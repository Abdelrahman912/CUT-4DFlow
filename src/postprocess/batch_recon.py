"""BatchRecon — reconstruct the 32 official ValidationSet cases with a checkpoint.

Submission writer for the CMRx4DFlow ValidationSet (no GT -> no metrics). For each
case it loads the SHIPPED undersampled k-space (kdata_ktGaussian{R}) at the case's
assigned R, reconstructs with the trained net over the full cardiac cycle, writes
the challenge directory layout (COO .npz in NATIVE orientation) and zips it.
Animations are NOT part of a submission: --anim delegates to
src.postprocess.animate (lazy import), and the default path never builds the
zero-filled volume they need.

Outputs go OUTSIDE test/ (default outputs/<run>/...). Configure via the .sh:
  --ckpt <best.ckpt>  --out-dir outputs/<run>  --val-root .../ValidationSet/Aorta
  --anim   (write animations)   --cpu   (force CPU)

Recon recipe matches the validated peek (Ω input, T=5 windows, phases = bins/T_size).
The model arch is read from the checkpoint's `meta` — point --ckpt at any smoke best.ckpt.
"""
from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path

import gc
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import scipy.fft as _sfft
import os as _os
_os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")  # gfx906: avoid frag OOM
import torch

# CMRX_TF32=1: TF32 tensor-core matmuls (CUDA Ampere+; ~2-4x on the GEMM share).
# DEFAULT OFF — quality-gate with scripts/bench_precision.py before enabling for
# any submission (10-bit matmul mantissa; measured metric deltas must be ~0).
if _os.environ.get("CMRX_TF32", "") not in ("", "0"):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("[batch_recon] TF32 ENABLED (CMRX_TF32=1)", flush=True)

# cuDNN conv autotuning: shapes are fixed within a case (few distinct shapes per run),
# so benchmarking the conv algorithms once pays for itself immediately. The transformer
# path never set this (only the dual/bcrnn constructors do).
torch.backends.cudnn.benchmark = True

# Fused ComplexLayerNorm (torch.compile, measured 1.43x on H100 in training) — enable
# by default at inference on CUDA builds only. probe-then-adopt already falls back to
# eager safely, but on ROCm/gfx906 inductor cannot lower it at all and the failed probe
# spams several alarming-looking "error: unsupported target: 'gfx906'" lines per run, so
# don't even attempt it there. Set CMRX_COMPILE_NORM=1 explicitly to override.
if torch.version.hip is None:
    _os.environ.setdefault("CMRX_COMPILE_NORM", "1")

from src.data.h5_reader import h5_complex, read_params_csv
from src.data.orientation import canonicalise_spatial, decanonicalise_spatial
from src.utils.cmrx_metrics import save_coo_npz
from src.models.conditioning import build_m
from src.recon.net_recon import build_cascade
from src.utils.mri_ops import ifftc2d


_FFT_WORKERS = int(_os.environ.get("CMRX_FFT_WORKERS", "-1"))  # -1 = all cores (standalone recon)


def _ifftc(x, axes):
    # scipy.fft (not numpy.fft): scipy PRESERVES the input dtype, so a complex64
    # volume stays complex64. numpy.fft always upcasts to complex128 -> 2x RAM and
    # ~20 GB temporaries on the big S2 volumes (Cerebro/Carotid), which OOM-killed
    # the recon on the 62 GB host. Numerically identical (both pocketfft, ortho norm).
    # workers=-1: pocketfft threads across ALL cores — the full-volume FE-IFFT and
    # the per-slice ZF IFFTs are the recon's main CPU cost (batch over the non-FFT
    # axes parallelizes well). CMRX_FFT_WORKERS caps it (training sets 1/worker).
    return _sfft.fftshift(
        _sfft.ifftn(_sfft.ifftshift(x, axes=axes), axes=axes, norm="ortho",
                    workers=_FFT_WORKERS), axes=axes)


def load_val_case(pdir: Path, no_canon: bool = False) -> dict:
    """Load one official val case: shipped undersampled k-space (Ω) at its assigned R.

    no_canon: skip orientation canonicalisation and recon in the scan's NATIVE
    frame. Needed for the S2 special-task organs (Carotid/Cerebro/PortalVein/
    RenalArtery), whose spatial_order is a PERMUTATION of the Aorta canonical
    order (HF,AP,LR) — canonicalise_spatial only handles axis flips, not the
    axis reordering these organs would need (and reordering would move the
    fully-sampled FE readout into an undersampled position, breaking recon).
    The network reconstructs the undersampled k-space regardless of anatomical
    orientation, so we feed native and keep the output native (decanon skipped).
    """
    kfile = next(pdir.glob("kdata_ktGaussian*.mat"))
    R = int(kfile.stem.replace("kdata_ktGaussian", ""))
    kdata = h5_complex(kfile, "kdata_ktGaussian")                       # (Nv,Nt,Nc,SPE,PE,FE) raw
    coil = h5_complex(pdir / "coilmap.mat", "coilmap")                  # (Nc,SPE,PE,FE)
    with h5py.File(pdir / "segmask.mat", "r") as f:
        seg_native = np.array(f["segmask"]).astype(bool)
    params = read_params_csv(pdir / "params.csv")
    so = params["spatial_order"]
    if not no_canon:
        kdata = canonicalise_spatial(kdata, so)                        # -> canonical (training frame)
        coil = canonicalise_spatial(coil, so)
    kdh = _ifftc(kdata, axes=[-1]).astype(np.complex64)                # FE iFFT -> hybrid (net input)
    return dict(R=R, spatial_order=so, no_canon=no_canon, kdata_hybrid=kdh,
                coilmap=coil.astype(np.complex64), segmask_native=seg_native,
                VENC=np.asarray(params["VENC"], np.float32),
                B0=float(params.get("field_strength") or 3.0))


@torch.no_grad()
def _prep_chunk(kfeb, cfeb, device):
    """Upload ONE raw FE-chunk and build every network input ON THE GPU.

    kfeb (B,Nv,Nt,Nc,SPE,PE) / cfeb (B,Nc,SPE,PE) numpy -> device tensors
    (kn, c, izf, norms). Previously the zero-filled image was built with CPU scipy
    FFTs (measured ~15-18 s/case, the single largest CPU block in the recon), then
    re-sliced and re-uploaded for EVERY T-window. Here the raw chunk crosses PCIe
    once and every derived tensor stays resident, so the window loop is pure GPU work.
    """
    k = torch.from_numpy(np.ascontiguousarray(kfeb)).to(device)     # (B,Nv,Nt,Nc,SPE,PE)
    c = torch.from_numpy(np.ascontiguousarray(cfeb)).to(device)     # (B,Nc,SPE,PE)
    B = k.shape[0]
    # per-slice normalization — same statistic as the CPU path (RMS over sampled points)
    absk = k.abs().reshape(B, -1)
    denom = (absk > 0).sum(dim=1).to(torch.float64).sqrt().clamp(min=1.0)
    norms = (absk.to(torch.float64).norm(dim=1) / denom)
    norms = torch.where(norms == 0, torch.ones_like(norms), norms).to(torch.float32)
    kn = (k / norms[:, None, None, None, None, None]).to(torch.complex64)
    del k, absk
    cconj = c.conj()
    izf = (ifftc2d(kn) * cconj[:, None, None]).sum(dim=-3)          # (B,Nv,Nt,SPE,PE)
    return kn, c, izf, norms


@torch.no_grad()          # inference only: no autograd graph anywhere in the recon
def recon_case(model, sample, device, T_size: int, overlap: int = 2, fe_chunk: int = 4,
               want_zf: bool = True, gpu_batch: int = 4):
    """Net recon of the full cycle from the shipped Ω, T-windows (phases = bins/T_size).

    overlap=0: non-overlapping windows (tail slides back), matches the original p2 submission.
    overlap>0 (default 2): windows stride by (T_size-overlap); frames seen by >1 window are
    complex-AVERAGED (the winning "ov2" setting).
    fe_chunk: FE slices batched per GPU forward (they are independent — batching them
    turns ~117 small forwards into ~30 big ones; same math, per-slice normalization
    kept). On CUDA OOM the chunk transparently falls back to slice-by-slice.
    gpu_batch: how many (FE slice, T window) PAIRS go through the network in ONE
    forward. FE slices and windows are both independent, so they share the batch
    axis: at fe_chunk=4 with ~7 windows a case drops from ~150-240 cascade
    invocations to ~30 (gpu_batch=16) or ~8 (gpu_batch=32), amortising the
    per-forward dispatch floor. Halves itself on CUDA OOM. Default 4 reproduces
    the one-window-at-a-time forward count and is safe on every card; it pays off
    only where dispatch (not compute) is the bottleneck — measured slightly
    NEGATIVE on a work-bound 16GB gfx906, so A/B it per GPU before submitting.
    want_zf: also return the zero-filled volume. It costs a FULL-volume CPU iFFT +
    coil-combine (seconds per Aorta case, tens on S2) and is consumed ONLY by the
    animations, so submission runs (--anim off) pass False and skip it.
    """
    kdh, coil = sample["kdata_hybrid"], sample["coilmap"]
    Nv, Nt, Nc, SPE, PE, FE = kdh.shape
    # Conditioning vector (one per case: R from the shipped mask/filename, B0 from params).
    m = None
    if getattr(model, "conditioning_enabled", False):
        m = build_m(model.conditioning_inputs, sample["R"], sample.get("B0"), device)
    ov = int(overlap) if Nt > T_size else 0
    if ov > 0:                                                          # precompute windows + coverage
        stride = max(1, T_size - ov)
        windows = [list(range(s0, s0 + T_size))
                   for s0 in sorted({min(s0, Nt - T_size) for s0 in range(0, Nt, stride)})]
        cnt = np.zeros(Nt, np.float32)
        for w in windows:
            for gf in w:
                cnt[gf] += 1.0
        cnt[cnt == 0] = 1.0
    else:
        windows = []
        for s0 in range(0, Nt, T_size):
            win = list(range(s0, min(s0 + T_size, Nt)))
            if len(win) < T_size and Nt >= T_size:
                win = list(range(Nt - T_size, Nt))                     # tail slides back to stay full-T
            windows.append(win)

    win_t = torch.as_tensor(np.asarray(windows), dtype=torch.long, device=device)  # (W,T)
    W = win_t.shape[0]
    cnt_t = (torch.from_numpy(cnt).to(device)[None, None, :, None, None] if ov > 0 else None)
    # Which (window, position) pairs write to the output. ov>0: all (index_add_ then
    # divide by coverage). ov=0: only the LAST window covering a frame owns it, which
    # reproduces the sequential "tail overwrites" semantics with unique scatter indices.
    own = np.ones((W, T_size), bool)
    if ov == 0:
        last = {}
        for wi, w in enumerate(windows):
            for li, gf in enumerate(w):
                last[gf] = (wi, li)
        own[:] = False
        for wi, li in last.values():
            own[wi, li] = True
    own_t = torch.from_numpy(own).to(device)

    out = np.zeros((Nv, Nt, SPE, PE, FE), np.complex64)
    Tw = win_t.shape[1]

    def _run(kfeb, cfeb, gb):
        """Prep + every (slice, window) pair for one (sub)chunk, entirely on device."""
        kn, c, izf, norms = _prep_chunk(kfeb, cfeb, device)
        B = izf.shape[0]
        # fold the frame axis next to the batch axis so a (slice, frame) pair is ONE
        # flat index -> gather/scatter for any (slice, window) group is a single op.
        izf_f = izf.permute(0, 2, 1, 3, 4).reshape(B * Nt, Nv, SPE, PE)
        kn_f = kn.permute(0, 2, 1, 3, 4, 5).reshape(B * Nt, Nv, Nc, SPE, PE)
        acc_f = torch.zeros(B * Nt, Nv, SPE, PE, dtype=torch.complex64, device=device)
        pairs = [(b, wi) for b in range(B) for wi in range(W)]
        for g0 in range(0, len(pairs), gb):
            grp = pairs[g0:g0 + gb]
            g = len(grp)
            b_idx = torch.as_tensor([p[0] for p in grp], dtype=torch.long, device=device)
            w_idx = torch.as_tensor([p[1] for p in grp], dtype=torch.long, device=device)
            bins = win_t.index_select(0, w_idx)                        # (g,Tw)
            flat = (b_idx[:, None] * Nt + bins).reshape(-1)            # (g*Tw,)
            xi = (izf_f.index_select(0, flat).reshape(g, Tw, Nv, SPE, PE)
                  .permute(0, 2, 1, 3, 4).contiguous())                # (g,Nv,Tw,SPE,PE)
            yt = (kn_f.index_select(0, flat).reshape(g, Tw, Nv, Nc, SPE, PE)
                  .permute(0, 2, 3, 1, 4, 5).contiguous())            # (g,Nv,Nc,Tw,SPE,PE)
            cg = c.index_select(0, b_idx)
            ph = (bins.to(torch.float32) / T_size).contiguous()
            r = model(xi, yt, cg, None, ph, m=m, R=sample["R"])
            rf = r.permute(0, 2, 1, 3, 4).reshape(g * Tw, Nv, SPE, PE)
            sel = own_t.index_select(0, w_idx).reshape(-1)             # (g*Tw,) bool
            acc_f.index_add_(0, flat[sel], rf[sel])
        acc = acc_f.reshape(B, Nt, Nv, SPE, PE).permute(0, 2, 1, 3, 4)
        if ov > 0:
            acc = acc / cnt_t
        acc = acc * norms[:, None, None, None, None]                   # de-normalize per slice
        return acc.cpu().numpy()                                       # ONE D2H per chunk

    step = max(1, int(fe_chunk))
    gb = max(1, int(gpu_batch))
    for f0 in range(0, FE, step):
        f1 = min(f0 + step, FE)
        kfeb = np.moveaxis(kdh[..., f0:f1], -1, 0)                     # (B,Nv,Nt,Nc,SPE,PE)
        cfeb = np.moveaxis(coil[..., f0:f1], -1, 0)                    # (B,Nc,SPE,PE)
        while True:
            try:
                res = _run(kfeb, cfeb, gb)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if gb > 1:
                    gb = max(1, gb // 2)
                    print(f"[recon] OOM -> retrying with gpu_batch={gb}", flush=True)
                    continue
                print(f"[recon] OOM at gpu_batch=1 -> slice-by-slice for this chunk", flush=True)
                res = np.concatenate([_run(kfeb[b:b + 1], cfeb[b:b + 1], 1)
                                      for b in range(f1 - f0)], axis=0)
                break
        out[..., f0:f1] = np.moveaxis(res, 0, -1)
    return out, (izf_full(sample) if want_zf else None)                # recon (canonical), zf (canonical)


def izf_full(sample) -> np.ndarray:
    """Zero-filled image (canonical) from the shipped Ω.

    Consumed by the diagnostic animations and by scripts/generic_recon.py (which
    hands users a zero-filled reference); submission runs skip it (want_zf=False).

    kdh (Nv,Nt,Nc,SPE,PE,FE) -> iFFT over SPE,PE -> coil-combine (sum over Nc)."""
    kdh, coil = sample["kdata_hybrid"], sample["coilmap"]
    return np.sum(_ifftc(kdh, axes=[-2, -3]) * np.conj(coil), axis=-4).astype(np.complex64)


def main() -> int:
    ap = argparse.ArgumentParser(description="BatchRecon: 32 official val cases -> submission (+anim)")
    ap.add_argument("--ckpt", required=True, help="net checkpoint (arch read from its meta)")
    ap.add_argument("--use-ema", action="store_true",
                    help="load ck['ema_model'] (EMA-averaged weights) instead of the raw weights")
    ap.add_argument("--val-root", default="/home/afathy/Thesis/code/flowmri_net-main/data/CMRx4DFlow/"
                                          "_extracted/TaskR1R2/ValidationSet/Aorta")
    ap.add_argument("--out-dir", required=True, help="output dir (NOT under test/), e.g. outputs/lv1")
    ap.add_argument("--anim", action="store_true",
                    help="DIAGNOSTIC: also write per-case cine gifs via src.postprocess.animate. Costs a full-volume zero-filled iFFT per case; off for submissions.")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--n", type=int, default=-1, help="first N cases (-1 = all 32)")
    ap.add_argument("--overlap", type=int, default=2,
                    help="temporal-window overlap (default 2 = the winning 'ov2'; 0 = non-overlap p2-style).")
    ap.add_argument("--prefetch", type=int, default=1,
                    help="1 = load the next case on a background thread (hides CPU load "
                         "behind GPU recon; one extra case in RAM). 0 = off.")
    ap.add_argument("--gpu-batch", type=int, default=4,
                    help="(FE slice, T window) pairs per network forward; halves on OOM. "
                         "4 = one window x fe_chunk slices (safe everywhere). RAISE on a "
                         "big CUDA card (16-32 on A6000/H100) where the per-forward "
                         "dispatch floor dominates; measured NEUTRAL-to-4%%-SLOWER on a "
                         "16GB gfx906, which is already work-bound.")
    ap.add_argument("--fe-chunk", type=int, default=4,
                    help="FE slices batched per GPU forward (independent slices; auto slice-by-slice on OOM).")
    ap.add_argument("--out-subpath", default="TaskR1R2/ValidationSet/Aorta",
                    help="submission layout under --out-dir (e.g. TaskS2/ValidationSet/Carotid for the special tasks).")
    ap.add_argument("--no-canon", action="store_true",
                    help="skip orientation canonicalise/decanonicalise; recon in the scan's NATIVE "
                         "frame. Required for S2 organs whose spatial_order permutes the Aorta canonical "
                         "(HF,AP,LR) order (Carotid/Cerebrovascular/PortalVein/RenalArtery).")
    ap.add_argument("--n-stages", type=int, default=-1,
                    help="TEST-TIME unroll count. -1 = use the trained depth. If > trained n_stages, the "
                         "shared denoiser is iterated extra times, reusing per-stage gates by --gate-policy. "
                         "Study only (stages beyond training were never optimised); unconditioned nets only.")
    ap.add_argument("--gate-policy", choices=["hold", "cycle"], default="hold",
                    help="gate reuse for the extra stages: 'hold' repeats the last trained gate, "
                         "'cycle' wraps the trained schedule.")
    a = ap.parse_args()

    device = torch.device("cpu" if (a.cpu or not torch.cuda.is_available()) else "cuda")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    meta = ck.get("meta") or ck.get("cfg")            # smoke: 'meta' | cluster train_cmrx: 'cfg'
    if a.use_ema:
        sd = ck.get("ema_model")
        if sd is None:
            raise SystemExit(f"--use-ema set but no 'ema_model' in {a.ckpt} (was it trained with EMA?)")
        print(f"[batch_recon] EMA weights (ema_model) <- {a.ckpt}")
    else:
        sd = ck.get("state_dict") or ck.get("model")  # smoke: 'state_dict' | cluster: 'model'
    T_size = int(meta.get("T_size", 5))
    model = build_cascade(meta, device); model.load_state_dict(sd); model.eval()

    # ── Optional test-time unroll override (base@N study) ──
    # The denoiser is weight-tied, so extra stages reuse it for free. The per-stage DC/WA gates
    # (and, for conditioned nets, the per-stage conditioning table) are reused by --gate-policy.
    trained_stages = model.n_stages
    if a.n_stages > 0 and a.n_stages != trained_stages:
        ns = trained_stages
        gidx = lambda i: (ns - 1) if a.gate_policy == "hold" else (i % ns)   # noqa: E731
        if a.n_stages < ns:                            # truncate
            model.dc = model.dc[:a.n_stages]
            model.wa = model.wa[:a.n_stages]
        else:                                          # extend by reusing gate modules
            for i in range(ns, a.n_stages):
                model.dc.append(model.dc[gidx(i)])
                model.wa.append(model.wa[gidx(i)])
        # conditioned nets: resize the per-stage cond_table (stage axis) by the same policy
        if getattr(model, "cond_table", None) is not None:
            with torch.no_grad():
                dp = model.cond_table.dp_table                     # (n_R, ns)
                if a.n_stages < ns:
                    new = dp[:, :a.n_stages]
                else:
                    extra = dp[:, [gidx(i) for i in range(ns, a.n_stages)]]
                    new = torch.cat([dp, extra], dim=1)            # (n_R, a.n_stages)
                model.cond_table.dp_table = torch.nn.Parameter(new.clone())
            model.cond_table.n_stages = a.n_stages
        model.n_stages = a.n_stages
        print(f"[stages] TEST-TIME override: {trained_stages} -> {model.n_stages} "
              f"(gate-policy={a.gate_policy}); stages beyond {trained_stages} were NOT trained.")

    # ── Provenance: which epoch produced this submission, and its losses ──
    # NOTE: 'epoch'/'best_val' are TOP-LEVEL checkpoint keys, not config keys —
    # reading them off `meta` (the cfg) is why this used to print epoch=None.
    prov = {"ckpt": str(a.ckpt), "use_ema": bool(a.use_ema),
            "epoch": ck.get("epoch"), "best_val": ck.get("best_val"),
            "train_loss": None, "val_loss": None,
            "d_model": meta.get("d_model"), "n_heads": meta.get("n_heads"),
            "n_stages": int(model.n_stages), "trained_n_stages": int(trained_stages),
            "gate_policy": (a.gate_policy if model.n_stages != trained_stages else None),
            "lambda_v": meta.get("lambda_v"), "supervised": meta.get("supervised")}
    _log = Path(a.ckpt).parent / "train_log.csv"      # sibling of the checkpoint
    if _log.is_file() and prov["epoch"] is not None:
        import csv as _csv
        with open(_log) as _f:
            for _row in _csv.DictReader(_f):
                try:
                    if int(float(_row["epoch"])) == int(prov["epoch"]):
                        prov["train_loss"] = float(_row["train_loss"]) if _row.get("train_loss") else None
                        prov["val_loss"] = float(_row["val_loss"]) if _row.get("val_loss") else None
                        break
                except (ValueError, KeyError):
                    continue
    _f2 = lambda v: "n/a" if v is None else f"{v:.6f}"
    print(f"[ckpt] {a.ckpt}")
    print(f"[ckpt] epoch={prov['epoch']}  train_loss={_f2(prov['train_loss'])}  "
          f"val_loss={_f2(prov['val_loss'])}  best_val={_f2(prov['best_val'])}")
    print(f"[ckpt] d_model={prov['d_model']} n_heads={prov['n_heads']} n_stages={prov['n_stages']} "
          f"lambda_v={prov['lambda_v']} supervised={prov['supervised']}")
    print(f"[ckpt] device={device} T_size={T_size} overlap={a.overlap} no_canon={a.no_canon}")
    if prov["epoch"] is None:
        print("[ckpt] WARNING: checkpoint has no 'epoch' key — provenance incomplete")

    out_dir = Path(a.out_dir); anim_dir = out_dir / "anim"
    cases = sorted(Path(a.val_root).glob("Center*/*/P*"))
    if a.n > 0:
        cases = cases[:a.n]
    print(f"[batch] {len(cases)} validation cases -> {out_dir}")
    _pool = ThreadPoolExecutor(max_workers=1) if int(a.prefetch) > 0 else None
    _fut = None
    _pf: list = []
    for i, pdir in enumerate(cases):
        centre, vendor, pid = pdir.parts[-3], pdir.parts[-2], pdir.name
        key = f"{centre}/{vendor}/{pid}"
        # Case loading (gzip inflate + canonicalise + full-volume FE-iFFT) is pure CPU
        # and was fully serialized with GPU recon — ~15 s/case on Aorta, minutes on the
        # S2 organs. A single-slot prefetch thread runs it for case N+1 while the GPU
        # reconstructs case N, so that cost disappears behind compute whenever the recon
        # is the longer of the two. Costs one extra case resident in host RAM.
        s = _pf.pop() if _pf else load_val_case(pdir, no_canon=a.no_canon)
        if _pool is not None and i + 1 < len(cases):
            _fut = _pool.submit(load_val_case, cases[i + 1], no_canon=a.no_canon)
        t0 = time.time()
        rec, zf = recon_case(model, s, device, T_size, overlap=a.overlap,
                             fe_chunk=a.fe_chunk, want_zf=bool(a.anim),
                             gpu_batch=a.gpu_batch)
        # no_canon: recon is already native (canonicalise was skipped) -> no decanon.
        rec_nat = rec if a.no_canon else decanonicalise_spatial(rec, s["spatial_order"])
        zf_nat = None if zf is None else (
            zf if a.no_canon else decanonicalise_spatial(zf, s["spatial_order"]))
        masked = (rec_nat * s["segmask_native"][None, None]).astype(np.complex64)
        outp = (out_dir.joinpath(*a.out_subpath.split("/"))
                / centre / vendor / pid / f"img_ktGaussian{s['R']}.npz")
        outp.parent.mkdir(parents=True, exist_ok=True)
        save_coo_npz(str(outp), masked)
        msg = f"  {key} R={s['R']} {time.time()-t0:5.0f}s -> {outp.name}"
        if a.anim:
            from src.postprocess.animate import animate_case   # diagnostic-only, lazy
            animate_case(key, rec_nat, zf_nat, s["segmask_native"], s["VENC"],
                         anim_dir / f"anim_{key.replace('/','__')}.gif", R=s["R"])
            msg += "  +anim"
        print(msg)
        # Free the big per-case arrays before loading the next case (the S2 volumes
        # are ~10 GB each; on a no-swap host, fragmentation across 10 cases can OOM).
        del s, rec, zf, rec_nat, zf_nat, masked
        if _fut is not None:                       # collect the prefetched next case
            _pf.append(_fut.result()); _fut = None
        gc.collect()
    if _pool is not None:
        _pool.shutdown(wait=True)

    # Record which model produced this submission, next to the zip, so a zip can
    # always be traced back to an epoch + its losses.
    prov.update(out_dir=str(out_dir), out_subpath=a.out_subpath, n_cases=len(cases),
                overlap=a.overlap, no_canon=bool(a.no_canon))
    with open(out_dir / "provenance.json", "w") as f:
        json.dump(prov, f, indent=2, default=str)
    print(f"[prov] epoch={prov['epoch']} train_loss={_f2(prov['train_loss'])} "
          f"val_loss={_f2(prov['val_loss'])} -> {out_dir / 'provenance.json'}")

    zp = out_dir / "RosettaFlow-Submission.zip"
    # ZIP_STORED, not DEFLATE: every member is a .npz whose arrays are ALREADY
    # deflate-compressed, so re-deflating spends seconds of CPU for ~0% gain.
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as zf:
        for p in (out_dir / a.out_subpath.split("/")[0]).rglob("*.npz"):
            zf.write(p, p.relative_to(out_dir))
    print(f"[zip] {zp}  ({zp.stat().st_size/1e6:.1f} MB, {len(zipfile.ZipFile(zp).namelist())} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

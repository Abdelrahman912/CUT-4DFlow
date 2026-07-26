"""Submission writer: reconstruct a ValidationSet with a checkpoint and write the challenge layout.

For each case it loads the shipped undersampled k-space at its assigned R, reconstructs the full
cardiac cycle, masks to the segmentation, writes a sparse COO ``.npz`` in native orientation, and
zips the result. Optionally writes per-case animations. The model architecture is read from the
checkpoint config.
"""
from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path

import gc

import h5py
import numpy as np
import scipy.fft as _sfft
import torch

from src.data.h5_reader import h5_complex, read_params_csv
from src.data.orientation import canonicalise_spatial, decanonicalise_spatial
from src.utils.cmrx_metrics import complex2magflow, save_coo_npz
from src.models.conditioning import build_m
from src.recon.net_recon import build_cascade


def _ifftc(x, axes):
    return _sfft.fftshift(_sfft.ifftn(_sfft.ifftshift(x, axes=axes), axes=axes, norm="ortho"), axes=axes)


def load_val_case(pdir: Path, no_canon: bool = False) -> dict:
    """Load one case: shipped undersampled k-space (Omega) at its assigned R, plus coil maps and seg.

    ``no_canon``: recon in the scan's native frame (needed when spatial_order is a permutation of the
    canonical order, e.g. the cross-anatomy organs).
    """
    kfile = next(pdir.glob("kdata_ktGaussian*.mat"))
    R = int(kfile.stem.replace("kdata_ktGaussian", ""))
    kdata = h5_complex(kfile, "kdata_ktGaussian")
    coil = h5_complex(pdir / "coilmap.mat", "coilmap")
    with h5py.File(pdir / "segmask.mat", "r") as f:
        seg_native = np.array(f["segmask"]).astype(bool)
    params = read_params_csv(pdir / "params.csv")
    so = params["spatial_order"]
    if not no_canon:
        kdata = canonicalise_spatial(kdata, so)
        coil = canonicalise_spatial(coil, so)
    kdh = _ifftc(kdata, axes=[-1]).astype(np.complex64)          # FE iFFT -> hybrid (net input)
    return dict(R=R, spatial_order=so, no_canon=no_canon, kdata_hybrid=kdh,
                coilmap=coil.astype(np.complex64), segmask_native=seg_native,
                VENC=np.asarray(params["VENC"], np.float32),
                B0=float(params.get("field_strength") or 3.0))


@torch.no_grad()
def _recon_window(model, izf_n, kfe_n, cfe, win, T_size, norm, device, m=None, R=None):
    """Reconstruct one T-window for one FE slice. Returns (Nv, len(win), SPE, PE)."""
    bins = np.asarray(win)
    xi = torch.from_numpy(izf_n[:, bins][None]).to(device)
    yt = torch.from_numpy(np.take(kfe_n, bins, axis=1).transpose(0, 2, 1, 3, 4)[None]).to(device)
    sn = torch.from_numpy(cfe[None]).to(device)
    ph = torch.tensor((bins.astype(np.float32) / T_size)[None, :], device=device)
    return model(xi, yt, sn, None, ph, m=m, R=R)[0].cpu().numpy() * norm


def recon_case(model, sample, device, T_size: int, overlap: int = 2):
    """Reconstruct the full cycle from the shipped Omega, tiled in T-windows (phases = bins/T_size).

    overlap>0: overlapping windows; frames seen by more than one are complex-averaged.
    """
    kdh, coil = sample["kdata_hybrid"], sample["coilmap"]
    Nv, Nt, Nc, SPE, PE, FE = kdh.shape
    m = None
    if getattr(model, "conditioning_enabled", False):
        m = build_m(model.conditioning_inputs, sample["R"], sample.get("B0"), device)
    ov = int(overlap) if Nt > T_size else 0
    if ov > 0:
        stride = max(1, T_size - ov)
        windows = [list(range(s0, s0 + T_size))
                   for s0 in sorted({min(s0, Nt - T_size) for s0 in range(0, Nt, stride)})]
        cnt = np.zeros(Nt, np.float32)
        for w in windows:
            for gf in w:
                cnt[gf] += 1.0
        cnt[cnt == 0] = 1.0
    out = np.zeros((Nv, Nt, SPE, PE, FE), np.complex64)
    for fe in range(FE):
        kfe, cfe = kdh[..., fe], coil[..., fe]
        img_zf = np.sum(_ifftc(kfe, axes=[-1, -2]) * np.conj(cfe), axis=-3)
        denom = float(np.linalg.norm(np.abs(kfe) != 0))
        norm = (float(np.linalg.norm(kfe)) / max(denom, 1.0)) or 1.0
        kfe_n = (kfe / norm).astype(np.complex64)
        izf_n = (img_zf / norm).astype(np.complex64)
        if ov > 0:
            acc = np.zeros((Nv, Nt, SPE, PE), np.complex64)
            for win in windows:
                r = _recon_window(model, izf_n, kfe_n, cfe, win, T_size, norm, device, m, sample["R"])
                for li, gf in enumerate(win):
                    acc[:, gf] += r[:, li]
            out[..., fe] = acc / cnt[None, :, None, None]
        else:
            for s0 in range(0, Nt, T_size):
                win = list(range(s0, min(s0 + T_size, Nt)))
                if len(win) < T_size and Nt >= T_size:
                    win = list(range(Nt - T_size, Nt))
                r = _recon_window(model, izf_n, kfe_n, cfe, win, T_size, norm, device, m, sample["R"])
                for li, gf in enumerate(win):
                    out[:, gf, :, :, fe] = r[:, li]
    return out, izf_full(sample)


def izf_full(sample) -> np.ndarray:
    """Coil-combined zero-filled image from the shipped Omega (animation baseline)."""
    kdh, coil = sample["kdata_hybrid"], sample["coilmap"]
    return np.sum(_ifftc(kdh, axes=[-2, -3]) * np.conj(coil), axis=-4).astype(np.complex64)


def animate_case(key, rec_nat, zf_nat, seg, venc, out_gif, R=None, fps=6):
    """3-panel cine over the cycle: anatomy |img| / recon |v| / zero-filled |v|."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as manim
    Nv, Nt, SPE, PE, FE = rec_nat.shape
    z = int(seg.sum(axis=(1, 2)).argmax()) if seg.any() else SPE // 2
    seg2d = seg[z]
    mag_r, _ = complex2magflow(rec_nat, None)
    _, flow_r = complex2magflow(rec_nat, venc)
    sp_r = np.linalg.norm(flow_r, axis=0)[:, z]
    _, flow_z = complex2magflow(zf_nat, venc)
    sp_z = np.linalg.norm(flow_z, axis=0)[:, z]
    mmax = float(np.percentile(mag_r[0, :, z], 99)) + 1e-6
    vmax = (float(np.percentile(sp_r[:, seg2d], 99)) if seg2d.any() else 1.0) + 1e-6
    panels = [("anatomy |img|", mag_r[0, :, z], "gray", 0, mmax),
              ("blood |v|", sp_r * seg2d[None], "jet", 0, vmax),
              ("zero-filled |v|", sp_z * seg2d[None], "jet", 0, vmax)]
    fig, ax = plt.subplots(1, 3, figsize=(9.5, 3.4), constrained_layout=True)
    ims = [ax[j].imshow(p[1][0], cmap=p[2], vmin=p[3], vmax=p[4]) for j, p in enumerate(panels)]
    for j, p in enumerate(panels):
        ax[j].set_title(p[0], fontsize=9)
        ax[j].set_xticks([])
        ax[j].set_yticks([])
    ttl = fig.suptitle("")

    def upd(t):
        for j, p in enumerate(panels):
            ims[j].set_data(p[1][t])
        ttl.set_text(f"{key}  R={R}  SPE={z}  frame {t+1}/{Nt}")
        return ims

    anim = manim.FuncAnimation(fig, upd, frames=Nt, interval=1000 / fps, blit=False)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_gif), writer=manim.PillowWriter(fps=fps))
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Reconstruct a ValidationSet into the submission layout")
    ap.add_argument("--ckpt", required=True, help="checkpoint (architecture read from its config)")
    ap.add_argument("--val-root", default="Data/TaskR1R2/ValidationSet/Aorta")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--anim", action="store_true", help="also write per-case animations")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--n", type=int, default=-1, help="first N cases (-1 = all)")
    ap.add_argument("--overlap", type=int, default=2, help="temporal-window overlap (0 = non-overlapping)")
    ap.add_argument("--out-subpath", default="TaskR1R2/ValidationSet/Aorta")
    ap.add_argument("--no-canon", action="store_true",
                    help="recon in native orientation (for anatomies whose spatial_order permutes the axes)")
    a = ap.parse_args()

    device = torch.device("cpu" if (a.cpu or not torch.cuda.is_available()) else "cuda")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    meta = ck.get("meta") or ck.get("cfg")
    sd = ck.get("state_dict") or ck.get("model")
    T_size = int(meta.get("T_size", 5))
    model = build_cascade(meta, device)
    model.load_state_dict(sd)
    model.eval()

    prov = {"ckpt": str(a.ckpt), "epoch": ck.get("epoch"), "best_val": ck.get("best_val"),
            "train_loss": None, "val_loss": None,
            "d_model": meta.get("d_model"), "n_heads": meta.get("n_heads"),
            "n_stages": meta.get("n_stages"), "lambda_v": meta.get("lambda_v"),
            "supervised": meta.get("supervised")}
    _log = Path(a.ckpt).parent / "train_log.csv"
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
    _f2 = lambda v: "n/a" if v is None else f"{v:.6f}"  # noqa: E731
    print(f"[ckpt] {a.ckpt}  epoch={prov['epoch']}  val_loss={_f2(prov['val_loss'])}")
    print(f"[ckpt] d_model={prov['d_model']} n_heads={prov['n_heads']} n_stages={prov['n_stages']} "
          f"lambda_v={prov['lambda_v']} supervised={prov['supervised']}")
    print(f"[ckpt] device={device} T_size={T_size} overlap={a.overlap} no_canon={a.no_canon}")

    out_dir = Path(a.out_dir)
    anim_dir = out_dir / "anim"
    cases = sorted(Path(a.val_root).glob("Center*/*/P*"))
    if a.n > 0:
        cases = cases[:a.n]
    print(f"[batch] {len(cases)} validation cases -> {out_dir}")
    for pdir in cases:
        centre, vendor, pid = pdir.parts[-3], pdir.parts[-2], pdir.name
        key = f"{centre}/{vendor}/{pid}"
        s = load_val_case(pdir, no_canon=a.no_canon)
        t0 = time.time()
        rec, zf = recon_case(model, s, device, T_size, overlap=a.overlap)
        rec_nat = rec if a.no_canon else decanonicalise_spatial(rec, s["spatial_order"])
        zf_nat = zf if a.no_canon else decanonicalise_spatial(zf, s["spatial_order"])
        masked = (rec_nat * s["segmask_native"][None, None]).astype(np.complex64)
        outp = (out_dir.joinpath(*a.out_subpath.split("/")) / centre / vendor / pid
                / f"img_ktGaussian{s['R']}.npz")
        outp.parent.mkdir(parents=True, exist_ok=True)
        save_coo_npz(str(outp), masked)
        msg = f"  {key} R={s['R']} {time.time()-t0:5.0f}s -> {outp.name}"
        if a.anim:
            animate_case(key, rec_nat, zf_nat, s["segmask_native"], s["VENC"],
                         anim_dir / f"anim_{key.replace('/','__')}.gif", R=s["R"])
            msg += "  +anim"
        print(msg)
        del s, rec, zf, rec_nat, zf_nat, masked
        gc.collect()

    prov.update(out_dir=str(out_dir), out_subpath=a.out_subpath, n_cases=len(cases),
                overlap=a.overlap, no_canon=bool(a.no_canon))
    with open(out_dir / "provenance.json", "w") as f:
        json.dump(prov, f, indent=2, default=str)

    zp = out_dir / "RosettaFlow-Submission.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in (out_dir / a.out_subpath.split("/")[0]).rglob("*.npz"):
            zf.write(p, p.relative_to(out_dir))
    print(f"[zip] {zp}  ({zp.stat().st_size/1e6:.1f} MB, {len(zipfile.ZipFile(zp).namelist())} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Training loop for the CMRx4DFlow2026 pure-transformer cascade (non-FiLM).

Derived from :mod:`src.training.train` (Arch-A trainer) and adapted for:

  - the CMRx unrolled non-FiLM cascade
    :class:`src.models.cascade.CMRxTransformerCascade`
  - the CMRx dataset ``src.data.dataset.CMRx4DFlowDataset`` (per-velocity sample;
    SSDU partition built per ``__getitem__``)
  - FlowMRI-Net-style train/val split loaded from
    ``configs/train_val_split.json`` (two patient lists)
  - SSDU loss on the held-out Lambda partition
  - scanner-balanced loss weighting (multiplied into the per-sample loss)
  - deterministic-by-path validation (same R / mask / venc / cardiac window
    every time a val patient is drawn)
  - mixed-precision toggle (default float32 — gfx906 reliability)
  - checkpointing (best + periodic)

Smoke-test entry point: ``--max-steps N`` overrides ``n_epochs`` and stops
training after ``N`` optimiser steps.
"""

from __future__ import annotations

import argparse
import csv as _csv
import glob
import math
import random
import sys
import time as _time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import CMRx4DFlowDataset
from src.data.splits import make_split
from src.data.samplers import BlockShuffleSampler, PatientInterleaveSampler
from src.models.attention import fuse_qkv_optimizer_state
from src.models.cascade import CMRxTransformerCascade
from src.models.conditioning import build_m
from src.training.loss import ssdu_phase_loss   # = ssdu_loss + lambda_v * velocity term
from src.utils.mri_ops import fftc2d


# ────────────────────────────────────────────────────────────────────────────
# Forward operator (k-space residual on the Lambda partition).
# ────────────────────────────────────────────────────────────────────────────


def mri_forward_with_mask(
    img: torch.Tensor,        # (B, V, T, SPE, PE) complex
    coil: torch.Tensor,       # (B, C, SPE, PE)   complex
    mask: torch.Tensor,       # (B, V, C, T, SPE, PE) float / bool
) -> torch.Tensor:
    """Apply coil sens, centred FFT2 over (SPE,PE), then mask in k-space.

    Returns (B, V, C, T, SPE, PE) complex.
    """
    coil_imgs = img.unsqueeze(2) * coil.unsqueeze(1).unsqueeze(3)  # (B, V, C, T, H, W)
    k = fftc2d(coil_imgs)
    return mask * k


# ssdu_loss / ssdu_phase_loss / velocity_difference_loss now live in
# src.training.loss (imported at the top of this module).


# ────────────────────────────────────────────────────────────────────────────
# Sample preparation: ndarray dict -> tensor dict suited to the cascade.
# ────────────────────────────────────────────────────────────────────────────


def _slice_to_tensors(
    sample: dict, device: torch.device, fe_idx: int | None = None,
) -> dict:
    """Wrap the dataset's per-sample ndarray dict into batched tensors.

    ``fe_idx`` selects which FE slice to extract:

      - ``None`` (default): use index 0. In train mode the dataset already
        slices one FE position (D_size=1), so the FE axis is length-1 and
        ``[..., 0]`` is the only valid slice.
      - integer: pick that FE index. Used by ``validate`` to take a
        deterministic middle slice per patient when D_size=-1 (whole volume).
    """
    fe = 0 if fe_idx is None else int(fe_idx)
    img_zf = torch.from_numpy(sample['img_zf'])[..., fe].unsqueeze(0).to(device)
    img_gt = torch.from_numpy(sample['img_gt'])[..., fe].unsqueeze(0).to(device)

    kl = torch.from_numpy(sample['kdata_lambda'])[..., fe]  # (V, T, C, SPE, PE)
    kl = kl.permute(0, 2, 1, 3, 4).unsqueeze(0).to(device)  # (B=1, V, C, T, SPE, PE)
    kt = torch.from_numpy(sample['kdata_theta'])[..., fe]
    kt = kt.permute(0, 2, 1, 3, 4).unsqueeze(0).to(device)

    # Masks broadcast over FE (last axis is length-1), so they don't index by fe.
    ml = torch.from_numpy(sample['mask_lambda'])[..., 0]  # (1, T, 1, SPE, PE)
    ml = ml.permute(0, 2, 1, 3, 4).unsqueeze(0).to(device)
    mth = torch.from_numpy(sample['mask_theta'])[..., 0]
    mth = mth.permute(0, 2, 1, 3, 4).unsqueeze(0).to(device)

    coil = torch.from_numpy(sample['coilmap'])[..., fe].unsqueeze(0).to(device)

    cardiac_bins = torch.from_numpy(sample['cardiac_bins']).to(device)

    return {
        'img_zf': img_zf, 'img_gt': img_gt,
        'kdata_theta': kt, 'kdata_lambda': kl,
        'mask_theta': mth, 'mask_lambda': ml,
        'coilmap': coil, 'cardiac_bins': cardiac_bins,
        'norm': float(sample['norm']),
        'R': int(sample['R']),
        'B0': float(sample['B0']),
        'seg_idx': int(sample['seg_idx']),
        'patient_dir': str(sample['patient_dir']),
        'scanner_weight': float(sample['scanner_weight']),
    }


# ────────────────────────────────────────────────────────────────────────────
# Per-sample training step (one batched item).
# ────────────────────────────────────────────────────────────────────────────


def training_step(
    model: CMRxTransformerCascade,
    batch: dict,
    device: torch.device,
    apply_scanner_weight: bool = True,
    lambda_v: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """One forward + loss. ``batch`` is a tensor dict from ``_slice_to_tensors``."""
    # The kdata tensor is authoritative for Nt: params.csv occasionally drifts
    # by ±1 (e.g. Center010/P004 lists Nt=26 but kdata has Nt=25). Use the
    # actual tensor T dimension to normalise cardiac_bins.
    # batch['kdata_theta'] shape: (B=1, V, C, T, SPE, PE)  -> T is axis 3.
    Nt = int(batch['kdata_theta'].shape[3])
    cardiac_bins = batch['cardiac_bins'].to(torch.float32) / max(int(Nt), 1)
    phases = cardiac_bins.unsqueeze(0)  # (B=1, T)

    # Conditioning vector m (None unless the model is conditioned).
    m = None
    if getattr(model, 'conditioning_enabled', False):
        m = build_m(model.conditioning_inputs, batch['R'], batch.get('B0'), device)

    x_recon = model(
        batch['img_zf'],
        batch['kdata_theta'],
        batch['coilmap'],
        batch['mask_theta'],
        phases,
        m=m,
        R=batch['R'],          # v2 table conditioning indexes by R (ignored by the MLP path)
    )

    kdata_lambda = batch['kdata_lambda']
    mask_lambda = batch['mask_lambda']
    k_pred = mri_forward_with_mask(x_recon, batch['coilmap'], mask_lambda)
    # lambda_v=0 -> plain SSDU; >0 -> + velocity-difference (phase) term (lv0/lv1 knob).
    loss = ssdu_phase_loss(
        k_pred, kdata_lambda,
        mask_lambda=mask_lambda, lambda_v=lambda_v,
    )
    if apply_scanner_weight:
        loss = loss * float(batch['scanner_weight'])
    return loss, {'R': batch['R'], 'seg_idx': batch['seg_idx']}


# ────────────────────────────────────────────────────────────────────────────
# Validation: SSDU k-space residual on val set (no scanner weighting).
# ────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def validate(
    model: CMRxTransformerCascade,
    val_dataset: CMRx4DFlowDataset,
    device: torch.device,
    max_samples: int = -1,
    desc: str = 'val',
    lambda_v: float = 0.0,
) -> dict:
    """Validation pass: average SSDU k-space residual across val patients.

    With ``seed_mode='deterministic_by_path'`` on the val dataset each
    patient always sees the same R + mask + SSDU partition + venc +
    cardiac window, so the metric is stable across runs.
    """
    model.eval()
    total_loss = 0.0
    n = 0
    n_iter = len(val_dataset) if max_samples < 0 else min(max_samples, len(val_dataset))
    pbar = tqdm(
        range(n_iter), desc=desc,
        dynamic_ncols=True, unit='sample', leave=False,
        mininterval=1.0, smoothing=0.1, file=sys.stdout,
    )
    for i in pbar:
        sample = val_dataset[i]
        # Deterministic FE slice: middle of the volume per patient so the
        # metric is consistent across epochs (instead of always FE=0).
        fe_size = sample['img_zf'].shape[-1]
        fe_idx = fe_size // 2
        batch = _slice_to_tensors(sample, device, fe_idx=fe_idx)
        loss, _ = training_step(
            model, batch, device, apply_scanner_weight=False, lambda_v=lambda_v,
        )
        total_loss += float(loss.item())
        n += 1
        pbar.set_postfix(
            loss=f'{(total_loss / max(n, 1)):.4f}',
            R=batch['R'],
        )
    model.train()
    return {'val_loss': total_loss / max(n, 1), 'n_val_samples': n}


@torch.no_grad()
def gate_table(model, r_list=(10, 20, 30, 40, 50), b0=3.0):
    """Per-R, per-stage learned gate values — the conditioning analysis figure.

    v[i]    = sigma(noise_lvl_i + delta_v_i(m))     (DC trust)
    para[i] = sigma(para0_i     + delta_para_i(m))  (WA blend)
    with m = [R/50, B0/3]. Unconditioned model -> delta=0 -> gates flat across R
    (the expected contrast). Returns a list of dict rows; also carries per-block
    gamma norms when Phase 2 (use_film) is on. Reusable on any loaded checkpoint.
    """
    model.eval()
    device = next(model.parameters()).device
    rows = []
    for R in r_list:
        dv = dp = None
        gnorm = {}
        if getattr(model, 'cond_table', None) is not None:      # v2: table -> WA gate only
            dp = model.cond_table(R)                           # (n_stages,); dv stays None
        elif getattr(model, 'conditioning_enabled', False) and getattr(model, 'cond_mlp', None) is not None:
            off = model.cond_mlp(build_m(model.conditioning_inputs, R, b0, device))
            dv, dp = off['delta_v'][0], off['delta_para'][0]
            if off['gamma'] is not None:
                g = off['gamma'][0]                            # (n_blocks, d_model)
                gnorm = {f'gamma{b}_norm': float(g[b].norm()) for b in range(g.shape[0])}
        for i in range(model.n_stages):
            dc, wa = model.dc[i], model.wa[i]
            v = torch.sigmoid(dc.noise_lvl + (0.0 if dv is None else dv[i]))
            if dc.min_v > 0.0:
                v = dc.min_v + (1.0 - dc.min_v) * v
            para = torch.sigmoid(wa.para + (0.0 if dp is None else dp[i]))
            if wa.min_para > 0.0:
                para = wa.min_para + (1.0 - wa.min_para) * para
            rows.append({'R': R, 'stage': i, 'v': float(v), 'para': float(para), **gnorm})
    model.train()
    return rows


# ────────────────────────────────────────────────────────────────────────────
# LR schedule (warmup + cosine).
# ────────────────────────────────────────────────────────────────────────────


def _lr_lambda(epoch: int, warmup_epochs: int, total_epochs: int, eta_min: float, lr: float) -> float:
    if epoch < warmup_epochs:
        return max((epoch + 1) / max(warmup_epochs, 1), 1e-6)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return eta_min / lr + (1 - eta_min / lr) * cosine


# ────────────────────────────────────────────────────────────────────────────
# Main training loop.
# ────────────────────────────────────────────────────────────────────────────


class _EMA:
    """Exponential moving average of model weights (params + buffers).

    Opt-in via cfg['ema'] = {'enabled': True, 'decay': 0.999}. The EMA weights are
    validated and saved alongside the raw weights so both can be A/B'd offline;
    model SELECTION stays on the raw SSDU val loss (unchanged).
    """

    def __init__(self, model, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model) -> None:
        d = self.decay
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.is_floating_point() or v.is_complex():
                s.mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                s.copy_(v)          # int/bool buffers: track latest
    def state_dict(self):
        return self.shadow

    def load(self, sd) -> None:
        for k, v in sd.items():
            if k in self.shadow:
                self.shadow[k].copy_(v)


def main() -> int:
    parser = argparse.ArgumentParser(description='CMRx4DFlow training')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to a checkpoint to resume from.')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Smoke-test cap on total optimiser steps.')
    cli = parser.parse_args()

    with open(cli.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Cluster convenience: let env vars override path keys without editing yaml.
    import os
    for _env, _key in (('CMRX_TRAIN_ROOT', 'train_root'),
                       ('CMRX_CACHE_DIR', 'patient_cache_dir'),
                       ('CMRX_SAVE_DIR', 'save_dir')):
        _v = os.environ.get(_env)
        if _v:
            cfg[_key] = _v
            print(f'[env-override] {_key} = {_v}')

    seed = int(cfg.get('seed', 1337))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[device] {device}')
    # TF32 on NVIDIA Ampere+ (H100/A30): big matmul speedup, harmless elsewhere.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    save_dir = Path(cfg['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    log_path = save_dir / 'train_log.csv'
    if not log_path.exists():
        with open(log_path, 'w', newline='') as f:
            _csv.writer(f).writerow([
                'step', 'epoch', 'lr', 'train_loss', 'val_loss',
                'epoch_time_sec', 'gpu_memory_peak_gb',
            ])

    # ── Datasets ─────────────────────────────────────────────────────────
    train_root = cfg['train_root']

    usrate_list = list(cfg.get('usrate_list', [10, 20, 30, 40, 50]))
    T_size = int(cfg.get('T_size', 5))
    D_size = int(cfg.get('D_size', 1))
    rho = float(cfg.get('rho', 0.2))
    ssdu_r2 = int(cfg.get('ssdu_r2', 9))
    orientation = str(cfg.get('orientation', 'canonical'))   # 'canonical' (flip_origin) | 'native'
    supervised = bool(cfg.get('supervised', False))
    print(f'[data] orientation: {orientation}  supervised: {supervised}')

    # ── Train/val split: deterministic, per-scanner hold-out (replaces the
    #    old static JSON). val_per_scanner patients are held out per scanner
    #    (every scanner appears in val); each is evaluated at every R in
    #    val_r_list -> (patient x R) val samples. All knobs are GA-tunable.
    val_per_scanner = int(cfg.get('val_per_scanner', 1))
    split_seed = int(cfg.get('split_seed', seed))
    val_r_list = list(cfg.get('val_r_list', usrate_list))
    _prospective = str(cfg.get('dataset', 'retrospective')).lower() == 'prospective'
    if not _prospective:                      # prospective builds its own patient split below
        train_dirs, val_dirs = make_split(
            train_root, val_per_scanner=val_per_scanner, seed=split_seed,
        )
        print(f'[split] {len(train_dirs)} train / {len(val_dirs)} val patients '
              f'(val_per_scanner={val_per_scanner}, split_seed={split_seed})')
        print(f'[split] val R per patient: {val_r_list} '
              f'-> {len(val_dirs) * len(val_r_list)} val (patient,R) samples')

    # ── Prospective SSDU fine-tuning (TaskS1/S2): shipped undersampled k-space +
    #    shipped Omega mask, NO ground truth. Patient-level holdout for validation. ──
    if _prospective:
        from src.data.prospective import ProspectiveSSDUDataset
        if supervised:
            raise ValueError("dataset='prospective' has no ground truth; set supervised: false")
        # S1 root = .../TaskS1/ValidationSet/Aorta      -> centre/vendor/Pxxx  (*/*/P*)
        # S2 root = .../TaskS2/ValidationSet            -> organ/centre/vendor/Pxxx (*/*/*/P*)
        all_dirs = sorted({Path(p) for pat in ('*/*/P*', '*/*/*/P*')
                           for p in glob.glob(str(Path(train_root) / pat)) if Path(p).is_dir()})
        if not all_dirs:
            raise FileNotFoundError(f'no patients under {train_root}/(*/*/P* | */*/*/P*)')
        n_val = int(cfg.get('val_patients', 6))
        rng = random.Random(split_seed)
        # STRATIFIED val split: S2 roots are Organ/Center/Vendor/P* — a plain shuffle
        # can leave an organ with zero val patients (4 anatomies, 6 val slots). Group
        # by the organ (top-level component; S1's Center/Vendor/P* collapses to one
        # group), shuffle within groups, then draw val ROUND-ROBIN across organs so
        # every anatomy is represented before any gets a second slot.
        _groups: dict[str, list[Path]] = {}
        for _p in all_dirs:
            _rel = _p.relative_to(train_root).parts
            _groups.setdefault(_rel[0] if len(_rel) >= 4 else '_all', []).append(_p)
        for _g in _groups.values():
            rng.shuffle(_g)
        _names = sorted(_groups)
        val_dirs = []
        _gi = 0
        while len(val_dirs) < n_val and any(_groups[_n] for _n in _names):
            _name = _names[_gi % len(_names)]; _gi += 1
            if _groups[_name]:
                val_dirs.append(_groups[_name].pop())
        train_dirs = [p for _n in _names for p in _groups[_n]]
        rng.shuffle(train_dirs)
        _vg = {}
        for _p in val_dirs:
            _rel = _p.relative_to(train_root).parts
            _vg[_rel[0] if len(_rel) >= 4 else '_all'] = \
                _vg.get(_rel[0] if len(_rel) >= 4 else '_all', 0) + 1
        print(f'[split] stratified val by group: {_vg}')
        _ds_kw = dict(T_size=T_size, rho=rho, ssdu_r2=ssdu_r2, orientation=orientation)
        train_dataset = ProspectiveSSDUDataset(train_dirs, mode='train', D_size=D_size, **_ds_kw)
        val_dataset = ProspectiveSSDUDataset(val_dirs, mode='val', **_ds_kw)
        print(f'[split] PROSPECTIVE {len(train_dirs)} train / {len(val_dirs)} val patients '
              f'(orientation={orientation}, root={train_root})')
        print(f'[data] training entries:   {len(train_dataset)}')
        print(f'[data] validation entries: {len(val_dataset)}')
    else:
        train_dataset = val_dataset = None

    patient_cache_dir = cfg.get('patient_cache_dir')
    if train_dataset is None:
        train_dataset = CMRx4DFlowDataset(
            patient_dirs=train_dirs,
            mode='train', D_size=D_size, T_size=T_size,
            usrate_list=usrate_list, rho=rho, ssdu_r2=ssdu_r2,
            seed_mode='random', supervised=supervised,
            cache_dir=patient_cache_dir, orientation=orientation,
            masks_per_sample=int(cfg.get('masks_per_sample', 1)),   # N: Omega-masks per (slice,R)
            r_per_sample=int(cfg.get('r_per_sample', 1)),           # M: R-values per slice
            fixed_masks=bool(cfg.get('fixed_masks', False)),        # deterministic N*M per slice
            require_cache=bool(cfg.get('require_cache', False)),    # raise (not .mat) if a .pt is missing
            oversample_rule=cfg.get('oversample_rule'),             # per-SCANNER rebalancing (v2)
        )
        val_dataset = CMRx4DFlowDataset(
            patient_dirs=val_dirs,
            mode='val', D_size=-1, T_size=T_size,
            usrate_list=usrate_list, rho=rho, ssdu_r2=ssdu_r2,
            seed_mode='deterministic_by_path', supervised=supervised,
            cache_dir=patient_cache_dir, orientation=orientation,
            val_r_list=val_r_list,
            require_cache=bool(cfg.get('require_cache', False)),
        )
        print(f'[data] training entries:   {len(train_dataset)}')
        print(f'[data] validation entries: {len(val_dataset)}')

    # ── Model ─────────────────────────────────────────────────────────────
    # block_ckpt nests a checkpoint inside grad_check's checkpoint. torch.compile'd
    # kernels can save a DIFFERENT tensor set between the original forward and the
    # nested recompute (inductor decides what to stash) -> CheckpointError "A
    # different number of tensors was saved". Force eager for such runs — the env
    # flags are read lazily at first forward, so disabling here is early enough.
    if bool(cfg.get('block_ckpt', False)):
        import os as _os_f
        for _v in ('CMRX_COMPILE_NORM', 'CMRX_COMPILE_ATTN'):
            if _os_f.environ.get(_v, '') not in ('', '0'):
                print(f'[fused] {_v} force-disabled: incompatible with block_ckpt '
                      f'(nested checkpoint recompute would mismatch saved tensors)')
            _os_f.environ[_v] = '0'
    model = CMRxTransformerCascade(
        n_stages=int(cfg.get('n_stages', 6)),
        d_model=int(cfg.get('d_model', 48)),
        n_heads=int(cfg.get('n_heads', 4)),
        n_blocks=int(cfg.get('n_blocks', 2)),
        mlp_ratio=int(cfg.get('mlp_ratio', 2)),
        patch_size=int(cfg.get('patch_size', 2)),
        in_channels=int(cfg.get('in_channels', 1)),
        K_pe=int(cfg.get('K_pe', 4)),
        K_cardiac=int(cfg.get('K_cardiac', 4)),
        dc_min_v=float(cfg.get('dc_min_v', 0.0)),
        wa_min_para=float(cfg.get('wa_min_para', 0.0)),
        init_scheme=str(cfg.get('init_scheme', 'uniform')),
        pe_learnable=bool(cfg.get('pe_learnable', False)),
        unshared_mlp=bool(cfg.get('unshared_mlp', False)),
        unshared_attn=bool(cfg.get('unshared_attn', False)),
        mlp_neg_bias_init=cfg.get('mlp_neg_bias_init'),
        embed_kernel=int(cfg.get('embed_kernel', 1)),
        denoiser=str(cfg.get('denoiser', 'transformer')),
        frame_chunk=int(cfg.get('frame_chunk', 0)),
        bcrnn_nf=int(cfg.get('bcrnn_nf', 24)),
        bcrnn_per_encoding=bool(cfg.get('bcrnn_per_encoding', True)),
        block_ckpt=bool(cfg.get('block_ckpt', False)),
        conditioning=cfg.get('conditioning'),
    )
    model.to(device)
    if getattr(model, 'conditioning_enabled', False):
        n_den = sum(p.numel() for p in model.denoiser.parameters())
        if getattr(model, 'cond_table', None) is not None:      # v2: anchored lookup table
            tbl = model.cond_table
            n_c = tbl.dp_table.numel()
            print(f"[cond] TABLE inputs={model.conditioning_inputs} r_values={tbl.r_values} "
                  f"anchor_R={tbl.anchor_R} | WA-gate only (DC + FiLM off) "
                  f"| params={n_c:,} ({100*n_c/max(n_den,1):.2f}% of denoiser)")
        elif getattr(model, 'cond_mlp', None) is not None:      # legacy MLP path (p1/p12)
            n_c = sum(p.numel() for p in model.cond_mlp.parameters())
            print(f"[cond] MLP enabled inputs={model.conditioning_inputs} use_film={model.use_film} "
                  f"| MLP params={n_c:,} ({100*n_c/max(n_den,1):.1f}% of denoiser)")
        else:
            print("[cond] enabled but no conditioning module built — check conditioning.mode")
    else:
        print("[cond] disabled (unconditioned baseline)")
    print(f"[pe] spatial_pe=norm K={cfg.get('K_pe', 4)} "
          f"learnable={bool(cfg.get('pe_learnable', False))}")

    # ── Parameter count ──────────────────────────────────────────────────
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_real = sum(
        2 * p.numel() if p.is_complex() else p.numel() for p in model.parameters()
    )
    print(f'[params] total numel: {total:,}')
    print(f'[params] trainable numel: {trainable:,}')
    print(f'[params] total real-equivalent: {total_real:,}')

    # ── Optimiser ────────────────────────────────────────────────────────
    lr = float(cfg.get('lr', 1e-4))
    wd = float(cfg.get('weight_decay', 0.0))
    param_groups = [p for p in model.parameters() if p.requires_grad]
    if wd > 0.0:
        opt = torch.optim.AdamW(param_groups, lr=lr, weight_decay=wd)
    else:
        opt = torch.optim.Adam(param_groups, lr=lr)
    n_epochs = int(cfg.get('n_epochs', cfg.get('epoch', 50)))
    warmup_epochs = int(cfg.get('warmup_epochs', 2))
    eta_min = float(cfg.get('eta_min', 1e-6))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda ep: _lr_lambda(ep, warmup_epochs, n_epochs, eta_min, lr),
    )

    # ── DataLoader ───────────────────────────────────────────────────────
    # Use BlockShuffleSampler so the dataset's @lru_cache(maxsize=2) becomes
    # effectively perfect — each patient's .pt is loaded ONCE per epoch
    # (i.e. ~122 disk loads instead of ~14k). set_epoch(epoch) is called per
    # epoch below to re-seed the shuffle deterministically.
    # IO-safe: always block-shuffle (each patient .pt loaded ONCE per epoch).
    # Scanner imbalance is handled via LOSS WEIGHTING (sampling_alpha, below),
    # NOT random resampling — random per-sample draws would thrash the patient
    # cache (thousands of 2.5 GB random reads/epoch on Lustre).
    # sampler: 'block' (default, unchanged) yields one patient at a time;
    # 'interleave' keeps num_workers patients in flight so an accumulation window
    # spans several subjects and each .pt is read once per epoch instead of once
    # per worker. Ordering only — no effect on the model or checkpoints.
    _num_workers = int(cfg.get('num_workers', 0))
    _sampler_kind = str(cfg.get('sampler', 'block')).lower()
    if _sampler_kind == 'interleave':
        train_sampler = PatientInterleaveSampler(
            train_dataset.patient_to_entries,
            num_workers=max(1, _num_workers), seed=seed,
        )
    elif _sampler_kind == 'block':
        train_sampler = BlockShuffleSampler(
            train_dataset.patient_to_entries, seed=seed,
        )
    else:
        raise ValueError(f"sampler must be 'block' or 'interleave'; got {_sampler_kind!r}")
    print(f'[data] sampler: {_sampler_kind} (num_workers={_num_workers})')

    def _seed_worker(worker_id):
        # PyTorch reseeds torch + Python `random` per worker (and per epoch), but NOT
        # numpy — and our masks use numpy. Without this, all workers draw IDENTICAL masks
        # and the same set every epoch, defeating fixed_masks=false ("fresh mask") training.
        import numpy as _np
        _np.random.seed(torch.initial_seed() % (2 ** 32))

    train_loader = DataLoader(
        train_dataset, batch_size=1, sampler=train_sampler,
        num_workers=_num_workers,
        pin_memory=False,
        collate_fn=lambda b: b[0],
        worker_init_fn=_seed_worker,
    )

    # ── Mixed precision ──────────────────────────────────────────────────
    use_amp = bool(cfg.get('mixed_precision', False)) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    print(f'[amp] mixed_precision={use_amp}')

    # Scanner-imbalance handling via per-sample LOSS WEIGHTING (IO-safe).
    #   sampling_alpha = 1.0 -> natural (all weights 1.0, no balancing)
    #                    0.5 -> inverse-sqrt (recommended, softened)
    #                    0.0 -> full inverse-frequency (equal mass/scanner)
    # weight(scanner) = count^(alpha-1), normalized so the MEAN per-sample
    # weight = 1 (keeps loss/LR scale stable regardless of alpha).
    from collections import Counter as _Counter
    sampling_alpha = float(cfg.get('sampling_alpha', 1.0))
    _counts = _Counter(
        f'{Path(p).parts[-3]}/{Path(p).parts[-2]}'
        for (p, _fe, _r) in train_dataset.filename
    )
    _ntot = sum(_counts.values())
    _denom = sum(c ** sampling_alpha for c in _counts.values()) or 1.0
    scanner_alpha_weight = {
        s: (c ** (sampling_alpha - 1.0)) * _ntot / _denom
        for s, c in _counts.items()
    }
    apply_scanner_weight = (sampling_alpha < 0.999)
    print(f'[loss] scanner-balance alpha={sampling_alpha} apply={apply_scanner_weight} '
          f'weights={ {k: round(v, 3) for k, v in scanner_alpha_weight.items()} }')

    # ── Resume ───────────────────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    best_val = float('inf')
    if cli.resume:
        ck = torch.load(cli.resume, map_location='cpu')
        model.load_state_dict(ck['model'])
        # Pre-fusion checkpoints carry three separate Q/K/V params per attention
        # block; remap their optimizer moments onto the fused W_QKV (no-op for
        # checkpoints already in the fused layout).
        opt.load_state_dict(fuse_qkv_optimizer_state(ck['optim'], model))
        sched.load_state_dict(ck['sched'])
        start_epoch = int(ck.get('epoch', 0))
        global_step = int(ck.get('step', 0))
        best_val = float(ck.get('best_val', float('inf')))
        print(f'[resume] loaded {cli.resume} at epoch={start_epoch} step={global_step}')

    # ── EMA (opt-in): weight average, validated/saved alongside raw weights ──
    ema = None
    _ema_cfg = cfg.get('ema') or {}
    if _ema_cfg.get('enabled', False):
        ema = _EMA(model, decay=float(_ema_cfg.get('decay', 0.999)))
        if cli.resume and 'ema_model' in ck:
            ema.load(ck['ema_model']); print('[ema] restored EMA weights from resume ckpt')
        print(f'[ema] enabled, decay={ema.decay}')

    val_every = int(cfg.get('val_every', 5))
    grad_clip = float(cfg.get('grad_clip', 1.0))
    accum_steps = int(cfg.get('batch_size', 1))
    lambda_v = float(cfg.get('lambda_v', 0.0))   # 0 = plain SSDU (lv0); >0 = + velocity term (lv1)
    max_steps_cap = cli.max_steps
    val_max_samples = int(cfg.get('val_max_samples', -1))

    # ── Training loop ────────────────────────────────────────────────────
    stop = False
    for epoch in range(start_epoch, n_epochs):
        if stop:
            break
        model.train()
        epoch_start = _time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        epoch_loss = 0.0
        epoch_n = 0
        opt.zero_grad(set_to_none=True)
        # Re-seed the BlockShuffleSampler so each epoch gets a different
        # (but deterministic-given-seed) patient + FE-slice order.
        train_sampler.set_epoch(epoch)
        _step_t0 = _time.time()
        pbar = tqdm(
            train_loader,
            desc=f'train epoch {epoch + 1}/{n_epochs}',
            dynamic_ncols=True, unit='step', leave=False,
            mininterval=1.0, smoothing=0.1, file=sys.stdout,
        )
        for i, raw_sample in enumerate(pbar):
            # A training draw is N*M mask/R realizations of ONE slice (list) — or a single
            # dict when masks_per_sample=r_per_sample=1. Average them into one accumulated
            # gradient: each realization backprops loss/(accum_steps*n_real).
            reals = raw_sample if isinstance(raw_sample, list) else [raw_sample]
            n_real = len(reals)
            slice_loss = 0.0
            for r_raw in reals:
                batch = _slice_to_tensors(r_raw, device)
                if apply_scanner_weight:
                    _sk = (f'{Path(batch["patient_dir"]).parts[-3]}/'
                           f'{Path(batch["patient_dir"]).parts[-2]}')
                    batch['scanner_weight'] = scanner_alpha_weight.get(_sk, 1.0)
                if use_amp:
                    with torch.cuda.amp.autocast():
                        loss, _ = training_step(
                            model, batch, device,
                            apply_scanner_weight=apply_scanner_weight, lambda_v=lambda_v,
                        )
                    scaler.scale(loss / (accum_steps * n_real)).backward()
                else:
                    loss, _ = training_step(
                        model, batch, device,
                        apply_scanner_weight=apply_scanner_weight, lambda_v=lambda_v,
                    )
                    (loss / (accum_steps * n_real)).backward()
                slice_loss += float(loss.item())
            loss_val = slice_loss / n_real   # mean loss over the slice's realizations (logging)

            epoch_loss += loss_val
            epoch_n += 1
            global_step += 1

            if global_step % accum_steps == 0:
                if use_amp:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    opt.step()
                opt.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)

            if max_steps_cap is not None and global_step >= max_steps_cap:
                print(f'[smoke-test] reached --max-steps={max_steps_cap}; stopping')
                stop = True
                break

            _step_dt = _time.time() - _step_t0
            _step_t0 = _time.time()
            # Live per-step status on the tqdm bar (throttled by mininterval=1s).
            pbar.set_postfix(
                loss=f'{loss_val:.4f}',
                R=batch['R'], n=n_real,
                dt=f'{_step_dt:.2f}s',
            )
            # During smoke tests (max-steps-cap set) print every step so we
            # can measure steady-state step time. Otherwise every 25 steps to
            # keep discrete checkpoints in the log file for offline analysis.
            _print_every = 1 if max_steps_cap is not None else 25
            if global_step % _print_every == 0:
                print(f'  step {global_step:5d} | loss {loss_val:.4f} '
                      f'| R {batch["R"]} n {n_real} '
                      f'| dt {_step_dt:.2f}s')

        # ── Flush any partial gradient accumulation at epoch end ────────
        # Otherwise the (epoch_n % accum_steps) tail samples have their
        # gradients silently dropped on the floor each epoch.
        if global_step % accum_steps != 0:
            if use_amp:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
            opt.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)

        sched.step()
        elapsed = _time.time() - epoch_start
        train_loss = epoch_loss / max(epoch_n, 1)
        gpu_mem = (
            torch.cuda.max_memory_allocated() / 1e9
            if torch.cuda.is_available() else 0.0
        )

        # ── Validation ───────────────────────────────────────────────
        val_loss: float | None = None
        if (
            not stop and val_dataset is not None
            and ((epoch + 1) % val_every == 0 or epoch == n_epochs - 1)
        ):
            vmetrics = validate(
                model, val_dataset, device, max_samples=val_max_samples,
                desc=f'val epoch {epoch + 1}/{n_epochs}', lambda_v=lambda_v,
            )
            val_loss = float(vmetrics['val_loss'])
            if ema is not None:
                _bak = {k: v.detach().clone() for k, v in model.state_dict().items()}
                model.load_state_dict(ema.shadow, strict=True)
                _vm_ema = validate(
                    model, val_dataset, device, max_samples=val_max_samples,
                    desc=f'val-ema {epoch + 1}/{n_epochs}', lambda_v=lambda_v,
                )
                model.load_state_dict(_bak, strict=True)
                print(f'[ema] val(ema)={float(_vm_ema["val_loss"]):.4f}  val(raw)={val_loss:.4f}')
            if val_loss < best_val:            # selection UNCHANGED: raw SSDU val loss
                best_val = val_loss
                ckpt = {
                    'model': model.state_dict(),
                    'optim': opt.state_dict(),
                    'sched': sched.state_dict(),
                    'epoch': epoch + 1,
                    'step': global_step,
                    'best_val': best_val,
                    'cfg': cfg,
                }
                if ema is not None:
                    ckpt['ema_model'] = ema.state_dict()
                torch.save(ckpt, save_dir / 'best.ckpt')
                print(f'[ckpt] new best val={best_val:.4f} -> {save_dir / "best.ckpt"}')

        # ── Log ──────────────────────────────────────────────────────
        cur_lr = opt.param_groups[0]['lr']
        with open(log_path, 'a', newline='') as f:
            _csv.writer(f).writerow([
                global_step, epoch + 1, f'{cur_lr:.3e}',
                f'{train_loss:.6f}',
                f'{val_loss:.6f}' if val_loss is not None else '',
                f'{elapsed:.1f}',
                f'{gpu_mem:.2f}',
            ])
        val_str = f'{val_loss:.4f}' if val_loss is not None else 'N/A'
        print(
            f'[epoch {epoch + 1:3d}/{n_epochs}] train={train_loss:.4f} '
            f'val={val_str} lr={cur_lr:.2e} time={elapsed:.1f}s '
            f'gpu={gpu_mem:.2f}GB'
        )

        # ── Latest checkpoint (every epoch, resume-friendly) ────────
        latest_ckpt = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'opt': opt.state_dict(),
            'sched': sched.state_dict(),
            'global_step': global_step,
            # Extras for resume parity with the periodic checkpoints.
            'optim': opt.state_dict(),
            'step': global_step,
            'best_val': best_val,
            'cfg': cfg,
        }
        if ema is not None:
            latest_ckpt['ema_model'] = ema.state_dict()
        torch.save(latest_ckpt, save_dir / 'latest.ckpt')
        # Plain-text progress marker so the SLURM wrapper can check completion
        # without loading torch (works for conda/apptainer/enroot runtimes).
        (save_dir / 'progress.txt').write_text(str(epoch + 1))

        # ── Periodic checkpoint every 5 epochs ──────────────────────
        if (epoch + 1) % 5 == 0 or epoch == n_epochs - 1:
            ckpt = {
                'model': model.state_dict(),
                'optim': opt.state_dict(),
                'sched': sched.state_dict(),
                'epoch': epoch + 1,
                'step': global_step,
                'best_val': best_val,
                'cfg': cfg,
            }
            if ema is not None:
                ckpt['ema_model'] = ema.state_dict()
            ck_path = save_dir / f'epoch_{epoch + 1:03d}.ckpt'
            torch.save(ckpt, ck_path)
            print(f'[ckpt] periodic -> {ck_path}')

    print(f'[done] best_val={best_val:.6f}')

    # ── Per-R gate curves for the conditioning analysis figure ──
    try:
        rows = gate_table(model)
        gpath = save_dir / 'gate_curves.csv'
        with open(gpath, 'w', newline='') as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'[gates] wrote per-R gate curves -> {gpath}')
    except Exception as e:  # never let a diagnostic break a finished run
        print(f'[gates] skipped gate_curves.csv: {e}')

    return 0


if __name__ == '__main__':
    sys.exit(main())

"""Training loop for CUT-4DFlow.

Config-driven (YAML). SSDU loss on the held-out Lambda partition, deterministic per-path
validation, checkpointing (best + periodic). Path keys may be overridden from the environment:
CMRX_TRAIN_ROOT, CMRX_CACHE_DIR, CMRX_SAVE_DIR.
"""

from __future__ import annotations

import argparse
import csv as _csv
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
from src.models.cascade import CMRxTransformerCascade
from src.models.conditioning import build_m
from src.training.loss import ssdu_phase_loss
from src.utils.mri_ops import fftc2d


def mri_forward_with_mask(
    img: torch.Tensor,        # (B, V, T, SPE, PE) complex
    coil: torch.Tensor,       # (B, C, SPE, PE)   complex
    mask: torch.Tensor,       # (B, V, C, T, SPE, PE) float / bool
) -> torch.Tensor:
    """Coil sens -> centred FFT2 over (SPE,PE) -> mask in k-space. Returns (B,V,C,T,SPE,PE)."""
    coil_imgs = img.unsqueeze(2) * coil.unsqueeze(1).unsqueeze(3)
    k = fftc2d(coil_imgs)
    return mask * k


def _slice_to_tensors(sample: dict, device: torch.device, fe_idx: int | None = None) -> dict:
    """Wrap the dataset's ndarray sample into batched tensors. ``fe_idx`` selects the FE slice."""
    fe = 0 if fe_idx is None else int(fe_idx)
    img_zf = torch.from_numpy(sample['img_zf'])[..., fe].unsqueeze(0).to(device)
    img_gt = torch.from_numpy(sample['img_gt'])[..., fe].unsqueeze(0).to(device)

    kl = torch.from_numpy(sample['kdata_lambda'])[..., fe].permute(0, 2, 1, 3, 4).unsqueeze(0).to(device)
    kt = torch.from_numpy(sample['kdata_theta'])[..., fe].permute(0, 2, 1, 3, 4).unsqueeze(0).to(device)

    ml = torch.from_numpy(sample['mask_lambda'])[..., 0].permute(0, 2, 1, 3, 4).unsqueeze(0).to(device)
    mth = torch.from_numpy(sample['mask_theta'])[..., 0].permute(0, 2, 1, 3, 4).unsqueeze(0).to(device)

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


def training_step(
    model: CMRxTransformerCascade,
    batch: dict,
    device: torch.device,
    apply_scanner_weight: bool = True,
    lambda_v: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """One forward + loss on a batched sample from ``_slice_to_tensors``."""
    Nt = int(batch['kdata_theta'].shape[3])                       # tensor T dim is authoritative
    phases = (batch['cardiac_bins'].to(torch.float32) / max(int(Nt), 1)).unsqueeze(0)

    m = None
    if getattr(model, 'conditioning_enabled', False):
        m = build_m(model.conditioning_inputs, batch['R'], batch.get('B0'), device)

    x_recon = model(batch['img_zf'], batch['kdata_theta'], batch['coilmap'],
                    batch['mask_theta'], phases, m=m, R=batch['R'])

    k_pred = mri_forward_with_mask(x_recon, batch['coilmap'], batch['mask_lambda'])
    loss = ssdu_phase_loss(k_pred, batch['kdata_lambda'],
                           mask_lambda=batch['mask_lambda'], lambda_v=lambda_v)
    if apply_scanner_weight:
        loss = loss * float(batch['scanner_weight'])
    return loss, {'R': batch['R'], 'seg_idx': batch['seg_idx']}


@torch.no_grad()
def validate(
    model: CMRxTransformerCascade,
    val_dataset: CMRx4DFlowDataset,
    device: torch.device,
    max_samples: int = -1,
    desc: str = 'val',
    lambda_v: float = 0.0,
) -> dict:
    """Average SSDU residual over the validation set (deterministic per path)."""
    model.eval()
    total_loss = 0.0
    n = 0
    n_iter = len(val_dataset) if max_samples < 0 else min(max_samples, len(val_dataset))
    pbar = tqdm(range(n_iter), desc=desc, dynamic_ncols=True, unit='sample', leave=False,
                mininterval=1.0, smoothing=0.1, file=sys.stdout)
    for i in pbar:
        sample = val_dataset[i]
        fe_idx = sample['img_zf'].shape[-1] // 2
        batch = _slice_to_tensors(sample, device, fe_idx=fe_idx)
        loss, _ = training_step(model, batch, device, apply_scanner_weight=False, lambda_v=lambda_v)
        total_loss += float(loss.item())
        n += 1
        pbar.set_postfix(loss=f'{(total_loss / max(n, 1)):.4f}', R=batch['R'])
    model.train()
    return {'val_loss': total_loss / max(n, 1), 'n_val_samples': n}


@torch.no_grad()
def gate_table(model, r_list=(10, 20, 30, 40, 50), b0=3.0):
    """Per-R, per-stage learned gate values (v = DC trust, para = WA blend)."""
    model.eval()
    device = next(model.parameters()).device
    rows = []
    for R in r_list:
        dv = dp = None
        gnorm = {}
        if getattr(model, 'cond_table', None) is not None:
            dp = model.cond_table(R)
        elif getattr(model, 'conditioning_enabled', False) and getattr(model, 'cond_mlp', None) is not None:
            off = model.cond_mlp(build_m(model.conditioning_inputs, R, b0, device))
            dv, dp = off['delta_v'][0], off['delta_para'][0]
            if off['gamma'] is not None:
                g = off['gamma'][0]
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


def _lr_lambda(epoch, warmup_epochs, total_epochs, eta_min, lr):
    """Linear warmup then cosine decay to eta_min."""
    if epoch < warmup_epochs:
        return max((epoch + 1) / max(warmup_epochs, 1), 1e-6)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return eta_min / lr + (1 - eta_min / lr) * cosine


def main() -> int:
    parser = argparse.ArgumentParser(description='CUT-4DFlow training')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--max-steps', type=int, default=None, help='cap total optimiser steps (smoke test)')
    cli = parser.parse_args()

    with open(cli.config, 'r') as f:
        cfg = yaml.safe_load(f)

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
            _csv.writer(f).writerow(['step', 'epoch', 'lr', 'train_loss', 'val_loss',
                                     'epoch_time_sec', 'gpu_memory_peak_gb'])

    # ── Datasets ──
    train_root = cfg['train_root']
    usrate_list = list(cfg.get('usrate_list', [10, 20, 30, 40, 50]))
    T_size = int(cfg.get('T_size', 5))
    D_size = int(cfg.get('D_size', 1))
    rho = float(cfg.get('rho', 0.2))
    ssdu_r2 = int(cfg.get('ssdu_r2', 9))
    orientation = str(cfg.get('orientation', 'canonical'))
    supervised = bool(cfg.get('supervised', False))
    print(f'[data] orientation: {orientation}  supervised: {supervised}')

    val_per_scanner = int(cfg.get('val_per_scanner', 1))
    split_seed = int(cfg.get('split_seed', seed))
    val_r_list = list(cfg.get('val_r_list', usrate_list))
    train_dirs, val_dirs = make_split(train_root, val_per_scanner=val_per_scanner, seed=split_seed)
    print(f'[split] {len(train_dirs)} train / {len(val_dirs)} val patients')

    patient_cache_dir = cfg.get('patient_cache_dir')
    train_dataset = CMRx4DFlowDataset(
        patient_dirs=train_dirs, mode='train', D_size=D_size, T_size=T_size,
        usrate_list=usrate_list, rho=rho, ssdu_r2=ssdu_r2, seed_mode='random',
        supervised=supervised, cache_dir=patient_cache_dir, orientation=orientation,
        masks_per_sample=int(cfg.get('masks_per_sample', 1)),
        r_per_sample=int(cfg.get('r_per_sample', 1)),
        fixed_masks=bool(cfg.get('fixed_masks', False)),
        require_cache=bool(cfg.get('require_cache', False)),
        oversample_rule=cfg.get('oversample_rule'),
    )
    val_dataset = CMRx4DFlowDataset(
        patient_dirs=val_dirs, mode='val', D_size=-1, T_size=T_size,
        usrate_list=usrate_list, rho=rho, ssdu_r2=ssdu_r2, seed_mode='deterministic_by_path',
        supervised=supervised, cache_dir=patient_cache_dir, orientation=orientation,
        val_r_list=val_r_list, require_cache=bool(cfg.get('require_cache', False)),
    )
    print(f'[data] training entries:   {len(train_dataset)}')
    print(f'[data] validation entries: {len(val_dataset)}')

    # ── Model ──
    model = CMRxTransformerCascade(
        n_stages=int(cfg.get('n_stages', 6)), d_model=int(cfg.get('d_model', 48)),
        n_heads=int(cfg.get('n_heads', 4)), n_blocks=int(cfg.get('n_blocks', 2)),
        mlp_ratio=int(cfg.get('mlp_ratio', 2)), patch_size=int(cfg.get('patch_size', 2)),
        in_channels=int(cfg.get('in_channels', 1)), K_pe=int(cfg.get('K_pe', 4)),
        K_cardiac=int(cfg.get('K_cardiac', 4)), dc_min_v=float(cfg.get('dc_min_v', 0.0)),
        wa_min_para=float(cfg.get('wa_min_para', 0.0)), init_scheme=str(cfg.get('init_scheme', 'uniform')),
        pe_learnable=bool(cfg.get('pe_learnable', False)), conditioning=cfg.get('conditioning'),
    )
    model.to(device)

    total = sum(p.numel() for p in model.parameters())
    total_real = sum(2 * p.numel() if p.is_complex() else p.numel() for p in model.parameters())
    print(f'[params] total numel: {total:,}  real-equivalent: {total_real:,}')

    # ── Optimiser ──
    lr = float(cfg.get('lr', 1e-4))
    wd = float(cfg.get('weight_decay', 0.0))
    opt = (torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd) if wd > 0.0
           else torch.optim.Adam(model.parameters(), lr=lr))
    n_epochs = int(cfg.get('n_epochs', cfg.get('epoch', 50)))
    warmup_epochs = int(cfg.get('warmup_epochs', 2))
    eta_min = float(cfg.get('eta_min', 1e-6))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda ep: _lr_lambda(ep, warmup_epochs, n_epochs, eta_min, lr))

    # ── DataLoader (one patient's slices per block, so each .pt is read once per epoch) ──
    _num_workers = int(cfg.get('num_workers', 0))
    _sampler_kind = str(cfg.get('sampler', 'block')).lower()
    if _sampler_kind == 'interleave':
        train_sampler = PatientInterleaveSampler(train_dataset.patient_to_entries,
                                                 num_workers=max(1, _num_workers), seed=seed)
    elif _sampler_kind == 'block':
        train_sampler = BlockShuffleSampler(train_dataset.patient_to_entries, seed=seed)
    else:
        raise ValueError(f"sampler must be 'block' or 'interleave'; got {_sampler_kind!r}")
    print(f'[data] sampler: {_sampler_kind} (num_workers={_num_workers})')

    def _seed_worker(worker_id):
        # numpy is not reseeded per worker by default, but the masks use it
        import numpy as _np
        _np.random.seed(torch.initial_seed() % (2 ** 32))

    train_loader = DataLoader(train_dataset, batch_size=1, sampler=train_sampler,
                              num_workers=_num_workers, pin_memory=False,
                              collate_fn=lambda b: b[0], worker_init_fn=_seed_worker)

    use_amp = bool(cfg.get('mixed_precision', False)) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    print(f'[amp] mixed_precision={use_amp}')

    # ── Optional per-scanner loss weighting (alpha=1 disables it) ──
    from collections import Counter as _Counter
    sampling_alpha = float(cfg.get('sampling_alpha', 1.0))
    _counts = _Counter(f'{Path(p).parts[-3]}/{Path(p).parts[-2]}' for (p, _fe, _r) in train_dataset.filename)
    _ntot = sum(_counts.values())
    _denom = sum(c ** sampling_alpha for c in _counts.values()) or 1.0
    scanner_alpha_weight = {s: (c ** (sampling_alpha - 1.0)) * _ntot / _denom for s, c in _counts.items()}
    apply_scanner_weight = (sampling_alpha < 0.999)
    print(f'[loss] scanner-balance alpha={sampling_alpha} apply={apply_scanner_weight}')

    start_epoch = 0
    global_step = 0
    best_val = float('inf')
    if cli.resume:
        ck = torch.load(cli.resume, map_location='cpu')
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optim'])
        sched.load_state_dict(ck['sched'])
        start_epoch = int(ck.get('epoch', 0))
        global_step = int(ck.get('step', 0))
        best_val = float(ck.get('best_val', float('inf')))
        print(f'[resume] loaded {cli.resume} at epoch={start_epoch} step={global_step}')

    val_every = int(cfg.get('val_every', 5))
    grad_clip = float(cfg.get('grad_clip', 1.0))
    accum_steps = int(cfg.get('batch_size', 1))
    lambda_v = float(cfg.get('lambda_v', 0.0))
    max_steps_cap = cli.max_steps
    val_max_samples = int(cfg.get('val_max_samples', -1))

    # ── Training loop ──
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
        train_sampler.set_epoch(epoch)
        _step_t0 = _time.time()
        pbar = tqdm(train_loader, desc=f'train epoch {epoch + 1}/{n_epochs}', dynamic_ncols=True,
                    unit='step', leave=False, mininterval=1.0, smoothing=0.1, file=sys.stdout)
        for i, raw_sample in enumerate(pbar):
            reals = raw_sample if isinstance(raw_sample, list) else [raw_sample]
            n_real = len(reals)
            slice_loss = 0.0
            for r_raw in reals:
                batch = _slice_to_tensors(r_raw, device)
                if apply_scanner_weight:
                    _sk = f'{Path(batch["patient_dir"]).parts[-3]}/{Path(batch["patient_dir"]).parts[-2]}'
                    batch['scanner_weight'] = scanner_alpha_weight.get(_sk, 1.0)
                if use_amp:
                    with torch.cuda.amp.autocast():
                        loss, _ = training_step(model, batch, device,
                                                apply_scanner_weight=apply_scanner_weight, lambda_v=lambda_v)
                    scaler.scale(loss / (accum_steps * n_real)).backward()
                else:
                    loss, _ = training_step(model, batch, device,
                                            apply_scanner_weight=apply_scanner_weight, lambda_v=lambda_v)
                    (loss / (accum_steps * n_real)).backward()
                slice_loss += float(loss.item())
            loss_val = slice_loss / n_real

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

            if max_steps_cap is not None and global_step >= max_steps_cap:
                print(f'[smoke-test] reached --max-steps={max_steps_cap}; stopping')
                stop = True
                break

            _step_dt = _time.time() - _step_t0
            _step_t0 = _time.time()
            pbar.set_postfix(loss=f'{loss_val:.4f}', R=batch['R'], n=n_real, dt=f'{_step_dt:.2f}s')
            _print_every = 1 if max_steps_cap is not None else 25
            if global_step % _print_every == 0:
                print(f'  step {global_step:5d} | loss {loss_val:.4f} | R {batch["R"]} n {n_real} | dt {_step_dt:.2f}s')

        # flush a partial accumulation window at epoch end
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

        sched.step()
        elapsed = _time.time() - epoch_start
        train_loss = epoch_loss / max(epoch_n, 1)
        gpu_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

        val_loss = None
        if not stop and val_dataset is not None and ((epoch + 1) % val_every == 0 or epoch == n_epochs - 1):
            vmetrics = validate(model, val_dataset, device, max_samples=val_max_samples,
                                desc=f'val epoch {epoch + 1}/{n_epochs}', lambda_v=lambda_v)
            val_loss = float(vmetrics['val_loss'])
            if val_loss < best_val:
                best_val = val_loss
                torch.save({'model': model.state_dict(), 'optim': opt.state_dict(),
                            'sched': sched.state_dict(), 'epoch': epoch + 1, 'step': global_step,
                            'best_val': best_val, 'cfg': cfg}, save_dir / 'best.ckpt')
                print(f'[ckpt] new best val={best_val:.4f} -> {save_dir / "best.ckpt"}')

        cur_lr = opt.param_groups[0]['lr']
        with open(log_path, 'a', newline='') as f:
            _csv.writer(f).writerow([global_step, epoch + 1, f'{cur_lr:.3e}', f'{train_loss:.6f}',
                                     f'{val_loss:.6f}' if val_loss is not None else '',
                                     f'{elapsed:.1f}', f'{gpu_mem:.2f}'])
        val_str = f'{val_loss:.4f}' if val_loss is not None else 'N/A'
        print(f'[epoch {epoch + 1:3d}/{n_epochs}] train={train_loss:.4f} val={val_str} '
              f'lr={cur_lr:.2e} time={elapsed:.1f}s gpu={gpu_mem:.2f}GB')

        torch.save({'epoch': epoch + 1, 'model': model.state_dict(), 'optim': opt.state_dict(),
                    'sched': sched.state_dict(), 'step': global_step, 'best_val': best_val, 'cfg': cfg},
                   save_dir / 'latest.ckpt')
        (save_dir / 'progress.txt').write_text(str(epoch + 1))

        if (epoch + 1) % 5 == 0 or epoch == n_epochs - 1:
            ck_path = save_dir / f'epoch_{epoch + 1:03d}.ckpt'
            torch.save({'model': model.state_dict(), 'optim': opt.state_dict(),
                        'sched': sched.state_dict(), 'epoch': epoch + 1, 'step': global_step,
                        'best_val': best_val, 'cfg': cfg}, ck_path)
            print(f'[ckpt] periodic -> {ck_path}')

    print(f'[done] best_val={best_val:.6f}')

    try:
        rows = gate_table(model)
        gpath = save_dir / 'gate_curves.csv'
        with open(gpath, 'w', newline='') as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'[gates] wrote per-R gate curves -> {gpath}')
    except Exception as e:
        print(f'[gates] skipped gate_curves.csv: {e}')

    return 0


if __name__ == '__main__':
    sys.exit(main())

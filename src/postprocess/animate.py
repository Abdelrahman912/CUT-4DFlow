"""Per-case cine animations — DIAGNOSTIC ONLY, never part of a submission.

Kept out of ``batch_recon`` so the submission path carries no matplotlib import,
no animation code and no zero-filled-volume dependency: ``batch_recon --anim``
imports this module lazily, and a plain submission run never touches it.

    from src.postprocess.animate import animate_case
"""
from __future__ import annotations

import numpy as np

from src.utils.cmrx_metrics import complex2magflow


def animate_case(key, rec_nat, zf_nat, seg, venc, out_gif, R=None, fps=6):
    """3-panel cine over the cardiac cycle: anatomy (recon |img|) / blood (recon |v|) / zf (|v|)."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib.animation as manim
    Nv, Nt, SPE, PE, FE = rec_nat.shape
    z = int(seg.sum(axis=(1, 2)).argmax()) if seg.any() else SPE // 2
    seg2d = seg[z]
    mag_r, _ = complex2magflow(rec_nat, None)
    _, flow_r = complex2magflow(rec_nat, venc); sp_r = np.linalg.norm(flow_r, axis=0)[:, z]      # (Nt,PE,FE)
    _, flow_z = complex2magflow(zf_nat, venc);  sp_z = np.linalg.norm(flow_z, axis=0)[:, z]
    mmax = float(np.percentile(mag_r[0, :, z], 99)) + 1e-6
    vmax = (float(np.percentile(sp_r[:, seg2d], 99)) if seg2d.any() else 1.0) + 1e-6
    panels = [("anatomy |img|", mag_r[0, :, z], "gray", 0, mmax),
              ("blood |v|", sp_r * seg2d[None], "jet", 0, vmax),
              ("zero-filled |v|", sp_z * seg2d[None], "jet", 0, vmax)]
    fig, ax = plt.subplots(1, 3, figsize=(9.5, 3.4), constrained_layout=True)
    ims = [ax[j].imshow(p[1][0], cmap=p[2], vmin=p[3], vmax=p[4]) for j, p in enumerate(panels)]
    for j, p in enumerate(panels):
        ax[j].set_title(p[0], fontsize=9); ax[j].set_xticks([]); ax[j].set_yticks([])
    ttl = fig.suptitle("")
    def upd(t):
        for j, p in enumerate(panels): ims[j].set_data(p[1][t])
        ttl.set_text(f"{key}  R={R}  SPE={z}  frame {t+1}/{Nt}")
        return ims
    anim = manim.FuncAnimation(fig, upd, frames=Nt, interval=1000 / fps, blit=False)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_gif), writer=manim.PillowWriter(fps=fps)); plt.close(fig)

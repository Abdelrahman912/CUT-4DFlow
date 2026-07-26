"""CS-LLR classical-baseline recon adapter — run the official CS-LLR on OUR sample.

Bridges our dataset sample to the official ``CS_LLR`` wrapper: builds the full
acquired Ω k-space (= Θ+Λ), reshapes to CS-LLR's ``(V,C,T,FE,PE,SPE)`` convention,
runs it with the official hyperparameters (lamb_tv=0, lamb_llr=0.5), and reshapes
the result back to our ``(Nv,Nt,SPE,PE,FE)`` layout.

NOTE: CS-LLR requires a CUDA/HIP device — the official wrapper hard-codes
``torch.cuda.*`` (synchronize / max_memory) calls, so it cannot run on CPU.
"""
from __future__ import annotations

import einops
import numpy as np
import scipy.fft

from src.baselines.cs_llr.cs_llr_exec import CS_LLR


def _fe_hybrid_to_kspace(x: np.ndarray) -> np.ndarray:
    """Undo the dataset's FE-axis iFFT: hybrid (FE in image) -> raw k-space (FE in k).

    Our dataset stores kdata in HYBRID space — the FE readout has already been
    inverse-FFT'd to image (only SPE/PE remain in k-space). The official CS-LLR
    wrapper expects RAW k-space and re-applies ``k2i`` over FE internally, so we
    must put FE back in k-space first (centered ortho FFT, the exact inverse of
    ``src.data.dataset._k2i_numpy`` over the last axis) — otherwise FE is
    transformed twice and the recon collapses to a streak.
    """
    ax = (-1,)
    return scipy.fft.fftshift(
        scipy.fft.fftn(scipy.fft.ifftshift(x, axes=ax), axes=ax, norm="ortho", workers=-1),
        axes=ax,
    )


def recon_cs_llr(sample, device: str = "cuda:0",
                 lamb_tv: float = 0.0, lamb_llr: float = 0.5) -> np.ndarray:
    """CS-LLR recon of the full acquired Ω k-space. Returns (Nv,Nt,SPE,PE,FE) complex."""
    k_omega = sample["kdata_theta"] + sample["kdata_lambda"]            # (Nv,Nt,Nc,SPE,PE,FE) hybrid
    k_omega = _fe_hybrid_to_kspace(k_omega).astype(np.complex64)        # -> raw k-space (FE in k)
    ksp = einops.rearrange(k_omega, "nv nt nc spe pe fe -> nv nc nt fe pe spe")
    coils = einops.rearrange(sample["coilmap"], "nc spe pe fe -> nc 1 fe pe spe")
    rec = CS_LLR(lamb_tv, lamb_llr, ksp, coils, seg=True, dev=device)   # (Nv,Nt,FE,PE,SPE)
    return einops.rearrange(rec, "nv nt fe pe spe -> nv nt spe pe fe").astype(np.complex64)

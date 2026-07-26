"""Centred 2-D FFT operators (orthonormal) over the last two axes."""

import numpy as np
import torch


def fftc2d(x):
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
    x = torch.fft.fftshift(x, dim=[-1, -2])
    return x


def ifftc2d(x):
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.ifft2(x, dim=(-2, -1), norm="ortho")
    x = torch.fft.fftshift(x, dim=[-1, -2])
    return x


def mriAdjointOp(rawdata, sens, mask):
    """Adjoint: masked k-space -> coil-combined image."""
    coil_sens = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(rawdata * mask), norm="ortho"))
    return np.sum(coil_sens * np.conj(sens), axis=1)


def mri_forward_op(u, coil_sens, sampling_mask):
    """Forward: image -> sampled k-space."""
    coil_imgs = u.unsqueeze(2) * coil_sens.unsqueeze(1).unsqueeze(3)
    Fu = fftc2d(coil_imgs)
    return sampling_mask.unsqueeze(2).unsqueeze(4) * Fu

"""Data-consistency (DC) and weighted-average (WA) blocks.

Both are closed-form steps with a single learnable gate. DC blends the network's k-space with
the acquired samples (trust ``sigma(noise_lvl)``); WA blends the denoised image with the DC
output (blend ``sigma(para)``).
"""

import torch
import torch.nn as nn
from src.utils.mri_ops import fftc2d, ifftc2d


class dataConsistencyTerm(nn.Module):

    def __init__(self, noise_lvl=None, min_v=0.0):
        super().__init__()
        self.min_v = float(min_v)
        self.noise_lvl = noise_lvl
        if noise_lvl is not None:
            self.noise_lvl = torch.nn.Parameter(data=torch.Tensor([noise_lvl]))

    def perform(self, x, k0, sensitivity, delta=None):
        """x: image (N,V,T,H,W); k0: acquired k-space; sensitivity: coil maps (N,C,D,H,W).

        ``delta``: optional conditioning offset added to ``noise_lvl`` before the sigmoid.
        """
        x = sensitivity[:, :, None, None] * x[:, None, :, :, None]
        k = fftc2d(x)

        if self.noise_lvl is not None:
            logit = self.noise_lvl if delta is None else (self.noise_lvl + delta)
            v_sig = torch.sigmoid(logit)
            v = self.min_v + (1.0 - self.min_v) * v_sig if self.min_v > 0.0 else v_sig
            out = torch.where(k0 != 0, v * k + (1 - v) * k0, k)
        else:
            out = torch.where(k0 != 0, k0, k)

        x = ifftc2d(out)
        Sx = torch.sum(x * sensitivity.conj()[:, :, None, None], axis=1)
        return Sx


class weightedAverageTerm(nn.Module):

    def __init__(self, para=None, min_para=0.0):
        super().__init__()
        self.min_para = float(min_para)
        self.para = para
        if para is not None:
            self.para = torch.nn.Parameter(torch.Tensor([para]))

    def perform(self, cnn, Sx, delta=None):
        """Blend denoised image ``cnn`` with DC output ``Sx``. ``delta``: optional offset."""
        logit = self.para if delta is None else (self.para + delta)
        para_sig = torch.sigmoid(logit)
        para = self.min_para + (1.0 - self.min_para) * para_sig if self.min_para > 0.0 else para_sig
        x = para * cnn + (1 - para) * Sx
        return x

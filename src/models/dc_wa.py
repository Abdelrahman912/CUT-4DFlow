"""
Data Consistency and Weighted Average blocks.
Copied verbatim from flowmri_net-main/networks/flowmri_net.py (lines 166-207).
"""

import torch
import torch.nn as nn
from src.utils.mri_ops import fftc2d, ifftc2d


class dataConsistencyTerm(nn.Module):

    def __init__(self, noise_lvl=None, min_v=0.0):
        """min_v: optional lower bound on σ(noise_lvl). 0.0 = default unbounded."""
        super(dataConsistencyTerm, self).__init__()
        self.min_v = float(min_v)
        self.noise_lvl = noise_lvl
        if noise_lvl is not None:
            self.noise_lvl = torch.nn.Parameter(data=torch.Tensor([noise_lvl]))

    def perform(self, x, k0, sensitivity, delta=None):
        """
        x    - input in image space (N V T H W)
        k0   - initially sampled elements in k-space
        sensitivity - coil sensitivities (N C D H W)
        delta - optional conditioning offset added to noise_lvl PRE-sigmoid
                (v = sigmoid(noise_lvl + delta)); None -> baseline.
        """
        x = sensitivity[:, :, None, None] * x[:, None, :, :, None]
        k = fftc2d(x)

        if self.noise_lvl is not None:
            logit = self.noise_lvl if delta is None else (self.noise_lvl + delta)
            v_sig = torch.sigmoid(logit)
            if self.min_v > 0.0:
                # rescale (0, 1) → [min_v, 1], preserving differentiability
                v = self.min_v + (1.0 - self.min_v) * v_sig
            else:
                v = v_sig
            out = torch.where(k0 != 0, v * k + (1 - v) * k0, k)
        else:
            out = torch.where(k0 != 0, k0, k)

        x = ifftc2d(out)
        Sx = torch.sum(x * sensitivity.conj()[:, :, None, None], axis=1)
        return Sx


class weightedAverageTerm(nn.Module):

    def __init__(self, para=None, min_para=0.0):
        """min_para: optional lower bound on σ(para). 0.0 = default unbounded."""
        super(weightedAverageTerm, self).__init__()
        self.min_para = float(min_para)
        self.para = para
        if para is not None:
            self.para = torch.nn.Parameter(torch.Tensor([para]))

    def perform(self, cnn, Sx, delta=None):
        """delta - optional conditioning offset added to para PRE-sigmoid
        (para = sigmoid(para0 + delta)); None -> baseline."""
        logit = self.para if delta is None else (self.para + delta)
        para_sig = torch.sigmoid(logit)
        if self.min_para > 0.0:
            para = self.min_para + (1.0 - self.min_para) * para_sig
        else:
            para = para_sig
        x = para * cnn + (1 - para) * Sx
        return x

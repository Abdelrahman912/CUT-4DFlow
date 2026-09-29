"""FlowMRI-Net's BCRNN regularizer as a drop-in denoiser for CMRxTransformerCascade.

Faithful port of networks/flowmri_net.py (flowmri_net-main), which itself follows
Deep-MRI-Reconstruction's CRNN-MRI: 4 stacked bidirectional-in-time BCRNN layers
(complex64, k=3, spatial modReLU) + a 3x3 complex output conv producing the residual
x_cnn. The 4 hidden feature maps RECUR ACROSS UNROLLED ITERATIONS (the ih2ih path):
this module keeps them as internal state, reset when the cascade calls stage_idx == 0.

Matched-capacity comparison for the paper: the regularizer's hidden width ``nf`` is
the single capacity knob (their published default: 24). ``per_encoding=True``
reproduces the original single-channel design (features_in=1): the V velocity
encodings are folded into the batch and processed independently — coupling between
encodings comes from DC/WA and the loss, exactly as in FlowMRI-Net.

Interface-compatible with CMRxTransformerDenoiser:
    forward(x (B,V,T,SPE,PE) complex, phases, gammas=None, stage_idx=0) -> delta_x
(phases/gammas accepted and ignored: the BCRNN has no PE / no conditioning.)

NOTE: incompatible with cascade grad_check — checkpoint recomputation would advance
the cross-stage hidden state twice. The cascade refuses that combination.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# The unfold path exists ONLY for gfx906/ROCm (no MIOpen conv kernels). It materialises a
# 9x patch tensor per conv — retained for backward, it OOMs the unrolled BCRNN even on an
# 80GB H100. On CUDA/cuDNN we use F.conv2d on real/imag components instead (no patch
# materialisation, ~9x less activation memory, faster). Same math either way.
_IS_ROCM = torch.version.hip is not None


def _complex_conv2d(x, weight, bias, pad):
    """Complex 2D conv, reflect-padded. cuDNN path on NVIDIA; unfold+einsum on ROCm."""
    xr = F.pad(x.real, [pad, pad, pad, pad], mode="reflect")
    xi = F.pad(x.imag, [pad, pad, pad, pad], mode="reflect")
    if not (_IS_ROCM and x.is_cuda):
        # (Wr + iWi)(xr + ixi) as ONE real conv2d with the block weight
        # [[Wr, -Wi], [Wi, Wr]] over stacked [xr; xi] channels (1 launch, not 4).
        wr, wi = weight.real, weight.imag
        w2 = torch.cat([torch.cat([wr, -wi], dim=1), torch.cat([wi, wr], dim=1)], dim=0)
        out2 = F.conv2d(torch.cat([xr, xi], dim=1), w2)
        C_out = weight.shape[0]
        return torch.complex(out2[:, :C_out], out2[:, C_out:]) + bias.reshape(1, -1, 1, 1)
    B, C_in, H, W = x.shape
    C_out, _, kH, kW = weight.shape
    x_pad = torch.complex(xr, xi)
    patches = x_pad.unfold(2, kH, 1).unfold(3, kW, 1)          # (B, C_in, H, W, kH, kW)
    patches = patches.contiguous().reshape(B, C_in, H * W, kH * kW)
    w = weight.reshape(C_out, C_in, kH * kW)
    out = torch.einsum("oip,bilp->bol", w, patches).reshape(B, C_out, H, W)
    return out + bias.reshape(1, -1, 1, 1)


class _SpatialModReLU(nn.Module):
    """modReLU over channel-first spatial maps (B, nf, H, W) — as in FlowMRI-Net."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros((1, hidden_size, 1, 1)))

    def forward(self, x):
        return x * F.relu(torch.abs(x) + self.bias) / (torch.abs(x) + 1e-6)


class CRNNcell(nn.Module):
    """input conv + hidden(time) conv + hidden(iteration) conv -> modReLU.

    The three convs are FUSED into one at forward time (conv is linear:
    sum of convs == conv of channel-concatenated inputs/weights) — exact same
    math and identical parameter tensors/keys, ~3x fewer kernel launches.
    """

    def __init__(self, input_size: int, hidden_size: int, kernel_size: int):
        super().__init__()
        self.pad = kernel_size // 2
        z = lambda *s: torch.randn(*s, dtype=torch.complex64) * 0.02
        self.i2h_w = nn.Parameter(z(hidden_size, input_size, kernel_size, kernel_size))
        self.i2h_b = nn.Parameter(torch.zeros(hidden_size, dtype=torch.complex64))
        self.h2h_w = nn.Parameter(z(hidden_size, hidden_size, kernel_size, kernel_size))
        self.h2h_b = nn.Parameter(torch.zeros(hidden_size, dtype=torch.complex64))
        self.ih2ih_w = nn.Parameter(z(hidden_size, hidden_size, kernel_size, kernel_size))
        self.ih2ih_b = nn.Parameter(torch.zeros(hidden_size, dtype=torch.complex64))
        self.act = _SpatialModReLU(hidden_size)

    def forward(self, input, hidden_iteration, hidden):
        z = torch.cat([input, hidden_iteration, hidden], dim=1)
        w = torch.cat([self.i2h_w, self.ih2ih_w, self.h2h_w], dim=1)
        b = self.i2h_b + self.ih2ih_b + self.h2h_b
        return self.act(_complex_conv2d(z, w, b, self.pad))


class BCRNNlayer(nn.Module):
    """Bidirectional-in-time CRNN layer (one shared cell, forward + backward sums)."""

    def __init__(self, input_size: int, hidden_size: int, kernel_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.CRNN_model = CRNNcell(input_size, hidden_size, kernel_size)

    def forward(self, input, input_iteration):
        nb, nc, nt, nx, ny = input.shape
        # Forward and backward time sweeps are independent chains through the SAME
        # cell -> run them as one batch of 2*nb (backward = time-flipped input),
        # halving the sequential depth. Exact same math as two separate sweeps.
        z2 = torch.cat([input, input.flip(2)], dim=0)
        it2 = torch.cat([input_iteration, input_iteration.flip(2)], dim=0)
        hidden = input.new_zeros((2 * nb, self.hidden_size, nx, ny))
        outs = []
        for i in range(nt):
            hidden = self.CRNN_model(z2[:, :, i], it2[:, :, i], hidden)
            outs.append(hidden)
        o = torch.stack(outs, dim=2)                       # (2nb, nf, nt, nx, ny)
        return o[:nb] + o[nb:].flip(2)                     # forward + time-unflipped backward


class BCRNNDenoiser(nn.Module):
    """FlowMRI-Net regularizer (4 BCRNN layers + output conv) for the CMRx cascade."""

    def __init__(self, in_channels: int = 4, nf: int = 24, per_encoding: bool = True, **_):
        super().__init__()
        self.in_channels = int(in_channels)
        self.nf = int(nf)
        self.per_encoding = bool(per_encoding)
        n_ch = 1 if self.per_encoding else self.in_channels   # features_in (original: 1)
        self.n_ch = n_ch

        self.bcrnn1 = BCRNNlayer(n_ch, self.nf, 3)
        self.bcrnn2 = BCRNNlayer(self.nf, self.nf, 3)
        self.bcrnn3 = BCRNNlayer(self.nf, self.nf, 3)
        self.bcrnn4 = BCRNNlayer(self.nf, self.nf, 3)
        self.output_w = nn.Parameter(torch.randn(n_ch, self.nf, 3, 3, dtype=torch.complex64) * 0.02)
        self.output_b = nn.Parameter(torch.zeros(n_ch, dtype=torch.complex64))

        self._hidden = None            # cross-stage (ih2ih) state; reset at stage_idx 0
        self._init_weights()
        # conv shapes are fixed per scanner (a handful total) -> let cuDNN autotune
        torch.backends.cudnn.benchmark = True

    def _init_weights(self) -> None:
        """FlowMRI-Net init: complex conv weights ~ N(0, 0.012), all biases 0."""
        for p in self.parameters():
            if p.is_complex():
                if p.dim() == 4:
                    p.data.real.normal_(0, 0.012)
                    p.data.imag.normal_(0, 0.012)
                else:
                    p.data.zero_()
            else:
                p.data.zero_()          # spatial modReLU biases

    def forward(self, x, phases=None, gammas=None, stage_idx: int = 0):
        """x: (B, V, T, SPE, PE) complex64 -> residual delta_x, same shape."""
        B, V, T, H, W = x.shape
        if self.per_encoding:
            z = x.reshape(B * V, 1, T, H, W)                 # V folded into batch
        else:
            z = x

        nb = z.shape[0]
        if stage_idx == 0 or self._hidden is None or self._hidden[0].shape[0] != nb \
                or self._hidden[0].shape[-2:] != (H, W):
            hid = z.new_zeros((nb, self.nf, T, H, W))
            self._hidden = (hid, hid, hid, hid)

        h1 = self.bcrnn1(z, self._hidden[0])
        h2 = self.bcrnn2(h1, self._hidden[1])
        h3 = self.bcrnn3(h2, self._hidden[2])
        h4 = self.bcrnn4(h3, self._hidden[3])
        self._hidden = (h1, h2, h3, h4)

        out = _complex_conv2d(
            h4.permute(0, 2, 1, 3, 4).reshape(nb * T, self.nf, H, W),
            self.output_w, self.output_b, pad=1,
        )
        delta = out.reshape(nb, T, self.n_ch, H, W).permute(0, 2, 1, 3, 4)
        if self.per_encoding:
            delta = delta.reshape(B, V, T, H, W)
        return delta

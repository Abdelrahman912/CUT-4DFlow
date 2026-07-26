"""Complex-valued layers (conv, linear, activation).

Convolutions are implemented with einsum rather than ``F.conv2d`` so they run on any
backend, including those without a native complex convolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._fused import maybe_compile


class ComplexConv2d(nn.Module):
    """Complex patch convolution (kernel == stride) via pixel-unshuffle + 1x1 einsum."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        assert stride == kernel_size, "only supports stride == kernel_size (patch conv)"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size,
                        dtype=torch.complex64) * 0.02
        )
        self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.complex64))

    def forward(self, x):
        # x: (B, C_in, H, W) complex
        B, C_in, H, W = x.shape
        k = self.kernel_size
        Hp, Wp = H // k, W // k

        x_p = x.reshape(B, C_in, Hp, k, Wp, k)
        x_p = x_p.permute(0, 1, 3, 5, 2, 4).contiguous()
        x_p = x_p.reshape(B, C_in * k * k, Hp, Wp)

        w_flat = self.weight.reshape(self.out_channels, C_in * k * k)
        out = torch.einsum('op,bphw->bohw', w_flat, x_p)

        if self.bias is not None:
            out = out + self.bias[None, :, None, None]
        return out


class ComplexConvTranspose2d(nn.Module):
    """Complex patch transposed convolution (kernel == stride) via 1x1 einsum + pixel-shuffle."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        assert stride == kernel_size, "only supports stride == kernel_size"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, kernel_size, kernel_size,
                        dtype=torch.complex64) * 0.02
        )
        self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.complex64))

    def forward(self, x):
        # x: (B, C_in, Hp, Wp) complex
        B, C_in, Hp, Wp = x.shape
        k = self.kernel_size
        C_out = self.out_channels

        w_flat = self.weight.reshape(C_in, C_out * k * k)
        out_flat = torch.einsum('cp,bchw->bphw', w_flat, x)

        out = out_flat.reshape(B, C_out, k, k, Hp, Wp)
        out = out.permute(0, 1, 4, 2, 5, 3).contiguous()
        out = out.reshape(B, C_out, Hp * k, Wp * k)

        if self.bias is not None:
            out = out + self.bias[None, :, None, None]
        return out


class ComplexUpsampleConv2d(nn.Module):
    """Nearest-neighbour upsample followed by a complex 1x1 or 3x3 conv (einsum).

    Avoids the checkerboard artefacts of a learned transposed convolution.
    """

    def __init__(self, in_channels, out_channels, scale_factor, kernel_size=3):
        super().__init__()
        assert kernel_size in (1, 3), "only 1x1 and 3x3 supported"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = scale_factor
        self.kernel_size = kernel_size

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size,
                        dtype=torch.complex64) * 0.02
        )
        self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.complex64))

    def forward(self, x):
        # x: (B, C_in, H, W) complex
        B, C_in, H, W = x.shape
        k = self.kernel_size
        s = self.scale
        C_out = self.out_channels

        x_up = x.repeat_interleave(s, dim=2).repeat_interleave(s, dim=3)
        Hs, Ws = H * s, W * s

        if k == 1:
            w_flat = self.weight.reshape(C_out, C_in)
            out = torch.einsum('oc,bchw->bohw', w_flat, x_up)
        else:
            x_pad = torch.zeros(B, C_in, Hs + 2, Ws + 2, dtype=x.dtype, device=x.device)
            x_pad[:, :, 1:Hs + 1, 1:Ws + 1] = x_up
            patches = torch.stack(
                [x_pad[:, :, i:i + Hs, j:j + Ws] for i in range(3) for j in range(3)],
                dim=2,
            )
            patches = patches.reshape(B, C_in * 9, Hs, Ws)
            w_flat = self.weight.reshape(C_out, C_in * 9)
            out = torch.einsum('op,bphw->bohw', w_flat, patches)

        out = out + self.bias[None, :, None, None]
        return out


class ComplexLinear(nn.Module):
    """Complex linear layer: ``x @ W^T + b``."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, dtype=torch.complex64) * 0.02
        )
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.complex64)) if bias else None

    def forward(self, x):
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out


def _modrelu_kernel(xr, bias, eps: float = 1e-8):
    """modReLU on a real ``(..., num_features, 2)`` view (kept complex-free for compile)."""
    re = xr[..., 0]
    im = xr[..., 1]
    mag = torch.hypot(re, im)
    gate = F.relu(mag + bias)
    denom = mag + eps
    return torch.stack((gate * re / denom, gate * im / denom), dim=-1)


def _modrelu_probe():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    return (torch.randn(4, 8, 2, device=dev, requires_grad=True),
            torch.randn(8, device=dev, requires_grad=True))


class modReLU(nn.Module):
    """Phase-preserving activation: ``ReLU(|z| + b) * z / (|z| + eps)`` with learnable bias b."""

    def __init__(self, num_features):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        kernel = maybe_compile(_modrelu_kernel, 'CMRX_COMPILE_ATTN', _modrelu_probe)
        return torch.view_as_complex(kernel(torch.view_as_real(x), self.bias))

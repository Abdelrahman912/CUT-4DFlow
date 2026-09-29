"""
Complex-valued operations for ROCm/gfx906 compatibility.

All convolutions use manual einsum (no F.conv2d / MIOpen) because
MIOpen on gfx906 (AMD Radeon VII) raises miopenStatusUnknownError.

For stride-2 conv: pixel-unshuffle + einsum (equivalent to strided conv).
For stride-2 transpose: einsum + pixel-shuffle.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._fused import maybe_compile

# ROCm/gfx906 has no MIOpen conv kernels -> keep the einsum fallbacks there.
# On CUDA (cuDNN) the fused real-conv paths below are exact and much leaner.
_IS_ROCM = torch.version.hip is not None


class ComplexConv2d(nn.Module):
    """Complex 2D convolution via manual pixel-unshuffle + einsum.

    For kernel=k, stride=k (non-overlapping patch conv):
      1. Rearrange input into k×k patches (pixel-unshuffle)
      2. 1×1 conv via einsum

    No MIOpen, no F.conv2d — works on gfx906.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        assert stride == kernel_size, (
            "Manual conv only supports stride == kernel_size (patch conv)"
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(
            torch.randn(
                out_channels, in_channels, kernel_size, kernel_size,
                dtype=torch.complex64,
            ) * 0.02
        )
        self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.complex64))

    def forward(self, x):
        # x: (B, C_in, H, W) complex
        B, C_in, H, W = x.shape
        k = self.kernel_size
        Hp, Wp = H // k, W // k

        # Pixel-unshuffle: extract non-overlapping k×k patches
        # (B, C_in, H, W) -> (B, C_in, Hp, k, Wp, k) -> (B, C_in*k*k, Hp, Wp)
        x_p = x.reshape(B, C_in, Hp, k, Wp, k)
        x_p = x_p.permute(0, 1, 3, 5, 2, 4).contiguous()  # (B, C_in, k, k, Hp, Wp)
        x_p = x_p.reshape(B, C_in * k * k, Hp, Wp)

        # 1x1 conv via einsum (no MIOpen)
        w_flat = self.weight.reshape(self.out_channels, C_in * k * k)
        out = torch.einsum('op,bphw->bohw', w_flat, x_p)

        if self.bias is not None:
            out = out + self.bias[None, :, None, None]
        return out


class ComplexConvTranspose2d(nn.Module):
    """Complex 2D transposed convolution via einsum + pixel-shuffle.

    For kernel=k, stride=k (non-overlapping patch deconv):
      1. 1×1 conv via einsum: (B, C_in, Hp, Wp) -> (B, C_out*k*k, Hp, Wp)
      2. Pixel-shuffle: -> (B, C_out, Hp*k, Wp*k)

    No MIOpen, no F.conv_transpose2d — works on gfx906.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        assert stride == kernel_size, (
            "Manual transpose conv only supports stride == kernel_size"
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        # ConvTranspose2d weight shape: (in_channels, out_channels, kH, kW)
        self.weight = nn.Parameter(
            torch.randn(
                in_channels, out_channels, kernel_size, kernel_size,
                dtype=torch.complex64,
            ) * 0.02
        )
        self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.complex64))

    def forward(self, x):
        # x: (B, C_in, Hp, Wp) complex
        B, C_in, Hp, Wp = x.shape
        k = self.kernel_size
        C_out = self.out_channels

        # 1x1 conv via einsum: (B, C_in, Hp, Wp) -> (B, C_out*k*k, Hp, Wp)
        w_flat = self.weight.reshape(C_in, C_out * k * k)
        out_flat = torch.einsum('cp,bchw->bphw', w_flat, x)

        # Pixel-shuffle: (B, C_out*k*k, Hp, Wp) -> (B, C_out, Hp*k, Wp*k)
        out = out_flat.reshape(B, C_out, k, k, Hp, Wp)
        out = out.permute(0, 1, 4, 2, 5, 3).contiguous()  # (B, C_out, Hp, k, Wp, k)
        out = out.reshape(B, C_out, Hp * k, Wp * k)

        if self.bias is not None:
            out = out + self.bias[None, :, None, None]
        return out


class ComplexUpsampleConv2d(nn.Module):
    """Nearest-neighbor upsample + complex 2D conv (kernel 1 or 3) via einsum.

    Drop-in replacement for ComplexConvTranspose2d to avoid checkerboard
    artifacts from learned subpixel weights (each subpixel position in a
    k×k block sharing the same filter response).

    No MIOpen, no F.conv2d, no F.pad — safe on gfx906.
    """

    def __init__(self, in_channels, out_channels, scale_factor, kernel_size=3):
        super().__init__()
        assert kernel_size in (1, 3), "Only 1x1 and 3x3 supported"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = scale_factor
        self.kernel_size = kernel_size

        self.weight = nn.Parameter(
            torch.randn(
                out_channels, in_channels, kernel_size, kernel_size,
                dtype=torch.complex64,
            ) * 0.02
        )
        self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.complex64))

    def forward(self, x):
        # x: (B, C_in, H, W) complex
        B, C_in, H, W = x.shape
        k = self.kernel_size
        s = self.scale
        C_out = self.out_channels

        # Nearest-neighbor upsample (memory op — complex-safe)
        x_up = x.repeat_interleave(s, dim=2).repeat_interleave(s, dim=3)
        Hs, Ws = H * s, W * s

        if k == 1:
            w_flat = self.weight.reshape(C_out, C_in)
            out = torch.einsum('oc,bchw->bohw', w_flat, x_up)
        elif not (_IS_ROCM and x.is_cuda):
            # cuDNN/CPU fast path: the complex conv as ONE real conv2d with the block
            # weight [[Wr, -Wi], [Wi, Wr]] over stacked [re; im] channels. Zero-pad=1
            # == the explicit zero-pad below, so the math is identical — but cuDNN
            # never materialises the 9x patch stack (saved-for-backward memory) and
            # runs one fused kernel instead of stack+reshape+einsum.
            wr, wi = self.weight.real, self.weight.imag
            w2 = torch.cat([torch.cat([wr, -wi], dim=1), torch.cat([wi, wr], dim=1)], dim=0)
            out2 = F.conv2d(torch.cat([x_up.real, x_up.imag], dim=1), w2, padding=1)
            out = torch.complex(out2[:, :C_out], out2[:, C_out:])
        else:
            # gfx906/ROCm fallback (no MIOpen): explicit shifts + einsum, unchanged.
            x_pad = torch.zeros(
                B, C_in, Hs + 2, Ws + 2, dtype=x.dtype, device=x.device,
            )
            x_pad[:, :, 1:Hs + 1, 1:Ws + 1] = x_up

            # Stack 9 shifts as explicit patches: patches[idx=i*3+j]
            patches = torch.stack(
                [x_pad[:, :, i:i + Hs, j:j + Ws]
                 for i in range(3) for j in range(3)],
                dim=2,
            )  # (B, C_in, 9, Hs, Ws)
            patches = patches.reshape(B, C_in * 9, Hs, Ws)

            # weight[o, c, i, j] with reshape order matches patches C_in*9
            w_flat = self.weight.reshape(C_out, C_in * 9)
            out = torch.einsum('op,bphw->bohw', w_flat, patches)

        out = out + self.bias[None, :, None, None]
        return out


class ComplexLinear(nn.Module):
    """Complex-valued linear layer: x @ W^T + b."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, dtype=torch.complex64) * 0.02
        )
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=torch.complex64)
            )
        else:
            self.bias = None

    def forward(self, x):
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out


def _modrelu_kernel(xr, bias, eps: float = 1e-8):
    """modReLU on a real ``(..., num_features, 2)`` view -> same shape.

    Complex-free so it is compilable; it is a pure elementwise chain, which is
    exactly what inductor fuses well (the eager version is ~16 separate kernels
    over the full activation).
    """
    re = xr[..., 0]
    im = xr[..., 1]
    # hypot, not sqrt(re*re + im*im): it matches what complex .abs() does, which
    # keeps the backward pass ~1.6x closer to the pre-change implementation.
    mag = torch.hypot(re, im)
    gate = F.relu(mag + bias)
    denom = mag + eps
    return torch.stack((gate * re / denom, gate * im / denom), dim=-1)


def _modrelu_probe():
    """Tiny sample args used to validate the compiled kernel before adopting it."""
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    return (torch.randn(4, 8, 2, device=dev, requires_grad=True),
            torch.randn(8, device=dev, requires_grad=True))


class modReLU(nn.Module):
    """Phase-preserving modReLU activation.

    modReLU(z) = ReLU(|z| + b) * z / (|z| + eps)
    where b is a learnable real bias per channel.
    """

    def __init__(self, num_features):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # x: (..., num_features) complex
        kernel = maybe_compile(_modrelu_kernel, 'CMRX_COMPILE_ATTN', _modrelu_probe)
        return torch.view_as_complex(kernel(torch.view_as_real(x), self.bias))

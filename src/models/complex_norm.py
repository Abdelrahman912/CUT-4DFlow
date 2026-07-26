"""Complex layer normalization with full 2x2 covariance whitening.

Following Eilers & Jiang (2023): the real/imaginary parts are whitened jointly by the
inverse square root of their 2x2 covariance, then scaled by a learnable positive-definite
matrix and shifted by a complex bias.

The maths lives in ``_cln_kernel``, written in real arithmetic on a ``(..., dim, 2)`` view
so it can be optionally ``torch.compile``'d (set ``CMRX_COMPILE_NORM=1``); default is eager.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._fused import maybe_compile


def _cln_kernel(xr, raw_a, raw_b, raw_c, beta_re, beta_im, eps, gamma):
    """CLN on a real ``(..., dim, 2)`` tensor -> ``(..., dim, 2)``."""
    # 1. complex mean over features
    mean = xr.mean(dim=-2, keepdim=True)
    xc = xr - mean
    re = xc[..., 0]
    im = xc[..., 1]

    # 2. 2x2 covariance per token
    vrr = (re * re).mean(dim=-1, keepdim=True).clamp(min=eps)
    vii = (im * im).mean(dim=-1, keepdim=True).clamp(min=eps)
    vri = (re * im).mean(dim=-1, keepdim=True)

    # 3. inverse square root of the covariance (closed-form 2x2)
    det = (vrr * vii - vri * vri).clamp(min=eps)
    s = torch.sqrt(det)
    sqrt_t = torch.sqrt((vrr + vii + 2 * s).clamp(min=eps))
    s_rr = (vrr + s) / sqrt_t
    s_ri = vri / sqrt_t
    s_ii = (vii + s) / sqrt_t
    det_sqrt = (s_rr * s_ii - s_ri * s_ri).clamp(min=eps)
    inv_rr = s_ii / det_sqrt
    inv_ri = -s_ri / det_sqrt
    inv_ii = s_rr / det_sqrt

    re_w = inv_rr * re + inv_ri * im
    im_w = inv_ri * re + inv_ii * im

    # 4. learnable positive-definite scale zeta^{1/2}
    a = F.softplus(raw_a)
    c = F.softplus(raw_c)
    b = torch.sqrt(a * c) * torch.tanh(raw_b)
    delta_z = (a * c - b * b).clamp(min=eps)
    s_z = torch.sqrt(delta_z)
    t_z = torch.sqrt((a + c + 2 * s_z).clamp(min=eps))
    z_rr = (a + s_z) / t_z
    z_ri = b / t_z
    z_ii = (c + s_z) / t_z

    # 5. scale + shift
    re_final = z_rr * re_w + z_ri * im_w + beta_re
    im_final = z_ri * re_w + z_ii * im_w + beta_im

    # 6. optional conditional magnitude scale (phase-preserving)
    if gamma is not None:
        scale = 1.0 + gamma
        re_final = re_final * scale
        im_final = im_final * scale

    return torch.stack((re_final, im_final), dim=-1)


def _cln_probe():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    r = lambda n: torch.randn(n, device=dev, requires_grad=True)  # noqa: E731
    return (torch.randn(4, 8, 2, device=dev, requires_grad=True),
            r(8), r(8), r(8), r(8), r(8), 1e-6, None)


def _get_kernel():
    return maybe_compile(_cln_kernel, 'CMRX_COMPILE_NORM', _cln_probe)


class ComplexLayerNorm2x2(nn.Module):
    """Complex layer norm with 2x2 covariance whitening, learnable scale and complex shift."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps

        self.beta_re = nn.Parameter(torch.zeros(dim))
        self.beta_im = nn.Parameter(torch.zeros(dim))

        # scale zeta initialised to identity: softplus(raw) = 1  =>  raw = log(e - 1)
        init_val = math.log(math.exp(1.0) - 1.0)
        self.raw_a = nn.Parameter(torch.full((dim,), init_val))
        self.raw_b = nn.Parameter(torch.zeros(dim))
        self.raw_c = nn.Parameter(torch.full((dim,), init_val))

    def forward(self, x, gamma=None):
        """x: (..., dim) complex64 -> (..., dim) complex64.

        ``gamma``: optional (dim,) real conditional magnitude scale (1 + gamma), applied
        identically to real and imaginary parts (phase-preserving). None -> unconditioned.
        """
        out = _get_kernel()(
            torch.view_as_real(x), self.raw_a, self.raw_b, self.raw_c,
            self.beta_re, self.beta_im, self.eps, gamma,
        )
        return torch.view_as_complex(out)

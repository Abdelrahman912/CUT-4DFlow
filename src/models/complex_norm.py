"""
Complex-valued Layer Normalization with full 2x2 covariance whitening.

Following Eilers & Jiang (2023) Eq. 14-16:
  [Re(CLN(X))]                   [Re(X - E(X))]
  [Im(CLN(X))] = Cov_C(X)^{-1/2} [Im(X - E(X))]

Then scaled by learnable PD matrix zeta^{1/2} and shifted by beta in C.

Performance note
----------------
This layer is the single largest optimisation target in the cascade: it runs
80x per forward (4 norms x 2 blocks x 10 stages) and, written as ~40 separate
tensor ops on complex tensors, expands to ~141 GPU kernels per call — each one
re-reading and re-writing the whole activation. Removing it entirely from an
H100 step saves 34% of the time, so fusing it is worth real effort.

The maths therefore lives in ``_cln_kernel``, which is deliberately free of
complex dtypes: it takes and returns a real ``(..., dim, 2)`` view. That matters
because inductor cannot lower complex64 — it reinterprets complex as pairs of
reals via ``aten.view.dtype``, which requires a stride-1 last dim and fails on
the cascade's permuted positional-encoding tensors. Keeping this function purely
real lets ``torch.compile`` fuse the elementwise chain even though the model as
a whole cannot be compiled.

Set ``CMRX_COMPILE_NORM=1`` to enable that fusion (default off, eager). The
reduction axis (``dim``) is static and only the token axis varies across
patients, so a single dynamic-shape compilation covers every subject.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._fused import maybe_compile


def _cln_kernel(xr, raw_a, raw_b, raw_c, beta_re, beta_im, eps, gamma):
    """CLN maths on a real ``(..., dim, 2)`` tensor -> ``(..., dim, 2)``.

    No complex dtype appears anywhere in here, so this is compilable.
    """
    # 1. Complex mean over features
    mean = xr.mean(dim=-2, keepdim=True)
    xc = xr - mean
    re = xc[..., 0]
    im = xc[..., 1]

    # 2. 2x2 covariance per token
    vrr = (re * re).mean(dim=-1, keepdim=True).clamp(min=eps)
    vii = (im * im).mean(dim=-1, keepdim=True).clamp(min=eps)
    vri = (re * im).mean(dim=-1, keepdim=True)

    # 3. C^{-1/2} via closed-form 2x2 matrix inverse square root
    #    (Trabelsi 2018 Appendix B)
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

    # 4. Learnable scale zeta^{1/2} (parameters only)
    a = F.softplus(raw_a)
    c = F.softplus(raw_c)
    b = torch.sqrt(a * c) * torch.tanh(raw_b)
    delta_z = (a * c - b * b).clamp(min=eps)
    s_z = torch.sqrt(delta_z)
    t_z = torch.sqrt((a + c + 2 * s_z).clamp(min=eps))
    z_rr = (a + s_z) / t_z
    z_ri = b / t_z
    z_ii = (c + s_z) / t_z

    # 5. Scale + shift
    re_final = z_rr * re_w + z_ri * im_w + beta_re
    im_final = z_ri * re_w + z_ii * im_w + beta_im

    # 6. Optional conditional real magnitude scale (FiLM scale, phase-safe).
    if gamma is not None:
        scale = 1.0 + gamma
        re_final = re_final * scale
        im_final = im_final * scale

    return torch.stack((re_final, im_final), dim=-1)


def _cln_probe():
    """Tiny sample args used to validate the compiled kernel before adopting it."""
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    r = lambda n: torch.randn(n, device=dev, requires_grad=True)  # noqa: E731
    return (torch.randn(4, 8, 2, device=dev, requires_grad=True),
            r(8), r(8), r(8), r(8), r(8), 1e-6, None)


def _get_kernel():
    """Return the CLN kernel, compiling it once if CMRX_COMPILE_NORM is set."""
    return maybe_compile(_cln_kernel, 'CMRX_COMPILE_NORM', _cln_probe)


class ComplexLayerNorm2x2(nn.Module):
    """Eilers-style complex LN with full 2x2 covariance whitening.

    Per token, across the d feature dimension:
    1. Compute complex mean and 2x2 real covariance over d features.
    2. Whiten via closed-form 2x2 matrix inverse square root.
    3. Scale by learnable PD matrix zeta^{1/2} and shift by complex beta.
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps

        # Learnable shift beta in C
        self.beta_re = nn.Parameter(torch.zeros(dim))
        self.beta_im = nn.Parameter(torch.zeros(dim))

        # Learnable scale zeta: PD 2x2 matrix per feature
        # a, c > 0 via softplus; b constrained via b = sqrt(a*c) * tanh(raw_b)
        # Init: zeta = I  =>  softplus(raw) = 1  =>  raw = log(e-1)
        init_val = math.log(math.exp(1.0) - 1.0)  # inv_softplus(1) ~ 0.5413
        self.raw_a = nn.Parameter(torch.full((dim,), init_val))
        self.raw_b = nn.Parameter(torch.zeros(dim))
        self.raw_c = nn.Parameter(torch.full((dim,), init_val))

    def forward(self, x, gamma=None):
        """x: (..., dim) complex64 -> (..., dim) complex64

        gamma: optional (dim,) REAL conditional scale. Applied as (1+gamma) to the
        WHOLE normalized output (zeta.whiten + beta), identically to Re and Im -> a
        pure per-channel magnitude scale (phase-exact: out_cond/out = 1+gamma). None
        -> unconditioned (baseline, bit-identical)."""
        out = _get_kernel()(
            torch.view_as_real(x), self.raw_a, self.raw_b, self.raw_c,
            self.beta_re, self.beta_im, self.eps, gamma,
        )
        return torch.view_as_complex(out)

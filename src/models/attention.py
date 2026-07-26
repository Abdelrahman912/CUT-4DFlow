"""C-Attention: complex attention with a real-part Hermitian-inner-product score.

The score is ``Re<Q, K> / sqrt(d_k)`` and the softmax weights are real, so applying them to V
scales V without rotating its phase, preserving the phase (velocity) information. This is the
CAtt variant of Eilers & Jiang (2023).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._fused import maybe_compile
from src.models.complex_ops import ComplexLinear


def _catt_kernel(Qr, Qi, Kr, Ki, Vr, Vi, scale):
    """CAtt core in real arithmetic -> (out_re, out_im)."""
    # Re<Q, K> = Qr@Kr^T + Qi@Ki^T
    score = (Qr @ Kr.transpose(-2, -1) + Qi @ Ki.transpose(-2, -1)) / scale
    w = F.softmax(score, dim=-1)
    return w @ Vr, w @ Vi


def _catt_probe():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    t = lambda: torch.randn(2, 2, 4, 4, device=dev, requires_grad=True)  # noqa: E731
    return (t(), t(), t(), t(), t(), t(), 2.0)


class CAtt(nn.Module):
    """Single-head C-Attention."""

    def __init__(self, d_k):
        super().__init__()
        self.scale = math.sqrt(d_k)

    def forward(self, Q, K, V):
        # Q, K, V: (..., N_tok, d_k) complex
        kernel = maybe_compile(_catt_kernel, 'CMRX_COMPILE_ATTN', _catt_probe)
        out_re, out_im = kernel(Q.real, Q.imag, K.real, K.imag, V.real, V.imag, self.scale)
        return torch.complex(out_re, out_im)


class MultiHeadCAtt(nn.Module):
    """Multi-head C-Attention with a fused Q/K/V projection."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_QKV = ComplexLinear(d_model, 3 * d_model, bias=False)
        self.W_O = ComplexLinear(d_model, d_model, bias=True)
        self.attn = CAtt(self.d_k)

    def forward(self, x):
        # x: (batch, seq_len, d_model) complex
        B, N, _ = x.shape
        qkv = self.W_QKV(x).view(B, N, 3, self.n_heads, self.d_k)
        Q, K, V = qkv.permute(2, 0, 3, 1, 4)          # each (B, heads, N, d_k)
        out = self.attn(Q, K, V)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, self.d_model)
        return self.W_O(out)

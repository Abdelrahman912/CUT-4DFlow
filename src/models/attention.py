"""
C-Attention (CAtt) — real-part Hermitian-inner-product softmax.

CAtt computes attention scores from the real part of the Hermitian
inner product: score = Re<Q, K> / sqrt(d_k). The softmax produces
REAL weights, so multiplying by V scales V without rotating its
phase — V's phase information (i.e. velocity) survives intact.

Empirically the strongest of the four complex-attention variants tested
in Eilers & Jiang (2023, arXiv:2306.09827, Table 2): CAtt > AAtt > RIAtt
> APAtt on both classification and sequence-generation benchmarks.
APAtt's per-entry phase factor sgn(<Q, K>) introduces destructive
interference in the V sum — the failure mode confirmed in our own
diagnostics (row phase coherence 0.4–0.7).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._fused import enabled, maybe_compile
from src.models.complex_ops import ComplexLinear


def _catt_kernel(Qr, Qi, Kr, Ki, Vr, Vi, scale):
    """CAtt core in pure real arithmetic -> (out_re, out_im).

    No complex dtype appears here, so this is compilable. The four matmuls stay
    as extern cuBLAS calls; what fusion buys is the surrounding plumbing (the
    add of the two score terms, the scale, and the softmax).
    """
    # Re<Q, K> = Qr@Kr^T + Qi@Ki^T -- two real GEMMs, not a full complex matmul
    # whose imaginary half would be discarded.
    score = (Qr @ Kr.transpose(-2, -1) + Qi @ Ki.transpose(-2, -1)) / scale
    # Real softmax -> real weights summing to 1 per row.
    w = F.softmax(score, dim=-1)
    # Real weights on complex V: applied to Re/Im separately (two real GEMMs)
    # rather than upcasting w and multiplying a zero imaginary part.
    return w @ Vr, w @ Vi


def _catt_sdpa(Q, K, V, scale):
    """CAtt via ``scaled_dot_product_attention`` on realified tensors.

    CAtt is *exactly* a real SDPA in disguise::

        score[i,j] = Qr[i].Kr[j] + Qi[i].Ki[j] = <[Qr|Qi][i], [Kr|Ki][j]>
        out        = (w@Vr, w@Vi)              = w @ [Vr|Vi]

    so concatenating re/im along the feature axis turns the whole thing into one
    real attention with head dim ``2*d_k``. ``scale`` stays ``1/sqrt(d_k)`` — the
    concatenation doubles the feature width but must NOT change the temperature.

    MEASURED ON H100: this is SLOWER — 265.5 -> 290.2 ms per fwd+bwd (~9%), and
    the attention ceiling rose 52% -> 55%. The hoped-for win was SDPA skipping the
    N x N score matrix, but in fp32 the flash backend is unavailable (it needs
    fp16/bf16) so no fused kernel engages, while the three ``torch.cat`` calls
    add real traffic by materialising doubled-width Q/K/V every call. Kept behind
    ``CMRX_SDPA_ATTN`` (default OFF) purely to document the negative result —
    do not enable without re-measuring.
    """
    Qp = torch.cat((Q.real, Q.imag), dim=-1)
    Kp = torch.cat((K.real, K.imag), dim=-1)
    Vp = torch.cat((V.real, V.imag), dim=-1)
    out = F.scaled_dot_product_attention(Qp, Kp, Vp, scale=1.0 / scale)
    d_k = V.shape[-1]
    return torch.complex(out[..., :d_k], out[..., d_k:])


def _catt_probe():
    """Tiny sample args used to validate the compiled kernel before adopting it."""
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    t = lambda: torch.randn(2, 2, 4, 4, device=dev, requires_grad=True)  # noqa: E731
    return (t(), t(), t(), t(), t(), t(), 2.0)


class CAtt(nn.Module):
    """Single-head C-Attention (real-part dot-product softmax)."""

    def __init__(self, d_k):
        super().__init__()
        self.scale = math.sqrt(d_k)

    def forward(self, Q, K, V):
        """
        Q, K, V: (..., N_tok, d_k) complex
        Returns:  (..., N_tok, d_k) complex
        """
        # CAtt: real-part of the Hermitian inner product as the score.
        # |Q||K| cos(phi_Q - phi_K) per entry — norm-weighted angular similarity.
        if enabled('CMRX_SDPA_ATTN'):
            return _catt_sdpa(Q, K, V, self.scale)
        kernel = maybe_compile(_catt_kernel, 'CMRX_COMPILE_ATTN', _catt_probe)
        out_re, out_im = kernel(
            Q.real, Q.imag, K.real, K.imag, V.real, V.imag, self.scale,
        )
        return torch.complex(out_re, out_im)


def fuse_qkv_optimizer_state(optim_sd: dict, model: nn.Module) -> dict:
    """Migrate a pre-fusion optimizer state dict to the fused-QKV layout.

    ``MultiHeadCAtt`` used to hold W_Q/W_K/W_V as three parameters and now holds
    one ``W_QKV`` = ``cat([W_Q, W_K, W_V], dim=0)``. Optimizer state is keyed by
    parameter INDEX, so a pre-fusion ``optim`` state has more entries than the
    fused model has parameters and ``load_state_dict`` rejects it outright.

    Adam/AdamW moments are elementwise, so the moment of the concatenated weight
    is the concatenation of the three moments — the migration is exact, and a
    resumed run keeps its momentum instead of restarting it.

    Returns ``optim_sd`` untouched when it already matches the model, so this is
    safe to call unconditionally.
    """
    new_names = [n for n, _ in model.named_parameters()]
    suffix = 'W_QKV.weight'
    # Expand back to pre-fusion order to recover how the checkpoint was indexed.
    old_names: list[str] = []
    for n in new_names:
        if n.endswith(suffix):
            base = n[: -len(suffix)]
            old_names += [f'{base}W_{c}.weight' for c in ('Q', 'K', 'V')]
        else:
            old_names.append(n)

    groups = optim_sd.get('param_groups') or []
    if (len(new_names) == len(old_names)          # nothing fused
            or len(groups) != 1
            or len(groups[0].get('params', [])) != len(old_names)):
        return optim_sd

    # Checkpointed state keys may be ints or strings depending on torch version.
    old_state = {int(k): v for k, v in optim_sd['state'].items()}
    old_index = {name: i for i, name in enumerate(old_names)}
    merge_keys = ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq')

    new_state = {}
    for new_i, n in enumerate(new_names):
        if n.endswith(suffix):
            base = n[: -len(suffix)]
            parts = [old_state.get(old_index[f'{base}W_{c}.weight'])
                     for c in ('Q', 'K', 'V')]
            if any(p is None for p in parts):
                continue                      # never stepped; leave state empty
            merged = dict(parts[0])           # carries 'step'
            for key in merge_keys:
                if all(key in p for p in parts):
                    merged[key] = torch.cat([p[key] for p in parts], dim=0)
            new_state[new_i] = merged
        else:
            st = old_state.get(old_index[n])
            if st is not None:
                new_state[new_i] = st

    out = dict(optim_sd)
    out['state'] = new_state
    out['param_groups'] = [dict(groups[0], params=list(range(len(new_names))))]
    return out


class MultiHeadCAtt(nn.Module):
    """Multi-head C-Attention (CAtt).

    Splits d_model into n_heads of d_k = d_model // n_heads,
    runs CAtt per head, concatenates, and applies output projection W_O.
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Q/K/V share one fused projection: one (d_model x 3*d_model) GEMM rather
        # than three (d_model x d_model) ones. Checkpoints written before the
        # fusion store W_Q/W_K/W_V separately and are migrated on load below.
        self.W_QKV = ComplexLinear(d_model, 3 * d_model, bias=False)
        self.W_O = ComplexLinear(d_model, d_model, bias=True)

        self.attn = CAtt(self.d_k)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Accept pre-fusion checkpoints by concatenating W_Q/W_K/W_V into W_QKV."""
        legacy = [f'{prefix}W_{n}.weight' for n in ('Q', 'K', 'V')]
        if all(k in state_dict for k in legacy):
            state_dict[f'{prefix}W_QKV.weight'] = torch.cat(
                [state_dict.pop(k) for k in legacy], dim=0,
            )
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x):
        """x: (batch, seq_len, d_model) complex -> same shape."""
        B, N, _ = x.shape

        # One projection, then split into Q/K/V and heads.
        qkv = self.W_QKV(x).view(B, N, 3, self.n_heads, self.d_k)
        Q, K, V = qkv.permute(2, 0, 3, 1, 4)   # each (B, heads, N, d_k)

        out = self.attn(Q, K, V)  # (B, heads, N, d_k)

        # Concat heads and project
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, self.d_model)
        return self.W_O(out)

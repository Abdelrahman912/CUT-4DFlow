"""Optional ``torch.compile`` helpers for the model's real-valued inner kernels.

The model is complex-valued end to end, which ``torch.compile`` cannot lower, so the
fusable kernels are written in real arithmetic and compiled individually. Each is gated
by an environment flag (default off):

  CMRX_COMPILE_NORM=1   ComplexLayerNorm2x2
  CMRX_COMPILE_ATTN=1   attention core + modReLU
"""
from __future__ import annotations

import os

import torch

_CACHE: dict = {}


def enabled(env_var: str) -> bool:
    return os.environ.get(env_var, '') not in ('', '0')


def maybe_compile(fn, env_var: str, probe=None):
    """Return ``fn``, or a cached ``torch.compile``'d version if ``env_var`` is set.

    Compilation is deferred to first use. When ``probe`` (a zero-arg callable returning
    sample args) is given, the compiled kernel is exercised forward *and* backward before
    being adopted; on any failure it falls back to eager permanently.
    """
    key = (fn.__module__, fn.__qualname__, env_var)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit

    if not enabled(env_var):
        _CACHE[key] = fn
        return fn

    compiled = torch.compile(fn, dynamic=True)

    if probe is not None:
        try:
            out = compiled(*probe())
            tensors = out if isinstance(out, (tuple, list)) else (out,)
            loss = sum(t.float().sum() for t in tensors if torch.is_tensor(t))
            loss.backward()
        except Exception as exc:          # noqa: BLE001
            print(f'[fused] {fn.__qualname__}: torch.compile unusable here '
                  f'({type(exc).__name__}); using eager.', flush=True)
            _CACHE[key] = fn
            return fn
        print(f'[fused] {fn.__qualname__}: compiled', flush=True)

    _CACHE[key] = compiled
    return compiled

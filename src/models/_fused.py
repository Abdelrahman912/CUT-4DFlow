"""Lazy ``torch.compile`` helpers for the cascade's fusable kernels.

Why individual kernels instead of compiling the model
-----------------------------------------------------
inductor cannot lower complex64: it reinterprets complex tensors as pairs of
reals via ``aten.view.dtype``, which requires a stride-1 last dim and fails on
the cascade's permuted positional-encoding tensors. The model is complex-valued
end to end, so ``torch.compile(model)`` is not available.

What works instead is compiling individual functions that are written in REAL
arithmetic only, with ``view_as_real``/``view_as_complex`` (or explicit re/im
pairs) at the boundary. The GEMMs inside stay as extern cuBLAS calls — inductor
fuses the pointwise and reduction plumbing around them, which is where the
non-GEMM time actually goes.

Each kernel has its own env flag so they can be A/B'd independently:

  CMRX_COMPILE_NORM=1   ComplexLayerNorm2x2   (measured 1.43x on H100)
  CMRX_COMPILE_ATTN=1   CAtt core + modReLU

``dynamic=True`` because the token axis varies across subjects (23 distinct
spatial shapes in the training set) while the reduction axes are fixed — one
compilation then covers every subject instead of recompiling per shape.
"""

from __future__ import annotations

import os

import torch

_CACHE: dict = {}


def enabled(env_var: str) -> bool:
    return os.environ.get(env_var, '') not in ('', '0')


def maybe_compile(fn, env_var: str, probe=None):
    """Return ``fn``, compiled once (and cached) if ``env_var`` is set.

    Compilation is deferred to first use so importing the module never pays for
    it, and so the flag can be set after import.

    ``probe`` is a zero-arg callable returning a tuple of small sample arguments.
    When given, the compiled kernel is exercised on them — forward AND backward —
    before being adopted. If anything fails, this falls back to the eager
    function permanently. The backward matters: AOT autograd compiles the
    backward graph lazily inside ``.backward()``, so a codegen failure there
    would otherwise escape any guard around the forward call and kill the run
    mid-training. Probing costs one tiny compile and makes the flag safe to
    enable on hardware where inductor cannot lower these kernels.
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
            # enable_grad: the first real call often happens inside @torch.no_grad()
            # (inference), where the backward probe would raise and silently cache
            # the eager fallback — exactly the path where the fusion is wanted.
            with torch.enable_grad():
                out = compiled(*probe())
                tensors = out if isinstance(out, (tuple, list)) else (out,)
                loss = sum(t.float().sum() for t in tensors if torch.is_tensor(t))
                loss.backward()           # exercise the backward codegen too
        except Exception as exc:          # noqa: BLE001 - any codegen failure
            print(f'[fused] {fn.__qualname__}: torch.compile unusable here '
                  f'({type(exc).__name__}: {str(exc).splitlines()[0][:100]}); '
                  f'using eager for this run.', flush=True)
            _CACHE[key] = fn
            return fn
        print(f'[fused] {fn.__qualname__}: compiled', flush=True)

    _CACHE[key] = compiled
    return compiled

"""Training losses for the CMRx 4D-flow SSDU pipeline.

Two losses live here:

- ``ssdu_loss`` — the standard self-supervised k-space residual on the held-out
  Lambda partition: normalized ``0.5*L2 + 0.5*L1`` (relative to ``||y_Lambda||``).
- ``ssdu_phase_loss`` — ``ssdu_loss`` PLUS a velocity-sensitive term
  ``lambda_v * L_v`` in the referenced (velocity-difference) basis. ``lambda_v=0``
  recovers plain ``ssdu_loss`` (the "lv0" vs "lv1" ablation).

Velocity-difference basis (still PURE SSDU — no supervision, no GT velocity):
re-express the SAME held-out acquired k-space ``y_Lambda`` in the referenced
4-point-encoding basis

    x  =  (x0,  x1,        x2,        x3)
       -> (x0,  x1 - x0,   x2 - x0,   x3 - x0)

whose last three channels ARE the velocity-encoding differences. Because the
forward operator ``A = M_Lambda . F . S`` is SHARED across the V velocity
encodings, differencing commutes with it:

    A_Lambda(x)_d - A_Lambda(x)_0  =  A_Lambda(x_d - x_0)
    =>  dy_hat_d = y_hat_d - y_hat_0            (prediction difference)
        dy_d     = y_{Lambda,d} - y_{Lambda,0} (target difference)

so the difference target ``dy_d`` is built ENTIRELY from the acquired Lambda
k-space already used by SSDU. No fully-sampled image and no GT velocity enter the
loss -> still PURE SSDU, just re-expressed in a velocity-sensitive basis. The
velocity-carrying difference is now the whole signal in those channels (not a
~6% perturbation), so it gets full gradient weight.

Loss:
    L = L_SSDU(per-enc)  +  lambda_v * L_v
    L_v = mean_{d=1..3} [ 0.5 * ||dy_hat_d - dy_d||_2 / ||dy_d||_2
                        + 0.5 * ||dy_hat_d - dy_d||_1 / ||dy_d||_1 ]
mirroring the normalized L2+L1 SSDU loss, on the difference channels. Both terms
are kept: the per-encoding term constrains magnitude / absolute reconstruction,
L_v constrains velocity — they are not redundant.

Mask-sharing caveat (handled): ``dy_d`` is only defined where encoding d AND the
reference encoding 0 are both sampled in Lambda. The CMRx kt-Gaussian mask is
broadcast across V (shape ``(1, T, 1, SPE, PE, 1)``), so every encoding shares
the per-frame Lambda pattern and the difference is valid wherever Lambda is
sampled. ``_intersection_mask`` enforces this regardless, so the term stays
correct even if a per-encoding mask is ever introduced.
"""
from __future__ import annotations

import torch


def ssdu_loss(k_pred: torch.Tensor, k_target: torch.Tensor) -> torch.Tensor:
    """0.5 L2 + 0.5 L1, both relative to ||k_target||."""
    pred = torch.view_as_real(k_pred)
    tgt = torch.view_as_real(k_target)
    n2 = torch.norm(tgt, p=2)
    n1 = torch.norm(tgt, p=1)
    eps = 1e-8
    l2 = torch.norm(tgt - pred, p=2) / (n2 + eps)
    l1 = torch.norm(tgt - pred, p=1) / (n1 + eps)
    return 0.5 * l2 + 0.5 * l1


def encoding_difference(k: torch.Tensor) -> torch.Tensor:
    """Referenced-basis transform along the V (velocity-encoding) axis.

    k : (B, V, C, T, SPE, PE) complex
    -> (B, V-1, C, T, SPE, PE) complex, where channel ``d-1`` = ``k[:, d] - k[:, 0]``
       for ``d = 1 .. V-1`` (the three velocity-encoding differences).
    """
    return k[:, 1:] - k[:, 0:1]


def _intersection_mask(mask_lambda: torch.Tensor) -> torch.Tensor:
    """Binary (B, V-1, C, T, SPE, PE) mask: points sampled in BOTH enc d and ref 0.

    ``mask_lambda`` : (B, V, C, T, SPE, PE) float/bool, OR V-broadcast (B, 1, ...).
    When V is broadcast (the CMRx case — a single shared Lambda pattern) this is
    just that shared pattern, and multiplying by it is a no-op on the already
    Lambda-masked differences. The guard only does real work if a per-encoding
    mask is ever supplied (then it restricts each difference to the intersection
    of the two encodings' sampled points).
    """
    base = mask_lambda.real if torch.is_complex(mask_lambda) else mask_lambda
    ref = base[:, 0:1] != 0
    enc = ref if base.shape[1] == 1 else (base[:, 1:] != 0)
    return (enc & ref).to(torch.float32)


def velocity_difference_loss(
    k_pred: torch.Tensor,
    k_target: torch.Tensor,
    mask_lambda: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """L_v only — normalized 0.5*L2 + 0.5*L1 on the V-difference channels.

    k_pred, k_target : (B, V, C, T, SPE, PE) complex (A_Lambda(x) and y_Lambda).
    Returns a real scalar.
    """
    dpred = encoding_difference(k_pred)
    dtgt = encoding_difference(k_target)
    if mask_lambda is not None:
        inter = _intersection_mask(mask_lambda).to(dpred.device)
        dpred = dpred * inter
        dtgt = dtgt * inter

    # Per-difference-channel norms, computed as one batched reduction over the
    # V-1 axis instead of a python loop over d. Each channel is still normalised
    # by its OWN target norm, so this is the same value the loop produced.
    p = torch.view_as_real(dpred)
    t = torch.view_as_real(dtgt)
    red = tuple(i for i in range(p.dim()) if i != 1)   # reduce all but the d axis
    diff = t - p
    n2 = torch.linalg.vector_norm(t, ord=2, dim=red)
    n1 = torch.linalg.vector_norm(t, ord=1, dim=red)
    l2 = torch.linalg.vector_norm(diff, ord=2, dim=red) / (n2 + eps)
    l1 = torch.linalg.vector_norm(diff, ord=1, dim=red) / (n1 + eps)
    return (0.5 * l2 + 0.5 * l1).mean()


def ssdu_phase_loss(
    k_pred: torch.Tensor,
    k_target: torch.Tensor,
    mask_lambda: torch.Tensor | None = None,
    lambda_v: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Pure-SSDU loss = per-encoding term + lambda_v * velocity-difference term.

    k_pred, k_target : (B, V, C, T, SPE, PE) complex  — A_Lambda(x) and y_Lambda.
    mask_lambda      : (B, V|1, C, T, SPE, PE) — used only by the difference guard.
    lambda_v         : weight on the velocity-difference term (0 = plain ssdu_loss).
    """
    base = ssdu_loss(k_pred, k_target)
    if lambda_v == 0.0:
        return base   # lv0: skip the velocity-difference term entirely (no wasted compute)
    lv = velocity_difference_loss(k_pred, k_target, mask_lambda=mask_lambda, eps=eps)
    return base + lambda_v * lv

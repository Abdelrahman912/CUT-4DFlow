"""Self-supervised (SSDU) k-space losses.

- ``ssdu_loss``: normalized ``0.5*L2 + 0.5*L1`` k-space residual on the held-out Lambda set.
- ``ssdu_phase_loss``: adds ``lambda_v`` times a velocity-difference term evaluated in the
  referenced basis (each velocity encoding minus the reference). Since the forward operator is
  shared across encodings, this difference is built entirely from the acquired Lambda k-space,
  so the loss remains fully self-supervised. ``lambda_v = 0`` recovers ``ssdu_loss``.
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
    """Referenced basis along V: ``k[:, d] - k[:, 0]`` for d = 1..V-1.  (B,V,C,T,SPE,PE) complex."""
    return k[:, 1:] - k[:, 0:1]


def _intersection_mask(mask_lambda: torch.Tensor) -> torch.Tensor:
    """Binary mask of points sampled in both encoding d and the reference (no-op when V is broadcast)."""
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
    """Normalized 0.5*L2 + 0.5*L1 on the V-difference channels (each normalized by its own norm)."""
    dpred = encoding_difference(k_pred)
    dtgt = encoding_difference(k_target)
    if mask_lambda is not None:
        inter = _intersection_mask(mask_lambda).to(dpred.device)
        dpred = dpred * inter
        dtgt = dtgt * inter

    p = torch.view_as_real(dpred)
    t = torch.view_as_real(dtgt)
    red = tuple(i for i in range(p.dim()) if i != 1)   # reduce all but the difference axis
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
    """SSDU loss + lambda_v * velocity-difference term. lambda_v=0 -> plain ssdu_loss."""
    base = ssdu_loss(k_pred, k_target)
    if lambda_v == 0.0:
        return base
    lv = velocity_difference_loss(k_pred, k_target, mask_lambda=mask_lambda, eps=eps)
    return base + lambda_v * lv

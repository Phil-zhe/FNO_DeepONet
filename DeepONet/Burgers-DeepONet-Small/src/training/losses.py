from __future__ import annotations

import torch
import torch.nn.functional as F


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def relative_l2_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    num = torch.linalg.vector_norm(pred - target, dim=-1)
    den = torch.linalg.vector_norm(target, dim=-1).clamp_min(eps)
    return (num / den).mean()

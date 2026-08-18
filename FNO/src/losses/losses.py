from __future__ import annotations

import torch
import torch.nn.functional as F


def relative_l2_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    numerator = torch.linalg.norm((pred - target).reshape(pred.shape[0], -1), dim=1)
    denominator = torch.linalg.norm(target.reshape(target.shape[0], -1), dim=1) + eps
    return (numerator / denominator).mean()


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)

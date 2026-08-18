from __future__ import annotations

import torch
import torch.nn.functional as F


def relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    numerator = torch.linalg.norm((pred - target).reshape(pred.shape[0], -1), dim=1)
    denominator = torch.linalg.norm(target.reshape(target.shape[0], -1), dim=1) + eps
    return (numerator / denominator).mean()


def relative_l2_per_sample(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    numerator = torch.linalg.norm((pred - target).reshape(pred.shape[0], -1), dim=1)
    denominator = torch.linalg.norm(target.reshape(target.shape[0], -1), dim=1) + eps
    return numerator / denominator


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def max_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.abs(pred - target))


def spectral_error(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred_fft = torch.abs(torch.fft.rfft(pred, dim=-1))
    target_fft = torch.abs(torch.fft.rfft(target, dim=-1))
    numerator = torch.linalg.norm((pred_fft - target_fft).reshape(pred.shape[0], -1), dim=1)
    denominator = torch.linalg.norm(target_fft.reshape(target.shape[0], -1), dim=1) + eps
    return (numerator / denominator).mean()

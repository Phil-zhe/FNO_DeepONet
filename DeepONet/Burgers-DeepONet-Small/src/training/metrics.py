from __future__ import annotations

import numpy as np
import torch


def compute_mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def compute_mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def compute_relative_l2(pred: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    pred2 = pred.reshape(pred.shape[0], -1)
    target2 = target.reshape(target.shape[0], -1)
    rel = np.linalg.norm(pred2 - target2, axis=1) / np.maximum(np.linalg.norm(target2, axis=1), eps)
    return float(np.mean(rel))


@torch.no_grad()
def evaluate_full_grid(model, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    preds = []
    targets = []
    for batch in loader:
        branch = batch["branch"].to(device)
        trunk = batch["trunk"].to(device)
        target = batch["target"].to(device)
        pred = model(branch, trunk)
        if hasattr(loader.dataset, "denormalize_uT"):
            pred = loader.dataset.denormalize_uT(pred)
            target = loader.dataset.denormalize_uT(target)
        preds.append(pred.cpu().numpy())
        targets.append(target.cpu().numpy())
    pred_np = np.concatenate(preds, axis=0)
    target_np = np.concatenate(targets, axis=0)
    return {
        "mse": compute_mse(pred_np, target_np),
        "mae": compute_mae(pred_np, target_np),
        "relative_l2": compute_relative_l2(pred_np, target_np),
    }

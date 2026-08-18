from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import BurgersDeepONetDataset
from src.models.deeponet import DeepONet
from src.training.metrics import compute_mae, compute_mse, compute_relative_l2
from src.utils.config import load_config, resolve_path
from src.utils.io import ensure_parent
from src.utils.plotting import plot_error_histogram, plot_prediction_examples
from src.utils.seed import set_seed


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(ROOT / args.config)
    set_seed(int(cfg["seed"]))
    device = torch.device(
        "cuda" if bool(cfg["device"]["use_gpu"]) and torch.cuda.is_available() else "cpu"
    )
    ckpt_path = ROOT / cfg["paths"]["best_model"]
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    dataset_path = resolve_path(cfg, Path(cfg["paths"]["data_dir"]) / cfg["paths"]["dataset_name"])
    test_ds = BurgersDeepONetDataset(
        dataset_path,
        "test",
        n_sensors=cfg["dataset"]["n_sensors"],
        n_query=cfg["dataset"]["n_query"],
        normalize=cfg["dataset"]["normalize"],
        stats=checkpoint["stats"],
        seed=cfg["seed"] + 2,
    )
    loader = DataLoader(test_ds, batch_size=cfg["evaluation"]["batch_size"], shuffle=False)
    model = DeepONet(n_sensors=cfg["dataset"]["n_sensors"], **cfg["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    preds, targets, sample_indices = [], [], []
    for batch in loader:
        branch = batch["branch"].to(device)
        trunk = batch["trunk"].to(device)
        target = batch["target"].to(device)
        pred = model(branch, trunk)
        pred = test_ds.denormalize_uT(pred)
        target = test_ds.denormalize_uT(target)
        preds.append(pred.cpu().numpy())
        targets.append(target.cpu().numpy())
        sample_indices.append(batch["sample_idx"].numpy())
    pred_np = np.concatenate(preds, axis=0)
    true_np = np.concatenate(targets, axis=0)
    sample_idx_np = np.concatenate(sample_indices, axis=0)
    rel_per_sample = np.linalg.norm(pred_np - true_np, axis=1) / np.maximum(
        np.linalg.norm(true_np, axis=1), 1e-12
    )
    metrics = {
        "mse": compute_mse(pred_np, true_np),
        "mae": compute_mae(pred_np, true_np),
        "relative_l2": compute_relative_l2(pred_np, true_np),
    }
    print(
        f"test MSE={metrics['mse']:.6e}, MAE={metrics['mae']:.6e}, "
        f"relative L2={metrics['relative_l2']:.6e}"
    )
    out_path = ROOT / cfg["paths"]["predictions"]
    ensure_parent(out_path)
    np.savez_compressed(
        out_path,
        x=test_ds.x,
        pred=pred_np,
        true=true_np,
        sample_idx=sample_idx_np,
        relative_l2_per_sample=rel_per_sample,
        mse=metrics["mse"],
        mae=metrics["mae"],
        relative_l2=metrics["relative_l2"],
    )
    fig_dir = ROOT / cfg["paths"]["figure_dir"]
    plot_prediction_examples(
        test_ds.x,
        true_np,
        pred_np,
        fig_dir / "prediction_examples.png",
        max_examples=cfg["evaluation"]["num_plot_examples"],
    )
    plot_error_histogram(rel_per_sample, fig_dir / "error_histogram.png")


if __name__ == "__main__":
    main()

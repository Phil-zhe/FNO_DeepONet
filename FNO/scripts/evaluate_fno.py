from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.burgers_dataset import BurgersDataset
from src.data.normalizer import UnitGaussianNormalizer
from src.metrics.metrics import max_error, mse, relative_l2, relative_l2_per_sample, spectral_error
from src.models.fno1d import FNO1d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "fno1d.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        file_config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = ROOT / "results" / "checkpoints" / "best_fno1d.pt"
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", file_config)
    seed = int(config["train"]["seed"])

    test_ds = BurgersDataset(
        ROOT / config["data_path"],
        "test",
        config["train_size"],
        config["val_size"],
        config["test_size"],
        seed,
    )
    test_loader = DataLoader(test_ds, batch_size=int(config["train"]["batch_size"]), shuffle=False, num_workers=0)

    input_normalizer = UnitGaussianNormalizer()
    output_normalizer = UnitGaussianNormalizer()
    input_normalizer.load_state_dict(checkpoint["input_normalizer"])
    output_normalizer.load_state_dict(checkpoint["output_normalizer"])
    input_normalizer.to(device)
    output_normalizer.to(device)

    model = FNO1d(**config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    a_all, u_true_all, u_pred_all, rel_all = [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            a = batch["a"].to(device)
            u = batch["u"].to(device)
            grid = batch["grid"].to(device)
            pred_norm = model(input_normalizer.encode(a), grid)
            pred = output_normalizer.decode(pred_norm)
            a_all.append(a.cpu())
            u_true_all.append(u.cpu())
            u_pred_all.append(pred.cpu())
            rel_all.append(relative_l2_per_sample(pred, u).cpu())

    a_test = torch.cat(a_all, dim=0)
    u_true = torch.cat(u_true_all, dim=0)
    u_pred = torch.cat(u_pred_all, dim=0)
    rel_l2_samples = torch.cat(rel_all, dim=0)

    metrics = {
        "test_relative_l2": float(relative_l2(u_pred, u_true).item()),
        "test_mse": float(mse(u_pred, u_true).item()),
        "test_max_error": float(max_error(u_pred, u_true).item()),
        "test_spectral_error": float(spectral_error(u_pred, u_true).item()),
    }

    table_dir = ROOT / "results" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    with (table_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    np.savez_compressed(
        table_dir / "test_predictions.npz",
        a_test=a_test.numpy().astype(np.float32),
        u_true=u_true.numpy().astype(np.float32),
        u_pred=u_pred.numpy().astype(np.float32),
        x=test_ds.x.astype(np.float32),
        rel_l2_per_sample=rel_l2_samples.numpy().astype(np.float32),
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

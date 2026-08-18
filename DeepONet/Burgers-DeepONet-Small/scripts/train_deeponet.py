from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import BurgersDeepONetDataset
from src.models.deeponet import DeepONet
from src.training.trainer import Trainer
from src.utils.config import load_config, resolve_path
from src.utils.plotting import plot_loss_curve
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(ROOT / args.config)
    set_seed(int(cfg["seed"]))

    dataset_path = resolve_path(cfg, Path(cfg["paths"]["data_dir"]) / cfg["paths"]["dataset_name"])
    train_ds = BurgersDeepONetDataset(
        dataset_path,
        "train",
        n_sensors=cfg["dataset"]["n_sensors"],
        n_query=cfg["dataset"]["n_query"],
        normalize=cfg["dataset"]["normalize"],
        seed=cfg["seed"],
    )
    val_ds = BurgersDeepONetDataset(
        dataset_path,
        "val",
        n_sensors=cfg["dataset"]["n_sensors"],
        n_query=cfg["dataset"]["n_query"],
        normalize=cfg["dataset"]["normalize"],
        stats=train_ds.stats,
        seed=cfg["seed"] + 1,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["device"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["evaluation"]["batch_size"],
        shuffle=False,
        num_workers=cfg["device"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device(
        "cuda" if bool(cfg["device"]["use_gpu"]) and torch.cuda.is_available() else "cpu"
    )
    model = DeepONet(n_sensors=cfg["dataset"]["n_sensors"], **cfg["model"])
    print(f"Using device: {device}")
    trainer = Trainer(model, train_loader, val_loader, cfg, device)
    trainer.fit()
    plot_loss_curve(
        ROOT / cfg["paths"]["loss_history"],
        ROOT / cfg["paths"]["figure_dir"] / "loss_curve.png",
    )


if __name__ == "__main__":
    main()

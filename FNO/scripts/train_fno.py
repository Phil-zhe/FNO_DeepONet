from __future__ import annotations

import argparse
import csv
import random
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
from src.losses.losses import relative_l2_loss
from src.metrics.metrics import relative_l2
from src.models.fno1d import FNO1d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "fno1d.yaml")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_loader(dataset: BurgersDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)


def fit_normalizers(train_ds: BurgersDataset) -> tuple[UnitGaussianNormalizer, UnitGaussianNormalizer]:
    a_train = torch.from_numpy(train_ds.a[train_ds.indices])
    u_train = torch.from_numpy(train_ds.u[train_ds.indices])
    input_normalizer = UnitGaussianNormalizer()
    output_normalizer = UnitGaussianNormalizer()
    input_normalizer.fit(a_train)
    output_normalizer.fit(u_train)
    return input_normalizer, output_normalizer


@torch.no_grad()
def evaluate_rel_l2(model, loader, input_normalizer, output_normalizer, device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        a = batch["a"].to(device)
        u = batch["u"].to(device)
        grid = batch["grid"].to(device)
        pred_norm = model(input_normalizer.encode(a), grid)
        pred = output_normalizer.decode(pred_norm)
        batch_rel = relative_l2(pred, u)
        total += batch_rel.item() * a.shape[0]
        count += a.shape[0]
    return total / count


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_cfg = config["train"]
    seed = int(train_cfg["seed"])
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(train_cfg["amp"]) and device.type == "cuda"

    data_path = ROOT / config["data_path"]
    train_ds = BurgersDataset(data_path, "train", config["train_size"], config["val_size"], config["test_size"], seed)
    val_ds = BurgersDataset(data_path, "val", config["train_size"], config["val_size"], config["test_size"], seed)
    batch_size = int(train_cfg["batch_size"])
    train_loader = make_loader(train_ds, batch_size, shuffle=True)
    val_loader = make_loader(val_ds, batch_size, shuffle=False)

    input_normalizer, output_normalizer = fit_normalizers(train_ds)
    input_normalizer.to(device)
    output_normalizer.to(device)

    model = FNO1d(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(train_cfg["epochs"]))
    # GradScaler cannot unscale complex-valued FNO weights in current PyTorch builds.
    # Autocast remains enabled for real-valued layers while spectral FFTs run in fp32.
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    checkpoint_dir = ROOT / "results" / "checkpoints"
    table_dir = ROOT / "results" / "tables"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    log_path = table_dir / "train_log.csv"
    best_val = float("inf")
    rows = []

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        loss_total = 0.0
        n_seen = 0
        for batch in train_loader:
            a = batch["a"].to(device)
            u = batch["u"].to(device)
            grid = batch["grid"].to(device)
            a_norm = input_normalizer.encode(a)
            u_norm = output_normalizer.encode(u)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_norm = model(a_norm, grid)
                loss = relative_l2_loss(pred_norm, u_norm)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_total += loss.item() * a.shape[0]
            n_seen += a.shape[0]

        scheduler.step()
        train_loss = loss_total / n_seen
        train_rel = evaluate_rel_l2(model, train_loader, input_normalizer, output_normalizer, device)
        val_rel = evaluate_rel_l2(model, val_loader, input_normalizer, output_normalizer, device)
        lr = optimizer.param_groups[0]["lr"]

        rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_rel_l2": train_rel,
            "val_rel_l2": val_rel,
            "lr": lr,
        })
        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_rel_l2", "val_rel_l2", "lr"])
            writer.writeheader()
            writer.writerows(rows)

        print(
            f"epoch {epoch:04d} | train_loss {train_loss:.6f} | "
            f"train_rel_l2 {train_rel:.6f} | val_rel_l2 {val_rel:.6f} | lr {lr:.3e}"
        )

        if val_rel < best_val:
            best_val = val_rel
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "input_normalizer": input_normalizer.state_dict(),
                    "output_normalizer": output_normalizer.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "best_val_loss": best_val,
                },
                checkpoint_dir / "best_fno1d.pt",
            )

    print(f"Best validation relative L2: {best_val:.6f}")


if __name__ == "__main__":
    main()

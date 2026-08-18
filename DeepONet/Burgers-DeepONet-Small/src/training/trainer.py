from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import torch
from torch.amp import GradScaler, autocast

from src.training.losses import mse_loss, relative_l2_loss
from src.training.metrics import evaluate_full_grid
from src.utils.io import ensure_parent


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        config: dict,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        tr = config["training"]
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(tr["learning_rate"]),
            weight_decay=float(tr["weight_decay"]),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=int(tr["epochs"])
        )
        self.use_amp = bool(config["device"]["use_amp"]) and device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.use_amp)
        self.best_path = Path(config["_project_root"]) / config["paths"]["best_model"]
        self.history_path = Path(config["_project_root"]) / config["paths"]["loss_history"]
        ensure_parent(self.best_path)
        ensure_parent(self.history_path)
        self.history: list[dict[str, float]] = []

    def train_one_epoch(self) -> float:
        self.model.train()
        losses = []
        for batch in self.train_loader:
            branch = batch["branch"].to(self.device, non_blocking=True)
            trunk = batch["trunk"].to(self.device, non_blocking=True)
            target = batch["target"].to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=self.use_amp):
                pred = self.model(branch, trunk)
                loss = mse_loss(pred, target)
            self.scaler.scale(loss).backward()
            grad_clip = float(self.config["training"].get("grad_clip_norm", 0.0))
            if grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            losses.append(float(loss.detach().cpu()))
        return float(sum(losses) / max(len(losses), 1))

    @torch.no_grad()
    def validate(self) -> tuple[float, float]:
        self.model.eval()
        losses = []
        rels = []
        for batch in self.val_loader:
            branch = batch["branch"].to(self.device)
            trunk = batch["trunk"].to(self.device)
            target = batch["target"].to(self.device)
            pred = self.model(branch, trunk)
            losses.append(float(mse_loss(pred, target).cpu()))
            pred_den = self.val_loader.dataset.denormalize_uT(pred)
            target_den = self.val_loader.dataset.denormalize_uT(target)
            rels.append(float(relative_l2_loss(pred_den, target_den).cpu()))
        return float(sum(losses) / max(len(losses), 1)), float(sum(rels) / max(len(rels), 1))

    def _save_checkpoint(self, epoch: int, val_loss: float, val_rel_l2: float) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "val_relative_l2": val_rel_l2,
                "config": self.config,
                "stats": self.train_loader.dataset.stats,
                "sensor_idx": self.train_loader.dataset.sensor_idx,
            },
            self.best_path,
        )

    def _write_history(self) -> None:
        pd.DataFrame(self.history).to_csv(self.history_path, index=False)

    def fit(self) -> dict[str, float]:
        epochs = int(self.config["training"]["epochs"])
        patience = int(self.config["training"]["early_stopping_patience"])
        best_val = float("inf")
        stale = 0
        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch()
            val_loss, val_rel_l2 = self.validate()
            self.scheduler.step()
            lr = float(self.scheduler.get_last_lr()[0])
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_relative_l2": val_rel_l2,
                "lr": lr,
            }
            self.history.append(row)
            if val_loss < best_val:
                best_val = val_loss
                stale = 0
                self._save_checkpoint(epoch, val_loss, val_rel_l2)
            else:
                stale += 1
            print(
                f"epoch {epoch:04d} | train {train_loss:.6e} | "
                f"val {val_loss:.6e} | relL2 {val_rel_l2:.6e} | lr {lr:.3e}"
            )
            self._write_history()
            if stale >= patience:
                print(f"Early stopping at epoch {epoch}; best val loss {best_val:.6e}.")
                break

        metrics = evaluate_full_grid(self.model, self.val_loader, self.device)
        return metrics

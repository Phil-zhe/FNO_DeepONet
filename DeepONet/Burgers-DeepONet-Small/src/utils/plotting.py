from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.io import ensure_parent


def plot_loss_curve(history_csv: str | Path, out_path: str | Path) -> None:
    df = pd.read_csv(history_csv)
    ensure_parent(out_path)
    plt.figure(figsize=(7, 4))
    plt.semilogy(df["epoch"], df["train_loss"], label="train")
    plt.semilogy(df["epoch"], df["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_prediction_examples(x, true, pred, out_path: str | Path, max_examples: int = 5) -> None:
    ensure_parent(out_path)
    n = min(max_examples, true.shape[0])
    fig, axes = plt.subplots(n, 2, figsize=(10, 2.4 * n), squeeze=False)
    for i in range(n):
        axes[i, 0].plot(x, true[i], label="true", lw=2)
        axes[i, 0].plot(x, pred[i], "--", label="pred", lw=2)
        axes[i, 0].set_ylabel(f"sample {i}")
        axes[i, 0].legend()
        axes[i, 1].plot(x, pred[i] - true[i], color="tab:red")
        axes[i, 1].set_ylabel("error")
    axes[-1, 0].set_xlabel("x")
    axes[-1, 1].set_xlabel("x")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_error_histogram(rel_errors: np.ndarray, out_path: str | Path) -> None:
    ensure_parent(out_path)
    plt.figure(figsize=(6, 4))
    plt.hist(rel_errors, bins=30, edgecolor="black")
    plt.xlabel("relative L2 error")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

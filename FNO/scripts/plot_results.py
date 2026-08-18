from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "fno1d.yaml")
    return parser.parse_args()


def plot_loss_curve(fig_dir: Path) -> None:
    log = np.genfromtxt(ROOT / "results" / "tables" / "train_log.csv", delimiter=",", names=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(log["epoch"], log["train_rel_l2"], label="train relative L2")
    ax.plot(log["epoch"], log["val_rel_l2"], label="val relative L2")
    ax.set_xlabel("epoch")
    ax.set_ylabel("relative L2")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "loss_curve.png", dpi=200)
    plt.close(fig)


def plot_predictions(fig_dir: Path) -> None:
    data = np.load(ROOT / "results" / "tables" / "test_predictions.npz")
    x = data["x"]
    a = data["a_test"]
    u_true = data["u_true"]
    u_pred = data["u_pred"]
    n_plot = min(6, len(a))
    fig, axes = plt.subplots(n_plot, 2, figsize=(10, 2.3 * n_plot), sharex=True)
    if n_plot == 1:
        axes = axes[None, :]
    for i in range(n_plot):
        axes[i, 0].plot(x, a[i], label="u0")
        axes[i, 0].plot(x, u_true[i], label="u_true")
        axes[i, 0].plot(x, u_pred[i], "--", label="u_pred")
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].set_ylabel(f"sample {i}")
        axes[i, 1].plot(x, u_pred[i] - u_true[i], color="tab:red", label="error")
        axes[i, 1].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 0].legend()
            axes[i, 1].legend()
    axes[-1, 0].set_xlabel("x")
    axes[-1, 1].set_xlabel("x")
    fig.tight_layout()
    fig.savefig(fig_dir / "test_predictions.png", dpi=200)
    plt.close(fig)


def plot_histogram(fig_dir: Path) -> None:
    data = np.load(ROOT / "results" / "tables" / "test_predictions.npz")
    rel = data["rel_l2_per_sample"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(rel, bins=20, edgecolor="black", alpha=0.8)
    ax.set_xlabel("relative L2")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "test_rel_l2_hist.png", dpi=200)
    plt.close(fig)


def plot_spectrum(fig_dir: Path) -> None:
    data = np.load(ROOT / "results" / "tables" / "test_predictions.npz")
    u_true = data["u_true"]
    u_pred = data["u_pred"]
    true_spec = np.abs(np.fft.rfft(u_true, axis=-1)).mean(axis=0)
    pred_spec = np.abs(np.fft.rfft(u_pred, axis=-1)).mean(axis=0)
    modes = np.arange(true_spec.shape[0])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(modes, true_spec + 1e-12, label="|FFT(u_true)|")
    ax.semilogy(modes, pred_spec + 1e-12, "--", label="|FFT(u_pred)|")
    ax.set_xlabel("Fourier mode")
    ax.set_ylabel("mean amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "spectrum_comparison.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        yaml.safe_load(f)
    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_loss_curve(fig_dir)
    plot_predictions(fig_dir)
    plot_histogram(fig_dir)
    plot_spectrum(fig_dir)
    print(f"Saved figures to {fig_dir}")


if __name__ == "__main__":
    main()

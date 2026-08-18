from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.solvers.burgers_spectral import BurgersConfig, generate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "data.yaml")
    return parser.parse_args()


def plot_samples(data: dict[str, np.ndarray | float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = data["x"]
    a = data["a"]
    u = data["u"]
    fig, axes = plt.subplots(5, 1, figsize=(8, 10), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(x, a[i], label="u0", linewidth=1.5)
        ax.plot(x, u[i], label="uT", linewidth=1.5)
        ax.set_ylabel(f"sample {i}")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
    axes[-1].set_xlabel("x")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    config = BurgersConfig(
        nx=int(cfg["nx"]),
        n_samples=int(cfg["n_samples"]),
        nu=float(cfg["nu"]),
        T=float(cfg["T"]),
        dt=float(cfg["dt"]),
        k_max=int(cfg["k_max"]),
        seed=int(cfg["seed"]),
        dealias=bool(cfg["dealias"]),
    )
    data = generate_dataset(config)

    output_npz = ROOT / cfg["output_npz"]
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **data)
    plot_samples(data, ROOT / "results" / "figures" / "check_generated_data.png")
    print(f"Saved dataset: {output_npz}")


if __name__ == "__main__":
    main()

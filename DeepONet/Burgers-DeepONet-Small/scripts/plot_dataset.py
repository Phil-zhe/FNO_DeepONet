from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config, resolve_path
from src.utils.io import ensure_parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(ROOT / args.config)
    data_path = resolve_path(cfg, Path(cfg["paths"]["data_dir"]) / cfg["paths"]["dataset_name"])
    data = np.load(data_path, allow_pickle=False)
    x, t, u, u0, uT = data["x"], data["t"], data["u"], data["u0"], data["uT"]
    fig_dir = ROOT / cfg["paths"]["figure_dir"]

    rng = np.random.default_rng(cfg["seed"])
    ids = rng.choice(u0.shape[0], size=min(5, u0.shape[0]), replace=False)
    ensure_parent(fig_dir / "initial_conditions.png")
    plt.figure(figsize=(7, 4))
    for idx in ids:
        plt.plot(x, u0[idx], label=f"u0 #{idx}")
    plt.xlabel("x")
    plt.ylabel("u0")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "initial_conditions.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    for idx in ids:
        plt.plot(x, uT[idx], label=f"uT #{idx}")
    plt.xlabel("x")
    plt.ylabel("u(x,T)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "terminal_solutions.png", dpi=180)
    plt.close()

    sample = int(ids[0])
    plt.figure(figsize=(7, 4))
    plt.imshow(
        u[sample],
        aspect="auto",
        origin="lower",
        extent=[float(x.min()), float(x.max()), float(t.min()), float(t.max())],
        cmap="viridis",
    )
    plt.colorbar(label="u(x,t)")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.tight_layout()
    plt.savefig(fig_dir / "dataset_xt_example.png", dpi=180)
    plt.close()
    print(f"Saved dataset figures to {fig_dir}")


if __name__ == "__main__":
    main()

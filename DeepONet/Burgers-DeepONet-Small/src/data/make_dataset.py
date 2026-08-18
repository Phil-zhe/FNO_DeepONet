from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.data.burgers_solver import solve_burgers_1d
from src.data.initial_conditions import generate_random_fourier_batch
from src.utils.config import resolve_path
from src.utils.io import ensure_parent


def generate_dataset(config: dict) -> Path:
    data_cfg = config["data"]
    path_cfg = config["paths"]
    out_path = resolve_path(config, Path(path_cfg["data_dir"]) / path_cfg["dataset_name"])
    ensure_parent(out_path)

    nx = int(data_cfg["nx"])
    nt = int(data_cfg["nt"])
    total = int(data_cfg["total_samples"])
    n_train = int(data_cfg["n_train"])
    n_val = int(data_cfg["n_val"])
    n_test = int(data_cfg["n_test"])
    if n_train + n_val + n_test != total:
        raise ValueError("n_train + n_val + n_test must equal total_samples.")

    x = np.linspace(0.0, 1.0, nx, endpoint=False, dtype=np.float64)
    t = np.linspace(0.0, float(data_cfg["T"]), nt, dtype=np.float64)
    u0_all = generate_random_fourier_batch(
        x,
        total,
        num_modes=int(data_cfg["num_modes"]),
        seed=int(config["seed"]),
        max_abs=float(data_cfg["ic_max_abs"]),
    )
    u = np.empty((total, nt, nx), dtype=np.float32)
    for i in tqdm(range(total), desc="Solving Burgers samples"):
        u[i] = solve_burgers_1d(
            u0_all[i],
            x,
            t,
            nu=float(data_cfg["nu"]),
            dealias=bool(data_cfg["dealias"]),
        )

    idx = np.arange(total)
    rng = np.random.default_rng(int(config["seed"]))
    rng.shuffle(idx)
    split = {
        "train": idx[:n_train].astype(np.int64),
        "val": idx[n_train : n_train + n_val].astype(np.int64),
        "test": idx[n_train + n_val :].astype(np.int64),
    }

    tmp_path = out_path.with_name(out_path.stem + ".tmp.npz")
    np.savez(
        tmp_path,
        x=x.astype(np.float32),
        t=t.astype(np.float32),
        u=u,
        u0=u[:, 0, :],
        uT=u[:, -1, :],
        split_train=split["train"],
        split_val=split["val"],
        split_test=split["test"],
        config=json.dumps(config, indent=2),
    )
    if out_path.exists():
        out_path.unlink()
    tmp_path.replace(out_path)
    print(f"Saved dataset: {out_path}")
    print(f"u shape={u.shape}, u0 shape={u[:, 0, :].shape}, uT shape={u[:, -1, :].shape}")
    print(
        "u stats: "
        f"min={u.min():.6f}, max={u.max():.6f}, mean={u.mean():.6f}, std={u.std():.6f}"
    )
    return out_path

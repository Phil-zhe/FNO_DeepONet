from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def split_indices(
    n_total: int,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
) -> dict[str, np.ndarray]:
    required = train_size + val_size + test_size
    if required > n_total:
        raise ValueError(f"Requested {required} samples, but dataset contains {n_total}.")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)
    train_end = train_size
    val_end = train_size + val_size
    return {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:val_end + test_size],
    }


class BurgersDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        split: str,
        train_size: int,
        val_size: int,
        test_size: int,
        seed: int = 2026,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")

        data = np.load(Path(data_path))
        self.a = data["a"].astype(np.float32)
        self.u = data["u"].astype(np.float32)
        self.x = data["x"].astype(np.float32)
        indices = split_indices(len(self.a), train_size, val_size, test_size, seed)
        self.indices = indices[split]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        i = self.indices[idx]
        return {
            "a": torch.from_numpy(self.a[i]),
            "u": torch.from_numpy(self.u[i]),
            "grid": torch.from_numpy(self.x),
        }

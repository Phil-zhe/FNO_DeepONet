from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class BurgersDeepONetDataset(Dataset):
    def __init__(
        self,
        npz_path: str | Path,
        split: str,
        n_sensors: int = 128,
        n_query: int = 128,
        normalize: bool = True,
        stats: dict[str, float] | None = None,
        seed: int = 42,
    ) -> None:
        data = np.load(npz_path, allow_pickle=False)
        self.x = data["x"].astype(np.float32)
        self.u0_all = data["u0"].astype(np.float32)
        self.uT_all = data["uT"].astype(np.float32)
        self.indices = data[f"split_{split}"].astype(np.int64)
        self.split = split
        self.n_query = int(n_query)
        self.normalize = bool(normalize)
        self.rng = np.random.default_rng(seed)

        nx = self.x.size
        self.sensor_idx = np.linspace(0, nx - 1, int(n_sensors), dtype=np.int64)
        train_idx = data["split_train"].astype(np.int64)
        if stats is None:
            stats = {
                "u0_mean": float(self.u0_all[train_idx].mean()),
                "u0_std": float(self.u0_all[train_idx].std() + 1e-8),
                "uT_mean": float(self.uT_all[train_idx].mean()),
                "uT_std": float(self.uT_all[train_idx].std() + 1e-8),
            }
        self.stats = stats

    def __len__(self) -> int:
        return int(self.indices.size)

    def _norm_u0(self, arr: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return arr
        return (arr - self.stats["u0_mean"]) / self.stats["u0_std"]

    def _norm_uT(self, arr: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return arr
        return (arr - self.stats["uT_mean"]) / self.stats["uT_std"]

    def denormalize_uT(self, arr: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return arr
        return arr * self.stats["uT_std"] + self.stats["uT_mean"]

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        idx = self.indices[item]
        u0 = self.u0_all[idx]
        uT = self.uT_all[idx]
        if self.split == "train":
            query_idx = self.rng.choice(self.x.size, size=self.n_query, replace=False)
            query_idx.sort()
        else:
            query_idx = np.arange(self.x.size, dtype=np.int64)

        branch = self._norm_u0(u0[self.sensor_idx])
        target = self._norm_uT(uT[query_idx])
        trunk = self.x[query_idx, None]
        return {
            "branch": torch.from_numpy(branch.astype(np.float32)),
            "trunk": torch.from_numpy(trunk.astype(np.float32)),
            "target": torch.from_numpy(target.astype(np.float32)),
            "query_idx": torch.from_numpy(query_idx.astype(np.int64)),
            "sample_idx": torch.tensor(int(idx), dtype=torch.long),
        }

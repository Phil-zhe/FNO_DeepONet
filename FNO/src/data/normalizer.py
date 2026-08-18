from __future__ import annotations

import torch


class UnitGaussianNormalizer:
    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

    def fit(self, x: torch.Tensor) -> None:
        x = x.detach().float()
        self.mean = x.mean(dim=0, keepdim=True)
        self.std = x.std(dim=0, keepdim=True) + self.eps

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self._check_fitted()
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        self._check_fitted()
        return x * self.std.to(x.device) + self.mean.to(x.device)

    def state_dict(self) -> dict[str, torch.Tensor | float]:
        self._check_fitted()
        return {"mean": self.mean.cpu(), "std": self.std.cpu(), "eps": self.eps}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor | float]) -> None:
        self.mean = state_dict["mean"].detach().float()
        self.std = state_dict["std"].detach().float()
        self.eps = float(state_dict.get("eps", self.eps))

    def to(self, device: torch.device | str) -> "UnitGaussianNormalizer":
        if self.mean is not None:
            self.mean = self.mean.to(device)
        if self.std is not None:
            self.std = self.std.to(device)
        return self

    def _check_fitted(self) -> None:
        if self.mean is None or self.std is None:
            raise RuntimeError("UnitGaussianNormalizer must be fitted or loaded before use.")

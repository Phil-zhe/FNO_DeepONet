from __future__ import annotations

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    if not hasattr(nn, name):
        raise ValueError(f"Unknown torch.nn activation: {name}")
    return getattr(nn, name)()


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        depth: int,
        activation: str = "GELU",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least 2.")
        layers: list[nn.Module] = []
        dims = [in_dim] + [hidden_dim] * (depth - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(_activation(activation))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BranchNet(MLP):
    pass


class TrunkNet(MLP):
    pass


class DeepONet(nn.Module):
    def __init__(
        self,
        n_sensors: int,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        branch_depth: int = 4,
        trunk_depth: int = 4,
        activation: str = "GELU",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.branch = BranchNet(n_sensors, latent_dim, hidden_dim, branch_depth, activation, dropout)
        self.trunk = TrunkNet(1, latent_dim, hidden_dim, trunk_depth, activation, dropout)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, branch_input: torch.Tensor, trunk_input: torch.Tensor) -> torch.Tensor:
        branch_out = self.branch(branch_input)
        b, n, c = trunk_input.shape
        trunk_out = self.trunk(trunk_input.reshape(b * n, c)).reshape(b, n, -1)
        return torch.einsum("bp,bnp->bn", branch_out, trunk_out) + self.bias

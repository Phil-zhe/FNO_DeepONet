from __future__ import annotations

import torch
from torch import nn


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_float = x.float()
            batch_size, _, nx = x_float.shape
            x_ft = torch.fft.rfft(x_float, dim=-1)
            out_ft = torch.zeros(
                batch_size,
                self.out_channels,
                nx // 2 + 1,
                device=x.device,
                dtype=torch.cfloat,
            )
            modes = min(self.modes, x_ft.shape[-1])
            out_ft[:, :, :modes] = torch.einsum(
                "bim,iom->bom", x_ft[:, :, :modes], self.weights[:, :, :modes]
            )
            out = torch.fft.irfft(out_ft, n=nx, dim=-1)
        return out.to(dtype=input_dtype)


class FNOBlock1d(nn.Module):
    def __init__(self, width: int, modes: int) -> None:
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.pointwise = nn.Conv1d(width, width, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.spectral(x) + self.pointwise(x))


class FNO1d(nn.Module):
    def __init__(self, modes: int = 16, width: int = 32, depth: int = 4, fc_dim: int = 128) -> None:
        super().__init__()
        self.lift = nn.Linear(2, width)
        self.blocks = nn.ModuleList([FNOBlock1d(width, modes) for _ in range(depth)])
        self.proj = nn.Sequential(
            nn.Linear(width, fc_dim),
            nn.GELU(),
            nn.Linear(fc_dim, 1),
        )

    def forward(self, a: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        if grid.dim() == 1:
            grid = grid.unsqueeze(0).expand(a.shape[0], -1)
        x = torch.stack((a, grid), dim=-1)
        x = self.lift(x)
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)
        x = x.permute(0, 2, 1)
        return self.proj(x).squeeze(-1)

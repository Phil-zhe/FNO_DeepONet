from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tqdm import tqdm


@dataclass(frozen=True)
class BurgersConfig:
    nx: int = 256
    n_samples: int = 1200
    nu: float = 0.01
    T: float = 1.0
    dt: float = 0.002
    k_max: int = 8
    seed: int = 2026
    dealias: bool = True


def make_grid(nx: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, nx, endpoint=False, dtype=np.float64)


def make_wavenumbers(nx: int) -> np.ndarray:
    freq = np.fft.fftfreq(nx, d=1.0 / nx)
    return 2.0 * np.pi * freq


def make_dealias_mask(nx: int) -> np.ndarray:
    mode_numbers = np.fft.fftfreq(nx) * nx
    return np.abs(mode_numbers) <= nx / 3


def random_initial_condition(
    x: np.ndarray,
    rng: np.random.Generator,
    k_max: int = 8,
    scale: float = 0.9,
) -> np.ndarray:
    u0 = np.zeros_like(x, dtype=np.float64)
    for k in range(1, k_max + 1):
        std = 1.0 / (k**2)
        a_k = rng.normal(0.0, std)
        b_k = rng.normal(0.0, std)
        u0 += a_k * np.sin(2.0 * np.pi * k * x)
        u0 += b_k * np.cos(2.0 * np.pi * k * x)

    max_abs = np.max(np.abs(u0))
    if max_abs > 0.0:
        u0 = scale * u0 / max_abs
    return u0


def solve_burgers_spectral(
    u0: np.ndarray,
    nu: float = 0.01,
    T: float = 1.0,
    dt: float = 0.002,
    dealias: bool = True,
) -> np.ndarray:
    u0 = np.asarray(u0, dtype=np.float64)
    nx = u0.shape[-1]
    n_steps = int(round(T / dt))
    if abs(n_steps * dt - T) > 1e-12:
        raise ValueError(f"T/dt must be an integer. Got T={T}, dt={dt}, n_steps={n_steps}.")

    k = make_wavenumbers(nx)
    mask = make_dealias_mask(nx) if dealias else None
    u_hat = np.fft.fft(u0)

    if mask is not None:
        u_hat = u_hat * mask

    denom = 1.0 + dt * nu * k**2
    for _ in range(n_steps):
        u = np.fft.ifft(u_hat).real
        nonlinear_hat = -0.5j * k * np.fft.fft(u**2)
        if mask is not None:
            nonlinear_hat = nonlinear_hat * mask
        u_hat = (u_hat + dt * nonlinear_hat) / denom
        if mask is not None:
            u_hat = u_hat * mask

    return np.fft.ifft(u_hat).real.astype(np.float64)


def generate_dataset(config: BurgersConfig) -> dict[str, np.ndarray | float]:
    x = make_grid(config.nx)
    rng = np.random.default_rng(config.seed)
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    progress = tqdm(total=config.n_samples, desc="Generating Burgers data")
    attempts = 0
    max_attempts = config.n_samples * 20
    while len(inputs) < config.n_samples:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(f"Generated only {len(inputs)} valid samples after {attempts} attempts.")

        u0 = random_initial_condition(x, rng, k_max=config.k_max)
        uT = solve_burgers_spectral(u0, nu=config.nu, T=config.T, dt=config.dt, dealias=config.dealias)
        if not (np.all(np.isfinite(u0)) and np.all(np.isfinite(uT))):
            continue

        inputs.append(u0.astype(np.float32))
        targets.append(uT.astype(np.float32))
        progress.update(1)

    progress.close()
    return {
        "a": np.stack(inputs, axis=0).astype(np.float32),
        "u": np.stack(targets, axis=0).astype(np.float32),
        "x": x.astype(np.float32),
        "nu": float(config.nu),
        "T": float(config.T),
        "dt": float(config.dt),
    }

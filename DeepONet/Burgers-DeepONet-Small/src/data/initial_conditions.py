from __future__ import annotations

import numpy as np


def generate_random_fourier_ic(
    x: np.ndarray,
    num_modes: int = 5,
    rng: np.random.Generator | None = None,
    max_abs: float = 1.2,
) -> np.ndarray:
    """Generate one smooth periodic initial condition with decaying Fourier modes."""
    rng = np.random.default_rng() if rng is None else rng
    u0 = np.zeros_like(x, dtype=np.float64)
    for k in range(1, num_modes + 1):
        scale = 1.0 / (k * k)
        a_k = rng.normal(0.0, scale)
        b_k = rng.normal(0.0, scale)
        u0 += a_k * np.sin(2.0 * np.pi * k * x) + b_k * np.cos(2.0 * np.pi * k * x)

    peak = float(np.max(np.abs(u0)))
    if peak > 1e-12:
        u0 = u0 / peak * min(peak, max_abs)
    return u0.astype(np.float64)


def generate_random_fourier_batch(
    x: np.ndarray,
    n_samples: int,
    num_modes: int = 5,
    seed: int = 42,
    max_abs: float = 1.2,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack(
        [generate_random_fourier_ic(x, num_modes, rng, max_abs) for _ in range(n_samples)],
        axis=0,
    )

from __future__ import annotations

from functools import partial

import numpy as np


def spectral_derivatives(u: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u_hat = np.fft.fft(u)
    ux = np.fft.ifft(1j * k * u_hat).real
    uxx = np.fft.ifft(-(k**2) * u_hat).real
    return ux, uxx


def _dealias_product(product: np.ndarray) -> np.ndarray:
    n = product.shape[-1]
    p_hat = np.fft.fft(product)
    cutoff = n // 3
    p_hat[cutoff + 1 : n - cutoff] = 0.0
    return np.fft.ifft(p_hat).real


def burgers_rhs(u: np.ndarray, k: np.ndarray, nu: float, dealias: bool = True) -> np.ndarray:
    ux, uxx = spectral_derivatives(u, k)
    nonlinear = u * ux
    if dealias:
        nonlinear = _dealias_product(nonlinear)
    rhs = -nonlinear + nu * uxx
    if not np.all(np.isfinite(rhs)):
        raise FloatingPointError("Burgers RHS produced NaN or Inf.")
    return rhs


def rk4_step(u: np.ndarray, dt: float, rhs_func) -> np.ndarray:
    k1 = rhs_func(u)
    k2 = rhs_func(u + 0.5 * dt * k1)
    k3 = rhs_func(u + 0.5 * dt * k2)
    k4 = rhs_func(u + dt * k3)
    out = u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("RK4 step produced NaN or Inf.")
    return out


def solve_burgers_1d(
    u0: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    nu: float = 0.01,
    dealias: bool = True,
) -> np.ndarray:
    nx = x.size
    dx = float(x[1] - x[0])
    k = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    kmax = float(np.max(np.abs(k)))
    sol = np.empty((t.size, nx), dtype=np.float32)
    u = np.asarray(u0, dtype=np.float64).copy()
    sol[0] = u.astype(np.float32)

    rhs = partial(burgers_rhs, k=k, nu=nu, dealias=dealias)
    for i in range(1, t.size):
        interval = float(t[i] - t[i - 1])
        max_speed = max(float(np.max(np.abs(u))), 1e-6)
        adv_dt = 0.4 * dx / max_speed
        diff_dt = 2.0 / max(nu * kmax * kmax, 1e-12)
        stable_dt = min(adv_dt, diff_dt, interval)
        n_substeps = max(1, int(np.ceil(interval / stable_dt)))
        dt = interval / n_substeps
        for _ in range(n_substeps):
            u = rk4_step(u, dt, rhs)
        sol[i] = u.astype(np.float32)
    return sol

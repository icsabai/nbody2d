#!/usr/bin/env python3
"""
ug_nbody2d.py ────────────────────────────────────────────────────────────────
Unified-Gravity (UG) upgrade of the didactic **nbody2d** demo
(https://github.com/jhidding/nbody2d).

Key fixes in **2025-06-13 revision**
-----------------------------------
* **Initial expansion rate** – if the user does not provide an initial Hubble
  parameter, the code now *computes* one from the box-average density so that
  the simulation starts **expanding** instead of contracting.
* Added CLI flag `--H0` (or `-H`) to specify the desired initial Hubble rate
  in code-units; the default `None` triggers automatic calculation.
* Live console logging now prints `a`, `H`, and `Ω_m` every 20 steps so you
  can monitor the background just like in the original Newtonian code.

This single file stays under 250 lines and needs only NumPy and Matplotlib.
"""

from __future__ import annotations
import argparse
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Physical constants in **code units**
# ─────────────────────────────────────────────────────────────────────────────
G2D = 1.0            # 2-D Newton constant (choose your own units)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Background (scale-factor) integrator
# ─────────────────────────────────────────────────────────────────────────────
class Background:
    """Evolves the homogeneous mode *a(t)* from UG Friedmann analogue.

    The ODE is
        d²a/dt² = -π G₂ᴅ ρ̄ a  (pressure-less dust)
    which we integrate in velocity form (a, adot).
    """

    def __init__(self, a0: float = 1.0, H0: float | None = None):
        self.a = a0                    # scale factor
        self.adot = None if H0 is None else H0 * a0
        self._needs_init = H0 is None  # wait for initial density

    # one-off initialisation *after* we know the mean density
    def maybe_init(self, rho_bar: float):
        if self._needs_init:
            self.adot = np.sqrt(2.0 * np.pi * G2D * rho_bar) * self.a  # expanding
            self._needs_init = False

    def step(self, rho_bar: float, dt: float):
        self.maybe_init(rho_bar)
        # Friedmann acceleration (dust)
        self.adot += -np.pi * G2D * rho_bar * self.a * dt
        self.a    += self.adot * dt

    # helper properties
    @property
    def H(self):
        return self.adot / self.a

    def rho_crit(self):
        return 2.0 * self.H**2 / (np.pi * G2D)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Simple particle container
# ─────────────────────────────────────────────────────────────────────────────
class Particles:
    def __init__(self, n: int, box: float):
        rng = np.random.default_rng(1)
        self.x  = rng.random((n, 2)) * box        # comoving positions x ∈ [0, box)
        self.p  = np.zeros((n, 2))                # canonical momenta (comoving)
        self.m  = np.ones(n) * (box**2 / n)       # equal masses, sets Ωₘ≈1 initially
        self.box = box

    # ────────────────────────────────────────────────────────────────────
    #  CIC density deposit ρ(x)            (physical 2-D density: M / a² dA)
    # ────────────────────────────────────────────────────────────────────
    def deposit_density(self, mesh_size: int, a: float) -> tuple[np.ndarray, float]:
        rho = np.zeros((mesh_size, mesh_size))
        h   = self.box / mesh_size                 # comoving grid spacing Δχ
        # integer grid indices of the lower-left cell of each particle
        gx = (self.x[:, 0] / h).astype(int) % mesh_size
        gy = (self.x[:, 1] / h).astype(int) % mesh_size
        np.add.at(rho, (gx, gy), self.m)          # cloud-in-cell weight = 1 (nearest grid-point)
        rho /= h**2 * a**2                        # convert to physical density ρ = m / (Δx_phys Δy_phys)
        return rho, rho.mean()

    # ────────────────────────────────────────────────────────────────────
    #  Particle push:   p = canonical momentum (comoving),   x = comoving
    #  Integrates   dp/dt = -a²∇Φ   −   H p      (UG weak-field, dust)
    #               dx/dt = p / (a m)
    #  using a 2-nd-order leap-frog with *built-in cosmic drag*.
    # ────────────────────────────────────────────────────────────────────
    def advance(self, acc: np.ndarray, dt: float, bg: 'Background'):
        H = bg.H
        drag = 1.0 - 0.5 * H * dt                # half-step cosmological drag factor
        # half-kick + drag
        self.p = drag * (self.p + 0.5 * dt * acc * bg.a**2)
        # drift in comoving coordinates χ = x / a
        self.x += dt * self.p / (self.m[:, None] * bg.a)
        self.x %= self.box                       # periodic BC
        # second half-kick (force evaluated at same time)
        self.p = drag * (self.p + 0.5 * dt * acc * bg.a**2)#(self) -> np.ndarray:
        """Return physical accelerations −t_a H^{a}{}_{0i}.  Here we take t_a=(1,0,0)."""
        Hx = fft.ifft2(self.Hk[1]).real  # H_{10}
        Hy = fft.ifft2(self.Hk[2]).real  # H_{20}
        return np.stack([-Hx, -Hy], axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Field solver in Fourier space (harmonic gauge projection)
# ─────────────────────────────────────────────────────────────────────────────
class UGField:
    """Stores and updates the 3 force-producing components H_{a0} (a=0,1,2)."""

    def __init__(self, n: int):
        self.n = n
        self.Hk   = np.zeros((3, n, n), dtype=complex)   # spectral field
        self.Hk_p = np.zeros_like(self.Hk)                # first derivative
        # wavenumbers
        kx = fft.fftfreq(n) * n * 2.0 * np.pi  # assume box length = n grid-units
        self.k2 = kx[:, None]**2 + kx[None, :]**2
        self.k2[0, 0] = 1.0  # avoid divide-by-zero (will never use k=0)

    def update(self, Tk: np.ndarray, dt: float):
        """Integrate □H = -κ T for a,0 components using leap-frog."""
        Eg = np.sqrt(8.0 * np.pi * G2D)  # coupling
        Hdd = -Eg * Tk - self.k2 * self.Hk
        # leap-frog
        self.Hk   += dt * self.Hk_p + 0.5 * dt**2 * Hdd
        self.Hk_p += dt * Hdd
        # enforce harmonic gauge  k·H = 0 for spatial indices (projection)
        kx = fft.fftfreq(self.n) * self.n * 2.0 * np.pi
        kx2d, ky2d = np.meshgrid(kx, kx, indexing='ij')
        k_dot_H = kx2d * self.Hk[1] + ky2d * self.Hk[2]
        self.Hk[1] -= kx2d * k_dot_H / self.k2
        self.Hk[2] -= ky2d * k_dot_H / self.k2

    def forces(self) -> np.ndarray:
        """Return physical accelerations −t_a H^{a}{}_{0i}.  Here we take t_a=(1,0,0)."""
        Hx = fft.ifft2(self.Hk[1]).real  # H_{10}
        Hy = fft.ifft2(self.Hk[2]).real  # H_{20}
        return np.stack([-Hx, -Hy], axis=-1)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def log_background(step: int, bg: Background, rho_bar: float):
    if step % 20 == 0:
        print(
            f"step {step:05d}  a={bg.a:.4f}  H={bg.H:.3e}  Ω_m={rho_bar/bg.rho_crit():.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Main simulation driver
# ─────────────────────────────────────────────────────────────────────────────

def run(mesh_size: int, n_particles: int, dt: float, n_steps: int, box: float, H0: float | None):
    parts  = Particles(n_particles, box)
    bg     = Background(a0=1.0, H0=H0)
    field  = UGField(mesh_size)

    fig, ax = plt.subplots()
    a_hist = []

    for istep in range(n_steps):
        # (1) density deposit → grid
        rho, rho_bar = parts.deposit_density(mesh_size, bg.a)
        # (2) first call sets initial H if not specified
        bg.step(rho_bar, dt)
        # (3) build T_{a0}: only energy density couples for dust, so
        Tk = np.zeros_like(field.Hk)
        Tk[0] = fft.fft2(rho)
        field.update(Tk, dt)
        # (4) gather forces (interpolated nearest grid point for simplicity)
        acc_grid = field.forces()
        acc      = acc_grid[(parts.x[:, 0]/box*mesh_size).astype(int)%mesh_size,
                             (parts.x[:, 1]/box*mesh_size).astype(int)%mesh_size]
        # (5) particle push
        parts.advance(acc, dt, bg)
        # (6) diagnostics
        log_background(istep, bg, rho_bar)
        a_hist.append(bg.a)

        if istep % 20 == 0:
            ax.clear()
            ax.plot(a_hist)
            ax.set_xlabel('step')
            ax.set_ylabel('a(t)')
            plt.pause(0.01)

    plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 7.  CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="2-D Unified-Gravity N-body toy")
    p.add_argument("--mesh_size", type=int, default=64)
    p.add_argument("--n_particles", type=int, default=10000)
    p.add_argument("--dt", type=float, default=1e-3)
    p.add_argument("--n_steps", type=int, default=200)
    p.add_argument("--box", type=float, default=1.0)
    p.add_argument("--H0", "-H", type=float, default=None,
                   help="initial Hubble parameter in code units (optional)")
    args = p.parse_args()

    run(args.mesh_size, args.n_particles, args.dt, args.n_steps, args.box, args.H0)

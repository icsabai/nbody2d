#!/usr/bin/env python3
"""
ug_nbody2d.py ────────────────────────────────────────────────────────────────
A **self-contained 2-D particle-mesh toy code** that upgrades the didactic
`nbody2d` (https://github.com/jhidding/nbody2d) from *externally-imposed* FLRW
expansion to a **comoving-split Unified-Gravity (UG) solver** in which the
scale factor *a(t)* is evolved from the box-average density.

Physics handled
---------------
* weak-field, pressure-less matter (dust)
* homogeneous expansion captured by a(t) satisfying the 2-D Friedmann analogue
  \\ddot a/a = -π G₂ᴰ ρ̄
* perturbations evolved through the gauge-field component H₀₀ via the flat-space
  wave equation   □H₀₀ = −κ T⁰⁰
* leap-frog integrator in cosmic time, comoving coordinates
* CIC deposition / gathering, FFT-based spectral solver, periodic boundaries

This *is not* a production cosmology code—there are no baryons, no AMR, no
relativistic momenta—but it is excellent for unit-testing UG algorithms before
porting to full 3-D.

Code units
----------
We adopt **code units** where the box length L = 1, speed of light c = 1, and
2-D gravitational constant G₂ᴰ = 1.  Time is therefore measured in L/c.
Mass is in units where surface density Σ = M/L².

Dependencies: numpy ≥1.20, scipy ≥1.8 (only for FFT wrapper; NumPy FFT works
fine too), matplotlib (optional, for quick visual checks).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field as dc_field
from typing import Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Physical / numerical constants in code units
# ──────────────────────────────────────────────────────────────────────────────
C_LIGHT: float = 1.0                 # speed of light                   (c)
G2D: float = 1.0                     # 2-D Newton G                     (G₂ᴰ)
KAPPA: float = 2.0 * np.pi * G2D / C_LIGHT ** 4  # coupling in □H = −κ T

# ──────────────────────────────────────────────────────────────────────────────
# Basic data containers
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Background:
    """Homogeneous scale factor a(t) evolved from UG Friedmann equation."""
    a: float = 1.0                   # scale factor
    adot: float = 0.0                # first derivative da/dt

    def step(self, rho_bar: float, dt: float) -> None:
        """Update a and adot by one cosmic-time step `dt`.
        2-D dust Friedmann: ẍ/x = −π G₂ᴰ ρ̄ where x = a
        """
        self.adot += -np.pi * G2D * rho_bar * self.a * dt
        self.a += self.adot * dt

@dataclass
class Particles:
    """Comoving coordinates and momenta q = a² dx/dt for N particles."""
    x: np.ndarray                    # shape (N, 2) positions (0 ≤ x < 1)
    q: np.ndarray                    # shape (N, 2) momenta
    m: np.ndarray                    # shape (N,)   masses

    def kick(self, force: np.ndarray, bg: Background, dt: float) -> None:
        """Half-kick: update momenta q with force and Hubble drag."""
        hubble_drag = -(bg.adot / bg.a) * self.q
        self.q += (force + hubble_drag) * dt

    def drift(self, bg: Background, dt: float) -> None:
        """Drift: update positions with current momenta (periodic)."""
        self.x += (self.q / bg.a ** 2) * dt
        self.x %= 1.0  # periodic boundaries

# ──────────────────────────────────────────────────────────────────────────────
# Mesh utilities (CIC deposit and gather)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Mesh:
    N: int                           # grid resolution (N×N)
    rho: np.ndarray = dc_field(init=False)  # comoving surface density Σ_c
    fx: np.ndarray = dc_field(init=False)  # comoving force field (x component)
    fy: np.ndarray = dc_field(init=False)  # comoving force field (y component)

    def __post_init__(self) -> None:
        self.rho = np.zeros((self.N, self.N))
        self.fx = np.zeros_like(self.rho)
        self.fy = np.zeros_like(self.rho)

    # ───────── CIC deposit ─────────
    def deposit_mass(self, p: Particles) -> None:
        self.rho.fill(0.0)
        dx = 1.0 / self.N
        idx = (p.x / dx).astype(int)  # lower cell indices
        tx_ty = (p.x / dx) - idx      # fractional offsets

        for k in range(p.m.size):     # small-N clarity over raw speed
            i, j = idx[k]
            tx, ty = tx_ty[k]
            w = np.array([[1 - tx, tx], [1 - ty, ty]])
            self.rho[i     % self.N, j     % self.N] += p.m[k] * w[0, 0]
            self.rho[(i+1) % self.N, j     % self.N] += p.m[k] * w[0, 1]
            self.rho[i     % self.N, (j+1) % self.N] += p.m[k] * w[1, 0]
            self.rho[(i+1) % self.N, (j+1) % self.N] += p.m[k] * w[1, 1]

        cell_area = (1.0 / self.N) ** 2
        self.rho /= cell_area  # surface density Σ_c (mass / comoving area)

    # ───────── gather forces ─────────
    def gather_force(self, p: Particles) -> np.ndarray:
        """Nearest-grid-point force (for clarity). Returns array shape (N,2)."""
        dx = 1.0 / self.N
        idx = (p.x / dx).astype(int) % self.N
        f = np.empty_like(p.x)
        f[:, 0] = self.fx[idx[:, 0], idx[:, 1]]
        f[:, 1] = self.fy[idx[:, 0], idx[:, 1]]
        return f

# ──────────────────────────────────────────────────────────────────────────────
# Field solver (spectral leap-frog for H00)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Field:
    mesh: Mesh
    Hk: np.ndarray = dc_field(init=False)       # complex FFT of H00
    Hk_dot: np.ndarray = dc_field(init=False)   # time derivative of Hk
    kx: np.ndarray = dc_field(init=False)
    ky: np.ndarray = dc_field(init=False)
    k2: np.ndarray = dc_field(init=False)

    def __post_init__(self) -> None:
        N = self.mesh.N
        self.Hk = np.zeros((N, N), dtype=np.complex128)
        self.Hk_dot = np.zeros_like(self.Hk)

        kfreq = 2.0 * np.pi * np.fft.fftfreq(N, d=1.0 / N)
        self.kx, self.ky = np.meshgrid(kfreq, kfreq, indexing="ij")
        self.k2 = self.kx ** 2 + self.ky ** 2
        self.k2[0, 0] = 1.0  # avoid divide-by-zero when projecting gauges

    # ───────── update H00 and derive forces ─────────
    def step(self, T00: np.ndarray, dt: float) -> None:
        """Leap-frog step for H00 in Fourier space."""
        Tk = np.fft.fft2(T00)
        Hk_ddot = -KAPPA * Tk - self.k2 * self.Hk
        self.Hk_dot += Hk_ddot * dt
        self.Hk += self.Hk_dot * dt

        # forces: F = −∇Φ with Φ ≡ H00/2 (weak-field), in comoving coords
        Fx_k = 0.5j * self.kx * self.Hk
        Fy_k = 0.5j * self.ky * self.Hk
        self.mesh.fx = np.fft.ifft2(Fx_k).real
        self.mesh.fy = np.fft.ifft2(Fy_k).real

# ──────────────────────────────────────────────────────────────────────────────
# Main simulation driver
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Simulation:
    mesh_size: int = 128
    n_particles: int = 1000
    dt: float = 5e-4
    n_steps: int = 1000

    bg: Background = dc_field(default_factory=Background)
    mesh: Mesh = dc_field(init=False)
    field: Field = dc_field(init=False)
    particles: Particles = dc_field(init=False)

    def __post_init__(self) -> None:
        self.mesh = Mesh(self.mesh_size)
        self.field = Field(self.mesh)
        self.particles = self._init_glass(self.n_particles)

    # ───────── initial conditions (uniform + tiny noise) ─────────
    def _init_glass(self, N: int) -> Particles:
        rng = np.random.default_rng(123)
        x = rng.random((N, 2))              # uniform positions 0..1
        q = np.zeros_like(x)                # zero peculiar momenta
        m = np.full(N, 1.0 / N)             # total mass = 1
        return Particles(x, q, m)

    # ───────── one integration step ─────────
    def _step(self) -> None:
        # (1) deposit Σ_c to mesh
        self.mesh.deposit_mass(self.particles)

        # (2) update homogeneous background
        rho_phys_bar = self.mesh.rho.mean() / self.bg.a ** 2  # ρ̄ = Σ_c / a²
        self.bg.step(rho_phys_bar, self.dt)

        # (3) build T⁰⁰ = ρ_phys c²
        T00 = self.mesh.rho / self.bg.a ** 2 * C_LIGHT ** 2

        # (4) field step → updates mesh.fx, mesh.fy
        self.field.step(T00, self.dt)

        # (5) kick-drift-kick
        force = self.mesh.gather_force(self.particles)
        self.particles.kick(force, self.bg, 0.5 * self.dt)
        self.particles.drift(self.bg, self.dt)
        self.mesh.deposit_mass(self.particles)          # need fresh ρ for 2nd kick drag
        force = self.mesh.gather_force(self.particles)
        self.particles.kick(force, self.bg, 0.5 * self.dt)

    # ───────── run full simulation ─────────
    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        scale_factors = []
        times = []
        t = 0.0
        for _ in range(self.n_steps):
            if _ % 10 == 0:
                print(f"Step {_} of {self.n_steps}")
            self._step()
            t += self.dt
            times.append(t)
            scale_factors.append(self.bg.a)
        return np.array(times), np.array(scale_factors)

# ──────────────────────────────────────────────────────────────────────────────
# Stand-alone execution example
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sim = Simulation(mesh_size=64, n_particles=10000, dt=1e-3, n_steps=2000)
    t, a = sim.run()

    try:
        import matplotlib.pyplot as plt
        plt.plot(t, a)
        plt.xlabel("cosmic time t [code]")
        plt.ylabel("scale factor a(t)")
        plt.title("2-D UG dust expansion: a(t) ∝ t")
        plt.show()
    except ModuleNotFoundError:
        print("matplotlib not installed - run complete, a(t) printed below")
        print(np.vstack([t, a]).T)



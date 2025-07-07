#!/usr/bin/env python3
"""
ug_nbody2d.py ────────────────────────────────────────────────────────────────
Unified-Gravity (UG) upgrade of the didactic **nbody2d** demo
(https://github.com/jhidding/nbody2d).

This modified version computes the scale factor 'a' locally for each grid
cell based on its density, then averages these values to get a global 'a'
for each timestep.

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
###G2D = 1.0            # 2-D Newton constant (choose your own units)
G2D = 1.0

OMEGAX = 1e-6

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Background (scale-factor) integrator
# ─────────────────────────────────────────────────────────────────────────────
class Background:
    """
    Evolves a grid of local scale factors a(t, x) based on the density in
    each cell, then provides the global average a(t) and H(t).

    The ODE for each cell is
        d²a/dt² = -π G₂ᴅ ρ(x) a  (pressure-less dust)
    which we integrate in velocity form (a, adot).
    """

    def __init__(self, mesh_size: int, a0: float = 1.0, H0: float | None = None):
        self.a = np.ones((mesh_size, mesh_size)) * a0
        self.adot = None if H0 is None else np.ones((mesh_size, mesh_size)) * H0 * a0
        
        self.aTRAD = a0                    # the traditional global scale factor
        self.adotTRAD = None if H0 is None else H0 * a0
        
        self._needs_init = H0 is None

    def maybe_init(self, rho: np.ndarray):
        """One-off initialisation of adot using the initial density grid."""
        if self._needs_init:
            # Use local density to set initial expansion rate in each cell
            self.adot = np.sqrt(2.0 * np.pi * G2D * rho) * self.a
            ###self.adotTRAD = np.sqrt(2.0 * np.pi * G2D * rho.mean()) * self.aTRAD  # expanding
            self.adotTRAD = np.sqrt(2.0 * np.pi * G2D * OMEGAX) * self.aTRAD  
            self._needs_init = False

    def step(self, rho: np.ndarray, dt: float):
        """Evolves the grid of local a and adot values."""
        self.maybe_init(rho)
        # Friedmann acceleration for each cell
        addot = -np.pi * G2D * rho * self.a
        self.adot += addot * dt
        self.a += self.adot * dt
        
        ###self.adotTRAD += -np.pi * G2D * rho.mean() * self.aTRAD * dt
        self.adotTRAD += -np.pi * G2D * OMEGAX * self.aTRAD * dt
        self.aTRAD    += self.adotTRAD * dt

    # --- Helper properties for global average values ---
    @property
    def a_global(self) -> float:
        return self.a.mean()

    @property
    def adot_global(self) -> float:
        return self.adot.mean()

    @property
    def H_global(self) -> float:
        # Avoid division by zero if a_global is somehow zero
        return self.adot_global / self.a_global if self.a_global != 0 else 0.0
    
    @property
    def HTRAD(self):
        return self.adotTRAD / self.aTRAD

    def rho_crit_global(self) -> float:
        return 2.0 * self.H_global**2 / (np.pi * G2D)

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

    def deposit_density(self, mesh_size: int, a: float) -> tuple[np.ndarray, float]:
        """CIC density deposit ρ(x) (physical 2-D density: M / a² dA)"""
        rho = np.zeros((mesh_size, mesh_size))
        h   = self.box / mesh_size
        gx = (self.x[:, 0] / h).astype(int) % mesh_size
        gy = (self.x[:, 1] / h).astype(int) % mesh_size
        np.add.at(rho, (gx, gy), self.m)
        rho /= h**2 * a**2
        return rho, rho.mean()

    def advance(self, acc: np.ndarray, dt: float, a: float, H: float):
        """
        Particle push with 2nd-order leap-frog and built-in cosmic drag.
        Integrates dp/dt = -a²∇Φ − H p and dx/dt = p / (a m).
        """
        drag = 1.0 - 0.5 * H * dt
        self.p = drag * (self.p + 0.5 * dt * acc * a**2)
        self.x += dt * self.p / (self.m[:, None] * a)
        self.x %= self.box
        self.p = drag * (self.p + 0.5 * dt * acc * a**2)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Field solver in Fourier space (harmonic gauge projection)
# ─────────────────────────────────────────────────────────────────────────────
class UGField:
    """Stores and updates the 3 force-producing components H_{a0} (a=0,1,2)."""
    def __init__(self, n: int):
        self.n = n
        self.Hk   = np.zeros((3, n, n), dtype=complex)
        self.Hk_p = np.zeros_like(self.Hk)
        kx = fft.fftfreq(n) * n * 2.0 * np.pi
        self.k2 = kx[:, None]**2 + kx[None, :]**2
        self.k2[0, 0] = 1.0

    def update(self, Tk: np.ndarray, dt: float):
        """Integrate □H = -κ T for a,0 components using leap-frog."""
        Eg = np.sqrt(8.0 * np.pi * G2D)
        Hdd = -Eg * Tk - self.k2 * self.Hk
        self.Hk   += dt * self.Hk_p + 0.5 * dt**2 * Hdd
        self.Hk_p += dt * Hdd
        kx = fft.fftfreq(self.n) * self.n * 2.0 * np.pi
        kx2d, ky2d = np.meshgrid(kx, kx, indexing='ij')
        k_dot_H = kx2d * self.Hk[1] + ky2d * self.Hk[2]
        self.Hk[1] -= kx2d * k_dot_H / self.k2
        self.Hk[2] -= ky2d * k_dot_H / self.k2

    def forces(self) -> np.ndarray:
        """Return physical accelerations −t_a H^{a}{}_{0i}. Here we take t_a=(1,0,0)."""
        Hx = fft.ifft2(self.Hk[1]).real
        Hy = fft.ifft2(self.Hk[2]).real
        return np.stack([-Hx, -Hy], axis=-1)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Diagnostics
# ─────────────────────────────────────────────────────────────────────────────
def log_background(step: int, a: float, H: float, rho_bar: float, rho_crit: float):
    if step % 20 == 0:
        Omega_m = rho_bar / rho_crit if rho_crit > 0 else 0
        print(f"step {step:05d}  a={a:.4f}  H={H:.3e}  Ω_m={Omega_m:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Main simulation driver
# ─────────────────────────────────────────────────────────────────────────────
def run(mesh_size: int, n_particles: int, dt: float, n_steps: int, box: float, H0: float | None):
    parts  = Particles(n_particles, box)
    bg     = Background(mesh_size, a0=0.1, H0=H0)
    field  = UGField(mesh_size)

    fig, ax = plt.subplots()
    a_hist = []
    H_hist = []
    aTRAD_hist = []
    HTRAD_hist = []

    for istep in range(n_steps):
        # (1) Use global 'a' from previous step to calculate current density
        a_current = bg.a_global
        rho, rho_bar = parts.deposit_density(mesh_size, a_current)

        # (2) Evolve local scale factor grid based on local density
        bg.step(rho, dt)
        
        # (3) Get new global a and H for this step's dynamics
        a_new = bg.a_global
        H_new = bg.H_global
        aTRAD_new = bg.aTRAD
        HTRAD_new = bg.HTRAD

        # (4) Update fields and gather forces
        Tk = np.zeros_like(field.Hk)
        Tk[0] = fft.fft2(rho)
        field.update(Tk, dt)
        acc_grid = field.forces()
        acc = acc_grid[(parts.x[:, 0]/box*mesh_size).astype(int)%mesh_size,
                       (parts.x[:, 1]/box*mesh_size).astype(int)%mesh_size]

        # (5) Particle push using the new global a and H
        parts.advance(acc, dt, a_new, H_new)

        # (6) Diagnostics and data recording
        log_background(istep, a_new, H_new, rho_bar, bg.rho_crit_global())
        a_hist.append(a_new)
        H_hist.append(H_new)
        aTRAD_hist.append(aTRAD_new)
        HTRAD_hist.append(HTRAD_new)

        if istep % 20 == 0:
            ax.clear()
            ax.plot(a_hist)
            ax.set_xlabel('step')
            ax.set_ylabel('a(t)')
            ax.set_title('Global Scale Factor Evolution')
            plt.pause(0.01)
    
    # Save results to a file
    time_array = np.arange(n_steps) * dt
    np.savez('output_data.npz', t=time_array, a=np.array(a_hist), H=np.array(H_hist), aTRAD=np.array(aTRAD_hist), HTRAD=np.array(HTRAD_hist))
    print("\nSimulation finished. Saved a(t) and H(t) to output_data.npz")

    plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 7.  CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="2-D Unified-Gravity N-body toy (local 'a' version)")
    p.add_argument("--mesh_size", type=int, default=64)
    p.add_argument("--n_particles", type=int, default=10000)
    p.add_argument("--dt", type=float, default=1e-3)
    p.add_argument("--n_steps", type=int, default=2000)
    p.add_argument("--box", type=float, default=1.0)
    p.add_argument("--H0", "-H", type=float, default=None,
                   help="initial Hubble parameter in code units (optional)")
    args = p.parse_args()

    run(args.mesh_size, args.n_particles, args.dt, args.n_steps, args.box, args.H0)
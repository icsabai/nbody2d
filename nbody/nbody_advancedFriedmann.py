from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.integrate import quad
import numba
import matplotlib.pyplot as plt

import sys
sys.path.append('.')
# Assuming cft.py is in the same directory or in the python path
try:
    import cft
except ImportError:
    print("Warning: 'cft' module not found. Please ensure it's in the python path.")
    # Create a dummy cft module for basic functionality if it's missing
    class DummyBox:
        def __init__(self, dim, N, L):
            self.dim = dim
            self.N = N
            self.L = L
            self.res = L / N
            self.shape = (N,) * dim
        @property
        def K(self):
            # A simplified K for the dummy module
            return np.fft.fftfreq(self.N, d=self.res) * 2 * np.pi

    class DummyPotential:
        def __call__(self, K):
            # A simplified potential for the dummy module
            return -1.0 / (np.sum(K**2, axis=0) + 1e-9)

    cft = type('cft', (), {'Box': DummyBox, 'Potential': DummyPotential})


from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Callable, Tuple
from functools import partial

# --- Start of Merged FriedmannIntegrator from friedmann_solver3.py ---
class FriedmannIntegrator:
    """
    Step-by-step Friedmann equation integrator for n-body simulations
    """
    def __init__(self, omega_m_func, omega_lambda=None, omega_k=None, h0=70.0, method='RK4'):
        self.omega_m_func = omega_m_func
        self.omega_lambda = omega_lambda
        self.omega_k = omega_k
        self.h0 = h0
        self.H0_inv_Gyr = h0 * 1.022e-3  # Convert to 1/Gyr
        self.method = method
        
        self.current_a = None
        self.current_t = None
        self.current_H = None
        self.current_omega_m = None
        
    def _get_density_params(self, a):
        omega_m_current = self.omega_m_func(a)
        
        if self.omega_lambda is None and self.omega_k is None:
            omega_lambda_current = 1.0 - omega_m_current
            omega_k_current = 0.0
        elif self.omega_lambda is None:
            omega_lambda_current = 1.0 - omega_m_current - self.omega_k
            omega_k_current = self.omega_k
        elif self.omega_k is None:
            omega_lambda_current = self.omega_lambda
            omega_k_current = 1.0 - omega_m_current - self.omega_lambda
        else:
            omega_lambda_current = self.omega_lambda
            omega_k_current = self.omega_k
            
        return omega_m_current, omega_lambda_current, omega_k_current
    
    def _friedmann_derivative(self, a, t):
        if a <= 0: return 0
        omega_m, omega_lambda, omega_k = self._get_density_params(a)
        H_squared = self.H0_inv_Gyr**2 * (omega_m/a**3 + omega_lambda + omega_k/a**2)
        if H_squared < 0: return 0
        H = np.sqrt(H_squared)
        return H * a
    
    def initialize(self, a_initial, t_initial=0.0):
        self.current_a = a_initial
        self.current_t = t_initial
        omega_m, omega_lambda, omega_k = self._get_density_params(a_initial)
        self.current_omega_m = omega_m
        H_squared = self.H0_inv_Gyr**2 * (omega_m/a_initial**3 + omega_lambda + omega_k/a_initial**2)
        self.current_H = np.sqrt(max(0, H_squared))
        
    def step(self, dt):
        if self.current_a is None:
            raise ValueError("Integrator not initialized. Call initialize() first.")
        
        a = self.current_a
        t = self.current_t
        k1 = self._friedmann_derivative(a, t)
        k2 = self._friedmann_derivative(a + 0.5*dt*k1, t + 0.5*dt)
        k3 = self._friedmann_derivative(a + 0.5*dt*k2, t + 0.5*dt)
        k4 = self._friedmann_derivative(a + dt*k3, t + dt)
        new_a = a + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
            
        self.current_a = new_a
        self.current_t += dt
        omega_m, omega_lambda, omega_k = self._get_density_params(self.current_a)
        self.current_omega_m = omega_m
        H_squared = self.H0_inv_Gyr**2 * (omega_m/self.current_a**3 + omega_lambda + omega_k/self.current_a**2)
        self.current_H = np.sqrt(max(0, H_squared))
        
        return self.get_state()
    
    def get_state(self):
        return {'a': self.current_a, 't': self.current_t, 'H': self.current_H, 'omega_m': self.current_omega_m}

# --- End of Merged FriedmannIntegrator ---


@dataclass
class Cosmology:
    H0 : float
    OmegaM : float
    OmegaL : float

    @property
    def OmegaK(self):
        return 1 - self.OmegaM - self.OmegaL
    @property
    def G(self):
        return 3./2 * self.OmegaM * self.H0**2
        
    # This is the original, simple Friedmann solution
    def da_original(self, a):
        return self.H0 * a * np.sqrt(
                  self.OmegaL \
                + self.OmegaM * a**-3 \
                + self.OmegaK * a**-2)
    def growing_mode(self, a):
        if isinstance(a, np.ndarray):
            return np.array([self.growing_mode(b) for b in a])
        elif a <= 0.001:
            return a
        else:
            # Note: The original da() is used here as it's part of the initial condition generation
            return self.da_original(a)/a * quad(lambda b: self.da_original(b)**(-3), 0.00001, a)[0] + 0.00001

LCDM = Cosmology(68.0, 0.31, 0.69)
EdS = Cosmology(70.0, 1.0, 0.0)


@numba.jit
def md_cic_2d(shape: Tuple[int], pos: np.ndarray, tgt: np.ndarray):
    for i in range(len(pos)):
        idx0, idx1 = int(np.floor(pos[i,0])), int(np.floor(pos[i,1]))
        f0, f1     = pos[i,0] - idx0, pos[i,1] - idx1
        tgt[idx0 % shape[0], idx1 % shape[1]] += (1 - f0) * (1 - f1)
        tgt[(idx0 + 1) % shape[0], idx1 % shape[1]] += f0 * (1 - f1)
        tgt[idx0 % shape[0], (idx1 + 1) % shape[1]] += (1 - f0) * f1
        tgt[(idx0 + 1) % shape[0], (idx1 + 1) % shape[1]] += f0 * f1

class Interp2D:
    "Reasonably fast bilinear interpolation routine"
    def __init__(self, data):
        self.data = data
        self.shape = data.shape

    def __call__(self, x):
        X1 = np.floor(x).astype(int) % self.shape[0]
        X2 = (X1 + 1) % self.shape[0]
        Y1 = np.floor(x).astype(int) % self.shape[1]
        Y2 = (Y1 + 1) % self.shape[1]
        
        xm = x % 1.0
        xn = 1.0 - xm

        f1 = self.data[X1[:,0], Y1[:,1]]
        f2 = self.data[X2[:,0], Y1[:,1]]
        f3 = self.data[X1[:,0], Y2[:,1]]
        f4 = self.data[X2[:,0], Y2[:,1]]

        return  f1 * xn[:,0] * xn[:,1] + \
                f2 * xm[:,0] * xn[:,1] + \
                f3 * xn[:,0] * xm[:,1] + \
                f4 * xm[:,0] * xm[:,1]


def gradient_2nd_order(F, i):
    return   (1./12 * np.roll(F, -2, axis=i) - 2./3  * np.roll(F, -1, axis=i) \
           + 2./3  * np.roll(F,  1, axis=i) - 1./12 * np.roll(F,  2, axis=i))

Vector = TypeVar("Vector", bound=np.ndarray)

@dataclass
class State(Generic[Vector]):
    time : float
    position : Vector
    momentum : Vector

    def kick(self, dt: float, h: 'HamiltonianSystem[Vector]') -> State[Vector]:
        self.momentum += dt * h.momentumEquation(self)
        return self

    def drift(self, dt: float, h: 'HamiltonianSystem[Vector]') -> State[Vector]:
        self.position += dt * h.positionEquation(self)
        return self

    def wait(self, dt: float) -> State[Vector]:
        self.time += dt
        return self


class HamiltonianSystem(ABC, Generic[Vector]):
    @abstractmethod
    def positionEquation(self, s: State[Vector]) -> Vector:
        raise NotImplementedError

    @abstractmethod
    def momentumEquation(self, s: State[Vector]) -> Vector:
        raise NotImplementedError

def leap_frog(dt: float, h: HamiltonianSystem[Vector], s: State[Vector]) -> State[Vector]:
    # Note: time is now managed by the Friedmann integrators
    return s.kick(dt, h).drift(dt, h)

def iterate_step(step: Callable[[State[Vector]], State[Vector]], halt: Callable[[State[Vector]], bool], init: State[Vector], system: 'PoissonVlasov') -> State[Vector]:
    state = init
    
    # --- History tracking for plotting ---
    history = {
        'time': [],
        'a_original': [], 'H_original': [],
        'a_avgdens': [], 'H_avgdens': [],
        'a_cellavg': [], 'H_cellavg': [],
    }

    while not halt(state):
        state = step(state)
        system.update_cosmology() # This is the new key step
        
        # --- Record data for plotting ---
        history['time'].append(system.cosmology_original['t'])
        history['a_original'].append(system.cosmology_original['a'])
        history['H_original'].append(system.cosmology_original['H'])
        history['a_avgdens'].append(system.cosmology_avgdens['a'])
        history['H_avgdens'].append(system.cosmology_avgdens['H'])
        history['a_cellavg'].append(system.cosmology_cellavg['a'])
        history['H_cellavg'].append(system.cosmology_cellavg['H'])

    return state, history


class PoissonVlasov(HamiltonianSystem[np.ndarray]):
    def __init__(self, box, cosmology, particle_mass, dt):
        self.box = box
        self.cosmology_params = cosmology
        self.particle_mass = particle_mass
        self.dt = dt
        self.delta = np.zeros(self.box.shape, dtype='f8')
        self.a_init = None

        # --- Initialize the three cosmology scenarios ---
        self.a_original = None
        self.da_original_val = None

        # 1. Original simple solver
        self.cosmology_original = {'a': None, 't': 0.0, 'H': None}


        # 2. Advanced solver with average density
        self.integrator_avgdens = FriedmannIntegrator(
            lambda a: np.mean(self.delta + 1) * self.cosmology_params.OmegaM, 
            self.cosmology_params.OmegaL, self.cosmology_params.OmegaK, self.cosmology_params.H0
        )
        self.cosmology_avgdens = None

        # 3. Advanced solver for each cell
        self.cell_density_funcs = [lambda a, i=i, j=j: (self.delta[i, j] + 1) * self.cosmology_params.OmegaM 
                                   for i in range(box.shape[0]) for j in range(box.shape[1])]
        self.integrators_cell = [FriedmannIntegrator(f, self.cosmology_params.OmegaL, self.cosmology_params.OmegaK, self.cosmology_params.H0) 
                                 for f in self.cell_density_funcs]
        self.cosmology_cellavg = None


    def initialize_cosmology(self, a_init, t_init=0.0):
        self.a_init = a_init
        # 1. Original
        self.cosmology_original['a'] = a_init
        self.cosmology_original['t'] = t_init
        self.da_original_val = self.cosmology_params.da_original(a_init)
        self.cosmology_original['H'] = self.da_original_val / a_init

        # 2. Avg Density
        self.integrator_avgdens.initialize(a_init, t_init)
        self.cosmology_avgdens = self.integrator_avgdens.get_state()
        
        # 3. Cell-by-cell
        for integrator in self.integrators_cell:
            integrator.initialize(a_init, t_init)
        
        # Calculate initial average for cellavg
        a_vals = [integ.current_a for integ in self.integrators_cell]
        H_vals = [integ.current_H for integ in self.integrators_cell]
        self.cosmology_cellavg = {
            'a': np.mean(a_vals), 't': t_init, 'H': np.mean(H_vals), 'omega_m': np.mean([integ.current_omega_m for integ in self.integrators_cell])
        }

    def update_cosmology(self):
        # 1. Original
        # We use a simple Euler step for the original model to keep it basic
        self.cosmology_original['a'] += self.dt * self.da_original_val
        self.cosmology_original['t'] += self.dt
        self.da_original_val = self.cosmology_params.da_original(self.cosmology_original['a'])
        self.cosmology_original['H'] = self.da_original_val / self.cosmology_original['a']

        # 2. Avg Density
        self.cosmology_avgdens = self.integrator_avgdens.step(self.dt)

        # 3. Cell-by-cell
        a_vals, H_vals, omega_m_vals = [], [], []
        for integrator in self.integrators_cell:
            state = integrator.step(self.dt)
            a_vals.append(state['a'])
            H_vals.append(state['H'])
            omega_m_vals.append(state['omega_m'])

        self.cosmology_cellavg['a'] = np.mean(a_vals)
        self.cosmology_cellavg['t'] += self.dt
        self.cosmology_cellavg['H'] = np.mean(H_vals)
        self.cosmology_cellavg['omega_m'] = np.mean(omega_m_vals)


    def positionEquation(self, s: State[np.ndarray]) -> np.ndarray:
        # For simplicity, we use the average density scale factor for particle movement
        a = self.cosmology_avgdens['a']
        da = self.cosmology_avgdens['H'] * a
        return s.momentum / (a**2 * da)

    def momentumEquation(self, s: State[np.ndarray]) -> np.ndarray:
        # We use the average density scale factor here as well
        a = self.cosmology_avgdens['a']
        da = self.cosmology_avgdens['H'] * a
        
        x_grid = s.position / self.box.res
        self.delta.fill(0.0)
        md_cic_2d(self.box.shape, x_grid, self.delta)
        self.delta *= self.particle_mass
        self.delta -= 1.0

        assert abs(self.delta.mean()) < 1e-6, "total mass should be normalised"

        delta_f = np.fft.fftn(self.delta)
        # Assuming cft.Potential works as in the original script
        kernel = cft.Potential()(self.box.K)
        phi = np.fft.ifftn(delta_f * kernel).real * self.cosmology_params.G / a
        
        acc_x = Interp2D(gradient_2nd_order(phi, 0))
        acc_y = Interp2D(gradient_2nd_order(phi, 1))
        acc = np.c_[acc_x(x_grid), acc_y(x_grid)] / self.box.res
        
        return -acc / da


def a2r(B, X):
    return X.transpose([1,2,0]).reshape([B.N**2, 2])

def r2a(B, x):
    return x.reshape([B.N, B.N, 2]).transpose([2,0,1])

class Zeldovich:
    def __init__(self, B_mass: cft.Box, B_force: cft.Box, cosmology: Cosmology, phi: np.ndarray):
        self.bm = B_mass
        self.bf = B_force
        self.cosmology  = cosmology
        self.u = np.array([-gradient_2nd_order(phi, 0),
                           -gradient_2nd_order(phi, 1)]) / self.bm.res

    def state(self, a_init: float) -> State[np.ndarray]:
        # Using the original cosmology's growing mode for initial conditions
        D_init = self.cosmology.growing_mode(a_init)
        
        # Correctly apply Zeldovich approximation
        # Position: q = x + D(t) * u(x)
        # Momentum: p = a^2 * d/dt(q) = a^2 * dD/dt * u(x)
        # We use time 'a' so d/dt = da/dt * d/da, momentum p = a^2 * da/dt * dD/da * u(x)
        # For simplicity, we use an approximation p ~ a * u
        
        X = a2r(self.bm, np.indices(self.bm.shape) * self.bm.res + D_init * self.u)
        P = a2r(self.bm, a_init**2 * self.cosmology.da_original(a_init) * D_init/a_init * self.u)
        
        return State(time=a_init, position=X, momentum=P)

    @property
    def particle_mass(self):
        return (self.bf.N / self.bm.N)**self.bm.dim

def plot_results(history):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot scale factors
    ax1.plot(history['time'], history['a_original'], 'k--', label='a_original (Simple Solver)')
    ax1.plot(history['time'], history['a_avgdens'], 'b-', label='a_avgdens (Advanced Solver, Avg Density)')
    ax1.plot(history['time'], history['a_cellavg'], 'r:', label='a_cellavg (Advanced Solver, Cell Avg)')
    ax1.set_ylabel('Scale Factor (a)')
    ax1.set_title('Evolution of Scale Factor and Hubble Parameter')
    ax1.legend()
    ax1.grid(True)

    # Plot Hubble parameters
    ax2.plot(history['time'], history['H_original'], 'k--', label='H_original')
    ax2.plot(history['time'], history['H_avgdens'], 'b-', label='H_avgdens')
    ax2.plot(history['time'], history['H_cellavg'], 'r:', label='H_cellavg')
    ax2.set_xlabel('Time (Gyr)')
    ax2.set_ylabel('Hubble Parameter (H) [1/Gyr]')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("cosmology_comparison.png")
    plt.show()

if __name__ == "__main__":
    if 'cft' in sys.modules and not isinstance(cft, type):
        import cft

    N = 64  # Reduced for faster execution
    B_m = cft.Box(2, N, 50.0)

    A = 10
    seed = 4
    # Ensure these functions from cft are available
    try:
        Power_spectrum = cft.Power_law(-0.5) * cft.Scale(B_m, 0.2) * cft.Cutoff(B_m)
        phi = cft.garfield(B_m, Power_spectrum, cft.Potential(), seed) * A
    except (NameError, AttributeError):
        print("Using dummy power spectrum and field due to missing 'cft' functions.")
        k = np.linalg.norm(B_m.K, axis=0)
        Power_spectrum = np.exp(-k**2 / (2 * (0.2 * 2 * np.pi / B_m.L)**2))
        Power_spectrum[k==0] = 0
        phi = np.fft.ifftn(np.sqrt(Power_spectrum) * (np.random.randn(*B_m.shape) + 1j * np.random.randn(*B_m.shape))).real * A


    force_box = cft.Box(2, N*2, B_m.L)
    
    # Use a cosmology with dark energy for a more interesting evolution
    cosmology_model = LCDM 
    
    za = Zeldovich(B_m, force_box, cosmology_model, phi)
    
    a_initial = 0.1 # Start at a later time
    dt = 0.01 # Timestep in Gyr
    
    state = za.state(a_initial)
    system = PoissonVlasov(force_box, cosmology_model, za.particle_mass, dt)
    system.initialize_cosmology(a_initial)
    
    stepper = partial(leap_frog, dt, system)
    final_state, history = iterate_step(stepper, lambda s: system.cosmology_avgdens['a'] > 2.0, state, system)

    print("Simulation finished. Plotting results...")
    plot_results(history)
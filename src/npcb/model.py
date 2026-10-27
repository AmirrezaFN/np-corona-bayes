"""Minimal nanoparticle transport model with an evolving protein corona.

State y = [c, B, L, T, R]
    c : corona coverage of the particle surface (0 = bare, 1 = fully coated)
    B : blood        (fraction of injected dose)
    L : liver/spleen (mononuclear phagocyte system, MPS)
    T : tumour
    R : rest of body

Time is in hours. Tumour washout returns to blood, so B + L + T + R = 1 at all times.
"""

from dataclasses import dataclass, asdict

import numpy as np
from scipy.integrate import solve_ivp

STATE_NAMES = ("c", "B", "L", "T", "R")


@dataclass
class Params:
    k_on0: float = 0.5      # corona formation rate for an un-PEGylated particle [1/h]
    k_mps0: float = 0.15    # MPS uptake rate of a bare particle [1/h]
    alpha: float = 4.0      # extra MPS uptake when fully coated (alpha = 0 -> fixed particle)
    k_tum0: float = 0.004   # max tumour uptake rate, at optimal size [1/h]
    d_opt: float = 100.0    # optimal diameter for tumour extravasation [nm]
    w: float = 60.0         # width of the size window [nm]
    k_te: float = 0.02      # tumour washout back to blood [1/h]
    k_r: float = 0.1        # blood -> rest of body [1/h]
    k_rb: float = 0.05      # rest of body -> blood [1/h]

    def as_dict(self):
        return asdict(self)


def rates(d, p, prm: Params):
    """Map particle properties (diameter d [nm], PEG density p in [0, 1]) to rate constants."""
    k_on = prm.k_on0 * (1.0 - p)
    k_tum = prm.k_tum0 * np.exp(-(((d - prm.d_opt) / prm.w) ** 2))
    return k_on, k_tum


def rhs(t, y, d, p, prm: Params):
    c, B, L, T, R = y
    k_on, k_tum = rates(d, p, prm)
    k_mps = prm.k_mps0 * (1.0 + prm.alpha * c)

    dc = k_on * (1.0 - c)
    dB = -(k_mps + k_tum + prm.k_r) * B + prm.k_rb * R + prm.k_te * T
    dL = k_mps * B
    dT = k_tum * B - prm.k_te * T
    dR = prm.k_r * B - prm.k_rb * R
    return np.array([dc, dB, dL, dT, dR])


Y0 = np.array([0.0, 1.0, 0.0, 0.0, 0.0])  # bare particle, whole dose in blood


def solve(d, p, prm: Params, t_eval):
    """Reference solution with an adaptive solver. Returns array (len(t_eval), 5)."""
    t_eval = np.asarray(t_eval, dtype=float)
    sol = solve_ivp(rhs, (0.0, t_eval[-1]), Y0, t_eval=t_eval, args=(d, p, prm),
                    method="LSODA", rtol=1e-8, atol=1e-10)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T


def solve_rk4(d, p, prm: Params, t_eval, dt=0.05):
    """Fixed-step RK4. Same scheme we will later write in JAX for inference."""
    t_eval = np.asarray(t_eval, dtype=float)
    n = int(np.ceil(t_eval[-1] / dt))
    ts = np.linspace(0.0, n * dt, n + 1)
    ys = np.empty((n + 1, 5))
    ys[0] = Y0
    y = Y0.copy()
    for i in range(n):
        t = ts[i]
        k1 = rhs(t, y, d, p, prm)
        k2 = rhs(t + dt / 2, y + dt / 2 * k1, d, p, prm)
        k3 = rhs(t + dt / 2, y + dt / 2 * k2, d, p, prm)
        k4 = rhs(t + dt, y + dt * k3, d, p, prm)
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        ys[i + 1] = y
    # linear interpolation onto the requested times
    return np.column_stack([np.interp(t_eval, ts, ys[:, j]) for j in range(5)])


def tumour_auc(d, p, prm: Params, t_end=48.0):
    """Area under the tumour curve, 0..t_end [%ID*h / 100]. Used by the counterfactual."""
    t = np.linspace(0.0, t_end, 481)
    return np.trapezoid(solve(d, p, prm, t)[:, 3], t)

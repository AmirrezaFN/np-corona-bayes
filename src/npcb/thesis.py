"""Bayesian ODE surrogate of my M.Sc. agent-based tumour-immune model (Sharif University of Technology).

Data: replicate-averaged ABM output for tumours with n = 1..6 antigenically distinct subclones
(total tumour cells N, total T cells T, activated T cells A; time step 2500 s).

Surrogate, fitted on the late (vascularised, immune-active) phase, day ~30 to ~40.5:
    dN/dt = r_n N ln(K_n / N) - kappa A          Gompertz growth minus T-cell killing
    dA/dt = a_n T(t) - delta A                    activation of the T cells present, and loss
T(t) is taken from the ABM output (forcing), so the activation rate a_n is "per T cell present".

The question for the posterior: activated T cells fall with heterogeneity, but is that because fewer T cells
are present, or because each T cell is less likely to be activated?  Encoded as
    log a_n = mu_a + beta_a * (n - 1)
beta_a < 0 means per-cell activation itself declines with the number of subclones.
"""

import csv
from pathlib import Path

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

STEP_S = 2500.0
DT = 0.05            # d, RK4 step
STEP0, STEP1, EVERY = 1050, 1400, 10  # fitting window (ABM steps) and thinning


def load_abm(folder="data/thesis_abm"):
    def read(name):
        rows = [r for r in csv.reader(open(Path(folder) / f"{name}.csv")) if r and r[0].replace(".", "").isdigit()]
        return np.array(rows, float)
    N, T, A = read("Total_Tumor_Cells"), read("Total_T_Cells"), read("Activated_T_Cells")
    day = N[:, 0] * STEP_S / 86400
    return day, N[:, 1:], T[:, 1:], A[:, 1:]  # columns: n = 1..6 subclones


def load_run_per_clone(run_dir):
    """Single ABM run with per-subclone counters (V16: 2 clones, V17: 4 clones).

    Returns day, N (steps, k) tumour cells per clone (active + quiescent), A (steps, k) activated T cells
    specific to each clone.
    """
    import scipy.io

    m = {k: v.ravel().astype(float) for k, v in scipy.io.loadmat(Path(run_dir) / "FinalResults.mat").items()
         if not k.startswith("__")}
    n_cl = sum(1 for k in m if k.startswith("total_TLa") and k[-1].isdigit())
    N = np.column_stack([m[f"total_cella{i}"] + m[f"total_cellq{i}"] for i in range(1, n_cl + 1)])
    A = np.column_stack([m[f"total_TLa{i}"] for i in range(1, n_cl + 1)])
    day = np.arange(len(N)) * STEP_S / 86400
    return day, N, A


def build_clone_model(day, N, A, per_clone_growth=False):
    """Per-subclone killing model for one ABM run.

        dN_k/dt = r_k N_k ln(K / sum_j N_j) - kappa A_k(t)

    Subclones share space (one K for the whole tumour) and one killing rate kappa; A_k(t) is taken from the ABM.
    per_clone_growth=False: r_k = r for all clones.  True: log r_k ~ Normal(mu_r, tau_r) (partial pooling), which
    lets a clone simply grow slower instead of being killed -- the test of whether kappa is really identified.
    """
    n_cl = N.shape[1]
    t0, t1 = day[STEP0], day[STEP1]
    n_steps = int(round((t1 - t0) / DT))
    grid = t0 + DT * np.arange(n_steps + 1)
    A_at = lambda tt: np.column_stack([np.interp(tt, day, A[:, j]) for j in range(n_cl)])
    A_force = np.stack([A_at(grid[:-1]), A_at(grid[:-1] + DT / 2), A_at(grid[1:])], axis=1)

    obs_steps = np.arange(STEP0, STEP1 + 1, EVERY)
    obs_idx = np.round((day[obs_steps] - t0) / DT).astype(int)

    with pm.Model() as model:
        if per_clone_growth:
            mu_r = pm.Normal("mu_r", np.log(0.1), 1.0)
            tau_r = pm.HalfNormal("tau_r", 0.2)                  # clones of one tumour: growth rates within ~20 %
            z_r = pm.Normal("z_r", 0.0, 1.0, shape=n_cl)          # non-centred: avoids the funnel at small tau_r
            r = pm.Deterministic("r", pt.exp(mu_r + tau_r * z_r))
        else:
            r = pm.LogNormal("r", np.log(0.1), 1.0)
        K = pm.LogNormal("K", np.log(5e5), 1.0)
        kappa = pm.LogNormal("kappa", np.log(1.0), 1.5)              # kills per activated T cell per day
        sig = pm.HalfNormal("sigma", 0.3)

        def rhs(y, Ak, r, K, kappa):
            return r * y * pt.log(K / pt.sum(y)) - kappa * Ak

        def step(Af, y, r, K, kappa):
            k1 = rhs(y, Af[0], r, K, kappa)
            k2 = rhs(y + DT / 2 * k1, Af[1], r, K, kappa)
            k3 = rhs(y + DT / 2 * k2, Af[1], r, K, kappa)
            k4 = rhs(y + DT * k3, Af[2], r, K, kappa)
            return pt.maximum(y + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4), 1.0)

        y0 = pt.as_tensor(N[STEP0])
        ys = pytensor.scan(step, sequences=[pt.as_tensor(A_force)], outputs_info=y0,
                           non_sequences=[r, K, kappa], return_updates=False)
        ys = pt.concatenate([y0[None], ys], axis=0)[obs_idx]      # (n_obs, n_cl)
        pm.Deterministic("N_pred", ys)
        pm.Normal("log_N", pt.log(ys), sig, observed=np.log(N[obs_steps]))
    return model, obs_steps


def build_model(day, N, T, A):
    n_cl = N.shape[1]
    t0, t1 = day[STEP0], day[STEP1]
    n_steps = int(round((t1 - t0) / DT))
    grid = t0 + DT * np.arange(n_steps + 1)
    # T(t) forcing at the start, midpoint and end of each RK4 step: (n_steps, 3, n_cl)
    T_at = lambda tt: np.column_stack([np.interp(tt, day, T[:, j]) for j in range(n_cl)])
    T_force = np.stack([T_at(grid[:-1]), T_at(grid[:-1] + DT / 2), T_at(grid[1:])], axis=1)

    obs_steps = np.arange(STEP0, STEP1 + 1, EVERY)
    obs_idx = np.round((day[obs_steps] - t0) / DT).astype(int)
    y0 = np.stack([N[STEP0], np.maximum(A[STEP0], 1.0)], axis=1)  # (n_cl, 2)
    n_sub = np.arange(n_cl)

    with pm.Model() as model:
        r = pm.LogNormal("r", np.log(0.1), 1.0, shape=n_cl)          # Gompertz rate [1/d]
        K = pm.LogNormal("K", np.log(5e5), 1.0, shape=n_cl)          # carrying capacity [cells]
        kappa = pm.LogNormal("kappa", np.log(1.0), 1.0)              # kills per activated T cell per day
        mu_a = pm.Normal("mu_a", np.log(0.01), 1.5)
        beta_a = pm.Normal("beta_a", 0.0, 0.5)                       # change in log activation per extra subclone
        a = pm.Deterministic("a", pt.exp(mu_a + beta_a * n_sub))
        delta = pm.LogNormal("delta", np.log(0.2), 1.0)              # loss of activated state [1/d]
        sig_N = pm.HalfNormal("sigma_N", 0.2)
        sig_A = pm.HalfNormal("sigma_A", 0.3)

        def rhs(y, Tt, r, K, kappa, a, delta):
            Nn, An = y[:, 0], y[:, 1]
            return pt.stack([r * Nn * pt.log(K / Nn) - kappa * An, a * Tt - delta * An], axis=1)

        def step(Tf, y, r, K, kappa, a, delta):
            k1 = rhs(y, Tf[0], r, K, kappa, a, delta)
            k2 = rhs(y + DT / 2 * k1, Tf[1], r, K, kappa, a, delta)
            k3 = rhs(y + DT / 2 * k2, Tf[1], r, K, kappa, a, delta)
            k4 = rhs(y + DT * k3, Tf[2], r, K, kappa, a, delta)
            return pt.maximum(y + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4), 1e-3)

        y0_t = pt.as_tensor(y0)
        ys = pytensor.scan(step, sequences=[pt.as_tensor(T_force)], outputs_info=y0_t,
                           non_sequences=[r, K, kappa, a, delta], return_updates=False)
        ys = pt.concatenate([y0_t[None], ys], axis=0)[obs_idx]   # (n_obs, n_cl, 2)

        pm.Normal("log_N", pt.log(ys[:, :, 0]), sig_N, observed=np.log(N[obs_steps]))
        pm.Normal("log_A", pt.log(ys[:, :, 1]), sig_A, observed=np.log(np.maximum(A[obs_steps], 1.0)))
    return model, obs_steps

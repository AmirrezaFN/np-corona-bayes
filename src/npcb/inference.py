"""PyMC model: Bayesian calibration of the transport ODE against biodistribution data.

The ODE is solved inside the PyMC graph with fixed-step RK4 (pytensor.scan), all formulations at once,
so the NUTS sampler (run through NumPyro/JAX) gets exact gradients through the solver.
"""

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

DT = 0.05  # h; every observation time is a multiple of DT
COLS = {"B": 1, "L": 2, "T": 3}
RATE_KEYS = ("k_on", "k_mps0", "alpha", "k_tum", "k_te", "k_r", "k_rb")  # what the ODE right-hand side uses

# Name, prior median, prior log-sd. Medians are deliberately NOT the true values:
# they encode order-of-magnitude knowledge only (log-sd 1 = a factor ~e either way).
PRIORS = {
    "k_on0":  (0.3, 1.0),    # corona forms over hours
    "k_mps0": (0.1, 1.0),    # MPS clearance over hours
    "alpha":  (2.0, 1.0),    # corona makes clearance a few times faster
    "k_tum0": (0.003, 1.0),  # tumour gets <1 % of the dose
    "k_te":   (0.03, 1.0),   # tumour retention over days
    "k_r":    (0.1, 1.0),
    "k_rb":   (0.1, 1.0),
}


def _rhs(y, k):
    """y: (n_form, 5); every rate in k is a (n_form,) vector or a scalar."""
    c, B, L, T, R = (y[:, i] for i in range(5))
    k_mps = k["k_mps0"] * (1 + k["alpha"] * c)
    return pt.stack([
        k["k_on"] * (1 - c),
        -(k_mps + k["k_tum"] + k["k_r"]) * B + k["k_rb"] * R + k["k_te"] * T,
        k_mps * B,
        k["k_tum"] * B - k["k_te"] * T,
        k["k_r"] * B - k["k_rb"] * R,
    ], axis=1)


def _rk4_solve(k, n_form, n_steps):
    y0 = pt.as_tensor(np.tile([0.0, 1.0, 0.0, 0.0, 0.0], (n_form, 1)))

    # scan needs the parameters passed explicitly (non_sequences), not captured from the enclosing scope
    keys = list(RATE_KEYS)
    params = [pt.as_tensor(k[n]) for n in keys]

    def step(y, *vals):
        kk = dict(zip(keys, vals))
        k1 = _rhs(y, kk)
        k2 = _rhs(y + DT / 2 * k1, kk)
        k3 = _rhs(y + DT / 2 * k2, kk)
        k4 = _rhs(y + DT * k3, kk)
        return y + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    ys = pytensor.scan(step, outputs_info=y0, non_sequences=params, n_steps=n_steps, return_updates=False)
    return pt.concatenate([y0[None], ys], axis=0)  # (n_steps + 1, n_form, 5)


def build_model(rows, fixed_particle=False):
    """rows: output of simulate.load(). fixed_particle=True sets alpha = 0 (corona has no effect)."""
    forms = sorted({r["formulation"] for r in rows})
    d = np.array([next(r["d"] for r in rows if r["formulation"] == f) for f in forms])
    p = np.array([next(r["p"] for r in rows if r["formulation"] == f) for f in forms])
    f_idx = np.array([forms.index(r["formulation"]) for r in rows])
    t_idx = np.array([int(round(r["t"] / DT)) for r in rows])
    c_idx = np.array([COLS[r["compartment"]] for r in rows])
    y_obs = np.array([r["observed"] for r in rows])
    n_steps = int(t_idx.max())

    with pm.Model() as model:
        k = {}
        for name, (med, sd) in PRIORS.items():
            if fixed_particle and name == "alpha":
                k[name] = 0.0
                continue
            k[name] = pm.LogNormal(name, mu=np.log(med), sigma=sd)
        d_opt = pm.Normal("d_opt", 100.0, 40.0)          # optimum somewhere in the EPR range
        w = pm.LogNormal("w", np.log(50.0), 0.5)
        sigma = pm.HalfNormal("sigma", 0.5)              # measurement noise on the log scale

        k["k_on"] = k["k_on0"] * (1 - p)
        k["k_tum"] = k["k_tum0"] * pt.exp(-(((d - d_opt) / w) ** 2))

        ys = _rk4_solve(k, len(forms), n_steps)
        pred = ys[t_idx, f_idx, c_idx]
        pm.Deterministic("pred", pred)
        pm.Normal("log_obs", mu=pt.log(pt.maximum(pred, 1e-9)), sigma=sigma, observed=np.log(y_obs))
    return model


def fit(model, draws=1000, tune=1000, chains=4, seed=1):
    with model:
        return pm.sample(draws=draws, tune=tune, chains=chains, random_seed=seed,
                         nuts_sampler="numpyro", target_accept=0.9, idata_kwargs={"log_likelihood": True})

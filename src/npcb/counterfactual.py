"""Counterfactual design: the smallest change to a particle that makes it reach the tumour.

For one parameter set, solve
    min  (dd / S_D)^2 + (dp / S_P)^2
    s.t. tumour_AUC(d0 + dd, p0 + dp) >= factor * tumour_AUC(d0, p0)
Repeating this for every posterior draw gives a distribution of minimal changes instead of one answer.
"""

import numpy as np
from scipy.optimize import minimize

from .model import Params, tumour_auc

S_D = 50.0   # nm counted as "one unit" of change in diameter
S_P = 0.5    # PEG density counted as "one unit" of change
BOUNDS = [(20.0, 250.0), (0.0, 0.95)]  # what can realistically be manufactured


def minimal_change(prm: Params, d0, p0, factor=2.0):
    """Return (d_new, p_new, cost); cost is nan if no feasible design was found."""
    target = factor * tumour_auc(d0, p0, prm)
    cost = lambda x: ((x[0] - d0) / S_D) ** 2 + ((x[1] - p0) / S_P) ** 2
    con = {"type": "ineq", "fun": lambda x: tumour_auc(x[0], x[1], prm) / target - 1.0}
    best = None
    # a few starting points: the constraint is non-convex (size window), so avoid a single local answer
    for x0 in [(d0 - 30, p0), (d0, min(p0 + 0.3, 0.95)), (d0 - 60, min(p0 + 0.3, 0.95))]:
        r = minimize(cost, x0, method="SLSQP", bounds=BOUNDS, constraints=[con], options={"maxiter": 100})
        if r.success and con["fun"](r.x) > -1e-4 and (best is None or r.fun < best.fun):
            best = r
    if best is None:
        return np.nan, np.nan, np.nan
    return best.x[0], best.x[1], best.fun


def robust_change(draws, d0, p0, factor=2.0, prob=0.9):
    """Smallest change that meets the target under a fraction `prob` of the posterior draws.

    The minimal fix of a single parameter set sits exactly on the success boundary, so under
    uncertainty it works only ~50 % of the time. Here the constraint is on the (1 - prob) quantile
    of the improvement ratio across draws instead.
    """
    base = np.array([tumour_auc(d0, p0, prm) for prm in draws])

    def margin(x):
        ratio = np.array([tumour_auc(x[0], x[1], prm) for prm in draws]) / base
        return np.quantile(ratio, 1 - prob) / factor - 1.0

    cost = lambda x: ((x[0] - d0) / S_D) ** 2 + ((x[1] - p0) / S_P) ** 2
    r = minimize(cost, (d0 - 30, p0), method="SLSQP", bounds=BOUNDS,
                 constraints=[{"type": "ineq", "fun": margin}], options={"maxiter": 50})
    return r.x[0], r.x[1], r.fun


def success_probability(draws, d, p, d0, p0, factor=2.0):
    """Fraction of posterior draws under which design (d, p) meets that draw's own target."""
    ok = [tumour_auc(d, p, prm) >= factor * tumour_auc(d0, p0, prm) for prm in draws]
    return float(np.mean(ok))


def extrapolation_distance(d, p, tested):
    """Distance (in S_D / S_P units) from (d, p) to the nearest formulation that was actually measured."""
    return min(np.hypot((d - td) / S_D, (p - tp) / S_P) for td, tp in tested)

"""Counterfactual for the badly behaving particle (200 nm, PEG 0.2): what is the smallest fix, and can we trust it?

Three trust checks:
  (a) success probability of the recommended design across the full posterior
  (b) distance of the recommendation from any formulation that was actually measured (extrapolation)
  (c) the same analysis with the tumour data removed: how much does the answer depend on them?

Results -> data/counterfactual.json (summary) and data/counterfactual_draws.npz (minimal fix per draw).
"""

import json

import numpy as np
import xarray as xr

from npcb import Params
from npcb.counterfactual import extrapolation_distance, minimal_change, robust_change, success_probability
from npcb.inference import build_model, fit
from npcb.simulate import FORMULATIONS, load

D0, P0, FACTOR, N_DRAWS = 200.0, 0.2, 2.0, 200
NAMES = ["k_on0", "k_mps0", "alpha", "k_tum0", "d_opt", "w", "k_te", "k_r", "k_rb"]
rng = np.random.default_rng(0)


def draws_from(posterior, n=N_DRAWS):
    flat = {k: posterior[k].values.ravel() for k in NAMES}
    idx = rng.choice(len(flat["k_mps0"]), n, replace=False)
    return [Params(**{k: float(v[i]) for k, v in flat.items()}) for i in idx]


def analyse(draws, label):
    sols = np.array([minimal_change(prm, D0, P0, FACTOR) for prm in draws])
    ok = ~np.isnan(sols[:, 0])
    d_rec, p_rec = np.median(sols[ok, 0]), np.median(sols[ok, 1])
    res = dict(
        label=label,
        feasible=float(ok.mean()),
        d_rec=float(d_rec), p_rec=float(p_rec),
        d_94=np.percentile(sols[ok, 0], [3, 97]).tolist(),
        p_94=np.percentile(sols[ok, 1], [3, 97]).tolist(),
        success_prob=success_probability(draws, d_rec, p_rec, D0, P0, FACTOR),
        extrapolation=float(extrapolation_distance(d_rec, p_rec, FORMULATIONS)),
    )
    # robust fix: 90 % success probability, optimised on 50 draws and checked on all of them
    d_rob, p_rob, _ = robust_change(draws[:50], D0, P0, FACTOR, prob=0.9)
    res.update(d_robust=float(d_rob), p_robust=float(p_rob),
               success_prob_robust=success_probability(draws, d_rob, p_rob, D0, P0, FACTOR))
    print(json.dumps(res, indent=2))
    return sols, res


# full data
post = xr.open_datatree("data/posterior_evolving.nc")["posterior"]
sols_full, res_full = analyse(draws_from(post), "all data")

# ground truth answer (only possible because the data are synthetic)
truth = Params(**json.load(open("data/truth.json"))["params"])
d_true, p_true, _ = minimal_change(truth, D0, P0, FACTOR)
print(f"true minimal change: d = {d_true:.1f} nm, PEG = {p_true:.2f}")

# trust check (c): refit without tumour measurements
rows = [r for r in load() if r["compartment"] != "T"]
no_tumour = fit(build_model(rows))
sols_nt, res_nt = analyse(draws_from(no_tumour.posterior), "no tumour data")

json.dump({"all_data": res_full, "no_tumour_data": res_nt, "truth": [d_true, p_true],
           "baseline": [D0, P0], "factor": FACTOR}, open("data/counterfactual.json", "w"), indent=2)
np.savez("data/counterfactual_draws.npz", all_data=sols_full, no_tumour_data=sols_nt)

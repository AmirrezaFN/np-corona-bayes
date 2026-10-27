"""Calibrate the evolving-particle model on all six formulations; posterior -> data/posterior_evolving.nc."""

import json
import time

import arviz as az

from npcb.inference import build_model, fit
from npcb.simulate import load

rows = load()
truth = json.load(open("data/truth.json"))
true = dict(truth["params"], sigma=truth["sigma"])
names = ["k_on0", "k_mps0", "alpha", "k_tum0", "d_opt", "w", "k_te", "k_r", "k_rb", "sigma"]

t0 = time.time()
idata = fit(build_model(rows))
print(f"sampling took {time.time() - t0:.0f} s")
idata.to_netcdf("data/posterior_evolving.nc")

summary = az.summary(idata, var_names=names)
summary.insert(0, "true", [true[n] for n in names])
print(summary.to_string())
print("divergences:", int(idata.sample_stats["diverging"].sum()))

"""Fit the Bayesian ODE surrogate to my M.Sc. agent-based model output (1..6 subclones).

Posterior -> data/posterior_thesis.nc, posterior predictive -> data/pp_thesis.nc.
"""

import arviz as az
import numpy as np
import pymc as pm

from npcb.thesis import build_model, load_abm

day, N, T, A = load_abm()
model, obs_steps = build_model(day, N, T, A)
with model:
    idata = pm.sample(1000, tune=1500, chains=4, random_seed=1, nuts_sampler="numpyro", target_accept=0.95)
    pp = pm.sample_posterior_predictive(idata, random_seed=2)
idata.to_netcdf("data/posterior_thesis.nc")
pp.to_netcdf("data/pp_thesis.nc")

print(az.summary(idata, var_names=["r", "K", "kappa", "mu_a", "beta_a", "delta", "sigma_N", "sigma_A"]).to_string())
print("divergences:", int(idata.sample_stats["diverging"].sum()))
beta = idata.posterior["beta_a"].values.ravel()
print(f"P(beta_a < 0) = {np.mean(beta < 0):.3f}")
print(f"per extra subclone, activation per T cell changes by {100 * (np.exp(np.median(beta)) - 1):+.1f} % "
      f"(94 % interval {100 * (np.exp(np.percentile(beta, 3)) - 1):+.1f} to {100 * (np.exp(np.percentile(beta, 97)) - 1):+.1f} %)")

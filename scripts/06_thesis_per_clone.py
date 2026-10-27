"""Is the small subclone of run V17 controlled by T cells, or does it just grow slower?

Fits the per-subclone killing model twice: shared growth rate, and per-clone growth rates (partial pooling).
If kappa stays well above zero when each clone may have its own growth rate, the killing is identified.

Usage: python scripts/06_thesis_per_clone.py <simulator run Data folder>
Posteriors -> data/posterior_<run>_shared.nc, data/posterior_<run>_perclone.nc; data -> data/thesis_abm/<run>_per_clone.npz
"""

import sys

import arviz as az
import numpy as np
import pymc as pm

from npcb.thesis import build_clone_model, load_run_per_clone

run = sys.argv[1] if len(sys.argv) > 1 else r"G:\Uni\M.Sc\Semesters\Thesis\MATLAB\V17-Fallahnejad\Data"
tag = "V17" if "V17" in run else "V16" if "V16" in run else "run"
day, N, A = load_run_per_clone(run)
np.savez(f"data/thesis_abm/{tag}_per_clone.npz", day=day, N=N, A=A)  # so figures do not need the simulator folder

fits = {}
for label, per_clone in [("shared", False), ("perclone", True)]:
    model, obs_steps = build_clone_model(day, N, A, per_clone_growth=per_clone)
    with model:
        idata = pm.sample(1000, tune=2000, chains=4, random_seed=1, nuts_sampler="numpyro", target_accept=0.99,
                          idata_kwargs={"log_likelihood": True})
    idata.to_netcdf(f"data/posterior_{tag}_{label}.nc")
    fits[label] = idata
    names = ["r", "K", "kappa", "sigma"] + (["mu_r", "tau_r"] if per_clone else [])
    print(f"\n=== {tag}: {label} growth ===")
    print(az.summary(idata, var_names=names).to_string())
    print("divergences:", int(idata.sample_stats["diverging"].sum()))
    k = idata.posterior["kappa"].values.ravel()
    print(f"kappa median {np.median(k):.3f}, 94 % interval {np.percentile(k, 3):.3f} - {np.percentile(k, 97):.3f}")

print("\n", az.compare(fits).to_string())

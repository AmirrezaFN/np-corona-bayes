"""Does it matter that the particle changes? Fit a fixed-particle model (alpha = 0) to the same data and compare.

Posterior -> data/posterior_fixed.nc.
"""

import arviz as az
import xarray as xr

from npcb.inference import build_model, fit
from npcb.simulate import load

rows = load()
evolving = xr.open_datatree("data/posterior_evolving.nc")
fixed = fit(build_model(rows, fixed_particle=True))
fixed.to_netcdf("data/posterior_fixed.nc")

print("divergences (fixed):", int(fixed.sample_stats["diverging"].sum()))
print(az.compare({"evolving particle": evolving, "fixed particle": fixed}).to_string())

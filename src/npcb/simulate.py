"""Synthetic biodistribution data generated from the evolving-particle model with known parameters."""

import csv
import json
from pathlib import Path

import numpy as np

from .model import Params, solve

# (diameter [nm], PEG density) for each formulation
FORMULATIONS = [
    (30, 0.0),
    (100, 0.0),
    (100, 0.5),
    (100, 0.9),
    (60, 0.7),
    (200, 0.2),  # the "badly behaving" particle used later for the counterfactual
]
T_OBS = np.array([0.25, 1, 4, 8, 24, 48])  # hours
OBSERVED = {"B": 1, "L": 2, "T": 3}       # compartment -> column in the state array
SIGMA = 0.2                               # log-normal noise (~20 % measurement error)


def generate(prm=None, sigma=SIGMA, seed=1):
    """Return a list of rows (formulation, d, p, t, compartment, true, observed)."""
    prm = prm or Params()
    rng = np.random.default_rng(seed)
    rows = []
    for f, (d, p) in enumerate(FORMULATIONS):
        y = solve(d, p, prm, T_OBS)
        for name, j in OBSERVED.items():
            for t, true in zip(T_OBS, y[:, j]):
                obs = true * np.exp(sigma * rng.standard_normal())
                rows.append(dict(formulation=f, d=d, p=p, t=t, compartment=name, true=true, observed=obs))
    return rows


def save(rows, prm, out_dir="data"):
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    with open(out / "biodistribution.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["formulation", "d", "p", "t", "compartment", "observed"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})
    with open(out / "truth.json", "w") as fh:
        json.dump({"params": prm.as_dict(), "sigma": SIGMA, "formulations": FORMULATIONS}, fh, indent=2)


def load(path="data/biodistribution.csv"):
    with open(path) as fh:
        return [dict(r, formulation=int(r["formulation"]), d=float(r["d"]), p=float(r["p"]),
                     t=float(r["t"]), observed=float(r["observed"])) for r in csv.DictReader(fh)]

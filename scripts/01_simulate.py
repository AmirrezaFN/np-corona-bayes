"""Generate the synthetic dataset (data/biodistribution.csv) and the ground truth (data/truth.json)."""

from npcb import Params
from npcb.simulate import generate, save

prm = Params()
rows = generate(prm)
save(rows, prm)
print(f"{len(rows)} observations saved to data/")

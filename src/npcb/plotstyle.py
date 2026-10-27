"""Shared figure style: LaTeX look (Computer Modern via mathtext, no LaTeX install needed), colour-blind-safe colours."""

from pathlib import Path

import matplotlib as mpl

# Okabe-Ito palette (colour-blind safe)
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "purple": "#CC79A7",
     "sky": "#56B4E9", "yellow": "#F0E442", "black": "#000000", "grey": "#999999"}
CYCLE = [C[k] for k in ("blue", "orange", "green", "red", "purple", "sky")]

# parameter names as they appear in the equations
TEX = {
    "k_on0": r"$k_{\mathrm{on},0}$", "k_mps0": r"$k_{\mathrm{MPS},0}$", "alpha": r"$\alpha$",
    "k_tum0": r"$k_{\mathrm{tum},0}$", "d_opt": r"$d_{\mathrm{opt}}$", "w": r"$w$", "k_te": r"$k_{\mathrm{te}}$",
    "k_r": r"$k_{r}$", "k_rb": r"$k_{rb}$", "sigma": r"$\sigma$", "kappa": r"$\kappa$",
}


def use():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["cmr10"],              # Computer Modern, shipped with matplotlib
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,  # tick labels like $10^{4}$
        "axes.unicode_minus": False,          # cmr10 has no unicode minus glyph
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=CYCLE),
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.6,
        "legend.frameon": False,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def panel_labels(axes, x=-0.12, y=1.04):
    for i, ax in enumerate(axes):
        ax.text(x, y, f"({chr(97 + i)})", transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom")


def save(fig, name, folder="figures"):
    Path(folder).mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(Path(folder) / f"{name}.{ext}")

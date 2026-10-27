"""Draw every figure from saved results (no sampling). PNG + PDF in figures/.

Usage: python scripts/make_figures.py            all figures
       python scripts/make_figures.py 2 4        only figures 2 and 4
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from npcb import Params, solve
from npcb.plotstyle import C, CYCLE, TEX, panel_labels, save, use
from npcb.simulate import FORMULATIONS, OBSERVED, load
from npcb.thesis import EVERY, STEP0, STEP1, load_abm

use()
COMPARTMENT = {"B": "Blood", "L": "Liver/spleen (MPS)", "T": "Tumour"}
ODE_NAMES = ["k_on0", "k_mps0", "alpha", "k_tum0", "d_opt", "w", "k_te", "k_r", "k_rb"]
rng = np.random.default_rng(0)


def posterior(path):
    return xr.open_datatree(path)["posterior"]


def flat(post, name):
    return post[name].values.reshape(-1, *post[name].shape[2:])


def form_label(d, p):
    return rf"$d = {d:g}$ nm, $p = {p:g}$"


def truth():
    t = json.load(open("data/truth.json"))
    return dict(t["params"], sigma=t["sigma"])


def ode_bands(post, d, p, t, alpha_fixed=None, n=150):
    draws = {k: flat(post, k) for k in ODE_NAMES if k in post}
    sims = []
    for i in rng.choice(len(draws["k_mps0"]), n, replace=False):
        prm = {k: float(v[i]) for k, v in draws.items()}
        if alpha_fixed is not None:
            prm["alpha"] = alpha_fixed
        sims.append(solve(d, p, Params(**prm), t))
    return np.percentile(np.array(sims), [3, 50, 97], axis=0)


def log_hist(ax, x, bins, **kw):
    """Fraction of samples per log-spaced bin, i.e. the density in log(x)."""
    ax.hist(x, bins=bins, weights=np.full(len(x), 1 / len(x)), **kw)


# --------------------------------------------------------------------------------------------------------------------
def fig1():
    rows, prm = load(), Params()
    t = np.linspace(0, 48, 400)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for f, (d, p) in enumerate(FORMULATIONS):
        y = solve(d, p, prm, t)
        for ax, (name, j) in zip(axes, OBSERVED.items()):
            ax.plot(t, 100 * y[:, j], color=CYCLE[f], label=form_label(d, p))
            pts = [r for r in rows if r["formulation"] == f and r["compartment"] == name]
            ax.plot([r["t"] for r in pts], [100 * r["observed"] for r in pts], "o", ms=4, color=CYCLE[f])
    for ax, name in zip(axes, OBSERVED):
        ax.set_title(COMPARTMENT[name])
        ax.set_xlabel(r"time $t$ (h)")
        ax.set_xticks([0, 12, 24, 36, 48])
    axes[0].set_ylabel("injected dose (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.1), handlelength=1.5)
    panel_labels(axes)
    fig.tight_layout()
    save(fig, "01_synthetic_data")


def fig2():
    post, true = posterior("data/posterior_evolving.nc"), truth()
    names = ["k_on0", "k_mps0", "alpha", "k_tum0", "d_opt", "w", "k_te", "k_r", "k_rb", "sigma"]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for i, n in enumerate(names):
        lo, q1, mid, q3, hi = np.percentile(flat(post, n) / true[n], [3, 25, 50, 75, 97])
        ax.plot([lo, hi], [i, i], color=C["blue"], lw=1.4)
        ax.plot([q1, q3], [i, i], color=C["blue"], lw=4, solid_capstyle="butt")
        ax.plot(mid, i, "o", color="white", mec=C["blue"], mew=1.4, ms=6, zorder=3)
    ax.axvline(1, color=C["black"], ls="--", lw=0.9)
    ax.set_xscale("log")
    ax.set_xticks([0.6, 0.8, 1, 1.25, 1.5], [r"$0.6$", r"$0.8$", r"$1$", r"$1.25$", r"$1.5$"])
    ax.minorticks_off()
    ax.set_yticks(range(len(names)), [TEX[n] for n in names])
    ax.set_xlabel(r"posterior / true value")
    ax.set_title("Parameter recovery (thin: 94%, thick: 50% interval)", fontsize=11)
    fig.tight_layout()
    save(fig, "02_parameter_recovery")

    pair = ["k_mps0", "alpha", "k_on0"]
    n = len(pair)
    fig, axes = plt.subplots(n, n, figsize=(6.5, 6.5))
    for i, yi in enumerate(pair):
        for j, xj in enumerate(pair):
            ax = axes[i, j]
            x = flat(post, xj)
            if i == j:
                ax.hist(x, bins=40, color=C["blue"], alpha=0.75)
                ax.axvline(true[xj], color=C["red"], lw=1.5)
                ax.set_yticks([])
                ax.spines["left"].set_visible(False)
            elif i > j:
                ax.plot(x, flat(post, yi), ".", ms=1.5, alpha=0.25, color=C["blue"], rasterized=True)
                ax.plot(true[xj], true[yi], "x", color=C["red"], ms=9, mew=2)
                corr = np.corrcoef(x, flat(post, yi))[0, 1]
                ax.text(0.97, 0.95, rf"$\rho = {corr:.2f}$", transform=ax.transAxes, ha="right", va="top")
            else:
                ax.axis("off")
            if i == n - 1:
                ax.set_xlabel(TEX[xj])
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(TEX[yi])
            elif j > 0 and i != j:
                ax.set_yticklabels([])
    fig.text(0.62, 0.78, "posterior draws\n" r"$\times$  true value", color=C["black"], fontsize=11)
    fig.tight_layout()
    save(fig, "02_pair_kmps_alpha")


def fig3():
    rows = load()
    evolving, fixed = posterior("data/posterior_evolving.nc"), posterior("data/posterior_fixed.nc")
    t = np.linspace(0, 48, 200)
    show = [1, 3]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), sharex=True)
    for row, f in enumerate(show):
        d, p = FORMULATIONS[f]
        for label, post, a, color in [("evolving particle", evolving, None, C["blue"]),
                                      (r"fixed particle ($\alpha = 0$)", fixed, 0.0, C["red"])]:
            lo, mid, hi = ode_bands(post, d, p, t, alpha_fixed=a)
            for ax, (name, j) in zip(axes[row], OBSERVED.items()):
                ax.fill_between(t, 100 * lo[:, j], 100 * hi[:, j], color=color, alpha=0.2, lw=0)
                ax.plot(t, 100 * mid[:, j], color=color, label=label)
        for ax, name in zip(axes[row], OBSERVED):
            pts = [r for r in rows if r["formulation"] == f and r["compartment"] == name]
            ax.plot([r["t"] for r in pts], [100 * r["observed"] for r in pts], "o", ms=4.5, color=C["black"],
                    zorder=3, label="data" if name == "B" else None)
            if row == 0:
                ax.set_title(COMPARTMENT[name])
        axes[row, 0].set_ylabel("injected dose (%)")
        axes[row, 0].text(-0.38, 0.5, form_label(d, p), transform=axes[row, 0].transAxes, rotation=90,
                          va="center", ha="center", fontsize=12)
    for ax in axes[-1]:
        ax.set_xlabel(r"time $t$ (h)")
        ax.set_xticks([0, 12, 24, 36, 48])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    panel_labels(axes.ravel(), x=-0.16)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "03_fixed_vs_evolving")


def fig4():
    res = json.load(open("data/counterfactual.json"))
    sols = np.load("data/counterfactual_draws.npz")
    d0, p0 = res["baseline"]
    d_true, p_true = res["truth"]
    full, nt = res["all_data"], res["no_tumour_data"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, zoom in zip(axes, [False, True]):
        ax.plot(sols["no_tumour_data"][:, 0], sols["no_tumour_data"][:, 1], "o", ms=3.5, alpha=0.45,
                color=C["orange"], mew=0, label="minimal fix per draw, no tumour data")
        ax.plot(sols["all_data"][:, 0], sols["all_data"][:, 1], "o", ms=3.5, color=C["blue"], mew=0, zorder=6,
                label="minimal fix per draw, all data")
        ax.plot(d0, p0, "s", ms=9, color=C["black"], label=rf"original particle ($d = {d0:g}$ nm, $p = {p0:g}$)")
        ax.plot(full["d_robust"], full["p_robust"], "D", ms=10, mfc="none", mec=C["green"], mew=2, zorder=5,
                label=rf"robust fix, all data ($P_{{\mathrm{{success}}}} = {full['success_prob_robust']:.2f}$)")
        ax.plot(nt["d_robust"], nt["p_robust"], "D", ms=10, mfc="none", mec=C["red"], mew=2, zorder=5,
                label=rf"robust fix, no tumour data ($P_{{\mathrm{{success}}}} = {nt['success_prob_robust']:.2f}$)")
        ax.plot(d_true, p_true, "*", ms=15, color=C["black"], mec="white", mew=0.6, zorder=7, label="true minimal fix")
        ax.set_xlabel(r"diameter $d$ (nm)")
        ax.set_ylabel(r"PEG density $p$")
        if zoom:
            ax.set_xlim(170, 202)
            ax.set_ylim(0.195, 0.25)
            ax.set_title("Zoom near the original particle")
            ax.annotate("all-data fixes\n(tight cluster)", xy=(187.5, 0.2075), xytext=(189, 0.232), fontsize=10,
                        color=C["blue"], arrowprops=dict(arrowstyle="->", color=C["blue"], lw=1))
        else:
            for d, p in FORMULATIONS:
                ax.plot(d, p, "+", ms=10, mew=1.5, color=C["grey"])
            ax.plot([], [], "+", ms=10, mew=1.5, color=C["grey"], label="measured formulations")
            ax.set_xlim(20, 250)
            ax.set_ylim(-0.03, 1.0)
            ax.set_title(r"Smallest change giving $2\times$ tumour exposure")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.13))
    panel_labels(axes)
    fig.tight_layout()
    save(fig, "04_counterfactual")


def fig5():
    day, N, T, A = load_abm()
    post = posterior("data/posterior_thesis.nc")
    pp = xr.open_datatree("data/pp_thesis.nc")["posterior_predictive"]
    obs = np.arange(STEP0, STEP1 + 1, EVERY)
    t_obs, win = day[obs], slice(STEP0, STEP1 + 1)
    colors = plt.cm.viridis(np.linspace(0, 0.9, 6))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
    for j in range(6):
        for ax, key, data in [(axes[0], "log_N", N), (axes[1], "log_A", A)]:
            sims = np.exp(pp[key].values.reshape(-1, len(obs), 6)[:, :, j])
            lo, hi = np.percentile(sims, [3, 97], axis=0)
            ax.fill_between(t_obs, lo, hi, color=colors[j], alpha=0.2, lw=0)
            ax.plot(day[win], data[win, j], color=colors[j], lw=1.3, label=rf"$n = {j + 1}$")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"tumour cells $N$")
    axes[0].set_title("Tumour cells")
    axes[1].set_ylabel(r"activated T cells $A$")
    axes[1].set_title("Activated T cells")
    for ax in axes[:2]:
        ax.set_xlabel(r"time $t$ (day)")
    axes[0].legend(title="subclones", ncol=2, fontsize=9, title_fontsize=10, loc="lower right")

    a = flat(post, "a")
    lo, mid, hi = np.percentile(a, [3, 50, 97], axis=0)
    n = np.arange(1, 7)
    axes[2].errorbar(n, 1e3 * mid, yerr=[1e3 * (mid - lo), 1e3 * (hi - mid)], fmt="o", color=C["blue"],
                     capsize=4, mfc="white", mew=1.4)
    beta = flat(post, "beta_a")
    pct = 100 * (np.exp(np.percentile(beta, [50, 3, 97])) - 1)
    axes[2].text(0.97, 0.95, rf"$\beta_a$: ${pct[0]:+.1f}\%$ per subclone" "\n"
                 rf"94% interval $[{pct[1]:+.1f}, {pct[2]:+.1f}]\%$",
                 transform=axes[2].transAxes, ha="right", va="top", fontsize=11)
    axes[2].set_xticks(n)
    axes[2].set_xlabel(r"number of subclones $n$")
    axes[2].set_ylabel(r"activation rate $a_n$ ($10^{-3}$ day$^{-1}$)")
    axes[2].set_title("Per-cell activation vs heterogeneity")
    panel_labels(axes)
    fig.text(0.5, -0.03, "Lines: agent-based simulator output.  Bands: 94% posterior predictive interval of the "
             "surrogate.", ha="center", fontsize=10)
    fig.tight_layout()
    save(fig, "05_thesis_surrogate")

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    bins = np.logspace(-2, 2, 60)
    log_hist(ax, rng.lognormal(0.0, 1.0, 40000), bins, color=C["grey"], alpha=0.5, label="prior")
    log_hist(ax, flat(post, "kappa"), bins, color=C["blue"], alpha=0.75, label="posterior")
    ax.set_xscale("log")
    ax.set_xlabel(r"killing rate $\kappa$ (kills per activated T cell per day)")
    ax.set_ylabel("fraction of samples")
    ax.set_title(r"$\kappa$ from bulk counts: upper bound only")
    ax.legend()
    fig.tight_layout()
    save(fig, "05_kappa_prior_vs_posterior")


def fig6(tag):
    data = np.load(f"data/thesis_abm/{tag}_per_clone.npz")
    day, N = data["day"], data["N"]
    fits = {k: posterior(f"data/posterior_{tag}_{k}.nc") for k in ("shared", "perclone")}
    obs = np.arange(STEP0, STEP1 + 1, EVERY)
    n_cl = N.shape[1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1.25, 1]})
    for j in range(n_cl):
        axes[0].plot(day[STEP0:STEP1 + 1], N[STEP0:STEP1 + 1, j], color=CYCLE[j], lw=5, alpha=0.3)
        for k, ls in [("shared", (0, (4, 2))), ("perclone", "-")]:
            pred = np.median(flat(fits[k], "N_pred")[:, :, j], axis=0)
            axes[0].plot(day[obs], pred, ls=ls, color=CYCLE[j], lw=1.4)
    # end-of-line labels, pushed apart in log space so they do not overlap
    order = np.argsort(N[STEP1])
    ys = np.log10(N[STEP1, order])
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + 0.09)
    for j, y in zip(order, ys):
        axes[0].text(day[STEP1] + 0.15, 10 ** y, rf"clone {j + 1}", color=CYCLE[j], va="center", fontsize=11)
    axes[0].plot([], [], color=C["grey"], lw=5, alpha=0.4, label="simulator")
    axes[0].plot([], [], color=C["grey"], ls=(0, (4, 2)), label=r"surrogate, shared $r$")
    axes[0].plot([], [], color=C["grey"], label=r"surrogate, clone-specific $r_k$")
    axes[0].set_yscale("log")
    axes[0].set_xlim(day[STEP0] - 0.2, day[STEP1] + 1.6)
    axes[0].set_xlabel(r"time $t$ (day)")
    axes[0].set_ylabel(r"tumour cells per subclone $N_k$")
    axes[0].set_title(f"Run {tag}: {n_cl} subclones")
    axes[0].legend(loc="lower right")

    bins = np.logspace(-2, 1.5, 180)  # fine bins: the V17 posterior is narrow
    log_hist(axes[1], rng.lognormal(0.0, 1.5, 40000), bins, color=C["grey"], alpha=0.4, label="prior")
    for k, c, lab in [("shared", C["blue"], r"posterior, shared $r$"),
                      ("perclone", C["red"], r"posterior, clone-specific $r_k$")]:
        log_hist(axes[1], flat(fits[k], "kappa"), bins, color=c, alpha=0.6, label=lab)
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"killing rate $\kappa$ (kills per activated T cell per day)")
    axes[1].set_ylabel("fraction of samples")
    axes[1].set_title(r"Is $\kappa$ identified?")
    axes[1].legend(loc="upper left")
    panel_labels(axes)
    fig.tight_layout()
    save(fig, f"06_per_clone_{tag}")


FIGS = {"1": fig1, "2": fig2, "3": fig3, "4": fig4, "5": fig5,
        "6": lambda: (fig6("V17"), fig6("V16"))}

if __name__ == "__main__":
    for k in sys.argv[1:] or FIGS:
        FIGS[k]()
        print(f"figure {k} done")

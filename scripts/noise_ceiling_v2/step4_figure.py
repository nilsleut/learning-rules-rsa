"""Step 4 figure: per-subject model rho with 95% bootstrap CIs against the
leave-one-out lower bound (band = its 95% CI). The upper bound is off-scale and
uninformative at N=3; it is marked at the panel edge with its permutation null.

Reads step1_bounds.json and step3c_summary.csv only.
Output: results/noise_ceiling_v2/noise_ceiling_v2.png (+ .pdf)
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import RESULTS, ROIS, ROI_LAYER

RULES = [("random_weights", "Random"), ("backprop", "BP"), ("feedback_alignment", "FA"),
         ("predictive_coding", "PC"), ("stdp", "STDP")]
SETS = [("original", "Table 2 run (27.04.)", "#2a78d6", "o"),
        ("bnfix", "BN-fixed rerun (08.08.)", "#eb6834", "D")]
INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#c9c8c2"


def main():
    s1 = json.loads((RESULTS / "step1_bounds.json").read_text())["rois"]
    sm = pd.read_csv(RESULTS / "step3c_summary.csv").fillna("")
    q = lambda quant, roi, st="", rule="": sm[(sm.quantity == quant) & (sm.roi == roi) &
                                              (sm.rdm_set == st) & (sm.rule == rule)].iloc[0]
    xmax = 1.15 * max(sm[sm.quantity.isin(["persub", "lower"])].ci_hi.max(), 0.07)
    xmin = min(sm[sm.quantity == "persub"].ci_lo.min(), 0) - 0.004

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK})
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.4), sharey=True, sharex=True)
    for ax, roi in zip(axes, ROIS):
        lo = q("lower", roi)
        ax.axvspan(lo.ci_lo, lo.ci_hi, color=BAND, alpha=0.55, lw=0, zorder=0)
        ax.axvline(lo.point, color=INK2, lw=1.2, zorder=1)
        ax.axvline(0, color=GRID, lw=1, zorder=0)
        for i, (rk, rn) in enumerate(RULES):
            for j, (st, _, col, mk) in enumerate(SETS):
                r = q("persub", roi, st, rk)
                y = i + (-0.14 if j == 0 else 0.14)
                ax.plot([r.ci_lo, r.ci_hi], [y, y], color=col, lw=2, solid_capstyle="round",
                        zorder=3)
                ax.plot(r.point, y, mk, ms=6, mfc=col if j == 0 else "white", mec=col,
                        mew=1.6, zorder=4)
        u = s1[roi]
        ax.text(0.985, 0.02, f"upper {u['upper']:.3f} → off-scale\n"
                f"(H0 {u['upper_null_mean']:.3f}; not informative, N=3)",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color=INK2)
        ax.axvline(xmax, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.set_title(f"{roi}  ({ROI_LAYER[roi]})", color=INK, fontsize=10, loc="left")
        ax.set_xlim(xmin, xmax)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_xlabel("Spearman ρ, per subject (mean of 3)")
    axes[0].set_yticks(range(len(RULES)))
    axes[0].set_yticklabels([rn for _, rn in RULES])
    axes[0].set_ylim(len(RULES) + 0.35, -0.7)

    handles = [plt.Line2D([], [], color=c, marker=m, mfc=c if k == 0 else "white", mec=c,
                          lw=2, label=lab) for k, (_, lab, c, m) in enumerate(SETS)]
    handles += [plt.Rectangle((0, 0), 1, 1, color=BAND, alpha=0.55,
                              label="LOO lower bound, 95% CI (line = estimate)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Model–brain RSA vs. leave-one-subject-out lower bound "
                 "(stimulus bootstrap, 1000×)", x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"noise_ceiling_v2.{ext}", dpi=200, facecolor="#fcfcfb")
    print("saved")


if __name__ == "__main__":
    main()

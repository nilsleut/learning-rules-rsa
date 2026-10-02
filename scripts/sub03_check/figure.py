"""sub03_check.png: (A) luminance rho per subject x ROI, (B) untrained-network rho per
subject x ROI (mean +- SD over 5 seeds), (C) trial-space shift test at V1 with the
1000-relabelling null, (D) sub-01 vs sub-02 on all pairs vs between-run pairs only.
Reads results/sub03_check/*.csv/json only.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_sub03 import RESULTS, ROIS, SUBJECTS

COL = {"sub-01": "#2a78d6", "sub-02": "#eb6834", "sub-03": "#1baf7a"}
MK = {"sub-01": "o", "sub-02": "s", "sub-03": "^"}
INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#c9c8c2"


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=9.5, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.axhline(0, color=INK2, lw=0.8)


def main():
    c2 = pd.read_csv(RESULTS / "check2_luminance.csv")
    c3 = pd.read_csv(RESULTS / "check3_models.csv")
    sh = pd.read_csv(RESULTS / "check5_shift_trial.csv")
    nul = json.loads((RESULTS / "check567.json").read_text())["null_V1"]
    c8 = json.loads((RESULTS / "check8.json").read_text())["restricted_pairs"]

    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK})
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.5), gridspec_kw={"width_ratios": [1, 1, 1.25, 0.9]})
    x = np.arange(len(ROIS))
    off = {s: d for s, d in zip(SUBJECTS, (-0.18, 0, 0.18))}

    for s in SUBJECTS:
        y = [c2[(c2.roi == r) & (c2.target == s)].spearman.iloc[0] for r in ROIS]
        ax[0].plot(x + off[s], y, MK[s], color=COL[s], ms=7, label=s, ls="none")
        m = [c3[(c3.rule == "random_weights") & (c3.roi == r) & (c3.subject == s)].iloc[0] for r in ROIS]
        ax[1].errorbar(x + off[s], [v["mean"] for v in m], yerr=[v.sd for v in m], fmt=MK[s],
                       color=COL[s], ms=7, lw=2, capsize=0, label=s)
    for a, t in [(ax[0], "A  Luminance vs. subject"),
                 (ax[1], "B  Untrained CNN vs. subject")]:
        a.set_xticks(x); a.set_xticklabels(ROIS); a.set_ylabel("Spearman ρ"); style(a, t)
    ax[0].legend(frameon=False, fontsize=7.5, loc="upper right")

    v = sh[sh.roi == "V1"]
    a = ax[2]
    a.axhspan(nul["rho_vs_luminance"]["q025"], nul["rho_vs_luminance"]["q975"], color=BAND,
              alpha=0.6, lw=0, label="sub-03 relabelled, 95% (null)")
    lines = [("sub-03", "rho_vs_luminance", "-", "sub-03 vs luminance"),
             ("sub-03", "rho_vs_sub01", "--", "sub-03 vs sub-01"),
             ("sub-02", "rho_vs_sub01", ":", "sub-02 vs sub-01 (control)")]
    for s, c, ls, lab in lines:
        d = v[v.subject == s].sort_values("k")
        a.plot(d.k, d[c], ls, color=COL[s], lw=2, marker=MK[s], ms=5, label=lab)
    a.set_xlabel("trial shift k within run"); a.set_ylabel("Spearman ρ (V1)")
    a.set_xticks(range(-5, 6)); style(a, "C  Shift test, V1")
    a.legend(frameon=False, fontsize=7, loc="upper left")

    a = ax[3]
    allp = [c8[r]["sub01_sub02_all_pairs"] for r in ROIS]
    btw = [c8[r]["sub01_sub02_between_run_pairs"] for r in ROIS]
    a.bar(x - 0.18, allp, 0.34, color=COL["sub-01"], label="all pairs")
    a.bar(x + 0.18, btw, 0.34, color="white", edgecolor=COL["sub-01"], lw=1.6,
          label="different-run pairs only")
    a.set_xticks(x); a.set_xticklabels(ROIS); a.set_ylabel("Spearman ρ")
    style(a, "D  sub-01 vs sub-02")
    a.legend(frameon=False, fontsize=7, loc="upper right")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"sub03_check.{ext}", dpi=200, facecolor="#fcfcfb")
    print("saved")


if __name__ == "__main__":
    main()

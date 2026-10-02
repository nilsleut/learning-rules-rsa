"""Figures for paper v3 (new file names; the v2 figures are not touched).

  rsa_comparison_cnn_v3.png   per-subject cross-run rho (bnfix) with stimulus-bootstrap CIs,
                              Nili lower bound as band (point + 95% CI); upper bound not shown
  subject_consistency_v3.png  single-subject cross-run rho with bootstrap CIs
  seed_variability_v3.png     per-seed cross-run rho (dots) and seed mean
  hierarchy_cnn_v3.png        cross-run rho along the fixed layer->ROI mapping
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PAPER = REPO / "paper" / "arxiv_upload_learning_rules_v3"
CR, P3 = REPO / "results" / "crossrun", REPO / "results" / "paper_v3"
RULES = ["random_weights", "backprop", "feedback_alignment", "predictive_coding", "stdp"]
NAME = {"random_weights": "Random (untrained)", "backprop": "BP", "feedback_alignment": "FA",
        "predictive_coding": "PC", "stdp": "STDP"}
# Random = neutral baseline; trained rules: Okabe-Ito, fixed order (validated, light mode)
COL = {"random_weights": "#8C8C8C", "backprop": "#0072B2", "feedback_alignment": "#E69F00",
       "predictive_coding": "#009E73", "stdp": "#CC79A7"}
ROIS = ["V1", "V2", "LOC", "IT"]
INK, MUTED = "#222222", "#666666"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False})


def bars(ax, means, lo, hi, x0, w):
    for i, r in enumerate(RULES):
        x = x0 + (i - 2) * w
        ax.bar(x, means[r], w * 0.86, color=COL[r], edgecolor="white", linewidth=1,
               label=NAME[r])
        ax.errorbar(x, means[r], yerr=[[means[r] - lo[r]], [hi[r] - means[r]]], fmt="none",
                    ecolor=INK, elinewidth=0.8, capsize=2)


def fig_main():
    s = pd.read_csv(CR / "step2_summary.csv").set_index("key")
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    w = 0.15
    for j, roi in enumerate(ROIS):
        lb = s.loc[f"lower|cr|{roi}"]
        ax.add_patch(plt.Rectangle((j - 0.42, lb.ci_lo), 0.84, lb.ci_hi - lb.ci_lo,
                                   color="#D9D9D9", zorder=0, lw=0))
        ax.plot([j - 0.42, j + 0.42], [lb.point] * 2, color=MUTED, lw=1.2, zorder=1)
        g = {r: s.loc[f"persub|cr|bnfix|{r}|{roi}"] for r in RULES}
        bars(ax, {r: g[r].point for r in RULES}, {r: g[r].ci_lo for r in RULES},
             {r: g[r].ci_hi for r in RULES}, j, w)
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(range(4)); ax.set_xticklabels(ROIS)
    ax.set_ylabel("Spearman $\\rho$ (per subject, cross-run)")
    h, l = ax.get_legend_handles_labels()
    band = plt.Rectangle((0, 0), 1, 1, color="#D9D9D9")
    ax.legend(h[:5] + [band], l[:5] + ["Nili lower bound (line) and 95% CI (band)"],
              ncol=6, fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(PAPER / "rsa_comparison_cnn_v3.png", dpi=200)
    plt.close(fig)


def fig_subjects():
    s = pd.read_csv(P3 / "subject_summary.csv").set_index(["subject", "rule", "roi"])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
    for ax, sub in zip(axes, ["sub-01", "sub-02", "sub-03"]):
        for j, roi in enumerate(ROIS):
            g = {r: s.loc[(sub, r, roi)] for r in RULES}
            bars(ax, {r: g[r].point for r in RULES}, {r: g[r].ci_lo for r in RULES},
                 {r: g[r].ci_hi for r in RULES}, j, 0.15)
        ax.axhline(0, color=INK, lw=0.6)
        ax.set_xticks(range(4)); ax.set_xticklabels(ROIS); ax.set_title(sub, fontsize=9)
    axes[0].set_ylabel("Spearman $\\rho$ (cross-run)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h[:5], l[:5], ncol=5, fontsize=7.5, frameon=False, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(PAPER / "subject_consistency_v3.png", dpi=200)
    plt.close(fig)


def fig_seeds():
    d = pd.read_csv(P3 / "seed_per_seed.csv")
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    w = 0.15
    for j, roi in enumerate(ROIS):
        for i, r in enumerate(RULES):
            v = d[(d.rule == r) & (d.roi == roi)].persub.values
            x = j + (i - 2) * w
            ax.scatter(np.full(len(v), x), v, s=14, color=COL[r], edgecolor="white", lw=0.6,
                       zorder=3, label=NAME[r] if j == 0 else None)
            ax.plot([x - w * 0.35, x + w * 0.35], [v.mean()] * 2, color=INK, lw=1, zorder=4)
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(range(4)); ax.set_xticklabels(ROIS)
    ax.set_ylabel("Spearman $\\rho$ (per subject, cross-run)")
    ax.legend(ncol=5, fontsize=7.5, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(PAPER / "seed_variability_v3.png", dpi=200)
    plt.close(fig)


def fig_hierarchy():
    s = pd.read_csv(CR / "step2_summary.csv").set_index("key")
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    x = np.arange(4)
    for r in RULES:
        y = [s.loc[f"persub|cr|bnfix|{r}|{roi}"].point for roi in ROIS]
        ax.plot(x, y, "o-", color=COL[r], lw=1.6, ms=5, label=NAME[r])
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(["Conv1→V1", "Conv1→V2", "Conv3→LOC", "FC1→IT"],
                                          fontsize=7.5)
    ax.set_ylabel("Spearman $\\rho$ (per subject, cross-run)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(PAPER / "hierarchy_cnn_v3.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    PAPER.mkdir(parents=True, exist_ok=True)
    fig_main(); fig_subjects(); fig_seeds(); fig_hierarchy()
    print("figures written")

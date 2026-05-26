"""
make_figures.py
===============
Publication-ready figures for the training dynamics study.
Saves directly to the current folder as PDF (300 Dpi).

  py make_figures.py
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import product

# ── Config ─────────────────────────────────────────────────────────────────────
DATA  = Path(__file__).parent / "training_dynamics_outputs" / "outputs" / \
        "training_dynamics_results.csv"
OUT   = Path(__file__).parent          # save figures here

ROIS  = ["V1", "V2", "V3", "V4", "LOC", "IT"]
ROI_LABELS = {"V1": "V1", "V2": "V2", "V3": "V3", "V4": "V4",
              "LOC": "LOC", "IT": "IT"}

RULE_ORDER = ["Random Weights", "Backprop", "Feedback Alignment",
              "Predictive Coding", "STDP"]
RULE_SHORT = {"Random Weights": "Rand.", "Backprop": "BP",
              "Feedback Alignment": "FA", "Predictive Coding": "PC", "STDP": "STDP"}

COLORS = {
    "Random Weights":     "#888888",
    "Backprop":           "#2E86AB",
    "Feedback Alignment": "#E84855",
    "Predictive Coding":  "#3BB273",
    "STDP":               "#F4A261",
}

COMPARE_EPOCH = 40
MM = 1 / 25.4

plt.rcParams.update({
    "font.family":         "sans-serif",
    "font.sans-serif":     ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":           10,
    "axes.labelsize":      11,
    "axes.titlesize":      12,
    "legend.fontsize":     9,
    "xtick.labelsize":     9,
    "ytick.labelsize":     9,
    "figure.dpi":          300,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "pdf.fonttype":        42,
    "ps.fonttype":         42,
})


# ══════════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load():
    df = pd.read_csv(str(DATA))
    print(f"Loaded {len(df)} rows — rules: {df['rule'].unique().tolist()}")
    return df


def best_layer(df, rule, roi, epoch):
    sub = df[(df["rule"] == rule) & (df["roi"] == roi) & (df["epoch"] == epoch)]
    if sub.empty:
        return "Conv1"
    return sub.groupby("layer")["rho"].mean().idxmax()


def seed_vals(df, rule, roi, epoch, layer=None):
    """Mean rho per seed (averaged across subjects)."""
    if layer is None:
        layer = best_layer(df, rule, roi, epoch)
    sub = df[(df["rule"] == rule) & (df["roi"] == roi)
             & (df["epoch"] == epoch) & (df["layer"] == layer)]
    return sub.groupby("seed_idx")["rho"].mean().values


def curve(df, rule, roi):
    """epochs, means, SEMs — best layer at each epoch."""
    epochs = sorted(df[(df["rule"] == rule) & (df["roi"] == roi)]["epoch"].unique())
    means, sems = [], []
    for ep in epochs:
        v = seed_vals(df, rule, roi, ep)
        means.append(v.mean())
        sems.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
    return np.array(epochs), np.array(means), np.array(sems)


def sign_test_onesided(a, b):
    """One-sided sign test: P(a > b), exact. Returns p-value."""
    diffs = np.array(a) - np.array(b)
    n_pos = (diffs > 0).sum()
    n     = len(diffs)
    # Exact binomial: p = P(X >= n_pos) under H0(p=0.5)
    from math import comb
    p = sum(comb(n, k) for k in range(n_pos, n + 1)) / 2 ** n
    return float(p)


def sig_star(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "†"
    return "ns"


# ══════════════════════════════════════════════════════════════════════════════
# SHARED AXIS SETUP
# ══════════════════════════════════════════════════════════════════════════════

EPOCH_TICKS = [0, 1, 2, 5, 10, 20, 30, 40]

def set_epoch_axis(ax, epochs):
    """Symlog x-axis with clean epoch tick labels."""
    ax.set_xscale("symlog", linthresh=1)
    ticks = [e for e in EPOCH_TICKS if e in epochs or e == 0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlim(-0.3, max(epochs) * 1.05)


def add_panel_label(ax, label):
    ax.text(-0.15, 1.05, label, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", ha="left")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — V1 dynamics (3 panels)
# ══════════════════════════════════════════════════════════════════════════════

def fig1(df):
    roi     = "V1"
    trained = [r for r in RULE_ORDER if r != "Random Weights" and r in df["rule"].values]
    ep_cmp  = COMPARE_EPOCH
    rw_ep0  = seed_vals(df, "Random Weights", roi, 0).mean()

    fig, axes = plt.subplots(1, 3, figsize=(180 * MM, 75 * MM),
                             constrained_layout=True)
    ax_a, ax_b, ax_c = axes

    # ── Panel A: V1 alignment across training ─────────────────────────────────
    all_epochs = sorted(df[df["roi"] == roi]["epoch"].unique())

    # Untrained reference line
    ax_a.axhline(rw_ep0, color=COLORS["Random Weights"], lw=1.2,
                 ls="--", alpha=0.8, zorder=1, label="Untrained (Rand.)")
    # Vertical dashed line at epoch 0
    ax_a.axvline(0, color="#aaaaaa", lw=0.8, ls=":", zorder=0)
    ax_a.text(0.22, rw_ep0 + 0.003, "Untrained", fontsize=8,
              color=COLORS["Random Weights"], va="bottom")

    for rule in trained:
        eps, means, sems = curve(df, rule, roi)
        ax_a.plot(eps, means, "o-", color=COLORS[rule], lw=1.8, ms=4,
                  label=RULE_SHORT[rule], zorder=3)
        ax_a.fill_between(eps, means - sems, means + sems,
                           color=COLORS[rule], alpha=0.15, zorder=2)

    set_epoch_axis(ax_a, all_epochs)
    ax_a.set_xlabel("Training epoch")
    ax_a.set_ylabel("Spearman r (best layer)")
    ax_a.set_title("V1 brain alignment", pad=6)
    ax_a.legend(frameon=False, loc="upper right", handlelength=1.5)
    add_panel_label(ax_a, "A")

    # ── Panel B: delta rho (normalised to epoch 0) ────────────────────────────
    ax_b.axhline(0, color="#aaaaaa", lw=1.0, ls="--", zorder=1)

    for rule in trained:
        eps, means, sems = curve(df, rule, roi)
        baseline = means[eps == 0][0] if 0 in eps else means[0]
        deltas   = means - baseline
        ax_b.plot(eps, deltas, "o-", color=COLORS[rule], lw=1.8, ms=4,
                  label=RULE_SHORT[rule], zorder=3)
        ax_b.fill_between(eps, deltas - sems, deltas + sems,
                           color=COLORS[rule], alpha=0.15, zorder=2)

    set_epoch_axis(ax_b, all_epochs)
    ax_b.set_xlabel("Training epoch")
    ax_b.set_ylabel("Δr  (relative to epoch 0)")
    ax_b.set_title("Change from untrained", pad=6)
    add_panel_label(ax_b, "B")

    # ── Panel C: bar chart at epoch 40 ────────────────────────────────────────
    bar_rules  = [r for r in RULE_ORDER if r in df["rule"].values]
    bar_vals   = []
    bar_sems   = []
    bar_colors = []
    bar_seeds  = []

    for rule in bar_rules:
        ep_use = ep_cmp if rule != "Random Weights" else 0
        v = seed_vals(df, rule, roi, ep_use)
        bar_vals.append(v.mean())
        bar_sems.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
        bar_colors.append(COLORS[rule])
        bar_seeds.append(v)

    x = np.arange(len(bar_rules))
    bars = ax_c.bar(x, bar_vals, 0.6, color=bar_colors, alpha=0.88,
                    yerr=bar_sems, capsize=3,
                    error_kw={"lw": 1.2, "ecolor": "black", "capthick": 1.2},
                    zorder=3)
    ax_c.axhline(0, color="black", lw=0.6)

    # Individual seed points
    rng = np.random.default_rng(0)
    for i, (rule, v) in enumerate(zip(bar_rules, bar_seeds)):
        jitter = rng.uniform(-0.08, 0.08, size=len(v))
        ax_c.scatter(x[i] + jitter, v, s=12, color=COLORS[rule],
                     edgecolors="white", linewidths=0.4, zorder=4, alpha=0.85)

    # Significance brackets: PC/STDP vs BP (one-sided: PC/STDP > BP?)
    bp_idx = bar_rules.index("Backprop")
    bp_v   = bar_seeds[bp_idx]
    top    = max(v + s for v, s in zip(bar_vals, bar_sems)) + 0.008
    step   = 0.014

    for i, rule in enumerate(bar_rules):
        if rule in ("Backprop", "Random Weights"):
            continue
        p = sign_test_onesided(bar_seeds[i], bp_v)
        s = sig_star(p)
        if s == "ns":
            continue
        y0 = top + step * abs(i - bp_idx) * 0.3
        ax_c.plot([bp_idx, bp_idx, i, i],
                   [y0, y0 + 0.004, y0 + 0.004, y0], lw=0.9, color="black")
        ax_c.text((bp_idx + i) / 2, y0 + 0.005, s,
                   ha="center", va="bottom", fontsize=9)
        top = y0 + 0.012

    ax_c.set_xticks(x)
    ax_c.set_xticklabels([RULE_SHORT[r] for r in bar_rules], rotation=30, ha="right")
    ax_c.set_ylabel("Spearman r (best layer)")
    ax_c.set_title(f"V1 at epoch {ep_cmp}", pad=6)
    add_panel_label(ax_c, "C")

    fig.savefig(str(OUT / "figure1_v1_dynamics.pdf"))
    fig.savefig(str(OUT / "figure1_v1_dynamics.png"), dpi=200)
    plt.close()
    print("Saved: figure1_v1_dynamics.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — All ROIs (6-panel, shared y-axis)
# ══════════════════════════════════════════════════════════════════════════════

def fig2(df):
    trained    = [r for r in RULE_ORDER if r != "Random Weights" and r in df["rule"].values]
    rois       = [r for r in ROIS if r in df["roi"].values]
    all_epochs = sorted(df["epoch"].unique())

    fig, axes = plt.subplots(2, 3, figsize=(180 * MM, 120 * MM),
                             constrained_layout=True, sharey=True)
    axes = axes.flatten()

    # Compute shared y limits
    all_means, all_sems = [], []
    for rule in trained:
        for roi in rois:
            _, m, s = curve(df, rule, roi)
            all_means.extend(m); all_sems.extend(s)
    rw_max = max(seed_vals(df, "Random Weights", roi, 0).mean() for roi in rois)
    ymin = min(all_means) - 0.005
    ymax = max(rw_max, max(m + s for m, s in zip(all_means, all_sems))) + 0.01

    for ax, roi in zip(axes, rois):
        rw_rho = seed_vals(df, "Random Weights", roi, 0).mean()
        ax.axhline(rw_rho, color=COLORS["Random Weights"], lw=1.0,
                   ls="--", alpha=0.7, zorder=1)

        for rule in trained:
            eps, means, sems = curve(df, rule, roi)
            ax.plot(eps, means, "o-", color=COLORS[rule], lw=1.5, ms=3,
                    label=RULE_SHORT[rule], zorder=3)
            ax.fill_between(eps, means - sems, means + sems,
                             color=COLORS[rule], alpha=0.13, zorder=2)

        set_epoch_axis(ax, all_epochs)
        ax.set_ylim(ymin, ymax)
        ax.axhline(0, color="black", lw=0.4, zorder=0)
        ax.set_title(ROI_LABELS[roi], fontweight="bold", pad=4)

    for ax in axes[3:]:
        ax.set_xlabel("Training epoch")
    for ax in [axes[0], axes[3]]:
        ax.set_ylabel("Spearman r (best layer)")

    # Shared legend
    handles = ([plt.Line2D([0], [0], color=COLORS[r], lw=1.5,
                            label=RULE_SHORT[r]) for r in trained] +
               [plt.Line2D([0], [0], color=COLORS["Random Weights"], lw=1.0,
                            ls="--", label="Untrained")])
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.suptitle("fMRI alignment across training — all ROIs", fontsize=12, y=1.01)

    fig.savefig(str(OUT / "figure2_all_rois.pdf"))
    fig.savefig(str(OUT / "figure2_all_rois.png"), dpi=200)
    plt.close()
    print("Saved: figure2_all_rois.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — V1 vs LOC (opposing trends)
# ══════════════════════════════════════════════════════════════════════════════

def fig3(df):
    trained    = [r for r in RULE_ORDER if r != "Random Weights" and r in df["rule"].values]
    all_epochs = sorted(df["epoch"].unique())

    fig, (ax_v1, ax_loc) = plt.subplots(1, 2, figsize=(130 * MM, 68 * MM),
                                         constrained_layout=True)

    for roi, ax, title in [("V1", ax_v1, "V1 (early visual)"),
                            ("LOC", ax_loc, "LOC (object recognition)")]:
        rw_rho = seed_vals(df, "Random Weights", roi, 0).mean()
        ax.axhline(rw_rho, color=COLORS["Random Weights"], lw=1.0,
                   ls="--", alpha=0.7, label="Untrained")

        for rule in trained:
            eps, means, sems = curve(df, rule, roi)
            ax.plot(eps, means, "o-", color=COLORS[rule], lw=1.8, ms=4,
                    label=RULE_SHORT[rule], zorder=3)
            ax.fill_between(eps, means - sems, means + sems,
                             color=COLORS[rule], alpha=0.15, zorder=2)

            # Annotate crossover: where does this rule's LOC curve cross its V1 curve?
            if roi == "LOC":
                eps_v1, means_v1, _ = curve(df, rule, "V1")
                for i in range(len(eps) - 1):
                    ep = eps[i]
                    if ep not in eps_v1:
                        continue
                    idx_v1 = np.where(eps_v1 == ep)[0]
                    idx_v1_n = np.where(eps_v1 == eps[i + 1])[0]
                    if len(idx_v1) == 0 or len(idx_v1_n) == 0:
                        continue
                    loc_now  = means[i];     loc_next  = means[i + 1]
                    v1_now   = means_v1[idx_v1[0]];  v1_next  = means_v1[idx_v1_n[0]]
                    if (loc_now < v1_now) and (loc_next >= v1_next):
                        # Crossover between ep[i] and ep[i+1]
                        x_cross = (eps[i] + eps[i + 1]) / 2
                        ax.annotate("",
                                    xy=(x_cross, (loc_now + v1_now) / 2),
                                    xytext=(x_cross, (loc_now + v1_now) / 2 + 0.008),
                                    arrowprops=dict(arrowstyle="->", color=COLORS[rule],
                                                    lw=1.0))

        set_epoch_axis(ax, all_epochs)
        ax.axhline(0, color="black", lw=0.4)
        ax.set_xlabel("Training epoch")
        ax.set_title(title, pad=6)

    ax_v1.set_ylabel("Spearman r (best layer)")
    ax_v1.legend(frameon=False, loc="upper right", fontsize=8, handlelength=1.2)
    add_panel_label(ax_v1, "A")
    add_panel_label(ax_loc, "B")

    # Shared y limits for direct comparison
    all_y = []
    for roi in ["V1", "LOC"]:
        for rule in trained:
            _, m, s = curve(df, rule, roi)
            all_y.extend(m - s); all_y.extend(m + s)
        all_y.append(seed_vals(df, "Random Weights", roi, 0).mean())
    ymin = min(all_y) - 0.005
    ymax = max(all_y) + 0.015
    ax_v1.set_ylim(ymin, ymax)
    ax_loc.set_ylim(ymin, ymax)

    fig.suptitle("Opposing alignment trends: V1 falls, LOC rises during training",
                 fontsize=10, y=1.02)

    fig.savefig(str(OUT / "figure3_v1_vs_loc.pdf"))
    fig.savefig(str(OUT / "figure3_v1_vs_loc.png"), dpi=200)
    plt.close()
    print("Saved: figure3_v1_vs_loc.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY S1 — Seed variability (box plots)
# ══════════════════════════════════════════════════════════════════════════════

def figS1(df):
    roi        = "V1"
    all_rules  = [r for r in RULE_ORDER if r in df["rule"].values]
    trained    = [r for r in all_rules if r != "Random Weights"]
    epochs_cmp = [0, COMPARE_EPOCH]

    fig, axes = plt.subplots(1, 2, figsize=(130 * MM, 70 * MM),
                              constrained_layout=True, sharey=True)

    for ax, ep in zip(axes, epochs_cmp):
        rules_here = all_rules if ep == 0 else trained
        data_list  = []
        labels     = []
        colors     = []
        for rule in rules_here:
            v = seed_vals(df, rule, roi, ep)
            data_list.append(v)
            labels.append(RULE_SHORT[rule])
            colors.append(COLORS[rule])

        bp = ax.boxplot(data_list, patch_artist=True, widths=0.5,
                        medianprops={"color": "black", "lw": 1.5},
                        whiskerprops={"lw": 0.8}, capprops={"lw": 0.8},
                        flierprops={"marker": "o", "ms": 3, "alpha": 0.6},
                        boxprops={"lw": 0.8})
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Overlay individual seed points
        rng = np.random.default_rng(ep)
        for i, (v, color) in enumerate(zip(data_list, colors), start=1):
            jitter = rng.uniform(-0.12, 0.12, size=len(v))
            ax.scatter(i + jitter, v, s=18, color=color, zorder=4,
                       edgecolors="white", linewidths=0.5)

        title = "Epoch 0 — Untrained" if ep == 0 else f"Epoch {ep} — Trained"
        ax.set_title(title, pad=6)
        ax.set_xticks(range(1, len(rules_here) + 1))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.axhline(0, color="black", lw=0.5)

    axes[0].set_ylabel("Spearman r (V1, best layer)")
    add_panel_label(axes[0], "A")
    add_panel_label(axes[1], "B")
    fig.suptitle(f"V1 alignment — seed variability (n = {len(df['seed_idx'].unique())} seeds)",
                 fontsize=10, y=1.02)

    fig.savefig(str(OUT / "figureS1_seed_variability.pdf"))
    fig.savefig(str(OUT / "figureS1_seed_variability.png"), dpi=200)
    plt.close()
    print("Saved: figureS1_seed_variability.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    df = load()
    print("\nGenerating figures...")
    fig1(df)
    fig2(df)
    fig3(df)
    figS1(df)
    print(f"\nAll figures saved to: {OUT}")


if __name__ == "__main__":
    main()

"""
training_dynamics_analysis.py
==============================
Statistical tests and paper-ready figures for the training dynamics study.

Input:  training_dynamics_outputs/outputs/training_dynamics_results.csv
Output:
  training_dynamics_outputs/stats/permutation_bp_vs_others.csv
  training_dynamics_outputs/stats/epoch0_vs_epoch1_drop.csv
  training_dynamics_outputs/stats/monotone_trend.csv
  training_dynamics_outputs/stats/all_tests_fdr.csv
  training_dynamics_outputs/figures/figure1_v1_dynamics.pdf

Usage:
  py training_dynamics_analysis.py
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from itertools import product
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE    = Path(__file__).parent / "training_dynamics_outputs"
DATA    = BASE / "outputs" / "training_dynamics_results.csv"
STAT_DIR = BASE / "stats"
FIG_DIR  = BASE / "figures"

COMPARE_EPOCH = 40       # final training epoch for bar chart / pairwise tests
ROIS          = ["V1", "V2", "V3", "V4", "LOC", "IT"]
LAYERS        = ["Conv1", "Conv2", "Conv3", "FC1"]

RULE_ORDER = ["Random Weights", "Backprop", "Feedback Alignment",
              "Predictive Coding", "STDP"]
RULE_SHORT = {
    "Random Weights":     "Rand",
    "Backprop":           "BP",
    "Feedback Alignment": "FA",
    "Predictive Coding":  "PC",
    "STDP":               "STDP",
}
COLORS = {
    "Random Weights":     "#999999",
    "Backprop":           "#2E86AB",
    "Feedback Alignment": "#E84855",
    "Predictive Coding":  "#3BB273",
    "STDP":               "#F4A261",
}

# ── Nature-style rcParams ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         7,
    "axes.labelsize":    8,
    "axes.titlesize":    8,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "legend.fontsize":   7,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "pdf.fonttype":      42,   # embeds fonts as TrueType in PDF
    "ps.fonttype":       42,
})

MM = 1 / 25.4   # mm to inches


# ══════════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    df = pd.read_csv(str(DATA))
    print(f"Loaded {len(df)} rows")
    print(f"  Rules:   {df['rule'].unique().tolist()}")
    print(f"  Seeds:   {sorted(df['seed_idx'].unique())}")
    print(f"  Epochs:  {sorted(df['epoch'].unique())}")
    return df


def nearest_epoch(df, target):
    """Return the available epoch closest to target."""
    epochs = sorted(df["epoch"].unique())
    return min(epochs, key=lambda e: abs(e - target))


def best_layer_for(df, rule, roi, epoch):
    """
    Fixed best layer: the layer with the highest mean rho across all seeds
    and subjects at the given epoch. Used consistently for all seed-level values.
    """
    sub = df[(df["rule"] == rule) & (df["roi"] == roi) & (df["epoch"] == epoch)]
    if sub.empty:
        return LAYERS[0]
    return sub.groupby("layer")["rho"].mean().idxmax()


def seed_rhos(df, rule, roi, epoch, layer=None):
    """
    Per-seed mean rho (averaged across subjects) at a given epoch.
    If layer is None, uses the fixed best layer.
    Returns array of length n_seeds.
    """
    if layer is None:
        layer = best_layer_for(df, rule, roi, epoch)
    sub = df[(df["rule"] == rule) & (df["roi"] == roi)
             & (df["epoch"] == epoch) & (df["layer"] == layer)]
    return sub.groupby("seed_idx")["rho"].mean().values


def epoch_curve(df, rule, roi, use_best_at_each_epoch=True):
    """
    For each available epoch: mean and SEM across seeds (using best layer).
    Returns (epochs, means, sems).
    """
    epochs = sorted(df[(df["rule"] == rule) & (df["roi"] == roi)]["epoch"].unique())
    means, sems = [], []
    for ep in epochs:
        if use_best_at_each_epoch:
            layer = best_layer_for(df, rule, roi, ep)
        else:
            layer = best_layer_for(df, rule, roi, COMPARE_EPOCH)
        vals = seed_rhos(df, rule, roi, ep, layer)
        means.append(float(np.mean(vals)))
        sems.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)
    return np.array(epochs), np.array(means), np.array(sems)


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def sign_flip_perm_test(a, b):
    """
    Exact paired permutation test via sign flipping (for n ≤ 15).
    H0: mean(a - b) = 0.
    Returns (observed_diff, two-sided p-value).
    """
    diffs = np.array(a) - np.array(b)
    obs   = float(diffs.mean())
    n     = len(diffs)
    signs = np.array(list(product([-1, 1], repeat=n)))   # 2^n rows
    null  = (signs * diffs).mean(axis=1)
    p     = float(np.mean(np.abs(null) >= np.abs(obs)))
    return obs, p


def cohens_d_paired(a, b):
    diffs = np.array(a) - np.array(b)
    return float(diffs.mean() / diffs.std(ddof=1)) if diffs.std(ddof=1) > 1e-10 else np.nan


def benjamini_hochberg(pvals):
    arr  = np.array(pvals, dtype=float)
    n    = len(arr)
    idx  = np.argsort(arr)
    adj  = np.zeros(n)
    cum  = 1.0
    for i in range(n - 1, -1, -1):
        cum = min(cum, arr[idx[i]] * n / (i + 1))
        adj[idx[i]] = min(cum, 1.0)
    return adj.tolist()


def sig_label(p):
    if np.isnan(p):    return "N/A"
    if p < 0.001:      return "***"
    if p < 0.01:       return "**"
    if p < 0.05:       return "*"
    return "ns"


# ── 1a: Pairwise permutation tests at COMPARE_EPOCH ──────────────────────────

def run_permutation_tests(df):
    rules   = [r for r in RULE_ORDER if r in df["rule"].values and r != "Random Weights"]
    ep      = nearest_epoch(df, COMPARE_EPOCH)
    rows    = []
    all_p   = []
    print(f"\n[1a] Permutation tests vs Backprop at epoch {ep}")
    for roi in ROIS:
        bp_vals = seed_rhos(df, "Backprop", roi, ep)
        if len(bp_vals) == 0:
            continue
        for rule_b in rules:
            if rule_b == "Backprop":
                continue
            other_vals = seed_rhos(df, rule_b, roi, ep)
            if len(other_vals) == 0:
                continue
            # align seeds
            n = min(len(bp_vals), len(other_vals))
            obs, p = sign_flip_perm_test(other_vals[:n], bp_vals[:n])
            d = cohens_d_paired(other_vals[:n], bp_vals[:n])
            all_p.append(p)
            rows.append({
                "roi": roi, "epoch": ep,
                "rule_a": rule_b, "rule_b": "Backprop",
                "rho_a":  round(float(other_vals.mean()), 5),
                "rho_b":  round(float(bp_vals.mean()), 5),
                "delta":  round(obs, 5),
                "p_uncorr": round(p, 4),
                "sig":    sig_label(p),
                "cohens_d": round(d, 3),
            })
            print(f"  {roi:4s}  {rule_b:22s} vs BP:  d={obs:+.4f}  p={p:.4f} {sig_label(p)}")
    fdr = benjamini_hochberg(all_p)
    for i, row in enumerate(rows):
        row["p_fdr"] = round(fdr[i], 4)
        row["sig_fdr"] = sig_label(fdr[i])
    return pd.DataFrame(rows), all_p


# ── 1b: Epoch 0 vs Epoch 1 drop ───────────────────────────────────────────────

def run_epoch_drop_tests(df):
    trained = [r for r in RULE_ORDER if r in df["rule"].values and r != "Random Weights"]
    rows    = []
    all_p   = []
    ep1     = nearest_epoch(df, 1)
    print(f"\n[1b] Epoch 0 vs epoch {ep1} drop test")
    for rule in trained:
        for roi in ROIS:
            vals0 = seed_rhos(df, rule, roi, 0)
            vals1 = seed_rhos(df, rule, roi, ep1)
            n = min(len(vals0), len(vals1))
            if n < 2:
                continue
            obs, p = sign_flip_perm_test(vals1[:n], vals0[:n])
            d = cohens_d_paired(vals1[:n], vals0[:n])
            all_p.append(p)
            rows.append({
                "rule": rule, "roi": roi,
                "rho_ep0": round(float(vals0.mean()), 5),
                "rho_ep1": round(float(vals1.mean()), 5),
                "delta":   round(obs, 5),
                "p_uncorr": round(p, 4),
                "sig":     sig_label(p),
                "cohens_d": round(d, 3),
            })
            if roi in ["V1", "LOC"]:
                print(f"  {rule:22s}  {roi:4s}  "
                      f"rho0={vals0.mean():.4f}  rho1={vals1.mean():.4f}  "
                      f"d={obs:+.4f}  p={p:.4f} {sig_label(p)}")
    fdr = benjamini_hochberg(all_p)
    for i, row in enumerate(rows):
        row["p_fdr"] = round(fdr[i], 4)
        row["sig_fdr"] = sig_label(fdr[i])
    return pd.DataFrame(rows), all_p


# ── 1c: Monotone trend ────────────────────────────────────────────────────────

def run_monotone_trend(df):
    rules = [r for r in RULE_ORDER if r in df["rule"].values and r != "Random Weights"]
    rows  = []
    all_p = []
    print(f"\n[1c] Monotone trend (Spearman epoch vs rho, epochs > 0)")
    for rule in rules:
        for roi in ROIS:
            epochs_avail = sorted(df[(df["rule"] == rule) & (df["roi"] == roi)
                                     & (df["epoch"] > 0)]["epoch"].unique())
            if len(epochs_avail) < 3:
                continue
            # mean rho per epoch (best layer at each epoch)
            rho_by_ep = []
            for ep in epochs_avail:
                vals = seed_rhos(df, rule, roi, ep)
                rho_by_ep.append(float(np.mean(vals)))
            rho_rho, p = spearmanr(epochs_avail, rho_by_ep)
            all_p.append(p)
            rows.append({
                "rule": rule, "roi": roi,
                "spearman_rho": round(float(rho_rho), 4),
                "p_uncorr":     round(float(p), 4),
                "sig":          sig_label(p),
                "direction":    "decrease" if rho_rho < 0 else "increase",
                "n_epochs":     len(epochs_avail),
            })
    fdr = benjamini_hochberg(all_p)
    for i, row in enumerate(rows):
        row["p_fdr"] = round(fdr[i], 4)
        row["sig_fdr"] = sig_label(fdr[i])
    # Print V1 summary
    df_trend = pd.DataFrame(rows)
    if not df_trend.empty:
        print("  V1 trend:")
        for _, r in df_trend[df_trend["roi"] == "V1"].iterrows():
            print(f"    {r['rule']:22s}  rho={r['spearman_rho']:+.3f}  "
                  f"p={r['p_uncorr']:.4f} {r['sig']}  {r['direction']}")
    return df_trend, all_p


# ── Combined FDR ──────────────────────────────────────────────────────────────

def run_all_stats(df):
    STAT_DIR.mkdir(parents=True, exist_ok=True)

    perm_df, p1  = run_permutation_tests(df)
    drop_df, p2  = run_epoch_drop_tests(df)
    trend_df, p3 = run_monotone_trend(df)

    perm_df.to_csv(str(STAT_DIR / "permutation_bp_vs_others.csv"), index=False)
    drop_df.to_csv(str(STAT_DIR / "epoch0_vs_epoch1_drop.csv"),    index=False)
    trend_df.to_csv(str(STAT_DIR / "monotone_trend.csv"),           index=False)

    # Global FDR across all tests
    all_p  = p1 + p2 + p3
    all_fdr = benjamini_hochberg(all_p)
    fdr_rows = (
        [{"test": "perm_bp_vs_other", **r} for r in perm_df.to_dict("records")] +
        [{"test": "epoch0_vs_ep1",    **r} for r in drop_df.to_dict("records")] +
        [{"test": "monotone_trend",   **r} for r in trend_df.to_dict("records")]
    )
    for i, row in enumerate(fdr_rows):
        row["p_global_fdr"] = round(all_fdr[i], 4)
        row["sig_global"] = sig_label(all_fdr[i])
    pd.DataFrame(fdr_rows).to_csv(str(STAT_DIR / "all_tests_fdr.csv"), index=False)
    print(f"\n  Stats saved to {STAT_DIR}")
    return perm_df, drop_df, trend_df


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — 3-panel V1 dynamics (Nature-style, PDF)
# ══════════════════════════════════════════════════════════════════════════════

def _epoch_xpos(epochs):
    """Map epoch values to evenly-spaced integer positions for display."""
    return np.arange(len(epochs))


def _rule_sig_at_epoch(perm_df, rule, roi, epoch):
    """Return FDR-corrected significance label for rule vs BP at epoch."""
    if perm_df is None or perm_df.empty:
        return ""
    sub = perm_df[(perm_df["rule_a"] == rule) & (perm_df["roi"] == roi)
                  & (perm_df["epoch"] == epoch)]
    return sub.iloc[0]["sig_fdr"] if not sub.empty else ""


def make_figure1(df, perm_df=None):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    roi     = "V1"
    ep_cmp  = nearest_epoch(df, COMPARE_EPOCH)
    rules   = [r for r in RULE_ORDER if r in df["rule"].values]
    trained = [r for r in rules if r != "Random Weights"]

    fig = plt.figure(figsize=(183 * MM, 72 * MM))
    gs  = gridspec.GridSpec(1, 3, figure=fig,
                            left=0.08, right=0.97, bottom=0.18, top=0.88,
                            wspace=0.42)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    # ── Panel A: V1 alignment across training ─────────────────────────────────
    # Use Random Weights epoch-0 value as horizontal reference line
    rw_rho = seed_rhos(df, "Random Weights", roi, 0).mean()
    ax_a.axhline(rw_rho, color=COLORS["Random Weights"], lw=1.0,
                 linestyle="--", alpha=0.7, zorder=1)

    for rule in trained:
        epochs, means, sems = epoch_curve(df, rule, roi)
        x = _epoch_xpos(epochs)
        ax_a.plot(x, means, "o-", color=COLORS[rule], label=RULE_SHORT[rule],
                  lw=1.5, ms=3, zorder=3)
        ax_a.fill_between(x, means - sems, means + sems,
                           color=COLORS[rule], alpha=0.15, zorder=2)

    epoch_vals = sorted(df[df["roi"] == roi]["epoch"].unique())
    epoch_vals = [e for e in epoch_vals if e > 0 or
                  any(df[(df["rule"] == r) & (df["roi"] == roi) &
                          (df["epoch"] == 0)].shape[0] > 0 for r in trained)]
    xpos = _epoch_xpos(epoch_vals)
    ax_a.set_xticks(xpos)
    ax_a.set_xticklabels([str(e) for e in epoch_vals], rotation=45, ha="right")
    ax_a.axvline(0, color="black", lw=0.6, linestyle=":", alpha=0.5)
    ax_a.text(0.04, rw_rho + 0.002, "Untrained", transform=ax_a.get_xaxis_transform(),
              fontsize=6, color=COLORS["Random Weights"], va="bottom")
    ax_a.set_xlabel("Training epoch")
    ax_a.set_ylabel("Spearman r (V1, best layer)")
    ax_a.set_title("A  V1 alignment during training", loc="left", pad=3, fontweight="bold")
    ax_a.legend(frameon=False, loc="upper right", handlelength=1.2)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # ── Panel B: Delta rho (V1, normalised to epoch 0) ───────────────────────
    for rule in trained:
        epochs, means, sems = epoch_curve(df, rule, roi)
        baseline = means[epochs == 0][0] if 0 in epochs else means[0]
        deltas   = means - baseline
        delta_sems = sems
        x = _epoch_xpos(epochs)
        ax_b.plot(x, deltas, "o-", color=COLORS[rule], label=RULE_SHORT[rule],
                  lw=1.5, ms=3, zorder=3)
        ax_b.fill_between(x, deltas - delta_sems, deltas + delta_sems,
                           color=COLORS[rule], alpha=0.15, zorder=2)

    ax_b.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.7)
    ax_b.set_xticks(xpos)
    ax_b.set_xticklabels([str(e) for e in epoch_vals], rotation=45, ha="right")
    ax_b.set_xlabel("Training epoch")
    ax_b.set_ylabel(r"$\Delta$r (relative to epoch 0)")
    ax_b.set_title("B  Change from untrained baseline", loc="left", pad=3, fontweight="bold")
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # ── Panel C: Bar chart at ep_cmp, all rules, V1 ──────────────────────────
    bar_rules = [r for r in RULE_ORDER if r in df["rule"].values]
    bar_vals, bar_sems, bar_colors = [], [], []
    for rule in bar_rules:
        ep_use = ep_cmp if rule != "Random Weights" else 0
        vals = seed_rhos(df, rule, roi, ep_use)
        bar_vals.append(float(np.mean(vals)))
        bar_sems.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)
        bar_colors.append(COLORS[rule])

    x_bar = np.arange(len(bar_rules))
    bars = ax_c.bar(x_bar, bar_vals, 0.65, color=bar_colors, alpha=0.85,
                    yerr=bar_sems, capsize=2.5,
                    error_kw={"lw": 0.8, "ecolor": "black"}, zorder=3)
    ax_c.axhline(0, color="black", lw=0.5)

    # Annotate rho values
    for bar, v in zip(bars, bar_vals):
        y = max(bar.get_height(), 0) + max(bar_sems[list(bar_vals).index(v)], 0) + 0.002
        ax_c.text(bar.get_x() + bar.get_width() / 2, y, f"{v:.3f}",
                  ha="center", va="bottom", fontsize=5.5)

    # Significance brackets vs BP
    bp_idx = bar_rules.index("Backprop") if "Backprop" in bar_rules else None
    if bp_idx is not None and perm_df is not None and not perm_df.empty:
        bracket_y = max(bar_vals) + max(bar_sems) + 0.012
        for i, rule in enumerate(bar_rules):
            if rule in ("Backprop", "Random Weights"):
                continue
            sig = _rule_sig_at_epoch(perm_df, rule, roi, ep_cmp)
            if sig and sig != "ns":
                y0 = bracket_y
                ax_c.plot([bp_idx, bp_idx, i, i],
                           [y0, y0 + 0.003, y0 + 0.003, y0],
                           lw=0.8, color="black")
                ax_c.text((bp_idx + i) / 2, y0 + 0.004, sig,
                           ha="center", va="bottom", fontsize=7)
                bracket_y += 0.014

    ax_c.set_xticks(x_bar)
    ax_c.set_xticklabels([RULE_SHORT[r] for r in bar_rules], rotation=30, ha="right")
    ax_c.set_ylabel("Spearman r (V1, best layer)")
    ax_c.set_title(f"C  V1 alignment at epoch {ep_cmp}", loc="left", pad=3, fontweight="bold")
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    out = FIG_DIR / "figure1_v1_dynamics.pdf"
    plt.savefig(str(out), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {out}")

    # Also save PNG for quick preview
    fig2 = plt.figure(figsize=(183 * MM, 72 * MM))
    gs2  = gridspec.GridSpec(1, 3, figure=fig2,
                             left=0.08, right=0.97, bottom=0.18, top=0.88,
                             wspace=0.42)
    # Replot (same code, cleaner approach: just save both formats)
    plt.close()
    # Re-run with PNG backend
    _save_figure1_png(df, perm_df, roi, ep_cmp, trained, bar_rules,
                      epoch_vals, xpos, rw_rho)


def _save_figure1_png(df, perm_df, roi, ep_cmp, trained, bar_rules,
                      epoch_vals, xpos, rw_rho):
    """PNG preview of Figure 1."""
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(13, 4.5))

    rw_rho_val = seed_rhos(df, "Random Weights", roi, 0).mean()
    ax_a.axhline(rw_rho_val, color=COLORS["Random Weights"], lw=1.0,
                 linestyle="--", alpha=0.7)

    for rule in trained:
        epochs, means, sems = epoch_curve(df, rule, roi)
        x = _epoch_xpos(epochs)
        ax_a.plot(x, means, "o-", color=COLORS[rule], label=RULE_SHORT[rule], lw=1.5, ms=3)
        ax_a.fill_between(x, means - sems, means + sems,
                           color=COLORS[rule], alpha=0.15)
    ax_a.set_xticks(xpos)
    ax_a.set_xticklabels([str(e) for e in epoch_vals], rotation=45, ha="right", fontsize=8)
    ax_a.axvline(0, color="black", lw=0.6, linestyle=":")
    ax_a.set_xlabel("Training epoch"); ax_a.set_ylabel("Spearman r (V1, best layer)")
    ax_a.set_title("A  V1 alignment during training", loc="left", fontweight="bold")
    ax_a.legend(frameon=False, fontsize=8)
    ax_a.spines["top"].set_visible(False); ax_a.spines["right"].set_visible(False)

    for rule in trained:
        epochs, means, sems = epoch_curve(df, rule, roi)
        baseline = means[epochs == 0][0] if 0 in epochs else means[0]
        deltas = means - baseline
        x = _epoch_xpos(epochs)
        ax_b.plot(x, deltas, "o-", color=COLORS[rule], label=RULE_SHORT[rule], lw=1.5, ms=3)
        ax_b.fill_between(x, deltas - sems, deltas + sems, color=COLORS[rule], alpha=0.15)
    ax_b.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.7)
    ax_b.set_xticks(xpos)
    ax_b.set_xticklabels([str(e) for e in epoch_vals], rotation=45, ha="right", fontsize=8)
    ax_b.set_xlabel("Training epoch"); ax_b.set_ylabel("Delta r (relative to epoch 0)")
    ax_b.set_title("B  Change from untrained baseline", loc="left", fontweight="bold")
    ax_b.spines["top"].set_visible(False); ax_b.spines["right"].set_visible(False)

    bar_vals, bar_sems, bar_colors = [], [], []
    for rule in bar_rules:
        ep_use = ep_cmp if rule != "Random Weights" else 0
        vals = seed_rhos(df, rule, roi, ep_use)
        bar_vals.append(float(np.mean(vals)))
        bar_sems.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)
        bar_colors.append(COLORS[rule])
    x_bar = np.arange(len(bar_rules))
    ax_c.bar(x_bar, bar_vals, 0.65, color=bar_colors, alpha=0.85,
             yerr=bar_sems, capsize=3, error_kw={"lw": 1.0})
    ax_c.axhline(0, color="black", lw=0.5)
    ax_c.set_xticks(x_bar)
    ax_c.set_xticklabels([RULE_SHORT[r] for r in bar_rules], rotation=30, ha="right", fontsize=8)
    ax_c.set_ylabel("Spearman r (V1, best layer)")
    ax_c.set_title(f"C  V1 alignment at epoch {ep_cmp}", loc="left", fontweight="bold")
    ax_c.spines["top"].set_visible(False); ax_c.spines["right"].set_visible(False)

    plt.tight_layout()
    out = FIG_DIR / "figure1_v1_dynamics.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY: all-ROI grid
# ══════════════════════════════════════════════════════════════════════════════

def make_all_rois_figure(df):
    trained = [r for r in RULE_ORDER if r in df["rule"].values and r != "Random Weights"]
    rois    = [r for r in ROIS if r in df["roi"].values]

    fig, axes = plt.subplots(2, 3, figsize=(183 * MM, 120 * MM))
    axes = axes.flatten()

    for ax, roi in zip(axes, rois):
        rw_rho = seed_rhos(df, "Random Weights", roi, 0).mean()
        ax.axhline(rw_rho, color=COLORS["Random Weights"], lw=0.8,
                   linestyle="--", alpha=0.6)
        for rule in trained:
            epochs, means, sems = epoch_curve(df, rule, roi)
            x = _epoch_xpos(epochs)
            ax.plot(x, means, "o-", color=COLORS[rule], label=RULE_SHORT[rule],
                    lw=1.2, ms=2.5)
            ax.fill_between(x, means - sems, means + sems,
                             color=COLORS[rule], alpha=0.12)
        epoch_vals = sorted(df[df["roi"] == roi]["epoch"].unique())
        xpos = _epoch_xpos(epoch_vals)
        ax.set_xticks(xpos)
        ax.set_xticklabels([str(e) for e in epoch_vals], rotation=45, ha="right", fontsize=6)
        ax.set_title(roi, fontweight="bold")
        ax.axhline(0, color="black", lw=0.4)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Spearman r (best layer)")
    axes[3].set_ylabel("Spearman r (best layer)")
    handles = [plt.Line2D([0], [0], color=COLORS[r], lw=1.5, label=RULE_SHORT[r])
               for r in trained]
    handles.append(plt.Line2D([0], [0], color=COLORS["Random Weights"],
                               lw=1.0, linestyle="--", label="Untrained"))
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7,
               bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.suptitle("fMRI alignment across training — all ROIs", fontsize=8, y=1.01)
    plt.tight_layout()

    for ext in ["pdf", "png"]:
        out = FIG_DIR / f"figureS1_all_rois.{ext}"
        plt.savefig(str(out), dpi=300 if ext == "pdf" else 150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: figureS1_all_rois.pdf/png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Data not found: {DATA}")

    print("=" * 60)
    print("Training Dynamics Analysis")
    print("=" * 60)

    df = load_data()

    # Stats
    perm_df, drop_df, trend_df = run_all_stats(df)

    # Figures
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("\n[2] Generating figures...")
    make_figure1(df, perm_df)
    make_all_rois_figure(df)

    # Print key results
    ep = nearest_epoch(df, COMPARE_EPOCH)
    print(f"\n{'='*60}")
    print(f"KEY RESULTS — V1 at epoch {ep}")
    print(f"{'='*60}")
    rw = seed_rhos(df, "Random Weights", "V1", 0).mean()
    print(f"  Random (untrained):  r = {rw:.4f}")
    for rule in [r for r in RULE_ORDER if r != "Random Weights" and r in df["rule"].values]:
        vals = seed_rhos(df, rule, "V1", ep)
        drop_from_ep0 = vals.mean() - seed_rhos(df, rule, "V1", 0).mean()
        print(f"  {rule:22s}  r = {vals.mean():.4f} (SD={vals.std():.4f})  "
              f"drop from ep0 = {drop_from_ep0:+.4f}")

    print(f"\nAll outputs in: {BASE}")


if __name__ == "__main__":
    main()

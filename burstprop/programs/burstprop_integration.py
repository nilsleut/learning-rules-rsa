"""
burstprop_integration.py
========================
Integrates Burstprop RSA results into the learning-rules-rsa analysis pipeline.

Outputs (in outputs/):
    rsa_results_cnn_with_burstprop.csv   -- combined results table
    burstprop_heatmap.png                -- layer × ROI heatmap for Burstprop
    comparison_v1_bar.png                -- V1 best-layer ρ per rule (bar chart)
    comparison_all_rois.png              -- all ROIs grouped bar chart
    burstprop_stats.csv                  -- permutation + Cohen's d vs Random

Usage:
    py burstprop_integration.py
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
LEARN_DIR   = Path(__file__).parent
FMRI_DIR    = LEARN_DIR / "outputs_720"
OUTPUTS_DIR = LEARN_DIR / "outputs"
BP_DIR      = Path(r"C:\Users\nilsl\Desktop\GitHub\Burstprop_RSA\all_outputs")

SUBJECTS = ["sub-01", "sub-02", "sub-03"]
ROIS     = ["V1", "V2", "V3", "V4", "LOC", "IT"]
LAYERS   = ["Conv1", "Conv2", "Conv3", "FC1"]

COLORS = {
    "Random Weights":     "#999999",
    "Backprop":           "#2E86AB",
    "Burstprop":          "#9B59B6",
    "Feedback Alignment": "#E84855",
    "Predictive Coding":  "#3BB273",
    "STDP":               "#F4A261",
}

RULE_ORDER = ["Random Weights", "Backprop", "Burstprop",
              "Feedback Alignment", "Predictive Coding", "STDP"]

N_PERM = 1000
rng    = np.random.default_rng(42)


# ── Utilities ──────────────────────────────────────────────────────────────────

def upper_tri(rdm):
    return rdm[np.triu_indices(rdm.shape[0], k=1)]

def rsa_score(rdm_a, rdm_b):
    n   = min(rdm_a.shape[0], rdm_b.shape[0])
    idx = np.triu_indices(n, k=1)
    r, _ = spearmanr(rdm_a[:n, :n][idx], rdm_b[:n, :n][idx])
    return float(r)

def load_fmri_rdm(roi, sub):
    p = FMRI_DIR / f"fmri_rdm_{roi}_{sub}.npy"
    return np.load(str(p)) if p.exists() else None

def mean_brain_rdm(roi):
    rdms = [load_fmri_rdm(roi, s) for s in SUBJECTS]
    rdms = [r for r in rdms if r is not None]
    if not rdms:
        return None
    n = min(r.shape[0] for r in rdms)
    return np.mean([r[:n, :n] for r in rdms], axis=0)


# ── Build Burstprop summary in rsa_results_cnn.csv format ─────────────────────

def build_burstprop_summary():
    """
    For each (layer, ROI): compute ρ from seed-averaged model RDM vs mean brain RDM,
    plus per-subject ρ and seed-level std for CI.
    """
    rows = []
    for layer in LAYERS:
        # Load per-seed RDMs
        seed_rdms = []
        for s in range(3):
            p = BP_DIR / "model_rdms" / f"seed_{s}" / f"rdm_burstprop_{layer}.npy"
            if p.exists():
                seed_rdms.append(np.load(str(p)))
        if not seed_rdms:
            print(f"  WARNING: no RDMs found for {layer}")
            continue

        n    = min(r.shape[0] for r in seed_rdms)
        mean_rdm = np.mean([r[:n, :n] for r in seed_rdms], axis=0)

        for roi in ROIS:
            brain = mean_brain_rdm(roi)
            if brain is None:
                continue

            # ρ from mean brain RDM (consistent with existing pipeline)
            rho_mean_brain = rsa_score(mean_rdm, brain)

            # Seed-level ρ values for std / CI
            seed_rhos = [rsa_score(r[:n, :n], brain) for r in seed_rdms]

            # Per-subject ρ (from mean model RDM)
            sub_rhos = {}
            for sub in SUBJECTS:
                b = load_fmri_rdm(roi, sub)
                sub_rhos[sub] = rsa_score(mean_rdm, b) if b is not None else np.nan

            rho_arr = np.array(seed_rhos)
            rows.append({
                "rule":         "Burstprop",
                "layer":        layer,
                "roi":          roi,
                "rho":          round(float(np.mean(rho_arr)), 5),
                "rho_std":      round(float(np.std(rho_arr, ddof=1)) if len(rho_arr) > 1 else 0.0, 5),
                "ci_lo":        round(float(np.mean(rho_arr) - 1.96 * np.std(rho_arr) / np.sqrt(len(rho_arr))), 5),
                "ci_hi":        round(float(np.mean(rho_arr) + 1.96 * np.std(rho_arr) / np.sqrt(len(rho_arr))), 5),
                "n_seeds":      len(seed_rdms),
                "n_subs":       3,
                "rho_sub01":    round(sub_rhos["sub-01"], 5),
                "rho_sub02":    round(sub_rhos["sub-02"], 5),
                "rho_sub03":    round(sub_rhos["sub-03"], 5),
                "rho_sub_mean": round(float(np.nanmean(list(sub_rhos.values()))), 5),
            })
    return pd.DataFrame(rows)


# ── Merge with existing results ────────────────────────────────────────────────

def merge_with_existing(bp_df):
    existing_path = OUTPUTS_DIR / "rsa_results_cnn.csv"
    if not existing_path.exists():
        print(f"  WARNING: {existing_path} not found — saving Burstprop-only CSV")
        return bp_df

    existing = pd.read_csv(str(existing_path))

    # Drop any old Burstprop rows if re-running
    existing = existing[existing["rule"] != "Burstprop"]

    combined = pd.concat([existing, bp_df], ignore_index=True)
    out = OUTPUTS_DIR / "rsa_results_cnn_with_burstprop.csv"
    combined.to_csv(str(out), index=False)
    print(f"  Saved: {out.name}  ({len(combined)} rows, "
          f"{combined['rule'].nunique()} rules)")
    return combined


# ── Best-layer-per-ROI ─────────────────────────────────────────────────────────

def best_layer(df, roi):
    sub = df[(df["roi"] == roi)]
    if sub.empty:
        return None, None
    idx = sub["rho"].idxmax()
    return sub.loc[idx, "layer"], sub.loc[idx, "rho"]


# ══════════════════════════════════════════════════════════════════════════════
# PLOT A — Bar chart: V1 best-layer ρ per rule
# ══════════════════════════════════════════════════════════════════════════════

def plot_v1_bar(combined_df):
    roi = "V1"
    rules_present = [r for r in RULE_ORDER if r in combined_df["rule"].values]

    vals, errs, colors = [], [], []
    for rule in rules_present:
        sub = combined_df[(combined_df["rule"] == rule) & (combined_df["roi"] == roi)]
        if sub.empty:
            continue
        best = sub.loc[sub["rho"].idxmax()]
        vals.append(best["rho"])
        errs.append(best.get("rho_std", 0))
        colors.append(COLORS.get(rule, "#AAAAAA"))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(rules_present))
    bars = ax.bar(x, vals, yerr=errs, capsize=4, color=colors, alpha=0.85,
                  error_kw={"elinewidth": 1.5})

    # Annotate bars with ρ value
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(rules_present, fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("Spearman ρ (best layer)", fontsize=11)
    ax.set_title(f"V1 alignment — best layer per rule (mean ± SD across seeds)",
                 fontsize=12)
    ax.axhline(0, color="black", lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = OUTPUTS_DIR / "comparison_v1_bar.png"
    plt.tight_layout()
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT B — Burstprop heatmap: layer × ROI
# ══════════════════════════════════════════════════════════════════════════════

def plot_burstprop_heatmap(bp_df):
    pivot = bp_df.pivot(index="layer", columns="roi", values="rho")
    # Order rows/cols
    layer_order = [l for l in LAYERS if l in pivot.index]
    roi_order   = [r for r in ROIS if r in pivot.columns]
    pivot = pivot.loc[layer_order, roi_order]

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                   vmin=-0.01, vmax=0.06)
    plt.colorbar(im, ax=ax, label="Spearman ρ")

    ax.set_xticks(range(len(roi_order)))
    ax.set_xticklabels(roi_order, fontsize=11)
    ax.set_yticks(range(len(layer_order)))
    ax.set_yticklabels(layer_order, fontsize=11)
    ax.set_title("Burstprop: RSA ρ by layer × ROI (mean across 3 seeds)", fontsize=12)

    for i in range(len(layer_order)):
        for j in range(len(roi_order)):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=8, color="black")

    out = OUTPUTS_DIR / "burstprop_heatmap.png"
    plt.tight_layout()
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT C — All rules × all ROIs grouped bar chart
# ══════════════════════════════════════════════════════════════════════════════

def plot_all_rois(combined_df):
    rules_present = [r for r in RULE_ORDER if r in combined_df["rule"].values]
    rois_plot     = ["V1", "V2", "V3", "V4", "LOC", "IT"]
    rois_plot     = [r for r in rois_plot if r in combined_df["roi"].values]

    # best-layer ρ per (rule, roi)
    best = {}
    for rule in rules_present:
        best[rule] = {}
        for roi in rois_plot:
            sub = combined_df[(combined_df["rule"] == rule) & (combined_df["roi"] == roi)]
            if sub.empty:
                best[rule][roi] = (np.nan, 0.0)
            else:
                row = sub.loc[sub["rho"].idxmax()]
                best[rule][roi] = (row["rho"], row.get("rho_std", 0.0))

    n_rules = len(rules_present)
    n_rois  = len(rois_plot)
    w       = 0.8 / n_rules
    x       = np.arange(n_rois)

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, rule in enumerate(rules_present):
        offset = (i - n_rules / 2 + 0.5) * w
        vals = [best[rule][roi][0] for roi in rois_plot]
        errs = [best[rule][roi][1] for roi in rois_plot]
        ax.bar(x + offset, vals, w * 0.9, yerr=errs, capsize=2,
               color=COLORS.get(rule, "#AAAAAA"), alpha=0.85, label=rule,
               error_kw={"elinewidth": 1.0})

    ax.set_xticks(x)
    ax.set_xticklabels(rois_plot, fontsize=11)
    ax.set_ylabel("Spearman ρ (best layer)", fontsize=11)
    ax.set_title("fMRI alignment per learning rule and ROI (best layer, mean ± SD across seeds)",
                 fontsize=11)
    ax.axhline(0, color="black", lw=0.5)
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = OUTPUTS_DIR / "comparison_all_rois.png"
    plt.tight_layout()
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# STATS — permutation test vs Random + Cohen's d vs BP
# ══════════════════════════════════════════════════════════════════════════════

def run_burstprop_stats(bp_df, combined_df):
    """
    For each ROI (best layer for Burstprop):
      1. Permutation test: Burstprop vs Random Weights
      2. Permutation test: Burstprop vs Backprop
      3. Cohen's d (from 3-subject values): Burstprop vs Random and vs Backprop
    """
    sub_cols = ["rho_sub01", "rho_sub02", "rho_sub03"]

    def cohens_d(a, b):
        a, b = np.array(a), np.array(b)
        pooled = np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2)
        return float((a.mean() - b.mean()) / pooled) if pooled > 1e-10 else np.nan

    def perm_rdm_test(rdm_a, rdm_b, brain, n=N_PERM):
        n_obs = min(rdm_a.shape[0], rdm_b.shape[0], brain.shape[0])
        idx   = np.triu_indices(n_obs, k=1)
        va    = rdm_a[:n_obs, :n_obs][idx]
        vb    = rdm_b[:n_obs, :n_obs][idx]
        vbr   = brain[:n_obs, :n_obs][idx]
        obs   = float(spearmanr(va, vbr)[0]) - float(spearmanr(vb, vbr)[0])
        null  = []
        for _ in range(n):
            p = rng.permutation(len(vbr))
            null.append(float(spearmanr(va, vbr[p])[0]) -
                        float(spearmanr(vb, vbr[p])[0]))
        p_val = float(np.mean(np.abs(null) >= np.abs(obs)))
        return obs, p_val

    rows = []
    for roi in ROIS:
        # Best layer for Burstprop at this ROI
        bp_sub = bp_df[bp_df["roi"] == roi]
        if bp_sub.empty:
            continue
        best_layer = bp_sub.loc[bp_sub["rho"].idxmax(), "layer"]

        brain = mean_brain_rdm(roi)
        if brain is None:
            continue

        # Load mean Burstprop RDM
        seed_rdms = []
        for s in range(3):
            p = BP_DIR / "model_rdms" / f"seed_{s}" / f"rdm_burstprop_{best_layer}.npy"
            if p.exists():
                seed_rdms.append(np.load(str(p)))
        if not seed_rdms:
            continue
        n_rdm = min(r.shape[0] for r in seed_rdms)
        bp_rdm = np.mean([r[:n_rdm, :n_rdm] for r in seed_rdms], axis=0)

        bp_rho = rsa_score(bp_rdm, brain)

        # Burstprop subject values
        bp_row = bp_sub.loc[bp_sub["rho"].idxmax()]
        bp_subs = [bp_row[c] for c in sub_cols]

        # Compare against each reference rule
        for ref_rule in ["Random Weights", "Backprop"]:
            ref_sub = combined_df[(combined_df["rule"] == ref_rule) &
                                  (combined_df["roi"] == roi) &
                                  (combined_df["layer"] == best_layer)]
            if ref_sub.empty:
                # Try best layer for that rule
                ref_all = combined_df[(combined_df["rule"] == ref_rule) &
                                      (combined_df["roi"] == roi)]
                if ref_all.empty:
                    continue
                ref_sub = ref_all.loc[[ref_all["rho"].idxmax()]]

            ref_rho  = float(ref_sub.iloc[0]["rho"])
            ref_subs = [float(ref_sub.iloc[0][c]) for c in sub_cols if c in ref_sub.columns]

            d = cohens_d(bp_subs, ref_subs) if len(ref_subs) == 3 else np.nan

            # Permutation test via RDMs if ref RDM exists
            ref_rule_key = ref_rule.lower().replace(" ", "_")
            ref_rdm_path = OUTPUTS_DIR / "model_rdms" / f"rdm_{ref_rule_key}_{best_layer}.npy"
            if not ref_rdm_path.exists():
                # Try seed subdirs
                seed_paths = [
                    OUTPUTS_DIR / "model_rdms" / f"seed_{s}" / f"rdm_{ref_rule_key}_{best_layer}.npy"
                    for s in range(10)
                ]
                seed_rdms_ref = [np.load(str(p)) for p in seed_paths if p.exists()]
                if seed_rdms_ref:
                    nr = min(r.shape[0] for r in seed_rdms_ref)
                    ref_rdm = np.mean([r[:nr, :nr] for r in seed_rdms_ref], axis=0)
                else:
                    ref_rdm = None
            else:
                ref_rdm = np.load(str(ref_rdm_path))

            if ref_rdm is not None:
                n_common = min(bp_rdm.shape[0], ref_rdm.shape[0], brain.shape[0])
                delta, p_val = perm_rdm_test(
                    bp_rdm[:n_common, :n_common],
                    ref_rdm[:n_common, :n_common],
                    brain[:n_common, :n_common]
                )
            else:
                delta, p_val = np.nan, np.nan

            sig = ("***" if p_val < 0.001 else "**" if p_val < 0.01
                   else "*" if p_val < 0.05 else "ns") if not np.isnan(p_val) else "N/A"

            rows.append({
                "roi":       roi,
                "layer":     best_layer,
                "rule_a":    "Burstprop",
                "rule_b":    ref_rule,
                "rho_a":     round(bp_rho, 5),
                "rho_b":     round(ref_rho, 5),
                "delta":     round(delta, 5) if not np.isnan(delta) else np.nan,
                "p_perm":    round(p_val, 4) if not np.isnan(p_val) else np.nan,
                "sig":       sig,
                "cohens_d":  round(d, 3) if not np.isnan(d) else np.nan,
            })
            msg = (f"  {roi} ({best_layer}) BP vs {ref_rule:20s}: "
                   f"rho={bp_rho:.4f} vs {ref_rho:.4f}  "
                   f"delta={delta:+.4f}  p={p_val:.4f} {sig}  d={d:.2f}"
                   if not np.isnan(d) else
                   f"  {roi} ({best_layer}) BP vs {ref_rule:20s}: "
                   f"rho={bp_rho:.4f} vs {ref_rho:.4f}  p={p_val:.4f} {sig}")
            print(msg)

    df = pd.DataFrame(rows)
    out = OUTPUTS_DIR / "burstprop_stats.csv"
    df.to_csv(str(out), index=False)
    print(f"  Saved: {out.name}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Burstprop Integration")
    print("=" * 60)

    # ── 1. Build Burstprop summary ────────────────────────────────────────────
    print("\n[1] Building Burstprop summary from seed RDMs...")
    bp_df = build_burstprop_summary()
    print(f"  {len(bp_df)} rows  (layers x ROIs)")
    print("\n  Best rho per ROI:")
    for roi in ROIS:
        layer, rho = best_layer(bp_df, roi)
        print(f"    {roi:4s}  {layer:5s}  rho={rho:.4f}" if rho is not None else f"    {roi}: no data")

    # ── 2. Merge with existing ────────────────────────────────────────────────
    print("\n[2] Merging with existing results...")
    combined_df = merge_with_existing(bp_df)

    # ── 3. Plots ──────────────────────────────────────────────────────────────
    print("\n[3] Creating plots...")
    plot_burstprop_heatmap(bp_df)
    plot_v1_bar(combined_df)
    plot_all_rois(combined_df)

    # ── 4. Stats ──────────────────────────────────────────────────────────────
    print(f"\n[4] Statistical tests (n_perm={N_PERM})...")
    stats_df = run_burstprop_stats(bp_df, combined_df)

    # ── 5. Summary for paper ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY FOR PAPER")
    print("=" * 60)
    print("\nBurstprop best-layer rho per ROI (mean across 3 seeds):")
    for roi in ROIS:
        layer, rho = best_layer(bp_df, roi)
        if rho is not None:
            print(f"  {roi:4s}  best={layer}  rho={rho:.4f}")

    if not stats_df.empty:
        print("\nBurstprop vs Random Weights (best Burstprop layer):")
        for _, r in stats_df[stats_df["rule_b"] == "Random Weights"].iterrows():
            print(f"  {r['roi']:4s}: Burstprop rho={r['rho_a']:.4f}, "
                  f"Random rho={r['rho_b']:.4f}, delta={r['delta']:+.4f}, "
                  f"p={r['p_perm']:.4f} {r['sig']}, d={r['cohens_d']}")

        print("\nBurstprop vs Backprop (best Burstprop layer):")
        for _, r in stats_df[stats_df["rule_b"] == "Backprop"].iterrows():
            print(f"  {r['roi']:4s}: Burstprop rho={r['rho_a']:.4f}, "
                  f"BP rho={r['rho_b']:.4f}, delta={r['delta']:+.4f}, "
                  f"p={r['p_perm']:.4f} {r['sig']}, d={r['cohens_d']}")

    print(f"\nAll outputs in: {OUTPUTS_DIR}")
    print("\nCAVEAT: Burstprop uses CIFAR10ConvNet (3 conv+1 FC, 32x32 input, 67.7% acc)")
    print("        vs ResNet-18 (224x224, ~93% acc) for all other rules.")
    print("        Architecture is not controlled — comparison is exploratory.")


if __name__ == "__main__":
    main()

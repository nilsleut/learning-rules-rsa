"""
stats_analysis_v2.py
=====================
Phase 3 (improved): Statistical Analysis of Learning Rules RSA Results.

Changes from v1:
    1. FDR correction (Benjamini-Hochberg) for multiple comparisons
    2. Cohen's d effect sizes from per-subject RSA distributions
    3. All labels/titles in English (consistency with paper)
    4. Random Weights baseline included in comparisons
    5. Summary table with FDR-adjusted p-values

Usage:
    py stats_analysis_v2.py

Outputs (in outputs/):
    permutation_results_fdr.csv  -- p-values with FDR correction
    permutation_forest_fdr.png   -- forest plot with FDR significance
    effect_sizes.csv             -- Cohen's d for all pairwise comparisons
    stats_summary_v2.csv         -- Final table for paper
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
FMRI_DIR    = BASE_DIR / "outputs_720"
SUBJECTS    = ["sub-01", "sub-02", "sub-03"]
ROIS        = ["V1", "V2", "LOC", "IT"]
N_PERM      = 1000
SEED        = 42
rng         = np.random.default_rng(SEED)

COLORS = {
    "Random Weights":     "#999999",
    "Backprop":           "#2E86AB",
    "Feedback Alignment": "#E84855",
    "Predictive Coding":  "#3BB273",
    "STDP":               "#F4A261",
}


# ── Utilities ──────────────────────────────────────────────────────────────────

def upper_tri(rdm):
    return rdm[np.triu_indices(rdm.shape[0], k=1)]

def rsa(rdm_a, rdm_b):
    n   = min(rdm_a.shape[0], rdm_b.shape[0])
    idx = np.triu_indices(n, k=1)
    r, _ = spearmanr(rdm_a[:n,:n][idx], rdm_b[:n,:n][idx])
    return float(r)

def load_fmri_rdm(roi, sub):
    p = FMRI_DIR / f"fmri_rdm_{roi}_{sub}.npy"
    return np.load(str(p)) if p.exists() else None

def mean_brain_rdm(roi):
    rdms = [load_fmri_rdm(roi, s) for s in SUBJECTS]
    rdms = [r for r in rdms if r is not None]
    if not rdms: return None
    n = min(r.shape[0] for r in rdms)
    return np.mean([r[:n,:n] for r in rdms], axis=0)


# ── FDR Correction ─────────────────────────────────────────────────────────────

def benjamini_hochberg(p_values):
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    n = len(p_values)
    if n == 0: return []
    sorted_idx = np.argsort(p_values)
    sorted_p   = np.array(p_values)[sorted_idx]
    adjusted   = np.zeros(n)
    cum_min    = 1.0
    for i in range(n-1, -1, -1):
        adj = sorted_p[i] * n / (i + 1)
        cum_min = min(cum_min, adj)
        adjusted[sorted_idx[i]] = min(cum_min, 1.0)
    return adjusted.tolist()


# ── Effect Size ────────────────────────────────────────────────────────────────

def cohens_d_from_subjects(rsa_df, rule_a, rule_b, roi, layer):
    """
    Compute Cohen's d using per-subject RSA scores.
    With only 3 subjects this is approximate — report with caveats.
    """
    sub_cols = ["rho_sub01", "rho_sub02", "rho_sub03"]

    def get_sub_vals(rule):
        row = rsa_df[(rsa_df["rule"]==rule) & (rsa_df["roi"]==roi)
                     & (rsa_df["layer"]==layer)]
        if row.empty: return []
        row = row.iloc[0]
        return [row[c] for c in sub_cols if c in row.index and not np.isnan(row[c])]

    a_vals = get_sub_vals(rule_a)
    b_vals = get_sub_vals(rule_b)

    if len(a_vals) < 2 or len(b_vals) < 2:
        return np.nan

    a, b = np.array(a_vals), np.array(b_vals)
    pooled_std = np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2)
    if pooled_std < 1e-10:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


# ── Permutation Test ───────────────────────────────────────────────────────────

def permutation_test(rdm_a, rdm_b, brain_rdm, n_perm=N_PERM):
    n   = min(rdm_a.shape[0], rdm_b.shape[0], brain_rdm.shape[0])
    idx = np.triu_indices(n, k=1)
    va  = rdm_a[:n,:n][idx]
    vb  = rdm_b[:n,:n][idx]
    vbr = brain_rdm[:n,:n][idx]

    r_a      = float(spearmanr(va, vbr)[0])
    r_b      = float(spearmanr(vb, vbr)[0])
    observed = r_a - r_b

    null = []
    for _ in range(n_perm):
        perm  = rng.permutation(len(vbr))
        vbr_p = vbr[perm]
        null.append(float(spearmanr(va, vbr_p)[0]) - float(spearmanr(vb, vbr_p)[0]))
    null  = np.array(null)
    p_val = np.mean(np.abs(null) >= np.abs(observed))
    return observed, p_val, null


def run_permutation_tests(model_rdms, rsa_df=None):
    """Pairwise permutation tests with FDR correction and effect sizes."""
    results    = []
    all_p      = []
    rules      = list(model_rdms.keys())
    layer_map  = {"V1": "Conv1", "V2": "Conv1", "LOC": "Conv3", "IT": "FC1"}

    for roi in ROIS:
        brain = mean_brain_rdm(roi)
        if brain is None: continue
        layer = layer_map[roi]
        print(f"\n  ROI: {roi} (Layer: {layer})")

        for rule_a, rule_b in combinations(rules, 2):
            if layer not in model_rdms.get(rule_a, {}): continue
            if layer not in model_rdms.get(rule_b, {}): continue

            rdm_a = model_rdms[rule_a][layer]
            rdm_b = model_rdms[rule_b][layer]
            delta, p, null = permutation_test(rdm_a, rdm_b, brain)

            # Effect size
            d = np.nan
            if rsa_df is not None:
                d = cohens_d_from_subjects(rsa_df, rule_a, rule_b, roi, layer)

            results.append({
                "roi":            roi,
                "layer":          layer,
                "rule_a":         rule_a,
                "rule_b":         rule_b,
                "r_a":            rsa(rdm_a[:720,:720], brain),
                "r_b":            rsa(rdm_b[:720,:720], brain),
                "delta":          round(delta, 5),
                "p_uncorrected":  round(p, 4),
                "cohens_d":       round(d, 3) if not np.isnan(d) else np.nan,
            })
            all_p.append(p)

            sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
            d_str = f"d={d:.2f}" if not np.isnan(d) else "d=N/A"
            print(f"    {rule_a:25s} vs {rule_b:25s}: "
                  f"Δ={delta:+.5f}  p={p:.4f} {sig}  {d_str}")

    # FDR correction
    fdr_adjusted = benjamini_hochberg(all_p) if all_p else []
    for i, row in enumerate(results):
        if i < len(fdr_adjusted):
            row["p_fdr"] = round(fdr_adjusted[i], 4)
            p_f = fdr_adjusted[i]
            row["sig_fdr"] = "***" if p_f<0.001 else "**" if p_f<0.01 else \
                             "*" if p_f<0.05 else "ns"
        else:
            row["p_fdr"]   = np.nan
            row["sig_fdr"] = "N/A"

    df = pd.DataFrame(results)

    # Report FDR impact
    if not df.empty:
        n_sig_raw = sum(1 for p in all_p if p < 0.05)
        n_sig_fdr = (df["sig_fdr"] != "ns").sum()
        print(f"\n  FDR correction: {n_sig_raw} → {n_sig_fdr} significant "
              f"(out of {len(all_p)} tests)")

    return df


# ── Load Model RDMs ───────────────────────────────────────────────────────────

def load_model_rdms():
    rdm_dir = OUTPUTS_DIR / "model_rdms"
    if not rdm_dir.exists():
        print("Model RDMs not found. Run learning_rules_v6.py first.")
        return None

    rules  = ["Random Weights", "Backprop", "Feedback Alignment",
              "Predictive Coding", "STDP"]
    layers = ["Conv1", "Conv2", "Conv3", "FC1"]

    model_rdms = {}
    for rule in rules:
        rule_key = rule.lower().replace(" ", "_")
        found = False
        for layer in layers:
            path = rdm_dir / f"rdm_{rule_key}_{layer}.npy"
            if path.exists():
                if rule not in model_rdms: model_rdms[rule] = {}
                model_rdms[rule][layer] = np.load(str(path))
                found = True
        if not found:
            print(f"  No RDMs found for: {rule}")

    return model_rdms


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_forest(df):
    """Forest plot with FDR-corrected significance markers."""
    rois = [r for r in ROIS if r in df["roi"].values]
    fig, axes = plt.subplots(1, len(rois), figsize=(4.5*len(rois), 6), sharey=True)
    if len(rois) == 1: axes = [axes]

    for ax, roi in zip(axes, rois):
        sub = df[df["roi"] == roi].copy()
        sub["label"] = sub["rule_a"].str[:4] + " vs " + sub["rule_b"].str[:4]
        sub = sub.sort_values("delta", ascending=True)

        colors = ["#2ecc71" if s != "ns" else "#e74c3c"
                  for s in sub.get("sig_fdr", sub.get("p_uncorrected", "ns"))]
        ax.barh(range(len(sub)), sub["delta"], color=colors, alpha=0.8)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub["label"], fontsize=7)
        ax.set_xlabel("Δρ (A − B)", fontsize=10)
        ax.set_title(f"ROI: {roi}", fontsize=11)

        for i, (_, row) in enumerate(sub.iterrows()):
            sig = row.get("sig_fdr", "ns")
            if sig != "ns":
                ax.text(row["delta"], i, f" {sig}", va="center",
                        fontsize=7, fontweight="bold")

    plt.suptitle("Pairwise Δρ with FDR correction (green = significant)", fontsize=12)
    plt.tight_layout()
    path = OUTPUTS_DIR / "permutation_forest_fdr.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


# ── Summary Table ──────────────────────────────────────────────────────────────

def make_summary_table(rsa_df, perm_df):
    """Final table for paper: RSA scores + FDR significance + effect size."""
    rules = ["Random Weights", "Backprop", "Feedback Alignment",
             "Predictive Coding", "STDP"]
    rules = [r for r in rules if r in rsa_df["rule"].values]

    # Fixed layer mapping — MUST match permutation test layer
    fixed_layer = {"V1": "Conv1", "V2": "Conv1", "LOC": "Conv3", "IT": "FC1"}

    rows = []
    for roi in ROIS:
        for rule in rules:
            layer = fixed_layer[roi]
            sub = rsa_df[(rsa_df["roi"] == roi) & (rsa_df["rule"] == rule)
                         & (rsa_df["layer"] == layer)]
            if sub.empty:
                # Fallback: any layer for this ROI
                sub = rsa_df[(rsa_df["roi"] == roi) & (rsa_df["rule"] == rule)]
            if sub.empty: continue
            best = sub.iloc[0]
            row  = {
                "ROI":   roi,
                "Rule":  rule,
                "Layer": best["layer"],
                "rho":   round(best["rho"], 4),
                "CI_lo": round(best.get("ci_lo", best["rho"]), 4),
                "CI_hi": round(best.get("ci_hi", best["rho"]), 4),
            }

            # Significance vs BP (FDR-corrected)
            if rule != "Backprop" and perm_df is not None:
                p_sub = perm_df[
                    (perm_df["roi"] == roi) &
                    (((perm_df["rule_a"]=="Backprop") & (perm_df["rule_b"]==rule)) |
                     ((perm_df["rule_a"]==rule) & (perm_df["rule_b"]=="Backprop")))
                ]
                if not p_sub.empty:
                    row["p_vs_BP_uncorr"] = round(p_sub.iloc[0]["p_uncorrected"], 4)
                    row["p_vs_BP_fdr"]    = round(p_sub.iloc[0]["p_fdr"], 4)
                    row["sig_vs_BP"]      = p_sub.iloc[0]["sig_fdr"]
                    row["d_vs_BP"]        = p_sub.iloc[0].get("cohens_d", np.nan)
            else:
                row["p_vs_BP_uncorr"] = "-"
                row["p_vs_BP_fdr"]    = "-"
                row["sig_vs_BP"]      = "-"
                row["d_vs_BP"]        = "-"

            rows.append(row)

    summary = pd.DataFrame(rows)
    path    = OUTPUTS_DIR / "stats_summary_v2.csv"
    summary.to_csv(str(path), index=False)
    print(f"\nSummary saved: {path.name}")
    print(summary.to_string())
    return summary


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Stats Analysis v2 (with FDR + effect sizes)\n")

    # Load RSA results
    rsa_csv = OUTPUTS_DIR / "rsa_results_cnn.csv"
    if not rsa_csv.exists():
        print(f"Not found: {rsa_csv}")
        print("Run learning_rules_v6.py first.")
        return
    rsa_df = pd.read_csv(str(rsa_csv))
    print(f"RSA results loaded: {len(rsa_df)} rows\n")

    # Load model RDMs
    print("Loading model RDMs...")
    model_rdms = load_model_rdms()
    if model_rdms is None:
        return

    # Permutation tests with FDR
    print(f"\nPermutation tests (N={N_PERM}) with FDR correction...")
    perm_df = run_permutation_tests(model_rdms, rsa_df)
    perm_df.to_csv(str(OUTPUTS_DIR / "permutation_results_fdr.csv"), index=False)

    # Plot
    if not perm_df.empty:
        plot_forest(perm_df)

    # Summary
    make_summary_table(rsa_df, perm_df)

    print("\nDone.")


if __name__ == "__main__":
    main()
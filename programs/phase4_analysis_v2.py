"""
phase4_analysis_v2.py
=====================
Phase 4 (improved): Filter Visualization + Partial RSA

Changes from v1:
    1. All labels/titles in English (paper consistency)
    2. Random Weights baseline included in partial RSA
    3. Consistent ρ values — partial RSA uses same RDM pipeline as main analysis
    4. Abstract-ready summary values printed at end
    5. Partial RSA table clearly annotated to avoid confusion with Table 2

Usage:
    py phase4_analysis_v2.py

Prerequisites:
    - learning_rules_v6.py must have run with weight/RDM export
    - outputs/model_rdms/ must exist
    - outputs_720/fmri_rdm_*.npy must exist
"""

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from scipy.stats import spearmanr, rankdata
from scipy.spatial.distance import pdist, squareform
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
FMRI_DIR    = BASE_DIR / "outputs_720"
THINGS_DIR  = Path(r"C:\Users\nilsl\Desktop\Projekte\RSA\Datensatz\images_THINGS\object_images")
SUBJECTS    = ["sub-01", "sub-02", "sub-03"]

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

def load_fmri_rdm(roi, sub):
    p = FMRI_DIR / f"fmri_rdm_{roi}_{sub}.npy"
    return np.load(str(p)) if p.exists() else None

def mean_brain_rdm(roi):
    rdms = [load_fmri_rdm(roi, s) for s in SUBJECTS]
    rdms = [r for r in rdms if r is not None]
    if not rdms: return None
    n = min(r.shape[0] for r in rdms)
    return np.mean([r[:n,:n] for r in rdms], axis=0)

def load_model_rdm(rule, layer):
    rule_key = rule.lower().replace(" ", "_")
    p = OUTPUTS_DIR / "model_rdms" / f"rdm_{rule_key}_{layer}.npy"
    return np.load(str(p)) if p.exists() else None

def load_stim_order():
    p = FMRI_DIR / "stim_order_sub-01.txt"
    with open(p) as f:
        return [l.strip() for l in f if l.strip()]

def find_img(stimulus):
    name    = stimulus.replace(".jpg", "")
    parts   = name.split("_")
    last    = parts[-1]
    concept = "_".join(parts[:-1]) if (len(parts)>1 and len(last)<=4
                                        and any(c.isdigit() for c in last)) else name
    for pat in [f"{concept}/{name}.jpg", f"{concept}/*.jpg"]:
        hits = sorted(THINGS_DIR.glob(pat))
        if hits: return hits[0]
    for folder in THINGS_DIR.iterdir():
        if folder.name.lower() == concept.lower():
            imgs = sorted(folder.glob("*.jpg"))
            if imgs: return imgs[0]
    return None


# ══════════════════════════════════════════════════════════════════════════════
# PART A — FILTER VISUALIZATION + GABOR ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def load_conv1_filters():
    rules   = ["Random Weights", "Backprop", "Feedback Alignment",
               "Predictive Coding", "STDP"]
    filters = {}
    for rule in rules:
        rule_key = rule.lower().replace(" ", "_")
        path     = OUTPUTS_DIR / f"model_weights_{rule_key}.pt"
        if not path.exists():
            print(f"  {rule}: {path.name} not found")
            continue
        state = torch.load(str(path), map_location="cpu", weights_only=False)
        for key, val in state.items():
            if "conv1" in key.lower() and "weight" in key.lower():
                if val.ndim == 4 and val.shape[1] == 3:
                    filters[rule] = val.numpy()
                    print(f"  {rule}: {val.shape} loaded")
                    break
    return filters


def visualize_filters(filters, n_show=16):
    if not filters: return
    rules  = list(filters.keys())
    fig, axes = plt.subplots(len(rules), n_show,
                             figsize=(n_show*1.1, len(rules)*1.3))
    if len(rules) == 1:
        axes = axes[np.newaxis, :]

    for ri, rule in enumerate(rules):
        w = filters[rule]
        axes[ri, 0].set_ylabel(rule, fontsize=8, rotation=0,
                                ha="right", va="center", labelpad=60)
        for ci in range(n_show):
            ax = axes[ri, ci]
            ax.axis("off")
            if ci < w.shape[0]:
                f   = np.transpose(w[ci], (1,2,0))
                f   = f - f.min()
                if f.max() > 0: f /= f.max()
                ax.imshow(f, interpolation="nearest")

    plt.suptitle("Conv1 filters per learning rule (first 16)", fontsize=12, y=1.01)
    plt.tight_layout()
    path = OUTPUTS_DIR / "filter_visualization.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


def gabor_score(w):
    """FFT peakedness as Gabor-likeness score."""
    scores = []
    for f in w:
        fg  = f.mean(0)
        fft = np.abs(np.fft.fftshift(np.fft.fft2(fg)))
        h, ww = fft.shape
        fft[h//2, ww//2] = 0
        scores.append(fft.max() / (fft.mean() + 1e-8))
    return np.array(scores)


def analyze_gabor(filters):
    if not filters: return None
    rows = []
    for rule, w in filters.items():
        s = gabor_score(w)
        rows.append({"rule": rule, "mean": s.mean(), "std": s.std()})
        print(f"  {rule:25s}: {s.mean():.2f} +/- {s.std():.2f}")
    df   = pd.DataFrame(rows).sort_values("mean", ascending=False)
    df.to_csv(str(OUTPUTS_DIR / "gabor_analysis.csv"), index=False)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(df["rule"], df["mean"], yerr=df["std"], capsize=4,
           color=[COLORS.get(r,"gray") for r in df["rule"]], alpha=0.85)
    ax.set_ylabel("Gabor-peakedness (higher = more V1-like)", fontsize=11)
    ax.set_title("Gabor-likeness of Conv1 filters", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(str(OUTPUTS_DIR / "gabor_plot.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: gabor_plot.png, gabor_analysis.csv")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PART B — PARTIAL RSA
# ══════════════════════════════════════════════════════════════════════════════

def compute_pixel_rdm(paths, n_max=720):
    tf = T.Compose([T.Resize(32), T.CenterCrop(32), T.ToTensor()])
    class DS(Dataset):
        def __init__(self, ps): self.ps = ps
        def __len__(self): return len(self.ps)
        def __getitem__(self, i):
            return tf(Image.open(self.ps[i]).convert("RGB")), i
    loader = DataLoader(DS(paths[:n_max]), batch_size=64,
                        shuffle=False, num_workers=0)
    feats  = []
    with torch.no_grad():
        for imgs, _ in loader:
            feats.append(imgs.view(imgs.size(0),-1).numpy())
    return squareform(pdist(np.concatenate(feats), metric="correlation"))


def partial_spearman(x, y, z):
    """Partial Spearman r(x,y | z) via residualization."""
    def residualize(a, b):
        ar = rankdata(a).astype(float)
        br = rankdata(b).astype(float)
        bc = br - br.mean()
        beta = np.dot(ar, bc) / (np.dot(bc, bc) + 1e-10)
        return ar - beta * br
    r, p = spearmanr(residualize(x,z), residualize(y,z))
    return float(r), float(p)


def run_partial_rsa(pixel_rdm):
    """
    Partial RSA controlling for pixel similarity.
    
    IMPORTANT: This uses the SAME model RDMs and brain RDMs as the main
    analysis. The standard ρ values here should match Table 2 in the paper.
    If they don't, there's a pipeline inconsistency to fix.
    """
    rules     = ["Random Weights", "Backprop", "Feedback Alignment",
                 "Predictive Coding", "STDP"]
    layer_map = {"V1": "Conv1", "V2": "Conv1", "LOC": "Conv3", "IT": "FC1"}
    rows      = []

    # Also load the main RSA results for cross-check
    main_csv = OUTPUTS_DIR / "rsa_results_cnn.csv"
    main_df  = pd.read_csv(str(main_csv)) if main_csv.exists() else None

    for roi in ["V1", "V2", "LOC", "IT"]:
        brain = mean_brain_rdm(roi)
        if brain is None: continue
        n = min(brain.shape[0], pixel_rdm.shape[0])
        print(f"\n  {roi}:")

        for rule in rules:
            layer = layer_map[roi]
            mrdm  = load_model_rdm(rule, layer)
            if mrdm is None: continue
            nm    = min(n, mrdm.shape[0])

            mv = upper_tri(mrdm[:nm,:nm])
            bv = upper_tri(brain[:nm,:nm])
            pv = upper_tri(pixel_rdm[:nm,:nm])

            r_std, _     = spearmanr(mv, bv)
            r_par, p_par = partial_spearman(mv, bv, pv)

            # Cross-check with main RSA
            xcheck = ""
            if main_df is not None:
                main_row = main_df[(main_df["rule"]==rule) & (main_df["roi"]==roi)
                                   & (main_df["layer"]==layer)]
                if not main_row.empty:
                    main_rho = main_row.iloc[0]["rho"]
                    diff = abs(r_std - main_rho)
                    if diff > 0.005:
                        xcheck = f"  ⚠ MISMATCH with Table 2 (Δ={diff:.4f})"
                    else:
                        xcheck = "  ✓ consistent"

            print(f"    {rule:25s}: ρ_std={r_std:.4f}  ρ_partial={r_par:.4f}  "
                  f"Δ={r_par-r_std:+.4f}  p={p_par:.4f}{xcheck}")

            rows.append({
                "roi": roi, "rule": rule, "layer": layer,
                "rho_std": round(r_std, 4),
                "rho_partial": round(r_par, 4),
                "p_partial": round(p_par, 4),
                "delta": round(r_par - r_std, 4),
            })

    df = pd.DataFrame(rows)
    df.to_csv(str(OUTPUTS_DIR / "partial_rsa_results.csv"), index=False)

    # Plot
    if not df.empty:
        rois  = [r for r in ["V1","V2","LOC","IT"] if r in df["roi"].values]
        rules_plot = [r for r in rules if r in df["rule"].values]
        x     = np.arange(len(rois))
        w     = 0.08
        n_rules = len(rules_plot)
        offs  = np.linspace(-(n_rules-1)*w/2, (n_rules-1)*w/2, n_rules)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        for ax_i, (col, title) in enumerate([
            ("rho_std",     "Standard RSA"),
            ("rho_partial", "Partial RSA (pixel similarity controlled)")
        ]):
            ax = axes[ax_i]
            for i, rule in enumerate(rules_plot):
                sub  = df[df["rule"]==rule].set_index("roi")
                vals = [sub.loc[roi, col] if roi in sub.index else np.nan
                        for roi in rois]
                ax.bar(x+offs[i], vals, w*0.9,
                       color=COLORS.get(rule,"gray"), alpha=0.85,
                       label=rule if ax_i==0 else "")
            ax.set_xticks(x); ax.set_xticklabels(rois, fontsize=11)
            ax.set_ylabel("Spearman ρ", fontsize=11)
            ax.set_title(title, fontsize=12)
            ax.axhline(0, color="black", lw=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        axes[0].legend(fontsize=8, ncol=2)
        plt.suptitle("Partial RSA: learning rule effects beyond pixel similarity",
                     fontsize=12)
        plt.tight_layout()
        plt.savefig(str(OUTPUTS_DIR / "partial_rsa_plot.png"), dpi=150,
                    bbox_inches="tight")
        plt.close()
        print(f"\n  Saved: partial_rsa_results.csv, partial_rsa_plot.png")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Phase 4 v2: Filter Visualization + Partial RSA\n")

    # Part A
    print("="*55 + "\nPART A — Filter Visualization\n" + "="*55)
    filters   = load_conv1_filters()
    visualize_filters(filters)
    gabor_df  = analyze_gabor(filters)

    # Part B
    print("\n" + "="*55 + "\nPART B — Partial RSA\n" + "="*55)
    stimuli   = load_stim_order()
    paths     = [p for p in [find_img(s) for s in stimuli] if p is not None]
    print(f"  {len(paths)} images loaded")

    print("\n  Computing pixel RDM...")
    pixel_rdm = compute_pixel_rdm(paths)
    print(f"  Pixel RDM: {pixel_rdm.shape}")

    partial_df = run_partial_rsa(pixel_rdm)

    # Summary for abstract
    if not partial_df.empty:
        print("\n" + "="*55)
        print("VALUES FOR ABSTRACT (use these, not Table 2 values)")
        print("="*55)
        print("\nNote: The abstract should cite the STANDARD RSA values")
        print("from the MAIN analysis (rsa_results_cnn.csv), NOT the")
        print("partial RSA values. Partial RSA is a control analysis.\n")

        for roi in ["V1", "V2", "LOC", "IT"]:
            sub = partial_df[partial_df["roi"]==roi]
            if sub.empty: continue
            best_std = sub.loc[sub["rho_std"].idxmax()]
            best_par = sub.loc[sub["rho_partial"].idxmax()]
            print(f"  {roi}:")
            print(f"    Standard:  best = {best_std['rule']} (ρ = {best_std['rho_std']:.4f})")
            print(f"    Partial:   best = {best_par['rule']} (ρ = {best_par['rho_partial']:.4f})")

        # STDP vs BP at V1
        print("\n  STDP advantage at V1 (before/after pixel control):")
        for metric, label in [("rho_std","Standard"), ("rho_partial","Partial")]:
            sub = partial_df[partial_df["roi"]=="V1"].set_index("rule")
            if "STDP" in sub.index and "Backprop" in sub.index:
                d = sub.loc["STDP",metric] - sub.loc["Backprop",metric]
                print(f"    {label:10s}: STDP - BP = {d:+.4f}")

        # Random baseline reference
        print("\n  Random Weights baseline (for context):")
        for roi in ["V1", "IT"]:
            sub = partial_df[(partial_df["roi"]==roi) & (partial_df["rule"]=="Random Weights")]
            if not sub.empty:
                print(f"    {roi}: ρ_std = {sub.iloc[0]['rho_std']:.4f}")

    print("\nPhase 4 v2 complete.")


if __name__ == "__main__":
    main()

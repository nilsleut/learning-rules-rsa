"""
training_dynamics_rsa.py
========================
RSA brain alignment at training checkpoints for all learning rules.

Phase 1 (current): BP + Random Weights, 1 seed, 10 milestones.
Phase 2: expand RULES and N_SEEDS below.

KAGGLE SETUP:
  1. Add fMRI dataset (learning-rules-fmri) and THINGS images (things-object-images)
  2. Upload this file to the kernel
  3. "Save & Run All" (commit) — results survive session end

Outputs in outputs/training_dynamics/:
  training_dynamics_results.csv
  checkpoints/{rule}_seed{s}_epoch{e}.pth
  rdms/{rule}_seed{s}_epoch{e}/rdm_{rule}_{layer}.npy
  plots/epoch_vs_rho_v1.png
  plots/epoch_vs_rho_all_rois.png
  plots/delta_rho_normalized.png
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset, Dataset
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import random

# ── Kaggle dataset slugs ───────────────────────────────────────────────────────
FMRI_DATASET   = "learning-rules-fmri"
THINGS_DATASET = "things-object-images"

ON_KAGGLE = Path("/kaggle/input").exists()

# ── Paths ──────────────────────────────────────────────────────────────────────
if ON_KAGGLE:
    BASE_DIR    = Path("/kaggle/working")
    FMRI_DIR    = Path(f"/kaggle/input/datasets/nilsleutenegger/{FMRI_DATASET}/outputs_720")
    THINGS_DIR  = Path(f"/kaggle/input/datasets/nilsleutenegger/{THINGS_DATASET}/images_THINGS/object_images")
    CIFAR_DIR   = Path("/kaggle/working/data")
else:
    BASE_DIR    = Path(__file__).parent
    FMRI_DIR    = BASE_DIR / "outputs_720"
    THINGS_DIR  = BASE_DIR / "images_THINGS" / "object_images"
    CIFAR_DIR   = BASE_DIR / "data"

OUT_DIR   = BASE_DIR / "outputs" / "training_dynamics"
CKPT_DIR  = OUT_DIR / "checkpoints"
RDM_DIR   = OUT_DIR / "rdms"
PLOT_DIR  = OUT_DIR / "plots"

SUBJECTS = ["sub-01", "sub-02", "sub-03"]
ROIS     = ["V1", "V2", "V3", "V4", "LOC", "IT"]
LAYERS   = ["Conv1", "Conv2", "Conv3", "FC1"]

# ── Experiment config ──────────────────────────────────────────────────────────
MILESTONES = [0, 1, 5, 10, 25, 50, 100, 200, 300, 500]
N_EPOCHS   = 500
IMG_SIZE   = 224

# Phase 1: BP + Random, 1 seed. Expand for Phase 2.
RULES = ['Random Weights','Backprop','Feedback Alignment','Predictive Coding','STDP']
N_SEEDS = 5
SEEDS  = [42]

BATCH    = 128
LR       = 1e-3
N_CIFAR  = 8000
C1, C2, C3 = 32, 64, 128
FC1_DIM  = 512
N_CLS    = 10
FEAT_SIZE = 4
FC1_IN   = C3 * FEAT_SIZE * FEAT_SIZE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

COLORS = {
    "Random Weights": "#999999",
    "Backprop":       "#2E86AB",
    "Feedback Alignment": "#E84855",
    "Predictive Coding":  "#3BB273",
    "STDP":               "#F4A261",
    "Burstprop":          "#9B59B6",
}


# ══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURES
# ══════════════════════════════════════════════════════════════════════════════

def make_conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )

def _pool_for_fc(c3):
    return F.adaptive_avg_pool2d(c3, FEAT_SIZE)


class BP_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = make_conv_block(3,  C1)
        self.conv2 = make_conv_block(C1, C2)
        self.conv3 = make_conv_block(C2, C3)
        self.fc1   = nn.Linear(FC1_IN, FC1_DIM)
        self.fc2   = nn.Linear(FC1_DIM, N_CLS)
        self.drop  = nn.Dropout(0.3)

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        return self.fc2(self.drop(F.relu(self.fc1(x.view(x.size(0), -1)))))

    def get_features(self, x):
        with torch.no_grad():
            c1 = self.conv1(x)
            c2 = self.conv2(c1)
            c3 = self.conv3(c2)
            h1 = F.relu(self.fc1(_pool_for_fc(c3).view(c3.size(0), -1)))
        return c1.mean([2, 3]), c2.mean([2, 3]), c3.mean([2, 3]), h1

# Random Weights reuses BP_CNN architecture (untrained)
Random_CNN = BP_CNN


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def make_bp_optimizer(model):
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_EPOCHS)
    return opt, sched


def train_one_epoch_bp(model, loader, opt, sched):
    model.train()
    tl, tc, tn = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            tl += loss.item()
            tc += (model(x).argmax(1) == y).sum().item()
            tn += y.size(0)
    sched.step()
    model.eval()
    return tl / len(loader), tc / tn


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def get_cifar_loader(seed):
    tf = T.Compose([
        T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4),
        T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    full = torchvision.datasets.CIFAR10(str(CIFAR_DIR), train=True,
                                         download=True, transform=tf)
    idx  = torch.randperm(len(full))[:N_CIFAR].tolist()
    return DataLoader(Subset(full, idx), batch_size=BATCH,
                      shuffle=True, num_workers=2, drop_last=True)


def load_stim_order(sub="sub-01"):
    p = FMRI_DIR / f"stim_order_{sub}.txt"
    with open(p) as f:
        return [l.strip() for l in f if l.strip()]


def find_img(stimulus):
    name    = stimulus.replace(".jpg", "")
    parts   = name.split("_")
    last    = parts[-1]
    concept = "_".join(parts[:-1]) if (len(parts) > 1 and len(last) <= 4
                                        and any(c.isdigit() for c in last)) else name
    for pat in [f"{concept}/{name}.jpg", f"{concept}/*.jpg"]:
        hits = sorted(THINGS_DIR.glob(pat))
        if hits:
            return hits[0]
    for folder in THINGS_DIR.iterdir():
        if folder.name.lower() == concept.lower():
            imgs = sorted(folder.glob("*.jpg"))
            if imgs:
                return imgs[0]
    return None


class ImgDS(Dataset):
    def __init__(self, paths, t):
        self.paths, self.t = paths, t

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.t(Image.open(self.paths[i]).convert("RGB")), i


def get_things_paths():
    stimuli = load_stim_order("sub-01")
    paths   = [p for p in [find_img(s) for s in stimuli] if p is not None]
    print(f"  THINGS: {len(paths)}/{len(stimuli)} images found")
    return paths


def get_things_transform():
    return T.Compose([
        T.Resize(IMG_SIZE), T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# RSA PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def compute_rdm(f):
    f = f.detach().cpu().numpy() if torch.is_tensor(f) else np.array(f)
    return squareform(pdist(f, metric="correlation"))


def rsa_score(a, b):
    n   = min(a.shape[0], b.shape[0])
    idx = np.triu_indices(n, k=1)
    r, p = spearmanr(a[:n, :n][idx], b[:n, :n][idx])
    return float(r), float(p)


def load_fmri_rdm(roi, sub):
    p = FMRI_DIR / f"fmri_rdm_{roi}_{sub}.npy"
    return np.load(str(p)) if p.exists() else None


def extract_features(model, paths, transform):
    loader = DataLoader(ImgDS(paths, transform), batch_size=64,
                        shuffle=False, num_workers=2)
    c1s, c2s, c3s, h1s = [], [], [], []
    model.eval()
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(DEVICE)
            c1, c2, c3, h1 = model.get_features(imgs)
            def np_(t):
                return t.cpu().numpy() if torch.is_tensor(t) else np.array(t)
            c1s.append(np_(c1)); c2s.append(np_(c2))
            c3s.append(np_(c3)); h1s.append(np_(h1))
    return (np.concatenate(c1s), np.concatenate(c2s),
            np.concatenate(c3s), np.concatenate(h1s))


def compute_rsa_rows(feats, rule, seed_idx, epoch):
    """
    Compute per-subject RSA rows for all layers x all ROIs at one epoch.
    Returns list of dicts: {rule, layer, roi, subject, seed_idx, epoch, rho, pval}
    """
    rdms  = [compute_rdm(f) for f in feats]
    rows  = []
    for layer_idx, layer_name in enumerate(LAYERS):
        for roi in ROIS:
            for sub in SUBJECTS:
                brain = load_fmri_rdm(roi, sub)
                if brain is None:
                    continue
                rho, pval = rsa_score(rdms[layer_idx], brain)
                rows.append({
                    "rule":     rule,
                    "layer":    layer_name,
                    "roi":      roi,
                    "subject":  sub,
                    "seed_idx": seed_idx,
                    "epoch":    epoch,
                    "rho":      round(rho, 6),
                    "pval":     round(pval, 6),
                })
    return rows, rdms


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT + CSV UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def rule_key(rule):
    return rule.lower().replace(" ", "_")


def save_checkpoint(model, rule, seed_idx, epoch):
    path = CKPT_DIR / f"{rule_key(rule)}_seed{seed_idx}_epoch{epoch}.pth"
    torch.save(model.state_dict(), str(path))


def save_rdms(rdms, rule, seed_idx, epoch):
    d = RDM_DIR / f"{rule_key(rule)}_seed{seed_idx}_epoch{epoch}"
    d.mkdir(parents=True, exist_ok=True)
    for layer_name, rdm in zip(LAYERS, rdms):
        np.save(str(d / f"rdm_{rule_key(rule)}_{layer_name}.npy"), rdm)


def save_partial(all_rows):
    df = pd.DataFrame(all_rows)
    df.to_csv(str(OUT_DIR / "training_dynamics_results_partial.csv"), index=False)


def already_done(all_rows, rule, seed_idx, epoch):
    """Check if this (rule, seed, epoch) was already extracted."""
    for r in all_rows:
        if r["rule"] == rule and r["seed_idx"] == seed_idx and r["epoch"] == epoch:
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def _best_layer_rho(df, rule, roi, epoch):
    sub = df[(df["rule"] == rule) & (df["roi"] == roi) & (df["epoch"] == epoch)]
    if sub.empty:
        return np.nan
    return sub.groupby("layer")["rho"].mean().max()


def plot_epoch_vs_rho_v1(df):
    """Main plot: Epoch (log) vs V1 rho, one line per rule."""
    rules   = [r for r in df["rule"].unique() if r in COLORS]
    epochs  = sorted(df["epoch"].unique())

    fig, ax = plt.subplots(figsize=(9, 5))
    for rule in rules:
        means, sems = [], []
        for ep in epochs:
            sub = df[(df["rule"] == rule) & (df["roi"] == "V1") & (df["epoch"] == ep)]
            if sub.empty:
                means.append(np.nan); sems.append(np.nan)
                continue
            # best layer at this epoch (mean across subjects)
            best = sub.groupby("layer")["rho"].mean().max()
            # SD across subjects at best layer
            best_layer = sub.groupby("layer")["rho"].mean().idxmax()
            sd = sub[sub["layer"] == best_layer]["rho"].std()
            means.append(float(best)); sems.append(float(sd))

        x = [e + 0.5 for e in epochs]  # avoid log(0)
        ax.plot(x, means, "o-", color=COLORS.get(rule, "#333333"),
                label=rule, linewidth=2, markersize=5)
        ax.fill_between(x,
                         [m - s for m, s in zip(means, sems)],
                         [m + s for m, s in zip(means, sems)],
                         color=COLORS.get(rule, "#333333"), alpha=0.15)

    ax.set_xscale("log")
    ax.set_xticks([e + 0.5 for e in epochs])
    ax.set_xticklabels([str(e) for e in epochs], rotation=45, fontsize=9)
    ax.set_xlabel("Epoch (log scale)", fontsize=11)
    ax.set_ylabel("Spearman rho (best layer, mean subjects)", fontsize=11)
    ax.set_title("V1 brain alignment across training — does learning degrade it?", fontsize=12)
    ax.axhline(0, color="black", lw=0.5)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = PLOT_DIR / "epoch_vs_rho_v1.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def plot_epoch_vs_rho_all_rois(df):
    """6-panel grid: one subplot per ROI."""
    rules  = [r for r in df["rule"].unique() if r in COLORS]
    epochs = sorted(df["epoch"].unique())
    rois   = [r for r in ROIS if r in df["roi"].values]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=False)
    axes = axes.flatten()

    for ax, roi in zip(axes, rois):
        for rule in rules:
            means = []
            for ep in epochs:
                sub = df[(df["rule"] == rule) & (df["roi"] == roi) & (df["epoch"] == ep)]
                if sub.empty:
                    means.append(np.nan)
                    continue
                means.append(float(sub.groupby("layer")["rho"].mean().max()))
            x = [e + 0.5 for e in epochs]
            ax.plot(x, means, "o-", color=COLORS.get(rule, "#333333"),
                    label=rule, linewidth=1.5, markersize=4)
        ax.set_xscale("log")
        ax.set_xticks([e + 0.5 for e in epochs])
        ax.set_xticklabels([str(e) for e in epochs], rotation=45, fontsize=7)
        ax.set_title(roi, fontsize=11, fontweight="bold")
        ax.axhline(0, color="black", lw=0.5)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Spearman rho (best layer)", fontsize=10)
    axes[3].set_ylabel("Spearman rho (best layer)", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("fMRI alignment across training epochs — all ROIs", fontsize=13, y=1.01)
    plt.tight_layout()
    out = PLOT_DIR / "epoch_vs_rho_all_rois.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def plot_delta_rho_normalized(df):
    """Delta rho relative to epoch 0 (normalized degradation curve)."""
    rules  = [r for r in df["rule"].unique() if r in COLORS]
    epochs = sorted(df["epoch"].unique())
    rois   = [r for r in ROIS if r in df["roi"].values]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
    axes = axes.flatten()

    for ax, roi in zip(axes, rois):
        for rule in rules:
            means = {}
            for ep in epochs:
                sub = df[(df["rule"] == rule) & (df["roi"] == roi) & (df["epoch"] == ep)]
                if sub.empty:
                    continue
                means[ep] = float(sub.groupby("layer")["rho"].mean().max())

            if 0 not in means or np.isnan(means[0]) or means[0] == 0:
                continue
            baseline = means[0]
            eps_plot = [e for e in epochs if e in means]
            deltas   = [means[e] - baseline for e in eps_plot]
            x = [e + 0.5 for e in eps_plot]
            ax.plot(x, deltas, "o-", color=COLORS.get(rule, "#333333"),
                    label=rule, linewidth=1.5, markersize=4)

        ax.set_xscale("log")
        ax.set_xticks([e + 0.5 for e in epochs])
        ax.set_xticklabels([str(e) for e in epochs], rotation=45, fontsize=7)
        ax.set_title(roi, fontsize=11, fontweight="bold")
        ax.axhline(0, color="black", lw=1, linestyle="--", label="epoch 0 baseline")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("delta rho = rho(epoch) - rho(epoch=0)", fontsize=9)
    axes[3].set_ylabel("delta rho = rho(epoch) - rho(epoch=0)", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Training-induced change in fMRI alignment (delta from epoch 0)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    out = PLOT_DIR / "delta_rho_normalized.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def generate_all_plots(all_rows):
    if not all_rows:
        return
    df = pd.DataFrame(all_rows)
    print("\nGenerating plots...")
    plot_epoch_vs_rho_v1(df)
    plot_epoch_vs_rho_all_rois(df)
    plot_delta_rho_normalized(df)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    for d in [OUT_DIR, CKPT_DIR, RDM_DIR, PLOT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Training Dynamics RSA — Phase 1")
    print(f"  Rules:      {RULES}")
    print(f"  Seeds:      {SEEDS}")
    print(f"  Milestones: {MILESTONES}")
    print(f"  ROIs:       {ROIS}")
    print(f"  Device:     {DEVICE}\n")

    # Resume from partial CSV if it exists
    partial_csv = OUT_DIR / "training_dynamics_results_partial.csv"
    if partial_csv.exists():
        all_rows = pd.read_csv(str(partial_csv)).to_dict("records")
        print(f"Resuming from partial CSV: {len(all_rows)} rows already saved\n")
    else:
        all_rows = []

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading THINGS image paths...")
    paths       = get_things_paths()
    tf_things   = get_things_transform()

    # ── Main loop ─────────────────────────────────────────────────────────────
    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n{'='*60}")
        print(f"SEED {seed_idx}/{len(SEEDS)-1}  (seed={seed})")
        print(f"{'='*60}")

        cifar_loader = get_cifar_loader(seed)

        for rule in RULES:
            print(f"\n--- Rule: {rule} ---")
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            model = BP_CNN().to(DEVICE)
            model.eval()

            if rule != "Random Weights":
                opt, sched = make_bp_optimizer(model)

            for epoch in range(N_EPOCHS + 1):
                if epoch not in MILESTONES:
                    if rule != "Random Weights":
                        loss, acc = train_one_epoch_bp(model, cifar_loader, opt, sched)
                    continue

                # ── Milestone reached ──────────────────────────────────────────
                if already_done(all_rows, rule, seed_idx, epoch):
                    print(f"  epoch {epoch:4d}: skip (already in partial CSV)")
                    if rule == "Random Weights":
                        break
                    continue

                print(f"  epoch {epoch:4d}: extracting features...", end=" ", flush=True)
                feats = extract_features(model, paths, tf_things)
                rows, rdms = compute_rsa_rows(feats, rule, seed_idx, epoch)
                all_rows.extend(rows)

                # V1 summary line
                v1_rows = [r for r in rows if r["roi"] == "V1"]
                if v1_rows:
                    by_layer = {}
                    for r in v1_rows:
                        by_layer.setdefault(r["layer"], []).append(r["rho"])
                    best_layer = max(by_layer, key=lambda l: np.mean(by_layer[l]))
                    best_rho   = np.mean(by_layer[best_layer])
                    print(f"V1 best={best_layer} rho={best_rho:.4f}")

                save_checkpoint(model, rule, seed_idx, epoch)
                save_rdms(rdms, rule, seed_idx, epoch)
                save_partial(all_rows)

                if rule == "Random Weights":
                    break  # no training for random baseline

                if epoch < N_EPOCHS:
                    loss, acc = train_one_epoch_bp(model, cifar_loader, opt, sched)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── Final CSV ─────────────────────────────────────────────────────────────
    final_df = pd.DataFrame(all_rows)
    out_csv  = OUT_DIR / "training_dynamics_results.csv"
    final_df.to_csv(str(out_csv), index=False)
    print(f"\nSaved: {out_csv}  ({len(final_df)} rows)")

    # ── Plots ─────────────────────────────────────────────────────────────────
    generate_all_plots(all_rows)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if not final_df.empty:
        summary = (final_df
                   .groupby(["rule", "epoch", "roi"])["rho"]
                   .mean()
                   .unstack("roi")
                   .round(4))
        print("\nMean rho (best layer) per rule x epoch x ROI:")
        for (rule, epoch), row in summary.iterrows():
            v1  = row.get("V1", np.nan)
            loc = row.get("LOC", np.nan)
            it  = row.get("IT", np.nan)
            print(f"  {rule:20s}  ep={epoch:4d}  V1={v1:.4f}  LOC={loc:.4f}  IT={it:.4f}")

    print(f"\nAll outputs: {OUT_DIR}")
    print("\nTo expand to all rules (Phase 2):")
    print("  Set RULES = ['Random Weights','Backprop','Feedback Alignment','Predictive Coding','STDP']")
    print("  Set N_SEEDS = 5")


if __name__ == "__main__":
    main()

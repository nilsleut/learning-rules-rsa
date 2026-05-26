"""
burstprop_rsa_kaggle.py  —  paste entire file into ONE Kaggle cell and run.

Kaggle setup:
  1. Settings → Accelerator → GPU T4 (or P100)
  2. Add datasets: "learning-rules-fmri" and "things-object-images"
  3. Paste this entire script into a cell and click Run

OUTPUT FORMAT (compatible with learning_rules_v8_kaggle.py):
  /kaggle/working/outputs/model_rdms/seed_{i}/rdm_burstprop_{layer}.npy
  /kaggle/working/outputs/model_rdms/rdm_burstprop_{layer}.npy   (mean)
  /kaggle/working/outputs/rsa_results_burstprop.csv

BACKUP:
  After each seed a ZIP is written to /kaggle/working/backup_after_seedN.zip
  so partial results survive even if the session dies.
"""

# ── Step 1: clone burstprop repo before any burstprop imports ─────────────────
import subprocess, sys, shutil
from pathlib import Path

ON_KAGGLE = Path("/kaggle").exists()

if ON_KAGGLE:
    _bp = Path("/kaggle/working/burstprop")
    if not (_bp / "networks.py").exists():
        if _bp.exists():
            shutil.rmtree(str(_bp))
        print("Cloning burstprop repo...")
        subprocess.run(
            ["git", "clone", "https://github.com/jordan-g/Burstprop", str(_bp)],
            check=True
        )
        print(f"  networks.py present: {(_bp / 'networks.py').exists()}")
    sys.path.insert(0, str(_bp))
    for _mod in list(sys.modules.keys()):
        if _mod in ("networks", "layers", "layers_imagenet", "networks_imagenet"):
            del sys.modules[_mod]
else:
    _local = Path(__file__).parent / "burstprop" if "__file__" in dir() else Path("burstprop")
    if str(_local) not in sys.path:
        sys.path.insert(0, str(_local))

# ── Step 2: PyTorch 2.x compatibility patches ─────────────────────────────────
try:
    import layers as _bp_layers
    _bp_layers.use_cudnn = False
except Exception:
    pass

try:
    import layers as _l
    def _output_update_no_bias(self, lr, momentum=0, weight_decay=0):
        self.delta_weight = -lr * self.grad_weight + momentum * self.delta_weight
        self.weight += self.delta_weight - weight_decay * self.weight
        if self.weight_fa_learning:
            self.delta_weight_fa = -lr * self.grad_weight_fa + momentum * self.delta_weight_fa
            self.weight_fa += self.delta_weight_fa - weight_decay * self.weight_fa
    _l.OutputLayer.update_weights = _output_update_no_bias
except Exception:
    pass

try:
    import networks as _chk; del _chk
    print("OK: burstprop importable")
except ImportError as e:
    raise ImportError(
        f"Cannot import burstprop (networks.py). "
        f"sys.path[0]={sys.path[0]}  "
        f"networks.py exists: {Path(sys.path[0]) / 'networks.py'}"
    ) from e

# ── Step 3: all other imports ─────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from PIL import Image
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
FMRI_DATASET   = "learning-rules-fmri"
THINGS_DATASET = "things-object-images"

OUTPUTS_DIR = Path("/kaggle/working/outputs") if ON_KAGGLE else Path("results/burstprop")
FMRI_DIR    = Path(f"/kaggle/input/datasets/nilsleutenegger/{FMRI_DATASET}/outputs_720") if ON_KAGGLE else \
              Path(r"C:\Users\nilsl\Desktop\Projekte\learning-rules-rsa\outputs_720")
THINGS_DIR  = Path(f"/kaggle/input/datasets/nilsleutenegger/{THINGS_DATASET}/images_THINGS/object_images") if ON_KAGGLE else \
              Path(r"C:\Users\nilsl\Desktop\Projekte\RSA\Datensatz\images_THINGS\object_images")
CIFAR_DIR   = Path("/kaggle/working/data") if ON_KAGGLE else Path("burstprop/Data")

SUBJECTS = ["sub-01", "sub-02", "sub-03"]
ROIS     = ["V1", "V2", "V3", "V4", "LOC", "IT"]

# ── Hyperparameters ────────────────────────────────────────────────────────────
N_EPOCHS   = 500
BATCH      = 32
SEEDS      = [0, 1, 2]
N_SEEDS    = len(SEEDS)
IMG_SIZE   = 32
NUM_WORKERS = 2 if ON_KAGGLE else 0

HIDDEN_LR        = 0.01
OUTPUT_LR        = 0.01
MOMENTUM         = 0.9
WEIGHT_DECAY     = 1e-6
RECURRENT_LR     = 0.0001
P_BASELINE       = 0.2
WEIGHT_FA_STD    = 1.0
WEIGHT_R_RANGE   = 0.01
WEIGHT_FA_LEARNING = True
RECURRENT_INPUT    = True
WEIGHT_R_LEARNING  = True

TARGET_WRONG = 4.0 / 9.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

def build_model():
    from networks import CIFAR10ConvNet
    return CIFAR10ConvNet(
        input_channels=3,
        p_baseline=P_BASELINE,
        weight_fa_std=WEIGHT_FA_STD,
        weight_r_range=WEIGHT_R_RANGE,
        weight_fa_learning=WEIGHT_FA_LEARNING,
        recurrent_input=RECURRENT_INPUT,
        weight_r_learning=WEIGHT_R_LEARNING,
        device=DEVICE,
    ).to(DEVICE)


def get_lr_list():
    return [HIDDEN_LR] * 4 + [OUTPUT_LR]


def train_epoch(net, loader, lr_list):
    net.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        t = TARGET_WRONG + (1.0 - TARGET_WRONG) * F.one_hot(targets, num_classes=10).float()
        outputs = net(inputs)
        loss = net.loss(outputs, t)
        net.backward(t)
        net.update_weights(lr=lr_list, momentum=MOMENTUM,
                           weight_decay=WEIGHT_DECAY, recurrent_lr=RECURRENT_LR)
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total   += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return total_loss / len(loader), correct / total


def get_features(net, imgs):
    with torch.no_grad():
        net(imgs)
        c1 = net.feature_layers[0].e.mean([2, 3]).cpu().numpy()
        c2 = net.feature_layers[1].e.mean([2, 3]).cpu().numpy()
        c3 = net.feature_layers[2].e.mean([2, 3]).cpu().numpy()
        h1 = net.classification_layers[0].e.cpu().numpy()
    return c1, c2, c3, h1


# ══════════════════════════════════════════════════════════════════════════════
# THINGS IMAGE LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_stim_order(sub="sub-01"):
    p = FMRI_DIR / f"stim_order_{sub}.txt"
    with open(p) as f:
        return [l.strip() for l in f if l.strip()]


def find_img(stimulus):
    name  = stimulus.replace(".jpg", "")
    parts = name.split("_")
    last  = parts[-1]
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
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB")), i


# ══════════════════════════════════════════════════════════════════════════════
# RSA
# ══════════════════════════════════════════════════════════════════════════════

def compute_rdm(features):
    return squareform(pdist(np.array(features), metric="correlation"))


def load_fmri_rdm(roi, sub):
    p = FMRI_DIR / f"fmri_rdm_{roi}_{sub}.npy"
    return np.load(str(p)) if p.exists() else None


def rsa_score(rdm_a, rdm_b):
    n   = min(rdm_a.shape[0], rdm_b.shape[0])
    idx = np.triu_indices(n, k=1)
    r, p = spearmanr(rdm_a[:n, :n][idx], rdm_b[:n, :n][idx])
    return float(r), float(p)


def run_rsa(net, paths, tf_things, seed_idx):
    loader = DataLoader(ImgDS(paths, tf_things), batch_size=64,
                        shuffle=False, num_workers=NUM_WORKERS)
    c1s, c2s, c3s, h1s = [], [], [], []
    net.eval()
    for imgs, _ in loader:
        c1, c2, c3, h1 = get_features(net, imgs.to(DEVICE))
        c1s.append(c1); c2s.append(c2); c3s.append(c3); h1s.append(h1)

    layer_names = ["Conv1", "Conv2", "Conv3", "FC1"]
    rdms = [compute_rdm(np.concatenate(a)) for a in [c1s, c2s, c3s, h1s]]

    rows = []
    for layer_name, rdm in zip(layer_names, rdms):
        for roi in ROIS:
            for sub in SUBJECTS:
                fmri_rdm = load_fmri_rdm(roi, sub)
                if fmri_rdm is None:
                    continue
                rho, pval = rsa_score(rdm, fmri_rdm)
                rows.append({"rule": "Burstprop", "layer": layer_name, "roi": roi,
                              "subject": sub, "rho": round(rho, 5), "pval": round(pval, 5),
                              "seed_idx": seed_idx})
        print(f"  {layer_name}: RDM {rdm.shape}")
    return rows, rdms


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nBurstprop RSA — device={DEVICE}, seeds={SEEDS}\n")

    # CIFAR-10
    print("Loading CIFAR-10...")
    tf_train = T.Compose([
        T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
        T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_set = torchvision.datasets.CIFAR10(str(CIFAR_DIR), train=True,
                                              download=True, transform=tf_train)
    train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True,
                              num_workers=NUM_WORKERS, drop_last=False)
    print(f"  {len(train_set)} samples, {len(train_loader)} batches/epoch")

    # THINGS images
    print("\nLoading THINGS image paths...")
    stimuli = load_stim_order("sub-01")
    paths   = [p for p in [find_img(s) for s in stimuli] if p is not None]
    print(f"  {len(paths)}/{len(stimuli)} images found")
    tf_things = T.Compose([
        T.Resize(IMG_SIZE), T.CenterCrop(IMG_SIZE),
        T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    all_rows  = []
    seed_rdms = {ln: [] for ln in ["Conv1", "Conv2", "Conv3", "FC1"]}
    lr_list   = get_lr_list()

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n{'='*55}\nSEED {seed_idx+1}/{N_SEEDS}  (seed={seed})\n{'='*55}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Check for existing checkpoint → skip training
        ckpt_path = OUTPUTS_DIR / f"checkpoint_burstprop_seed{seed_idx}.pth"
        if ckpt_path.exists():
            print(f"  Checkpoint found, loading {ckpt_path.name}...")
            net = torch.load(str(ckpt_path), map_location=DEVICE)
        else:
            net = build_model()
            for epoch in range(N_EPOCHS):
                loss, acc = train_epoch(net, train_loader, lr_list)
                if (epoch + 1) % 50 == 0 or epoch == 0:
                    print(f"  Epoch {epoch+1:3d}/{N_EPOCHS}: loss={loss:.4f}  acc={acc:.3f}")
            torch.save(net, str(ckpt_path))
            print(f"  Checkpoint saved: {ckpt_path.name}")

        # RSA
        print(f"\n  Extracting activations for {len(paths)} THINGS images...")
        rdm_dir_seed = OUTPUTS_DIR / "model_rdms" / f"seed_{seed_idx}"
        rdm_dir_seed.mkdir(parents=True, exist_ok=True)

        rows, rdms = run_rsa(net, paths, tf_things, seed_idx)
        all_rows.extend(rows)

        layer_names = ["Conv1", "Conv2", "Conv3", "FC1"]
        for ln, rdm in zip(layer_names, rdms):
            rdm_path = rdm_dir_seed / f"rdm_burstprop_{ln}.npy"
            np.save(str(rdm_path), rdm)
            seed_rdms[ln].append(rdm)
            print(f"  Saved {rdm_path.name}  shape={rdm.shape}")

        # Partial CSV
        pd.DataFrame(all_rows).to_csv(
            str(OUTPUTS_DIR / "rsa_results_burstprop_partial.csv"), index=False)

        # ═══ ZIP BACKUP AFTER EVERY SEED ═══
        zip_name = f"/kaggle/working/backup_after_seed{seed_idx}"
        shutil.make_archive(zip_name, "zip", str(OUTPUTS_DIR))
        print(f"\n  ★ BACKUP: {zip_name}.zip ★\n")

        del net
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Mean RDMs
    print(f"\n{'='*55}\nMean RDMs across {N_SEEDS} seeds\n{'='*55}")
    rdm_dir = OUTPUTS_DIR / "model_rdms"
    for ln in ["Conv1", "Conv2", "Conv3", "FC1"]:
        mean_rdm = np.mean(seed_rdms[ln], axis=0)
        np.save(str(rdm_dir / f"rdm_burstprop_{ln}.npy"), mean_rdm)
        print(f"  rdm_burstprop_{ln}.npy  shape={mean_rdm.shape}")

    # Final CSV
    df = pd.DataFrame(all_rows)
    df.to_csv(str(OUTPUTS_DIR / "rsa_results_burstprop.csv"), index=False)
    print(f"\nRSA results: {len(df)} rows")

    # Summary
    print(f"\n{'='*55}\nSUMMARY\n{'='*55}")
    best = df.groupby(["roi", "layer"])["rho"].mean().reset_index()
    best = best.loc[best.groupby("roi")["rho"].idxmax()].sort_values("rho", ascending=False)
    print("Best layer per ROI (mean rho across seeds & subjects):")
    for _, row in best.iterrows():
        print(f"  {row['roi']:4s}  {row['layer']:5s}  rho={row['rho']:.4f}")

    # Final ZIP
    shutil.make_archive("/kaggle/working/burstprop_final", "zip", str(OUTPUTS_DIR))
    print(f"\n★ FINAL ZIP: /kaggle/working/burstprop_final.zip ★")
    print(f"Done. Outputs in: {OUTPUTS_DIR}")


main()
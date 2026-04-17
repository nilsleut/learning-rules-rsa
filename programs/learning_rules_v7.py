"""
learning_rules_v6.py
=====================
Verbesserungen gegenueber v5:
    1. Random-Weights Baseline (untrained CNN)
    2. Subject-Level Analyse (Rankings pro Subject)
    3. Best-Layer-per-ROI Robustheitscheck
    4. Multiple-Comparisons-Korrektur (FDR)
    5. Konsistente Figuren-Sprache (alles Englisch)
    6. Effect Sizes (Cohen's d auf Bootstrap-Verteilungen)

Architektur (identisch fuer alle fuenf Bedingungen):
    Conv1: 3  -> 32  Filter, 3x3  -> V1-analog
    Conv2: 32 -> 64  Filter, 3x3  -> V2-analog
    Conv3: 64 -> 128 Filter, 3x3  -> LOC-analog
    FC1:   128*4*4 -> 512          -> IT-analog
    FC2:   512 -> 10 (Classifier)

Lernregeln:
    0. Random Weights (untrained, He-init)  <-- NEU
    1. Backpropagation (BP)
    2. Feedback Alignment (FA)
    3. Predictive Coding (PC)
    4. STDP
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset, Dataset
from scipy.stats import spearmanr, bootstrap
from scipy.spatial.distance import pdist, squareform
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import random
from itertools import combinations

# ── Pfade ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
PC_OUTPUTS  = Path(r"C:\Users\nilsl\Desktop\Projekte\learning-rules-rsa\outputs_720")
THINGS_DIR  = Path(r"C:\Users\nilsl\Desktop\Projekte\RSA\Datensatz\images_THINGS\object_images")
SUBJECTS    = ["sub-01", "sub-02", "sub-03"]

# ── Hyperparameter ─────────────────────────────────────────────────────────────
SEED     = 42
N_EPOCHS = 40
BATCH    = 64
LR       = 1e-3
N_CIFAR  = 8000

C1, C2, C3 = 32, 64, 128
FC1_DIM    = 512
N_CLS      = 10
FEAT_SIZE  = 4
FC1_IN     = C3 * FEAT_SIZE * FEAT_SIZE

# PC
T_INF  = 10
LR_R   = 0.02
LR_W   = 1e-4

# STDP
A_P    = 0.003
A_M    = 0.003
TAU_P  = 20.0
TAU_M  = 20.0
T_SIM  = 10

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


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

# ── 0. Random Weights (untrained) ─────────────────────────────────────────────

class Random_CNN(nn.Module):
    """Identical architecture, He-initialized, never trained."""
    def __init__(self):
        super().__init__()
        self.conv1 = make_conv_block(3,   C1)
        self.conv2 = make_conv_block(C1,  C2)
        self.conv3 = make_conv_block(C2,  C3)
        self.fc1   = nn.Linear(FC1_IN, FC1_DIM)
        self.fc2   = nn.Linear(FC1_DIM, N_CLS)

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def get_features(self, x):
        with torch.no_grad():
            c1 = self.conv1(x)
            c2 = self.conv2(c1)
            c3 = self.conv3(c2)
            f  = c3.view(c3.size(0), -1)
            h1 = F.relu(self.fc1(f))
        return (c1.mean([2,3]), c2.mean([2,3]),
                c3.mean([2,3]), h1)


def train_random(loader):
    """No training — just initialize and return."""
    torch.manual_seed(SEED)
    model = Random_CNN()
    model.eval()
    # Compute chance-level accuracy
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    hist = [{"epoch": 0, "loss": float('nan'), "acc": correct/total}]
    print(f"    Random baseline accuracy: {correct/total:.3f} (expected ~0.1)")
    return model, hist


# ── 1. Backpropagation ─────────────────────────────────────────────────────────

class BP_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = make_conv_block(3,   C1)
        self.conv2 = make_conv_block(C1,  C2)
        self.conv3 = make_conv_block(C2,  C3)
        self.fc1   = nn.Linear(FC1_IN, FC1_DIM)
        self.fc2   = nn.Linear(FC1_DIM, N_CLS)
        self.drop  = nn.Dropout(0.3)

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)

    def get_features(self, x):
        with torch.no_grad():
            c1 = self.conv1(x)
            c2 = self.conv2(c1)
            c3 = self.conv3(c2)
            f  = c3.view(c3.size(0), -1)
            h1 = F.relu(self.fc1(f))
        return (c1.mean([2,3]), c2.mean([2,3]),
                c3.mean([2,3]), h1)


def train_bp(loader):
    model = BP_CNN()
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_EPOCHS)
    hist  = []
    for epoch in range(N_EPOCHS):
        model.train()
        tl, tc, tn = 0.0, 0, 0
        for x, y in loader:
            opt.zero_grad()
            out  = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item(); tc += (out.argmax(1)==y).sum().item(); tn += y.size(0)
        sched.step()
        hist.append({"epoch": epoch+1, "loss": tl/len(loader), "acc": tc/tn})
        if (epoch+1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}: loss={tl/len(loader):.4f}  acc={tc/tn:.3f}")
    model.eval()
    return model, hist


# ── 2. Feedback Alignment ──────────────────────────────────────────────────────

class FAConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, b, B_feedback, stride, padding):
        ctx.save_for_backward(x, W, b, B_feedback)
        ctx.stride  = stride
        ctx.padding = padding
        return F.conv2d(x, W, b, stride=stride, padding=padding)

    @staticmethod
    def backward(ctx, grad_output):
        x, W, b, B = ctx.saved_tensors
        stride, padding = ctx.stride, ctx.padding
        grad_W = torch.nn.grad.conv2d_weight(x, W.shape, grad_output,
                                              stride=stride, padding=padding)
        grad_b = grad_output.sum([0,2,3]) if b is not None else None
        grad_x = F.conv_transpose2d(grad_output, B, stride=stride, padding=padding)
        return grad_x, grad_W, grad_b, None, None, None


class FAConv2d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.W = nn.Parameter(
            nn.init.kaiming_normal_(torch.empty(out_c, in_c, kernel_size, kernel_size)))
        self.b = nn.Parameter(torch.zeros(out_c))
        B = torch.randn(out_c, in_c, kernel_size, kernel_size)
        nn.init.xavier_normal_(B)
        self.register_buffer('B_feedback', B)
        self.stride  = stride
        self.padding = padding

    def forward(self, x):
        return FAConvFunction.apply(x, self.W, self.b, self.B_feedback,
                                    self.stride, self.padding)


def make_fa_conv_block(in_c, out_c):
    return nn.Sequential(
        FAConv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class FA_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = make_fa_conv_block(3,  C1)
        self.conv2 = make_fa_conv_block(C1, C2)
        self.conv3 = make_fa_conv_block(C2, C3)
        self.fc1   = nn.Linear(FC1_IN, FC1_DIM)
        self.fc2   = nn.Linear(FC1_DIM, N_CLS)
        self.drop  = nn.Dropout(0.3)

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)

    def get_features(self, x):
        with torch.no_grad():
            c1 = self.conv1(x)
            c2 = self.conv2(c1)
            c3 = self.conv3(c2)
            f  = c3.view(c3.size(0), -1)
            h1 = F.relu(self.fc1(f))
        return (c1.mean([2,3]), c2.mean([2,3]),
                c3.mean([2,3]), h1)


def train_fa(loader):
    model = FA_CNN()
    opt   = torch.optim.SGD(model.parameters(), lr=LR*0.5,
                            momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_EPOCHS)
    hist  = []
    for epoch in range(N_EPOCHS):
        model.train()
        tl, tc, tn = 0.0, 0, 0
        for x, y in loader:
            opt.zero_grad()
            out  = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item(); tc += (out.argmax(1)==y).sum().item(); tn += y.size(0)
        sched.step()
        hist.append({"epoch": epoch+1, "loss": tl/len(loader), "acc": tc/tn})
        if (epoch+1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}: loss={tl/len(loader):.4f}  acc={tc/tn:.3f}")
    model.eval()
    return model, hist


# ── 3. Predictive Coding ──────────────────────────────────────────────────────

class PC_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Conv2d(3,  C1, 3, padding=1, bias=False)
        self.W2 = nn.Conv2d(C1, C2, 3, padding=1, bias=False)
        self.W3 = nn.Conv2d(C2, C3, 3, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2)
        self.bn1  = nn.BatchNorm2d(C1)
        self.bn2  = nn.BatchNorm2d(C2)
        self.bn3  = nn.BatchNorm2d(C3)
        self.P2 = nn.ConvTranspose2d(C2, C1, 3, padding=1, bias=False)
        self.P3 = nn.ConvTranspose2d(C3, C2, 3, padding=1, bias=False)
        self.fc1 = nn.Linear(FC1_IN, FC1_DIM)
        self.fc2 = nn.Linear(FC1_DIM, N_CLS)
        self.clf_opt = torch.optim.Adam(
            list(self.fc1.parameters()) + list(self.fc2.parameters()), lr=LR)

    def infer(self, x):
        with torch.no_grad():
            r1 = self.pool(F.relu(self.bn1(self.W1(x))))
            r2 = self.pool(F.relu(self.bn2(self.W2(r1))))
            r3 = self.pool(F.relu(self.bn3(self.W3(r2))))
            for _ in range(T_INF):
                pred1 = torch.tanh(F.interpolate(
                    self.P2(r2), size=r1.shape[2:], mode='nearest'))
                pred2 = torch.tanh(F.interpolate(
                    self.P3(r3), size=r2.shape[2:], mode='nearest'))
                e1 = r1 - pred1
                e2 = r2 - pred2
                dr1 = -e1
                dr2 = -e2 + F.avg_pool2d(
                    F.conv2d(e1, self.W2.weight, padding=1), 2)
                dr3 = F.avg_pool2d(
                    F.conv2d(e2, self.W3.weight, padding=1), 2)
                r1 = F.relu(r1 + LR_R * dr1)
                r2 = F.relu(r2 + LR_R * dr2)
                r3 = F.relu(r3 + LR_R * dr3)
            return r1, r2, r3

    def weight_update(self, x, r1, r2, r3):
        with torch.no_grad():
            r1_init = self.pool(F.relu(self.bn1(self.W1(x))))
            r2_init = self.pool(F.relu(self.bn2(self.W2(r1_init))))
            e1 = (r1 - r1_init).clamp(-0.5, 0.5)
            e2 = (r2 - r2_init).clamp(-0.5, 0.5)
            de1 = e1.mean([0,2,3])
            din = x.mean([0,2,3])
            dW1 = (de1.unsqueeze(1) * din.unsqueeze(0)).unsqueeze(-1).unsqueeze(-1)
            dW1 = dW1.expand_as(self.W1.weight).clamp(-0.01, 0.01)
            self.W1.weight.data += LR_W * dW1
            de2 = e2.mean([0,2,3])
            din2 = r1.mean([0,2,3])
            dW2 = (de2.unsqueeze(1) * din2.unsqueeze(0)).unsqueeze(-1).unsqueeze(-1)
            dW2 = dW2.expand_as(self.W2.weight).clamp(-0.01, 0.01)
            self.W2.weight.data += LR_W * dW2

    def forward_features(self, x):
        with torch.no_grad():
            r1, r2, r3 = self.infer(x)
            f  = r3.view(r3.size(0), -1)
            h1 = F.relu(self.fc1(f))
        return r1, r2, r3, h1

    def get_features(self, x):
        r1, r2, r3, h1 = self.forward_features(x)
        return (r1.mean([2,3]), r2.mean([2,3]),
                r3.mean([2,3]), h1)

    def step(self, x, y):
        r1, r2, r3 = self.infer(x)
        self.weight_update(x, r1, r2, r3)
        self.clf_opt.zero_grad()
        f     = r3.detach().view(r3.size(0), -1)
        logit = self.fc2(F.relu(self.fc1(f)))
        loss  = F.cross_entropy(logit, y)
        loss.backward()
        self.clf_opt.step()
        acc = (logit.argmax(1) == y).float().mean().item()
        return loss.item(), acc

    def eval(self): pass


def train_pc(loader):
    model = PC_CNN()
    hist  = []
    for epoch in range(N_EPOCHS):
        tl, ta, n = 0.0, 0.0, 0
        for x, y in loader:
            loss, acc = model.step(x, y)
            tl += loss; ta += acc; n += 1
        hist.append({"epoch": epoch+1, "loss": tl/n, "acc": ta/n})
        if (epoch+1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}: loss={tl/n:.4f}  acc={ta/n:.3f}")
    return model, hist


# ── 4. STDP ────────────────────────────────────────────────────────────────────

class STDP_Conv:
    def __init__(self, in_c, out_c, kernel_size=3, padding=1):
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, padding=padding, bias=False)
        nn.init.kaiming_normal_(self.conv.weight)

    def poisson_spikes(self, rates):
        r = rates.clamp(0, 1).unsqueeze(-1).expand(*rates.shape, T_SIM)
        return (torch.rand_like(r) < r / T_SIM).float()

    def first_spike(self, spikes):
        has = spikes.any(-1)
        t   = torch.argmax(spikes, dim=-1).float()
        t[~has] = float(T_SIM + 1)
        return t

    def stdp_update(self, pre_act, post_act, lr=5e-4):
        with torch.no_grad():
            pre_r  = torch.sigmoid(pre_act)
            post_r = torch.sigmoid(post_act)
            pre_s  = self.poisson_spikes(pre_r)
            post_s = self.poisson_spikes(post_r)
            t_pre  = self.first_spike(pre_s)
            t_post = self.first_spike(post_s)
            dt  = t_post.unsqueeze(1) - t_pre.unsqueeze(2)
            ltp = A_P * torch.exp(-dt.clamp(min=0) / TAU_P)
            ltd = -A_M * torch.exp(dt.clamp(max=0) / TAU_M)
            dW  = (ltp + ltd).mean(0).clamp(-0.002, 0.002)
            dW_conv = dW.T.view(
                self.conv.weight.size(0), self.conv.weight.size(1), 1, 1
            ).expand_as(self.conv.weight)
            self.conv.weight.data += lr * dW_conv
            self.conv.weight.data.clamp_(-1.0, 1.0)

    def forward(self, x, do_stdp=False, pre_act=None):
        out = F.relu(self.conv(x))
        if do_stdp and pre_act is not None:
            self.stdp_update(pre_act, out.mean([2, 3]))
        return out


class STDP_CNN:
    def __init__(self):
        self.L1   = STDP_Conv(3,  C1)
        self.L2   = STDP_Conv(C1, C2)
        self.L3   = STDP_Conv(C2, C3)
        self.pool = nn.MaxPool2d(2)
        self.bn1  = nn.BatchNorm2d(C1)
        self.bn2  = nn.BatchNorm2d(C2)
        self.bn3  = nn.BatchNorm2d(C3)
        self.fc1  = nn.Linear(FC1_IN, FC1_DIM)
        self.fc2  = nn.Linear(FC1_DIM, N_CLS)
        self.clf_opt = torch.optim.Adam(
            list(self.fc1.parameters()) + list(self.fc2.parameters()) +
            [self.bn1.weight, self.bn1.bias,
             self.bn2.weight, self.bn2.bias,
             self.bn3.weight, self.bn3.bias], lr=LR)

    def forward(self, x, do_stdp=False):
        pre1 = x.mean([2, 3])
        c1   = self.pool(F.relu(self.bn1(
                    self.L1.forward(x, do_stdp=do_stdp, pre_act=pre1))))
        pre2 = c1.mean([2, 3])
        c2   = self.pool(F.relu(self.bn2(
                    self.L2.forward(c1, do_stdp=do_stdp, pre_act=pre2))))
        pre3 = c2.mean([2, 3])
        c3   = self.pool(F.relu(self.bn3(
                    self.L3.forward(c2, do_stdp=do_stdp, pre_act=pre3))))
        f    = c3.view(c3.size(0), -1)
        return self.fc2(F.relu(self.fc1(f))), c1, c2, c3

    def step(self, x, y):
        logit, c1, c2, c3 = self.forward(x, do_stdp=True)
        self.clf_opt.zero_grad()
        loss = F.cross_entropy(logit.detach(), y)
        logit2, _, _, _ = self.forward(x, do_stdp=False)
        loss2 = F.cross_entropy(logit2, y)
        loss2.backward()
        self.clf_opt.step()
        acc = (logit.argmax(1) == y).float().mean().item()
        return loss.item(), acc

    def get_features(self, x):
        with torch.no_grad():
            _, c1, c2, c3 = self.forward(x, do_stdp=False)
            f  = c3.view(c3.size(0), -1)
            h1 = F.relu(self.fc1(f))
        return (c1.mean([2,3]), c2.mean([2,3]),
                c3.mean([2,3]), h1)

    def eval(self): pass


def train_stdp(loader):
    model = STDP_CNN()
    hist  = []
    for epoch in range(N_EPOCHS):
        tl, ta, n = 0.0, 0.0, 0
        for x, y in loader:
            loss, acc = model.step(x, y)
            tl += loss; ta += acc; n += 1
        hist.append({"epoch": epoch+1, "loss": tl/n, "acc": ta/n})
        if (epoch+1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}: loss={tl/n:.4f}  acc={ta/n:.3f}")
    return model, hist


# ══════════════════════════════════════════════════════════════════════════════
# RSA PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def compute_rdm(f):
    f = f.detach().cpu().numpy() if torch.is_tensor(f) else np.array(f)
    return squareform(pdist(f, metric="correlation"))

def rsa_score(a, b):
    n   = min(a.shape[0], b.shape[0])
    idx = np.triu_indices(n, k=1)
    r, p = spearmanr(a[:n,:n][idx], b[:n,:n][idx])
    return float(r), float(p)

def bootstrap_ci(rdm_a, rdm_b, n_boot=500, ci=0.95):
    n   = min(rdm_a.shape[0], rdm_b.shape[0])
    idx = np.triu_indices(n, k=1)
    va  = rdm_a[:n,:n][idx]
    vb  = rdm_b[:n,:n][idx]
    rng = np.random.default_rng(SEED)
    boot_rs = []
    for _ in range(n_boot):
        samp = rng.integers(0, len(va), len(va))
        r, _ = spearmanr(va[samp], vb[samp])
        boot_rs.append(r)
    lo = np.percentile(boot_rs, (1-ci)/2 * 100)
    hi = np.percentile(boot_rs, (1+ci)/2 * 100)
    return lo, hi, boot_rs  # <-- NEU: auch die Verteilung zurueckgeben

def load_fmri_rdm(roi, sub):
    p = PC_OUTPUTS / f"fmri_rdm_{roi}_{sub}.npy"
    return np.load(str(p)) if p.exists() else None

def noise_ceiling(sub_ids, roi, n_splits=200):
    rdms = [load_fmri_rdm(roi, s) for s in sub_ids]
    rdms = [r for r in rdms if r is not None]
    if len(rdms) < 2:
        return np.nan, np.nan
    rng  = np.random.default_rng(SEED)
    n    = rdms[0].shape[0]
    idx  = np.triu_indices(n, k=1)
    rhos = []
    for _ in range(n_splits):
        perm   = rng.permutation(len(rdms))
        half1  = np.mean([rdms[i][idx] for i in perm[:len(rdms)//2]], 0)
        half2  = np.mean([rdms[i][idx] for i in perm[len(rdms)//2:]], 0)
        r, _   = spearmanr(half1, half2)
        rhos.append(2*r / (1+r) if r < 1 else 1.0)
    lo = np.percentile(rhos, 2.5)
    hi = np.mean(rhos)
    return float(lo), float(hi)

def load_stim_order(sub="sub-01"):
    p = PC_OUTPUTS / f"stim_order_{sub}.txt"
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

class ImgDS(Dataset):
    def __init__(self, paths, t): self.paths, self.t = paths, t
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        return self.t(Image.open(self.paths[i]).convert("RGB")), i

def extract_features(model, paths, transform):
    loader = DataLoader(ImgDS(paths, transform), batch_size=32,
                        shuffle=False, num_workers=0)
    c1s, c2s, c3s, h1s = [], [], [], []
    model.eval()
    with torch.no_grad():
        for imgs, _ in loader:
            c1, c2, c3, h1 = model.get_features(imgs)
            def n(t): return t.cpu().numpy() if torch.is_tensor(t) else np.array(t)
            c1s.append(n(c1)); c2s.append(n(c2))
            c3s.append(n(c3)); h1s.append(n(h1))
    return (np.concatenate(c1s), np.concatenate(c2s),
            np.concatenate(c3s), np.concatenate(h1s))

# ── Layer-ROI Mapping ──────────────────────────────────────────────────────────

# Fixed mapping (as in paper)
LAYER_ROI_FIXED = {
    "Conv1": (0, ["V1", "V2"]),
    "Conv2": (1, ["V1", "V2"]),
    "Conv3": (2, ["LOC"]),
    "FC1":   (3, ["IT"]),
}

# All layers x all ROIs (for best-layer-per-ROI analysis)
LAYER_ROI_ALL = {
    "Conv1": (0, ["V1", "V2", "LOC", "IT"]),
    "Conv2": (1, ["V1", "V2", "LOC", "IT"]),
    "Conv3": (2, ["V1", "V2", "LOC", "IT"]),
    "FC1":   (3, ["V1", "V2", "LOC", "IT"]),
}


def run_rsa(model, paths, transform, rule, layer_roi_map=None):
    """
    RSA pipeline. Returns rows for DataFrame and raw RDMs.
    
    NEU in v6:
      - Subject-level scores statt nur Mittelwert
      - Bootstrap-Verteilungen gespeichert fuer Effect-Size-Berechnung
      - Optionales layer_roi_map fuer Best-Layer-per-ROI
    """
    if layer_roi_map is None:
        layer_roi_map = LAYER_ROI_FIXED

    print(f"\n  RSA: {rule}")
    feats = extract_features(model, paths, transform)
    rdms  = [compute_rdm(f) for f in feats]

    rows = []
    for layer_name, (feat_idx, rois) in layer_roi_map.items():
        for roi in rois:
            sub_scores = []
            for sub in SUBJECTS:
                fmri = load_fmri_rdm(roi, sub)
                if fmri is None:
                    continue
                r, _ = rsa_score(rdms[feat_idx], fmri)
                sub_scores.append({"sub": sub, "rho": r})

            if sub_scores:
                # Mean brain RDM — used for rho, CI, and permutation test
                brain_mean = np.mean([load_fmri_rdm(roi, s) for s in SUBJECTS
                                      if load_fmri_rdm(roi, s) is not None], axis=0)
                n = min(rdms[feat_idx].shape[0], brain_mean.shape[0])

                # PRIMARY rho: model vs mean-brain (consistent with CI + perm test)
                rho_mean_brain, _ = rsa_score(rdms[feat_idx][:n,:n], brain_mean[:n,:n])

                # CI on same metric
                lo, hi, boot_dist = bootstrap_ci(rdms[feat_idx][:n,:n],
                                                  brain_mean[:n,:n])

                rows.append({
                    "rule":      rule,
                    "layer":     layer_name,
                    "roi":       roi,
                    "rho":       float(rho_mean_brain),  # consistent with CI + perm
                    "ci_lo":     lo,
                    "ci_hi":     hi,
                    "n_subs":    len(sub_scores),
                    # Subject-level scores (for subject-level analysis)
                    "rho_sub01": next((s["rho"] for s in sub_scores if s["sub"]=="sub-01"), np.nan),
                    "rho_sub02": next((s["rho"] for s in sub_scores if s["sub"]=="sub-02"), np.nan),
                    "rho_sub03": next((s["rho"] for s in sub_scores if s["sub"]=="sub-03"), np.nan),
                    # Per-subject mean (for reference, NOT the primary metric)
                    "rho_sub_mean": float(np.mean([s["rho"] for s in sub_scores])),
                    # Bootstrap distribution stats
                    "boot_mean": float(np.mean(boot_dist)),
                    "boot_std":  float(np.std(boot_dist)),
                })

                mean_sub = float(np.mean([s["rho"] for s in sub_scores]))
                print(f"    {layer_name:5s} vs {roi:3s}: "
                      f"rho={rho_mean_brain:.4f} [{lo:.4f},{hi:.4f}]  "
                      f"sub_mean={mean_sub:.4f}  "
                      f"per-sub: {[round(s['rho'],4) for s in sub_scores]}")

    return rows, rdms


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL ANALYSES (NEU in v6 — integriert statt separates Skript)
# ══════════════════════════════════════════════════════════════════════════════

def benjamini_hochberg(p_values):
    """FDR correction (Benjamini-Hochberg). Returns adjusted p-values."""
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p   = np.array(p_values)[sorted_idx]
    adjusted   = np.zeros(n)
    cum_min    = 1.0
    for i in range(n-1, -1, -1):
        adj = sorted_p[i] * n / (i + 1)
        cum_min = min(cum_min, adj)
        adjusted[sorted_idx[i]] = min(cum_min, 1.0)
    return adjusted.tolist()


def permutation_test(rdm_a, rdm_b, brain_rdm, n_perm=1000):
    """Permutation test for H0: RSA(A,brain) == RSA(B,brain)."""
    rng = np.random.default_rng(SEED)
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
    return observed, p_val


def cohens_d_bootstrap(boot_a, boot_b):
    """Cohen's d between two bootstrap distributions."""
    a, b = np.array(boot_a), np.array(boot_b)
    pooled_std = np.sqrt((a.std()**2 + b.std()**2) / 2)
    if pooled_std < 1e-10:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def run_all_stats(model_rdms, rsa_df):
    """
    Run permutation tests + FDR correction + effect sizes.
    model_rdms: dict[rule][layer] -> ndarray
    rsa_df: DataFrame from run_rsa
    """
    rules = [r for r in ["Random Weights", "Backprop", "Feedback Alignment",
                          "Predictive Coding", "STDP"]
             if r in model_rdms]
    layer_map = {"V1": "Conv1", "V2": "Conv1", "LOC": "Conv3", "IT": "FC1"}
    rois = ["V1", "V2", "LOC", "IT"]

    perm_rows = []
    all_p_values = []

    for roi in rois:
        brain = np.mean([load_fmri_rdm(roi, s) for s in SUBJECTS
                         if load_fmri_rdm(roi, s) is not None], axis=0)
        if brain is None:
            continue
        layer = layer_map[roi]
        print(f"\n  Permutation tests — {roi} (layer {layer}):")

        for rule_a, rule_b in combinations(rules, 2):
            if layer not in model_rdms[rule_a] or layer not in model_rdms[rule_b]:
                continue
            delta, p = permutation_test(
                model_rdms[rule_a][layer], model_rdms[rule_b][layer], brain)

            # Effect size from bootstrap distributions
            sub_a = rsa_df[(rsa_df["rule"]==rule_a) & (rsa_df["roi"]==roi)
                           & (rsa_df["layer"]==layer)]
            sub_b = rsa_df[(rsa_df["rule"]==rule_b) & (rsa_df["roi"]==roi)
                           & (rsa_df["layer"]==layer)]
            d = np.nan
            if not sub_a.empty and not sub_b.empty:
                # Use per-subject rho values as pseudo-bootstrap
                a_vals = [sub_a.iloc[0].get(f"rho_sub0{i}", np.nan) for i in [1,2,3]]
                b_vals = [sub_b.iloc[0].get(f"rho_sub0{i}", np.nan) for i in [1,2,3]]
                a_vals = [v for v in a_vals if not np.isnan(v)]
                b_vals = [v for v in b_vals if not np.isnan(v)]
                if len(a_vals) >= 2 and len(b_vals) >= 2:
                    d = cohens_d_bootstrap(a_vals, b_vals)

            perm_rows.append({
                "roi": roi, "layer": layer,
                "rule_a": rule_a, "rule_b": rule_b,
                "delta": round(delta, 5),
                "p_uncorrected": round(p, 4),
                "cohens_d": round(d, 3) if not np.isnan(d) else np.nan,
            })
            all_p_values.append(p)

            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"    {rule_a:25s} vs {rule_b:25s}: "
                  f"delta={delta:+.5f}  p={p:.4f} {sig}  d={d:.2f}")

    # FDR correction
    if all_p_values:
        fdr_adjusted = benjamini_hochberg(all_p_values)
        for i, row in enumerate(perm_rows):
            row["p_fdr"] = round(fdr_adjusted[i], 4)
            p_fdr = fdr_adjusted[i]
            row["sig_fdr"] = ("***" if p_fdr < 0.001 else "**" if p_fdr < 0.01
                              else "*" if p_fdr < 0.05 else "ns")

    perm_df = pd.DataFrame(perm_rows)
    return perm_df


# ══════════════════════════════════════════════════════════════════════════════
# SUBJECT-LEVEL ANALYSIS (NEU in v6)
# ══════════════════════════════════════════════════════════════════════════════

def subject_level_analysis(rsa_df):
    """
    Check if learning-rule rankings are stable across individual subjects.
    Returns a summary DataFrame.
    """
    print("\n" + "="*55)
    print("SUBJECT-LEVEL ANALYSIS")
    print("="*55)

    rois = ["V1", "V2", "LOC", "IT"]
    rows = []

    for roi in rois:
        sub_df = rsa_df[rsa_df["roi"] == roi].copy()
        if sub_df.empty:
            continue

        # Pick best layer per rule for this ROI (avoid duplicates from Conv1+Conv2)
        best_per_rule = sub_df.loc[sub_df.groupby("rule")["rho"].idxmax()]

        print(f"\n  {roi}:")
        for sub_col, sub_name in [("rho_sub01","sub-01"),
                                   ("rho_sub02","sub-02"),
                                   ("rho_sub03","sub-03")]:
            if sub_col not in best_per_rule.columns:
                continue
            ranking = best_per_rule[["rule", sub_col]].dropna().sort_values(
                sub_col, ascending=False)
            rank_order = ranking["rule"].tolist()
            print(f"    {sub_name}: {' > '.join(rank_order)}")

            for _, row in ranking.iterrows():
                rows.append({
                    "roi": roi, "subject": sub_name,
                    "rule": row["rule"], "rho": round(row[sub_col], 4),
                })

    sub_level_df = pd.DataFrame(rows)

    # Check ranking consistency
    if not sub_level_df.empty:
        print("\n  Ranking consistency (best rule per subject):")
        for roi in rois:
            roi_data = sub_level_df[sub_level_df["roi"] == roi]
            if roi_data.empty:
                continue
            best_per_sub = roi_data.groupby("subject")["rho"].idxmax()
            best_per_sub = [roi_data.loc[idx, "rule"] for idx in best_per_sub.values]
            unanimous = len(set(best_per_sub)) == 1
            print(f"    {roi}: {best_per_sub}  "
                  f"{'UNANIMOUS' if unanimous else 'INCONSISTENT'}")

    return sub_level_df


# ══════════════════════════════════════════════════════════════════════════════
# BEST-LAYER-PER-ROI ANALYSIS (NEU in v6)
# ══════════════════════════════════════════════════════════════════════════════

def best_layer_analysis(rsa_df_all):
    """
    For each rule x ROI, find the best-performing layer.
    Checks whether fixed mapping results are robust.
    """
    print("\n" + "="*55)
    print("BEST-LAYER-PER-ROI ANALYSIS")
    print("="*55)

    rois  = ["V1", "V2", "LOC", "IT"]
    rules = rsa_df_all["rule"].unique()

    rows = []
    for roi in rois:
        print(f"\n  {roi}:")
        for rule in rules:
            sub = rsa_df_all[(rsa_df_all["roi"]==roi) & (rsa_df_all["rule"]==rule)]
            if sub.empty:
                continue
            best_row  = sub.loc[sub["rho"].idxmax()]
            best_layer = best_row["layer"]
            best_rho   = best_row["rho"]

            # Fixed mapping layer for comparison
            fixed_map = {"V1": "Conv1", "V2": "Conv1", "LOC": "Conv3", "IT": "FC1"}
            fixed_layer = fixed_map[roi]
            fixed_sub = sub[sub["layer"] == fixed_layer]
            fixed_rho = fixed_sub["rho"].values[0] if not fixed_sub.empty else np.nan

            match = "MATCH" if best_layer == fixed_layer else "MISMATCH"
            print(f"    {rule:25s}: best={best_layer} (rho={best_rho:.4f})  "
                  f"fixed={fixed_layer} (rho={fixed_rho:.4f})  {match}")

            rows.append({
                "roi": roi, "rule": rule,
                "best_layer": best_layer, "best_rho": round(best_rho, 4),
                "fixed_layer": fixed_layer, "fixed_rho": round(fixed_rho, 4),
                "match": match == "MATCH",
            })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS — All in English (consistency fix)
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {
    "Random Weights":     "#999999",
    "Backprop":           "#2E86AB",
    "Feedback Alignment": "#E84855",
    "Predictive Coding":  "#3BB273",
    "STDP":               "#F4A261",
}

def plot_training(histories):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for rule, hist in histories.items():
        if rule == "Random Weights":
            continue  # no training curve
        df  = pd.DataFrame(hist)
        col = COLORS.get(rule, "gray")
        axes[0].plot(df["epoch"], df["loss"], color=col, lw=2, label=rule)
        axes[1].plot(df["epoch"], df["acc"],  color=col, lw=2, label=rule)
    for ax, title, ylabel in zip(axes,
        ["Training Loss (CNN)", "Training Accuracy (CNN)"], ["Loss", "Accuracy"]):
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(str(OUTPUTS_DIR / "training_curves_cnn.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  training_curves_cnn.png saved")


def plot_rsa(df, nc_dict=None):
    """RSA comparison — now all labels in English."""
    rois  = ["V1", "V2", "LOC", "IT"]
    rois  = [r for r in rois if r in df["roi"].values]
    rules = list(COLORS.keys())
    rules = [r for r in rules if r in df["rule"].values]
    x     = np.arange(len(rois))
    w     = 0.15
    offs  = np.linspace(-(len(rules)-1)*w/2, (len(rules)-1)*w/2, len(rules))

    fig, ax = plt.subplots(figsize=(13, 5.5))

    if nc_dict:
        for i, roi in enumerate(rois):
            if roi in nc_dict:
                lo, hi = nc_dict[roi]
                ax.fill_between([i-0.4, i+0.4], lo, hi,
                                alpha=0.12, color="gray", zorder=1)

    for i, rule in enumerate(rules):
        sub  = df[df["rule"] == rule].groupby("roi")
        for j, roi in enumerate(rois):
            if roi not in sub.groups:
                continue
            row  = sub.get_group(roi).iloc[0]
            rho  = row["rho"]
            lo   = row.get("ci_lo", rho)
            hi   = row.get("ci_hi", rho)
            bar  = ax.bar(j + offs[i], rho, w*0.85,
                          color=COLORS[rule], alpha=0.85,
                          label=rule if j == 0 else "")
            # CI from bootstrap on mean-brain RDM; rho is mean of per-subject
            # scores — clamp to avoid negative yerr
            err_lo = max(0, rho - lo)
            err_hi = max(0, hi - rho)
            ax.errorbar(j + offs[i], rho,
                        yerr=[[err_lo], [err_hi]],
                        fmt='none', color='black', capsize=3, linewidth=1)

    ax.set_xticks(x); ax.set_xticklabels(rois, fontsize=12)
    ax.set_ylabel("Spearman rho (model-brain)", fontsize=12)
    ax.set_xlabel("ROI (early → late)", fontsize=12)
    ax.set_title("Which learning rule produces the most brain-like representations?",
                 fontsize=13)
    ax.axhline(0, color="black", lw=0.5)

    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen: seen[l] = h

    if nc_dict:
        from matplotlib.patches import Patch
        ax.legend(list(seen.values()) + [Patch(color='gray', alpha=0.3)],
                  list(seen.keys()) + ['Noise Ceiling'],
                  fontsize=9, ncol=3, loc='upper right')
    else:
        ax.legend(seen.values(), seen.keys(), fontsize=9, ncol=3)

    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(str(OUTPUTS_DIR / "rsa_comparison_cnn.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  rsa_comparison_cnn.png saved")


def plot_hierarchy(df):
    """Hierarchy gradient — English labels."""
    layers = ["Conv1", "Conv2", "Conv3", "FC1"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for rule in COLORS:
        if rule not in df["rule"].values:
            continue
        sub  = df[df["rule"] == rule].groupby("layer")["rho"].mean()
        vals = [sub.get(l, np.nan) for l in layers]
        ax.plot(layers, vals, "o-", color=COLORS[rule],
                lw=2, markersize=8, label=rule)
    ax.set_xlabel("Layer (early → late)", fontsize=12)
    ax.set_ylabel("Spearman rho (mean across ROIs)", fontsize=12)
    ax.set_title("Hierarchy gradient per learning rule", fontsize=13)
    ax.axhline(0, color="black", lw=0.4)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(str(OUTPUTS_DIR / "hierarchy_cnn.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  hierarchy_cnn.png saved")


def plot_subject_consistency(sub_df):
    """Bar plot showing per-subject RSA scores for each rule x ROI."""
    rois = ["V1", "V2", "LOC", "IT"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)

    for ax, roi in zip(axes, rois):
        roi_data = sub_df[sub_df["roi"] == roi]
        if roi_data.empty:
            ax.set_title(roi); continue

        rules = roi_data["rule"].unique()
        subjects = roi_data["subject"].unique()
        x = np.arange(len(rules))
        w = 0.25

        for si, sub in enumerate(subjects):
            sub_data = roi_data[roi_data["subject"] == sub]
            vals = [sub_data[sub_data["rule"]==r]["rho"].values[0]
                    if not sub_data[sub_data["rule"]==r].empty else 0
                    for r in rules]
            ax.bar(x + (si-1)*w, vals, w*0.9, alpha=0.7, label=sub if roi=="V1" else "")

        short_names = [r.split()[0][:4] for r in rules]
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, fontsize=8, rotation=30)
        ax.set_title(roi, fontsize=11)
        ax.axhline(0, color="black", lw=0.4)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Spearman rho", fontsize=11)
    axes[0].legend(fontsize=8)
    plt.suptitle("Subject-level RSA scores per learning rule and ROI", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(OUTPUTS_DIR / "subject_consistency.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  subject_consistency.png saved")


def plot_permutation_forest(perm_df):
    """Forest plot with FDR-corrected significance."""
    rois = perm_df["roi"].unique()
    fig, axes = plt.subplots(1, len(rois), figsize=(4.5*len(rois), 5), sharey=True)
    if len(rois) == 1: axes = [axes]

    for ax, roi in zip(axes, rois):
        sub = perm_df[perm_df["roi"] == roi].copy()
        sub["label"] = sub["rule_a"].str[:4] + " vs " + sub["rule_b"].str[:4]
        sub = sub.sort_values("delta", ascending=True)

        colors = ["#2ecc71" if s != "ns" else "#e74c3c" for s in sub["sig_fdr"]]
        ax.barh(range(len(sub)), sub["delta"], color=colors, alpha=0.8)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub["label"], fontsize=8)
        ax.set_xlabel("Δρ (A − B)", fontsize=10)
        ax.set_title(f"ROI: {roi}", fontsize=11)

        for i, (_, row) in enumerate(sub.iterrows()):
            if row.get("sig_fdr", "ns") != "ns":
                ax.text(row["delta"], i, row["sig_fdr"],
                        va="center", fontsize=8, fontweight="bold")

    plt.suptitle("Pairwise Δρ (green = FDR-significant, red = ns)", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(OUTPUTS_DIR / "permutation_forest_fdr.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  permutation_forest_fdr.png saved")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(plots_only=False):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── PLOTS-ONLY MODE ───────────────────────────────────────────────────
    if plots_only:
        print("Learning Rules v6 — PLOTS-ONLY MODE\n")
        print("Loading saved results from", OUTPUTS_DIR)

        rsa_csv = OUTPUTS_DIR / "rsa_results_cnn.csv"
        if not rsa_csv.exists():
            print(f"ERROR: {rsa_csv} not found. Run full pipeline first.")
            return
        df = pd.read_csv(str(rsa_csv))
        print(f"  RSA results: {len(df)} rows")

        perm_csv = OUTPUTS_DIR / "permutation_results_fdr.csv"
        perm_df = pd.read_csv(str(perm_csv)) if perm_csv.exists() else pd.DataFrame()

        sub_csv = OUTPUTS_DIR / "subject_level_rsa.csv"
        sub_df = pd.read_csv(str(sub_csv)) if sub_csv.exists() else pd.DataFrame()

        nc_dict = {}
        for roi in ["V1", "V2", "LOC", "IT"]:
            lo, hi = noise_ceiling(SUBJECTS, roi)
            nc_dict[roi] = (lo, hi)

        print("\nRegenerating plots...")
        plot_rsa(df, nc_dict)
        plot_hierarchy(df)
        if not sub_df.empty:
            plot_subject_consistency(sub_df)
        if not perm_df.empty:
            plot_permutation_forest(perm_df)
        print("\nPlots regenerated. Done.")
        return

    # ── FULL PIPELINE ─────────────────────────────────────────────────────
    print("Learning Rules v6 — CNN RSA (with improvements)\n")

    # CIFAR-10
    print("Loading CIFAR-10...")
    tf = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    tf_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    full = torchvision.datasets.CIFAR10(str(BASE_DIR/"data"), train=True,
                                         download=True, transform=tf)
    idx  = torch.randperm(len(full))[:N_CIFAR].tolist()
    train_loader = DataLoader(Subset(full, idx), batch_size=BATCH,
                              shuffle=True, num_workers=0, drop_last=True)
    print(f"  {N_CIFAR} samples, {len(train_loader)} batches\n")

    # THINGS
    print("Loading THINGS image paths...")
    stimuli = load_stim_order("sub-01")
    paths   = [p for p in [find_img(s) for s in stimuli] if p is not None]
    print(f"  {len(paths)}/{len(stimuli)} images\n")

    tf_things = T.Compose([
        T.Resize(32), T.CenterCrop(32), T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    # Noise Ceilings
    print("Computing noise ceilings...")
    nc_dict = {}
    for roi in ["V1", "V2", "LOC", "IT"]:
        lo, hi = noise_ceiling(SUBJECTS, roi)
        nc_dict[roi] = (lo, hi)
        if not np.isnan(lo):
            print(f"  {roi}: NC [{lo:.3f}, {hi:.3f}]")

    # ── Training + RSA (fixed mapping) ──
    train_fns = {
        "Random Weights":     train_random,     # <-- NEU
        "Backprop":           train_bp,
        "Feedback Alignment": train_fa,
        "Predictive Coding":  train_pc,
        "STDP":               train_stdp,
    }

    histories    = {}
    all_rows     = []
    all_rdms     = {}
    model_rdms   = {}

    for rule, fn in train_fns.items():
        print(f"\n{'='*55}\n{'Training' if rule != 'Random Weights' else 'Initializing'}: {rule}\n{'='*55}")
        torch.manual_seed(SEED); np.random.seed(SEED)
        model, hist     = fn(train_loader)
        histories[rule] = hist
        rows, rdms      = run_rsa(model, paths, tf_things, rule)
        all_rows.extend(rows)

        # Save model RDMs
        rdm_dir  = OUTPUTS_DIR / "model_rdms"
        rdm_dir.mkdir(exist_ok=True)
        rule_key = rule.lower().replace(" ", "_")
        layer_names = ["Conv1", "Conv2", "Conv3", "FC1"]
        model_rdms[rule] = {}
        for layer_name, rdm in zip(layer_names, rdms):
            np.save(str(rdm_dir / f"rdm_{rule_key}_{layer_name}.npy"), rdm)
            model_rdms[rule][layer_name] = rdm

        # Save weights
        if hasattr(model, "state_dict"):
            torch.save(model.state_dict(),
                       str(OUTPUTS_DIR / f"model_weights_{rule_key}.pt"))

    # Save fixed-mapping RSA results
    df = pd.DataFrame(all_rows)
    df.to_csv(str(OUTPUTS_DIR / "rsa_results_cnn.csv"), index=False)
    print(f"\nCSV saved: rsa_results_cnn.csv")

    # ── Best-Layer-per-ROI (all layers x all ROIs) ──
    print("\n" + "="*55)
    print("Running best-layer-per-ROI analysis...")
    all_rows_full = []
    for rule in train_fns:
        rule_key = rule.lower().replace(" ", "_")
        # Re-load RDMs from saved files
        rdms_list = []
        for ln in ["Conv1", "Conv2", "Conv3", "FC1"]:
            p = OUTPUTS_DIR / "model_rdms" / f"rdm_{rule_key}_{ln}.npy"
            rdms_list.append(np.load(str(p)))

        for layer_name, (feat_idx, rois) in LAYER_ROI_ALL.items():
            for roi in rois:
                brain = np.mean([load_fmri_rdm(roi, s) for s in SUBJECTS
                                 if load_fmri_rdm(roi, s) is not None], axis=0)
                if brain is None: continue
                n = min(rdms_list[feat_idx].shape[0], brain.shape[0])
                r, _ = rsa_score(rdms_list[feat_idx][:n,:n], brain[:n,:n])
                all_rows_full.append({
                    "rule": rule, "layer": layer_name, "roi": roi, "rho": round(r, 4)
                })

    df_all = pd.DataFrame(all_rows_full)
    df_all.to_csv(str(OUTPUTS_DIR / "rsa_all_layers_all_rois.csv"), index=False)
    best_layer_df = best_layer_analysis(df_all)
    best_layer_df.to_csv(str(OUTPUTS_DIR / "best_layer_per_roi.csv"), index=False)

    # ── Statistical Analysis ──
    print("\n" + "="*55)
    print("Statistical Analysis (permutation + FDR + effect sizes)")
    print("="*55)
    perm_df = run_all_stats(model_rdms, df)
    perm_df.to_csv(str(OUTPUTS_DIR / "permutation_results_fdr.csv"), index=False)

    # ── Subject-Level Analysis ──
    sub_df = subject_level_analysis(df)
    sub_df.to_csv(str(OUTPUTS_DIR / "subject_level_rsa.csv"), index=False)

    # ── Plots ──
    print("\nGenerating plots...")
    plot_training(histories)
    plot_rsa(df, nc_dict)
    plot_hierarchy(df)
    plot_subject_consistency(sub_df)
    if not perm_df.empty:
        plot_permutation_forest(perm_df)

    # ── Summary ──
    print("\n" + "="*55)
    print("SUMMARY — CNN RSA v6")
    print("="*55)
    if not df.empty:
        # Fixed-mapping pivot
        pivot = df.pivot_table(index="roi", columns="rule", values="rho")
        print("\nFixed-mapping RSA scores:")
        print(pivot.round(4).to_string())

        print("\nBest learning rule per ROI:")
        for roi in ["V1", "V2", "LOC", "IT"]:
            sub = df[df["roi"] == roi]
            if sub.empty: continue
            best = sub.loc[sub["rho"].idxmax(), "rule"]
            rho  = sub["rho"].max()
            nc   = nc_dict.get(roi, (np.nan, np.nan))
            # Random baseline for comparison
            rand_sub = sub[sub["rule"] == "Random Weights"]
            rand_rho = rand_sub["rho"].values[0] if not rand_sub.empty else np.nan
            print(f"  {roi:5}: {best:25s} rho={rho:.4f}  "
                  f"random={rand_rho:.4f}  NC=[{nc[0]:.3f},{nc[1]:.3f}]")

        # FDR summary
        if not perm_df.empty:
            n_sig = (perm_df["sig_fdr"] != "ns").sum()
            n_tot = len(perm_df)
            print(f"\nPermutation tests: {n_sig}/{n_tot} significant after FDR correction")

    print("\nDone. All outputs saved to:", OUTPUTS_DIR)


if __name__ == "__main__":
    import sys
    plots_only = "--plots-only" in sys.argv or "--plots" in sys.argv
    main(plots_only=plots_only)
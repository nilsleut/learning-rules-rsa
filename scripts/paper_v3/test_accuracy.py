"""CIFAR-10 test accuracy of the repaired (bnfix) models.

Only seed 0 has a checkpoint: learning_rules_v10_sweep_modal.py saves weights only when
seed_idx == 0 (line ~553). Seeds 1-4 cannot be evaluated without retraining, which is not
done here. The architectures are taken verbatim from learning_rules_v10_sweep_modal.py
(source block "Architectures" ... "Training functions", exec'd with v10's constants), so
nothing is retyped. Evaluation: CIFAR-10 test set (10,000 images, 32x32), ToTensor +
the training normalisation, no augmentation, eval mode (BN running statistics).

Output: results/paper_v3/accuracy.csv
"""
import re
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SRC = REPO / "learning_rules_v10_sweep_modal.py"
CKPT = REPO / "learning_rules_outputs_bnfix" / "checkpoints"
OUT = REPO / "results" / "paper_v3" / "accuracy.csv"
RULES = ["random_weights", "backprop", "feedback_alignment", "predictive_coding", "stdp"]


def load_classes():
    s = SRC.read_text(encoding="utf-8")
    consts = s[s.index("# ── Constants matching v8"):s.index("# fixed layer->ROI mapping")]
    a = s.index("    # ── Architectures (exact v8)")
    b = s.index("    # ── Training functions (exact v8)")
    ns = {"torch": torch, "nn": nn, "F": F}
    exec(consts, ns)
    exec(textwrap.dedent(s[a:b]), ns)
    return ns


def logits(ns, rule, model, x):
    if rule in ("random_weights", "backprop", "feedback_alignment"):
        return model(x)
    if rule == "predictive_coding":                 # as in PC_CNN.step
        _, _, r3 = model.infer(x)
        return model.fc2(F.relu(model.fc1(r3.view(r3.size(0), -1))))
    _, _, c3 = model._forward(x, False)             # as in STDP_CNN.step
    return model.fc2(F.relu(model.fc1(c3.view(c3.size(0), -1))))


def build(ns, rule):
    sd = torch.load(CKPT / f"model_weights_{rule}.pt", map_location="cpu")
    if rule in ("random_weights", "backprop"):
        m = ns["BP_CNN"](); m.load_state_dict(sd); m.eval(); return m
    if rule == "feedback_alignment":
        m = ns["FA_CNN"](); m.load_state_dict(sd); m.eval(); return m
    if rule == "predictive_coding":
        m = ns["PC_CNN"](); m.load_state_dict(sd); m.eval(); return m
    m = ns["STDP_CNN"]()                            # plain class: load tensors by hand
    for i, L in enumerate([m.L1, m.L2, m.L3], 1):
        L.conv.weight.data = sd[f"conv{i}.weight"].clone()
    for i, bn in enumerate([m.bn1, m.bn2, m.bn3], 1):
        for k in ("weight", "bias"):
            getattr(bn, k).data = sd[f"bn{i}.{k}"].clone()
        bn.running_mean.data = sd[f"bn{i}.running_mean"].clone()
        bn.running_var.data = sd[f"bn{i}.running_var"].clone()
    for nm in ("fc1", "fc2"):
        getattr(m, nm).weight.data = sd[f"{nm}.weight"].clone()
        getattr(m, nm).bias.data = sd[f"{nm}.bias"].clone()
    m.eval()
    return m


def main():
    torch.manual_seed(0)
    ns = load_classes()
    tf = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    test = torchvision.datasets.CIFAR10(str(REPO / "data"), train=False, download=False, transform=tf)
    loader = DataLoader(test, batch_size=500, shuffle=False, num_workers=0)
    rows = []
    for rule in RULES:
        m = build(ns, rule)
        correct = n = 0
        with torch.no_grad():
            for x, y in loader:
                correct += int((logits(ns, rule, m, x).argmax(1) == y).sum()); n += len(y)
        rows.append({"rule": rule, "seed_idx": 0, "seed": 42, "n_test": n, "test_acc": correct / n})
        print(rule, correct / n, flush=True)
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)


if __name__ == "__main__":
    main()

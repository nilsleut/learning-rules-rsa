"""Sensitivity: V2 mapped to Conv2 instead of Conv1 (repaired set, cross-run, per subject).

Same pair set P, the same 1000 shared stimulus resamples (noise_ceiling_v2 index matrix,
seed 20261002) and the same per-subject convention as Table 2. Conv1 -> V2 is recomputed
alongside as validation: it must reproduce results/crossrun/step2_draws.csv (persub|cr|bnfix)
for every draw.

Primary V2 statements checked under Conv2 (sign and significance compared with Conv1):
  Random > BP;  FA lowest (Random, BP, PC, STDP each > FA).
Result: Random > BP, Random > FA and STDP > FA hold; BP - FA reverses and PC - FA becomes
non-significant, so the paper restricts "FA lowest" to V1 (appendix table tab:v2conv2).
  py -3 v2_conv2_sensitivity.py              # compute draws (~30 min, 10 processes)
  py -3 v2_conv2_sensitivity.py --from-draws # re-summarise the stored draws

Output: results/paper_v3/v2_conv2_draws.csv, v2_conv2_summary.csv, v2_conv2.json
"""
import json
import sys
import time
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts" / "crossrun"))
from common_cr import N_STIM, TRI, SUBJECTS, load, run_labels, ranks  # noqa: E402

P3 = REPO / "results" / "paper_v3"
CR = REPO / "results" / "crossrun"
RDMS = REPO / "learning_rules_outputs_bnfix" / "rdms" / "res224"
RULES = ["random_weights", "backprop", "feedback_alignment", "predictive_coding", "stdp"]
LAYERS = ["Conv1", "Conv2"]
SEEDS = range(5)
BOOT_SEED, N_BOOT = 20261002, 1000
PRIMARY = [("random_weights", "backprop")] + [(r, "feedback_alignment") for r in
                                                 ("random_weights", "backprop", "predictive_coding", "stdp")]
_W = {}


def _init():
    _W["runs"] = run_labels()
    _W["boot"] = np.random.default_rng(BOOT_SEED).integers(0, N_STIM, size=(N_BOOT, N_STIM))
    _W["fmri"] = load("old", "V2")
    _W["M"] = {(r, l, s): np.load(RDMS / f"seed_{s}" / f"rdm_{r}_{l}.npy")
               for r in RULES for l in LAYERS for s in SEEDS}


def one(b):
    s = np.arange(N_STIM) if b < 0 else _W["boot"][b]
    a, c = s[TRI[0]], s[TRI[1]]
    keep = a != c
    for r in _W["runs"].values():
        keep &= r[a] != r[c]
    a, c = a[keep], c[keep]
    sub = [ranks(x[a, c]) for x in _W["fmri"]]
    out = {"boot": b}
    for r in RULES:
        for l in LAYERS:
            v = [float(np.mean([ranks(np.asarray(_W["M"][(r, l, sd)][a, c], float)) @ x for x in sub]))
                 for sd in SEEDS]
            out[f"{l}|{r}"] = float(np.mean(v))
    return out


def main():
    t0 = time.time()
    if "--from-draws" in sys.argv:          # re-summarise an existing draws file
        D = pd.read_csv(P3 / "v2_conv2_draws.csv")
    else:
        with Pool(10, initializer=_init) as pool:
            rows = sorted(pool.map(one, range(-1, N_BOOT), chunksize=4), key=lambda r: r["boot"])
        D = pd.DataFrame(rows)
        D.to_csv(P3 / "v2_conv2_draws.csv", index=False)

    # validation: Conv1 -> V2 must reproduce crossrun step2 draws (persub|cr|bnfix|rule|V2)
    ref = pd.read_csv(CR / "step2_draws.csv").set_index("boot")
    dev = max(float((D.set_index("boot")[f"Conv1|{r}"] - ref[f"persub|cr|bnfix|{r}|V2"]).abs().max())
              for r in RULES)
    assert dev < 1e-10, f"Conv1 validation failed: {dev}"

    pt, bt = D[D.boot == -1].iloc[0], D[D.boot >= 0]
    summ, checks = [], {}
    for l in LAYERS:
        for r in RULES:
            k = f"{l}|{r}"
            summ.append(dict(layer=l, quantity=r, point=pt[k], ci_lo=bt[k].quantile(.025), ci_hi=bt[k].quantile(.975)))
        for a, b in combinations(RULES, 2):
            d = bt[f"{l}|{a}"] - bt[f"{l}|{b}"]
            summ.append(dict(layer=l, quantity=f"{a} - {b}", point=pt[f"{l}|{a}"] - pt[f"{l}|{b}"],
                             ci_lo=d.quantile(.025), ci_hi=d.quantile(.975)))
    for l in LAYERS:                         # primary pairs in their stated direction
        for a, b in PRIMARY:
            if (a, b) not in combinations(RULES, 2):
                d = bt[f"{l}|{a}"] - bt[f"{l}|{b}"]
                summ.append(dict(layer=l, quantity=f"{a} - {b}", point=pt[f"{l}|{a}"] - pt[f"{l}|{b}"],
                                 ci_lo=d.quantile(.025), ci_hi=d.quantile(.975)))
    S = pd.DataFrame(summ)
    S.to_csv(P3 / "v2_conv2_summary.csv", index=False)
    s = S.set_index(["layer", "quantity"])
    same = True
    for a, b in PRIMARY:
        q = f"{a} - {b}"
        r1, r2 = s.loc[("Conv1", q)], s.loc[("Conv2", q)]
        sig1, sig2 = (r1.ci_lo > 0 or r1.ci_hi < 0), (r2.ci_lo > 0 or r2.ci_hi < 0)
        ok = bool(np.sign(r1.point) == np.sign(r2.point) and sig1 == sig2)
        same &= ok
        checks[q] = {"conv1": [round(float(r1.point), 4), round(float(r1.ci_lo), 4), round(float(r1.ci_hi), 4)],
                     "conv2": [round(float(r2.point), 4), round(float(r2.ci_lo), 4), round(float(r2.ci_hi), 4)],
                     "same_sign_and_significance": ok}
    out = {"validation_conv1_vs_step2_max_dev": dev, "primary_same": bool(same), "checks": checks,
           "runtime_s": round(time.time() - t0, 1)}
    (P3 / "v2_conv2.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
    if not same:
        # Outcome of this check (Oct 2026): BP - FA and PC - FA change under Conv2. The paper
        # therefore states "FA lowest" at V1 only and reports this table in the appendix.
        print("NOTE: a primary V2 statement changes sign or significance under Conv2")


if __name__ == "__main__":
    main()

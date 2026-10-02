"""Additional cross-run quantities for paper v3 (bnfix model set, stored RDMs only).

B1  per-subject rho (sub-01, -02, -03) per rule x ROI (fixed layer mapping), seed mean,
    with the 1000 shared stimulus resamples of noise_ceiling_v2 (seed 20261002); pairwise
    rule differences per subject from the same resamples.
B2  per-seed rho per rule x ROI (cross-run; per-subject and mean-RDM convention):
    seed mean, SD, min, max.
H   hierarchy: per-subject cross-run rho for every layer (Conv1..FC1) x ROI, seed mean.

Pair set: cross-run intersection P (scripts/crossrun/common_cr.py), recomputed per resample.
Output: results/paper_v3/subject_draws.csv, subject_summary.csv, subject_pairwise.csv,
        seed_spread.csv, hierarchy.csv, compute_extra.json
"""
import hashlib
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts" / "crossrun"))
from common_cr import ROIS, SUBJECTS, N_STIM, load, run_labels, ranks, pair_set  # noqa: E402
from common import ROI_LAYER  # noqa: E402
from step2_bootstrap import CACHE, MODEL_ITEMS  # noqa: E402

OUT = REPO / "results" / "paper_v3"
BNFIX = REPO / "learning_rules_outputs_bnfix" / "rdms" / "res224"
RULES = ["random_weights", "backprop", "feedback_alignment", "predictive_coding", "stdp"]
LAYERS_ALL = ["Conv1", "Conv2", "Conv3", "FC1"]
SEEDS = range(5)
BOOT_SEED, N_BOOT = 20261002, 1000
_W = {}


def _init():
    _W["runs"] = run_labels()
    _W["fmri"] = {roi: load("old", roi) for roi in ROIS}
    # shared float64 memmap built by step2_bootstrap.py (same files, bit-identical)
    M = np.load(CACHE, mmap_mode="r")
    _W["M"] = {(r, roi, sd): M[MODEL_ITEMS.index(("bnfix", r, ROI_LAYER[roi], f"seed_{sd}"))]
               for r in RULES for roi in ROIS for sd in SEEDS}
    _W["boot"] = np.random.default_rng(BOOT_SEED).integers(0, N_STIM, size=(N_BOOT, N_STIM))


def subject_draw(b):
    """b = -1: identity; else bootstrap resample b. Seed-mean rho per subject x rule x ROI."""
    s = None if b < 0 else _W["boot"][b]
    rp, cp, _ = pair_set(s, _W["runs"])
    out = {"boot": b}
    for roi in ROIS:
        subj = [ranks(x[rp, cp]) for x in _W["fmri"][roi]]
        for r in RULES:
            mv = [ranks(np.asarray(_W["M"][(r, roi, sd)][rp, cp], float)) for sd in SEEDS]
            for si, sub in enumerate(SUBJECTS):
                out[f"{sub}|{r}|{roi}"] = float(np.mean([m @ subj[si] for m in mv]))
    return out


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    _init()
    v2 = json.loads((REPO / "results/noise_ceiling_v2/step3c_summary.json").read_text())
    assert hashlib.sha256(_W["boot"].tobytes()).hexdigest() == v2["boot_idx_sha256"]
    runs = _W["runs"]
    rp, cp, _ = pair_set(None, runs)

    # B2 + H: point estimates, identity sample
    seed_rows, hier = [], []
    for roi in ROIS:
        F = _W["fmri"][roi]
        subj = [ranks(x[rp, cp]) for x in F]
        meanr = ranks(np.mean([x[rp, cp] for x in F], axis=0))
        for r in RULES:
            for sd in SEEDS:
                for layer in LAYERS_ALL:
                    X = (np.asarray(_W["M"][(r, roi, sd)]) if layer == ROI_LAYER[roi]
                         else np.load(BNFIX / f"seed_{sd}" / f"rdm_{r}_{layer}.npy"))
                    mv = ranks(X[rp, cp])
                    ps = float(np.mean([mv @ q for q in subj]))
                    hier.append({"rule": r, "layer": layer, "roi": roi, "seed": sd, "persub": ps})
                    if layer == ROI_LAYER[roi]:
                        seed_rows.append({"rule": r, "roi": roi, "seed": sd, "persub": ps,
                                          "meanrdm": float(mv @ meanr)})
    seed = pd.DataFrame(seed_rows)
    seed.to_csv(OUT / "seed_per_seed.csv", index=False)
    spread = seed.groupby(["rule", "roi"]).agg(
        persub_mean=("persub", "mean"), persub_sd=("persub", lambda x: x.std(ddof=1)),
        persub_min=("persub", "min"), persub_max=("persub", "max"),
        meanrdm_mean=("meanrdm", "mean"), meanrdm_sd=("meanrdm", lambda x: x.std(ddof=1)),
        meanrdm_min=("meanrdm", "min"), meanrdm_max=("meanrdm", "max")).reset_index()
    spread.to_csv(OUT / "seed_spread.csv", index=False)
    h = pd.DataFrame(hier).groupby(["rule", "layer", "roi"]).persub.mean().reset_index()
    h.to_csv(OUT / "hierarchy.csv", index=False)
    print(f"seed + hierarchy done {time.time() - t0:.0f}s", flush=True)

    # B1: per-subject bootstrap
    with Pool(10, initializer=_init) as pool:
        draws = pool.map(subject_draw, range(-1, N_BOOT), chunksize=4)
    d = pd.DataFrame(draws).sort_values("boot")
    d.to_csv(OUT / "subject_draws.csv", index=False)
    pt, bt = d[d.boot == -1].iloc[0], d[d.boot >= 0]
    summ, pw = [], []
    for sub in SUBJECTS:
        for roi in ROIS:
            for r in RULES:
                k = f"{sub}|{r}|{roi}"
                summ.append({"subject": sub, "rule": r, "roi": roi, "point": float(pt[k]),
                             "ci_lo": float(bt[k].quantile(.025)), "ci_hi": float(bt[k].quantile(.975))})
            for i, a in enumerate(RULES):
                for b in RULES[i + 1:]:
                    ka, kb = f"{sub}|{a}|{roi}", f"{sub}|{b}|{roi}"
                    x = bt[ka] - bt[kb]
                    pw.append({"subject": sub, "roi": roi, "rule_a": a, "rule_b": b,
                               "diff": float(pt[ka] - pt[kb]), "ci_lo": float(x.quantile(.025)),
                               "ci_hi": float(x.quantile(.975))})
    pd.DataFrame(summ).to_csv(OUT / "subject_summary.csv", index=False)
    pd.DataFrame(pw).to_csv(OUT / "subject_pairwise.csv", index=False)
    # consistency check: mean over subjects of the identity draw = step2 persub|cr|bnfix
    s2 = pd.read_csv(REPO / "results/crossrun/step2_summary.csv").set_index("key")
    dev = max(abs(np.mean([pt[f"{s}|{r}|{roi}"] for s in SUBJECTS]) - s2.loc[f"persub|cr|bnfix|{r}|{roi}", "point"])
              for r in RULES for roi in ROIS)
    meta = {"boot_seed": BOOT_SEED, "n_boot": N_BOOT, "max_dev_vs_crossrun_step2": float(dev),
            "runtime_s": round(time.time() - t0, 1)}
    (OUT / "compute_extra.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))
    assert dev < 1e-9, dev


if __name__ == "__main__":
    main()

"""Step 3c: stimulus bootstrap of lower bound, per-subject model rho, and their difference.

Every resample draws 720 stimuli with replacement (shared by all quantities), reduces
every RDM to the sample and drops pairs whose two entries are the same stimulus
(they would be diagonal zeros). Ranks are computed once per vector per resample;
Spearman = dot product of centred unit-norm ranks.

Model rho per subject is averaged over subjects and over the 5 seeds, matching
Table 2's seed averaging (Table 2 itself uses the mean-RDM convention, which is
computed alongside as a control column).

Resample index -1 is the identity sample: it must reproduce step1 / step3a exactly.

Output: results/noise_ceiling_v2/step3c_bootstrap_draws.csv, step3c_summary.csv/.json
"""
import hashlib
import json
import os
import tempfile
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import REPO, RESULTS, ROIS, ROI_LAYER, SUBJECTS, N_STIM, FMRI_DIR, ranks

SEED = 20261002
N_BOOT = 1000
N_WORKERS = 10
SETS = {"original": REPO / "outputs/model_rdms",
        "bnfix": REPO / "learning_rules_outputs_bnfix/rdms/res224"}
RULES = ["random_weights", "backprop", "feedback_alignment", "predictive_coding", "stdp"]
LAYERS = sorted(set(ROI_LAYER.values()))
SEEDS = [f"seed_{i}" for i in range(5)]
# The cache name depends on the subject-RDM directory, so a run on other RDMs
# (NC_FMRI_DIR) can never silently reuse a stack built from different files.
CACHE = os.path.join(tempfile.gettempdir(), "noise_ceiling_v2_rdm_stack_"
                     + hashlib.sha256(str(FMRI_DIR.resolve()).encode()).hexdigest()[:12] + ".npy")


def items():
    out = [("fmri", roi, s, FMRI_DIR / f"fmri_rdm_{roi}_{s}.npy")
           for roi in ROIS for s in SUBJECTS]
    out += [("model", (st, r, l), sd, root / sd / f"rdm_{r}_{l}.npy")
            for st, root in SETS.items() for r in RULES for l in LAYERS for sd in SEEDS]
    return out


ITEMS = items()
_W = {}


def _init(boot_idx):
    _W["M"] = np.load(CACHE, mmap_mode="r")
    _W["boot"] = boot_idx
    _W["tri"] = np.triu_indices(N_STIM, k=1)


def one(b):
    M, (ta, tb) = _W["M"], _W["tri"]
    s = np.arange(N_STIM) if b < 0 else _W["boot"][b]
    keep = s[ta] != s[tb]
    rows, cols = s[ta][keep], s[tb][keep]
    res = {"boot": b, "n_pairs": int(keep.sum())}

    subj = {}
    for k, (kind, roi, _, _) in enumerate(ITEMS):
        if kind == "fmri":
            subj.setdefault(roi, []).append(np.asarray(M[k][rows, cols], float))
    sr = {}
    for roi, vecs in subj.items():
        rk = [ranks(v) for v in vecs]
        tot = np.sum(vecs, axis=0)
        r_mean = ranks(tot / 3)
        res[f"lower|{roi}"] = float(np.mean(
            [r @ ranks((tot - v) / 2) for r, v in zip(rk, vecs)]))
        res[f"upper|{roi}"] = float(np.mean([r @ r_mean for r in rk]))
        sr[roi] = (rk, r_mean)

    acc = {}
    for k, (kind, key, _, _) in enumerate(ITEMS):
        if kind != "model":
            continue
        st, rule, layer = key
        mv = ranks(np.asarray(M[k][rows, cols], float))
        for roi in ROIS:
            if ROI_LAYER[roi] != layer:
                continue
            rk, r_mean = sr[roi]
            a = acc.setdefault((st, rule, roi), {"persub": [], "meanrdm": []})
            a["persub"].append(float(np.mean([mv @ r for r in rk])))
            a["meanrdm"].append(float(mv @ r_mean))
    for (st, rule, roi), a in acc.items():
        assert len(a["persub"]) == len(SEEDS)
        res[f"persub|{st}|{rule}|{roi}"] = float(np.mean(a["persub"]))
        res[f"meanrdm|{st}|{rule}|{roi}"] = float(np.mean(a["meanrdm"]))
        res[f"diff|{st}|{rule}|{roi}"] = res[f"persub|{st}|{rule}|{roi}"] - res[f"lower|{roi}"]
    return res


def build_cache():
    if os.path.exists(CACHE):
        M = np.load(CACHE, mmap_mode="r")
        if M.shape == (len(ITEMS), N_STIM, N_STIM):
            return
    M = np.lib.format.open_memmap(CACHE, mode="w+", dtype=np.float64,
                                  shape=(len(ITEMS), N_STIM, N_STIM))
    for k, it in enumerate(ITEMS):
        a = np.load(it[3])
        assert a.shape == (N_STIM, N_STIM), it
        M[k] = a
    M.flush()
    del M


def main():
    t0 = time.time()
    build_cache()
    boot_idx = np.random.default_rng(SEED).integers(0, N_STIM, size=(N_BOOT, N_STIM))
    with Pool(N_WORKERS, initializer=_init, initargs=(boot_idx,)) as pool:
        draws = []
        for i, r in enumerate(pool.imap(one, range(-1, N_BOOT), chunksize=4)):
            draws.append(r)
            if i % 50 == 0:
                print(f"{i}/{N_BOOT + 1}  {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(draws).sort_values("boot")
    df.to_csv(RESULTS / "step3c_bootstrap_draws.csv", index=False)
    finalize(boot_idx, round(time.time() - t0, 1))


def runtime_from_log():
    """Elapsed seconds of the last progress line in step3c.log (for --finalize-only)."""
    lines = [l for l in (RESULTS / "step3c.log").read_text().splitlines()
             if l.split("/")[0].isdigit()]
    return float(lines[-1].split()[-1].rstrip("s"))


def finalize(boot_idx, runtime_s):
    df = pd.read_csv(RESULTS / "step3c_bootstrap_draws.csv")
    point = df[df.boot == -1].iloc[0]
    bt = df[df.boot >= 0]
    summ = []
    for c in df.columns:
        if "|" not in c:
            continue
        parts = c.split("|")
        q = bt[c].to_numpy()
        summ.append({"quantity": parts[0],
                     "rdm_set": parts[1] if len(parts) == 4 else "",
                     "rule": parts[2] if len(parts) == 4 else "",
                     "roi": parts[-1], "point": float(point[c]),
                     "ci_lo": float(np.percentile(q, 2.5)),
                     "ci_hi": float(np.percentile(q, 97.5)),
                     "boot_sd": float(q.std(ddof=1)),
                     "frac_boot_gt0": float((q > 0).mean())})
    summ = pd.DataFrame(summ)
    summ.to_csv(RESULTS / "step3c_summary.csv", index=False)

    # identity sample must reproduce steps 1 and 3a
    s1 = json.loads((RESULTS / "step1_bounds.json").read_text())["rois"]
    ctrl = pd.read_csv(RESULTS / "step3a_control.csv")
    dev_lower = max(abs(point[f"lower|{r}"] - s1[r]["lower"]) for r in ROIS)
    dev_model = 0.0
    for _, row in ctrl[ctrl.rule.isin(RULES)].iterrows():
        dev_model = max(dev_model,
                        abs(point[f"persub|{row.rdm_set}|{row.rule}|{row.roi}"] - row.rho_persub),
                        abs(point[f"meanrdm|{row.rdm_set}|{row.rule}|{row.roi}"] - row.rho_meanrdm))
    meta = {"seed": SEED, "n_boot": N_BOOT, "n_workers": N_WORKERS,
            "boot_idx_sha256": hashlib.sha256(boot_idx.tobytes()).hexdigest(),
            "n_pairs_median": int(bt.n_pairs.median()),
            "runtime_s": float(runtime_s),
            "identity_check": {"max_dev_lower_vs_step1": float(dev_lower),
                               "max_dev_model_vs_step3a": float(dev_model),
                               "ok": bool(dev_lower < 1e-9 and dev_model < 1e-9)}}
    (RESULTS / "step3c_summary.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=1))
    assert meta["identity_check"]["ok"], meta


if __name__ == "__main__":
    import sys
    if "--finalize-only" in sys.argv:  # recompute summaries from saved draws
        finalize(np.random.default_rng(SEED).integers(0, N_STIM, size=(N_BOOT, N_STIM)),
                 runtime_from_log())
    else:
        main()

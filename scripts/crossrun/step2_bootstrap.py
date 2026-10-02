"""Step 2: bounds, model RSA, model - lower; variants on shared stimulus resamples.

Variants
  all  all pairs (= noise_ceiling_v2; validation target)         persub + meanrdm
  cr   pair set P (cross-run in every subject)                    persub + meanrdm
  crb  P, block-cleaned per subject (subject RDM and model RDM    persub only
       demeaned with that subject's run-pair blocks; Nili bounds: each subject RDM with
       its own blocks, then leave-one-out). Mean-RDM convention not defined: the three
       subjects have different block structures, so no single cleaning applies to their mean.

Resamples: the noise_ceiling_v2 index matrix (seed 20261002, 1000 x 720), so `all` must
reproduce results/noise_ceiling_v2 exactly and all variants are paired.
Permutation null for the bounds: 1000x independent stimulus relabelling per subject
(seed 20261005); P and the blocks stay attached to positions.

Resumable (each finished draw / permutation is appended as one JSON line to
results/crossrun/_parts/{draws,perm}.jsonl; a rerun skips what is already there):
  py -3 step2_bootstrap.py draws --start -1 --stop 500   # identity sample + resamples 0..499
  py -3 step2_bootstrap.py draws --start 500 --stop 1000
  py -3 step2_bootstrap.py perm
  py -3 step2_bootstrap.py finalize                       # needs all 1001 draws + 1000 perms
Results do not depend on how the work is split (every draw is a pure function of its index).

Output: results/crossrun/step2_draws.csv, step2_perm.csv, step2_summary.csv, step2.json
"""
import argparse
import ctypes
import hashlib
import json
import os
import tempfile
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common_cr import (RESULTS, REPO, ROIS, SUBJECTS, N_STIM, TRI, SEED, load, run_labels,
                       ranks, pair_set, block_ids, demean_blocks)
from common import ROI_LAYER

BOOT_SEED, N_BOOT, N_PERM = 20261002, 1000, 1000
SETS = {"original": REPO / "outputs/model_rdms", "bnfix": REPO / "learning_rules_outputs_bnfix/rdms/res224"}
RULES = ["random_weights", "backprop", "feedback_alignment", "predictive_coding", "stdp"]
LAYERS = sorted(set(ROI_LAYER.values()))
SEEDS = [f"seed_{i}" for i in range(5)]
CACHE = os.path.join(tempfile.gettempdir(), "crossrun_model_stack.npy")
PARTS = RESULTS / "_parts"
LUM = RESULTS.parents[0] / "sub03_check" / "luminance.csv"
MODEL_ITEMS = [(st, r, l, sd) for st in SETS for r in RULES for l in LAYERS for sd in SEEDS]
_W = {}


def _init(boot):
    _W["boot"], _W["runs"] = boot, run_labels()
    _W["fmri"] = {roi: load("old", roi) for roi in ROIS}
    _W["M"] = np.load(CACHE, mmap_mode="r")
    _W["lum"] = pd.read_csv(LUM).luminance.values


def bounds(vecs):
    rk = [ranks(v) for v in vecs]
    tot = np.sum(vecs, axis=0)
    rm = ranks(tot / 3)
    return (float(np.mean([r @ ranks((tot - v) / 2) for r, v in zip(rk, vecs)])),
            float(np.mean([r @ rm for r in rk])), rk, rm)


def one(b):
    """b < 0: identity sample; b >= 0: bootstrap resample b."""
    s = None if b < 0 else _W["boot"][b]
    runs = _W["runs"]
    out = {"boot": b}
    # pair sets
    ss = np.arange(N_STIM) if s is None else s
    a_all, b_all = ss[TRI[0]], ss[TRI[1]]
    k_all = a_all != b_all
    ra, ca = a_all[k_all], b_all[k_all]
    rp, cp, _ = pair_set(s, runs)
    bid = {sub: block_ids(rp, cp, runs[sub]) for sub in SUBJECTS}
    out["n_pairs_all"], out["n_pairs_cr"] = int(len(ra)), int(len(rp))
    lum = _W["lum"]
    lv = {"all": np.abs(lum[ra] - lum[ca]), "cr": np.abs(lum[rp] - lum[cp])}
    lvr = {k: ranks(v) for k, v in lv.items()}
    brain = {}
    for roi in ROIS:
        F = _W["fmri"][roi]
        for var, (r_, c_) in {"all": (ra, ca), "cr": (rp, cp)}.items():
            vecs = [x[r_, c_] for x in F]
            lo, up, rk, rm = bounds(vecs)
            out[f"lower|{var}|{roi}"], out[f"upper|{var}|{roi}"] = lo, up
            brain[(var, roi)] = (rk, rm)
            out[f"lum_persub|{var}|{roi}"] = float(np.mean([lvr[var] @ r for r in rk]))
            out[f"lum_meanrdm|{var}|{roi}"] = float(lvr[var] @ rm)
        vecs = [demean_blocks(x[rp, cp], bid[sub]) for x, sub in zip(F, SUBJECTS)]
        lo, up, rk, _ = bounds(vecs)
        out[f"lower|crb|{roi}"], out[f"upper|crb|{roi}"] = lo, up
        brain[("crb", roi)] = (rk, None)
        out[f"lum_persub|crb|{roi}"] = float(np.mean(
            [ranks(demean_blocks(lv["cr"], bid[sub])) @ r for sub, r in zip(SUBJECTS, rk)]))

    acc = {}
    M = _W["M"]
    for k, (st, rule, layer, sd) in enumerate(MODEL_ITEMS):
        X = M[k]
        mv = {"all": ranks(np.asarray(X[ra, ca], float)), "cr": ranks(np.asarray(X[rp, cp], float))}
        xc = np.asarray(X[rp, cp], float)
        mcrb = [ranks(demean_blocks(xc, bid[sub])) for sub in SUBJECTS]
        for roi in ROIS:
            if ROI_LAYER[roi] != layer:
                continue
            for var in ("all", "cr"):
                rk, rm = brain[(var, roi)]
                acc.setdefault((var, "persub", st, rule, roi), []).append(
                    float(np.mean([mv[var] @ r for r in rk])))
                acc.setdefault((var, "meanrdm", st, rule, roi), []).append(float(mv[var] @ rm))
            rk, _ = brain[("crb", roi)]
            acc.setdefault(("crb", "persub", st, rule, roi), []).append(
                float(np.mean([m @ r for m, r in zip(mcrb, rk)])))
    for (var, conv, st, rule, roi), v in acc.items():
        assert len(v) == len(SEEDS)
        out[f"{conv}|{var}|{st}|{rule}|{roi}"] = float(np.mean(v))
        if conv == "persub":
            out[f"diff|{var}|{st}|{rule}|{roi}"] = float(np.mean(v)) - out[f"lower|{var}|{roi}"]
    return out


def perm(p):
    """Permutation null of the bounds (cr, crb); P and blocks fixed on positions."""
    rng = np.random.default_rng([SEED, p])
    runs = _W["runs"]
    rp, cp, _ = pair_set(None, runs)
    bid = {sub: block_ids(rp, cp, runs[sub]) for sub in SUBJECTS}
    out = {"perm": p}
    for roi in ROIS:
        vecs = []
        for x in _W["fmri"][roi]:
            q = rng.permutation(N_STIM)
            vecs.append(x[np.ix_(q, q)][rp, cp])
        lo, up, _, _ = bounds(vecs)
        out[f"lower|cr|{roi}"], out[f"upper|cr|{roi}"] = lo, up
        lo, up, _, _ = bounds([demean_blocks(v, bid[s]) for v, s in zip(vecs, SUBJECTS)])
        out[f"lower|crb|{roi}"], out[f"upper|crb|{roi}"] = lo, up
    return out


def build_cache():
    if os.path.exists(CACHE) and np.load(CACHE, mmap_mode="r").shape[0] == len(MODEL_ITEMS):
        return
    M = np.lib.format.open_memmap(CACHE, mode="w+", dtype=np.float64,
                                  shape=(len(MODEL_ITEMS), N_STIM, N_STIM))
    for k, (st, r, l, sd) in enumerate(MODEL_ITEMS):
        M[k] = np.load(SETS[st] / sd / f"rdm_{r}_{l}.npy")
    M.flush()


def _boot_idx():
    boot = np.random.default_rng(BOOT_SEED).integers(0, N_STIM, size=(N_BOOT, N_STIM))
    v2 = json.loads((REPO / "results/noise_ceiling_v2/step3c_summary.json").read_text())
    assert hashlib.sha256(boot.tobytes()).hexdigest() == v2["boot_idx_sha256"], "resamples differ from v2"
    return boot


def _read_parts(name, key):
    """Completed records of a part file; a truncated last line (killed run) is ignored."""
    f = PARTS / f"{name}.jsonl"
    done = {}
    if f.exists():
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[r[key]] = r
    return done


def _run(name, key, func, todo, boot, t0):
    PARTS.mkdir(parents=True, exist_ok=True)
    # keep Windows from sleeping while this process runs (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    with open(PARTS / f"{name}.jsonl", "a") as fh, \
            Pool(10, initializer=_init, initargs=(boot,)) as pool:
        for i, r in enumerate(pool.imap_unordered(func, todo, chunksize=2)):
            fh.write(json.dumps(r) + "\n"); fh.flush()
            if i % 50 == 0:
                print(f"{name} {i + 1}/{len(todo)} (last {key}={r[key]}) {time.time() - t0:.0f}s", flush=True)


def finalize(boot, t0):
    d, p = _read_parts("draws", "boot"), _read_parts("perm", "perm")
    miss_d = sorted(set(range(-1, N_BOOT)) - set(d)); miss_p = sorted(set(range(N_PERM)) - set(p))
    assert not miss_d and not miss_p, f"missing draws {miss_d[:10]}... perms {miss_p[:10]}..."
    draws = pd.DataFrame([d[b] for b in range(-1, N_BOOT)])
    draws.to_csv(RESULTS / "step2_draws.csv", index=False)
    perms = pd.DataFrame([p[k] for k in range(N_PERM)])
    perms.to_csv(RESULTS / "step2_perm.csv", index=False)

    pt, bt = draws[draws.boot == -1].iloc[0], draws[draws.boot >= 0]
    summ = []
    for c in draws.columns:
        if "|" not in c:
            continue
        q = bt[c]
        summ.append({"key": c, "point": float(pt[c]), "ci_lo": float(q.quantile(0.025)),
                     "ci_hi": float(q.quantile(0.975)), "boot_sd": float(q.std(ddof=1))})
    summ = pd.DataFrame(summ)
    summ.to_csv(RESULTS / "step2_summary.csv", index=False)

    # validation: `all` must reproduce noise_ceiling_v2 (point and bootstrap CI)
    v2s = pd.read_csv(REPO / "results/noise_ceiling_v2/step3c_summary.csv").fillna("")
    dev = 0.0
    for _, r in v2s.iterrows():
        key = (f"{r.quantity}|all|{r.roi}" if r.quantity in ("lower", "upper")
               else f"{r.quantity}|all|{r.rdm_set}|{r.rule}|{r.roi}")
        row = summ[summ.key == key].iloc[0]
        dev = max(dev, abs(row.point - r.point), abs(row.ci_lo - r.ci_lo), abs(row.ci_hi - r.ci_hi))
    meta = {"boot_seed": BOOT_SEED, "perm_seed": SEED, "n_boot": N_BOOT, "n_perm": N_PERM,
            "n_pairs_cr_identity": int(pt.n_pairs_cr), "n_pairs_all_identity": int(pt.n_pairs_all),
            "n_pairs_cr_boot_median": int(bt.n_pairs_cr.median()),
            "validation_max_dev_vs_noise_ceiling_v2": float(dev),
            "perm_null": {c: {"mean": float(perms[c].mean()), "sd": float(perms[c].std(ddof=1)),
                              "q975": float(perms[c].quantile(0.975))}
                          for c in perms.columns if c != "perm"},
            "finalize_runtime_s": round(time.time() - t0, 1)}
    (RESULTS / "step2.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "perm_null"}, indent=1))
    assert dev < 1e-9, dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["draws", "perm", "finalize"])
    ap.add_argument("--start", type=int, default=-1)
    ap.add_argument("--stop", type=int, default=N_BOOT)
    a = ap.parse_args()
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    boot = _boot_idx()
    if a.stage == "draws":
        build_cache()
        done = _read_parts("draws", "boot")
        todo = [b for b in range(max(a.start, -1), min(a.stop, N_BOOT)) if b not in done]
        print(f"draws: {len(todo)} to do, {len(done)} already done", flush=True)
        if todo:
            _run("draws", "boot", one, todo, boot, t0)
    elif a.stage == "perm":
        build_cache()
        done = _read_parts("perm", "perm")
        todo = [k for k in range(N_PERM) if k not in done]
        print(f"perm: {len(todo)} to do, {len(done)} already done", flush=True)
        if todo:
            _run("perm", "perm", perm, todo, boot, t0)
    else:
        finalize(boot, t0)
    print(f"{a.stage} finished {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

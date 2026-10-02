"""Step 1 diagnostics (before any model or effect number).

a) Lag within run (old RDMs, all within-run pairs of each subject): mean dissimilarity
   per |position difference| in the 82-trial run; Spearman(dissimilarity, lag).
b) Run-pair structure on P: per subject x ROI the 10x10 matrix of mean dissimilarity
   per run pair (diagonal left out). 01-02 (identical order): Spearman between their 45
   run-pair means; exact null over all 10! relabellings of sub-02's runs.
c) Luminance vs subject RDM on P vs on P block-cleaned (luminance RDM and subject RDM
   demeaned with that subject's blocks). STOP criterion (fixed): block cleaning lowers the
   V1 luminance fit of sub-01 or sub-02 by more than 30 %.
d) Subject agreement: all pairs | P | P block-cleaned (each RDM with its own blocks).

Output: results/crossrun/step1_*.csv, step1.json
"""
import itertools
import json

import numpy as np
import pandas as pd

from common_cr import (RESULTS, ROIS, SUBJECTS, N_STIM, TRI, N_RUNS, load, run_labels, ranks,
                       stim_order, pair_set, block_ids, demean_blocks)
from common_runz import DATA_DIR

PAIRS = [(0, 1), (0, 2), (1, 2)]


def exact_null(m1, m2):
    """Spearman of the 45 upper-triangle run-pair means of m1 vs m2, and its distribution
    over all 10! relabellings of m2's runs."""
    iu = np.triu_indices(N_RUNS, 1)
    x = ranks(m1[iu])
    obs = float(x @ ranks(m2[iu]))
    perms = np.array(list(itertools.permutations(range(N_RUNS))), dtype=np.int8)
    null = np.empty(len(perms))
    for c in range(0, len(perms), 200_000):
        P = perms[c:c + 200_000]
        V = m2[P[:, iu[0]], P[:, iu[1]]]                      # (n, 45)
        R = V.argsort(1).argsort(1).astype(float)              # no ties among block means
        R -= R.mean(1, keepdims=True)
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        null[c:c + len(P)] = R @ x
    return obs, null


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    runs = run_labels()
    rows, cols, keep = pair_set(runs=runs)
    lum = pd.read_csv(REPO_LUM).luminance.values
    lumv = np.abs(lum[rows] - lum[cols])
    bid = {s: block_ids(rows, cols, runs[s]) for s in SUBJECTS}
    out = {"n_pairs_total": int(len(TRI[0])), "n_pairs_P": int(keep.sum()),
           "frac_pairs_P": float(keep.mean())}

    # position within the run for (a)
    order = stim_order()
    pos = {}
    for s in SUBJECTS:
        st = pd.read_csv(DATA_DIR / f"{s}_task-things_stimulus-metadata.csv")
        st["pos"] = st.groupby(["session", "run"]).cumcount()
        pos[s] = st.set_index("stimulus").loc[order, "pos"].values

    lag_rows, lag_sum, bl_rows, b_rows, c_rows, d_rows = [], [], [], [], [], []
    for roi in ROIS:
        R = load("old", roi)
        # a) lag
        for si, s in enumerate(SUBJECTS):
            same = runs[s][TRI[0]] == runs[s][TRI[1]]
            lag = np.abs(pos[s][TRI[0]] - pos[s][TRI[1]])[same]
            v = R[si][TRI][same]
            df = pd.DataFrame({"lag": lag, "d": v}).groupby("lag").d.agg(["mean", "count"])
            for lg, r in df.iterrows():
                lag_rows.append({"roi": roi, "subject": s, "lag": int(lg),
                                 "mean_dissimilarity": float(r["mean"]), "n": int(r["count"])})
            lag_sum.append({"roi": roi, "subject": s, "spearman_diss_vs_lag": float(ranks(v) @ ranks(lag.astype(float))),
                            "mean_lag1": float(df.loc[1, "mean"]), "mean_lag_ge10": float(v[lag >= 10].mean()),
                            "mean_cross_run_P": float(R[si][rows, cols].mean())})
        # b) run-pair matrices
        mats = {}
        for si, s in enumerate(SUBJECTS):
            v = R[si][rows, cols]
            sm = np.bincount(bid[s], weights=v, minlength=100); n = np.bincount(bid[s], minlength=100)
            M = np.full((N_RUNS, N_RUNS), np.nan)
            for k in range(100):
                i, j = divmod(k, N_RUNS)
                if i < j:
                    M[i, j] = M[j, i] = sm[k] / n[k]
                    bl_rows.append({"roi": roi, "subject": s, "run_i": i + 1, "run_j": j + 1,
                                    "mean_dissimilarity": sm[k] / n[k], "n_pairs": int(n[k])})
            mats[s] = M
        obs, null = exact_null(mats["sub-01"], mats["sub-02"])
        b_rows.append({"roi": roi, "spearman_01_02_runpair_means": obs,
                       "null_mean": float(null.mean()), "null_sd": float(null.std()),
                       "p_exact_one_sided": float((null >= obs).mean()),
                       "n_permutations": int(len(null))})
        # c) luminance
        for si, s in enumerate(SUBJECTS):
            v = R[si][rows, cols]
            cr = float(ranks(lumv) @ ranks(v))
            crb = float(ranks(demean_blocks(lumv, bid[s])) @ ranks(demean_blocks(v, bid[s])))
            allp = float(ranks(np.abs(lum[TRI[0]] - lum[TRI[1]])) @ ranks(R[si][TRI]))
            c_rows.append({"roi": roi, "subject": s, "all_pairs": allp, "cross_run": cr,
                           "cross_run_block_cleaned": crb,
                           "frac_change_cr_to_crb": (crb - cr) / cr if cr != 0 else np.nan})
        # d) agreement
        for i, j in PAIRS:
            vi, vj = R[i][rows, cols], R[j][rows, cols]
            d_rows.append({"roi": roi, "pair": f"{i+1:02d}-{j+1:02d}",
                           "all_pairs": float(ranks(R[i][TRI]) @ ranks(R[j][TRI])),
                           "cross_run": float(ranks(vi) @ ranks(vj)),
                           "cross_run_block_cleaned": float(
                               ranks(demean_blocks(vi, bid[SUBJECTS[i]])) @
                               ranks(demean_blocks(vj, bid[SUBJECTS[j]])))})
        print(roi, "done", flush=True)

    for name, rws in [("lag", lag_rows), ("lag_summary", lag_sum), ("runpair_means", bl_rows),
                      ("runpair_test", b_rows), ("luminance", c_rows), ("agreement", d_rows)]:
        pd.DataFrame(rws).to_csv(RESULTS / f"step1_{name}.csv", index=False)
    c = pd.DataFrame(c_rows)
    v1 = c[(c.roi == "V1") & c.subject.isin(["sub-01", "sub-02"])]
    b = pd.DataFrame(b_rows)
    out.update({
        "runpair_significant_any_roi": bool((b.p_exact_one_sided < 0.05).any()),
        "runpair_significant_rois": b.loc[b.p_exact_one_sided < 0.05, "roi"].tolist(),
        "primary_variant": ("cross_run_block_cleaned" if (b.p_exact_one_sided < 0.05).any()
                            else "cross_run"),
        "stop_luminance_drop_gt_30pct": bool((v1.frac_change_cr_to_crb < -0.30).any()),
        "luminance_V1_frac_change_sub01_sub02": v1.set_index("subject").frac_change_cr_to_crb.to_dict(),
    })
    (RESULTS / "step1.json").write_text(json.dumps(out, indent=2))
    pd.set_option("display.width", 200)
    for name in ("lag_summary", "runpair_test", "luminance", "agreement"):
        print(pd.read_csv(RESULTS / f"step1_{name}.csv").round(4).to_string())
    print(json.dumps(out, indent=1))


REPO_LUM = RESULTS.parents[0] / "sub03_check" / "luminance.csv"

if __name__ == "__main__":
    main()

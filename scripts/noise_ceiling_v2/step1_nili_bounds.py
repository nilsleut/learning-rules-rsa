"""Step 1: Nili bounds per ROI + stimulus-permutation null for the upper bound.

Also reproduces the original estimator (learning_rules_v8.py:500-513) so the
report's "old" table is generated, not typed.

Output: results/noise_ceiling_v2/step1_bounds.json
"""
import json

import numpy as np
from scipy.stats import spearmanr

from common import RESULTS, ROIS, N_STIM, load_fmri, triu, nili_bounds, spearman

SEED = 20261001
N_PERM = 1000
PROVENANCE_V1 = {"lower": 0.0570, "upper": 0.5752}  # NOISE_CEILING_PROVENANCE.md sec. 3


def original_estimator(rdms, n_splits=200):
    """Verbatim logic of learning_rules_v8.py:noise_ceiling (seed 42)."""
    rng = np.random.default_rng(42)
    idx = np.triu_indices(rdms[0].shape[0], k=1)
    rhos = []
    for _ in range(n_splits):
        perm = rng.permutation(len(rdms))
        half1 = np.mean([rdms[i][idx] for i in perm[:len(rdms) // 2]], 0)
        half2 = np.mean([rdms[i][idx] for i in perm[len(rdms) // 2:]], 0)
        r, _ = spearmanr(half1, half2)
        rhos.append(2 * r / (1 + r) if r < 1 else 1.0)
    return float(np.percentile(rhos, 2.5)), float(np.mean(rhos))


def main():
    rng = np.random.default_rng(SEED)
    out = {"seed": SEED, "n_perm": N_PERM, "rois": {}}
    for roi in ROIS:
        rdms = load_fmri(roi)
        vecs = [triu(r) for r in rdms]
        # helper sanity check against scipy
        assert abs(spearman(vecs[0], vecs[1]) - spearmanr(vecs[0], vecs[1])[0]) < 1e-10

        b = nili_bounds(vecs)
        pairwise = {f"{i+1}-{j+1}": spearman(vecs[i], vecs[j])
                    for i in range(3) for j in range(i + 1, 3)}
        p25, mean_sb = original_estimator(rdms)

        null_u, null_l = [], []
        for _ in range(N_PERM):
            pv = []
            for r in rdms:  # relabel stimuli independently per subject (rows+cols)
                p = rng.permutation(N_STIM)
                pv.append(triu(r[np.ix_(p, p)]))
            nb = nili_bounds(pv)
            null_u.append(nb["upper"])
            null_l.append(nb["lower"])
        null_u, null_l = np.array(null_u), np.array(null_l)

        out["rois"][roi] = {
            **b,
            "pairwise_spearman": pairwise,
            "upper_null_mean": float(null_u.mean()),
            "upper_null_sd": float(null_u.std(ddof=1)),
            "upper_null_q975": float(np.percentile(null_u, 97.5)),
            "upper_minus_null_mean": b["upper"] - float(null_u.mean()),
            "upper_perm_p": float((1 + (null_u >= b["upper"]).sum()) / (N_PERM + 1)),
            "lower_null_mean": float(null_l.mean()),
            "lower_null_sd": float(null_l.std(ddof=1)),
            "lower_perm_p": float((1 + (null_l >= b["lower"]).sum()) / (N_PERM + 1)),
            "old_code_p2p5": p25,
            "old_code_mean_sb": mean_sb,
        }
        print(f"{roi}: lower={b['lower']:.4f} upper={b['upper']:.4f} "
              f"null_upper={null_u.mean():.4f}±{null_u.std(ddof=1):.4f} "
              f"old=({p25:.4f},{mean_sb:.4f})", flush=True)

    v1 = out["rois"]["V1"]
    out["provenance_check"] = {
        k: {"provenance": PROVENANCE_V1[k], "recomputed": v1[k],
            "abs_diff": abs(PROVENANCE_V1[k] - v1[k]),
            "ok": abs(PROVENANCE_V1[k] - v1[k]) <= 0.002}
        for k in PROVENANCE_V1}
    print("provenance check:", out["provenance_check"])

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "step1_bounds.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

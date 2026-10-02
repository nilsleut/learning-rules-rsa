"""Checks 1-3 on the stored RDMs (same loader as noise_ceiling_v2/step1_nili_bounds.py).

1. Subject-subject Spearman and Pearson, all ROIs.
2. Luminance RDM vs each subject's RDM, all ROIs (and vs the mean RDM, to
   reproduce the paper's rho = 0.075 at V1).
3. Model vs each subject: read from results/noise_ceiling_v2/step3a_control_per_seed.csv
   (already computed per subject and seed for the reference set, not recomputed).

Output: results/sub03_check/check1_pairs.csv, check2_luminance.csv,
        check3_models.csv, luminance.csv, check123.json
"""
import json

import numpy as np
import pandas as pd

from common_sub03 import (RESULTS, REPO, ROIS, SUBJECTS, load_fmri, triu, spearman,
                          pearson, stim_order, image_paths, luminance, csc)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    vec = {roi: [triu(r) for r in load_fmri(roi)] for roi in ROIS}

    # 1 ────────────────────────────────────────────────────────────────────────
    rows = []
    for roi in ROIS:
        v = vec[roi]
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            rows.append({"roi": roi, "pair": f"{i+1:02d}-{j+1:02d}",
                         "spearman": spearman(v[i], v[j]), "pearson": pearson(v[i], v[j])})
    c1 = pd.DataFrame(rows)
    c1.to_csv(RESULTS / "check1_pairs.csv", index=False)

    # 2 ────────────────────────────────────────────────────────────────────────
    orders = {s: stim_order(s) for s in SUBJECTS}
    assert all(orders[s] == orders["sub-01"] for s in SUBJECTS), "stim orders differ"
    paths = image_paths(orders["sub-01"])
    exact = [p is not None and p.stem == s.replace(".jpg", "")
             for p, s in zip(paths, orders["sub-01"])]
    lum = luminance(paths)
    pd.DataFrame({"stimulus": orders["sub-01"], "image": [p.name if p is not None else "" for p in paths],
                  "exact_filename_match": exact, "luminance": lum}) \
        .to_csv(RESULTS / "luminance.csv", index=False)
    lv = triu(csc.abs_rdm(lum))
    rows = []
    for roi in ROIS:
        v = vec[roi]
        for s, x in zip(SUBJECTS, v):
            rows.append({"roi": roi, "target": s, "spearman": spearman(lv, x),
                         "pearson": pearson(lv, x)})
        m = np.mean(v, axis=0)
        rows.append({"roi": roi, "target": "mean-RDM", "spearman": spearman(lv, m),
                     "pearson": pearson(lv, m)})
    c2 = pd.DataFrame(rows)
    c2.to_csv(RESULTS / "check2_luminance.csv", index=False)

    # 3 ────────────────────────────────────────────────────────────────────────
    ps = pd.read_csv(REPO / "results/noise_ceiling_v2/step3a_control_per_seed.csv")
    ps = ps[(ps.rdm_set == "original")]
    long = ps.melt(id_vars=["rule", "roi", "seed"],
                   value_vars=["rho_sub01", "rho_sub02", "rho_sub03"],
                   var_name="subject", value_name="rho")
    long["subject"] = long.subject.str.replace("rho_sub", "sub-")
    c3 = (long.groupby(["rule", "roi", "subject"]).rho
          .agg(mean="mean", sd=lambda x: x.std(ddof=1), n="count").reset_index())
    c3.to_csv(RESULTS / "check3_models.csv", index=False)

    out = {
        "n_stimuli": len(paths),
        "n_exact_image_match": int(sum(exact)),
        "luminance_vs_meanrdm_V1": float(c2[(c2.roi == "V1") &
                                            (c2.target == "mean-RDM")].spearman.iloc[0]),
        "sub03_ratio_luminance": {roi: float(
            c2[(c2.roi == roi) & (c2.target == "sub-03")].spearman.iloc[0] /
            c2[(c2.roi == roi) & c2.target.isin(["sub-01", "sub-02"])].spearman.mean())
            for roi in ROIS},
    }
    (RESULTS / "check123.json").write_text(json.dumps(out, indent=2))
    print(c1.round(4).to_string()); print(c2.round(4).to_string())
    print(c3[c3.rule == "random_weights"].round(4).to_string()); print(out)


if __name__ == "__main__":
    main()

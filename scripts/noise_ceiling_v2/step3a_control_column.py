"""Step 3a: control column. Can the published Table 2 values be reproduced?

For every condition and seed: rho vs. the 3-subject mean RDM (published
convention) and rho per subject (averaged over subjects). Done for two RDM sets:
  bnfix    = learning_rules_outputs_bnfix/rdms/res224/seed_*   (post BN fix, Aug)
  original = outputs/model_rdms/seed_*                         (run behind Table 2)
Published values are parsed from the paper's Table 2, not typed.

Output: results/noise_ceiling_v2/step3a_control.csv / .json
"""
import json
import re

import numpy as np
import pandas as pd

from common import REPO, RESULTS, ROIS, ROI_LAYER, load_fmri, triu, ranks

TEX = REPO / "paper/arxiv_upload_learning_rules_v2/learning_rules_rsa_paper_v2.tex"
SETS = {
    "bnfix": REPO / "learning_rules_outputs_bnfix/rdms/res224",
    "original": REPO / "outputs/model_rdms",
}
RULES = {  # file key -> paper column
    "random_weights": "Random Weights", "backprop": "BP",
    "feedback_alignment": "FA", "predictive_coding": "PC", "stdp": "STDP",
    "random_weights_(bn-calibrated)": None,  # not in the paper
}
PAPER_COLS = ["Random Weights", "BP", "FA", "PC", "STDP"]


def parse_table2():
    lines = TEX.read_text(encoding="utf-8").splitlines()
    pub = {}
    for ln_no, line in enumerate(lines, 1):
        m = re.match(r"^(V1|V2|LOC|IT)\s*&\s*(\w+)\s*&(.*)\\\\", line.strip())
        if not m:
            continue
        cells = m.group(3).split("&")
        vals = [float(re.search(r"(-?\d*\.\d+)", c).group(1)) for c in cells]
        assert len(vals) == 5 and m.group(2) == ROI_LAYER[m.group(1)], line
        pub[m.group(1)] = {"line": ln_no, **dict(zip(PAPER_COLS, vals))}
    assert set(pub) == set(ROIS), pub
    return pub


def main():
    pub = parse_table2()
    brain = {}
    for roi in ROIS:
        vecs = [triu(r) for r in load_fmri(roi)]
        brain[roi] = {"subj": [ranks(v) for v in vecs],
                      "mean": ranks(np.mean(vecs, axis=0))}

    rows = []
    for set_name, root in SETS.items():
        seeds = sorted(p for p in root.glob("seed_*") if p.is_dir())
        for key, col in RULES.items():
            for roi in ROIS:
                f = f"rdm_{key}_{ROI_LAYER[roi]}.npy"
                if not (seeds[0] / f).exists():
                    continue
                for sd in seeds:
                    mv = ranks(triu(np.load(sd / f)))
                    per = [float(mv @ s) for s in brain[roi]["subj"]]
                    rows.append({"rdm_set": set_name, "rule": key, "paper_col": col,
                                 "roi": roi, "layer": ROI_LAYER[roi], "seed": sd.name,
                                 "rho_meanrdm": float(mv @ brain[roi]["mean"]),
                                 "rho_sub01": per[0], "rho_sub02": per[1],
                                 "rho_sub03": per[2], "rho_persub": float(np.mean(per))})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "step3a_control_per_seed.csv", index=False)

    agg = (df.groupby(["rdm_set", "rule", "paper_col", "roi"], dropna=False)
             .agg(n_seeds=("seed", "count"), rho_meanrdm=("rho_meanrdm", "mean"),
                  rho_persub=("rho_persub", "mean")).reset_index())
    agg["published"] = [pub[r][c] if isinstance(c, str) else np.nan
                        for r, c in zip(agg.roi, agg.paper_col)]
    agg["tex_line"] = [pub[r]["line"] for r in agg.roi]
    agg["abs_diff"] = (agg.rho_meanrdm - agg.published).abs()
    # published values are printed to 3 decimals
    agg["reproduced"] = np.where(agg.published.isna(), np.nan,
                                 agg.abs_diff <= 0.0005 + 1e-9)
    agg.to_csv(RESULTS / "step3a_control.csv", index=False)

    verdict = {}
    for s in SETS:
        a = agg[(agg.rdm_set == s) & agg.published.notna()]
        verdict[s] = {"n_cells": int(len(a)), "n_reproduced": int(a.reproduced.sum()),
                      "failed": a.loc[~a.reproduced.astype(bool),
                                      ["paper_col", "roi", "rho_meanrdm", "published"]]
                               .round(4).to_dict("records")}
    (RESULTS / "step3a_control.json").write_text(json.dumps(
        {"published_table2": pub, "verdict": verdict}, indent=2))
    print(agg.round(4).to_string())
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()

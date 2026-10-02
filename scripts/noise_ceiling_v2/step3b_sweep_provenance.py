"""Step 3b: provenance of the sweeps that report BP / FA / Random at 224 px.

Seed statistics per sweep and ROI, and whether Table - bnfix lies within 2 SE.
SE of a sweep mean = seed SD / sqrt(n_seeds); SE of a difference between two
independently trained sweeps = sqrt(SE_a^2 + SE_b^2).
Convention: rho vs. the 3-subject mean RDM (the Table 2 convention).

Inventory facts (mtime, config) are recorded here from the files named below;
the config lines are quoted in the report with file:line references.

Output: results/noise_ceiling_v2/step3b_sweeps.csv / .json
"""
import datetime
import hashlib
import json
import os

import numpy as np
import pandas as pd

from common import REPO, RESULTS, ROIS, ROI_LAYER

SWEEPS = {
    "table": {
        "csv": "outputs/rsa_results_seeds.csv",
        "script": "learning_rules_v8.py (Kaggle notebook learning_rules_v8.ipynb, T4)",
        "config": "N_EPOCHS=40, BATCH=64, LR=1e-3, N_CIFAR=8000; CIFAR subset drawn once "
                  "with seed 42 and shared by all seeds (learning_rules_v8.py:987-992); "
                  "num_workers=0; pre-BN-fix",
    },
    "june_sweep": {
        "csv": "outputs/rsa_resolution_sweep.csv",
        "script": "learning_rules_v9_sweep_modal.py, June version "
                  "(copy: Projekte_2/merged_paper/learning_rules_v9_sweep_modal.py, 2026-06-04)",
        "config": "N_EPOCHS=40, BATCH=128, LR=1e-3, N_CIFAR=8000; CIFAR subset re-drawn "
                  "per seed (manual_seed(seed) before randperm); num_workers=4; Modal T4; "
                  "pre-BN-fix; retrained from scratch",
    },
    "bnfix": {
        "csv": "learning_rules_outputs_bnfix/rsa_resolution_sweep.csv",
        "script": "learning_rules_v10_sweep_modal.py (= v9 August version + merge check)",
        "config": "as june_sweep, plus BN-mode fix for PC/STDP and explicit train/eval "
                  "mode per phase; retrained from scratch",
    },
}
DUPLICATES = {"old_sweep.csv": "outputs/rsa_resolution_sweep.csv",
              "bnfix_sweep.csv": "learning_rules_outputs_bnfix/rsa_resolution_sweep.csv"}
RULES = ["Backprop", "Feedback Alignment", "Random Weights"]


def sha(p):
    return hashlib.sha256((REPO / p).read_bytes()).hexdigest()


def load(name):
    d = pd.read_csv(REPO / SWEEPS[name]["csv"])
    if "res" in d:
        d = d[d.res == 224]
    d = d[d.rule.isin(RULES)]
    return d[[ROI_LAYER[r] == l for r, l in zip(d.roi, d.layer)]]


def main():
    inv = {}
    for name, meta in SWEEPS.items():
        d = load(name)
        p = REPO / meta["csv"]
        inv[name] = {**meta,
                     "mtime": datetime.datetime.fromtimestamp(os.path.getmtime(p))
                                      .strftime("%Y-%m-%d %H:%M"),
                     "seeds": sorted(int(s) for s in d.seed.unique()),
                     "n_seeds": int(d.seed.nunique())}
    dup = {k: {"same_as": v, "identical": sha(k) == sha(v)} for k, v in DUPLICATES.items()}

    stats = []
    for name in SWEEPS:
        g = load(name).groupby(["rule", "roi"]).rho
        s = g.agg(mean="mean", sd=lambda x: x.std(ddof=1), n="count").reset_index()
        s["se"] = s.sd / np.sqrt(s.n)
        s["sweep"] = name
        stats.append(s)
    stats = pd.concat(stats)

    rows = []
    for rule in RULES:
        for roi in ROIS:
            c = {sw: stats[(stats.sweep == sw) & (stats.rule == rule) & (stats.roi == roi)]
                 .iloc[0] for sw in SWEEPS}
            row = {"rule": rule, "roi": roi, "layer": ROI_LAYER[roi]}
            for sw in SWEEPS:
                row[f"{sw}_mean"] = c[sw]["mean"]
                row[f"{sw}_sd"] = c[sw]["sd"]
            for a, b in [("table", "bnfix"), ("table", "june_sweep"), ("june_sweep", "bnfix")]:
                diff = c[a]["mean"] - c[b]["mean"]
                se = float(np.hypot(c[a]["se"], c[b]["se"]))
                row[f"{a}_minus_{b}"] = diff
                row[f"{a}_minus_{b}_se"] = se
                row[f"{a}_minus_{b}_z"] = diff / se if se > 0 else 0.0
                row[f"{a}_minus_{b}_within_2se"] = bool(abs(diff) <= 2 * se) if se > 0 \
                    else bool(abs(diff) < 1e-12)
            rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(RESULTS / "step3b_sweeps.csv", index=False)

    # Re-derive the correction note's "Delta rho <= 0.0013 at V1 across six resolutions"
    # (learning_rules_rsa_paper_v2.tex:61-63): June sweep vs bnfix, all resolutions.
    ja = pd.read_csv(REPO / SWEEPS["june_sweep"]["csv"])
    jb = pd.read_csv(REPO / SWEEPS["bnfix"]["csv"])
    k = ["rule", "layer", "roi", "res"]
    dr = (jb.groupby(k).rho.mean() - ja.groupby(k).rho.mean()).dropna().reset_index()
    dr = dr[dr.rule.isin(RULES) & (dr.roi == "V1") & (dr.layer == "Conv1")]
    worst = dr.loc[dr.rho.abs().idxmax()]
    note_check = {"max_abs_june_vs_bnfix_V1_all_res": float(dr.rho.abs().max()),
                  "worst_cell": {"rule": worst.rule, "res": int(worst.res),
                                 "bnfix_minus_june": float(worst.rho)},
                  "resolutions": sorted(int(r) for r in dr.res.unique())}

    flagged = tab.loc[~tab.table_minus_bnfix_within_2se, ["rule", "roi"]].to_dict("records")
    v1 = tab[tab.roi == "V1"].set_index("rule")
    out = {"inventory": inv, "duplicates": dup, "flagged_table_vs_bnfix": flagged,
           "max_abs_table_minus_bnfix": float(tab.table_minus_bnfix.abs().max()),
           "max_abs_june_minus_bnfix_V1": float(v1.june_sweep_minus_bnfix.abs().max()),
           "max_abs_june_minus_bnfix_all": float(tab.june_sweep_minus_bnfix.abs().max()),
           "correction_note_delta_check": note_check}
    (RESULTS / "step3b_sweeps.json").write_text(json.dumps(out, indent=2))
    print(tab.round(4).to_string())
    print(json.dumps({k: out[k] for k in out if k != "inventory"}, indent=1))


if __name__ == "__main__":
    main()

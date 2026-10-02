"""Step 2b: core effects of arXiv:2608.12408 per variant, from stored bnfix model RDMs.

  E1  Random - Backprop, Conv1 -> V1, every resolution
  E2  Backprop - Random, Conv3 -> LOC, every resolution
  E3  luminance scalar vs V1 RDM (and vs the untrained network at 224 px)
Variants: all (all pairs) | cr (pair set P) | crb (P block-cleaned, per subject only).
Seed CI: paired per-seed difference, mean +- t_{0.975,4} SD/sqrt(5). Validation: `all`,
mean-RDM must reproduce bnfix_sweep.csv.

Output: results/crossrun/step2_effects.csv, step2_effects_per_seed.csv, step2_effects.json
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

from common_cr import RESULTS, REPO, SUBJECTS, N_STIM, TRI, load, run_labels, ranks, pair_set, \
    block_ids, demean_blocks

RES = [32, 64, 96, 128, 160, 224]
ROOT = REPO / "learning_rules_outputs_bnfix" / "rdms"
CASES = {"V1": ("Conv1", "random_weights", "backprop"), "LOC": ("Conv3", "backprop", "random_weights")}
T = tdist.ppf(0.975, 4)
LUM = RESULTS.parents[0] / "sub03_check" / "luminance.csv"


def main():
    runs = run_labels()
    rp, cp, _ = pair_set(None, runs)
    bid = {s: block_ids(rp, cp, runs[s]) for s in SUBJECTS}
    sel = {"all": (TRI[0], TRI[1]), "cr": (rp, cp)}

    brain = {}
    for roi in CASES:
        F = load("old", roi)
        for var, (r_, c_) in sel.items():
            vecs = [x[r_, c_] for x in F]
            brain[(var, roi)] = ([ranks(v) for v in vecs], ranks(np.mean(vecs, axis=0)))
        brain[("crb", roi)] = ([ranks(demean_blocks(x[rp, cp], bid[s])) for x, s in zip(F, SUBJECTS)], None)

    def score(X, roi):
        out = {}
        for var, (r_, c_) in sel.items():
            mv = ranks(X[r_, c_])
            sub, mean = brain[(var, roi)]
            out[(var, "persub")] = float(np.mean([mv @ s for s in sub]))
            out[(var, "meanrdm")] = float(mv @ mean)
        xc = X[rp, cp]
        out[("crb", "persub")] = float(np.mean(
            [ranks(demean_blocks(xc, bid[s])) @ r for s, r in zip(SUBJECTS, brain[("crb", roi)][0])]))
        return out

    rows = []
    for px in RES:
        for seed in range(5):
            d = ROOT / f"res{px}" / f"seed_{seed}"
            for roi, (layer, a, b) in CASES.items():
                for rule in (a, b):
                    for (var, conv), v in score(np.load(d / f"rdm_{rule}_{layer}.npy"), roi).items():
                        rows.append({"res": px, "seed": seed, "roi": roi, "rule": rule,
                                     "variant": var, "convention": conv, "rho": v})
    ps = pd.DataFrame(rows)
    ps.to_csv(RESULTS / "step2_effects_per_seed.csv", index=False)

    sw = pd.read_csv(REPO / "bnfix_sweep.csv")
    nm = {"random_weights": "Random Weights", "backprop": "Backprop"}
    v = ps[(ps.variant == "all") & (ps.convention == "meanrdm")].copy()
    v["rule_csv"], v["layer"] = v.rule.map(nm), v.roi.map({k: c[0] for k, c in CASES.items()})
    j = v.merge(sw, left_on=["rule_csv", "roi", "layer", "res", "seed"],
                right_on=["rule", "roi", "layer", "res", "seed_idx"], suffixes=("", "_sw"))
    max_dev = float((j.rho - j.rho_sw).abs().max())

    eff = []
    for (px, roi, var, conv), g in ps.groupby(["res", "roi", "variant", "convention"]):
        layer, a, b = CASES[roi]
        d = g[g.rule == a].set_index("seed").rho - g[g.rule == b].set_index("seed").rho
        se = d.std(ddof=1) / np.sqrt(len(d))
        eff.append({"effect": "E1 Random-BP V1" if roi == "V1" else "E2 BP-Random LOC",
                    "res": px, "variant": var, "convention": conv, "mean": float(d.mean()),
                    "ci_lo": float(d.mean() - T * se), "ci_hi": float(d.mean() + T * se),
                    "n_seeds_positive": int((d > 0).sum())})
    eff = pd.DataFrame(eff)
    eff.to_csv(RESULTS / "step2_effects.csv", index=False)

    lum = pd.read_csv(LUM).luminance.values
    L = np.abs(lum[:, None] - lum[None, :])
    lumd = {f"{k[0]}|{k[1]}": val for k, val in score(L, "V1").items()}
    rnd = ps[(ps.res == 224) & (ps.roi == "V1") & (ps.rule == "random_weights")] \
        .groupby(["variant", "convention"]).rho.mean()
    out = {"validation_max_abs_diff_vs_bnfix_sweep": max_dev, "n_validated": int(len(j)),
           "luminance_vs_V1": lumd,
           "random_224_V1": {f"{a}|{b}": float(x) for (a, b), x in rnd.items()}}
    (RESULTS / "step2_effects.json").write_text(json.dumps(out, indent=2))
    pd.set_option("display.width", 200)
    print(eff[eff.res.isin([32, 224])].round(4).to_string()); print(json.dumps(out, indent=1))
    assert max_dev < 1e-4, max_dev


if __name__ == "__main__":
    main()

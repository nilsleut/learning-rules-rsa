"""Multiple-comparison correction of the primary comparisons: family of 7 (v4) vs 11 (before
the V2 layer-assignment check removed the four FA-at-V2 comparisons).

p-values from the shared stimulus bootstrap (results/crossrun/step2_draws.csv, per subject,
cross-run, repaired set), two-sided:  p = (2 * min(#{d_b <= 0}, #{d_b >= 0}) + 1) / (B + 1),
capped at 1. Corrections: Benjamini-Hochberg (FDR, q = 0.05) and Holm (FWER, alpha = 0.05),
each applied within the family of 7 and within the family of 11.

Output: results/paper_v3/primary_fdr.csv, primary_fdr.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
P3 = REPO / "results" / "paper_v3"
RULES = {"rnd": "random_weights", "bp": "backprop", "fa": "feedback_alignment",
         "pc": "predictive_coding", "stdp": "stdp"}
FAM7 = [("rnd", "bp", "V1"), ("rnd", "bp", "V2"), ("rnd", "bp", "LOC")] + \
       [(x, y, "V1") for (x, y) in (("rnd", "fa"), ("bp", "fa"), ("fa", "pc"), ("fa", "stdp"))]
FAM11 = FAM7 + [(x, y, "V2") for (x, y) in (("rnd", "fa"), ("bp", "fa"), ("fa", "pc"), ("fa", "stdp"))]
ALPHA = 0.05


def bh(p, q=ALPHA):
    p = np.asarray(p); m = len(p); o = np.argsort(p)
    thr = q * np.arange(1, m + 1) / m
    ok = p[o] <= thr
    k = np.max(np.nonzero(ok)[0]) + 1 if ok.any() else 0
    rej = np.zeros(m, bool); rej[o[:k]] = True
    adj = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m); out[o] = np.minimum(adj, 1)
    return rej, out


def holm(p, a=ALPHA):
    p = np.asarray(p); m = len(p); o = np.argsort(p)
    adj = np.maximum.accumulate(p[o] * (m - np.arange(m)))
    out = np.empty(m); out[o] = np.minimum(adj, 1)
    return out <= a, out


def main():
    D = pd.read_csv(REPO / "results" / "crossrun" / "step2_draws.csv")
    pt, bt = D[D.boot == -1].iloc[0], D[D.boot >= 0]
    B = len(bt)
    rows = {}
    for a, b, roi in FAM11:
        ka, kb = f"persub|cr|bnfix|{RULES[a]}|{roi}", f"persub|cr|bnfix|{RULES[b]}|{roi}"
        d = (bt[ka] - bt[kb]).to_numpy()
        p = min(1.0, (2 * min((d <= 0).sum(), (d >= 0).sum()) + 1) / (B + 1))
        rows[(a, b, roi)] = dict(comparison=f"{a}-{b}", roi=roi, diff=pt[ka] - pt[kb],
                                 ci_lo=np.quantile(d, .025), ci_hi=np.quantile(d, .975), p_boot=p,
                                 uncorrected_sig=bool(np.quantile(d, .025) > 0 or np.quantile(d, .975) < 0))
    for name, fam in (("7", FAM7), ("11", FAM11)):
        ps = [rows[k]["p_boot"] for k in fam]
        rb, pb = bh(ps); rh, ph = holm(ps)
        for k, r1, p1, r2, p2 in zip(fam, rb, pb, rh, ph):
            rows[k].update({f"bh_p_fam{name}": p1, f"bh_sig_fam{name}": bool(r1),
                            f"holm_p_fam{name}": p2, f"holm_sig_fam{name}": bool(r2)})
    df = pd.DataFrame([rows[k] for k in FAM11])
    df["in_family7"] = [k in FAM7 for k in FAM11]
    df.to_csv(P3 / "primary_fdr.csv", index=False)
    f7 = df[df.in_family7]
    changed = bool((f7.bh_sig_fam7 != f7.bh_sig_fam11).any() or (f7.holm_sig_fam7 != f7.holm_sig_fam11).any()
                   or (f7.bh_sig_fam7 != f7.uncorrected_sig).any())
    out = {"n_boot": B, "alpha": ALPHA, "any_decision_differs_7_vs_11_or_vs_uncorrected": changed,
           "min_p_attainable": 1 / (B + 1)}
    (P3 / "primary_fdr.json").write_text(json.dumps(out, indent=2))
    pd.set_option("display.width", 220)
    print(df.round(5).to_string(index=False)); print(out)


if __name__ == "__main__":
    main()

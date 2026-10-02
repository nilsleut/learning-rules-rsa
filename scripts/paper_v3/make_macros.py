"""Every number in paper v3, generated from saved result files.

Writes into paper/arxiv_upload_learning_rules_v3/:
  numbers.tex        \\NV{key} macros (single rounding, ROUND_HALF_UP, from full precision)
  tab_main.tex       Table 2 (bnfix, cross-run; per-subject main column, mean-RDM extra column)
  tab_pairwise.tex   all pairwise differences (bnfix, cross-run, per subject)
  tab_sens.tex       sensitivity: bounds + model rho, all pairs | cross-run | block-cleaned
  tab_diag.tex       diagnostics 1b, 1c, STOP criterion 4
  tab_errata.tex     rounding / inconsistency errata of v2
  tab_partial.tex    partial RSA (original analysis, unaffected conditions only)
and results/paper_v3/numbers_manifest.csv (key, value, decimals, text, source, selector).

The script asserts every qualitative statement the text makes (sign / CI significance);
if one fails it stops, and the text has to change.
"""
import json
import math
from decimal import Decimal, ROUND_HALF_UP
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PAPER = REPO / "paper" / "arxiv_upload_learning_rules_v3"
RES = REPO / "results"
CR = RES / "crossrun"
P3 = RES / "paper_v3"

RULES = {"rnd": "random_weights", "bp": "backprop", "fa": "feedback_alignment",
         "pc": "predictive_coding", "stdp": "stdp"}
RNAME = {"rnd": "Random", "bp": "BP", "fa": "FA", "pc": "PC", "stdp": "STDP"}
CSVNAME = {"rnd": "Random Weights", "bp": "Backprop", "fa": "Feedback Alignment",
           "pc": "Predictive Coding", "stdp": "STDP"}
ROIS = ["V1", "V2", "LOC", "IT"]
LAYER = {"V1": "Conv1", "V2": "Conv1", "LOC": "Conv3", "IT": "FC1"}
SETS = {"bn": "bnfix", "ref": "original"}
CONV = {"ps": "persub", "mr": "meanrdm"}
VARS = ["all", "cr", "crb"]
SUBJ = ["sub-01", "sub-02", "sub-03"]
# primary comparisons (the three statements that hold): Random vs BP at V1/V2, BP vs Random at
# LOC, FA vs every other condition at V1. Keys are in table order (a before b in RULES).
# FA at V2 is not primary: its ordering against BP and PC depends on the layer assigned to V2
# (Conv1 vs Conv2; results/paper_v3/v2_conv2.json, appendix table tab_v2conv2.tex).
PRIMARY = {("rnd", "bp", "V1"), ("rnd", "bp", "V2"), ("rnd", "bp", "LOC")} | {
    (x, y, roi) for roi in ("V1",) for (x, y) in
    (("rnd", "fa"), ("bp", "fa"), ("fa", "pc"), ("fa", "stdp"))}

MAN = []          # manifest rows
MAC = {}          # key -> text
RAW = {}          # key -> full-precision value


def rnd(x, d):
    q = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    return format(q, "f")


def txt(x, d, lead=True, sign=False):
    s = rnd(x, d)
    if sign and not s.startswith("-"):
        s = "+" + s
    if not lead:
        s = s.replace("0.", ".", 1) if s.lstrip("+-").startswith("0.") else s
    return s


def put(key, x, d, src, sel, lead=True, sign=False, pct=False):
    """Register a number; returns its text."""
    t = txt(100 * x if pct else x, d, lead, sign)
    if key in MAC:
        assert MAC[key] == t, (key, MAC[key], t)
    MAC[key] = t
    RAW[key] = float(x)
    MAN.append({"key": key, "value": float(x), "decimals": d, "text": t, "source": src,
                "selector": sel, "pct": pct})
    return t


def puts(key, s, src, sel):
    """Register a non-rounded literal (integer count, label)."""
    MAC[key] = s
    MAN.append({"key": key, "value": np.nan, "decimals": -1, "text": s, "source": src,
                "selector": sel, "pct": False})
    return s


def sig(lo, hi):
    return lo > 0 or hi < 0


def rel(p):
    return str(p.relative_to(REPO)).replace("\\", "/")


def main():
    PAPER.mkdir(parents=True, exist_ok=True)
    summ = pd.read_csv(CR / "step2_summary.csv").set_index("key")
    draws = pd.read_csv(CR / "step2_draws.csv")
    pt, bt = draws[draws.boot == -1].iloc[0], draws[draws.boot >= 0]
    j2 = json.loads((CR / "step2.json").read_text())
    j1 = json.loads((CR / "step1.json").read_text())
    S2, D2, J2, J1 = (rel(CR / "step2_summary.csv"), rel(CR / "step2_draws.csv"),
                      rel(CR / "step2.json"), rel(CR / "step1.json"))

    def ci(key):
        r = summ.loc[key]
        return r.point, r.ci_lo, r.ci_hi

    def dci(ka, kb):
        x = bt[ka] - bt[kb]
        return pt[ka] - pt[kb], x.quantile(.025), x.quantile(.975)

    # ── model rho, bounds, differences ────────────────────────────────────────
    for sk, st in SETS.items():
        for ck, conv in CONV.items():
            for var in VARS:
                if var == "crb" and ck == "mr":
                    continue
                for rk, rule in RULES.items():
                    for roi in ROIS:
                        k = f"{conv}|{var}|{st}|{rule}|{roi}"
                        p, lo, hi = ci(k)
                        b = f"rho.{ck}.{var}.{sk}.{rk}.{roi}"
                        put(b, p, 3, S2, k)
                        put(b + ".lo", lo, 3, S2, k + " ci_lo", lead=False)
                        put(b + ".hi", hi, 3, S2, k + " ci_hi", lead=False)
                        put(b + ".4", p, 4, S2, k)
                        if ck == "ps":
                            kd = f"diff|{var}|{st}|{rule}|{roi}"
                            p, lo, hi = ci(kd)
                            put(f"ml.{var}.{sk}.{rk}.{roi}", p, 3, S2, kd, sign=True)
                            put(f"ml.{var}.{sk}.{rk}.{roi}.lo", lo, 3, S2, kd + " ci_lo")
                            put(f"ml.{var}.{sk}.{rk}.{roi}.hi", hi, 3, S2, kd + " ci_hi")
                for roi in ROIS:
                    for a, b in combinations(RULES, 2):
                        ka = f"{conv}|{var}|{st}|{RULES[a]}|{roi}"
                        kb = f"{conv}|{var}|{st}|{RULES[b]}|{roi}"
                        d, lo, hi = dci(ka, kb)
                        base = f"d.{ck}.{var}.{sk}.{a}.{b}.{roi}"
                        sel = f"{ka} - {kb}"
                        rbase, rsel = f"d.{ck}.{var}.{sk}.{b}.{a}.{roi}", f"{kb} - {ka}"
                        for dd, suf in ((3, ""), (4, ".4")):
                            put(base + suf, d, dd, D2, sel, sign=True)
                            put(base + ".lo" + suf, lo, dd, D2, sel + " q.025")
                            put(base + ".hi" + suf, hi, dd, D2, sel + " q.975")
                            put(rbase + suf, -d, dd, D2, rsel, sign=True)
                            put(rbase + ".lo" + suf, -hi, dd, D2, rsel + " q.025")
                            put(rbase + ".hi" + suf, -lo, dd, D2, rsel + " q.975")
    for var in VARS:
        for roi in ROIS:
            for q in ("lower", "upper"):
                k = f"{q}|{var}|{roi}"
                p, lo, hi = ci(k)
                b = f"{'lb' if q == 'lower' else 'ub'}.{var}.{roi}"
                put(b, p, 3, S2, k)
                put(b + ".lo", lo, 3, S2, k + " ci_lo")
                put(b + ".hi", hi, 3, S2, k + " ci_hi")
            if var != "all":
                for q, short in (("lower", "lb"), ("upper", "ub")):
                    nn = j2["perm_null"][f"{q}|{var}|{roi}"]
                    put(f"{short}null.{var}.{roi}.mean", nn["mean"], 3, J2, f"perm_null.{q}|{var}|{roi}.mean")
                    put(f"{short}null.{var}.{roi}.sd", nn["sd"], 3, J2, f"perm_null.{q}|{var}|{roi}.sd")
                    put(f"{short}null.{var}.{roi}.q", nn["q975"], 3, J2, f"perm_null.{q}|{var}|{roi}.q975")

    # ranges over rules
    for roi in ROIS:
        for ck in CONV:
            vals = {rk: summ.loc[f"{CONV[ck]}|cr|bnfix|{r}|{roi}"].point for rk, r in RULES.items()}
            put(f"range.{ck}.{roi}.min", min(vals.values()), 3, S2, f"min over rules {CONV[ck]}|cr|bnfix|*|{roi}")
            put(f"range.{ck}.{roi}.max", max(vals.values()), 3, S2, f"max over rules {CONV[ck]}|cr|bnfix|*|{roi}")
    for roi in ROIS:
        c = summ.loc[f"lower|cr|{roi}"].point / summ.loc[f"lower|all|{roi}"].point - 1
        put(f"lbchg.{roi}", -c, 0, S2, f"1 - lower|cr|{roi} / lower|all|{roi}", pct=True)

    # ── assertions behind the text (bnfix, cross-run) ─────────────────────────
    def d_(ck, a, b, roi, var="cr", st="bnfix"):
        return dci(f"{CONV[ck]}|{var}|{st}|{RULES[a]}|{roi}", f"{CONV[ck]}|{var}|{st}|{RULES[b]}|{roi}")
    for ck in CONV:
        for roi in ("V1", "V2"):
            assert d_(ck, "rnd", "bp", roi)[1] > 0                         # Random > BP
            for o in ("rnd", "bp", "pc", "stdp"):
                assert d_(ck, o, "fa", roi)[1] > 0                         # FA lowest
        assert d_(ck, "bp", "rnd", "LOC")[1] > 0                           # BP > Random at LOC
        for o in ("pc", "stdp"):
            assert d_(ck, o, "rnd", "LOC")[1] > 0
        for a, b in combinations(["bp", "fa", "pc", "stdp"], 2):
            assert not sig(*d_(ck, a, b, "LOC")[1:])                       # trained n.s. at LOC
        for a, b in combinations(RULES, 2):
            assert not sig(*d_(ck, a, b, "IT")[1:])                        # IT: nothing
    assert not sig(*d_("ps", "fa", "rnd", "LOC")[1:])                      # FA borderline:
    assert d_("mr", "fa", "rnd", "LOC")[1] > 0                             #  ps n.s., mr sig
    for a, b in combinations(RULES, 2):                                    # V1: all sig (ps)
        assert sig(*d_("ps", a, b, "V1")[1:])
    v1_mr_ns = [(a, b) for a, b in combinations(RULES, 2) if not sig(*d_("mr", a, b, "V1")[1:])]
    v2_ns = {ck: [(a, b) for a, b in combinations(RULES, 2) if not sig(*d_(ck, a, b, "V2")[1:])] for ck in CONV}
    assert v1_mr_ns == [("bp", "stdp")] and v2_ns == {"ps": [("bp", "stdp")], "mr": [("bp", "stdp")]}
    # model - lower bound (ps, cr, bnfix)
    for roi in ("V1", "V2"):
        assert not sig(*ci(f"diff|cr|bnfix|random_weights|{roi}")[1:])    # Random reaches lb
    for rk in ("bp", "fa", "pc", "stdp"):
        assert ci(f"diff|cr|bnfix|{RULES[rk]}|V1")[2] < 0                  # trained below at V1
    for rk in RULES:
        for roi in ("LOC", "IT"):
            assert ci(f"diff|cr|bnfix|{RULES[rk]}|{roi}")[2] < 0           # all below at LOC/IT
    assert not sig(*ci("diff|cr|bnfix|stdp|V2")[1:])                       # STDP at V2: n.s.
    for rk in ("bp", "fa", "pc"):
        assert ci(f"diff|cr|bnfix|{RULES[rk]}|V2")[2] < 0
    put("ml.cr.bn.bp.V2.hi.4", ci("diff|cr|bnfix|backprop|V2")[2], 4, S2, "diff|cr|bnfix|backprop|V2 ci_hi")
    assert -1e-4 < ci("diff|cr|bnfix|backprop|V2")[2] < 0        # text: "ends below zero by less than 10^-4"
    # V1: PC below BP (both conventions); STDP above BP only per subject
    for ck in CONV:
        assert d_(ck, "bp", "pc", "V1")[1] > 0
    assert d_("ps", "stdp", "bp", "V1")[1] > 0 and not sig(*d_("mr", "stdp", "bp", "V1")[1:])
    # reliability order of the lower bound: LOC < IT < V2 < V1
    lbp = {roi: summ.loc[f"lower|cr|{roi}"].point for roi in ROIS}
    assert lbp["LOC"] < lbp["IT"] < lbp["V2"] < lbp["V1"]
    for roi in ROIS:                                                        # far from chance
        assert summ.loc[f"lower|cr|{roi}"].ci_lo > j2["perm_null"][f"lower|cr|{roi}"]["q975"]
    # hierarchy statements
    pts = {roi: {rk: summ.loc[f"persub|cr|bnfix|{r}|{roi}"].point for rk, r in RULES.items()} for roi in ROIS}
    for roi in ("V1", "V2"):
        assert max(pts[roi], key=pts[roi].get) == "rnd"
    assert min(pts["LOC"], key=pts["LOC"].get) == "rnd" and pts["LOC"]["rnd"] < 0
    assert min(pts["IT"], key=pts["IT"].get) == "rnd"
    for rk in RULES:
        assert pts["LOC"][rk] < pts["V1"][rk]
    # appendix: statements hold in every pair set (repaired set), except one cell
    exc = []
    for var in VARS:
        for ck in (["ps"] if var == "crb" else ["ps", "mr"]):
            for roi in ("V1", "V2"):
                assert d_(ck, "rnd", "bp", roi, var)[1] > 0
                for o in ("rnd", "bp", "pc", "stdp"):
                    assert d_(ck, o, "fa", roi, var)[1] > 0
            assert d_(ck, "bp", "rnd", "LOC", var)[1] > 0
            for a, b in combinations(["bp", "fa", "pc", "stdp"], 2):
                if sig(*d_(ck, a, b, "LOC", var)[1:]):
                    exc.append((var, ck, a, b))
            for a, b in combinations(RULES, 2):
                assert not sig(*d_(ck, a, b, "IT", var)[1:])
    assert exc == [("crb", "ps", "bp", "fa")], exc
    md = max(abs(summ.loc[f"persub|{v}|bnfix|{r}|{roi}"].point - summ.loc[f"persub|cr|bnfix|{r}|{roi}"].point)
             for v in ("all", "crb") for r in RULES.values() for roi in ROIS)
    put("sens.maxdiff", md, 3, S2, "max |rho_subj(variant) - rho_subj(cr)|, variants all/crb, bnfix")
    AG = CR / "step1_agreement.csv"
    ag = pd.read_csv(AG, dtype={"pair": str}).set_index(["roi", "pair"])
    for roi in ROIS:
        for pr_ in ("01-02", "01-03", "02-03"):
            for c, s_ in (("all_pairs", "all"), ("cross_run", "cr")):
                put(f"agree.{pr_.replace('-', '')}.{roi}.{s_}", ag.loc[(roi, pr_), c], 3, rel(AG), f"{roi} {pr_} {c}")
    # upper bound: within its permutation null range (not informative)
    for roi in ROIS:
        nn = j2["perm_null"][f"upper|cr|{roi}"]
        put(f"ubexcess.cr.{roi}", summ.loc[f"upper|cr|{roi}"].point - nn["mean"], 3, S2 + " ; " + J2,
            f"upper|cr|{roi} point - perm_null mean")
    # borderline bnfix cell, block-cleaned
    k1, k2 = "persub|crb|bnfix|backprop|LOC", "persub|crb|bnfix|feedback_alignment|LOC"
    d, lo, hi = dci(k1, k2)
    assert lo > 0 and lo < 0.0005
    put("border.d", d, 4, D2, f"{k1} - {k2}", sign=True)
    put("border.lo", lo, 4, D2, f"{k1} - {k2} q.025")
    put("border.hi", hi, 4, D2, f"{k1} - {k2} q.975")
    d, lo, hi = dci(k1.replace("crb", "cr"), k2.replace("crb", "cr"))
    put("border.cr.d", d, 4, D2, "cr analogue", sign=True)
    put("border.cr.lo", lo, 4, D2, "cr analogue q.025")
    put("border.cr.hi", hi, 4, D2, "cr analogue q.975")

    # ── pairs, diagnostics ────────────────────────────────────────────────────
    puts("pairs.total", f"{j1['n_pairs_total']:,}".replace(",", "{,}"), J1, "n_pairs_total")
    puts("pairs.cr", f"{j1['n_pairs_P']:,}".replace(",", "{,}"), J1, "n_pairs_P")
    put("pairs.cr.pct", j1["frac_pairs_P"], 1, J1, "frac_pairs_P", pct=True)
    put("pairs.excl.pct", 1 - j1["frac_pairs_P"], 1, J1, "1 - frac_pairs_P", pct=True)
    puts("nboot", str(j2["n_boot"]), J2, "n_boot")
    puts("nperm", str(j2["n_perm"]), J2, "n_perm")
    puts("bootseed", str(j2["boot_seed"]), J2, "boot_seed")
    puts("permseed", str(j2["perm_seed"]), J2, "perm_seed")
    LAG = CR / "step1_lag_summary.csv"
    lag = pd.read_csv(LAG)
    put("lag.rho.min", lag.spearman_diss_vs_lag.min(), 2, rel(LAG), "min spearman_diss_vs_lag")
    put("lag.rho.max", lag.spearman_diss_vs_lag.max(), 2, rel(LAG), "max spearman_diss_vs_lag")
    hi_ = lag[lag.roi.isin(["LOC", "IT"])]
    assert (hi_.mean_lag1 > 1).sum() >= 5 and (hi_.mean_cross_run_P < 1).all()
    put("lag.one.min", hi_.mean_lag1.min(), 3, rel(LAG), "min mean_lag1 over LOC/IT")
    put("lag.one.max", hi_.mean_lag1.max(), 3, rel(LAG), "max mean_lag1 over LOC/IT")
    puts("lag.one.n", str(int((hi_.mean_lag1 > 1).sum())), rel(LAG), "count mean_lag1 > 1, LOC/IT")
    RPT = CR / "step1_runpair_test.csv"
    rpt = pd.read_csv(RPT).set_index("roi")
    for roi in ROIS:
        put(f"rp.{roi}.rho", rpt.loc[roi, "spearman_01_02_runpair_means"], 3, rel(RPT), f"{roi} spearman")
        put(f"rp.{roi}.p", rpt.loc[roi, "p_exact_one_sided"], 3, rel(RPT), f"{roi} p_exact_one_sided")
    puts("rp.nperm", f"{int(rpt.n_permutations.iloc[0]):,}".replace(",", "{,}"), rel(RPT), "n_permutations")
    LUMF = CR / "step1_luminance.csv"
    lumf = pd.read_csv(LUMF).set_index(["subject", "roi"])
    for s in SUBJ:
        for roi in ("V1", "V2"):
            r = lumf.loc[(s, roi)]
            sk = s.replace("sub-0", "s")
            put(f"lum.{sk}.{roi}.all", r.all_pairs, 3, rel(LUMF), f"{s} {roi} all_pairs")
            put(f"lum.{sk}.{roi}.cr", r.cross_run, 3, rel(LUMF), f"{s} {roi} cross_run")
            put(f"lum.{sk}.{roi}.crb", r.cross_run_block_cleaned, 3, rel(LUMF), f"{s} {roi} crb")
            put(f"lum.{sk}.{roi}.chg", r.frac_change_cr_to_crb, 0, rel(LUMF), f"{s} {roi} frac_change", pct=True, sign=True)
    assert not j1["stop_luminance_drop_gt_30pct"]
    EFF = CR / "step2_effects.json"
    eff = json.loads(EFF.read_text())
    for ck in CONV:
        put(f"lumv1.{ck}.cr", eff["luminance_vs_V1"][f"cr|{CONV[ck]}"], 3, rel(EFF), f"luminance_vs_V1 cr|{CONV[ck]}")
        put(f"lumv1.{ck}.all", eff["luminance_vs_V1"][f"all|{CONV[ck]}"], 3, rel(EFF), f"luminance_vs_V1 all|{CONV[ck]}")

    # ── per subject (B1), seeds (B2) ──────────────────────────────────────────
    SS, SP = P3 / "subject_summary.csv", P3 / "subject_pairwise.csv"
    ss = pd.read_csv(SS).set_index(["subject", "rule", "roi"])
    sp = pd.read_csv(SP).set_index(["subject", "roi", "rule_a", "rule_b"])
    sub_loc = {}
    for s in SUBJ:
        sk = s.replace("sub-0", "s")
        for rk, r in RULES.items():
            for roi in ROIS:
                x = ss.loc[(s, r, roi)]
                put(f"sub.{sk}.{rk}.{roi}", x.point, 3, rel(SS), f"{s} {r} {roi} point")
                put(f"sub.{sk}.{rk}.{roi}.lo", x.ci_lo, 3, rel(SS), f"{s} {r} {roi} ci_lo")
                put(f"sub.{sk}.{rk}.{roi}.hi", x.ci_hi, 3, rel(SS), f"{s} {r} {roi} ci_hi")
        for roi in ROIS:
            for a, b in combinations(RULES, 2):
                x = sp.loc[(s, roi, RULES[a], RULES[b])]
                base = f"subd.{sk}.{a}.{b}.{roi}"
                put(base, x["diff"], 3, rel(SP), f"{s} {roi} {a}-{b} diff", sign=True)
                put(base + ".lo", x.ci_lo, 3, rel(SP), f"{s} {roi} {a}-{b} ci_lo")
                put(base + ".hi", x.ci_hi, 3, rel(SP), f"{s} {roi} {a}-{b} ci_hi")
        # LOC: is BP the top point estimate, and does any CI support "BP leading"?
        pts = {rk: ss.loc[(s, RULES[rk], "LOC")].point for rk in RULES}
        top = max(pts, key=pts.get)
        def pair(a, b, roi="LOC"):
            ia = list(RULES).index(a) < list(RULES).index(b)
            x = sp.loc[(s, roi, RULES[a], RULES[b])] if ia else sp.loc[(s, roi, RULES[b], RULES[a])]
            return (x["diff"], x.ci_lo, x.ci_hi) if ia else (-x["diff"], -x.ci_hi, -x.ci_lo)
        sub_loc[s] = {"top": top, "points": {k: round(v, 4) for k, v in pts.items()},
                      "bp_minus": {o: [round(float(v), 4) for v in pair("bp", o)] for o in ("rnd", "fa", "pc", "stdp")},
                      "bp_vs_sig": {o: sig(*pair("bp", o)[1:]) for o in ("rnd", "fa", "pc", "stdp")}}
    SEED = P3 / "seed_spread.csv"
    sd = pd.read_csv(SEED).set_index(["rule", "roi"])
    for rk, r in RULES.items():
        for roi in ROIS:
            x = sd.loc[(r, roi)]
            for c in ("persub_sd", "persub_min", "persub_max"):
                put(f"seed.{rk}.{roi}.{c.split('_')[1]}", x[c], 4, rel(SEED), f"{r} {roi} {c}")
    v12 = sd.loc[[(r, roi) for r in RULES.values() for roi in ("V1", "V2")]]
    put("seedsd.v12.min", v12.persub_sd.min(), 3, rel(SEED), "min persub_sd V1/V2")
    put("seedsd.v12.max", v12.persub_sd.max(), 3, rel(SEED), "max persub_sd V1/V2")
    it = sd.xs("IT", level="roi")
    put("seedsd.it.min", it.persub_sd.min(), 4, rel(SEED), "min persub_sd IT")
    put("seedsd.it.max", it.persub_sd.max(), 4, rel(SEED), "max persub_sd IT")
    overl = {}
    for a, b in combinations(RULES, 2):
        A, B = it.loc[RULES[a]], it.loc[RULES[b]]
        overl[(a, b)] = bool(A.persub_min <= B.persub_max and B.persub_min <= A.persub_max)
    # B2 statements at IT: BP vs FA and BP vs Random ranges do not overlap, all others do
    assert [p for p, v in overl.items() if not v] == [("rnd", "bp"), ("bp", "fa")]
    for k_, (r, c) in {"seed5.fa.IT.max": ("fa", "persub_max"), "seed5.bp.IT.min": ("bp", "persub_min"),
                       "seed5.rnd.IT.max": ("rnd", "persub_max")}.items():
        put(k_, it.loc[RULES[r], c], 5, rel(SEED), f"{RULES[r]} IT {c}")
    # B1 statements
    ord_group = sorted(RULES, key=lambda r: -summ.loc[f"persub|cr|bnfix|{RULES[r]}|V1"].point)
    for s in ("sub-01", "sub-02"):
        o = sorted(RULES, key=lambda r: -ss.loc[(s, RULES[r], "V1")].point)
        assert o == ord_group == ["rnd", "stdp", "bp", "pc", "fa"], (s, o)
        for b in ("bp", "fa", "pc", "stdp"):
            assert sp.loc[(s, "V1", "random_weights", RULES[b])].ci_lo > 0
    for b in ("rnd", "bp", "stdp"):
        a_, b_ = (b, "fa") if list(RULES).index(b) < list(RULES).index("fa") else ("fa", b)
        x = sp.loc[("sub-01", "V1", RULES[a_], RULES[b_])]
        assert (x.ci_lo > 0) if a_ == b else (x.ci_hi < 0)
    assert sp.loc[("sub-01", "V1", "feedback_alignment", "predictive_coding")].ci_hi < 0
    x = sp.loc[("sub-02", "V1", "feedback_alignment", "predictive_coding")]
    assert -0.0005 < x.ci_hi < 0
    assert sub_loc["sub-01"]["top"] == "stdp" and sub_loc["sub-02"]["top"] == "bp" and sub_loc["sub-03"]["top"] == "bp"
    assert not any(sub_loc[s]["bp_vs_sig"][o] for s in SUBJ for o in ("fa", "pc", "stdp"))
    assert [s for s in SUBJ if sub_loc[s]["bp_vs_sig"]["rnd"]] == ["sub-02"]
    for s in SUBJ:                                    # no trained-trained difference at LOC
        for a, b in combinations(["bp", "fa", "pc", "stdp"], 2):
            x = sp.loc[(s, "LOC", RULES[a], RULES[b])]
            assert not sig(x.ci_lo, x.ci_hi)
    trained_pairs = [p for p in overl if "rnd" not in p]
    seed_it = {"trained_all_overlap": all(overl[p] for p in trained_pairs),
               "rnd_overlaps": {b: overl[("rnd", b)] for b in ("bp", "fa", "pc", "stdp")}}

    # ── best layer per ROI (bnfix, cross-run, per subject; point estimates) ──
    HI = P3 / "hierarchy.csv"
    hh = pd.read_csv(HI)
    best = {(r, roi): g.loc[g.persub.idxmax()] for (r, roi), g in hh.groupby(["rule", "roi"])}
    bp_match = [roi for roi in ROIS if best[("backprop", roi)].layer == LAYER[roi]]
    assert bp_match == ["V1", "V2", "IT"] and best[("backprop", "LOC")].layer == "FC1"
    fc1_v1 = [rk for rk, r in RULES.items() if best[(r, "V1")].layer == "FC1"]
    assert fc1_v1 == ["rnd", "fa", "pc"]
    puts("bl.bp.nmatch", str(len(bp_match)), rel(HI), "ROIs where BP's best layer = fixed layer")
    L = ["\\begin{tabular}{l" + "c" * 4 + "}", "\\toprule", "Condition & " + " & ".join(ROIS) + " \\\\", "\\midrule"]
    for rk, r in RULES.items():
        cells = []
        for roi in ROIS:
            b = best[(r, roi)]
            fx = hh[(hh.rule == r) & (hh.roi == roi) & (hh.layer == LAYER[roi])].persub.iloc[0]
            tb = put(f"bl.{rk}.{roi}", b.persub, 3, rel(HI), f"{r} {roi} max over layers")
            tf = put(f"blfix.{rk}.{roi}", fx, 3, rel(HI), f"{r} {roi} fixed layer {LAYER[roi]}")
            cells.append(f"{b.layer} ${tb}$" + ("" if b.layer == LAYER[roi] else f" (${tf}$)"))
        L.append(f"{RNAME[rk]} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (PAPER / "tab_bestlayer.tex").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ── calibration control (all pairs, mean-RDM; original analysis) ─────────
    CAL, SW = REPO / "bncal_rows.csv", REPO / "bnfix_sweep.csv"
    cal, sw = pd.read_csv(CAL), pd.read_csv(SW)
    bp = sw[(sw.rule == "Backprop") & (sw.layer == "Conv1") & (sw.roi == "V1") & (sw.res == 224)].set_index("seed_idx").rho
    for v, k in (("B CIFAR@eval-res", "B"), ("D THINGS-disjoint@res", "D")):
        x = cal[(cal.variant == v) & (cal.res == 224) & (cal.layer == "Conv1") & (cal.roi == "V1")].set_index("seed_idx").rho
        g = x - bp
        put(f"cal.{k}", g.mean(), 3, f"{rel(CAL)} - {rel(SW)}", f"{v} minus Backprop, Conv1 V1 224px, seed mean", sign=True)
        puts(f"cal.{k}.n", f"{int((g > 0).sum())}/{len(g)}", rel(CAL), f"{v} seeds with gap > 0")
    put("init.ratio", math.sqrt(2) / math.sqrt(1 / 3), 1, "derived", "kaiming_normal(relu) std / kaiming_uniform(a=sqrt5) std = sqrt(6)")
    rep = sw[(sw.layer == "Conv1") & (sw.roi == "V1")]
    old = pd.read_csv(REPO / "old_sweep.csv")
    m = rep.merge(old, on=["rule", "layer", "roi", "res", "seed_idx"], suffixes=("", "_old"))
    m = m[m.rule.isin(["Random Weights", "Backprop", "Feedback Alignment"])]
    g = m.groupby(["rule", "res"])[["rho", "rho_old"]].mean()
    put("repair.max", (g.rho - g.rho_old).abs().max(), 4, f"{rel(SW)} vs old_sweep.csv",
        "max |seed-mean rho change|, Random/BP/FA, Conv1 V1, all res")

    # ── original analysis: partial RSA, Gabor, v2 table cells (errata) ───────
    PR, GB, RC = REPO / "outputs/partial_rsa_results.csv", REPO / "outputs/gabor_analysis.csv", REPO / "outputs/rsa_results_cnn.csv"
    pr = pd.read_csv(PR)
    pr = pr[(pr.roi == "V1")].set_index("rule")
    for rk in ("rnd", "bp", "fa"):
        x = pr.loc[CSVNAME[rk]]
        put(f"part.{rk}.std", x.rho_std, 3, rel(PR), f"V1 {CSVNAME[rk]} rho_std")
        put(f"part.{rk}.par", x.rho_partial, 3, rel(PR), f"V1 {CSVNAME[rk]} rho_partial")
        put(f"part.{rk}.d", x.delta, 3, rel(PR), f"V1 {CSVNAME[rk]} delta")
    prall = pd.read_csv(PR).set_index(["roi", "rule"])
    put("part.fa.V2.par", prall.loc[("V2", "Feedback Alignment"), "rho_partial"], 3, rel(PR), "V2 FA rho_partial")
    put("part.fa.V2.p", prall.loc[("V2", "Feedback Alignment"), "p_partial"], 3, rel(PR), "V2 FA p_partial")
    sub3 = pr.loc[[CSVNAME[k] for k in ("rnd", "bp", "fa")]]
    put("part.d.min", sub3.delta.min(), 3, rel(PR), "min delta rnd/bp/fa")
    put("part.d.max", sub3.delta.max(), 3, rel(PR), "max delta rnd/bp/fa")
    assert (sub3.rho_partial.rank() == sub3.rho_std.rank()).all()
    gb = pd.read_csv(GB).set_index("rule")
    for rk in RULES:
        put(f"gabor.{rk}.mean", gb.loc[CSVNAME[rk], "mean"], 2, rel(GB), f"{CSVNAME[rk]} mean")
        put(f"gabor.{rk}.std", gb.loc[CSVNAME[rk], "std"], 2, rel(GB), f"{CSVNAME[rk]} std")
    rc = pd.read_csv(RC)
    def rcv(rule, roi, col="rho"):
        return rc[(rc.rule == rule) & (rc.roi == roi) & (rc.layer == LAYER[roi])][col].iloc[0]
    put("err.rnd.V1", rcv("Random Weights", "V1"), 3, rel(RC), "Random Weights V1 rho")
    put("err.bp.V1", rcv("Backprop", "V1"), 3, rel(RC), "Backprop V1 rho")
    put("err.bp.V2", rcv("Backprop", "V2"), 3, rel(RC), "Backprop V2 rho")
    put("err.bp.V2.hi", rcv("Backprop", "V2", "ci_hi"), 3, rel(RC), "Backprop V2 ci_hi", lead=False)
    put("err.gap.V1", rcv("Random Weights", "V1") - rcv("Backprop", "V1"), 3, rel(RC), "Random - Backprop V1", sign=True)
    put("err.gap.sweep", sw[(sw.rule == "Random Weights") & (sw.layer == "Conv1") & (sw.roi == "V1") & (sw.res == 224)].rho.mean()
        - bp.mean(), 3, rel(SW), "Random - Backprop V1 224px (resolution study)", sign=True)
    std12 = rc[rc.roi.isin(["V1", "V2"]) & (rc.layer == "Conv1")].rho_std
    put("err.seedsd.min", std12.min(), 3, rel(RC), "min rho_std V1/V2")
    put("err.seedsd.max", std12.max(), 3, rel(RC), "max rho_std V1/V2")
    PF = REPO / "outputs/permutation_results_fdr.csv"
    pf = pd.read_csv(PF)
    puts("err.nsig", str(int((pf.sig_fdr != "ns").sum())), rel(PF), "count sig_fdr != ns")
    put("err.pmax", pf[pf.roi.isin(["V1", "V2"])].p_fdr.max(), 4, rel(PF), "max p_fdr V1/V2")

    # ── CIFAR-10 test accuracy (bnfix, seed 0 only) and v2 training accuracy ──
    ACC = P3 / "accuracy.csv"
    acc = pd.read_csv(ACC).set_index("rule")
    assert (acc.seed_idx == 0).all() and (acc.n_test == 10000).all()
    for rk, r in RULES.items():
        put(f"acc.{rk}", acc.loc[r, "test_acc"], 1, rel(ACC), f"{r} test_acc (seed 0)", pct=True)
    puts("acc.ntest", "10{,}000", rel(ACC), "n_test")
    OLDACC = {"bp": "82.4", "stdp": "63.2", "pc": "56.6", "fa": "39.0", "rnd": "10.0"}  # v2 Table 1 (quoted)
    L = ["\\begin{tabular}{lc}", "\\toprule", "Condition & Test accuracy (\\%) \\\\", "\\midrule"]
    for rk in ("bp", "stdp", "pc", "fa", "rnd"):
        L.append(f"{CSVNAME[rk]} & ${MAC[f'acc.{rk}']}$ \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (PAPER / "tab_accuracy.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    L = ["\\begin{tabular}{lcc}", "\\toprule",
         "Condition & v2: training accuracy (\\%) & v4: test accuracy (\\%) \\\\", "\\midrule"]
    for i, rk in enumerate(("bp", "stdp", "pc", "fa", "rnd")):
        puts(f"v2acc.{rk}", OLDACC[rk], "learning_rules_rsa_paper_v2.tex", f"Table 1, {CSVNAME[rk]}")
        L.append(f"{CSVNAME[rk]} & ${OLDACC[rk]}$ & ${MAC[f'acc.{rk}']}$ \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (PAPER / "tab_acc_old.tex").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ── partial RSA (pixel RDM controlled; bnfix, cross-run, per subject) ───
    PS, PDR, PJ = P3 / "partial_summary.csv", P3 / "partial_draws.csv", P3 / "partial_rsa.json"
    ps_ = pd.read_csv(PS).set_index(["rule", "roi"])
    pdr = pd.read_csv(PDR)
    ppt, pbt = pdr[pdr.boot == -1].iloc[0], pdr[pdr.boot >= 0]
    pj = json.loads(PJ.read_text())
    assert pj["validation_max_dev_partial"] < 6e-5 and pj["validation_max_dev_std"] < 6e-5
    put("pval.dev", max(pj["validation_max_dev_partial"], pj["validation_max_dev_std"]), 5, rel(PJ), "max validation deviation vs v2")
    for rk, r in RULES.items():
        for roi in ROIS:
            x = ps_.loc[(r, roi)]
            for c in ("std", "partial", "delta"):
                dp = 4 if c == "delta" else 3
                put(f"pr.{c}.{rk}.{roi}", x[c], dp, rel(PS), f"{r} {roi} {c}", sign=(c == "delta"))
                put(f"pr.{c}.{rk}.{roi}.lo", x[c + "_lo"], dp, rel(PS), f"{r} {roi} {c}_lo")
                put(f"pr.{c}.{rk}.{roi}.hi", x[c + "_hi"], dp, rel(PS), f"{r} {roi} {c}_hi")
    v1d = ps_.xs("V1", level="roi").delta
    put("pr.d.V1.min", v1d.min(), 3, rel(PS), "min delta V1")
    put("pr.d.V1.max", v1d.max(), 3, rel(PS), "max delta V1")
    assert (ps_.xs("V1", level="roi").delta_hi < 0).all()                     # every V1 decrease significant
    o_std = sorted(RULES, key=lambda k: -ps_.loc[(RULES[k], "V1"), "std"])
    o_par = sorted(RULES, key=lambda k: -ps_.loc[(RULES[k], "V1"), "partial"])
    assert o_std == o_par == ["rnd", "stdp", "bp", "pc", "fa"]
    for roi in ROIS:                                    # same significance pattern with and without pixels
        for a, b in combinations(RULES, 2):
            x = pbt[f"partial|{RULES[a]}|{roi}"] - pbt[f"partial|{RULES[b]}|{roi}"]
            s_par = sig(x.quantile(.025), x.quantile(.975))
            y = dci(f"persub|cr|bnfix|{RULES[a]}|{roi}", f"persub|cr|bnfix|{RULES[b]}|{roi}")
            assert s_par == sig(y[1], y[2]), (a, b, roi)
            if a == "rnd" and b == "bp":
                put(f"prd.rnd.bp.{roi}", ppt[f"partial|random_weights|{roi}"] - ppt[f"partial|backprop|{roi}"], 3, rel(PDR),
                    f"partial Random - BP {roi}", sign=True)
                put(f"prd.rnd.bp.{roi}.lo", x.quantile(.025), 3, rel(PDR), f"partial Random - BP {roi} q.025")
                put(f"prd.rnd.bp.{roi}.hi", x.quantile(.975), 3, rel(PDR), f"partial Random - BP {roi} q.975")
    assert not sig(ps_.loc[("feedback_alignment", "V2"), "partial_lo"], ps_.loc[("feedback_alignment", "V2"), "partial_hi"])
    assert not sig(ps_.loc[("feedback_alignment", "V2"), "std_lo"], ps_.loc[("feedback_alignment", "V2"), "std_hi"])
    L = ["\\begin{tabular}{l" + "ccc" * 2 + "}", "\\toprule",
         " & \\multicolumn{3}{c}{V1} & \\multicolumn{3}{c}{V2} \\\\",
         "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}",
         "Condition & " + " & ".join(["$\\rho_{\\text{subj}}$ & $\\rho_{\\text{partial}}$ [95\\% CI] & $\\Delta$ [95\\% CI]"] * 2) + " \\\\",
         "\\midrule"]
    for rk in ("rnd", "stdp", "bp", "pc", "fa"):
        cells = []
        for roi in ("V1", "V2"):
            g = lambda c: f"{MAC[f'pr.{c}.{rk}.{roi}']}\\;[{MAC[f'pr.{c}.{rk}.{roi}.lo']},{MAC[f'pr.{c}.{rk}.{roi}.hi']}]"
            cells += [f"${MAC[f'pr.std.{rk}.{roi}']}$", f"${g('partial')}$", f"${g('delta')}$"]
        L.append(f"{RNAME[rk]} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (PAPER / "tab_partial.tex").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ── resolution dependence of the V1 gap: model set of the resolution study ──
    # (evaluation-resolution-rsa, commit in external/PROVENANCE.txt; v12 checkpoints, same
    # architecture and training, 5 seeds). Cross-run, per subject, NATIVE evaluation,
    # Random - BP at Conv1 -> V1, seed mean with 95% stimulus-bootstrap CI.
    import hashlib
    EXT = P3 / "external"
    prov = (EXT / "PROVENANCE.txt").read_text()
    for f in ["step4_gaps.csv"] + [f"rsa_seed{i}.csv" for i in range(5)]:
        h = hashlib.sha256((EXT / f).read_bytes()).hexdigest()
        assert f"{f}  <- results/upsampling/{f}  sha256 {h}" in prov, f"{f}: sha256 differs from PROVENANCE.txt"
    G4 = EXT / "step4_gaps.csv"
    g4 = pd.read_csv(G4)
    g4 = g4[(g4.convention == "persub") & (g4.roi == "V1") & (g4.arm == "NATIVE")].set_index("res")
    for px in (32, 224):
        x = g4.loc[px]
        sel = f"persub V1 NATIVE {px}px"
        put(f"res.gap.{px}", x.gap, 4, rel(G4), f"{sel} gap", sign=True)
        put(f"res.gap.{px}.lo", x.boot_lo, 4, rel(G4), f"{sel} boot_lo", sign=True)
        put(f"res.gap.{px}.hi", x.boot_hi, 4, rel(G4), f"{sel} boot_hi", sign=True)
        put(f"res.gap.{px}.3", x.gap, 3, rel(G4), f"{sel} gap", sign=True)          # abstract: 3 decimals
        put(f"res.gap.{px}.lo.3", x.boot_lo, 3, rel(G4), f"{sel} boot_lo")
        put(f"res.gap.{px}.hi.3", x.boot_hi, 3, rel(G4), f"{sel} boot_hi")
        if px == 32:
            assert not sig(x.boot_lo, x.boot_hi)             # text: "do not differ / vanishes at 32 px"
        else:
            assert x.boot_lo > 0                              # text: "exceeds at 224 px"
    RS = [EXT / f"rsa_seed{i}.csv" for i in range(5)]
    rs = pd.concat([pd.read_csv(f) for f in RS])
    rs = rs[(rs.arm == "NATIVE") & (rs.variant == "cr") & (rs.convention == "persub") &
            (rs.layer == "Conv1") & (rs.roi == "V1")]
    m = rs.pivot_table(index=["res", "seed_idx"], columns="rule", values="rho")
    assert m.groupby("res").size().eq(5).all()
    for px in (32, 224):                                       # same source: gap = Random - BP
        assert abs((m.loc[px]["Random Weights"] - m.loc[px]["Backprop"]).mean() - g4.loc[px].gap) < 1e-12
    srcs = "results/paper_v3/external/rsa_seed{0..4}.csv"
    put("res.rnd.32", m.loc[32]["Random Weights"].mean(), 3, srcs, "Random Weights NATIVE cr persub Conv1 V1 32px, seed mean")
    put("res.bp.32", m.loc[32]["Backprop"].mean(), 3, srcs, "Backprop NATIVE cr persub Conv1 V1 32px, seed mean")

    # ── V2 layer assignment: Conv1 (fixed mapping) vs Conv2 (sensitivity) ──────
    VJ, VS = P3 / "v2_conv2.json", P3 / "v2_conv2_summary.csv"
    vj = json.loads(VJ.read_text())
    vs = pd.read_csv(VS).set_index(["layer", "quantity"])
    assert vj["validation_conv1_vs_step2_max_dev"] < 1e-10
    V2PAIRS = [("rnd", "bp"), ("rnd", "fa"), ("bp", "fa"), ("pc", "fa"), ("stdp", "fa")]
    same = {}
    for a, b in V2PAIRS:
        q = f"{RULES[a]} - {RULES[b]}"
        for l in ("Conv1", "Conv2"):
            r = vs.loc[(l, q)]
            assert abs(round(r.point, 4) - vj["checks"][q][l.lower()][0]) < 1e-9   # json == summary
            k = f"v2c.{l}.{a}.{b}"
            put(k, r.point, 4, rel(VS), f"{l} {q} point", sign=True)
            put(k + ".lo", r.ci_lo, 4, rel(VS), f"{l} {q} ci_lo")
            put(k + ".hi", r.ci_hi, 4, rel(VS), f"{l} {q} ci_hi")
        # the Conv1 column is the main analysis: must equal the pairwise-table difference
        assert abs(vs.loc[("Conv1", q)].point - d_("ps", a, b, "V2")[0]) < 1e-12
        same[(a, b)] = vj["checks"][q]["same_sign_and_significance"]
    # text: Random > BP under both; ordering among FA, BP and PC depends on the assignment
    assert same == {("rnd", "bp"): True, ("rnd", "fa"): True, ("bp", "fa"): False,
                    ("pc", "fa"): False, ("stdp", "fa"): True}, same
    assert vs.loc[("Conv2", "random_weights - backprop")].ci_lo > 0
    L = ["\\begin{tabular}{lccc}", "\\toprule",
         "Difference at V2 & Conv1$\\to$V2 (main) & Conv2$\\to$V2 & same sign and significance \\\\", "\\midrule"]
    for a, b in V2PAIRS:
        cells = [f"${MAC[f'v2c.{l}.{a}.{b}']}\\;[{MAC[f'v2c.{l}.{a}.{b}.lo']},{MAC[f'v2c.{l}.{a}.{b}.hi']}]$"
                 for l in ("Conv1", "Conv2")]
        L.append(f"{RNAME[a]} $-$ {RNAME[b]} & " + " & ".join(cells) + f" & {'yes' if same[(a, b)] else 'no'} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (PAPER / "tab_v2conv2.tex").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ── multiple-comparison correction: primary family of 7 vs the earlier 11 ─────
    FD = P3 / "primary_fdr.csv"
    fd = pd.read_csv(FD)
    assert not json.loads((P3 / "primary_fdr.json").read_text())["any_decision_differs_7_vs_11_or_vs_uncorrected"]
    f7 = fd[fd.in_family7]
    assert len(f7) == 7 and len(fd) == 11
    # text: no significance decision changes, in either family, under either correction --
    # including the four FA-at-V2 comparisons of the earlier family of 11
    assert f7[["bh_sig_fam7", "holm_sig_fam7"]].all().all()
    assert fd[["uncorrected_sig", "bh_sig_fam11", "holm_sig_fam11"]].all().all()
    L = ["\\begin{tabular}{llcccccc}", "\\toprule",
         "Comparison & ROI & $\\Delta\\rho_{\\text{subj}}$ [95\\% CI] & $p_{\\text{boot}}$ & BH, 7 & BH, 11 & Holm, 7 & Holm, 11 \\\\",
         "\\midrule"]
    for i, (_, r) in enumerate(fd.iterrows()):
        if i == 7:
            L.append("\\midrule")                    # rows below: family of 11 only (FA at V2)
        a, b = r.comparison.split("-")
        k = f"fdr.{a}.{b}.{r.roi}"
        sel = f"{r.comparison} {r.roi}"
        put(k + ".d", r["diff"], 4, rel(FD), f"{sel} diff", sign=True)
        put(k + ".lo", r.ci_lo, 4, rel(FD), f"{sel} ci_lo")
        put(k + ".hi", r.ci_hi, 4, rel(FD), f"{sel} ci_hi")
        cells = []
        for c in ("p_boot", "bh_p_fam7", "bh_p_fam11", "holm_p_fam7", "holm_p_fam11"):
            if c.endswith("fam7") and not r.in_family7:
                cells.append("--")
            else:
                cells.append(f"${put(f'{k}.{c}', r[c], 4, rel(FD), f'{sel} {c}')}$")
        L.append(f"{RNAME[a]} $-$ {RNAME[b]} & {r.roi} & ${MAC[k + '.d']}\\;[{MAC[k + '.lo']},{MAC[k + '.hi']}]$ & "
                 + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (PAPER / "tab_fdr.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    puts("nprimary.old", "11", rel(FD), "size of the earlier primary family")

    puts("nprimary", str(len(PRIMARY)), "make_macros.py PRIMARY", "number of primary comparisons")
    assert len(PRIMARY) == 7
    for (a, b, roi) in PRIMARY:                       # every primary comparison is significant
        x = dci(f"persub|cr|bnfix|{RULES[a]}|{roi}", f"persub|cr|bnfix|{RULES[b]}|{roi}")
        assert sig(x[1], x[2]), (a, b, roi)

    # ── tables ────────────────────────────────────────────────────────────────
    write_tables(summ, dci, ci, sd)
    lint_tex()

    # ── macros file + manifest ────────────────────────────────────────────────
    lines = ["% generated by scripts/paper_v3/make_macros.py -- do not edit", "\\makeatletter"]
    for k, t in sorted(MAC.items()):
        lines.append(f"\\expandafter\\def\\csname nv@{k}\\endcsname{{{t}}}")
    lines += ["\\makeatother",
              "\\newcommand{\\NV}[1]{\\ifcsname nv@#1\\endcsname\\csname nv@#1\\endcsname"
              "\\else\\errmessage{Undefined number: #1}\\fi}"]
    (PAPER / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    man = pd.DataFrame(MAN).drop_duplicates("key")
    P3.mkdir(parents=True, exist_ok=True)
    man.to_csv(P3 / "numbers_manifest.csv", index=False)
    facts = {"sub_loc": sub_loc, "seed_it": {"trained_all_overlap": seed_it["trained_all_overlap"],
                                             "rnd_overlaps": seed_it["rnd_overlaps"]},
             "overlap_pairs_IT": {f"{a}-{b}": v for (a, b), v in overl.items()},
             "v1_mr_ns": v1_mr_ns, "v2_ns": v2_ns}
    (P3 / "facts.json").write_text(json.dumps(facts, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"{len(MAC)} macros; facts:", (P3 / "facts.json").read_text())


def lint_tex():
    """Qualitative checks on the v3 text (data facts are asserted in main):
    no 'convergence' wording; no sentence has PC exceeding/outperforming BP unless it says
    'no longer'; every sentence with STDP exceeding/outperforming BP names the per-subject
    convention (in the mean-RDM convention STDP - BP at V1 is not significant)."""
    import re
    tex = (PAPER / "learning_rules_rsa_paper_v3.tex").read_text(encoding="utf-8")
    body = "\n".join(l for l in tex.splitlines() if not l.lstrip().startswith("%"))
    body = body[body.index("\\begin{document}"):]
    assert not re.search(r"converg", body, re.I), "convergence wording in v3"
    # the untrained network's V1 advantage at 224 px is a train/eval resolution effect (2608.12408),
    # not evidence that training moves early layers away from V1: that reading must not return
    flat = re.sub(r"\\emph\{([^}]*)\}", r"\1", re.sub(r"\s+", " ", body))
    assert not re.search(r"away from V1-like", flat, re.I), "'away from V1-like' wording in v3"
    assert not re.search(r"moves? early-layer representations", flat, re.I), "'moves early-layer representations' in v3"
    # architecture claims beyond what the 224 px / 32 px results support
    assert not re.search(r"reveals the dominant role", flat, re.I), "'reveals the dominant role' in v3"
    assert not re.search(r"more important than the weight-update rule", flat, re.I), "'more important than the weight-update rule' in v3"
    # dataset: THINGS-fMRI is 3T (Hebart et al. 2023, eLife); Gifford & Cichy 2022 is THINGS-EEG2
    assert not re.search(r"\b7\s*T\b|gifford2022", flat), "7T / gifford2022 in v3"
    # FA lowest holds at V1 only (V2 ordering depends on the layer assignment, tab_v2conv2)
    assert not re.search(r"(FA[^.;]{0,60}lowest|lowest[^.;]{0,40}FA)[^.;]{0,60}V1 and V2", flat), "'FA lowest at V1 and V2' in v3"
    sents = re.split(r"(?<=[.;])\s+", re.sub(r"\s+", " ", body))
    verb = r"\b(exceeds?|exceeding|outperforms?|outperforming|leads?|is above|lies above|higher than)\b"
    def claims(rule, t):   # rule is the grammatical subject: optional (...), at most two words, verb, then BP
        return re.search(r"\b" + rule + r"\b(?:\s*\([^)]*\))?\s+(?:[A-Za-z-]+\s+){0,2}" + verb + r"[^.;]{0,30}\bBP\b", t)
    assert claims("PC", "PC ($0.012$) exceeds BP at V1") and claims("STDP", "STDP outperforms BP")
    assert claims("PC", "PC no longer exceeds BP") and not claims("PC", "PC $>$ FA), and the untrained network still exceeds BP")
    for snt in sents:
        if claims("PC", snt):
            assert "no longer" in snt, ("PC > BP claim", snt[:200])
        if claims("STDP", snt):
            assert "per-subject" in snt, ("STDP > BP without convention", snt[:200])


def cell(t, lo, hi, bold=False, mark=""):
    core = f"\\mathbf{{{t}}}" if bold else t
    return f"${core}\\;[{lo},{hi}]{mark}$"


def write_tables(summ, dci, ci, sd):
    # Table 2 (per subject) and appendix table (mean-RDM) ----------------------
    for ck, fname in (("ps", "tab_main.tex"), ("mr", "tab_main_mr.tex")):
        conv = CONV[ck]
        L = ["\\begin{tabular}{ll" + "c" * 6 + "}", "\\toprule",
             "ROI & Layer & lower bound & " + " & ".join(RNAME[r] for r in RULES) + " \\\\", "\\midrule"]
        for roi in ROIS:
            pts = {rk: summ.loc[f"{conv}|cr|bnfix|{r}|{roi}"].point for rk, r in RULES.items()}
            best = max(pts, key=pts.get) if roi != "IT" else None
            row = [roi, LAYER[roi], cell(MAC[f"lb.cr.{roi}"], txt(summ.loc[f"lower|cr|{roi}"].ci_lo, 3, False),
                                         txt(summ.loc[f"lower|cr|{roi}"].ci_hi, 3, False))
                   if ck == "ps" else "--"]
            for rk, r in RULES.items():
                b = f"rho.{ck}.cr.bn.{rk}.{roi}"
                mark = ""
                if rk != "rnd":
                    d, lo, hi = dci(f"{conv}|cr|bnfix|{r}|{roi}", f"{conv}|cr|bnfix|random_weights|{roi}")
                    mark = "^{\\dagger}" if (lo > 0 or hi < 0) else ""
                row.append(cell(MAC[b], MAC[b + ".lo"], MAC[b + ".hi"], bold=(rk == best), mark=mark))
            L.append(" & ".join(row) + (" \\\\" if roi != "IT" else "$^{\\ddagger}$ \\\\"))
        L += ["\\bottomrule", "\\end{tabular}"]
        (PAPER / fname).write_text("\n".join(L) + "\n", encoding="utf-8")

    # pairwise -----------------------------------------------------------------
    L = ["\\begin{tabular}{lcccc}", "\\toprule", "Pair (A $-$ B) & " + " & ".join(ROIS) + " \\\\", "\\midrule"]
    for a, b in combinations(RULES, 2):
        row = [f"{RNAME[a]} $-$ {RNAME[b]}"]
        for roi in ROIS:
            k = f"d.ps.cr.bn.{a}.{b}.{roi}"
            s = sig(RAW[k + ".lo"], RAW[k + ".hi"])
            core = "\\mathbf{" + MAC[k + ".4"] + "}" if s else MAC[k + ".4"]
            prim = "^{\\mathrm{P}}" if (a, b, roi) in PRIMARY else ""
            row.append(f"${core}\\;[{MAC[k + '.lo.4']},{MAC[k + '.hi.4']}]{prim}$")
        L.append(" & ".join(row) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (PAPER / "tab_pairwise.tex").write_text("\n".join(L) + "\n", encoding="utf-8")

    # sensitivity --------------------------------------------------------------
    L = ["\\begin{tabular}{llcccc}", "\\toprule",
         "ROI & Quantity & all pairs & cross-run & block-cleaned & cross-run, original set \\\\", "\\midrule"]
    for roi in ROIS:
        L.append(f"{roi} & lower bound & " + " & ".join(
            f"${MAC[f'lb.{v}.{roi}']}\\;[{MAC[f'lb.{v}.{roi}.lo']},{MAC[f'lb.{v}.{roi}.hi']}]$" for v in VARS) + " & -- \\\\")
        L.append(f" & upper bound & " + " & ".join(f"${MAC[f'ub.{v}.{roi}']}$" for v in VARS) + " & -- \\\\")
        for rk in RULES:
            L.append(f" & {RNAME[rk]} & " + " & ".join(
                f"${MAC[f'rho.ps.{v}.bn.{rk}.{roi}']}$" for v in VARS) + f" & ${MAC[f'rho.ps.cr.ref.{rk}.{roi}']}$ \\\\")
        if roi != "IT":
            L.append("\\addlinespace")
    L += ["\\bottomrule", "\\end{tabular}"]
    (PAPER / "tab_sens.tex").write_text("\n".join(L) + "\n", encoding="utf-8")

    # diagnostics --------------------------------------------------------------
    L = ["\\begin{tabular}{lcc}", "\\toprule",
         "ROI & $\\rho$ (sub-01 vs sub-02, 45 run-pair means) & $p$ (exact, one-sided) \\\\", "\\midrule"]
    for roi in ROIS:
        L.append(f"{roi} & ${MAC[f'rp.{roi}.rho']}$ & ${MAC[f'rp.{roi}.p']}$ \\\\")
    L += ["\\midrule", "Subject & \\multicolumn{2}{c}{V1 luminance fit: cross-run $\\to$ block-cleaned (change)} \\\\", "\\midrule"]
    for s in SUBJ:
        sk = s.replace("sub-0", "s")
        L.append(f"{s} & \\multicolumn{{2}}{{c}}{{${MAC[f'lum.{sk}.V1.cr']} \\to {MAC[f'lum.{sk}.V1.crb']}$ "
                 f"(${MAC[f'lum.{sk}.V1.chg']}\\,\\%$)}} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (PAPER / "tab_diag.tex").write_text("\n".join(L) + "\n", encoding="utf-8")

    # errata -------------------------------------------------------------------
    rows = [  # (v2 location, quantity, printed in v2 (quoted), correct key)
        ("Abstract, \\S4.2, Tab.~2, \\S5", "Random, V1", "0.076", "err.rnd.V1"),
        ("Abstract, \\S4.2, Tab.~2, \\S5", "BP, V1", "0.034", "err.bp.V1"),
        ("Tab.~2", "BP, V2", "0.019", "err.bp.V2"),
        ("Tab.~2", "BP, V2, CI upper", ".023", "err.bp.V2.hi"),
        ("Abstract, \\S1, \\S4.2, \\S5", "$\\Delta\\rho$ Random $-$ BP, V1", "+0.044", "err.gap.V1"),
        ("\\S4.8", "Gabor peakedness, BP", "2.00", "gabor.bp.mean"),
        ("\\S4.5", "seed SD range, V1/V2", "0.003--0.007", None),
        ("\\S4.4", "FDR-significant comparisons of 40", "30", "err.nsig"),
        ("\\S4.4", "largest FDR $p$ at V1/V2", "$p \\le 0.014$", "err.pmax"),
    ]
    L = ["\\begin{tabular}{llll}", "\\toprule", "Location in v2 & Quantity & printed & correct \\\\", "\\midrule"]
    for i, (loc, q, old, k) in enumerate(rows):
        puts(f"v2quote.{i}", old, "learning_rules_rsa_paper_v2.tex", f"printed value: {q}")
        new = f"${MAC['err.seedsd.min']}$--${MAC['err.seedsd.max']}$" if k is None else f"${MAC[k]}$"
        L.append(f"{loc} & {q} & {old} & {new} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (PAPER / "tab_errata.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

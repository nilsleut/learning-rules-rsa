"""List of every number that changed from paper v2 to v4 (arXiv v3 is the August 2026
correction note only), and the arXiv comment.

Old values are quotations of learning_rules_rsa_paper_v2.tex; the script checks that each
quoted literal really occurs on the cited v2 line. New values come from the manifest
written by make_macros.py (key -> text, source file).

Output: results/paper_v3/CHANGED_NUMBERS.md, paper/arxiv_upload_learning_rules_v3/ARXIV_COMMENT.txt
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
V2 = REPO / "paper/arxiv_upload_learning_rules_v2/learning_rules_rsa_paper_v2.tex"
P3 = REPO / "results/paper_v3"
PAPER = REPO / "paper/arxiv_upload_learning_rules_v3"

# (quantity, v2 lines, old literal as printed, [new keys: main, mean-RDM or None], note)
R = {"rnd": "Random", "bp": "BP", "fa": "FA", "pc": "PC", "stdp": "STDP"}
SPEC = [
    ("Random, V1", [42, 279, 398], "0.076", ["rho.ps.cr.bn.rnd.V1", "rho.mr.cr.bn.rnd.V1"], ""),
    ("BP, V1", [42, 279, 285, 398, 402], "0.034", ["rho.ps.cr.bn.bp.V1", "rho.mr.cr.bn.bp.V1"], ""),
    ("Random - BP, V1", [42, 121, 279, 398], "+0.044", ["d.ps.cr.bn.rnd.bp.V1", "d.mr.cr.bn.rnd.bp.V1"], "v2 quoted the resolution study's gap"),
    ("STDP, V1", [42, 279, 285, 398, 402], "0.064", ["rho.ps.cr.bn.stdp.V1", "rho.mr.cr.bn.stdp.V1"], "repaired evaluation; 'highest among trained' kept only per subject"),
    ("PC, V1", [279, 285, 398, 402], "0.056", ["rho.ps.cr.bn.pc.V1", "rho.mr.cr.bn.pc.V1"], "repaired evaluation; 'PC outperforms BP' withdrawn"),
    ("FA, V1", [42, 287], "0.012", ["rho.ps.cr.bn.fa.V1", "rho.mr.cr.bn.fa.V1"], ""),
    ("Random, V2", [279], "0.043", ["rho.ps.cr.bn.rnd.V2", "rho.mr.cr.bn.rnd.V2"], ""),
    ("BP, V2", [279], "0.019", ["rho.ps.cr.bn.bp.V2", "rho.mr.cr.bn.bp.V2"], ""),
    ("Random - BP, V2", [279], "+0.024", ["d.ps.cr.bn.rnd.bp.V2", "d.mr.cr.bn.rnd.bp.V2"], ""),
    ("BP, LOC", [42, 281], "0.012", ["rho.ps.cr.bn.bp.LOC", "rho.mr.cr.bn.bp.LOC"], ""),
    ("Random, LOC", [42, 281, 312, 316], "-0.005", ["rho.ps.cr.bn.rnd.LOC", "rho.mr.cr.bn.rnd.LOC"], ""),
    ("BP - Random, LOC", [281], "+0.017", ["d.ps.cr.bn.bp.rnd.LOC", "d.mr.cr.bn.bp.rnd.LOC"], ""),
    ("IT range, min", [42, 283, 312, 316], "0.008", ["range.ps.IT.min", "range.mr.IT.min"], ""),
    ("IT range, max", [42, 283, 312, 316], "0.014", ["range.ps.IT.max", "range.mr.IT.max"], ""),
    ("Random, IT (FC1)", [316], "0.008", ["rho.ps.cr.bn.rnd.IT", "rho.mr.cr.bn.rnd.IT"], ""),
    ("noise ceiling lower, V1", [275], "0.07", ["lb.cr.V1", None], "Nili lower bound, cross-run"),
    ("noise ceiling lower, V2", [275], "0.05", ["lb.cr.V2", None], ""),
    ("noise ceiling lower, LOC", [275], "0.03", ["lb.cr.LOC", None], ""),
    ("noise ceiling lower, IT", [275], "0.04", ["lb.cr.IT", None], ""),
    ("noise ceiling upper, V1", [275], "0.11", ["ub.cr.V1", None], "not informative at N=3; appendix only"),
    ("noise ceiling upper, V2", [275], "0.09", ["ub.cr.V2", None], "appendix only"),
    ("noise ceiling upper, LOC", [275], "0.06", ["ub.cr.LOC", None], "appendix only"),
    ("noise ceiling upper, IT", [275, 400], "0.07", ["ub.cr.IT", None], "appendix only"),
    ("seed SD range V1/V2, min", [349], "0.003", ["seedsd.v12.min", None], ""),
    ("seed SD range V1/V2, max", [349], "0.007", ["seedsd.v12.max", None], "v2 source gave 0.011"),
    ("partial RSA decrease V1, smallest", [357], "-0.004", ["pr.d.V1.max", None], "recomputed: repaired set, cross-run, per subject"),
    ("partial RSA decrease V1, largest", [357], "-0.008", ["pr.d.V1.min", None], ""),
    ("Gabor peakedness, BP", [390, 394], "2.00", ["gabor.bp.mean", None], "rounding"),
    ("FA partial V2", [404], "0.001", ["pr.partial.fa.V2", None], "recomputed"),
    ("luminance vs V1 (correction note)", [90], "0.075", ["lumv1.ps.cr", "lumv1.mr.cr"], "now per subject, cross-run"),
    ("untrained network vs V1 (correction note)", [91], "0.076", ["rho.ps.cr.bn.rnd.V1", "rho.mr.cr.bn.rnd.V1"], ""),
    ("calibration control, CIFAR statistics", [82], "+0.041", ["cal.B", None], "unchanged (original analysis)"),
    ("calibration control, THINGS statistics", [82], "+0.033", ["cal.D", None], "unchanged (original analysis)"),
    ("initialisation scale factor", [100], "2.4", ["init.ratio", None], "unchanged"),
    ("accuracy, BP (Table 1)", [250], "82.4", ["acc.bp", None], "v2: training batches; v4: test set, seed 0"),
    ("accuracy, STDP (Table 1)", [251], "63.2", ["acc.stdp", None], "v2: training batches; v4: test set, seed 0"),
    ("accuracy, PC (Table 1)", [252], "56.6", ["acc.pc", None], "v2: training batches; v4: test set, seed 0"),
    ("accuracy, FA (Table 1)", [253], "39.0", ["acc.fa", None], "v2: training batches; v4: test set, seed 0"),
    ("accuracy, Random (Table 1)", [254], "10.0", ["acc.rnd", None], "chance level in both"),
    ("STDP accuracy in Methods", [216], "63", ["acc.stdp", None], ""),
    ("FA accuracy in text", [287, 404], "39", ["acc.fa", None], ""),
    ("BP / Random accuracy in Discussion", [406], "82", ["acc.bp", None], "Random: 10 -> see acc.rnd"),
]
TABLE = {  # v2 Table 2, lines 299-302: (rule, roi): (rho, lo, hi)
    ("rnd", "V1"): ("0.076", ".072", ".080"), ("bp", "V1"): ("0.034", ".029", ".037"),
    ("fa", "V1"): ("0.012", ".008", ".016"), ("pc", "V1"): ("0.056", ".052", ".060"),
    ("stdp", "V1"): ("0.064", ".060", ".068"),
    ("rnd", "V2"): ("0.043", ".040", ".047"), ("bp", "V2"): ("0.019", ".015", ".023"),
    ("fa", "V2"): ("0.004", ".000", ".008"), ("pc", "V2"): ("0.028", ".024", ".032"),
    ("stdp", "V2"): ("0.036", ".032", ".040"),
    ("rnd", "LOC"): ("-0.005", "-.009", "-.001"), ("bp", "LOC"): ("0.012", ".008", ".016"),
    ("fa", "LOC"): ("0.006", ".002", ".009"), ("pc", "LOC"): ("0.006", ".002", ".010"),
    ("stdp", "LOC"): ("0.006", ".002", ".009"),
    ("rnd", "IT"): ("0.008", ".004", ".012"), ("bp", "IT"): ("0.013", ".009", ".017"),
    ("fa", "IT"): ("0.012", ".008", ".015"), ("pc", "IT"): ("0.014", ".010", ".017"),
    ("stdp", "IT"): ("0.012", ".008", ".016"),
}
TLINE = {"V1": 299, "V2": 300, "LOC": 301, "IT": 302}


def main():
    v2 = V2.read_text(encoding="utf-8").splitlines()
    man = pd.read_csv(P3 / "numbers_manifest.csv", keep_default_na=False).set_index("key")

    def on_line(lit, ln):
        t = v2[ln - 1].replace("$-$", "-").replace("--", "-")
        return lit.lstrip("+") in t

    rows = []
    for q, lines, old, keys, note in SPEC:
        miss = [ln for ln in lines if not on_line(old, ln)]
        assert not miss, (q, old, miss)
        main_k, mr_k = keys
        rows.append({"quantity": q, "v2 lines": ", ".join(map(str, lines)), "v2 printed": old,
                     "v4 (per subject, cross-run)": man.loc[main_k, "text"],
                     "v4 mean-RDM": man.loc[mr_k, "text"] if mr_k else "",
                     "key": main_k, "source": man.loc[main_k, "source"], "note": note})
    for (rk, roi), (rho, lo, hi) in TABLE.items():
        ln = TLINE[roi]
        for lit in (rho, lo, hi):
            assert on_line(lit, ln), (rk, roi, lit)
        k = f"rho.ps.cr.bn.{rk}.{roi}"
        rows.append({"quantity": f"Table 2: {R[rk]}, {roi}", "v2 lines": str(ln),
                     "v2 printed": f"{rho} [{lo}, {hi}]",
                     "v4 (per subject, cross-run)": f"{man.loc[k, 'text']} [{man.loc[k + '.lo', 'text']}, {man.loc[k + '.hi', 'text']}]",
                     "v4 mean-RDM": man.loc[f"rho.mr.cr.bn.{rk}.{roi}", "text"], "key": k,
                     "source": man.loc[k, "source"], "note": "v2: mean-RDM, all pairs, original set"})
    removed = [("FDR-significant comparisons", "327", "30", "permutation/FDR replaced by bootstrap CIs"),
               ("largest FDR p at V1/V2", "323, 327", "p \\le 0.014", "removed"),
               ("BP vs trained rules at LOC, p", "281, 327", "0.051-0.079", "removed (bootstrap CIs instead)"),
               ("Random vs BP at IT, p", "283, 327", "0.051", "removed (bootstrap CI instead)"),
               ("PC vs STDP at V1, p", "279", "0.004", "removed"),
               ("partial RSA rows PC and STDP", "368-369", "0.058/0.054, 0.067/0.061", "removed (evaluation defect)"),
               ("correction note PC/STDP before/after", "64-65", "0.056 -> 0.016, 0.064 -> 0.037", "superseded by main table"),
               ("repair bound", "62", "0.0013", "not printed in v4")]
    df = pd.DataFrame(rows)
    df.to_csv(P3 / "changed_numbers.csv", index=False)
    md = ["# Changed numbers, paper v2 -> v4", "",
          "Generated by `scripts/paper_v3/make_changes.py`. *v2 printed* is quoted from "
          "`learning_rules_rsa_paper_v2.tex` (checked against the cited line); v4 values are from "
          "`results/paper_v3/numbers_manifest.csv`. v2 values used all stimulus pairs, the mean-RDM "
          "convention and the original model set; v4's main column uses cross-run pairs, the "
          "per-subject convention and the repaired model set.", "",
          "| Quantity | v2 line(s) | v2 printed | v4 per subject | v4 mean-RDM | Source | Note |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['quantity']} | {r['v2 lines']} | {r['v2 printed']} | {r['v4 (per subject, cross-run)']} | "
                  f"{r['v4 mean-RDM']} | `{r['source']}` | {r['note']} |")
    md += ["", "## Removed in v4", "", "| Quantity | v2 line(s) | v2 printed | Why |", "|---|---|---|---|"]
    md += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in removed]
    (P3 / "CHANGED_NUMBERS.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    comment = ("v4 replaces the noise ceiling by the leave-one-subject-out lower bound of Nili et al. "
               "(2014), since the split-half bounds of earlier versions were not valid with three subjects "
               "and could not be reproduced, restricts all analyses to stimulus pairs from different fMRI "
               "runs, because a presentation order shared by two subjects inflated the bound at LOC and IT, "
               "updates predictive coding and STDP after an evaluation-mode fix, and corrects double-rounded "
               "values and the accuracy table. The claim that the models exhaust the available signal is "
               "withdrawn, predictive coding no longer exceeds backpropagation at V1, STDP exceeds it only in "
               "the per-subject convention, and IT differences are reported as not resolvable at the low "
               "reliability of IT rather than as convergence. The untrained network still exceeds "
               "backpropagation at V1 and V2 at 224 px evaluation, backpropagation still exceeds it at LOC, and "
               "feedback alignment remains lowest at V1.")
    n_sent = comment.replace("et al.", "et al").count(". ") + 1
    assert n_sent <= 3, n_sent
    (PAPER / "ARXIV_COMMENT.txt").write_text(comment + "\n", encoding="utf-8")
    print(f"{len(rows)} changed numbers; {len(removed)} removed; comment {len(comment)} chars, {n_sent} sentences")


if __name__ == "__main__":
    main()

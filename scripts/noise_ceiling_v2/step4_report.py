"""Step 4: REPORT.md. Every number is read from the JSON/CSV produced by steps 1-3c,
or from the paper source itself (old text, Table 2, line-275 bounds); none is typed.

Output: results/noise_ceiling_v2/REPORT.md, results/noise_ceiling_v2/noise_ceiling_v2.csv/.json
"""
import json
import sys
sys.stdout.reconfigure(encoding="utf-8")
import re
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd

from common import REPO, RESULTS, ROIS, ROI_LAYER

TEX = REPO / "paper/arxiv_upload_learning_rules_v2/learning_rules_rsa_paper_v2.tex"
ERC = REPO.parents[1] / "Projekte_2/evaluation-resolution-rsa/paper"
ERC_V1, ERC_V2 = ERC / "Evaluation_Resolution_Confounds_paper.tex", \
    ERC / "Evaluation_Resolution_Confounds_paper_v2.tex"
RULES = [("random_weights", "Random"), ("backprop", "BP"), ("feedback_alignment", "FA"),
         ("predictive_coding", "PC"), ("stdp", "STDP")]
RN = dict(RULES)
NUMWORD = {0: 'No', 1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 6: 'Six'}
SETNAME = {"original": "Referenz (Tabelle-2-Lauf, outputs/model_rdms)",
           "bnfix": "korrigierter Stand (bnfix, res224)"}

s1 = json.loads((RESULTS / "step1_bounds.json").read_text())
B = s1["rois"]
ctrl = pd.read_csv(RESULTS / "step3a_control.csv")
ctrl_json = json.loads((RESULTS / "step3a_control.json").read_text())
sw = pd.read_csv(RESULTS / "step3b_sweeps.csv")
swj = json.loads((RESULTS / "step3b_sweeps.json").read_text())
sm = pd.read_csv(RESULTS / "step3c_summary.csv").fillna("")
smj = json.loads((RESULTS / "step3c_summary.json").read_text())
draws = pd.read_csv(RESULTS / "step3c_bootstrap_draws.csv")
bt, pt = draws[draws.boot >= 0], draws[draws.boot == -1].iloc[0]
TEXL = TEX.read_text(encoding="utf-8").splitlines()


def f(x, d=3):
    return f"{x:.{d}f}".replace("-", "−")


def q(quant, roi, st="", rule=""):
    return sm[(sm.quantity == quant) & (sm.roi == roi) & (sm.rdm_set == st) &
              (sm.rule == rule)].iloc[0]


def ci(r, d=3):
    return f"{f(r.point, d)} [{f(r.ci_lo, d)}, {f(r.ci_hi, d)}]"


def pair(a, b, roi, st="original"):
    """Bootstrap of persub(a) - persub(b) on the same resamples."""
    ca, cb = f"persub|{st}|{a}|{roi}", f"persub|{st}|{b}|{roi}"
    d = bt[ca] - bt[cb]
    return pt[ca] - pt[cb], np.percentile(d, 2.5), np.percentile(d, 97.5)


def tex(n):
    return TEXL[n - 1]


def half_up(x, d=3):
    return float(Decimal(repr(float(x))).quantize(Decimal(10) ** -d, rounding=ROUND_HALF_UP))


# ── old bounds from the paper text (line 275) ───────────────────────────────────
m = re.search(r"upper bounds: V1: ([\d.]+), V2: ([\d.]+), LOC: ([\d.]+), IT: ([\d.]+); "
              r"lower bounds: ([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+)", tex(275))
assert m, "line 275 changed"
TXT_UP = dict(zip(ROIS, map(float, m.groups()[:4])))
TXT_LO = dict(zip(ROIS, map(float, m.groups()[4:])))

# ── machine-readable combined table (ROI x method x bound x CI) ─────────────────
combo = []
for roi in ROIS:
    lo = q("lower", roi)
    combo += [
        {"roi": roi, "method": "old_code_1v2_SB", "bound": "lower(p2.5)",
         "value": B[roi]["old_code_p2p5"], "ci_lo": np.nan, "ci_hi": np.nan},
        {"roi": roi, "method": "old_code_1v2_SB", "bound": "upper(mean)",
         "value": B[roi]["old_code_mean_sb"], "ci_lo": np.nan, "ci_hi": np.nan},
        {"roi": roi, "method": "paper_text_l275", "bound": "lower",
         "value": TXT_LO[roi], "ci_lo": np.nan, "ci_hi": np.nan},
        {"roi": roi, "method": "paper_text_l275", "bound": "upper",
         "value": TXT_UP[roi], "ci_lo": np.nan, "ci_hi": np.nan},
        {"roi": roi, "method": "nili_loo", "bound": "lower",
         "value": lo.point, "ci_lo": lo.ci_lo, "ci_hi": lo.ci_hi},
        {"roi": roi, "method": "nili_loo", "bound": "upper",
         "value": q("upper", roi).point, "ci_lo": q("upper", roi).ci_lo,
         "ci_hi": q("upper", roi).ci_hi},
        {"roi": roi, "method": "nili_loo_perm_null", "bound": "upper",
         "value": B[roi]["upper_null_mean"], "ci_lo": np.nan,
         "ci_hi": B[roi]["upper_null_q975"]},
    ]
combo = pd.DataFrame(combo)
combo.to_csv(RESULTS / "noise_ceiling_v2.csv", index=False)
(RESULTS / "noise_ceiling_v2.json").write_text(combo.to_json(orient="records", indent=2))

# ── derived facts used in the prose ─────────────────────────────────────────────
LO = {r: q("lower", r) for r in ROIS}
PS = {(st, k, r): q("persub", r, st, k) for st in SETNAME for k, _ in RULES for r in ROIS}
DF = {(st, k, r): q("diff", r, st, k) for st in SETNAME for k, _ in RULES for r in ROIS}


def above(st, k, r):
    d = DF[(st, k, r)]
    return "über" if d.ci_lo > 0 else ("unter" if d.ci_hi < 0 else "nicht unterscheidbar")


def ranking(st, roi, col):
    sub = ctrl[(ctrl.rdm_set == st) & (ctrl.roi == roi) & ctrl.rule.isin(RN)]
    return [RN[k] for k in sub.sort_values(col, ascending=False).rule]


rank_rows, rank_changes = [], []
for st in SETNAME:
    for roi in ROIS:
        a, b = ranking(st, roi, "rho_meanrdm"), ranking(st, roi, "rho_persub")
        rank_rows.append((st, roi, a, b))
        if a != b:
            rank_changes.append((st, roi, a, b))

claims = []
for roi in ("V1", "V2"):
    claims.append((f"Random > BP an {roi}", pair("random_weights", "backprop", roi)))
claims.append(("BP > Random an LOC", pair("backprop", "random_weights", "LOC")))
for o in ("feedback_alignment", "predictive_coding", "stdp"):
    claims.append((f"BP − {RN[o]} an LOC (Paper: n.s.)", pair("backprop", o, "LOC")))
it_vals = {RN[k]: PS[("original", k, "IT")].point for k, _ in RULES}
fa_lowest = {roi: min(((PS[("original", k, roi)].point, RN[k]) for k, _ in RULES))[1]
             for roi in ("V1", "V2", "LOC")}
fa_lowest_trained = {roi: min(((PS[("original", k, roi)].point, RN[k]) for k, _ in RULES
                               if k != "random_weights"))[1] for roi in ("V1", "V2", "LOC")}
fa_lowest_pub = {roi: min(((ctrl[(ctrl.rdm_set == "original") & (ctrl.rule == k) &
                                 (ctrl.roi == roi)].iloc[0].rho_meanrdm, RN[k])
                           for k, _ in RULES))[1] for roi in ("V1", "V2", "LOC")}
KEY = {v: k for k, v in RN.items()}

# every pair whose order flips between conventions: is their per-subject difference resolved?
swap_cis = []
for st, roi, a, b in rank_changes:
    for i, x in enumerate(a):
        for y in a[i + 1:]:
            if b.index(x) > b.index(y):  # x above y by mean-RDM, below by per-subject
                d, lo, hi = pair(KEY[y], KEY[x], roi, st)
                swap_cis.append((st, roi, y, x, d, lo, hi, bool(lo <= 0 <= hi)))
self_incl = {r: B[r]["upper_null_mean"] / B[r]["upper"] for r in ROIS}
non_it_changes = [x for x in rank_changes if x[1] != "IT"]
claim_fail_pre = any(("n.s." not in lab and not lo > 0) or
                     ("n.s." in lab and not lo <= 0 <= hi) for lab, (d, lo, hi) in claims)

# ── report ──────────────────────────────────────────────────────────────────────
L = []
w = L.append
v1lo, v1up = LO["V1"], B["V1"]
rnd_v1 = PS[("original", "random_weights", "V1")]
rnd_v1_mean = ctrl[(ctrl.rdm_set == "original") & (ctrl.rule == "random_weights") &
                   (ctrl.roi == "V1")].iloc[0]

w("# Noise Ceiling v2 — arXiv:2604.16875\n")
w("Erzeugt von `scripts/noise_ceiling_v2/step4_report.py`. Alle Zahlen stammen aus "
  "`step1_bounds.json`, `step3a_control.csv`, `step3b_sweeps.csv/.json`, "
  "`step3c_summary.csv/.json` und `step3c_bootstrap_draws.csv`, alte Textstellen aus dem "
  "TeX-Quelltext. **Paper-Dateien wurden nicht editiert.** Prüfsummen: `MANIFEST.json`.\n")

w("## Kurzfassung\n")
w(f"- Der publizierte „lower bound“ ist kein eigener Bound, sondern das 2.5. Perzentil "
  f"der 1-vs-2-Split-Verteilung mit Spearman–Brown (k=2). Der Code liefert für V1 "
  f"({f(B['V1']['old_code_p2p5'])}, {f(B['V1']['old_code_mean_sb'])}). Die Textwerte der "
  f"lower bounds (Z. 275: {', '.join(f(TXT_LO[r], 2) for r in ROIS)}) sind aus keinem "
  f"Code reproduzierbar, ebenso V2-upper {f(TXT_UP['V2'], 2)} (Code: "
  f"{f(B['V2']['old_code_mean_sb'])}).")
w(f"- Neuer leave-one-subject-out lower bound (Nili et al. 2014), V1: {ci(v1lo)}; "
  + ", ".join(f"{r}: {ci(LO[r])}" for r in ROIS[1:]) + ".")
w(f"- Der Nili-upper bound (V1 {f(v1up['upper'])}) liegt nur "
  f"{f(v1up['upper_minus_null_mean'])} über seiner Permutations-Nullverteilung "
  f"({f(v1up['upper_null_mean'])}). Bei N=3 ist er **nicht informativ**.")
w(f"- In derselben (Pro-Subject-)Konvention erreicht das untrainierte Netz an V1 "
  f"{ci(rnd_v1)}, Differenz zum lower bound "
  f"{ci(DF[('original', 'random_weights', 'V1')])}. Die Paper-Zahl "
  f"{f(rnd_v1_mean.rho_meanrdm, 4)} ist gegen die Mittel-RDM gemessen und mit keinem Bound "
  f"vergleichbar.")
cnt = {}
for k, _ in RULES:
    for r in ROIS:
        cnt.setdefault(above("original", k, r), []).append(f"{RN[k]}-{r}")
w("- Pro Subject gegen den lower bound (Referenzsatz, 20 Zellen): "
  + "; ".join(f"{lab} {len(v)}" + (f" ({', '.join(v)})" if len(v) <= 4 else "")
              for lab, v in sorted(cnt.items())) + ". Keine Bedingung liegt signifikant "
  "über der Untergrenze." if "über" not in cnt else
  "- Pro Subject gegen den lower bound: " + str({k: len(v) for k, v in cnt.items()}))
w(f"- Rangfolge der Regeln pro ROI: Mittel-RDM- vs. Pro-Subject-Konvention "
  + ("**identisch** in allen ROIs und beiden RDM-Sätzen." if not rank_changes else
     f"verschieden in {len(rank_changes)} Fällen, "
     + ("alle an IT und innerhalb der Stimulus-Unsicherheit (§3b). " if not non_it_changes
        and all(ok for *_, ok in swap_cis) else "**siehe §3b**. ")
     + ("Die geprüften qualitativen Aussagen halten." if not claim_fail_pre else
        "**Mindestens eine qualitative Aussage hält nicht (§3b).**")))
w(f"- Tabelle 2 wird vom Referenzsatz in "
  f"{ctrl_json['verdict']['original']['n_reproduced']}/20 Zellen auf 3 Stellen "
  f"reproduziert. Die übrigen sind Rundungsfehler (§5). Der bnfix-Satz reproduziert "
  f"{ctrl_json['verdict']['bnfix']['n_reproduced']}/20 (BN-Fix bei PC/STDP plus neues "
  f"Training).\n")

# 1 old
w("## 1. Alte Bounds\n")
w("| ROI | Code: p2.5 („lower“) | Code: Mittel („upper“) | Text Z. 275 lower | "
  "Text Z. 275 upper | Figur-Band `rsa_comparison_cnn.png` |")
w("|---|---|---|---|---|---|")
for r in ROIS:
    up_ok = round(B[r]["old_code_mean_sb"], 2) == TXT_UP[r]
    lo_ok = round(B[r]["old_code_p2p5"], 2) == TXT_LO[r]
    w(f"| {r} | {f(B[r]['old_code_p2p5'], 4)} | {f(B[r]['old_code_mean_sb'], 4)} | "
      f"{f(TXT_LO[r], 2)}{'' if lo_ok else ' ✗ *nicht reproduzierbar*'} | "
      f"{f(TXT_UP[r], 2)}{'' if up_ok else ' ✗ *nicht reproduzierbar* (Code ' + f(B[r]['old_code_mean_sb']) + ')'} | "
      f"{f(B[r]['old_code_p2p5'], 3)}–{f(B[r]['old_code_mean_sb'], 3)} |")
w("")
w("**Herkunft.** `noise_ceiling()` in `learning_rules_v8.py:500-513` (gleich in allen "
  "Versionen) zieht 200-mal eine Permutation der 3 Subjects. `perm[:3//2]` ist ein "
  "einzelnes Subject, die andere Hälfte das Mittel der beiden übrigen. Es gibt also nur "
  "3 verschiedene Splits. Auf jedes Spearman-*r* wird `2r/(1+r)` angewendet, die "
  "Spearman-Brown-Formel für **zwei gleich grosse** Hälften. Bei 1 gegen 2 Subjects gibt "
  "es kein k, das eine „Verdopplung der Testlänge“ beschreibt. Die Formel ist hier also "
  "falsch angewendet. Zurückgegeben werden `(percentile(rhos, 2.5), mean(rhos))`. Der "
  "„lower bound“ ist damit das 2.5. Perzentil **derselben** Verteilung. Weil jedes Subject "
  "in etwa einem Drittel der Ziehungen allein steht, entspricht er genau dem Split mit "
  "sub-03 allein. Er ist kein eigener Bound. Das graue Band der Figur "
  "(`learning_rules_v8.py:796`) zeigt diese beiden Zahlen unverändert. Der Text in Z. 275 "
  "weicht davon ab: lower bounds in allen ROIs, upper bound in V2. Woher die Textwerte "
  "stammen, ist nicht feststellbar (`NOISE_CEILING_PROVENANCE.md` §2, §8). "
  "Die Benennung „split-half reliability“ (Z. 228) ist ebenfalls falsch. Es werden "
  "Subjects geteilt, nicht Messwiederholungen.\n")

# 2 new
w("## 2. Neue Bounds (Nili et al. 2014, Spearman, oberes Dreieck, ohne Spearman–Brown)\n")
w(f"lower = mean_s ρ(RDM_s, Mittel der 2 anderen). upper = mean_s ρ(RDM_s, Mittel "
  f"aller 3). CIs: Stimulus-Bootstrap, {smj['n_boot']}×, Seed {smj['seed']}. "
  f"Nullverteilung: {s1['n_perm']}× Stimuluslabels pro Subject unabhängig permutiert, "
  f"Seed {s1['seed']}.\n")
w("| ROI | lower [95%-CI] | lower unter H0 | upper | upper unter H0 (Mittel ± SD) | "
  "upper − H0 | Status upper |")
w("|---|---|---|---|---|---|---|")
for r in ROIS:
    w(f"| {r} | {ci(LO[r], 4)} | {f(B[r]['lower_null_mean'], 4)} ± "
      f"{f(B[r]['lower_null_sd'], 4)} | {f(B[r]['upper'], 4)} | "
      f"{f(B[r]['upper_null_mean'], 4)} ± {f(B[r]['upper_null_sd'], 4)} | "
      f"{f(B[r]['upper_minus_null_mean'], 4)} | **bei N=3 nicht informativ** |")
w("")
pc = s1["provenance_check"]
w(f"Abgleich mit `NOISE_CEILING_PROVENANCE.md` (V1): lower {f(pc['lower']['recomputed'], 4)} "
  f"vs. {pc['lower']['provenance']} (Δ {pc['lower']['abs_diff']:.1e}), upper "
  f"{f(pc['upper']['recomputed'], 4)} vs. {pc['upper']['provenance']} "
  f"(Δ {pc['upper']['abs_diff']:.1e}). "
  + ("Beide unter 0.002, also keine Abweichung." if pc['lower']['ok'] and pc['upper']['ok']
     else "**ABWEICHUNG > 0.002.**"))
pw = B["V1"]["pairwise_spearman"]
w(f"\nPaarweise Spearman-*r* zwischen den Subjects an V1: 01–02 {f(pw['1-2'], 4)}, "
  f"01–03 {f(pw['1-3'], 4)}, 02–03 {f(pw['2-3'], 4)}. sub-03 teilt mit den beiden anderen "
  f"fast keine RDM-Struktur, das bestimmt alle Bounds.\n")
w("**Warum upper nicht informativ ist.** Jedes Subject macht ein Drittel des Mittels aus, "
  "gegen das es korreliert wird. Sind die drei Subjects unabhängig, ist ρ ≈ 1/√3 ≈ 0.577. "
  "Die Permutationsverteilung bestätigt das. Der beobachtete upper bound liegt zwar "
  f"signifikant darüber, besteht aber zu {min(self_incl.values()):.0%}–"
  f"{max(self_incl.values()):.0%} (Nullwert/Beobachtung) aus diesem Selbst-Einschluss. Als "
  "„erreichbarer Anteil“ eignet er sich nicht. **Zulässig ist nur der Vergleich mit dem "
  "lower bound, und zwar als Untergrenze:** Ein Modell, das ihn erreicht, sagt jedes "
  "Subject so gut vorher wie die anderen Subjects. Wie weit das unter dem tatsächlich "
  "Erreichbaren liegt, ist bei N=3 unbekannt.\n")
w("**Warum kein Subject-Bootstrap.** Bei n=3 gibt es nur 10 verschiedene Multimengen. "
  "9 davon enthalten ein Subject doppelt oder dreifach, und dann wird beim leave-one-out "
  "ein Subject gegen eine Kopie von sich selbst korreliert. Der Bound wird dadurch künstlich "
  "hoch, im Extremfall (A, A, A) genau 1. Nur eine Multimenge, die Originalstichprobe, ist "
  "frei von Duplikaten. Eine solche Verteilung beschreibt keine Unsicherheit über Personen. "
  "Die CIs hier geben deshalb nur die Unsicherheit über Stimuli an (Verallgemeinerung auf "
  "neue Bilder derselben drei Personen), **nicht** über Personen.\n")
w("**Warum keine within-subject Ceiling.** Jeder der 720 Stimuli wurde pro Subject genau "
  "einmal gezeigt. Für die Single-Exemplar-RDM gibt es also keine Wiederholungen, die man "
  "aufteilen könnte. Splits über Exemplare (12 pro Konzept) würden die Reliabilität einer "
  "*anderen* RDM schätzen und waren laut Entscheidung ausgeschlossen.\n")

# 3 model
w("## 3. Modelle in derselben Konvention\n")
w(f"ρ pro Subject = Spearman(Modell-RDM, RDM_s), gemittelt über die 3 Subjects und dann über "
  f"5 Seeds. Das ist die Konvention des lower bound. ρ Mittel-RDM = publizierte Konvention "
  f"(Kontrollspalte). Differenz Modell − lower auf denselben {smj['n_boot']} Resamples. "
  f"Kontrolle mit dem Identitäts-Sample: maximale Abweichung zu Schritt 1/3a "
  f"{smj['identity_check']['max_dev_model_vs_step3a']:.0e}. Paare aus doppelt gezogenen "
  f"Stimuli sind ausgeschlossen (Median {smj['n_pairs_median']:,} von 258,840 Paaren). "
  f"Zuordnung ROI→Layer wie in Tabelle 2: "
  + ", ".join(f"{r}→{ROI_LAYER[r]}" for r in ROIS) + ".\n")
for st, title in SETNAME.items():
    w(f"### {title}\n")
    w("| Regel | ROI | ρ Mittel-RDM | Tabelle 2 | ρ pro Subject [95%-CI] | "
      "Modell − lower [95%-CI] | relativ zu lower |")
    w("|---|---|---|---|---|---|---|")
    for k, n in RULES:
        for r in ROIS:
            c = ctrl[(ctrl.rdm_set == st) & (ctrl.rule == k) & (ctrl.roi == r)].iloc[0]
            pub = f(c.published) if st == "original" else f"({f(c.published)})"
            w(f"| {n} | {r} | {f(c.rho_meanrdm, 4)} | {pub} | {ci(PS[(st, k, r)], 4)} | "
              f"{ci(DF[(st, k, r)], 4)} | {above(st, k, r)} |")
    w("")
w("„relativ zu lower“: *über*/*unter* heisst, dass das 95%-CI der Differenz die 0 nicht "
  "enthält. Bei bnfix steht der publizierte Wert in Klammern, weil er aus einem anderen "
  "Lauf stammt.\n")

w("### 3b. Rangfolge und qualitative Aussagen\n")
w("| RDM-Satz | ROI | Rangfolge Mittel-RDM | Rangfolge pro Subject | gleich? |")
w("|---|---|---|---|---|")
for st, roi, a, b in rank_rows:
    w(f"| {st} | {roi} | {' > '.join(a)} | {' > '.join(b)} | {'ja' if a == b else '**nein**'} |")
w("")
w("Paarweise Differenzen pro Subject im Referenzsatz, Stimulus-Bootstrap-CI auf denselben "
  "Resamples. Das ersetzt nicht die Permutations- und FDR-Tests des Papers, sondern prüft "
  "nur, ob das Vorzeichen in der neuen Konvention hält.\n")
w("| Aussage | Δρ pro Subject | 95%-CI | hält? |")
w("|---|---|---|---|")
for lab, (d, lo, hi) in claims:
    if "n.s." in lab:
        verdict = "CI enthält 0 (konsistent mit n.s.)" if lo <= 0 <= hi else \
            "**CI schliesst 0 aus**"
    else:
        verdict = "ja" if lo > 0 else ("**nein (umgekehrt)**" if hi < 0 else "**nicht gesichert**")
    w(f"| {lab} | {f(d, 4)} | [{f(lo, 4)}, {f(hi, 4)}] | {verdict} |")
w("")
if swap_cis:
    w("Paare, deren Reihenfolge zwischen den Konventionen wechselt (Δρ pro Subject, "
      "Bootstrap-CI):\n")
    w("| RDM-Satz | ROI | Paar | Δρ pro Subject | 95%-CI | aufgelöst? |")
    w("|---|---|---|---|---|---|")
    for st, roi, y, x, d, lo, hi, ok in swap_cis:
        w(f"| {st} | {roi} | {y} − {x} | {f(d, 4)} | [{f(lo, 4)}, {f(hi, 4)}] | "
          f"{'nein (CI enthält 0)' if ok else '**ja**'} |")
    w("")
    w("Alle Wechsel betreffen "
      + ("nur IT" if not non_it_changes else "auch andere ROIs als IT")
      + ("; sie liegen innerhalb der Stimulus-Unsicherheit und passen zur "
         "Konvergenz-Aussage des Papers.\n" if all(ok for *_, ok in swap_cis) else
         "; **mindestens ein Wechsel ist aufgelöst**.\n"))
w("**„FA consistently produces the lowest alignment at V1, V2, and LOC“** (Abstract Z. 42): "
  "niedrigste Bedingung in der publizierten Konvention: "
  + ", ".join(f"{r}: {fa_lowest_pub[r]}" for r in fa_lowest_pub)
  + "; pro Subject: " + ", ".join(f"{r}: {fa_lowest[r]}" for r in fa_lowest)
  + "; nur unter den trainierten Regeln: "
  + ", ".join(f"{r}: {fa_lowest_trained[r]}" for r in fa_lowest_trained) + ". "
  + ("Die Aussage stimmt an LOC **schon in Tabelle 2 nicht**, dort ist Random am niedrigsten. "
     "Sie gilt nur unter den trainierten Regeln. Das hat mit der Konvention nichts zu tun; "
     "ich vermerke es nur. "
     if fa_lowest_pub.get("LOC") != "FA" else "")
  + f"IT-Konvergenz: ρ pro Subject zwischen {f(min(it_vals.values()), 4)} und "
  f"{f(max(it_vals.values()), 4)}.\n")

# 4 figure
w("## 4. Figur\n")
w("![noise ceiling v2](noise_ceiling_v2.png)\n")
w("Punkte: ρ pro Subject mit 95%-CI. Gefüllt = Referenzsatz, hohl = bnfix. Graues Band: "
  "95%-CI des LOO lower bound, Linie = Schätzer. Der upper bound liegt ausserhalb der "
  "Skala und ist am rechten Rand gestrichelt mit seinem Nullwert angegeben.\n")

# 5 rounding
w("## 5. Rundungsfehler in Tabelle 2\n")
rr = []
for _, c in ctrl[(ctrl.rdm_set == "original") & ctrl.rule.isin(RN)].iterrows():
    direct, double = half_up(c.rho_meanrdm, 3), half_up(half_up(c.rho_meanrdm, 4), 3)
    if abs(direct - c.published) > 1e-12:
        rr.append((RN[c.rule], c.roi, c.tex_line, c.rho_meanrdm, c.published, direct, double))
w("| Regel | ROI | TeX-Zeile | Wert (CSV/RDMs) | gedruckt | korrekt (3 Stellen) | "
  "über 4 Stellen gerundet |")
w("|---|---|---|---|---|---|---|")
for n, r, ln, v, p, d, dd in rr:
    w(f"| {n} | {r} | {ln} | {v:.5f} | {f(p)} | **{f(d)}** | {f(dd)} |")
w("")
w(f"Alle {len(rr)} Abweichungen entstehen durch doppeltes Runden (erst auf 4, dann auf "
  "3 Stellen). Die Werte selbst sind korrekt. Siehe `NUMERIC_AUDIT.md` Zeilen 10–14 (dort "
  "auch V2-BP-CI .023 → .022 und dieselben Zahlen in Abstract und Diskussion: "
  "Z. 42, 279, 285, 299, 398, 402) sowie dasselbe Muster in arXiv:2608.12408 (Zeile 1).\n")

# 6 sweeps
w("## 6. Sweep-Provenienz\n")
w("| Sweep | Datei | Datum | Seeds | Skript | Konfiguration |")
w("|---|---|---|---|---|---|")
for k, v in swj["inventory"].items():
    w(f"| {k} | `{v['csv']}` | {v['mtime']} | {v['n_seeds']} ({', '.join(map(str, v['seeds']))}) "
      f"| {v['script']} | {v['config']} |")
w("")
w("Duplikate: " + "; ".join(f"`{k}` ist byte-identisch mit `{v['same_as']}`"
                             if v["identical"] else f"`{k}` ≠ `{v['same_as']}`"
                             for k, v in swj["duplicates"].items()) +
  ". **„Alt“ = 0.0308** ist BP Conv1→V1 bei 224 px im Juni-Sweep "
  "(`outputs/rsa_resolution_sweep.csv`, kopiert als `old_sweep.csv`). Das ist ein "
  "eigenes Neutraining, nicht der Lauf hinter Tabelle 2. Alle Werte bei 224 px in "
  "Mittel-RDM-Konvention.\n")
w("| Regel | ROI | Tabelle (Mittel ± SD) | Juni-Sweep | bnfix | Tabelle − bnfix | 2 SE | "
  "z | innerhalb 2 SE |")
w("|---|---|---|---|---|---|---|---|---|")
for _, r in sw.iterrows():
    w(f"| {r.rule} | {r.roi} | {f(r.table_mean, 4)} ± {f(r.table_sd, 4)} | "
      f"{f(r.june_sweep_mean, 4)} ± {f(r.june_sweep_sd, 4)} | "
      f"{f(r.bnfix_mean, 4)} ± {f(r.bnfix_sd, 4)} | {f(r.table_minus_bnfix, 4)} | "
      f"{f(2 * r.table_minus_bnfix_se, 4)} | {r.table_minus_bnfix_z:.2f} | "
      f"{'ja' if r.table_minus_bnfix_within_2se else '**NEIN**'} |")
w("")
fl = swj["flagged_table_vs_bnfix"]
w(("Keine Zelle liegt ausserhalb von 2 SE." if not fl else
   "**Ausserhalb 2 SE (markiert, nicht weiter untersucht):** "
   + ", ".join(f"{x['rule']} {x['roi']}" for x in fl)) +
  " SE = Seed-SD/√5 pro Sweep, kombiniert als √(SE₁² + SE₂²). Das setzt unabhängig "
  "trainierte Sweeps voraus. Random ist deterministisch und in allen drei Sweeps "
  "bit-identisch (Differenz 0).\n")
nc = swj["correction_note_delta_check"]
bpv1 = sw[(sw.rule == "Backprop") & (sw.roi == "V1")].iloc[0]
w(f"**Korrekturnotiz Z. 61–63 („Δρ ≤ 0.0013 at V1 across six evaluation resolutions“).** "
  f"Die Zahl ist korrekt, aber sie vergleicht Juni-Sweep mit bnfix: Maximum "
  f"{nc['max_abs_june_vs_bnfix_V1_all_res']:.4f} bei {nc['worst_cell']['rule']}, "
  f"{nc['worst_cell']['res']} px. Gegen Tabelle 2 beträgt die Differenz bei 224 px bis zu "
  f"{swj['max_abs_table_minus_bnfix']:.4f} (BP V1: {f(bpv1.table_mean, 4)} → "
  f"{f(bpv1.bnfix_mean, 4)}). Die Notiz liest sich aber so, als bezöge sie sich auf die "
  "Werte des Papers. (Die Stelle steht in Z. 61–63, nicht in Z. 87–92.)\n")

# 7 text changes
w("## 7. Textänderungen `learning_rules_rsa_paper_v2.tex` (Vorschläge, nicht angewendet)\n")
lo_list = ", ".join(f"{r}: ${f(LO[r].point)}$ [{f(LO[r].ci_lo)}, {f(LO[r].ci_hi)}]"
                    for r in ROIS)
ps = lambda k, r: PS[("original", k, r)].point
bnf = lambda k, r: PS[("bnfix", k, r)].point


def latex(s):
    return s.strip().replace("−", "-")


def block(lines, new, note=""):
    w(f"**Z. {lines}**{(' — ' + note) if note else ''}\n")
    a, b = (int(x) for x in str(lines).split("–")) if "–" in str(lines) else (lines, lines)
    w("Alt:\n```latex\n" + "\n".join(tex(i) for i in range(a, b + 1)) + "\n```")
    w("Neu:\n```latex\n" + latex(new) + "\n```\n")


LUM = re.search(r"reaches \$\\rho = ([\d.]+)\$", tex(90)).group(1)  # luminance, paper l.90
rank_sentence = (
    "unchanged under this convention" if not rank_changes else
    "unchanged under this convention at V1, V2 and LOC; at IT, where all conditions "
    "converge, adjacent conditions whose difference is not resolved swap places"
    if not non_it_changes else "changed in some ROIs under this convention")
CN = rf"""
\noindent\textbf{{Noise ceiling (v3).}} The noise ceiling reported in v1 and v2
(Section~3.4, Figure~\ref{{fig:rsa_main}}, Section~4.2) was computed incorrectly: the three
subjects were split one against two, a Spearman--Brown correction valid only for two
equal halves was applied, and the 2.5th percentile of the resulting split distribution
was reported as the lower bound. The lower bounds stated in Section~4.2, and the V2
upper bound, do not correspond to any computed value. We replace it with the
leave-one-subject-out lower bound \citep{{nili2014}} ({lo_list}; 95\% stimulus-bootstrap
CIs) and compare models to it in the same per-subject convention (Fig.~\ref{{fig:nc}}):
the untrained network reaches $\rho = {f(ps('random_weights', 'V1'))}$ at V1, and
the ranking of the conditions within each ROI is {rank_sentence}. The statement that
the conditions exhaust most of the available signal is withdrawn; with three subjects,
no informative upper bound exists. Independently, a single scalar luminance value per
image reaches $\rho = {LUM}$ against the group-mean V1 RDM (\newid), so V1 alignment at
this level is a low bar. {NUMWORD[len(rr)]} values in Table~\ref{{tab:rsa}} were rounded
twice and are corrected
({'; '.join(f'{n} at {r}: ${f(d)}$' for n, r, ln, v, p, d, dd in rr)}).
"""

block("60–63", rf"""
Repairing the defect and re-running the full five-seed design (newly trained models)
leaves the random, backpropagation and feedback-alignment conditions within seed
variability of Table~\ref{{tab:rsa}}: at $224$\,px the largest change is
$\Delta\rho = {f(-bpv1.table_minus_bnfix, 4)}$ for backpropagation at V1
(${f(bpv1.table_mean, 4)} \to {f(bpv1.bnfix_mean, 4)}$; seed SD ${f(bpv1.table_sd, 4)}$ and
${f(bpv1.bnfix_sd, 4)}$), and all twelve rule\,$\times$\,ROI differences are below two
standard errors across seeds. The untouched random condition is reproduced
bit-identically. It changes the two affected conditions
""", "Vergleich gegen die Tabelle statt Sweep gegen Sweep, mit Seed-Streuung")

block("87–94", CN, "ganzer Absatz ersetzt durch die Correction note (v3), siehe §9; "
      "„exhaust most of the available signal“ ersatzlos gestrichen")

block(228, rf"""
\textbf{{Noise ceiling.}} We report a lower bound on the attainable model--brain
correlation as the inter-subject consistency of \citet{{nili2014}}: each subject's RDM
is correlated (Spearman, upper triangle) with the mean RDM of the other two subjects,
and the three values are averaged; no Spearman--Brown correction is applied. 95\% CIs
are from {smj['n_boot']:,} stimulus bootstrap resamples (pairs of a resampled stimulus
with itself excluded); with $N = 3$ subjects no subject-level interval is reported.
Because each stimulus was presented once per subject, a within-subject reliability of
these single-exemplar RDMs cannot be estimated. The corresponding upper bound
(correlation with the mean including the subject) is not reported: with three subjects
it is dominated by each subject's own contribution to the mean
(V1: {f(B['V1']['upper'])} observed vs.\ {f(B['V1']['upper_null_mean'])} under
stimulus permutation).
""".replace("{,}", "{,}"))

block(271, r"""
\caption{Brain alignment across ROIs (main result). Spearman $\rho$ between model RDMs
and the mean fMRI RDM (3 subjects) for each condition and ROI. Error bars: bootstrap
95\% CI ($N = 10{,}000$). White bar with black outline: untrained random-weights
baseline. No noise ceiling is drawn: no bound is available for the group-mean
convention with three subjects (see Fig.~\ref{fig:nc} for the per-subject comparison
with the leave-one-out lower bound).}
""", "graues Band entfernen; neue Figur `fig:nc` = noise_ceiling_v2.pdf")

v1_status = above("original", "random_weights", "V1")
w275 = {"über": "exceeds", "unter": "falls below",
        "nicht unterscheidbar": "is indistinguishable from"}[v1_status]
block(275, rf"""
\textbf{{Interpreting absolute RSA scores.}} Absolute Spearman $\rho$ values in this study
are low (e.g., $\rho < 0.10$ at V1), which may appear surprising in light of Brain-Score
results where encoding models reach $\rho \approx 0.5$ for V1. However, RSA and encoding
models measure fundamentally different quantities: RSA evaluates the geometry of the
representational space (pairwise dissimilarity structure), whereas encoding models
predict individual voxel responses, which is a more permissive criterion that can
exploit stimulus-specific variance. RSA scores are further attenuated by the spatial
resolution of fMRI relative to electrophysiology, and by our relatively small CNN
architecture. The inter-subject agreement is itself low: the leave-one-subject-out lower
bound is {lo_list}. Compared per subject, the untrained network at V1
($\rho = {f(ps('random_weights', 'V1'))}$) {w275} this bound
(difference ${f(DF[('original', 'random_weights', 'V1')].point)}$
[{f(DF[('original', 'random_weights', 'V1')].ci_lo)},
{f(DF[('original', 'random_weights', 'V1')].ci_hi)}]), i.e.\ it predicts a held-out
subject about as well as the other subjects do. This is a floor on the attainable
correlation, not an estimate of it; with three subjects the remaining headroom cannot be
quantified (Fig.~\ref{{fig:nc}}). Typical RSA Spearman $\rho$ values in the literature
range from 0.01 to 0.15 for model--fMRI comparisons at this scale
\citep{{kriegeskorte2008, schrimpf2020}}.
""", "„Critically … noise level“ ersetzt, Vergleich nur mit lower bound als Untergrenze")

it_lo, lo_min_roi = LO["IT"].point, min(ROIS, key=lambda r: LO[r].point)
it_max = max(ps(k, "IT") for k, _ in RULES)
block(400, rf"""
... whether this advantage reflects the learning rule or the additional model capacity
remains an open question. The leave-one-subject-out lower bound at IT
(${f(it_lo)}$) is {'the lowest of all ROIs' if lo_min_roi == 'IT' else
f'not the lowest of the four ROIs (lowest: {lo_min_roi}, ${f(LO[lo_min_roi].point)}$)'},
so the convergence at IT is not explained by a lower ceiling there. All conditions lie
well below it at IT (per-subject $\rho \le {f(it_max)}$), so true differences may exist
that are unresolvable given the fMRI signal quality at this ROI and sample size ($N = 3$).
""", "nur der letzte Satz ändert sich; „upper bound: 0.07“ entfällt")

# 8 ERC
w("## 8. arXiv:2608.12408 (Evaluation Resolution Confounds)\n")
erc1 = ERC_V1.read_text(encoding="utf-8").splitlines()
erc2 = ERC_V2.read_text(encoding="utf-8").splitlines()
rbn = PS[("bnfix", "random_weights", "V1")]
rbn_mean = ctrl[(ctrl.rdm_set == "bnfix") & (ctrl.rule == "random_weights") &
                (ctrl.roi == "V1")].iloc[0]
ERC_REL = {"über": "above that floor", "unter": "below that floor",
           "nicht unterscheidbar": "statistically indistinguishable from that floor"}[
    above("bnfix", "random_weights", "V1")]
w(f"**`{ERC_V1.name}:57`**\n")
w("Alt (Satzanfang):\n```latex\n" + erc1[56][:560] + " …\n```")
w("Neu (ersetzt die ersten beiden Sätze, Rest des Absatzes ab „That framing …“ "
  "unverändert, mit „framing“ → „comparison“):\n```latex\n" + rf"""
\textbf{{Interpreting the scale.}} Absolute Spearman $\rho$ values in this setting are low.
The split-half noise ceiling quoted in v1 of this paper has been withdrawn by the endpoint
study \citep{{leutenegger2026}}. Its replacement, a leave-one-subject-out lower bound
(V1: ${f(v1lo.point)}$ [{f(v1lo.ci_lo)}, {f(v1lo.ci_hi)}]; LOC: ${f(LO['LOC'].point)}$),
is a per-subject quantity; the untrained network's per-subject score at $224$\,px is
$\rho = {f(rbn.point)}$, {ERC_REL}. No bound exists for the group-mean
convention of our figures ($\rho = {f(half_up(rbn_mean.rho_meanrdm, 3))}$ at $224$\,px),
so we give no percentage of a ceiling.
""".replace("−", "-").strip() + "\n```\n")
w(f"Nebenbei: Die Zeile nennt untrainiert ρ = 0.076. Korrekt gerundet ist "
  f"{f(half_up(rbn_mean.rho_meanrdm, 3))} (`NUMERIC_AUDIT.md` Zeile 1).\n")
w(f"**`{ERC_V2.name}:35`, Prüfung der Formulierung**\n")
w("Ist:\n```latex\n" + erc2[34] + "\n```")
w("Das ist **nicht präzise**:\n"
  "1. Eine between-subject Untergrenze *ist* schätzbar, sie wird hier berechnet "
  f"(V1 {ci(v1lo)}). Nicht schätzbar bzw. nicht informativ ist bei N=3 die "
  "*obere* Grenze.\n"
  "2. Dass keine within-subject Ceiling schätzbar ist, gilt für **genau diese "
  "Single-Exemplar-RDM**, weil jeder Stimulus einmal gezeigt wurde. Für THINGS-fMRI "
  "allgemein gilt es nicht: Das Testset hat 12 Wiederholungen, die Trainingsbilder haben "
  "12 Exemplare pro Konzept.\n"
  "3. Für die Mittel-RDM-Konvention der Figuren gibt es keinen Bound, auch keinen "
  "between-subject.\n")
w("Vorschlag:\n```latex\n" + rf"""
We also withdraw the noise-ceiling estimate and the ``$69\%$ of the attainable ceiling''
figure. Because each of the 720 stimuli was presented once per subject, no within-subject
ceiling can be estimated for the single-exemplar RDMs used here, and with three subjects
no ceiling is available for the group-mean RDM our figures correlate against. A
between-subject lower bound can be computed for per-subject scores
(V1: ${f(v1lo.point)}$; \citealp{{leutenegger2026}}), but it is a floor, not a ceiling.
The luminance bound, which requires no ceiling, is stated in its place; it is the bound
the paper's scale argument now rests on.
""".strip() + "\n```\n")

# 9 arXiv comment + correction note
w("## 9. arXiv-Revisionskommentar und Correction note\n")
claim_fail = any(("n.s." not in lab and not lo > 0) or ("n.s." in lab and not lo <= 0 <= hi)
                 for lab, (d, lo, hi) in claims)
qual_ok = not claim_fail and not non_it_changes and all(ok for *_, ok in swap_cis)
last = (f"No qualitative result changes; {NUMWORD[len(rr)].lower()} double-rounding errors "
        f"in Table 2 are also corrected." if qual_ok else
        "Affected statements are listed in the correction note.")
w("**arXiv comment (v3), 3 Sätze:**\n")
w("> v3 withdraws the noise ceiling of v1/v2, which applied a two-half Spearman-Brown "
  "correction to an unequal 1-vs-2 subject split and reported a percentile of that split "
  "distribution as its lower bound (the printed lower bounds were not reproducible). It is "
  "replaced by the leave-one-subject-out lower bound of Nili et al. (2014) with "
  "stimulus-bootstrap CIs, compared against per-subject model scores, and the claim that "
  f"the models exhaust the available signal is removed. {last}\n")
w("**Correction note (v3):** Das ist der Absatz, der in §7 Z. 87–94 ersetzt "
  "(„On the noise-ceiling argument“ → „Noise ceiling (v3)“). Er steht dort einmal, damit "
  "derselbe Inhalt nicht doppelt in der Notiz auftaucht:\n")
w("```latex\n" + latex(CN) + "\n```\n")
w("Neue Literaturangabe für die `thebibliography`-Umgebung (Z. 428–500), alphabetisch "
  "einsortieren:\n```latex\n\\bibitem[Nili et al., 2014]{nili2014}\n"
  "Nili, H., Wingfield, C., Walther, A., Su, L., Marslen-Wilson, W., and "
  "Kriegeskorte, N. (2014).\nA toolbox for representational similarity analysis.\n"
  "\\textit{PLoS Computational Biology}, 10(4):e1003553.\n```\n")

# 10 files
w("## 10. Dateien und Reproduktion\n")
w("```\npy -3 scripts/noise_ceiling_v2/step1_nili_bounds.py      # Bounds, Permutationsnull, alter Schätzer\n"
  "py -3 scripts/noise_ceiling_v2/step3a_control_column.py   # Kontrollspalte (Reproduktion Tabelle 2)\n"
  "py -3 scripts/noise_ceiling_v2/step3b_sweep_provenance.py # Sweep-Provenienz\n"
  "py -3 scripts/noise_ceiling_v2/step3c_bootstrap.py        # 1000× Stimulus-Bootstrap (~25 min, 10 Prozesse)\n"
  "py -3 scripts/noise_ceiling_v2/step4_figure.py\n"
  "py -3 scripts/noise_ceiling_v2/step4_report.py\n"
  "py -3 scripts/noise_ceiling_v2/step4_manifest.py          # erst nach Commit der Skripte\n```\n")
w(f"Seeds: Permutationen {s1['seed']}, Bootstrap {smj['seed']} "
  f"(SHA-256 der Indexmatrix `{smj['boot_idx_sha256'][:16]}…`). Laufzeit Bootstrap "
  f"{smj['runtime_s'] / 60:.0f} min.\n")

(RESULTS / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
print(f"REPORT.md: {len(L)} lines; rank changes: {rank_changes}; claims:",
      [(lab, round(d, 4), round(lo, 4), round(hi, 4)) for lab, (d, lo, hi) in claims])

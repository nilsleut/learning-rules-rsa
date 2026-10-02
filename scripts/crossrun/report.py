"""CROSSRUN_REPORT.md from the stored outputs (no number typed by hand).

Claim criterion (as before): a claim HOLDS in a variant if it has the same sign and the
same CI significance (95% CI excludes 0 or not) as in the reference analysis (`all`, the
published pair set). HOLDS but the point estimate moves by more than 25 % (relative to
`all`) -> CHANGES SIZE. Otherwise -> FLIPS.
"""
import json
import sys

import numpy as np
import pandas as pd

from common_cr import RESULTS, REPO, ROIS, SUBJECTS

sys.stdout.reconfigure(encoding="utf-8")
R = RESULTS
j1 = json.loads((R / "step1.json").read_text())
lagS = pd.read_csv(R / "step1_lag_summary.csv")
rpt = pd.read_csv(R / "step1_runpair_test.csv")
lum1 = pd.read_csv(R / "step1_luminance.csv")
agr = pd.read_csv(R / "step1_agreement.csv")
j2 = json.loads((R / "step2.json").read_text())
draws = pd.read_csv(R / "step2_draws.csv")
summ = pd.read_csv(R / "step2_summary.csv").set_index("key")
eff = pd.read_csv(R / "step2_effects.csv")
je = json.loads((R / "step2_effects.json").read_text())
rz5 = pd.read_csv(REPO / "results/runz/step5_effects.csv")
pt, bt = draws[draws.boot == -1].iloc[0], draws[draws.boot >= 0]

RN = {"random_weights": "Random", "backprop": "BP", "feedback_alignment": "FA",
      "predictive_coding": "PC", "stdp": "STDP"}
VAR = {"all": "alle Paare (publiziert)", "cr": "Cross-Run (primär)", "crb": "Cross-Run blockbereinigt"}
CONVS = {"all": ["meanrdm", "persub"], "cr": ["meanrdm", "persub"], "crb": ["persub"]}
SETNAME = {"original": "Referenzsatz", "bnfix": "bnfix"}


def f(x, d=4):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}".replace("-", "−")


def diff_ci(conv, var, st, a, b, roi):
    ca, cb = f"{conv}|{var}|{st}|{a}|{roi}", f"{conv}|{var}|{st}|{b}|{roi}"
    d = bt[ca] - bt[cb]
    return float(pt[ca] - pt[cb]), float(d.quantile(0.025)), float(d.quantile(0.975))


def sig(lo, hi):
    return lo > 0 or hi < 0


def verdict(ref, new):
    (d0, l0, h0), (d1, l1, h1) = ref, new
    same = np.sign(d0) == np.sign(d1) and sig(l0, h0) == sig(l1, h1)
    if not same:
        return "**kippt**"
    if d0 != 0 and abs(d1 - d0) / abs(d0) > 0.25:
        return "ändert Grösse"
    return "hält"


# ── Paper-1 claims (stimulus bootstrap, rule differences) ──────────────────────
claims = []
for roi in ("V1", "V2"):
    claims.append((f"Random > BP an {roi}", "random_weights", "backprop", roi, "P1"))
claims.append(("BP > Random an LOC", "backprop", "random_weights", "LOC", "P1"))
for o in ("feedback_alignment", "predictive_coding", "stdp"):
    claims.append((f"BP vs {RN[o]} an LOC n.s.", "backprop", o, "LOC", "P1"))
for roi in ("V1", "V2"):
    for o in ("random_weights", "backprop", "predictive_coding", "stdp"):
        claims.append((f"FA < {RN[o]} an {roi}", o, "feedback_alignment", roi, "P1"))
rk = list(RN)
for i, a in enumerate(rk):
    for b in rk[i + 1:]:
        claims.append((f"IT konvergiert: {RN[a]} vs {RN[b]} n.s.", a, b, "IT", "P1"))


def claim_rows(st):
    rows = []
    for lab, a, b, roi, _ in claims:
        for conv in ("meanrdm", "persub"):
            ref = diff_ci(conv, "all", st, a, b, roi)
            row = {"claim": lab, "conv": conv, "all": ref}
            for var in ("cr", "crb"):
                if conv in CONVS[var]:
                    new = diff_ci(conv, var, st, a, b, roi)
                    row[var] = new
                    row[f"v_{var}"] = verdict(ref, new)
            rows.append(row)
    return rows


L = []
w = L.append
w("# Cross-Run-Analyse (Run-Konfundierung) — arXiv:2604.16875 und arXiv:2608.12408\n")
w("Erzeugt von `scripts/crossrun/report.py` aus `results/crossrun/*`. Paper-Dateien wurden "
  "nicht editiert. Die alten RDMs (`outputs_720`, `--zscore all`) und `rdms_runz/` bleiben "
  "unverändert.\n")
w("**Varianten.** *alle Paare* = publizierte Analyse. *Cross-Run (primär)* = nur Paare, die in "
  f"**keinem** Subject im selben Run liegen: {j1['n_pairs_P']:,} von {j1['n_pairs_total']:,} Paaren "
  f"({j1['frac_pairs_P']:.1%}). *Cross-Run blockbereinigt* = dieselben Paare. Pro Subject wird von "
  "Fach-RDM und Modell- bzw. Luminanz-RDM der Mittelwert jedes Run-Paar-Blocks abgezogen, "
  "berechnet auf der Schnittmenge, mit den Runs dieses Subjects. Das geschieht nur in der "
  "Pro-Subject-Konvention. **Die Mittel-RDM-Konvention ist für die Blockbereinigung nicht "
  "definiert:** Die drei Subjects haben verschiedene Blockstrukturen (sub-03 eine andere "
  "Reihenfolge als sub-01/02), und keine einzelne Bereinigung passt auf ihr Mittel.\n")
w("**Kriterium.** Eine Aussage *hält*, wenn sie in der Variante dasselbe Vorzeichen und dieselbe "
  "CI-Signifikanz hat (95 %-CI schliesst 0 aus oder nicht) wie auf *allen Paaren*. Hält sie, "
  "aber der Punktschätzer ändert sich um mehr als 25 %, steht *ändert Grösse*. Sonst *kippt*. "
  "CIs: Stimulus-Bootstrap (1000 gemeinsame Resamples, dieselben wie in noise_ceiling_v2) für "
  "Paper 1, Seed-CI (t, n = 5) für Paper 2.\n")

# summary
w("## Kurzfassung\n")
w(f"- **Diagnostik (Schritt 1):** Die Run-Paar-Struktur ist bei sub-01/02 nicht geteilt (exaktes p "
  f"über 10! Permutationen: "
  + ", ".join(f"{r.roi} {r.p_exact_one_sided:.2f}" for _, r in rpt.iterrows())
  + f"). Primärvariante bleibt deshalb **{VAR[{'cross_run': 'cr', 'cross_run_block_cleaned': 'crb'}[j1['primary_variant']]]}**. Die Blockbereinigung "
  f"senkt den Luminanz-Fit an V1 bei sub-01/02 um "
  + ", ".join(f"{-v:.0%}" for v in j1["luminance_V1_frac_change_sub01_sub02"].values())
  + " (STOPP-Schwelle 30 %, nicht erreicht).")
lo = {v: {r: summ.loc[f"lower|{v}|{r}"] for r in ROIS} for v in VAR}
w("- **Lower bound:** "
  + "; ".join(f"{r}: {f(lo['all'][r].point, 3)} → {f(lo['cr'][r].point, 3)} (Cross-Run) / "
              f"{f(lo['crb'][r].point, 3)} (blockbereinigt)" for r in ROIS)
  + ". Der Reihenfolge-Anteil sass im Bound, vor allem an LOC und IT.")
for st in ("original",):
    cr = claim_rows(st)
    nflip = {v: sum(1 for r in cr if r.get(f"v_{v}") == "**kippt**") for v in ("cr", "crb")}
    nsize = {v: sum(1 for r in cr if r.get(f"v_{v}") == "ändert Grösse") for v in ("cr", "crb")}
    ntot = {v: sum(1 for r in cr if f"v_{v}" in r) for v in ("cr", "crb")}
w(f"- **Paper 1 (Referenzsatz):** Von {ntot['cr']} geprüften Aussage × Konvention-Zellen kippen "
  f"unter Cross-Run {nflip['cr']}, {nsize['cr']} ändern die Grösse. Blockbereinigt "
  f"({ntot['crb']} Zellen, nur pro Subject): {nflip['crb']} kippen, {nsize['crb']} ändern die "
  "Grösse. Die kippenden Zellen betreffen: "
  + (", ".join(sorted({r["claim"] for r in cr for v in ("cr", "crb")
                       if r.get(f"v_{v}") == "**kippt**"})) or "keine").rstrip(".")
  + ". Art der Wechsel in §3.")
e224 = eff[(eff.res == 224) & (eff.effect.str.startswith("E1"))].set_index(["variant", "convention"])
w(f"- **Paper 2:** Random − BP an V1 bei 224 px (Mittel-RDM) "
  f"{f(e224.loc[('all', 'meanrdm')]['mean'], 3)} → {f(e224.loc[('cr', 'meanrdm')]['mean'], 3)} "
  f"(Cross-Run). Pro Subject {f(e224.loc[('all', 'persub')]['mean'], 3)} → "
  f"{f(e224.loc[('cr', 'persub')]['mean'], 3)} / {f(e224.loc[('crb', 'persub')]['mean'], 3)}. "
  f"Luminanz an V1 (Mittel-RDM) {f(je['luminance_vs_V1']['all|meanrdm'], 3)} → "
  f"{f(je['luminance_vs_V1']['cr|meanrdm'], 3)}. Details in §4.\n")

# 1 diagnostics
w("## 1. Diagnostik (vor allen Modellzahlen)\n")
w("**a) Lag innerhalb eines Runs** (alte RDMs, alle Paare im selben Run; diese Paare sind in der "
  "Cross-Run-Analyse ausgeschlossen):\n")
w("| ROI | Subject | Spearman(Dissimilarität, Lag) | Mittel Lag 1 | Mittel Lag ≥ 10 | Mittel Cross-Run |")
w("|---|---|---|---|---|---|")
for _, r in lagS.iterrows():
    w(f"| {r.roi} | {r.subject} | {f(r.spearman_diss_vs_lag, 3)} | {f(r.mean_lag1)} | "
      f"{f(r.mean_lag_ge10)} | {f(r.mean_cross_run_P)} |")
w("\nIm selben Run nimmt die Dissimilarität mit dem Abstand ab. Benachbarte Trials sind "
  "unähnlicher als entfernte, an LOC und IT sogar über 1 (also antikorreliert). Das passt zu "
  "negativ gekoppelten Einzeltrial-Schätzungen bei schneller Trialfolge. Paare aus "
  "verschiedenen Runs liegen nahe 1. Das ist nur berichtet, es geht nicht in die Analyse ein.\n")
w("**b) Run-Paar-Struktur** (Schnittmenge P, 10×10-Mittel pro Run-Paar, 45 Werte). 01–02 haben "
  "identische Run-Labels. Exakte Null über alle 10! = 3 628 800 Umbenennungen der Runs von "
  "sub-02:\n")
w("| ROI | Spearman 01–02 | Null Mittel ± SD | p (einseitig, exakt) |")
w("|---|---|---|---|")
for _, r in rpt.iterrows():
    w(f"| {r.roi} | {f(r.spearman_01_02_runpair_means, 3)} | {f(r.null_mean, 3)} ± "
      f"{f(r.null_sd, 3)} | {r.p_exact_one_sided:.3f} |")
w(f"\nIn keiner ROI signifikant. Cross-Run allein reicht also; die Blockbereinigung bleibt "
  "Robustheitscheck. Die Power ist mit 45 Run-Paaren begrenzt; die Blockbereinigung wird "
  "deshalb trotzdem für alle Aussagen mitgerechnet.\n")
w("**c) Luminanz-Fit pro Subject** (*Diagnostik mit Effektcharakter*, siehe §6):\n")
w("| ROI | Subject | alle Paare | Cross-Run | blockbereinigt | Änderung Cross-Run → bereinigt |")
w("|---|---|---|---|---|---|")
for _, r in lum1.iterrows():
    ch = f"{r.frac_change_cr_to_crb:+.0%}" if abs(r.cross_run) > 0.005 else "– (kein Fit)"
    w(f"| {r.roi} | {r.subject} | {f(r.all_pairs)} | {f(r.cross_run)} | "
      f"{f(r.cross_run_block_cleaned)} | {ch} |")
s3 = lum1[(lum1.roi == "V1") & (lum1.subject == "sub-03")].iloc[0]
w(f"\nBei sub-03 (Luminanz–Run-Kopplung p = 0.005, siehe `results/runz`) entfällt an V1 "
  f"{-(s3.cross_run_block_cleaned - s3.cross_run) / s3.cross_run:.0%} des Luminanz-Fits auf die "
  "Run-Paar-Mittel, also praktisch nichts. STOPP-Kriterium (Abnahme > 30 % bei sub-01/02 an "
  "V1): nicht erfüllt.\n")
w("**d) Übereinstimmung zwischen Subjects:**\n")
w("| ROI | Paar | alle Paare | Cross-Run | blockbereinigt |")
w("|---|---|---|---|---|")
for _, r in agr.iterrows():
    w(f"| {r.roi} | {r.pair} | {f(r.all_pairs)} | {f(r.cross_run)} | {f(r.cross_run_block_cleaned)} |")
a12 = agr[agr.pair == "01-02"].set_index("roi")
above = all(min(a12.loc[r, c] for c in ("cross_run", "cross_run_block_cleaned")) >
            agr[(agr.roi == r) & (agr.pair != "01-02")][["cross_run", "cross_run_block_cleaned"]].values.max()
            for r in ROIS)
w(f"\n01–02 liegt in allen Varianten {'über' if above else '**nicht überall über**'} "
  "01–03/02–03. Änderung von allen Paaren auf Cross-Run für 01–02: "
  + ", ".join(f"{r} {a12.loc[r, 'cross_run'] / a12.loc[r, 'all_pairs'] - 1:+.0%}" for r in ROIS)
  + ". Das ist der Reihenfolge-Anteil aus SUB03_CHECK.\n")

# 2 bounds
w("## 2. Noise-Ceiling-Bounds\n")
w(f"Bootstrap: {j2['n_boot']} Resamples, dieselbe Indexmatrix wie noise_ceiling_v2. Die Variante "
  f"*alle Paare* reproduziert v2 exakt (max. Abweichung "
  f"{j2['validation_max_dev_vs_noise_ceiling_v2']:.0e}). Cross-Run-Paare pro Resample: Median "
  f"{j2['n_pairs_cr_boot_median']:,}. Permutations-Null: {j2['n_perm']}× Stimuluslabels pro "
  f"Subject unabhängig permutiert, Seed {j2['perm_seed']}.\n")
w("| ROI | Variante | lower [95%-CI] | lower H0 (Mittel ± SD) | upper | upper H0 | Status upper |")
w("|---|---|---|---|---|---|---|")
v2b = json.loads((REPO / "results/noise_ceiling_v2/step1_bounds.json").read_text())["rois"]
for roi in ROIS:
    for v in VAR:
        l_, u_ = summ.loc[f"lower|{v}|{roi}"], summ.loc[f"upper|{v}|{roi}"]
        if v == "all":
            nl = f"{f(v2b[roi]['lower_null_mean'])} ± {f(v2b[roi]['lower_null_sd'])}"
            nu = f"{f(v2b[roi]['upper_null_mean'])}"
        else:
            pn = j2["perm_null"]
            nl = f"{f(pn[f'lower|{v}|{roi}']['mean'])} ± {f(pn[f'lower|{v}|{roi}']['sd'])}"
            nu = f(pn[f"upper|{v}|{roi}"]["mean"])
        w(f"| {roi} | {VAR[v]} | {f(l_.point)} [{f(l_.ci_lo)}, {f(l_.ci_hi)}] | {nl} | "
          f"{f(u_.point)} | {nu} | bei N=3 nicht informativ |")
w("")

# 3 models
w("## 3. Modell-RSA und Paper-1-Aussagen\n")
for st in ("original", "bnfix"):
    w(f"### Modell − lower bound, pro Subject ({SETNAME[st]})\n")
    w("| Regel | ROI | " + " | ".join(f"ρ {VAR[v]}" for v in VAR) + " | "
      + " | ".join(f"Δ {VAR[v]} [CI]" for v in VAR) + " |")
    w("|---|---|" + "---|" * (2 * len(VAR)))
    for k, n in RN.items():
        for roi in ROIS:
            cells = [f(summ.loc[f"persub|{v}|{st}|{k}|{roi}"].point) for v in VAR]
            ds = []
            for v in VAR:
                d = summ.loc[f"diff|{v}|{st}|{k}|{roi}"]
                tag = "über" if d.ci_lo > 0 else ("unter" if d.ci_hi < 0 else "n.u.")
                ds.append(f"{f(d.point)} [{f(d.ci_lo)}, {f(d.ci_hi)}] {tag}")
            w(f"| {n} | {roi} | " + " | ".join(cells) + " | " + " | ".join(ds) + " |")
    w("")
    w(f"### Mittel-RDM-Konvention ({SETNAME[st]})\n")
    w("| Regel | ROI | alle Paare | Cross-Run |")
    w("|---|---|---|---|")
    for k, n in RN.items():
        for roi in ROIS:
            w(f"| {n} | {roi} | {f(summ.loc[f'meanrdm|all|{st}|{k}|{roi}'].point)} | "
              f"{f(summ.loc[f'meanrdm|cr|{st}|{k}|{roi}'].point)} |")
    w("")

w("### Rangfolge der Regeln pro ROI (Kendall τ gegen alle Paare)\n")
from scipy.stats import kendalltau  # noqa: E402
w("| Satz | ROI | Konvention | Variante | Rangfolge | τ vs alle Paare |")
w("|---|---|---|---|---|---|")
for st in SETNAME:
    for roi in ROIS:
        for conv in ("meanrdm", "persub"):
            base = [summ.loc[f"{conv}|all|{st}|{k}|{roi}"].point for k in RN]
            for v in VAR:
                if conv not in CONVS[v]:
                    continue
                vals = [summ.loc[f"{conv}|{v}|{st}|{k}|{roi}"].point for k in RN]
                order = " > ".join(RN[k] for _, k in sorted(zip(vals, RN), reverse=True))
                tau = kendalltau(base, vals)[0]
                w(f"| {SETNAME[st]} | {roi} | {conv} | {VAR[v]} | {order} | {tau:.2f} |")
w("")

for st in SETNAME:
    w(f"### Paper-1-Aussagen × Variante ({SETNAME[st]})\n")
    w("Δρ = erste − zweite Bedingung, 95%-CI aus dem Stimulus-Bootstrap auf denselben Resamples.\n")
    w("| Aussage | Konvention | alle Paare | Cross-Run | Urteil | blockbereinigt | Urteil |")
    w("|---|---|---|---|---|---|---|")
    for r in claim_rows(st):
        cell = lambda t: f"{f(t[0])} [{f(t[1])}, {f(t[2])}]"
        crb = (cell(r["crb"]), r["v_crb"]) if "crb" in r else ("nicht definiert", "–")
        w(f"| {r['claim']} | {r['conv']} | {cell(r['all'])} | {cell(r['cr'])} | {r['v_cr']} | "
          f"{crb[0]} | {crb[1]} |")
    w("")
w("Hinweis: Die Signifikanzangaben des Papers stammen aus Permutationstests mit FDR. Hier wird "
  "nur geprüft, ob Vorzeichen und Stimulus-Bootstrap-Signifikanz zwischen den Varianten "
  "gleich bleiben. Wo das CI schon auf *allen Paaren* von der Paper-Aussage abweicht (z. B. "
  "„n.s.“, aber CI schliesst 0 aus), ist das ein Befund zur Aussage selbst und unabhängig von "
  "der Variante.\n")
# what the "kippt" cells actually are (computed, not typed)
flips = [(SETNAME[st], r["claim"], r["conv"], v, r["all"], r[v]) for st in SETNAME
         for r in claim_rows(st) for v in ("cr", "crb") if r.get(f"v_{v}") == "**kippt**"]
sign_only = [x for x in flips if not sig(*x[4][1:]) and not sig(*x[5][1:])]
sig_change = [x for x in flips if x not in sign_only]
w(f"**Art der {len(flips)} „kippt“-Zellen (beide Modellsätze).** {len(sign_only)} davon sind "
  "reine Vorzeichenwechsel eines Punktschätzers nahe 0 bei einer „n.s.“-Aussage: Das CI "
  "schliesst 0 in *beiden* Varianten ein (grösster Betrag "
  f"{f(max(max(abs(x[4][0]), abs(x[5][0])) for x in sign_only)) if sign_only else '–'}). "
  "Die Aussage „n.s.“ bleibt dort inhaltlich bestehen; das Kriterium wertet sie wie vorab "
  "festgelegt trotzdem als „kippt“. Dasselbe gilt für „ändert Grösse“ bei „n.s.“-Aussagen: "
  "Eine relative Änderung über 25 % eines Werts nahe 0 sagt wenig. "
  + (f"Echte Signifikanzwechsel ({len(sig_change)}): "
     + "; ".join(f"{s}, {c} ({cv}, {VAR[v]}): {f(a[0])} [{f(a[1])}, {f(a[2])}] → "
                 f"{f(n[0])} [{f(n[1])}, {f(n[2])}]" for s, c, cv, v, a, n in sig_change)
     + "." if sig_change else "Echte Signifikanzwechsel: keine.") + "\n")

# 4 paper 2
w("## 4. arXiv:2608.12408: Kerneffekte über die Auflösungen (bnfix, Seed-CI)\n")
w(f"Validierung: *alle Paare*, Mittel-RDM reproduziert `bnfix_sweep.csv` (max. Abweichung "
  f"{je['validation_max_abs_diff_vs_bnfix_sweep']:.0e}, {je['n_validated']} Werte).\n")
for e in ("E1 Random-BP V1", "E2 BP-Random LOC"):
    w(f"**{e}**\n")
    w("| px | Konvention | alle Paare | Cross-Run | Urteil | blockbereinigt | Urteil |")
    w("|---|---|---|---|---|---|---|")
    g = eff[eff.effect == e]
    for px in sorted(g.res.unique()):
        for conv in ("meanrdm", "persub"):
            x = g[(g.res == px) & (g.convention == conv)].set_index("variant")
            ref = tuple(x.loc["all", ["mean", "ci_lo", "ci_hi"]])
            c = lambda v: f"{f(x.loc[v, 'mean'])} [{f(x.loc[v, 'ci_lo'])}, {f(x.loc[v, 'ci_hi'])}] ({x.loc[v, 'n_seeds_positive']}/5)"
            crv = verdict(ref, tuple(x.loc["cr", ["mean", "ci_lo", "ci_hi"]]))
            if "crb" in x.index:
                crb, crbv = c("crb"), verdict(ref, tuple(x.loc["crb", ["mean", "ci_lo", "ci_hi"]]))
            else:
                crb, crbv = "nicht definiert", "–"
            w(f"| {px} | {conv} | {c('all')} | {c('cr')} | {crv} | {crb} | {crbv} |")
    w("")
lu, rnd = je["luminance_vs_V1"], je["random_224_V1"]
w("**E3 Luminanz-Schranke an V1** (Paper: Luminanz 0.074 ≈ untrainiertes Netz 0.075). "
  "Kriterium „hält“: Die Luminanz liegt innerhalb von 10 % des untrainierten Netzes bei 224 px, "
  "wie auf allen Paaren.\n")
w("| Variante | Konvention | Luminanz | untrainiert 224 px | Verhältnis | Urteil |")
w("|---|---|---|---|---|---|")
for k in lu:
    ratio = lu[k] / rnd[k]
    w(f"| {VAR[k.split('|')[0]]} | {k.split('|')[1]} | {f(lu[k])} | {f(rnd[k])} | {ratio:.2f} | "
      f"{'hält' if abs(ratio - 1) <= 0.10 else '**kippt**'} |")
w("")

# 5 nothing changes
w("## 5. Wo sich nichts ändert\n")
allc = claim_rows("original")
held = [r for r in allc if r["v_cr"] == "hält" and r.get("v_crb", "hält") == "hält"]
notheld = [r for r in allc if r not in held]
w(f"**Paper 1 (Referenzsatz):** {len(held)} von {len(allc)} Aussage × Konvention-Zellen halten in "
  "beiden robusten Varianten ohne Grössenänderung über 25 %. "
  + ("Alle anderen: " + "; ".join(f"{r['claim']} ({r['conv']}: Cross-Run {r['v_cr']}, "
                                   f"blockbereinigt {r.get('v_crb', '–')})" for r in notheld) + "."
     if notheld else "Keine Ausnahme."))
pv = []
for e in ("E1 Random-BP V1", "E2 BP-Random LOC"):
    g = eff[eff.effect == e]
    bad = []
    for (px, conv), x in g.groupby(["res", "convention"]):
        x = x.set_index("variant")
        ref = tuple(x.loc["all", ["mean", "ci_lo", "ci_hi"]])
        for v in x.index:
            if v != "all":
                vv = verdict(ref, tuple(x.loc[v, ["mean", "ci_lo", "ci_hi"]]))
                if vv != "hält":
                    bad.append(f"{px} px {conv} {VAR[v]}: {vv}")
    pv.append(f"{e}: " + ("hält bei allen 6 Auflösungen, in allen Varianten und Konventionen"
                          if not bad else "Ausnahmen: " + "; ".join(bad)))
lum_ok = all(abs(lu[k] / rnd[k] - 1) <= 0.10 for k in lu)
pv.append("E3 Luminanz-Schranke: " + ("hält in allen Varianten" if lum_ok else "siehe §4"))
w("\n**Paper 2:** " + ". ".join(pv) + ".\n")
w("**Was sich ändert:** der lower bound. "
  + ", ".join(f"{r} {f(lo['all'][r].point, 3)} → {f(lo['cr'][r].point, 3)} "
              f"({lo['cr'][r].point / lo['all'][r].point - 1:+.0%})" for r in ROIS)
  + ". Dadurch verschiebt sich die Lage der Modelle relativ zum Bound (§3, Spalten Δ). Die "
  "Modellwerte selbst ändern sich kaum.\n")

# 6 deviations
w("## 6. Abweichungen von der Vorab-Planung\n")
w("1. **runz → Cross-Run.** Vorab festgelegt war die Voxel-z-Normierung pro Run als "
  "Primärvariante. Sie hat beide vorab definierten Kriterien verfehlt (K1: Δ = +1/81 in allen "
  "Zellen, das −1/(n−1)-Artefakt des Demeaning; K2: 01–02 auf allen Paaren ≠ Cross-Run, siehe "
  "`results/runz/RUNZ_REPORT.md`). Primärvariante ist jetzt Cross-Run auf den alten RDMs, die "
  "Blockbereinigung ist Robustheitscheck. **Die runz-Effektzahlen (Schritt 5, z. B. V1-Gap "
  f"224 px {f(rz5[(rz5.res==224)&(rz5.roi=='V1')&(rz5.rdms=='runz')&(rz5.convention=='meanrdm')]['mean'].iloc[0], 3)} "
  "statt 0.044) wurden gesehen, bevor diese Entscheidung fiel.** Die Entscheidung ist mit "
  "den Kriterien aus Schritt 2 begründet, nicht mit den Effektzahlen. Unabhängig ist sie "
  "dennoch nicht.")
w("2. **Schritt 1c ist Diagnostik mit Effektcharakter.** Der Luminanz-Fit ist zugleich eine "
  "Kernaussage von arXiv:2608.12408 (E3) und STOPP-Kriterium. Er wurde vor allen "
  "Modellzahlen gerechnet, gibt aber schon Auskunft über ein Ergebnis.")
w("3. **Paarmenge.** Statt der Cross-Run-Paare pro Subject-Paar wird für alle Analysen die "
  "Schnittmenge über alle drei Subjects verwendet (Korrektur zum Auftrag).")
w("4. **Blockbereinigung nur pro Subject.** Für die Mittel-RDM-Konvention ist sie nicht "
  "definiert (siehe oben).")
w("5. **Run-Offset aus Test-Trials** wurde nicht umgesetzt (Entscheidung: der Schätzfehler "
  "würde positive Korrelation innerhalb des Runs erzeugen).\n")

w("## 7. Dateien\n")
w("```\npy -3 scripts/crossrun/step1_diagnostics.py   # Lag, Run-Paare (10! exakt), Luminanz, Übereinstimmung\n"
  "py -3 scripts/crossrun/step2_bootstrap.py draws --start -1 --stop 500    # Bootstrap Teil 1 (~1 h, 10 Prozesse)\n"
  "py -3 scripts/crossrun/step2_bootstrap.py draws --start 500 --stop 1000  # Teil 2 (~1 h); wiederaufnehmbar\n"
  "py -3 scripts/crossrun/step2_bootstrap.py perm       # Permutations-Null der Bounds (~7 min)\n"
  "py -3 scripts/crossrun/step2_bootstrap.py finalize   # Zusammenfassung + Validierung gegen v2\n"
  "py -3 scripts/crossrun/step2_effects.py       # Kerneffekte 2608.12408 über Auflösungen\n"
  "py -3 scripts/crossrun/report.py\n```\n")
(R / "CROSSRUN_REPORT.md").write_text("\n".join(L), encoding="utf-8")
print("ok", len(L))

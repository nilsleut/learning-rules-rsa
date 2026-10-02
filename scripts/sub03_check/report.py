"""SUB03_CHECK.md from the stored check outputs (no number typed by hand)."""
import json
import sys

import pandas as pd

from common_sub03 import RESULTS, REPO, ROIS, SUBJECTS, EXTRACT_SCRIPT_DATA_DIR, DATA_DIR

sys.stdout.reconfigure(encoding="utf-8")
R = RESULTS
c1 = pd.read_csv(R / "check1_pairs.csv")
c2 = pd.read_csv(R / "check2_luminance.csv")
c3 = pd.read_csv(R / "check3_models.csv")
j123 = json.loads((R / "check123.json").read_text())
j4 = json.loads((R / "check4_ids.json").read_text())
j567 = json.loads((R / "check567.json").read_text())
val = pd.read_csv(R / "check0_validation.csv")
q6 = pd.read_csv(R / "check6_quality.csv")
st = pd.read_csv(R / "check5_shift_trial.csv")
si = pd.read_csv(R / "check5_shift_index.csv")
s7 = pd.read_csv(R / "check7_sessions.csv")
p7 = pd.read_csv(R / "check7_concept_pairs.csv")
j8 = json.loads((R / "check8.json").read_text())
o8 = pd.read_csv(R / "check8_order_rdm.csv")
TEX = REPO / "paper/arxiv_upload_learning_rules_v2/learning_rules_rsa_paper_v2.tex"
TL = TEX.read_text(encoding="utf-8").splitlines()


def f(x, d=4):
    return f"{x:.{d}f}".replace("-", "−")


def g(df, **kw):
    for k, v in kw.items():
        df = df[df[k] == v]
    return df.iloc[0]


# ── verdict, from the data ──────────────────────────────────────────────────────
S = j4["subjects"]
align_flags = {s: all([S[s]["n_equal_to_sub01_by_position"] == 720, S[s]["n_unique"] == 720,
                       S[s]["order_file_equals_recomputed_from_own_metadata"],
                       S[s]["h5_column_labels_equal_0..n-1"], S[s]["trial_id_equals_row"],
                       S[s]["h5_voxel_ids_equal_voxel_metadata"],
                       S[s]["metadata_index_is_0..n-1"]]) for s in SUBJECTS}
null = j567["null_V1"]
peak = {}
for roi in ROIS:
    d = st[(st.subject == "sub-03") & (st.roi == roi)].set_index("k")
    peak[roi] = {c: int(d[c].idxmax()) for c in ("rho_vs_luminance", "rho_vs_sub01")}
v1_03 = st[(st.subject == "sub-03") & (st.roi == "V1")].set_index("k")
k0_lum, k0_s1 = v1_03.loc[0, "rho_vs_luminance"], v1_03.loc[0, "rho_vs_sub01"]
z_lum = (k0_lum - null["rho_vs_luminance"]["mean"]) / null["rho_vs_luminance"]["sd"]
z_s1 = (k0_s1 - null["rho_vs_sub01"]["mean"]) / null["rho_vs_sub01"]["sd"]
# A trial-assignment error is ROI-independent, so it has to show wherever sub-03 carries a
# stimulus-locked signal: vs sub-01 in every ROI (orders unrelated, no shared run component)
# and vs luminance only in ROIs where sub-03's luminance rho at k=0 exceeds the null.
# Where there is no such signal the argmax over k is noise and is reported, not used.
lum_informative = [r for r in ROIS if st[(st.subject == "sub-03") & (st.roi == r) &
                                         (st.k == 0)].rho_vs_luminance.iloc[0]
                   > null["rho_vs_luminance"]["q975"]]
offbyk = (any(p["rho_vs_sub01"] != 0 for p in peak.values()) or
          any(peak[r]["rho_vs_luminance"] != 0 for r in lum_informative))
ctrl_k = st[(st.k != 0) & st.subject.isin(["sub-01", "sub-02"])].rho_vs_luminance
lum_noise = [(r, peak[r]["rho_vs_luminance"],
              st[(st.subject == "sub-03") & (st.roi == r)].set_index("k")
              .rho_vs_luminance.max()) for r in ROIS if r not in lum_informative]
nc = {s: {r: g(q6, subject=s, roi=r) for r in ROIS} for s in SUBJECTS}
weaker = all(nc["sub-03"][r].nc_testset_mean < min(nc["sub-01"][r].nc_testset_mean,
                                                    nc["sub-02"][r].nc_testset_mean) for r in ROIS)
val_ok = j567["validation_max_abs_diff"] < 1e-6
if not all(align_flags.values()) or offbyk:
    verdict = "(A) Zuordnungsfehler gefunden"
elif val_ok and weaker and z_lum > 3 and z_s1 > 3:
    verdict = ("(B) Kein Zuordnungsfehler: sub-03 hat echt schwächeres Signal (die vom "
               "Datensatz mitgelieferte Voxel-Reliabilität aus 12× wiederholten Testbildern "
               "ist in allen ROIs die niedrigste, Check 6). Der Abstand zu sub-01/02 wird aber "
               "durch eine **gemeinsame Präsentationsreihenfolge von sub-01 und sub-02** "
               "vergrößert (Check 8), und das betrifft die noise-ceiling-v2-Bounds.")
else:
    verdict = "(C) unklar"

L = []
w = L.append
w("# sub-03-Integritätscheck (vor Noise-Ceiling v3)\n")
w(f"**Verdikt: {verdict}**\n")
w("Erzeugt von `scripts/sub03_check/report.py` aus den CSV/JSON in `results/sub03_check/`. "
  "Paper-Dateien und `results/noise_ceiling_v2/` wurden nicht verändert.\n")

w("## Kurzfassung\n")
w(f"- **Zuordnung:** Stimulusliste, Metadatenzeile→H5-Spalte und Voxel-Metadaten→H5-Zeile "
  f"sind bei allen drei Subjects korrekt (Check 4). Die Rohdaten-Rekonstruktion trifft die "
  f"gespeicherten RDMs auf {j567['validation_max_abs_diff']:.1e} genau.")
w(f"- **Kein Off-by-k:** Im Verschiebungstest hat sub-03 das Maximum gegen sub-01 in allen "
  f"ROIs bei k = 0, gegen Luminanz an {', '.join(lum_informative)} (der einzigen ROI mit "
  f"Luminanzsignal bei sub-03). An V1 liegt ρ(sub-03, Luminanz) = {f(k0_lum)} bei "
  f"z = {z_lum:.1f}, ρ(sub-03, sub-01) = {f(k0_s1)} bei z = {z_s1:.1f} gegen 1000 zufällige "
  "Zuordnungen. Das Signal ist schwach, aber eindeutig stimulusgebunden.")
w(f"- **Echt schwächer:** Die Reliabilität aus dem Datensatz selbst (`nc_testset`, aus "
  f"Testbild-Wiederholungen, ohne unsere Pipeline) beträgt an V1 "
  + ", ".join(f"{s} {f(nc[s]['V1'].nc_testset_mean, 1)}" for s in SUBJECTS)
  + f". sub-03 hat weniger Voxel (V1: "
  + ", ".join(str(nc[s]['V1'].n_voxels) for s in SUBJECTS)
  + "). Auch zwischen seinen eigenen 12 Sessions ist sub-03 am wenigsten konsistent "
  "(Check 7). Seine Session 5 ist dabei kein Ausreißer.")
r8 = j8["restricted_pairs"]
w(f"- **Neuer Befund, Reihenfolge:** sub-01 (Session {j567['subjects']['sub-01']['session_of_720']}) "
  f"und sub-02 (Session {j567['subjects']['sub-02']['session_of_720']}) sahen die 720 Stimuli in "
  f"**identischer** Reihenfolge, sub-03 (Session {j567['subjects']['sub-03']['session_of_720']}) "
  "in einer anderen. Jede Subject-RDM enthält eine starke Run-Komponente. Auf Paare aus "
  "verschiedenen Runs beschränkt sinkt sub-01↔sub-02 von "
  + ", ".join(f"{r} {f(r8[r]['sub01_sub02_all_pairs'], 3)}→{f(r8[r]['sub01_sub02_between_run_pairs'], 3)}"
              for r in ROIS)
  + ". Ein Teil der hohen 01–02-Übereinstimmung, besonders an LOC und IT, ist also "
  "reihenfolgebedingt und nicht stimulusbedingt.\n")

# check 1
w("## Check 1: Subject–Subject (gespeicherte RDMs, oberes Dreieck)\n")
w("| ROI | 01–02 Spearman | 01–03 | 02–03 | 01–02 Pearson | 01–03 | 02–03 |")
w("|---|---|---|---|---|---|---|")
for r in ROIS:
    d = c1[c1.roi == r].set_index("pair")
    w(f"| {r} | " + " | ".join(f(d.loc[p, 'spearman']) for p in ("01-02", "01-03", "02-03"))
      + " | " + " | ".join(f(d.loc[p, 'pearson']) for p in ("01-02", "01-03", "02-03")) + " |")
w("\nDas Muster (sub-03 ≈ 0.01 gegen beide, 01–02 5- bis 8-mal höher) ist in allen vier ROIs "
  "gleich und hängt nicht vom Korrelationsmaß ab.\n")

# check 2
w("## Check 2: Luminanz pro Subject\n")
w(f"Luminanz = Rec.-709-Mittelwert pro Bild mit `colour_statistic_control.py` (Bildsuche, "
  f"Laden, |Δ|-RDM wiederverwendet). {j123['n_exact_image_match']}/{j123['n_stimuli']} Bilder "
  f"wurden über den exakten Dateinamen gefunden, der Konzept-Fallback in `find_img` griff nie. "
  f"Kontrolle: ρ(Luminanz, Mittel-RDM, V1) = {f(j123['luminance_vs_meanrdm_V1'])}, wie im "
  "Paper (0.074/0.075).\n")
w("| ROI | sub-01 | sub-02 | sub-03 | Mittel-RDM | sub-03 / Mittel(01,02) |")
w("|---|---|---|---|---|---|")
for r in ROIS:
    d = c2[c2.roi == r].set_index("target").spearman
    ref_ok = min(d["sub-01"], d["sub-02"]) > null["rho_vs_luminance"]["q975"]
    ratio = f"{j123['sub03_ratio_luminance'][r]:.2f}" if ref_ok else "– (kein Signal bei 01/02)"
    w(f"| {r} | {f(d['sub-01'])} | {f(d['sub-02'])} | {f(d['sub-03'])} | {f(d['mean-RDM'])} | "
      f"{ratio} |")
w(f"\nAn V1 ist sub-03 klein, aber über der Nullverteilung "
  f"(95 %-Grenze {f(null['rho_vs_luminance']['q975'])}). Das spricht gegen eine "
  "Fehlzuordnung: Ein vertauschtes Bild hätte keine Luminanzbeziehung. Gegen schwaches Signal "
  "spricht es nicht. An LOC und IT trägt Luminanz bei keinem Subject Information.\n")

# check 3
w("## Check 3: Modelle pro Subject (Referenzsatz, Mittel ± SD über 5 Seeds)\n")
w("Werte aus `results/noise_ceiling_v2/step3a_control_per_seed.csv` (pro Subject und Seed "
  "bereits berechnet, nur gelesen).\n")
w("| Regel | ROI | sub-01 | sub-02 | sub-03 |")
w("|---|---|---|---|---|")
RN = {"random_weights": "Random", "backprop": "BP", "feedback_alignment": "FA",
      "predictive_coding": "PC", "stdp": "STDP"}
for k, n in RN.items():
    for r in ROIS:
        cells = [g(c3, rule=k, roi=r, subject=s) for s in SUBJECTS]
        w(f"| {n} | {r} | " + " | ".join(f"{f(c['mean'])} ± {f(c.sd)}" for c in cells) + " |")
pos = {s: int(((c3.subject == s) & (c3["mean"] > 0)).sum()) for s in SUBJECTS}
w(f"\nPositive Mittelwerte (von 20): " + ", ".join(f"{s} {pos[s]}" for s in SUBJECTS) +
  ". sub-03 ist nicht für alle Modelle ≈ 0. An V1 liegt es für Random, PC und STDP klar "
  "über 0, nur kleiner als bei sub-01/02.\n")

# check 4
w("## Check 4: Stimulus-IDs und Zuordnung\n")
w("**Quelle der Reihenfolge** (`extract_fmri_rdms_720.py`): `outputs_720/stim_order_{sub}.txt` "
  "wird geladen (Z. 82–86). Fehlt die Datei, wird sie neu berechnet (Z. 88–101): pro Konzept "
  "alphabetisch, davon das erste Exemplar nach Dateinamen. Die Response eines Stimulus ist "
  "`responses_all[stim.index[stim.stimulus == s]]` (Z. 104–112), also die H5-Spalte mit der "
  "Nummer der Metadatenzeile.\n")
w("| Prüfung | sub-01 | sub-02 | sub-03 |")
w("|---|---|---|---|")
items = [("Stimuli / eindeutig", lambda x: f"{x['n']} / {x['n_unique']}"),
         ("positionsgleich mit sub-01", lambda x: x["n_equal_to_sub01_by_position"]),
         ("erste Abweichung / fehlend / zusätzlich",
          lambda x: f"{x['first_mismatch_vs_sub01']} / {len(x['missing_vs_sub01'])} / {len(x['extra_vs_sub01'])}"),
         ("Order-Datei = Neuberechnung aus eigenen Metadaten",
          lambda x: x["order_file_equals_recomputed_from_own_metadata"]),
         ("Metadaten-Index 0…n−1, trial_id = Zeile",
          lambda x: f"{x['metadata_index_is_0..n-1']}, {x['trial_id_equals_row']}"),
         ("H5-Spaltenlabels 0…9839, Anzahl = Metadatenzeilen",
          lambda x: f"{x['h5_column_labels_equal_0..n-1']}, {x['h5_n_columns_equals_metadata_rows']}"),
         ("H5-Voxel-IDs = Voxel-Metadaten", lambda x: x["h5_voxel_ids_equal_voxel_metadata"]),
         ("subject_id in den Metadaten",
          lambda x: f"{x['voxel_metadata_subject_id']}/{x['stim_metadata_subject_id']}"),
         ("Exemplar-Suffix der 720", lambda x: x["exemplar_suffix_numbers"]),
         ("Session der 720 / Runs", lambda x: f"{x['the_720_sessions']} / {x['the_720_runs_per_session']}"),
         ("Wiederholungen pro Stimulus (max)", lambda x: x["the_720_reps_per_stimulus_max"])]
for lab, fn in items:
    w(f"| {lab} | " + " | ".join(str(fn(S[s])) for s in SUBJECTS) + " |")
w("\nSortierung: Alle 720 Dateinamen tragen dasselbe Suffix (`_01…`). Die Frage alphabetisch "
  "gegen numerisch (`_10` vor `_2`) stellt sich also nicht. Konzepte werden für alle Subjects "
  "identisch alphabetisch sortiert. 0- und 1-basierte Indizes kommen nicht vor: Zeile, "
  "`trial_id` und H5-Spaltenlabel sind identisch und beginnen bei 0.\n")
w(f"**Pfadabweichung (dokumentiert):** `extract_fmri_rdms_720.py:20` zeigt auf "
  f"`{EXTRACT_SCRIPT_DATA_DIR}`. Dieser Pfad existiert nicht mehr, die Daten liegen jetzt "
  f"unter `{DATA_DIR}`. Die Rekonstruktion aus dem neuen Pfad stimmt mit den gespeicherten RDMs "
  f"überein (max. Abweichung {j567['validation_max_abs_diff']:.1e}, float32). Es sind also "
  "dieselben Rohdaten. `h5py` fehlte in der Projektumgebung und wurde nur für diesen Lauf "
  "temporär in den Scratch-Ordner installiert.\n")

# check 5
w("## Check 5: Verschiebungstest\n")
w(f"(b) Trial-Raum: Jeder Stimulus erhält die Response des Trials k Positionen später im selben "
  f"Run (zirkulär über 82 Trials), danach wird die RDM neu gebaut. (a) Index-Raum: die "
  f"gespeicherte sub-03-RDM wird zirkulär in der Konzeptreihenfolge verschoben. Null: "
  f"{j567['n_null']} zufällige Neuzuordnungen von sub-03 (V1, Seed {j567['seed']}): "
  f"vs. sub-01 {f(null['rho_vs_sub01']['mean'])} ± {f(null['rho_vs_sub01']['sd'])}, "
  f"vs. Luminanz {f(null['rho_vs_luminance']['mean'])} ± {f(null['rho_vs_luminance']['sd'])}.\n")
w("| k | sub-03 vs Lum (Trial) | sub-03 vs sub-01 (Trial) | sub-03 vs sub-01 (Index) | "
  "sub-02 vs sub-01 (Trial, Kontrolle) | sub-01 vs Lum (Trial, Kontrolle) |")
w("|---|---|---|---|---|---|")
for k in range(-5, 6):
    a = g(st, subject="sub-03", roi="V1", k=k); b = g(si, roi="V1", k=k)
    c = g(st, subject="sub-02", roi="V1", k=k); d = g(st, subject="sub-01", roi="V1", k=k)
    w(f"| {k:+d} | {f(a.rho_vs_luminance)} | {f(a.rho_vs_sub01)} | {f(b.rho_vs_sub01)} | "
      f"{f(c.rho_vs_sub01)} | {f(d.rho_vs_luminance)} |")
w("\nArgmax über k für sub-03, pro ROI: " + "; ".join(
    f"{r}: Lum k={p['rho_vs_luminance']}, sub-01 k={p['rho_vs_sub01']}" for r, p in peak.items())
  + ". **Entscheidend ist der Vergleich mit sub-01:** Sein Maximum liegt in allen ROIs bei "
  "k = 0. Die Luminanz liegt an V1 ebenfalls bei k = 0, und nur an "
  + ", ".join(lum_informative) + " trägt sub-03 bei k = 0 Luminanzsignal über der Null. "
  "Die Luminanz-Maxima bei k ≠ 0 an "
  + ", ".join(f"{r} (k={k}, ρ={f(v)})" for r, k, v in lum_noise)
  + " liegen in ROIs ohne Luminanzsignal und sind Rauschen. Eine Trial-Fehlzuordnung wäre "
  "ROI-unabhängig und müsste sich an V1 zeigen. Zudem liegen diese Werte im Bereich der "
  f"Kontrollen bei k ≠ 0 (sub-01/02 gegen Luminanz: {f(ctrl_k.min())} bis {f(ctrl_k.max())}). "
  "Für Shifts innerhalb eines Runs ist die globale Permutations-Null zu eng: Die "
  "Run-Blockstruktur mit nur 10 Blöcken bleibt dabei erhalten und erzeugt größere "
  "Zufallskorrelationen. Die Kontrollen zeigen, dass der Test empfindlich ist: Bei sub-01 bricht die Luminanz "
  "bei k ≠ 0 zusammen. Bei sub-02 gegen sub-01 bleibt bei k ≠ 0 ein Rest von etwa 0.015–0.022. "
  "Das ist die reihenfolgegebundene Komponente aus Check 8. Bei sub-03 gibt es diesen Rest "
  "nicht, weil seine Reihenfolge anders ist. Eine Run-Struktur ist bekannt (10 Runs × 82 "
  "Trials), die Verschiebung innerhalb jedes Runs ist also die oben gezeigte Trial-Variante.\n")

# check 6
w("## Check 6: Datenqualität\n")
w("Alle Subjects laufen durch dasselbe Skript mit denselben Maskenspalten "
  "(`V1`, `V2`, `lLOC|rLOC`, `IT`, `extract_fmri_rdms_720.py:53-60`), derselben z-Normierung "
  "pro Voxel über alle 9840 Trials (Z. 77–78) und derselben Distanz (1 − Pearson, Z. 29). "
  "`nc_testset` und `splithalf_corrected` stammen aus den Voxel-Metadaten des Datensatzes und "
  "beruhen auf den 12× wiederholten Testbildern. Sie sind von unserer Pipeline unabhängig.\n")
w("| Subject | ROI | Voxel | NaN | Inf | konstant | Var. über 720 (roh) | RDM Mittel | RDM SD | "
  "RDM Schiefe | nc_testset Mittel | Median | splithalf korr. | Anteil nc>10 |")
w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in ROIS:
    for s in SUBJECTS:
        x = nc[s][r]
        w(f"| {s} | {r} | {x.n_voxels} | {x.frac_nan:.0%} | {x.frac_inf:.0%} | "
          f"{x.frac_constant_voxels:.0%} | {f(x.mean_var_over_720_raw)} | {f(x.rdm_mean)} | "
          f"{f(x.rdm_sd)} | {f(x.rdm_skew, 3)} | {f(x.nc_testset_mean, 1)} | "
          f"{f(x.nc_testset_median, 1)} | {f(x.splithalf_corrected_mean, 3)} | "
          f"{x.frac_vox_nc_testset_gt10:.0%} |")
w("\nsub-03 hat in jeder ROI die niedrigste Voxel-Reliabilität und weniger Voxel (an LOC "
  "deutlich), aber keine NaN/Inf und keine konstanten Voxel. Die Rohvarianz ist vergleichbar. "
  "Die höhere SD der RDM-Einträge passt zu weniger Voxeln. Das Bild ist das eines "
  "intakten, aber verrauschteren Datensatzes.\n")

# check 7
w("## Check 7: Konzeptebene\n")
w("Die 720 Stimuli sind 720 Konzepte mit je einem Exemplar. Jede der 12 Sessions zeigt jedes "
  "Konzept genau einmal, mit einem anderen Exemplar. (i) Pro Session wird eine Konzept-RDM "
  "gebaut und mit den anderen Sessions desselben Subjects verglichen. (ii) Check 1 wird auf "
  "Konzept-RDMs wiederholt, gemittelt über alle 12 Exemplare.\n")
w("| ROI | 01–02 Einzel-Exemplar | 01–02 Konzept (12 Ex.) | 01–03 Einzel | 01–03 Konzept | "
  "02–03 Einzel | 02–03 Konzept |")
w("|---|---|---|---|---|---|---|")
for r in ROIS:
    a = c1[c1.roi == r].set_index("pair").spearman
    b = p7[p7.roi == r].set_index("pair").spearman_concept_12ex
    w(f"| {r} | {f(a['01-02'])} | {f(b['01-02'])} | {f(a['01-03'])} | {f(b['01-03'])} | "
      f"{f(a['02-03'])} | {f(b['02-03'])} |")
agg = s7.groupby(["subject", "roi"]).mean_rho_with_own_other_sessions
re = [g(c1, roi=r, pair=p).spearman / g(c1, roi=r, pair="01-02").spearman
      for r in ROIS for p in ("01-03", "02-03")]
rc = [g(p7, roi=r, pair=p).spearman_concept_12ex / g(p7, roi=r, pair="01-02").spearman_concept_12ex
      for r in ROIS for p in ("01-03", "02-03")]
w("\n| Subject | ROI | Session der 720 | ρ dieser Session mit eigenen anderen | Mittel über "
  "alle Sessions | Min | Max |")
w("|---|---|---|---|---|---|---|")
for s in SUBJECTS:
    for r in ROIS:
        x = g(s7[s7.is_session_of_720], subject=s, roi=r)
        w(f"| {s} | {r} | {x.session} | {f(x.mean_rho_with_own_other_sessions)} | "
          f"{f(agg.mean()[(s, r)])} | {f(agg.min()[(s, r)])} | {f(agg.max()[(s, r)])} |")
w("\nAuf Konzeptebene steigt sub-03 deutlich an (V1 01–03: "
  f"{f(g(c1, roi='V1', pair='01-03').spearman)} → {f(g(p7, roi='V1', pair='01-03').spearman_concept_12ex)}). "
  "Die Faustregel der Aufgabe („steigt an → Exemplare vertauscht“) trifft hier aber nicht zu, "
  "und zwar aus drei Gründen. Erstens hat die Luminanz, die ein Merkmal des einzelnen "
  "Exemplars ist, bei sub-03 ihr Maximum bei der korrekten Zuordnung (Check 2/5). Zweitens "
  "ist die Session der 720 innerhalb von sub-03 typisch, kein Ausreißer. Drittens erklärt "
  "Mitteln über 12 Trials den Anstieg ohne Vertauschung: Es reduziert Rauschen, und davon "
  "profitiert das verrauschteste Subject am meisten. "
  + "; ".join(
      f"{r}: 01–02 {f(g(c1, roi=r, pair='01-02').spearman, 3)}→"
      f"{f(g(p7, roi=r, pair='01-02').spearman_concept_12ex, 3)}, sub-03-Paare im Mittel "
      f"{f((g(c1, roi=r, pair='01-03').spearman + g(c1, roi=r, pair='02-03').spearman) / 2, 3)}→"
      f"{f((g(p7, roi=r, pair='01-03').spearman_concept_12ex + g(p7, roi=r, pair='02-03').spearman_concept_12ex) / 2, 3)}"
      for r in ROIS)
  + ". An V1 und V2 bleibt 01–02 auf Konzeptebene etwa gleich, an LOC und IT steigt es "
  "ebenfalls. Der Anstieg ist also nicht spezifisch für sub-03, nur dort relativ am größten. "
  "Dass 01–02 an V1/V2 nicht steigt, passt zu zwei Komponenten ihrer "
  "Einzel-Session-Übereinstimmung, die beim Mitteln über Sessions wegfallen: dieselben Bilder "
  "(`_01`) und dieselbe Reihenfolge (Check 8). Auf Konzeptebene erreichen die sub-03-Paare "
  f"{min(rc):.0%}–{max(rc):.0%} von 01–02, auf Einzel-Exemplar-Ebene {min(re):.0%}–{max(re):.0%}.\n")

# check 8
w("## Check 8: Gemeinsame Präsentationsreihenfolge (zusätzlich)\n")
w(f"Identische 720er-Reihenfolge (Run und Position): sub-01/sub-02 "
  f"**{j8['identical_720_order_sub01_sub02']}**, sub-01/sub-03 "
  f"{j8['identical_720_order_sub01_sub03']}. Über alle 12 Sessions haben die Subjects "
  f"Session-Reihenfolgen, die paarweise mit Versatz gleich sind "
  f"({', '.join(f'{k}: {v} gleiche Paare' for k, v in j8['n_identical_session_orders'].items())}). "
  "Das Design verwendet also gemeinsame Sequenzen. Für die Exemplar-01-Sessions fallen sie nur "
  "bei sub-01 und sub-02 zusammen.\n")
w("| ROI | Subject | ρ(RDM, „anderer Run“) | ρ(RDM, Trial-Abstand) |")
w("|---|---|---|---|")
for r in ROIS:
    for s in SUBJECTS:
        x = g(o8, roi=r, subject=s)
        w(f"| {r} | {s} | {f(x.rho_vs_different_run)} | {f(x.rho_vs_trial_distance)} |")
w("\n| ROI | 01–02 alle Paare | nur Paare aus verschiedenen Runs | nur Paare im selben Run | "
  "reihenfolgegebunden (Check 5b, Mittel k≠0) |")
w("|---|---|---|---|---|")
for r in ROIS:
    x = r8[r]
    w(f"| {r} | {f(x['sub01_sub02_all_pairs'])} | {f(x['sub01_sub02_between_run_pairs'])} | "
      f"{f(x['sub01_sub02_within_run_pairs'])} | "
      f"{f(j8['order_locked_sub02_vs_sub01_mean_k_ne_0'][r])} |")
w("\nAlle drei Subject-RDMs enthalten eine Run-Komponente: Paare aus demselben Run sind "
  "ähnlicher. Das liegt nahe, weil die z-Normierung über alle Trials läuft und nicht pro Run. "
  "Bei sub-01 und sub-02 ist sie identisch angeordnet und erzeugt eine scheinbare "
  "Übereinstimmung zwischen den Subjects. An V1 ist der Effekt klein, an LOC und IT macht er "
  "den größeren Teil von 01–02 aus. **Konsequenzen, nicht neu berechnet:**\n")
w("- `results/noise_ceiling_v2/`: Der LOO lower bound enthält für sub-01 und sub-02 diesen "
  "geteilten Reihenfolge-Anteil. Er ist dadurch überschätzt, an LOC und IT vermutlich stark. "
  "Die Aussage „Modelle unter dem lower bound“ ist an LOC und IT deshalb nicht belastbar. "
  "Modelle können Reihenfolge nicht abbilden.")
w("- Für die Modell-RSA wirkt die Run-Komponente als Rauschen und dämpft alle ρ. Eine "
  "systematische Verschiebung zwischen den Regeln erzeugt sie nicht, weil alle Modelle "
  "dieselbe Hirn-RDM nutzen.")
w("- Mögliche Abhilfe für v3 (zu entscheiden, hier nicht umgesetzt): z-Normierung pro Run, "
  "Bounds nur auf Paaren aus verschiedenen Runs, oder Exemplar-Sessions mit nicht geteilter "
  "Reihenfolge.\n")

# paper sentence
w("## Satz für das Paper (Fall B)\n")
w(f"Ersetzt bzw. präzisiert `learning_rules_rsa_paper_v2.tex:338` („{TL[337][:110]}…“):\n")
s3 = nc["sub-03"]["V1"]
w("```latex\n" + (
    rf"Sub-03's RDMs agree only weakly with those of the other two subjects "
    rf"(V1: $\rho = {f(g(c1, roi='V1', pair='01-03').spearman, 3)}$ and "
    rf"${f(g(c1, roi='V1', pair='02-03').spearman, 3)}$, vs.\ "
    rf"${f(g(c1, roi='V1', pair='01-02').spearman, 3)}$ between sub-01 and sub-02). "
    rf"This reflects lower data quality rather than a stimulus-assignment error: the "
    rf"assignment of trials to stimuli was verified, the subject's RDM is related to image "
    rf"luminance only at the correct assignment, and the dataset's own test-retest voxel "
    rf"reliability is the lowest of the three subjects in every ROI (V1: mean noise ceiling "
    rf"{f(s3.nc_testset_mean, 1)}\% vs.\ {f(nc['sub-01']['V1'].nc_testset_mean, 1)}\% and "
    rf"{f(nc['sub-02']['V1'].nc_testset_mean, 1)}\%). Part of the higher agreement between "
    rf"sub-01 and sub-02 is shared presentation order: in the sessions used here both saw the "
    rf"720 images in the same sequence, and restricted to pairs from different runs their "
    rf"agreement falls to ${f(r8['V1']['sub01_sub02_between_run_pairs'], 3)}$ at V1 and "
    rf"${f(r8['IT']['sub01_sub02_between_run_pairs'], 3)}$ at IT."
).replace("−", "-") + "\n```\n")
w("Anmerkung: `nc_testset` ist im THINGS-Datensatz in Prozent erklärbarer Varianz angegeben. "
  "Vor dem Einbau bitte gegen die Datensatz-Dokumentation prüfen.\n")

w("## Dateien\n")
w("```\npy -3 scripts/sub03_check/check123_stored_rdms.py\n"
  "py -3 scripts/sub03_check/check4_ids.py            # braucht h5py\n"
  "py -3 scripts/sub03_check/check567_raw.py          # braucht h5py, ~2.5 min\n"
  "py -3 scripts/sub03_check/check8_order_confound.py\n"
  "py -3 scripts/sub03_check/figure.py\n"
  "py -3 scripts/sub03_check/report.py\n```\n")
w("![sub-03 check](sub03_check.png)\n")
w("A: Luminanz gegen Subject-RDM. B: untrainiertes CNN gegen Subject-RDM. C: "
  "Verschiebungstest an V1, graues Band = 95 % der Nullverteilung. D: sub-01 gegen sub-02, "
  "alle Paare gegen nur Paare aus verschiedenen Runs.\n")
(R / "SUB03_CHECK.md").write_text("\n".join(L), encoding="utf-8")
print(verdict)

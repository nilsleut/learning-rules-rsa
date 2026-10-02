# Cross-Run-Analyse (Run-Konfundierung) — arXiv:2604.16875 und arXiv:2608.12408

Erzeugt von `scripts/crossrun/report.py` aus `results/crossrun/*`. Paper-Dateien wurden nicht editiert. Die alten RDMs (`outputs_720`, `--zscore all`) und `rdms_runz/` bleiben unverändert.

**Varianten.** *alle Paare* = publizierte Analyse. *Cross-Run (primär)* = nur Paare, die in **keinem** Subject im selben Run liegen: 210,205 von 258,840 Paaren (81.2%). *Cross-Run blockbereinigt* = dieselben Paare. Pro Subject wird von Fach-RDM und Modell- bzw. Luminanz-RDM der Mittelwert jedes Run-Paar-Blocks abgezogen, berechnet auf der Schnittmenge, mit den Runs dieses Subjects. Das geschieht nur in der Pro-Subject-Konvention. **Die Mittel-RDM-Konvention ist für die Blockbereinigung nicht definiert:** Die drei Subjects haben verschiedene Blockstrukturen (sub-03 eine andere Reihenfolge als sub-01/02), und keine einzelne Bereinigung passt auf ihr Mittel.

**Kriterium.** Eine Aussage *hält*, wenn sie in der Variante dasselbe Vorzeichen und dieselbe CI-Signifikanz hat (95 %-CI schliesst 0 aus oder nicht) wie auf *allen Paaren*. Hält sie, aber der Punktschätzer ändert sich um mehr als 25 %, steht *ändert Grösse*. Sonst *kippt*. CIs: Stimulus-Bootstrap (1000 gemeinsame Resamples, dieselben wie in noise_ceiling_v2) für Paper 1, Seed-CI (t, n = 5) für Paper 2.

## Kurzfassung

- **Diagnostik (Schritt 1):** Die Run-Paar-Struktur ist bei sub-01/02 nicht geteilt (exaktes p über 10! Permutationen: V1 0.34, V2 0.32, LOC 0.16, IT 0.23). Primärvariante bleibt deshalb **Cross-Run (primär)**. Die Blockbereinigung senkt den Luminanz-Fit an V1 bei sub-01/02 um 1%, 10% (STOPP-Schwelle 30 %, nicht erreicht).
- **Lower bound:** V1: 0.057 → 0.051 (Cross-Run) / 0.054 (blockbereinigt); V2: 0.033 → 0.024 (Cross-Run) / 0.026 (blockbereinigt); LOC: 0.034 → 0.018 (Cross-Run) / 0.019 (blockbereinigt); IT: 0.035 → 0.021 (Cross-Run) / 0.021 (blockbereinigt). Der Reihenfolge-Anteil sass im Bound, vor allem an LOC und IT.
- **Paper 1 (Referenzsatz):** Von 48 geprüften Aussage × Konvention-Zellen kippen unter Cross-Run 4, 13 ändern die Grösse. Blockbereinigt (24 Zellen, nur pro Subject): 0 kippen, 11 ändern die Grösse. Die kippenden Zellen betreffen: IT konvergiert: BP vs PC n.s., IT konvergiert: FA vs PC n.s., IT konvergiert: FA vs STDP n.s. Art der Wechsel in §3.
- **Paper 2:** Random − BP an V1 bei 224 px (Mittel-RDM) 0.044 → 0.043 (Cross-Run). Pro Subject 0.032 → 0.031 / 0.033. Luminanz an V1 (Mittel-RDM) 0.074 → 0.076. Details in §4.

## 1. Diagnostik (vor allen Modellzahlen)

**a) Lag innerhalb eines Runs** (alte RDMs, alle Paare im selben Run; diese Paare sind in der Cross-Run-Analyse ausgeschlossen):

| ROI | Subject | Spearman(Dissimilarität, Lag) | Mittel Lag 1 | Mittel Lag ≥ 10 | Mittel Cross-Run |
|---|---|---|---|---|---|
| V1 | sub-01 | −0.133 | 0.9580 | 0.9556 | 0.9924 |
| V1 | sub-02 | −0.164 | 0.9717 | 0.9397 | 0.9881 |
| V1 | sub-03 | −0.101 | 0.9620 | 0.9719 | 0.9983 |
| V2 | sub-01 | −0.143 | 0.9544 | 0.9575 | 0.9949 |
| V2 | sub-02 | −0.162 | 0.9407 | 0.9429 | 0.9833 |
| V2 | sub-03 | −0.115 | 0.9587 | 0.9686 | 0.9988 |
| LOC | sub-01 | −0.259 | 1.0282 | 0.9702 | 0.9998 |
| LOC | sub-02 | −0.232 | 1.0027 | 0.9392 | 0.9902 |
| LOC | sub-03 | −0.197 | 1.0066 | 0.9716 | 0.9996 |
| IT | sub-01 | −0.277 | 1.0299 | 0.9750 | 0.9999 |
| IT | sub-02 | −0.300 | 1.0113 | 0.9580 | 0.9944 |
| IT | sub-03 | −0.204 | 0.9978 | 0.9822 | 0.9997 |

Im selben Run nimmt die Dissimilarität mit dem Abstand ab. Benachbarte Trials sind unähnlicher als entfernte, an LOC und IT sogar über 1 (also antikorreliert). Das passt zu negativ gekoppelten Einzeltrial-Schätzungen bei schneller Trialfolge. Paare aus verschiedenen Runs liegen nahe 1. Das ist nur berichtet, es geht nicht in die Analyse ein.

**b) Run-Paar-Struktur** (Schnittmenge P, 10×10-Mittel pro Run-Paar, 45 Werte). 01–02 haben identische Run-Labels. Exakte Null über alle 10! = 3 628 800 Umbenennungen der Runs von sub-02:

| ROI | Spearman 01–02 | Null Mittel ± SD | p (einseitig, exakt) |
|---|---|---|---|
| V1 | 0.055 | 0.000 ± 0.228 | 0.336 |
| V2 | 0.065 | 0.000 ± 0.181 | 0.323 |
| LOC | 0.171 | −0.000 ± 0.160 | 0.160 |
| IT | 0.097 | 0.000 ± 0.129 | 0.226 |

In keiner ROI signifikant. Cross-Run allein reicht also; die Blockbereinigung bleibt Robustheitscheck. Die Power ist mit 45 Run-Paaren begrenzt; die Blockbereinigung wird deshalb trotzdem für alle Aussagen mitgerechnet.

**c) Luminanz-Fit pro Subject** (*Diagnostik mit Effektcharakter*, siehe §6):

| ROI | Subject | alle Paare | Cross-Run | blockbereinigt | Änderung Cross-Run → bereinigt |
|---|---|---|---|---|---|
| V1 | sub-01 | 0.0520 | 0.0544 | 0.0537 | -1% |
| V1 | sub-02 | 0.0940 | 0.0958 | 0.0858 | -10% |
| V1 | sub-03 | 0.0107 | 0.0107 | 0.0105 | -2% |
| V2 | sub-01 | 0.0335 | 0.0336 | 0.0328 | -2% |
| V2 | sub-02 | 0.0608 | 0.0612 | 0.0513 | -16% |
| V2 | sub-03 | 0.0015 | 0.0023 | 0.0021 | – (kein Fit) |
| LOC | sub-01 | 0.0019 | 0.0006 | 0.0015 | – (kein Fit) |
| LOC | sub-02 | 0.0024 | 0.0046 | −0.0094 | – (kein Fit) |
| LOC | sub-03 | −0.0021 | −0.0024 | −0.0026 | – (kein Fit) |
| IT | sub-01 | 0.0032 | 0.0023 | 0.0023 | – (kein Fit) |
| IT | sub-02 | 0.0061 | 0.0069 | −0.0033 | -149% |
| IT | sub-03 | 0.0017 | 0.0029 | 0.0025 | – (kein Fit) |

Bei sub-03 (Luminanz–Run-Kopplung p = 0.005, siehe `results/runz`) entfällt an V1 2% des Luminanz-Fits auf die Run-Paar-Mittel, also praktisch nichts. STOPP-Kriterium (Abnahme > 30 % bei sub-01/02 an V1): nicht erfüllt.

**d) Übereinstimmung zwischen Subjects:**

| ROI | Paar | alle Paare | Cross-Run | blockbereinigt |
|---|---|---|---|---|
| V1 | 01-02 | 0.1152 | 0.0970 | 0.1050 |
| V1 | 01-03 | 0.0141 | 0.0171 | 0.0171 |
| V1 | 02-03 | 0.0142 | 0.0135 | 0.0146 |
| V2 | 01-02 | 0.0633 | 0.0414 | 0.0481 |
| V2 | 01-03 | 0.0074 | 0.0068 | 0.0064 |
| V2 | 02-03 | 0.0056 | 0.0058 | 0.0064 |
| LOC | 01-02 | 0.0618 | 0.0229 | 0.0235 |
| LOC | 01-03 | 0.0076 | 0.0095 | 0.0099 |
| LOC | 02-03 | 0.0090 | 0.0092 | 0.0094 |
| IT | 01-02 | 0.0659 | 0.0292 | 0.0298 |
| IT | 01-03 | 0.0086 | 0.0098 | 0.0100 |
| IT | 02-03 | 0.0074 | 0.0095 | 0.0098 |

01–02 liegt in allen Varianten über 01–03/02–03. Änderung von allen Paaren auf Cross-Run für 01–02: V1 -16%, V2 -35%, LOC -63%, IT -56%. Das ist der Reihenfolge-Anteil aus SUB03_CHECK.

## 2. Noise-Ceiling-Bounds

Bootstrap: 1000 Resamples, dieselbe Indexmatrix wie noise_ceiling_v2. Die Variante *alle Paare* reproduziert v2 exakt (max. Abweichung 2e-16). Cross-Run-Paare pro Resample: Median 209,937. Permutations-Null: 1000× Stimuluslabels pro Subject unabhängig permutiert, Seed 20261005.

| ROI | Variante | lower [95%-CI] | lower H0 (Mittel ± SD) | upper | upper H0 | Status upper |
|---|---|---|---|---|---|---|
| V1 | alle Paare (publiziert) | 0.0570 [0.0455, 0.0688] | 0.0001 ± 0.0020 | 0.5752 | 0.5477 | bei N=3 nicht informativ |
| V1 | Cross-Run (primär) | 0.0514 [0.0391, 0.0639] | 0.0000 ± 0.0021 | 0.5721 | 0.5477 | bei N=3 nicht informativ |
| V1 | Cross-Run blockbereinigt | 0.0540 [0.0411, 0.0663] | 0.0000 ± 0.0021 | 0.5725 | 0.5477 | bei N=3 nicht informativ |
| V2 | alle Paare (publiziert) | 0.0326 [0.0247, 0.0407] | −0.0001 ± 0.0018 | 0.5677 | 0.5526 | bei N=3 nicht informativ |
| V2 | Cross-Run (primär) | 0.0236 [0.0158, 0.0323] | −0.0001 ± 0.0019 | 0.5634 | 0.5525 | bei N=3 nicht informativ |
| V2 | Cross-Run blockbereinigt | 0.0258 [0.0180, 0.0340] | −0.0001 ± 0.0019 | 0.5637 | 0.5525 | bei N=3 nicht informativ |
| LOC | alle Paare (publiziert) | 0.0339 [0.0267, 0.0418] | −0.0001 ± 0.0016 | 0.5659 | 0.5502 | bei N=3 nicht informativ |
| LOC | Cross-Run (primär) | 0.0183 [0.0110, 0.0267] | −0.0000 ± 0.0018 | 0.5583 | 0.5502 | bei N=3 nicht informativ |
| LOC | Cross-Run blockbereinigt | 0.0189 [0.0115, 0.0269] | −0.0000 ± 0.0018 | 0.5587 | 0.5503 | bei N=3 nicht informativ |
| IT | alle Paare (publiziert) | 0.0347 [0.0283, 0.0419] | 0.0001 ± 0.0016 | 0.5630 | 0.5468 | bei N=3 nicht informativ |
| IT | Cross-Run (primär) | 0.0211 [0.0139, 0.0284] | 0.0000 ± 0.0018 | 0.5564 | 0.5467 | bei N=3 nicht informativ |
| IT | Cross-Run blockbereinigt | 0.0213 [0.0142, 0.0284] | 0.0000 ± 0.0018 | 0.5564 | 0.5467 | bei N=3 nicht informativ |

## 3. Modell-RSA und Paper-1-Aussagen

### Modell − lower bound, pro Subject (Referenzsatz)

| Regel | ROI | ρ alle Paare (publiziert) | ρ Cross-Run (primär) | ρ Cross-Run blockbereinigt | Δ alle Paare (publiziert) [CI] | Δ Cross-Run (primär) [CI] | Δ Cross-Run blockbereinigt [CI] |
|---|---|---|---|---|---|---|---|
| Random | V1 | 0.0526 | 0.0532 | 0.0534 | −0.0045 [−0.0194, 0.0100] n.u. | 0.0017 [−0.0137, 0.0170] n.u. | −0.0006 [−0.0150, 0.0150] n.u. |
| Random | V2 | 0.0289 | 0.0288 | 0.0282 | −0.0037 [−0.0157, 0.0075] n.u. | 0.0052 [−0.0069, 0.0170] n.u. | 0.0024 [−0.0090, 0.0140] n.u. |
| Random | LOC | −0.0021 | −0.0030 | −0.0049 | −0.0359 [−0.0454, −0.0275] unter | −0.0213 [−0.0325, −0.0118] unter | −0.0238 [−0.0342, −0.0145] unter |
| Random | IT | 0.0048 | 0.0040 | 0.0029 | −0.0299 [−0.0391, −0.0214] unter | −0.0171 [−0.0268, −0.0082] unter | −0.0184 [−0.0276, −0.0092] unter |
| BP | V1 | 0.0222 | 0.0240 | 0.0221 | −0.0349 [−0.0488, −0.0217] unter | −0.0275 [−0.0418, −0.0125] unter | −0.0319 [−0.0444, −0.0176] unter |
| BP | V2 | 0.0123 | 0.0133 | 0.0112 | −0.0203 [−0.0310, −0.0094] unter | −0.0103 [−0.0223, 0.0013] n.u. | −0.0146 [−0.0254, −0.0039] unter |
| BP | LOC | 0.0070 | 0.0075 | 0.0074 | −0.0269 [−0.0364, −0.0174] unter | −0.0107 [−0.0210, −0.0008] unter | −0.0114 [−0.0213, −0.0024] unter |
| BP | IT | 0.0090 | 0.0100 | 0.0097 | −0.0257 [−0.0339, −0.0177] unter | −0.0111 [−0.0196, −0.0025] unter | −0.0117 [−0.0196, −0.0035] unter |
| FA | V1 | 0.0075 | 0.0087 | 0.0065 | −0.0496 [−0.0647, −0.0351] unter | −0.0427 [−0.0588, −0.0274] unter | −0.0475 [−0.0617, −0.0318] unter |
| FA | V2 | 0.0028 | 0.0036 | 0.0015 | −0.0298 [−0.0409, −0.0196] unter | −0.0201 [−0.0323, −0.0087] unter | −0.0243 [−0.0352, −0.0136] unter |
| FA | LOC | 0.0033 | 0.0036 | 0.0026 | −0.0305 [−0.0399, −0.0220] unter | −0.0146 [−0.0247, −0.0056] unter | −0.0163 [−0.0261, −0.0075] unter |
| FA | IT | 0.0067 | 0.0079 | 0.0063 | −0.0280 [−0.0371, −0.0198] unter | −0.0132 [−0.0225, −0.0046] unter | −0.0150 [−0.0237, −0.0065] unter |
| PC | V1 | 0.0384 | 0.0387 | 0.0396 | −0.0187 [−0.0312, −0.0046] unter | −0.0127 [−0.0261, 0.0018] n.u. | −0.0144 [−0.0276, 0.0000] n.u. |
| PC | V2 | 0.0186 | 0.0185 | 0.0188 | −0.0140 [−0.0239, −0.0045] unter | −0.0051 [−0.0153, 0.0055] n.u. | −0.0070 [−0.0172, 0.0035] n.u. |
| PC | LOC | 0.0040 | 0.0035 | 0.0036 | −0.0299 [−0.0387, −0.0218] unter | −0.0148 [−0.0247, −0.0057] unter | −0.0153 [−0.0251, −0.0064] unter |
| PC | IT | 0.0085 | 0.0083 | 0.0082 | −0.0262 [−0.0346, −0.0186] unter | −0.0128 [−0.0216, −0.0041] unter | −0.0132 [−0.0217, −0.0049] unter |
| STDP | V1 | 0.0440 | 0.0445 | 0.0443 | −0.0131 [−0.0251, −0.0003] unter | −0.0069 [−0.0202, 0.0068] n.u. | −0.0096 [−0.0221, 0.0040] n.u. |
| STDP | V2 | 0.0237 | 0.0238 | 0.0233 | −0.0089 [−0.0189, 0.0007] n.u. | 0.0002 [−0.0101, 0.0104] n.u. | −0.0026 [−0.0128, 0.0076] n.u. |
| STDP | LOC | 0.0038 | 0.0035 | 0.0035 | −0.0301 [−0.0388, −0.0218] unter | −0.0147 [−0.0239, −0.0059] unter | −0.0154 [−0.0247, −0.0066] unter |
| STDP | IT | 0.0073 | 0.0070 | 0.0077 | −0.0274 [−0.0357, −0.0199] unter | −0.0141 [−0.0226, −0.0059] unter | −0.0137 [−0.0219, −0.0057] unter |

### Mittel-RDM-Konvention (Referenzsatz)

| Regel | ROI | alle Paare | Cross-Run |
|---|---|---|---|
| Random | V1 | 0.0755 | 0.0767 |
| Random | V2 | 0.0433 | 0.0435 |
| Random | LOC | −0.0051 | −0.0071 |
| Random | IT | 0.0078 | 0.0063 |
| BP | V1 | 0.0335 | 0.0363 |
| BP | V2 | 0.0185 | 0.0202 |
| BP | LOC | 0.0116 | 0.0124 |
| BP | IT | 0.0133 | 0.0154 |
| FA | V1 | 0.0117 | 0.0137 |
| FA | V2 | 0.0039 | 0.0051 |
| FA | LOC | 0.0056 | 0.0063 |
| FA | IT | 0.0116 | 0.0141 |
| PC | V1 | 0.0561 | 0.0565 |
| PC | V2 | 0.0279 | 0.0280 |
| PC | LOC | 0.0060 | 0.0050 |
| PC | IT | 0.0136 | 0.0137 |
| STDP | V1 | 0.0641 | 0.0648 |
| STDP | V2 | 0.0358 | 0.0361 |
| STDP | LOC | 0.0058 | 0.0051 |
| STDP | IT | 0.0119 | 0.0114 |

### Modell − lower bound, pro Subject (bnfix)

| Regel | ROI | ρ alle Paare (publiziert) | ρ Cross-Run (primär) | ρ Cross-Run blockbereinigt | Δ alle Paare (publiziert) [CI] | Δ Cross-Run (primär) [CI] | Δ Cross-Run blockbereinigt [CI] |
|---|---|---|---|---|---|---|---|
| Random | V1 | 0.0526 | 0.0532 | 0.0534 | −0.0045 [−0.0194, 0.0100] n.u. | 0.0017 [−0.0137, 0.0170] n.u. | −0.0006 [−0.0150, 0.0150] n.u. |
| Random | V2 | 0.0289 | 0.0288 | 0.0282 | −0.0037 [−0.0157, 0.0075] n.u. | 0.0052 [−0.0069, 0.0170] n.u. | 0.0024 [−0.0090, 0.0140] n.u. |
| Random | LOC | −0.0021 | −0.0030 | −0.0049 | −0.0359 [−0.0454, −0.0275] unter | −0.0213 [−0.0325, −0.0118] unter | −0.0238 [−0.0342, −0.0145] unter |
| Random | IT | 0.0048 | 0.0040 | 0.0029 | −0.0299 [−0.0391, −0.0214] unter | −0.0171 [−0.0268, −0.0082] unter | −0.0184 [−0.0276, −0.0092] unter |
| BP | V1 | 0.0207 | 0.0224 | 0.0207 | −0.0364 [−0.0502, −0.0232] unter | −0.0291 [−0.0432, −0.0142] unter | −0.0333 [−0.0460, −0.0188] unter |
| BP | V2 | 0.0113 | 0.0123 | 0.0104 | −0.0213 [−0.0318, −0.0107] unter | −0.0113 [−0.0233, −0.0000] unter | −0.0154 [−0.0260, −0.0046] unter |
| BP | LOC | 0.0075 | 0.0080 | 0.0076 | −0.0264 [−0.0360, −0.0169] unter | −0.0102 [−0.0205, −0.0003] unter | −0.0113 [−0.0211, −0.0023] unter |
| BP | IT | 0.0083 | 0.0094 | 0.0090 | −0.0264 [−0.0345, −0.0184] unter | −0.0117 [−0.0199, −0.0034] unter | −0.0123 [−0.0201, −0.0042] unter |
| FA | V1 | 0.0073 | 0.0086 | 0.0063 | −0.0497 [−0.0648, −0.0353] unter | −0.0428 [−0.0589, −0.0276] unter | −0.0477 [−0.0620, −0.0320] unter |
| FA | V2 | 0.0028 | 0.0036 | 0.0015 | −0.0298 [−0.0408, −0.0195] unter | −0.0200 [−0.0322, −0.0087] unter | −0.0243 [−0.0352, −0.0135] unter |
| FA | LOC | 0.0033 | 0.0036 | 0.0026 | −0.0305 [−0.0400, −0.0221] unter | −0.0146 [−0.0248, −0.0056] unter | −0.0163 [−0.0261, −0.0075] unter |
| FA | IT | 0.0066 | 0.0078 | 0.0063 | −0.0281 [−0.0373, −0.0199] unter | −0.0133 [−0.0227, −0.0046] unter | −0.0150 [−0.0237, −0.0065] unter |
| PC | V1 | 0.0106 | 0.0119 | 0.0096 | −0.0464 [−0.0616, −0.0321] unter | −0.0395 [−0.0549, −0.0241] unter | −0.0444 [−0.0582, −0.0291] unter |
| PC | V2 | 0.0050 | 0.0057 | 0.0036 | −0.0276 [−0.0385, −0.0171] unter | −0.0179 [−0.0297, −0.0066] unter | −0.0222 [−0.0328, −0.0113] unter |
| PC | LOC | 0.0039 | 0.0043 | 0.0029 | −0.0300 [−0.0387, −0.0216] unter | −0.0140 [−0.0240, −0.0053] unter | −0.0159 [−0.0256, −0.0075] unter |
| PC | IT | 0.0068 | 0.0077 | 0.0066 | −0.0279 [−0.0366, −0.0194] unter | −0.0134 [−0.0225, −0.0049] unter | −0.0147 [−0.0235, −0.0064] unter |
| STDP | V1 | 0.0252 | 0.0265 | 0.0242 | −0.0319 [−0.0460, −0.0187] unter | −0.0249 [−0.0389, −0.0105] unter | −0.0298 [−0.0430, −0.0155] unter |
| STDP | V2 | 0.0138 | 0.0146 | 0.0123 | −0.0188 [−0.0291, −0.0084] unter | −0.0090 [−0.0208, 0.0023] n.u. | −0.0135 [−0.0243, −0.0030] unter |
| STDP | LOC | 0.0037 | 0.0043 | 0.0032 | −0.0301 [−0.0395, −0.0215] unter | −0.0140 [−0.0238, −0.0049] unter | −0.0156 [−0.0249, −0.0068] unter |
| STDP | IT | 0.0080 | 0.0087 | 0.0079 | −0.0266 [−0.0355, −0.0191] unter | −0.0124 [−0.0213, −0.0041] unter | −0.0135 [−0.0223, −0.0055] unter |

### Mittel-RDM-Konvention (bnfix)

| Regel | ROI | alle Paare | Cross-Run |
|---|---|---|---|
| Random | V1 | 0.0755 | 0.0767 |
| Random | V2 | 0.0433 | 0.0435 |
| Random | LOC | −0.0051 | −0.0071 |
| Random | IT | 0.0078 | 0.0063 |
| BP | V1 | 0.0314 | 0.0341 |
| BP | V2 | 0.0169 | 0.0186 |
| BP | LOC | 0.0126 | 0.0133 |
| BP | IT | 0.0127 | 0.0149 |
| FA | V1 | 0.0115 | 0.0135 |
| FA | V2 | 0.0040 | 0.0052 |
| FA | LOC | 0.0056 | 0.0063 |
| FA | IT | 0.0115 | 0.0140 |
| PC | V1 | 0.0163 | 0.0183 |
| PC | V2 | 0.0075 | 0.0085 |
| PC | LOC | 0.0062 | 0.0070 |
| PC | IT | 0.0109 | 0.0130 |
| STDP | V1 | 0.0372 | 0.0393 |
| STDP | V2 | 0.0207 | 0.0220 |
| STDP | LOC | 0.0057 | 0.0065 |
| STDP | IT | 0.0131 | 0.0147 |

### Rangfolge der Regeln pro ROI (Kendall τ gegen alle Paare)

| Satz | ROI | Konvention | Variante | Rangfolge | τ vs alle Paare |
|---|---|---|---|---|---|
| Referenzsatz | V1 | meanrdm | alle Paare (publiziert) | Random > STDP > PC > BP > FA | 1.00 |
| Referenzsatz | V1 | meanrdm | Cross-Run (primär) | Random > STDP > PC > BP > FA | 1.00 |
| Referenzsatz | V1 | persub | alle Paare (publiziert) | Random > STDP > PC > BP > FA | 1.00 |
| Referenzsatz | V1 | persub | Cross-Run (primär) | Random > STDP > PC > BP > FA | 1.00 |
| Referenzsatz | V1 | persub | Cross-Run blockbereinigt | Random > STDP > PC > BP > FA | 1.00 |
| Referenzsatz | V2 | meanrdm | alle Paare (publiziert) | Random > STDP > PC > BP > FA | 1.00 |
| Referenzsatz | V2 | meanrdm | Cross-Run (primär) | Random > STDP > PC > BP > FA | 1.00 |
| Referenzsatz | V2 | persub | alle Paare (publiziert) | Random > STDP > PC > BP > FA | 1.00 |
| Referenzsatz | V2 | persub | Cross-Run (primär) | Random > STDP > PC > BP > FA | 1.00 |
| Referenzsatz | V2 | persub | Cross-Run blockbereinigt | Random > STDP > PC > BP > FA | 1.00 |
| Referenzsatz | LOC | meanrdm | alle Paare (publiziert) | BP > PC > STDP > FA > Random | 1.00 |
| Referenzsatz | LOC | meanrdm | Cross-Run (primär) | BP > FA > STDP > PC > Random | 0.40 |
| Referenzsatz | LOC | persub | alle Paare (publiziert) | BP > PC > STDP > FA > Random | 1.00 |
| Referenzsatz | LOC | persub | Cross-Run (primär) | BP > FA > STDP > PC > Random | 0.40 |
| Referenzsatz | LOC | persub | Cross-Run blockbereinigt | BP > PC > STDP > FA > Random | 1.00 |
| Referenzsatz | IT | meanrdm | alle Paare (publiziert) | PC > BP > STDP > FA > Random | 1.00 |
| Referenzsatz | IT | meanrdm | Cross-Run (primär) | BP > FA > PC > STDP > Random | 0.40 |
| Referenzsatz | IT | persub | alle Paare (publiziert) | BP > PC > STDP > FA > Random | 1.00 |
| Referenzsatz | IT | persub | Cross-Run (primär) | BP > PC > FA > STDP > Random | 0.80 |
| Referenzsatz | IT | persub | Cross-Run blockbereinigt | BP > PC > STDP > FA > Random | 1.00 |
| bnfix | V1 | meanrdm | alle Paare (publiziert) | Random > STDP > BP > PC > FA | 1.00 |
| bnfix | V1 | meanrdm | Cross-Run (primär) | Random > STDP > BP > PC > FA | 1.00 |
| bnfix | V1 | persub | alle Paare (publiziert) | Random > STDP > BP > PC > FA | 1.00 |
| bnfix | V1 | persub | Cross-Run (primär) | Random > STDP > BP > PC > FA | 1.00 |
| bnfix | V1 | persub | Cross-Run blockbereinigt | Random > STDP > BP > PC > FA | 1.00 |
| bnfix | V2 | meanrdm | alle Paare (publiziert) | Random > STDP > BP > PC > FA | 1.00 |
| bnfix | V2 | meanrdm | Cross-Run (primär) | Random > STDP > BP > PC > FA | 1.00 |
| bnfix | V2 | persub | alle Paare (publiziert) | Random > STDP > BP > PC > FA | 1.00 |
| bnfix | V2 | persub | Cross-Run (primär) | Random > STDP > BP > PC > FA | 1.00 |
| bnfix | V2 | persub | Cross-Run blockbereinigt | Random > STDP > BP > PC > FA | 1.00 |
| bnfix | LOC | meanrdm | alle Paare (publiziert) | BP > PC > STDP > FA > Random | 1.00 |
| bnfix | LOC | meanrdm | Cross-Run (primär) | BP > PC > STDP > FA > Random | 1.00 |
| bnfix | LOC | persub | alle Paare (publiziert) | BP > PC > STDP > FA > Random | 1.00 |
| bnfix | LOC | persub | Cross-Run (primär) | BP > PC > STDP > FA > Random | 1.00 |
| bnfix | LOC | persub | Cross-Run blockbereinigt | BP > STDP > PC > FA > Random | 0.80 |
| bnfix | IT | meanrdm | alle Paare (publiziert) | STDP > BP > FA > PC > Random | 1.00 |
| bnfix | IT | meanrdm | Cross-Run (primär) | BP > STDP > FA > PC > Random | 0.80 |
| bnfix | IT | persub | alle Paare (publiziert) | BP > STDP > PC > FA > Random | 1.00 |
| bnfix | IT | persub | Cross-Run (primär) | BP > STDP > FA > PC > Random | 0.80 |
| bnfix | IT | persub | Cross-Run blockbereinigt | BP > STDP > PC > FA > Random | 1.00 |

### Paper-1-Aussagen × Variante (Referenzsatz)

Δρ = erste − zweite Bedingung, 95%-CI aus dem Stimulus-Bootstrap auf denselben Resamples.

| Aussage | Konvention | alle Paare | Cross-Run | Urteil | blockbereinigt | Urteil |
|---|---|---|---|---|---|---|
| Random > BP an V1 | meanrdm | 0.0420 [0.0241, 0.0594] | 0.0404 [0.0218, 0.0581] | hält | nicht definiert | – |
| Random > BP an V1 | persub | 0.0304 [0.0184, 0.0424] | 0.0292 [0.0164, 0.0412] | hält | 0.0313 [0.0196, 0.0422] | hält |
| Random > BP an V2 | meanrdm | 0.0249 [0.0107, 0.0397] | 0.0233 [0.0078, 0.0383] | hält | nicht definiert | – |
| Random > BP an V2 | persub | 0.0166 [0.0074, 0.0260] | 0.0155 [0.0058, 0.0252] | hält | 0.0170 [0.0083, 0.0256] | hält |
| BP > Random an LOC | meanrdm | 0.0167 [0.0026, 0.0314] | 0.0195 [0.0037, 0.0363] | hält | nicht definiert | – |
| BP > Random an LOC | persub | 0.0090 [0.0010, 0.0180] | 0.0105 [0.0017, 0.0202] | hält | 0.0123 [0.0045, 0.0209] | ändert Grösse |
| BP vs FA an LOC n.s. | meanrdm | 0.0060 [−0.0035, 0.0161] | 0.0060 [−0.0044, 0.0171] | hält | nicht definiert | – |
| BP vs FA an LOC n.s. | persub | 0.0036 [−0.0015, 0.0093] | 0.0039 [−0.0018, 0.0100] | hält | 0.0048 [−0.0003, 0.0104] | ändert Grösse |
| BP vs PC an LOC n.s. | meanrdm | 0.0056 [−0.0076, 0.0180] | 0.0074 [−0.0064, 0.0226] | ändert Grösse | nicht definiert | – |
| BP vs PC an LOC n.s. | persub | 0.0030 [−0.0044, 0.0107] | 0.0041 [−0.0039, 0.0127] | ändert Grösse | 0.0039 [−0.0029, 0.0115] | ändert Grösse |
| BP vs STDP an LOC n.s. | meanrdm | 0.0058 [−0.0055, 0.0170] | 0.0072 [−0.0047, 0.0203] | hält | nicht definiert | – |
| BP vs STDP an LOC n.s. | persub | 0.0032 [−0.0031, 0.0101] | 0.0040 [−0.0032, 0.0115] | hält | 0.0039 [−0.0019, 0.0105] | hält |
| FA < Random an V1 | meanrdm | 0.0637 [0.0449, 0.0828] | 0.0630 [0.0433, 0.0825] | hält | nicht definiert | – |
| FA < Random an V1 | persub | 0.0451 [0.0324, 0.0579] | 0.0445 [0.0313, 0.0578] | hält | 0.0469 [0.0334, 0.0590] | hält |
| FA < BP an V1 | meanrdm | 0.0217 [0.0143, 0.0296] | 0.0226 [0.0149, 0.0309] | hält | nicht definiert | – |
| FA < BP an V1 | persub | 0.0147 [0.0098, 0.0199] | 0.0152 [0.0102, 0.0206] | hält | 0.0156 [0.0103, 0.0210] | hält |
| FA < PC an V1 | meanrdm | 0.0443 [0.0305, 0.0588] | 0.0428 [0.0284, 0.0574] | hält | nicht definiert | – |
| FA < PC an V1 | persub | 0.0309 [0.0214, 0.0407] | 0.0300 [0.0197, 0.0398] | hält | 0.0331 [0.0233, 0.0422] | hält |
| FA < STDP an V1 | meanrdm | 0.0524 [0.0395, 0.0662] | 0.0511 [0.0375, 0.0654] | hält | nicht definiert | – |
| FA < STDP an V1 | persub | 0.0365 [0.0277, 0.0458] | 0.0358 [0.0267, 0.0451] | hält | 0.0379 [0.0286, 0.0465] | hält |
| FA < Random an V2 | meanrdm | 0.0394 [0.0243, 0.0552] | 0.0384 [0.0225, 0.0549] | hält | nicht definiert | – |
| FA < Random an V2 | persub | 0.0261 [0.0163, 0.0362] | 0.0253 [0.0151, 0.0360] | hält | 0.0267 [0.0172, 0.0360] | hält |
| FA < BP an V2 | meanrdm | 0.0145 [0.0087, 0.0211] | 0.0151 [0.0087, 0.0223] | hält | nicht definiert | – |
| FA < BP an V2 | persub | 0.0095 [0.0058, 0.0135] | 0.0098 [0.0060, 0.0142] | hält | 0.0097 [0.0059, 0.0140] | hält |
| FA < PC an V2 | meanrdm | 0.0240 [0.0118, 0.0356] | 0.0229 [0.0098, 0.0355] | hält | nicht definiert | – |
| FA < PC an V2 | persub | 0.0158 [0.0082, 0.0234] | 0.0150 [0.0068, 0.0229] | hält | 0.0173 [0.0100, 0.0245] | hält |
| FA < STDP an V2 | meanrdm | 0.0318 [0.0213, 0.0427] | 0.0310 [0.0198, 0.0428] | hält | nicht definiert | – |
| FA < STDP an V2 | persub | 0.0209 [0.0140, 0.0282] | 0.0202 [0.0131, 0.0277] | hält | 0.0217 [0.0147, 0.0285] | hält |
| IT konvergiert: Random vs BP n.s. | meanrdm | −0.0055 [−0.0190, 0.0076] | −0.0091 [−0.0245, 0.0062] | ändert Grösse | nicht definiert | – |
| IT konvergiert: Random vs BP n.s. | persub | −0.0041 [−0.0128, 0.0037] | −0.0060 [−0.0153, 0.0032] | ändert Grösse | −0.0068 [−0.0151, 0.0019] | ändert Grösse |
| IT konvergiert: Random vs FA n.s. | meanrdm | −0.0038 [−0.0155, 0.0072] | −0.0079 [−0.0210, 0.0048] | ändert Grösse | nicht definiert | – |
| IT konvergiert: Random vs FA n.s. | persub | −0.0018 [−0.0091, 0.0050] | −0.0039 [−0.0117, 0.0038] | ändert Grösse | −0.0034 [−0.0104, 0.0038] | ändert Grösse |
| IT konvergiert: Random vs PC n.s. | meanrdm | −0.0059 [−0.0151, 0.0030] | −0.0075 [−0.0180, 0.0023] | ändert Grösse | nicht definiert | – |
| IT konvergiert: Random vs PC n.s. | persub | −0.0037 [−0.0092, 0.0016] | −0.0044 [−0.0108, 0.0015] | hält | −0.0052 [−0.0112, 0.0004] | ändert Grösse |
| IT konvergiert: Random vs STDP n.s. | meanrdm | −0.0041 [−0.0125, 0.0037] | −0.0052 [−0.0145, 0.0038] | ändert Grösse | nicht definiert | – |
| IT konvergiert: Random vs STDP n.s. | persub | −0.0025 [−0.0076, 0.0021] | −0.0030 [−0.0087, 0.0021] | hält | −0.0047 [−0.0099, 0.0001] | ändert Grösse |
| IT konvergiert: BP vs FA n.s. | meanrdm | 0.0017 [−0.0074, 0.0106] | 0.0013 [−0.0086, 0.0113] | hält | nicht definiert | – |
| IT konvergiert: BP vs FA n.s. | persub | 0.0023 [−0.0032, 0.0075] | 0.0021 [−0.0039, 0.0084] | hält | 0.0034 [−0.0023, 0.0086] | ändert Grösse |
| IT konvergiert: BP vs PC n.s. | meanrdm | −0.0004 [−0.0112, 0.0092] | 0.0016 [−0.0099, 0.0123] | **kippt** | nicht definiert | – |
| IT konvergiert: BP vs PC n.s. | persub | 0.0005 [−0.0056, 0.0064] | 0.0016 [−0.0054, 0.0079] | ändert Grösse | 0.0015 [−0.0048, 0.0074] | ändert Grösse |
| IT konvergiert: BP vs STDP n.s. | meanrdm | 0.0014 [−0.0096, 0.0120] | 0.0040 [−0.0085, 0.0157] | ändert Grösse | nicht definiert | – |
| IT konvergiert: BP vs STDP n.s. | persub | 0.0017 [−0.0052, 0.0083] | 0.0030 [−0.0046, 0.0101] | ändert Grösse | 0.0020 [−0.0046, 0.0090] | hält |
| IT konvergiert: FA vs PC n.s. | meanrdm | −0.0021 [−0.0101, 0.0065] | 0.0004 [−0.0092, 0.0101] | **kippt** | nicht definiert | – |
| IT konvergiert: FA vs PC n.s. | persub | −0.0018 [−0.0066, 0.0033] | −0.0004 [−0.0062, 0.0055] | ändert Grösse | −0.0018 [−0.0068, 0.0030] | hält |
| IT konvergiert: FA vs STDP n.s. | meanrdm | −0.0003 [−0.0104, 0.0091] | 0.0027 [−0.0090, 0.0133] | **kippt** | nicht definiert | – |
| IT konvergiert: FA vs STDP n.s. | persub | −0.0006 [−0.0066, 0.0050] | 0.0009 [−0.0062, 0.0069] | **kippt** | −0.0013 [−0.0073, 0.0042] | ändert Grösse |
| IT konvergiert: PC vs STDP n.s. | meanrdm | 0.0018 [−0.0025, 0.0066] | 0.0023 [−0.0026, 0.0075] | ändert Grösse | nicht definiert | – |
| IT konvergiert: PC vs STDP n.s. | persub | 0.0012 [−0.0015, 0.0040] | 0.0013 [−0.0017, 0.0044] | hält | 0.0005 [−0.0023, 0.0034] | ändert Grösse |

### Paper-1-Aussagen × Variante (bnfix)

Δρ = erste − zweite Bedingung, 95%-CI aus dem Stimulus-Bootstrap auf denselben Resamples.

| Aussage | Konvention | alle Paare | Cross-Run | Urteil | blockbereinigt | Urteil |
|---|---|---|---|---|---|---|
| Random > BP an V1 | meanrdm | 0.0441 [0.0268, 0.0613] | 0.0426 [0.0245, 0.0597] | hält | nicht definiert | – |
| Random > BP an V1 | persub | 0.0319 [0.0202, 0.0440] | 0.0308 [0.0186, 0.0428] | hält | 0.0327 [0.0211, 0.0437] | hält |
| Random > BP an V2 | meanrdm | 0.0264 [0.0120, 0.0413] | 0.0249 [0.0099, 0.0398] | hält | nicht definiert | – |
| Random > BP an V2 | persub | 0.0176 [0.0086, 0.0269] | 0.0165 [0.0072, 0.0261] | hält | 0.0178 [0.0091, 0.0262] | hält |
| BP > Random an LOC | meanrdm | 0.0177 [0.0036, 0.0320] | 0.0205 [0.0047, 0.0372] | hält | nicht definiert | – |
| BP > Random an LOC | persub | 0.0096 [0.0014, 0.0181] | 0.0110 [0.0023, 0.0206] | hält | 0.0125 [0.0050, 0.0206] | ändert Grösse |
| BP vs FA an LOC n.s. | meanrdm | 0.0070 [−0.0019, 0.0167] | 0.0070 [−0.0030, 0.0176] | hält | nicht definiert | – |
| BP vs FA an LOC n.s. | persub | 0.0042 [−0.0009, 0.0095] | 0.0044 [−0.0011, 0.0102] | hält | 0.0050 [0.0001, 0.0102] | **kippt** |
| BP vs PC an LOC n.s. | meanrdm | 0.0064 [−0.0025, 0.0160] | 0.0063 [−0.0037, 0.0171] | hält | nicht definiert | – |
| BP vs PC an LOC n.s. | persub | 0.0036 [−0.0014, 0.0092] | 0.0037 [−0.0017, 0.0097] | hält | 0.0047 [−0.0001, 0.0099] | ändert Grösse |
| BP vs STDP an LOC n.s. | meanrdm | 0.0069 [−0.0020, 0.0166] | 0.0068 [−0.0029, 0.0177] | hält | nicht definiert | – |
| BP vs STDP an LOC n.s. | persub | 0.0038 [−0.0013, 0.0094] | 0.0037 [−0.0017, 0.0097] | hält | 0.0043 [−0.0007, 0.0096] | hält |
| FA < Random an V1 | meanrdm | 0.0639 [0.0449, 0.0832] | 0.0631 [0.0435, 0.0828] | hält | nicht definiert | – |
| FA < Random an V1 | persub | 0.0453 [0.0325, 0.0581] | 0.0446 [0.0313, 0.0580] | hält | 0.0471 [0.0336, 0.0591] | hält |
| FA < BP an V1 | meanrdm | 0.0199 [0.0132, 0.0269] | 0.0206 [0.0137, 0.0279] | hält | nicht definiert | – |
| FA < BP an V1 | persub | 0.0134 [0.0090, 0.0178] | 0.0138 [0.0092, 0.0186] | hält | 0.0144 [0.0097, 0.0193] | hält |
| FA < PC an V1 | meanrdm | 0.0048 [0.0017, 0.0078] | 0.0048 [0.0014, 0.0079] | hält | nicht definiert | – |
| FA < PC an V1 | persub | 0.0033 [0.0012, 0.0053] | 0.0033 [0.0011, 0.0054] | hält | 0.0033 [0.0010, 0.0053] | hält |
| FA < STDP an V1 | meanrdm | 0.0257 [0.0188, 0.0334] | 0.0258 [0.0187, 0.0334] | hält | nicht definiert | – |
| FA < STDP an V1 | persub | 0.0179 [0.0131, 0.0229] | 0.0179 [0.0130, 0.0229] | hält | 0.0179 [0.0128, 0.0229] | hält |
| FA < Random an V2 | meanrdm | 0.0394 [0.0241, 0.0553] | 0.0383 [0.0222, 0.0549] | hält | nicht definiert | – |
| FA < Random an V2 | persub | 0.0260 [0.0162, 0.0362] | 0.0252 [0.0150, 0.0361] | hält | 0.0267 [0.0172, 0.0361] | hält |
| FA < BP an V2 | meanrdm | 0.0129 [0.0077, 0.0186] | 0.0134 [0.0076, 0.0198] | hält | nicht definiert | – |
| FA < BP an V2 | persub | 0.0084 [0.0052, 0.0120] | 0.0087 [0.0053, 0.0125] | hält | 0.0089 [0.0054, 0.0128] | hält |
| FA < PC an V2 | meanrdm | 0.0035 [0.0011, 0.0059] | 0.0033 [0.0008, 0.0057] | hält | nicht definiert | – |
| FA < PC an V2 | persub | 0.0022 [0.0006, 0.0037] | 0.0021 [0.0005, 0.0037] | hält | 0.0021 [0.0005, 0.0035] | hält |
| FA < STDP an V2 | meanrdm | 0.0167 [0.0118, 0.0222] | 0.0168 [0.0115, 0.0226] | hält | nicht definiert | – |
| FA < STDP an V2 | persub | 0.0110 [0.0077, 0.0144] | 0.0110 [0.0077, 0.0146] | hält | 0.0108 [0.0075, 0.0142] | hält |
| IT konvergiert: Random vs BP n.s. | meanrdm | −0.0049 [−0.0187, 0.0082] | −0.0087 [−0.0234, 0.0058] | ändert Grösse | nicht definiert | – |
| IT konvergiert: Random vs BP n.s. | persub | −0.0035 [−0.0120, 0.0045] | −0.0054 [−0.0144, 0.0036] | ändert Grösse | −0.0061 [−0.0144, 0.0023] | ändert Grösse |
| IT konvergiert: Random vs FA n.s. | meanrdm | −0.0037 [−0.0158, 0.0074] | −0.0078 [−0.0212, 0.0048] | ändert Grösse | nicht definiert | – |
| IT konvergiert: Random vs FA n.s. | persub | −0.0018 [−0.0091, 0.0051] | −0.0039 [−0.0118, 0.0039] | ändert Grösse | −0.0034 [−0.0104, 0.0039] | ändert Grösse |
| IT konvergiert: Random vs PC n.s. | meanrdm | −0.0032 [−0.0145, 0.0082] | −0.0067 [−0.0195, 0.0061] | ändert Grösse | nicht definiert | – |
| IT konvergiert: Random vs PC n.s. | persub | −0.0019 [−0.0088, 0.0050] | −0.0037 [−0.0115, 0.0041] | ändert Grösse | −0.0037 [−0.0103, 0.0032] | ändert Grösse |
| IT konvergiert: Random vs STDP n.s. | meanrdm | −0.0053 [−0.0167, 0.0052] | −0.0085 [−0.0208, 0.0040] | ändert Grösse | nicht definiert | – |
| IT konvergiert: Random vs STDP n.s. | persub | −0.0032 [−0.0102, 0.0031] | −0.0048 [−0.0123, 0.0028] | ändert Grösse | −0.0049 [−0.0114, 0.0016] | ändert Grösse |
| IT konvergiert: BP vs FA n.s. | meanrdm | 0.0012 [−0.0073, 0.0094] | 0.0009 [−0.0084, 0.0103] | ändert Grösse | nicht definiert | – |
| IT konvergiert: BP vs FA n.s. | persub | 0.0017 [−0.0035, 0.0068] | 0.0016 [−0.0041, 0.0075] | hält | 0.0027 [−0.0025, 0.0079] | ändert Grösse |
| IT konvergiert: BP vs PC n.s. | meanrdm | 0.0017 [−0.0063, 0.0100] | 0.0019 [−0.0070, 0.0108] | hält | nicht definiert | – |
| IT konvergiert: BP vs PC n.s. | persub | 0.0016 [−0.0031, 0.0065] | 0.0017 [−0.0036, 0.0072] | hält | 0.0024 [−0.0023, 0.0075] | ändert Grösse |
| IT konvergiert: BP vs STDP n.s. | meanrdm | −0.0004 [−0.0081, 0.0067] | 0.0002 [−0.0080, 0.0080] | **kippt** | nicht definiert | – |
| IT konvergiert: BP vs STDP n.s. | persub | 0.0003 [−0.0045, 0.0049] | 0.0006 [−0.0043, 0.0056] | ändert Grösse | 0.0012 [−0.0035, 0.0058] | ändert Grösse |
| IT konvergiert: FA vs PC n.s. | meanrdm | 0.0005 [−0.0052, 0.0067] | 0.0011 [−0.0058, 0.0086] | ändert Grösse | nicht definiert | – |
| IT konvergiert: FA vs PC n.s. | persub | −0.0001 [−0.0036, 0.0036] | 0.0001 [−0.0041, 0.0044] | **kippt** | −0.0003 [−0.0039, 0.0034] | ändert Grösse |
| IT konvergiert: FA vs STDP n.s. | meanrdm | −0.0016 [−0.0077, 0.0040] | −0.0007 [−0.0075, 0.0059] | ändert Grösse | nicht definiert | – |
| IT konvergiert: FA vs STDP n.s. | persub | −0.0014 [−0.0049, 0.0020] | −0.0009 [−0.0050, 0.0029] | ändert Grösse | −0.0016 [−0.0053, 0.0020] | hält |
| IT konvergiert: PC vs STDP n.s. | meanrdm | −0.0022 [−0.0077, 0.0028] | −0.0018 [−0.0075, 0.0037] | hält | nicht definiert | – |
| IT konvergiert: PC vs STDP n.s. | persub | −0.0013 [−0.0046, 0.0018] | −0.0011 [−0.0045, 0.0022] | hält | −0.0012 [−0.0043, 0.0018] | hält |

Hinweis: Die Signifikanzangaben des Papers stammen aus Permutationstests mit FDR. Hier wird nur geprüft, ob Vorzeichen und Stimulus-Bootstrap-Signifikanz zwischen den Varianten gleich bleiben. Wo das CI schon auf *allen Paaren* von der Paper-Aussage abweicht (z. B. „n.s.“, aber CI schliesst 0 aus), ist das ein Befund zur Aussage selbst und unabhängig von der Variante.

**Art der 7 „kippt“-Zellen (beide Modellsätze).** 6 davon sind reine Vorzeichenwechsel eines Punktschätzers nahe 0 bei einer „n.s.“-Aussage: Das CI schliesst 0 in *beiden* Varianten ein (grösster Betrag 0.0027). Die Aussage „n.s.“ bleibt dort inhaltlich bestehen; das Kriterium wertet sie wie vorab festgelegt trotzdem als „kippt“. Dasselbe gilt für „ändert Grösse“ bei „n.s.“-Aussagen: Eine relative Änderung über 25 % eines Werts nahe 0 sagt wenig. Echte Signifikanzwechsel (1): bnfix, BP vs FA an LOC n.s. (persub, Cross-Run blockbereinigt): 0.0042 [−0.0009, 0.0095] → 0.0050 [0.0001, 0.0102].

## 4. arXiv:2608.12408: Kerneffekte über die Auflösungen (bnfix, Seed-CI)

Validierung: *alle Paare*, Mittel-RDM reproduziert `bnfix_sweep.csv` (max. Abweichung 5e-07, 120 Werte).

**E1 Random-BP V1**

| px | Konvention | alle Paare | Cross-Run | Urteil | blockbereinigt | Urteil |
|---|---|---|---|---|---|---|
| 32 | meanrdm | −0.0009 [−0.0212, 0.0193] (3/5) | −0.0025 [−0.0225, 0.0176] (3/5) | ändert Grösse | nicht definiert | – |
| 32 | persub | 0.0002 [−0.0141, 0.0145] (3/5) | −0.0009 [−0.0151, 0.0132] (3/5) | **kippt** | 0.0017 [−0.0130, 0.0164] (3/5) | ändert Grösse |
| 64 | meanrdm | 0.0137 [−0.0059, 0.0333] (3/5) | 0.0123 [−0.0071, 0.0316] (3/5) | hält | nicht definiert | – |
| 64 | persub | 0.0108 [−0.0030, 0.0246] (4/5) | 0.0097 [−0.0039, 0.0233] (3/5) | hält | 0.0118 [−0.0025, 0.0261] (4/5) | hält |
| 96 | meanrdm | 0.0250 [0.0065, 0.0436] (5/5) | 0.0236 [0.0053, 0.0418] (5/5) | hält | nicht definiert | – |
| 96 | persub | 0.0187 [0.0056, 0.0318] (5/5) | 0.0176 [0.0048, 0.0305] (5/5) | hält | 0.0196 [0.0060, 0.0331] (5/5) | hält |
| 128 | meanrdm | 0.0326 [0.0147, 0.0504] (5/5) | 0.0311 [0.0136, 0.0486] (5/5) | hält | nicht definiert | – |
| 128 | persub | 0.0240 [0.0114, 0.0366] (5/5) | 0.0229 [0.0105, 0.0353] (5/5) | hält | 0.0248 [0.0118, 0.0379] (5/5) | hält |
| 160 | meanrdm | 0.0370 [0.0197, 0.0544] (5/5) | 0.0356 [0.0186, 0.0525] (5/5) | hält | nicht definiert | – |
| 160 | persub | 0.0271 [0.0148, 0.0394] (5/5) | 0.0260 [0.0140, 0.0380] (5/5) | hält | 0.0279 [0.0152, 0.0406] (5/5) | hält |
| 224 | meanrdm | 0.0441 [0.0274, 0.0607] (5/5) | 0.0426 [0.0263, 0.0589] (5/5) | hält | nicht definiert | – |
| 224 | persub | 0.0319 [0.0201, 0.0437] (5/5) | 0.0308 [0.0193, 0.0424] (5/5) | hält | 0.0327 [0.0205, 0.0449] (5/5) | hält |

**E2 BP-Random LOC**

| px | Konvention | alle Paare | Cross-Run | Urteil | blockbereinigt | Urteil |
|---|---|---|---|---|---|---|
| 32 | meanrdm | 0.0190 [0.0152, 0.0228] (5/5) | 0.0212 [0.0168, 0.0257] (5/5) | hält | nicht definiert | – |
| 32 | persub | 0.0109 [0.0087, 0.0132] (5/5) | 0.0120 [0.0093, 0.0147] (5/5) | hält | 0.0131 [0.0110, 0.0151] (5/5) | hält |
| 64 | meanrdm | 0.0223 [0.0193, 0.0253] (5/5) | 0.0258 [0.0222, 0.0293] (5/5) | hält | nicht definiert | – |
| 64 | persub | 0.0124 [0.0105, 0.0142] (5/5) | 0.0141 [0.0119, 0.0163] (5/5) | hält | 0.0153 [0.0138, 0.0169] (5/5) | hält |
| 96 | meanrdm | 0.0206 [0.0183, 0.0230] (5/5) | 0.0240 [0.0210, 0.0270] (5/5) | hält | nicht definiert | – |
| 96 | persub | 0.0113 [0.0098, 0.0128] (5/5) | 0.0130 [0.0111, 0.0149] (5/5) | hält | 0.0143 [0.0129, 0.0157] (5/5) | ändert Grösse |
| 128 | meanrdm | 0.0193 [0.0171, 0.0216] (5/5) | 0.0224 [0.0196, 0.0252] (5/5) | hält | nicht definiert | – |
| 128 | persub | 0.0105 [0.0091, 0.0119] (5/5) | 0.0121 [0.0103, 0.0139] (5/5) | hält | 0.0135 [0.0120, 0.0150] (5/5) | ändert Grösse |
| 160 | meanrdm | 0.0191 [0.0169, 0.0214] (5/5) | 0.0221 [0.0193, 0.0249] (5/5) | hält | nicht definiert | – |
| 160 | persub | 0.0104 [0.0090, 0.0118] (5/5) | 0.0119 [0.0101, 0.0137] (5/5) | hält | 0.0133 [0.0118, 0.0148] (5/5) | ändert Grösse |
| 224 | meanrdm | 0.0177 [0.0155, 0.0199] (5/5) | 0.0205 [0.0177, 0.0232] (5/5) | hält | nicht definiert | – |
| 224 | persub | 0.0096 [0.0081, 0.0110] (5/5) | 0.0110 [0.0092, 0.0128] (5/5) | hält | 0.0125 [0.0109, 0.0141] (5/5) | ändert Grösse |

**E3 Luminanz-Schranke an V1** (Paper: Luminanz 0.074 ≈ untrainiertes Netz 0.075). Kriterium „hält“: Die Luminanz liegt innerhalb von 10 % des untrainierten Netzes bei 224 px, wie auf allen Paaren.

| Variante | Konvention | Luminanz | untrainiert 224 px | Verhältnis | Urteil |
|---|---|---|---|---|---|
| alle Paare (publiziert) | persub | 0.0522 | 0.0526 | 0.99 | hält |
| alle Paare (publiziert) | meanrdm | 0.0745 | 0.0755 | 0.99 | hält |
| Cross-Run (primär) | persub | 0.0536 | 0.0532 | 1.01 | hält |
| Cross-Run (primär) | meanrdm | 0.0764 | 0.0767 | 1.00 | hält |
| Cross-Run blockbereinigt | persub | 0.0500 | 0.0534 | 0.94 | hält |

## 5. Wo sich nichts ändert

**Paper 1 (Referenzsatz):** 25 von 48 Aussage × Konvention-Zellen halten in beiden robusten Varianten ohne Grössenänderung über 25 %. Alle anderen: BP > Random an LOC (persub: Cross-Run hält, blockbereinigt ändert Grösse); BP vs FA an LOC n.s. (persub: Cross-Run hält, blockbereinigt ändert Grösse); BP vs PC an LOC n.s. (meanrdm: Cross-Run ändert Grösse, blockbereinigt –); BP vs PC an LOC n.s. (persub: Cross-Run ändert Grösse, blockbereinigt ändert Grösse); IT konvergiert: Random vs BP n.s. (meanrdm: Cross-Run ändert Grösse, blockbereinigt –); IT konvergiert: Random vs BP n.s. (persub: Cross-Run ändert Grösse, blockbereinigt ändert Grösse); IT konvergiert: Random vs FA n.s. (meanrdm: Cross-Run ändert Grösse, blockbereinigt –); IT konvergiert: Random vs FA n.s. (persub: Cross-Run ändert Grösse, blockbereinigt ändert Grösse); IT konvergiert: Random vs PC n.s. (meanrdm: Cross-Run ändert Grösse, blockbereinigt –); IT konvergiert: Random vs PC n.s. (persub: Cross-Run hält, blockbereinigt ändert Grösse); IT konvergiert: Random vs STDP n.s. (meanrdm: Cross-Run ändert Grösse, blockbereinigt –); IT konvergiert: Random vs STDP n.s. (persub: Cross-Run hält, blockbereinigt ändert Grösse); IT konvergiert: BP vs FA n.s. (persub: Cross-Run hält, blockbereinigt ändert Grösse); IT konvergiert: BP vs PC n.s. (meanrdm: Cross-Run **kippt**, blockbereinigt –); IT konvergiert: BP vs PC n.s. (persub: Cross-Run ändert Grösse, blockbereinigt ändert Grösse); IT konvergiert: BP vs STDP n.s. (meanrdm: Cross-Run ändert Grösse, blockbereinigt –); IT konvergiert: BP vs STDP n.s. (persub: Cross-Run ändert Grösse, blockbereinigt hält); IT konvergiert: FA vs PC n.s. (meanrdm: Cross-Run **kippt**, blockbereinigt –); IT konvergiert: FA vs PC n.s. (persub: Cross-Run ändert Grösse, blockbereinigt hält); IT konvergiert: FA vs STDP n.s. (meanrdm: Cross-Run **kippt**, blockbereinigt –); IT konvergiert: FA vs STDP n.s. (persub: Cross-Run **kippt**, blockbereinigt ändert Grösse); IT konvergiert: PC vs STDP n.s. (meanrdm: Cross-Run ändert Grösse, blockbereinigt –); IT konvergiert: PC vs STDP n.s. (persub: Cross-Run hält, blockbereinigt ändert Grösse).

**Paper 2:** E1 Random-BP V1: Ausnahmen: 32 px meanrdm Cross-Run (primär): ändert Grösse; 32 px persub Cross-Run (primär): **kippt**; 32 px persub Cross-Run blockbereinigt: ändert Grösse. E2 BP-Random LOC: Ausnahmen: 96 px persub Cross-Run blockbereinigt: ändert Grösse; 128 px persub Cross-Run blockbereinigt: ändert Grösse; 160 px persub Cross-Run blockbereinigt: ändert Grösse; 224 px persub Cross-Run blockbereinigt: ändert Grösse. E3 Luminanz-Schranke: hält in allen Varianten.

**Was sich ändert:** der lower bound. V1 0.057 → 0.051 (-10%), V2 0.033 → 0.024 (-28%), LOC 0.034 → 0.018 (-46%), IT 0.035 → 0.021 (-39%). Dadurch verschiebt sich die Lage der Modelle relativ zum Bound (§3, Spalten Δ). Die Modellwerte selbst ändern sich kaum.

## 6. Abweichungen von der Vorab-Planung

1. **runz → Cross-Run.** Vorab festgelegt war die Voxel-z-Normierung pro Run als Primärvariante. Sie hat beide vorab definierten Kriterien verfehlt (K1: Δ = +1/81 in allen Zellen, das −1/(n−1)-Artefakt des Demeaning; K2: 01–02 auf allen Paaren ≠ Cross-Run, siehe `results/runz/RUNZ_REPORT.md`). Primärvariante ist jetzt Cross-Run auf den alten RDMs, die Blockbereinigung ist Robustheitscheck. **Die runz-Effektzahlen (Schritt 5, z. B. V1-Gap 224 px 0.009 statt 0.044) wurden gesehen, bevor diese Entscheidung fiel.** Die Entscheidung ist mit den Kriterien aus Schritt 2 begründet, nicht mit den Effektzahlen. Unabhängig ist sie dennoch nicht.
2. **Schritt 1c ist Diagnostik mit Effektcharakter.** Der Luminanz-Fit ist zugleich eine Kernaussage von arXiv:2608.12408 (E3) und STOPP-Kriterium. Er wurde vor allen Modellzahlen gerechnet, gibt aber schon Auskunft über ein Ergebnis.
3. **Paarmenge.** Statt der Cross-Run-Paare pro Subject-Paar wird für alle Analysen die Schnittmenge über alle drei Subjects verwendet (Korrektur zum Auftrag).
4. **Blockbereinigung nur pro Subject.** Für die Mittel-RDM-Konvention ist sie nicht definiert (siehe oben).
5. **Run-Offset aus Test-Trials** wurde nicht umgesetzt (Entscheidung: der Schätzfehler würde positive Korrelation innerhalb des Runs erzeugen).

## 7. Dateien

```
py -3 scripts/crossrun/step1_diagnostics.py   # Lag, Run-Paare (10! exakt), Luminanz, Übereinstimmung
py -3 scripts/crossrun/step2_bootstrap.py draws --start -1 --stop 500    # Bootstrap Teil 1 (~1 h, 10 Prozesse)
py -3 scripts/crossrun/step2_bootstrap.py draws --start 500 --stop 1000  # Teil 2 (~1 h); wiederaufnehmbar
py -3 scripts/crossrun/step2_bootstrap.py perm       # Permutations-Null der Bounds (~7 min)
py -3 scripts/crossrun/step2_bootstrap.py finalize   # Zusammenfassung + Validierung gegen v2
py -3 scripts/crossrun/step2_effects.py       # Kerneffekte 2608.12408 über Auflösungen
py -3 scripts/crossrun/report.py
```

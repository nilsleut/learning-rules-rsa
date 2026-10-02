"""Step D: check every number in paper v3 against its macro source.

1. Every \\NV{key} used in the .tex sources exists in the manifest.
2. Every manifest entry is a single rounding (ROUND_HALF_UP) of its full-precision value.
3. Entries whose source is results/crossrun/step2_summary.csv are re-read from that file by
   their selector (independent of make_macros' code path).
4. Every decimal number in the compiled PDF text is either a manifest value or listed in
   ALLOWED with its reason; anything else is reported.
5. Decimal literals typed directly in the v3 .tex body (outside \\NV) are listed.

Output: results/paper_v3/check_numbers.json (+ exit code 1 on any failure)
"""
import json
import re
import subprocess
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PAPER = REPO / "paper" / "arxiv_upload_learning_rules_v3"
TEX = PAPER / "learning_rules_rsa_paper_v3.tex"
PDF = PAPER / "learning_rules_rsa_paper_v3.pdf"
P3 = REPO / "results" / "paper_v3"

# numbers that are not results: design constants, untraceable-but-kept values, literature
ALLOWED = {
    "0.02": "PC inference rate (code constant)",
    "0.003": "STDP A+/A- (code constant)",
    "0.10": "'rho < 0.10 at V1' (bound on the reported values, descriptive)",
    "82.4": "accuracy, original training stdout (NUMERIC_AUDIT section 3)",
    "63.2": "accuracy, original training stdout", "56.6": "accuracy, original training stdout",
    "39.0": "accuracy, original training stdout", "10.0": "accuracy, chance level",
    "0.11": "v2 bound quoted in errata", "0.09": "v2 bound quoted in errata",
    "0.06": "v2 bound quoted in errata", "0.07": "v2 bound quoted in errata",
    "0.05": "v2 bound quoted in errata", "0.03": "v2 bound quoted in errata",
    "0.04": "v2 bound quoted in errata",
    "10.1016": "DOI prefix of Walther et al. (2016)",
    "10.7554": "DOI prefix of Hebart et al. (2023)",
}


def single(x, d):
    return format(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP), "f")


def norm(t):
    t = t.replace("\u2212", "-").replace("{,}", ",").lstrip("+")
    if t.startswith("."):
        t = "0" + t
    if t.startswith("-."):
        t = "-0" + t[1:]
    return t


def main():
    man = pd.read_csv(P3 / "numbers_manifest.csv", keep_default_na=False)
    keys = dict(zip(man.key, man.text))
    fails = {}

    # 1. keys used
    srcs = [TEX] + sorted(PAPER.glob("tab_*.tex"))
    used = set()
    for f in srcs:
        used |= set(re.findall(r"\\NV\{([^}]+)\}", f.read_text(encoding="utf-8")))
    body = TEX.read_text(encoding="utf-8")
    # expand the shorthand macros of the preamble
    short = {"rs": "rho.ps.cr.bn.{0}.{1}", "rmean": "rho.mr.cr.bn.{0}.{1}", "dps": "d.ps.cr.bn.{0}.{1}.{2}",
             "lb": "lb.cr.{0}", "ml": "ml.cr.bn.{0}.{1}"}
    for m, pat in short.items():
        n = pat.count("{")
        for args in re.findall(r"\\" + m + r"(?![a-zA-Z])" + r"\{([^}]*)\}" * n, body):
            args = (args,) if isinstance(args, str) else args
            used.add(pat.format(*args))
    for m, pats in {"rsci": ["rho.ps.cr.bn.{0}.{1}.lo", "rho.ps.cr.bn.{0}.{1}.hi"],
                    "dpsci": ["d.ps.cr.bn.{0}.{1}.{2}.lo", "d.ps.cr.bn.{0}.{1}.{2}.hi"],
                    "dmrci": ["d.mr.cr.bn.{0}.{1}.{2}.lo", "d.mr.cr.bn.{0}.{1}.{2}.hi"],
                    "lbci": ["lb.cr.{0}.lo", "lb.cr.{0}.hi"],
                    "mlci": ["ml.cr.bn.{0}.{1}.lo", "ml.cr.bn.{0}.{1}.hi"]}.items():
        n = pats[0].count("{")
        for args in re.findall(r"\\" + m + r"(?![a-zA-Z])" + r"\{([^}]*)\}" * n, body):
            args = (args,) if isinstance(args, str) else args
            used |= {p.format(*args) for p in pats}
    used = {k for k in used if "#" not in k and k != "key"}      # macro definitions, comments
    missing = sorted(k for k in used if k not in keys)
    if missing:
        fails["undefined_keys"] = missing

    # 2. single rounding
    bad = []
    for r in man.itertuples():
        if int(r.decimals) < 0:
            continue
        v = float(r.value) * (100 if str(r.pct) == "True" else 1)
        if norm(r.text) != norm(single(v, int(r.decimals))):
            bad.append((r.key, r.text, single(v, int(r.decimals))))
    if bad:
        fails["not_single_rounding"] = bad

    # 3. step2_summary re-read
    s2 = pd.read_csv(REPO / "results/crossrun/step2_summary.csv").set_index("key")
    mism = []
    n3 = 0
    for r in man[man.source == "results/crossrun/step2_summary.csv"].itertuples():
        sel = r.selector
        col = "point"
        for c in ("ci_lo", "ci_hi"):
            if sel.endswith(" " + c):
                sel, col = sel[: -len(c) - 1], c
        if sel in s2.index:
            n3 += 1
            if abs(s2.loc[sel, col] - float(r.value)) > 1e-12:
                mism.append((r.key, sel, col))
    if mism:
        fails["step2_mismatch"] = mism

    # 4. PDF text
    txt = subprocess.run(["pdftotext", "-enc", "UTF-8", str(PDF), "-"], capture_output=True).stdout.decode("utf-8")
    txt = txt.replace("\u2212", "-")
    found = re.findall(r"(?<![\w.])[-+]?\d*\.\d+(?![\w.])", txt)
    valid = {norm(t) for t in man.text}
    aux = (PAPER / "learning_rules_rsa_paper_v3.aux").read_text(encoding="utf-8")
    secnums = set(re.findall(r"\\numberline \{([0-9]+\.[0-9]+)\}", aux))  # section numbers
    valid |= secnums | {"2608.12408"}                                          # + arXiv id
    unknown = sorted({t for t in found if norm(t) not in valid and norm(t) not in ALLOWED and t not in ALLOWED})
    if unknown:
        fails["pdf_numbers_without_source"] = unknown

    # 5. literals typed in the tex body (outside the generated files)
    lit = []
    quotes = {norm(t) for t in man[man.key.str.startswith("v2quote.")].text}
    in_tikz = False
    for i, line in enumerate(body.splitlines(), 1):
        if r"\begin{tikzpicture}" in line:
            in_tikz = True
        if r"\end{tikzpicture}" in line:
            in_tikz = False
            continue
        if line.lstrip().startswith("%") or in_tikz or r"\newcommand{\newid}" in line:
            continue
        clean = re.sub(r"\\NV\{[^}]*\}|\\[a-zA-Z]+\{[^}]*\}|\[[^\]]*cm\]", "", line)
        for t in re.findall(r"(?<![\w.{])\d*\.\d+(?![\w.])", clean):
            why = "v2 quotation (errata)" if norm(t) in quotes else ALLOWED.get(t, ALLOWED.get(norm(t), "NOT ALLOWED"))
            lit.append((i, t, why))
    if any(x[2] == "NOT ALLOWED" for x in lit):
        fails["typed_literals"] = [x for x in lit if x[2] == "NOT ALLOWED"]

    out = {"n_keys_used": len(used), "n_manifest": len(man), "n_step2_reread": n3,
           "n_pdf_decimals": len(found), "typed_literals_allowed": lit, "fails": fails}
    (P3 / "check_numbers.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k != "typed_literals_allowed"}, indent=1, ensure_ascii=False))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

"""Regenerate the v3 numbers, compile the paper, then check every number in the PDF.

Order matters: check_numbers.py reads the compiled PDF and its .aux, so the paper is
compiled (three pdflatex passes) after make_macros.py and before check_numbers.py.
Same steps as `make check` (Makefile at the repository root), for systems without make.

    python scripts/paper_v3/build_and_check.py
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "paper" / "arxiv_upload_learning_rules_v3"


def run(cmd, cwd=REPO, quiet=False):
    print("+", " ".join(map(str, cmd)), flush=True)
    r = subprocess.run(cmd, cwd=cwd, stdout=subprocess.DEVNULL if quiet else None)
    if r.returncode:
        sys.exit(f"failed ({r.returncode}): {' '.join(map(str, cmd))}")


def main():
    run([sys.executable, "scripts/paper_v3/make_macros.py"], quiet=True)
    for _ in range(3):
        run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "learning_rules_rsa_paper_v3.tex"],
            cwd=PAPER, quiet=True)
    run([sys.executable, "scripts/paper_v3/check_numbers.py"])
    print("build_and_check: OK")


if __name__ == "__main__":
    main()

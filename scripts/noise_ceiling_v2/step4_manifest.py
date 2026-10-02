"""MANIFEST.json: SHA-256 of every RDM used (fMRI + model RDMs of both sets), of every
script and result file, and the commit that holds the scripts.

Usage: py -3 step4_manifest.py  (after the scripts are committed; refuses if the
scripts differ from HEAD, so the recorded hash really is the scripts' commit)
"""
import hashlib
import json
import subprocess

from common import REPO, RESULTS
from step3c_bootstrap import ITEMS

SCRIPTS = REPO / "scripts" / "noise_ceiling_v2"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout.strip()


def main():
    dirty = git("status", "--porcelain", "--", "scripts/noise_ceiling_v2")
    if dirty:
        raise SystemExit(f"scripts not committed:\n{dirty}")
    head = git("log", "-1", "--format=%H", "--", "scripts/noise_ceiling_v2")
    rel = lambda p: p.relative_to(REPO).as_posix()
    man = {
        "scripts_commit": head,
        "python": subprocess.run(["py", "-3", "--version"], capture_output=True,
                                 text=True).stdout.strip(),
        "rdms": {rel(it[3]): sha(it[3]) for it in ITEMS},
        "scripts": {rel(p): sha(p) for p in sorted(SCRIPTS.glob("*.py"))},
        "results": {rel(p): sha(p) for p in sorted(RESULTS.iterdir())
                    if p.is_file() and p.name not in ("MANIFEST.json", "REPORT.md")},
        "paper_sources": {rel(p): sha(p) for p in [
            REPO / "paper/arxiv_upload_learning_rules_v2/learning_rules_rsa_paper_v2.tex",
            REPO / "outputs/rsa_results_cnn.csv", REPO / "outputs/rsa_results_seeds.csv"]},
    }
    man["n_rdms"] = len(man["rdms"])
    (RESULTS / "MANIFEST.json").write_text(json.dumps(man, indent=2))
    print(f"MANIFEST: {man['n_rdms']} RDMs, scripts commit {head}")


if __name__ == "__main__":
    main()

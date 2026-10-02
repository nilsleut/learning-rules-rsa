"""Cross-run analysis on the original (--zscore all) subject RDMs.

Pair set P (all analyses): stimulus pairs that lie in different runs for EVERY subject
(intersection over sub-01, -02, -03; run = run within the session the 720 come from).
Block cleaning (robustness variant): for subject s, every pair in P belongs to the
unordered run pair {run_s(a), run_s(b)} of that subject; the block mean (over the pairs
in P that fall into the block) is subtracted, in the subject RDM and in whatever RDM is
compared with it, using subject s's blocks.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts" / "runz"))
sys.path.insert(0, str(REPO / "scripts" / "noise_ceiling_v2"))
from common_runz import (ROIS, SUBJECTS, N_STIM, TRI, load, run_labels, ranks,  # noqa: E402
                         stim_order)

RESULTS = REPO / "results" / "crossrun"
SEED = 20261005
N_RUNS = 10


def pair_set(stims=None, runs=None):
    """Indices (rows, cols) of the stimulus pairs in P for a stimulus sample.

    stims: array of stimulus indices (a bootstrap sample) or None for 0..719.
    Pairs of a stimulus with itself are dropped. Returns rows, cols, keep-mask on TRI."""
    runs = runs or run_labels()
    s = np.arange(N_STIM) if stims is None else np.asarray(stims)
    a, b = s[TRI[0]], s[TRI[1]]
    keep = a != b
    for r in runs.values():
        keep &= r[a] != r[b]
    return a[keep], b[keep], keep


def block_ids(rows, cols, run):
    """Unordered run-pair id (0..44) of each pair, for one subject's run labels (1..10)."""
    i, j = run[rows] - 1, run[cols] - 1
    lo, hi = np.minimum(i, j), np.maximum(i, j)
    return lo * N_RUNS + hi


def demean_blocks(v, bid):
    """Subtract the mean of each block from v."""
    s = np.bincount(bid, weights=v, minlength=N_RUNS * N_RUNS)
    n = np.bincount(bid, minlength=N_RUNS * N_RUNS)
    return v - (s / np.maximum(n, 1))[bid]

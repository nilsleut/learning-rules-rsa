"""Shared helpers for noise_ceiling_v2.

Conventions (match the paper, learning_rules_v8.py:476-482):
  * RDMs are 720x720 correlation-distance matrices.
  * RSA = Spearman rho on the strict upper triangle (average ranks for ties).
"""
import os
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[2]
# Defaults reproduce the noise_ceiling_v2 run. NC_FMRI_DIR / NC_RESULTS point the same
# scripts at other subject RDMs (e.g. rdms_runz) without touching the v2 results.
FMRI_DIR = Path(os.environ.get("NC_FMRI_DIR", REPO / "outputs_720"))
RESULTS = Path(os.environ.get("NC_RESULTS", REPO / "results" / "noise_ceiling_v2"))
SUBJECTS = ["sub-01", "sub-02", "sub-03"]
ROIS = ["V1", "V2", "LOC", "IT"]
# ROI -> model layer, as in Table 2 of learning_rules_rsa_paper_v2.tex:297-302
ROI_LAYER = {"V1": "Conv1", "V2": "Conv1", "LOC": "Conv3", "IT": "FC1"}
N_STIM = 720


def load_fmri(roi):
    """List of per-subject 720x720 RDMs for one ROI."""
    rdms = [np.load(FMRI_DIR / f"fmri_rdm_{roi}_{s}.npy") for s in SUBJECTS]
    for r in rdms:
        assert r.shape == (N_STIM, N_STIM)
    return rdms


def triu(rdm):
    return rdm[np.triu_indices(rdm.shape[0], k=1)]


def ranks(v):
    """Centred, unit-norm average ranks, so spearman(a, b) == ranks(a) @ ranks(b)."""
    r = rankdata(v)
    r -= r.mean()
    return r / np.linalg.norm(r)


def spearman(a, b):
    return float(ranks(a) @ ranks(b))


def nili_bounds(vecs):
    """Nili et al. (2014) bounds from a list of per-subject dissimilarity vectors.

    upper = mean_s rho(v_s, mean of all subjects)          (s included)
    lower = mean_s rho(v_s, mean of the other subjects)    (leave-one-out)
    No Spearman-Brown correction.
    """
    vecs = [np.asarray(v, float) for v in vecs]
    rk = [ranks(v) for v in vecs]
    total = np.sum(vecs, axis=0)
    r_all = ranks(total / len(vecs))
    upper = [float(r @ r_all) for r in rk]
    lower = [float(r @ ranks((total - v) / (len(vecs) - 1))) for r, v in zip(rk, vecs)]
    return {"upper": float(np.mean(upper)), "lower": float(np.mean(lower)),
            "upper_per_subject": upper, "lower_per_subject": lower}

"""Shared helpers for the sub-03 integrity check.

Reuses, does not re-implement:
  * fMRI RDM loading, Spearman, ROI list         -> scripts/noise_ceiling_v2/common.py
  * stimulus order, image lookup, image loading,
    abs-difference RDM for the luminance scalar  -> colour_statistic_control.py
The only new code is raw-data access (h5) for checks 4-7, which mirrors
extract_fmri_rdms_720.py line for line (see build_rdm_from_raw) and is validated
against the stored RDMs at shift 0 before it is used for anything else.

Requires h5py (not in the project environment); see SUB03_CHECK.md.
"""
import pickle
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts" / "noise_ceiling_v2"))
sys.path.insert(0, str(REPO))

from common import (FMRI_DIR, ROIS, ROI_LAYER, SUBJECTS, N_STIM,  # noqa: E402
                    load_fmri, triu, ranks, spearman)
import colour_statistic_control as csc  # noqa: E402  (main() is guarded)

RESULTS = REPO / "results" / "sub03_check"
SEED = 20261003
# Raw data (THINGS-fMRI metadata + voxel responses): env THINGS_FMRI_DIR, default
# ../RSA/Datensatz next to the repository. extract_fmri_rdms_720.py once hard-coded an
# older location of the same data (documented in the report).
DATA_DIR = Path(os.environ.get("THINGS_FMRI_DIR", REPO.parents[0] / "RSA" / "Datensatz"))
EXTRACT_SCRIPT_DATA_DIR = r"<old location>\RSA\Datensatz"


def pearson(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def stim_order(sub):
    return csc.load_stim_order(sub)


def luminance(paths):
    """Per-image Rec.709 luminance, exactly as colour_statistic_control.build_references."""
    w = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return np.array([(csc.load_rgb(p, csc.REF_PX) @ w).mean() for p in paths])


def image_paths(order):
    return [csc.find_img(s) for s in order]


# ── raw data (mirrors extract_fmri_rdms_720.py) ─────────────────────────────────
ROI_COLUMNS = {"V1": ["V1"], "V2": ["V2"], "V3": ["V3"], "V4": ["hV4"],
               "LOC": ["lLOC", "rLOC"], "IT": ["IT"]}          # extract_fmri_rdms_720.py:53-60


def meta(sub):
    stim = pd.read_csv(DATA_DIR / f"{sub}_task-things_stimulus-metadata.csv")
    vox = pd.read_csv(DATA_DIR / f"{sub}_task-things_voxel-metadata.csv")
    return stim, vox


def h5_path(sub):
    return DATA_DIR / f"{sub}_task-things_voxel-wise-responses.h5"


def h5_labels(sub):
    """Column (trial) and row (voxel) labels stored in the pandas-HDF file."""
    import h5py
    with h5py.File(h5_path(sub), "r") as f:
        rd = f["ResponseData"]
        cols = pickle.loads(rd["block0_items"][0].tobytes())
        vox_ids = rd["block1_values"][:, 0]
        shape = rd["block0_values"].shape
    return np.asarray(cols), vox_ids, shape


def load_raw(sub):
    """Raw (un-normalised) responses of the union of ROI voxels, as in
    extract_fmri_rdms_720.py:62-76. Returns (responses trials x voxels, masks local)."""
    import h5py
    stim, vox = meta(sub)
    masks = {r: np.logical_or.reduce([vox[c].values.astype(bool) for c in cols])
             for r, cols in ROI_COLUMNS.items()}
    comb = np.logical_or.reduce(list(masks.values()))
    idx = np.where(comb)[0]
    with h5py.File(h5_path(sub), "r") as f:
        raw = f["ResponseData/block0_values"][idx, :].astype(np.float32)
    local = {r: np.where(m[idx])[0] for r, m in masks.items()}
    return raw.T, local, stim, vox, idx


def zscore_trials(raw):
    """extract_fmri_rdms_720.py:77-78 (z-score each voxel over all 9840 trials)."""
    return (raw - raw.mean(0)) / (raw.std(0) + 1e-8)


def rows_for_order(stim, order):
    """extract_fmri_rdms_720.py:103-114: metadata row index of each stimulus."""
    rows = []
    for s in order:
        ix = stim.index[stim["stimulus"] == s].tolist()
        assert len(ix) == 1, (s, ix)
        rows.append(ix[0])
    return np.array(rows)


def corr_dist_rdm(X):
    """1 - Pearson between rows; identical to squareform(pdist(X, 'correlation'))."""
    return csc.corr_rdm(X)

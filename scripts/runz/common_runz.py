"""Shared helpers for the run-wise z-normalisation (runz) analysis."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts" / "noise_ceiling_v2"))
sys.path.insert(0, str(REPO / "scripts" / "sub03_check"))
sys.path.insert(0, str(REPO))

from common import ROIS, SUBJECTS, N_STIM, triu, ranks, spearman, nili_bounds  # noqa: E402

RESULTS = REPO / "results" / "runz"
OLD_DIR = REPO / "outputs_720"
RUNZ_DIR = REPO / "rdms_runz"
DATA_DIR = Path(os.environ.get("THINGS_FMRI_DIR", REPO.parents[0] / "RSA" / "Datensatz"))  # THINGS-fMRI metadata
SEED = 20261004
TRI = np.triu_indices(N_STIM, 1)


def load(kind, roi):
    d = {"old": OLD_DIR, "runz": RUNZ_DIR}[kind]
    return [np.load(d / f"fmri_rdm_{roi}_{s}.npy") for s in SUBJECTS]


def stim_order():
    with open(OLD_DIR / "stim_order_sub-01.txt") as f:
        return [l.strip() for l in f if l.strip()]


def run_labels():
    """Run (within the session the 720 come from) of each of the 720 stimuli, per subject."""
    order = stim_order()
    out = {}
    for s in SUBJECTS:
        st = pd.read_csv(DATA_DIR / f"{s}_task-things_stimulus-metadata.csv")
        m = st.set_index("stimulus").loc[order]
        assert m.session.nunique() == 1
        out[s] = m.run.values
    return out

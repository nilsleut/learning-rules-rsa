"""
extract_fmri_rdms_720.py
========================
Berechnet fMRI-RDMs fuer alle 720 THINGS-Konzepte und speichert sie.
Kein PC-Training noetig.

    py -3 extract_fmri_rdms_720.py --data-dir <THINGS-fMRI Datensatz> --out-dir <Ziel> \
        [--zscore all|run] [--stim-order-dir outputs_720]

--data-dir      Ordner mit sub-0X_task-things_{voxel-wise-responses.h5,
                voxel-metadata.csv,stimulus-metadata.csv}. Alternativ Umgebungsvariable
                THINGS_FMRI_DIR. (Frueher hartkodiert: Projekte\\RSA\\Datensatz, heute
                Projekte_1\\RSA\\Datensatz.)
--out-dir       Zielordner fuer fmri_rdm_{ROI}_{sub}.npy. Pflicht, damit nie versehentlich
                outputs_720/ ueberschrieben wird; existierende RDM-Dateien werden nicht
                ueberschrieben (--force, um das zu erlauben).
--zscore all    Voxel-z-Normierung ueber alle 9840 Trials (Verhalten bis Oktober 2026;
                reproduziert outputs_720/).
--zscore run    Voxel-z-Normierung innerhalb jedes Runs (session x run, je 82 Trials inkl.
                Test-Trials), sonst identisch.
--stim-order-dir  Ordner mit stim_order_{sub}.txt (Default outputs_720). Fehlt die Datei,
                wird die Reihenfolge wie bisher aus den Metadaten berechnet und dort abgelegt.

Schreibt zusaetzlich <out-dir>/MANIFEST.json (SHA-256 jeder RDM, Parameter, Trials pro Run).
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

SUBJECTS  = ["sub-01", "sub-02", "sub-03"]
ROI_NAMES = ("V1", "V2", "V3", "V4", "LOC", "IT")
N_IMAGES  = 720
HERE      = Path(__file__).resolve().parent


def compute_rdm(features):
    return squareform(pdist(features, metric="correlation"))


def roi_masks(vox):
    return {
        "V1":  vox["V1"].values.astype(bool),
        "V2":  vox["V2"].values.astype(bool),
        "V3":  vox["V3"].values.astype(bool),
        "V4":  vox["hV4"].values.astype(bool),
        "LOC": (vox["lLOC"].values.astype(bool) | vox["rLOC"].values.astype(bool)),
        "IT":  vox["IT"].values.astype(bool),
    }


def zscore(responses, stim, mode):
    """Voxel-wise z-normalisation. responses: trials x voxels, rows = metadata rows."""
    if mode == "all":
        return (responses - responses.mean(axis=0)) / (responses.std(axis=0) + 1e-8), None
    run_id = stim["session"].astype(str) + "_" + stim["run"].astype(str)
    out = np.empty_like(responses)
    sizes = {}
    for rid, rows in stim.groupby(run_id).indices.items():
        block = responses[rows]
        out[rows] = (block - block.mean(axis=0)) / (block.std(axis=0) + 1e-8)
        sizes[rid] = len(rows)
    return out, sizes


def stim_order_for(sub, stim, order_dir):
    stim_file = order_dir / f"stim_order_{sub}.txt"
    if stim_file.exists():
        with open(stim_file) as f:
            order = [l.strip() for l in f if l.strip()]
        print(f"  Stimulus-Reihenfolge geladen: {len(order)} Stimuli")
        return order
    print("  Berechne Stimulus-Reihenfolge...")
    stim_unique = stim.copy().drop_duplicates(subset="stimulus").copy()
    concepts = sorted(stim_unique["concept"].unique().tolist())
    order = []
    for concept in concepts:
        order.extend(sorted(stim_unique[stim_unique["concept"] == concept]["stimulus"].tolist())[:1])
    order = order[:N_IMAGES]
    order_dir.mkdir(parents=True, exist_ok=True)
    with open(stim_file, "w") as f:
        for s in order:
            f.write(s + "\n")
    print(f"  Stimulus-Reihenfolge gespeichert: {len(order)} Stimuli")
    return order


def extract_subject(sub, data_dir, out_dir, mode, order_dir, force=False):
    print(f"\n{'='*55}\nSubject: {sub}   (zscore={mode})\n{'='*55}")
    h5_file = data_dir / f"{sub}_task-things_voxel-wise-responses.h5"
    vox = pd.read_csv(data_dir / f"{sub}_task-things_voxel-metadata.csv")
    stim = pd.read_csv(data_dir / f"{sub}_task-things_stimulus-metadata.csv")
    assert (stim.index.values == np.arange(len(stim))).all()

    masks = roi_masks(vox)
    combined = np.zeros(len(vox), dtype=bool)
    for m in masks.values():
        combined |= m
    roi_voxel_indices = np.where(combined)[0]
    global_to_local = {int(g): l for l, g in enumerate(roi_voxel_indices)}

    with h5py.File(h5_file, "r") as f:
        raw = f["ResponseData/block0_values"][roi_voxel_indices, :].astype(np.float32)
    responses_all, run_sizes = zscore(raw.T, stim, mode)
    print(f"  Responses: {responses_all.shape}")

    order = stim_order_for(sub, stim, order_dir)
    stim_responses = []
    for s in tqdm(order, desc="  fMRI mitteln"):
        idx = stim.index[stim["stimulus"] == s].tolist()
        if not idx:
            continue
        stim_responses.append(responses_all[idx].mean(axis=0))
    responses = np.array(stim_responses)
    assert responses.shape[0] == N_IMAGES, responses.shape

    files = {}
    for roi in ROI_NAMES:
        l_idx = np.array([global_to_local[int(g)] for g in np.where(masks[roi])[0]
                          if int(g) in global_to_local])
        if len(l_idx) == 0:
            print(f"    {roi}: keine Voxel gefunden")
            continue
        rdm = compute_rdm(responses[:, l_idx])
        path = out_dir / f"fmri_rdm_{roi}_{sub}.npy"
        if path.exists() and not force:
            raise FileExistsError(f"{path} existiert; --force zum Ueberschreiben")
        np.save(str(path), rdm)
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        tri = rdm[np.triu_indices(rdm.shape[0], k=1)]
        print(f"    {roi}: {rdm.shape}  mean={tri.mean():.3f}  std={tri.std():.3f}")
    info = {"n_trials": int(len(stim)), "n_roi_union_voxels": int(len(roi_voxel_indices))}
    if run_sizes is not None:
        sz = np.array(list(run_sizes.values()))
        info.update({"n_runs": int(len(sz)), "trials_per_run_min": int(sz.min()),
                     "trials_per_run_max": int(sz.max()),
                     "trials_per_run_unique": sorted(set(int(x) for x in sz))})
    return files, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-dir", default=os.environ.get("THINGS_FMRI_DIR"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--zscore", choices=["all", "run"], default="all")
    ap.add_argument("--stim-order-dir", default=str(HERE / "outputs_720"))
    ap.add_argument("--subjects", nargs="+", default=SUBJECTS)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if not a.data_dir:
        ap.error("--data-dir oder THINGS_FMRI_DIR angeben")
    data_dir, out_dir = Path(a.data_dir), Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"zscore": a.zscore, "data_dir": str(data_dir),
                "stim_order_dir": str(a.stim_order_dir), "subjects": {}, "files": {}}
    for sub in a.subjects:
        files, info = extract_subject(sub, data_dir, out_dir, a.zscore,
                                      Path(a.stim_order_dir), a.force)
        manifest["files"].update(files)
        manifest["subjects"][sub] = info
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nFertig. Dateien in: {out_dir}")


if __name__ == "__main__":
    main()

"""Partial RSA (pixel similarity controlled) for paper v3.

Pixel RDM: definition of phase4_analysis_v3.py (v2): THINGS image of each stimulus in the
order of outputs_720/stim_order_sub-01.txt, Resize(224) -> CenterCrop(224) -> ToTensor,
flattened, correlation distance. Partial Spearman: v2's partial_spearman (rank
residualisation of x and y on z, then Spearman of the residuals).

Stage 1  validation: the v2 analysis (all pairs, mean-RDM convention, seed-averaged model
         RDM of the original set) must reproduce outputs/partial_rsa_results.csv (stored at
         4 decimals). If it does not, the pixel RDM is not the v2 one -> stop.
Stage 2  v3 analysis: repaired set (bnfix), cross-run pairs, per-subject convention (partial
         rho per seed and subject, averaged), 1000 shared stimulus resamples (seed 20261002).

Output: results/paper_v3/pixel_rdm_images.csv, partial_validation.csv, partial_draws.csv,
        partial_summary.csv, partial_rsa.json   (pixel RDM itself: results/paper_v3/pixel_rdm.npy,
        not committed; SHA-256 in partial_rsa.json)
"""
import hashlib
import os
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts" / "crossrun"))
from common_cr import ROIS, SUBJECTS, N_STIM, TRI, load, run_labels, ranks, pair_set, stim_order  # noqa: E402
from common import ROI_LAYER  # noqa: E402
from step2_bootstrap import CACHE, MODEL_ITEMS  # noqa: E402

OUT = REPO / "results" / "paper_v3"
THINGS = Path(os.environ.get("THINGS_IMAGES_DIR",
                            REPO.parents[0] / "RSA" / "Datensatz" / "images_THINGS" / "object_images"))
RULES = ["random_weights", "backprop", "feedback_alignment", "predictive_coding", "stdp"]
CSVNAME = {"random_weights": "Random Weights", "backprop": "Backprop", "feedback_alignment": "Feedback Alignment",
           "predictive_coding": "Predictive Coding", "stdp": "STDP"}
BOOT_SEED, N_BOOT = 20261002, 1000
_W = {}


def find_img(stimulus):            # verbatim logic of phase4_analysis_v3.find_img
    name = stimulus.replace(".jpg", "")
    parts = name.split("_")
    last = parts[-1]
    concept = "_".join(parts[:-1]) if (len(parts) > 1 and len(last) <= 4
                                       and any(c.isdigit() for c in last)) else name
    for pat in [f"{concept}/{name}.jpg", f"{concept}/*.jpg"]:
        hits = sorted(THINGS.glob(pat))
        if hits:
            return hits[0]
    for folder in THINGS.iterdir():
        if folder.name.lower() == concept.lower():
            imgs = sorted(folder.glob("*.jpg"))
            if imgs:
                return imgs[0]
    return None


def pixel_rdm():
    import torch
    import torchvision.transforms as T
    from PIL import Image
    order = stim_order()
    paths = [find_img(s) for s in order]
    assert all(p is not None for p in paths) and len(paths) == N_STIM
    pd.DataFrame({"stimulus": order, "image": [str(p.relative_to(THINGS)) for p in paths]}) \
        .to_csv(OUT / "pixel_rdm_images.csv", index=False)
    tf = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor()])
    with torch.no_grad():
        X = np.stack([tf(Image.open(p).convert("RGB")).view(-1).numpy() for p in paths])
    return squareform(pdist(X, metric="correlation"))


def residualize(a, b):             # verbatim v2 partial_spearman helper
    ar = rankdata(a).astype(float)
    br = rankdata(b).astype(float)
    bc = br - br.mean()
    beta = np.dot(ar, bc) / (np.dot(bc, bc) + 1e-10)
    return ar - beta * br


def validate(P):
    tri = np.triu_indices(N_STIM, 1)
    pv = P[tri]
    ref = pd.read_csv(REPO / "outputs/partial_rsa_results.csv")
    rows = []
    for roi in ROIS:
        brain = np.mean(load("old", roi), axis=0)[tri]
        for r in RULES:
            seeds = sorted((REPO / "outputs/model_rdms").glob("seed_*"))
            m = np.mean([np.load(d / f"rdm_{r}_{ROI_LAYER[roi]}.npy") for d in seeds], axis=0)[tri]
            std = spearmanr(m, brain)[0]
            par = spearmanr(residualize(m, pv), residualize(brain, pv))[0]
            x = ref[(ref.roi == roi) & (ref.rule == CSVNAME[r])].iloc[0]
            rows.append({"roi": roi, "rule": r, "rho_std": std, "rho_partial": par,
                         "v2_rho_std": x.rho_std, "v2_rho_partial": x.rho_partial,
                         "dev_std": abs(std - x.rho_std), "dev_partial": abs(par - x.rho_partial)})
    v = pd.DataFrame(rows)
    v.to_csv(OUT / "partial_validation.csv", index=False)
    return v


def _init():
    _W["runs"] = run_labels()
    _W["fmri"] = {roi: load("old", roi) for roi in ROIS}
    _W["P"] = np.load(OUT / "pixel_rdm.npy")
    M = np.load(CACHE, mmap_mode="r")
    _W["M"] = {(r, roi, sd): M[MODEL_ITEMS.index(("bnfix", r, ROI_LAYER[roi], f"seed_{sd}"))]
               for r in RULES for roi in ROIS for sd in range(5)}
    _W["boot"] = np.random.default_rng(BOOT_SEED).integers(0, N_STIM, size=(N_BOOT, N_STIM))


def draw(b):
    s = None if b < 0 else _W["boot"][b]
    rp, cp, _ = pair_set(s, _W["runs"])
    pv = _W["P"][rp, cp]
    out = {"boot": b}
    for roi in ROIS:
        subj = [x[rp, cp] for x in _W["fmri"][roi]]
        s_std = [ranks(v) for v in subj]
        s_par = [ranks(residualize(v, pv)) for v in subj]
        for r in RULES:
            std, par = [], []
            for sd in range(5):
                mv = np.asarray(_W["M"][(r, roi, sd)][rp, cp], float)
                ms, mp = ranks(mv), ranks(residualize(mv, pv))
                std.append(np.mean([ms @ q for q in s_std]))
                par.append(np.mean([mp @ q for q in s_par]))
            out[f"std|{r}|{roi}"] = float(np.mean(std))
            out[f"partial|{r}|{roi}"] = float(np.mean(par))
    return out


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    P = pixel_rdm()
    np.save(OUT / "pixel_rdm.npy", P)
    v = validate(P)
    meta = {"pixel_rdm_sha256": hashlib.sha256(np.ascontiguousarray(P).tobytes()).hexdigest(),
            "validation_max_dev_std": float(v.dev_std.max()),
            "validation_max_dev_partial": float(v.dev_partial.max())}
    print(json.dumps(meta), flush=True)
    # v2 stored 4 decimals: a faithful reconstruction must agree to rounding (<= 5e-5 + float noise)
    if meta["validation_max_dev_partial"] > 6e-5 or meta["validation_max_dev_std"] > 6e-5:
        (OUT / "partial_rsa.json").write_text(json.dumps({**meta, "STOP": "pixel RDM does not reproduce v2"}, indent=2))
        raise SystemExit("STOP: reconstruction does not reproduce v2 partial RSA")

    with Pool(10, initializer=_init) as pool:
        draws = pool.map(draw, range(-1, N_BOOT), chunksize=4)
    d = pd.DataFrame(draws).sort_values("boot")
    d.to_csv(OUT / "partial_draws.csv", index=False)
    pt, bt = d[d.boot == -1].iloc[0], d[d.boot >= 0]
    rows = []
    for roi in ROIS:
        for r in RULES:
            ks, kp = f"std|{r}|{roi}", f"partial|{r}|{roi}"
            dd = bt[kp] - bt[ks]
            rows.append({"rule": r, "roi": roi,
                         "std": pt[ks], "std_lo": bt[ks].quantile(.025), "std_hi": bt[ks].quantile(.975),
                         "partial": pt[kp], "partial_lo": bt[kp].quantile(.025), "partial_hi": bt[kp].quantile(.975),
                         "delta": pt[kp] - pt[ks], "delta_lo": dd.quantile(.025), "delta_hi": dd.quantile(.975)})
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "partial_summary.csv", index=False)
    s2 = pd.read_csv(REPO / "results/crossrun/step2_summary.csv").set_index("key")
    dev = max(abs(pt[f"std|{r}|{roi}"] - s2.loc[f"persub|cr|bnfix|{r}|{roi}", "point"]) for r in RULES for roi in ROIS)
    meta.update({"boot_seed": BOOT_SEED, "n_boot": N_BOOT, "std_max_dev_vs_crossrun_step2": float(dev),
                 "runtime_s": round(time.time() - t0, 1)})
    (OUT / "partial_rsa.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=1))
    assert dev < 1e-9, dev


if __name__ == "__main__":
    main()

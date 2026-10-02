"""Checks 5-7 on the raw responses.

0. Validation: rebuild every stored 720-stimulus RDM from the h5 with the mirrored
   extraction (common_sub03) and require agreement with outputs_720/*.npy. Nothing
   below is trusted unless this passes.
5. Shift test.
   (a) index space: sub-03 RDM rows/cols shifted circularly by k = -5..5 in the
       sorted-concept order, vs sub-01 and vs luminance;
   (b) trial space: each stimulus takes the response of the trial k positions later
       in the same run (circular within the 82-trial run). This is the realistic
       off-by-k bug. Done for sub-03 and, as a sensitivity control, for sub-01;
   (c) 1000 random relabellings of sub-03 (seed 20261003) as null.
6. Data quality per subject and ROI: voxel count, NaN/Inf, constant voxels, raw
   response variance over the 720 stimuli, RDM entry moments, and the dataset's own
   per-voxel reliability from the 12x-repeated test images (voxel metadata columns
   nc_testset, splithalf_corrected) -- an estimate that involves no pipeline step of ours.
7. Concept level: every session shows each of the 720 concepts once (a different
   exemplar per session). Per subject and session a 720-concept RDM is built; we report
   (i) within-subject between-session consistency, i.e. whether sub-03's session 5 (the
   one in the paper) is an outlier among its own sessions, and (ii) Check 1 repeated on
   concept RDMs averaged over all 12 exemplars.

Output: results/sub03_check/check0_validation.csv, check5_*.csv, check6_quality.csv,
        check7_*.csv, check567.json
"""
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import skew

from common_sub03 import (RESULTS, SEED, SUBJECTS, ROIS, N_STIM, load_fmri, triu, ranks,
                          stim_order, csc, load_raw, zscore_trials, rows_for_order,
                          corr_dist_rdm)

K = list(range(-5, 6))
N_NULL = 1000


def rdm_rank(Z, rows, cols):
    return ranks(triu(corr_dist_rdm(Z[np.ix_(rows, cols)])))


def main():
    t0 = time.time()
    order = stim_order("sub-01")
    stored = {roi: load_fmri(roi) for roi in ROIS}
    lum = pd.read_csv(RESULTS / "luminance.csv")
    assert list(lum.stimulus) == order
    lum_r = ranks(triu(csc.abs_rdm(lum.luminance.values)))
    sub01_r = {roi: ranks(triu(stored[roi][0])) for roi in ROIS}

    val, quality, shift_trial, sess_rows, concept_rdm_r = [], [], [], [], {}
    info = {}
    for si, sub in enumerate(SUBJECTS):
        raw, local, stim, vox, idx = load_raw(sub)
        Z = zscore_trials(raw)
        rows = rows_for_order(stim, order)
        sess_720 = int(stim.loc[rows, "session"].iloc[0])
        info[sub] = {"session_of_720": sess_720, "n_union_voxels": int(len(idx))}

        # 0 validation ------------------------------------------------------------
        for roi in ROIS:
            R = corr_dist_rdm(Z[np.ix_(rows, local[roi])])
            val.append({"subject": sub, "roi": roi,
                        "max_abs_diff_vs_stored": float(np.abs(R - stored[roi][si]).max())})

        # 6 quality ---------------------------------------------------------------
        for roi in ROIS:
            X = raw[:, local[roi]]
            X720 = raw[np.ix_(rows, local[roi])]
            e = triu(stored[roi][si])
            vm = vox.iloc[idx[local[roi]]]
            quality.append({
                "subject": sub, "roi": roi, "n_voxels": int(len(local[roi])),
                "frac_nan": float(np.isnan(X).mean()), "frac_inf": float(np.isinf(X).mean()),
                "frac_constant_voxels": float((np.nanstd(X, 0) == 0).mean()),
                "mean_var_over_720_raw": float(np.nanvar(X720, 0).mean()),
                "rdm_mean": float(e.mean()), "rdm_sd": float(e.std()), "rdm_skew": float(skew(e)),
                "nc_testset_mean": float(vm.nc_testset.mean()),
                "nc_testset_median": float(vm.nc_testset.median()),
                "splithalf_corrected_mean": float(vm.splithalf_corrected.mean()),
                "frac_vox_nc_testset_gt10": float((vm.nc_testset > 10).mean()),
            })

        # 5b trial-space shift (V1, and every ROI for completeness) ----------------
        meta_sr = stim[["session", "run"]].values
        for k in K:
            shifted = []
            for r in rows:
                s_, ru = meta_sr[r]
                run_rows = np.where((meta_sr[:, 0] == s_) & (meta_sr[:, 1] == ru))[0]
                p = np.searchsorted(run_rows, r)
                shifted.append(run_rows[(p + k) % len(run_rows)])
            shifted = np.array(shifted)
            for roi in ROIS:
                rr = rdm_rank(Z, shifted, local[roi])
                shift_trial.append({"subject": sub, "roi": roi, "k": k,
                                    "rho_vs_luminance": float(rr @ lum_r),
                                    "rho_vs_sub01": float(rr @ sub01_r[roi]) if sub != "sub-01"
                                    else np.nan})

        # 7 concept level per session ---------------------------------------------
        tr = stim[stim.trial_type == "train"]
        concepts = sorted(tr.concept.unique())
        acc = {roi: None for roi in ROIS}
        sess_ranks = {roi: {} for roi in ROIS}
        for sess in sorted(tr.session.unique()):
            # metadata row (= h5 column, verified in check 4) of each concept in this session
            r_rows = (tr[tr.session == sess].reset_index()
                      .set_index("concept").loc[concepts, "index"].values)
            for roi in ROIS:
                R = corr_dist_rdm(Z[np.ix_(r_rows, local[roi])])
                sess_ranks[roi][int(sess)] = ranks(triu(R))
                acc[roi] = R if acc[roi] is None else acc[roi] + R
        for roi in ROIS:
            S = sorted(sess_ranks[roi])
            M = np.array([[sess_ranks[roi][a] @ sess_ranks[roi][b] for b in S] for a in S])
            for i, a in enumerate(S):
                others = [M[i, j] for j in range(len(S)) if j != i]
                sess_rows.append({"subject": sub, "roi": roi, "session": a,
                                  "is_session_of_720": a == sess_720,
                                  "mean_rho_with_own_other_sessions": float(np.mean(others))})
            concept_rdm_r[(sub, roi)] = ranks(triu(acc[roi] / len(S)))
        print(f"{sub} done {time.time() - t0:.0f}s", flush=True)
        del raw, Z

    pd.DataFrame(val).to_csv(RESULTS / "check0_validation.csv", index=False)
    max_dev = max(v["max_abs_diff_vs_stored"] for v in val)
    assert max_dev < 1e-6, f"rebuilt RDMs do not match stored ones: {max_dev}"

    # 5a index-space circular shift + 5c null --------------------------------------
    rng = np.random.default_rng(SEED)
    shift_idx, null = [], []
    for roi in ROIS:
        R3 = stored[roi][2]
        for k in K:
            p = (np.arange(N_STIM) + k) % N_STIM
            rr = ranks(triu(R3[np.ix_(p, p)]))
            shift_idx.append({"roi": roi, "k": k, "rho_vs_sub01": float(rr @ sub01_r[roi]),
                              "rho_vs_luminance": float(rr @ lum_r)})
    R3 = stored["V1"][2]
    for _ in range(N_NULL):
        p = rng.permutation(N_STIM)
        rr = ranks(triu(R3[np.ix_(p, p)]))
        null.append({"rho_vs_sub01": float(rr @ sub01_r["V1"]), "rho_vs_luminance": float(rr @ lum_r)})
    null = pd.DataFrame(null)

    # 7 between-subject at concept level -------------------------------------------
    cpairs = []
    for roi in ROIS:
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            cpairs.append({"roi": roi, "pair": f"{i+1:02d}-{j+1:02d}",
                           "spearman_concept_12ex": float(
                               concept_rdm_r[(SUBJECTS[i], roi)] @ concept_rdm_r[(SUBJECTS[j], roi)])})

    pd.DataFrame(quality).to_csv(RESULTS / "check6_quality.csv", index=False)
    pd.DataFrame(shift_trial).to_csv(RESULTS / "check5_shift_trial.csv", index=False)
    pd.DataFrame(shift_idx).to_csv(RESULTS / "check5_shift_index.csv", index=False)
    null.to_csv(RESULTS / "check5_null_V1.csv", index=False)
    pd.DataFrame(sess_rows).to_csv(RESULTS / "check7_sessions.csv", index=False)
    pd.DataFrame(cpairs).to_csv(RESULTS / "check7_concept_pairs.csv", index=False)
    out = {"seed": SEED, "n_null": N_NULL, "validation_max_abs_diff": max_dev, "subjects": info,
           "null_V1": {c: {"mean": float(null[c].mean()), "sd": float(null[c].std(ddof=1)),
                           "q975": float(null[c].quantile(0.975)),
                           "q025": float(null[c].quantile(0.025))} for c in null},
           "runtime_s": round(time.time() - t0, 1)}
    (RESULTS / "check567.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

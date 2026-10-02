"""Check 4: stimulus IDs and trial/voxel alignment, line by line.

Provenance chain of a brain RDM row (extract_fmri_rdms_720.py):
  stim_order_{sub}.txt (l.82-86, loaded; computed at l.88-101 only if missing)
  -> stim.index[stim.stimulus == s]   metadata row of that stimulus (l.106)
  -> responses_all[row]               h5 column `row` (l.70-76, transposed)
So an assignment error per subject can arise (a) in the order file, (b) in the metadata
row -> h5 column correspondence, (c) in the voxel metadata row -> h5 row correspondence.
All three are checked here, plus whether the order file equals what l.88-101 would
recompute from each subject's own metadata.

Output: results/sub03_check/check4_ids.json, check4_order_table.csv
"""
import json
import re

import numpy as np
import pandas as pd

from common_sub03 import RESULTS, SUBJECTS, N_STIM, stim_order, meta, h5_labels


def recompute_order(stim):
    """extract_fmri_rdms_720.py:92-99 verbatim."""
    stim_unique = stim.copy().drop_duplicates(subset="stimulus").copy()
    concepts = sorted(stim_unique["concept"].unique().tolist())
    order = []
    for c in concepts:
        order.extend(sorted(stim_unique[stim_unique["concept"] == c]["stimulus"].tolist())[:1])
    return order[:N_STIM]


def main():
    out = {"subjects": {}}
    orders = {s: stim_order(s) for s in SUBJECTS}
    ref = orders["sub-01"]
    table = pd.DataFrame({"pos": range(N_STIM)})
    for s in SUBJECTS:
        o = orders[s]
        stim, vox = meta(s)
        cols, vox_ids, shape = h5_labels(s)
        rec = recompute_order(stim)
        sub = stim[stim.stimulus.isin(o)].set_index("stimulus").loc[o]
        mism = [i for i, (a, b) in enumerate(zip(o, ref)) if a != b]
        # numeric vs lexical exemplar suffix: does sorted() pick e.g. _10 before _2?
        suffix = [re.search(r"_(\d+)(\w?)\.jpg$", x) for x in o]
        num = [int(m.group(1)) if m else None for m in suffix]
        out["subjects"][s] = {
            "n": len(o), "n_unique": len(set(o)),
            "n_equal_to_sub01_by_position": int(sum(a == b for a, b in zip(o, ref))),
            "first_mismatch_vs_sub01": mism[0] if mism else None,
            "missing_vs_sub01": sorted(set(ref) - set(o)),
            "extra_vs_sub01": sorted(set(o) - set(ref)),
            "order_file_equals_recomputed_from_own_metadata": rec == o,
            "metadata_rows": int(len(stim)),
            "metadata_index_is_0..n-1": bool((stim.index.values == np.arange(len(stim))).all()),
            "trial_id_equals_row": bool((stim.trial_id.values == np.arange(len(stim))).all()),
            "h5_shape_voxels_x_trials": list(shape),
            "h5_column_labels_equal_0..n-1": bool((cols == np.arange(len(cols))).all()),
            "h5_n_columns_equals_metadata_rows": int(len(cols)) == int(len(stim)),
            "h5_voxel_ids_equal_voxel_metadata": bool(
                len(vox_ids) == len(vox) and (vox_ids == vox.voxel_id.values).all()),
            "voxel_metadata_subject_id": sorted(vox.subject_id.unique().tolist()),
            "stim_metadata_subject_id": sorted(stim.subject_id.unique().tolist()),
            "the_720_trial_type": sub.trial_type.value_counts().to_dict(),
            "the_720_reps_per_stimulus_max": int(stim[stim.stimulus.isin(o)]
                                                 .stimulus.value_counts().max()),
            "the_720_sessions": sorted(sub.session.unique().tolist()),
            "the_720_runs_per_session": int(sub.groupby("session").run.nunique().max()),
            "exemplar_suffix_numbers": sorted(set(n for n in num if n is not None)),
            "suffix_pattern_unparsed": int(sum(m is None for m in suffix)),
        }
        table[f"{s}_stimulus"] = o
        table[f"{s}_trial_id"] = sub.trial_id.values
        table[f"{s}_session"] = sub.session.values
        table[f"{s}_run"] = sub.run.values
    # Does presentation order differ between subjects for the 720? (it may, legitimately)
    out["trial_id_identical_across_subjects"] = bool(
        (table["sub-01_trial_id"] == table["sub-03_trial_id"]).all() and
        (table["sub-01_trial_id"] == table["sub-02_trial_id"]).all())
    out["session_identical_across_subjects"] = bool(
        (table["sub-01_session"] == table["sub-03_session"]).all())
    table.to_csv(RESULTS / "check4_order_table.csv", index=False)
    (RESULTS / "check4_ids.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()

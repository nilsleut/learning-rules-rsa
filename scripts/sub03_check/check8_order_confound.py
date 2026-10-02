"""Check 8 (added; follows from check 4/5): shared presentation order.

The 720 stimuli come from one session per subject (sub-01: 3, sub-02: 2, sub-03: 5).
In those sessions sub-01 and sub-02 saw the 720 in the identical run and position;
sub-03 in an unrelated order. Order-locked structure (run offsets, drift, carry-over
of the previous trial's response) is then shared by sub-01/02 and not by sub-03.

Quantified without changing any stored RDM:
  (a) order identity per session pair across all 12 sessions (does it extend to the
      concept-level check 7?);
  (b) rho(RDM_s, same-run RDM_s) and rho(RDM_s, |trial-position difference| RDM_s)
      with each subject's own order;
  (c) sub-01 vs sub-02 restricted to pairs whose stimuli lie in different runs
      (no shared run membership) vs all pairs;
  (d) order-locked component from check 5b: sub-02 vs sub-01 at k != 0.

Output: results/sub03_check/check8_*.csv, check8.json
"""
import json

import numpy as np
import pandas as pd

from common_sub03 import RESULTS, SUBJECTS, ROIS, N_STIM, load_fmri, triu, ranks, \
    stim_order, meta


def main():
    order = stim_order("sub-01")
    pos = {}
    sess_orders = {}
    for s in SUBJECTS:
        st, _ = meta(s)
        st["pos_in_run"] = st.groupby(["session", "run"]).cumcount()
        st["pos_in_session"] = st.groupby("session").cumcount()
        m = st.set_index("stimulus").loc[order]
        pos[s] = m[["session", "run", "pos_in_run", "pos_in_session"]].reset_index(drop=True)
        tr = st[st.trial_type == "train"]
        sess_orders[s] = {int(k): g.concept.tolist() for k, g in tr.groupby("session")}

    # (a) which sessions share the same concept order across subjects
    rows = []
    for a, b in [("sub-01", "sub-02"), ("sub-01", "sub-03"), ("sub-02", "sub-03")]:
        for sa, oa in sess_orders[a].items():
            for sb, ob in sess_orders[b].items():
                if oa == ob:
                    rows.append({"pair": f"{a}/{b}", "session_a": sa, "session_b": sb})
    same = pd.DataFrame(rows, columns=["pair", "session_a", "session_b"])
    same.to_csv(RESULTS / "check8_identical_session_orders.csv", index=False)

    tri = np.triu_indices(N_STIM, 1)
    res, out = [], {}
    for roi in ROIS:
        rdms = load_fmri(roi)
        for si, s in enumerate(SUBJECTS):
            p = pos[s]
            same_run = (p.run.values[:, None] != p.run.values[None, :]).astype(float)
            tdist = np.abs(p.pos_in_session.values[:, None] - p.pos_in_session.values[None, :])
            r = ranks(triu(rdms[si]))
            res.append({"roi": roi, "subject": s,
                        "rho_vs_different_run": float(r @ ranks(triu(same_run))),
                        "rho_vs_trial_distance": float(r @ ranks(triu(tdist.astype(float))))})
        # (c) sub-01 vs sub-02 on between-run pairs only
        p = pos["sub-01"]
        between = p.run.values[tri[0]] != p.run.values[tri[1]]
        v1, v2 = triu(rdms[0]), triu(rdms[1])
        out[roi] = {"sub01_sub02_all_pairs": float(ranks(v1) @ ranks(v2)),
                    "sub01_sub02_between_run_pairs": float(ranks(v1[between]) @ ranks(v2[between])),
                    "sub01_sub02_within_run_pairs": float(ranks(v1[~between]) @ ranks(v2[~between])),
                    "frac_between_run_pairs": float(between.mean())}
    ord_df = pd.DataFrame(res)
    ord_df.to_csv(RESULTS / "check8_order_rdm.csv", index=False)

    sh = pd.read_csv(RESULTS / "check5_shift_trial.csv")
    sh = sh[(sh.subject == "sub-02") & (sh.k != 0)]
    order_locked = sh.groupby("roi").rho_vs_sub01.mean().to_dict()
    j = {"identical_720_order_sub01_sub02": bool(
            (pos["sub-01"][["run", "pos_in_run"]] == pos["sub-02"][["run", "pos_in_run"]]).all().all()),
         "identical_720_order_sub01_sub03": bool(
            (pos["sub-01"][["run", "pos_in_run"]] == pos["sub-03"][["run", "pos_in_run"]]).all().all()),
         "n_identical_session_orders": same.groupby("pair").size().to_dict(),
         "restricted_pairs": out,
         "order_locked_sub02_vs_sub01_mean_k_ne_0": order_locked}
    (RESULTS / "check8.json").write_text(json.dumps(j, indent=2))
    print(same.to_string()); print(ord_df.round(4).to_string()); print(json.dumps(j, indent=1))


if __name__ == "__main__":
    main()

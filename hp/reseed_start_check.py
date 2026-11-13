#!/usr/bin/env python3
"""The re-seed's quad-double polish from the prediction's own data (Reseeder(qd_start="pred"), #693) against the
double-double ladder's start (qd_start="ladder", the default), on stored predictions of a chain.

    cd hp && python3 reseed_start_check.py RAWDIR RECORDS.jsonl PRED_DIR [--threads 2] [--max_ok 3] [--json OUT]

RECORDS.jsonl is the chain's anm_run.jsonl, whose attempts carry the re-seed's outcome (reseed.ok) and D_str;
PRED_DIR holds the predictions it saved (--save_pred 1, pred_pin<k>_D<D>.npz), relative to RAWDIR. Pass (exit 0):
every prediction whose re-seed failed in the chain is re-seeded by the "pred" start, and the "ladder" start still
fails on it (the control: the old start must reproduce the chain's failure); on up to --max_ok predictions whose
re-seed was ok, both starts converge and their horizon data agree to 1e-30 relative. A prediction file that was
written again by a later attempt at the same D is skipped and named.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402

from hpkernel.driver import load_q  # noqa: E402
from reseed import Reseeder  # noqa: E402


def horizon(z, rs):
    sl = slice(rs.off["mh"], rs.off["mh"] + 2 * rs.M)
    return [np.asarray(p, dtype=float)[sl] for p in z]


def rel(a, b):
    """max relative difference of two four-part vectors, summed in mpmath"""
    with mp.workdps(80):
        d = [abs(sum(mp.mpf(float(p[i])) for p in a) - sum(mp.mpf(float(p[i])) for p in b)) for i in range(len(a[0]))]
        s = [abs(sum(mp.mpf(float(p[i])) for p in b)) for i in range(len(b[0]))]
        return float(max(x for x in d) / max(max(s), mp.mpf("1e-300")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("records")
    ap.add_argument("pred_dir")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--max_ok", type=int, default=3)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    outcome = {}                         # D_str -> list of re-seed outcomes of the attempts at that D
    for ln in open(os.path.join(a.raw, a.records)):
        if not ln.strip():
            continue
        r = json.loads(ln)
        for at in r.get("attempts", []):
            rsd = at.get("reseed")
            if rsd is not None:
                outcome.setdefault(str(at["D_str"]), []).append(bool(rsd["ok"]))
    preds = sorted(glob.glob(os.path.join(a.raw, a.pred_dir, "pred_pin*_D*.npz")))
    failed, ok, skipped = [], [], []
    for p in preds:
        z, m = load_q(p)
        outs = outcome.get(str(m["D_str"]))
        if not outs:
            continue
        if len(set(outs)) > 1:
            skipped.append(os.path.basename(p))         # attempts at this D disagree: the file is the last one's
        elif outs[0]:
            ok.append(p)
        else:
            failed.append(p)
    bad, recs = [], []
    for p in failed + ok[:a.max_ok]:
        z, m = load_q(p)
        D = mp.mpf(m["D_str"])
        res = {}
        for start in ("pred", "ladder"):
            rs = Reseeder(m, D, threads=a.threads, qd=True, qd_start=start)
            zr, diag = rs.reseed(tuple(np.asarray(q, dtype=float) for q in z))
            res[start] = (bool(diag["ok"]), horizon(zr, rs), diag.get("seconds"))
        was_ok = p in ok
        rec = dict(pred=os.path.basename(p), D=float(m["D"]), chain_ok=was_ok, pred_ok=res["pred"][0],
                   ladder_ok=res["ladder"][0], seconds=[res["pred"][2], res["ladder"][2]])
        if was_ok:
            agree = rel(res["pred"][1], res["ladder"][1]) if res["pred"][0] and res["ladder"][0] else float("nan")
            rec["rel_diff"] = agree
            good = res["pred"][0] and res["ladder"][0] and agree <= 1e-30
        else:
            good = res["pred"][0] and not res["ladder"][0]
        rec["pass"] = bool(good)
        recs.append(rec)
        print(json.dumps(rec), flush=True)
        if not good:
            bad.append(rec["pred"])
    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(records=recs, skipped=skipped, failed=bad), f, indent=1)
    print(f"{len(failed)} failed and {min(len(ok), a.max_ok)} ok predictions checked, {len(skipped)} skipped "
          f"({', '.join(skipped) or 'none'}): {'PASS' if not bad and failed else 'FAIL'}"
          + (f" ({', '.join(bad)})" if bad else "") + ("" if failed else " (no failed prediction found)"), flush=True)
    return 0 if (not bad and failed) else 1


if __name__ == "__main__":
    sys.exit(main())

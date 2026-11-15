#!/usr/bin/env python3
"""The error of an attempt's points in p1's odd harmonics against a converged root of the same plan (#752, #779):
the prediction (before its re-seed), the iterate at Newton step 1, and its full-step trial z1 + d1, as |(a_k, b_k)|
of p1 minus the root's (crawl_probe.harmonics' convention), with the max over all unknowns for scale. Numpy only.

    cd hp && python3 junk_err.py RAW --case NAME=PRED,ITER,ROOT[,W] [--case ...] [--mutate k17] [--json OUT]

Each case's files are relative to RAW: PRED the attempt's stored prediction (pred_pin<k>_D<D>.npz, four parts), ITER
crawl_probe.py anm's iterate at Newton step 1 (its z0..z7 and step d0..d7), ROOT a converged state of the same plan
(crawl_probe.py fixrun's output, eight parts z0..z7, or an anm_run.py state, zh..z7). W, optional: the own test's
correction at the step-1 trial lam = 1 (its |dz_t|, or the next iterate's |dz| where the trial was accepted), against
which the trial's harmonic-13 error is compared: equal to 10% or not (#779). --mutate k17 adds 1e-20 cos(17 theta) to
the last part of the first case's iterate p1, where its error is about 1e-30: the iterate's k 17 must then read 1e-20
within a factor 2 (the control of the reading). Exit 1 when the control reading misses.

#752 ran this with its two cases fixed in the script (raw I752-mac/junk_err.py): the crawl's root from fixrun and the
control's #742's M state at D 117.5.
"""
import json
import sys

import numpy as np

M = 32
TH = np.pi * np.arange(M) / M
KS = np.arange(1, M, 2)
Q = ("zh", "zl", "z2", "z3")
O = ("zh", "zl", "z2", "z3", "z4", "z5", "z6", "z7")


def harm(v):
    a = (2.0 / M) * np.cos(np.outer(KS, TH)) @ v
    b = (2.0 / M) * np.sin(np.outer(KS, TH)) @ v
    return np.hypot(a, b)


def parts(raw, path, keys):
    d = np.load(f"{raw}/{path}")
    return [np.asarray(d[k], dtype=float) for k in keys if k in d.files]


def state_parts(raw, path):
    """a state's parts in either layout: z0..z7 (fixrun, anm's iterates) or zh, zl, z2..z7 (anm_run.py, save_q)"""
    d = np.load(f"{raw}/{path}")
    keys = [f"z{q}" for q in range(8)] if "z0" in d.files else O
    return [np.asarray(d[k], dtype=float) for k in keys if k in d.files]


def lead_diff(x, y):
    """x - y as the sum of the parts' differences, smallest first: the leading parts' difference is exact for
    neighbouring values (Sterbenz), so the result is good to about 1e-32 of the values"""
    n = max(len(x), len(y))
    z = np.zeros_like(x[0])
    x, y = list(x) + [z] * (n - len(x)), list(y) + [z] * (n - len(y))
    e = np.zeros_like(z)
    for q in range(n - 1, -1, -1):
        e = e + (x[q] - y[q])
    return e


def main(argv):
    raw = argv[0]
    cases = []
    for i, a in enumerate(argv):
        if a == "--case":
            name, files = argv[i + 1].rsplit("=", 1)
            f = files.split(",")
            cases.append(dict(name=name, pred=f[0], it=f[1], root=f[2], W=float(f[3]) if len(f) > 3 and f[3] else None))
    if not cases:
        sys.exit(__doc__)
    mutate = "--mutate" in argv and argv[argv.index("--mutate") + 1] == "k17"
    out_json = argv[argv.index("--json") + 1] if "--json" in argv else ""
    bad, recs = 0, []
    for ci, c in enumerate(cases):
        root = state_parts(raw, c["root"])
        pred = parts(raw, c["pred"], Q)
        it = parts(raw, c["it"], [f"z{q}" for q in range(8)])
        if mutate and ci == 0:
            it[7] = it[7].copy()
            it[7][:M] += 1e-20 * np.cos(17 * TH)
        dd = parts(raw, c["it"], [f"d{q}" for q in range(8)])
        print(f"{c['name']}: root {c['root']}")
        e_it = lead_diff(it, root)
        rec = dict(name=c["name"], root=c["root"], W=c["W"])
        for lab, e in (("prediction", lead_diff(pred, root)), ("iterate z1", e_it), ("trial z1 + d1", e_it + (dd[0] + dd[1]))):
            h = harm(np.asarray(e[:M], dtype=float))
            row = ", ".join(f"k {k} {x:.2e}" for k, x in zip(KS, h) if 7 <= k <= 17)
            print(f"  {lab}: max|z - root| {float(np.abs(e).max()):.2e}; p1 {float(np.abs(e[:M]).max()):.2e}; harmonics {row}")
            rec[lab] = dict(max=float(np.abs(e).max()), p1=float(np.abs(e[:M]).max()),
                            harmonics={int(k): float(x) for k, x in zip(KS, h)})
            if mutate and ci == 0 and lab == "iterate z1":
                h17 = h[KS.tolist().index(17)]
                ok = 0.5e-20 < h17 < 2e-20
                print(f"  control: the planted 1e-20 in k 17 reads {h17:.2e}: {'ok' if ok else 'MISSED'}")
                bad += not ok
        if c["W"] is not None:
            e13 = rec["trial z1 + d1"]["harmonics"][13]
            r = c["W"] / e13
            rec.update(e13=e13, ratio=r, within=bool(abs(r - 1) <= 0.1))
            print(f"  W {c['W']:.4e} against the trial's harmonic-13 error {e13:.4e}: W / e13 = {r:.4f}, "
                  f"{'equal to 10%' if rec['within'] else 'NOT equal to 10%'}")
        recs.append(rec)
    if out_json:
        with open(out_json, "w") as fh:
            json.dump(recs, fh, indent=1)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

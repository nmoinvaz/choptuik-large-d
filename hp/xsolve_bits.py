#!/usr/bin/env python3
"""Bit identity of hp's quad- and octo-double solvers across two builds (#487): every entry point on stored states.

    python3 xsolve_bits.py dump RAWDIR OUT.json [--perturb]   # with the build under test first on sys.path
    python3 xsolve_bits.py compare A.json B.json               # exit 1 on any difference

dump calls residual_q, jacobian_q, newton_q and residual_jet_q, and residual_o, jacobian_o, newton_o and
residual_jet_o (residuals and jets with the mutation codes 0 and the state mutation, qsolve::QD_MUT_STATE_DD = 9 and
osolve::OD_MUT_STATE_QD = 10), at the states of STATES, and records the sha256 of every returned array's bytes. The
jets take z and z/1000 as their two coefficient vectors and return two orders. --perturb adds one ulp to the first
entry of each state (the control: compare must then report differences). Run it once per build, each with its own
hpkernel first on PYTHONPATH, and compare the two files.
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# every entry point at the first state; residuals, jets and the octo-double Jacobian at the second (the LUs and
# their solves are multidouble's, untouched by #487, and an octo-double LU takes minutes)
STATES = {"anm-mac/run_trend_pin3_qd3/anm_pin3_D79.7657.npz": "all",
          "I461-win2/crawlG/anm_pin3_D232.6312.npz": "residuals"}


def digest(x):
    if isinstance(x, (list, tuple)):
        return [digest(v) for v in x]
    if isinstance(x, float):
        return np.float64(x).tobytes().hex()
    a = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    return hashlib.sha256(a.tobytes()).hexdigest()[:20] + f":{a.shape}"


def dump(raw, out, perturb):
    import hpkernel
    from hpkernel.driver import make_solver
    sys.path.insert(0, HERE)
    from octo_corrector import load_o
    rec = {"hpkernel": hpkernel.__file__, "states": {}}
    for st, scope in STATES.items():
        z8, m = load_o(os.path.join(raw, st))
        z8 = [np.ascontiguousarray(np.array(p, dtype=float)) for p in z8]
        if perturb:
            z8[0][0] = np.nextafter(z8[0][0], np.inf)
        z4 = z8[:4]
        D = m.get("D_str", repr(float(m["D"])))
        s = make_solver(D, int(m["M"]), nodes=list(m["nodes"]), kt=m.get("kt", 36), kc=m.get("kc", 146), ks=m.get("ks", 130),
                        threads=4, kfreeze=m.get("kfreeze", 3))
        s.set_segs(s.plan(z4[0], z4[1], 1e-32, 0.05))
        r = {}
        for mut_q, mut_o in ((0, 0), (9, 10)):
            r[f"residual_q/{mut_q}"] = digest(s.residual_q(*z4, mutate=mut_q))
            r[f"residual_jet_q/{mut_q}"] = digest(s.residual_jet_q([tuple(z4), tuple(p / 1000 for p in z4)], 2, mutate=mut_q))
            r[f"residual_o/{mut_o}"] = digest(s.residual_o(z8, mutate=mut_o))
            r[f"residual_jet_o/{mut_o}"] = digest(s.residual_jet_o([z8, [p / 1000 for p in z8]], 2, mutate=mut_o))
        r["jacobian_o"] = digest(s.jacobian_o(z8))
        if scope == "all":
            r["jacobian_q"] = digest(s.jacobian_q(*z4))
            nq = s.newton_q(*z4)
            r["newton_q"] = digest(list(nq[:6]) + [float(nq[6]), float(nq[7])])
            no = s.newton_o(z8)
            r["newton_o"] = digest(list(no[0]) + [no[1], no[2], float(no[3]), float(no[4])])
        rec["states"][st] = r
        print(st, "done", flush=True)
    json.dump(rec, open(out, "w"), indent=1)
    print("wrote", out)


def compare(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    bad = 0
    for st in A["states"]:
        for k, v in A["states"][st].items():
            w = B["states"].get(st, {}).get(k)
            if v != w:
                bad += 1
                print(f"DIFFERS  {st}  {k}")
    n = sum(len(v) for v in A["states"].values())
    print(f"{n - bad} of {n} outputs byte-identical")
    return 1 if bad else 0


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump(sys.argv[2], sys.argv[3], "--perturb" in sys.argv)
    else:
        sys.exit(compare(sys.argv[2], sys.argv[3]))

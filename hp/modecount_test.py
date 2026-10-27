#!/usr/bin/env python3
"""Count the growth rates of a test scalar field on the DSS solution (notes/GHOST_ADMIXTURE.md): the zeros of the
test field's determinant det M_psi(lambda) (hp/modecount_cx, CSolver.test_det) in boxes of the right half plane, by
the argument principle of hp/modecount_wind.py (function count), for one saved double-double state.

    python3 modecount_test.py STATE.npz [--M=N] [--kf=none|13] [--threads=N] [--tag=NAME] [--boxes=B0,full]
                              [--plant=RE+IMj] [count options as modecount_wind.py: --h0 --tol --Im2 --aR --aL --aLL
                              --a1 --a2]

The expected zero is lambda = 0 (psi = the background's own scalar field; its copies + 2 i k omega). The candidate
poles are those of the full det M (hp/modecount_common.horizon_poles): the test field's p goes through the same
horizon solves. --boxes: the boxes to count (default all of modecount_wind.py's; B0,full covers the same region with
half the edges: B0 around lambda = 0, full = (aL, aR) x (-y2, y2), both sectors).

--plant=z: the control of the count on this det M_psi: multiply it by (lambda - z)(lambda - conj z), so a box holding
z must count two more zeros (one for a real z). Evaluations of det M_psi are cached in
hp/data/modecount_test_<tag>_raw.jsonl and shared between plain and planted runs.

Records go to <data>/modecount_test_results.jsonl and the log to <data>/modecount_test_<tag>.log, --data (default
hp/data) in a raw-data folder (raw_out.py; --allow_unraw writes anyway).
"""
import cmath
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from modecount_wind import count, parse, DATA  # noqa: E402


def main():
    from modecount_common import setup, horizon_poles
    from gamma_dd import gamma_formula
    args, opts = parse(sys.argv[1:])
    path = args[0]
    threads = int(opts.get("threads", 1))
    data = opts.get("data", DATA)
    res_file = opts.get("res", os.path.join(data, "modecount_test_results.jsonl"))
    from raw_out import require_raw
    require_raw(res_file)
    logs = []

    def log(m):
        print(m, flush=True)
        logs.append(m)

    kf = "default" if "kf" not in opts else (None if opts["kf"] == "none" else int(opts["kf"]))
    st = setup(path, M=int(opts["M"]) if "M" in opts else None, threads=threads, kf=kf, log=log,
               fmax=float(opts.get("fmax", 1e-20)))
    D, M = st.D, st.M
    tag = opts.get("tag", f"D{D:.4f}_M{M}")
    plant = complex(opts["plant"].replace(" ", "")) if "plant" in opts else None
    run = tag + (f"_plant{plant.real:g}{plant.imag:+g}j" if plant is not None else "")
    logf = open(os.path.join(data, f"modecount_test_{tag}.log"), "a")
    for m in logs:
        logf.write(m + "\n")

    def log2(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    raw_path = os.path.join(data, f"modecount_test_{tag}_raw.jsonl")
    raw = {}
    if os.path.exists(raw_path):
        for line in open(raw_path):
            r = json.loads(line)
            raw[(r["re"], r["im"])] = r
    rf = open(raw_path, "a")

    def det_psi(re, im):
        if (re, im) not in raw:
            t = time.time()
            la, arg, pmin, pmax = st.c.test_det(st.zh, st.zl, re, im)
            r = dict(re=re, im=im, lnabs=la, arg=arg, pmin=pmin, pmax=pmax, sec=round(time.time() - t, 2))
            raw[(re, im)] = r
            rf.write(json.dumps(r) + "\n")
            rf.flush()
        r = raw[(re, im)]
        la, arg = r["lnabs"], r["arg"]
        if plant is not None:
            lam = complex(re, im)
            f = cmath.log(lam - plant) + cmath.log(lam - plant.conjugate())
            la, arg = la + f.real, arg + f.imag
        return la, arg, r["pmin"], r["pmax"]

    lam_f = 1.0 / gamma_formula(D)
    log2(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')}  test field, {os.path.basename(path)}: D {D}, M {M}, kfreeze "
         f"{st.kf}, |F| {st.F:.1e}; Delta {st.Delta:.12f}, omega {st.omega:.6f}" +
         (f"; planted zeros at {plant} and its conjugate" if plant is not None else ""))
    poles = [p for _, p in horizon_poles(st)]
    # lam_g and lam_f only place the default vertical edges a1, a2 (modecount_wind.py); they are not zeros here
    info = dict(D=D, M=M, kfreeze=st.kf, F=st.F, state=os.path.basename(path), solver="dd (hp + modecount_cx), test field",
                Delta=st.Delta, omega=st.omega, lam_g=st.lam_g, lam_f=lam_f, sector="test field",
                plant=[plant.real, plant.imag] if plant is not None else None)
    boxes = opts["boxes"].split(",") if "boxes" in opts else None
    count(det_psi, info, poles, dict(opts, cache=f"test_{run}"), log2, f"test_{run}", res_file, expected=[0.0],
          boxes_only=boxes)


if __name__ == "__main__":
    main()

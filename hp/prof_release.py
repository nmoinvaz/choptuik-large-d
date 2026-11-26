"""Where one release outer step spends its time (#1098, #1096): at a saved state and each horizon order ks given, wall
seconds of the quad-double residual and its horizon block, the quad-double Jacobian (full, and one horizon column task:
the 33 tasks of m_h and Delta each run it), the qq solve (QDLu factor and solve, and the dd solve of Delta's gain), the
double-double Jacobian, and the octo-double residual's horizon block, term by term and with relaxed products
(Solver.set_horizon_relaxed). One json line per measurement into OUT/prof.jsonl.

    python3 prof_release.py OUT STATE --ks 2602,5218 [--threads 4] [--full_ks 2602]

--full_ks: the ks at which the full Jacobians and the qq solve are timed (the Taylor part does not depend on ks).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402

from form_s import make_solver_s, opts  # noqa: E402

M = 32


def main(argv):
    from octo_corrector import load_o
    out = argv[0]
    pos, o = opts(argv[1:], dict(ks="", threads="4", full_ks="", what="all"))
    os.makedirs(out, exist_ok=True)
    path = pos[0]
    z8, meta = load_o(path)
    with mp.workdps(60):
        Dt = mp.mpf(meta["D_str"])
    thr = int(o["threads"])
    import hpkernel
    print(f"hpkernel {hpkernel.__file__}", flush=True)
    K = len(meta["nodes"])
    fh = open(os.path.join(out, "prof.jsonl"), "a")

    def rec(ks, kind, mode, sec, **kw):
        r = dict(state=os.path.basename(path), D=float(meta["D"]), ks=ks, kind=kind, relaxed=mode, seconds=round(sec, 2),
                 threads=thr, utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), **kw)
        fh.write(json.dumps(r) + "\n")
        fh.flush()
        print(f"D {meta['D']:g} ks {ks} {kind} relaxed {mode}: {sec:.1f} s {kw if kw else ''}", flush=True)

    def timed(f):
        t0 = time.time()
        v = f()
        return v, time.time() - t0

    for ks in [int(x) for x in o["ks"].split(",")]:
        s = make_solver_s(Dt, M, list(meta["nodes"]), kfreeze=3, kt=int(meta["kt"]), kc=int(meta["kc"]), ks=ks,
                          threads=thr, segs=meta["segs"], mob=float(meta["hmob"]))
        L = s.size_s
        rows = list(range(K * 3 * M, (K + 1) * 3 * M))
        mh0 = 2 * M + 3 * M * K
        z4 = [np.ascontiguousarray(z8[q]) for q in range(4)]
        z2 = z4[:2]
        z8c = [np.ascontiguousarray(z8[q]) for q in range(8)]
        full = o["full_ks"] and int(o["full_ks"]) == ks
        for mode in (0, 1):
            s.set_horizon_relaxed(mode)
            rec(ks, "qd horizon block", mode, timed(lambda: s.horizon_s(z4))[1])
            rec(ks, "qd residual", mode, timed(lambda: s.residual_s(z4))[1])
            rec(ks, "qd J, one horizon column task", mode, timed(lambda: s.jacobian_sub_s(z4, rows, [mh0], 0))[1])
            rec(ks, "dd J, one horizon column task", mode, timed(lambda: s.jacobian_sub_s(z2, rows, [mh0], 0))[1])
            if full:
                rec(ks, "od horizon block", mode, timed(lambda: s.horizon_s(z8c))[1])
                J, t = timed(lambda: s.jacobian_s(z4, 0))
                rec(ks, "qd J, full", mode, t, size=L)
                if mode == 0:
                    from multidouble import QDLu, dd_solve
                    lu, t = timed(lambda: QDLu(*[np.ascontiguousarray(np.asarray(J[q])) for q in range(4)]))
                    rec(ks, "qq: QDLu factor", mode, t)
                    F = s.residual_s(z4)
                    _, t = timed(lambda: lu.solve(*[np.ascontiguousarray(-np.asarray(F[q])) for q in range(4)]))
                    rec(ks, "qq: QDLu solve", mode, t)
                    eD = np.zeros(L)
                    eD[-1] = 1.0
                    _, t = timed(lambda: dd_solve(np.ascontiguousarray(np.asarray(J[0]).T),
                                                  np.ascontiguousarray(np.asarray(J[1]).T), eD, np.zeros(L)))
                    rec(ks, "qq: dd solve of the gain", mode, t)
                del J
                rec(ks, "dd J, full", mode, timed(lambda: s.jacobian_s(z2, 0))[1], size=L)
                rec(ks, "od residual", mode, timed(lambda: s.residual_s(z8c))[1])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

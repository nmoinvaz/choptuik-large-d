"""#1076: a point's M = 3 value at xi1 with the inner solve in mpmath (inner_mp.py), for points whose float64 value
the 10 sigma guard blocks at its rounding floor.

For a point (A, w_f) with its float64 record's F_3, v = (B_3, C_3) = (-2 Im F_3, 2 Re F_3) starts a Newton solve of
r_3 = 0 in mpmath (central differences at 1e-12 |v| for the Jacobian); then R_1 = r[:2], and the same at a second
precision. sigma_mp = |R_1(dps) - R_1(dps2)| + |J_1 J_3^-1| |r_3| (the residual r_3 left in R_1, as argcount's dr1),
with a floor of 10^-(dps - 6) |R_1|. A value is valid when |R_1| >= 10 sigma_mp, max|Y| <= 100 (fu_value's tests) and
|r_3| < 1e-8 (argcount's valued3 reads it as `info`); the |dF_3| neighbour test is applied by the recount's merge.

    python3 value_mp.py point A WF F3RE F3IM [--r1=0.17] [--dps=34] [--dps2=42] [--order=40] [--order2=50] [--iters=8]
    python3 value_mp.py pointv A WF --M=5 --v=B3,C3,B5,C5 [same options]   M harmonics: v solved from the given start
    python3 value_mp.py check1 CACHE.jsonl [--n=3] [...]   check 1 of #1076 on the n best-valued float64 points
    python3 value_mp.py check2 CACHE.jsonl [--n=1]         check 2: R_1 at (34, 40, margin 12) against (42, 50, 14)
    python3 value_mp.py batch CACHE ROWS --points="A,w;A,w" [--procs=N]   value the points, write every row to ROWS,
        append the valued ones to CACHE (as argcount reads it: later records of a key win)
"""
import json
import os
import sys
import time

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import inner_mp as IM  # noqa: E402

PROGRESS = os.environ.get("VALUE_MP_PROGRESS") == "1"


def solve(A, wf, v0, r1, dps, p, iters=8, hrel="1e-12", margin=12, J=None, chord="1e-4", M=3, damp=0):
    """Newton on r_3 = ... = r_M = 0 for v = (B_3, C_3, ..., B_M, C_M) at precision dps, with chord steps
    (Shamanskii): the Jacobian (central differences, 2(M - 1) calls) is refreshed only after a step whose contraction
    max|r'|/max|r| exceeds `chord`, or after a step that fails; J = (J1, J3) starts from a given Jacobian (the second
    precision reuses the first's). chord=None is plain Newton. damp=k halves a step that fails with a fresh Jacobian up
    to k times (Armijo-free backtracking on max|r|) while max|r| > 1e-12; damp=0 (M = 3's default) stops there instead.
    Returns (R_1, (r_3, ..., r_M), v, J1, J3, info, calls)"""
    mp.mp.dps = dps
    n = M - 1                                   # unknowns, and rows r_3 .. r_M (real and imaginary parts)
    v = [mp.mpf(x) for x in v0]
    assert len(v) == n
    calls = 0

    t0 = time.time()

    def res(vv):
        nonlocal calls
        calls += 1
        out = IM.residual_demixed([A, wf] + list(vv), M, r1, p=p, margin=margin)
        if PROGRESS:                            # VALUE_MP_PROGRESS=1: one stderr line per residual call (#1080)
            print(f"  solve dps {dps}: call {calls}, max|r| {mp.nstr(max(abs(x) for x in out[0][2:]), 3)}, "
                  f"{time.time() - t0:.0f} s", file=sys.stderr, flush=True)
        return out

    def jac(vv):
        hs = mp.mpf(hrel) * max(max(abs(x) for x in vv), mp.mpf("1e-3"))
        cols = []
        for k in range(n):
            vp, vm = list(vv), list(vv)
            vp[k] += hs
            vm[k] -= hs
            rp, _ = res(vp)
            rm, _ = res(vm)
            cols.append([(rp[i] - rm[i]) / (2 * hs) for i in range(2 + n)])
        return (mp.matrix([[cols[k][i] for k in range(n)] for i in range(2)]),
                mp.matrix([[cols[k][2 + i] for k in range(n)] for i in range(n)]))

    r, info = res(v)
    J1, J3 = J if J is not None else (None, None)
    stale = J is None
    fresh = False
    tol = mp.mpf(10) ** (-(dps - 6))
    for _ in range(iters):
        if max(abs(x) for x in r[2:]) < tol:
            break
        if stale:
            J1, J3 = jac(v)
            stale, fresh = False, True
        dv = mp.lu_solve(J3, mp.matrix([-x for x in r[2:]]))
        vn = [v[k] + dv[k] for k in range(n)]
        rn, infon = res(vn)
        n0, n1 = max(abs(x) for x in r[2:]), max(abs(x) for x in rn[2:])
        if n1 >= n0 and fresh and damp and n0 > mp.mpf("1e-12"):
            t = mp.mpf(1)                  # far from the solution only: at the floor a failed step ends the solve
            for _ in range(damp):          # backtrack along the fresh Newton step
                t /= 2
                vn = [v[k] + t * dv[k] for k in range(n)]
                rn, infon = res(vn)
                n1 = max(abs(x) for x in rn[2:])
                if n1 < n0:
                    break
        if n1 >= n0:
            if fresh:
                break                      # no progress with a fresh Jacobian
            stale = True                   # a stale Jacobian failed: refresh and retry from v
            continue
        v, r, info = vn, rn, infon
        fresh = False
        if chord is None or n1 > mp.mpf(chord) * n0:
            stale = True
    if J3 is None:
        J1, J3 = jac(v)
    return mp.mpc(r[0], r[1]), tuple(r[2:]), v, J1, J3, info, calls


def value_point(A, wf, f3re, f3im, r1="0.17", dps=34, dps2=42, p=40, p2=50, iters=12, chord="1e-4"):
    return value_v(A, wf, [-2 * mp.mpf(f3im), 2 * mp.mpf(f3re)], 3, r1, dps, dps2, p, p2, iters, chord)


def value_v(A, wf, v0, M=3, r1="0.17", dps=34, dps2=42, p=40, p2=50, iters=12, chord="1e-4", damp=None, J=None,
            single=False, want_J=False):
    """R_1 at (A, w_f) with the harmonics v = (B_3, C_3, ..., B_M, C_M) solved from v0 at two precisions. J = (J1, J3)
    as nested lists of decimal strings starts the chord iteration from a given Jacobian (a walk's offset points take
    the centre's); single=True skips the second precision (sig is then the r-residual term and the floor only);
    want_J=True returns the Jacobian used, as strings, in out["J"] (not JSON-written by the CLI)."""
    t0 = time.time()
    mp.mp.dps = dps
    v0 = [mp.mpf(x) for x in v0]
    damp = (0 if M == 3 else 8) if damp is None else damp       # M = 3 keeps #1076's undamped solve
    Jm = None if J is None else tuple(mp.matrix([[mp.mpf(x) for x in row] for row in B]) for B in J)
    R, r3, v, J1, J3, info, n1 = solve(A, wf, v0, r1, dps, p, iters, chord=chord, M=M, damp=damp, J=Jm)
    if single:
        R2, r32, v2, info2, n2 = R, r3, v, info, 0
    else:
        R2, r32, v2, _, _, info2, n2 = solve(A, wf, [str(x) for x in v], r1, dps2, p2, iters,
                                             J=None if chord is None else (J1, J3), chord=chord, M=M, damp=damp)
    mp.mp.dps = dps2
    dR = abs(R2 - R)
    if J1 is not None and J3 is not None:
        G = J1 * mp.inverse(J3)
        r3n = max(abs(x) for x in r3)
        dr1 = max(sum(abs(G[i, k]) for k in range(M - 1)) for i in range(2)) * r3n
    else:
        dr1 = mp.inf
    sig = max(dR + dr1, mp.mpf(10) ** (-(dps - 6)) * abs(R2))
    r3n2 = max(abs(x) for x in r32)
    ok = abs(R2) >= 10 * sig and info2["ymax"] <= 100 and r3n2 < mp.mpf("1e-8")   # argcount valued3: info < 1e-8
    out = dict(A=float(A), wf=float(wf), re=float(mp.re(R2)), im=float(mp.im(R2)), sig=float(sig),
               info=float(r3n2),
               R_over_sig=float(abs(R2) / sig), valued=bool(ok), dR=float(dR), dr1=float(dr1),
               r3=[float(r32[0]), float(r32[1])], F3re=float(v2[1] / 2), F3im=float(-v2[0] / 2),
               ymax=float(info2["ymax"]), r1=float(r1), mp_dps=[dps, dps2], order=[p, p2], calls=[n1, n2],
               steps=int(info2["steps"]), terr=float(info2["terr"]), seconds=round(time.time() - t0, 1), fu="mp")
    if M > 3:                                   # the higher harmonics' residuals and the solved v, as strings
        out.update(M=M, r_rest=[float(x) for x in r32], v=[mp.nstr(x, dps2) for x in v2], single=bool(single))
    if want_J and J1 is not None:
        out["J"] = tuple([[mp.nstr(B[i, k], dps) for k in range(B.cols)] for i in range(B.rows)] for B in (J1, J3))
    return out


def _batch_one(a):
    (A, wf, f3re, f3im), kw = a
    return value_point(A, wf, f3re, f3im, **kw)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    for m_ in opts.get("mutate", "").split(","):
        if m_:
            IM.MUTATE.add(m_)
    kw = dict(r1=opts.get("r1", "0.17"), dps=int(opts.get("dps", 34)), dps2=int(opts.get("dps2", 42)),
              p=int(opts.get("order", 40)), p2=int(opts.get("order2", 50)), iters=int(opts.get("iters", 8)))
    if args and args[0] == "point":
        A, wf, f3re, f3im = args[1:5]
        print(json.dumps(value_point(A, wf, f3re, f3im, **kw)), flush=True)
        return 0
    if args and args[0] == "pointv":                # pointv A WF --M=5 --v=B3,C3,B5,C5 [--damp=K] (#1080)
        M = int(opts.get("M", "3"))
        dmp = int(opts["damp"]) if "damp" in opts else None
        print(json.dumps(value_v(args[1], args[2], opts["v"].split(","), M, damp=dmp, **kw)), flush=True)
        return 0
    if args and args[0] == "check1":
        recs = [json.loads(l) for l in open(args[1]) if l.strip()]
        good = [r for r in recs if r.get("sig") and r.get("re") is not None and abs(complex(r["re"], r["im"])) >= 1e3 * r["sig"]
                and abs(r.get("r1", 0.17) - float(kw["r1"])) < 1e-12]
        good.sort(key=lambda r: -abs(complex(r["re"], r["im"])) / r["sig"])
        ok_all = True
        for r in good[: int(opts.get("n", 3))]:
            row = value_point(str(r["A"]), str(r["wf"]), str(r["F3re"]), str(r["F3im"]), **kw)
            dR = abs(complex(row["re"], row["im"]) - complex(r["re"], r["im"]))
            dv = max(abs(row["F3re"] - r["F3re"]), abs(row["F3im"] - r["F3im"])) * 2
            ok = dR <= r["sig"] and dv <= 1e-9
            ok_all &= ok
            print(f"({r['A']}, {r['wf']}): |R_mp - R_f64| {dR:.3e} vs f64 sigma {r['sig']:.3e}; |v_mp - v_f64| {dv:.2e} "
                  f"(<= 1e-9); sigma_mp {row['sig']:.2e}, |R|/sigma_mp {row['R_over_sig']:.2e}, {row['seconds']} s: "
                  f"{'PASS' if ok else 'FAIL'}", flush=True)
        print(f"check 1: {'PASS' if ok_all else 'FAIL'}", flush=True)
        return 0 if ok_all else 1
    if args and args[0] == "batch":
        cache, rows = args[1], args[2]
        recs = {}
        for line in open(cache):
            r = json.loads(line)
            recs[(round(r["A"], 12), round(r["wf"], 12))] = r
        pts = [tuple(float(t) for t in q.split(",")) for q in opts.get("points", "").split(";") if q]
        if any(len(q) != 2 for q in pts) or len(args) != 3:
            print("batch takes CACHE ROWS --points=\"A,w;A,w\"", file=sys.stderr)
            return 2
        jobs = []
        for A, wf in pts:
            r = recs.get((round(A, 12), round(wf, 12)))
            if r is None or r.get("F3re") is None or not all(map(lambda t: isinstance(t, float) and t == t, (r["F3re"], r["F3im"]))):
                with open(rows, "a") as fh:
                    fh.write(json.dumps(dict(A=A, wf=wf, skipped="no float64 F_3 to start from", fu="mp")) + "\n")
                continue
            jobs.append((repr(r["A"]), repr(r["wf"]), repr(r["F3re"]), repr(r["F3im"])))
        procs = int(opts.get("procs", 1))
        fn = _batch_one
        it = map(fn, [(j, kw) for j in jobs]) if procs <= 1 else __import__("multiprocessing").get_context("spawn").Pool(procs).imap_unordered(fn, [(j, kw) for j in jobs])
        nval = 0
        for row in it:
            with open(rows, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            if row.get("valued"):
                nval += 1
                with open(cache, "a") as fh:
                    fh.write(json.dumps({k: row[k] for k in ("A", "wf", "re", "im", "info", "r1", "F3re", "F3im", "ymax", "sig", "fu", "mp_dps")}) + "\n")
            print(f"({row['A']}, {row['wf']}): |R|/sigma_mp {row['R_over_sig']:.3g}, valued {row['valued']}, {row['seconds']} s", flush=True)
        print(f"valued {nval} of {len(jobs)} points (skipped {len(pts) - len(jobs)}); rows in {rows}", flush=True)
        return 0
    if args and args[0] == "check2":
        recs = [json.loads(l) for l in open(args[1]) if l.strip()]
        good = [r for r in recs if r.get("sig") and r.get("re") is not None and abs(complex(r["re"], r["im"])) >= 1e3 * r["sig"]
                and abs(r.get("r1", 0.17) - float(kw["r1"])) < 1e-12]
        good.sort(key=lambda r: -abs(complex(r["re"], r["im"])) / r["sig"])
        ok_all = True
        for r in good[: int(opts.get("n", 1))]:
            v0 = [-2 * mp.mpf(str(r["F3im"])), 2 * mp.mpf(str(r["F3re"]))]
            Ra, _, va, *_ = solve(str(r["A"]), str(r["wf"]), v0, kw["r1"], 34, 40, kw["iters"], margin=12)
            Rb, _, vb, *_ = solve(str(r["A"]), str(r["wf"]), [str(va[0]), str(va[1])], kw["r1"], 42, 50, kw["iters"], margin=14)
            mp.mp.dps = 42
            rel = abs(Rb - Ra) / abs(Rb)
            ok = rel <= mp.mpf("1e-25")
            ok_all &= ok
            print(f"({r['A']}, {r['wf']}): R_1 at (dps 34, order 40, margin 12) and (42, 50, 14) differ by "
                  f"{mp.nstr(rel, 3)} relative (<= 1e-25): {'PASS' if ok else 'FAIL'}", flush=True)
        print(f"check 2: {'PASS' if ok_all else 'FAIL'}", flush=True)
        return 0 if ok_all else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())

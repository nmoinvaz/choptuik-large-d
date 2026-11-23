"""#1076: a point's M = 3 value at xi1 with the inner solve in mpmath (inner_mp.py), for points whose float64 value
the 10 sigma guard blocks at its rounding floor.

For a point (A, w_f) with its float64 record's F_3, v = (B_3, C_3) = (-2 Im F_3, 2 Re F_3) starts a Newton solve of
r_3 = 0 in mpmath (central differences at 1e-12 |v| for the Jacobian); then R_1 = r[:2], and the same at a second
precision. sigma_mp = |R_1(dps) - R_1(dps2)| + |J_1 J_3^-1| |r_3| (the residual r_3 left in R_1, as argcount's dr1),
with a floor of 10^-(dps - 6) |R_1|. A value is valid when |R_1| >= 10 sigma_mp, max|Y| <= 100 (fu_value's tests) and
|r_3| < 1e-8 (argcount's valued3 reads it as `info`); the |dF_3| neighbour test is applied by the recount's merge.

    python3 value_mp.py point A WF F3RE F3IM [--r1=0.17] [--dps=34] [--dps2=42] [--order=40] [--order2=50] [--iters=8]
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


def solve(A, wf, v0, r1, dps, p, iters=8, hrel="1e-12", margin=12, J=None, chord="1e-4"):
    """Newton on r_3 = 0 at precision dps, with chord steps (Shamanskii): the Jacobian (central differences, 4 calls)
    is refreshed only after a step whose contraction |r_3'|/|r_3| exceeds `chord`, or after a step that fails; J =
    (J1, J3) starts from a given Jacobian (the second precision reuses the first's). chord=None is plain Newton.
    Returns (R_1, r_3, v, J1, J3, info, calls)"""
    mp.mp.dps = dps
    v = [mp.mpf(v0[0]), mp.mpf(v0[1])]
    calls = 0

    def res(vv):
        nonlocal calls
        calls += 1
        return IM.residual_demixed([A, wf, vv[0], vv[1]], 3, r1, p=p, margin=margin)

    def jac(vv):
        hs = mp.mpf(hrel) * max(abs(vv[0]), abs(vv[1]), mp.mpf("1e-3"))
        cols = []
        for k in range(2):
            vp, vm = list(vv), list(vv)
            vp[k] += hs
            vm[k] -= hs
            rp, _ = res(vp)
            rm, _ = res(vm)
            cols.append([(rp[i] - rm[i]) / (2 * hs) for i in range(4)])
        return (mp.matrix([[cols[0][0], cols[1][0]], [cols[0][1], cols[1][1]]]),
                mp.matrix([[cols[0][2], cols[1][2]], [cols[0][3], cols[1][3]]]))

    r, info = res(v)
    J1, J3 = J if J is not None else (None, None)
    stale = J is None
    fresh = False
    tol = mp.mpf(10) ** (-(dps - 6))
    for _ in range(iters):
        if max(abs(r[2]), abs(r[3])) < tol:
            break
        if stale:
            J1, J3 = jac(v)
            stale, fresh = False, True
        dv = mp.lu_solve(J3, mp.matrix([-r[2], -r[3]]))
        vn = [v[0] + dv[0], v[1] + dv[1]]
        rn, infon = res(vn)
        n0, n1 = max(abs(r[2]), abs(r[3])), max(abs(rn[2]), abs(rn[3]))
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
    return mp.mpc(r[0], r[1]), (r[2], r[3]), v, J1, J3, info, calls


def value_point(A, wf, f3re, f3im, r1="0.17", dps=34, dps2=42, p=40, p2=50, iters=12, chord="1e-4"):
    t0 = time.time()
    v0 = [-2 * mp.mpf(f3im), 2 * mp.mpf(f3re)]
    R, r3, v, J1, J3, info, n1 = solve(A, wf, v0, r1, dps, p, iters, chord=chord)
    R2, r32, v2, _, _, info2, n2 = solve(A, wf, [str(v[0]), str(v[1])], r1, dps2, p2, iters,
                                         J=None if chord is None else (J1, J3), chord=chord)
    mp.mp.dps = dps2
    dR = abs(R2 - R)
    if J1 is not None and J3 is not None:
        G = J1 * mp.inverse(J3)
        r3n = max(abs(r3[0]), abs(r3[1]))
        dr1 = max(abs(G[0, 0]) + abs(G[0, 1]), abs(G[1, 0]) + abs(G[1, 1])) * r3n
    else:
        dr1 = mp.inf
    sig = max(dR + dr1, mp.mpf(10) ** (-(dps - 6)) * abs(R2))
    r3n2 = max(abs(r32[0]), abs(r32[1]))
    ok = abs(R2) >= 10 * sig and info2["ymax"] <= 100 and r3n2 < mp.mpf("1e-8")   # argcount valued3: info < 1e-8
    return dict(A=float(A), wf=float(wf), re=float(mp.re(R2)), im=float(mp.im(R2)), sig=float(sig),
                info=float(r3n2),
                R_over_sig=float(abs(R2) / sig), valued=bool(ok), dR=float(dR), dr1=float(dr1),
                r3=[float(r32[0]), float(r32[1])], F3re=float(v2[1] / 2), F3im=float(-v2[0] / 2),
                ymax=float(info2["ymax"]), r1=float(r1), mp_dps=[dps, dps2], order=[p, p2], calls=[n1, n2],
                steps=int(info2["steps"]), terr=float(info2["terr"]), seconds=round(time.time() - t0, 1), fu="mp")


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

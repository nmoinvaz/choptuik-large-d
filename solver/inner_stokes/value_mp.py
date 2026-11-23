"""#1076: a point's M = 3 value at xi1 with the inner solve in mpmath (inner_mp.py), for points whose float64 value
the 10 sigma guard blocks at its rounding floor.

For a point (A, w_f) with its float64 record's F_3, v = (B_3, C_3) = (-2 Im F_3, 2 Re F_3) starts a Newton solve of
r_3 = 0 in mpmath (central differences at 1e-12 |v| for the Jacobian); then R_1 = r[:2], and the same at a second
precision. sigma_mp = |R_1(dps) - R_1(dps2)| + |J_1 J_3^-1| |r_3| (the residual r_3 left in R_1, as argcount's dr1),
with a floor of 10^-(dps - 6) |R_1|. A value is valid when |R_1| >= 10 sigma_mp and max|Y| <= 100 (fu_value's tests;
the |dF_3| neighbour test is applied by the recount's merge, not here).

    python3 value_mp.py point A WF F3RE F3IM [--r1=0.17] [--dps=34] [--dps2=42] [--order=40] [--order2=50] [--iters=8]
    python3 value_mp.py check1 CACHE.jsonl [--n=3] [...]   check 1 of #1076 on the n best-valued float64 points
    python3 value_mp.py check2 CACHE.jsonl [--n=1]         check 2: R_1 at (34, 40, margin 12) against (42, 50, 14)
"""
import json
import os
import sys
import time

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import inner_mp as IM  # noqa: E402


def solve(A, wf, v0, r1, dps, p, iters=8, hrel="1e-12", margin=12):
    """Newton on r_3 = 0 at precision dps; returns (R_1, r_3, v, J1, J3, info, calls)"""
    mp.mp.dps = dps
    v = [mp.mpf(v0[0]), mp.mpf(v0[1])]
    calls = 0

    def res(vv):
        nonlocal calls
        calls += 1
        return IM.residual_demixed([A, wf, vv[0], vv[1]], 3, r1, p=p, margin=margin)

    r, info = res(v)
    J1 = J3 = None
    for _ in range(iters):
        hs = mp.mpf(hrel) * max(abs(v[0]), abs(v[1]), mp.mpf("1e-3"))
        cols = []
        for k in range(2):
            vp, vm = list(v), list(v)
            vp[k] += hs
            vm[k] -= hs
            rp, _ = res(vp)
            rm, _ = res(vm)
            cols.append([(rp[i] - rm[i]) / (2 * hs) for i in range(4)])
        J1 = mp.matrix([[cols[0][0], cols[1][0]], [cols[0][1], cols[1][1]]])
        J3 = mp.matrix([[cols[0][2], cols[1][2]], [cols[0][3], cols[1][3]]])
        dv = mp.lu_solve(J3, mp.matrix([-r[2], -r[3]]))
        vn = [v[0] + dv[0], v[1] + dv[1]]
        rn, infon = res(vn)
        if max(abs(rn[2]), abs(rn[3])) >= max(abs(r[2]), abs(r[3])):
            break
        v, r, info = vn, rn, infon
        if max(abs(r[2]), abs(r[3])) < mp.mpf(10) ** (-(dps - 6)):
            break
    return mp.mpc(r[0], r[1]), (r[2], r[3]), v, J1, J3, info, calls


def value_point(A, wf, f3re, f3im, r1="0.17", dps=34, dps2=42, p=40, p2=50, iters=8):
    t0 = time.time()
    v0 = [-2 * mp.mpf(f3im), 2 * mp.mpf(f3re)]
    R, r3, v, J1, J3, info, n1 = solve(A, wf, v0, r1, dps, p, iters)
    R2, r32, v2, _, _, info2, n2 = solve(A, wf, [str(v[0]), str(v[1])], r1, dps2, p2, iters)
    mp.mp.dps = dps2
    dR = abs(R2 - R)
    if J1 is not None and J3 is not None:
        G = J1 * mp.inverse(J3)
        r3n = max(abs(r3[0]), abs(r3[1]))
        dr1 = max(abs(G[0, 0]) + abs(G[0, 1]), abs(G[1, 0]) + abs(G[1, 1])) * r3n
    else:
        dr1 = mp.inf
    sig = max(dR + dr1, mp.mpf(10) ** (-(dps - 6)) * abs(R2))
    ok = abs(R2) >= 10 * sig and info2["ymax"] <= 100
    return dict(A=float(A), wf=float(wf), re=float(mp.re(R2)), im=float(mp.im(R2)), sig=float(sig),
                R_over_sig=float(abs(R2) / sig), valued=bool(ok), dR=float(dR), dr1=float(dr1),
                r3=[float(r32[0]), float(r32[1])], F3re=float(v2[1] / 2), F3im=float(-v2[0] / 2),
                ymax=float(info2["ymax"]), r1=float(r1), mp_dps=[dps, dps2], order=[p, p2], calls=[n1, n2],
                steps=int(info2["steps"]), terr=float(info2["terr"]), seconds=round(time.time() - t0, 1), fu="mp")


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

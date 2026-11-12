#!/usr/bin/env python3
"""A corrector whose first step puts the weak subspace on its root (#499, step 4), A/B against newton_od.

    cd hp && python3 weak_corrector.py RAWDIR DZ0 BASE [--mode=weak|newton] [--k=4] [--hs=1e-23,1e-24] [--tol=1e-60]
        [--it=12] [--lam_stall=0.04] [--threads=2] [--json=OUT]

DZ0 is anm_run.py --save_dz0's file of an attempt (its corrector start z0 and first Newton step dz0), BASE the state
it stepped from (plan and sizes), both relative to RAWDIR. The kernel at the attempt's D is rebuilt as dz0_weak.py
does, and the attempt's pinned system is recovered from the saved step: the held offset is c = F_bare(z0) + J dz0,
with J the exact octo-double Jacobian summed at 140 digits, kept on the horizon block's p rows (elsewhere it must be
below 1e-6 of its largest entry, which is dz0_weak.py's check). Check: the recovered system's max|F| at z0 equals the
attempt's logged start max|F| to 1% (printed beside the value for the comparison).

--mode=newton runs newton_od (hp/octo_corrector.py, the chain's corrector rule) from z0 on that system: the control,
which must repeat the attempt's logged Newton sequence and its refusal. --mode=weak replaces the first step's part on
the four weakest directions: z1 = z0 + V c* + (dz0 - V a), with a = V^T dz0 and c* the root of the cubic reduced
system B (c - a) + T(c, c, c) / 6 = 0 (dz0_weak.cubic) whose actual weak residual is smallest, at most 5e-2 of
|B a|. Then newton_od runs from z1. Printed: max|F| at z0 and z1, newton_od's sequence and outcome, Delta.
"""
import json
import os
import sys
import time

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dz0_weak import cubic  # noqa: E402
from freeze_round import mpv, parts8  # noqa: E402
from hpkernel.driver import make_solver  # noqa: E402
from multidouble import od_axpy  # noqa: E402
from near_null import subspace  # noqa: E402
from octo_corrector import PinnedO, load_o, newton_od, o8, omax  # noqa: E402
from weak_curv import mgs  # noqa: E402

mp.mp.dps = 140


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    if len(args) != 3:
        sys.exit(__doc__)
    raw, dzf, base = args
    mode, k = opts.get("mode", "weak"), int(opts.get("k", 4))
    tol, it, lam_stall = float(opts.get("tol", 1e-60)), int(opts.get("it", 12)), float(opts.get("lam_stall", 0.04))
    if mode not in ("weak", "newton"):
        sys.exit(f"--mode={mode}: weak or newton")
    f = np.load(os.path.join(raw, dzf))
    z0 = o8([f[f"z{q}"] for q in range(8)])
    d0 = o8([f[f"d{q}"] for q in range(8)])
    md = json.loads(str(f["meta"]))
    _, mb = load_o(os.path.join(raw, base))
    M, nodes, kf = int(mb["M"]), list(mb["nodes"]), int(mb.get("kfreeze", 3))
    K, L = len(nodes), len(z0[0])
    D = mp.mpf(md["D_str"]) if md.get("D_str") else mp.mpf(mb["D_str"]) + mp.mpf(md["a"])
    s = make_solver(str(D), M, nodes=nodes, kt=mb.get("kt", 36), kc=mb.get("kc", 146), ks=mb.get("ks", 130),
                    threads=int(opts.get("threads", 2)), kfreeze=kf)
    s.set_segs(mb["segs"])
    rec = dict(dz0=dzf, base=base, D=float(D), a=md.get("a"), mode=mode, k=k, tol=tol)
    t0 = time.time()

    # the held offset c = F_bare(z0) + J dz0, on the horizon block's p rows
    J = [np.asarray(p) for p in s.jacobian_o(list(z0))]
    dv = mpv(d0, range(L))
    nz = [c for c in range(L) if dv[c] != 0]
    jd = [mp.fsum(mp.fsum(mp.mpf(float(J[q][r, c])) for q in range(8) if J[q][r, c] != 0) * dv[c] for c in nz)
          for r in range(L)]
    Fb = mpv(o8(s.residual_o(list(z0))), range(L))
    coff = [x + y for x, y in zip(Fb, jd)]
    hp = set(range(4 * M * K, 4 * M * K + M))
    on, off = max(abs(coff[r]) for r in hp), max(abs(coff[r]) for r in range(L) if r not in hp)
    ok = bool(off < mp.mpf("1e-6") * on)
    c8 = parts8([coff[r] if r in hp else mp.mpf(0) for r in range(L)])
    so = PinnedO(s, c8[0], c8[1], c8=c8)
    F0 = so.residual_o(z0)
    f0 = omax(F0)
    print(f"{dzf}: D {mp.nstr(D, 12)} (a {md.get('a')}), mode {mode}; held offset on its rows {mp.nstr(on, 3)}, "
          f"elsewhere {mp.nstr(off, 3)} (bound 1e-6 of the first): {'pass' if ok else 'FAIL'}; the recovered system's "
          f"max|F| at z0 {f0:.3e} (compare with the attempt's logged start) ({time.time() - t0:.0f} s)", flush=True)
    rec.update(offset_check=ok, F0=f0)
    if not ok:
        print("check: FAIL")
        return 1

    start = z0
    if mode == "weak":
        t0 = time.time()
        s.newton_o(list(z0))           # the LU at z0, for the basis

        def solve(b, transpose=False):
            x = s.olu_solve([np.ascontiguousarray(b)] + [np.zeros(L)] * 7, transpose)
            return np.asarray(x[0]) + np.asarray(x[1])
        V, _ = subspace(solve, L, k, int(opts.get("its", 16)))
        X = mgs([[mp.mpf(float(x)) for x in V[:, j]] for j in range(k)])[0]
        for _ in range(3):
            X = mgs([mpv(o8(s.olu_solve(list(o8(s.olu_solve(list(parts8(x)), True))))), range(L)) for x in X])[0]
        a = [mp.fsum(x * y for x, y in zip(X[j], dv)) for j in range(k)]
        hs = [float(x) for x in opts.get("hs", "1e-23,1e-24").split(",")]
        cr = {}
        ok_t = cubic(s, z0, X, a, L, k, hs, int(opts.get("seed", 3)), cr)
        roots, actual = cr["cubic"]["roots"], cr["cubic"]["actual"]
        best = min(range(len(actual)), key=lambda i: actual[i]) if actual else None
        if not ok_t or best is None or actual[best] > 5e-2:
            print(f"  no verified weak root (T check {'pass' if ok_t else 'FAIL'}, actual residuals {actual}): stopping")
            print("check: FAIL")
            return 1
        cs = [mp.mpf(x) for x in roots[best]]
        # z1 = z0 + V c* + (dz0 - V a): the first step with its weak part on the root
        p = [dv[r] + mp.fsum((cs[j] - a[j]) * X[j][r] for j in range(k)) for r in range(L)]
        start = o8(od_axpy(1.0, *parts8(p), *z0))
        f1 = omax(so.residual_o(start))
        print(f"  weak first step: |a| {mp.nstr(mp.sqrt(mp.fsum(x * x for x in a)), 3)} -> the root |c*| "
              f"{mp.nstr(mp.sqrt(mp.fsum(x * x for x in cs)), 3)} (actual weak residual {actual[best]:.2e} of |B a|); "
              f"max|F| at z1 {f1:.3e} ({time.time() - t0:.0f} s)", flush=True)
        rec.update(root=[float(x) for x in cs], root_actual=actual[best], F1=f1)

    t0 = time.time()
    first = {}

    def on_first(zz, dd):
        first["z"], first["d"] = o8(zz), o8(dd)
    z, hist, conv = newton_od(so, start, tol=tol, it_max=it, verbose=True, lam_stall=lam_stall, on_first=on_first)
    if mode == "weak" and first:
        # the first step newton_od takes from z1: its span(V) part and the rest, and the weak residual of the reduced
        # model's coordinates (U^T [F - F(z0)] - B a) at z1 and at z1 + dz
        Um, Rm = mgs([mpv(o8(s.olu_solve(list(parts8(x)), True)), range(L)) for x in X])
        B = (mp.matrix(Rm) ** -1).T
        Ba = B * mp.matrix(a)
        nBa = mp.sqrt(mp.fsum(x * x for x in Ba))
        F0b = o8(s.residual_o(list(z0)))

        def weak_res(zz):
            dv2 = mpv(o8(od_axpy(-1.0, *F0b, *o8(s.residual_o(list(zz))))), range(L))
            gv = [mp.fsum(u * f for u, f in zip(Um[i], dv2)) - Ba[i] for i in range(k)]
            return mp.sqrt(mp.fsum(x * x for x in gv)) / nBa

        dd = mpv(first["d"], range(L))
        cd = [mp.fsum(x * y for x, y in zip(X[j], dd)) for j in range(k)]
        on_v = mp.sqrt(mp.fsum(x * x for x in cd))
        tot = mp.sqrt(mp.fsum(x * x for x in dd))
        z2 = o8(od_axpy(1.0, *first["d"], *first["z"]))
        print(f"  newton_od's first step from z1: 2-norm {mp.nstr(tot, 3)}, on span(V) {mp.nstr(on_v, 3)} (coefficients "
              + ", ".join(mp.nstr(x, 3) for x in cd) + f"), the rest {mp.nstr(mp.sqrt(max(tot ** 2 - on_v ** 2, 0)), 3)}; "
              f"weak residual / |B a| at z1 {mp.nstr(weak_res(first['z']), 3)}, at z1 + dz {mp.nstr(weak_res(z2), 3)}",
              flush=True)
        rec.update(step1=dict(norm=float(tot), on_V=float(on_v), coeffs=[float(x) for x in cd]))
    Delta = float(z[0][-1] + z[1][-1])
    print(f"  newton_od from {'z1' if mode == 'weak' else 'z0'}: {'converged' if conv else 'NOT converged'}, "
          f"{len(hist)} steps, max|F| {omax(so.residual_o(z)):.3e}, Delta {mp.nstr(mp.fsum(mp.mpf(float(z[q][-1])) for q in range(8)), 25)} "
          f"({time.time() - t0:.0f} s)", flush=True)
    rec.update(converged=bool(conv), steps=len(hist), hist=hist, Delta=Delta)
    print("result:", "converged" if conv else "not converged")
    if opts.get("json"):
        with open(opts["json"], "w") as fh:
            json.dump(rec, fh, indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())

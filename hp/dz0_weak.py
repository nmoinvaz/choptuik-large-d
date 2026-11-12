#!/usr/bin/env python3
"""The weak subspace's part of a corrector attempt's first Newton step (#499).

    cd hp && python3 dz0_weak.py RAWDIR DZ0 BASE [--k=4] [--its=16] [--refine=3] [--threads=2] [--json=OUT]
        [--mutate=D]

DZ0 is anm_run.py --save_dz0's file (the attempt's corrector start z0 and its first Newton step dz0, eight parts each),
BASE the stored state the attempt stepped from (its plan and the solver's sizes), both relative to RAWDIR. The kernel
at the attempt's D, on the base's plan, is factored at z0 in octo-double (Solver.newton_o). Its k weakest right
singular vectors come from near_null.subspace and are refined as weak_curv.py refines them (block inverse iteration
through the LU, orthonormal at 140 digits). Printed: |dz0| in max and 2-norms, its coefficients on the refined basis,
the 2-norms of its part on span(V) and of the rest, and the rest's largest components by block and harmonic.

Check (exit 1 on failure): the rebuilt system is the attempt's. Below ElimO's switch the saved step is the pinned
step -J^-1 (F - c), and the bare kernel's step at z0 is -J^-1 F, so their difference is J^-1 c, with the held offset c
only on the horizon block's p rows. J (dz0 - dz_bare), with J the exact octo-double Jacobian (Solver.jacobian_o) summed
at 140 digits, must stay below 1e-6 of its largest entry on those rows everywhere else. Control --mutate=D rebuilds
the kernel at D + 1e-6, and the check must fail.
"""
import json
import os
import sys
import time

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from freeze_round import mpv, parts8  # noqa: E402
from hpkernel.driver import make_solver  # noqa: E402
from multidouble import od_axpy  # noqa: E402
from near_null import harmonics, layout, subspace  # noqa: E402
from octo_corrector import load_o, o8  # noqa: E402
from weak_curv import mgs  # noqa: E402

mp.mp.dps = 140


def reduced(s, z0, X, a, L, k, h, seed, rec):
    """--reduced=1: the system on span(V) to second order at z0, g(c) = B (c - a) + 1/2 Q(c, c), and its real roots.
    U^T F(z0) = -U^T J dz0 = -B a, as dz0 lies on span(V) (its share is printed); B = U^T J V = R^-T with
    J^-T V = U R, and Q_ijk = u_i . F''(x_j, x_k) at z0 from central second differences of the octo-double residual
    (the held offset is constant, so the bare kernel's residual serves). Q is formed at h and at h / 10, and must
    agree to 1e-2 (u_i . F'' keeps only about 1e-13 of F'', so the differences' truncation could swamp it). The roots
    are sought by Newton at 140 digits from c = 0 and from random starts of the size sqrt(|B a| / ||Q||). Returns
    whether the two Q agree."""
    zero = np.zeros(L)
    Um, Rm = mgs([mpv(o8(s.olu_solve(list(parts8(x)), True)), range(L)) for x in X])
    B = (mp.matrix(Rm) ** -1).T
    F0 = o8(s.residual_o(list(z0)))

    def at(c):
        p = [mp.fsum(mp.mpf(cj) * X[j][r] for j, cj in c.items()) for r in range(L)]
        return o8(s.residual_o(list(o8(od_axpy(1.0, *parts8(p), *z0)))))

    def comb(terms, scale):
        acc = [zero] * 8
        for c, F in terms:
            acc = o8(od_axpy(float(c), *F, *acc))
        return [v * scale for v in mpv(acc, range(L))]

    def Qof(hh):
        Fpp = {}
        for j in range(k):
            Fpp[(j, j)] = comb([(1, at({j: hh})), (-2, F0), (1, at({j: -hh}))], mp.mpf(1) / mp.mpf(hh) ** 2)
            for q in range(j + 1, k):
                Fpp[(j, q)] = Fpp[(q, j)] = comb([(1, at({j: hh, q: hh})), (-1, at({j: hh, q: -hh})),
                                                  (-1, at({j: -hh, q: hh})), (1, at({j: -hh, q: -hh}))],
                                                 mp.mpf(1) / (4 * mp.mpf(hh) ** 2))
        return [[[mp.fsum(u * f for u, f in zip(Um[i], Fpp[(j, q)])) for q in range(k)] for j in range(k)] for i in range(k)]

    t0 = time.time()
    Q, Q10 = Qof(h), Qof(h / 10)
    qmax = max(abs(Q[i][j][q]) for i in range(k) for j in range(k) for q in range(k))
    dq = max(abs(Q[i][j][q] - Q10[i][j][q]) for i in range(k) for j in range(k) for q in range(k)) / qmax
    ok = bool(dq < mp.mpf("1e-2"))
    am = mp.matrix(a)
    Ba = B * am
    nBa = mp.sqrt(sum(x ** 2 for x in Ba))
    scale = mp.sqrt(nBa / qmax)
    print(f"  reduced system: |B a| {mp.nstr(nBa, 3)}, max|Q| {mp.nstr(qmax, 3)}, Q at h against h/10 {mp.nstr(dq, 2)} "
          f"(bound 1e-2): {'pass' if ok else 'FAIL'} ({time.time() - t0:.0f} s); the root scale sqrt(|B a| / |Q|) "
          f"{mp.nstr(scale, 3)}", flush=True)

    def g(c):
        return [mp.fsum(B[i, j] * (c[j] - a[j]) for j in range(k))
                + mp.fsum(Q[i][j][q] * c[j] * c[q] for j in range(k) for q in range(k)) / 2 for i in range(k)]

    def jg(c):
        return mp.matrix([[B[i, j] + mp.fsum(Q[i][j][q] * c[q] for q in range(k)) for j in range(k)] for i in range(k)])

    rng = np.random.default_rng(seed)
    starts = [[mp.mpf(0)] * k] + [[mp.mpf(float(x)) * scale for x in rng.standard_normal(k)] for _ in range(24)]
    roots, best = [], mp.inf
    for c in starts:
        for _ in range(200):
            gv = g(c)
            ng = mp.sqrt(mp.fsum(x * x for x in gv))
            best = min(best, ng)
            if ng < mp.mpf("1e-30") * nBa:
                break
            try:
                dc = mp.lu_solve(jg(c), mp.matrix(gv))
            except ZeroDivisionError:
                break
            c = [c[j] - dc[j] for j in range(k)]
        if ng < mp.mpf("1e-30") * nBa and not any(max(abs(x - y) for x, y in zip(c, r)) < mp.mpf("1e-6") * scale for r in roots):
            roots.append(c)
    roots.sort(key=lambda r: mp.sqrt(mp.fsum(x * x for x in r)))
    for r in roots[:6]:
        print(f"  root |c| {mp.nstr(mp.sqrt(mp.fsum(x * x for x in r)), 3)}: " + ", ".join(mp.nstr(x, 3) for x in r), flush=True)
    print(f"  {len(roots)} real roots from {len(starts)} starts; the smallest |g| reached {mp.nstr(best / nBa, 3)} of |B a|",
          flush=True)
    rec.update(reduced=dict(Ba=float(nBa), Qmax=float(qmax), Q_h_h10=float(dq), ok=ok, scale=float(scale),
                            roots=[[float(x) for x in r] for r in roots], best=float(best / nBa)))
    return ok


def cubic(s, z0, X, a, L, k, hs, seed, rec):
    """--cubic=1: the third-order reduced system at z0, g(c) = B (c - a) + 1/6 T(c, c, c), with T_ijlm =
    u_i . F'''(x_j, x_l, x_m). Q = U^T F'' is left out: at h = 1e-20, 1e-22, 1e-24 it is c h^2 exactly, the
    truncation of the second differences (#499). The directional third derivative D3(v) = F'''(v, v, v) is the
    antisymmetric difference [F(z+2hv) - 2F(z+hv) + 2F(z-hv) - F(z-2hv)] / 2h^3, and the mixed terms come by
    polarization: T(x, x, y) = [D3(x+y) - D3(x-y) - 2 D3(y)] / 6 and T(x, y, w) = sum over e2, e3 = +-1 of
    e2 e3 D3(x + e2 y + e3 w) / 24. T is formed at each h of hs; the check is that the last two agree to 1e-2 (a T
    that falls as h^2 is the truncation again). Returns whether they agree."""
    zero = np.zeros(L)
    Um, Rm = mgs([mpv(o8(s.olu_solve(list(parts8(x)), True)), range(L)) for x in X])
    B = (mp.matrix(Rm) ** -1).T

    def at(c):
        p = [mp.fsum(mp.mpf(cj) * X[j][r] for j, cj in c.items()) for r in range(L)]
        return o8(s.residual_o(list(o8(od_axpy(1.0, *parts8(p), *z0)))))

    def d3(c, hh):
        """U^T F'''(v, v, v) for v = sum_j c_j x_j, by the antisymmetric four-point difference at step hh"""
        acc = [zero] * 8
        for w, t in ((1, 2), (-2, 1), (2, -1), (-1, -2)):
            F = at({j: t * hh * cj for j, cj in c.items()})
            acc = o8(od_axpy(float(w), *F, *acc))
        dv = [x / (2 * mp.mpf(hh) ** 3) for x in mpv(acc, range(L))]
        return [mp.fsum(u * f for u, f in zip(Um[i], dv)) for i in range(k)]

    def tensor(hh):
        T = {}
        D = {}

        def D3(key):
            if key not in D:
                D[key] = d3(dict(key), hh)
            return D[key]
        for j in range(k):
            T[(j, j, j)] = D3(((j, 1),))
        for j in range(k):
            for q in range(k):
                if q != j:
                    plus, minus, yy = D3(((j, 1), (q, 1))), D3(((j, 1), (q, -1))), D3(((q, 1),))
                    T[(j, j, q)] = [(p - m - 2 * y) / 6 for p, m, y in zip(plus, minus, yy)]
        for j in range(k):
            for q in range(j + 1, k):
                for w in range(q + 1, k):
                    acc = [mp.mpf(0)] * k
                    for e2 in (1, -1):
                        for e3 in (1, -1):
                            v = D3(((j, 1), (q, e2), (w, e3)))
                            acc = [x + e2 * e3 * y for x, y in zip(acc, v)]
                    T[(j, q, w)] = [x / 24 for x in acc]
        full = {}
        for j in range(k):
            for q in range(k):
                for w in range(k):
                    idx = tuple(sorted((j, q, w)))
                    jj, qq, ww = idx
                    if jj == qq == ww:
                        full[(j, q, w)] = T[(jj, jj, jj)]
                    elif jj == qq:
                        full[(j, q, w)] = T[(jj, jj, ww)]
                    elif qq == ww:
                        full[(j, q, w)] = T[(qq, qq, jj)]
                    else:
                        full[(j, q, w)] = T[idx]
        return full, len(D)

    t0 = time.time()
    Ts = [tensor(hh) for hh in hs]
    tmax = [max(abs(x) for v in T.values() for x in v) for T, _ in Ts]
    T, nres = Ts[-1]
    dt = max(abs(x - y) for key in T for x, y in zip(Ts[-1][0][key], Ts[-2][0][key])) / tmax[-1]
    ok = bool(dt < mp.mpf("1e-2"))
    am = mp.matrix(a)
    nBa = mp.sqrt(sum(x ** 2 for x in B * am))
    scale = mp.cbrt(6 * nBa / tmax[-1])
    print("  cubic term max|U^T F'''| by h: " + ", ".join(f"{hh:.0e}: {mp.nstr(t, 4)}" for hh, t in zip(hs, tmax))
          + f"; the last two agree to {mp.nstr(dt, 2)} (bound 1e-2): {'pass' if ok else 'FAIL'} "
          f"({4 * nres} residuals per h, {time.time() - t0:.0f} s)", flush=True)

    def g(c):
        return [mp.fsum(B[i, j] * (c[j] - a[j]) for j in range(k))
                + mp.fsum(T[(j, q, w)][i] * c[j] * c[q] * c[w] for j in range(k) for q in range(k) for w in range(k)) / 6
                for i in range(k)]

    def jg(c):
        return mp.matrix([[B[i, j] + mp.fsum(T[(j, q, w)][i] * c[q] * c[w] for q in range(k) for w in range(k)) / 2
                           for j in range(k)] for i in range(k)])

    rng = np.random.default_rng(seed)
    starts = [[mp.mpf(0)] * k, list(a)] + [[mp.mpf(float(x)) * scale for x in rng.standard_normal(k)] for _ in range(24)]
    roots, best = [], mp.inf
    for c in starts:
        ng = mp.inf
        for _ in range(200):
            gv = g(c)
            ng = mp.sqrt(mp.fsum(x * x for x in gv))
            best = min(best, ng)
            if ng < mp.mpf("1e-30") * nBa:
                break
            try:
                dc = mp.lu_solve(jg(c), mp.matrix(gv))
            except ZeroDivisionError:
                break
            c = [c[j] - dc[j] for j in range(k)]
        if ng < mp.mpf("1e-30") * nBa and not any(max(abs(x - y) for x, y in zip(c, r)) < mp.mpf("1e-6") * scale for r in roots):
            roots.append(c)
    roots.sort(key=lambda r: mp.sqrt(mp.fsum(x * x for x in r)))
    print(f"  cubic reduced system: |B a| {mp.nstr(nBa, 3)}, |a| {mp.nstr(mp.sqrt(mp.fsum(x * x for x in a)), 3)}, "
          f"root scale (6 |B a| / |T|)^(1/3) {mp.nstr(scale, 3)}", flush=True)
    for r in roots[:6]:
        print(f"  root |c| {mp.nstr(mp.sqrt(mp.fsum(x * x for x in r)), 3)}: " + ", ".join(mp.nstr(x, 3) for x in r), flush=True)
    print(f"  {len(roots)} real roots from {len(starts)} starts; the smallest |g| reached {mp.nstr(best / nBa, 3)} of |B a|",
          flush=True)
    # each root against the actual weak residual, U^T [F(z0 + V c) - F(z0)] - B a, with every order of F in it: small
    # against |B a| where the cubic model holds at that root (the held offset cancels in the difference)
    F0 = o8(s.residual_o(list(z0)))
    Ba = B * am
    actual = []
    for r in roots[:4]:
        acc = o8(od_axpy(-1.0, *F0, *at({j: r[j] for j in range(k)})))
        dv = mpv(acc, range(L))
        gv = [mp.fsum(u * f for u, f in zip(Um[i], dv)) - Ba[i] for i in range(k)]
        rel = mp.sqrt(mp.fsum(x * x for x in gv)) / nBa
        actual.append(float(rel))
        print(f"  root |c| {mp.nstr(mp.sqrt(mp.fsum(x * x for x in r)), 3)}: the actual weak residual there is "
              f"{mp.nstr(rel, 3)} of |B a|", flush=True)
    rec.update(cubic=dict(hs=list(hs), Tmax=[float(t) for t in tmax], agree=float(dt), ok=ok, Ba=float(nBa),
                          scale=float(scale), roots=[[float(x) for x in r] for r in roots], best=float(best / nBa),
                          actual=actual))
    return ok


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    if len(args) != 3:
        sys.exit(__doc__)
    raw, dzf, base = args
    k, its = int(opts.get("k", 4)), int(opts.get("its", 16))
    mutate = opts.get("mutate", "")
    if mutate not in ("", "D"):
        sys.exit(f"--mutate={mutate}: D")
    f = np.load(os.path.join(raw, dzf))
    z0 = o8([f[f"z{q}"] for q in range(8)])
    d0 = o8([f[f"d{q}"] for q in range(8)])
    md = json.loads(str(f["meta"]))
    _, mb = load_o(os.path.join(raw, base))
    M, nodes, kf = int(mb["M"]), list(mb["nodes"]), int(mb.get("kfreeze", 3))
    K, L = len(nodes), len(z0[0])
    # the attempt's D as anm_run.py forms it, the base's D (its string) plus the step a, exact in mpmath
    D = mp.mpf(mb["D_str"]) + mp.mpf(md["a"]) + (mp.mpf("1e-6") if mutate == "D" else 0)
    if abs(D - mp.mpf(md["D"])) > (mp.mpf("1e-5") if mutate == "D" else mp.mpf("1e-9")):
        sys.exit(f"{dzf}: the base's D plus a, {mp.nstr(D, 15)}, is not the file's D {md['D']}")
    # D in full: the horizon block moves by about 1e31 per unit of D, so 30 digits leave it 3e3 off
    s = make_solver(str(D), M, nodes=nodes, kt=mb.get("kt", 36), kc=mb.get("kc", 146), ks=mb.get("ks", 130),
                    threads=int(opts.get("threads", 2)), kfreeze=kf)
    s.set_segs(mb["segs"])
    zero = np.zeros(L)
    t0 = time.time()
    st = s.newton_o(list(z0))
    print(f"{dzf}: D {mp.nstr(D, 12)} (a {md.get('a')}), from {base}; L {L}, octo-double LU {time.time() - t0:.0f} s, "
          f"pivots {st[3]:.1e}..{st[4]:.1e}", flush=True)
    rec = dict(dz0=dzf, base=base, D=float(D), a=md.get("a"), k=k, mutate=mutate)

    # the check: J (dz0 - dz_bare) lives on the horizon block's p rows only
    t0 = time.time()
    dlt = mpv(o8(od_axpy(-1.0, *o8(st[0]), *d0)), range(L))
    J = [np.asarray(p) for p in s.jacobian_o(list(z0))]
    nz = [c for c in range(L) if dlt[c] != 0]
    jd = [mp.fsum(mp.fsum(mp.mpf(float(J[q][r, c])) for q in range(8) if J[q][r, c] != 0) * dlt[c] for c in nz)
          for r in range(L)]
    hp = set(range(4 * M * K, 4 * M * K + M))
    on = max(abs(jd[r]) for r in hp)
    off = max(abs(jd[r]) for r in range(L) if r not in hp)
    ok = bool(on > 0 and off < mp.mpf("1e-6") * on)
    print(f"  check: J (dz0 - dz_bare) on the held rows {mp.nstr(on, 3)}, elsewhere {mp.nstr(off, 3)} (bound 1e-6 of "
          f"the first, {time.time() - t0:.0f} s): {'pass' if ok else 'FAIL'}", flush=True)
    rec["check"] = dict(on=float(on), off=float(off), ok=ok)

    def solve(b, transpose=False):
        x = s.olu_solve([np.ascontiguousarray(b)] + [zero] * 7, transpose)
        return np.asarray(x[0]) + np.asarray(x[1])

    V, hist = subspace(solve, L, k, its)
    X = mgs([[mp.mpf(float(x)) for x in V[:, j]] for j in range(k)])[0]
    for _ in range(int(opts.get("refine", 3))):
        X = mgs([mpv(o8(s.olu_solve(list(o8(s.olu_solve(list(parts8(x)), True))))), range(L)) for x in X])[0]
    dv = mpv(d0, range(L))
    a = [mp.fsum(x * y for x, y in zip(X[j], dv)) for j in range(k)]
    part = [mp.fsum(a[j] * X[j][r] for j in range(k)) for r in range(L)]
    rest = [y - p for y, p in zip(dv, part)]
    n2 = lambda v: mp.sqrt(mp.fsum(x * x for x in v))  # noqa: E731
    nd, nv, nr = n2(dv), n2(part), n2(rest)
    print(f"  sigma (iteration) " + ", ".join(f"{x:.4e}" for x in hist[-1]), flush=True)
    print(f"  |dz0|: max {float(max(abs(x) for x in dv)):.4e}, 2-norm {mp.nstr(nd, 4)}; on span(V) {mp.nstr(nv, 4)} "
          f"(share {mp.nstr(nv / nd, 3)}), the rest {mp.nstr(nr, 4)} (share {mp.nstr(nr / nd, 3)})", flush=True)
    print("  coefficients on the refined basis: " + ", ".join(mp.nstr(x, 4) for x in a), flush=True)
    rf = np.array([float(x) for x in rest])
    comps = sorted(((amp, name, h) for name, o, par in layout(M, K) for h, amp in harmonics(rf[o:o + M], par).items()),
                   reverse=True)[:6]
    print("  the rest's largest components: " + ", ".join(f"{n} k{h} {v:.2e}" for v, n, h in comps)
          + f"; Delta {float(rest[-1]):.2e}", flush=True)
    rec.update(sigma=[float(x) for x in hist[-1]], dz_max=float(max(abs(x) for x in dv)), dz_2=float(nd),
               on_V=float(nv), rest=float(nr), coeffs=[float(x) for x in a], rest_top=[(n, h, v) for v, n, h in comps])
    if opts.get("reduced", "0") != "0":
        ok_q = reduced(s, z0, X, a, L, k, float(opts.get("h", 1e-20)), int(opts.get("seed", 3)), rec)
        ok = ok and ok_q
    if opts.get("cubic", "0") != "0":
        hs = [float(x) for x in opts.get("hs", "1e-22,1e-23,1e-24").split(",")]
        ok_t = cubic(s, z0, X, a, L, k, hs, int(opts.get("seed", 3)), rec)
        ok = ok and ok_t
    print("check:", "pass" if ok else "FAIL")
    if opts.get("json"):
        with open(opts["json"], "w") as fh:
            json.dump(rec, fh, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

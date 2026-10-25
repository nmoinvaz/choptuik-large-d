#!/usr/bin/env python3
"""Zeros of the test field's det M_psi(lambda) (hp/modecount_cx, notes/GHOST_ADMIXTURE.md) inside a box: where a
count (modecount_test.py) finds zeros, this says where they are.

    python3 modecount_test_zeros.py STATE.npz --re=LO,HI --im=LO,HI [--panel=0.1] [--gl=8] [--tol=0.785]
                                    [--plant=RE+IMj] [--threads=N] [--M=N] [--cache=PATH]
                                    [--out=hp/data/modecount_test_zeros.jsonl]
    python3 modecount_test_zeros.py --selftest

The method is that of Delves and Lyness (Math. Comp. 21 (1967) 543, doi:10.1090/S0025-5718-1967-0228165-4).
g = det M_psi prod (lambda - p), p the candidate poles (modecount_common.horizon_poles) with |Im p| < max |Im| of the
box + 3 omega, as in modecount_wind.count: g has the zeros of det M_psi and no poles near the box. ln g is evaluated on
the box's boundary, counterclockwise from its lower left corner lambda_0, at the Gauss-Legendre nodes of panels (at most
--panel long, --gl nodes each), its argument unwrapped from point to point. A panel is halved while a step of the
argument between neighbouring points exceeds --tol (default pi/4, count's), down to a length of 1e-4. Before that, g is
divided by exp p, p(lambda) the quadratic whose real part fits ln|g| at the first panels' nodes in least squares (as
count fits a trend on the real axis): exp p has no zeros, and the power sums below are unchanged (Gauss-Legendre
integrates p's terms exactly), but arg g's drift along the edges, hundreds of radians per unit length at D 4 (the
count's trend there, 215 lambda), mostly goes, and with it most of the panels. The change of
the unwrapped argument around the boundary is 2 pi N, N the number of zeros inside (the count, at this resolution).
The power sums of the zeros about the box's centre c follow by parts, ln g continuous along the path:
    s_k = sum_j (z_j - c)^k = (lambda_0 - c)^k N - k / (2 pi i) oint (lambda - c)^(k-1) ln g dlambda,  k = 1 .. N.
The zeros are the roots of the polynomial with these power sums (Newton's identities). Each is then polished by
Newton's method on g, lambda <- lambda - g / g', g' by the central difference at step 1e-7 (on g, which is smooth at
its zeros, not on ln g), until a step is below 1e-11 (converged; at most 12 iterations).

Pass: the argument resolved (every step <= --tol), and every polished zero converged, inside the box, distinct (1e-7)
and within 1e-3 of its estimate (the quadrature's accuracy). With --plant=z, which multiplies det M_psi by
(lambda - z)(lambda - conj z) as modecount_test.py does, also z among the zeros to 1e-8: the control.

--selftest: the same on a synthetic g = exp(80 lambda + 3 lambda^2) prod (lambda - z_j), three zeros inside the unit
box (one 0.02 from an edge) and one outside 0.02 from it: the three must be found to 1e-9 and the outside one not.
Controls: with one panel per edge, 3 nodes and no halving the check must fail, and so must it with every estimate
moved by 2e-3 (the polished zeros then lie 2e-3 from their estimates).

Evaluations of det M_psi are cached in hp/data/modecount_test_D<D>_M<M>_raw.jsonl (modecount_test.py's file; --cache
for another), shared between plain and planted runs.

Newton from a grid of starts, this script's first version, does not work here: (ln g)' has a smooth background of
about 1e2 from the zeros and poles away from the box, so Newton's steps stay near 1e-2 and walk off unless a start is
within about 1e-2 of a zero.
"""
import cmath
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def wrap(x):
    return math.remainder(x, 2 * math.pi)


def trend_fit(points, vals, c):
    """The quadratic p(lambda) = c0 + c1 mu + c2 mu^2, mu = lambda - c, c0 real, whose real part fits ln|g| at the
    points in least squares (as count's trend on the real axis, here in the plane): exp(-p) has no zeros, and taking it
    out of g leaves the zeros and the power sums unchanged but removes most of arg g's drift along the edges."""
    mu = np.array(points) - c
    A = np.stack([np.ones(len(mu)), mu.real, -mu.imag, (mu * mu).real, -(mu * mu).imag], axis=1)
    k = np.linalg.lstsq(A, np.array([v.real for v in vals]), rcond=None)[0]
    return complex(k[0], 0.0), complex(k[1], k[2]), complex(k[3], k[4])


def boundary(lng, box, panel, ng, tol, hmin=1e-4, trend=True):
    """ln g on the boundary of box = (r0, r1, i0, i1), counterclockwise from (r0, i0), at the Gauss-Legendre nodes of
    panels halved until each step of arg g is <= tol. With trend, g is replaced by g exp(-p), p the quadratic of
    trend_fit to ln|g| at the first panels' nodes. Returns dict(panels, points, L (ln g continuous along the path, at
    the points), N, max_step, w, t, corner, trend, lt (lambda -> ln g exp(-p), argument wrapped))."""
    r0, r1, i0, i1 = box
    c = complex(0.5 * (r0 + r1), 0.5 * (i0 + i1))
    corners = [complex(r0, i0), complex(r1, i0), complex(r1, i1), complex(r0, i1)]
    panels = []
    for e in range(4):
        a, b = corners[e], corners[(e + 1) % 4]
        n = max(1, math.ceil(abs(b - a) / panel - 1e-9))
        panels += [(a + (b - a) * i / n, a + (b - a) * (i + 1) / n) for i in range(n)]
    t, w = np.polynomial.legendre.leggauss(ng)
    coef = (0j, 0j, 0j)
    lt = lng
    first = True
    while True:
        seq = [(corners[0], None)]
        for pi, (a, b) in enumerate(panels):
            seq += [(0.5 * (a + b) + 0.5 * (b - a) * tj, pi) for tj in t]
        seq.append((corners[0], None))
        if first and trend:
            coef = trend_fit([lam for lam, _ in seq], [lng(lam) for lam, _ in seq], c)

            def lt(lam, coef=coef):
                mu = lam - c
                v = lng(lam) - (coef[0] + coef[1] * mu + coef[2] * mu * mu)
                return complex(v.real, wrap(v.imag))
        first = False
        vals = [lt(lam) for lam, _ in seq]
        steps = [wrap(vals[i + 1].imag - vals[i].imag) for i in range(len(seq) - 1)]
        bad = set()
        for i, s in enumerate(steps):
            if abs(s) > tol:
                for k in (i, i + 1):
                    p = seq[k][1]
                    if p is not None and abs(panels[p][1] - panels[p][0]) >= 2 * hmin:
                        bad.add(p)
        if not bad:
            break
        new = []
        for pi, (a, b) in enumerate(panels):
            new += [(a, 0.5 * (a + b)), (0.5 * (a + b), b)] if pi in bad else [(a, b)]
        panels = new
    L = [vals[0]]
    for i, s in enumerate(steps):
        L.append(L[-1] + complex(vals[i + 1].real - vals[i].real, s))
    N = int(round((L[-1].imag - L[0].imag) / (2 * math.pi)))
    return dict(panels=panels, points=[lam for lam, _ in seq], L=L, N=N, max_step=max(abs(s) for s in steps), w=w,
                t=t, corner=corners[0], trend=coef, lt=lt)


def zeros_from_boundary(B, box):
    """The power sums s_1 .. s_N of the zeros about the box's centre by parts, and the roots of their polynomial"""
    r0, r1, i0, i1 = box
    c = complex(0.5 * (r0 + r1), 0.5 * (i0 + i1))
    N, ng = B["N"], len(B["t"])
    if N <= 0:
        return [], []
    s = []
    for k in range(1, N + 1):
        acc = 0j
        for pi, (a, b) in enumerate(B["panels"]):
            lam = B["points"][1 + pi * ng: 1 + (pi + 1) * ng]
            L = B["L"][1 + pi * ng: 1 + (pi + 1) * ng]
            acc += 0.5 * (b - a) * sum(wj * (lj - c) ** (k - 1) * Lj for wj, lj, Lj in zip(B["w"], lam, L))
        s.append((B["corner"] - c) ** k * N - k * acc / (2j * math.pi))
    e = [1 + 0j]
    for k in range(1, N + 1):
        e.append(sum((-1) ** (i - 1) * e[k - i] * s[i - 1] for i in range(1, k + 1)) / k)
    roots = np.roots([(-1) ** k * e[k] for k in range(N + 1)])
    return [complex(z) + c for z in roots], s


def polish(lng, z, h=1e-7):
    """Newton on g from z, g' by the central difference at h (g itself is smooth at its zero, ln g is not): the step
    -g / g' = -2 h exp(L(z) - L(z + h)) / (1 - exp(L(z - h) - L(z + h))), L = ln g. Returns (z, status, iterations,
    last step)."""
    last = math.inf
    for it in range(1, 13):
        L0, Lp, Lm = lng(z), lng(z + h), lng(z - h)
        if L0.real == -math.inf:
            return z, "converged", it, 0.0
        step = -2 * h * cmath.exp(L0 - Lp) / (1 - cmath.exp(Lm - Lp))
        z = z + step
        last = abs(step)
        if last < 1e-11:
            return z, "converged", it, last
    return z, "iterations", 12, last


def find(lng, box, panel, ng, tol, log=print, plant=None, poles=(), perturb=0.0, trend=True):
    """The count, the estimates and the polished zeros in box; returns (record, fails). perturb: added to every
    estimate (the self-test's control)"""
    r0, r1, i0, i1 = box
    B = boundary(lng, box, panel, ng, tol, trend=trend)
    est, s = zeros_from_boundary(B, box)
    est = [z + perturb for z in est]
    tr = B["trend"]
    log(f"  boundary: {len(B['panels'])} panels of {ng} nodes, max step of arg g {B['max_step']:.3f} rad, "
        f"N = {B['N']}; trend taken out: p = {tr[0].real:.3f} + ({tr[1]:.3f}) mu + ({tr[2]:.3f}) mu^2")
    zs, fails = [], []
    if B["max_step"] > tol:
        fails.append(f"the argument is not resolved (a step of {B['max_step']:.3f} rad)")
    if B["N"] < 0:
        fails.append(f"N = {B['N']} < 0: a pole inside the box")
    for z0 in est:
        z, status, its, last = polish(B["lt"], z0)
        inside = r0 <= z.real <= r1 and i0 <= z.imag <= i1
        dp = min([abs(z - p) for p in poles] + [9.0])
        rec = dict(estimate=[z0.real, z0.imag], z=[z.real, z.imag], status=status, its=its, last_step=last,
                   moved=abs(z - z0), inside=inside, nearest_candidate_pole=dp)
        zs.append(rec)
        log(f"  zero {z.real:+.12f} {z.imag:+.12f}i: estimate moved by {abs(z - z0):.1e}, {status} in {its} "
            f"(last step {last:.1e}), inside {inside}, nearest candidate pole {dp:.3f}")
        if status != "converged":
            fails.append(f"Newton did not converge from {z0:.6f}")
        if not inside:
            fails.append(f"the zero {z:.8f} is outside the box")
        if abs(z - z0) > 1e-3:
            fails.append(f"the zero {z:.8f} is {abs(z - z0):.1e} from its estimate")
    for i in range(len(zs)):
        for j in range(i):
            if abs(complex(*zs[i]["z"]) - complex(*zs[j]["z"])) < 1e-7:
                fails.append(f"two estimates polish to the same zero {complex(*zs[i]['z']):.8f}")
    if plant is not None:
        hit = [q for q in zs if abs(complex(*q["z"]) - plant) < 1e-8]
        log(f"  control: the planted zero {plant} {'found' if hit else 'NOT found'}")
        if not hit:
            fails.append("the planted zero was not found")
    rec = dict(box=list(box), panel=panel, gl=ng, tol=tol, N=B["N"], max_step=B["max_step"],
               trend=[[v.real, v.imag] for v in B["trend"]],
               panels=len(B["panels"]), power_sums=[[v.real, v.imag] for v in s], zeros=zs)
    return rec, fails


def selftest():
    zin = [0.3 + 0.4j, 0.7 + 0.75j, 0.98 + 0.5j]
    zout = [0.5 + 1.02j]

    def lng(lam):
        if any(lam == z for z in zin + zout):
            return complex(-math.inf, 0.0)
        v = 80 * lam + 3 * lam * lam + sum(cmath.log(lam - z) for z in zin + zout)
        return complex(v.real, wrap(v.imag))

    box = (0.0, 1.0, 0.0, 1.0)
    rec, fails = find(lng, box, 0.1, 8, math.pi / 4)
    got = sorted((complex(*q["z"]) for q in rec["zeros"]), key=lambda z: (z.real, z.imag))
    err = max([min(abs(z - g) for g in got) for z in zin] + [0.0]) if got else 9.0
    ok = not fails and rec["N"] == 3 and len(got) == 3 and err < 1e-9
    print(f"selftest: N {rec['N']}, {len(got)} zeros, worst error {err:.1e}, fails {fails}: {'ok' if ok else 'WRONG'}")
    crec, cfails = find(lng, box, 1.0, 3, 1e9, log=lambda m: None, trend=False)
    cgot = [complex(*q["z"]) for q in crec["zeros"]]
    cerr = max([min([abs(z - g) for g in cgot] + [9.0]) for z in zin])
    print(f"control (one 3-node panel per edge, no halving): N {crec['N']}, worst error {cerr:.1e}, fails {cfails}: "
          f"{'fails as it must' if cfails else 'NOT CAUGHT'}")
    prec, pfails = find(lng, box, 0.1, 8, math.pi / 4, log=lambda m: None, perturb=2e-3)
    print(f"control (the estimates moved by 2e-3): fails {pfails}: {'fails as it must' if pfails else 'NOT CAUGHT'}")
    return 0 if ok and cfails and pfails else 1


def main():
    if "--selftest" in sys.argv[1:]:
        sys.exit(selftest())
    from modecount_common import setup, horizon_poles
    from modecount_wind import parse
    args, opts = parse(sys.argv[1:])
    st = setup(args[0], M=int(opts["M"]) if "M" in opts else None, threads=int(opts.get("threads", 1)), log=print)
    r0, r1 = (float(v) for v in opts["re"].split(","))
    i0, i1 = (float(v) for v in opts["im"].split(","))
    box = (r0, r1, i0, i1)
    panel, ng, tol = float(opts.get("panel", 0.1)), int(opts.get("gl", 8)), float(opts.get("tol", math.pi / 4))
    plant = complex(opts["plant"].replace(" ", "")) if "plant" in opts else None
    out = opts.get("out", os.path.join(HERE, "data", "modecount_test_zeros.jsonl"))
    ymax = max(abs(i0), abs(i1))
    Q = [p for _, p in horizon_poles(st) if abs(p.imag) < ymax + 3 * st.omega]
    near = [p for p in Q if r0 - st.omega < p.real < r1 + st.omega and i0 - st.omega < p.imag < i1 + st.omega]
    cache_path = opts.get("cache", os.path.join(HERE, "data", f"modecount_test_D{st.D:.4f}_M{st.M}_raw.jsonl"))
    raw = {}
    if os.path.exists(cache_path):
        for line in open(cache_path):
            r = json.loads(line)
            raw[(r["re"], r["im"])] = r
    rf = open(cache_path, "a")
    nev = [0, 0.0]
    t0 = time.time()

    def lng(lam):
        key = (float(lam.real), float(lam.imag))
        if key not in raw:
            t = time.time()
            la, arg, pmin, pmax = st.c.test_det(st.zh, st.zl, key[0], key[1])
            r = dict(re=key[0], im=key[1], lnabs=la, arg=arg, pmin=pmin, pmax=pmax, sec=round(time.time() - t, 2))
            raw[key] = r
            rf.write(json.dumps(r) + "\n")
            rf.flush()
            nev[0] += 1
            nev[1] += time.time() - t
            if nev[0] % 50 == 0:
                print(f"    {nev[0]} evaluations, {nev[1] / nev[0]:.1f} s each", flush=True)
        r = raw[key]
        if plant is not None and lam in (plant, plant.conjugate()):
            return complex(-math.inf, 0.0)     # exactly on the planted zero (the polish can land there)
        v = complex(r["lnabs"], r["arg"]) + sum(cmath.log(lam - p) for p in Q)
        if plant is not None:
            v += cmath.log(lam - plant) + cmath.log(lam - plant.conjugate())
        return complex(v.real, wrap(v.imag))

    print(f"D {st.D}: M {st.M}, |F| {st.F:.1e}, omega {st.omega:.6f}; box Re {r0} .. {r1}, Im {i0} .. {i1}; panels "
          f"<= {panel}, {ng} nodes, tol {tol:.3f}; plant {plant}; {len(Q)} candidate poles multiplied out, within "
          f"omega of the box: {[(round(p.real, 4), round(p.imag, 4)) for p in near]}", flush=True)
    rec, fails = find(lng, box, panel, ng, tol, plant=plant, poles=Q, log=lambda m: print(m, flush=True))
    rec.update(state=os.path.basename(args[0]), D=st.D, M=st.M, F=st.F, omega=st.omega,
               plant=None if plant is None else [plant.real, plant.imag], evaluations=nev[0],
               sec=round(time.time() - t0, 1), fails=fails, time=time.strftime("%Y-%m-%d %H:%M:%S"),
               poles_near=[[p.real, p.imag] for p in near])
    with open(out, "a") as fo:
        fo.write(json.dumps(rec) + "\n")
    print(f"{nev[0]} evaluations, {time.time() - t0:.0f} s; " + ("PASS" if not fails else "FAIL: " + "; ".join(fails)),
          flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

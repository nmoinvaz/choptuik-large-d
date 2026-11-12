#!/usr/bin/env python3
"""Issue #545: march the perturbed solution of notes/QUAD_DOUBLE.md 13.6 outward from its horizon series, past the
series radius R, and find where it breaks down, beside the zeros of x - c (where the p equation is singular) and of
B = 1/a^2 (an apparent horizon).

At a stored state the horizon datum a_h is scaled by 1 + delta (13.6's perturbation, hp/pole_test.py), the kernel's
horizon series in u = 1 - x is recomputed, and summed at u = R/2 in double-double (mpmath per lane, R from 13.6's
estimator, pole_test.ratio_radius). From there the state is Taylor-integrated towards the centre (u increasing) by
Solver.integrate, the kernel's own double-double Taylor steps of order kt with the tau coupling, with step control: a
step h is kept when one step and two half steps agree to TOL relative, else halved; after a kept step with error
below TOL / 1e4 the next is 1.5 times longer. The march breaks down where h falls below HMIN times u (the local radius
of the Taylor series goes to zero there) or a field passes BIG. Along the march, per point, min over lanes of x - c
and of B (max of x - c: it is negative and the first lane to reach 0 marks the zero) are recorded; their zeros are located by linear interpolation where they change sign, or extrapolated
linearly from the last two points where they do not before the breakdown.

Conditioning: the horizon's modes grow like u^kappa away from it, kappa of order n, and the centre's like
x^-(n-1) towards it, so the march amplifies its start's rounding. A shadow march, the start moved by one
double-double rounding and stepped on the same grid, gives u_trust, where the two first differ by more than TRUST
relative; zeros and breakdowns are read only below u_trust, and a march trusted only below R is "not resolved in
double-double".

Verdict per (state, delta), fixed before the run (#545): the breakdown u_b lies within 3% of the zero of x - c and
more than 10% from the zero of B, or the record says where it lies instead.

Control (delta = 0): the march from u1/2 to u1, u1 = min(u0, R/2) (inside the series' radius and the last node),
must reproduce the series summed at u1 to 1e-20 relative (it checks the layout of the state and the direction of u,
and that the horizon series solves the kernel's equations to double-double). The
solution itself (delta = 0) is marched too, for reference.

Planted (`planted`, run first): the same stepper and breakdown rule on y(u) = (u_p - u)^(-alpha), alpha = 1/2 (a
regular singular point at u_p), stepped by its own Taylor series truncated at the kernel's order, in f64, so with
the step test at 1e-12 instead of TOL (the arithmetic's level, as TOL is double-double's). u_b must lie within 1% of
u_p, and --mutate=move (the check against 1.05 u_p) must fail the 3% test.

The arithmetic is double-double (Solver.integrate has no quad-double path); delta >= 1e-8 sits 8 orders above the
start's rounding (1e-16 relative where the series' high parts dominate).

    cd hp && python3 pole_march.py planted [--mutate=move]
    cd hp && python3 pole_march.py OUT.jsonl STATE.npz [STATE.npz ...] [--deltas=1e-8,1e-6,1e-4] [--threads N]
"""
import json
import math
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

TOL = 1e-22          # relative agreement of one step and two half steps
HMIN = 1e-7          # breakdown: the kept step below HMIN u
BIG = 1e100          # breakdown: a field above this (far above the mass mode's growth, so a step collapse shows first)
H0 = 1e-3            # first step, as a fraction of R
TRUST = 1e-6         # the march is trusted while one rounding of its start moves it by at most this, relative
U_END = 0.995        # the march ends here, short of the centre x = 0 (a singular point of the equations)
FIELDS = ("p", "m", "a", "c")


def diff_rel(y1, y2, parts):
    """max |y1 - y2| / max |y2| of two states in parts (the vector holds the parts one after the other): the parts'
    differences are added, so two double-double values that agree to 1e-32 give 1e-32, not an f64 ulp"""
    n = len(y1) // parts
    d = sum(y1[k * n:(k + 1) * n] - y2[k * n:(k + 1) * n] for k in range(parts))
    v = sum(y2[k * n:(k + 1) * n] for k in range(parts))
    return float(np.max(np.abs(d))) / max(float(np.max(np.abs(v))), 1e-300)


def march(step, y0, u0, u_end, scale, info, h0, tol=TOL, parts=1):
    """the adaptive march from u0 towards u_end: step(y, ua, ub) -> y at ub from y at ua (the half steps end at the
    full step's own endpoint, so the grid's rounding does not enter the step test); info(y, u) -> (max x - c, min B,
    max|y|).
    Returns (points [(u, xc, B, ymax, h)], u_b or None, why, the last state)"""
    y, u, h = y0, u0, h0
    pts = [(u,) + info(y, u) + (h,)]
    while u < u_end:
        h = min(h, u_end - u)
        ub = u_end if h >= u_end - u else u + h
        um = u + 0.5 * (ub - u)
        y1 = step(y, u, ub)
        y2 = step(step(y, u, um), um, ub)
        with np.errstate(invalid="ignore", over="ignore"):
            err = diff_rel(y1, y2, parts)
        if not np.all(np.isfinite(y2)) or err > tol:
            h /= 2
            if h < HMIN * max(u, scale):
                return pts, u, f"step below {HMIN:g} u (error {err:.1e})", y
            continue
        y, u = y2, ub
        xc, B, ym = info(y, u)
        pts.append((u, xc, B, ym, h))
        if ym > BIG:
            return pts, u, f"a field above {BIG:g}", y
        if err < tol * 1e-4:
            h *= 1.5
    return pts, None, "reached the end", y


def shadow(step, y0, pts, rng, parts=2):
    """the march's own conditioning: the start moved by one double-double rounding (each low part by a uniform
    relative 2^-106 of its high part) and both stepped along the march's grid by its two half steps (so the unmoved
    replay is the march itself). Returns the relative divergence at each grid point."""
    n = len(y0) // parts
    y1 = np.array(y0, dtype=float, copy=True)
    y1[n:2 * n] += y0[:n] * rng.uniform(-1, 1, n) * 2.0 ** -106
    ya, yb, out = np.array(y0, dtype=float, copy=True), y1, [0.0]
    for a, b in zip(pts[:-1], pts[1:]):
        um = a[0] + 0.5 * (b[0] - a[0])
        ya = step(step(ya, a[0], um), um, b[0])
        yb = step(step(yb, a[0], um), um, b[0])
        with np.errstate(invalid="ignore", over="ignore"):
            out.append(diff_rel(yb, ya, parts))
    return out


def trusted(pts, div, limit):
    """the first u where the shadow's divergence exceeds limit (None if it never does)"""
    for p_, d in zip(pts, div):
        if not (d <= limit):
            return p_[0]
    return None


def near_sonic(pts):
    """the closest approach to x = c along the march: the largest max-over-lanes x - c at an interior local maximum
    (after the march has moved away from its start), with the field peak and the smallest step near it"""
    xs = [p_[1] for p_ in pts]
    best = None
    for i in range(1, len(pts) - 1):
        if xs[i] >= xs[i - 1] and xs[i] >= xs[i + 1] and (best is None or xs[i] > xs[best]):
            best = i
    if best is None:
        return None
    lo, hi = max(0, best - 10), min(len(pts), best + 11)
    iy = max(range(lo, hi), key=lambda i: pts[i][3])
    ih = min(range(lo, hi), key=lambda i: pts[i][4])
    return dict(u=pts[best][0], xc=xs[best], u_field_peak=pts[iy][0], field_peak=pts[iy][3], u_hmin=pts[ih][0],
                hmin=pts[ih][4])


def zero_of(pts, k):
    """the first zero of column k (1: x - c, 2: B) along the march: interpolated where it changes sign, else
    extrapolated linearly from the last two points (None if it moves away from zero)"""
    for a, b in zip(pts[:-1], pts[1:]):
        if a[k] * b[k] <= 0 and a[k] != b[k]:
            return a[0] + (b[0] - a[0]) * a[k] / (a[k] - b[k]), "crossed"
    if len(pts) < 2:
        return None, "too few points"
    a, b = pts[-2], pts[-1]
    if b[k] == a[k] or (b[0] - a[0]) == 0:
        return None, "flat"
    slope = (b[k] - a[k]) / (b[0] - a[0])
    uz = b[0] - b[k] / slope
    return (uz, "extrapolated") if uz > b[0] else (None, "moving away")


def verdict(ub, us, uB):
    """#545's rule: u_b within 3% of the zero of x - c and more than 10% from the zero of B"""
    if ub is None:
        return "no breakdown"
    near_s = us is not None and abs(ub - us) <= 0.03 * us
    far_B = uB is None or abs(ub - uB) > 0.10 * uB
    if near_s and far_B:
        return "sonic point (x = c)"
    return f"neither: u_b {ub:.4g}, x - c zero {us}, B zero {uB}"


# ------------------------------------------------------------------ planted truth
def planted(mutate=""):
    """y = (u_p - u)^(-1/2) stepped by its own Taylor series of order 36, through march(); x - c := u_p - u (zero at
    u_p) and B := 1 + u (no zero), so the verdict must name the sonic point"""
    from math import lgamma
    alpha, kt = 0.5, 36
    ok = True
    for up in (0.05, 0.3, 0.7):
        def step(y, u, ub):
            d = up - u
            # (d - t)^-alpha = d^-alpha sum_k (alpha)_k / k! (t / d)^k, truncated at kt
            r = (ub - u) / d
            if r >= 1:
                return np.array([np.inf])
            s = sum(math.exp(lgamma(alpha + k) - lgamma(alpha) - lgamma(k + 1)) * r ** k for k in range(kt + 1))
            return np.array([y[0] * s])
        info = lambda y, u: (up - u, 1 + u, float(abs(y[0])))
        u0 = 0.4 * up
        pts, ub, why, _ = march(step, np.array([(up - u0) ** -alpha]), u0, 2 * up, up, info, H0 * up, tol=1e-12)
        us, _ = zero_of(pts, 1)
        target = 1.05 * up if mutate == "move" else up
        e = abs(ub - target) / target if ub is not None else float("inf")
        good = e <= 0.01 and abs(ub - target) <= 0.03 * target
        print(f"  planted u_p {up}: breakdown u_b {ub} ({why}), {len(pts)} points; |u_b - {target:.4g}| / {target:.4g} "
              f"= {e:.2e}; x - c zero {us}; verdict {verdict(ub, target, None)}: {'pass' if good else 'FAIL'}",
              flush=True)
        ok = ok and good
    return ok


# ------------------------------------------------------------------ the states
def run(out, paths, deltas, threads):
    import mpmath as mp
    from hpkernel.driver import load, make_solver
    from pole_test import ratio_radius
    mp.mp.dps = 40
    status = 0
    with open(out, "a") as fo:
        for path in paths:
            zh, zl, meta = load(path)
            D, M, nodes = meta["D"], meta["M"], list(meta["nodes"])
            K = len(nodes)
            u_last = 1.0 - nodes[-1]
            kf = meta.get("kfreeze", 13 if D >= 35 else None)
            s = make_solver(D, M, nodes=nodes, kt=meta.get("kt", 36), kc=meta.get("kc", 146), ks=meta.get("ks", 130),
                            threads=threads, kfreeze=kf)
            s.set_segs(meta["segs"] if "segs" in meta else s.plan(zh, zl, 1e-32, 0.05))
            sig = s.series_scale(1)
            oa = 3 * M + 4 * M * K
            rng = np.random.default_rng(545)
            for delta in (0.0,) + tuple(deltas):
                t0 = time.time()
                z2h, z2l = zh.copy(), zl.copy()
                z2h[oa:oa + M] *= 1.0 + delta
                z2l[oa:oa + M] *= 1.0 + delta
                hh, hl = s.series(z2h, z2l, 1, True)
                hh, hl = np.asarray(hh).reshape(-1, 4, M), np.asarray(hl).reshape(-1, 4, M)
                R, _ = ratio_radius(hh + hl, u_last, sig)

                def summed(u):
                    """the series at u in double-double, per lane (mpmath), as (sh, sl) of 4M entries"""
                    t = mp.mpf(u) / mp.mpf(sig)
                    sh, sl = np.zeros(4 * M), np.zeros(4 * M)
                    for q in range(4):
                        for j in range(M):
                            v = mp.mpf(0)
                            for k in range(hh.shape[0] - 1, -1, -1):
                                v = v * t + (mp.mpf(float(hh[k, q, j])) + mp.mpf(float(hl[k, q, j])))
                            sh[q * M + j] = float(v)
                            sl[q * M + j] = float(v - mp.mpf(sh[q * M + j]))
                    return sh, sl

                def step(y, u, ub):
                    """one Taylor path of the kernel from x = 1 - u to 1 - ub; y is the 8M vector (hi, lo)"""
                    oh, ol = s.integrate(z2h, z2l, np.ascontiguousarray(y[:4 * M]), np.ascontiguousarray(y[4 * M:]),
                                         [1.0 - u, 1.0 - ub])
                    return np.concatenate([np.asarray(oh), np.asarray(ol)])

                def info(y, u):
                    v = y[:4 * M] + y[4 * M:]
                    a, c = v[2 * M:3 * M], v[3 * M:]
                    # x - c is negative on the march and the first lane to reach 0 is the zero: the max over lanes
                    return float(np.max((1.0 - u) - c)), float(np.min(1.0 / a ** 2)), float(np.max(np.abs(v)))

                u_s = R / 2
                y0 = np.concatenate(summed(u_s))
                rec = dict(state=os.path.basename(path), D=D, M=M, delta=delta, R=R, u_start=u_s, u_last=u_last)
                if delta == 0.0:
                    # the control: from u1 / 2 to u1, u1 = min(u0, R / 2), inside both R and the last node; it passes
                    # when the march meets the series to 1e-20, or to 10 times its own shadow's divergence there
                    u1 = min(u_last, R / 2)
                    uc = u1 / 2
                    yc = np.concatenate(summed(uc))
                    pts, ub, why, yend = march(step, yc, uc, u1, R, info, H0 * R, parts=2)
                    rel = diff_rel(yend, np.concatenate(summed(u1)), 2)
                    sd = shadow(step, yc, pts, rng)[-1]
                    ok = bool(ub is None and rel <= max(1e-20, 10 * sd))
                    rec.update(control=dict(u_from=uc, u_to=u1, rel=rel, shadow=sd, points=len(pts), breakdown=ub, why=why,
                                            ok=ok))
                    print(f"D {D:.4f} delta 0: R {R:.4f}; the march from {uc:.4f} to {u1:.4f} against the series there: "
                          f"{rel:.1e} relative, its shadow {sd:.1e} ({len(pts)} points): {'pass' if ok else 'FAIL'}",
                          flush=True)
                    status |= 0 if ok else 1
                if True:
                    # every delta, 0 included (the solution itself, for reference), from R/2 towards the centre
                    pts, ub, why, _ = march(step, y0, u_s, U_END, R, info, H0 * R, parts=2)
                    div = shadow(step, y0, pts, rng)
                    ut = trusted(pts, div, TRUST)
                    tp = [p_ for p_ in pts if ut is None or p_[0] < ut]
                    us, hows = zero_of(tp, 1)
                    uB, howB = zero_of(tp, 2)
                    v = verdict(ub if ut is None or (ub is not None and ub < ut) else None, us, uB)
                    ns = near_sonic(tp)
                    if ut is not None and ut < R:
                        v = f"not resolved in double-double: the march is trusted only to u {ut:.4g} < R {R:.4g}"
                    rec.update(breakdown=ub, why=why, points=len(pts), u_trust=ut, near_sonic=ns, u_sonic=us, how_sonic=hows, u_AH=uB,
                               how_AH=howB,
                               last=dict(zip(("u", "xc", "B", "ymax", "h"), pts[-1])), verdict=v,
                               march=[list(p_) for p_ in pts[:: max(1, len(pts) // 60)]])
                    nstext = "none" if not ns else (f"u {ns['u']:.4f} (x - c {ns['xc']:+.4f}, {ns['u'] / R - 1:+.1%} from R), "
                                                    f"field peak {ns['field_peak']:.3g} at u {ns['u_field_peak']:.4f}, "
                                                    f"smallest step {ns['hmin']:.1e} at u {ns['u_hmin']:.4f}")
                    print(f"D {D:.4f} delta {delta:g}: R {R:.4f}, start {u_s:.4f}; trusted to u {ut}; breakdown u_b {ub} ({why}), "
                          f"{len(pts)} points; x - c zero {us} ({hows}); B zero {uB} ({howB}); last max x - c "
                          f"{pts[-1][1]:+.3e}, min B {pts[-1][2]:.3e}; closest to x = c: {nstext}"
                          f"; verdict {v} ({time.time() - t0:.0f} s)",
                          flush=True)
                fo.write(json.dumps(rec) + "\n")
                fo.flush()
    return status


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    if args and args[0] == "planted":
        ok = planted(opts.get("mutate", ""))
        print("planted:", "pass" if ok else "FAIL")
        return 0 if ok else 1
    if len(args) < 2:
        sys.exit(__doc__)
    if not planted():
        print("planted: FAIL, the states are not read")
        return 1
    deltas = tuple(float(x) for x in opts.get("deltas", "1e-8,1e-6,1e-4").split(","))
    return run(args[0], args[1:], deltas, int(opts.get("threads", 1)))


if __name__ == "__main__":
    sys.exit(main())

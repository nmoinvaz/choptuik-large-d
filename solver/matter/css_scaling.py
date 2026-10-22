#!/usr/bin/env python3
"""How the CSS solution scales with D: the finite-D solutions, continued past the self-similarity horizon, against the
D = infinity inner solution at fixed xi = x/sqrt(n) (matter/CSS.md, section 8).

    python3 css_scaling.py BG.jsonl KAPPA F0_INF OMEGA_INF [D1,D2,...]

The finite-D solution is analytic at x_h, so it continues uniquely along the real axis beyond x_h (integrated along
the upper half circle around x_h, then the real axis; the lower circle gives the same values to |J|).  The inner
solution is the lateral solution of css_inner.py on the real xi axis (its jump vanishes at (F0_INF, OMEGA_INF)),
integrated along real xi.  Printed: b and |f| (and arg f) at xi = 0.1 ... 0.8 for each D and for D = infinity.
"""
import sys, os, json
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from scipy.integrate import solve_ivp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import css_lateral as cl
import css_bg as cb
import css_inner as ci

XIS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]


def finite_D(n, kap, om, f0, xh, xis, Rf=0.3, x0=1e-4):
    cl._NEV[0] = 0
    st = cb.centre_start(n, kap, om, f0, x0)
    f = st[0] + 1j * st[1]; q = st[2] + 1j * st[3]; b = st[4]
    y = np.array([f, np.conj(f), q, np.conj(q), b + 0j])
    R = Rf * xh
    out = {}
    xs = [np.sqrt(n) * xi for xi in xis]
    inside = [x for x in xs if x < xh - R]
    s = solve_ivp(lambda t, z: cl.rhs_c(t, z, n, kap, om), (x0, xh - R), y, method='DOP853', rtol=1e-12, atol=1e-14,
                  dense_output=True)
    for x in inside:
        out[x] = s.sol(x)
    y = s.y[:, -1]
    def arc(phi, z):
        xx = xh + R * np.exp(1j * (np.pi - phi))
        return cl.rhs_c(xx, z, n, kap, om) * (-1j * R * np.exp(1j * (np.pi - phi)))
    cl._NEV[0] = 0
    y = solve_ivp(arc, (0, np.pi), y, method='DOP853', rtol=1e-12, atol=1e-14).y[:, -1]
    outside = [x for x in xs if x >= xh - R]
    if outside:
        x1 = xh + R
        cl._NEV[0] = 0
        s = solve_ivp(lambda t, z: cl.rhs_c(t, z, n, kap, om), (x1, max(max(outside), x1 + 1e-9)), y, method='DOP853',
                      rtol=1e-12, atol=1e-14, dense_output=True)
        for x in outside:
            if x >= x1:
                out[x] = s.sol(x)
            else:
                out[x] = None       # inside the circle: not evaluated
    return [out[x] for x in xs]


def inner(kap, f0, om, xis, r0=0.1, r1=0.25):
    J, Z, err = ci.jump(kap, f0, om, r0, r1)
    # Z: upper lateral solution at xi = r1 (real); integrate along real xi both ways
    res = {}
    lo = [x for x in xis if x < r1]; hi = [x for x in xis if x >= r1]
    if hi:
        s = solve_ivp(lambda t, z: ci.rhs(t, z, kap, om), (r1, max(hi)), Z, method='DOP853', rtol=1e-12, atol=1e-14,
                      dense_output=True)
        for x in hi:
            res[x] = s.sol(x)
    if lo:
        s = solve_ivp(lambda t, z: ci.rhs(t, z, kap, om), (r1, min(lo)), Z, method='DOP853', rtol=1e-12, atol=1e-14,
                      dense_output=True)
        for x in lo:
            res[x] = s.sol(x)
    return [res[x] for x in xis], abs(J)


if __name__ == "__main__":
    bg = {r["D"]: r for r in (json.loads(l) for l in open(sys.argv[1]))}
    kap, f0i, omi = map(float, sys.argv[2:5])
    Dl = [float(s) for s in sys.argv[5].split(",")] if len(sys.argv) > 5 else [D for D in (6, 10, 16, 24, 40, 60, 80) if D in bg]
    rows = {}
    for D in Dl:
        r = bg[D]
        vals = finite_D(D - 2, kap, r["omega"], r["f0"], r["X"], XIS)
        rows[D] = vals
    vi, aJ = inner(kap, f0i, omi, XIS)
    print("inner solution: |J| = %.1e at (|f(0)|, omega) = (%.10f, %.10f)" % (aJ, f0i, omi))
    print("b(x = sqrt(n) xi):")
    print("   xi  " + " ".join("  D=%-6g" % D for D in Dl) + "   D=inf")
    for k, xi in enumerate(XIS):
        print("  %.2f " % xi + " ".join(" %8.5f" % (rows[D][k][4].real if rows[D][k] is not None else np.nan) for D in Dl)
              + "  %8.5f" % vi[k][4].real)
    print("|f(x = sqrt(n) xi)|:")
    for k, xi in enumerate(XIS):
        print("  %.2f " % xi + " ".join(" %8.5f" % (abs(rows[D][k][0]) if rows[D][k] is not None else np.nan) for D in Dl)
              + "  %8.5f" % abs(vi[k][0]))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "css_scaling_k%g.json" % kap)
    json.dump(dict(xi=XIS, D=Dl,
                   b={str(D): [None if v is None else v[4].real for v in rows[D]] for D in Dl},
                   absf={str(D): [None if v is None else abs(v[0]) for v in rows[D]] for D in Dl},
                   b_inf=[v[4].real for v in vi], absf_inf=[abs(v[0]) for v in vi]), open(out, "w"), indent=1)
    print("written", out)

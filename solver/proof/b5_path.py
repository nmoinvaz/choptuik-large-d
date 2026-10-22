#!/usr/bin/env python3
"""Lemma B5(1) (proof/B5_inner.md, section 6): end-to-end numerical check, NOT rigorous. For a moderate n the exact
M = 1 model (b5_system.Model.f_eps, with the b-mode deviation vt slaved to first order, vt = eps(-E0 + z[QQ]/2 -
[P Phi]), so that the stiff b-mode is not integrated) is integrated from the upper end of the inner arc,
u0 = r_n e^{i(pi/2 + kappa)}, along the ray arg z = pi/2 + kappa out to |z| = Zt and then along the arc |z| = Zt down to
the Stokes line arg z = 0 (in the variable U = 1/z, where the fast mode has a constant rate). The data at u0 are the
inner solution plus the leading O(1/n) correction (b5_data.py: P^(1) = 0, Phi^(1) = phi_1 u0, C^(1) = -[F^2]). The
inner solution Z_in along the path is the formal series (accurate to e^{-1/(2|z|)}; at the M = 1 zero the multiplier
vanishes, so the series is also the lateral sum on the Stokes side to that accuracy).

Reported: max over the path of |W_P|/eps, |W_Phi|/(eps |z|), |W_C|/eps, compared with the certified bound x of
b5_bounds.py (kappa = 0.3, Z = 0.0085) for these data (forcing + data), and with eps Z^(1) (W = eps Z^(1) + O(eps^2)).

    python3 b5_path.py [--n=1e8] [--Zt=0.008]
    python3 b5_path.py --mutate=stokes     start on the Stokes line (u0 = r_n, then along arg z = 0 out to Zt): a free
                                           fast mode of relative size 1e-12 at u0 grows like e^{(1/r_n - 1/z)/2} and the
                                           bound must be violated
"""
import json
import os
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b5_system as BS  # noqa: E402
import b5_numeric as BN  # noqa: E402

ODD, EVEN = BS.ODD, BS.EVEN
OPTS = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
MUT = OPTS.get("mutate")
HERE = os.path.dirname(os.path.abspath(__file__))


class Shim:
    @staticmethod
    def matrix(A):
        return np.array(A, dtype=complex)

    @staticmethod
    def lu_solve(A, b):
        return np.linalg.solve(A, b)


def run(n=1e8, Zt=0.008, kappa=0.3, gamma=0.3):
    t0 = time.time()
    eps = 1.0 / n
    w = BN.omega()
    mdl = BS.Model(w, eps, Shim())
    inner = BN.Inner()
    rn = n ** (-1 + 2 * gamma)
    az = np.pi / 2 + kappa
    # leading O(1/n) data (b5_data.py)
    phi1 = {1: 74.9299 + 15.7144j, -1: 74.9299 - 15.7144j}
    A = float(BN.A_STAR)
    F2 = {0: A * A / 2, 2: -A * A / 4, -2: -A * A / 4}

    def slaved_vt(z, P, Phi, C):
        E0 = {k: v / 2 for k, v in BS.proj(P, P, EVEN).items()}
        Th = dict(E0)
        Q = mdl.Q_of(z, P, Phi, C, Th)
        QQ = BS.proj(Q, Q, EVEN)
        PPh = BS.proj(P, Phi, EVEN)
        return {k: eps * (-E0[k] + z * QQ[k] / 2 - PPh[k]) for k in EVEN}

    def pack(P, Phi, C):
        v = [P[m] for m in ODD] + [Phi[m] for m in ODD] + [C[k] for k in EVEN]
        return np.concatenate([np.real(v), np.imag(v)])

    def unpack(y):
        c = y[:7] + 1j * y[7:]
        return dict(zip(ODD, c[:2])), dict(zip(ODD, c[2:4])), dict(zip(EVEN, c[4:]))

    def rates(z, y):
        P, Phi, C = unpack(y)
        vt = slaved_vt(z, P, Phi, C)
        dP, dPhi, dC, _ = mdl.f_eps(z, P, Phi, C, vt)
        return dP, dPhi, dC

    def rhs_factory(zfun, dzfun):
        def f(s, y):
            z = zfun(s)
            dP, dPhi, dC = rates(z, y)
            dz = dzfun(s)
            return pack({m: dP[m] * dz for m in ODD}, {m: dPhi[m] * dz for m in ODD}, {k: dC[k] * dz for k in EVEN})
        return f

    def inner_state(z):
        Y, q = BN.state_in(inner, complex(z), w)
        return q["P"], q["Phi"], q["C"]

    # start point and data
    if MUT == "stokes":
        u0 = rn + 0j
    else:
        u0 = rn * np.exp(1j * az)
    P0, Phi0, C0 = inner_state(u0)
    P0 = dict(P0)
    Phi0 = {m: Phi0[m] + eps * phi1[m] * u0 for m in ODD}
    C0 = {k: C0[k] - eps * F2[k] for k in EVEN}
    if MUT == "stokes":
        Phi0 = {m: Phi0[m] * (1 + 1e-12) for m in ODD}      # a tiny free fast mode
    y0 = pack(P0, Phi0, C0)
    stats = {"P": 0.0, "F": 0.0, "C": 0.0, "P_vs_Z1": 0.0, "F_vs_Z1": 0.0, "C_vs_Z1": 0.0}
    track = []

    def record(z, y):
        P, Phi, C = unpack(y)
        Pi, Phii, Ci = inner_state(z)
        wP = max(abs(P[m] - Pi[m]) for m in ODD) / eps
        wF = max(abs(Phi[m] - Phii[m]) for m in ODD) / (eps * abs(z))
        wC = max(abs(C[k] - Ci[k]) for k in EVEN) / eps
        stats["P"], stats["F"], stats["C"] = max(stats["P"], wP), max(stats["F"], wF), max(stats["C"], wC)
        track.append((abs(z), float(np.angle(z)), wP, wF, wC))

    if MUT == "stokes":
        legs = [("real", None)]
    else:
        legs = [("ray", None), ("arc", None)]
    y = y0
    for leg, _ in legs:
        if leg == "ray":
            Ua, Ub = 1 / rn, 1 / Zt
            e = np.exp(-1j * az)
            zfun = lambda s: 1 / ((Ua - s) * e)
            dzfun = lambda s: e / ((Ua - s) * e) ** 2
            span = (0.0, Ua - Ub)
        elif leg == "arc":
            zfun = lambda s: Zt * np.exp(1j * (az - s))
            dzfun = lambda s: -1j * Zt * np.exp(1j * (az - s))
            span = (0.0, az)
        else:   # along the real axis in U, from 1/rn down to 1/Zt (z real increasing)
            Ua, Ub = 1 / rn, 1 / Zt
            zfun = lambda s: 1 / (Ua - s)
            dzfun = lambda s: 1 / (Ua - s) ** 2
            span = (0.0, Ua - Ub)
        npts = 60

        def blowup(s, yy):
            # terminal event: the state has left any neighbourhood of the inner solution (|Phi| > 1e3; |Phi| ~ 0.1 on
            # the path); only the mutation reaches it
            return 1e3 - np.max(np.abs(yy[2:4] + 1j * yy[9:11]))
        blowup.terminal = True
        sol = solve_ivp(rhs_factory(zfun, dzfun), span, y, method="DOP853", rtol=1e-10, atol=1e-14,
                        dense_output=True, events=blowup)
        if not sol.success:
            stats["fail"] = sol.message
            break
        s_end = sol.t[-1]
        for s in np.linspace(span[0], s_end, npts):
            record(zfun(s), sol.sol(s))
        if sol.status == 1:
            stats["fail"] = f"blow-up of the fast mode at |z| = {abs(zfun(s_end)):.4g}"
            break
        y = sol.y[:, -1]
    stats["n"], stats["rn"], stats["Zt"], stats["time"] = n, rn, Zt, time.time() - t0
    return stats, track


if __name__ == "__main__":
    n = float(OPTS.get("n", "1e8"))
    Zt = float(OPTS.get("Zt", "0.008"))
    stats, track = run(n, Zt)
    # the certified bound for these data (b5_bounds_k0.3_Z0.0085.json: forcing + phi_1 x^(F) + |c0| x^(C) + A^2 x^(v))
    with open(os.path.join(HERE, "b5_bounds_k0.3_Z0.0085.json")) as fh:
        ref = json.load(fh)[0]["supersolution"]
    dF, dC, dv = 76.56, 3.2313, 6.4625
    bound = {}
    for key, comps in (("P", ("P-1", "P+1")), ("F", ("F-1", "F+1")), ("C", ("C-2", "C+0", "C+2"))):
        bound[key] = max(ref["forcing"]["x"][c] + dF * ref["unit_F"]["x"][c] + dC * ref["unit_C"]["x"][c]
                         + dv * ref["unit_v"]["x"][c] for c in comps)
    print(f"n = {n:.0e}, r_n = {stats['rn']:.3e}, path to |z| = {Zt} then the arc to arg z = 0"
          + (" [MUTATION: Stokes-side start]" if MUT else "") + f"  ({stats['time']:.0f} s)")
    if "fail" in stats:
        print(f"  integration failed: {stats['fail']}")
    ok = "fail" not in stats
    for key in ("P", "F", "C"):
        lab = {"P": "|W_P|/eps", "F": "|W_Phi|/(eps|z|)", "C": "|W_C|/eps"}[key]
        within = stats[key] <= bound[key]
        ok = ok and within
        print(f"  max {lab:18s} = {stats[key]:.4g}   certified bound {bound[key]:.4g}   ratio {stats[key] / bound[key]:.2e}"
              f"   {'within' if within else 'VIOLATED'}")
    print(f"  expected from eps Z^(1): |W_Phi|/(eps|z|) ~ |phi_1| = 76.6 near z = 0, |W_C|/eps ~ |[F^2]| = 3.23")
    if MUT:
        print(f"MUTATION {MUT}: the bound must be violated: {'caught' if not ok else 'NOT caught'}")
        sys.exit(0 if not ok else 3)
    sys.exit(0 if ok else 1)

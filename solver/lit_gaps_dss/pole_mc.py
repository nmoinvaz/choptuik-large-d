"""Is H1's pole the momentum-constraint violation? (notes/LIT_GAPS_DSS.md, lead 1, test 3 in part)

At a stored state the horizon series is recomputed by the kernel (hpkernel, as hp/pole_test.py does) for horizon data
moved three ways, each by a relative delta:
  A   a_h scaled by 1 + delta (pole_test's move; violates the momentum constraint at the SSH);
  S   m_h scaled by 1 + delta and a_h re-slaved from it by the momentum constraint at x = c = 1,
        (omega d/dtheta - (n - 1) - 2 m_h^2)(B - 1) = 2 m_h^2,  a_h = B^(-1/2),
      solved in mpmath at 60 digits and split into double-double (no violation beyond the state's rounding);
  S0  m_h scaled by 1 + delta with a_h held (a violation of size about delta * q / n; the control of S).
For each, the series' radius R by pole_test.py's ratio test (its functions are imported). H1 (QUAD_DOUBLE 13.6) says
R = R0 + C ln(1/delta)/(n - 1) for A. The lead predicts R about the unperturbed radius for S at every delta, and a
fall for S0. First the slaved a_h is compared with the stored one (it must agree to near the state's precision).
Usage, from the repository root:  python3 solver/lit_gaps_dss/pole_mc.py STATE.npz [...]
"""
import json
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402
from hpkernel.driver import load, make_solver  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "hp"))
from pole_test import ratio_radius  # noqa: E402

DELTAS = (0.0, 1e-8, 1e-6, 1e-4, 1e-2)
mp.mp.dps = 60


def dmat(N):
    """Fourier differentiation matrix on N (even) points of [0, 2 pi), Nyquist derivative zero."""
    D = mp.matrix(N, N)
    for j in range(N):
        for k in range(N):
            if j != k:
                D[j, k] = mp.mpf(0.5) * (-1) ** (j - k) * mp.cot((j - k) * mp.pi / N)
    return D


def slave_ah(mh, n, omega, Dm):
    """a_h on the half grid from m_h (half grid, odd), in mpmath; returns mp list."""
    M = len(mh)
    full = [mp.mpf(v) for v in mh] + [-mp.mpf(v) for v in mh]
    N = 2 * M
    q = [2 * v * v for v in full]
    L = omega * Dm
    for j in range(N):
        L[j, j] -= (n - 1) + q[j]
    beta = mp.lu_solve(L, mp.matrix(q))
    return [1 / mp.sqrt(1 + beta[j]) for j in range(M)]


def violation(mh, ah, n, omega, Dm):
    """max|(omega D - (n - 1) - q)(B - 1) - q| / max|(n - 1)(B - 1)| at x = c = 1 (the momentum residual of the data)."""
    M = len(mh)
    full_m = [mp.mpf(v) for v in mh] + [-mp.mpf(v) for v in mh]
    full_a = [mp.mpf(v) for v in ah] + [mp.mpf(v) for v in ah]
    N = 2 * M
    q = [2 * v * v for v in full_m]
    b1 = [1 / (v * v) - 1 for v in full_a]
    r = omega * (Dm * mp.matrix(b1))
    res = max(abs(r[j] - ((n - 1) + q[j]) * b1[j] - q[j]) for j in range(N))
    return res / max(abs((n - 1) * v) for v in b1)


def split(vals):
    hi = np.array([float(v) for v in vals])
    lo = np.array([float(v - mp.mpf(h)) for v, h in zip(vals, hi)])
    return hi, lo


def main():
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    for path in paths:
        zh, zl, m = load(path)
        D, M, nodes = m["D"], m["M"], list(m["nodes"])
        kc, ks, kt = m.get("kc", 146), m.get("ks", 130), m.get("kt", 36)
        K = len(nodes)
        u0 = 1.0 - nodes[-1]
        n = mp.mpf(m.get("D_str", repr(D))) - 2
        Delta = mp.mpf(zh[-1]) + mp.mpf(zl[-1])
        omega = 2 * mp.pi / Delta
        om_, oa = 2 * M + 4 * M * K, 3 * M + 4 * M * K
        mh = [mp.mpf(zh[om_ + j]) + mp.mpf(zl[om_ + j]) for j in range(M)]
        ah = [mp.mpf(zh[oa + j]) + mp.mpf(zl[oa + j]) for j in range(M)]
        Dm = dmat(2 * M)
        ah_s = slave_ah(mh, n, omega, Dm)
        dev = max(abs(x - y) for x, y in zip(ah_s, ah)) / max(abs(y - 1) for y in ah)
        print(f"D {D:.4f}  slaved a_h against stored: max|diff| / max|a_h - 1| = {mp.nstr(dev, 3)}", flush=True)
        s = make_solver(D, M, nodes=nodes, kt=kt, kc=kc, ks=ks, threads=1)
        s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
        sig = s.series_scale(1)
        print(f"{'delta':>7} {'R_A':>7} {'R_S':>7} {'R_S0':>7}   {'E_A':>8} {'E_S':>8} {'E_S0':>8}"
              "   (R: ratio-test radius; E: the data's momentum residual at the SSH)", flush=True)
        for delta in DELTAS:
            Rs, Es = [], []
            for mode in ("A", "S", "S0"):
                h2, l2 = zh.copy(), zl.copy()
                if mode == "A":
                    h2[oa:oa + M] *= 1.0 + delta
                    l2[oa:oa + M] *= 1.0 + delta
                else:
                    mh2 = [v * (1 + mp.mpf(delta)) for v in mh]
                    hi, lo = split(mh2)
                    h2[om_:om_ + M], l2[om_:om_ + M] = hi, lo
                    if mode == "S":
                        hi, lo = split(slave_ah(mh2, n, omega, Dm))
                        h2[oa:oa + M], l2[oa:oa + M] = hi, lo
                mh_d = [mp.mpf(h2[om_ + j]) + mp.mpf(l2[om_ + j]) for j in range(M)]
                ah_d = [mp.mpf(h2[oa + j]) + mp.mpf(l2[oa + j]) for j in range(M)]
                Es.append(violation(mh_d, ah_d, n, omega, Dm))
                hh, hl = s.series(h2, l2, 1, True)
                c = (np.asarray(hh) + np.asarray(hl)).reshape(-1, 4, M)
                R, _ = ratio_radius(c, u0, sig)
                Rs.append(R)
            print(f"{delta:7.0e} " + " ".join(f"{r if r else float('nan'):7.3f}" for r in Rs) + "   "
                  + " ".join(f"{float(e):8.1e}" for e in Es), flush=True)
            print(json.dumps(dict(state=os.path.basename(path), D=D, delta=delta, R_A=Rs[0], R_S=Rs[1], R_S0=Rs[2],
                                  E_A=float(Es[0]), E_S=float(Es[1]), E_S0=float(Es[2]))), file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()

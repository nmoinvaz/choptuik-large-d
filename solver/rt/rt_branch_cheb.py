"""Chebyshev coefficients of the horizon branches s^lambda (s = 1 + xi), and how well degree-N polynomials hold them
(issue #163).

    python3 solver/rt/rt_branch_cheb.py [--lam 5.68,26.23] [--om 0,3.87] [--planted]

For each lambda (with the oscillation cos(om ln s), om = k / mu for harmonic k) it prints |c_m| for m = 0, 10, ..., 120
of f = s^lambda cos(om ln s) / max|f|, from a 400-point Chebyshev-Lobatto transform at 120 digits, and the max error of
the interpolant on N = 61, 81, 101, 121 points (on a grid dense near xi = -1). Those numbers are what a polynomial
basis does not hold of the branch. Planted check: --planted replaces lambda by its nearest integer with om = 0, a
polynomial, whose coefficients beyond m = lambda must then vanish to 1e-80 (exit 1 otherwise; without it
the script only prints and exits 0).
"""
import argparse
import sys
import mpmath as mp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", default="5.68,26.23")
    ap.add_argument("--om", default="0,3.87")
    ap.add_argument("--planted", action="store_true")
    a = ap.parse_args()
    mp.mp.dps = 120
    lams = [float(v) for v in a.lam.split(",")]
    oms = [float(v) for v in a.om.split(",")]
    ok = True
    M = 400
    xs = [mp.cos(mp.pi * j / M) for j in range(M + 1)]
    for lam, om in zip(lams, oms):
        if a.planted:
            lam, om = float(round(lam)), 0.0
        f = lambda x: (1 + x) ** lam * mp.cos(om * mp.log(1 + x)) if x > -1 else mp.mpf(0)
        fs = [f(x) for x in xs]
        mx = max(abs(v) for v in fs)
        c = []
        for m in range(0, 121, 10):
            s = mp.fsum(fs[j] * (mp.mpf(1) / 2 if j in (0, M) else 1) * mp.cos(mp.pi * j * m / M) for j in range(M + 1))
            c.append(abs(s * 2 / M) / mx)
        print(f"lambda {lam} om {om}: |c_m|/max|f| m = 0, 10, ..., 120: " + " ".join(f"{float(v):.0e}" for v in c),
              flush=True)
        errs = []
        for N in (61, 81, 101, 121):
            xg = [mp.cos(mp.pi * j / N) for j in range(N + 1)]
            fg = [f(x) for x in xg]
            w = [(-1) ** j * (mp.mpf(1) / 2 if j in (0, N) else 1) for j in range(N + 1)]
            e = 0
            for i in range(1, 300):
                x = -1 + 2 * (mp.mpf(i) / 300) ** 3
                q = [w[j] / (x - xg[j]) for j in range(N + 1)]
                p = mp.fsum(q[j] * fg[j] for j in range(N + 1)) / mp.fsum(q)
                e = max(e, abs(p - f(x)))
            errs.append(e / mx)
        print(f"  interpolation error / max|f| on N = 61, 81, 101, 121: " + " ".join(f"{float(v):.1e}" for v in errs),
              flush=True)
        if a.planted:
            tail = max(c[int(lam) // 10 + 1:]) if int(lam) // 10 + 1 < len(c) else 0
            ok &= tail < 1e-80
            print(f"  polynomial control: largest |c_m| beyond m = {int(lam)} is {float(tail):.0e} (< 1e-80: {tail < 1e-80})")
    print(f"exit {0 if ok else 1}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

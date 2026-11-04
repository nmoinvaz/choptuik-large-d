#!/usr/bin/env python3
"""A toy of the horizon block's branch problem: can an integral formulation replace the horizon series?

The model has the horizon's two features: a regular singular point at u = 0 whose exponents are 0 (the regular
branch) and s (the singular one, s ~ D/2 at the real horizon), and a coupling between the two:

    u y1' = u (al y1 + be y2),    u y2' = s y2 + u (ga y1 + de y2),    the regular solution: y1(0) = 1, y2(0) = 0.

Three ways to get the regular solution at the matching point u0 = 0.3, each at P = 16, 32 and 64 digits, against a
300-digit reference:
  (a) the series (as the solver's horizon block): y = sum a_k u^k from the recurrence k a1_k = al a1_(k-1) + be a2_(k-1),
      (k - s) a2_k = ga a1_(k-1) + de a2_(k-1), summed at u0 until the terms fall below 10^-P;
  (b) the integral equation from the horizon, y2 = u^s int_0^u t^(-s) (ga y1 + de y2) dt: its integrand behaves as t^-s
      at t -> 0, so it converges only for s < 1; it becomes usable after the first ceil(s) Taylor terms of the integrand
      are subtracted, which are (a)'s first ceil(s) coefficients;
  (c) shooting outward from u_e = 0.01 (where the series is exact) to u0: a rounding error at u_e excites the singular
      branch u^s (1 + O(u)), which grows by its own ratio between u_e and u0, so the shot loses that many digits; the
      ratio is computed from the singular branch's own Frobenius series.
Check (exit 1 on failure): (a) gives at least P - 3 correct digits at every s and P. --mutate stops the series at
half the digits (terms below 10^-(P/2)) at P = 32: (a)'s digits must then fall below P - 3 (the control).
    cd hp && python3 horizon_toy.py [--mutate]
"""
import sys

import mpmath as mp

AL, BE, GA, DE = mp.mpf("0.7"), mp.mpf("-1.3"), mp.mpf("2.1"), mp.mpf("-0.4")
U0, UE = mp.mpf("0.3"), mp.mpf("0.01")
SS = ("0.5", "2.5", "10.5", "20.5", "40.5", "98.5")


def regular(s, u, dps, half=False):
    """the regular solution at u by its series, at dps digits; returns (y1, y2, terms used)"""
    with mp.workdps(dps):
        s, u = mp.mpf(s), mp.mpf(u)
        a1, a2 = mp.mpf(1), mp.mpf(0)
        y1, y2, k, small = a1, a2, 0, 0
        tol = mp.mpf(10) ** (-(dps // 2) if half else -dps - 5)
        last = (mp.mpf(0), mp.mpf(0))
        while small < 3 and k < 5000:
            k += 1
            a1, a2 = (AL * a1 + BE * a2) / k, (GA * a1 + DE * a2) / (k - s)
            t1, t2 = a1 * u ** k, a2 * u ** k
            y1 += t1
            y2 += t2
            last = (t1, t2)
            small = small + 1 if max(abs(t1), abs(t2)) < tol * max(abs(y1), abs(y2), 1) else 0
        return +y1, +y2, k


def singular_ratio(s, ue, u0, dps=60):
    """|y_s(u0)| / |y_s(ue)| of the singular branch y = u^s (b0 + b1 u + ...), b0 = (0, 1): the growth of a rounding
    error at ue that excites it"""
    with mp.workdps(dps):
        s = mp.mpf(s)

        def at(u):
            b1, b2, y1, y2 = mp.mpf(0), mp.mpf(1), mp.mpf(0), mp.mpf(1)
            for k in range(1, 400):
                # u y' = A0 y + u A1 y on y = u^s sum b_k u^k: (k + s) b1_k = al b1_(k-1) + be b2_(k-1) (exponent 0
                # row: (k + s - 0)), (k + s - s) b2_k = ga b1_(k-1) + de b2_(k-1)
                b1, b2 = (AL * b1 + BE * b2) / (k + s), (GA * b1 + DE * b2) / k
                y1 += b1 * u ** k
                y2 += b2 * u ** k
            return u ** s * mp.sqrt(y1 ** 2 + y2 ** 2)
        return at(u0) / at(ue)


def main():
    mutate = "--mutate" in sys.argv
    ok = True
    print("| s (D ~ 2s + 4) | P | (a) series: terms, correct digits at u0 | (b) integral from u = 0 | (c) outward shot from 0.01: digits lost |")
    print("|---|---|---|---|---|")
    for s in SS:
        ref = regular(s, U0, 300)
        lost = float(mp.log10(singular_ratio(s, UE, U0)))
        for P in (16, 32, 64):
            y1, y2, k = regular(s, U0, P, half=(mutate and P == 32))
            err = max(abs(y1 - ref[0]) / abs(ref[0]), abs(y2 - ref[1]) / max(abs(ref[1]), mp.mpf(10) ** -300))
            dig = float(-mp.log10(err)) if err > 0 else 300.0
            good = dig >= P - 3
            ok &= good
            integ = "converges" if mp.mpf(s) < 1 else f"diverges as t^-{s} unless its first {int(mp.ceil(mp.mpf(s)))} Taylor terms are subtracted"
            print(f"| {s} (~{2 * float(s) + 4:.0f}) | {P} | {k} terms, {min(dig, P + 20):.1f} digits{'' if good else ' FAIL'} | {integ} | "
                  f"{lost:.1f}{' (more than P: no digits left)' if lost > P else ''} |")
    print(f"(a) every case keeps at least P - 3 digits: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

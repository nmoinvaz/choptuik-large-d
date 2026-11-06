#!/usr/bin/env python3
"""Does phi = artanh(w), w a solution of the flat heat equation w_s = w_tt, solve the inner problem in general?

Rozali and Way's exact oscillaton (JHEP 11 (2018) 106, Sec. 2.3, Eq. (10)) is phi = artanh(eps e^{-w^2 r^2/2} sin(w tau)),
i.e. phi = artanh(w) with w = eps e^{-w^2 s} sin(w t) a single heat mode (s = r^2/2, t = tau). If every heat solution
gave a solution this way, the inner problem would be linearizable, against the Painleve test of painleve.py. This
script tests that, in Rozali and Way's form of eq. (8): phi_s = c^-1 d_t(c^-1 phi_t), c_s = phi_t^2/c, with
q = c^2 -> 1 as s -> infinity (their gauge, delta_0 -> 0 at large r).

    q(s, t) = 1 - int_s^inf 2 phi_t^2 ds'         (from c_s = phi_t^2/c),
    residual R = c^2 phi_s - phi_tt + (c_t/c) phi_t   (zero iff the field equation holds),

evaluated by quadrature at a few points. Planted truth: the single mode must give R = 0 (and q equal to the closed
form (1 - eps^2 e^{-2 w^2 s})/(1 - w^2)); the two-mode superposition is the test.

    python3 oscillaton_lin.py [--mutate=coef]     --mutate=coef uses c_s = 2 phi_t^2/c for the single mode, which must
                                                  then fail (exit status 1 if the planted check does not behave)
"""
import sys

import mpmath as mp

mp.mp.dps = 30


def make(modes):
    def w(s, t):
        return sum(e * mp.exp(-k * k * s) * mp.sin(k * t) for e, k in modes)
    return w


def residual(w, s0, t0, coef=2):
    def phi(s, t):
        return mp.atanh(w(s, t))

    def phit(s, t):
        return mp.diff(lambda tt: phi(s, tt), t)

    def q(s, t):
        return 1 - mp.quad(lambda x: coef * phit(x, t) ** 2, [s, s + 2, mp.inf])

    qv = q(s0, t0)
    qt = mp.diff(lambda tt: q(s0, tt), t0)
    ph_s = mp.diff(lambda ss: phi(ss, t0), s0)
    ph_t = phit(s0, t0)
    ph_tt = mp.diff(lambda tt: phi(s0, tt), t0, 2)
    # c^2 phi_s = phi_tt - (c_t/c) phi_t, c_t/c = q_t/(2q)
    R = qv * ph_s - ph_tt + qt / (2 * qv) * ph_t
    return R, qv


def main(argv):
    coef = 4 if "--mutate=coef" in argv else 2
    pts = [(mp.mpf("0.3"), mp.mpf("0.7")), (mp.mpf("0.1"), mp.mpf("1.9")), (mp.mpf("0.5"), mp.mpf("-0.4"))]
    single = [(mp.mpf("0.6"), 1)]
    double = [(mp.mpf("0.5"), 1), (mp.mpf("0.3"), 3)]
    bad = 0
    print("single mode (planted: R must vanish, q must match the closed form)")
    w1 = make(single)
    for s0, t0 in pts:
        R, qv = residual(w1, s0, t0, coef)
        eps = single[0][0]
        qc = (1 - eps ** 2 * mp.exp(-2 * s0)) / (1 - w1(s0, t0) ** 2)
        print(f"   s {mp.nstr(s0, 3)} t {mp.nstr(t0, 3)}: R = {mp.nstr(R, 5)}, q - closed form = {mp.nstr(qv - qc, 5)}")
        if abs(R) > 1e-15:
            bad += 1
    print("two modes (the test)")
    w2 = make(double)
    worst = 0
    for s0, t0 in pts:
        R, qv = residual(w2, s0, t0)
        worst = max(worst, abs(R))
        print(f"   s {mp.nstr(s0, 3)} t {mp.nstr(t0, 3)}: R = {mp.nstr(R, 5)}")
    print(f"two-mode residual max {mp.nstr(worst, 3)}: "
          f"{'artanh of a heat solution is NOT a solution in general' if worst > 1e-6 else 'consistent with a linearization'}")
    if bad:
        print("planted single-mode check FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

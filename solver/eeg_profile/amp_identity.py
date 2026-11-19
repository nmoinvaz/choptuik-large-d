#!/usr/bin/env python3
"""The selection rule's amplitude against the gauge rate's 1/n coefficient (#638; notes/DIRECTION_REVIEW.md 2.4).

Ecker, Ecker and Grumiller's Eq. (15) (arXiv:2611.07007v1, notes/EEG_PROFILE.md section 1) gives
N0 = (Delta_1 - 4 pi mu_1)/(4 pi mu_0) = <bt^2>, and with lam_g = 4 pi mu / Delta and bt = sqrt(2) F at D = infinity
this is lim n (1 - lam_g) = 2 <F^2> = A^2 + sum_{nu >= 3} 2 |F_nu|^2 ... in this repository's normalization
F = A sin(theta) + sum_{nu >= 3} f_nu(A) e^{i nu theta} + c.c. (solver/eeg_profile/tuned_profile.py), 2 <F^2> =
4 sum_{nu >= 1 odd} |F_nu|^2 with F_1 = -iA/2. The fitted c1 = 7.4165644 +- 4.4e-6 (a spread over fit choices,
notes/MODE_COUNT.md section 10, hp/lam_gauge_fit.py; c0 and c1 are the same in 1/n and 1/D) comes from finite-D
states and shares no machinery with the D = infinity inner problem.

Inputs, from the amplitude expansion at omega_inf (solver/inner_amp/scan45J1250 and scan45J1500, K = 45, sigma_0 ..
sigma_22 and the tuned harmonics f):
- x0, the zero of S_1(x) = sum sigma_j x^j, x = A^2, from the near-diagonal Pade approximants [10/10], [10/11],
  [11/10], [11/11], [10/12] and [12/10], at each J: the set whose spread notes/PADE_FAR_END.md 14 quotes (8.4-8.8e-5),
  fixed before #638. The estimator range is their hull over both J (a spread over estimators, not a bound);
- 2 <F^2> at A0 = sqrt(Re x0), summed from f to K = 45 (tuned_profile.harmonics), through nu = 25.

Pass, fixed before the run (#638, comment of 2026-11-19): c1 lies inside the range of 2 <F^2>(x0) over the hull of x0.
Controls, each of which must fail the same test: <F^2> in place of 2 <F^2>, and the fundamental alone (A0^2, the
harmonics dropped), which shows the test resolves the harmonic share. (#638's control n - 1 -> n cannot fail: the
leading coefficient c1 is the same in 1/n and 1/(n + 1).) Planted: a series with a known real zero through the same
Pade path must return it to 1e-10.

Context, not part of the test: the diagonal Pade zero by order (16 to 22), and the amplitude c1 implies through the
identity (2 <F^2>(A) = c1 with the expansion's harmonics) against the finite-D states' A_inf = 2.71920535
(notes/EEG_PROFILE.md 3.5.1). Result of the first run (#638): the planted zero and both controls pass; c1 lies 6.2e-6
below the range of 2 <F^2>(x0), so the test fails (exit 1); c1 implies A = 2.71920533, 1.9e-8 from the states' A_inf
(notes/EEG_PROFILE.md 3.5.2).

    python3 amp_identity.py            exit 0 iff the planted zero is found, c1 passes and both controls fail
"""
import contextlib
import io
import os
import sys

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
with contextlib.redirect_stdout(io.StringIO()):     # tuned_profile runs its own analysis on import
    _argv, sys.argv = sys.argv, sys.argv[:1]
    import tuned_profile as TP  # noqa: E402
    sys.argv = _argv

C1, C1_ERR = mp.mpf("7.4165644"), mp.mpf("4.4e-6")     # MODE_COUNT.md section 10 (README "1 - 7.4165644/n + ...")
NEAR = mp.mpf("7.394")
JSONS = TP.DEFAULT
NUMAX = 25


def load_sigma(path):
    import json
    d = json.load(open(path))
    return [mp.mpc(mp.mpf(re_), mp.mpf(im_)) for re_, im_ in d["sigma_str"]]


def pade_zero(c, m, n, near=NEAR):
    """the zero of [m/n]'s numerator nearest `near`"""
    p, _ = mp.pade(c, m, n)
    roots = mp.polyroots(p[::-1], maxsteps=400, extraprec=400)
    return min(roots, key=lambda z: abs(z - near))


ORDERS = [(10, 10), (10, 11), (11, 10), (11, 11), (10, 12), (12, 10)]


def zeros(c):
    return {mn: pade_zero(c[:sum(mn) + 1], *mn) for mn in ORDERS}


def two_F2(f, K, A):
    F, _, _ = TP.harmonics(f, K, float(A), NUMAX)
    return 4 * sum(abs(v) ** 2 for v in F.values())


def planted():
    """S(x) = (1 - x/xp) exp(-0.3 x) (1 + 0.1 i x), Taylor to order 22: the ORDERS' zeros must give xp"""
    xp = mp.mpf("7.39123456789")
    c = mp.taylor(lambda x: (1 - x / xp) * mp.exp(-0.3 * x) * (1 + 0.1j * x), 0, 22)
    z = zeros(c)
    err = max(abs(v - xp) for v in z.values())
    ok = err < 1e-10
    print(f"planted: zero {xp}, the six near-diagonal Pade zeros within {mp.nstr(err, 3)}: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    mp.mp.dps = 60
    ok_plant = planted()
    xs, vals = [], []
    for path in JSONS:
        sig = load_sigma(path)
        K, J, f = TP.load_f(path)
        z = zeros(sig)
        re = [mp.re(v) for v in z.values()]
        print(f"J {J}: " + ", ".join(f"[{m}/{n}] {mp.nstr(mp.re(v), 10)}" for (m, n), v in z.items())
              + f"; Im at [11/11] {mp.nstr(mp.im(z[(11, 11)]), 3)}")
        for zz in re:
            xs.append(zz)
            vals.append((J, zz, two_F2(f, K, mp.sqrt(zz))))
        trend = []
        for top in range(16, 23):
            m = top // 2
            zz = pade_zero(sig[:top + 1], m, top - m)
            trend.append(f"[{m}/{top - m}] {mp.nstr(mp.re(zz), 9)}")
        print(f"  context, the diagonal zero by order: " + ", ".join(trend))
    lo, hi = min(xs), max(xs)
    t = [v for _, _, v in vals]
    tlo, thi = min(t), max(t)
    mid = [v for J, x, v in vals if x == sorted(xs)[len(xs) // 2]][0]
    share = (mid - float(sorted(xs)[len(xs) // 2])) / mid
    print(f"x0 hull {mp.nstr(lo, 10)} .. {mp.nstr(hi, 10)} (A0 {mp.nstr(mp.sqrt(lo), 9)} .. {mp.nstr(mp.sqrt(hi), 9)})")
    print(f"2<F^2>(x0) {tlo:.9f} .. {thi:.9f}; harmonic share sum_(nu>=3) 2|F_nu|^2 / 2<F^2> {share:.6e}")
    d_lo, d_hi = float(C1) - thi, float(C1) - tlo
    passed = tlo <= float(C1) <= thi or abs(d_lo) <= float(C1_ERR) or abs(d_hi) <= float(C1_ERR)
    print(f"c1 {mp.nstr(C1, 8)} +- {mp.nstr(C1_ERR, 2)}: c1 - 2<F^2>(x0) in [{d_lo:+.2e}, {d_hi:+.2e}]: {'PASS' if passed else 'FAIL'}")
    # context: the amplitude c1 implies through the identity (2<F^2>(A) = c1, the harmonics of the amplitude expansion),
    # against the finite-D states' fundamental amplitude A_inf = 2.71920535 (notes/EEG_PROFILE.md 3.5.1,
    # solver/inner_pert/ainf_from_states.py, half-range 2e-8 over the converged fit choices)
    K, J, f = TP.load_f(JSONS[0])
    def x_of(c):
        """x with 2<F^2>(sqrt x) = c, by secant in double precision (two_F2 is a double sum)"""
        a, b = 7.3940, 7.3942
        ga, gb = two_F2(f, K, a ** 0.5) - c, two_F2(f, K, b ** 0.5) - c
        for _ in range(30):
            if gb == ga:
                break
            a, b, ga = b, b - gb * (b - a) / (gb - ga), gb
            gb = two_F2(f, K, b ** 0.5) - c
        return mp.mpf(b)
    xc = x_of(float(C1))
    Ac = mp.sqrt(xc)
    dA = lambda c: mp.sqrt(x_of(float(c))) - Ac
    print(f"context: c1 implies A {mp.nstr(Ac, 9)} (+- {mp.nstr(abs(dA(C1 + C1_ERR)), 2)} from c1's spread); the states' "
          f"A_inf 2.71920535 differs by {mp.nstr(mp.mpf('2.71920535') - Ac, 2)}; the selection's x0 hull gives A0 "
          f"{mp.nstr(mp.sqrt(lo), 9)} .. {mp.nstr(mp.sqrt(hi), 9)}")
    # controls
    half = (tlo / 2 <= float(C1) <= thi / 2)
    fund = (float(lo) <= float(C1) <= float(hi))
    print(f"control <F^2>: {tlo / 2:.6f} .. {thi / 2:.6f}, c1 inside {'yes (DID NOT FIRE)' if half else 'no (fired)'}")
    print(f"control fundamental alone: {float(lo):.9f} .. {float(hi):.9f}, c1 - x0 {float(C1 - hi):+.3e} .. "
          f"{float(C1 - lo):+.3e}, inside {'yes (DID NOT FIRE)' if fund else 'no (fired)'}")
    good = ok_plant and passed and not half and not fund
    print("ALL PASS" if good else "FAIL")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())

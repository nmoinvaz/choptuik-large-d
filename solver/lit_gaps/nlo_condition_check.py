"""Does the profile of arXiv:2611.07007 Eq. (16) satisfy the NLO condition of arXiv:2601.14358 Eq. (20)? (#1089)

Eq. (20) of 2601.14358 asks beta'' + 3 beta' = 0 (tau derivatives) at every zero of beta. This script evaluates
R = (beta'' + 3 beta') / (3 beta') at each zero of a periodic profile, so R = 0 is the condition and R = 1 means the
second derivative vanishes there. Profiles:

  eeg16    2611.07007 Eq. (16): A Re[e^{i th} / (1 + q e^{i(2 th + delta)})], q 0.0551, delta 0.0985, th = 2 pi tau / Delta,
           Delta = 1.6439 (their Eq. (9)); A drops out of R.
  laiwen   2610.04960 Eq. (4) (= 2601.14358 Eq. (21)): cos 2 pi tau + sin(6 pi tau)/15.9476, Delta = 1. Built to satisfy
           Eq. (20), so it is the planted control: |R| must be below 1e-4 at every zero.
  tuned    the D = infinity tuned profile F of notes/SECOND_ORDER.md 1.2 (harmonics nu <= 7, b = sum 2 Re(c e^{i nu th})),
           Delta = 1.643893746.

Usage: python3 nlo_condition_check.py [--mutate=drop3]
  --mutate=drop3: removes the third harmonic from the laiwen control; its |R| must then exceed 0.1 (exit 1 otherwise).
Exit 1 if the control fails.
"""
import sys

import mpmath as mp

mp.mp.dps = 40


def eeg16(th):
    q, d = mp.mpf('0.0551'), mp.mpf('0.0985')
    return mp.re(mp.exp(1j * th) / (1 + q * mp.exp(1j * (2 * th + d))))


TUNED = {1: mp.mpc(0, '-1.35959'), 3: mp.mpc('0.007383', '-0.074488'), 5: mp.mpc('0.000756', '-0.004212'),
         7: mp.mpc('0.0000613', '-0.0002377')}


def tuned(th):
    return sum(2 * mp.re(c * mp.exp(1j * nu * th)) for nu, c in TUNED.items())


def zeros_R(f_th, Delta, label):
    """f_th is a function of th = 2 pi tau / Delta; derivatives in tau carry factors of w = 2 pi / Delta."""
    w = 2 * mp.pi / Delta
    out = []
    grid = [2 * mp.pi * k / 400 for k in range(401)]
    for a, b in zip(grid[:-1], grid[1:]):
        if f_th(a) * f_th(b) < 0:
            z = mp.findroot(f_th, (a, b), solver='anderson')
            d1 = w * mp.diff(f_th, z, 1)
            d2 = w ** 2 * mp.diff(f_th, z, 2)
            # the period Eq. (20) would select for this shape: w^2 F'' + 3 w F' = 0 (theta derivatives)
            dsel = -2 * mp.pi * mp.diff(f_th, z, 2) / (3 * mp.diff(f_th, z, 1))
            out.append((z, (d2 + 3 * d1) / (3 * d1), dsel))
    print(f"{label}: Delta {mp.nstr(Delta, 10)}, {len(out)} zeros per 2 pi in theta")
    for z, R, dsel in out:
        print(f"  theta {mp.nstr(z, 8):>12}  R = {mp.nstr(R, 6)}  Delta selected by Eq. (20) for this shape "
              f"{mp.nstr(dsel, 6)}")
    return out


def main():
    mutate = '--mutate=drop3' in sys.argv
    zeros_R(eeg16, mp.mpf('1.6439'), "2611.07007 Eq. (16)")
    zeros_R(tuned, mp.mpf('1.643893746'), "tuned D = inf profile F (SECOND_ORDER 1.2)")
    c3 = 0 if mutate else 1 / mp.mpf('15.9476')
    # Lai-Wen: tau-period 1 = Delta, th = 2 pi tau
    lw = lambda th: mp.cos(th) + c3 * mp.sin(3 * th)
    res = zeros_R(lw, mp.mpf(1), "2610.04960 Eq. (4) control" + (" MUTATED drop3" if mutate else ""))
    worst = max(abs(R) for _, R, _ in res)
    ok = worst > 0.1 if mutate else worst < 1e-4
    print(f"control: max |R| = {mp.nstr(worst, 4)} -> {'as expected' if ok else 'CONTROL FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()

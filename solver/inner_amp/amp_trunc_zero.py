#!/usr/bin/env python3
"""The zero (A, omega/omega_inf) of a model's amplitude series S_1/S_1^(1) = sum_n sigma_n x^n, x = A^2, from
amp_flint_par.py outputs at three omegas (#1016's calibrate-then-predict test).

At each omega the zero x0(omega) is a zero near the guess of the Pade approximants [m/n] of S(x), m + n = N (N + 1 the
number of sigma_n), for the approximants with m >= N/2; their median is x0 and their spread its Pade error. x0(omega)
is interpolated quadratically through the three omegas, and the selection is the omega with Im x0 = 0, A0 = sqrt(Re x0).
The error is the largest change of (A0, omega0) over the Pade orders used (the interpolation is redone with each
order's x0) and over dropping the top two sigma_n.

    python3 amp_trunc_zero.py FILE_LO FILE_MID FILE_HI [--guess=7.43] [--drop=2]
    python3 amp_trunc_zero.py --planted         a series with a known zero: must be recovered to 1e-10 (exit 0)
    python3 amp_trunc_zero.py --planted --mutate=shift   the planted zero moved by 1e-6: must fail (exit 1)
"""
import json
import sys

import mpmath as mp

mp.mp.dps = 50
W0 = 2 * mp.pi / mp.mpf("1.64389375")
MUT = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--mutate=")), "")


def opt(name, default):
    return type(default)(next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default))


def load(fn):
    d = json.load(open(fn))
    return mp.mpf(d["wf"]), [mp.mpc(mp.mpf(a), mp.mpf(b)) for a, b in d["sigma_str"]]


def pade_zeros(sig, guess):
    """{(m, n): the zero of the [m/n] numerator nearest the guess}, m + n = len(sig) - 1, m >= (len - 1)/2"""
    N = len(sig) - 1
    out = {}
    for m in range((N + 1) // 2, N + 1):
        n = N - m
        try:
            p, q = mp.pade(sig, m, n)
        except Exception:
            continue
        roots = mp.polyroots(p[::-1], maxsteps=400, extraprec=400)
        r = min(roots, key=lambda z: abs(z - guess))
        out[(m, n)] = r
    return out


def interp_root(ws, xs):
    """quadratic interpolation x0(w) through three points; returns (w0, x0(w0)) with Im x0(w0) = 0"""
    (w1, w2, w3), (x1, x2, x3) = ws, xs
    L = lambda w: (x1 * (w - w2) * (w - w3) / ((w1 - w2) * (w1 - w3)) + x2 * (w - w1) * (w - w3) / ((w2 - w1) * (w2 - w3))
                   + x3 * (w - w1) * (w - w2) / ((w3 - w1) * (w3 - w2)))
    w0 = mp.findroot(lambda w: mp.im(L(w)), w2)
    return w0, L(w0)


def solve(series, guess, drop):
    """series: [(wf, sigma list)] at three omegas; returns (A0, wf0, errA, errw, detail)"""
    ws = [w for w, _ in series]
    results = {}
    for dr in (0, drop):
        pz = [pade_zeros(s[:len(s) - dr], guess) for _, s in series]
        common = set(pz[0]) & set(pz[1]) & set(pz[2])
        for key in sorted(common):
            w0, x0 = interp_root(ws, [pz[i][key] for i in range(3)])
            results[(dr, key)] = (mp.sqrt(mp.re(x0)), w0)
    base = [v for (dr, k), v in results.items() if dr == 0]
    A = sorted(v[0] for v in base)[len(base) // 2]
    w = sorted(v[1] for v in base)[len(base) // 2]
    eA = max(abs(v[0] - A) for v in results.values())
    ew = max(abs(v[1] - w) for v in results.values())
    return A, w, eA, ew, results


def planted():
    """S(x) = (1 - x/x0(w)) exp(a x + b x^2) with x0(w) = 7.43 (1 + 0.3 i (w - 1.0156)), sampled at three w; the exact
    selection is w = 1.0156, A = sqrt(7.43)"""
    x00 = mp.mpf("7.43") * (1 + (mp.mpf("1e-6") if "shift" in MUT else 0))
    a, b = mp.mpc("-0.21", "0.13"), mp.mpc("0.004", "-0.002")
    series = []
    for w in (mp.mpf("1.0136"), mp.mpf("1.0156"), mp.mpf("1.0176")):
        x0 = mp.mpf("7.43") * (1 + 0.3j * (w - mp.mpf("1.0156")))
        x0 = x0 * x00 / mp.mpf("7.43")
        f = lambda x: (1 - x / x0) * mp.exp(a * x + b * x * x)
        sig = mp.taylor(f, 0, 30)
        series.append((w, sig))
    A, wf, eA, ew, _ = solve(series, mp.mpf("7.43"), 2)
    dA, dw = abs(A - mp.sqrt(mp.mpf("7.43"))), abs(wf - mp.mpf("1.0156"))
    good = dA < 1e-10 and dw < 1e-10
    print(f"planted: A0 {mp.nstr(A, 15)} (exact {mp.nstr(mp.sqrt(mp.mpf('7.43')), 15)}), wf0 {mp.nstr(wf, 15)} (exact 1.0156); "
          f"|dA| {mp.nstr(dA, 3)}, |dw| {mp.nstr(dw, 3)}, method spreads {mp.nstr(eA, 3)}, {mp.nstr(ew, 3)}  (bar 1e-10)")
    return good


if __name__ == "__main__":
    if "--planted" in sys.argv:
        good = planted()
        print("PASS" if good else "FAIL")
        sys.exit(0 if good else 1)
    files = [a for a in sys.argv[1:] if not a.startswith("--")]
    series = sorted((load(f) for f in files), key=lambda t: t[0])
    A, wf, eA, ew, res = solve(series, mp.mpf(opt("guess", 7.43)), opt("drop", 2))
    for (dr, key), (a, w) in sorted(res.items()):
        print(f"   drop {dr}, Pade {key}: A0 {mp.nstr(a, 12)}  omega0/omega_inf {mp.nstr(w, 12)}")
    print(f"zero: A0 = {mp.nstr(A, 12)} +- {mp.nstr(eA, 2)},  omega0/omega_inf = {mp.nstr(wf, 12)} +- {mp.nstr(ew, 2)}  "
          f"({len(series[0][1])} sigma_n; the errors are the largest moves over Pade orders and the dropped orders)")

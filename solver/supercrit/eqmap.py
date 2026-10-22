#!/usr/bin/env python3
"""Ecker, Ecker & Grumiller's self-similar equations are this repository's E1-E4 under an explicit variable map.

EEG (PRL 136, 191401 (2026), arXiv:2601.14358, Eqs. (2)-(4)), with their eps_E = 1/(D-1) = 1/(n+1):
    (3a) x Om_x + x^2 Om^2 = (1/eps_E)((1-2 eps_E) Pi^2 - Om) + Om (Pi^2 x^2 + x^4 Psi^2) + (1/eps_E - 2) x^2 Psi^2
    (3b) (1/x) d_x ln f = Om            (printed with 1/r; 1/x is the only reading consistent with Eq. (11))
    (3c) (d_tau + x d_x) Pi  = f x Psi_x + f (1/eps_E + x^2 Om) Psi - Pi
    (3d) (d_tau + x d_x) Psi = f (1/x) Pi_x + f Om Pi - 2 Psi
Repository (hp/src/model.rs; README "Large D"), n = D - 2, with y = p - m, sig = p + m, u = (n-1)(b-1):
    E1  x c_x = c u
    E2  x u_x = (n - 1 + u)(P - u),            P = p^2 + m^2 = (y^2 + sig^2)/2       (E2 of the README, b = 1 + u/(n-1))
    (A) x^2 y_x + x y_tau - x c sig_x = c (n - 1 + u) sig        (E3 - E4)
    (B) x^2 sig_x + x sig_tau - x c y_x = c (u - 1) y              (E3 + E4)
The map:   c = f,   u = x^2 Om,   y = sqrt(2) x Pi,   sig = sqrt(2) x^2 Psi,   eps_E = 1/(n+1).
The script substitutes the map into EEG's equations and reduces each to a multiple of a repository equation; it also
checks that (A), (B) are E3 -/+ E4 of hp/src/model.rs. Exit status 1 if any check fails.

    python3 eqmap.py                 # all residuals 0
    python3 eqmap.py --mutate=eps    # eps_E = 1/n instead of 1/(n+1): must FAIL
    python3 eqmap.py --mutate=psi    # sig = sqrt(2) x Psi: must FAIL
"""
import sys
import sympy as sp

mut = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--mutate=")), None)
x, tau, n = sp.symbols("x tau n", positive=True)
y, sg, u, c = (sp.Function(s)(x, tau) for s in ("y", "sig", "u", "c"))
r2 = sp.sqrt(2)

# the map
Pi = y / (r2 * x)
Psi = sg / (r2 * x ** (1 if mut == "psi" else 2))
Om = u / x ** 2
f = c
epsE = 1 / n if mut == "eps" else 1 / (n + 1)

D1 = lambda e: sp.diff(e, tau) + x * sp.diff(e, x)
eeg = {
    "3a": x * sp.diff(Om, x) + x ** 2 * Om ** 2 - ((1 / epsE) * ((1 - 2 * epsE) * Pi ** 2 - Om)
                                                   + Om * (Pi ** 2 * x ** 2 + x ** 4 * Psi ** 2) + (1 / epsE - 2) * x ** 2 * Psi ** 2),
    "3b": sp.diff(sp.log(f), x) / x - Om,
    "3c": D1(Pi) - (f * x * sp.diff(Psi, x) + f * (1 / epsE + x ** 2 * Om) * Psi - Pi),
    "3d": D1(Psi) - (f * sp.diff(Pi, x) / x + f * Om * Pi - 2 * Psi),
}
P = (y ** 2 + sg ** 2) / 2
repo = {
    "E2": x * sp.diff(u, x) - (n - 1 + u) * (P - u),
    "E1": x * sp.diff(c, x) - c * u,
    "A": x ** 2 * sp.diff(y, x) + x * sp.diff(y, tau) - x * c * sp.diff(sg, x) - c * (n - 1 + u) * sg,
    "B": x ** 2 * sp.diff(sg, x) + x * sp.diff(sg, tau) - x * c * sp.diff(y, x) - c * (u - 1) * y,
}
# EEG eq = factor * repo eq (factor read off by hand from the substitution; the check is that the difference is 0)
pairs = [("3a", "E2", 1 / x ** 2), ("3b", "E1", 1 / (x ** 2 * c)), ("3c", "A", 1 / (r2 * x ** 2)),
         ("3d", "B", 1 / (r2 * x ** 3))]
ok = True
for e, r, fac in pairs:
    res = sp.simplify(sp.expand(eeg[e] - fac * repo[r]))
    print(f"EEG ({e}) - {sp.sstr(fac)} * repo ({r}) = {res}")
    ok &= res == 0

# (A), (B) against model.rs's E3, E4 written in p, m, b, c (d/dtau = omega d/dtheta), G = (n-1) c (b-1)
p, m, b = (sp.Function(s)(x, tau) for s in ("p", "m", "b"))
G = (n - 1) * c * (b - 1)
E3 = x * (x - c) * sp.diff(p, x) + x * sp.diff(p, tau) - (G * p - c * p + n / 2 * c * (p + m))
E4 = x * (x + c) * sp.diff(m, x) + x * sp.diff(m, tau) - (-G * m + c * m - n / 2 * c * (p + m))
sub = {p: (sg + y) / 2, m: (sg - y) / 2, b: 1 + u / (n - 1)}
for name, comb, r in (("E3 - E4", E3 - E4, "A"), ("E3 + E4", E3 + E4, "B")):
    res = sp.simplify(sp.expand(comb.subs(sub).doit() - repo[r]))
    print(f"model.rs {name} - ({r}) = {res}")
    ok &= res == 0
Eb = x * sp.diff(b, x) - (b * (p ** 2 + m ** 2) - (n - 1) * b * (b - 1))
res = sp.simplify(sp.expand((n - 1) * Eb.subs(sub).doit() - repo["E2"]))
print(f"model.rs b equation * (n-1) - (E2) = {res}")
ok &= res == 0
print("PASS" if ok else "FAIL", f"(mutation: {mut})" if mut else "")
sys.exit(0 if ok else 1)

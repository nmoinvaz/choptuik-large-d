#!/usr/bin/env python3
"""Rozali and Way's exact oscillaton (JHEP 11 (2018) 106, arXiv:1807.10283, Sec. 2.3, Eq. (10)) and where it diverges.

Their leading-order equations (Sec. 2.2, Eqs. (9b)-(9c); as in solver/inner_pert/rw_map.py), with c^2 = 1 - delta_0:
    d_r phi = r (1-delta)^(-1/2) d_tau[(1-delta)^(-1/2) d_tau phi],     d_r delta = -2 r (d_tau phi)^2,
and the solution as transcribed from their Eq. (10):
    phi_0 = artanh(eps e^{-w^2 r^2/2} sin(w tau - tau0)),
    delta_0 = eps^2 e^{-w^2 r^2} cos^2(w tau - tau0) / (1 - eps^2 e^{-w^2 r^2} sin^2(w tau - tau0)).
The script checks the transcription (residuals 0, for the stated w-dependence or with w = 1), then shows that
    c^2 = 1 - delta_0 = (1 - eps^2 e^{-w^2 r^2}) / (1 - eps^2 e^{-w^2 r^2} sin^2(...)),
which vanishes at r_div (eps^2 e^{-w^2 r_div^2} = 1) for every tau, while d_tau phi stays finite there: for eps > 1 the
divergence of A_0 = r^2 (d_tau phi)^2/(1 - delta_0) is a STATIC front (fixed r) at which the lapse vanishes and
Y = 2 phi_tau / c diverges. Compare layer.py: a moving front is resolved (Y finite, c -> 0 behind it); the layer width
and the compactness behind the front scale with 1/|front speed|, so the static case is the degenerate one.

    python3 rw_oscillaton.py
    python3 rw_oscillaton.py --mutate=sign    # delta_0 with +eps^2 e^{..} sin^2 in the denominator: must FAIL
"""
import sys
import sympy as sp

mut = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--mutate=")), None)
r, tau, eps, w, t0 = sp.symbols("r tau epsilon omega tau0", positive=True)
ok = True
for wv in (w, 1):
    E = eps * sp.exp(-wv ** 2 * r ** 2 / 2)
    ph = sp.atanh(E * sp.sin(wv * tau - t0))
    den = 1 + (E * sp.sin(wv * tau - t0)) ** 2 if mut == "sign" else 1 - (E * sp.sin(wv * tau - t0)) ** 2
    de = E ** 2 * sp.cos(wv * tau - t0) ** 2 / den
    c = sp.sqrt(1 - de)
    R1 = sp.diff(ph, r) - r / c * sp.diff(sp.diff(ph, tau) / c, tau)
    R2 = sp.diff(de, r) + 2 * r * sp.diff(ph, tau) ** 2
    vals = []
    for (rv, tv, ev, wn) in [tuple(sp.Rational(v) for v in pt) for pt in
                             (("0.7", "0.3", "0.6", "1.3"), ("1.1", "2.0", "0.9", "0.8"), ("0.4", "1.0", "1.7", "1.1"))]:
        sb = {r: rv, tau: tv, eps: ev, t0: sp.Rational(1, 5)}
        if wv == w:
            sb[w] = wn
        vals.append(max(abs(complex(R1.subs(sb).evalf(60))), abs(complex(R2.subs(sb).evalf(60)))))
    print(f"omega {'symbolic' if wv == w else '= 1'}: max |residual| of (9b), (9c) at three points = {max(vals):.2e}")
    if wv == w:
        okw = max(vals) < 1e-40
    else:
        ok1 = max(vals) < 1e-40
ok &= okw or ok1
E = eps * sp.exp(-w ** 2 * r ** 2 / 2)
S2 = sp.sin(w * tau - t0) ** 2
c2 = sp.simplify(1 - E ** 2 * (1 - S2) / (1 - E ** 2 * S2))
print("c^2 = 1 - delta_0 =", sp.factor(c2))
rdiv = sp.sqrt(2 * sp.log(eps)) / w
at = sp.simplify(c2.subs(r, rdiv))
print("c^2 at r_div = sqrt(2 ln eps)/omega:", at, " (for every tau)")
ok &= at == 0
phit = sp.simplify(sp.diff(sp.atanh(E * sp.sin(w * tau - t0)), tau).subs(r, rdiv))
print("d_tau phi at r_div:", phit, " (finite where cos != 0): Y = 2 phi_tau / c diverges there")
print("PASS" if ok else "FAIL", f"(mutation: {mut})" if mut else "")
sys.exit(0 if ok else 1)

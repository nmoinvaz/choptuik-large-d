"""Which late-order basis makes the fitted constant window-independent? (order eps^3, harmonic 1)"""
import mpmath as mp
from amp_flint import Amp, to_mp
A = Amp(1.0, 3, 220, jlo=40)
A.P = {}
for k in range(1, 4):
    if k % 2 == 0: A.pass_even(k)
    else:
        A.pass_odd(k); A.P[('d', k)] = A.poly(A.d[k]); A.P[('s', k)] = A.poly(A.s[k])
# get the raw data y_j (order 1 and 3) over the full window via S_at internals: monkeypatch qr_solve to capture y
cap = {}
orig = mp.qr_solve
def grab(Am, y):
    cap['y'] = [y[i] for i in range(y.rows)]
    return orig(Am, y)
mp.qr_solve = grab
A.S_at(1, 1); y1 = cap['y']
A.S_at(1, 3); y3 = cap['y']
mp.qr_solve = orig
S1 = y1[-1]
js = list(range(40, 221))
w = A.wmp; g0 = 1j * w + mp.mpf(1) / 2
def col(kind, j, m, sh=0):
    if kind == "g":
        t = mp.mpc(1)
        for l in range(1, m + 1): t /= (j + g0 - l)
        return t
    if kind == "f":
        gs = g0 + sh * 1j * w
        t = mp.exp(mp.loggamma(j + gs) - mp.loggamma(j + g0))
        for l in range(1, m + 1): t /= (j + gs - l)
        return t
    if kind == "L":   # log-type: psi(j+g0) Gamma(j+g0-m)/Gamma(j+g0)
        return mp.psi(0, j + g0) * col("g", j, m)
def fit(lo, hi, spec):
    idx = [i for i, j in enumerate(js) if lo <= j <= hi]
    rows = []
    for i in idx:
        j = js[i]; r = [mp.mpc(1)]
        for kind, m, sh in spec: r.append(col(kind, j, m, sh))
        rows.append(r)
    Am = mp.matrix(rows); sc = [max(abs(Am[i, c]) for i in range(Am.rows)) for c in range(Am.cols)]
    for c in range(Am.cols):
        for i in range(Am.rows): Am[i, c] /= sc[c]
    y = mp.matrix([y3[i] for i in idx])
    c, res = mp.qr_solve(Am, y)
    return c[0] / sc[0] / S1, res / mp.norm(y)
specs = {
 "G12": [("g", m, 0) for m in range(1, 13)],
 "G12+f2x4": [("g", m, 0) for m in range(1, 13)] + [("f", m, s) for s in (-2, 2) for m in range(1, 5)],
 "G12+f2x4+L4": [("g", m, 0) for m in range(1, 13)] + [("f", m, s) for s in (-2, 2) for m in range(1, 5)] + [("L", m, 0) for m in range(1, 5)],
 "G12+f2x4+f2m0": [("g", m, 0) for m in range(1, 13)] + [("f", m, s) for s in (-2, 2) for m in range(0, 5)],
 "G12+f2,4x4+f2m0": [("g", m, 0) for m in range(1, 13)] + [("f", m, s) for s in (-2, 2) for m in range(0, 5)] + [("f", m, s) for s in (-4, 4) for m in range(1, 5)],
}
for name, spec in specs.items():
    out = []
    for lo, hi in ((60, 160), (80, 200), (100, 220)):
        v, r = fit(lo, hi, spec)
        out.append(f"[{lo},{hi}] {mp.nstr(v, 13)} (res {mp.nstr(r, 2)})")
    print(name, " | ".join(out), flush=True)

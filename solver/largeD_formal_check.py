"""Plug the formal solution (P, M to order 2 plus S_3; c to G_3; B to B_4) into the exact equations for a fixed
F(theta), K_k = 0 and any omega0 (higher w_k = 0). The scaled residual should fall like eps^3 if the expansion is
right and the family is free in (F, omega)."""
import json, os, sys
import numpy as np
import sympy as sp
sys.argv = ["x", "3"]
src = open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "largeD_expand.py")).read()
exec(compile(src.replace('print(f"order', 'pass  # print(f"order').replace('print(f"   ', 'pass  # print(f"   '), "expand", "exec"))
NDER = 12
derivs = [sp.Derivative(F, (th, j)) for j in range(NDER, 0, -1)]
Fd = sp.symbols(f"F0:{NDER + 1}")
rep = {dd: Fd[j] for dd, j in zip(derivs, range(NDER, 0, -1))}
def lam(e):
    e = sp.expand(sp.sympify(e).subs({Kf[j]: 0 for j in range(len(Kf))}).doit().subs({w[j]: 0 for j in range(1, len(w))}))
    e = e.subs(rep).subs(F, Fd[0])
    return sp.lambdify((x, w[0]) + Fd, e, "numpy"), sp.lambdify((x, w[0]) + Fd, sp.diff(e, x), "numpy")
fields = {f"S{k}": lam(S[k]) for k in range(1, 4)}
fields.update({f"D{k}": lam(D[k]) for k in range(0, 3)})
fields.update({f"G{k}": lam(G[k]) for k in range(1, 4)})
fields.update({f"B{k}": lam(Bs[k]) for k in range(2, 5)})
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hp"))
from states import path as state_path   # hp/data/states, see hp/states.py
d = np.load(state_path("up13_D60.0020"), allow_pickle=True)
meta = json.loads(str(d["meta"])); M = meta["M"]; n0 = meta["D"] - 2
p1 = d["zh"][:M] + d["zl"][:M]
full = np.r_[p1, -p1] * np.sqrt(n0)
Nt = len(full); kk = np.fft.fftfreq(Nt, 1 / Nt)
Fh = np.fft.fft(full)
Fder = [np.real(np.fft.ifft(Fh * (1j * kk) ** j)) for j in range(NDER + 1)]
dth = lambda f: np.real(np.fft.ifft(np.fft.fft(f) * 1j * kk))
def ev(name, xv, wv, der=False):
    f = fields[name][1 if der else 0]
    v = f(xv, wv, *Fder)
    return np.broadcast_to(v, (Nt,)).astype(float)
for wv in (2 * np.pi / 1.6438937, 2 * np.pi / 1.4):
    print(f"omega0 = {wv:.5f}  (Delta {2*np.pi/wv:.4f})")
    for n in (50, 100, 200, 400, 800):
        e = 1.0 / n
        worst = np.zeros(4)
        for xv in (0.2, 0.5, 0.8, 0.95):
            Sv = sum(e ** k * ev(f"S{k}", xv, wv) for k in range(1, 4)); Sx = sum(e ** k * ev(f"S{k}", xv, wv, True) for k in range(1, 4))
            Dv = sum(e ** k * ev(f"D{k}", xv, wv) for k in range(0, 3)); Dx = sum(e ** k * ev(f"D{k}", xv, wv, True) for k in range(0, 3))
            c = 1 - sum(e ** k * ev(f"G{k}", xv, wv) for k in range(1, 4)); cx = -sum(e ** k * ev(f"G{k}", xv, wv, True) for k in range(1, 4))
            B = sum(e ** k * ev(f"B{k}", xv, wv) for k in range(2, 5)); Bx = sum(e ** k * ev(f"B{k}", xv, wv, True) for k in range(2, 5))
            P, Mm = (Sv + Dv) / 2, (Sv - Dv) / 2
            Px, Mx = (Sx + Dx) / 2, (Sx - Dx) / 2
            p, m, px, mx = np.sqrt(e) * P, np.sqrt(e) * Mm, np.sqrt(e) * Px, np.sqrt(e) * Mx
            nn = 1 / e
            Gx = (nn - 1) * c * B
            E1 = xv * cx - Gx
            E2 = xv * Bx - (1 + B) * (p ** 2 + m ** 2) + (nn - 1) * (1 + B) * B
            E3 = xv * (xv - c) * px + wv * xv * dth(p) - Gx * p + c * p - nn / 2 * c * (p + m)
            E4 = xv * (xv + c) * mx + wv * xv * dth(m) + Gx * m - c * m + nn / 2 * c * (p + m)
            worst = np.maximum(worst, [np.abs(E1).max() / e, np.abs(E2).max() / e, np.abs(E3).max() / np.sqrt(e), np.abs(E4).max() / np.sqrt(e)])
        print(f"   n {n:4d}: relative residuals E1..E4 " + " ".join(f"{v:.2e}" for v in worst) + "   x n^3: " + " ".join(f"{v * n**3:.1f}" for v in worst), flush=True)

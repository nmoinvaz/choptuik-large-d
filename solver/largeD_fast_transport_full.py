"""fast transport check with the O(eps) background (b0 = 1 + 2 eps^2 x^2 F^2, G0 = (n-1) c0 (b0-1),
p0 = sqrt(eps) x F = -m0) and the linearized db, dc equations included."""
import os
import numpy as np
from scipy.integrate import solve_ivp
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import largeD_fast_transport as T
NT, th, w, Ff, dPhi, dth, A0 = T.NT, T.th, T.w, T.Ff, T.dPhi, T.dth, T.A0

def exact_full(n, x1, x2, full=True):
    eps = 1 / n; h = np.sqrt(eps)
    Fv = Ff(th); F2 = Fv ** 2
    def rhs(x, Y):
        a, bm, be, ga = Y[:NT], Y[NT:2*NT], Y[2*NT:3*NT], Y[3*NT:]
        c0 = 1 - eps * F2 * (1 - x * x)
        b0 = 1 + (2 * eps**2 * x * x * F2 if full else 0)
        G0 = (n - 1) * c0 * (b0 - 1)
        p0 = h * x * Fv if full else 0 * Fv; m0 = -p0; p0x = h * Fv if full else 0 * Fv; m0x = -p0x
        if not full:
            be = 0 * be; ga = 0 * ga
        dG = (n - 1) * (ga * (b0 - 1) + c0 * be)
        q = n * dPhi(x)
        # rescaled: d(field)/dx = e^{n Phi}(d delta/dx) + q * field
        ga_x = (n - 1) * (ga * (b0 - 1) + c0 * be) / x + q * ga
        be_x = (be * (p0**2 + m0**2) + 2 * b0 * (p0 * a + m0 * bm) - (n - 1) * (2 * b0 - 1) * be) / x + q * be
        a_x = (x * ga * p0x - w * x * dth(a) + dG * p0 + G0 * a - ga * p0 - c0 * a
               + n / 2 * (ga * (p0 + m0) + c0 * (a + bm))) / (x * (x - c0)) + q * a
        b_x = (-x * ga * m0x - w * x * dth(bm) - dG * m0 - G0 * bm + ga * m0 + c0 * bm
               - n / 2 * (ga * (p0 + m0) + c0 * (a + bm))) / (x * (x + c0)) + q * bm
        return np.r_[a_x, b_x, be_x, ga_x]
    a1 = A0(th).astype(complex); b1 = a1 * (1 - x1) / (1 + x1)
    beta = -2 * (1 - x1**2) * Fv * (a1 - b1) / x1 * eps**1.5
    Y0 = np.r_[a1, b1, beta, -beta * (1 - x1**2)]
    sol = solve_ivp(rhs, (x1, x2), Y0, method="DOP853", rtol=1e-11, atol=1e-14)
    return sol.y[:NT, -1]

x1, x2 = 0.9, 0.72
At = T.transported(x1, x2, True)
for n in (100, 200, 400, 800, 1600):
    a = exact_full(n, x1, x2, True)
    err = np.abs(a - At).max() / np.abs(At).max()
    print(f"full background + db, dc: n {n:5d}  rel err {err:.3e}  n*err {n*err:.2f}", flush=True)

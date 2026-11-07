"""Pilot: classify the smooth branch through P_s (BCG's ODE) by the first line it reaches, D_W = 0 or D_Z = 0.

Double precision (scipy DOP853) on the psi-system dW/dpsi = N_W D_Z, dZ/dpsi = N_Z D_W, started on the analytic
invariant manifold of the slow eigendirection (Taylor series of order NT about P_s). Prints, on a grid of r in
(r_3, r_4), what each half-branch reaches. BCG (arXiv:2301.10101 Sec. 3) state: at r_3 + eps the right solution meets
D_W = 0, at r_4 - eps it meets D_Z = 0.
"""
import sys
import numpy as np
from scipy.integrate import solve_ivp

gam = float(sys.argv[1]) if len(sys.argv) > 1 else 1.4
al = (gam - 1)/2


def NW(r, W, Z): return -(r + ((1 + 2*al)*W + (1 - al)*Z)/2)*W + al/2*Z**2
def NZ(r, W, Z): return -(r + ((1 - al)*W + (1 + 2*al)*Z)/2)*Z + al/2*W**2
def DW(W, Z): return 1 + (W + Z + al*(W - Z))/2
def DZ(W, Z): return 1 + (W + Z - al*(W - Z))/2


def P_s(r):
    g = gam
    R1 = np.sqrt(g**2*(r - 3)**2 - 2*g*(3*r**2 - 6*r + 7) + (9*r**2 - 14*r + 9))
    W0 = (g**2*r + (g + 1)*R1 - 3*g**2 - 2*g*r + 10*g - 3*r - 3)/(4*(g - 1)**2)
    Z0 = (g**2*r + (g - 3)*R1 - 3*g**2 - 6*g*r + 6*g + 9*r - 7)/(4*(g - 1)**2)
    return np.array([W0, Z0])


def F(r, X):
    W, Z = X
    return np.array([NW(r, W, Z)*DZ(W, Z), NZ(r, W, Z)*DW(W, Z)])


def jac(r, X, h=1e-7):
    J = np.zeros((2, 2))
    for k in range(2):
        e = np.zeros(2); e[k] = h
        J[:, k] = (F(r, X + e) - F(r, X - e))/(2*h)
    return J


def manifold(r, NT=40):
    """Coefficients a_j of X(u) = P_s + sum a_j u^j with F(X(u)) = lam_a u X'(u) (slow eigenvalue lam_a)."""
    P = P_s(r)
    J = jac(r, P)
    ev, V = np.linalg.eig(J)
    ia = int(np.argmin(np.abs(ev)))
    la, lb = ev[ia].real, ev[1 - ia].real
    a = [P, V[:, ia].real/np.linalg.norm(V[:, ia].real)]
    # order-j coefficient of F(P + sum a_i u^i) by exact polynomial arithmetic in u (truncated)
    def series_F(a, N):
        W = np.zeros(N + 1); Z = np.zeros(N + 1)
        for i, c in enumerate(a[:N + 1]):
            W[i], Z[i] = c
        mul = lambda x, y: np.convolve(x, y)[:N + 1]
        one = np.zeros(N + 1); one[0] = 1
        nw = -(r*one + ((1 + 2*al)*W + (1 - al)*Z)/2)
        nw = mul(nw, W) + al/2*mul(Z, Z)
        nz = -(r*one + ((1 - al)*W + (1 + 2*al)*Z)/2)
        nz = mul(nz, Z) + al/2*mul(W, W)
        dw = one + (W + Z + al*(W - Z))/2
        dz = one + (W + Z - al*(W - Z))/2
        return np.array([mul(nw, dz), mul(nz, dw)])
    for j in range(2, NT + 1):
        a.append(np.zeros(2))
        S = series_F(a, j)[:, j]          # includes J a_j = 0 term, so S is R_j
        a[j] = np.linalg.solve(j*la*np.eye(2) - J, S)
    return a, la, lb


def run(r, side, u0=None, NT=40, smax=1e6):
    a, la, lb = manifold(r, NT)
    if u0 is None:
        # radius estimate from coefficient decay
        nr = [np.linalg.norm(c) for c in a[1:]]
        rad = (nr[-1]/nr[len(nr)//2])**(-1.0/(len(nr) - len(nr)//2 - 1)) if nr[-1] > 0 else 1.0
        u0 = 0.3*rad
    u = side*u0
    X0 = sum(c*u**j for j, c in enumerate(a))
    dz0, dw0 = np.sign(DZ(*X0)), np.sign(DW(*X0))
    ev_w = lambda t, X: DW(*X); ev_w.terminal = True
    ev_z = lambda t, X: DZ(*X); ev_z.terminal = True
    ev_big = lambda t, X: np.abs(X).max() - smax; ev_big.terminal = True
    direction = 1.0 if la > 0 else -1.0
    sol = solve_ivp(lambda t, X: direction*F(r, X), (0, 1e4), X0, method="DOP853", rtol=1e-13, atol=1e-15,
                    events=(ev_w, ev_z, ev_big))
    hit = [len(e) > 0 for e in sol.t_events]
    tag = "D_W" if hit[0] else "D_Z" if hit[1] else "big" if hit[2] else "none"
    Xe = sol.y[:, -1]
    return tag, Xe, la, lb, u0


if __name__ == "__main__":
    import mpmath as mp
    r3 = 1.070374378675834; r4 = 1.094974963188878   # kres_planted.log at gamma 7/5
    if abs(gam - 1.4) > 1e-12:
        print("grid only meaningful at 7/5 here"); sys.exit(2)
    for side in (+1, -1):
        print(f"side {side:+d}")
        for r in np.linspace(r3 + 1e-4, r4 - 1e-4, 13):
            tag, Xe, la, lb, u0 = run(r, side)
            U, S = (Xe[0] + Xe[1])/2, (Xe[0] - Xe[1])/2
            print(f"  r {r:.6f} k {lb/la:.4f} -> {tag:4s} end (W,Z) = ({Xe[0]: .4e}, {Xe[1]: .4e})  U {U: .3e} S {S: .3e}")

"""Multiprecision shooting for r^(n)(gamma): the smooth branch through P_s (analytic invariant manifold of the slow
eigendirection, order-NT series) is integrated by an order-p Taylor method on the psi-system and classified by the
first line it reaches, D_W = 0 or D_Z = 0 (BCG, arXiv:2301.10101 Sec. 3). Bisection on the classification.
Usage: shoot_mp.py gamma rlo rhi dps p NT tol   (gamma as a fraction, e.g. 7/5)"""
import sys, time
import mpmath as mp
from fractions import Fraction

g_s, rlo_s, rhi_s, dps, p, NT, tol_s = sys.argv[1:8]
mp.mp.dps = int(dps); p = int(p); NT = int(NT)
gam = mp.mpf(Fraction(g_s).numerator)/Fraction(g_s).denominator
al = (gam - 1)/2
aW, bW = (1 + 2*al)/2, (1 - al)/2       # N_W = -(r + aW W + bW Z) W + al/2 Z^2
aZ, bZ = (1 - al)/2, (1 + 2*al)/2       # N_Z = -(r + aZ W + bZ Z) Z + al/2 W^2
cWw, cWz = (1 + al)/2, (1 - al)/2       # D_W = 1 + cWw W + cWz Z
cZw, cZz = (1 - al)/2, (1 + al)/2       # D_Z = 1 + cZw W + cZz Z


def F(r, W, Z):
    nw = -(r + aW*W + bW*Z)*W + al/2*Z**2
    nz = -(r + aZ*W + bZ*Z)*Z + al/2*W**2
    return nw*(1 + cZw*W + cZz*Z), nz*(1 + cWw*W + cWz*Z)


def series_coeffs(r, W, Z, sgn, K, fixed=None):
    """Taylor coefficients 0..K of the psi-flow (times sgn) through (W, Z); fixed: preset leading coefficients."""
    Ws, Zs = [W], [Z]
    LW, LZ, NWs, NZs, DWs, DZs = [], [], [], [], [], []
    for k in range(K):
        LW.append((r if k == 0 else 0) + aW*Ws[k] + bW*Zs[k])
        LZ.append((r if k == 0 else 0) + aZ*Ws[k] + bZ*Zs[k])
        DWs.append((1 if k == 0 else 0) + cWw*Ws[k] + cWz*Zs[k])
        DZs.append((1 if k == 0 else 0) + cZw*Ws[k] + cZz*Zs[k])
        NWs.append(-mp.fsum(LW[i]*Ws[k - i] for i in range(k + 1)) + al/2*mp.fsum(Zs[i]*Zs[k - i] for i in range(k + 1)))
        NZs.append(-mp.fsum(LZ[i]*Zs[k - i] for i in range(k + 1)) + al/2*mp.fsum(Ws[i]*Ws[k - i] for i in range(k + 1)))
        f1 = mp.fsum(NWs[i]*DZs[k - i] for i in range(k + 1))
        f2 = mp.fsum(NZs[i]*DWs[k - i] for i in range(k + 1))
        Ws.append(sgn*f1/(k + 1)); Zs.append(sgn*f2/(k + 1))
    return Ws, Zs


def P_s(r):
    g = gam
    R1 = mp.sqrt(g**2*(r - 3)**2 - 2*g*(3*r**2 - 6*r + 7) + (9*r**2 - 14*r + 9))
    W0 = (g**2*r + (g + 1)*R1 - 3*g**2 - 2*g*r + 10*g - 3*r - 3)/(4*(g - 1)**2)
    Z0 = (g**2*r + (g - 3)*R1 - 3*g**2 - 6*g*r + 6*g + 9*r - 7)/(4*(g - 1)**2)
    return W0, Z0


def manifold(r):
    W0, Z0 = P_s(r)
    J = mp.matrix(2, 2)
    for k, (dW, dZ) in enumerate(((1, 0), (0, 1))):
        f = lambda e: F(r, W0 + e*dW, Z0 + e*dZ)
        d = mp.diff(lambda e: f(e)[0], 0), mp.diff(lambda e: f(e)[1], 0)
        J[0, k], J[1, k] = d
    E, V = mp.eig(J)
    ia = 0 if abs(E[0]) < abs(E[1]) else 1
    la, lb = mp.re(E[ia]), mp.re(E[1 - ia])
    v = [mp.re(V[0, ia]), mp.re(V[1, ia])]
    nv = mp.sqrt(v[0]**2 + v[1]**2)
    if v[0] - v[1] < 0:      # orient: u > 0 is the half-branch on which S = (W - Z)/2 grows (toward P_0)
        nv = -nv
    a = [(W0, Z0), (v[0]/nv, v[1]/nv)]
    # order-j terms of F(P + sum a_i u^i): exact via truncated polynomial products
    for j in range(2, NT + 1):
        Wc = [c[0] for c in a] + [0]; Zc = [c[1] for c in a] + [0]
        def mul(x, y):
            return [mp.fsum(x[i]*y[k - i] for i in range(k + 1)) for k in range(j + 1)]
        one = [1] + [0]*j
        LWc = [r*one[k] + aW*Wc[k] + bW*Zc[k] for k in range(j + 1)]
        LZc = [r*one[k] + aZ*Wc[k] + bZ*Zc[k] for k in range(j + 1)]
        nw = [-x + al/2*y for x, y in zip(mul(LWc, Wc), mul(Zc, Zc))]
        nz = [-x + al/2*y for x, y in zip(mul(LZc, Zc), mul(Wc, Wc))]
        dw = [one[k] + cWw*Wc[k] + cWz*Zc[k] for k in range(j + 1)]
        dz = [one[k] + cZw*Wc[k] + cZz*Zc[k] for k in range(j + 1)]
        R = mp.matrix([mul(nw, dz)[j], mul(nz, dw)[j]])
        x = mp.lu_solve(j*la*mp.eye(2) - J, R)
        a.append((x[0], x[1]))
    return a, la, lb


def run(r, side=1, smax=mp.mpf(10)**30, maxsteps=20000):
    a, la, lb = manifold(r)
    tol = mp.mpf(10)**(-mp.mp.dps + 4)
    nrm = [abs(c[0]) + abs(c[1]) for c in a]
    rad = min((nrm[j]/nrm[NT])**(mp.mpf(1)/(NT - j)) for j in (NT//2, 3*NT//4))
    u0 = rad*(tol/nrm[NT])**(mp.mpf(1)/NT)/rad if nrm[NT] > 0 else mp.mpf("0.1")
    u0 = min(u0, rad/3)
    u = side*u0
    W = mp.fsum(c[0]*u**j for j, c in enumerate(a)); Z = mp.fsum(c[1]*u**j for j, c in enumerate(a))
    sgn = 1 if la > 0 else -1
    dw0 = mp.sign(1 + cWw*W + cWz*Z); dz0 = mp.sign(1 + cZw*W + cZz*Z)
    for step in range(maxsteps):
        Ws, Zs = series_coeffs(r, W, Z, sgn, p)
        m1 = max(abs(Ws[p]) + abs(Zs[p]), mp.mpf(10)**(-mp.mp.dps*2))
        m2 = max(abs(Ws[p - 1]) + abs(Zs[p - 1]), mp.mpf(10)**(-mp.mp.dps*2))
        scale = 1 + abs(W) + abs(Z)
        h = min((tol*scale/m1)**(mp.mpf(1)/p), (tol*scale/m2)**(mp.mpf(1)/(p - 1)))
        while True:
            Wn = mp.polyval(Ws[::-1], h); Zn = mp.polyval(Zs[::-1], h)
            dw = 1 + cWw*Wn + cWz*Zn; dz = 1 + cZw*Wn + cZz*Zn
            cw, cz = mp.sign(dw) != dw0, mp.sign(dz) != dz0
            if cw and cz:
                h /= 2; continue
            break
        if cw: return "D_W", step
        if cz: return "D_Z", step
        W, Z = Wn, Zn
        if abs(W) + abs(Z) > smax: return "big", step
        if abs(W) + abs(Z) < mp.mpf(10)**-12: return "P_inf", step
    return "none", maxsteps


if __name__ == "__main__":
    lo, hi = mp.mpf(rlo_s), mp.mpf(rhi_s)
    tol_r = mp.mpf(tol_s)
    t0 = time.time()
    tlo, s1 = run(lo); thi, s2 = run(hi)
    print(f"gamma {g_s} dps {dps} p {p} NT {NT}: {tlo} at lo ({s1} steps), {thi} at hi ({s2} steps) "
          f"[{time.time()-t0:.1f} s]", flush=True)
    assert tlo != thi and "big" not in (tlo, thi) and "none" not in (tlo, thi)
    it = 0
    while hi - lo > tol_r:
        mid = (lo + hi)/2
        t, s = run(mid)
        it += 1
        if t == tlo: lo = mid
        elif t == thi: hi = mid
        else:
            print(f"  undecided {t} at {mid}"); break
        if it % 10 == 0:
            print(f"  it {it}: [{mp.nstr(lo, 34)}, {mp.nstr(hi, 34)}] steps {s} [{time.time()-t0:.0f} s]", flush=True)
    print(f"r^(n)({g_s}) in [{mp.nstr(lo, 40)}, {mp.nstr(hi, 40)}]  width {mp.nstr(hi-lo, 3)}  "
          f"{it} bisections [{time.time()-t0:.0f} s]")

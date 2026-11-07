"""Planted control: rederive BCG's eigenvalue ratio k(r) at the sonic point P_s from their ODE.

Source of the ODE and the planted values: Buckmaster, Cao-Labora, Gomez-Serrano, arXiv:2301.10101 (review),
eqs. for N_W, N_Z, D_W, D_Z, and k(r) = (r-2-sqrt(2r-2))/(r-2+sqrt(2r-2)) at gamma = 5/3, with
r_2 = 11-3 sqrt 11, r_3 = 6-2 sqrt 6, r_4 = (43-5 sqrt 43)/9; r*(gamma) from arXiv:2208.09445 eq. (1.16).
--mutate=alpha uses alpha = (gamma-1)/3; the planted checks must then fail (exit 1).
"""
import sys
import mpmath as mp

mp.mp.dps = 50
MUT = "--mutate=alpha" in sys.argv


def system(gam):
    al = (gam - 1) / (3 if MUT else 2)

    def NW(r, W, Z): return -(r + (( 1 + 2*al)*W + (1 - al)*Z)/2)*W + al/2*Z**2
    def NZ(r, W, Z): return -(r + (( 1 - al)*W + (1 + 2*al)*Z)/2)*Z + al/2*W**2
    def DW(W, Z): return 1 + (W + Z + al*(W - Z))/2
    def DZ(W, Z): return 1 + (W + Z - al*(W - Z))/2
    return al, NW, NZ, DW, DZ


def sonic_points(gam, r):
    """Both intersections of D_Z = 0 and N_Z = 0 (D_Z linear: eliminate W)."""
    al, NW, NZ, DW, DZ = system(gam)
    # D_Z = 0  ->  W(1-al) + Z(1+al) = -2  ->  Z = (-2 - (1-al) W)/(1+al)
    Zof = lambda W: (-2 - (1 - al)*W)/(1 + al)
    f = lambda W: NZ(r, W, Zof(W))
    # f is quadratic in W: fit three points exactly
    w0, w1, w2 = mp.mpf(0), mp.mpf(1), mp.mpf(-1)
    f0, f1, f2 = f(w0), f(w1), f(w2)
    a = (f1 + f2 - 2*f0)/2; b = (f1 - f2)/2; c = f0
    disc = mp.sqrt(b*b - 4*a*c)
    out = []
    for W in ((-b + disc)/(2*a), (-b - disc)/(2*a)):
        out.append((W, Zof(W)))
    return out


def kratio(gam, r, which):
    al, NW, NZ, DW, DZ = system(gam)
    W, Z = sonic_points(gam, r)[which]
    F1 = lambda W_, Z_: NW(r, W_, Z_)*DZ(W_, Z_)
    F2 = lambda W_, Z_: NZ(r, W_, Z_)*DW(W_, Z_)
    J = mp.matrix([[mp.diff(lambda w: F1(w, Z), W), mp.diff(lambda z: F1(W, z), Z)],
                   [mp.diff(lambda w: F2(w, Z), W), mp.diff(lambda z: F2(W, z), Z)]])
    ev = mp.eig(J)[0]
    l1, l2 = sorted(ev, key=lambda e: abs(e))
    return l2/l1, (W, Z), (l1, l2)


def k_closed_53(r):
    s = mp.sqrt(2*r - 2)
    return (r - 2 - s)/(r - 2 + s)


def rstar_116(g):
    g = mp.mpf(g)
    if g < mp.mpf(5)/3:
        return 2/(mp.sqrt(2)*mp.sqrt(1/(g - 1)) + 1)**2 + 1
    return (3*g - 1)/(2 + mp.sqrt(3)*(g - 1))


def P_s_21(g, r):
    """BCG25 eq. (2.1), (2.3) as transcribed by the fetch."""
    R1 = mp.sqrt(g**2*(r - 3)**2 - 2*g*(3*r**2 - 6*r + 7) + (9*r**2 - 14*r + 9))
    W0 = (g**2*r + (g + 1)*R1 - 3*g**2 - 2*g*r + 10*g - 3*r - 3)/(4*(g - 1)**2)
    Z0 = (g**2*r + (g - 3)*R1 - 3*g**2 - 6*g*r + 6*g + 9*r - 7)/(4*(g - 1)**2)
    return W0, Z0


def which_21(gam, r):
    """Index of the sonic point that is BCG25's P_s, eq. (2.1)."""
    W0, Z0 = P_s_21(gam, r)
    P = sonic_points(gam, r)
    return min((0, 1), key=lambda w: abs(P[w][0] - W0) + abs(P[w][1] - Z0))


def r_of_k(gam, j, which, lo, hi):
    """Bisection on k(r) = j (k increasing on (1, r*), BCG25 Lemma 2.1); which=None: the (2.1) point."""
    def k(r):
        w = which_21(gam, r) if which is None else which
        return mp.re(kratio(gam, r, w)[0])
    for _ in range(190):
        mid = (lo + hi)/2
        lo, hi = (mid, hi) if k(mid) < j else (lo, mid)
    return (lo + hi)/2


fails = 0
g53 = mp.mpf(5)/3
print("gamma = 5/3: k(r) from the Jacobian vs the closed form")
for which in (0, 1):
    errs = []
    for r in (mp.mpf("1.02"), mp.mpf("1.08"), mp.mpf("1.12"), mp.mpf("1.2")):
        k, P, _ = kratio(g53, r, which)
        errs.append(abs(k - k_closed_53(r)))
    print(f"  sonic point #{which}: max |k - closed| = {mp.nstr(max(errs), 3)}")
which = min((0, 1), key=lambda w: abs(kratio(g53, mp.mpf("1.08"), w)[0] - k_closed_53(mp.mpf("1.08"))))
print(f"  using sonic point #{which}")
planted = {2: 11 - 3*mp.sqrt(11), 3: 6 - 2*mp.sqrt(6), 4: (43 - 5*mp.sqrt(43))/9}
for j, rj in planted.items():
    got = r_of_k(g53, j, None, mp.mpf("1.001"), mp.mpf("1.26"))
    d = abs(got - rj)
    ok = d < mp.mpf(10)**-30
    fails += not ok
    print(f"  r_{j}: computed {mp.nstr(got, 20)}  planted {mp.nstr(rj, 20)}  diff {mp.nstr(d, 3)}  {'ok' if ok else 'FAIL'}")

print("P_s against BCG25 (2.1) and r* against (1.16), several gamma")
for g in (mp.mpf("1.2"), mp.mpf(7)/5, g53, mp.mpf(2), mp.mpf(3)):
    rr = (1 + rstar_116(g))/2  # a point inside (1, r*)
    P = sonic_points(g, rr)
    W0, Z0 = P_s_21(g, rr)
    dP = min(abs(p[0] - W0) + abs(p[1] - Z0) for p in P)
    # r* = where k -> infinity: lambda_- -> 0; locate by k = 1e6
    w2 = min((0, 1), key=lambda w: abs(P[w][0] - W0) + abs(P[w][1] - Z0))
    kk = lambda r: mp.re(kratio(g, r, w2)[0])
    # bisection for k = 1e8 between 1+tiny and r*(1.16)+0.05
    lo, hi = mp.mpf(1) + mp.mpf(10)**-6, rstar_116(g) + mp.mpf("0.05")
    for _ in range(200):
        mid = (lo + hi)/2
        try:
            v = kk(mid)
            good = (v > 0 and v < 1e8 and mp.im(kratio(g, mid, w2)[0]) == 0)
        except Exception:
            good = False
        lo, hi = (mid, hi) if good else (lo, mid)
    ok1 = dP < mp.mpf(10)**-30
    ok2 = abs(lo - rstar_116(g)) < 1e-6
    fails += (not ok1) + (not ok2)
    print(f"  gamma {mp.nstr(g, 6)}: |P_s - (2.1)| = {mp.nstr(dP, 3)}; r*(k=1e8) {mp.nstr(lo, 12)} vs (1.16) "
          f"{mp.nstr(rstar_116(g), 12)}  {'ok' if ok1 and ok2 else 'FAIL'}")

print("gamma = 7/5 resonances r_j (k(r_j) = j), j = 2..12, and r*")
g75 = mp.mpf(7)/5
rs = []
for j in range(2, 13):
    rj = r_of_k(g75, j, None, mp.mpf(1) + mp.mpf(10)**-12, rstar_116(g75) - mp.mpf(10)**-12)
    rs.append(rj)
    print(f"  r_{j}(7/5) = {mp.nstr(rj, 25)}")
print(f"  r*(7/5) = {mp.nstr(rstar_116(g75), 25)}")
print(f"  BCG's Figure-1 value r = 1.079404 lies in (r_3, r_4) = ({mp.nstr(rs[1], 8)}, {mp.nstr(rs[2], 8)}): "
      f"{rs[1] < mp.mpf('1.079404') < rs[2]}")
print("PLANTED", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)

"""Our analogue of BCG's k(r): the local exponent ratio at the characteristic singular point, and the zeros and
poles of C(n,s) in omega. Same resonance finder for both problems (complex Newton on k(p) = j from a grid of starts).

Planted control: the finder applied to BCG's k(r) at gamma = 5/3 must return the real r_2, r_3, r_4 of
kres_planted.py. Ours: (a) the Frobenius exponents at x = 1 of the leading-order mode equation (math paper,
Prop. 2 proof), derived here with sympy, against the paper's {0, n/2 - 1 - s}; (b) C(n,s) of eq. (C) and K'(s) of
eq. (Kprime) on the real omega axis; control: the same evaluator must see the zero at omega = i (mu = 1) and the
pole at omega = -i(n/2 - 1)/mu. Exit 1 if any check fails.
"""
import sys
import mpmath as mp
import sympy as sp

mp.mp.dps = 40
fails = 0


def resonances(k, j, starts):
    """Complex roots of k(p) = j from the given starts, deduplicated."""
    roots = []
    for z0 in starts:
        try:
            z = mp.findroot(lambda z: k(z) - j, mp.mpc(z0), tol=mp.mpf(10)**-30, maxsteps=80)
        except Exception:
            continue
        if abs(k(z) - j) < mp.mpf(10)**-25 and all(abs(z - w) > 1e-12 for w in roots):
            roots.append(z)
    return roots


starts = [mp.mpc(a, b) for a in (0.2, 1.05, 1.2, 2, 4) for b in (-3, -0.5, 0.01, 0.5, 3)]

# planted: BCG gamma = 5/3, k(r) = (r-2-sqrt(2r-2))/(r-2+sqrt(2r-2)), principal sqrt
kB = lambda r: (r - 2 - mp.sqrt(2*r - 2))/(r - 2 + mp.sqrt(2*r - 2))
planted = {2: 11 - 3*mp.sqrt(11), 3: 6 - 2*mp.sqrt(6), 4: (43 - 5*mp.sqrt(43))/9}
print("planted: BCG k(r) = j, gamma 5/3")
for j, rj in planted.items():
    R = resonances(kB, j, starts)
    real = [z for z in R if abs(mp.im(z)) < 1e-25]
    ok = any(abs(z - rj) < 1e-25 for z in real)
    fails += not ok
    print(f"  j={j}: roots {[mp.nstr(z, 12) for z in R]}  real root = r_{j}: {ok}")

# ours (a): exponents at x = 1 of x^2(1-x^2)y'' + x(n-2-2(s+1)x^2)y' - (n-2+s(s+1)x^2)y = 0
x, t, n, s, rho = sp.symbols("x t n s rho")
P2 = x**2*(1 - x**2); P1 = x*(n - 2 - 2*(s + 1)*x**2); P0 = -(n - 2 + s*(s + 1)*x**2)
# regular singular point x = 1: indicial rho(rho-1) + p0 rho + q0 = 0 with p0 = lim (x-1) P1/P2, q0 = lim (x-1)^2 P0/P2
p0 = sp.limit((x - 1)*P1/P2, x, 1); q0 = sp.limit((x - 1)**2*P0/P2, x, 1)
expo = sp.solve(sp.Eq(rho*(rho - 1) + p0*rho + q0, 0), rho)
expo = [sp.simplify(e) for e in expo]
target = {sp.Integer(0), sp.simplify(n/2 - 1 - s)}
ok = set(expo) == target
fails += not ok
print(f"ours (a): exponents at x = 1: {expo}; paper's {{0, n/2-1-s}}: {ok}")
Lam = sp.lambdify((n, s), sp.simplify(n/2 - 1 - s), "mpmath")
for nn in (2, 18, 58, 198):
    for mu in (1, 3):
        kO = lambda w: Lam(nn, 1j*mu*w)
        R = []
        for j in range(0, 6):
            R += resonances(kO, j, [mp.mpc(a, b) for a in (0.5, 3.8, 10) for b in (-20, -2, 2, 20)])
        # omega = 0 is the static case, excluded by Prop. 2's hypothesis omega != 0
        real = [z for z in R if abs(mp.im(z)) < 1e-20 and abs(z) > 1e-20]
        dmin = min(abs(Lam(nn, 1j*mu*mp.mpf(2)*mp.pi/mp.mpf("1.64389375")) - j) for j in range(0, 400))
        print(f"  n={nn:3d} mu={mu}: roots of Lambda(omega)=j, j=0..5: {[mp.nstr(z, 6) for z in R[:3]]}... "
              f"real ones: {len(real)}; min_j |Lambda(omega_inf) - j| = {mp.nstr(dmin, 6)}")
        fails += len(real) > 0

# ours (b): C(n,s) and K'(s) on the real omega axis, and the control at the imaginary-axis zero and pole
C = lambda nn, s: mp.gamma((nn + 1)/mp.mpf(2))*mp.gamma(s + 1 - mp.mpf(nn)/2)*mp.rgamma((1 + s)/2)*mp.rgamma((2 + s)/2)
Kp = lambda s: 1/(mp.sqrt(2*mp.pi)*mp.gamma(1 + s))
print("ours (b): |C(n, i mu omega)| on omega in [0.05, 12], 2400 points")
for nn in (18, 58, 200):
    for mu in (1, 3):
        vals = [abs(C(nn, 1j*mu*w)) for w in mp.linspace(mp.mpf("0.05"), 12, 2400)]
        print(f"  n={nn:3d} mu={mu}: min |C| {mp.nstr(min(vals), 4)}  max |C| {mp.nstr(max(vals), 4)}")
        fails += not (min(vals) > 0 and mp.isfinite(max(vals)))
w_inf = 2*mp.pi/mp.mpf("1.64389375")
print(f"  |C(18, i omega_inf)| = {mp.nstr(abs(C(18, 1j*w_inf)), 4)} (paper: 0.0121)")
fails += abs(abs(C(18, 1j*w_inf)) - mp.mpf("0.0121")) > 5e-5
# n odd: Gamma(s+1-n/2) has its poles at half-integers, so s = -1 (omega = i, mu = 1) is a zero of 1/Gamma((1+s)/2).
# n even: the zeros at negative integer s cancel poles of Gamma(s+1-n/2); the poles s = n/2-1, ..., 0 remain.
zc = abs(C(19, mp.mpf(-1)))
pc = abs(C(18, mp.mpf(18)/2 - 1 + mp.mpf(10)**-20))   # s = n/2-1: Gamma(s+1-n/2) = Gamma(~0)
print(f"  control: |C(19, s=-1)| (omega = i): {mp.nstr(zc, 3)} (zero expected); |C(18, s=8+1e-20)| "
      f"(omega = -8i): {mp.nstr(pc, 3)} (pole expected)")
fails += not (zc < 1e-30 and pc > 1e15)
print("  K'(s) zeros: s = -1, -2, ... i.e. omega = i j/mu; |K'(i omega_inf)| =", mp.nstr(abs(Kp(1j*w_inf)), 6))
print("RESULT", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)

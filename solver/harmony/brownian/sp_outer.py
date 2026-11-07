"""Systematic 1/D outer expansion of the Sire-Chavanis scaling equation, SC02 Eq. (97).

  alpha S + x S' = S'' + (D+1)/x S' + S (x S' + D S)

With S = sigma/D, w = x^2/D, eps = 1/D it is

  eps (4 w s_ww + 4 s_w + 2 w s s_w) + 2 (1-w) s_w + s^2 - alpha s = 0 .

Order 0, alpha_0 = 2: sigma_0 = 2z/(1+(z-1)w), z = D S(0)/2 (a one-parameter family).
Order k: 2(1-w) s_k' + (2 s_0 - 2) s_k = R_k, solved by variation of parameters with the
homogeneous solution h = d sigma_0/dz ∝ (1-w)/(1+(z-1)w)^2, normalised by s_k(0) = 0.
A resonant term in R_k gives (1-w) log(1-w) at the turning point w = 1; its absence fixes
alpha_k(z). Then alpha_max = max_z alpha(z), order by order.

--mutate=diffusion doubles the eps*s_ww coefficient (the check against SC02 (112),(117),(119)
must then fail).
"""
import sys
import sympy as sp

mut = '--mutate=diffusion' in sys.argv
K = 3
for a in sys.argv[1:]:
    if a.startswith('--order='):
        K = int(a.split('=')[1])

w, z, eps = sp.symbols('w z epsilon', positive=True)
al = sp.symbols('a1:%d' % (K + 1))
s0 = 2*z/(1 + (z - 1)*w)
h = (1 - w)/(1 + (z - 1)*w)**2   # homogeneous solution, h(0) = 1

def full_residual(sig, alpha):
    c = 8 if mut else 4
    return (eps*(c*w*sp.diff(sig, w, 2) + 4*sp.diff(sig, w) + 2*w*sig*sp.diff(sig, w))
            + 2*(1 - w)*sp.diff(sig, w) + sig**2 - alpha*sig)

sig = [s0]
alph = [sp.Integer(2)]
for k in range(1, K + 1):
    ak = al[k - 1]
    sig_trial = sum(eps**j*sig[j] for j in range(k)) 
    alpha_trial = sum(eps**j*alph[j] for j in range(k)) + eps**k*ak
    # order-k part of the residual with s_k = 0; the linear operator acting on s_k is added separately
    res = sp.expand(full_residual(sig_trial, alpha_trial))
    Rk = -sp.simplify(res.coeff(eps, k))           # L s_k = Rk, L s = 2(1-w)s' + (2 s0 - 2) s
    # check operator: L s_k = d/dw-structure; integrand for variation of parameters
    # resonance: the integrand R (1+(z-1)w)^2 / (2 (1-w)^2) of variation of parameters has a
    # double pole at w = 1; its residue, -(1/2) d/dw [R (1+(z-1)w)^2] at w = 1, gives the log.
    g = Rk*(1 + (z - 1)*w)**2
    cond = sp.diff(g, w).subs(w, 1)
    sol = sp.solve(sp.Eq(sp.simplify(cond), 0), ak)
    assert len(sol) == 1, sol
    akz = sp.simplify(sol[0])
    alph.append(akz)
    print('alpha_%d(z) =' % k, akz, flush=True)
    if k == K:
        break
    Rk_s = Rk.subs(ak, akz)
    integrand = sp.together(Rk_s/(2*(1 - w)*h))
    try:
        I = sp.integrate(sp.apart(sp.factor(integrand), w), w)
    except sp.PolynomialError:
        I = sp.integrate(sp.expand(integrand), w)
    sk = sp.simplify(h*(I - I.subs(w, 0)))
    chk = sp.simplify(2*(1 - w)*sp.diff(sk, w) + (2*s0 - 2)*sk - Rk_s)
    assert chk == 0, chk
    # the solution must be free of log(1-w): check analyticity at w = 1
    assert not sk.has(sp.log(1 - w)) and not sk.has(sp.log(w - 1)), sk
    sig.append(sk)
    print('sigma_%d(w) =' % k, sk, flush=True)

# alpha_max order by order
zs = sp.symbols('z1:%d' % (K + 1))
zz = 4 + sum(eps**j*zs[j - 1] for j in range(1, K))
A = sum(eps**j*alph[j] for j in range(1, K + 1))
z0 = sp.solve(sp.diff(alph[1], z), z)
print('z0 from d alpha_1/dz = 0:', z0)
ser = sp.series(A.subs(z, zz), eps, 0, K + 1).removeO()
ser = sp.expand(ser)
# stationarity in z1..: d/dz_j
eqs = []
subsd = {}
coef = {j: sp.simplify(ser.coeff(eps, j)) for j in range(1, K + 1)}
for j in range(1, K):
    # z_j is fixed by stationarity at order j+2: coefficient is quadratic in z_j
    cj = coef[j + 2].subs(subsd) if j + 2 <= K else None
    if cj is None:
        break
    zj = sp.solve(sp.diff(cj, zs[j - 1]), zs[j - 1])
    subsd[zs[j - 1]] = sp.simplify(zj[0])
    print('z_%d =' % j, sp.nsimplify(subsd[zs[j - 1]]), '=', sp.N(subsd[zs[j - 1]], 20))
out = []
for j in range(1, K + 1):
    c = sp.nsimplify(sp.simplify(coef[j].subs(subsd)))
    out.append(c)
    print('alpha_max coefficient of D^-%d:' % j, c, '=', sp.N(c, 25))

ok = (sp.simplify(alph[1] - 4*(1/z - 2/z**2)) == 0 and out[0] == sp.Rational(1, 2)
      and out[1] == sp.Rational(11, 16))
a2_sc = 8*(5/z - 26/z**2 + 31/z**3 - 6/z**4 - (1/z - 7/z**2 + 14/z**3 - 8/z**4)*sp.log(z))
print('alpha_2(z) - SC02 (117):', sp.simplify(alph[2] - a2_sc) if K >= 2 else None)
ok = ok and (K < 2 or sp.simplify(alph[2] - a2_sc) == 0)
print('agrees with SC02 (112), (117), (119):', ok)
sys.exit(0 if ok else 1)

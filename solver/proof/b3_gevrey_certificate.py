"""Certified Gevrey-1 bound for the M = 1 outer formal series (proof/B3_borel.md, Lemma B3.7).

Norm: for a polynomial p(x) = sum p_j x^j, ||p|| = sum |p_j| rho^j, so |p(x)| <= ||p|| on |x| <= rho; ||pq|| <= ||p|| ||q||,
||x p|| = rho ||p||, ||x p'|| <= deg(p) ||p||. Applied to the recursion of b3_outer_m1.py (centre gauge G_k(0) = 0),
with m_k >= ||D_k||, s_k >= ||S_k||, g_k >= ||G_k||, beta_k >= ||B_k||, mu_k >= ||[c(b-1)]_k||, kappa_k >= ||[cb]_k||,
q_k >= ||[S Sbar + D Dbar]_k||, and divided by k! (b~_k = beta_{k+1}/k!, mu~_k = mu_{k+1}/k!, the others x_k/k!),
every derivative loss becomes a shift by t and every convolution a Cauchy product (because i!(k-i)! <= k!). The
generating functions of the normalized majorants are then bounded coefficientwise by the minimal solution of the
positive polynomial system V = Phi(t, V) below (B3_borel.md, Section 7, gives the term-by-term derivation).

Certificate: a vector V > 0 of exact binary numbers and t1 > 0 (rational) with Phi(t1, V) <= V componentwise,
checked in ball arithmetic (V: the float fixed point at t' = (t1 + t_max)/2, then three steps of V <- Phi(t1, V)
rounded up). Then
the minimal solution at t1 is <= V, so for all k and |x| <= rho
    |D_k(x)| <= k! V_m t1^-k,  |S_k(x)| <= k! V_s t1^-k,  |B_{k+1}(x)| <= k! V_b t1^-k,  |G_k(x)| <= k! V_g t1^-k :
the series is Gevrey-1 uniformly on the disc, and its Borel transform is analytic on |zeta| < t1 for every |x| <= rho.
Trusted code: this file (Phi and the check) and python-flint's arb.

Independent check (can fail): the actual ||D_k||, ||S_k|| for k <= KCHK from b3_outer_m1.run() must obey the bounds.
Mutation: --mutate=m0 uses one tenth of the true ||D_0|| in Phi (a wrong majorant); the comparison must then fail.

    python3 b3_gevrey_certificate.py [--gauge=centre|lc] [--rho=0.75] [--A=2.72] [--w=1.0] [--margin=0.97] [--kchk=40]
                                     [--mutate=m0]
In the light-cone gauge (--gauge=lc, rho >= 1) the constant of G_k is -g(1) with |g(1)| <= ||g||, so the G row of Phi
becomes (1 + t) mu~ instead of (1 + t) mu~ / 2; nothing else changes (B3_borel.md, Lemma B3.7).
"""
import math
import sys

from flint import arb, acb, ctx, fmpq

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
ctx.prec = 200
RHO = ARGS.get("rho", "0.75")
A = ARGS.get("A", "2.72")
WF = ARGS.get("w", "1.0")
MARGIN = float(ARGS.get("margin", "0.97"))
KCHK = int(ARGS.get("kchk", "40"))
MUT = ARGS.get("mutate", "")
GAUGE = ARGS.get("gauge", "centre")      # lc: G_k(1) = 0, needs rho >= 1 (then |G_k(1)| <= ||g||, so ||G_k|| <= 2 ||g||)
GF = 1.0 if GAUGE == "lc" else 0.5      # G row: (1 + t) mu~ (lc) or (1 + t) mu~ / 2 (centre)
if GAUGE == "lc" and float(ARGS.get("rho", "0.75")) < 1:
    sys.exit("--gauge=lc needs rho >= 1")


def Phi(t, V, rho, m0, om):
    """the positive system; every coefficient is >= 0, so Phi is monotone in V and in t."""
    m, s, be, mu, g, ka, q = V
    return [m0 + rho * (1 + om / 2) * s + (1 + t) * mu * m / 2 + g * m,          # D:  ||D_k||
            (2 * rho + om * rho) * t * m + 2 * t * (1 + g) * s                     # S:  x^2 D', w x D, x c S'
            + (g + t * be + t * g * be) * s + t * (1 + g) * (1 + t * be) * s,       #     [(1 - eps) c b]_l S_j
            t * (1 + t * be) * q + 2 * t * be + t * be * be + t * be + t * t * be * be,   # B_{k+1}
            (1 + g) * be,                                                           # [c (b-1)]_{k+1}
            (1 + t) * mu * GF,                                                      # G_k
            (1 + g) * (1 + t * be),                                                 # [c b]_k
            s * s + m * m]                                                          # [S Sbar + D Dbar]_k


def float_fixed(t, rho, m0, om, it=20000):
    V = [0.0] * 7
    for _ in range(it):
        W = Phi(t, V, rho, m0, om)
        if max(W) > 1e12:
            return None
        if max(abs(a - b) / (1 + abs(a)) for a, b in zip(V, W)) < 1e-14:
            return W
        V = W
    return None


def main():
    rho_f, A_f, w_f = float(RHO), float(A), float(WF)
    om_f = 2 * math.pi / 1.64389375 * w_f
    m0_f = A_f * rho_f * (0.1 if MUT == "m0" else 1.0)
    lo, hi = 1e-6, 1.0
    for _ in range(45):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if float_fixed(mid, rho_f, m0_f, om_f) is not None else (lo, mid)
    tmax = lo
    t1_f = MARGIN * tmax
    tprime = 0.5 * (t1_f + tmax)                 # V from the fixed point at t' > t1: Phi(t1, V) < Phi(t', V) = V
    Vf = float_fixed(tprime, rho_f, m0_f, om_f)
    # rational upper roundings
    from fractions import Fraction
    def q(f, den):
        fr = Fraction(f).limit_denominator(den)
        return arb(fmpq(fr.numerator, fr.denominator))
    V = [q(v * (1 + 1e-9), 10 ** 12) + arb("1e-12") for v in Vf]     # exact rationals, rounded up
    t1 = q(t1_f, 10 ** 9)
    rho, om = arb(RHO), 2 * arb.pi() / arb("1.64389375") * arb(WF)
    m0 = arb(A) * rho * (arb("0.1") if MUT == "m0" else 1)
    # V0 = fixed point at t' > t1 is a supersolution at t1 except for the t-independent rows (mu~, q), where it is an
    # equality; three steps of the decreasing iteration V <- Phi(t1, V), rounded up, give every row strict slack.
    for _ in range(3):
        V = [arb(w.upper()) for w in Phi(t1, V, rho, m0, om)]
    W = Phi(t1, V, rho, m0, om)
    names = ["m (D)", "s (S)", "b~ (B_k+1)", "mu~", "g (G)", "kappa", "q"]
    ok = True
    print(f"gauge = {GAUGE}, rho = {RHO}, A = {A}, w/w_inf = {WF}{'  [MUTATED: m0/10]' if MUT else ''}")
    print(f"largest t with a finite fixed point (float bisection): {tmax:.6f};  certified t1 = {float(t1.mid()):.6f}")
    for nm, v, w in zip(names, V, W):
        good = w.upper() <= v.lower()
        ok &= bool(good)
        print(f"  {nm:12s} V = {float(v.mid()):.6e}   Phi(t1, V) <= {float(w.upper()):.6e}   {'ok' if good else 'VIOLATED'}")
    print(("CERTIFIED" if ok else "NOT CERTIFIED") + f": Phi(t1, V) <= V at t1 = {float(t1.mid()):.6f}")
    # consequences
    x0 = 1 / math.sqrt(2)
    phi = lambda x: math.log(x) - 0.5 * math.log(1 - x * x)
    lo_x, hi_x = x0, x0
    while hi_x + 1e-5 <= rho_f and abs(2 * phi(hi_x + 1e-5)) < float(t1.lower()):
        hi_x += 1e-5
    while abs(2 * phi(lo_x - 1e-5)) < float(t1.lower()):
        lo_x -= 1e-5
    print(f"Borel transform analytic on |zeta| < {float(t1.lower()):.5f} for |x| <= {RHO}; |2 Phi(x)| < t1 on "
          f"[{lo_x:.5f}, {hi_x:.5f}] (around 1/sqrt 2)")
    # independent check against the actual coefficients
    sys.argv = [sys.argv[0], str(KCHK), f"--gauge={GAUGE}", "--prec=256", f"--A={A}", f"--w={WF}"]
    import importlib.util, os
    spec = importlib.util.spec_from_file_location("b3m1", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                        "b3_outer_m1.py"))
    b3m1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b3m1)
    D, S, G, b = b3m1.run(verbose=False)

    def norm(p):
        return sum((abs(c).upper() * rho ** j for j, c in enumerate(p.coeffs())), arb(0))
    worst = 0.0
    fact = arb(1)
    for k in range(1, KCHK + 1):
        fact = fact * k
        for nm, p, v in (("D", D[k], V[0]), ("S", S[k], V[1])):
            bound = fact * v / t1 ** k
            r = float((norm(p) / bound).upper())
            worst = max(worst, r)
            if r > 1:
                ok = False
                print(f"  k = {k}: ||{nm}_k|| exceeds the certified bound (ratio {r:.3g})")
    print(f"actual ||D_k||, ||S_k|| (k <= {KCHK}) / certified bound: max {worst:.3e}  ({'ok' if worst <= 1 else 'FAIL'})")
    sys.exit(0 if ok else 1)


main()

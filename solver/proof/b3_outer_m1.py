"""The outer 1/n series of the one-harmonic (M = 1) model in ball arithmetic (proof/B3_borel.md).

Independent re-implementation of the recursion of Proposition 1 (README) for M = 1: p, m carry the harmonics +-1 only,
b and c are theta-independent. It shares no code with solver/largeD_late_orders.py (that one is double precision on a
theta grid with a series inverse of c(1 - eps)b; this one works with the harmonic-1 polynomial coefficients in
python-flint acb_poly and solves the S equation by subtraction).

Unknowns (eps = 1/n, p = eps^(1/2) P, m = eps^(1/2) M, S = P + M, Delta = P - M; harmonic +1 coefficient of each):
    Delta = sum_{k>=0} eps^k D_k,  S = sum_{k>=1} eps^k S_k,  c = 1 - sum_{k>=1} eps^k G_k,  b = 1 + sum_{k>=2} eps^k B_k
with D_0 = 2 x F_1 (F = A sin theta, F_1 = A/(2i)) and the coefficient of x in D_k (k >= 1) zero (K_k = 0). The
equations (README, Proposition 1; harmonic-0 part of P^2 + M^2 is S Sbar + D Dbar for M = 1):
    x c Delta' - c Delta = x^2 S' + i w x S - ((1-eps)/eps) c (b-1) Delta
    (1 - eps) c b S      = eps (x^2 Delta' - x c S' + i w x Delta)
    eps x c'             = (1 - eps) c (b - 1)
    eps x b'             = eps^2 b (S Sbar + Delta Dbar) - (1 - eps) b (b - 1)
The gauge is fixed by
    --gauge=lc      G_k(1) = 0  (c(1) = 1: the light cone at x = 1; the gauge of the README and C_selection.md)
    --gauge=centre  G_k(0) = 0  (c(0) = 1: every order is fixed by data at x = 0 alone; Lemma S1 of B3_borel.md)

Output (JSON): midpoints and radii of
    g0[k] = G_k(0), g1[k] = G_k(1); D_k, S_k, G_k, B_{k+1} at the x in --xs; Taylor coefficients of D_k (x^3, x^5,
    x^7) and G_k (x^0, x^2, x^4); in the centre gauge also xstar[k], the coefficients of the light-cone position
    X* = 1 + sum eps^k xstar_k, C(X*) = X*.

    python3 b3_outer_m1.py K [--gauge=lc|centre] [--A=2.72] [--w=1.0] [--prec=320] [--out=file.json] [--xs=...]
           [--mutate=omega_S|drop_bQ|scale_G]

--mutate breaks the recursion on purpose (mutation controls): omega_S flips the sign of w in the S equation only;
drop_bQ scales the source of b at order 3 by 0.999; scale_G scales the gauge constant of every G_k by 1.001.
"""
import json
import sys
import time

from flint import acb, acb_poly, arb, ctx

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
POS = [a for a in sys.argv[1:] if not a.startswith("--")]
K = int(POS[0]) if POS else 40
GAUGE = ARGS.get("gauge", "lc")
ctx.prec = int(ARGS.get("prec", 320))
A = arb(ARGS.get("A", "2.72"))
WF = arb(ARGS.get("w", "1.0"))
MUT = ARGS.get("mutate", "")
OUT = ARGS.get("out", f"b3_outer_m1_{GAUGE}_K{K}.json")
XS = [arb(s) for s in ARGS.get("xs", "0.5,0.7071067811865475244,0.8,0.95,1").split(",")]

OMEGA = 2 * arb.pi() / arb("1.64389375") * WF
I = acb(0, 1)
X = acb_poly([0, 1])
ZERO = acb_poly([])
ONE = acb_poly([1])


def conjp(p):
    """the harmonic -1 polynomial: coefficient-wise conjugate (x real)."""
    return acb_poly([c.conjugate() for c in p.coeffs()])


def deriv(p):
    return p.derivative() if p.degree() > 0 else ZERO


def xd(p):
    return X * deriv(p)


def coef(a, b, k, lo_a=0, lo_b=0):
    """[a b]_k for series a, b given as lists (index = order) of polynomials"""
    acc = ZERO
    for i in range(lo_a, k + 1 - lo_b):
        if i < len(a) and k - i < len(b):
            acc = acc + a[i] * b[k - i]
    return acc


def small(z, scale):
    return abs(z).upper() < arb("1e-25") * (1 + scale)


def run(verbose=True):
    t0 = time.time()
    F1 = A / (2 * I)
    D = [2 * F1 * X]
    Dbar = [conjp(D[0])]
    S = [ZERO]
    Sbar = [ZERO]
    Sd = [ZERO]                  # S_k'
    c = [ONE]                    # c_k = -G_k
    G = [ZERO]
    b = [ONE, ZERO]              # b_k = B_k for k >= 2
    bm1 = [ZERO, ZERO]           # (b - 1)_k
    Q = []                       # Q_j = [S Sbar + D Dbar]_j
    cbm1 = [ZERO, ZERO]          # [c (b - 1)]_j, final once b_j and c_{j-2} are known
    cb = [ONE]                   # [c b]_j
    for k in range(1, K + 1):
        Q.append(coef(S, Sbar, k - 1) + coef(D, Dbar, k - 1))
        if MUT == "drop_bQ" and k == 3:
            Q[-1] = Q[-1] * arb("0.999")
        # order eps^{k+1} of the b equation: B_{k+1} = [b Q]_{k-1} - x B_k' - sum_{i=2}^{k-1} B_i B_{k+1-i} + [b (b-1)]_k
        Bk1 = coef(b, Q, k - 1) - xd(b[k]) - coef(bm1, bm1, k + 1, lo_a=2, lo_b=2) + coef(b, bm1, k)
        b.append(Bk1)
        bm1.append(Bk1)
        # order eps^{k+1} of the c equation: x c_k' = [c (b-1)]_{k+1} - [c (b-1)]_k
        cbm1.append(coef(c, bm1, k + 1))       # uses c_0 .. c_{k-1}
        rhs = cbm1[k + 1] - cbm1[k]
        cs = rhs.coeffs()
        scale = max([abs(z).upper() for z in cs] + [arb(0)])
        if cs and not small(cs[0], scale):
            print(f"order {k}: E1 obstruction, constant term {cs[0]}")
        g = acb_poly([acb(0)] + [-cs[j] / j for j in range(1, len(cs))])     # -x G' = rhs  =>  G = -int rhs/x
        const = -g(acb(1)) if GAUGE == "lc" else acb(0)
        if MUT == "scale_G":
            const = const * arb("1.001")
        Gk = g + const
        G.append(Gk)
        c.append(-Gk)
        cb.append(coef(c, b, k))
        # order eps^k of the S equation: S_k = [num]_{k-1} - sum_{j=1}^{k-1} [(1 - eps) c b]_{k-j} S_j
        om = -OMEGA if MUT == "omega_S" else OMEGA
        num = X * X * deriv(D[k - 1]) - X * coef(c, Sd, k - 1) + I * om * X * D[k - 1]
        Sk = num
        for j in range(1, k):
            Sk = Sk - (cb[k - j] - cb[k - j - 1]) * S[j]
        S.append(Sk)
        Sbar.append(conjp(Sk))
        Sd.append(deriv(Sk))
        # order eps^k of the Delta equation: x D_k' - D_k = R_k
        R = X * X * Sd[k] + I * OMEGA * X * Sk - coef(cbm1, D, k + 1) + coef(cbm1, D, k)
        for i in range(1, k + 1):
            R = R - c[i] * (xd(D[k - i]) - D[k - i])
        rc = R.coeffs()
        scale = max([abs(z).upper() for z in rc] + [arb(0)])
        if len(rc) > 1 and not small(rc[1], scale):
            print(f"order {k}: solvability, x^1 term {rc[1]}")
        dk = [acb(0)] * len(rc)
        for j in range(len(rc)):
            if j != 1:
                dk[j] = rc[j] / (j - 1)
        D.append(acb_poly(dk))
        Dbar.append(conjp(D[-1]))
        if verbose and k % 10 == 0:
            print(f"  order {k}  deg D_k {D[-1].degree()}  {time.time() - t0:.1f}s", flush=True)
    return D, S, G, b


def xstar_series(G):
    """X* = 1 + d(eps), d = -sum_{k>=1} eps^k G_k(1 + d): order by order, with g[k][m] = G_k^(m)(1)/m! and
    P[m][j] = [d^m]_j."""
    shift = acb_poly([1, 1])
    g = [None] + [G[k](shift).coeffs() for k in range(1, len(G))]   # Taylor coefficients of G_k at x = 1
    Kx = len(G) - 1
    d = [acb(0)] * (Kx + 1)
    P = [[acb(1)] + [acb(0)] * Kx]                                    # P[0] = 1
    for k in range(1, Kx + 1):
        tot = acb(0)
        for kp in range(1, k + 1):
            j = k - kp
            for m in range(0, min(j, len(g[kp]) - 1) + 1):
                if m < len(P):
                    tot = tot + g[kp][m] * P[m][j]
        d[k] = -tot
        # extend P: new row m = k (lowest order k) and column k of every row
        P.append([acb(0)] * (Kx + 1))
        for m in range(1, k + 1):
            P[m][k] = sum((d[i] * P[m - 1][k - i] for i in range(1, k - m + 2)), acb(0))
    return d[1:]


def sm(a):
    return a.mid().str(60, radius=False)


def mr(z):
    return [sm(z.real), sm(z.imag), float(z.rad())]


def main():
    D, S, G, b = run()
    out = dict(K=K, gauge=GAUGE, A=sm(A), w=sm(WF), prec=ctx.prec, mutate=MUT, xs=[sm(x) for x in XS])
    out["g0"] = [mr(G[k](acb(0))) for k in range(1, K + 1)]
    out["g1"] = [mr(G[k](acb(1))) for k in range(1, K + 1)]
    ser = {}
    for x in XS:
        xa = acb(x)
        ser[sm(x)] = dict(D=[mr(D[k](xa)) for k in range(1, K + 1)], S=[mr(S[k](xa)) for k in range(1, K + 1)],
                                 G=[mr(G[k](xa)) for k in range(1, K + 1)], B=[mr(b[k + 1](xa)) for k in range(1, K + 1)])
    out["series"] = ser

    def tc(p, j):
        cs = p.coeffs()
        return mr(cs[j]) if j < len(cs) else ["0", "0", 0.0]
    out["taylorD"] = {str(j): [tc(D[k], j) for k in range(1, K + 1)] for j in (3, 5, 7)}
    out["taylorG"] = {str(j): [tc(G[k], j) for k in range(1, K + 1)] for j in (0, 2, 4)}
    if GAUGE == "centre":
        out["xstar"] = [mr(z) for z in xstar_series(G)]
    with open(OUT, "w") as f:
        json.dump(out, f)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

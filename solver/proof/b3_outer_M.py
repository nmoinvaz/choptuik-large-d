"""The outer 1/n series of the M-harmonic model in ball arithmetic, harmonic by harmonic (proof/B3_borel.md).

Generalizes b3_outer_m1.py (and reproduces it at M = 1). The M-harmonic model applies P_M to every right-hand side of
the self-similar equations written as polynomials in (p, m, b, c) (C_selection.md, Section 1): odd harmonics |mu| <= M
for p, m; even harmonics |mu| <= M - 1 for b, c. Products are formed in full (all harmonics) and projected once per
equation, which is what "P_M applied to the right-hand side" means. In particular the S equation is solved as
P_M[(1 - eps) c b S] = P_M[num], order by order, not by dividing by c b (division and projection do not commute when
c, b depend on theta; solver/largeD_late_orders.py divides on the theta grid and then projects, a slightly different
truncation for M >= 3; at M = 1 the two agree).

Fields are dicts {mu: acb_poly in x}; only mu >= 0 is stored, mu < 0 is the coefficient-wise conjugate (real fields).

    python3 b3_outer_M.py M K [--A=2.72] [--w=1.0] [--prec=400] [--out=file.json] [--xs=0.5,0.8,0.95,1]
           [--mutate=omega_S|drop_bQ|scale_G]

Output (JSON): for every x in --xs and every stored harmonic, D_k, S_k (odd mu), G_k, B_{k+1} (even mu) as
midpoint/radius triples; G_k(0) and G_k(1) per even harmonic.
"""
import json
import sys
import time

from flint import acb, acb_poly, arb, ctx

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
POS = [a for a in sys.argv[1:] if not a.startswith("--")]
MH = int(POS[0]) if POS else 3
K = int(POS[1]) if len(POS) > 1 else 30
ctx.prec = int(ARGS.get("prec", 400))
A = arb(ARGS.get("A", "2.72"))
WF = arb(ARGS.get("w", "1.0"))
MUT = ARGS.get("mutate", "")
OUT = ARGS.get("out", f"b3_outer_M{MH}_K{K}.json")
XS = [arb(s) for s in ARGS.get("xs", "0.5,0.8,0.95,1").split(",")]
OMEGA = 2 * arb.pi() / arb("1.64389375") * WF
I = acb(0, 1)
X = acb_poly([0, 1])
ZERO = acb_poly([])
ONE = acb_poly([1])
ODD = list(range(1, MH + 1, 2))
EVEN = list(range(0, MH, 2))


def conjp(p):
    return acb_poly([c.conjugate() for c in p.coeffs()])


def get(f, mu):
    """harmonic mu of a field stored for mu >= 0"""
    if mu >= 0:
        return f.get(mu, ZERO)
    v = f.get(-mu)
    return conjp(v) if v is not None else ZERO


def mul(f, g, keep):
    """full product of two real fields, harmonics mu in keep (>= 0) of the result"""
    fk = [m for m in f if f[m].length() > 0]
    gk = [m for m in g if g[m].length() > 0]
    fall = sorted(set(fk + [-m for m in fk]))
    gall = sorted(set(gk + [-m for m in gk]))
    out = {}
    for mu in keep:
        acc = ZERO
        for a in fall:
            b = mu - a
            if b in gall:
                acc = acc + get(f, a) * get(g, b)
        out[mu] = acc
    return out


def add(*fs, coeff=None):
    out = {}
    for i, f in enumerate(fs):
        cf = 1 if coeff is None else coeff[i]
        for mu, p in f.items():
            out[mu] = out.get(mu, ZERO) + (p if cf == 1 else p * cf)
    return out


def scal(f, s):
    return {mu: p * s for mu, p in f.items()}


def fmap(f, op):
    return {mu: op(p) for mu, p in f.items()}


def deriv(p):
    return p.derivative() if p.degree() > 0 else ZERO


def dth(f, om):
    return {mu: p * (I * om * mu) for mu, p in f.items()}


def xmul(f, k=1):
    return {mu: p * (X if k == 1 else X * X) for mu, p in f.items()}


def coef(a, b, k, keep, lo_a=0, lo_b=0):
    acc = {mu: ZERO for mu in keep}
    for i in range(lo_a, k + 1 - lo_b):
        if i < len(a) and k - i < len(b):
            acc = add(acc, mul(a[i], b[k - i], keep))
    return acc


ALLH = list(range(0, 3 * MH + 1))      # harmonics kept in unprojected intermediate products


def proj(f, keep):
    return {mu: f.get(mu, ZERO) for mu in keep}


def run(verbose=True):
    t0 = time.time()
    F1 = A / (2 * I)
    D = [{1: 2 * F1 * X}]
    S = [{}]
    c = [{0: ONE}]
    G = [{}]
    b = [{0: ONE}, {}]
    bm1 = [{}, {}]
    Q = []
    cbm1 = [{}, {}]          # [c (b-1)]_j, unprojected (harmonics up to 2(M-1))
    cb = [{0: ONE}]          # [c b]_j, unprojected
    for k in range(1, K + 1):
        # Q_{k-1} = [(S^2 + D^2)/2]_{k-1}, harmonics even, up to 2M (unprojected)
        q = add(coef(S, S, k - 1, ALLH), coef(D, D, k - 1, ALLH))
        Q.append(scal(q, arb("0.5")))
        if MUT == "drop_bQ" and k == 3:
            Q[-1] = scal(Q[-1], arb("0.999"))
        # B_{k+1} = P[[b Q]_{k-1}] - x B_k' - P[sum B_i B_{k+1-i}] + P[[b (b-1)]_k]
        t1 = coef(b, Q, k - 1, EVEN)
        t2 = fmap(b[k], lambda p: X * deriv(p))
        t3 = coef(bm1, bm1, k + 1, EVEN, lo_a=2, lo_b=2)
        t4 = coef(b, bm1, k, EVEN)
        Bk1 = proj(add(t1, scal(t2, -1), scal(t3, -1), t4), EVEN)
        b.append(Bk1)
        bm1.append(Bk1)
        # x c_k' = P[[c (b-1)]_{k+1} - [c (b-1)]_k]
        cbm1.append(coef(c, bm1, k + 1, ALLH))
        rhs = proj(add(cbm1[k + 1], scal(cbm1[k], -1)), EVEN)
        Gk = {}
        for mu in EVEN:
            cs = rhs[mu].coeffs()
            if cs and not abs(cs[0]).upper() < arb("1e-25") * (1 + max(abs(z).upper() for z in cs)):
                print(f"order {k} mu {mu}: E1 obstruction {cs[0]}")
            g = acb_poly([acb(0)] + [-cs[j] / j for j in range(1, len(cs))])
            const = -g(acb(1))
            if MUT == "scale_G":
                const = const * arb("1.001")
            Gk[mu] = g + const
        G.append(Gk)
        c.append(scal(Gk, -1))
        cb.append(coef(c, b, k, ALLH))
        # S_k: P[(1-eps) c b S]_k = P[num_{k-1}], num = x^2 D' - x [c S']  + w x D_theta
        om = -OMEGA if MUT == "omega_S" else OMEGA
        Sd = [fmap(s, deriv) for s in S]
        num = add(xmul(fmap(D[k - 1], deriv), 2), scal(xmul(coef(c, Sd, k - 1, ODD)), -1),
                  xmul(dth(D[k - 1], om)))
        Sk = proj(num, ODD)
        for j in range(1, k):
            w = add(cb[k - j], scal(cb[k - j - 1], -1))
            Sk = add(Sk, scal(mul(w, S[j], ODD), -1))
        S.append(proj(Sk, ODD))
        # D_k: x D_k' - D_k = R_k
        Sdk = fmap(S[k], deriv)
        R = add(xmul(Sdk, 2), xmul(dth(S[k], OMEGA)), scal(coef(cbm1, D, k + 1, ODD), -1), coef(cbm1, D, k, ODD))
        for i in range(1, k + 1):
            R = add(R, scal(mul(c[i], add(fmap(D[k - i], lambda p: X * deriv(p)), scal(D[k - i], -1)), ODD), -1))
        Dk = {}
        for mu in ODD:
            rc = R[mu].coeffs()
            if len(rc) > 1 and not abs(rc[1]).upper() < arb("1e-25") * (1 + max(abs(z).upper() for z in rc)):
                print(f"order {k} mu {mu}: solvability x^1 term {rc[1]}")
            Dk[mu] = acb_poly([rc[j] / (j - 1) if j != 1 else acb(0) for j in range(len(rc))])
        D.append(Dk)
        if verbose and k % 5 == 0:
            print(f"  order {k}  {time.time() - t0:.1f}s", flush=True)
    return D, S, G, b


def sm(a):
    return a.mid().str(60, radius=False)


def mr(z):
    return [sm(z.real), sm(z.imag), float(z.rad())]


def main():
    D, S, G, b = run()
    out = dict(M=MH, K=K, A=sm(A), w=sm(WF), prec=ctx.prec, mutate=MUT, xs=[sm(x) for x in XS], series={})
    for x in XS:
        xa = acb(x)
        ser = {}
        for mu in ODD:
            ser[f"D{mu}"] = [mr(get(D[k], mu)(xa)) for k in range(1, K + 1)]
            ser[f"S{mu}"] = [mr(get(S[k], mu)(xa)) for k in range(1, K + 1)]
        for mu in EVEN:
            ser[f"G{mu}"] = [mr(get(G[k], mu)(xa)) for k in range(1, K + 1)]
            ser[f"B{mu}"] = [mr(get(b[k + 1], mu)(xa)) for k in range(1, K + 1)]
        out["series"][sm(x)] = ser
    out["g0"] = {str(mu): [mr(get(G[k], mu)(acb(0))) for k in range(1, K + 1)] for mu in EVEN}
    out["g1"] = {str(mu): [mr(get(G[k], mu)(acb(1))) for k in range(1, K + 1)] for mu in EVEN}
    with open(OUT, "w") as f:
        json.dump(out, f)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

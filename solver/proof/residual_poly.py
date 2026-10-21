"""Rigorous residual polynomial of the truncated formal M = 1 inner series (exact in ball arithmetic):
R(xi) = ODE residual of (D_N, S_N, C_N) = sum_k rho_k xi^k with k >= kmin; bound |R(xi)| <= (|xi|/r0)^kmin sum |rho_k| r0^k
for |xi| <= r0 (any direction). This is the forcing M_0 of the fixed-point lemma (step i of A_certify.md)."""
from flint import acb, acb_poly, arb, ctx
from series_arb import SeriesArb


def poly_series(sa, d, s, g, n):
    X = acb_poly([0, 1])
    D = {}
    for mu in sa.odd:
        c = [acb(0)] * (2 * n + 1)
        for j in range(n):
            c[2 * j + 1] = d[j][mu]
        D[mu] = acb_poly(c)
    S = {}
    for mu in sa.odd:
        c = [acb(0)] * (2 * n + 2)
        for j in range(n):
            c[2 * j + 2] = s[j][mu]
        S[mu] = acb_poly(c)
    C = {}
    for nu in sa.even:
        c = [acb(0)] * (2 * n + 1)
        for j in range(n):
            c[2 * j] = g[j][nu]
        C[nu] = acb_poly(c)
    return D, S, C, X


def pmul(A, B, keep):
    out = {k: acb_poly([0]) for k in keep}
    for i, ai in A.items():
        for j, bj in B.items():
            if i + j in out:
                out[i + j] = out[i + j] + ai * bj
    return out


def residual_poly(sa, d, s, g, n):
    D, S, C, X = poly_series(sa, d, s, g, n)
    w, I = sa.w, sa.I
    dD = {mu: D[mu].derivative() for mu in sa.odd}
    dS = {mu: S[mu].derivative() for mu in sa.odd}
    dC = {nu: C[nu].derivative() for nu in sa.even}
    cs = pmul(C, S, sa.odd)
    r1 = {mu: X * X * dD[mu] + (I * mu * w) * X * D[mu] - cs[mu] for mu in sa.odd}
    E = {k: v * acb(0.5) for k, v in pmul(D, D, sa.even).items()}
    cE = pmul(C, E, sa.even)
    cEd = pmul(cE, D, sa.odd)
    cd = pmul(C, D, sa.odd)
    cxdD = pmul(C, {mu: X * dD[mu] for mu in sa.odd}, sa.odd)
    r2 = {mu: X * X * dS[mu] + (I * mu * w) * X * S[mu] - (cxdD[mu] + cEd[mu] - cd[mu]) for mu in sa.odd}
    r3 = {nu: X * dC[nu] - cE[nu] for nu in sa.even}
    return r1, r2, r3


def weighted_bound(p, r0):
    """returns (kmin, sum_k |rho_k| r0^k) with kmin the lowest k whose coefficient ball does not contain 0"""
    cs = p.coeffs()
    kmin = next((k for k, c in enumerate(cs) if not c.contains(0)), len(cs))
    zero_low = all(abs(c).upper() < arb(10) ** -60 for c in cs[:kmin])
    tot = sum((abs(c) * arb(r0) ** k for k, c in enumerate(cs)), arb(0))
    return kmin, tot, zero_low


if __name__ == "__main__":
    import sys
    ctx.prec = 320
    r0 = sys.argv[1] if len(sys.argv) > 1 else "0.1"
    W0 = 2 * arb.pi() / arb("1.64389375")
    A, wf = arb("2.5421480475"), arb("0.7769940136")
    sa = SeriesArb({1: acb(0, -A / 2), -1: acb(0, A / 2)}, W0 * wf, prec=320)
    J = int(1 / (2 * float(r0) ** 2)) + 30
    d, s, g = sa.series(J)
    x2 = arb(r0) ** 2
    mags = [max(abs(d[j][mu]) for mu in sa.odd) * x2 ** j for j in range(J)]
    n = min(range(1, J), key=lambda j: float(mags[j].mid()))
    r1, r2, r3 = residual_poly(sa, d, s, g, n)
    worst = arb(0)
    for name, dct in (("I1", r1), ("I2", r2), ("I3", r3)):
        for k, p in dct.items():
            kmin, tot, zl = weighted_bound(p, r0)
            worst = max(worst, tot, key=lambda a: float(a.upper()))
            print(f"{name} harmonic {k:+d}: residual starts at xi^{kmin} (lower coefficients enclose 0, |.| < 1e-60: {zl}); "
                  f"sum |rho_k| r0^k = {tot.str(3)}")
    print(f"N = {n} terms; for |xi| <= {r0}: |R(xi)| <= (|xi|/{r0})^kmin * {float(worst.upper()):.2e}")

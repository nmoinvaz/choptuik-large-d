#!/usr/bin/env python3
"""Review of fixed_point_M_b.py (#867), check K1: the system in z, the residual and the series.

The reference is inner_np.py: its rhs (the solver's code, double precision) and a transcription of its equations in
ball arithmetic written here by the reviewer,
    xi^2 D' + i mu w xi D = [C S],   xi^2 S' + i mu w xi S = [C xi D'] + [[C E] D] - [C D],   xi C' = [C E],
    E = [D D] / 2,
D, S on odd |mu| <= M, C and E on even |nu| <= M + 1, each product projected. For polynomials P, Q, C in z = xi^2 and
D = xi P, S = z Q, the defects in these equations are, in the module's residual numerators (V = Q - (1 + s) P),
    xi^2 D' + s xi D - [C S]                            = -z N_P
    xi^2 S' + s xi S - [C xi D'] - [[C E] D] + [C D]    = xi (-N_V - (1 + s) z N_P + [C N_P])
    xi C' - [C E]                                       = -2 z N_C
(the reviewer's algebra; the [C N_P] term is there because the module writes 2 [C P'] as [C W] / z). Against
inner_np's rhs, which uses its own D' inside the S equation, the same reads
    (D' - rhs_D) xi^2 = -z N_P,   (S' - rhs_S) xi^2 = -xi (N_V + (1 + s) z N_P),   (C' - rhs_C) xi = -2 z N_C.

  (a) random polynomials of degree 6 with coefficients of size 1, complex w: every term of the system is of size 1.
      The first three identities at a point on the ray and at a point off it, to a relative 1e-120 (512 bits).
  (b) the same polynomials with real w against inner_np.Inner.rhs, to a relative 1e-11 (double precision).
  (c) the module's series Y_N at the M = 5 zero: the low orders of the residual vanish, and the first identities
      hold with the true residual, to 1e-20 of its size.
  (d) the module's N terms at |xi| = 0.9 r0 against inner_np.series_at's sums (its own, longer truncation; the
      terms past N are below 1e-50 there), relative 1e-11.

    python3 rv_system.py [--M=5]
    python3 rv_system.py --mutate=scale    the module's residual with its [G P] times 1 + 1e-9: (a) must fail. It is
                                           below the double-precision resolution of (b), which passes.
    python3 rv_system.py --mutate=sign     the module's residual with -[G P]: (a) and (b) must fail
Exit status 0 only if every check passes.
"""
import os
import random
import sys

import numpy as np
from flint import acb, acb_poly, arb, ctx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "inner_stokes"))
import fixed_point_M_b as B  # noqa: E402  (the module under review)
import inner_np  # noqa: E402  (the solver)

I = acb(0, 1)
ZEROS = {3: ["2.726475886", "1.015603095", "0.144419478488", "0.011087425887"],
         5: ["2.718579189", "0.999228404", "0.149203574672", "0.014719052460", "0.008405346879", "0.001470345176"]}


def sets(M):
    return list(range(-M, M + 1, 2)), list(range(-(M + 1), M + 2, 2))


def proj(A, Bd, target):
    """[A B] on the harmonics of target (values, dicts over harmonics)"""
    out = {k: acb(0) for k in target}
    for i, a in A.items():
        for j, b in Bd.items():
            if i + j in out:
                out[i + j] = out[i + j] + a * b
    return out


def defects(M, w, P, Q, C, xi):
    """the defects of (xi P, z Q, C) in inner_np's three equations at the point xi (the reviewer's transcription)"""
    odd, even = sets(M)
    z = xi * xi
    D = {mu: xi * P[mu](z) for mu in odd}
    dD = {mu: P[mu](z) + 2 * z * P[mu].derivative()(z) for mu in odd}
    S = {mu: z * Q[mu](z) for mu in odd}
    dS = {mu: 2 * xi * (Q[mu](z) + z * Q[mu].derivative()(z)) for mu in odd}
    Cv = {nu: C[nu](z) for nu in even}
    dC = {nu: 2 * xi * C[nu].derivative()(z) for nu in even}
    E = {nu: v / 2 for nu, v in proj(D, D, even).items()}
    cE = proj(Cv, E, even)
    CS, CD = proj(Cv, S, odd), proj(Cv, D, odd)
    CxdD = proj(Cv, {mu: xi * dD[mu] for mu in odd}, odd)
    cED = proj(cE, D, odd)
    eD = {mu: z * dD[mu] + I * mu * w * xi * D[mu] - CS[mu] for mu in odd}
    eS = {mu: z * dS[mu] + I * mu * w * xi * S[mu] - CxdD[mu] - cED[mu] + CD[mu] for mu in odd}
    eC = {nu: xi * dC[nu] - cE[nu] for nu in even}
    return eD, eS, eC, (D, S, Cv, dD, dS, dC)


def module_side(M, w, P, Q, C, xi, gp=None):
    """the same three defects from the module's residual numerators"""
    odd, even = sets(M)
    s = {mu: w * (I * mu) for mu in odd}
    V = {mu: Q[mu] - (1 + s[mu]) * P[mu] for mu in odd}
    NP, NV, NC = B.residual_numerators(M, s, P, Q, C, V, True, gp)
    z = xi * xi
    nP = {mu: NP[mu](z) for mu in odd}
    CNP = proj({nu: C[nu](z) for nu in even}, nP, odd)
    mD = {mu: -z * nP[mu] for mu in odd}
    mS = {mu: xi * (-NV[mu](z) - (1 + s[mu]) * z * nP[mu] + CNP[mu]) for mu in odd}
    mS_rhs = {mu: -xi * (NV[mu](z) + (1 + s[mu]) * z * nP[mu]) for mu in odd}
    mC = {nu: -2 * z * NC[nu](z) for nu in even}
    return mD, mS, mC, mS_rhs, (NP, NV, NC)


def rel(a, b):
    """max |a - b| / max |a| over the harmonics, as floats"""
    num = max(float(abs(a[k] - b[k]).upper()) for k in a)
    den = max(float(abs(a[k]).upper()) for k in a)
    return num / den if den > 0 else num


def rand_polys(M, rng, deg=6):
    odd, even = sets(M)

    def rp():
        return acb_poly([acb(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(deg + 1)])
    return {mu: rp() for mu in odd}, {mu: rp() for mu in odd}, {nu: rp() for nu in even}


def main():
    opts = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
    M = int(opts.get("M", "5"))
    mutate = opts.get("mutate")
    ctx.prec = 512
    gp = {"scale": 1 + acb("1e-9"), "sign": acb(-1)}.get(mutate)
    rng = random.Random(867)
    odd, even = sets(M)
    ok = True
    print(f"rv_system: M = {M}, mutation {mutate}")

    # (a) random polynomials, complex w, ball arithmetic
    P, Q, C = rand_polys(M, rng)
    w = acb("3.8", "0.004")
    worst = 0.0
    for name, xi in (("on the ray", acb("0.05") * (I * 3 * arb.pi() / 8).exp()), ("off it", acb("0.21", "-0.13"))):
        eD, eS, eC, _ = defects(M, w, P, Q, C, xi)
        mD, mS, mC, _, _ = module_side(M, w, P, Q, C, xi, gp)
        r = (rel(eD, mD), rel(eS, mS), rel(eC, mC))
        worst = max(worst, *r)
        print(f"(a) random polynomials, {name}: relative differences D {r[0]:.1e}, S {r[1]:.1e}, C {r[2]:.1e}")
    oka = worst < 1e-120
    print(f"(a) the module's residual numerators are the defects of inner_np's equations: {oka}")
    ok = ok and oka

    # (b) inner_np's rhs, real w, double precision
    wr = 3.8
    inn = inner_np.Inner(M, wr, {1: 0.3 - 0.2j, -1: 0.3 + 0.2j})
    L = inn.L
    worst = 0.0
    for name, xif in (("on the ray", 0.05 * np.exp(3j * np.pi / 8)), ("off it", 0.21 - 0.13j)):
        xi = acb(xif.real, xif.imag)
        _, _, _, (D, S, Cv, dD, dS, dC) = defects(M, acb(wr), P, Q, C, xi)
        _, _, mC, mS_rhs, _ = module_side(M, acb(wr), P, Q, C, xi, gp)
        mD = module_side(M, acb(wr), P, Q, C, xi, gp)[0]

        def arr(d):
            a = np.zeros(2 * L + 1, complex)
            for k, v in d.items():
                a[k + L] = complex(float(v.real.mid()), float(v.imag.mid()))
            return a
        f = inn.rhs(xif, inn.pack(arr(D), arr(S), arr(Cv)))
        no, ne = len(odd), len(even)
        lhs = np.concatenate([(arr(dD)[inn.odd] - f[:no]) * xif ** 2, (arr(dS)[inn.odd] - f[no:2 * no]) * xif ** 2,
                              (arr(dC)[inn.even] - f[2 * no:]) * xif])
        rhs = np.concatenate([arr(mD)[inn.odd], arr(mS_rhs)[inn.odd], arr(mC)[inn.even]])
        r = [float(np.abs(lhs[a:b] - rhs[a:b]).max() / np.abs(rhs[a:b]).max())
             for a, b in ((0, no), (no, 2 * no), (2 * no, 2 * no + ne))]
        worst = max(worst, *r)
        print(f"(b) against inner_np.Inner.rhs, {name}: relative differences D {r[0]:.1e}, S {r[1]:.1e}, C {r[2]:.1e}")
    okb = worst < 1e-11
    print(f"(b) the same against the solver's own rhs: {okb}")
    ok = ok and okb

    # (c) the module's series at the zero: low orders, and the value of the true residual
    st = B.Setup([acb(arb(v)) for v in ZEROS[M]], M, "0.07" if M == 5 else "0.1", "3pi/8")
    xi = st.r0 * acb("0.9") * (I * st.arg).exp()
    eD, eS, eC, _ = defects(M, st.w, st.P, st.Q, st.C, xi)
    mD, mS, mC, _, (NP, NV, NC) = module_side(M, st.w, st.P, st.Q, st.C, xi, gp)
    r = (rel(eD, mD), rel(eS, mS), rel(eC, mC))
    size = max(float(abs(v).upper()) for v in mD.values())
    okc = st.low_ok and max(r) < 1e-20
    print(f"(c) the series at the M = {M} zero, N = {st.N}: low orders of the residual contain 0: {st.low_ok} (largest "
          f"|ball| R^j {st.low_max:.1e}); the true defect at |xi| = 0.9 r0 is {size:.1e}, and the identities hold to a "
          f"relative D {r[0]:.1e}, S {r[1]:.1e}, C {r[2]:.1e}: {okc}")
    ok = ok and okc

    # (d) the series against inner_np.series_at, at its own truncation
    xf = [float(v) for v in ZEROS[M]]
    inn = inner_np.Inner(M, inner_np.W0 * xf[1], inner_np.Fh_from(xf, M))
    xi0 = 0.9 * float(st.r0.mid()) * np.exp(3j * np.pi / 8)
    Y, jt, _ = inn.series_at(xi0, int(1.2 / (2 * abs(xi0) ** 2)) + 40)
    z0 = acb(xi0.real, xi0.imag) ** 2
    xa = acb(xi0.real, xi0.imag)

    def trunc(p):
        return p(z0)
    mine = [xa * trunc(st.P[mu]) for mu in odd] + [z0 * trunc(st.Q[mu]) for mu in odd] + [trunc(st.C[nu]) for nu in even]
    mf = np.array([complex(float(v.real.mid()), float(v.imag.mid())) for v in mine])
    rd = float(np.abs(mf - Y).max() / np.abs(Y).max())
    okd = rd < 1e-11
    print(f"(d) the module's {st.N} terms against inner_np.series_at's sums ({jt} terms) at |xi| = 0.9 r0: relative "
          f"difference {rd:.1e}: {okd}")
    ok = ok and okd
    print(f"RV_SYSTEM: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

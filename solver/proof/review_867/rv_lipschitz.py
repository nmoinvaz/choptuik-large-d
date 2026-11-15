#!/usr/bin/env python3
"""Review of fixed_point_M_b.py (#867), check K3: the Lipschitz matrix and the forcing, from below.

The module's L_ij bounds the weighted norm of d(T eta)_i / d eta_j over the set X_w, and F_i that of T(0)_i. Here
the reviewer linearizes the integral operator T at eta = 0 by hand, from the system in z,
    dPhi_P = dW / (2z),      dW = dv + d[ct Q],    d[ct Q] = [dc Q] + [ct dQ],    dQ = dv + (1 + s) dp,
    dPhi_V = (d[ct Q] + [dc W] + [ct dW]) / (2z^2) + (-(2 + s)(1 + s) dp - (1 + s) d[ct Q] + [dG P] + [G dp]) / (2z),
    dPhi_C = dG / 2,         dG = dE + [dc E] + [ct dE],    dE = [P dp],
applies it to the direction eta_j = z^(k_j), whose norm is 1, and evaluates (DT eta_j)_i at z = R e^(i phi): the slow
rows by integrating the polynomial, the fast rows by mpmath's quadrature of the kernel in tau = 1/rho - 1/r. That gives
a lower bound l_ij of the true norm, so l_ij <= L_ij must hold in every entry, and l_ij = 0 wherever L_ij = 0. The same
for the forcing: the size of T(0)_i at |z| = R, from the residual numerators that check K1 ties to inner_np.py, must
not exceed F_i, nor the proved radius w2_i of eta.

    python3 rv_lipschitz.py [--M=5] [--k=K] [--sector=d]
    python3 rv_lipschitz.py --mutate=nolabel   the module's majorant without the label term (2 + s)(1 + s) P of Phi_V.
        Written before the run: the entries v_mu <- p_mu must then fall below the true derivative (exit 1).
    python3 rv_lipschitz.py --mutate=gpdrop    the module's majorant without [G P]. Whether this check or the author's
        Lipschitz comparison (fixed_point_M_b_check.py lip) notices is reported; no verdict was fixed beforehand.
Exit status 0 only if no entry exceeds its bound.
"""
import os
import sys

import mpmath as mp
from flint import acb, acb_poly, arb, ctx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import fixed_point_M_b as B  # noqa: E402  (the module under review)

ZEROS = {3: ["2.726475886", "1.015603095", "0.144419478488", "0.011087425887"],
         5: ["2.718579189", "0.999228404", "0.149203574672", "0.014719052460", "0.008405346879", "0.001470345176"]}
Z = acb_poly([0])


def pj(A, Bd, target):
    """[A B] on the harmonics of target (polynomials)"""
    out = {k: Z for k in target}
    for i, a in A.items():
        for j, b in Bd.items():
            if i + j in out:
                out[i + j] = out[i + j] + a * b
    return out


def down(p, n):
    """p / z^n for a polynomial whose n lowest coefficients are 0"""
    cs = p.coeffs()
    assert all(c == 0 for c in cs[:n]), "the lowest coefficients must vanish"
    return acb_poly(cs[n:])


def mpc(z):
    return mp.mpc(mp.mpf(z.real.mid().str(40, radius=False)), mp.mpf(z.imag.mid().str(40, radius=False)))


def linearized(st, dp, dv, dc):
    """dPhi_P z, dPhi_V z^2 and dPhi_C at Y_N in the direction (dp, dv, dc): polynomials"""
    odd, even, s = st.odd, st.even, st.s
    one = acb_poly([1])
    ct = {nu: st.C[nu] - (one if nu == 0 else Z) for nu in even}
    W = {mu: st.V[mu] + v for mu, v in pj(ct, st.Q, odd).items()}
    E = {nu: v * acb(0.5) for nu, v in pj(st.P, st.P, even).items()}
    G = {nu: E[nu] + v for nu, v in pj(ct, E, even).items()}
    dQ = {mu: dv[mu] + (1 + s[mu]) * dp[mu] for mu in odd}
    a1, a2 = pj(dc, st.Q, odd), pj(ct, dQ, odd)
    dcQ = {mu: a1[mu] + a2[mu] for mu in odd}
    dW = {mu: dv[mu] + dcQ[mu] for mu in odd}
    b1, b2 = pj(dc, W, odd), pj(ct, dW, odd)
    dE = pj(st.P, dp, even)
    e1, e2 = pj(dc, E, even), pj(ct, dE, even)
    dG = {nu: dE[nu] + e1[nu] + e2[nu] for nu in even}
    g1, g2 = pj(dG, st.P, odd), pj(G, dp, odd)
    zP = {mu: dW[mu] * acb(0.5) for mu in odd}                                # 2z dPhi_P / 2
    z2V = {mu: (dcQ[mu] + b1[mu] + b2[mu]) * acb(0.5)
           + acb_poly([0, 1]) * (-(2 + s[mu]) * (1 + s[mu]) * dp[mu] - (1 + s[mu]) * dcQ[mu] + g1[mu] + g2[mu])
           * acb(0.5) for mu in odd}
    return zP, z2V, {nu: dG[nu] * acb(0.5) for nu in even}


def slow_value(num, drop, k, R, phi):
    """R^-k |int_0^z num(t) / t^drop dt| at z = R e^(i phi), for a polynomial num"""
    cs = down(num, drop).coeffs()
    tot = mp.mpc(0)
    for j, c in enumerate(cs):
        tot += mpc(c) * mp.expjpi((j + 1) * phi / mp.pi) * R ** (j + 1 - k) / (j + 1)
    return abs(tot)


def fast_value(num, s_mu, k, R, phi):
    """R^-k |int_0^z K_mu(t, z) num(t) / t^2 dt| at z = R e^(i phi), K_mu = e^(1/(2t) - 1/(2z)) (t/z)^((3 + 2s)/2)"""
    cs = down(num, 2).coeffs()
    h = [mpc(c) * mp.expjpi(j * phi / mp.pi) * R ** (j + 2 - k) for j, c in enumerate(cs)]
    c0 = mp.expjpi(-phi / mp.pi) / 2                  # 1/(2t) - 1/(2z) = c0 tau on the ray, Re c0 = -a
    ex = (3 + 2 * mpc(s_mu)) / 2 + 2
    jlow = next((j for j, v in enumerate(h) if v != 0), None)
    if jlow is None:
        return mp.mpf(0)

    def f(tau):
        y = 1 / (1 + R * tau)
        acc = mp.mpc(0)
        for v in reversed(h[jlow:]):
            acc = acc * y + v
        return mp.exp(c0 * tau) * mp.exp((ex + jlow) * mp.log(y)) * acc
    a = -c0.real
    return abs(mp.quad(f, [0, 1 / a, 5 / a, 20 / a, 100 / a]))


def true_columns(st, R, phi):
    """l[i][j]: the size at |z| = R of (DT eta_j)_i for eta_j = z^(k_j)"""
    comps, k = st.comps, st.k
    n = len(comps)
    ell = [[mp.mpf(0)] * n for _ in range(n)]
    for j, cj in enumerate(comps):
        mono = acb_poly([0] * st.kw[cj] + [1])
        dp = {mu: (mono if cj == ("p", mu) else Z) for mu in st.odd}
        dv = {mu: (mono if cj == ("v", mu) else Z) for mu in st.odd}
        dc = {nu: (mono if cj == ("c", nu) else Z) for nu in st.even}
        zP, z2V, dC = linearized(st, dp, dv, dc)
        for i, ci in enumerate(comps):
            if ci[0] == "p":
                ell[i][j] = slow_value(zP[ci[1]], 1, k, R, phi)
            elif ci[0] == "v":
                ell[i][j] = fast_value(z2V[ci[1]], st.s[ci[1]], k, R, phi)
            else:
                ell[i][j] = slow_value(dC[ci[1]], 0, k + 1, R, phi)
    return ell


def true_forcing(st, R, phi):
    NP, NV, NC = B.residual_numerators(st.M, st.s, st.P, st.Q, st.C, st.V, True, st.gp)
    N, k = st.N, st.k

    def tail(p, n):                                    # the orders below n vanish identically (check K1 (c))
        cs = p.coeffs()
        return acb_poly([0] * n + cs[n:])
    out = []
    for cp in st.comps:
        if cp[0] == "p":
            out.append(slow_value(tail(NP[cp[1]], N) * acb(0.5), 1, k, R, phi))
        elif cp[0] == "v":
            out.append(fast_value(tail(NV[cp[1]], N) * acb(0.5), st.s[cp[1]], k, R, phi))
        else:
            out.append(slow_value(tail(NC[cp[1]], N - 1), 0, k + 1, R, phi))
    return out


def main():
    opts = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
    M = int(opts.get("M", "5"))
    mutate = opts.get("mutate")
    mp.mp.dps = 30
    ctx.prec = 512
    st = B.Setup([acb(arb(v)) for v in ZEROS[M]], M, "0.07" if M == 5 else "0.1", "3pi/8", opts.get("sector"), None,
                 int(opts["k"]) if "k" in opts else None)
    orig_mconv, count = B.mconv, {"n": 0}
    if mutate == "nolabel":
        st.s2 = {mu: arb(0) for mu in st.odd}          # MUTATION: no (2 + s)(1 + s) P in the majorant of Phi_V
    if mutate == "gpdrop":
        def mconv(A, Bd, left, right, target):          # MUTATION: the fifth product of lipschitz() is [G P]
            count["n"] += 1
            if count["n"] % 5 == 0:
                return {key: B.MJ({}) for key in target}
            return orig_mconv(A, Bd, left, right, target)
        B.mconv = mconv
    out = B.prove(st, False)
    print(f"rv_lipschitz: M = {M}, N = {st.N}, k = {st.k}, a = {float(st.a):.5f}, mutation {mutate}; the module's "
          f"contraction constant {float(out['lam']):.4f}, proved {out['ok']}")
    R, phi = mp.mpf(st.R.mid().str(40, radius=False)), 2 * mp.mpf(st.arg.mid().str(40, radius=False))
    comps, n = st.comps, len(st.comps)
    L = [[mp.mpf(v.mid().str(30, radius=False)) for v in row] for row in out["L"]]
    ell = true_columns(st, R, phi)
    bad, zero_bad, worst, tight, pos = [], [], mp.mpf(0), 0, 0
    for i in range(n):
        for j in range(n):
            if L[i][j] == 0:
                if ell[i][j] > mp.mpf("1e-20"):
                    zero_bad.append((comps[i], comps[j], float(ell[i][j])))
                continue
            pos += 1
            r = ell[i][j] / L[i][j]
            tight += r > 0.5
            if r > worst:
                worst, at = r, (comps[i], comps[j])
            if r > 1:
                bad.append((comps[i], comps[j], float(ell[i][j]), float(L[i][j])))
    print(f"  {pos} entries of L are positive: the true derivative at eta = 0 is at most {float(worst):.4f} of the "
          f"bound (at {at[0]} <- {at[1]}), and above half of it in {tight} entries")
    print(f"  entries above their bound: {len(bad)}" + "".join(
        f"\n      {b[0]} <- {b[1]}: true {b[2]:.4g}, bound {b[3]:.4g}" for b in bad[:8]))
    print(f"  entries where L is 0 and the true derivative is not: {len(zero_bad)}" + "".join(
        f"\n      {b[0]} <- {b[1]}: true {b[2]:.3g}" for b in zero_bad[:8]))
    tf = true_forcing(st, R, phi)
    F = [mp.mpf(v.mid().str(30, radius=False)) for v in out["F"]]
    w2 = [mp.mpf(v.mid().str(30, radius=False)) for v in out["w2"]]
    fr = [tf[i] / F[i] for i in range(n) if F[i] > 0]
    wr = [tf[i] / w2[i] for i in range(n)]
    okf = max(fr) <= 1 and max(wr) <= 1
    print(f"  the forcing: |T(0)_i| at |z| = R is between {float(min(fr)):.4f} and {float(max(fr)):.4f} of F_i, and at "
          f"most {float(max(wr)):.4f} of the proved radius w2_i: {okf}")
    ok = not bad and not zero_bad and okf
    print(f"RV_LIPSCHITZ: {'PASS' if ok else 'FAIL'}")
    if mutate == "gpdrop":
        import fixed_point_M as FPM
        import fixed_point_M_b_check as CHK
        count["n"] = 0
        print("the author's Lipschitz comparison with the same mutation:")
        passed = CHK.lip(M, FPM.ZEROS[M], "0.07" if M == 5 else "0.1", None)
        print(f"  fixed_point_M_b_check.py lip: {'PASS (the mutation is not noticed)' if passed else 'FAIL (noticed)'}")
    B.mconv = orig_mconv
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

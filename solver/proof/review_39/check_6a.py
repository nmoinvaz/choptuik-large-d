#!/usr/bin/env python3
"""Independent re-evaluation of lemma 6a(5)'s final step, from the stored backward enclosure B to rho_6a, in mpmath
interval arithmetic (review of #39). No python-flint, no code under review.

Inputs: B (the enclosure of d(state at xi_a)/d(state at xi_1) of the backward flow, SysXiM variables
[D_mu, S_mu, C_nu], as saved by lemma6_jump_M.py --save-B), and the flat estimate's Lambda, weights and L
(export_6a_flat.py; the flat estimate itself is not re-implemented). Steps, from proof/A_lemmas_M.md section 1 and 4.1
and the docstring of lemma6_jump_M.py:
  1. Lambda = max_i sum_j w_i L_ij / w_j, recomputed from the exported L and weights (must not exceed the export's).
  2. Coordinates at xi_a: p = D/xi, v = S/xi^2 - (1 + i mu w) D/xi, c = C (D = xi P, S = z Q, V = Q - (1 + s) P),
     G = T B on the u-columns (S and C at xi_1, the D-components of psi vanish there), slow rows (p, c), fast rows v.
  3. G_slow^-1 enclosed by preconditioning: R ~ mid(G_slow)^-1 (mpmath, not rigorous), E = I - R G_slow,
     ||E||_inf < 1, then X = (R G_slow)^-1 in I + E X (iterated from a Neumann ball), G_slow^-1 in X R.
  4. H = G_fast G_slow^-1, rho_6a = Lambda/(1 - Lambda) max_mu w_v,mu sum_j |H_mu,j| Z^(k_j - k)/w_j, k_j = k (p),
     k + 1 (c).
w = W0 wf with W0 = 2 pi / 1.64389375 and wf the parameter ball: --centre=xhat uses the stored hier.py zero (what
lemma6_jump_M.py used, its box file having no "x"), --centre=phat the certificate's p^ (the box file's "p").

    python3 check_6a.py --B=B.json --box=A_M2_6a.json --flat=flat.json [--centre=xhat|phat] [--json=out.json]
        [--mutate=coords | partial | Bscale] [--write_B=perturbed.json]
Exit status 0 only if rho_6a < 1 (for a mutation: 0 only if the mutation is caught).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import civ as C  # noqa: E402
from civ import (Z0, CZ, ri, ri_dec, radd, rsub, rmul, rdiv, rup, rlt, rmaxup, cadd, csub, cmul, cmulr,  # noqa: E402
                 cabsup, rpi, ccontains0)
from mpmath.libmp import from_str, round_floor, round_ceiling, mpf_add, mpf_neg, mpf_mul, mpf_abs, to_float  # noqa
from mpmath.libmp import from_man_exp, from_float  # noqa: E402
import mpmath  # noqa: E402

ONE = (ri(1), Z0)


def harmonics(M):
    return [m for m in range(-M, M + 1) if m % 2], [n for n in range(-(M + 1), M + 2) if n % 2 == 0]


def r_from_dec(mid, rad):
    """a stored [mid, rad] pair of decimal strings (mid to about 106 digits, rad to 25): an interval containing the
    ball the original held (mid widened by 1e-100 |mid| for its decimal rounding, rad by 1e-20 relative)"""
    p = C.PREC
    m = (from_str(mid, p, round_floor), from_str(mid, p, round_ceiling))
    r = from_str(rad, p, round_ceiling)
    r = mpf_mul(r, from_str("1.00000000000000000001", p, round_ceiling), p, round_ceiling)
    am = mpf_add(mpf_abs(m[0]), mpf_abs(m[1]), p, round_ceiling)
    r = mpf_add(r, mpf_mul(am, from_str("1e-100", p, round_ceiling), p, round_ceiling), p, round_ceiling)
    return (mpf_add(m[0], mpf_neg(r), p, round_floor), mpf_add(m[1], r, p, round_ceiling))


def c_from_dec(p):
    return (r_from_dec(*p[0]), r_from_dec(*p[1]))


def r_from_me(q):
    return C.ri_ball(q[0], q[1])


def cnorm_row(row):
    acc = Z0
    for x in row:
        acc = radd(acc, cabsup(x))
    return rup(acc)


def matmul(A, B):
    n, m, l = len(A), len(B), len(B[0])
    out = []
    for i in range(n):
        r = []
        for j in range(l):
            acc = CZ
            for k in range(m):
                acc = cadd(acc, cmul(A[i][k], B[k][j]))
            r.append(acc)
        out.append(r)
    return out


def inv_enclosure(G):
    """an entrywise enclosure of G^-1 for every matrix in the interval matrix G, or None"""
    n = len(G)
    mp = mpmath.mp
    mp.prec = C.PREC
    Gm = mpmath.matrix(n, n)
    for i in range(n):
        for j in range(n):
            re = (mpmath.mpf(G[i][j][0][0]) + mpmath.mpf(G[i][j][0][1])) / 2
            im = (mpmath.mpf(G[i][j][1][0]) + mpmath.mpf(G[i][j][1][1])) / 2
            Gm[i, j] = mpmath.mpc(re, im)
    Rm = Gm ** -1                                                   # approximate inverse (any matrix works)
    R = [[((Rm[i, j].real._mpf_,) * 2, (Rm[i, j].imag._mpf_,) * 2) for j in range(n)] for i in range(n)]
    RG = matmul(R, G)
    E = [[csub(ONE if i == j else CZ, RG[i][j]) for j in range(n)] for i in range(n)]
    rows = [cnorm_row(E[i]) for i in range(n)]
    delta = rmaxup(rows)
    if not rlt(delta, ri(1)):
        return None, to_float(delta[1])
    # X = (R G)^-1 satisfies X = I + E X and ||X||_inf <= 1/(1 - delta): every entry of X - I is at most eps
    eps = rup(rdiv(delta, rsub(ri(1), delta)))
    box = lambda c, e: (C.rinflate(c[0], e[1]), C.rinflate(c[1], e[1]))  # noqa: E731
    X = [[box(ONE if i == j else CZ, eps) for j in range(n)] for i in range(n)]
    for _ in range(4):
        EX = matmul(E, X)
        X = [[cadd(ONE if i == j else CZ, EX[i][j]) for j in range(n)] for i in range(n)]
    return matmul(X, R), to_float(delta[1])


def main():
    opts = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
    C.setprec(256)
    mutate = opts.get("mutate")
    M = 5
    odd, even = harmonics(M)
    no, ne = len(odd), len(even)
    nd = 2 * no + ne
    with open(opts["flat"]) as fh:
        fl = json.load(fh)
    with open(opts["box"]) as fh:
        bx = json.load(fh)
    with open(opts["B"]) as fh:
        sb = json.load(fh)
    assert sb["M"] == M and sb["xi1"] == bx["xi1"] and sb["xia"] == fl["xia"] and int(bx["M"]) == M
    k = fl["k"]
    Z = ri_dec(fl["Z"])
    xa = ri_dec(fl["xia"])
    # 1. Lambda from L and the weights
    wts = [(from_float(v),) * 2 for v in fl["wts"]]
    L = [[r_from_me(q) for q in row] for row in fl["L"]]
    lam_rows = []
    for i in range(len(L)):
        acc = Z0
        for j in range(len(L)):
            acc = radd(acc, rdiv(rmul(wts[i], L[i][j]), wts[j]))
        lam_rows.append(acc)
    lam = rmaxup(lam_rows)
    lam_exp = r_from_me(fl["lam"])
    lam_ok = bool(C.rle(lam, rup(lam_exp))) or abs(to_float(lam[1]) - to_float(lam_exp[1])) < 1e-15
    if not rlt(lam, ri(1)):
        print("Lambda >= 1")
        return 2
    # w = W0 wf
    centre = opts.get("centre", "xhat")
    xs = fl["x_hat"] if centre == "xhat" else bx["p"]
    W0 = rdiv(rmul(ri(2), rpi()), ri_dec(fl["W0_str"]))
    h = ri_dec(bx["h"])
    wf = C.rinflate(ri_dec(xs[1]), h[1])
    w = (rmul(W0, wf), Z0)
    # B
    B = [[c_from_dec(sb["B"][i][j]) for j in range(nd)] for i in range(nd)]
    if mutate == "Bscale":
        # MUTATION: the S rows of B (the fast components at xi_a) times 5
        B = [[cmulr(B[i][j], ri(5)) if no <= i < 2 * no else B[i][j] for j in range(nd)] for i in range(nd)]
    if "write_B" in opts:
        # the (perturbed) B in the save format, for lemma6_jump_M.py --load-B
        def enc(x):
            def one(r):
                m = (mpmath.mpf(r[0]) + mpmath.mpf(r[1])) / 2
                rad = (mpmath.mpf(r[1]) - mpmath.mpf(r[0])) / 2 * (1 + mpmath.mpf("1e-15"))
                return [mpmath.nstr(m, 110, strip_zeros=False), mpmath.nstr(rad, 25)]
            return [one(x[0]), one(x[1])]
        out = dict(sb)
        out["B"] = [[enc(B[i][j]) for j in range(nd)] for i in range(nd)]
        out["note"] = f"B of {opts['B']} with mutation {mutate}"
        with open(opts["write_B"], "w") as fh:
            json.dump(out, fh)
    # 2. T at xi_a and G = T B on the u-columns
    I = (Z0, ri(1))
    sgn = -1 if mutate == "coords" else 1                     # MUTATION coords: (1 - i mu w)
    ixa = rdiv(ri(1), xa)
    ixa2 = rmul(ixa, ixa)
    T = [[CZ] * nd for _ in range(nd)]
    for r_, mu in enumerate(odd):
        T[r_][r_] = (ixa, Z0)
        T[no + r_][no + r_] = (ixa2, Z0)
        T[no + r_][r_] = cmulr(C.cneg(cadd(ONE, cmulr(cmul(I, w), ri(sgn * mu)))), ixa)
    for r_ in range(ne):
        T[2 * no + r_][2 * no + r_] = ONE
    G = matmul(T, B)
    ucols = list(range(no, nd))
    if mutate == "partial":
        ucols = [i for i, m in enumerate(odd) if abs(m) >= 3] + ucols   # MUTATION partial: only J_D,1 = 0 assumed
    slow = list(range(no)) + list(range(2 * no, nd))
    fast = list(range(no, 2 * no))
    Gs = [[G[i][j] for j in ucols] for i in slow]
    Gf = [[G[i][j] for j in ucols] for i in fast]
    res = {"centre": centre, "mutate": mutate, "Lambda_recomputed": to_float(lam[1]), "Lambda_export": fl["lam_str"],
           "Lambda_le_export": lam_ok}
    if len(Gs) != len(Gs[0]):
        print(f"G_slow is {len(Gs)} x {len(Gs[0])}: not square, beta not determined; lemma 6a NOT established"
              + (f"   [MUTATION {mutate}: caught]" if mutate else ""))
        res.update(ok=False, why="G_slow not square")
        if "json" in opts:
            with open(opts["json"], "w") as fh:
                json.dump(res, fh, indent=1)
        return 0 if mutate else 2
    Gsi, delta = inv_enclosure(Gs)
    if Gsi is None:
        print(f"G_slow^-1 not enclosed (||I - R G_slow|| = {delta:.3g}); lemma 6a NOT established")
        res.update(ok=False, why="G_slow inverse", delta=delta)
        return 0 if mutate else 2
    H = matmul(Gf, Gsi)
    # 3. rho_6a
    cidx = {tuple(c): i for i, c in enumerate(fl["comps"])}
    slow_comps = [("p", m) for m in odd] + [("c", n) for n in even]
    kj = [k if t == "p" else k + 1 for (t, _) in slow_comps]
    ws = [wts[cidx[cp]] for cp in slow_comps]
    wv = [wts[cidx[("v", m)]] for m in odd]
    Zp = [ri(1), Z]
    rowv = []
    for r_ in range(no):
        acc = Z0
        for j in range(len(slow_comps)):
            acc = radd(acc, rdiv(rmul(cabsup(H[r_][j]), Zp[kj[j] - k]), ws[j]))
        rowv.append(rmul(wv[r_], acc))
    rho = rmul(rdiv(lam, rsub(ri(1), lam)), rmaxup(rowv))
    ok = bool(rlt(rho, ri(1)))
    hmax = max(to_float(cabsup(H[i][j])[1]) for i in range(no) for j in range(len(slow_comps)))
    gmax = max(to_float(cabsup(Gsi[i][j])[1]) for i in range(len(Gsi)) for j in range(len(Gsi)))
    bmax = max(C.crad(B[i][j]) for i in range(nd) for j in range(nd))
    res.update(ok=ok, rho6a_lo=to_float(rho[0]), rho6a_hi=to_float(rho[1]), H_max=hmax, Gslow_inv_max=gmax,
               delta=delta, B_maxrad=bmax)
    print(f"Lambda recomputed from L: {to_float(lam[1]):.12f} (export {fl['lam_str']}, not above it: {lam_ok}); "
          f"w centre {centre}; max radius of B {bmax:.3g}; ||I - R G_slow|| <= {delta:.3g}; |G_slow^-1| <= {gmax:.4g}; "
          f"|H| <= {hmax:.4g}")
    print(f"rho_6a in [{to_float(rho[0]):.12f}, {to_float(rho[1]):.12f}] < 1: {ok}"
          + (f"   [MUTATION {mutate}]" if mutate else ""))
    if "json" in opts:
        with open(opts["json"], "w") as fh:
            json.dump(res, fh, indent=1)
    if mutate:
        return 0 if not ok else 3
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())

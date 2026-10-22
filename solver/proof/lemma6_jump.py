#!/usr/bin/env python3
"""Lemma 6a (proof/A_certify.md section 11): J_D(xi1) = 0 forces the whole Stokes jump to vanish. Rigorous (arb).

psi = y+ - y- on the real axis, y-(x) = sigma(y+(x)) (sigma: mu -> -mu and complex conjugation). If
J_D(xi1) = D+_{+1}(xi1) - conj(D+_{-1}(xi1)) = 0, then both D components of psi(xi1) vanish
(psi_{D,+1} = J_D, psi_{D,-1} = -conj(J_D)), so psi(xi1) = (0, 0, u) with u = (psi_S, psi_C) in C^5.

1. Backward flow. The C^1 Lohner integrator of validated_int.py (unchanged) runs along the real axis from xi1 = 0.3
   down to xi_a, from a box containing y+(xi1; x) and y-(xi1; x) = sigma(y+(xi1; x)) for every x in X (the state set of
   the lower-path box run, lemma6_path.py), with the 7 columns of the identity as variational columns. The result B
   encloses D Phi(y) for every y in that box, Phi the flow xi1 -> xi_a, so by the mean value theorem
   psi(xi_a) = Phi(y+) - Phi(y-) in B psi(xi1). (Backward along the real axis the fast modes decay, by about e^-106.)
2. Coordinates at xi_a: p = D/xi, v = S/xi^2 - (1 + i mu w) D/xi, c = C (those of fixed_point.py), G = T_a B (columns
   of u only), split into the 5 slow rows (p, c) and the 2 fast rows (v): (p, c)(xi_a) = G_slow u, v(xi_a) = G_fast u.
3. The flat estimate (lemma6_sector.py, part R) at Z = xi_a^2 gives, with beta = v_psi(Z) and eps = ||psi - b||:
   eps <= Lambda/(1 - Lambda) Z^-k max_mu w_{v,mu} |beta_mu|, |p_mu(Z)| <= Z^k eps / w_{p,mu}, |c_nu(Z)| <= Z^(k+1) eps / w_{c,nu}.
   If G_slow is invertible (every matrix in the ball), beta = G_fast G_slow^-1 (p, c)(Z), hence
       max_mu w_v |beta_mu| <= rho_6a max_mu w_v |beta_mu|,
       rho_6a = Lambda/(1 - Lambda) max_mu w_{v,mu} sum_j |H_{mu j}| Z^(k_j - k) / w_j,   H = G_fast G_slow^-1.
   rho_6a < 1 gives beta = 0, then psi = 0 on (0, Z] (psi = b + T0 psi with b = 0 and ||T0|| < 1), so y+ = y-.

    python3 lemma6_jump.py --box=lemma6_path_box_1e-9.json [--xia=0.067] [--mutate=coords|swap]
    python3 lemma6_jump.py --point=lemma6_path_point.json     (the same at x^ only: a quicker preview)
    mutations: swap   = y- taken as conj(y+) without the reflection mu -> -mu (a wrong mode labelling); then y-(xi1) is
                        not a solution near y+(xi1) and the check must fail;
               coords = the slow/fast change of variables at xi_a with (1 - i mu w) instead of (1 + i mu w) (a wrong
                        mode normalization). This one is NOT caught (rho_6a = 0.12 < 1): the wrong v differs from the
                        right one by 2 i mu w p, and p(Z) is itself bounded by the flat estimate, so the error is of
                        second order; recorded as a limitation of the harness, see section 11.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import validated_int as VI  # noqa: E402
import lemma6_common  # noqa: E402
lemma6_common.install()            # a-priori inflation with a common floor (see lemma6_common.py)
import lemma6_sector as L6S  # noqa: E402
from flint import arb, acb, acb_mat, ctx  # noqa: E402

AH, WH = "2.5421480475", "0.7769940136"
SIG = [1, 0, 3, 2, 6, 5, 4]              # sigma on [D-1, D+1, S-1, S+1, C-2, C0, C+2]


def parse(p):
    return acb(arb(p[0]), arb(p[1]))


def start_box(y, w_box, A_box, mutate=None):
    """hull of y and sigma(y), plus w and A"""
    ys = [y[SIG[i]].conjugate() for i in range(7)]
    if mutate == "swap":
        ys = [y[i].conjugate() for i in range(7)]
    H = [y[i].union(ys[i]) for i in range(7)]
    return H + [acb(w_box), acb(A_box)]


def backward(H, xia, verbose=True):
    s = VI.Sys(gravity=True)
    n, m = s.n, 7
    c = [acb(z.mid()) for z in H]
    q = [H[i] - c[i] for i in range(n)]
    Q = acb_mat([[1 if i == j else 0 for j in range(n)] for i in range(n)])
    cV = acb_mat([[1 if i == j else 0 for j in range(m)] for i in range(n)])
    qV = acb_mat(n, m)
    lo = VI.Lohner(s, c, Q, q, 24, var=(cV, Q, qV))
    t0 = time.time()
    steps = lo.run(VI.Leg("real", "0.3", xia), 1e-3, verbose)
    V = lo.var_box()
    X = lo.box()
    return V, X, steps, time.time() - t0


def main():
    opts = {a.split("=", 1)[0][2:]: a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--")}
    xia = opts.get("xia", "0.067")
    mutate = opts.get("mutate")
    ctx.prec = 320
    from decimal import Decimal
    Z = str(Decimal(xia) ** 2)                                   # exact decimal square
    box = "0"
    if "box" in opts:
        with open(opts["box"]) as fh:
            d = json.load(fh)
        y = [parse(p) for p in d["yX_xi1"]]
        box = d["h"]
    else:
        with open(opts["point"]) as fh:
            d = json.load(fh)
        y = [parse(p) for p in d["y_xi1"]]
    # lemma S and the flat estimate over the same parameter box
    S = L6S.lemma_S(L6S.RP, L6S.KW, box, verbose=False)
    R = L6S.flat_estimate(S, Z, verbose=False)
    if not (S["ok"] and R["ok"]):
        print("lemma S or the flat estimate failed")
        return 1
    k, lam, wts, comps = R["k"], R["lam"], R["wts"], R["comps"]
    print(f"lemma S: contraction {float(S['lip'].upper()):.4f}; flat estimate on 0 < z <= Z = {Z}: Lambda = "
          f"{float(lam.upper()):.4f}, k = {k}")
    ctx.prec = 320
    wf = arb(WH, arb(box)) if float(box) > 0 else arb(WH)
    W0 = 2 * arb.pi() / arb("1.64389375")
    w = W0 * wf
    A_box = arb(AH, arb(box)) if float(box) > 0 else arb(AH)
    ctx.prec = VI.PREC
    H = start_box(y, w, A_box, mutate)
    print(f"start box at xi1 = 0.3 (y+ and sigma(y+) over the parameter box {box}): radius "
          f"{max(VI.rad_of(z) for z in H[:7]):.2e}")
    try:
        V, X, steps, sec = backward(H, xia, verbose=False)
    except RuntimeError as e:
        print(f"backward integration failed ({e}): lemma 6a NOT established" +
              (f"   [MUTATION {mutate}: failure expected]" if mutate else ""))
        return 2
    print(f"backward along the real axis 0.3 -> {xia}: {steps} steps, {sec:.0f} s; final state radius "
          f"{max(VI.rad_of(z) for z in X[:7]):.2e}; max radius of B {max(VI.rad_of(V[i, j]) for i in range(7) for j in range(7)):.2e}")
    ctx.prec = 320
    xa = acb(arb(xia))
    I = acb(0, 1)
    # T_a: rows [p-1, p+1, v-1, v+1, c-2, c0, c+2] from columns [D-1, D+1, S-1, S+1, C-2, C0, C+2]
    T = acb_mat(7, 7)
    for r, mu in enumerate((-1, 1)):
        sgn = -1 if mutate == "coords" else 1
        T[r, r] = 1 / xa
        T[2 + r, 2 + r] = 1 / (xa * xa)
        T[2 + r, r] = -(1 + sgn * I * mu * acb(w)) / xa
    for r in range(3):
        T[4 + r, 4 + r] = acb(1)
    B = acb_mat([[V[i, j] for j in range(7)] for i in range(7)])
    G = T * B
    ucols = [2, 3, 4, 5, 6]
    slow_rows, fast_rows = [0, 1, 4, 5, 6], [2, 3]
    Gs = acb_mat([[G[i, j] for j in ucols] for i in slow_rows])
    Gf = acb_mat([[G[i, j] for j in ucols] for i in fast_rows])
    try:
        Gsi = Gs.inv()
    except ZeroDivisionError:
        print("G_slow: invertibility not proved")
        return 1
    Hm = Gf * Gsi
    kj = [k, k, k + 1, k + 1, k + 1]                       # weight exponents of the slow rows (p-1, p+1, c-2, c0, c+2)
    ws = [wts[i] for i in slow_rows]
    wv = [wts[2], wts[3]]
    Za = arb(Z)
    rows = []
    for r in range(2):
        acc = arb(0)
        for j in range(5):
            acc = acc + abs(Hm[r, j]) * Za ** (kj[j] - k) / arb(ws[j])
        rows.append(arb(wv[r]) * acc)
    rho6a = lam / (1 - lam) * arb(max(x.upper() for x in rows))
    sv = [float(abs(Hm[r, j]).upper()) for r in range(2) for j in range(5)]
    print(f"|H| = |G_fast G_slow^-1| entries <= {max(sv):.2e}; G_slow^-1 entries <= "
          f"{max(float(abs(Gsi[i, j]).upper()) for i in range(5) for j in range(5)):.2e}")
    ok = bool(rho6a < 1)
    print(f"rho_6a = {float(rho6a.upper()):.3e} < 1: {ok}  => J_D(0.3) = 0 implies psi = 0 (whole jump): {ok}"
          + (f"   [MUTATION {mutate}]" if mutate else ""))
    out = {"xia": xia, "Z": Z, "box": box, "Lambda": lam.str(10), "rho6a": rho6a.str(10), "ok": ok, "steps": steps,
           "sec": sec, "H_max": max(sv), "mutate": mutate}
    if "json" in opts:
        with open(opts["json"], "w") as fh:
            json.dump(out, fh, indent=1)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())

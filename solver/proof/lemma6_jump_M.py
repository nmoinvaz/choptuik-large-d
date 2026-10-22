#!/usr/bin/env python3
"""Lemma 6a for M harmonics (proof/A_lemmas_M.md section 4): if every D-component of the jump vanishes at xi1,
    J_{D,mu}(xi1) = D+_mu(xi1) - conj(D+_{-mu}(xi1)) = 0   for mu = 1, 3, .., M,
then the whole jump vanishes, psi = y+ - y- = 0, i.e. every Stokes multiplier S_1, .., S_M is zero. Rigorous (arb),
given a box B1 that contains y+(xi1; x) for every x in the parameter box X.

psi = y+ - sigma(y+) on the real axis (sigma: mu -> -mu with complex conjugation). If all J_{D,mu}(xi1) = 0 then
psi(xi1) = (0, u), u = (psi_S, psi_C) in C^{(M+1) + (M+2)}.
1. Backward flow: the C^1 Lohner integrator of validated_int.py (unchanged), on lemma6_usys_M.SysXiM, along the real
   axis from xi1 to xi_a, from the hull of B1 and sigma(B1), with the identity as variational columns: B encloses
   D Phi(y) for every y in the hull (convex, so it contains the segment from y- to y+), and psi(xi_a) in B psi(xi1).
2. Coordinates at xi_a: p = D/xi, v = S/xi^2 - (1 + i mu w) D/xi, c = C; G = T_a B on the u-columns, split into the
   2M + 3 slow rows (p, c) and the M + 1 fast rows (v). G_slow is square, (2M+3) x (2M+3).
3. The flat estimate (lemma6_sector_M.py, part R) at Z = xi_a^2: with beta = v_psi(Z),
       rho_6a = Lambda/(1 - Lambda) max_mu w_{v,mu} sum_j |H_{mu j}| Z^(k_j - k) / w_j,   H = G_fast G_slow^-1,
   and rho_6a < 1 forces beta = 0, then psi = 0 on (0, Z], so y+ = y-.
What changes with M: only the dimensions. On the real axis the fast modes of every label decay alike, |xi^{-2 s_mu}| = 1.
What does not carry over: the number of conditions. One complex D-component suffices at M = 1; at M harmonics all
M + 1 complex D-components (the J_{D,mu}, mu > 0) are needed, because u must have as many entries as the slow rows
(--mutate=partial drops J_{D,3} and must fail).

Input (from the point certificate's box run, the analogue of lemma6_path_box_1e-9.json):
    --box=f.json with keys "M", "h", "xi1", "yX_xi1" (the nd = 2(M+1) + (M+2) state components at xi1 as [re, im]
    arb strings, in lateral_np's order [D_mu ascending, S_mu, C_nu]) and optionally "x" (the box centre, decimal
    strings; default the stored hier.py zero). The parameter box is x +- h (the axis-aligned hull of a tilted box is
    fine); lemma S and the flat estimate are proved over x^ +- hbox (--hbox, default 1e-6), which must contain it.
PILOT (NOT rigorous): --pilot=1e-10 builds B1 from a double-precision y+(xi1) at the hier.py zero (lateral_np's path:
    the ray arg xi = 3pi/8 from 0.085 to rho, the arc |xi| = rho to the real axis, then the real axis to xi1),
    inflated by the given radius. It measures the cost and rho_6a; it proves nothing.

    python3 lemma6_jump_M.py --M=3 --box=f.json [--xia=0.067] [--rp=0.1] [--k=20] [--json=out.json]
    python3 lemma6_jump_M.py --M=3 --pilot=1e-10 [--xi1=0.2] [--rho=0.2] [--save-B=b.json | --load-B=b.json]
        --save-B stores the backward enclosure B (and the final state box); --load-B reuses it, for the mutations that
        change only the post-processing (partial, coords)
    mutations: --mutate=swap    y- = conj(y+) without the reflection mu -> -mu: must fail
               --mutate=partial only J_{D,1} = 0 assumed (psi_{D,+-3} free): G_slow is not square, must fail
               --mutate=coords  (1 - i mu w) in the change of variables at xi_a: at M = 1 not caught (second order,
                                A_certify.md 11.6); reported
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import validated_int as VI  # noqa: E402
import lemma6_common  # noqa: E402
lemma6_common.install()            # a-priori inflation with a common floor (see lemma6_common.py)
import lemma6_sector_M as L6SM  # noqa: E402
import lemma6_usys_M as UM  # noqa: E402
import fixed_point_M as FPM  # noqa: E402
from flint import arb, acb, acb_mat, ctx  # noqa: E402


def parse(p):
    return acb(arb(p[0]), arb(p[1]))


def sigma_index(M):
    odd, even = UM.harmonics(M)
    no = len(odd)
    idx = []
    for m in odd:
        idx.append(odd.index(-m))
    for m in odd:
        idx.append(no + odd.index(-m))
    for n in even:
        idx.append(2 * no + even.index(-n))
    return idx


def start_box(M, y, w_box, A_box, mutate=None):
    """hull of y and sigma(y), plus w and A"""
    sig = sigma_index(M)
    nd = len(sig)
    ys = [y[sig[i]].conjugate() for i in range(nd)]
    if mutate == "swap":
        ys = [y[i].conjugate() for i in range(nd)]
    return [y[i].union(ys[i]) for i in range(nd)] + [acb(w_box), acb(A_box)]


def backward(M, H, xi1, xia, verbose=False):
    s = UM.SysXiM(M)
    n, m = s.n, s.nd
    c = [acb(z.mid()) for z in H]
    q = [H[i] - c[i] for i in range(n)]
    Q = acb_mat([[1 if i == j else 0 for j in range(n)] for i in range(n)])
    cV = acb_mat([[1 if i == j else 0 for j in range(m)] for i in range(n)])
    qV = acb_mat(n, m)
    lo = VI.Lohner(s, c, Q, q, int(os.environ.get("LOHNER_ORDER", "24")), var=(cV, Q, qV))
    t0 = time.time()
    steps = lo.run(VI.Leg("real", xi1, xia), 1e-3, verbose)
    return lo.var_box(), lo.box(), steps, time.time() - t0


def pilot_state(M, xi1, rho, rad):
    """NOT rigorous: y+(xi1) at the hier.py zero in double precision (lateral_np), inflated by rad"""
    sys.path.insert(0, os.path.join(HERE, "..", "inner_stokes"))
    import lateral_np as LNP
    import inner_np
    x = np.array([float(v) for v in FPM.ZEROS[M]])
    Fh = inner_np.Fh_from(x, M)
    w = LNP.W0 * x[1]
    r0 = 0.085
    phi0 = 3 * np.pi / 8
    Y0, jt, err = LNP.series_data(Fh, w, M, r0 * np.exp(1j * phi0))
    nm = LNP.NP(M, w)
    segs = [("ray", phi0, r0, rho), ("arc", rho, phi0, 0.0)]
    if xi1 > rho:
        segs.append(("real", rho, xi1))
    Y = nm.path(Y0, segs, rtol=1e-13, atol=1e-16)
    return [acb(arb(v.real, rad), arb(v.imag, rad)) for v in Y]


def main():
    opts = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
    M = int(opts.get("M", "3"))
    xia = opts.get("xia", "0.067")
    mutate = opts.get("mutate")
    rp, k = opts.get("rp", "0.1"), int(opts.get("k", "20"))
    ctx.prec = 320
    from decimal import Decimal
    Z = str(Decimal(xia) ** 2)
    pilot = "pilot" in opts
    if "x" in opts:
        # another centre (decimal strings), e.g. the point certificate's p^, which is closer to the zero than the stored
        # double-precision zero; lemma S and the flat estimate are then over that centre +- hbox
        FPM.ZEROS[M] = opts["x"].split(",")
    if pilot:
        xi1 = opts.get("xi1", "0.2")
        rho = float(opts.get("rho", "0.2"))
        y = pilot_state(M, float(xi1), rho, float(opts["pilot"]))
        box = opts.get("hbox", "1e-6")
        print(f"PILOT (not rigorous): B1 = double-precision y+({xi1}) at the hier.py zero (path: ray 3pi/8 to {rho}, "
              f"arc, real axis to {xi1}) inflated by {opts['pilot']}; lemma S and the flat estimate over x^ +- {box}")
        xs = FPM.ZEROS[M]
        hpar = box
    else:
        with open(opts["box"]) as fh:
            d = json.load(fh)
        assert int(d.get("M", M)) == M, "box file is for another M"
        y = [parse(p) for p in d["yX_xi1"]]
        xi1 = str(d.get("xi1", "0.3"))
        # the certificate's box: centre d["x"] (default: the stored zero) +- d["h"] in every parameter (the axis-aligned
        # hull of a tilted box is fine). Lemma S and the flat estimate are over x^ +- hbox (default 1e-6), which must
        # contain it.
        xs, hpar = d.get("x", FPM.ZEROS[M]), d["h"]
        box = opts.get("hbox", "1e-6")
        ctx.prec = 320
        inside = all(bool(abs(arb(a) - arb(b)) + arb(hpar) <= arb(box)) for a, b in zip(xs, FPM.ZEROS[M]))
        if not inside:
            print(f"the certificate's box is not inside x^ +- {box}: rerun with a larger --hbox")
            return 1
    S = L6SM.lemma_S_M(M, rp, k, box, verbose=False)
    R = L6SM.flat_estimate_M(S, Z, diag=False, verbose=False)
    if not (S["ok"] and R["ok"]):
        print("lemma S or the flat estimate failed")
        return 1
    lam, wts, comps = R["lam"], R["wts"], R["comps"]
    print(f"lemma S (M = {M}, rp = {rp}, k = {k}): contraction {float(S['lip'].upper()):.4f}; flat estimate on "
          f"0 < z <= Z = {Z}: Lambda = {float(lam.upper()):.4f}")
    ctx.prec = 320
    wf = arb(xs[1], arb(hpar)) if float(hpar) > 0 else arb(xs[1])
    w = FPM.W0() * wf
    A_box = arb(xs[0], arb(hpar)) if float(hpar) > 0 else arb(xs[0])
    ctx.prec = VI.PREC
    H = start_box(M, y, w, A_box, mutate)
    nd = len(y)
    print(f"start box at xi1 = {xi1} (y+ and sigma(y+)): radius {max(VI.rad_of(z) for z in H[:nd]):.2e}", flush=True)
    try:
        if "load-B" in opts:
            # the saved enclosure of a clean run with the same start box (the post-processing mutations reuse it)
            with open(opts["load-B"]) as fh:
                sb = json.load(fh)
            assert sb["M"] == M and sb["xi1"] == xi1 and sb["xia"] == xia and mutate != "swap"
            V = acb_mat([[parse(sb["B"][i][j]) for j in range(nd)] for i in range(nd)])
            X, steps, sec = [parse(p) for p in sb["X"]], sb["steps"], 0.0
        else:
            V, X, steps, sec = backward(M, H, xi1, xia, verbose="verbose" in opts)
            if "save-B" in opts:
                def sj(z):
                    return [z.real.str(80, radius=True), z.imag.str(80, radius=True)]
                with open(opts["save-B"], "w") as fh:
                    json.dump({"M": M, "xi1": xi1, "xia": xia, "steps": steps, "sec": sec,
                               "X": [sj(z) for z in X[:nd]], "B": [[sj(V[i, j]) for j in range(nd)] for i in range(nd)]},
                              fh)
    except RuntimeError as e:
        print(f"backward integration failed ({e}): lemma 6a NOT established"
              + (f"   [MUTATION {mutate}: failure expected]" if mutate else ""))
        return 2 if not mutate else (0 if mutate == "swap" else 2)
    print(f"backward along the real axis {xi1} -> {xia}: {steps} steps, {sec:.0f} s; final state radius "
          f"{max(VI.rad_of(z) for z in X[:nd]):.2e}; max radius of B "
          f"{max(VI.rad_of(V[i, j]) for i in range(nd) for j in range(nd)):.2e}", flush=True)
    ctx.prec = 320
    odd, even = UM.harmonics(M)
    no, ne = len(odd), len(even)
    xa = acb(arb(xia))
    I = acb(0, 1)
    T = acb_mat(nd, nd)
    for r, mu in enumerate(odd):
        sgn = -1 if mutate == "coords" else 1
        T[r, r] = 1 / xa
        T[no + r, no + r] = 1 / (xa * xa)
        T[no + r, r] = -(1 + sgn * I * mu * acb(w)) / xa
    for r in range(ne):
        T[2 * no + r, 2 * no + r] = acb(1)
    B = acb_mat([[V[i, j] for j in range(nd)] for i in range(nd)])
    G = T * B
    ucols = list(range(no, nd))
    if mutate == "partial":
        # only J_{D,+-1} = 0 assumed: psi_{D,mu}(xi1) with |mu| >= 3 is free and joins u
        ucols = [i for i, m in enumerate(odd) if abs(m) >= 3] + ucols
    slow_rows = list(range(no)) + list(range(2 * no, nd))
    fast_rows = list(range(no, 2 * no))
    Gs = acb_mat([[G[i, j] for j in ucols] for i in slow_rows])
    Gf = acb_mat([[G[i, j] for j in ucols] for i in fast_rows])
    if Gs.nrows() != Gs.ncols():
        print(f"G_slow is {Gs.nrows()} x {Gs.ncols()}: not square, beta is not determined by the slow data; lemma 6a "
              f"NOT established" + (f"   [MUTATION {mutate}: failure expected]" if mutate else ""))
        return 0 if mutate == "partial" else 2
    try:
        Gsi = Gs.inv()
    except ZeroDivisionError:
        print("G_slow: invertibility not proved; lemma 6a NOT established")
        return 2
    Hm = Gf * Gsi
    # weight exponents of the slow rows: p -> k, c -> k + 1; weights of the flat estimate
    cidx = {cp: i for i, cp in enumerate(comps)}
    slow_comps = [("p", m) for m in odd] + [("c", n) for n in even]
    kj = [k if t == "p" else k + 1 for (t, _) in slow_comps]
    ws = [wts[cidx[cp]] for cp in slow_comps]
    wv = [wts[cidx[("v", m)]] for m in odd]
    Za = arb(Z)
    rowv = []
    for r in range(no):
        acc = arb(0)
        for j in range(len(slow_comps)):
            acc = acc + abs(Hm[r, j]) * Za ** (kj[j] - k) / arb(ws[j])
        rowv.append(arb(wv[r]) * acc)
    rho6a = lam / (1 - lam) * arb(max(x.upper() for x in rowv))
    hmax = max(float(abs(Hm[r, j]).upper()) for r in range(no) for j in range(len(slow_comps)))
    gsi = max(float(abs(Gsi[i, j]).upper()) for i in range(Gsi.nrows()) for j in range(Gsi.ncols()))
    ok = bool(rho6a < 1)
    print(f"|H| = |G_fast G_slow^-1| entries <= {hmax:.2e}; G_slow^-1 entries <= {gsi:.2e}")
    print(f"rho_6a = {float(rho6a.upper()):.3e} < 1: {ok}  => all J_D,mu({xi1}) = 0 implies psi = 0 (whole jump): {ok}"
          + (f"   [MUTATION {mutate}]" if mutate else "") + ("   [PILOT: not rigorous]" if pilot else ""))
    out = {"M": M, "xi1": xi1, "xia": xia, "Z": Z, "box": box, "pilot": opts.get("pilot"), "Lambda": lam.str(10),
           "rho6a": rho6a.str(10), "ok": ok, "steps": steps, "sec": sec, "H_max": hmax, "Gslow_inv_max": gsi,
           "mutate": mutate}
    if "json" in opts:
        with open(opts["json"], "w") as fh:
            json.dump(out, fh, indent=1)
    if mutate == "swap":
        return 0 if not ok else 3
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())

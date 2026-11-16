#!/usr/bin/env python3
"""Comparisons of fixed_point_M_b.py (the second implementation of the start-value contraction, issue #427) with the
first one, fixed_point.py and fixed_point_M.py. This file is a harness: it imports both. The proof code,
fixed_point_M_b.py, imports neither.

    python3 fixed_point_M_b_check.py m1
        planted: M = 1 at fixed_point.py's zero, r0 = 0.1. The second implementation's enclosures of
        (D, S, C)(xi0), xi0 = 0.1 e^{3 i pi/8}, must overlap fixed_point.py's start values in all 7 components.
    python3 fixed_point_M_b_check.py lip [--M=5] [--x=...] [--r0=0.07] [--sector=0.2]
        the two Lipschitz matrices entry by entry, at the first implementation's N, k and ball, with the kernel
        bound of A_certify.md 4.4 (--kernel=classic) in the second. Criteria, written before the first run:
          (a) every entry of the second is <= the first's (1 + 1e-9): its majorants keep each power of t and
              integrate term by term, the first collapses them to one power, so the second can only be smaller;
          (b) every entry of the first above 1e-6 is matched to within 20% (second >= 0.8 first): a term missing
              from either matrix would show as a larger gap;
          (c) the same for the forcing in the first's weighted norm.
    python3 fixed_point_M_b_check.py cert --box=data/A_M2_box_M5_1e-10.json [--second] [--N=..]
        the start data of a stored point certificate (issue #866). The certificate's four parameter sets (its centre
        p^, the hull of its box, and the Cauchy box of each) are rebuilt by start_cover_M2.cert_data. The first
        implementation is run by the certificate's own calls of fixed_point_M.start_data_M (as start_cover_M2 --rerun
        does), the second by fixed_point_M_b.start_data. Criteria, written before the first run: the second proves
        the contraction on all four sets; every start value at p^, every derivative entry at p^ and every derivative
        entry over the hull overlaps the first's, per component (and the values over the hull, which the certificate
        does not use). --second (kind box2): also the second-order jets of Y_N over the hull along the chart's
        directions, by the second implementation's recursion on series of length 3, against certify_M2.series2.
    python3 fixed_point_M_b_check.py cert2 --box=data/A_M2_box_M5_1e-10.json
        the C^2 start data of a second-order box run (issue #869): certify_M2.start_data2's values at p^, its
        first-derivative columns V0_k = Dy+(xi0; p^) T_k and its 21 second-derivative columns
            W0_jk = D^2 Y_N(p(x))[d_j, d_k] + (Cauchy bound of D^2 eta) + Dy+ Gamma_jk     over the box,
        against the same three assembled from the second implementation: D^2 Y_N from its recursion on series of
        length 3 over the hull, the Cauchy bound e_q (2 nu / (c - hw))^2 with e_q its own bound of eta on the Cauchy
        box at p^, and the Gamma term from its own derivatives over the hull. The chart geometry (T, Gamma, the
        directions d_k, the pair list) is the certificate's. Criteria, written before the first run: the four
        contractions hold and every entry overlaps (values, V0, and 21 pairs x nd components of W0).
    python3 fixed_point_M_b_check.py cli
        the plain command line (issue #878): python3 fixed_point_M_b.py --M=5 --x=<the M = 5 certificate's centre>,
        run as a subprocess, must print the N, k and contraction constant that Setup and prove give for the point
        read at 512 bits. (Read at python-flint's default 53 bits the decimals are balls of radius 6e-17, and the
        lemma is then proved on that ball with another truncation.)
    python3 fixed_point_M_b_check.py box2 --box=data/A_M2_box_M5_1e-10.json --out=new.json [--log=f] [--pilot=N]
        the second-order box run of that certificate again, from the second implementation's start data (issue #880):
        certify_M2.box2_run with the stored chart, h and w, its start_data2 replaced for this call by the assembly that
        cert2 compares. certify_M2.py is not edited. About 40 min at M = 5; --pilot stops after N steps.
    python3 fixed_point_M_b_check.py kraw --box=new.json --stored=data/A_M2_box_M5_1e-10.json [--sub=1e-18,1e-26,1e-27]
        the Krawczyk test on the new run, on its whole box and on the sub-box, with the relative margins beside the
        stored run's; the two enclosures of the zero must overlap. PASS only if the test holds on the new run.
    mutations (each must make its check fail, exit status 3):
        m1 --mutate=coef | sign | scale     as in fixed_point_M_b.py
        box2 --mutate=sign                  the run from start data with the sign of [G P] flipped; kraw on it must fail
        cli --mutate=read53                 the comparison made with the point read at 53 bits: must differ
        cert2 --mutate=w0 | sign            the second implementation's D^2 Y_N dropped; the sign of [G P]
        lip --mutate=drop                   the second implementation without the [ct W] term of Phi_V: (b) must fail
        cert --mutate=sign | radius         the sign of [G P]: the start data must miss the certificate's;
                                            r0 = 0.3: no contraction
"""
import sys

import fixed_point as FP
import fixed_point_M as FPM
import fixed_point_M_b as B
from fixed_point import Maj
from flint import arb, acb, ctx


def ov(a, b):
    return a.real.overlaps(b.real) and a.imag.overlaps(b.imag)


def m1(mutate=None):
    ctx.prec = 320
    Maj.BOX = False
    o1 = FP.run(acb(FP.A_STAR), acb(FP.WF_STAR), "0.1", verbose=False)
    xi0, D1, S1, C1 = FP.start_values(o1, +1)
    ctx.prec = 512
    st = B.Setup([acb(arb(FP.A_STAR)), acb(arb(FP.WF_STAR))], 1, "0.1", "3pi/8", mutate=mutate)
    out = B.prove(st, True, "second implementation: ")
    if not out["ok"]:
        print("M = 1 PLANTED: FAIL (no contraction)")
        return False
    R = st.R
    phi = 3 * arb.pi() / 8
    xi = acb(st.r0 * phi.cos(), st.r0 * phi.sin())
    z0 = xi * xi
    ok = True
    print(f"first implementation: N = {o1['N']}, k = {o1['k']}, contraction {float(o1['lip'].upper()):.4f}")
    print("  component, overlap, |mid - first's mid|, radius, first's radius")
    for mu in st.odd:
        ep = B.eta_radius(out, ("p", mu), R)
        ev = B.eta_radius(out, ("v", mu), R)
        Dm = xi * (st.P[mu](z0) + B.disc(ep))
        # S = z Q, Q = V + (1 + s) P
        Sm = z0 * (st.Q[mu](z0) + B.disc(B.up(ev + st.s1[mu] * ep)))
        for nm, a, b in ((f"D{mu:+d}", Dm, D1[mu]), (f"S{mu:+d}", Sm, S1[mu])):
            o = ov(a, b)
            ok = ok and o
            print(f"  {nm} {o} {float(abs(a.mid() - b.mid()).upper()):.2e} {B.rad(a):.2e} {B.rad(b):.2e}")
    for nu in st.even:
        a = st.C[nu](z0) + B.disc(B.eta_radius(out, ("c", nu), R))
        o = ov(a, C1[nu])
        ok = ok and o
        print(f"  C{nu:+d} {o} {float(abs(a.mid() - C1[nu].mid()).upper()):.2e} {B.rad(a):.2e} {B.rad(C1[nu]):.2e}")
    print(f"M = 1 PLANTED (fixed_point.py's start values): {'PASS' if ok else 'FAIL'}")
    return ok


def lip(M, xs, r0, sector, mutate=None):
    ctx.prec = 320
    Maj.BOX = False
    o1 = FPM.run_M([acb(v) for v in xs], M, r0, "3pi/8", sector, verbose=False, diag=True)
    comps, wts, rho, k, N = o1["comps"], o1["wts"], o1["rho"], o1["k"], o1["N"]
    ctx.prec = 512
    st = B.Setup([acb(arb(v)) for v in xs], M, r0, "3pi/8", sector, N, k, True, "classic")
    assert st.comps == comps
    w = {cp: B.up(rho / arb(wts[i])) for i, cp in enumerate(comps)}
    if mutate == "drop":
        orig = B.mconv
        calls = {"n": 0}

        def mconv(A, Bd, left, right, target):
            calls["n"] += 1
            if calls["n"] == 2:          # the second product of lipschitz() is [ct W]
                return {key: B.MJ({}) for key in target}
            return orig(A, Bd, left, right, target)
        B.mconv = mconv
    L2 = st.lipschitz(w)
    if mutate == "drop":
        B.mconv = orig
    L1 = o1["L"]
    n = len(comps)
    worst_hi, worst_lo, bad_a, bad_b = 0.0, 1.0, [], []
    for i in range(n):
        for j in range(n):
            a, b = float(L1[i][j].upper()), float(L2[i][j])
            if b > a * (1 + 1e-9) + 1e-300:
                bad_a.append((comps[i], comps[j], a, b))
            if a > 1e-6:
                worst_hi = max(worst_hi, b / a)
                worst_lo = min(worst_lo, b / a)
                if b < 0.8 * a:
                    bad_b.append((comps[i], comps[j], a, b))
    lam1 = float(o1["lip"].upper())
    lam2 = max(sum(wts[i] * float(L2[i][j]) / wts[j] for j in range(n)) for i in range(n))
    F1 = float(o1["F"].upper())
    F2 = max(wts[i] * float(st.Fv[cp]) for i, cp in enumerate(comps))
    okc = F2 <= F1 * (1 + 1e-9) and F2 >= 0.8 * F1
    print(f"M = {M}, r0 = {r0}, sector {sector}: N = {N}, k = {k}, a = {float(o1['a']):.5f}")
    print(f"  entries of the first above 1e-6: ratio second / first between {worst_lo:.4f} and {worst_hi:.6f}")
    print(f"  (a) second <= first in every entry: {not bad_a}" + "".join(
        f"\n      {x[0]}<-{x[1]}: first {x[2]:.6g}, second {x[3]:.6g}" for x in bad_a[:8]))
    print(f"  (b) second >= 0.8 first: {not bad_b}" + "".join(
        f"\n      {x[0]}<-{x[1]}: first {x[2]:.6g}, second {x[3]:.6g}" for x in bad_b[:8]))
    print(f"  (c) forcing in the first's norm: first {F1:.6e}, second {F2:.6e}, ratio {F2 / F1:.6f}: {okc}")
    print(f"  contraction constant in the first's weights: first {lam1:.6f}, second {lam2:.6f}")
    ok = not bad_a and not bad_b and okc
    print(f"LIPSCHITZ COMPARISON: {'PASS' if ok else 'FAIL'}")
    return ok


def cert(boxfile, mutate=None, second=False, N_in=None):
    import json
    import time
    import certify_M as CM
    import certify_M2 as C2M
    import start_cover_M2 as SC
    from flint import acb_series
    with open(boxfile) as fh:
        bx = json.load(fh)
    ce = SC.cert_data(bx)
    M, r0, arg, diag, kind, c = ce["M"], ce["r0"], ce["arg"], ce["diag"], ce["kind"], ce["c"]
    m = M + 1
    p, hull = ce["p"], ce["hull"]
    names = FPM.state_order(M)
    nd = len(names)
    print(f"{boxfile}: kind {kind}, M = {M}, ray {arg}, r0 = {r0}, Cauchy radius {c}, the certificate's N = {ce['N']}, "
          f"hull half-width {ce['hw']:.3e}")
    # the first implementation, by the certificate's own calls
    t0 = time.time()
    cen = SC.centre_run(p, M, r0, arg, diag) if kind == "box2" else None
    ctx.prec = SC.PREC
    _, v1c, d1c, e1c = FPM.start_data_M([arb(v) for v in p], M, r0, arg, +1, c, centre=cen, diag=diag)
    _, v1b, d1b, e1b = FPM.start_data_M([arb(v) for v in hull], M, r0, arg, +1, c, centre=cen, diag=diag)
    print(f"first implementation (start_data_M at p^ and over the hull): sup|eta| {float(e1c[0].upper()):.3e} and "
          f"{float(e1b[0].upper()):.3e}, Cauchy parts {float(e1c[1].upper()):.3e} and {float(e1b[1].upper()):.3e}  "
          f"[{time.time() - t0:.0f} s]", flush=True)
    # the second implementation
    t0 = time.time()
    okc, _, v2c, d2c, ic = B.start_data(p, M, r0, arg, c, N=N_in, mutate=mutate, label="p^, ")
    Np, kp = ic["N"], ic["k"]
    okb, _, v2b, d2b, ib = B.start_data(hull, M, r0, arg, c, N=Np, k=kp, mutate=mutate, label="hull, ")
    print(f"second implementation: N = {Np} (the certificate's {ce['N']}), k = {kp}; four contractions hold: "
          f"{okc and okb}  [{time.time() - t0:.0f} s]", flush=True)
    if not (okc and okb):
        print("CERTIFICATE START DATA: FAIL (no contraction)")
        return False
    print(f"  sup|eta| in (p, v, c): {ic['eta']:.3e} at p^, {ib['eta']:.3e} over the hull; on the Cauchy boxes "
          f"{ic['eta_cauchy']:.3e} and {ib['eta_cauchy']:.3e}")
    ok = True

    def block(title, A1, A2):
        """A1, A2: lists of acb (first, second); overlap per entry"""
        nonlocal ok
        miss = [i for i in range(len(A1)) if not ov(A2[i], A1[i])]
        inside = sum(bool(A1[i].contains(A2[i])) for i in range(len(A1)))
        dist = max(float(abs(A2[i].mid() - A1[i].mid()).upper()) for i in range(len(A1)))
        r1, r2 = max(B.rad(z) for z in A1), max(B.rad(z) for z in A2)
        print(f"  {title}: {len(A1) - len(miss)} of {len(A1)} overlap ({inside} of the second's balls inside the "
              f"first's); largest midpoint distance {dist:.2e}; largest radius first {r1:.2e}, second {r2:.2e}"
              + (f"; MISSES at entries {miss[:8]}" if miss else ""))
        ok = ok and not miss
    block("values at p^", v1c, v2c)
    block("derivatives at p^", [z for col in d1c for z in col], [z for col in d2c for z in col])
    block("derivatives over the hull", [z for col in d1b for z in col], [z for col in d2b for z in col])
    block("values over the hull (not used by the certificate)", v1b, v2b)
    if second and kind == "box2":
        # second-order jets of Y_N over the hull along the chart's directions and their pair sums (start_data2's jd, js)
        import certify_M2_lohner as C2
        t0 = time.time()
        ctx.prec = CM.KPREC
        Cm = [[arb(x) for x in r] for r in bx["C"]]
        G = [[arb(x) for x in r] for r in bx["Gamma"]]
        ws = list(bx["w"])
        widths = [arb(bx["h"]), arb(bx["h"])] + [arb(ws[(i // 2) % len(ws)]) for i in range(M - 1)]
        ctx.prec = SC.PREC
        T = CM.T_matrix(Cm, M)
        X = [arb(0, wd) for wd in widths]
        dirs = C2M.dp_dirs(T, G, X, M)
        dirs = dirs + [[dirs[j][i] + dirs[k][i] for i in range(m)] for (j, k) in C2.pair_list(m) if j != k]
        path = CM.path_for(M)
        j1 = C2M.series2(hull, dirs, M, r0, path[1], ce["N"])
        t1 = time.time() - t0
        ctx.prec = 512
        ctx.cap = 3
        odd, even = B.harmonics(M)
        xi0 = acb(arb(r0)) * (B.I * B.angle(arg)).exp()
        z0 = xi0 * xi0
        zero, one = acb_series([0], prec=3), acb_series([1], prec=3)
        a1, a2 = [], []
        for di, dv in enumerate(dirs):
            pj = [acb_series([acb(hull[i]), acb(dv[i])], prec=3) for i in range(m)]
            F, w, sl = B.profile(pj, M)
            d, q, g = B.formal_series(F, sl, M, ce["N"], zero, one, True, st_gp(mutate))
            st2 = [B.horner([d[j][mu] for j in range(ce["N"])], z0, zero) * xi0 for mu in odd]
            st2 += [B.horner([q[j][mu] for j in range(ce["N"])], z0, zero) * z0 for mu in odd]
            st2 += [B.horner([g[j][nu] for j in range(ce["N"])], z0, zero) for nu in even]
            for qi in range(nd):
                cs = st2[qi].coeffs()
                for o in range(3):
                    a1.append(j1[di][qi][o])
                    a2.append(cs[o] if len(cs) > o else acb(0))
        print(f"  second-order jets: {len(dirs)} directions, first {t1:.0f} s, second {time.time() - t0 - t1:.0f} s")
        block("jets of Y_N over the hull (orders 0, 1, 2)", a1, a2)
    print(f"CERTIFICATE START DATA ({boxfile}): {'PASS' if ok else 'FAIL'}")
    return ok


def assemble_b(M, pstr, Cm, G, widths, msys=None, drop_d2=False, verbose=True):
    """the C^2 start data of a second-order box run from the second implementation, with certify_M2.start_data2's
    arguments and return values (vals, cols, wcol, W0cols, info). The chart geometry (p^, T, Gamma, the directions,
    the pair list, the hull of p(X)) is built by the certificate's helpers as start_data2 builds it. Everything that
    start_data2 takes from fixed_point_M.py comes from fixed_point_M_b.py here: the values and first derivatives at
    p^, the first derivatives over the hull, D^2 Y_N over the hull (the recursion on series of length 3, by
    polarization) and the bound e_q of eta on the Cauchy box at p^ in the Cauchy estimate e_q (2 nu / (c - hw))^2.
    msys: a mutation of the second implementation's system ("sign"); drop_d2: D^2 Y_N set to zero."""
    import hashlib
    import time
    import certify_M as CM
    import certify_M2 as C2M
    import certify_M2_lohner as C2
    import certify_M_sys as CS
    from flint import acb_series
    t0 = time.time()
    m = M + 1
    path = CM.path_for(M)
    r0, arg, c = path[0], CM.ARG[M], C2M.CAUCHY
    ctx.prec = 320
    p = CM.exact_p(pstr)
    ctx.prec = 320
    T = CM.T_matrix(Cm, M)
    X = [arb(0, wd) for wd in widths]
    hull = C2M.p_of_x(p, T, G, X, M)
    hw = max(float(abs(hull[i] - p[i]).upper()) for i in range(m))
    assert hw < 0.5 * float(arb(c).mid()), "the box must lie well inside the Cauchy box"
    okc, xi0, v2, d2c, ic = B.start_data(p, M, r0, arg, c, mutate=msys, verbose=False)
    okb, _, _, d2b, ib = B.start_data(hull, M, r0, arg, c, N=ic["N"], k=ic["k"], mutate=msys, verbose=False)
    lams = [float(o["lam"]) for o in (ic["real"], ic["cbox"], ib["real"], ib["cbox"]) if "lam" in o]
    if verbose:
        print(f"second implementation: N = {ic['N']}, k = {ic['k']}; four contractions hold: {okc and okb} (constants "
              + ", ".join(f"{v:.4f}" for v in lams) + f")  [{time.time() - t0:.0f} s]", flush=True)
    if not (okc and okb):
        raise RuntimeError("second implementation: fixed point not verified on the certificate's four sets")
    nd = len(v2)
    ctx.prec = 320
    dirs = C2M.dp_dirs(T, G, X, M)
    pairs = C2.pair_list(m)
    sums = [[dirs[j][i] + dirs[k][i] for i in range(m)] for (j, k) in pairs if j != k]
    nu = max(float(abs(dirs[k][i]).upper()) for k in range(m) for i in range(m))
    wcol = [CS.W0() * T[1][k] for k in range(m)]
    ctx.prec = 512
    cc = arb(c)
    fac = (2 * arb(nu) / (cc - arb(hw))) ** 2
    cols = [[sum((acb(T[i][k]) * d2c[i][q] for i in range(m)), acb(0)) for q in range(nd)] for k in range(m)]
    ctx.cap = 3
    odd, even = B.harmonics(M)
    z0 = xi0 * xi0
    zero, one = acb_series([0], prec=3), acb_series([1], prec=3)
    N = ic["N"]

    def jets(dv):
        pj = [acb_series([acb(hull[i]), acb(dv[i])], prec=3) for i in range(m)]
        F, _, sl = B.profile(pj, M)
        d, q, g = B.formal_series(F, sl, M, N, zero, one, True, st_gp(msys))
        st = [B.horner([d[j][mu] for j in range(N)], z0, zero) * xi0 for mu in odd]
        st += [B.horner([q[j][mu] for j in range(N)], z0, zero) * z0 for mu in odd]
        st += [B.horner([g[j][nu_] for j in range(N)], z0, zero) for nu_ in even]
        out = []
        for v in st:
            cs = v.coeffs()
            out.append(cs[2] if len(cs) > 2 else acb(0))
        return out
    jd = [jets(dv) for dv in dirs]
    js = iter([jets(dv) for dv in sums])
    e_q = ic["e_cauchy"]
    W = []
    for (j, k) in pairs:
        if j == k:
            dd = [2 * jd[j][q] for q in range(nd)]
        else:
            jsk = next(js)
            dd = [jsk[q] - jd[j][q] - jd[k][q] for q in range(nd)]
        if drop_d2:
            dd = [acb(0) for _ in range(nd)]
        col = []
        for q in range(nd):
            v = dd[q] + B.disc(B.up(e_q[q] * fac))
            if j < 2 and k < 2:
                for i in range(2, m):
                    v = v + d2b[i][q] * C2M.gam(G, i - 2, j, k)
            col.append(v)
        W.append(col)
    sha = hashlib.sha256(open(B.__file__, "rb").read()).hexdigest()[:16]
    info = {"start": f"fixed_point_M_b.start_data (the second implementation; sha256 {sha}), arg {arg}, diag "
                     f"splitting, N = {N}, k = {ic['k']}; second derivatives: its series jets + Cauchy"
                     + (f"; MUTATION {msys}" if msys else "") + ("; MUTATION w0" if drop_d2 else ""),
            "rigorous": not (msys or drop_d2), "N": N, "k": ic["k"], "eta": ic["eta"], "eta_cauchy": ic["eta_cauchy"],
            "eta_d2_bound": max(float((e * fac).upper()) for e in e_q), "hull_halfwidth": hw, "nu": nu,
            "contraction": lams, "sec": time.time() - t0, "mutate": [x for x in (msys, "w0" if drop_d2 else None) if x]}
    if verbose:
        print(f"  second implementation's assembly: hull half-width {hw:.3e}, nu = {nu:.3f}, Cauchy factor "
              f"{float(fac.mid()):.4e}, its largest bound of D^2 eta {info['eta_d2_bound']:.3e}  "
              f"[{time.time() - t0:.0f} s]", flush=True)
    return v2, cols, wcol, W, info


def chart_of(bx):
    """(Cm, G, widths) of a stored box2 run, as certify_M2.load_chart and box2_run build them"""
    import certify_M as CM
    ctx.prec = CM.KPREC
    Cm = [[arb(x) for x in r] for r in bx["C"]]
    G = [[arb(x) for x in r] for r in bx["Gamma"]]
    ws = list(bx["w"])
    widths = [arb(bx["h"]), arb(bx["h"])] + [arb(ws[(i // 2) % len(ws)]) for i in range(bx["M"] - 1)]
    return Cm, G, widths


def cert2(boxfile, mutate=None):
    import json
    import time
    import certify_M2 as C2M
    import certify_M2_lohner as C2
    with open(boxfile) as fh:
        bx = json.load(fh)
    assert bx["kind"] == "box2", "cert2 is for the second-order (box2) runs"
    M = bx["M"]
    m = M + 1
    nd = len(FPM.state_order(M))
    Cm, G, widths = chart_of(bx)
    # the first implementation: the certificate's own call
    t0 = time.time()
    v1, cols1, _, W1, info1 = C2M.start_data2(M, bx["p"], Cm, G, widths, verbose=False)
    pairs = C2.pair_list(m)
    print(f"{boxfile}: M = {M}, h = {bx['h']}, w = {list(bx['w'])}; start_data2: N = {info1['N']}, "
          f"second-derivative bound of eta {info1['eta_d2_bound']:.3e}, {len(pairs)} pairs  [{time.time() - t0:.0f} s]",
          flush=True)
    try:
        v2, cols2, _, W2, _ = assemble_b(M, bx["p"], Cm, G, widths, mutate if mutate == "sign" else None,
                                         mutate == "w0")
    except RuntimeError as e:
        print(f"C2 START DATA: FAIL ({e})")
        return False
    ok = True

    def block(title, A1, A2):
        nonlocal ok
        miss = [i for i in range(len(A1)) if not ov(A2[i], A1[i])]
        inside = sum(bool(A1[i].contains(A2[i])) for i in range(len(A1)))
        dist = max(float(abs(A2[i].mid() - A1[i].mid()).upper()) for i in range(len(A1)))
        r1, r2 = max(B.rad(z) for z in A1), max(B.rad(z) for z in A2)
        print(f"  {title}: {len(A1) - len(miss)} of {len(A1)} overlap ({inside} of the second's balls inside the "
              f"first's); largest midpoint distance {dist:.2e}; largest radius first {r1:.2e}, second {r2:.2e}")
        ok = ok and not miss
        return miss
    block("values at p^", v1, v2)
    block("first-derivative columns V0", [z for col in cols1 for z in col], [z for col in cols2 for z in col])
    miss = block("second-derivative columns W0", [z for col in W1 for z in col], [z for col in W2 for z in col])
    bad = sorted({pairs[i // nd] for i in miss})
    print(f"  W0 pair by pair: {len(pairs) - len(bad)} of {len(pairs)} pairs overlap in all {nd} components"
          + (f"; pairs with a miss: {bad[:10]}" if bad else ""))
    for pi, (j, k) in enumerate(pairs):
        r1, r2 = max(B.rad(z) for z in W1[pi]), max(B.rad(z) for z in W2[pi])
        mx = max(float(abs(z).upper()) for z in W1[pi])
        dist = max(float(abs(W2[pi][q].mid() - W1[pi][q].mid()).upper()) for q in range(nd))
        print(f"    pair ({j}, {k}): max |W0| {mx:.3e}, midpoints within {dist:.2e}, radius first {r1:.2e}, second "
              f"{r2:.2e}")
    print(f"C2 START DATA ({boxfile}): {'PASS' if ok else 'FAIL'}")
    return ok


def box2(boxfile, out, mutate=None, pilot=None, log=None):
    """the second-order box run of a stored certificate, from the second implementation's start data (#880):
    certify_M2.box2_run with the stored run's chart, h and w, and with certify_M2.start_data2 replaced, for this call,
    by assemble_b. pilot: stop after that many steps (certify_M2's own pilot)."""
    import json
    import certify_M2 as C2M
    with open(boxfile) as fh:
        bx = json.load(fh)
    orig = C2M.start_data2

    def start_b(M, pstr, Cm, G, widths, verbose=True):
        return assemble_b(M, pstr, Cm, G, widths, mutate if mutate == "sign" else None, mutate == "w0")
    C2M.start_data2 = start_b
    try:
        res = C2M.box2_run(bx["M"], bx["chart"], bx["h"], list(bx["w"]), out, log=log, max_steps=pilot, verbose=True)
    finally:
        C2M.start_data2 = orig
    return res.get("kind") in ("box2", "pilot2")


def kraw(newfile, storedfile, sub=None):
    """the Krawczyk test of a box2 run made from the second implementation's start data, beside the stored run's"""
    import json
    import certify_M2 as C2M
    import start_cover_M2 as SC
    with open(newfile) as fh:
        bn = json.load(fh)
    with open(storedfile) as fh:
        bo = json.load(fh)
    assert bn["p"] == bo["p"] and bn["h"] == bo["h"] and list(bn["w"]) == list(bo["w"]) and bn["C"] == bo["C"], \
        "the two runs have another centre, chart or box"
    names = SC.pnames(bn["M"])
    print(f"new run: {newfile}: {bn['steps']} steps, {bn['sec']:.0f} s; start: {bn['start']['start']}")
    print(f"stored run: {storedfile}: {bo['steps']} steps; start: {bo['start']['start'][:60]}...")
    ok = True
    for sb in [None] + ([sub] if sub else []):
        rn = C2M.krawczyk2_test(bn, sub=sb, verbose=False)
        ro = C2M.krawczyk2_test(bo, sub=sb, verbose=False)
        ctx.prec = 320
        over = all(bool(rn["pstar"][i].overlaps(ro["pstar"][i])) for i in range(len(names)))
        where = "the run's whole box" if sb is None else "the sub-box +-(" + ", ".join(sb) + ")"
        print(f"Krawczyk on {where}: K(X) in int X: new {rn['ok']}, stored {ro['ok']}")
        print("  relative margins, new:    " + ", ".join(f"{r:+.3f}" for r in rn["rel_margins"]))
        print("  relative margins, stored: " + ", ".join(f"{r:+.3f}" for r in ro["rel_margins"]))
        print("  zero's enclosure, radius new / stored: " + ", ".join(
            f"{names[i]} {float(rn['pstar'][i].rad()):.2e} / {float(ro['pstar'][i].rad()):.2e}" for i in range(len(names)))
            + f"; the two enclosures overlap in every parameter: {over}")
        ok = ok and rn["ok"] and over
    print(f"KRAWCZYK ON THE SECOND IMPLEMENTATION'S START DATA: {'PASS' if ok else 'FAIL'}")
    return ok


def cli(mutate=None):
    import json
    import os
    import re
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "data", "A_M2_box_M5_1e-10.json")) as fh:
        X = json.load(fh)["p"]
    cmd = [sys.executable, os.path.join(here, "fixed_point_M_b.py"), "--M=5", "--x=" + ",".join(X)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    m1_ = re.search(r"N = (\d+), k = (\d+)", out.stdout)
    m2_ = re.search(r"contraction constant max_i \(L w\)_i / w_i <= ([0-9.]+)", out.stdout)
    if not (m1_ and m2_):
        print("the command line printed no N, k and constant:\n" + out.stdout[-600:] + out.stderr[-600:])
        return False
    got = (int(m1_.group(1)), int(m1_.group(2)), m2_.group(1))
    ctx.prec = 53 if mutate == "read53" else 512
    x = [acb(arb(v)) for v in X]
    r0 = float(x[0].real.rad())
    ctx.prec = 512
    st = B.Setup(x, 5, "0.07", "3pi/8")
    o = B.prove(st, False)
    want = (st.N, st.k, f"{float(o['lam']):.4f}")
    ok = got == want and out.returncode == 0 and o["ok"]
    print(f"command line (exit {out.returncode}): N = {got[0]}, k = {got[1]}, constant {got[2]}")
    print(f"Setup and prove on the point read at {53 if mutate == 'read53' else 512} bits (radius of A {r0:.1e}): "
          f"N = {want[0]}, k = {want[1]}, constant {want[2]}")
    print(f"COMMAND LINE PRECISION: {'PASS' if ok else 'FAIL'}")
    return ok


def st_gp(mutate):
    return {"sign": acb(-1), "scale": 1 + acb("1e-9")}.get(mutate)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
    cmd = args[0] if args else "m1"
    if cmd == "m1":
        ok = m1(opts.get("mutate"))
    elif cmd == "box2":
        ok = box2(opts["box"], opts["out"], opts.get("mutate"), int(opts["pilot"]) if "pilot" in opts else None,
                  opts.get("log"))
    elif cmd == "kraw":
        ok = kraw(opts["box"], opts["stored"], opts["sub"].split(",") if "sub" in opts else None)
    elif cmd == "cli":
        ok = cli(opts.get("mutate"))
    elif cmd == "cert2":
        ok = cert2(opts["box"], opts.get("mutate"))
    elif cmd == "cert":
        ok = cert(opts["box"], opts.get("mutate"), "second" in opts, int(opts["N"]) if "N" in opts else None)
    else:
        M = int(opts.get("M", "5"))
        xs = opts["x"].split(",") if "x" in opts else FPM.ZEROS[M]
        ok = lip(M, xs, opts.get("r0", "0.07" if M == 5 else "0.1"), opts.get("sector"), opts.get("mutate"))
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()

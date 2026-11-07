#!/usr/bin/env python3
"""Independent check of the M = 5 C^2 certificate (proof/A_M2_review.md): Y D^2 gt(0) and Y Dgt(0) from point
integrations, compared with the certificate's enclosures Y l W(X) and Y l V0 at xi1.

Written for the review, without the code under review. What it computes:
  * the M = 5 system of certify_M_sys.py's docstring (xi^2 D' + i mu w xi D = [C S], xi^2 S' + i mu w xi S =
    [C xi D'] + [(C E) D] - [C D], xi C' = [C E], E = [D D]/2), coded anew here;
  * a non-rigorous Taylor integrator (Picard iteration on power series in the leg parameter, order NT), on the path
    of proof/A_M.md section 1 (ray at 5 pi/16 from 0.0622 to 0.2, arc to the real axis, real axis to 0.3), with one
    step sequence chosen at the centre and reused at every parameter point, so that the numerical flow is an analytic
    function of the parameters;
  * start values: the truncated series Y_N of the start-value lemma (fixed_point_M.series_jet_M, values only), at
    complex parameters;
  * derivatives without differences of first derivatives: the chart p(x) = p^ + T x + (0, Gamma[x_o, x_o]) / 2 and
    gt(x) = g(p(x)) extend holomorphically to complex x, with J_nu(z) = D_nu(p(z)) - conj(D_-nu(p(conj z))). Along a
    real direction u, the Taylor coefficients of z -> J(z u) come from four points on a circle (Cauchy's formula,
    trapezoid rule): c1 = mean J(z_k) / z_k, c2 = mean J(z_k) / z_k^2, z_k = r i^k, exact up to the aliased terms
    c5 r^4 and c6 r^4. Then Dgt[u] = c1 and D^2 gt[u, u] = 2 c2 (Re and Im parts give the real rows).
  * the certificate's enclosures from the saved box file (the Lohner sets at xi1), assembled anew here in arb at 768
    bits: Y = inverse of the midpoint of l V0 (rounded to an exact matrix), Y l (cW + QW qW), Y l (cV + QV qV).

Shares with the certificate: the definition of the system, path, jump and chart (and their data: p^, C, Gamma), the
lemma's series code for Y_N (fixed_point_M.series_jet_M, an assumed fact of the claim, not the series code that
certify_M2.series2 uses), python-flint, and the saved Lohner sets it compares with. Not shared: certify_M_sys.SysM,
certify_M_lohner, validated_int, certify_M2*, certify_M2_fd.py (no variational equations, no finite differences).

    python3 review_M2_indep.py centre --out=f.json            the step sequence and g(p^) at the centre
    python3 review_M2_indep.py dir --steps=f.json --u=j[,k,lam] --r=R --out=g.json   the 4 contour runs along u
    python3 review_M2_indep.py compare --box=data/A_M2_box_M5_1e-10.json g1.json g2.json ...
"""
import json
import os
import sys
import time

from flint import acb, arb, acb_series, arb_mat, ctx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

M = 5
ODD = [m for m in range(-M, M + 1) if m % 2]
EVEN = [v for v in range(-(M + 1), M + 2) if v % 2 == 0]
PREC = 320
NT = 30                  # Taylor order of the integrator
NSER = 132               # the lemma's truncation at M = 5 (the box file's start info)
R0, RHO, XI1 = "0.0622", "0.2", "0.3"
CHART = os.path.join(HERE, "data", "A_M2_chart_M5.json")
Ii = acb(0, 1)


def phi0():
    return 5 * arb.pi() / 16


def winf():
    return 2 * arb.pi() / arb("1.64389375")


# ---------------------------------------------------------------- the system, coded from its equations

def conv(X, Y, out):
    """{k: sum_{a + b = k} X_a Y_b} for k in out"""
    r = {}
    for k in out:
        acc = None
        for a, xa in X.items():
            yb = Y.get(k - a)
            if yb is not None:
                acc = xa * yb if acc is None else acc + xa * yb
        r[k] = acc
    return r


def F(D, S, C, w, xs, ixs):
    """dD/dxi, dS/dxi, dC/dxi from xi^2 D' + i mu w xi D = [C S]; xi^2 S' + i mu w xi S = [C xi D'] + [(C E) D]
    - [C D]; xi C' = [C E]; E = [D D] / 2"""
    ixs2 = ixs * ixs
    cs = conv(C, S, ODD)
    dD = {m: (cs[m] - Ii * m * w * xs * D[m]) * ixs2 for m in ODD}
    dd = conv(D, D, EVEN)
    E = {v: dd[v] / 2 for v in EVEN}
    cE = conv(C, E, EVEN)
    dC = {v: cE[v] * ixs for v in EVEN}
    xdD = {m: xs * dD[m] for m in ODD}
    t1, t2, t3 = conv(C, xdD, ODD), conv(cE, D, ODD), conv(C, D, ODD)
    dS = {m: (t1[m] + t2[m] - t3[m] - Ii * m * w * xs * S[m]) * ixs2 for m in ODD}
    return dD, dS, dC


# ---------------------------------------------------------------- the path

def leg_xi(kind, t):
    """xi(t) and dxi/dt for t an acb (point) or acb_series"""
    p0 = phi0()
    if kind == "ray":
        u = (Ii * p0).exp()
        return (acb(arb(R0)) + acb(arb(RHO) - arb(R0)) * t) * u, acb(arb(RHO) - arb(R0)) * u
    if kind == "arc":
        x = acb(arb(RHO)) * (Ii * p0 * (1 - t)).exp()
        return x, -Ii * p0 * x
    return acb(arb(RHO)) + acb(arb(XI1) - arb(RHO)) * t, acb(arb(XI1) - arb(RHO))


def leg_series(kind, t0, L):
    """xi(t0 + tau) and dxi/dt as series in tau of length L"""
    x0, v = leg_xi(kind, acb(t0))
    if kind in ("ray", "real"):
        return acb_series([x0, v], prec=L), acb_series([v], prec=L)
    e = acb_series([0, -Ii * phi0()], prec=L).exp()
    xs = x0 * e
    return xs, -Ii * phi0() * xs


# ---------------------------------------------------------------- the Taylor integrator (non-rigorous)

def taylor(kind, t0, y, w, N):
    """Taylor coefficients (order N) in tau of the solution through y at t0: Picard iteration on series"""
    L = N + 1
    ctx.cap = L
    xs, dxs = leg_series(kind, t0, L)
    ixs = 1 / xs
    D0, S0, C0 = y
    D = {m: acb_series([D0[m]], prec=L) for m in ODD}
    S = {m: acb_series([S0[m]], prec=L) for m in ODD}
    C = {v: acb_series([C0[v]], prec=L) for v in EVEN}
    for it in range(N + 1):
        Lk = min(it + 2, L)
        ctx.cap = Lk

        def tr(z):
            return acb_series(z.coeffs()[:Lk], prec=Lk)
        Dk, Sk, Ck = ({k: tr(v) for k, v in X.items()} for X in (D, S, C))
        xk, ixk, dxk = tr(xs), tr(ixs), tr(dxs)
        dD, dS, dC = F(Dk, Sk, Ck, w, xk, ixk)
        D = {m: acb_series([D0[m]], prec=Lk) + (dD[m] * dxk).integral() for m in ODD}
        S = {m: acb_series([S0[m]], prec=Lk) + (dS[m] * dxk).integral() for m in ODD}
        C = {v: acb_series([C0[v]], prec=Lk) + (dC[v] * dxk).integral() for v in EVEN}
    ctx.cap = L
    return D, S, C


def evaluate(s, h, N):
    cs = s.coeffs()
    acc = acb(0)
    for k in range(min(N, len(cs) - 1), -1, -1):
        acc = acc * h + cs[k]
    return acc


def coefmax(Ser, k):
    m = 0.0
    for X in Ser:
        for v in X.values():
            cs = v.coeffs()
            if k < len(cs):
                m = max(m, float(abs(cs[k]).mid()))
    return m


def integrate(y, w, steps=None, tol=1e-45, rate_fac=0.5, N=NT, log=None):
    """along the three legs; steps None: choose them (from the last Taylor coefficients and the fast-mode rate
    |xi'| / |xi|^3) and return them; else use the given list [(kind, h), ...] exactly"""
    chosen = []
    it = iter(steps) if steps is not None else None
    for kind in ("ray", "arc", "real"):
        t = arb(0)
        while t < 1:
            Dser, Sser, Cser = taylor(kind, t, y, w, N)
            if it is not None:
                k2, h = next(it)
                assert k2 == kind
            else:
                x0, v = leg_xi(kind, acb(t))
                rate = float(abs(v).mid()) * (1 / float(abs(x0).mid()) ** 3 + 2 / float(abs(x0).mid()) ** 2
                                              + M * float(abs(w).mid()) / float(abs(x0).mid()))
                h = rate_fac / rate
                for kk in (N - 1, N):
                    cm = coefmax((Dser, Sser, Cser), kk) + 1e-300
                    h = min(h, (tol / cm) ** (1.0 / kk))
                rem = float((1 - t).mid())
                if h >= rem * 0.999:
                    h = None                       # last step: exactly to t = 1
                chosen.append((kind, h))
            if h is None:
                hh = 1 - t
            else:
                hh = arb(h)
            ha = acb(hh)
            y = ({m: evaluate(Dser[m], ha, N) for m in ODD}, {m: evaluate(Sser[m], ha, N) for m in ODD},
                 {v: evaluate(Cser[v], ha, N) for v in EVEN})
            t = t + hh if h is not None else arb(1)
        if log:
            log(f"   leg {kind} done")
    return y, chosen


# ---------------------------------------------------------------- start values, chart, jump

def load_chart():
    d = json.load(open(CHART))
    ctx.prec = PREC
    p = [arb(arb(x).mid()) for x in d["p"]]
    Cm = [[arb(arb(x).mid()) for x in r] for r in d["C"]]
    G = [[arb(x) for x in r] for r in d["Gamma"]]
    return p, Cm, G


def p_of(z, chart):
    """p(z) = p^ + T z + (0, Gamma[z_o, z_o]) / 2 for a complex vector z"""
    p, Cm, G = chart
    out = [acb(p[0]) + z[0], acb(p[1]) + z[1]]
    for i in range(M - 1):
        q = (acb(G[i][0]) * z[0] * z[0] + 2 * acb(G[i][1]) * z[0] * z[1] + acb(G[i][2]) * z[1] * z[1]) / 2
        out.append(acb(p[2 + i]) + acb(Cm[i][0]) * z[0] + acb(Cm[i][1]) * z[1] + z[2 + i] + q)
    return out


def start(pz):
    """Y_N at xi0 = r0 e^{i phi0} for parameters pz (acb), values only"""
    import fixed_point_M as FPM
    odd, even, d, s, g = FPM.series_jet_M(pz, M, NSER)
    assert odd == ODD and even == EVEN
    xi0 = acb(arb(R0)) * (Ii * phi0()).exp()
    z0 = xi0 * xi0
    D, S, C = {}, {}, {}
    for m in ODD:
        a, b, pw = acb(0), acb(0), acb(1)
        for j in range(NSER):
            a += d[j][m].v * pw
            b += s[j][m].v * pw
            pw = pw * z0
        D[m], S[m] = a * xi0, b * z0
    for v in EVEN:
        a, pw = acb(0), acb(1)
        for j in range(NSER):
            a += g[j][v].v * pw
            pw = pw * z0
        C[v] = a
    return (D, S, C), winf() * pz[1]


def run_point(z, chart, steps=None, log=None):
    pz = p_of(z, chart)
    y0, w = start(pz)
    y1, chosen = integrate(y0, w, steps=steps, log=log)
    return y1[0], chosen


def s_c(z):
    return [z.real.mid().str(90, radius=False), z.imag.mid().str(90, radius=False)]


def p_c(v):
    return acb(arb(v[0]), arb(v[1]))


# ---------------------------------------------------------------- comparison with the certificate's sets

def jump_rows(n):
    """l (6 x 2n) on z = (Re y, Im y): Re J = Re D_nu - Re D_-nu, Im J = Im D_nu + Im D_-nu (state order D, S, C, w)"""
    rows = []
    for nu in (1, 3, 5):
        ip, im = ODD.index(nu), ODD.index(-nu)
        re, imr = [0] * (2 * n), [0] * (2 * n)
        re[ip], re[im] = 1, -1
        imr[n + ip], imr[n + im] = 1, 1
        rows += [re, imr]
    return arb_mat([[arb(x) for x in r] for r in rows])


def lohner_apply(R, sets, ck, Qk, qk):
    cc = sets[ck]
    n, m = len(cc), len(cc[0])
    Z = arb_mat(2 * n, m)
    for i in range(n):
        for j in range(m):
            Z[i, j], Z[n + i, j] = arb(cc[i][j][0]), arb(cc[i][j][1])
    Q = arb_mat([[arb(float.fromhex(x)) for x in row] for row in sets[Qk]])
    q = arb_mat([[arb(x) for x in row] for row in sets[qk]])
    V = R * Z + (R * Q) * q
    return [[V[i, j] for j in range(m)] for i in range(V.nrows())]


def assemble(bx):
    ctx.prec = 768
    S = bx["sets"]
    n = len(S["c"])
    ell = jump_rows(n)
    LV0 = lohner_apply(ell, S, "cV", "QV", "qV")
    m = 6
    Dm = arb_mat([[arb(LV0[i][j].mid()) for j in range(m)] for i in range(m)])
    Yb = Dm.inv()
    Y = arb_mat([[arb(Yb[i, j].mid()) for j in range(m)] for i in range(m)])
    Yl = Y * ell
    G1 = lohner_apply(Yl, S, "cV", "QV", "qV")
    G2 = lohner_apply(Yl, S, "cW", "QW", "qW")
    c = S["c"]
    z = arb_mat([[arb(v[0])] for v in c] + [[arb(v[1])] for v in c])
    Q = arb_mat([[arb(float.fromhex(x)) for x in row] for row in S["Q"]])
    q = arb_mat([[arb(x)] for x in S["q"]])
    gv = ell * z + (ell * Q) * q
    return Y, G1, G2, [gv[i, 0] for i in range(m)]


def pairs():
    return [(j, k) for j in range(6) for k in range(j, 6)]


def compare(bxfile, files):
    bx = json.load(open(bxfile))
    Y, G1, G2, g0 = assemble(bx)
    pidx = {pk: i for i, pk in enumerate(pairs())}
    Yn = [[Y[i, j] for j in range(6)] for i in range(6)]
    allok = True
    diag = {}
    print(f"comparison with {bxfile} (Y = mid of (mid l V0)^-1, assembled here at 768 bits)")
    for f in files:
        d = json.load(open(f))
        u = [arb(x) for x in d["u"]]
        r = arb(d["r"])
        Dk = [[p_c(v) for v in row] for row in d["D"]]       # Dk[k][mu index] at z_k = r i^k
        zk = [acb(r) * Ii ** k for k in range(4)]
        J = []
        for k in range(4):
            kb = (4 - k) % 4                                  # conj(z_k) = z_kb
            J.append([Dk[k][ODD.index(nu)] - Dk[kb][ODD.index(-nu)].conjugate() for nu in (1, 3, 5)])
        c1 = [sum((J[k][a] / zk[k] for k in range(4)), acb(0)) / 4 for a in range(3)]
        c2 = [sum((J[k][a] / (zk[k] * zk[k]) for k in range(4)), acb(0)) / 4 for a in range(3)]
        d1 = [x for a in range(3) for x in (c1[a].real, c1[a].imag)]
        d2 = [x for a in range(3) for x in (2 * c2[a].real, 2 * c2[a].imag)]
        Yd1 = [sum((Yn[i][j] * d1[j] for j in range(6)), arb(0)) for i in range(6)]
        Yd2 = [sum((Yn[i][j] * d2[j] for j in range(6)), arb(0)) for i in range(6)]
        # the certificate's enclosures along u: Y l V0 u and sum_{j<=k} mult (Y l W)_{jk} u_j u_k
        e1 = [sum((G1[i][j] * u[j] for j in range(6)), arb(0)) for i in range(6)]
        e2 = []
        for i in range(6):
            acc = arb(0)
            for (j, k) in pairs():
                if u[j] == 0 or u[k] == 0:
                    continue
                acc += (1 if j == k else 2) * G2[i][pidx[(j, k)]] * u[j] * u[k]
            e2.append(acc)
        tag = d["tag"]
        print(f"direction {tag} (u = {[float(x.mid()) for x in u]}, r = {float(r.mid()):.1e}):")
        ok = True
        for i in range(6):
            in1 = e1[i].contains(arb(Yd1[i].mid()))
            in2 = e2[i].contains(arb(Yd2[i].mid()))
            dev2 = float(abs(arb(Yd2[i].mid()) - arb(e2[i].mid())).mid())
            rad2 = float(e2[i].rad())
            ok &= in1 and in2
            print(f"   row {i}: Y D^2 gt[u,u] here {float(Yd2[i].mid()):+.9e}  enclosure {float(e2[i].mid()):+.9e} "
                  f"+- {rad2:.2e}  |dev| {dev2:.2e} (= {dev2 / (rad2 + 1e-300):.2e} radius, "
                  f"{dev2 / (abs(float(Yd2[i].mid())) + 1e-300):.2e} relative)  contained {in2};  "
                  f"Y Dgt[u] {float(Yd1[i].mid()):+.6e} in {float(e1[i].mid()):+.6e} +- {float(e1[i].rad()):.1e}: {in1}")
        allok &= ok
        diag[tag] = (u, Yd2)
    # polarization: entries (j, k) from the directions e_j, e_k, e_j + lam e_k (scaled)
    print("entries by polarization, D_jk = (D[u,u] - D[a,a] lam_a^2 - D[b,b] lam_b^2) / (2 lam_a lam_b):")
    for tag, (u, Yd2) in diag.items():
        nz = [j for j in range(6) if u[j] != 0]
        if len(nz) != 2:
            continue
        j, k = nz
        ta, tb = f"e{j}", f"e{k}"
        if ta not in diag or tb not in diag:
            continue
        ua, Ya = diag[ta]
        ub, Yb_ = diag[tb]
        la, lb = u[j] / ua[j], u[k] / ub[k]
        for i in range(6):
            est = (Yd2[i] - la * la * Ya[i] - lb * lb * Yb_[i]) / (2 * u[j] * u[k])
            z = G2[i][pidx[(j, k)]]
            dev = float(abs(arb(est.mid()) - arb(z.mid())).mid())
            print(f"   (Y D^2 gt)_{i},{j}{k}: here {float(est.mid()):+.9e}  enclosure {float(z.mid()):+.9e} +- "
                  f"{float(z.rad()):.2e}  |dev| / radius {dev / (float(z.rad()) + 1e-300):.2e}  relative "
                  f"{dev / (abs(float(est.mid())) + 1e-300):.2e}  contained {z.contains(arb(est.mid()))}")
    print("COMPARE " + ("ALL CONTAINED" if allok else "NOT ALL CONTAINED"))
    return allok


# ---------------------------------------------------------------- driver

def main():
    a = sys.argv[1:]
    mode = a[0]
    o = {x.split("=", 1)[0][2:]: (x.split("=", 1)[1] if "=" in x else True) for x in a if x.startswith("--")}
    pos = [x for x in a[1:] if not x.startswith("--")]
    ctx.prec = PREC
    chart = load_chart()
    t0 = time.time()

    def log(msg):
        print(f"{msg} [{time.time() - t0:.0f} s]", flush=True)
    if mode == "centre":
        z = [acb(0)] * 6
        D, chosen = run_point(z, chart, log=log)
        J = [D[nu] - D[-nu].conjugate() for nu in (1, 3, 5)]
        out = {"steps": [(k, (h.hex() if h is not None else None)) for k, h in chosen], "D": [s_c(D[m]) for m in ODD],
               "g": [s_c(x) for x in J], "NT": NT, "prec": PREC, "sec": time.time() - t0}
        json.dump(out, open(o["out"], "w"), indent=1)
        log(f"centre: {len(chosen)} steps ({sum(1 for k, _ in chosen if k == 'ray')} ray, "
            f"{sum(1 for k, _ in chosen if k == 'arc')} arc, {sum(1 for k, _ in chosen if k == 'real')} real); "
            f"J = {[complex(x.mid()) for x in J]}")
        return 0
    if mode == "dir":
        st = json.load(open(o["steps"]))
        steps = [(k, (float.fromhex(h) if h is not None else None)) for k, h in st["steps"]]
        spec = o["u"].split(",")
        scale = [1e-10, 1e-10, 1e-18, 1e-18, 1e-19, 1e-19]
        u = [0.0] * 6
        u[int(spec[0])] = scale[int(spec[0])]
        tag = f"e{spec[0]}"
        if len(spec) > 1:
            lam = float(spec[2]) if len(spec) > 2 else 1.0
            u[int(spec[1])] = lam * scale[int(spec[1])]
            tag = f"e{spec[0]}+e{spec[1]}"
        r = float(o.get("r", "1e-3"))
        Ds = []
        for k in range(4):
            zk = acb(r) * Ii ** k
            z = [zk * acb(x) for x in u]
            D, _ = run_point(z, chart, steps=steps)
            Ds.append([s_c(D[m]) for m in ODD])
            log(f"   {tag} point {k} done")
        json.dump({"tag": tag, "u": [repr(x) for x in u], "r": repr(r), "D": Ds, "steps_from": o["steps"],
                   "NT": NT, "prec": PREC, "sec": time.time() - t0}, open(o["out"], "w"), indent=1)
        return 0
    if mode == "compare":
        return 0 if compare(o["box"], pos) else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())

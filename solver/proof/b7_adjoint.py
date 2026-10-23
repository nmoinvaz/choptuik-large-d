#!/usr/bin/env python3
"""Lemma P (proof/B_chain.md, section 8): the adjoint of the inner secant flow along the real axis from xi1 = 0.3 down
to xi_L = 0.09 (z = u* = 0.0081), M = 1, in ball arithmetic, over a COMPLEX parameter box (so that Cauchy estimates
bound its derivatives on Theorem A's real box).

Statement certified here. Let X_C = x^ + [-hr, hr]^2 + i [-hi, hi]^2 in (A, w/w_inf), y+ the lateral solution (the
lower path of proof/A_certify.md section 11: ray 0.1 -> 0.2, arc 0.2, real axis 0.2 -> 0.3; lemma 6b gives it is the
Borel sum), y-(xi; p) = sigma(y+(xi; p^bar)) and psi = y+ - y- on the real axis. For every p in X_C,
    J_D(p) := psi_{D,+1}(0.3; p) = chi(p)^T psi(0.09; p)   with   chi(p) in the enclosure printed below,
in the variables (D_-1, D+1, S_-1, S+1, C_-2, C_0, C_2) of validated_int.py. chi(p) is holomorphic in p.

Proof (B_chain.md section 8). The backward run starts from the hull H of {y+(0.3; p), y-(0.3; p) : p in X_C} (with w
and A as constant states over X_C) and integrates along the real axis down to 0.09 with validated_int's QR-Lohner
integrator (unchanged; lemma6_common's a-priori enclosure). Each step k maps the Lohner set X_k (convex, containing
y+(xi_k; p), y-(xi_k; p) and the segment between them) and encloses the step Jacobian over X_k in an interval matrix
J_k. By the mean value theorem in integral form, psi(xi_{k+1}) = Bbar_k psi(xi_k) with Bbar_k = int_0^1 D Phi_k(y- + s
psi) ds in J_k (its dynamic 7 x 7 block: y+ and y- share the parameters). So psi(0.3) = Bbar_0^-1 ... Bbar_{K-1}^-1
psi(0.09) and J_D = e_{D+1}^T psi(0.3) = chi^T psi(0.09) with chi^T = e^T Bbar_0^-1 ... Bbar_{K-1}^-1, i.e.
chi_{k+1} = Bbar_k^-T chi_k in [J_k]^-T chi_k. The product is carried as a QR-Lohner set (the update of validated_int's
variational columns, with [J_k]^-T in place of J_k). Backward along the real axis the fast modes of psi decay by about
e^-56 between 0.3 and 0.09, so chi grows by about e^56 in the dual fast direction: the direction in which the QR-Lohner
representation keeps relative accuracy.

    python3 b7_adjoint.py box [--hr=1.1e-8] [--hi=1e-8] [--out=b7_box.json]     lower-path run over X_C (state at 0.3)
    python3 b7_adjoint.py adjoint --box=b7_box.json [--out=b7_adjoint.json]       the backward adjoint run
    python3 b7_adjoint.py point [--out=b7_point.json]                             the same two runs at x^ only
    python3 b7_adjoint.py check --adj=b7_adjoint_point.json                       chi vs a double-precision forward
                                         integration of the variational equation along y+ (not rigorous): the
                                         enclosure must contain chi^T v for unit v, to the integration tolerance
    python3 b7_adjoint.py check --adj=... --mutate=transpose     uses [J_k]^-1 in place of [J_k]^-T: must FAIL
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
lemma6_common.install()
import fixed_point as FP  # noqa: E402
from flint import arb, acb, acb_mat, arb_mat, ctx  # noqa: E402

AH, WH = "2.5421480475", "0.7769940136"
R0, RHO, XI1, XIL = "0.1", "0.2", "0.3", "0.09"
SIG = [1, 0, 3, 2, 6, 5, 4]              # sigma on [D-1, D+1, S-1, S+1, C-2, C0, C+2]
OPTS = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
MUT = OPTS.get("mutate")


def s_acb(z):
    return [z.real.str(80, radius=True, more=True), z.imag.str(80, radius=True, more=True)]


def p_acb(p):
    return acb(arb(p[0]), arb(p[1]))


# ----------------------------------------------------------------------------------------------------------------------
# start data over a complex parameter box (fixed_point.start_data, unchanged in substance; accepts acb A, wf)
# ----------------------------------------------------------------------------------------------------------------------
def start_data_c(A, wf, r0="0.1", theta_sign=+1, cauchy=None):
    """as fixed_point.start_data, for complex boxes A, wf (acb): value Y(xi0) over the box and derivatives dY/dA,
    dY/dwf by Cauchy's estimate on the box + cauchy*(1 + i). Returns (xi0, vals, dAs, dws, eta)."""
    ctx.prec = 320
    A, wf = acb(A), acb(wf)
    if cauchy is None:
        cauchy = arb("1e-2")
    cA = A + acb(arb(0, cauchy), arb(0, cauchy))
    cw = wf + acb(arb(0, cauchy), arb(0, cauchy))
    FP.Maj.BOX = False
    centre = FP.run(acb(A.mid()), acb(wf.mid()), r0, verbose=False)
    FP.Maj.BOX = bool(A.rad() > 0 or wf.rad() > 0)
    outX = FP.run(A, wf, r0, verbose=False, kmins_in=centre["kmins"], N_in=centre["N"])
    FP.Maj.BOX = True
    outC = FP.run(cA, cw, r0, verbose=False, kmins_in=centre["kmins"], N_in=centre["N"])
    FP.Maj.BOX = False
    if not (outX["ok"] and outC["ok"]):
        raise RuntimeError("fixed point not verified")

    def eta_bounds(out):
        R_, k_, rho_, wts_, comps_ = out["R"], out["k"], out["rho"], out["wts"], out["comps"]
        return {cp: rho_ / arb(wts_[i]) * R_ ** (k_ + 1 if cp[0] == "c" else k_) for i, cp in enumerate(comps_)}
    N, R = centre["N"], outX["R"]
    eb, ebC = eta_bounds(outX), eta_bounds(outC)
    odd, even, d, s, g = FP.series_jet(A, wf, N)
    phi = theta_sign * 3 * arb.pi() / 8
    r0a = arb(r0)
    xi0 = acb(r0a * phi.cos(), r0a * phi.sin())
    z0 = xi0 * xi0
    zp = [acb(1)]
    for _ in range(N):
        zp.append(zp[-1] * z0)
    W0 = 2 * arb.pi() / arb("1.64389375")
    sm_up = {mu: FP.up(1 + acb(0, 1) * mu * acb(W0) * cw) for mu in odd}
    vals, dAs, dws = [], [], []

    def ball(e):
        r = arb(0, e.upper())
        return acb(r, r)

    def add_eta(jv, e, eC):
        return jv.v + ball(e), jv.dA + ball(eC / cauchy), jv.dw + ball(eC / cauchy)

    for mu in odd:
        Dj = sum((d[j][mu] * zp[j] for j in range(N)), FP.Jet(0)) * xi0
        v, a, w_ = add_eta(Dj, r0a * eb[("p", mu)], r0a * ebC[("p", mu)])
        vals.append(v); dAs.append(a); dws.append(w_)
    for mu in odd:
        Sj = sum((s[j][mu] * zp[j] for j in range(N)), FP.Jet(0)) * z0
        v, a, w_ = add_eta(Sj, R * (eb[("v", mu)] + sm_up[mu] * eb[("p", mu)]),
                           R * (ebC[("v", mu)] + sm_up[mu] * ebC[("p", mu)]))
        vals.append(v); dAs.append(a); dws.append(w_)
    for nu in even:
        Cj = sum((g[j][nu] * zp[j] for j in range(N)), FP.Jet(0))
        v, a, w_ = add_eta(Cj, eb[("c", nu)], ebC[("c", nu)])
        vals.append(v); dAs.append(a); dws.append(w_)
    etab = max(eb.values(), key=lambda x: float(x.upper()))
    etabC = max(ebC.values(), key=lambda x: float(x.upper()))
    return xi0, vals, dAs, dws, (etab, etabC / cauchy)


def fk_start_c(A, wf, r0="0.1", cauchy="1e-3"):
    ctx.prec = 320
    _, vals, dAs, dws, eta = start_data_c(A, wf, r0, +1, arb(cauchy))
    ctx.prec = VI.PREC
    return vals, dAs, dws, eta


VI.fk_start = fk_start_c          # start_lohner_fk calls fk_start; real boxes give the same numbers as before


def legs():
    return [VI.Leg("ray", R0, RHO), VI.Leg("arc", R0, RHO), VI.Leg("real", RHO, XI1)]


def box_run(hr, hi, verbose=False):
    """the lower-path run over X_C in centre-set form (validated_int.start_lohner_fk, var columns): the state set
    {y+(0.3; p) : p in X_C}"""
    t0 = time.time()
    s = VI.Sys(gravity=True)
    ctx.prec = 320
    Ahat, wfhat = arb(AH), arb(WH)
    if float(hr) > 0 or float(hi) > 0:
        A = acb(Ahat + arb(0, arb(hr)), arb(0, arb(hi)))
        wf = acb(wfhat + arb(0, arb(hr)), arb(0, arb(hi)))
        var, cs = True, True
    else:
        A, wf = acb(Ahat), acb(wfhat)
        var, cs = False, False
    ctx.prec = VI.PREC
    lo, eta = VI.start_lohner_fk(s, Ahat, wfhat, A, wf, R0, var=var, centre_set=cs)
    steps = [lo.run(leg, 1e-3, verbose) for leg in legs()]
    X = lo.box()
    return {"A": AH, "wf": WH, "hr": hr, "hi": hi, "steps": steps, "yX_xi1": [s_acb(x) for x in X[:s.nd]],
            "w": s_acb(X[s.nd]), "Abox": s_acb(X[s.nd + 1]), "state_rad": max(VI.rad_of(x) for x in X[:s.nd]),
            "sec": time.time() - t0}


# ----------------------------------------------------------------------------------------------------------------------
# the adjoint run
# ----------------------------------------------------------------------------------------------------------------------
class AdjLohner(VI.Lohner):
    """validated_int.Lohner with an adjoint vector chi carried along: chi_{k+1} in [J_k]^-T chi_k, J_k the enclosure
    of the step Jacobian over the whole set (dynamic 7 x 7 block). The step itself is validated_int.Lohner.step,
    copied unchanged except for the marked block."""

    def set_adjoint(self, chi0):
        nd = self.s.nd
        self.adj = (acb_mat([[chi0[i]] for i in range(nd)]), arb_mat([[1 if i == j else 0 for j in range(2 * nd)]
                                                                      for i in range(2 * nd)]), arb_mat(2 * nd, 1))

    def adj_box(self):
        cA, QA, qA = self.adj
        return cA + VI.unstack(QA * qA)

    def step(self, leg, t0, h):
        s, N, n = self.s, self.N, self.s.n
        X = self.box()
        Yt, T = VI.apriori(s, leg, t0, h, X)
        if Yt is None:
            return False
        Vt = VI.apriori_V(s, leg, T, Yt, h)
        if Vt is None:
            return False
        hh = acb(h)
        hN1 = hh ** (N + 1)
        Yc, _ = VI.taylor_Y(s, leg, acb(t0), self.c, N)
        Yr, ctr = VI.taylor_Y(s, leg, T, Yt, N + 1)
        nd = s.nd
        phic = [VI.evalpoly(Yc[i], hh, N) + hN1 * VI.coef(Yr[i], N + 1) for i in range(nd)] + self.c[nd:]
        _, ctxX = VI.taylor_Y(s, leg, acb(t0), X, N)
        Vk = VI.taylor_V(s, ctxX, N)
        VR = VI.taylor_V(s, ctr, N + 1)
        J = Vk[N]
        for k in range(N - 1, -1, -1):
            J = J * hh + Vk[k]
        PhiN1 = VR[N + 1]
        J = J + hN1 * (PhiN1 * Vt)
        JR = VI.realify(J)
        cnew = [acb(v.mid()) for v in phic]
        e = VI.stack(acb_mat([[phic[i] - cnew[i]] for i in range(n)]))
        A = JR * self.Q
        try:
            Qn = VI.from_np_r(VI.qr_pivot(VI.to_np_r(A), [float(abs(x).upper()) for x in self.q]))
            Qi = Qn.inv()
        except Exception:
            return False
        qn = (Qi * A) * arb_mat([[x] for x in self.q]) + Qi * e
        # ---- adjoint block (new) ----------------------------------------------------------------------------------
        Jyy = acb_mat([[J[i, j] for j in range(nd)] for i in range(nd)])
        try:
            Jinv = Jyy.inv()
        except ZeroDivisionError:
            return False
        M = Jinv if MUT == "transpose" else Jinv.transpose()
        cA, QA, qA = self.adj
        MC = M * cA
        cAn = acb_mat([[acb(MC[i, 0].mid())] for i in range(nd)])
        AV = VI.realify(M) * QA
        try:
            QAn = VI.from_np_r(VI.qr_pivot(VI.to_np_r(AV), [float(abs(qA[i, 0]).upper()) for i in range(2 * nd)]))
            QAi = QAn.inv()
        except Exception:
            return False
        newadj = (cAn, QAn, (QAi * AV) * qA + QAi * VI.stack(MC - cAn))
        # ---- end of the adjoint block --------------------------------------------------------------------------------
        if self.var is not None:
            cV, QV, qV = self.var
            m = cV.ncols()
            JC = J * cV
            cVn = acb_mat([[acb(JC[i, j].mid()) for j in range(m)] for i in range(n)])
            AVv = JR * QV
            QVn = VI.from_np_r(VI.qr_pivot(VI.to_np_r(AVv), [max(float(abs(qV[i, j]).upper()) for j in range(m))
                                                               for i in range(2 * n)]))
            QVi = QVn.inv()
            self.var = (cVn, QVn, (QVi * AVv) * qV + QVi * VI.stack(JC - cVn))
        self.c, self.Q, self.q = cnew, Qn, [qn[i, 0] for i in range(2 * n)]
        self.adj = newadj
        return True


def adjoint_run(ybox, w, Abox, verbose=False):
    """from the hull of y+(0.3) and sigma(y+(0.3)) (over the box) backward along the real axis to 0.09"""
    t0 = time.time()
    ctx.prec = VI.PREC
    s = VI.Sys(gravity=True)
    ys = [ybox[SIG[i]].conjugate() for i in range(7)]
    H = [ybox[i].union(ys[i]) for i in range(7)] + [w, Abox]
    n = s.n
    c = [acb(z.mid()) for z in H]
    q = [H[i] - c[i] for i in range(n)]
    Q = acb_mat([[1 if i == j else 0 for j in range(n)] for i in range(n)])
    lo = AdjLohner(s, c, Q, q, 24)
    e = [acb(0)] * 7
    e[1] = acb(1)                      # chi(0.3) = e_{D,+1}
    lo.set_adjoint(e)
    steps = lo.run(VI.Leg("real", XI1, XIL), 1e-3, verbose)
    chi = lo.adj_box()
    X = lo.box()
    return {"steps": steps, "sec": time.time() - t0, "chi": [s_acb(chi[i, 0]) for i in range(7)],
            "y_xiL": [s_acb(x) for x in X[:7]], "state_rad": max(VI.rad_of(x) for x in X[:7]),
            "chi_rad_rel": max(VI.rad_of(chi[i, 0]) for i in range(7)) / max(float(abs(chi[i, 0]).upper()) for i in range(7)),
            "mutate": MUT}


# ----------------------------------------------------------------------------------------------------------------------
# check against a double-precision forward integration of the variational equation along y+ (NOT rigorous)
# ----------------------------------------------------------------------------------------------------------------------
def check(adj_json, start_json):
    from scipy.integrate import solve_ivp
    import mpmath as mp
    with open(adj_json) as fh:
        adj = json.load(fh)
    with open(start_json) as fh:
        st = json.load(fh)
    chi = [p_acb(p) for p in adj["chi"]]
    W0 = float(2 * mp.pi / mp.mpf("1.64389375"))
    w = W0 * float(WH)
    ODD, EVEN = (-1, 1), (-2, 0, 2)

    def pr(X, Y, keep):
        out = {k: 0j for k in keep}
        for a, xa in X.items():
            for b, yb in Y.items():
                if a + b in out:
                    out[a + b] += xa * yb
        return out

    def rhs(xi, Y):
        D = {-1: Y[0], 1: Y[1]}; S = {-1: Y[2], 1: Y[3]}; C = {-2: Y[4], 0: Y[5], 2: Y[6]}
        CS = pr(C, S, ODD)
        dD = {m: (CS[m] - 1j * m * w * xi * D[m]) / xi ** 2 for m in ODD}
        E = {k: v / 2 for k, v in pr(D, D, EVEN).items()}
        cE = pr(C, E, EVEN)
        xdD = {m: xi * dD[m] for m in ODD}
        a, b, c = pr(C, xdD, ODD), pr(cE, D, ODD), pr(C, D, ODD)
        r2 = {m: a[m] + b[m] - c[m] for m in ODD}
        dS = {m: (r2[m] - 1j * m * w * xi * S[m]) / xi ** 2 for m in ODD}
        return np.array([dD[-1], dD[1], dS[-1], dS[1], cE[-2] / xi, cE[0] / xi, cE[2] / xi])

    def jac(xi, Y, h=1e-7):
        Jm = np.zeros((7, 7), complex)
        for j in range(7):
            e = np.zeros(7, complex); e[j] = h
            Jm[:, j] = (rhs(xi, Y + e) - rhs(xi, Y - e)) / (2 * h)
        return Jm
    y03 = np.array([complex(float(arb(p[0]).mid()), float(arb(p[1]).mid())) for p in st["yX_xi1"]])
    back = solve_ivp(lambda t, Y: rhs(t, Y), (0.3, 0.09), y03, method="DOP853", rtol=1e-13, atol=1e-15,
                     dense_output=True)
    worst = 0.0
    for j in range(7):
        v = np.zeros(7, complex); v[j] = 1.0
        fw = solve_ivp(lambda t, d: jac(t, back.sol(t)) @ d, (0.09, 0.3), v, method="DOP853", rtol=1e-12,
                       atol=1e-10)
        num = fw.y[1, -1]                # e_{D+1}^T Phi(0.09 -> 0.3) e_j
        ref = chi[j]
        err = abs(complex(float(ref.real.mid()), float(ref.imag.mid())) - num)
        rad = VI.rad_of(ref)
        rel = max(err - rad, 0) / abs(num)
        worst = max(worst, rel)
        print(f"   j = {j}: chi_j = {complex(float(ref.real.mid()), float(ref.imag.mid())):.10e} (radius {rad:.1e}); "
              f"forward double precision {num:.10e}; excess beyond the ball, relative {rel:.1e}")
    return worst


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "point"
    if cmd == "box":
        out = box_run(OPTS.get("hr", "1.1e-8"), OPTS.get("hi", "1e-8"), verbose=bool(OPTS.get("verbose")))
        with open(os.path.join(HERE, OPTS.get("out", "b7_box.json")), "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"lower-path run over X_C: steps {out['steps']}, state radius at 0.3 {out['state_rad']:.2e}, "
              f"{out['sec']:.0f} s")
    elif cmd == "point":
        out = box_run("0", "0")
        with open(os.path.join(HERE, OPTS.get("out", "b7_box_point.json")), "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"point run: steps {out['steps']}, state radius {out['state_rad']:.2e}, {out['sec']:.0f} s")
    elif cmd == "adjoint":
        with open(os.path.join(HERE, OPTS["box"])) as fh:
            bx = json.load(fh)
        yb = [p_acb(p) for p in bx["yX_xi1"]]
        res = adjoint_run(yb, p_acb(bx["w"]), p_acb(bx["Abox"]), verbose=bool(OPTS.get("verbose")))
        res["box"] = OPTS["box"]
        with open(os.path.join(HERE, OPTS.get("out", "b7_adjoint.json")), "w") as fh:
            json.dump(res, fh, indent=1)
        print(f"adjoint run 0.3 -> 0.09: {res['steps']} steps, {res['sec']:.0f} s; state radius at 0.09 "
              f"{res['state_rad']:.2e}; chi relative radius {res['chi_rad_rel']:.2e}")
        for i, c in enumerate(res["chi"]):
            print(f"   chi[{i}] = {c[0][:24]} + i {c[1][:24]}")
    elif cmd == "check":
        worst = check(os.path.join(HERE, OPTS["adj"]), os.path.join(HERE, OPTS.get("start", "b7_box_point.json")))
        ok = worst < 1e-6
        print(f"adjoint vs forward integration: worst relative excess {worst:.2e} (< 1e-6): {'PASS' if ok else 'FAIL'}")
        if MUT:
            print(f"MUTATION {MUT}: the check must FAIL: {'caught' if not ok else 'NOT caught'}")
            sys.exit(0 if not ok else 3)
        sys.exit(0 if ok else 1)

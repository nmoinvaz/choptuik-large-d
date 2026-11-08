"""Second-order coefficients omega_2 (Delta_2) and lambda_2 of the M-harmonic inner model with deflated jets, for
M = 5, where the one-shot linear algebra of second_order_run.py is beyond double precision (as first_order_hier.py at
first order; notes/SECOND_ORDER.md, section 4). It starts from first_order_hier.py's results
(data/first_order_hier_M{M}_xi{xi1}.json: the zero, its Jacobian K, dx = (K_1, omega_1) and dx_gauge0, lambda*, the
deflated eigenprofile f_0, ell_l, the denominator ell_l . Sch'(lambda*) e and lambda_1).

Deflation (first_order_hier.py): a direction whose own higher-harmonic jump r_H is large carries, through the Schur
correction Lambda = K_lH K_HH^-1 (~1e8 at M = 5), an error rtol |r_H| into the harmonic-1 rows. So the mixed
component e_1 e_2 (twice the 1/n^2 part) gets higher-harmonic profile data C added, refined over integrations until
its r_H stops falling, and the harmonic-1 condition is then solved with the deflated A and omega directions:
    2 dx2 = C + s_A c_A + s_w c_w,   (s_A, s_w) = -Sch^-1 corr(J_12(C)),   omega_2 = omega_inf s_w / 2.
lambda_2: the p e_1 component (f_1 = 0) gets inner profile data refined until its r_H is small, and its harmonic-1
part t (f_1[0] = 0) from Sch(lambda*) (0, t) = -corr(J_p1); the p e_1 e_2 component (f_2 = 0) is deflated the same
way and lambda_2 = -(ell_l . corr(J_p12)/2) / (ell_l . Sch' e).

    python3 -u second_order_hier.py M [--xi1=0.2] [--rtol=1e-12] [--raw=DIR] [--redo=omega2,gauge,lam2] [--no-gauge]
                                     [--omega-only]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import first_order_run as R  # noqa: E402
import first_order_hier as H  # noqa: E402
import second_order_run as S2  # noqa: E402
from second_order_jet import JetModel2, Dir, Fh_from, dFh, W0  # noqa: E402

T0 = time.time()


def log(*a):
    print(*a, f"[{time.time() - T0:.0f} s]", flush=True)


def inner_prof(C, M, scale=1.0):
    """profile harmonics of inner (higher-harmonic) parameter coefficients C (parameters 2..)"""
    out = {}
    for i, c in enumerate(C):
        for k, v in dFh(i + 2, M).items():
            out[k] = out.get(k, 0) + scale * c * v
    return out


def omega2_deflated(x, M, xi1, dx, K, L, gscale=1.0):
    n = R.npar(M)
    gauge1, gauge2 = S2.gauge_bg(x, M, dx, gscale)
    K1 = S2.prof_of_dx(dx, M)
    Fh = Fh_from(x, M)
    w = W0 * x[1]

    def run(C):
        cA = np.zeros(n); cA[0] = 1; cA[2:] = C[0]
        cw = np.zeros(n); cw[1] = 1; cw[2:] = C[1]
        dirs, masks, data = H.param_dirs(M, [cA, cw])
        e1, e2 = 1 << 2, 1 << 3
        dirs += [Dir("e1", q1=True, omega=W0 * dx[1]), Dir("e2", q1=True, omega=W0 * dx[1])]
        masks += [e1, e2, e1 | e2]
        data[e1] = (K1, gauge1)
        data[e2] = (K1, gauge1)
        data[e1 | e2] = (inner_prof(C[2], M), {k: 2 * v for k, v in gauge2.items()})
        jm = JetModel2(M, w, Fh, dirs, masks, data)
        Rr, _ = R.residuals(jm, x, M, xi1)
        run.J1 = Rr[jm.idx[e1]]
        return [Rr[1], Rr[2], Rr[jm.idx[e1 | e2]]]
    C0 = [L.deflate(K[2:, 0]), L.deflate(K[2:, 1]), np.zeros(n - 2)]
    rows0 = run(C0)
    log(f"   pass 1: |r_H| of e_1 e_2 {np.linalg.norm(rows0[2][2:]):.2e}, |J_e1| {np.linalg.norm(run.J1):.2e} "
        f"(r_H {np.linalg.norm(run.J1[2:]):.2e})")
    C0[2] = L.deflate(rows0[2][2:])
    rows, C, hist = H.refine(run, C0, L, nit=6, label="e1e2 deflation")
    Sch = np.column_stack([L.corr(rows[0]), L.corr(rows[1])])
    Jc = L.corr(rows[2])
    st = -np.linalg.solve(Sch, Jc)
    cA = np.zeros(n); cA[0] = 1; cA[2:] = C[0]
    cw = np.zeros(n); cw[1] = 1; cw[2:] = C[1]
    c2 = np.zeros(n); c2[2:] = C[2]
    dx2 = (c2 + st[0] * cA + st[1] * cw) / 2
    return dx2, dict(J12_final=rows[2].tolist(), J12H_final=float(np.linalg.norm(rows[2][2:])), e12_hist=hist,
                     J1=run.J1.tolist(), Sch_omega2=Sch.tolist(), J12_corr=Jc.tolist(),
                     J12H_pass1=float(np.linalg.norm(rows0[2][2:])))


def lam2_deflated(x, M, xi1, dx, dx2, first):
    lam, lam1 = first["lam"], first["lam1"]
    f0vec = np.array(first["fvec"])
    ll = np.array(first["ell_l"])
    den = first["lam_den"]
    Fh = Fh_from(x, M)
    w = W0 * x[1]
    f0 = R.profile_of(f0vec, M)
    gauge1, gauge2 = S2.gauge_bg(x, M, dx)
    pre = JetModel2(M, w, Fh, [Dir("pert", lam=lam)], [0, 1], {1: (f0, {})})
    pre.series_at(1.0, 3, stop=False)
    dgauge1 = S2.harm_dict(-pre.ser["G"][1][1], pre.L, 0, M)
    S, _, _ = R.pert_matrix(x, M, xi1, lam)
    Lp = H.Lin(S)
    Sch, Cm, _, resH = H.sch_lam(x, M, xi1, lam, S=S)
    log(f"   Sch(lambda*) singular values {np.linalg.svd(Sch, compute_uv=False)} (r_H of its deflated columns {resH:.1e})")
    basis = R.pert_basis(M)

    def vec_prof(v):
        return R.profile_of(v, M)

    def run(f1vec, top_prof, dgauge2):
        dirs, masks, data, wextra, (e1, e2) = S2.bg_dirs(x, M, dx, gauge1, gauge2, dx2=dx2, with_params=False)
        p = 1 << len(dirs)
        dirs.append(Dir("pert", lam=lam))
        masks += [p, p | e1, p | e2]
        data[p] = (f0, {})
        data[p | e1] = (vec_prof(f1vec), dgauge1)
        data[p | e2] = (vec_prof(f1vec), dgauge1)
        if top_prof is not None:
            masks.append(p | e1 | e2)
            data[p | e1 | e2] = (top_prof, {k: 2 * v for k, v in dgauge2.items()})
        jm = JetModel2(M, w, Fh, dirs, masks, data, wextra=wextra, lamj={e1: lam1, e2: lam1})
        Rr, _ = R.residuals(jm, x, M, xi1)
        return jm, Rr, (p, e1, e2)
    # f_1: deflate the p e_1 component's inner profile, then its harmonic-1 part t from Sch (0, t) = -corr(J_p1)
    nb = len(basis)

    def run1(Cin):
        fv = np.concatenate([[0.0, 0.0], Cin[0]])
        jm, Rr, (p, e1, e2) = run(fv, None, None)
        run1.bg = (Rr[jm.idx[e1]], Rr[jm.idx[e1 | e2]])
        return [Rr[jm.idx[p | e1]]]
    rows, Cin, hist1 = H.refine(run1, [np.zeros(nb - 2)], Lp, nit=6, label="p e1 deflation")
    J1c = Lp.corr(rows[0])
    log(f"   background with the second-order selection: |J_e1| {np.linalg.norm(run1.bg[0]):.2e}, "
        f"|J_e12| {np.linalg.norm(run1.bg[1]):.2e}")
    ellJ1 = float(ll @ J1c)
    # columns of the deflated harmonic-1 profiles (sch_lam): Cm (2N x 2); f_1 = inner + t * Cm[:, 1] (f_1[0] = 0)
    t = float(np.linalg.lstsq(Sch[:, 1:2], -J1c, rcond=None)[0][0])
    f1vec = np.concatenate([[0.0, 0.0], Cin[0]]) + t * Cm[:, 1]
    log(f"   f_1 = {f1vec.tolist()};  ell_l . corr(J_p1) {ellJ1:.3e} against |corr(J_p1)| {np.linalg.norm(J1c):.3e} "
        f"(zero by the definition of lambda_1)")
    # the perturbation's second gauge datum
    pre = JetModel2(M, w, Fh, [Dir("pert", lam=lam), Dir("e1", q1=True, omega=W0 * dx[1])], [0, 1, 2, 3],
                    {1: (f0, {}), 2: (S2.prof_of_dx(dx, M), gauge1), 3: (vec_prof(f1vec), dgauge1)}, lamj={2: lam1})
    pre.series_at(1.0, 3, stop=False)
    dgauge2 = S2.harm_dict(-(pre.ser["G"][2][pre.idx[1]] + pre.ser["G"][1][pre.idx[3]]), pre.L, 0, M)

    def run2(Cin):
        fv = np.concatenate([[0.0, 0.0], Cin[0]])
        jm, Rr, (p, e1, e2) = run(f1vec, vec_prof(fv), dgauge2)
        run2.Jp1 = Rr[jm.idx[p | e1]]
        return [Rr[jm.idx[p | e1 | e2]]]
    rows2, Cin2, hist2 = H.refine(run2, [np.zeros(nb - 2)], Lp, nit=6, label="p e1 e2 deflation")
    num = float(ll @ Lp.corr(rows2[0])) / 2
    lam2 = -num / den
    return dict(lam2=lam2, lam2_num=num, f1vec=f1vec.tolist(), ellJp1=ellJ1, Jp1_after=float(np.linalg.norm(Lp.corr(run2.Jp1))),
                p1_hist=hist1, p12_hist=hist2, Jp12_final=rows2[0].tolist(), lam=lam, lam1=lam1, lam_den=den)


def main():
    argv = sys.argv[1:]
    opts = {a.split("=")[0]: (a.split("=", 1)[1] if "=" in a else True) for a in argv if a.startswith("--")}
    args = [a for a in argv if not a.startswith("--")]
    M = int(args[0])
    xi1 = float(opts.get("--xi1", 0.2))
    R.RTOL = float(opts.get("--rtol", 1e-12))
    tag = f"M{M}_xi{xi1:g}" + (f"_rtol{R.RTOL:g}" if R.RTOL != 1e-12 else "")
    first = json.load(open(os.path.join(HERE, "data", f"first_order_hier_{tag}.json")))
    raw = opts.get("--raw", os.path.join(HERE, "data"))
    fn = os.path.join(raw, f"second_order_hier_{tag}.json")
    out = json.load(open(fn)) if os.path.exists(fn) else dict(M=M, xi1=xi1, rtol=R.RTOL)
    for key in (opts.get("--redo", "").split(",") if "--redo" in opts else []):
        for k2 in {"omega2": ["dx2"], "gauge": ["omega2_gauge0"], "lam2": ["lam2", "lam2_gauge0"]}[key]:
            out.pop(k2, None)

    def save():
        json.dump(out, open(fn, "w"), indent=1)
    x, dx, K = np.array(first["x"]), np.array(first["dx"]), np.array(first["K"])
    L = H.Lin(K)
    w, w1 = W0 * x[1], W0 * dx[1]
    log(f"M {M} xi1 {xi1} rtol {R.RTOL}: zero {x.tolist()}  omega* {w:.9f}  omega_1 {w1:.6f} (first order)")
    if "dx2" not in out:
        dx2, info = omega2_deflated(x, M, xi1, dx, K, L)
        out.update(dx2=dx2.tolist(), omega2=W0 * dx2[1], **info)
        save()
    dx2 = np.array(out["dx2"])
    w2 = W0 * dx2[1]
    D1, D2 = S2.deltas(w, w1, w2)
    out.update(Delta_star=2 * np.pi / w, Delta1=D1, Delta2=D2)
    save()
    log(f"OMEGA2: dx2 {dx2.tolist()}  (r_H of e_1 e_2: pass 1 {out['J12H_pass1']:.1e}, deflated {out['J12H_final']:.1e}; "
        f"|J_e1| {np.linalg.norm(out['J1']):.1e})")
    log(f"   omega = {w:.9f} + ({w1:.6f})/n + ({w2:.5f})/n^2;  Delta = {2 * np.pi / w:.9f} + ({D1:.5f})/n + ({D2:.4f})/n^2")
    if "omega2_gauge0" not in out and "--no-gauge" not in opts:
        dxg = np.array(first["dx_gauge0"])
        dx2g, _ = omega2_deflated(x, M, xi1, dxg, K, L, gscale=0.0)
        out.update(omega2_gauge0=W0 * dx2g[1], dx2_gauge0=dx2g.tolist(), omega1_gauge0=W0 * dxg[1])
        save()
    if "omega2_gauge0" in out:
        D1g, D2g = S2.deltas(w, out["omega1_gauge0"], out["omega2_gauge0"])
        out.update(Delta2_gauge0=D2g)
        save()
        log(f"GAUGE: omega_2 {w2:.4f} (light-cone gauge) against {out['omega2_gauge0']:.4f} (all gauge data 0); "
            f"Delta_2 {D2:.4f} against {D2g:.4f}")
    if "--omega-only" in opts:
        return
    if "lam2" not in out:
        out.update(lam2_deflated(x, M, xi1, dx, dx2, first))
        save()
    log(f"   p e1 deflation {out['p1_hist']}; p e1 e2 deflation {out['p12_hist']}; |corr(J_p1)| after f_1 {out['Jp1_after']:.2e}")
    log(f"LAMBDA2: lambda = {out['lam']:.9f} + ({out['lam1']:.5f})/n + ({out['lam2']:.4f})/n^2")
    log(f"wrote {fn}")


if __name__ == "__main__":
    main()

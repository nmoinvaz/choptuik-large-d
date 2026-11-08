"""Second-order (1/n^2) coefficients of omega and of the growth rate lambda = 1/gamma from the M-harmonic inner model,
with one-shot jets (M = 1, 3; notes/SECOND_ORDER.md, section 4). It starts from the first-order results of
first_order_run.py (data/first_order_M{M}_xi{xi1}.json: the zero x, dx = (K_1, omega_1), lambda*, its eigenprofile
f_0, the left null vector ell, lambda_1).

Steps (e_1, e_2 nilpotent with e = e_1 + e_1 = 1/n; the e_1 e_2 component holds twice the 1/n^2 part):
  1. omega2  the background to O(1/n^2): e_1, e_2 carry the selected first-order data (K_1, omega_1, c_1(0) = -g_1),
             e_1 e_2 the gauge 2 c_2(0) = -2 (g_2 + h_1) and no K_2, omega_2. Its jump J_12 = 2 S_rest; the
             corrections dx2 = (A_2, omega_2/omega_inf, B_3,2, ...) solve (dr/dx) dx2 = -J_12/2, and omega = omega* +
             omega_1/n + omega_2/n^2 + ... A rerun with dx2 entered must give J_12 = 0 (check).
  2. gauge   the same with every gauge datum 0 (c_1(0) = c_2(0) = 0, with the gauge-0 first-order selection):
             independent of the gauge in the full problem, so the difference measures the truncation.
  3. lam2    the perturbation e^{lambda tau} f to second order: f = f_0 + e f_1 + e^2 f_2 (f[0] = 1 at every
             order), lambda = lambda* + e lambda_1 + e^2 lambda_2 as a jet. First f_1 from S(lambda*) f_1 = -J_p1
             (J_p1: the jump of the p e_1 component with f_1 = 0, which holds S_1 f_0 + lambda_1 dS/dlambda f_0;
             ell . J_p1 = 0 by the definition of lambda_1, a check). Then the gauge 2 dc_2(0) = -2 (dg_2 + dh_1), and
             the p e_1 e_2 component's jump J_p12 = 2 (S_2 f_0 + S_1 f_1 + ... with f_2 = 0, lambda_2 = 0) gives
             lambda_2 = -(ell . J_p12/2) / (ell . dS/dlambda f_0).

    python3 -u second_order_run.py M [--xi1=0.2] [--rtol=1e-12] [--redo=omega2,gauge,lam2] [--raw=DIR]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import first_order_run as R  # noqa: E402
from second_order_jet import JetModel2, Dir, Fh_from, dFh, W0, F2h  # noqa: E402

T0 = time.time()
MUTATE = set()        # 'pgauge2': the perturbation's second gauge datum dropped (second_order_checks.py modes)


def log(*a):
    print(*a, f"[{time.time() - T0:.0f} s]", flush=True)


def prof_of_dx(dx, M, scale=1.0):
    """profile harmonics of a parameter vector (p != 1)"""
    out = {}
    for p in range(R.npar(M)):
        if p == 1 or dx[p] == 0:
            continue
        for k, v in dFh(p, M).items():
            out[k] = out.get(k, 0) + scale * dx[p] * v
    return out


def harm_dict(vec, L, parity, M):
    lim = M if parity == 1 else M + 1
    return {int(k): complex(vec[k + L]) for k in range(-lim, lim + 1) if (k % 2 == parity) and vec[k + L] != 0}


def gauge_bg(x, M, dx, gscale=1.0):
    """c_1(0) = -g_1 and c_2(0) = -(g_2 + h_1) from a preliminary series (value + one 1/n direction with K_1,
    omega_1, c_1(0)); gscale = 0 sets both to zero"""
    Fh = Fh_from(x, M)
    w = W0 * x[1]
    g1 = F2h(Fh, M)
    gauge1 = {k: -gscale * v for k, v in g1.items()}
    jm = JetModel2(M, w, Fh, [Dir("e1", q1=True, omega=W0 * dx[1])], [0, 1], {1: (prof_of_dx(dx, M), gauge1)})
    jm.series_at(1.0, 3, stop=False)
    k0 = -gscale * (jm.ser["G"][2][0] + jm.ser["G"][1][1])
    return gauge1, harm_dict(k0, jm.L, 0, M)


def bg_dirs(x, M, dx, gauge1, gauge2, dx2=None, with_params=True):
    """parameter directions (optional) + e_1, e_2 (+ e_1 e_2 with 2 K_2, 2 omega_2 from dx2, 2 c_2(0))"""
    n = R.npar(M)
    dirs, masks, data = [], [0], {}
    if with_params:
        for p in range(n):
            dirs.append(Dir(f"x{p}", omega=W0 if p == 1 else 0.0))
            masks.append(1 << p)
            data[1 << p] = (dFh(p, M), {})
    b = len(dirs)
    e1, e2 = 1 << b, 1 << (b + 1)
    dirs += [Dir("e1", q1=True, omega=W0 * dx[1]), Dir("e2", q1=True, omega=W0 * dx[1])]
    K1 = prof_of_dx(dx, M)
    masks += [e1, e2, e1 | e2]
    K2 = prof_of_dx(dx2, M, 2.0) if dx2 is not None else {}
    data[e1] = (K1, gauge1)
    data[e2] = (K1, gauge1)
    data[e1 | e2] = (K2, {k: 2 * v for k, v in gauge2.items()})
    wextra = {e1 | e2: 2 * W0 * dx2[1]} if dx2 is not None else {}
    return dirs, masks, data, wextra, (e1, e2)


def omega2_step(x, M, xi1, dx, gscale=1.0):
    n = R.npar(M)
    gauge1, gauge2 = gauge_bg(x, M, dx, gscale)
    dirs, masks, data, wextra, (e1, e2) = bg_dirs(x, M, dx, gauge1, gauge2)
    jm = JetModel2(M, W0 * x[1], Fh_from(x, M), dirs, masks, data, wextra=wextra)
    Rr, info = R.residuals(jm, x, M, xi1)
    r0, K = Rr[0], Rr[1:1 + n].T
    J1, J12 = Rr[jm.idx[e1]], Rr[jm.idx[e1 | e2]]
    dx2 = -R.schur_solve(K, J12 / 2)
    return dx2, dict(J1=J1.tolist(), J12=J12.tolist(), r0=r0.tolist(), gauge2={str(k): [v.real, v.imag] for k, v in gauge2.items()})


def verify_omega2(x, M, xi1, dx, dx2, gscale=1.0):
    gauge1, gauge2 = gauge_bg(x, M, dx, gscale)
    dirs, masks, data, wextra, (e1, e2) = bg_dirs(x, M, dx, gauge1, gauge2, dx2=dx2, with_params=False)
    jm = JetModel2(M, W0 * x[1], Fh_from(x, M), dirs, masks, data, wextra=wextra)
    Rr, info = R.residuals(jm, x, M, xi1)
    return Rr[jm.idx[e1]], Rr[jm.idx[e1 | e2]]


def lam2_step(x, M, xi1, dx, dx2, lam, f0vec, ell, lam1, ellJel, lam1_free=False, gscale=1.0):
    """lam1_free: lambda_1 is computed here from the p e_1 jump (-ell . J_p1 / ell . dS f_0) instead of given"""
    Fh = Fh_from(x, M)
    w = W0 * x[1]
    f0 = R.profile_of(f0vec, M)
    gauge1, gauge2 = gauge_bg(x, M, dx, gscale)
    # the perturbation's first gauge datum: dc_1(0) = -dg_1 (xi^2 coefficient of dc_0)
    pre = JetModel2(M, w, Fh, [Dir("pert", lam=lam)], [0, 1], {1: (f0, {})})
    pre.series_at(1.0, 3, stop=False)
    dgauge1 = harm_dict(-gscale * pre.ser["G"][1][1], pre.L, 0, M)

    def run(f1vec, dgauge2, top):
        dirs, masks, data, wextra, (e1, e2) = bg_dirs(x, M, dx, gauge1, gauge2, dx2=dx2, with_params=False)
        p = 1 << len(dirs)
        dirs.append(Dir("pert", lam=lam))
        f1 = R.profile_of(f1vec, M)
        masks += [p, p | e1, p | e2] + ([p | e1 | e2] if top else [])
        data[p] = (f0, {})
        data[p | e1] = (f1, dgauge1)
        data[p | e2] = (f1, dgauge1)
        if top:
            data[p | e1 | e2] = ({}, {k: 2 * v for k, v in dgauge2.items()})
        jm = JetModel2(M, w, Fh, dirs, masks, data, wextra=wextra, lamj={e1: lam1, e2: lam1})
        Rr, info = R.residuals(jm, x, M, xi1)
        return jm, Rr, (p, e1, e2)
    # f_1
    S0, dS0, _ = R.pert_matrix(x, M, xi1, lam)
    # the denominator ell . dS/dlambda f_0, from this model's own d/dlambda direction (the value passed in is only
    # compared: the stored first-order 'Jel' of data/first_order_M3_xi0.2.json does not match its 'ell')
    ellJel_in = ellJel
    ellJel = float(ell @ dS0 @ f0vec)
    log(f"   ell . dS/dlambda f_0 = {ellJel:.6e} (passed in: {ellJel_in:.6e})")
    jm, Rr, (p, e1, e2) = run(np.zeros(len(f0vec)), None, False)
    Jp1 = Rr[jm.idx[p | e1]]
    if lam1_free:
        lam1 = -float(ell @ Jp1) / ellJel
        jm, Rr, (p, e1, e2) = run(np.zeros(len(f0vec)), None, False)
        Jp1 = Rr[jm.idx[p | e1]]
    ellJp1 = float(ell @ Jp1)
    sol, res, rk, sv = np.linalg.lstsq(S0[:, 1:], -Jp1, rcond=None)
    f1vec = np.concatenate([[0.0], sol])
    log(f"   f_1 = {f1vec.tolist()};  ell . J_p1 = {ellJp1:.3e} against |J_p1| {np.linalg.norm(Jp1):.3e} "
        f"(zero by the definition of lambda_1); residual of the least-squares solve {np.linalg.norm(S0 @ f1vec + Jp1):.2e}")
    # the perturbation's second gauge datum: dc_2(0) = -(dg_2 + dh_1)
    f1 = R.profile_of(f1vec, M)
    pre = JetModel2(M, w, Fh, [Dir("pert", lam=lam), Dir("e1", q1=True, omega=W0 * dx[1])], [0, 1, 2, 3],
                    {1: (f0, {}), 2: (prof_of_dx(dx, M), gauge1), 3: (f1, dgauge1)}, lamj={2: lam1})
    pre.series_at(1.0, 3, stop=False)
    dgauge2 = harm_dict(-gscale * (pre.ser["G"][2][pre.idx[1]] + pre.ser["G"][1][pre.idx[3]]), pre.L, 0, M)
    if "pgauge2" in MUTATE:
        dgauge2 = {}
    jm, Rr, (p, e1, e2) = run(f1vec, dgauge2, True)
    Jp1b = Rr[jm.idx[p | e1]]
    Jp12 = Rr[jm.idx[p | e1 | e2]]
    num = float(ell @ (Jp12 / 2))
    lam2 = -num / ellJel
    return dict(lam2=lam2, lam2_num=num, f1vec=f1vec.tolist(), ellJp1=ellJp1, lam1_used=lam1, ellJel_used=ellJel,
                Jp1_norm=float(np.linalg.norm(Jp1)), Jp1_after=float(np.linalg.norm(Jp1b)),
                Jp12=Jp12.tolist(), dgauge1={str(k): [v.real, v.imag] for k, v in dgauge1.items()},
                dgauge2={str(k): [v.real, v.imag] for k, v in dgauge2.items()})


def deltas(w, w1, w2):
    D1 = -2 * np.pi * w1 / w ** 2
    D2 = 2 * np.pi * (w1 ** 2 / w ** 3 - w2 / w ** 2)
    return D1, D2


def main():
    argv = sys.argv[1:]
    opts = {a.split("=")[0]: (a.split("=", 1)[1] if "=" in a else True) for a in argv if a.startswith("--")}
    args = [a for a in argv if not a.startswith("--")]
    M = int(args[0])
    xi1 = float(opts.get("--xi1", 0.2))
    R.RTOL = float(opts.get("--rtol", 1e-12))
    tag = f"M{M}_xi{xi1:g}" + (f"_rtol{R.RTOL:g}" if R.RTOL != 1e-12 else "")
    first = json.load(open(os.path.join(HERE, "data", f"first_order_{tag}.json")))
    hier = os.path.join(HERE, "data", f"first_order_hier_{tag}.json")
    if "dx_gauge0" not in first and os.path.exists(hier):
        first["dx_gauge0"] = json.load(open(hier))["dx_gauge0"]        # the same model's gauge-0 selection
    raw = opts.get("--raw", os.path.join(HERE, "data"))
    fn = os.path.join(raw, f"second_order_{tag}.json")
    out = json.load(open(fn)) if os.path.exists(fn) else dict(M=M, xi1=xi1, rtol=R.RTOL)
    for key in (opts.get("--redo", "").split(",") if "--redo" in opts else []):
        for k2 in {"omega2": ["dx2"], "gauge": ["omega2_gauge0"], "lam2": ["lam2", "lam2_gauge0"]}[key]:
            out.pop(k2, None)

    def save():
        json.dump(out, open(fn, "w"), indent=1)
    x, dx = np.array(first["x"]), np.array(first["dx"])
    w = W0 * x[1]
    w1 = W0 * dx[1]
    log(f"M {M} xi1 {xi1} rtol {R.RTOL}: zero {x.tolist()}  omega* {w:.9f}  omega_1 {w1:.6f} (first order)")
    if "dx2" not in out:
        dx2, info = omega2_step(x, M, xi1, dx)
        out.update(dx2=dx2.tolist(), omega2=W0 * dx2[1], **info)
        save()
    dx2 = np.array(out["dx2"])
    w2 = W0 * dx2[1]
    log(f"   |J_e1| (first-order selected: ~0) {np.linalg.norm(out['J1']):.2e};  |J_e12| {np.linalg.norm(out['J12']):.3e}")
    D1, D2 = deltas(w, w1, w2)
    log(f"OMEGA2: dx2 {dx2.tolist()}")
    log(f"   omega = {w:.9f} + ({w1:.6f})/n + ({w2:.5f})/n^2;  Delta = {2 * np.pi / w:.9f} + ({D1:.5f})/n + ({D2:.4f})/n^2")
    out.update(Delta_star=2 * np.pi / w, Delta1=D1, Delta2=D2)
    save()
    if "verify_J12" not in out:
        J1v, J12v = verify_omega2(x, M, xi1, dx, dx2)
        out.update(verify_J1=float(np.linalg.norm(J1v)), verify_J12=float(np.linalg.norm(J12v)))
        save()
    log(f"   check: with dx2 entered, |J_e12| = {out['verify_J12']:.2e} (before {np.linalg.norm(out['J12']):.2e})")
    if "omega2_gauge0" not in out and "--no-gauge" not in opts and "dx_gauge0" in first:
        dxg = np.array(first["dx_gauge0"])
        dx2g, _ = omega2_step(x, M, xi1, dxg, gscale=0.0)
        out.update(omega2_gauge0=W0 * dx2g[1], dx2_gauge0=dx2g.tolist(), omega1_gauge0=W0 * dxg[1])
        save()
    if "omega2_gauge0" in out:
        D1g, D2g = deltas(w, out["omega1_gauge0"], out["omega2_gauge0"])
        out.update(Delta2_gauge0=D2g)
        save()
        log(f"GAUGE: omega_2 {w2:.4f} (light-cone gauge) against {out['omega2_gauge0']:.4f} (all gauge data 0); "
            f"Delta_2 {D2:.4f} against {D2g:.4f}")
    if "--omega-only" in opts:
        return
    if "lam2" not in out and "lam" in first:
        lam, lam1 = first["lam"], first["lam1"]
        f0vec, ell = np.array(first["fvec"]), np.array(first["ell"])
        ellJel = float(ell @ np.array(first["Jel"]))
        res = lam2_step(x, M, xi1, dx, dx2, lam, f0vec, ell, lam1, ellJel)
        out.update(res, lam=lam, lam1=lam1, ellJel=res["ellJel_used"], ellJel_first_json=ellJel)
        save()
    if "lam2" in out:
        log(f"   check: ell . J_p1 {out['ellJp1']:.2e}; |J_p1| after f_1 {out['Jp1_after']:.2e}")
        log(f"LAMBDA2: lambda = {out['lam']:.9f} + ({out['lam1']:.5f})/n + ({out['lam2']:.4f})/n^2")
    if "lam2" in out and "lam2_gauge0" not in out and "dx2_gauge0" in out and "--no-gauge" not in opts:
        # every gauge datum 0 (background and perturbation), with the gauge-0 selections; lambda_1 recomputed
        f0vec, ell = np.array(first["fvec"]), np.array(first["ell"])
        g0 = lam2_step(x, M, xi1, np.array(first["dx_gauge0"]), np.array(out["dx2_gauge0"]), out["lam"], f0vec, ell,
                       0.0, out["ellJel"], lam1_free=True, gscale=0.0)
        out.update(lam1_gauge0=g0["lam1_used"], lam2_gauge0=g0["lam2"])
        save()
    if "lam2_gauge0" in out:
        log(f"GAUGE: lambda_1 {out['lam1']:.5f}, lambda_2 {out['lam2']:.4f} (light-cone gauge) against "
            f"{out['lam1_gauge0']:.5f}, {out['lam2_gauge0']:.4f} (all gauge data 0)")
    log(f"wrote {fn}")


if __name__ == "__main__":
    main()

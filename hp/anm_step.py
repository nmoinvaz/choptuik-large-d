#!/usr/bin/env python3
"""The exact Taylor continuation in D (the asymptotic numerical method; notes/QUAD_DOUBLE.md 13.15): from a converged
pinned state z_0 at D_0 the branch's Taylor coefficients z_k in h = D - D_0 to order N, each from one right-hand side
(Solver.residual_jet on the jet state z_0 + ... + z_{k-1} h^{k-1} with the dimension n + h: its order-k coefficient
with z_k absent) solved with the Jacobian's LU at z_0 (one factorisation, one refinement step per order through the
exact product Solver.jvp), the step rule a_max = (eps |z_1| / |z_N|)^{1/(N-1)}, the radius from the coefficients'
ratios, and the prediction z(h) at the given h with the Newton of hpkernel.driver from it. Tests J3 and J4 of 13.15.

    python3 anm_step.py RAWDIR OUT.jsonl --state T75/pin5_D71.0000.npz --N 24 --hs 0.5,1,2 [--threads 4] [--mutate 21]
        [--it_max 8] [--tol 1e-25]

RAWDIR is a clone of the raw-data repository (the state's path is relative to it); OUT.jsonl lies in a task folder of
it (hp/raw_out.py). --mutate 21 is the control (model.rs JET_MUT_FREEZE_N, the dimension frozen: every z_k is then
zero and the prediction is z_0; the branch test must fail). Prints and records |z_k|, the ratios, a_max, and per h the
predicted state's residual, Newton's steps and final residual, the prediction error against the converged state (max
norm) and Delta against the README's formula.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import mpmath as mp  # noqa: E402
from hpkernel.driver import make_solver, newton_solve, rnorm  # noqa: E402
from multidouble import dd_add, dd_scale, dd_norm  # noqa: E402
from raw_out import require_raw  # noqa: E402
from taylor_predict import formula_delta  # noqa: E402

JET_MUT_FREEZE_N = 21


def load_state(path):
    d = np.load(path)
    m = json.loads(str(d["meta"]))
    return d["zh"].astype(float), d["zl"].astype(float), m


def main():
    args = list(sys.argv[1:])
    opt = dict(state="T75/pin5_D71.0000.npz", N=24, hs="0.5,1,2", threads=4, mutate=0, it_max=8, tol=1e-25, eps="1e-16,1e-20")
    for flag in list(opt):
        f = "--" + flag
        if f in args:
            i = args.index(f); v = args[i + 1]; del args[i:i + 2]
            opt[flag] = type(opt[flag])(v) if not isinstance(opt[flag], str) else v
    raw, out = args[0], args[1]
    require_raw(out)
    N, threads, mutate = opt["N"], opt["threads"], opt["mutate"]
    hs = [float(x) for x in opt["hs"].split(",")]
    epss = [float(x) for x in opt["eps"].split(",")]
    zh, zl, m = load_state(os.path.join(raw, opt["state"]))
    D0, M, nodes = float(m["D"]), int(m["M"]), list(m["nodes"])
    kt, kc, ks = m.get("kt", 36), m.get("kc", 146), m.get("ks", 130)
    kf = 13 if D0 >= 35 else None
    mk = lambda Dx: make_solver(Dx, M, nodes=nodes, kt=kt, kc=kc, ks=ks, threads=threads, kfreeze=kf)  # noqa: E731
    s = mk(D0)
    segs = s.plan(zh, zl, 1e-32, 0.05)
    s.set_segs(segs)
    rec = dict(state=opt["state"], D0=D0, M=M, nodes=nodes, N=N, mutate=mutate, hs=hs)
    # the base state converged to tol (a stored state may sit above it), then the LU at it
    Fh, Fl = s.residual(zh, zl)
    F0 = float(rnorm(Fh, Fl))
    if F0 > opt["tol"]:
        zh, zl, hist = newton_solve(s, zh, zl, tol=opt["tol"], it_max=6, verbose=False)
        Fh, Fl = s.residual(zh, zl)
        rec["base_newton_steps"] = len(hist)
    rec["F_base"] = float(rnorm(Fh, Fl))
    t0 = time.time()
    dh, dl, Fh, Fl, pmin, pmax = s.newton(zh, zl)        # stores the double-double LU of the pinned Jacobian at z_0
    rec.update(t_lu=time.time() - t0, piv=(float(pmin), float(pmax)), dz_base=float(dd_norm(dh, dl)))
    print(f"{opt['state']}: D0 {D0}, M {M}, |F| {rec['F_base']:.1e}, LU {rec['t_lu']:.1f} s, pivots {pmin:.1e}..{pmax:.1e}, "
          f"|dz| {rec['dz_base']:.1e}", flush=True)

    # the coefficients, order by order
    Zh, Zl = [zh], [zl]
    norms, rhs_norms, refine, times = [float(dd_norm(zh, zl))], [None], [None], [None]
    for k in range(1, N + 1):
        t0 = time.time()
        Hs, Ls = s.residual_jet(Zh, Zl, k + 1, mutate)
        rh, rl = Hs[k], Ls[k]
        xh, xl = s.lu_solve(-rh, -rl)
        # one refinement step with the exact product: r = -rhs - J x, x += J^-1 r
        jh, jl = s.jvp(zh, zl, xh, xl)[:2]
        r0h, r0l = dd_add(-rh, -rl, -jh, -jl)
        ch, cl = s.lu_solve(r0h, r0l)
        xh, xl = dd_add(xh, xl, ch, cl)
        jh, jl = s.jvp(zh, zl, xh, xl)[:2]
        r1h, r1l = dd_add(-rh, -rl, -jh, -jl)
        Zh.append(xh); Zl.append(xl)
        norms.append(float(dd_norm(xh, xl)))
        rhs_norms.append(float(dd_norm(rh, rl)))
        refine.append((float(dd_norm(r0h, r0l)), float(dd_norm(r1h, r1l))))
        times.append(time.time() - t0)
        print(f"  order {k:2d}: |rhs| {rhs_norms[-1]:.2e}, |z_k| {norms[-1]:.3e}, ratio |z_{k-1}|/|z_k| "
              f"{norms[-2] / norms[-1] if norms[-1] else float('inf'):.3f}, solve residual {refine[-1][0]:.1e} -> {refine[-1][1]:.1e}, "
              f"{times[-1]:.1f} s", flush=True)
    rec.update(norms=norms, rhs_norms=rhs_norms, refine=refine, times=times)
    ratios = [norms[k - 1] / norms[k] if norms[k] else None for k in range(1, N + 1)]
    rec["ratios"] = ratios
    rec["a_max"] = {str(e): (e * norms[1] / norms[N]) ** (1.0 / (N - 1)) if norms[N] else None for e in epss}
    rec["R_D_last"] = ratios[-1]
    print(f"a_max: " + ", ".join(f"eps {e}: {a:.2f}" for e, a in rec["a_max"].items() if a) + f"; last ratio {ratios[-1]}", flush=True)

    # predictions
    mp.mp.dps = 50
    preds = []
    for h in hs:
        ph, pl = np.array(zh), np.array(zl)
        for k in range(1, N + 1):
            ph, pl = dd_add(ph, pl, *dd_scale(Zh[k], Zl[k], h ** k))
        sh = mk(D0 + h)
        sh.set_segs(segs)
        Fh, Fl = sh.residual(ph, pl)
        Fp = float(rnorm(Fh, Fl))
        p = dict(h=h, D=D0 + h, F_pred=Fp, Delta_pred=float(ph[-1] + pl[-1]))
        print(f"h {h}: predicted |F| {Fp:.2e}, Delta {p['Delta_pred']:.12f}", flush=True)
        if np.isfinite(Fp):
            t0 = time.time()
            ch, cl, hist = newton_solve(sh, ph, pl, tol=opt["tol"], it_max=opt["it_max"], verbose=False)
            Fh, Fl = sh.residual(ch, cl)
            Fc = float(rnorm(Fh, Fl))
            fd = float(formula_delta(D0 + h))
            eh, el = dd_add(ch, cl, -ph, -pl)
            p.update(converged=bool(Fc <= opt["tol"]), steps=len(hist), F_final=Fc, Delta=float(ch[-1] + cl[-1]), Delta_formula=fd,
                     dDelta_formula=float(ch[-1] + cl[-1]) - fd, pred_error=float(dd_norm(eh, el)), t_newton=time.time() - t0)
            print(f"      Newton: {len(hist)} steps, |F| {Fp:.1e} -> {Fc:.1e}, {'CONVERGED' if p['converged'] else 'not converged'}; "
                  f"Delta {p['Delta']:.12f} (formula {fd:.6f}, diff {p['dDelta_formula']:+.1e}); prediction error {p['pred_error']:.2e}; "
                  f"{p['t_newton']:.0f} s", flush=True)
        else:
            p.update(converged=False, note="residual not finite at the predicted state")
        preds.append(p)
    rec["predictions"] = preds
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()

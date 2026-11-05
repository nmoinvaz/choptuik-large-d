#!/usr/bin/env python3
"""The exact Taylor continuation in D (the asymptotic numerical method of notes/QUAD_DOUBLE.md 13.15; Cochelin, Comput.
Struct. 53 (1994) 1181) on the low-D branch at M = 320: the branch's Taylor coefficients z_1..z_N at a converged state
from one Jacobian (the mixed step's f64 LU in block storage; each order's solve J z_k = -[F(z_<k(h), D0 + h)]_k refined
by GMRES with exact double-double products, driver.gmres_refine), the predictions at the D of stored converged states
and at --target, and Newton from the target's prediction. Pilot T106, procedure notes/LOW_D.md section 14.

    python3 anm_lowd.py RAWDIR OUTDIR [--state T82-win/states/lowDhm_x0.000157-0.975_D3.0450_M320.npz]
        [--checks T100-win2/states/lowDhm_x0.000157-0.975_D3.0444_M320.npz,T82-win/states/lowDhm_x0.000157-0.975_D3.0475_M320.npz]
        [--target 3.04375] [--N 15] [--threads 8] [--newton_its 6] [--step_tol 1e-20]

RAWDIR is a clone of the raw-data repository; OUTDIR lies in a task folder of it (raw_out.py). Writes
OUTDIR/anm_lowd.jsonl (one record per stage, flushed as it goes) and, when Newton converges at --target, its state.
Residuals are max |F| over the rows, as hp/lowD_dd.py reports them. The control: the first order's right-hand side
-[F]_1 with the dimension frozen in the jets (Ctx::mutate 21, JET_MUT_FREEZE_N), which then has no h term, so it must
vanish against the real one (the D-derivative of F at the base).
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from hpkernel.driver import gmres_refine, load, make_solver, newton_solve, save  # noqa: E402
from multidouble import dd_add, dd_norm, dd_scale  # noqa: E402
from raw_out import FLAG, HELP, require_raw  # noqa: E402,F401

JET_MUT_FREEZE_N = 21


def fmax(s, zh, zl):
    fh, fl = s.residual(zh, zl)
    return float(np.abs(np.asarray(fh) + np.asarray(fl)).max())


def solver_for(meta, D, threads, segs=None):
    s = make_solver(D, meta["M"], nodes=meta["nodes"], kt=meta["kt"], kc=meta["kc"], ks=meta["ks"], threads=threads)
    if segs is not None:
        s.set_segs(segs)
    return s


def coefficients(s, zh, zl, N, rec, mutate=0, step_tol=1e-20, max_products=40):
    """z_1..z_N at (zh, zl) with the LU stored in s; per order the GMRES products, its relative changes and time.
    step_tol must lie above the solve's double-double floor: past it GMRES's basis degenerates and the iterate
    leaves the solution (at 1e-26, D 4, M 48: changes 1, 7e-8, 6e-16, 3e-24, then NaN; hp/jet_lowd_check.py)."""
    Zh, Zl = [zh], [zl]
    for k in range(1, N + 1):
        t0 = time.time()
        Hs, Ls = s.residual_jet(Zh, Zl, k + 1, mutate)
        bh, bl = -np.asarray(Hs[k]), -np.asarray(Ls[k])
        t1 = time.time()
        if not float(dd_norm(bh, bl)) > 0:
            raise SystemExit(f"z_{k}: the right-hand side is zero (a frozen dimension?)")
        xh, xl, info = gmres_refine(s, zh, zl, bh, bl, max_products=max_products, step_tol=step_tol)
        Zh.append(np.asarray(xh)); Zl.append(np.asarray(xl))
        rec.append(dict(k=k, norm=float(dd_norm(xh, xl)), rhs=float(dd_norm(bh, bl)), products=info["products"],
                        converged=bool(info["converged"]), changes=[float(c) for c in info.get("changes", [])],
                        t_jet=round(t1 - t0, 1), t_solve=round(time.time() - t1, 1)))
        print(f"  z_{k}: |z| {rec[-1]['norm']:.3e}, |rhs| {rec[-1]['rhs']:.2e}, {info['products']} products, "
              f"converged {info['converged']}, {t1 - t0:.0f} + {time.time() - t1:.0f} s", flush=True)
    return Zh, Zl


def evaluate(Zh, Zl, h):
    """sum_k z_k h^k in double-double (h exact: a difference of two nearby doubles)"""
    ph, pl = np.array(Zh[0], copy=True), np.array(Zl[0], copy=True)
    for k in range(1, len(Zh)):
        ph, pl = dd_add(ph, pl, *dd_scale(Zh[k], Zl[k], h ** k))
    return np.asarray(ph), np.asarray(pl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--state", default="T82-win/states/lowDhm_x0.000157-0.975_D3.0450_M320.npz")
    ap.add_argument("--checks", default="T100-win2/states/lowDhm_x0.000157-0.975_D3.0444_M320.npz,"
                                        "T82-win/states/lowDhm_x0.000157-0.975_D3.0475_M320.npz")
    ap.add_argument("--target", type=float, default=3.04375)
    ap.add_argument("--N", type=int, default=15)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--newton_its", type=int, default=6)
    ap.add_argument("--tol_step", type=float, default=1e-32)
    ap.add_argument("--hmax", type=float, default=0.05)
    ap.add_argument("--step_tol", type=float, default=1e-20)
    ap.add_argument(FLAG, action="store_true", help=HELP)
    a = ap.parse_args()
    require_raw(a.out)
    os.makedirs(a.out, exist_ok=True)
    jf = open(os.path.join(a.out, "anm_lowd.jsonl"), "a")

    def put(**r):
        jf.write(json.dumps(r) + "\n"); jf.flush()

    zh, zl, meta = load(os.path.join(a.raw, a.state))
    D0 = float(meta["D"])
    t0 = time.time()
    s = solver_for(meta, D0, a.threads, meta.get("segs"))
    if not meta.get("segs"):                                   # a state stored without its plan: plan from it
        s.set_segs(s.plan(zh, zl, a.tol_step, a.hmax))
    F0 = fmax(s, zh, zl)
    _, _, _, _, pmin, pmax = s.newton_mixed(zh, zl)            # the f64 Jacobian and its block LU at the base
    put(stage="base", state=a.state, D=D0, M=meta["M"], L=len(zh), steps=sum(len(g) - 1 for g in s.get_segs()), F=F0,
        pivots=[pmin, pmax], seconds=round(time.time() - t0, 1), threads=a.threads)
    print(f"base {a.state}: D {D0}, M {meta['M']}, |F| {F0:.1e}, pivots {pmin:.1e}..{pmax:.1e}, {time.time() - t0:.0f} s", flush=True)

    # the control: with the dimension frozen in the jets, F(z0, D) has no h term, so the first order's right-hand side
    # must vanish against the real one
    Hc, Lc = s.residual_jet([zh], [zl], 2, JET_MUT_FREEZE_N)
    ctl = float(dd_norm(np.asarray(Hc[1]), np.asarray(Lc[1])))
    orders = []
    Zh, Zl = coefficients(s, zh, zl, a.N, orders, step_tol=a.step_tol)
    ratio = ctl / orders[0]["rhs"]
    norms = [float(dd_norm(zh, zl))] + [o["norm"] for o in orders]
    radii = [norms[k - 1] / norms[k] for k in range(2, len(norms)) if norms[k]]
    put(stage="coefficients", N=a.N, orders=orders, control_rhs1=ctl, control_ratio=ratio, radii=radii)
    print(f"control (dimension frozen): |[F]_1| {ctl:.2e}, {ratio:.1e} of the real one "
          f"({'fires' if ratio < 1e-10 else 'DOES NOT FIRE'}); radii |z_k-1|/|z_k| {', '.join(f'{r:.3g}' for r in radii)}", flush=True)
    if not ratio < 1e-10:
        put(stage="stop", reason="the control did not fire"); sys.exit(1)

    # predictions at the stored converged states: against the state, and its residual on the state's own plan
    for path in [p for p in a.checks.split(",") if p]:
        ch, cl, mc = load(os.path.join(a.raw, path))
        Dc = float(mc["D"]); h = Dc - D0
        ph, pl = evaluate(Zh, Zl, h)
        dh, dl = dd_add(ph, pl, -np.asarray(ch), -np.asarray(cl))
        err = float(np.abs(np.asarray(dh) + np.asarray(dl)).max())
        dD = float((ph[-1] - ch[-1]) + (pl[-1] - cl[-1]))
        t1 = time.time()
        sc = solver_for(mc, Dc, a.threads, mc.get("segs"))
        if not mc.get("segs"):
            sc.set_segs(sc.plan(ch, cl, a.tol_step, a.hmax))
        Fp, Fc = fmax(sc, ph, pl), fmax(sc, ch, cl)
        put(stage="check", state=path, D=Dc, h=h, max_err=err, dDelta=dD, F_pred=Fp, F_state=Fc, seconds=round(time.time() - t1, 1))
        print(f"check {path}: h {h:+.6f}, max |z(h) - state| {err:.2e}, Delta {dD:+.2e}, |F| of the prediction {Fp:.2e} "
              f"on the state's plan (the state's own {Fc:.1e})", flush=True)

    # the target: the prediction on its own plan, then Newton (f64 Jacobian, as the low-D driver)
    h = a.target - D0
    ph, pl = evaluate(Zh, Zl, h)
    t1 = time.time()
    st = solver_for(meta, a.target, a.threads)
    segs = st.plan(ph, pl, a.tol_step, a.hmax)
    st.set_segs(segs)
    Fp = fmax(st, ph, pl)
    put(stage="target", D=a.target, h=h, steps=sum(len(g) - 1 for g in st.get_segs()), F_pred=Fp, seconds=round(time.time() - t1, 1))
    print(f"target D {a.target}: h {h:+.6f}, |F| of the prediction {Fp:.2e} on its own plan "
          f"({sum(len(g) - 1 for g in st.get_segs())} steps)", flush=True)
    nh, nl, hist = newton_solve(st, ph, pl, tol=1e-28, it_max=a.newton_its, verbose=True, jac="f64")
    Fn = fmax(st, nh, nl)
    lams = [x.get("lam") for x in hist if "lam" in x]
    put(stage="newton", D=a.target, F=Fn, calls=len(hist), lams=lams, Delta=float(nh[-1]), Delta_lo=float(nl[-1]),
        seconds=round(time.time() - t1, 1))
    print(f"Newton at D {a.target}: |F| {Fn:.2e} after {len(hist)} calls, line-search factors {lams}, "
          f"Delta {nh[-1]:.15f}", flush=True)
    if Fn < 1e-26:
        save(os.path.join(a.out, f"anm_lowd_D{a.target:.6f}_M{meta['M']}.npz"), nh, nl,
             dict(D=a.target, M=meta["M"], nodes=meta["nodes"], kt=meta["kt"], kc=meta["kc"], ks=meta["ks"], segs=st.get_segs(),
                  source=f"anm_lowd from {a.state}, N {a.N}"))
    put(stage="done", seconds=round(time.time() - t0, 1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""T106's diagnostic of anm_lowd.py's first order at the pilot's base (D 3.045, M 320), each link against a
computation that does not use it (notes/LOW_D.md section 14):
  (a) the jet's order 0 (Solver.residual_jet) against the residual;
  (b) its order 1, dF/dD at fixed z, against central differences of the residual in D on the same plan;
  (c) Solver.jvp, with cut=True (gmres_refine's) and cut=False, against central differences of the residual along a
      random direction;
  (d) the branch's tangent t = dz/dD from four stored converged states (the derivative of their cubic interpolant at
      the base) against J t = -F_1, through |J t + F_1|, with |J (t4 - t3)| (t3 from three of the states) as the
      interpolation's own size;
  (e) the stored f64 LU's solve of J x = -F_1 and gmres_refine's (cut=True as anm_lowd.py, and cut=False) against t.
Norms are 2-norms. Each link prints PASS or FAIL against the bound named in its line; exit 1 when any fails. --mutate is the control:
the jet with the dimension frozen (JET_MUT_FREEZE_N, so F_1 = 0 and (b) fails) and jvp without the horizon block's
state tangent (its mutate 2, so (c), (c_cut) and (d) fail); use it with --no_gmres.

    python jet_lowd_check.py RAWDIR OUT.jsonl [--threads 4] [--eps 1e-6,1e-8,1e-10] [--no_gmres] [--mutate]
        [--base T82-win/states/..._D3.0450_M320.npz] [--others A.npz,B.npz,C.npz]

--base and --others name the states (relative to RAWDIR): the base and three converged states of the same layout on
the branch near it (two below the base for t3, as the defaults).

RAWDIR is a clone of the raw-data repository; OUT.jsonl lies in a task folder of it (raw_out.py).
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402
from hpkernel.driver import gmres_refine, load, make_solver  # noqa: E402
from multidouble import dd_add, dd_scale  # noqa: E402
from raw_out import FLAG, HELP, require_raw  # noqa: E402,F401

BASE = "T82-win/states/lowDhm_x0.000157-0.975_D3.0450_M320.npz"
JET_MUT_FREEZE_N, JVP_MUT_HORIZON = 21, 2
OTHERS = ["T105-win2/states/lowDhm_x0.000157-0.975_p1_D3.0438_M320.npz",
          "T100-win2/states/lowDhm_x0.000157-0.975_D3.0444_M320.npz",
          "T82-win/states/lowDhm_x0.000157-0.975_D3.0475_M320.npz"]


def v(h, l):
    """a double-double vector as f64 (the comparisons here are at relative 1e-6 and coarser)"""
    return np.asarray(h) + np.asarray(l)


def n2(x):
    return float(np.linalg.norm(x))


def deriv_weights(xs, x0):
    """weights w_i with sum w_i f(x_i) = p'(x0), p the interpolant through (x_i, f(x_i))"""
    n = len(xs)
    w = []
    for i in range(n):
        rest = [j for j in range(n) if j != i]
        den = np.prod([xs[i] - xs[j] for j in rest])
        num = sum(np.prod([x0 - xs[k] for k in rest if k != m]) for m in rest)
        w.append(num / den)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--eps", default="1e-6,1e-8,1e-10")
    ap.add_argument("--no_gmres", action="store_true")
    ap.add_argument("--mutate", action="store_true")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--others", default=",".join(OTHERS))
    ap.add_argument(FLAG, action="store_true", help=HELP)
    a = ap.parse_args()
    require_raw(a.out)
    mp.mp.dps = 50                                             # D +- eps exact enough for make_solver's double-double n
    jf = open(a.out, "a")
    fails = []

    def put(**r):
        jf.write(json.dumps(r) + "\n"); jf.flush()

    def verdict(name, ok, line):
        print(f"({name}) {line}: {'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            fails.append(name)

    zh, zl, meta = load(os.path.join(a.raw, a.base))
    D0, M, nodes, segs = float(meta["D"]), meta["M"], meta["nodes"], meta["segs"]
    mk = lambda Dx: make_solver(Dx, M, nodes=nodes, kt=meta["kt"], kc=meta["kc"], ks=meta["ks"], segs=segs,  # noqa: E731
                                threads=a.threads)
    t0 = time.time()
    s0 = mk(mp.mpf(D0))
    F0 = v(*s0.residual(zh, zl))
    t_res = time.time() - t0
    print(f"base {a.base}: D {D0}, M {M}, L {len(zh)}, {sum(len(g) - 1 for g in segs)} steps, |F| {n2(F0):.2e}, "
          f"residual {t_res:.1f} s", flush=True)

    # (a), (b): the jet's orders 0 and 1
    t0 = time.time()
    mv = JVP_MUT_HORIZON if a.mutate else 0
    Hs, Ls = s0.residual_jet([zh], [zl], 2, 0)
    c0, F1 = v(Hs[0], Ls[0]), v(Hs[1], Ls[1])
    t_jet = time.time() - t0
    F1b = v(*(x[1] for x in s0.residual_jet([zh], [zl], 2, JET_MUT_FREEZE_N))) if a.mutate else F1   # (b)'s
    d0 = n2(c0 - F0)
    verdict("a", d0 <= 1e-3 * n2(F0) + 1e-40, f"jet order 0 against the residual: |c0 - F| {d0:.2e} (|F| {n2(F0):.2e}), "
            f"bound 1e-3 |F|")
    fd = []
    for eps in [float(x) for x in a.eps.split(",")]:
        sp, sm = mk(mp.mpf(D0) + mp.mpf(eps)), mk(mp.mpf(D0) - mp.mpf(eps))
        d1 = (v(*sp.residual(zh, zl)) - v(*sm.residual(zh, zl))) / (2 * eps)
        fd.append(dict(eps=eps, rel=n2(d1 - F1b) / max(n2(F1b), 1e-300), delta_row=float(d1[-1])))
        print(f"    dF/dD by central differences, eps {eps:.0e}: |d1 - F_1| / |F_1| {fd[-1]['rel']:.2e}", flush=True)
    best = min(r["rel"] for r in fd)
    verdict("b", best <= 1e-6, f"jet order 1 against central differences in D: best {best:.2e} (|F_1| {n2(F1b):.3e}, "
            f"jet {t_jet:.0f} s), bound 1e-6")
    put(stage="jet", mutate=bool(a.mutate), D=D0, M=M, F0=n2(F0), c0_minus_F=d0, F1=n2(F1), fd=fd, t_residual=t_res, t_jet=t_jet)

    # (c): jvp along a random direction (max |u| = 1)
    u = np.random.default_rng(0).standard_normal(len(zh))
    u /= np.abs(u).max()
    zero = np.zeros_like(u)

    def fd_dir(w, eps):
        ph, pl = dd_add(zh, zl, *dd_scale(w, zero, eps))
        mh, ml = dd_add(zh, zl, *dd_scale(w, zero, -eps))
        return (v(*s0.residual(ph, pl)) - v(*s0.residual(mh, ml))) / (2 * eps)

    fdu = {e: fd_dir(u, e) for e in (1e-8, 1e-9, 1e-10)}
    jv = {}
    for cut in (False, True):
        t0 = time.time()
        Ju = v(*s0.jvp(zh, zl, u, zero, mutate=mv, cut=cut)[:2])
        rel = {e: n2(Ju - fdu[e]) / n2(Ju) for e in fdu}
        jv[cut] = dict(rel=min(rel.values()), seconds=round(time.time() - t0, 1))
        verdict(f"c{'_cut' if cut else ''}", jv[cut]["rel"] <= 1e-6,
                f"jvp(cut={cut}) along a random u against central differences (eps 1e-8, 1e-9, 1e-10): "
                f"{', '.join(f'{r:.1e}' for r in rel.values())}, {jv[cut]['seconds']} s, bound 1e-6")
    put(stage="jvp", rel_nocut=jv[False]["rel"], rel_cut=jv[True]["rel"],
        seconds=[jv[False]["seconds"], jv[True]["seconds"]])

    # (d): the tangent of the stored states
    Ds, Zs = [D0], [v(zh, zl)]
    for p in a.others.split(","):
        h, l, m = load(os.path.join(a.raw, p))
        if m["M"] != M or list(m["nodes"]) != list(nodes) or len(h) != len(zh):
            raise SystemExit(f"{p}: another layout (M {m['M']}, {len(m['nodes'])} nodes), not comparable")
        Ds.append(float(m["D"])); Zs.append(v(h, l))
    order = np.argsort(Ds)
    Ds, Zs = [Ds[i] for i in order], [Zs[i] for i in order]
    t4 = sum(w * z for w, z in zip(deriv_weights(Ds, D0), Zs))
    near = [i for i in range(len(Ds)) if Ds[i] <= D0]           # the three states at or below the base
    t3 = sum(w * Zs[i] for w, i in zip(deriv_weights([Ds[i] for i in near], D0), near))
    Jt = v(*s0.jvp(zh, zl, t4, zero, mutate=mv)[:2])
    Jdt = v(*s0.jvp(zh, zl, t4 - t3, zero)[:2])
    r_t, r_unc = n2(Jt + F1) / n2(F1), n2(Jdt) / n2(F1)
    print(f"    tangent from D {', '.join(f'{d:g}' for d in Ds)}: |t| {n2(t4):.4e}, dDelta/dD {t4[-1]:.6f}, "
          f"|t4 - t3| / |t4| {n2(t4 - t3) / n2(t4):.1e}", flush=True)
    verdict("d", n2(Jt + F1) <= n2(Jdt) + 1e-8 * n2(F1),
            f"|J t + F_1| / |F_1| {r_t:.2e} against the interpolation's |J (t4 - t3)| / |F_1| {r_unc:.2e}")
    put(stage="tangent", Ds=Ds, t=n2(t4), dDelta=float(t4[-1]), t4_t3=n2(t4 - t3) / n2(t4), Jt_plus_F1=r_t, J_dt=r_unc)

    # (e): the linear solves
    if not a.no_gmres:
        t0 = time.time()
        _, _, _, _, pmin, pmax = s0.newton_mixed(zh, zl)
        t_lu = time.time() - t0
        tol_t = max(10 * n2(t4 - t3) / n2(t4), 1e-6)

        def judge(name, x, extra):
            Jx = v(*s0.jvp(zh, zl, x, zero)[:2])
            rec = dict(x=n2(x), dDelta=float(x[-1]), res=n2(Jx + F1) / n2(F1), vs_t=n2(x - t4) / n2(t4), **extra)
            verdict(name, rec["vs_t"] <= tol_t, f"{extra['how']}: |x| {rec['x']:.4e}, dDelta/dD {rec['dDelta']:.6f}, "
                    f"|J x + F_1| / |F_1| {rec['res']:.1e}, |x - t| / |t| {rec['vs_t']:.1e}, bound {tol_t:.0e}")
            put(stage=name, **rec)

        xh, xl = s0.lu_solve(-np.asarray(Hs[1]), -np.asarray(Ls[1]))
        judge("e_lu", v(xh, xl), dict(how=f"f64 LU solve (pivots {pmin:.1e}..{pmax:.1e}, {t_lu:.0f} s)"))
        # anm_lowd.py's call (cut=True, step_tol 1e-26), then step_tol 1e-20 with and without the cut
        for cut, tol in ((True, 1e-26), (True, 1e-20), (False, 1e-20)):
            t0 = time.time()
            gh, gl, info = gmres_refine(s0, zh, zl, -np.asarray(Hs[1]), -np.asarray(Ls[1]), max_products=40,
                                        step_tol=tol, cut=cut)
            ch = [float(f"{c:.1e}") for c in info["changes"]]
            judge(f"e_gmres{'_cut' if cut else ''}_{tol:.0e}", v(gh, gl),
                  dict(how=f"gmres_refine cut={cut} step_tol {tol:.0e} ({info['products']} products, converged "
                           f"{info['converged']}, {time.time() - t0:.0f} s, changes {ch[:6]}..{ch[-2:]})",
                       products=info["products"], converged=bool(info["converged"]), changes=ch, cut=cut, step_tol=tol))
    put(stage="done", fails=fails)
    print(f"failed links: {', '.join(fails) if fails else 'none'}", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

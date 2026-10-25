#!/usr/bin/env python3
"""Can a corrector get past the curvature that stalls Newton at large D? (qdh_check.py check C; qdh_diag.py found
F(z + lam d) = (1 - lam) F0 + lam^2 Q exactly along the Newton step d, |Q| about 5 |F0| at D 67.72, the step in the
near-null centre directions.) Three correctors, each with the quad-double-horizon residual (driver.QdhSolver) and
the double-double Jacobian's LU, from the same state, with the same budget:

    cd hp && python3 qdh_corrector_test.py [STATE.npz] [--its=12] [--threads=N] [--out=data/large_d/qdh_corrector_test.jsonl]

- newton (the control): v = -J^-1 F, backtracking on |F|_2 along z + t v (t = 1, 1/2, ..., 12 halvings), as
  driver.newton_solve.
- geodesic: v as above and the geodesic acceleration a = -J^-1 r, r the second directional derivative of F along v
  from F at z + h v (h = 1/4): r = 2 (F(z + h v) - (1 - h) F(z)) / h^2 (exact when J v = -F and F is quadratic along v,
  as measured); the step is the parabola t v + t^2 a / 2, backtracked like newton's (Transtrum, Machta and Sethna, PRL
  104, 060201 (2010), whose step is v + a/2; this is its Newton (undamped Levenberg-Marquardt) form, i.e. a damped
  Chebyshev step).
- anderson: Newton-Anderson of depth 1 (Dallas and Pollock, arXiv:2207.12334): with w_k the Newton step and s_{k-1}
  the previous step taken, gamma = w_k.(w_k - w_{k-1}) / |w_k - w_{k-1}|^2 and the step w_k - gamma (s_{k-1} + w_k -
  w_{k-1}); gamma is set to 0 (a Newton step) when |gamma| > 1; backtracked like newton's.
Every state is made in quad-double (QdhSolver.zadd).

Pass (written before the run), per method: |F| (max norm) <= 1e-25 within --its iterations, with Delta moved by at most
1e-14. The control must stay above 1e-23 (otherwise this state does not test anything). Exit 0 if the control stalls
and at least one method passes, else 1.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hpkernel.driver import QdhSolver, dd_add, dd_axpy, dd_scale, load, make_solver, rnorm  # noqa: E402


def fmax(F):
    return float(np.abs(np.asarray(F[0]) + np.asarray(F[1])).max())


def combine(c1, x, c2, y):
    """c1 x + c2 y for double-double vectors x, y and f64 c1, c2 (powers of two where exactness matters)"""
    a = dd_scale(np.asarray(x[0]), np.asarray(x[1]), c1)
    return dd_axpy(c2, np.asarray(y[0]), np.asarray(y[1]), a[0], a[1])


def run(s, zh, zl, method, its, log):
    qs = QdhSolver(s)
    hist = []
    prev_w = prev_s = None
    D0 = float(zh[-1] + zl[-1])
    for it in range(its):
        dh, dl, Fh, Fl, pmin, pmax = qs.newton(zh, zl)
        n0, f0 = rnorm(Fh, Fl), fmax((Fh, Fl))
        w = (np.asarray(dh), np.asarray(dl))
        rec = dict(it=it, F=f0)
        if f0 <= 1e-25:
            hist.append(rec)
            break
        accel = None
        if method == "geodesic":
            h = 0.25
            ph, pl = qs.zadd(zh, zl, h * w[0], h * w[1])
            Fp = qs.residual(ph, pl)
            rh, rl = combine(1.0, Fp, -(1 - h), (Fh, Fl))
            rh, rl = dd_scale(rh, rl, 2 / h ** 2)
            ah, al = s.lu_solve(np.ascontiguousarray(-rh), np.ascontiguousarray(-rl))
            accel = (np.asarray(ah), np.asarray(al))
            rec["a_over_v"] = float(np.abs(accel[0] + accel[1]).max() / np.abs(w[0] + w[1]).max())
        base = w
        if method == "anderson" and prev_w is not None:
            dw = (w[0] + w[1]) - (prev_w[0] + prev_w[1])
            den = float(dw @ dw)
            g = float((w[0] + w[1]) @ dw) / den if den > 0 else 0.0
            if abs(g) > 1:
                g = 0.0
            rec["gamma"] = g
            # w - g (s_prev + w - w_prev)
            t1 = dd_add(prev_s[0], prev_s[1], w[0], w[1])
            t1 = dd_add(t1[0], t1[1], -prev_w[0], -prev_w[1])
            base = dd_axpy(-g, np.asarray(t1[0]), np.asarray(t1[1]), w[0], w[1])
        t, accepted = 1.0, False
        for _ in range(13):
            step = dd_scale(np.asarray(base[0]), np.asarray(base[1]), t)
            if accel is not None:
                step = dd_axpy(0.5 * t * t, accel[0], accel[1], np.asarray(step[0]), np.asarray(step[1]))
            th, tl = qs.zadd(zh, zl, np.asarray(step[0]), np.asarray(step[1]))
            Ft = qs.residual(th, tl)
            if rnorm(*Ft) <= (1 - 1e-4 * t) * n0:
                accepted = True
                break
            t *= 0.5
        rec.update(t=t, accepted=accepted, step=float(np.abs(np.asarray(step[0]) + np.asarray(step[1])).max()),
                   F_after=fmax(Ft) if accepted else None)
        hist.append(rec)
        log(f"    {method} {it}: |F| {f0:.3e} -> {fmax(Ft) if accepted else float('nan'):.3e}, t {t:g}, |step| "
            f"{rec['step']:.2e}" + (f", |a|/|v| {rec['a_over_v']:.2f}" if 'a_over_v' in rec else "")
            + (f", gamma {rec['gamma']:+.3f}" if 'gamma' in rec else ""))
        if not accepted:
            break
        prev_w, prev_s = w, (np.asarray(step[0]), np.asarray(step[1]))
        zh, zl = th, tl
    Ff = qs.residual(zh, zl)
    z2, _ = qs.parts(zh, zl)
    dD = float((zh[-1] - D0) + zl[-1] + z2[-1])
    return dict(method=method, F=fmax(Ff), its=len(hist), Delta_moved=dD, hist=hist)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    path = args[0] if args else os.path.join(HERE, "data", "large_d", "up70_D67.7224.npz")
    out = opts.get("out", os.path.join(HERE, "data", "large_d", "qdh_corrector_test.jsonl"))
    its = int(opts.get("its", 12))
    zh, zl, meta = load(path)
    zh, zl = np.ascontiguousarray(zh), np.ascontiguousarray(zl)
    s = make_solver(meta["D"], meta["M"], nodes=meta["nodes"], kt=meta["kt"], kc=meta["kc"], ks=meta["ks"],
                    threads=int(opts.get("threads", 2)), kfreeze=13)
    s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
    log = lambda m: print(m, flush=True)  # noqa: E731
    log(f"D {meta['D']}: M {meta['M']}, ks {meta['ks']}, kc {meta['kc']}; {its} iterations each")
    res = {}
    t0 = time.time()
    for method in ("newton", "geodesic", "anderson"):
        r = run(s, zh, zl, method, its, log)
        res[method] = r
        log(f"  {method}: |F| {r['F']:.3e} after {r['its']} iterations, Delta moved {r['Delta_moved']:+.2e}")
    ok = {m: res[m]["F"] <= 1e-25 and abs(res[m]["Delta_moved"]) <= 1e-14 for m in ("geodesic", "anderson")}
    stalled = res["newton"]["F"] > 1e-23
    log(f"control stalls: {stalled}; geodesic passes: {ok['geodesic']}; anderson passes: {ok['anderson']} "
        f"({time.time() - t0:.0f} s)")
    with open(out, "a") as fo:
        fo.write(json.dumps(dict(state=os.path.basename(path), D=meta["D"], its=its, results=res, control_stalls=stalled,
                                 passes=ok, time=time.strftime("%Y-%m-%d %H:%M:%S"))) + "\n")
    good = stalled and any(ok.values())
    log("PASS" if good else "FAIL" + ("" if stalled else " (the control did not stall: the state tests nothing)"))
    sys.exit(0 if good else 1)


if __name__ == "__main__":
    main()

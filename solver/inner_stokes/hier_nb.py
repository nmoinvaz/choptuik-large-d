"""hier.py with the compiled backend: the same hierarchical solve (hier.main, unchanged), with inner_np.Inner replaced by
inner_nb.InnerNB (numba series and DOP853 transcribed from scipy; same right-hand side, same step-size control, the
same number of right-hand-side calls; residuals agree to rounding, `inner_nb.py check`). In spawned workers the
replacement is made again at import (this file is the workers' __mp_main__).

One extra, optional step before hier.main: --pretune solves the top harmonic's condition S_M = 0 alone, at fixed lower
unknowns, first at its approach radius and then at its polish radius, integrating only to that radius. Why: hier
evaluates every residual at all radii from one integration, and while the top harmonic is far from tuned
(|r_M| ~ 10^2 at M = 7), its jump at xi = 0.2 is large and nonlinear and the arc there costs ~10^6 right-hand sides
(minutes) instead of ~10^4. The pretuned x is only a better starting point; hier's own level-M solve then starts in
its polish regime, and the zero it converges to does not depend on it.

    python -u hier_nb.py M A wf [hier.py options] [--pretune]
e.g. the M = 7 selection from the M = 5 zero:
    python -u hier_nb.py 7 2.718579189229888 0.9992284031945021 --x=0.1492035749173264,0.01471905249087869,\
0.008405346906375517,0.0014703451817031647 --pretune --schur --predict --floor --procs 2 --verbose=2
"""
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import inner_nb                                         # noqa: E402

inner_nb.install()
import hier as H                                        # noqa: E402
import inner_np as I                                    # noqa: E402


def pretune(x, M, xi_top=0.2, maxit=12):
    """Newton for (B_M, C_M) on r_M at the approach radius, then at the polish radius, integrating only to that
    radius; lower unknowns fixed. Returns x."""
    w = I.W0 * x[1]
    F2 = I.F2mean(I.Fh_from(x, M))
    rlow = H.level_radius(M, w, xi_top)
    rpol = H.polish_radius(M, w, F2, xi_top)
    k = H.idx(M)
    for rr, tag in ((rlow, "approach"), (rpol, "polish")):
        def g(xx):
            return H.demixed_multi(xx, M, [rr])[rr][k]
        t0 = time.time()
        r = g(x)
        print(f"pretune level {M} ({tag}, xi {rr:.4f}): start |r_{M}| {np.linalg.norm(r):.3e}  [{time.time() - t0:.0f} s]", flush=True)
        Jc = None
        for it in range(maxit):
            if Jc is None:
                hs = 1e-6 * np.maximum(np.abs(x[k]), 1e-3 * abs(x[0]))
                cols = []
                for i, kk in enumerate(k):
                    xt = x.copy()
                    xt[kk] += hs[i]
                    cols.append((g(xt) - r) / hs[i])
                Jc = np.array(cols).T
            step = -np.linalg.solve(Jc, r)
            ok = False
            for lam in (1.0, 0.5, 0.25, 0.125):
                xt = x.copy()
                xt[k] += lam * step
                rt = g(xt)
                if np.all(np.isfinite(rt)) and np.linalg.norm(rt) < (1 - 1e-4 * lam) * np.linalg.norm(r):
                    ok = True
                    break
            if not ok:
                print(f"   it {it}: no decrease; stop at |r_{M}| {np.linalg.norm(r):.3e}", flush=True)
                break
            slow = np.linalg.norm(rt) > 0.3 * np.linalg.norm(r)
            x, r = xt, rt
            print(f"   it {it}: |r_{M}| {np.linalg.norm(r):.3e}  lam {lam}  B,C_{M} {x[k[0]]:.12e} {x[k[1]]:.12e}  "
                  f"[{time.time() - t0:.0f} s]", flush=True)
            if tag == "approach" and np.linalg.norm(r) < 1e-6:
                break
            if slow:
                Jc = None
    return x


def main():
    argv = sys.argv[1:]
    if "--pretune" in argv:
        argv = [a for a in argv if a != "--pretune"]
        args = [a for a in argv if not a.startswith("--")]
        if "--procs" in argv:
            args = [a for a in args if a != argv[argv.index("--procs") + 1]]
        M, A, wf = int(args[0]), float(args[1]), float(args[2])
        opts = {a.split("=")[0]: a.split("=", 1)[1] for a in argv if a.startswith("--") and "=" in a}
        r1 = float(opts.get("--xi1", 0.2))
        if "--x" in opts:
            v = np.array([float(t) for t in opts["--x"].split(",")])
            if len(v) < M - 1:
                v = np.concatenate([v, H.seed_from_C(opts.get("--seed", H.DEFAULT_SEED), A, M)[len(v):]])
        else:
            v = H.seed_from_C(opts.get("--seed", H.DEFAULT_SEED), A, M)
        x = np.concatenate([[A, wf], v])
        x = pretune(x, M, r1)
        xs = ",".join(repr(float(t)) for t in x[2:])
        print(f"pretuned x {x.tolist()}", flush=True)
        argv = [a for a in argv if not a.startswith("--x=")] + [f"--x={xs}"]
    sys.argv = [sys.argv[0]] + argv
    H.main()


if __name__ == "__main__":
    main()

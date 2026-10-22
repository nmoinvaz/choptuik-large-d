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

A second optimisation, LazyHier (below): while a level is in its approach mode only the radii whose entries hier can
read are integrated; the used entries are bit-identical to hier's (`hier_nb.py selftest`).

    python -u hier_nb.py M A wf [hier.py options] [--pretune]
    python3 hier_nb.py selftest [--mutate=radius]       lazy against full residual (the mutation must fail)
Options of this wrapper (removed before hier.main sees the arguments): --pretune; --h1=ABS, --h3=REL (finite-difference
steps of levels 1 and 3, default hier's 1e-6); the environment variable INNER_NB_RTOL (integrator tolerance, inner_nb.py).
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

LAZY = [os.environ.get("HIER_NB_LAZY", "1") == "1"]
FD_STEP = {}                                            # --h1=ABS, --h3=REL (see LazyHier.h); also read in workers via env
for _k, _env in ((1, "HIER_NB_H1"), (3, "HIER_NB_H3")):
    if os.environ.get(_env):
        FD_STEP[_k] = float(os.environ[_env])
MUT_RADIUS = [False]                                    # selftest --mutate=radius: read the approach level at the smallest radius


class LazyHier(H.Hier):
    """hier.Hier whose residual integrates only up to the largest radius whose entries can be used.

    While some level l >= 5 is in its approach ('low') mode, hier uses only r_l at its approach radius and r_m (m > l)
    at their polish radii: the entries of the levels below the highest 'low' level are never read (the level-l Newton
    reads r[idx(l)]; a level returns to the caller only after a fresh evaluation in polish mode; correct() skips rows
    below a 'low' level). So the integration stops at R = max(approach radius of that level, polish radii above it), and
    the unused entries are set to 0. The radii below R are hier's own list (same staged ray segments), so every used
    entry is bit-identical to hier's. Why: with the top harmonic far from tuned, the arc at xi = 0.2 costs ~10^6
    right-hand sides. When no level is in approach mode the evaluation is hier's, unchanged. HIER_NB_LAZY=0 disables."""

    def h(self, l, x):
        """finite-difference steps: hier's, unless --h1=ABS (level 1, absolute steps in A and w/w_inf) or --h3=REL (level
        3, relative steps as hier's 1e-6) are given. Why: at M = 7 the corrected r_3 and r_1 have noise floors far above
        M = 5's, and hier's steps (|dr_3| ~ 1e-2 for C_3) can fall below them."""
        if l == 1 and FD_STEP.get(1):
            return np.array([FD_STEP[1], FD_STEP[1]])
        if l == 3 and FD_STEP.get(3):
            b = x[H.idx(l)]
            return FD_STEP[3] * np.maximum(np.abs(b), 1e-3 * np.abs(x[0]))
        return super().h(l, x)

    def res_raw(self, x):
        lows = [nu for nu in range(5, self.M + 1, 2) if self.mode.get(nu, "low") == "low"]
        if not LAZY[0] or not lows:
            return super().res_raw(x)
        H.NEV[0] += 1
        Ls = max(lows)
        low, pol = self.radii(x)
        allr = [self.r1] + list(low.values()) + list(pol.values())
        need = max([low[Ls]] + [pol[nu] for nu in range(Ls + 2, self.M + 1, 2)])
        R = H.demixed_multi(x, self.M, [r for r in allr if r <= need])
        out = np.zeros(self.M + 1)
        out[H.idx(Ls)] = R[min(R) if MUT_RADIUS[0] else low[Ls]][H.idx(Ls)]
        for nu in range(Ls + 2, self.M + 1, 2):
            out[H.idx(nu)] = R[pol[nu]][H.idx(nu)]
        return out


H.Hier = LazyHier


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


def selftest(mutate=False):
    """LazyHier.res_raw against hier.Hier.res_raw at the pretuned M = 7 start (M = 5 zero + tuned F_7), for every
    combination of approach/polish modes of levels 5 and 7: the used entries must be bit-identical. Exit 1 otherwise.
    --mutate=radius reads the approach level at the smallest integrated radius instead and must fail."""
    MUT_RADIUS[0] = mutate
    x = np.array([2.718579189229888, 0.9992284031945021, 0.1492035749173264, 0.01471905249087869,
                  0.008405346906375517, 0.0014703451817031647, 0.0004739782315836618, 0.00011921162548210427])
    ok = True
    for m5 in ("low", "top"):
        for m7 in ("low", "top"):
            h = LazyHier(7, verbose=0)
            h.mode = {5: m5, 7: m7}
            a = h.res_raw(x)
            b = H.Hier.__mro__[1].res_raw(h, x)
            lows = [nu for nu in (5, 7) if h.mode[nu] == "low"]
            used = list(range(len(a))) if not lows else sum((H.idx(nu) for nu in range(max(lows), 8, 2)), [])
            d = float(np.abs(a[used] - b[used]).max())
            print(f"modes 5:{m5} 7:{m7}: used entries {used}, max |lazy - full| {d:.1e}", flush=True)
            ok = ok and d == 0.0
    print("PASS" if ok else "FAIL")
    return ok


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "selftest":
        sys.exit(0 if selftest("--mutate=radius" in argv) else 1)
    for k in (1, 3):                                    # --h1/--h3 -> environment, so spawned workers see them too
        a = [t for t in argv if t.startswith(f"--h{k}=")]
        if a:
            os.environ[f"HIER_NB_H{k}"] = a[0].split("=", 1)[1]
            FD_STEP[k] = float(os.environ[f"HIER_NB_H{k}"])
    argv = [t for t in argv if not (t.startswith("--h1=") or t.startswith("--h3="))]
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

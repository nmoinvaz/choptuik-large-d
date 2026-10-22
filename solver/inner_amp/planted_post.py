#!/usr/bin/env python3
"""Post-freeze additions to the planted-truth test (notes/PLANTED_SELECTION.md, section 3.7; NOT part of the frozen
battery, added after the frozen completions C-W1..3 put their zeros outside the bracket).

C-ARC*: completions of the real sigma_0..sigma_12 with a smooth arc-density tail instead of a ratio-law tail. The
completed function is, at each node,
    S(x) = P(x) + sum_{k<=12} (sigma_k^real - sigma_k^P) x^k,
with P an arc plant of planted_plants (fitted to the real coefficients with its zero placed at x0(wf; wt_fit)); so its
first 13 coefficients are the real ones and its tail is P's. The truth is the zero of this closed form near 7.4 at each
node and the root of the quadratic through Im x_R(wf), as for the frozen completions.

    nice -n 10 python3 planted_post.py [--scratch DIR]
"""
import argparse
import os
import sys

import mpmath as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import planted_plants as PP  # noqa: E402
import planted_run as R  # noqa: E402


def arc_completion(kind, wt_fit, M=6):
    mp.mp.dps = PP.DPS
    real = PP.load_real()
    nodes, info, xr = {}, {}, {}
    for wf in PP.NODES:
        arc = PP.Arc(wf, "arc" if kind == "beta2" else kind)
        ex = mp.mpc(1) if kind == "beta2" else PP.mpc(PP.BETA[wf]) - 1
        lap = PP.Laplace(arc, ex, M)
        x0 = PP.x0_planted(wf, wt_fit)
        res = lap.fit(real[wf][0], PP.factor([x0]))
        pc = lap.coeffs(PP.factor([x0]), normalise=False)
        norm = pc[0]
        pc = [c / norm for c in pc]
        sig = [real[wf][0][k] if k <= 12 else pc[k] for k in range(PP.KMAX + 1)]
        corr = [real[wf][0][k] - pc[k] for k in range(13)]
        L0 = lap.L(0)

        def f(x, lap=lap, x0=x0, corr=corr, L0=L0):
            return lap.L(x) * (1 - x / x0) / L0 + mp.polyval(corr[::-1], x)
        z = mp.findroot(f, mp.mpc("7.40", "0"), tol=mp.mpf(10) ** -40, maxsteps=200)
        xr[wf] = z
        nodes[wf] = sig
        info[str(wf)] = {"fit_resid_max": max(res), "x_R": [float(z.real), float(z.imag)],
                         "corr_at_x0_max": float(max(abs(c) * 7.4 ** k for k, c in enumerate(corr)))}
    truth = {}
    for lab, fn in (("im_x", lambda z: mp.im(z)), ("im_sqrt", lambda z: mp.im(mp.sqrt(z)))):
        c = np.polyfit(PP.NODES, [float(fn(xr[w])) for w in PP.NODES], 2)
        rts = [r.real for r in np.roots(c) if abs(r.imag) < 1e-12 and 0.95 < r.real < 1.05]
        truth[lab] = min(rts, key=lambda r: abs(r - 1)) if rts else None
    return nodes, info, truth


def refit_scan(kind, wts, M=6):
    """weighted L2 and max relative residual of the arc fit to the real sigma_0..12, against the planted wt
    (the moments do not depend on wt, so one Laplace object per node is refitted)"""
    mp.mp.dps = PP.DPS
    real = PP.load_real()
    laps = {}
    for wf in PP.NODES:
        arc = PP.Arc(wf, "arc" if kind == "beta2" else kind)
        ex = mp.mpc(1) if kind == "beta2" else PP.mpc(PP.BETA[wf]) - 1
        laps[wf] = PP.Laplace(arc, ex, M)
    out = {}
    for wt in wts:
        l2, mx = 0.0, 0.0
        for wf in PP.NODES:
            r = laps[wf].fit(real[wf][0], PP.factor([PP.x0_planted(wf, wt)]))
            l2 += sum(x * x for x in r)
            mx = max(mx, max(r))
        out[wt] = (float(np.sqrt(l2)), mx)
    return out


def best_wt(scan):
    """minimum of the L2 residual: the vertex of a parabola through the smallest point and its neighbours"""
    w = sorted(scan, key=float)
    v = [scan[x][0] for x in w]
    i = int(np.argmin(v))
    i = min(max(i, 1), len(w) - 2)
    xs = [float(w[j]) for j in (i - 1, i, i + 1)]
    c = np.polyfit(xs, [v[j] ** 2 for j in (i - 1, i, i + 1)], 2)
    return -c[1] / (2 * c[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", default=os.path.join(os.environ.get("TMPDIR", "/tmp"), "planted_scratch"))
    a = ap.parse_args()
    R.DONE.update(R.done_names())
    chk = R.load_checks()
    # the wt each arc shape prefers (residual minimum), and the completion with that shape's best-fit tail
    wts = [f"{0.9990 + 0.0001 * i:.4f}" for i in range(21)]
    for kind in ("arc", "chord", "mirror", "beta2"):
        key = f"REFIT-SCAN {kind}"
        if key not in chk:
            scan = refit_scan(kind, wts)
            chk = R.load_checks()
            chk[key] = {"scan": scan, "best_wt": best_wt(scan)}
            R.save_checks(chk)
        bw = f"{chk[key]['best_wt']:.5f}"
        name = f"C-ARC {kind} fit-wt=best({bw})"
        if name not in R.DONE:
            nodes, info, truth = arc_completion(kind, bw)
            chk = R.load_checks()
            chk[name] = {"info": info, "truth": truth}
            R.save_checks(chk)
            R.run_plant(name, "C-ARC (post-freeze)", nodes, truth["im_x"], a.scratch,
                        info={"truth_im_sqrt": truth["im_sqrt"], **info})
    chk = R.load_checks()
    for kind, wt in (("arc", "1.0000"), ("chord", "1.0000"), ("mirror", "1.0000"), ("beta2", "1.0000"),
                     ("arc", "0.9980"), ("arc", "1.0020")):
        name = f"C-ARC {kind} fit-wt={wt}"
        if name in R.DONE:
            continue
        nodes, info, truth = arc_completion(kind, wt)
        chk = R.load_checks()
        chk[name] = {"info": info, "truth": truth}
        R.save_checks(chk)
        R.run_plant(name, "C-ARC (post-freeze)", nodes, truth["im_x"], a.scratch,
                    info={"truth_im_sqrt": truth["im_sqrt"], **info})


if __name__ == "__main__":
    main()

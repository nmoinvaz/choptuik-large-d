#!/usr/bin/env python3
"""The kappa-model's change of stability in D: the CSS's oscillatory mode followed in kappa at fixed D (issue #377).

    python3 css_kappa_crit.py path D LAMOSC LAMPHYS [--k1=0.5] [--h=0.01] [--hmax=0.05] [--out=F]
          background (css_lateral.newton) and the growth rates lambda_osc, lambda_phys (css_pert.detM, secant) followed
          from kappa = 0 (data/css_bg_k0.jsonl) in steps of kappa, at the circle radii R = 0.85 x_h and 0.6 x_h; stops
          one step after Re lambda_osc < 0 at both radii, or at --k1
    python3 css_kappa_crit.py crit PATHFILE D [--out=F]
          kappa_c, where Re lambda_osc = 0, refined at each radius separately from the path's sign change (secant in
          kappa on Re lambda_osc, each evaluation a new background and a new root), and lambda_phys there
    python3 css_kappa_crit.py bgpath D K1 [--h=0.05] [--out=F]
          the background alone continued to kappa = K1 (control: D = 4, K1 = 1 must end on HHS96's axion-dilaton
          solution, omega = 1.17695272, |f(0)| = 0.89255541)
    python3 css_kappa_crit.py selftest
          the crossing finder on a planted Re lambda(kappa) with a known zero (exit 1 if it misses, or if the control,
          a planted function with no sign change at one radius, is accepted)

A step is accepted when the background converges (|J| < 1e-10), each root converges at both radii to the same lambda
(to --spread, default 1e-7) and lies within 0.02 (1 + |lambda|) of its extrapolation; otherwise the step is halved.
Growth rates are zeros of det M at every radius; the jump method's artefacts move with R (matter/CSS.md section 6), so
the crossing is accepted only if Re lambda_osc changes sign at both radii.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import css_lateral as cl  # noqa: E402
import css_pert as cp  # noqa: E402

RADII = (0.85, 0.6)


def secant(F, lam0, tol=1e-11, maxit=40):
    l0 = complex(lam0)
    l1 = l0 * (1 + 1e-4) + 1e-5
    d0, d1 = F(l0), F(l1)
    for _ in range(maxit):
        l2 = l1 - d1 * (l1 - l0) / (d1 - d0)
        if abs(l2 - l1) < tol * max(1, abs(l2)):
            return l2, True
        l0, d0 = l1, d1
        l1, d1 = l2, F(l2)
    return l2, False


def solve_bg(D, kap, om_g, f_g):
    n = D - 2
    xg = cl.horizon_estimate(n, kap, om_g, f_g)
    p, aJ, _ = cl.newton(n, kap, om_g, f_g, 0.85 * xg, verbose=False)
    xh = cl.horizon_estimate(n, kap, p[0], p[1])
    return p[0], p[1], xh, aJ


def roots(D, kap, om, f0, xh, guesses, radii=RADII):
    """Secant roots of det M from each guess at each radius: list over guesses of [(Rf, lam, ok), ...]."""
    n = D - 2
    out = []
    for g in guesses:
        res = []
        for Rf in radii:
            lam, ok = secant(lambda l: cp.detM(n, kap, om, f0, l, xh, Rf * xh), g)
            res.append((Rf, lam, ok))
        out.append(res)
    return out


def extrap(hist, x):
    """Quadratic (or lower) extrapolation of the last points (x_i, v_i) to x."""
    if len(hist) >= 3:
        xs = np.array([q[0] for q in hist[-3:]])
        V = np.array([q[1] for q in hist[-3:]])
        return sum(V[i] * np.prod([(x - xs[j]) / (xs[i] - xs[j]) for j in range(3) if j != i]) for i in range(3))
    if len(hist) == 2:
        (xa, va), (xb, vb) = hist
        return vb + (vb - va) * (x - xb) / (xb - xa)
    return hist[-1][1]


def load_k0(D):
    for l in open(os.path.join(HERE, "data", "css_bg_k0.jsonl")):
        r = json.loads(l)
        if abs(r["D"] - D) < 1e-9:
            return r
    raise SystemExit("no kappa = 0 background at D = %g in data/css_bg_k0.jsonl" % D)


def opts_of(argv):
    pos, o = [], {}
    for a in argv:
        if a.startswith("--"):
            k, _, v = a[2:].partition("=")
            o[k] = v
        else:
            pos.append(a)
    return pos, o


def crossing(ks, res, ):
    """Index i with res[i] > 0 >= res[i+1] (first sign change from + to -), or None."""
    for i in range(len(ks) - 1):
        if res[i] > 0 and res[i + 1] <= 0:
            return i
    return None


def refine(F, ka, kb, fa, fb, tol=1e-10, maxit=40):
    """Illinois regula falsi for F(k) = 0 on [ka, kb] with fa > 0 >= fb; returns (k, F(k), iterations)."""
    side = 0
    kprev = None
    for it in range(maxit):
        k = (ka * fb - kb * fa) / (fb - fa)
        if kprev is not None and abs(k - kprev) < tol:
            return kprev, fprev, it         # the last evaluated point (F(k) is cached by the caller there)
        kprev = k
        fk = F(k)
        fprev = fk
        if fk > 0:
            ka, fa = k, fk
            if side == 1:
                fb *= 0.5
            side = 1
        else:
            kb, fb = k, fk
            if side == -1:
                fa *= 0.5
            side = -1
        if abs(kb - ka) < tol:
            break
    return k, fk, it + 1


def main():
    pos, o = opts_of(sys.argv[1:])
    out = o.get("out")

    def emit(rec):
        print(json.dumps(rec), flush=True)
        if out:
            with open(out, "a") as fo:
                fo.write(json.dumps(rec) + "\n")

    cmd = pos[0] if pos else ""
    if cmd == "selftest":
        # planted: Re lambda(kappa; R) = 0.25 - 3.3 kappa + 2 kappa^2 (+ tiny R dependence), zero at k* below
        kstar = (3.3 - np.sqrt(3.3 ** 2 - 4 * 2 * 0.25)) / 4
        F = lambda k: 0.25 - 3.3 * k + 2 * k * k
        ks = list(np.arange(0, 0.2, 0.03))
        i = crossing(ks, [F(k) for k in ks])
        k, fk, it = refine(F, ks[i], ks[i + 1], F(ks[i]), F(ks[i + 1]))
        ok = abs(k - kstar) < 1e-9
        G = lambda k: 0.25 + 0.1 * k                  # control: never crosses
        ctl = crossing(ks, [G(k) for k in ks]) is None
        print("selftest: planted kappa_c %.12f, found %.12f (%d iterations): %s; control (no crossing) %s"
              % (kstar, k, it, "PASS" if ok else "FAIL", "fires" if ctl else "DOES NOT FIRE"))
        sys.exit(0 if ok and ctl else 1)

    if cmd == "bgpath":
        D, k1 = float(pos[1]), float(pos[2])
        h = float(o.get("h", "0.05"))
        r = load_k0(D)
        hist_om, hist_f = [(0.0, r["omega"])], [(0.0, r["f0"])]
        kap = 0.0
        while kap < k1 - 1e-12:
            kn = min(kap + h, k1)
            om, f0, xh, aJ = solve_bg(D, kn, extrap(hist_om, kn), extrap(hist_f, kn))
            if aJ > 1e-10 or abs(om - extrap(hist_om, kn)) > 0.05 * om:
                h *= 0.5
                if h < 1e-4:
                    raise SystemExit("background continuation failed at kappa %g" % kn)
                continue
            kap = kn
            hist_om.append((kap, om)); hist_f.append((kap, f0))
            emit(dict(kind="bgpath", D=D, kappa=kap, omega=om, f0=f0, xh=xh, absJ=aJ))
        return

    if cmd == "path":
        D = float(pos[1])
        lo0 = complex(pos[2].replace(" ", ""))
        lp0 = complex(pos[3].replace(" ", ""))
        k1 = float(o.get("k1", "0.5"))
        h = float(o.get("h", "0.01"))
        hmax = float(o.get("hmax", "0.05"))
        smax = float(o.get("spread", "1e-7"))
        r = load_k0(D)
        hist = dict(om=[(0.0, r["omega"])], f0=[(0.0, r["f0"])], osc=[], phys=[])
        kap, below = 0.0, 0
        # kappa = 0: the roots from the given starts
        first = True
        while kap < k1 + 1e-12:
            kn = 0.0 if first else min(kap + h, k1 + 1e-9)
            t = time.time()
            ok = True
            try:
                if first:
                    om, f0, xh, aJ = solve_bg(D, 0.0, r["omega"], r["f0"])
                    go, gp = lo0, lp0
                else:
                    om, f0, xh, aJ = solve_bg(D, kn, extrap(hist["om"], kn), extrap(hist["f0"], kn))
                    go, gp = complex(extrap(hist["osc"], kn)), complex(extrap(hist["phys"], kn))
                ok = aJ < 1e-10
            except Exception as e:
                ok = False
            if ok:
                rr = roots(D, kn, om, f0, xh, [go, gp])
                for res, g in zip(rr, (go, gp)):
                    lam = res[0][1]
                    sp_ = max(abs(q[1] - lam) for q in res)
                    if not all(q[2] for q in res) or sp_ > smax or abs(lam - g) > 0.02 * (1 + abs(lam)):
                        ok = False
            if not ok:
                if first:
                    raise SystemExit("kappa = 0 start failed")
                h *= 0.5
                if h < 1e-5:
                    emit(dict(kind="path_stop", D=D, kappa=kap, reason="step below 1e-5"))
                    break
                continue
            first = False
            kap = kn
            lo = rr[0][0][1]; lp = rr[1][0][1]
            hist["om"].append((kap, om)); hist["f0"].append((kap, f0))
            hist["osc"].append((kap, lo)); hist["phys"].append((kap, lp))
            emit(dict(kind="path", D=D, kappa=kap, omega=om, f0=f0, xh=xh, absJ=aJ,
                      osc=[[q[0], q[1].real, q[1].imag, q[2]] for q in rr[0]],
                      phys=[[q[0], q[1].real, q[1].imag, q[2]] for q in rr[1]],
                      osc_spread=max(abs(q[1] - lo) for q in rr[0]), gamma=1 / lp.real,
                      seconds=round(time.time() - t, 1)))
            if all(q[1].real < 0 for q in rr[0]):
                below += 1
                if below >= 1:
                    break
            h = min(hmax, 1.5 * h)
        return

    if cmd == "crit":
        P = [json.loads(l) for l in open(pos[1])]
        D = float(pos[2])
        P = [p for p in P if p.get("kind") == "path" and abs(p["D"] - D) < 1e-9]
        ks = [p["kappa"] for p in P]
        for ir, Rf in enumerate(RADII):
            res = [p["osc"][ir][1] for p in P]
            i = crossing(ks, res)
            if i is None:
                emit(dict(kind="crit", D=D, R_over_xh=Rf, crossing=False))
                continue
            pa, pb = P[i], P[i + 1]
            cache = {}

            def F(k, Rf=Rf, ir=ir):
                hk = [(p["kappa"], p["omega"]) for p in (pa, pb)]
                hf = [(p["kappa"], p["f0"]) for p in (pa, pb)]
                hl = [(p["kappa"], complex(p["osc"][ir][1], p["osc"][ir][2])) for p in (pa, pb)]
                om, f0, xh, aJ = solve_bg(D, k, extrap(hk, k), extrap(hf, k))
                (res_,) = roots(D, k, om, f0, xh, [complex(extrap(hl, k))], radii=(Rf,))
                lam = res_[0][1]
                cache[k] = (om, f0, xh, aJ, lam, res_[0][2])
                return lam.real
            k, fk, it = refine(F, pa["kappa"], pb["kappa"], pa["osc"][ir][1], pb["osc"][ir][1])
            om, f0, xh, aJ, lam, conv = cache[k]
            # lambda_phys at kappa_c, at this radius
            lp_g = complex(extrap([(p["kappa"], complex(p["phys"][ir][1], p["phys"][ir][2])) for p in (pa, pb)], k))
            (rp,) = roots(D, k, om, f0, xh, [lp_g], radii=(Rf,))
            lp = rp[0][1]
            emit(dict(kind="crit", D=D, R_over_xh=Rf, crossing=True, kappa_c=k, re_at_kappa_c=fk, iterations=it,
                      lam_osc=[lam.real, lam.imag], converged=conv, omega=om, f0=f0, xh=xh, absJ=aJ,
                      im_over_omega=lam.imag / om, bracket=[pa["kappa"], pb["kappa"]],
                      re_bracket=[pa["osc"][ir][1], pb["osc"][ir][1]], lam_phys=[lp.real, lp.imag],
                      gamma=1 / lp.real, omega_BD=(-1.5 + 1 / (8 * k)) if k > 0 else None))
        return
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main()

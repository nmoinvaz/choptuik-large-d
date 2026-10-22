#!/usr/bin/env python3
"""Continue the CSS solution of the HE kappa-model in D and compute its perturbation spectrum.

    python3 css_scan.py KAPPA DLIST [--N 40,56] [--A 5] [--kscale] [--out FILE]

    KAPPA   target curvature in HE97 normalization (1 = axion-dilaton, 0 = free complex scalar)
    DLIST   comma-separated target dimensions, e.g. 4,5,6,8,10,12,16,20
    --kscale  hold kappa*n/2 fixed instead of kappa (kappa(D) = KAPPA * 2/n), i.e. a fixed curvature in the
              repository's solver normalization (G = 2n T), where F_HE = sqrt(n) phi
    --N     two resolutions (resolution check; the spectrum is compared between them)
    --A     node-clustering parameter of the map w = (e^{A zeta}-1)/(e^A-1)

For each D in DLIST: Newton at both resolutions (continuation in D from the previous point in steps <= dDmax),
the generalized eigenproblem of the linearized evolution at both resolutions, and the eigenvalues that agree
between the two to 1e-6 (relative) are kept as converged.  Output: one JSON line per D.
"""
import sys, os, json, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import css_model as cm
import css_bg as cb

D4_GUESS = {  # published D = 4 data used only as the initial guess
    1.0: (1.17695272200, 2.60909347510, 0.892555411872),             # HHS96 eq. (4.6)
    0.0: (1.9154446, 5.0035380, np.sqrt(4 * np.pi) * 0.67217263 / 1.9154446),   # HE95a
}


def kappa_of(D, kap, kscale):
    n = D - 2
    return kap * 2.0 / n if kscale else kap


def converged_modes(l1, l2, tol=1e-6):
    out = []
    for z in l1:
        d = np.min(np.abs(l2 - z)) if len(l2) else np.inf
        if d <= tol * max(1.0, abs(z)):
            out.append((z, d))
    return out


def classify(z, om, tol=1e-6):
    if abs(z) < tol:
        return "phase (lambda=0)"
    if abs(z - 1) < tol:
        return "time translation (lambda=1)"
    if abs(z.real) < tol and abs(abs(z.imag) - om) < tol * max(1, om):
        return "target isometry (lambda=+-i omega)"
    return ""


def run(args):
    Ds = [float(s) for s in args.D.split(",")]
    N1, N2 = [int(s) for s in args.N.split(",")]
    A = args.A
    kap0 = args.kappa
    if args.guess:
        om, X, f0 = [float(s) for s in args.guess.split(",")]
        D0 = args.D0
    else:
        om, X, f0 = D4_GUESS[kap0 if not args.kscale else 1.0]
        D0 = 4.0
    out = open(args.out, "a") if args.out else None
    g1 = cm.Grid(N1, A)
    kap = kappa_of(D0, kap0, args.kscale)
    if args.guess and args.kscale:
        pass
    Y0 = cb.guess_from_shooting(g1, D0 - 2, kap, om, f0, X, wcut=0.999)
    Y1, res, it = cm.newton_bg(g1, Y0, D0 - 2, kap, tol=args.tol)
    print("start D=%.4f kappa=%.6f  res %.1e  omega %.10f" % (D0, kap, res, Y1[cm.NF * g1.M]), flush=True)
    Dcur = D0
    for Dt in Ds:
        # continuation in D
        while abs(Dt - Dcur) > 1e-12:
            step = np.clip(Dt - Dcur, -args.dD, args.dD)
            Dn = Dcur + step
            kap = kappa_of(Dn, kap0, args.kscale)
            Yn, res, it = cm.newton_bg(g1, Y1, Dn - 2, kap, tol=args.tol)
            if res > 1e-7:
                # halve the step
                args_dD = step / 2
                Dn = Dcur + args_dD
                kap = kappa_of(Dn, kap0, args.kscale)
                Yn, res, it = cm.newton_bg(g1, Y1, Dn - 2, kap, tol=args.tol)
                if res > 1e-7:
                    raise RuntimeError("continuation failed at D=%.4f (res %.1e)" % (Dn, res))
            Y1, Dcur = Yn, Dn
        n = Dt - 2
        kap = kappa_of(Dt, kap0, args.kscale)
        t0 = time.time()
        g2 = cm.Grid(N2, A)
        Y2, res2, _ = cm.newton_bg(g2, cm.regrid(g1, Y1, g2), n, kap, tol=args.tol)
        _, res1, _ = (Y1, np.max(np.abs(cm.bg_residual(g1, Y1, n, kap))), 0)
        ob1 = cb.observables(g1, Y1, n, kap)
        ob2 = cb.observables(g2, Y2, n, kap)
        l1, _ = cm.spectrum(g1, Y1, n, kap)
        l2, V2 = cm.spectrum(g2, Y2, n, kap)
        sel1 = l1[l1.real > args.remin]
        sel2 = l2[l2.real > args.remin]
        conv = converged_modes(sel2, sel1, tol=args.etol)
        conv.sort(key=lambda t: -t[0].real)
        unconv_pos = [z for z in sel2 if z.real > 1e-6 and all(abs(z - c[0]) > 1e-12 for c in conv)]
        phys = [c for c in conv if classify(c[0], ob2["omega"], 1e-5) == "" and c[0].real > 1e-6]
        lam = phys[0][0].real if phys and abs(phys[0][0].imag) < 1e-8 else float("nan")
        d2 = cm.unpack(g2, Y2)
        # profile diagnostics
        fabs = np.hypot(d2["sR"], d2["sI"])
        rec = dict(D=Dt, n=n, kappa=kap, N=[N1, N2], A=A,
                   omega=ob2["omega"], omega_N1=ob1["omega"], X=ob2["X"], X_N1=ob1["X"],
                   f0=ob2["f0"], fh=ob2["fh"], uh=ob2["uh"], res=[float(res1), float(res2)],
                   kfmax=float(kap * fabs.max() ** 2), b_h=float(d2["b"][-1]),
                   lam=lam, gamma_r=1.0 / lam if lam == lam else float("nan"),
                   modes=[[c[0].real, c[0].imag, c[1], classify(c[0], ob2["omega"], 1e-5)] for c in conv],
                   unconverged_positive=[[z.real, z.imag] for z in unconv_pos][:10],
                   n_phys_unstable=len([c for c in phys if c[0].real > 1e-6]),
                   time=time.time() - t0)
        print("D=%6.2f kappa=%.4f omega=%.10f (dN %.1e) X=%.8f |f0|=%.8f lam=%.10f gamma=%.8f  #phys_unstable=%d  unconv+=%d  (%.1fs)"
              % (Dt, kap, rec["omega"], rec["omega"] - rec["omega_N1"], rec["X"], rec["f0"], lam, rec["gamma_r"],
                 rec["n_phys_unstable"], len(unconv_pos), rec["time"]), flush=True)
        for c in conv[:8]:
            print("      %+.10f %+.10fi   (dN %.1e) %s" % (c[0].real, c[0].imag, c[1], classify(c[0], ob2["omega"], 1e-5)))
        if out:
            out.write(json.dumps(rec) + "\n"); out.flush()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kappa", type=float)
    ap.add_argument("D")
    ap.add_argument("--N", default="40,56")
    ap.add_argument("--A", type=float, default=5.0)
    ap.add_argument("--dD", type=float, default=0.5)
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--etol", type=float, default=1e-6)
    ap.add_argument("--remin", type=float, default=-0.5)
    ap.add_argument("--kscale", action="store_true")
    ap.add_argument("--guess", default=None, help="omega,X,f0 at D0 instead of the published D = 4 data")
    ap.add_argument("--D0", type=float, default=4.0)
    ap.add_argument("--out", default=None)
    run(ap.parse_args())

#!/usr/bin/env python3
"""Chained steps of the exact Taylor continuation in D (notes/QUAD_DOUBLE.md 13.15-13.16): from a converged pinned
state, the branch's Taylor coefficients to order N (anm_step.py's loop: Solver.residual_jet for the right-hand sides,
the Jacobian's LU at the base, one refinement per order), the step a = min(a_max, h_max) with
a_max = (eps |z_1| / |z_N|)^{1/(N-1)}, the prediction z(a), the corrector at D + a with T75's recipe (scan_hp.py's
"dnp": deflated_newton, then up to 4 plain Newton steps from below 1e-12; the tolerance grows 1.45x per unit of D
from the base's, and a stall below --accept_stall is accepted as T75 accepted its), and on convergence the new base;
on failure the step is halved, three times at most. Every
converged state is saved (npz, hpkernel.driver.save's layout) and every step recorded (jsonl), in a task folder of the
raw-data repository. p1's odd harmonics in (kfreeze, 13] stay at the first base's values throughout (the pinned
formulation of hp/pinning.py; their coefficients are zero).

    python3 anm_run.py RAWDIR OUTDIR --state T75/pin7_D73.2657.npz --kfreeze 3 [--N 24] [--eps 1e-16] [--h_max 2]
        [--steps 20] [--D_max 200] [--corrector dnp|deflated|newton] [--tol 1e-20] [--tol_growth 1.45]
        [--accept_stall 1e-19] [--junk_only 0] [--threads 4] [--w 24]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from hpkernel.driver import make_solver, newton_solve, deflated_newton, delta_correction, rnorm, save  # noqa: E402
from multidouble import dd_add, dd_scale, dd_norm  # noqa: E402
from pinning import PinnedSolver, band_coeffs, band_field  # noqa: E402
from raw_out import require_raw  # noqa: E402
from taylor_predict import formula_delta  # noqa: E402


def load_state(path):
    d = np.load(path)
    m = json.loads(str(d["meta"]))
    return d["zh"].astype(float), d["zl"].astype(float), m


def coefficients(s, zh, zl, N, threads_note=""):
    """z_1..z_N from the stored LU at (zh, zl): [(zh_k, zl_k)], norms, the refinement residuals"""
    Zh, Zl, norms, refine = [zh], [zl], [float(dd_norm(zh, zl))], []
    for k in range(1, N + 1):
        Hs, Ls = s.residual_jet(Zh, Zl, k + 1, 0)
        rh, rl = Hs[k], Ls[k]
        xh, xl = s.lu_solve(-rh, -rl)
        jh, jl = s.jvp(zh, zl, xh, xl)[:2]
        r0h, r0l = dd_add(-rh, -rl, -jh, -jl)
        ch, cl = s.lu_solve(r0h, r0l)
        xh, xl = dd_add(xh, xl, ch, cl)
        jh, jl = s.jvp(zh, zl, xh, xl)[:2]
        r1h, r1l = dd_add(-rh, -rl, -jh, -jl)
        Zh.append(xh); Zl.append(xl)
        norms.append(float(dd_norm(xh, xl)))
        refine.append((float(dd_norm(r0h, r0l)), float(dd_norm(r1h, r1l))))
    return Zh, Zl, norms, refine


def main():
    args = list(sys.argv[1:])
    opt = dict(state="T75/pin7_D73.2657.npz", kfreeze=3, N=24, eps=1e-16, h_max=2.0, steps=20, D_max=200.0, corrector="dnp",
               tol=1e-20, tol_growth=1.45, accept_stall=1e-17, junk_only=0, threads=4, w=24)
    for k in list(opt):
        f = "--" + k
        if f in args:
            i = args.index(f); opt[k] = type(opt[k])(args[i + 1]); del args[i:i + 2]
    raw, out = args[0], args[1]
    os.makedirs(out, exist_ok=True)
    rec_path = os.path.join(out, "anm_run.jsonl")
    require_raw(rec_path)
    zh, zl, m = load_state(os.path.join(raw, opt["state"]))
    D, M, nodes = float(m["D"]), int(m["M"]), list(m["nodes"])
    kt, kc, ks = m.get("kt", 36), m.get("kc", 146), m.get("ks", 130)
    kf, N = opt["kfreeze"], opt["N"]
    D_start = D
    junk_only = bool(opt["junk_only"])

    def tol_at(Dx):
        return opt["tol"] * opt["tol_growth"] ** (Dx - D_start)

    def correct(spx, ph, pl, tol):
        """T75's corrector (scan_hp.py dnp): deflated Newton, then plain Newton steps from below 1e-12 if they lower |F|;
        returns zh, zl, |F|, steps, accepted (converged, or stalled below accept_stall)"""
        if opt["corrector"] == "newton":
            ch, cl, hist = newton_solve(spx, ph, pl, tol=tol, it_max=12, verbose=False)
        else:
            ch, cl, hist = deflated_newton(spx, ph, pl, tol=tol, it_max=30, w=opt["w"], verbose=False, junk_only=junk_only)
            if opt["corrector"] == "dnp" and hist and hist[-1]["F"] < 1e-12:
                zh3, zl3, h3 = newton_solve(spx, ch, cl, tol=1e-40, it_max=4, verbose=False)
                if h3 and h3[-1]["F"] < hist[-1]["F"]:
                    ch, cl = zh3, zl3
                    hist = hist + h3
        Fh, Fl = spx.residual(ch, cl)
        Fc = float(rnorm(Fh, Fl))
        # a stall below accept_stall is accepted (T75's rule, at 1e-19 there; here 1e-17: the pinned horizon block's
        # floor after a prediction at h <= 1, 13.16), with the linearised Delta correction of the remaining residual
        # recorded as that point's Delta uncertainty
        dcorr = float(delta_correction(spx, Fh, Fl)) if Fc > tol and np.isfinite(Fc) else 0.0
        return ch, cl, Fc, len(hist), bool(Fc <= tol or Fc <= opt["accept_stall"]), dcorr
    mk = lambda Dx: make_solver(Dx, M, nodes=nodes, kt=kt, kc=kc, ks=ks, threads=opt["threads"], kfreeze=kf)  # noqa: E731
    ks_held = [k for k in range(kf + 1, 14) if k % 2 == 1]
    h_hi, h_lo = band_field(band_coeffs(zh[:M], zl[:M], M, ks_held), ks_held, M)
    wrap = lambda sx: PinnedSolver(sx, M, h_hi, h_lo)  # noqa: E731
    s = mk(D); s.set_segs(s.plan(zh, zl, 1e-32, 0.05)); sp = wrap(s)
    Fh, Fl = sp.residual(zh, zl)
    F0 = float(rnorm(Fh, Fl))
    if F0 > opt["tol"]:
        zh, zl, F0, _, _, _ = correct(sp, zh, zl, opt["tol"])
    print(f"base {opt['state']}: D {D}, M {M}, pin {kf} (held {ks_held}), |F| {F0:.1e}; N {N}, eps {opt['eps']}, h_max {opt['h_max']}, "
          f"corrector {opt['corrector']} (junk_only {junk_only}), tol {opt['tol']:.1e} x {opt['tol_growth']}^(D - {D_start}), "
          f"stalls accepted below {opt['accept_stall']:.0e}", flush=True)
    a = None
    for step in range(opt["steps"]):
        if D >= opt["D_max"]:
            break
        t0 = time.time()
        dh, dl, Fh, Fl, pmin, pmax = sp.newton(zh, zl)
        Zh, Zl, norms, refine = coefficients(s, zh, zl, N)
        t_coef = time.time() - t0
        a_max = (opt["eps"] * norms[1] / norms[N]) ** (1.0 / (N - 1)) if norms[N] else opt["h_max"]
        a = min(a_max, opt["h_max"], opt["D_max"] - D) if a is None else min(a, a_max, opt["h_max"], opt["D_max"] - D)
        ratios = [norms[k - 1] / norms[k] if norms[k] else None for k in range(2, N + 1)]
        rec = dict(step=step, D0=D, pin=kf, N=N, pivots=(float(pmin), float(pmax)), norms=norms, a_max=a_max, t_coef=t_coef,
                   refine_last=refine[-1], ratios_2_5=ratios[:4])
        print(f"step {step}: D {D:.6f}, pivots {pmin:.1e}..{pmax:.1e}, |z_1| {norms[1]:.2e}, |z_N| {norms[N]:.2e}, ratios 2-5 "
              + " ".join(f"{r:.1f}" for r in ratios[:4] if r) + f", a_max {a_max:.3f}, {t_coef:.0f} s", flush=True)
        converged = False
        for attempt in range(4):
            ph, pl = np.array(zh), np.array(zl)
            for k in range(1, N + 1):
                ph, pl = dd_add(ph, pl, *dd_scale(Zh[k], Zl[k], a ** k))
            Dn = D + a
            sn = mk(Dn)
            sn.set_segs(s.segs())
            spn = wrap(sn)
            Fh, Fl = spn.residual(ph, pl)
            Fp = float(rnorm(Fh, Fl))
            if np.isfinite(Fp) and Fp <= 1e-6:
                sn.set_segs(sn.plan(ph, pl, 1e-32, 0.05))
                Fh, Fl = spn.residual(ph, pl); Fp = float(rnorm(Fh, Fl))
            t1 = time.time()
            tol = tol_at(Dn)
            if np.isfinite(Fp):
                ch, cl, Fc, nsteps, accepted, dcorr = correct(spn, ph, pl, tol)
            else:
                ch, cl, Fc, nsteps, accepted, dcorr = ph, pl, float("nan"), 0, False, 0.0
            t_corr = time.time() - t1
            fd = float(formula_delta(Dn))
            att = dict(attempt=attempt, a=a, D=Dn, F_pred=Fp, F_final=Fc, tol=tol, steps=nsteps, accepted=accepted, t_corr=t_corr,
                       Delta=float(ch[-1] + cl[-1]), dDelta_formula=float(ch[-1] + cl[-1]) - fd, dDelta_corr=dcorr,
                       moved=float(dd_norm(*dd_add(ch, cl, -ph, -pl))))
            rec.setdefault("attempts", []).append(att)
            print(f"    a {a:.3f} -> D {Dn:.6f}: predicted |F| {Fp:.2e}, {opt['corrector']} {nsteps} steps -> {Fc:.2e} (tol {tol:.1e}"
                  f"{', stall accepted, Delta correction ' + format(dcorr, '+.1e') if accepted and Fc > tol else ''}), "
                  f"Delta {att['Delta']:.12f} (formula diff {att['dDelta_formula']:+.1e}), "
                  f"moved {att['moved']:.1e}, {t_corr:.0f} s", flush=True)
            if accepted:
                converged = True
                break
            a *= 0.5
        rec["converged"] = converged
        with open(rec_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if not converged:
            print("stopped: no convergence after three halvings", flush=True)
            break
        D, zh, zl, s, sp = Dn, ch, cl, sn, spn
        save(os.path.join(out, f"anm_pin{kf}_D{D:.4f}.npz"), zh, zl, dict(D=D, M=M, nodes=nodes, kt=kt, kc=kc, ks=ks, kfreeze=kf,
                                                                           pin_held=ks_held, N=N, a=a, source=opt["state"]))
        a = min(2 * a, opt["h_max"])          # the next step may grow again
    print("done", flush=True)


if __name__ == "__main__":
    main()

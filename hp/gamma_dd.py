#!/usr/bin/env python3
"""The critical exponent gamma(D) = 1/lam at double-double precision, from saved double-double states.

lam is the growth rate of the unstable perturbation e^{lam tau} (periodic) of the DSS solution: the root of
det M(lam), where M(lam) = Solver.mode_matrix is the exact Jacobian of the multiple-shooting residual for
perturbations (every tau-derivative of a perturbation gains + lam, Delta held fixed, no phase row), built by
forward-mode duals in double-double (hp/src/model.rs: Ctx::lam, dtau, psolve with the lam-shifted horizon LUs).
The f64 solver does the same with mode_matrix / gamma_root.py; this is its double-double version.

For each state the solver is rebuilt, with the kfreeze-13 cut from D = 35 on (as in the runs; without the cut, weak
directions bias the root by up to 2e-9 at D = 61), --kf=none|13|auto overrides (auto: the layout with the smaller
residual); det M is bracketed around the rational formula's lam = 1/gamma(D)
and the sign change located with brentq on sign * exp(ln|det| - ref). --polish[=N] first re-converges the state in the
rebuilt layout (up to N Newton steps, default 4; the up6 states at D = 27-35 come down from |F| ~ 1e-12 to 1e-25, and lam does
not change). States whose residual still exceeds --fmax (default 1e-20) are skipped and recorded as such.

    python3 gamma_dd.py out.jsonl STATE.npz [STATE.npz ...] [--threads=N] [--M48 | --Mhi=64] [--kf=none|13|auto] [--fmax=1e-20] [--polish[=N]]
        [--newton_it=12] [--lm_it=20] [--bracket=A,B] [--xtol=1e-14] [--ball=PREC] [--series_override]
        [--scan=CENTRE,STEP,K0,K1 [--scan_at=base|hi] [--scan_only]] [--mutate=old_rule|no_tail_gate]

--bracket=A,B searches the root of det M in [A, B] only, at every M, instead of the grids about the formula's value,
and a bracket without a sign change gives lam None and exit status 1 (#458: the root's resolution is the spread of lam
over brackets). --xtol is brentq's absolute tolerance (its relative tolerance is 1e-15). --ball=PREC takes det M as
the exact determinant of the double-double mode matrix (python-flint's ball LU at PREC bits, hp/modecount_qdet.py)
instead of its double-double LU's, at every M; the log's pivot range then holds the ball radius of ln|det| (#590).
--scan=CENTRE,STEP,K0,K1 takes the same det M at lam = CENTRE + k STEP for k = K0..K1, at the state's own M
(--scan_at=base) or at the resampled one (hi, the default with --M48 or --Mhi), into the record's scan list of (k, lam,
sign, ln|det|) (#625: whether det M crosses zero once near the root or many times in a band). --scan_only skips the
root searches, so that several processes can each take a range of k on the same background.

--M48 (or --Mhi=N) also resamples the state to M = 48 (N) tau points, re-converges it and repeats the root search,
bracketed afresh from the formula (self-convergence; the record's M48 field). The M = 48 solve uses the kfreeze-13 cut
from D = 35 on (without it Newton from the resampled state stalls at D = 40), with the resampled p1 projected onto the
cut first; --kf48=none|13 overrides. The resampling is in double-double (resample_hi) and Newton runs to the floor,
until a step no longer lowers |F| (solve_hi; Levenberg-Marquardt if that stops above --fmax). The record holds the
Newton step's Delta component at the solution (est), and the floor check wants |est| <= EST_MAX; the exit status is 1
when it fails. --mutate=old_rule restores the path before #271, the resampling in f64 with lo dropped and Newton to
max(10 |F|, 1e-27), which at D 50 stopped one step short along p1's harmonics 13 and 11 (Delta 1.2e-11 off, #266).
The record also holds both series' truncation (modecount_common.series_tail) at the resampled start and at the
solution, and a state where either exceeds SERIES_TAIL (1e-15) is refused there (no lam; refused gives the reason; exit
status 1) unless --series_override, which records the override (#267). --mutate=no_tail_gate removes that check.
Needs the hpkernel wheel built from this tree (hp/build.sh) for mode_det.
"""
import json
import math
import os
import sys
import time

import mpmath as mp
import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "review"))
from hpkernel.driver import dd_from_mp, make_solver, load, lm_solve, newton_solve, plow_odd, resample_z  # noqa: E402
import modecount_common as mc  # noqa: E402

# The largest |Newton step's Delta component| accepted at a re-converged high-M state. At the floor it is 2e-19 (D 50)
# and 5e-18 (D 58) (#266); Newton stopped one step short gives 1.2e-11 at D 50.
EST_MAX = 1e-15

# README rational formula for gamma (solver/formula_all.py), used only for the starting guess
E = [0.5, 1.8083629635e1, 1.5241588758e2, -5.2341588541e1, 9.9259594505, -9.8386450353e1, 7.4529506074e1,
     -4.8272358903e1]
FQ = [1, 3.7683686025e1, 3.4245505235e2]


def gamma_formula(D):
    x = 1.0 / (D - 2)
    return sum(c * x ** i for i, c in enumerate(E)) / sum(c * x ** i for i, c in enumerate(FQ))


def build(meta, zh, zl, threads, M=None, kfreeze=None):
    s = make_solver(meta["D"], M or meta["M"], nodes=meta["nodes"], kt=meta.get("kt", 36), kc=meta.get("kc", 420),
                    ks=meta.get("ks", 100), threads=threads, kfreeze=kfreeze)
    s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
    fh, fl = s.residual(zh, zl)
    return s, float(np.abs(np.asarray(fh) + np.asarray(fl)).max())


def best_solver(meta, zh, zl, threads, M=None, force=None):
    kfs = (None, 13) if force in (None, "auto") else ((None,) if force == "none" else (int(force),))
    cands = [(build(meta, zh, zl, threads, M, kf), kf) for kf in kfs]
    (s, F), kf = min(cands, key=lambda c: c[0][1])
    return s, F, kf


def det_fn(s, zh, zl, ball=0, log=None, cache=None):
    """lam -> (sign, ln|det M|) at the state, by the double-double LU (mode_det) or, with ball > 0, the exact
    determinant of the double-double matrix at ball bits; memoized in cache"""
    cache = {} if cache is None else cache

    def sd(lam):
        if lam not in cache:
            t = time.time()
            if ball:
                # the exact determinant of the double-double matrix (modecount_qdet.ball_det_real), not its LU's
                from modecount_qdet import ball_det_real
                mh, ml = s.mode_matrix(zh, zl, float(lam), 0.0)
                sg, la, rad = ball_det_real(np.asarray(mh), np.asarray(ml), ball)
                if sg == 0:
                    raise ValueError(f"the ball of det M at lam {lam!r} contains 0 at {ball} bits")
                pmin, pmax = rad, float("nan")
            else:
                sg, la, pmin, pmax = s.mode_det(zh, zl, float(lam), 0.0)
            cache[lam] = (sg, la)
            if log:
                log(f"    lam {lam:.15f}: sign {sg:+.0f} ln|det| {la:.6f} pivots {pmin:.2e}..{pmax:.2e} ({time.time() - t:.1f} s)")
        return cache[lam]
    return sd


def scan_det(s, zh, zl, scan, ball=0, log=None):
    """[(k, lam, sign, ln|det|)] at lam = c + k step for k = k0..k1 (--scan)"""
    c, step, k0, k1 = scan
    sd = det_fn(s, zh, zl, ball, log)
    out = []
    for k in range(k0, k1 + 1):
        lam = c + k * step
        sg, la = sd(lam)
        out.append([k, lam, float(sg), float(la)])
    if log:
        ch = sum(1 for a, b in zip(out, out[1:]) if a[2] != b[2])
        log(f"  scan: {len(out)} points, {ch} sign changes between neighbours")
    return out


def find_lam(s, zh, zl, lam0, log=None, bracket=None, xtol=1e-14, ball=0):
    """the root of det M near lam0, from the widening grids about lam0, or in bracket (a, b) when given (#458: the
    root's resolution as its spread over brackets); None when the bracket has no sign change"""
    cache = {}
    sd = det_fn(s, zh, zl, ball, log, cache)

    if bracket:
        a, b = bracket
        if sd(a)[0] == sd(b)[0]:
            return None, 0, len(cache)
        ref = 0.5 * (sd(a)[1] + sd(b)[1])
        f = lambda l: sd(l)[0] * math.exp(sd(l)[1] - ref)
        return brentq(f, a, b, xtol=xtol, rtol=1e-15, maxiter=80), 1, len(cache)
    for w in (3e-4, 2e-3, 1e-2, 4e-2):
        grid = [lam0 - w, lam0, lam0 + w]
        sg = [sd(l)[0] for l in grid]
        ch = [i for i in range(2) if sg[i] != sg[i + 1]]
        if ch:
            i = min(ch, key=lambda i: abs(0.5 * (grid[i] + grid[i + 1]) - lam0))
            a, b = grid[i], grid[i + 1]
            ref = 0.5 * (sd(a)[1] + sd(b)[1])
            f = lambda l: sd(l)[0] * math.exp(sd(l)[1] - ref)
            lam = brentq(f, a, b, xtol=xtol, rtol=1e-15, maxiter=60)
            return lam, len(ch), len(cache)
    return None, 0, len(cache)


def project_p1_dd(zh, zl, M, kf):
    """p1 (the first M entries) projected onto its odd harmonics k <= kf (plow_odd), at 40 digits, rounded to
    double-double"""
    a_, b_ = plow_odd(M, kf)
    zh, zl = np.array(zh, float), np.array(zl, float)
    with mp.workdps(40):
        v = [mp.mpf(float(h)) + mp.mpf(float(l)) for h, l in zip(zh[:M], zl[:M])]
        out = [mp.fsum(v[j] * (mp.mpf(float(a_[i, j])) + mp.mpf(float(b_[i, j]))) for j in range(M)) for i in range(M)]
        zh[:M], zl[:M] = dd_from_mp(out)
    return zh, zl


def resample_hi(zh, zl, M0, M, K, kf, mutate=""):
    """The state on M tau points, with p1 projected onto the cut kf (if set). Trigonometric interpolation at 40 digits
    (rv2_resolve.resample_dd), the projection in double-double. mutate=old_rule: resample_z in f64 with lo dropped and
    the projection in f64 (the path before #271), which leaves 1e-16 in the weak directions."""
    if mutate == "old_rule":
        hh, hl = resample_z(np.asarray(zh) + np.asarray(zl), M0, M, K)
        if kf:
            a_, b_ = plow_odd(M, int(kf))
            hh[:M] = (a_ + b_) @ hh[:M]
        return hh, hl
    from rv2_resolve import resample_dd
    hh, hl = resample_dd(np.asarray(zh), np.asarray(zl), M0, M, K)
    if kf:
        hh, hl = project_p1_dd(hh, hl, M, int(kf))
    return np.ascontiguousarray(hh), np.ascontiguousarray(hl)


def solve_hi(s, zh, zl, F_ref, fmax, verbose=True, mutate="", newton_it=12, lm_it=20):
    """Re-converge a resampled state on the solver s. Newton to tol 1e-40, so it stops at the floor, where the line
    search finds no step that lowers |F| (as #266's R2 path), then Levenberg-Marquardt if that is above fmax.
    mutate=old_rule: tol max(10 F_ref, 1e-27), the rule before #271. Returns zh, zl and a record: |F|, the steps, the
    stop, and the Newton step at the solution (its Delta component est, the linearised Delta correction). newton_it and
    lm_it are the Newton and Levenberg-Marquardt iteration limits (--newton_it, --lm_it)."""
    tol = max(10 * F_ref, 1e-27) if mutate == "old_rule" else 1e-40
    zh, zl, hist = newton_solve(s, zh, zl, tol=tol, it_max=newton_it, verbose=verbose)
    rec = dict(tol=tol, newton=sum("lam" in h for h in hist))
    if hist[-1]["F"] > fmax:
        if verbose:
            print("  Newton stalled; Levenberg-Marquardt", flush=True)
        zh, zl, hist = lm_solve(s, zh, zl, tol=tol, it_max=lm_it, verbose=verbose, newton_after=1e-22)
        rec["lm"] = sum(1 for h in hist if h.get("lm"))
        rec["newton_after_lm"] = sum("lam" in h for h in hist)
    last = hist[-1]
    # floor: the line search found no decrease; lm: LM ended without handing over; other: it_max or a step below 1e-31
    rec["stop"] = ("floor" if last.get("failed") else "lm" if last.get("lm") else "tol" if "lam" not in last
                   else "other")
    zh, zl = np.ascontiguousarray(zh), np.ascontiguousarray(zl)
    # the Newton step at the solution, with a fresh LU there: its Delta component is how far one more step moves Delta
    dh, dl, fh, fl, pmin, pmax = s.newton(zh, zl)
    rec.update(F=float(np.abs(np.asarray(fh) + np.asarray(fl)).max()), est=float(dh[-1] + dl[-1]),
               step=float(np.abs(np.asarray(dh) + np.asarray(dl)).max()))
    rec["floor"] = "ok" if abs(rec["est"]) <= EST_MAX else "ABOVE"
    return zh, zl, rec


def tails_of(s, zh, zl, meta):
    """both series' truncation on the layout of s (modecount_common.series_tail): the centre series at the first node,
    the horizon series at the last"""
    return [mc.series_tail(s, zh, zl, meta, w) for w in (0, 1)]


def tail_gate(tails, when, mutate=""):
    """None when both series' truncation is at most SERIES_TAIL (1e-15), else the reason for refusing the state.
    mutate=no_tail_gate removes the check (its control: a planted state above the gate is then accepted)."""
    worst = max(t["tail"] for t in tails)
    if mutate == "no_tail_gate" or worst <= mc.SERIES_TAIL:
        return None
    return f"series truncation {worst:.1e} > {mc.SERIES_TAIL:.0e} at the {when} ({mc.tails_text(tails)})"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    threads = int(opts.get("threads", 1))
    mutate = opts.get("mutate", "")
    bracket = tuple(float(x) for x in opts["bracket"].split(",")) if "bracket" in opts else None
    xtol = float(opts.get("xtol", 1e-14))
    ball = int(opts.get("ball", 0))
    if ball:
        from flint import ctx
        ctx.threads = threads
    root = dict(bracket=list(bracket) if bracket else None, xtol=xtol, **({"ball": ball} if ball else {}))
    scan = None
    if "scan" in opts:
        c, step, k0, k1 = opts["scan"].split(",")
        scan = (float(c), float(step), int(k0), int(k1))
    scan_at = opts.get("scan_at", "hi" if ("M48" in opts or "Mhi" in opts) else "base")
    scan_only = "scan_only" in opts
    if scan_only and not scan:
        sys.exit("--scan_only needs --scan")
    out = args[0]
    log = lambda m: print(m, flush=True)
    status = 0
    for f in args[1:]:
        zh, zl, meta = load(f)
        D = meta["D"]
        t0 = time.time()
        # the kfreeze-13 cut from D = 35 on: without it weak (junk) directions bias the det root by up to 2e-9 at
        # D = 61 (four states within dD = 0.005 scatter by 2e-9 about a line; with the cut by 3e-11)
        s, F, kf = best_solver(meta, zh, zl, threads, force=opts.get("kf", "13" if D >= 35 else "none"))
        F_in = F
        if "polish" in opts:
            # re-converge in the rebuilt layout (its segment plan differs from the scan's)
            zh, zl, hist = newton_solve(s, zh, zl, tol=1e-29, it_max=int(opts["polish"]) if opts["polish"] != "1" else 4,
                                        verbose=True)
            fh, fl = s.residual(zh, zl)
            F = float(np.abs(np.asarray(fh) + np.asarray(fl)).max())
        lam0 = 1.0 / gamma_formula(D)
        log(f"D {D:.6f}: |F| {F:.1e} (kfreeze {kf}), formula lam {lam0:.9f}")
        fmax = float(opts.get("fmax", 1e-20))
        if F > fmax:
            rec = dict(D=D, M=meta["M"], kfreeze=kf, F=F, lam=None, gamma=None, lam_formula=lam0,
                       skipped=f"|F| {F:.1e} > {fmax:.0e}: state not converged in this layout", state=os.path.basename(f))
            log(json.dumps(rec))
            with open(out, "a") as fh_:
                fh_.write(json.dumps(rec) + "\n")
            continue
        lam, nch, nev = (None, 0, 0) if scan_only else find_lam(s, zh, zl, lam0, log, bracket, xtol, ball)
        rec = dict(D=D, M=meta["M"], kfreeze=kf, F=F, **({"F_in": F_in} if "polish" in opts else {}), lam=lam, gamma=(1 / lam if lam else None), lam_formula=lam0,
                   n_signchanges=nch, n_det=nev, state=os.path.basename(f), seconds=round(time.time() - t0, 1),
                   **(root if bracket or "xtol" in opts or ball else {}))
        if scan and scan_at == "base":
            rec["scan"] = scan_det(s, zh, zl, scan, ball, log)
        if bracket and lam is None and not scan_only:
            log(f"  no sign change of det M in the bracket {bracket}")
            status = 1
        if ("M48" in opts or "Mhi" in opts) and (lam or scan_only):
            # self-convergence: resample to Mhi (default 48) collocation points in tau, re-converge, find lam
            t1 = time.time()
            rec.update(Delta=float(zh[-1]), Delta_lo=float(zl[-1]))
            Mhi = int(opts.get("Mhi", 48))
            K = len(meta["nodes"])
            m48 = dict(meta, M=Mhi)
            kf48 = opts.get("kf48", "13" if D >= 35 else "none")  # at D = 40 Newton from the resampled state stalls without the cut
            # the frozen formulation pins p1's harmonics above the cut: the resampled p1 is projected onto k <= kf
            z48h, z48l = resample_hi(zh, zl, meta["M"], Mhi, K, None if kf48 == "none" else int(kf48), mutate)
            s48, F48s, kf48 = best_solver(m48, z48h, z48l, threads, M=Mhi, force=kf48)
            hi = dict(M=Mhi, kfreeze=kf48, resample="f64" if mutate == "old_rule" else "dd", F_start=F48s,
                      tails_start=tails_of(s48, z48h, z48l, m48), **({"mutate": mutate} if mutate else {}))
            log(f"  M = {Mhi}: resampled ({'f64' if mutate == 'old_rule' else 'double-double'}) |F| {F48s:.1e} "
                f"(kfreeze {kf48}); series truncation {mc.tails_text(hi['tails_start'])}")
            # the series gate (#267): a layout whose series are truncated above SERIES_TAIL is refused at the start
            # and at the solution, unless --series_override
            why = tail_gate(hi["tails_start"], "start", mutate)
            if why and "series_override" not in opts:
                hi.update(refused=why, lam=None, lam_diff=None)
            else:
                if why:
                    hi["override"] = [why]
                z48h, z48l, sol = solve_hi(s48, z48h, z48l, F, fmax, verbose=True, mutate=mutate,
                                           newton_it=int(opts.get("newton_it", 12)), lm_it=int(opts.get("lm_it", 20)))
                F48, est, floor = sol.pop("F"), sol.pop("est"), sol.pop("floor")
                log(f"  M = {Mhi}: |F| {F48:.1e} ({sol['stop']}); Newton step at the solution: Delta {est:+.2e}, "
                    f"max {sol['step']:.1e}; floor check {floor} (|est| <= {EST_MAX:.0e})")
                hi.update(F=F48, solve=sol, est=est, floor=floor, tails=tails_of(s48, z48h, z48l, m48),
                          Delta=float(z48h[-1]), Delta_lo=float(z48l[-1]),
                          Delta_diff=float((z48h[-1] - zh[-1]) + (z48l[-1] - zl[-1])))
                log(f"  M = {Mhi}: series truncation at the solution {mc.tails_text(hi['tails'])}")
                why = tail_gate(hi["tails"], "solution", mutate)
                if why and "series_override" not in opts:
                    hi.update(refused=why, lam=None, lam_diff=None)
                else:
                    if why:
                        hi["override"] = hi.get("override", []) + [why]
                    lam48 = None
                    if F48 <= fmax and scan and scan_at == "hi":
                        hi["scan"] = scan_det(s48, z48h, z48l, scan, ball, log)
                    if F48 <= fmax and not scan_only:
                        lam48, nch48, _ = find_lam(s48, z48h, z48l, lam0, log, bracket, xtol, ball)  # independently of lam
                        if bracket and lam48 is None:
                            log(f"  M = {Mhi}: no sign change of det M in the bracket {bracket}")
                            status = 1
                    hi.update(lam=lam48, lam_diff=(lam48 - lam) if lam48 and lam else None)
            if "refused" in hi:
                log(f"  M = {Mhi}: refused, {hi['refused']}; --series_override accepts it")
            hi["seconds"] = round(time.time() - t1, 1)
            rec["M48"] = hi
            if "refused" in hi or hi.get("floor", "ok") != "ok":
                status = 1
        log(json.dumps(rec))
        with open(out, "a") as fh_:
            fh_.write(json.dumps(rec) + "\n")
    return status


if __name__ == "__main__":
    sys.exit(main())

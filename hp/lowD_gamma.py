#!/usr/bin/env python3
"""gamma(D) = 1/lam below D = 4 from double-double states (hp/data/lowD_states/*.npz): the root of det M(lam) by
hp/gamma_dd.find_lam, with the starting guess interpolated from a table of earlier low-D values (the f64 kc 47 sweep,
solver/data/scan_lowD3_kc47.jsonl, and the README value at D = 4) instead of the rational formula, which is not fitted
below D = 4. The guess only places the bracket; the root is located by sign changes of det M(lam).

    python3 lowD_gamma.py OUT.jsonl STATE.npz [STATE.npz ...] [--threads=2] [--polish=N] [--lam0=L] [--xtol=1e-11]
                          [--hmax=0.05]   (the step plan's largest step; with --polish the state re-converges on it)
                          [--scan=LO,HI,N]   det M(lam)'s sign and ln|det| at N points of [LO, HI], written to OUT; no root
        [--mutate_map=gp|phase_gp]
    python3 lowD_gamma.py --selftest UNIFORM.npz [MAPPED.npz] [--threads=2] [--break]

--polish=N: re-converge the state first in the rebuilt layout (N Newton steps), as gamma_dd.py --polish.
The solver is rebuilt with the state's own centre/horizon series lengths (meta kc, ks) and a fresh step plan.

A state saved on the tau-mapped grid (hp/tau_map_gain.py solve, meta["map"] = [s0, L]) is read on that grid
(tau_map.make_solver_mapped): the kernel's d/dtau is omega times the solver's derivative rows, which the map replaces,
and a mode matrix adds lam times each perturbation pointwise in tau (model.rs dtau, Rate::shift), so det M(lam) needs no
other change. --mutate_map breaks the mapped inputs as tau_map.MUTATIONS (a control: gamma must move).

--selftest: (1) on UNIFORM, det M(lam) from the uniform solver and from the mapped one at the identity map (L 1, s0 0)
are equal bit for bit (sign and ln|det|); --break uses L 0.99 instead, and the comparison must then fail. (2) on
MAPPED, the same lam with --mutate_map gp (d/dtheta without 1/g') must move ln|det| by more than 1e-6. Exit 1 on failure.
"""
import json
import math
import os
import sys
import time

import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hpkernel.driver import load, make_solver, newton_solve  # noqa: E402


def find_lam(s, zh, zl, lam0, log=None, xtol=1e-11, widths=(1e-4, 1e-3, 1e-2, 4e-2, 1e-1)):
    """root of det M(lam) near lam0 (as hp/gamma_dd.find_lam, with a tighter first bracket and xtol, since each det
    costs one double-double Jacobian and LU: about 6-8 evaluations from a guess within 1e-4)"""
    cache = {}

    def sd(lam):
        if lam not in cache:
            t = time.time()
            sg, la, pmin, pmax = s.mode_det(zh, zl, float(lam), 0.0)
            cache[lam] = (sg, la)
            if log:
                log(f"    lam {lam:.12f}: sign {sg:+.0f} ln|det| {la:.6f} pivots {pmin:.2e}..{pmax:.2e} ({time.time() - t:.1f} s)")
        return cache[lam]

    for w in widths:
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

# guesses for lam = 1/gamma, by D: the M 128 double-double values below D 3.6 (D 3.15 and 3.425 raw I1070-macm3,
# 392ae38d; D 3.475 raw I1070-dh170; D 3.5 #628), the f64 kc 47 values (scan_lowD3_kc47.jsonl) and D = 4
TABLE = [(3.15, 1 / 0.2273872422), (3.425, 1 / 0.3092042856), (3.475, 1 / 0.3182480362), (3.5, 1 / 0.3223855828),
         (3.6425, 1 / 0.34213711), (3.675, 1 / 0.34590386), (3.7, 1 / 0.34864959), (3.7938, 1 / 0.35792558),
         (4.0, 1 / 0.37396062), (4.25, 1 / 0.38807371), (4.5, 1 / 0.39853855), (5.0, 1 / 0.41312694)]


def lam_guess(D):
    Ds = np.array([t[0] for t in TABLE])
    ls = np.array([t[1] for t in TABLE])
    if D < Ds[0]:
        c = np.polyfit(Ds[:3], ls[:3], 2)
        return float(np.polyval(c, D))
    return float(np.interp(D, Ds, ls))


def solver_kw(meta, threads):
    return dict(nodes=meta["nodes"], kt=meta.get("kt", 36), kc=meta.get("kc", 60), ks=meta.get("ks", 100),
                threads=threads)


def build_solver(D, M, meta, threads, mutate=None):
    """(solver, grid): the uniform solver, or the mapped one when the state was saved on a mapped grid (meta["map"] =
    [s0, L]) or a mutation of the mapped inputs is asked for (on the identity map when the state has none)"""
    grid = meta.get("map")
    if grid is None and mutate is None:
        return make_solver(D, M, **solver_kw(meta, threads)), None
    import tau_map
    s0, L = (float(grid[0]), float(grid[1])) if grid is not None else (0.0, 1.0)
    return tau_map.make_solver_mapped(D, M, s0, L, mutate=mutate, **solver_kw(meta, threads)), [s0, L]


def selftest(args, opts, log):
    """(1) identity: uniform and mapped (L 1, s0 0; with --break L 0.99) det M(lam) on the same state, plan and lam,
    equal bit for bit; (2) on a mapped state, --mutate_map gp moves ln|det| by more than 1e-6 (relative)"""
    import tau_map
    threads = int(opts.get("threads", 2))
    ok = True
    zh, zl, meta = load(args[0])
    D, M = meta["D"], meta["M"]
    if meta.get("map") is not None:
        raise SystemExit("selftest: UNIFORM must be a state on the uniform grid")
    lam = float(opts["lam0"]) if "lam0" in opts else lam_guess(D)
    su, _ = build_solver(D, M, meta, threads)
    segs = su.plan(zh, zl, 1e-32, 0.05)
    su.set_segs(segs)
    L = 0.99 if "break" in opts else 1.0
    sm = tau_map.make_solver_mapped(D, M, 0.0, L, **solver_kw(meta, threads))
    sm.set_segs(segs)
    a, b = su.mode_det(zh, zl, lam, 0.0), sm.mode_det(zh, zl, lam, 0.0)
    same = a[0] == b[0] and a[1] == b[1]
    log(f"(1) {os.path.basename(args[0])} D {D} M {M}, lam {lam:.10f}: uniform sign {a[0]:+.0f} ln|det| {a[1]!r}; "
        f"mapped L {L} sign {b[0]:+.0f} ln|det| {b[1]!r}: {'equal' if same else 'DIFFERENT'} (must be equal)")
    ok &= same
    if len(args) > 1:
        zh2, zl2, m2 = load(args[1])
        if m2.get("map") is None:
            raise SystemExit("selftest: MAPPED must be a state saved with meta['map']")
        D2, M2 = m2["D"], m2["M"]
        lam2 = lam_guess(D2)
        s1, grid = build_solver(D2, M2, m2, threads)
        segs2 = s1.plan(zh2, zl2, 1e-32, 0.05)
        s1.set_segs(segs2)
        s2, _ = build_solver(D2, M2, m2, threads, mutate="gp")
        s2.set_segs(segs2)
        d1, d2 = s1.mode_det(zh2, zl2, lam2, 0.0), s2.mode_det(zh2, zl2, lam2, 0.0)
        moved = d1[0] != d2[0] or abs(d1[1] - d2[1]) > 1e-6 * abs(d1[1])
        log(f"(2) {os.path.basename(args[1])} D {D2} M {M2} map {grid}, lam {lam2:.10f}: ln|det| {d1[1]!r}, with "
            f"--mutate_map gp {d2[1]!r}: {'moved' if moved else 'UNMOVED'} (must move)")
        ok &= moved
    print(f"lowD_gamma selftest{' --break' if 'break' in opts else ''}: {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    log = lambda m: print(m, flush=True)
    if "selftest" in opts:
        return selftest(args, opts, log)
    threads = int(opts.get("threads", 2))
    out = args[0]
    for f in args[1:]:
        zh, zl, meta = load(f)
        D, M = meta["D"], meta["M"]
        t0 = time.time()
        s, grid = build_solver(D, M, meta, threads, mutate=opts.get("mutate_map"))
        hmax = float(opts.get("hmax", 0.05))           # the plan's largest step; --polish re-converges on it
        s.set_segs(s.plan(zh, zl, 1e-32, hmax))
        if "polish" in opts:
            zh, zl, _ = newton_solve(s, zh, zl, tol=1e-29, it_max=int(opts["polish"]), verbose=True)
        fh, fl = s.residual(zh, zl)
        F = float(np.abs(np.asarray(fh) + np.asarray(fl)).max())
        lam0 = float(opts["lam0"]) if "lam0" in opts else lam_guess(D)
        log(f"{os.path.basename(f)}: D {D} M {M} |F| {F:.1e}, guess lam {lam0:.6f}")
        if "scan" in opts:                          # --scan=LO,HI,N: det M(lam)'s sign and ln|det| on a grid, no root
            lo, hi, n = opts["scan"].split(",")
            lo, hi, n = float(lo), float(hi), int(n)
            for k in range(n):
                lam = lo + (hi - lo) * k / (n - 1)
                t = time.time()
                sg, la, pmin, pmax = s.mode_det(zh, zl, lam, 0.0)
                row = dict(D=D, M=M, F=F, lam=lam, sign=float(sg), ln=float(la), pmin=float(pmin), pmax=float(pmax),
                           hmax=hmax, state=os.path.basename(f), map=grid, sec=round(time.time() - t, 1), scan=True)
                with open(out, "a") as fh:
                    fh.write(json.dumps(row) + "\n")
                log(f"    scan lam {lam:.12f}: sign {sg:+.0f} ln|det| {la:.6f} pivots {pmin:.2e}..{pmax:.2e} ({row['sec']} s)")
            continue
        lam, nch, nev = find_lam(s, zh, zl, lam0, log, xtol=float(opts.get("xtol", 1e-11)))
        rec = dict(D=D, M=M, F=F, lam=lam, gamma=(1 / lam if lam else None), lam_guess=lam0, n_signchanges=nch,
                   n_det=nev, state=os.path.basename(f), map=grid, mutate_map=opts.get("mutate_map"), hmax=hmax,
                   seconds=round(time.time() - t0, 1))
        log(json.dumps(rec))
        with open(out, "a") as fh_:
            fh_.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    sys.exit(main())

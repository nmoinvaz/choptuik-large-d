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
        [--accept_stall 1e-19] [--junk_only 0] [--threads 4] [--w 24] [--trend T75] [--tol_qd 1e-28] [--accept_stall_qd 0]
        [--qd_it 12] [--qd_w 0] [--refine 1] [--probe 0 --probe_c C] [--plan_tol 1e-32] [--fix1 0] [--F_pred_max 0]
        [--horizon_alpha 0] [--fix1_qlu 0] [--jet_q 0] [--reseed 0] [--pred_q 0] [--reseed_q 0] [--save_pred 0]
        [--jet_o 0] [--corrector od --tol_od 1e-40] [--base_adopt 0] [--base_polish 1] [--keep_plan 1] [--held_fit FILE]
        [--held_o 0] [--reseed_o 0] [--a0 0] [--a_grow 2] [--a_hold 0] [--lam_stall 0] [--save_base 0]

--corrector qd runs the quad-double Newton of the kernel (hpkernel.driver.newton_qd, Solver.newton_q) on the pinned
system (pinning.PinnedQ: the held harmonics' offset subtracted from the quad-double residual and added to the step, as
PinnedSolver does in double-double) from the double-double prediction, to --tol_qd on max |F| in at most --qd_it steps
(the quadratic phase begins only below |dz| ~ 1e-11, 13.19, so 8 cut off converging iterations). The base is polished the
same way first. The result's first two parts are the next base (the jets are double-double) and all four parts are
saved (save_q). Where the double-double Newton takes no step from a residual of 1e-14 in the horizon block's weak
directions (13.17), the quad-double one sees those directions. (Freezing the held harmonics as directions on the
unpinned kernel instead diverges: the rows the pinning drops stay, 13.19.) --qd_w W > 0 starts that Newton deflated
(newton_qd's phase 'all': the step without its components along the Jacobian's W weakest right singular directions),
and plain Newton finishes once that phase stalls, so a converged state solves the whole pinned system.

--trend FOLDER holds the harmonics on a trend instead of at the base's values (T75's pin-to-trend, in the
continuation's terms): for each held harmonic k the coefficients (a_k, b_k) of p1 across the stored states of FOLDER are
fitted as |v| = C n^s (n = D - 2) when their sign is constant and the fit's residual in ln|v| is below 0.05 (harmonics
3-9 at D 71-73: residuals 1e-5 to 5e-3); the others stay constant. The trend is taken relative to the base, v(D) =
v(D0) (n/n0)^s, so the base state is unchanged, its Taylor coefficients in h enter the jet state order by order (the
held rows of each right-hand side carry them, as PinnedSolver's offset carries the values), and the corrector at D + a
holds the harmonics at the trend's values there. Records carry the exponents.
--held_fit FILE holds them instead on a fit of the full-system states (hp/held_fit.py fit; pinning.HeldFit: sqrt(n) v
a Chebyshev series in 1/n per coefficient), also relative to the base, v(D) = v(D0) + fit(D) - fit(D0), its Taylor
coefficients in h from the fit's power series; it replaces --trend. The trend's harmonic-5 power law is what puts the
chain 5e-8 below the series at D 79.8 (notes/HELD_HARMONICS.md); the fit is meant to start from a full-system state.

--refine R takes R refinement steps per order against the stored LU (1, the default, is the loop above). --probe 1
(--probe 2: only the two links of the order-1 equation, against quad-double central differences)
measures the series' first-order error at the base and stops (13.19's correction): the order-1 residual in the jet's
model after each refinement; |F_pred| / a at a = 1/64, 1/32, 1/16 in double-double and quad-double; the quad-double
model's order-1 residual of z_1 by central differences along (z_1, 1) in (z, D), at steps 2^-30 and 2^-33, with the held
harmonics' slope subtracted on their rows as the series' right-hand side carries it; z_1 corrected twice by that
residual through the stored LU, the series rebuilt on it, and its predictions at a = 1/64 to 1. --probe_c C is the
chain's recorded |F_pred| / a at this base: the probe exits 1 unless its own double-double value at a = 1/16 is within
1% of it (the check that the probe reproduces the chain). Results go to OUTDIR/probe.jsonl.
--plan_tol T is the tolerance of the first base's Taylor plan (1e-32; the chain replans each prediction at 1e-32).
A run started from a saved four-part state polishes it from all four parts, so a restart continues where the chain
stopped (from the first two alone the polish starts eps_dd G off the root, outside its basin past D ~ 131).
--fix1 1 corrects z_1 at every base by the quad-double model's order-1 residual (central differences at 2^-29 and
2^-30, Richardson-combined, two passes through the stored LU) and rebuilds the series on it; records carry the
residual norms (fix1). The double-double jet's own z_1 is wrong by 1e-4 relative at D 82.16 (13.19).
--F_pred_max P (0: off) skips the corrector for a prediction above P and shrinks the step by 0.9 (P / |F_pred|)^(1/2)
(at least 1/4) at once: with --fix1 the prediction error is second order in a, and from D 79.3 to 84.5 every attempt
predicted below 1.53e-14 converged while a failed correction costs up to 12 min. Records carry the skipped (a, |F|).
--horizon_alpha A is the horizon series' form for every solver of the run (make_solver's horizon_alpha; 0, the
default, as before; 1 carries alpha = a - 1, notes/LARGE_D_MIXED.md 10). The model is the same in both forms, only the
rounding differs, so the quad-double corrector solves the same system. The jet's order-1 error is rounding amplified
by the mass mode in either form, and which form is smaller depends on the base.
--jet_q 1 builds the series with the quad-double jet (Solver.residual_jet_q, multidouble 2.0.0's QJet) through the
quad-double LU at the base's four parts (Solver.newton_q, once per base), so every z_k carries the quad-double
rounding instead of the double-double one amplified by the mass mode; --fix1 is then not applied. hp/qjet_check.py
checks the quad-double jet against the quad-double residual on a stored state.
--jet_o 1 builds the series in octo-double (notes/OCTO_DOUBLE.md): the octo-double jet (Solver.residual_jet_o,
multidouble 4.0.0's OJet; M = 32 only) through the octo-double LU at the base (Solver.newton_o, once per base, at the
base's four quad-double parts promoted to eight), the held rows added in octo-double (od_add_dd). The prediction is
summed by Horner's rule in octo-double (od_axpy) and rounded to quad-double, where the quad-double corrector starts
(--corrector od starts from all eight parts), so it needs --corrector qd or od and with --reseed also --reseed_q 1; it
replaces --jet_q, --pred_q and --fix1, and --probe does not apply. Its series errs by about the octo-double unit times
G^2 where the quad-double one errs by the quad-double unit times G^2, which reaches the chain's tolerance near D
170-185; hp/ojet_check.py compares the two on stored states.
--reseed 1 replaces each prediction's horizon data (m_h, a_h) by those of the regular solution through its last node
before the residual, the plan and the corrector (hp/reseed.py, notes/RESEED.md: the node integrated toward the horizon,
the series inverted on the rungs u0/32 .. u0 in the m and a rows); 0, the default, changes nothing. Records carry each
re-seed's rungs (reseed). --reseed 2 does the same and runs the quad-double corrector with the horizon data eliminated
(reseed.ElimQ: every point the corrector steps from or tries is re-seeded at u0 while its max |F| exceeds 1e-15), which
converged from the series at a = 5 where the single re-seed did not (notes/RESEED.md 4.4). The re-seed's floor test
is reseed.py's default 'stall' (amendment 10), which converged in 19 of 20 trials at D 106 where the first design's
converged in 3; a re-seed that fails keeps the prediction, and the attempt then usually fails. With --reseed 2 the
step's solution, stored and carried on as the next base, is the projection of the corrector's last iterate, whose
residual is the one the corrector converged (notes/QUAD_DOUBLE.md 13.26). --save_pred 1 stores each prediction before
its re-seed as pred_pin<k>_D<D>.npz, an input for testing the re-seed (13.26). --save_dz0 1 (with --corrector od)
stores each attempt's corrector start and first Newton step as dz0_pin<k>_D<D>_a<a>.npz, for #499.
--corrector od is the corrector in octo-double (hp/octo_corrector.py, notes/OCTO_DOUBLE.md 5): newton_od, newton_qd's
rule on the octo-double kernel (PinnedO: Solver.newton_o, olu_solve and residual_o with the held offset), to --tol_od
on max |F| in at most --qd_it steps, with --reseed 2 inside ElimO (ElimQ's quad-double projection put back in eight
parts). Under --jet_o 1 it starts from the octo-double prediction in all eight parts, its horizon data re-seeded. The
base is polished from all its stored parts (eight from a state saved by it), with --reseed 2 inside ElimO at the
base's D (the quad-double polish runs without the projection). A base polish that neither converges nor lowers its
max |F| is not adopted: the stored state stays the base, and the line "base polish not adopted" says so (#461, where
a polish took the D 215.03125 state from 4.4e-36 to 1.5e-18 and the series of that base had |z_N| 1e-10 against
2e-30). --base_adopt 1 adopts every polish, which is the control of that rule. --base_polish 0 skips the polish, so
the stored state is the base as it stands: a state reloaded near D 215 starts about 4e-36 off in the held rows
(their values are taken again from its first two parts), and its polish costs 12 octo-double steps before it is
dropped. Under --corrector od a base whose double-double |F| is at most --tol is still polished when the stored
state's pinned octo-double max |F| exceeds --tol_od, and the line "base: ..." gives both tests (#576). The step: the
first is --a0 (h_max when 0), a refused attempt halves it, and after an accepted step the next is
--a_grow times longer, up to h_max; with --a_hold 1 it grows only after a step accepted at its first attempt, so a
step found by halving is kept (past D 232 steps of 1.0 are refused at every base and 0.5 accepted, #461).
--od_accept natural judges the octo-double corrector's trials by Deuflhard's natural monotonicity test (the simplified
Newton correction through the LU at the iterate, octo_corrector.newton_od) instead of each trial's own Newton
correction (own, the default; #644). --od_strong k takes the corrector's first step whole and then solves only the
strong rows, by the chord through the start's LU with the k weakest right directions of the start's Jacobian
removed from each correction (octo_corrector.chord_strong; #528, #674). 0, the default, is off.
--lam_stall L ends an attempt, as refused, when two Newton steps in a row had lam at or below L. A state is saved
with the plan of the run that converged it, and a run started from it keeps that plan
(--keep_plan 0 plans again at the state, the control). Its states are saved in eight parts
(octo_corrector.save_o: save_q's four, which load_q reads, and four more), and the series of the next base (--jet_o)
is taken at all eight. M = 32 only; --accept_stall_qd does not apply. --od_junk w freezes, in every step of that
corrector, the junk among the w weakest directions of its first Jacobian (octo_corrector.junk_subspace_o: no Delta
content; p1's highest harmonics past D ~ 210, notes/QUAD_DOUBLE.md 13.29), as newton_qd's junk mode does.
--save_base 1 saves the base after its octo-double polish as OUTDIR/base_pin<k>_D<D>.npz (eight parts, its meta
recording the polish), when the polish was adopted; with --steps 0 the polish alone is the run (#489).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402
from hpkernel.driver import make_solver, newton_solve, deflated_newton, delta_correction, newton_qd, qparts, rnorm, save, save_q, load_q  # noqa: E402
from multidouble import dd_add, dd_horner, dd_scale, dd_norm, qd_add_dd  # noqa: E402
from pinning import HeldFit, PinnedQ, PinnedSolver, band_coeffs, band_field, qd_axpy  # noqa: E402
from raw_out import require_raw  # noqa: E402
from octo_corrector import ElimO, PinnedO, delta_digits, has_o, load_o, newton_od, o8, o8_of_q, omax, q4_of_o, save_o, with_horizon  # noqa: E402,F401
from reseed import ElimQ, Reseeder, StartNotProjected  # noqa: E402
from taylor_predict import formula_delta  # noqa: E402


def load_state(path):
    d = np.load(path)
    m = json.loads(str(d["meta"]))
    return d["zh"].astype(float), d["zl"].astype(float), m


# D is carried in mpmath precision: the solver forms n = D - 2 in double-double from it, and at fixed data the
# residual changes by about 2e6 per unit of D (13.16), so the float64 rounding of D + a (7e-15 at D 73) would put
# 1e-8 into every prediction (found on the pilot's first run, 2026-11-04 00:10 UTC)
mp.mp.dps = 50
# 1/3 in double-double. The float64 weights 4/3 and -1/3 are each 2^-54 below their values, so a Richardson value formed
# with them is (1 - 2^-54) times the true one, a 5.6e-17 relative error in every row.
THIRD = (1.0 / 3.0, float(mp.mpf(1) / 3 - mp.mpf(1.0 / 3.0)))


def richardson(r1, r2):
    """(4 r2 - r1) / 3 in double-double, r1 at step e and r2 at e / 2: the central differences' e^2 term cancelled"""
    th, tl = dd_add(*dd_scale(np.asarray(r2[0]), np.asarray(r2[1]), 4.0), -np.asarray(r1[0]), -np.asarray(r1[1]))
    return dd_add(*dd_scale(th, tl, THIRD[0]), *dd_scale(th, tl, THIRD[1]))


def series_at(zh, zl, Zh, Zl, a):
    """z0 + sum_k z_k a^k by Horner's rule in double-double (a float64; multidouble's dd_horner, bit for bit the
    dd_add / dd_scale loop it replaced): each product rounds to double-double, where float64 powers a**k would carry
    1e-16 of themselves into the prediction unless a is a power of 2"""
    return dd_horner([zh] + list(Zh[1:]), [zl] + list(Zl[1:]), a)


def coefficients(s, zh, zl, N, held=None, refine_it=1, z1=None):
    """z_1..z_N from the stored LU at (zh, zl): [(zh_k, zl_k)], norms, the refinement residuals. held = (fn, rows):
    fn(k) gives the held harmonics' order-k Taylor coefficients as a field (hi, lo) on the horizon block's p rows (the
    slice rows), added to the right-hand side there as PinnedSolver's offset is, so that z_k carries their motion.
    refine_it: refinement steps per order; refine gets the residual norms before and after each. z1 = (hi, lo): z_1
    given instead of solved (its residual is still recorded)"""
    Zh, Zl, norms, refine = [zh], [zl], [float(dd_norm(zh, zl))], []
    for k in range(1, N + 1):
        Hs, Ls = s.residual_jet(Zh, Zl, k + 1, 0)
        bh, bl = -np.asarray(Hs[k]), -np.asarray(Ls[k])
        if held is not None:
            fn, rows = held
            th, tl = fn(k)
            bh, bl = bh.copy(), bl.copy()
            ah, al = dd_add(bh[rows], bl[rows], np.asarray(th), np.asarray(tl))
            bh[rows], bl[rows] = ah, al
        res = []
        if k == 1 and z1 is not None:
            xh, xl = np.asarray(z1[0]), np.asarray(z1[1])
        else:
            xh, xl = s.lu_solve(bh, bl)
            for _ in range(refine_it):
                jh, jl = s.jvp(zh, zl, xh, xl)[:2]
                r0h, r0l = dd_add(bh, bl, -jh, -jl)
                res.append(float(dd_norm(r0h, r0l)))
                ch, cl = s.lu_solve(r0h, r0l)
                xh, xl = dd_add(xh, xl, ch, cl)
        jh, jl = s.jvp(zh, zl, xh, xl)[:2]
        r1h, r1l = dd_add(bh, bl, -jh, -jl)
        res.append(float(dd_norm(r1h, r1l)))
        Zh.append(xh); Zl.append(xl)
        norms.append(float(dd_norm(xh, xl)))
        refine.append(tuple(res))
    return Zh, Zl, norms, refine


def coefficients_q(s, zq, N, held=None):
    """z_1..z_N by the quad-double jet (Solver.residual_jet_q) and the quad-double LU of the last Solver.newton_q, which
    must have been taken at zq (the base's four parts): the list of four-part z_k (z_0 first) and their norms. held as
    in coefficients, added in quad-double. No refinement: the LU is the exact Jacobian's in quad-double."""
    Zq, norms = [tuple(np.ascontiguousarray(np.asarray(p)) for p in zq)], [float(dd_norm(zq[0], zq[1]))]
    for k in range(1, N + 1):
        F = s.residual_jet_q(Zq, k + 1)
        b = [-np.asarray(p) for p in F[k]]
        if held is not None:
            fn, rows = held
            th, tl = fn(k)
            r = qd_add_dd(*(np.ascontiguousarray(p[rows]) for p in b), np.asarray(th), np.asarray(tl))
            for p, v in zip(b, r):
                p[rows] = v
        x = s.qlu_solve(*(np.ascontiguousarray(p) for p in b))
        Zq.append(tuple(np.ascontiguousarray(np.asarray(p)) for p in x))
        norms.append(float(dd_norm(Zq[-1][0], Zq[-1][1])))
    return Zq, norms


def coefficients_o(s, zo, N, held=None):
    """z_1..z_N by the octo-double jet (Solver.residual_jet_o) and the octo-double LU of the last Solver.newton_o, which
    must have been taken at zo (eight parts): the list of eight-part z_k (z_0 first) and their norms. held as in
    coefficients, added in octo-double (od_add_dd). No refinement: the LU is the exact Jacobian's in octo-double."""
    # the octo-double helpers are in multidouble's Python package from f092018 on; imported here so that the other
    # paths run on the older packages the runners have
    from multidouble import od_add_dd, od_axpy
    Zo, norms = [o8(zo)], [float(dd_norm(zo[0], zo[1]))]
    for k in range(1, N + 1):
        F = s.residual_jet_o(list(Zo), k + 1)
        b = [-np.asarray(p) for p in F[k]]
        if held is not None:
            fn, rows = held
            t = fn(k)
            if len(t) == 2:
                r = od_add_dd(*(np.ascontiguousarray(p[rows]) for p in b), np.asarray(t[0]), np.asarray(t[1]))
            else:
                # the held field's order-k coefficient in eight parts (--held_o 1)
                r = od_axpy(1.0, *(np.ascontiguousarray(np.asarray(v)) for v in t), *(np.ascontiguousarray(p[rows]) for p in b))
            for p, v in zip(b, r):
                p[rows] = v
        Zo.append(o8(s.olu_solve([np.ascontiguousarray(p) for p in b])))
        norms.append(float(dd_norm(Zo[-1][0], Zo[-1][1])))
    return Zo, norms


def horner_o(Zo, a):
    """z_0 + sum_k z_k a^k by Horner's rule in octo-double (od_axpy, y + a x), a a float64"""
    from multidouble import od_axpy
    p = Zo[-1]
    for k in range(len(Zo) - 2, -1, -1):
        p = o8(od_axpy(a, *p, *Zo[k]))
    return p


def main():
    args = list(sys.argv[1:])
    opt = dict(state="T75/pin7_D73.2657.npz", kfreeze=3, N=24, eps=1e-16, h_max=2.0, steps=20, D_max=200.0, corrector="dnp",
               tol=1e-20, tol_growth=1.45, accept_stall=1e-17, junk_only=0, threads=4, w=24, trend="", tol_qd=1e-28, accept_stall_qd=0.0, qd_it=12, qd_w=0,
               refine=1, probe=0, probe_c=0.0, plan_tol=1e-32, fix1=0, F_pred_max=0.0, horizon_alpha=0, fix1_qlu=0, jet_q=0, reseed=0, pred_q=0, reseed_q=0,
               save_pred=0, jet_o=0, save_dz0=0)
    opt.update(tol_od=1e-40, od_junk=0)  # --corrector od (hp/octo_corrector.py); --od_junk w freezes junk (QUAD_DOUBLE 13.29)
    opt.update(base_adopt=0)            # --base_adopt 1: the base's od polish is adopted whatever it did
    opt.update(base_polish=1)           # --base_polish 0: no polish, the stored state is the base
    opt.update(keep_plan=1)             # --keep_plan 0: plan the base again even when its state stores a plan
    opt.update(a0=0.0, a_grow=2.0, a_hold=0, lam_stall=0.0)    # the step's start, growth and hold; the corrector's stall rule
    opt.update(od_accept="own")                                 # the octo-double corrector's trial test (#644)
    opt.update(od_strong=0)             # --od_strong k: the strong-row chord after the first step (#674)
    opt.update(held_fit="")             # --held_fit FILE (pinning.HeldFit; notes/HELD_HARMONICS.md)
    # --held_o 1: the held values and their Taylor coefficients as mpf instead of floats, and the held field in eight
    # parts instead of two (with --held_fit and --corrector od); --reseed_o K: K octo-double corrections of the
    # re-seeded horizon data (reseed.Reseeder.refine_o)
    opt.update(held_o=0, reseed_o=0)
    opt.update(save_base=0)             # --save_base 1: the polished base saved (with --corrector od)
    for k in list(opt):
        f = "--" + k
        if f in args:
            i = args.index(f); opt[k] = type(opt[k])(args[i + 1]); del args[i:i + 2]
    raw, out = args[0], args[1]
    if opt["pred_q"] and (not opt["jet_q"] or (opt["reseed"] and not opt["reseed_q"])):
        raise SystemExit("--pred_q needs --jet_q 1, and with --reseed also --reseed_q 1")
    if opt["jet_o"] and (opt["corrector"] not in ("qd", "od") or opt["jet_q"] or opt["pred_q"] or opt["fix1"] or opt["probe"]
                         or (opt["reseed"] and not opt["reseed_q"])):
        raise SystemExit("--jet_o 1 needs --corrector qd or od, and with --reseed also --reseed_q 1; it replaces --jet_q, --pred_q "
                         "and --fix1, and --probe does not apply")
    out_dir = out
    os.makedirs(out, exist_ok=True)
    rec_path = os.path.join(out, "anm_run.jsonl")
    require_raw(rec_path)
    zh, zl, m = load_state(os.path.join(raw, opt["state"]))
    D, M, nodes = (mp.mpf(m["D_str"]) if "D_str" in m else mp.mpf(float(m["D"]))), int(m["M"]), list(m["nodes"])
    kt, kc, ks = m.get("kt", 36), m.get("kc", 146), m.get("ks", 130)
    kf, N = opt["kfreeze"], opt["N"]
    if opt["jet_o"] and (M != 32 or not 1 <= N <= 31):
        raise SystemExit(f"--jet_o 1: the octo-double solver is built for M = 32 and N <= 31, not M {M}, N {N}")
    if opt["corrector"] == "od" and (M != 32 or (opt["reseed"] and not opt["reseed_q"])):
        raise SystemExit(f"--corrector od: the octo-double solver is built for M = 32 (not {M}), and with --reseed it needs --reseed_q 1")
    D_start = D
    junk_only = bool(opt["junk_only"])

    def tol_at(Dx):
        return opt["tol"] * opt["tol_growth"] ** float(Dx - D_start)

    last_q = {}
    elim = {}             # --reseed 2: the attempt's Reseeder, for ElimQ in the corrector
    dz0_at = {}           # the attempt in progress (its target D and step), for --save_dz0

    def save_dz0(z, d):
        """--save_dz0 1: the corrector's start and first Newton step of the attempt, eight parts each (#499)"""
        name = f"dz0_pin{kf}_D{dz0_at.get('D', 0.0):.4f}_a{dz0_at.get('a', 0.0):.6f}.npz"
        np.savez(os.path.join(out, name), **{f"z{q}": np.asarray(z[q]) for q in range(8)},
                 **{f"d{q}": np.asarray(d[q]) for q in range(8)},
                 meta=json.dumps(dict(D=dz0_at.get("D"), D_str=dz0_at.get("D_str"), a=dz0_at.get("a"), M=M, nodes=nodes, kfreeze=kf, source=opt["state"])))
        print(f"      saved the start and first step as {name}", flush=True)

    def correct(spx, ph, pl, tol, z4_start=None, z8_start=None):
        """T75's corrector (scan_hp.py dnp): deflated Newton, then plain Newton steps from below 1e-12 if they lower |F|;
        returns zh, zl, |F|, steps, accepted (converged, or stalled below accept_stall). z4_start: the quad-double
        corrector's start in four parts (--pred_q), else (ph, pl). z8_start: the octo-double corrector's in eight"""
        if opt["corrector"] == "od":
            # --corrector od: the octo-double Newton (hp/octo_corrector.py), with --reseed 2 inside ElimO
            so = PinnedO(spx.s, spx.ch, spx.cl, c8=spx.c8)
            if elim.get("rs") is not None:
                so = ElimO(so, elim["rs"])
            z0 = z8_start if z8_start is not None else o8_of_q(z4_start if z4_start is not None else qparts(ph, pl))
            try:
                z8, hist, conv = newton_od(so, z0, tol=opt["tol_od"], it_max=opt["qd_it"], verbose=True, junk_w=opt["od_junk"],
                                           lam_stall=opt["lam_stall"], on_first=save_dz0 if opt["save_dz0"] else None,
                                           accept=opt["od_accept"], strong_chord=int(opt["od_strong"]))
            except StartNotProjected as e:
                print(f"      elim projections {e.args[0]}: the start's projection failed, attempt stopped", flush=True)
                Fh, Fl = spx.residual(z0[0], z0[1])
                return np.asarray(z0[0]), np.asarray(z0[1]), float(rnorm(Fh, Fl)), 0, False, 0.0
            if elim.get("rs") is not None:
                print(f"      elim projections {so.counts}", flush=True)
                z8 = so.project(z8)             # the point whose residual converged (unprojected below the switch)
            last_q["z8"], last_q["z4"] = z8, q4_of_o(z8)
            Fh, Fl = spx.residual(z8[0], z8[1])
            Fc = float(rnorm(Fh, Fl))
            fo = float(hist[-1]["F"]) if hist else float("nan")
            # max |F| at the start and after the last step, for the base's adoption rule
            last_q["od_F"] = (float(hist[0]["F"]), float(hist[-1].get("F_new", hist[-1]["F"]))) if hist else None
            print(f"      octo-double Newton: {len(hist)} steps, max|F| {fo:.1e}, {'converged' if conv else 'not converged'}; double-double |F| "
                  f"of its first two parts {Fc:.1e}", flush=True)
            return np.asarray(z8[0]), np.asarray(z8[1]), Fc, len(hist), bool(conv), 0.0
        if opt["corrector"] == "qd":
            sq = PinnedQ(spx.s, spx.ch, spx.cl)
            if elim.get("rs") is not None:
                # --reseed 2: the corrector on the system with the horizon data eliminated (hp/reseed.py ElimQ)
                sq = ElimQ(sq, elim["rs"])
            z0 = z4_start if z4_start is not None else qparts(ph, pl)
            try:
                z4, hist, conv = newton_qd(sq, z0, tol=opt["tol_qd"], it_max=opt["qd_it"], w=opt["qd_w"], verbose=True)
            except StartNotProjected as e:
                # the start's projection failed, so the attempt stops and the step is halved (hp/reseed.py ElimQ)
                print(f"      elim projections {e.args[0]}: the start's projection failed, attempt stopped", flush=True)
                Fh, Fl = spx.residual(z0[0], z0[1])
                return np.asarray(z0[0]), np.asarray(z0[1]), float(rnorm(Fh, Fl)), 0, False, 0.0
            if elim.get("rs") is not None:
                print(f"      elim projections {sq.counts}", flush=True)
                # ElimQ's residual is that of the point's projection, so the solution is the projected iterate (the
                # iterate itself is off the root by its horizon data's last step, 2.9e-10 in |F| at 153.43, 13.26)
                z4 = elim["rs"].project(z4)[0]
            last_q["z4"] = z4
            Fh, Fl = spx.residual(z4[0], z4[1])
            Fc = float(rnorm(Fh, Fl))
            fq = float(hist[-1]["F"]) if hist else float("nan")
            if not conv and opt["accept_stall_qd"] > 0 and np.isfinite(fq) and fq <= opt["accept_stall_qd"]:
                # --accept_stall_qd: a stall at the quad-double floor, which rises past tol_qd near D 194 (1.1e-28 there,
                # the iterates moving Delta by ~1e-19), is accepted below this level (notes/QUAD_DOUBLE.md 13.28)
                print(f"      quad-double stall at max|F| {fq:.1e} accepted (accept_stall_qd {opt['accept_stall_qd']:.0e})", flush=True)
                conv = True
            print(f"      quad-double Newton: {len(hist)} steps, max|F| {fq:.1e}, {'converged' if conv else 'not converged'}; double-double |F| of its "
                  f"first two parts {Fc:.1e}", flush=True)
            return np.asarray(z4[0]), np.asarray(z4[1]), Fc, len(hist), bool(conv), 0.0
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
    mk = lambda Dx: make_solver(Dx, M, nodes=nodes, kt=kt, kc=kc, ks=ks, threads=opt["threads"], kfreeze=kf,  # noqa: E731
                                horizon_alpha=opt["horizon_alpha"])
    ks_held = [k for k in range(kf + 1, 14) if k % 2 == 1]
    if opt["held_o"] and not (opt["held_fit"] and opt["corrector"] == "od" and opt["jet_o"]):
        raise SystemExit("--held_o 1 needs --held_fit, --corrector od and --jet_o 1")
    held_x = bool(opt["held_o"])
    hparts = 8 if held_x else 2

    def coeffs_of(zh_, zl_):
        """[(a_k, b_k)] of the base's held harmonics, exact there: floats, or with --held_o mpf from all the parts the
        base has (the corrector's eight, or the stored state's)"""
        rest = ()
        if held_x:
            z8b = last_q.get("z8")
            if z8b is None and has_o(os.path.join(raw, opt["state"])):
                z8b = load_o(os.path.join(raw, opt["state"]))[0]
            if z8b is not None:
                rest = [np.asarray(p)[:M] for p in z8b[2:]]
        return band_coeffs(zh_[:M], zl_[:M], M, ks_held, exact=held_x, rest=rest)
    base_coeffs = coeffs_of(zh, zl)
    # the trend: exponents s of |v| ~ n^s per held coefficient, from the stored states of --trend (0 = held constant)
    slopes = [[0.0, 0.0] for _ in ks_held]
    if opt["trend"]:
        import glob
        ns, vals = [], []
        for f in sorted(glob.glob(os.path.join(raw, opt["trend"], "pin*_D*.npz")))[::4]:
            dd = np.load(f); mm = json.loads(str(dd["meta"]))
            if int(mm["M"]) != M:
                continue
            ns.append(float(mm["D"]) - 2.0)
            vals.append(band_coeffs(dd["zh"][:M], dd["zl"][:M], M, ks_held))
        ns = np.array(ns)
        for i, k in enumerate(ks_held):
            for j in (0, 1):
                v = np.array([c[i][j] for c in vals])
                if len(v) >= 6 and (np.all(v > 0) or np.all(v < 0)):
                    sl, c0 = np.polyfit(np.log(ns), np.log(np.abs(v)), 1)
                    res = float(np.max(np.abs(np.log(np.abs(v)) - (c0 + sl * np.log(ns)))))
                    if res < 0.05:
                        slopes[i][j] = float(sl)
        print("trend exponents (a, b) per held harmonic " + ", ".join(f"{k}: {sa:+.2f}/{sb:+.2f}" for k, (sa, sb) in zip(ks_held, slopes)), flush=True)
    n0 = float(D) - 2.0
    # --held_fit FILE: the held values' increments from a fit of the full-system states (pinning.HeldFit), in place
    # of the trend; relative to the base as the trend is (notes/HELD_HARMONICS.md)
    hfit = HeldFit(opt["held_fit"]) if opt["held_fit"] else None
    hfit_jets = {}
    if hfit is not None:
        if opt["trend"]:
            raise SystemExit("--held_fit replaces --trend; give one of them")
        print(f"held values from the fit {opt['held_fit']} (degree {hfit.meta.get('deg')}, {hfit.meta.get('n_points')} full-system states, "
              f"D {hfit.meta.get('Dmin')}-{hfit.meta.get('Dmax')}), relative to the base", flush=True)

    def held_at(Dx):
        """the held coefficients at D = Dx on the trend, relative to the base"""
        if hfit is not None:
            # a float here is half an ulp from the fit's value, 1e-18 on the largest held coefficient, which the
            # corrector's first step then answers with a move 1 / (smallest pivot) times larger (#461)
            num = (lambda v: v) if held_x else float
            return [(num(mp.mpf(a) + da), num(mp.mpf(b) + db)) for (a, b), (da, db) in zip(base_coeffs, hfit.increment(D, Dx, ks_held))]
        r = float(Dx) - 2.0
        return [(a * (r / n0) ** sa, b * (r / n0) ** sb) for (a, b), (sa, sb) in zip(base_coeffs, slopes)]

    def held_jet(j):
        """the order-j Taylor coefficients in h of the held values, v(D0) binom(s, j) n0^-j"""
        if hfit is not None:
            key = str(D)
            if key not in hfit_jets:
                hfit_jets.clear()
                hfit_jets[key] = hfit.jets(D, N, ks_held, exact=held_x)
            return hfit_jets[key][j]
        out = []
        for (a, b), (sa, sb) in zip(base_coeffs, slopes):
            ca = math.comb if False else None  # noqa: F841
            def binom(sx, jj):
                r = 1.0
                for t in range(jj):
                    r *= (sx - t) / (t + 1)
                return r
            out.append((a * binom(sa, j) / n0 ** j, b * binom(sb, j) / n0 ** j))
        return out

    def wrap_at(sx, Dx):
        h = band_field(held_at(Dx), ks_held, M, parts=hparts)
        return PinnedSolver(sx, M, h[0], h[1], h_rest=h[2:])
    wrap = lambda sx: wrap_at(sx, D)  # noqa: E731
    # a state saved with its plan keeps it. A run plans once, at its first base (a prediction is planned again only
    # when its double-double |F| is below 1e-6, which it is not past D ~ 130), so a restart that plans again changes
    # the steps: that leaves the state 5e-36 off its own equations at D 226, and past D 230 the weak direction turns
    # it into a first correction of 1e-15 (#461)
    segs = m.get("segs") if opt["keep_plan"] else None
    if segs:
        print(f"the base keeps its stored plan ({sum(len(sg) - 1 for sg in segs)} steps)", flush=True)
    else:
        segs = None
    s = mk(D)
    if not segs:
        segs = s.plan(zh, zl, opt["plan_tol"], 0.05)
    s.set_segs(segs); sp = wrap(s)
    Kb = (len(zh) - 1) // (4 * M) - 1                                   # the horizon block
    rows = slice(Kb * 4 * M, Kb * 4 * M + M)
    # the held harmonics as directions to freeze in the quad-double Newton: unit band fields in the p1 block
    Fh, Fl = sp.residual(zh, zl)
    F0 = float(rnorm(Fh, Fl))
    need = F0 > opt["tol"]
    if opt["corrector"] == "od" and opt["base_polish"] and not need:
        # a start converged in double-double can be far off in octo-double: the gate is then the stored state's pinned
        # octo-double max |F| against --tol_od (#576)
        Fo = omax(PinnedO(sp.s, sp.ch, sp.cl, c8=sp.c8).residual_o(load_o(os.path.join(raw, opt["state"]))[0]))
        need = Fo > opt["tol_od"]
        print(f"base: double-double |F| {F0:.1e} <= --tol {opt['tol']:.0e}, octo-double max|F| {Fo:.1e} "
              f"{'>' if need else '<='} --tol_od {opt['tol_od']:.0e}: base polish {'runs' if need else 'skipped'}", flush=True)
    if need and not opt["base_polish"]:
        print("base polish skipped (--base_polish 0): the stored state is the base", flush=True)
    elif need:
        # the quad-double polish starts from all four stored parts: from the first two alone it starts eps_dd G off the
        # root, outside the basin past D ~ 131 (the restart of T113 at 135.06, notes/QUAD_DOUBLE.md 13.22)
        z4_base = tuple(np.ascontiguousarray(np.asarray(p)) for p in load_q(os.path.join(raw, opt["state"]))[0])
        if opt["corrector"] == "od":
            # --corrector od: the octo-double polish from all eight stored parts (a four-part state's promoted), with
            # --reseed 2 inside ElimO: from a quad-double state at D 188 a full step leaves 4e-14 in the horizon rows
            # alone (its quadratic term there; 1e-41 in the others), which the projection removes
            if opt["reseed"] == 2:
                elim["rs"] = Reseeder(dict(M=M, nodes=nodes, kt=kt, kc=kc, ks=ks), D, threads=opt["threads"],
                                      horizon_alpha=opt["horizon_alpha"], qd=bool(opt["reseed_q"]), od=opt["reseed_o"])
            zh_p, zl_p, F0_p, _, conv_p, _ = correct(sp, zh, zl, opt["tol"],
                                                     z8_start=load_o(os.path.join(raw, opt["state"]))[0])
            elim["rs"] = None
            Fs = last_q.get("od_F")
            if conv_p or Fs is None or Fs[1] < Fs[0] or opt["base_adopt"]:
                zh, zl, F0 = zh_p, zl_p, F0_p
                if opt["save_base"] and "z8" in last_q:
                    p = os.path.join(out, f"base_pin{kf}_D{float(D):.4f}.npz")
                    save_o(p, last_q["z8"], dict(D=float(D), D_str=str(D), M=M, nodes=nodes, kt=kt, kc=kc, ks=ks, kfreeze=kf,
                                                 pin_held=ks_held, N=N, source=opt["state"], parts=8,
                                                 segs=[[float(x) for x in sg] for sg in segs],
                                                 base_polish=dict(converged=bool(conv_p), max_F=Fs, tol_od=opt["tol_od"])))
                    print(f"base polish saved as {p}", flush=True)
            else:
                # the stored state stays the base: the series and the first prediction load it again
                last_q.pop("z8", None)
                last_q.pop("z4", None)
                print(f"base polish not adopted (max|F| {Fs[0]:.1e} -> {Fs[1]:.1e}): the stored state is the base", flush=True)
        else:
            zh, zl, F0, _, _, _ = correct(sp, zh, zl, opt["tol"], z4_base if opt["corrector"] == "qd" else None)
    print(f"base {opt['state']}: D {mp.nstr(D, 12)}, M {M}, pin {kf} (held {ks_held}), |F| {F0:.1e}; N {N}, eps {opt['eps']}, h_max {opt['h_max']}, "
          f"corrector {opt['corrector']} (junk_only {junk_only}), tol {opt['tol']:.1e} x {opt['tol_growth']}^(D - {mp.nstr(D_start, 8)}), "
          f"stalls accepted below {opt['accept_stall']:.0e}", flush=True)
    def predicted(D0, zh0, zl0, Zh_, Zl_, a_, powers=False):
        """the series' prediction at D0 + a_ and its max |F| in double-double and quad-double, as an attempt forms it
        (powers: the sum with float64 powers a_**k instead of Horner's rule, the control)"""
        if powers:
            ph, pl = np.array(zh0), np.array(zl0)
            for k in range(1, N + 1):
                ph, pl = dd_add(ph, pl, *dd_scale(Zh_[k], Zl_[k], a_ ** k))
        else:
            ph, pl = series_at(zh0, zl0, Zh_, Zl_, a_)
        Dn = D0 + mp.mpf(a_)
        sn = mk(Dn)
        sn.set_segs(segs)
        spn = wrap_at(sn, Dn)
        Fh, Fl = spn.residual(ph, pl)
        if float(rnorm(Fh, Fl)) <= 1e-6:
            sn.set_segs(sn.plan(ph, pl, 1e-32, 0.05))
            Fh, Fl = spn.residual(ph, pl)
        Fq = PinnedQ(sn, spn.ch, spn.cl).residual_q(*qparts(ph, pl))
        return float(rnorm(Fh, Fl)), float(rnorm(Fq[0], Fq[1]))

    def order1_q(D0, zq, z1h, z1l, e):
        """the pinned quad-double model's order-1 residual of z_1: central differences of the kernel's residual along
        (z_1, 1) in (z, D) at step e (a power of 2, so z +- e z_1 and D +- e are exact), minus the held slope on its rows"""
        Ks = []
        for sg in (1.0, -1.0):
            Dx = D0 + sg * mp.mpf(e)
            sx = mk(Dx)
            sx.set_segs(segs)
            Ks.append(sx.residual_q(*qd_axpy(sg * e, *qparts(z1h, z1l), *zq)))
        d = qd_axpy(-1.0, *(np.ascontiguousarray(np.asarray(p)) for p in Ks[1]), *(np.ascontiguousarray(np.asarray(p)) for p in Ks[0]))
        rh, rl = (np.array(p) for p in dd_scale(np.asarray(d[0]), np.asarray(d[1]), 0.5 / e))
        th, tl = band_field(held_jet(1), ks_held, M)
        rh[rows], rl[rows] = dd_add(rh[rows], rl[rows], -np.asarray(th), -np.asarray(tl))
        return rh, rl

    def order1_r(D0, zq, z1h, z1l):
        """order1_q at steps 2^-29 and 2^-30 combined as (4 r(e/2) - r(e)) / 3, which cancels the e^2 term (the step
        2^-33 alone left 2.1e-14 at D 82.16, 13.19)"""
        r1, r2 = order1_q(D0, zq, z1h, z1l, 2.0 ** -29), order1_q(D0, zq, z1h, z1l, 2.0 ** -30)
        return richardson(r1, r2)

    def fix_z1(D0, zq, z1h, z1l, passes=2):
        """z_1 corrected by the quad-double order-1 residual (order1_r) through the stored LU; returns z_1 and the
        residual norms before and after each pass. --fix1_qlu 1: through the quad-double LU at the base instead
        (Solver.newton_q factors it, once per step), whose contraction stays at 1e-10 to 1e-13 from D 79.8 to 86.4
        where the double-double LU's grows from 7e-7 to 7e-5, about 2.4x per unit of D"""
        x = order1_r(D0, zq, z1h, z1l)
        hist = [float(rnorm(*x))]
        if opt["fix1_qlu"]:
            s.newton_q(*zq)
            zero = np.zeros(len(z1h))
        for _ in range(passes):
            if opt["fix1_qlu"]:
                q = s.qlu_solve(-np.asarray(x[0]), -np.asarray(x[1]), zero, zero)
                ch, cl = np.asarray(q[0]), np.asarray(q[1])
            else:
                ch, cl = s.lu_solve(-np.asarray(x[0]), -np.asarray(x[1]))
            z1h, z1l = dd_add(z1h, z1l, ch, cl)
            x = order1_r(D0, zq, z1h, z1l)
            hist.append(float(rnorm(*x)))
        return np.asarray(z1h), np.asarray(z1l), hist

    def probe_links(D0, zh0, zl0, Zh, Zl, zq):
        """--probe 2: which link of the order-1 equation J z_1 = -dF/dD carries the error. The jet's order-1
        coefficient at fixed z (dF/dD) against quad-double central differences in D at fixed z, and the jvp along z_1
        against central differences along z_1 at fixed D, both Richardson-combined at steps e and e/2 and printed at
        two e, with the row of the largest difference and its block (the held offset enters neither)"""
        c4 = lambda v: tuple(np.ascontiguousarray(np.asarray(p)) for p in v)  # noqa: E731
        Hs, Ls = s.residual_jet([np.array(zh0)], [np.array(zl0)], 2, 0)
        jet = (np.asarray(Hs[1]), np.asarray(Ls[1]))
        jv = tuple(np.asarray(v) for v in s.jvp(zh0, zl0, np.array(Zh[1]), np.array(Zl[1]))[:2])

        def central(e, along_z):
            Ks = []
            for sg in (1.0, -1.0):
                sx = mk(D0 if along_z else D0 + sg * mp.mpf(e))
                sx.set_segs(segs)
                zz = c4(qd_axpy(sg * e, *qparts(Zh[1], Zl[1]), *zq)) if along_z else zq
                Ks.append(sx.residual_q(*zz))
            d = qd_axpy(-1.0, *c4(Ks[1]), *c4(Ks[0]))
            return dd_scale(np.asarray(d[0]), np.asarray(d[1]), 0.5 / e)

        out = dict(D0=float(D0), links=[])
        diffs = {}
        for name, ref, along_z in (("dF/dD", jet, False), ("J z_1", jv, True)):
            for e in (2.0 ** -42, 2.0 ** -45):
                r1, r2 = central(e, along_z), central(e / 2, along_z)
                rr = richardson(r1, r2)
                dh, dl = dd_add(ref[0], ref[1], -np.asarray(rr[0]), -np.asarray(rr[1]))
                dv = np.abs(np.asarray(dh) + np.asarray(dl))
                i = int(np.argmax(dv))
                rec = dict(link=name, e=e, fd=float(rnorm(*rr)), jet=float(rnorm(*ref)), diff=float(dv[i]), row=i, block=i // (4 * M))
                out["links"].append(rec)
                print(f"  probe: {name}, step {e:.2e}: |finite difference| {rec['fd']:.4e}, |jet| {rec['jet']:.4e}, "
                      f"max |jet - fd| {rec['diff']:.4e} at row {i} (block {rec['block']})", flush=True)
                diffs[name] = (np.asarray(dh), np.asarray(dl), np.asarray(ref[0]) + np.asarray(ref[1]))
        # the order-1 residual's error, jet (dF/dD + J z_1) minus finite differences, and each link's share by row
        sh, sl = dd_add(*diffs["dF/dD"][:2], *diffs["J z_1"][:2])
        sv = np.abs(np.asarray(sh) + np.asarray(sl))
        top = [int(i) for i in np.argsort(sv)[::-1][:6]]
        out["order1_error"] = float(sv.max())
        out["order1_rows"] = [dict(row=i, block=i // (4 * M), field=(i % (4 * M)) // M, sum=float(sv[i]),
                                   dFdD=float(diffs["dF/dD"][0][i] + diffs["dF/dD"][1][i]), Jz1=float(diffs["J z_1"][0][i] + diffs["J z_1"][1][i]),
                                   dFdD_value=float(diffs["dF/dD"][2][i])) for i in top]
        print(f"  probe: order-1 residual, jet minus finite differences: max {sv.max():.4e}", flush=True)
        for r in out["order1_rows"]:
            print(f"  probe:   row {r['row']} (block {r['block']}, field {r['field']}): sum {r['sum']:.3e}, dF/dD link {r['dFdD']:+.3e}, "
                  f"J z_1 link {r['Jz1']:+.3e}, dF/dD there {r['dFdD_value']:+.3e}", flush=True)
        with open(os.path.join(out_dir, "probe.jsonl"), "a") as f:
            f.write(json.dumps(out) + "\n")
        return 0

    def probe(D0, zh0, zl0, Zh, Zl, refine, held_arg):
        """--probe (the docstring): prints, records OUTDIR/probe.jsonl, returns the exit code"""
        zq = last_q["z4"] if "z4" in last_q else load_q(os.path.join(raw, opt["state"]))[0]
        zq = tuple(np.ascontiguousarray(np.asarray(p)) for p in zq)
        if opt["probe"] == 2:
            return probe_links(D0, zh0, zl0, Zh, Zl, zq)
        out = dict(D0=float(D0), D0_str=str(D0), refine_it=opt["refine"], refine_orders_1_2_N=[refine[0], refine[1], refine[-1]])
        print(f"  probe: jet-model residual norms, order 1 {refine[0]}, order 2 {refine[1]}, order N {refine[-1]}", flush=True)
        out["pred"] = []
        for a_ in (1 / 64, 1 / 32, 1 / 16):
            Fd, Fq = predicted(D0, zh0, zl0, Zh, Zl, a_)
            out["pred"].append(dict(a=a_, Fd=Fd, Fq=Fq))
            print(f"  probe: a {a_:.6g}: |F_pred| dd {Fd:.4e} ({Fd / a_:.4e} per unit a), qd {Fq:.4e} ({Fq / a_:.4e})", flush=True)
        r1 = {}
        for e in (2.0 ** -30, 2.0 ** -33):
            r1[e] = order1_q(D0, zq, Zh[1], Zl[1], e)
            print(f"  probe: quad-double order-1 residual of z_1, central step {e:.3e}: max {float(rnorm(*r1[e])):.4e}", flush=True)
        dr = dd_add(r1[2.0 ** -30][0], r1[2.0 ** -30][1], -r1[2.0 ** -33][0], -r1[2.0 ** -33][1])
        out["order1_q"] = [float(rnorm(*r1[e])) for e in r1]
        out["order1_q_step_diff"] = float(rnorm(*dr))
        print(f"  probe: the two steps differ by {out['order1_q_step_diff']:.3e}", flush=True)
        z1h, z1l, x = np.array(Zh[1]), np.array(Zl[1]), r1[2.0 ** -33]
        out["corrected"] = []
        for it in range(2):
            ch, cl = s.lu_solve(-x[0], -x[1])
            z1h, z1l = dd_add(z1h, z1l, ch, cl)
            x = order1_q(D0, zq, z1h, z1l, 2.0 ** -33)
            out["corrected"].append(dict(dz1=float(dd_norm(ch, cl)), order1_q=float(rnorm(*x))))
            print(f"  probe: z_1 corrected ({it + 1}): |dz_1| {out['corrected'][-1]['dz1']:.3e}, quad-double order-1 residual "
                  f"{out['corrected'][-1]['order1_q']:.4e}", flush=True)
        Zc_h, Zc_l, _, refine_c = coefficients(s, zh0, zl0, N, held=held_arg, refine_it=opt["refine"], z1=(z1h, z1l))
        out["pred_corrected"] = []
        for a_ in (1 / 64, 1 / 16, 1 / 4, 1 / 2, 1.0):
            Fd, Fq = predicted(D0, zh0, zl0, Zc_h, Zc_l, a_)
            out["pred_corrected"].append(dict(a=a_, Fd=Fd, Fq=Fq))
            print(f"  probe: corrected series, a {a_:.6g}: |F_pred| dd {Fd:.4e} ({Fd / a_:.4e} per unit a), qd {Fq:.4e} ({Fq / a_:.4e})", flush=True)
        z1h, z1l, out["richardson"] = fix_z1(D0, zq, np.array(Zh[1]), np.array(Zl[1]))
        print(f"  probe: z_1 corrected by the Richardson residual, its norms {out['richardson']}", flush=True)
        Zr_h, Zr_l, _, _ = coefficients(s, zh0, zl0, N, held=held_arg, refine_it=opt["refine"], z1=(z1h, z1l))
        out["pred_richardson"] = []
        for a_ in (1 / 64, 1 / 16, 1 / 4, 1 / 2, 1.0, 1.5, 2.0):
            Fd, Fq = predicted(D0, zh0, zl0, Zr_h, Zr_l, a_)
            out["pred_richardson"].append(dict(a=a_, Fd=Fd, Fq=Fq))
            print(f"  probe: Richardson series, a {a_:.6g}: |F_pred| dd {Fd:.4e} ({Fd / a_:.4e} per unit a), qd {Fq:.4e} ({Fq / a_:.4e})", flush=True)
        out["horner"] = []
        for a_ in (0.4341, 0.1085):
            Fh_, _ = predicted(D0, zh0, zl0, Zr_h, Zr_l, a_)
            Fw_, _ = predicted(D0, zh0, zl0, Zr_h, Zr_l, a_, powers=True)
            out["horner"].append(dict(a=a_, horner=Fh_, powers=Fw_))
            print(f"  probe: Richardson series, a {a_}: |F_pred| by Horner {Fh_:.4e} ({Fh_ / a_ ** 2:.4e} per unit a^2), "
                  f"by float64 powers {Fw_:.4e}", flush=True)
        code = 0
        if opt["probe_c"]:
            c16 = out["pred"][2]["Fd"] * 16
            out["reproduces"] = abs(c16 / opt["probe_c"] - 1) <= 0.01
            print(f"  probe: dd |F_pred| / a at 1/16 {c16:.4e} against the chain's {opt['probe_c']:.4e}: "
                  f"{'reproduced' if out['reproduces'] else 'NOT reproduced'}", flush=True)
            code = 0 if out["reproduces"] else 1
        with open(os.path.join(out_dir, "probe.jsonl"), "a") as f:
            f.write(json.dumps(out) + "\n")
        return code

    a = opt["a0"] or None
    for step in range(opt["steps"]):
        if D >= opt["D_max"]:
            break
        t0 = time.time()
        held_arg = ((lambda j: band_field(held_jet(j), ks_held, M, parts=hparts if opt["jet_o"] else 2), rows)
                    if (any(any(sl) for sl in slopes) or hfit is not None) else None)
        fix1 = None
        Zo = None
        if opt["jet_o"]:
            # --jet_o 1: the series from the octo-double jet through the octo-double LU at the base's four parts
            # promoted to eight
            zq = last_q["z4"] if "z4" in last_q else load_q(os.path.join(raw, opt["state"]))[0]
            zo = o8_of_q(zq)
            if "z8" in last_q or ("z4" not in last_q and has_o(os.path.join(raw, opt["state"]))):
                # the base in eight parts: the octo-double corrector's, or a state it stored
                zo = last_q["z8"] if "z8" in last_q else load_o(os.path.join(raw, opt["state"]))[0]
            pmin, pmax = s.newton_o(list(zo))[3:5]
            Zo, norms = coefficients_o(s, zo, N, held=held_arg)
            Zh, Zl, refine = [np.asarray(z[0]) for z in Zo], [np.asarray(z[1]) for z in Zo], [()]
        elif opt["jet_q"]:
            # --jet_q 1: the series from the quad-double jet through the quad-double LU at the base's four parts
            zq = last_q["z4"] if "z4" in last_q else load_q(os.path.join(raw, opt["state"]))[0]
            zq = tuple(np.ascontiguousarray(np.asarray(p)) for p in zq)
            pmin, pmax = s.newton_q(*zq)[6:8]
            Zq, norms = coefficients_q(s, zq, N, held=held_arg)
            Zh, Zl, refine = [np.asarray(z[0]) for z in Zq], [np.asarray(z[1]) for z in Zq], [()]
        else:
            dh, dl, Fh, Fl, pmin, pmax = sp.newton(zh, zl)
            Zh, Zl, norms, refine = coefficients(s, zh, zl, N, held=held_arg, refine_it=opt["refine"])
        if opt["fix1"] and not opt["probe"] and not opt["jet_q"]:
            zq = last_q["z4"] if "z4" in last_q else load_q(os.path.join(raw, opt["state"]))[0]
            zq = tuple(np.ascontiguousarray(np.asarray(p)) for p in zq)
            z1h, z1l, fix1 = fix_z1(D, zq, np.array(Zh[1]), np.array(Zl[1]))
            Zh, Zl, norms, refine = coefficients(s, zh, zl, N, held=held_arg, refine_it=opt["refine"], z1=(z1h, z1l))
        t_coef = time.time() - t0
        a_max = (opt["eps"] * norms[1] / norms[N]) ** (1.0 / (N - 1)) if norms[N] else opt["h_max"]
        room = float(opt["D_max"] - D)
        a = min(a_max, opt["h_max"], room) if a is None else min(a, a_max, opt["h_max"], room)
        ratios = [norms[k - 1] / norms[k] if norms[k] else None for k in range(2, N + 1)]
        rec = dict(step=step, D0=float(D), D0_str=str(D), pin=kf, N=N, pivots=(float(pmin), float(pmax)), norms=norms, a_max=a_max,
                   t_coef=t_coef, refine_last=refine[-1], fix1=fix1, ratios_2_5=ratios[:4], trend=opt["trend"], slopes=slopes, held=ks_held,
                   horizon_alpha=opt["horizon_alpha"], jet_q=opt["jet_q"], jet_o=opt["jet_o"], held_fit=opt["held_fit"])
        print(f"step {step}: D {mp.nstr(D, 12)}, pivots {pmin:.1e}..{pmax:.1e}, |z_1| {norms[1]:.2e}, |z_N| {norms[N]:.2e}, ratios 2-5 "
              + " ".join(f"{r:.1f}" for r in ratios[:4] if r) + f", a_max {a_max:.3f}, {t_coef:.0f} s", flush=True)
        if opt["probe"]:
            sys.exit(probe(D, zh, zl, Zh, Zl, refine, held_arg))
        converged = False
        for attempt in range(4):
            screens = []
            while True:
                ph, pl = series_at(zh, zl, Zh, Zl, a)
                pred4 = None
                pred8 = None                       # --corrector od: the octo-double prediction, the start
                if opt["pred_q"]:
                    # --pred_q 1: the prediction summed in quad-double from the quad-double series, the base in its four
                    # parts, so the corrector starts from it unrounded (a double-double start carries ~eps_dd G in F)
                    pq = Zq[-1]
                    for k in range(len(Zq) - 2, -1, -1):
                        pq = tuple(np.ascontiguousarray(np.asarray(v)) for v in qd_axpy(a, *pq, *Zq[k]))
                    pred4 = pq
                    ph, pl = np.asarray(pq[0]), np.asarray(pq[1])
                elif opt["jet_o"]:
                    # --jet_o 1: the prediction summed in octo-double from the octo-double series and rounded to
                    # quad-double, the corrector's start
                    pred8 = horner_o(Zo, a)
                    pred4 = q4_of_o(pred8)
                    ph, pl = np.asarray(pred4[0]), np.asarray(pred4[1])
                Dn = D + mp.mpf(a)                 # exact: the series variable is a to double-double precision
                rsd = None
                elim["rs"] = None
                if opt["reseed"]:
                    # --reseed 1: the prediction's horizon data from its last node (hp/reseed.py); 2: also eliminated
                    # inside the corrector
                    rs_n = Reseeder(dict(M=M, nodes=nodes, kt=kt, kc=kc, ks=ks), Dn, threads=opt["threads"],
                                    horizon_alpha=opt["horizon_alpha"], qd=bool(opt["reseed_q"]), od=opt["reseed_o"])
                    dz0_at.update(D=float(Dn), D_str=str(Dn), a=float(a))
                    if opt["save_pred"] and pred4 is not None:
                        # the prediction before its re-seed, for testing the re-seed on it (13.26)
                        save_q(os.path.join(out, f"pred_pin{kf}_D{float(Dn):.4f}.npz"), pred4,
                               dict(D=float(Dn), D_str=str(Dn), M=M, nodes=nodes, kt=kt, kc=kc, ks=ks, kfreeze=kf, pin_held=ks_held,
                                    N=N, a=a, source=opt["state"], prediction=True))
                    zr, rsd = rs_n.reseed(pred4 if pred4 is not None else (ph, pl))
                    elim["rs"] = rs_n if opt["reseed"] == 2 else None
                    rsd = dict(ok=rsd["ok"], dropped=rsd["dropped"], seconds=rsd["seconds"],
                               rungs=[dict(u=r["u"], its=r.get("its"), R=r.get("R"), conv=r.get("conv")) for r in rsd["rungs"]])
                    bad = [r for r in rsd["rungs"] if not r.get("conv")]
                    print(f"    a {a:.4f}: re-seeded ({'ok' if rsd['ok'] else 'FAILED, prediction kept'}), {rsd['seconds']} s"
                          + (f"; rung u {bad[0]['u']:.4f} not converged at |R| {bad[0].get('R', float('nan')):.2e} after {bad[0].get('its')} steps" if bad and not rsd["ok"] else "")
                          + (f"; rungs dropped {rsd['dropped']}" if rsd["dropped"] else ""), flush=True)
                    ph, pl = np.asarray(zr[0]), np.asarray(zr[1])
                    if pred4 is not None:
                        pred4 = tuple(np.ascontiguousarray(np.asarray(v)) for v in zr)
                    if pred8 is not None and rsd["ok"]:
                        pred8 = with_horizon(pred8, zr, rs_n)          # the re-seeded horizon data in eight parts
                        if rs_n.od:
                            print("      horizon data refined in octo-double, rung max|R| "
                                  + " -> ".join(f"{r:.2e}" for r in rs_n.last_o), flush=True)
                sn = mk(Dn)
                segs_n = segs
                sn.set_segs(segs_n)
                spn = wrap_at(sn, Dn)
                Fh, Fl = spn.residual(ph, pl)
                Fp = float(rnorm(Fh, Fl))
                if np.isfinite(Fp) and Fp <= 1e-6:
                    segs_n = sn.plan(ph, pl, 1e-32, 0.05)
                    sn.set_segs(segs_n)
                    Fh, Fl = spn.residual(ph, pl); Fp = float(rnorm(Fh, Fl))
                # --F_pred_max: a prediction above it is not corrected; with --fix1 the error is second order in a,
                # so the step shrinks by the square root of the excess (at most 6 times per attempt)
                if not (opt["F_pred_max"] and np.isfinite(Fp) and Fp > opt["F_pred_max"] and len(screens) < 6):
                    break
                screens.append((a, Fp))
                print(f"    a {a:.4f}: predicted |F| {Fp:.2e} above {opt['F_pred_max']:.1e}, not corrected", flush=True)
                a *= max(0.25, 0.9 * (opt["F_pred_max"] / Fp) ** 0.5)
            t1 = time.time()
            tol = tol_at(Dn)
            if np.isfinite(Fp):
                ch, cl, Fc, nsteps, accepted, dcorr = correct(spn, ph, pl, tol, pred4, z8_start=pred8)
            else:
                ch, cl, Fc, nsteps, accepted, dcorr = ph, pl, float("nan"), 0, False, 0.0
            t_corr = time.time() - t1
            fd = float(formula_delta(float(Dn)))
            att = dict(attempt=attempt, a=a, D=float(Dn), D_str=str(Dn), F_pred=Fp, F_final=Fc, tol=tol, steps=nsteps, accepted=accepted, t_corr=t_corr,
                       Delta=float(ch[-1] + cl[-1]), dDelta_formula=float(ch[-1] + cl[-1]) - fd, dDelta_corr=dcorr,
                       moved=float(dd_norm(*dd_add(ch, cl, -ph, -pl))), screens=screens, reseed=rsd)
            rec.setdefault("attempts", []).append(att)
            print(f"    a {a:.3f} -> D {mp.nstr(Dn, 12)}: predicted |F| {Fp:.2e}, {opt['corrector']} {nsteps} steps -> {Fc:.2e} (tol {tol:.1e}"
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
        D, zh, zl, s, sp, segs = Dn, ch, cl, sn, spn, segs_n
        # re-base the trend on the new state (its harmonics sit on the trend's values at D, to the corrector's floor)
        base_coeffs = coeffs_of(zh, zl)
        n0 = float(D) - 2.0
        meta_out = dict(D=float(D), D_str=str(D), M=M, nodes=nodes, kt=kt, kc=kc, ks=ks, kfreeze=kf, pin_held=ks_held, N=N, a=a, source=opt["state"],
                        segs=[[float(x) for x in sg] for sg in segs])       # the run's plan, which the state solves
        if opt["corrector"] == "qd" and "z4" in last_q:
            save_q(os.path.join(out, f"anm_pin{kf}_D{float(D):.4f}.npz"), last_q["z4"], meta_out)
        elif opt["corrector"] == "od" and "z8" in last_q:
            save_o(os.path.join(out, f"anm_pin{kf}_D{float(D):.4f}.npz"), last_q["z8"], dict(meta_out, parts=8))
            print(f"      Delta of the saved state, its eight parts summed exactly: {delta_digits(last_q['z8'])}", flush=True)
        else:
            save(os.path.join(out, f"anm_pin{kf}_D{float(D):.4f}.npz"), zh, zl, meta_out)
        if not opt["a_hold"] or attempt == 0:
            a = min(opt["a_grow"] * a, opt["h_max"])          # the next step may grow again
    print("done", flush=True)


if __name__ == "__main__":
    main()

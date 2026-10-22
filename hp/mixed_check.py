#!/usr/bin/env python3
"""Checks of the mixed-precision Newton (f64 Jacobian, double-double residual; notes/MIXED_PRECISION.md).

    cd hp && python3 mixed_check.py agree [--strict] [--state=S.npz] [--M=..] [--variant=f64x4] [--mutate=dtau|omega|kappa] [--expect_fail]
    cd hp && python3 mixed_check.py controls
    cd hp && python3 mixed_check.py converge [--M=64] [--variants=f64x4,f64x4+ddlu] [--dDelta=1e-6] [--dz=1e-8] [--tol=1e-28]
    cd hp && python3 mixed_check.py ref --save=REF.npz | --compare=REF.npz

Variants of the f64 Jacobian (function variant): f64xN (all-f64 lanes, N columns per evaluation; f64x4 is the
default of Solver.jacobian_f64 and newton_mixed) and ddN (a double-double base trajectory with N f64 tangents); for
newton_mixed a suffix +ddlu factors the f64 Jacobian by the double-double LU instead of the f64 LU.
agree: the f64 Jacobian (Solver.jacobian_f64, --variant) against the hi part of the double-double Jacobian
  (Solver.jacobian) at a saved state, with the solver settings of its meta and its Taylor plan (plan(z, 1e-30, 0.05)),
  as profile_newton.py. Reports the relative difference per column (max |dJ| / max |J| over the column; overall and
  per residual block), in the Frobenius norm, entrywise on the nonzero entries, and the relative difference of the
  Newton step -J^-1 F with F the residual at a perturbed state (Delta + 1e-6, fields + 1e-8; both solved by the
  double-double LU, Solver.lu_factor), an estimate of the mixed Newton iteration's contraction factor, next to the
  same difference for the double-double Jacobian rounded to f64 (the floor for any f64 Jacobian, about the condition
  number times 2^-53).
    Two tiers. --strict (the bug detector): the D = 3.7, M = 48 state resampled to M = 16, where the f64 Jacobian has
  no amplified rounding; pass: per-column and Frobenius differences <= 1e-12. Default (the production
  configuration): the D = 3.7, M = 48 state at its own M (or --state, --M); pass: per-column and Frobenius <= 1e-6.
  At M >= 48 the Taylor steps planned for the double-double solution do not resolve the top tau-harmonics of a
  tangent, and the f64 base trajectory's rounding inside a step is amplified to ~1e-7 (~5e-10 with a double-double
  base); see notes/MIXED_PRECISION.md. --tol overrides the threshold.
    --mutate breaks the f64 path on purpose (dtau: no tau derivative; omega: omega scaled by 1 + 1e-9; kappa: the
  horizon solves lose the kappa tangent term, so only the horizon block's columns change); the check must then fail
  (exit 1; --expect_fail turns a failure into exit 0 and a pass into exit 1).
controls: every mutation in the strict tier (f64x4, and dtau also with f64x1 and dd4), and dtau and kappa in the
  production tier, each must fire; exit 0 if all fire.
converge: Newton with the exact double-double Jacobian (driver.newton_solve with Solver.newton) and with the f64
  Jacobian (Solver.newton_mixed, for each of --variants) from the same perturbed start to |F| <= --tol: Delta moved
  by --dDelta and every field entry by a deterministic pseudo-random amount of size --dz (default start: the D = 3.7
  state at --M). Reports iterations (Newton calls, the last one only confirming |F| <= tol), final |F|, seconds, and
  the differences between the solutions in Delta and in max |z|, next to a control: the same difference between two
  exact-Jacobian solves from two different starts (the precision to which the equations fix the solution; --no_control
  skips it). Pass: all converge, and the Delta and max |z| differences are at most max(1e-27, 10 x the control)
  relative (1e-27 without the control). --json=FILE writes the records.
ref: bit-identity of the default double-double path (residual, jacobian, newton, chord, lu_solve_t, mode_matrix at
  lam = 0.3, a Jacobian with frozen harmonics and one with an inward matching node) at the D = 3.7, M = 48 state
  against a file written by an earlier build (--save writes it; --compare exits 1 on any differing bit).
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from hpkernel.driver import dd_add, dd_norm, load, make_solver, newton_solve, resample_z  # noqa: E402

STATE = lambda M: os.path.join(HERE, "data", "lowD_states", f"lowDv_D3.7000_M{M}.npz")  # noqa: E731
MUTATIONS = {"dtau": 1, "omega": 2, "kappa": 3}
DEFAULT_VARIANT = "f64x4"


def variant(v):
    """keyword arguments of Solver.newton_mixed for a variant name (see the module doc)"""
    kw = {}
    if v.endswith("+ddlu"):
        kw["lu"] = "dd"
        v = v[:-5]
    if v.startswith("dd"):
        kw.update(tangents=int(v[2:]), dd_base=True)
    elif v.startswith("f64x"):
        kw["tangents"] = int(v[4:])
    else:
        raise ValueError(f"unknown variant {v}")
    return kw


def jac_kw(v):
    """keyword arguments of Solver.jacobian_f64 for a variant name"""
    return {k: x for k, x in variant(v).items() if k != "lu"}


def opts_of(argv):
    pos = [a for a in argv if not a.startswith("--")]
    opts = {}
    for a in argv:
        if a.startswith("--"):
            k, _, v = a[2:].partition("=")
            opts[k] = v if v else True
    return pos, opts


def solver_for(path, threads, M=None, **kw):
    """solver with the state's meta settings and its plan; the state resampled to M if given (driver.resample_z)"""
    zh, zl, m = load(path)
    if M and M != m["M"]:
        zh, zl = resample_z(zh + zl, m["M"], M, len(m["nodes"]))
    else:
        M = m["M"]
    s = make_solver(m["D"], M, nodes=m["nodes"], kt=m["kt"], kc=m["kc"], ks=m["ks"], threads=threads, **kw)
    s.set_segs(m.get("segs") if M == m["M"] and m.get("segs") else s.plan(zh, zl, 1e-30, 0.05))
    return s, np.asarray(zh), np.asarray(zl), dict(m, M=M)


def perturb(zh, zl, dDelta, dz, seed):
    """Delta + dDelta, every other entry + dz * u, u uniform in [-1, 1] (fixed seed)"""
    u = np.random.default_rng(seed).uniform(-1, 1, len(zh))
    u[-1] = 0.0
    e = dz * u
    e[-1] = dDelta
    return dd_add(zh, zl, e, np.zeros_like(e))


def compare_jacobians(Jd, Jf, M, K):
    """relative differences of Jf against Jd (both f64 arrays); residual blocks of 4M rows (K + 1 of them)"""
    dJ = np.abs(Jf - Jd)
    aJ = np.abs(Jd)
    nz = aJ > 0
    rel = dJ[nz] / aJ[nz]
    colmax = aJ.max(axis=0)
    big = aJ >= 1e-8 * colmax[None, :]
    rel_big = dJ[big] / aJ[big]
    percol = dJ.max(axis=0) / np.where(colmax > 0, colmax, 1.0)
    blocks = []
    for b in range(K + 1):
        rows = slice(b * 4 * M, (b + 1) * 4 * M)
        cm = aJ[rows].max(axis=0)
        sel = cm > 0
        blocks.append(float((dJ[rows].max(axis=0)[sel] / cm[sel]).max()))
    i_worst = np.unravel_index(np.argmax(np.where(nz, dJ / np.where(nz, aJ, 1.0), 0.0)), Jd.shape)
    return dict(
        nonzero=int(nz.sum()), new_nonzero=int(((Jf != 0) & ~nz).sum()),
        max_new_nonzero=float(np.abs(Jf[~nz]).max()) if (~nz).any() else 0.0,
        entry_max=float(rel.max()), entry_q=[float(np.quantile(rel, q)) for q in (0.5, 0.9, 0.99, 0.999)],
        entry_big_max=float(rel_big.max()), worst_entry=[int(i_worst[0]), int(i_worst[1])],
        worst_entry_value=float(Jd[i_worst]), worst_entry_colmax=float(colmax[i_worst[1]]),
        column_max=float(percol.max()), column_worst=int(np.argmax(percol)), block_column_max=blocks,
        frobenius=float(np.linalg.norm(Jf - Jd) / np.linalg.norm(Jd)))


def agree(opts, quiet=False):
    strict = bool(opts.get("strict"))
    path = opts.get("state", STATE(48))
    M = int(opts.get("M", 16 if strict else 0)) or None
    threads = int(opts.get("threads", 2))
    tol = float(opts.get("tol", 1e-12 if strict else 1e-6))
    var = opts.get("variant", DEFAULT_VARIANT)
    mut = opts.get("mutate")
    s, zh, zl, m = solver_for(path, threads, M)
    t = time.time(); Jh, Jl = s.jacobian(zh, zl); t_dd = time.time() - t
    t = time.time(); Jf = s.jacobian_f64(zh, zl, MUTATIONS[mut] if mut else 0, **jac_kw(var)); t_f = time.time() - t
    Jh, Jf = np.asarray(Jh), np.asarray(Jf)
    r = compare_jacobians(Jh, Jf, m["M"], len(m["nodes"]))
    # one Newton step from a perturbed state with each Jacobian, both solved by the double-double LU
    Fh, Fl = s.residual(*perturb(zh, zl, 1e-6, 1e-8, 1))
    Fh, Fl = -np.asarray(Fh), -np.asarray(Fl)
    s.lu_factor(Jh, np.asarray(Jl))
    x_dd = np.add(*s.lu_solve(Fh, Fl))
    s.lu_factor(Jf)
    x_f = np.add(*s.lu_solve(Fh, Fl))
    r["step"] = float(np.linalg.norm(x_f - x_dd) / max(np.linalg.norm(x_dd), 1e-300))
    # the floor for any f64 Jacobian: the same step with the double-double Jacobian rounded to f64 (cond(J) x 2^-53)
    s.lu_factor(Jh)
    x_r = np.add(*s.lu_solve(Fh, Fl))
    r["step_rounding_floor"] = float(np.linalg.norm(x_r - x_dd) / max(np.linalg.norm(x_dd), 1e-300))
    ok = r["column_max"] <= tol and r["frobenius"] <= tol
    r.update(state=os.path.basename(path), M=m["M"], N=s.size, threads=threads, variant=var, mutate=mut,
             tier="strict" if strict else "production", t_jacobian=round(t_dd, 2), t_jacobian_f64=round(t_f, 2),
             tol=tol, passed=bool(ok))
    tag = f"{r['state']}{' resampled to M = %d' % m['M'] if M else ''} (N = {r['N']}), {r['tier']}, variant {var}"
    print(f"{tag}{', MUTATED: ' + mut if mut else ''}: jacobian {t_dd:.1f} s, jacobian_f64 {t_f:.2f} s")
    if not quiet:
        print(f"  entrywise on the {r['nonzero']} nonzero entries: max {r['entry_max']:.2e} (median {r['entry_q'][0]:.1e}, "
              f"90% {r['entry_q'][1]:.1e}, 99% {r['entry_q'][2]:.1e}, 99.9% {r['entry_q'][3]:.1e}); worst at {r['worst_entry']}, "
              f"|J| {abs(r['worst_entry_value']):.1e} in a column of max {r['worst_entry_colmax']:.1e}")
        print(f"  entries >= 1e-8 of their column max: max {r['entry_big_max']:.2e}; nonzero in f64 only: {r['new_nonzero']} "
              f"(max {r['max_new_nonzero']:.1e})")
        print("  per column, by residual block (centre, segments 1..K-1, horizon): "
              + " ".join(f"{v:.0e}" for v in r["block_column_max"]))
    print(f"  per column max|dJ|/max|J|: {r['column_max']:.2e} (column {r['column_worst']}); Frobenius {r['frobenius']:.2e}; "
          f"Newton step at a perturbed state: {r['step']:.2e} (with the dd Jacobian rounded to f64: {r['step_rounding_floor']:.2e})")
    print(f"  {'PASS' if ok else 'FAIL'} (per-column and Frobenius <= {tol:g})", flush=True)
    if "json" in opts:
        json.dump(r, open(opts["json"], "w"), indent=1)
    if opts.get("expect_fail"):
        print("  expected to fail:", "it did (the control fires)" if not ok else "it PASSED: the control does not fire")
        return 0 if not ok else 1
    return 0 if ok else 1


def controls(opts):
    runs = ([dict(strict=True, mutate=k) for k in MUTATIONS]
            + [dict(strict=True, mutate="dtau", variant=v) for v in ("f64x1", "dd4")]
            + [dict(mutate="dtau"), dict(mutate="kappa")])
    fired = []
    for o in runs:
        o.update(expect_fail=True, threads=opts.get("threads", 2))
        fired.append(agree(o, quiet=True) == 0)
    print(f"controls: {sum(fired)} of {len(fired)} fire", "PASS" if all(fired) else "FAIL")
    return 0 if all(fired) else 1


def run_newton(s, zh, zl, tol, var, it_max):
    """driver.newton_solve with s.newton (var None) or s.newton_mixed (variant var); returns z, history, seconds"""
    t = time.time()
    if var is None:
        zh2, zl2, hist = newton_solve(s, zh, zl, tol=tol, it_max=it_max, verbose=True)
    else:
        zh2, zl2, hist = newton_solve(s, zh, zl, tol=tol, it_max=it_max, verbose=True, jac="f64", mixed=variant(var))
    return np.asarray(zh2), np.asarray(zl2), hist, time.time() - t


def zdiff(a, b):
    """(relative Delta difference, max |z difference| / max |z|) of two dd vectors"""
    dh, dl = dd_add(a[0], a[1], -b[0], -b[1])
    d = dh + dl
    return abs(float(d[-1])) / abs(float(a[0][-1])), float(np.abs(d[:-1]).max() / np.abs(a[0][:-1]).max())


def converge(opts):
    M = int(opts.get("M", 64))
    path = opts.get("state", STATE(M))
    threads = int(opts.get("threads", 2))
    tol = float(opts.get("tol", 1e-28))
    dD, dz = float(opts.get("dDelta", 1e-6)), float(opts.get("dz", 1e-8))
    it_max = int(opts.get("it_max", 12))
    variants = str(opts.get("variants", DEFAULT_VARIANT)).split(",")
    s, zh0, zl0, m = solver_for(path, threads, M)
    out = dict(state=os.path.basename(path), M=m["M"], N=s.size, threads=threads, dDelta=dD, dz=dz, tol=tol, runs={})
    zh, zl = perturb(zh0, zl0, dD, dz, 1)
    print(f"{out['state']} at M = {m['M']}: start |F| {dd_norm(*s.residual(zh, zl)):.2e} (Delta {dD:g}, fields {dz:g})")
    sols = {}
    for name, var in [("newton", None)] + [(f"newton_mixed {v}", v) for v in variants]:
        print(f" {name}:", flush=True)
        a, b, hist, sec = run_newton(s, zh, zl, tol, var, it_max)
        Fh, Fl = s.residual(a, b)
        sols[name] = (a, b)
        out["runs"][name] = dict(newton_calls=len(hist), F=dd_norm(Fh, Fl), seconds=round(sec, 1), Delta=float(a[-1]),
                                 Delta_lo=float(b[-1]), converged=dd_norm(Fh, Fl) <= tol,
                                 history=[dict(it=h["it"], F=h["F"], lam=h.get("lam"), t=h.get("t")) for h in hist])
    ctl = None
    if not opts.get("no_control"):
        zh2, zl2 = perturb(zh0, zl0, -dD, dz, 2)
        print(" control (newton from a second start):", flush=True)
        a, b, hist, sec = run_newton(s, zh2, zl2, tol, None, it_max)
        out["control"] = dict(newton_calls=len(hist), F=dd_norm(*s.residual(a, b)), seconds=round(sec, 1))
        ctl = zdiff(sols["newton"], (a, b))
        out.update(control_Delta_rel_diff=ctl[0], control_z_rel_diff=ctl[1])
    ok = all(r["converged"] for r in out["runs"].values())
    for name in out["runs"]:
        r = out["runs"][name]
        print(f"  {name:22s}: {r['newton_calls']} Newton calls, |F| {r['F']:.1e}, {r['seconds']:.1f} s, "
              f"Delta {r['Delta']!r} {r['Delta_lo']:+.3e}")
        if name != "newton":
            dDm, dzm = zdiff(sols["newton"], sols[name])
            r.update(Delta_rel_diff=dDm, z_rel_diff=dzm)
            lim = (max(1e-27, 10 * ctl[0]), max(1e-27, 10 * ctl[1])) if ctl else (1e-27, 1e-27)
            ok = ok and dDm <= lim[0] and dzm <= lim[1]
            print(f"    vs newton: Delta {dDm:.1e} relative, max|dz|/max|z| {dzm:.1e} (limits {lim[0]:.0e}, {lim[1]:.0e})")
    if ctl:
        print(f"  control, newton from two starts: Delta {ctl[0]:.1e}, max|dz|/max|z| {ctl[1]:.1e}")
    out["passed"] = bool(ok)
    print(f"  {'PASS' if ok else 'FAIL'}")
    if "json" in opts:
        json.dump(out, open(opts["json"], "w"), indent=1)
    return 0 if ok else 1


def ref_outputs(path, threads):
    """every default-path output that must not change, as named float64 arrays"""
    out = {}
    s, zh, zl, m = solver_for(path, threads)
    out["residual_hi"], out["residual_lo"] = s.residual(zh, zl)
    out["jacobian_hi"], out["jacobian_lo"] = s.jacobian(zh, zl)
    dh, dl, Fh, Fl, pmin, pmax = s.newton(zh, zl)
    out["newton_dz_hi"], out["newton_dz_lo"], out["newton_piv"] = dh, dl, np.array([pmin, pmax])
    th, tl = dd_add(zh, zl, np.asarray(dh), np.asarray(dl))
    ch, cl, _, _ = s.chord(th, tl)
    out["chord_hi"], out["chord_lo"] = ch, cl
    e = np.zeros(s.size)
    e[-1] = 1.0
    out["lu_solve_t_hi"], out["lu_solve_t_lo"] = s.lu_solve_t(e, np.zeros(s.size))
    out["mode_matrix_hi"], out["mode_matrix_lo"] = s.mode_matrix(zh, zl, 0.3, 0.0)
    s2, _, _, _ = solver_for(path, threads, kfreeze=13)
    out["jacobian_freeze_hi"], out["jacobian_freeze_lo"] = s2.jacobian(zh, zl)
    s3, _, _, _ = solver_for(path, threads, match_node=8)
    out["jacobian_inward_hi"], out["jacobian_inward_lo"] = s3.jacobian(zh, zl)
    return {k: np.ascontiguousarray(np.asarray(v, dtype=np.float64)) for k, v in out.items()}


def ref(opts):
    path = opts.get("state", STATE(48))
    threads = int(opts.get("threads", 2))
    t = time.time()
    out = ref_outputs(path, threads)
    sha = {k: hashlib.sha256(v.tobytes()).hexdigest()[:16] for k, v in out.items()}
    print(f"{os.path.basename(path)}: {len(out)} outputs in {time.time() - t:.0f} s")
    if "save" in opts:
        np.savez_compressed(opts["save"], **out)
        for k in out:
            print(f"  {k:22s} sha256 {sha[k]}")
        return 0
    want = np.load(opts["compare"])
    bad = 0
    for k, v in out.items():
        w = want[k]
        same = w.shape == v.shape and np.array_equal(w.view(np.uint64), v.view(np.uint64))
        bad += not same
        extra = "" if same else f"  DIFFERS: {int((w.view(np.uint64) != v.view(np.uint64)).sum()) if w.shape == v.shape else 'shape'} entries"
        print(f"  {k:22s} sha256 {sha[k]} {'bit-identical' if same else ''}{extra}")
    print("PASS: bit-identical" if not bad else f"FAIL: {bad} outputs differ")
    return 1 if bad else 0


def main():
    pos, opts = opts_of(sys.argv[1:])
    cmd = pos[0] if pos else "agree"
    return {"agree": agree, "controls": controls, "converge": converge, "ref": ref}[cmd](opts)


if __name__ == "__main__":
    sys.exit(main())

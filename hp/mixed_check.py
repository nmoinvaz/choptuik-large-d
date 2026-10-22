#!/usr/bin/env python3
"""Checks of the mixed-precision Newton (f64 Jacobian, double-double residual; notes/MIXED_PRECISION.md).

    cd hp && python3 mixed_check.py agree [--state=S.npz] [--threads=2] [--mutate=dtau|omega|kappa] [--expect_fail]
    cd hp && python3 mixed_check.py converge [--state=S.npz] [--M=64] [--dDelta=1e-6] [--dz=1e-8] [--tol=1e-28]
    cd hp && python3 mixed_check.py ref --save=REF.npz | --compare=REF.npz [--state=S.npz]

agree: the f64 Jacobian (Solver.jacobian_f64) against the hi part of the double-double one (Solver.jacobian) at a
  saved state (default D = 3.7, M = 48; solver settings from the state's meta, as profile_newton.py). Reports the
  relative difference entrywise on the nonzero entries (maximum and quantiles, also restricted to entries above
  1e-8 of their column's maximum), per column (max |dJ| / max |J| over each column) and in the Frobenius norm, and
  the relative difference of the Newton step -J^-1 F solved with each (f64 LAPACK LU of both; the contraction factor
  of the mixed Newton iteration). Pass: the per-column and the Frobenius differences are both below --tol (1e-12).
  --mutate breaks the f64 path on purpose (dtau: no tau derivative; omega: omega scaled by 1 + 1e-9; kappa: the
  horizon solves lose the kappa tangent term, so only the horizon block's a_h and Delta columns change); the check
  must then fail (exit 1; --expect_fail turns a failure into exit 0 and a pass into exit 1).
converge: Newton with the exact double-double Jacobian (driver.newton_solve, Solver.newton) and with the f64
  Jacobian (Solver.newton_mixed) from the same perturbed start to |F| <= --tol: Delta moved by --dDelta and every
  field entry by a deterministic pseudo-random amount of size --dz (default start: the D = 3.7 state at --M). Reports
  iterations, final |F|, seconds, and the differences between the two solutions in Delta and in max |z|, next to a
  control: the same difference between two exact-Jacobian solves from two different starts (the precision to which
  the equations fix the solution). Pass: both converge, and the Delta and max |z| differences are at most
  max(1e-27, 10 x the control) relative. --json=FILE writes the records.
ref: bit-identity of the default double-double path (residual, jacobian, newton, chord, lu_solve_t, mode_matrix at
  lam = 0.3, a Jacobian with frozen harmonics and one with an inward matching node) against a file written by an
  earlier build (--save writes it; --compare exits 1 on any differing bit).
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
from hpkernel.driver import dd_add, dd_norm, load, make_solver, newton_solve, rnorm  # noqa: E402

STATE = lambda M: os.path.join(HERE, "data", "lowD_states", f"lowDv_D3.7000_M{M}.npz")  # noqa: E731
MUTATIONS = {"dtau": 1, "omega": 2, "kappa": 3}


def opts_of(argv):
    pos = [a for a in argv if not a.startswith("--")]
    opts = {}
    for a in argv:
        if a.startswith("--"):
            k, _, v = a[2:].partition("=")
            opts[k] = v if v else True
    return pos, opts


def solver_for(path, threads, **kw):
    zh, zl, m = load(path)
    s = make_solver(m["D"], m["M"], nodes=m["nodes"], kt=m["kt"], kc=m["kc"], ks=m["ks"], threads=threads, **kw)
    s.set_segs(m.get("segs") or s.plan(zh, zl, 1e-30, 0.05))
    return s, zh, zl, m


def compare_jacobians(Jd, Jf):
    """relative differences of Jf against Jd (both f64 arrays)"""
    dJ = np.abs(Jf - Jd)
    aJ = np.abs(Jd)
    nz = aJ > 0
    rel = dJ[nz] / aJ[nz]
    colmax = aJ.max(axis=0)
    big = aJ >= 1e-8 * colmax[None, :]
    rel_big = dJ[big] / aJ[big]
    percol = dJ.max(axis=0) / np.where(colmax > 0, colmax, 1.0)
    i_worst = np.unravel_index(np.argmax(np.where(nz, dJ / np.where(nz, aJ, 1.0), 0.0)), Jd.shape)
    return dict(
        nonzero=int(nz.sum()), new_nonzero=int(((Jf != 0) & ~nz).sum()), max_new_nonzero=float(np.abs(Jf[~nz]).max()),
        entry_max=float(rel.max()), entry_q=[float(np.quantile(rel, q)) for q in (0.5, 0.9, 0.99, 0.999)],
        entry_big_max=float(rel_big.max()), worst_entry=[int(i_worst[0]), int(i_worst[1])],
        worst_entry_value=float(Jd[i_worst]), worst_entry_colmax=float(colmax[i_worst[1]]),
        column_max=float(percol.max()), column_worst=int(np.argmax(percol)),
        frobenius=float(np.linalg.norm(Jf - Jd) / np.linalg.norm(Jd)))


def agree(opts):
    import scipy.linalg as sla
    path = opts.get("state", STATE(48))
    threads = int(opts.get("threads", 2))
    tol = float(opts.get("tol", 1e-12))
    mut = opts.get("mutate")
    s, zh, zl, m = solver_for(path, threads)
    t = time.time(); Jh, Jl = s.jacobian(zh, zl); t_dd = time.time() - t
    t = time.time()
    Jf = s.jacobian_f64(zh, zl, MUTATIONS[mut] if mut else 0, int(opts.get("lanes", 0)))
    t_f = time.time() - t
    Jh, Jf = np.asarray(Jh), np.asarray(Jf)
    r = compare_jacobians(Jh, Jf)
    Fh, Fl = s.residual(zh, zl)
    F = np.asarray(Fh) + np.asarray(Fl)
    x_dd = sla.lu_solve(sla.lu_factor(Jh, check_finite=False), -F, check_finite=False)
    x_f = sla.lu_solve(sla.lu_factor(Jf, check_finite=False), -F, check_finite=False)
    r["step"] = float(np.linalg.norm(x_f - x_dd) / max(np.linalg.norm(x_dd), 1e-300))
    ok = r["column_max"] <= tol and r["frobenius"] <= tol
    r.update(state=os.path.basename(path), M=m["M"], N=s.size, threads=threads, mutate=mut, t_jacobian=round(t_dd, 2),
             t_jacobian_f64=round(t_f, 2), tol=tol, passed=bool(ok))
    print(f"{r['state']} (N = {r['N']}){' MUTATED: ' + mut if mut else ''}: jacobian {t_dd:.1f} s, jacobian_f64 {t_f:.2f} s")
    print(f"  entrywise on the {r['nonzero']} nonzero entries: max {r['entry_max']:.2e} (median {r['entry_q'][0]:.1e}, "
          f"90% {r['entry_q'][1]:.1e}, 99% {r['entry_q'][2]:.1e}, 99.9% {r['entry_q'][3]:.1e}); worst at {r['worst_entry']}, "
          f"|J| {abs(r['worst_entry_value']):.1e} in a column of max {r['worst_entry_colmax']:.1e}")
    print(f"  entries >= 1e-8 of their column max: max {r['entry_big_max']:.2e}; nonzero in f64 only: {r['new_nonzero']} "
          f"(max {r['max_new_nonzero']:.1e})")
    print(f"  per column max|dJ|/max|J|: {r['column_max']:.2e} (column {r['column_worst']}); Frobenius {r['frobenius']:.2e}; "
          f"Newton step -J^-1 F: {r['step']:.2e}")
    print(f"  {'PASS' if ok else 'FAIL'} (per-column and Frobenius <= {tol:g})")
    if "json" in opts:
        json.dump(r, open(opts["json"], "w"), indent=1)
    if opts.get("expect_fail"):
        print("  expected to fail:", "it did (the control fires)" if not ok else "it PASSED: the control does not fire")
        return 0 if not ok else 1
    return 0 if ok else 1


def perturb(zh, zl, M, dDelta, dz, seed):
    """Delta + dDelta, every other entry + dz * u, u uniform in [-1, 1] (fixed seed)"""
    u = np.random.default_rng(seed).uniform(-1, 1, len(zh))
    u[-1] = 0.0
    e = dz * u
    e[-1] = dDelta
    return dd_add(zh, zl, e, np.zeros_like(e))


def run_newton(s, zh, zl, tol, mixed, it_max, lanes=0):
    """driver.newton_solve with s.newton or s.newton_mixed; returns z, history, seconds"""
    if mixed:
        class Mixed:
            def __getattr__(self, k):
                return getattr(s, k)

            def newton(self, a, b):
                return s.newton_mixed(a, b, lanes)
        so = Mixed()
    else:
        so = s
    t = time.time()
    zh2, zl2, hist = newton_solve(so, zh, zl, tol=tol, it_max=it_max, verbose=True)
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
    s, zh0, zl0, m = solver_for(path, threads)
    out = dict(state=os.path.basename(path), M=m["M"], N=s.size, threads=threads, dDelta=dD, dz=dz, tol=tol)
    zh, zl = perturb(zh0, zl0, m["M"], dD, dz, 1)
    print(f"{out['state']}: start |F| {dd_norm(*s.residual(zh, zl)):.2e} (Delta {dD:g}, fields {dz:g})")
    res = {}
    for name, mixed in (("newton", False), ("newton_mixed", True)):
        print(f" {name}:")
        a, b, hist, sec = run_newton(s, zh, zl, tol, mixed, it_max, int(opts.get("lanes", 0)))
        Fh, Fl = s.residual(a, b)
        res[name] = (a, b)
        out[name] = dict(iterations=len(hist), newton_calls=len(hist), F=dd_norm(Fh, Fl), seconds=round(sec, 1),
                         Delta=float(a[-1]), Delta_lo=float(b[-1]), converged=dd_norm(Fh, Fl) <= tol,
                         history=[dict(it=h["it"], F=h["F"], lam=h.get("lam")) for h in hist])
    # control: a second exact-Jacobian solve from a different start (another seed, same sizes)
    zh2, zl2 = perturb(zh0, zl0, m["M"], -dD, dz, 2)
    print(" control (newton from a second start):")
    a, b, hist, sec = run_newton(s, zh2, zl2, tol, False, it_max)
    out["control"] = dict(iterations=len(hist), F=dd_norm(*s.residual(a, b)), seconds=round(sec, 1))
    dD_mix, dz_mix = zdiff(res["newton"], res["newton_mixed"])
    dD_ctl, dz_ctl = zdiff(res["newton"], (a, b))
    out.update(Delta_rel_diff=dD_mix, z_rel_diff=dz_mix, control_Delta_rel_diff=dD_ctl, control_z_rel_diff=dz_ctl)
    ok = (out["newton"]["converged"] and out["newton_mixed"]["converged"]
          and dD_mix <= max(1e-27, 10 * dD_ctl) and dz_mix <= max(1e-27, 10 * dz_ctl))
    out["passed"] = bool(ok)
    for k in ("newton", "newton_mixed"):
        r = out[k]
        print(f"  {k:13s}: {r['iterations']} iterations, |F| {r['F']:.1e}, {r['seconds']:.1f} s, Delta {r['Delta']!r} {r['Delta_lo']:+.3e}")
    print(f"  newton vs newton_mixed: Delta {dD_mix:.1e} relative, max|dz|/max|z| {dz_mix:.1e}")
    print(f"  control, newton from two starts: Delta {dD_ctl:.1e}, max|dz|/max|z| {dz_ctl:.1e}")
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
    return {"agree": agree, "converge": converge, "ref": ref}[cmd](opts)


if __name__ == "__main__":
    sys.exit(main())

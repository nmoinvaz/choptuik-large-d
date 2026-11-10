#!/usr/bin/env python3
"""gamma_dd's D 50 M 48 self-convergence gap (#266): two M = 48 solutions of one layout, the direction their difference
lies in, and which of them is converged.

    python3 hp/gamma_m48_gap.py solve OUTDIR STATE --path=gamma|r2|polish [--M=48] [--kf=13] [--threads=1] [--it=14]
    python3 hp/gamma_m48_gap.py analyze OUT.jsonl A.npz B.npz [--w=24] [--lam] [--threads=1] [--mutate=plant]
        [--save_dirs=DIRS.npz]
    python3 hp/gamma_m48_gap.py qd OUTDIR STATE.npz [--plan=REF.npz] [--kf=13] [--it=8] [--tol=1e-45] [--mutate=N]
    python3 hp/gamma_m48_gap.py lam OUT.jsonl STATE.npz [...] [--plan=REF.npz] [--kf=13|none]
    python3 hp/gamma_m48_gap.py lamres OUT.jsonl STATE.npz --base=LAM [--h=5e-14] [--eps=1e-30,...] [--n=2] [--plan=REF]

solve: an M = 32 state taken to M = 48 by one of two paths, saved as OUTDIR/<stem>_M<M>_<path>.npz with a record in
  OUTDIR/solve.jsonl.
  gamma: gamma_dd.py --M48's path before #271 (its --mutate=old_rule), call for call (resample_z in f64 with lo dropped, p1 projected onto the cut, Newton to
    max(10 |F_32|, 1e-27), Levenberg-Marquardt when Newton stops above 1e-20).
  r2: review R2's path (hp/review/rv2_resolve.py --M 48 --corrector dnp: resample_dd at 40 digits, deflated Newton to
    1e-40 with w = 16, then Newton to 1e-40).
  polish: STATE is already at M; plain Newton to 1e-40 on its own layout (tells whether the first path stopped early).
analyze: A and B on one solver (B's segment plan; A's own plan is compared with it). For each: max |F| in double-double
  and in quad-double (the same dd state, evaluated in quad-double), per block, the linearised Delta correction
  -dDelta/dF . F, the Newton step's Delta component and norm, both series' truncation (series_truncation.measure), and
  with --lam lam located as gamma_dd.py does. Then at B the w weakest right singular directions of the Jacobian
  (inverse subspace iteration on (J^T J)^-1 with the double-double LU, then Rayleigh-Ritz; the unscaled Jacobian as the
  solver uses it), the difference d = A - B projected on them, the Newton step at A projected on -d, and d's and the
  weakest directions' tau-harmonic content per field.
  Control (--mutate=plant): A is replaced by B plus 1e-12 along the third weakest direction and 1e-20 along a strong
  one (random, orthogonal to the weak subspace); the projection must return 1e-12 on that direction to 1e-6, below
  1e-18 on the others, and the strong part outside the weak subspace (exit 1 if not). --mutate=plant_strong swaps the
  two amplitudes, and the check must fail.
qd: STATE polished by the quad-double Newton (Solver.newton_q, plain, hpkernel.driver.newton_qd) on the M = 48 layout
  with the cut, the segment plan of --plan's state (default STATE's own), so that two states can be polished on one
  system. Saves OUTDIR/<stem>_qd.npz (four parts) and a record in OUTDIR/qd.jsonl with Delta to 30 digits.
  --mutate=N passes Ctx::mutate N to every call (the controls of the quad-double lanes).
lam: lam located on each state as gamma_dd.py does (find_lam, brentq xtol 1e-14), then refined inside +-2e-13 in
  double-double lam (lam_fine: base + t, brentq on t with xtol 1e-19); a four-part state is rounded to its first two parts. Also records |F| and the linearised Delta
  correction -dDelta/dF . F at the state (est).
lamres: the det root's resolution on one state (see cmd_lamres): the root interpolated inside base +- h, on the state
  and on copies with relative noise eps.
"""
import json
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "review"))
from hpkernel.driver import (_normal_solve, block_norms, dd_add, deflated_newton, delta_correction, load, lm_solve,  # noqa: E402
                             make_solver, newton_qd, newton_solve, plow_odd, qparts, resample_z, save, save_q,
                             weak_subspace)

# field layout of z (driver.py): p1, c0 | node_k (p, m, a, c) | m_h, a_h | Delta
FIELD_NAMES = lambda K: ["p1", "c0"] + [f"{f}@{k}" for k in range(K) for f in ("p", "m", "a", "c")] + ["m_h", "a_h"]
PARITY = lambda K: ["odd", "even"] + ["odd", "odd", "even", "even"] * K + ["odd", "even"]


def opts_of(argv):
    args = [a for a in argv if not a.startswith("--")]
    opts = dict((a[2:].partition("=")[0], a[2:].partition("=")[2] or "1") for a in argv if a.startswith("--"))
    return args, opts


def solver_for(meta, zh, zl, M, kf, threads, plan_from=None):
    s = make_solver(meta["D"], M, nodes=meta["nodes"], kt=meta.get("kt", 36), kc=meta.get("kc", 420),
                    ks=meta.get("ks", 100), threads=threads, kfreeze=kf)
    ph, pl = plan_from if plan_from is not None else (zh, zl)
    s.set_segs(s.plan(ph, pl, 1e-32, 0.05))
    return s


def fmax(s, zh, zl):
    fh, fl = s.residual(zh, zl)
    return float(np.abs(np.asarray(fh) + np.asarray(fl)).max()), fh, fl


def delta_str(*parts):
    """Delta (the last entry) of a state of 2 or 4 parts, summed at 40 digits"""
    with mp.workdps(40):
        return mp.nstr(sum(mp.mpf(float(p[-1])) for p in parts), 30)


# ------------------------------------------------------------------ solve
def cmd_solve(args, opts):
    from raw_out import require_raw
    outdir, path = args
    how = opts.get("path", "gamma")
    require_raw(os.path.join(outdir, "solve.jsonl"))
    threads = int(opts.get("threads", 1))
    kf = None if opts.get("kf", "13") == "none" else int(opts.get("kf", "13"))
    zh, zl, meta = load(path)
    D, M0, K = meta["D"], meta["M"], len(meta["nodes"])
    M = int(opts.get("M", 48))
    log = lambda m: print(m, flush=True)
    t0 = time.time()
    rec = dict(path=how, state=os.path.basename(path), D=D, M0=M0, M=M, kf=kf, Delta0=float(zh[-1]),
               Delta0_lo=float(zl[-1]))
    if how == "gamma":
        from gamma_dd import best_solver
        s, F, kfu = best_solver(meta, zh, zl, threads, force=str(kf) if kf else "none")
        rec.update(F0=F)
        hh, hl = resample_z(np.asarray(zh) + np.asarray(zl), M0, M, K)
        mh = dict(meta, M=M)
        if kf:
            a_, b_ = plow_odd(M, kf)
            hh[:M] = (a_ + b_) @ hh[:M]
        s2, F2, _ = best_solver(mh, hh, hl, threads, M=M, force=str(kf) if kf else "none")
        log(f"D {D}: M {M0} |F| {F:.1e}; resampled to M {M} (f64), |F| {F2:.1e}; Newton to {max(10 * F, 1e-27):.1e}")
        rec.update(F_start=F2, tol=max(10 * F, 1e-27))
        hh, hl, hist = newton_solve(s2, hh, hl, tol=max(10 * F, 1e-27), it_max=12, verbose=True)
        rec["newton"] = len(hist)
        if hist[-1]["F"] > 1e-20:
            log("  Newton stalled; Levenberg-Marquardt")
            hh, hl, hist2 = lm_solve(s2, hh, hl, tol=max(10 * F, 1e-27), it_max=20, verbose=True, newton_after=1e-22)
            rec["lm"] = len(hist2)
        zh2, zl2 = hh, hl
    elif how == "r2":
        from rv2_resolve import resample_dd
        hh, hl = resample_dd(zh, zl, M0, M, K) if M != M0 else (zh, zl)
        mh = dict(meta, M=M)
        s2 = solver_for(mh, hh, hl, M, kf, threads)
        rec.update(F_start=fmax(s2, hh, hl)[0])
        log(f"D {D}: resampled to M {M} at 40 digits, |F| {rec['F_start']:.1e}; deflated Newton, then Newton, to 1e-40")
        it = int(opts.get("it", 14))
        hh, hl, h0 = deflated_newton(s2, hh, hl, tol=1e-40, it_max=it, w=16, verbose=True)
        zh2, zl2, hist = newton_solve(s2, hh, hl, tol=1e-40, it_max=it, verbose=True)
        rec.update(deflated=len(h0), newton=len(hist))
    elif how == "polish":
        mh = dict(meta)
        M = M0
        s2 = solver_for(mh, zh, zl, M, kf, threads)
        rec.update(F_start=fmax(s2, zh, zl)[0])
        zh2, zl2, hist = newton_solve(s2, zh, zl, tol=1e-40, it_max=int(opts.get("it", 14)), verbose=True)
        rec.update(newton=len(hist), hist=[dict(F=h["F"], Delta=h["Delta"], dDelta=h.get("dDelta")) for h in hist])
    else:
        raise SystemExit(f"--path={how}: gamma, r2 or polish")
    zh2, zl2 = np.ascontiguousarray(zh2), np.ascontiguousarray(zl2)
    F2 = fmax(s2, zh2, zl2)[0]          # on the path's own solver (its plan from the start), as the paths report it
    src_h, src_l = zh, zl
    rec.update(F=F2, Delta=float(zh2[-1]), Delta_lo=float(zl2[-1]), Delta_str=delta_str(zh2, zl2),
               Delta_diff=float((zh2[-1] - src_h[-1]) + (zl2[-1] - src_l[-1])),
               steps=sum(len(g) - 1 for g in s2.get_segs()), seconds=round(time.time() - t0, 1))
    stem = os.path.basename(path)[:-4]
    out = os.path.join(outdir, f"{stem}_M{M}_{how}.npz" if how != "polish" else f"{stem}_polish.npz")
    save(out, zh2, zl2, dict(mh, M=M, F=F2, source=os.path.basename(path), path=how, kfreeze=kf))
    rec["saved"] = os.path.basename(out)
    log(json.dumps(rec))
    with open(os.path.join(outdir, "solve.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    return 0


# ------------------------------------------------------------------ analyze
def harmonics(v, M, K):
    """per field, |coefficient| of each tau harmonic k = 0..M (the half-grid field extended odd or even about pi)"""
    out = []
    for j, par in enumerate(PARITY(K)):
        f = v[j * M:(j + 1) * M]
        full = np.r_[f, -f] if par == "odd" else np.r_[f, f]
        c = np.abs(np.fft.rfft(full)) / (2 * M)
        c[1:M] *= 2
        out.append(c)
    return np.array(out)


def band_table(v, M, K, kf):
    """the largest harmonic amplitude per field group and band: k <= kf, kf < k < M/2, k >= M/2"""
    H = harmonics(v, M, K)
    names = FIELD_NAMES(K)
    groups = {"p1": [0], "c0": [1], "node p,m": [i for i, n in enumerate(names) if "@" in n and n[0] in "pm"],
              "node a,c": [i for i, n in enumerate(names) if "@" in n and n[0] in "ac"],
              "m_h,a_h": [len(names) - 2, len(names) - 1]}
    k = np.arange(M + 1)
    bands = {"k<=%d" % kf: k <= kf, "%d<k<%d" % (kf, M // 2): (k > kf) & (k < M // 2), "k>=%d" % (M // 2): k >= M // 2}
    return {g: {b: float(H[idx][:, m].max()) for b, m in bands.items()} for g, idx in groups.items()}, \
        float(abs(v[-1]))


def fmt_bands(tab):
    g0 = next(iter(tab))
    head = "    " + "".join(f"{b:>12s}" for b in tab[g0])
    return "\n".join([head] + [f"    {g:9s}" + "".join(f"{x:12.2e}" for x in r.values()) for g, r in tab.items()])


def per_state(s, meta, zh, zl, kf, opts, log, label):
    from series_truncation import measure, tails_rec
    F, fh, fl = fmax(s, zh, zl)
    fq = s.residual_q(*qparts(zh, zl))
    Fq = float(np.abs(np.asarray(fq[0]) + np.asarray(fq[1])).max())
    blocks_dd = block_norms(s, fh, fl)
    blocks_qd = block_norms(s, np.asarray(fq[0]), np.asarray(fq[1]))
    dh, dl, _, _, pmin, pmax = s.newton(zh, zl)
    dz = np.asarray(dh) + np.asarray(dl)
    est = delta_correction(s, fh, fl)
    _, _, tails = measure(dict(meta), zh, zl, kf, int(opts.get("threads", 1)))
    rec = dict(label=label, F=F, F_qd=Fq, blocks_dd=blocks_dd, blocks_qd=blocks_qd, est=est, newton_dDelta=float(dz[-1]),
               newton_dz=float(np.abs(dz).max()), piv=[pmin, pmax], Delta=float(zh[-1]), Delta_lo=float(zl[-1]),
               Delta_str=delta_str(zh, zl), tails=tails_rec(tails))
    log(f"  {label}: |F| dd {F:.2e}, qd {Fq:.2e}; blocks qd " + " ".join(f"{b:.1e}" for b in blocks_qd)
        + f"\n    linearised Delta correction {est:+.3e}; Newton step: dDelta {dz[-1]:+.3e}, max|dz| {rec['newton_dz']:.2e};"
        f" pivots {pmin:.1e}..{pmax:.1e}\n    Delta {rec['Delta_str']}; tails centre {tails[0]['tail']:.1e} horizon "
        f"{tails[1]['tail']:.1e}")
    return rec, dz


def weak_directions(s, w, iters=12):
    """the w weakest right singular directions of the Jacobian whose LU s holds, with their singular values
    (Rayleigh-Ritz on the subspace from inverse iteration on (J^T J)^-1), ascending in sigma"""
    V = weak_subspace(s, w, iters=iters)
    W = _normal_solve(s, V)
    H = V.T @ W
    ev, U = np.linalg.eigh((H + H.T) / 2)
    X = V @ U
    X /= np.linalg.norm(X, axis=0)
    sig = 1 / np.sqrt(np.maximum(ev, 1e-300))
    o = np.argsort(sig)
    return X[:, o], sig[o]


def cmd_analyze(args, opts):
    from raw_out import require_raw
    out, pa, pb = args
    require_raw(out)
    threads = int(opts.get("threads", 1))
    w = int(opts.get("w", 24))
    mutate = opts.get("mutate", "")
    log = lambda m: print(m, flush=True)
    zb_h, zb_l, mb = load(pb)
    za_h, za_l, ma = load(pa)
    kf = mb.get("kfreeze", 13 if mb["D"] >= 35 else None)
    M, K = mb["M"], len(mb["nodes"])
    assert ma["M"] == M and ma["nodes"] == mb["nodes"], "A and B must share M and the nodes"
    t0 = time.time()
    s = solver_for(mb, zb_h, zb_l, M, kf, threads)
    sa = solver_for(ma, za_h, za_l, M, kf, threads)
    segs_same = [list(map(float, g)) for g in s.get_segs()] == [list(map(float, g)) for g in sa.get_segs()]
    rec = dict(A=os.path.basename(pa), B=os.path.basename(pb), D=mb["D"], M=M, kf=kf, w=w, mutate=mutate,
               steps_B=sum(len(g) - 1 for g in s.get_segs()), steps_A=sum(len(g) - 1 for g in sa.get_segs()),
               segs_same=segs_same)
    log(f"D {mb['D']} M {M} kf {kf}: A {rec['A']}, B {rec['B']}; Taylor steps B {rec['steps_B']}, A's own plan "
        f"{rec['steps_A']} ({'the same' if segs_same else 'different'} segments); both evaluated on B's plan")
    del sa
    # B first: its LU stays in s for the weak directions
    recB, dzB = per_state(s, mb, zb_h, zb_l, kf, opts, log, "B")
    X, sig = weak_directions(s, w)
    if mutate in ("plant", "plant_strong"):
        # A := B + a known combination: 1e-12 along the third weakest direction, 1e-20 along a strong one (random,
        # orthogonalised against the weak subspace). plant_strong (the control's mutation) swaps the two, so the
        # check must fail.
        rng = np.random.default_rng(1)
        r = rng.standard_normal(len(zb_h))
        r -= X @ (X.T @ r)
        r /= np.linalg.norm(r)
        a_w, a_s = (1e-12, 1e-20) if mutate == "plant" else (1e-20, 1e-12)
        dplant = a_w * X[:, 2] + a_s * r
        za_h, za_l = dd_add(zb_h, zb_l, dplant, np.zeros_like(dplant))
        za_h, za_l = np.ascontiguousarray(za_h), np.ascontiguousarray(za_l)
        rec["planted"] = dict(weak_index=2, weak_coef=a_w, strong_coef=a_s)
    recA, dzA = per_state(s, ma, za_h, za_l, kf, opts, log, "A")
    d = (za_h - zb_h) + (za_l - zb_l)
    nd = float(np.linalg.norm(d))
    c = X.T @ d
    resid = d - X @ c
    # F(A) - F(B) on one solver: the nonlinear image of d, against sigma |c|
    fa = s.residual(za_h, za_l)
    fb = s.residual(zb_h, zb_l)
    dF = (np.asarray(fa[0]) - np.asarray(fb[0])) + (np.asarray(fa[1]) - np.asarray(fb[1]))
    cosN = float(-(dzA @ d) / (np.linalg.norm(dzA) * nd)) if nd > 0 else None
    rec.update(B_rec=recB, A_rec=recA, d_norm=nd, d_Delta=float(d[-1]), dF_norm=float(np.linalg.norm(dF)),
               sigma=[float(x) for x in sig], coef=[float(x) for x in c], dir_Delta=[float(x) for x in X[-1, :]],
               outside_weak=float(np.linalg.norm(resid)), newtonA_cos_minus_d=cosN,
               newtonA_ratio=float(np.linalg.norm(dzA) / nd) if nd > 0 else None)
    log(f"\n  d = A - B: |d|_2 {nd:.3e}, Delta component {d[-1]:+.3e}; |F(A) - F(B)|_2 {rec['dF_norm']:.2e} "
        f"(effective sigma along d {rec['dF_norm'] / nd if nd else float('nan'):.1e})")
    log(f"  the Newton step at A against -d: cosine {cosN}, |dz_A| / |d| {rec['newtonA_ratio']}")
    log(f"  part of d outside the {w} weakest directions: {rec['outside_weak']:.2e} ({rec['outside_weak'] / nd if nd else 0:.1e} "
        "of |d|)")
    log("  k  sigma      coef(d)     |coef| frac  Delta comp")
    for i in range(w):
        log(f"  {i:2d} {sig[i]:.3e}  {c[i]:+.3e}  {abs(c[i]) / nd if nd else 0:.2e}  {X[-1, i]:+.2e}")
    tab_d, _ = band_table(d, M, K, kf or 13)
    log("  tau-harmonic content of d (largest |coefficient| per group and band):\n" + fmt_bands(tab_d))
    rec["d_bands"] = tab_d
    # p1's odd harmonics k = 1..2 kf + 1 in d and in the six weakest directions (which harmonic each pair is)
    kk = list(range(1, 2 * (kf or 13) + 2, 2))
    rec["p1_harmonics_k"] = kk
    rec["p1_harmonics_d"] = [float(harmonics(d, M, K)[0][k]) for k in kk]
    rec["p1_harmonics_dirs"] = [[float(harmonics(X[:, i], M, K)[0][k]) for k in kk] for i in range(min(6, w))]
    log("  p1's harmonic amplitudes, k = " + " ".join(f"{k:8d}" for k in kk))
    log("    d          " + " ".join(f"{v:8.1e}" for v in rec["p1_harmonics_d"]))
    for i, row in enumerate(rec["p1_harmonics_dirs"]):
        log(f"    dir {i} {sig[i]:.0e} " + " ".join(f"{v:8.1e}" for v in row))
    if "save_dirs" in opts:
        np.savez(opts["save_dirs"], X=X[:, :min(8, w)], sigma=sig, d=d, dzA=dzA, dzB=dzB)
    rec["dir_bands"] = []
    for i in range(min(3, w)):
        tab, dD = band_table(X[:, i], M, K, kf or 13)
        rec["dir_bands"].append(tab)
        log(f"  weakest direction {i} (sigma {sig[i]:.2e}, Delta component {X[-1, i]:+.1e}):\n" + fmt_bands(tab))
    ok = True
    if mutate in ("plant", "plant_strong"):
        got = c[2]
        ok = abs(got / 1e-12 - 1) < 1e-6 and rec["outside_weak"] > 0.5e-20 and np.abs(np.delete(c, 2)).max() < 1e-18
        pl = rec["planted"]
        log(f"  control (planted): coefficient on direction 2 {got:.6e} (planted {pl['weak_coef']:.0e}), the part outside "
            f"the weak subspace {rec['outside_weak']:.2e} (planted {pl['strong_coef']:.0e} along a strong direction), "
            f"other coefficients max {np.abs(np.delete(c, 2)).max():.1e}; the check wants 1e-12 on direction 2 and 1e-20 "
            f"outside -> {'PASS' if ok else 'FAIL'}")
        rec["control_pass"] = bool(ok)
    if "lam" in opts:
        from gamma_dd import find_lam, gamma_formula
        lam0 = 1.0 / gamma_formula(mb["D"])
        for lab, (h, l_) in (("B", (zb_h, zb_l)), ("A", (za_h, za_l))):
            lam, nch, nev = find_lam(s, h, l_, lam0)
            rec[f"lam_{lab}"] = lam
            rec[f"nch_{lab}"] = nch
            log(f"  lam at {lab}: {lam!r} ({nch} sign change, {nev} determinants)")
        if rec.get("lam_A") and rec.get("lam_B"):
            log(f"  lam(A) - lam(B) {rec['lam_A'] - rec['lam_B']:+.3e}")
    rec["seconds"] = round(time.time() - t0, 1)
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return 0 if ok else 1


# ------------------------------------------------------------------ lam
def lam_fine(s, zh, zl, lam, h=2e-13, log=None):
    """the sign change of det M(lam) refined below find_lam's xtol 1e-14 and below the spacing of doubles near 2
    (4.4e-16): lam = base + t in double-double (two_sum), base = find_lam's root, brentq on t in [-h, h] with xtol 1e-19.
    Returns (base, t) or (None, n) when +-h has no sign change; the resolution is then set by the determinant's own
    accuracy, not by the root finder."""
    import math
    from scipy.optimize import brentq
    from multidouble import two_sum
    cache = {}

    def sd(t):
        if t not in cache:
            hi, lo = two_sum(np.float64(lam), np.float64(t))
            sg, la, _, _ = s.mode_det(zh, zl, float(hi), float(lo))
            cache[t] = (sg, la)
        return cache[t]

    if sd(-h)[0] == sd(h)[0]:
        return None, len(cache)
    ref = 0.5 * (sd(-h)[1] + sd(h)[1])
    t = brentq(lambda t: sd(t)[0] * math.exp(sd(t)[1] - ref), -h, h, xtol=1e-19, maxiter=100)
    return (float(lam), float(t)), len(cache)


def cmd_lam(args, opts):
    """lam on each state (a four-part state is rounded to its first two parts), with gamma_dd.find_lam and then
    lam_fine; all on the layout of --plan's state (default each state's own plan)."""
    from raw_out import require_raw
    from gamma_dd import find_lam, gamma_formula
    out, paths = args[0], args[1:]
    require_raw(out)
    threads = int(opts.get("threads", 1))
    plan_from = None
    if "plan" in opts:
        ph, pl, _ = load(opts["plan"])
        plan_from = (ph, pl)
    for p in paths:
        t0 = time.time()
        zh, zl, meta = load(p)
        zh, zl = np.ascontiguousarray(zh), np.ascontiguousarray(zl)
        kf = None if opts.get("kf", "13") == "none" else int(opts.get("kf", "13"))
        s = solver_for(meta, zh, zl, meta["M"], kf, threads, plan_from)
        F, fh, fl = fmax(s, zh, zl)
        s.newton(zh, zl)                       # the LU at the state, for the linearised Delta correction
        est = delta_correction(s, fh, fl)
        lam0 = 1.0 / gamma_formula(meta["D"])
        lam, nch, nev = find_lam(s, zh, zl, lam0)
        fine, nev2 = lam_fine(s, zh, zl, lam) if lam else (None, 0)
        if fine is not None:
            with mp.workdps(40):
                fine = dict(base=fine[0], t=fine[1], str=mp.nstr(mp.mpf(fine[0]) + mp.mpf(fine[1]), 25))
        rec = dict(state=os.path.basename(p), plan=os.path.basename(opts.get("plan", p)), D=meta["D"], M=meta["M"], kf=kf,
                   F=F, est=est, Delta_str=delta_str(zh, zl), lam=lam, n_signchanges=nch, lam_fine=fine, n_det=nev + nev2,
                   seconds=round(time.time() - t0, 1))
        print(json.dumps(rec), flush=True)
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return 0


def cmd_lamres(args, opts):
    """the resolution of the det root on one state: the root by linear interpolation of det M(base + t) between
    t = -h and +h (double-double lam), on the state and on n copies with relative noise eps added to every entry
    (eps 1e-30: double-double rounding; larger eps: how far a state error moves the root). The spread of the roots is
    the root's resolution at that eps; a single det evaluation's noise shows at eps 0 as the root's distance from
    lam_fine's."""
    import math
    from raw_out import require_raw
    from multidouble import two_sum
    out, path = args
    require_raw(out)
    threads = int(opts.get("threads", 1))
    zh, zl, meta = load(path)
    zh, zl = np.ascontiguousarray(zh), np.ascontiguousarray(zl)
    kf = None if opts.get("kf", "13") == "none" else int(opts.get("kf", "13"))
    plan_from = None
    if "plan" in opts:
        ph, pl, _ = load(opts["plan"])
        plan_from = (ph, pl)
    s = solver_for(meta, zh, zl, meta["M"], kf, threads, plan_from)
    base = float(opts["base"])
    h = float(opts.get("h", 5e-14))
    rng = np.random.default_rng(int(opts.get("seed", 0)))
    t0 = time.time()

    def root(h_, l_):
        v = []
        for t in (-h, h):
            hi, lo = two_sum(np.float64(base), np.float64(t))
            v.append(s.mode_det(h_, l_, float(hi), float(lo))[:2])
        ref = 0.5 * (v[0][1] + v[1][1])
        f = [sg * math.exp(la - ref) for sg, la in v]
        return (-h + 2 * h * f[0] / (f[0] - f[1])) if f[0] != f[1] else None, [x[0] for x in v]

    recs = []
    for eps in [0.0] + [float(e) for e in opts.get("eps", "1e-30").split(",")] * int(opts.get("n", 2)):
        if eps:
            noise = eps * np.abs(zh) * rng.standard_normal(len(zh))
            h_, l_ = dd_add(zh, zl, noise, np.zeros_like(noise))
            h_, l_ = np.ascontiguousarray(h_), np.ascontiguousarray(l_)
        else:
            h_, l_ = zh, zl
        t, signs = root(h_, l_)
        recs.append(dict(eps=eps, t=t, signs=signs))
        print(f"  eps {eps:.0e}: root base {'%+.3e' % t if t is not None else 'none in +-h'} (signs {signs})", flush=True)
    ts = [r["t"] for r in recs if r["t"] is not None]
    rec = dict(state=os.path.basename(path), plan=os.path.basename(opts.get("plan", path)), D=meta["D"], M=meta["M"],
               kf=kf, base=base, h=h, roots=recs, spread=(max(ts) - min(ts)) if ts else None,
               seconds=round(time.time() - t0, 1))
    print(json.dumps(rec), flush=True)
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return 0


# ------------------------------------------------------------------ qd polish
def cmd_qd(args, opts):
    from raw_out import require_raw
    outdir, path = args
    require_raw(os.path.join(outdir, "qd.jsonl"))
    threads = int(opts.get("threads", 1))
    zh, zl, meta = load(path)
    kf = None if opts.get("kf", "13") == "none" else int(opts.get("kf", str(meta.get("kfreeze") or 13)))
    plan_from = None
    if "plan" in opts:
        ph, pl, pm = load(opts["plan"])
        assert pm["M"] == meta["M"] and pm["nodes"] == meta["nodes"]
        plan_from = (ph, pl)
    s = solver_for(meta, zh, zl, meta["M"], kf, threads, plan_from)
    mutate = int(opts.get("mutate", 0))
    t0 = time.time()
    z = qparts(zh, zl)
    print(f"D {meta['D']} M {meta['M']} kf {kf}: {os.path.basename(path)}, plan of "
          f"{os.path.basename(opts.get('plan', path))} ({sum(len(g) - 1 for g in s.get_segs())} steps); quad-double Newton",
          flush=True)
    z, hist, conv = newton_qd(s, z, tol=float(opts.get("tol", 1e-45)), it_max=int(opts.get("it", 8)), verbose=True,
                              mutate=mutate)
    Fq = s.residual_q(*z)
    F = float(np.abs(np.asarray(Fq[0]) + np.asarray(Fq[1])).max())
    rec = dict(state=os.path.basename(path), plan=os.path.basename(opts.get("plan", path)), D=meta["D"], M=meta["M"],
               kf=kf, mutate=mutate, converged=conv, F_qd=F, Delta_start=delta_str(zh, zl), Delta_qd=delta_str(*z),
               steps=len(hist), hist=hist, seconds=round(time.time() - t0, 1))
    with mp.workdps(40):
        rec["Delta_change"] = float(mp.mpf(rec["Delta_qd"]) - mp.mpf(rec["Delta_start"]))
    stem = os.path.basename(path)[:-4]
    out = os.path.join(outdir, f"{stem}_qd" + (f"_mut{mutate}" if mutate else "") + ".npz")
    save_q(out, z, dict(meta, F_qd=F, source=os.path.basename(path), plan=rec["plan"], kfreeze=kf))
    rec["saved"] = os.path.basename(out)
    print(json.dumps(rec, default=float), flush=True)
    with open(os.path.join(outdir, "qd.jsonl"), "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
    return 0 if conv else 1


def main():
    args, opts = opts_of(sys.argv[1:])
    if not args:
        sys.exit(__doc__)
    cmd = args.pop(0)
    return dict(solve=cmd_solve, analyze=cmd_analyze, qd=cmd_qd, lam=cmd_lam, lamres=cmd_lamres)[cmd](args, opts)


if __name__ == "__main__":
    sys.exit(main())

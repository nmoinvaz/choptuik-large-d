#!/usr/bin/env python3
"""Issue #1043: checks of the horizon series' exact 1/k (Solver.set_exact_recip, model.rs Ctx::exact_recip) against
#1035's transcription of the series (hp/factored_horizon.py), and Delta's change between two re-solved states.

Commands, from hp/ (STATE an .npz of the chain; read with numpy, its parts summed exactly):
    python3 exact_recip_check.py recip [--kmax 4096] [--mutate=off]
    python3 exact_recip_check.py miss STATE [--prec 288] [--tol 1e-6] [--threads 4] [--mutate=planted [--k K]]
    python3 exact_recip_check.py plant STATE [--k 3,5,7,30,60,100] [--prec 288] [--tol 1e-6] [--threads 4] [--mutate=shift]
    python3 exact_recip_check.py delta OFF.npz ON.npz [--csv data/delta_all.csv]
    python3 exact_recip_check.py bits STATE --save=F.json | --compare=F.json [--threads 4] [--mutate=on]
    python3 exact_recip_check.py gain STATE --k K [--prec 288]
    python3 exact_recip_check.py predict OFF.npz ON.npz [--threads 4] [--prec 600] [--mutate=offr|prow]
--out FILE.json writes the command's record. Exit 1 when a check fails. Criteria fixed before the first run.

recip: Solver.horizon_recip(k), k = 1..kmax, on a small solver. With the flag on its eight parts must be 1/k's greedy
round-to-nearest expansion, each part the f64 nearest to 1/k less the parts before it (mpmath at 1200 bits; float()
of an mpf rounds to nearest); with it off its parts must be factored_horizon.recip_int(k)'s two (#1035's transcription
of multidouble's double-double division) and zeros. Control --mutate=off: the flag-off parts in place of the flag-on
ones, which must fail at every k that is not a power of two.

miss: the horizon block's rows of the kernel in octo-double (Solver.residual_o at the state's parts) with the flag on
less those with it off, dR_hp, against the transcription's old rows with an exact 1/k less those with hp's recip_int,
dR_py (factored_horizon.series at --prec bits, the p rows projected on the harmonics k <= kfreeze as the kernel's
frozen rows are; the held and frozen offsets cancel in the difference). Pass: max |dR_hp - dR_py| <= tol max |dR_py|,
tol = 1e-6: far above both computations' rounding (the rows' octo-double floor and 2^-prec, each times the mass
mode's gain) and below order 3's share of the miss. Control --mutate=planted: the kernel's flag-on rows with 1/k at
order --k (3 by default) also scaled by 1 + 2^-106 (Solver.set_exact_recip's planted change), which must fail. The miss
comes from the lowest orders whose 1/k is not dyadic: at D 242.36 the planted change moves the rows by 3.3e-6 at k = 3,
1.2e-10 at 5 and below 1e-20 from 10 on. The controls first tried, at orders 60 and ks, moved them by 1e-16 and 3e-33
of the miss and did not fire (notes/EXACT_RECIP.md 2).

plant: the planted check of #1043. The kernel's rows with the flag on and 1/K scaled by 1 + 2^-106 less those without,
against the transcription's rows with an exact 1/k and 1/K scaled the same less those without, for each K of --k.
Pass as for miss, at every K. Control --mutate=shift: the transcription scales 1/(K+1) instead, which must fail.

delta: Delta of two stored states (their parts summed exactly at 600 bits) and its change OFF -> ON, with the three rows
of --csv nearest in D and their quoted errors.

bits: the flag off leaves the octo-double path bit for bit, which highm_check.py's ref, scaled and jets do not run:
SHA-256 of every output's float64 bytes at STATE with the default solver (residual, residual_q and residual_o at the
state's parts, residual_jet_o to order 2, newton_o's step and pivots, and series_o's first three orders from that
LU). --save with a build of main, --compare with this branch's build: exit 1 on any differing hash. Control
--mutate=on: the flag on (set_exact_recip(1)) with --compare, which must change the hashes of the outputs that read
the horizon series in quad- or octo-double.

gain: the transcription alone, at an exact 1/k: max |dR| for 1/K times 1 + eps (eps = 2^-106, 2^-110, 2^-120,
2^-140; linear if max |dR| / eps is the same), for recip_int at K alone, and for 1 + 2^-106 at orders K..ks + 1 and at
every order. It prints; it checks nothing.

predict: Delta's change from the flag, predicted to first order by a second solve path. At the flag-off solution
OFF (a polished base), r = F_exact(OFF) is the kernel's flag-on residual in octo-double and J its flag-on
octo-double Jacobian (Solver.jacobian_o, eight parts). A state with held harmonics (kfreeze < 13) takes r =
F_exact(OFF) - F_recip_int(OFF) instead, in which the held offset cancels; F_recip_int(OFF) is the flag-off polish's
floor there (its max is printed). The row of J^-1 for Delta is
y with J^T y = e_Delta, solved by FLINT's arb_mat at --prec bits (approximate LU), not by hp's LUs, and the
prediction is dDelta = -y^T r in arb. Measured: Delta(ON) - Delta(OFF), each the exact sum of its parts. Pass:
|dDelta - measured| / |measured| < 1% (criterion fixed before the first run, #1043), and ||y||_1 max|r| >= |measured|.
Controls, which must fail: --mutate=offr takes r from the flag-off residual (no held harmonics only), and
--mutate=prow the row of J^-1 for unknown 0 (a p1 point) in place of Delta's.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402

import factored_horizon as fh  # noqa: E402

PLANT = mp.mpf(2) ** -106      # the planted relative change of 1/k (model.rs RECIP_PLANT)


def kernel(meta, threads, mode, planted=0):
    """a solver of the state's layout with Solver.set_exact_recip(mode, planted); its horizon rows need no plan"""
    from hpkernel.driver import make_solver
    D = meta.get("D_str", meta["D"])
    s = make_solver(D, int(meta["M"]), nodes=list(meta["nodes"]), kt=meta.get("kt", 36), kc=meta.get("kc", 146),
                    ks=meta.get("ks", 130), threads=threads, kfreeze=meta.get("kfreeze"))
    if mode or planted:
        if not hasattr(s, "set_exact_recip"):
            raise SystemExit("this hpkernel has no Solver.set_exact_recip (built before #1043); rebuild hp")
        s.set_exact_recip(int(mode), int(planted))
    return s


def parts8(path):
    """the state's parts as eight contiguous arrays, the missing lower ones zero"""
    d = np.load(path, allow_pickle=False)
    keys = ["zh", "zl"] + [f"z{i}" for i in range(2, 8)]
    zero = np.zeros(len(d["zh"]))
    return [np.ascontiguousarray(np.asarray(d[k], dtype=float)) if k in d.files else zero for k in keys]


def hp_rows(meta, z8, threads, mode, planted=0):
    """the kernel's horizon block (4M rows) in octo-double, each row its eight parts summed exactly"""
    M, K = int(meta["M"]), len(meta["nodes"])
    F = kernel(meta, threads, mode, planted).residual_o(z8)
    lo, hi = 4 * M * K, 4 * M * (K + 1)
    with mp.workprec(600):
        return [mp.fsum(mp.mpf(float(np.asarray(p)[i])) for p in F) for i in range(lo, hi)]


class Transcription:
    """factored_horizon's old series at a stored state: its horizon rows for a choice of 1/k"""

    def __init__(self, path, prec):
        mp.mp.prec = prec
        self.meta, z = fh.load_state(path)
        self.seg = fh.seg_of(self.meta)
        self.mh, self.ah, delta, self.node = fh.horizon_data(self.meta, z)
        self.om = self.seg.omega(delta)
        kf = self.meta.get("kfreeze")
        self.P = fh.plow_odd(self.seg.M, int(kf)) if kf else None

    def rows(self, recip, planted=0):
        """the rows p (projected on k <= kfreeze), m, a, c with 1/k as recip ('dd': hp's recip_int; 'exact': 1/k at
        the working precision), 1/planted times 1 + 2^-106"""
        orig = fh.recip_int
        base = orig if recip == "dd" else (lambda k: 1 / mp.mpf(k))
        # factored_horizon.series reads the module's recip_int when its recip is True
        fh.recip_int = lambda k: base(k) * (1 + PLANT) if k == planted else base(k)
        try:
            s = fh.series(self.seg, self.mh, self.om, a0=self.ah, recip=True)
        finally:
            fh.recip_int = orig
        R, _ = fh.rows_old(self.seg, s, self.node)
        M = self.seg.M
        if self.P is not None:
            R = [mp.fdot(self.P[i], R[:M]) for i in range(M)] + R[M:]
        return R


def compare(dhp, dpy, M, tol):
    """max |dhp - dpy| over max |dpy|, the fields' max |dpy|, and the pass"""
    scale = max(abs(v) for v in dpy)
    err = max(abs(a - b) for a, b in zip(dhp, dpy)) / scale if scale else mp.inf
    fields = {f: float(max(abs(v) for v in dpy[i * M:(i + 1) * M])) for i, f in enumerate("pmac")}
    return float(err), float(scale), fields, bool(err <= tol)


def cmd_recip(kmax, mutate):
    from hpkernel.driver import make_solver
    s = make_solver(4, 8, nodes=[0.1, 0.7], kt=8, kc=8, ks=8, threads=1)
    if not hasattr(s, "set_exact_recip"):
        raise SystemExit("this hpkernel has no Solver.set_exact_recip (built before #1043); rebuild hp")
    hexes = lambda v: [x.hex() for x in v]  # noqa: E731
    bad_on, bad_off, pow2 = [], [], 0
    with mp.workprec(1200):
        for k in range(1, kmax + 1):
            s.set_exact_recip(0)
            off = s.horizon_recip(k)
            s.set_exact_recip(1)
            # --mutate=off: the flag-off parts in place of the flag-on ones
            on = off if mutate == "off" else s.horizon_recip(k)
            want, r = [], 1 / mp.mpf(k)
            for _ in range(8):
                want.append(float(r))
                r -= mp.mpf(want[-1])
            fh.recip_int(k)
            want_off = list(fh._RECIP[k]) + [0.0] * 6
            if hexes(on) != hexes(want):
                bad_on.append(k)
            if hexes(off) != hexes(want_off):
                bad_off.append(k)
            pow2 += (k & (k - 1)) == 0
    s.set_exact_recip(0)
    ok = not bad_on and not bad_off
    print(f"recip k = 1..{kmax}{' (--mutate=off: the flag-off parts as the flag-on ones)' if mutate else ''}: flag on, "
          f"{len(bad_on)} k not 1/k's greedy eight parts{' ' + str(bad_on[:10]) if bad_on else ''} ({kmax - pow2} k are "
          f"not powers of two); flag off, {len(bad_off)} k not factored_horizon.recip_int's"
          f"{' ' + str(bad_off[:10]) if bad_off else ''}: {'PASS' if ok else 'FAIL'}")
    return dict(cmd="recip", kmax=kmax, mutate=mutate, bad_on=len(bad_on), bad_on_first=bad_on[:50], bad_off=len(bad_off),
                bad_off_first=bad_off[:50], not_pow2=kmax - pow2, ok=ok), ok


def cmd_miss(path, prec, tol, threads, mutate, k):
    t0 = time.time()
    tr = Transcription(path, prec)
    meta, M = tr.meta, tr.seg.M
    z8 = parts8(path)
    off = hp_rows(meta, z8, threads, 0)
    on = hp_rows(meta, z8, threads, 1, k if mutate == "planted" else 0)
    dhp = [a - b for a, b in zip(on, off)]
    r_dd, r_ex = tr.rows("dd"), tr.rows("exact")
    dpy = [a - b for a, b in zip(r_ex, r_dd)]
    err, scale, fields, ok = compare(dhp, dpy, M, tol)
    rec = dict(cmd="miss", state=os.path.basename(path), D=tr.seg.D_str, ks=tr.seg.K, prec=prec, tol=tol, mutate=mutate,
               k=k if mutate else None, miss=scale, fields=fields, err=err, ok=ok, G=float(tr.seg.G),
               off_rows=float(max(abs(v) for v in off[M:])), seconds=round(time.time() - t0, 1))
    print(f"miss {os.path.basename(path)} D {tr.seg.D_str[:12]} ks {tr.seg.K} ({prec} bits{', --mutate=planted at k ' + str(k) if mutate else ''}): "
          f"the transcription's rows, exact 1/k less recip_int, max {scale:.3e} (p {fields['p']:.2e}, m {fields['m']:.2e}, "
          f"a {fields['a']:.2e}, c {fields['c']:.2e}); the kernel's, flag on less off, differ by {err:.2e} of it "
          f"(tol {tol:g}): {'PASS' if ok else 'FAIL'}; the kernel's flag-off m, a, c rows max {rec['off_rows']:.2e} ({rec['seconds']} s)")
    return rec, ok


def cmd_plant(path, prec, tol, threads, mutate, ks_):
    t0 = time.time()
    tr = Transcription(path, prec)
    meta, M = tr.meta, tr.seg.M
    z8 = parts8(path)
    on = hp_rows(meta, z8, threads, 1)
    r_ex = tr.rows("exact")
    out, ok_all = [], True
    for K in ks_:
        dhp = [a - b for a, b in zip(hp_rows(meta, z8, threads, 1, K), on)]
        Kp = K + 1 if mutate == "shift" else K
        dpy = [a - b for a, b in zip(tr.rows("exact", Kp), r_ex)]
        err, scale, fields, ok = compare(dhp, dpy, M, tol)
        ok_all &= ok
        out.append(dict(K=K, K_transcription=Kp, move=scale, fields=fields, err=err, ok=ok))
        print(f"plant {os.path.basename(path)} D {tr.seg.D_str[:12]} K {K}{' (transcription at ' + str(Kp) + ')' if mutate else ''}: "
              f"the transcription's rows move by {scale:.3e} (p {fields['p']:.2e}, m {fields['m']:.2e}, a {fields['a']:.2e}, "
              f"c {fields['c']:.2e}); the kernel's differ by {err:.2e} of it (tol {tol:g}): {'PASS' if ok else 'FAIL'}", flush=True)
    rec = dict(cmd="plant", state=os.path.basename(path), D=tr.seg.D_str, ks=tr.seg.K, prec=prec, tol=tol, mutate=mutate,
               runs=out, ok=ok_all, G=float(tr.seg.G), seconds=round(time.time() - t0, 1))
    print(f"plant: {'PASS' if ok_all else 'FAIL'} ({rec['seconds']} s)")
    return rec, ok_all


def delta_of(path):
    """(Delta, D string, meta) of a stored state: the last entry of its parts summed exactly"""
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    keys = ["zh", "zl"] + [f"z{i}" for i in range(2, 8)]
    with mp.workprec(600):
        dl = mp.fsum(mp.mpf(float(d[k][-1])) for k in keys if k in d.files)
    return dl, str(meta.get("D_str", meta["D"])), meta


def cmd_delta(a, b, csv):
    with mp.workprec(600):
        da, Da, ma = delta_of(a)
        db, Db, mb = delta_of(b)
        diff = db - da
        # the dimensions compared at 600 bits, so that their low digits count
        same_D = mp.mpf(Da) == mp.mpf(Db)
    rows = []
    if csv and os.path.exists(csv):
        import csv as csvmod
        with open(csv) as fh_:
            for r in csvmod.DictReader(fh_):
                rows.append(dict(D=r["D"], kind=r["kind"], err=r["err"], err_kind=r["err_kind"], state=r.get("state", "")))
        rows = sorted((r for r in rows if r["err"]), key=lambda r: abs(float(r["D"]) - float(mp.mpf(Da))))[:3]
    print(f"delta: {os.path.basename(a)} (exact_recip {ma.get('exact_recip', 0)}) Delta {mp.nstr(da, 40)}")
    print(f"       {os.path.basename(b)} (exact_recip {mb.get('exact_recip', 0)}) Delta {mp.nstr(db, 40)}")
    print(f"       D {Da[:20]}{'' if same_D else ' and ' + Db[:20] + ' (NOT the same D)'}; Delta(b) - Delta(a) = {mp.nstr(diff, 6)}")
    for r in rows:
        print(f"       {csv} row D {r['D']}: {r['kind']}, quoted error {r['err']} ({r['err_kind']}), |change| / error "
              f"{float(abs(diff)) / float(r['err']):.2e}")
    rec = dict(cmd="delta", a=a, b=b, D_a=Da, D_b=Db, same_D=bool(same_D), Delta_a=mp.nstr(da, 45), Delta_b=mp.nstr(db, 45),
               change=float(diff), csv_rows=rows, ok=bool(same_D))
    return rec, bool(same_D)


def bits_of(path, threads, mode):
    """{output: sha256} at the state for the solver with set_exact_recip(mode); the state's own plan if it has one"""
    import hashlib
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    s = kernel(meta, threads, mode)
    if meta.get("segs"):
        s.set_segs(meta["segs"])
    z8 = parts8(path)
    h = lambda arrs: hashlib.sha256(b"".join(np.ascontiguousarray(np.asarray(a, dtype=float)).tobytes() for a in arrs)).hexdigest()  # noqa: E731
    out = {}
    out["residual"] = h(s.residual(z8[0], z8[1]))
    out["residual_q"] = h(s.residual_q(*z8[:4]))
    out["residual_o"] = h(s.residual_o(z8))
    out["residual_jet_o"] = h([p for f in s.residual_jet_o([z8], 3) for p in f])
    st = s.newton_o(z8)
    out["newton_o"] = h([np.asarray(v, dtype=float).ravel() for v in st])
    out["series_o"] = h([p for z in s.series_o(z8, 3) for p in z])
    return out, meta


def cmd_bits(path, threads, save, cmp_path, mutate):
    t0 = time.time()
    mode = 1 if mutate == "on" else 0
    out, meta = bits_of(path, threads, mode)
    rec = dict(cmd="bits", state=os.path.basename(path), D=str(meta.get("D_str", meta["D"])), exact_recip=mode, hashes=out,
               seconds=round(time.time() - t0, 1))
    if save:
        with open(save, "w") as f:
            json.dump(rec, f, indent=1)
        print(f"bits {os.path.basename(path)}: {len(out)} hashes saved to {save} ({rec['seconds']} s)")
        return rec, True
    with open(cmp_path) as f:
        ref = json.load(f)
    if ref["state"] != rec["state"]:
        raise SystemExit(f"bits: {cmp_path} is of {ref['state']}, not {rec['state']}")
    differ = sorted(k for k in out if out[k] != ref["hashes"].get(k))
    ok = not differ
    rec["differ"] = differ
    print(f"bits {os.path.basename(path)} (exact_recip {mode}) against {os.path.basename(cmp_path)}: "
          f"{len(differ)} of {len(out)} outputs differ {differ}: {'PASS' if ok else 'FAIL'} ({rec['seconds']} s)")
    return rec, ok


def cmd_gain(path, prec, K):
    tr = Transcription(path, prec)
    base = tr.rows("exact")

    def move(f):
        orig = fh.recip_int
        fh.recip_int = f
        try:
            R = tr.rows("dd")      # "dd" reads fh.recip_int, here f
        finally:
            fh.recip_int = orig
        return max(abs(a - b) for a, b in zip(R, base))

    ex, ri, out = (lambda k: 1 / mp.mpf(k)), fh.recip_int, {}
    for e in (-106, -110, -120, -140):
        d = move(lambda k, e=e: ex(k) * (1 + mp.mpf(2) ** e) if k == K else ex(k))
        out[f"eps 2^{e}"] = (float(d), float(d / mp.mpf(2) ** e))
    out["recip_int at K alone"] = (float(move(lambda k: ri(k) if k == K else ex(k))), None)
    for lab, lo in ((f"2^-106 at orders {K}..ks+1", K), ("2^-106 at every order", 1)):
        out[lab] = (float(move(lambda k, lo=lo: ex(k) * (1 + PLANT) if k >= lo else ex(k))), None)
    print(f"gain {os.path.basename(path)} D {tr.seg.D_str[:12]} ks {tr.seg.K} K {K} ({prec} bits), max |dR| against an exact 1/k:")
    for lab, (d, g) in out.items():
        print(f"  {lab}: {d:.4e}" + (f" ({g:.4e} per unit relative change)" if g is not None else ""))
    return dict(cmd="gain", state=os.path.basename(path), D=tr.seg.D_str, ks=tr.seg.K, K=K, prec=prec, moves=out), True


def cmd_predict(off_path, on_path, threads, prec, mutate):
    import flint
    t0 = time.time()
    meta = json.loads(str(np.load(off_path, allow_pickle=False)["meta"]))
    held = int(meta.get("kfreeze") or 0) < 13
    if held and mutate == "offr":
        raise SystemExit("predict --mutate=offr: a state with held harmonics has no flag-off r without the held offset")
    z8 = parts8(off_path)
    L = len(z8[0])
    iD = L - 1
    row = 0 if mutate == "prow" else iD
    sv = {}
    for mode in (0, 1):
        sv[mode] = kernel(meta, threads, mode)
        if meta.get("segs"):
            sv[mode].set_segs(meta["segs"])
    F_on = [np.asarray(p, dtype=float) for p in sv[1].residual_o(z8)]
    F_off = [np.asarray(p, dtype=float) for p in sv[0].residual_o(z8)]
    t1 = time.time()
    J = [np.asarray(p, dtype=float) for p in sv[1].jacobian_o(z8)]
    t_jac = time.time() - t1
    cond = float(np.linalg.cond(J[0]))
    old = flint.ctx.prec
    flint.ctx.prec = prec
    try:
        t1 = time.time()
        nz = [q for q in range(8) if np.any(J[q])]
        # J^T, row-major: entry (i, j) is J[j, i], its octo-double parts summed exactly
        JT = [Jq.T.ravel() for Jq in (J[q] for q in nz)]
        ent = [flint.arb(0)] * (L * L)
        for idx in range(L * L):
            v = flint.arb(JT[0][idx])
            for Jq in JT[1:]:
                x = Jq[idx]
                if x != 0.0:
                    v += flint.arb(x)
            ent[idx] = v
        A = flint.arb_mat(L, L, ent)
        e = flint.arb_mat(L, 1, [1 if i == row else 0 for i in range(L)])
        y = A.solve(e, algorithm="approx")
        res = A * y - e
        solve_res = max(abs(float(res[i, 0].mid())) for i in range(L))
        t_solve = time.time() - t1
        def sum8(F, i):
            return sum((flint.arb(float(F[q][i])) for q in range(8)), flint.arb(0))
        if held:
            r = [sum8(F_on, i) - sum8(F_off, i) for i in range(L)]
        else:
            r = [sum8(F_off if mutate == "offr" else F_on, i) for i in range(L)]
        foff = max(abs(float(sum8(F_off, i).mid())) for i in range(L))
        pred = -sum((y[i, 0] * r[i] for i in range(L)), flint.arb(0))
        y1 = sum((abs(y[i, 0]) for i in range(L)), flint.arb(0))
        rmax = max(abs(float(r[i].mid())) for i in range(L))
        pred_f, y1_f = float(pred.mid()), float(y1.mid())
    finally:
        flint.ctx.prec = old
    with mp.workprec(600):
        da, _, _ = delta_of(off_path)
        db, _, _ = delta_of(on_path)
        meas = float(db - da)
        pa, pb = (mp.fsum(mp.mpf(float(np.asarray(z)[0])) for z in parts8(x)) for x in (off_path, on_path))
        meas_row = float(pb - pa)
    rel = abs(pred_f - meas) / abs(meas)
    ok = rel < 0.01 and y1_f * rmax >= abs(meas)
    name = "Delta" if row == iD else "unknown 0 (p1)"
    rec = dict(cmd="predict", off=off_path, on=on_path, prec=prec, mutate=mutate, L=L, row=row, held=held,
               cond_f64=cond, solve_residual=solve_res, y_l1=y1_f, r_max=rmax, F_off_max=foff, bound=y1_f * rmax, predicted=pred_f,
               measured=meas, measured_row=meas_row, rel=rel, ok=ok, t_jac=round(t_jac, 1),
               t_solve=round(t_solve, 1), seconds=round(time.time() - t0, 1))
    print(f"predict {os.path.basename(off_path)} (L {L}, {prec} bits{', --mutate=' + mutate if mutate else ''}"
          f"{', r = F_exact - F_recip_int (held harmonics)' if held else ''}): J's f64 condition {cond:.2e}; J^T y = "
          f"e({name}) solved by arb_mat, max |J^T y - e| {solve_res:.2e}; ||y||_1 {y1_f:.4e}, max|r| {rmax:.4e}, their "
          f"product {y1_f * rmax:.4e}; max|F_recip_int| at OFF {foff:.2e}")
    print(f"  predicted -y^T r = {pred_f:.6e}; measured Delta(ON) - Delta(OFF) = {meas:.6e}"
          + (f" (the row's own change {meas_row:.6e})" if row != iD else "")
          + f"; |pred - measured| / |measured| = {rel:.3e} (bound 1e-2): {'PASS' if ok else 'FAIL'} "
          f"(Jacobian {t_jac:.0f} s, arb solve {t_solve:.0f} s)")
    return rec, ok


def main(argv):
    args, opt = [], {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            k, eq, v = a[2:].partition("=")
            if not eq:
                v = argv[i + 1]
                i += 1
            opt[k] = v
        else:
            args.append(a)
        i += 1
    if not args:
        print(__doc__)
        return 2
    known = {"kmax", "mutate", "prec", "tol", "threads", "k", "csv", "out", "save", "compare"}
    if set(opt) - known:
        raise SystemExit(f"exact_recip_check.py: unknown options {sorted(set(opt) - known)}")
    cmd, mutate = args[0], opt.get("mutate", "")
    prec, tol, threads = int(opt.get("prec", 288)), float(opt.get("tol", 1e-6)), int(opt.get("threads", 4))
    if cmd == "recip" and mutate in ("", "off"):
        rec, ok = cmd_recip(int(opt.get("kmax", 4096)), mutate)
    elif cmd == "miss" and len(args) == 2 and mutate in ("", "planted"):
        rec, ok = cmd_miss(args[1], prec, tol, threads, mutate, int(opt.get("k", 3)))
    elif cmd == "plant" and len(args) == 2 and mutate in ("", "shift"):
        rec, ok = cmd_plant(args[1], prec, tol, threads, mutate, [int(x) for x in opt.get("k", "3,5,7,30,60,100").split(",")])
    elif cmd == "bits" and len(args) == 2 and mutate in ("", "on") and bool(opt.get("save")) != bool(opt.get("compare")) \
            and not (mutate and opt.get("save")):
        rec, ok = cmd_bits(args[1], threads, opt.get("save"), opt.get("compare"), mutate)
    elif cmd == "gain" and len(args) == 2 and "k" in opt:
        rec, ok = cmd_gain(args[1], prec, int(opt["k"]))
    elif cmd == "predict" and len(args) == 3 and mutate in ("", "offr", "prow"):
        rec, ok = cmd_predict(args[1], args[2], threads, int(opt.get("prec", 600)), mutate)
    elif cmd == "delta" and len(args) == 3:
        rec, ok = cmd_delta(args[1], args[2], opt.get("csv", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "delta_all.csv")))
    else:
        print(__doc__)
        return 2
    rec["mpmath"] = mp.__version__
    if opt.get("out"):
        with open(opt["out"], "w") as f:
            json.dump(rec, f, indent=1, default=str)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

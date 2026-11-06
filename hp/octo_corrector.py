"""The octo-double corrector of the large-D chain (notes/OCTO_DOUBLE.md 5): the pinned system's Newton in octo-double
(eight f64 parts, Solver.residual_o, newton_o and olu_solve, M = 32 only), with the horizon data eliminated by the
quad-double projection above its switch, as the quad-double corrector (hpkernel.driver.newton_qd on pinning.PinnedQ,
reseed.ElimQ) does one level down. anm_run.py --corrector od runs it.

PinnedO: the held harmonics' offset c (PinnedSolver's ch, cl) on the octo-double methods: F - c, and the step
-J^-1 (F - c) = dz + J^-1 c through the octo-double LU. newton_od: newton_qd's 'full' mode in octo-double (a trial
z + lam dz is accepted when its own correction is smaller than dz, or its max |F| <= tol). ElimO: ElimQ for
eight-part iterates. save_o, load_o: eight-part states in save_q's layout plus four more parts, which load_q reads as
the first four.

The octo-double helpers (od_add_dd, od_axpy) are multidouble's from f092018 on; they are imported where they are
used, so that a module importing this one runs on the older packages the runners have.

    python3 octo_corrector.py planted RAW STATE.npz [--threads 3] [--rel 1e-12] [--seed 1] [--qd 1] [--Delta_ref D]
        [--mutate_offset qd] [--mutate 10] [--out PREFIX]
    python3 octo_corrector.py wall RAW PRED.npz [--threads 3] [--qd 0] [--F_max 1e-36] [--mutate_offset qd]
        [--mutate 10] [--out PREFIX]

`planted` is check (a) of notes/OCTO_DOUBLE.md 5 (a converged state moved by a known amount, corrected in
octo-double and in quad-double), `wall` check (b) (a stored prediction where the quad-double corrector stalls). The
controls --mutate_offset qd and --mutate 10 must make them fail. Each exits 1 when its check fails."""
import json

import numpy as np
from hpkernel.driver import qmax, qparts
from multidouble import qd_add_dd


def o8(z):
    """an octo-double vector's eight parts as contiguous float64 arrays"""
    return tuple(np.ascontiguousarray(np.asarray(p, dtype=float)) for p in z)


def o8_of_q(zq):
    """a quad-double vector's four parts promoted to eight (the lower four zero): the same values exactly"""
    zero = np.zeros(len(zq[0]))
    return o8(tuple(zq) + (zero, zero, zero, zero))


def q4_of_o(zo):
    """an octo-double vector rounded to quad-double: its first four parts plus the rest's sum (float64, from the
    smallest up), renormalized by qd_add_dd, as OD::to_qd does; the rounding is about 2^-212 of each entry"""
    zo = o8(zo)
    tail = zo[4] + (zo[5] + (zo[6] + zo[7]))
    return tuple(np.ascontiguousarray(np.asarray(p)) for p in qd_add_dd(zo[0], zo[1], zo[2], zo[3], tail, np.zeros(len(tail))))


def omax(F):
    """max |F| of a residual in parts (two, four or eight), to f64 accuracy"""
    return float(np.abs(np.asarray(F[0]) + np.asarray(F[1])).max())


def save_o(path, z, meta):
    """an eight-part state: save_q's keys (zh, zl, z2, z3: load_q reads the first four parts) and z4..z7"""
    z = o8(z)
    np.savez(path, zh=z[0], zl=z[1], z2=z[2], z3=z[3], z4=z[4], z5=z[5], z6=z[6], z7=z[7], meta=json.dumps(meta))


def load_o(path):
    """(z, meta) with z eight parts; a state saved by save_q or save has zero lower parts"""
    d = np.load(path)
    z = qparts(d["zh"], d["zl"], d["z2"] if "z2" in d else None, d["z3"] if "z3" in d else None)
    zero = np.zeros(len(z[0]))
    z = tuple(z) + tuple(np.asarray(d[k]) if k in d else zero for k in ("z4", "z5", "z6", "z7"))
    return o8(z), json.loads(str(d["meta"]))


def has_o(path):
    """whether a stored state carries eight parts (save_o)"""
    return "z7" in np.load(path)


class PinnedO:
    """PinnedQ's offset for the octo-double methods: the kernel's frozen rows read coefficient - held value, so the
    octo-double Newton converges to the pinned solution. ch, cl: the PinnedSolver's offset. residual_o returns F - c in
    eight parts. newton_o returns (dz, F hi, F lo, pivots) as Solver.newton_o does, with dz + J^-1 c in octo-double and
    F - c formed from an octo-double residual_o: the kernel's F there is rounded to double-double, 2^-106 |c| ~ 1e-35
    on the held rows (|c| ~ 8e-4), above the tolerance. mutate_offset (the control): 'qd' adds J^-1 c through the
    quad-double LU at the point's quad-double rounding instead of the octo-double one."""

    def __init__(self, s, ch, cl, mutate_offset=""):
        self.s = s
        self.ch, self.cl = np.ascontiguousarray(ch), np.ascontiguousarray(cl)
        self.zero = np.zeros(len(self.ch))
        self.mutate_offset = mutate_offset

    def __getattr__(self, name):
        return getattr(self.s, name)

    def residual_o(self, z, mutate=0):
        from multidouble import od_add_dd
        F = self.s.residual_o(list(o8(z)), mutate=mutate)
        return o8(od_add_dd(*o8(F), -self.ch, -self.cl))

    def newton_o(self, z, mutate=0):
        from multidouble import od_axpy
        z = o8(z)
        d, _, _, pmin, pmax = self.s.newton_o(list(z), mutate=mutate)
        if self.mutate_offset == "qd":
            self.s.newton_q(*q4_of_o(z), mutate=mutate)
            x = o8_of_q(self.s.qlu_solve(self.ch, self.cl, self.zero, self.zero))
        else:
            x = o8(self.s.olu_solve([self.ch, self.cl] + [self.zero] * 6))        # J^-1 c
        d = o8(od_axpy(1.0, *x, *o8(d)))
        F = self.residual_o(z, mutate=mutate)
        return d, F[0], F[1], pmin, pmax


def newton_od(s, z, tol=1e-40, it_max=12, max_halvings=8, verbose=True, mutate=0):
    """Newton in octo-double from the eight-part state z on s (PinnedO, or ElimO around it): newton_qd's mode 'full'
    one level up (hpkernel.driver.newton_qd, its acceptance rule). A trial z + lam dz (lam = 1, 1/2, ...,
    2^-max_halvings) is accepted when the Newton correction there, with its own Jacobian, is smaller than dz (max norms
    of the leading parts), or when its max |F| <= tol. Stops when max |F| <= tol (converged), when no trial is accepted
    (stalled), or after it_max steps. A trial whose Jacobian is not finite or singular is rejected. Returns (z, hist,
    converged), hist one record per step as newton_qd's."""
    import time
    from multidouble import od_axpy
    z = o8(z)
    st = s.newton_o(z, mutate=mutate)
    hist = []
    for it in range(it_max + 1):
        nf = omax(st[1:3])
        if nf <= tol:
            if verbose:
                print(f"  on {it}: |F| {nf:.2e} converged, Delta {z[0][-1]:.17f} {z[1][-1]:+.4e}", flush=True)
            hist.append(dict(it=it, F=nf, dz=float(np.abs(st[0][0]).max()), mode="full"))
            return z, hist, True
        if it == it_max:
            break
        d = st[0]
        nd = float(np.abs(d[0]).max())
        t0 = time.time()
        rec = dict(it=it, F=nf, dz=nd, dDelta=float(d[0][-1] + d[1][-1]), piv=[st[3], st[4]], mode="full")
        lam = 1.0
        for _ in range(max_halvings + 1):
            zt = o8(od_axpy(lam, *d, *z))
            try:
                stt = s.newton_o(zt, mutate=mutate)
                good = np.isfinite(stt[0][0]).all() and np.isfinite(stt[1]).all()
            except ValueError:          # a non-finite or singular Jacobian at the trial point
                good = False
            if good and (float(np.abs(stt[0][0]).max()) < nd or omax(stt[1:3]) <= tol):
                break
            lam *= 0.5
        else:
            rec.update(lam=0.0, seconds=round(time.time() - t0, 2))
            hist.append(rec)
            if verbose:
                print(f"  on {it}: |F| {nf:.2e}, |dz| {nd:.1e}: no trial step accepted, stopping", flush=True)
            return z, hist, False
        z, st = zt, stt
        rec.update(lam=lam, F_new=omax(st[1:3]), dz_new=float(np.abs(st[0][0]).max()), seconds=round(time.time() - t0, 2))
        hist.append(rec)
        if verbose:
            print(f"  on {it}: |F| {nf:.2e} -> {rec['F_new']:.2e}, |dz| {nd:.1e} -> {rec['dz_new']:.1e}, lam {lam:g}, "
                  f"dDelta {rec['dDelta']:+.2e}, piv {st[3]:.1e}..{st[4]:.1e}, {rec['seconds']} s", flush=True)
    return z, hist, False


def with_horizon(z8, z4, rs):
    """z8 (eight parts) with its horizon data (m_h, a_h) taken from z4 (four parts), their parts 4-7 zero: a re-seed or
    projection (rs, a reseed.Reseeder of the layout, changes only those) put back on an octo-double point"""
    sl = slice(rs.off["mh"], rs.off["mh"] + 2 * rs.M)
    out = [np.array(p, dtype=float) for p in o8(z8)]
    for q in range(8):
        out[q][sl] = np.asarray(z4[q])[sl] if q < 4 else 0.0
    return o8(out)


class ElimO:
    """reseed.ElimQ for eight-part iterates: residual_o and newton_o project the point first while its pinned max |F|
    exceeds switch, and newton_o's step includes the projection's displacement. The projection is the quad-double one
    (Reseeder.project at the point's quad-double rounding); its horizon data are put back in eight parts with parts
    4-7 zero, the other unknowns keeping all eight. Below switch the point is not projected (plain octo-double Newton
    finishes). start_F: a start whose projection fails at max |F| above start_F raises reseed.StartNotProjected
    (None: never), as ElimQ. The switch test is the pinned quad-double residual of the rounding (PinnedQ), far above
    its floor there."""

    def __init__(self, so, rs, switch=1e-15, start_F=1e-6):
        from pinning import PinnedQ
        self.so, self.rs, self.switch, self.start_F = so, rs, switch, start_F
        self.sq = PinnedQ(so.s, so.ch, so.cl)
        self.counts = dict(u0=0, u0q=0, ladder=0, failed=0, skipped=0)
        self.first = True

    def __getattr__(self, name):
        return getattr(self.so, name)

    def project(self, z):
        """z (eight parts) projected when its pinned max |F| exceeds switch (the point ElimO's residual is that of),
        else z; counted"""
        from reseed import StartNotProjected
        first, self.first = self.first, False
        z = o8(z)
        z4 = q4_of_o(z)
        nf = qmax(self.sq.residual_q(*z4))
        if nf <= self.switch:
            self.counts["skipped"] += 1
            return z
        zp, how = self.rs.project(z4)
        self.counts[how] += 1
        if how == "failed" and first and self.start_F is not None and nf > self.start_F:
            raise StartNotProjected(dict(self.counts))
        return z if how == "failed" else with_horizon(z, zp, self.rs)

    def residual_o(self, z, mutate=0):
        return self.so.residual_o(self.project(z), mutate=mutate)

    def newton_o(self, z, mutate=0):
        from multidouble import od_axpy
        z = o8(z)
        zp = self.project(z)
        st = self.so.newton_o(zp, mutate=mutate)
        disp = o8(od_axpy(-1.0, *z, *zp))                         # z' - z
        return (o8(od_axpy(1.0, *disp, *st[0])), *st[1:])


def pinned_o(meta, z, threads=2, plan_tol=1e-32, mutate_offset=""):
    """reseed.pinned's solver at the state's D, with the segments planned at z at plan_tol, and its PinnedO wrapper:
    (s, PinnedSolver, PinnedQ, PinnedO)"""
    import mpmath as mp
    from hpkernel.driver import make_solver
    from pinning import PinnedQ, PinnedSolver, band_coeffs, band_field
    M = int(meta["M"])
    D = mp.mpf(meta["D_str"])
    s = make_solver(D, M, nodes=meta["nodes"], kt=meta.get("kt", 36), kc=meta.get("kc", 146), ks=meta.get("ks", 130),
                    threads=threads, kfreeze=meta["kfreeze"])
    s.set_segs(s.plan(np.asarray(z[0]), np.asarray(z[1]), plan_tol, 0.05))
    ks_held = [k for k in range(int(meta["kfreeze"]) + 1, 14) if k % 2 == 1]
    hh, hl = band_field(band_coeffs(np.asarray(z[0])[:M], np.asarray(z[1])[:M], M, ks_held), ks_held, M)
    sp = PinnedSolver(s, M, hh, hl)
    return s, sp, PinnedQ(s, sp.ch, sp.cl), PinnedO(s, sp.ch, sp.cl, mutate_offset=mutate_offset)



# ------------------------------------------------------------------ the checks (notes/OCTO_DOUBLE.md 5)
def _delta(z):
    """Delta (the last unknown) summed from its parts in mpmath, 40 digits"""
    import mpmath as mp
    with mp.workdps(140):
        return sum((mp.mpf(float(p[-1])) for p in z), mp.mpf(0))


def _parse(argv, opt):
    for k in list(opt):
        if "--" + k in argv:
            i = argv.index("--" + k)
            opt[k] = type(opt[k])(argv[i + 1])
    return opt


def _run_od(so, rs, z8, opt):
    """ElimO around so (PinnedO), newton_od to opt's tol and it_max; the final point (its projection, the point whose
    residual converged), the history, converged, the octo-double pinned max |F| there and the projections"""
    el = ElimO(so, rs)
    z, hist, conv = newton_od(el, z8, tol=opt["tol"], it_max=opt["it_max"], mutate=opt["mutate"])
    z = el.project(z)
    return z, hist, conv, omax(so.residual_o(z)), dict(el.counts)


def _run_qd(sq, so, rs, z4, tol, it_max):
    """the quad-double corrector of anm_run.py --corrector qd --reseed 2 (ElimQ, newton_qd, the last iterate
    projected) from z4; its final point, history, converged and the octo-double pinned max |F| there"""
    from hpkernel.driver import newton_qd
    from reseed import ElimQ
    el = ElimQ(sq, rs)
    z, hist, conv = newton_qd(el, z4, tol=tol, it_max=it_max, verbose=True)
    z = rs.project(z)[0]
    return o8(z), hist, conv, omax(so.residual_o(o8_of_q(z))), dict(el.counts)


def planted(argv):
    """check (a): the converged chain state RAW/STATE with its centre data and nodes (not m_h, a_h, Delta) moved by
    rel r relative, r uniform in +-[0.5, 1] (--seed), is the start. Pass, fixed before the first run: the octo-double
    corrector (ElimO, newton_od, tol 1e-40) converges with the octo-double pinned max |F| below 1e-40, and its Delta
    agrees with the quad-double corrector's from the same start (ElimQ, newton_qd to tol 1e-45 in 12 steps, its
    floor) to 1e-30. The pinned system is the state's own (held harmonics at its values, the plan at it at 1e-32).
    Controls, which must fail: --mutate_offset qd (PinnedO's J^-1 c through the quad-double LU), --mutate 10 (the
    kernel's OD_MUT_STATE_QD, the state read in its quad-double part). --qd 0 skips the quad-double run; --Delta_ref
    then gives the quad-double Delta to compare with (a passing run's), else only the residual is judged. Exit 1 on
    failure."""
    import os
    import time
    import mpmath as mp
    from hpkernel.driver import load_q, qd_axpy
    from reseed import Reseeder
    raw, path = argv[0], argv[1]
    opt = _parse(argv[2:], dict(threads=3, rel=1e-12, seed=1, mutate_offset="", mutate=0, it_max=12, qd=1, tol=1e-40, out="",
                                Delta_ref=""))
    z4, meta = load_q(os.path.join(raw, path))
    z4 = o8(z4)
    D = mp.mpf(meta["D_str"])
    s, sp, sq, so = pinned_o(meta, z4, threads=opt["threads"], mutate_offset=opt["mutate_offset"])
    rs = Reseeder(meta, D, threads=opt["threads"], qd=True)
    n = rs.off["mh"]
    rng = np.random.default_rng(opt["seed"])
    r = rng.uniform(0.5, 1.0, n) * rng.choice([-1.0, 1.0], n)
    dz = np.zeros(len(z4[0]))
    dz[:n] = opt["rel"] * r * z4[0][:n]
    zero = np.zeros(len(dz))
    zs = o8(qd_axpy(1.0, dz, zero, zero, zero, *z4))
    print(f"planted {path}: D {mp.nstr(D, 12)}, start = truth moved by {opt['rel']:.0e} relative in [0, {n}); pinned max|F| (octo) "
          f"truth {omax(so.residual_o(o8_of_q(z4))):.2e}, start {omax(so.residual_o(o8_of_q(zs))):.2e}; mutate_offset "
          f"'{opt['mutate_offset']}', mutate {opt['mutate']}", flush=True)
    t0 = time.time()
    zo, ho, co, Fo, counts = _run_od(so, rs, o8_of_q(zs), opt)
    Do = _delta(zo)
    print(f"  od: {len(ho)} steps, {'converged' if co else 'NOT converged'}, pinned max|F| (octo) {Fo:.2e}, Delta {mp.nstr(Do, 40)}, "
          f"projections {counts}, {time.time() - t0:.0f} s", flush=True)
    ok = bool(co) and Fo < 1e-40
    rec = dict(test="planted", state=path, D_str=meta["D_str"], rel=opt["rel"], seed=opt["seed"], mutate_offset=opt["mutate_offset"],
               mutate=opt["mutate"], od=dict(steps=len(ho), converged=bool(co), F=Fo, Delta=mp.nstr(Do, 40), hist=ho, counts=counts))
    print(f"  od against the truth: dDelta {mp.nstr(Do - _delta(z4), 5)}", flush=True)
    if opt["qd"]:
        t0 = time.time()
        zq, hq, cq, Fq, cntq = _run_qd(sq, so, rs, q4_of_o(zs), 1e-45, 12)
        Dq = _delta(zq)
        dD = abs(Do - Dq)
        print(f"  qd: {len(hq)} steps, {'converged' if cq else 'stopped'}, its last max|F| {hq[-1]['F']:.2e} (octo at the projected point "
              f"{Fq:.2e}), Delta {mp.nstr(Dq, 40)}, projections {cntq}, {time.time() - t0:.0f} s; |Delta_od - Delta_qd| {mp.nstr(dD, 5)}",
              flush=True)
        ok = ok and dD <= 1e-30
        rec["qd"] = dict(steps=len(hq), converged=bool(cq), F_last=hq[-1]["F"], F_octo=Fq, Delta=mp.nstr(Dq, 40), dDelta=float(dD))
    elif opt["Delta_ref"]:
        with mp.workdps(60):
            dD = abs(Do - mp.mpf(opt["Delta_ref"]))
        print(f"  |Delta_od - Delta_ref| {mp.nstr(dD, 5)} (Delta_ref {opt['Delta_ref']})", flush=True)
        ok = ok and dD <= 1e-30
        rec["Delta_ref"], rec["dDelta_ref"] = opt["Delta_ref"], float(dD)
    rec["passed"] = bool(ok)
    print(f"  (a) {'PASS' if ok else 'FAIL'}: od max|F| {Fo:.2e} < 1e-40" + (", |Delta_od - Delta_qd| <= 1e-30" if opt["qd"] or opt["Delta_ref"] else ""),
          flush=True)
    if opt["out"]:
        import json
        save_o(opt["out"] + ".npz", zo, dict(meta, D_str=meta["D_str"], planted=True))
        with open(opt["out"] + ".json", "w") as f:
            f.write(json.dumps(rec, default=float) + "\n")
    return 0 if ok else 1


def wall(argv):
    """check (b): a stored prediction RAW/PRED (anm_run.py --save_pred 1, four parts, before its re-seed), re-seeded
    (Reseeder, quad-double polish) and corrected by the octo-double corrector (ElimO, newton_od, tol 1e-40). Pass,
    fixed before the first run: the octo-double pinned max |F| of the result below 1e-36. The pinned system is that
    of reseed.pinned on the re-seeded prediction (its own held values, the plan at it at 1e-32). --qd 1 also runs the
    chain's quad-double corrector (ElimQ, newton_qd, tol 1e-28, 12 steps) from the same start, for its Delta.
    Controls as planted's. Exit 1 on failure."""
    import os
    import time
    import mpmath as mp
    from hpkernel.driver import load_q
    from reseed import Reseeder
    raw, path = argv[0], argv[1]
    opt = _parse(argv[2:], dict(threads=3, mutate_offset="", mutate=0, it_max=12, qd=0, tol=1e-40, out="", F_max=1e-36))
    z4, meta = load_q(os.path.join(raw, path))
    D = mp.mpf(meta["D_str"])
    rs = Reseeder(meta, D, threads=opt["threads"], qd=True)
    t0 = time.time()
    zr, diag = rs.reseed(o8(z4))
    zr = o8(zr)
    s, sp, sq, so = pinned_o(meta, zr, threads=opt["threads"], mutate_offset=opt["mutate_offset"])
    print(f"wall {path}: D {mp.nstr(D, 12)}, re-seed {'ok' if diag['ok'] else 'FAILED'} ({diag['seconds']} s), pinned max|F| (octo) "
          f"{omax(so.residual_o(o8_of_q(zr))):.2e}; mutate_offset '{opt['mutate_offset']}', mutate {opt['mutate']}", flush=True)
    zo, ho, co, Fo, counts = _run_od(so, rs, o8_of_q(zr), opt)
    Do = _delta(zo)
    print(f"  od: {len(ho)} steps, {'converged' if co else 'NOT converged'}, pinned max|F| (octo) {Fo:.2e}, Delta {mp.nstr(Do, 40)}, "
          f"projections {counts}, {time.time() - t0:.0f} s", flush=True)
    ok = Fo < opt["F_max"]
    rec = dict(test="wall", pred=path, D_str=meta["D_str"], mutate_offset=opt["mutate_offset"], mutate=opt["mutate"],
               od=dict(steps=len(ho), converged=bool(co), F=Fo, Delta=mp.nstr(Do, 40), hist=ho, counts=counts))
    if opt["qd"]:
        t0 = time.time()
        zq, hq, cq, Fq, cntq = _run_qd(sq, so, rs, q4_of_o(zr), 1e-28, 12)
        Dq = _delta(zq)
        print(f"  qd: {len(hq)} steps, {'converged' if cq else 'stopped'}, its last max|F| {hq[-1]['F']:.2e} (octo at the projected point "
              f"{Fq:.2e}), Delta {mp.nstr(Dq, 40)}, projections {cntq}, {time.time() - t0:.0f} s; Delta_od - Delta_qd {mp.nstr(Do - Dq, 5)}",
              flush=True)
        rec["qd"] = dict(steps=len(hq), converged=bool(cq), F_last=hq[-1]["F"], F_octo=Fq, Delta=mp.nstr(Dq, 40))
    rec["passed"] = bool(ok)
    print(f"  (b) {'PASS' if ok else 'FAIL'}: od max|F| {Fo:.2e} < {opt['F_max']:.0e}", flush=True)
    if opt["out"]:
        import json
        save_o(opt["out"] + ".npz", zo, dict(meta, D_str=meta["D_str"], prediction=False, corrected="od"))
        with open(opt["out"] + ".json", "w") as f:
            f.write(json.dumps(rec, default=float) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if len(sys.argv) > 3 and sys.argv[1] in ("planted", "wall"):
        sys.exit((planted if sys.argv[1] == "planted" else wall)(sys.argv[2:]))
    print(__doc__)
    sys.exit(2)

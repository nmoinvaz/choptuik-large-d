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

    python3 octo_corrector.py planted RAW STATE.npz [--threads 3] [--rel 1e-12] [--seed 1] [--qd 1] [--od 1]
        [--Delta_ref D] [--last 0] [--mutate_offset qd] [--mutate 10] [--out PREFIX]
    python3 octo_corrector.py wall RAW PRED.npz [--threads 3] [--qd 0] [--F_max 1e-36] [--mutate_offset qd]
        [--mutate 10] [--out PREFIX] [--reseed_o 0] [--held_o 0]
    python3 octo_corrector.py horizon RAW STATE.npz [--threads 3] [--steps 3] [--mutate_h sign]
    python3 octo_corrector.py digits [--mutate 2]

`planted` is check (a) of notes/OCTO_DOUBLE.md 5 (a converged state moved by a known amount, corrected in
octo-double and in quad-double), `wall` check (b) (a stored prediction where the quad-double corrector stalls). The
controls --mutate_offset qd and --mutate 10 must make them fail. `horizon` is the planted check of
Reseeder.refine_o (its docstring); --steps 0, --mutate_h sign and --mutate_h badlu_keep must make it fail, and
--mutate_h badlu (a poor first LU that refine_o refactors) must pass. Each exits 1 when its check
fails. `digits` checks delta_digits, the 40-digit Delta anm_run.py prints (#556); --mutate 2 must make it fail."""
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


def delta_digits(z, digits=40, parts=None):
    """Delta, the last unknown, as the exact sum of the last entries of all parts of z (or of the first `parts`), to
    `digits` significant digits (#556). The sum is taken at 64 x 53 bits, beyond any exponent spread that matters
    at 40 digits."""
    import mpmath as mp
    with mp.workprec(64 * 53):
        s = mp.fsum(mp.mpf(float(np.asarray(p)[-1])) for p in list(z)[:parts])
        return mp.nstr(s, digits, strip_zeros=False)


def digits_check(argv):
    """delta_digits on a planted eight-part value (140 digits split greedily into doubles) must give its first 40
    digits, and the first two parts alone must give a different value (the print reads more than double-double).
    One ulp of the last part does not show at 40 digits (it is about 1e-128 relative), which is printed. --mutate 2
    sums only two parts, and the check must then fail."""
    import mpmath as mp
    mutate = int(argv[argv.index("--mutate") + 1]) if "--mutate" in argv else None
    mp.mp.dps = 150
    x = mp.pi / 2 + mp.e * mp.mpf(10) ** -37
    r, z = x, []
    for _ in range(8):
        h = float(r)
        z.append(np.array([0.0, h]))
        r -= mp.mpf(h)
    want = mp.nstr(x, 40, strip_zeros=False)
    got = delta_digits(z, parts=mutate)
    two = delta_digits(z, parts=2)
    z_ulp = list(z)
    z_ulp[7] = np.array([0.0, float(np.nextafter(z[7][-1], np.inf))])
    ulp = delta_digits(z_ulp)
    ok = got == want and two != want
    print(f"  planted {want}\n  printed {got}{'  (--mutate ' + str(mutate) + ')' if mutate else ''}\n"
          f"  two parts {two}\n  last part one ulp up {ulp} ({'same' if ulp == delta_digits(z) else 'differs'} at 40 digits)", flush=True)
    print(f"  {'PASS' if ok else 'FAIL'}: the print equals the planted value to 40 digits and the two-part sum differs",
          flush=True)
    return 0 if ok else 1


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
    quad-double LU at the point's quad-double rounding instead of the octo-double one. c8: the offset in eight parts
    (PinnedSolver's c8, anm_run.py --held_o 1), used instead of ch, cl in residual_o and in the step."""

    def __init__(self, s, ch, cl, mutate_offset="", c8=None):
        self.s = s
        self.ch, self.cl = np.ascontiguousarray(ch), np.ascontiguousarray(cl)
        self.zero = np.zeros(len(self.ch))
        self.mutate_offset = mutate_offset
        self.c8 = None if c8 is None else o8(c8)

    def __getattr__(self, name):
        return getattr(self.s, name)

    def residual_o(self, z, mutate=0):
        from multidouble import od_add_dd, od_axpy
        F = self.s.residual_o(list(o8(z)), mutate=mutate)
        if self.c8 is not None:
            return o8(od_axpy(-1.0, *self.c8, *o8(F)))
        return o8(od_add_dd(*o8(F), -self.ch, -self.cl))

    def newton_o(self, z, mutate=0):
        from multidouble import od_axpy
        z = o8(z)
        d, _, _, pmin, pmax = self.s.newton_o(list(z), mutate=mutate)
        if self.mutate_offset == "qd":
            self.s.newton_q(*q4_of_o(z), mutate=mutate)
            x = o8_of_q(self.s.qlu_solve(self.ch, self.cl, self.zero, self.zero))
        elif self.c8 is not None:
            x = o8(self.s.olu_solve(list(self.c8)))
        else:
            x = o8(self.s.olu_solve([self.ch, self.cl] + [self.zero] * 6))        # J^-1 c
        d = o8(od_axpy(1.0, *x, *o8(d)))
        F = self.residual_o(z, mutate=mutate)
        return d, F[0], F[1], pmin, pmax


def junk_subspace_o(s, w, delta_max=1e-2, iters=6, seed=0):
    """the w weakest right singular directions of the Jacobian of s's last octo-double LU (subspace iteration on
    (J^T J)^-1 by olu_solve, transposed then plain, in the leading part), Rayleigh-Ritz, keeping those whose Delta
    component is below delta_max ("junk": at large D p1's highest harmonics, which carry no Delta;
    notes/QUAD_DOUBLE.md 13.30), orthonormal; with the kept and the other singular values (junk_subspace_q's octo twin)"""
    L = s.size
    zero = np.zeros(L)
    rng = np.random.default_rng(seed)

    def apply(v):
        y = s.olu_solve([np.ascontiguousarray(v)] + [zero] * 7, True)
        return np.asarray(s.olu_solve([np.ascontiguousarray(np.asarray(y[0]))] + [zero] * 7, False)[0])

    V = np.linalg.qr(rng.standard_normal((L, w)))[0]
    for _ in range(iters):
        V = np.linalg.qr(np.column_stack([apply(V[:, j]) for j in range(w)]))[0]
    W = np.column_stack([apply(V[:, j]) for j in range(w)])
    H = V.T @ W
    ev, U = np.linalg.eigh((H + H.T) / 2)
    X = V @ U
    X /= np.linalg.norm(X, axis=0)
    sig = 1 / np.sqrt(np.maximum(ev, 1e-300))
    keep = (ev > 0) & (np.abs(X[-1, :]) < delta_max)
    return (np.linalg.qr(X[:, keep])[0] if keep.any() else np.zeros((L, 0))), sig[keep], sig[~keep]


def newton_od(s, z, tol=1e-40, it_max=12, max_halvings=8, verbose=True, mutate=0, junk_w=0, delta_max=1e-2, tol_dz=1e-45,
              lam_stall=0.0, on_first=None, scale_c=None, trace=None):
    """Newton in octo-double from the eight-part state z on s (PinnedO, or ElimO around it): newton_qd's mode 'full'
    one level up (hpkernel.driver.newton_qd, its acceptance rule). A trial z + lam dz (lam = 1, 1/2, ...,
    2^-max_halvings) is accepted when the Newton correction there, with its own Jacobian, is smaller than dz (max norms
    of the leading parts), or when its max |F| <= tol. Stops when max |F| <= tol (converged), when no trial is accepted
    (stalled), or after it_max steps. A trial whose Jacobian is not finite or singular is rejected. Returns (z, hist,
    converged), hist one record per step as newton_qd's. junk_w > 0: as newton_qd's junk mode, the junk among the
    junk_w weakest directions of the first Jacobian (junk_subspace_o: no Delta content) is removed from every step and
    every correction it is judged by, and the iteration also ends converged when that projected correction falls below
    tol_dz (the residual left is the junk's). lam_stall > 0: the iteration also stops, not converged, when two
    accepted steps in a row had lam at or below lam_stall: past D 232 a step that is too long creeps at lam 1/64 to
    1/32 with max |F| falling one percent per iteration, and it_max of those cost an hour (#461). on_first: called
    once as on_first(z, dz) with the start and its first Newton step (eight parts each), before any trial.

    scale_c (f64, one per unknown): the test compares the corrections in the unknowns y = D_c^-1 z of the equilibrated
    system D_r J D_c (near_null_eq.py, #504), max |D_c^-1 dz_trial| < max |D_c^-1 dz| instead of the leading parts' max
    norms. A row scaling does not change the Newton correction, a column scaling does. Each record then also carries
    dz_eq, dz_eq_new and every trial (lam, max |F|, both sides of both tests). verbose >= 2 prints every trial. trace: a
    list that gets each trial's leading parts of dz and of its correction (#504's planted check)."""
    import time
    from multidouble import od_axpy
    z = o8(z)
    eq = scale_c is not None
    if eq:
        scale_c = np.asarray(scale_c, dtype=float)

    def size(d, scaled=eq):
        """the max norm the test compares: of the leading part, or of D_c^-1 times it"""
        return float(np.abs(np.asarray(d[0]) / scale_c).max()) if scaled else float(np.abs(d[0]).max())
    st = s.newton_o(z, mutate=mutate)
    hist = []
    V = None
    if junk_w:
        V, sk, sd = junk_subspace_o(s, junk_w, delta_max)
        if verbose:
            print(f"  od: {V.shape[1]} junk directions frozen (sigma {sk.min() if len(sk) else 0:.0e}..{sk.max() if len(sk) else 0:.0e}), "
                  f"{len(sd)} weak ones with Delta content kept", flush=True)
    zero8 = np.zeros(len(z[0]))
    small = 0

    def proj(d):
        """the eight-part step with its components along V removed"""
        if V is None or V.shape[1] == 0:
            return d
        c = V @ (V.T @ (np.asarray(d[0]) + np.asarray(d[1])))
        return o8(od_axpy(-1.0, np.ascontiguousarray(c), *([zero8] * 7), *o8(d)))

    for it in range(it_max + 1):
        nf = omax(st[1:3])
        if V is not None and V.shape[1] and float(np.abs(proj(st[0])[0]).max()) < tol_dz:
            if verbose:
                print(f"  on {it}: projected correction below {tol_dz:.0e} with the junk frozen: converged at |F| {nf:.2e}, "
                      f"Delta {z[0][-1]:.17f} {z[1][-1]:+.4e}", flush=True)
            hist.append(dict(it=it, F=nf, dz=float(np.abs(proj(st[0])[0]).max()), mode="sel", junk_floor=True))
            return z, hist, True
        if nf <= tol:
            if verbose:
                print(f"  on {it}: |F| {nf:.2e} converged, Delta {z[0][-1]:.17f} {z[1][-1]:+.4e}", flush=True)
            hist.append(dict(it=it, F=nf, dz=float(np.abs(st[0][0]).max()), mode="full"))
            return z, hist, True
        if it == it_max:
            break
        d = proj(st[0])
        if it == 0 and on_first is not None:
            on_first(z, d)
        nd = size(d)
        t0 = time.time()
        rec = dict(it=it, F=nf, dz=float(np.abs(d[0]).max()) if eq else nd, dDelta=float(d[0][-1] + d[1][-1]), piv=[st[3], st[4]],
                   mode="sel" if V is not None else "full")
        if eq:
            rec.update(dz_eq=nd, trials=[])
        lam = 1.0
        for _ in range(max_halvings + 1):
            zt = o8(od_axpy(lam, *d, *z))
            t1 = time.time()
            try:
                stt = s.newton_o(zt, mutate=mutate)
                good = np.isfinite(stt[0][0]).all() and np.isfinite(stt[1]).all()
            except ValueError:          # a non-finite or singular Jacobian at the trial point
                good = False
            ok = good and (size(proj(stt[0])) < nd or omax(stt[1:3]) <= tol)
            if eq or verbose >= 2 or trace is not None:
                dt = proj(stt[0]) if good else None
                tr = dict(lam=lam, good=bool(good), accepted=bool(ok), F=omax(stt[1:3]) if good else float("nan"),
                          dz=rec["dz"], dz_t=size(dt, False) if good else float("nan"), seconds=round(time.time() - t1, 2))
                if eq:
                    tr.update(dz_eq=nd, dz_eq_t=size(dt, True) if good else float("nan"))
                    rec["trials"].append(tr)
                if verbose >= 2:
                    line = (f"    on {it} trial lam {lam:g}: max|F| {tr['F']:.2e}; raw |dz_t| {tr['dz_t']:.3e} vs |dz| {tr['dz']:.3e}"
                            f" ({'<' if tr['dz_t'] < tr['dz'] else 'not <'})")
                    if eq:
                        line += (f"; eq |dz_t/D_c| {tr['dz_eq_t']:.3e} vs |dz/D_c| {nd:.3e} "
                                 f"({'<' if tr['dz_eq_t'] < nd else 'not <'})")
                    print(line + f"; {'accepted' if ok else 'refused'}, {tr['seconds']} s", flush=True)
                if trace is not None:
                    trace.append(dict(it=it, lam=lam, accepted=bool(ok), F=tr["F"], d=np.array(d[0]),
                                      dt=np.array(dt[0]) if good else None))
            if ok:
                break
            lam *= 0.5
        else:
            rec.update(lam=0.0, seconds=round(time.time() - t0, 2))
            hist.append(rec)
            if verbose:
                print(f"  on {it}: |F| {nf:.2e}, |dz| {rec['dz']:.1e}: no trial step accepted, stopping", flush=True)
            return z, hist, False
        z, st = zt, stt
        rec.update(lam=lam, F_new=omax(st[1:3]), dz_new=float(np.abs(proj(st[0])[0]).max()), seconds=round(time.time() - t0, 2))
        if eq:
            rec["dz_eq_new"] = size(proj(st[0]))
        hist.append(rec)
        if verbose:
            print(f"  on {it}: |F| {nf:.2e} -> {rec['F_new']:.2e}, |dz| {rec['dz']:.1e} -> {rec['dz_new']:.1e}, lam {lam:g}, "
                  f"dDelta {rec['dDelta']:+.2e}, piv {st[3]:.1e}..{st[4]:.1e}, {rec['seconds']} s"
                  + (f"; |dz/D_c| {nd:.1e} -> {rec['dz_eq_new']:.1e}" if eq else ""), flush=True)
        small = small + 1 if (lam_stall and lam <= lam_stall) else 0
        if small >= 2:
            if verbose:
                print(f"  on {it}: lam at or below {lam_stall:g} twice in a row, stopping (stalled)", flush=True)
            return z, hist, False
    return z, hist, False


def with_horizon(z8, z4, rs):
    """z8 (eight parts) with its horizon data (m_h, a_h) taken from z4 (four parts), their parts 4-7 zero: a re-seed or
    projection (rs, a reseed.Reseeder of the layout, changes only those) put back on an octo-double point. With
    rs.od > 0 the data are then corrected in octo-double against the point's own eight parts (Reseeder.refine_o), and
    the rung residuals of that are kept in rs.last_o."""
    sl = slice(rs.off["mh"], rs.off["mh"] + 2 * rs.M)
    out = [np.array(p, dtype=float) for p in o8(z8)]
    for q in range(8):
        out[q][sl] = np.asarray(z4[q])[sl] if q < 4 else 0.0
    if getattr(rs, "od", 0):
        out, rs.last_o = rs.refine_o(out)
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


def pinned_o(meta, z, threads=2, plan_tol=1e-32, mutate_offset="", held_exact=False):
    """reseed.pinned's solver at the state's D, with the segments planned at z at plan_tol, and its PinnedO wrapper:
    (s, PinnedSolver, PinnedQ, PinnedO). The held values are z's own, as floats, or with held_exact taken from all
    of z's parts and carried in eight (anm_run.py --held_o 1)."""
    import mpmath as mp
    from hpkernel.driver import make_solver
    from pinning import PinnedQ, PinnedSolver, band_coeffs, band_field
    M = int(meta["M"])
    D = mp.mpf(meta["D_str"])
    s = make_solver(D, M, nodes=meta["nodes"], kt=meta.get("kt", 36), kc=meta.get("kc", 146), ks=meta.get("ks", 130),
                    threads=threads, kfreeze=meta["kfreeze"])
    s.set_segs(s.plan(np.asarray(z[0]), np.asarray(z[1]), plan_tol, 0.05))
    ks_held = [k for k in range(int(meta["kfreeze"]) + 1, 14) if k % 2 == 1]
    co = band_coeffs(np.asarray(z[0])[:M], np.asarray(z[1])[:M], M, ks_held, exact=held_exact,
                     rest=[np.asarray(p)[:M] for p in z[2:]] if held_exact else ())
    h = band_field(co, ks_held, M, parts=8 if held_exact else 2)
    sp = PinnedSolver(s, M, h[0], h[1], h_rest=h[2:])
    return s, sp, PinnedQ(s, sp.ch, sp.cl), PinnedO(s, sp.ch, sp.cl, mutate_offset=mutate_offset, c8=sp.c8)



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
    z, hist, conv = newton_od(el, z8, tol=opt["tol"], it_max=opt["it_max"], mutate=opt["mutate"], junk_w=opt.get("junk", 0))
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


def plant_start(z4, rs, rel, seed, last):
    """planted's start (four parts) from the converged four-part state z4 and the number n of moved unknowns: the
    centre data and the nodes but the last (last: all nodes) moved by rel r relative, r uniform in +-[0.5, 1] (seed). A
    last node moved by 1e-12 leaves the regular family by G 1e-12 at the horizon, and its projection fails (180.5246,
    2026-11-06)"""
    from hpkernel.driver import qd_axpy
    n = rs.off["mh"] if last else rs.off["node"](rs.K - 1)
    rng = np.random.default_rng(seed)
    r = rng.uniform(0.5, 1.0, n) * rng.choice([-1.0, 1.0], n)
    dz = np.zeros(len(z4[0]))
    dz[:n] = rel * r * z4[0][:n]
    zero = np.zeros(len(dz))
    return o8(qd_axpy(1.0, dz, zero, zero, zero, *z4)), n


def planted(argv):
    """check (a): the converged chain state RAW/STATE with its centre data and nodes but the last (--last 1: all
    nodes; not m_h, a_h, Delta) moved by rel r relative, r uniform in +-[0.5, 1] (--seed), is the start. Pass, fixed
    before the first run: the octo-double
    corrector (ElimO, newton_od, tol 1e-40) converges with the octo-double pinned max |F| below 1e-40, and its Delta
    agrees with the quad-double corrector's from the same start (ElimQ, newton_qd to tol 1e-45 in 12 steps, its
    floor) to 1e-30. The pinned system is the state's own (held harmonics at its values, the plan at it at 1e-32).
    Controls, which must fail: --mutate_offset qd (PinnedO's J^-1 c through the quad-double LU), --mutate 10 (the
    kernel's OD_MUT_STATE_QD, the state read in its quad-double part). --qd 0 skips the quad-double run and --od 0 the
    octo-double one; --Delta_ref then gives the other's Delta (a passing run's) to compare with, else only the
    residual is judged. Exit 1 on failure."""
    import os
    import time
    import mpmath as mp
    from hpkernel.driver import load_q
    from reseed import Reseeder
    raw, path = argv[0], argv[1]
    opt = _parse(argv[2:], dict(threads=3, rel=1e-12, seed=1, mutate_offset="", mutate=0, it_max=12, qd=1, tol=1e-40, out="",
                                Delta_ref="", last=0, od=1))
    z4, meta = load_q(os.path.join(raw, path))
    z4 = o8(z4)
    D = mp.mpf(meta["D_str"])
    s, sp, sq, so = pinned_o(meta, z4, threads=opt["threads"], mutate_offset=opt["mutate_offset"])
    rs = Reseeder(meta, D, threads=opt["threads"], qd=True)
    zs, n = plant_start(z4, rs, opt["rel"], opt["seed"], opt["last"])
    print(f"planted {path}: D {mp.nstr(D, 12)}, start = truth moved by {opt['rel']:.0e} relative in [0, {n}); pinned max|F| (octo) "
          f"truth {omax(so.residual_o(o8_of_q(z4))):.2e}, start {omax(so.residual_o(o8_of_q(zs))):.2e}; mutate_offset "
          f"'{opt['mutate_offset']}', mutate {opt['mutate']}", flush=True)
    import json
    rec = dict(test="planted", state=path, D_str=meta["D_str"], rel=opt["rel"], seed=opt["seed"], mutate_offset=opt["mutate_offset"],
               mutate=opt["mutate"], last=opt["last"])

    def dump():
        if opt["out"]:
            with open(opt["out"] + ".json", "w") as f:
                f.write(json.dumps(rec, default=float) + "\n")
    ok, judged = True, []
    if opt["od"]:
        t0 = time.time()
        zo, ho, co, Fo, counts = _run_od(so, rs, o8_of_q(zs), opt)
        Do = _delta(zo)
        print(f"  od: {len(ho)} steps, {'converged' if co else 'NOT converged'}, pinned max|F| (octo) {Fo:.2e}, Delta {mp.nstr(Do, 40)}, "
              f"projections {counts}, {time.time() - t0:.0f} s", flush=True)
        print(f"  od against the truth: dDelta {mp.nstr(Do - _delta(z4), 5)}", flush=True)
        ok = bool(co) and Fo < 1e-40
        judged.append(f"od max|F| {Fo:.2e} < 1e-40")
        rec["od"] = dict(steps=len(ho), converged=bool(co), F=Fo, Delta=mp.nstr(Do, 40), hist=ho, counts=counts)
        if opt["out"]:
            save_o(opt["out"] + ".npz", zo, dict(meta, D_str=meta["D_str"], planted=True))
        dump()
    else:
        # --od 0: the quad-double run alone, against the octo-double Delta of an earlier run (--Delta_ref)
        Do = None
    ref = None
    if opt["Delta_ref"]:
        with mp.workdps(60):
            ref = mp.mpf(opt["Delta_ref"])
    if opt["qd"]:
        t0 = time.time()
        zq, hq, cq, Fq, cntq = _run_qd(sq, so, rs, zs, 1e-45, 12)
        Dq = _delta(zq)
        print(f"  qd: {len(hq)} steps, {'converged' if cq else 'stopped'}, its last max|F| {hq[-1]['F']:.2e} (octo at the projected point "
              f"{Fq:.2e}), Delta {mp.nstr(Dq, 40)}, projections {cntq}, {time.time() - t0:.0f} s", flush=True)
        rec["qd"] = dict(steps=len(hq), converged=bool(cq), F_last=hq[-1]["F"], F_octo=Fq, Delta=mp.nstr(Dq, 40), hist=hq)
        Da, Db = (Do, Dq) if Do is not None else (ref, Dq)
    else:
        Da, Db = Do, ref
    if Da is not None and Db is not None:
        dD = abs(Da - Db)
        print(f"  |Delta_od - Delta_qd| {mp.nstr(dD, 5)}" + (f" (Delta_ref {opt['Delta_ref']})" if ref is not None else ""), flush=True)
        ok = ok and dD <= 1e-30
        judged.append("|Delta_od - Delta_qd| <= 1e-30")
        rec["dDelta"] = float(dD)
    rec["passed"] = bool(ok)
    dump()
    print(f"  (a) {'PASS' if ok else 'FAIL'}: " + ", ".join(judged), flush=True)
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
    opt = _parse(argv[2:], dict(threads=3, mutate_offset="", mutate=0, it_max=12, qd=0, tol=1e-40, out="", F_max=1e-36, junk=0,
                                reseed_o=0, held_o=0))
    z4, meta = load_q(os.path.join(raw, path))
    D = mp.mpf(meta["D_str"])
    rs = Reseeder(meta, D, threads=opt["threads"], qd=True, od=opt["reseed_o"])
    t0 = time.time()
    zr, diag = rs.reseed(o8(z4))
    zr = o8(zr)
    s, sp, sq, so = pinned_o(meta, zr, threads=opt["threads"], mutate_offset=opt["mutate_offset"],
                             held_exact=bool(opt["held_o"]))
    print(f"wall {path}: D {mp.nstr(D, 12)}, re-seed {'ok' if diag['ok'] else 'FAILED'} ({diag['seconds']} s), pinned max|F| (octo) "
          f"{omax(so.residual_o(o8_of_q(zr))):.2e}; mutate_offset '{opt['mutate_offset']}', mutate {opt['mutate']}", flush=True)
    start = o8_of_q(zr)
    if rs.od:
        # as anm_run.py does at the prediction: the re-seeded horizon data refined before the corrector starts
        start, hr = rs.refine_o(start)
        print("  horizon data refined in octo-double, rung max|R| " + " -> ".join(f"{r:.2e}" for r in hr), flush=True)
    zo, ho, co, Fo, counts = _run_od(so, rs, start, opt)
    Do = _delta(zo)
    print(f"  od: {len(ho)} steps, {'converged' if co else 'NOT converged'}, pinned max|F| (octo) {Fo:.2e}, Delta {mp.nstr(Do, 40)}, "
          f"projections {counts}, {time.time() - t0:.0f} s", flush=True)
    import json
    ok = Fo < opt["F_max"]
    rec = dict(test="wall", pred=path, D_str=meta["D_str"], mutate_offset=opt["mutate_offset"], mutate=opt["mutate"],
               od=dict(steps=len(ho), converged=bool(co), F=Fo, Delta=mp.nstr(Do, 40), hist=ho, counts=counts), passed=bool(ok))

    def dump():
        if opt["out"]:
            with open(opt["out"] + ".json", "w") as f:
                f.write(json.dumps(rec, default=float) + "\n")
    if opt["out"]:
        save_o(opt["out"] + ".npz", zo, dict(meta, D_str=meta["D_str"], prediction=False, corrected="od"))
    dump()
    print(f"  (b) {'PASS' if ok else 'FAIL'}: od max|F| {Fo:.2e} < {opt['F_max']:.0e}", flush=True)
    if opt["qd"]:
        t0 = time.time()
        zq, hq, cq, Fq, cntq = _run_qd(sq, so, rs, zr, 1e-28, 12)
        Dq = _delta(zq)
        print(f"  qd: {len(hq)} steps, {'converged' if cq else 'stopped'}, its last max|F| {hq[-1]['F']:.2e} (octo at the projected point "
              f"{Fq:.2e}), Delta {mp.nstr(Dq, 40)}, projections {cntq}, {time.time() - t0:.0f} s; Delta_od - Delta_qd {mp.nstr(Do - Dq, 5)}",
              flush=True)
        rec["qd"] = dict(steps=len(hq), converged=bool(cq), F_last=hq[-1]["F"], F_octo=Fq, Delta=mp.nstr(Dq, 40))
        dump()
    return 0 if ok else 1


def horizon(argv):
    """the planted check of Reseeder.refine_o. The truth is built from a converged eight-part state RAW/STATE (anm_run.py
    --corrector od): its horizon data refined until the rung residual stops falling, so that they solve the rung's
    equations far below the state's own residual (the state's data solve them only to its residual, and along the
    block's weak directions that leaves them 1e-62 from the rung's solution at D 221). The truth's data are then
    rounded to quad-double (parts 4-7 zero, what with_horizon makes of the best quad-double re-seed) and refined
    again. Pass, fixed before the run: the plant shows (the rung's max |R| at the rounded data is at least 1e6 times
    the truth's and at least 1e-36; exit 2 if not, since nothing is tested then), the refined residual is at most
    1e-10 of the rounded one, and the data's distance from the truth's falls to at most 1e-10 of the rounding's.
    Controls: --steps 0 (no correction) and --mutate_h sign (each correction with the wrong sign) must fail. Exit 1 on
    failure."""
    import os
    import time
    import mpmath as mp
    from reseed import Reseeder
    raw, path = argv[0], argv[1]
    opt = _parse(argv[2:], dict(threads=3, steps=3, mutate_h=""))
    if not has_o(os.path.join(raw, path)):
        print(f"horizon {path}: not an eight-part state, nothing to test")
        return 2
    z8, meta = load_o(os.path.join(raw, path))
    D = mp.mpf(meta["D_str"])
    rs = Reseeder(meta, D, threads=opt["threads"], qd=True, od=opt["steps"])
    sl = slice(rs.off["mh"], rs.off["mh"] + 2 * rs.M)
    t0 = time.time()
    R_state = rs.refine_o(o8(z8), 0)[1][0]
    zt, ht = rs.refine_o(o8(z8), 4)
    zq = [np.array(p) for p in zt]
    for q in range(4, 8):
        zq[q][sl] = 0.0
    zr, hist = rs.refine_o(zq, opt["steps"], mutate=opt["mutate_h"])

    def dist(z):
        """the largest distance of z's horizon data from the truth's, relative to the largest datum"""
        with mp.workdps(160):
            v = [abs(sum(mp.mpf(float(z[q][i])) for q in range(8)) - sum(mp.mpf(float(zt[q][i])) for q in range(8)))
                 for i in range(sl.start, sl.stop)]
            top = max(abs(mp.mpf(float(zt[0][i]))) for i in range(sl.start, sl.stop))
            return float(max(v) / top)
    e_q, e_r = dist(zq), dist(zr)
    print(f"horizon {path}: D {mp.nstr(D, 12)}, steps {opt['steps']}, mutate_h '{opt['mutate_h']}', {time.time() - t0:.0f} s", flush=True)
    print(f"  rung max|R|: the state's own {R_state:.2e}; the truth's {ht[-1]:.2e}; rounded to quad-double {hist[0]:.2e}; after "
          "each correction " + (", ".join(f"{r:.2e}" for r in hist[1:]) or "none"), flush=True)
    print(f"  horizon data's distance from the truth's: rounded {e_q:.2e}, refined {e_r:.2e}", flush=True)
    if not (hist[0] >= 1e6 * ht[-1] and hist[0] >= 1e-36):
        print("  the plant does not show in the rung residual: nothing tested", flush=True)
        return 2
    ok = bool(np.isfinite(hist[-1]) and hist[-1] <= 1e-10 * hist[0] and e_r <= 1e-10 * e_q)
    print(f"  {'PASS' if ok else 'FAIL'}: residual {hist[-1]:.2e} against 1e-10 x {hist[0]:.2e}, distance {e_r:.2e} against "
          f"1e-10 x {e_q:.2e}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if len(sys.argv) > 1 and sys.argv[1] == "digits":
        sys.exit(digits_check(sys.argv[2:]))
    if len(sys.argv) > 3 and sys.argv[1] in ("planted", "wall", "horizon"):
        sys.exit(dict(planted=planted, wall=wall, horizon=horizon)[sys.argv[1]](sys.argv[2:]))
    print(__doc__)
    sys.exit(2)

#!/usr/bin/env python3
"""The crawl of the octo-double corrector at the chain's front (#700, notes/CRAWL.md). At a crawl iterate z with its
Newton step dz (ElimO.newton_o: the projection's displacement plus the step from the projected point), the residual
the corrector sees along the step, F(lam) = F(z + lam dz), is evaluated on a ladder of lam and bisected to lam*, the
point where it stops being affine. The rows that jump there are named by block, field and harmonic.

    python3 crawl_probe.py selftest [--mutate shift|comp|smooth|control|coarse]
    python3 crawl_probe.py iterate RAW BASE.npz --a A (--jet JET.npz | --pred PRED.npz) --out DIR [--threads 2]
        [--strong 4] [--steps 2] [--check_pred PRED.npz]
    python3 crawl_probe.py probe RAW ITER.npz --out DIR [--threads 2] [--lo 0.5] [--hi 1.0] [--rungs 6]
        [--tol_lam 1e-6] [--arms U,E] [--theta 1] [--theta_half 0] [--hint 1]
    python3 crawl_probe.py fixrun RAW ITER.npz --out DIR [--threads 2] [--switch 0] [--steps 6] [--tol 1e-60]
    python3 crawl_probe.py offset RAW ITER.npz [--threads 2]
    python3 crawl_probe.py chord RAW BASE.npz --a A (--jet JET.npz | --pred PRED.npz) --out DIR [--threads 2]
        [--strong 4] [--chord_it 10] [--check_pred PRED.npz] [--check_iter ITER.npz]
    python3 crawl_probe.py chordstat RAW DIR/chord_trace_D<D>.npz [--threads 2]

`selftest` is the planted check (fixed before the first real run): a smooth map with a quadratic remainder along the
exact Newton step, plus a jump of known rows switched on at a known lam_true along that step. Pass: lam* within 1e-6 of
lam_true, the rows carrying 99% of the jump (in the squared 2-norm) equal to the planted rows, and the control (the
same map without the jump) reported affine to the ladder's end. The detector runs three times: without a hint, with
a bracket hint around lam_true, and with a hint 0.01 away, which it must refuse and still find lam*. The mutations must
make it fail: shift (the jump planted 5e-6 past the lam_true the check expects), comp (planted on the neighbouring
rows), smooth (no jump planted, one expected), control (a jump planted in the control), coarse (the bisection stopped
at 1e-2), hint_trust (a hinted bracket taken without evaluating its ends). Exit 1 on failure.

`iterate` rebuilds one attempt of anm_run.py with #603's flags (--corrector od --tol_od 1e-60 --reseed 2 --reseed_q 1
--reseed_o 3 --held_o 1 --held_fit data/held_fit.json --jet_o 1 --od_accept natural --od_strong 4 --lam_stall 0.04,
kfreeze 3): the base state, its plan, the held field at D + a, the prediction (horner_o of the stored series, or the
stored four-part prediction promoted to eight), its re-seed, PinnedO inside ElimO, and newton_od for --steps steps.
It saves the iterate after the first step (the chord's point under --strong) with its Newton step, as ElimO.newton_o
returned them inside newton_od, to DIR/crawl_iter_D<D>.npz. --check_pred: the stored prediction the run saved, which
the rebuilt one must equal bit for bit (exit 1 otherwise).

`probe` evaluates, at the saved iterate, lam -> F(z + lam dz) on the ladder lo..hi and bisects the first jump to
tol_lam (locate_jump). Arm E is the corrector's own residual (ElimO: the trial projected when its pinned quad-double
max |F| exceeds the switch, as newton_od's natural test evaluates it), arm U the residual never projected (PinnedO),
arm P (on request) the residual always projected; after arm E the point just above lam* is also evaluated projected.
Each evaluation records the switch's test value, whether the point was projected, and max |F - (1 - lam) F0| per
block. --theta 1 also forms the natural test's simplified correction at lam = 1 with
the trial projected (what a corrector that projects every trial of a projected iterate would see), and --theta_half 1
at lam = 1/2 as the corrector forms it. --hint 1 (the default) brackets arm E's jump first by bisecting the switch's
test value along the line (no projection), then evaluates F at the bracket's ends (locate_jump's hint); a projection
costs about 5.5 min on the Mac.

`fixrun` continues the corrector from the saved iterate (newton_od, the natural test, no chord, at most --steps steps
to --tol) with ElimO's switch set to --switch: 0 projects every point the corrector evaluates (the proposed fix),
1e-15 is the corrector as it runs (the control, which must crawl: lam 1/2 at ratio 0.5000). It prints each trial and
the final pinned octo-double max |F| of the converged point's projection and of the point itself, and Delta to 40
digits.

`offset` splits the saved step dz into the projection's displacement P(z) - z (the iterate's horizon data against those
its last node asks for) and the step from the projected point, and prints each by block and field, with the
displacement's m_h and a_h by harmonic and the unprojected residual's largest rows (one projection).

`chord` (#715) rebuilds the attempt's start and its first Newton step as `iterate` does, and runs chord_strong as
newton_od calls it under --strong, through an ElimO that keeps each point it is asked to project (trace_elim). It keeps
each chord point, its projection (formed again with the switch off where the switch skipped it), each chord solve
x = J0^-1 F and the weak subspace V, in DIR/chord_trace_D<D>.npz, and then runs `chordstat` on them. --check_iter: the
iterate `iterate` saved, which the chord's last point must equal bit for bit (exit 1 otherwise). `chordstat` prints,
at each point, the switch's test value, the point's own residual and its projection's by block (the horizon rows by
harmonic), and P(z) - z by unknown group with m_h and a_h by harmonic. At each chord step it prints x, V^T x and
dw = drop_V(x) by group, checks that the step reproduces the next point bit for bit (exit 1 otherwise), and splits the
step: dw against drop_V in mpmath with the same V (drop_V's f64 rounding), and the switch's test value at z - dw with
one group of dw applied, with the exact dw, with the rounding alone, and with V rounded differently (V plus 1e-16
noise, orthonormalized again).
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

FIELDS = ("p", "m", "a", "c")


# ------------------------------------------------------------------ the detector (shared by the planted check and the probe)
def growth(e_a, e_b, lam_a, lam_b, floor):
    """e_b against e_a carried to lam_b as a quadratic remainder would grow, (lam_b / lam_a)^2, above floor"""
    return e_b / max(e_a * (lam_b / lam_a) ** 2, floor)


HINT_MUT = {}         # the control of the hint: {"trust": True} takes a hinted bracket without evaluating its ends


def locate_jump(ev, F0, lams, tol_lam=1e-6, K=1e3, floor_rel=1e-12, max_bisect=60, verbose=True, hint=None):
    """The first lam on the ladder lams (increasing) past which lam -> ev(lam) stops being affine. ev(lam) returns
    (F, info): F the residual's leading values (float64), info a dict kept with the record. The deviation from the
    affine model is e(lam) = max |F(lam) - (1 - lam) F0|. A smooth map along a Newton step deviates by its quadratic
    remainder, which grows like lam^2; a jump is a rung whose e exceeds K times the last rung's e carried by
    (lam_b / lam_a)^2 (and the floor floor_rel max |F0|). The bracket is then bisected with the threshold
    tau = sqrt(e_a (lam_b / lam_a)^2 e_b) until it is narrower than tol_lam. Returns a dict: lam_star (None when
    every rung is affine), the bracket's ends with their F and e, every evaluation in order, and the largest growth
    on the ladder. hint(lam_a, lam_b), when given, proposes a narrower bracket (h_a, h_b) inside the rungs' bracket
    (from a cheaper quantity than ev): ev is evaluated at both ends, and the bracket is taken only when h_a reads
    affine (e <= tau) and h_b jumped (e > tau); otherwise the two evaluations narrow the rungs' bracket as bisection
    steps would, and the bisection goes on from there."""
    F0 = np.asarray(F0, dtype=float)
    floor = floor_rel * float(np.abs(F0).max())
    recs = []

    def e_of(lam):
        F, info = ev(lam)
        F = np.asarray(F, dtype=float)
        e = float(np.abs(F - (1.0 - lam) * F0).max())
        recs.append(dict(lam=lam, e=e, maxF=float(np.abs(F).max()), **info))
        if verbose:
            extra = " ".join(f"{k} {v:.3e}" if isinstance(v, float) else f"{k} {v}" for k, v in info.items()
                             if not isinstance(v, (list, dict)))
            print(f"    lam {lam:.9f} (1 - lam {1 - lam:.4e}): max|F| {recs[-1]['maxF']:.3e}, e {e:.3e}  {extra}", flush=True)
        return F, e
    rung = [e_of(lam) for lam in lams]
    g = [growth(rung[i][1], rung[i + 1][1], lams[i], lams[i + 1], floor) for i in range(len(lams) - 1)]
    out = dict(lams=list(lams), growth=g, g_max=max(g) if g else 0.0, K=K, floor=floor, lam_star=None)
    i = next((i for i, x in enumerate(g) if x > K), None)
    if i is None:
        out.update(evals=recs, e_end=rung[-1][1])
        return out
    la, lb = lams[i], lams[i + 1]
    (Fa, ea), (Fb, eb) = rung[i], rung[i + 1]
    tau = float(np.sqrt(max(ea * (lb / la) ** 2, floor) * eb))
    out["hint"] = None
    h = hint(la, lb) if hint is not None else None
    if h is not None and la < h[0] < h[1] < lb:
        if HINT_MUT.get("trust"):
            (Fa, ea), (Fb, eb) = e_of(h[0]), e_of(h[1])
            la, lb = h
            out["hint"] = dict(bracket=list(h), taken=True, trusted=True)
        else:
            (Fha, eha), (Fhb, ehb) = e_of(h[0]), e_of(h[1])
            taken = eha <= tau < ehb
            if eha > tau:                    # the jump lies below h_a
                lb, Fb, eb = h[0], Fha, eha
            else:
                la, Fa, ea = h[0], Fha, eha
                if ehb > tau:
                    lb, Fb, eb = h[1], Fhb, ehb
                else:                        # the jump lies above h_b
                    la, Fa, ea = h[1], Fhb, ehb
            out["hint"] = dict(bracket=list(h), taken=bool(taken))
    for _ in range(max_bisect):
        if lb - la <= tol_lam:
            break
        mid = 0.5 * (la + lb)
        Fm, em = e_of(mid)
        if em > tau:
            lb, Fb, eb = mid, Fm, em
        else:
            la, Fa, ea = mid, Fm, em
    out.update(lam_star=0.5 * (la + lb), lam_a=la, lam_b=lb, width=lb - la, e_a=ea, e_b=eb, tau=tau, F_a=Fa, F_b=Fb,
               evals=recs)
    return out


def carriers(delta, frac=0.99):
    """the fewest rows whose squared entries hold frac of |delta|^2, largest first"""
    d2 = np.asarray(delta, dtype=float) ** 2
    order = np.argsort(-d2)
    c = np.cumsum(d2[order])
    n = int(np.searchsorted(c, frac * c[-1]) + 1)
    return [int(r) for r in order[:n]]


# ------------------------------------------------------------------ the planted check
def plant(n=161, seed=7, lam_true=None, rows=(140, 141, 150), amp=(0.7, 0.7, 0.1), J=1.0, eps=1e-6):
    """a smooth map F(x) = A x + b + eps x*x on R^n, a start z and its exact Newton step d (F(z + lam d) = (1 - lam) F(z)
    + eps lam^2 d*d), and a jump J sum_r amp_r e_r switched on where <w, x> passes its value at z + lam_true d. The
    remainder eps lam^2 |d|^2 is about 1e-5 at lam = 1, so the jump stands 1e4 to 1e5 above it, past locate_jump's
    K = 1e3 (with eps 1e-3 the jump is 135 times the remainder there, under K, and is reported affine)."""
    rng = np.random.default_rng(seed)
    A = 3.0 * np.eye(n) + 0.1 * rng.standard_normal((n, n)) / np.sqrt(n)
    b = rng.standard_normal(n)
    z = rng.standard_normal(n)
    w = rng.standard_normal(n)
    if lam_true is None:
        lam_true = float(rng.uniform(0.55, 0.95))

    def smooth(x):
        return A @ x + b + eps * x * x
    d = -np.linalg.solve(A + 2 * eps * np.diag(z), smooth(z))
    v = np.zeros(n)
    for r, s in zip(rows, amp):
        v[r] = s
    theta = float(w @ (z + lam_true * d))
    sgn = 1.0 if w @ d > 0 else -1.0

    def F(x):
        return smooth(x) + (J * v if sgn * (w @ x - theta) > 0 else 0.0)
    return F, z, d, lam_true


def selftest(argv):
    """the planted check of locate_jump and carriers (the module docstring's criteria and mutations)"""
    mutate = argv[argv.index("--mutate") + 1] if "--mutate" in argv else ""
    rows, lam_true = (140, 141, 150), 0.7316807
    ladder = [0.5 + 0.05 * i for i in range(11)]
    lam_plant = lam_true + (5e-6 if mutate == "shift" else 0.0)
    rows_plant = tuple(r + 1 for r in rows) if mutate == "comp" else rows
    F, z, d, _ = plant(lam_true=lam_plant, rows=rows_plant, J=0.0 if mutate == "smooth" else 1.0)
    tol = 1e-2 if mutate == "coarse" else 1e-7
    if mutate == "hint_trust":
        HINT_MUT["trust"] = True
    print(f"planted: n {len(z)}, jump at lam {lam_plant:.9f} on rows {list(rows_plant)}"
          f"{'  (--mutate ' + mutate + ')' if mutate else ''}", flush=True)
    F0 = F(z)
    ok_lam = ok_rows = True
    # three runs of the detector: no hint, a hint around lam_true, and a hint 0.01 away (which must be refused)
    for name, hint in (("no hint", None), ("hint at lam_true", lambda a, b: (lam_true - 3e-8, lam_true + 3e-8)),
                       ("hint 0.01 off", lambda a, b: (lam_true + 0.01 - 3e-8, lam_true + 0.01 + 3e-8))):
        r = locate_jump(lambda lam: (F(z + lam * d), {}), F0, ladder, tol_lam=tol, verbose=False, hint=hint)
        ok_l = r["lam_star"] is not None and abs(r["lam_star"] - lam_true) <= 1e-6
        # the jump net of the affine change across the bracket, (F_b - (1 - lam_b) F0) - (F_a - (1 - lam_a) F0)
        rows_found = (sorted(carriers((r["F_b"] - (1 - r["lam_b"]) * F0) - (r["F_a"] - (1 - r["lam_a"]) * F0)))
                      if r["lam_star"] is not None else [])
        ok_r = rows_found == sorted(rows)
        ok_lam, ok_rows = ok_lam and ok_l, ok_rows and ok_r
        hn = r.get("hint")
        print(f"  jump ({name}): lam* {r['lam_star'] if r['lam_star'] is None else format(r['lam_star'], '.9f')} "
              f"(expected {lam_true:.9f}, bound 1e-6), largest ladder growth {r['g_max']:.2e} (K {r['K']:.0e}); "
              f"rows carrying 99% {rows_found} (expected {sorted(rows)}); {len(r['evals'])} evaluations"
              + (f"; hint {'taken' if hn['taken'] else 'refused'}" if hn else ""), flush=True)
    Fc, zc, dc, _ = plant(lam_true=lam_true, rows=rows, J=1.0 if mutate == "control" else 0.0)
    rc = locate_jump(lambda lam: (Fc(zc + lam * dc), {}), Fc(zc), ladder, tol_lam=tol, verbose=False)
    ok_ctl = rc["lam_star"] is None
    print(f"  control (no jump): {'affine to lam = ' + format(ladder[-1], 'g') if ok_ctl else 'jump at ' + format(rc['lam_star'], '.9f')}, "
          f"largest ladder growth {rc['g_max']:.2e}, e at the end {rc.get('e_end', float('nan')):.2e}", flush=True)
    ok = ok_lam and ok_rows and ok_ctl
    print(f"  {'PASS' if ok else 'FAIL'}: lam* {'ok' if ok_lam else 'off'}, rows {'ok' if ok_rows else 'wrong'}, "
          f"control {'affine' if ok_ctl else 'jumped'}", flush=True)
    return 0 if ok else 1


# ------------------------------------------------------------------ the real attempt
def _opt(argv, key, default):
    f = "--" + key
    return type(default)(argv[argv.index(f) + 1]) if f in argv else default


def build_attempt(raw, base, a, threads=2, jet=None, pred=None, held_fit=None, reseed_o=3):
    """anm_run.py's attempt at D + a from the base state RAW/BASE with #603's flags: the solver at D + a on the base's
    plan, the held field (HeldFit increments from the base's own coefficients, in eight parts), PinnedO, the
    re-seeder, and the prediction in eight parts (horner_o of the stored series RAW/JET, or the stored four-part
    prediction RAW/PRED promoted to eight) re-seeded as anm_run does it. Returns a dict of the pieces."""
    import mpmath as mp
    from anm_run import horner_o, load_state
    from hpkernel.driver import load_q, make_solver
    from octo_corrector import PinnedO, load_o, o8, o8_of_q, q4_of_o, with_horizon
    from pinning import HeldFit, PinnedSolver, band_coeffs, band_field
    from reseed import Reseeder
    mp.mp.dps = 50
    path = os.path.join(raw, base)
    zh, zl, m = load_state(path)
    z8b = load_o(path)[0]
    M, nodes = int(m["M"]), list(m["nodes"])
    kt, kc, ks = m.get("kt", 36), m.get("kc", 146), m.get("ks", 130)
    kf = int(m["kfreeze"])
    ks_held = [k for k in range(kf + 1, 14) if k % 2 == 1]
    D = mp.mpf(m["D_str"])
    Dn = D + mp.mpf(a)
    base_coeffs = band_coeffs(zh[:M], zl[:M], M, ks_held, exact=True, rest=[np.asarray(p)[:M] for p in z8b[2:]])
    hfit = HeldFit(held_fit or os.path.join(HERE, "data", "held_fit.json"))
    held = [(mp.mpf(av) + da, mp.mpf(bv) + db) for (av, bv), (da, db) in zip(base_coeffs, hfit.increment(D, Dn, ks_held))]
    h = band_field(held, ks_held, M, parts=8)
    sn = make_solver(Dn, M, nodes=nodes, kt=kt, kc=kc, ks=ks, threads=threads, kfreeze=kf, horizon_alpha=0)
    sn.set_segs(m["segs"])
    spn = PinnedSolver(sn, M, h[0], h[1], h_rest=h[2:])
    so = PinnedO(sn, spn.ch, spn.cl, c8=spn.c8)
    rs = Reseeder(dict(M=M, nodes=nodes, kt=kt, kc=kc, ks=ks), Dn, threads=threads, horizon_alpha=0, qd=True, od=reseed_o)
    if jet:
        dj = np.load(os.path.join(raw, jet))
        Zo = [o8(tuple(dj[f"z{q}"][k] for q in range(8))) for k in range(dj["z0"].shape[0])]
        pred8 = horner_o(Zo, a)
        pred4 = q4_of_o(pred8)
    else:
        pred4 = tuple(np.ascontiguousarray(p) for p in load_q(os.path.join(raw, pred))[0])
        pred8 = o8_of_q(pred4)
    return dict(M=M, K=len(nodes), nodes=nodes, D=D, Dn=Dn, m=m, sn=sn, spn=spn, so=so, rs=rs, pred8=pred8, pred4=pred4,
                with_horizon=with_horizon)


def reseeded_start(att):
    """the prediction re-seeded as anm_run.py does it (--reseed 2 --reseed_q 1 --reseed_o 3): its horizon data put back
    in eight parts and refined when the re-seed converged, the prediction kept otherwise"""
    zr, rsd = att["rs"].reseed(att["pred4"])
    z0 = att["with_horizon"](att["pred8"], zr, att["rs"]) if rsd["ok"] else att["pred8"]
    return z0, rsd


def make_elim(att, record=False):
    """ElimO around the attempt's PinnedO (switch 1e-15), past its start (no StartNotProjected); with record every
    newton_o call keeps (z, st) in .calls"""
    from octo_corrector import ElimO

    class RecElimO(ElimO):
        def newton_o(self, z, mutate=0):
            st = super().newton_o(z, mutate=mutate)
            if record:
                self.calls.append((z, st))
            return st

        def project(self, z):
            # with cache (the probe): a point's projection is formed once (deterministic; counts only the misses)
            if self.cache is None:
                return super().project(z)
            key = b"".join(np.ascontiguousarray(np.asarray(p, dtype=float)).tobytes() for p in z)
            if key not in self.cache:
                self.cache[key] = super().project(z)
            return self.cache[key]
    el = RecElimO(att["so"], att["rs"])
    el.calls = []
    el.cache = None if record else {}
    el.first = False
    return el


def iterate(argv):
    from hpkernel.driver import qmax
    from octo_corrector import newton_od, o8, omax, q4_of_o
    from raw_out import require_raw
    raw, base = argv[0], argv[1]
    a = float(argv[argv.index("--a") + 1])
    jet = _opt(argv, "jet", "") or None
    pred = _opt(argv, "pred", "") or None
    out = _opt(argv, "out", "")
    threads, strong, steps = _opt(argv, "threads", 2), _opt(argv, "strong", 4), _opt(argv, "steps", 2)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    att = build_attempt(raw, base, a, threads=threads, jet=jet, pred=pred)
    name = os.path.join(out, f"crawl_iter_D{float(att['Dn']):.4f}.npz")
    require_raw(name)
    print(f"attempt: base {base}, D {att['D']}, a {a!r}, D + a {att['Dn']}, prediction from {'jet ' + jet if jet else 'pred ' + pred}, "
          f"{time.time() - t0:.1f} s", flush=True)
    code = 0
    chk = _opt(argv, "check_pred", "")
    if chk:
        d = np.load(os.path.join(raw, chk))
        same = all(np.array_equal(np.asarray(att["pred4"][q]), d[k]) for q, k in enumerate(("zh", "zl", "z2", "z3")))
        diff = max(float(np.abs(np.asarray(att["pred4"][q]) - d[k]).max()) for q, k in enumerate(("zh", "zl", "z2", "z3")))
        print(f"  the rebuilt prediction against {chk}: {'bit for bit equal' if same else 'DIFFERS, largest part difference ' + format(diff, '.2e')}",
              flush=True)
        code = 0 if same else 1
    z0, rsd = reseeded_start(att)
    print(f"  re-seed {'ok' if rsd['ok'] else 'FAILED, prediction kept'}, {rsd['seconds']} s; start max|F| (PinnedO) "
          f"{omax(att['so'].residual_o(z0)):.2e}", flush=True)
    el = make_elim(att, record=True)
    el.first = True
    z, hist, conv = newton_od(el, z0, tol=1e-60, it_max=steps, verbose=True, junk_w=0, lam_stall=0.04, accept="natural",
                              strong_chord=strong)
    print(f"  newton_od: {len(hist)} steps, converged {conv}, projections {el.counts}, {time.time() - t0:.0f} s", flush=True)
    if len(el.calls) < 2:
        print("  no second newton_o call (the chord was refused): the first step's trials ran instead; saving the start",
              flush=True)
        zi, st = el.calls[0]
    else:
        zi, st = el.calls[1]             # the iterate after the first step and its Newton step (the step 'on 1')
    zi, d = o8(zi), o8(st[0])
    sw = qmax(el.sq.residual_q(*q4_of_o(zi)))
    meta = dict(raw_base=base, a=a, D_str=str(att["D"]), Dn_str=str(att["Dn"]), jet=jet, pred=pred, threads=threads,
                strong=strong, steps=steps, reseed=dict(ok=rsd["ok"], seconds=rsd["seconds"]), counts=dict(el.counts),
                hist=hist,
                F_iter=float(np.abs(np.asarray(st[1]) + np.asarray(st[2])).max()), dz=float(np.abs(d[0]).max()),
                sw_iter=sw, seconds=round(time.time() - t0, 1))
    np.savez(name, **{f"z{q}": np.asarray(zi[q]) for q in range(8)}, **{f"d{q}": np.asarray(d[q]) for q in range(8)},
             meta=json.dumps(meta, default=float))
    print(f"  saved {name}: the iterate (max|F| of its projection {meta['F_iter']:.2e}, its own pinned quad-double max|F| "
          f"{sw:.2e}) and its step (|dz| {meta['dz']:.2e})", flush=True)
    return code


def row_name(r, M, K):
    """(block, field, point) of residual row r; the last row is the phase row"""
    if r == 4 * M * (K + 1):
        return ("phase", "", 0)
    b, rem = divmod(r, 4 * M)
    return (b, FIELDS[rem // M], rem % M)


def harmonics(v, M):
    """odd harmonics k = 1, 3, ..., M - 1 of a field on the half grid (pinning.band_coeffs' convention): |(a_k, b_k)|"""
    th = np.pi * np.arange(M) / M
    ks = np.arange(1, M, 2)
    a = (2.0 / M) * np.cos(np.outer(ks, th)) @ v
    b = (2.0 / M) * np.sin(np.outer(ks, th)) @ v
    return ks, np.hypot(a, b)


def per_block(dev, M, K):
    """max |dev| per (block, field), the phase row apart"""
    out = {}
    for b in range(K + 1):
        for f in range(4):
            seg = dev[4 * M * b + M * f:4 * M * b + M * (f + 1)]
            out[f"{b}{FIELDS[f]}"] = float(np.abs(seg).max())
    out["phase"] = float(abs(dev[-1]))
    return out


def probe(argv):
    from hpkernel.driver import qmax
    from multidouble import od_axpy
    from octo_corrector import o8, omax, q4_of_o
    from raw_out import require_raw
    raw, it_path = argv[0], argv[1]
    out = _opt(argv, "out", "")
    threads = _opt(argv, "threads", 2)
    lo, hi, rungs = _opt(argv, "lo", 0.5), _opt(argv, "hi", 1.0), _opt(argv, "rungs", 6)
    tol_lam = _opt(argv, "tol_lam", 1e-6)
    arms = _opt(argv, "arms", "U,E").split(",")
    os.makedirs(out, exist_ok=True)
    dd = np.load(it_path)
    meta = json.loads(str(dd["meta"]))
    z = o8(tuple(dd[f"z{q}"] for q in range(8)))
    d = o8(tuple(dd[f"d{q}"] for q in range(8)))
    tag = os.path.basename(it_path).replace("crawl_iter_", "").replace(".npz", "")
    rec_path = os.path.join(out, f"crawl_probe_{tag}.jsonl")
    require_raw(rec_path)
    t0 = time.time()
    att = build_attempt(raw, meta["raw_base"], meta["a"], threads=threads, jet=meta.get("jet"), pred=meta.get("pred"))
    M, K, so, rs = att["M"], att["K"], att["so"], att["rs"]
    el = make_elim(att)
    print(f"probe: {it_path}, D + a {att['Dn']}, |dz| {float(np.abs(d[0]).max()):.3e}, arms {arms}, {time.time() - t0:.1f} s", flush=True)

    def sw_of(x):
        """the switch's test value: the pinned quad-double max |F| of x's quad-double rounding (ElimO.project)"""
        return qmax(el.sq.residual_q(*q4_of_o(x)))

    def forced(x):
        """x with its horizon data projected whatever its residual (ElimO.project without the switch)"""
        zp, how = rs.project(q4_of_o(x))
        return (x if how == "failed" else att["with_horizon"](x, zp, rs)), how

    def lead(F):
        return np.asarray(F[0]) + np.asarray(F[1])

    def ev_arm(arm):
        def ev(lam):
            zt = o8(od_axpy(lam, *d, *z))
            t1 = time.time()
            sw = sw_of(zt)
            if arm == "E":
                n0 = dict(el.counts)
                ztp = el.project(zt)
                how = next((k for k in el.counts if el.counts[k] != n0[k]), "?")
            elif arm == "P":
                ztp, how = forced(zt)
            else:
                ztp, how = zt, "never"
            F = so.residual_o(ztp)
            return lead(F), dict(sw=sw, how=how, s=round(time.time() - t1, 1))
        return ev
    ladder = [lo + (hi - lo) * i / (rungs - 1) for i in range(rungs)]
    res = dict(iter=it_path, meta=meta, ladder=ladder, arms={})
    sws = {}

    def sw_at(lam):
        if lam not in sws:
            sws[lam] = sw_of(o8(od_axpy(lam, *d, *z)))
        return sws[lam]

    def sw_hint(la, lb):
        """the bracket lb - (1 +- 0.1) mu* around the lam = lb - mu* where the switch's test value crosses el.switch
        between la and lb, bisected on that value (which needs no projection) in log mu, mu = lb - lam, so that a
        crossing a few ulps below lb is resolved; None when it does not cross there"""
        mu_hi, mu_lo = lb - la, 8 * np.spacing(lb)
        if not (sw_at(la) > el.switch >= sw_at(lb)) or sw_at(lb - mu_lo) > el.switch:
            return None
        for _ in range(48):
            mu_m = float(np.sqrt(mu_hi * mu_lo))
            if sw_at(lb - mu_m) > el.switch:
                mu_hi = mu_m
            else:
                mu_lo = mu_m
            if mu_hi / mu_lo < 1.02:
                break
        mu = float(np.sqrt(mu_hi * mu_lo))
        res["sw_crossing"] = dict(mu=mu, lam=lb - mu, n_sw=len(sws), sw=sorted([[k, v] for k, v in sws.items()]))
        print(f"    hint: the switch's test value crosses {el.switch:.0e} at lam = {lb:g} - {mu:.4e} ({len(sws)} evaluations of it)",
              flush=True)
        return (lb - 1.1 * mu, lb - 0.9 * mu)
    for arm in arms:
        ev = ev_arm(arm)
        F0, info0 = ev(0.0)
        print(f"  arm {arm}: at lam 0 max|F| {float(np.abs(F0).max()):.3e} (switch test {info0['sw']:.3e}, {info0['how']})", flush=True)
        r = locate_jump(ev, F0, ladder, tol_lam=tol_lam, hint=sw_hint if (arm == "E" and _opt(argv, "hint", 1)) else None)
        if r.get("hint"):
            print(f"    hint {'taken' if r['hint']['taken'] else 'refused'}: {r['hint']['bracket']}", flush=True)
        ar = dict(F0_max=float(np.abs(F0).max()), lam0=info0, lam_star=r["lam_star"], g_max=r["g_max"], growth=r["growth"],
                  evals=r["evals"], hint=r.get("hint"))
        if r["lam_star"] is None:
            print(f"  arm {arm}: affine to lam = {hi:g} (largest ladder growth {r['g_max']:.2e}, K {r['K']:.0e}); "
                  f"e at the end {r['e_end']:.3e}", flush=True)
        else:
            dev_a = r["F_a"] - (1 - r["lam_a"]) * F0
            dev_b = r["F_b"] - (1 - r["lam_b"]) * F0
            jump = dev_b - dev_a
            rows = carriers(jump)
            ar.update(lam_a=r["lam_a"], lam_b=r["lam_b"], width=r["width"], e_a=r["e_a"], e_b=r["e_b"],
                      blocks_a=per_block(dev_a, M, K), blocks_b=per_block(dev_b, M, K), n_rows99=len(rows),
                      rows99=[(r_, *row_name(r_, M, K), float(jump[r_])) for r_ in rows[:40]])
            print(f"  arm {arm}: lam* {r['lam_star']:.9f}, 1 - lam* {1 - r['lam_star']:.4e} (bracket 1 - lam {1 - r['lam_a']:.4e} .. "
                  f"{1 - r['lam_b']:.4e}, width {r['width']:.2e}), e {r['e_a']:.3e} -> {r['e_b']:.3e}", flush=True)
            top_a = sorted(ar["blocks_a"].items(), key=lambda kv: -kv[1])[:6]
            top_b = sorted(ar["blocks_b"].items(), key=lambda kv: -kv[1])[:6]
            print("    |F - (1 - lam) F0| per block, below lam*: " + ", ".join(f"{k} {v:.2e}" for k, v in top_a), flush=True)
            print("    |F - (1 - lam) F0| per block, above lam*: " + ", ".join(f"{k} {v:.2e}" for k, v in top_b), flush=True)
            print(f"    {len(rows)} rows carry 99% of the jump; the first: "
                  + ", ".join(f"{b}{f}[{j}] {v:+.2e}" for _, b, f, j, v in ar["rows99"][:10]), flush=True)
            # the jump's harmonic content per (block, field) that carries it
            hb = {}
            for r_ in rows:
                b, f, _ = row_name(r_, M, K)
                if b != "phase":
                    hb.setdefault((b, f), None)
            ar["harmonics"] = {}
            for (b, f) in hb:
                o = 4 * M * b + M * FIELDS.index(f)
                ks_, amp = harmonics(jump[o:o + M], M)
                ar["harmonics"][f"{b}{f}"] = {int(k): float(x) for k, x in zip(ks_, amp)}
                top = sorted(zip(ks_, amp), key=lambda kv: -kv[1])[:4]
                print(f"    block {b} field {f}: largest harmonics " + ", ".join(f"k {int(k)} {x:.2e}" for k, x in top), flush=True)
        if arm == "E" and r["lam_star"] is not None:
            # the same point just above lam* with its horizon data projected (arm P there): whether the projection
            # alone takes the residual back to the affine model
            Fp, infop = ev_arm("P")(r["lam_b"])
            dev_p = Fp - (1 - r["lam_b"]) * F0
            ar.update(forced_b=dict(lam=r["lam_b"], e=float(np.abs(dev_p).max()), blocks=per_block(dev_p, M, K), **infop))
            top_p = sorted(ar["forced_b"]["blocks"].items(), key=lambda kv: -kv[1])[:6]
            print(f"    at lam_b (1 - lam {1 - r['lam_b']:.4e}) projected ({infop['how']}): e {ar['forced_b']['e']:.3e}; per block "
                  + ", ".join(f"{k} {v:.2e}" for k, v in top_p), flush=True)
        res["arms"][arm] = ar
        with open(rec_path, "a") as fh:
            fh.write(json.dumps(dict(arm=arm, **ar), default=float) + "\n")
    if _opt(argv, "theta", 1):
        # the natural test at lam = 1 with the trial projected: the LU at the iterate's projection (ElimO.newton_o, as
        # newton_od forms it), the correction (zt' - zt) - J^-1 F(zt') against |dz|
        st = el.newton_o(z)
        dz_new = o8(st[0])
        same = all(np.array_equal(np.asarray(dz_new[q]), np.asarray(d[q])) for q in range(8))
        nd = float(np.abs(np.asarray(d[0])).max())
        th = {}
        for lam, modes in ((1.0, ("switch", "forced")),) + (((0.5, ("switch",)),) if _opt(argv, "theta_half", 0) else ()):
            zt = o8(od_axpy(lam, *d, *z))
            for mode in modes:
                ztp = el.project(zt) if mode == "switch" else forced(zt)[0]
                disp = o8(od_axpy(-1.0, *zt, *ztp))
                Ft = o8(so.residual_o(ztp))
                xb = o8(so.olu_solve([np.ascontiguousarray(q) for q in Ft]))
                db = o8(od_axpy(-1.0, *xb, *disp))
                th[f"{lam:g}_{mode}"] = dict(theta=float(np.abs(np.asarray(db[0])).max()) / nd, maxF=omax(Ft),
                                            bound=1 - lam / 4)
                print(f"  natural test at lam {lam:g}, trial {'as the corrector projects it' if mode == 'switch' else 'projected'}: "
                      f"max|F| {omax(Ft):.3e}, simplified correction / |dz| {th[f'{lam:g}_{mode}']['theta']:.3e} (bound {1 - lam / 4:.3f})",
                      flush=True)
        res["theta"] = dict(step_reproduced=same, cases=th)
        print(f"  the step at the iterate formed again: {'bit for bit the stored one' if same else 'DIFFERS from the stored one'}", flush=True)
        with open(rec_path, "a") as fh:
            fh.write(json.dumps(dict(theta=res["theta"]), default=float) + "\n")
    print(f"done, {time.time() - t0:.0f} s; records {rec_path}", flush=True)
    return 0


def fixrun(argv):
    from octo_corrector import delta_digits, newton_od, o8, omax
    from raw_out import require_raw
    raw, it_path = argv[0], argv[1]
    out = _opt(argv, "out", "")
    threads, steps, tol, switch = _opt(argv, "threads", 2), _opt(argv, "steps", 6), _opt(argv, "tol", 1e-60), _opt(argv, "switch", 0.0)
    os.makedirs(out, exist_ok=True)
    dd = np.load(it_path)
    meta = json.loads(str(dd["meta"]))
    z = o8(tuple(dd[f"z{q}"] for q in range(8)))
    tag = os.path.basename(it_path).replace("crawl_iter_", "").replace(".npz", "")
    name = os.path.join(out, f"fixrun_{tag}_switch{switch:g}.npz")
    require_raw(name)
    t0 = time.time()
    att = build_attempt(raw, meta["raw_base"], meta["a"], threads=threads, jet=meta.get("jet"), pred=meta.get("pred"))
    el = make_elim(att, record=True)
    el.cache = None
    el.switch = switch
    print(f"fixrun: {it_path}, D + a {att['Dn']}, ElimO switch {switch:g}, natural test, at most {steps} steps to {tol:.0e}",
          flush=True)
    zf, hist, conv = newton_od(el, z, tol=tol, it_max=steps, verbose=True, junk_w=0, lam_stall=0.04, accept="natural")
    zp = el.project(zf)
    Fp, Fz = omax(att["so"].residual_o(zp)), omax(att["so"].residual_o(zf))
    print(f"  {len(hist)} steps, converged {conv}, projections {el.counts}; pinned octo-double max|F| of the final point's "
          f"projection {Fp:.2e}, of the point itself {Fz:.2e}; Delta {delta_digits(zp)}; {time.time() - t0:.0f} s", flush=True)
    np.savez(name, **{f"z{q}": np.asarray(zp[q]) for q in range(8)},
             meta=json.dumps(dict(iter=it_path, switch=switch, steps=steps, tol=tol, converged=bool(conv), hist=hist,
                                  F_proj=Fp, F_point=Fz, Delta=delta_digits(zp), counts=dict(el.counts),
                                  seconds=round(time.time() - t0, 1)), default=float))
    return 0 if conv else 1


def unknown_blocks(v, M, K):
    """max |v| per unknown group of the layout: centre data (p1, c0), node i's fields, m_h, a_h, Delta"""
    v = np.asarray(v, dtype=float)
    out = {"p1": float(np.abs(v[:M]).max()), "c0": float(np.abs(v[M:2 * M]).max())}
    for i in range(K):
        for f in range(4):
            o = 2 * M + 4 * M * i + M * f
            out[f"n{i}{FIELDS[f]}"] = float(np.abs(v[o:o + M]).max())
    o = 2 * M + 4 * M * K
    out["m_h"] = float(np.abs(v[o:o + M]).max())
    out["a_h"] = float(np.abs(v[o + M:o + 2 * M]).max())
    out["Delta"] = float(abs(v[-1]))
    return out


def offset(argv):
    from multidouble import od_axpy
    from octo_corrector import o8, omax
    raw, it_path = argv[0], argv[1]
    threads = _opt(argv, "threads", 2)
    dd = np.load(it_path)
    meta = json.loads(str(dd["meta"]))
    z = o8(tuple(dd[f"z{q}"] for q in range(8)))
    d = o8(tuple(dd[f"d{q}"] for q in range(8)))
    t0 = time.time()
    att = build_attempt(raw, meta["raw_base"], meta["a"], threads=threads, jet=meta.get("jet"), pred=meta.get("pred"))
    M, K, so = att["M"], att["K"], att["so"]
    el = make_elim(att)
    el.switch = 0.0
    zp = el.project(z)
    disp = o8(od_axpy(-1.0, *z, *zp))
    step = o8(od_axpy(-1.0, *disp, *d))
    lead = lambda v: np.asarray(v[0]) + np.asarray(v[1])  # noqa: E731
    print(f"offset: {it_path}, projection {el.counts}, {time.time() - t0:.0f} s", flush=True)
    for name, v in (("dz", d), ("P(z) - z", disp), ("dz - (P(z) - z), the step from P(z)", step)):
        ub = unknown_blocks(lead(v), M, K)
        print(f"  {name}: max {max(ub.values()):.3e}; " + ", ".join(f"{k} {x:.2e}" for k, x in ub.items() if x > 0), flush=True)
    o = 2 * M + 4 * M * K
    for nm, sl in (("m_h", slice(o, o + M)), ("a_h", slice(o + M, o + 2 * M))):
        ks_, amp = harmonics(lead(disp)[sl], M)
        print(f"  P(z) - z in {nm} by harmonic: " + ", ".join(f"k {int(k)} {x:.2e}" for k, x in zip(ks_, amp)), flush=True)
    Fz, Fpz = lead(so.residual_o(z)), lead(so.residual_o(zp))
    for nm, F in (("F(z)", Fz), ("F(P(z))", Fpz)):
        pb = per_block(F, M, K)
        top = sorted(pb.items(), key=lambda kv: -kv[1])[:8]
        print(f"  {nm}: max {omax((F, np.zeros_like(F))):.3e}; per block " + ", ".join(f"{k} {x:.2e}" for k, x in top), flush=True)
    return 0


# ------------------------------------------------------------------ the strong-row chord, point by point (#715)
def groups(M, K):
    """index arrays of the unknown groups: centre data, node i, the last node, m_h, a_h, Delta"""
    o = 2 * M + 4 * M * K
    g = {"centre": np.arange(0, 2 * M), "nodes": np.arange(2 * M, o), "last node": np.arange(o - 4 * M, o),
         "m_h": np.arange(o, o + M), "a_h": np.arange(o + M, o + 2 * M), "Delta": np.array([o + 2 * M])}
    g["horizon data"] = np.r_[g["m_h"], g["a_h"]]
    return g


def trace_elim(att):
    """ElimO around the attempt's PinnedO (switch 1e-15) that keeps each point it is asked to project, in order, with
    its switch's test value, how it was projected and the projection (formed once per point), and each one-argument
    olu_solve (the chord's J0^-1 F and the natural test's) with its result. The corrector's arithmetic is unchanged."""
    from hpkernel.driver import qmax
    from octo_corrector import ElimO, o8, q4_of_o

    class TraceElimO(ElimO):
        def project(self, z):
            z = o8(z)
            key = b"".join(np.ascontiguousarray(np.asarray(p, dtype=float)).tobytes() for p in z)
            if key not in self.cache:
                n0 = dict(self.counts)
                t1 = time.time()
                zp = super().project(z)
                how = next((k for k in self.counts if self.counts[k] != n0[k]), "?")
                self.cache[key] = (zp, how, qmax(self.sq.residual_q(*q4_of_o(z))), round(time.time() - t1, 1))
            zp, how, sw, sec = self.cache[key]
            self.points.append(dict(z=z, zp=zp, how=how, sw=sw, s=sec))
            return zp

        def olu_solve(self, *args):
            r = self.so.olu_solve(*args)
            if len(args) == 1:
                self.solves.append((o8(args[0]), o8(r)))
            return r
    el = TraceElimO(att["so"], att["rs"])
    el.points, el.solves, el.cache = [], [], {}
    return el


def drop_V(V, x):
    """chord_strong's drop_V, the same two lines: x less V V^T x (leading parts, f64), twice"""
    from multidouble import od_axpy
    from octo_corrector import o8
    zero8 = np.zeros(len(x[0]))
    for _ in range(2):
        c = V @ (V.T @ (np.asarray(x[0]) + np.asarray(x[1])))
        x = o8(od_axpy(-1.0, np.ascontiguousarray(c), *([zero8] * 7), *x))
    return x


def drop_V_exact(V, x, dps=160):
    """x less V V^T x twice in mpmath at dps digits, with V's f64 entries and all eight parts of x: drop_V without
    its f64 rounding (the same V). Returned in eight parts."""
    import mpmath as mp
    from octo_corrector import o8
    with mp.workdps(dps):
        L, w = V.shape
        xs = [mp.fsum(mp.mpf(float(x[q][i])) for q in range(8)) for i in range(L)]
        Vm = [[mp.mpf(float(V[i, j])) for j in range(w)] for i in range(L)]
        for _ in range(2):
            c = [mp.fsum(Vm[i][j] * xs[i] for i in range(L)) for j in range(w)]
            xs = [xs[i] - mp.fsum(Vm[i][j] * c[j] for j in range(w)) for i in range(L)]
        out = [np.zeros(L) for _ in range(8)]
        for i in range(L):
            v = xs[i]
            for q in range(8):
                f = float(v)
                out[q][i] = f
                v = v - mp.mpf(f)
    return o8(out)


def chord(argv):
    """the attempt's start, its first Newton step and the strong-row chord (chord_strong, as newton_od calls it under
    --od_strong), with every chord point, its projection (forced where the switch skipped it), each chord solve
    J0^-1 F and the weak subspace V kept; then chordstat on them"""
    import octo_corrector as oc
    from octo_corrector import o8
    from raw_out import require_raw
    raw, base = argv[0], argv[1]
    a = float(argv[argv.index("--a") + 1])
    jet, pred, out = _opt(argv, "jet", "") or None, _opt(argv, "pred", "") or None, _opt(argv, "out", "")
    threads, strong, chord_it = _opt(argv, "threads", 2), _opt(argv, "strong", 4), _opt(argv, "chord_it", 10)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    att = build_attempt(raw, base, a, threads=threads, jet=jet, pred=pred)
    name = os.path.join(out, f"chord_trace_D{float(att['Dn']):.4f}.npz")
    require_raw(name)
    print(f"chord: base {base}, D + a {att['Dn']}, prediction from {'jet ' + jet if jet else 'pred ' + pred}", flush=True)
    code = 0
    chk = _opt(argv, "check_pred", "")
    if chk:
        d = np.load(os.path.join(raw, chk))
        same = all(np.array_equal(np.asarray(att["pred4"][q]), d[k]) for q, k in enumerate(("zh", "zl", "z2", "z3")))
        print(f"  the rebuilt prediction against {chk}: {'bit for bit equal' if same else 'DIFFERS'}", flush=True)
        code |= 0 if same else 1
    z0, rsd = reseeded_start(att)
    el = trace_elim(att)
    st = el.newton_o(z0)                      # newton_od's first call, the start's LU (J0)
    dz = o8(st[0])
    nd = float(np.abs(np.asarray(dz[0])).max())
    print(f"  re-seed {'ok' if rsd['ok'] else 'FAILED, prediction kept'}; start max|F| (PinnedO) "
          f"{float(np.abs(np.asarray(st[1]) + np.asarray(st[2])).max()):.2e}, |dz| {nd:.3e}, {time.time() - t0:.0f} s",
          flush=True)
    junk = {}
    orig = oc.junk_subspace_o

    def junk_rec(s, w, *args, **kw):
        r = orig(s, w, *args, **kw)
        junk.update(V=r[0], sig=r[1], sig_other=r[2])
        return r
    oc.junk_subspace_o = junk_rec
    try:
        zc, ok_c, info = oc.chord_strong(el, z0, dz, strong, chord_it, 1e-60, 1e-45, 0, True)
    finally:
        oc.junk_subspace_o = orig
    n_chord = len(el.solves)
    Fz = o8(el.residual_o(zc))                # newton_od's natural test at the chord's point, through J0's LU
    xb = o8(el.olu_solve([np.ascontiguousarray(q) for q in Fz]))
    theta = float(np.abs(np.asarray(xb[0])).max()) / nd
    print(f"  chord: {len(info['steps'])} steps, ok {ok_c}; theta at its point {theta:.3e} (#700: 7.993e-12), "
          f"{time.time() - t0:.0f} s", flush=True)
    chk = _opt(argv, "check_iter", "")
    if chk:
        di = np.load(os.path.join(raw, chk))
        same = all(np.array_equal(np.asarray(zc[q]), di[f"z{q}"]) for q in range(8))
        print(f"  the chord's point against {chk}: {'bit for bit equal' if same else 'DIFFERS'}", flush=True)
        code |= 0 if same else 1
    # the points: the start, then each point the chord evaluated, then its last point (the natural test's)
    pts = el.points
    print("  points projected (in order): " + ", ".join(f"{p['how']} (switch test {p['sw']:.2e})" for p in pts), flush=True)
    forced = []
    for k, p in enumerate(pts):
        if p["how"] == "skipped" and not any(p["z"] is q["z"] for q in pts[:k]):
            t1 = time.time()
            zp4, how = att["rs"].project(oc.q4_of_o(p["z"]))
            zp = p["z"] if how == "failed" else att["with_horizon"](p["z"], zp4, att["rs"])
            forced.append(k)
            p.update(zp_forced=zp, how_forced=how, s_forced=round(time.time() - t1, 1))
            print(f"  point {k} forced: projection {how}, {p['s_forced']} s", flush=True)
    sav = {"meta": json.dumps(dict(raw_base=base, a=a, jet=jet, pred=pred, threads=threads, strong=strong,
                                   chord_it=chord_it, n_chord_solves=n_chord, chord=info, theta=theta, ok=bool(ok_c),
                                   points=[dict(how=p["how"], sw=p["sw"], s=p["s"], how_forced=p.get("how_forced"))
                                           for p in pts], counts=dict(el.counts), seconds=round(time.time() - t0, 1)),
                              default=float),
           "V": junk["V"], "sig": junk["sig"], "sig_other": junk["sig_other"]}
    for k, p in enumerate(pts):
        zp = p.get("zp_forced", p["zp"])
        for q in range(8):
            sav[f"p{k}_z{q}"] = np.asarray(p["z"][q])
            sav[f"p{k}_P{q}"] = np.asarray(zp[q])
    for j, (F, x) in enumerate(el.solves[:n_chord]):
        for q in range(8):
            sav[f"x{j}_{q}"] = np.asarray(x[q])
    for q in range(8):
        sav[f"zc_{q}"] = np.asarray(zc[q])
        sav[f"dz_{q}"] = np.asarray(dz[q])
    np.savez(name, **sav)
    print(f"  saved {name}, {time.time() - t0:.0f} s", flush=True)
    code |= chordstat([raw, name] + (["--threads", str(threads)]))
    return code


def chordstat(argv):
    """the trace of chord(): at each point the switch's test value, its own residual and its projection's by block,
    and P(z) - z by unknown group with m_h and a_h by harmonic; at each chord step x = J0^-1 F, V^T x, dw by group,
    drop_V's f64 rounding (dw against drop_V in mpmath with the same V), and the switch's test value at z - dw with
    only one group of dw applied, with the exact dw, and with V rounded differently (V + 1e-16 noise, QR again)"""
    from hpkernel.driver import qmax
    from multidouble import od_axpy
    from octo_corrector import PinnedO, o8, q4_of_o  # noqa: F401
    from pinning import PinnedQ
    from raw_out import require_raw
    raw, path = argv[0], argv[1]
    threads = _opt(argv, "threads", 2)
    dd = np.load(path)
    meta = json.loads(str(dd["meta"]))
    att = build_attempt(raw, meta["raw_base"], meta["a"], threads=threads, jet=meta.get("jet"), pred=meta.get("pred"))
    M, K, so = att["M"], att["K"], att["so"]
    sq = PinnedQ(so.s, so.ch, so.cl)
    rec_path = path.replace(".npz", ".jsonl")
    require_raw(rec_path)
    V = dd["V"]
    G = groups(M, K)
    n = len(meta["points"])
    Z = [o8(tuple(dd[f"p{k}_z{q}"] for q in range(8))) for k in range(n)]
    P = [o8(tuple(dd[f"p{k}_P{q}"] for q in range(8))) for k in range(n)]
    X = [o8(tuple(dd[f"x{j}_{q}"] for q in range(8))) for j in range(meta["n_chord_solves"])]
    lead = lambda v: np.asarray(v[0]) + np.asarray(v[1])  # noqa: E731

    def sw(z):
        return qmax(sq.residual_q(*q4_of_o(z)))

    def sub(z, v):
        return o8(od_axpy(-1.0, *v, *z))

    def mask(v, idx):
        m = np.zeros(len(v[0]), dtype=bool)
        m[idx] = True
        return o8(tuple(np.where(m, np.asarray(p), 0.0) for p in v))

    def gmax(v):
        lv = lead(v)
        return {g: float(np.abs(lv[ix]).max()) for g, ix in G.items()}

    def harm(v, g):
        ks_, amp = harmonics(lead(v)[G[g]], M)
        return {int(k): float(x) for k, x in zip(ks_, amp)}

    def hrows(F):
        """the horizon block's rows of a residual by field, max and by harmonic"""
        o = 4 * M * K
        Fl = lead(F)
        out = {}
        for f in range(4):
            seg = Fl[o + M * f:o + M * (f + 1)]
            ks_, amp = harmonics(seg, M)
            out[FIELDS[f]] = dict(max=float(np.abs(seg).max()), harm={int(k): float(x) for k, x in zip(ks_, amp)})
        return out
    names = ["start z0"] + [f"chord point {k - 1}" for k in range(1, n)]
    recs = []
    print(f"chordstat: {path}; M {M}, K {K}; V {V.shape}, sigma {', '.join(f'{s:.2e}' for s in dd['sig'])}", flush=True)
    print("  the points (the switch's test value is the pinned quad-double max |F| of the point itself):", flush=True)
    for k in range(n):
        disp = o8(od_axpy(-1.0, *Z[k], *P[k]))
        Fo, Fp = so.residual_o(Z[k]), so.residual_o(P[k])
        r = dict(kind="point", k=k, name=names[k], how=meta["points"][k]["how"], how_forced=meta["points"][k]["how_forced"],
                 sw=sw(Z[k]), F_own=per_block(lead(Fo), M, K), F_proj_max=float(np.abs(lead(Fp)).max()),
                 F_proj=per_block(lead(Fp), M, K), hrows_own=hrows(Fo), disp=gmax(disp), disp_mh=harm(disp, "m_h"),
                 disp_ah=harm(disp, "a_h"))
        recs.append(r)
        top = sorted(r["F_own"].items(), key=lambda kv: -kv[1])[:4]
        print(f"  [{k}] {names[k]}: switch test {r['sw']:.3e} ({r['how']}{', forced ' + r['how_forced'] if r['how_forced'] else ''}); "
              f"own F by block " + ", ".join(f"{b} {v:.2e}" for b, v in top) + f"; F(P(z)) {r['F_proj_max']:.2e}", flush=True)
        print("      own F in the horizon rows by harmonic: " + "; ".join(
            f"{f} " + ", ".join(f"k{kk} {v:.1e}" for kk, v in sorted(r["hrows_own"][f]["harm"].items(), key=lambda kv: -kv[1])[:4])
            for f in ("m", "a", "c")), flush=True)
        print(f"      P(z) - z: m_h {r['disp']['m_h']:.3e}, a_h {r['disp']['a_h']:.3e}, other {max(r['disp'][g] for g in ('centre', 'nodes', 'Delta')):.1e}; "
              f"a_h by harmonic " + ", ".join(f"k{kk} {v:.1e}" for kk, v in r["disp_ah"].items()), flush=True)
        print("      m_h by harmonic " + ", ".join(f"k{kk} {v:.1e}" for kk, v in r["disp_mh"].items()), flush=True)
    # the chord steps: point j + 1 (pts index) is chord point j; chord point j + 1 = chord point j - dw_j
    print("  the chord steps:", flush=True)
    rng = np.random.default_rng(1)
    Vp = np.linalg.qr(V + 1e-16 * rng.standard_normal(V.shape))[0]
    for j, x in enumerate(X):
        zj, zn = Z[j + 1], Z[j + 2]
        dw = drop_V(V, x)
        same = all(np.array_equal(np.asarray(p), np.asarray(q)) for p, q in zip(sub(zj, dw), zn))
        dwx = drop_V_exact(V, x)
        res_ = o8(od_axpy(-1.0, *dwx, *dw))         # drop_V's rounding, dw - dw_exact
        dwp = drop_V(Vp, x)
        vtx = V.T @ lead(x)
        r = dict(kind="step", j=j, reproduced=bool(same), x=gmax(x), x_max=float(np.abs(lead(x)).max()),
                 VTx=[float(v) for v in vtx], dw=gmax(dw), dw_max=float(np.abs(lead(dw)).max()),
                 dw_ah=harm(dw, "a_h"), dw_mh=harm(dw, "m_h"), rounding=gmax(res_), rounding_ah=harm(res_, "a_h"),
                 dw_exact_max=float(np.abs(lead(dwx)).max()), dw_Vp_max=float(np.abs(lead(dwp)).max()),
                 sw_before=sw(zj), sw_after=sw(zn))
        part = {}
        for g in ("centre", "nodes", "last node", "m_h", "a_h", "horizon data", "Delta"):
            part[g] = sw(sub(zj, mask(dw, G[g])))
        notg = np.setdiff1d(np.arange(len(dw[0])), G["horizon data"])
        part["all but horizon data"] = sw(sub(zj, mask(dw, notg)))
        part["exact dw"] = sw(sub(zj, dwx))
        part["exact dw, own F max"] = float(np.abs(lead(so.residual_o(sub(zj, dwx)))).max())
        part["dw with V rounded differently"] = sw(sub(zj, dwp))
        part["rounding only (dw - exact dw)"] = sw(sub(zj, res_))
        r["sw_partial"] = part
        recs.append(r)
        print(f"  step {j} (chord point {j} -> {j + 1}): reproduced {'bit for bit' if same else 'NOT'}; |x| {r['x_max']:.3e} "
              f"(a_h {r['x']['a_h']:.2e}, m_h {r['x']['m_h']:.2e}, nodes {r['x']['nodes']:.2e}); V^T x "
              + ", ".join(f"{v:+.2e}" for v in vtx), flush=True)
        print(f"      |dw| {r['dw_max']:.3e} (a_h {r['dw']['a_h']:.2e}, m_h {r['dw']['m_h']:.2e}, nodes {r['dw']['nodes']:.2e}, "
              f"Delta {r['dw']['Delta']:.2e}); exact drop_V |dw| {r['dw_exact_max']:.3e}; with V rounded differently "
              f"{r['dw_Vp_max']:.3e}", flush=True)
        print(f"      drop_V's rounding (dw - exact): a_h {r['rounding']['a_h']:.2e}, m_h {r['rounding']['m_h']:.2e}, "
              f"nodes {r['rounding']['nodes']:.2e}, centre {r['rounding']['centre']:.2e}", flush=True)
        print(f"      switch test {r['sw_before']:.3e} -> {r['sw_after']:.3e}; with one group of dw only: "
              + ", ".join(f"{g} {v:.2e}" for g, v in part.items()), flush=True)
        print("      dw in a_h by harmonic " + ", ".join(f"k{kk} {v:.1e}" for kk, v in r["dw_ah"].items()), flush=True)
    with open(rec_path, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r, default=float) + "\n")
    print(f"  records {rec_path}", flush=True)
    return 0 if all(r["reproduced"] for r in recs if r["kind"] == "step") else 1


if __name__ == "__main__":
    args = sys.argv[1:]
    cmds = {"selftest": selftest, "iterate": iterate, "probe": probe, "fixrun": fixrun, "offset": offset, "chord": chord,
            "chordstat": chordstat}
    if not args or args[0] not in cmds:
        print(__doc__)
        sys.exit(2)
    sys.exit(cmds[args[0]](args[1:]))

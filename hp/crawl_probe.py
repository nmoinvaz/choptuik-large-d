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


if __name__ == "__main__":
    args = sys.argv[1:]
    cmds = {"selftest": selftest, "iterate": iterate, "probe": probe, "fixrun": fixrun}
    if not args or args[0] not in cmds:
        print(__doc__)
        sys.exit(2)
    sys.exit(cmds[args[0]](args[1:]))

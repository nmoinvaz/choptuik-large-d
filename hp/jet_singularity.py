#!/usr/bin/env python3
"""The chain's Taylor series in D: noise in its middle orders, or a singularity of z(D) near the real axis (#634).

#603's chain (anm_run.py --jet_o 1, N = 24) records |z_5| to |z_8| growing 20 to 1000 times from D 225 to 245 while
|z_1| to |z_4| stay smooth. Noise moves when the precision or the base moves; a property of z(D) does not. This script
recomputes the series in such ways, compares them, and locates the nearest singularity of a scalar component's series
(Delta's coefficients, the last entry of each z_k, or one row) by Pade poles and by a two-term recurrence fit.

    python3 jet_singularity.py planted [--mutate drop_pole]
    python3 jet_singularity.py jet RAW STATE OUTPREFIX --variants od,qd,q4,pert:L:S,inject:L:S [--N 24] [--threads 4]
        [--held_fit data/held_fit.json]
    python3 jet_singularity.py compare REF.npz VAR.npz ... [--ks 1-12] [--mutate scale]
    python3 jet_singularity.py locate JET.npz ... [--row delta|max5|<index>] [--json OUT]
    python3 jet_singularity.py truth RAW JET.npz STATE.npz [STATE.npz ...]
    python3 jet_singularity.py preds RAW FOLDER [--mutate scale5] [--D_max D]
    python3 jet_singularity.py reexpand JET_B1.npz JET_B2.npz [--mutate scale5]
    python3 jet_singularity.py refusals RAW FOLDER [--log LOG] [--D_max D]
    python3 jet_singularity.py summary RAW [--out data/jet_634.json] [--D_max 245.5] [--D_beyond 253.6]

planted: the estimator on series with a known answer, through the code path that locate uses, with the criteria fixed
before any real series was read. Background: Delta-like ln(1 + a/243) with c_1 = -1.7e-4, plus a conjugate pair at
radius 7 (the records' tail decays about 7 per order), c_24 ~ 1e-28. P1: a conjugate pole pair at |s| 1.0, arg 60
degrees; P2: at 2.5, 25 degrees; P3: square-root branch points at 1.5, 45 degrees; each hidden below the background at
orders 1 to 4 and dominant from order 5 on. Relative noise 1e-6 on every coefficient. N1: the background with that
noise and no singularity within 4. N2: noise alone shaped like the data, orders 5 to 24 given the records' |z_k| at
D 245.14 with random signs (5 seeds). N3 (added after the records' norms were read, before any real series): the
background plus an entire oscillating component 2 Re C exp((0.1 + 2.3 i) a), which has no singularity. Pass: P1 and P2 recovered by Pade to 3% in |s| and 3 degrees in arg, and by
the recurrence fit to 5% and 5 degrees; P3 to 10% and 10 degrees by both; N1, N2 and N3 give no stable Pade pole
with |s| <= 4 at any seed. Each case also prints growth_law (singularity against entire fall, reported, not judged). --mutate drop_pole builds P1 without its pole, which must fail. Exit 1 on failure.

jet: the series at STATE as anm_run.py --jet_o 1 forms it (its coefficients_o and coefficients_q, the held fit's
Taylor coefficients in eight parts as --held_o 1 gives them), in each variant of --variants:
    od      the stored state: the chain's series (equal to anm_run.py --save_jet 2's bit for bit)
    qd      the quad-double series (coefficients_q, quad-double LU) at the state's first four parts
    q4      the octo-double series at the state rounded to quad-double (the qd series' base, in octo-double)
    pert:L:S  the octo-double series at z0 + J^-1 r, r random (seed S) with max |r| = L: the root of F = r, a base
            converged to level instead of to its own residual (the check: F(z0 + J^-1 r) - F(z0) - r is printed)
    inject:L:S  each order's right-hand side changed by L max|b_k| times a random vector (seed S, max 1) before
            its solve
Each saved as OUTPREFIX_<variant>.npz in the layout of anm_run.py's series files (keys z0..z7, (N + 1, L) each), the
variant in the meta.

compare: per order k, max|z_k|, the largest change max|z_k' - z_k| / max|z_k| and Delta's (the last entry), the
differences formed in octo-double. --mutate scale multiplies the second series' order 5 by 1 + 1e-6, which the
table must show (the comparison's control).

locate: for the chosen row's coefficients c_1..c_N, the Pade approximants [L/M] of sum c_k a^(k-1) with L + M + 1 = N,
M = 2..10, their poles with Im >= 0 that are not exact doublets (a numerator zero within 1e-30 of the pole, the
approximant's excess degrees) and not inside 0.85 of the root-test radius (the coefficients' upper envelope fitted
over the upper half of the orders: a pole well inside the disc of convergence is the approximant's, since the nearest
singularity lies on its circle); a pole is stable when poles of at least three M lie within 5% of it, and the nearest stable one is reported as (|s|, arg s) with the
spread of its cluster. The recurrence fit: c_k = p c_(k-1) + q c_(k-2) by least squares over windows of 6 orders, its
roots 1/s; the median over windows from order 5 on and the windows' spread. Domb-Sykes: the ratios c_(k-1)/c_k
against 1/k. The singularity's place in D is D0 + s.

truth: the series at a base summed at each later state's a (Horner in octo-double, anm_run.horner_o) against that
converged state: the prediction's error per component, its largest entry and Delta's, against the middle orders'
terms |z_k| a^k (k 5..8) and the truncation |z_N| a^N. A converged state reached by a later step tests the series at a
refused step's a as well.

preds: the chain's own stored predictions against its converged states, from the records alone (preds' docstring).
summary: hp/data/jet_634.json, the numbers notes/LARGE_D_WALL.md quotes, from the raw-data repository's I634-mac
(clone it first: git clone https://github.com/nmoinvaz/choptuik-large-d-raw ../choptuik-large-d-raw).

Methods: Pade approximants (G. A. Baker, Jr. and P. Graves-Morris, Pade Approximants, 2nd ed., Cambridge University
Press, 1996); the ratio method of C. Domb and M. F. Sykes, Proc. R. Soc. Lond. A 240, 214 (1957),
doi:10.1098/rspa.1957.0078; a conjugate pair's modulated coefficients after C. Hunter and B. Guerrieri, SIAM J. Appl.
Math. 39, 248 (1980), doi:10.1137/0139022.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402

DPS = 80                 # the estimator's working precision
DOUBLET = 1e-30          # a numerator zero within this fraction of a pole (in mpmath) marks an exact doublet, the
                         # excess degrees of an [L/M] above the series' own; a genuine pole of small residue keeps its
                         # zero much farther (6e-11 of |s| for P1, residue 1e-14 against c_1 1.7e-4)
STABLE_TOL = 0.05        # poles of different M within this fraction of each other belong to one cluster
STABLE_MIN = 3           # a cluster of at least this many M is a stable pole
INSIDE = 0.85            # a pole nearer than this fraction of the root-test radius lies inside the disc of convergence
R_RANGE = 4.0            # "within the range": the step bound's reach (a_max 2.8-3.3 in #603's records)
MS = range(2, 11)
WIN = 6


# ------------------------------------------------------------------ estimators
def _roots(cs):
    """roots of the polynomial with coefficients cs (lowest first), [] when it has degree < 1 or does not converge"""
    cs = list(cs)
    while len(cs) > 1 and cs[-1] == 0:
        cs.pop()
    if len(cs) < 2:
        return []
    for steps in (200, 800):
        try:
            return list(mp.polyroots(cs[::-1], maxsteps=steps, extraprec=4 * DPS))
        except mp.libmp.libhyper.NoConvergence:
            continue
    return []


def pade_poles(c, M):
    """the [L/M] Pade approximant of g(a) = sum_k c_k a^(k-1) (c = [c_1..c_N], mpf), L = N - 1 - M: (poles, zeros)"""
    g = list(c)
    L = len(g) - 1 - M
    if L < 0:
        return [], []
    try:
        p, q = mp.pade(g, L, M)
    except ZeroDivisionError:
        return [], []
    return _roots(q), _roots(p)


def root_radius(c, k_lo=None):
    """the root test's radius of sum c_k a^k: exp(-slope) of ln of the coefficients' upper envelope (the largest |c|
    of k - 1, k, k + 1, which a conjugate pair's cos(k theta) dips need) fitted over orders k_lo..N (N/2 by default)"""
    v = np.abs(np.array([float(x) for x in c]))
    N = len(v)
    k = np.arange(1, N + 1)
    env = np.array([v[max(0, i - 1):i + 2].max() for i in range(N)])
    sel = (k >= (k_lo or N // 2)) & (env > 0)
    return float(np.exp(-np.polyfit(k[sel], np.log(env[sel]), 1)[0]))


def stable_poles(c, Ms=MS, r_max=R_RANGE * 2):
    """the clusters of Pade poles (Im >= 0, not doublets, INSIDE rho <= |s| <= r_max with rho the root-test radius)
    recurring over the M of Ms: a list of dicts (s median, abs, arg_deg, n, spread, Ms), nearest first. Doublets: a
    numerator zero within DOUBLET |s|. A pole well inside the disc of convergence is not a singularity of the series'
    function (the nearest one lies on the circle, Cauchy-Hadamard), so it is the approximant's"""
    r_min = INSIDE * root_radius(c)
    cand = []
    for M in Ms:
        poles, zeros = pade_poles(c, M)
        for sm in poles:
            s = complex(sm)
            if s.imag < -1e-12 * abs(s) or abs(s) > r_max or abs(s) < r_min:
                continue
            if any(abs(sm - z) <= DOUBLET * abs(sm) for z in zeros):
                continue
            cand.append((M, complex(s.real, abs(s.imag))))
    clusters = []
    for M, s in sorted(cand, key=lambda t: abs(t[1])):
        for cl in clusters:
            if abs(s - cl["seed"]) <= STABLE_TOL * abs(cl["seed"]) and M not in cl["Ms"]:
                cl["Ms"].append(M)
                cl["pts"].append(s)
                break
        else:
            clusters.append(dict(seed=s, Ms=[M], pts=[s]))
    out = []
    for cl in clusters:
        if len(cl["Ms"]) < STABLE_MIN:
            continue
        pts = np.array(cl["pts"])
        med = complex(np.median(pts.real), np.median(pts.imag))
        out.append(dict(s=med, abs=abs(med), arg_deg=float(np.degrees(np.angle(med))), n=len(cl["Ms"]),
                        spread=float(np.max(np.abs(pts - med)) / abs(med)), Ms=sorted(cl["Ms"])))
    return sorted(out, key=lambda d: d["abs"])


def recurrence_fit(c, k_from=5, w=WIN):
    """c_k = p c_(k-1) + q c_(k-2) by least squares over windows of w orders (k counted from 1), the root 1/s of
    x^2 - p x - q of largest modulus per window: [(k0, s)], and the median s over the windows starting at k_from"""
    v = np.array([float(x) for x in c])
    out = []
    n = len(v)
    for k0 in range(3, n - w + 2):                          # window k0..k0+w-1 needs c_(k0-2)
        ks = range(k0, k0 + w)
        A = np.array([[v[k - 2], v[k - 3]] for k in ks])  # c_(k-1), c_(k-2) with c_k at v[k-1]
        b = np.array([v[k - 1] for k in ks])
        sc = np.abs(A).max()
        if sc == 0:
            continue
        (p, q), *_ = np.linalg.lstsq(A / sc, b / sc, rcond=None)
        xs = np.roots([1.0, -p, -q])
        x = xs[np.argmax(np.abs(xs))]
        if x == 0:
            continue
        s = 1 / x
        out.append((k0, complex(s.real, abs(s.imag))))
    sel = [s for k0, s in out if k0 >= k_from]
    med = complex(np.median([s.real for s in sel]), np.median([s.imag for s in sel])) if sel else None
    spread = float(np.max([abs(s - med) for s in sel]) / abs(med)) if sel else None
    return out, med, spread


def domb_sykes(c):
    """the ratios c_(k-1)/c_k (k >= 2) and their 1/k extrapolation from the last half of the orders: (ratios, s_DS)"""
    v = [float(x) for x in c]
    r = [(k, v[k - 2] / v[k - 1]) for k in range(2, len(v) + 1) if v[k - 1] != 0]
    tail = r[len(r) // 2:]
    if len(tail) < 3:
        return r, None
    x = np.array([1.0 / k for k, _ in tail])
    y = np.array([1.0 / rr for _, rr in tail])             # c_k / c_(k-1) ~ (1/s)(1 + b/k)
    sl, ic = np.polyfit(x, y, 1)
    return r, (1.0 / ic if ic != 0 else None)


# ------------------------------------------------------------------ planted controls
def planted_series(kind, seed=1, noise=1e-6, N=24, drop_pole=False, envelope=None):
    """c_1..c_N (mpf) of a planted series and its planted singularity (s or None)"""
    rng = np.random.default_rng(seed)
    with mp.workdps(DPS):
        c = []
        S = mp.mpf(243)
        st = mp.mpf(7) * mp.expj(mp.mpf("2.2"))             # the background's radius-7 pair
        B = mp.mpf("1.9e-8")
        for k in range(1, N + 1):
            v = -mp.mpf("1.7e-4") * S * (-1) ** (k + 1) / (k * S ** k)
            v += B * 2 * mp.re(st ** (-k))
            c.append(v)
        plant = None
        if kind in ("P1", "P2"):
            R, th, C = (mp.mpf(1), 60, mp.mpf("1e-14")) if kind == "P1" else (mp.mpf("2.5"), 25, mp.mpf("1e-11"))
            s = R * mp.expj(mp.radians(th))
            plant = complex(s)
            if not drop_pole:
                for k in range(1, N + 1):
                    c[k - 1] += C * 2 * mp.re(s ** (-k - 1))     # 1/(s - a) + conjugate: c_k = 2 Re s^-(k+1)
        elif kind == "P3":
            R, th, C = mp.mpf("1.5"), 45, mp.mpf("1e-12")
            s = R * mp.expj(mp.radians(th))
            plant = complex(s)
            for k in range(1, N + 1):
                c[k - 1] += C * 2 * mp.re(mp.binomial(mp.mpf(1) / 2, k) * (-1 / s) ** k)
        elif kind == "N3":
            # an entire component, 2 Re C exp((lam + i w) a) with w 2.3 (a period of 2.7 in D): no singularity at all
            C, lam, w = mp.mpf("1e-11"), mp.mpf("0.1"), mp.mpf("2.3")
            for k in range(1, N + 1):
                c[k - 1] += C * 2 * mp.re(mp.mpc(lam, w) ** k / mp.factorial(k))
        elif kind == "N2":
            for k in range(5, N + 1):
                c[k - 1] = mp.mpf(float(envelope[k])) * (1 if rng.random() < 0.5 else -1) * (1 + 0.3 * rng.standard_normal())
        if noise:
            c = [v * (1 + noise * float(rng.standard_normal())) for v in c]
        return c, plant


# the records' |z_k| at D 245.1366 (#603's anm_run.jsonl, the step from that base), the envelope of N2
ENV_245 = [None, 3.6e-4, 1.1e-6, 4.0e-9, 1.5e-11, 1.88e-12, 8.79e-13, 1.69e-13, 6.73e-15, 4.0e-15, 1.2e-15, 2.1e-16,
           2.6e-17, 3.1e-19, 5.5e-19, 7.9e-20, 1.3e-21, 4.5e-22, 9.0e-24, 7.0e-24, 5.7e-26, 6.9e-26, 7.9e-28, 7.7e-28,
           2.3e-28]


def judge_pole(c, plant, tol_r, tol_deg, tol_r_rec, tol_deg_rec):
    """(ok, line): the nearest stable Pade pole and the recurrence fit against the planted s"""
    sp = stable_poles(c)
    _, rec, rspread = recurrence_fit(c)
    pr, pth = abs(plant), np.degrees(np.angle(plant))
    near = sp[0] if sp else None
    okp = near is not None and abs(near["abs"] / pr - 1) <= tol_r and abs(near["arg_deg"] - pth) <= tol_deg
    okr = rec is not None and abs(abs(rec) / pr - 1) <= tol_r_rec and abs(np.degrees(np.angle(rec)) - pth) <= tol_deg_rec
    line = law_line(c) + "; " + (f"planted |s| {pr:.3f} arg {pth:.1f}; Pade " + (f"|s| {near['abs']:.4f} arg {near['arg_deg']:.2f} (n {near['n']}, "
            f"spread {near['spread']:.1e})" if near else "no stable pole") + f"; recurrence " +
            (f"|s| {abs(rec):.4f} arg {np.degrees(np.angle(rec)):.2f} (spread {rspread:.2f})" if rec is not None else "none")
            + f"; root-test radius {root_radius(c):.3f}")
    return okp and okr, line


def growth_law(c, k_lo=5):
    """how the coefficients fall from order k_lo on: the upper envelope's least-squares fit as a singularity at
    distance rho (ln|c_k| = A - k ln rho) and as an entire function (ln|c_k| = A + k ln w - ln k!), each with its rms in
    ln, and the slope d|s|/dk of the recurrence fit's |s| over its windows (about 0 for a singularity, about 1/w for an
    entire function, whose ratios grow like k/w)"""
    from math import lgamma
    v = np.abs(np.array([float(x) for x in c]))
    N = len(v)
    k = np.arange(1, N + 1)
    env = np.array([v[max(0, i - 1):i + 2].max() for i in range(N)])
    sel = (k >= k_lo) & (env > 0)
    kk, y = k[sel], np.log(env[sel])
    A = np.vstack([np.ones(len(kk)), kk]).T
    cg = np.linalg.lstsq(A, y, rcond=None)[0]
    y2 = y + np.array([lgamma(x + 1.0) for x in kk])
    ce = np.linalg.lstsq(A, y2, rcond=None)[0]
    wins, _, _ = recurrence_fit(c, k_from=k_lo)
    ws = [(k0, abs(sv)) for k0, sv in wins if k0 >= k_lo]
    slope = float(np.polyfit([w[0] for w in ws], [w[1] for w in ws], 1)[0]) if len(ws) >= 3 else None
    return dict(rho=float(np.exp(-cg[1])), rms_geo=float(np.sqrt(np.mean((A @ cg - y) ** 2))),
                w=float(np.exp(ce[1])), rms_ent=float(np.sqrt(np.mean((A @ ce - y2) ** 2))), slope=slope)


def law_line(c):
    g = growth_law(c)
    return (f"fall from order 5: singularity rho {g['rho']:.2f} (rms {g['rms_geo']:.2f}), entire w {g['w']:.2f} "
            f"(rms {g['rms_ent']:.2f}), recurrence d|s|/dk " + (f"{g['slope']:+.2f}" if g["slope"] is not None else "n/a"))


def planted(argv):
    mutate = argv[argv.index("--mutate") + 1] if "--mutate" in argv else ""
    mp.mp.dps = DPS
    ok_all = True
    print("planted controls of the estimator (criteria fixed before any real series was read)", flush=True)
    for kind, tol in (("P1", (0.03, 3, 0.05, 5)), ("P2", (0.03, 3, 0.05, 5)), ("P3", (0.10, 10, 0.10, 10))):
        c, plant = planted_series(kind, drop_pole=(mutate == "drop_pole" and kind == "P1"))
        ok, line = judge_pole(c, plant, *tol)
        ok_all &= ok
        print(f"  {kind}: {'PASS' if ok else 'FAIL'}: {line}", flush=True)
        for nz in (1e-12, 1e-3):
            c2, _ = planted_series(kind, noise=nz, drop_pole=(mutate == "drop_pole" and kind == "P1"))
            ok2, line2 = judge_pole(c2, plant, *tol)
            print(f"      noise {nz:.0e} (reported, not judged): {'recovered' if ok2 else 'not recovered'}: {line2}", flush=True)
    for kind in ("N1", "N2", "N3"):
        for seed in range(1, 6):
            c, _ = planted_series(kind, seed=seed, envelope=ENV_245)
            sp = [d for d in stable_poles(c) if d["abs"] <= R_RANGE]
            _, rec, rspread = recurrence_fit(c)
            ok = not sp
            ok_all &= ok
            far = stable_poles(c, r_max=1e3)
            print(f"  {kind} seed {seed}: {'PASS' if ok else 'FAIL'}: stable Pade poles within {R_RANGE}: "
                  + (", ".join(f"|s| {d['abs']:.3f} arg {d['arg_deg']:.1f} (n {d['n']})" for d in sp) or "none")
                  + "; nearest beyond: " + (f"|s| {far[0]['abs']:.3f} arg {far[0]['arg_deg']:.1f} (n {far[0]['n']})" if far else "none")
                  + (f"; recurrence |s| {abs(rec):.3f} arg {np.degrees(np.angle(rec)):.1f} (spread {rspread:.2f})" if rec is not None else "")
                  + "; " + law_line(c), flush=True)
    print(f"{'PASS' if ok_all else 'FAIL'}: planted controls" + (f" (--mutate {mutate})" if mutate else ""), flush=True)
    return 0 if ok_all else 1


# ------------------------------------------------------------------ the series at a stored state
def parts_mp(parts, i):
    """entry i of a vector in parts as an exact mpf sum"""
    with mp.workprec(64 * 53):
        return mp.fsum(mp.mpf(float(np.asarray(p)[i])) for p in parts)


def load_jet(path):
    """(list of z_k as tuples of parts, meta)"""
    d = np.load(path)
    m = json.loads(str(d["meta"]))
    P = m["parts"]
    Z = [d[f"z{q}"] for q in range(P)]
    return [tuple(np.ascontiguousarray(Z[q][k]) for q in range(P)) for k in range(Z[0].shape[0])], m


def save_jet(path, jet, meta):
    np.savez(path, **{f"z{q}": np.array([np.asarray(z[q]) for z in jet]) for q in range(len(jet[0]))},
             meta=json.dumps(dict(meta, parts=len(jet[0]))))


def jet(argv):
    """the series at STATE in each variant of --variants (comma-separated: od, qd, q4, pert:LEVEL:SEED,
    inject:LEVEL:SEED), saved as OUT_<variant>.npz; od and inject share the octo-double LU at the base, and pert's
    J^-1 r are all formed through it before their own LUs replace it"""
    import time
    from hpkernel.driver import make_solver
    from multidouble import od_axpy
    from pinning import HeldFit, band_field
    from octo_corrector import load_o, o8, o8_of_q, q4_of_o
    import anm_run
    raw, state, out = argv[0], argv[1], argv[2]
    opt = dict(variants="od", N=24, threads=4, held_fit="data/held_fit.json", kfreeze=3)
    for k in list(opt):
        if "--" + k in argv:
            opt[k] = type(opt[k])(argv[argv.index("--" + k) + 1])
    anm_run.require_raw(out + "_od.npz")
    zo, m = load_o(os.path.join(raw, state))
    D = mp.mpf(m["D_str"])
    M, nodes, kf, N = int(m["M"]), list(m["nodes"]), opt["kfreeze"], opt["N"]
    s = make_solver(D, M, nodes=nodes, kt=m.get("kt", 36), kc=m.get("kc", 146), ks=m.get("ks", 130),
                    threads=opt["threads"], kfreeze=kf, horizon_alpha=0)
    s.set_segs(m["segs"])
    ks_held = [k for k in range(kf + 1, 14) if k % 2 == 1]
    hj = HeldFit(opt["held_fit"]).jets(D, N, ks_held, exact=True)
    Kb = (len(zo[0]) - 1) // (4 * M) - 1
    rows = slice(Kb * 4 * M, Kb * 4 * M + M)
    held = (lambda j: band_field(hj[j], ks_held, M, parts=8), rows)
    L = len(zo[0])
    zero = np.zeros(L)
    specs = []
    for v in opt["variants"].split(","):
        f = v.split(":")
        specs.append(dict(name=f[0], level=float(f[1]) if len(f) > 1 else 0.0, seed=int(f[2]) if len(f) > 2 else 1,
                          tag=v.replace(":", "_")))

    def finish(sp, Z, norms, piv, t, info):
        meta = dict(D=float(D), D_str=str(D), M=M, N=N, kfreeze=kf, jet=("q" if sp["name"] == "qd" else "o"),
                    variant=sp["name"], level=sp["level"], seed=sp["seed"], norms=norms, pivots=[float(x) for x in piv],
                    state=state, held_fit=opt["held_fit"], held_o=1, t_coef=t, info=info)
        path = f"{out}_{sp['tag']}.npz"
        save_jet(path, Z, meta)
        print(f"  {sp['tag']}: series at {state} (D {mp.nstr(D, 12)}, N {N}) saved as {path}, {t:.0f} s; |z_k| "
              + " ".join(f"{x:.3e}" for x in norms[1:9]), flush=True)

    t0 = time.time()
    piv0 = s.newton_o(list(zo))[3:5]
    print(f"  octo-double LU at the base: pivots {piv0[0]:.2e}..{piv0[1]:.2e}, {time.time() - t0:.0f} s", flush=True)
    bases = {}
    if any(sp["name"] == "pert" for sp in specs):
        F0 = o8(s.residual_o(list(zo)))
    for sp in specs:
        if sp["name"] != "pert":
            continue
        # the root of F = r: z0 + J^-1 r, its residual change checked against r
        rng = np.random.default_rng(sp["seed"])
        r = rng.standard_normal(L)
        r *= sp["level"] / np.abs(r).max()
        dz = o8(s.olu_solve([r] + [zero] * 7))
        z1 = o8(od_axpy(1.0, *dz, *zo))
        dF = o8(od_axpy(-1.0, *F0, *o8(s.residual_o(list(z1)))))
        rem = o8(od_axpy(-1.0, *o8([r] + [zero] * 7), *dF))
        info = dict(dz_max=float(np.abs(dz[0]).max()), dz_delta=float(dz[0][-1]), dF_max=float(np.abs(dF[0]).max()),
                    remainder=float(np.abs(rem[0]).max()), F0_max=float(np.abs(F0[0]).max()))
        print(f"  {sp['tag']}: max|r| {sp['level']:.1e}, max|J^-1 r| {info['dz_max']:.2e} (Delta {info['dz_delta']:+.2e}); "
              f"max|F(z0 + dz) - F(z0)| {info['dF_max']:.2e}, its departure from r {info['remainder']:.2e}", flush=True)
        bases[sp["tag"]] = (z1, info)
    for sp in [x for x in specs if x["name"] in ("od", "inject")]:
        t1 = time.time()
        perturb = None
        if sp["name"] == "inject":
            rng = np.random.default_rng(sp["seed"])

            def perturb(k, b, rng=rng, lev=sp["level"]):
                e = rng.standard_normal(L)
                e *= lev * float(np.abs(np.asarray(b[0])).max()) / np.abs(e).max()
                return o8(od_axpy(1.0, *o8([e] + [zero] * 7), *o8(b)))
        Z, norms = anm_run.coefficients_o(s, zo, N, held=held, perturb=perturb)
        finish(sp, Z, norms, piv0, time.time() - t1, {})
    for sp in [x for x in specs if x["name"] in ("pert", "q4")]:
        t1 = time.time()
        zb, info = bases[sp["tag"]] if sp["name"] == "pert" else (o8_of_q(q4_of_o(zo)), {})
        piv = s.newton_o(list(zb))[3:5]
        Z, norms = anm_run.coefficients_o(s, zb, N, held=held)
        finish(sp, Z, norms, piv, time.time() - t1, info)
    for sp in [x for x in specs if x["name"] == "qd"]:
        t1 = time.time()
        zq = q4_of_o(zo)
        piv = s.newton_q(*zq)[6:8]
        Zq, norms = anm_run.coefficients_q(s, zq, N, held=held)
        finish(sp, [o8_of_q(z) for z in Zq], norms, piv, time.time() - t1, {})
    print(f"done, {time.time() - t0:.0f} s", flush=True)
    return 0


# ------------------------------------------------------------------ comparisons
def od_diff_max(a, b):
    """max |b - a| and (b - a)[-1] for vectors in parts (promoted to eight), in octo-double"""
    from multidouble import od_axpy
    from octo_corrector import o8
    zero = np.zeros(len(a[0]))
    a8 = o8(list(a) + [zero] * (8 - len(a)))
    b8 = o8(list(b) + [zero] * (8 - len(b)))
    d = o8(od_axpy(-1.0, *a8, *b8))
    v = np.asarray(d[0]) + np.asarray(d[1])
    return float(np.abs(v).max()), float(v[-1])


def compare(argv):
    paths = [p for p in argv if p.endswith(".npz")]
    ks = argv[argv.index("--ks") + 1] if "--ks" in argv else "1-12"
    k_lo, k_hi = (int(x) for x in ks.split("-"))
    mutate = argv[argv.index("--mutate") + 1] if "--mutate" in argv else ""
    ref, mr = load_jet(paths[0])
    rows = []
    for p in paths[1:]:
        var, mv = load_jet(p)
        if mutate == "scale":
            var = list(var)
            var[5] = tuple(np.asarray(x) * (1 + 1e-6) for x in var[5])
        tag = f"{mv.get('variant', 'anm_run ' + mv.get('jet', ''))}" + (f" {mv['level']:.0e} s{mv['seed']}" if mv.get("level") else "")
        print(f"{os.path.basename(p)} ({tag}) against {os.path.basename(paths[0])} ({mr.get('variant', mr.get('jet'))}), D {mv['D']:.4f}", flush=True)
        print("   k   max|z_k| ref   Delta_k ref        max|dz_k|/max|z_k|  |dDelta_k/Delta_k|", flush=True)
        for k in range(k_lo, min(k_hi, len(ref) - 1, len(var) - 1) + 1):
            dmax, dd = od_diff_max(ref[k], var[k])
            zmax = float(np.abs(np.asarray(ref[k][0])).max())
            dk = float(parts_mp(ref[k], -1))
            rows.append(dict(path=p, k=k, zmax=zmax, delta=dk, rel=dmax / zmax, rel_delta=abs(dd / dk) if dk else None))
            print(f"  {k:2d}   {zmax:.3e}   {dk:+.6e}   {dmax / zmax:.2e}            {abs(dd / dk) if dk else float('nan'):.2e}", flush=True)
    return rows


# ------------------------------------------------------------------ the singularity
def row_series(jetZ, row):
    """c_1..c_N of one entry of the series (mpf); row 'delta' (the last entry), 'max5' (the entry of the largest
    |z_5|) or an index"""
    if row == "delta":
        i = -1
    elif row == "max5":
        i = int(np.argmax(np.abs(np.asarray(jetZ[5][0]))))
    else:
        i = int(row)
    return [parts_mp(z, i) for z in jetZ[1:]], i


def locate_one(c, D0, label=""):
    sp = stable_poles(c, r_max=50)
    recs, rec, rspread = recurrence_fit(c)
    _, ds = domb_sykes(c)
    rho = root_radius(c)
    res = dict(D0=D0, label=label, root_radius=rho, pade=[dict(abs=d["abs"], arg_deg=d["arg_deg"], n=d["n"], spread=d["spread"],
                                              Ms=d["Ms"], D_re=D0 + d["s"].real, D_im=d["s"].imag) for d in sp[:4]],
               recurrence=(dict(abs=abs(rec), arg_deg=float(np.degrees(np.angle(rec))), spread=rspread,
                                D_re=D0 + rec.real, D_im=rec.imag) if rec is not None else None),
               recurrence_windows=[(k0, abs(s), float(np.degrees(np.angle(s)))) for k0, s in recs],
               domb_sykes=ds)
    print(f"  D0 {D0:.4f} {label}: Pade " + (", ".join(f"|s| {d['abs']:.3f} arg {d['arg_deg']:.1f} (n {d['n']}, spread {d['spread']:.1e})"
                                                    for d in sp[:3]) or "no stable pole")
          + (f"; recurrence |s| {abs(rec):.3f} arg {np.degrees(np.angle(rec)):.1f} (spread {rspread:.2f})" if rec is not None else "")
          + (f"; Domb-Sykes s {ds:.3f}" if ds else "") + f"; root-test radius {rho:.3f}; " + law_line(c), flush=True)
    res["growth_law"] = growth_law(c)
    return res


def locate(argv):
    mp.mp.dps = DPS
    row = argv[argv.index("--row") + 1] if "--row" in argv else "delta"
    out = argv[argv.index("--json") + 1] if "--json" in argv else None
    res = []
    for p in [p for p in argv if p.endswith(".npz")]:
        Z, m = load_jet(p)
        c, i = row_series(Z, row)
        print(f"{os.path.basename(p)}: row {i}, c_1..c_{len(c)}: " + " ".join(mp.nstr(x, 3) for x in c), flush=True)
        res.append(dict(locate_one(c, m["D"], label=f"row {i}"), path=p, row=i, coeffs=[mp.nstr(x, 20) for x in c]))
    if out:
        with open(out, "w") as f:
            json.dump(res, f, indent=1, default=str)
    return 0


# ------------------------------------------------------------------ the series against converged states
def truth(argv):
    from anm_run import horner_o
    from octo_corrector import load_o
    raw, jp = argv[0], argv[1]
    states = [p for p in argv[2:] if p.endswith(".npz")]
    Z, m = load_jet(jp)
    D0 = mp.mpf(m["D_str"])
    N = len(Z) - 1
    zmax = [float(np.abs(np.asarray(z[0])).max()) for z in Z]
    out = []
    for sp in states:
        z, ms = load_o(os.path.join(raw, sp))
        a = float(mp.mpf(ms["D_str"]) - D0)
        if not 0 < a <= 2:
            continue
        pred = horner_o(Z, a)
        emax, edelta = od_diff_max(pred, z)
        mid = max(zmax[k] * a ** k for k in (5, 6, 7, 8))
        trunc = zmax[N] * a ** N
        dmid = max(abs(float(parts_mp(Z[k], -1))) * a ** k for k in (5, 6, 7, 8))
        rec = dict(D0=float(D0), state=sp, a=a, err_max=emax, err_delta=edelta, mid_terms=mid, mid_terms_delta=dmid, trunc=trunc)
        out.append(rec)
        print(f"  D0 {float(D0):.4f} -> {ms['D_str'][:10]} (a {a:.4f}): max|z - series| {emax:.2e}, Delta {edelta:+.2e}; "
              f"max_k=5..8 |z_k| a^k {mid:.2e} (Delta's {dmid:.2e}); |z_N| a^N {trunc:.2e}", flush=True)
    return out


def preds(argv):
    """the chain's stored predictions (pred_pin<k>_D<D>.npz, before their re-seed, four parts) against the converged
    states at the same D, per component, with the records' |z_k| a^k for k 5..8 and |z_N| a^N of the step's base.
    --mutate scale5 adds the base's |z_5| a^5 to every prediction's Delta, which the check must see (exit 1). Exit 1
    unless every prediction's error lies below its middle orders' terms by 1e6 or more (the middle orders are
    confirmed by the corrector to that factor)"""
    import glob
    from octo_corrector import load_o
    raw, folder = argv[0], argv[1]
    mutate = argv[argv.index("--mutate") + 1] if "--mutate" in argv else ""
    d_max = float(argv[argv.index("--D_max") + 1]) if "--D_max" in argv else float("inf")
    recs = [r for r in (json.loads(x) for x in open(os.path.join(raw, folder, "anm_run.jsonl"))) if r["D0"] < d_max]
    by_target = {}
    for r in recs:
        for a in r["attempts"]:
            if a["accepted"]:
                by_target[a["D_str"]] = (r, a)
    ok, out = True, []
    print(" target D     a       max|z - pred|  Delta's error   max_k=5..8 |z_k| a^k  |z_N| a^N   factor", flush=True)
    for pp in sorted(glob.glob(os.path.join(raw, folder, "pred_pin*_D*.npz"))):
        d = np.load(pp)
        pm = json.loads(str(d["meta"]))
        sp = pp.replace("pred_pin", "anm_pin")
        if pm["D_str"] not in by_target or not os.path.exists(sp):
            continue
        r, att = by_target[pm["D_str"]]
        if abs(att["a"] - pm["a"]) > 1e-12:
            continue                                           # the file holds a later attempt's prediction
        z, _ = load_o(sp)
        pr = [d["zh"], d["zl"], d["z2"], d["z3"]]
        a = pm["a"]
        if mutate == "scale5":
            pr = [x.copy() for x in pr]
            pr[0][-1] += r["norms"][5] * a ** 5
        e = (z[0] - pr[0]) + (z[1] - pr[1]) + (z[2] - pr[2]) + (z[3] - pr[3]) + (z[4] + z[5])
        mid = max(r["norms"][k] * a ** k for k in (5, 6, 7, 8))
        N = r["N"]
        trunc = r["norms"][N] * a ** N
        emax = float(np.abs(e).max())
        fac = mid / emax if emax else float("inf")
        ok &= fac >= 1e6
        out.append(dict(D=pm["D_str"], a=a, err_max=emax, err_delta=float(e[-1]), mid=mid, trunc=trunc, factor=fac))
        print(f" {float(pm['D_str']):10.4f}  {a:.4f}  {emax:.2e}       {float(e[-1]):+.2e}       {mid:.2e}            {trunc:.2e}   {fac:.1e}",
              flush=True)
    print(f"{'PASS' if ok and out else 'FAIL'}: {len(out)} predictions, each error below its middle orders' terms by 1e6 or more"
          + (f" (--mutate {mutate})" if mutate else ""), flush=True)
    return 0 if ok and out else 1


def reexpand(argv):
    """the series at base B1 re-expanded about B2 = B1 + h, z_k(B2) = sum_(j >= k) C(j, k) z_j(B1) h^(j - k) in
    octo-double (h a dyadic float, so each coefficient is exact), against B2's own series: per order, the relative
    difference of the vector (max-norm) and of Delta's entry, and the truncation's size C(N + 1, k) |z_N(B1)| h^(N+1-k).
    The two series share the kernel and the held fit but neither base, LU nor jet, so a coefficient that both give is a
    property of z(D) (noise in either would not re-expand into the other). --mutate scale5 multiplies B1's z_5 by
    1 + 1e-6, which must show at order 5. Exit 1 unless orders 1..8 agree within 1e-9 relative"""
    from math import comb
    from multidouble import od_axpy
    from octo_corrector import o8
    p1, p2 = [x for x in argv if x.endswith(".npz")][:2]
    mutate = argv[argv.index("--mutate") + 1] if "--mutate" in argv else ""
    Z1, m1 = load_jet(p1)
    Z2, m2 = load_jet(p2)
    h = float(mp.mpf(m2["D_str"]) - mp.mpf(m1["D_str"]))
    if float(mp.mpf(h)) != float(mp.mpf(m2["D_str"]) - mp.mpf(m1["D_str"])) or h != h:
        raise SystemExit("the bases' distance is not a float")
    N = len(Z1) - 1
    zero = np.zeros(len(Z1[0][0]))
    Z1 = [o8(list(z) + [zero] * (8 - len(z))) for z in Z1]
    if mutate == "scale5":
        Z1[5] = o8(od_axpy(1e-6, *Z1[5], *Z1[5]))
    n1 = [float(np.abs(z[0]).max()) for z in Z1]
    print(f"{os.path.basename(p1)} (D {m1['D']:.4f}) re-expanded by h {h} against {os.path.basename(p2)} (D {m2['D']:.4f})", flush=True)
    print("   k   max|z_k(B2)|   rel. difference   Delta_k(B2)        Delta's rel. difference   truncation/|z_k|", flush=True)
    ok, rows = True, []
    for k in range(1, min(len(Z2) - 1, N) + 1):
        acc = o8([zero] * 8)
        for j in range(N, k - 1, -1):
            acc = o8(od_axpy(float(comb(j, k)) * h ** (j - k), *Z1[j], *acc))
        dmax, dd = od_diff_max(acc, Z2[k])
        zmax = float(np.abs(np.asarray(Z2[k][0])).max())
        dk = float(parts_mp(Z2[k], -1))
        trunc = comb(N + 1, k) * n1[N] * abs(h) ** (N + 1 - k)
        rows.append(dict(k=k, zmax=zmax, rel=dmax / zmax, delta=dk, rel_delta=abs(dd / dk), trunc_rel=trunc / zmax))
        if k <= 8:
            ok &= dmax / zmax <= 1e-9
        print(f"  {k:2d}   {zmax:.3e}      {dmax / zmax:.2e}          {dk:+.6e}      {abs(dd / dk):.2e}                  {trunc / zmax:.1e}", flush=True)
    print(f"{'PASS' if ok else 'FAIL'}: orders 1..8 re-expand within 1e-9" + (f" (--mutate {mutate})" if mutate else ""), flush=True)
    return (0 if ok else 1), rows


def attempts_of_log(path):
    """per corrector attempt of an anm_run.py --corrector od log: the base D, a, the target D, the start max|F|, the
    first Newton move's max|dz| and Delta part, max|F| after it, the steps, whether it was accepted and the max|F| it
    converged to (F_conv, None when it did not)"""
    import re
    out, base, first, conv = [], None, None, None
    for ln in open(path):
        m = re.match(r"step \d+: D ([0-9.]+),", ln)
        if m:
            base = float(m.group(1))
            continue
        m = re.match(r"\s+on \d+: \|F\| ([0-9.e+-]+) converged", ln)
        if m and base is not None:
            conv = float(m.group(1))
            continue
        m = re.match(r"\s+on 0: \|F\| ([0-9.e+-]+) -> ([0-9.e+-]+), \|dz\| ([0-9.e+-]+) -> [0-9.e+-]+, lam [0-9.]+, dDelta ([0-9.e+-]+)", ln)
        if m and base is not None:
            first = dict(F0=float(m.group(1)), F1=float(m.group(2)), dz=float(m.group(3)), dDelta=float(m.group(4)))
            continue
        m = re.match(r"\s+a ([0-9.]+) -> D ([0-9.]+): predicted .* od (\d+) steps -> .*?\((tol [0-9.e+-]+)(, stall accepted)?", ln)
        if m and base is not None and first is not None:
            out.append(dict(D0=base, a=float(m.group(1)), D=float(m.group(2)), steps=int(m.group(3)),
                            accepted=bool(m.group(5)), F_conv=conv if m.group(5) else None, **first))
            first, conv = None, None
    return out


def refusals(argv):
    """the refused attempts of a chain against the accepted ones: the prediction's error as the corrector's first move
    measures it (max|dz|, the step from the prediction toward the root, which matches the prediction's distance from the
    converged state at every accepted step, preds), against the records' truncation |z_N| a^N and middle-order terms,
    and the start residual. RAW FOLDER: the chain's folder in the raw-data repository (its anm_run.jsonl and the log
    LOG, default FOLDER/../chain.log)"""
    raw, folder = argv[0], argv[1]
    logp = argv[argv.index("--log") + 1] if "--log" in argv else os.path.join(raw, folder, os.pardir, "chain.log")
    d_max = float(argv[argv.index("--D_max") + 1]) if "--D_max" in argv else float("inf")
    recs = {round(r["D0"], 4): r for r in (json.loads(x) for x in open(os.path.join(raw, folder, "anm_run.jsonl")))
            if r["D0"] < d_max}
    att = attempts_of_log(logp)
    print(" base D     a       target D   start max|F|  first move  its Delta part  max|F| after  steps  accepted"
          "   |z_N| a^N   max_k=5..8 |z_k| a^k", flush=True)
    rows = []
    for x in att:
        r = recs.get(round(x["D0"], 4))
        if r is None:
            continue
        N = r["N"]
        x["trunc"] = r["norms"][N] * x["a"] ** N
        x["mid"] = max(r["norms"][k] * x["a"] ** k for k in (5, 6, 7, 8))
        rows.append(x)
        print(f" {x['D0']:9.4f}  {x['a']:.4f}  {x['D']:9.4f}  {x['F0']:.2e}     {x['dz']:.2e}    {x['dDelta']:+.2e}      "
              f"{x['F1']:.2e}     {x['steps']:2d}     {'yes' if x['accepted'] else 'NO '}      {x['trunc']:.2e}   {x['mid']:.2e}", flush=True)
    acc1 = [x for x in rows if x["accepted"] and x["a"] > 0.9]
    ref = [x for x in rows if not x["accepted"]]
    if acc1 and ref:
        print(f"accepted at a > 0.9: first move {min(x['dz'] for x in acc1):.1e} to {max(x['dz'] for x in acc1):.1e}, start "
              f"{min(x['F0'] for x in acc1):.1e} to {max(x['F0'] for x in acc1):.1e}; refused: first move "
              f"{min(x['dz'] for x in ref):.1e} to {max(x['dz'] for x in ref):.1e}, start {min(x['F0'] for x in ref):.1e} to "
              f"{max(x['F0'] for x in ref):.1e}", flush=True)
    return rows


def summary(argv):
    """hp/data/jet_634.json from the raw-data repository's I634-mac and I603-win2 folders: the predictions' check, every
    variant against its base's od series (k 1..8), Delta's series located at every 24-order base, and the refused
    steps' series against the converged states their targets later reached. RAW is the raw-data checkout"""
    import glob
    import io
    from contextlib import redirect_stdout
    mp.mp.dps = DPS
    raw = argv[0]
    # #603's records are used up to the bases below D_max (245.5: the 27 steps from D 225.13 to 245.64 that #634 set),
    # and its later records to the bases below D_beyond (253.6: the steps to D 253.53 of raw commit d2948f5b)
    d_max = float(argv[argv.index("--D_max") + 1]) if "--D_max" in argv else 245.5
    d_beyond = float(argv[argv.index("--D_beyond") + 1]) if "--D_beyond" in argv else 253.6
    out = argv[argv.index("--out") + 1] if "--out" in argv else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "jet_634.json")
    R = os.path.join(raw, "I634-mac")
    J = dict(source="raw-data repository, folders I634-mac and I603-win2 (#634)")
    buf = io.StringIO()
    with redirect_stdout(buf):
        preds([raw, "I603-win2/chain", "--D_max", str(d_max)])
    rows = [ln.split() for ln in buf.getvalue().splitlines() if ln.strip() and ln.split()[0][0].isdigit()]
    pr = [dict(D=float(r[0]), a=float(r[1]), err=float(r[2]), mid=float(r[4]), factor=float(r[6])) for r in rows]
    one = [x for x in pr if x["a"] > 0.9]
    small = [x for x in pr if x["a"] < 0.65]
    J["preds"] = dict(n=len(pr), factor_min=min(x["factor"] for x in pr), err_a1_min=min(x["err"] for x in one),
                      err_a1_max=max(x["err"] for x in one), mid_a1_min=min(x["mid"] for x in one),
                      mid_a1_max=max(x["mid"] for x in one), err_small_min=min(x["err"] for x in small),
                      err_small_max=max(x["err"] for x in small), rows=pr)
    var = {}
    for ref in sorted(glob.glob(os.path.join(R, "jets", "D*_od.npz"))):
        Dk = os.path.basename(ref).split("_")[0][1:]
        Zr, _ = load_jet(ref)
        var[Dk] = {}
        for vp in sorted(glob.glob(os.path.join(R, "jets", f"D{Dk}_*.npz"))) + sorted(glob.glob(os.path.join(R, "polish", f"jet_pin3_D{Dk}.npz"))) \
                + sorted(glob.glob(os.path.join(R, "repro", f"jet_pin3_D{Dk}.npz"))):
            if vp == ref:
                continue
            Zv, mv = load_jet(vp)
            tag = (os.path.basename(vp)[len(f"D{Dk}_"):-4] if "jets" in vp else ("polish" if "polish" in vp else "repro_N24"))
            ks = range(1, min(len(Zr), len(Zv)))
            rel, reld = {}, {}
            for k in ks:
                dmax, dd = od_diff_max(Zr[k], Zv[k])
                zmax = float(np.abs(np.asarray(Zr[k][0])).max())
                dk = float(parts_mp(Zr[k], -1))
                rel[k], reld[k] = dmax / zmax, (abs(dd / dk) if dk else None)
            var[Dk][tag] = dict(rel=rel, rel_delta=reld, rel_max_5_8=max(rel[k] for k in (5, 6, 7, 8)),
                                rel_delta_max_5_8=max(reld[k] for k in (5, 6, 7, 8)), info=mv.get("info", {}))
        var[Dk]["norms"] = [float(np.abs(np.asarray(z[0])).max()) for z in Zr]
        var[Dk]["delta"] = [float(parts_mp(z, -1)) for z in Zr]
    J["variants"] = var
    # the polishes: anm_run.py --tol_od 1e-80's Newton steps on the base, from its log
    import re
    J["polish"] = {}
    for lp in sorted(glob.glob(os.path.join(R, "polish_D*.log"))):
        steps = [(float(a), float(b), float(c)) for a, b, c in
                 re.findall(r"on \d+: \|F\| ([0-9.e+-]+) -> ([0-9.e+-]+), \|dz\| ([0-9.e+-]+) ->", open(lp).read())]
        conv = re.findall(r"on \d+: \|F\| ([0-9.e+-]+) converged", open(lp).read())
        if steps:
            J["polish"][os.path.basename(lp)[len("polish_D"):-4]] = dict(F_start=steps[0][0], F_final=float(conv[-1]) if conv else None,
                                                                         moves=[x[2] for x in steps], n_steps=len(steps))
    loc = {}
    for jp in sorted(glob.glob(os.path.join(R, "repro", "jet_pin3_D*.npz"))) + sorted(glob.glob(os.path.join(R, "repro16", "jet_pin3_D*.npz"))):
        Z, m = load_jet(jp)
        with redirect_stdout(io.StringIO()):
            c, _ = row_series(Z, "delta")
            r = locate_one(c, m["D"], label="delta")
            c5, i5 = row_series(Z, "max5")
            r5 = locate_one(c5, m["D"], label=f"row {i5}")
        # the rows of the 24 largest |z_8|: each row's nearest stable Pade pole, a singularity of z(D) showing in
        # every component at one place; a 24-order series is also read at its first 16 orders, the 16-order series'
        # bias
        rows = [int(i) for i in np.argsort(np.abs(np.asarray(Z[8][0])))[::-1][:24]]
        L = dict(delta=r, max5=dict(r5, row=i5), delta_coeffs=[float(x) for x in c], norms=m["norms"], N=m["N"])
        for tag, Zt in (("", Z),) + ((("_16", Z[:17]),) if len(Z) > 17 else ()):
            near = []
            for i in rows:
                ci, _ = row_series(Zt, str(i))
                sp = stable_poles(ci, r_max=50)
                near.append(dict(row=i, abs=sp[0]["abs"], arg_deg=sp[0]["arg_deg"], n=sp[0]["n"]) if sp else dict(row=i, abs=None))
            found = [x for x in near if x["abs"] is not None]
            L.update({f"rows{tag}": near, f"rows_found{tag}": len(found),
                      f"rows_abs{tag}": [min(x["abs"] for x in found), float(np.median([x["abs"] for x in found])),
                                         max(x["abs"] for x in found)] if found else None,
                      f"rows_arg{tag}": [min(x["arg_deg"] for x in found), float(np.median([x["arg_deg"] for x in found])),
                                         max(x["arg_deg"] for x in found)] if found else None,
                      f"root_rows{tag}": [root_radius(row_series(Zt, str(i))[0]) for i in rows]})
        loc[f"{m['D']:.4f}"] = L
    J["locate"] = loc
    # the records' norms at every base: the root test over orders 12..N and the fall law from order 5 (max-norms, so
    # no argument and no sign)
    recs = [json.loads(x) for x in open(os.path.join(raw, "I603-win2", "chain", "anm_run.jsonl"))]
    J["records"] = [dict(D0=r["D0"], radius_12=root_radius(r["norms"][1:], k_lo=12), law=growth_law(r["norms"][1:]),
                         z=r["norms"][1:9]) for r in recs if r["D0"] < d_max]
    # the records past D_max, when the checkout has them (the chain went on): the same root test, records only
    J["records_beyond"] = [dict(D0=r["D0"], radius_12=root_radius(r["norms"][1:], k_lo=12), z=r["norms"][1:9],
                                z_N=r["norms"][-1], a_max=r["a_max"],
                                a_accepted=[a["a"] for a in r["attempts"] if a["accepted"]],
                                a_refused=[a["a"] for a in r["attempts"] if not a["accepted"]])
                           for r in recs if d_max <= r["D0"] < d_beyond]
    # the driver against anm_run.py: the od series of jet (N 8) and anm_run.py --save_jet 2's at the same base, every
    # part of every entry at orders 0..8
    J["driver"] = {}
    for jp in sorted(glob.glob(os.path.join(R, "jets", "D*_od.npz"))):
        Dk = os.path.basename(jp).split("_")[0][1:]
        rp = os.path.join(R, "repro", f"jet_pin3_D{Dk}.npz")
        if os.path.exists(rp):
            A, _ = load_jet(jp)
            B, _ = load_jet(rp)
            J["driver"][Dk] = bool(all(np.array_equal(A[k][q], B[k][q]) for k in range(len(A)) for q in range(len(A[k]))))
    # each od series (N 8) against the record of the step from its base: equal norms (of 9) and pivots
    J["od_vs_record"] = {}
    for jp in sorted(glob.glob(os.path.join(R, "jets", "D*_od.npz"))):
        _, m = load_jet(jp)
        r = min(recs, key=lambda x: abs(x["D0"] - m["D"]))
        J["od_vs_record"][f"{m['D']:.4f}"] = dict(equal=sum(a == b for a, b in zip(m["norms"], r["norms"])),
                                                  n=len(m["norms"]), pivots_equal=list(m["pivots"]) == list(r["pivots"]))
    # the reproduction: each 24-order series' norms and pivots against the record of the step from that base
    J["repro"] = {}
    for jp in sorted(glob.glob(os.path.join(R, "repro", "jet_pin3_D*.npz"))) + sorted(glob.glob(os.path.join(R, "repro16", "jet_pin3_D*.npz"))):
        _, m = load_jet(jp)
        r = min(recs, key=lambda x: abs(x["D0"] - m["D"]))
        J["repro"][f"{m['D']:.4f}_N{m['N']}"] = dict(n_norms=len(m["norms"]), equal=sum(a == b for a, b in zip(m["norms"], r["norms"])),
                                            pivots_equal=list(m["pivots"]) == list(r["pivots"]),
                                            max_rel=max(abs(a - b) / b for a, b in zip(m["norms"], r["norms"]) if b))
    with redirect_stdout(io.StringIO()):
        J["attempts"] = refusals([raw, "I603-win2/chain", "--D_max", str(d_max)])
    with open(out, "w") as f:
        json.dump(J, f, indent=1, default=lambda o: [o.real, o.imag] if isinstance(o, complex) else str(o))
    print(f"wrote {out}: {len(pr)} predictions, variants at {sorted(var)}, Delta located at {sorted(loc)}", flush=True)
    return 0


def main():
    cmd, argv = sys.argv[1], sys.argv[2:]
    if cmd == "summary":
        return summary(argv)
    if cmd == "preds":
        return preds(argv)
    if cmd == "reexpand":
        return reexpand(argv)[0]
    if cmd == "refusals":
        refusals(argv)
        return 0
    if cmd == "planted":
        return planted(argv)
    if cmd == "jet":
        return jet(argv)
    if cmd == "compare":
        compare(argv)
        return 0
    if cmd == "locate":
        return locate(argv)
    if cmd == "truth":
        truth(argv)
        return 0
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())

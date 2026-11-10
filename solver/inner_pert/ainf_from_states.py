"""The D = infinity amplitude of the finite-D states against the selection's (agent-comms#54, notes/EEG_PROFILE.md
3.5.1).

States (numerical evidence). A_n = 2|c_1(n)| is the fundamental amplitude of F_n = beta~/sqrt(2 (1 + 1/n)),
beta~ = Pi(tau, 0)/sqrt(eps), at each state (nnlo_closed_form.profile, gauge F_1 = -i A/2). A_inf is the intercept of
the least-squares fit A_n = A_inf + sum_{k=1..kmax} a_k n^-k over the states with D >= dlo (nnlo_closed_form.fit_K,
free=True). The grid G is kmax 5-9 x dlo 15, 20, 30 x the test states 40.02, 60.00, 100, 200 held out or included (30
choices). The central value is the median over G, and s = max_G |A_inf - median| is a spread over fit choices, not an
error bar. The reference choice for sensitivities and diagnostics is kmax 8, dlo 20, all states.

Selection (conditional on Conjecture 1). A_7 is the affine-fit zero of the ten M = 7 logs (m7_outer.py --fit,
proof/B_tail_M7.md 2 and 10), with noise +-2e-5 (the scatter of repeated solves, not an enclosure). Its truncation tail
is trunc_seq_M7.py's geometric estimator on the A sequence M = 3, 5, 7, L_A = A_7 + d_7 k/(1 - k) with k = d_7/d_5, and
that estimator's planted error formula |A_7 - L_A| (|e| + 0.05)/(1 - |k|), with e = k/k_5 - 1 the drift of the ratio.
The limit under five scenarios for the ratios past M = 7 is given too, since the measured drift is far outside the
+-0.1 at which that formula was tested. Post hoc, as information, the range of the converged fit choices (CONVERGED) is
reported beside the pre-registered median, with the selection's values against its midpoint.

Criteria (agent-comms#54, the started comment, written before the runs): R0 (#42 reproduced), P1-P3 and Pm (planted
truth), C (the 1e-5 shift), the mutations slope and step, R1 and R2 (the selection's inputs reproduced).

    python3 ainf_from_states.py planted FIELDS... [--family=alt12,cpx7,alt20] [--noise=SIGMA --seeds=20]
                                         [--mutate=offset] [--json=F] [--summary=F]
    python3 ainf_from_states.py states FIELDS... [--chains=DIR,...] [--mutate=shift|shift<v>|slope|step] [--json=F]
                                         [--summary=F]
    python3 ainf_from_states.py controls FIELDS... [--json=F] [--summary=F]   # C, slope, step and Pm together
    python3 ainf_from_states.py selection [--json=F]               # A_7, its tail, R1, R2 (no fields needed)

--summary=F merges the command's few numbers into F, keeping the other commands' keys (data/ainf_from_states.json is
the notes' copy).

planted exits 1 when P1 fails (or P3, with --noise) and with --mutate=offset exits 0 only when P1 fails, which means
the control fires. states exits 1 when R0, R1 or R2 fails; with a mutation it exits 0 only when that mutation's
criterion holds. controls exits 0 only when every control behaves. selection exits 1 when R1 or R2 fails.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import nnlo_closed_form as CF            # noqa: E402
import nnlo_vs_solution as NV            # noqa: E402

sys.path.insert(0, os.path.join(REPO, "solver", "proof"))
import trunc_seq_M7 as TS                # noqa: E402

sys.path.insert(0, os.path.join(REPO, "solver", "inner_stokes"))
import m7_outer as MO                    # noqa: E402

arg = CF.arg
GRID = [(k, d, h) for k in (5, 6, 7, 8, 9) for d in (15.0, 20.0, 30.0) for h in (True, False)]
REF = (8, 20.0, False)                   # kmax, dlo, test states held out
# post hoc (chosen after the real run, from the planted calibration): the choices that have converged in kmax and dlo
CONVERGED = [(k, d, False) for k in (9, 10, 11, 12) for d in (20.0, 30.0)] + [(8, 30.0, False), (8, 40.0, False)]
R0_REF = dict(main=2.7192088, lo=2.7192055, hi=2.7192530)    # #42's cross-check (AC42-mac/compare.log)
R0_GRID = [(k, d, True) for k in (5, 6, 7) for d in (15.0, 20.0, 30.0)]
R0_TOL = 1e-7
STEP_D = 76.5                            # between the full-system state at 73 and the held-harmonic chains from 80
GROUPS = (("D <= 60", 0.0, 60.5), ("D = 73", 60.5, 76.5), ("D 80-170", 76.5, 175.0), ("D 180-215", 175.0, 1e9))

# planted families: A_inf = 2.7180, #42's 1/n and 1/n^2 coefficients, and non-terminating tails sized to #42's offsets
A_PL, A1_PL, A2_PL = 2.7180, 14.52, -33.0
FAMILIES = {"alt12": lambda t: A_PL + A1_PL * t + A2_PL * t * t / (1 + 12.0 * t),
            "cpx7": lambda t: A_PL + A1_PL * t + A2_PL * t * t / (1 - 7.0 * t + 49.0 * t * t),
            "alt20": lambda t: A_PL + A1_PL * t + A2_PL * t * t / (1 + 20.0 * t)}
P1_FAMILIES = ("alt12", "cpx7")          # alt20 is harsher than the data (information)
TOL_P, OFFSET = 2e-6, 5e-6
SHIFT, SHIFT_TOL = 1e-5, 0.2
SLOPE, SLOPE_TOL = 1e-3, 1e-8
STEP, STEP_TOL = 1e-5, 0.2

# the selection: the ten logs of B_tail_M7.md 10's affine fit, its quoted zero and noise, the quoted omega limit
M7_LOGS = ["hier_nb_M7_fixA_h3_xi015_at_M5zero.log", "hier_nb_M7_fixA_xi015_newton1.log",
           "hier_nb_M7_fixA_xi015_newton2.log", "hier_nb_M7_fixA_xi015_newton2_repeat_a.log",
           "hier_nb_M7_fixA_xi015_newton2_repeat_b.log", "hier_nb_M7_fixA_xi015_newton3.log",
           "hier_nb_M7_fixA_xi015_local_dA.log", "hier_nb_M7_fixA_xi015_local_dw.log",
           "hier_nb_M7_fixA_xi015_fit3.log", "hier_nb_M7_fixA_xi015_fit3_repeat.log"]
A7_QUOTED, W7_QUOTED, LW_QUOTED = 2.719219, 1.000022, 0.999985
A7_NOISE = 2e-5
R1_TOL, R2_TOL = 1e-6, 2e-6
R1_SCATTER = 0.1                         # the repeats' scatter in r_1 (B_tail_M7.md 2)


# ------------------------------------------------------------------ the states' amplitudes and the fit
def amps(profs):
    """A_n = Re(2i chat_1) = 2|c_1| (the fundamental is aligned to -i|c_1|)"""
    return np.array([float(np.real(2j * p["chat"][0])) for p in profs])


def a_inf(profs, kmax, dlo, held, var="n"):
    """one fit choice: A_inf (fit_K's free intercept) and the fundamental's residual in A; var 'eps' fits in
    1/(D - 1) instead of 1/n"""
    ps = profs if var == "n" else [dict(p, n=p["D"] - 1.0) for p in profs]
    ft = CF.fit_K(ps, None, kmax, dlo, hold=CF.TEST_D if held else (), free=True)
    use = [p for p in ps if any(abs(p["D"] - d) < 1e-9 for d in ft["D"])]
    t = np.array([1.0 / p["n"] for p in use])
    model = ft["F_fit"][0] + sum(ft["a"][k - 1][0] * t ** k for k in range(1, kmax + 1))
    res = np.real(2j * (np.array([p["chat"][0] for p in use]) - model))
    return dict(A=float(np.real(2j * ft["F_fit"][0])), rms=float(np.sqrt(np.mean(res ** 2))),
                max_res=float(np.abs(res).max()), n_states=len(use), D=[p["D"] for p in use], res=res.tolist())


def grid(profs, var="n", choices=GRID):
    rows = []
    for (k, d, h) in choices:
        f = a_inf(profs, k, d, h, var)
        rows.append(dict(kmax=k, dlo=d, held=h, A=f["A"], rms=f["rms"], max_res=f["max_res"], n_states=f["n_states"]))
    A = np.array([r["A"] for r in rows])
    med = float(np.median(A))
    return dict(rows=rows, median=med, s=float(np.abs(A - med).max()), lo=float(A.min()), hi=float(A.max()))


def print_grid(g, label, truth=None):
    """the grid as a kmax x (dlo, held) table; with truth, the errors"""
    ref = truth if truth is not None else 0.0
    print(f"\n== {label}: A_inf over the grid" + (f" minus the truth {truth}" if truth is not None else ""))
    cols = [(d, h) for d in (15.0, 20.0, 30.0) for h in (True, False)]
    print("   kmax " + "".join(f"{'D>=%g %s' % (d, 'held' if h else 'all'):>16s}" for d, h in cols))
    for k in sorted({r["kmax"] for r in g["rows"]}):
        line = f"   {k:4d} "
        for d, h in cols:
            r = next(x for x in g["rows"] if x["kmax"] == k and x["dlo"] == d and x["held"] == h)
            line += f"{r['A'] - ref:16.3e}" if truth is not None else f"{r['A']:16.9f}"
        print(line)
    print(f"   median {g['median'] - ref:.9g}  spread s = max |A - median| {g['s']:.2e}  range "
          f"{g['lo'] - ref:.9g} .. {g['hi'] - ref:.9g}")


# ------------------------------------------------------------------ records changed in the field Pi(tau, 0)
def with_beta(r, b):
    """a copy of record r whose central Pi is sqrt(eps) b (profile reads only Pi at x = 0)"""
    i0 = int(np.argmin(np.abs(r["xs"])))
    q = dict(r)
    q["Pi"] = r["Pi"].copy()
    q["Pi"][i0] = np.sqrt(1.0 / (r["D"] - 1.0)) * b
    return q


def add_amplitude(recs, dA):
    """each state's fundamental amplitude raised by dA(D) in the field: beta~ += sqrt(2 (1 + 1/n)) dA sin(theta + phi)
    along the fundamental's phase phi, so that A_n -> A_n + dA exactly"""
    out = []
    for r in recs:
        p = CF.profile(r)
        th = np.pi * np.arange(p["M"]) / p["M"]
        i0 = int(np.argmin(np.abs(r["xs"])))
        b = r["Pi"][i0] / np.sqrt(p["eps"]) + np.sqrt(2 * (1 + 1 / p["n"])) * dA(r["D"]) * np.sin(th + p["phi"])
        out.append(with_beta(r, b))
    return out


def planted_records(recs, fam, noise=0.0, rng=None):
    """each state's central profile replaced by the tuned profile scaled to amplitude fam(1/n) (+ Gaussian noise),
    shifted in time by a known phase; only D, M and the x grid of the state are used"""
    F = CF.tuned_F()
    out, truth = [], []
    for r in recs:
        n = r["D"] - 2.0
        A = fam(1.0 / n) + (rng.normal(0.0, noise) if noise else 0.0)
        phi = 2 * np.pi * ((0.37 * r["D"] + 0.11) % 1.0)
        b = np.sqrt(2 * (1 + 1 / n)) * CF.synth(F * (A / CF.A_TUNED) * np.exp(1j * CF.nus() * phi), r["Pi"].shape[1])
        out.append(with_beta(r, b))
        truth.append(A)
    return out, np.array(truth)


MUTATIONS = {"shift": lambda v: (lambda D: v), "slope": lambda v: (lambda D: v / (D - 2.0)),
             "step": lambda v: (lambda D: v if D >= STEP_D else 0.0)}


def parse_mut(mut):
    for name, default in (("shift", SHIFT), ("slope", SLOPE), ("step", STEP)):
        if mut.startswith(name):
            rest = mut[len(name):]
            return name, float(rest) if rest else default
    raise SystemExit(f"unknown mutation {mut} (shift[<v>], slope[<v>], step[<v>])")


# ------------------------------------------------------------------ diagnostics
def step_fit(profs, kmax=REF[0], dlo=REF[1], d0=STEP_D):
    """the reference fit with an extra column H(D - d0): the step's coefficient and A_inf with it"""
    use = [p for p in profs if p["D"] >= dlo]
    t = np.array([1.0 / p["n"] for p in use])
    X = np.array([t ** k for k in range(kmax + 1)] + [np.array([1.0 if p["D"] >= d0 else 0.0 for p in use])]).T
    sc = np.abs(X).max(axis=0)
    sol, *_ = np.linalg.lstsq(X / sc, amps(use), rcond=None)
    sol = sol / sc
    return dict(A=float(sol[0]), step=float(sol[-1]), kmax=kmax, dlo=dlo, d0=d0)


def groups_of(fit):
    out = []
    for name, lo, hi in GROUPS:
        r = [x for d, x in zip(fit["D"], fit["res"]) if lo <= d < hi]
        if r:
            out.append(dict(group=name, n=len(r), mean=float(np.mean(r)), max=float(np.abs(r).max())))
    return out


def sensitivities(profs, g):
    """one factor at a time around the reference choice"""
    k0, d0, h0 = REF
    base = a_inf(profs, k0, d0, h0)["A"]
    ps2 = [dict(p, chat=p["chat"] * np.sqrt(1 + 1 / p["n"])) for p in profs]
    rows = [("reference (kmax 8, D >= 20, all states)", base),
            ("variable 1/(D - 1)", a_inf(profs, k0, d0, h0, var="eps")["A"]),
            ("prefactor sqrt2 (A_n sqrt(1 + 1/n))", a_inf(ps2, k0, d0, h0)["A"]),
            ("dlo 40", a_inf(profs, k0, 40.0, h0)["A"])]
    rows += [(f"kmax {k}", a_inf(profs, k, d0, h0)["A"]) for k in (4, 10, 11, 12)]
    gv = grid(profs, var="eps")
    gp = grid(ps2)
    rows += [("grid median, variable 1/(D - 1)", gv["median"]), ("grid median, prefactor sqrt2", gp["median"])]
    return [dict(what=w, A=a, dA=a - base) for w, a in rows], dict(eps=gv, sqrt2=gp)


def chain_swap(recs, dirs):
    """the T140 chains in place of the main chain at equal D: A_n differences, and A_inf at the reference and the
    grid median with the swapped states"""
    out = []
    for d in dirs:
        others = {round(o["D"], 6): o for o in (NV.load_fields(os.path.join(d, f)) for f in sorted(os.listdir(d))
                                                 if f.endswith(".npz"))}
        sw, diffs = [], []
        for r in recs:
            o = others.get(round(r["D"], 6))
            if o is None:
                sw.append(r)
                continue
            sw.append(o)
            a_m, a_o = amps([CF.profile(r)])[0], amps([CF.profile(o)])[0]
            diffs.append(dict(D=r["D"], A_main=a_m, A_other=a_o, dA=a_o - a_m))
        ps = [CF.profile(r) for r in sw]
        out.append(dict(dir=os.path.basename(os.path.normpath(d)), diffs=diffs, A_ref=a_inf(ps, *REF)["A"],
                        median=grid(ps)["median"]))
    return out


# ------------------------------------------------------------------ the selection's A (conditional on Conjecture 1)
def geometric(seq):
    """trunc_seq_M7.py's geometric estimator on the last three increments: L = c_7 + d_7 k/(1 - k), k = d_7/d_5, the
    drift e = k/k_5 - 1, and the planted error formula |c_7 - L| (|e| + 0.05)/(1 - |k|)"""
    d = np.diff(seq)
    k, k5 = d[-1] / d[-2], d[-2] / d[-3]
    L = seq[-1] + d[-1] * k / (1 - k)
    e = k / k5 - 1
    return dict(seq=list(seq), d=d.tolist(), k=float(k), k5=float(k5), L=float(L), tail=float(L - seq[-1]),
                e=float(e), bound=float(abs(seq[-1] - L) * (abs(e) + 0.05) / (1 - abs(k))))


def estimator_scenarios(g, k_other):
    """information: the true limit of the measured sequence under scenarios for the ratios past M = 7 (held at k_7,
    back to k_5, the other sequence's k_7, the drift k_7/k_5 continued up to |k| = 1/4, and -1/4 throughout, the
    ratio B_tail.md would need proved), against the estimate L and its error formula"""
    k5, k7, d7, c7 = g["k5"], g["k"], g["d"][-1], g["seq"][-1]
    drift = k7 / k5

    def limit(ks):
        tail, prod = 0.0, d7
        for k in ks:
            prod *= k
            tail += prod
        return c7 + tail

    sc = {"held at k_7": [k7] * 200, "back to k_5": [k5] * 200, "the other sequence's k_7": [k_other] * 200,
          "drift continued, |k| <= 1/4": [float(np.clip(k7 * drift ** (j + 1), -0.25, 0.25)) for j in range(200)],
          "-1/4 throughout": [-0.25] * 200}
    rows = []
    for name, ks in sc.items():
        L = limit(ks)
        rows.append(dict(scenario=name, L=L, tail=L - c7, err=g["L"] - L,
                         covered=bool(abs(g["L"] - L) <= g["bound"])))
    return rows


def selection(verbose=True):
    pts = [MO.point(os.path.join(REPO, "solver", "inner_stokes", "logs", f), True) for f in M7_LOGS]
    use = [t for t in pts if t[3] < 1e-6]
    pref = np.mean([t[1] for t in use], axis=0)
    X = np.array([[1.0, *(t[1] - pref)] for t in use])
    C, *_ = np.linalg.lstsq(X, np.array([t[2] for t in use]), rcond=None)
    J = C[1:].T
    A7, w7 = pref - np.linalg.solve(J, C[0])
    noise_row = float(R1_SCATTER * np.linalg.norm(np.linalg.inv(J)[0]))
    gA = geometric([TS.M1["A"], TS.M3["A"], TS.M5["A"], A7])
    gW = geometric([TS.M1["c"], TS.M3["c"], TS.M5["c"], w7])
    est = estimator_scenarios(gA, gW["k"])
    r1 = bool(abs(A7 - A7_QUOTED) <= R1_TOL and abs(w7 - W7_QUOTED) <= R1_TOL)
    r2 = bool(abs(gW["L"] - LW_QUOTED) <= R2_TOL)
    out = dict(A7=float(A7), w7=float(w7), n_solves=len(use), noise=A7_NOISE, noise_A_row=noise_row, A_seq=gA,
               w_seq=gW, scenarios=est, R1=r1, R2=r2)
    if verbose:
        print("\n== the selection's A (conditional on Conjecture 1): the M = 7 zero and the truncation sequence")
        print(f"   M = 7 affine-fit zero ({len(use)} solves with |r_3| < 1e-6): A_7 = {A7:.9f}, w/w_inf = {w7:.9f}")
        print(f"   R1: B_tail_M7.md's 2.719219, 1.000022 within {R1_TOL:g}: {'PASS' if r1 else 'FAIL'}")
        print(f"   noise: +-{A7_NOISE:g} (B_tail_M7.md 2); the A row of J^-1 times the scatter {R1_SCATTER}: "
              f"{noise_row:.2e} (information)")
        for name, g in (("A", gA), ("w/w_inf", gW)):
            print(f"   {name:8s} M = 1, 3, 5, 7: " + ", ".join(f"{v:.9f}" for v in g["seq"])
                  + f";  d_3, d_5, d_7 = " + ", ".join(f"{v:+.4e}" for v in g["d"]))
            print(f"            ratio d_5/d_3 {g['k5']:+.4f}, d_7/d_5 {g['k']:+.4f} (drift e {g['e']:+.3f});  "
                  f"geometric limit L {g['L']:.9f}, tail L - c_7 {g['tail']:+.2e}, estimator formula {g['bound']:.1e}")
        print(f"   R2: the omega limit {gW['L']:.6f} against B_tail_M7.md's {LW_QUOTED} within {R2_TOL:g}: "
              f"{'PASS' if r2 else 'FAIL'}")
        print("   A's limit under scenarios for the ratios past M = 7 (information), against L and its formula:")
        for x in est:
            print(f"      {x['scenario']:30s} limit {x['L']:.9f}  tail {x['tail']:+.2e}  L - limit {x['err']:+.2e}  "
                  f"{'covered' if x['covered'] else 'NOT covered'} by {gA['bound']:.1e}")
    return out


def gaps(central, s, sel):
    """the gap between the selection's A and the states' central value, in units of s, and against the combined
    sqrt(s^2 + noise^2 + tail^2) (A_7: the tail |L_A - A_7|; L_A: the estimator formula)"""
    gA = sel["A_seq"]
    out = []
    for name, A, tail in (("A_7", sel["A7"], abs(gA["tail"])), ("L_A", gA["L"], gA["bound"])):
        G = A - central
        sig = float(np.sqrt(s ** 2 + sel["noise"] ** 2 + tail ** 2))
        out.append(dict(what=name, A=A, gap=float(G), in_s=float(G / s), combined=sig, in_combined=float(G / sig),
                        consistent=bool(abs(G) <= 2 * sig)))
    return out


def converged(profs):
    """information, post hoc: A_inf over the converged choices, their range and midpoint"""
    A = [a_inf(profs, *c)["A"] for c in CONVERGED]
    return dict(lo=float(min(A)), hi=float(max(A)), mid=float((min(A) + max(A)) / 2), half=float((max(A) - min(A)) / 2))


def implied_ratio(central, sel):
    """the next ratio k of the A sequence for which its limit is the states' value, A_inf - A_7 = d_7 k/(1 - k), with
    A_7 moved by its noise (which moves d_7 too)"""
    A5 = sel["A_seq"]["seq"][2]
    ks = []
    for dA7 in (0.0, -sel["noise"], sel["noise"]):
        A7 = sel["A7"] + dA7
        x = (central - A7) / (A7 - A5)
        ks.append(float(x / (1 + x)))
    return dict(k=ks[0], lo=min(ks), hi=max(ks), k_A=sel["A_seq"]["k"], k_w=sel["w_seq"]["k"])


# ------------------------------------------------------------------ commands
def load(files):
    recs = sorted([NV.load_fields(f) for f in files], key=lambda r: r["D"])
    return recs, [CF.profile(r) for r in recs]


def dump(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=lambda z: z.tolist() if isinstance(z, np.ndarray) else
                  [z.real, z.imag] if isinstance(z, complex) else str(z))
    print(f"\nwrote {path}")


def merge_summary(path, part):
    """the notes' summary file: each command replaces its own keys and keeps the others"""
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(json.loads(json.dumps(part, default=str)))

    def short(z):                        # ten significant digits keep the file at a few KB
        if isinstance(z, float):
            return float(f"{z:.10g}")
        if isinstance(z, dict):
            return {k: short(v) for k, v in z.items()}
        return [short(v) for v in z] if isinstance(z, list) else z

    with open(path, "w") as fh:
        fh.write("{\n" + ",\n".join(f"{json.dumps(k)}: {json.dumps(short(v))}" for k, v in old.items()) + "\n}\n")
    print(f"\nwrote {path}")


def planted_family(recs, name, noise=0.0, seed=0, verbose=True, target=A_PL):
    rng = np.random.default_rng(seed) if noise else None
    precs, truth = planted_records(recs, FAMILIES[name], noise, rng)
    profs = [CF.profile(r) for r in precs]
    got = amps(profs)
    g = grid(profs)
    if verbose:
        print_grid(g, f"planted {name}" + (f", noise {noise:g}, seed {seed}" if noise else ""), truth=target)
    g["converged"] = converged(profs)
    return g, float(np.abs(got - truth).max())


def cmd_planted(argv):
    files = [a for a in argv if not a.startswith("--")]
    recs = sorted([NV.load_fields(f) for f in files], key=lambda r: r["D"])
    fams = arg(argv, "family", ",".join(FAMILIES)).split(",")
    noise, seeds = float(arg(argv, "noise", 0.0)), int(arg(argv, "seeds", 20))
    mut = arg(argv, "mutate")
    if mut not in (None, "offset"):
        raise SystemExit(f"unknown mutation {mut} (planted takes --mutate=offset)")
    target = A_PL + (OFFSET if mut else 0.0)
    out, ver = dict(families={}, mutation=mut, target=target), []
    for name in fams:
        g, amp_err = planted_family(recs, name, target=target)
        err = g["median"] - target
        out["families"][name] = dict(grid=g, err=err, amp_err=amp_err)
        cv = g["converged"]
        cerr = max(abs(cv["lo"] - target), abs(cv["hi"] - target))
        out["families"][name]["converged_err"] = cerr
        print(f"   {name}: the converged choices (post hoc, information) lie within {cerr:.1e} of the truth")
        ver.append(dict(c=f"Pz {name}: planted A_n recovered by profile within 1e-12", v=amp_err, ok=amp_err <= 1e-12))
        if name in P1_FAMILIES:
            ver.append(dict(c=f"P1 {name}: |median - truth| <= {TOL_P:g}", v=abs(err), ok=abs(err) <= TOL_P))
        print(f"   {name}: median - truth {err:+.2e}, spread s {g['s']:.2e}; P2 (information): s "
              f"{'covers' if abs(err) <= g['s'] else 'does not cover'} the error; max |A_n - planted| {amp_err:.1e}")
    if noise:
        for name in [f for f in fams if f in P1_FAMILIES]:
            errs = np.array([planted_family(recs, name, noise, s, verbose=False)[0]["median"] - target
                             for s in range(seeds)])
            med, p90 = float(np.median(np.abs(errs))), float(np.quantile(np.abs(errs), 0.9))
            out["families"][name]["noise"] = dict(sigma=noise, seeds=seeds, errs=errs.tolist(), median_abs=med,
                                                  p90_abs=p90)
            ver.append(dict(c=f"P3 {name}, noise {noise:.1e}: median |err| over {seeds} seeds <= {TOL_P:g}", v=med,
                            ok=med <= TOL_P))
            print(f"   {name} with noise {noise:.2e} over {seeds} seeds: median |median - truth| {med:.2e}, "
                  f"90th percentile {p90:.2e}")
    for v in ver:
        v["ok"] = bool(v["ok"])
    print("\n== planted verdicts" + (f" (mutation {mut}: compared against the truth + {OFFSET:g})" if mut else ""))
    for v in ver:
        print(f"   {v['c']:62s} {v['v']:9.2e}  {'PASS' if v['ok'] else 'FAIL'}")
    out["verdicts"] = ver
    js = arg(argv, "json")
    if js:
        dump(js, out)
    sm = arg(argv, "summary")
    if sm:
        # planted: errors at three choices; offset: its verdicts; noise: the P3 numbers only
        key = "planted" + (f"_{mut}" if mut else "") + ("_noise" if noise else "")
        keep = ("P3",) if noise else ("P1",) if mut else ("P1", "Pz")
        fam = {}
        for name, f in out["families"].items():
            if noise:
                if "noise" in f:
                    fam[name] = {k: v for k, v in f["noise"].items() if k != "errs"}
                continue
            pick = {f"{r['kmax']},{r['dlo']:g},{'held' if r['held'] else 'all'}": r["A"] - target
                    for r in f["grid"]["rows"] if (r["kmax"], r["dlo"], r["held"]) in
                    ((5, 15.0, True), (6, 20.0, True), (9, 30.0, False))}
            fam[name] = (dict(median_err=f["err"], s=f["grid"]["s"], err_at=pick, converged_err=f["converged_err"])
                         if not mut else f["err"])
        merge_summary(sm, {key: dict(target=target, families=fam, verdicts=[
            [v["c"], v["v"], v["ok"]] for v in ver if v["c"].startswith(keep)])})
    p1 = all(v["ok"] for v in ver if v["c"].startswith("P1"))
    rest = all(v["ok"] for v in ver if not v["c"].startswith("P1"))
    if mut:
        fired = not p1 and rest
        print(f"\nmutation {mut}: {'P1 fails, the control fires (exit 0)' if fired else 'CONTROL FAILED (exit 1)'}")
        return 0 if fired else 1
    ok = p1 and rest
    print(f"\n{'ALL PASS' if ok else 'FAILED'}")
    return 0 if ok else 1


def criterion(name, v, dA, step):
    """a mutation's criterion on the grid's changes dA and the step fit's change: (label, measure, ok)"""
    if name == "shift":
        dev = float(np.abs(dA / v - 1).max())
        return f"C shift {v:g}: every choice moves by {v:g} within {SHIFT_TOL:.0%}", dev, bool(dev <= SHIFT_TOL)
    if name == "slope":
        dev = float(np.abs(dA).max())
        return f"slope {v:g}/n: every choice moves by <= {SLOPE_TOL:g}", dev, bool(dev <= SLOPE_TOL)
    dev = float(abs(step / v - 1))
    return f"step {v:g} at D >= {STEP_D}: the step fit recovers it within {STEP_TOL:.0%}", dev, bool(dev <= STEP_TOL)


def mutation_check(recs, g0, mut, verbose=True):
    """the mutated grid against the unmutated one, the step fit's change, and the mutation's criterion"""
    name, v = parse_mut(mut)
    mprofs = [CF.profile(r) for r in add_amplitude(recs, MUTATIONS[name](v))]
    g1 = grid(mprofs)
    dA = np.array([b["A"] - a["A"] for a, b in zip(g0["rows"], g1["rows"])])
    step = step_fit(mprofs)["step"] - step_fit([CF.profile(r) for r in recs])["step"]
    c, dev, ok = criterion(name, v, dA, step)
    out = dict(mutation=mut, name=name, value=v, dA=dA.tolist(), dA_min=float(dA.min()), dA_max=float(dA.max()),
               dA_median=float(g1["median"] - g0["median"]), dA_ref=float(dA[GRID.index(REF)]), step_fitted=step,
               c=c, v=dev, ok=ok)
    if verbose:
        print(f"\n== mutation {mut}: A_inf moves by {dA.min():+.3e} .. {dA.max():+.3e} over the grid (median "
              f"{out['dA_median']:+.3e}, reference {out['dA_ref']:+.3e}); the step fit's step moves {step:+.3e}")
        print(f"   {c}: {dev:.2e}  {'PASS' if ok else 'FAIL'}")
    return out


def check_r0(profs):
    main = a_inf(profs, 6, 20.0, True)["A"]
    A = [a_inf(profs, k, d, h)["A"] for (k, d, h) in R0_GRID]
    dev = max(abs(main - R0_REF["main"]), abs(min(A) - R0_REF["lo"]), abs(max(A) - R0_REF["hi"]))
    return dict(main=main, lo=min(A), hi=max(A), dev=dev, ok=bool(dev <= R0_TOL))


def analysis(recs, profs, argv):
    out = {}
    r0 = check_r0(profs)
    out["R0"] = r0
    print(f"\n== R0: kmax 6, D >= 20, test held out: {r0['main']:.7f} (#42: {R0_REF['main']}); #42's nine choices "
          f"{r0['lo']:.7f} .. {r0['hi']:.7f} (#42: {R0_REF['lo']} .. {R0_REF['hi']}); max dev {r0['dev']:.1e}  "
          f"{'PASS' if r0['ok'] else 'FAIL'}")
    g = grid(profs)
    out["grid"] = g
    print_grid(g, "the states (numerical evidence)")
    ref = a_inf(profs, *REF)
    out["reference"] = ref
    out["groups"] = groups_of(ref)
    print(f"\n== reference fit (kmax 8, D >= 20, all states): A_inf {ref['A']:.9f}, {ref['n_states']} states, "
          f"residual rms {ref['rms']:.2e}, max {ref['max_res']:.2e}")
    for x in out["groups"]:
        print(f"   {x['group']:10s} {x['n']:3d} states: mean residual {x['mean']:+.2e}, max |residual| {x['max']:.2e}")
    sens, extra = sensitivities(profs, g)
    out["sensitivities"] = sens
    out["sens_grids"] = {k: dict(median=v["median"], s=v["s"], lo=v["lo"], hi=v["hi"]) for k, v in extra.items()}
    print("\n== sensitivities (information): one factor at a time around the reference")
    for x in sens:
        print(f"   {x['what']:44s} {x['A']:.9f}  {x['dA']:+.2e}")
    st = step_fit(profs)
    out["step"] = st
    print(f"\n== step diagnostic: a step at D >= {STEP_D} fitted with the reference: {st['step']:+.3e}; A_inf with it "
          f"{st['A']:.9f} ({st['A'] - ref['A']:+.2e} against the reference)")
    dirs = [d for d in arg(argv, "chains", "").split(",") if d]
    if dirs:
        out["chains"] = chain_swap(recs, dirs)
        print("\n== the T140 chains in place of the main chain at equal D (information)")
        for c in out["chains"]:
            print(f"   {c['dir']}: " + "  ".join(f"D {x['D']:.0f} dA_n {x['dA']:+.2e}" for x in c["diffs"])
                  + f";  A_inf reference {c['A_ref']:.9f} ({c['A_ref'] - ref['A']:+.2e}), grid median "
                    f"{c['median']:.9f} ({c['median'] - g['median']:+.2e})")
    sel = selection()
    out["selection"] = sel
    gp = gaps(g["median"], g["s"], sel)
    out["gaps"] = gp
    print(f"\n== the gap: selection minus the states' median {g['median']:.9f} (s = {g['s']:.2e}); combined "
          "= sqrt(s^2 + noise^2 + tail^2)")
    for x in gp:
        print(f"   {x['what']} = {x['A']:.9f}: gap {x['gap']:+.2e} = {x['in_s']:+.2f} s; combined {x['combined']:.2e}, "
              f"gap {x['in_combined']:+.2f} of it: {'consistent' if x['consistent'] else 'DISCREPANCY'}")
    cv = converged(profs)
    cv["gap_A7"] = sel["A7"] - cv["mid"]
    cv["gap_L"] = {x["scenario"]: x["L"] - cv["mid"] for x in sel["scenarios"]}
    cv["minus_tuned"], cv["R0_minus"] = cv["mid"] - CF.A_TUNED, out["R0"]["main"] - cv["mid"]
    out["converged"] = cv
    print(f"   information, post hoc: the converged choices (kmax 9-12 at D >= 20 and 30, kmax 8 at D >= 30 and 40, "
          f"all states) give {cv['lo']:.9f} .. {cv['hi']:.9f}; A_7 minus their midpoint {cv['gap_A7']:+.2e}; the "
          "limits of A minus it: " + ", ".join(f"{k} {v:+.2e}" for k, v in cv["gap_L"].items()))
    print(f"   their midpoint minus the tuned profile's {CF.A_TUNED}: {cv['minus_tuned']:+.2e}; #42's kmax 6, D >= 20 "
          f"value minus their midpoint: {cv['R0_minus']:+.2e}")
    ir = implied_ratio(g["median"], sel)
    out["implied_ratio"] = ir
    print(f"   the A sequence's next ratio that puts its limit on the median: {ir['k']:+.4f} ({ir['lo']:+.4f} .. "
          f"{ir['hi']:+.4f} with A_7 +- its noise), against the last A ratio {ir['k_A']:+.4f} and omega's "
          f"{ir['k_w']:+.4f}")
    ast = np.sqrt(2) * g["median"]
    out["A_star"] = dict(A_star=float(ast), minus_tuned=float(g["median"] - CF.A_TUNED))
    print(f"   sqrt2 x median = {ast:.6f} (EEG's A_* is printed as 3.8455); median minus the tuned profile's "
          f"{CF.A_TUNED}: {g['median'] - CF.A_TUNED:+.2e}")
    return out


def summary(out):
    """the few numbers the notes cite (a few KB)"""
    g, sel = out["grid"], out["selection"]
    return dict(what="agent-comms#54: the states' D = infinity amplitude against the selection's (ainf_from_states.py)",
                status=dict(states="numerical evidence", selection="conditional on Conjecture 1"),
                grid={f"{r['kmax']},{r['dlo']:g},{'held' if r['held'] else 'all'}": r["A"] for r in g["rows"]},
                median=g["median"], s=g["s"], range=[g["lo"], g["hi"]], R0=out["R0"],
                reference=dict(A=out["reference"]["A"], rms=out["reference"]["rms"]),
                sensitivities=[dict(what=x["what"], dA=x["dA"]) for x in out["sensitivities"]],
                step=out["step"], chains=[dict(dir=c["dir"], dA_ref=c["A_ref"] - out["reference"]["A"],
                                               dA_n=[x["dA"] for x in c["diffs"]]) for c in out.get("chains", [])],
                selection=dict(A7=sel["A7"], w7=sel["w7"], noise=sel["noise"], A_ratios=[sel["A_seq"]["k5"],
                               sel["A_seq"]["k"]], L_A=sel["A_seq"]["L"], tail=sel["A_seq"]["tail"],
                               formula=sel["A_seq"]["bound"], L_w=sel["w_seq"]["L"],
                               scenarios=[dict(scenario=x["scenario"], L=x["L"], tail=x["tail"])
                                          for x in sel["scenarios"]]),
                gaps=out["gaps"], implied_ratio=out["implied_ratio"], A_star=out["A_star"],
                converged=out["converged"])


def cmd_states(argv):
    files = [a for a in argv if not a.startswith("--")]
    recs, profs = load(files)
    out = analysis(recs, profs, argv)
    mut = arg(argv, "mutate")
    if mut:
        out["mutation"] = mutation_check(recs, out["grid"], mut)
    js = arg(argv, "json")
    if js:
        dump(js, out)
    sm = arg(argv, "summary")
    if sm:
        merge_summary(sm, summary(out))
    ok = out["R0"]["ok"] and out["selection"]["R1"] and out["selection"]["R2"]
    if mut:
        fired = out["mutation"]["ok"]
        print(f"\nmutation {mut}: {'its criterion holds (exit 0)' if fired else 'CRITERION FAILED (exit 1)'}"
              + ("" if ok else "; R0, R1 or R2 FAILED"))
        return 0 if fired and ok else 1
    print(f"\n{'R0, R1, R2 PASS' if ok else 'FAILED'}")
    return 0 if ok else 1


def cmd_controls(argv):
    files = [a for a in argv if not a.startswith("--")]
    recs, profs = load(files)
    g0 = grid(profs)
    res = [mutation_check(recs, g0, m) for m in ("shift", "slope", "step")]
    # each criterion applied to another mutation's changes must fail: the step for C, the shift for slope, the slope
    # for step (a criterion only ever seen to pass is untested)
    by = {r["name"]: r for r in res}
    for crit, other in (("shift", "step"), ("slope", "shift"), ("step", "slope")):
        v = dict(shift=SHIFT, slope=SLOPE, step=STEP)[crit]
        c, dev, ok = criterion(crit, v, np.array(by[other]["dA"]), by[other]["step_fitted"])
        res.append(dict(c=f"{c.split(':')[0]} on the {other} mutation: must fail", v=dev, ok=not ok))
    print("\n== Pm: planted truth compared against the truth + 5e-6 (P1 must fail)")
    pm = cmd_planted(files + ["--family=" + ",".join(P1_FAMILIES), "--mutate=offset"]) == 0
    res.append(dict(c="Pm planted offset 5e-6: P1 fails", v=float("nan"), ok=pm))
    print("\n== controls")
    for r in res:
        print(f"   {r['c']:66s} {r['v']:9.2e}  {'PASS' if r['ok'] else 'FAIL'}")
    js = arg(argv, "json")
    if js:
        dump(js, res)
    sm = arg(argv, "summary")
    if sm:
        merge_summary(sm, dict(controls=[dict(c=r["c"], v=None if np.isnan(r["v"]) else r["v"], ok=r["ok"])
                                         for r in res]))
    ok = all(r["ok"] for r in res)
    print(f"\n{'every control behaves (exit 0)' if ok else 'A CONTROL FAILED (exit 1)'}")
    return 0 if ok else 1


def cmd_selection(argv):
    sel = selection()
    js = arg(argv, "json")
    if js:
        dump(js, sel)
    ok = sel["R1"] and sel["R2"]
    print(f"\n{'R1, R2 PASS' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    cmds = dict(planted=cmd_planted, states=cmd_states, controls=cmd_controls, selection=cmd_selection)
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print(__doc__)
        sys.exit(2)
    sys.exit(cmds[sys.argv[1]](sys.argv[2:]))

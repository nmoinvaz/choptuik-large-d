#!/usr/bin/env python3
"""Newton's correction test in the equilibrated unknowns (#504): octo_corrector.newton_od with scale_c = D_c, the column
scaling of near_null_eq.py's result (Curtis-Reid then Ruiz on |J|, #502), on a planted state and on #489's polish.

    cd hp && python3 eq_test.py planted RAW STATE --out PREFIX [--test raw|eq] [--S none|random|unmoved] [--S_seed 7]
        [--threads 2]
        [--rel 1e-12] [--seed 1] [--it_max 12] [--tol 1e-40] [--J0 FILE.npz] [--newton_from FILE.py]
    python3 eq_test.py predict PREFIX --J0 FILE.npz [--seeds 7,8,unmoved] [--n N] [--tol 1e-40] [--U 20,40] [--draws 500]
    python3 eq_test.py verdict A B C D
    python3 eq_test.py same X Y [--mutate_bit 1]
    python3 eq_test.py polish [--Dc_ref L08.json] [--Dc_out FILE.npy] [--hist_out PREFIX] ANM_RUN_ARGS...

planted: octo_corrector.planted's start (the converged state with its centre data and its nodes but the last moved by
--rel relative, --seed) corrected by newton_od inside ElimO, as planted's octo-double run does, every trial printed.
--S random or unmoved gives the unknowns a planted column scaling: the corrector runs on w = S^-1 z, the system J S
(ScaledO), with S_j powers of two, so that z = S w is exact in every part and the kernel sees the same points as without
S. random: S_j the power of two nearest 10^u_j, u_j uniform in [-20, 20] (numpy default_rng(--S_seed), #502's plant).
unmoved: S_j = 2^-66 (1.4e-20) on the unknowns the plant leaves at the truth (the last node, m_h, a_h, Delta) and 2^66
on those it moves. The raw test's max norms then fall on the unmoved unknowns, whose first correction (the first step's
O(p^2) error there) the next correction undoes. --test raw is newton_od's own test, the
leading parts' max norms of the corrections in w. --test eq is the test in the equilibrated unknowns, with D_c from
near_null_eq.scalings on |J S| = |J| S, J the f64 leading part of the octo-double Jacobian at the start (Solver.jacobian_o,
cached in --J0). --newton_from runs the newton_od of another copy of octo_corrector.py (the bit-identity check of
scale_c None against the code before it). Writes PREFIX.json (the lam sequence, the history with every trial, the
final max|F| and Delta), PREFIX.npz (the iterate S w, eight parts) and PREFIX_trace.npz (each trial's leading parts of
dz and of its correction, in z).

predict: the decisions the raw test under S, and the equilibrated test with and without S, make at the trials of the
plain run PREFIX (--test raw, no S), from its trace. Along that run's path only: a test that decides differently there
leaves it. Then each trial's two sides of both tests and, with --U, the largest raw ratio under --draws random S with u
uniform in [-U, U].

verdict (exit 0 pass, 1 fail, 2 the plant does not show), fixed before the runs: A plain (raw, no S), B (eq, no S),
C (eq, S), D (raw, S), from one start and one S. The plant shows when D's lam sequence differs from A's (the control:
the raw test changes with the scaling of the unknowns). The check passes when C's lam sequence equals B's, step by step,
with the same outcome (converged or not). The final iterates' bit equality and Delta difference are printed beside it.

same: the bit-identity check of two planted runs, the iterates' eight parts and the histories without their times.
Exit 1 when they differ. Control: --mutate_bit 1 flips the last bit of Y's eighth part at one entry, and it must fail.

polish: anm_run.main with the given arguments (#489's run_polish.sh), its newton_od given scale_c = D_c from |J| at the
first corrector call's start and verbose 2. --Dc_ref: near_null_eq's json at the same state (#502's L08_plant7.json),
whose D_c is compared with this one up to a constant factor (the scaling is unique up to D_r t, D_c / t).
--hist_out PREFIX appends each corrector call's history to PREFIX.jsonl and its trace to PREFIX_trace<call>.npz.
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from near_null_eq import RESULT, scalings  # noqa: E402  (sets mp.dps 140 at import, reset below)
import mpmath as mp  # noqa: E402
from octo_corrector import ElimO, _delta, load_o, newton_od, o8, o8_of_q, omax, pinned_o, plant_start, save_o  # noqa: E402
from raw_out import require_raw  # noqa: E402

mp.mp.dps = 50          # reseed.py's and anm_run.py's precision


def plant_S(seed, L):
    """the planted column scaling: the power of two nearest 10^u_j, u_j uniform in [-20, 20] (#502's draw)"""
    u = np.random.default_rng(seed).uniform(-20, 20, L)
    return np.exp2(np.round(u * np.log2(10.0)))


def eq_scaling(A, tol=1e-12):
    """near_null_eq's result scaling of |J| = A: (D_r, D_c, Ruiz iterations, largest |1 - max-norm|)"""
    return scalings(A, tol)[RESULT]


class ScaledO:
    """s (ElimO or PinnedO) in the unknowns w = S^-1 z, S a power of two per unknown: residual_o(w) = F(S w) and
    newton_o(w) = S^-1 times the step at S w, the Newton step of the system J S. Every product is checked exact."""

    def __init__(self, s, S):
        self.s, self.S = s, np.asarray(S, dtype=float)

    def __getattr__(self, name):
        return getattr(self.s, name)

    def _map(self, x, f, g):
        out = tuple(np.ascontiguousarray(f(np.asarray(p))) for p in o8(x))
        for p, q in zip(out, o8(x)):
            if not np.array_equal(g(p), q, equal_nan=True):
                raise ArithmeticError("ScaledO: a scaling by a power of two was not exact")
        return out

    def z_of(self, w):
        return self._map(w, lambda p: p * self.S, lambda p: p / self.S)

    def w_of(self, z):
        return self._map(z, lambda p: p / self.S, lambda p: p * self.S)

    def residual_o(self, w, mutate=0):
        return self.s.residual_o(self.z_of(w), mutate=mutate)

    def newton_o(self, w, mutate=0):
        st = self.s.newton_o(self.z_of(w), mutate=mutate)
        return (self.w_of(st[0]), *st[1:])


def _opts(argv, opt):
    for k in list(opt):
        if "--" + k in argv:
            i = argv.index("--" + k)
            opt[k] = type(opt[k])(argv[i + 1])
    return opt


def _lams(rec):
    """(the accepted lam of each step, converged)"""
    return [h["lam"] for h in rec["hist"] if "lam" in h], bool(rec["converged"])


def planted(argv):
    from hpkernel.driver import load_q
    from reseed import Reseeder
    raw, path = argv[0], argv[1]
    opt = _opts(argv[2:], dict(threads=2, rel=1e-12, seed=1, S="none", S_seed=7, test="raw", it_max=12, tol=1e-40, J0="",
                               newton_from="", out=""))
    if not opt["out"] or opt["test"] not in ("raw", "eq") or opt["S"] not in ("none", "random", "unmoved"):
        raise SystemExit("planted: --out PREFIX, --test raw or eq, --S none, random or unmoved")
    Sname = dict(none="none", random=f"random:{opt['S_seed']}", unmoved="unmoved")[opt["S"]]
    require_raw(opt["out"] + ".json")
    nod = newton_od
    if opt["newton_from"]:
        spec = importlib.util.spec_from_file_location("octo_from", opt["newton_from"])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        nod = mod.newton_od
    z4, meta = load_q(os.path.join(raw, path))
    z4 = o8(z4)
    D = mp.mpf(meta["D_str"])
    s, sp, sq, so = pinned_o(meta, z4, threads=opt["threads"])
    rs = Reseeder(meta, D, threads=opt["threads"], qd=True)
    zs, n = plant_start(z4, rs, opt["rel"], opt["seed"], 0)
    z0 = o8_of_q(zs)
    L = len(z0[0])
    S = None
    if opt["S"] == "random":
        S = plant_S(opt["S_seed"], L)
    elif opt["S"] == "unmoved":
        S = np.where(np.arange(L) >= n, 2.0 ** -66, 2.0 ** 66)
    print(f"planted {path}: D {mp.nstr(D, 12)}, L {L}, start = truth moved by {opt['rel']:.0e} relative in [0, {n}), seed "
          f"{opt['seed']}; pinned max|F| (octo) truth {omax(so.residual_o(o8_of_q(z4))):.2e}, start {omax(so.residual_o(z0)):.2e}; "
          f"test {opt['test']}, S " + (f"{Sname}: 2^k in {S.min():.1e}..{S.max():.1e}" if S is not None else "none")
          + (f", newton_od from {opt['newton_from']}" if opt["newton_from"] else ""), flush=True)
    dc = None
    rec = dict(test=opt["test"], state=path, D_str=meta["D_str"], rel=opt["rel"], seed=opt["seed"], S=Sname,
               tol=opt["tol"], it_max=opt["it_max"], newton_from=opt["newton_from"])
    A = None
    if opt["J0"] and os.path.exists(opt["J0"]):
        A = np.abs(np.load(opt["J0"])["J0"])
        print(f"  |J| at the start from {opt['J0']}", flush=True)
    elif opt["J0"] or opt["test"] == "eq":
        # the f64 leading part of the octo-double Jacobian at the start, kept in --J0 for the other runs and predict
        t0 = time.time()
        J0 = np.asarray(s.jacobian_o(list(z0))[0])
        A = np.abs(J0)
        print(f"  jacobian_o at the start {time.time() - t0:.0f} s", flush=True)
        if opt["J0"]:
            np.savez_compressed(opt["J0"], J0=J0)
    if opt["test"] == "eq":
        if S is not None:
            A = A * S[None, :]
        dr, dc, its, err = eq_scaling(A)
        print(f"  D_c ({RESULT}, {its} Ruiz iterations, largest |1 - max-norm| {err:.1e}) {dc.min():.2e}..{dc.max():.2e}, "
              f"D_r {dr.min():.2e}..{dr.max():.2e}", flush=True)
        rec.update(D_c=[float(dc.min()), float(dc.max())], ruiz_its=its, ruiz_err=err)
    el = ElimO(so, rs)
    sy = el if S is None else ScaledO(el, S)
    w0 = z0 if S is None else sy.w_of(z0)
    trace = []
    t0 = time.time()
    # the code before scale_c has neither scale_c nor trace, and prints only the accepted steps
    kw = dict(tol=opt["tol"], it_max=opt["it_max"], verbose=True) if opt["newton_from"] else \
        dict(tol=opt["tol"], it_max=opt["it_max"], verbose=2, trace=trace)
    if dc is not None:
        kw["scale_c"] = dc
    w, hist, conv = nod(sy, w0, **kw)
    z = w if S is None else sy.z_of(w)
    zp = el.project(z)
    F, Dl = omax(so.residual_o(zp)), _delta(zp)
    rec.update(hist=hist, converged=bool(conv), F=F, Delta=mp.nstr(Dl, 40), counts=dict(el.counts),
               seconds=round(time.time() - t0, 1))
    rec["lams"] = _lams(rec)[0]
    print(f"  {len(hist)} records, {'converged' if conv else 'NOT converged'}, lam sequence {rec['lams']}; pinned max|F| (octo) "
          f"{F:.2e}, Delta {mp.nstr(Dl, 40)}, projections {el.counts}, {rec['seconds']:.0f} s", flush=True)
    save_o(opt["out"] + ".npz", z, dict(meta, planted=True, test=opt["test"], S=Sname))
    d = np.array([t["d"] for t in trace]) if trace else np.zeros((0, L))
    dt = np.array([t["dt"] if t["dt"] is not None else np.full(L, np.nan) for t in trace]) if trace else np.zeros((0, L))
    if S is not None:                    # the trace in z
        d, dt = d * S[None, :], dt * S[None, :]
    np.savez_compressed(opt["out"] + "_trace.npz", it=np.array([t["it"] for t in trace]), lam=np.array([t["lam"] for t in trace]),
                        accepted=np.array([t["accepted"] for t in trace]), F=np.array([t["F"] for t in trace]), d=d, dt=dt)
    with open(opt["out"] + ".json", "w") as f:
        f.write(json.dumps(rec, default=float) + "\n")
    return 0


def predict(argv):
    """the decisions of the raw test under S and of the equilibrated test (with and without S) at the plain run's trials"""
    prefix = argv[0]
    opt = _opts(argv[1:], dict(J0="", seeds="7", tol=1e-40, n=0, U="", draws=500))
    t = np.load(prefix + "_trace.npz")
    A = np.abs(np.load(opt["J0"])["J0"])
    L = A.shape[0]
    dc0 = eq_scaling(A)[1]

    def rule(d, dt, F, c):
        return bool(np.isfinite(F) and (np.abs(dt / c).max() < np.abs(d / c).max() or F <= opt["tol"]))
    out = {}
    base = [rule(t["d"][i], t["dt"][i], t["F"][i], np.ones(L)) for i in range(len(t["lam"]))]
    eq0 = [rule(t["d"][i], t["dt"][i], t["F"][i], dc0) for i in range(len(t["lam"]))]
    print(f"{prefix}: {len(base)} trials, plain decisions {base} (recorded {list(map(bool, t['accepted']))})")
    print(f"  eq, no S: {eq0}{'' if eq0 == base else '  <- differs from the plain test'}")
    for k in opt["seeds"].split(","):
        # a seed of the random S, or 'unmoved' (--n: the plant's moved unknowns)
        S = np.where(np.arange(L) >= opt["n"], 2.0 ** -66, 2.0 ** 66) if k == "unmoved" else plant_S(int(k), L)
        dck = eq_scaling(A * S[None, :])[1]
        raw = [rule(t["d"][i] / S, t["dt"][i] / S, t["F"][i], np.ones(L)) for i in range(len(base))]
        eqk = [rule(t["d"][i] / S, t["dt"][i] / S, t["F"][i], dck) for i in range(len(base))]
        out[k] = dict(raw=raw, eq=eqk)
        print(f"  S {k}: raw {raw}{'' if raw == base else '  <- the raw test changes'}; eq {eqk}"
              f"{'' if eqk == eq0 else '  <- the eq test changes'}")
    for i in range(len(base)):
        d, dt = np.abs(t["d"][i]), np.abs(t["dt"][i])
        print(f"  trial {i} (step {t['it'][i]}, lam {t['lam'][i]:g}, max|F| {t['F'][i]:.2e}): ratio of the test's two sides raw "
              f"{dt.max() / d.max():.10f}, eq {(dt / dc0).max() / (d / dc0).max():.10f}; largest |dz_t,j| / |dz_j| "
              f"{np.max(dt / d):.4e} at j {int(np.argmax(dt / d))}", flush=True)
        for U in [float(x) for x in opt["U"].split(",") if x]:
            # the raw ratio under draws of S = 10^u, u uniform in [-U, U] (seeds 5000 on), its largest value
            r = []
            for k in range(opt["draws"]):
                u = np.random.default_rng(5000 + k).uniform(-U, U, L)
                S = np.exp2(np.round(u * np.log2(10.0)))
                r.append((dt / S).max() / (d / S).max())
            r = np.array(r)
            print(f"    {opt['draws']} draws of S, u in [-{U:g}, {U:g}]: raw ratio at most {r.max():.3e}, at or above 1 in "
                  f"{int((r >= 1).sum())}", flush=True)
    return 0


def verdict(argv):
    recs = []
    for p in argv[:4]:
        with open(p if p.endswith(".json") else p + ".json") as f:
            recs.append(json.loads(f.readline()))
    a, b, c, d = recs
    for r in recs:      # a record written before --S has S_seed, -1 for none
        r.setdefault("S", "none" if r.get("S_seed", -1) < 0 else f"random:{r['S_seed']}")
    for name, r, want in (("A", a, ("raw", "none")), ("B", b, ("eq", "none")), ("C", c, ("eq", "S")), ("D", d, ("raw", "S"))):
        print(f"{name}: test {r['test']}, S {r['S']}, lams {_lams(r)[0]}, converged {r['converged']}, max|F| {r['F']:.2e}, "
              f"Delta {r['Delta']}")
        if r["test"] != want[0] or (want[1] == "none") != (r["S"] == "none"):
            raise SystemExit(f"{name} is not the run verdict expects ({want})")
    if not (c["S"] == d["S"] and a["state"] == b["state"] == c["state"] == d["state"]
            and len({(r["rel"], r["seed"]) for r in recs}) == 1):
        raise SystemExit("the four runs do not share the start and S")
    shows = _lams(d) != _lams(a)
    la, ld = _lams(a)[0], _lams(d)[0]
    k = next((i for i, (x, y) in enumerate(zip(la, ld)) if x != y), min(len(la), len(ld)))
    print(f"control (D raw under S against A raw): {'changed, the plant shows' if shows else 'unchanged'}"
          + (f"; first at step {k}: lam {la[k] if k < len(la) else '-'} against {ld[k] if k < len(ld) else '-'}"
             + (f" (D ran {d['it_max']} steps)" if d["it_max"] < a["it_max"] else "") if shows else ""))
    if not shows:
        print("the plant does not show: nothing tested")
        return 2
    ok = _lams(c) == _lams(b)
    zb, _ = load_o(argv[1] + ".npz")
    zc, _ = load_o(argv[2] + ".npz")
    bits = all(np.array_equal(p, q) for p, q in zip(zb, zc))
    with mp.workdps(60):
        dD = abs(mp.mpf(b["Delta"]) - mp.mpf(c["Delta"]))
    print(f"check (C eq under S against B eq): lam sequences {'equal' if ok else 'DIFFER'}; iterates bit-identical {bits}, "
          f"|Delta_B - Delta_C| {mp.nstr(dD, 5)}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def same(argv):
    x, y = argv[0], argv[1]
    opt = _opts(argv[2:], dict(mutate_bit=0))
    zx, _ = load_o(x + ".npz")
    zy, _ = load_o(y + ".npz")
    zy = [np.array(p) for p in zy]
    if opt["mutate_bit"]:
        i = int(np.argmax(np.abs(zy[7])))
        zy[7][i] = np.nextafter(zy[7][i], np.inf)
        print(f"control: entry {i} of the eighth part moved by one ulp")
    with open(x + ".json") as f:
        hx = json.loads(f.readline())["hist"]
    with open(y + ".json") as f:
        hy = json.loads(f.readline())["hist"]

    def strip(h):
        return [{k: v for k, v in r.items() if k != "seconds"} for r in h]
    zeq = all(np.array_equal(p, q) for p, q in zip(zx, zy))
    heq = json.dumps(strip(hx)) == json.dumps(strip(hy))
    print(f"{x} against {y}: iterates bit-identical {zeq}, histories (without times) equal {heq}")
    print("same" if zeq and heq else "DIFFERENT")
    return 0 if zeq and heq else 1


def polish(argv):
    opt = dict(Dc_ref="", Dc_out="", hist_out="")
    rest = list(argv)
    for k in list(opt):
        if "--" + k in rest:
            i = rest.index("--" + k)
            opt[k] = rest[i + 1]
            del rest[i:i + 2]
    import anm_run
    state = {}

    def nod(s, z, **kw):
        if "dc" not in state:
            t0 = time.time()
            A = np.abs(np.asarray(s.jacobian_o(list(o8(z)))[0]))
            dr, dc, its, err = eq_scaling(A)
            state["dc"] = dc
            print(f"#504: D_c from |J| at the corrector's start ({RESULT}, {its} Ruiz iterations, largest |1 - max-norm| "
                  f"{err:.1e}, {time.time() - t0:.0f} s): D_c {dc.min():.2e}..{dc.max():.2e}, D_r {dr.min():.2e}..{dr.max():.2e}",
                  flush=True)
            if opt["Dc_out"]:
                np.save(opt["Dc_out"], dc)
            if opt["Dc_ref"]:
                with open(opt["Dc_ref"]) as f:
                    ref = np.asarray(json.load(f)["D_c"])
                r = np.log10(dc / ref)
                print(f"#504: log10(D_c / D_c of {os.path.basename(opt['Dc_ref'])}) {r.min():+.3e}..{r.max():+.3e} (spread "
                      f"{np.ptp(r):.1e})", flush=True)
        kw["verbose"] = 2
        trace = []
        out = newton_od(s, z, scale_c=state["dc"], trace=trace, **kw)
        if opt["hist_out"]:
            # every call's history (one json line) and the trace of its trials (each call's npz)
            state["calls"] = state.get("calls", 0) + 1
            with open(opt["hist_out"] + ".jsonl", "a") as f:
                f.write(json.dumps(dict(call=state["calls"], converged=bool(out[2]), hist=out[1]), default=float) + "\n")
            L = len(state["dc"])
            np.savez_compressed(f"{opt['hist_out']}_trace{state['calls']}.npz", it=np.array([t["it"] for t in trace]),
                                lam=np.array([t["lam"] for t in trace]), accepted=np.array([t["accepted"] for t in trace]),
                                F=np.array([t["F"] for t in trace]), d=np.array([t["d"] for t in trace]).reshape(-1, L),
                                dt=np.array([t["dt"] if t["dt"] is not None else np.full(L, np.nan) for t in trace]).reshape(-1, L),
                                D_c=state["dc"])
        return out
    anm_run.newton_od = nod
    sys.argv = ["anm_run.py"] + rest
    anm_run.main()
    return 0


if __name__ == "__main__":
    cmds = dict(planted=planted, predict=predict, verdict=verdict, same=same, polish=polish)
    if len(sys.argv) < 3 or sys.argv[1] not in cmds:
        sys.exit(__doc__)
    sys.exit(cmds[sys.argv[1]](sys.argv[2:]))

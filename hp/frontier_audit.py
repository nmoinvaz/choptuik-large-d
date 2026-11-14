#!/usr/bin/env python3
"""chain_fit's error past its comparison frontier (#805): #603's states from D 245 to 259.7 against track C's series
(#801, notes/LARGE_D_WALL.md "Track C"); notes/LARGE_D_WALL.md "The chain past the comparison frontier".

    cd hp && python3 frontier_audit.py table RAWDIR TRACKC_REAL_JSON --out=TABLE.json
    cd hp && python3 frontier_audit.py jet RAWDIR --table=TABLE.json [--real=0] [--jets=253.9266,254.3086] [--out=F]
    cd hp && python3 frontier_audit.py causes --table=TABLE.json [--out=F]
    cd hp && python3 frontier_audit.py rule RAWDIR TRACKC_REAL_JSON [--out=F]
    cd hp && python3 frontier_audit.py held RAWDIR [--out=F]
    cd hp && python3 frontier_audit.py envelope RAWDIR TRACKC_REAL_JSON --table=TABLE.json [--out=F]
    cd hp && python3 frontier_audit.py planted RAWDIR TRACKC_REAL_JSON --table=TABLE.json [--out=F]

RAWDIR is a clone of the raw-data repository (I603-win2/ at 1a3c1d04 or later); TRACKC_REAL_JSON is I801-mac/real.json.

table: per state from 240 to 255.8, chain - series, a local cubic residual of Delta in 1/(D - 2) through its six
nearest neighbours, the accepted attempt's octo-double Newton steps and max|F| and first Delta correction (logs), the
step's a, refusals and move (records), and the state's LU pivots and series' last norm (the record of the step from it).
jet (test J, pre-registered on #805): the stored series of chain_nsr's bases continued to the states below them, at 200
digits; the states are one analytic function where |state - J| <= 10 T + 1e-30, T the last three terms' largest. The
plants W (an entire oscillating term growing as 1/p_min) and S (a square-root pair at 258.6 +- 2.79i) must pass and N
(W plus per-state errors c (refused + 0.2)/p_min) must fail, and each jet must return its next prediction to 1e-60;
exit 1 otherwise. causes (test 2): Spearman rho of |local residual| p_min against per-state quantities, a permutation p;
its plants fail on this grid (every one names a: a smooth term's local residual scales as the span^4), exit 1.
rule (test 3): err = max(fits extrapolated, K/p_min), K = 2 max |chain - series| p_min over 245.6-253.53, tested on
the states from 253.64 to 259.74; exit 1 when one is not covered. held: data/delta_fits.csv's degree differences
against each chain_fit state's p_min. envelope (post hoc): log|v| on log(1/p_min) and the range of |v| p_min's windowed
maximum, for the chain and the plants. planted (exploratory, superseded by jet): candidates ranked by log-log fits.
"""
import glob
import hashlib
import json
import math
import os
import random
import re
import sys

import mpmath as mp
import numpy as np

mp.mp.dps = 50
args = dict(a.lstrip("-").split("=", 1) if "=" in a else (a.lstrip("-"), "1") for a in sys.argv[2:] if a.startswith("--"))
pos = [a for a in sys.argv[1:] if not a.startswith("--")]
LO, HI = 240.0, 256.0


FOLDERS = ("chain", "chain_nat", "chain_ns", "chain_nsr")   # #603's runs in order: chain to 253.73, then the restarts


def states(raw):
    """{D: dict(Delta, plan, folder)} from #603's four runs"""
    out = {}
    for d in FOLDERS:
        for p in glob.glob(os.path.join(raw, "I603-win2", d, "anm_pin3_D*.npz")):
            z = np.load(p)
            m = json.loads(str(z["meta"]))
            parts = [z[k][-1] for k in ("zh", "zl", "z2", "z3", "z4", "z5", "z6", "z7") if k in z.files]
            out[m["D"]] = dict(Delta=mp.fsum(mp.mpf(float(x)) for x in parts), folder=d,
                               plan=hashlib.md5(json.dumps(m.get("segs")).encode()).hexdigest()[:8])
    return out


def log_info(raw):
    """{D (rounded 1e-6): dict(od_steps, od_F, first_dDelta)} for each accepted attempt, from the run logs"""
    info = {}
    for lg in [d + ".log" for d in FOLDERS]:
        path = os.path.join(raw, "I603-win2", lg)
        if not os.path.exists(path):
            continue
        lines = open(path, errors="replace").read().splitlines()
        first, od = None, None
        for ln in lines:
            m0 = re.match(r"\s+on 0: \|F\| \S+ -> \S+, \|dz\| \S+ -> \S+, lam \S+, dDelta ([-+0-9.e]+)", ln)
            if m0:
                first = float(m0.group(1))
            m1 = re.search(r"octo-double Newton: (\d+) steps, max\|F\| ([0-9.e+-]+), (converged|not converged)", ln)
            if m1:
                od = (int(m1.group(1)), float(m1.group(2)), m1.group(3) == "converged")
            m2 = re.search(r"-> D ([0-9.]+): predicted .*?(stall accepted|\(tol [^)]*\))", ln)
            if m2 and od is not None:
                D = round(float(m2.group(1)), 6)
                if od[2]:
                    info[D] = dict(od_steps=od[0], od_F=od[1], first_dDelta=first)
                first, od = None, None
    return info


def record_info(raw):
    """{D (rounded 1e-6): dict(a, refused, moved, min_pivot, max_pivot, zN)}: the step that reached D (its accepted
    attempt's a, the refused attempts before it, and the corrector's move from the prediction), and the state's own
    LU pivots and series' last norm, which the record of the step from D (its base, D0) carries"""
    step, base = {}, {}
    for d in FOLDERS:
        for line in open(os.path.join(raw, "I603-win2", d, "anm_run.jsonl")):
            if not line.strip():
                continue
            r = json.loads(line)
            k0 = round(r["D0"], 6)
            if k0 not in base and r.get("pivots"):
                base[k0] = dict(min_pivot=r["pivots"][0], max_pivot=r["pivots"][1],
                                zN=r["norms"][-1] if r.get("norms") else None)
            att = r.get("attempts", [])
            acc = [x for x in att if x.get("accepted")]
            if acc:
                a = acc[-1]
                step[round(a["D"], 6)] = dict(a=a["a"], refused=len(att) - 1, moved=a.get("moved"))
    out = {}
    for k in set(step) | set(base):
        out[k] = dict(a=None, refused=None, moved=None, min_pivot=None, max_pivot=None, zN=None)
        out[k].update(step.get(k, {}))
        out[k].update(base.get(k, {}))
    return out


def local_residual(Ds, vals, i, k=6):
    """value at i minus the cubic in 1/(D - 2) through its k nearest neighbours (i itself left out)"""
    order = sorted((j for j in range(len(Ds)) if j != i), key=lambda j: abs(Ds[j] - Ds[i]))[:k]
    x = [1 / (mp.mpf(Ds[j]) - 2) for j in order]
    y = [vals[j] for j in order]
    x0 = 1 / (mp.mpf(Ds[i]) - 2)
    xm = sum(x) / len(x)
    A = mp.matrix([[(xx - xm) ** p for p in range(4)] for xx in x])
    c = mp.qr_solve(A, mp.matrix(y))[0]
    return vals[i] - sum(c[p] * (x0 - xm) ** p for p in range(4))


def series_at(real_json):
    """track C's series at each state's D: the rows of #801's real.json 'beyond' give chain - series"""
    d = json.load(open(real_json))
    return {round(D, 6): (diff, E) for D, diff, E in d["beyond"]}


def frontier_err(D):
    """chain_fit's 'fits extrapolated' err at D: 4.0e-11 at 232.13 to 4.4e-11 at 245.64, log-linear (#801's reading of
    delta_all.csv); past 245.64 the same law continued"""
    return 4.0e-11 * (4.4e-11 / 4.0e-11) ** ((D - 232.131) / (245.637 - 232.131))


def cmd_table(raw, real_json):
    st = states(raw)
    li = log_info(raw)
    ri = record_info(raw)
    cs = series_at(real_json)
    Ds = sorted(D for D in st if LO <= D <= HI)
    vals = [st[D]["Delta"] for D in Ds]
    rows = []
    for i, D in enumerate(Ds):
        k = round(D, 6)
        lr = float(local_residual(Ds, vals, i))
        L, R = li.get(k, {}), ri.get(k, {})
        cmS = cs.get(k, (None, None))[0]
        rows.append(dict(D=D, folder=st[D]["folder"], plan=st[D]["plan"], chain_minus_series=cmS,
                         err=frontier_err(D), local_residual=lr, od_steps=L.get("od_steps"), od_F=L.get("od_F"),
                         first_dDelta=L.get("first_dDelta"), a=R.get("a"), refused=R.get("refused"),
                         moved=R.get("moved"), min_pivot=R.get("min_pivot"), max_pivot=R.get("max_pivot"),
                         zN=R.get("zN")))
    print("   D        chain-series    err      local res  od steps max|F|   first dDelta  a      refused min piv   |z_N|"
          "     moved    |dep| p_min")
    for r in rows:
        f = lambda v, fmt: (fmt % v) if v is not None else "-"
        dp = r["chain_minus_series"] * r["min_pivot"] if r["chain_minus_series"] is not None and r["min_pivot"] else None
        print("%9.4f %12s %9.1e %11s  %2s %8s %12s %6s %3s %9s %9s %9s %10s" % (
            r["D"], f(r["chain_minus_series"], "%+.3e"), r["err"], "%+.2e" % r["local_residual"],
            f(r["od_steps"], "%d"), f(r["od_F"], "%.0e"), f(r["first_dDelta"], "%+.1e"), f(r["a"], "%.3f"),
            f(r["refused"], "%d"), f(r["min_pivot"], "%.1e"), f(r["zN"], "%.1e"), f(r["moved"], "%.1e"),
            f(dp, "%+.2e")))
    if "out" in args:
        json.dump(rows, open(args["out"], "w"), indent=1)
    return rows


CANDS = ("1/p_min", "|z_N|", "exp(D)", "od_F", "od_steps", "refused", "1/a")


def cand_value(r, c):
    return {"1/p_min": 1 / r["min_pivot"], "|z_N|": r["zN"], "exp(D)": math.exp(r["D"] - 245), "od_F": r["od_F"],
            "od_steps": r["od_steps"], "refused": r["refused"] + 1, "1/a": 1 / r["a"]}[c]


def analyse(rows, dep_key="chain_minus_series", lo=245.6, hi=253.75, quiet=False):
    """(1) each candidate: least squares of log|dep| on log(candidate) over lo..hi, its slope and rms (the smaller the
    rms, the better the candidate tracks the departure); (2) smoothness: slope of log(|local residual|/|dep|) on log h,
    h the span of the six neighbours (4 for a smooth departure, 0 for per-state noise)"""
    R = [r for r in rows if lo <= r["D"] <= hi and r.get(dep_key) is not None and r[dep_key] != 0]
    y = np.log(np.array([abs(r[dep_key]) for r in R]))
    out = {}
    for c in CANDS:
        x = np.log(np.array([float(cand_value(r, c)) for r in R]))
        if np.ptp(x) == 0:
            out[c] = dict(slope=float("nan"), rms=float(np.std(y)))
            continue
        A = np.stack([x, np.ones_like(x)], 1)
        cf, *_ = np.linalg.lstsq(A, y, rcond=None)
        out[c] = dict(slope=float(cf[0]), rms=float(np.sqrt(np.mean((A @ cf - y) ** 2))))
    h = np.log(np.array([r["span"] for r in R]))
    q = np.log(np.array([max(abs(r["local_residual"]), 1e-300) / abs(r[dep_key]) for r in R]))
    A = np.stack([h, np.ones_like(h)], 1)
    cf, *_ = np.linalg.lstsq(A, q, rcond=None)
    out["smooth_slope"] = float(cf[0])
    best = min(CANDS, key=lambda c: out[c]["rms"])
    out["best"] = best
    if not quiet:
        print("  %d states, D %.2f..%.2f; log|dep| against log(candidate): " % (len(R), lo, hi) + "; ".join(
            "%s slope %.2f rms %.2f" % (c, out[c]["slope"], out[c]["rms"]) for c in CANDS))
        print("  best tracker: %s; smoothness: log(|local residual|/|dep|) against log(span) slope %.2f "
              "(4: smooth, 0: per-state noise)" % (best, out["smooth_slope"]))
    return out


def add_local(rows, key):
    """local residual of key (per state, the cubic through six neighbours in 1/(D-2)) and the neighbours' span"""
    Ds = [r["D"] for r in rows]
    vals = [mp.mpf(r[key]) for r in rows]
    for i, r in enumerate(rows):
        order = sorted((j for j in range(len(Ds)) if j != i), key=lambda j: abs(Ds[j] - Ds[i]))[:6]
        r["span"] = max(abs(Ds[j] - Ds[i]) for j in order)
        r["local_residual"] = float(local_residual(Ds, vals, i))
    return rows


def cmd_planted(raw, real_json):
    rows = [r for r in json.load(open(args["table"])) if r["chain_minus_series"] is not None]
    for r in rows:
        r["dep_real"] = r["chain_minus_series"]
    add_local(rows, "dep_real")
    print("real departure (chain - series):")
    real = analyse(rows, "dep_real")
    rng = random.Random(805)
    # K from the real states: |dep| p_min over 245.6..253.75 (median), for the plants' envelopes
    K = float(np.median([abs(r["dep_real"]) * r["min_pivot"] for r in rows if 245.6 <= r["D"] <= 253.75]))
    plants = {
        "S (smooth, envelope K/p_min, oscillating)": lambda r: K / r["min_pivot"] * math.cos(2.3 * (r["D"] - 246.9)),
        "N (per-state noise scaled by the refusals)": lambda r: K / 1.1e-29 * (r["refused"] + 0.2) * rng.choice((-1, 1))
                                                              * math.exp((r["D"] - 249) / 1.5),
        "H (smooth drift, no weak-direction amplification)": lambda r: 3e-11 * ((r["D"] - 240) / 5) ** 5,
    }
    out = {"real": real, "K": K}
    for name, f in plants.items():
        for r in rows:
            r["dep_p"] = f(r) + rng.gauss(0, 1e-12)
        add_local(rows, "dep_p")
        print("planted %s:" % name)
        out[name] = analyse(rows, "dep_p")
    if "out" in args:
        json.dump(out, open(args["out"], "w"), indent=1)


# ---- test J: the stored jets' analytic continuation against the states (pre-registered on #805) ----

PARTS = ("zh", "zl", "z2", "z3", "z4", "z5", "z6", "z7")


def load_state(path):
    """(D as mpf from D_str, Delta as the exact sum of the stored parts) of a state or prediction file"""
    z = np.load(path)
    m = json.loads(str(z["meta"]))
    return mp.mpf(m["D_str"]), mp.fsum(mp.mpf(float(z[k][-1])) for k in PARTS if k in z.files)


def load_jet(path):
    """(D_b, [delta_n]): Delta's Taylor coefficients in a = D - D_b, the exact sum of the jet's eight parts"""
    z = np.load(path)
    m = json.loads(str(z["meta"]))
    ks = sorted(k for k in z.files if re.fullmatch(r"z\d", k))
    return mp.mpf(m["D_str"]), [mp.fsum(mp.mpf(float(z[k][n][-1])) for k in ks) for n in range(z[ks[0]].shape[0])]


def jet_eval(Db, d, D):
    a = D - Db
    return mp.fsum(c * a ** n for n, c in enumerate(d)), max(abs(d[n]) * abs(a) ** n for n in range(len(d) - 3, len(d)))


def j_test(Db, d, pts, T_max=1e-12, label=""):
    """pts: [(D, Delta)]; each with D < D_b and T <= T_max is compared: pass when |d| <= 10 T + 1e-30 at all"""
    rows, ok = [], True
    for D, v in sorted(pts):
        if D >= Db:
            continue
        J, T = jet_eval(Db, d, D)
        if T > T_max:
            continue
        dd = v - J
        good = abs(dd) <= 10 * T + mp.mpf("1e-30")
        ok &= bool(good)
        rows.append(dict(D=float(D), d=float(dd), T=float(T), ratio=float(abs(dd) / (10 * T + mp.mpf("1e-30")))))
    worst = max(rows, key=lambda r: r["ratio"]) if rows else None
    print("  %s: %d states from D %.4f to %.4f within reach (T <= %.0e); worst |d|/(10T + 1e-30) = %.3g at D %.4f "
          "(d %+.2e, T %.1e): %s" % (label, len(rows), rows[0]["D"] if rows else 0, rows[-1]["D"] if rows else 0,
                                    T_max, worst["ratio"], worst["D"], worst["d"], worst["T"],
                                    "one analytic function" if ok else "NOT one analytic function"))
    return ok, rows


def plant_funcs(tab):
    """the plants' truths as (value(D), jet(D_b, N)) in mp; W's K, b, omega and phi from the table's envelope"""
    R = [r for r in tab if 245.6 <= r["D"] <= 255.8 and r["min_pivot"]]
    x = np.array([r["D"] for r in R])
    b = -float(np.polyfit(x, np.log([r["min_pivot"] for r in R]), 1)[0])     # p_min ~ e^{-b D}
    p250 = math.exp(float(np.polyval(np.polyfit(x, np.log([r["min_pivot"] for r in R]), 1), 250.0)))
    K = mp.mpf(2e-37) / p250
    kap = mp.mpc(b, 2 * mp.pi / 5.3)
    ph = mp.pi / 2 - kap.imag * mp.mpf("0.3")       # cos(omega (D - 250) + phi) = 0 at D 250.3
    W = lambda D: (K * mp.exp(1j * ph) * mp.exp(kap * (D - 250))).real   # noqa: E731
    Wj = lambda Db, N: [(K * mp.exp(1j * ph) * mp.exp(kap * (Db - 250)) * kap ** n / mp.factorial(n)).real  # noqa: E731
                        for n in range(N + 1)]
    Dc, c = mp.mpc("258.6", "2.79"), mp.mpf("1e-6") * mp.exp(1j * mp.mpf("0.7"))
    Sf = lambda D: 2 * (c * mp.sqrt(Dc - D)).real     # noqa: E731

    def Sj(Db, N):
        w = Dc - Db
        return [2 * (c * mp.sqrt(w) * mp.binomial(mp.mpf(1) / 2, n) * (-1 / w) ** n).real for n in range(N + 1)]
    return dict(W=(W, Wj), S=(Sf, Sj), b=b, K=K, p250=p250)


def cmd_jet(raw):
    mp.mp.dps = 200          # the jets' sums and the plants at 200 digits, as registered (50 rounds Delta at 1e-51)
    tab = json.load(open(args["table"]))
    W = os.path.join(raw, "I603-win2")
    jets = [os.path.join(W, "chain_nsr", "jet_pin3_D%s.npz" % k) for k in args.get("jets", "253.9266,254.3086").split(",")]
    out = {"plants": {}, "real": {}}
    pf = plant_funcs(tab)
    rng = random.Random(805)
    st = {r["D"]: r for r in tab}
    # the grid: every stored state (any folder), exact D
    pts = []
    for d in FOLDERS:
        for f in glob.glob(os.path.join(W, d, "anm_pin3_D*.npz")):
            D, v = load_state(f)
            if LO <= D <= 255.8:
                pts.append((D, v))
    print("plants (b = %.4f from log p_min, K = %.2e, p_min(250) = %.2e):" % (pf["b"], float(pf["K"]), pf["p250"]))
    verdict = True
    # N's errors: c (refused + 0.2)/p_min with random signs, c set so that their median magnitude over the states the
    # first jet reaches equals the real local residuals' median there
    Db0, d0 = load_jet(jets[0])
    reach = [D for D, _ in pts if D < Db0 and jet_eval(Db0, d0, D)[1] <= 1e-12]
    keyed = {round(float(D), 6): st.get(min(st, key=lambda x: abs(x - float(D)))) for D, _ in pts}
    lr_med = float(np.median([abs(keyed[round(float(D), 6)]["local_residual"]) for D in reach]))
    raw_eps = {round(float(D), 6): ((keyed[round(float(D), 6)]["refused"] or 0) + 0.2) / keyed[round(float(D), 6)]["min_pivot"]
               * rng.choice((-1, 1)) for D, _ in pts}
    cN = lr_med / float(np.median([abs(raw_eps[round(float(D), 6)]) for D in reach]))
    for name, expect in (("W", True), ("S", True), ("N", False)):
        f, fj = pf["S"] if name == "S" else pf["W"]
        for jp in jets:
            Db, d = load_jet(jp)
            N = len(d) - 1
            jet_p = fj(Db, N)
            P = [(D, f(D) + (cN * raw_eps[round(float(D), 6)] if name == "N" else 0)) for D, _ in pts]
            if name == "N":
                jet_p[0] += cN * raw_eps[round(float(Db), 6)]
            ok, rows = j_test(Db, jet_p, P, label="plant %s, jet at %s" % (name, mp.nstr(Db, 10)))
            good = ok == expect
            verdict &= good
            print("    expected %s: %s" % ("pass" if expect else "fail", "as expected" if good else "NOT AS EXPECTED"))
            out["plants"]["%s@%s" % (name, mp.nstr(Db, 10))] = dict(ok=ok, expected=expect, rows=rows)
    if "real" in args and args["real"] == "0":
        json.dump(out, open(args["out"], "w"), indent=1) if "out" in args else None
        sys.exit(0 if verdict else 1)
    print("real chain:")
    for jp in jets:
        Db, d = load_jet(jp)
        # control: the jet returns its own next prediction
        nxt = min((D for D, _ in pts if D > Db), default=None)
        pf_ = glob.glob(os.path.join(W, "chain_nsr", "pred_pin3_D%.4f.npz" % float(nxt)))
        Dp, vp = load_state(pf_[0])
        Jp, _ = jet_eval(Db, d, Dp)
        ctl = abs(vp - Jp) <= mp.mpf("1e-60")
        verdict &= bool(ctl)
        print("  control, jet at %s against the prediction at %s: |pred - J| = %s, %s" % (
            mp.nstr(Db, 10), mp.nstr(Dp, 10), mp.nstr(abs(vp - Jp), 3), "passes" if ctl else "FAILS"))
        ok, rows = j_test(Db, d, pts, label="real, jet at %s" % mp.nstr(Db, 10))
        for r in rows:
            k = min(st, key=lambda x: abs(x - r["D"]))
            r["local_residual"] = st[k]["local_residual"] if abs(k - r["D"]) < 1e-6 else None
        print("    D         d=state-J    T        |local residual|")
        for r in rows:
            print("    %9.4f %+.2e %.1e %s" % (r["D"], r["d"], r["T"], "%.2e" % abs(r["local_residual"])
                                             if r["local_residual"] is not None else "-"))
        out["real"][mp.nstr(Db, 10)] = dict(ok=ok, control=bool(ctl), rows=rows)
    # each step's Delta correction from the stored files: Delta_state - Delta(pred)
    corr = []
    for d in FOLDERS:
        for f in glob.glob(os.path.join(W, d, "pred_pin3_D*.npz")):
            sf = f.replace("pred_pin3", "anm_pin3")
            if os.path.exists(sf):
                Dp, vp = load_state(f)
                Ds, vs = load_state(sf)
                if abs(Dp - Ds) < mp.mpf("1e-30"):
                    corr.append((float(Ds), float(vs - vp)))
    corr.sort()
    c2 = [abs(c) for D, c in corr if 240 <= D]
    print("each step's Delta correction from the files, D 240-259.74: %d steps, max |Delta_state - Delta_pred| = %.1e, "
          "median %.1e" % (len(c2), max(c2), float(np.median(c2))))
    out["corrections"] = corr
    if "out" in args:
        json.dump(out, open(args["out"], "w"), indent=1)
    sys.exit(0 if verdict else 1)


# ---- test 2: per-state causes, Spearman with a permutation p (pre-registered on #805) ----

PER_STATE = ("refused", "od_steps", "a", "log od_F")


def spearman_perm(q, c, n=10000, seed=805):
    from scipy.stats import spearmanr
    rho = spearmanr(q, c)[0]
    rng = np.random.default_rng(seed)
    hits = sum(abs(spearmanr(rng.permutation(q), c)[0]) >= abs(rho) for _ in range(n))
    return float(rho), (hits + 1) / (n + 1)


def causes(rows, label):
    q = [abs(r["local_residual"]) * r["min_pivot"] for r in rows]
    named, res = [], {}
    for c in PER_STATE:
        v = [math.log(r["od_F"]) if c == "log od_F" else r[c] for r in rows]
        rho, pv = spearman_perm(q, v)
        res[c] = dict(rho=rho, p=pv)
        if pv < 0.01:
            named.append(c)
    print("  %s: %d states; " % (label, len(rows)) + "; ".join("%s rho %+.2f p %.4f" % (c, res[c]["rho"], res[c]["p"])
                                                            for c in PER_STATE) + "; named: %s" % (", ".join(named) or "none"))
    return named, res


def cmd_causes():
    tab = [r for r in json.load(open(args["table"])) if LO <= r["D"] <= 255.8 and r["min_pivot"] and r["od_F"]
           and r["refused"] is not None]
    pf = plant_funcs(tab)
    rng = random.Random(805)
    lr_med = float(np.median([abs(r["local_residual"]) for r in tab]))
    eps = [((r["refused"]) + 0.2) / r["min_pivot"] * rng.choice((-1, 1)) for r in tab]
    cN = lr_med / float(np.median(np.abs(eps)))
    out, verdict = {}, True
    print("plants:")
    for name, f, expect in (("W", pf["W"][0], []), ("S", pf["S"][0], []),
                            ("N", lambda D, i: pf["W"][0](D) + cN * eps[i], ["refused"])):
        rows = [dict(r) for r in tab]
        for i, r in enumerate(rows):
            r["v"] = f(mp.mpf(r["D"]), i) if name == "N" else f(mp.mpf(r["D"]))
        add_local(rows, "v")
        named, res = causes(rows, "plant " + name)
        good = named == expect
        verdict &= good
        print("    expected %s: %s" % (", ".join(expect) or "none", "as expected" if good else "NOT AS EXPECTED"))
        out[name] = dict(named=named, res=res, expected=expect)
    print("real chain (local residuals of Delta, table.json):")
    named, res = causes(tab, "real")
    out["real"] = dict(named=named, res=res)
    if "out" in args:
        json.dump(out, open(args["out"], "w"), indent=1)
    sys.exit(0 if verdict else 1)


# ---- test 3: rule R out of sample (pre-registered on #805) ----

def cmd_rule(raw, real_json):
    ri = record_info(raw)
    cs = series_at(real_json)
    rows = []
    for k, (dep, E) in sorted(cs.items()):
        r = ri.get(k)
        if r is None or not r["min_pivot"]:
            print("  no pivot for D %.6f" % k)
            continue
        rows.append(dict(D=k, dep=dep, E=E, p_min=r["min_pivot"]))
    cal = [r for r in rows if 245.6 <= r["D"] <= 253.53]
    test = [r for r in rows if 253.64 <= r["D"] <= 259.74]
    Kc = max(abs(r["dep"]) * r["p_min"] for r in cal)
    K = 2 * Kc
    print("rule R: err = max(fits extrapolated, K/p_min), K = 2 x %.3e = %.3e from %d states, D %.4f to %.4f" % (
        Kc, K, len(cal), cal[0]["D"], cal[-1]["D"]))
    fails = []
    print("   D          chain-series    K/p_min    err       |dep|/err")
    for r in cal + test:
        r["err"] = max(frontier_err(r["D"]), K / r["p_min"])
        r["ratio"] = abs(r["dep"]) / r["err"]
        r["set"] = "cal" if r in cal else "test"
        if r["set"] == "test" and r["ratio"] > 1:
            fails.append(r)
        print("  %9.4f %s %+.3e %9.2e %9.2e %8.3f%s" % (r["D"], r["set"], r["dep"], K / r["p_min"], r["err"], r["ratio"],
                                                       "  NOT COVERED" if r["ratio"] > 1 else ""))
    mt = max(test, key=lambda r: r["ratio"])
    print("out of sample: %d states, D %.4f to %.4f; largest |dep|/err %.3f at D %.4f; %s" % (
        len(test), test[0]["D"], test[-1]["D"], mt["ratio"], mt["D"],
        "R covers every state" if not fails else "R fails at %d states, from D %.4f" % (len(fails), fails[0]["D"])))
    if "out" in args:
        json.dump(dict(K=K, Kc=Kc, rows=rows, fails=[r["D"] for r in fails]), open(args["out"], "w"), indent=1)
    sys.exit(0 if not fails else 1)


# ---- the held fit's degree response (data/delta_fits.csv) against each chain_fit state's p_min ----

def cmd_held(raw):
    import csv
    here = os.path.dirname(os.path.abspath(__file__))
    fits = {round(float(r["D"]), 5): r for r in csv.DictReader(open(os.path.join(here, "data", "delta_fits.csv")))}
    rows = []
    cache = {}
    for r in csv.DictReader(open(os.path.join(here, "data", "delta_all.csv"))):
        D = float(r["D"])
        if r["kind"] != "chain_fit" or D < 150 or round(D, 5) not in fits:
            continue
        src = os.path.join(raw, r["source"])
        if src not in cache:
            cache[src] = {round(json.loads(x)["D0"], 4): json.loads(x).get("pivots") for x in open(src) if x.strip()}
        pv = cache[src].get(round(D, 4))
        f = fits[round(D, 5)]
        rows.append(dict(D=D, d6=float(f["diff_d6"]) if f["diff_d6"] else None,
                         d10=float(f["diff_d10"]) if f["diff_d10"] else None, p_min=pv[0] if pv else None))
    print("   D        d6 - d8     d10 - d8    p_min      |d6| p_min")
    for r in rows[::3] + [rows[-1]]:
        print("  %8.3f %+.3e %s %9s %10s" % (r["D"], r["d6"], "%+.3e" % r["d10"] if r["d10"] is not None else "-",
                                            "%.1e" % r["p_min"] if r["p_min"] else "-",
                                            "%.1e" % (abs(r["d6"]) * r["p_min"]) if r["p_min"] else "-"))
    if "out" in args:
        json.dump(rows, open(args["out"], "w"), indent=1)


# ---- post hoc (not pre-registered): the 1/p_min envelope on the plants and the real chain ----

def env_stats(D, v, pm, win=3.0):
    """slope of log|v| on log(1/p_min), and the range (largest over smallest) of the windowed maximum of |v| p_min
    over windows of win in D (the windows skip the zeros of an oscillating v)"""
    D, v, pm = np.array(D), np.abs(np.array(v, dtype=float)), np.array(pm)
    sl = float(np.polyfit(np.log(1 / pm), np.log(v), 1)[0])
    wm = [max(v[(D >= d0) & (D < d0 + win)] * pm[(D >= d0) & (D < d0 + win)]) for d0 in D if d0 + win <= D[-1]]
    return sl, float(max(wm) / min(wm))


def cmd_envelope(raw, real_json):
    tab = [r for r in json.load(open(args["table"])) if LO <= r["D"] <= 255.8 and r["min_pivot"]]
    ri, cs = record_info(raw), series_at(real_json)
    G = [(k, dep, ri[k]["min_pivot"]) for k, (dep, E) in sorted(cs.items()) if k in ri and ri[k]["min_pivot"]]
    pf = plant_funcs(tab)
    rng = random.Random(805)
    lr_med = float(np.median([abs(r["local_residual"]) for r in tab]))
    eps = {round(r["D"], 6): (r["refused"] + 0.2) / r["min_pivot"] * rng.choice((-1, 1)) for r in tab}
    cN = lr_med / float(np.median(np.abs(list(eps.values()))))
    out = {}
    for name in ("real", "W", "S", "N"):
        if name == "real":
            v = [g[1] for g in G]
            lr = [(r["local_residual"], r["chain_minus_series"]) for r in tab if 248 <= r["D"] <= 253.0
                  and r["chain_minus_series"]]
        else:
            f = pf["S"][0] if name == "S" else pf["W"][0]
            v = [float(f(mp.mpf(g[0]))) + (cN * eps.get(round(g[0], 6), 0) if name == "N" else 0) for g in G]
            rows = [dict(r) for r in tab]
            for r in rows:
                r["v"] = f(mp.mpf(r["D"])) + (cN * eps[round(r["D"], 6)] if name == "N" else 0)
            add_local(rows, "v")
            lr = [(r["local_residual"], float(r["v"])) for r in rows if 248 <= r["D"] <= 253.0]
        sl, rg = env_stats([g[0] for g in G], v, [g[2] for g in G])
        q = float(np.median([abs(a_) / abs(b_) for a_, b_ in lr]))
        out[name] = dict(slope=sl, range=rg, lr_over_v=q)
        print("  %-4s D %.2f-%.2f (%d states): log|v| on log(1/p_min) slope %.3f; windowed max of |v| p_min ranges over "
              "a factor %.3g; median |local residual|/|v| on 248-253: %.3f" % (name, G[0][0], G[-1][0], len(G), sl, rg, q))
    if "out" in args:
        json.dump(out, open(args["out"], "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "jet":
        cmd_jet(pos[1])
    if cmd == "envelope":
        cmd_envelope(pos[1], pos[2])
        sys.exit(0)
    if cmd == "causes":
        cmd_causes()
    if cmd == "rule":
        cmd_rule(pos[1], pos[2])
    if cmd == "held":
        cmd_held(pos[1])
        sys.exit(0)
    if cmd == "planted":
        cmd_planted(pos[1], pos[2])
        sys.exit(0)
    if cmd == "table":
        cmd_table(pos[1], pos[2])
        sys.exit(0)
    raise SystemExit(__doc__)

#!/usr/bin/env python3
"""The held harmonics of the large-D chain (anm_run.py, kfreeze 3: p1's odd harmonics 5-13) from a fit of the
full-system states, and the bias that each way of holding them puts into Delta (notes/HELD_HARMONICS.md).

The fit: for each held coefficient v (a_k cos + b_k sin of p1 on the half grid), sqrt(n) v as a Chebyshev series of
degree --deg in x = 1/n (n = D - 2), least squares on the full-system states (every harmonic solved): the
double-double states of hp/data/states and T12-win, T13-win (D >= --Dmin), and the quad-double states of T84-mac and
T81 (68.5-73.0). The scaling is the centre's, p ~ x F(theta) / sqrt(n) with F analytic in 1/n (README, "How the
solution scales with D"). pinning.HeldFit evaluates the JSON and its Taylor coefficients in D; anm_run.py --held_fit
holds the harmonics on it.

The bias of a held value h against the full system's h_full at the same D is, to first order, s . (h - h_full), with
s = dDelta/dh the pinned system's response (one quad-double LU, `sens`). The pinned solution with h = h_full is the full
solution, so this is the whole bias to first order in h - h_full.

    cd hp && python3 held_fit.py fit RAWDIR OUT.json [--deg=8] [--Dmin=30] [--Dmax=73.01]
    cd hp && python3 held_fit.py sens RAWDIR OUT.json STATE [STATE ...] [--threads=2]
    cd hp && python3 held_fit.py planted RAWDIR SENS.json [--deg=8] [--mutate=scale5]
    cd hp && python3 held_fit.py series RAWDIR SENS.json FIT.json [--plant=0]

`sens` (kernel needed) writes s at each state with the state's own p1 coefficients. `planted` (criteria fixed before
the first run, notes/HELD_HARMONICS.md 2): with the full-system quad-double states as truth,
  P1 the fit on D <= 71.0 predicts 71.5, 72.8, 73.0: |bias| <= 1e-12 at each;
  P2 the fit on the double-double states alone (D <= 61.03) predicts 68.5-73.0, 7.5-12 units ahead: |bias| <= 1e-10;
  B1 the chain's trend (anm_run.py --trend T75 from T75's 73.27 state) at the same D, reported;
  B2 the trend's rule (power law in n per coefficient, fitted as anm_run.py fits it) on the double-double states
     58-61.03, relative to the 61.03 state, at 68.5-73.0, reported.
--mutate=scale5 multiplies every fitted value of harmonic 5 by 1 + 1e-6 (the control: P1 must fail). Exit 1 when P1 or P2
fails. `series`: at each chain state of SENS.json, Delta_chain + s . (h_fit - h_chain), against the 1/D series of
degree 20 fitted on the double-double points D <= 67.77 (chain_vs_series.py's reference); --plant adds to h_fit's
harmonic-5 cosine coefficient (the control: it must move the column by s_5a times the plant).
"""
import glob
import json
import os
import sys

import mpmath as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
M = 32
NODES = [0.3, 0.4, 0.5, 0.6, 0.7]
KS = [5, 7, 9, 11, 13]
KEYS = [f"{k}{c}" for k in KS for c in "ab"]
TH = np.pi * np.arange(M) / M


def coeffs_of(z):
    """{key: v} of p1 (the first M entries of a state's float64 sum of parts)"""
    f = z[:M]
    out = {}
    for k in KS:
        out[f"{k}a"] = float((2 / M) * np.sum(f * np.cos(k * TH)))
        out[f"{k}b"] = float((2 / M) * np.sum(f * np.sin(k * TH)))
    return out


def load(p):
    d = np.load(p)
    m = json.loads(str(d["meta"]))
    parts = [d["zh"], d["zl"]] + ([d["z2"], d["z3"]] if "z2" in d else [])
    z = parts[-1].astype(float)
    for q in parts[-2::-1]:
        z = z + q
    D = mp.mpf(m["D_str"]) if "D_str" in m else mp.mpf(repr(float(m["D"])))
    delta = sum(mp.mpf(float(q[-1])) for q in parts)
    return D, z, m, delta


def full_states(raw):
    """[(D, path, kind)] of the full-system states, M 32, nodes 0.3-0.7"""
    out = []
    for p in sorted(glob.glob(os.path.join(HERE, "data", "states", "*.npz"))):
        out.append((p, "dd"))
    for fo in ("T12-win/scratch/t12", "T13-win/scratch/t13"):
        for p in sorted(glob.glob(os.path.join(raw, fo, "*.npz"))):
            out.append((p, "dd"))
    for p in sorted(glob.glob(os.path.join(raw, "T84-mac", "*_full.npz"))):
        out.append((p, "qd"))
    for p in ("T81/conv_pin7_D72.8000_D72.800000.npz", "T81/conv_pin5_D73.0000_D73.000000.npz"):
        out.append((os.path.join(raw, p), "qd"))
    res, seen = [], set()
    for p, kind in out:
        D, z, m, delta = load(p)
        if int(m["M"]) != M or list(m["nodes"]) != NODES:
            continue
        key = (kind, mp.nstr(D, 10))
        if key in seen:          # T84's pin5 and pin7 legs converge to the same state (spread 3e-28)
            continue
        seen.add(key)
        res.append(dict(D=D, path=p, kind=kind, c=coeffs_of(z), delta=delta))
    res.sort(key=lambda r: r["D"])
    return res


def fit(pts, deg, mutate=""):
    """{key: Chebyshev coefficients} of sqrt(n) v in x = 1/n on the points' domain, and the domain"""
    n = np.array([float(r["D"]) - 2.0 for r in pts])
    x = 1.0 / n
    dom = [float(x.min()), float(x.max())]
    out, res = {}, {}
    for key in KEYS:
        y = np.sqrt(n) * np.array([r["c"][key] for r in pts])
        if mutate == "scale5" and key.startswith("5"):
            y = y * (1 + 1e-6)
        c = np.polynomial.chebyshev.Chebyshev.fit(x, y, deg, domain=dom)
        out[key] = [float(v) for v in c.coef]
        res[key] = float(np.max(np.abs(c(x) - y) / np.sqrt(n)))
    return out, dom, res


def write_fit(path, pts, deg, mutate=""):
    coeffs, dom, res = fit(pts, deg, mutate)
    d = dict(kind="held_fit", model="sqrt(n) v = Chebyshev series in x = 1/n, n = D - 2", deg=deg, domain=dom, coeffs=coeffs,
             n_points=len(pts), Dmin=float(pts[0]["D"]), Dmax=float(pts[-1]["D"]), max_residual=res,
             states=[os.path.relpath(r["path"], HERE) if r["path"].startswith(HERE) else r["path"].split("choptuik-large-d-raw/")[-1] for r in pts])
    with open(path, "w") as fh:
        json.dump(d, fh, indent=1)
    return d


def powerlaw_trend(pts_trend, base):
    """anm_run.py --trend's rule on the states pts_trend (every 4th of the folder there; here all given): per
    coefficient, |v| = C n^s when its sign is constant over >= 6 states and the fit's residual in ln|v| < 0.05, else held
    constant; relative to base: v(D) = v(D0) (n / n0)^s. Returns (fn(D) -> {key: v}, {key: s})"""
    ns = np.array([float(r["D"]) - 2.0 for r in pts_trend])
    slopes = {}
    for key in KEYS:
        v = np.array([r["c"][key] for r in pts_trend])
        slopes[key] = 0.0
        if len(v) >= 6 and (np.all(v > 0) or np.all(v < 0)):
            sl, c0 = np.polyfit(np.log(ns), np.log(np.abs(v)), 1)
            if float(np.max(np.abs(np.log(np.abs(v)) - (c0 + sl * np.log(ns))))) < 0.05:
                slopes[key] = float(sl)
    n0 = float(base["D"]) - 2.0
    return (lambda D: {key: base["c"][key] * ((float(D) - 2.0) / n0) ** slopes[key] for key in KEYS}), slopes


def bias(sens, h, h_ref):
    return sum(sens[key] * (h[key] - h_ref[key]) for key in KEYS)


def sens_at(raw, rel, threads):
    """s = dDelta/dh for each held coefficient at a state (pinned kernel, kfreeze 3; one Solver.newton_q for the LU)"""
    from hpkernel.driver import make_solver, load_q
    from pinning import band_field
    z, m = load_q(os.path.join(raw, rel))
    D = mp.mpf(m["D_str"]) if "D_str" in m else mp.mpf(repr(float(m["D"])))
    s = make_solver(D, M, nodes=list(m["nodes"]), kt=m.get("kt", 36), kc=m.get("kc", 146), ks=m.get("ks", 130), threads=threads,
                    kfreeze=3)
    s.set_segs(s.plan(np.asarray(z[0]), np.asarray(z[1]), 1e-32, 0.05))
    out = s.newton_q(*z)
    L = len(z[0])
    r0 = ((L - 1) // (4 * M) - 1) * 4 * M
    zero = np.zeros(L)
    res = {}
    for k in KS:
        for c, ab in zip("ab", [(1.0, 0.0), (0.0, 1.0)]):
            fh, fl = band_field([ab], [k], M)
            ch, cl = zero.copy(), zero.copy()
            ch[r0:r0 + M], cl[r0:r0 + M] = fh, fl
            x = s.qlu_solve(ch, cl, zero, zero)
            res[f"{k}{c}"] = float(np.asarray(x[0])[-1] + np.asarray(x[1])[-1])
    Dz, zf, _, delta = load(os.path.join(raw, rel))
    return dict(state=rel, D=str(D), sens=res, h=coeffs_of(zf), delta=str(delta), piv=[float(out[6]), float(out[7])])


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    if len(args) < 2:
        sys.exit(__doc__)
    cmd, raw = args[0], args[1]
    deg = int(opts.get("deg", 8))
    if cmd == "fit":
        pts = [r for r in full_states(raw) if float(opts.get("Dmin", 30)) <= r["D"] <= float(opts.get("Dmax", 73.01))]
        d = write_fit(args[2], pts, deg)
        print(f"{args[2]}: degree {deg}, {d['n_points']} states D {d['Dmin']:.4f}-{d['Dmax']:.4f}; max residual per coefficient "
              + ", ".join(f"{k} {v:.1e}" for k, v in d["max_residual"].items()))
        return 0
    if cmd == "sens":
        out = [sens_at(raw, rel, int(opts.get("threads", 2))) for rel in args[3:]]
        for r in out:
            print(f"{r['state']}: D {r['D']}, pivots {r['piv'][0]:.1e}..{r['piv'][1]:.1e}; s = " + ", ".join(f"{k} {v:+.4e}" for k, v in r["sens"].items()),
                  flush=True)
        prev = json.load(open(args[2])) if os.path.exists(args[2]) else []
        keep = [r for r in prev if r["state"] not in {o["state"] for o in out}]
        json.dump(keep + out, open(args[2], "w"), indent=1)
        return 0
    from pinning import HeldFit
    if cmd == "check":
        # HeldFit's Taylor coefficients in D against its values: the order-N sum at D0 + u must equal the value at D0 + u
        # to the series' truncation (u = 0.5, N = 24: (u / n)^25 ~ 1e-45 relative). --mutate=jet scales order 1 by
        # 1 + 1e-9 (the control: must fail). Exit 1 on failure.
        hf = HeldFit(args[2])
        worst = 0.0
        for D0 in ("68.5", "73.0", "79.765722", "120.0"):
            D0 = mp.mpf(D0)
            J = hf.jets(D0, 24, KS)
            if opts.get("mutate") == "jet":
                J[1] = [(a * (1 + 1e-9), b * (1 + 1e-9)) for a, b in J[1]]
            for u in (mp.mpf("0.125"), mp.mpf("0.5")):
                ex = hf.values(D0 + u, KS)
                for i in range(len(KS)):
                    for c in (0, 1):
                        ser = mp.fsum(mp.mpf(J[j][i][c]) * u ** j for j in range(25))
                        worst = max(worst, float(abs(ser - ex[i][c]) / abs(ex[i][c])))
        print(f"HeldFit {args[2]}: largest relative difference between the order-24 Taylor sum and the value: {worst:.1e}"
              " (pass below 1e-14)")
        return 0 if worst < 1e-14 else 1
    sens = json.load(open(args[2]))
    sens_by = {round(float(r["D"]), 4): r for r in sens}
    if cmd == "planted":
        mutate = opts.get("mutate", "")
        allp = full_states(raw)
        qd = [r for r in allp if r["kind"] == "qd"]
        dd = [r for r in allp if r["kind"] == "dd" and r["D"] >= float(opts.get("Dmin", 30))]

        def s_near(D):
            ks = sorted(sens_by, key=lambda d: abs(d - float(D)))
            return sens_by[ks[0]]["sens"], ks[0]
        tmp = os.path.join(os.path.dirname(os.path.abspath(args[2])), "_held_fit_planted.json")
        ok = True
        rows = []
        # P1: fit on every full state with D <= 71.0, predict the later quad-double ones
        write_fit(tmp, [r for r in dd + qd if r["D"] <= 71.0 + 1e-9], deg, mutate)
        hf = HeldFit(tmp)
        for r in [q for q in qd if q["D"] > 71.0 + 1e-9]:
            s, Ds = s_near(r["D"])
            h = dict(zip(KEYS, [v for ab in hf.values(r["D"], KS) for v in ab]))
            b = bias(s, h, r["c"])
            ok &= abs(b) <= 1e-12
            rows.append(("P1 fit D <= 71.0", r["D"], b, abs(b) <= 1e-12, Ds))
        # P2: fit on the double-double states alone
        write_fit(tmp, dd, deg, mutate)
        hf = HeldFit(tmp)
        for r in qd:
            s, Ds = s_near(r["D"])
            h = dict(zip(KEYS, [v for ab in hf.values(r["D"], KS) for v in ab]))
            b = bias(s, h, r["c"])
            ok &= abs(b) <= 1e-10
            rows.append((f"P2 fit on dd D <= {float(dd[-1]['D']):.2f}", r["D"], b, abs(b) <= 1e-10, Ds))
        # B1: the chain's trend (T75's states, T75's 73.27 base), as anm_run.py --trend T75 forms it
        t75 = []
        for p in sorted(glob.glob(os.path.join(raw, "T75", "pin*_D*.npz")))[::4]:
            Dp, zp, mp_, dl = load(p)
            if int(mp_["M"]) == M:
                t75.append(dict(D=Dp, c=coeffs_of(zp)))
        Db, zb, _, _ = load(os.path.join(raw, "T75", "pin7_D73.2657.npz"))
        fn, sl = powerlaw_trend(t75, dict(D=Db, c=coeffs_of(zb)))
        print("B1 exponents (anm_run.py's log: 5 -1.21/+0.07, 7 -1.61/-0.71, 9 -1.99/-1.30, 11 and 13 0): "
              + ", ".join(f"{k} {v:+.2f}" for k, v in sl.items()))
        for r in qd:
            s, Ds = s_near(r["D"])
            rows.append(("B1 chain's trend (T75)", r["D"], bias(s, fn(r["D"]), r["c"]), None, Ds))
        # B2: the trend's rule on the double-double states 58-61.03, from the last
        win = [r for r in dd if r["D"] >= 58.0]
        fn2, sl2 = powerlaw_trend(win, win[-1])
        for r in qd:
            s, Ds = s_near(r["D"])
            rows.append((f"B2 trend's rule on dd {float(win[0]['D']):.2f}-{float(win[-1]['D']):.2f}", r["D"], bias(s, fn2(r["D"]), r["c"]), None, Ds))
        print(f"degree {deg}, mutate '{mutate}'; bias = s . (h - h_full), s from the nearest of SENS.json")
        print("| method | D | bias in Delta | pass | s at D |")
        print("|---|---|---|---|---|")
        for name, D, b, p, Ds in rows:
            print(f"| {name} | {mp.nstr(D, 6)} | {b:+.2e} | {'' if p is None else ('yes' if p else 'NO')} | {Ds} |")
        os.remove(tmp)
        print("planted truth:", "pass" if ok else "FAIL")
        return 0 if ok else 1
    if cmd == "series":
        import dd_fits_mp as F
        mp.mp.dps = 60
        hf = HeldFit(args[3])
        plant = float(opts.get("plant", 0))
        pts, _ = F.load_points()
        Ds = F.clean(pts, sorted(d for d in pts if 10 <= d <= 67.77), verbose=False)
        ev, _, _ = F.fit(pts, Ds, "1/D", 20)
        print(f"fit {args[3]}, plant {plant:g} on 5a; reference: the 1/D series of degree 20 on the double-double points 10 <= D <= 67.77")
        print("| D | chain state | chain - series | s . (h_fit - h_chain) | fit's Delta - series |")
        print("|---|---|---|---|---|")
        for r in sorted(sens, key=lambda r: float(r["D"])):
            D = mp.mpf(r["D"])
            h = dict(zip(KEYS, [v for ab in hf.values(D, KS) for v in ab]))
            h["5a"] += plant
            corr = bias(r["sens"], h, r["h"])
            ser = ev(D)
            d0 = mp.mpf(r["delta"])
            print(f"| {mp.nstr(D, 8)} | {r['state']} | {float(d0 - ser):+.3e} | {corr:+.3e} | {float(d0 + corr - ser):+.2e} |")
        return 0
    sys.exit(__doc__)


if __name__ == "__main__":
    sys.exit(main())

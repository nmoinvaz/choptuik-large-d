#!/usr/bin/env python3
"""lam_g past the top of notes/MODE_COUNT.md 10.4's fit (#815): the fold's square root, the 1/p_min term, or both.

    cd hp && python3 lam_fold_fit.py planted RAWDIR [--json=OUT]   the plants first (exits 1 when a verdict is not the
                                                                   expected one); --mutate=dstar judges D*_C against
                                                                   259.9033 + 0.5, and must exit 1
    cd hp && python3 lam_fold_fit.py real [--json=OUT]             the chain's states, with the verdict rule below

RAWDIR is a clone of the raw-data repository (I603-win2/ at 1a3c1d04 or later), read for p_min only (the smallest LU
pivot of each state, frontier_audit.record_info), which sets the 1/p_min plant's rate and size.

The data. delta = lam_g - R(D) at the backbone states of hp/data/lam_gauge.csv past lam_gauge_fit.DMAX (244.64), where
R is 10.4's reference fit (c0 = 1, 1/n, degree 14, D 20 to 244.64). sigma(D) is the spread of R(D) over 10.4's kept
c0 = 1 fits, a spread over fit choices, used as each state's weight. The states' own lam_g is good to about 5e-31
(notes/LARGE_D_WALL.md, #807), far below sigma.

The forms, each fitted by weighted least squares (residuals in units of sigma), the linear coefficients solved for
each value of the nonlinear ones, which a grid and Nelder-Mead minimise:
  S       sum_{k=0}^{3} a_k u^k, u = sqrt(D* - D), D* free above the window's top: a square root at a fold, with a
          regular part of degree 1 in D* - D
  W       Re[A e^{kappa (D - 250)}], A complex, kappa = beta + i omega free: #805's entire 1/p_min term
  C       W + S, with D* free
  C_even  W + b0 + b1 (D - 259): C without the odd powers of u
r is the largest |residual|/sigma over the window. Windows: N = [258, 259.75] (40 states, near the fold, D* - D below
the 3.1 to the nearest complex singularity seen from D 255) and G = [248, 259.75] (92 states, past 248).

The verdict rule (fixed before the real fit, posted on #815):
  the fold's square root is carried when r(C_even)/r(C) >= 100 and |D*_C - 259.9033| <= 0.01, where 259.9033 is the
      fold of Delta's series (hp/data/fold_259.csv, #603's kept plan, 259.9029 to 259.9037 over estimators and bases);
  the 1/p_min term is carried when r(S)/r(C) >= 100;
  the window's verdict is "sqrt", "pmin", "both" or "neither" from those two, and C is complete when r(C) <= 10.
A form fits when r <= 10 and fails when r >= 1000.

Plants (on the real D grid and sigma, noise N(0, sigma) of seed 815, fixed before the real fit):
  PS      1e-4 (1 - 2u + u^2/2), u = sqrt(259.9033 - D): expected "sqrt", C complete, D*_C within 1e-3 and a1 within 1%,
          and W fails
  PW      #807's plant W with lam_g's constant: Re[K e^{i phi} e^{kappa (D - 250)}], K = 3e-39/p_min(250), kappa =
          b + 2 pi i/5.3, b and p_min(250) from log p_min on 245.6 to 255.8, cos zero at 250.3: expected "pmin", C
          complete, beta and omega within 1%, and S fails
  PM      PS + PW scaled to PS's largest value: expected "both", C complete, D*, beta and omega recovered
  PM10    PS + 0.1 of that: the same
  Smoved  PS with its fold at 260.4033 (control): expected "neither", since D*_C is 0.5 from 259.9033
  Snoisy  PS with noise 1e4 sigma (control): expected C incomplete
  Wchirp  PW's rate with its period lengthening by 2% per unit D (reported, not judged)
"""
import csv
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import minimize

import frontier_audit as FA
import lam_gauge_fit as LF
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
DSTAR = 259.9033                 # the fold of Delta's series, hp/data/fold_259.csv (kept plan)
DSTAR_TOL = 0.01
WINDOWS = dict(N=(258.0, 259.75), G=(248.0, 259.75))
NEED, FIT, FAIL = 100.0, 10.0, 1000.0
SEED = 815
args = dict(a.lstrip("-").split("=", 1) if "=" in a else (a.lstrip("-"), "1") for a in sys.argv[2:] if a.startswith("--"))
pos = [a for a in sys.argv[2:] if not a.startswith("--")]
DREF = DSTAR + (0.5 if args.get("mutate") == "dstar" else 0.0)     # the fold the rule judges D*_C against


def data():
    """[(D, delta, sigma)] of the backbone past DMAX, delta = lam_g - R, sigma the kept fits' spread of R"""
    mp.mp.dps = 40
    pts = LF.backbone()
    Rs = []
    for f in LF.fits(pts, fixed0=mp.mpf(1)):
        if not f["sel"]:
            continue
        sel = [(D, y) for D, y in pts if D >= f["Dlo"]]
        tf = (lambda D: 1 / (D - 2)) if f["var"] == "n" else (lambda D: 1 / D)
        c, _ = LF.lsq([tf(D) for D, _ in sel], [y for _, y in sel], f["d"], mp.mpf(1))
        Rs.append(lambda D, c=c, tf=tf: mp.polyval(c[::-1], tf(D)))
    c, _ = LF.lsq([1 / (D - 2) for D, _ in pts], [y for _, y in pts], LF.REF_DEG, fixed0=mp.mpf(1))
    out = []
    for r in csv.DictReader(x for x in open(os.path.join(HERE, "data", "lam_gauge.csv")) if not x.startswith("#")):
        D = mp.mpf(r["D_str"]) if r["D_str"] else mp.mpf(float(r["D"]))
        if r["role"] != "backbone" or D <= LF.DMAX:
            continue
        r0 = mp.polyval(c[::-1], 1 / (D - 2))
        out.append((float(D), float(mp.mpf(r["lam_g"]) - r0), float(max(abs(R(D) - r0) for R in Rs))))
    return sorted(out)


def lin(cols, y, sig):
    A = np.vstack(cols).T
    c, *_ = np.linalg.lstsq(A / sig[:, None], y / sig, rcond=None)
    return c, (y - A @ c) / sig


def wcols(D, b, w):
    x = D - 250.0
    return [np.exp(b * x) * np.cos(w * x), np.exp(b * x) * np.sin(w * x)]


def scols(D, Ds, K=3):
    u = np.sqrt(Ds - D)
    return [u ** k for k in range(K + 1)]


def nm(f, starts, ok):
    best = None
    for p0 in starts:
        m = minimize(lambda p: f(p) if ok(p) else 1e300, p0, method="Nelder-Mead",
                     options=dict(xatol=1e-12, fatol=1e-12, maxiter=40000, maxfev=40000))
        if best is None or m.fun < best.fun:
            best = m
    return best


def lrms(res):
    return math.log(float(np.sqrt(np.mean(res ** 2))))


def form(D, y, sig, name, K=3):
    """the fit of one form: dict(r, and its parameters)"""
    top = float(D.max())
    if name == "S":
        f = lambda p: lrms(lin(scols(D, p[0], K), y, sig)[1])        # noqa: E731
        g = np.linspace(top + 1e-3, top + 2, 200)
        m = nm(f, [[g[i]] for i in np.argsort([f([x]) for x in g])[:3]], lambda p: p[0] > top)
        c, res = lin(scols(D, m.x[0], K), y, sig)
        return dict(r=float(np.max(np.abs(res))), Dstar=float(m.x[0]), a=[float(x) for x in c])
    if name in ("W", "C_even"):
        extra = [np.ones_like(D), D - 259.0] if name == "C_even" else []
        f = lambda p: lrms(lin(wcols(D, *p) + extra, y, sig)[1])     # noqa: E731
        G = [(b, w) for b in np.linspace(0, 5, 26) for w in np.linspace(0.05, 5, 50)]
        m = nm(f, [list(G[i]) for i in np.argsort([f(p) for p in G])[:4]], lambda p: True)
        c, res = lin(wcols(D, *m.x) + extra, y, sig)
        return dict(r=float(np.max(np.abs(res))), beta=float(m.x[0]), omega=float(abs(m.x[1])), a=[float(x) for x in c])
    if name == "C":
        f = lambda p: lrms(lin(wcols(D, p[0], p[1]) + scols(D, p[2], K), y, sig)[1])   # noqa: E731
        G = [(b, w, s) for b in np.linspace(0, 5, 11) for w in np.linspace(0.1, 5, 21)
             for s in top + np.geomspace(1e-3, 2, 12)]
        m = nm(f, [list(G[i]) for i in np.argsort([f(p) for p in G])[:6]], lambda p: p[2] > top)
        c, res = lin(wcols(D, m.x[0], m.x[1]) + scols(D, m.x[2], K), y, sig)
        Wv = np.vstack(wcols(D, m.x[0], m.x[1])).T @ c[:2]
        u = np.sqrt(m.x[2] - D)
        odd = c[3] * u + (c[5] * u ** 3 if K >= 3 else 0)
        return dict(r=float(np.max(np.abs(res))), beta=float(m.x[0]), omega=float(abs(m.x[1])), Dstar=float(m.x[2]),
                    a=[float(x) for x in c[2:]], A=[float(x) for x in c[:2]],
                    W_top=float(Wv[-1]), odd_top=float(odd[-1]), W_max=float(np.max(np.abs(Wv))),
                    odd_max=float(np.max(np.abs(odd))))
    raise ValueError(name)


def judge(D, y, sig):
    """every form on one window and the verdict of the rule"""
    out = {k: form(D, y, sig, k) for k in ("S", "W", "C", "C_even")}
    sq = out["C_even"]["r"] / out["C"]["r"] >= NEED and abs(out["C"]["Dstar"] - DREF) <= DSTAR_TOL
    pm = out["S"]["r"] / out["C"]["r"] >= NEED
    out["verdict"] = {(True, True): "both", (True, False): "sqrt", (False, True): "pmin"}.get((sq, pm), "neither")
    out["complete"] = out["C"]["r"] <= FIT
    return out


def line(name, o):
    c = o["C"]
    return ("  %-7s r: S %.1e  W %.1e  C %.1e  C_even %.1e | C_even/C %.1e  S/C %.1e | C: D* %.5f beta %.4f omega "
            "%.4f a1 %+.3e | %s, C %s" % (name, o["S"]["r"], o["W"]["r"], c["r"], o["C_even"]["r"],
                                          o["C_even"]["r"] / c["r"], o["S"]["r"] / c["r"], c["Dstar"], c["beta"],
                                          c["omega"], c["a"][1], o["verdict"], "complete" if o["complete"] else
                                          "incomplete"))


def pivot_law(raw):
    """b and p_min(250) from log p_min on 245.6 to 255.8, as frontier_audit.plant_funcs takes them"""
    ri = FA.record_info(raw)
    R = sorted((k, v["min_pivot"]) for k, v in ri.items() if 245.6 <= k <= 255.8 and v["min_pivot"])
    pf = np.polyfit([k for k, _ in R], np.log([p for _, p in R]), 1)
    return float(-pf[0]), math.exp(float(np.polyval(pf, 250.0)))


def plants(D, sig, b, p250):
    n = sig * np.random.default_rng(SEED).standard_normal(len(D))
    u = np.sqrt(DSTAR - D)
    PS = 1e-4 * (1 - 2 * u + u ** 2 / 2)
    kap = complex(b, 2 * math.pi / 5.3)
    ph = math.pi / 2 - kap.imag * 0.3
    PW = np.real(3e-39 / p250 * np.exp(1j * ph) * np.exp(kap * (D - 250)))
    om = 2 * math.pi / (5.3 * (1 + 0.02 * (D - 250)))
    phase = np.cumsum(np.r_[0, np.diff(D) * (om[1:] + om[:-1]) / 2]) + om[0] * (D[0] - 250) + ph
    PWc = np.exp(b * (D - 250)) * np.cos(phase)
    sc = lambda v, ref: v / np.max(np.abs(v)) * np.max(np.abs(ref))         # noqa: E731
    u5 = np.sqrt(DSTAR + 0.5 - D)
    return dict(PS=PS + n, PW=PW + n, PM=PS + sc(PW, PS) + n, PM10=PS + 0.1 * sc(PW, PS) + n,
                Smoved=1e-4 * (1 - 2 * u5 + u5 ** 2 / 2) + n, Snoisy=PS + 1e4 * n, Wchirp=sc(PWc, PW) + n)


def expected(name, o, b):
    """True when the plant's outcome is the registered one (None: reported only)"""
    c = o["C"]
    near = lambda x, t: abs(x / t - 1) <= 0.01                             # noqa: E731
    if name == "PS":
        return (o["verdict"] == "sqrt" and o["complete"] and abs(c["Dstar"] - DSTAR) <= 1e-3 and near(c["a"][1], -2e-4)
                and o["W"]["r"] >= FAIL)
    if name == "PW":
        return (o["verdict"] == "pmin" and o["complete"] and near(c["beta"], b) and near(c["omega"], 2 * math.pi / 5.3)
                and o["S"]["r"] >= FAIL)
    if name in ("PM", "PM10"):
        return (o["verdict"] == "both" and o["complete"] and abs(c["Dstar"] - DSTAR) <= 1e-3 and near(c["beta"], b)
                and near(c["omega"], 2 * math.pi / 5.3))
    if name == "Smoved":
        return o["verdict"] == "neither"
    if name == "Snoisy":
        return not o["complete"]
    return None


def window(pts, w):
    lo, hi = WINDOWS[w]
    sel = [p for p in pts if lo <= p[0] <= hi]
    return tuple(np.array([p[i] for p in sel]) for i in range(3))


def cmd_planted(raw):
    pts = data()
    b, p250 = pivot_law(raw)
    print("plants: 1/p_min rate b %.4f, p_min(250) %.3e, omega %.4f; fold D* %.4f" % (b, p250, 2 * math.pi / 5.3, DSTAR))
    ok, out = True, {}
    for w in WINDOWS:
        D, _, sig = window(pts, w)
        print("window %s, D %.2f to %.2f, %d states, sigma %.1e to %.1e:" % (w, D[0], D[-1], len(D), sig.min(), sig.max()))
        for name, y in plants(D, sig, b, p250).items():
            o = judge(D, y, sig)
            e = expected(name, o, b)
            ok &= e is not False
            print(line(name, o) + " | " + {True: "as expected", False: "NOT AS EXPECTED", None: "reported"}[e])
            out[w + ":" + name] = dict(o, expected=e)
    print("planted:", "PASS" if ok else "FAIL")
    if "json" in args:
        json.dump(dict(b=b, p250=p250, plants=out), open(args["json"], "w"), indent=1)
    return 0 if ok else 1


def cmd_real():
    pts = data()
    out = {}
    for w in WINDOWS:
        D, y, sig = window(pts, w)
        print("window %s, D %.4f to %.4f, %d states, |delta| up to %.2e, sigma %.1e to %.1e:" % (
            w, D[0], D[-1], len(D), np.max(np.abs(y)), sig.min(), sig.max()))
        o = judge(D, y, sig)
        print(line("real", o))
        c = o["C"]
        print("    C's parts: the odd powers of u up to %.2e (%.2e at the top), W up to %.2e (%.2e at the top)" % (
            c["odd_max"], c["odd_top"], c["W_max"], c["W_top"]))
        # diagnostics, not judged: the fold part's degree
        for K in (1, 2, 4, 5):
            ck = form(D, y, sig, "C", K)
            print("    diagnostic C with K = %d: r %.1e, D* %.5f, beta %.4f, omega %.4f" % (
                K, ck["r"], ck["Dstar"], ck["beta"], ck["omega"]))
            o["C_K%d" % K] = ck
        out[w] = o
    if "json" in args:
        json.dump(out, open(args["json"], "w"), indent=1)
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "planted" and pos:
        sys.exit(cmd_planted(pos[0]))
    if cmd == "real":
        sys.exit(cmd_real())
    print(__doc__)
    sys.exit(2)

#!/usr/bin/env python3
"""The type of lam_g's nearest real singularity in D, from its own series at chain_nsr's bases from D 258.78 (#818),
by hp/singularity_type.py's estimators (#782), beside Delta's series from the same jets.

    cd hp && python3 lam_singularity.py control RAWDIR [--mutate=n]   the series' construction (exits 1 on a fail)
    cd hp && python3 lam_singularity.py planted RAWDIR [--json=OUT]   plants of lam_g's size at each base (exits 1 on a
                                                                      fail)
    cd hp && python3 lam_singularity.py real RAWDIR --planted=PLANTED.json [--json=OUT]
                                                                      lam_g's and Delta's series at each base
    cd hp && python3 lam_singularity.py posthoc RAWDIR [--json=OUT]   post hoc: the fold's coefficient from the high
                                                                      orders, plants first

RAWDIR is a clone of the raw-data repository; the jets are I603-win2/chain_nsr/jet_pin3_D*.npz (25 orders, eight
parts), read by lam_frontier.load_ah_jet for the horizon block a_h and by the same sums for Delta (the last entry).

lam_g's series. With a_h(D_b + h) = sum_n z_n h^n at each of the M points, <a_h^2 - 1> = sum_k m_k h^k by the Cauchy
product (exact to order N = 24, since it needs z_0..z_k only), and lam_g = 1 - (D_b - 3 + h) <a_h^2 - 1> gives
l_0 = 1 - (D_b - 3) m_0 and l_k = -(D_b - 3) m_k - m_(k-1). The estimators take c = l_1..l_24, as singularity_type's
`real` takes Delta's z_1..z_24. Sums at 200 digits, the estimators at singularity_type.DPS.

control: at each jet, (1) l_0 equals lam_gauge.lam_g of the base state to 1e-60, and (2) the full product (orders 0 to
2N + 1) at the next prediction's own float a equals lam_frontier.lam_jet's order-N value there to 1e-60 (both are
lam_g of the order-N a_h); --mutate=n puts n in place of n - 1 in this script's series only, and both must fail.

planted (fixed before the real series were read by this script): at each base D0 the fold at s1 = 259.9033 - D0, with
#782's near contamination at the same D (a second real singularity at 260.17 and a pair at 259.47 +- 1.17i, each of
amplitude 0.3 against the first and of the same type) and relative noise 1e-12 (singularity_type.plant), scaled to
lam_g's size, A = 5e-4 sqrt(s1) (#815's fold coefficient a1, 4.7e-4 to 8.7e-4 over the fold part's degree), plus a
regular part: the Taylor coefficients at D0 of 10.4's reference fit R and of an entire term
6.29e-5 e^(1.019 (D - 259.74)) cos(1.607 (D - 259.74)) (#815's C on window N, hp/data/lam_fold.json).
  S       the fold a square root (mu = 1/2)
  P       the fold a simple pole (mu = -1), the same amplitudes
  Smoved  S with every singularity 0.5 further from the base (control)
The criterion at a base (the issue's): both estimators put D* = D0 + s within 0.01 of 259.9033 and mu in the S band
[0.25, 0.75]. A base discriminates when S meets it and P and Smoved do not. planted passes when at least 4 bases
discriminate (the issue's count); exit 1 otherwise.

real: lam_g's series at each base, both estimators, the criterion against 259.903 at the bases that discriminate; the
verdict "a square root at D 259.903" when 4 or more of them meet it, otherwise "not a square root" as the issue words
it. Delta's series from the same jets are typed beside it, reported (#782's result on other builds of these bases).
"""
import csv
import glob
import json
import math
import os
import sys

import mpmath as mp
import numpy as np

import lam_frontier as LFR
import lam_gauge as LG
import lam_gauge_fit as LF
import singularity_type as ST

HERE = os.path.dirname(os.path.abspath(__file__))
DSTAR = mp.mpf("259.9033")       # Delta's fold, hp/data/fold_259.csv (#603's kept plan)
TOL = 0.01
D_MIN = 258.78
NCOUNT = 4
FOLD_A = 5e-4                    # #815's fold coefficient a1 (lam_gauge sqrt(D* - D) coefficient)
W_REG = (6.29e-5, 1.019, 1.607, 259.74)   # #815's C, window N: |W| at the top, beta, omega, top D
args = dict(a.lstrip("-").split("=", 1) if "=" in a else (a.lstrip("-"), "1") for a in sys.argv[2:] if a.startswith("--"))
pos = [a for a in sys.argv[2:] if not a.startswith("--")]


def jets(raw):
    fs = sorted(glob.glob(os.path.join(raw, "I603-win2", "chain_nsr", "jet_pin3_D*.npz")))
    return [f for f in fs if float(f.rsplit("_D", 1)[1][:-4]) >= D_MIN - 1e-9]


def delta_series(path):
    """Delta's z_0..z_N (the state's last entry), the eight parts summed"""
    z = np.load(path)
    ks = sorted(k for k in z.files if k[0] == "z" and k[1:].isdigit())
    N1 = z[ks[0]].shape[0]
    return [mp.fsum(mp.mpf(float(z[k][n][-1])) for k in ks) for n in range(N1)]


def lam_series(Db, c, M, full=False, mutate=""):
    """l_0..l_N of lam_g (or of the full product, orders 0..2N + 1, when full)"""
    N = len(c) - 1
    top = 2 * N if full else N
    m = []
    for k in range(top + 1):
        s = mp.fsum(c[n][j] * c[k - n][j] for j in range(M) for n in range(max(0, k - N), min(k, N) + 1))
        m.append(s / M - (1 if k == 0 else 0))
    b = (Db - 2) if mutate == "n" else (Db - 3)
    out = [1 - b * m[0]] + [-b * m[k] - m[k - 1] for k in range(1, top + 1)]
    if full:
        out.append(-m[top])
    return out


def at(series, h):
    return mp.fsum(x * h ** k for k, x in enumerate(series))


def est(c):
    """singularity_type's two estimators on c_1..c_N, at its precision"""
    with mp.workdps(ST.DPS):
        cc = [+x for x in c]
        return ST.ratio_fit(cc), ST.dlog_fit(cc)


def meets(D0, e):
    """both estimators: D* within TOL of DSTAR and mu in the S band"""
    return all(x is not None and abs(D0 + x[0] - float(DSTAR)) <= TOL and 0.25 <= x[1] <= 0.75 for x in e)


def row(D0, e):
    r, d = e
    f = lambda x: "none" if x is None else "D* %.4f mu %+.3f %s" % (D0 + x[0], x[1], ST.kind(x[1]))   # noqa: E731
    return "ratio %-30s dlog %-30s" % (f(r), f(d))


_CR = []


def regular(D0, N):
    """r_1..r_N: the Taylor coefficients at D0 of 10.4's reference fit and of #815's entire term"""
    if not _CR:
        with mp.workdps(60):
            pts = LF.backbone()
            _CR.extend(LF.lsq([1 / (D - 2) for D, _ in pts], [y for _, y in pts], LF.REF_DEG, fixed0=mp.mpf(1))[0])
    cR = _CR
    n0 = mp.mpf(D0) - 2
    tau = [(-1) ** k / n0 ** (k + 1) for k in range(N + 1)]       # 1/(n0 + h)
    pw, R = [mp.mpf(1)] + [mp.mpf(0)] * N, [mp.mpf(0)] * (N + 1)
    for j, cj in enumerate(cR):
        if j:
            pw = [mp.fsum(pw[i] * tau[k - i] for i in range(k + 1)) for k in range(N + 1)]
        R = [R[k] + cj * pw[k] for k in range(N + 1)]
    amp, b, w, Dt = W_REG
    kap = mp.mpc(b, w)
    W = [mp.re(amp * mp.exp(kap * (mp.mpf(D0) - Dt)) * kap ** k / mp.factorial(k)) for k in range(N + 1)]
    return [R[k] + W[k] for k in range(1, N + 1)]


def plant_at(D0, mu, shift=0.0, N=24, seed=818, amp=FOLD_A):
    s1 = float(DSTAR) - D0 + shift
    s2 = 260.17 - D0 + shift
    p3 = complex(259.47 - D0 + shift, 1.17)
    with mp.workdps(ST.DPS):
        sing = ST.plant(mu, s1, True, N=N, seed=seed, s2=s2, s3=abs(p3), arg3=float(np.degrees(np.angle(p3))))
        A = amp * math.sqrt(float(DSTAR) - D0)
        reg = regular(D0, N)
        return [A * x + y for x, y in zip(sing, reg)]


def cmd_control(raw):
    mp.mp.dps = 200
    mut = args.get("mutate", "")
    ok1, ok2, w1, w2, n = True, True, 0.0, 0.0, 0
    for jp in jets(raw):
        Db, c, M = LFR.load_ah_jet(jp)
        ser = lam_series(Db, c, M, mutate=mut)
        st = jp.replace("jet_pin3_", "anm_pin3_")
        if os.path.exists(st):
            parts, meta = LG.load(st)
            with mp.workprec(LG.PREC):
                lam0 = LG.lam_g(parts, meta)
            e1 = abs(ser[0] - lam0)
            w1 = max(w1, float(e1))
            ok1 &= e1 <= mp.mpf("1e-60")
        nxt = sorted(glob.glob(os.path.join(os.path.dirname(jp), "pred_pin3_D*.npz")))
        nxt = [p for p in nxt if float(p.rsplit("_D", 1)[1][:-4]) > float(Db) + 1e-9]
        if not nxt:
            continue
        a, _ = LFR.pred_ah(nxt[0])
        full = lam_series(Db, c, M, full=True, mutate=mut)
        e2 = abs(at(full, a) - LFR.lam_jet(Db, c, M, Db + a)[-1])
        w2 = max(w2, float(e2))
        ok2 &= e2 <= mp.mpf("1e-60")
        n += 1
    print("control%s: l_0 against the base state's lam_g, largest %.1e (bound 1e-60): %s; the full product against "
          "lam_jet at %d next predictions, largest %.1e: %s" % (" (--mutate=%s)" % mut if mut else "", w1,
                                                                 "pass" if ok1 else "FAIL", n, w2, "pass" if ok2 else "FAIL"))
    return 0 if ok1 and ok2 else 1


def cmd_planted(raw):
    out, disc = {}, []
    print("plants of lam_g's size (A = %.0e sqrt(s1), #782's near contamination, regular part of #815's size):" % FOLD_A)
    for jp in jets(raw):
        D0 = float(jp.rsplit("_D", 1)[1][:-4])
        res = {}
        for name, mu, sh in (("S", 0.5, 0.0), ("P", -1.0, 0.0), ("Smoved", 0.5, 0.5)):
            e = est(plant_at(D0, mu, sh))
            res[name] = dict(ratio=e[0], dlog=e[1], meets=meets(D0, e))
        a10 = est(plant_at(D0, 0.5, amp=FOLD_A / 10))
        res["S_A/10"] = dict(ratio=a10[0], dlog=a10[1], meets=meets(D0, a10))
        d = res["S"]["meets"] and not res["P"]["meets"] and not res["Smoved"]["meets"]
        if d:
            disc.append(D0)
        out[D0] = dict(res, discriminates=d)
        print("  D0 %.4f s1 %.3f  S: %s %s | P: %s | Smoved: %s | S, A/10: %s | %s" % (
            D0, float(DSTAR) - D0, row(D0, (res["S"]["ratio"], res["S"]["dlog"])), "meets" if res["S"]["meets"] else "misses",
            "meets" if res["P"]["meets"] else "misses", "meets" if res["Smoved"]["meets"] else "misses",
            "meets" if res["S_A/10"]["meets"] else "misses", "discriminates" if d else "does not discriminate"))
    ok = len(disc) >= NCOUNT
    print("planted: %d of %d bases discriminate (%s): %s" % (len(disc), len(out), ", ".join("%.4f" % x for x in disc),
                                                           "PASS" if ok else "FAIL"))
    if "json" in args:
        json.dump(dict(discriminating=disc, bases=out), open(args["json"], "w"), indent=1)
    return 0 if ok else 1


def cmd_real(raw):
    mp.mp.dps = 200
    pl = json.load(open(args["planted"])) if "planted" in args else None
    disc = set(round(x, 4) for x in pl["discriminating"]) if pl else None
    out, hits = {}, []
    print("lam_g's and Delta's series (criterion: both estimators D* within %.2f of %s, mu in [0.25, 0.75])" % (TOL, DSTAR))
    for jp in jets(raw):
        Db, c, M = LFR.load_ah_jet(jp)
        D0 = float(Db)
        ser = lam_series(Db, c, M)
        el = est(ser[1:])
        ed = est(delta_series(jp)[1:])
        m = meets(D0, el)
        usable = disc is None or round(D0, 4) in disc
        if m and usable:
            hits.append(D0)
        out[D0] = dict(lam=dict(ratio=el[0], dlog=el[1], meets=m), delta=dict(ratio=ed[0], dlog=ed[1],
                                                                              meets=meets(D0, ed)), usable=usable)
        print("  D0 %.4f lam_g: %s %s | Delta: %s %s%s" % (D0, row(D0, el), "meets" if m else "misses", row(D0, ed),
                                                         "meets" if meets(D0, ed) else "misses",
                                                         "" if usable else " (base does not discriminate)"))
    v = "a square root at D 259.903" if len(hits) >= NCOUNT else "not a square root"
    print("real: %d discriminating bases meet the criterion (%s): %s" % (len(hits), ", ".join("%.4f" % x for x in hits), v))
    if "json" in args:
        json.dump(dict(hits=hits, verdict=v, bases=out), open(args["json"], "w"), indent=1)
    return 0


def amplitude(c, D0, ks=range(16, 25)):
    """post hoc: the fold's coefficient a1 of sqrt(D* - D) from c_k = A binom(1/2, k) (-1)^k / s^k at the high orders,
    A = a1 sqrt(s), s = DSTAR - D0; (median a1 over ks, its relative spread)"""
    s = DSTAR - mp.mpf(D0)
    v = [float(c[k - 1] * s ** k / (mp.binomial(mp.mpf(1) / 2, k) * (-1) ** k) / mp.sqrt(s)) for k in ks]
    med = float(np.median(v))
    return med, (max(v) - min(v)) / abs(med)


def cmd_posthoc(raw):
    """post hoc (after the registered real run): the fold's coefficient from the high orders, on the plants S (a1
    5e-4) and S A/10 (5e-5) first, then on lam_g's and Delta's series, at the bases from D 259.36"""
    mp.mp.dps = 200
    out = {}
    print("post hoc: a1, the coefficient of sqrt(D* - D), from c_16..c_24 (median, relative spread over k)")
    for jp in jets(raw):
        D0 = float(jp.rsplit("_D", 1)[1][:-4])
        if D0 < 259.36:
            continue
        with mp.workdps(ST.DPS):
            pS = amplitude(plant_at(D0, 0.5), D0)
            pA = amplitude(plant_at(D0, 0.5, amp=FOLD_A / 10), D0)
        Db, c, M = LFR.load_ah_jet(jp)
        lam = amplitude(lam_series(Db, c, M)[1:], D0)
        dl = amplitude(delta_series(jp)[1:], D0)
        out[D0] = dict(plant_S=pS, plant_A10=pA, lam=lam, delta=dl)
        print("  D0 %.4f  plant S %.3e (%.1e)  plant A/10 %.3e (%.1e) | lam_g %+.3e (%.1e) | Delta %+.3e (%.1e)" % (
            D0, pS[0], pS[1], pA[0], pA[1], lam[0], lam[1], dl[0], dl[1]))
    if "json" in args:
        json.dump(out, open(args["json"], "w"), indent=1)
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    fn = dict(control=cmd_control, planted=cmd_planted, real=cmd_real, posthoc=cmd_posthoc).get(cmd)
    if fn is None or not pos:
        print(__doc__)
        sys.exit(2)
    sys.exit(fn(pos[0]))

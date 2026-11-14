#!/usr/bin/env python3
"""The Borel-Pade order K at which P1's zeros serve the corrected zero law out to |x| = 35 (#788; pre-registered on
the issue).

P1's exact sigma_0..sigma_K (and those of the control C = P1 (x - x0 - 1)/(x - x0)), each with complex relative noise
1e-14 (stokes_lines.noisy, seeds 1 + 10K and 3 + 10K), are summed by Borel-Pade [m/(K-m)], m = ceil(0.35K) ..
floor(K/2) + 1 (zerolib.borel_pade, 60 digits): S(x) = c0 + sum_p r_p exp(t_p x). The zeros of each sum in
Omega = {|x| <= 42, Re x >= 2} are grid minima (h = 0.2) of |S| in arb (256 bits) refined by Newton, with the argument
principle on the boundary as the completeness check. The arm (#357's rule, continued) must reach x6; the order-3 law
of #773 on x1..x6 gives x0_hat. Decisive at K: P1 and C each have a sum that enters, every P1 sum that enters has
e1 = |x0_hat - x0| <= 0.15, and every C sum that enters has |x0_hat - (x0 + 1)| > 3 max e1.

    cd solver/harmony/stokes
    python3 stokes_bp.py selftest [--mutate=coarse]     the zero search on e^{ax} - c e^{bx} (known zeros); coarse
                                                        (h = 2) must fail
    python3 stokes_bp.py exact                          the law on P1's exact arm (#773: e = 0.0117)
    python3 stokes_bp.py run --K=40,60,80,100 --out=FILE   the sums (resumable), then the verdict
    python3 stokes_bp.py verdict --out=FILE             the verdict from FILE alone
    python3 stokes_bp.py posthoc --K=100 --noise=0,1e-20 --out=FILE   post hoc (not part of the test): P1's sums at
                                                        other noise levels, to tell the noise floor from K
Exit codes: selftest 0/1; run and verdict 0 if some K is decisive, 3 if none; 1 if K = 40 is decisive (the control).
"""
import json
import math
import os
import sys
import time

import mpmath as mp
import numpy as np

import stokes_series as T
from flint import acb, ctx

SL = T.SL
Z = SL.Z
mp.mp.dps = 60
PREC = 256
R_BP, RE_MIN, H = 42.0, 2.0, 0.2
TOL_E, D_CTRL = 0.15, 1
args = dict(a.lstrip("-").split("=", 1) if "=" in a else (a.lstrip("-"), "1") for a in sys.argv[2:])


# ---- exponential sums in arb ---------------------------------------------------------------------------------
class ExpSum:
    def __init__(self, ts, rs, c0):
        ctx.prec = PREC
        self.t = [T.to_acb(t) for t in ts]
        self.r = [T.to_acb(r) for r in rs]
        self.rt = [r * t for r, t in zip(self.r, self.t)]
        self.c0 = T.to_acb(c0)

    def f(self, x):
        s = self.c0
        for t, r in zip(self.t, self.r):
            s += r * (t * x).exp()
        return s

    def fd(self, x):
        s, d = self.c0, acb(0)
        for t, r, rt in zip(self.t, self.r, self.rt):
            e = (t * x).exp()
            s += r * e
            d += rt * e
        return s, d


def logabs(v):
    a = abs(v)
    return float(a.log().mid()) if a > 0 else -1e300


def zeros_region(F, h=H, R=R_BP):
    """grid minima of log|F| on {|x| <= R, Re x >= RE_MIN}, Newton in arb; the winding on the boundary"""
    ctx.prec = PREC
    xs = np.arange(RE_MIN - h, R + 2 * h, h)
    ys = np.arange(-R - h, R + 2 * h, h)
    G = np.full((len(ys), len(xs)), np.inf)
    for i, y in enumerate(ys):
        for k, x in enumerate(xs):
            G[i, k] = logabs(F.f(acb(float(x), float(y))))
    found = []
    for i in range(1, len(ys) - 1):
        for k in range(1, len(xs) - 1):
            if G[i, k] > G[i - 1:i + 2, k - 1:k + 2].min():
                continue
            x = acb(float(xs[k]), float(ys[i]))
            conv = False
            for _ in range(60):
                s, d = F.fd(x)
                dx = s / d
                x = (x - dx).mid()
                if abs(complex(dx)) < 1e-40 * max(1.0, abs(complex(x))):
                    conv = True
                    break
            z = complex(x)
            if conv and abs(z) <= R and z.real >= RE_MIN and all(abs(z - complex(w)) > 1e-10 for w in found):
                found.append(x)
    return sorted(found, key=lambda a: abs(complex(a))), winding(F, R)


def winding(F, R):
    yb = math.sqrt(R ** 2 - RE_MIN ** 2)
    thb = math.atan2(yb, RE_MIN)

    def pt(t):
        if t < 1:
            return complex(RE_MIN, yb - 2 * yb * t)
        th = -thb + 2 * thb * (t - 1)
        return R * complex(math.cos(th), math.sin(th))
    cache = {}

    def ang(t):
        if t not in cache:
            z = pt(t)
            cache[t] = float(F.f(acb(z.real, z.imag)).arg().mid())
        return cache[t]
    hits = 0

    def step(a, b, depth):
        nonlocal hits
        d = (ang(b) - ang(a) + math.pi) % (2 * math.pi) - math.pi
        if abs(d) > math.pi / 8:
            if depth < 30:
                m = 0.5 * (a + b)
                return step(a, m, depth + 1) + step(m, b, depth + 1)
            hits += 1
        return d
    ts = np.linspace(0, 2, 4001)
    tot = sum(step(float(a), float(b), 0) for a, b in zip(ts[:-1], ts[1:]))
    return int(round(tot / (2 * math.pi))), hits


# ---- selftest ------------------------------------------------------------------------------------------------
def selftest(coarse=False):
    a, b, c = mp.mpc("1.70", "1.34"), mp.mpc("0.80", "0.01"), mp.mpc("3.0", "-2.0")
    F = ExpSum([a, b], [mp.mpc(1), -c], mp.mpc(0))
    want = []
    for k in range(-60, 61):
        x = complex((mp.log(c) + 2j * mp.pi * k) / (a - b))
        if abs(x) <= R_BP and x.real >= RE_MIN:
            want.append(x)
    zs, (w, hits) = zeros_region(F, h=2.0 if coarse else H)
    got = [complex(z) for z in zs]
    miss = [x for x in want if not any(abs(x - g) < 1e-20 for g in got)]
    ok = not miss and len(got) == len(want) and w == len(want) and hits == 0
    print("selftest%s: e^{ax} - c e^{bx}, %d zeros in the region; found %d (all to 1e-20: %s), winding %d: %s" % (
        " (coarse, h = 2)" if coarse else "", len(want), len(got), "yes" if not miss else "no, %d missed" % len(miss),
        w, "PASS" if ok else "FAIL"))
    return ok


# ---- the planted functions -------------------------------------------------------------------------------------
def planted():
    s1, _ = T.functions(200)
    z1, _, _ = T.zeros_omega(T.poly_of(s1), s1.name)
    arm = T.arm_of([T.to_mp(z) for z in z1])
    x0 = T.refine_root(s1, arm[0])
    sdps = SL.SDPS
    SL.SDPS = T.XDPS
    sc = SL.displaced(s1, x0, mp.mpf(D_CTRL))
    SL.SDPS = sdps
    with mp.workdps(60):
        return s1, sc, [+x for x in arm], +x0


def cmd_exact():
    s1, sc, arm, x0 = planted()
    xh, par = T.predict(arm, 3)
    e = float(abs(xh - arm[0]))
    print("exact arm: " + ", ".join(Z.cnum(x, 4) for x in arm[:7]) + "; order-3 law on x1..x6: x0_hat %s, e = %.4f "
          "(#773: 0.0117)" % (Z.cnum(xh, 5), e))
    return abs(e - 0.0117) < 5e-5


# ---- the run ---------------------------------------------------------------------------------------------------
def one_sum(sig, K, m, exact_arm, target):
    t0 = time.time()
    try:
        ts, rs, c0 = Z.borel_pade(sig, m, K - m)
    except ZeroDivisionError:                       # the Pade system is singular (exact data of lower degree)
        return dict(K=K, m=m, zeros=0, winding=None, hits=0, complete=False, arm=None, entered=False,
                    pade="singular", seconds=round(time.time() - t0, 1))
    F = ExpSum(ts, rs, c0)
    zs, (wnd, hits) = zeros_region(F)
    zm = [T.to_mp(z) for z in zs]
    rec = dict(K=K, m=m, zeros=len(zs), winding=wnd, hits=hits, complete=(wnd == len(zs) and hits == 0))
    arm = T.arm_of(zm) if zm else None
    rec["arm"] = None if arm is None else [[float(x.real), float(x.imag)] for x in arm[:7]]
    rec["entered"] = bool(rec["complete"] and arm is not None and len(arm) >= 7)
    if arm is not None:
        rec["zero_err"] = [float(abs(arm[j] - exact_arm[j])) for j in range(min(7, len(arm)))]
    if rec["entered"]:
        xh, _ = T.predict(arm, 3)
        rec["x0_hat"] = [float(xh.real), float(xh.imag)]
        rec["e"] = float(abs(xh - target))
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def cmd_run():
    out = args["out"]
    Ks = [int(k) for k in args.get("K", "40,60,80,100").split(",")]
    done = set()
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line)
            done.add((r["fn"], r["K"], r["m"]))
    s1, sc, arm, x0 = planted()
    arm_c = [x0 + D_CTRL] + arm[1:]
    for K in Ks:
        for fn, s, seed, ex, tgt in (("P1", s1, 1, arm, x0), ("C", sc, 3, arm_c, x0 + D_CTRL)):
            sig = SL.noisy(s.sigma(K), seed + 10 * K)
            for m in range(math.ceil(0.35 * K), K // 2 + 2):
                if (fn, K, m) in done:
                    continue
                rec = dict(fn=fn, **one_sum(sig, K, m, ex, tgt))
                with open(out, "a") as fo:
                    fo.write(json.dumps(rec) + "\n")
                ze = rec.get("zero_err")
                print("%-2s K %3d [%2d/%2d]: zeros %3d winding %3d entered %-5s max zero error x1..x6 %s  e %s  "
                      "(%.0f s)" % (
                    fn, K, m, K - m, rec["zeros"], rec["winding"], rec["entered"],
                    "%.1e" % max(ze[1:7]) if ze and len(ze) >= 7 else "-",
                    "%.4f" % rec["e"] if "e" in rec else "-", rec["seconds"]), flush=True)
    return verdict(out)


def cmd_posthoc():
    out = args["out"]
    done = set()
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line)
            done.add((r["noise"], r["K"], r["m"]))
    s1, sc, arm, x0 = planted()
    for K in [int(k) for k in args.get("K", "100").split(",")]:
        for nz in [float(v) for v in args.get("noise", "0,1e-20").split(",")]:
            SL.NOISE = nz
            sig = SL.noisy(s1.sigma(K), 1 + 10 * K)
            for m in range(math.ceil(0.35 * K), K // 2 + 2):
                if (nz, K, m) in done:
                    continue
                rec = dict(fn="P1", noise=nz, **one_sum(sig, K, m, arm, x0))
                with open(out, "a") as fo:
                    fo.write(json.dumps(rec) + "\n")
                ze = rec.get("zero_err")
                print("P1 noise %.0e K %3d [%2d/%2d]: entered %-5s zero errors x1..x6 %s  e %s  (%.0f s)" % (
                    nz, K, m, K - m, rec["entered"], " ".join("%.1e" % v for v in ze[1:7]) if ze and len(ze) >= 7
                    else "-", "%.4f" % rec["e"] if "e" in rec else "-", rec["seconds"]), flush=True)
    SL.NOISE = 1e-14
    return 0


def verdict(out):
    recs = [json.loads(line) for line in open(out)]
    Ks = sorted({r["K"] for r in recs})
    first = None
    res = {}
    for K in Ks:
        p = [r for r in recs if r["K"] == K and r["fn"] == "P1"]
        c = [r for r in recs if r["K"] == K and r["fn"] == "C"]
        pe = [r for r in p if r["entered"]]
        ce = [r for r in c if r["entered"]]
        e1max = max(r["e"] for r in pe) if pe else None
        fires = bool(ce) and e1max is not None and all(r["e"] > 3 * e1max for r in ce)
        dec = bool(pe) and bool(ce) and e1max <= TOL_E and fires
        zmax = [max(r["zero_err"][1:7]) for r in pe]
        res[K] = dec
        if dec and first is None:
            first = K
        print("K = %3d: P1 sums %d, entered %d (e1 %s); C sums %d, entered %d (|x0_hat - (x0+1)| %s); control %s; "
              "zero errors x1..x6 of the entered P1 sums %s; decisive: %s" % (
                  K, len(p), len(pe), ("%.4f-%.4f" % (min(r["e"] for r in pe), e1max)) if pe else "-", len(c),
                  len(ce), ("%.3f-%.3f" % (min(r["e"] for r in ce), max(r["e"] for r in ce))) if ce else "-",
                  "fires" if fires else "does not fire", ("%.1e-%.1e" % (min(zmax), max(zmax))) if zmax else "-",
                  dec))
    print("verdict: %s; the control K = 40 is %s" % (
        "decisive first at K = %d" % first if first else "K = 100 does not suffice",
        "not decisive (right)" if not res.get(40) else "DECISIVE (the control fails)"))
    if res.get(40):
        return 1
    return 0 if first else 3


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "selftest":
        sys.exit(0 if selftest(coarse=args.get("mutate") == "coarse") else 1)
    if cmd == "exact":
        sys.exit(0 if cmd_exact() else 1)
    if cmd == "run":
        sys.exit(cmd_run())
    if cmd == "posthoc":
        sys.exit(cmd_posthoc())
    if cmd == "verdict":
        sys.exit(verdict(args["out"]))
    raise SystemExit(__doc__)

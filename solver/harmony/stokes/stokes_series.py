#!/usr/bin/env python3
"""The two-endpoint zero law with its 1/x corrections, on planted zeros only (#773; pre-registered on the issue).

Two endpoint contributions C_k e^{kappa x} x^{-a} (1 + O(1/x)) + C_z e^{zeta x} x^{-b} (1 + O(1/x)) vanish where
    F_N(x) = Lam x + q ln x - L + sum_{k=1..N} c_k x^{-k} = 2 pi i j,
the c_k being the coefficients of the logarithm of the ratio of the endpoints' series (law of order N; N = 0 is
#357's law). The N + 3 unknowns are fitted exactly on the arm zeros x_1 .. x_{N+3} (stokes_lines.arm, continued by
arm_long), and F_N = 0 at j = 0, by Newton from x_1 - (x_2 - x_1), predicts x_0.

Functions (stokes_lines): P1 (the #214 arc family at w = 1.10), P2 (the segment), C = P1 (x - x0 - d)/(x - x0)
with d = 1 and C0 with d = 0 (must not fire). Taylor series to n = NSER at 160 digits; zeros in
Omega = {|x| <= 46, Re x >= 2} as grid minima (h = 0.2) of |S| in arb at 640 bits, refined by Newton in arb;
completeness by the argument principle on the boundary of Omega.

    cd solver/harmony/stokes
    python3 stokes_series.py selftest [--mutate=nolog]    the fitter on an arm of a known order-3 law
    python3 stokes_series.py zeros --nser=200 --out=FILE  zeros of P1, P2, C, C0 in Omega, with the winding count
    python3 stokes_series.py checks --zeros=F200 --zeros2=F300 [--out=FILE]
                                                           arm zeros at 200 against 300 terms; S by quadrature
                                                           (P1) and Kummer functions (P2) at the outermost zero
    python3 stokes_series.py fit --zeros=FILE [--out=FILE]  the pre-registered test at X = 30, 35, 40
Exit codes: selftest 0 pass, 1 fail; zeros 0 if every winding count matches, 1 if not; checks 0/1; fit 0 if some
X is decisive, 3 if none ("no X <= 40 suffices"), 1 if C0 fires.
"""
import json
import math
import sys
import time

import mpmath as mp
import numpy as np
from flint import acb, acb_poly, arb, ctx

import stokes_lines as SL

mp.mp.dps = 60
ctx.prec = 640
R_OMEGA, RE_MIN, H = 46.0, 2.0, 0.2
XS = (30, 35, 40)
TOL_E, D_CTRL = 0.15, mp.mpf(1)
args = dict(a.lstrip("-").split("=", 1) if "=" in a else (a.lstrip("-"), "1") for a in sys.argv[2:])
MUTATE = args.get("mutate")


# ---- the law of order N ---------------------------------------------------------------------------------------
def law_fit(pts, js, N):
    """(Lam, q, L, c_1..c_N) solving F_N(x_i) = 2 pi i j_i exactly; N + 3 points"""
    rows = []
    for x in pts:
        r = [x, mp.log(x) if MUTATE != "nolog" else mp.mpc(0), mp.mpc(-1)] + [x ** (-k) for k in range(1, N + 1)]
        if MUTATE == "nolog":                       # control: the q ln x column replaced by x^(-N-1)
            r[1] = x ** (-N - 1)
        rows.append(r)
    sol = mp.lu_solve(mp.matrix(rows), mp.matrix([2j * mp.pi * j for j in js]))
    return list(sol)


def law_F(par, x, j=0):
    lam, q, L, cs = par[0], par[1], par[2], par[3:]
    lnx = mp.log(x) if MUTATE != "nolog" else x ** (-len(cs) - 1)
    return lam * x + q * lnx - L + mp.fsum(c * x ** (-k) for k, c in enumerate(cs, 1)) - 2j * mp.pi * j


def law_dF(par, x):
    lam, q, cs = par[0], par[1], par[3:]
    dln = 1 / x if MUTATE != "nolog" else (-len(cs) - 1) * x ** (-len(cs) - 2)
    return lam + q * dln - mp.fsum(k * c * x ** (-k - 1) for k, c in enumerate(cs, 1))


def law_solve(par, x, j=0):
    for _ in range(200):
        dx = law_F(par, x, j) / law_dF(par, x)
        x -= dx
        if abs(dx) < mp.mpf(10) ** -40:
            return x
    return x


def predict(arm, N):
    """x0_hat from the law of order N fitted on arm[1..N+3]"""
    pts = arm[1:N + 4]
    par = law_fit(pts, list(range(1, N + 4)), N)
    return law_solve(par, arm[1] - (arm[2] - arm[1])), par


# ---- selftest: a known order-3 law --------------------------------------------------------------------------
def selftest():
    # planted law with P1's scale: q = -(be - al), spacing about 5 near |x| = 13; its own code, not law_F
    lam, q = mp.mpc("0.93", "1.36"), mp.mpc("-7.3", "-1.4")
    cs = [mp.mpc(3, -2), mp.mpc(-5, 4), mp.mpc(8, 1)]
    x1 = mp.mpc("13.5", "0.2")
    G = lambda x: lam * x + q * mp.log(x) + mp.fsum(c * x ** (-k) for k, c in enumerate(cs, 1))
    dG = lambda x: lam + q / x - mp.fsum(k * c * x ** (-k - 1) for k, c in enumerate(cs, 1))
    L = G(x1) - 2j * mp.pi

    def root(x, j):
        for _ in range(200):
            dx = (G(x) - L - 2j * mp.pi * j) / dG(x)
            x -= dx
            if abs(dx) < mp.mpf(10) ** -50:
                break
        return x
    xs = [mp.mpc(0)] * 8
    xs[1] = root(x1, 1)
    for j in range(2, 8):                           # continuation outward, then inward to x0
        xs[j] = root(xs[j - 1] + 2j * mp.pi / dG(xs[j - 1]), j)
    xs[0] = root(xs[1] - 2j * mp.pi / dG(xs[1]), 0)
    res = {}
    for N in (2, 3):
        xh, _ = predict(xs, N)
        res[N] = float(abs(xh - xs[0]))
    pts = xs[1:7]
    bad = law_fit(pts, [1, 2, 3, 4, 5, 7], 3)       # x_6 labelled 7
    res["label"] = float(abs(law_solve(bad, xs[1] - (xs[2] - xs[1])) - xs[0]))
    ok = res[3] < 1e-20 and res[2] > 1e-6 and res["label"] > 0.1
    print("selftest: planted order-3 arm x0..x7 = " + ", ".join(SL.Z.cnum(x, 3) for x in xs))
    print(f"  N = 3 on x1..x6: |x0_hat - x0| = {res[3]:.2e} (must be < 1e-20); N = 2 on x1..x5: {res[2]:.2e} "
          f"(must be > 1e-6); x6 labelled 7: {res['label']:.2e} (must be > 0.1); "
          f"{'PASS' if ok else 'FAIL'}{' (mutate=' + MUTATE + ')' if MUTATE else ''}")
    return ok


# ---- functions and zeros ----------------------------------------------------------------------------------
def to_acb(v):
    return acb(arb(mp.nstr(mp.re(v), 170)), arb(mp.nstr(mp.im(v), 170)))


def to_mp(a):
    a = a.mid()
    return mp.mpc(mp.mpf(a.real.mid().str(70, radius=False)), mp.mpf(a.imag.mid().str(70, radius=False)))


def functions(nser):
    """P1, P2, C (d = 1), C0 (d = 0) as Series with nser + 1 coefficients; C needs P1's x0, found first"""
    SL.NSER = nser
    with mp.workdps(SL.SDPS):
        s1, s2 = SL.p1(), SL.p2()
    return s1, s2


def poly_of(s):
    with mp.workdps(SL.SDPS):
        a = [s.c[n] / mp.factorial(n) for n in range(len(s.c))]
    return acb_poly([to_acb(v) for v in a])


def in_omega(z):
    return abs(z) <= R_OMEGA and z.real >= RE_MIN


def zeros_omega(P, label):
    """grid minima of |P| on Omega refined by Newton in arb; the winding of P on the boundary of Omega"""
    t0 = time.time()
    DP = P.derivative()
    xs = np.arange(RE_MIN - H, R_OMEGA + 2 * H, H)
    ys = np.arange(-R_OMEGA - H, R_OMEGA + 2 * H, H)
    F = np.full((len(ys), len(xs)), np.inf)
    for i, y in enumerate(ys):
        for k, x in enumerate(xs):
            F[i, k] = abs(complex(P(acb(float(x), float(y)))))
    found = []
    nmin = 0
    for i in range(1, len(ys) - 1):
        for k in range(1, len(xs) - 1):
            if F[i, k] > F[i - 1:i + 2, k - 1:k + 2].min():
                continue
            nmin += 1
            x = acb(float(xs[k]), float(ys[i]))
            conv = False
            for _ in range(80):
                dx = P(x) / DP(x)
                x = (x - dx).mid()
                if abs(complex(dx)) < 1e-50 * max(1.0, abs(complex(x))):
                    conv = True
                    break
            z = complex(x)
            if not conv or not in_omega(z):
                continue
            if all(abs(z - complex(w)) > 1e-10 for w in found):
                found.append(x)
    wind, hits, nev = winding(P)
    print(f"  {label}: {len(found)} zeros in Omega from {nmin} grid minima; winding on the boundary {wind} "
          f"({nev} evaluations, {hits} refinement-limit hits); {time.time() - t0:.0f} s", flush=True)
    return sorted(found, key=lambda a: abs(complex(a))), wind, hits


def winding(P):
    """change of arg P along the boundary of Omega (segment Re x = 2, then the arc), steps refined to <= pi/8"""
    yb = math.sqrt(R_OMEGA ** 2 - RE_MIN ** 2)
    thb = math.atan2(yb, RE_MIN)

    def pt(t):                                       # t in [0, 2]: segment for t < 1, arc for t >= 1
        if t < 1:
            return complex(RE_MIN, yb - 2 * yb * t)
        th = -thb + 2 * thb * (t - 1)
        return R_OMEGA * complex(math.cos(th), math.sin(th))
    cache = {}

    def ang(t):
        if t not in cache:
            z = pt(t)
            cache[t] = float(np.angle(complex(P(acb(z.real, z.imag)))))
        return cache[t]
    hits = 0
    tot = 0.0

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
    for a, b in zip(ts[:-1], ts[1:]):
        tot += step(float(a), float(b), 0)
    return int(round(tot / (2 * math.pi))), hits, len(cache)


def cmd_zeros():
    nser = int(args.get("nser", "200"))
    s1, s2 = functions(nser)
    out = {"nser": nser, "R": R_OMEGA, "re_min": RE_MIN, "h": H, "functions": {}}
    ok = True
    P1 = poly_of(s1)
    z1, w1, h1 = zeros_omega(P1, s1.name)
    arm1 = SL.arm([to_mp(z) for z in z1])
    x0 = arm1[0]
    print(f"  P1 x0 = {mp.nstr(x0, 20)}", flush=True)
    todo = [("P1", s1, P1, (z1, w1, h1)), ("P2", s2, poly_of(s2), None)]
    for nm, d in (("C", D_CTRL), ("C0", mp.mpf(0))):
        sc = SL.displaced(s1, x0, d)
        todo.append((nm, sc, poly_of(sc), None))
    for nm, s, P, pre in todo:
        zs, w, hits = pre if pre else zeros_omega(P, s.name)
        ok &= (w == len(zs)) and hits == 0
        out["functions"][nm] = {"name": s.name, "winding": w, "hits": hits,
                                "zeros": [[mp.nstr(mp.re(to_mp(z)), 50), mp.nstr(mp.im(to_mp(z)), 50)] for z in zs]}
    out["P1_x0"] = [mp.nstr(mp.re(x0), 50), mp.nstr(mp.im(x0), 50)]
    json.dump(out, open(args["out"], "w"), indent=1)
    print("zeros: every winding count equals the zeros found" if ok else "zeros: a winding count differs: FAIL")
    return ok


def load_zeros(path):
    d = json.load(open(path))
    return d, {k: [mp.mpc(mp.mpf(a), mp.mpf(b)) for a, b in v["zeros"]] for k, v in d["functions"].items()}


def arm_of(zs, n=12):
    """#357's arm, continued (arm_long); with the distance of each pick from its target"""
    xs = SL.arm_long(zs, n)
    return xs


def n_within(arm, X):
    n = 0
    for x in arm[1:]:
        if abs(x) > X:
            break
        n += 1
    return n


# ---- the pre-registered test --------------------------------------------------------------------------------
def cmd_fit():
    d, Zs = load_zeros(args["zeros"])
    arms = {k: arm_of(v) for k, v in Zs.items()}
    x0 = arms["P1"][0]
    for k, a in arms.items():
        print(f"{k}: arm " + ", ".join(f"x{j} {SL.Z.cnum(x, 4)} (|x| {float(abs(x)):.2f})" for j, x in enumerate(a)))
    same = max(float(abs(a - b)) for a, b in zip(arms["P1"][1:], arms["C"][1:]))
    print(f"C's arm beyond x0 against P1's: largest difference {same:.1e}; C's x0 - P1's x0 = "
          f"{SL.Z.cnum(arms['C'][0] - x0, 6)}")
    out = {"zeros": args["zeros"], "nser": d["nser"], "X": {}}
    first = None
    c0_fires = False
    for X in XS:
        n = n_within(arms["P1"], X)
        N = n - 3
        row = {"n": n, "N": N}
        if N < 0:
            print(f"X = {X}: {n} arm zeros, no law")
            continue
        es = {}
        for k in ("P1", "P2", "C", "C0"):
            a = arms[k]
            if len(a) < N + 4:
                es[k] = None
                continue
            xh, par = predict(a, N)
            tgt = a[0]
            es[k] = dict(x0_hat=xh, e=float(abs(xh - tgt)), outer=float(abs(a[N + 3])), par=par)
        e1, e2 = es["P1"]["e"], es["P2"]["e"] if es["P2"] else None
        ctrl = float(abs(es["C"]["x0_hat"] - (x0 + D_CTRL)))
        ctrl0 = float(abs(es["C0"]["x0_hat"] - x0))
        fires = ctrl > 3 * e1
        fires0 = ctrl0 > 3 * e1
        c0_fires |= fires0
        dec = e2 is not None and e1 <= TOL_E and e2 <= TOL_E and fires
        if dec and first is None:
            first = X
        print(f"X = {X}: n = {n}, N = {N} | P1 e1 = {e1:.4f} (x0_hat {SL.Z.cnum(es['P1']['x0_hat'], 4)}, "
              f"outer |x| {es['P1']['outer']:.2f}) | P2 e2 = {e2:.4f} (outer |x| {es['P2']['outer']:.2f}) | "
              f"C: |x0_hat - (x0 + d)| = {ctrl:.4f} vs 3 e1 = {3 * e1:.4f}, {'fires' if fires else 'does not fire'}"
              f" | C0 {'fires (wrong)' if fires0 else 'does not fire'} | decisive: {dec}")
        lam = es["P1"]["par"]
        print(f"   P1 fit: Lam = {SL.Z.cnum(lam[0], 5)}, q = {SL.Z.cnum(lam[1], 4)}, c = "
              + ", ".join(SL.Z.cnum(c, 3) for c in lam[3:]))
        row.update({k: (None if v is None else {"x0_hat": str(v["x0_hat"]), "e": v["e"], "outer_abs": v["outer"],
                                                "par": [str(p) for p in v["par"]]}) for k, v in es.items()})
        row.update(control=ctrl, fires=fires, control_d0=ctrl0, fires_d0=fires0, decisive=dec)
        out["X"][str(X)] = row
    # post hoc, not part of the test: every order at each X
    print("\npost hoc: e at x0 by order (law fitted on x1..x_{N+3})")
    ph = {}
    for k in ("P1", "P2"):
        a = arms[k]
        ph[k] = []
        for N in range(0, len(a) - 3):
            xh, _ = predict(a, N)
            ph[k].append({"N": N, "outer_abs": float(abs(a[N + 3])), "e": float(abs(xh - a[0]))})
        print(f"  {k}: " + "; ".join(f"N {r['N']} (|x_{r['N'] + 3}| {r['outer_abs']:.1f}) e {r['e']:.4f}"
                                     for r in ph[k]))
    out["post_hoc"] = ph
    out["first_decisive_X"] = first
    out["C0_fires"] = c0_fires
    if "out" in args:
        json.dump(out, open(args["out"], "w"), indent=1)
    verdict = f"decisive at X = {first}" if first else "no X <= 40 suffices"
    print(f"\nverdict: {verdict}; C0 (d = 0) {'fires (wrong)' if c0_fires else 'never fires'}")
    if c0_fires:
        return 1
    return 0 if first else 3


# ---- checks 3 and 4 ---------------------------------------------------------------------------------------
def cmd_checks():
    _, Za = load_zeros(args["zeros"])
    _, Zb = load_zeros(args["zeros2"])
    ok = True
    out = {}
    for k in ("P1", "P2", "C", "C0"):
        a, b = arm_of(Za[k]), arm_of(Zb[k])
        n = n_within(arm_of(Za["P1"]), max(XS)) + 1
        dif = [float(abs(u - v)) for u, v in zip(a[:n], b[:n])]
        good = len(b) >= n and max(dif) < 1e-8
        ok &= good
        out[k] = {"shift_200_300": dif}
        print(f"  {k}: arm zeros x0..x{n - 1}, 200 against 300 terms: largest shift {max(dif):.1e} "
              f"(must be < 1e-8) {'OK' if good else 'BAD'}", flush=True)
    s1, s2 = functions(int(json.load(open(args["zeros"]))["nser"]))
    for k, s in (("P1", s1), ("P2", s2)):
        a = arm_of(Za[k])
        n = n_within(arm_of(Za["P1"]), max(XS))
        x = a[n]
        with mp.workdps(SL.SDPS):
            P = poly_of(s)
            der = to_mp(P.derivative()(to_acb(x)))
        t0 = time.time()
        if k == "P1":
            with mp.workdps(80):
                Sx = quad_arc(x)
                Sy = quad_arc(x + mp.mpf("0.5"))
        else:
            Sx = kummer(x)
            Sy = kummer(x + mp.mpf("0.5"))
        ser_y = to_mp(P(to_acb(x + mp.mpf("0.5"))))
        r = float(abs(Sx) / abs(der))
        ry = float(abs(Sy / ser_y - 1))
        good = r < 1e-8 and ry < 1e-8
        ok &= good
        out[k + "_independent"] = {"x": str(x), "S_over_dS": r, "rel_diff_at_x_plus_0.5": ry}
        print(f"  {k}: at x{n} = {SL.Z.cnum(x, 4)}: |S|/|S'| by {'quadrature' if k == 'P1' else 'Kummer'} "
              f"{r:.1e} (must be < 1e-8); at x + 0.5 the two routes differ by {ry:.1e} "
              f"{'OK' if good else 'BAD'} ({time.time() - t0:.0f} s)", flush=True)
    if "out" in args:
        json.dump(out, open(args["out"], "w"), indent=1)
    print("checks: PASS" if ok else "checks: FAIL")
    return ok


def quad_arc(x):
    """P1's S(x) by tanh-sinh quadrature of the arc integral, on 16 subintervals"""
    b = SL.arc_family.ArcBase()
    p = b.params(mp.mpf("1.10"))
    za, zp, kap, al, be, e0, e1 = [p[k] for k in SL.arc_family.NAMES]
    rho = lambda s: s ** (al - 1) * (1 - s) ** (be - 1) * (e0 + e1 * s)
    gam = lambda s: za + s * (zp - za) + kap * s * (1 - s)
    nodes = [mp.mpf(i) / 16 for i in range(17)]
    num = mp.quad(lambda s: mp.exp(x * gam(s)) * rho(s), nodes)
    den = mp.quad(rho, [0, mp.mpf(1) / 2, 1])
    return num / den


def kummer(x):
    """P2's S(x) by Kummer functions (as stokes_lines.p2_checks)"""
    z, k, a, b, g = [SL.P2_PAR[n] for n in ("zeta", "kappa", "alpha", "beta", "g")]
    lam = k - z
    with mp.workdps(80):
        v = mp.exp(z * x) * (mp.beta(a, b - 1) * mp.hyp1f1(a, a + b - 1, lam * x)
                             + g * lam * mp.beta(a + 1, b - 1) * mp.hyp1f1(a + 1, a + b, lam * x))
        return v / (mp.beta(a, b - 1) + g * lam * mp.beta(a + 1, b - 1))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "selftest":
        sys.exit(0 if selftest() else 1)
    if cmd == "zeros":
        sys.exit(0 if cmd_zeros() else 1)
    if cmd == "checks":
        sys.exit(0 if cmd_checks() else 1)
    if cmd == "fit":
        sys.exit(cmd_fit())
    raise SystemExit(__doc__)

#!/usr/bin/env python3
"""Issue #538: do the variables B = 1/a^2 (grad r . grad r) and C = c a^2 remove the horizon series' pole of
notes/QUAD_DOUBLE.md 13.6? At each stored state the horizon datum a_h is scaled by 1 + delta (13.6's perturbation,
hp/pole_test.py), the kernel's double-double horizon series is recomputed, and B, C and b = a^2 (g_rr) are formed from
its a and c coefficients by series arithmetic (python-flint arb_series at PREC bits, Cauchy products and the series
inverse), lane by lane in tau, with no kernel change. In these variables the equations are x B_x = (n-1)(1-B) -
B(p^2+m^2), x C_x = C(p^2+m^2), c = C B: a pole of b at an apparent horizon (B = 0) is a simple zero of B.

Radius estimator (one variable f at a time): 13.6's late-order ratio test, t_k = max over tau of |f_k| u0^k and
R = u0 (t_{K-W} / t_K)^(1/W), with W = min(NR, K // 2). For a derived variable the rounding of a, amplified by the pole
that B and C cancel, can exceed |B_k|, so K is the last order before the first run of three orders with t_k below
FLOOR_MARGIN times the noise floor. The floor is the larger of (1) the difference between the two horizon_alpha modes
of the kernel (0: the series in a, 13.6's; 1: in a - 1, the same solution with other rounding) and (2) the change of
the derived coefficients when every a and c coefficient is perturbed by a relative 2^-104 (the double-double unit). K
below KMIN: not read. The ± of R is an estimator range, the min and max over K' in {K, K-5, K-10} and W' in {W, 3W/4,
W/2}. R_new = min(R_p, R_m, R_B, R_C), the radius a series in the new variables would have.

13.6's own estimator (pole_test.ratio_radius, the max over p, m, a, c at the last order) is recomputed as R136 and
checked against route1-mac/pole_test.jsonl (--ref). Per state, R = R0 + b ln(1/delta) is fitted on 13.6's grid (1e-8,
1e-6, 1e-4, 1e-2), C = b (n - 1), and the log-slope s = d ln R / d ln(1/delta) on the full grid, ± its standard error
(fit spread). Where: x - c and B summed on the real u axis up to 0.95 R per lane, their first zeros (min over lanes).

Pass (#538): |s(R_new)| < 0.1 s(R136) at every state, and R_new > 0.3 for delta <= 1e-4 at D 73.27.
Controls (#538): (1) a alone gives C within 10% of 0.88 at every state of D 40-73; (2) --mutate=b puts b = a^2 in B's
place (no inverse) and must restore 13.6's law, C(B) within 10% of 0.88.
Planted truth (`planted`, written before the real states): per lane B = Bh (1 - u/u_p) g(u), with g's complex pair at
|u| = 0.55 and C with a real pole at 0.45, a = B^(-1/2) and c = C B rounded to double-double with two independent
relative noises of 2^-104 standing for the two modes. Pass: R(a), R(b) within 5% of u_p; R(B) within 10% of 0.55 and
R(C) within 10% of 0.45 wherever read, read at u_p >= 0.14; B's first real zero within 1% of u_p where it lies inside
0.95 R(B). With --mutate=b the same checks must fail (B's radius falls to u_p). Exit 1 on a failed check.

    cd hp && python3 pole_bc.py planted [--mutate=b]
    cd hp && python3 pole_bc.py OUT.jsonl STATE.npz [STATE.npz ...] [--mutate=b] [--threads N] [--ref pole_test.jsonl] [--ks N]
"""
import json
import math
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402
from flint import arb, arb_series, ctx  # noqa: E402

PREC = 256
NR = 40                     # 13.6's ratio-test window
NDS = 60                    # orders in the Domb-Sykes fit (pole_test.py)
KMIN = 30                   # fewest reliable orders for a reading
FLOOR_MARGIN = 100.0        # signal over noise floor for an order to count
ETA = 2.0 ** -104           # the double-double unit, the relative perturbation of floor (2)
DELTAS = (0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
GRID136 = (1e-8, 1e-6, 1e-4, 1e-2)
C136 = 0.88
FIELDS = ("p", "m", "a", "c")
DERIVED = ("B", "C", "b")
NEW = ("p", "m", "B", "C")
ctx.prec = PREC


# ------------------------------------------------------------------ series arithmetic
def lanes_of(h, l, q):
    """field q of a kernel series (nk, 4, M), hi and lo parts -> per lane, its coefficients as exact arb sums"""
    nk, _, M = h.shape
    return [[arb(float(h[k, q, j])) + arb(float(l[k, q, j])) for k in range(nk)] for j in range(M)]


def pad(cs, nk):
    """arb_series.coeffs() drops trailing zeros"""
    return list(cs) + [arb(0)] * (nk - len(cs))


def derive(a, c, nk, mutate="", noise=None):
    """B = 1/a^2, C = c a^2, b = a^2 per lane, in the kernel's scaled variable (products of scaled series are scaled).
    mutate='b': B := a^2. noise: (rng, eta) multiplies every a and c coefficient by 1 + eta N(0, 1) first."""
    ctx.cap = nk                # arb_series truncates every product and inverse to ctx.cap terms (default 10)
    out = {v: [] for v in DERIVED}
    for aj, cj in zip(a, c):
        if noise is not None:
            rng, eta = noise
            # x (1 + eta g) as x + x (eta g): 1 + eta g would round to 1 in f64 at eta = 2^-104
            aj = [x + x * arb(eta * g) for x, g in zip(aj, rng.standard_normal(nk))]
            cj = [x + x * arb(eta * g) for x, g in zip(cj, rng.standard_normal(nk))]
        sa = arb_series(aj, prec=nk)
        sc = arb_series(cj, prec=nk)
        a2 = sa * sa
        out["b"].append(pad(a2.coeffs(), nk))
        out["B"].append(pad((a2 if mutate == "b" else a2.inv()).coeffs(), nk))
        out["C"].append(pad((sc * a2).coeffs(), nk))
    return out


def tmax(lanes, w):
    """t_k = max over lanes |f_k| w_k, w_k = (u0 / sigma)^k"""
    nk = len(w)
    t = np.zeros(nk)
    for cs in lanes:
        t = np.maximum(t, np.array([abs(float(x.mid())) for x in cs[:nk]]))
    return t * w


def dmax(l1, l2, w):
    """max over lanes |f1_k - f2_k| w_k (the noise floor between two computations of one variable)"""
    nk = len(w)
    t = np.zeros(nk)
    for c1, c2 in zip(l1, l2):
        t = np.maximum(t, np.array([abs(float((x - y).mid())) for x, y in zip(c1[:nk], c2[:nk])]))
    return t * w


# ------------------------------------------------------------------ the radius
def radius(t, floor, u0):
    """the ratio test on the reliable orders: dict(R, K, W, lo, hi); R None when fewer than KMIN orders are reliable"""
    ok = np.isfinite(t) & (t > 0)
    if floor is not None:
        ok &= t > FLOOR_MARGIN * np.nan_to_num(floor, nan=np.inf)
    nk = len(t)
    kend = nk
    for k in range(5, nk - 2):
        if not (ok[k] or ok[k + 1] or ok[k + 2]):
            kend = k
            break
    idx = np.flatnonzero(ok[:kend])
    if len(idx) == 0:
        return dict(R=None, K=None, W=None, lo=None, hi=None)
    K = int(idx[-1])
    if K < KMIN:
        return dict(R=None, K=K, W=None, lo=None, hi=None)
    W = min(NR, K // 2)

    def est(K1, W1):
        if W1 < 5 or K1 - W1 < 0 or not (ok[K1] and ok[K1 - W1]):
            return None
        return float(u0 / (t[K1] / t[K1 - W1]) ** (1.0 / W1))

    R = est(K, W)
    rs = [r for r in (est(K - dk, w1) for dk in (0, 5, 10) for w1 in (W, (3 * W) // 4, W // 2)) if r is not None]
    return dict(R=R, K=K, W=W, lo=min(rs) if rs else None, hi=max(rs) if rs else None)


def first_zero(lanes, sig, umax, shift=None, skip=0.0):
    """the first real zero in (skip, umax] over lanes of f(u) (+ shift(u)), the series summed in u / sigma, and the
    largest value of f at umax over lanes: (u_zero or None, f_end)"""
    us = np.linspace(0.0, umax, 200)[1:]
    best, fend = None, -np.inf
    for cs in lanes:
        co = np.array([float(x.mid()) for x in cs])
        v = np.polyval(co[::-1], us / sig)
        if shift is not None:
            v = v + shift(us)
        fend = max(fend, float(v[-1]))
        sgn = np.sign(v)
        start = np.flatnonzero(us > skip)
        if len(start) == 0:
            continue
        i0 = start[0]
        ch = np.flatnonzero(sgn[i0 + 1:] != sgn[i0])
        if len(ch):
            j = i0 + 1 + ch[0]
            # linear interpolation between the grid points around the sign change
            u = us[j - 1] + (us[j] - us[j - 1]) * v[j - 1] / (v[j - 1] - v[j])
            best = u if best is None else min(best, u)
    return best, fend


def fits(rows, n, key):
    """R = R0 + b ln(1/delta) on 13.6's grid (C = b (n - 1)) and ln R = c0 + s ln(1/delta) on the full grid, s ± se"""
    out = {}
    pts = [(math.log(1 / r["delta"]), r[key]) for r in rows if r["delta"] > 0 and r[key] is not None]
    p136 = [(x, y) for x, y in pts if any(abs(x - math.log(1 / d)) < 1e-9 for d in GRID136)]
    if len(p136) >= 3:
        x, y = map(np.array, zip(*p136))
        b, a0 = np.polyfit(x, y, 1)
        out.update(R0=float(a0), b=float(b), C=float(b * (n - 1)), resid=float(np.max(np.abs(np.polyval([b, a0], x) - y))))
    if len(pts) >= 3 and all(y > 0 for _, y in pts):
        x, y = map(np.array, zip(*pts))
        ly = np.log(y)
        (s, c0), cov = np.polyfit(x, ly, 1, cov=True) if len(pts) > 3 else (np.polyfit(x, ly, 1), np.zeros((2, 2)))
        out.update(s=float(s), s_se=float(math.sqrt(max(cov[0, 0], 0.0))), npts=len(pts))
    return out


# ------------------------------------------------------------------ one perturbed state
def analyse(f0, f1, nk, u0, sig, mutate, rng, n=None):
    """f0, f1: dict of field lanes (p, m, a, c) from the two modes. Returns per-variable radii and the where-test."""
    w = (u0 / sig) ** np.arange(nk)
    d0 = derive(f0["a"], f0["c"], nk, mutate)
    d1 = derive(f1["a"], f1["c"], nk, mutate)
    dn = derive(f0["a"], f0["c"], nk, mutate, noise=(rng, ETA))
    rec = {}
    for v in FIELDS + DERIVED:
        l0 = f0[v] if v in FIELDS else d0[v]
        l1 = f1[v] if v in FIELDS else d1[v]
        t = tmax(l0, w)
        fl = dmax(l0, l1, w)
        if v in DERIVED:
            fl = np.maximum(fl, dmax(l0, dn[v], w))
        rec[v] = radius(t, fl, u0)
    rs = [rec[v]["R"] for v in NEW]
    rec["new"] = dict(R=min(rs) if all(r is not None for r in rs) else None)
    rec["where"] = where_test(f0, d0, rec, sig, n) if n else {}
    if rec["B"]["K"]:
        mean = np.array([np.mean([float(cs[k].mid()) for cs in d0["B"]]) for k in range(rec["B"]["K"] + 1)])
        rec["B"]["ds"] = domb_sykes(mean, sig)
    return rec, d0


def domb_sykes(mean, sig):
    """pole_test.domb_sykes on a tau-mean in the scaled variable: rho_k = A + B/k over the last NDS orders"""
    rho = mean[1:] / mean[:-1] / sig
    ks = np.arange(1, len(mean))
    ok = np.isfinite(rho) & (mean[1:] != 0) & (mean[:-1] != 0)
    ks, rho = ks[ok][-NDS:], rho[ok][-NDS:]
    if len(ks) < 10:
        return None
    Bf, A = np.polyfit(1.0 / ks, rho, 1)
    return dict(R_ds=float(1 / A) if A else None, alpha=float(1 + Bf / A) if A else None, neg=float(np.mean(rho < 0)),
                orders=[int(ks[0]), int(ks[-1])])


def sum_at(cs, u, sig, K):
    """f(u) and f'(u) from the orders 0..K of a scaled series"""
    co = np.array([float(x.mid()) for x in cs[:K + 1]])
    s = u / sig
    return float(np.polyval(co[::-1], s)), float(np.polyval((np.arange(len(co)) * co)[1:][::-1], s) / sig)


def where_test(f0, d0, rec, sig, n):
    """B and x - c on the real u axis at u1 = R min(0.95, 1 - 5/K) of each variable (orders 0..K summed), per lane:
    the first zero inside, the value closest to zero at u1, and the zero each extrapolates to with the mass mode,
    1 - B and the growing part of x - c taken proportional to x^-(n-1) = (1 - u)^-(n-1) from u1 on. conv: the change
    of the sum at u1 when its last 20 orders are dropped (the summation's own error)."""
    out = {}
    for v, lanes, shift in (("B", d0["B"], None), ("xc", f0["c"], True)):
        r = rec["B"] if v == "B" else rec["c"]
        if not r["R"] or not r["K"]:
            continue
        K = r["K"]
        u1 = r["R"] * min(0.95, 1 - 5 / K)
        uz, _ = first_zero([cs[:K + 1] for cs in lanes], sig, u1, shift=(lambda u: -(1.0 - u)) if shift else None,
                           skip=0.02 * u1 if shift else 0.0)
        best, ext, conv = None, None, 0.0
        for cs in lanes:
            f, df = sum_at(cs, u1, sig, K)
            f20, _ = sum_at(cs, u1, sig, K - 20)
            conv = max(conv, abs(f - f20))
            if shift:
                g, dg = (1.0 - u1) - f, -1.0 - df          # x - c and its u-derivative
                best = g if best is None else max(best, g)
                # g = A + beta (1 - u)^-(n-1): its zero beyond u1
                beta = dg * (1 - u1) ** n / (n - 1)
                A = g - beta * (1 - u1) ** -(n - 1)
                if A < 0 < beta:
                    e = 1 - (-A / beta) ** (-1.0 / (n - 1))
                    ext = e if ext is None else min(ext, e)
            else:
                best = f if best is None else min(best, f)
                if 0 < f < 1:
                    e = 1 - (1 - u1) * (1 - f) ** (1.0 / (n - 1))
                    ext = e if ext is None else min(ext, e)
        key = "AH" if v == "B" else "sonic"
        out.update({f"u1_{v}": u1, f"{v}_at_u1": best, f"u_{key}_zero": uz, f"u_{key}_extrap": ext, f"conv_{v}": conv})
    return out


# ------------------------------------------------------------------ planted truth
def planted(mutate):
    """the estimator on a series with a known zero of B (pole of b) and known other singularities"""
    nk, M, sig, u0 = 131, 6, 0.25, 0.30
    ctx.cap = nk
    Rg, Rc, Bh, gam = 0.55, 0.45, 0.95, 0.5
    rng = np.random.default_rng(538)
    fails = []
    print(f"{'u_p':>6} {'var':>3} {'R':>7} {'range':>15} {'K':>4} {'W':>3} {'expect':>7}  verdict")
    for up0 in (0.30, 0.20, 0.14, 0.07, 0.03):
        f = [{v: [] for v in FIELDS} for _ in range(2)]
        for j in range(M):
            tau = 2 * math.pi * j / M
            up = up0 * (1 + 0.1 * (1 - math.cos(tau)))
            th = 0.5 + 0.15 * j
            lin = arb_series([arb(1), arb(-sig) / arb(up)], prec=nk)
            den = arb_series([arb(1), -2 * arb(sig) * arb(th).cos() / arb(Rg), (arb(sig) / arb(Rg)) * (arb(sig) / arb(Rg))], prec=nk)
            g = arb_series([arb(0), arb(gam) * arb(sig)], prec=nk) * den.inv() + 1
            B = lin * g * arb(Bh)
            a = B.rsqrt()
            C = arb_series([arb(0), arb(0.2) * arb(sig)], prec=nk) * arb_series([arb(1), -arb(sig) / arb(Rc)], prec=nk).inv() + 1
            c = C * B
            for mode in (0, 1):
                for v, s in (("a", a), ("c", c)):
                    cs = pad(s.coeffs(), nk)
                    # the mode's rounding: a relative 2^-104 noise, then double-double
                    cs = [x + x * arb(ETA * gg) for x, gg in zip(cs, rng.standard_normal(nk))]
                    dd = []
                    for x in cs:
                        hi = float(x.mid())
                        lo = float((x - arb(hi)).mid())
                        dd.append(arb(hi) + arb(lo))
                    f[mode][v].append(dd)
                f[mode]["p"].append([arb(0)] * nk)
                f[mode]["m"].append([arb(0)] * nk)
        rec, _ = analyse(f[0], f[1], nk, u0, sig, mutate, rng, n=38.0)
        exp = dict(a=up0, b=up0, B=Rg, C=Rc, c=min(Rg, Rc))
        for v in ("a", "b", "B", "C", "c"):
            r = rec[v]
            tol = 0.05 if v in ("a", "b") else 0.10
            if r["R"] is None:
                verdict = "not read"
                if v in ("a", "b") or (v in ("B", "C") and up0 >= 0.14):
                    verdict += " FAIL"
                    fails.append((up0, v, None))
            else:
                good = abs(r["R"] - exp[v]) <= tol * exp[v]
                verdict = "OK" if good else "FAIL"
                if not good and v != "c":
                    fails.append((up0, v, r["R"]))
            rg = f"[{r['lo']:.3f}, {r['hi']:.3f}]" if r.get("lo") else ""
            print(f"{up0:6.2f} {v:>3} {r['R'] if r['R'] else float('nan'):7.3f} {rg:>15} {r['K'] if r['K'] is not None else -1:4d} "
                  f"{r['W'] or 0:3d} {exp[v]:7.3f}  {verdict}", flush=True)
        uz = rec["where"].get("u_AH_zero")
        if uz is not None or up0 < rec["where"].get("u1_B", 0.0):
            ok = uz is not None and abs(uz - up0) <= 0.01 * up0
            print(f"{up0:6.2f} B's first real zero {uz if uz else float('nan'):.4f} against {up0:.4f}: {'OK' if ok else 'FAIL'}")
            if not ok:
                fails.append((up0, "Bzero", uz))
    if fails:
        print("planted FAILED:", fails)
        sys.exit(1)
    print("planted: all checks passed")


# ------------------------------------------------------------------ the stored states
def run_states(out, paths, mutate, threads, ref, ks_set=None):
    from hpkernel.driver import load, make_solver
    from pole_test import ratio_radius
    refR = {}
    if ref:
        for line in open(ref):
            if line.strip():
                r = json.loads(line)
                if "delta" in r and r.get("R") is not None:
                    refR[(r["state"], r["delta"])] = r["R"]
    rng = np.random.default_rng(538)
    verdict = []
    hdr = ("R: per-variable ratio test (reliable orders only); R136: 13.6's estimator; new = min(p, m, B, C); "
           "B(u1), x-c(u1): the value closest to zero over lanes at u1 ~ 0.95 R; AH, sonic: the zeros of B and x - c "
           "extrapolated with the mass mode; Rds_B: Domb-Sykes of B's tau-mean")
    print(f"{'D':>8} {'delta':>7} {'R136':>6} {'a':>6} {'b':>6} {'B':>6} {'C':>6} {'p':>6} {'m':>6} {'c':>6} {'new':>6} "
          f"{'K_B':>4} {'B(u1)':>6} {'xc(u1)':>7} {'AH':>6} {'sonic':>6} {'Rds_B':>6}   ({hdr})", flush=True)
    with open(out, "a") as fo:
        for path in paths:
            zh, zl, meta = load(path)
            D, M, nodes = meta["D"], meta["M"], list(meta["nodes"])
            kc, ks, kt = meta.get("kc", 146), ks_set or meta.get("ks", 130), meta.get("kt", 36)
            K = len(nodes)
            u0 = 1.0 - nodes[-1]
            n = D - 2
            kf = meta.get("kfreeze", 13 if D >= 35 else None)
            sv = []
            for ha in (0, 1):
                s = make_solver(D, M, nodes=nodes, kt=kt, kc=kc, ks=ks, threads=threads, kfreeze=kf, horizon_alpha=ha)
                s.set_segs(meta["segs"] if "segs" in meta else s.plan(zh, zl, 1e-32, 0.05))
                sv.append(s)
            sig = sv[0].series_scale(1)
            oa = 3 * M + 4 * M * K
            rows = []
            name = os.path.basename(path)
            for delta in DELTAS:
                zh2, zl2 = zh.copy(), zl.copy()
                zh2[oa:oa + M] *= 1.0 + delta
                zl2[oa:oa + M] *= 1.0 + delta
                ser = []
                for s in sv:
                    hh, hl = s.series(zh2, zl2, 1, True)
                    ser.append((np.asarray(hh).reshape(-1, 4, M), np.asarray(hl).reshape(-1, 4, M)))
                nk = min(len(ser[0][0]), len(ser[1][0]))
                R136, _ = ratio_radius(ser[0][0] + ser[0][1], u0, sig)
                f = [{v: lanes_of(h[:nk], l[:nk], q) for q, v in enumerate(FIELDS)} for h, l in ser]
                rec, _ = analyse(f[0], f[1], nk, u0, sig, mutate, rng, n=n)
                row = dict(state=name, D=D, n=n, u0=u0, ks=ks, nk=nk, delta=delta, mutate=mutate, R136=R136,
                           R136_ref=refR.get((name, delta)), **{v: rec[v] for v in FIELDS + DERIVED + ("new",)},
                           where=rec["where"])
                fo.write(json.dumps(row) + "\n")
                fo.flush()
                flat = dict(delta=delta, R136=R136, **{v: rec[v]["R"] for v in FIELDS + DERIVED + ("new",)})
                rows.append(flat)
                g = lambda x: f"{x:6.3f}" if x is not None else "     -"  # noqa: E731
                wz = rec["where"]
                print(f"{D:8.4f} {delta:7.0e} {g(R136)} {g(rec['a']['R'])} {g(rec['b']['R'])} {g(rec['B']['R'])} "
                      f"{g(rec['C']['R'])} {g(rec['p']['R'])} {g(rec['m']['R'])} {g(rec['c']['R'])} {g(rec['new']['R'])} "
                      f"{rec['B']['K'] if rec['B']['K'] is not None else -1:4d} {g(wz.get('B_at_u1'))} "
                      f"{wz['xc_at_u1'] if wz.get('xc_at_u1') is not None else float('nan'):+7.3f} "
                      f"{g(wz.get('u_AH_zero') or wz.get('u_AH_extrap'))} {g(wz.get('u_sonic_zero') or wz.get('u_sonic_extrap'))} "
                      f"{g((rec['B'].get('ds') or {}).get('R_ds'))}", flush=True)
                if row["R136_ref"] is not None and R136 is not None and abs(R136 - row["R136_ref"]) > 1e-3:
                    print(f"  R136 {R136:.4f} does not reproduce pole_test.jsonl {row['R136_ref']:.4f}", flush=True)
                    verdict.append((D, delta, "R136 reproduction"))
            st = dict(state=name, D=D, n=n, mutate=mutate, fits={k: fits(rows, n, k) for k in ("R136",) + FIELDS + DERIVED + ("new",)})
            fo.write(json.dumps(st) + "\n")
            fo.flush()
            fz = st["fits"]
            line = f"{D:8.4f} fits: " + "; ".join(
                f"{k} C {fz[k]['C']:.2f} s {fz[k]['s']:.4f}±{fz[k]['s_se']:.4f}" if "C" in fz[k] and "s" in fz[k]
                else (f"{k} s {fz[k]['s']:.4f}±{fz[k]['s_se']:.4f}" if "s" in fz[k] else f"{k} -")
                for k in ("R136", "a", "b", "B", "C", "p", "m", "new"))
            print(line, flush=True)
            # the controls and the pass criteria
            ref_s = fz["R136"].get("s")
            if D <= 74 and "C" in fz["a"] and abs(fz["a"]["C"] - C136) > 0.1 * C136:
                verdict.append((D, "control 1: C(a)", fz["a"]["C"]))
            if mutate == "b":
                cb = fz["B"].get("C")
                if D <= 74 and (cb is None or abs(cb - C136) > 0.1 * C136):
                    verdict.append((D, "control 2: C(B := b)", cb))
            else:
                sn = fz["new"].get("s")
                passed = sn is not None and ref_s and abs(sn) < 0.1 * ref_s
                print(f"{D:8.4f} pass test: |s(new)| {abs(sn) if sn is not None else float('nan'):.4f} against 0.1 s(R136) "
                      f"{0.1 * ref_s if ref_s else float('nan'):.4f}: {'PASS' if passed else 'FAIL'}", flush=True)
                if abs(D - 73.27) < 0.01:
                    low = [r for r in rows if 0 < r["delta"] <= 1e-4]
                    ok = all(r["new"] is not None and r["new"] > 0.3 for r in low)
                    print(f"{D:8.4f} pass test: R_new > 0.3 for delta <= 1e-4: "
                          + ", ".join(f"{r['delta']:.0e}: {r['new'] if r['new'] else float('nan'):.3f}" for r in low)
                          + f": {'PASS' if ok else 'FAIL'}", flush=True)
    if verdict:
        print("checks that did not hold:", verdict, flush=True)
        sys.exit(1)


def main():
    args = list(sys.argv[1:])
    mutate = ""
    for a in list(args):
        if a.startswith("--mutate="):
            mutate = a.split("=", 1)[1]
            args.remove(a)
    if mutate not in ("", "b"):
        raise SystemExit(f"--mutate={mutate}: only --mutate=b exists")
    threads, ref, ks_set = 2, None, None
    if "--threads" in args:
        i = args.index("--threads"); threads = int(args[i + 1]); del args[i:i + 2]
    if "--ref" in args:
        i = args.index("--ref"); ref = args[i + 1]; del args[i:i + 2]
    if "--ks" in args:
        # a diagnostic: the horizon series to N orders instead of the state's ks (R136 then differs from 13.6's)
        i = args.index("--ks"); ks_set = int(args[i + 1]); del args[i:i + 2]
    if args and args[0] == "planted":
        planted(mutate)
        return
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from raw_out import pop_allow, require_raw
    allow = pop_allow(args)
    out, paths = args[0], args[1:]
    require_raw(out, allow=allow)
    run_states(out, paths, mutate, threads, ref, ks_set)


if __name__ == "__main__":
    main()

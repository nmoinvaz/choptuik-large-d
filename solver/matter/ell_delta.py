#!/usr/bin/env python3
"""The l = 1 multiplet's growth rate against the scalar's at the same D: delta = lambda_1 - lambda_0 and its limit by
ELL.md 7's windows (#1025).

    python3 ell_delta.py planted       the window code on synthetic delta(n) = c + a/n + b/n^2 + e/n^3 with known c
    python3 ell_delta.py planted --mutate=window    a window off by one point: the exactness checks must fail
    python3 ell_delta.py collect RAWDIR   data/ell_lam_highD.json from the k_c runs: per D the highest k_c's lambda and,
                                       as lam_err, its change from the next lower k_c (RAWDIR: a checkout of
                                       choptuik-large-d-raw with I660-mac, I810-mac and I1025-mac)
    python3 ell_delta.py calib         the same windows on the scalar's own lambda_0(D), whose limit is exactly 2
                                       (GAMMA.md 5), at integer D 5 .. 20 (lambda_0 interpolated as below): how far
                                       the windows are from their limit at these n for a real sequence of this kind
    python3 ell_delta.py fold [FILE]   a fold of the branch in D from ell_leap --fold records (default
                                       data/l1kc47F1025.jsonl): D* from sigma_fold^2 = c (D* - D) on the last 3, 4, 5
                                       points, then Delta and lambda at the fold from x(D) = x* + k sqrt(D* - D) + m (D* - D)
    python3 ell_delta.py fold --planted   the same code on a synthetic fold (D* 16.37): D*, Delta* and lambda* to 1e-9;
                                       --mutate=sigma fits sigma instead of sigma^2 and must miss D* (exit 1)
    python3 ell_delta.py table [FILE]  delta, n delta and the windows from the l = 1 points in FILE
                                       (default data/ell_lam_highD.json)

Windows (as ELL.md 7 and ell_fit.py): on consecutive n, the exact interpolants c + a/n (two points) and
c + a/n + b/n^2 (three points), and c is their limit n -> infinity. In x = 1/n they are polynomial extrapolations to
x = 0. For delta = c + a x + b x^2 + e x^3 the two-point value is c - b x1 x2 - e x1 x2 (x1 + x2), and the
three-point value is c + e x1 x2 x3. planted checks the code against these to 1e-12, and with e = 0 the three-point
window must return c to 1e-12.

lambda_0(D) is not stored at the l = 1 points' D. It is interpolated in x = 1/n by the cubic through the four nearest
stored points (solver/data/scan_ne6_lowD_kc39_rk2.jsonl, then gamma_rust.jsonl). The cubic minus the quadratic through
the three nearest points is quoted as the interpolation error. ell_fit.py interpolated linearly in D.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MUT = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--mutate=")), "")


def windows(n, d, k):
    """c from the exact interpolant c + a/n + ... (k terms) on each run of k consecutive points"""
    out = []
    off = 1 if "window" in MUT else 0
    for i in range(len(n) - k + 1):
        nn = n[i:i + k]
        dd = d[i + off:i + k + off] if i + k + off <= len(n) else d[i:i + k]
        A = np.stack([nn ** -j for j in range(k)], 1)
        out.append((nn.tolist(), float(np.linalg.solve(A, dd)[0])))
    return out


def planted():
    ok = True
    n = np.arange(3.0, 19.0)                          # D 5 .. 20
    x = 1 / n
    for c, a, b, e in ((0.2, 5.6, -3.0, 0.0), (-0.003, 5.9, -2.1, 4.0), (0.15, 4.0, 6.0, -12.0)):
        d = c + a * x + b * x ** 2 + e * x ** 3
        w2, w3 = windows(n, d, 2), windows(n, d, 3)
        err2 = max(abs(cc - (c - b * x[i] * x[i + 1] - e * x[i] * x[i + 1] * (x[i] + x[i + 1]))) for i, (_, cc) in enumerate(w2))
        err3 = max(abs(cc - (c + e * x[i] * x[i + 1] * x[i + 2])) for i, (_, cc) in enumerate(w3))
        good = err2 < 1e-12 and err3 < 1e-12
        ok &= good
        print(f"c {c:+.3f}, a {a}, b {b}, e {e}: two-point windows against c - b x1 x2 - e x1 x2 (x1 + x2): max error "
              f"{err2:.1e}; three-point against c + e x1 x2 x3: {err3:.1e} (bar 1e-12: {'OK' if good else 'BAD'})")
        for lab, w in (("two-point", w2), ("three-point", w3)):
            sel = [w[0], w[2], w[4], w[len(w) // 2], w[-1]]
            print(f"   {lab}: " + "; ".join(f"n {int(q[0][0])}-{int(q[0][-1])}: {q[1]:+.5f}" for q in sel))
    print("PASS" if ok else "FAIL")
    return ok


def lam0_table():
    pts = {}
    for f in ("scan_ne6_lowD_kc39_rk2.jsonl", "gamma_rust.jsonl"):
        for line in open(os.path.join(HERE, "..", "data", f)):
            r = json.loads(line)
            if r.get("lam"):
                D = round(r["D"], 6)
                if all(abs(D - q) > 1e-6 for q in pts):
                    pts[D] = r["lam"]
    Ds = np.array(sorted(pts))
    return Ds, np.array([pts[d] for d in Ds])


def lam0_at(D, Ds, L):
    """cubic in x = 1/n through the four nearest stored points; error = cubic - quadratic (three nearest)"""
    i = np.argsort(np.abs(Ds - D))
    x0 = 1 / (D - 2)
    xs = 1 / (Ds - 2)
    p3 = np.polyfit(xs[i[:4]], L[i[:4]], 3)
    p2 = np.polyfit(xs[i[:3]], L[i[:3]], 2)
    v3, v2 = np.polyval(p3, x0), np.polyval(p2, x0)
    exact = np.min(np.abs(Ds - D)) < 1e-6
    return (float(L[i[0]]), 0.0) if exact else (float(v3), float(abs(v3 - v2)))


def table(fn):
    """the delta table against delta(inf) = lambda_inf(l = 1) - 2 (#994's M = 3 value, data/ell_lambda_l1.json), with the
    k_c bar (|lambda(top k_c) - lambda(next lower)| <= 1e-6) and the psi-constraint of every record, the D where delta
    crosses 0 (linear interpolation between the bracketing points, and the spread from moving each by its k_c change),
    and the windows"""
    rows = json.load(open(fn))["points"]
    Ds, L = lam0_table()
    l1inf = json.load(open(os.path.join(HERE, "data", "ell_lambda_l1.json")))["M3"]["lam"]
    dinf = l1inf - 2
    nn, dd, ee, DD, hh = [], [], [], [], []
    print(f"delta(inf) = lambda_inf(l = 1) - lambda_0(inf) = {l1inf:.6f} - 2 = {dinf:+.6f} (#994, M = 3; "
          f"lambda_0(inf) = 2, GAMMA.md 5)\n")
    print("| D | n | λ₁ | k_c | k_c bar: change from the next lower k_c | ψ-constraint | λ₀ (interp. error) | δ = λ₁ − λ₀ | "
          "δ − δ(∞) | n δ | n δ(∞) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda q: q["D"]):
        D, l1 = r["D"], r["lam"]
        l0, e0 = lam0_at(D, Ds, L)
        n = D - 2
        d = l1 - l0
        nn.append(n); dd.append(d); ee.append(r.get("lam_err", 0.0)); DD.append(D); hh.append(r.get("has_err", True))
        bar = (("OK" if r["lam_err"] <= 1e-6 else "BAD") + f" ({r['lam_err']:.1e})") if r.get("has_err", True) else "single k_c"
        con = r.get("constraint")
        print(f"| {D:g} | {n:g} | {l1:.7f} | {r.get('kc', '')} | {bar} | {con:.1e} | {l0:.7f} ({e0:.1e}) | {d:.6f} | "
              f"{d - dinf:+.6f} | {n * d:.4f} | {n * dinf:.4f} |" if con is not None else
              f"| {D:g} | {n:g} | {l1:.7f} | {r.get('kc', '')} | {bar} | | {l0:.7f} ({e0:.1e}) | {d:.6f} | "
              f"{d - dinf:+.6f} | {n * d:.4f} | {n * dinf:.4f} |")
    nn, dd, ee, DD = map(np.array, (nn, dd, ee, DD))
    cr = [i for i in range(len(dd) - 1) if dd[i] > 0 >= dd[i + 1]]
    if cr:
        i = cr[0]
        lin = lambda a, b: DD[i] + (DD[i + 1] - DD[i]) * a / (a - b)
        Dc = lin(dd[i], dd[i + 1])
        alts = [lin(dd[i] + s1 * ee[i], dd[i + 1] + s2 * ee[i + 1]) for s1 in (-1, 1) for s2 in (-1, 1)]
        spread = (f"k_c spread {min(alts):.4f} to {max(alts):.4f}" if hh[i] and hh[i + 1] else
                  "k_c spread not measured (single k_c at " + ", ".join(f"D {DD[j]:g}" for j in (i, i + 1) if not hh[j]) + ")")
        print(f"\ndelta crosses 0 between D {DD[i]:g} and {DD[i + 1]:g}: D = {Dc:.4f} (linear), {spread}")
    else:
        print(f"\ndelta has not crossed 0 (last D {DD[-1]:g}, delta {dd[-1]:+.6f})")
    sl = np.diff(dd) / np.diff(DD)
    print("slope of delta per unit D over the last steps: "
          + ", ".join(f"D {DD[j]:g}-{DD[j + 1]:g} {sl[j]:+.4f}" for j in range(max(0, len(sl) - 6), len(sl))))
    if dd[-1] < -0.05:
        slope = (dd[-1] - dd[-2]) / (DD[-1] - DD[-2])
        print(f"delta {dd[-1]:+.4f} is past -0.05, slope {slope:+.4f} per unit D at the last step: "
              + ("NOT levelling" if slope < -0.02 else "levelling"))
    print("\nwindows (c = the limit of delta):")
    for k, lab in ((2, "c + a/n"), (3, "c + a/n + b/n^2")):
        for ns, c in windows(nn, dd, k)[-6:]:
            print(f"   {lab:16s} n = {', '.join(f'{v:g}' for v in ns)}: c {c:+.4f}")
    return True


def calib():
    Ds, L = lam0_table()
    D = np.arange(5.0, 21.0)
    n = D - 2
    lam = np.array([lam0_at(d, Ds, L)[0] for d in D])
    e0 = max(lam0_at(d, Ds, L)[1] for d in D)
    print(f"lambda_0 at D 5 .. 20 (max interpolation error {e0:.1e}); n (lambda_0 - 2): "
          + ", ".join(f"{v:.4f}" for v in n * (lam - 2)))
    for k, lab in ((2, "c + a/n"), (3, "c + a/n + b/n^2")):
        w = windows(n, lam - 2, k)
        print(f"   {lab:16s} c - 0 on consecutive n: " + "; ".join(f"n {int(q[0][0])}-{int(q[0][-1])}: {q[1]:+.4f}" for q in w))
    Dmax = float(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--Dmax=")), 12))
    print(f"   least-squares polynomials in 1/n, limit free, on D 5 .. {Dmax:g} (the limit is 0):")
    for Dlo in (5.0, 6.0, 7.0):
        m = (D >= Dlo) & (D <= Dmax)
        for deg in range(2, 6):
            if deg + 2 > m.sum():
                continue
            A = np.stack([n[m] ** -j for j in range(deg + 1)], 1)
            cf, *_ = np.linalg.lstsq(A, lam[m] - 2, rcond=None)
            rms = float(np.sqrt(np.mean((A @ cf - (lam[m] - 2)) ** 2)))
            print(f"      D {Dlo:g} .. {Dmax:g}, degree {deg}: limit {cf[0]:+.4f} (rms {rms:.1e})")
    return True


def fold_fit(D, sv, X, k):
    """D* from the last k points of sigma^2 = c (D* - D) (sigma itself under --mutate=sigma), then each series in X
    fitted as x* + kk sqrt(D* - D) + m (D* - D) on the same points; returns D*, {name: x*}"""
    Dk, sk = D[-k:], sv[-k:]
    y = sk if "sigma" in MUT else sk ** 2
    A = np.stack([np.ones(k), Dk], 1)
    (c0, c1), *_ = np.linalg.lstsq(A, y, rcond=None)
    Ds = -c0 / c1
    out = {}
    for name, x in X.items():
        r = np.sqrt(np.maximum(Ds - Dk, 0))
        B = np.stack([np.ones(k), r, Ds - Dk], 1)
        cf, *_ = np.linalg.lstsq(B, x[-k:], rcond=None)
        out[name] = float(cf[0])
    return float(Ds), out


def fold(fn):
    if "--planted" in sys.argv:
        Dst, Xs = 16.37, dict(Delta=0.9812, lam=2.21)
        D = np.array([14.0, 14.5, 15.0, 15.5, 15.75, 16.0, 16.125])
        sv = 3.1e-7 * np.sqrt(Dst - D)
        X = dict(Delta=Xs["Delta"] - 0.05 * np.sqrt(Dst - D) + 0.004 * (Dst - D),
                 lam=Xs["lam"] + 0.21 * np.sqrt(Dst - D) - 0.01 * (Dst - D))
        good = True
        for k in (3, 4, 5):
            Ds, xs = fold_fit(D, sv, X, k)
            e = max(abs(Ds - Dst), abs(xs["Delta"] - Xs["Delta"]), abs(xs["lam"] - Xs["lam"]))
            good &= e < 1e-9
            print(f"planted fold D* {Dst}: last {k} points give D* {Ds:.10f}, Delta* {xs['Delta']:.10f}, lambda* {xs['lam']:.10f}; "
                  f"max error {e:.1e} (bar 1e-9)" + (f"  [MUTATION {MUT}]" if MUT else ""))
        print("PASS" if good else "FAIL")
        return good
    rows = [json.loads(l_) for l_ in open(fn)]
    rows = [r for r in rows if "sv_fold" in r]
    D = np.array([r["D"] for r in rows])
    sv = np.array([r["sv_fold"] for r in rows])
    print("| D | Delta | sigma_fold | cos | sigma_1(J) | sigma_min(bordered) / max | abs(J+ dF/dn) | abs(dDelta/dF) | constraint |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['D']:g} | {r['Delta']:.7f} | {r['sv_fold']:.3e} | {r['cos_fold']:.3f} | {r['sv_J_small'][0]:.3e} | "
              f"{r['sv_B_min'] / r['sv_B_max']:.2e} | {r['tan_norm']:.2e} | {r['dDelta_dF']:.2e} | {r['constraint']:.1e} |")
    X = dict(Delta=np.array([r["Delta"] for r in rows]))
    for k in (3, 4, 5):
        if len(D) >= k:
            Ds, xs = fold_fit(D, sv, X, k)
            print(f"last {k} points: D* {Ds:.4f}, Delta* {xs['Delta']:.6f}")
    return True


def collect(raw):
    srcs = [os.path.join(HERE, "data", f) for f in ("l1_D4_kc47.jsonl", "l1_kc23r.jsonl", "l1_kc31r.jsonl")]
    srcs += [os.path.join(raw, "I810-mac", "out", f"l1kc{k}{s_}s810.jsonl") for k in (23, 31, 39) for s_ in ("d4", "d5", "")]
    srcs += [os.path.join(raw, "I660-mac", "out", f"l1kc{k}t660.jsonl") for k in (23, 31, 39)]
    srcs += [os.path.join(raw, "I1025-mac", "out", f"l1kc{k}{t_}1025.jsonl") for k in (31, 39, 47, 55) for t_ in ("r", "L", "Lr", "Fr", "Fs", "Fb")]
    by = {}
    for f in srcs:
        if not os.path.exists(f):
            continue
        for line in open(f):
            r = json.loads(line)
            if r.get("lam") and r.get("l", 1) == 1:
                by.setdefault(round(r["D"], 6), {})[r["kc"]] = (r["lam"], os.path.relpath(f, raw) if f.startswith(raw) else os.path.relpath(f, HERE),
                                                                r.get("constraint"))
    pts = []
    for D in sorted(by):
        ks = sorted(by[D])
        top = ks[-1]
        err = abs(by[D][top][0] - by[D][ks[-2]][0]) if len(ks) > 1 else None
        pts.append(dict(D=D, lam=by[D][top][0], kc=top, lam_err=err if err is not None else 0.0, has_err=err is not None,
                        kcs={str(k): by[D][k][0] for k in ks}, source=by[D][top][1], constraint=by[D][top][2]))
        print(f"D {D:g}: k_c {ks}, lambda {by[D][top][0]:.10f}" + (f", change from k_c {ks[-2]} {err:.1e}" if err is not None else ""))
    json.dump(dict(points=pts, note="built by ell_delta.py collect"), open(os.path.join(HERE, "data", "ell_lam_highD.json"), "w"), indent=1)
    return True


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "planted":
        good = planted()
    elif cmd == "fold":
        rest = [a_ for a_ in sys.argv[2:] if not a_.startswith("--")]
        good = fold(rest[0] if rest else os.path.join(HERE, "data", "l1kc47F1025.jsonl"))
    elif cmd == "calib":
        good = calib()
    elif cmd == "collect":
        good = collect(os.path.abspath(sys.argv[2]))
    else:
        rest = [a for a in sys.argv[2:] if not a.startswith("--")]
        good = table(rest[0] if rest else os.path.join(HERE, "data", "ell_lam_highD.json"))
    sys.exit(0 if good else 1)

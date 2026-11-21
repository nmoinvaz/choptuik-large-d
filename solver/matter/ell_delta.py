#!/usr/bin/env python3
"""The l = 1 multiplet's growth rate against the scalar's at the same D: delta = lambda_1 - lambda_0 and its limit by
ELL.md 7's windows (#1025).

    python3 ell_delta.py planted       the window code on synthetic delta(n) = c + a/n + b/n^2 + e/n^3 with known c
    python3 ell_delta.py planted --mutate=window    a window off by one point: the exactness checks must fail
    python3 ell_delta.py collect RAWDIR   data/ell_lam_highD.json from the k_c runs: per D the highest k_c's lambda and,
                                       as lam_err, its change from the next lower k_c, and the same from P4's hybrid
                                       re-roots (RAWDIR: a checkout of choptuik-large-d-raw with I660-mac, I810-mac and
                                       the I1025 folders)
    python3 ell_delta.py calib         the same windows on the scalar's own lambda_0(D), whose limit is exactly 2
                                       (GAMMA.md 5), at integer D 5 .. 20 (lambda_0 interpolated as below): how far
                                       the windows are from their limit at these n for a real sequence of this kind
    python3 ell_delta.py fold [FILE]   a fold of the branch in D from ell_leap --fold records (default
                                       data/l1kc47F1025.jsonl): D* from sigma_fold^2 = c (D* - D) on the last 3, 4, 5
                                       points, then Delta and lambda at the fold from x(D) = x* + k sqrt(D* - D) + m (D* - D)
    python3 ell_delta.py fold --planted   the same code on a synthetic fold (D* 16.37): D*, Delta* and lambda* to 1e-9;
                                       --mutate=sigma fits sigma instead of sigma^2 and must miss D* (exit 1)
    python3 ell_delta.py table [FILE]  delta, n delta and the windows from the l = 1 points in FILE
                                       (default data/ell_lam_highD.json): P4's hybrid roots and hybrid k_c spreads where
                                       they exist (D 14.5 to 17), float64 elsewhere, with the float64 values beside them
    python3 ell_delta.py table --f64   the same from the float64 roots alone (the table before P4)
    python3 ell_delta.py steep [FILE]  the k_c test of lambda_1's steepening past D 15: the move from k_c 47 to 55 at
                                       each D against the steepening S(D), lambda_47 minus its line through D 14.5, 15;
                                       resolution if |move| > |S| at some D (bar fixed before the k_c 55 runs)
    python3 ell_delta.py steep --planted [--mutate=steep]   the same on a synthetic quadratic lambda: a move of
                                       0.1 S reads real, -1.5 S resolution; the mutation flips the comparison (exit 1)
    python3 ell_delta.py kcneed [FILE] per D (13 on) with two or more k_c: the k_c at which lambda's error, taken as
                                       A e^(-alpha k_c), falls to 1e-6 and to 0.1 |S|; alpha from lambda's moves when
                                       three k_c shrink, else from the psi-constraint's decay (flagged when the moves grow)
    python3 ell_delta.py kcneed --planted [--mutate=kcneed]   planted exponential errors: alpha and the k_c to 1e-9;
                                       the mutation inverts alpha (exit 1)
    python3 ell_delta.py leak          |K'_nu/K'_1| at the l = 0 and l = 1 M = 3 zeros: the weight with which an error
                                       in the harmonic-nu condition leaks into the harmonic-1 one, and that times eps

Windows (as ELL.md 7 and ell_fit.py): on consecutive n, the exact interpolants c + a/n (two points) and
c + a/n + b/n^2 (three points), and c is their limit n -> infinity. In x = 1/n they are polynomial extrapolations to
x = 0. For delta = c + a x + b x^2 + e x^3 the two-point value is c - b x1 x2 - e x1 x2 (x1 + x2), and the
three-point value is c + e x1 x2 x3. planted checks the code against these to 1e-12, and with e = 0 the three-point
window must return c to 1e-12.

lambda_0(D) is not stored at the l = 1 points' D. It is interpolated in x = 1/n by the cubic through the four nearest
stored points (solver/data/scan_ne6_lowD_kc39_rk2.jsonl, then gamma_rust.jsonl). The cubic minus the quadratic through
the three nearest points is quoted as the interpolation error. ell_fit.py interpolated linearly in D.
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P4_AMEND5 = {("l1kc63Fw1025.jsonl", 17.0)}           # P4 rows whose system check is read under P4 amendment 5 (p4_criteria.log)
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
    and the windows. The crossing and the slopes use only rows with two or more valid k_c; single-k_c rows are printed
    and marked."""
    rows = json.load(open(fn))["points"]
    Ds, L = lam0_table()
    l1inf = json.load(open(os.path.join(HERE, "data", "ell_lambda_l1.json")))["M3"]["lam"]
    dinf = l1inf - 2
    nn, dd, ee, DD, hh = [], [], [], [], []
    print(f"delta(inf) = lambda_inf(l = 1) - lambda_0(inf) = {l1inf:.6f} - 2 = {dinf:+.6f} (#994, M = 3; "
          f"lambda_0(inf) = 2, GAMMA.md 5)\n")
    f64m = "--f64" in sys.argv                       # the float64 table (ell_refine's roots), as quoted before P4 was in
    if f64m:
        print("| D | n | λ₁ | λ₁ from | k_c | signtest | k_c bar: change from the next lower k_c | ψ-constraint | λ₀ (interp. error) | "
              "δ = λ₁ − λ₀ | δ − δ(∞) | n δ | n δ(∞) |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    else:                                            # P4's hybrid roots and hybrid k_c spreads wherever a row has one
        rows = [dict(r, lam=r["lam_hybrid"], kc=r["kc_hybrid"], lam_err=r["lam_err_hybrid"], has_err=r["has_err_hybrid"],
                     constraint=r["constraint_hybrid"], signtested=r["signtested_hybrid"], note=r["note_hybrid"],
                     method="hybrid" if (r["err_hybrid_from"] or "hybrid").startswith("hybrid") else
                     f"hybrid; spread to {r['err_hybrid_from']}",
                     source=f"{r['record_hybrid'].removesuffix('.jsonl')}, P4 {r['source_hybrid'].split('/')[0]}",
                     f64=(r["lam"], r["kc"], r["lam_err"] if r.get("has_err", True) else None))
                if r.get("lam_hybrid") is not None else dict(r, method="float64", source=r["source"].split("/")[-1].removesuffix(".jsonl"))
                for r in rows]
        print("| D | n | λ₁ | from | k_c | source (record, P4 file's folder) | k_c bar: change from the next lower k_c | λ₁ minus its "
              "record's float64 root | float64 table: λ₁ (k_c; spread) | the record's signtest | ψ-constraint | λ₀ (interp. error) | "
              "δ = λ₁ − λ₀ | n δ |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda q: q["D"]):
        D, l1 = r["D"], r["lam"]
        l0, e0 = lam0_at(D, Ds, L)
        n = D - 2
        d = l1 - l0
        nn.append(n); dd.append(d); ee.append(r.get("lam_err", 0.0)); DD.append(D); hh.append(r.get("has_err", True))
        bar = (("OK" if r["lam_err"] <= 1e-6 else "BAD") + f" ({r['lam_err']:.1e})") if r.get("has_err", True) else \
            "single k_c (not in crossing or slopes)"
        con = r.get("constraint")
        cs = f"{con:.1e}" if con is not None else ""
        if f64m:
            sgt = "passed" if r.get("signtested") else "no signtest yet"
            how = r.get("method", "float64") + (f", P4 gap {r['p4_gap']:+.2e}" if r.get("p4_gap") is not None else "") + \
                (" (note)" if r.get("note") else "")
            print(f"| {D:g} | {n:g} | {l1:.7f} | {how} | {r.get('kc', '')} | {sgt} | {bar} | {cs} | {l0:.7f} ({e0:.1e}) | {d:.6f} | "
                  f"{d - dinf:+.6f} | {n * d:.4f} | {n * dinf:.4f} |")
            continue
        sgt = "passed" if r.get("signtested") else "none"
        sp = bar
        f6 = r.get("f64")
        mr = f"{r['minus_record_hybrid']:+.2e}" if f6 else ""
        ft = (f"{f6[0]:.7f} ({f6[1]}; " + (f"{f6[2]:.1e})" if f6[2] is not None else "single k_c)")) if f6 else "same"
        print(f"| {D:g} | {n:g} | {l1:.7f} | {r['method']}{' (note)' if r.get('note') else ''} | {r.get('kc', '')} | {r['source']} | {sp} | "
              f"{mr} | {ft} | {sgt} | {cs} | {l0:.7f} ({e0:.1e}) | {d:.6f} | {n * d:.4f} |")
    for r in sorted(rows, key=lambda q: q["D"]):
        if r.get("note"):
            print(f"\nnote, D {r['D']:g} (k_c {r.get('kc', '')}, {r.get('source', '')}): {r['note']}")
    if not f64m:
        hr = [r for r in sorted(rows, key=lambda q: q["D"]) if r.get("f64")]
        print(f"\nλ₁ is P4's hybrid root (float64 blocks 0-11, double-double horizon block) on each record's own state, at D "
              f"{hr[0]['D']:g} to {hr[-1]['D']:g}, where every root passed P4's rule (raw I1025-mac/p4_criteria.log); the other rows "
              "are float64. The k_c spread is between the two highest hybrid k_c, except where marked 'spread to float64'. "
              "'the record's signtest' is the float64 signtest at lam -/+ 0.02 of the record whose state the root is on.")
        print("\nhybrid δ by k_c (P4 roots; where k_c 55 and 63 both exist, δ(63) − δ(55)):")
        ks = sorted({int(k) for r in hr for k in r["kcs_hybrid"]})
        print("| D | " + " | ".join(f"δ at k_c {k}" for k in ks) + " | δ(63) − δ(55) |")
        print("|---|" + "---|" * (len(ks) + 1))
        for r in hr:
            l0 = lam0_at(r["D"], Ds, L)[0]; kh = r["kcs_hybrid"]
            dk = f"{kh['63'] - kh['55']:+.2e}" if "55" in kh and "63" in kh else ""
            print(f"| {r['D']:g} | " + " | ".join(f"{kh[str(k)] - l0:+.6f}" if str(k) in kh else "" for k in ks) + f" | {dk} |")
        for k in ("55", "63"):
            pk = [(r["D"], r["kcs_hybrid"][k] - lam0_at(r["D"], Ds, L)[0]) for r in hr if k in r["kcs_hybrid"]]
            print(f"slopes of the hybrid δ at k_c {k}: " + ", ".join(f"D {a[0]:g}-{b[0]:g} {(b[1] - a[1]) / (b[0] - a[0]):+.4f}"
                                                                     for a, b in zip(pk, pk[1:])))
    gaps = [(r["D"], r["p4_gap"]) for r in sorted(rows, key=lambda q: q["D"]) if r.get("p4_gap") is not None] if f64m else []
    if gaps:
        print("\nthe k_c bar and the spreads below are float64 k_c spreads. P4's hybrid re-roots differ from the float64 records by "
              + ", ".join(f"{g:+.2e} at D {D:g}" for D, g in gaps) + ", which the float64 spreads do not cover")
    nn, dd, ee, DD = map(np.array, (nn, dd, ee, DD))
    nw, dw = nn, dd                                  # the windows keep every row
    m = np.array(hh, bool)                           # rows with two or more valid k_c
    if (~m).any():
        print("\nleft out of the crossing and the slopes (single k_c): " + ", ".join(f"D {v:g}" for v in DD[~m]))
    dd, ee, DD = dd[m], ee[m], DD[m]
    cr = [i for i in range(len(dd) - 1) if dd[i] > 0 >= dd[i + 1]]
    if cr:
        i = cr[0]
        lin = lambda a, b: DD[i] + (DD[i + 1] - DD[i]) * a / (a - b)
        Dc = lin(dd[i], dd[i + 1])
        alts = [lin(dd[i] + s1 * ee[i], dd[i + 1] + s2 * ee[i + 1]) for s1 in (-1, 1) for s2 in (-1, 1)]
        print(f"\ndelta crosses 0 between D {DD[i]:g} and {DD[i + 1]:g}: D = {Dc:.4f} (linear), "
              f"k_c spread {min(alts):.4f} to {max(alts):.4f}")
    else:
        print(f"\ndelta has not crossed 0 (last D {DD[-1]:g}, delta {dd[-1]:+.6f})")
    sl = np.diff(dd) / np.diff(DD)
    sb = (ee[:-1] + ee[1:]) / np.diff(DD)            # the slope's k_c spread: both ends moved by their k_c change
    print("slope of delta per unit D over the last steps (± its k_c spread, the change between the two highest k_c, "
          "not a converged bound): "
          + ", ".join(f"D {DD[j]:g}-{DD[j + 1]:g} {sl[j]:+.4f} ± {sb[j]:.4f}" for j in range(max(0, len(sl) - 6), len(sl))))
    if dd[-1] < -0.05:
        slope, bar = sl[-1], sb[-1]
        held = "holds" if dd[-1] + ee[-1] < -0.05 else "does not hold"
        verdict = ("levelling" if slope >= -0.02 else "NOT levelling" if slope + bar < -0.02 else
                   f"NOT levelling at the central value; its k_c spread reaches {slope + bar:+.4f}, so levelling is "
                   "not excluded")
        print(f"delta {dd[-1]:+.4f} is past -0.05 (within its k_c spread {ee[-1]:.1e}, not a converged bound: {held}), "
              f"slope {slope:+.4f} ± {bar:.4f} per unit D at the last step: {verdict}")
    print("\nwindows (c = the limit of delta):")
    for k, lab in ((2, "c + a/n"), (3, "c + a/n + b/n^2")):
        for ns, c in windows(nw, dw, k)[-6:]:
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
    srcs += [os.path.join(raw, "I1025-win", "out", f) for f in ("l1kc63Fw1025.jsonl", "l1kc63W1025.jsonl",
                                                               "l1kc63Fw1025b.jsonl")]   # winb's k_c 63 lane
    srcs += [os.path.join(raw, "I1025-mac", "out", f"l1kc{k}{t_}1025.jsonl") for k in (47, 55) for t_ in ("Fq", "Fp")]
    srcs += [os.path.join(raw, "I1025-mac", "out", "l1kc63Fb1025.jsonl")]   # lane J: D 16.25, 16.75 and k_c 63 at D 16.5
    srcs += [os.path.join(raw, "I1025-win2b", "out", f"l1kc55F{t_}1025.jsonl") for t_ in ("v", "u")]   # win2b: D 16.75, 16.25
    srcs += [os.path.join(raw, "I1025-mac", "out", f) for f in ("l1kc71Fm1025.jsonl", "l1kc63Fm1025.jsonl")]   # lane T
    srcs += [os.path.join(raw, "I1025-win", "out", "l1kc71Fw1025.jsonl")]   # winb: k_c 71 at D 16.5
    srcs += [os.path.join(raw, "I1025-win2b", "out", f"l1kc{k}Fw1025.jsonl") for k in (47, 55)]   # win2b redos: D 17, D 16.5
    srcs += [os.path.join(raw, "I1025-win2c", "out", f"l1kc63F{t_}1025.jsonl") for t_ in ("y", "z")]   # win2c: D 15, D 15.5
    srcs += [os.path.join(raw, "I1025-win2b", "out", "l1kc63Fu1025.jsonl")]   # win2b: k_c 63 at D 16.25
    srcs += [os.path.join(raw, "I1025-win2b", "out", "l1kc71Fw1025c.jsonl")]   # win2b: k_c 71 at D 16.5 (the recheck)
    exf = os.path.join(HERE, "data", "ell_lam_excluded.json")   # records that failed a check (signtest, guess, sign changes)
    excl = {(e["source"], round(e["D"], 6)): e["reason"] for e in json.load(open(exf))["excluded"]} if os.path.exists(exf) else {}
    by = {}
    for f in srcs:
        if not os.path.exists(f):
            continue
        for line in open(f):
            r = json.loads(line)
            key = (os.path.relpath(f, raw) if f.startswith(raw) else os.path.relpath(f, HERE), round(r.get("D", 0), 6))
            if key in excl:
                print(f"excluded: {key[0]} D {key[1]:g} lambda {r.get('lam')}: {excl[key]}")
                continue
            if r.get("lam") and r.get("l", 1) == 1:
                by.setdefault(round(r["D"], 6), {})[r["kc"]] = (r["lam"], os.path.relpath(f, raw) if f.startswith(raw) else os.path.relpath(f, HERE),
                                                                r.get("constraint"))
    stf = os.path.join(HERE, "data", "ell_lam_signtested.json")   # quoted rows whose own record passed signtest.py
    stset = {(e["source"], round(e["D"], 6)) for e in json.load(open(stf))["signtested"]} if os.path.exists(stf) else set()
    ntf = os.path.join(HERE, "data", "ell_lam_notes.json")   # remarks printed beside a quoted row
    notes = {(e["source"], round(e["D"], 6)): e for e in json.load(open(ntf))["notes"]} if os.path.exists(ntf) else {}
    # P4's hybrid re-roots (float64 blocks 0-11, dd block 12) of records on their own states (raw I1025-mac/p4_criteria.log), each
    # taken only if it passes P4's rule; the first file wins for a record re-rooted on several machines (the cross-check row)
    hyb = {}
    for f in [os.path.join(raw, "I1025-mac", "out", "ell_p4_hybrid_1025.jsonl")] + \
            [os.path.join(raw, m_, "p4", "ell_p4_hybrid_1025_w.jsonl") for m_ in ("I1025-win", "I1025-win2b", "I1025-dh170")]:
        if not os.path.exists(f):
            continue
        for line in open(f):
            r = json.loads(line)
            if (r["mode"] == "hybrid" and r["lam"] is not None and (r["record_system"] or (r["record"], r["D"]) in P4_AMEND5)
                    and r["n_signchanges_grid"] == 1 and r["n_changes_all"] == 1 and r["noise_robust"] and (r["bracket_width"] or 1) <= 1e-7):
                hyb.setdefault(round(r["D"], 6), {}).setdefault(r["kc"], (r["lam"], os.path.relpath(f, raw), r["record"], r["minus_record"]))
    rpath = {os.path.basename(f): f for f in srcs if os.path.exists(f)}   # a hybrid row's own record file, for its constraint
    for f in glob.glob(os.path.join(raw, "I1025-*", "out", "l1kc*.jsonl")):
        rpath.setdefault(os.path.basename(f), f)

    def rec_at(name, D):
        f = rpath.get(name)
        rs = [json.loads(q) for q in open(f) if json.loads(q).get("D") == D and json.loads(q).get("lam")] if f else []
        return (os.path.relpath(f, raw), rs[-1]) if rs else (None, {})
    pts = []
    for D in sorted(by):
        ks = sorted(by[D])
        top = ks[-1]
        err = abs(by[D][top][0] - by[D][ks[-2]][0]) if len(ks) > 1 else None
        h, hp = hyb.get(D, {}), {}
        if h:                                        # the hybrid row: its top k_c, and the spread to the next lower hybrid k_c
            hk = sorted(h); ht = hk[-1]              # (or, with one hybrid k_c, to the next lower float64 one, marked)
            lower = [k for k in ks if k < ht]
            he, hf = ((abs(h[ht][0] - h[hk[-2]][0]), f"hybrid k_c {hk[-2]}") if len(hk) > 1 else
                      (abs(h[ht][0] - by[D][lower[-1]][0]), f"float64 k_c {lower[-1]}") if lower else (None, None))
            rp, rr = rec_at(h[ht][2], D)
            hp = dict(lam_hybrid=h[ht][0], kc_hybrid=ht, lam_err_hybrid=he if he is not None else 0.0, has_err_hybrid=he is not None,
                      err_hybrid_from=hf, kcs_hybrid={str(k): h[k][0] for k in hk}, source_hybrid=h[ht][1], record_hybrid=h[ht][2],
                      record_hybrid_path=rp, minus_record_hybrid=h[ht][3], constraint_hybrid=rr.get("constraint"),
                      signtested_hybrid=(rp, D) in stset,
                      note_hybrid=next((e.get("note_hybrid") for (src, d), e in notes.items() if src == rp and d == D), None))
        pts.append(dict(D=D, lam=by[D][top][0], kc=top, lam_err=err if err is not None else 0.0, has_err=err is not None,
                        kcs={str(k): by[D][k][0] for k in ks}, kcs_constraint={str(k): by[D][k][2] for k in ks},
                        source=by[D][top][1], constraint=by[D][top][2], signtested=(by[D][top][1], D) in stset,
                        method="float64", note=notes.get((by[D][top][1], D), {}).get("note"),   # ell_refine's float64 roots
                        p4_gap=notes.get((by[D][top][1], D), {}).get("p4_gap"), **hp))   # P4's hybrid minus this record
        print(f"D {D:g}: k_c {ks}, lambda {by[D][top][0]:.10f}" + (f", change from k_c {ks[-2]} {err:.1e}" if err is not None else "")
              + (f"; hybrid k_c {sorted(h)}, lambda {hp['lam_hybrid']:.10f}" + (f", change from {hp['err_hybrid_from']} "
                 f"{hp['lam_err_hybrid']:.1e}" if hp["has_err_hybrid"] else "") if h else ""))
    for D in sorted(set(hyb) - set(by)):
        print(f"hybrid rows at D {D:g} with no float64 row collected (not in the table): {sorted(hyb[D])}")
    json.dump(dict(points=pts, note="built by ell_delta.py collect"), open(os.path.join(HERE, "data", "ell_lam_highD.json"), "w"), indent=1)
    return True


def steep_verdict(lo, hi, D0=14.5, D1=15.0, Ds=(15.5, 16.0, 16.5, 17.0, 17.5)):
    """the k_c test of lambda_1's steepening (#1025, coordinator): S(D) = lambda_lo(D) minus the line through lambda_lo
    at D0, D1 (the steepening's size at D), against the move to the higher k_c, c(D) = lambda_hi(D) - lambda_lo(D).
    The steepening is resolution if |c| > |S| at any D in Ds where both are known."""
    s0 = (lo[D1] - lo[D0]) / (D1 - D0)
    rows = []
    for D in Ds:
        if D in lo and D in hi:
            S = lo[D] - (lo[D1] + s0 * (D - D1))
            c = hi[D] - lo[D]
            res = abs(c) < abs(S) if "steep" in MUT else abs(c) > abs(S)
            rows.append((D, lo[D], hi[D], c, S, res))
    return rows, any(r[-1] for r in rows)


def steep(fn, planted=False):
    if planted:
        D = [14.5, 15.0, 15.5, 16.0, 16.5, 17.0]
        lo = {d: 2.5 - 0.15 * (d - 14.5) - 0.02 * (d - 14.5) ** 2 for d in D}
        S = {d: lo[d] - (lo[15.0] + (lo[15.0] - lo[14.5]) / 0.5 * (d - 15.0)) for d in D}
        good = True
        for lab, hi, want in (("k_c move 0.1 S + 1e-4 (real)", {d: lo[d] + 0.1 * S[d] + 1e-4 for d in D}, False),
                              ("k_c move -1.5 S (resolution)", {d: lo[d] - 1.5 * S[d] for d in D}, True)):
            _, res = steep_verdict(lo, hi)
            ok = res == want
            good &= ok
            print(f"planted, {lab}: verdict {'resolution' if res else 'real'} ({'OK' if ok else 'BAD'})"
                  + (f"  [MUTATION {MUT}]" if MUT else ""))
        return good
    pts = json.load(open(fn))["points"]
    lo = {p["D"]: p["kcs"]["47"] for p in pts if "47" in p["kcs"]}
    hi = {p["D"]: p["kcs"]["55"] for p in pts if "55" in p["kcs"]}
    rows, res = steep_verdict(lo, hi)
    print("| D | λ₁ k_c 47 | λ₁ k_c 55 | c = move to k_c 55 | S = steepening at k_c 47 | \\|c\\| > \\|S\\| |")
    print("|---|---|---|---|---|---|")
    for D, a, b, c, S, r in rows:
        print(f"| {D:g} | {a:.7f} | {b:.7f} | {c:+.2e} | {S:+.2e} | {'yes' if r else 'no'} |")
    ks = sorted(hi)
    if len(ks) > 1:
        print("k_c 55 slopes per unit D: " + ", ".join(
            f"D {ks[i]:g}-{ks[i + 1]:g} {(hi[ks[i + 1]] - hi[ks[i]]) / (ks[i + 1] - ks[i]):+.4f}" for i in range(len(ks) - 1)))
    print(("verdict: the steepening is resolution (k_c 55 moves lambda_1 by more than it at some D)" if res else
           "verdict: the k_c 55 move is below the steepening at every D") if rows else "no D with both k_c 47 and 55")
    return True


def kc_need(lam, con, target):
    """the k_c at which lambda's remaining k_c error falls to target, for an error A e^(-alpha k_c). lam, con: {k_c:
    lambda, psi-constraint} at one D, k_c spaced equally. alpha from lambda's last two moves when there are three k_c
    (and they shrink), else from the psi-constraint's decay over the top two k_c (an assumption: lambda's error decays
    as the constraint does). The error left at the top k_c is its last move times r / (1 - r), r = e^(-alpha dk)."""
    ks = sorted(lam)
    dk = ks[-1] - ks[-2]
    m1 = abs(lam[ks[-1]] - lam[ks[-2]])
    src = "constraint"
    grow = len(ks) >= 3 and abs(lam[ks[-2]] - lam[ks[-3]]) <= m1     # lambda's moves do not shrink: not the model
    if len(ks) >= 3 and not grow:
        m0 = abs(lam[ks[-2]] - lam[ks[-3]])
        alpha, src = (np.log(m1 / m0) if "kcneed" in MUT else np.log(m0 / m1)) / dk, "lambda"
    else:
        alpha = (np.log(con[ks[-1]] / con[ks[-2]]) if "kcneed" in MUT else np.log(con[ks[-2]] / con[ks[-1]])) / dk
    if alpha <= 0:
        return dict(alpha=float(alpha), src=src, err=None, k=None, grow=grow)
    r = np.exp(-alpha * dk)
    err = m1 * r / (1 - r)
    return dict(alpha=float(alpha), src=src, err=float(err), k=float(ks[-1] + np.log(err / target) / alpha), grow=grow)


def kcneed(fn, planted=False):
    if planted:
        good = True
        for lab, alpha in (("three k_c", 0.3), ("constraint", 0.25)):
            ks = (39, 47, 55) if lab == "three k_c" else (47, 55)
            lam = {k: 2.0 + 50 * np.exp(-alpha * k) for k in ks}
            con = {k: 7 * np.exp(-alpha * k) for k in ks}
            T = 1e-6
            want = np.log(50 / T) / alpha
            o = kc_need(lam, con, T)
            ok = o["k"] is not None and abs(o["k"] - want) < 1e-9 and abs(o["err"] - 50 * np.exp(-alpha * ks[-1])) < 1e-12
            good &= ok
            print(f"planted, {lab} (alpha {alpha}): alpha {o['alpha']:.6f} from {o['src']}, k_c needed for 1e-6 "
                  f"{o['k'] if o['k'] is None else round(o['k'], 6)} (want {want:.6f}) ({'OK' if ok else 'BAD'})"
                  + (f"  [MUTATION {MUT}]" if MUT else ""))
        return good
    pts = json.load(open(fn))["points"]
    print("| D | k_c (λ₁) | ψ-constraint by k_c | α per k_c (from) | error left at top k_c | k_c for \\|S\\| | k_c for 1e-6 |")
    print("|---|---|---|---|---|---|---|")
    lo = {p["D"]: p["kcs"]["47"] for p in pts if "47" in p["kcs"]}
    rows, _ = steep_verdict(lo, lo)
    S = {r[0]: abs(r[4]) for r in rows}
    for p in pts:
        if p["D"] < 13 or len(p["kcs"]) < 2 or not p.get("kcs_constraint"):
            continue
        lam = {int(k): v for k, v in p["kcs"].items()}
        con = {int(k): v for k, v in p["kcs_constraint"].items()}
        if any(v is None for v in con.values()):
            continue
        ks = sorted(lam)
        if len(set(np.diff(ks))) > 1:
            ks = ks[-2:]
            lam, con = {k: lam[k] for k in ks}, {k: con[k] for k in ks}
        o = kc_need(lam, con, 1e-6)
        oS = kc_need(lam, con, 0.1 * S[p["D"]]) if p["D"] in S else None
        print(f"| {p['D']:g} | " + ", ".join(f"{k}: {lam[k]:.7f}" for k in ks) + " | "
              + ", ".join(f"{con[k]:.1e}" for k in ks) + f" | {o['alpha']:.3f} ({o['src']}"
              + ("; lambda's moves grow, so not this model" if o["grow"] else "") + ") | "
              + (f"{o['err']:.1e} | " if o["err"] else "- | ")
              + (f"{oS['k']:.0f} | " if oS and oS["k"] else "- | ") + (f"{o['k']:.0f} |" if o["k"] else "- |"))
    print("k_c for |S|: the k_c at which the remaining error is 0.1 |S| (S as in steep)")
    return True


def leak():
    """|K'_nu / K'_1| (inner_np.Kp at i nu w + l) at the M = 3 zeros: the weight with which an error in the harmonic-nu
    condition leaks into the harmonic-1 one (nested_ell.py, hier.py). l = 0: FIRST_ORDER.md's zero (w/w_inf
    1.015603095); l = 1: #994's (data/ell_lambda_l1.json)."""
    sys.path.insert(0, os.path.join(HERE, "..", "inner_stokes"))
    import inner_np as IN
    wf1 = json.load(open(os.path.join(HERE, "data", "ell_lambda_l1.json")))["zero"]["M3"]["wf"]
    eps = np.finfo(float).eps
    for l, wf in ((0, 1.015603095), (1, wf1)):
        w = IN.W0 * wf
        k1 = abs(IN.Kp(1j * w + l))
        r = {n: abs(IN.Kp(1j * n * w + l)) / k1 for n in (3, 5, 7)}
        print(f"l = {l}, w/w_inf {wf}: " + ", ".join(f"|K'_{n}/K'_1| {v:.2e} (times eps {v * eps:.1e})" for n, v in r.items()))
    return True


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "planted":
        good = planted()
    elif cmd == "steep":
        rest = [a_ for a_ in sys.argv[2:] if not a_.startswith("--")]
        good = steep(rest[0] if rest else os.path.join(HERE, "data", "ell_lam_highD.json"), "--planted" in sys.argv)
    elif cmd == "leak":
        good = leak()
    elif cmd == "kcneed":
        rest = [a_ for a_ in sys.argv[2:] if not a_.startswith("--")]
        good = kcneed(rest[0] if rest else os.path.join(HERE, "data", "ell_lam_highD.json"), "--planted" in sys.argv)
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

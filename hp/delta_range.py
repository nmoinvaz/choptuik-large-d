#!/usr/bin/env python3
"""The horizon rows' linear range in Delta against D (#832): how far Delta can move, at fixed horizon data, before the
horizon block's response stops being linear.

    cd hp && python3 delta_range.py RAW STATE.npz [STATE.npz ...] [--threads=3] [--ds=12:36:2] [--out=CSV]
    cd hp && python3 delta_range.py --selftest [--mutate=oneside]

Per state (a chain state of the raw-data repository, relative to RAW): the pinned system at the state's D, as
octo_corrector.pinned_o builds it (the plan at the state at 1e-32, the held harmonics at its own values), and per move
d = 10^-k (k over --ds, first:last:step) the octo-double residual at z + d e_Delta and z - d e_Delta, the horizon
data and every other unknown held. Per horizon field f (the block's p, m, a and fourth rows, M rows each),
    S_f(d) = max |F(+d) - F(-d)| / (2 d)          (the odd part: s_f, the slope, as d -> 0)
    C_f(d) = max |F(+d) + F(-d) - 2 F(0)| / (2 d^2)  (the even part: c_f, half the curvature)
and r_f = S_f / C_f at the smallest d, the move where the two terms are equal: the linear range. A state whose
max|F| exceeds --F_max (1e-15) is not a solution of this system and is left out of the rate. A field is resolved
when S_f and C_f at the two smallest d agree within 10%; r is reported only for resolved fields, and the state's r is
the smallest over them. Max norms over a field's rows, so s, c and r are scales, not one row's derivatives. Printed
per state: D, u0 = 1 - x_last, n - 1 = D - 3 (the mass mode's exponent), and per field S, C and r at each d. The run
ends with the least-squares slope of ln r against D over the resolved states, the rate at which r falls, beside
#805's 1.44 (the weak pivot's) and 0.34 (13.6's pole law for a_h, (n - 1) u0 / 0.88 per unit of n, from
notes/HORIZON_RADIUS.md).

The odd and even parts separate s and c whatever d is against r, as long as the cubic and quartic terms are small
against them; where they are not, S or C does not plateau and the field is not resolved.

--selftest: planted responses F(d) = s d + c d^2 + e d^3 + q d^4 per row (random signs, rounding noise 1e-91), read
by the same estimator at d 1e-30 .. 1e-36. Plant 1 (s 1e32, c 1e58, e 1e84, q 1e110; r 1e-26): s and c recovered within
1e-3 and resolved. Plant 2 (e 1e100, so e d^2 exceeds s up to d 1e-34): not resolved. --mutate=evenone reads the
curvature from F(+d) - F(0) alone, which carries s / d, and must fail plant 1's recovery. Exit 1 on a failed
check."""
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FIELDS = ("p", "m", "a", "4th")
PLATEAU = 0.10


def estimate(F0, Fp, Fm, ds, rows, mutate=""):
    """{field: dict(S=[...], C=[...], r, resolved)} from residual changes per d (lists of numpy arrays: the summed
    eight parts, differences formed in octo-double by the caller); rows: {field: index array}"""
    out = {}
    for f, ix in rows.items():
        S, C = [], []
        for d, dp, dm in zip(ds, Fp, Fm):
            odd = (dp[ix] - dm[ix]) / (2 * d)
            even = (dp[ix] + dm[ix]) / (2 * d * d) if mutate != "evenone" else dp[ix] / (d * d)
            S.append(float(np.max(np.abs(odd))))
            C.append(float(np.max(np.abs(even))))
        i1, i2 = int(np.argmin(ds)), int(np.argsort(ds)[1])
        ok = (C[i1] > 0 and S[i1] > 0 and abs(S[i1] / S[i2] - 1) < PLATEAU and abs(C[i1] / C[i2] - 1) < PLATEAU)
        out[f] = dict(S=S, C=C, r=S[i1] / C[i1] if C[i1] > 0 else float("inf"), resolved=bool(ok))
    return out


def selftest(mutate=""):
    rng = np.random.default_rng(832)
    n = 32
    ds = [10.0 ** -k for k in range(30, 38, 2)]
    ok = True
    for name, s, c, e3, want in (("plant 1, resolved", 1e32, 1e58, 1e84, True),
                                 ("plant 2, a cubic above s", 1e32, 1e58, 1e100, False)):
        sv = s * rng.choice([-1, 1], n) * rng.uniform(0.2, 1, n)
        cv = c * rng.choice([-1, 1], n) * rng.uniform(0.2, 1, n)
        ev, qv = e3 * rng.uniform(0.5, 1, n), 1e110 * rng.standard_normal(n)
        sv[0], cv[0] = s, c                                       # the max norms are s and c
        resp = lambda d: sv * d + cv * d * d + ev * d ** 3 + qv * d ** 4 + 1e-91 * rng.standard_normal(n)  # noqa: E731
        Fp = [resp(d) for d in ds]
        Fm = [resp(-d) for d in ds]
        est = estimate(None, Fp, Fm, ds, {"p": np.arange(n)}, mutate)["p"]
        i1 = int(np.argmin(ds))
        es, ec = abs(est["S"][i1] / s - 1), abs(est["C"][i1] / c - 1)
        good = est["resolved"] == want and (not want or (es < 1e-3 and ec < 1e-3))
        ok &= good
        print(f"  {name}: S {est['S'][i1]:.4e} (planted {s:.1e}, off {es:.1e}), C {est['C'][i1]:.4e} (planted {c:.1e}, "
              f"off {ec:.1e}), resolved {est['resolved']} (want {want}): {'ok' if good else 'WRONG'}")
    print(f"delta_range selftest{' --mutate=' + mutate if mutate else ''}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def measure(raw, path, ds, threads, log):
    import mpmath as mp
    from multidouble import od_axpy
    from octo_corrector import load_o, o8, pinned_o
    z, meta = load_o(os.path.join(raw, path))
    z = o8(z)
    t0 = time.time()
    s, sp, sq, so = pinned_o(meta, z, threads=threads)
    M, K = int(meta["M"]), len(meta["nodes"])
    L = len(z[0])
    r0 = 4 * M * K
    rows = {f: np.arange(r0 + j * M, r0 + (j + 1) * M) for j, f in enumerate(FIELDS)}
    e = [np.zeros(L) for _ in range(8)]
    zero = [np.zeros(L) for _ in range(8)]
    F0 = o8(so.residual_o(z))

    def change(d):
        e[0][:] = 0.0
        e[0][-1] = d
        zd = o8(od_axpy(1.0, *e, *z))
        Fd = o8(so.residual_o(zd))
        df = o8(od_axpy(-1.0, *F0, *Fd))                           # F(z + d e) - F(z) in octo-double
        return np.sum(np.array([np.asarray(p) for p in df]), axis=0)
    Fp, Fm = [], []
    for d in ds:
        Fp.append(change(d))
        Fm.append(change(-d))
    est = estimate(None, Fp, Fm, ds, rows)
    with mp.workdps(60):
        D = mp.mpf(meta["D_str"])
    u0 = 1.0 - float(meta["nodes"][-1])
    log(f"{path}: D {mp.nstr(D, 10)}, u0 {u0:.4f}, n - 1 {float(D) - 3:.3f}, max|F| at the state "
        f"{float(np.max(np.abs(np.asarray(F0[0])))):.2e} ({time.time() - t0:.0f} s)")
    for f in FIELDS:
        x = est[f]
        log(f"  {f}: " + ", ".join(f"d 1e{np.log10(d):.0f}: S {sv:.3e} C {cv:.3e}" for d, sv, cv in
                                   zip(ds, x["S"], x["C"]))
            + f"; r {x['r']:.3e}, {'resolved' if x['resolved'] else 'NOT resolved'}")
    del zero
    return dict(state=path, D=float(D), D_str=meta["D_str"], u0=u0, est=est,
                Fmax=float(np.max(np.abs(np.asarray(F0[0])))))


def main(argv):
    import mpmath as mp
    mp.mp.dps = 60      # pinned_o parses D_str at the caller's precision before #831: at 15 digits D is rounded
    opts = dict(a[2:].split("=", 1) for a in argv if a.startswith("--") and "=" in a)
    if "--selftest" in argv:
        return selftest(opts.get("mutate", ""))
    args = [a for a in argv if not a.startswith("--")]
    if len(args) < 2:
        sys.exit(__doc__)
    raw, states = args[0], args[1:]
    a, b, c = (int(x) for x in opts.get("ds", "12:36:2").split(":"))
    ds = [10.0 ** -k for k in range(a, b + 1, c)]

    def log(s):
        print(s, flush=True)
    recs = [measure(raw, p, ds, int(opts.get("threads", 3)), log) for p in states]
    if opts.get("out"):
        with open(opts["out"], "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["state", "D", "u0", "field", "d", "S", "C", "r", "resolved"])
            for r in recs:
                for f in FIELDS:
                    x = r["est"][f]
                    for d, sv, cv in zip(ds, x["S"], x["C"]):
                        w.writerow([r["state"], f"{r['D']:.6f}", f"{r['u0']:.6f}", f, f"{d:.0e}", f"{sv:.6e}",
                                    f"{cv:.6e}", f"{x['r']:.6e}", int(x["resolved"])])
    pts = []
    fmax = float(opts.get("F_max", 1e-15))
    for r in recs:
        rs = [r["est"][f]["r"] for f in FIELDS if r["est"][f]["resolved"]]
        if r["Fmax"] > fmax:
            log(f"  D {r['D']:.4f}: max|F| {r['Fmax']:.2e} above {fmax:g}, not a solution of this system: left out")
            continue
        if rs:
            pts.append((r["D"], min(rs)))
            log(f"  D {r['D']:.4f}: r (the smallest over {len(rs)} resolved fields) {min(rs):.3e}")
        else:
            log(f"  D {r['D']:.4f}: no field resolved")
    if len(pts) >= 2:
        x = np.array([p[0] for p in pts])
        y = np.log([p[1] for p in pts])
        slope = float(np.polyfit(x, y, 1)[0])
        log(f"rate: ln r falls by {-slope:.3f} per unit of D over {len(pts)} states (#805's weak pivot 1.44; 13.6's "
            f"pole law for a_h 0.34)")
    nres = len(pts)
    log(f"delta_range: {nres} of {len(recs)} states resolved and solutions (max|F| <= {fmax:g})")
    return 0 if nres == len(recs) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

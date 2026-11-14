#!/usr/bin/env python3
"""The type of the chain's nearest real singularity in D: a square-root point (a fold) or a pole (#782).

    python3 singularity_type.py planted [--mutate=move]
    python3 singularity_type.py real JET.json [JET.json ...] [--D_min 256] [--json OUT]
    python3 singularity_type.py rows JET.npz [JET.npz ...] [--rows 24] [--delta]

At a singularity f ~ A (1 - h/s)^mu, h = D - D0, the series' coefficients obey c_k / c_(k-1) = (1/s)(1 - (1 + mu)/k)
when it is alone. mu = 1/2 is a square-root point, where the real branch turns (a fold in D); mu = -1 is a simple pole,
where it ends. Two estimators, each from c_1..c_N:
  ratio  the ratios fitted as (1/s)(1 + b/k + e/k^2) by least squares over the orders k0..N, k0 = 8..14; mu = -1 - b.
         The median over the windows and their spread.
  dlog   Pade approximants [L/M] (M = 3..9, L = M - 1, M, M + 1) of g'/g, g = df/dh (g_k = (k + 1) c_(k+1)): a
         singularity (1 - h/s)^mu is a pole of g'/g at s with residue mu - 1. The approximant's real pole (|Im| below
         5% of its modulus, Re > 0) nearest the origin that is not a doublet (a numerator zero within 1e-6 of it); the
         median over the approximants and their spread. A pole whose residue lies within 0.05 of 0 (a Froissart
         doublet) or of +1 (a zero of g) is skipped.
The quoted spreads are spreads over windows or approximants of one method, not error bars.

planted (the criteria were fixed in #782 before any real series was read by this script; the series' Pade poles at
D 258.78 had been read by #760): a square-root point (S, mu = 1/2) or a simple pole (P, mu = -1) at s1 = 1.17, alone
(clean) and with the contamination #760 saw from D 258.78 (a second real singularity of the same type at 1.39 and a
conjugate pair at |s| 1.45, arg 60 degrees, each of amplitude 0.3 against the first), relative noise 1e-12, N = 24.
Pass: in the clean plants both estimators return mu to 10% (0.05 for S, 0.1 for P) and s to 2%; in the contaminated
plants at least one estimator does, and the other is reported. --mutate=move builds every plant at s1 + 0.5 and judges it
against s1, which must fail. Exit 1 on failure.
The near plant (added before #776's series were read): the same two types at s1 = 259.95 - 259.4928, with a second
real singularity at 260.17 and a conjugate pair at 259.47 +- 1.17i, amplitudes 0.3, judged as the contaminated ones.
First run (2026-11-14): the clean plants pass to 1e-9; the contaminated ones FAIL the 10% bar, the Dlog median giving
mu = +0.574 for S and -0.83 for P, the ratio fit worse. So this estimator is validated to 10% only for an isolated
singularity. Two changes followed, before any real series was read by this script: the residue filter above (it moved
no contaminated median past the bar), and a type band, reported beside the bar and not judged: S when mu lies in
[0.25, 0.75], P when it lies in [-1.25, -0.75], otherwise unclassified.

real: Delta's coefficients (delta_coeffs, double precision) of each locate entry with N = 24 and D0 >= D_min in a
jet_singularity.py summary file (I760-dh370/jet_760.json, I776-dh370/jet_776.json): both estimators, D* = D0 + s.
rows: the same for the 24 entries with the largest |z_8| of each saved series (anm_run.py --save_jet 2), from the
octo-double coefficients; the median and range over rows. --delta adds Delta's own series (the last entry) from the same
file, in octo-double.

Methods: the ratio method of C. Domb and M. F. Sykes, Proc. R. Soc. Lond. A 240, 214 (1957),
doi:10.1098/rspa.1957.0078; Dlog Pade approximants, G. A. Baker, Jr. and P. Graves-Morris, Pade Approximants, 2nd ed.,
Cambridge University Press, 1996, ch. 8.
"""
import json
import statistics
import sys

import mpmath as mp
import numpy as np

DPS = 60
S1, S2, S3, ARG3, AMP = 1.17, 1.39, 1.45, 60.0, 0.3
NOISE = 1e-12
K0S = range(8, 15)
MS = range(3, 10)


# the near plant (#776's last base, D 259.4928, fixed before its series was read): the real singularity at 259.95, a
# second real one at 260.17 and the pair 259.47 +- 1.17i that Delta's Dlog approximants show from D 258.78
NEAR = (259.95 - 259.4928, 260.17 - 259.4928, abs(complex(259.47 - 259.4928, 1.17)),
        float(np.degrees(np.angle(complex(259.47 - 259.4928, 1.17)))))


def plant(mu, s1=S1, contaminated=False, N=24, seed=1, s2=S2, s3=S3, arg3=ARG3):
    """c_1..c_N (mpf) of A (1 - h/s1)^mu, plus the contamination when asked, with relative noise NOISE"""
    rng = np.random.default_rng(seed)
    mu = mp.mpf(mu)
    s3 = mp.mpf(s3) * mp.expj(mp.radians(arg3))
    c = []
    for k in range(1, N + 1):
        b = mp.binomial(mu, k) * (-1) ** k
        v = b / mp.mpf(s1) ** k
        if contaminated:
            v += AMP * b / mp.mpf(s2) ** k + AMP * 2 * mp.re(b / s3 ** k)
        c.append(v * (1 + NOISE * rng.standard_normal()))
    return c


def ratio_fit(c):
    """(s, mu, spread of s, spread of mu) from the ratios over the windows k0..N"""
    v = [mp.mpf(x) for x in c]
    N = len(v)
    out = []
    for k0 in K0S:
        ks = [k for k in range(max(k0, 2), N + 1) if v[k - 2] != 0]
        if len(ks) < 4:
            continue
        A = np.array([[1.0, 1.0 / k, 1.0 / k ** 2] for k in ks])
        y = np.array([float(v[k - 1] / v[k - 2]) for k in ks])
        a0, a1, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        if a0 == 0:
            continue
        out.append((1.0 / a0, -1.0 - a1 / a0))
    if not out:
        return None
    s = [x for x, _ in out]
    m = [x for _, x in out]
    return statistics.median(s), statistics.median(m), max(s) - min(s), max(m) - min(m)


def dlog_series(c):
    """the coefficients of g'/g, g = sum c_k k h^(k-1) = sum g_j h^j, to the order the c allow"""
    v = [mp.mpf(x) for x in c]
    g = [(j + 1) * v[j] for j in range(len(v))]          # g_j = (j + 1) c_(j+1)
    dg = [(j + 1) * g[j + 1] for j in range(len(g) - 1)]
    q = []
    for k in range(len(dg)):
        q.append((dg[k] - sum(g[j] * q[k - j] for j in range(1, k + 1))) / g[0])
    return q


def dlog_fit(c):
    """(s, mu, spread of s, spread of mu, n) from the real pole of the Dlog Pade approximants nearest the origin"""
    q = dlog_series(c)
    res = []
    for M in MS:
        for L in (M - 1, M, M + 1):
            if L + M + 1 > len(q) or L < 0:
                continue
            p, d = mp.pade(q[:L + M + 1], L, M)
            if len(d) < 2:
                continue
            poles = mp.polyroots(d[::-1], maxsteps=200, extraprec=200)
            zeros = mp.polyroots(p[::-1], maxsteps=200, extraprec=200) if len(p) > 1 else []
            dp = [k * d[k] for k in range(1, len(d))]
            cand = []
            for z in poles:
                z = mp.mpc(z)
                if abs(z.imag) > 0.05 * abs(z) or z.real <= 0:
                    continue
                if any(abs(z - w) < 1e-6 * abs(z) for w in zeros):
                    continue                              # a doublet: the approximant's excess degree
                r = mp.re(mp.polyval(p[::-1], z) / mp.polyval(dp[::-1], z))
                if abs(r) < 0.05 or abs(r - 1) < 0.05:
                    continue                              # a Froissart doublet, or a zero of g
                cand.append((z, r))
            if not cand:
                continue
            z, r = min(cand, key=lambda x: abs(x[0]))
            res.append((float(z.real), float(r) + 1.0))
    if not res:
        return None
    s = [x for x, _ in res]
    m = [x for _, x in res]
    return statistics.median(s), statistics.median(m), max(s) - min(s), max(m) - min(m), len(res)


def judge(est, s_true, mu_true):
    if est is None:
        return False
    return abs(est[0] - s_true) <= 0.02 * s_true and abs(est[1] - mu_true) <= 0.1 * abs(mu_true)


def kind(mu):
    return "S" if 0.25 <= mu <= 0.75 else "P" if -1.25 <= mu <= -0.75 else "unclassified"


def fmt(est):
    if est is None:
        return "none"
    return f"s {est[0]:.4f} (spread {est[2]:.1e}), mu {est[1]:+.4f} (spread {est[3]:.1e})"


def planted(argv):
    mp.mp.dps = DPS
    move = "--mutate=move" in argv
    ok = True
    print(f"planted: s1 {S1}" + (" (--mutate=move: built at s1 + 0.5, judged against s1)" if move else ""))
    for name, mu in (("S", 0.5), ("P", -1.0)):
        for cont in (False, True):
            c = plant(mu, S1 + (0.5 if move else 0.0), cont)
            r, d = ratio_fit(c), dlog_fit(c)
            jr, jd = judge(r, S1, mu), judge(d, S1, mu)
            good = (jr and jd) if not cont else (jr or jd)
            ok &= good
            print(f"  {name} {'contaminated' if cont else 'clean':12s} mu {mu:+.1f}: ratio {fmt(r)} -> {'ok' if jr else 'off'}"
                  f" ({kind(r[1]) if r else '-'}); dlog {fmt(d)}{f' over {d[4]} approximants' if d else ''} -> "
                  f"{'ok' if jd else 'off'} ({kind(d[1]) if d else '-'}); {'pass' if good else 'FAIL'}")
    s1, s2, s3, a3 = NEAR
    for name, mu in (("S", 0.5), ("P", -1.0)):
        c = plant(mu, s1 + (0.5 if move else 0.0), True, s2=s2, s3=s3, arg3=a3)
        r, d = ratio_fit(c), dlog_fit(c)
        jr, jd = judge(r, s1, mu), judge(d, s1, mu)
        ok &= jr or jd
        print(f"  {name} near (s1 {s1:.3f}) mu {mu:+.1f}: ratio {fmt(r)} -> {'ok' if jr else 'off'} ({kind(r[1]) if r else '-'}); "
              f"dlog {fmt(d)}{f' over {d[4]} approximants' if d else ''} -> {'ok' if jd else 'off'} ({kind(d[1]) if d else '-'}); "
              f"{'pass' if jr or jd else 'FAIL'}")
    print("planted:", "pass" if ok else "FAIL")
    return 0 if ok else 1


def real(argv):
    mp.mp.dps = DPS
    d_min = float(argv[argv.index("--D_min") + 1]) if "--D_min" in argv else 256.0
    files = [a for a in argv if a.endswith(".json") and not (argv.index(a) > 0 and argv[argv.index(a) - 1] == "--json")]
    out = {}
    print("D0          ratio: D* (s), mu, spreads                      dlog: D* (s), mu, spreads, approximants")
    for f in files:
        for k, L in sorted(json.load(open(f))["locate"].items(), key=lambda x: float(x[0])):
            if L["N"] != 24 or float(k) < d_min or k in out:
                continue
            c = L["delta_coeffs"]
            D0 = L["delta"]["D0"]
            r, d = ratio_fit(c), dlog_fit(c)
            out[k] = dict(D0=D0, ratio=r, dlog=d)
            rs = f"{D0 + r[0]:.3f} ({r[0]:.3f}), mu {r[1]:+.3f} {kind(r[1])}, {r[2]:.1e} {r[3]:.1e}" if r else "none"
            ds = f"{D0 + d[0]:.3f} ({d[0]:.3f}), mu {d[1]:+.3f} {kind(d[1])}, {d[2]:.1e} {d[3]:.1e}, {d[4]}" if d else "none"
            print(f"{D0:9.4f}   {rs:44s}   {ds}")
    if "--json" in argv:
        json.dump(out, open(argv[argv.index("--json") + 1], "w"), indent=1)
    return 0


def rows(argv):
    sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
    from jet_singularity import load_jet, row_series
    mp.mp.dps = DPS
    n = int(argv[argv.index("--rows") + 1]) if "--rows" in argv else 24
    for f in [a for a in argv if a.endswith(".npz")]:
        Z, m = load_jet(f)
        if "--delta" in argv:
            c, _ = row_series(Z, "delta")
            r, d = ratio_fit(c), dlog_fit(c)
            for name, e in (("ratio", r), ("dlog", d)):
                print(f"D0 {m['D']:.4f} Delta {name}: " + (f"D* {m['D'] + e[0]:.4f} (s {e[0]:.4f}), mu {e[1]:+.4f} {kind(e[1])}, "
                      f"spreads {e[2]:.1e} {e[3]:.1e}" if e else "no estimate"), flush=True)
        idx = [int(i) for i in np.argsort(np.abs(np.asarray(Z[8][0])))[::-1][:n]]
        rr, dd = [], []
        for i in idx:
            c, _ = row_series(Z, str(i))
            r, d = ratio_fit(c), dlog_fit(c)
            if r:
                rr.append(r)
            if d:
                dd.append(d)
        D0 = m["D"]
        for name, e in (("ratio", rr), ("dlog", dd)):
            if not e:
                print(f"D0 {D0:.4f} {name}: no row gives an estimate")
                continue
            s = [x[0] for x in e]
            mu = [x[1] for x in e]
            print(f"D0 {D0:.4f} {name}: {len(e)} of {n} rows; D* median {D0 + statistics.median(s):.3f} "
                  f"({D0 + min(s):.3f} to {D0 + max(s):.3f}); mu median {statistics.median(mu):+.3f} ({min(mu):+.3f} to {max(mu):+.3f})")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "planted":
        sys.exit(planted(sys.argv[2:]))
    if cmd == "real":
        sys.exit(real(sys.argv[2:]))
    if cmd == "rows":
        sys.exit(rows(sys.argv[2:]))
    sys.exit(__doc__)

"""Which side of the 0.5-4.3e-6 gap between our chain's Delta and EEG26b's Table I at D 100-200 does the D <= 61
1/D series favour? (#1086, notes/LARGE_D_BASICS.md section 6)

Fits a polynomial in 1/D of degree DEG by least squares (mpmath QR at 60 digits) to the double-double points
10 <= D <= 61.1 of delta_all.csv (kind dd, every harmonic solved or pinned above 13 only), evaluates it at the D of
the chain points that carry a published EEG value, and prints chain - fit and EEG - fit. The fit's own extrapolation
spread is shown by two degrees. This is not an independent computation of Delta: the fit and the chain share the
solver; the chain's held harmonics and the fit's points do not share the held fit.

With --at=FILE (lines "D Delta ks path", as sigma_vs_ks.py prints them: columns 1, 6, 2, 8) it compares those states
with the fits instead of the EEG rows.

Usage: python3 eeg_gap.py DELTA_ALL_CSV [--deg=16,18] [--at=FILE] [--mutate=shift]
  --mutate=shift: control; adds 1e-6 to every fitted point, which must move chain - fit by -1e-6.
"""
import csv
import sys

import mpmath as mp

mp.mp.dps = 60


def fit(points, deg):
    A = mp.matrix(len(points), deg + 1)
    b = mp.matrix(len(points), 1)
    for i, (D, v) in enumerate(points):
        y = 1 / mp.mpf(D)
        for k in range(deg + 1):
            A[i, k] = y ** k
        b[i] = v
    c, res = mp.qr_solve(A, b)
    return c


def ev(c, D):
    y = 1 / mp.mpf(D)
    return sum(c[k] * y ** k for k in range(len(c)))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    degs = [int(d) for d in opts.get("deg", "16,18").split(",")]
    shift = mp.mpf("1e-6") if opts.get("mutate") == "shift" else 0
    rows = list(csv.DictReader(open(args[0])))
    pts = [(mp.mpf(r["D"]), mp.mpf(r["Delta_q0"]) + mp.mpf(r["Delta_q1"] or 0) + shift)
           for r in rows if r["kind"] == "dd" and 10 <= float(r["D"]) <= 61.1]
    fits = {d: fit(pts, d) for d in degs}
    print("# %d fitted points; limit D->inf: %s" % (len(pts), "  ".join("deg %d %s" % (d, mp.nstr(fits[d][0], 12))
                                                                       for d in degs)))
    if "at" in opts:
        print("# D  Delta  ks  " + "  ".join("state-fit(d%d)" % d for d in degs) + "  path")
        for line in open(opts["at"]):
            if line.startswith("#"):
                continue
            w = line.split()
            D, X = mp.mpf(w[0]), mp.mpf(w[5])
            print("%s %s %s  %s  %s" % (w[0], w[5], w[1], "  ".join("%+.2e" % float(X - ev(fits[d], D)) for d in degs),
                                        w[7]))
        return
    print("# D  chain  EEG  " + "  ".join("chain-fit(d%d) EEG-fit(d%d)" % (d, d) for d in degs))
    for r in rows:
        if r["published_ref"] != "EEG2026" or float(r["D"]) < 62:
            continue
        D = mp.mpf(r["D"])
        ch = mp.mpf(r["Delta_q0"]) + mp.mpf(r["Delta_q1"] or 0)
        ee = mp.mpf(r["published_Delta"])
        cols = []
        for d in degs:
            f = ev(fits[d], D)
            cols.append("%+.2e %+.2e" % (float(ch - f), float(ee - f)))
        print("%s %s %s %s  %s" % (mp.nstr(D, 7), mp.nstr(ch, 12), mp.nstr(ee, 7), r["kind"], "  ".join(cols)))


if __name__ == "__main__":
    main()

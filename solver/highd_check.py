"""Held-out test of the large-D series: fit D <= 30.2 (final_fit.step_converged, weighted series in x / max(x)), predict
the x_K 0.7 junk-solved points above, then refit with them included.

    python3 ../highd_check.py            (from solver/data)
"""
import numpy as np

from final_fit import step_converged, weighted_series

FAMILIES = [(lo, xn, xf, m) for lo in (10, 12) for xn, xf in (("1/D", lambda d: 1 / d), ("1/(D-1)", lambda d: 1 / (d - 1)),
                                                                ("1/(D-2)", lambda d: 1 / (d - 2))) for m in (9, 10)]


def main():
    D, y, s, g = step_converged()
    lo_m, hi_m = D <= 30.2, D > 30.2
    print(f"{lo_m.sum()} points D <= 30.2, {hi_m.sum()} held out: D {D[hi_m].min():.3f} .. {D[hi_m].max():.3f}")
    preds, lims = [], []
    for lo, xn, xf, m in FAMILIES:
        k = lo_m & (D >= lo)
        c, chi = weighted_series(xf(D[k]), y[k], 1 / s[k], m)
        preds.append(np.polyval(c[::-1], xf(D[hi_m])))
        lims.append(c[0])
    preds = np.array(preds)
    mid = np.median(preds, 0)
    print(f"{'D':>8} {'Delta':>14} {'- median':>10} {'family spread':>14}")
    for d, v, p, sp in zip(D[hi_m], y[hi_m], mid, preds.max(0) - preds.min(0)):
        print(f"{d:8.3f} {v:14.10f} {v - p:+10.1e} {sp:14.1e}")
    print(f"Delta_inf from D <= 30.2: {min(lims):.6f} .. {max(lims):.6f}")
    lims2 = []
    for lo, xn, xf, m in FAMILIES:
        k = D >= lo
        c, chi = weighted_series(xf(D[k]), y[k], 1 / s[k], m)
        lims2.append((c[0], chi, lo, xn, m))
    for c0, chi, lo, xn, m in lims2:
        print(f"  all data, D >= {lo} {xn:8s} m = {m:2d}: Delta_inf {c0:.6f}  chi2/dof {chi:.1f}")
    print(f"Delta_inf with the held-out points: {min(l[0] for l in lims2):.6f} .. {max(l[0] for l in lims2):.6f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Graph how the large-D behaviour is an assumption: several formulas that all reproduce the computed points
(D = 4-40) to their precision, extended to D = 10^5 on a log axis, with the limits they head for.

  - the 1/D power series (degree 10 and 11) and the all-D rational formula: analytic in 1/D at D = infinity;
  - power series in n^(-1/2) (degree 14, 16, 18): a non-analytic approach (half-integer powers of 1/n);
  - the rational formula with its limit forced to pi^2/6 or sqrt(e) and one extra denominator term, which it uses for a
    pole at D ~ -400 / -6000: a slow bend that starts in the hundreds.

All of them fit 10 <= D <= 30.2 (the rational ones 4 <= D <= 30.2) and predict the held-out D = 34-40 points; they
agree to 1e-5 up to D ~ 60 and fan out above D ~ 100.

Usage (from solver/data):
    python3 ../plot_extrapolation.py [-o ../extrapolation.svg]
"""
import argparse
import math
import os
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from final_fit import weighted_series  # noqa: E402
from formula_all import fit_delta  # noqa: E402
from plot_delta import BAND, DINF_DD, GRID, INK, INK_SOFT, NEUTRAL, SERIES, Svg, dd_points, esc, marker  # noqa: E402
from rational_fit import data, rat  # noqa: E402

LX0, LX1 = math.log10(20), 5.0     # log10 D range (the computed points go down to D = 4, off the left edge)
Y0, Y1 = 1.58, 2.15


def forced_far_pole(D, y, s, v, starts=8):
    """P8/Q3 rational in 1/(D-2) on 4 <= D <= 30.2 with the limit forced to v; best of several starts, no pole at D > 2."""
    pf, qf = fit_delta()
    m = (D >= 4) & (D <= 30.2)
    x, w = 1 / (D[m] - 2), 1 / np.maximum(s[m], 1e-7)
    rng = np.random.default_rng(1)
    base = np.r_[pf[1:], qf, 0.0]
    best = None
    for i in range(starts):
        c0 = base * (1 + (0 if i == 0 else 0.3 * rng.standard_normal(len(base))))
        f = lambda c: (rat(np.r_[v, c[:8]], c[8:], x) - y[m]) * w
        try:
            r = least_squares(f, c0, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=20000)
        except Exception:
            continue
        q = r.x[8:]
        poles = [2 + 1 / z.real for z in np.roots(np.r_[1, q][::-1]) if abs(z.imag) < 1e-9 and abs(z.real) > 1e-12]
        if any(pp > 2 for pp in poles):
            continue
        chi = np.sum(r.fun ** 2)
        if best is None or chi < best[0]:
            best = (chi, np.r_[v, r.x[:8]], q, poles)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--output", default="../extrapolation.svg")
    args = ap.parse_args()
    D, y, s, _ = data()
    fit = (D >= 10) & (D <= 30.2)

    models = []                                   # (label, color, dash, function of D, limit)
    for deg, dash in ((10, None), (11, "8 3")):
        c, _ = weighted_series(1 / D[fit], y[fit], 1 / s[fit], deg)
        models.append((f"1/D series, degree {deg}", SERIES[0], dash, lambda d, c=c: np.polyval(c[::-1], 1 / d), c[0]))
    pf, qf = fit_delta()
    models.append(("all-D rational formula (README)", SERIES[0], "2 3", lambda d: rat(pf, qf, 1 / (d - 2)), pf[0]))
    for deg, dash in ((14, None), (16, "8 3"), (18, "2 3")):
        c, _ = weighted_series((D[fit] - 2) ** -0.5, y[fit], 1 / s[fit], deg)
        models.append((f"series in n^(-1/2), degree {deg}", SERIES[2], dash,
                       lambda d, c=c: np.polyval(c[::-1], (d - 2) ** -0.5), c[0]))
    for name, v, color in (("π²/6", math.pi ** 2 / 6, SERIES[1]), ("√e", math.e ** 0.5, SERIES[4])):
        b = forced_far_pole(D, y, s, v)
        if b:
            _, p, q, poles = b
            far = min(poles, key=lambda pp: -abs(pp))
            models.append((f"rational, limit forced to {name} (pole at D ≈ {far:.0f})", color, None,
                           lambda d, p=p, q=q: rat(p, q, 1 / (d - 2)), v))

    W, px, py, pw, ph = 1080, 84, 186, 820, 480
    svg = Svg(W)
    ddp = dd_points(40.35, every=2.0)
    Dtop = float(dd_points(40.35, every=0.0)[:, 0].max())
    svg.text(16, 28, "The large-D forms the f64 points allowed, and the one the double-double points leave",
             size=15, fill=INK, weight="bold")
    svg.text(16, 48, "Every curve reproduces the f64 Δ for D = 4–40 to its precision; they agree to 1e-5 up to D ≈ 60 "
                     "and fan out above D ≈ 100 toward different limits.", size=12)
    svg.text(16, 66, f"The double-double points up to D = {Dtop:.0f} decide it: fitted to them, half-integer series predict "
                     "unseen points 40–300× worse than the 1/D series and forced limits 100–3500× worse (README).",
             size=12)
    svg.text(16, 84, f"The analytic family (blue) is the one left, with Δ∞ = {DINF_DD:.9f}; the faded curves are ruled out.",
             size=12, fill=INK)
    sx = lambda d: px + (math.log10(d) - LX0) / (LX1 - LX0) * pw
    sy = lambda v: py + ph - (v - Y0) / (Y1 - Y0) * ph
    # computed range
    a, b = sx(20), sx(Dtop)
    svg.add(f'<rect x="{a:.1f}" y="{py}" width="{b - a:.1f}" height="{ph}" fill="{BAND}"/>')
    svg.text((a + b) / 2, py + 14, "computed", size=10, anchor="middle")
    svg.text((a + b) / 2, py + 28, f"(D = 4–{Dtop:.0f})", size=10, anchor="middle")
    for v in np.arange(1.6, 2.151, 0.05):
        svg.line(px, sy(v), px + pw, sy(v), GRID)
        svg.text(px - 8, sy(v) + 4, f"{v:.2f}", size=10, anchor="end")
    for e in range(1, 6):
        for mant in (1, 2, 5):
            d = mant * 10 ** e
            if 20 <= d <= 10 ** LX1:
                svg.line(sx(d), py, sx(d), py + ph, GRID)
                svg.text(sx(d), py + ph + 16, f"{d:g}" if d < 1e4 else f"1e{e}" if mant == 1 else f"{mant}e{e}", size=10,
                         anchor="middle")
    svg.line(px, py + ph, px + pw, py + ph, INK_SOFT)
    svg.text(px + pw / 2, py + ph + 38, "spacetime dimension D (log scale)", size=11, anchor="middle")
    svg.add(f'<text x="{px - 50:.1f}" y="{py + ph / 2:.1f}" font-size="11" fill="{INK_SOFT}" text-anchor="middle" '
            f'transform="rotate(-90 {px - 50:.1f} {py + ph / 2:.1f})">echo period Δ</text>')
    # candidate limits as reference lines, labelled at the right
    refs = (("π²/6", math.pi ** 2 / 6), ("√e", math.e ** 0.5), ("2π/j₁,₁", 2 * math.pi / 3.8317059702075125),
            ("φ", (1 + 5 ** 0.5) / 2), ("Δ∞ (double-double)", DINF_DD))
    ys_lab = []
    for name, v in sorted(refs, key=lambda r: -r[1]):              # top to bottom, labels pushed 13 px apart
        dd_ref = name.startswith("Δ∞")
        svg.line(px, sy(v), px + pw, sy(v), SERIES[0] if dd_ref else NEUTRAL, dash=None if dd_ref else "2 3",
                 opacity=0.9 if dd_ref else 0.7)
        yl = max(sy(v) + 4, ys_lab[-1] + 13) if ys_lab else sy(v) + 4
        ys_lab.append(yl)
        svg.text(px + pw + 6, yl, f"{name} = {v:.9f}" if name.startswith("Δ∞") else f"{name} = {v:.4f}", size=10,
                 fill=INK if name.startswith("Δ∞") else INK_SOFT)
    # models
    dd = np.logspace(LX0, LX1, 700)
    for label, color, dash, f, lim in models:
        vals = f(dd)
        ok = (vals >= Y0) & (vals <= Y1)
        pts = " ".join(f"{'M' if i == 0 or not ok[i - 1] else 'L'}{sx(d):.1f},{sy(v):.1f}"
                       for i, (d, v) in enumerate(zip(dd, vals)) if ok[i])
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        out = color != SERIES[0]                                    # everything but the analytic family
        svg.add(f'<g><path d="{pts}" fill="none" stroke="{color}" stroke-width="2" stroke-opacity="{0.35 if out else 0.9}"'
                f'{extra}/><title>{esc(label)}{" (ruled out by the double-double points)" if out else ""}: limit {lim:.5f}; '
                f'Δ(100) = {f(100.0):.5f}, Δ(1000) = {f(1000.0):.5f}</title></g>')
    for d, v in zip(D, y):
        if 20 <= d and Y0 <= v <= Y1:
            marker(svg, sx(d), sy(v), INK_SOFT, f"D = {d:.3f}, Δ = {v:.10f}", r=3.2)
    for d, v in ddp:
        marker(svg, sx(d), sy(v), INK_SOFT, f"D = {d:.3f}, Δ = {v:.13f} (double-double)", shape="diamond", r=3.0)
    # legend
    kx, ky = px, py - 46
    for label, color, dash in (("analytic in 1/D: 1/D series, all-D formula", SERIES[0], None),
                               ("series in n^(-1/2): ruled out", SERIES[2], None),
                               ("limit forced to π²/6: ruled out", SERIES[1], None),
                               ("limit forced to √e: ruled out", SERIES[4], None)):
        svg.line(kx, ky - 4, kx + 22, ky - 4, color, width=2, dash=dash)
        svg.text(kx + 28, ky, label, size=10)
        kx += 28 + 5.6 * len(label) + 18
    svg.text(px, ky + 20, "dash styles within a family: different degrees (hover a curve for its limit and values); "
                          "points: f64 (dots) and double-double (diamonds)", size=10)
    ty = py + ph + 64
    svg.text(16, ty, "Limits:", size=11, fill=INK)
    col = 0
    for label, color, dash, f, lim in models:
        svg.line(80 + col * 330, ty + 16 - 4, 100 + col * 330, ty + 16 - 4, color, width=2, dash=dash)
        svg.text(106 + col * 330, ty + 16, f"{lim:.5f}  {label}", size=10)
        col += 1
        if col == 3:
            col, ty = 0, ty + 18
    ty += 40
    svg.text(16, ty, "solver/plot_extrapolation.py. Points: the computed Δ for D ≥ 20 (f64), and the double-double points "
                     "above D = 40 (hp/data/delta_dd.csv). The curves are the fits to the f64 points.", size=10)
    open(args.output, "w").write(svg.finish(ty + 16))
    print(f"wrote {args.output}")
    for label, color, dash, f, lim in models:
        print(f"  {label:58s} limit {lim:.5f}  D=60 {f(60.0):.6f}  D=100 {f(100.0):.6f}  D=1000 {f(1000.0):.5f}")


if __name__ == "__main__":
    main()

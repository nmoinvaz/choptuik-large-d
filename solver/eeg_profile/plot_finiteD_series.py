#!/usr/bin/env python3
"""How far the finite-D ratios' D = infinity limits lie from the tuned profile's, as one SVG (agent-comms#38).

    cd solver/eeg_profile && python3 plot_finiteD_series.py STATEDIR [-o .]

finiteD_series.svg, two panels.
  Left: the polynomial fit's limit |r_nu(0) - r_nu(tuned)| against its degree K, past the grid (eps, D >= 15, step at
        35), for nu = 1, 3, 5, 7. The grid G1 is K 7-9 (shaded).
  Right: for nu = 3, 5, 7, the grid's median with its spread s (finiteD_series.py states), against the intercept that
        the shape (r - c)/eps = P_L/Q_M fits best over [L/M] = [3/3], [4/3], [5/3], [4/4] (finiteD_series.py info,
        post hoc).
STATEDIR as in finiteD_series.py. Standard library and numpy; the primitives are hp/data_graphs.py. Prints the
plotted numbers.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "hp"))
import finiteD_series as FS  # noqa: E402
from data_graphs import INK, INK_SOFT, SERIES, SHAPES, Svg, axes, better_hint, callout, footnote, legend, marker  # noqa: E402

REPO = "https://github.com/nmoinvaz/choptuik-large-d"
KS = list(range(5, 13))
LM = ((3, 3), (4, 3), (5, 3), (4, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("statedir")
    ap.add_argument("-o", default=HERE)
    a = ap.parse_args()
    recs = FS.add_ratios(FS.load_states(a.statedir))
    tuned = FS.tuned_ratios(FS.A_TARGET)

    seq = {nu: [(k, abs(v - tuned[nu])) for k, v in FS.k_sequence(recs, nu, ks=KS)] for nu in FS.NU_ACC}
    print("| K | " + " | ".join(f"r_{nu}" for nu in FS.NU_ACC) + " |\n|---|" + "---|" * len(FS.NU_ACC))
    for i, k in enumerate(KS):
        print(f"| {k} | " + " | ".join(f"{seq[nu][i][1]:.2e}" for nu in FS.NU_ACC) + " |")
    right = {}
    for nu in (3, 5, 7):
        g = FS.grid_fit(recs, nu)
        cs = [abs(FS.best_intercept(recs, nu, L, M, [tuned[nu], g["median"]])[0] - tuned[nu]) for L, M in LM]
        right[nu] = dict(med=abs(g["median"] - tuned[nu]), s=g["s"], cs=cs)
        print(f"r_{nu}: grid median - tuned {right[nu]['med']:.2e}, s {g['s']:.2e}; best rational intercept - tuned "
              + ", ".join(f"[{L}/{M}] {c:.2e}" for (L, M), c in zip(LM, cs)))

    W = 1040
    svg = Svg(W)
    svg.text(16, 26, "The finite-D harmonic ratios' D = ∞ limits against the tuned profile", size=15, fill=INK,
             weight="bold")
    svg.text(16, 46, "r_ν = F_(ν+2)/F_ν of the centre profile, fitted in ε = 1/(D − 1) over the full-system states "
             "D 15–73 (r₁ also the chain to D 215), extrapolated to ε = 0.", size=11)
    svg.text(16, 61, "r₁ and r₃ settle on the tuned values. r₅ and r₇ are still moving toward them as the degree "
             "grows, the sign of a singularity nearer to ε = 0 than the data reach.", size=11)

    # left: the K sequence
    x, y, w, h = 96, 120, 380, 300
    svg.text(x, y - 14, "Polynomial fit: distance of its limit from the tuned ratio", size=12, fill=INK)
    xm, ym = axes(svg, x, y, w, h, (KS[0] - 0.5, KS[-1] + 0.5), (1e-7, 1e-2, "log"),
                  "polynomial degree K in ε", "|r_ν(0) − r_ν(tuned)|", xfmt=lambda v: f"{v:g}",
                  yfmt=lambda v: f"{v:.0e}")
    svg.add(f'<rect x="{xm(6.5):.1f}" y="{y}" width="{xm(9.5) - xm(6.5):.1f}" height="{h}" fill="{INK_SOFT}" '
            f'opacity="0.10"/>')
    svg.text(xm(8), y + 14, "grid G1", size=10, anchor="middle")
    items = []
    for i, nu in enumerate(FS.NU_ACC):
        col = SERIES[i]
        pts = [(xm(k), ym(max(d, 1.01e-7))) for k, d in seq[nu]]
        svg.add('<polyline points="' + " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
                + f'" fill="none" stroke="{col}" stroke-width="2"/>')
        for (k, d), (px, py) in zip(seq[nu], pts):
            marker(svg, SHAPES[i], px, py, col, f"r_{nu}, K {k}: {d:.2e}")
        items.append((f"r{'₁₃₅₇'[i]}", col))
    lx = x + w - 4
    for lab, col in reversed(items):
        svg.text(lx, y + h - 10, lab, size=11, fill=INK, anchor="end")
        lx -= len(lab) * 7 + 6
        svg.add(f'<circle cx="{lx:.1f}" cy="{y + h - 14}" r="4.5" fill="{col}"/>')
        lx -= 14
    k12 = seq[7][-1]
    callout(svg, xm(k12[0]), ym(k12[1]), "r₇: 1.7e-3 at K 12", dx=-30, dy=-34)
    better_hint(svg, x, y, w, h, where="right", label="closer", reverse=True)

    # right: grid median and spread against the rational intercepts
    x2, w2 = 620, 360
    svg.text(x2, y - 14, "D = ∞ limit: polynomial grid against rational shapes", size=12, fill=INK)
    xm2, ym2 = axes(svg, x2, y, w2, h, (0.5, 3.5), (1e-6, 1e-2, "log"), "ratio",
                    "|limit − r_ν(tuned)|", xfmt=lambda v: {1: "r₃", 2: "r₅", 3: "r₇"}.get(round(v), ""),
                    yfmt=lambda v: f"{v:.0e}")
    for j, nu in enumerate((3, 5, 7)):
        cx = xm2(j + 1)
        r = right[nu]
        svg.line(cx - 22, ym2(r["s"]), cx - 6, ym2(r["s"]), SERIES[1], 2)
        marker(svg, "circle", cx - 14, ym2(r["med"]), SERIES[0], f"r_{nu}: grid median - tuned {r['med']:.2e}")
        for q, c in enumerate(r["cs"]):
            marker(svg, "diamond", cx + 6 + 6 * q, ym2(max(c, 1.01e-6)), SERIES[2],
                   f"r_{nu} [{LM[q][0]}/{LM[q][1]}]: best intercept - tuned {c:.2e}")
    legend(svg, x2 + w2, y + h + 52, [("grid median", SERIES[0]), ("spread s", SERIES[1]),
                                       ("rational shapes, post hoc", SERIES[2])])
    svg.text(x2 + w2, y + h + 68, "rational: (r − c)/ε = P_L/Q_M, [L/M] = [3/3], [4/3], [5/3], [4/4], left to right", size=10,
             anchor="end")
    callout(svg, xm2(2) - 14, ym2(right[5]["med"]), "r₅ outside s", dx=-50, dy=-30)
    better_hint(svg, x2, y, w2, h, where="right", label="closer", reverse=True)

    body = y + h + 84
    hgt = footnote(svg, ["states: hp/data/states, raw T84-mac, T81, T138-win2 (b904462b)",
                         "fits: solver/eeg_profile/finiteD_series.py", "tuned: scan45J1500 at A 2.71918"],
                   "", [], body, REPO)
    out = os.path.join(a.o, "finiteD_series.svg")
    open(out, "w").write(svg.finish(hgt))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

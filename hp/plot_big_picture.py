#!/usr/bin/env python3
"""The big picture of the large-D computation in six panels: the problem, and how the held-harmonics fit solves it.

    cd hp && python3 plot_big_picture.py RAWDIR [-o .]

big_picture.svg, read left to right and top to bottom:
  1. every D has its own critical solution and echo period: Delta(D) from data/delta_all.csv;
  2. a solution is a wave over one echo period, a sum of odd harmonics: p1 at the clean chain's state D 232.63
     (raw I461-win2/crawlG) and the size of each harmonic, by role (solved, held, zero);
  3. past D 73 the full equations barely fix the higher harmonics: the singular values of the Schur complement of the
     held rows, from the exact Jacobian (data/held_grip.csv, made by held_grip.py from held_schur.py --jac=exact, raw
     I479-mac at dd9ba550);
  4. the solutions form a smooth family: sqrt(D-2) a5 of the 138 full solutions against 1/(D-2), with the fit of
     data/held_fit.json carried past D 73;
  5. the chain holds harmonics 5-13 on that fit and steps in D: the size of each Taylor term |z_k| a^k of one step
     (crawlG's anm_run.jsonl, base D 228.13125) and the corrector's move;
  6. the price: Delta's error bar, the change with resolution of the full system against the spread over the fit's
     degree on the chain (delta_all.csv's err, err_kind check and fits).
Every number is read from those files and printed. Standard library and numpy; the primitives are data_graphs.py.
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data_graphs as dg  # noqa: E402
import plot_algorithm as pa  # noqa: E402
from data_graphs import INK, INK_SOFT, SERIES, Svg, axes, footnote, marker  # noqa: E402
from held_fit import full_states  # noqa: E402
from hpkernel.driver import load_q  # noqa: E402
from pinning import HeldFit  # noqa: E402

REPO = "https://github.com/nmoinvaz/choptuik-large-d"
FRONT = "I461-win2/crawlG/anm_pin3_D232.6312.npz"
JSONL = "I461-win2/crawlG/anm_run.jsonl"
BASE_D = 228.13125
M = 32
KS = list(range(1, 15, 2))
TICKS = {}        # (lo, hi) of an axis -> its tick values, for the axes that need their own
PI_FMT = {0.0: "0", np.pi: "π", 2 * np.pi: "2π"}


class Ax(dg.Axis):
    def ticks(self):
        if (self.lo, self.hi) in TICKS:
            return TICKS[(self.lo, self.hi)]
        if self.log:
            e0, e1 = math.ceil(math.log10(self.lo)), math.floor(math.log10(self.hi))
            step = max(1, math.ceil((e1 - e0) / 5))
            return [10.0 ** e for e in range(e0, e1 + 1) if e % step == 0]
        return super().ticks()


dg.Axis = Ax


def coeffs(f):
    th = np.pi * np.arange(M) / M
    return {k: ((2 / M) * np.sum(f * np.cos(k * th)), (2 / M) * np.sum(f * np.sin(k * th))) for k in KS}


def p10(v):
    return pa.pow10(math.log10(v))


def sci(v, digits=1):
    """2.4×10⁻¹¹"""
    e = math.floor(math.log10(abs(v)))
    m = v / 10 ** e
    return f"{m:.{digits}f}×{pa.pow10(e)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("-o", default=HERE)
    a = ap.parse_args()

    # data
    rows = list(csv.DictReader(open(os.path.join(HERE, "data", "delta_all.csv"))))
    full_kinds = ("dd", "qd", "pinned")
    dfull = sorted((float(r["D"]), float(r["Delta_q0"])) for r in rows if r["kind"] in full_kinds)
    dchain = sorted((float(r["D"]), float(r["Delta_q0"])) for r in rows if r["kind"] == "chain_fit")
    d4 = min(dfull, key=lambda p: abs(p[0] - 4.0))
    err_full = sorted((float(r["D"]), float(r["err"])) for r in rows if r["kind"] == "dd" and r["err_kind"] == "check")
    err_chain = sorted((float(r["D"]), float(r["err"])) for r in rows if r["kind"] == "chain_fit" and r["err_kind"] == "fits"
                       and float(r["err"]) > 0)
    zf, mf = load_q(os.path.join(a.raw, FRONT))
    f = np.asarray(zf[0], float) + np.asarray(zf[1], float)
    co = coeffs(f[:M])
    D_front = float(mf["D"])
    sizes = {k: math.hypot(*co[k]) for k in KS}
    ratios = [sizes[k] / sizes[k + 2] for k in KS[:-1]]
    grip = list(csv.DictReader(open(os.path.join(HERE, "data", "held_grip.csv"))))
    gD = sorted({float(r["D"]) for r in grip})
    fit = HeldFit(os.path.join(HERE, "data", "held_fit.json"))
    fitd = json.load(open(os.path.join(HERE, "data", "held_fit.json")))
    pts = [p for p in full_states(a.raw) if float(p["D"]) >= float(fitd["Dmin"]) - 1e-6]
    rec = next(json.loads(l) for l in open(os.path.join(a.raw, JSONL)) if l.strip() and abs(json.loads(l)["D0"] - BASE_D) < 1e-9)
    acc = next(x for x in rec["attempts"] if x["accepted"])
    step = float(acc["a"])
    terms = [rec["norms"][k] * step ** k for k in range(1, len(rec["norms"]))]

    def sq_a5(D, v):
        return math.sqrt(D - 2) * v
    fit_miss = max(abs(sq_a5(float(p["D"]), p["c"]["5a"]) - sq_a5(float(p["D"]), fit.values(float(p["D"]), [5])[0][0])) for p in pts)

    print(f"1. Delta: full system {len(dfull)} points D {dfull[0][0]:g}-{dfull[-1][0]:g}; chain {len(dchain)} points to D {dchain[-1][0]:g}; "
          f"D {d4[0]:g}: Delta {d4[1]:.6f}")
    print(f"2. D {D_front:g}: |harmonic k| " + " ".join(f"{k}:{sizes[k]:.2e}" for k in KS) + "; ratios " + " ".join(f"{r:.1f}" for r in ratios))
    print("3. grip (exact singular values of S) by D and harmonic:")
    for D in gD:
        print(f"   D {D:g}: " + " ".join(f"{r['harmonic']}:{float(r['sigma_exact']):.2e}" for r in grip if float(r["D"]) == D))
    print(f"4. {len(pts)} full solutions; largest |sqrt(D-2) a5 - fit| {fit_miss:.1e}")
    print(f"5. base D {BASE_D}, step a {step}, Newton steps {acc['steps']}, corrector move {acc['moved']:.1e}; |z_k| a^k: "
          + " ".join(f"{t:.0e}" for t in terms))
    print(f"6. err: full system {min(e for _, e in err_full):.1e}..{max(e for _, e in err_full):.1e}; "
          f"chain fits {min(e for _, e in err_chain):.1e}..{max(e for _, e in err_chain):.1e}")

    W = 1040
    svg = Svg(W)
    svg.text(16, 26, "The big picture: what is computed, why it gets hard past D 73, and how a fit gets around it", size=15, fill=INK,
             weight="bold")
    svg.text(16, 44, "Read left to right, then the second row. Each panel has a detailed figure further down this page.", size=11,
             fill=INK_SOFT)
    CW, PX, PW, PH = 344, 62, 258, 168
    rows_y = [(78, "The problem"), (408, "Our solution")]

    def cell(i, row):
        x0 = 16 + i * CW
        y0 = rows_y[row][0]
        return x0, y0 + 60, x0 + PX, PW, PH

    def claim(i, row, num, l1, l2):
        x0, ytop, _, _, _ = cell(i, row)
        svg.text(x0, ytop - 30, f"{num}  {l1}", size=12, fill=INK, weight="bold")
        svg.text(x0 + 17, ytop - 15, l2, size=12, fill=INK, weight="bold")

    for y, label in rows_y:
        svg.text(16, y, label.upper(), size=11, fill=SERIES[1], weight="bold")
        svg.line(110, y - 4, W - 16, y - 4, INK_SOFT, 0.5)

    # 1. Delta against D
    _, y, x, w, h = cell(0, 0)
    claim(0, 0, "1.", "Each D has its own critical solution,", "with its own echo period Δ")
    xm, ym = axes(svg, x, y, w, h, (0, 240), (1.5, 3.7), "spacetime dimension D", "echo period Δ", xfmt=lambda v: f"{v:g}",
                  yfmt=lambda v: f"{v:.1f}")
    svg.add(f'<path d="{pa.path_d([(xm(D), ym(v)) for D, v in dfull])}" fill="none" stroke="{SERIES[2]}" stroke-width="2.2"/>')
    svg.add(f'<path d="{pa.path_d([(xm(D), ym(v)) for D, v in dchain])}" fill="none" stroke="{SERIES[0]}" stroke-width="2.2"/>')
    marker(svg, "circle", xm(d4[0]), ym(d4[1]), SERIES[1], f"D {d4[0]:g}: Δ {d4[1]:.6f}")
    svg.text(xm(d4[0]) + 10, ym(d4[1]) + 4, f"D 4, our world: Δ = {d4[1]:.3f}", size=11, fill=INK)
    svg.add(f'<line x1="{xm(73.27):.1f}" y1="{y}" x2="{xm(73.27):.1f}" y2="{y + h}" stroke="{INK_SOFT}" stroke-dasharray="4 3"/>')
    svg.text(xm(73.27) + 6, y + h - 52, "full system", size=10, fill=SERIES[2])
    svg.text(xm(73.27) + 6, y + h - 40, f"to D {dfull[-1][0]:.0f}, then", size=10, fill=SERIES[2])
    svg.text(xm(73.27) + 6, y + h - 28, f"the chain to D {dchain[-1][0]:.0f}", size=10, fill=SERIES[0])

    # 2. the wave and its harmonics
    _, y, x, w, h = cell(1, 0)
    claim(1, 0, "2.", "A solution is a wave over one period,", f"odd harmonics, each {min(ratios):.0f}–{max(ratios):.0f}× smaller")
    th = np.linspace(0, 2 * np.pi, 161)
    wv = sum(a_ * np.cos(k * th) + b_ * np.sin(k * th) for k, (a_, b_) in co.items())
    amp = float(np.max(np.abs(wv)))
    wy, wh = y + 4, 46
    svg.line(x, wy + wh / 2, x + w, wy + wh / 2, INK_SOFT, 0.5)
    svg.add(f'<path d="{pa.path_d([(x + w * t / (2 * np.pi), wy + wh / 2 - (wh / 2) * v / amp) for t, v in zip(th, wv)])}" '
            f'fill="none" stroke="{INK}" stroke-width="2"/>')
    svg.text(x + w, wy + wh + 13, f"p1 over one echo period, D {D_front:.1f}", size=10, fill=INK_SOFT, anchor="end")
    by, bh = y + wh + 18, h - wh - 18
    xb, yb = axes(svg, x, by, w, bh, (0, 16), (1e-8, 1, "log"), "harmonic k", "size of harmonic k", xfmt=lambda v: f"{v:g}", yfmt=p10)
    bw = 0.55 * (xb(2) - xb(0))
    for k in KS:
        col = SERIES[0] if k <= 3 else SERIES[1]
        top = yb(sizes[k])
        svg.add(f'<rect x="{xb(k) - bw / 2:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{by + bh - top:.1f}" fill="{col}">'
                f'<title>harmonic {k}: {sizes[k]:.3e}</title></rect>')
    svg.text(xb(15), by + bh - 6, "0", size=10, fill=INK_SOFT, anchor="middle")
    svg.text(xb(3) + bw / 2 + 4, yb(sizes[3]) + 10, "solved", size=10, fill=SERIES[0])
    svg.text(xb(7) + 6, yb(sizes[5]) + 2, "held on the fit", size=10, fill=SERIES[1])
    svg.text(xb(15), by + bh - 18, "15+", size=10, fill=INK_SOFT, anchor="middle")

    # 3. the full equations' grip on each held harmonic
    _, y, x, w, h = cell(2, 0)
    claim(2, 0, "3.", "Past D 73 the equations barely fix", "the higher harmonics")
    sig = [float(r["sigma_exact"]) for r in grip]
    lo, hi = 10 ** math.floor(math.log10(min(sig))), 10 ** (math.ceil(math.log10(max(sig))) + 6)   # room for the note
    TICKS[(4, 14)] = [5, 7, 9, 11, 13]
    xg, yg = axes(svg, x, y, w, h, (4, 14), (lo, hi, "log"), "held harmonic k", "the equations' grip on it",
                  xfmt=lambda v: f"{v:g}", yfmt=p10)
    per = []
    for i, D in enumerate(gD):
        rr = [r for r in grip if float(r["D"]) == D]
        by_k = {}
        for r in rr:
            by_k.setdefault(int(r["harmonic"]), []).append(float(r["sigma_exact"]))
        ks = sorted(by_k)
        gm = [math.sqrt(by_k[k][0] * by_k[k][-1]) for k in ks]
        per += [u_ / v_ for u_, v_ in zip(gm, gm[1:])]
        svg.add(f'<path d="{pa.path_d([(xg(k), yg(v)) for k, v in zip(ks, gm)])}" fill="none" stroke="{SERIES[i]}" stroke-width="1.6"/>')
        for k in ks:
            for v in by_k[k]:
                marker(svg, "circle", xg(k), yg(v), SERIES[i], f"D {D:g}, harmonic {k}: σ {v:.3e}")
        lab = lambda d: f"{d:.1f}".rstrip("0").rstrip(".")  # noqa: E731
        lx = x + 22 + sum(len(lab(d)) * 6.0 + 10 for d in gD[:i])       # one row under the axis title
        svg.text(lx, y + h + 52, lab(D), size=10, fill=SERIES[i])
    D3 = gD[0]
    h13 = [r for r in grip if float(r["D"]) == D3 and int(r["harmonic"]) == 13]
    move13 = max(abs(float(r["newton_step"])) for r in h13)
    size13 = math.hypot(*fit.values(D3, [13])[0])
    svg.text(x + 8, y + h + 52, "D", size=10, fill=INK_SOFT)
    for j, line in enumerate([f"each harmonic {min(per):.0f} to {max(per):.0f}× weaker than the last;",
                              f"at D {D3:.1f}, Newton would move harmonic 13",
                              f"by {move13:.2f}, {pa.pow10(round(math.log10(move13 / size13)))} times its size ({sci(size13)})"]):
        svg.text(x + 8, y + 12 + 13 * j, line, size=10, fill=INK)
    print(f"3. weaker per harmonic {min(per):.0f}x to {max(per):.0f}x; D {D3:g}: harmonic 13's full Newton move {move13:.3f}, "
          f"its size on the fit {size13:.2e}")

    # the step between the rows
    yc = rows_y[1][0] - 26
    pa.arrow(svg, W / 2 - 330, yc - 4, W / 2 - 330, yc + 12, SERIES[1], 2)
    svg.text(W / 2 - 316, yc + 8, "So the chain solves what the equations fix, and takes the held harmonics from the smooth family of the "
             "solutions that were solved in full.", size=12, fill=INK)

    # 4. the smooth family: sqrt(D-2) a5 against 1/(D-2)
    _, y, x, w, h = cell(0, 1)
    claim(0, 1, "4.", "But the solutions form a smooth family,", "so a fit carries harmonic 5 past D 73")
    u = lambda D: 1 / (D - 2)  # noqa: E731
    Dg = np.linspace(float(fitd["Dmin"]), 240, 300)
    curve = [(D, sq_a5(D, fit.values(float(D), [5])[0][0])) for D in Dg]
    vals = [sq_a5(float(p["D"]), p["c"]["5a"]) for p in pts] + [v for _, v in curve]
    pad = (max(vals) - min(vals)) * 0.12
    xl = (u(36.0), u(242.0))
    TICKS[xl] = [u(D) for D in (40, 50, 73, 100, 240)]
    xf, yf = axes(svg, x, y, w, h, xl, (min(vals) - pad, max(vals) + pad), "D, spaced as 1/(D − 2)", "√(D−2) · a5",
                  xfmt=lambda v: f"{1 / v + 2:.0f}", yfmt=lambda v: f"{v:.4f}")
    inside = [(xf(u(D)), yf(v)) for D, v in curve if D <= 73.0]
    outside = [(xf(u(D)), yf(v)) for D, v in curve if D >= 73.0]
    svg.add(f'<path d="{pa.path_d(outside)}" fill="none" stroke="{SERIES[1]}" stroke-width="2" stroke-dasharray="5 3"/>')
    for p in pts:
        D = float(p["D"])
        svg.add(f'<circle cx="{xf(u(D)):.1f}" cy="{yf(sq_a5(D, p["c"]["5a"])):.1f}" r="2.6" fill="{SERIES[2]}">'
                f'<title>D {D:.2f}: √(D−2) a5 {sq_a5(D, p["c"]["5a"]):.8f}</title></circle>')
    svg.add(f'<path d="{pa.path_d(inside)}" fill="none" stroke="{SERIES[1]}" stroke-width="1.4"/>')
    vF = sq_a5(D_front, fit.values(D_front, [5])[0][0])
    marker(svg, "diamond", xf(u(D_front)), yf(vF), SERIES[0], f"D {D_front:g}: held √(D−2) a5 {vF:.8f}")
    svg.text(xf(u(D_front)) - 8, yf(vF) + 4, f"held at D {D_front:.1f}", size=10, fill=SERIES[0], anchor="end")
    svg.text(x + w - 6, y + 14, f"dots: {len(pts)} full solutions", size=10, fill=SERIES[2], anchor="end")
    svg.text(x + w - 6, y + 28, f"line: the fit, within {sci(fit_miss, 0)} of each,", size=10, fill=SERIES[1], anchor="end")
    svg.text(x + w - 6, y + 42, "dashed past D 73", size=10, fill=SERIES[1], anchor="end")

    # 5. the chain's step: the Taylor terms and the corrector
    _, y, x, w, h = cell(1, 1)
    claim(1, 1, "5.", "The chain holds 5–13 on the fit, solves the rest,", "and steps in D by a Taylor series")
    xt, yt = axes(svg, x, y, w, h, (0, 25), (1e-32, 1e-2, "log"), f"order k of one step, D {BASE_D:.0f} → {acc['D']:.0f}",
                  "size of term k", xfmt=lambda v: f"{v:g}", yfmt=p10)
    bw = 0.62 * (xt(1) - xt(0))
    for k, v in enumerate(terms, start=1):
        top = yt(max(v, 1e-32))
        svg.add(f'<rect x="{xt(k) - bw / 2:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{y + h - top:.1f}" '
                f'fill="{SERIES[0] if k == 1 else SERIES[3]}"><title>order {k}: {v:.2e}</title></rect>')
    svg.text(x + w - 4, y + 14, f"{len(terms)} terms, summed: the prediction", size=10, fill=INK, anchor="end")
    svg.text(x + w - 4, y + 28, f"Newton corrects it by {sci(acc['moved'], 0)} in {acc['steps']} steps", size=10, fill=INK, anchor="end")

    # 6. the price: Delta's error bar
    _, y, x, w, h = cell(2, 1)
    emax = max(e for _, e in err_chain)
    claim(2, 1, "6.", "The price: Δ's error bar grows", f"from about 10⁻¹⁵ to {sci(emax)}")
    allerr = [e for _, e in err_full] + [e for _, e in err_chain]
    xe, ye = axes(svg, x, y, w, h, (0, 240), (10 ** math.floor(math.log10(min(allerr))), 10 ** math.ceil(math.log10(emax)), "log"),
                  "spacetime dimension D", "error bar of Δ", xfmt=lambda v: f"{v:g}", yfmt=p10)
    for (series, col, what) in ((err_full, SERIES[2], "full system: change with resolution"),
                                (err_chain, SERIES[0], "chain: spread over the fit's degree")):
        dots = "".join(f'<circle cx="{xe(D):.1f}" cy="{ye(e):.1f}" r="1.8" fill="{col}"/>' for D, e in series)
        svg.add(f"<g><title>{what}</title>{dots}</g>")
    for j, (line, col) in enumerate((("full system:", SERIES[2]), ("change with resolution", SERIES[2]),
                                     ("chain: spread over", SERIES[0]), ("the fit's degree", SERIES[0]))):
        svg.text(x + 8, y + 14 + 13 * j + (6 if j > 1 else 0), line, size=10, fill=col)

    hgt = footnote(svg, ["Δ and error bars: data/delta_all.csv", "grip: data/held_grip.csv (held_schur.py, raw I479-mac)",
                         f"wave, step: raw {FRONT.split('/anm')[0]}", "fit: data/held_fit.json"], "", [], rows_y[1][0] + 60 + PH + 46, REPO)
    out = os.path.join(a.o, "big_picture.svg")
    open(out, "w").write(svg.finish(hgt))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

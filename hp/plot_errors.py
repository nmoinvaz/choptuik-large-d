#!/usr/bin/env python3
"""The error graphs: lowD_errors.svg (how much Delta moves when the tau resolution M goes up one step, below D = 4)
and largeD_errors.svg (the continuation points' error indicators: distance from the 1/(D - 1) fit, Newton's remaining
linearised correction, and the frozen-harmonic cut). Standard library only; the primitives are data_graphs.py,
vendored unchanged from github.com/nmoinvaz/speedy-gonzales (skills/graph-data).

Data: data/delta_dd.csv (export_dd.py: Delta_err of the dM rows, dDelta_est, run, F) and data/fit_residuals.csv
(fit_residuals.py). The frozen-harmonic cut is the checks table's (README; notes/NUMERICS_REVIEW.md F5). Prints the
plotted numbers.

    cd hp && python3 export_dd.py && python3 fit_residuals.py && python3 plot_errors.py

The y axes are linear in log10 of the value, labelled 1e<k>: a log axis would put 1-2-5 gridlines on every one of
the 12 to 20 decades these errors span.
"""
import csv
import math
import os
import subprocess

from data_graphs import (INK, INK_SOFT, SERIES, Svg, axes, better_hint, callout, caption, dodge_markers, esc,
                         footnote, legend, marker)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "https://github.com/nmoinvaz/choptuik-large-d"
WIDTH = 1080
X, W = 80, 952  # plot box: room for the y title on the left, the better arrow on the right
FREEZE_D = 35.353229  # the runs pin the centre data's harmonics above 13 from this point on
# Freeing harmonic 15 as well, with full Newton steps (README checks table; notes/NUMERICS_REVIEW.md F5)
CUT = [(40.020065, 1.52e-15), (50.007491, -1.06e-15), (60.001952, -1.20e-15)]
MACHINES = "runs on the local i7-1185G7 and in the cloud"


def sci(v, digits=1):
    m, e = f"{v:.{digits}e}".split("e")
    return f"{m}e{int(e)}"


def dec(v):
    return f"1e{int(round(v))}"


def data_commit():
    try:
        return subprocess.check_output(["git", "log", "-1", "--format=%h", "--", "data/delta_dd.csv"], cwd=HERE,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def low_d(rows, commit):
    low = [dict(D=float(r["D"]), run=r["run"], err=float(r["Delta_err"])) for r in rows
           if r["err_kind"] == "dM" and r["Delta_err"]]
    svg = Svg(WIDTH)
    svg.text(16, 28, "Low D: change in Δ from the next lower τ resolution", size=15, fill=INK, weight="bold")
    y = caption(svg, X, 58, "|change in Δ| when M goes up one step",
                "At each D: Δ at the highest M run minus Δ at the next lower M (M = number of τ modes). "
                "Hover a point for its value.")
    h = 360
    logs = [math.log10(r["err"]) for r in low]
    ylo, yhi = 5 * math.floor(min(logs) / 5), 5 * math.ceil(max(logs) / 5)
    xmap, ymap = axes(svg, X, y, W, h, (3.0, 4.0), (ylo, yhi), "dimension D", "|change in Δ| (log scale)",
                      xfmt=lambda v: f"{v:.1f}", yfmt=dec)
    better_hint(svg, X, y, W, h, "right", reverse=True)
    pts = dodge_markers({0: {r["D"]: (xmap(r["D"]), ymap(math.log10(r["err"]))) for r in low}})[0]
    print("low D: |change in Delta| from the next lower M")
    print(f"{'D':>8} {'M':>5} {'change':>9}  status")
    flagged = []
    for r in low:
        r["M"] = r["run"].split("-M")[1].split("-")[0]
        r["bad"] = "unresolved" in r["run"]
        # Resolution status is a second variable, so it takes a shape, not a color
        marker(svg, "triangle" if r["bad"] else "circle", *pts[r["D"]], SERIES[0],
               f"D = {r['D']:g}, M = {r['M']}: Δ changed by {sci(r['err'])}"
               + ("; not resolved, row flagged" if r["bad"] else ""))
        print(f"{r['D']:8g} {r['M']:>5} {r['err']:9.1e}  {'not resolved, flagged' if r['bad'] else 'resolved'}")
        if r["bad"]:
            flagged.append(r)
            callout(svg, *pts[r["D"]], f"D = {r['D']:g}: not resolved at M = {r['M']}", dx=26, dy=-14)
    for r in sorted(low, key=lambda r: r["err"])[:2]:
        callout(svg, *pts[r["D"]], f"D = {r['D']:g}, M = {r['M']}: {sci(r['err'], 0)}", dx=-26, dy=-26)
    warnings = [f"D = {r['D']:g}: not resolved at M = {r['M']}, its row is flagged as not usable" for r in flagged]
    height = footnote(svg, [f"Δ: hp/data/delta_dd.csv (Delta_err of the dM rows) at choptuik-large-d {commit}"],
                      MACHINES, warnings, y + h + 46, repo_url=REPO)
    return svg.finish(height)


def least_squares(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return slope, my - slope * mx


def large_d(rows, fits, commit):
    newton = [(float(r["D"]), float(r["dDelta_est"])) for r in rows
              if not r["run"].startswith("lowD") and r["dDelta_est"] and float(r["dDelta_est"])]
    fitted = [(f["D"], f["v"]) for f in fits if f["fitted"] and f["v"]]
    beyond = [(f["D"], f["v"]) for f in fits if not f["fitted"] and f["D"] > max(d for d, _ in fitted) and f["v"]]
    dropped = sum(1 for f in fits if not f["fitted"] and f["D"] <= max(d for d, _ in fitted))
    tr = [(d, math.log(abs(v))) for d, v in newton if d >= FREEZE_D]
    slope, icpt = least_squares([d for d, _ in tr], [v for _, v in tr])
    factor = math.exp(slope)

    svg = Svg(WIDTH)
    svg.text(16, 28, "Large D: error indicators of the continuation points", size=15, fill=INK, weight="bold")
    cfit, cnewt, ccut = SERIES[0], SERIES[1], SERIES[2]
    legend(svg, WIDTH - 16, 58, [("distance from the 1/(D−1) fit", cfit),
                                 ("Newton's remaining correction", cnewt),
                                 ("frozen-harmonic cut", ccut)])
    dfit = max(d for d, _ in fitted)
    sub = (f"Fit: degree 18 in 1/(D−1) on the cleaned points 10 ≤ D ≤ {dfit:.2f}"
           + (" (hollow: points past it, extrapolated)" if beyond else "")
           + ". Newton: the linearised Δ step left at the end of each solve. Hover a point for its value.")
    y = caption(svg, X, 58, "size of each error indicator", sub)
    h = 420
    alld = [d for d, _ in fitted + beyond + newton]
    logs = [math.log10(abs(v)) for _, v in fitted + beyond + newton] + [math.log10(abs(v)) for _, v in CUT]
    xmap, ymap = axes(svg, X, y, W, h, (math.floor(min(alld) * 10) / 10 - 0.1, math.ceil(max(alld) * 10) / 10),
                      (5 * math.floor(min(logs) / 5), math.ceil(max(logs)) + 1), "dimension D",
                      "|error in Δ| (log scale)", xfmt=lambda v: f"{v:g}", yfmt=dec)
    better_hint(svg, X, y, W, h, "right", reverse=True)

    def at(D, v):
        return xmap(D), ymap(math.log10(abs(v)))

    # Hundreds of points per series: small dots, each with its hover title, instead of 11px ringed markers
    for series, color, what, hollow in ((fitted, cfit, "Δ − fit", False), (beyond, cfit, "Δ − fit", True),
                                        (newton, cnewt, "Newton's remaining correction", False)):
        paint = f'fill="none" stroke="{color}" stroke-width="1"' if hollow else f'fill="{color}"'
        for d, v in series:
            px, py = at(d, v)
            tip = esc(f"D = {d:.4f}: {what} {sci(v)}")
            svg.add(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.2" {paint}><title>{tip}</title></circle>')

    def trend(D):
        return math.exp(icpt + slope * D)
    d0, d1 = tr[0][0], tr[-1][0]
    (x0, y0), (x1, y1) = at(d0, trend(d0)), at(d1, trend(d1))
    svg.add(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="{INK_SOFT}" stroke-width="2" '
            f'stroke-opacity="0.7" stroke-dasharray="5 3"><title>least-squares trend of log |Newton correction| '
            f'on D ≥ {FREEZE_D:.2f}: ×{factor:.3f} per unit D</title></line>')
    for d, v in CUT:
        marker(svg, "circle", *at(d, v), ccut, f"D = {d:.4f}: freeing harmonic 15 moves Δ by {sci(v, 2)}")
    first_frozen = min((p for p in fitted if p[0] > FREEZE_D), key=lambda p: p[0])
    callout(svg, *at(*first_frozen), f"harmonics above 13 frozen from D = {FREEZE_D:.2f}", dx=24, dy=-30)
    callout(svg, *at(*CUT[0]), f"freeing harmonic 15 moves Δ by {sci(CUT[0][1], 2)}", dx=24, dy=-14)
    callout(svg, *at(45.0, trend(45.0)), f"Newton's correction grows ×{factor:.2f} per unit D", dx=24, dy=70)

    up6 = [r for r in rows if r["run"] == "up6"]
    warnings = [f"the fit's cleaning drops {dropped} points, not shown"]
    if up6:
        ds = [float(r["D"]) for r in up6]
        warnings.append(f"D {min(ds):.1f}–{max(ds):.1f} (run up6) were accepted at |F| up to "
                        f"{sci(max(float(r['F']) for r in up6))}")
    height = footnote(svg, [f"Δ and Newton corrections: hp/data/delta_dd.csv at choptuik-large-d {commit}",
                            "fit: fit_residuals.py", "cut: notes/NUMERICS_REVIEW.md F5"],
                      MACHINES, warnings, y + h + 46, repo_url=REPO)

    print("\nlarge D: Delta minus the degree-18 1/(D-1) fit (fitted points, then the ones past the fit) and Newton's "
          "remaining correction")
    nw = dict(newton)
    print(f"{'D':>10} {'Delta - fit':>12} {'Newton':>10}")
    for d, v in fitted + beyond:
        n = nw.get(d)
        print(f"{d:10.4f} {v:12.2e} {'' if n is None else format(n, '.2e'):>10}")
    print(f"trend of |Newton| on D >= {FREEZE_D:.2f} ({len(tr)} points): x{factor:.3f} per unit D")
    print("frozen-harmonic cut: " + ", ".join(f"D {d:.4f} {v:+.2e}" for d, v in CUT))
    return svg.finish(height)


def main():
    rows = list(csv.DictReader(open(os.path.join(HERE, "data", "delta_dd.csv"))))
    fits = [dict(D=float(f["D"]), v=float(f["Delta_minus_fit"]), fitted=f["fitted"] == "1")
            for f in csv.DictReader(open(os.path.join(HERE, "data", "fit_residuals.csv")))]
    commit = data_commit()
    for name, svg in (("lowD_errors.svg", low_d(rows, commit)), ("largeD_errors.svg", large_d(rows, fits, commit))):
        with open(os.path.join(HERE, name), "w") as fh:
            fh.write(svg)


if __name__ == "__main__":
    main()

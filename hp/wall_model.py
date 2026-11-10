#!/usr/bin/env python3
"""The large-D walls as the horizon series' off-solution radius against each layout's matching distance.

T84 measured, at the saved 0.7-layout states (D 40 to 73.27), the radius R of the horizon series in u = 1 - x for data
on the solution and for data about 4e-5 dD off it (dD = 1e-8, 1e-6, 1e-4, 1e-2; notes/QUAD_DOUBLE.md 12.4,
T84-mac/horizon_radius.jsonl). The series is summed at the matching distance u0 = 1 - x_last (0.30, 0.20 and 0.15 for
the last node at 0.7, 0.8 and 0.85), so a layout's continuation stalls where R for the deviations its iterates have
falls below u0. This script fits R = A D^p per deviation (least squares in log D, log R; a linear fit R = c0 + c1 D
beside it as the pessimistic reading), predicts the crossing D* = (u0 / A)^(1/p) per layout and deviation, reads the
observed walls from the continuation records (T73/T75's frontier in T84-mac/basin.jsonl, T85's x80 and x85 attempts,
T101's ladders, the plain double-double run in data/delta_dd.csv) and the Newton-basin records (T84-mac/basin.jsonl,
T101-mac/basin_x80.jsonl and basin_x85.jsonl), prints them, writes data/large_d_walls.json (the numbers the notes
quote) and large_d_walls.svg.

    cd hp && python3 wall_model.py ../../choptuik-large-d-raw [--svg large_d_walls.svg] [--json data/large_d_walls.json]
"""
import csv
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data_graphs import GRID, INK, INK_SOFT, SERIES, Svg, axes, better_hint, callout, caption, footnote, host_specs, legend, marker  # noqa: E402

KEYS = ["0e+00", "1e-08", "1e-06", "1e-04", "1e-02"]
DEV = {"0e+00": "on the solution", "1e-08": "4×10⁻¹³ off", "1e-06": "4×10⁻¹¹ off", "1e-04": "4×10⁻⁹ off", "1e-02": "4×10⁻⁷ off"}
LAYOUTS = [(0.30, "0.7"), (0.20, "0.8"), (0.15, "0.85")]
REPO = "https://github.com/nmoinvaz/choptuik-large-d"


def rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def fit_power(Ds, Rs):
    """R = A D^p by least squares in (ln D, ln R)"""
    xs, ys = [math.log(d) for d in Ds], [math.log(r) for r in Rs]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    p = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return math.exp(my - p * mx), p


def fit_linear(Ds, Rs):
    n = len(Ds)
    mx, my = sum(Ds) / n, sum(Rs) / n
    c1 = sum((x - mx) * (y - my) for x, y in zip(Ds, Rs)) / sum((x - mx) ** 2 for x in Ds)
    return my - c1 * mx, c1


def main():
    args = list(sys.argv[1:])
    svg_path, json_path = os.path.join(HERE, "large_d_walls.svg"), os.path.join(HERE, "data", "large_d_walls.json")
    if "--svg" in args:
        i = args.index("--svg"); svg_path = args[i + 1]; del args[i:i + 2]
    if "--json" in args:
        i = args.index("--json"); json_path = args[i + 1]; del args[i:i + 2]
    raw = args[0]
    rad = rows(os.path.join(raw, "T84-mac", "horizon_radius.jsonl"))
    Ds = [r["D"] for r in rad]
    fits = {}
    print(f"{'deviation':>16} {'A':>8} {'p':>7} {'R(73.3)':>8} {'R(100)':>7} {'R(200)':>7} {'linear zero':>12}")
    for k in KEYS:
        Rs = [r["R"][k] for r in rad]
        A, p = fit_power(Ds, Rs)
        c0, c1 = fit_linear(Ds, Rs)
        fits[k] = dict(A=A, p=p, c0=c0, c1=c1, D_zero_linear=-c0 / c1, R100=A * 100 ** p, R200=A * 200 ** p)
        print(f"{DEV[k]:>16} {A:8.3f} {p:7.3f} {A * 73.27 ** p:8.3f} {fits[k]['R100']:7.3f} {fits[k]['R200']:7.3f} {-c0 / c1:12.1f}")
    cross = {}
    print(f"\n{'last node':>10} {'u0':>5} " + " ".join(f"{DEV[k]:>14}" for k in KEYS[1:]) + "   (D where the power-law R crosses u0)")
    for u0, node in LAYOUTS:
        cross[node] = {k: (u0 / fits[k]["A"]) ** (1 / fits[k]["p"]) for k in KEYS[1:]}
        print(f"{node:>10} {u0:5.2f} " + " ".join(f"{cross[node][k]:14.1f}" for k in KEYS[1:]))
    # observed walls
    walls = {}
    with open(os.path.join(HERE, "data", "delta_dd.csv")) as fh:
        walls["0.7 plain"] = max(float(r["D"]) for r in csv.DictReader(fh) if r["run"] == "T48-up70")
    walls["0.7 pinned"] = max(r["D"] for r in rows(os.path.join(raw, "T84-mac", "basin.jsonl")))
    walls["0.8 pinned"] = max(r["D_to"] for r in rows(os.path.join(raw, "T85-mac", "x80_attempts.jsonl")) if r.get("ok"))
    walls["0.85 pinned"] = max(r["D_to"] for r in rows(os.path.join(raw, "T85-mac", "x85_attempts.jsonl")) if r.get("ok"))
    walls["0.85 ladder"] = max(r["D_to"] for r in rows(os.path.join(raw, "T101-mac", "ladder_x85_attempts.jsonl")) if r.get("ok"))
    print("\nobserved walls (last converged D):", ", ".join(f"{k} {v:.2f}" for k, v in walls.items()))
    # basins, full system
    basins = {}
    for name, f in (("0.7", "T84-mac/basin.jsonl"), ("0.8", "T101-mac/basin_x80.jsonl"), ("0.85", "T101-mac/basin_x85.jsonl")):
        p = os.path.join(raw, f)
        if os.path.exists(p):
            basins[name] = [(r["D"], r["pivot_min"], r["c"], r["basin"]) for r in rows(p) if r["system"] == "full"]
    print("\nfull-system Newton basin sigma^2/c (smallest pivot sigma) per layout:")
    for name, bs in basins.items():
        print(f"  last node {name}: " + "; ".join(f"D {d:.2f}: sigma {s:.1e}, basin {b:.1e}" for d, s, c, b in bs))
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    json.dump(dict(fits=fits, cross=cross, walls=walls, basins=basins, D_fit=[min(Ds), max(Ds)]), open(json_path, "w"), indent=1)

    # the graph
    svg = Svg(1080)
    X, W, H = 92, 900, 420
    y = caption(svg, X, 58, "Where the large-D continuations stop: the horizon series' radius against the matching distance",
                "R for data a given distance off the solution (T84, points), fitted R = A·D^p (lines, dashed past the data); a layout stalls where R falls below its u₀")
    xm, ym = axes(svg, X, y, W, H, (40, 200), (0, 0.8), "dimension D", "radius R of the horizon series in u = 1 − x")
    cols = {"0e+00": SERIES[-1], "1e-08": SERIES[0], "1e-06": SERIES[1], "1e-04": SERIES[2], "1e-02": SERIES[3]}
    Dmax = max(Ds)
    for k in KEYS:
        A, p = fits[k]["A"], fits[k]["p"]
        pts_in = [(xm(d), ym(A * d ** p)) for d in range(40, int(Dmax) + 1)] + [(xm(Dmax), ym(A * Dmax ** p))]
        pts_out = [(xm(d), ym(A * d ** p)) for d in range(int(Dmax), 201)]
        svg.add('<polyline fill="none" stroke="%s" stroke-width="2" points="%s"/>' % (cols[k], " ".join(f"{x:.1f},{yy:.1f}" for x, yy in pts_in)))
        svg.add('<polyline fill="none" stroke="%s" stroke-width="2" stroke-dasharray="6 5" points="%s"/>' % (cols[k], " ".join(f"{x:.1f},{yy:.1f}" for x, yy in pts_out)))
        for r in rad:
            marker(svg, "circle", xm(r["D"]), ym(r["R"][k]), cols[k], f"D {r['D']:.2f}: R = {r['R'][k]:.3f} ({DEV[k]})")
    for u0, node in LAYOUTS:
        svg.add(f'<line x1="{xm(40):.1f}" y1="{ym(u0):.1f}" x2="{xm(200):.1f}" y2="{ym(u0):.1f}" stroke="{INK_SOFT}" stroke-width="1" stroke-dasharray="2 4"/>')
        svg.text(xm(200) - 4, ym(u0) - 5, f"u₀ = {u0:.2f}, last node {node}", size=10, anchor="end")
    wall_marks = [("0.7 plain", 0.30, "square", "plain double-double, T48-up70: its floor"), ("0.7 pinned", 0.30, "diamond", "pinned double-double, T73/T75: floor and basin"),
                  ("0.8 pinned", 0.20, "diamond", "pinned, last node 0.8, T85"), ("0.85 pinned", 0.15, "diamond", "pinned, last node 0.85, T85 (left the branch)"),
                  ("0.85 ladder", 0.15, "triangle", "pinned, last node 0.85, T101 ladder (off the branch)")]
    for key, u0, shape, title in wall_marks:
        marker(svg, shape, xm(walls[key]), ym(u0), INK, f"stopped at D {walls[key]:.2f}: {title}")
    callout(svg, xm(walls["0.8 pinned"]), ym(0.20), f"0.8 layout stops at D {walls['0.8 pinned']:.1f}", dx=40, dy=52)
    callout(svg, xm(walls["0.7 pinned"]), ym(0.30), f"0.7 layouts stop at D {walls['0.7 plain']:.1f} and {walls['0.7 pinned']:.1f}", dx=-30, dy=-70)
    k4 = "1e-04"
    callout(svg, xm(cross["0.8"][k4]), ym(0.20), f"4×10⁻⁹ off crosses u₀ = 0.20 at D {cross['0.8'][k4]:.0f}", dx=60, dy=-60)
    better_hint(svg, X, y, W, H, where="right", label="converges")
    legend(svg, X + W, y - 10, [(DEV[k], cols[k]) for k in KEYS[1:]] + [(DEV["0e+00"], cols["0e+00"])])
    srcs = ["T84-mac/horizon_radius.jsonl f18a936 (horizon_radius.py, the 0.7-layout states, ks 100-130)",
            "walls: T84-mac/basin.jsonl, T85-mac/*_attempts.jsonl 3b6572f, T101-mac 7c06884, hp/data/delta_dd.csv"]
    warns = ["fits of 9 points at D 40-73 extrapolated to 200", "the deviation scale 4e-5 per unit of dD is 12.2's estimate"]
    height = footnote(svg, srcs, host_specs(), warns, y + H + 50, REPO)
    with open(svg_path, "w") as fh:
        fh.write(svg.finish(height))
    print(f"\nwrote {svg_path} and {json_path}")


if __name__ == "__main__":
    main()

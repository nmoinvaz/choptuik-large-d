#!/usr/bin/env python3
"""The echo itself: the scalar field at the centre over one period of the discretely self-similar solution, p(tau, x)
~ p1(tau) x near x = 0, at several D, each curve scaled by its largest value so the shapes compare (README, the large-D
graphs show the period Delta(D); this shows what repeats with that period). p1 is the first M entries of a stored
state (the half grid theta_j = pi j / M of the odd sector; the second half of the period is the first with the sign
flipped). States from the raw-data repository (hp/raw_out.py's clone).

    python3 plot_echo.py RAWDIR [-o echo_shapes.svg]

It also writes hp/data/echo_shapes.csv, the table it prints (the README quotes it; check_numbers.py reads it).
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data_graphs import GRID, SERIES, Svg, axes, callout, caption, footnote, host_specs, legend  # noqa: E402

REPO = "https://github.com/nmoinvaz/choptuik-large-d"
# (file in the raw-data repository, label): converged double-double states across D
STATES = [
    ("T100-win2/states/lowDhm_x0.000157-0.975_p1_D3.0444_M384.npz", "D = 3.044"),
    ("T79-win/down_D4.0000.npz", "D = 4"),
    ("T79-win/m64_D8.0000.npz", "D = 8"),
    ("T78/layoutB_D16.7600_M32.npz", "D = 16.76"),
    ("T77/relayout_test_D45.0507_n7.npz", "D = 45.05"),
    ("T12-win/scratch/t12/up17_D60.9626.npz", "D = 60.96"),
    ("T75/pin7_D73.2657.npz", "D = 73.27"),
]


def load(path):
    d = np.load(path)
    m = json.loads(str(d["meta"]))
    zh, zl = d["zh"], d["zl"]
    M = int(m["M"])
    p1 = zh[:M] + zl[:M]
    delta = float(zh[-1] + zl[-1])
    return float(m["D"]), M, delta, p1


def full_period(p1):
    """samples over theta in [0, 2 pi): the half grid and its sign-flipped copy, rolled so that the largest value sits
    at a quarter of the period (the states' own tau origins differ by their phase conditions)"""
    full = np.concatenate([p1, -p1])
    return np.roll(full, len(full) // 4 - int(np.argmax(full)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("-o", default=os.path.join(HERE, "echo_shapes.svg"))
    a = ap.parse_args()
    rows = []
    for f, label in STATES:
        D, M, delta, p1 = load(os.path.join(a.raw, f))
        full = full_period(p1)
        # harmonic content of the odd field: the fraction of its power in the first harmonic
        c = np.fft.rfft(full) / len(full)
        power = np.abs(c[1:]) ** 2
        frac1 = float(power[0] / power.sum())
        imax = int(np.argmax(np.abs(full)))
        amp = float(np.abs(full).max())
        # the distance from a pure first harmonic with the same peak and phase (the shape the curves approach)
        n = len(full)
        sine = np.sin(2 * np.pi * np.arange(n) / n)
        dev = float(np.max(np.abs(full / amp - sine)))
        rows.append((D, M, delta, label, full, amp, frac1, imax / len(full), dev))
    svg = Svg(900)
    X, W, H = 70, 700, 380
    y = caption(svg, X, 40, "The echo itself: the centre field over one period, D = 3 to 73",
                "p(τ, x) ≈ p1(τ) x near the centre; each curve divided by its largest value and its peak put at τ/Δ = 1/4; Δ(D) is the large-D graphs' line")
    xm, ym = axes(svg, X, y, W, H, (0, 1), (-1.05, 1.05), "fraction of one echo, τ / Δ",
                  "p1(τ) / max |p1|", xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}")
    svg.line(xm(0), ym(0), xm(1), ym(0), GRID, width=1)
    # the pure first harmonic, peak 1 at tau / Delta = 1/4: the shape every curve approaches as D grows
    sine_pts = " ".join(f"{'M' if i == 0 else 'L'}{xm(i / 400):.1f},{ym(math.sin(2 * math.pi * i / 400)):.1f}" for i in range(401))
    svg.add(f'<path d="{sine_pts}" fill="none" stroke="{SERIES[-1]}" stroke-width="2.4" stroke-dasharray="8 5">'
            "<title>a pure first harmonic, sin(2π τ/Δ): the limit shape</title></path>")
    items = []
    for k, (D, M, delta, label, full, amp, frac1, xmax, dev) in enumerate(rows):
        color = SERIES[k]
        n = len(full)
        pts = [f"{xm(j / n):.1f},{ym(full[j] / amp):.1f}" for j in range(n)] + [f"{xm(1.0):.1f},{ym(full[0] / amp):.1f}"]
        svg.add(f'<path d="M{pts[0]} ' + " ".join("L" + p for p in pts[1:]) + f'" fill="none" stroke="{color}" stroke-width="1.8">'
                f"<title>{label}: Δ = {delta:.6f}, M = {M}, max |p1| = {amp:.4g}, first harmonic {100 * frac1:.1f}% of the power</title></path>")
        items.append((f"{label}, Δ = {delta:.4f}", color))
    items.append(("one harmonic, sin(2πτ/Δ): the limit shape", SERIES[-1]))
    first, last = rows[0], rows[-1]
    callout(svg, xm(0.25), ym(1.0), f"{first[3]}: a pulse, first harmonic {100 * first[6]:.0f}% of the power", dx=60, dy=30)
    callout(svg, xm(0.75), ym(-1.0), f"{last[3]}: {100 * last[6]:.0f}%, within {last[8]:.3f} of the sinusoid", dx=-90, dy=-40)
    legend(svg, X + W, y - 10, items[:4])
    legend(svg, X + W, y + H + 52, items[4:])
    srcs = ["states: " + ", ".join(f.split("/")[0] for f, _ in STATES) + " (the raw-data repository)"]
    height = footnote(svg, srcs, host_specs(), [], y + H + 80, REPO)
    open(a.o, "w").write(svg.finish(height))
    print("wrote", a.o)
    print("| D | M | Δ | max p1 | first harmonic (power) | max deviation from sin(2πτ/Δ) |")
    print("|---|---|---|---|---|---|")
    with open(os.path.join(HERE, "data", "echo_shapes.csv"), "w") as f:
        f.write("# hp/plot_echo.py: the centre field's echo at several D from stored states of the raw-data repository: the state's\n"
                "# Delta, the largest |p1|, the fraction of the odd field's power in its first harmonic, and the largest deviation\n"
                "# of p1 / max |p1| from sin(2 pi tau / Delta) with the peaks aligned\n"
                "D,M,Delta,max_p1,first_harmonic_power_fraction,max_deviation_from_sine,state\n")
        for (D, M, delta, label, full, amp, frac1, xmax, dev), (path, _) in zip(rows, STATES):
            print(f"| {D} | {M} | {delta:.6f} | {amp:.4g} | {100 * frac1:.1f}% | {dev:.3f} |")
            f.write(f"{D},{M},{delta!r},{amp!r},{frac1!r},{dev!r},{path}\n")


if __name__ == "__main__":
    main()

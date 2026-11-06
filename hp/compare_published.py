#!/usr/bin/env python3
"""Every published value of Delta (data/published_delta.csv) against ours, in data/delta_published_compare.csv.

    cd hp && python3 compare_published.py

At a published D that has a row of data/delta_all.csv, ours is that row (how "row", err the row's err). Elsewhere, ours
is interpolated in D through the nine nearest rows of one kind (the kind of the nearest row): Lagrange through all
nine, in mpmath, with err the difference from the interpolant through the nearest seven (an estimate of the
interpolation error, not a bound; how "interp-<kind>"). No value is extrapolated: a published D outside the span of
those rows, or with fewer than nine rows of that kind around it, gets none. Columns: D, published_ref, published_Delta,
published_unc (the uncertainty the file gives), ours_Delta, ours_how, ours_err, diff (ours minus published) and within
("yes" when |diff| <= published_unc). The chain rows (D > 75) carry the held harmonics' bias, about 1e-8 to 1e-7 at
D 79-86 and growing with D (notes/QUAD_DOUBLE.md 13.19), which ours_err does not include. Prints the largest |diff|
per reference.
"""
import csv
import os
from collections import defaultdict

import mpmath as mp

mp.mp.dps = 50
HERE = os.path.dirname(os.path.abspath(__file__))


def value(r):
    return sum((mp.mpf(float(r[f"Delta_q{i}"])) for i in range(4) if r[f"Delta_q{i}"] != ""), mp.mpf(0))


def lagrange(pts, x):
    s = mp.mpf(0)
    for i, (xi, yi) in enumerate(pts):
        w = mp.mpf(1)
        for j, (xj, _) in enumerate(pts):
            if j != i:
                w *= (x - xj) / (xi - xj)
        s += w * yi
    return s


def main():
    rows = list(csv.DictReader(open(os.path.join(HERE, "data", "delta_all.csv"))))
    by_d = {round(float(r["D"]), 6): r for r in rows}
    kinds = defaultdict(list)
    for r in rows:
        kinds[r["kind"]].append((mp.mpf(r["D"]), value(r)))
    pub = []
    for line in open(os.path.join(HERE, "data", "published_delta.csv")):
        if line.startswith("#") or line.startswith("D,"):
            continue
        d, v, u, ref = line.strip().split(",")
        pub.append((float(d), v, u, ref))
    out, worst = [], defaultdict(lambda: (0.0, None))
    for d, v, u, ref in sorted(pub):
        row = by_d.get(round(d, 6))
        ours = how = err = ""
        if row is not None:
            ours, how, err = value(row), "row", row["err"]
        else:
            near = min(rows, key=lambda r: abs(float(r["D"]) - d))
            pts = sorted(kinds[near["kind"]], key=lambda p: abs(p[0] - d))
            if len(pts) >= 9:
                p9, p7 = sorted(pts[:9]), sorted(pts[:7])
                if p9[0][0] <= d <= p9[-1][0]:
                    ours = lagrange(p9, mp.mpf(d))
                    how, err = f"interp-{near['kind']}", format(float(abs(ours - lagrange(p7, mp.mpf(d)))), ".1e")
        diff = within = ""
        if ours != "":
            dv = float(ours - mp.mpf(v))
            diff, within = format(dv, ".2e"), ("yes" if abs(dv) <= float(u) else "no")
            if abs(dv) > worst[ref][0]:
                worst[ref] = (abs(dv), d)
            ours = mp.nstr(ours, 20)
        out.append(dict(D=repr(d), published_ref=ref, published_Delta=v, published_unc=u, ours_Delta=ours, ours_how=how,
                        ours_err=err, diff=diff, within=within))
    path = os.path.join(HERE, "data", "delta_published_compare.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    n = defaultdict(lambda: [0, 0, 0])
    for o in out:
        c = n[o["published_ref"]]
        c[0] += 1
        c[1] += o["ours_Delta"] != ""
        c[2] += o["within"] == "yes"
    print(f"{len(out)} published values -> {path}")
    for ref, (tot, have, ok) in n.items():
        m, at = worst[ref]
        print(f"  {ref}: {tot} values, {have} compared, {ok} within the stated uncertainty; largest |diff| {m:.2e} at D {at}")


if __name__ == "__main__":
    main()

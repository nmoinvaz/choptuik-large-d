#!/usr/bin/env python3
"""Every published value of matter/published_values.csv against ours, in matter/published_compare.csv.

matter/published_values.csv holds the published values for the matter models of matter/: the kappa-model CSS
(matter/CSS.md; kappa = 0 the free complex scalar, kappa = 1 the axion-dilaton), with the uncertainty each publication
states and the file ours is read from. It is plain CSV with no comment lines, so that GitHub renders it. Its references
are the keys of matter/CSS.md's reference list, where each was verified; ref_url is the paper's arXiv page (the IDs of
matter/LITERATURE.md's reference list). unc_abs is the stated uncertainty, or half a unit of the last printed digit
where none is stated (unc_kind "rounding bound (none stated)"). ours_source is a path in this repository or raw:<path>
in the raw-data repository; ours_key names the field of the record with the row's D and kappa.

    python3 solver/matter/compare_published_matter.py RAW            write the table
    python3 solver/matter/compare_published_matter.py RAW --check [--mutate=ours]

RAW is a clone of the raw-data repository (git clone https://github.com/nmoinvaz/choptuik-large-d-raw), for the rows
whose ours_source is raw:<path>. Ours is the field ours_key of the record of ours_source with the row's D and kappa
(a file without a kappa field is one kappa's). Columns: the published side as given, then ours, ours_source, diff
(ours minus published) and within ("yes" when |diff| <= unc_abs). --check rebuilds the table from the sources and
compares it with the written file, field by field, exiting 1 on any difference; --mutate=ours adds one unit of the
tenth digit to the first row's ours before the comparison, which must then fail.
"""
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "matter", "published_values.csv")
OUT = os.path.join(ROOT, "matter", "published_compare.csv")
COLS = ["model", "D", "kappa", "quantity", "published_ref", "ref_url", "published", "published_unc", "unc_abs", "unc_kind",
        "convention", "locator", "ours", "ours_source", "diff", "within"]


def ours(raw, src, key, D, kappa):
    path = os.path.join(raw, src[4:]) if src.startswith("raw:") else os.path.join(ROOT, src)
    if not os.path.exists(path):
        sys.exit(f"compare_published_matter: {path} is missing (clone the raw-data repository and pass it as RAW)")
    hits = []
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if abs(float(r["D"]) - D) < 1e-9 and ("kappa" not in r or abs(float(r["kappa"]) - kappa) < 1e-9) and key in r:
            hits.append(float(r[key]))
    if not hits:
        sys.exit(f"compare_published_matter: no record with D {D}, kappa {kappa} and {key} in {src}")
    return hits[0]


def build(raw, mutate=""):
    rows = []
    for r in csv.DictReader(l for l in open(SRC) if not l.startswith("#")):
        v = ours(raw, r["ours_source"], r["ours_key"], float(r["D"]), float(r["kappa"]))
        if mutate == "ours" and not rows:
            v += 10.0 ** (len(f"{v:.10g}".split(".")[0]) - 10)
        d = v - float(r["published"])
        rows.append(dict(r, ours=repr(v), diff=f"{d:+.3e}", within="yes" if abs(d) <= float(r["unc_abs"]) else "no"))
    return rows


def main(argv):
    if not argv or argv[0].startswith("--"):
        sys.exit(__doc__)
    raw = argv[0]
    mutate = next((a.split("=", 1)[1] for a in argv if a.startswith("--mutate=")), "")
    rows = build(raw, mutate)
    if "--check" in argv:
        old = list(csv.DictReader(open(OUT)))
        bad = [f"{a['published_ref']} {a['quantity']}({a['D']}, kappa {a['kappa']}): {c} {a[c]} against {b[c]}"
               for a, b in zip(rows, old) for c in COLS if a[c] != b[c]]
        if len(rows) != len(old):
            bad.append(f"{len(rows)} rows against {len(old)}")
        print("\n".join(bad) if bad else f"check: {len(rows)} rows equal to {os.path.relpath(OUT, ROOT)}")
        return 1 if bad else 0
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    n_no = sum(r["within"] == "no" for r in rows)
    print(f"wrote {os.path.relpath(OUT, ROOT)}: {len(rows)} rows, {n_no} outside the published uncertainty")
    for r in rows:
        print(f"  {r['published_ref']:6s} {r['model']:19s} D {r['D']:>2s} kappa {r['kappa']:>2s} {r['quantity']:9s} "
              f"published {r['published']:>14s}  ours {r['ours']:>16s}  diff {r['diff']}  within {r['within']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

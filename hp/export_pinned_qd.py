#!/usr/bin/env python3
"""The quad-double Delta_full1 of the pinned solutions as exact sums, hp/data/delta_pinned_qd.csv, one row per record
of T84-mac/*_fullcorr.jsonl (qd_fullcorr.py, notes/QUAD_DOUBLE.md 12.4; #758).

    cd hp && python3 export_pinned_qd.py RAWDIR [--verify] [--mutate=repr]

RAWDIR is a clone of the raw-data repository (git clone https://github.com/nmoinvaz/choptuik-large-d-raw). --verify
rebuilds every row from RAWDIR and data/delta_qd.csv and exits 1 on any difference from the written file. --mutate=repr
is its control, which must fail: the first record's eight parts are summed as their shortest decimal forms (repr)
instead of their binary values.

Each record holds the pinned solution's Delta and the Delta component of one full-system Newton step from it, four
f64 parts each. Delta_full1 is the exact sum of the eight parts (mpmath, 1200 bits, export_delta_all.full1_qd).
Columns: D and pin; F_full, the full system's |F| at the pinned solution; dz, the full step's |dz|; correction, the
step's Delta component (the f64 nearest its exact sum); Delta_pinned, the pinned Delta's exact sum (DIGITS significant
digits); Delta_q0..Delta_q3, Delta_full1 as four f64 parts, each the nearest f64 to what the ones before leave of it;
Delta, Delta_full1 to DIGITS significant digits; minus_other_pin, this row's Delta_full1 minus the other pin's at this D
(a spread between two systems, not a bound); minus_qd, this row's Delta_full1 minus the converged full-system
quad-double Delta of data/delta_qd.csv at this D, the exact sum of its four parts (empty where none); dd_pin7 and
dd_pin5, T75's double-double Delta_full1 at this D (DD, Delta_full1 + Delta_full1_lo summed exactly; empty where that
pin has none); source, the record's file.
"""
import csv
import os
import sys

import mpmath as mp

import export_delta_all as E

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "delta_pinned_qd.csv")
QD = os.path.join(HERE, "data", "delta_qd.csv")
COLS = ["D", "pin", "F_full", "dz", "correction", "Delta_pinned", "Delta_q0", "Delta_q1", "Delta_q2", "Delta_q3",
        "Delta", "minus_other_pin", "minus_qd", "dd_pin7", "dd_pin5", "source"]
# T75's double-double runs, by pin
DD = {7: "T75/pin7.jsonl", 5: "T75/pin5.jsonl"}
DIGITS = 40
PREC = 1200


def dec(x):
    return mp.nstr(x, DIGITS, strip_zeros=False)


def exact(parts):
    """the exact sum of f64 parts"""
    with mp.workprec(PREC):
        return sum((mp.mpf(p) for p in parts), mp.mpf(0))


def build(mutate=""):
    with mp.workprec(PREC):
        qd = {round(float(r["D"]), 6): exact([float(r[f"Delta_q{i}"]) for i in range(4)])
              for r in csv.DictReader(open(QD))}
        dd = {pin: {round(r["D"], 6): exact([r["Delta_full1"], r["Delta_full1_lo"]])
                    for r in E.records(f) if "Delta_full1" in r} for pin, f in DD.items()}
        sols = E.pinned_qd()
        if mutate == "repr":
            k = min(sols)
            pin, _, f, r = sols[k][-1]
            v = sum((mp.mpf(repr(p)) for p in r["Delta_pinned_q"] + r["correction_q"]), mp.mpf(0))
            sols[k][-1] = (pin, v, f, r)
            print(f"mutated: D {r['D']} pin {pin}, its parts summed as their shortest decimal forms")
        rows = []
        for k in sorted(sols):
            for pin, v, f, r in sorted(sols[k], key=lambda t: -t[0]):
                other = [w for p, w, _, _ in sols[k] if p != pin]
                if len(other) != 1:
                    sys.exit(f"D {r['D']}: {len(sols[k])} records, expected two pins")
                rows.append(dict(D=E.fmt(r["D"]), pin=str(pin), F_full=E.fmt(r["F_full"]), dz=E.fmt(r["dz"]),
                                 correction=E.fmt(exact(r["correction_q"])), Delta_pinned=dec(exact(r["Delta_pinned_q"])),
                                 **dict(zip(("Delta_q0", "Delta_q1", "Delta_q2", "Delta_q3"), E.parts4(v))),
                                 Delta=dec(v), minus_other_pin=E.fmt(v - other[0]),
                                 minus_qd=E.fmt(v - qd[k]) if k in qd else "",
                                 dd_pin7=dec(dd[7][k]) if k in dd[7] else "", dd_pin5=dec(dd[5][k]) if k in dd[5] else "",
                                 source=f))
    return rows


def verify(mutate=""):
    have = list(csv.DictReader(open(OUT)))
    want = build(mutate)
    bad = 0
    for i in range(max(len(have), len(want))):
        h = have[i] if i < len(have) else None
        w = want[i] if i < len(want) else None
        if h != w:
            bad += 1
            cols = [c for c in COLS if (h or {}).get(c) != (w or {}).get(c)]
            print(f"DIFFERS: row {i + 1} (D {(h or w)['D']} pin {(h or w)['pin']}): {', '.join(cols)}")
            for c in cols[:3]:
                print(f"   {c}: file {(h or {}).get(c)!r}, rebuilt {(w or {}).get(c)!r}")
    print(f"verify: {len(want) - bad} of {len(want)} rows of {OUT} equal their rebuild")
    return bad == 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict((a[2:] + "=").split("=")[:2] for a in sys.argv[1:] if a.startswith("--"))
    if not args:
        sys.exit(__doc__)
    E.RAW = os.path.abspath(args[0])
    if opts.get("mutate", "repr") != "repr":
        sys.exit(f"unknown --mutate={opts['mutate']}")
    if "verify" in opts or "mutate" in opts:
        sys.exit(0 if verify(opts.get("mutate", "")) else 1)
    rows = build()
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {OUT}")

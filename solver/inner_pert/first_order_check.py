#!/usr/bin/env python3
"""Check that a first_order_run.py file's stored ell, Jen and Jel belong to its stored lam1 (#385).

    python3 first_order_check.py FILE.json [FILE.json ...] [--tol=1e-9] [--mutate=ell|jen|lam1]

first_order_run.py's direct route gives lam1 = -(ell . J_en) / (ell . J_el), with ell the left null vector of S(lam*).
A file passes when that quotient, formed from the stored vectors, equals the stored lam1 within --tol (relative), and
when the stored ellJel, if present, equals ell . J_el within --tol. Exit 0 when every file passes, 1 otherwise.

--mutate breaks the input before the check, which must then fail: ell (first component scaled by 1.01), jen (J_en's
largest component scaled by 1.01), lam1 (the stored lam1 scaled by 1 + 1e-6).
"""
import json
import math
import sys


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


def check(fn, tol, mutate):
    d = json.load(open(fn))
    missing = [k for k in ("ell", "Jen", "Jel", "lam1") if k not in d]
    if missing:
        print(f"{fn}: FAIL, missing {missing}")
        return False
    ell, Jen, Jel, lam1 = list(d["ell"]), list(d["Jen"]), list(d["Jel"]), d["lam1"]
    if not len(ell) == len(Jen) == len(Jel):
        print(f"{fn}: FAIL, lengths ell {len(ell)}, Jen {len(Jen)}, Jel {len(Jel)}")
        return False
    if mutate == "ell":
        ell[0] *= 1.01
    elif mutate == "jen":
        i = max(range(len(Jen)), key=lambda j: abs(Jen[j]))
        Jen[i] *= 1.01
    elif mutate == "lam1":
        lam1 *= 1 + 1e-6
    den = dot(ell, Jel)
    q = -dot(ell, Jen) / den
    rel = abs(q - lam1) / abs(lam1)
    ok = rel <= tol
    line = f"{fn}: lam1 stored {lam1:.15g}, -ell.Jen/ell.Jel {q:.15g}, relative difference {rel:.2e}"
    if "ellJel" in d:
        rel2 = abs(den - d["ellJel"]) / abs(d["ellJel"])
        ok = ok and rel2 <= tol
        line += f"; ell.Jel stored {d['ellJel']:.9g}, recomputed {den:.9g}, relative difference {rel2:.2e}"
    print(line + (": ok" if ok else ": FAIL"))
    return ok


def main():
    opts = {a.split("=")[0]: (a.split("=", 1)[1] if "=" in a else True) for a in sys.argv[1:] if a.startswith("--")}
    files = [a for a in sys.argv[1:] if not a.startswith("--")]
    unknown = set(opts) - {"--tol", "--mutate"}
    if not files or unknown or opts.get("--mutate") not in (None, "ell", "jen", "lam1"):
        sys.exit(__doc__)
    tol = float(opts.get("--tol", 1e-9))
    results = [check(fn, tol, opts.get("--mutate")) for fn in files]
    print(f"{sum(results)} of {len(results)} files pass (tol {tol:g})")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()

"""#375: the tables of matter/COMPLEX_IMAG_DD.md section 4 from a runner's raw folder (run_i375.sh's files).

    python3 i375_summary.py RAWDIR [--other=EVALS_A,EVALS_B]

RAWDIR is a folder of choptuik-large-d-raw (I375-dh370). Prints the tolerance scan (phase of det M at tol 1e-12
against 1e-14), the box counts with the planted pair and the mutated coupling, and the acceptance verdicts; exits 1
when one fails. --other compares two evaluation logs of the same contour (two machines) point by point.
"""
import glob
import json
import math
import os
import sys

EXPECT = {"B0": 1, "U0": 0, "FULL": 0}      # the crate: lambda = 0 only (GHOST_ADMIXTURE.md 3, COMPLEX_CRITICAL.md 3)
PHASE_TOL = 1e-3                            # #375's acceptance, rad
INT_TOL = 1e-6                              # a winding is an integer to this


def recs(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def wrap(x):
    return math.remainder(x, 2 * math.pi)


def main():
    raw = sys.argv[1]
    other = next((a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith("--other=")), None)
    ok = True
    print("tolerance scan: phase of det M, tol 1e-12 against 1e-14 (limit %.0e rad)" % PHASE_TOL)
    for f in sorted(glob.glob(os.path.join(raw, "tolscan_*.jsonl")), key=lambda f: float(f.split("_D")[-1][:-6])):
        by = {}
        for r in recs(f):
            by.setdefault(tuple(r["lam"]), {})[r["tol"]] = r
        dph = [abs(wrap(d[1e-12]["lndet"][1] - d[1e-14]["lndet"][1])) for d in by.values()]
        dln = [abs(d[1e-12]["lndet"][0] - d[1e-14]["lndet"][0]) for d in by.values()]
        r0 = next(iter(by.values()))[1e-14]
        good = len(by) == 6 and max(dph) <= PHASE_TOL
        ok &= good
        print("  D %.2f  %d lambda  max dphase %.1e  max dln|det| %.1e  %s" %
              (r0["D"], len(by), max(dph), max(dln), "OK" if good else "BAD"))
    print("counts (winding of g; expected B0 1, U0 0, FULL 0; plant +2 in FULL; coupling changes B0)")
    names = glob.glob(os.path.join(raw, "count_*_U0.jsonl"))
    states = sorted({os.path.basename(f)[6:].rsplit("_", 1)[0] for f in names}, key=lambda s: float(s.split("_D")[1]))
    for s in states:
        row, D = {}, None
        for tag in ("B0", "U0", "FULL", "FULL_plant", "B0_coupling"):
            r = recs(os.path.join(raw, "count_%s_%s.jsonl" % (s, tag)))[-1]
            D = r["D"]
            w = r["winding_g"]
            n = round(w)
            ok &= abs(w - n) < INT_TOL and r["refinement_limit_hits"] == 0 and r["largest_step"] <= math.pi / 4
            row[tag] = (n, abs(w - n), r["largest_step"], r["candidate_poles_inside"], r["evaluations"])
        good = all(row[b][0] == EXPECT[b] for b in EXPECT) and row["FULL_plant"][0] == row["FULL"][0] + 2 \
            and row["B0_coupling"][0] != row["B0"][0]
        ok &= good
        print("  D %.2f  B0 %d  U0 %d  FULL %d (poles %d)  plant %d  coupling B0 %d  |w - n| <= %.0e  "
              "largest step %.2f (mutated %.2f)  evaluations %d  %s" %
              (D, row["B0"][0], row["U0"][0], row["FULL"][0], row["FULL"][3], row["FULL_plant"][0],
               row["B0_coupling"][0], max(v[1] for v in row.values()),
               max(row[b][2] for b in ("B0", "U0", "FULL", "FULL_plant")), row["B0_coupling"][2],
               sum(row[b][4] for b in ("B0", "U0", "FULL")), "OK" if good else "BAD"))
    if other:
        fa, fb = other.split(",")
        ev = lambda f: {tuple(r["lam"]): complex(*r["lndet"]) for r in recs(f) if r.get("kind") == "eval"}
        a, b = ev(fa), ev(fb)
        com = sorted(set(a) & set(b))
        print("two machines, %d common lambda: max dln|det| %.1e, max dphase %.1e rad" %
              (len(com), max(abs(a[k].real - b[k].real) for k in com),
               max(abs(wrap(a[k].imag - b[k].imag)) for k in com)))
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

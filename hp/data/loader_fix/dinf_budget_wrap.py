"""Run hp/dinf_budget.py's main() unchanged (same stdout) and also dump every grid row at 20 digits to a JSON file.
The only addition is the dump, made before report() prints."""
import json
import os
import sys

WT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))  # repository root
OUT = sys.argv[1]
sys.path.insert(0, os.path.join(WT, "hp"))
sys.argv = [os.path.join(WT, "hp", "dinf_budget.py")]
import dinf_budget as B  # noqa: E402

mp = B.mp
orig = B.report


def report(rows, T0):
    json.dump([{"dmax": r[0], "dmin": r[1], "clean": r[2], "model": r[3], "deg": r[4], "limit": mp.nstr(r[5], 20),
                "res": mp.nstr(r[6], 6), "used": r[7]} for r in rows], open(OUT, "w"), indent=0)
    orig(rows, T0)


B.report = report
B.main()

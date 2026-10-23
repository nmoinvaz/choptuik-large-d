"""Run an hp/ script with dd_fits_mp.load_points replaced by the pre-fix parse (the CSV strings read as decimals at
60 digits), to give the old output of a documented command on today's file. Kept with the logs in hp/data/loader_fix/ (notes/LOADER_FIX.md).
    python3 oldrun.py hp/dd_fits_mp.py 10 45 --clean --dmax=60.1"""
import os
import runpy
import sys

WT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))  # repository root
sys.path.insert(0, os.path.join(WT, "hp"))
import dd_fits_mp as F  # noqa: E402

mp = F.mp


def old_load(scan=None):
    assert scan is None
    pts, src = {}, {}
    for line in list(open(os.path.join(F.HERE, "data", "delta_dd.csv")))[1:]:
        c = line.strip().split(",")
        if c[6].startswith("lowD"):
            continue
        k = round(float(c[0]), 6)
        pts[k] = mp.mpf(c[1]) + mp.mpf(c[2])
        src[k] = c[6]
    return pts, src


F.load_points = old_load
script = os.path.join(WT, sys.argv[1])
sys.argv = sys.argv[1:]
if os.path.basename(script) == "dd_fits_mp.py":
    F.main()
else:
    sys.path.insert(0, os.path.dirname(script))
    runpy.run_path(script, run_name="__main__")

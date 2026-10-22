"""Attribute the changes in the held-out table (dd_fits_mp.py 10 45 --clean --dmax=60.1) to (a) the values (exact
vs decimal-parsed) and (b) the cleaning set (D = 10.75 kept by the fixed loader, dropped by the old one).
Four variants: values {fixed, old} x set {fixed (731), old (730)}. Kept with the logs in hp/data/loader_fix/ (notes/LOADER_FIX.md)."""
import os
import sys
import time

WT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))  # repository root
sys.path.insert(0, os.path.join(WT, "hp"))
import dd_fits_mp as F  # noqa: E402

mp = F.mp
T0 = time.time()
cut = float(sys.argv[1]) if len(sys.argv) > 1 else 45.0
new, _ = F.load_points()
old = {}
for line in list(open(os.path.join(WT, "hp", "data", "delta_dd.csv")))[1:]:
    c = line.strip().split(",")
    old[round(float(c[0]), 6)] = mp.mpf(c[1]) + mp.mpf(c[2])
base = sorted(d for d in new if 10 <= d <= 60.1)
set_new = F.clean(new, base, verbose=False)
set_old = [d for d in set_new if d != 10.75]
assert len(set_new) == 731 and len(set_old) == 730, (len(set_new), len(set_old))
for vname, pts in (("fixed values", new), ("old values", old)):
    for sname, Ds in (("fixed set 731", set_new), ("old set 730", set_old)):
        fitD = [d for d in Ds if d <= cut]
        hold = [d for d in Ds if d > cut]
        for model, deg in (("1/D", 16), ("1/D", 18), ("1/n", 18), ("n^-1/2", 20), ("D^-1/2", 20)):
            ev, res, _ = F.fit(pts, fitD, model, deg)
            h = max(abs(ev(d) - pts[d]) for d in hold)
            print(f"cut {cut:g}  {vname:12s} {sname:13s} {model:6s} {deg}: res {mp.nstr(res, 3):>8s}  held-out {mp.nstr(h, 3):>8s}"
                  f"  limit {mp.nstr(F.limit(ev), 12)}  ({time.time() - T0:.0f} s)", flush=True)

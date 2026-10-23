"""Cleaning sets of dd_fits_mp.clean with the fixed loader and with the old decimal-string parse, for every
(Dmin, Dmax) the quoted commands use; plus point-level parse errors. Kept with the logs in hp/data/loader_fix/ (notes/LOADER_FIX.md)."""
import json
import os
import sys
import time

WT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))  # repository root
sys.path.insert(0, os.path.join(WT, "hp"))
import dd_fits_mp as F  # noqa: E402

mp = F.mp
T0 = time.time()
new, _ = F.load_points()
old = {}
for line in list(open(os.path.join(WT, "hp", "data", "delta_dd.csv")))[1:]:
    c = line.strip().split(",")
    if c[6].startswith("lowD"):
        continue
    old[round(float(c[0]), 6)] = mp.mpf(c[1]) + mp.mpf(c[2])
err = {d: old[d] - new[d] for d in new}
print(f"parse error: max {mp.nstr(max(abs(v) for v in err.values()), 3)}, rms "
      f"{mp.nstr(mp.sqrt(mp.fsum(v * v for v in err.values()) / len(err)), 3)}; D > 45: max "
      f"{mp.nstr(max(abs(v) for d, v in err.items() if d > 45), 3)}", flush=True)
out = {}
cases = [(10, 60.1), (10, 1e9), (10, 59.8), (10, 45), (12, 45), (15, 45), (12, 60.1), (15, 60.1), (12, 61.03),
         (15, 61.03)]
if len(sys.argv) > 1:
    cases = [tuple(float(x) for x in a.split(":")) for a in sys.argv[1:]]
for dmin, dmax in cases:
    base = sorted(d for d in new if dmin <= d <= dmax)
    kn = F.clean(new, base, verbose=False)
    ko = F.clean(old, base, verbose=False)
    dn, do = sorted(set(base) - set(kn)), sorted(set(base) - set(ko))
    print(f"D {dmin}..{dmax}: N {len(base)}; fixed drops {len(dn)} keeps {len(kn)}; old drops {len(do)} keeps {len(ko)}; "
          f"only old drops {sorted(set(do) - set(dn))}; only fixed drops {sorted(set(dn) - set(do))}  "
          f"({time.time() - T0:.0f} s)", flush=True)
    out[f"{dmin}:{dmax}"] = {"N": len(base), "fixed_drops": dn, "old_drops": do}
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cleaning_compare.json"), "w"), indent=1)

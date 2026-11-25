"""Track C's Delta(D) (hp/track_c.py: load_real, all_fits, levels, estimate; LARGE_D_WALL, notes/LIT_GAPS_LARGED.md
§7 on wip/lit-gaps-larged) at the D of the #1090 runs, written as JSON: D -> value, E (the spread over fit choices),
h, s. Conditional on Delta analytic in 1/n (Track C's premise).

Usage (from hp/): python3 ../solver/inner_large_d/trackc_at.py OUT.json D1,D2,...
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "hp"))
import track_c as T  # noqa: E402

out, Ds = sys.argv[1], [float(v) for v in sys.argv[2].split(",")]
real = T.load_real()
fits = T.all_fits(real)
D0, used = T.levels(fits)
res = {"points": len(real), "D0": D0, "fits_used": len(used), "values": {}}
for D in Ds:
    e = T.estimate(fits, D)
    res["values"]["%g" % D] = {"Delta": e["value"].mid().str(17, radius=False), "E": e["E"], "h": e["h"], "s": e["s"]}
    print("D %-8g Delta %s  E %.2e" % (D, res["values"]["%g" % D]["Delta"], e["E"]), flush=True)
json.dump(res, open(out, "w"), indent=1)

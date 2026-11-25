"""Tables of the #1090 continuations from their JSON records (solve_cont.py's sol_*.json, solve_dd_cont.py's
sdd_*.json), with Delta against Track C at the same D (trackc_at.py's JSON).

Usage: python3 tabulate.py TRACKC.json DIR [DIR ...] [--md]
"""
import glob
import json
import os
import sys

import mpmath as mp

mp.mp.dps = 30
args = [a for a in sys.argv[1:] if not a.startswith("--")]
md = "--md" in sys.argv
tc = json.load(open(args[0]))["values"]
for d in args[1:]:
    recs = []
    for fn in glob.glob(os.path.join(d, "sol_*.json")) + glob.glob(os.path.join(d, "sdd_*.json")):
        recs.append(json.load(open(fn)))
    recs.sort(key=lambda r: r["D"])
    print("# %s (%d records)" % (d, len(recs)))
    if md:
        print("| D | M | a | ks | max\\|F\\| | Δ | Δ − Track C | Track C's E | σ_min (scaled) | grip | noise, Moebius | noise, affine | s |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in recs:
        key = "%g" % r["D"]
        dl = mp.mpf(r["Delta"]) + mp.mpf(r.get("Delta_lo", 0.0))
        c = tc.get(key)
        diff = float(dl - mp.mpf(c["Delta"])) if c else float("nan")
        E = c["E"] if c else float("nan")
        sv = r["sv_min"][-1] if "sv_min" in r else r["sv_scaled"][-1]
        na = r.get("noise_amp")
        n0 = r.get("noise_amp_a0")
        vals = (r["D"], r["M"], r.get("a", 1.0), r["ks"], r["maxF"], mp.nstr(dl, 16 if "Delta_lo" in r else 12), diff, E, sv, r["grip"],
                "%.1e" % na if na is not None else "", "%.1e" % n0 if n0 is not None else "", r["seconds"])
        if md:
            print("| %g | %d | %g | %d | %.1e | %s | %+.2e | %.1e | %.1e | %.1e | %s | %s | %.0f |" % vals)
        else:
            print("D %8g M %2d a %g ks %5d max|F| %.1e Delta %s  -C %+.2e (E %.1e)  sv_min %.1e grip %.1e noise %s / %s  %.0f s" % vals)

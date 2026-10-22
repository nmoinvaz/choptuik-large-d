"""Full-precision summary of the dinf_budget grid rows dumped by dinf_budget_wrap.py."""
import json
import sys

from mpmath import mp, mpf, nstr

mp.dps = 30
rows = json.load(open(sys.argv[1]))
for label, sel in (("D <= 45", lambda r: r["dmax"] == 45), ("D <= 60.1", lambda r: r["dmax"] == 60.1),
                   ("D <= 61.03", lambda r: r["dmax"] == 61.03),
                   ("quoted (60.1 and 61.03)", lambda r: r["dmax"] > 50)):
    L = [mpf(r["limit"]) for r in rows if r["used"] and sel(r)]
    lo, hi = min(L), max(L)
    print(f"{label:24s} {len(L):2d} fits: {nstr(lo, 12)} .. {nstr(hi, 12)}  centre {nstr((lo + hi) / 2, 12)}  "
          f"half-width {nstr((hi - lo) / 2, 3)}")
q = [r for r in rows if r["used"] and r["dmax"] > 50]
for r in sorted(q, key=lambda r: mpf(r["limit"]))[:2] + sorted(q, key=lambda r: mpf(r["limit"]))[-2:]:
    print(f"   extreme: D {r['dmin']}..{r['dmax']} {'clean' if r['clean'] else 'all'} {r['model']} deg {r['deg']}: "
          f"{nstr(mpf(r['limit']), 14)} res {r['res']}")

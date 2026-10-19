"""Best Delta(D), gamma(D) after the nonlinear-elimination re-solves.

Below D = 17.2 the older solutions (data_best.load) agree with ne_solve to ~1e-13 and are kept. Everything
above comes from the NE scans (scan_ne*.jsonl, ok records only, matching at x_K = 0.6, kc = 23 unless
another setup is asked for). Duplicate D (re-solves of the same point) are averaged; their spread is kept.

    rows = load()                      # {D: dict(D, Delta, gamma, src, spread)}
    rows = load(xK=0.7) / load(kc=31) / load(smooth=True)  # other setups, NE records only
"""
import glob, json
import numpy as np
from data_best import load as _old


def ne_records(xK=0.6, kc=23, smooth=False):
    recs = []
    for fn in sorted(glob.glob("scan_ne*.jsonl")):
        for l in open(fn):
            r = json.loads(l)
            if not r.get("ok"):
                continue
            if abs(r.get("xK", 0.6) - xK) > 1e-9:
                continue
            rkc = r.get("kc", 31 if "kc31" in fn else 23)
            if rkc != kc or bool(r.get("smooth", False)) != smooth:
                continue
            recs.append(r)
    return recs


def load(xK=0.6, kc=23, dmin_ne=17.2, smooth=False):
    by = {}
    for r in ne_records(xK, kc, smooth):
        by.setdefault(round(r["D"], 4), []).append(r)
    rows = {}
    if xK == 0.6 and kc == 23 and not smooth:
        for k, r in _old().items():
            if k < dmin_ne and r.get("Delta") is not None:
                rows[k] = dict(D=r["D"], Delta=r["Delta"], gamma=r.get("gamma"), src="old", spread=0.0)
    for k, rs in by.items():
        if k < dmin_ne and k in rows:
            continue
        d = np.array([r["Delta"] for r in rs])
        g = [r["gamma"] for r in rs if r.get("gamma")]
        rows[k] = dict(D=rs[0]["D"], Delta=float(d.mean()), gamma=(float(np.mean(g)) if g else None), src="ne",
                       spread=float(d.max() - d.min()))
    return rows


if __name__ == "__main__":
    rows = load()
    for k in sorted(rows):
        r = rows[k]
        print(f"{r['D']:8.4f} {r['Delta']:.10f} {r['gamma'] if r['gamma'] is None else round(r['gamma'], 8)} {r['src']} {r['spread']:.0e}")

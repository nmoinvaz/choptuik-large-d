"""All computed points D = 3.7 .. 40: (D, Delta, sigma, gamma) for fits over the whole range.

D < 8: downward sweeps at x_K 0.6 (scan_ne6_lowD_kc*_rk*.jsonl): kc 39 with Richardson in the RK step
(RK2 + (RK2 - RK1) / 15). The tau resolution matters at low D: kc 23 -> 31 -> 39 change Delta by -4.3e-4, -3.4e-5 at
D = 4 (ratio ~13), by <= 1e-8 above D = 5.5. sigma = max(3e-11, |kc39 - kc31| / 13), the next step of a geometric
convergence in kc. Where kc 47 re-solves exist (D <= 4.25) Delta is extrapolated in kc from 31, 39, 47 and sigma is
the size of that extrapolation step.
D >= 8: final_fit.step_converged (kc 23 + Richardson; agrees with kc 39 + Richardson to <= 4e-10 at D = 8 - 8.7).

    from data_all import load_all
    D, y, s, g = load_all()
"""
import glob
import json
import os

import numpy as np

from final_fit import step_converged


def _load(f):
    out = {}
    if os.path.exists(f):
        for l in open(f):
            r = json.loads(l)
            if r.get("ok"):
                out[round(r["D"], 4)] = r
    return out


def _rich(r1, r2):
    return r2["Delta"] + (r2["Delta"] - r1["Delta"]) / 15


def low_d():
    """{D: (Delta, sigma, gamma)} for D < 8 from the kc 31 / 39 / 47 sweeps."""
    k31 = [_load("scan_ne6_lowD_kc31_rk1.jsonl"), _load("scan_ne6_lowD_kc31_rk2.jsonl")]
    k39 = [_load("scan_ne6_lowD_kc39_rk1.jsonl"), _load("scan_ne6_lowD_kc39_rk2.jsonl")]
    k47 = [{}, {}]
    for f in glob.glob("scan_ne6_lowD_kc47*.jsonl"):
        for d, r in _load(f).items():
            k47[0 if r.get("rk", 1) == 1 else 1][d] = r
    out = {}
    for d in sorted(set(k39[0]) & set(k39[1])):
        if d >= 8.0:
            continue
        v39 = _rich(k39[0][d], k39[1][d])
        g = k39[1][d].get("gamma")
        if d in k31[0] and d in k31[1]:
            v31 = _rich(k31[0][d], k31[1][d])
            sig = max(3e-11, abs(v39 - v31) / 13)
        else:
            v31, sig = None, 1e-6
        if d in k47[1]:
            v47 = _rich(k47[0][d], k47[1][d]) if d in k47[0] else k47[1][d]["Delta"]
            if v31 is not None and abs(v39 - v31) > 0:
                q = (v47 - v39) / (v39 - v31)                     # geometric ratio of the kc steps
                if 0 < q < 0.5:
                    ext = v47 + (v47 - v39) * q / (1 - q)
                    out[d] = (ext, max(3e-11, abs(ext - v47)), k47[1][d].get("gamma", g))
                    continue
            out[d] = (v47, max(3e-11, abs(v47 - v39) / 13), k47[1][d].get("gamma", g))
            continue
        out[d] = (v39, sig, g)
    return out


def load_all():
    D, y, s, g = step_converged(Dmin=8.0)
    lo = low_d()
    Dl = np.array(sorted(lo))
    return (np.r_[Dl, D], np.r_[[lo[d][0] for d in Dl], y], np.r_[[lo[d][1] for d in Dl], s],
            np.r_[[lo[d][2] if lo[d][2] is not None else np.nan for d in Dl], g])


if __name__ == "__main__":
    D, y, s, g = load_all()
    for d, v, e, gg in zip(D, y, s, g):
        if d < 9:
            print(f"{d:7.4f} {v:.10f} +- {e:.1e}  gamma {gg:.8f}")
    print(f"{len(D)} points, D = {D.min():.2f} .. {D.max():.2f}")

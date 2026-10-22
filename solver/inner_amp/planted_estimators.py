#!/usr/bin/env python3
"""Harness for the planted-truth test (notes/PLANTED_SELECTION.md): runs the production selection estimators on a
directory of scan JSONs (the scan25phys format: one file per wf with "wf" and "sigma_str") and returns their numbers.

Nothing here re-implements an estimator except the Pade crossing rule, which exists only as prose in comms.txt (T24,
T25: "Im(eps) = 0 crossings, linear between the bracketing omegas"):
  - arg tau_k = pi crossings and ratio estimates, quadratic in wf: plot_selection.crossings (the README table);
  - the same, linear, plus the Aitken line: arg_crossing.py, run as a subprocess and parsed (the crossing.txt lines);
  - Pade poles of (ln S_1)': zero_selection.zeros, then the reconstructed crossing rule (accepted pole nearest X0 per
    approximant and node, sign change of Im sqrt(x) between adjacent nodes, linear interpolation);
  - the factored series G = exp(-kbar x) S_1: solver/proof/entire_cert.resummed (SCAN and WFS pointed at the plant).
mpmath precision is set to each production script's own value before each call (crossings 50, Pade 40, G 60).

    python3 planted_estimators.py DIR [K]      prints the estimates for one scan directory
"""
import contextlib
import io
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOLVER = os.path.dirname(HERE)
for p in (HERE, SOLVER, os.path.join(SOLVER, "proof")):
    if p not in sys.path:
        sys.path.insert(0, p)

import mpmath as mp  # noqa: E402

import zero_selection  # noqa: E402
import plot_selection  # noqa: E402
import entire_cert  # noqa: E402
from arg_crossing import taus  # noqa: E402

X0 = entire_cert.X0                  # 7.397, the production prior for the zero (findroot start, kbar objective)
ARG_CROSSING = os.path.join(HERE, "arg_crossing.py")


def write_scan(dirpath, nodes, K=None):
    """nodes: {wf_label: [sigma_0, sigma_1, ...] (mpc)} -> one JSON per node, truncated to sigma_0..sigma_K"""
    os.makedirs(dirpath, exist_ok=True)
    for fn in os.listdir(dirpath):
        if fn.endswith(".json"):
            os.remove(os.path.join(dirpath, fn))
    for wf, sig in nodes.items():
        s = sig if K is None else sig[:K + 1]
        rec = {"wf": wf, "K": 2 * (len(s) - 1) + 1, "J": 0, "prec": 0,
               "sigma_str": [[mp.nstr(mp.re(z), 30), mp.nstr(mp.im(z), 30)] for z in s],
               "sigma": [[float(mp.re(z)), float(mp.im(z))] for z in s]}
        with open(os.path.join(dirpath, f"sigF_w{wf:.5f}_planted.json"), "w") as f:
            json.dump(rec, f)


def aitken(x0, x1, x2):
    den = (x2 - x1) - (x1 - x0)
    return None if den == 0 else x2 - (x2 - x1) ** 2 / den


def quad_crossings(dirpath):
    mp.mp.dps = 50
    above, below = plot_selection.crossings(dirpath)
    return above, below


def linear_crossings(dirpath):
    """arg_crossing.py stdout, parsed"""
    r = subprocess.run([sys.executable, ARG_CROSSING, dirpath], capture_output=True, text=True, timeout=600)
    out = r.stdout
    arg = {int(k): float(w) for k, w in re.findall(r"arg\(tau_(\d+)\) = pi at wf = ([-\d.]+)", out)}
    ait = re.findall(r"Aitken limit of the last three crossings: wf = ([-\d.]+)", out)
    rat = {int(k) + 1: float(w) for k, w in re.findall(r"ratio tau_(\d+)/tau_\d+: eps0 real at wf = ([-\d.]+)", out)}
    return {"arg": arg, "aitken": float(ait[0]) if ait else None, "ratio": rat, "stdout": out,
            "returncode": r.returncode, "stderr": r.stderr[-2000:]}


def pade_crossings(dirpath):
    """zero_selection.zeros per node; per [m/q] the accepted pole (|res - 1| < 0.3) nearest X0; Im sqrt(x) sign change
    between adjacent nodes, linear. Returns {(m, q): (wf_cross or None, {wf: x})}"""
    mp.mp.dps = 40
    data = zero_selection.load([dirpath])
    poles = {}
    for wf, sig, _ in data:
        for m, q, r, res in zero_selection.zeros(sig, verbose=False):
            if abs(res - 1) < 0.3:
                key = (m, q)
                cur = poles.setdefault(key, {})
                if wf not in cur or abs(r - X0) < abs(cur[wf] - X0):
                    cur[wf] = r
    wfs = [d[0] for d in data]
    out = {}
    for key, per in sorted(poles.items()):
        cross = None
        if all(w in per for w in wfs):
            ims = [float(mp.im(mp.sqrt(per[w]))) for w in wfs]
            for i in range(len(wfs) - 1):
                if ims[i] == 0:
                    cross = wfs[i]
                    break
                if ims[i] * ims[i + 1] < 0:
                    cross = wfs[i] + ims[i] / (ims[i] - ims[i + 1]) * (wfs[i + 1] - wfs[i])
                    break
        out[key] = (cross, {w: complex(per[w]) for w in per})
    return out, len(data[0][1]) - 2 if data else 0     # n = len(dL) - 1, dL = (k tau_k), k = 1..K


def factored(dirpath, wfs, Kmax, mid_key=None):
    """entire_cert.resummed on the plant. Returns {K: (kbar, [roots], last_terms)} for K = 9..Kmax, or an error string.
    mid_key: if the middle label is not exactly 1.0 (shifted-grid corruption), production raises; with mid_key given,
    the middle node is also offered under the key 1.0, which resummed() hard-codes for kbar."""
    mp.mp.dps = 60
    entire_cert.SCAN = dirpath
    entire_cert.WFS = tuple(wfs)
    try:
        S = {w: entire_cert.load(w) for w in wfs}
        if mid_key is not None:
            S[1.0] = S[mid_key]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = entire_cert.resummed(S, Kmax)
        last = {int(k): float(v) for k, v in re.findall(r"K (\d+): kbar .*?, last terms ([-\d.e+]+);", buf.getvalue())}
        return {K: (complex(kb), [float(x) for x in r], last.get(K)) for K, (kb, r) in res.items()}
    except Exception as e:                                      # noqa: BLE001  (reported, not swallowed)
        return f"{type(e).__name__}: {e}"
    finally:
        entire_cert.SCAN = os.path.join(SOLVER, "inner_amp", "scan25phys")
        entire_cert.WFS = (0.99, 1.0, 1.01)


def nearest(vals, target=1.0):
    return min(vals, key=lambda v: abs(v - target)) if vals else None


def run_all(dirs_by_K, wfs, Kmax_G=None, mid_key=None, linear=True):
    """dirs_by_K: {K: directory with sigma_0..sigma_K}. Returns a dict of estimates per K."""
    out = {}
    for K, d in sorted(dirs_by_K.items()):
        errors = {}
        try:
            above, below = quad_crossings(d)
        except Exception as e:                                  # noqa: BLE001  (recorded in "errors", counted in report)
            errors["crossings"] = f"{type(e).__name__}: {e}"
            above, below = {}, {}
            try:                                                # the arg part alone (crossings() computes both)
                mp.mp.dps = 50
                rows = []
                for fn in sorted(os.listdir(d)):
                    r = json.load(open(os.path.join(d, fn)))
                    rows.append((r["wf"], taus([mp.mpc(mp.mpf(a), mp.mpf(b)) for a, b in r["sigma_str"]])))
                rows.sort()
                ws = [w for w, _ in rows]
                import numpy as np
                for k in range(4, len(rows[0][1])):
                    vals = [float(mp.arg(t[k]) / mp.pi) % 2 - 1 for _, t in rows]
                    c = np.polyfit(ws, vals, 2)
                    rts = [x.real for x in np.roots(c) if abs(x.imag) < 1e-12 and ws[0] - 0.03 < x.real < ws[-1] + 0.03]
                    above[k] = min(rts, key=lambda x: abs(x - 1)) if rts else None
            except Exception as e2:                             # noqa: BLE001
                errors["crossings_arg_only"] = f"{type(e2).__name__}: {e2}"
        rec = {"arg": {k: v for k, v in above.items()}, "ratio": {k: v for k, v in below.items()}, "errors": errors}
        ks = sorted(k for k, v in above.items() if v is not None)
        rec["aitken_quad"] = {k: aitken(above[k - 2], above[k - 1], above[k]) for k in ks
                              if k - 2 in above and None not in (above[k - 2], above[k - 1], above[k])}
        if linear:
            lin = linear_crossings(d)
            rec["linear"] = {k: lin[k] for k in ("arg", "aitken", "ratio", "returncode")}
            rec["linear_stdout"] = lin["stdout"]
        try:
            pc, n = pade_crossings(d)
        except Exception as e:                                  # noqa: BLE001
            errors["pade"] = f"{type(e).__name__}: {e}"
            pc, n = {}, K - 1
        rec["pade"] = {f"{m}/{q}": c for (m, q), (c, _) in pc.items()}
        rec["pade_poles_mid"] = {f"{m}/{q}": [per.get(w) for w in wfs] for (m, q), (_, per) in pc.items()}
        central = [f"{m}/{n - m}" for m in range(3, n - 3)]
        cv = [rec["pade"].get(c) for c in central]
        rec["pade_central"] = central
        rec["pade_central_vals"] = cv
        ok = [v for v in cv if v is not None]
        rec["pade_central_mid"] = (min(ok) + max(ok)) / 2 if ok and len(ok) == len(cv) else None
        rec["pade_central_range"] = (min(ok), max(ok)) if ok else None
        out[K] = rec
    Kmax = Kmax_G or max(dirs_by_K)
    G = factored(dirs_by_K[Kmax], wfs, Kmax, mid_key=mid_key)
    for K in out:
        if isinstance(G, str):
            out[K]["G"], out[K]["G_err"] = None, G
        elif K in G:
            kb, roots, last = G[K]
            out[K]["G"] = nearest(roots)
            out[K]["G_roots"] = roots
            out[K]["G_last"] = last
            out[K]["G_kbar"] = [kb.real, kb.imag]
            out[K]["G_prev"] = nearest(G[K - 1][1]) if K - 1 in G else None
            out[K]["G_last_prev"] = G[K - 1][2] if K - 1 in G else None
    return out


def main():
    d = sys.argv[1]
    K = int(sys.argv[2]) if len(sys.argv) > 2 else None
    import glob
    wfs = sorted(json.load(open(f))["wf"] for f in glob.glob(os.path.join(d, "*.json")))
    n = min(len(json.load(open(f))["sigma_str"]) for f in glob.glob(os.path.join(d, "*.json"))) - 1
    K = K or n
    res = run_all({K: d}, wfs)
    r = res[K]
    print(f"{d}: K = {K}, wf nodes {wfs}")
    print("  arg crossings (quadratic):", {k: round(v, 6) if v else v for k, v in r["arg"].items()})
    print("  ratio estimates (quadratic):", {k: round(v, 6) if v else v for k, v in r["ratio"].items()})
    print("  Aitken (quadratic):", {k: round(v, 6) for k, v in r["aitken_quad"].items()})
    print("  linear (arg_crossing.py):", r["linear"])
    print("  Pade crossings:", {k: round(v, 6) if v else v for k, v in r["pade"].items()})
    print("  Pade central", r["pade_central"], "range", r["pade_central_range"], "mid", r["pade_central_mid"])
    print("  G:", r.get("G"), "prev", r.get("G_prev"), "last", r.get("G_last"), r.get("G_err", ""))


if __name__ == "__main__":
    main()

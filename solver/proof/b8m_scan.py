#!/usr/bin/env python3
"""Lemma B8 (M = 1): scan of the bootstrap closure over ln n_0 (b8m_bounds.py), warm-started.

For each L0 = ln n_0: iterate b <- P(b) from the previous fixed point (or 0) until the relative change is < 1e-5
(at most --iters).  Closure: P(h) < h componentwise, which certifies Lemma B8 with the constants h.  The ray h = s b*
cannot close in the V component (P_V(h) = c h_S is linear and homogeneous, so P_V(s b*) = s P_V(b*) >= s b*_V for
an iterate from below), so the test descends from above: h_0 = s b*, h_{k+1} = (1 + eps) P(h_k), and stops at the
first k with (1 + eps) P(h_k) <= h_k, i.e. P(h_k) < h_k strictly (s in --slacks, eps = --eps, at most --down steps).
The run is appended to the log as JSON lines.

    python3 b8m_scan.py K L0,L0,... [--outer=full|arc] [--slacks=1.1,1.5] [--eps=0.01] [--down=25] [--iters=120]
            [--log=file]
            [b8m_bounds options: --kappa --kappa1 --gamma0 --delta --xF --r1 --rA --rW]
"""
import json
import sys
import time
from flint import arb
import b8m_atoms as AT
import b8m_bounds as MB

OPTS = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
pos = [a for a in sys.argv[1:] if not a.startswith("--")]
K = int(pos[0])
Ls = [float(v) for v in pos[1].split(",")]
slacks = [arb(s) for s in OPTS.get("slacks", "1.1,1.5").split(",")]
eps = arb(OPTS.get("eps", "0.01"))
down = int(OPTS.get("down", "25"))
iters = int(OPTS.get("iters", "120"))
kw = {k: OPTS[k] for k in ("outer", "kappa", "kappa1", "gamma0", "delta", "xF", "r1") if k in OPTS}
log = OPTS.get("log")
t0 = time.time()
at = AT.atoms(K, OPTS.get("rA", "0"), OPTS.get("rW", "0"))
b = dict(V=arb(0), S=arb(0), b=arb(0), c=arb(0))
for L in Ls:
    g = MB.Geo(K, repr(L), atoms=at, **kw)
    rec = dict(K=K, L0=L, outer=g.outer, rho_max=float(g.rho_max), **{k: OPTS.get(k) for k in kw})
    try:
        k = MB.known(g)
        kc = MB.kernel_consts(g, k)
        rec["kernel"] = {a: float(v) for a, v in kc.items()}
        diffs = []
        for it in range(iters):
            nb, _ = MB.improve(g, k, kc, b)
            rel = max(abs(float(nb[s]) - float(b[s])) / (1e-30 + float(nb[s])) for s in b)
            diffs.append(rel)
            b = nb
            if rel < 1e-5 or any(float(v) > 1e40 for v in b.values()):
                break
        rec["iters"] = it + 1
        rec["last_rel_changes"] = diffs[-3:]
        rec["b_fixed"] = {s: float(v) for s, v in b.items()}
        rec["closes"] = None
        for sl in slacks:
            h = {s: MB.up(b[s] * sl) for s in b}
            ok = False
            try:
                for st in range(down):
                    Pt, dg = MB.improve(g, k, kc, h, verbose=True)
                    Th = {s: MB.up(Pt[s] * (1 + eps)) for s in b}
                    if all(Th[s] <= h[s] for s in b):
                        ok = all(Pt[s] < h[s] for s in b)
                        break
                    h = Th
            except ValueError as ex:
                continue
            if ok:
                rec["closes"] = float(sl)
                rec["down_steps"] = st
                rec["hyp"] = {s: float(v) for s, v in h.items()}
                rec["P_hyp"] = {s: float(v) for s, v in Pt.items()}
                rec["ratio"] = {s: float(Pt[s] / h[s]) for s in b}
                rec["diag"] = {a: float(v) for a, v in dg.items()}
                break
    except ValueError as ex:
        rec["fail"] = str(ex)
        b = dict(V=arb(0), S=arb(0), b=arb(0), c=arb(0))
    rec["time"] = round(time.time() - t0, 1)
    line = json.dumps(rec)
    print(line, flush=True)
    if log:
        with open(log, "a") as fh:
            fh.write(line + "\n")

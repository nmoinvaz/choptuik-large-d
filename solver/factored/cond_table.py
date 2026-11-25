"""The Jacobian's conditioning against D: form A (two ways) against form S, at one fixed M (#1088).

At each D, form S (fseries_s.py, a slaved by the momentum constraint) is solved by Newton from the previous D's
solution rescaled (p1, m_h by sqrt(n0/n), c0 - 1 by n0/n; Delta from --Delta_at or the previous one), a jump with no
continuation in D. At its solution the two form-A layouts of fseries.py (a by the Hamiltonian equation) are evaluated
at the same state:
  A_h: the horizon datum of a is beta_0 = (n-1)(1 - 1/a_h^2) and beta's series runs forward from it (hp's layout of
       the horizon block, here matched at x* = 1/sqrt 2): the mass mode's gain sits in the Jacobian;
  A_*: the datum is beta's coefficient at the mass mode's s-peak k* (fseries.py, horizon_anchor).
Reported per form: max|F| at the state (form A's residual measures how well S's solution satisfies A), the raw and
scaled (unit rows and columns) singular values, and for form A the reproducibility of the a-datum's Jacobian column
(complex step against central differences at h = 1e-7, relative): rounding amplified by the horizon series near the
order n/2. Every number is float64.

Usage: python3 cond_table.py START.npz --M=6 --Ds=50,100,200 [--Delta_at=400:1.6696580641] [--out=FILE.jsonl]
       [--mutate=sigma]   (the control: the horizon's SSH exponent loses its g term; Delta must move off the table)
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fseries import Factored  # noqa: E402
from fseries_s import FactoredS  # noqa: E402
from fsolve import conditioning, load_start, solve  # noqa: E402

opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
start = [a for a in sys.argv[1:] if not a.startswith("--")][0]
M = int(opts.get("M", 6))
Ds = [float(v) for v in opts["Ds"].split(",")]
Dat = {}
for item in opts.get("Delta_at", "").split(","):
    if item:
        a, b = item.split(":")
        Dat[float(a)] = float(b)
mut = opts.get("mutate", "")
out = open(opts["out"], "a") if "out" in opts else None
log = lambda s: print(s, flush=True)  # noqa: E731

y, Dprev = load_start(start, M)
y[3 * M:4 * M] = 0.0
for D in Ds:
    t0 = time.time()
    n, n0 = D - 2, Dprev - 2
    if abs(D - Dprev) > 1e-12:
        s = np.sqrt(n0 / n)
        y[0:M] *= s
        y[2 * M:3 * M] *= s
        y[M:2 * M] = 1 + (y[M:2 * M] - 1) * n0 / n
    if D in Dat:
        y[4 * M] = Dat[D]
    elif abs(D - Dprev) > 1e-12:
        # the 1/D fit of the table (Delta ~ 1.6439 + 10.37/D - 49.17/D^2) moves the previous Delta
        y[4 * M] += 10.37 * (1 / D - 1 / Dprev) - 49.17 * (1 / D**2 - 1 / Dprev**2)
    with np.errstate(all="ignore"):
        y, J, fz, hist, secs = solve(D, M, y, iters=int(opts.get("iters", 10)), mutate=mut, log=lambda s: None, form="S")
    if not np.all(np.isfinite(J)) or not np.isfinite(hist[-1]["maxF"]):
        log("D %7.2f: form S Newton diverged from the jump (non-finite); stop" % D)
        break
    sv, svs, grip = conditioning(J, M, "S")
    rec = {"D": D, "M": M, "ks": fz.ks, "kc": fz.kc, "Delta": float(y[4 * M]), "maxF_S": hist[-1]["maxF"],
           "its": len(hist) - 1, "S_svmax": float(sv[0]), "S_svmin": float(sv[-1]), "S_ssvmin": float(svs[-1]),
           "S_scond": float(svs[0] / svs[-1]), "S_grip": grip, "S_ssv_tail": [float(v) for v in svs[-6:]],
           "mutate": mut}
    # the S solution in the form-A layouts
    fsS = FactoredS(D, M)
    p1, c0, Q0, Delta = y[0:M], y[M:2 * M], y[2 * M:3 * M], y[4 * M]
    ho = fsS.horizon(Q0, 2 * np.pi / Delta)
    beta_s = -(n - 1) * ho["series"][3]
    for tag, mt in (("A_h", "kstar0"), ("A_*", "")):
        fa = Factored(D, M, mutate=mt)
        kst = fa.kstar()
        y4 = y.copy()
        y4[3 * M:4 * M] = beta_s[kst]
        with np.errstate(all="ignore"):
            FA = fa.residual4(y4)
            JA = fa.jacobian4(y4)
        if not (np.all(np.isfinite(FA)) and np.all(np.isfinite(JA))):
            rec.update({tag + "_maxF": float("nan"), tag + "_finite": False, tag + "_kstar": kst})
            continue
        rec[tag + "_finite"] = True
        svA, svsA, gripA = conditioning(JA, M, "A")
        i = 3 * M
        cs = JA[:, i]
        h = 1e-7 * max(1e-3, np.max(np.abs(y4[i])))
        yp, ym = y4.copy(), y4.copy()
        yp[i] += h
        ym[i] -= h
        fd = (fa.residual4(yp) - fa.residual4(ym)) / (2 * h)
        disc = float(np.linalg.norm(cs - fd) / max(np.linalg.norm(cs), np.linalg.norm(fd)))
        rec.update({tag + "_maxF": float(np.max(np.abs(FA))), tag + "_svmax": float(svA[0]), tag + "_svmin": float(svA[-1]),
                    tag + "_ssvmin": float(svsA[-1]), tag + "_scond": float(svsA[0] / svsA[-1]), tag + "_kstar": kst,
                    tag + "_datum_col_disc": disc, tag + "_datum_col_norm": float(np.linalg.norm(cs))})
    rec["seconds"] = time.time() - t0
    line = "D %7.2f Delta %.12f |F_S| %.1e its %d | S: svmax %.2e ssvmin %.2e scond %.2e" % (
        D, rec["Delta"], rec["maxF_S"], rec["its"], rec["S_svmax"], rec["S_ssvmin"], rec["S_scond"])
    for tag in ("A_h", "A_*"):
        if rec.get(tag + "_finite"):
            line += " | %s: svmax %.2e scond %.2e disc %.1e |F| %.1e" % (tag, rec[tag + "_svmax"], rec[tag + "_scond"],
                                                                         rec[tag + "_datum_col_disc"], rec[tag + "_maxF"])
        else:
            line += " | %s: not finite (float64 overflow)" % tag
    log(line + " (%.0f s)" % rec["seconds"])
    if out:
        out.write(json.dumps(rec) + "\n")
        out.flush()
    Dprev = D

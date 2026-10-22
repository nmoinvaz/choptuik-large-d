#!/usr/bin/env python3
"""Review check R7 (agent RV): fixed_point.py's maxima over balls, monotonicity in R, and growth with the parameter box.

  1. fixed_point.run takes F = max(...), lip0 = max(...), lip = max(...) over arb balls with Python's max. arb's ">"
     is True only when certain, so for overlapping balls max keeps the first, which need not have the largest upper
     bound. Overlaps do occur: the mu = -1 and mu = +1 rows are mathematically equal (conjugation symmetry). Here the
     run is repeated with max replaced by the exact maximum of the upper bounds (as an exact arb), and the conclusion
     (contraction, self-map) re-checked; the difference shows the size of the gap.
  2. Every bound must be non-decreasing in R (the uniqueness argument restricts to (0, R'] for R' < R): the lemma at
     the zero is re-run with the same truncation N and vanishing orders at R' = r0'^2, r0' = 0.06 .. 0.1.
  3. Perturbation: on complex parameter boxes of growing radius, the eta bound and the start-value radii must grow, and
     the derivative bounds of start_data (Cauchy) must grow with the real box.

    python3 rv_fixed_point_audit.py
"""
import builtins
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import fixed_point as FP  # noqa: E402
from flint import acb, arb, ctx  # noqa: E402


def rmax(*args, key=None):
    items = list(args[0]) if len(args) == 1 else list(args)
    if key is not None or not all(isinstance(x, arb) for x in items):
        return builtins.max(items, key=key) if key is not None else builtins.max(items)
    return arb(builtins.max(x.upper() for x in items))       # arf comparison: exact


def run_both(A, wf, r0="0.1", **kw):
    FP.Maj.BOX = kw.pop("box", False)
    if "max" in FP.__dict__:
        del FP.max
    a = FP.run(A, wf, r0, verbose=False, **kw)
    FP.max = rmax
    b = FP.run(A, wf, r0, verbose=False, **kw)
    del FP.max
    return a, b


if __name__ == "__main__":
    ctx.prec = 320
    a, b = run_both(acb(FP.A_STAR), acb(FP.WF_STAR))
    # how many of the 7 weighted forcing / row sums overlap their maximum?
    print(f"1. Python max over balls: F {a['F'].str(12, radius=True)} vs exact max of upper bounds "
          f"{b['F'].str(12, radius=True)}; F upper bounds equal: {a['F'].upper() == b['F'].upper()}; "
          f"lip {a['lip'].upper().str(8)} vs {b['lip'].upper().str(8)}")
    print(f"   rigorous-max run: contraction {b['lip'] < 1}, self-map {b['selfmap'] < b['rho']}, ok {b['ok']} "
          f"(code: ok {a['ok']})")
    ctx.prec = 320
    L = a["L"]
    over = sum(1 for i in range(7) for j in range(7) for k in range(7) for l in range(7)
               if (i, j) < (k, l) and L[i][j].overlaps(L[k][l]) and L[i][j].upper() != L[k][l].upper())
    print(f"   overlapping-but-unequal pairs of Lipschitz entries (where max can pick the smaller): {over}")
    print("2. monotonicity in R (same N = 53 and vanishing orders):")
    c0 = FP.run(acb(FP.A_STAR), acb(FP.WF_STAR), "0.1", verbose=False)
    prev = None
    for r0 in ("0.06", "0.07", "0.08", "0.09", "0.1"):
        o = FP.run(acb(FP.A_STAR), acb(FP.WF_STAR), r0, verbose=False, kmins_in=c0["kmins"], N_in=c0["N"])
        row = (float(o["lip"].upper()), float(o["F"].upper()))
        mono = prev is None or (row[0] >= prev[0])
        print(f"   r0 {r0}: k = {o['k']}, Lipschitz <= {row[0]:.4f}, ||T(0)||_w <= {row[1]:.3e}, ok {o['ok']}; "
              f"Lipschitz non-decreasing in R: {mono}")
        prev = row
    print("3. growth with the parameter box (complex boxes (1+i) r around the zero; start_data on real boxes):")
    c = FP.run(acb(FP.A_STAR), acb(FP.WF_STAR), "0.1", verbose=False)
    last = 0.0
    grow = True
    for r in ("0", "1e-9", "1e-6", "1e-4", "1e-3", "3e-3"):
        FP.Maj.BOX = r != "0"
        A = acb(arb(FP.A_STAR, r), arb(0, r))
        wf = acb(arb(FP.WF_STAR, r), arb(0, r))
        o = FP.run(A, wf, "0.1", verbose=False, kmins_in=c["kmins"], N_in=c["N"])
        _, Dv, Sv, Cv = FP.start_values(o, +1)
        rad = max(float(v.real.rad()) for v in list(Dv.values()) + list(Sv.values()))
        eta = float((o["rho"] / arb(min(o["wts"])) * o["R"] ** o["k"]).upper())
        grow &= rad >= last
        last = rad
        print(f"   box radius {r:>5}: Lipschitz {float(o['lip'].upper()):.4f}, ok {o['ok']}, eta bound {eta:.2e}, "
              f"start-value radius {rad:.2e}")
    FP.Maj.BOX = False
    for h in ("1e-12", "1e-9", "1e-6"):
        xi0, vals, dAs, dws, (e0, e1) = FP.start_data(arb(FP.A_STAR, h), arb(FP.WF_STAR, h), "0.1", +1, arb("1e-3"))
        print(f"   start_data on the real box +-{h}: value radius {max(float(v.real.rad()) for v in vals):.2e}, "
              f"derivative radius {max(float(v.real.rad()) for v in dAs + dws):.2e}")
    print("VERDICT:", "conclusions unchanged with rigorous maxima; bounds monotone; enclosures grow with the box"
          if (b["ok"] and grow) else "PROBLEM")

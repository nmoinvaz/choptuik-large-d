"""Independent check of hp's Moebius horizon (hp/src/form_s.rs horizon_mob, wip/large-d-form-s; Solver.set_horizon_mobius)
against this branch's double-double recurrence (mobius_sdd.py), at large D (#1088, #1090).

Written before any run (2026-11-25). Inputs at each D: M = 32 collocation points; m_h = Q0/2 of this branch's float64
M-6 solution at D 646 (I1090-mac/f64_M6_cont3/sol_D646_M6_a8.json) resampled to M = 32 and rescaled by sqrt(644/n);
Delta from Track C at D (I1090-mac/diag/trackc_at.json); the last node x_l = 0.7; ks = form_s.ks_mob(D) = ceil(1.5 n) + 20
in both codes; a = 8 for D <= 3200 and n/400 beyond (the rule of notes/LARGE_D_INNER.md §3). hp is evaluated with its
last node at x_l and that node's state zero, so its horizon block (the series at x_l minus the last node) is its sum S = (p, m, c) at
x_l (hp needs two nodes: 0.65 and x_l, both with zero state). Both codes take the same float64 m_h and Delta. hp in octo-double (8 parts in, the result summed in mpmath), this
branch in double-double, both with every Cauchy product in full (kb = None here; hp has no banding).

Checks and pass bars:
  C1  agreement: max |S_hp - S_dd| / max |S_dd| <= 1e-24 at every D.
  C2  the gain from m_h (hprobe's: m_h moved by eps times |m_h| with random signs, default_rng(1); the largest change of
      S over eps): hp's against this branch's within 1e-3 relative at eps = 1e-20; hp's gains at eps = 1e-20 and 1e-10
      within a factor 2 of each other (linear) and below 1e3.
  C3  invariance: this branch's sums at a and at 1.25 a agree within 1e-24 (the series is the same function in any
      Moebius variable, so a consistent change of a must not move the sum).
  Control (must fail C1): hp with FS_MUT_MOBIUS (mutate 52: z's series with a's sign flipped, h and e kept), and this
      branch with z's series at 1.25 a while h and e keep a (--mutate=zseries). Each must differ from the other code by
      more than 1e-10.
  Information (not pass/fail): hp at a = 8 at D 5000 and 10^4, against this branch at the rule's a: whether a fixed a
      fails past D 5000 in hp's octo-double as it does in this branch's float64 (exponent range).

Usage (PYTHONPATH must hold hp's form-S build, and FORMS_HP the worktree's hp/ with form_s.py):
    python3 hp_mob_check.py SOL.json TRACKC.json --Ds=1000,2000 [--parts=8] [--threads=2] [--gain=1]
                            [--a_fixed=8] [--mutate=zseries|hp52]
  --a_fixed: hp's a only (this branch keeps the rule's a), for the information rows
"""
import json
import os
import sys
import time

import numpy as np
import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("FORMS_HP", ""))
import ddnp as d  # noqa: E402
from ddnp import DD  # noqa: E402
import mobius_s as ms  # noqa: E402
import mobius_sdd as msd  # noqa: E402

opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
fn, tcf = [a for a in sys.argv[1:] if not a.startswith("--")][:2]
Ds = [float(v) for v in opts.get("Ds", "1000,2000").split(",")]
parts = int(opts.get("parts", 8))
threads = int(opts.get("threads", 2))
want_gain = int(opts.get("gain", 1))
a_fixed = float(opts["a_fixed"]) if "a_fixed" in opts else None
mut = opts.get("mutate", "")
M = 32
XL = 0.7
r = json.load(open(fn))
tc = json.load(open(tcf))["values"]
mh0 = ms.resample_odd(np.array(r["Q0"]), M) / 2.0
D0 = r["D"]

from form_s import make_solver_s, ks_mob  # noqa: E402

mp.mp.dps = 140


def split_parts(v, parts):
    """an mpmath number as `parts` non-overlapping doubles (a normalized expansion)"""
    out = []
    for _ in range(parts):
        f = float(v)
        out.append(f)
        v = v - f
    return out


def hp_sum(D, a, mh, Dl, mutate=0, dmh=None):
    """hp's horizon sum at x_l in octo-double: one node at x_l with zero state, the horizon block of residual_s"""
    with mp.workdps(60):
        Dt = mp.mpf(repr(D))
    K = 2  # hp needs two nodes; the second is x_l, and its state is zero, so the horizon block is the sum at x_l
    s = make_solver_s(Dt, M, [XL - 0.05, XL], kt=36, kc=200, ks=ks_mob(D), threads=threads, mob=a)
    L = 3 * M * (K + 1) + 1
    z = np.zeros(L)
    z[M:2 * M] = 1.0  # c0 (p1 = 0; the centre and segment blocks are not used)
    o = 2 * M + 3 * M * K  # m_h after p1, c0 and the nodes' p, m, c
    z[o:o + M] = mh
    z[-1] = Dl
    zp = [np.ascontiguousarray(z)] + [np.zeros(L) for _ in range(parts - 1)]
    if dmh is not None:  # m_h + dmh carried in all the parts (a float64 m_h cannot hold a move of 1e-20)
        for i in range(M):
            ps = split_parts(mp.mpf(float(mh[i])) + mp.mpf(float(dmh[i])), parts)
            for p_ in range(parts):
                zp[p_][o + i] = ps[p_]
    F = s.residual_s(zp, mutate)
    Fm = [sum(mp.mpf(float(F[p][i])) for p in range(len(F))) for i in range(3 * M * K, 3 * M * (K + 1))]
    return Fm


def dd_sum(D, a, mh, Dl, zmut=None, dmh=None):
    """this branch's sum at x_l in double-double (kb None), as p = (U + x V)/2, m = (U - x V)/2, c"""
    zK = mp.mpf(XL) * mp.mpf(XL)  # x_l^2 of the float64 x_l, exactly, as hp forms it in the lane's precision
    f = msd.FormSDD(D, M, a=a, ks=ks_mob(D), kb=None, zK=zK)
    if zmut is not None:
        g = msd.FormSDD(D, M, a=zmut, ks=ks_mob(D), kb=None, zK=zK)
        f.ts_dd, f.ats_dd = f.ts_dd, g.ats_dd * f.ts_dd / g.ts_dd  # z's series (via a t*) at zmut, h and e at a
    Q0 = DD(2.0 * mh) if dmh is None else DD(2.0 * mh) + DD(2.0 * dmh)  # the move added in double-double
    om = (f.pi * 2.0) / DD(np.array(Dl))
    h = f.horizon(Q0, om)
    xl = DD(np.array(XL))
    p = (h["U"] + h["V"] * xl) * 0.5
    m = (h["U"] - h["V"] * xl) * 0.5
    out = []
    for X in (p, m, h["c"]):
        out += [mp.mpf(float(u)) + mp.mpf(float(v)) for u, v in zip(X.h, X.l)]
    return out


def rel(A, B):
    sc = max(abs(v) for v in B)
    return float(max(abs(x - y) for x, y in zip(A, B)) / sc)


def gain(fun, D, a, mh, Dl, eps):
    """the largest change of S over eps for m_h moved by eps |m_h| with random signs (default_rng(1)), the move
    carried in the extra parts, not added to the float64 m_h"""
    dirn = np.abs(mh) * np.random.default_rng(1).choice([-1, 1], M)
    S0 = fun(D, a, mh, Dl)
    S1 = fun(D, a, mh, Dl, dmh=eps * dirn)
    g = float(max(abs(x - y) for x, y in zip(S1, S0)) / eps)
    return g


ok = True
print("hp_mob_check: M %d, x_l %g, parts %d, start %s (D %g), mutate %r" % (M, XL, parts, os.path.basename(fn), D0, mut))
for D in Ds:
    n = D - 2
    a = 8.0 if D <= 3200 else n / 400.0  # the rule (notes/LARGE_D_INNER.md §3)
    a_hp = a_fixed if a_fixed is not None else a  # --a_fixed moves hp's a only (the information rows)
    mh = mh0 * np.sqrt((D0 - 2) / n)
    Dl = float(tc["%g" % D]["Delta"])
    t0 = time.time()
    Shp = hp_sum(D, a_hp, mh, Dl, 52 if mut == "hp52" else 0)
    t_hp = time.time() - t0
    t0 = time.time()
    Sdd = dd_sum(D, a, mh, Dl, zmut=1.25 * a if mut == "zseries" else None)
    t_dd = time.time() - t0
    c1 = rel(Shp, Sdd)
    per = []
    for nm, sl in (("p", slice(0, M)), ("m", slice(M, 2 * M)), ("c", slice(2 * M, 3 * M))):
        A, B_ = Shp[sl], Sdd[sl]
        sc = max(abs(v) for v in B_) if nm != "c" else mp.mpf(1)
        per.append("%s %.1e (|%s| %.2e)" % (nm, float(max(abs(x - y) for x, y in zip(A, B_)) / sc), nm,
                                           float(max(abs(v) for v in B_)) if nm != "c" else float(max(abs(v - 1) for v in B_))))
    print("   per field, relative to its size (c: to 1): " + ", ".join(per), flush=True)
    line = "D %g n %g a %g (hp %g) ks %d: |S| %.4e; C1 hp od vs dd %.2e (%s); hp %.0f s, dd %.0f s" % (
        D, n, a, a_hp, ks_mob(D), float(max(abs(v) for v in Sdd)), c1, "PASS" if c1 <= 1e-24 else "FAIL", t_hp, t_dd)
    if mut:
        line += "  [control %s: must FAIL C1 by more than 1e-10: %s]" % (mut, "fired" if c1 > 1e-10 else "DID NOT FIRE")
        ok &= c1 > 1e-10
    else:
        ok &= c1 <= 1e-24
    print(line, flush=True)
    if not mut and want_gain:
        Sdd2 = dd_sum(D, 1.25 * a, mh, Dl)
        c3 = rel(Sdd2, Sdd)
        g_hp20 = gain(hp_sum, D, a, mh, Dl, 1e-20)
        g_hp10 = gain(hp_sum, D, a, mh, Dl, 1e-10)
        g_dd20 = gain(dd_sum, D, a, mh, Dl, 1e-20)
        c2 = (g_dd20 > 0 and min(g_hp20, g_hp10) > 0 and abs(g_hp20 - g_dd20) / g_dd20 <= 1e-3
              and max(g_hp20, g_hp10) / min(g_hp20, g_hp10) <= 2 and g_hp20 < 1e3)
        print("   C2 gain from m_h: hp %.4e (eps 1e-20), %.4e (eps 1e-10); dd %.4e (eps 1e-20): %s" % (
            g_hp20, g_hp10, g_dd20, "PASS" if c2 else "FAIL"), flush=True)
        print("   C3 dd at a and 1.25 a: %.2e (%s)" % (c3, "PASS" if c3 <= 1e-24 else "FAIL"), flush=True)
        ok &= c2 and c3 <= 1e-24
print("all checks: %s" % ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)

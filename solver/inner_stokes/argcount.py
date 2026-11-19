#!/usr/bin/env python3
"""Adaptive winding count of the zeros of the selection multiplier R_1(A, w/w_inf) (issue #995,
notes/DIRECTION_REVIEW.md 2.3): excited echoing solutions at D = infinity would be further zeros.

R_1 is a complex function of two real variables (A, wf = w/w_inf), so its zeros are counted by the degree of the map:
the winding number of R_1 along the boundary of a cell is the sum of the signs (+1 or -1, the sign of the Jacobian
determinant) of the zeros inside. A cell of winding 0 may still hold a pair of opposite sign.

The multiplier:
  M = 1: lateral_np.jump(1, w_inf wf, A sin theta) at xi1 = --r1, R_1 times xi1^(-A^2) (a positive factor, so no
         phase change). A point whose path reaches max |Y| > --ymax (1e2) has no value: the path then passes near a
         movable singularity of y+, and R_1 jumps where the parameters carry one across it (at xi1 = 0.3, 0.25 and
         0.2 on lines of A up to 5; none at 0.17). A point with |R_1| < 10 sigma (its rounding floor) has no value.
         The zero set does not depend on xi1 (J = 0 means y+ = y-), so each zero is also located at a second xi1.
  M = 3: nested.inner at xi1 = --r1 (F_3 tuned so that S_3 = 0, warm-started from the nearest point already computed;
         the base grid is computed in rings outward from the known M = 3 zero), r_1; a point whose inner solve leaves
         |r_3| >= 1e-8 has no value, nor one whose final path exceeds --ymax or whose |r_1| is under 10 sigma.
  Cells with a point of no value are reported unresolved.

The count:
  1. a base grid of cells over the box (offset by a small irrational amount so that no known zero lies on an edge);
  2. every cell edge refined by bisection until each phase step along it is below --dphi (0.5 rad), or until a piece
     is shorter than --hmin (then the edge is reported as unresolved, and so is every cell that uses it);
  3. each cell's winding W and boundary phase variation TV (the sum of |steps|); a cell is split into four while
     W != 0 and it is larger than --hzero (0.02), or W = 0 and it is larger than --hpair (0.01) and either TV >= pi
     or R at its centre differs from the bilinear interpolant of its corners by more than 25% (the resolution test:
     a pair of opposite sign well inside a cell leaves the winding at 0, so cells are refined until R is nearly
     linear on them, and a nearly linear map has at most one zero, of winding +1 or -1);
  4. each leaf cell with W != 0 is a zero of sign W; it is located by Newton's method on (Re R, Im R) with
     finite-difference Jacobians from the cell centre (accepted if it stays in the cell's double).
--plant=A0,wf0 multiplies R_1 by (A - A0) + 2i (wf - wf0), which adds a zero of sign +1 at (A0, wf0); the count must
find it. Values are cached (--cache, one JSON line per point), so a planted run reuses the plain run's points.

    python3 argcount.py --M=1 --A=0.5,6 --wf=0.2,3 --nA=22 --nwf=28 --out=DIR [--procs=2] [--plant=A0,wf0]
    python3 argcount.py --M=1 --recount=DIR/leaves_TAG.jsonl --r1=0.2 --r1b=0.25 --out=DIR --tag=TAG_r0.2
    python3 argcount.py --selftest      # a planted map with four zeros, two of them a close pair of opposite sign
    python3 argcount.py --selftest --mutate=nosplit   # no splitting of winding-0 cells: the pair is missed, FAIL
    python3 argcount.py --selftest --mutate=conj      # the map conjugated: every sign flips, FAIL
    python3 argcount.py --selftest --mutate=norecount # the recount keeps the hole: the pair stays unresolved, FAIL
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OFF = (0.0123456, 0.00654321)        # grid offset in (A, wf), so that the known zeros avoid the edges


# ------------------------------------------------------------------------------------------- the multipliers
def m1_value(A, wf, r1):
    """R_1 of lateral_np.jump(1, ...) along the same ray and arc (same solve_ivp calls, so the same bits), max |Y|
    on the path, and sigma, R_1's rounding floor. A large max |Y| means the path passes near a movable singularity
    of y+, whose side decides the branch, so R_1 jumps as the parameters move it across the path (#995). J is a
    difference of two O(|Y|) values, and its error is about 4e-16 |Y| at every rtol from 1e-12 to 3e-14, so
    sigma = 1e-15 |Y(xi1)| / |reference| (in R_1's units; at large A, J ~ xi1^(A^2) e^(-1/(2 xi1^2)) nears it)."""
    from scipy.integrate import solve_ivp
    from lateral_np import NP, Fharm, Kp, W0, fast_lin, series_data
    w, Fh, r0, phi0 = W0 * wf, Fharm(A), 0.1, 3 * np.pi / 8
    Y0, jt, err = series_data(Fh, w, 1, r0 * np.exp(1j * phi0), True)
    nm = NP(1, w, True)
    u = np.exp(1j * phi0)
    s1 = solve_ivp(lambda t, Z: nm.rhs(t * u, Z) * u, (r0, r1), Y0, method="DOP853", rtol=1e-12, atol=1e-15)
    s2 = solve_ivp(lambda t, Z: nm.rhs(r1 * np.exp(1j * t), Z) * 1j * r1 * np.exp(1j * t), (phi0, 0.0),
                   s1.y[:, -1], method="DOP853", rtol=1e-12, atol=1e-15)
    if not (s1.success and s2.success):
        raise RuntimeError(s1.message if not s1.success else s2.message)
    D = s2.y[:nm.no, -1]
    i = {mu: k for k, mu in enumerate(nm.odd)}
    s = 1j * w
    ref = 2j * np.pi * Kp(s) * fast_lin(s, r1) * (2 * Fh[1]) * r1 ** (A * A)
    R = (D[i[1]] - np.conj(D[i[-1]])) / ref
    sig = 1e-15 * np.abs(s2.y[:, -1]).max() / abs(ref)
    return complex(R), float(max(np.abs(s1.y).max(), np.abs(s2.y).max())), float(sig), float(err)


def f_M1(args):
    """[Re, Im, series error, max |Y|, sigma] (the guards are applied by Multiplier, so the cache keeps values)"""
    p, r1 = args
    try:
        v, ym, sig, err = m1_value(p[0], p[1], r1)
        return [v.real, v.imag, err, ym, sig]
    except Exception as ex:
        return [float("nan"), float("nan"), str(ex)[:80], float("nan"), float("nan")]


_PATH = {}


def _probe_inner_np():
    """inner_np with its solve_ivp wrapped to record max |Y| and the end |Y| of each jumps() path (the same call,
    so the same bits); jumps() resets the record, so after nested.inner it holds the final evaluation's path"""
    import inner_np as I
    if getattr(I, "_argcount_probe", False):
        return I
    ivp, jumps = I.solve_ivp, I.Inner.jumps

    def solve_ivp(*a, **k):
        sol = ivp(*a, **k)
        _PATH["ymax"] = max(_PATH.get("ymax", 0.0), float(np.abs(sol.y).max()))
        _PATH["yend"] = float(np.abs(sol.y[:, -1]).max())
        return sol

    def jumps_probe(self, *a, **k):
        _PATH.clear()
        return jumps(self, *a, **k)

    I.solve_ivp, I.Inner.jumps, I._argcount_probe = solve_ivp, jumps_probe, True
    return I


def f_M3(args):
    """[Re, Im, |r_3|, F3re, F3im, max |Y|, sigma]; sigma = 1e-15 |Y(xi1)| times residual_demixed's scale for r_1"""
    I = _probe_inner_np()
    import nested
    (A, wf), F3, r1 = args
    nan = float("nan")
    try:
        F3n, R, r3 = nested.inner(A, wf, complex(*F3), r1)
        w = I.W0 * wf
        Fh = I.Fh_from(np.array([A, wf, -2 * F3n.imag, 2 * F3n.real]), 3)
        sc = np.exp(1 / (2 * r1 * r1)) * r1 ** (-2 * I.F2mean(Fh)) / (2 * np.pi * abs(I.Kp(1j * w)))
        ym, sig = _PATH.get("ymax", nan), 1e-15 * _PATH.get("yend", nan) * sc
        if not float(r3) < 1e-8:                  # the inner solve did not tune S_3 to zero: no value
            return [nan, nan, float(r3), F3[0], F3[1], ym, sig]
        return [R.real, R.imag, float(r3), F3n.real, F3n.imag, ym, sig]
    except Exception as ex:
        return [nan, nan, str(ex)[:80], F3[0], F3[1], nan, nan]


class Multiplier:
    def __init__(self, M, cache, procs, plant=None, mutate=None, r1=0.2, ymax=1e2):
        self.M, self.procs, self.plant, self.mutate = M, procs, plant, mutate
        self.r1, self.ymax = r1, ymax
        self.vals, self.extra = {}, {}
        self.cache = cache
        self.pool = None
        if cache and os.path.exists(cache):
            for line in open(cache):
                r = json.loads(line)
                if r.get("r1") != r1 and (M == 1 or "r1" in r):
                    raise SystemExit(f"cache {cache} holds points at r1 {r.get('r1')}, not {r1}")
                self.store((r["A"], r["wf"]), r)
        self.F3seed = {}

    @staticmethod
    def key(p):
        return (round(p[0], 12), round(p[1], 12))

    def raw(self, pts):
        """the multiplier at the points not yet cached"""
        todo, seen = [], set()
        for p in pts:                               # each new point once
            k = self.key(p)
            if k not in self.vals and k not in seen:
                seen.add(k)
                todo.append(p)
        if not todo:
            return
        if self.M == 1:
            fn, args = f_M1, [(p, self.r1) for p in todo]
        else:
            fn = f_M3
            args = [(p, self.seed(p), self.r1) for p in todo]
        if self.procs > 1:
            if self.pool is None:
                import multiprocessing as mp_
                self.pool = mp_.get_context("spawn").Pool(self.procs)
            res = self.pool.map(fn, args, chunksize=1)
        else:
            res = [fn(a) for a in args]
        with open(self.cache, "a") as fh:
            for p, r in zip(todo, res):
                k = self.key(p)
                rec = dict(A=k[0], wf=k[1], re=r[0], im=r[1], info=r[2])
                if self.M == 1:
                    rec.update(r1=self.r1, ymax=r[3], sig=r[4])
                if self.M == 3:
                    rec.update(r1=self.r1, F3re=r[3], F3im=r[4], ymax=r[5], sig=r[6])
                self.store(k, rec)
                fh.write(json.dumps(rec) + "\n")

    def store(self, k, r):
        """a point's value, or none (NaN) when its path came near a singularity (max |Y| > ymax) or |R_1| is within
        10 sigma of its rounding floor (the phase is then unknown)"""
        v = complex(r["re"], r["im"])
        if not (r.get("ymax", 0.0) <= self.ymax and abs(v) >= 10 * r.get("sig", 0.0)):
            v = complex(float("nan"), float("nan"))
        self.vals[k] = v
        self.extra[k] = r

    def seed(self, p):
        """M = 3: F_3 from the nearest computed point (its tuned F_3), else the known zero's ratio"""
        best, bd = None, 1e300
        for k, r in self.extra.items():
            if "F3re" in r and math.isfinite(r["F3re"]) and isinstance(r["info"], float) and r["info"] < 1e-8:
                d = (k[0] - p[0]) ** 2 + (k[1] - p[1]) ** 2
                if d < bd:
                    best, bd = (r["F3re"], r["F3im"]), d
        if best is None:
            z = (p[0] / 2j) * (0.0532 + 0.0041j)
            best = (z.real, z.imag)
        return best

    def value(self, p):
        """the value without the guards"""
        r = self.extra[self.key(p)]
        return self.shape(p, complex(r["re"], r["im"]))

    def floor(self, p):
        """the rounding floor sigma of the value (M = 1), times the planted factor"""
        s = self.extra[self.key(p)].get("sig", 0.0)
        return float(s * (abs(self.shape(p, 1.0)) if self.plant is not None else 1.0))

    def __call__(self, p):
        return self.shape(p, self.vals[self.key(p)])

    def shape(self, p, v):
        if self.plant is not None:
            v = v * complex(p[0] - self.plant[0], 2 * (p[1] - self.plant[1]))
        if self.mutate == "conj":
            v = v.conjugate()
        return v

    def close(self):
        if self.pool is not None:
            self.pool.close()


# ------------------------------------------------------------------------------------------- the count
def wrap(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


class Counter:
    def __init__(self, F, dphi, hmin):
        self.F, self.dphi, self.hmin = F, dphi, hmin
        self.edges = {}

    def edge(self, p, q):
        """(phase increment from p to q, total variation, min |F|, resolved) with the pieces refined"""
        key = (Multiplier.key(p), Multiplier.key(q))
        rkey = (key[1], key[0])
        if key not in self.edges and rkey not in self.edges:
            self.prefetch([(p, q)])
        if key in self.edges:
            return self.edges[key]
        d, tv, mn, ok = self.edges[rkey]
        return (-d, tv, mn, ok)

    def prefetch(self, pairs):
        """refine several edges together, each by the same rule, so that each round's new points go to F in one call:
        a piece is bisected while its phase step is >= dphi and it is longer than hmin; a point with no value ends
        its edge as unresolved (bisecting would not give it one)"""
        act = {}
        for p, q in pairs:
            key = (Multiplier.key(p), Multiplier.key(q))
            rkey = (key[1], key[0])
            if not (key in self.edges or rkey in self.edges or key in act or rkey in act):
                act[key] = [[p, q], True]
        while act:
            self.F.raw([x for pts, _ in act.values() for x in pts])
            for key in list(act):
                pts, ok = act[key]
                vals = [self.F(x) for x in pts]
                new, bad = [pts[0]], False
                for i in range(len(pts) - 1):
                    a, b = vals[i], vals[i + 1]
                    step = wrap(math.atan2(b.imag, b.real) - math.atan2(a.imag, a.real)) if (
                        np.isfinite(a) and np.isfinite(b)) else float("nan")
                    L = math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
                    if not math.isfinite(step):
                        ok = False
                    elif not (abs(step) < self.dphi):
                        if L > self.hmin:
                            new.append(((pts[i][0] + pts[i + 1][0]) / 2, (pts[i][1] + pts[i + 1][1]) / 2))
                            bad = True
                        else:
                            ok = False
                    new.append(pts[i + 1])
                if bad:
                    act[key] = [new, ok]
                    continue
                steps = [wrap(math.atan2(b.imag, b.real) - math.atan2(a.imag, a.real))
                         for a, b in zip(vals[:-1], vals[1:])]
                fin = [abs(v) for v in vals if np.isfinite(v)]
                self.edges[key] = (sum(steps), sum(abs(s) for s in steps), min(fin) if fin else float("nan"),
                                   ok and all(map(np.isfinite, vals)))
                del act[key]

    @staticmethod
    def corners(a0, a1, w0, w1):
        return [(a0, w0), (a1, w0), (a1, w1), (a0, w1)]      # counter-clockwise in (A, wf)

    def cell_edges(self, a0, a1, w0, w1):
        c = self.corners(a0, a1, w0, w1)
        return [(c[i], c[(i + 1) % 4]) for i in range(4)]

    def cell(self, a0, a1, w0, w1):
        tot, tv, mn, ok = 0.0, 0.0, 1e300, True
        for p, q in self.cell_edges(a0, a1, w0, w1):
            d, v, m, o = self.edge(p, q)
            tot, tv, mn, ok = tot + d, tv + v, min(mn, m), ok and o
        return (round(tot / (2 * math.pi)) if ok else 0), tot / (2 * math.pi), tv, mn, ok


def locate(F, a0, a1, w0, w1):
    return locate_many(F, [(a0, a1, w0, w1)])[0]


def locate_many(F, cells):
    """Newton on (Re F, Im F) from each cell's centre with finite-difference Jacobians (steps 1e-2 of the cell, on the
    unguarded values: near the zero |R| falls below the rounding floor by design); None if it leaves the double.
    Near the floor the iterates jitter at sigma / |grad R|, so it ends after 20 steps if they do not settle. The
    cells advance in lockstep, so each step's points go to F in one call."""
    st = []
    for a0, a1, w0, w1 in cells:
        st.append(dict(c=(a0, a1, w0, w1), x=np.array([(a0 + a1) / 2, (w0 + w1) / 2]), h=((a1 - a0) * 1e-2,
                       (w1 - w0) * 1e-2), J=None, live=True, fail=False))
    pts = lambda s: [tuple(s["x"]), (s["x"][0] + s["h"][0], s["x"][1]), (s["x"][0], s["x"][1] + s["h"][1])]
    for _ in range(20):
        act = [s for s in st if s["live"]]
        if not act:
            break
        F.raw([p for s in act for p in pts(s)])
        for s in act:
            (a0, a1, w0, w1), (ha, hw) = s["c"], s["h"]
            f0, fa, fw = (F.value(p) for p in pts(s))
            if not all(map(np.isfinite, (f0, fa, fw))):
                s["live"], s["fail"] = False, True
                continue
            J = np.array([[(fa - f0).real / ha, (fw - f0).real / hw], [(fa - f0).imag / ha, (fw - f0).imag / hw]])
            try:
                dx = np.linalg.solve(J, -np.array([f0.real, f0.imag]))
            except np.linalg.LinAlgError:
                s["live"], s["fail"] = False, True
                continue
            s["x"], s["J"] = s["x"] + dx, J
            x = s["x"]
            if not (2 * a0 - a1 <= x[0] <= 2 * a1 - a0 and 2 * w0 - w1 <= x[1] <= 2 * w1 - w0):
                s["live"], s["fail"] = False, True
            elif np.hypot(dx[0] / (a1 - a0), dx[1] / (w1 - w0)) < 1e-9:
                s["live"] = False
    F.raw([tuple(s["x"]) for s in st if not s["fail"]])
    return [None if s["fail"] else dict(A=float(s["x"][0]), wf=float(s["x"][1]), absF=abs(F.value(tuple(s["x"]))),
                                        floor=F.floor(tuple(s["x"])), sign=int(np.sign(np.linalg.det(s["J"]))))
            for s in st]


def count(F, box, nA, nwf, dphi=0.5, hmin=1e-5, hzero=0.02, hpair=0.01, tvsplit=math.pi, eta=0.25, log=print,
          maxcells=20000, off=OFF):
    (A0, A1), (W0_, W1) = box
    As = np.linspace(A0 + off[0], A1 + off[0], nA + 1)
    Ws = np.linspace(W0_ + off[1], W1 + off[1], nwf + 1)
    base = [(a, w) for a in As for w in Ws]
    if getattr(F, "M", 1) == 3:
        # continuation: rings outward from the known M = 3 zero, so each batch is seeded by the previous ones
        base.sort(key=lambda p: (p[0] - 2.7265) ** 2 + 4 * (p[1] - 1.0156) ** 2)
        for i in range(0, len(base), max(1, 2 * F.procs)):
            F.raw(base[i:i + max(1, 2 * F.procs)])
    F.raw(base)
    C = Counter(F, dphi, hmin)
    queue = [(As[i], As[i + 1], Ws[k], Ws[k + 1], 0) for i in range(nA) for k in range(nwf)]
    leaves, splits = [], 0
    while queue:
        # one wave: every cell of the queue at once, so that each refinement round goes to F in one call
        wave, queue = queue, []
        C.prefetch([e for c in wave for e in C.cell_edges(*c[:4])])
        dec = []
        for a0, a1, w0, w1, depth in wave:
            W, wraw, tv, mn, ok = C.cell(a0, a1, w0, w1)
            size = max((a1 - a0), (w1 - w0))
            split = ok and ((W != 0 and size > hzero) or (W == 0 and tv >= tvsplit and size > hpair))
            # resolution: R at the centre against the bilinear interpolant of the corners; a pair of opposite sign
            # well inside a cell leaves the boundary's winding at 0 but shows here
            test = ok and not split and W == 0 and size > hpair and tvsplit < float("inf")
            dec.append(((a0, a1, w0, w1, depth), (W, wraw, tv, mn, ok), split, test))
        F.raw([((c[0] + c[1]) / 2, (c[2] + c[3]) / 2) for c, _, _, test in dec if test])
        for (a0, a1, w0, w1, depth), (W, wraw, tv, mn, ok), split, test in dec:
            if test:
                cen = ((a0 + a1) / 2, (w0 + w1) / 2)
                fb = sum(F(c) for c in C.corners(a0, a1, w0, w1)) / 4
                split = abs(F(cen) - fb) > eta * abs(fb)
            if split and splits < maxcells:
                am, wm = (a0 + a1) / 2, (w0 + w1) / 2
                queue += [(a0, am, w0, wm, depth + 1), (am, a1, w0, wm, depth + 1), (a0, am, wm, w1, depth + 1),
                          (am, a1, wm, w1, depth + 1)]
                splits += 1
                continue
            leaves.append(dict(a0=a0, a1=a1, w0=w0, w1=w1, W=W, wraw=wraw, tv=tv, minabs=mn, ok=ok, depth=depth))
    # the boundary of the whole box
    Wb = 0.0
    for e in [((As[0], Ws[0]), (As[-1], Ws[0])), ((As[-1], Ws[0]), (As[-1], Ws[-1])),
              ((As[-1], Ws[-1]), (As[0], Ws[-1])), ((As[0], Ws[-1]), (As[0], Ws[0]))]:
        # sum of the refined pieces along the side (the base grid's edges)
        p, q = e
        n = nA if p[1] == q[1] else nwf
        for i in range(n):
            pa = (p[0] + (q[0] - p[0]) * i / n, p[1] + (q[1] - p[1]) * i / n)
            pb = (p[0] + (q[0] - p[0]) * (i + 1) / n, p[1] + (q[1] - p[1]) * (i + 1) / n)
            Wb += C.edge(pa, pb)[0]
    zl = [L for L in leaves if L["W"] != 0]
    zs = locate_many(F, [(L["a0"], L["a1"], L["w0"], L["w1"]) for L in zl])
    zeros = [dict(cell=[L["a0"], L["a1"], L["w0"], L["w1"]], W=L["W"], located=z) for L, z in zip(zl, zs)]
    unresolved = [L for L in leaves if not L["ok"]]
    return dict(zeros=zeros, boundary_winding=Wb / (2 * math.pi), nleaves=len(leaves), splits=splits,
                unresolved=[[L["a0"], L["a1"], L["w0"], L["w1"]] for L in unresolved], npoints=len(F.vals),
                leaf_sum=sum(L["W"] for L in leaves), leaves=leaves)


def recount(F, leaves, dphi=0.5):
    """count each unresolved leaf of an earlier count as its own box (corners unchanged, no offset)"""
    boxes = [((L["a0"], L["a1"]), (L["w0"], L["w1"])) for L in leaves if not L["ok"]]
    res = dict(zeros=[], leaves=[], unresolved=[], nleaves=0, splits=0, leaf_sum=0, boundary_winding=0.0, boxes=[])
    for bx in boxes:
        r = count(F, bx, 1, 1, dphi=dphi, off=(0.0, 0.0))
        res["boxes"].append(dict(box=bx, boundary_winding=r["boundary_winding"], leaf_sum=r["leaf_sum"],
                                 unresolved=len(r["unresolved"])))
        for k in ("zeros", "leaves", "unresolved"):
            res[k] += r[k]
        for k in ("nleaves", "splits", "leaf_sum", "boundary_winding"):
            res[k] += r[k]
    res["npoints"] = len(F.vals)
    return res


# ------------------------------------------------------------------------------------------- selftest
class Analytic:
    """a planted map with known zeros: prod (dA + i s_k dwf) or its conjugate per zero, times a zero-free factor"""
    def __init__(self, zeros, mutate=None, hole=None):
        self.zs, self.vals, self.plant, self.mutate, self.hole = zeros, {}, None, mutate, hole

    def raw(self, pts):
        for p in pts:
            k = Multiplier.key(p)
            if k in self.vals:
                continue
            v = complex(1.0, 0.3) * np.exp(complex(0.4 * p[0], 1.7 * p[1]))
            for (a, w, sgn) in self.zs:
                d = complex(p[0] - a, 2 * (p[1] - w))
                v *= d if sgn > 0 else d.conjugate()
            if self.hole and math.hypot(p[0] - self.hole[0], 2 * (p[1] - self.hole[1])) < self.hole[2]:
                v = complex(float("nan"), float("nan"))     # no value here, as below a rounding floor
            self.vals[k] = v

    def __call__(self, p):
        v = self.vals[Multiplier.key(p)]
        return v.conjugate() if self.mutate == "conj" else v

    value = __call__

    def floor(self, p):
        return 0.0


def selftest(mutate=None):
    zs = [(2.54, 0.777, +1), (3.4, 0.36, -1), (3.45, 0.37, +1), (1.2, 2.2, +1)]
    F = Analytic(zs, mutate)
    res = count(F, ((0.5, 6.0), (0.2, 3.0)), 22, 28, tvsplit=float("inf") if mutate == "nosplit" else math.pi)
    found = sorted((round(z["located"]["A"], 3), round(z["located"]["wf"], 3), z["W"]) for z in res["zeros"]
                   if z["located"])
    want = sorted((round(a, 3), round(w, 3), s) for a, w, s in zs)
    ok = found == want and round(res["boundary_winding"]) == 2 and not res["unresolved"]
    print(f"selftest{' --mutate=' + mutate if mutate else ''}: planted {want}; found {found}; boundary winding "
          f"{res['boundary_winding']:.3f}; points {res['npoints']}: {'PASS' if ok else 'FAIL'}")
    # the radius ladder: a hole of no value over the close pair leaves its cells unresolved; a recount of them
    # with the hole gone (a second radius) must recover the pair, and the union must hold all four zeros
    Fh = Analytic(zs, mutate, hole=(3.425, 0.365, 0.15))
    r1 = count(Fh, ((0.5, 6.0), (0.2, 3.0)), 22, 28)
    r2 = recount(Analytic(zs, mutate, hole=None if mutate != "norecount" else (3.425, 0.365, 0.15)), r1["leaves"])
    got = sorted((round(z["located"]["A"], 3), round(z["located"]["wf"], 3), z["W"])
                 for z in r1["zeros"] + r2["zeros"] if z["located"])
    ok2 = bool(r1["unresolved"]) and not r2["unresolved"] and got == want and r1["leaf_sum"] + r2["leaf_sum"] == 2
    print(f"selftest recount: {len(r1['unresolved'])} cells unresolved by the hole, {len(r2['unresolved'])} after the "
          f"recount; found {got}: {'PASS' if ok2 else 'FAIL'}")
    return ok and ok2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--M", type=int, default=1)
    ap.add_argument("--A", default="0.5,6")
    ap.add_argument("--wf", default="0.2,3")
    ap.add_argument("--nA", type=int, default=22)
    ap.add_argument("--nwf", type=int, default=28)
    ap.add_argument("--dphi", type=float, default=0.5)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--plant", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--r1", type=float, default=None,
                    help="comparison radius xi1 (M = 1: 0.17; M = 3: 0.2, nested.py's)")
    ap.add_argument("--r1b", type=float, default=None,
                    help="second xi1 at which each zero is located again (M = 1: 0.2; M = 3: 0.17)")
    ap.add_argument("--ymax", type=float, default=1e2, help="no value where max |Y| on the path exceeds this")
    ap.add_argument("--recount", default=None,
                    help="leaves file of an earlier count: recount each of its unresolved cells as its own box")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mutate", default=None, choices=["conj", "nosplit", "norecount"])
    a = ap.parse_args()
    if a.selftest:
        return 0 if selftest(a.mutate) else 1
    box = (tuple(float(v) for v in a.A.split(",")), tuple(float(v) for v in a.wf.split(",")))
    plant = tuple(float(v) for v in a.plant.split(",")) if a.plant else None
    os.makedirs(a.out, exist_ok=True)
    tag = a.tag or f"M{a.M}_A{a.A}_wf{a.wf}" + (f"_plant{a.plant}" if plant else "")

    a.r1 = a.r1 or (0.17 if a.M == 1 else 0.2)
    a.r1b = a.r1b or (0.2 if a.M == 1 else 0.17)
    cache = a.cache or os.path.join(a.out, f"points_M{a.M}_r{a.r1}.jsonl")
    F = Multiplier(a.M, cache, a.procs, plant, r1=a.r1, ymax=a.ymax)
    if a.M == 3:
        # seed the continuation at the known M = 3 zero's tuned F_3
        F.extra[(-1.0, -1.0)] = dict(F3re=float(((2.7264758856 / 2j) * (0.0532 + 0.0041j)).real),
                                     F3im=float(((2.7264758856 / 2j) * (0.0532 + 0.0041j)).imag), info=0.0)
    t0 = time.time()
    if a.recount:
        # a rung of the radius ladder: the zero set does not depend on xi1, and each cell is counted at one xi1,
        # so a cell left unresolved at one radius (its rounding floor or a path near a singularity) is counted
        # again at another, with no grid offset (its corners are the earlier cell's)
        res = recount(F, list(map(json.loads, open(a.recount))), dphi=a.dphi)
        res["recount"] = a.recount
        box = f"{len(res['boxes'])} unresolved cells of {a.recount}"
    else:
        res = count(F, box, a.nA, a.nwf, dphi=a.dphi)
    F.close()
    if a.r1b:
        # the zero set does not depend on xi1: locate each zero again at r1b, from the same cell
        Fb = Multiplier(a.M, os.path.join(a.out, f"points_M{a.M}_r{a.r1b}.jsonl"), a.procs, plant, r1=a.r1b,
                        ymax=a.ymax)
        if a.M == 3:                              # seed F_3 from the first radius's tuned values
            for k, r in F.extra.items():
                Fb.extra.setdefault(k, dict(F3re=r.get("F3re", float("nan")), F3im=r.get("F3im", float("nan")),
                                            info=r.get("info")))
        for z, zb in zip(res["zeros"], locate_many(Fb, [z["cell"] for z in res["zeros"]])):
            z["located_r1b"] = zb
        Fb.close()
    res["sec"] = time.time() - t0
    res.update(M=a.M, box=box, plant=plant, dphi=a.dphi, nA=a.nA, nwf=a.nwf, r1=a.r1, r1b=a.r1b, ymax=a.ymax)
    leaves = res.pop("leaves")
    with open(os.path.join(a.out, f"count_{tag}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    with open(os.path.join(a.out, f"leaves_{tag}.jsonl"), "w") as fh:
        for L in leaves:
            fh.write(json.dumps(L) + "\n")
    where = f"recount of {box}" if a.recount else f"box A {box[0]}, wf {box[1]}"
    print(f"M = {a.M} at xi1 {a.r1}, {where}, plant {plant}: boundary winding {res['boundary_winding']:.3f}, "
          f"sum over leaves {res['leaf_sum']}, {res['nleaves']} leaves after {res['splits']} splits, "
          f"{res['npoints']} points, {len(res['unresolved'])} unresolved cells, {res['sec']:.0f} s")
    for z in res["zeros"]:
        loc = z["located"]
        print(f"  zero of sign {z['W']:+d} in cell A {z['cell'][0]:.4f}-{z['cell'][1]:.4f}, wf {z['cell'][2]:.4f}-"
              f"{z['cell'][3]:.4f}: " + (f"located at A {loc['A']:.6f}, wf {loc['wf']:.6f} (|R| {loc['absF']:.1e}, "
                                          f"Jacobian sign {loc['sign']:+d})" if loc else "not located")
              + (f"; at xi1 {a.r1b}: " + (f"A {z['located_r1b']['A']:.6f}, wf {z['located_r1b']['wf']:.6f}"
                                         if z.get("located_r1b") else "not located") if a.r1b else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

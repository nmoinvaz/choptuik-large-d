#!/usr/bin/env python3
"""Adaptive winding count of the zeros of the selection multiplier R_1(A, w/w_inf) (issue #995,
notes/DIRECTION_REVIEW.md 2.3): excited echoing solutions at D = infinity would be further zeros.

R_1 is a complex function of two real variables (A, wf = w/w_inf), so its zeros are counted by the degree of the map:
the winding number of R_1 along the boundary of a cell is the sum of the signs (+1 or -1, the sign of the Jacobian
determinant) of the zeros inside. A cell of winding 0 may still hold a pair of opposite sign.

The multiplier:
  M = 1: lateral_np.jump(1, w_inf wf, A sin theta), R_1 times xi1^(-A^2) (a positive factor, so no phase change);
  M = 3: nested.inner (F_3 tuned so that S_3 = 0, warm-started from the nearest point already computed), r_1.

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
    python3 argcount.py --selftest      # a planted map with four zeros, two of them a close pair of opposite sign
    python3 argcount.py --selftest --mutate=nosplit   # no splitting of winding-0 cells: the pair is missed, FAIL
    python3 argcount.py --selftest --mutate=conj      # the map conjugated: every sign flips, FAIL
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
def f_M1(p):
    from lateral_np import jump, Fharm, W0
    A, wf = p
    try:
        R, JD, JS, err = jump(1, W0 * wf, Fharm(A), r1=0.3)
        v = complex(R[1]) * 0.3 ** (-A * A)
        return [v.real, v.imag, err]
    except Exception as ex:
        return [float("nan"), float("nan"), str(ex)[:80]]


def f_M3(args):
    import nested
    (A, wf), F3 = args
    try:
        F3n, R, r3 = nested.inner(A, wf, complex(*F3), 0.2)
        return [R.real, R.imag, float(r3), F3n.real, F3n.imag]
    except Exception as ex:
        return [float("nan"), float("nan"), str(ex)[:80], F3[0], F3[1]]


class Multiplier:
    def __init__(self, M, cache, procs, plant=None, mutate=None):
        self.M, self.procs, self.plant, self.mutate = M, procs, plant, mutate
        self.vals, self.extra = {}, {}
        self.cache = cache
        self.pool = None
        if cache and os.path.exists(cache):
            for line in open(cache):
                r = json.loads(line)
                k = (r["A"], r["wf"])
                self.vals[k] = complex(r["re"], r["im"])
                self.extra[k] = r
        self.F3seed = {}

    @staticmethod
    def key(p):
        return (round(p[0], 12), round(p[1], 12))

    def raw(self, pts):
        """the multiplier at the points not yet cached"""
        todo = [p for p in pts if self.key(p) not in self.vals]
        if not todo:
            return
        if self.M == 1:
            fn, args = f_M1, todo
        else:
            fn = f_M3
            args = [(p, self.seed(p)) for p in todo]
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
                self.vals[k] = complex(r[0], r[1])
                rec = dict(A=k[0], wf=k[1], re=r[0], im=r[1], info=r[2])
                if self.M == 3:
                    rec.update(F3re=r[3], F3im=r[4])
                self.extra[k] = rec
                fh.write(json.dumps(rec) + "\n")

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

    def __call__(self, p):
        v = self.vals[self.key(p)]
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
        if key in self.edges:
            return self.edges[key]
        rkey = (key[1], key[0])
        if rkey in self.edges:
            d, tv, mn, ok = self.edges[rkey]
            return (-d, tv, mn, ok)
        pts = [p, q]
        ok = True
        while True:
            self.F.raw(pts)
            vals = [self.F(x) for x in pts]
            new, bad = [pts[0]], False
            for i in range(len(pts) - 1):
                a, b = vals[i], vals[i + 1]
                step = wrap(math.atan2(b.imag, b.real) - math.atan2(a.imag, a.real)) if (
                    np.isfinite(a) and np.isfinite(b)) else float("nan")
                L = math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
                if not (abs(step) < self.dphi):
                    if L > self.hmin:
                        new.append(((pts[i][0] + pts[i + 1][0]) / 2, (pts[i][1] + pts[i + 1][1]) / 2))
                        bad = True
                    else:
                        ok = False
                new.append(pts[i + 1])
            if not bad:
                break
            pts = new
        vals = [self.F(x) for x in pts]
        steps = [wrap(math.atan2(b.imag, b.real) - math.atan2(a.imag, a.real)) for a, b in zip(vals[:-1], vals[1:])]
        res = (sum(steps), sum(abs(s) for s in steps), min(abs(v) for v in vals), ok and all(map(np.isfinite, vals)))
        self.edges[key] = res
        return res

    def cell(self, a0, a1, w0, w1):
        corners = [(a0, w0), (a1, w0), (a1, w1), (a0, w1)]      # counter-clockwise in (A, wf)
        tot, tv, mn, ok = 0.0, 0.0, 1e300, True
        for i in range(4):
            d, v, m, o = self.edge(corners[i], corners[(i + 1) % 4])
            tot, tv, mn, ok = tot + d, tv + v, min(mn, m), ok and o
        return round(tot / (2 * math.pi)), tot / (2 * math.pi), tv, mn, ok


def locate(F, a0, a1, w0, w1):
    """Newton on (Re F, Im F) from the cell centre with finite-difference Jacobians; None if it leaves the double"""
    x = np.array([(a0 + a1) / 2, (w0 + w1) / 2])
    ha, hw = (a1 - a0) * 1e-3, (w1 - w0) * 1e-3
    for _ in range(12):
        pts = [tuple(x), (x[0] + ha, x[1]), (x[0], x[1] + hw)]
        F.raw(pts)
        f0, fa, fw = (F(p) for p in pts)
        J = np.array([[(fa - f0).real / ha, (fw - f0).real / hw], [(fa - f0).imag / ha, (fw - f0).imag / hw]])
        try:
            dx = np.linalg.solve(J, -np.array([f0.real, f0.imag]))
        except np.linalg.LinAlgError:
            return None
        x = x + dx
        if not (2 * a0 - a1 <= x[0] <= 2 * a1 - a0 and 2 * w0 - w1 <= x[1] <= 2 * w1 - w0):
            return None
        if np.hypot(dx[0] / (a1 - a0), dx[1] / (w1 - w0)) < 1e-9:
            break
    F.raw([tuple(x)])
    return dict(A=float(x[0]), wf=float(x[1]), absF=abs(F(tuple(x))), sign=int(np.sign(np.linalg.det(J))))


def count(F, box, nA, nwf, dphi=0.5, hmin=1e-5, hzero=0.02, hpair=0.01, tvsplit=math.pi, eta=0.25, log=print,
          maxcells=20000):
    (A0, A1), (W0_, W1) = box
    As = np.linspace(A0 + OFF[0], A1 + OFF[0], nA + 1)
    Ws = np.linspace(W0_ + OFF[1], W1 + OFF[1], nwf + 1)
    F.raw([(a, w) for a in As for w in Ws])
    C = Counter(F, dphi, hmin)
    queue = [(As[i], As[i + 1], Ws[k], Ws[k + 1], 0) for i in range(nA) for k in range(nwf)]
    leaves, splits = [], 0
    while queue:
        a0, a1, w0, w1, depth = queue.pop()
        W, wraw, tv, mn, ok = C.cell(a0, a1, w0, w1)
        size = max((a1 - a0), (w1 - w0))
        split = ok and ((W != 0 and size > hzero) or (W == 0 and tv >= tvsplit and size > hpair))
        if ok and not split and W == 0 and size > hpair and tvsplit < float("inf"):
            # resolution: R at the centre against the bilinear interpolant of the corners; a pair of opposite sign
            # well inside a cell leaves the boundary's winding at 0 but shows here
            cen = ((a0 + a1) / 2, (w0 + w1) / 2)
            F.raw([cen])
            fb = sum(F(c) for c in [(a0, w0), (a1, w0), (a1, w1), (a0, w1)]) / 4
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
    zeros = []
    for L in leaves:
        if L["W"] != 0:
            z = locate(F, L["a0"], L["a1"], L["w0"], L["w1"])
            zeros.append(dict(cell=[L["a0"], L["a1"], L["w0"], L["w1"]], W=L["W"], located=z))
    unresolved = [L for L in leaves if not L["ok"]]
    return dict(zeros=zeros, boundary_winding=Wb / (2 * math.pi), nleaves=len(leaves), splits=splits,
                unresolved=[[L["a0"], L["a1"], L["w0"], L["w1"]] for L in unresolved], npoints=len(F.vals),
                leaf_sum=sum(L["W"] for L in leaves), leaves=leaves)


# ------------------------------------------------------------------------------------------- selftest
class Analytic:
    """a planted map with known zeros: prod (dA + i s_k dwf) or its conjugate per zero, times a zero-free factor"""
    def __init__(self, zeros, mutate=None):
        self.zs, self.vals, self.plant, self.mutate = zeros, {}, None, mutate

    def raw(self, pts):
        for p in pts:
            k = Multiplier.key(p)
            if k in self.vals:
                continue
            v = complex(1.0, 0.3) * np.exp(complex(0.4 * p[0], 1.7 * p[1]))
            for (a, w, sgn) in self.zs:
                d = complex(p[0] - a, 2 * (p[1] - w))
                v *= d if sgn > 0 else d.conjugate()
            self.vals[k] = v

    def __call__(self, p):
        v = self.vals[Multiplier.key(p)]
        return v.conjugate() if self.mutate == "conj" else v


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
    return ok


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
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mutate", default=None, choices=["conj", "nosplit"])
    a = ap.parse_args()
    if a.selftest:
        return 0 if selftest(a.mutate) else 1
    box = (tuple(float(v) for v in a.A.split(",")), tuple(float(v) for v in a.wf.split(",")))
    plant = tuple(float(v) for v in a.plant.split(",")) if a.plant else None
    os.makedirs(a.out, exist_ok=True)
    tag = a.tag or f"M{a.M}_A{a.A}_wf{a.wf}" + (f"_plant{a.plant}" if plant else "")
    cache = a.cache or os.path.join(a.out, f"points_M{a.M}.jsonl")
    F = Multiplier(a.M, cache, a.procs, plant)
    if a.M == 3:
        # seed the continuation at the known M = 3 zero's tuned F_3
        F.extra[(-1.0, -1.0)] = dict(F3re=float(((2.7264758856 / 2j) * (0.0532 + 0.0041j)).real),
                                     F3im=float(((2.7264758856 / 2j) * (0.0532 + 0.0041j)).imag), info=0.0)
    t0 = time.time()
    res = count(F, box, a.nA, a.nwf, dphi=a.dphi)
    F.close()
    res["sec"] = time.time() - t0
    res.update(M=a.M, box=box, plant=plant, dphi=a.dphi, nA=a.nA, nwf=a.nwf)
    leaves = res.pop("leaves")
    with open(os.path.join(a.out, f"count_{tag}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    with open(os.path.join(a.out, f"leaves_{tag}.jsonl"), "w") as fh:
        for L in leaves:
            fh.write(json.dumps(L) + "\n")
    print(f"M = {a.M}, box A {box[0]}, wf {box[1]}, plant {plant}: boundary winding {res['boundary_winding']:.3f}, "
          f"sum over leaves {res['leaf_sum']}, {res['nleaves']} leaves after {res['splits']} splits, "
          f"{res['npoints']} points, {len(res['unresolved'])} unresolved cells, {res['sec']:.0f} s")
    for z in res["zeros"]:
        loc = z["located"]
        print(f"  zero of sign {z['W']:+d} in cell A {z['cell'][0]:.4f}-{z['cell'][1]:.4f}, wf {z['cell'][2]:.4f}-"
              f"{z['cell'][3]:.4f}: " + (f"located at A {loc['A']:.6f}, wf {loc['wf']:.6f} (|R| {loc['absF']:.1e}, "
                                          f"Jacobian sign {loc['sign']:+d})" if loc else "not located"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Algebraic approximants of Delta(D) fitted to points: sum_k A_k(eps) Delta^k = 0, eps = 1 / (D - 2).

Degree 1 in Delta is a rational function [L/M] (A_1 of degree 0: a polynomial), degree 2 the quadratic approximant
P Delta^2 + Q Delta + R = 0 with degrees (p, q, r) (notes/QUADRATIC_FORMULA.md). One code path fits every class: a
linearized weighted least squares (the smallest eigenvector of the normal matrix, reweighted by 1 / |dE/dDelta|),
then Gauss-Newton on the misfit in Delta itself, in ball arithmetic (python-flint, 1024 bits by default).

    cd solver/quadform && python3 algfit.py selftest [--mutate=coef]
    cd solver/quadform && python3 algfit.py A --data=T_cut|T_pow|T_pole|T_pole7|real [--mutate=root|shift] [--out=F]
    cd solver/quadform && python3 algfit.py B --data=... [--out=F]
    cd solver/quadform && python3 algfit.py C --data=... [--out=F]

A, B and C are the three experiments of notes/QUADRATIC_FORMULA.md 2, whose criteria were fixed before the real points
were fitted. The plants go through the same path on the real D grid with noise N(0, sigma). Exit 1 when a selftest or a
mutation's expected verdict is not met.

Methods: quadratic approximants, Shafer 1974, SIAM J. Numer. Anal. 11, 447 (as cited in
solver/inner_amp/pade_chord_arc.py); the reweighted linear least squares for rational fits, Sanathanan and Koerner
1963, IEEE Trans. Autom. Control 8, 56 (unverified).
"""
import csv
import json
import math
import os
import random
import sys
import time

import mpmath as mp
from flint import arb, arb_mat, ctx

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
ARGS = dict(a.lstrip("-").split("=", 1) if "=" in a else (a.lstrip("-"), "1") for a in sys.argv[2:])
PREC = int(ARGS.get("prec", 1024))
ctx.prec = PREC
mp.mp.prec = PREC
DMAX = 245.64                      # the comparison frontier (README, "The family to D 245.64")
SIG_FLOOR = 1e-15
SEED = 20261119
DINF = "1.6438937"                 # the plants' limit, a round number near the real one


# ---------------------------------------------------------------- points

def dropped():
    """the continuation points D >= 10 that the production cleaning drops, as hp/formula_cut_fit.py takes them"""
    cache = os.path.join(HERE, "data", "cleaned_drop.json")
    if os.path.exists(cache):
        return set(json.load(open(cache))["dropped"])
    sys.path.insert(0, os.path.join(ROOT, "hp"))
    import dd_fits_mp as F
    mp.mp.dps = 60
    pts, _ = F.load_points()
    Ds = sorted(d for d in pts if d >= 10)
    drop = sorted(set(round(d, 6) for d in Ds) - set(round(d, 6) for d in F.clean(pts, Ds, verbose=False)))
    mp.mp.prec = PREC
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump(dict(dropped=drop, produced_by="solver/quadform/algfit.py dropped(): hp/dd_fits_mp.clean on D >= 10"),
              open(cache, "w"), indent=1)
    return set(drop)


def load_points():
    """[(D, Delta arb, sigma, kind)] sorted by D: the rows of hp/data/delta_all.csv named in the note's section 2"""
    drop = dropped()
    best = {}
    for r in csv.DictReader(open(os.path.join(ROOT, "hp", "data", "delta_all.csv"))):
        D = float(r["D"])
        k, ek = r["kind"], r["err_kind"]
        if D > DMAX + 1e-9:
            continue
        ok = (k == "dd" and ek in ("check", "dM", "dM neighbours")) or k in ("qd", "pinned") or \
             (k == "chain_fit" and ek.startswith("fits d"))
        if not ok or (k == "dd" and ek == "check" and round(D, 6) in drop):
            continue
        # each part is the repr of a double: parse it to that double, then add exactly
        v = sum((arb(float(r["Delta_q%d" % i])) for i in range(4) if r["Delta_q%d" % i]), arb(0))
        s = max(float(r["err"]), SIG_FLOOR)
        key = round(D, 9)
        if key not in best or s < best[key][2]:
            best[key] = (D, v, s, k)
    out = [best[k] for k in sorted(best)]
    if "no_amend" in ARGS:
        return out
    # amendment 1 (note, section 3): a qd row within 1e-4 of another row's D carries a D rounded to 4 decimals (its
    # state file's name) with the Delta of the unrounded D, up to 8.4e-8 off at its quoted 2.4e-23
    keep = []
    for i, p in enumerate(out):
        near = [q for q in out[max(0, i - 2):i + 3] if q is not p and abs(q[0] - p[0]) < 1e-4]
        if not (p[3] == "qd" and near):
            keep.append(p)
    return keep


# ---------------------------------------------------------------- plants

def _cheb(x, k):
    T = [arb(1), x]
    for _ in range(k - 1):
        T.append(2 * x * T[-1] - T[-2])
    return T[:k + 1]


def plant_fn(name):
    """the plant as a function D (float) -> arb"""
    if name == "T_cut":
        f = json.load(open(os.path.join(ROOT, "hp", "data", "formula_cut.json")))
        c, ns = [arb(v) for v in f["c"]], arb(f["n_s"])

        def fn(D):
            n = arb(D) - 2
            u = (2 * n - (1 + ns)) / (1 - ns)
            z = u - (u * u - 1).sqrt()
            return sum((ci * t for ci, t in zip(c, _cheb(2 * z - 1, len(c) - 1))), arb(0))
        return fn
    if name == "T_pow":
        cc = arb("10.375") / 3              # the slope in eps: ((1 - e)/(1 + 9 e))^0.3 = 1 - 3 e + ...

        def fn(D):
            D = arb(D)
            return arb(DINF) + cc * (1 - ((D - 3) / (D + 7)) ** arb("0.3"))
        return fn
    if name == "T_pole":
        f = json.load(open(os.path.join(ROOT, "solver", "data", "formula_coefficients.json")))
        a, b = [arb(v) for v in f["a"]], [arb(1)] + [arb(v) for v in f["b"]]

        def fn(D):
            e = 1 / (arb(D) - 2)
            return sum((ci * e ** k for k, ci in enumerate(a)), arb(0)) / sum((ci * e ** k for k, ci in enumerate(b)), arb(0))
        return fn
    if name == "T_pole7":
        poles = [-2, -4, -6, -8, -10, -13, -17]
        res = ["1", "1.5", "2", "2", "1.875", "1.25", "0.75"]

        def fn(D):
            e = 1 / (arb(D) - 2)
            return arb(DINF) + sum((arb(r) * e / (1 - e * (arb(p) - 2)) for p, r in zip(poles, res)), arb(0))
        return fn
    raise SystemExit(f"unknown plant {name}")


def dataset(name):
    """the real points, or a plant on their D grid with noise N(0, sigma)"""
    pts = load_points()
    if name == "real":
        return pts
    fn, rng = plant_fn(name), random.Random(SEED)
    return [(D, fn(D) + arb(s * rng.gauss(0, 1)), s, k) for D, _, s, k in pts]


# ---------------------------------------------------------------- the model

class Model:
    """sum_k A_k(eps) Delta^k = 0; A_k = sum_j c[off_k + j] T_j(u), u the fit interval's eps mapped to [-1, 1]"""

    def __init__(self, degs, a, b):
        self.degs, self.a, self.b = list(degs), arb(a), arb(b)
        self.off = [sum(m + 1 for m in degs[:k]) for k in range(len(degs))]
        self.n = sum(m + 1 for m in degs)
        self.c = None

    def trow(self, D):
        e = 1 / (arb(D) - 2) if D != "inf" else arb(0)
        return _cheb((2 * e - (self.a + self.b)) / (self.b - self.a), max(max(self.degs), 1))

    def brow(self, T, y):
        """the row of basis functions T_j(u) y^k"""
        out, yk = [], arb(1)
        for m in self.degs:
            out.extend(T[j] * yk for j in range(m + 1))
            yk = yk * y
        return out

    def A(self, T):
        return [sum((self.c[o + j] * T[j] for j in range(m + 1)), arb(0)) for o, m in zip(self.off, self.degs)]

    def E(self, Ak, y):
        v = arb(0)
        for a_ in reversed(Ak):
            v = v * y + a_
        return v

    def dE(self, Ak, y):
        v = arb(0)
        for k in range(len(Ak) - 1, 0, -1):
            v = v * y + k * Ak[k]
        return v

    def root(self, T, y0, other=False):
        """the root of E(eps, .) reached by Newton from y0; other: the quadratic's second root. (root, converged)"""
        Ak, y, ok = self.A(T), arb(y0), False
        for _ in range(60):
            d = self.dE(Ak, y)
            if float(abs(d)) == 0:
                break
            st = self.E(Ak, y) / d
            y = arb((y - st).mid())
            if float(abs(st)) < 2.0 ** (-PREC + 80) * max(1.0, float(abs(y))):
                ok = True
                break
        if other and len(Ak) == 3:
            y = -Ak[1] / Ak[2] - y
        return y, ok

    def value(self, D, y0, other=False):
        return self.root(self.trow(D), y0, other)

    # the quadratic's discriminant, its zeros and the zeros of P, as D
    def _mono(self, k):
        """A_k as monomial coefficients in u (mpmath)"""
        m = self.degs[k]
        cs = [mp.mpf(self.c[self.off[k] + j].mid().str(PREC // 3, radius=False)) for j in range(m + 1)]
        polys = [[mp.mpf(1)], [mp.mpf(0), mp.mpf(1)]]
        for _ in range(m - 1):
            p1, p0 = polys[-1], polys[-2]
            nxt = [mp.mpf(0)] + [2 * v for v in p1]
            for i, v in enumerate(p0):
                nxt[i] -= v
            polys.append(nxt)
        out = [mp.mpf(0)] * (m + 1)
        for j in range(m + 1):
            for i, v in enumerate(polys[j]):
                out[i] += cs[j] * v
        return out

    def _roots_D(self, mono):
        """the zeros of a polynomial in u (ascending monomials), as complex D"""
        while len(mono) > 1 and abs(mono[-1]) < mp.mpf(2) ** (-PREC // 2) * max(abs(v) for v in mono):
            mono = mono[:-1]
        if len(mono) < 2:
            return []
        rts = mp.polyroots(list(reversed(mono)), maxsteps=2000, extraprec=4 * PREC)
        a, b = mp.mpf(self.a.mid().str(PREC // 3, radius=False)), mp.mpf(self.b.mid().str(PREC // 3, radius=False))
        out = []
        for u in rts:
            e = ((b - a) * u + (a + b)) / 2
            out.append(complex(2 + 1 / e) if abs(e) > mp.mpf(10) ** -200 else complex("inf"))
        return out

    def branch_points(self):
        R, Q, P = self._mono(0), self._mono(1), self._mono(2)
        n = max(2 * len(Q) - 1, len(P) + len(R) - 1)
        d = [mp.mpf(0)] * n
        for i, x in enumerate(Q):
            for j, y in enumerate(Q):
                d[i + j] += x * y
        for i, x in enumerate(P):
            for j, y in enumerate(R):
                d[i + j] -= 4 * x * y
        return self._roots_D(d)

    def zeros_of(self, k):
        return self._roots_D(self._mono(k))


def _min_eigvec(N):
    """the eigenvector of the smallest eigenvalue of a symmetric positive matrix, by inverse iteration"""
    n = N.nrows()
    tr = sum(float(N[i, i]) for i in range(n))
    Nm = arb_mat([[arb(N[i, j].mid()) + (arb(tr) * arb(2) ** (-PREC + 120) if i == j else 0) for j in range(n)]
                  for i in range(n)])
    x = arb_mat([[arb(1 + 0.37 * math.sin(1 + 2.1 * i))] for i in range(n)])
    prev = None
    for it in range(80):
        x = Nm.solve(x)
        nrm = max(float(abs(x[i, 0])) for i in range(n))
        x = arb_mat([[arb((x[i, 0] / nrm).mid())] for i in range(n)])
        cur = [float(x[i, 0]) for i in range(n)]
        if prev is not None and max(abs(abs(u) - abs(v)) for u, v in zip(cur, prev)) < 1e-14 and it >= 3:
            break
        prev = cur
    nrm = sum((x[i, 0] * x[i, 0] for i in range(n)), arb(0)).sqrt()
    return [arb((x[i, 0] / nrm).mid()) for i in range(n)]


def fit(pts, degs, uniform=False, verbose=False):
    """fit the form `degs` (degrees of A_0..A_d) to pts [(D, y, sigma, kind)]; returns the Model with .info"""
    es = [1 / (arb(D) - 2) for D, _, _, _ in pts]
    M = Model(degs, min(es, key=float), max(es, key=float))
    Ts = [M.trow(D) for D, _, _, _ in pts]
    ys = [y for _, y, _, _ in pts]
    sg = [arb(1) if uniform else arb(s) for _, _, s, _ in pts]
    B = [M.brow(T, y) for T, y in zip(Ts, ys)]
    n, npts = M.n, len(pts)
    # 1. the linearized least squares, reweighted by 1 / |dE/dDelta| at the points
    w = [1 / s for s in sg]
    for it in range(12):
        Am = arb_mat([[v * w[i] for v in B[i]] for i in range(npts)])
        c = _min_eigvec(Am.transpose() * Am)
        if M.c is not None:
            ch = min(max(float(abs(u - v)) for u, v in zip(c, M.c)), max(float(abs(u + v)) for u, v in zip(c, M.c)))
        else:
            ch = 1.0
        M.c = c
        if ch < 1e-25:
            break
        w = [1 / (s * abs(M.dE(M.A(T), y))) for s, T, y in zip(sg, Ts, ys)]

    # 2. Gauss-Newton with damping on the misfit in Delta
    def state(cvec):
        M.c = cvec
        r, rows, ok = [], [], True
        for i in range(npts):
            ym, conv = M.root(Ts[i], ys[i])
            ok = ok and conv and float(abs(ym - ys[i])) < 0.05
            dEi = M.dE(M.A(Ts[i]), ym)
            r.append((ym - ys[i]) / sg[i])
            rows.append([-v / (dEi * sg[i]) for v in M.brow(Ts[i], ym)])
        return r, rows, ok, sum(float(v) ** 2 for v in r)

    r, rows, ok, chi = state(M.c)
    lam, hist = 0.0, [chi]
    for it in range(40):
        if not ok:
            break
        J = arb_mat(rows)
        Jt = J.transpose()
        H, g = Jt * J, Jt * arb_mat([[v] for v in r])
        tau = sum(float(H[i, i]) for i in range(n)) / n
        Hm = arb_mat([[arb(H[i, j].mid()) + tau * M.c[i] * M.c[j] + (lam * H[i, i] if i == j else 0)
                       for j in range(n)] for i in range(n)])
        try:
            dl = Hm.solve(arb_mat([[-arb(g[i, 0].mid())] for i in range(n)]))
        except ZeroDivisionError:
            break
        c_old = M.c
        c_new = [arb((c_old[i] + dl[i, 0]).mid()) for i in range(n)]
        nrm = sum((v * v for v in c_new), arb(0)).sqrt()
        c_new = [arb((v / nrm).mid()) for v in c_new]
        r2, rows2, ok2, chi2 = state(c_new)
        if ok2 and chi2 <= chi:
            done = chi - chi2 <= 1e-10 * chi
            r, rows, chi = r2, rows2, chi2
            lam = lam / 10 if lam > 1e-12 else 0.0
            hist.append(chi)
            if done:
                break
        else:
            M.c = c_old
            lam = 1e-6 if lam == 0 else lam * 100
            if lam > 1e12:
                break
    miss = [float(v * s) for v, s in zip(r, sg)]
    rsig = [m / s for m, (_, _, s, _) in zip(miss, pts)]
    M.info = dict(degs=list(degs), n_const=n - 1, npts=npts, ok=bool(ok), ss=chi, max_miss=max(abs(v) for v in miss),
                  max_over_sigma=max(abs(v) for v in rsig), chi2dof=sum(v * v for v in rsig) / max(1, npts - (n - 1)),
                  gn_steps=len(hist) - 1)
    if verbose:
        print(f"    {form_name(degs):>12s} N {n - 1:2d}: ss {chi:.3e} max miss {M.info['max_miss']:.2e} "
              f"max/sigma {M.info['max_over_sigma']:.3g} ok {ok}", flush=True)
    return M


def form_name(degs):
    if len(degs) == 2:
        return f"[{degs[0]}/{degs[1]}]"
    return "(" + ",".join(str(m) for m in reversed(degs)) + ")"     # (p, q, r)


def held_out(M, pts, mutate=None):
    """the largest |model - point| over pts, with the D where it is, and whether every root was found"""
    worst, at, ok = 0.0, None, True
    for D, y, _, _ in pts:
        Dm = D + 0.01 if mutate == "shift" else D
        v, conv = M.value(Dm, y, other=(mutate == "root"))
        e = float(abs(v - y))
        ok = ok and conv
        if not (e <= worst):
            worst, at = e, D
    return worst, at, ok


def real_axis_defect(M, Dlo):
    """True when a quadratic's discriminant has a zero at a real D >= Dlo (or the form is not a quadratic: False)"""
    if len(M.degs) != 3:
        return False
    return any(abs(z.imag) < 1e-9 * max(1.0, abs(z.real)) and z.real >= Dlo for z in M.branch_points()
               if z == z and abs(z) != float("inf"))


def rational_forms(N, Mmax=4):
    return [(N - 1 - m, m) for m in range(1, Mmax + 1) if N - 1 - m >= m]


def quadratic_forms(N, spread):
    """(r, q, p) for every p + q + r + 2 = N with the degrees within `spread` of each other"""
    out, s = [], N - 2
    for p in range(s + 1):
        for q in range(s + 1 - p):
            r = s - p - q
            if max(p, q, r) - min(p, q, r) <= spread:
                out.append((r, q, p))
    return out


# ---------------------------------------------------------------- experiments

def exp_A(name, mutate):
    pts = dataset(name)
    win = [p for p in pts if 4 - 1e-9 <= p[0] <= 30.2 + 1e-9]
    h1 = [p for p in pts if 30.2 + 1e-9 < p[0] <= 67.77 + 1e-9]
    h2 = [p for p in pts if p[0] > 67.77 + 1e-9]
    low = [p for p in pts if p[0] < 4 - 1e-9]
    print(f"A, {name}: {len(win)} points in 4 <= D <= 30.2; held out {len(h1)} to 67.77, {len(h2)} to {DMAX}; "
          f"{len(low)} below 4", flush=True)
    out = dict(experiment="A", data=name, mutate=mutate or "", window=[4, 30.2], n_window=len(win),
               n_held=len(h1) + len(h2), rows=[], verdicts={})
    for wt in ("sigma", "uniform"):
        wins = 0
        for N in (9, 11, 13):
            entry = {}
            for cls, forms in (("rational", rational_forms(N)), ("quadratic", quadratic_forms(N, 3))):
                cands = []
                for degs in forms:
                    t0 = time.time()
                    M = fit(win, degs, uniform=(wt == "uniform"))
                    s1, a1, k1 = held_out(M, h1, mutate)
                    s2, a2, k2 = held_out(M, h2, mutate)
                    vinf, _ = M.value("inf", arb(DINF))
                    row = dict(weights=wt, N=N, cls=cls, form=form_name(degs), ss=M.info["ss"], ok=M.info["ok"],
                               in_max=M.info["max_miss"], held1=s1, held2=s2, score=max(s1, s2),
                               at=a1 if s1 >= s2 else a2, roots_found=bool(k1 and k2),
                               defect=real_axis_defect(M, 30.2), dinf=float(vinf), secs=round(time.time() - t0, 1))
                    out["rows"].append(row)
                    cands.append(row)
                    print(f"  {wt:7s} N {N:2d} {cls:9s} {row['form']:>10s} ss {row['ss']:.3e} in {row['in_max']:.1e} "
                          f"held {s1:.1e} {s2:.1e} defect {row['defect']} ({row['secs']} s)", flush=True)
                good = [c for c in cands if c["ok"]]
                entry[cls] = min(good or cands, key=lambda c: c["ss"])
            q, r = entry["quadratic"], entry["rational"]
            win_here = q["score"] <= r["score"] / 10
            wins += win_here
            out["verdicts"][f"{wt}:N{N}"] = dict(rational=r["form"], rational_score=r["score"], quadratic=q["form"],
                                                 quadratic_score=q["score"], ratio=r["score"] / q["score"],
                                                 quadratic_wins=bool(win_here))
            print(f"  => {wt} N {N}: rational {r['form']} {r['score']:.2e}, quadratic {q['form']} {q['score']:.2e}, "
                  f"ratio {r['score'] / q['score']:.3g}: {'quadratic wins' if win_here else 'no win'}", flush=True)
        out["verdicts"][f"{wt}:pass"] = bool(wins >= 2)
    out["pass_A"] = bool(out["verdicts"]["sigma:pass"] and out["verdicts"]["uniform:pass"])
    print(f"A, {name}{' mutate ' + mutate if mutate else ''}: {'PASS' if out['pass_A'] else 'no pass'}")
    if mutate:
        # the control: the entered quadratics' scores must be broken (root: above 1e-3; shift: above 1e-5)
        lim = 1e-3 if mutate == "root" else 1e-5
        sc = [v["quadratic_score"] for k, v in out["verdicts"].items() if isinstance(v, dict)]
        fired = all(s > lim for s in sc)
        out["control_fired"] = bool(fired)
        print(f"control {mutate}: quadratic scores {min(sc):.1e} to {max(sc):.1e}, all above {lim:g}: "
              f"{'fired' if fired else 'DID NOT FIRE'}")
    return out, (1 if (mutate and not out.get("control_fired")) else 0)


def near_diag_forms(N):
    return quadratic_forms(N, 1)


def exp_B(name):
    pts = dataset(name)
    print(f"B, {name}: {len(pts)} points, D {pts[0][0]:.4f} to {pts[-1][0]:.2f}", flush=True)
    out = dict(experiment="B", data=name, n=len(pts), rows=[])
    Ns = [int(v) for v in ARGS.get("Ns", "11,13,15,17,19,21,23,25").split(",")]
    for N in Ns:
        for degs in near_diag_forms(N):
            t0 = time.time()
            M = fit(pts, degs)
            bps = [z for z in M.branch_points() if z == z and abs(z) != float("inf")]
            real_bp = sorted(z.real for z in bps if abs(z.imag) < 1e-9 * max(1.0, abs(z.real)))
            defect = any(v >= pts[0][0] for v in real_bp)
            up = [v for v in real_bp if v < pts[0][0] and v > 2]
            neg = [v for v in real_bp if v < 2]
            vinf, _ = M.value("inf", arb(DINF))
            at_up = None
            if up:                                  # Delta at the upper branch point, -Q / (2 P) there (post hoc)
                Ak = M.A(M.trow(max(up)))
                at_up = float(-Ak[1] / (2 * Ak[2]))
            row = dict(N=N, form=form_name(degs), chi2dof=M.info["chi2dof"], max_over_sigma=M.info["max_over_sigma"],
                       delta_at_upper=at_up,
                       ok=M.info["ok"], defect=bool(defect), accepted=bool(M.info["ok"] and not defect and M.info["chi2dof"] <= 4),
                       bp_upper=max(up) if up else None, bp_neg=max(neg) if neg else None, real_bp=real_bp,
                       complex_bp=[[z.real, z.imag] for z in bps if z.imag > 1e-9 * max(1.0, abs(z.real))],
                       P_zeros=[[z.real, z.imag] for z in M.zeros_of(2) if z == z and abs(z) != float("inf")],
                       dinf=vinf.mid().str(14, radius=False), secs=round(time.time() - t0, 1))
            out["rows"].append(row)
            print(f"  N {N:2d} {row['form']:>10s} chi2/dof {row['chi2dof']:.3g} max/sigma {row['max_over_sigma']:.3g} "
                  f"defect {defect} bp_upper {row['bp_upper']} bp_neg {row['bp_neg']} dinf {row['dinf']} "
                  f"({row['secs']} s)", flush=True)
    acc = [r for r in out["rows"] if r["accepted"]]
    hit = [r for r in acc if r["bp_upper"] is not None and abs(r["bp_upper"] - 3) <= 0.05 and
           any(abs(v + 7) <= 1 for v in r["real_bp"])]
    out["accepted"], out["hit_cut"] = len(acc), len(hit)
    out["hit_fraction"] = len(hit) / len(acc) if acc else None

    def iqr(vs):
        vs = sorted(vs)
        if len(vs) < 4:
            return None
        return vs[(3 * len(vs)) // 4] - vs[len(vs) // 4]
    out["upper"] = dict(values=[r["bp_upper"] for r in acc if r["bp_upper"] is not None])
    out["upper"]["iqr"] = iqr(out["upper"]["values"])
    out["neg"] = dict(values=[r["bp_neg"] for r in acc if r["bp_neg"] is not None])
    out["neg"]["iqr"] = iqr(out["neg"]["values"])
    print(f"B, {name}: {len(acc)} accepted of {len(out['rows'])}; with branch points within 0.05 of 3 and 1 of -7: "
          f"{len(hit)}; upper {out['upper']}, negative {out['neg']}")
    return out, 0


def exp_C(name):
    pts = [p for p in dataset(name) if p[0] <= 67.77 + 1e-9]
    print(f"C, {name}: {len(pts)} points, D {pts[0][0]:.4f} to {pts[-1][0]:.2f}; the bar is 3.47 sigma", flush=True)
    out = dict(experiment="C", data=name, n=len(pts), rows=[], first={})
    Ns = [int(v) for v in ARGS.get("Ns", "12,14,16,18,20,22,24,26,28").split(",")]
    for cls in ("quadratic", "rational"):
        for N in Ns:
            forms = near_diag_forms(N) if cls == "quadratic" else [f for f in rational_forms(N, 12) if f[1] in (2, 4, 6)]
            best = None
            for degs in forms:
                t0 = time.time()
                M = fit(pts, degs)
                defect = real_axis_defect(M, pts[0][0])
                row = dict(cls=cls, N=N, form=form_name(degs), chi2dof=M.info["chi2dof"], ok=M.info["ok"],
                           max_over_sigma=M.info["max_over_sigma"], defect=bool(defect), secs=round(time.time() - t0, 1))
                out["rows"].append(row)
                print(f"  {cls:9s} N {N:2d} {row['form']:>10s} chi2/dof {row['chi2dof']:.3g} max/sigma "
                      f"{row['max_over_sigma']:.3g} defect {defect} ({row['secs']} s)", flush=True)
                if row["ok"] and not defect and row["max_over_sigma"] <= 3.47 and best is None:
                    best = row
            if best is not None and cls not in out["first"]:
                out["first"][cls] = dict(N=N, form=best["form"], max_over_sigma=best["max_over_sigma"])
                if "all" not in ARGS:
                    break
    q = out["first"].get("quadratic")
    out["pass_C"] = bool(q and q["N"] <= 24)
    print(f"C, {name}: first within 3.47 sigma: {out['first']}; {'PASS' if out['pass_C'] else 'no pass'}")
    return out, 0


def selftest():
    """an in-class quadratic plant without noise is recovered to 1e-20; --mutate=coef must fail"""
    rng = random.Random(1)
    Ds = [4 + 0.25 * i for i in range(105)]
    es = [1 / (arb(D) - 2) for D in Ds]
    T0 = Model((2, 2, 2), min(es, key=float), max(es, key=float))
    # R, Q, P in the Chebyshev basis: the discriminant stays positive, the roots are near 2 to 5 and beyond 20
    T0.c = [arb(v) for v in ("2.5", "0.9", "0.1", "-1", "0.1", "0.03", "0.03", "0.01", "0.002")]
    pts = [(D, T0.value(D, arb(3))[0], 1.0, "plant") for D in Ds]
    held = [(D, T0.value(D, arb(2))[0], 1.0, "plant") for D in (40.0, 80.0, 160.0, 245.0)]
    if ARGS.get("mutate") == "coef":
        T0.c[4] = T0.c[4] * (1 + arb("1e-6"))
        held = [(D, T0.value(D, y)[0], 1.0, "plant") for D, y, _, _ in held]
    M = fit(pts, (2, 2, 2), uniform=True)
    s, at, ok = held_out(M, held)
    good = ok and s < 1e-20 and M.info["max_miss"] < 1e-20
    print(f"selftest: in-sample max miss {M.info['max_miss']:.2e}, held-out {s:.2e} (at D {at}); "
          f"{'PASS' if good else 'FAIL'}")
    return dict(experiment="selftest", in_max=M.info["max_miss"], held=s, ok=bool(good)), (0 if good else 1)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    name, mutate = ARGS.get("data", "T_cut"), ARGS.get("mutate")
    if cmd == "selftest":
        out, code = selftest()
    elif cmd == "A":
        out, code = exp_A(name, mutate)
    elif cmd == "B":
        out, code = exp_B(name)
    elif cmd == "C":
        out, code = exp_C(name)
    else:
        raise SystemExit(__doc__)
    out["prec_bits"] = PREC
    if "out" in ARGS:
        os.makedirs(os.path.dirname(os.path.abspath(ARGS["out"])), exist_ok=True)
        json.dump(out, open(ARGS["out"], "w"), indent=1)
    sys.exit(code)


if __name__ == "__main__":
    main()

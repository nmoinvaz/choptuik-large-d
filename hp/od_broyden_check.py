#!/usr/bin/env python3
"""Checks of newton_od's Broyden updates (octo_corrector.newton_od broyden=1, anm_run.py --od_broyden 1) on planted
toy systems in mpmath at 1200 bits, behind the octo-double interface newton_od uses (residual_o, newton_o and
olu_solve in eight parts). Each system is F(z) = A e + N(e), e = z - z*, with its root z* planted. No kernel is needed.

    cd hp && python3 od_broyden_check.py [--base OLD_OCTO_CORRECTOR.py]
        [--mutate=norank1|norefuse|bits|count|deferred|border]

Cases, with the pass criteria fixed before the first run:
  A  planted root, mild quadratic N, start |e| ~ 1e-3, tol 1e-60: the natural test with and without the updates both
     converge, each to max |z - z*| <= 2 ||J(z*)^-1|| tol, their Delta (the last unknown) agrees to 4 ||J^-1|| tol,
     and with the updates the run forms fewer factorizations and takes at least two update steps.
  B  the same run against an independent dense Broyden iteration (the general good update
     B += (y - B s) s^T / |s|^2 on explicit matrices, mpmath solves): every trial point of newton_od equals the
     reference's to 1e-25 of the reference's step, in the same order, and both end at the same iteration.
  C  a planted case that needs a refactorization: N(e) = gamma (p.e)(q.e) p and a start along q with a small p part,
     so the first Newton step is accepted with a small contraction while the step after it is not, since the Jacobian
     changes along p, which no step has seen (Theta about g / (1 + g), g = gamma q.e0 = 3). The dense reference
     predicts the refusal at update 1. Pass: newton_od refuses there, forms the factors again, and converges to the
     root (max |z - z*| <= 2 ||J^-1|| tol).
  D  ElimO around case A's system with the last three unknowns eliminated (a projection that solves the last three
     rows for them): both paths converge to the root, Delta agrees as in A, and the updates form fewer factorizations.
  G  the updates with the chain's other rules, on case A's system: with a strong chord of 2 directions the run
     converges to the root (as A) with fewer factorizations than without the updates; with the floor rule (floor_delta
     1e-30, floor_F 1e-40, tol 1e-70) it ends by the floor rule, at an iterate where the updates formed the factors
     again for it, within 2 ||J^-1|| max |F| of the root.
  E  (with --base) the off path bit for bit: cases A, C and D, a strong chord, the floor rule, the own test, junk
     directions frozen, the weak-first chord, and the full step's drop with the abort rule, run by the given older
     octo_corrector.py and by this one with broyden 0, must return the same eight parts, the same records and the same
     sequence of evaluation points.
  F  the counts od_broyden_ab.py reads from a log: for every off run, its count of factorizations from the printed
     lines equals the toy's count of newton_o calls; for every run with the updates, the printed "od Broyden" line's
     counts equal the toy's and newton_od's info; for every run, od_broyden_ab.deferred (converged without factors
     at the end) holds exactly when the toy's last evaluation is a residual; and case D with times=1 (--od_times 1),
     off and on, prints an "od times" line whose factorizations are the toy's, and is bit for bit the run without it.
     The off runs of A and of G's strong chord are counted again with a border line printed at each newton_o call, as
     arclength.py's BorderedCore prints one, and the chord's log must hold an accepted chord followed by such a line.

Controls, each of which must make the check fail (exit 1):
  --mutate=norank1   the updates' rank-one terms dropped (each update step is the chord's): B fails.
  --mutate=norefuse  every update step taken whatever its contraction: C fails.
  --mutate=bits      with --base, one start entry of this module's runs moved by an ulp: E fails.
  --mutate=count     od_broyden_ab's count without the start's factorization: F fails.
  --mutate=deferred  od_broyden_ab.deferred always False: F fails.
  --mutate=border    od_broyden_ab's count with the border lines not skipped after a strong chord: F fails.
"""
import argparse
import contextlib
import importlib.util
import io
import os
import random
import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import octo_corrector as oc  # noqa: E402
import od_broyden_ab as ab  # noqa: E402

PREC = 1200
TOL = 1e-60


def to8(v):
    """an mpf vector as eight float64 parts, greedily (each part the nearest double of what is left)"""
    parts = [np.zeros(len(v)) for _ in range(8)]
    with mp.workprec(PREC):
        for i, x in enumerate(v):
            r = mp.mpf(x)
            for q in range(8):
                f = float(r)
                parts[q][i] = f
                r -= f
    return tuple(np.ascontiguousarray(p) for p in parts)


def of8(z):
    """eight parts summed exactly into an mpf vector"""
    with mp.workprec(PREC):
        return mp.matrix([mp.fsum(mp.mpf(float(np.asarray(p)[i])) for p in z) for i in range(len(z[0]))])


class Toy:
    """F(z) = A e + N(e), e = z - z*, in mpmath at PREC bits behind newton_od's interface. newton_o factors J(z) and
    keeps it for olu_solve; calls lists every evaluation point (residual_o and newton_o) as eight parts"""

    def __init__(self, zstar, A, N, dN):
        self.zstar, self.A, self.N, self.dN = zstar, A, N, dN
        self.size = len(zstar)
        self.J = None
        self.n_newton = 0
        self.calls = []
        self.border = False             # print arclength's border line at each newton_o (BorderedCore.newton_o)

    def F(self, x):
        with mp.workprec(PREC):
            e = x - self.zstar
            return self.A * e + self.N(e)

    def Jac(self, x):
        with mp.workprec(PREC):
            return self.A + self.dN(x - self.zstar)

    def residual_o(self, z, mutate=0):
        self.calls.append(("F", tuple(np.array(p) for p in oc.o8(z))))
        return to8(self.F(of8(z)))

    def newton_o(self, z, mutate=0):
        self.calls.append(("J", tuple(np.array(p) for p in oc.o8(z))))
        self.n_newton += 1
        if self.border:
            print("      border at D 259.383: g +0.00e+00, dD -1.47e-17 -> held, den 1.0000, |b| 0.0030, |a| 1.00e-16",
                  flush=True)
        x = of8(z)
        with mp.workprec(PREC):
            self.J = self.Jac(x)
            f = self.F(x)
            d = -mp.lu_solve(self.J, f)
            piv = [abs(self.J[i, i]) for i in range(self.size)]
        f8 = to8(f)
        return to8(d), f8[0], f8[1], float(min(piv)), float(max(piv))

    def olu_solve(self, rhs, trans=False):
        with mp.workprec(PREC):
            b = of8(rhs)
            return to8(mp.lu_solve(self.J.T if trans else self.J, b))


def make_elim(module, m):
    """an ElimO of `module` (so that its newton_od takes the projection paths) whose projection solves the last m rows
    of its system for the last m unknowns, by Newton in mpmath from the point's own values"""

    class ToyElim(module.ElimO):
        def __init__(self, so):         # noqa: super().__init__ builds a PinnedQ and a re-seeder this toy has not
            self.so, self.log, self.why, self.first = so, False, [], True
            self.counts = dict(u0=0, u0q=0, ladder=0, failed=0, skipped=0)
            self.m = m

        def project(self, z):
            t = self.so
            x = of8(z)
            n = t.size
            with mp.workprec(PREC):
                for _ in range(60):
                    f = t.F(x)
                    r = mp.matrix([f[i] for i in range(n - m, n)])
                    if mp.norm(r, mp.inf) < mp.mpf(2) ** (-PREC + 40):
                        break
                    Jf = t.Jac(x)
                    Jb = mp.matrix([[Jf[i, j] for j in range(n - m, n)] for i in range(n - m, n)])
                    dx = mp.lu_solve(Jb, r)
                    for k in range(m):
                        x[n - m + k] -= dx[k]
            self.counts["u0"] += 1
            return to8(x)
    return ToyElim


def rand_matrix(rnd, n, delta):
    return mp.eye(n) + mp.matrix([[delta * (rnd.random() - 0.5) for _ in range(n)] for _ in range(n)])


def system_mild(n=10, seed=11, gamma=2.0):
    """case A: N_i(e) = gamma sum_j B_ij e_j^2, A = I + 0.4 R, the start e0 ~ 1e-3"""
    rnd = random.Random(seed)
    with mp.workprec(PREC):
        zstar = mp.matrix([mp.mpf(rnd.random() - 0.5) for _ in range(n)])
        zstar[n - 1] = mp.mpf("1.68") + mp.mpf(rnd.random()) / 100        # Delta, the last unknown
        A = rand_matrix(rnd, n, 0.4)
        B = mp.matrix([[rnd.random() - 0.5 for _ in range(n)] for _ in range(n)])
        e0 = mp.matrix([mp.mpf(rnd.random() - 0.5) * mp.mpf("2e-3") for _ in range(n)])

    def N(e):
        return gamma * (B * mp.matrix([x * x for x in e]))

    def dN(e):
        return gamma * mp.matrix([[B[i, j] * 2 * e[j] for j in range(n)] for i in range(n)])
    return zstar, A, N, dN, zstar + e0


def system_refactor(n=8, seed=3, gamma=10.0, eps="1e-3"):
    """case C: N(e) = gamma (p.e)(q.e) p, p = e_1, q = e_0, A = I + 0.1 R; the start e0 = 0.3 q + eps p + a small
    rest, so g = gamma q.e0 = 3 and the Jacobian changes by about -g p p^T over the first step"""
    rnd = random.Random(seed)
    with mp.workprec(PREC):
        zstar = mp.matrix([mp.mpf(rnd.random() - 0.5) for _ in range(n)])
        A = rand_matrix(rnd, n, 0.1)
        e0 = mp.matrix([0] * n)
        e0[0], e0[1] = mp.mpf("0.3"), mp.mpf(eps)
        for i in range(2, n):
            e0[i] = mp.mpf("0.3") * (rnd.random() - 0.5) * 0.3

    def N(e):
        v = mp.matrix([0] * n)
        v[1] = gamma * e[1] * e[0]
        return v

    def dN(e):
        M = mp.matrix(n, n)
        M[1, 0], M[1, 1] = gamma * e[1], gamma * e[0]
        return M
    return zstar, A, N, dN, zstar + e0


def inv_norm(toy):
    """||J(z*)^-1|| in the max norm"""
    with mp.workprec(PREC):
        Ji = mp.inverse(toy.Jac(toy.zstar))
        return float(mp.norm(Ji, mp.inf))


def run(module, sysdef, broyden, elim=0, start_bump=False, border=False, **kw):
    """newton_od of `module` on the toy, from the system's start (one entry moved by an ulp with start_bump), with
    its printed log (with border, a border line at each newton_o): (z, hist, conv, toy, log, info)"""
    zstar, A, N, dN, x0 = sysdef
    toy = Toy(zstar, A, N, dN)
    toy.border = border
    s = make_elim(module, elim)(toy) if elim else toy
    z0 = to8(x0)
    if start_bump:
        z0 = tuple(np.array(p) for p in z0)
        z0[0][0] = np.nextafter(z0[0][0], np.inf)
    info = {}
    buf = io.StringIO()
    args = dict(tol=TOL, it_max=30, verbose=True, accept="natural")
    args.update(kw)
    if broyden:
        args.update(broyden=1, info=info)
    with contextlib.redirect_stdout(buf):
        z, hist, conv = module.newton_od(s, z0, **args)
    return z, hist, conv, toy, buf.getvalue(), info


def err_root(z, toy):
    with mp.workprec(PREC):
        return float(mp.norm(of8(z) - toy.zstar, mp.inf))


def delta_of(z):
    with mp.workprec(PREC):
        return of8(z)[len(z[0]) - 1]


def reference(sysdef, tol=TOL, theta_max=0.5, it_max=30):
    """the dense Broyden iteration (independent of newton_od's recursion): the first Newton step, then
    B_(k+1) = B_k + (y - B_k s) s^T / |s|^2 on explicit matrices, each step s = -B^-1 F, taken while
    Theta = |B^-1 F(x + s)|_2 / |s|_2 < theta_max or max |F(x + s)| <= tol. Returns the trial points (Newton's, then the
    updates') and where it ended ('converged', 'refused' at an update index)"""
    zstar, A, N, dN, x0 = sysdef
    toy = Toy(zstar, A, N, dN)
    with mp.workprec(PREC):
        x, F0 = x0, toy.F(x0)
        J0 = toy.Jac(x0)
        s = -mp.lu_solve(J0, F0)
        x1 = x + s
        F1 = toy.F(x1)
        trials = [x1]
        th0 = mp.norm(mp.lu_solve(J0, F1), 2) / mp.norm(s, 2)
        if mp.norm(F1, mp.inf) <= tol:
            return trials, ("converged", 0)
        if not th0 < theta_max:
            return trials, ("no updates", 0)
        B = J0 + (F1 - F0 - J0 * s) * s.T / (s.T * s)[0]
        x, Fx = x1, F1
        for k in range(1, it_max):
            s = -mp.lu_solve(B, Fx)
            xt = x + s
            Ft = toy.F(xt)
            trials.append(xt)
            Th = mp.norm(mp.lu_solve(B, Ft), 2) / mp.norm(s, 2)
            if mp.norm(Ft, mp.inf) <= tol:
                return trials, ("converged", k)
            if not Th < theta_max:
                return trials, ("refused", k)
            B = B + (Ft - Fx - B * s) * s.T / (s.T * s)[0]
            x, Fx = xt, Ft
    return trials, ("cap", it_max)


def trial_points(toy):
    """the points of the toy's residual_o calls (newton_od's trials, without ElimO), as mpf vectors"""
    return [of8(z) for kind, z in toy.calls if kind == "F"]


def same_run(r1, r2):
    """bit for bit: the eight parts returned, the records' numbers, and the evaluation points"""
    z1, h1, c1, t1 = r1[:4]
    z2, h2, c2, t2 = r2[:4]
    if c1 != c2 or len(h1) != len(h2) or len(t1.calls) != len(t2.calls):
        return False
    if not all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(z1, z2)):
        return False
    for a, b in zip(h1, h2):
        for k in ("F", "dz", "lam", "F_new", "dz_new", "dDelta"):
            if k in a or k in b:
                x, y = a.get(k), b.get(k)
                if not (x == y or (isinstance(x, float) and isinstance(y, float) and np.isnan(x) and np.isnan(y))):
                    return False
    for (k1, p1), (k2, p2) in zip(t1.calls, t2.calls):
        if k1 != k2 or not all(np.array_equal(a, b) for a, b in zip(p1, p2)):
            return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="", help="an older octo_corrector.py, for case E")
    ap.add_argument("--mutate", default="", choices=["", "norank1", "norefuse", "bits", "count", "deferred", "border"])
    a = ap.parse_args()
    mp.mp.prec = PREC
    if a.mutate in ("norank1", "norefuse"):
        oc.QN_MUT[a.mutate] = True
    if a.mutate == "count":
        ab.MUT["nostart"] = True
    if a.mutate == "border":
        ab.MUT["border"] = True
    if a.mutate == "deferred":
        ab.deferred = lambda log: False
    results = []

    def judge(name, ok, line):
        results.append(ok)
        print(f"{name}: {line}: {'pass' if ok else 'FAIL'}", flush=True)

    runs = []           # (label, run) for case F
    # A: the planted root, with and without the updates
    sA = system_mild()
    off = run(oc, sA, 0)
    on = run(oc, sA, 1)
    runs += [("A off", off, 0), ("A on", on, 1)]
    bound = 2 * inv_norm(off[3]) * TOL
    eo, en = err_root(off[0], off[3]), err_root(on[0], on[3])
    dD = abs(float(delta_of(off[0]) - delta_of(on[0])))
    judge("A planted root", bool(off[2] and on[2] and eo <= bound and en <= bound and dD <= 2 * bound
                                 and on[5]["lus"] < off[3].n_newton and on[5]["quasi"] >= 2),
          f"converged off {off[2]} on {on[2]}; max|z - z*| off {eo:.1e}, on {en:.1e} (bound {bound:.1e}); "
          f"|Delta_on - Delta_off| {dD:.1e} (bound {2 * bound:.1e}); factorizations off {off[3].n_newton}, on "
          f"{on[5]['lus']}; update steps {on[5]['quasi']}, refused {on[5]['refused']}")

    # B: the trial points against the dense reference
    ref, end = reference(sA)
    pts = trial_points(on[3])
    with mp.workprec(PREC):
        steps = [mp.norm(ref[0] - sA[4], mp.inf)] + [mp.norm(ref[k] - ref[k - 1], mp.inf) for k in range(1, len(ref))]
        rel = [float(mp.norm(p - r, mp.inf) / st) for p, r, st in zip(pts, ref, steps)]
    worst = max(rel) if rel else float("inf")
    judge("B dense reference", bool(len(pts) == len(ref) and end[0] == "converged" and worst <= 1e-25),
          f"trial points newton_od {len(pts)}, reference {len(ref)} (ends {end[0]} at update {end[1]}); worst "
          f"|x - x_ref| / |s_ref| {worst:.1e} (bound 1e-25)" + (f"; per trial {', '.join(f'{r:.0e}' for r in rel)}"
                                                                  if rel else ""))

    # C: a case whose updates must be refused once
    sC = system_refactor()
    refC, endC = reference(sC)
    onC = run(oc, sC, 1)
    offC = run(oc, sC, 0)
    runs += [("C off", offC, 0), ("C on", onC, 1)]
    bC = 2 * inv_norm(onC[3]) * TOL
    eC = err_root(onC[0], onC[3])
    first_ref = [h.get("quasi_refused", {}).get("update") for h in onC[1] if "quasi_refused" in h]
    judge("C refactor", bool(endC == ("refused", 1) and onC[2] and eC <= bC and onC[5]["refused"] >= 1
                             and first_ref[:1] == [1]),
          f"reference ends {endC[0]} at update {endC[1]}; newton_od converged {onC[2]}, refused {onC[5]['refused']} "
          f"(first at update {first_ref[0] if first_ref else None}), factorizations {onC[5]['lus']} (off "
          f"{offC[3].n_newton}); max|z - z*| {eC:.1e} (bound {bC:.1e})")

    # D: ElimO, the last three unknowns eliminated
    offD = run(oc, sA, 0, elim=3)
    onD = run(oc, sA, 1, elim=3)
    runs += [("D off", offD, 0), ("D on", onD, 1)]
    bD = 2 * inv_norm(offD[3]) * TOL
    zDo = make_elim(oc, 3)(offD[3]).project(offD[0])
    zDn = make_elim(oc, 3)(onD[3]).project(onD[0])
    eDo, eDn = err_root(zDo, offD[3]), err_root(zDn, onD[3])
    dDD = abs(float(delta_of(zDo) - delta_of(zDn)))
    judge("D ElimO", bool(offD[2] and onD[2] and eDo <= bD and eDn <= bD and dDD <= 2 * bD
                          and onD[5]["lus"] < offD[3].n_newton),
          f"converged off {offD[2]} on {onD[2]}; projected max|z - z*| off {eDo:.1e}, on {eDn:.1e} (bound {bD:.1e}); "
          f"|Delta_on - Delta_off| {dDD:.1e}; factorizations off {offD[3].n_newton}, on {onD[5]['lus']}; update "
          f"steps {onD[5]['quasi']}, refused {onD[5]['refused']}")

    # G: the updates with the strong chord and with the floor rule
    kwc = dict(strong_chord=2)
    kwf = dict(floor_delta=1e-30, floor_F=1e-40, floor_dz=1e-30, tol=1e-70)
    offG1, onG1 = run(oc, sA, 0, **kwc), run(oc, sA, 1, **kwc)
    onG2 = run(oc, sA, 1, **kwf)
    runs += [("G chord off", offG1, 0), ("G chord on", onG1, 1), ("G floor on", onG2, 1)]
    eG1 = err_root(onG1[0], onG1[3])
    h2 = onG2[1]
    FG2 = h2[-1]["F"]
    bG2 = 2 * inv_norm(onG2[3]) * FG2
    eG2 = err_root(onG2[0], onG2[3])
    by_floor = bool(h2[-1].get("floor") and len(h2) >= 2 and "floor" in str(h2[-2].get("refactor", "")))
    judge("G chord and floor rule", bool(onG1[2] and eG1 <= bound and onG1[5]["lus"] < offG1[3].n_newton and onG2[2]
                                         and by_floor and eG2 <= bG2),
          f"chord: converged {onG1[2]}, max|z - z*| {eG1:.1e} (bound {bound:.1e}), factorizations {onG1[5]['lus']} "
          f"(off {offG1[3].n_newton}); floor: converged {onG2[2]}, by the floor rule after factors formed again "
          f"{by_floor}, max|F| {FG2:.1e}, max|z - z*| {eG2:.1e} (bound {bG2:.1e})")

    # E: the off path bit for bit against an older module
    if a.base:
        spec = importlib.util.spec_from_file_location("octo_corrector_base", a.base)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        sF = system_refactor(eps="2e-2")
        cases = [("A", sA, {}, 0), ("C", sC, {}, 0), ("D", sA, {}, 3),
                 ("strong chord", sA, dict(strong_chord=2), 0),
                 ("floor rule", sA, dict(floor_delta=1e-30, floor_F=1e-40, floor_dz=1e-30, tol=1e-70), 0),
                 ("own test", sF, dict(accept="own"), 0), ("junk", sA, dict(junk_w=2), 0),
                 ("weak first", sA, dict(strong_chord=2, weak_first=1), 0),
                 ("drop and abort", sC, dict(full_drop=1e-3, abort_theta=1e8, abort_it=2), 0)]
        bad = []
        for name, sd, kw, el in cases:
            r_old = run(old, sd, 0, elim=el, **kw)
            r_new = run(oc, sd, 0, elim=el, start_bump=a.mutate == "bits", **kw)
            if not same_run(r_old, r_new):
                bad.append(name)
            if not kw.get("accept") and not kw.get("junk_w"):
                runs.append((f"E {name} off", r_new, 0))
        judge("E off path bit for bit", not bad, f"{len(cases)} runs against {a.base}" + (f"; differ: {bad}" if bad else
                                                                                         "; all equal"))
    else:
        print("E off path bit for bit: skipped (no --base)", flush=True)

    # F: the counts od_broyden_ab.py reads from logs, also with arclength's border lines
    rbA, rbG = run(oc, sA, 0, border=True), run(oc, sA, 0, border=True, **kwc)
    runs += [("A off border", rbA, 0), ("G chord off border", rbG, 0)]
    badF = []
    gl = [x for x in rbG[4].splitlines() if x.strip()]
    if not any(m and m.group(1) == "accepted" and ab.RE_BORDER.match(y) for m, y in zip(map(ab.RE_CHORD.match, gl), gl[1:])):
        badF.append("G chord off border: no accepted chord followed by a border line")
    for label, r, br in runs:
        if r is None:
            continue
        toy, log, info = r[3], r[4], r[5]
        ends_without = r[2] and toy.calls[-1][0] == "F"         # the toy's last evaluation a residual, no newton_o
        if br:
            printed = ab.printed_counts(log)
            okF = len(printed) == 1 and printed[0]["lus"] == toy.n_newton == info.get("lus")
        else:
            okF = ab.count_lus(log, runs=1) == toy.n_newton
        okF = okF and ab.deferred(log) == ends_without
        if not okF:
            badF.append(label)
    # the times line (--od_times 1): its count of factorizations is the toy's, and timing changes no bit
    for br in (0, 1):
        rt = run(oc, sA, br, elim=3, times=1)
        x = ab.times_of(rt[4])
        if not (x and x["n_lu"] == rt[3].n_newton and x["n_proj"] > 0 and same_run(rt, run(oc, sA, br, elim=3))):
            badF.append(f"D {'on' if br else 'off'} with times")
    runs.append(("D times", None, None))
    judge("F log counts", not badF, f"{len(runs)} runs" + (f"; differ: {badF}" if badF else "; every count equals the toy's"))

    ok = all(results)
    print(("PASS" if ok else "FAIL") + (f" (--mutate={a.mutate})" if a.mutate else ""), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

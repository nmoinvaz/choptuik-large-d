#!/usr/bin/env python3
"""Checks of the corrector's cheaper projections (#1032): Reseeder.project under proj_start 'pred', refine_o's stop
at its floor (od_stop), ElimO's octo-double projection (proj_o), and the rung residual formed from its own blocks
(reseed.res_rows, Solver.residual_sub).

    cd hp && python3 proj_check.py mock [--mutate=ulp|nopred|nostop|stall4|sign|o_nobar]
    cd hp && python3 proj_check.py bits RAW STATE.npz [--threads 10] [--mutate ulp]
    cd hp && python3 proj_check.py time RAW STATE.npz [--threads 10] [--rep 2] [--paths default,pred,stop,o]
    cd hp && python3 proj_check.py accuracy RAW STATE.npz [--threads 10] [--mutate sign|o_nobar]
    cd hp && python3 proj_check.py floor RAW STATE.npz [--threads 10]

mock runs no kernel: a rung of M = 2 (MockRung) whose m and a rows are a smooth map of the horizon data, evaluated in
mpmath from the parts given and rounded to as many, with the a_h columns times a gain of 1e30 (the mass mode's role).
Its cases and pass criteria, fixed before the first run:
  A (od_stop at a floor): residual_o reads the data in their quad-double rounding, a hard floor. From the root's
    data moved by 1e-20, the default refine_o (3 steps) ends by its cap with 10 residuals and 8 rung Jacobians; with
    od_stop it ends at 'floor' with at most 4 residuals and 2 Jacobians and |R| at most twice the default's.
  B (od_stop, a poor first LU, mutate 'badlu'): with od_stop the corrections still end within 10 times A's floor.
  C (od_stop, no floor): refine_o with and without od_stop give the same eight parts bit for bit.
  D (proj_start 'pred' where the u0 path fails): the double-double rung at its floor (every trial 1e6) and the
    quad-double polish of its rounded data not finite (#693's pole). The default project() ends 'ladder' (the re-seed's
    polish of the point's own data); with 'pred' it ends 'pred' with the same four parts bit for bit, and fewer
    residual calls.
  E (proj_start 'pred' where every path fails, qd_bar 1e-80): both end 'failed' with the point unchanged, bit for
    bit, and the same number of quad-double residuals (the own-data polish runs once on each path).
  F (ElimO proj_o, planted truth): from the root's data moved by 1e-20 relative (m_h; a_h by 1e-20 over the gain),
    the projection is counted 'u0o' and its data lie within 1e-60 of the root's (relative, max over the data).
  G (res_rows): with Solver.residual_sub the rung rows come from it, and they equal res_rows_full's bit for bit.
Controls, each of which must fail (exit 1): ulp (one ulp of D's 'pred' output), nopred (D's second run with
proj_start 'u0'), nostop (A's od_stop off), stall4 (OD_STALL at OD_REFACTOR: B stops at the poor LU), sign (F's
corrections with the wrong sign: refused, so not 'u0o'), o_nobar (with sign, the bar off: accepted, and the truth
test fails).

bits, time, accuracy and floor run the kernel on a stored eight-part chain state (octo_corrector.truth_system), with
the chain's Reseeder settings (qd, od 3, qd_start pred, qd_bar 1e-13); STATE is relative to RAW, the raw-data clone.
  bits: the state's horizon data moved by 0, 1e-24 and 1e-20 (relative, random signs), each projected by
    Reseeder.project and with_horizon on the default path and under proj_start 'pred': every part of the result
    compared by its bits where the default ended 'ladder'. Then res_rows against res_rows_full on the rung point in
    two, four and eight parts. Exit 1 on a difference; --mutate ulp (one ulp of the 'pred' result) must fail.
  time: one projection per path and repeat, with the seconds of each stage (the double-double rung, the quad-double
    polishes, the re-seed, refine_o) and the kernel calls by kind: default, pred (proj_start 'pred'), stop ('pred' and
    od_stop), o (refine_o alone with the stop, ElimO's proj_o).
  accuracy: from the state's data moved by 1e-24 and 1e-20, the projections of the default path, of 'pred' with
    od_stop and of proj_o: the rung's max |R|, the pinned octo-double max |F| at the projected point, and the data's
    distance from the default's projection of the unmoved state. Pass: on each plant the cheaper paths' pinned
    max |F| is at most 10 times the default's or below 2e-34 (the chain's tol_od), their distance at most 10 times the
    default's plus 1e-60, and proj_o accepted. --mutate sign (refine_o's corrections with the wrong sign) and
    --mutate o_nobar (sign with the bar off) must fail.
  floor: refine_o from the state's data rounded to quad-double, as with_horizon leaves them, with per correction the
    rung's |R|, whether R and the data changed bit for bit, and the correction's size in the m_h and a_h columns;
    then the change of the rung residual when one m_h and one a_h datum are moved by 2^-k of themselves (k 200 to
    320), against the quad-double Jacobian's prediction: where the octo-double residual stops seeing the data.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402

import reseed  # noqa: E402

DPS = 160


# ------------------------------------------------------------------ the mock rung
def split(v, k):
    """an mpf as k float64 parts, each the nearest double to what is left"""
    out = []
    for _ in range(k):
        f = float(v)
        out.append(f)
        v -= mp.mpf(f)
    return out


def total(parts, i):
    """entry i of a part array list, summed exactly"""
    return sum((mp.mpf(float(p[i])) for p in parts), mp.mpf(0))


class MockRung:
    """a two-node rung solver of M = 2 (L = 25, the layout of hpkernel.driver), whose horizon block's m and a rows are
    g(d) minus node 1's m and a fields, d the 2M horizon data: g = g0 + A u + c u*u with u = S (d - d0), S the gain on
    the a_h columns. Every method evaluates in mpmath from the parts given and rounds to as many parts. floor_qd:
    residual_o reads d in its quad-double rounding (a hard floor). pole_dd: residual_q is not finite where the data's
    parts 2 and 3 are zero. dd_fail: the double-double residual's rung rows are 1e6 everywhere. sub: has
    residual_sub. calls counts the calls by name and parts."""

    M = 2

    def __init__(self, gain=1e30, floor_qd=False, pole_dd=False, dd_fail=False, sub=False):
        self.gain, self.floor_qd, self.pole_dd, self.dd_fail = gain, floor_qd, pole_dd, dd_fail
        M = self.M
        self.L = 4 * M * 3 + 1
        self.r0, self.c0, self.t0 = 8 * M + M, 2 * M + 8 * M, 2 * M + 4 * M + M     # rung rows, data, node 1's m
        n = 2 * M
        with mp.workdps(DPS):
            self.S = [mp.mpf(1)] * M + [mp.mpf(gain)] * M
            self.A = [[mp.mpf(1) + mp.mpf(i == j) * 3 + mp.mpf(i + 2 * j) / 7 for j in range(n)] for i in range(n)]
            self.d0 = [mp.mpf("0.1"), mp.mpf("0.05"), mp.mpf("1.0001"), mp.mpf("1.00005")]
            self.g0 = [mp.mpf("0.3"), mp.mpf("-0.2"), mp.mpf("1.1"), mp.mpf("0.9")]
        self.c = mp.mpf("0.25")
        self.calls = {}
        if sub:
            self.residual_sub = self._residual_sub

    def count(self, name, parts):
        k = f"{name}/{parts}"
        self.calls[k] = self.calls.get(k, 0) + 1

    def data(self, z, parts):
        """the 2M data of z in mpmath, z read in its first `parts` parts"""
        return [total(z[:parts], self.c0 + j) for j in range(2 * self.M)]

    def g(self, d):
        n = 2 * self.M
        u = [(d[j] - self.d0[j]) * self.S[j] for j in range(n)]
        return [self.g0[i] + sum(self.A[i][j] * u[j] for j in range(n)) + self.c * u[i] * u[i] for i in range(n)]

    def jac(self, d):
        n = 2 * self.M
        u = [(d[j] - self.d0[j]) * self.S[j] for j in range(n)]
        return [[(self.A[i][j] + (2 * self.c * u[i] if i == j else 0)) * self.S[j] for j in range(n)] for i in range(n)]

    def full(self, z, parts, read):
        """the residual of z (`parts` parts in, as many out), the data read in their first `read` parts"""
        with mp.workdps(DPS):
            d = self.data(z, parts)
            if read < parts:
                d = [sum((mp.mpf(f) for f in split(v, read)), mp.mpf(0)) for v in d]
            g = self.g(d)
            F = [np.zeros(self.L) for _ in range(parts)]
            for i in range(2 * self.M):
                v = split(g[i] - total(z[:parts], self.t0 + i), parts)
                for q in range(parts):
                    F[q][self.r0 + i] = v[q]
        return F

    def residual(self, zh, zl):
        self.count("residual", 2)
        F = self.full([zh, zl], 2, 2)
        if self.dd_fail:
            F[0][self.r0:self.r0 + 2 * self.M], F[1][self.r0:self.r0 + 2 * self.M] = 1e6, 0.0
        return F

    def residual_q(self, z0, z1, z2, z3, mutate=0):
        self.count("residual", 4)
        z = [z0, z1, z2, z3]
        if self.pole_dd and all(not np.any(np.asarray(z[q])[self.c0:self.c0 + 2 * self.M]) for q in (2, 3)):
            return [np.full(self.L, np.nan) for _ in range(4)]
        return self.full(z, 4, 4)

    def residual_o(self, z, mutate=0):
        self.count("residual", 8)
        return self.full(list(z), 8, 4 if self.floor_qd else 8)

    def _residual_sub(self, z, rows):
        self.count("residual_sub", len(z))
        F = (self.residual(*z) if len(z) == 2 else self.residual_q(*z) if len(z) == 4 else self.residual_o(z))
        self.count("residual", len(z))          # residual_sub's own call above is not a full residual's
        self.calls[f"residual/{len(z)}"] -= 2
        return [np.asarray(p)[list(rows)] for p in F]

    def jacobian_sub(self, z, rows, cols):
        parts = len(z)
        self.count("jacobian_sub", parts)
        with mp.workdps(DPS):
            J = self.jac(self.data(z, parts))
            out = [np.zeros((len(rows), len(cols))) for _ in range(parts)]
            for a, r in enumerate(rows):
                for b, c in enumerate(cols):
                    i, j = r - self.r0, c - self.c0
                    v = split(J[i][j] if 0 <= i < 2 * self.M and 0 <= j < 2 * self.M else mp.mpf(0), parts)
                    for q in range(parts):
                        out[q][a, b] = v[q]
        return out


def mock_point(mk, rel=1e-20, parts=8, seed=1):
    """the mock's point in `parts` parts: node 1's m and a fields set so that d* = d0 + 1e-3 (m_h) and d0 + 1e-3 / gain
    (a_h) is the root, the data d* moved by rel (m_h) and rel / gain (a_h), random signs; returns (z, d*)"""
    M, n = mk.M, 2 * mk.M
    rng = np.random.default_rng(seed)
    sg = rng.choice([-1.0, 1.0], n)
    with mp.workdps(DPS):
        dstar = [mk.d0[j] + mp.mpf("1e-3") / mk.S[j] for j in range(n)]
        g = mk.g(dstar)
        vals = [mp.mpf(0)] * mk.L
        for i in range(2 * M):
            vals[i] = mp.mpf("0.01") * (i + 1)
        for i in range(8 * M):
            vals[2 * M + i] = mp.mpf("0.5") + mp.mpf(i) / 100
        for i in range(n):
            vals[mk.t0 + i] = g[i]
            moved = dstar[i] * mp.mpf(rel) * sg[i] if i < M else mp.mpf(rel) / mk.S[i] * sg[i]
            vals[mk.c0 + i] = dstar[i] + moved
        vals[-1] = mp.mpf("1.6846521667419536")
        z = [np.zeros(mk.L) for _ in range(parts)]
        for i, v in enumerate(vals):
            for q, f in enumerate(split(v, parts)):
                z[q][i] = f
    return tuple(np.ascontiguousarray(p) for p in z), dstar


def mock_reseeder(mk, **kw):
    """a Reseeder of the mock's layout (two nodes, u0 0.3) whose solvers are the mock"""
    rs = reseed.Reseeder(dict(M=mk.M, nodes=[0.005, 0.7], kt=36, kc=146, ks=130), mp.mpf(251), threads=1, **kw)
    rs.solver = lambda nodes, kt=None: mk
    rs.rung_solver = lambda u: mk
    return rs


def same_bits(a, b):
    return len(a) == len(b) and all(np.asarray(x).tobytes() == np.asarray(y).tobytes() for x, y in zip(a, b))


def dist_root(z, dstar, mk):
    """the largest distance of z's horizon data from d* relative to the largest datum"""
    with mp.workdps(DPS):
        d = mk.data(list(z), len(z))
        return float(max(abs(d[j] - dstar[j]) for j in range(len(d))) / max(abs(x) for x in dstar))


def last_node_target(rs):
    """Reseeder.targets without the stepper: the u0 rung's target (the last node) alone, no rungs below"""
    def targets(zh, zl):
        nk = rs.off["node"](rs.K - 1)
        return {rs.us[-1]: (np.array(zh[nk:nk + 4 * rs.M]), np.array(zl[nk:nk + 4 * rs.M]))}, []
    return targets


def mock(argv):
    mutate = next((a.split("=", 1)[1] for a in argv if a.startswith("--mutate=")), "")
    from octo_corrector import ElimO, PROJ_MUT
    ok = {}
    if mutate == "stall4":
        reseed.OD_STALL = reseed.OD_REFACTOR

    # A: od_stop at a floor
    mk = MockRung(floor_qd=True)
    z8, _ = mock_point(mk, 1e-20)
    rs = mock_reseeder(mk, qd=True, od=3)
    mk.calls = {}
    za, ha = rs.refine_o(z8)
    end_a, calls_a = rs.last_o_end, dict(mk.calls)
    mk.calls = {}
    zb, hb = rs.refine_o(z8, stop=mutate != "nostop")
    end_b, calls_b = rs.last_o_end, dict(mk.calls)
    ok["A"] = (end_a == "cap" and calls_a.get("residual/8") == 10 and calls_a.get("jacobian_sub/4") == 8
               and end_b == "floor" and calls_b.get("residual/8", 0) <= 4 and calls_b.get("jacobian_sub/4", 0) <= 2
               and hb[-1] <= 2 * ha[-1])
    print(f"A od_stop at a floor: default |R| {' -> '.join(f'{r:.2e}' for r in ha)}, end {end_a}, calls {calls_a}; "
          f"od_stop {' -> '.join(f'{r:.2e}' for r in hb)}, end {end_b}, calls {calls_b}: {'PASS' if ok['A'] else 'FAIL'}")

    # B: od_stop with a poor first LU
    zc, hc = rs.refine_o(z8, stop=True, mutate="badlu")
    ok["B"] = hc[-1] <= 10 * ha[-1]
    print(f"B od_stop, poor first LU: {' -> '.join(f'{r:.2e}' for r in hc)}, end {rs.last_o_end}: "
          f"{'PASS' if ok['B'] else 'FAIL'}")

    # C: no floor, the same bits
    mk = MockRung()
    z8, _ = mock_point(mk, 1e-20)
    rs = mock_reseeder(mk, qd=True, od=3)
    zd, hd = rs.refine_o(z8)
    end_d = rs.last_o_end
    ze, he = rs.refine_o(z8, stop=True)
    ok["C"] = same_bits(zd, ze) and hd == he
    print(f"C no floor: default {' -> '.join(f'{r:.2e}' for r in hd)} ({end_d}), od_stop "
          f"{' -> '.join(f'{r:.2e}' for r in he)} ({rs.last_o_end}), the same bits: {same_bits(zd, ze)}: "
          f"{'PASS' if ok['C'] else 'FAIL'}")

    # D: proj_start 'pred' where the u0 path fails
    mk = MockRung(pole_dd=True, dd_fail=True)
    z4 = tuple(np.ascontiguousarray(p) for p in mock_point(mk, 1e-20, parts=4)[0])
    rs0 = mock_reseeder(mk, qd=True, od=3, qd_start="pred")
    mk.calls = {}
    p0, how0 = rs0.project(z4)
    calls0 = dict(mk.calls)
    rs1 = mock_reseeder(mk, qd=True, od=3, qd_start="pred", proj_start="u0" if mutate == "nopred" else "pred")
    mk.calls = {}
    p1, how1 = rs1.project(z4)
    calls1 = dict(mk.calls)
    if mutate == "ulp":
        p1 = [np.array(p) for p in p1]
        j = rs1.off["mh"]
        p1[3][j] = np.nextafter(p1[3][j], np.inf)
    n0, n1 = (sum(v for k, v in c.items() if k.startswith("residual")) for c in (calls0, calls1))
    ok["D"] = how0 == "ladder" and how1 == "pred" and same_bits(p0, p1) and n1 < n0
    print(f"D proj_start pred, the u0 path failing: default '{how0}' {calls0}; pred '{how1}' {calls1}; the same bits: "
          f"{same_bits(p0, p1)}: {'PASS' if ok['D'] else 'FAIL'}")

    # E: every path failing
    out = []
    for ps in ("u0", "pred"):
        r = mock_reseeder(mk, qd=True, od=3, qd_start="pred", qd_bar=1e-80, proj_start=ps)
        r.targets = last_node_target(r)
        mk.calls = {}
        p, how = r.project(z4)
        out.append((p, how, dict(mk.calls)))
    ok["E"] = (out[0][1] == out[1][1] == "failed" and same_bits(out[0][0], z4) and same_bits(out[1][0], z4)
               and out[0][2].get("residual/4") == out[1][2].get("residual/4"))
    print(f"E every path failing: default '{out[0][1]}' {out[0][2]}; pred '{out[1][1]}' {out[1][2]}: "
          f"{'PASS' if ok['E'] else 'FAIL'}")

    # F: ElimO's octo-double projection, planted truth
    mk = MockRung()
    z8, dstar = mock_point(mk, 1e-20)
    rs = mock_reseeder(mk, qd=True, od=3)
    if mutate in ("sign", "o_nobar"):
        ro = rs.refine_o
        rs.refine_o = lambda z, steps=None, mutate="", stop=None: ro(z, steps, "sign", stop)
    PROJ_MUT["o_nobar"] = mutate == "o_nobar"

    class SO:
        s, ch, cl = mk, np.zeros(mk.L), np.zeros(mk.L)

    class SQ:
        def residual_q(self, *z):
            return [np.ones(mk.L)] + [np.zeros(mk.L)] * 3

    el = ElimO(SO(), rs, switch=0.0, start_F=None, proj_o=1)
    el.sq = SQ()
    zp = el.project(z8)
    e0, e1 = dist_root(z8, dstar, mk), dist_root(zp, dstar, mk)
    ok["F"] = el.counts.get("u0o", 0) == 1 and e1 <= 1e-60
    print(f"F ElimO proj_o: counts {el.counts}, rung |R| {' -> '.join(f'{r:.2e}' for r in rs.last_o)}, distance from "
          f"the root {e0:.2e} -> {e1:.2e}: {'PASS' if ok['F'] else 'FAIL'}")
    PROJ_MUT["o_nobar"] = False

    # G: res_rows from residual_sub
    mk = MockRung(sub=True)
    z8, _ = mock_point(mk, 1e-20)
    rows = list(range(mk.r0, mk.r0 + 2 * mk.M))
    good = True
    for parts in (2, 4, 8):
        z = [np.ascontiguousarray(p) for p in (z8[:parts] if parts < 8 else z8)]
        mk.calls = {}
        a = reseed.res_rows(mk, z, rows)
        used = mk.calls.get(f"residual_sub/{parts}", 0) == 1
        b = reseed.res_rows_full(mk, z, rows)
        good = good and used and same_bits(a, b)
    ok["G"] = good
    print(f"G res_rows through residual_sub, two, four and eight parts: {'PASS' if good else 'FAIL'}")

    passed = all(ok.values())
    print(f"mock{' --mutate=' + mutate if mutate else ''}: {'PASS' if passed else 'FAIL'} "
          f"({', '.join(k for k, v in ok.items() if not v) or 'all cases'}{' failed' if not passed else ''})")
    return 0 if passed else 1


# ------------------------------------------------------------------ the kernel checks on a stored chain state
def _opts(argv, **defaults):
    unknown = [a for a in argv[2:] if a.startswith("--") and a[2:] not in defaults]
    if unknown or len(argv) < 2:
        raise SystemExit(f"proj_check.py: expected RAW STATE.npz [options], unknown {unknown}; known {sorted(defaults)}")
    opt = dict(defaults)
    for k in opt:
        if "--" + k in argv:
            opt[k] = type(opt[k])(argv[argv.index("--" + k) + 1])
    return argv[0], argv[1], opt


def chain_rs(meta, threads, **kw):
    """the chain's Reseeder at the state's D (anm_run.py --reseed 2 --reseed_q 1 --reseed_o 3 --reseed_start pred
    --reseed_qd_bar 1e-13, ks_chain3b), with kw on top"""
    from octo_corrector import D_of
    return reseed.Reseeder(meta, D_of(meta), threads=threads, qd=True, od=3, qd_start="pred", qd_bar=1e-13, **kw)


SMOKE = {}      # --smoke_ks K: every check's rung at horizon and centre order K, a smoke test of the code (no meaning)


def load8(raw, path):
    from octo_corrector import has_o, load_o
    fp = os.path.join(raw, path)
    if not has_o(fp):
        raise SystemExit(f"{path}: not an eight-part state")
    z, meta = load_o(fp)
    if SMOKE.get("ks"):
        meta = dict(meta, ks=SMOKE["ks"], kc=SMOKE["ks"])
    return z, meta


def plant(z8, rs, rel, seed=1):
    """z8 with m_h moved by rel and a_h by rel / G relative (G = x_K-1^-(n-1), the mass mode's gain), random signs
    and sizes 0.5-1, so that the rung's |R| is about rel: a corrector's trial, whose data sit within the Newton step's
    second-order term of the rung's solution"""
    from multidouble import od_axpy
    from octo_corrector import o8
    if rel == 0:
        return o8(z8)
    M, sl = rs.M, slice(rs.off["mh"], rs.off["mh"] + 2 * rs.M)
    G = float(rs.x_node) ** -(float(rs.D) - 3.0)
    rng = np.random.default_rng(seed)
    r = rng.uniform(0.5, 1.0, 2 * M) * rng.choice([-1.0, 1.0], 2 * M)
    dz = np.zeros(len(z8[0]))
    dz[sl] = rel * r * np.asarray(z8[0])[sl] * np.r_[np.ones(M), np.full(M, 1.0 / G)]
    zero = np.zeros(len(dz))
    return o8(od_axpy(1.0, dz, *([zero] * 7), *o8(z8)))


class Timed:
    """a solver whose residual and Jacobian calls are counted and timed into rec[name/parts] = (calls, seconds)"""

    NAMES = dict(residual=2, residual_q=4, residual_o=8, residual_sub=0, jacobian_sub=0, jacobian=2, jacobian_q=4)

    def __init__(self, s, rec):
        self._s, self._rec = s, rec

    def __getattr__(self, name):
        f = getattr(self._s, name)
        if name not in self.NAMES or not callable(f):
            return f

        def g(*a, **k):
            t = time.time()
            r = f(*a, **k)
            key = f"{name}/{self.NAMES[name] or len(a[0])}"
            n, sec = self._rec.get(key, (0, 0.0))
            self._rec[key] = (n + 1, sec + time.time() - t)
            return r
        return g


def instrument(rs, rec):
    """rs's rung solvers wrapped in Timed and its stages timed into rec['stage:<name>'] (inclusive seconds)"""
    orig = rs.rung_solver
    rs.rung_solver = lambda u: Timed(orig(u), rec)
    for name in ("invert", "invert_q", "own_polish", "reseed", "refine_o", "targets"):
        f = getattr(rs, name)

        def g(*a, _f=f, _n=name, **k):
            t = time.time()
            try:
                return _f(*a, **k)
            finally:
                n, sec = rec.get("stage:" + _n, (0, 0.0))
                rec["stage:" + _n] = (n + 1, sec + time.time() - t)
        setattr(rs, name, g)


def fmt(rec):
    return ", ".join(f"{k} {n}x {sec:.1f} s" for k, (n, sec) in sorted(rec.items()))


def bits(argv):
    """the default projection against proj_start 'pred' and res_rows against res_rows_full, bit for bit"""
    from octo_corrector import D_of, o8, q4_of_o, with_horizon
    raw, path, opt = _opts(argv, threads=10, mutate="", plants="0,1e-24,1e-20", sub=0)
    z8, meta = load8(raw, path)
    print(f"bits {path}: D {mp.nstr(D_of(meta), 12)}, ks {meta.get('ks')}, threads {opt['threads']}, mutate "
          f"'{opt['mutate']}'", flush=True)
    ok, tested = True, 0
    for k, rel in enumerate(float(x) for x in opt["plants"].split(",")):
        rs0, rs1 = chain_rs(meta, opt["threads"]), chain_rs(meta, opt["threads"], proj_start="pred")
        zp = plant(z8, rs0, rel)
        z4 = q4_of_o(zp)
        t0 = time.time()
        p0, how0 = rs0.project(z4)
        t1 = time.time()
        p1, how1 = rs1.project(z4)
        t2 = time.time()
        if opt["mutate"] == "ulp":
            p1 = [np.array(p) for p in p1]
            j = rs1.off["mh"] + rs1.M
            p1[3][j] = np.nextafter(p1[3][j], np.inf)
        same = same_bits(p0, p1)
        line = f"  plant {rel:g}: default '{how0}' {t1 - t0:.1f} s, pred '{how1}' {t2 - t1:.1f} s, four parts the same: {same}"
        if how0 == "ladder":
            tested += 1
            ok = ok and same and how1 == "pred"
        elif how0 == "u0q" and not same:
            from multidouble import qd_axpy
            sl = slice(rs0.off["mh"], rs0.off["mh"] + 2 * rs0.M)
            dq = qd_axpy(-1.0, *(np.ascontiguousarray(np.asarray(x)[sl]) for x in p0),
                         *(np.ascontiguousarray(np.asarray(x)[sl]) for x in p1))
            d = float(np.max(np.abs(np.asarray(dq[0]) + np.asarray(dq[1]))))
            line += f" (the default converged at u0q, so the paths differ as documented: horizon data by {d:.2e})"
        print(line, flush=True)
        if k == 0:
            t3 = time.time()
            w0, w1 = with_horizon(zp, p0, rs0), with_horizon(zp, p1, rs1)
            print(f"    with_horizon (refine_o, 3 steps) on both: eight parts the same: {same_bits(w0, w1)}, "
                  f"{time.time() - t3:.1f} s; rung |R| {' -> '.join(f'{r:.2e}' for r in rs0.last_o)}", flush=True)
            ok = ok and (same_bits(w0, w1) or opt["mutate"] == "ulp")
    # the rung residual from its own blocks
    rs = chain_rs(meta, opt["threads"])
    s = rs.rung_solver(rs.us[-1])
    if hasattr(s, "residual_sub"):
        from dup_eval_check import rung_point
        rows = rs.rung_rows()
        for parts in (2, 4, 8):
            p = rung_point(rs, o8(z8), parts)
            t0 = time.time()
            a = reseed.res_rows(s, p, rows)
            t1 = time.time()
            b = reseed.res_rows_full(s, p, rows)
            t2 = time.time()
            if opt["mutate"] == "ulp":
                a = [np.array(x) for x in a]
                a[parts - 1][0] = np.nextafter(a[parts - 1][0], np.inf)
            same = same_bits(a, b)
            print(f"  res_rows in {parts} parts: residual_sub {t1 - t0:.2f} s, full residual {t2 - t1:.2f} s, the same: "
                  f"{same}", flush=True)
            ok, tested = ok and same, tested + 1
    else:
        print("  this hpkernel has no Solver.residual_sub: res_rows not tested", flush=True)
        if opt["sub"]:
            return 2
    print(f"bits: {'PASS' if ok and tested else 'FAIL'} ({tested} comparisons)", flush=True)
    return 0 if ok and tested else 1


def timing(argv):
    """the seconds of one projection by path, stage and kernel call"""
    from octo_corrector import D_of, project_o, q4_of_o, with_horizon
    raw, path, opt = _opts(argv, threads=10, rep=2, paths="default,pred,stop,o", plant=1e-24)
    z8, meta = load8(raw, path)
    print(f"time {path}: D {mp.nstr(D_of(meta), 12)}, ks {meta.get('ks')}, threads {opt['threads']}, plant "
          f"{opt['plant']:g}", flush=True)
    zp = plant(z8, chain_rs(meta, opt["threads"]), opt["plant"])
    z4 = q4_of_o(zp)
    for name in opt["paths"].split(","):
        for r in range(opt["rep"]):
            kw = dict(default={}, pred=dict(proj_start="pred"), stop=dict(proj_start="pred", od_stop=True), o={})[name]
            rs = chain_rs(meta, opt["threads"], **kw)
            rec = {}
            instrument(rs, rec)
            t0 = time.time()
            if name == "o":
                _, ho, end, acc = project_o(rs, zp)
                how = f"u0o {'accepted' if acc else 'refused'}, end {end}"
            else:
                p, how = rs.project(z4)
                if how != "failed":
                    with_horizon(zp, p, rs)
                ho = rs.last_o if how != "failed" else []
            print(f"  {name} #{r + 1}: {time.time() - t0:.1f} s, '{how}', rung |R| "
                  f"{' -> '.join(f'{x:.2e}' for x in ho)}; {fmt(rec)}", flush=True)
    return 0


def accuracy(argv):
    """the cheaper projections against the default on planted trials of a stored state"""
    from octo_corrector import D_of, PROJ_MUT, omax, project_o, q4_of_o, truth_system, with_horizon
    raw, path, opt = _opts(argv, threads=10, mutate="", plants="1e-24,1e-20", tol=2e-34)
    z8, meta = load8(raw, path)
    if SMOKE.get("ks"):
        from octo_corrector import pinned_o
        so, info = pinned_o(meta, z8, threads=opt["threads"], held_exact=True, segs=meta["segs"])[3], "smoke"
    else:
        _, _, (_, _, _, so), info = truth_system(raw, path, threads=opt["threads"])
    PROJ_MUT["o_nobar"] = opt["mutate"] == "o_nobar"
    sl = slice(chain_rs(meta, 1).off["mh"], chain_rs(meta, 1).off["mh"] + 2 * int(meta["M"]))

    def dist(a, b):
        with mp.workdps(DPS):
            return float(max(abs(total(a, i) - total(b, i)) for i in range(sl.start, sl.stop))
                         / max(abs(total(b, i)) for i in range(sl.start, sl.stop)))

    rs = chain_rs(meta, opt["threads"])
    t0 = time.time()
    zref = with_horizon(z8, rs.project(q4_of_o(z8))[0], rs)
    print(f"accuracy {path}: D {mp.nstr(D_of(meta), 12)}, {info}; the reference (the default projection of the state) "
          f"pinned max|F| {omax(so.residual_o(zref)):.2e}, rung |R| {rs.last_o[-1]:.2e}, {time.time() - t0:.0f} s",
          flush=True)
    ok = True
    for rel in (float(x) for x in opt["plants"].split(",")):
        out = {}
        for name in ("default", "stop", "o"):
            rs = chain_rs(meta, opt["threads"], **(dict(proj_start="pred", od_stop=True) if name == "stop" else {}))
            zp = plant(z8, rs, rel)
            t0 = time.time()
            if name == "o":
                zo, ho, end, acc = project_o(rs, zp, mutate="sign" if opt["mutate"] in ("sign", "o_nobar") else "")
                R, how = ho[-1], f"u0o {'accepted' if acc else 'refused'} ({end})"
            else:
                p, how = rs.project(q4_of_o(zp))
                zo = with_horizon(zp, p, rs)
                R, acc = rs.last_o[-1], how != "failed"
            out[name] = dict(F=omax(so.residual_o(zo)), R=R, dist=dist(zo, zref), how=how, acc=bool(acc),
                             sec=time.time() - t0)
            print(f"  plant {rel:g} {name}: '{how}', rung |R| {R:.2e}, pinned max|F| {out[name]['F']:.2e}, distance from "
                  f"the reference {out[name]['dist']:.2e}, {out[name]['sec']:.0f} s", flush=True)
        d = out["default"]
        for name in ("stop", "o"):
            c = out[name]
            good = (c["acc"] and (c["F"] <= 10 * d["F"] or c["F"] <= opt["tol"])
                    and c["dist"] <= 10 * d["dist"] + 1e-60)
            ok = ok and good
            print(f"    {name} against the default: {'ok' if good else 'NOT ok'}", flush=True)
    PROJ_MUT["o_nobar"] = False
    print(f"accuracy{' --mutate ' + opt['mutate'] if opt['mutate'] else ''}: {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


def floor(argv):
    """refine_o's stall at its floor, correction by correction, and the octo-double residual's sensitivity to the data
    (item 3 of #1032; the docstring)"""
    from multidouble import QDLu, od_axpy
    from octo_corrector import D_of
    raw, path, opt = _opts(argv, threads=10, steps=8)
    z8, meta = load8(raw, path)
    rs = chain_rs(meta, opt["threads"])
    M, a, o = rs.M, rs.aoff, rs.off
    s = rs.rung_solver(rs.us[-1])
    rows, cols = rs.rung_rows(), np.r_[a["mh"]:a["mh"] + 2 * M]
    sl = slice(o["mh"], o["mh"] + 2 * M)
    n0, n1, nk = o["node"](0), o["node"](1), o["node"](rs.K - 1)
    base = [np.zeros(a["L"]) for _ in range(8)]
    for q in range(8):
        zq = np.asarray(z8[q])
        base[q][:2 * M], base[q][a["node"](0):a["node"](1)] = zq[:2 * M], zq[n0:n1]
        base[q][a["node"](1):a["node"](1) + 4 * M], base[q][-1] = zq[nk:nk + 4 * M], zq[-1]

    def put(d, parts):
        v = [b.copy() for b in base[:parts]]
        for q in range(parts):
            v[q][cols] = d[q]
        return [np.ascontiguousarray(x) for x in v]

    def R_of(d):
        return [np.ascontiguousarray(p) for p in reseed.res_rows(s, put(d, 8), rows)]

    zero = np.zeros(2 * M)
    d_state = [np.ascontiguousarray(np.asarray(z8[q])[sl]) for q in range(8)]
    d = [d_state[q] if q < 4 else zero.copy() for q in range(8)]           # with_horizon's rounding to quad-double
    print(f"floor {path}: D {mp.nstr(D_of(meta), 12)}, ks {meta.get('ks')}, G {float(rs.x_node) ** -(float(rs.D) - 3.0):.3e}",
          flush=True)
    R_st = R_of(d_state)
    print(f"  the state's own data: rung max|R| {np.max(np.abs(R_st[0] + R_st[1])):.2e}", flush=True)
    R = R_of(d)
    lu = None
    for k in range(opt["steps"]):
        nR = float(np.max(np.abs(R[0] + R[1])))
        if lu is None or k >= 2:
            lu = QDLu(*[np.ascontiguousarray(b) for b in reseed.jac_block(s, put(d, 4), rows, cols)])
        x = [np.ascontiguousarray(np.asarray(p)) for p in lu.solve(*[-r for r in R[:4]])]
        dn = [np.ascontiguousarray(np.asarray(p)) for p in od_axpy(1.0, *x, *([zero] * 4), *d)]
        Rn = R_of(dn)
        xm, xa = float(np.max(np.abs(x[0][:M]))), float(np.max(np.abs(x[0][M:])))
        moved = [int(np.sum(dn[q] != d[q])) for q in range(8)]
        print(f"  correction {k + 1}: |R| {nR:.3e} -> {np.max(np.abs(Rn[0] + Rn[1])):.3e}; R the same bits: "
              f"{same_bits(R, Rn)}; |x| m_h {xm:.2e}, a_h {xa:.2e}; data entries changed per part {moved}", flush=True)
        d, R = dn, Rn
    # does the octo-double residual read the data's parts 4-7?
    dq = [d[q] if q < 4 else zero for q in range(8)]
    Rq = R_of(dq)
    print(f"  R at the data and at their parts 0-3 alone: the same bits: {same_bits(R, Rq)}, |R| "
          f"{np.max(np.abs(R[0] + R[1])):.3e} and {np.max(np.abs(Rq[0] + Rq[1])):.3e}", flush=True)
    J = [np.asarray(b) for b in reseed.jac_block(s, put(d, 4), rows, cols)]
    for j in (0, M):
        for k in (200, 212, 220, 230, 240, 260, 280, 300, 320):
            t = float(d[0][j]) * 2.0 ** -k
            e = np.zeros(2 * M)
            e[j] = t
            dt = [np.ascontiguousarray(np.asarray(p)) for p in od_axpy(1.0, e, *([zero] * 7), *d)]
            Rt = R_of(dt)
            dR = od_axpy(-1.0, *R, *Rt)
            got = float(np.max(np.abs(np.asarray(dR[0]) + np.asarray(dR[1]))))
            want = float(np.max(np.abs(J[0][:, j] * t)))
            print(f"  column {'m_h' if j < M else 'a_h'} {j % M}, moved by 2^-{k} of itself ({t:.2e}): |dR| {got:.3e}, "
                  f"the Jacobian's {want:.3e}, ratio {got / want if want else float('nan'):.3e}", flush=True)
    return 0


if __name__ == "__main__":
    mp.mp.dps = 50
    if "--smoke_ks" in sys.argv:
        i = sys.argv.index("--smoke_ks")
        SMOKE["ks"] = int(sys.argv[i + 1])
        del sys.argv[i:i + 2]
    cmds = dict(mock=mock, bits=bits, time=timing, accuracy=accuracy, floor=floor)
    if len(sys.argv) > 1 and sys.argv[1] in cmds:
        sys.exit(cmds[sys.argv[1]](sys.argv[2:]))
    print(__doc__)
    sys.exit(2)

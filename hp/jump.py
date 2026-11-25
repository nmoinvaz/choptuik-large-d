#!/usr/bin/env python3
"""Issue #1088, Track A: the critical solution at a target D by one direct solve, with no continuation in D.

guess   The unknowns of stored large-D states, scaled by their large-D laws (p, m, p1 and m_h times sqrt(n); c0, a, c and
        a_h as (v - 1) n; n = D - 2), are fitted by a polynomial in t = 1/n of low degree, component by component, and
        the fit is evaluated at the target D. Delta is Track C's series value at that D (hp/data/delta_all.csv, kind
        series), or the fit's with --delta fit. The segments are planned at the guess at --plan_tol, and the horizon
        order is tied to D (ks_rule). The horizon data are then re-seeded at that order from the guess's last node by
        the octo-double ladder below. The guess is saved with eight parts (octo_corrector.save_o).
solve   Newton in octo-double on the system of --kfreeze (13: p1's harmonics 1-13 solved, 15 and up set to zero, no
        harmonic held; the kernel's Solver.newton_o, residual_o; M = 32 only), with every trial point projected: its
        horizon data re-seeded from its own last node by the ladder before its residual is read. A trial is accepted
        when its max |F| falls (halving the step up to --halvings times). One record per step in OUT/jump.jsonl.
planted A converged state moved by a known relative amount (random signs, seed) in every unknown but the horizon data,
        which the projection sets, then solved by solve's iteration. Pass: Delta within --pass_delta of the state's and
        max |F| below --tol. --mutate step (the step applied with its sign flipped) must make it fail.

The ladder (OLadder). The horizon data (m_h, a_h) enter through the horizon series summed at u0 = 1 - x_last, where
the mass mode amplifies an error delta in a_h by G(u) = (1 - u)^-(n-1), 10^38 at D 250 and 10^62 at D 400. Data off
by more than about 1/G(u0) put a singularity of the series inside u0, so the data of a guess, or of a Newton trial,
cannot be polished at u0 alone. The ladder inverts the series on rungs u_1 < ... < u_J = u0 spaced in log10 G by at
most --rung_dec decades: each rung's target is the last node integrated outward to x = 1 - u (double-double Taylor
steps, Reseeder.integrate; the mass mode decays outward) or the node itself at u0 (all eight parts), its equations
the m and a rows of the horizon block (Reseeder.rows 'ma'), and its Newton runs on octo-double residuals
(Solver.residual_sub, eight parts) with the rung block's quad-double LU (jacobian_sub, QDLu), each rung starting
from the last one's data. A rung's data are off by its target's error over G(u), so the next rung starts about
1e-32 G(u_j+1)/G(u_j) from its root.

    python3 jump.py guess OUT.npz --D 250 --states 'GLOB' [--Dmin 226 --Dmax 245.64 --deg 3 --kfreeze 13
        --plan_tol 1e-40 --ks auto --threads 2 --delta series|fit]
    python3 jump.py solve GUESS.npz OUTDIR [--tol 1e-40 --it_max 16 --halvings 8 --threads 2 --mutate none|step]
    python3 jump.py planted STATE.npz OUTDIR [--rel 1e-10 --seed 1 --pass_delta 1e-30 ...]
    python3 jump.py ks D [--tail 2^-212]

Output paths must be in a task folder of the raw-data repository (raw_out.py; --allow_unraw for scratch)."""
import glob
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402
from hpkernel.driver import make_solver  # noqa: E402
from multidouble import ODLu, QDLu, od_axpy  # noqa: E402
from octo_corrector import delta_digits, load_o, o8, omax, save_o  # noqa: E402
from raw_out import pop_allow, require_raw  # noqa: E402
from reseed import Reseeder, jac_block, res_rows  # noqa: E402

M_ = 32


# ------------------------------------------------------------------ orders tied to D
def ks_rule(D, x_last=0.7, tail=2.0 ** -212, gbar=None):
    """the horizon order at D: the larger of #731's ceil(1.95 Re s) + 10, with Re s = (n/2 - 1 + gbar) / (1 - gbar)
    (notes/LARGE_D_BASICS.md 2; gbar = (n - 1) <a_h^2 - 1>, 7 / n if not given, its size on the chain), and the
    smallest order whose mass-mode relative tail sum_{k > ks} C(n-2+k, k) u^k (1-u)^(n-1) at u = 1 - x_last is below
    tail (LARGE_D_BASICS 4, solver/basics/order_laws.py). Returns (ks, the first rule's, the second's)."""
    n = float(D) - 2.0
    g = 7.0 / n if gbar is None else float(gbar)
    ks_s = math.ceil(1.95 * (n / 2 - 1 + g) / (1 - g)) + 10
    with mp.workdps(160):
        u = 1 - mp.mpf(x_last)
        m = mp.mpf(n) - 2
        term = (1 - u) ** (m + 1)                   # the negative binomial's k = 0 term; all terms sum to 1
        tot, k = term, 0
        while True:
            k += 1
            term = term * (m + k) / k * u
            tot += term
            if k > 0.43 * n and 1 - tot < tail:
                break
    return max(ks_s, k), ks_s, k


# ------------------------------------------------------------------ layout and scaling
def groups(M, K):
    o = 2 * M + 4 * M * K
    g = dict(p1=slice(0, M), c0=slice(M, 2 * M), mh=slice(o, o + M), ah=slice(o + M, o + 2 * M))
    for i in range(K):
        b = 2 * M + 4 * M * i
        for f, nm in enumerate("pmac"):
            g[f"n{i}{nm}"] = slice(b + f * M, b + (f + 1) * M)
    return g


def scale(z, D, M, K):
    """the scaled unknowns: p1, p, m, m_h times sqrt(n); c0, a, c, a_h as (v - 1) n; Delta as is"""
    n = float(D) - 2.0
    y = np.array(z, dtype=float)
    for nm, sl in groups(M, K).items():
        if nm in ("p1", "mh") or nm[-1] in "pm" and nm.startswith("n"):
            y[sl] = y[sl] * math.sqrt(n)
        else:
            y[sl] = (y[sl] - 1.0) * n
    return y


def unscale(y, D, M, K):
    n = float(D) - 2.0
    z = np.array(y, dtype=float)
    for nm, sl in groups(M, K).items():
        if nm in ("p1", "mh") or nm[-1] in "pm" and nm.startswith("n"):
            z[sl] = y[sl] / math.sqrt(n)
        else:
            z[sl] = 1.0 + y[sl] / n
    return z


def series_delta(D, path=os.path.join(HERE, "data", "delta_all.csv")):
    """Track C's series value at D from delta_all.csv (kind series): (Delta as mpf, err) or None"""
    import csv
    for r in csv.DictReader(open(path)):
        if r["kind"] == "series" and abs(float(r["D"]) - float(D)) < 1e-9:
            with mp.workdps(40):
                return mp.mpf(r["Delta_q0"]) + mp.mpf(r["Delta_q1"] or 0), float(r["err"])
    return None


# ------------------------------------------------------------------ a_h from the momentum constraint
_DM = {}


def _dmat_full(N, dps):
    """d/dtheta on the full grid of N points over 2 pi (trigonometric interpolation), mpmath"""
    key = (N, dps)
    if key not in _DM:
        with mp.workdps(dps):
            D = mp.matrix(N, N)
            for j in range(N):
                for k in range(N):
                    if j != k:
                        D[j, k] = mp.mpf(0.5) * (-1) ** (j - k) * mp.cot((j - k) * mp.pi / N)
            _DM[key] = D
    return _DM[key]


def slave_ah(z8, M, K, D, dps=160):
    """a_h from m_h by the momentum constraint at the SSH, x = c = 1 (notes/LIT_GAPS_DSS.md 2, form S of Gundlach 1997
    Eq. (19) in D dimensions): (omega d/dtheta - (n - 1) - q)(B - 1) = q with q = 2 m_h^2 and B = 1/a_h^2, solved on the
    full grid of 2M points in mpmath at dps digits from all eight parts of m_h and Delta. Data with a_h so slaved carry
    no momentum violation at the SSH, which form A's horizon series would turn into the mass mode x^-(n-1). Returns
    a_h's eight parts on the half grid."""
    o = 2 * M + 4 * M * K
    with mp.workdps(dps):
        n = mp.mpf(D) - 2
        mh = [mp.fsum(mp.mpf(float(np.asarray(p)[o + j])) for p in z8) for j in range(M)]
        Dl = mp.fsum(mp.mpf(float(np.asarray(p)[-1])) for p in z8)
        om = 2 * mp.pi / Dl
        full = mh + [-v for v in mh]
        N = 2 * M
        q = [2 * v * v for v in full]
        Dm = _dmat_full(N, dps)
        L = om * Dm
        for j in range(N):
            L[j, j] -= (n - 1) + q[j]
        b = mp.lu_solve(L, mp.matrix(q))
        ah = [1 / mp.sqrt(1 + b[j]) for j in range(M)]
        out = [np.zeros(M) for _ in range(8)]
        for j, v in enumerate(ah):
            for qq in range(8):
                out[qq][j] = float(v)
                v -= mp.mpf(out[qq][j])
    return out


# ------------------------------------------------------------------ the octo-double ladder
class OLadder:
    """the re-seed of the horizon data on rungs spaced in the mass mode's gain (the module docstring)"""

    def __init__(self, meta, D, threads=2, rung_dec=12.0, first_dec=3.0, plan_tol=1e-32, hmax=0.05, it_max=12,
                 floor=1e-80, conv_tol=1e-26, lu="o", verbose=False):
        self.rs = Reseeder(meta, D, threads=threads, levels=0, rows="ma", plan_tol=plan_tol, hmax=hmax)
        self.M, self.K = self.rs.M, self.rs.K
        n = float(D) - 2.0
        u0 = 1.0 - self.rs.x_node
        g0 = -(n - 1) * math.log10(1.0 - u0)
        decs = []
        d = first_dec
        while d < g0 - 0.5 * rung_dec:
            decs.append(d)
            d += rung_dec
        self.us = [1.0 - 10.0 ** (-dd / (n - 1)) for dd in decs] + [u0]
        self.decs = decs + [g0]
        self.plan_tol, self.hmax, self.it_max, self.floor, self.verbose = plan_tol, hmax, it_max, floor, verbose
        # a rung converges when its |R| ends at or below conv_tol: the u0 rung's quad-double block LU stops near 2e-29
        # at D 245.6 (I1088-mac-a/probes/ladtest3.log), the inner rungs' far below
        self.conv_tol = conv_tol
        # lu: the rung block's LU, "o" octo-double (jacobian_sub at eight parts, ODLu) or "q" quad-double (four parts,
        # QDLu). The block's columns all carry gains near 1e44-1e47 at D 245.6 while the regular family's directions
        # move the node by O(1), so a quad-double LU leaves errors of about 1e-17 of a step in the amplified
        # directions (I1088-mac-a/probes)
        self.lu = lu
        self.last = None

    def targets(self, z8):
        """the last node integrated (double-double) to x = 1 - u for every rung below u0, and the node at u0 in eight
        parts; {u: eight-part target}"""
        M, o = self.M, self.rs.off
        nk = o["node"](self.K - 1)
        zh, zl = np.asarray(z8[0]), np.asarray(z8[1])
        sh, sl = np.array(zh[nk:nk + 4 * M]), np.array(zl[nk:nk + 4 * M])
        inner = sorted(self.us[:-1], reverse=True)          # increasing x
        T, info = self.rs.integrate(zh, zl, sh, sl, [1.0 - u for u in inner], self.plan_tol, self.hmax, self.rs.kt)
        zero = np.zeros(4 * M)
        tg = {}
        for u, t, rec in zip(inner, T, info):
            if rec["max_x_minus_c"] < 0.0 and np.all(np.isfinite(t[0])):
                tg[u] = [np.asarray(t[0]), np.asarray(t[1])] + [zero] * 6
        tg[self.us[-1]] = [np.array(np.asarray(p)[nk:nk + 4 * M]) for p in z8]
        return tg

    def rung(self, u, z8, target, d):
        """Newton on the eight-part data d at rung u against the eight-part target; returns (d, hist, converged), converged
        when |R| ends at or below conv_tol"""
        M, a, o = self.M, self.rs.aoff, self.rs.off
        s = self.rs.rung_solver(u)
        rows = self.rs.rung_rows()
        cols = np.r_[a["mh"]:a["mh"] + 2 * M]
        L = a["L"]
        n0, n1 = o["node"](0), o["node"](1)
        base = [np.zeros(L) for _ in range(8)]
        for q in range(8):
            zq = np.asarray(z8[q])
            base[q][:2 * M] = zq[:2 * M]
            base[q][a["node"](0):a["node"](1)] = zq[n0:n1]
            base[q][a["node"](1):a["node"](1) + 4 * M] = target[q]
            base[q][-1] = zq[-1]

        def put(dd, parts):
            v = [b.copy() for b in base[:parts]]
            for q in range(parts):
                v[q][cols] = dd[q]
            return [np.ascontiguousarray(x) for x in v]

        def res(dd):
            R = [np.ascontiguousarray(p) for p in res_rows(s, put(dd, 8), rows)]
            return R, float(np.max(np.abs(R[0] + R[1])))

        zero = np.zeros(2 * M)
        d = [np.ascontiguousarray(np.asarray(p, dtype=float)) for p in d]
        R, nR = res(d)
        hist = [nR]
        if not np.isfinite(nR):
            return d, hist, False
        lu, fresh = None, False
        for it in range(self.it_max):
            if nR <= self.floor:
                break
            if lu is None:
                if self.lu == "o":
                    lu = ODLu(*[np.ascontiguousarray(b) for b in jac_block(s, put(d, 8), rows, cols)])
                else:
                    lu = QDLu(*[np.ascontiguousarray(b) for b in jac_block(s, put(d, 4), rows, cols)])
                fresh = True
            if self.lu == "o":
                x = [np.ascontiguousarray(np.asarray(p)) for p in lu.solve(*[-r for r in R])]
            else:
                x = [np.ascontiguousarray(np.asarray(p)) for p in lu.solve(*[-r for r in R[:4]])] + [zero] * 4
            lam, acc = 1.0, False
            for _ in range(10):
                dn = [np.ascontiguousarray(np.asarray(p)) for p in od_axpy(lam, *x, *d)]
                Rn, nn = res(dn)
                if np.isfinite(nn) and nn < nR:
                    acc = True
                    break
                lam *= 0.5
            if not acc:
                if fresh:
                    break                       # a fresh LU's step lowers no |R|: the floor
                lu = None
                continue
            gain = nn / nR
            d, R, nR = dn, Rn, nn
            hist.append(nR)
            if fresh and lam == 1.0 and gain > 0.5:
                break                           # a fresh LU's full step does not halve |R|: the floor
            if gain > 1e-3 or lam < 1.0:
                lu = None                       # refactor; otherwise the LU is kept (a chord)
            fresh = False
        return d, hist, bool(np.isfinite(nR) and nR <= self.conv_tol)

    def project(self, z8):
        """z8 with its horizon data re-seeded on the ladder from its own data and last node; (z8', info). A rung that does
        not converge ends the ladder (info ok False), and z8 is returned unchanged."""
        t0 = time.time()
        M, o = self.M, self.rs.off
        sl_ = slice(o["mh"], o["mh"] + 2 * M)
        z8 = [np.array(np.asarray(p), dtype=float) for p in z8]
        tg = self.targets(z8)
        d = [np.array(p[sl_]) for p in z8]
        recs = []
        ok = True
        for u, dec in zip(self.us, self.decs):
            if u not in tg:
                recs.append(dict(u=u, skipped=True))
                continue
            t1 = time.time()
            d, h, okr = self.rung(u, z8, tg[u], d)
            recs.append(dict(u=round(u, 6), G=round(dec, 1), R0=h[0], R=h[-1], its=len(h) - 1, s=round(time.time() - t1, 1)))
            if self.verbose:
                print(f"    rung u {u:.5f} (G 1e{dec:.1f}): |R| {h[0]:.2e} -> {h[-1]:.2e} in {len(h) - 1}, "
                      f"{time.time() - t1:.1f} s", flush=True)
            if not okr:
                ok = False
                break
        out = [np.array(p) for p in z8]
        if ok:
            for q in range(8):
                out[q][sl_] = d[q]
        info = dict(ok=ok, R=recs[-1].get("R") if recs else None, rungs=recs, s=round(time.time() - t0, 1))
        self.last = info
        return o8(out), info


# ------------------------------------------------------------------ the system
def system(meta, D, threads, kfreeze):
    s = make_solver(D, int(meta["M"]), nodes=meta["nodes"], kt=meta.get("kt", 36), kc=meta["kc"], ks=meta["ks"],
                    threads=threads, kfreeze=kfreeze)
    s.set_segs(meta["segs"])
    return s


def D_mp(meta):
    with mp.workdps(60):
        return mp.mpf(meta["D_str"])


def opts(argv, defaults):
    o = dict(defaults)
    pos = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            k = a[2:]
            if "=" in k:
                k, v = k.split("=", 1)
            else:
                v = argv[i + 1]
                i += 1
            o[k] = v
        else:
            pos.append(a)
        i += 1
    return pos, o


# ------------------------------------------------------------------ guess
def cmd_guess(argv):
    pos, o = opts(argv, dict(D="250", states="", Dmin="0", Dmax="1e9", deg="3", kfreeze="13", plan_tol="1e-40",
                             hmax="0.05", ks="auto", kc="146", threads="2", delta="series", rung_dec="12",
                             reseed="1", slave="0"))
    out = pos[0]
    require_raw(out)
    paths = sorted(set(p for g in o["states"].split(",") for p in glob.glob(g)))
    rows = []
    for p in paths:
        z, m = load_o(p)
        if float(o["Dmin"]) <= m["D"] <= float(o["Dmax"]):
            rows.append((m["D"], z, m, p))
    rows.sort(key=lambda r: r[0])
    m0 = rows[-1][2]
    M, K = int(m0["M"]), len(m0["nodes"])
    Ds = np.array([r[0] for r in rows])
    Y = np.array([scale(sum(np.asarray(p) for p in r[1][:2]), r[0], M, K) for r in rows])
    with mp.workdps(60):
        Dt = mp.mpf(o["D"])
    deg = int(o["deg"])
    t = 1.0 / (Ds - 2.0)
    c = np.polyfit(t, Y, deg)
    tt = 1.0 / (float(Dt) - 2.0)
    y = np.array([np.polyval(c[:, j], tt) for j in range(Y.shape[1])])
    # the fit's own spread: degree deg - 1 against deg at the target
    c1 = np.polyfit(t, Y, max(deg - 1, 0))
    y1 = np.array([np.polyval(c1[:, j], tt) for j in range(Y.shape[1])])
    z = unscale(y, float(Dt), M, K)
    spread = np.abs(unscale(y1, float(Dt), M, K) - z)
    print(f"guess at D {o['D']} from {len(rows)} states, D {Ds[0]:.4f} .. {Ds[-1]:.4f}, degree {deg} in 1/n; "
          f"degree {deg - 1} differs by {spread[:-1].max():.2e} (unknowns), {spread[-1]:.2e} (Delta)", flush=True)
    z8 = [z] + [np.zeros(len(z)) for _ in range(7)]
    if o["delta"] == "series":
        sd = series_delta(o["D"])
        if sd is None:
            raise SystemExit(f"no series row at D {o['D']} in delta_all.csv")
        with mp.workdps(60):
            v = sd[0]
            for q in range(8):
                f = float(v)
                z8[q][-1] = f
                v -= mp.mpf(f)
        print(f"  Delta from Track C's series: {mp.nstr(sd[0], 16)} +- {sd[1]:.2e}; the fit's {z[-1]:.12f}", flush=True)
    if o["ks"] == "auto":
        n_t = float(Dt) - 2.0
        ah = z[groups(M, K)["ah"]]
        ks, ks_s, ks_m = ks_rule(float(Dt), gbar=(n_t - 1) * float(np.mean(ah * ah - 1.0)))
        print(f"  ks {ks} (sigma rule {ks_s}, mass-mode tail rule {ks_m})", flush=True)
    else:
        ks = int(o["ks"])
    meta = dict(D=float(Dt), D_str=mp.nstr(Dt, 50), M=M, nodes=m0["nodes"], kt=int(m0.get("kt", 36)),
                kc=int(o["kc"]), ks=ks, kfreeze=int(o["kfreeze"]) if o["kfreeze"] != "none" else None,
                parts=8, source="jump.py guess", states=[os.path.basename(r[3]) for r in rows], deg=deg,
                plan_tol=float(o["plan_tol"]))
    s = make_solver(Dt, M, nodes=meta["nodes"], kt=meta["kt"], kc=meta["kc"], ks=ks, threads=int(o["threads"]),
                    kfreeze=meta["kfreeze"])
    segs = s.plan(np.ascontiguousarray(z8[0]), np.ascontiguousarray(z8[1]), float(o["plan_tol"]), float(o["hmax"]))
    meta["segs"] = [list(map(float, g)) for g in segs]
    print(f"  plan at {o['plan_tol']}: {sum(len(g) - 1 for g in segs)} Taylor steps", flush=True)
    s.set_segs(segs)
    if int(o["slave"]):
        ah = slave_ah(z8, M, K, Dt)
        for q in range(8):
            z8[q][groups(M, K)["ah"]] = ah[q]
        print("  a_h slaved to m_h by the momentum constraint at the SSH", flush=True)
    F = s.residual_o(list(o8(z8)))
    Fv = np.asarray(F[0]) + np.asarray(F[1])
    print(f"  max |F| before the re-seed {omax(F):.3e}; per block "
          + " ".join(f"{np.abs(Fv[b * 4 * M:(b + 1) * 4 * M]).max():.1e}" for b in range(K + 1)), flush=True)
    zp, info = o8(z8), dict(ok=None, skipped=True)
    if int(o["reseed"]):
        lad = OLadder(meta, Dt, threads=int(o["threads"]), rung_dec=float(o["rung_dec"]), verbose=True)
        print(f"  ladder: {len(lad.us)} rungs, log10 G {', '.join(f'{d:.1f}' for d in lad.decs)}", flush=True)
        zp, info = lad.project(z8)
        F = s.residual_o(list(zp))
        Fv = np.asarray(F[0]) + np.asarray(F[1])
        print(f"  re-seed {'ok' if info['ok'] else 'FAILED'} in {info['s']} s; max |F| {omax(F):.3e}; per block "
              + " ".join(f"{np.abs(Fv[b * 4 * M:(b + 1) * 4 * M]).max():.1e}" for b in range(K + 1)), flush=True)
    meta["guess_F"] = omax(F)
    meta["reseed"] = info
    save_o(out, zp, meta)
    print(f"  saved {out}", flush=True)
    return 0


# ------------------------------------------------------------------ solve
def newton(s, lad, z8, tol, it_max, halvings, log, mutate="none", verbose=True):
    """projected Newton (module docstring); returns (z8, converged, records)"""
    z8, info = lad.project(z8)
    F = s.residual_o(list(z8))
    nF = omax(F)
    recs = []
    t_start = time.time()
    if verbose:
        print(f"  start: max |F| {nF:.3e} (projected, {info['s']} s), Delta {delta_digits(z8, 40)}", flush=True)
    conv = nF <= tol
    for it in range(it_max):
        if conv:
            break
        t0 = time.time()
        d, _, _, pmin, pmax = s.newton_o(list(z8))
        d = o8(d)
        t_lu = time.time() - t0
        sign = -1.0 if mutate == "step" else 1.0
        dz = float(np.abs(np.asarray(d[0])).max())
        dD = float(np.asarray(d[0])[-1])
        lam, acc, trials = 1.0, False, []
        for _ in range(halvings + 1):
            zt = o8(od_axpy(sign * lam, *d, *z8))
            zt, pinfo = lad.project(zt)
            Ft = s.residual_o(list(zt)) if pinfo["ok"] else None
            nt = omax(Ft) if Ft is not None else float("inf")
            trials.append(dict(lam=lam, F=nt, proj_R=pinfo["R"], proj_s=pinfo["s"], proj_ok=pinfo["ok"]))
            if np.isfinite(nt) and nt < nF:
                acc = True
                break
            lam *= 0.5
        rec = dict(it=it, F=nF, dz=dz, dDelta=dD, pmin=pmin, pmax=pmax, lu_s=round(t_lu, 1), trials=trials,
                   accepted=acc, s=round(time.time() - t0, 1))
        if acc:
            z8, F, nF = zt, Ft, nt
            conv = nF <= tol
        rec.update(F_new=nF, Delta=delta_digits(z8, 40), utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
        recs.append(rec)
        if log:
            with open(log, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        if verbose:
            print(f"  it {it}: |F| {rec['F']:.3e} -> {nF:.3e}, |dz| {dz:.2e}, dDelta {dD:+.3e}, lam {lam if acc else 0:g}, "
                  f"pivots {pmin:.1e}..{pmax:.1e}, {rec['s']} s (LU {t_lu:.0f} s), Delta {rec['Delta']}", flush=True)
        if not acc:
            break
    if verbose:
        print(f"  {'converged' if conv else 'NOT converged'}: max |F| {nF:.3e}, {time.time() - t_start:.0f} s", flush=True)
    return z8, conv, recs


def cmd_solve(argv):
    pos, o = opts(argv, dict(tol="1e-40", it_max="16", halvings="8", threads="2", mutate="none", rung_dec="12",
                             kfreeze="meta", ref_delta=""))
    path, outdir = pos[0], pos[1]
    os.makedirs(outdir, exist_ok=True)
    log = os.path.join(outdir, "jump.jsonl")
    require_raw(log)
    z8, meta = load_o(path)
    D = D_mp(meta)
    kf = meta.get("kfreeze") if o["kfreeze"] == "meta" else (None if o["kfreeze"] == "none" else int(o["kfreeze"]))
    thr = int(o["threads"])
    s = system(meta, D, thr, kf)
    lad = OLadder(meta, D, threads=thr, rung_dec=float(o["rung_dec"]))
    print(f"solve at D {meta['D_str'][:20]}: ks {meta['ks']}, kc {meta['kc']}, kfreeze {kf}, "
          f"{sum(len(g) - 1 for g in meta['segs'])} Taylor steps, ladder of {len(lad.us)} rungs; {path}", flush=True)
    with open(log, "a") as fh:
        fh.write(json.dumps(dict(start=path, D_str=meta["D_str"], ks=meta["ks"], kc=meta["kc"], kfreeze=kf,
                                 tol=float(o["tol"]), mutate=o["mutate"], threads=thr,
                                 utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))) + "\n")
    z8, conv, recs = newton(s, lad, z8, float(o["tol"]), int(o["it_max"]), int(o["halvings"]), log, o["mutate"])
    meta = dict(meta, source=path, solved=bool(conv), F=omax(s.residual_o(list(z8))), kfreeze=kf)
    meta.pop("reseed", None)
    out = os.path.join(outdir, f"jump_D{float(D):.4f}.npz")
    save_o(out, z8, meta)
    Dl = delta_digits(z8, 40)
    print(f"  Delta {Dl}  saved {out}", flush=True)
    if o["ref_delta"]:
        with mp.workdps(60):
            diff = mp.mpf(Dl) - mp.mpf(o["ref_delta"])
        print(f"  Delta - reference {mp.nstr(diff, 6)}", flush=True)
    return 0 if conv else 1


def main():
    argv = sys.argv[1:]
    pop_allow(argv)
    cmd = argv[0]
    if cmd == "ks":
        print(ks_rule(float(argv[1])))
        return 0
    return dict(guess=cmd_guess, solve=cmd_solve)[cmd](argv[1:])


if __name__ == "__main__":
    mp.mp.dps = 50
    sys.exit(main())

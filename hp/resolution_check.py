#!/usr/bin/env python3
"""The chain's resolution error at large D (notes/BIAS.md, "Open"): a pinned chain state (M = 32) resampled in tau to
other M, its pinned system (the chain's own, kfreeze 3, the held harmonics at the state's values) solved again by the
quad-double Newton, and Delta compared across M (the quad-double solver is built for M = 32 and 48). Also prints the LU's smallest and
largest pivot at every M, at the solution, to see
whether the conditioning wall that stops the chain near D 215 (notes/QUAD_DOUBLE.md 13.25) depends on M.

The resampling is resample_z's trigonometric interpolation (hpkernel.driver), done in mpmath on the sum of all four
parts and split back into four float64 parts, so the start keeps the state's quad-double accuracy (a float64 start is
outside the quad-double polish's basin past D ~ 131, notes/QUAD_DOUBLE.md 13.22).

    cd hp && python3 resolution_check.py RAWDIR STATE [--Ms=32,48] [--threads=2] [--it=12] [--tol=1e-28]
        [--junk=0] [--mutate=resample]

Check (fails with --mutate=resample, which halves every non-constant term's weight in the first resampling):
resampling M -> M' -> M returns the state's p1 block to 1e-40 relative for M' >= M. Exits 1 when the check fails or a solve does not converge.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402
from hpkernel.driver import load_q, make_solver, newton_qd  # noqa: E402
from pinning import PinnedQ, PinnedSolver, band_coeffs, band_field  # noqa: E402

mp.mp.dps = 80


def mp_of(z):
    """the four parts summed in mpmath, entry by entry"""
    return [mp.mpf(float(a)) + mp.mpf(float(b)) + mp.mpf(float(c)) + mp.mpf(float(d)) for a, b, c, d in zip(*z)]


def split4(v):
    """an mpmath vector into four float64 parts, each the rounding of what is left"""
    parts = []
    r = list(v)
    for _ in range(4):
        p = np.array([float(x) for x in r])
        parts.append(p)
        r = [x - mp.mpf(float(y)) for x, y in zip(r, p)]
    return tuple(np.ascontiguousarray(p) for p in parts)


def resample_mp(v, M_old, M_new, K, mutate=""):
    """resample_z's interpolation in mpmath: every field block of the half grid (odd or even about theta = pi)"""
    nf = 4 * (K + 1)
    parity = ["odd", "even"] + ["odd", "odd", "even", "even"] * K + ["odd", "even"]
    N = 2 * M_old
    kmax = min(M_old, M_new)
    cos_t = {}
    out = []
    for j in range(nf):
        f = v[j * M_old:(j + 1) * M_old]
        full = f + ([-x for x in f] if parity[j] == "odd" else list(f))
        coef = []
        for k in range(kmax + 1):
            c = mp.mpc(0)
            for n, x in enumerate(full):
                c += x * mp.expjpi(-mp.mpf(2 * n * k) / N)
            coef.append(c / N)
        for n in range(M_new):
            th = mp.mpf(2 * n) / (2 * M_new)            # theta / pi
            s = mp.mpf(0)
            for k, c in enumerate(coef):
                w = 1 if (k == 0 or k == M_old or mutate == "resample") else 2
                key = (k, n, M_new)
                if key not in cos_t:
                    cos_t[key] = mp.expjpi(k * th)
                s += w * (c * cos_t[key]).real
            out.append(s)
    return out + list(v[nf * M_old:])


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    opts = dict((a[2:] + "=").split("=")[:2] for a in argv if a.startswith("--"))
    if len(args) != 2:
        sys.exit(__doc__)
    raw, st = args
    Ms = [int(x) for x in opts.get("Ms", "32,48").split(",")]
    threads, it, tol = int(opts.get("threads", 2)), int(opts.get("it", 12)), float(opts.get("tol", 1e-28))
    junk = int(opts.get("junk", 0))                # newton_qd's junk mode: the Delta-free weakest directions frozen
    z, m = load_q(os.path.join(raw, st))
    D, M0 = mp.mpf(m["D_str"]) if "D_str" in m else mp.mpf(float(m["D"])), int(m["M"])
    kf = int(opts.get("kfreeze", m.get("kfreeze") or 3))         # --kfreeze for states that do not store it
    K = (len(z[0]) - 1) // (4 * M0) - 1
    v0 = mp_of(z)
    ks = [k for k in range(kf + 1, 14) if k % 2 == 1]
    d0 = v0[-1]
    print(f"{st}: D {mp.nstr(D, 12)}, M {M0}, K {K}, kfreeze {kf}, Delta {mp.nstr(d0, 25)}", flush=True)
    ok = True
    # the check: M0 -> M -> M0 returns p1 to 1e-40
    for M in Ms:
        if M <= M0:
            continue
        back = resample_mp(resample_mp(v0, M0, M, K, opts.get("mutate", "")), M, M0, K)
        err = max(abs(a - b) for a, b in zip(back[:M0], v0[:M0])) / max(abs(x) for x in v0[:M0])
        good = err < 1e-40
        ok &= good
        print(f"  round trip {M0} -> {M} -> {M0}: p1 relative {float(err):.1e} ({'pass' if good else 'FAIL'})", flush=True)
    if opts.get("mutate"):
        print(f"{'pass' if ok else 'FAIL'}: the round trip under --mutate={opts['mutate']} (no solves)")
        return 0 if ok else 1
    res = {}
    for M in Ms:
        z4 = z if M == M0 else split4(resample_mp(v0, M0, M, K))
        s = make_solver(D, M, nodes=list(m["nodes"]), kt=m.get("kt", 36), kc=m.get("kc", 146), ks=m.get("ks", 130),
                        threads=threads, kfreeze=kf)
        s.set_segs(s.plan(np.asarray(z4[0]), np.asarray(z4[1]), 1e-32, 0.05))
        held = band_coeffs(np.asarray(z4[0])[:M], np.asarray(z4[1])[:M], M, ks)
        hh, hl = band_field(held, ks, M)
        ps = PinnedSolver(s, M, hh, hl)
        sq = PinnedQ(s, ps.ch, ps.cl)
        try:
            F0 = sq.residual_q(*z4)
        except ValueError as e:                     # a size the quad-double solver is not built for
            print(f"M {M}: skipped ({e})", flush=True)
            continue
        f0 = float(np.abs(np.asarray(F0[0]) + np.asarray(F0[1])).max())
        st = sq.newton_q(*(np.ascontiguousarray(np.asarray(p)) for p in z4))
        lin = float(sum(mp.mpf(float(p[-1])) for p in st[:4]))       # the first Newton step's Delta, a linear estimate
        print(f"M {M}: start |F| {f0:.2e}, the first Newton step's |dz| {float(np.abs(np.asarray(st[0])).max()):.1e} "
              f"and dDelta {lin:+.3e} (linear estimate of Delta(M) - Delta of the start)", flush=True)
        z2, hist, conv = newton_qd(sq, z4, tol=tol, it_max=it, verbose=True, w=junk, junk=junk > 0)
        d = sum(mp.mpf(float(p[-1])) for p in z2)
        pmin, pmax = sq.newton_q(*(np.ascontiguousarray(np.asarray(p)) for p in z2))[-2:]   # the LU's pivots there
        res[M] = d
        ok &= bool(conv)
        print(f"M {M}: {'converged' if conv else 'NOT converged'} in {len(hist)} steps, |F| {hist[-1]['F'] if hist else f0:.2e}, "
              f"pivots {pmin:.1e}..{pmax:.1e}, Delta {mp.nstr(d, 25)}, minus M {M0}'s state {float(d - d0):+.3e}", flush=True)
    for M in Ms:
        if M != Ms[0] and M in res:
            print(f"Delta(M {M}) - Delta(M {Ms[0]}) = {float(res[M] - res[Ms[0]]):+.3e}")
    print(f"{'pass' if ok else 'FAIL'}: D {mp.nstr(D, 12)}, Ms {Ms}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

#!/usr/bin/env python3
"""How well the full system determines each held harmonic, at a pinned chain state (notes/HELD_HARMONICS.md 4).

The pinned system (kfreeze 3) replaces the matching of p's harmonics 5-13 at the horizon block by "p1's coefficient =
held value h". At its solution z(h) the full system's residual (kfreeze 13) is zero outside the horizon p rows and g(h)
there, g being those rows' harmonics 5-13: ten numbers. Block elimination of the full Newton system on these rows
(the bordering algorithm, Keller, SIAM J. Sci. Stat. Comput. 4 (1983) 573, doi:10.1137/0904039) gives the Schur
complement S = dg/dh, 10 x 10: column j is the full Jacobian applied to dz_j = J_pinned^-1 c(e_j), the pinned
solution's response to a unit change of held coefficient j (one quad-double LU, qlu_solve), by central differences of
the full quad-double residual at +-eps dz_j. The full Newton step is dh = -S^-1 g, and its Delta change s . dh with
s_j = Delta's component of dz_j. Truncating the SVD of S (Hansen, BIT 27 (1987) 534, doi:10.1007/BF01937276) keeps
the directions the full equations determine and holds the rest: dh_tau = -sum over sigma_i > tau sigma_1 of
v_i (u_i . g) / sigma_i, a linear estimate of "solve the well-determined held directions, hold the near-null ones".

    cd hp && python3 held_schur.py RAWDIR STATE [--threads=2] [--mutate=pinned_rows] [--plant=5a:1e-8,13a:1e-6]
        [--json=OUT]

--plant adds a held error dh to the state in the linear model (g + S dh): the planted truth at a full-system state,
where g = 0, is that the truncated step recovers the planted error's Delta in the solved directions and leaves the
held ones' part.

Check (exit 1 on failure): S from eps and eps/2 agree to 1e-6 relative in every singular value above 1e-12 sigma_1,
and sigma_1 > 1e-14 (a full system that sees its held rows). --mutate=pinned_rows takes g and S from the pinned kernel's
own rows, which do not contain the dropped matching (the control: sigma_1 is then 0 and the check fails).
"""
import json
import os
import sys
import time

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hpkernel.driver import make_solver, load_q, qmax, qd_axpy  # noqa: E402
from pinning import band_field  # noqa: E402

mp.mp.dps = 80
KS = [5, 7, 9, 11, 13]
LAB = [f"{k}{c}" for k in KS for c in "ab"]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    if len(args) < 2:
        sys.exit(__doc__)
    raw, st = args[0], args[1]
    z, m = load_q(os.path.join(raw, st))
    z = tuple(np.ascontiguousarray(np.asarray(p)) for p in z)
    D = mp.mpf(m["D_str"]) if "D_str" in m else mp.mpf(repr(float(m["D"])))
    M = int(m["M"])
    th = int(opts.get("threads", 2))

    def mk(kf):
        return make_solver(D, M, nodes=list(m["nodes"]), kt=m.get("kt", 36), kc=m.get("kc", 146), ks=m.get("ks", 130), threads=th, kfreeze=kf)
    sp = mk(3)
    sf = sp if opts.get("mutate") == "pinned_rows" else mk(13)
    segs = sp.plan(z[0], z[1], 1e-32, 0.05)
    sp.set_segs(segs)
    sf.set_segs(segs)
    L = len(z[0])
    r0 = ((L - 1) // (4 * M) - 1) * 4 * M
    rows = list(range(r0, r0 + M))
    zero = np.zeros(L)
    tht = [mp.pi * j / M for j in range(M)]

    def proj(parts):
        f = [mp.fsum(mp.mpf(float(np.asarray(p)[r])) for p in parts) for r in rows]
        out = []
        for k in KS:
            out.append(mp.fsum(f[j] * mp.cos(k * tht[j]) for j in range(M)) * 2 / M)
            out.append(mp.fsum(f[j] * mp.sin(k * tht[j]) for j in range(M)) * 2 / M)
        return out
    t0 = time.time()
    out = sp.newton_q(*z)
    print(f"{st}: D {mp.nstr(D, 10)}, pinned LU {time.time() - t0:.0f} s, pivots {out[6]:.1e}..{out[7]:.1e}", flush=True)
    Ff = sf.residual_q(*z)
    if opts.get("mutate") == "pinned_rows":
        # the pinned rows read P_high(3) p1 here (the kernel's own offset is zero); the held values subtracted
        Fp = [np.array(np.asarray(p), dtype=float) for p in Ff]
        hfield = [np.asarray(p)[:M] for p in z]
        for q in range(4):
            Fp[q][r0:r0 + M] -= hfield[q]
        Ff = Fp
    g0 = proj(Ff)
    print(f"  full residual max {qmax(Ff[:2]):.2e}; g (harmonics 5-13 of the horizon p rows) " + " ".join(mp.nstr(v, 3) for v in g0), flush=True)
    S = {1: mp.matrix(10, 10), 2: mp.matrix(10, 10)}
    sens = []
    for i, k in enumerate(KS):
        for j, ab in enumerate([(1.0, 0.0), (0.0, 1.0)]):
            col = 2 * i + j
            fh, fl = band_field([ab], [k], M)
            ch, cl = zero.copy(), zero.copy()
            ch[r0:r0 + M], cl[r0:r0 + M] = fh, fl
            x = tuple(np.ascontiguousarray(np.asarray(p)) for p in sp.qlu_solve(ch, cl, zero, zero))
            sens.append(mp.fsum(mp.mpf(float(p[-1])) for p in x))
            for div in (1, 2):
                eps = 2.0 ** -40 / float(np.max(np.abs(x[0]))) / div
                gp = proj(sf.residual_q(*qd_axpy(eps, *x, *z)))
                gm = proj(sf.residual_q(*qd_axpy(-eps, *x, *z)))
                if opts.get("mutate") == "pinned_rows":
                    gp = [a - (eps * (1 if r == col else 0)) for r, a in enumerate(gp)]
                    gm = [a + (eps * (1 if r == col else 0)) for r, a in enumerate(gm)]
                for r in range(10):
                    S[div][r, col] = (gp[r] - gm[r]) / (2 * mp.mpf(eps))
    plant = {}
    if opts.get("plant"):
        # a planted held error dh (KEY:VALUE,...): g becomes g + S dh, the residual of a pinned state held at
        # h_state + dh in the linear model, and the exact correction's dDelta is -s . dh
        for item in opts["plant"].split(","):
            key, val = item.split(":")
            plant[LAB.index(key)] = mp.mpf(val)
        for r in range(10):
            g0[r] += mp.fsum(S[1][r, c] * v for c, v in plant.items())
        print(f"  planted dh {opts['plant']}: the exact correction moves Delta by {mp.nstr(-mp.fsum(sens[c] * v for c, v in plant.items()), 6)}; "
              + ", ".join(f"{LAB[c]} part {mp.nstr(-sens[c] * v, 4)}" for c, v in plant.items()))
    U, sig, V = mp.svd_r(S[1])
    sig2 = mp.svd_r(S[2], compute_uv=False)
    smax = sig[0]
    agree = max([abs(sig[i] - sig2[i]) / sig[i] for i in range(10) if sig[i] > 1e-12 * smax] or [mp.inf])
    ok = bool(smax > 1e-14 and agree < 1e-6)
    print(f"  singular values of S (eps): " + " ".join(mp.nstr(v, 4) for v in sig))
    print(f"  eps vs eps/2: largest relative difference {mp.nstr(agree, 3)} over sigma > 1e-12 sigma_1")
    print("| sigma | direction (largest components) | dDelta per unit along v | u . g | full step along v | its dDelta |")
    print("|---|---|---|---|---|---|")
    rec = []
    for i in range(10):
        vi = [V[i, c] for c in range(10)]
        big = sorted(range(10), key=lambda c: -abs(vi[c]))[:2]
        sv = mp.fsum(sens[c] * vi[c] for c in range(10))
        ug = mp.fsum(U[r, i] * g0[r] for r in range(10))
        step = -ug / sig[i] if sig[i] else mp.inf
        print(f"| {mp.nstr(sig[i], 3)} | " + ", ".join(f"{LAB[c]} {mp.nstr(vi[c], 3)}" for c in big)
              + f" | {mp.nstr(sv, 3)} | {mp.nstr(ug, 3)} | {mp.nstr(step, 3)} | {mp.nstr(step * sv, 3)} |")
        rec.append(dict(sigma=str(sig[i]), dDelta_per_unit=str(sv), ug=str(ug)))
    print("| kept: sigma > tau sigma_1 | harmonics solved | dDelta of the truncated step | max abs dh |")
    print("|---|---|---|---|")
    trunc = {}
    for tau in (1e-2, 1e-5, 1e-8, 1e-10, 0.0):
        dt = [mp.mpf(0)] * 10
        kept = []
        for i in range(10):
            if sig[i] > tau * smax and sig[i] > 0:
                ug = mp.fsum(U[r, i] * g0[r] for r in range(10))
                for c in range(10):
                    dt[c] -= V[i, c] * ug / sig[i]
                kept.append(LAB[max(range(10), key=lambda c: abs(V[i, c]))][:-1])
        dd = mp.fsum(sens[c] * dt[c] for c in range(10))
        trunc[str(tau)] = str(dd)
        print(f"| {tau:g} | {', '.join(sorted(set(kept), key=int))} | {mp.nstr(dd, 6)} | {mp.nstr(max(abs(v) for v in dt), 3)} |")
    outp = opts.get("json")
    if outp:
        json.dump(dict(state=st, D=str(D), g0=[str(v) for v in g0], sens=[str(v) for v in sens], directions=rec, truncated=trunc,
                       S=[[str(S[1][r, c]) for c in range(10)] for r in range(10)]), open(outp, "w"), indent=1)
    print("check:", "pass" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

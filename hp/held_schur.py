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

    cd hp && python3 held_schur.py RAWDIR STATE [--threads=2] [--mutate=pinned_rows|jac_scale] [--plant=5a:1e-8,13a:1e-6]
        [--jac=exact] [--eps_pow=40] [--od] [--sigma_min=1e-14] [--json=OUT]

--plant adds a held error dh to the state in the linear model (g + S dh): the planted truth at a full-system state,
where g = 0, is that the truncated step recovers the planted error's Delta in the solved directions and leaves the
held ones' part.

Check (exit 1 on failure): S from eps and eps/2 agree to 1e-6 relative in every singular value above 1e-12 sigma_1,
and sigma_1 > 1e-14 (a full system that sees its held rows). --mutate=pinned_rows takes g and S from the pinned kernel's
own rows, which do not contain the dropped matching (the control: sigma_1 is then 0 and the check fails).

--jac=exact also forms S without differences, as the horizon p rows of the full system's exact quad-double Jacobian
(Solver.jacobian_q, forward-mode) applied to each dz_j, summed in mpmath. Its singular values are then the ones
reported and used by the table, and its check replaces the one above: sigma_1 > 1e-14, and the exact singular values
agree with those of S at eps to 1e-3 relative wherever eps and eps/2 agree to 1e-3 (the directions the differences
resolve). The finite-difference check is still printed. --mutate=jac_scale multiplies the exact rows by 1.01 (the
control: the agreement then fails). --eps_pow=P sets the difference step to 2^-P / max |dz_j| (default 40). Past D 80 the
residual's cubic term along dz_j outweighs S at P = 40 (S at eps/2 is then a quarter of S at eps); P = 70 keeps the
differences in quad-double's range there. --od does everything in octo-double (newton_o, olu_solve, residual_o,
jacobian_o; od_axpy from multidouble's Python package), from the state's eight parts (octo_corrector.load_o; a
four-part state gets zero lower parts). Past D 190, where the pinned LU's pivots span 10^42 and more, quad-double
breaks the cosine-sine pairs of S's singular values; the chain itself is corrected in octo-double there.
--sigma_min sets the floor of "a full system that sees its held rows" (sigma_1 > 1e-14 by default, set from the
states near D 80). The full equations' grip on harmonic 5 falls to 3e-17 at D 200, so past D 190 the floor is 1e-25,
still eight orders above the control --mutate=pinned_rows (sigma_1 about 1e-33).
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
    od = "--od" in sys.argv[1:]
    if od:
        from multidouble import od_axpy
        from octo_corrector import load_o
        z, m = load_o(os.path.join(raw, st))
    else:
        z, m = load_q(os.path.join(raw, st))
    z = tuple(np.ascontiguousarray(np.asarray(p)) for p in z)
    P = len(z)

    def residual(s_, zz):
        return s_.residual_o(list(zz)) if od else s_.residual_q(*zz)

    def axpy(a_, x_, zz):
        return od_axpy(a_, *x_, *zz) if od else qd_axpy(a_, *x_, *zz)
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
        return proj_f([mp.fsum(mp.mpf(float(np.asarray(p)[r])) for p in parts) for r in rows])

    def proj_f(f):
        out = []
        for k in KS:
            out.append(mp.fsum(f[j] * mp.cos(k * tht[j]) for j in range(M)) * 2 / M)
            out.append(mp.fsum(f[j] * mp.sin(k * tht[j]) for j in range(M)) * 2 / M)
        return out
    t0 = time.time()
    out = sp.newton_o(list(z)) if od else sp.newton_q(*z)
    piv = (out[3], out[4]) if od else (out[6], out[7])
    print(f"{st}: D {mp.nstr(D, 10)}, pinned LU ({'octo' if od else 'quad'}-double) {time.time() - t0:.0f} s, "
          f"pivots {piv[0]:.1e}..{piv[1]:.1e}", flush=True)
    Ff = residual(sf, z)
    if opts.get("mutate") == "pinned_rows":
        # the pinned rows read P_high(3) p1 here (the kernel's own offset is zero); the held values subtracted
        Fp = [np.array(np.asarray(p), dtype=float) for p in Ff]
        hfield = [np.asarray(p)[:M] for p in z]
        for q in range(P):
            Fp[q][r0:r0 + M] -= hfield[q]
        Ff = Fp
    g0 = proj(Ff)
    print(f"  full residual max {qmax(Ff[:2]):.2e}; g (harmonics 5-13 of the horizon p rows) " + " ".join(mp.nstr(v, 3) for v in g0), flush=True)
    S = {1: mp.matrix(10, 10), 2: mp.matrix(10, 10)}
    sens = []
    xs = []
    for i, k in enumerate(KS):
        for j, ab in enumerate([(1.0, 0.0), (0.0, 1.0)]):
            col = 2 * i + j
            fh, fl = band_field([ab], [k], M)
            ch, cl = zero.copy(), zero.copy()
            ch[r0:r0 + M], cl[r0:r0 + M] = fh, fl
            x = tuple(np.ascontiguousarray(np.asarray(p)) for p in
                      (sp.olu_solve([ch, cl] + [zero] * 6) if od else sp.qlu_solve(ch, cl, zero, zero)))
            xs.append(x)
            sens.append(mp.fsum(mp.mpf(float(p[-1])) for p in x))
            for div in (1, 2):
                eps = 2.0 ** -int(opts.get("eps_pow", 40)) / float(np.max(np.abs(x[0]))) / div
                gp = proj(residual(sf, axpy(eps, x, z)))
                gm = proj(residual(sf, axpy(-eps, x, z)))
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
    exact = opts.get("jac") == "exact"
    if exact:
        # S from the exact Jacobian: the horizon p rows of J_full times dz_j, in mpmath from the four parts
        t1 = time.time()
        Jq = [np.asarray(p) for p in (sf.jacobian_o(list(z)) if od else sf.jacobian_q(*z))]
        scale = mp.mpf("1.01") if opts.get("mutate") == "jac_scale" else mp.mpf(1)
        Jr = [[scale * mp.fsum(mp.mpf(float(Jq[q][r, c])) for q in range(P)) for c in range(L)] for r in rows]
        S[3] = mp.matrix(10, 10)
        for col, x in enumerate(xs):
            xm = [mp.fsum(mp.mpf(float(p[c])) for p in x) for c in range(L)]
            gx = proj_f([mp.fsum(Jr[i][c] * xm[c] for c in range(L) if xm[c]) for i in range(M)])
            if opts.get("mutate") == "pinned_rows":
                gx[col] -= 1
            for r in range(10):
                S[3][r, col] = gx[r]
        print(f"  exact Jacobian ({'jacobian_o' if od else 'jacobian_q'}) and S from it: {time.time() - t1:.0f} s", flush=True)
    U1, sig1, V1 = mp.svd_r(S[1])
    U2, sig2, V2 = mp.svd_r(S[2])
    smax = sig1[0]
    agree = max([abs(sig1[i] - sig2[i]) / sig1[i] for i in range(10) if sig1[i] > 1e-12 * smax] or [mp.inf])
    sig_min = float(opts.get("sigma_min", 1e-14))
    ok_fd = bool(smax > sig_min and agree < 1e-6)
    print(f"  singular values of S (eps): " + " ".join(mp.nstr(v, 4) for v in sig1))
    print(f"  singular values of S (eps/2): " + " ".join(mp.nstr(v, 4) for v in sig2))
    print(f"  eps vs eps/2, relative difference per singular value: " + " ".join(mp.nstr(abs(sig1[i] - sig2[i]) / sig1[i], 2) for i in range(10)))
    print(f"  eps vs eps/2: largest relative difference {mp.nstr(agree, 3)} over sigma > 1e-12 sigma_1"
          + ("" if ok_fd else " (the finite-difference check fails)"))
    if exact:
        U, sig, V = mp.svd_r(S[3])
        resolved = [i for i in range(10) if abs(sig1[i] - sig2[i]) / sig1[i] < 1e-3]
        agree_x = max([abs(sig[i] - sig1[i]) / sig[i] for i in resolved] or [mp.inf])
        ok = bool(sig[0] > sig_min and resolved and agree_x < 1e-3)
        print(f"  singular values of S (exact): " + " ".join(mp.nstr(v, 4) for v in sig))
        print(f"  exact vs eps, relative difference per singular value: " + " ".join(mp.nstr(abs(sig[i] - sig1[i]) / sig[i], 2) for i in range(10)))
        print(f"  exact vs eps: largest relative difference {mp.nstr(agree_x, 3)} over the {len(resolved)} values that eps and eps/2 "
              f"resolve to 1e-3")
        U2, sig2, V2 = U1, sig1, V1
    else:
        U, sig, V, ok = U1, sig1, V1, ok_fd
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
        rec.append(dict(sigma=str(sig[i]), dDelta_per_unit=str(sv), ug=str(ug), step=str(step),
                        harmonic=int(LAB[big[0]][:-1]), v={LAB[c]: str(vi[c]) for c in range(10)}))
    print("| kept: sigma > tau sigma_1 | harmonics solved | dDelta of the truncated step | the same from S at "
          + ("eps" if exact else "eps/2") + " | max abs dh |")
    print("|---|---|---|---|---|")
    trunc = {}

    def truncated(Ux, sx, Vx, tau):
        dt = [mp.mpf(0)] * 10
        kept = []
        for i in range(10):
            if sx[i] > tau * sx[0] and sx[i] > 0:
                ug = mp.fsum(Ux[r, i] * g0[r] for r in range(10))
                for c in range(10):
                    dt[c] -= Vx[i, c] * ug / sx[i]
                kept.append(LAB[max(range(10), key=lambda c: abs(Vx[i, c]))][:-1])
        return mp.fsum(sens[c] * dt[c] for c in range(10)), dt, kept
    for tau in (1e-2, 1e-5, 1e-8, 1e-10, 0.0):
        dd, dt, kept = truncated(U, sig, V, tau)
        dd2 = truncated(U2, sig2, V2, tau)[0]
        trunc[str(tau)] = [str(dd), str(dd2)]
        print(f"| {tau:g} | {', '.join(sorted(set(kept), key=int))} | {mp.nstr(dd, 6)} | {mp.nstr(dd2, 6)} | {mp.nstr(max(abs(v) for v in dt), 3)} |")
    outp = opts.get("json")
    if outp:
        json.dump(dict(state=st, D=str(D), g0=[str(v) for v in g0], sens=[str(v) for v in sens], directions=rec, truncated=trunc,
                       S=[[str(S[1][r, c]) for c in range(10)] for r in range(10)], jac="exact" if exact else "differences",
                       sigma_eps=[str(v) for v in sig1], sigma_eps2=[str(v) for v in (mp.svd_r(S[2])[1])],
                       sigma_exact=[str(v) for v in sig] if exact else None, check_fd="pass" if ok_fd else "FAIL",
                       S_exact=[[str(S[3][r, c]) for c in range(10)] for r in range(10)] if exact else None),
                  open(outp, "w"), indent=1)
    print("check:", "pass" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

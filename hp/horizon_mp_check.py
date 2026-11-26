"""Independent check of form S's horizon values (#1098): the fields (p, m, c) at the last node x_l that the horizon
series gives from a saved state's m_h and Delta, by the kernel (the u series, form_s.rs horizon_series_s, and the
Moebius series, horizon_mob, in octo-double) and by mpmath at 150 digits, which shares none of the kernel's arithmetic.
The mpmath series is horizon_series_s's recursion in the scaled u = (1 - x)/sigma, written again with exact solves of
its slaving and p systems, summed to --ks_mp orders (its tail printed); both use the kernel's double-double d/dtheta
matrices, read exactly, and omega = dd(2 pi)/Delta. The kernel's values are the last node's fields minus the horizon
block of the residual (kfreeze none, so that no row is projected).

Reported: the kernel's u sum at the state's ks against mpmath's at the same ks (the u recurrence's arithmetic), and the
kernel's u and Moebius values against mpmath's converged sum. PASS when the Moebius values agree with mpmath to --bar
(1e-40); the control moves m_h by 1e-30 in the kernel only, which must exceed --bar.

    python3 hp/horizon_mp_check.py STATE [--ks_mp 0] [--mob 8] [--ks_mob 0] [--dps 150] [--bar 1e-40]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mpmath import mp  # noqa: E402

import form_s  # noqa: E402
from octo_corrector import load_o  # noqa: E402
from taylor_mp_check import kernel_dmats  # noqa: E402


def opts(argv):
    o = dict(ks_mp="0", mob="8", ks_mob="0", dps="150", bar="1e-40", u_scan="", mob_scan="", split="0")
    pos = []
    i = 0
    while i < len(argv):
        if argv[i].startswith("--"):
            o[argv[i][2:]] = argv[i + 1]
            i += 2
        else:
            pos.append(argv[i])
            i += 1
    return pos, o


def mp_of(hi, lo):
    return [mp.mpf(float(a)) + mp.mpf(float(b)) for a, b in zip(np.ravel(hi), np.ravel(lo))]


def kernel_right(z8, meta, mob, ks, dmh=0.0):
    """the kernel's (p, m, c) at x_l from the horizon series (u when mob is 0), octo-double, as mpf (3M,)"""
    with mp.workdps(60):
        D = mp.mpf(meta["D_str"])
    M, nodes = int(meta["M"]), list(meta["nodes"])
    K = len(nodes)
    s = form_s.make_solver_s(D, M, nodes, kt=int(meta.get("kt", 36)), kc=int(meta["kc"]), ks=ks, threads=2,
                             kfreeze=None, segs=meta["segs"], mob=mob)
    z = [np.array(z8[q], dtype=float) for q in range(8)]
    if dmh:
        o = 2 * M + 3 * M * K
        z[1][o:o + M] += dmh
    F = s.residual_s([np.ascontiguousarray(p) for p in z])
    F = np.array([np.asarray(f) for f in F])
    on = 2 * M + 3 * M * (K - 1)
    out = []
    with mp.workdps(160):
        # the eight parts summed exactly (at the caller's precision a 53-bit sum would cap them at 1e-17)
        for i in range(3 * M):
            node = mp.fsum(mp.mpf(float(z[q][on + i])) for q in range(8))
            fh = mp.fsum(mp.mpf(float(F[q][K * 3 * M + i])) for q in range(8))
            out.append((node, fh))
    return out, s


def mp_horizon(z8, meta, ks_mp, sig, dps, split=False):
    """horizon_series_s in mpmath: the coefficients of p, m, c in the scaled u to order ks_mp"""
    M, nodes = int(meta["M"]), list(meta["nodes"])
    K = len(nodes)
    with mp.workdps(dps):
        D = mp.mpf(meta["D_str"])
        n = D - 2
        n1, nh = n - 1, n / 2
        vo, ve = kernel_dmats(M)
        Do, De = mp.matrix(M, M), mp.matrix(M, M)
        for i in range(M):
            for j in range(M):
                Do[i, j] = vo[i * M + j]
                De[i, j] = ve[i * M + j]
        tp = mp.mpf(float(mp.pi * 2))
        tp += mp.mpf(float(mp.pi * 2 - tp))
        delta = mp.fsum(mp.mpf(float(z8[q][-1])) for q in range(8))
        om = tp / delta
        o = 2 * M + 3 * M * K
        m0 = [mp.fsum(mp.mpf(float(z8[q][o + i])) for q in range(8)) for i in range(M)]
        sg, isg = mp.mpf(sig), 1 / mp.mpf(sig)
        zero = [mp.mpf(0)] * M

        def dth(f):
            return [om * mp.fsum(Do[i, j] * f[j] for j in range(M)) for i in range(M)]

        def cs(a, b, k, lo=0):
            return [mp.fsum(a[j][i] * b[k - j][i] for j in range(lo, k + 1)) for i in range(M)]

        def wfun(c, i):
            return [1 - v for v in c[0]] if i == 0 else ([-v - sg for v in c[1]] if i == 1 else [-v for v in c[i]])

        def zfun(c, i):
            return [v + 1 for v in c[0]] if i == 0 else ([v - sg for v in c[1]] if i == 1 else list(c[i]))

        L = ks_mp + 2
        pp, mm, cc = [list(zero) for _ in range(L)], [list(zero) for _ in range(L)], [list(zero) for _ in range(L)]
        bb, ia, qs, p2, m2 = ([list(zero) for _ in range(L)] for _ in range(5))
        dp, dm, pm, vv, ww, zc, wr, zr = ([list(zero) for _ in range(L)] for _ in range(8))
        cc[0] = [mp.mpf(1)] * M
        mm[0] = list(m0)
        m2[0] = [v * v for v in m0]
        dm[0] = dth(mm[0])
        ww[0], zc[0] = wfun(cc, 0), zfun(cc, 0)
        izc0 = [1 / v for v in zc[0]]
        lub = None
        for kk in range(ks_mp + 1):
            wr[kk], zr[kk] = wfun(cc, kk), zfun(cc, kk)
            s1, s2 = cs(wr, p2, kk), cs(zr, m2, kk)
            qs[kk] = [s1[i] + s2[i] + (sg * qs[kk - 1][i] if kk else 0) for i in range(M)]
            if kk == 0:
                lub = om * De
                for i in range(M):
                    lub[i, i] += -qs[0][i] - n1
                if split:
                    # horizon_mob's split B_0 = 1 + X, (omega D_e + kappa) X = q_0: equal to the direct solve only
                    # when D_e 1 = 0, which the double-double D_e meets to its row sums
                    x = mp.lu_solve(lub, mp.matrix(qs[0]))
                    x = mp.matrix([1 + x[i] for i in range(M)])
                else:
                    x = mp.lu_solve(lub, mp.matrix([-n1] * M))
            else:
                x = mp.lu_solve(lub, mp.matrix(cs(qs, bb, kk, 1)))
            bb[kk] = [x[i] for i in range(M)]
            if kk == 0:
                ia[0] = [1 / v for v in bb[0]]
            else:
                t = cs(bb, ia, kk, 1)
                ia[kk] = [-t[i] * ia[0][i] for i in range(M)]
            ca = cs(cc, ia, kk)
            ca1 = [ca[i] - cc[kk][i] for i in range(M)]
            cc[kk + 1] = [(cc[kk][i] * kk - ca1[i] * n1) / (kk + 1) * sg for i in range(M)]
            w1, w0 = wfun(cc, kk + 1), wfun(cc, kk)
            ww[kk + 1] = [w1[i] - sg * w0[i] for i in range(M)]
            vv[kk] = [cc[kk + 1][i] * (kk + 1) * isg - cc[kk][i] * kk for i in range(M)]
            z1, z0 = zfun(cc, kk + 1), zfun(cc, kk)
            zc[kk + 1] = [z1[i] - sg * z0[i] for i in range(M)]
            pm[kk] = [pp[kk][i] + mm[kk][i] for i in range(M)]
            wpu = [mp.fsum(j * pp[j][i] * ww[kk + 1 - j][i] for j in range(kk + 1)) * isg for i in range(M)]
            tau = [-sg * dp[kk - 1][i] for i in range(M)] if kk >= 1 else zero
            a1, a2, a3 = cs(vv, pp, kk), cs(cc, pp, kk), cs(cc, pm, kk)
            r = [tau[i] - wpu[i] + a1[i] + a2[i] - nh * a3[i] for i in range(M)]
            c1 = [v * isg for v in cc[1]]
            A = om * Do
            for i in range(M):
                A[i, i] += (c1[i] + 1) * kk + c1[i] + 1 - nh
            x = mp.lu_solve(A, mp.matrix([-v for v in r]))
            pp[kk] = [x[i] for i in range(M)]
            dp[kk] = dth(pp[kk])
            p2[kk] = cs(pp, pp, kk)
            if kk == ks_mp:
                break
            tau = [dm[kk][i] - sg * dm[kk - 1][i] for i in range(M)] if kk >= 1 else list(dm[kk])
            pm[kk] = [pp[kk][i] + mm[kk][i] for i in range(M)]
            zmu = [mp.fsum(j * mm[j][i] * zc[kk + 1 - j][i] for j in range(kk + 1)) * isg for i in range(M)]
            b1, b2, b3 = cs(vv, mm, kk), cs(cc, mm, kk), cs(cc, pm, kk)
            rest = [tau[i] - zmu[i] - b1[i] - b2[i] + nh * b3[i] for i in range(M)]
            mm[kk + 1] = [rest[i] * izc0[i] / (kk + 1) * sg for i in range(M)]
            dm[kk + 1] = dth(mm[kk + 1])
            m2[kk + 1] = cs(mm, mm, kk + 1)
        return pp[:ks_mp + 1], mm[:ks_mp + 1], cc[:ks_mp + 1]


def main():
    pos, o = opts(sys.argv[1:])
    z8, meta = load_o(pos[0])
    dps, ks = int(o["dps"]), int(meta["ks"])
    ks_mp = int(o["ks_mp"]) or 2 * ks
    mob = float(o["mob"])
    ks_mob = int(o["ks_mob"]) or form_s.ks_mob(float(meta["D"])) + 130
    if float(meta.get("hmob", 0.0) or 0.0):
        raise SystemExit("the state must be a u state (hmob 0): its own residual is the u series'")
    ku, s = kernel_right(z8, meta, 0.0, ks)
    sig = s.series_scale(1)
    M = int(meta["M"])
    K = len(meta["nodes"])
    with mp.workdps(dps):
        u = (1 - mp.mpf(float(meta["nodes"][-1]))) / mp.mpf(sig)
        P, Mm, C = mp_horizon(z8, meta, ks_mp, sig, dps, split=o["split"] == "1")
        ref, ref_ks, tail = [], [], 0
        for F in (P, Mm, C):
            for i in range(M):
                terms = [F[k][i] * u ** k for k in range(ks_mp + 1)]
                ref.append(mp.fsum(terms))
                ref_ks.append(mp.fsum(terms[:ks + 1]))
                tail = max(tail, max(abs(t) for t in terms[-10:]))
        print(f"mpmath u series to order {ks_mp} at u/sigma = {mp.nstr(u, 6)} (sigma {sig}): last ten terms at most "
              f"{mp.nstr(tail, 3)}", flush=True)
        # the residual's horizon rows: node - right or right - node, whichever matches mpmath at the state's ks
        dn = [max(abs((nd - fh) - r) for (nd, fh), r in zip(ku, ref_ks)), max(abs((nd + fh) - r) for (nd, fh), r in
                                                                                  zip(ku, ref_ks))]
        sgn = -1 if dn[0] <= dn[1] else 1
        kv = [nd + sgn * fh for nd, fh in ku]
        d_u_ks = max(abs(a - b) for a, b in zip(kv, ref_ks))
        d_u = max(abs(a - b) for a, b in zip(kv, ref))
        d_tr = max(abs(a - b) for a, b in zip(ref_ks, ref))
        print(f"kernel u (ks {ks}) against mpmath at ks {ks}: {mp.nstr(d_u_ks, 4)} (the u recurrence's arithmetic)",
              flush=True)
        print(f"mpmath at ks {ks} against ks {ks_mp}: {mp.nstr(d_tr, 4)} (u's truncation)", flush=True)
        print(f"kernel u (ks {ks}) against mpmath converged: {mp.nstr(d_u, 4)} (p, m, c: " + ", ".join(
            mp.nstr(max(abs(a - b) for a, b in zip(kv[f * M:(f + 1) * M], ref[f * M:(f + 1) * M])), 3) for f in range(3))
            + ")", flush=True)
        for kj in [int(v) for v in o.get("u_scan", "").split(",") if v]:
            kuj, _ = kernel_right(z8, meta, 0.0, kj)
            part = []
            for F in (P, Mm, C):
                for i in range(M):
                    part.append(mp.fsum(F[k][i] * u ** k for k in range(kj + 1)))
            dj = [max(abs((nd + sgn * fh) - b) for (nd, fh), b in zip(kuj[f * M:(f + 1) * M], part[f * M:(f + 1) * M]))
                  for f in range(3)]
            dc = max(abs((nd + sgn * fh) - b) for (nd, fh), b in zip(kuj, ref))
            print(f"  kernel u at ks {kj} against mpmath's partial sum to {kj}: p, m, c "
                  f"{', '.join(mp.nstr(v, 3) for v in dj)}; against the converged sum {mp.nstr(dc, 3)}", flush=True)
        for spec in [v for v in o.get("mob_scan", "").split(",") if v]:
            aj, kj = spec.split(":")
            kmj, _ = kernel_right(z8, meta, float(aj), int(kj))
            dj = [max(abs((nd + sgn * fh) - b) for (nd, fh), b in zip(kmj[f * M:(f + 1) * M], ref[f * M:(f + 1) * M]))
                  for f in range(3)]
            print(f"  kernel Moebius a {aj}, ks {kj} against mpmath converged: p, m, c "
                  f"{', '.join(mp.nstr(v, 3) for v in dj)}", flush=True)
        kmv, _ = kernel_right(z8, meta, mob, ks_mob)
        km = [nd + sgn * fh for nd, fh in kmv]
        d_m = max(abs(a - b) for a, b in zip(km, ref))
        rows = [max(abs(a - b) for a, b in zip(km[f * M:(f + 1) * M], ref[f * M:(f + 1) * M])) for f in range(3)]
        print(f"kernel Moebius (a {mob:g}, ks {ks_mob}) against mpmath converged: {mp.nstr(d_m, 4)} (p, m, c: "
              f"{', '.join(mp.nstr(r, 3) for r in rows)})", flush=True)
        kc_, _ = kernel_right(z8, meta, mob, ks_mob, dmh=1e-30)
        d_c = max(abs((nd + sgn * fh) - b) for (nd, fh), b in zip(kc_, ref))
        print(f"control (m_h + 1e-30 in the kernel only): {mp.nstr(d_c, 4)}", flush=True)
    ok = d_m <= float(o["bar"]) and d_c > float(o["bar"])
    print(f"  {'PASS' if ok else 'FAIL'}: Moebius against mpmath {mp.nstr(d_m, 3)} (bar {o['bar']}), the control "
          f"{mp.nstr(d_c, 3)} (must exceed it); {K} nodes, x_l {meta['nodes'][-1]}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

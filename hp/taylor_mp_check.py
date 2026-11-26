"""Independent check of form S's octo-double Taylor step (#1098): one step of a saved state's last node, by the kernel
(Solver.path_s, octo-double) and by mpmath at 150 digits, which shares none of the kernel's arithmetic (no
multidouble lanes, reciprocals, Cauchy sums or LU). The mpmath recursion is form_s.rs taylor_coeffs_s's equations
written again from their definitions:

    x q = x (p^2 + m^2) - c (p^2 - m^2),   (omega D_e - q - (n - 1)) B = -(n - 1),   a^2 = 1/B,
    x c' = g = (n - 1) c (a^2 - 1),
    x (x - c) p' = g p - x omega D_o p - c p + (n/2) c (p + m),
    x (x + c) m' = c m - x omega D_o m - g m - (n/2) c (p + m),

as series in h = x - x0 to order kt, summed at x1 - x0. Both use the same d/dtheta matrices, the kernel's stored
values read exactly (hi + lo, plus the f64 remainder c when the solver applies it: form_s.exact_dmat), and
omega = dd(2 pi)/Delta, so that the difference is the kernel's arithmetic alone; the matrices' own rounding is
hp/dmat_perturb.py's term. PASS when the largest difference over the 3M values is
at most --bar (1e-60), and the control, the kernel with double-double reciprocals (HP_EXACT_RECIP=0, the 1/x0 and
1/(k + 1) of every run before #1098's fix), must exceed --bar_ctl (1e-40).

    python3 hp/taylor_mp_check.py STATE [--h -0.02] [--dps 150] [--bar 1e-60] [--bar_ctl 1e-40]
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hpkernel.driver as drv  # noqa: E402
from mpmath import mp  # noqa: E402

import form_s  # noqa: E402
from octo_corrector import load_o  # noqa: E402


def opts(argv):
    o = dict(h="-0.02", dps="150", bar="1e-60", bar_ctl="1e-40", kernel_only="")
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


def remainders_applied():
    """whether make_solver_s gives the solver the d/dtheta remainders (form_s.exact_dmat and Solver.set_dmat_corr)"""
    from hpkernel._hpkernel import Solver
    return bool(getattr(form_s, "exact_dmat", lambda: False)()) and hasattr(Solver, "set_dmat_corr")


def kernel_dmats(M):
    """the odd and even d/dtheta exactly as the kernel's quad- and octo-double lanes apply them, as mpf lists
    (row-major): hi + lo, plus the f64 remainder c when it is applied (remainders_applied)"""
    oh, ol = drv.dmat_odd(M)
    eh, el = form_s.dmat_even(M)
    vo, ve = mp_of(oh, ol), mp_of(eh, el)
    if remainders_applied():
        co, ce = form_s.dmat_corr(M)
        vo = [v + mp.mpf(float(c)) for v, c in zip(vo, np.ravel(co))]
        ve = [v + mp.mpf(float(c)) for v, c in zip(ve, np.ravel(ce))]
    return vo, ve


def kernel_step(path, h):
    """the kernel's octo-double step from the state's last node to x_l + h: (3M,) mpf values, x_l, x1"""
    z8, meta = load_o(path)
    with mp.workdps(60):
        D = mp.mpf(meta["D_str"])
    M, nodes = int(meta["M"]), list(meta["nodes"])
    s = form_s.make_solver_s(D, M, nodes, kt=int(meta.get("kt", 36)), kc=int(meta["kc"]), ks=int(meta["ks"]),
                             threads=1, kfreeze=13, segs=meta["segs"], mob=float(meta.get("hmob", 0.0) or 0.0))
    x0 = float(nodes[-1])
    x1 = x0 + h
    P = s.path_s([np.ascontiguousarray(z8[q]) for q in range(8)], [x0, x1], [1])
    parts = np.array([np.asarray(p).ravel() for p in P])
    return parts


def mp_step(path, h, dps):
    z8, meta = load_o(path)
    M, nodes, kt = int(meta["M"]), list(meta["nodes"]), int(meta.get("kt", 36))
    K = len(nodes)
    with mp.workdps(dps):
        D = mp.mpf(meta["D_str"])
        n = D - 2
        n1, nh = n - 1, n / 2
        Do = mp.matrix(M, M)
        De = mp.matrix(M, M)
        vo, ve = kernel_dmats(M)
        for i in range(M):
            for j in range(M):
                Do[i, j] = vo[i * M + j]
                De[i, j] = ve[i * M + j]
        tp = mp.mpf(float(mp.pi * 2))
        tp += mp.mpf(float(mp.pi * 2 - tp))
        delta = mp.fsum(mp.mpf(float(z8[q][-1])) for q in range(8))
        om = tp / delta
        o = 2 * M + 3 * M * (K - 1)
        val = [mp.fsum(mp.mpf(float(z8[q][o + i])) for q in range(8)) for i in range(3 * M)]
        x0 = mp.mpf(float(nodes[-1]))
        hh = mp.mpf(float(nodes[-1]) + h) - x0
        P = [[val[i] for i in range(M)]]
        Mm = [[val[M + i] for i in range(M)]]
        C = [[val[2 * M + i] for i in range(M)]]
        Q, Bs, IA, G, D2, Q2, W, V, TP, TM = [], [], [], [], [], [], [], [], [], []

        def cs(a, b, k, lo=0):
            return [mp.fsum(a[j][i] * b[k - j][i] for j in range(lo, k + 1)) for i in range(M)]

        def dth(f):
            return [om * mp.fsum(Do[i, j] * f[j] for j in range(M)) for i in range(M)]

        x2 = [x0 * x0, 2 * x0, mp.mpf(1)]
        lu_a = None
        for k in range(kt):
            pp2, mm2 = cs(P, P, k), cs(Mm, Mm, k)
            Q2.append([a + b for a, b in zip(pp2, mm2)])
            D2.append([a - b for a, b in zip(pp2, mm2)])
            cd = cs(C, D2, k)
            s = [x0 * Q2[k][i] - cd[i] + (Q2[k - 1][i] if k else 0) for i in range(M)]
            Q.append([(s[i] - (Q[k - 1][i] if k else 0)) / x0 for i in range(M)])
            if k == 0:
                lu_a = om * De
                for i in range(M):
                    lu_a[i, i] += -Q[0][i] - n1
                rhs = mp.matrix([-n1] * M)
            else:
                rhs = mp.matrix(cs(Q, Bs, k, 1))
            x = mp.lu_solve(lu_a, rhs)
            Bs.append([x[i] for i in range(M)])
            if k == 0:
                IA.append([1 / b for b in Bs[0]])
            else:
                t = cs(Bs, IA, k, 1)
                IA.append([-t[i] * IA[0][i] for i in range(M)])
            ca = cs(C, IA, k)
            G.append([n1 * (ca[i] - C[k][i]) for i in range(M)])
            TP.append(dth(P[k]))
            TM.append(dth(Mm[k]))
            xp = [x0 * TP[k][i] + (TP[k - 1][i] if k else 0) for i in range(M)]
            xm = [x0 * TM[k][i] + (TM[k - 1][i] if k else 0) for i in range(M)]
            cp, cm = cs(C, P, k), cs(C, Mm, k)
            gp, gm = cs(G, P, k), cs(G, Mm, k)
            rp = [gp[i] - xp[i] - cp[i] + nh * (cp[i] + cm[i]) for i in range(M)]
            rm = [cm[i] - xm[i] - gm[i] - nh * (cp[i] + cm[i]) for i in range(M)]
            C.append([(G[k][i] - k * C[k][i]) / (x0 * (k + 1)) for i in range(M)])
            xc = [x0 * C[k][i] + (C[k - 1][i] if k else 0) for i in range(M)]
            W.append([(x2[k] if k <= 2 else 0) - xc[i] for i in range(M)])
            V.append([(x2[k] if k <= 2 else 0) + xc[i] for i in range(M)])
            sp = [mp.fsum((k + 1 - j) * W[j][i] * P[k + 1 - j][i] for j in range(1, k + 1)) for i in range(M)]
            sm = [mp.fsum((k + 1 - j) * V[j][i] * Mm[k + 1 - j][i] for j in range(1, k + 1)) for i in range(M)]
            P.append([(rp[i] - sp[i]) / (W[0][i] * (k + 1)) for i in range(M)])
            Mm.append([(rm[i] - sm[i]) / (V[0][i] * (k + 1)) for i in range(M)])
        out = []
        for F in (P, Mm, C):
            for i in range(M):
                out.append(mp.fsum(F[k][i] * hh ** k for k in range(kt + 1)))
        return out


def main():
    pos, o = opts(sys.argv[1:])
    path, h = pos[0], float(o["h"])
    if o["kernel_only"]:
        parts = kernel_step(path, h)
        np.save(o["kernel_only"], parts)
        return 0
    dps = int(o["dps"])
    ref = mp_step(path, h, dps)
    res = {}
    for er in ("1", "0"):
        with tempfile.TemporaryDirectory() as td:
            tmp = os.path.join(td, "kernel.npy")
            env = dict(os.environ, HP_EXACT_RECIP=er)
            subprocess.run([sys.executable, os.path.abspath(__file__), path, "--h", o["h"], "--kernel_only", tmp],
                           env=env, check=True)
            parts = np.load(tmp)
        with mp.workdps(dps):
            ker = [mp.fsum(mp.mpf(float(parts[q][i])) for q in range(parts.shape[0])) for i in range(parts.shape[1])]
            res[er] = max(abs(a - b) for a, b in zip(ker, ref))
            big = max(abs(b) for b in ref)
        print(f"HP_EXACT_RECIP={er}: max |kernel - mpmath| {mp.nstr(res[er], 4)} over {len(ref)} values (largest "
              f"|value| {mp.nstr(big, 4)})", flush=True)
    print(f"the matrices on both sides: {'hi + lo + c (the remainders applied)' if remainders_applied() else 'hi + lo'}",
          flush=True)
    ok = res["1"] <= float(o["bar"]) and res["0"] > float(o["bar_ctl"])
    print(f"  {'PASS' if ok else 'FAIL'}: lane-precision reciprocals {mp.nstr(res['1'], 3)} (bar {o['bar']}), the "
          f"control {mp.nstr(res['0'], 3)} (must exceed {o['bar_ctl']})", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

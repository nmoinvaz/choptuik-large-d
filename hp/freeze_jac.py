#!/usr/bin/env python3
"""How much the freeze projector's double-double rounding moves the pinned Jacobian on its weakest directions (#562).

    cd hp && python3 freeze_jac.py RAWDIR STATE SETUP [--threads=2] [--h=1e-30] [--json=OUT] [--mutate=scale] [--uw=1]

The kernel's frozen horizon rows of p read Pi_low d + Pi_high p1 with the projectors rounded to double-double
(freeze_round.py). Their error e(z) = E_lo d(z) + E_hi p1(z), E = rounded minus exact, is linear in d and p1, so the
rounded system's Jacobian differs from the exact one by dJ = E_lo dd/dz + E_hi dp1/dz, with dd/dz the unfrozen
kernel's horizon p rows (Solver.jacobian_o, summed at 140 digits). SETUP is a held_homotopy.py OUT_setup.npz of the
same state: its V (the k weakest right singular vectors) and their sigma. Printed: ||dJ v_i|| against sigma_i, and the
2-norm of dJ (f64), which bounds how far each singular value of the rounded system's J can lie from the exact one's
(Weyl's inequality).

--uw=1 factors the rounded system's J at the state (Solver.newton_o) and forms the weak block's left vectors u_i =
J^-T v_i / |J^-T v_i|, so that U^T J V = diag(sigma) for singular pairs. The exact rows' block is then diag(sigma) -
U^T dJ V, a first-order model (its coupling to the strong directions, about |dJ| over the gap sigma_5 / sigma_4, is
dropped). Printed: U^T dJ V and both blocks' singular values. Its check: 1 / |J^-T v_i| equals the setup's sigma_i to
1e-3.

Check (exit 1): dJ v_i from the Jacobian equals the change of the emulated e along v_i, (e(z + h v_i) - e(z)) / h
from two unfrozen residuals at 160 digits, to 1e-3 relative for every i. Control --mutate=scale reports dJ v_i 1.01
times too large to the comparison, and the check must fail.
"""
import json
import os
import sys
import time

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from freeze_round import matvec, mpv  # noqa: E402
from hpkernel.driver import make_solver, plow_odd  # noqa: E402
from multidouble import dd_add, od_axpy  # noqa: E402
from octo_corrector import load_o, o8  # noqa: E402


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    if len(args) != 3:
        sys.exit(__doc__)
    raw, st, setup = args
    mut = opts.get("mutate", "")
    mp.mp.dps = 160
    z, m = load_o(os.path.join(raw, st))
    M, nodes = int(m["M"]), list(m["nodes"])
    K, kf, L = len(nodes), int(m.get("kfreeze", 3)), len(z[0])
    u = make_solver(m.get("D_str", repr(float(m["D"]))), M, nodes=nodes, kt=m.get("kt", 36), kc=m.get("kc", 146),
                    ks=m.get("ks", 130), threads=int(opts.get("threads", 2)), kfreeze=None)
    u.set_segs(m["segs"])
    hp = list(range(4 * M * K, 4 * M * K + M))
    S = np.load(setup)
    V, sig = S["V"], S["sig"]
    a, b = plow_odd(M, kf)
    hh, hl = dd_add(np.eye(M).ravel(), np.zeros(M * M), -a.ravel(), -b.ravel())
    N = 2 * M
    th = [2 * mp.pi * j / N for j in range(M)]
    Pex = [[mp.fsum(4 * mp.cos(k * (th[i] - th[j])) / N for k in range(1, kf + 1, 2)) for j in range(M)] for i in range(M)]
    Elo = [[mp.mpf(float(a[i, j])) + mp.mpf(float(b[i, j])) - Pex[i][j] for j in range(M)] for i in range(M)]
    Ehi = [[mp.mpf(float(hh[i * M + j])) + mp.mpf(float(hl[i * M + j])) - ((1 if i == j else 0) - Pex[i][j])
            for j in range(M)] for i in range(M)]
    print(f"{st}: D {float(m['D']):.10g}, nodes {nodes}, L {L}; max|E_lo| "
          f"{mp.nstr(max(abs(x) for r in Elo for x in r), 3)}, max|E_hi| {mp.nstr(max(abs(x) for r in Ehi for x in r), 3)}",
          flush=True)
    t0 = time.time()
    J = [np.asarray(p) for p in u.jacobian_o(list(z))]
    Jh = [[mp.fsum(mp.mpf(float(J[q][r, c])) for q in range(8)) for c in range(L)] for r in hp]
    print(f"  unfrozen Jacobian's horizon p rows ({time.time() - t0:.0f} s): max entry "
          f"{mp.nstr(max(abs(x) for r in Jh for x in r), 3)}", flush=True)

    def e_of(zz):
        Fu = o8(u.residual_o(list(zz)))
        return [x + y for x, y in zip(matvec(Elo, mpv(Fu, hp)), matvec(Ehi, mpv(zz, range(M))))]
    h = float(opts.get("h", 1e-30))
    e0 = e_of(z)
    rec = dict(state=st, D=float(m["D"]), h=h, mutate=mut, vectors=[])
    ok = True
    dJV = np.zeros((L, V.shape[1]))
    for i in range(V.shape[1]):
        v = [mp.mpf(float(x)) for x in V[:, i]]
        dv = [x + y for x, y in zip(matvec(Elo, matvec(Jh, v)), matvec(Ehi, v[:M]))]
        dJV[hp, i] = [float(x) for x in dv]
        n = mp.sqrt(mp.fsum(x * x for x in dv))
        zh = o8(od_axpy(h, *o8([np.ascontiguousarray(V[:, i])] + [np.zeros(L)] * 7), *z))
        fd = [(x - y) / h for x, y in zip(e_of(zh), e0)]
        sc = mp.mpf("1.01") if mut == "scale" else 1
        err = mp.sqrt(mp.fsum((sc * x - y) ** 2 for x, y in zip(dv, fd))) / mp.sqrt(mp.fsum(y * y for y in fd))
        good = bool(err <= mp.mpf("1e-3"))
        ok &= good
        rec["vectors"].append(dict(sigma=float(sig[i]), dJv=float(n), ratio=float(n / sig[i]), fd_rel_err=float(err),
                                   lo_part=float(mp.sqrt(mp.fsum(x * x for x in matvec(Elo, matvec(Jh, v)))))))
        print(f"  v_{i + 1}: sigma {sig[i]:.3e}, ||dJ v|| {mp.nstr(n, 4)} ({mp.nstr(n / sig[i], 3)} sigma; E_lo dd/dz part "
              f"{rec['vectors'][-1]['lo_part']:.3e}); against the residuals' difference {mp.nstr(err, 2)} "
              f"({'ok' if good else 'FAIL'})", flush=True)
    # dJ in f64 for its norm: E_lo (M x M) times the rows (M x L), plus E_hi on p1's columns
    dJ = np.array([[float(x) for x in r] for r in Elo]) @ np.array([[float(x) for x in r] for r in Jh])
    dJ[:, :M] += np.array([[float(x) for x in r] for r in Ehi])
    nrm = float(np.linalg.norm(dJ, 2))
    rec["dJ_norm2"] = nrm
    print(f"  ||dJ||_2 {nrm:.3e} (f64): every singular value of the rounded system's J lies within it of the exact one's",
          flush=True)
    if opts.get("uw") == "1":
        # the weak block of the exact system: with u_i = J^-T v_i / |J^-T v_i| (J v_i = sigma_i u_i for singular
        # pairs), U^T J V = diag(sigma) and U^T J_exact V = diag(sigma) - U^T dJ V
        s = make_solver(m.get("D_str", repr(float(m["D"]))), M, nodes=nodes, kt=m.get("kt", 36), kc=m.get("kc", 146),
                        ks=m.get("ks", 130), threads=int(opts.get("threads", 2)), kfreeze=kf)
        s.set_segs(m["segs"])
        t0 = time.time()
        s.newton_o(list(z))
        Y = np.column_stack([np.asarray(s.olu_solve([np.ascontiguousarray(V[:, i])] + [np.zeros(L)] * 7, True)[0])
                             for i in range(V.shape[1])])
        sig2 = 1 / np.linalg.norm(Y, axis=0)
        U = Y * sig2
        ok_s = bool(np.all(np.abs(sig2 / sig - 1) < 1e-3))
        ok &= ok_s
        W = U.T @ dJV
        B = np.diag(sig2)
        sv_dd, sv_ex = np.linalg.svd(B, compute_uv=False), np.linalg.svd(B - W, compute_uv=False)
        rec.update(sigma_transpose=[float(x) for x in sig2], W=W.tolist(), sv_exact_block=[float(x) for x in sv_ex],
                   U_orth=float(np.abs(U.T @ U - np.eye(V.shape[1])).max()))
        print(f"  LU {time.time() - t0:.0f} s; sigma from J^-T v: " + ", ".join(f"{x:.4e}" for x in sig2)
              + f" ({'ok' if ok_s else 'FAIL'} against the setup's to 1e-3); max|U^T U - I| {rec['U_orth']:.1e}", flush=True)
        print("  W = U^T dJ V:\n" + "\n".join("    " + " ".join(f"{x:+.3e}" for x in row) for row in W), flush=True)
        print("  singular values of the weak block, rounded rows (diag sigma): " + ", ".join(f"{x:.3e}" for x in sv_dd)
              + "; exact rows (diag sigma - W): " + ", ".join(f"{x:.3e}" for x in sv_ex), flush=True)
    rec["ok"] = bool(ok)
    print("check:", "pass" if ok else "FAIL")
    if opts.get("json"):
        json.dump(rec, open(opts["json"], "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

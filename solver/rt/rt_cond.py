"""Condition number of the RT prototype's Jacobian at solved states (rt_run.py's npz files).

    python3 solver/rt/rt_cond.py OUT.jsonl STATE.npz ... [--svd]     (one record per state)
    python3 solver/rt/rt_cond.py --selftest [--mutate=transpose]      (planted matrices; must pass, and fail mutated)

kappa = sigma_max / sigma_min of J (the matrix Newton factors: rows as rt_model.residual, columns the unknowns),
sigma_max by power iteration on J^T J, sigma_min by inverse iteration with the LU of J (J^-T J^-1); with --svd also
the full singular spectrum's bottom. kappa_eq is the same after Ruiz equilibration (rows and columns scaled to unit
max norm, 10 sweeps), the scaling-free measure. In f64 a computed sigma_min below about 1e-15 sigma_max is the
rounding of J itself, not the operator's: such values are flagged (resolved = false).
For the weakest right singular vector the record names the field and the t harmonic holding most of its norm.
Selftest: J = U diag(s) V^T with s from 1 to 1e-10 (and a Ruiz-sensitive row scaling); kappa must come out to 1e-6
relative. --mutate=transpose uses J^-1 J^-1 in the inverse iteration (the wrong singular value) and must fail.
"""
import os
for _v in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import sys
import json
import time
import numpy as np
import scipy.linalg as sl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
MUT = None
for a in sys.argv[1:]:
    if a.startswith("--mutate="):
        MUT = a.split("=", 1)[1]


def extremes(J, it=60, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=J.shape[1])
    for _ in range(it):
        w = J.T @ (J @ v)
        v = w / np.linalg.norm(w)
    smax = np.linalg.norm(J @ v)
    lu = sl.lu_factor(J)
    v = rng.normal(size=J.shape[1])
    for _ in range(it):
        y = sl.lu_solve(lu, v, trans=0 if MUT == "transpose" else 1)     # J^-T v
        w = sl.lu_solve(lu, y)                                            # J^-1 J^-T v = (J^T J)^-1 v
        v = w / np.linalg.norm(w)
    smin = np.linalg.norm(J @ v)
    return smax, smin, v, lu


def ruiz(J, sweeps=10):
    A = J.copy()
    r = np.ones(J.shape[0])
    c = np.ones(J.shape[1])
    for _ in range(sweeps):
        dr = 1 / np.sqrt(np.abs(A).max(axis=1))
        dc = 1 / np.sqrt(np.abs(A).max(axis=0))
        A = dr[:, None] * A * dc[None, :]
        r *= dr
        c *= dc
    return A


def selftest():
    rng = np.random.default_rng(1)
    n = 400
    U, _ = np.linalg.qr(rng.normal(size=(n, n)))
    V, _ = np.linalg.qr(rng.normal(size=(n, n)))
    s = np.logspace(0, -10, n)
    J = (U * s) @ V.T
    smax, smin, _, _ = extremes(J)
    k = smax / smin
    ok = abs(k / 1e10 - 1) < 1e-6
    print(f"selftest: planted kappa 1e10, measured {k:.6e}  {'ok' if ok else 'FAIL'}")
    D = np.diag(np.logspace(0, 6, n))
    A = ruiz(D @ J)
    ka = np.linalg.cond(A)
    ok2 = ka < 1e11
    print(f"selftest: rows scaled over 1e6, Ruiz-equilibrated kappa {ka:.3e} (< 1e11)  {'ok' if ok2 else 'FAIL'}")
    sys.exit(0 if ok and ok2 else 1)


def weakest_label(rt, v):
    w1, w2, w3, w4, mu = rt.unpack(v)
    best = None
    for name, f, anti in (("omega_1", w1, False), ("omega_2", w2, False), ("omega_3", w3, False),
                          ("omega_4", w4, True)):
        g = f[0] * (np.exp(-0.5j * rt.t)[:, None] if anti else 1)
        G = np.abs(np.fft.fft(g, axis=0)) ** 2
        e = G.sum(axis=1)
        k = np.fft.fftfreq(rt.Nt, 1.0 / rt.Nt) + (0.5 if anti else 0.0)
        tot = e.sum()
        kk = int(np.argmax(e))
        # t-harmonic in units of Delta's frequency (t has period 2 pi for Delta / 2): 2 |k|
        if best is None or tot > best[1]:
            best = (name, tot, float(2 * abs(k[kk])))
    xi_share = None
    return dict(field=best[0], harmonic_of_Delta=best[2], mu_share=float(abs(mu[0, 0, 0])))


def main():
    from rt_model import RT
    if "--selftest" in sys.argv:
        selftest()
    out = sys.argv[1]
    files = [f for f in sys.argv[2:] if not f.startswith("--")]
    grid = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--grid=")]
    from rt_scan import localize
    for f in files:
        d = np.load(f)
        rt = RT(float(d["D"]), int(d["Nt"]), int(d["N"]))
        z = d["z"]
        if grid:                                  # resampled to another grid (not re-solved there)
            new = RT(float(d["D"]), *(int(v) for v in grid[0].split("x")))
            z = rt.resample_to(z, new)
            rt = new
        t0 = time.time()
        F = np.abs(rt.residual(z)[0]).max()
        J = rt.jacobian(z)
        smax, smin, v, lu = extremes(J)
        piv = np.abs(np.diag(lu[0]))
        A = ruiz(J)
        smax_e, smin_e, _, _ = extremes(A)
        rec = dict(state=f, D=float(d["D"]), grid=[rt.Nt, rt.N], L=rt.L, F=float(F), sigma_max=float(smax),
                   sigma_min=float(smin), kappa=float(smax / smin), kappa_eq=float(smax_e / smin_e),
                   resolved=bool(smin > 1e-14 * smax), resolved_eq=bool(smin_e > 1e-14 * smax_e),
                   pivots=[float(piv.min()), float(piv.max())], weakest=weakest_label(rt, v), loc=localize(rt, v),
                   resampled=bool(grid))
        if "--svd" in sys.argv:
            s = np.linalg.svd(J, compute_uv=False)
            rec["svd_bottom"] = [float(x) for x in s[-8:]]
            rec["svd_top"] = float(s[0])
        rec["secs"] = round(time.time() - t0, 1)
        with open(out, "a") as fo:
            fo.write(json.dumps(rec) + "\n")
        print(f"{f}: D {rec['D']} grid {rt.Nt}x{rt.N} L {rt.L} |F| {F:.1e} sigma {smax:.3e} .. {smin:.3e} kappa "
              f"{rec['kappa']:.3e} (eq {rec['kappa_eq']:.3e}) pivots {piv.min():.1e}..{piv.max():.1e} weakest "
              f"{rec['weakest']} {rec['secs']} s", flush=True)


if __name__ == "__main__":
    main()

"""Condition number of the RT prototype's Jacobian where f64 cannot resolve it: the Jacobian in arb (exact central
differences of rt_mp's residual, which is quadratic), sigma_min = 1 / |J^-1|_2 from an arb approximate inverse and
power iteration, at two precisions that must agree.

    python3 solver/rt/rt_cond_mp.py OUT.jsonl --profile PROF.npz --grid 16x21 [--prec 512]   (our state, mapped)
    python3 solver/rt/rt_cond_mp.py OUT.jsonl --state RT.npz --grid 16x21 [--prec 512]       (an RT solution)
    python3 solver/rt/rt_cond_mp.py --selftest [--mutate=lowprec]

The state is either an hp profile mapped into RT gauge (gg_to_rt.GGMap, on the given grid; the Jacobian is then that
of the collocation at the true solution's grid values, not at the discrete solution) or an rt_run.py state resampled
to the grid. kappa = sigma_max / sigma_min, sigma_max by power iteration on the f64 rounding of J (well conditioned).
resolved: sigma_min at prec and at prec + 128 bits agree to 1e-6 relative.
Selftest: an arb matrix U diag(s) V^T with s from 1 to 1e-40 (U, V products of Householder reflections, exact in
arb); the measured kappa must be 1e40 to 1e-6. --mutate=lowprec runs it at 64 bits, which cannot resolve it: the two
precisions then disagree or kappa is off, and the selftest must fail.
"""
import os
for _v in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import sys
import json
import time
import argparse
import numpy as np
import flint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
arb, arb_mat = flint.arb, flint.arb_mat


def norm2_inv(A, prec, it=40, seed=0, k=4):
    """|A^-1|_2 at prec bits: X = A^-1 (approximate, arb), then block power iteration on X^T X with k vectors,
    orthonormalized in arb. Returns the largest; norm2_inv.all holds the k largest singular values of X (the k
    smallest of A, inverted) and norm2_inv.vecs the corresponding right singular vectors of A (unit, f64)."""
    flint.ctx.prec = prec
    L = A.nrows()
    I = arb_mat(L, L, [1 if i == j else 0 for i in range(L) for j in range(L)])
    X = A.solve(I, algorithm="approx")
    XT = X.transpose()
    rng = np.random.default_rng(seed)
    V = [arb_mat(L, 1, [arb(float(x)) for x in rng.normal(size=L)]) for _ in range(k)]
    dot = lambda u, v: (u.transpose() * v)[0, 0]

    def orth(W):
        Q = []
        for w in W:
            for q in Q:
                w = w - q * dot(q, w)
            Q.append(w * (1 / dot(w, w).sqrt()))
        return Q
    V = orth(V)
    for _ in range(it):
        V = orth([XT * (X * v) for v in V])
    XV = [X * v for v in V]
    # the converged, Gram-Schmidt ordered block: |X v_j| are the k largest singular values of X in order (no
    # Rayleigh-Ritz rotation: their ratios can exceed f64's range of an eigensolver)
    sig = [dot(x, x).sqrt() for x in XV]
    norm2_inv.all = [float(s.mid()) for s in sig]
    norm2_inv.vecs = []
    for x, s in zip(XV, sig):
        u = np.array([float((x[i, 0] / s).mid()) for i in range(L)])
        norm2_inv.vecs.append(u / np.linalg.norm(u))
    norm2_inv.weak = norm2_inv.vecs[0]
    # the largest, directly in arb (no f64 Gram matrix): |X v_1|
    return float(dot(XV[0], XV[0]).sqrt().mid())


def phys_kappa(smax, lowest, Nt):
    """kappa with the smallest singular value whose vector is not the Nyquist artifact of the collocation (a periodic
    field at t harmonic Nt / 2, i.e. harmonic Nt of Delta, whose discrete t derivative is zero), or None."""
    ph = [l for l in lowest if not (l["weakest"]["field"] != "omega_4" and l["weakest"]["harmonic_of_Delta"] == Nt)]
    return float(smax / ph[0]["sigma"]) if ph else None


def to_arb_mat(J_obj, prec):
    flint.ctx.prec = prec
    L = J_obj.shape[0]
    return arb_mat(L, L, [J_obj[i, j] for i in range(L) for j in range(L)])


def jac_arb(rm, z):
    """Columns (F(z + e_k) - F(z - e_k)) / 2 in arb (exact for the quadratic residual)."""
    from rt_mp import obj
    zo = obj(z)
    L = len(z)
    J = np.empty((L, L), dtype=object)
    one = arb(1)
    for k in range(L):
        zp = zo.copy()
        zm = zo.copy()
        zp[k] = zp[k] + one
        zm[k] = zm[k] - one
        J[:, k] = (rm.residual_m(zp) - rm.residual_m(zm)) / 2
    return J


def selftest(prec):
    flint.ctx.prec = prec
    n = 60
    rng = np.random.default_rng(3)

    def householder(u):
        u = [arb(float(x)) for x in u]
        nn = sum((x * x for x in u), arb(0))
        return arb_mat(n, n, [(1 if i == j else 0) - 2 * u[i] * u[j] / nn for i in range(n) for j in range(n)])
    U = householder(rng.normal(size=n)) * householder(rng.normal(size=n))
    V = householder(rng.normal(size=n)) * householder(rng.normal(size=n))
    S = arb_mat(n, n, [arb(10) ** (-40 * arb(i) / (n - 1)) if i == j else 0 for i in range(n) for j in range(n)])
    A = U * S * V.transpose()
    Af = np.array([[float(A[i, j].mid()) for j in range(n)] for i in range(n)])
    smax = np.linalg.norm(Af, 2)
    k1 = smax * norm2_inv(A, prec)
    k2 = smax * norm2_inv(A, prec + 128)
    ok = abs(k1 / 1e40 - 1) < 1e-6 and abs(k1 / k2 - 1) < 1e-6
    print(f"selftest at {prec} bits: planted kappa 1e40, measured {k1:.8e} (at {prec + 128} bits {k2:.8e})  "
          f"{'ok' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?")
    ap.add_argument("--profile")
    ap.add_argument("--state")
    ap.add_argument("--grid", default="16x21")
    ap.add_argument("--prec", type=int, default=512)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mutate", default="")
    a = ap.parse_args()
    if a.selftest:
        selftest(64 if a.mutate == "lowprec" else 256)
    from rt_model import RT
    from rt_mp import RTmp
    from rt_cond import extremes, weakest_label
    from rt_scan import localize
    Nt, N = (int(v) for v in a.grid.split("x"))
    t0 = time.time()
    if a.profile:
        from gg_to_rt import GGMap
        g = GGMap(a.profile, verbose=False)
        D = g.D
        rt = RT(D, Nt, N)
        z, err, _ = g.to_rt_phased(rt)
        src = a.profile
    else:
        d = np.load(a.state)
        D = float(d["D"])
        rt = RT(D, Nt, N)
        z = RT(D, int(d["Nt"]), int(d["N"])).resample_to(d["z"], rt)
        src = a.state
    F = float(np.abs(rt.residual(z)[0]).max())
    rm = RTmp(D, Nt, N, prec=a.prec + 128)
    Jo = jac_arb(rm, z)
    Jf = np.array([[float(v.mid()) for v in row] for row in Jo])
    smax, smin_f64, _, _ = extremes(Jf)
    t1 = time.time()
    n1 = norm2_inv(to_arb_mat(Jo, a.prec), a.prec)
    n2 = norm2_inv(to_arb_mat(Jo, a.prec + 128), a.prec + 128)
    smin = 1 / n2
    lowest = [dict(sigma=1 / s, loc=localize(rt, v), weakest=weakest_label(rt, v))
              for s, v in zip(norm2_inv.all, norm2_inv.vecs)]
    rec = dict(src=src, D=D, grid=[Nt, N], L=rt.L, F_f64=F, prec=a.prec, sigma_max=float(smax), sigma_min=smin,
               sigma_min_at_prec=1 / n1, kappa=float(smax * n2), kappa_f64=float(smax / smin_f64),
               resolved=bool(abs(n1 / n2 - 1) < 1e-6), lowest=lowest, loc=localize(rt, norm2_inv.weak),
               weakest=weakest_label(rt, norm2_inv.weak), secs=round(time.time() - t0, 1),
               kappa_phys=phys_kappa(smax, lowest, Nt),
               secs_inverse=round(time.time() - t1, 1))
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"{src}: D {D} grid {Nt}x{N} L {rt.L} |F_f64| {F:.1e} sigma_max {smax:.3e} sigma_min {smin:.3e} kappa "
          f"{rec['kappa']:.3e} (f64 says {rec['kappa_f64']:.2e}) resolved {rec['resolved']} {rec['secs']} s",
          flush=True)


if __name__ == "__main__":
    main()

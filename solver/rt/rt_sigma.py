"""An upper bound on the smallest gain of the RT Jacobian at a state, and the direction that attains it (issue #163).

    python3 solver/rt/rt_sigma.py STATE.npz [--starts 3] [--rough] [--prec 192] [--horizon K] [--max_products 250]
    python3 solver/rt/rt_sigma.py --planted [--mutate=shift]

For each of --starts smooth random right-hand sides v (max|v| = 1) it solves J x = v with rt_mpsolve's flexible GMRES
(products by exact central differences of the arb residual, the f64 LU of the f64 Jacobian as preconditioner,
relative residual 1e-30). Then max|J x| / max|x| = 1 / max|x| bounds the smallest gain of J in the max norm from
above (a bound, not an SVD; it is near sigma_min when v is not nearly orthogonal to the weakest left singular vector),
and x is dominated by the weakest right singular directions. Printed per start: max|x|, the bound, the solve's
residual max|J x - v|, and x's content in mu, by field, by harmonic and at xi = +1, near 0 and -1. Repeated inverse
iteration is not used: J is not normal, and J^-1 applied to its weakest right vector need not be large. --rough draws
each entry of v independently (normal), instead of smooth fields: a weak direction whose left vector is rough on the
grid (as for a polynomial copy of a horizon branch) is then seen, while smooth v miss it.
Planted check: a diagonal operator A = diag(s) with s log-uniform in [1e-3, 1] and one entry 1e-22, through the same
fgmres and LUdd (the f64 LU is exact for it); the best bound over three starts must lie in [1e-22, 1e-20], with x
along that entry to 1e-6. (Revised after the first run of this version, whose window was a factor 3: the bound's
slack is 1 / |u . v| for the weakest left vector u, and it came out 14.) --mutate=shift adds 1e-20 to the tiny entry
of the operator applied (not to the preconditioner), which moves the bound by a factor 100 (exit 1 then).
"""
import os
for _v in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import sys
import argparse
import numpy as np
import flint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rt_model import RT
from rt_mp import RTmp
import rt_mpsolve as mps

arb = flint.arb


def smooth_start(rt, seed=163):
    rng = np.random.default_rng(seed)
    co = RT(rt.D, 8, 11)
    T8, X8 = np.meshgrid(co.t, co.xi, indexing="ij")
    f = lambda: sum(rng.normal() * np.cos(k * T8 + rng.uniform(0, 6.3)) * X8 ** j for k in range(3) for j in range(3))
    w1 = np.sin(np.pi * X8 / 2) * np.cos(T8 + 0.3)
    w4 = sum(rng.normal() * np.cos((k + 0.5) * T8 + rng.uniform(0, 6.3)) * X8 ** j for k in range(2) for j in range(3))
    return co.resample_to(co.pack(w1, f(), f(), w4, rng.normal()), rt)


def describe(rt, x):
    w1, w2, w3, w4, mu = rt.unpack(x)
    out = [f"mu {mu[0, 0, 0]:+.2e}"]
    for nm, f, anti in (("w1", w1[0], False), ("w2", w2[0], False), ("w3", w3[0], False), ("w4", w4[0], True)):
        g = f * (np.exp(-0.5j * rt.t)[:, None] if anti else 1)
        H = np.abs(np.fft.fft(g, axis=0) / rt.Nt).max(axis=1)
        col = np.abs(f).max(axis=0)
        out.append(f"{nm} max {np.abs(f).max():.1e} (harmonics 0-3 {' '.join(f'{v:.0e}' for v in H[:4])}, 16 "
                   f"{H[rt.Nt // 2]:.0e}; xi=+1 {col[0]:.0e}, ~0 {col[len(col) // 2]:.0e}, -1 {col[-1]:.0e})")
    return "; ".join(out)


def bounds(apply, prec, vs, max_products, log=print):
    """For each v (max|v| scaled to 1): x = J^-1 v and the bound 1 / max|x|. Returns [(bound, x / max|x|)]."""
    out = []
    for i, v in enumerate(vs):
        v = mps.mids(v * arb(1.0 / mps.amax(v)))
        x, info = mps.fgmres(apply, prec, v, tol=1e-30, max_products=max_products)
        mx = mps.amax(x)
        res = mps.amax(mps.mids(apply(x) - v))
        log(f"  start {i}: max|x| {mx:.3e}, bound on the smallest gain {1.0 / mx:.3e}, max|J x - v| {res:.1e}, "
            f"{info['products']} products", flush=True)
        out.append((1.0 / mx, mps.mids(x * arb(1.0 / mx))))
    return out


def planted(mutate=None):
    flint.ctx.prec = 192
    rng = np.random.default_rng(1)
    L = 400
    s = np.exp(rng.uniform(np.log(1e-3), 0, size=L))
    s[123] = 1e-22
    so = mps.obj(s)
    if mutate == "shift":
        so[123] = so[123] + arb("1e-20")
    apply = lambda u: mps.mids(so * u)
    P = mps.LUdd(np.diag(s))
    res = bounds(apply, P, [mps.obj(rng.normal(size=L)) for _ in range(3)], 80)
    best, x = min(res, key=lambda r: r[0])
    xf = np.abs(mps.to_f64(x))
    off = np.delete(xf, 123).max()
    ok = 1e-22 <= best * (1 + 1e-12) and best < 1e-20 and off < 1e-6
    print(f"planted: best bound {best:.4e} (in [1e-22, 1e-20]), largest other component {off:.1e} (< 1e-6): "
          f"{'ok' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state", nargs="?")
    ap.add_argument("--starts", type=int, default=3)
    ap.add_argument("--rough", action="store_true", help="v with independent normal entries instead of smooth fields")
    ap.add_argument("--prec", type=int, default=192)
    ap.add_argument("--horizon", type=int, default=None, help="default: the state's own")
    ap.add_argument("--max_products", type=int, default=250)
    ap.add_argument("--planted", action="store_true")
    ap.add_argument("--mutate", default=None, choices=["shift"])
    a = ap.parse_args()
    if a.planted:
        sys.exit(0 if planted(a.mutate) else 1)
    flint.ctx.prec = a.prec
    d = np.load(a.state)
    D, Nt, N = float(d["D"]), int(d["Nt"]), int(d["N"])
    K = a.horizon if a.horizon is not None else (int(d["horizon"]) if "horizon" in d else 0)
    rt = RT(D, Nt, N, horizon=K)
    rm = RTmp(D, Nt, N, prec=a.prec, horizon=K)
    z = mps.from_dd(d["zh"], d["zl"])
    F = rm.residual_m(z)
    print(f"{a.state.rsplit('/', 1)[-1]}: D {D} {Nt}x{N} K {K}, max|F| {mps.amax(F):.2e}")
    P = mps.LUdd(rt.jacobian(mps.to_f64(z)), overwrite=True)
    apply = lambda u: mps.jv_central(rm.residual_m, z, u)
    if a.rough:
        rng = np.random.default_rng(163)
        vs = [mps.obj(rng.normal(size=rt.L)) for _ in range(a.starts)]
    else:
        vs = [mps.obj(smooth_start(rt, seed=163 + i)) for i in range(a.starts)]
    print(f"  right-hand sides: {'rough (independent normal entries)' if a.rough else 'smooth fields'}")
    res = bounds(apply, P, vs, a.max_products)
    for i, (b, x) in enumerate(res):
        print(f"  start {i}, x / max|x|: " + describe(rt, mps.to_f64(x)))
    best = min(r[0] for r in res)
    print(f"  best bound on the smallest gain (max norm) {best:.3e}; the bounds of the starts span a factor "
          f"{max(r[0] for r in res) / best:.1f}")


if __name__ == "__main__":
    main()

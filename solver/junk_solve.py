"""Solve the junk equations too. Gauss-Newton on the junk coordinates (s/smax < 3e-14 at the start: the centre-p
harmonics k >= 5..7 at large D) with the strong and weak equations re-solved by ne_solve at every evaluation.
The f64 Jacobian does not see these directions, but the residual responds to them at finite amplitude: NE
solutions that leave the junk residual at 1e-12 instead of 1e-14 carry Delta errors of ~1e-5 (D = 23).
The junk Jacobian is a forward difference with step h, its columns evaluated on a process pool.

    z, info = junk_solve(D, z, nodes, kc, workers)          # from scan_ne (env NE_JUNK)
    python junk_solve.py D z.npy xK [kc] [workers] [h] [out.npy]
"""
import json, os, sys, time, warnings
import numpy as np
from multiprocessing import Pool
warnings.filterwarnings("ignore")

CUT = 3e-14
_W = {}


def _setup(D, Delta, nodes, kc, Ks, basis, zb, Vj, Uj, n_junk=None):
    import fast; fast.install()
    from rust_solver import make
    S, R = make(D, Delta, nodes=nodes, threads=1, kc=kc, Ks=Ks)
    _W.update(R=R, basis=basis, zb=zb, Vj=Vj, Uj=Uj, n_junk=n_junk)


def _eval(a):
    """Re-solve strong and weak with the junk coordinates at a; return the junk residual and the solution."""
    from ne_solver import ne_solve
    R = _W["R"]
    try:
        z, info = ne_solve(R, _W["zb"] + _W["Vj"].T @ a, basis=_W["basis"], n_junk=_W["n_junk"])
        h, l = R.residual_dd(z)
        r = _W["Uj"].T @ (h + l)
    except (ValueError, np.linalg.LinAlgError):     # a step far outside the valley: NaN in the inner solve
        r = None
    if r is None or not np.all(np.isfinite(r)):
        return np.full(len(a), np.inf), _W["zb"], dict(F=np.inf)
    return r, z, info


def junk_solve(D, z0, nodes, kc=23, workers=None, h=1e-2, it_max=8, tol=1e-15, verbose=False, Ks=60, n_junk=None,
               damp=False, damp_mu=(0.0, 1e-3, 1e-2, 0.1, 0.3, 1.0)):
    """Returns (z, info): info has r0 / r (junk residual before / after), F, its (iterations) and hist.
    h: the difference step, or several tried in turn, each from where the previous one stopped, until tol is met.
    At D ~ 27 a step of 1e-3 can stall where 1e-2 converges; at D ~ 29 1e-2 stops short of the valley bottom.
    n_junk: take the n_junk smallest singular directions as the junk set (see ne_solve) instead of s/s_max < CUT.
    damp: instead of shortening the Gauss-Newton step (1, 1/2, 1/4, 1/10), try Levenberg-damped steps
    -sum_i sigma_i / (sigma_i^2 + mu^2) q_i p_i^T r for mu / sigma_max in (0, 1e-3, 1e-2, 0.1, 0.3, 1). Near the valley
    bottom most of the difference Jacobian is noise (D = 30.98, h = 1e-2..3e-3: full steps of 0.2-0.6 whose best tenth
    still raised the junk residual from 5.5e-15 to 2-3e-14); damping cuts the noisy directions, not the whole step.
    damp_mu: the mu / sigma_max values tried. The undamped trial (0) is the far one whose inner solve can take most of an
    iteration's time (x_K 0.65, Ks 100), so a run may leave it out."""
    from rust_solver import make
    S, R = make(D, z0[-1], nodes=nodes, threads=2, kc=kc, Ks=Ks)
    _, J = R.jacobian(z0)
    U, s, Vt = np.linalg.svd(J)
    jm = s / s[0] < CUT
    if n_junk is not None:
        jm = np.arange(len(s)) >= len(s) - n_junk
    args = (D, z0[-1], nodes, kc, Ks, (U, s, Vt), z0, Vt[jm], U[:, jm], n_junk)
    _setup(*args)
    t0 = time.time()
    a = np.zeros(jm.sum())
    r, z, inf = _eval(a)
    info = dict(n_junk=int(jm.sum()), r0=float(np.linalg.norm(r)), hist=[])
    with Pool(workers or os.cpu_count(), initializer=_setup, initargs=args) as pool:
        for hh in np.atleast_1d(h):
            for it in range(it_max):
                if np.linalg.norm(r) < tol:
                    break
                cols = pool.map(_eval, [a + hh * e for e in np.eye(len(a))])
                Jr = np.stack([(c[0] - r) / hh for c in cols], 1)
                if not np.all(np.isfinite(Jr)):
                    break
                if damp:
                    P, sg, Qt = np.linalg.svd(Jr)
                    steps = tuple(damp_mu)
                    das = [-Qt.T @ ((sg / (sg ** 2 + (m * sg[0]) ** 2)) * (P.T @ r)) for m in steps]
                else:
                    da = -np.linalg.lstsq(Jr, r, rcond=1e-10)[0]
                    steps = (1.0, 0.5, 0.25, 0.1)
                    das = [t * da for t in steps]
                (t, dat), (rt, zt, it_inf) = min(zip(zip(steps, das), pool.map(_eval, [a + d for d in das])),
                                                 key=lambda x: np.linalg.norm(x[1][0]))
                rec = dict(it=it, h=float(hh), step=t, da=float(np.linalg.norm(dat)), r_junk=float(np.linalg.norm(rt)),
                           F=it_inf["F"], Delta=float(zt[-1]), s=round(time.time() - t0))
                info["hist"].append(rec)
                if verbose:
                    print(json.dumps(rec), flush=True)
                if np.linalg.norm(rt) >= np.linalg.norm(r):
                    break
                a, r, z, inf = a + dat, rt, zt, it_inf
            if np.linalg.norm(r) < tol:
                break
    info.update(r=float(np.linalg.norm(r)), F=inf["F"], its=len(info["hist"]), seconds=round(time.time() - t0))
    return z, info


def main():
    D, z0, xK = float(sys.argv[1]), np.load(sys.argv[2]), float(sys.argv[3])
    kc = int(sys.argv[4]) if len(sys.argv) > 4 else 23
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else os.cpu_count()
    h = float(sys.argv[6]) if len(sys.argv) > 6 else 1e-2
    out = sys.argv[7] if len(sys.argv) > 7 else f"junk_D{D:.4f}.npy"
    import fast; fast.install()
    from rust_solver import NODES_L
    z, info = junk_solve(D, z0, NODES_L + ([0.7] if xK > 0.65 else []), kc, workers, h, verbose=True)
    print(f"{info['n_junk']} junk directions: |r_junk| {info['r0']:.2e} -> {info['r']:.2e}, |F| {info['F']:.1e}, "
          f"Delta {z0[-1]:.10f} -> {z[-1]:.10f} ({z[-1] - z0[-1]:+.2e}), {info['seconds']} s", flush=True)
    np.save(out, z)


if __name__ == "__main__":
    main()

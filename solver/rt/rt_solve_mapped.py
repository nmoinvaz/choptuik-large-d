"""Solve the RT prototype at one D from our state mapped into RT gauge, on several grids, and compare Delta with ours.

    python3 solver/rt/rt_solve_mapped.py OUT.jsonl --profile PROF.npz --grids 24x21,24x31,24x41 [--mp 192]

For each grid: the map (gg_to_rt.GGMap) gives the first guess, f64 Newton, then the refinement with the arb residual
(rt_mp.newton_mixed, f64 Jacobian; it converges only while kappa 2^-53 < 1, and the record says whether it did).
Records the mapped residual, kappa from rt_cond.extremes at the solution, Delta from xi = 0 and xi = 1, and
Delta_RT - Delta_hp (hp/data/delta_all.csv at the state's D). Exits 1 if no grid converged.
"""
import os
for _v in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rt_model import RT
from gg_to_rt import GGMap
from rt_run import delta_hp
from rt_cond import extremes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--grids", default="24x21")
    ap.add_argument("--mp", type=int, default=192)
    ap.add_argument("--arb_its", type=int, default=0, help="Newton steps in arb from the mapped state (0: f64 + mixed)")
    a = ap.parse_args()
    from rt_mp import RTmp, newton_mixed, to_f64, obj
    g = GGMap(a.profile, verbose=False)
    D = g.D
    ref = delta_hp(D)
    any_ok = False
    for gr in a.grids.split(","):
        Nt, N = (int(v) for v in gr.split("x"))
        t0 = time.time()
        rt = RT(D, Nt, N)
        z0, err, _ = g.to_rt_phased(rt)
        F0 = float(np.abs(rt.residual(z0)[0]).max())
        # the first full Newton step from the mapped state (f64): its size, its change of Delta, and its predicted
        # quadratic remainder |F(z0 + dz)|; a step far larger than the map's error means the discrete solution on this
        # grid is not near the true one, or not in Newton's basin from it
        J0 = rt.jacobian(z0)
        dz0 = np.linalg.solve(J0, -rt.residual(z0)[0])
        step = dict(max_dz=float(np.abs(dz0).max()), dDelta=float(rt.delta(z0 + dz0)[0] - rt.delta(z0)[0]),
                    F_after=float(np.abs(rt.residual(z0 + dz0)[0]).max()))
        sa, sb, _, _ = extremes(rt.jacobian(z0 + dz0))
        step["kappa"] = float(sa / sb)          # kappa after the first step (near the discrete solution if it holds)
        rm = RTmp(D, Nt, N, prec=a.mp)
        if a.arb_its:
            # Newton entirely in arb from the mapped state: the Jacobian at the arb state (rt_cond_mp.jac_arb) and an
            # arb solve, for kappa beyond f64's reach (the f64 Jacobian's rounding then exceeds sigma_min)
            import flint
            from rt_cond_mp import jac_arb, to_arb_mat
            hist = []
            zm = obj(z0)
            mh = []
            for _ in range(a.arb_its):
                F = rm.residual_m(zm)
                mh.append(float(max(abs(v.mid()) for v in F)))
                if mh[-1] < 1e-28:
                    break
                A = to_arb_mat(jac_arb(rm, zm), a.mp)
                dz = A.solve(flint.arb_mat(rt.L, 1, [-v for v in F]), algorithm="approx")
                zm = zm + np.array([dz[i, 0] for i in range(rt.L)], dtype=object)
            mh.append(float(max(abs(v.mid()) for v in rm.residual_m(zm))))
            okm = mh[-1] < 1e-25
        else:
            z, ok, hist = rt.newton(z0.copy(), tol=1e-11, it_max=15, verbose=False)
            zm, okm, mh = newton_mixed(rm, z, tol=1e-28, it_max=40, log=lambda s: None)
        d0, d1 = (float(v.mid()) for v in rm.delta_m(zm))
        zf = to_f64(zm)
        smax, smin, _, _ = extremes(rt.jacobian(zf))
        rec = dict(D=D, grid=[Nt, N], L=rt.L, profile=a.profile, F_mapped=F0, newton=[float(h) for h in hist],
                   mixed=[float(h) for h in mh], converged=bool(okm), mu=float(zf[-1]), mu_map=float(g.mu),
                   Delta_xi0=d0, Delta_xi1=d1, Delta_hp=ref, diff=(d0 - ref) if ref is not None else None,
                   Delta_state=float(g.Delta), first_step=step,
                   Delta_map=float(rt.delta(z0)[1]), sigma_max=float(smax), sigma_min=float(smin),
                   kappa=float(smax / smin), secs=round(time.time() - t0, 1))
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        any_ok |= okm
        how = f"f64 Newton {len(hist) - 1} its to {hist[-1]:.1e}" if hist else f"arb Newton {len(mh) - 1} its"
        print(f"D {D} grid {gr} L {rt.L}: mapped |F| {F0:.1e}, {how}, "
              f"mixed {'converged' if okm else 'NOT converged'} to {mh[-1]:.1e}; Delta {d0:.14f} (xi=1 {d1:.14f}) "
              f"hp {ref} diff {rec['diff']:+.2e}; kappa {rec['kappa']:.2e}; first step max dz {step['max_dz']:.1e} "
              f"dDelta {step['dDelta']:+.1e} F after {step['F_after']:.1e}; {rec['secs']} s", flush=True)
    sys.exit(0 if any_ok else 1)


if __name__ == "__main__":
    main()

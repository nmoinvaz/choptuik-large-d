"""The RT prototype's Jacobian condition number against D on one grid: continuation in D (steps dD, the guess
linearly extrapolated) from a solved state, and at every D a multiple of --every the record of rt_cond.extremes
(sigma_max, sigma_min, kappa, the Ruiz-equilibrated kappa_eq) with where the weakest right singular vector lives.

    python3 solver/rt/rt_scan.py OUT.jsonl START.npz --grid 16x21 --D_to 20 [--dD 0.5] [--every 1] [--kappa_stop 1e15]

START.npz: an rt_run.py state (any grid; resampled to --grid and re-solved). Stops when Newton fails, or once
kappa > --kappa_stop (f64 cannot resolve sigma_min below about 1e-15 sigma_max, so larger values are rounding).
The localization of the weakest vector v: the share of |v|^2 in xi < -0.8 (the + equations' horizon image, where D^+
degenerates), |xi| < 0.2 (the centre) and xi > 0.8, its field and t harmonic (rt_cond.weakest_label).
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
from rt_cond import extremes, ruiz, weakest_label
from rt_run import delta_hp


def localize(rt, v):
    w = rt.unpack(v)
    e = np.zeros(rt.N + 1)
    for f in w[:4]:
        e += (f[0] ** 2).sum(axis=0)
    e /= e.sum()
    xi = rt.xi
    return dict(hor_minus=float(e[xi < -0.8].sum()), centre=float(e[np.abs(xi) < 0.2].sum()),
                hor_plus=float(e[xi > 0.8].sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("start")
    ap.add_argument("--grid", default="16x21")
    ap.add_argument("--D_to", type=float, default=20.0)
    ap.add_argument("--dD", type=float, default=0.5)
    ap.add_argument("--every", type=float, default=1.0)
    ap.add_argument("--kappa_stop", type=float, default=1e15)
    ap.add_argument("--save_dir", default=None)
    a = ap.parse_args()
    Nt, N = (int(v) for v in a.grid.split("x"))
    d = np.load(a.start)
    D = float(d["D"])
    src = RT(D, int(d["Nt"]), int(d["N"]))
    rt = RT(D, Nt, N)
    z = src.resample_to(d["z"], rt)
    hist = []
    while True:
        rt = RT(D, Nt, N)
        if len(hist) >= 2:
            (Da, za), (Db, zb) = hist[-2], hist[-1]
            z = zb + (zb - za) * (D - Db) / (Db - Da)
        t0 = time.time()
        z, ok, h = rt.newton(z, tol=1e-11, it_max=15, verbose=False)
        if not ok:
            print(f"D {D}: Newton failed {h}", flush=True)
            break
        hist.append((D, z))
        if abs(D / a.every - round(D / a.every)) < 1e-9:
            J = rt.jacobian(z)
            smax, smin, v, lu = extremes(J)
            se, sm, _, _ = extremes(ruiz(J))
            d0, d1 = rt.delta(z)
            ref = delta_hp(D)
            rec = dict(D=D, grid=[Nt, N], L=rt.L, F=float(h[-1]), its=len(h) - 1, mu=float(z[-1]), Delta=float(d0),
                       Delta_xi1=float(d1), Delta_hp=ref, sigma_max=float(smax), sigma_min=float(smin),
                       kappa=float(smax / smin), kappa_eq=float(se / sm), loc=localize(rt, v),
                       weakest=weakest_label(rt, v), secs=round(time.time() - t0, 1))
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"D {D:6.2f} {Nt}x{N}: its {rec['its']} |F| {h[-1]:.1e} Delta {d0:.10f} (hp {ref}) kappa "
                  f"{rec['kappa']:.2e} eq {rec['kappa_eq']:.2e} loc {rec['loc']} {rec['weakest']} {rec['secs']} s",
                  flush=True)
            if a.save_dir:
                np.savez(os.path.join(a.save_dir, f"scan_D{D:.4f}_{Nt}x{N}.npz"), z=z, D=D, Nt=Nt, N=N)
            if rec["kappa"] > a.kappa_stop:
                print(f"kappa above {a.kappa_stop:g}: stop", flush=True)
                break
        if D >= a.D_to - 1e-12:
            break
        D = min(D + a.dD, a.D_to)


if __name__ == "__main__":
    main()

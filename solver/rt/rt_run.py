"""Solve the RT formulation (rt_model.py) at D = 4 from an hp state mapped into RT gauge (gg_to_rt.py), continue it in
D with EEG's protocol (steps of 1/2, the initial guess linearly extrapolated from the two previous solutions), and
compare Delta with hp/data/delta_all.csv.

    python3 solver/rt/rt_run.py --profile PROFILE.npz --grids 32x41,48x61 --D_to 10 --out DIR [--mutate=coef]

Exits 1 when |Delta_RT - Delta_hp| > --tol (1e-8) at the start D or at D_to, on the last grid. --mutate=coef scales the
coefficient (D - 3)(D - 2)/2 of Omega_3 by 1 + 1e-3; the run must then exit 1 with Delta moved by more than 1e-6.
mu at D = 4 is also compared with Reiterer and Trubowitz's enclosure (arXiv:1203.3766v1, Eq. (3), radius 1e-80).
"""
import os
for _v in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")          # a shared machine: two BLAS threads
import sys
import csv
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rt_model import RT
from gg_to_rt import GGMap

MU_RT4 = 0.16830707896344996951013497904285742072100199080892966476395293134873313662587505
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


def delta_hp(D):
    with open(os.path.join(REPO, "hp", "data", "delta_all.csv")) as f:
        for r in csv.DictReader(f):
            if abs(float(r["D"]) - D) < 1e-9:
                return float(r["Delta_q0"]) + float(r["Delta_q1"] or 0.0)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--grids", default="32x41")
    ap.add_argument("--cont_grid", default=None, help="grid of the continuation (default: the first)")
    ap.add_argument("--D_to", type=float, default=None)
    ap.add_argument("--dD", type=float, default=0.5)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="rt")
    ap.add_argument("--mutate", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    coef = 1.0 + 1e-3 if a.mutate == "coef" else 1.0
    tag = a.tag + ("_mutcoef" if a.mutate == "coef" else "")
    log = open(os.path.join(a.out, f"{tag}.log"), "a")
    rec = open(os.path.join(a.out, f"{tag}.jsonl"), "a")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    grids = [tuple(int(v) for v in g.split("x")) for g in a.grids.split(",")]
    g = GGMap(a.profile, verbose=False)
    D0 = g.D
    say(f"== {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC  profile {a.profile} D {D0} grids {grids} "
        f"mutate {a.mutate}  map: lam {g.lam:.15f} mu {g.mu:.15f} {g.check}")
    failed = False

    def solve(rt, z, D, label):
        t0 = time.time()
        z, ok, hist = rt.newton(z, tol=1e-11, it_max=15, verbose=False)
        d0, d1 = rt.delta(z)
        ref = delta_hp(D)
        r = dict(D=D, grid=[rt.Nt, rt.N], L=rt.L, mu=float(z[-1]), Delta_xi0=float(d0), Delta_xi1=float(d1),
                 Delta_hp=ref, diff=(float(d0 - ref) if ref is not None else None), newton=[float(h) for h in hist],
                 ok=bool(ok), secs=round(time.time() - t0, 1), label=label, mutate=a.mutate)
        rec.write(json.dumps(r) + "\n")
        rec.flush()
        say(f"  {label} D {D:.3f} grid {rt.Nt}x{rt.N} L {rt.L}: newton {len(hist) - 1} its, max|F| {hist[-1]:.1e}, "
            f"mu {z[-1]:.15f}, Delta {d0:.13f} (xi=1: {d1:.13f}), hp {ref}, diff "
            f"{(d0 - ref) if ref is not None else float('nan'):.2e}, {r['secs']} s")
        return z, ok, r

    def check(r, where):
        nonlocal failed
        if r["Delta_hp"] is None:
            return
        bad = abs(r["diff"]) > a.tol or not r["ok"]
        say(f"  CHECK {where}: |Delta_RT - Delta_hp| = {abs(r['diff']):.2e} {'>' if bad else '<='} {a.tol:g}"
            f" {'FAIL' if bad else 'ok'}")
        failed |= bad

    # ------------------------------------------------ start D: map, then each grid
    rt = RT(D0, *grids[0], coef_mut=coef)
    z, err, s = g.to_rt_phased(rt)
    say(f"  mapped initial guess: inversion error {err:.1e}, t shift {s:.4f}, max|F(z0)| "
        f"{np.abs(rt.residual(z)[0]).max():.2e}")
    states = {}
    for i, gr in enumerate(grids):
        if i > 0:
            new = RT(D0, *gr, coef_mut=coef)
            z = rt.resample_to(z, new)
            rt = new
        z, ok, r = solve(rt, z, D0, "start")
        states[gr] = z
    if abs(D0 - 4) < 1e-12:
        say(f"  mu - mu_RT(D=4) = {z[-1] - MU_RT4:.2e}")
    check(r, f"D = {D0}")
    np.savez(os.path.join(a.out, f"{tag}_D{D0:.4f}_{rt.Nt}x{rt.N}.npz"), z=z, D=D0, Nt=rt.Nt, N=rt.N)
    if a.D_to is None:
        sys.exit(1 if failed else 0)

    # ------------------------------------------------ continuation in D
    cg = tuple(int(v) for v in a.cont_grid.split("x")) if a.cont_grid else grids[0]
    rtc = RT(D0, *cg, coef_mut=coef)
    zc = states[cg] if cg in states else rt.resample_to(z, rtc)
    hist = [(D0, zc)]
    D = D0
    while D < a.D_to - 1e-12:
        Dn = min(D + a.dD, a.D_to)
        if len(hist) >= 2:
            (Da, za), (Db, zb) = hist[-2], hist[-1]
            zg = zb + (zb - za) * (Dn - Db) / (Db - Da)
        else:
            zg = hist[-1][1]
        rtc = RT(Dn, *cg, coef_mut=coef)
        zc, ok, r = solve(rtc, zg, Dn, "cont")
        if not ok:
            say("  continuation step failed; stopping")
            failed = True
            break
        hist.append((Dn, zc))
        D = Dn
        np.savez(os.path.join(a.out, f"{tag}_D{D:.4f}_{rtc.Nt}x{rtc.N}.npz"), z=zc, D=D, Nt=rtc.Nt, N=rtc.N)
    # ------------------------------------------------ the end point on every grid
    z, rt = zc, rtc
    for gr in grids:
        if gr == cg:
            zz, ok, r = solve(rtc, zc, D, "end")
        else:
            new = RT(D, *gr, coef_mut=coef)
            zz, ok, r = solve(new, rtc.resample_to(zc, new), D, "end")
            np.savez(os.path.join(a.out, f"{tag}_D{D:.4f}_{gr[0]}x{gr[1]}.npz"), z=zz, D=D, Nt=gr[0], N=gr[1])
    check(r, f"D = {D}")
    say(f"== exit {1 if failed else 0}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

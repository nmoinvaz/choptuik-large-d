#!/usr/bin/env python3
"""Check of driver.renode's horizon-series path for a node beyond the last old one (T45, D 3.075).

    cd hp && python3 renode_horizon_check.py [--Ms 256,320] [--add 0.975] [--threads 6]

At D 3.075 the 19-node states (0.000313 .. 0.95, kc 100, ks 250) had an M 320 horizon tail of 5e-33 (> --tail_max
1e-33); --auto_nodes added a node at 0.975, and the re-noded M 256 state was NaN: its state at 0.975 came from Taylor
integration outward from 0.95, past the horizon series' nearest singularity (radius about 0.068 in u). renode now takes
a node beyond the last old one from the horizon series at u = 1 - x (u 0.025, about 0.37 per order there).
For each M: the stopped state re-noded both ways, each start's |F| on the plan of the horizon-path start (a NaN state
has no plan of its own), and the horizon-path start's series tails. Pass: the horizon path is finite with |F| below
1e-10; the control (beyond="integrate", the old path, h 0.001 as lowD_highM.py uses) is not finite. Exits 1 otherwise.
The convergence of the re-noded starts is checked by highm_check.py lift --nodes (two passes), see the T45 log."""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

from hpkernel.driver import load, make_solver, renode  # noqa: E402
import lowD_dd  # noqa: E402

STATE = os.path.join(HERE, "data", "lowD_states", "lowDhm_x0.000313-0.95_D3.0750_M{}.npz")


def fnorm(s, zh, zl):
    Fh, Fl = s.residual(zh, zl)
    return float(np.abs(np.asarray(Fh) + np.asarray(Fl)).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ms", default="256,320")
    ap.add_argument("--add", type=float, default=0.975)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    ok = True
    for M in [int(v) for v in a.Ms.split(",")]:
        zh, zl, m = load(STATE.format(M))
        old = list(m["nodes"])
        new = old + [a.add]
        s0 = make_solver(m["D"], M, nodes=old, kt=m["kt"], kc=m["kc"], ks=m["ks"], threads=a.threads)
        s0.set_segs(m["segs"])
        F0 = fnorm(s0, zh, zl)
        hz = renode(s0, zh, zl, old, new, M, h=0.001)
        ig = renode(s0, zh, zl, old, new, M, h=0.001, beyond="integrate")
        s1 = make_solver(m["D"], M, nodes=new, kt=m["kt"], kc=m["kc"], ks=m["ks"], threads=a.threads)
        segs = s1.plan(np.asarray(hz[0]), np.asarray(hz[1]), 1e-30, 0.05)
        s1.set_segs(segs)
        Fh = fnorm(s1, *hz)
        Fi = fnorm(s1, *ig)
        tc, th = lowD_dd.tails(s1, np.asarray(hz[0]), np.asarray(hz[1]), new, M)
        good = np.isfinite(Fh) and Fh < 1e-10
        fires = not np.isfinite(Fi)
        ok = ok and good and fires
        print(f"M {M}: stopped state |F| {F0:.1e} on its own {len(old)}-node plan; re-noded to {len(new)} nodes "
              f"(+{a.add}) on the horizon start's plan ({sum(len(g) - 1 for g in segs)} steps): horizon series |F| "
              f"{Fh:.1e} ({'pass' if good else 'FAIL'}), tails {tc:.1e} (centre) {th:.1e} (horizon); control, "
              f"outward integration: |F| {Fi:.1e} ({'fires' if fires else 'DOES NOT FIRE'})", flush=True)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

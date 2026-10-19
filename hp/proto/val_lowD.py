"""Validation at low D against the kc 39 / kc 47 references (data_all.low_d)."""
import sys
import numpy as np
sys.path.insert(0, "/home/claude/dss-residual-kernel/solver")
from hpkernel.driver import make_solver, seed_from_f64, newton_solve, save, resample_z
NL = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6]
import os
os.chdir("/home/claude/dss-residual-kernel/solver/data")
from data_all import low_d
ref = low_d()
for D0, f in ((7.75, "neB_D7.7500.npy"), (7.0, "neB_D7.0000.npy")):
    r = ref.get(round(D0, 4))
    D, zh, zl = seed_from_f64("/home/claude/dss/" + f, 32, nodes=NL)
    out = []
    for M in (32, 48, 64):
        if M != 32:
            zh, zl = resample_z(zh, Mo, M, len(NL))
        s = make_solver(D, M, nodes=NL, kt=36, kc=60, ks=100)
        s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
        zh, zl, hist = newton_solve(s, zh, zl, tol=1e-29, it_max=10, verbose=False)
        out.append(zh[-1] + zl[-1]); Mo = M
        print(f"D {D} M {M}: Delta {zh[-1]:.15f} (+{zl[-1]:.2e}) |F| {hist[-1]['F']:.1e} its {len(hist)}", flush=True)
        save(f"/home/claude/dss/hp_D{D:.2f}_M{M}.npz", zh, zl, dict(D=D, M=M, nodes=NL))
    print(f"   reference {r}  -> DD minus reference {out[-1] - r[0]:+.2e} (sigma {r[1]:.1e})", flush=True)

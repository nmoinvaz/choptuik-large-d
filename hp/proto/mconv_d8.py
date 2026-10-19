import sys
import numpy as np
from hpkernel.driver import make_solver, newton_solve, load, save, resample_z, block_norms
NL = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6]
zh, zl, meta = load("/home/claude/dss/hp_D8_M32.npz")
D = meta["D"]; Mo = 32
for M in (48, 64, 80):
    zh, zl = resample_z(zh, Mo, M, len(NL))
    s = make_solver(D, M, nodes=NL, kt=36, kc=40, ks=100)
    s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
    zh, zl, hist = newton_solve(s, zh, zl, tol=1e-29, it_max=8, verbose=False)
    print(f"M {M}: Delta {zh[-1]!r} + {zl[-1]:+.3e}   iterations {len(hist)}  |F| {hist[-1]['F']:.1e}  t {hist[-1]['t']}s", flush=True)
    save(f"/home/claude/dss/hp_D8_M{M}.npz", zh, zl, dict(D=D, M=M, nodes=NL))
    Mo = M
# parameter checks at M = 64
zh, zl, meta = load("/home/claude/dss/hp_D8_M64.npz")
for kt, kc, ks, tol in ((48, 40, 100, 1e-32), (36, 60, 100, 1e-32), (36, 40, 140, 1e-32), (36, 40, 100, 1e-34)):
    s = make_solver(D, 64, nodes=NL, kt=kt, kc=kc, ks=ks)
    s.set_segs(s.plan(zh, zl, tol, 0.05))
    z2h, z2l, hist = newton_solve(s, zh.copy(), zl.copy(), tol=1e-29, it_max=6, verbose=False)
    print(f"kt {kt} kc {kc} ks {ks} tol {tol:.0e}: dDelta vs M64 base {((z2h[-1] - zh[-1]) + (z2l[-1] - zl[-1])):+.2e}", flush=True)

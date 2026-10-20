"""Independent double-double solve at D = 5, seeded from the f64 kc 47 (RK x2) solution, at tau resolutions M = 32, 48, 64,
(80): Delta(5) and its convergence in M. Same recipe as hp/proto/val_lowD.py (which validated D = 7, 7.75 against kc 39/47 to 3e-12).
    python3 dd_D5.py [seedfile] [kc_of_seed]"""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "solver"))
from hpkernel.driver import make_solver, seed_from_f64, newton_solve, save, resample_z
NL = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6]
seed = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "solver", "data", "neB_kc47_rk2_D5.0000.npy")
kcs = int(sys.argv[2]) if len(sys.argv) > 2 else 47
Ms = [int(m) for m in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["32", "48", "64"])]
D, zh, zl = seed_from_f64(seed, Ms[0], kc=kcs, nodes=NL)
print(f"seed {seed}: D {D}, f64 Delta {zh[-1]:.12f}", flush=True)
Mo = Ms[0]
for M in Ms:
    if M != Mo:
        zh, zl = resample_z(zh, Mo, M, len(NL))
    t0 = time.time()
    s = make_solver(D, M, nodes=NL, kt=36, kc=60, ks=100)
    s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
    zh, zl, hist = newton_solve(s, zh, zl, tol=1e-29, it_max=14, verbose=False)
    Mo = M
    print(f"D {D} M {M}: Delta {zh[-1]:.16f} (+{zl[-1]:.2e})  |F| {hist[-1]['F']:.1e}  its {len(hist)}  {time.time()-t0:.0f}s", flush=True)
    save(os.path.join(HERE, f"dd_D{D:.4f}_M{M}.npz"), zh, zl, dict(D=D, M=M, nodes=NL))

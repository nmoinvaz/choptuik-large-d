import sys
import numpy as np
from hpkernel.driver import make_solver, seed_from_f64, newton_solve, block_norms, save, resample_z
NL = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6]
M = int(sys.argv[1]) if len(sys.argv) > 1 else 16
D, zh, zl = seed_from_f64("/home/claude/dss/longB_D20.1460.npy", M, nodes=NL, kmax=7, kmax_all=None)
s = make_solver(D, M, nodes=NL, kt=36, kc=40, ks=100)
s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
print(f"D {D} M {M} L {s.size} seed Delta {zh[-1]:.12f} (reference 2.108951698328)", flush=True)
zh, zl, hist = newton_solve(s, zh, zl, tol=1e-29, it_max=25)
Fh, Fl = s.residual(zh, zl)
print("blocks", " ".join(f"{v:.0e}" for v in block_norms(s, Fh, Fl)))
print(f"Delta = {zh[-1]!r} + {zl[-1]!r}")
save(f"/home/claude/dss/hp_D20_M{M}.npz", zh, zl, dict(D=D, M=M, nodes=NL))

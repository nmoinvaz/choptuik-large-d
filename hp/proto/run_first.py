import sys, time
import numpy as np
from hpkernel.driver import make_solver, seed_from_f64, newton_solve, block_norms, save
M = int(sys.argv[1]) if len(sys.argv) > 1 else 48
D, zh, zl = seed_from_f64("/home/claude/dss/ne7_js_rk3_s111111118_D38.1740.npy", M, kmax=11)
s = make_solver(D, M, kt=36, kc=40, ks=150)
s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
print(f"D {D} M {M} L {s.size}; seed Delta {zh[-1]:.12f}", flush=True)
zh, zl, hist = newton_solve(s, zh, zl, tol=1e-29, it_max=10)
Fh, Fl = s.residual(zh, zl)
print("final blocks", " ".join(f"{v:.0e}" for v in block_norms(s, Fh, Fl)))
print(f"Delta = {zh[-1]!r} + {zl[-1]!r}")
save(f"/home/claude/dss/hp_first_M{M}.npz", zh, zl, dict(D=D, M=M, kt=36, kc=40, ks=150, segs=s.get_segs()))

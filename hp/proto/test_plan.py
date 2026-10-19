import time
import numpy as np
from hpkernel.driver import make_solver, seed_from_f64, block_norms
M = 48
D, zh, zl = seed_from_f64("/home/claude/dss/ne7_js_rk3_s111111118_D38.1740.npy", M, kmax=11)
for kt in (24, 36, 48):
    s = make_solver(D, M, kt=kt, kc=40, ks=100)
    t0 = time.time(); segs = s.plan(zh, zl, 1e-32, 0.05); t1 = time.time()
    s.set_segs(segs)
    t2 = time.time(); Fh, Fl = s.residual(zh, zl); t3 = time.time()
    print(f"kt {kt}: steps per segment {[len(g) - 1 for g in segs]} (total {sum(len(g) - 1 for g in segs)}), plan {t1 - t0:.2f}s, residual {t3 - t2:.3f}s")
    print("   blocks", " ".join(f"{v:.1e}" for v in block_norms(s, Fh, Fl)))

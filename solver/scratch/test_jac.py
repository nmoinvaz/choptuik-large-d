import numpy as np, time, warnings
warnings.filterwarnings("ignore")
from dss import DSSMulti, DSS
from refine import nodes
z = np.load("/home/claude/dss/sol_ms_N32.npy")
S = DSSMulti(nodes=nodes, n=2.0, N=32, Ks=8, u0=0.03, seg_h=0.002, log_steps=150)
t0 = time.time(); F1, J1 = S.jacobian(z); t1 = time.time() - t0
t0 = time.time(); F2, J2 = DSS.jacobian(S, z); t2 = time.time() - t0
print(f"structured {t1:.1f} s, dense {t2:.1f} s; max |J1 - J2| = {np.abs(J1 - J2).max():.2e} (max |J| {np.abs(J2).max():.1e})")

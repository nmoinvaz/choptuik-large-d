"""
Initial guess for the periodic solve, taken from our own near-critical collapse simulation
(the run just below threshold from /home/claude/collapse).

1. Re-run the near-critical evolution, saving Phi, Pi, a, alpha on every polar slice in the echo phase.
2. Locate the accumulation point T* (central proper time) from the echo timing.
3. Trace the past light cone of (T*, r = 0) backwards: the self-similarity horizon r_s(t).
4. Re-express each slice in self-similar variables: tau = -ln r_s, x = r / r_s, and read off
   p1(tau) = r_s Pi(t, 0), m(tau, 1), a(tau, 1), c(tau, 0) = alpha(0) a(r_s) / alpha(r_s).
"""
import sys, json
import numpy as np
sys.path.insert(0, "/home/claude/collapse")
from gr_collapse import make_grid, initial_data_family, evolve

ps = json.load(open("/home/claude/collapse/pstar_A.json"))
p = ps["p_lo"]
dx, L, rmax, t_end, cfl = 0.0004, 0.05, 30.0, 26.0, 0.3
r = make_grid(dx, L, rmax)
phi, Phi, Pi = initial_data_family(r, p, "A")
nsteps_total = int(np.ceil(t_end / (cfl * r[1])))
dt = t_end / nsteps_total
idx = np.zeros(1, dtype=np.int64)

chunk = 10
t, Tc = 0.0, 0.0
snaps = []
while t < 17.3:
    out = evolve(r, phi, Phi, Pi, dt, chunk, 0.5, 2.0, 0, idx, chunk, 1e9, 0.1)
    central = out[7]; m, a, alpha = out[8], out[9], out[10]
    Tc += central[-1, 1]
    t += chunk * dt
    if t > 10.0:
        snaps.append((t, Tc, Phi.copy(), Pi.copy(), a.copy(), alpha.copy()))
print("snapshots:", len(snaps), "grid points:", len(r), "dt", dt)
T = np.array([s[0] for s in snaps]); TC = np.array([s[1] for s in snaps])
PHI = np.array([s[2] for s in snaps]); PI = np.array([s[3] for s in snaps])
A = np.array([s[4] for s in snaps]); AL = np.array([s[5] for s in snaps])
np.savez_compressed("/home/claude/dss/nearcrit_slices.npz", t=T, Tc=TC, Phi=PHI, Pi=PI, a=A, alpha=AL, r=r)

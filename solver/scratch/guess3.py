"""
Initial guess, second attempt: find the self-similarity horizon locally, within one echo period,
instead of tracing the light cone through the poorly resolved end of the run.

In central-proper-time similarity coordinates tau_c = -ln(T*-T), x_c = r/(T*-T), ingoing light rays obey
dx/dtau_c = x - c_c(tau_c, x),  c_c = (alpha/a)/alpha(0).  The SSH is the periodic ray: shoot over one period.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

d = np.load("/home/claude/dss/nearcrit_slices.npz")
t, Tc, Phi, Pi, a, al, r = d["t"], d["Tc"], d["Phi"], d["Pi"], d["a"], d["alpha"], d["r"]
# successive extrema of the central field, half a period apart
E1, E2, E3 = 8.588415, 9.027686, 9.110792
Ts = ((E2 ** 2) - E1 * E3) / (2 * E2 - E1 - E3)
Dl = 2 * np.log((Ts - E1) / (Ts - E2))
Dl2 = 2 * np.log((Ts - E2) / (Ts - E3))
print(f"T* = {Ts:.5f}   Delta estimate = {Dl:.4f} (check {Dl2:.4f})")
tauc = -np.log(Ts - Tc)
tA, tB = -np.log(Ts - E1), -np.log(Ts - E3)
cc = (al / a) / al[:, :1]

def c_at(tc, x):
    j = np.clip(np.searchsorted(tauc, tc) - 1, 0, len(tauc) - 2)
    f = (tc - tauc[j]) / (tauc[j + 1] - tauc[j])
    R = x * (Ts - (Tc[j] * (1 - f) + Tc[j + 1] * f))
    return (1 - f) * np.interp(R, r, cc[j]) + f * np.interp(R, r, cc[j + 1])

def ray(x0, nsteps=4000):
    h = (tB - tA) / nsteps; x = x0; tc = tA; xs = [x0]
    for _ in range(nsteps):
        k1 = x - c_at(tc, x); k2 = (x + h / 2 * k1) - c_at(tc + h / 2, x + h / 2 * k1)
        k3 = (x + h / 2 * k2) - c_at(tc + h / 2, x + h / 2 * k2); k4 = (x + h * k3) - c_at(tc + h, x + h * k3)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); tc += h; xs.append(x)
        if x <= 0 or x > 1e3: break
    return np.array(xs)

lo, hi = 0.2, 50.0
for _ in range(60):
    mid = np.sqrt(lo * hi); xs = ray(mid)
    if len(xs) < 4001 or xs[-1] < mid: lo = mid
    else: hi = mid
xs = ray(np.sqrt(lo * hi))
tc_grid = np.linspace(tA, tB, len(xs))
print(f"SSH: x_s at start {xs[0]:.4f}, end {xs[-1]:.4f}, range {xs.min():.3f}..{xs.max():.3f}")

# fields in SSH gauge along the window
sel = (tauc >= tA) & (tauc <= tB)
xs_s = np.interp(tauc[sel], tc_grid, xs)
rs = xs_s * (Ts - Tc[sel])
tau = -np.log(rs)
def at(F, R):
    return np.array([np.interp(R_, r, F_) for F_, R_ in zip(F, R)])
p1 = rs * Pi[sel, 0]
mS = rs * (at(Phi[sel], rs) - at(Pi[sel], rs))
aS = at(a[sel], rs)
c0 = al[sel, 0] * aS / at(al[sel], rs)
print(f"tau span {tau[0]:.4f} .. {tau[-1]:.4f} = {tau[-1]-tau[0]:.4f}")
np.savez("/home/claude/dss/guess3.npz", tau=tau, p1=p1, mS=mS, aS=aS, c0=c0, Delta=Dl, Tstar=Ts)
fig, ax = plt.subplots(5, 1, figsize=(8, 10), sharex=False)
ax[0].plot(tc_grid, xs); ax[0].set_ylabel("x_s (central gauge)"); ax[0].set_xlabel("tau_c")
for axx, f, nm in zip(ax[1:], (p1, mS, aS, c0), ("p1", "m at SSH", "a at SSH", "c at centre")):
    axx.plot(tau, f, lw=1); axx.set_ylabel(nm)
fig.tight_layout(); fig.savefig("/home/claude/dss/guess3.png", dpi=80)

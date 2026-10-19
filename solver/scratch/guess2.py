import numpy as np, json
from scipy.optimize import least_squares, minimize_scalar
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

d = np.load("/home/claude/dss/nearcrit_slices.npz")
t, Tc, Phi, Pi, a, al, r = d["t"], d["Tc"], d["Phi"], d["Pi"], d["a"], d["alpha"], d["r"]
Pi0 = Pi[:, 0]
# central field phi0(T) by integrating dphi0/dT = Pi(0) (a = 1 at the centre)
phi0 = np.concatenate([[0.0], np.cumsum(0.5 * (Pi0[1:] + Pi0[:-1]) * np.diff(Tc))])
# echo events: extrema of phi0 (zeros of Pi0) and zeros of phi0 - mean, in the critical phase
ev = []
for f in (Pi0, phi0 - np.median(phi0)):
    s = np.where(np.sign(f[:-1]) != np.sign(f[1:]))[0]
    for i in s:
        T0 = Tc[i] - f[i] * (Tc[i + 1] - Tc[i]) / (f[i + 1] - f[i])
        ev.append((T0, i, f is Pi0))
ev.sort()
print("events (Tc, kind):")
for e in ev:
    print(f"  {e[0]:.6f} {'extremum' if e[2] else 'zero'}  alpha0={al[e[1],0]:.3e}")
np.save("/home/claude/dss/events.npy", np.array([[e[0], e[2]] for e in ev]))

# ---- accumulation time from the well-resolved events ----
good = np.array([e[0] for e in ev if 8.3 < e[0] < 9.115])
k = np.arange(len(good))
def res(q):
    Ts, t0, Dl = q
    return -np.log(np.maximum(Ts - good, 1e-12)) - (t0 + k * Dl / 4)
sol = least_squares(res, [good[-1] + 0.02, 0.0, 3.4])
Tstar, tau0, Dfit = sol.x
print(f"T* = {Tstar:.6f}, Delta from event timing = {Dfit:.4f}, rms {np.sqrt(np.mean(sol.fun**2)):.3e}, events used {len(good)}")
for sub in (slice(0, -1), slice(1, None)):
    s2 = least_squares(lambda q: (-np.log(np.maximum(q[0] - good[sub], 1e-12)) - (q[1] + k[sub] * q[2] / 4)), sol.x)
    print("   subset:", s2.x)

# ---- the past light cone of (T*, r=0), traced backwards in coordinate time ----
tstar = np.interp(Tstar, Tc, t)
ca = al / a                       # alpha / a on each slice
def speed(tt, rr):
    j = np.clip(np.searchsorted(t, tt) - 1, 0, len(t) - 2)
    f = (tt - t[j]) / (t[j + 1] - t[j])
    v0 = np.interp(rr, r, ca[j]); v1 = np.interp(rr, r, ca[j + 1])
    return (1 - f) * v0 + f * v1
rs = np.zeros_like(t)
mask = t < tstar
# integrate dr/d(-t) = alpha/a from (tstar, 0) backwards with RK4, recording at every slice time
rr, tt = 0.0, tstar
order = np.where(mask)[0][::-1]
for i in order:
    t_target = t[i]
    nsub = max(1, int(np.ceil((tt - t_target) / 2e-4)))
    h = (tt - t_target) / nsub
    for _ in range(nsub):
        k1 = speed(tt, rr); k2 = speed(tt - h / 2, rr + h / 2 * k1); k3 = speed(tt - h / 2, rr + h / 2 * k2); k4 = speed(tt - h, rr + h * k3)
        rr += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); tt -= h
    rs[i] = rr
sel = mask & (rs > 25 * (r[1] - r[0]))      # SSH resolved by at least ~25 grid cells
print(f"tstar (coordinate) = {tstar:.5f}; usable slices {sel.sum()}; r_s range {rs[sel].min():.4f} .. {rs[sel].max():.4f}")

tau = -np.log(rs[sel])
def at(F, R):
    return np.array([np.interp(R_, r, F_) for F_, R_ in zip(F, R)])
R = rs[sel]
p1 = R * Pi[sel, 0]
mS = R * (at(Phi[sel], R) - at(Pi[sel], R))
aS = at(a[sel], R)
c0 = al[sel, 0] * aS / at(al[sel], R)
np.savez("/home/claude/dss/guess_tau.npz", tau=tau, p1=p1, mS=mS, aS=aS, c0=c0, Tstar=Tstar, Dfit=Dfit,
         Tc=Tc[sel], rs=R)
fig, ax = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
for axx, f, nm in zip(ax, (p1, mS, aS, c0), ("p1 = r_s Pi(0)", "m at SSH", "a at SSH", "c at centre")):
    axx.plot(tau, f, lw=1); axx.set_ylabel(nm)
ax[-1].set_xlabel("tau = -ln r_s")
fig.tight_layout(); fig.savefig("/home/claude/dss/guess_tau.png", dpi=90)
print("tau range", tau.min(), tau.max())

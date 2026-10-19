import numpy as np
d = np.load("/home/claude/dss/nearcrit_slices.npz"); g = np.load("/home/claude/dss/guess3.npz")
t, Tc, Phi, Pi, a, al, r = d["t"], d["Tc"], d["Phi"], d["Pi"], d["a"], d["alpha"], d["r"]
Ts = float(g["Tstar"]); tau = g["tau"]
tauc = -np.log(np.maximum(Ts - Tc, 1e-30))
E1, E3 = 8.588415, 9.110792
sel = np.where((Tc >= E1) & (Tc <= E3))[0]
rs = np.exp(-tau)
worst = []
for k, i in enumerate(sel):
    R = rs[k]; msk = r <= R
    xx = r[msk] / R
    cc = (al[i, msk] / a[i, msk]) / np.interp(R, r, al[i] / a[i])
    j = np.argmin(cc - xx)
    worst.append((tau[k], (cc - xx)[j], xx[j], cc[0]))
w = np.array(worst)
for row in w[::len(w)//25]:
    print(f"tau={row[0]:6.3f}  min(c-x)={row[1]:+.4f} at x={row[2]:.3f}   c(0)={row[3]:.3f}")
print("overall min(c-x) over the window:", w[:,1].min())

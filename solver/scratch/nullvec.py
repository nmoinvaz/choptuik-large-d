import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
from spec_long import make_long
from spec_c import freeze_steps, jacobian_cs
NODES_L = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6]
D = float(sys.argv[1]); z = np.load(sys.argv[2])
S = freeze_steps(make_long(23, D - 2, nodes=NODES_L, Ks=60), z[-1])
F, J = jacobian_cs(S, z); np.save(sys.argv[2].replace(".npy","_Jcs.npy"), J)
U, s, Vt = np.linalg.svd(J); c = U.T @ F
sizes, K, L = S.layout(); offs = np.concatenate([[0], np.cumsum(sizes)])
names = ["p1c", "c0c"] + [f"{f}@{x}" for x in NODES_L for f in "pmac"] + ["mS", "aS"]
pars = S.parities()
print("sigma/smax  |U.F|  Vdelta  | top blocks (energy)  | dominant harmonic in top block")
for i in range(len(s) - 26, len(s)):
    v = Vt[i]; e = np.array([np.sum(v[offs[b]:offs[b+1]]**2) for b in range(len(sizes))])
    top = np.argsort(e)[::-1][:3]
    b = top[0]; cb = v[offs[b]:offs[b+1]]
    if pars[b] == "odd": m = len(S.kodd); amp = np.hypot(cb[:m], cb[m:]); kk = S.kodd
    else: m = len(S.keven) - 1; amp = np.r_[abs(cb[0]), np.hypot(cb[1:1+m], cb[1+m:])]; kk = S.keven
    print(f"{s[i]/s[0]:.1e}  {abs(c[i]):.1e}  {abs(v[-1]):.1e} | " + ", ".join(f"{names[t]}:{e[t]:.2f}" for t in top) + f" | k={kk[np.argmax(amp)]} spread {np.round(amp/amp.max(),2)[:8]}")

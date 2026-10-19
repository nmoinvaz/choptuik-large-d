import numpy as np
from hpkernel.driver import make_solver, seed_from_f64
M = 48; K = 10
D, zh, zl = seed_from_f64("/home/claude/dss/ne7_js_rk3_s111111118_D38.1740.npy", M, kmax=11)
s = make_solver(D, M, kt=36, kc=40, ks=150)
s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
dh, dl, Fh, Fl, a, b = s.newton(zh, zl)
names = ["p1", "c0"] + [f"{q}{i}" for i in range(K) for q in "pmac"] + ["m_h", "a_h"]
def harm(f, odd):
    full = np.r_[f, -f] if odd else np.r_[f, f]
    return np.abs(np.fft.rfft(full)) / M
for j, nm in enumerate(names):
    d = dh[j * M:(j + 1) * M]
    z = zh[j * M:(j + 1) * M]
    odd = nm[0] in "pm"
    H = harm(d, odd)
    ks = np.arange(len(H))
    sel = ks[(ks % 2 == 1) if odd else (ks % 2 == 0)]
    top = sel[np.argsort(H[sel])[::-1][:3]]
    print(f"{nm:4s} |dz| {np.abs(d).max():.1e}  |z| {np.abs(z).max():.1e}  biggest harmonics " + " ".join(f"k{k}:{H[k]:.0e}" for k in top))
print("dDelta", dh[-1])

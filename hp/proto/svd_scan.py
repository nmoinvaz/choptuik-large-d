import sys, time
import numpy as np
from hpkernel.driver import make_solver, seed_from_f64
K = 10
for M in (16, 24, 32, 48):
    D, zh, zl = seed_from_f64("/home/claude/dss/ne7_js_rk3_s111111118_D38.1740.npy", M, kmax=11)
    s = make_solver(D, M, kt=36, kc=40, ks=150)
    s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
    Jh, Jl = s.jacobian(zh, zl)
    U, sv, Vt = np.linalg.svd(Jh)
    r = sv / sv[0]
    print(f"M {M}: L {s.size} smax {sv[0]:.1e}; smallest s/smax: " + " ".join(f"{v:.0e}" for v in r[-8:][::-1]), flush=True)
    # structure of the smallest right singular vector: which field / harmonic
    names = ["p1", "c0"] + [f"{q}{i}" for i in range(K) for q in "pmac"] + ["m_h", "a_h"]
    for idx in (-1, -2, -3):
        v = Vt[idx]
        e = [np.linalg.norm(v[j * M:(j + 1) * M]) for j in range(len(names))]
        top = np.argsort(e)[::-1][:4]
        fld = names[top[0]]; j = top[0]
        f = v[j * M:(j + 1) * M]
        full = np.r_[f, -f] if fld[0] in "pm" else np.r_[f, f]
        H = np.abs(np.fft.rfft(full)); kmaxh = np.argmax(H)
        print(f"   v[{idx}] s/smax {r[idx]:.0e}: fields " + " ".join(f"{names[t]}:{e[t]:.2f}" for t in top) + f"  dominant harmonic {kmaxh}  Delta comp {v[-1]:.1e}")

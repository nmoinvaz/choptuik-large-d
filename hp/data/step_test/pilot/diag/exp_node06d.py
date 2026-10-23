"""T43 pilot diagnosis, part 4: are the dd Jacobian's horizon-data columns right? At z* (base's solution at D 60.023452,
in the 0.3-0.7 and 0.3-0.6 layouts), each horizon column of s.jacobian against central differences of the dd residual
at steps 1e-18, 1e-20, 1e-22, and against the same column at z* + 1e-30 in Delta (noise).
    python exp_node06d.py HP_DIR PILOT_DIR"""
import ctypes, os, sys
import numpy as np

H, P = sys.argv[1], sys.argv[2]
ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)    # below normal
sys.path.insert(0, H)
os.chdir(H)
from hpkernel.driver import dd_add, load, make_solver, renode  # noqa: E402

M = 32
N6, N7 = [0.3, 0.4, 0.5, 0.6], [0.3, 0.4, 0.5, 0.6, 0.7]
bh, bl, bm = load(os.path.join(P, "base_D60.00_D60.0235.npz"))
D = bm["D"]


def Fd(s, zh, zl, j, d):
    """central difference of the dd residual along coordinate j, step d (added in double-double)"""
    u = np.zeros_like(zh)
    u[j] = d
    ph, pl = dd_add(zh, zl, u, np.zeros_like(u))
    mh, ml = dd_add(zh, zl, -u, np.zeros_like(u))
    Ph, Pl = s.residual(ph, pl)
    Mh, Ml = s.residual(mh, ml)
    return ((np.asarray(Ph) - np.asarray(Mh)) + (np.asarray(Pl) - np.asarray(Ml))) / (2 * d)


for nodes, ks in ((N7, 100), (N6, 160)):
    if nodes == N7:
        sh, sl = bh, bl
    else:
        s7 = make_solver(D, M, nodes=N7, kt=36, kc=220, ks=100, threads=6)
        sh, sl = renode(s7, bh, bl, N7, N6, M)
    s = make_solver(D, M, nodes=nodes, kt=36, kc=220, ks=ks, threads=6, kfreeze=13)
    s.set_segs(s.plan(sh, sl, 1e-32, 0.05))
    Jh, Jl = s.jacobian(sh, sl)
    J = np.asarray(Jh) + np.asarray(Jl)
    u = np.zeros_like(sh)
    u[-1] = 1e-30
    th, tl = dd_add(sh, sl, u, np.zeros_like(u))
    J2h, J2l = s.jacobian(th, tl)
    J2 = np.asarray(J2h) + np.asarray(J2l)
    h0 = 2 * M + 4 * M * len(nodes)
    print(f"\n==== layout {nodes}, ks {ks}: horizon columns {h0}..{sh.size - 2}", flush=True)
    worst = {d: 0.0 for d in (1e-18, 1e-20, 1e-22)}
    noise = 0.0
    rows = []
    for j in range(h0, sh.size - 1):
        cn = np.abs(J[:, j]).max()
        if cn == 0:
            continue
        noise = max(noise, np.abs(J2[:, j] - J[:, j]).max() / cn)
        errs = {}
        for d in worst:
            errs[d] = np.abs(Fd(s, sh, sl, j, d) - J[:, j]).max() / cn
            worst[d] = max(worst[d], errs[d])
        rows.append((j, cn, errs))
    for j, cn, errs in rows[:: max(1, len(rows) // 8)] + rows[-2:]:
        print(f"col {j}: |J col| {cn:.1e}; |FD - J|/|J col| " + " ".join(f"h {d:.0e}: {e:.1e}" for d, e in errs.items()),
              flush=True)
    print("worst over the horizon columns: " + " ".join(f"h {d:.0e}: {e:.1e}" for d, e in worst.items())
          + f"; J(z* + 1e-30 Delta) - J(z*): {noise:.1e} relative", flush=True)
print("done", flush=True)

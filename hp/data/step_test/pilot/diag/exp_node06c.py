"""T43 pilot diagnosis, part 3: is the dd Jacobian consistent with the residual near z*? At D 60.023452, in both
layouts, with e the run's seed error (as in exp_node06b.py) split into the horizon data and Delta: the residual
change F(z*+e) - F(z*) against J(z*) e and J(z*+e) e, and how far J moves between z* and z*+e.
    python exp_node06c.py HP_DIR PILOT_DIR"""
import ctypes, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
H, P = sys.argv[1], sys.argv[2]
sys.argv = [sys.argv[0], H, P]
sys.path.insert(0, HERE)
import exp_node06b as b  # noqa: E402  (its module body builds the seeds and prints part 2 again)
from hpkernel.driver import dd_add, dd_norm, load, make_solver, renode  # noqa: E402
from pinning import predict_dd  # noqa: E402

M = 32
bh, bl, bm = load(os.path.join(P, "base_D60.00_D60.0235.npz"))
D = bm["D"]
for nodes, a in ((b.N7, b.a7), (b.N6, b.a6)):
    print(f"\n==== layout {nodes}, ks {a.ks}", flush=True)
    if nodes == b.N7:
        sh, sl = bh, bl
    else:
        s7 = make_solver(D, M, nodes=b.N7, kt=36, kc=220, ks=100, threads=6)
        sh, sl = renode(s7, bh, bl, b.N7, b.N6, M)
    s = make_solver(D, M, nodes=nodes, kt=36, kc=220, ks=a.ks, threads=6, kfreeze=13)
    s.set_segs(s.plan(sh, sl, 1e-32, 0.05))
    zp, zpl = predict_dd(b.build(nodes, a), D, 4)
    e = (zp - sh) + (zpl - sl)
    K = len(nodes)
    h0 = 2 * M + 4 * M * K
    print(f"horizon block {h0}..{e.size - 2}: |e| {np.abs(e[h0:-1]).max():.1e}; Delta: e {e[-1]:+.1e}", flush=True)
    F0h, F0l = s.residual(sh, sl)
    F0 = np.asarray(F0h) + np.asarray(F0l)
    J0h, J0l = s.jacobian(sh, sl)
    J0 = np.asarray(J0h) + np.asarray(J0l)
    for name, idx in (("all", slice(None)), ("horizon data", slice(h0, e.size - 1)), ("Delta", slice(e.size - 1, None))):
        eb = np.zeros_like(e)
        eb[idx] = e[idx]
        th, tl = dd_add(sh, sl, eb, np.zeros_like(eb))
        F1h, F1l = s.residual(th, tl)
        dF = (np.asarray(F1h) - np.asarray(F0h)) + (np.asarray(F1l) - np.asarray(F0l))
        J1h, J1l = s.jacobian(th, tl)
        J1 = np.asarray(J1h) + np.asarray(J1l)
        r0 = np.abs(J0 @ eb - dF).max() / np.abs(dF).max()
        r1 = np.abs(J1 @ eb - dF).max() / np.abs(dF).max()
        dJ = np.abs(J1 - J0)
        i, j = np.unravel_index(np.argmax(dJ), dJ.shape)
        print(f"{name:13s}: |dF| {np.abs(dF).max():.1e}; |J(z*) e - dF|/|dF| {r0:.1e}; |J(z*+e) e - dF|/|dF| {r1:.1e}; "
              f"max|J(z*+e) - J(z*)| {dJ.max():.1e} at ({i}, {j}) where |J| {abs(J0[i, j]):.1e}; "
              f"columns changed by > 1e-6 relative: {int((dJ.max(axis=0) > 1e-6 * np.abs(J0).max(axis=0)).sum())}",
              flush=True)
print("done", flush=True)

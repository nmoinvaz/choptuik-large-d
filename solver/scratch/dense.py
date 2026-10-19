"""Evaluate a converged critical solution on a dense (tau, x) grid, for figures."""
import sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
from run_spec import make

def dense(D, kc, fname, nx_per_seg=12, ntau=None):
    S = make(kc, n=D - 2)
    z = np.load(fname)
    parts, Dl = S.split(z[None, :])
    pars = S.parities()
    F = [S.phys(c, p) for c, p in zip(parts, pars)]
    om = 2 * np.pi / Dl
    K = len(S.nodes)
    states = [np.stack(F[2 + 4 * i:6 + 4 * i]) for i in range(K)]
    xs, fields = [], []
    # node-to-node segments, recording intermediate points
    for i in range(K - 1):
        xa, xb = S.nodes[i], S.nodes[i + 1]
        s = states[i]
        sub = np.linspace(xa, xb, nx_per_seg + 1)
        for j in range(nx_per_seg):
            xs.append(sub[j]); fields.append(s[:, 0])
            s = S.seg(s, sub[j], sub[j + 1], om)
    # last node to the SSH: integrate backwards from the SSH start and record
    P, M, A, C = S.ssh_series(F[-2], F[-1], om)
    s = S.trunc(np.stack([S.eval_series(Q, S.u0) for Q in (P, M, A, C)]))
    sub = np.linspace(1 - S.u0, S.nodes[-1], nx_per_seg + 1)
    tail = []
    for j in range(nx_per_seg):
        tail.append((sub[j], s[:, 0])); s = S.seg(s, sub[j], sub[j + 1], om)
    xs.append(S.nodes[-1]); fields.append(states[-1][:, 0])
    for x, f in reversed(tail):
        xs.append(x); fields.append(f)
    xs.append(1.0)
    st1 = np.stack([S.eval_series(Q, 0.0) for Q in (P, M, A, C)])[:, 0]
    fields.append(st1)
    # centre to first node
    Pc, Mc, Ac, Cc = S.centre_series(F[0], F[1], om)
    xc = np.geomspace(1e-3, S.nodes[0], 6)[:-1]
    cen = [(0.0, np.stack([np.zeros(S.N), np.zeros(S.N), np.ones(S.N), F[1][0]]))]
    s = S.trunc(np.stack([S.eval_series(Q, S.x0) for Q in (Pc, Mc, Ac, Cc)]))
    x_prev = S.x0
    for x in xc:
        s = S.integrate_log(s, x_prev, x, 20, om); x_prev = x
        cen.append((x, s[:, 0]))
    xs = [c[0] for c in cen] + xs
    fields = [c[1] for c in cen] + fields
    X = np.array(xs); Fd = np.array(fields)                # (nx, 4, N)
    return X, Fd, float(Dl[0]), S.N

if __name__ == "__main__":
    D, kc, fname, out = float(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
    X, Fd, Dl, N = dense(D, kc, fname)
    p, m, a = Fd[:, 0], Fd[:, 1], Fd[:, 2]
    mu = 1 - 1 / a ** 2                                    # = (2m/r)^... compactness-like (D=4: 2m/r)
    phi_r = (p + m) / 2                                     # r Phi
    json.dump(dict(D=D, Delta=Dl, x=X.round(5).tolist(), mu=mu.round(5).tolist(), rPhi=phi_r.round(5).tolist(), N=N),
              open(out, "w"))
    print("x points", len(X), "tau points", N, "mu range", float(mu.min()), float(mu.max()))

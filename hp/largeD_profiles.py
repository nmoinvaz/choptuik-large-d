"""Dense profiles of p, m, a, c in x for one DD solution: centre series (x <= x0), Taylor stepping between nodes,
horizon series in u = 1 - x (x >= x_K). Fields on the half tau grid (M points); p, m odd, a, c even."""
import numpy as np
from hpkernel.driver import make_solver, load


def solver_for(f, ks=None):
    zh, zl, meta = load(f)
    D, M, nodes = meta["D"], meta["M"], meta["nodes"]
    s = make_solver(D, M, nodes=nodes, kt=meta.get("kt", 36), kc=meta.get("kc", 420), ks=ks or meta.get("ks", 100),
                    kfreeze=13)
    s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
    return s, zh, zl, meta


def profile(f, xs):
    s, zh, zl, meta = solver_for(f)
    M, nodes = meta["M"], meta["nodes"]
    K = len(nodes)
    z = zh + zl
    ch, cl = s.series(zh, zl, 0)
    C = (np.asarray(ch) + np.asarray(cl)).reshape(-1, 4, M)
    hh, hl = s.series(zh, zl, 1)
    H = (np.asarray(hh) + np.asarray(hl)).reshape(-1, 4, M)
    out = {}
    for x in xs:
        if x <= nodes[0]:
            out[x] = np.einsum("k,kqm->qm", x ** np.arange(C.shape[0]), C)
        elif x >= nodes[-1]:
            u = 1 - x
            out[x] = np.einsum("k,kqm->qm", u ** np.arange(H.shape[0]), H)
        else:
            b = max(i for i in range(K) if nodes[i] <= x)
            o = 2 * M + 4 * M * b
            sh, sl = zh[o:o + 4 * M].copy(), zl[o:o + 4 * M].copy()
            if x == nodes[b]:
                st = sh + sl
            else:
                pts = list(np.linspace(nodes[b], x, max(2, int(np.ceil((x - nodes[b]) / 0.01)) + 1)))
                rh, rl = s.integrate(zh, zl, sh, sl, pts)
                st = np.asarray(rh) + np.asarray(rl)
            out[x] = st.reshape(4, M)
    return meta["D"], float(z[-1]), M, out, (C, H, nodes)


def harmonics(f_half, odd):
    full = np.r_[f_half, -f_half] if odd else np.r_[f_half, f_half]
    return np.fft.rfft(full) / len(f_half)

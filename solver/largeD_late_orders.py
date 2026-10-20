"""The 1/n expansion of the DSS equations to high order, numerically (same recursion as solver/largeD_expand.py,
checked against it at first order to 1e-12).

Each order is a polynomial in x with theta-grid coefficients, stored as an array (deg + 1, NT); theta derivatives
are spectral. For a given F(theta) and omega (higher w_k and the free K_k set to zero) this builds S_k, D_k, G_k,
B_k up to order K and reports their growth, to find the factorial divergence that locates the beyond-all-orders
terms.

    python3 late_orders.py K [sine|path.npz] [omega] [NT]
"""
import sys
import numpy as np

NT = 256
KK = np.fft.fftfreq(NT, 1 / NT)


def z(deg=0):
    return np.zeros((deg + 1, NT))


def pmul(a, b):
    out = np.zeros((a.shape[0] + b.shape[0] - 1, NT))
    for i in range(a.shape[0]):
        if a[i].any():
            out[i:i + b.shape[0]] += a[i][None, :] * b
    return out


def padd(*ps):
    n = max(p.shape[0] for p in ps)
    out = np.zeros((n, NT))
    for p in ps:
        out[:p.shape[0]] += p
    return out


def dx(a):
    if a.shape[0] == 1:
        return z()
    return a[1:] * np.arange(1, a.shape[0])[:, None]


def mulx(a, k=1):
    return np.vstack([np.zeros((k, NT)), a])


def dth(a):
    return np.real(np.fft.ifft(np.fft.fft(a, axis=1) * (1j * KK)[None, :], axis=1))


def at(a, xv):
    return (a * (xv ** np.arange(a.shape[0]))[:, None]).sum(0)


def coef(a, b, k, lo=0):
    """[a b]_k = sum_{i=lo..k} a_i b_{k-i} for series given as lists of polynomials."""
    acc = z()
    for i in range(lo, k + 1):
        if i < len(a) and k - i < len(b):
            acc = padd(acc, pmul(a[i], b[k - i]))
    return acc


def project(a, parity, jmax):
    """keep harmonics of the given parity (1 odd, 0 even) with |j| <= jmax (removes roundoff at high wavenumbers,
    which the spectral theta derivatives would otherwise amplify order by order)"""
    j = np.abs(KK)
    mask = ((j % 2) == parity) & (j <= jmax)
    return np.real(np.fft.ifft(np.fft.fft(a, axis=1) * mask[None, :], axis=1))


def run(Fg, omega, K, verbose=True, jmax_of=lambda k: 2 * k + 3):
    one = np.ones((1, NT))
    S = [z()] * (K + 2)
    D = [mulx(2 * Fg[None, :])] + [z()] * (K + 1)
    G = [z()] * (K + 2)
    B = [z()] * (K + 3)
    c = [one] + [z()] * (K + 1)               # c = 1 - sum eps^k G_k
    inv = [one]                                # 1 / (c (1 - eps)(1 + B)), extended as orders complete
    stats = []
    for k in range(1, K + 1):
        P = [padd(S[i], D[i]) / 2 for i in range(k)]
        M = [padd(S[i], -D[i]) / 2 for i in range(k)]
        # B_{k+1} = [(1+B) q2]_{k-1} - x B_k,x - ([(1+B) B]_{k+1} - [(1+B) B]_k), excluding B_{k+1} itself
        q2 = [padd(coef(P, P, j), coef(M, M, j)) for j in range(k)]
        opB = [one] + B[1:]
        t2 = coef(opB, q2, k - 1)
        BB = lambda m: padd(B[m] if m <= k else z(), coef(B, B, m, lo=1))
        t3 = padd(BB(k + 1), -BB(k))
        B[k + 1] = project(padd(t2, -mulx(dx(B[k])), -t3), 0, jmax_of(k + 1))
        # G_k: -x G_k,x = [(1-eps) c B]_{k+1} = [cB]_{k+1} - [cB]_k ; G_k(1) = 0
        rhs = padd(coef(c, B, k + 1), -coef(c, B, k))
        if np.abs(rhs[0]).max() > 1e-12 * max(1.0, np.abs(rhs).max()):
            print(f"  order {k}: E1 x^0 term {np.abs(rhs[0]).max():.2e}")
        gk = np.zeros_like(rhs)
        for j in range(1, rhs.shape[0]):
            gk[j] = -rhs[j] / j
        gk[0] = -gk[1:].sum(0)                # G_k(1) = 0
        gk = project(gk, 0, jmax_of(k))
        G[k] = gk
        c[k] = -gk
        # inverse of den = c (1 - eps)(1 + B) up to order k-1
        den = lambda m: padd(coef(c, [one] + B[1:], m), -coef(c, [one] + B[1:], m - 1) if m >= 1 else z())
        while len(inv) < k:
            m = len(inv)
            acc = z()
            for i in range(1, m + 1):
                acc = padd(acc, pmul(den(i), inv[m - i]))
            inv.append(-acc)
        # S_k = [num inv]_{k-1}, num_m = x^2 D_m,x - x [c S_x]_m + omega x D_m,th
        num = [padd(mulx(dx(D[m]), 2), -mulx(coef(c, [dx(s) for s in S], m)), omega * mulx(dth(D[m])))
               for m in range(k)]
        S[k] = project(coef(num, inv, k - 1), 1, jmax_of(k))
        # D_k: x D_k,x - D_k = R_k
        cB = [coef(c, B, m) for m in range(k + 2)]
        cBD = lambda m: coef(cB, D[:k] + [z()], m)
        R = padd(mulx(dx(S[k]), 2), omega * mulx(dth(S[k])), -cBD(k + 1), cBD(k))
        for i in range(1, k + 1):
            R = padd(R, -pmul(c[i], padd(mulx(dx(D[k - i])), -D[k - i])))
        if R.shape[0] > 1 and np.abs(R[1]).max() > 1e-9 * max(1.0, np.abs(R).max()):
            print(f"  order {k}: SOLVABILITY x^1 term {np.abs(R[1]).max():.2e} (|R| {np.abs(R).max():.2e})")
        dk = np.zeros_like(R)
        for j in range(R.shape[0]):
            if j != 1:
                dk[j] = R[j] / (j - 1)
        D[k] = project(dk, 1, jmax_of(k))
        st = dict(k=k, S=np.abs(S[k]).max(), D=np.abs(D[k]).max(), G=np.abs(G[k]).max(), B=np.abs(B[k + 1]).max(),
                  S1=np.abs(at(S[k], 1.0)).max(), D1=np.abs(at(D[k], 1.0)).max(), Dhalf=np.abs(at(D[k], 0.5)).max())
        stats.append(st)
        if verbose:
            r = (st["D1"] / stats[-2]["D1"]) if len(stats) > 1 else 0
            print(f"order {k:3d}: |D_k(1)| {st['D1']:.3e} |D_k(.5)| {st['Dhalf']:.3e} |S_k(1)| {st['S1']:.3e} "
                  f"|G_k| {st['G']:.3e} |B_k+1| {st['B']:.3e}  D(1) ratio {r:.3e}  ratio/k {r / k:.3f}", flush=True)
    return S, D, G, B, stats


if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    src = sys.argv[2] if len(sys.argv) > 2 else "sine"
    omega = float(sys.argv[3]) if len(sys.argv) > 3 else 2 * np.pi / 1.64389375
    th = 2 * np.pi * np.arange(NT) / NT
    if src == "sine":
        Fg = 2.72 * np.sin(th)
    else:
        import json
        d = np.load(src, allow_pickle=True)
        meta = json.loads(str(d["meta"])); Mh = meta["M"]
        p1 = d["zh"][:Mh] + d["zl"][:Mh]
        full = np.r_[p1, -p1] * np.sqrt(meta["D"] - 2)
        Fg = np.fft.irfft(np.fft.rfft(full), NT) * NT / len(full)
    run(Fg, omega, K)

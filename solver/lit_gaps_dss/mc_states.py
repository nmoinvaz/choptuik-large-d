"""The momentum constraint in slaved form on stored hp states (notes/LIT_GAPS_DSS.md, lead 1).

In dss.py's variables (n = D - 2, B = 1/a^2, q = ((x - c) p^2 + (x + c) m^2)/x) the momentum constraint with a_x
taken from the Hamiltonian one is the periodic linear tau-equation
    omega dB/dtheta = [(n - 1) + q] B - (n - 1)                                  (S)
(Gundlach 1997 Eq. (19) in D dimensions; MGG 2003 Eq. (17) at n = 2). At the SSH x = c = 1, q = 2 m_h^2.
For each state and each node (and the SSH) this prints
  r_S   = max|residual of (S)| / max|(n - 1)(B - 1)|      (the stored a against (S))
  r_sl  = max|B_slaved - B| / max|B - 1|                   (B solved from (S) given p, m, c, against the stored B)
  cond  = (n - 1) * ||L^-1||_2 for L = omega d/dtheta - (n - 1) - q    (1 = perfectly conditioned)
B - 1 is formed from a - 1 summed part by part, so the cancellation in 1/a^2 - 1 costs no digits.
Usage (from the repository root, states from the raw-data repository):
  git -C ../choptuik-large-d-raw archive origin/main T73/t73/pin5_D68.0835.npz T139-macm3/chain/anm_pin3_D100.0000.npz \
    I903-macm3/chain_k135/anm_pin3_D226.1312.npz I966-dh370/ks_chain3b/leg15_D263.4862_ks293/anm_pin3_D263.5987.npz \
    | tar -x -C STATES
  python3 solver/lit_gaps_dss/mc_states.py STATES/T73/t73/pin5_D68.0835.npz ... [--mutate=flux|n1]
  flux: swaps (x - c) and (x + c) in q; n1: uses 1 in place of n - 1 (the D = 4 coefficient). Both must fail.
Pass: worst r_S <= 1e-12. Exits 1 on a fail.
"""
import json
import sys

import numpy as np


def load(f):
    z = np.load(f, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    parts = [k for k in ("zh", "zl", "z2", "z3", "z4", "z5", "z6", "z7") if k in z.files]
    y = sum(z[k].astype(float) for k in parts)
    ym1 = (z["zh"].astype(float) - 1.0) + sum(z[k].astype(float) for k in parts[1:])   # y - 1 without cancellation
    return meta, y, ym1


def full(f, odd):
    return np.concatenate([f, -f if odd else f])


def dtheta(f):
    N = len(f)
    F = np.fft.rfft(f)
    k = np.arange(len(F))
    F = F * 1j * k
    F[-1] = 0.0
    return np.fft.irfft(F, n=N)


def Lmat(q, n1, omega):
    """Matrix of B -> omega B_theta - (n1 + q) B on the full grid (N points over 2 pi)."""
    N = len(q)
    E = np.eye(N)
    Dm = np.stack([dtheta(E[i]) for i in range(N)], axis=1)      # column i = d/dtheta of basis vector i
    return omega * Dm - np.diag(n1 + q)


def check(x, p, m, a, c, n, omega, mutate, am1):
    n1 = 1.0 if mutate == "n1" else n - 1
    B = 1.0 / a**2
    Bm1 = -am1 * (a + 1) / a**2                     # B - 1 from a - 1, accurate to the state's precision
    if mutate == "flux":
        q = ((x + c) * p**2 + (x - c) * m**2) / x
    else:
        q = ((x - c) * p**2 + (x + c) * m**2) / x
    res = omega * dtheta(Bm1) - n1 * Bm1 - q * B
    r_S = np.abs(res).max() / np.abs((n - 1) * Bm1).max()
    L = Lmat(q, n1, omega)
    Bs_m1 = np.linalg.solve(L, q)                 # (S) for B - 1:  L (B - 1) = q
    r_sl = np.abs(Bs_m1 - Bm1).max() / np.abs(Bm1).max()
    s = np.linalg.svd(L, compute_uv=False)
    return r_S, r_sl, (n - 1) / s.min()


def main():
    files = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    mutate = opts.get("mutate", "")
    worst = 0.0
    for f in files:
        meta, y, ym1 = load(f)
        M, nodes, D = meta["M"], meta["nodes"], meta["D"]
        n = D - 2
        K = len(nodes)
        Delta = y[-1]
        omega = 2 * np.pi / Delta
        print("D %.4f  Delta %.12f  M %d  nodes %s" % (D, Delta, M, nodes))
        for i, x in enumerate(nodes):
            o = 2 * M + 4 * M * i
            p, m, a, c = (y[o + j * M: o + (j + 1) * M] for j in range(4))
            am1 = ym1[o + 2 * M: o + 3 * M]
            r = check(x, full(p, 1), full(m, 1), full(a, 0), full(c, 0), n, omega, mutate, full(am1, 0))
            worst = max(worst, r[0])
            print("  x %.2f  r_S %.2e  r_sl %.2e  cond %.4f" % ((x,) + r))
        o = 2 * M + 4 * M * K
        mh, ah = y[o: o + M], y[o + M: o + 2 * M]
        r = check(1.0, np.zeros(2 * M), full(mh, 1), full(ah, 0), np.ones(2 * M), n, omega, mutate,
                  full(ym1[o + M: o + 2 * M], 0))
        worst = max(worst, r[0])
        print("  SSH     r_S %.2e  r_sl %.2e  cond %.4f" % r)
    ok = worst <= 1e-12
    print("worst r_S %.2e: %s" % (worst, "pass" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

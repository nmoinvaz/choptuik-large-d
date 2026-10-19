import sys, time, warnings
import numpy as np
from scipy.optimize import least_squares
warnings.filterwarnings("ignore")
from dss import DSSMulti
from guess_ms import build

def parities(K):
    return ["odd", "even"] + ["odd", "odd", "even", "even"] * K + ["odd", "even"]

def filt(z, N, kcut, K):
    H = N // 2; out = z.copy()
    for b, par in enumerate(parities(K)):
        h = z[b * H:(b + 1) * H]
        f = np.concatenate([h, -h]) if par == "odd" else np.concatenate([h, h])
        F = np.fft.rfft(f); k = np.arange(len(F))
        F *= np.exp(-36 * (k / kcut) ** 8)
        out[b * H:(b + 1) * H] = np.fft.irfft(F, n=N)[:H]
    return out

def solve(S, z, maxfev=60, verbose=2):
    cache = {}
    def fun(z):
        return S.residual(z[None, :])[0]
    def jac(z):
        F, J = S.jacobian(z)
        return J
    t0 = time.time()
    r = least_squares(fun, z, jac=jac, method="lm", x_scale="jac", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=maxfev, verbose=verbose)
    return r.x, np.linalg.norm(r.fun), time.time() - t0

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    nodes = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    S = DSSMulti(nodes=nodes, n=2.0, N=N, Ks=8, u0=0.03, seg_h=0.002, log_steps=150)
    z = build(N, nodes)
    z = filt(z, N, 0.75 * N / 2, len(nodes))
    print(f"unknowns {len(z)}; guess |F| = {np.linalg.norm(S.residual(z[None, :])[0]):.3e}", flush=True)
    z, nF, w = solve(S, z, maxfev=int(sys.argv[2]) if len(sys.argv) > 2 else 60)
    print(f"final |F| = {nF:.3e}   Delta = {z[-1]!r}   ({w:.0f} s)")
    np.save(f"/home/claude/dss/sol_ms_N{N}.npy", z)

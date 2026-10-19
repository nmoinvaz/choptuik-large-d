import sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from dss import DSSMulti
from solve_ms import parities, solve

def regrid(z, N_old, N_new, K):
    H = N_old // 2; out = []
    for b, par in enumerate(parities(K)):
        h = z[b * H:(b + 1) * H]
        f = np.concatenate([h, -h]) if par == "odd" else np.concatenate([h, h])
        F = np.fft.rfft(f); G = np.zeros(N_new // 2 + 1, complex)
        m = min(len(F) - 1, len(G) - 1); G[:m] = F[:m] * (N_new / N_old)
        out.append(np.fft.irfft(G, n=N_new)[:N_new // 2])
    return np.concatenate(out + [z[-1:]])

nodes = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
if __name__ == "__main__":
    N_old, N_new = int(sys.argv[1]), int(sys.argv[2])
    seg_h = float(sys.argv[3]) if len(sys.argv) > 3 else 0.002
    tag = sys.argv[4] if len(sys.argv) > 4 else ""
    z = np.load(f"/home/claude/dss/sol_ms_N{N_old}.npy")
    z = regrid(z, N_old, N_new, len(nodes))
    S = DSSMulti(nodes=nodes, n=2.0, N=N_new, Ks=8, u0=0.03, seg_h=seg_h, log_steps=int(150 * 0.002 / seg_h))
    print(f"N={N_new} seg_h={seg_h}: start |F| = {np.linalg.norm(S.residual(z[None, :])[0]):.3e}", flush=True)
    z, nF, w = solve(S, z, maxfev=40, verbose=0)
    print(f"N={N_new} seg_h={seg_h}: final |F| = {nF:.3e}   Delta = {z[-1]!r}   ({w:.0f} s)", flush=True)
    np.save(f"/home/claude/dss/sol_ms_N{N_new}{tag}.npy", z)

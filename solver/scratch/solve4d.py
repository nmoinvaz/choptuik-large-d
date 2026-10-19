import sys, time, json
import numpy as np
from dss import DSS

def resample(tau, f, N, D, parity):
    tg = tau[0] + np.arange(N) / N * D
    g = np.interp(tg, tau, f)
    F = np.fft.rfft(g)
    k = np.arange(len(F))
    F[(k % 2 == 0) if parity == "odd" else (k % 2 == 1)] = 0
    return np.fft.irfft(F, n=N)

def shift(f, s, D):
    """f(tau + s) by Fourier interpolation."""
    N = f.shape[-1]; F = np.fft.rfft(f); k = np.arange(len(F))
    return np.fft.irfft(F * np.exp(1j * k * 2 * np.pi / D * s), n=N)

def make_guess(N):
    g = np.load("/home/claude/dss/guess3.npz")
    D = float(g["Delta"]); tau = g["tau"]
    p1 = resample(tau, g["p1"], N, D, "odd"); c0 = resample(tau, g["c0"], N, D, "even")
    mS = resample(tau, g["mS"], N, D, "odd"); aS = resample(tau, g["aS"], N, D, "even")
    # rotate so the first sine coefficient of p1 vanishes (the phase condition)
    F1 = np.fft.rfft(p1)[1]
    s = np.angle(F1) / (2 * np.pi / D)   # shift such that phase of mode 1 becomes 0
    p1, c0, mS, aS = (shift(f, -s, D) for f in (p1, c0, mS, aS))
    H = N // 2
    return np.concatenate([p1[:H], c0[:H], mS[:H], aS[:H], [D]])

def refine(z, N_old, N_new):
    H = N_old // 2; D = z[-1]
    out = []
    for i, par in enumerate(("odd", "even", "odd", "even")):
        h = z[i * H:(i + 1) * H]
        f = np.concatenate([h, -h]) if par == "odd" else np.concatenate([h, h])
        F = np.fft.rfft(f); G = np.zeros(N_new // 2 + 1, complex); G[:len(F) - 1] = F[:-1] * (N_new / N_old)
        out.append(np.fft.irfft(G, n=N_new)[:N_new // 2])
    return np.concatenate(out + [[D]])

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    S = DSS(n=2.0, N=N)
    z = make_guess(N)
    t0 = time.time()
    F = S.residual(z[None, :])[0]
    print(f"guess residual |F| = {np.linalg.norm(F):.3e} ({time.time()-t0:.2f} s per evaluation); finite: {np.all(np.isfinite(F))}")
    z, nF, ok = S.newton(z, tol=1e-10, maxit=40)
    print("converged:", ok, "Delta =", z[-1], f"({time.time()-t0:.0f} s)")
    np.save(f"/home/claude/dss/sol4d_N{N}.npy", z)

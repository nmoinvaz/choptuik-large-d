import sys, time, warnings
import numpy as np
from scipy.optimize import least_squares
warnings.filterwarnings("ignore")
from spec import DSSSpec
NODES = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def make(kc, n=2.0, **kw):
    opts = dict(Ks=8, u0=0.03, seg_h=0.002, log_steps=150)
    opts.update(kw)
    return DSSSpec(kc=kc, nodes=NODES, n=n, **opts)

def solve(S, z, maxfev=40, verbose=0):
    t0 = time.time()
    r = least_squares(lambda z: S.residual(z[None, :])[0], z, jac=lambda z: S.jacobian(z)[1], method="lm",
                      x_scale="jac", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=maxfev, verbose=verbose)
    return r.x, np.linalg.norm(r.fun), time.time() - t0

def tail(S, z):
    """Spectral tail: largest coefficient magnitude in the top quarter of harmonics, over all blocks."""
    parts, _ = S.split(z[None, :]); worst = 0.0
    for c, par in zip(parts, S.parities()):
        f = S.phys(c, par)[0]; F = np.abs(np.fft.rfft(f)) / S.N
        k = np.arange(len(F)); top = (k > 0.75 * S.kc) & (k <= S.kc)
        worst = max(worst, F[top].max())
    return worst

if __name__ == "__main__":
    kcs = [int(a) for a in sys.argv[1:]]
    z32 = np.load("sol_ms_N32.npy")
    prev = None
    for kc in kcs:
        S = make(kc)
        z = S.from_halfsamples(z32, 32, len(NODES)) if prev is None else S.regrid(*prev)
        F0 = np.linalg.norm(S.residual(z[None, :])[0])
        z, nF, w = solve(S, z)
        print(f"kc={kc:3d} (N={S.N}, unknowns {len(z)}): start |F| {F0:.2e} -> {nF:.2e}   Delta = {z[-1]:.10f}   tail {tail(S, z):.1e}   ({w:.0f} s)", flush=True)
        np.save(f"spec_kc{kc}.npy", z)
        prev = (z, S)

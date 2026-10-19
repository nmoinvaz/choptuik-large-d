import numpy as np, warnings
warnings.filterwarnings("ignore")
from dense import dense
from run_spec import make, tail
for D, f in [(4.0, "spec_kc31.npy"), (5.5, "solD_5.500_kc31.npy"), (6.0, "solD_6.000_kc31.npy"), (6.5, "solD_6.500_kc31.npy")]:
    X, Fd, Dl, N = dense(D, 31, f)
    c = Fd[:, 3]; gap = c - X[:, None]
    i, j = np.unravel_index(np.argmin(gap[X < 0.97]), gap[X < 0.97].shape)
    S = make(31, n=D - 2); z = np.load(f); parts, _ = S.split(z[None, :])
    names = ["p1", "c0"] + [f"{q}@{x}" for x in S.nodes for q in "pmac"] + ["mS", "aS"]
    worst = []
    for c_, par, nm in zip(parts, S.parities(), names):
        g = S.phys(c_, par)[0]; Fk = np.abs(np.fft.rfft(g)) / S.N; k = np.arange(len(Fk))
        worst.append((Fk[(k > 0.75 * S.kc) & (k <= S.kc)].max() / max(Fk.max(), 1e-30), nm))
    worst.sort(reverse=True)
    print(f"D={D}: min(c-x) inside = {gap[X < 0.97].min():.4f} at x={X[X < 0.97][i]:.3f};  c0 range [{c[0].min():.3f},{c[0].max():.3f}];  worst relative tails: {[(nm, f'{w:.1e}') for w, nm in worst[:4]]}")

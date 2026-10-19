import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
from run_spec import make
for D, f in [(4.0, "spec_kc31.npy"), (5.0, "solD_5.000_kc31.npy"), (5.5, "solD_5.500_kc31.npy"), (6.0, "solD_6.000_kc31.npy"), (6.5, "solD_6.500_kc31.npy")]:
    S = make(31, n=D - 2, Ks=12); z = np.load(f)
    parts, Dl = S.split(z[None, :]); F = [S.phys(c, p) for c, p in zip(parts, S.parities())]
    om = 2 * np.pi / Dl
    P, M, A, C = S.ssh_series(F[-2], F[-1], om)
    c1 = C[1][0]; a0 = A[0][0]
    pk = [np.abs(x).max() for x in P]
    ratio = [pk[k + 1] / pk[k] for k in range(len(pk) - 1)]
    print(f"D={D}: 1+c1 in [{(1+c1).min():.3f},{(1+c1).max():.3f}]  a_SSH in [{a0.min():.3f},{a0.max():.3f}]  |p_k| ratios {np.round(ratio[-6:],2)}  -> radius ~{1/np.mean(ratio[-4:]):.3f}")
    # kappa_K means (resonance check)
    print("      mean kappa_K, K=0..5:", [round(float(np.mean(K * (1 + c1) + c1 + 1 - 0.5 * S.n)), 3) for K in range(6)])

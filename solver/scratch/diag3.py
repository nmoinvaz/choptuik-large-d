import numpy as np, warnings
warnings.filterwarnings("ignore")
from dss import DSSMulti
from guess_ms import build
nodes = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for N in (32, 64):
    S = DSSMulti(nodes=nodes, n=2.0, N=N, Ks=8, u0=0.03, seg_h=0.002, log_steps=150)
    z = build(N, nodes)
    F = S.residual(z[None, :])[0]
    H = N // 2
    names = ["centre->0.02"] + [f"{nodes[i]}->{nodes[i+1]}" for i in range(len(nodes)-1)] + ["SSH->0.9"]
    print(f"N={N}: total {np.linalg.norm(F):.3e}")
    for b, nm in enumerate(names):
        blk = F[b*4*H:(b+1)*4*H].reshape(4, H)
        print(f"   {nm:14s} p {np.abs(blk[0]).max():.2e}  m {np.abs(blk[1]).max():.2e}  a {np.abs(blk[2]).max():.2e}  c {np.abs(blk[3]).max():.2e}")
    p1 = np.concatenate([z[:H], -z[:H]])
    print("   |Fourier p1| (odd modes):", np.abs(np.fft.rfft(p1))[1::2][:12].round(3))

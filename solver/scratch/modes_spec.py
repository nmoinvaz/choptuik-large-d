import sys, time, numpy as np, warnings
warnings.filterwarnings("ignore")
from run_spec import make
kc = int(sys.argv[1]); lo, hi, npts = float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
S = make(kc); z = np.load(f"/home/claude/dss/spec_kc{kc}.npy")
t0 = time.time(); prev = None
for lam in np.linspace(lo, hi, npts):
    M = S.mode_matrix(z, lam)
    sg, ld = np.linalg.slogdet(M)
    sv = np.linalg.svd(M, compute_uv=False)
    flag = "  <-- sign change" if prev is not None and sg != prev else ""
    print(f"lambda={lam:7.4f} sign={int(sg):+d} log|det|={ld:10.3f} smin={sv[-1]:.3e}{flag}", flush=True)
    prev = sg
print(f"({time.time()-t0:.0f} s)")

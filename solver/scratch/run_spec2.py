import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
from run_spec import make, solve, tail, NODES
prev_kc = int(sys.argv[1]); kcs = [int(a) for a in sys.argv[2:]]
prev = (np.load(f"/home/claude/dss/spec_kc{prev_kc}.npy"), make(prev_kc))
for kc in kcs:
    S = make(kc); z = S.regrid(*prev)
    F0 = np.linalg.norm(S.residual(z[None, :])[0])
    z, nF, w = solve(S, z, maxfev=20)
    print(f"kc={kc:3d} (N={S.N}, unknowns {len(z)}): start |F| {F0:.2e} -> {nF:.2e}   Delta = {z[-1]:.12f}   tail {tail(S, z):.1e}   ({w:.0f} s)", flush=True)
    np.save(f"/home/claude/dss/spec_kc{kc}.npy", z); prev = (z, S)

"""Is the D>26 drift a horizon-series truncation artefact? Re-solve one point with more series terms (Ks) and compare Delta."""
import sys, numpy as np, warnings, json, time
warnings.filterwarnings("ignore")
import fast; fast.install()
from rust_solver import make, jacobian_rust
D = float(sys.argv[1]); z = np.load(sys.argv[2]); Ks_list = [int(k) for k in sys.argv[3].split(",")]
for Ks in Ks_list:
    t = time.time()
    S, R = make(D, z[-1], Ks=Ks)
    F0 = np.linalg.norm(R(z))
    zz, nF, _ = fast.lm_svd3(S, z.copy(), maxjac=12, tol=1e-12, jac=lambda S_, z_, e: jacobian_rust(S_, R, z_, e), res=R)
    P, M, A, C = S.ssh_series(S.phys(S.split(zz[None, :])[0][-2], "odd"), S.phys(S.split(zz[None, :])[0][-1], "even"), np.array([2 * np.pi / zz[-1]]))
    last = max(np.abs(Q[-1]).max() for Q in (P, M, A, C)) * S.u0 ** Ks
    print(f"D={D:.4f} Ks={Ks}: start |F|={F0:.1e} -> {nF:.1e}  Delta={zz[-1]:.10f}  (input {z[-1]:.10f}, diff {zz[-1]-z[-1]:+.2e})  last term {last:.1e}  {time.time()-t:.0f}s", flush=True)
    np.save(f"ks_D{D:.4f}_Ks{Ks}.npy", zz)

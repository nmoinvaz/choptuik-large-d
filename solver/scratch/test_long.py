import sys, time, numpy as np, warnings
warnings.filterwarnings("ignore")
from run_spec import make
from spec_long import make_long, from_standard
from spec_c import lm_solve, jacobian_c, freeze_steps
D = float(sys.argv[1]); f = sys.argv[2]; kcf = int(sys.argv[3]); kc = 23
S_std = make(kcf, n=D - 2); z = np.load(f)
if kcf != kc:
    S23 = make(kc, n=D - 2); z = S23.regrid(z, S_std); S_std = S23
for Ks, xK in ((60, 0.6),):
    nodes = [x for x in [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6, 0.7] if x <= xK + 1e-12]
    S = freeze_steps(make_long(kc, D - 2, nodes=nodes, Ks=Ks), z[-1])
    zl = from_standard(S_std, z, S)
    t0 = time.time()
    F0 = np.linalg.norm(S.residual(zl[None, :])[0])
    zl, nF, _ = lm_solve(S, zl, maxjac=10)
    F, J = jacobian_c(S, zl, 3e-7)
    y = np.linalg.lstsq(J[:, :-1], J[:, -1], rcond=1e-13)[0]; sD = np.linalg.norm(J[:, -1] - J[:, :-1] @ y)
    print(f"D={D} x_K={xK} Ks={Ks}: start |F|={F0:.1e} -> {nF:.1e}; Delta={zl[-1]:.9f} (standard {z[-1]:.9f}); s_Delta={sD:.2e}; dDelta~{nF/sD:.1e}  ({time.time()-t0:.0f}s)", flush=True)
    np.save(f"long_D{D:.3f}_xK{xK}.npy", zl)

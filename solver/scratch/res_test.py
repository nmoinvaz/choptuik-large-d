"""Resolution tests at one D: (a) RK4 steps doubled everywhere, (b) tau resolution kc=31. Rust kernel, chord LM."""
import sys, numpy as np, warnings, time
warnings.filterwarnings("ignore")
import fast; fast.install()
import rustkernel
from rustkernel import RustResidual, kernel_args, Kernel
from rust_solver import jacobian_rust, NODES_L
from spec_long import make_long
from spec_c import freeze_steps
D = float(sys.argv[1]); z = np.load(sys.argv[2]); which = sys.argv[3]
class RR(RustResidual):
    def __init__(self, S, args):
        self.folded, self.threads, self.dedup, self.backend = True, 2, False, "simd"
        self.kernel = Kernel(*args); self.L = S.layout()[2]
def solve(S, R, z):
    return fast.lm_svd3(S, z.copy(), maxjac=14, tol=1e-12, jac=lambda S_, z_, e: jacobian_rust(S_, R, z_, e), res=R)
t = time.time()
if which == "steps":
    for mult in (1, 2, 4):
        S = freeze_steps(make_long(23, D - 2, nodes=NODES_L, Ks=60), z[-1])
        a = list(kernel_args(S)); a[11] = a[11] * mult; a[12] = [s * mult for s in a[12]]   # log_steps, seg_steps
        R = RR(S, a)
        zz, nF, _ = solve(S, R, z)
        print(f"D={D:.4f} RK4 steps x{mult}: |F|={nF:.1e} Delta={zz[-1]:.10f} ({time.time()-t:.0f}s)", flush=True)
elif which == "kc":
    S23 = make_long(23, D - 2, nodes=NODES_L, Ks=60)
    for kc in (23, 31):
        S = freeze_steps(make_long(kc, D - 2, nodes=NODES_L, Ks=60), z[-1])
        z0 = S.regrid(z, S23) if kc != 23 else z.copy()
        R = RustResidual(S, threads=2)
        print(f"  kc={kc} start |F|={np.linalg.norm(R(z0)):.1e}", flush=True)
        zz, nF, _ = solve(S, R, z0)
        print(f"D={D:.4f} kc={kc}: |F|={nF:.1e} Delta={zz[-1]:.10f} ({time.time()-t:.0f}s)", flush=True)
        np.save(f"kc{kc}_D{D:.4f}.npy", zz)

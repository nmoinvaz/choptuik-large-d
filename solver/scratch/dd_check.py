import numpy as np, warnings, time, sys
warnings.filterwarnings("ignore")
from spec_long import make_long
from spec_c import freeze_steps
from rustkernel import RustResidual
from rust_solver import NODES_L
D = float(sys.argv[1]); z = np.load(sys.argv[2]); x7 = len(sys.argv) > 3
S = freeze_steps(make_long(23, D - 2, nodes=NODES_L + ([0.7] if x7 else []), Ks=60), z[-1]); R = RustResidual(S, threads=2)
t = time.time(); F = R(z); t64 = time.time() - t
t = time.time(); Fh, Fl = R.residual_dd(z); tdd = time.time() - t
print(f"D={D:.3f}: f64 {t64*1e3:.1f} ms, dd {tdd*1e3:.1f} ms; |F64 - Fdd| = {np.linalg.norm(F - (Fh + Fl)):.1e}  |Fdd| = {np.linalg.norm(Fh):.2e}")
rng = np.random.default_rng(0); V = rng.standard_normal((4, len(z))); V[:, -1] = 0; V /= np.linalg.norm(V, axis=1)[:, None]
for d in (1e-10, 1e-14, 1e-18, 1e-22):
    # perturbation represented exactly as a lo part (z_hi unchanged) so it is not lost to f64 rounding
    Zh = np.repeat(z[None, :], 4, 0)
    Fp = R.residual_dd(Zh, d * V); Fm = R.residual_dd(Zh, -d * V)
    s2 = [np.linalg.norm((Fp[0][i] - Fh) + (Fp[1][i] - Fl) + (Fm[0][i] - Fh) + (Fm[1][i] - Fl)) for i in range(4)]
    print(f"  dd second difference d={d:.0e}: median {np.median(s2):.1e}")

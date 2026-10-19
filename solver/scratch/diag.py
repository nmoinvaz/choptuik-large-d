import numpy as np, warnings
warnings.filterwarnings("ignore")
from dss import DSS
from solve4d import make_guess
N = 32; S = DSS(n=2.0, N=N); z = make_guess(N)
p1, c0, mS, aS, om = S.unpack(z[None, :])
print("c0 range", c0.min(), c0.max(), " aS range", aS.min(), aS.max(), " p1 range", p1.min(), p1.max(), " mS", mS.min(), mS.max())
# centre side
P, M, A, C = S.centre_series(p1, c0, om)
print("centre series coefficient sizes |p_k|:", [f"{np.abs(x).max():.1e}" for x in P])
x = S.x0; s = np.stack([S.eval_series(F, x) for F in (P, M, A, C)]); h = 0.002
while x < 0.999:
    k1 = S.rhs(x, s, om); k2 = S.rhs(x + h/2, s + h/2*k1, om); k3 = S.rhs(x + h/2, s + h/2*k2, om); k4 = S.rhs(x + h, s + h*k3, om)
    s = s + h/6*(k1+2*k2+2*k3+k4); x += h
    if abs(round(x / 0.05) * 0.05 - x) < h / 2 or not np.all(np.isfinite(s)):
        print(f"  centre side x={x:.3f}: min(c-x)={np.min(s[3]-x):+.3f}  max|p|={np.abs(s[0]).max():.2e}  a in [{s[2].min():.3f},{s[2].max():.3f}]")
    if not np.all(np.isfinite(s)): break
# SSH side
P, M, A, C = S.ssh_series(mS, aS, om)
print("SSH series coefficient sizes |p_k|:", [f"{np.abs(x).max():.1e}" for x in P])
u = S.u0; s = np.stack([S.eval_series(F, u) for F in (P, M, A, C)]); x = 1 - u; h = -0.002
while x > 0.01:
    k1 = S.rhs(x, s, om); k2 = S.rhs(x + h/2, s + h/2*k1, om); k3 = S.rhs(x + h/2, s + h/2*k2, om); k4 = S.rhs(x + h, s + h*k3, om)
    s = s + h/6*(k1+2*k2+2*k3+k4); x += h
    if abs(round(x / 0.05) * 0.05 - x) < abs(h) / 2 or not np.all(np.isfinite(s)):
        print(f"  SSH side x={x:.3f}: min(c-x)={np.min(s[3]-x):+.3f}  max|p|={np.abs(s[0]).max():.2e}  c in [{s[3].min():.3f},{s[3].max():.3f}]")
    if not np.all(np.isfinite(s)): break

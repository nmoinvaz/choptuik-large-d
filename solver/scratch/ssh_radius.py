"""How far does the horizon (SSH) Taylor series in u = 1 - x converge?  If its radius is ~1, the solution near the
horizon can be represented by the series out to u ~ 0.3-0.5, replacing the stiff integration that sets the
precision floor."""
import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
from run_spec import make
D = float(sys.argv[1]); f = sys.argv[2]; kc = int(sys.argv[3])
S = make(kc, n=D - 2); z = np.load(f)
parts, Dl = S.split(z[None, :]); om = np.array([2 * np.pi / Dl[0]])
mS = S.phys(parts[-2], "odd"); aS = S.phys(parts[-1], "even")
for Ks in (40,):
    S.Ks = Ks
    P, M, A, C = S.ssh_series(mS, aS, om)
    for name, Q in (("P", P), ("M", M), ("A", A), ("C", C)):
        mags = np.array([np.abs(q).max() for q in Q])
        r = [mags[k] ** (-1 / k) if mags[k] > 0 else np.inf for k in range(5, Ks + 1, 5)]
        print(f"{name}: |coef_K| at K=0,5,..,{Ks}: " + " ".join(f"{mags[k]:.1e}" for k in range(0, Ks + 1, 5)) + "   radius est |c_K|^(-1/K): " + " ".join(f"{v:.2f}" for v in r))
# compare series evaluated at u vs integrated state from the last node: consistency check at u = 0.1
S.Ks = 40
P, M, A, C = S.ssh_series(mS, aS, om)
for u in (0.03, 0.1, 0.2, 0.3):
    st_series = np.stack([S.eval_series(Q, u) for Q in (P, M, A, C)])
    S8 = make(kc, n=D - 2)       # default Ks=8 + integration from u0 = 0.03
    P8, M8, A8, C8 = S8.ssh_series(mS, aS, om)
    s = S8.trunc(np.stack([S8.eval_series(Q, S8.u0) for Q in (P8, M8, A8, C8)]))
    st_int = S8.seg(s, 1 - S8.u0, 1 - u, om) if u > S8.u0 else s
    print(f"u={u}: |series(Ks=40) - (Ks=8 series + integration)| = {np.abs(S.trunc(st_series) - st_int).max():.2e}")

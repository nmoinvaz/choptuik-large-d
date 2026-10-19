"""Pointwise horizon coefficient g(tau) = 1 - (n-1)(a_SSH^2 - 1) (its mean is the gauge mode lambda_g).
If min g -> 0 the self-similarity horizon becomes degenerate at some tau and the SSH expansion is singular."""
import numpy as np, warnings, glob, re
warnings.filterwarnings("ignore")
from run_spec import make
files = sorted(glob.glob("solD_*_kc23.npy") + glob.glob("solD_*_kc31.npy"), key=lambda f: float(re.findall(r"solD_([\d.]+)_", f)[0]))
for f in files + ["diag_D9.350_mu1e-08.npy"]:
    D = float(re.findall(r"D_?([\d.]+?)_", f.replace("solD", "D").replace("diag_D", "D"))[0]) if "diag" not in f else 9.35
    kc = 31 if "kc31" in f else 23
    S = make(kc, n=D - 2); z = np.load(f)
    parts, Dl = S.split(z[None, :]); a = S.phys(parts[-1], "even")[0]; n = D - 2
    g = 1 - (n - 1) * (a ** 2 - 1)
    s0p = n / (2 * g.min()) - 1
    print(f"D={D:7.4f} kc={kc}  mean g={g.mean():.4f}  min g={g.min():.4f}  max g={g.max():.4f}  s0={n/(2*g.mean())-1:.3f}")

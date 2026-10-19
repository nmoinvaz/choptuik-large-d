"""Exponent of the non-smooth solutions at the horizon: p ~ (1-x)^s0 with s0 = -<kappa_0>/<1+c1>,
kappa_0 = c1 + 1 - n/2, c1 = -(n-1)(a_SSH^2 - 1). Regularity at the horizon is what selects Delta; the larger s0,
the weaker that condition is."""
import glob, re, numpy as np, warnings
warnings.filterwarnings("ignore")
from run_spec import make
files = {4.0: "spec_kc31.npy"}
for f in glob.glob("solD_*_kc31.npy"):
    m = re.match(r"solD_([\d.]+)_kc31\.npy", f)
    if m: files[float(m.group(1))] = f
for D in sorted(files):
    if D < 4 or D not in (4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0) and D < 8.1: continue
    n = D - 2; S = make(31, n=n); z = np.load(files[D]); parts, _ = S.split(z[None, :])
    aS = S.phys(parts[-1], "even")[0]
    c1 = -(n - 1) * (aS ** 2 - 1)
    s0 = -np.mean(c1 + 1 - n / 2) / np.mean(1 + c1)
    print(f"D={D:5.2f}: <1+c1> = {np.mean(1+c1):.4f}   s0 = {s0:.3f}   s0/n = {s0/n:.3f}")

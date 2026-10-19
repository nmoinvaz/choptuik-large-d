import glob, re, json, numpy as np, warnings
warnings.filterwarnings("ignore")
from run_spec import make
rows = []
files = {}
for f in glob.glob("solD_*_kc31.npy"):
    m = re.match(r"solD_([\d.]+)_kc31\.npy", f)
    if m: files[float(m.group(1))] = f
files[4.0] = "spec_kc31.npy"
for D in sorted(files):
    if D < 4: continue
    S = make(31, n=D - 2); z = np.load(files[D]); parts, Dl = S.split(z[None, :])
    c0 = S.phys(parts[1], "even")[0]; p1 = S.phys(parts[0], "odd")[0]
    N = S.N; Delta = float(Dl[0])
    # fraction of the period with c0 below 0.5 * its max ("collapsed" plateau)
    frac = np.mean(c0 < 0.5 * c0.max())
    rows.append((D, D - 2, Delta, c0.min(), c0.max(), frac))
    print(f"D={D:5.2f} n={D-2:5.2f}  Delta={Delta:.5f}  c0 min={c0.min():.4f} max={c0.max():.4f}  plateau fraction={frac:.3f}  n*c0min={(D-2)*c0.min():.3f}  sqrt(n)*c0min={np.sqrt(D-2)*c0.min():.3f}")
json.dump(rows, open("c0_stats.json", "w"))

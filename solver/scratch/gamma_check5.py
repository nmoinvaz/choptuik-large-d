import sys, time, numpy as np, warnings
warnings.filterwarnings("ignore")
from spec_long import make_long
from spec_c import freeze_steps
from gamma import growth_rate
D = float(sys.argv[1]); f = sys.argv[2]; xk = float(sys.argv[3]); KS = int(sys.argv[4]); pred = float(sys.argv[5])
nodes = [x for x in [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6] if x <= xk + 1e-9]
S = freeze_steps(make_long(23, D - 2, nodes=nodes, Ks=KS), 2.2); z = np.load(f); t0 = time.time()
for w in (0.002, 0.01, 0.03):
    try:
        lam, nev = growth_rate(S, z, pred - w, pred + w, xtol=1e-12); break
    except Exception as e:
        lam = None
print(f"D={D} x_K={xk} Ks={KS}: lambda0={lam}  gamma={1/lam if lam else None}  ({time.time()-t0:.0f}s)", flush=True)

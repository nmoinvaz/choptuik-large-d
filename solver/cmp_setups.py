"""Delta from two setups (x_K = 0.6 vs 0.7, or kc 23 vs 31) at the same D: interpolate the first (local cubic
through its 4 nearest points) to the D values of the second. The difference is the discretisation error bar."""
import numpy as np, sys
from data_ne import load
a = load(); b = load(xK=0.7) if len(sys.argv) < 2 else load(kc=int(sys.argv[1]))
Da = np.array(sorted(k for k in a if k >= 17)); ya = np.array([a[k]["Delta"] for k in Da])
ga = np.array([a[k]["gamma"] or np.nan for k in Da])
for k in sorted(b):
    if k < Da[0] or k > Da[-1]:
        continue
    i = np.argsort(abs(Da - k))[:4]
    p = np.polyfit(Da[i] - k, ya[i], 3); pg = np.polyfit(Da[i] - k, ga[i], 3)
    gb = b[k]["gamma"]
    print(f"D={k:8.4f}  Delta {b[k]['Delta']:.10f}  diff {b[k]['Delta'] - p[-1]:+.2e}   gamma diff "
          + ("-" if gb is None else f"{gb - pg[-1]:+.2e}"))

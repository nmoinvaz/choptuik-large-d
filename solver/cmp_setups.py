"""Delta from two setups at the same D (usage: cmp_setups.py [A] B, setups as xK,kc[,sj]; default 0.6,23 vs 0.7,23): interpolate the first (local cubic
through its 4 nearest points) to the D values of the second. The difference is the discretisation error bar."""
import numpy as np, sys
from data_ne import load
def setup(arg):   # "xK,kc[,sj]" e.g. 0.7,23 or 0.6,31,sj
    p = arg.split(","); return load(xK=float(p[0]), kc=int(p[1]), smooth=len(p) > 2 and p[2] == "sj")
a = setup(sys.argv[1]) if len(sys.argv) > 2 else load()
b = setup(sys.argv[-1]) if len(sys.argv) > 1 else load(xK=0.7)
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

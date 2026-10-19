"""Compute growth rates (gamma) for saved long-series solutions, in parallel with the scan."""
import json, time, os, numpy as np, warnings
warnings.filterwarnings("ignore")
from spec_long import make_long
from spec_c import freeze_steps
from gamma import growth_rate
NODES_L = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6]
done = {}
for fn in ("scan_long_kc23.jsonl", "gamma_long.jsonl"):
    if os.path.exists(fn):
        for l in open(fn):
            r = json.loads(l)
            if r.get("lam"): done[round(r["D"], 4)] = r["lam"]
idle = 0
while idle < 240:
    todo = []
    for l in open("scan_long_kc23.jsonl"):
        r = json.loads(l); d = round(r["D"], 4)
        if d not in done and os.path.exists(f"long_D{r['D']:.4f}.npy"): todo.append(r["D"])
    if not todo:
        idle += 1; time.sleep(30); continue
    idle = 0
    D = sorted(todo)[0]
    ks = sorted(done); 
    near = [k for k in ks if k < D][-3:]
    if len(near) >= 2:
        c = np.polyfit(near, [done[k] for k in near], len(near) - 1); pred = float(np.polyval(c, D))
    else:
        pred = done[near[-1]]
    S = freeze_steps(make_long(23, D - 2, nodes=NODES_L, Ks=60), 2.4)
    z = np.load(f"long_D{D:.4f}.npy"); lam = None; t0 = time.time()
    for w in (0.01, 0.03, 0.08):
        try:
            lam, _ = growth_rate(S, z, pred - w, pred + w, xtol=1e-11); break
        except Exception:
            continue
    rec = dict(D=D, lam=lam, gamma=(1 / lam if lam else None), seconds=round(time.time() - t0))
    print(json.dumps(rec), flush=True); open("gamma_long.jsonl", "a").write(json.dumps(rec) + "\n")
    done[round(D, 4)] = lam if lam else pred

import json, numpy as np, warnings, sys
warnings.filterwarnings("ignore")
import fast; fast.install()
from spec_long import make_long
from spec_c import freeze_steps, jacobian_cs
NODES_L = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6]
old = sorted({round(json.loads(l)['D'], 4): json.loads(l) for l in open('scan_long_kc23.jsonl')}.values(), key=lambda r: r['D'])
out = open('sdelta_exact.jsonl', 'a')
for r in old:
    D = r['D']
    if D > 16.0 or D < 7.9: continue
    z = np.load(f"long_D{D:.4f}.npy"); S = freeze_steps(make_long(23, D - 2, nodes=NODES_L, Ks=60), z[-1])
    F, J = jacobian_cs(S, z)
    y = np.linalg.lstsq(J[:, :-1], J[:, -1], rcond=1e-13)[0]; sD = float(np.linalg.norm(J[:, -1] - J[:, :-1] @ y))
    rec = dict(D=D, s_Delta=sD, s0=r['s0'], residual=float(np.linalg.norm(F))); print(json.dumps(rec), flush=True); out.write(json.dumps(rec) + "\n")

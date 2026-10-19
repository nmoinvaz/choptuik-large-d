"""Re-converge old long-series scan points with the exact horizon Jacobian + chord-corrected LM."""
import sys, json, time, numpy as np, warnings
warnings.filterwarnings("ignore")
from spec_long import make_long
from spec_c import freeze_steps, lm_svd2, jacobian_cs
NODES_L = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6]
Dmin, Dmax = float(sys.argv[1]), float(sys.argv[2]); out = sys.argv[3]
rs = [json.loads(l) for l in open('scan_long_kc23.jsonl')]
rs = sorted({round(r['D'], 6): r for r in rs}.values(), key=lambda r: r['D'])
for r in rs:
    D = r['D']
    if not (Dmin <= D <= Dmax): continue
    t0 = time.time(); z = np.load(f"long_D{D:.4f}.npy")
    S = freeze_steps(make_long(23, D - 2, nodes=NODES_L, Ks=60), z[-1])
    z2, nF, _ = lm_svd2(S, z, maxjac=6, tol=1e-12)
    F, J = jacobian_cs(S, z2)
    y = np.linalg.lstsq(J[:, :-1], J[:, -1], rcond=1e-13)[0]; sD = float(np.linalg.norm(J[:, -1] - J[:, :-1] @ y))
    rec = dict(D=D, Delta=float(z2[-1]), Delta_old=r['Delta'], change=float(z2[-1] - r['Delta']), residual=float(nF),
               residual_old=r['residual'], s_Delta=sD, seconds=round(time.time() - t0))
    print(json.dumps(rec), flush=True); open(out, 'a').write(json.dumps(rec) + "\n")
    np.save(f"longP_D{D:.4f}.npy", z2)

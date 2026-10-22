"""Singular values of the Jacobian of a sigma-model DSS solution, and where the weakest directions live (which
unknowns: centre data q, c0 by harmonic; node states; horizon data; Delta). Used to tell a fold (one isolated small
singular value, smooth vector, large Delta component) from the 'junk' near-null band of high centre harmonics.
usage: python sigdss_svd.py SOL D eps kc [--kappa 0.7] [--nlow 8]
Appends to data/sigdss_svd.jsonl."""
import os, sys, json, time, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sigdss_solver import make    # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("sol"); ap.add_argument("D", type=float); ap.add_argument("eps", type=float); ap.add_argument("kc", type=int)
ap.add_argument("--kappa", type=float, default=0.7); ap.add_argument("--nlow", type=int, default=8)
a = ap.parse_args()
z = np.load(a.sol)
S = make(a.D, a.eps, z, kc=a.kc, kappa=a.kappa)
t0 = time.time()
F, J = S.jacobian(z)
U, s, Vt = np.linalg.svd(J)
sizes, K, L = S.layout()
offs = np.concatenate([[0], np.cumsum(sizes)])
no = S.n_odd
low = []
for i in range(1, a.nlow + 1):
    v = Vt[-i]
    w_centre_q = float(np.linalg.norm(v[:no]))
    w_centre_c = float(np.linalg.norm(v[no:offs[2]]))
    w_ssh = float(np.linalg.norm(v[offs[-1] - sum(sizes[-3:]):offs[-1]]))
    w_D = float(abs(v[-1]))
    qh = np.abs(v[:no // 2]) + np.abs(v[no // 2:no])           # |Re| + |Im| per odd harmonic of q
    low.append(dict(s_rel=float(s[-i] / s[0]), w_q=w_centre_q, w_c0=w_centre_c, w_ssh=w_ssh, w_Delta=w_D,
                    q_harm=[round(float(x), 4) for x in qh]))
    print(f"s/smax {s[-i] / s[0]:.2e}: centre q {w_centre_q:.3f}, c0 {w_centre_c:.3f}, horizon {w_ssh:.3f}, "
          f"Delta {w_D:.2e}; q by harmonic {np.round(qh, 3).tolist()}", flush=True)
rec = dict(sol=os.path.basename(a.sol), D=a.D, eps=a.eps, kc=a.kc, F=float(np.linalg.norm(F)),
           s_rel_lowest=[float(x) for x in (s[::-1][:20] / s[0])], low=low, seconds=round(time.time() - t0))
print(json.dumps(dict(D=a.D, s_rel_lowest=rec["s_rel_lowest"][:10])), flush=True)
open(os.path.join(HERE, "data", "sigdss_svd.jsonl"), "a").write(json.dumps(rec) + "\n")

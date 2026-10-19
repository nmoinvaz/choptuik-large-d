"""Track the complex singularity pair of the horizon series across the long-series scan."""
import json, glob, re, numpy as np, warnings
warnings.filterwarnings("ignore")
from spec_long import make_long
recs = {round(json.loads(l)["D"], 4): json.loads(l) for l in open("scan_long_kc23.jsonl")}
out = []
for D in sorted(recs):
    f = f"long_D{D:.4f}.npy"
    S = make_long(23, D - 2, nodes=[0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6], Ks=100); z = np.load(f)
    parts, Dl = S.split(z[None, :]); om = np.array([2 * np.pi / Dl[0]])
    P, M, A, C = S.ssh_series(S.phys(parts[-2], "odd"), S.phys(parts[-1], "even"), om)
    us = []
    for Q, k in ((C, 0), (A, 0), (P, 1)):
        c = np.array([np.fft.rfft(q[0])[k] / S.N for q in Q]); lo = 40
        X = np.stack([c[lo:-1], c[lo - 1:-2]], 1); y = c[lo + 1:]
        (a1, a2), *_ = np.linalg.lstsq(X, y, rcond=None)
        u = 1 / np.roots([1, -a1, -a2]); u = u[np.argmax(np.abs(u.imag))]
        us.append(complex(u.real, abs(u.imag)))
    u = np.mean(us); spread = np.max(np.abs(np.array(us) - u))
    r = recs[D]; n = D - 2
    rec = dict(D=D, Delta=r["Delta"], gamma=r["gamma"], s0=r["s0"], lam_g=r["lam_gauge"], u_abs=abs(u), u_arg=float(np.angle(u)), spread=float(spread),
               x_c=[1 - u.real, -u.imag])
    out.append(rec)
    print(f"D={D:7.3f} Delta={r['Delta']:.6f} s0={r['s0']:.3f} | u_c = {u.real:.4f}+{u.imag:.4f}i |u_c|={abs(u):.4f} arg={np.degrees(np.angle(u)):.2f} deg (spread {spread:.1e}) | x_c = {1-u.real:.4f} -/+ {u.imag:.4f}i  |x_c|={abs(1-u):.4f} arg x_c={np.degrees(np.angle(1-u)):.1f}", flush=True)
json.dump(out, open("sing_track.json", "w"), indent=1)

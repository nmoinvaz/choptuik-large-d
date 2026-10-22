"""Number of growing modes of a sigma-model DSS solution by the argument principle.

Perturbations ~ exp(lam tau) x (periodic, with the background's half-period parity). In this sector lam is defined
modulo 2 i omega (omega = 2 pi/Delta); the opposite-parity sector appears shifted by i omega. A box of height 2 omega
therefore contains every mode of both sectors once. The box used is
    a <= Re lam <= b,   -omega/2 <= Im lam <= 3 omega/2,
whose edges avoid the lines Im lam = +-omega, +-3 omega on which the light-cone series is resonant: there
det M(lam) has poles at lam = n/2 - (K+1) lam_g + i omega (K = 0..Ks, lam_g = mean(1 - c_x(tau, 1)), the gauge rate),
see sigdss_solver.ssh_series. Those inside the box are removed by multiplying det M by (lam - pole).
The winding of det M around the boundary (adaptive: steps are halved until the phase changes by < pi/6) is then the
number of zeros inside, gauge mode included.
usage: python sigdss_count.py SOL D eps kc [--a 0.05] [--b 12] [--h0 0.5] [--kappa 0.7]
Appends to data/sigdss_count.jsonl."""
import os, sys, json, time, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sigdss_solver import SigDSS, NODES_STD    # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("sol"); ap.add_argument("D", type=float); ap.add_argument("eps", type=float); ap.add_argument("kc", type=int)
ap.add_argument("--a", type=float, default=0.05); ap.add_argument("--b", type=float, default=12.0)
ap.add_argument("--h0", type=float, default=0.5); ap.add_argument("--kappa", type=float, default=0.7)
ap.add_argument("--Ks", type=int, default=16); ap.add_argument("--u0", type=float, default=0.01)
ap.add_argument("--maxpts", type=int, default=600)
a = ap.parse_args()

z = np.load(a.sol)
S = SigDSS(kc=a.kc, nodes=NODES_STD, n=a.D - 2, eps=a.eps, Delta_ref=z[-1], Ks=a.Ks, u0=a.u0)
S.freeze_from(z, a.kappa)
om = 2 * np.pi / z[-1]
lg = S.gauge_lambda(z)
poles = [S.n / 2 - (K + 1) * lg + 1j * om for K in range(a.Ks + 1)]
poles = [p for p in poles if a.a < p.real < a.b]
print(f"D {a.D} eps {a.eps} Delta {z[-1]:.8f} omega {om:.5f} lam_gauge {lg:.6f}; poles inside: "
      f"{[round(p.real, 4) for p in poles]}", flush=True)
cache = {}


def f(lam):
    """log det M(lam) * prod(lam - pole): returns (phase, log|.|)."""
    if lam not in cache:
        s, l = np.linalg.slogdet(S.mode_matrix_c(z, lam))
        ph = np.angle(s); la = l
        for p in poles:
            ph += np.angle(lam - p); la += np.log(abs(lam - p))
        cache[lam] = (ph, la)
    return cache[lam]


corners = [complex(a.a, -om / 2), complex(a.b, -om / 2), complex(a.b, 1.5 * om), complex(a.a, 1.5 * om)]
t0 = time.time()
total = 0.0
edges = []
for e in range(4):
    A, B = corners[e], corners[(e + 1) % 4]
    L = abs(B - A)
    m = max(4, int(np.ceil(L / a.h0)))
    pts = [A + (B - A) * k / m for k in range(m + 1)]
    i = 0
    wind = 0.0
    while i < len(pts) - 1:
        p0, p1 = pts[i], pts[i + 1]
        d = np.angle(np.exp(1j * (f(p1)[0] - f(p0)[0])))
        if abs(d) > np.pi / 6 and abs(p1 - p0) > 1e-4 and len(cache) < a.maxpts:
            pts.insert(i + 1, 0.5 * (p0 + p1))
            continue
        wind += d
        i += 1
    total += wind
    edges.append(dict(A=[A.real, A.imag], B=[B.real, B.imag], wind=wind / (2 * np.pi), pts=len(pts)))
    print(f"  edge {e}: {A:.3f} -> {B:.3f}: winding {wind / (2 * np.pi):+.4f} ({len(pts)} points, "
          f"{time.time() - t0:.0f} s)", flush=True)
count = total / (2 * np.pi)
rec = dict(sol=os.path.basename(a.sol), D=a.D, eps=a.eps, kc=a.kc, Delta=float(z[-1]), omega=om, lam_gauge=lg,
           box=[a.a, a.b, -om / 2, 1.5 * om], poles=[[p.real, p.imag] for p in poles], winding=count,
           count=int(round(count)), n_eval=len(cache), edges=edges, seconds=round(time.time() - t0),
           capped=len(cache) >= a.maxpts)
print(json.dumps(rec), flush=True)
open(os.path.join(HERE, "data", "sigdss_count.jsonl"), "a").write(json.dumps(rec) + "\n")

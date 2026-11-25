"""Form S at one D in double-double: float64 damped Newton, then the split Newton of lsnewton.py in dd (#1088).

Steps: (1) the start (an hp npz or a json of this script or fsolve.py) rescaled to D; (2) fsolve.py's float64 damped
Newton with the complex-step Jacobian (cheap, puts the strong directions at float64 accuracy); (3) the dd difference
Jacobian and its scaled SVD (mpmath, 40 digits); (4) lsnewton.ls_solve in dd. Written: a json with the dd unknowns,
Delta, the residual history and the scaled singular values.

Usage: python3 solve_dd.py --D=100 --M=8 --start=STATE [--Delta=...] [--f64iters=12] [--outer=6] [--rounds=1]
                           [--tau=1e-8] [--tol=1e-26] [--ks=...] [--kb=...] [--out=FILE.json] [--mutate=sigma]
  --rounds: the dd Jacobian is formed again and the split Newton restarted up to this many times
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import ddnp as d  # noqa: E402
from ddnp import DD  # noqa: E402
from fseries_sdd import FactoredSDD, load_y  # noqa: E402
from fsolve import conditioning, solve  # noqa: E402
from lsnewton import ls_solve  # noqa: E402

opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
D = float(opts["D"])
M = int(opts.get("M", 8))
ks = int(opts["ks"]) if "ks" in opts else None
mut = opts.get("mutate", "")
kb = int(opts["kb"]) if "kb" in opts else None  # the banded horizon recurrence (fseries_sdd.py, jset)
log = lambda s: print(s, flush=True)  # noqa: E731
t0 = time.time()
y, D0 = load_y(opts["start"], M, D)
if "Delta" in opts:
    y[3 * M] = float(opts["Delta"])
log("solve_dd: D %.6f M %d start %s (D0 %.6f) Delta0 %.12f" % (D, M, opts["start"].rsplit("/", 1)[-1], D0, float(y.h[3 * M])))

# (2) float64 damped Newton (form S through fsolve's 4M+1 layout)
f64 = np.concatenate([y.h[:3 * M], np.zeros(M), y.h[3 * M:]])
with np.errstate(all="ignore"):
    f64, J64, fz, hist64, secs = solve(D, M, f64, ks=ks, iters=int(opts.get("f64iters", 12)), mutate=mut,
                                       log=lambda s: log("  f64 " + s), form="S", kb=kb)
sv64, svs64, grip64 = conditioning(J64, M, "S")
log("f64: max|F| %.3e Delta %.15f scaled cond %.3e (%.0f s)" % (hist64[-1]["maxF"], f64[4 * M], svs64[0] / svs64[-1], time.time() - t0))
y = DD(np.concatenate([f64[:3 * M], f64[4 * M:]]))

# (3)-(4) dd
f = FactoredSDD(D, M, ks=ks, mutate=mut, kb=kb)
hist = []
for rnd in range(int(opts.get("rounds", 1))):
    J = f.jacobian(y)
    log("dd Jacobian, round %d (%.0f s)" % (rnd, time.time() - t0))
    y, F, h1, S = ls_solve(f.residual, y, J, tau=float(opts.get("tau", 1e-8)), outer=int(opts.get("outer", 6)), log=log)
    hist += h1
    if float(np.max(np.abs(F.h))) < float(opts.get("tol", 1e-26)):
        break
nF = float(np.max(np.abs(F.h)))
# the Jacobian at the solution, for the conditioning
J = f.jacobian(y)
from lsnewton import scaled_svd  # noqa: E402
r, c, U, Ssol, Vt = scaled_svd(J)
# grip on Delta: |e_Delta^T J^-1| with unit rows
Jf = (J.h + J.l) / r[:, None]
e = np.zeros(3 * M + 1)
e[3 * M] = 1.0
grip = float(np.linalg.norm(np.linalg.lstsq(Jf.T, e, rcond=None)[0]))
log("dd: max|F| %.3e Delta %.16f (+%.2e) ; scaled sv max %.3e min %.3e cond %.3e ; smallest %s ; grip %.3e (%.0f s)" % (
    nF, float(y.h[3 * M]), float(y.l[3 * M]), Ssol[0], Ssol[-1], Ssol[0] / Ssol[-1],
    " ".join("%.2e" % v for v in Ssol[-6:]), grip, time.time() - t0))
if "out" in opts:
    js = {"D": D, "M": M, "kc": f.kc, "ks": f.ks, "kb": kb, "form": "S-dd", "p1": list(y.h[0:M]), "c0": list(y.h[M:2 * M]),
          "Q0": list(y.h[2 * M:3 * M]), "Delta": float(y.h[3 * M]), "p1_lo": list(y.l[0:M]), "c0_lo": list(y.l[M:2 * M]),
          "Q0_lo": list(y.l[2 * M:3 * M]), "Delta_lo": float(y.l[3 * M]), "maxF": nF, "ls_hist": hist,
          "sv_scaled": list(map(float, Ssol)), "grip": grip, "f64_maxF": hist64[-1]["maxF"], "f64_Delta": float(f64[4 * M]),
          "f64_sv_scaled": list(map(float, svs64)), "start": opts["start"], "seconds": time.time() - t0, "mutate": mut}
    json.dump(js, open(opts["out"], "w"), indent=1)

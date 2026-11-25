"""Check fseries.py's two series against a stored hp state (#1088).

The centre series in z, from hp's p1 and c0, is summed at hp's first node x = 0.3 and compared with the node values;
the horizon series in s = w/w_K, from hp's m_h, with beta matched to hp's node value of a at x = 0.7, is summed there
and compared with hp's last node (all harmonics of U and c; V's odd harmonics 1 and 3 only, since hp's pinned rows
drop the p-matching of harmonics 5 and up). hp's a_h is compared with the horizon side's output a_h.

Usage: python3 check_state.py STATE.npz [--mutate=sigma]   (the control replaces the SSH exponent's g term; FAIL)
Exit status 1 when a comparison exceeds its bound.
"""
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fseries import Factored, harmonics_odd  # noqa: E402

opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
path = [a for a in sys.argv[1:] if not a.startswith("--")][0]
z = np.load(path, allow_pickle=False)
meta = json.loads(str(z["meta"]))
M = meta["M"]
K = len(meta["nodes"])
zz = z["zh"] + z["zl"]
D = meta["D"]
n = D - 2
p1 = zz[0:M]
c0 = zz[M:2 * M]
off = lambda b: 2 * M + 4 * M * b  # noqa: E731
mh = zz[off(K):off(K) + M]
ah = zz[off(K) + M:off(K) + 2 * M]
Delta = zz[-1]
om = 2 * np.pi / Delta
ok = True


def node(b):
    o = off(b)
    p, m, a, c = (zz[o + i * M:o + (i + 1) * M] for i in range(4))
    x = meta["nodes"][b]
    return {"U": p + m, "V": (p - m) / x, "a": a, "c": c}


x0 = meta["nodes"][0]
fc = Factored(D, M, kc=int(opts.get("kc", 200)), zK=x0**2)
ce = fc.centre(p1, c0, om)
nd = node(0)
print("D %.6f n %.4f M %d Delta %.15f" % (D, n, M, Delta))
for nm in ("U", "V", "c"):
    err = np.max(np.abs(ce[nm] - nd[nm])) / np.max(np.abs(nd[nm]))
    print("centre at x %.2f: %s rel err %.2e" % (x0, nm, err))
    ok = ok and err < 1e-12
err = np.max(np.abs(ce["alpha"] - (nd["a"] - 1))) / np.max(np.abs(nd["a"] - 1))
print("centre at x %.2f: a - 1 rel err %.2e" % (x0, err))
ok = ok and err < 1e-10

xK = meta["nodes"][-1]
ks = int(opts.get("ks", n + 14 * np.sqrt(n) + 80))
fh = Factored(D, M, ks=ks, zK=xK**2, mutate=opts.get("mutate", ""))
ndK = node(K - 1)
aK = ndK["a"]
betaK = (n - 1) * (1 - 1 / aK**2)
ho = fh.horizon(2 * mh, om, betaK)
print("horizon (ks %d, %d sweeps, last beta change %.1e)" % (ks, fh.stats["sweeps"], fh.stats["beta_dif"]))
for nm in ("U", "c"):
    err = np.max(np.abs(ho[nm] - ndK[nm])) / np.max(np.abs(ndK[nm]))
    print("horizon at x %.2f: %s rel err %.2e" % (xK, nm, err))
hU = harmonics_odd(ho["V"])
hN = harmonics_odd(ndK["V"])
for j in range(min(4, len(hU))):
    print("   V harmonic %2d: horizon %.6e node %.6e rel diff %.2e" % (2 * j + 1, hU[j], hN[j], abs(hU[j] - hN[j]) / hN[j]))
errV = max(abs(hU[j] - hN[j]) / hN[j] for j in range(2))
err_ah = np.max(np.abs(ho["a_h"] - ah)) / np.max(np.abs(ah - 1))
print("a_h: horizon side output against hp's a_h, rel to a_h - 1: %.2e" % err_ah)
print("tails: centre %.1e horizon %.1e" % (ce["tail"], ho["tail"]))
ok = ok and errV < 1e-8 and err_ah < 1e-6
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

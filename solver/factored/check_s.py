"""Checks of form S (fseries_s.py) against a stored hp state (#1088).

  1. dmat_even differentiates cos(2 theta), sin(4 theta) and a smooth even field to rounding (spectral accuracy)
  2. the slaving: at each of hp's nodes, X = B - 1 from (omega D - (n-1) - q_S) X = q_S with the node's p, m, c
     equals the node's 1/a^2 - 1 (hp integrates a by the Hamiltonian equation; the momentum constraint is not imposed)
  3. form S's centre series at hp's first node and its horizon series (from hp's m_h, with nothing for a) at hp's
     last node agree with the node values (V's harmonics 1 and 3, since hp's pinned rows drop the p-matching of
     harmonics 5 and up; U and c in full)
  4. the Hamiltonian equation, which form S does not impose, holds order by order in its centre series (from hp's
     p1, c0 and Delta: any data, since H_tau = (n-1) a^2 H has no periodic solution but 0)

Usage: python3 check_s.py STATE.npz [--mutate=noslave|evenD]   (each control makes a check FAIL)
"""
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fseries import harmonics_odd  # noqa: E402
from fseries_s import FactoredS, dmat_even  # noqa: E402

opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
path = [a for a in sys.argv[1:] if not a.startswith("--")][0]
mut = opts.get("mutate", "")
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
Delta = zz[-1]
om = 2 * np.pi / Delta
ok_all = True


def report(k, name, ok):
    global ok_all
    ok_all = ok_all and ok
    print("%d. %-70s %s" % (k, name, "PASS" if ok else "FAIL"), flush=True)


# 1
th = np.pi * np.arange(M) / M
De = dmat_even(M) * (0.5 if mut == "evenD" else 1.0)
f = np.exp(0.3 * np.cos(2 * th))
fd = -0.6 * np.sin(2 * th) * f
e1 = max(np.max(np.abs(De @ np.cos(2 * th) + 2 * np.sin(2 * th))), np.max(np.abs(De @ np.sin(4 * th) - 4 * np.cos(4 * th))),
         np.max(np.abs(De @ f - fd)))
print("   dmat_even max error %.2e" % e1)
report(1, "dmat_even differentiates even fields", e1 < 1e-12)

# 2
fs = FactoredS(D, M, mutate=mut)
errs = []
for b in range(K):
    o = off(b)
    p, m, a, c = (zz[o + i * M:o + (i + 1) * M] for i in range(4))
    x = meta["nodes"][b]
    qs = ((x - c) * p * p + (x + c) * m * m) / x
    if mut == "noslave":
        qs = 0 * qs
    Xs = fs.xsolve(om, qs, qs)
    Xn = 1 / a**2 - 1
    errs.append(np.max(np.abs(Xs - Xn)) / np.max(np.abs(Xn)))
    print("   node x %.2f: |X_S - X_hp| / |X_hp| = %.2e  (|X| %.2e)" % (x, errs[-1], np.max(np.abs(Xn))))
report(2, "momentum-constraint slaving holds at hp's nodes", max(errs) < 1e-9)

# 3


def node(b):
    o = off(b)
    p, m, a, c = (zz[o + i * M:o + (i + 1) * M] for i in range(4))
    x = meta["nodes"][b]
    return {"U": p + m, "V": (p - m) / x, "c": c, "X": 1 / a**2 - 1}


x0 = meta["nodes"][0]
fc = FactoredS(D, M, kc=200, zK=x0**2, mutate=mut)
ce = fc.centre(p1, c0, om)
nd = node(0)
e3 = []
for nm in ("U", "V", "c"):
    e3.append(np.max(np.abs(ce[nm] - nd[nm])) / np.max(np.abs(nd[nm])))
    print("   centre at x %.2f: %s rel err %.2e" % (x0, nm, e3[-1]))
xK = meta["nodes"][-1]
fh = FactoredS(D, M, zK=xK**2, mutate=mut)
ho = fh.horizon(2 * mh, om)
ndK = node(K - 1)
for nm in ("U", "c"):
    e3.append(np.max(np.abs(ho[nm] - ndK[nm])) / np.max(np.abs(ndK[nm])))
    print("   horizon at x %.2f: %s rel err %.2e" % (xK, nm, e3[-1]))
hU = harmonics_odd(ho["V"])
hN = harmonics_odd(ndK["V"])
for j in range(min(4, len(hU))):
    print("   V harmonic %2d: horizon %.6e node %.6e rel diff %.2e" % (2 * j + 1, hU[j], hN[j], abs(hU[j] - hN[j]) / hN[j]))
e3.append(max(abs(hU[j] - hN[j]) / hN[j] for j in range(2)))
print("   horizon tail %.1e, ks %d" % (ho["tail"], fh.ks))
report(3, "form-S centre and horizon series reproduce hp's first and last nodes", max(e3) < 1e-7)

# 4. the Hamiltonian equation, not imposed in form S, holds order by order in the centre series: with B = 1 + X,
#    2z B_z = (n-1)(1 - B) - q_H B, q_H = (U^2 + z V^2)/2, i.e. at order k: (2k + n - 1) X_k + (q_H B)_k = 0
Uc, Vc, cc, Xc = ce["series"]
Kc = Uc.shape[0] - 1
qH = np.zeros_like(Uc)
for k in range(1, Kc + 1):
    qH[k] = 0.5 * (sum(Uc[j] * Uc[k - j] for j in range(1, k)) + sum(Vc[j] * Vc[k - 1 - j] for j in range(0, k)))
Bc = Xc.copy()
Bc[0] = Bc[0] + 1.0
worst = 0.0
for k in range(1, 40):
    res = (2 * k + n - 1) * Xc[k] + sum(qH[j] * Bc[k - j] for j in range(0, k + 1))
    worst = max(worst, np.max(np.abs(res)) / max(1e-300, np.max(np.abs((n - 1) * Xc[k]))))
print("   centre series, orders 1..39: max |Hamiltonian residual| / |(n-1) X_k| = %.2e" % worst)
report(4, "the Hamiltonian equation holds order by order in form S's centre series", worst < 1e-10)
print("ALL PASS" if ok_all else "SOME FAIL")
sys.exit(0 if ok_all else 1)

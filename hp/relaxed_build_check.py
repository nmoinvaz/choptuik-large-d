"""A build's check of the relaxed horizon sums (#1096, #1098): which hpkernel loads, and Solver.set_horizon_relaxed's effect on one
quad-double residual at a state with the Moebius horizon (relaxed against term by term within 1e-50, and the control
mutate 56, which must differ by more than 1e-40), then one Solver at M 64 with one horizon block through form S's
size dispatch (Ctx<8>). The u = 1 - x horizon (states without hmob, such as D 300's) has no relaxed sums.

    PYTHONPATH=<site> python3 relaxed_build_check.py STATE      (from hp/; exit 1 on FAIL)
"""
import sys
import time

import mpmath as mp
import numpy as np

import hpkernel
from form_s import make_solver_s
from octo_corrector import load_o

print("hpkernel", hpkernel.__file__, flush=True)
z8, m = load_o(sys.argv[1])
with mp.workdps(60):
    D = mp.mpf(m["D_str"])
if not m.get("hmob"):
    sys.exit("FAIL: this state has no Moebius horizon (hmob), so no relaxed sums to check")
s = make_solver_s(D, int(m["M"]), list(m["nodes"]), kfreeze=int(m.get("kfreeze", 3)), kt=int(m["kt"]), kc=int(m["kc"]),
                  ks=int(m["ks"]), threads=4, segs=m["segs"], mob=float(m["hmob"]))
if not hasattr(s, "set_horizon_relaxed"):
    sys.exit("FAIL: this hpkernel has no set_horizon_relaxed")
z4 = [np.ascontiguousarray(z8[q]) for q in range(4)]
mp.mp.dps = 80
val = lambda F: [mp.fsum(mp.mpf(float(np.asarray(F[q])[i])) for q in range(4)) for i in range(len(F[0]))]  # noqa: E731
res = {}
for tag, mode, mut in (("naive", 0, 0), ("relaxed", 1, 0), ("control", 1, 56)):
    s.set_horizon_relaxed(mode)
    t0 = time.time()
    res[tag] = (val(s.residual_s(z4, mut)), time.time() - t0)
s.set_horizon_relaxed(0)
n, r, c = res["naive"][0], res["relaxed"][0], res["control"][0]
d = float(max(abs(a - b) for a, b in zip(r, n)))
dc = float(max(abs(a - b) for a, b in zip(c, n)))
ok = d <= 1e-50 and dc > 1e-40
print(f"D {float(D):g} ks {m['ks']}: qd max|F| {float(max(abs(v) for v in n)):.2e}; |F_relaxed - F_naive| {d:.2e} "
      f"(<= 1e-50), control |F_mut56 - F_naive| {dc:.2e} (> 1e-40); {res['naive'][1]:.1f} s naive, "
      f"{res['relaxed'][1]:.1f} s relaxed: {'PASS' if ok else 'FAIL'}", flush=True)
# form S at M 64 (Ctx<8>): one Solver and one horizon block through the size dispatch, with m_h zero and this Delta
s64 = make_solver_s(D, 64, list(m["nodes"]), kfreeze=3, kt=int(m["kt"]), kc=int(m["kc"]), ks=200, threads=4,
                    segs=m["segs"], mob=float(m["hmob"]))
z = np.zeros(s64.size_s)
z[-1] = float(np.asarray(z8[0])[-1])
h = s64.horizon_s([np.ascontiguousarray(z), np.zeros(s64.size_s)])
ok64 = s64.size_s == 3 * 64 * (len(m["nodes"]) + 1) + 1 and len(h[0]) == 3 * 64 and bool(np.all(np.isfinite(h[0])))
print(f"M 64: size_s {s64.size_s}, horizon block {len(h[0])} values, finite {bool(np.all(np.isfinite(h[0])))}: "
      f"{'PASS' if ok64 else 'FAIL'}", flush=True)
sys.exit(0 if ok and ok64 else 1)

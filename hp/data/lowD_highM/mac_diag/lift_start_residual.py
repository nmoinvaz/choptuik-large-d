"""Residual of a low-D start state at its own M (saved plan if any, and a fresh plan), and after driver.resample_z to
M2 on its own plan: the start |F| of highm_check.py lift.

    cd hp && python3 data/lowD_highM/mac_diag/lift_start_residual.py STATE.npz M2
"""
import sys, time
import numpy as np
from hpkernel.driver import load, make_solver, resample_z, dd_norm
st, M2 = sys.argv[1], int(sys.argv[2])
zh, zl, m = load(st)
D, M, nodes = m["D"], m["M"], list(m["nodes"])
print("meta", {k: (v if k != "segs" else f"{len(v)} segs") for k, v in m.items() if k not in ("nodes",)}, "nodes", len(nodes))
kw = dict(nodes=nodes, kt=m["kt"], kc=m["kc"], ks=m["ks"], threads=2)
s = make_solver(D, M, **kw)
if m.get("segs"):
    s.set_segs(m["segs"]); t = time.time()
    print(f"M {M} saved plan: |F| {dd_norm(*s.residual(zh, zl)):.3e} ({time.time()-t:.1f}s)", flush=True)
t = time.time(); seg = s.plan(zh, zl, 1e-30, 0.05); s.set_segs(seg)
print(f"M {M} fresh plan ({time.time()-t:.1f}s): |F| {dd_norm(*s.residual(zh, zl)):.3e}", flush=True)
zh2, zl2 = resample_z(zh + zl, M, M2, len(nodes))
s2 = make_solver(D, M2, **kw)
t = time.time(); seg2 = s2.plan(zh2, zl2, 1e-30, 0.05); s2.set_segs(seg2)
Fh, Fl = s2.residual(zh2, zl2)
F = np.asarray(Fh) + np.asarray(Fl)
print(f"M {M2} resampled, its plan ({time.time()-t:.1f}s, {sum(len(g)-1 for g in seg2)} steps): |F| {dd_norm(Fh, Fl):.3e}, max |F_i| {np.max(np.abs(F)):.3e} at {int(np.argmax(np.abs(F)))} of {F.size}", flush=True)

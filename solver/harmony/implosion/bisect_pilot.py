"""Bisection of the D_W / D_Z classification of shoot_pilot.run (side -1) for r^(n)(gamma) in (r_n, r_{n+1}).
Usage: bisect_pilot.py gamma rlo rhi [NT] [smax]"""
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
sys.argv, args = sys.argv[:2], sys.argv[1:]
import shoot_pilot as sp
lo, hi = float(args[1]), float(args[2])
NT = int(args[3]) if len(args) > 3 else 40
smax = float(args[4]) if len(args) > 4 else 1e6
tlo = sp.run(lo, -1, NT=NT, smax=smax)[0]; thi = sp.run(hi, -1, NT=NT, smax=smax)[0]
print(f"gamma {sp.gam}: ends {tlo} at {lo}, {thi} at {hi}")
assert tlo != thi
t0 = time.time()
for it in range(60):
    mid = (lo + hi)/2
    t = sp.run(mid, -1, NT=NT, smax=smax)[0]
    if t == tlo: lo = mid
    elif t == thi: hi = mid
    else:
        print(f"  undecided ({t}) at {mid!r}"); break
    if hi - lo < 4e-16*hi: break
print(f"r = {(lo+hi)/2!r}  bracket width {hi-lo:.2e}  NT {NT} smax {smax:g}  [{time.time()-t0:.1f} s]")

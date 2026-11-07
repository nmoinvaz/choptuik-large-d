"""Outcome of the other half-branch (toward P_inf = (0,0)) at the grid points of intervals_grid.py, gamma = 7/5.
Usage: left_grid.py dps p NT npts"""
import sys, os, time
dps, p, NT, npts = sys.argv[1:5]
sys.argv = ["x", "7/5", "1", "1", dps, p, NT, "1e-10"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shoot_mp as s
import mpmath as mp
rn = {2: "1.034210797210342108181121", 3: "1.070374378675834393936063", 4: "1.094974963188878288595314",
      5: "1.111900086191899177148834", 6: "1.124064439344963857725882", 7: "1.133166815190844531464624",
      8: "1.140208448655939460936351"}
t0 = time.time()
for n in range(2, 8):
    a, b = mp.mpf(rn[n]), mp.mpf(rn[n + 1])
    outs = []
    for i in range(1, int(npts) + 1):
        r = a + (b - a)*mp.mpf(i)/(int(npts) + 1)
        outs.append(s.run(r, side=-1, maxsteps=4000)[0])
    print(f"(r_{n}, r_{n+1}): {' '.join(outs)}  [{time.time()-t0:.0f} s]", flush=True)

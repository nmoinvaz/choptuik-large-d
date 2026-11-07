"""Outcome (D_W or D_Z first) of the P_0-side smooth branch on a grid in each (r_n, r_{n+1}), gamma = 7/5.
r_n from kres_planted.log. Usage: intervals_grid.py dps p NT npts"""
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
        outs.append(s.run(r, maxsteps=6000)[0])
    flips = sum(1 for u, v in zip(outs, outs[1:]) if u != v)
    print(f"(r_{n}, r_{n+1}) = ({mp.nstr(a, 8)}, {mp.nstr(b, 8)}): {' '.join(o.replace('D_', '') for o in outs)}"
          f"   switches {flips}  [{time.time()-t0:.0f} s]", flush=True)

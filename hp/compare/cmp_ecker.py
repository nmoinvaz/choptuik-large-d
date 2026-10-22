"""Ecker et al. (CriticalSpacetimeCrystal repo, FortranData/dimX/{256,512}/Delta.out and DeltaList.dat) against our
Delta(D) (README rational formula, fitted on the f64 points 4 <= D <= 30.2; direct f64 points where available).

    git clone https://github.com/EckerChristian/CriticalSpacetimeCrystal ecc     # their published data (not included)
    python3 cmp_ecker.py ecc
"""
import os, glob, json, sys
import numpy as np
A = [1.6438975180, 4.0378045077e1, 2.5233285989e2, 2.0673502321e2, -3.5709850575e2, 3.1347320996e2,
     -4.1891374574e2, 3.2695252782e2, -1.6050467699e2]
B = [1, 1.8251453626e1, 6.8182050167e1]
E = [0.5, 1.8083629635e1, 1.5241588758e2, -5.2341588541e1, 9.9259594505, -9.8386450353e1, 7.4529506074e1, -4.8272358903e1]
Fq = [1, 3.7683686025e1, 3.4245505235e2]
poly = lambda c, x: sum(ci * x ** i for i, ci in enumerate(c))
ours = lambda D: poly(A, 1 / (D - 2)) / poly(B, 1 / (D - 2))
oursg = lambda D: poly(E, 1 / (D - 2)) / poly(Fq, 1 / (D - 2))
ECC = sys.argv[1] if len(sys.argv) > 1 else "ecc"   # a clone of github.com/EckerChristian/CriticalSpacetimeCrystal
root = os.path.join(ECC, "complete_data", "FortranData") + "/"
rows = []
for d in sorted([e for e in os.listdir(root) if e.startswith("dim")], key=lambda s: float(s[3:])):
    D = float(d[3:])
    v = {}
    for res in ("256", "512", "1024"):
        f = os.path.join(root, d, res, "Delta.out")
        if os.path.exists(f):
            try:
                v[res] = float(open(f).read().split()[0])
            except Exception:
                pass
    if D >= 4.0 and "512" in v:
        rows.append((D, v.get("256"), v["512"], ours(D)))
print("   D      Ecker 256        Ecker 512        ours (formula)    512-256     512-ours    (512-ours)/(256-512)")
for D, a, b, o in [r for r in rows if r[0] in (4.0,4.15,4.3,4.43,4.5,4.6,4.7,4.75,4.8,4.85,4.9,4.95,5.0,5.05,5.1,5.2,5.3,5.4,5.5) or abs(r[0]-round(r[0]*10)/10)<1e-9]:
    s = f"{D:7.4f}  {('-' if a is None else format(a, '.9f')):>14}  {b:.9f}  {o:.9f}  "
    s += (f"{b - a:+.2e}  " if a else " " * 11) + f"{b - o:+.2e}" + (f"  {(b - o) / (a - b):+.3f}" if a and a != b else "")
    print(s)
# our direct f64 points
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "solver", "data"))
sys.path.insert(0, "..")
from data_all import low_d
lo = low_d()
print("\nour direct f64 points vs Ecker DeltaList and gamma list:")
DL = {round(float(a), 4): float(b) for a, b in (l.split() for l in open(os.path.join(ECC, "complete_data", "DeltaList.dat")))}
GL = {round(float(a), 4): float(b) for a, b in (l.split() for l in open(os.path.join(ECC, "complete_data", "GammaList.dat")))}
for d in sorted(lo):
    if 3.7 <= d <= 7.9:
        e = DL.get(round(d, 4)); g = GL.get(round(d, 4))
        print(f"  D {d:6.4f}: ours {lo[d][0]:.9f} (sigma {lo[d][1]:.0e})  Ecker {e if e is None else f'{e:.9f}'}"
              + (f"  diff {e - lo[d][0]:+.2e}" if e else "")
              + f"   gamma ours {lo[d][2]:.8f}" + (f" Ecker {g:.8f} diff {g - lo[d][2]:+.2e}" if g else ""))

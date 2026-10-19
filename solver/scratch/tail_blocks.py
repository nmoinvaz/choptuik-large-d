import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
from spec_long import make_long
NODES_L = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6]
D = float(sys.argv[1]); z = np.load(sys.argv[2])
S = make_long(23, D - 2, nodes=NODES_L, Ks=60)
names = ["p1c", "c0c"] + [f"{f}@{x}" for x in NODES_L for f in "pmac"] + ["mS", "aS"]
parts, _ = S.split(z[None, :])
for nm, c, par in zip(names, parts, S.parities()):
    f = S.phys(c, par)[0]; F = np.abs(np.fft.rfft(f)) / S.N
    ks = S.kodd if par == "odd" else S.keven
    amp = F[ks]
    print(f"{nm:8s} " + " ".join(f"{a:.0e}" for a in amp))

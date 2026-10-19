import numpy as np, warnings
warnings.filterwarnings('ignore')
from spec_long import make_long
import fast, dss
D = 19.846037499999998; z = np.load('long_D19.8460_svd2.npy')
S = make_long(23, D - 2, nodes=[0.02,0.05,0.1,0.16,0.23,0.3,0.4,0.5,0.6], Ks=60)
parts, _ = S.split(z[None, :]); mS = S.phys(parts[-2], 'odd'); aS = S.phys(parts[-1], 'even'); om = np.array([2*np.pi/z[-1]])
o = dss.DSS.ssh_series(S, mS, aS, om); f = fast.ssh_series_fast(S, mS, aS, om)
for name, a, b in zip("PMAC", o, f):
    for k in range(0, 61, 1):
        d = np.abs(np.asarray(a[k]) - b[k]).max()
        if d > 1e-12 * max(1, np.abs(a[k]).max()):
            print(name, k, "first mismatch", d, np.abs(a[k]).max()); break
    else: print(name, "ok")

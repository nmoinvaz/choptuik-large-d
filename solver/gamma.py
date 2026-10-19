import sys, time, numpy as np, warnings
from scipy.optimize import brentq
warnings.filterwarnings("ignore")
from run_spec import make

def growth_rate(S, z, a, b, xtol=1e-11):
    cache = {}
    def ldet(lam):
        if lam not in cache:
            cache[lam] = np.linalg.slogdet(S.mode_matrix(z, lam))
        return cache[lam]
    sa, la = ldet(a); sb, lb = ldet(b)
    assert sa != sb, f"no sign change in [{a}, {b}]"
    ref = 0.5 * (la + lb)
    g = lambda lam: (lambda s_l: s_l[0] * np.exp(s_l[1] - ref))(ldet(lam))
    return brentq(g, a, b, xtol=xtol, rtol=1e-14), len(cache)

if __name__ == "__main__":
    kc = int(sys.argv[1]); a, b = float(sys.argv[2]), float(sys.argv[3])
    S = make(kc); z = np.load(f"spec_kc{kc}.npy")
    t0 = time.time(); lam, nev = growth_rate(S, z, a, b)
    print(f"kc={kc}: lambda = {lam:.10f}   gamma = 1/lambda = {1/lam:.10f}   ({nev} evaluations, {time.time()-t0:.0f} s)", flush=True)

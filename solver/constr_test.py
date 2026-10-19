import numpy as np, sys
exec(open("refit_rk.py").read().split("print(f\"{len(D)}")[0].replace("Dmax = float(sys.argv[1]) if len(sys.argv) > 1 else 25.5", f"Dmax = {float(sys.argv[1]) if len(sys.argv) > 1 else 25.5}"))
y = yR
print(f"{len(D)} points D = {D.min():.2f}..{D.max():.2f}")
cands = {"free": None, "pi^2/6": np.pi**2/6, "1.6440": 1.6440, "1.6442": 1.6442, "1.6445": 1.6445}
for lo in (8, 10):
    m_ = D >= lo
    for xn, xf in (("1/n", lambda d: 1/(d-2)), ("1/(D-1)", lambda d: 1/(d-1)), ("1/D", lambda d: 1/d)):
        out = []
        for name, L in cands.items():
            best = None
            for m in (8, 9, 10, 11):
                xs = xf(D[m_]); sc = xs.max(); A = np.vander(xs / sc, m + 1, increasing=True)
                if L is None:
                    c, *_ = np.linalg.lstsq(A, y[m_], rcond=None); r = A @ c - y[m_]; c0 = c[0]
                else:
                    c, *_ = np.linalg.lstsq(A[:, 1:], y[m_] - L, rcond=None); r = A[:, 1:] @ c - (y[m_] - L); c0 = L
                rms = np.sqrt(np.mean(r ** 2))
                if m == 10: best = (c0, rms)
            out.append(f"{name}: {best[1]:.1e}" + (f" (c0 {best[0]:.6f})" if L is None else ""))
        print(f"D>={lo:2d} {xn:8s} m=10 rms -> " + "  ".join(out))

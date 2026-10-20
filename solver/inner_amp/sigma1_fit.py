"""Look for a closed form of sigma_1(omega) (first amplitude correction of the inner Stokes multiplier) by linear
least squares over digamma-type basis functions, on the scan_sigma1.py values (6-7 digits)."""
import numpy as np, mpmath as mp, re, itertools
W0 = 2 * np.pi / 1.64389375
data = []
for line in open("scan.txt"):
    m = re.search(r"omega ([\d.]+): sigma_1 = ([+-][\d.]+) ([+-][\d.]+)i", line)
    if m: data.append((float(m.group(1)), complex(float(m.group(2)), float(m.group(3)))))
def basis(w):
    s = 1j * w; a = (1 + s) / 2; b = (2 + s) / 2
    f = {"1": 1, "psi(a)": mp.psi(0, a), "psi(b)": mp.psi(0, b), "psi(1+s)": mp.psi(0, 1 + s), "psi(1/2+s)": mp.psi(0, 0.5 + s),
         "1/s": 1 / s, "1/s^2": 1 / s ** 2, "s": s, "psi1(a)": mp.psi(1, a), "psi1(b)": mp.psi(1, b), "1/(1+s)": 1/(1+s),
         "1/(1+2s)": 1/(1+2*s), "ln s": mp.log(s)}
    return {k: complex(v) for k, v in f.items()}
names = list(basis(1.0).keys())
best = []
for r in (1, 2, 3):
    for combo in itertools.combinations(names, r):
        A = np.array([[basis(w)[c] for c in combo] for w, _ in data])
        y = np.array([v for _, v in data])
        # complex least squares with REAL coefficients (rational-looking combos)
        Ar = np.vstack([A.real, A.imag]); yr = np.r_[y.real, y.imag]
        c, *_ = np.linalg.lstsq(Ar, yr, rcond=None)
        res = np.abs(Ar @ c - yr).max()
        best.append((res, combo, c))
best.sort(key=lambda t: t[0])
for res, combo, c in best[:12]:
    print(f"{res:.2e}  " + " + ".join(f"{ci:+.6f}*{n}" for ci, n in zip(c, combo)))

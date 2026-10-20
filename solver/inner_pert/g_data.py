"""Load the f64 gamma(D) points (rational_fit.gamma_data) and save them as gamma_points.npz (D, gamma, sigma)."""
import os, sys
import numpy as np
S = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, S)
os.chdir(os.path.join(S, "data"))
from rational_fit import gamma_data
D, g, s = gamma_data()
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gamma_points.npz"), D=D, g=g, s=s)
print(len(D), D.min(), D.max())
for d in (4, 5, 6, 8, 10, 15, 20, 25, 30, 35, 40):
    i = np.argmin(abs(D - d)); print(f"D {D[i]:.4f} gamma {g[i]:.9f} sigma {s[i]:.1e}")

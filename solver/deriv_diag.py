"""Derivative diagnostics: g(n) = -n^2 dDelta/dn. If Delta = L + a/n + b n^-1.5 + ..., g -> a; if Delta ~ n^-alpha, g ~ n^(1-alpha)."""
import numpy as np
from data_best import load
rows = load(); D = np.array(sorted(d for d in rows if d >= 7.9)); y = np.array([rows[d]["Delta"] for d in D]); n = D - 2
# local quadratic fit derivative over 5-point windows
g = []; nn = []
for i in range(2, len(n) - 2):
    s = slice(i - 2, i + 3); c = np.polyfit(n[s] - n[i], y[s], 3); g.append(-n[i] ** 2 * c[-2]); nn.append(n[i])
g = np.array(g); nn = np.array(nn)
q = nn * (-g / nn ** 2) / np.interp(nn, n, y) * -1  # local exponent -dlnDelta/dlnn
print(f"{'D':>7} {'g=-n^2 dDelta/dn':>17} {'-dlnD/dlnn':>11} {'d g/d(n^-1/2)':>14}")
dg = np.gradient(g, nn ** -0.5)
for k in range(0, len(nn), max(1, len(nn) // 30)):
    print(f"{nn[k]+2:7.3f} {g[k]:17.6f} {g[k]/nn[k]/np.interp(nn[k], n, y):11.5f} {dg[k]:14.4f}")
np.save("deriv_diag.npy", np.c_[nn, g])

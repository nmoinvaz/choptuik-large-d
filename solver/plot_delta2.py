import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from data_best import load
DREL = 25.5   # above this, Delta is not pinned by the current solver (profile test)
rows = load(); Dall = np.array(sorted(d for d in rows if d >= 3.69)); yall = np.array([rows[d]["Delta"] for d in Dall])
rel = Dall <= DREL
D, y = Dall[rel], yall[rel]; n = D - 2
Du, yu = Dall[~rel], yall[~rel]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C1, C2, C3, NEUT = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold", "figure.facecolor": SURF, "axes.facecolor": SURF})
m = D >= 10
# fits on D >= 10
c4 = np.polyfit(n[m] ** -0.5, y[m], 4)
m8 = D >= 8
A8 = np.stack([np.ones(m8.sum())] + [n[m8] ** -p for p in (1, 1.5, 2)], 1); cL, *_ = np.linalg.lstsq(A8, y[m8], rcond=None)
fL = lambda nn: cL[0] + cL[1] / nn + cL[2] * nn ** -1.5 + cL[3] / nn ** 2
f2 = lambda p, nn: p[0] + p[1] * nn ** -p[2] * (1 + p[3] / nn)
p2 = least_squares(lambda p: f2(p, n[m]) - y[m], [1.5, 2, 0.5, 0]).x
DD = np.linspace(D.min(), 60, 600); NN = DD - 2
fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
for a in ax:
    a.grid(True, color=GRID, lw=0.8); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)
# A: Delta vs D with extrapolations
a = ax[0]
a.plot(DD[DD > 8], fL(NN[DD > 8]), color=C2, lw=2, ls="--", label=f"Δ∞ + a/(D−2) + b/(D−2)^1.5 + c/(D−2)², Δ∞ = {cL[0]:.3f}")
a.plot(DD[DD > 10], f2(p2, NN[DD > 10]), color=C3, lw=2, ls="--", label=f"L + A(D−2)^−α(1+b/(D−2)), α = {p2[2]:.2f}")
a.plot(DD, 4.24 * NN ** -0.25, color=NEUT, lw=1.5, ls=":", label="old crossover law 4.24 (D−2)^−1/4")
a.plot(D, y, "o", color=C1, ms=5, label="computed, pinned (D ≤ 25)")
a.plot(Du, yu, "o", color=C1, ms=5, mfc="none", mew=1.2, label="computed, not pinned (D > 26)")
a.axhline(cL[0], color=C2, lw=1, alpha=0.6)
a.axvspan(DREL, 32, color=GRID, alpha=0.5, lw=0)
a.set_xlim(3, 60); a.set_ylim(1.2, 3.6); a.set_xlabel("spacetime dimension D"); a.set_ylabel("echo period Δ")
a.set_title("Δ levels off instead of falling to zero")
a.text(5, 1.575, f"limit as D → ∞: Δ∞ ≈ {cL[0]:.3f}", color=INK2, fontsize=10)
a.legend(frameon=False, fontsize=9, loc="upper right")
# B: Delta vs 1/sqrt(n): the D -> infinity limit is the intercept at 0
a = ax[1]
X = np.linspace(0, 0.75, 300)
XX = X[(X > 0) & (X < 0.41)]; a.plot(np.r_[0, XX], np.r_[cL[0], fL(XX ** -2)], color=C2, lw=2, ls="--", label="Δ∞ + a/(D−2) + … fit (D ≥ 8), extended to D = ∞")
a.plot(n ** -0.5, y, "o", color=C1, ms=5, label="computed, pinned")
a.plot((Du - 2) ** -0.5, yu, "o", color=C1, ms=5, mfc="none", mew=1.2, label="not pinned (D > 26)")
a.plot([0], [cL[0]], "D", color=C2, ms=8)
a.annotate(f"D = ∞:  Δ ≈ {cL[0]:.3f}", (0, cL[0]), (0.06, 1.52), color=INK, fontsize=10,
           arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
for d in (4, 6, 10, 18):
    a.annotate(f"D={d}", ((d - 2) ** -0.5, np.interp((d - 2) ** -0.5, (n ** -0.5)[::-1], y[::-1])), textcoords="offset points", xytext=(4, -14), color=INK2, fontsize=9)
a.axvspan((32 - 2) ** -0.5, (DREL - 2) ** -0.5, color=GRID, alpha=0.5, lw=0)
a.set_xlim(-0.02, 0.75); a.set_ylim(1.4, 3.6); a.set_xlabel("1/√(D−2)   (D = ∞ at the left edge)"); a.set_ylabel("echo period Δ")
a.set_title("Against 1/√(D−2), the curve flattens onto a finite value")
a.legend(frameon=False, fontsize=9, loc="upper left")
# C: local exponent
a = ax[2]
nall = Dall - 2
q = -np.gradient(np.log(yall), np.log(nall))
a.plot(Dall[rel], q[rel], "o-", color=C1, ms=4, lw=1.5, label="pinned")
a.plot(Dall[~rel], q[~rel], "o", color=C1, ms=4, mfc="none", mew=1.0, label="not pinned")
a.axvspan(DREL, 32, color=GRID, alpha=0.5, lw=0)
a.legend(frameon=False, fontsize=9, loc="upper right")
a.axhline(0.25, color=NEUT, lw=1.2, ls=":"); a.text(15.2, 0.253, "¼ (old law)", color=INK2, fontsize=9)
a.set_xlim(3.5, 32); a.set_xlabel("spacetime dimension D"); a.set_ylabel("local exponent  −d ln Δ / d ln(D−2)")
a.set_title("The power-law exponent peaks at D ≈ 7, then falls")
fig.suptitle(f"Choptuik echo period Δ(D), D = 3.7–{Dall.max():.1f} ({len(Dall)} points; fits use only the {len(D)} pinned points, D ≤ 25; shaded: not pinned)", color=INK, fontsize=13, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig("delta_graph2.png", dpi=150, facecolor=SURF)
print("L-fit coeffs:", np.round(cL, 4)); print("coeffs quartic (high->low):", np.round(c4, 4), " fit2:", np.round(p2, 4))

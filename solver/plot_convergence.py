import json, re, glob, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data_best import load
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C1, C2, C3, NEUT = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
plt.rcParams.update({"font.size": 10.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlecolor": INK, "axes.titlesize": 11.5, "axes.titleweight": "bold", "figure.facecolor": SURF, "axes.facecolor": SURF})
rd = lambda f: [json.loads(l) for l in open(f)]
ex = {round(r['D'], 4): r for r in rd('sdelta_exact.jsonl')}
pol = {round(r['D'], 4): r for r in rd('scan_long_pol.jsonl')}
B = {round(r['D'], 4): r for r in rd('scan_longB.jsonl')}
old = {round(r['D'], 4): r for r in rd('scan_long_kc23.jsonl')}
pts = {}
for k, r in ex.items(): pts[k] = (r['D'], r['residual'], r['s_Delta'], r['s0'])
for k, r in pol.items(): pts[k] = (r['D'], r['residual'], r['s_Delta'], old[k]['s0'])
for k, r in B.items(): pts[k] = (r['D'], r['residual'], r['s_Delta'], r['s0'])
P = np.array([pts[k] for k in sorted(pts)])
D, res, sD, s0 = P.T
# old (finite-difference Jacobian) residuals for comparison, D >= 16
Do = np.array([old[k]['D'] for k in sorted(old) if old[k]['D'] >= 15.9]); ro = np.array([old[k]['residual'] for k in sorted(old) if old[k]['D'] >= 15.9])
fails = []
for f in sorted(glob.glob('scan_long*.log')):
    for l in open(f):
        m = re.match(r'step to D=([\d.]+) failed \(\|F\|=([\deE.+-]+)\)', l)
        if m: fails.append((float(m.group(1)), float(m.group(2))))
fails = np.array(fails)
fig, ax = plt.subplots(2, 2, figsize=(14, 9.4))
for a in ax.flat:
    a.grid(True, color=GRID, lw=0.8); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)
# (a) residuals
a = ax[0, 0]
a.semilogy(Do, ro, "s", color=NEUT, ms=5, mfc="none", label="old solver (finite-difference Jacobian)")
a.semilogy(D, res, "o-", color=C1, ms=5, lw=1.2, label="accepted points, exact horizon Jacobian")
fm = fails[:, 0] > 19
a.semilogy(fails[fm, 0], fails[fm, 1], "x", color=C2, ms=8, mew=2, label="rejected attempts (step then halved)")
a.axhline(5e-11, color=INK2, lw=0.8, ls=":"); a.text(8.2, 6.5e-11, "original pass threshold 5e-11", color=INK2, fontsize=9)
a.set_xlabel("spacetime dimension D"); a.set_ylabel("final residual |F|")
a.set_title("Solver residual: the floor rises with D")
a.legend(frameon=False, fontsize=9, loc="upper left")
# (b) selection strength
a = ax[0, 1]
a.semilogy(D, sD, "o-", color=C3, ms=5, lw=1.2, label="selection strength s_Δ (exact Jacobian)")
for k in range(int(np.ceil(s0.min())), int(s0.max()) + 1):
    xk = np.interp(k, s0, D); a.axvline(xk, color=NEUT, lw=0.8, ls=":")
    a.text(xk, sD.max() * 1.25, f"s0={k}", color=INK2, fontsize=8, ha="center")
a.set_ylim(sD.min() / 2, sD.max() * 2.2)
a.set_xlabel("spacetime dimension D"); a.set_ylabel("how strongly the equations pin Δ")
a.set_title("Why convergence gets harder: Δ is pinned more weakly")
a.legend(frameon=False, fontsize=9, loc="lower left")
a.text(0.98, 0.04, "dotted: horizon exponent s0 crosses an integer", transform=a.transAxes, ha="right", color=INK2, fontsize=8.5)
# (c) Delta_inf estimate vs data range
rows = load(); Dd = np.array(sorted(rows)); y = np.array([rows[d]["Delta"] for d in Dd]); n = Dd - 2
models = [((1, 1.5, 2), C1, "Δ∞ + a/n + b/n^1.5 + c/n²"), ((1, 1.5, 2, 2.5), C2, "… + d/n^2.5"), ((1, 2, 3, 4), C3, "Δ∞ + a/n + b/n² + c/n³ + d/n⁴ (whole powers)")]
Dmaxs = [d for d in Dd if d >= 15]
a = ax[1, 0]
for ps, col, lab in models:
    Ls = []
    for dm in Dmaxs:
        m = (Dd >= 10) & (Dd <= dm); A = np.stack([np.ones(m.sum())] + [n[m] ** -p for p in ps], 1)
        Ls.append(np.linalg.lstsq(A, y[m], rcond=None)[0][0])
    a.plot(Dmaxs, Ls, "o-", color=col, ms=3.5, lw=1.4, label=lab)
a.set_xlabel("largest D included in the fit (fit uses 10 ≤ D ≤ that)"); a.set_ylabel("fitted limit Δ∞")
a.set_title("Convergence of the Δ∞ estimate as data are added")
a.legend(frameon=False, fontsize=9, loc="upper right")
# (d) g diagnostic
a = ax[1, 1]
nn, g = np.load("deriv_diag.npy").T
x = nn ** -0.5
a.plot(x, g, "o", color=C1, ms=4, label="(D−2)² · |dΔ/dD| from the data")
m = nn + 2 >= 18; c = np.polyfit(x[m], g[m], 1); xx = np.linspace(0, x.max(), 50)
a.plot(xx, np.polyval(c, xx), "--", color=C2, lw=1.5, label=f"straight-line extension (D ≥ 18) → {c[1]:.1f} at D = ∞")
a.plot([0], [c[1]], "D", color=C2, ms=7)
a.set_xlim(-0.01, x.max() + 0.02); a.set_xlabel("1/√(D−2)   (D = ∞ at the left edge)"); a.set_ylabel("(D−2)² · |dΔ/dD|")
a.set_title("Slope test: levels off ⇒ Δ has a finite limit")
a.legend(frameon=False, fontsize=9, loc="upper right")
fig.suptitle(f"Convergence of the Δ(D) computation, D = 8–{D.max():.1f}", color=INK, fontsize=13, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig("convergence.png", dpi=140, facecolor=SURF)
print("saved; g-intercept", c[1])

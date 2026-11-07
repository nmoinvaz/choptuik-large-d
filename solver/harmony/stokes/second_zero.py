"""(i) the type drift: 1/increment of sigma_k/sigma_{k+1} and the paper's tail bound C a^k/Gamma(k+b) at k <= 22;
(ii) the zeros x0, x1, x2 of the ring-free Borel-Pade sums at K = 21 (three nodes) and K = 22 (wf 1.00), their spread
over sums, and the velocity d x/d wf of x1 at K = 21; extrapolation of Im x1 = 0 in wf (linear and quadratic)."""
import json
import numpy as np
import mpmath as mp
import zerolib as Z
mp.mp.dps = 60
out = {}
sig = Z.load(1.0)
a, b, C = mp.mpf("1.675"), mp.mpf("4.16"), mp.mpf("27.3")
print("k   |sigma_k| Gamma(k+4.16)/1.675^k   (paper: bound C = 27.3 calibrated on k <= 12)")
drift = {}
for k in range(4, len(sig)):
    v = abs(sig[k]) * mp.gamma(k + b) / a ** k
    drift[k] = float(v)
print("  " + ", ".join(f"{k}: {drift[k]:.2f}" for k in drift))
out["normalised"] = drift
inc = {k: complex(1 / (sig[k] / sig[k + 1] - sig[k - 1] / sig[k])) for k in range(2, len(sig) - 1)}
print("1/increment (kappa estimate):", ", ".join(f"{k}: {inc[k].real:.3f}{inc[k].imag:+.3f}i |{abs(inc[k]):.3f}|"
                                                for k in (8, 10, 12, 14, 16, 18, 20, 21)))
out["kappa_est"] = {k: [v.real, v.imag] for k, v in inc.items()}

def zeros_near(sig, m, n, starts):
    ts, rs, c0 = Z.borel_pade(sig, m, n)
    f = Z.bp_func(ts, rs, c0)
    zs = []
    for s in starts:
        zs.append(mp.findroot(f, mp.mpc(*s), tol=mp.mpf(10) ** -40, maxsteps=200))
    return zs, max(abs(t) for t in ts)

starts = [(7.39, 0), (12.75, 1.12), (3.78, 14.3), (-2.0, 15.0)]
res = {}
for key, K, sums in ((1.0, 22, [(6, 16), (7, 15), (8, 14), (9, 13), (10, 12)]),):
    sg = Z.load(key)
    for m, n in sums:
        zs, tmax = zeros_near(sg, m, n, starts)
        res[f"K22 wf1.00 [{m}/{n}]"] = [[float(z.real), float(z.imag)] for z in zs]
        print(f"K22 wf 1.00 [{m}/{n}] max|t| {float(tmax):.3f}: " + "  ".join(Z.cnum(z, 5) for z in zs))
vel = {}
for m, n in [(6, 15), (7, 14), (8, 13), (9, 12), (10, 11)]:
    row = {}
    for w in (0.99, "1.0K43", 1.01):
        sg = Z.load(w)
        zs, tmax = zeros_near(sg, m, n, starts[:2])
        row[w] = zs
        res[f"K21 wf{w} [{m}/{n}]"] = [[float(z.real), float(z.imag)] for z in zs]
    wfs = [0.99, 1.00, 1.01]
    for i, lab in ((0, "x0"), (1, "x1")):
        xs = [row[w][i] for w in (0.99, "1.0K43", 1.01)]
        v = (xs[2] - xs[0]) / 0.02
        acc = (xs[2] - 2 * xs[1] + xs[0]) / 0.01 ** 2
        im = [float(x.imag) for x in xs]
        c2 = np.polyfit(wfs, im, 2)
        r2 = [r.real for r in np.roots(c2) if abs(r.imag) < 1e-12]
        lin = 1.0 - float(xs[1].imag) / float(v.imag)
        A_lin = mp.sqrt(xs[1] + v * (lin - 1.0))
        print(f"K21 [{m}/{n}] {lab}(0.99,1.00,1.01) = " + "  ".join(Z.cnum(x, 5) for x in xs)
              + f"; dx/dwf = {Z.cnum(v,3)}; Im = 0 at wf {lin:.4f} (linear; A there {Z.cnum(A_lin,3)}), quadratic roots {['%.4f' % r for r in r2]}")
        vel[f"[{m}/{n}] {lab}"] = {"x": [[float(x.real), float(x.imag)] for x in xs], "v": [float(v.real), float(v.imag)],
                                   "acc": [float(acc.real), float(acc.imag)], "wf_lin": lin, "wf_quad": r2}
out["zeros"] = res
out["velocity"] = vel
json.dump(out, open("../second_zero.json", "w"), indent=1)

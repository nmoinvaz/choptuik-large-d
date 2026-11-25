"""Solve form S at one D in the Galerkin layout of galerkin.py: float64 damped Newton, then dd split Newton (#1088).

Usage: python3 galsolve.py --D=100 --Mg=16 --ju=7 --start=STATE [--Delta=...] [--f64iters=15] [--dd=1] [--outer=8]
                           [--tau=1e-8] [--ks=...] [--kc=160] [--out=FILE.json] [--mutate=sigma]
  STATE: an hp npz (projected onto the harmonics) or a json of this script (coefficients, any ju: missing ones zero).
  A start at another D is rescaled: p1, m_h coefficients by sqrt(n0/n), c0 - 1 by n0/n.
Output json: the coefficients (hi and lo), Delta, the residual history, the scaled singular values at the end.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fseries_s import FactoredS  # noqa: E402
from galerkin import GalerkinS, bases  # noqa: E402

opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
D = float(opts["D"])
n = D - 2
Mg = int(opts.get("Mg", 16))
ju = int(opts.get("ju", 7))
ks = int(opts["ks"]) if "ks" in opts else None
kc = int(opts.get("kc", 160))
mut = opts.get("mutate", "")
log = lambda s: print(s, flush=True)  # noqa: E731
t0 = time.time()

g64 = GalerkinS(FactoredS(D, Mg, kc=kc, ks=ks, mutate=mut), Mg, ju)
No, Ne = g64.No, g64.Ne
start = opts["start"]
ylo = None
if start.endswith(".npz"):
    z = np.load(start, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    M0 = meta["M"]
    K = len(meta["nodes"])
    zz = z["zh"] + z["zl"]
    off = 2 * M0 + 4 * M0 * K
    yg = np.concatenate([zz[0:M0], zz[M0:2 * M0], 2 * zz[off:off + M0], [zz[-1]]])
    _, _, Ao, Ae, _, _ = bases(M0, ju)
    y = np.concatenate([Ao @ yg[0:M0], Ae @ yg[M0:2 * M0], Ao @ yg[2 * M0:3 * M0], [yg[3 * M0]]])
    D0 = meta["D"]
else:
    js = json.load(open(start))
    D0 = js["D"]
    y = np.zeros(g64.L)
    ylo = np.zeros(g64.L)

    def put(dst, src, names_dst, names_src):
        for i, nm in enumerate(names_src):
            nm = tuple(nm)
            if nm in names_dst:
                dst[names_dst.index(nm)] = src[i]
    for key, off_, names in (("p1", 0, g64.no), ("c0", No, g64.ne), ("Q0", No + Ne, g64.no)):
        dst = np.zeros(len(names))
        dlo = np.zeros(len(names))
        put(dst, js[key], names, js["names_" + ("e" if key == "c0" else "o")])
        put(dlo, js.get(key + "_lo", [0.0] * len(js[key])), names, js["names_" + ("e" if key == "c0" else "o")])
        y[off_:off_ + len(names)] = dst
        ylo[off_:off_ + len(names)] = dlo
    y[-1] = js["Delta"]
    ylo[-1] = js.get("Delta_lo", 0.0)
n0 = D0 - 2
if abs(D - D0) > 1e-12:
    s = np.sqrt(n0 / n)
    y[0:No] *= s
    y[No + Ne:2 * No + Ne] *= s
    y[No] = 1 + (y[No] - 1) * n0 / n  # the mean of c0
    y[No + 1:No + Ne] *= n0 / n
    ylo = None
if "Delta" in opts:
    y[-1] = float(opts["Delta"])
    ylo = None
log("galsolve: D %.6f n %.4f Mg %d ju %d (L %d) ks %d kc %d start %s (D0 %.6f) Delta0 %.12f" % (
    D, n, Mg, ju, g64.L, g64.inner.ks, kc, start.rsplit("/", 1)[-1], D0, y[-1]))

# float64 damped Newton
hist = []
for it in range(int(opts.get("f64iters", 15)) + 1):
    with np.errstate(all="ignore"):
        F = g64.residual(y)
    nf = float(np.max(np.abs(F)))
    hist.append({"stage": "f64", "it": it, "maxF": nf, "Delta": float(y[-1])})
    log("  f64 it %d max|F| %.3e Delta %.15f (%.0f s)" % (it, nf, y[-1], time.time() - t0))
    if not np.isfinite(nf):
        log("non-finite residual; stop")
        sys.exit(2)
    if it == int(opts.get("f64iters", 15)) or nf < 1e-15:
        break
    J = g64.jacobian(y)
    dy = np.linalg.solve(J, -F)
    rs = np.linalg.norm(J, axis=1)
    r0 = np.linalg.norm(F / rs)
    alpha = 1.0
    for _ in range(16):
        with np.errstate(all="ignore"):
            rt = np.linalg.norm(g64.residual(y + alpha * dy) / rs)
        if np.isfinite(rt) and rt < (1 - 1e-4 * alpha) * r0:
            break
        alpha *= 0.5
    y = y + alpha * dy
J64 = g64.jacobian(y)
r = np.linalg.norm(J64, axis=1)
Js = J64 / r[:, None]
Js = Js / np.linalg.norm(Js, axis=0)[None, :]
sv64 = np.linalg.svd(Js, compute_uv=False)
log("f64: max|F| %.3e Delta %.15f scaled sv min %.2e cond %.2e (%.0f s)" % (hist[-1]["maxF"], y[-1], sv64[-1], sv64[0] / sv64[-1],
                                                                            time.time() - t0))
out = {"D": D, "Mg": Mg, "ju": ju, "ks": g64.inner.ks, "kc": kc, "names_o": g64.no, "names_e": g64.ne,
       "f64_Delta": float(y[-1]), "f64_maxF": hist[-1]["maxF"], "f64_sv_scaled": list(map(float, sv64)), "mutate": mut,
       "start": start}
yfin, ylofin = y, np.zeros_like(y)
if opts.get("dd", "1") == "1":
    import ddnp as d
    from ddnp import DD
    from fseries_sdd import FactoredSDD
    from lsnewton import ls_solve, scaled_svd
    gdd = GalerkinS(FactoredSDD(D, Mg, kc=kc, ks=ks, mutate=mut), Mg, ju, dd=True)
    ydd = DD(y, ylo if (ylo is not None and opts.get("use_lo", "0") == "1") else None)
    lshist = []
    for rnd in range(int(opts.get("rounds", 4))):
        Jd = gdd.jacobian(ydd)
        log("dd Jacobian, round %d (%.0f s)" % (rnd, time.time() - t0))
        ydd, Fd, lh, S0 = ls_solve(gdd.residual, ydd, Jd, tau=float(opts.get("tau", 1e-8)), outer=int(opts.get("outer", 8)),
                                   log=lambda s: log("  " + s))
        lshist += lh
        if float(np.max(np.abs(Fd.h))) < float(opts.get("tol", 1e-26)):
            break
    Jd = gdd.jacobian(ydd)
    rr, cc, U, Sf, Vt = scaled_svd(Jd)
    Jr = (Jd.h + Jd.l) / rr[:, None]
    e = np.zeros(gdd.L)
    e[-1] = 1.0
    grip = float(np.linalg.norm(np.linalg.lstsq(Jr.T, e, rcond=None)[0]))
    nF = float(np.max(np.abs(Fd.h)))
    log("dd: max|F| %.3e Delta %.16f (+%.2e) ; scaled sv max %.3e min %.3e cond %.3e ; smallest %s ; grip %.3e (%.0f s)" % (
        nF, float(ydd.h[-1]), float(ydd.l[-1]), Sf[0], Sf[-1], Sf[0] / Sf[-1], " ".join("%.2e" % v for v in Sf[-6:]), grip,
        time.time() - t0))
    out.update({"Delta": float(ydd.h[-1]), "Delta_lo": float(ydd.l[-1]), "maxF": nF, "ls_hist": lshist,
                "sv_scaled": list(map(float, Sf)), "grip": grip})
    yfin, ylofin = ydd.h, ydd.l
else:
    out.update({"Delta": float(y[-1]), "Delta_lo": 0.0, "maxF": hist[-1]["maxF"], "sv_scaled": list(map(float, sv64))})
out.update({"p1": list(yfin[0:No]), "c0": list(yfin[No:No + Ne]), "Q0": list(yfin[No + Ne:2 * No + Ne]),
            "p1_lo": list(ylofin[0:No]), "c0_lo": list(ylofin[No:No + Ne]), "Q0_lo": list(ylofin[No + Ne:2 * No + Ne]),
            "hist": hist, "seconds": time.time() - t0})
if "out" in opts:
    json.dump(out, open(opts["out"], "w"), indent=1)

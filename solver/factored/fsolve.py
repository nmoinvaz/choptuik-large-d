"""Newton solve of the factored formulation at one D (#1088, notes/LARGE_D_FACTORED.md).

Unknowns p1, c0, Q0 = 2 m_h, the mass-mode anchor (M each) and Delta; rows U, V, c, beta matched at z_K = 1/2 and
the phase row (fseries.py, residual4). The Jacobian is the complex-step one (exact to float64 rounding). Reported per
iteration: max|F|, Delta, the step; at the end the singular values of the Jacobian, raw and with rows and columns
scaled to unit 2-norm, and the grip on Delta, |e_Delta^T J^-1| (scaled rows), the change of Delta per unit change of
the rows.

Usage: python3 fsolve.py --D=100 --M=8 --start=STATE [--Delta=...] [--kc=160] [--ks=...] [--iters=8] [--out=FILE.json]
                         [--mutate=sigma]
  STATE: an hp state (npz) or a json written by --out. When its D differs, p1 and m_h are rescaled by sqrt(n0/n), c0 - 1
  by n0/n, the anchor set to zero, and Delta taken from --Delta (default: the start's).
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fseries import Factored, from_hp_state, harmonics_odd, resample_even, resample_odd  # noqa: E402
from fseries_s import FactoredS  # noqa: E402


def load_start(start, M):
    if start.endswith(".npz"):
        y, meta, raw = from_hp_state(start, M)
        y4 = np.concatenate([y[:3 * M], np.zeros(M), y[3 * M:]])
        return y4, meta["D"]
    js = json.load(open(start))
    y4 = np.concatenate([resample_odd(np.array(js["p1"]), M), resample_even(np.array(js["c0"]), M),
                         resample_odd(np.array(js["Q0"]), M), np.zeros(M), [js["Delta"]]])
    if js["M"] == M and "anchor" in js:
        y4[3 * M:4 * M] = js["anchor"]
    return y4, js["D"]


class FormS:
    """FactoredS in the 4M+1 layout of form A (the anchor entries kept at zero, their rows the identity), so that
    the driver, the conditioning and the output files are shared; conditioning() drops them."""

    def __init__(self, D, M, **kw):
        self.f = FactoredS(D, M, **kw)
        self.M = M
        self.ks = self.f.ks
        self.kc = self.f.kc
        self.stats = {"sweeps": 0, "beta_dif": 0.0}

    def _y3(self, y):
        M = self.M
        return np.concatenate([y[:3 * M], y[4 * M:]])

    def residual4(self, y):
        M = self.M
        R = self.f.residual(self._y3(y))
        self.last = self.f.last
        return np.concatenate([R[:3 * M], y[3 * M:4 * M], R[3 * M:]])

    def jacobian4(self, y):
        M = self.M
        J3 = self.f.jacobian(self._y3(y))
        L = 4 * M + 1
        J = np.zeros((L, L))
        idx = list(range(3 * M)) + [4 * M]
        J[np.ix_(list(range(3 * M)) + [4 * M], idx)] = J3
        J[3 * M:4 * M, 3 * M:4 * M] = np.eye(M)
        return J


def solve(D, M, y, kc=160, ks=None, iters=8, mutate="", log=print, zK=0.5, form="A"):
    fz = Factored(D, M, kc=kc, ks=ks, zK=zK, mutate=mutate) if form == "A" else FormS(D, M, kc=kc, ks=ks, zK=zK, mutate=mutate)
    t0 = time.time()
    hist = []
    for it in range(iters + 1):
        F = fz.residual4(y)
        ce, ho = fz.last
        nf = np.max(np.abs(F))
        log("it %d max|F| %.3e Delta %.15f block %d (%.0e) tails c %.1e h %.1e  (%.0f s)" % (
            it, nf, y[4 * M], fz.stats["sweeps"], fz.stats["beta_dif"], ce["tail"], ho["tail"], time.time() - t0))
        hist.append({"it": it, "maxF": float(nf), "Delta": float(y[4 * M])})
        if it == iters:
            break
        J = fz.jacobian4(y)
        dy = np.linalg.solve(J, -F)
        # backtracking on the row-scaled residual (the weak directions are strongly curved: a full step can overshoot)
        rs = np.linalg.norm(J, axis=1)
        r0 = np.linalg.norm(F / rs)
        alpha = 1.0
        for _ in range(16):
            with np.errstate(all="ignore"):
                Ft = fz.residual4(y + alpha * dy)
            rt = np.linalg.norm(Ft / rs)
            if np.isfinite(rt) and rt < (1 - 1e-4 * alpha) * r0:
                break
            alpha *= 0.5
        y = y + alpha * dy
        log("   step max|dy| %.3e dDelta %.3e alpha %.3g" % (np.max(np.abs(dy)), dy[4 * M], alpha))
        if abs(dy[4 * M]) < 1e-15 and np.max(np.abs(dy)) < 1e-14 and it > 0:
            F = fz.residual4(y)
            nf = np.max(np.abs(F))
            log("it %d max|F| %.3e Delta %.15f (step below 1e-15)" % (it + 1, nf, y[4 * M]))
            hist.append({"it": it + 1, "maxF": float(nf), "Delta": float(y[4 * M])})
            break
    J = fz.jacobian4(y)
    return y, J, fz, hist, time.time() - t0


def conditioning(J, M, form="A"):
    if form == "S":
        keep = list(range(3 * M)) + [4 * M]
        J = J[np.ix_(keep, keep)]
        M4 = 3 * M
    else:
        M4 = 4 * M
    sv = np.linalg.svd(J, compute_uv=False)
    r = np.linalg.norm(J, axis=1)
    Js = J / r[:, None]
    cn = np.linalg.norm(Js, axis=0)
    Js = Js / cn[None, :]
    svs = np.linalg.svd(Js, compute_uv=False)
    # grip: e_Delta^T J^-1 in the scaled rows (unscaled Delta column)
    Jr = J / r[:, None]
    e = np.zeros(J.shape[0])
    e[M4] = 1.0
    gD = np.linalg.solve(Jr.T, e)
    return sv, svs, float(np.linalg.norm(gD))


if __name__ == "__main__":
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    D = float(opts["D"])
    M = int(opts.get("M", 8))
    n = D - 2
    y, D0 = load_start(opts["start"], M)
    n0 = D0 - 2
    if abs(D - D0) > 1e-12:
        s = np.sqrt(n0 / n)
        y[0:M] *= s
        y[2 * M:3 * M] *= s
        y[M:2 * M] = 1 + (y[M:2 * M] - 1) * n0 / n
        y[3 * M:4 * M] = 0.0
    if "Delta" in opts:
        y[4 * M] = float(opts["Delta"])
    log = lambda s: print(s, flush=True)  # noqa: E731
    fz0 = Factored(D, M, kc=int(opts.get("kc", 160)), ks=int(opts["ks"]) if "ks" in opts else None)
    log("D %.6f n %.4f M %d kc %d ks %d start %s (D0 %.6f)" % (D, n, M, fz0.kc, fz0.ks, opts["start"].rsplit("/", 1)[-1], D0))
    form = opts.get("form", "A")
    if form == "S":
        y[3 * M:4 * M] = 0.0
    y, J, fz, hist, secs = solve(D, M, y, kc=int(opts.get("kc", 160)), ks=int(opts["ks"]) if "ks" in opts else None,
                                 iters=int(opts.get("iters", 8)), mutate=opts.get("mutate", ""), log=log, form=form)
    sv, svs, grip = conditioning(J, M, form)
    log("singular values (raw): max %.3e min %.3e cond %.3e" % (sv[0], sv[-1], sv[0] / sv[-1]))
    log("singular values (rows, columns unit): max %.3e min %.3e cond %.3e" % (svs[0], svs[-1], svs[0] / svs[-1]))
    log("  smallest five (scaled): " + " ".join("%.3e" % v for v in svs[-5:]))
    log("grip |e_Delta^T J^-1| (unit rows): %.3e" % grip)
    log("p1 harmonics: " + " ".join("%.3e" % v for v in harmonics_odd(y[0:M])))
    log("Delta %.15f" % y[4 * M])
    if "out" in opts:
        out = {"D": D, "M": M, "kc": fz.kc, "ks": fz.ks, "p1": list(y[0:M]), "c0": list(y[M:2 * M]),
               "Q0": list(y[2 * M:3 * M]), "anchor": list(y[3 * M:4 * M]), "Delta": float(y[4 * M]), "hist": hist,
               "sv": list(map(float, sv)), "sv_scaled": list(map(float, svs)), "grip": grip,
               "maxF": hist[-1]["maxF"], "mutate": opts.get("mutate", ""), "start": opts["start"], "seconds": secs,
               "form": form}
        json.dump(out, open(opts["out"], "w"), indent=1)

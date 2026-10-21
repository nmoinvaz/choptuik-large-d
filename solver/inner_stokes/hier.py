"""Inner selection at M harmonics by a hierarchical (back-substitution) Newton solve: S_M = 0 innermost, then S_{M-2},
..., S_3, and S_1 = 0 for (A, w) outermost.

Why: the de-mixed residual r_nu (inner_np.residual_demixed) is block upper-triangular with enormous couplings. Measured
at M = 5, A = 2.72, w_inf: dr_5/dB5 ~ 4e8, dr_3/dB5 ~ 1e13, dr_1/dB5 ~ 1e17, against dr_3/dB3 ~ 5e5 and dr_1/dA ~ 1e4
(each S_mu leaks into the lower r_nu with weight ~|K'_mu/K'_nu| ~ e^{pi (mu - nu) w / 2}). Away from S_M ~ 0 the
harmonic-M jump is also strongly nonlinear. A joint solve (full_M.py, LM with absolute tolerances) therefore stalls.
Here every level is a 2 x 2 damped Newton on its own residual, with all higher levels re-solved inside each
evaluation (warm-started), chord Jacobians cached per level and refreshed when a step fails, finite-difference steps
scaled to the level, and convergence judged by the step size (no absolute residual tolerance).

Each condition r_nu is evaluated at its own radius from one integration: nu <= 3 at xi_top (0.2); nu >= 5 first at a
small radius where its jump is linear (approach, about 0.119 for nu = 5, 0.103 for nu = 7), then polished at xi_top,
where near the zero it is resolved ~1e8 times better (r_5 floor 2e-10 at 0.2 vs 5e-7 at 0.119). The polish matters:
dr_3/dB5 ~ 1e13 turns the small-radius floor of r_5 into noise ~30 in r_3.

Seeds: F_3, F_5, F_7, ... from agent C's amplitude expansion (solver/inner_amp/data/sigF_w1.0000_K15_J1000_phys.json,
tuned f_nu summed to eps^15 at the given A; computed at w = w_inf; falls back to the legacy K17_J1500 file summed to
eps^11), or given explicitly with --x (missing higher harmonics are filled from the seed file).

Result (2026-10-20, M = 5, A = 2.72, w_inf, --fixA, 403 s on one core): F3/F1 = 0.054859 + 0.005407i,
F5/F1 = 0.003089 + 0.000540i, r_1 = 6.50 + 1.81i (M = 3: -58.8 - 99.1i).

    python -u hier.py M A wf [--fixA] [--seed=FILE] [--x=B3,C3,B5,C5,...] [--xi1=0.2] [--procs N]
      --fixA     keep (A, w) fixed: solve S_3..S_M = 0, print F_nu/F_1 and the harmonic-1 residual r_1
      --procs N  finite-difference columns of the levels run in the main process go to N worker processes
                 (spawn start method; Windows-safe). Only 2 columns per level, so N = 2..3 is all it can use.
F_nu = B_nu/(2i) + C_nu/2 is the e^{i nu theta} coefficient of F = A sin(theta) + ... (F_1 = A/(2i)).
"""
import os
import sys
import time
import json
import numpy as np
import inner_np as I

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(HERE, "..", "inner_amp", "data")
# converged physical-basis fit first (main e5406bf); the legacy files are mis-tuned from order 11 (nu = 7) / 13 (nu = 3)
DEFAULT_SEED = next((os.path.join(_DATA, f) for f in ("sigF_w1.0000_K15_J1000_phys.json", "sigF_w1.0000_K17_J1500.json")
                     if os.path.exists(os.path.join(_DATA, f))), os.path.join(_DATA, "sigF_w1.0000_K15_J1000_phys.json"))
POOL = None
NEV = [0]


def seed_from_C(path, A, M, kmax=None):
    d = json.load(open(path))
    kmax = kmax or (d["K"] if "phys" in os.path.basename(path) else 11)
    f = {tuple(int(t) for t in k.split("_")): complex(*v) for k, v in d["f"].items()}
    out = []
    for nu in range(3, M + 1, 2):
        Fn = sum(f.get((k, nu), 0) * A ** k for k in range(3, kmax + 1, 2))
        out += [-2 * Fn.imag, 2 * Fn.real]
    return np.array(out)


def demixed_multi(x, M, radii, NTH=64):
    """inner_np.residual_demixed at several radii from one integration: dict r1 -> residual vector"""
    w = I.W0 * x[1]
    Fh = I.Fh_from(x, M)
    m = I.Inner(M, w, Fh, True)
    J, info = m.jumps(tuple(sorted(set(radii))))
    th = 2 * np.pi * np.arange(NTH) / NTH
    kk = np.fft.fftfreq(NTH, 1 / NTH)
    Fth = sum(v * np.exp(1j * mu * th) for mu, v in Fh.items()).real
    F2h = np.fft.fft(Fth ** 2) / NTH
    Pi = np.real(np.fft.ifft(np.where(kk != 0, F2h / (1j * kk + (kk == 0)), 0)) * NTH) / w
    out = {}
    for r1, Jd in J.items():
        Jth = sum(Jd[k + m.L] * np.exp(1j * k * th) for k in range(-M, M + 1) if k % 2)
        q = np.fft.fft(Jth * np.exp(-Pi)) / NTH
        lsc = 1 / (2 * r1 * r1) - 2 * I.F2mean(Fh) * np.log(r1)
        v = []
        for nu in range(1, M + 1, 2):
            z = q[nu % NTH] * r1 ** (2j * nu * w) * np.exp(lsc - np.log(2 * np.pi * abs(I.Kp(1j * nu * w))))
            v += [z.real, z.imag]
        out[r1] = np.array(v)
    return out


def level_radius(nu, w, xi_top=0.2, margin=1e3):
    """radius at which the harmonic-nu condition is evaluated: the jump there is ~margin^-1 of its multiplier
    (linear regime) but still resolvable; levels 1 and 3 use xi_top"""
    if nu <= 3:
        return xi_top
    L = np.log(margin * 2 * np.pi * abs(I.Kp(1j * nu * w)))
    return min(xi_top, 1 / np.sqrt(2 * L))


def idx(l):
    return [0, 1] if l == 1 else [l - 1, l]


class Hier:
    def __init__(self, M, r1=0.2, top=1, verbose=1):
        self.M, self.r1, self.top, self.verbose = M, r1, top, verbose
        self.J = {}            # chord Jacobian per (level, mode)
        self.fresh = {}
        self.mode = {}         # level >= 5: 'low' (small radius, approach) or 'top' (xi_top, polish)

    def radii(self, x):
        w = I.W0 * x[1]
        return {nu: level_radius(nu, w, self.r1) for nu in range(1, self.M + 1, 2)}

    def res(self, x):
        """packed residual: r_nu for nu <= 3 at xi_top; for nu >= 5 the level's current mode ('low' radius while
        approaching, 'top' once near the zero). Also keeps both sets in self.last."""
        NEV[0] += 1
        rad = self.radii(x)
        R = demixed_multi(x, self.M, [self.r1] + list(rad.values()))
        out = R[self.r1].copy()
        for nu in range(5, self.M + 1, 2):
            if self.mode.get(nu, "low") == "low":
                out[idx(nu)] = R[rad[nu]][idx(nu)]
        self.last = (R, rad)
        return out

    def solve_from(self, l, x):
        """solve levels l..M (l > M: just evaluate). Returns x, full residual vector."""
        if l > self.M:
            return x, self.res(x)
        return self.level(l, x)

    def h(self, l, x):
        if l == 1:
            return np.array([1e-4, 1e-4])
        b = x[idx(l)]
        return 1e-6 * np.maximum(np.abs(b), 1e-3 * np.abs(x[0]))

    def fdjac(self, l, x, g0):
        hs = self.h(l, x)
        xs = []
        for i, k in enumerate(idx(l)):
            xt = x.copy()
            xt[k] += hs[i]
            xs.append(xt)
        if POOL is not None and l >= self.top:
            outs = POOL.map(_worker, [(self.M, self.r1, l + 2, xt, dict(self.mode), dict(self.J)) for xt in xs])
            NEV[0] += sum(o[2] for o in outs)
        else:
            outs = [self.solve_from(l + 2, xt) + (0,) for xt in xs]
        cols = [(o[1][idx(l)] - g0) / hs[i] for i, o in enumerate(outs)]
        return np.array(cols).T

    def level(self, l, x, maxit=25):
        if l >= 5:
            self.mode[l] = "top"
            x1, r = self.solve_from(l + 2, x.copy())
            if not np.all(np.isfinite(r)) or np.abs(r[idx(l)]).max() > 1e-4:
                self.mode[l] = "low"
                x, r = self._newton(l, x1)
                self.mode[l] = "top"
                x, r = self.solve_from(l + 2, x)
            else:
                x = x1
            return self._newton(l, x, r)
        return self._newton(l, x)

    def _newton(self, l, x, r=None, maxit=25):
        key = (l, self.mode.get(l, "top"))
        if r is None:
            x, r = self.solve_from(l + 2, x.copy())
        g = r[idx(l)]
        if key not in self.J:
            self.J[key] = self.fdjac(l, x, g)
            self.fresh[key] = True
        for it in range(maxit):
            try:
                step = -np.linalg.solve(self.J[key], g)
            except np.linalg.LinAlgError:
                step = -np.linalg.lstsq(self.J[key], g, rcond=None)[0]
            scale = np.abs(x[idx(l)]).max() + 1e-3 * abs(x[0])
            if np.abs(step).max() < 2e-16 * scale:
                break
            accepted = False
            for lam in ((1.0, 0.5, 0.25, 0.125, 1 / 16, 1 / 32) if not self.fresh[key] or it == 0 else (1.0, 0.5)):
                xt = x.copy()
                xt[idx(l)] += lam * step
                try:
                    xt, rt = self.solve_from(l + 2, xt)
                    gt = rt[idx(l)]
                    ok = np.all(np.isfinite(rt)) and np.linalg.norm(gt) < (1 - 1e-4 * lam) * np.linalg.norm(g)
                except Exception:
                    ok = False
                if ok:
                    accepted = True
                    break
            if not accepted:
                if self.fresh[key]:
                    if self.verbose and l <= self.top + 2:
                        print(f"   level {l} ({key[1]}): stop at |g| {np.linalg.norm(g):.2e} (noise floor)", flush=True)
                    break
                self.J[key] = self.fdjac(l, x, g)
                self.fresh[key] = True
                continue
            slow = np.linalg.norm(gt) > 0.3 * np.linalg.norm(g)
            x, r, g = xt, rt, gt
            self.fresh[key] = False
            if slow:
                self.J[key] = self.fdjac(l, x, g)
                self.fresh[key] = True
            if self.verbose and (l == self.top or self.verbose > 1):
                print(f"level {l} it {it}: |r_{l}| {np.linalg.norm(g):.3e}  lam {lam:.3g}  x {np.round(x, 9).tolist()}  "
                      f"[{time.time() - T0:.0f} s, {NEV[0]} residuals]", flush=True)
        return x, r


def _worker(args):
    M, r1, l, x, mode, Jc = args
    n0 = NEV[0]
    h = Hier(M, r1, top=99, verbose=0)
    h.mode = mode
    h.J = Jc                                   # warm chord Jacobians from the main process
    h.fresh = {k: False for k in Jc}
    x, r = h.solve_from(l, x)
    return x, r, NEV[0] - n0


def report(x, r, M, tag):
    Fh = I.Fh_from(x, M)
    w = x[1] * I.W0
    print(f"{tag} M {M}: A {x[0]:.6f}  w/w_inf {x[1]:.6f}  (w {w:.5f}, Delta {2 * np.pi / w:.5f})  <F^2> {I.F2mean(Fh):.4f}  "
          f"r_1 {r[0]:+.4e}{r[1]:+.4e}i  [{time.time() - T0:.0f} s, {NEV[0]} residuals]", flush=True)
    for nu in range(3, M + 1, 2):
        rr = Fh[nu] / Fh[1]
        rn = r[nu - 1] + 1j * r[nu]
        print(f"      F_{nu}/F_1 = {rr.real:+.6f}{rr.imag:+.6f}i   |.| {abs(rr):.6f}  arg {np.angle(rr):+.4f}   r_{nu} {abs(rn):.1e}",
              flush=True)


def main():
    global POOL
    argv = sys.argv[1:]
    opts = {a.split("=")[0]: (a.split("=", 1)[1] if "=" in a else True) for a in argv if a.startswith("--")}
    args = [a for a in argv if not a.startswith("--")]
    if "--procs" in argv:
        k = argv.index("--procs")
        opts["--procs"] = argv[k + 1]
        args = [a for a in args if a != argv[k + 1]]
    M, A, wf = int(args[0]), float(args[1]), float(args[2])
    r1 = float(opts.get("--xi1", 0.2))
    if "--x" in opts:
        v = np.array([float(t) for t in opts["--x"].split(",")])
        if len(v) < M - 1:
            v = np.concatenate([v, seed_from_C(opts.get("--seed", DEFAULT_SEED), A, M)[len(v):]])
    else:
        v = seed_from_C(opts.get("--seed", DEFAULT_SEED), A, M)
    x = np.concatenate([[A, wf], v])
    procs = int(opts.get("--procs", 1))
    if procs > 1:
        import multiprocessing as mpc
        POOL = mpc.get_context("spawn").Pool(procs)
    top = 3 if "--fixA" in opts else 1
    print(f"M {M}, xi1 {r1}, {'fixed (A, w)' if top == 3 else 'full selection'}, procs {procs}; start x {np.round(x, 9).tolist()}",
          flush=True)
    h = Hier(M, r1, top=top)
    print("level radii: " + ", ".join(f"r_{nu} at xi {v:.4f}" for nu, v in h.radii(x).items()), flush=True)
    r0 = h.res(x)
    print("start residuals: " + "  ".join(f"|r_{nu}| {np.hypot(r0[nu - 1 if nu > 1 else 0], r0[nu if nu > 1 else 1]):.2e}"
                                         for nu in range(1, M + 1, 2)), flush=True)
    x, r = h.solve_from(top, x)
    report(x, r, M, "FIXED-A" if top == 3 else "ZERO")


if __name__ == "__main__":
    main()

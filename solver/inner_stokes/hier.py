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
F5/F1 = 0.003089 + 0.000540i, r_1 = 6.50 + 1.81i (M = 3: -58.8 - 99.1i); with --schur 218 s, r_1 = 6.484 + 1.854i.

M = 7 (2026-10-20): the polish radius is per level (polish_radius: nu = 5 at 0.2 as before, nu = 7 at ~0.15; at 0.2
the harmonic-7 jump left over by the approach is ~1e3, far outside the linear regime), the Newton stops as soon as a
fresh Jacobian no longer halves |g| (before: up to 25 noise-floor iterations with Jacobian refreshes, ~75 residuals
per level-7 solve, which is why the first M = 7 run printed nothing for 45 min), and --schur removes the ulp leakage:
near the zeros one ulp of B7 (5.4e-20) moves r_1 by ~200 at xi 0.2 (~15-20 at 0.15), r_3 by ~0.05, r_5 by ~5e-6,
i.e. a 0.3-3% floor on w, while r_7 at the same radius resolves sub-ulp offsets of B7 to ~1e-11 ulp. The correction
r_l - K_lH K_HH^-1 r_H (K by finite differences, M - 1 residuals) is r_l at the exact higher-level zeros.
Use --schur for M >= 7 (it also helps at M = 5).

    python -u hier.py M A wf [--fixA] [--x=B3,C3,...] [--seed=FILE] [--xi1=0.2] [--procs N] [--verbose=2]
                             [--schur] [--predict] [--floor] [--tol=T] [--fasttrial]
  --fixA       (A, w) fixed: solve S_3..S_M = 0, report F_nu/F_1 and r_1
  --schur      leakage Jacobian K after the first inner solve; Schur-corrected residuals from then on
  --predict    (implies --schur) K also over (A, w); every nested re-solve starts from the linear prediction of the
               higher-level zeros; outer FD step 1e-6
  --floor      (with K) stop every nested solve (levels below the top) at 3x its level's noise floor, measured once by
               the first solves after K that end by failed trials (instead of rediscovering it in every solve)
  --tol=T      inner levels stop at |r_l| < T once K exists (default 0). T = 1e-7 tested at M = 5: worse (r_5 leaks
               into r_3 with weight ~6e4 and K is only ~1e-4 accurate)
  --fasttrial  near the floor try lam = 1, 0.25 then refresh J (default: halvings down to 1/32); not validated
Without --predict/--floor/--tol/--fasttrial the solver is the one on main (047ea84).
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
DEFAULT_SEED = next((os.path.join(_DATA, f) for f in ("sigF_w1.0000_K17_J750_phys.json", "sigF_w1.0000_K15_J1000_phys.json", "sigF_w1.0000_K17_J1500.json")
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


def polish_radius(nu, w, F2, xi_top=0.2, sc_min=1e-2):
    """radius at which a level nu >= 5 is polished after the small-radius approach: the largest xi <= xi_top at which
    the normalization e^{1/(2xi^2)} xi^{-2<F^2>} / (2 pi |K'_nu|) is still >= sc_min, i.e. where the harmonic-nu jump
    left over by the approach (r_nu ~ 1e-6) is still linear (J ~ 1e-4). nu = 5: xi_top (0.2), as in the converged M = 5
    run; nu = 7: about 0.15 (at 0.2 the leftover jump would be ~1e3, far from linear)."""
    if nu <= 3:
        return xi_top
    L = np.log(2 * np.pi * abs(I.Kp(1j * nu * w))) + np.log(sc_min)
    f = lambda xi: 1 / (2 * xi * xi) - 2 * F2 * np.log(xi) - L
    if f(xi_top) >= 0:
        return xi_top
    lo, hi = 0.05, xi_top
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) >= 0 else (lo, mid)
    return lo


def idx(l):
    return [0, 1] if l == 1 else [l - 1, l]


class Hier:
    def __init__(self, M, r1=0.2, top=1, verbose=1):
        self.M, self.r1, self.top, self.verbose = M, r1, top, verbose
        self.J = {}            # chord Jacobian per (level, mode)
        self.fresh = {}
        self.mode = {}         # level >= 5: 'low' (small radius, approach) or 'top' (xi_top, polish)
        self.K = None          # leakage Jacobian for the Schur correction (--schur)
        self.schur = False
        self.predict = False   # --predict: linear predictor for the higher levels (needs K, implies --schur)
        self.tol = 0.0         # --tol: inner levels stop at |r_l| < tol once K exists (default 1e-7 with --predict)
        self.xref, self.xorder, self.count = {}, {}, 0
        self.fasttrial = False # --fasttrial: near the floor try lam = 1, 0.25 only, then refresh J (default: 6 halvings)
        self.floorstop = False # --floor: stop nested solves at 3x the measured noise floor of their level
        self.floor = {}        # (level, mode) -> measured floor of |r_l| (median end value of floor-discovery solves)
        self.floorhist = {}
        self.ltop = top        # the run's top level (workers: the main process's), for --tol / --floor

    def radii(self, x):
        """(low, polish) radius per level"""
        w = I.W0 * x[1]
        F2 = I.F2mean(I.Fh_from(x, self.M))
        low = {nu: level_radius(nu, w, self.r1) for nu in range(1, self.M + 1, 2)}
        pol = {nu: polish_radius(nu, w, F2, self.r1) for nu in range(1, self.M + 1, 2)}
        return low, pol

    def res(self, x):
        """residual used by the solver: the raw residual with the sub-ulp (Schur) leakage correction when available"""
        r = self.res_raw(x)
        return self.correct(r) if self.K is not None else r

    def correct(self, r):
        """r_l -> r_l - K_lH K_HH^-1 r_H for every level l, H = the levels above l (all in polish mode): the value r_l
        would have if the higher-harmonic unknowns were moved (by far less than one ulp) onto their exact zeros.
        Removes the leakage of the ulp-limited S_mu (mu > l) into r_l (M = 7: one ulp of B7 moves r_1 by ~200)."""
        out = r.copy()
        for l in range(1, self.M - 1, 2):
            H = list(range(l + 2, self.M + 1, 2))
            if any(self.mode.get(nu, "low") != "top" for nu in H if nu >= 5):
                continue
            rows = sum((idx(nu) for nu in H), [])
            dv = np.linalg.solve(self.K[np.ix_(rows, rows)], r[rows])
            out[idx(l)] = r[idx(l)] - self.K[np.ix_(idx(l), rows)] @ dv
        return out

    def leak_jac(self, x):
        """K = d r_raw / d (B3, C3, ..., B_M, C_M) at x, with every level >= 5 in polish mode (linear regime near the
        zeros); steps 1e-12 relative (thousands of ulps, still linear)"""
        saved = dict(self.mode)
        for nu in range(5, self.M + 1, 2):
            self.mode[nu] = "top"
        K = np.zeros((self.M + 1, self.M + 1))
        r0 = self.res_raw(x)
        for k in range(0 if self.predict else 2, self.M + 1):
            rel = 1e-10 if (self.predict and k < 4) else 1e-12   # --predict: B3, C3, A, w at 1e-10 (accuracy)
            hk = rel * max(abs(x[k]), 1e-3 * abs(x[0])) if k >= 2 else rel * abs(x[k])
            xp = x.copy()
            xp[k] += hk
            K[:, k] = (self.res_raw(xp) - r0) / hk
        self.mode = saved
        self.K = K
        if self.verbose:
            print(f"   Jacobian K computed ({self.M + 1 if self.predict else self.M - 1} residuals) [{time.time() - T0:.0f} s]", flush=True)

    def res_raw(self, x):
        """packed residual: r_nu for nu <= 3 at xi_top; for nu >= 5 at the level's current mode radius ('low' while
        approaching, 'top' = polish radius once near the zero); all radii from one integration."""
        NEV[0] += 1
        low, pol = self.radii(x)
        R = demixed_multi(x, self.M, [self.r1] + list(low.values()) + list(pol.values()))
        out = R[self.r1].copy()
        for nu in range(5, self.M + 1, 2):
            rr = low[nu] if self.mode.get(nu, "low") == "low" else pol[nu]
            out[idx(nu)] = R[rr][idx(nu)]
        return out

    def solve_from(self, l, x):
        """solve levels l..M (l > M: just evaluate). Returns x, full residual vector."""
        if l > self.M:
            return x, self.res(x)
        return self.level(self.predicted(l, x), l)

    def predicted(self, l, x):
        """--predict: move the unknowns of levels >= l onto the linearized zeros of r_l..r_M, given the change of the
        lower unknowns since the last state at which levels >= l were solved (K = full raw Jacobian). The nested
        re-solve then starts in the linear (polish) regime and needs a few residuals instead of a fresh approach."""
        if not self.predict or self.K is None or l < 3:
            return x
        cands = [(self.xorder[k], self.xref[k]) for k in self.xref if k <= l]
        if not cands:
            return x
        xr = max(cands, key=lambda t: t[0])[1]
        lo = list(range(0, l - 1))
        H = list(range(l - 1, self.M + 1))
        d = x[lo] - xr[lo]
        if not np.any(d):
            return x
        x = x.copy()
        x[H] = xr[H] - np.linalg.solve(self.K[np.ix_(H, H)], self.K[np.ix_(H, lo)] @ d)
        return x

    def h(self, l, x):
        if l == 1:
            return np.array([1e-6, 1e-6]) if self.predict else np.array([1e-4, 1e-4])
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
            st = (self.predict, dict(self.xref), dict(self.xorder), self.count, self.tol, self.ltop, self.fasttrial,
                  self.floorstop, dict(self.floor))
            outs = POOL.map(_worker, [(self.M, self.r1, l + 2, xt, dict(self.mode), dict(self.J), self.K, st) for xt in xs])
            NEV[0] += sum(o[2] for o in outs)
        else:
            outs = [self.solve_from(l + 2, xt) + (0,) for xt in xs]
        cols = [(o[1][idx(l)] - g0) / hs[i] for i, o in enumerate(outs)]
        return np.array(cols).T

    def level(self, x, l, maxit=15):
        t0, n0 = time.time(), NEV[0]
        x, r = self._level(l, x)
        self.count += 1
        self.xref[l], self.xorder[l] = x.copy(), self.count
        if self.verbose >= 2 or (self.verbose and l <= self.top + 2):
            print(f"   level {l} solved: |r_{l}| {np.linalg.norm(r[idx(l)]):.2e}  ({NEV[0] - n0} residuals, {time.time() - t0:.0f} s)  "
                  f"[{time.time() - T0:.0f} s total]", flush=True)
        return x, r

    def _level(self, l, x):
        if l == self.top and self.schur and self.K is None and l < self.M:
            x, _ = self.solve_from(l + 2, x.copy())
            self.leak_jac(x)
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

    def at_floor(self, key, l, g):
        """--floor: a nested level (l > top, polish mode) whose |r_l| is within 3x its measured floor is done"""
        return (self.floorstop and self.K is not None and l > self.ltop and key in self.floor and key[1] == "top"
                and np.linalg.norm(g) < 3 * self.floor[key])

    def record_floor(self, key, l, g):
        """--floor: the end value of a solve that discovered its noise floor (failed trials with a fresh Jacobian)"""
        if self.floorstop and self.K is not None and l > self.ltop and key[1] == "top":
            new = key not in self.floor
            hist = self.floorhist.setdefault(key, [])
            hist.append(float(np.linalg.norm(g)))
            self.floor[key] = float(np.median(hist))     # median: one early stop must not inflate the floor
            if new and self.verbose:
                print(f"   floor of level {l}: |r_{l}| ~ {self.floor[key]:.2e} (stop at 3x from now on)", flush=True)

    def _newton(self, l, x, r=None, maxit=15):
        key = (l, self.mode.get(l, "top"))
        if r is None:
            x, r = self.solve_from(l + 2, x.copy())
        g = r[idx(l)]
        if key not in self.J:
            self.J[key] = self.fdjac(l, x, g)
            self.fresh[key] = True
        for it in range(maxit):
            if self.at_floor(key, l, g):
                break
            if self.tol and self.K is not None and l > self.ltop and np.linalg.norm(g) < self.tol:
                break                          # inner level in the linear regime: the Schur correction does the rest
            try:
                step = -np.linalg.solve(self.J[key], g)
            except np.linalg.LinAlgError:
                step = -np.linalg.lstsq(self.J[key], g, rcond=None)[0]
            scale = np.abs(x[idx(l)]).max() + 1e-3 * abs(x[0])
            if np.abs(step).max() < 2e-16 * scale:
                self.record_floor(key, l, g)
                break
            accepted = False
            if not self.fasttrial:
                lams = (1.0, 0.5, 0.25, 0.125, 1 / 16, 1 / 32) if not self.fresh[key] or it == 0 else (1.0, 0.5)
            elif it == 0 or key[1] == "low":
                lams = (1.0, 0.5, 0.25, 0.125, 1 / 16, 1 / 32)     # approach: damping may be needed
            else:
                lams = (1.0, 0.25)                                 # near the floor: refresh J instead of halving
            for lam in lams:
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
                    if self.verbose >= 2:
                        print(f"   level {l} ({key[1]}): stop at |g| {np.linalg.norm(g):.2e} (noise floor)", flush=True)
                    self.record_floor(key, l, g)
                    break
                self.J[key] = self.fdjac(l, x, g)
                self.fresh[key] = True
                continue
            slow = np.linalg.norm(gt) > 0.3 * np.linalg.norm(g)
            floor = self.fresh[key] and np.linalg.norm(gt) > 0.5 * np.linalg.norm(g)
            x, r, g = xt, rt, gt
            if self.at_floor(key, l, g):
                break
            if floor:                          # a fresh Jacobian no longer halves |g|: noise floor
                self.record_floor(key, l, g)
                break
            self.fresh[key] = False
            if slow:
                self.J[key] = self.fdjac(l, x, g)
                self.fresh[key] = True
            if self.verbose and (l == self.top or self.verbose > 1):
                print(f"level {l} it {it}: |r_{l}| {np.linalg.norm(g):.3e}  lam {lam:.3g}  x {np.round(x, 9).tolist()}  "
                      f"[{time.time() - T0:.0f} s, {NEV[0]} residuals]", flush=True)
        return x, r


def _worker(args):
    M, r1, l, x, mode, Jc, K, st = args
    n0 = NEV[0]
    h = Hier(M, r1, top=99, verbose=0)
    h.mode = mode
    h.J = Jc                                   # warm chord Jacobians from the main process
    h.fresh = {k: False for k in Jc}
    h.K = K
    (h.predict, h.xref, h.xorder, h.count, h.tol, h.ltop, h.fasttrial, h.floorstop, h.floor) = st
    x, r = h.solve_from(l, x)
    return x, r, NEV[0] - n0


def report(x, r, M, tag):
    Fh = I.Fh_from(x, M)
    w = x[1] * I.W0
    print(f"{tag} M {M}: A {x[0]:.9f}  w/w_inf {x[1]:.9f}  (w {w:.5f}, Delta {2 * np.pi / w:.5f})  <F^2> {I.F2mean(Fh):.4f}  "
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
    h = Hier(M, r1, top=top, verbose=int(opts.get("--verbose", 1)))
    h.predict = "--predict" in opts
    h.tol = float(opts.get("--tol", 0.0))
    h.fasttrial = "--fasttrial" in opts
    h.floorstop = "--floor" in opts
    h.schur = "--schur" in opts or h.predict
    low, pol = h.radii(x)
    print("level radii (approach -> polish): " + ", ".join(f"r_{nu} {low[nu]:.4f} -> {pol[nu]:.4f}" for nu in low), flush=True)
    r0 = h.res(x)
    print("start residuals: " + "  ".join(f"|r_{nu}| {np.hypot(r0[nu - 1 if nu > 1 else 0], r0[nu if nu > 1 else 1]):.2e}"
                                         for nu in range(1, M + 1, 2)), flush=True)
    x, r = h.solve_from(top, x)
    report(x, r, M, "FIXED-A" if top == 3 else "ZERO")
    if h.K is not None:
        rr = h.res_raw(x)
        print(f"      (uncorrected r_1 {rr[0]:+.4e}{rr[1]:+.4e}i)", flush=True)


if __name__ == "__main__":
    main()

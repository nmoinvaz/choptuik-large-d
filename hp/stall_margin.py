#!/usr/bin/env python3
"""Stall detector for the continuation: Newton's margins along the Jacobian's weakest directions (README, "Why the
steps collapse"; along one direction the model is F(t) = F0 - sigma t + c t^2 / 2).

At a state z, with J factored there (s.newton), for the k weakest right singular directions v_k (Rayleigh-Ritz on
driver.weak_subspace, as driver.junk_subspace), sigma_k = 1 / |J^-T v_k| and u_k = sigma_k J^-T v_k, the left
direction (J v_k / sigma_k would carry v_k's rounding in the strong directions times sigma_max / sigma_k):
  c_k      the curvature along v_k, |u_k . (F(z + h v_k) + F(z - h v_k) - 2 F(z))| / h^2, at the h of a ladder
           (1e-9 .. 1e-3) where c(h) and c(2h) agree best; c_dev is that relative difference
  eps_k    the residual's rounding along u_k, |u_k . (F(z + d) - F(z) - J d)| for random d of relative size 1e-32
  floor_k  c_k eps_k^2 / (2 sigma_k^2): what a Newton step leaves along u_k when its error eps_k / sigma_k meets the
           curvature; the solve cannot go below the largest floor_k, so it stalls once that exceeds the tolerance
  F0_k     |u_k . F(z)|, taken as sigma_k |v_k . dz| with the Newton step dz = -J^-1 F(z) (equal for an exact
           singular pair, and free of F's strong components), and kappa_k = c_k F0_k / sigma_k^2: in the model a
           Newton step lowers the residual along
           v_k only while kappa_k < 2 (the basin F0 < 2 sigma^2 / c)
  theta_k  the measured contraction along v_k, |v_k . J^-1 F(z + dz)| / |v_k . dz|: kappa_k / 2 in the model,
           with what the other directions' second-order terms leak in
  harm_k   the odd p1 harmonic with the largest share of v_k (band_coeffs), and p1_k, the share of |v_k| in p1
and for the state, sigma_max = |J|_2 (SVD of the solver's double-double Jacobian, s.jacobian, in f64) and
u_kappa = u sigma_max / sigma_min with u = 2^-104, double-double's unit roundoff: a backward-stable solve's step has
relative error about u_kappa along the weakest direction, so Newton (an inexact Newton method with that forcing
term) can no longer converge once u_kappa nears 1.
At a converged point the floor is the quantity; at a step's seed (a predictor's state) kappa and theta are.

    python3 stall_margin.py --selftest [--mutate=half_curv|no_proj]
    python3 stall_margin.py OUT.jsonl STATE.npz ... [--kfreeze 13] [--qdh 1] [--pin K] [--k 4] [--threads N]
    python3 stall_margin.py OUT.jsonl --attempts ATTEMPTS.jsonl --states_dir DIR --tag up70 [--D LO,HI] [--every N]

STATE.npz: driver.save files (zh, zl, meta with D, M, nodes and the run's kt, kc, ks, which the solver is built
with). --kfreeze: default 13 from D 35.35 on (as the runs), none below. --pin K: the system T73's runs solve, p1's
harmonics in (K, kfreeze] held at the state's own values (pinning.PinnedSolver) on a kfreeze = K solver. OUT must be
in a raw-data folder (raw_out.py). Each state gives one record in OUT and one line on stdout.
--attempts: a scan's attempted steps (scan_hp --attempts) in D_from LO..HI, every Nth (all failed ones near
the end are wanted, so N = 1 there): the step's seed rebuilt as scan_hp --predictor lagrange_dd (default, T48's)
or lagrange does it, through the 4 saved states <tag>_D<D>.npz up to D_from, and the margins at that
seed with the solver at D_to in the newest state's layout; the record has the attempt's outcome and its logged
F_seed beside the rebuilt one (they agree only if the run used no other seed adjustment, e.g. --junk_hold).

Control (--selftest): two planted problems F(z) = A (z - z*) + (b + c ((z - z*).v)^2 / 2) u with known sigma, c and
b, seeded off z* along strong directions, kappa = c b / sigma^2 = 0.3 and 2.5; sigma, c, F0, kappa and theta
(= kappa / 2) must come out to 1e-6, and floor within a factor 30 of c eta^2 / (2 sigma^2) for a planted rounding
eta. --mutate=half_curv halves the curvature estimate, no_proj takes F0 as |F| instead of its u component; each must
make the selftest fail. Exit 1 on failure.
"""
import argparse
import hashlib
import json
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

from hpkernel.driver import junk_subspace, load, make_solver, predict, zadd  # noqa: E402
from pinning import band_coeffs, band_field, dd_add, predict_dd  # noqa: E402
from raw_out import FLAG, HELP, require_raw  # noqa: E402

HS = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]


def f64(v):
    return np.asarray(v[0]) + np.asarray(v[1])


def dd_diff(a, b):
    """a - b for (hi, lo) pairs, in double-double, as f64"""
    h, l_ = dd_add(np.asarray(a[0]), np.asarray(a[1]), -np.asarray(b[0]), -np.asarray(b[1]))
    return np.asarray(h) + np.asarray(l_)


def margins(s, zh, zl, k=4, M=0, mutate=""):
    """the quantities of the docstring at (zh, zl) for solver s; returns a list of dicts, weakest first"""
    L = len(zh)
    dh, dl, Fh, Fl, _, _ = s.newton(zh, zl)              # J and its LU at z, the Newton step, F(z)
    F0 = (np.asarray(Fh), np.asarray(Fl))
    X, sig_rr, _ = junk_subspace(s, k, delta_max=np.inf)   # the k weakest right singular directions (LU solves)
    Y = np.column_stack([f64(s.lu_solve_t(np.ascontiguousarray(X[:, j]), np.zeros(L))) for j in range(X.shape[1])])
    th, tl = zadd(s, zh, zl, np.asarray(dh), np.asarray(dl))
    F1 = s.residual(th, tl)
    eh, el = s.lu_solve(np.ascontiguousarray(-np.asarray(F1[0])), np.ascontiguousarray(-np.asarray(F1[1])))
    dz, dz1 = np.asarray(dh) + np.asarray(dl), f64((eh, el))     # Newton step and the simplified one after it
    rng = np.random.default_rng(1)
    d = 1e-32 * np.abs(np.asarray(zh)) * rng.choice([-1.0, 1.0], L)
    Fd = s.residual(*zadd(s, zh, zl, d, np.zeros(L)))
    noise = dd_diff(Fd, F0) - f64(s.jvp(zh, zl, d, np.zeros(L)))  # (jvp after every LU solve)
    out = []
    for j in range(X.shape[1]):
        v = X[:, j]
        sig = 1.0 / float(np.linalg.norm(Y[:, j]))       # J^-T v = u / sigma
        u = sig * Y[:, j]
        best = None
        for h in HS:
            cs = []
            for hh in (h, 2 * h):
                Fp = s.residual(*zadd(s, zh, zl, hh * v, np.zeros(L)))
                Fm = s.residual(*zadd(s, zh, zl, -hh * v, np.zeros(L)))
                second = dd_diff(Fp, F0) + dd_diff(Fm, F0)
                cs.append(abs(float(u @ second)) / hh ** 2 / (2.0 if mutate == "half_curv" else 1.0))
            dev = abs(cs[0] - cs[1]) / max(cs[0], cs[1], 1e-300)
            if best is None or dev < best[1]:
                best = (cs[0], dev, h)
        c, c_dev, h_used = best
        eps = abs(float(u @ noise))
        F0k = float(np.linalg.norm(f64(F0))) if mutate == "no_proj" else sig * abs(float(v @ dz))
        rec = dict(sigma=sig, sigma_rr=float(sig_rr[j]) if j < len(sig_rr) else None, c=c, c_dev=c_dev, h=h_used,
                   eps=eps, floor=c * eps ** 2 / (2 * sig ** 2), F0=F0k, kappa=c * F0k / sig ** 2,
                   theta=abs(float(v @ dz1)) / max(abs(float(v @ dz)), 1e-300))
        if M:
            ks = list(range(1, M, 2))
            amp = np.hypot(*np.asarray(band_coeffs(v[:M], np.zeros(M), M, ks), dtype=float).T)
            rec.update(harm=ks[int(np.argmax(amp))], p1=float(np.linalg.norm(v[:M]) / np.linalg.norm(v)))
        out.append(rec)
    out.sort(key=lambda r: r["sigma"])
    smax = None
    if hasattr(s, "jacobian"):
        smax = float(np.linalg.norm(f64(s.jacobian(zh, zl)), 2))
    for r in out:
        r.update(sigma_max=smax, u_kappa=None if smax is None else 2.0 ** -104 * smax / r["sigma"])
    return out, float(np.linalg.norm(f64(F0)))


class Planted:
    """F(z) = A (z - z*) + (b + c ((z - z*).v)^2 / 2) u, A = U diag(sigmas) V^T with v, u its weakest pair. The
    residual is (F(z_hi), J z_lo + rounding): the low word carries the state's low part to first order and a planted
    rounding, eta times a fixed pseudo-random vector per distinct state (hash of its bytes)"""

    def __init__(self, sigmas, c, b, eta, seed=3):
        rng = np.random.default_rng(seed)
        n = len(sigmas)
        self.U = np.linalg.qr(rng.standard_normal((n, n)))[0]
        self.V = np.linalg.qr(rng.standard_normal((n, n)))[0]
        self.A = self.U @ np.diag(sigmas) @ self.V.T
        self.u, self.v = self.U[:, -1], self.V[:, -1]
        self.zs = rng.standard_normal(n)
        self.c, self.b, self.eta, self.size = c, b, eta, n
        self.J = None

    def F(self, z):
        t = (z - self.zs) @ self.v
        return self.A @ (z - self.zs) + (self.b + 0.5 * self.c * t * t) * self.u

    def jac(self, z):
        return self.A + self.c * ((z - self.zs) @ self.v) * np.outer(self.u, self.v)

    def residual(self, zh, zl):
        zh, zl = np.asarray(zh), np.asarray(zl)
        g = np.random.default_rng(int.from_bytes(hashlib.sha256(zh.tobytes() + zl.tobytes()).digest()[:8], "little"))
        return self.F(zh), self.jac(zh) @ zl + self.eta * g.standard_normal(self.size)

    def newton(self, zh, zl):
        self.J = self.jac(np.asarray(zh))
        Fh, Fl = self.residual(zh, zl)
        return -np.linalg.solve(self.J, Fh + Fl), np.zeros(self.size), Fh, Fl, 0.0, 0.0

    def lu_solve(self, rh, rl):
        return np.linalg.solve(self.J, np.asarray(rh) + np.asarray(rl)), np.zeros(self.size)

    def lu_solve_t(self, rh, rl):
        return np.linalg.solve(self.J.T, np.asarray(rh) + np.asarray(rl)), np.zeros(self.size)

    def jacobian(self, zh, zl):
        return self.jac(np.asarray(zh)), np.zeros((self.size, self.size))

    def jvp(self, zh, zl, uh, ul):
        return self.jac(np.asarray(zh) + np.asarray(zl)) @ (np.asarray(uh) + np.asarray(ul)), np.zeros(self.size)


def selftest(mutate=""):
    ok = True
    sig_w, c, eta = 1e-6, 1.0, 1e-20
    for kappa in (0.3, 2.5):
        p = Planted([3.0, 1.0, 0.4, 0.1, 0.02, sig_w], c, kappa * sig_w ** 2 / c, eta)
        z0 = p.zs + 1e-3 * p.V[:, 0] - 2e-3 * p.V[:, 1]        # off z* in strong directions, t = 0 along v
        r = margins(p, z0, np.zeros(p.size), k=2, mutate=mutate)[0][0]
        want = dict(sigma=sig_w, c=c, F0=p.b, kappa=kappa, theta=kappa / 2)
        for key, w in want.items():
            good = abs(r[key] - w) <= 1e-6 * abs(w)
            print(f"  kappa {kappa}: {key} {r[key]:.6g} (planted {w:.6g}): {'PASS' if good else 'FIRE'}")
            ok &= good
        good = abs(r["sigma_max"] - 3.0) <= 1e-6 * 3.0
        print(f"  kappa {kappa}: sigma_max {r['sigma_max']:.6g} (planted 3): {'PASS' if good else 'FIRE'}")
        ok &= good
        fl = c * eta ** 2 / (2 * sig_w ** 2)
        good = fl / 30 <= r["floor"] <= 30 * fl
        print(f"  kappa {kappa}: floor {r['floor']:.2e} (c eta^2 / 2 sigma^2 = {fl:.2e}): {'PASS' if good else 'FIRE'}")
        ok &= good
    return ok


def seeds_of(a):
    """(label, attempt) jobs for --attempts: each attempt's seed rebuilt from the saved states (see the docstring)"""
    import glob
    lo, hi = (float(v) for v in a.D.split(","))
    states = sorted((load(f)[2]["D"], f) for f in glob.glob(os.path.join(a.states_dir, f"{a.tag}_D*.npz")))
    atts = [r for r in (json.loads(x) for x in open(a.attempts) if x.strip())
            if r.get("D_from") is not None and lo <= r["D_from"] <= hi]
    jobs = []
    for i, r in enumerate(atts):
        if i % a.every and r["ok"]:
            continue                                   # failed attempts always, successes every Nth
        hist = [st for st in states if st[0] <= r["D_from"] + 1e-9][-4:]
        if len(hist) < 4 or abs(hist[-1][0] - r["D_from"]) > 1e-9:
            continue                                   # the state at D_from was not saved
        loaded = [load(f) for _, f in hist]
        if a.predictor == "lagrange_dd":
            zp, zpl = (np.asarray(v) for v in predict_dd([(m["D"], h, l_) for h, l_, m in loaded], r["D_to"], 4))
        else:
            zp = predict([(m["D"], h + l_) for h, l_, m in loaded], r["D_to"])
            zpl = np.zeros_like(zp)
        meta = dict(loaded[-1][2], D=r["D_to"])
        att = {k: r[k] for k in ("D_from", "D_to", "ok", "F_seed", "F_last")}
        att["seed"] = (zp, zpl, meta)
        jobs.append((f"seed {r['D_from']:.6f} -> {r['D_to']:.6f}", att))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?")
    ap.add_argument("states", nargs="*")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mutate", default="", choices=["", "half_curv", "no_proj"])
    ap.add_argument("--kfreeze", type=int, default=None)
    ap.add_argument("--qdh", type=int, default=0)
    ap.add_argument("--pin", type=int, default=None)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--tol_step", type=float, default=1e-32)
    ap.add_argument("--hmax", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--attempts", default="")
    ap.add_argument("--states_dir", default="")
    ap.add_argument("--tag", default="up70")
    ap.add_argument("--D", default="0,1e9")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--predictor", default="lagrange_dd", choices=["lagrange_dd", "lagrange"])
    ap.add_argument(FLAG, action="store_true", help=HELP)
    a = ap.parse_args()
    if a.selftest:
        good = selftest(a.mutate)
        print(f"selftest: {'PASS' if good else 'FAIL'}")
        sys.exit(0 if good else 1)
    if not a.out or not (a.states or a.attempts):
        ap.error("OUT.jsonl and STATE.npz files or --attempts are needed")
    require_raw(a.out, allow=a.allow_unraw)
    print(f"{'D':>10} {'|F|':>8} {'sigma':>8} {'harm':>4} {'smax':>8} {'u_kappa':>8} {'c':>8} {'floor':>8} "
          f"{'kappa':>8} {'theta':>8}")
    jobs = [(path, None) for path in a.states]
    if a.attempts:
        jobs += seeds_of(a)
    for path, att in jobs:
        if att is None:
            zh, zl, meta = load(path)
        else:
            zh, zl, meta = att.pop("seed")
        D, M, nodes = meta["D"], meta["M"], list(meta["nodes"])
        kf = a.kfreeze if a.kfreeze is not None else (13 if D >= 35.35 else None)
        kw = dict(nodes=nodes, kt=meta.get("kt", 36), kc=meta.get("kc", 40), ks=meta.get("ks", 120), threads=a.threads)
        s = make_solver(D, M, **kw, kfreeze=a.pin if a.pin is not None else kf)
        s.set_segs(s.plan(zh, zl, a.tol_step, a.hmax))
        if a.qdh:
            from hpkernel.driver import QdhSolver
            s = QdhSolver(s)
        if a.pin is not None:
            from pinning import PinnedSolver
            ks = list(range(a.pin + 2, kf + 1, 2))
            s = PinnedSolver(s, M, *band_field(band_coeffs(zh[:M], zl[:M], M, ks), ks, M))
        rows, nF = margins(s, zh, zl, k=a.k, M=M)
        rec = dict(state=path, D=D, F=nF, kfreeze=kf, pin=a.pin, qdh=a.qdh, kt=kw["kt"], kc=kw["kc"], ks=kw["ks"],
                   dirs=rows, floor_max=max(r["floor"] for r in rows), kappa_max=max(r["kappa"] for r in rows),
                   theta_max=max(r["theta"] for r in rows), u_kappa=rows[0]["u_kappa"], attempt=att)
        with open(a.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        w = rows[0]
        print(f"{D:10.4f} {nF:8.1e} {w['sigma']:8.1e} {w.get('harm', '-'):>4} {w['sigma_max'] or 0:8.1e} "
              f"{w['u_kappa'] or 0:8.1e} {w['c']:8.1e} "
              f"{rec['floor_max']:8.1e} {rec['kappa_max']:8.1e} {rec['theta_max']:8.1e}"
              + ("" if att is None else f"  seed from {att['D_from']:.6f}, {'ok' if att['ok'] else 'FAILED'}, "
                 f"logged F_seed {att['F_seed']:.2e}"), flush=True)


if __name__ == "__main__":
    main()

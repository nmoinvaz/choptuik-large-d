"""Form S with the Moebius horizon variable at one D in double-double, the weak harmonics held and then released one at
a time (#1090; the method of Track A's release_ls in hp/form_s.py, #1088).

Coordinates. p1 on the M collocation points is written in odd harmonics, p1 = T h (k = 1, 3, ..., M - 1, a cos and a
sin part each). The rows are rewritten the same way for p: R_p = (R_U + x* R_V)/2 (the mismatch of p = (U + x V)/2 at
x*) in harmonics, r_p = T^-1 R_p, beside R_V, R_c and the phase row. Both maps are invertible, so the system is the same.

Held system. For every held harmonic k its two components of h are fixed and harmonic k of r_p is dropped (hp's kfreeze
drops the horizon block's p-matching rows of the held harmonics the same way). The held system is square and has none of
the weakest directions; it is solved by Newton in double-double with the exact forward-mode Jacobian (mobius_sdd.py),
reused as a chord while it contracts.

Release of harmonic k (the lowest held one first). At the held solution, the Newton step dz of the full system
(every row, every unknown) is formed; its harmonic-k part moves only the held value, h_k <- h_k + lam dz_k; the held
system is re-solved; lam is halved while harmonic k's dropped rows do not fall. When they are below --tol_k, k is
freed and the next one released.

Usage: python3 release_dd.py START.json --D=200 --M=10 --hold=5,7,9 --out=FILE.json [--a=1] [--kb=150]
                             [--tol=1e-24] [--tol_k=1e-24] [--outer=6] [--log=FILE.log]
  START.json: a form-S record (solve_cont.py, solve_dd_cont.py or Track B's); resampled to M; rescaled to D as in
  solve_cont.py when its D differs.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ddnp as d  # noqa: E402
from ddnp import DD  # noqa: E402
import mobius_s as ms  # noqa: E402
import mobius_sdd as msd  # noqa: E402

opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
start = [a for a in sys.argv[1:] if not a.startswith("--")][0]
D = float(opts["D"])
M = int(opts.get("M", 10))
hold = [int(v) for v in opts.get("hold", "").split(",") if v]
a_m = float(opts.get("a", 1.0))
kb = int(opts.get("kb", 150))
tol = float(opts.get("tol", 1e-24))
tol_k = float(opts.get("tol_k", 1e-24))
outer = int(opts.get("outer", 6))
logf = open(opts["log"], "a") if "log" in opts else None
t_start = time.time()


def say(s):
    s = "[%6.0f s] %s" % (time.time() - t_start, s)
    print(s, flush=True)
    if logf:
        logf.write(s + "\n")
        logf.flush()


# ---------------------------------------------------------------- start
js = json.load(open(start))
D0, M0 = js["D"], js["M"]
get = lambda k: np.array(js[k]) + np.array(js.get(k + "_lo", [0.0] * len(js[k])))  # noqa: E731
if M0 == M:
    yh = np.concatenate([js["p1"], js["c0"], js["Q0"], [js["Delta"]]])
    yl = np.concatenate([js.get("p1_lo", [0.0] * M), js.get("c0_lo", [0.0] * M), js.get("Q0_lo", [0.0] * M),
                         [js.get("Delta_lo", 0.0)]])
else:
    yh = np.concatenate([ms.resample_odd(get("p1"), M), ms.resample_even(get("c0"), M), ms.resample_odd(get("Q0"), M),
                         [js["Delta"] + js.get("Delta_lo", 0.0)]])
    yl = np.zeros_like(yh)
if abs(D - D0) > 1e-12:
    n, n0 = D - 2, D0 - 2
    s = np.sqrt(n0 / n)
    yh = yh + yl
    yh[:M] *= s
    yh[2 * M:3 * M] *= s
    yh[M:2 * M] = 1 + (yh[M:2 * M] - 1) * n0 / n
    yl = np.zeros_like(yh)

# ---------------------------------------------------------------- harmonic coordinates
th = np.pi * np.arange(M) / M
ks_odd = list(range(1, M, 2))
cols = []
lab = []
for k in ks_odd:
    cols.append(np.cos(k * th))
    lab.append((k, "c"))
    cols.append(np.sin(k * th))
    lab.append((k, "s"))
T = np.array(cols).T  # p1 = T h
Tinv = np.linalg.inv(T)
xs = 1 / np.sqrt(2.0)
L = 3 * M + 1
# y = B z, B = blockdiag(T, I_M, I_M, 1); F_z = A F_y, A = [[Tinv/2, Tinv xs/2, 0, 0], [0, I, 0, 0], [0, 0, I, 0], [0, 0, 0, 1]]
B = np.eye(L)
B[:M, :M] = T
A = np.eye(L)
A[:M, :M] = Tinv / 2
A[:M, M:2 * M] = Tinv * xs / 2
Binv = np.linalg.inv(B)


def to_z(y):
    return d.matvec(DD(Binv), y)


def to_y(z):
    return d.matvec(DD(B), z)


f = msd.FormSDD(D, M, a=a_m, kb=kb)


def Fz(z):
    return d.matvec(DD(A), f.residual(to_y(z)))


def Jz(z):
    """the exact Jacobian in z, A J_y B, in double-double"""
    Jy = f.jacobian(to_y(z))
    return _mm(_mm(DD(A), Jy), DD(B))


def _mm(X, Y):
    """X @ Y in double-double, column by column with matvec"""
    out = d.zeros((X.shape[0], Y.shape[1]))
    for j in range(Y.shape[1]):
        col = d.matvec(X, DD(Y.h[:, j], Y.l[:, j]))
        out.h[:, j] = col.h
        out.l[:, j] = col.l
    return out


def held_index(held):
    """rows and columns of the held system: drop h's components and r_p's rows of the held harmonics"""
    drop = [i for i, (k, _) in enumerate(lab) if k in held]
    keep = [i for i in range(L) if i not in drop]
    return keep, drop


def sub(J, rows, cols_):
    return DD(J.h[np.ix_(rows, cols_)], J.l[np.ix_(rows, cols_)])


def solve_held(z, held, J=None, iters=8, tolr=None):
    """Newton (chord with J reused while it contracts by 4 per step) on the held system; returns z, J, |F_keep|"""
    keep, drop = held_index(held)
    tolr = tol if tolr is None else tolr
    if J is None:
        J = Jz(z)
    F = Fz(z)
    r = float(np.max(np.abs(F.f()[keep])))
    for it in range(iters):
        if r < tolr:
            break
        Jk = sub(J, keep, keep)
        dz = d.solve(Jk, DD(-F.h[keep], -F.l[keep]))
        lam = 1.0
        ok = False
        for _ in range(12):
            zt = z.copy()
            zt.h[keep] = (DD(z.h[keep], z.l[keep]) + dz * lam).h
            zt.l[keep] = (DD(z.h[keep], z.l[keep]) + dz * lam).l
            with np.errstate(all="ignore"):
                Ft = Fz(zt)
            rt = float(np.max(np.abs(Ft.f()[keep])))
            if np.isfinite(rt) and rt < r:
                ok = True
                break
            lam *= 0.5
        if not ok:
            J = Jz(z)  # refresh once, then give up
            Jk = sub(J, keep, keep)
            dz = d.solve(Jk, DD(-F.h[keep], -F.l[keep]))
            zt = z.copy()
            zt.h[keep] = (DD(z.h[keep], z.l[keep]) + dz).h
            zt.l[keep] = (DD(z.h[keep], z.l[keep]) + dz).l
            Ft = Fz(zt)
            rt = float(np.max(np.abs(Ft.f()[keep])))
            if not (np.isfinite(rt) and rt < r):
                break
        if rt > 0.25 * r:
            J = Jz(zt)  # slow contraction: a fresh Jacobian for the next step
        z, F, r = zt, Ft, rt
    return z, J, r, F


def dropped_norm(F, held_k):
    rows = [i for i, (k, _) in enumerate(lab) if k == held_k]
    return float(np.max(np.abs(F.f()[rows])))


# ---------------------------------------------------------------- run
z = to_z(DD(yh, yl))
held = sorted(hold)
say("release_dd: D %g M %d a %g hold %s start %s (D %g, M %d)" % (D, M, a_m, held, os.path.basename(start), D0, M0))
z, J, r, F = solve_held(z, held)
say("held system: max|F_keep| %.2e  dropped rows: %s  Delta %.16f" % (
    r, " ".join("h%d %.1e" % (k, dropped_norm(F, k)) for k in held), float(to_y(z).f()[3 * M])))
history = [dict(stage="held", held=list(held), r=r, Delta=float(to_y(z).f()[3 * M]))]
for k in list(held):
    rows_k = [i for i, (kk, _) in enumerate(lab) if kk == k]
    gk = dropped_norm(F, k)
    for it in range(outer):
        if gk < tol_k:
            break
        Jf = Jz(z)
        dz = d.solve(Jf, -F)  # the full system's Newton step
        lam = 1.0
        acc = False
        for _ in range(14):
            zt = z.copy()
            for i in rows_k:
                v = DD(z.h[i], z.l[i]) + DD(dz.h[i], dz.l[i]) * lam
                zt.h[i], zt.l[i] = v.h, v.l
            with np.errstate(all="ignore"):
                zt, Jt, rt, Ft = solve_held(zt, held, J=Jf)
            gt = dropped_norm(Ft, k) if np.all(np.isfinite(Ft.f())) else np.inf
            if np.isfinite(gt) and gt < gk and rt < max(1e3 * tol, 10 * r):
                acc = True
                break
            lam *= 0.5
        if not acc:
            say("  release h%d: no step lowers its rows (|g| %.2e); stop" % (k, gk))
            break
        z, J, r, F, gk = zt, Jt, rt, Ft, gt
        say("  release h%d, step %d: lambda %g  its rows %.2e  held residual %.2e  Delta %.16f" % (
            k, it, lam, gk, r, float(to_y(z).f()[3 * M])))
        history.append(dict(stage="release", k=k, step=it, lam=lam, g=gk, r=r, Delta=float(to_y(z).f()[3 * M])))
    if gk < tol_k:
        held.remove(k)
        say("h%d released; still held: %s" % (k, held))
    else:
        say("h%d not released (|g| %.2e); stop" % (k, gk))
        break
y = to_y(z)
Fy = f.residual(y)
nF = float(np.max(np.abs(Fy.f())))
say("end: max|F| (full system) %.2e  Delta %.16f (%+.1e)  held %s" % (nF, float(y.h[3 * M]), float(y.l[3 * M]), held))
rec = dict(D=D, M=M, a=a_m, ks=f.ks, kb=kb, maxF=nF, Delta=float(y.h[3 * M]), Delta_lo=float(y.l[3 * M]), held=held,
           hold0=hold, history=history, seconds=time.time() - t_start, start=os.path.basename(start),
           p1=list(y.h[:M]), c0=list(y.h[M:2 * M]), Q0=list(y.h[2 * M:3 * M]),
           p1_lo=list(y.l[:M]), c0_lo=list(y.l[M:2 * M]), Q0_lo=list(y.l[2 * M:3 * M]))
json.dump(rec, open(opts["out"], "w"), indent=1)

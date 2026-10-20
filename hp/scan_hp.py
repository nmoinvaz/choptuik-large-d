#!/usr/bin/env python3
"""Continuation in D with the double-double solver.

    python3 scan_hp.py OUT.jsonl START.npz D_END [--dD 0.5] [--M 48] [--kt 36] [--kc 40] [--ks 100]
                       [--nodes 0.02,0.05,...] [--tag name] [--maxdD 1.0]

START.npz: a converged solution (hpkernel.driver.save format: zh, zl, meta with D, M, nodes). Every converged point
is appended to OUT.jsonl and saved as <tag>_D<D>.npz next to OUT. A --nodes / --M different from the start file's
re-nodes / resamples the start solution first (and re-converges it at its own D).
"""
import argparse
import json
import os
import time

import numpy as np

from hpkernel.driver import deflated_newton, delta_correction, hermite, lm_solve, load, make_solver, newton_solve, plow_odd, predict, renode, resample_z, save, tangent


def tails(s, zh, zl, nodes, M):
    """max over fields/lanes of |c_k| x^k for the last 3 orders: centre series at nodes[0], horizon series at
    u = 1 - nodes[-1]"""
    out = []
    for which, x in ((0, nodes[0]), (1, 1 - nodes[-1])):
        h, l = s.series(zh, zl, which)
        c = np.abs((h + l).reshape(-1, 4, M)).max(axis=(1, 2))
        k = np.arange(len(c))
        out.append(float((c * x ** k)[-3:].max()))
    return out


def sensitivity(s, M):
    """dDelta/dF (needs the LU of the last Newton step): max |y| per residual block"""
    L = s.size
    e = np.zeros(L)
    e[-1] = 1.0
    yh, yl = s.lu_solve_t(e, np.zeros(L))
    nb = (L - 1) // (4 * M)
    return [float(np.abs(yh[b * 4 * M:(b + 1) * 4 * M]).max()) for b in range(nb)]


def freeze_p1(zh, zl, a, M):
    """p1's harmonics above kfreeze set to zero (they are pinned there by the frozen formulation)"""
    if not a.kfreeze:
        return zh, zl
    lo, lo2 = plow_odd(M, a.kfreeze)
    zh = zh.copy()
    zl = zl.copy()
    zh[:M] = (lo + lo2) @ (zh[:M] + zl[:M])
    zl[:M] = 0.0
    return zh, zl


def solve_at(D, zh, zl, a, nodes, M, it_max):
    s = make_solver(D, M, nodes=nodes, kt=a.kt, kc=a.kc, ks=a.ks, threads=a.threads, kfreeze=a.kfreeze)
    s.set_segs(s.plan(zh, zl, a.tol_step, a.hmax))
    t0 = time.time()
    if a.corrector in ("dn", "dnp"):
        zh2, zl2, hist = deflated_newton(s, zh, zl, tol=a.tol, it_max=it_max, w=a.w, verbose=a.verbose,
                                         junk_only=a.junk_only, delta_tol=a.delta_tol)
    else:
        zh2, zl2, hist = newton_solve(s, zh, zl, tol=a.tol, it_max=it_max, verbose=a.verbose)
    if a.corrector == "dnp" and hist and hist[-1]["F"] < a.polish_from:
        # polish: plain Newton to the bottom (deflated Newton stops at the first iterate under tol, or stalls just
        # above it, and leaves that point's error in the predictor's history)
        zh3, zl3, h3 = newton_solve(s, zh2, zl2, tol=1e-40, it_max=a.polish_its, verbose=a.verbose)
        if h3 and h3[-1]["F"] < hist[-1]["F"]:
            zh2, zl2 = zh3, zl3
            hist = hist + [dict(r, polish=True) for r in h3]
    ok = bool(hist) and (hist[-1]["F"] < a.tol or bool(hist[-1].get("accepted")))
    if not ok and a.lm:
        print(f"   D={D}: Newton stopped at |F| {hist[-1]['F']:.1e}; Levenberg-Marquardt from the predictor", flush=True)
        zh2, zl2, h2 = lm_solve(s, zh, zl, tol=a.tol, it_max=a.lm, verbose=a.verbose, newton_after=1e-22)
        hist = hist + h2
        ok = hist[-1]["F"] < a.tol
    return s, zh2, zl2, hist, ok, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("start")
    ap.add_argument("D_end", type=float)
    ap.add_argument("--dD", type=float, default=0.5)
    ap.add_argument("--maxdD", type=float, default=1.0)
    ap.add_argument("--dD0", type=float, default=None, help="first step (default dD)")
    ap.add_argument("--history", default="", help="comma list of converged npz files at earlier D (same M), oldest first: "
                    "re-noded to the scan's layout and used for the predictor")
    ap.add_argument("--predictor", default="lagrange", choices=["lagrange", "hermite"],
                    help="lagrange: extrapolation through the last 4 converged points (default; the J-based tangent "
                    "is unreliable along near-null directions at larger D); hermite: exact tangent + cubic Hermite")
    ap.add_argument("--M", type=int, default=None)
    ap.add_argument("--kt", type=int, default=36)
    ap.add_argument("--kc", type=int, default=40)
    ap.add_argument("--ks", type=int, default=100)
    ap.add_argument("--nodes", default=None)
    ap.add_argument("--tag", default="hp")
    ap.add_argument("--tol", type=float, default=1e-27)
    ap.add_argument("--tol_step", type=float, default=1e-32)
    ap.add_argument("--hmax", type=float, default=0.05)
    ap.add_argument("--it_max", type=int, default=8)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--kfreeze", type=int, default=None, help="pin p1 harmonics above this to zero and drop the matching of "
                    "p's harmonics above it at the horizon block (removes the near-null centre directions)")
    ap.add_argument("--corrector", default="dn", choices=["dn", "dnp", "newton"],
                    help="dn: deflated Newton (default); dnp: deflated Newton, then plain Newton to the bottom")
    ap.add_argument("--polish_its", type=int, default=4, help="dnp: plain Newton steps after the deflated phase")
    ap.add_argument("--polish_from", type=float, default=1e-12, help="dnp: polish only from |F| below this")
    ap.add_argument("--w", type=int, default=24, help="weak directions frozen in the deflated phase")
    ap.add_argument("--delta_tol", type=float, default=1e-14, help="junk_only: accept a stalled step if the linearised "
                    "Delta correction is below this")
    ap.add_argument("--junk_hold", type=int, default=None, help="predictor: carry p1 harmonics above this over from the "
                    "last point instead of extrapolating them")
    ap.add_argument("--junk_trend", type=int, default=0, help="with junk_hold: put the held harmonics on a least-squares "
                    "line through the last N points instead of the last value")
    ap.add_argument("--junk_only", action="store_true", help="freeze only weak directions without Delta content, and keep "
                    "them frozen (they stay on the predictor's smooth trend)")
    ap.add_argument("--lm", type=int, default=15, help="Levenberg-Marquardt iterations after a failed Newton (0: off)")
    ap.add_argument("--Dlist", default="", help="explicit D values to visit in order (comma list or @file); "
                    "D_end is then ignored")
    ap.add_argument("--schedule", default="", help="node changes during the scan: 'D1=x,x,...;D2=x,x,...' "
                    "(at the first converged point with D >= D_i the solution is re-noded and re-converged)")
    a = ap.parse_args()
    zh, zl, meta = load(a.start)
    D, M0, nodes0 = meta["D"], meta["M"], list(meta["nodes"])
    M = a.M or M0
    nodes = [float(v) for v in a.nodes.split(",")] if a.nodes else nodes0
    if M != M0:
        zh, zl = resample_z(zh, M0, M, len(nodes0))
    if nodes != nodes0:
        s0 = make_solver(D, M, nodes=nodes0, kt=a.kt, kc=a.kc, ks=a.ks, threads=a.threads)
        zh, zl = renode(s0, zh, zl, nodes0, nodes, M)
    zh, zl = freeze_p1(zh, zl, a, M)
    s, zh, zl, hist, ok, dt = solve_at(D, zh, zl, a, nodes, M, a.it_max)
    print(f"(re-)converged start D={D} M={M} nodes={nodes}: Delta {zh[-1]:.16f} ok {ok} ({dt:.0f}s)", flush=True)
    if not ok:
        raise SystemExit("start did not converge")
    sched = []
    for item in filter(None, a.schedule.split(";")):
        d, xs = item.split("=")
        sched.append((float(d), [float(v) for v in xs.split(",")]))
    sched.sort()
    outdir = os.path.dirname(os.path.abspath(a.out))
    s_of = lambda Dx: make_solver(Dx, M, nodes=nodes, kt=a.kt, kc=a.kc, ks=a.ks, threads=a.threads, kfreeze=a.kfreeze)
    hist_z = []
    for f in filter(None, a.history.split(",")):
        hh, hl, hm = load(f)
        if hm["M"] != M:
            raise SystemExit(f"{f}: M {hm['M']} != {M}")
        if list(hm["nodes"]) != nodes:
            s0 = make_solver(hm["D"], M, nodes=list(hm["nodes"]), kt=a.kt, kc=a.kc, ks=a.ks, threads=a.threads)
            hh, hl = renode(s0, hh, hl, list(hm["nodes"]), nodes, M)
        hh, hl = freeze_p1(hh, hl, a, M)
        sh, hh, hl, hst, okh, _ = solve_at(hm["D"], hh, hl, a, nodes, M, a.it_max)
        print(f"history point D={hm['D']}: ok {okh}, Delta {hh[-1]:.16f}", flush=True)
        if okh:
            hist_z.append((hm["D"], hh + hl, None))
    t0_ = tangent(s_of, s, D, zh, zl) if a.predictor == "hermite" or not hist_z else None
    hist_z.append((D, zh + zl, t0_))
    dD = a.dD0 or a.dD
    targets = []
    if a.Dlist:
        txt = open(a.Dlist[1:]).read() if a.Dlist.startswith("@") else a.Dlist
        targets = [float(v) for v in txt.replace("\n", ",").split(",") if v.strip()]
        a.D_end = targets[-1]
    direction = 1.0 if a.D_end > D else -1.0
    while (a.D_end - D) * direction > 1e-12:
        while targets and (targets[0] - D) * direction <= 1e-12:
            targets.pop(0)
        goal = targets[0] if targets else a.D_end
        Dn = D + direction * min(dD, abs(goal - D))
        Dn = round(Dn, 6)
        if a.predictor == "lagrange" and len(hist_z) > 1:
            zp = predict([(h_[0], h_[1]) for h_ in hist_z], Dn)
            if a.junk_hold:
                # p1 harmonics above junk_hold: carried over from the last point (they are frozen junk; extrapolating
                # them compounds their noise from point to point)
                lo, lo2 = plow_odd(M, a.junk_hold)
                P = lo + lo2
                zlast = hist_z[-1][1]
                zp = zp.copy()
                held = zlast[:M] - P @ zlast[:M]
                if a.junk_trend and len(hist_z) >= 3:
                    # least-squares line through the held harmonics of the last junk_trend points: their values
                    # jitter from point to point at |F| / sigma, and a straight fit averages that out
                    pts_ = hist_z[-a.junk_trend:]
                    Dh_ = np.array([h_[0] for h_ in pts_])
                    Y = np.array([h_[1][:M] - P @ h_[1][:M] for h_ in pts_])
                    A_ = np.vander(Dh_ - Dh_[-1], 2)
                    coef, *_ = np.linalg.lstsq(A_, Y, rcond=None)
                    held = coef[0] * (Dn - Dh_[-1]) + coef[1]
                zp[:M] = P @ zp[:M] + held
        elif len(hist_z) > 1 and hist_z[-2][2] is not None:
            (D0, z0, t0), (D1, z1, t1) = hist_z[-2], hist_z[-1]
            zp = hermite(D0, z0, t0, D1, z1, t1, Dn)
        else:
            D1, z1, t1 = hist_z[-1]
            zp = z1 + (Dn - D1) * t1
        s, zh2, zl2, hist, ok, dt = solve_at(Dn, zp, np.zeros_like(zp), a, nodes, M, a.it_max)
        if not ok:
            print(f"D={Dn}: no convergence (|F| {hist[-1]['F'] if hist else float('nan'):.1e}); dD {dD} -> {dD / 2}", flush=True)
            dD /= 2
            if dD < 1e-3:
                raise SystemExit("step too small")
            continue
        D, zh, zl = Dn, zh2, zl2
        hist_z.append((D, zh + zl, tangent(s_of, s, D, zh, zl) if a.predictor == "hermite" else None))
        # the stored LU is from the last Newton call, made at the converged z
        sens = sensitivity(s, M)
        tc, th = tails(s, zh, zl, nodes, M)
        rec = dict(D=D, Delta=float(zh[-1]), Delta_lo=float(zl[-1]), F=hist[-1]["F"], its=len(hist), seconds=round(dt, 1),
                   M=M, nodes=nodes, kt=a.kt, kc=a.kc, ks=a.ks, tol_step=a.tol_step, steps=sum(len(g) - 1 for g in s.get_segs()),
                   piv=hist[0].get("piv"), sens=sens, tail_c=tc, tail_h=th, dDelta_est=hist[-1].get("dDelta_est"))
        if rec["dDelta_est"] is None:
            rec["dDelta_est"] = delta_correction(s, *s.residual(zh, zl))
        rec["polish"] = sum(1 for r in hist if r.get("polish"))
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        save(os.path.join(outdir, f"{a.tag}_D{D:.4f}.npz"), zh, zl, dict(D=D, M=M, nodes=nodes, kt=a.kt, kc=a.kc, ks=a.ks))
        print(f"D={D:.4f} Delta={zh[-1]:.16f}{zl[-1]:+.2e} its {len(hist)} |F| {hist[-1]['F']:.1e} {dt:.0f}s "
              f"piv {rec['piv'][0]:.1e}..{rec['piv'][1]:.1e} dDelta/dF max {max(sens):.1e} tails {tc:.0e} {th:.0e}", flush=True)
        if tc > 1e-33:
            a.kc = int(a.kc * 1.25) + 2
            print(f"   centre series tail {tc:.1e}: kc -> {a.kc}", flush=True)
        if th > 1e-33:
            a.ks = int(a.ks * 1.25) + 2
            print(f"   horizon series tail {th:.1e}: ks -> {a.ks}", flush=True)
        its_main = len(hist) - rec["polish"]
        if its_main <= 8:
            dD = min(a.maxdD, dD * 1.25)
        elif its_main >= 14:
            dD *= 0.8
        while sched and (D - sched[0][0]) * direction >= -1e-9:
            new_nodes = sched.pop(0)[1]
            if new_nodes == nodes:
                continue
            zh, zl = renode(s, zh, zl, nodes, new_nodes, M)
            s, zh, zl, hist, ok, dt = solve_at(D, zh, zl, a, new_nodes, M, a.it_max + 4)
            print(f"re-noded at D={D}: {nodes} -> {new_nodes}: Delta {zh[-1]:.16f} ok {ok} ({dt:.0f}s)", flush=True)
            if not ok:
                raise SystemExit("re-noding did not converge")
            with open(a.out, "a") as f:
                f.write(json.dumps(dict(D=D, Delta=float(zh[-1]), Delta_lo=float(zl[-1]), F=hist[-1]["F"], its=len(hist),
                                        M=M, nodes=new_nodes, kt=a.kt, kc=a.kc, ks=a.ks, renoded=True,
                                        sens=sensitivity(s, M), tail_c=tails(s, zh, zl, new_nodes, M)[0],
                                        tail_h=tails(s, zh, zl, new_nodes, M)[1])) + "\n")
            nodes = new_nodes
            s_of = lambda Dx: make_solver(Dx, M, nodes=nodes, kt=a.kt, kc=a.kc, ks=a.ks, threads=a.threads, kfreeze=a.kfreeze)
            hist_z = [(D, zh + zl, tangent(s_of, s, D, zh, zl))]   # one point: Euler with the tangent
            dD = min(dD, 0.5)


if __name__ == "__main__":
    main()

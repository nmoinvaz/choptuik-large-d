#!/usr/bin/env python3
"""A homotopy in the held offset of the pinned system, from a start that solves its system exactly (#562).

    cd hp && python3 held_homotopy.py RAWDIR STATE OUT [--mode real|control|planted] [--threads 4] [--rho FILE]
        [--eps E] [--mutate sign] [--probe_only 1] [--h0 H] [--h_max 0.05] [--rungs 400] [--hours 24]
        [--resume 1] [--allow_unraw]

STATE is relative to RAWDIR; OUT is a path prefix (OUT.jsonl, OUT_setup.npz, OUT_ckpt.npz, OUT_end.npz), which must
lie in a task folder of the raw-data repository (hp/raw_out.py) unless --allow_unraw is given.

The pinned system F(z) - c is anm_run.py's base polish at the state's D (renode_o.pinned: the held offset c at the
state's own coefficients of p1's harmonics (kfreeze, 13], in eight parts, on the stored plan). The homotopy is

    H(z, t) = F(z) - c - a - (1 - t) b,

with a and b chosen so that the start z0 solves H(z0, 0) = 0 exactly.
- real: a = 0, b = r0 = F(z0) - c, the start's pinned residual. t = 1 is the pinned system itself.
- control: a = r0 - rho, b = rho, with rho the b of a real run (--rho, its OUT_setup.npz) kept on the horizon block's
  p rows. The target moves by -rho from the one z0 solves, as the real mode's moves by -r0.
- planted: zP = z0 + w, w = eps tau0 / |tau0| (tau0 = -J^-1 r0, the real mode's first tangent; eps signed), a =
  F(zP) - c, b = F(z0) - F(zP). Then zP solves H(z, 1) = 0: the known answer.
--mutate sign uses -b in H, so that z0 no longer solves H(z, 0) = 0, and the t = 0 check must fail.

Probe, at z0 with its octo-double LU (Solver.newton_o): the t = 0 check, the first tangent tau0 = -J^-1 b, the k = 4
weakest right singular vectors V (near_null.subspace), Delta's linear response by block of b, and a chord ladder:
for h = +-10^-j, H(z0 + h tau0, h) and the chord correction delta = -J(z0)^-1 H, whose size relative to h |tau0|
says where the linear model of the path ends.

Trace (the rules, fixed before the first real run and written on #562):
- A rung goes from t to t' = min(t + h, 1). Predictor z0' = z + (t' - t) tau, tau = -J(z)^-1 b by the LU at z.
- Corrector: full Newton steps on H(., t') from the predictor (PinnedO with the offset c + a + (1 - t') b). It
  converges when a correction's max falls to tol = max(1e-12 (t' - t) max|tau|, 1e-70). The rung is refused when the
  first correction exceeds half the predicted move (t' - t) max|tau|, when a correction exceeds half the one before,
  when a Jacobian or correction is not finite, or after 8 corrections.
- Refused: h <- (t' - t) / 4 from the same z and tau; 12 refusals in a row stop the trace (exit 3).
- Accepted after n corrections: h <- 4 (t' - t) if n <= 3, (t' - t) if n <= 5, else (t' - t) / 2; h <= h_max.
- h0: --h0, or the largest h of the probe's ladder (+ side) whose chord correction, and every smaller h's, is at
  most 0.1 h max|tau0|, capped at h_max.
- At t = 1 Newton continues until max|H| <= 1e-60 or a correction stops contracting (the endpoint polish).
Each rung's record (OUT.jsonl, written as it goes): t, h, the corrections, max|H|, Delta(t) to 40 digits, Delta(t) -
Delta(0), |z(t) - z0| (2-norm and max), V^T (z(t) - z0) on z0's V and its share of |z(t) - z0|.

Checks (exit 1): the t = 0 check, max|H(z0, 0)| <= 1e-20 max|b|. Planted: the trace reaches t = 1, and the endpoint
is zP to max|z(1) - zP| <= 1e-6 max|w| and |Delta(1) - Delta(zP)| <= 1e-6 max|w|.
"""
import argparse
import json
import os
import sys
import time

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hpkernel.driver import make_solver  # noqa: E402
from multidouble import od_axpy  # noqa: E402
from near_null import subspace  # noqa: E402
from octo_corrector import PinnedO, load_o, o8, save_o  # noqa: E402
from raw_out import require_raw  # noqa: E402
from renode_o import blocks, pinned  # noqa: E402

mp.mp.dps = 140


def add(x8, y8, a=1.0):
    """y + a x in eight parts"""
    return o8(od_axpy(float(a), *o8(x8), *o8(y8)))


def f64(x8):
    return np.asarray(x8[0]) + np.asarray(x8[1])


def mx(x8):
    return float(np.abs(f64(x8)).max())


def zero8(L):
    return o8([np.zeros(L)] * 8)


def delta_mp(z8):
    """Delta, the last unknown, summed from its eight parts"""
    return mp.fsum(mp.mpf(float(p[-1])) for p in z8)


def dist(z8, z08):
    """z - z0 in f64 (the difference formed in octo-double, so exact to f64 rounding of the difference)"""
    return f64(add(z08, z8, -1.0))


class System:
    """H(z, t) = F(z) - c - a - (1 - t) b (sign -1: + (1 - t) b, the control --mutate sign) on the solver s"""

    def __init__(self, s, c8, a8, b8, sign=1.0):
        self.s, self.ca8, self.b8, self.sign = s, add(a8, c8), o8(b8), sign

    def at(self, t):
        off = add(self.b8, self.ca8, self.sign * (1.0 - t))
        return PinnedO(self.s, off[0], off[1], c8=off)

    def tangent(self):
        """dz/dt = -J^-1 b by the current LU (sign as H's)"""
        x = o8(self.s.olu_solve(list(self.b8)))
        return add(x, zero8(len(x[0])), -self.sign)


def correct(so, zp, tol, first_max, it_max=8):
    """full Newton steps on so from zp: (z, corrections, ok, why); z's LU is the solver's on success"""
    corr = []
    try:
        d, Fh, Fl, _, _ = so.newton_o(zp)
    except ValueError as e:
        return zp, corr, False, f"Jacobian: {e}"
    z = zp
    for j in range(it_max + 1):
        nd = mx(d)
        nF = float(np.abs(np.asarray(Fh) + np.asarray(Fl)).max())
        corr.append(dict(F=nF, dz=nd))
        if not (np.isfinite(nd) and np.isfinite(nF)):
            return z, corr, False, "not finite"
        if nd <= tol:
            return z, corr, True, "converged"
        if j == 0 and nd > first_max:
            return z, corr, False, "first correction above half the predicted move"
        if j and nd > 0.5 * corr[-2]["dz"]:
            return z, corr, False, "no contraction"
        if j == it_max:
            return z, corr, False, "8 corrections"
        z = add(d, z)
        try:
            d, Fh, Fl, _, _ = so.newton_o(z)
        except ValueError as e:
            return z, corr, False, f"Jacobian: {e}"
    return z, corr, False, "8 corrections"


def endpoint_polish(so, z, tol=1e-60, it_max=8):
    """Newton at t = 1 until max|H| <= tol or a correction stops contracting; (z, its max|H|, records)"""
    rec = []
    d, Fh, Fl, _, _ = so.newton_o(z)
    for _ in range(it_max + 1):
        nF = float(np.abs(np.asarray(Fh) + np.asarray(Fl)).max())
        rec.append(dict(F=nF, dz=mx(d)))
        if nF <= tol or (len(rec) > 1 and rec[-1]["dz"] > 0.5 * rec[-2]["dz"]):
            break
        zn = add(d, z)
        dn, Fhn, Fln, _, _ = so.newton_o(zn)
        if float(np.abs(np.asarray(Fhn) + np.asarray(Fln)).max()) > nF:
            rec.append(dict(F=float(np.abs(np.asarray(Fhn) + np.asarray(Fln)).max()), dz=mx(dn), refused=True))
            break
        z, d, Fh, Fl = zn, dn, Fhn, Fln
    return z, rec[-1]["F"] if not rec[-1].get("refused") else rec[-2]["F"], rec


def describe(z8, z08, V, D0):
    dz = dist(z8, z08)
    n = float(np.linalg.norm(dz))
    c = V.T @ dz
    Dt = delta_mp(z8)
    return dict(Delta=mp.nstr(Dt, 40), dDelta=float(Dt - D0), dist=n, dist_max=float(np.abs(dz).max()),
                VT=[float(x) for x in c], share=float(np.linalg.norm(c) / n) if n else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("state")
    ap.add_argument("out")
    ap.add_argument("--mode", default="real", choices=("real", "control", "planted"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--rho", default="")
    ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument("--mutate", default="", choices=("", "sign"))
    ap.add_argument("--probe_only", type=int, default=0)
    ap.add_argument("--h0", type=float, default=0.0)
    ap.add_argument("--h_max", type=float, default=0.05)
    ap.add_argument("--rungs", type=int, default=400)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--resume", type=int, default=0)
    ap.add_argument("--ladder", default="0:46:2")
    ap.add_argument("--allow_unraw", action="store_true")
    a = ap.parse_args()
    if not a.allow_unraw:
        require_raw(a.out + ".jsonl")
    T0 = time.time()
    z0, m = load_o(os.path.join(a.raw, a.state))
    M, nodes, kf = int(m["M"]), list(m["nodes"]), int(m.get("kfreeze", 3))
    K, L = len(nodes), len(z0[0])
    D = m.get("D_str", repr(float(m["D"])))
    s = make_solver(D, M, nodes=nodes, kt=m.get("kt", 36), kc=m.get("kc", 146), ks=m.get("ks", 130), threads=a.threads,
                    kfreeze=kf)
    s.set_segs(m["segs"])
    hrows = slice(4 * M * K, 4 * M * K + M)
    pin = pinned(s, z0, M, kf)
    c8 = o8(pin.c8)
    r0 = o8(pin.residual_o(z0))                                  # F(z0) - c, eight parts
    D0 = delta_mp(z0)
    log = open(a.out + ".jsonl", "a" if a.resume else "w")

    def emit(rec):
        log.write(json.dumps(rec, default=float) + "\n")
        log.flush()
    print(f"{a.state}: D {float(m['D']):.10g}, nodes {nodes}, L {L}, mode {a.mode}, mutate '{a.mutate}'; pinned max|F| "
          f"{mx(r0):.3e}; by block: " + "; ".join(f"{k} {v:.2e}" for k, v in blocks(r0, M, K, kf).items()), flush=True)
    setup_path = a.out + "_setup.npz"
    if a.resume:
        st = np.load(setup_path)
        a8 = o8([st[f"a{q}"] for q in range(8)])
        b8 = o8([st[f"b{q}"] for q in range(8)])
        tau0 = o8([st[f"tau{q}"] for q in range(8)])
        V = st["V"]
        zP = o8([st[f"zP{q}"] for q in range(8)]) if "zP0" in st else None
        sysH = System(s, c8, a8, b8, -1.0 if a.mutate == "sign" else 1.0)
        h0 = float(st["h0"])
    else:
        # ---- the system's a and b
        zP = None
        if a.mode == "real":
            a8, b8 = zero8(L), r0
        elif a.mode == "control":
            rf = np.load(a.rho)
            rho = [np.array(rf[f"b{q}"]) for q in range(8)]
            mask = np.zeros(L)
            mask[hrows] = 1.0
            rho = o8([p * mask for p in rho])
            a8, b8 = add(rho, r0, -1.0), rho
            print(f"  rho from {a.rho}: max {mx(rho):.3e} on the horizon block's p rows (the rest of that b, "
                  f"{float(np.abs(f64([np.array(rf[f'b{q}']) for q in range(8)]) * (1 - mask)).max()):.2e}, dropped)",
                  flush=True)
        t1 = time.time()
        pin.newton_o(z0)                                              # the LU at z0
        print(f"  octo-double LU at z0 {time.time() - t1:.0f} s", flush=True)

        def solve(v, transpose=False):
            x = s.olu_solve([np.ascontiguousarray(v)] + [np.zeros(L)] * 7, transpose)
            return np.asarray(x[0]) + np.asarray(x[1])
        tau_r = add(o8(s.olu_solve(list(r0))), zero8(L), -1.0)      # -J^-1 r0, the real mode's first tangent
        if a.mode == "planted":
            w = a.eps * f64(tau_r) / float(np.linalg.norm(f64(tau_r)))
            zP = add(o8([w] + [np.zeros(L)] * 7), z0)
            FP = o8(s.residual_o(list(zP)))
            Fz = o8(s.residual_o(list(z0)))
            a8 = add(c8, FP, -1.0)                                    # F(zP) - c
            b8 = add(FP, Fz, -1.0)                                    # F(z0) - F(zP)
            print(f"  planted zP = z0 + eps tau0/|tau0|, eps {a.eps:+.3e}: max|w| {float(np.abs(w).max()):.3e}, Delta "
                  f"part {w[-1]:+.3e}; max|b| {mx(b8):.3e}", flush=True)
        sysH = System(s, c8, a8, b8, -1.0 if a.mutate == "sign" else 1.0)
        tau0 = sysH.tangent()
        t1 = time.time()
        V, hist = subspace(solve, L, 4, 16)
        sig = hist[-1]
        print(f"  weak basis at z0 ({time.time() - t1:.0f} s): sigma " + ", ".join(f"{x:.3e}" for x in sig), flush=True)
        # ---- probe
        H00 = sysH.at(0.0).residual_o(z0)
        ok0 = mx(H00) <= 1e-20 * mx(b8)
        rec = dict(kind="probe", state=a.state, mode=a.mode, mutate=a.mutate, D=float(m["D"]), nodes=nodes, L=L,
                   r0_max=mx(r0), r0_blocks=blocks(r0, M, K, kf), b_max=mx(b8), b_blocks=blocks(b8, M, K, kf),
                   H00=mx(H00), t0_check=bool(ok0), sigma=[float(x) for x in sig], Delta0=mp.nstr(D0, 40))
        print(f"  t = 0 check: max|H(z0, 0)| {mx(H00):.3e} against 1e-20 max|b| = {1e-20 * mx(b8):.2e}: "
              f"{'pass' if ok0 else 'FAIL'}", flush=True)
        if not ok0:
            emit(rec)
            print("check: FAIL (t = 0 is not exact)")
            return 1
        tf = f64(tau0)
        rec["tau0"] = dict(max=float(np.abs(tf).max()), norm=float(np.linalg.norm(tf)), Delta=float(tf[-1]),
                           VT=[float(x) for x in V.T @ tf], share=float(np.linalg.norm(V.T @ tf) / np.linalg.norm(tf)))
        print(f"  tangent tau0 = -J^-1 b: max {rec['tau0']['max']:.4e}, norm {rec['tau0']['norm']:.4e}, Delta "
              f"{rec['tau0']['Delta']:+.4e}; V^T tau0 " + " ".join(f"{x:+.3e}" for x in rec["tau0"]["VT"])
              + f", share {rec['tau0']['share']:.4f}", flush=True)
        # Delta's linear response by block of b
        resp = {}
        for name, rows in (("horizon p rows", hrows), ("the other rows", None)):
            mask = np.zeros(L)
            if rows is None:
                mask[:] = 1.0
                mask[hrows] = 0.0
            else:
                mask[rows] = 1.0
            x = add(o8(s.olu_solve(list(o8([np.asarray(p) * mask for p in b8])))), zero8(L), -1.0)
            resp[name] = dict(b_max=float(np.abs(f64(b8) * mask).max()), Delta=float(f64(x)[-1]),
                              norm=float(np.linalg.norm(f64(x))))
        rec["linear_response"] = resp
        print("  Delta's linear response -(J^-1 b)_Delta by rows of b: " + "; ".join(
            f"{k} (max|b| {v['b_max']:.2e}): {v['Delta']:+.3e}, |dz| {v['norm']:.3e}" for k, v in resp.items()), flush=True)
        # chord ladder along tau0
        j0, j1, js = (int(x) for x in a.ladder.split(":"))
        lad = []
        nt = float(np.abs(tf).max())
        for j in range(j0, j1 + 1, js):
            for sg in (1.0, -1.0):
                h = sg * 10.0 ** -j
                zh = add(tau0, z0, h)
                try:
                    Hh = sysH.at(h).residual_o(zh)
                except ValueError:
                    lad.append(dict(h=h, ok=False))
                    continue
                dl = f64(add(o8(s.olu_solve(list(o8(Hh)))), zero8(L), -1.0))
                nd = float(np.abs(dl).max())
                cos = float(dl @ tf / (np.linalg.norm(dl) * np.linalg.norm(tf))) if nd else 0.0
                lad.append(dict(h=h, ok=bool(np.isfinite(nd)), H=mx(Hh), delta=nd, ratio=nd / (abs(h) * nt), cos=cos,
                                VT=[float(x) for x in V.T @ dl], dDelta=float(dl[-1])))
        rec["ladder"] = lad
        print("  chord ladder: h, max|H(z0 + h tau0, h)|, max|delta|, ratio max|delta| / (|h| max|tau0|), cos(delta, "
              "tau0), delta's Delta part", flush=True)
        for e in lad:
            if e.get("ok"):
                print(f"    {e['h']:+.0e}  {e['H']:.3e}  {e['delta']:.3e}  {e['ratio']:.3e}  {e['cos']:+.4f}  "
                      f"{e['dDelta']:+.3e}", flush=True)
            else:
                print(f"    {e['h']:+.0e}  not finite", flush=True)
        good = [e for e in lad if e["h"] > 0]
        good.sort(key=lambda e: e["h"])
        h0 = 0.0
        for e in good:
            if e.get("ok") and e["ratio"] <= 0.1:
                h0 = e["h"]
            else:
                break
        h0 = min(a.h0 or h0, a.h_max)
        rec["h0"] = h0
        print(f"  h0 {h0:.1e} ({'--h0' if a.h0 else 'the ladder rule'}, h_max {a.h_max:g})", flush=True)
        emit(rec)
        np.savez(setup_path, **{f"a{q}": np.asarray(a8[q]) for q in range(8)}, **{f"b{q}": np.asarray(b8[q]) for q in range(8)},
                 **{f"tau{q}": np.asarray(tau0[q]) for q in range(8)}, V=V, sig=np.asarray(sig), h0=h0,
                 **({f"zP{q}": np.asarray(zP[q]) for q in range(8)} if zP is not None else {}))
        if a.probe_only:
            print(f"probe only, {time.time() - T0:.0f} s")
            return 0
        if h0 <= 0:
            print("no h on the ladder meets the h0 rule: give --h0")
            return 3

    # ---- the trace
    ck = a.out + "_ckpt.npz"
    if a.resume and os.path.exists(ck):
        c = np.load(ck)
        z = o8([c[f"z{q}"] for q in range(8)])
        tau = o8([c[f"tau{q}"] for q in range(8)])
        t, h, k = float(c["t"]), float(c["h"]), int(c["k"])
        print(f"  resumed at rung {k}, t {t:.6e}, h {h:.3e}", flush=True)
    else:
        z, tau, t, h, k = z0, tau0, 0.0, h0, 0
    refusals = 0
    ok_trace = False
    while k < a.rungs and time.time() - T0 < a.hours * 3600:
        t2 = min(t + h, 1.0)
        dt = t2 - t
        mt = mx(tau)
        zp = add(tau, z, dt)
        tol = max(1e-12 * dt * mt, 1e-70)
        t1 = time.time()
        so = sysH.at(t2)
        zn, corr, ok, why = correct(so, zp, tol, 0.5 * dt * mt)
        rec = dict(kind="rung", k=k, t=t, t_new=t2, h=dt, tau_max=mt, tol=tol, corr=corr, ok=ok, why=why,
                   seconds=round(time.time() - t1, 1))
        if not ok:
            refusals += 1
            h = dt / 4
            emit(rec)
            print(f"  rung {k}: t {t:.4e} -> {t2:.4e} refused ({why}; corrections "
                  + ", ".join(f"{c_['dz']:.1e}" for c_ in corr) + f"), h -> {h:.2e} [{refusals}]", flush=True)
            if refusals >= 12:
                print(f"trace stopped: 12 refusals in a row at t {t:.6e}")
                np.savez(ck, **{f"z{q}": np.asarray(z[q]) for q in range(8)}, **{f"tau{q}": np.asarray(tau[q]) for q in range(8)},
                         t=t, h=h, k=k)
                return 3
            continue
        refusals = 0
        n = len(corr) - 1
        z_prev, z = z, zn
        t = t2
        tau = sysH.tangent()                                          # the LU at z is the corrector's last
        rec.update(describe(z, z0, V, D0), n=n, move=float(np.linalg.norm(dist(z, z_prev))),
                   pred_err=float(np.linalg.norm(dist(z, zp))))
        emit(rec)
        print(f"  rung {k}: t {rec['t_new']:.4e} (h {dt:.2e}), {n} corrections " + ", ".join(f"{c_['dz']:.1e}" for c_ in corr)
              + f"; |H| {corr[-1]['F']:.2e}; Delta - Delta0 {rec['dDelta']:+.4e}; |z - z0| {rec['dist']:.4e} (max "
              f"{rec['dist_max']:.3e}), share on V {rec['share']:.4f}, V^T " + " ".join(f"{x:+.2e}" for x in rec["VT"])
              + f" ({rec['seconds']} s)", flush=True)
        k += 1
        h = (4 * dt if n <= 3 else dt if n <= 5 else dt / 2)
        h = min(h, a.h_max)
        np.savez(ck, **{f"z{q}": np.asarray(z[q]) for q in range(8)}, **{f"tau{q}": np.asarray(tau[q]) for q in range(8)},
                 t=t, h=h, k=k)
        if t >= 1.0:
            ok_trace = True
            break
    if not ok_trace:
        print(f"trace stopped at t {t:.6e} after {k} rungs ({time.time() - T0:.0f} s): rung or time limit")
        return 3
    # ---- the endpoint
    so = sysH.at(1.0)
    z, F_end, pol = endpoint_polish(so, z)
    end = describe(z, z0, V, D0)
    end.update(kind="end", F=F_end, polish=pol, rungs=k, seconds=round(time.time() - T0))
    meta = {kk: v for kk, v in m.items()}
    meta.update(parts=8, source=a.state, homotopy=dict(mode=a.mode, F=F_end, dDelta=end["dDelta"], dist=end["dist"]))
    save_o(a.out + "_end.npz", z, meta)
    print(f"endpoint: max|H(z, 1)| {F_end:.3e} after the polish (" + ", ".join(f"{p['F']:.1e}" for p in pol) + f"); Delta "
          f"{end['Delta']}; Delta - Delta0 {end['dDelta']:+.6e}; |z - z0| {end['dist']:.6e} (max {end['dist_max']:.3e}), "
          f"share on V {end['share']:.4f}", flush=True)
    ok = True
    if zP is not None:
        e = float(np.abs(dist(z, zP)).max())
        eD = float(abs(delta_mp(z) - delta_mp(zP)))
        wm = float(np.abs(dist(zP, z0)).max())
        ok = e <= 1e-6 * wm and eD <= 1e-6 * wm
        end.update(planted=dict(err=e, err_Delta=eD, w_max=wm, ok=bool(ok)))
        print(f"planted check: max|z(1) - zP| {e:.3e}, |Delta(1) - Delta(zP)| {eD:.3e}, against 1e-6 max|w| = {1e-6 * wm:.3e}: "
              f"{'pass' if ok else 'FAIL'}", flush=True)
    emit(end)
    print(f"check: {'pass' if ok else 'FAIL'} ({time.time() - T0:.0f} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

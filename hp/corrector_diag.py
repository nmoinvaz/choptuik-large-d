#!/usr/bin/env python3
"""Corrector diagnostic for the large-D continuation: the residual floor at a converged state, and a Newton trace from
the scan's predictor seed, both with the quad-double horizon (QdhSolver).

    python3 corrector_diag.py OUT.jsonl S1.npz,...,Sn.npz D_TARGET [--kt 36 --kc 146 --ks 130 --kfreeze 13]
        [--tol_step 1e-32 --hmax 0.05] [--its 12] [--deflated TOL] [--pin 11,7,5] [--threads N] [--mutate=jvp_scale]

S1..Sn: converged states in one layout, oldest first. The seed at D_TARGET is pinning.predict_dd through the last 4
(scan_hp --predictor lagrange_dd).

1. Floor, at Sn: z moved by delta = r |z| (random signs, seed 1, quad-double parts kept by QdhSolver.zadd) for
   r = 1e-32 .. 1e-28; per residual block, |J delta| (QdhSolver.jvp) and the remainder |F(z + delta) - F(z) - J delta|.
   A remainder that does not fall with r is the evaluation's rounding (the floor); |J delta| at r = 1e-32 is what
   one unit in the last place of a double-double z moves F by.
2. Trace, from the seed: plain Newton, the step -J^-1 F (QdhSolver.newton) with halving on |F|_2 (at most 12), per
   iteration |F| per block, the step's size in p1's harmonics up to and above 7, the other unknowns and Delta.
   --deflated TOL also runs driver.deflated_newton from the seed with scan_hp's settings (w 24) and prints its lines.
3. --pin K,...: deflated Newton (as 2) on the kfreeze = K system with p1's harmonics in (K, kfreeze] held at the
   seed's values (pinning.PinnedSolver, as scan_hp --pin with the seed for the trend); then at its end the full
   (kfreeze) system's residual per block and one full-system Newton step's Delta change (what the pin costs).

Controls: L, at r = 1e-28 the remainder is below 1e-2 of |J delta| in every block (the product and the residual
agree; at smaller r the inner blocks' remainder is their rounding, about 3e-31, flat in r); --mutate=jvp_scale scales
J delta by 1.1 and must make L fire. The trace has no pass criterion; run it
on a step known to converge (D 62.0098 -> 62.0306, T48) as its positive control.
"""
import argparse
import json
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

from hpkernel.driver import QdhSolver, deflated_newton, load, load_parts, make_solver, plow_odd  # noqa: E402
from pinning import PinnedSolver, band_coeffs, band_field, dd_add, predict_dd  # noqa: E402


def as_f(r):
    return np.asarray(r[0]) + np.asarray(r[1])


def blocks(F, M):
    """max |F| per residual block (4M rows each; the last block is the horizon), then the final row"""
    nb = (len(F) - 1) // (4 * M)
    return [float(np.abs(F[b * 4 * M:(b + 1) * 4 * M]).max()) for b in range(nb)] + [float(abs(F[-1]))]


def fmt(v):
    return " ".join(f"{x:.1e}" for x in v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("states")
    ap.add_argument("D_target", type=float)
    ap.add_argument("--kt", type=int, default=36)
    ap.add_argument("--kc", type=int, default=146)
    ap.add_argument("--ks", type=int, default=130)
    ap.add_argument("--kfreeze", type=int, default=13)
    ap.add_argument("--tol_step", type=float, default=1e-32)
    ap.add_argument("--hmax", type=float, default=0.05)
    ap.add_argument("--its", type=int, default=12)
    ap.add_argument("--deflated", type=float, default=0.0)
    ap.add_argument("--pin", default="")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--mutate", default="", choices=["", "jvp_scale"])
    a = ap.parse_args()
    t_start = time.time()

    paths = [p for p in a.states.split(",") if p]
    pts = [load(p) for p in paths]
    zh, zl, meta = pts[-1]
    D1, M, nodes = meta["D"], meta["M"], list(meta["nodes"])
    L = len(zh)
    kw = dict(nodes=nodes, kt=a.kt, kc=a.kc, ks=a.ks, threads=a.threads, kfreeze=a.kfreeze)
    lo, lo2 = plow_odd(M, 7)
    P = np.asarray(lo) + np.asarray(lo2)
    rec = dict(states=paths, D_target=a.D_target, mutate=a.mutate)

    # 1. floor at the last state
    s = make_solver(D1, M, **kw)
    s.set_segs(s.plan(zh, zl, a.tol_step, a.hmax))
    q = QdhSolver(s)
    q._parts[q._key(zh, zl)] = load_parts(paths[-1])
    F0 = as_f(q.residual(zh, zl))
    print(f"floor at D={D1}: |F| per block {fmt(blocks(F0, M))}", flush=True)
    rng = np.random.default_rng(1)
    sg = rng.choice([-1.0, 1.0], L)
    z = zh + zl
    rows = []
    ok_L = True
    for r in (1e-32, 1e-31, 1e-30, 1e-29, 1e-28):
        d = r * np.abs(z) * sg
        th, tl = q.zadd(zh, zl, d, np.zeros(L))
        F1 = as_f(q.residual(th, tl))
        Jd = as_f(q.jvp(zh, zl, d, np.zeros(L)))
        if a.mutate == "jvp_scale":
            Jd = Jd * 1.1
        jb, nb = blocks(Jd, M), blocks(F1 - F0 - Jd, M)
        rows.append(dict(r=r, Jd=jb, rem=nb))
        print(f"  r {r:.0e}: |J delta| {fmt(jb)}", flush=True)
        print(f"          remainder {fmt(nb)}", flush=True)
        if r == 1e-28:
            ok_L = all(n <= 1e-2 * j for n, j in zip(nb[:-1], jb[:-1]) if j > 0)
    print(f"L (remainder below 1e-2 of |J delta| at r 1e-28): {'PASS' if ok_L else 'FIRE'}", flush=True)
    rec.update(floor=dict(D=D1, F=blocks(F0, M), rows=rows, L=ok_L))

    # 2. trace from the seed
    seed = predict_dd([(p[2]["D"], p[0], p[1]) for p in pts], a.D_target, 4)
    sh, sl = (np.asarray(v) for v in seed)
    st = make_solver(a.D_target, M, **kw)
    st.set_segs(st.plan(sh, sl, a.tol_step, a.hmax))
    qt = QdhSolver(st)
    Fs = as_f(qt.residual(sh, sl))
    print(f"trace D {D1} -> {a.D_target} (dD {a.D_target - D1:.3e}): seed |F| {np.abs(Fs).max():.2e}, blocks "
          f"{fmt(blocks(Fs, M))}", flush=True)
    trace = []
    ch, cl = sh, sl
    for it in range(a.its):
        dh, dl, Fh, Fl, pmin, pmax = qt.newton(ch, cl)
        F = np.asarray(Fh) + np.asarray(Fl)
        n0 = float(np.linalg.norm(F))
        dz = np.asarray(dh) + np.asarray(dl)
        lam, best = 1.0, None
        for _ in range(12):
            th, tl = qt.zadd(ch, cl, lam * np.asarray(dh), lam * np.asarray(dl))
            n1 = float(np.linalg.norm(as_f(qt.residual(th, tl))))
            if n1 < n0:
                best = (th, tl, n1)
                break
            lam *= 0.5
        t = dict(it=it, F=float(np.abs(F).max()), blocks=blocks(F, M), piv=(float(pmin), float(pmax)),
                 dz_p1_low=float(np.abs(P @ dz[:M]).max()), dz_p1_high=float(np.abs(dz[:M] - P @ dz[:M]).max()),
                 dz_rest=float(np.abs(dz[M:-1]).max()), dz_Delta=float(dz[-1]), lam=lam if best else 0.0,
                 Delta=float(ch[-1]) + float(cl[-1]))
        trace.append(t)
        print(f"  it {it}: |F| {t['F']:.2e} [{fmt(t['blocks'])}]  dz p1<=7 {t['dz_p1_low']:.1e} p1>7 "
              f"{t['dz_p1_high']:.1e} rest {t['dz_rest']:.1e} Delta {t['dz_Delta']:+.1e}  lam {t['lam']:g}  "
              f"Delta {t['Delta']:.16f}", flush=True)
        if best is None:
            print("  no decrease along the step: stop", flush=True)
            break
        ch, cl = best[0], best[1]
    Fe = as_f(qt.residual(ch, cl))
    print(f"  end: |F| {np.abs(Fe).max():.2e} [{fmt(blocks(Fe, M))}]", flush=True)
    rec.update(trace=trace, F_end=blocks(Fe, M))

    if a.deflated:
        print(f"deflated Newton from the seed (tol {a.deflated:g}, w 24):", flush=True)
        st2 = make_solver(a.D_target, M, **kw)
        st2.set_segs(st2.plan(sh, sl, a.tol_step, a.hmax))
        _, _, h = deflated_newton(QdhSolver(st2), sh, sl, tol=a.deflated, it_max=30, w=24, verbose=True)
        rec["deflated"] = [dict(F=r_.get("F"), mode=r_.get("mode")) for r_ in h]
    tol = a.deflated or 1.41e-21
    rec["pins"] = []
    for K in [int(v) for v in a.pin.split(",") if v]:
        ks = list(range(K + 2, a.kfreeze + 1, 2))
        sp = make_solver(a.D_target, M, **dict(kw, kfreeze=K))
        sp.set_segs(sp.plan(sh, sl, a.tol_step, a.hmax))
        hh, hl = band_field(band_coeffs(sh[:M], sl[:M], M, ks), ks, M)
        ps = PinnedSolver(QdhSolver(sp), M, hh, hl)
        print(f"pin {K} (p1 harmonics {ks[0]}..{ks[-1]} held at the seed's), deflated Newton, tol {tol:g}:", flush=True)
        t0 = time.time()
        ph, pl, h = deflated_newton(ps, sh, sl, tol=tol, it_max=30, w=24, verbose=False)
        # the full system at the pinned solution: residual, and one Newton step (its Delta change is the pin's cost)
        sf = make_solver(a.D_target, M, **kw)
        sf.set_segs(sf.plan(ph, pl, a.tol_step, a.hmax))
        qf = QdhSolver(sf)
        dh, dl, Fh, Fl, _, _ = qf.newton(ph, pl)
        Ff = np.asarray(Fh) + np.asarray(Fl)
        r_ = dict(K=K, its=len(h), F_pinned=h[-1]["F"] if h else None, Delta=float(ph[-1]) + float(pl[-1]),
                  F_full=blocks(Ff, M), dDelta_full_step=float(dh[-1]) + float(dl[-1]), seconds=round(time.time() - t0, 1))
        rec["pins"].append(r_)
        print(f"  {r_['its']} its, pinned |F| {r_['F_pinned']:.2e}, Delta {r_['Delta']:.16f}; full system: |F| per block "
              f"{fmt(r_['F_full'])}, one full step moves Delta by {r_['dDelta_full_step']:+.2e} ({r_['seconds']:.0f}s)",
              flush=True)
    rec["seconds"] = round(time.time() - t_start, 1)
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"done in {rec['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()

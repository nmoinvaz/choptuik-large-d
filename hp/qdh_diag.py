#!/usr/bin/env python3
"""Why Newton with the quad-double horizon residual stalls at |F| ~ 5e-22 (qdh_check.py check C, comms 2026-10-25
19:52 UTC mac): the Jacobian, or the residual's curvature along the step?

    cd hp && python3 qdh_diag.py [STATE.npz] [--threads=N] [--out=data/large_d/qdh_diag.jsonl]

At the state (as qdh_check.py: kfreeze 13, the scan's Taylor plan) with QdhSolver: the Newton step d = -J^-1 F0 from
the double-double Jacobian J and the quad-double-horizon F0, then
1. along the step: F(z + lam d) for lam = 2^-k, k = 0..14, every state made in quad-double, and
   R(lam) = F(z + lam d) - (1 - lam) F0, the departure from the linear model with J. If J were the residual's exact
   Jacobian, R(lam) ~ lam^2 (curvature, |R| / lam^2 constant as lam -> 0). An error E of J gives R(lam) ~ -lam E F0
   (|R| / lam constant). Both ratios are printed, with R split into the horizon rows and the others.
2. the derivative along d: (F(z + h d) - F(z - h d)) / (2 h) for h = 2^-k, k = 0..30 (central differences of the
   quad-double-horizon residual, the states in quad-double), against J d = -F0 (Solver.jvp, the double-double
   Jacobian's product, also printed): their difference is E d at the h where the differences plateau.
3. the step refined by GCR whose products have quad-double horizon rows (QdhSolver with gcr; --products, default 12),
   against d; then up to --its (default 4) full Newton steps with it, without a line search, while |F| decreases.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hpkernel.driver import QdhSolver, load, make_solver  # noqa: E402


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    path = args[0] if args else os.path.join(HERE, "data", "large_d", "up70_D67.7224.npz")
    out = opts.get("out", os.path.join(HERE, "data", "large_d", "qdh_diag.jsonl"))
    zh, zl, meta = load(path)
    zh, zl = np.ascontiguousarray(zh), np.ascontiguousarray(zl)
    D, M, nodes = meta["D"], meta["M"], meta["nodes"]
    s = make_solver(D, M, nodes=nodes, kt=meta["kt"], kc=meta["kc"], ks=meta["ks"], threads=int(opts.get("threads", 2)),
                    kfreeze=13)
    s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
    L, K = s.size, len(nodes)
    hb = np.zeros(L, bool)
    hb[K * 4 * M:(K + 1) * 4 * M] = True
    qs = QdhSolver(s)
    dh, dl, Fh, Fl, pmin, pmax = qs.newton(zh, zl)
    dh, dl = np.asarray(dh), np.asarray(dl)
    F0 = np.asarray(Fh) + np.asarray(Fl)
    d = dh + dl
    print(f"D {D}: M {M}, K {K}, L {L}; |F0| {np.abs(F0).max():.3e} (horizon {np.abs(F0[hb]).max():.3e}, others "
          f"{np.abs(F0[~hb]).max():.3e}); |d| {np.abs(d).max():.3e}, d_Delta {d[-1]:+.3e}; pivots {pmin:.1e}..{pmax:.1e}",
          flush=True)
    jh, jl = s.jvp(zh, zl, dh, dl)[:2]
    Jd = np.asarray(jh) + np.asarray(jl)
    print(f"  J d + F0 (Solver.jvp, the double-double Jacobian): {np.abs(Jd + F0).max():.3e} (horizon "
          f"{np.abs((Jd + F0)[hb]).max():.3e})", flush=True)

    def Fat(t):
        a, b = qs.zadd(zh, zl, t * dh, t * dl)
        h, l_ = qs.residual(a, b)
        return np.asarray(h) + np.asarray(l_)

    rec = dict(state=os.path.basename(path), D=D, M=M, K=K, F0=float(np.abs(F0).max()), d=float(np.abs(d).max()),
               d_Delta=float(d[-1]), pivots=[pmin, pmax], jvp_check=float(np.abs(Jd + F0).max()), line=[], deriv=[])
    print("  1. along the step: lam, |F(lam)|, |R|, |R| / lam, |R| / lam^2 (all rows; horizon rows; others)", flush=True)
    for k in range(15):
        lam = 2.0 ** -k
        Fl_ = Fat(lam)
        R = Fl_ - (1 - lam) * F0
        r, rh, ro = (float(np.abs(v).max()) for v in (R, R[hb], R[~hb]))
        rec["line"].append(dict(lam=lam, F=float(np.abs(Fl_).max()), R=r, R_h=rh, R_o=ro))
        print(f"    2^-{k:<2d} |F| {np.abs(Fl_).max():.3e}  |R| {r:.3e}  /lam {r / lam:.3e}  /lam^2 {r / lam ** 2:.3e}   "
              f"horizon {rh:.2e} ({rh / lam:.2e}, {rh / lam ** 2:.2e})  others {ro:.2e} ({ro / lam:.2e}, {ro / lam ** 2:.2e})",
              flush=True)
    print("  2. central differences along d: h, |(F(+h) - F(-h)) / 2h + F0| (all; horizon; others), the J error on d",
          flush=True)
    for k in range(0, 31, 2):
        h = 2.0 ** -k
        g = (Fat(h) - Fat(-h)) / (2 * h)
        e = g + F0
        v = [float(np.abs(x).max()) for x in (e, e[hb], e[~hb])]
        rec["deriv"].append(dict(h=h, E=v[0], E_h=v[1], E_o=v[2]))
        print(f"    2^-{k:<2d} {v[0]:.3e}  horizon {v[1]:.3e}  others {v[2]:.3e}", flush=True)
    # 3. the step refined by GCR with quad-double horizon products (QdhSolver gcr), then full Newton steps with it
    qg = QdhSolver(s, horizon=slice(K * 4 * M, (K + 1) * 4 * M), gcr=dict(max_products=int(opts.get("products", 12)),
                                                                          step_tol=1e-8, verbose=True))
    gh, gl, _, _, _, _ = qg.newton(zh, zl)
    gd = np.asarray(gh) + np.asarray(gl)
    info = qg.last_gcr
    print(f"  3. GCR step: {info['products']} products, changes {', '.join(f'{c:.1e}' for c in info['changes'])}; "
          f"|dz_gcr| {np.abs(gd).max():.3e}, |dz_gcr - d| {np.abs(gd - d).max():.3e}, dDelta {gd[-1]:+.3e}", flush=True)
    rec["gcr"] = dict(products=info["products"], changes=info["changes"], dz=float(np.abs(gd).max()),
                      diff=float(np.abs(gd - d).max()), dDelta=float(gd[-1]), steps=[])
    zc_h, zc_l = zh, zl
    for it in range(int(opts.get("its", 4))):
        gh, gl, Fh, Fl, _, _ = qg.newton(zc_h, zc_l)
        n0 = float(np.abs(np.asarray(Fh) + np.asarray(Fl)).max())
        th, tl = qg.zadd(zc_h, zc_l, np.asarray(gh), np.asarray(gl))
        Th, Tl = qg.residual(th, tl)
        n1 = float(np.abs(np.asarray(Th) + np.asarray(Tl)).max())
        dz = float(np.abs(np.asarray(gh) + np.asarray(gl)).max())
        rec["gcr"]["steps"].append(dict(F=n0, F_after=n1, dz=dz, products=qg.last_gcr["products"]))
        print(f"    full step {it}: |F| {n0:.3e} -> {n1:.3e}, |dz| {dz:.3e}, {qg.last_gcr['products']} products", flush=True)
        if not n1 < n0:
            break
        zc_h, zc_l = th, tl
    with open(out, "a") as fo:
        fo.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()

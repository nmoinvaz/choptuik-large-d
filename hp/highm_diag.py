#!/usr/bin/env python3
"""Diagnostics for the large-M solves (notes/HIGH_M.md, section 1.4).

    cd hp && python3 highm_diag.py jac STATE.npz M            (the state's meta must have its plan, segs)
    cd hp && python3 highm_diag.py radius STATE.npz M
    cd hp && python3 highm_diag.py start STATE.npz M_new [--nodes=x0,x1,...] [--renode_h=0.001] [--fill]
    cd hp && python3 highm_diag.py orders STATE.npz M [--orders=kc/ks,...] [--newton]

jac: do f64 Jacobians that differ only in rounding agree? Newton steps of Solver.newton_mixed in block storage with
  f64x4, f64x1 and dd4 at the state: their relative differences in dz and in the sensitivity y = J^-T e_Delta; the
  linear-model mismatch |F(z + t dz) - (1 - t) F(z)| / t at t = 1e-3 (= |(J - J~) dz| to first order) against |F|;
  and |F(z + dz)|.
radius: radius of convergence of the horizon series (in u = 1 - x) and of the centre series per tau-harmonic: the
  series coefficients of p, m, a, c transformed to harmonics, max over the fields; for harmonic k, exp(-slope) of a
  least-squares line of log |c_n(k)| over the second half of the orders; next to the tail max_k |c_n(k)| x^n of the
  last 3 orders at the evaluation point (x0 = first node, u0 = 1 - last node).
start: the start of a lift: the state (optionally re-noded at its own M with driver.renode, Taylor steps --renode_h)
  resampled to M_new (with --fill also lowD_dd.fill_tail: the harmonics above the old M by each field's geometric
  trend), its Taylor plan, |F| per residual block and the series tails.
orders: do lower centre / horizon series orders (kc, ks) change anything at a converged state? The residual on the
  state's own plan with its (kc, ks) and with each --orders pair: seconds (best of 3), |F|, the series tails, and
  whether the residual is bitwise identical; with --newton also one newton_mixed step (seconds, its Delta correction,
  whether the step is bitwise identical).
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

import lowD_dd  # noqa: E402
from hpkernel.driver import block_norms, dd_add, dd_norm, load, make_solver, renode, resample_z  # noqa: E402


def solver(m, M, nodes=None):
    return make_solver(m["D"], M, nodes=nodes or m["nodes"], kt=m["kt"], kc=m["kc"], ks=m["ks"], threads=2)


def jac(path, M):
    zh, zl, m = load(path)
    s = solver(m, M)
    s.set_segs(m["segs"])
    F0 = dd_norm(*s.residual(zh, zl))
    e = np.zeros(s.size)
    e[-1] = 1
    out = {}
    for name, kw in (("f64x4", dict(tangents=4)), ("f64x1", dict(tangents=1)), ("dd4", dict(tangents=4, dd_base=True))):
        t = time.time()
        dh, dl, Fh, Fl, p0, p1 = s.newton_mixed(zh, zl, block=True, **kw)
        dz = np.asarray(dh) + np.asarray(dl)
        y = np.add(*s.lu_solve_t(e, np.zeros(s.size)))
        tt = 1e-3
        th, tl = dd_add(zh, zl, np.asarray(dh) * tt, np.asarray(dl) * tt)
        Fh2, Fl2 = s.residual(th, tl)
        mis = np.abs((np.asarray(Fh2) + np.asarray(Fl2)) - (1 - tt) * (np.asarray(Fh) + np.asarray(Fl))).max() / tt
        th, tl = dd_add(zh, zl, np.asarray(dh), np.asarray(dl))
        F1 = dd_norm(*s.residual(th, tl))
        out[name] = (dz, y)
        print(f"{name}: {time.time() - t:.0f} s, |F| {F0:.2e}, |dz| {np.abs(dz).max():.2e}, dDelta {dz[-1]:+.2e}, pivots "
              f"{p0:.1e}..{p1:.1e}, model mismatch |(J - J~) dz| ~ {mis:.2e}, |F(z + dz)| {F1:.2e}", flush=True)
    for a, b in (("f64x4", "f64x1"), ("f64x4", "dd4"), ("f64x1", "dd4")):
        da = np.abs(out[a][0] - out[b][0]).max() / np.abs(out[b][0]).max()
        ya = np.abs(out[a][1] - out[b][1]).max() / np.abs(out[b][1]).max()
        print(f"{a} vs {b}: dz {da:.2e}, sensitivity {ya:.2e}", flush=True)


def radius(path, M):
    zh, zl, m = load(path)
    s = solver(m, M)
    for which, x, name in ((1, 1 - m["nodes"][-1], "horizon (u0)"), (0, m["nodes"][0], "centre (x0)")):
        h, l = s.series(zh, zl, which)
        c = (np.asarray(h) + np.asarray(l)).reshape(-1, 4, M)
        nmax = c.shape[0] - 1
        amp = np.zeros((nmax + 1, M))
        for q in range(4):
            f = c[:, q, :]
            full = np.concatenate([f, -f if q < 2 else f], axis=1)       # p, m odd; a, c even
            amp = np.maximum(amp, np.abs(np.fft.rfft(full, axis=1))[:, :M] / M)
        n = np.arange(nmax + 1)
        n0 = nmax // 2
        print(f"{name} = {x:.4g}: {nmax} terms; tail max_k |c_n(k)| x^n over the last 3 n: {(amp.max(1) * x ** n)[-3:].max():.1e}")
        for k in list(range(1, 12, 2)) + list(range(M // 2 - 1, M, M // 8)) + [M - 1]:
            y = amp[n0:, k]
            ok = y > 0
            if ok.sum() < 5:
                continue
            r = np.exp(-np.polyfit(n[n0:][ok], np.log(y[ok]), 1)[0])
            print(f"  k {k:3d}: |c_0| {amp[0, k]:.1e}, radius {r:.3f} (x/r = {x / r:.2f})")


def start(path, M, opts):
    zh, zl, m = load(path)
    Mo, nodes = m["M"], list(m["nodes"])
    if opts.get("nodes"):
        new = [float(v) for v in opts["nodes"].split(",")]
        s0 = solver(m, Mo)
        s0.set_segs(m.get("segs") or s0.plan(zh, zl, 1e-30, 0.05))
        zh, zl = renode(s0, zh, zl, nodes, new, Mo, h=float(opts.get("renode_h", 0.001)))
        zh, zl = np.asarray(zh), np.asarray(zl)
        nodes = new
    z = np.asarray(zh) + np.asarray(zl)
    if M != Mo:
        z, _ = resample_z(z, Mo, M, len(nodes))
    if opts.get("fill"):
        z = lowD_dd.fill_tail(z, M, len(nodes), Mo - 1, klo=Mo - 40)
    s = solver(m, M, nodes)
    segs = s.plan(z, 0 * z, 1e-30, 0.05)
    s.set_segs(segs)
    Fh, Fl = s.residual(z, 0 * z)
    print(f"{os.path.basename(path)} (M {Mo}) -> M {M}, nodes {nodes[0]}..{nodes[-1]} ({len(nodes)}){' filled' if opts.get('fill') else ''}: "
          f"plan {sum(len(g) - 1 for g in segs)} steps, |F| {dd_norm(Fh, Fl):.2e} (per block "
          f"{' '.join('%.0e' % v for v in block_norms(s, np.asarray(Fh), np.asarray(Fl)))}), tails (centre, horizon) "
          f"{' '.join('%.1e' % v for v in lowD_dd.tails(s, z, 0 * z, nodes, M))}", flush=True)


def orders(path, M, opts):
    zh, zl, m = load(path)
    pairs = [(m["kc"], m["ks"])] + [tuple(int(v) for v in p.split("/")) for p in str(opts.get("orders", "60/150")).split(",")]
    base = None
    for kc, ks in pairs:
        s = make_solver(m["D"], M, nodes=m["nodes"], kt=m["kt"], kc=kc, ks=ks, threads=2)
        s.set_segs(m["segs"])
        tr = []
        for _ in range(3):
            t = time.time()
            Fh, Fl = s.residual(zh, zl)
            tr.append(time.time() - t)
        Fh, Fl = np.asarray(Fh), np.asarray(Fl)
        tl = lowD_dd.tails(s, zh, zl, m["nodes"], M)
        line = (f"kc {kc} ks {ks}: residual {min(tr):.2f} s, |F| {dd_norm(Fh, Fl):.2e}, tails (centre, horizon) "
                f"{tl[0]:.1e} {tl[1]:.1e}")
        if base is not None:
            same = np.array_equal(Fh, base[0]) and np.array_equal(Fl, base[1])
            line += f", residual {'bitwise identical' if same else 'differs by %.1e' % np.abs((Fh - base[0]) + (Fl - base[1])).max()}"
        if opts.get("newton"):
            t = time.time()
            dh, dl, _, _, p0, p1 = s.newton_mixed(zh, zl)
            line += f"; Newton step {time.time() - t:.0f} s, dDelta {float(dh[-1]) + float(dl[-1]):+.1e}"
            if base is not None and base[2] is not None:
                line += f", step {'bitwise identical' if np.array_equal(np.asarray(dh), base[2]) else 'differs by %.1e' % np.abs(np.asarray(dh) - base[2]).max()}"
            dh = np.asarray(dh)
        else:
            dh = None
        if base is None:
            base = (Fh, Fl, dh)
        print(line + f" (load {os.getloadavg()[0]:.1f})", flush=True)


def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict((a[2:].split("=", 1) + [True])[:2] for a in sys.argv[1:] if a.startswith("--"))
    cmd, path, M = pos[0], pos[1], int(pos[2])
    {"jac": lambda: jac(path, M), "radius": lambda: radius(path, M), "start": lambda: start(path, M, opts),
     "orders": lambda: orders(path, M, opts)}[cmd]()


if __name__ == "__main__":
    main()

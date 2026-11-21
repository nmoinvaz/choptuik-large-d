#!/usr/bin/env python3
"""hp's low-D multiple shooting with one segment, the band through the sonic valley, marched on warped curves
(band.py) instead of curves of constant x (notes/LOW_D_SONIC.md).

The unknowns, the layout and every residual block are hp's (hpkernel.Solver, hp/src/model.rs): centre series, Taylor
steps between nodes, horizon series, the phase row and Delta. Only the rows of the band segment, nodes x_A -> x_B
(residual block of node x_B: F = Phi(s_A) - s_B, rows p, m, a, c on the half grid), are replaced by Phi_warp(s_A) - s_B
from band.py, in f64. The solve is a chord iteration with one LU: hp's f64 Jacobian at the start state, at the start D
(Solver.newton_mixed, with hp's own plan, so the start must lie above the closing), used at every D. A chord step
converges to the zero of the residual it is given whatever the LU, only its rate depends on the LU.

    python3 hybrid.py OUT.jsonl --state S.npz [--history A.npz,B.npz] --Ds 3.045,3.044 [--band 0.01,0.02]
        [--warp none|test|design|FILE.npy] [--A 1.5e-3] [--eps0 2e-3] [--kappa 1.0] [--tol 1e-12] [--it_max 40]
        [--threads 4] [--drop_nodes x1,x2] [--add_nodes x1,x2] [--save DIR] [--expect D:VALUE:TOL,...] [--mutate none|nowarp]
        [--time_cap SECONDS] [--allow_unraw]

--Ds: the D values in order; each starts from the Lagrange extrapolation through the (at most) four states nearest in D
among the start state, the --history states (same M and nodes; renoded with the start when --drop_nodes or --add_nodes
is given) and the solutions of this run, or from the state itself when one is at that D. --warp: none
(phi = 0, hp's own curves), test (phi = A (sin(2 theta + 0.3) + 0.3 cos(4 theta - 1.1)), an arbitrary warp for the
planted checks), design (from the start state's unwarped band at its D: omega b phi_theta = softplus(eps0 - g*) minus its
mean, g* the smallest c - x over the band at each theta; band.design_phi), or a saved phi. --expect D:VALUE:TOL: exit 1
unless the solve at D converged with |Delta - VALUE| <= TOL. --mutate nowarp: band.py's control (the warped curves
treated as curves of constant x). Exit 1 if a D does not converge, or an --expect fails.

Each D appends a record: Delta, the residual history, the band's step count, min g_p and min g_m (the warped curves'
distance from the characteristics, > 0 by construction), min (c - x) over the band's step points (negative where c < x,
inside the sonic patch), the time, and the kernel used.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
HP = os.path.normpath(os.path.join(HERE, "..", "..", "hp"))
sys.path.insert(0, HERE)
sys.path.insert(0, HP)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

from band import Band, Warp, design_phi, half_rows, node_full  # noqa: E402


def lagrange(Ds, zs, D):
    out = np.zeros_like(zs[0])
    for i, (Di, zi) in enumerate(zip(Ds, zs)):
        w = 1.0
        for j, Dj in enumerate(Ds):
            if j != i:
                w *= (D - Dj) / (Di - Dj)
        out = out + w * zi
    return out


class Hybrid:
    def __init__(self, D, M, nodes, kt, kc, ks, segs, band, phi, threads, kappa, mutate="none", full_band_plan=False):
        from hpkernel.driver import make_solver
        self.D, self.M, self.nodes = D, M, list(nodes)
        self.ia = self.nodes.index(band[0])
        self.ib = self.ia + 1
        if abs(self.nodes[self.ib] - band[1]) > 1e-15:
            raise SystemExit(f"band {band}: not two consecutive nodes of {nodes}")
        segs = [list(s) for s in segs]
        if not full_band_plan:
            segs[self.ia] = [band[0], 0.5 * (band[0] + band[1]), band[1]]   # its rows are replaced
        self.s = make_solver(D, M, nodes=self.nodes, kt=kt, kc=kc, ks=ks, threads=threads)
        self.s.set_segs(segs)
        self.segs = segs
        self.band = Band(D, 2 * M, Warp(band[0], band[1], phi), mutate=mutate)
        self.kappa = kappa
        self.rows = slice(4 * M * self.ib, 4 * M * (self.ib + 1))
        self.pts = None
        self.last = {}

    def band_rows(self, z):
        M = self.M
        sA, sB = node_full(z, M, self.ia), node_full(z, M, self.ib)
        om = 2 * np.pi / z[-1]
        prof = []
        if self.pts is None:
            self.pts, end, gp, gm = self.band.plan(sA, om, kappa=self.kappa, record=prof)
        else:
            end, gp, gm = self.band.run(sA, om, self.pts, record=prof)
        x = np.array([self.band.warp.at(zz)[0] for zz, _ in prof])
        nat = float(min(np.min(st[3] - xx) for (zz, st), xx in zip(prof, x)))
        self.last = dict(band_steps=len(self.pts) - 1, min_gp=gp, min_gm=gm, min_c_minus_x=nat)
        return half_rows(end) - half_rows(sB)

    def residual(self, zh, zl):
        Fh, Fl = self.s.residual(zh, zl)
        Fh, Fl = np.array(Fh, float), np.array(Fl, float)
        Fh[self.rows] = self.band_rows(zh + zl)
        Fl[self.rows] = 0.0
        return Fh, Fl

    def band_jacobian(self, z, eps=1e-7, chunk=None):
        """d(Phi_warp(s_A))/d(s_A half-grid unknowns, Delta) by forward differences on the band's plan, in batches of
        chunk columns: (4M, 4M + 1)"""
        M = self.M
        sA = node_full(z, M, self.ia)
        base = half_rows(sA)
        D = z[-1]
        hD = eps * max(1.0, abs(D))
        h = eps * np.maximum(1.0, np.abs(base))
        ncol = 4 * M + 1
        chunk = chunk or ncol
        G = np.zeros((4 * M, ncol))
        end0, _, _ = self.band.run(sA, 2 * np.pi / D, self.pts)
        r0 = half_rows(end0)
        odd = np.array([True, True, False, False])
        for c0 in range(0, ncol, chunk):
            cols = list(range(c0, min(ncol, c0 + chunk)))
            S = np.repeat(sA[None], len(cols), axis=0)
            om = np.full((len(cols), 1), 2 * np.pi / D)
            for b, j in enumerate(cols):
                if j == 4 * M:
                    om[b, 0] = 2 * np.pi / (D + hD)
                    continue
                f, k = divmod(j, M)
                S[b, f, k] += h[j]
                S[b, f, k + M] += -h[j] if odd[f] else h[j]
            end, _, _ = self.band.run(S, om, self.pts)
            for b, j in enumerate(cols):
                G[:, j] = (half_rows(end[b]) - r0) / (hD if j == 4 * M else h[j])
        return G

    def assemble(self, J, G):
        """J with the band rows replaced: G on s_A's columns and Delta's, -I on s_B's, 0 elsewhere"""
        M = self.M
        r = self.rows
        J[r, :] = 0.0
        oA = 2 * M + 4 * M * self.ia
        oB = 2 * M + 4 * M * self.ib
        J[r, oA:oA + 4 * M] = G[:, :4 * M]
        J[r, -1] = G[:, 4 * M]
        J[r, oB:oB + 4 * M] -= np.eye(4 * M)
        return J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--state", required=True)
    ap.add_argument("--history", default="")
    ap.add_argument("--Ds", required=True)
    ap.add_argument("--band", default="0.01,0.02")
    ap.add_argument("--warp", default="design")
    ap.add_argument("--A", type=float, default=1.5e-3)
    ap.add_argument("--eps0", type=float, default=2e-3)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--it_max", type=int, default=40)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--drop_nodes", default="")
    ap.add_argument("--add_nodes", default="")
    ap.add_argument("--save", default="")
    ap.add_argument("--expect", default="")
    ap.add_argument("--mutate", default="none")
    ap.add_argument("--jac", default="band", choices=["band", "hp"])
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--time_cap", type=float, default=6 * 3600)
    ap.add_argument("--allow_unraw", action="store_true")
    a = ap.parse_args()
    from raw_out import require_raw
    require_raw(a.out, os.path.join(a.save, "x") if a.save else None, allow=a.allow_unraw)
    import hpkernel
    from hpkernel.driver import dd_add, load, make_solver, renode

    t0 = time.time()
    log = lambda *m: print(f"[{time.time() - t0:8.1f}s]", *m, flush=True)
    log("hpkernel", hpkernel.__file__)
    zh, zl, meta = load(a.state)
    D0, M, nodes = float(meta["D"]), int(meta["M"]), list(meta["nodes"])
    kt, kc, ks = meta.get("kt", 36), meta.get("kc", 100), meta.get("ks", 250)
    segs = meta.get("segs")
    band = tuple(float(v) for v in a.band.split(","))
    hist = []
    for f in [h for h in a.history.split(",") if h]:
        hh, hl, hm = load(f)
        if hm["M"] != M or list(hm["nodes"]) != nodes:
            raise SystemExit(f"{f}: M or nodes differ from the start state's")
        hist.append((float(hm["D"]), hh + hl))
    s0 = make_solver(D0, M, nodes=nodes, kt=kt, kc=kc, ks=ks, threads=a.threads)
    if segs is None:
        segs = s0.plan(zh, zl, 1e-30, 0.05)
    s0.set_segs(segs)
    if a.drop_nodes or a.add_nodes:
        drop = [float(v) for v in a.drop_nodes.split(",") if v]
        add = [float(v) for v in a.add_nodes.split(",") if v]
        new = sorted(set(x for x in nodes if x not in drop) | set(add))
        zh, zl = renode(s0, zh, zl, nodes, new, M)
        hist2 = []
        for (Dh, zv), f in zip(hist, [h for h in a.history.split(",") if h]):
            hh, hl, hm = load(f)
            sh = make_solver(Dh, M, nodes=nodes, kt=kt, kc=kc, ks=ks, threads=a.threads)
            sh.set_segs(hm.get("segs") or sh.plan(hh, hl, 1e-30, 0.05))
            rh, rl = renode(sh, hh, hl, nodes, new, M)
            hist2.append((Dh, np.asarray(rh) + np.asarray(rl)))
        hist = hist2
        nodes = new
        s0 = make_solver(D0, M, nodes=nodes, kt=kt, kc=kc, ks=ks, threads=a.threads)
        segs = s0.plan(zh, zl, 1e-30, 0.05)
        s0.set_segs(segs)
        log(f"renoded to {len(nodes)} nodes {nodes}; plan steps {sum(len(q) - 1 for q in segs)}")
    F0 = np.asarray(s0.residual(zh, zl)[0])
    log(f"start D {D0} M {M} K {len(nodes)} L {s0.size}: hp |F| {np.abs(F0).max():.3e}, Delta {zh[-1] + zl[-1]:.16f}")
    N = 2 * M
    th = 2 * np.pi * np.arange(N) / N
    om0 = 2 * np.pi / (zh[-1] + zl[-1])
    if a.warp == "none":
        phi = np.zeros(N)
    elif a.warp == "test":
        phi = a.A * (np.sin(2 * th + 0.3) + 0.3 * np.cos(4 * th - 1.1))
    elif a.warp == "design":
        bd = Band(D0, N, Warp.none(band[0], band[1], N))
        z = zh + zl
        sA = node_full(z, M, nodes.index(band[0]))
        prof = []
        pts, end, gp, gm = bd.plan(sA, om0, kappa=a.kappa, record=prof)
        phi, g = design_phi([(band[0], sA)] + prof, om0, a.eps0, kmax=M // 2)
        log(f"warp designed at D {D0}: unwarped band {len(pts) - 1} steps, g* min {g.min():.4e}, max|phi| "
            f"{np.abs(phi).max():.3e}")
    else:
        phi = np.load(a.warp)
    if a.save:
        os.makedirs(a.save, exist_ok=True)
        np.save(os.path.join(a.save, "phi.npy"), phi)
    tj = time.time()
    if a.jac == "hp":
        log(f"Jacobian at D {D0}: hp's f64 Jacobian with its own plan, LU kept for the chord steps")
        s0.newton_mixed(zh, zl)
        solve = lambda Fh, Fl: tuple(np.asarray(v) for v in s0.lu_solve(-Fh, -Fl))
    else:
        import scipy.linalg as sla
        log(f"Jacobian at D {D0}: hp's f64 Jacobian, its band rows replaced by the warped band's (forward differences)")
        hy0 = Hybrid(D0, M, nodes, kt, kc, ks, segs, band, phi, a.threads, a.kappa, a.mutate)
        s0 = None
        J = np.asarray(hy0.s.jacobian_f64(zh, zl))
        log(f"  hp Jacobian (short band plan): {time.time() - tj:.0f}s, {J.nbytes / 1e9:.2f} GB")
        hy0.band_rows(zh + zl)
        tb = time.time()
        G = hy0.band_jacobian(zh + zl, chunk=a.chunk)
        log(f"  band Jacobian: {time.time() - tb:.0f}s ({hy0.last})")
        J = hy0.assemble(J, G)
        del G
        tl = time.time()
        LU = sla.lu_factor(J.T, overwrite_a=True, check_finite=False)     # J.T is Fortran-ordered: factored in place
        del J
        dU = np.abs(np.diag(LU[0]))
        log(f"  LU: {time.time() - tl:.0f}s, |u_kk| {dU.min():.3e}..{dU.max():.3e}")
        solve = lambda Fh, Fl: (-sla.lu_solve(LU, Fh, trans=1, check_finite=False), np.zeros_like(Fh))
    log(f"  Jacobian and LU: {time.time() - tj:.0f}s")
    expects = {}
    for e in [e for e in a.expect.split(",") if e]:
        d_, v_, t_ = e.split(":")
        expects[float(d_)] = (float(v_), float(t_))
    hist.append((D0, zh + zl))
    ok_all = True
    for D in [float(v) for v in a.Ds.split(",")]:
        if time.time() - t0 > a.time_cap:
            log("time cap reached; stopping")
            ok_all = False
            break
        td = time.time()
        near = sorted(hist, key=lambda h: abs(h[0] - D))[:4]
        zp = near[0][1].copy() if near[0][0] == D else lagrange([h[0] for h in near], [h[1] for h in near], D)
        zh_, zl_ = zp.copy(), np.zeros_like(zp)
        hy = Hybrid(D, M, nodes, kt, kc, ks, segs, band, phi, a.threads, a.kappa, a.mutate)
        hist_F, conv, it = [], False, 0
        err = None
        try:
            for it in range(a.it_max):
                Fh, Fl = hy.residual(zh_, zl_)
                nF = float(np.abs(Fh).max())
                if not np.isfinite(nF):
                    raise FloatingPointError("residual not finite")
                xh, xl = solve(Fh, Fl)
                zh_, zl_ = dd_add(zh_, zl_, np.asarray(xh), np.asarray(xl))
                zh_, zl_ = np.asarray(zh_), np.asarray(zl_)
                dD = float(np.asarray(xh)[-1])
                hist_F.append((nF, dD))
                log(f"  D {D} it {it}: |F| {nF:.3e} (band rows {np.abs(Fh[hy.rows]).max():.3e}), dDelta {dD:+.3e}, "
                    f"band {hy.last}")
                if nF < a.tol and abs(dD) < a.tol:
                    conv = True
                    break
                if it >= 6 and nF > 0.7 * hist_F[-4][0] and nF > 1e3 * a.tol:
                    log("  chord stalled")
                    break
        except (FloatingPointError, RuntimeError) as e:
            err = str(e)
            log(f"  D {D}: {err}")
        Delta = float(zh_[-1] + zl_[-1])
        rec = dict(D=D, M=M, nodes=nodes, band=band, warp=a.warp, eps0=a.eps0, A=a.A, kappa=a.kappa, mutate=a.mutate,
                   converged=conv, its=it + 1, Delta=Delta, Delta_hi=float(zh_[-1]), Delta_lo=float(zl_[-1]),
                   F_hist=hist_F, error=err, seconds=round(time.time() - td, 1), D_start=D0,
                   predictor_Delta=float(zp[-1]), max_phi=float(np.abs(phi).max()), hpkernel=hpkernel.__file__,
                   **hy.last)
        if D in expects:
            v, tol_ = expects[D]
            rec["expect"] = v
            rec["expect_tol"] = tol_
            rec["expect_ok"] = bool(conv and abs(Delta - v) <= tol_)
            log(f"  D {D}: Delta {Delta:.16f}, expected {v} +- {tol_}: {'OK' if rec['expect_ok'] else 'FAIL'} "
                f"(diff {Delta - v:+.3e})")
            ok_all &= rec["expect_ok"]
        with open(a.out, "a") as fo:
            fo.write(json.dumps(rec) + "\n")
        log(f"D {D}: converged {conv} its {it + 1} Delta {Delta:.16f} band {hy.last}")
        if not conv:
            ok_all = False
            break
        if a.save:
            m2 = dict(meta)
            m2.update(D=D, M=M, nodes=nodes, segs=hy.segs, band=list(band), warp=a.warp, band_pts=list(hy.pts))
            np.savez(os.path.join(a.save, f"hyb_D{D:.6f}_M{M}.npz"), zh=zh_, zl=zl_, meta=json.dumps(m2))
        hist = [h for h in hist if h[0] != D] + [(D, zh_ + zl_)]
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Re-seeding the horizon data from the last node (notes/RESEED.md; the toy study is notes/QUAD_DOUBLE.md 13.21).

The solver's horizon data (m_h, a_h, 2M unknowns) enter the residual through the horizon series summed at u0 = 1 - x_K-1,
where the mass mode's gain G = x_K-1^-(n-1) turns an error delta in a_h into G delta at the node. Re-seeding replaces
them, before the corrector, by those of the regular solution through the state's last node: the node is integrated
toward the horizon with the kernel's Taylor stepper to u0 / 2^J, and the horizon series is inverted at the rungs
u0 / 2^J, ..., u0 / 2, u0 in turn, each Newton starting from the last rung's data. A rung's target is the integrated
state there (the node itself at u0), and its equations are the m and a rows of the horizon block (2M equations in the
2M unknowns; notes/RESEED.md 2 gives the reasons). Delta, the centre data and the nodes are not changed.

    python3 reseed.py check STATE.npz [--threads 2] [--levels 5] [--rows ma|pm|all] [--mutate none|skip|wrongu]
        [--plan_tol 1e-32] [--out OUT.jsonl] [--delta_h 1e-6] [--seed 1] [--floor_rule stall|abs|rel]
    python3 reseed.py jac STATE.npz
    python3 reseed.py checkq PRED.npz [--threads 4] [--legacy 1] [--rule136 1] [--sweeps 1]
    python3 reseed.py truthq STATE.npz [--threads 4] [--sweeps 1] [--delta 1e-32] [--seed 1]
    python3 reseed.py checkstart PRED.npz [--threads 4] [--start_F 1e-6] [--budget 900]
    python3 reseed.py projlog [--mutate=silent|stage]

`check` is the unit check and the stepper check (notes/RESEED.md 3.1, 3.2): the converged state's own node re-seeded
from horizon data off by delta_h (relative, random signs) must return the state's horizon data to the floor of 3.1,
and the node integrated to each rung must agree with the state's horizon series there (3.2); exit 1 otherwise. The
mutations must make 3.1 fail: skip (only the last rung, u0), wrongu (each rung's series summed at 1.001 u against the
target at u); --plan_tol 1e-12 must make 3.2 fail. `jac` prints the singular values of the horizon block's m_h, a_h
columns by row groups (notes/RESEED.md 2.1). `checkq` re-seeds a stored prediction (anm_run.py --save_pred 1) with
the quad-double polish and must converge; --legacy 1, the earlier polish rule, must make it fail where that rule fell
short (its docstring), and --sweeps 1 past D 210. `truthq` plants an error in a converged state's horizon data and
the re-seed must remove it (its docstring). `checkstart` runs the corrector on a stored prediction after its re-seed
and must end within a budget; --start_F 0 must make it fail on a prediction whose start projection fails (its
docstring). `projlog` checks the log line of a failed projection (`anm_run.py --log_proj 1`, #743) on stubs, and its
mutations silent and stage must make it fail.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402
from hpkernel.driver import make_solver, load_q, qparts, qmax  # noqa: E402
from multidouble import QDLu, dd_add, dd_solve, qd_add_dd, qd_axpy  # noqa: E402

# refine_o: the corrections allowed past the requested steps at most, the least reduction of |R| per correction under
# which the rung block's LU is factored again, and the |R| at the arithmetic's floor, below which it is not
OD_EXTRA, OD_REFACTOR, OD_FLOOR = 6, 1e-4, 1e-80

mp.mp.dps = 50


def offsets(M, K):
    """z offsets of the layout (hpkernel.driver's docstring): node i, m_h, a_h, and the length"""
    return dict(node=lambda i: 2 * M + 4 * M * i, mh=2 * M + 4 * M * K, ah=3 * M + 4 * M * K, L=4 * M * (K + 1) + 1)


def cols_least_squares(B, R, lo, hi, dps=90):
    """the correction (four parts, the block's width) minimizing |B x + R| over the block's columns lo..hi-1 with the
    others zero, by the normal equations in mpmath at dps digits (B, R in four parts). The rung block's a_h columns
    have singular values all about the mass mode's gain and its m_h columns all about 1, so each set alone is well
    conditioned (notes/QUAD_DOUBLE.md 13.27)."""
    n, w = len(R[0]), B[0].shape[1]
    with mp.workdps(dps):
        A = mp.matrix(n, hi - lo)
        for i in range(n):
            for j in range(hi - lo):
                A[i, j] = mp.mpf(float(B[0][i, lo + j])) + mp.mpf(float(B[1][i, lo + j])) + mp.mpf(float(B[2][i, lo + j])) + mp.mpf(float(B[3][i, lo + j]))
        r = mp.matrix([mp.mpf(float(R[0][i])) + mp.mpf(float(R[1][i])) + mp.mpf(float(R[2][i])) + mp.mpf(float(R[3][i])) for i in range(n)])
        xs = mp.lu_solve(A.T * A, -(A.T * r))
        out = [np.zeros(w) for _ in range(4)]
        for j in range(hi - lo):
            v = xs[j]
            for q in range(4):
                f = float(v)
                out[q][lo + j] = f
                v = v - mp.mpf(f)
    return out


class Reseeder:
    """The re-seed for one layout and D. meta: a stored state's meta (M, nodes, kt, kc, ks); D an mpmath number.
    levels J: the first rung is u0 / 2^J. rows: 'ma' (the m and a rows of the horizon block, the default), 'pm' (p and
    m) or 'all' (least squares over the 4M rows, through the augmented system).
    plan_tol, hmax: the stepper's plan toward the horizon (model.rs plan_path: the truncation term of each step at
    about plan_tol, steps at most hmax). check_tol, check_kt: the second integration whose difference from the first
    is the stepper's error estimate (a plan at check_tol with steps at most hmax / 2, at Taylor order check_kt)."""

    def __init__(self, meta, D, threads=2, levels=5, rows="ma", plan_tol=1e-32, hmax=0.05, check_tol=1e-36,
                 check_kt=None, it_max=30, horizon_alpha=0, mutate="none", floor_rule="stall", qd=False, od=0,
                 qd_start="ladder"):
        self.M, self.nodes = int(meta["M"]), list(meta["nodes"])
        self.K = len(self.nodes)
        self.kt, self.kc, self.ks = meta.get("kt", 36), meta.get("kc", 146), meta.get("ks", 130)
        self.D, self.threads, self.levels, self.rows = mp.mpf(D), threads, levels, rows
        self.plan_tol, self.hmax, self.check_tol = plan_tol, hmax, check_tol
        self.check_kt = check_kt or self.kt + 12
        self.it_max, self.horizon_alpha, self.mutate = it_max, horizon_alpha, mutate
        # the rung's floor test: 'abs' (the design: a step below 1e-24 with the residual no longer halving); 'stall'
        # (amendment 10: also the residual no longer halving with the step no longer shrinking tenfold, or no trial
        # lowering the residual after such a step; the steps stall at the floor eps_dd G above 1e-24 past D ~ 95); 'rel'
        # (amendment 9, withdrawn: also a residual below 1e-29 G_u that stops halving; it stops the u0 rung before its
        # data converge and fails the unit check at D 101.1, notes/RESEED.md), kept runnable as a control
        self.floor_rule = floor_rule
        # qd: after the double-double ladder, the u0 rung polished in quad-double (invert_q): the quad-double residual
        # against the node's four parts, steps from the double-double Jacobian, so that the horizon data agree with the
        # node to the quad-double floor eps_qd G rather than eps_dd G (the double-double rung's floor exceeds what the
        # corrector tolerates past D ~ 128, notes/QUAD_DOUBLE.md 13.22)
        self.qd = qd
        # qd_start (#693): "ladder" polishes the double-double ladder's u0 data in quad-double, as before; "pred" first
        # polishes the point's own four-part horizon data (a prediction carries a_h to about 1e-39, the ladder's data
        # are rounded to double-double, about 1e-32 off, whose pole lies inside u0 past D ~ 249), and runs the ladder
        # only when that does not converge
        self.qd_start = qd_start
        # od: the number of octo-double corrections of the horizon data after the quad-double polish (refine_o), taken
        # where the data are put on an eight-part point (octo_corrector.with_horizon); 0: none
        self.od = int(od)
        self.x_node = self.nodes[-1]
        u0 = 1.0 - self.x_node
        # the rungs u0 / 2^j, j = J..0 (exact in f64: u0 is the f64 value 1 - x_K-1 and the divisions are by powers of 2)
        self.us = [u0 / 2 ** j for j in range(levels, -1, -1)]
        if mutate == "skip":
            self.us = [u0]
        # the mutation wrongu: the rung's series is summed at (1 + 1e-3) u, its target integrated to u
        self.u_scale = 1.001 if mutate == "wrongu" else 1.0
        self.off = offsets(self.M, self.K)
        self.aoff = offsets(self.M, 2)
        self._aux = {}

    # ------------------------------------------------------------------ solvers
    def solver(self, nodes, kt=None):
        """a solver of the same D, M and orders on the given nodes, without the freeze projection (the p rows are the
        plain mismatch) and with one Taylor step per segment (its rows are not used)"""
        key = (tuple(nodes), kt or self.kt)
        if key not in self._aux:
            s = make_solver(self.D, self.M, nodes=list(nodes), kt=kt or self.kt, kc=self.kc, ks=self.ks, threads=self.threads,
                            kfreeze=None, horizon_alpha=self.horizon_alpha)
            s.set_segs([[nodes[i], nodes[i + 1]] for i in range(len(nodes) - 1)])
            self._aux[key] = s
        return self._aux[key]

    def rung_solver(self, u):
        """nodes (x_0, 1 - u): block 2, the horizon block, is the horizon series at u minus node 1"""
        return self.solver((self.nodes[0], 1.0 - u * self.u_scale))

    # ------------------------------------------------------------------ the stepper toward the horizon
    def integrate(self, zh, zl, sh, sl, xs_to, tol, hmax, kt, x_start=None):
        """the state (sh, sl) at the last node carried to each x in xs_to (increasing), with Delta from z: per piece
        a plan (model.rs plan_path, from a two-node solver) and the kernel's Taylor steps along it, step by step.
        Returns the states at xs_to and per piece the steps' x points and max (x - c) over tau at each of them."""
        M = self.M
        out, info = [], []
        x = self.x_node if x_start is None else x_start
        for xb in xs_to:
            s = self.solver((x, xb), kt=kt)
            a = offsets(M, 2)
            zz_h = np.zeros(a["L"]); zz_l = np.zeros(a["L"])
            zz_h[a["node"](0):a["node"](0) + 4 * M] = sh; zz_l[a["node"](0):a["node"](0) + 4 * M] = sl
            zz_h[-1], zz_l[-1] = zh[-1], zl[-1]
            xs = s.plan(zz_h, zz_l, tol, hmax)[0]
            xmc = []
            for i in range(len(xs) - 1):
                sh, sl = (np.asarray(v) for v in s.integrate(zh, zl, sh, sl, [xs[i], xs[i + 1]]))
                c = sh[3 * M:4 * M] + sl[3 * M:4 * M]
                xmc.append(float(np.max(xs[i + 1] - c)))
            out.append((sh.copy(), sl.copy()))
            info.append(dict(x=xb, steps=len(xs) - 1, h_min=float(np.min(np.diff(xs))), max_x_minus_c=max(xmc)))
            x = xb
            if max(xmc) >= 0.0 or not np.all(np.isfinite(sh)):
                break                    # the sonic point x = c crossed (or the state not finite): no rungs beyond
        return out, info

    def targets(self, zh, zl):
        """the rung targets: the last node integrated to x = 1 - u for the rungs u < u0 (the node itself at u0), the
        stepper's error estimate at each (max |difference| from the second integration), and the plan records"""
        M, o = self.M, self.off
        sh = np.array(zh[o["node"](self.K - 1):o["node"](self.K - 1) + 4 * M])
        sl = np.array(zl[o["node"](self.K - 1):o["node"](self.K - 1) + 4 * M])
        inner = sorted([u for u in self.us if u < self.us[-1]], reverse=True)
        xs_to = [1.0 - u for u in inner]       # increasing x
        T, info = self.integrate(zh, zl, sh, sl, xs_to, self.plan_tol, self.hmax, self.kt)
        T2, _ = self.integrate(zh, zl, sh, sl, xs_to[:len(T)], self.check_tol, self.hmax / 2, self.check_kt)
        for u, rec in zip(inner, info):
            rec["u"] = u
        for t, t2, rec in zip(T, T2, info):
            rec["err_est"] = float(np.max(np.abs((t[0] - t2[0]) + (t[1] - t2[1])))) if len(t2) else float("nan")
        ok = [i for i, rec in enumerate(info) if rec["max_x_minus_c"] < 0.0 and np.all(np.isfinite(T[i][0]))]
        n_ok = 0
        while n_ok < len(info) and n_ok in ok:
            n_ok += 1
        tg = {u: T[i] for i, u in enumerate(inner[:n_ok])}
        tg[self.us[-1]] = (sh, sl)
        return tg, info

    # ------------------------------------------------------------------ the inversion at a rung
    def rung_rows(self):
        M = self.M
        r0 = 8 * M                                    # block 2 of the two-node layout
        if self.rows == "ma":
            return np.r_[r0 + M:r0 + 3 * M]
        if self.rows == "pm":
            return np.r_[r0:r0 + 2 * M]
        return np.r_[r0:r0 + 4 * M]

    def series_at(self, u, zh, zl, dh, dl):
        """the horizon series of the data (dh, dl) summed at u (4M, hi and lo), with Delta from z"""
        M, a = self.M, self.aoff
        s = self.solver((self.nodes[0], 1.0 - u))
        vh = np.zeros(a["L"]); vl = np.zeros(a["L"])
        vh[:2 * M], vl[:2 * M] = zh[:2 * M], zl[:2 * M]
        vh[a["node"](0):a["node"](1)] = zh[self.off["node"](0):self.off["node"](1)]
        vl[a["node"](0):a["node"](1)] = zl[self.off["node"](0):self.off["node"](1)]
        vh[a["mh"]:a["mh"] + 2 * M], vl[a["mh"]:a["mh"] + 2 * M] = dh, dl
        vh[-1], vl[-1] = zh[-1], zl[-1]
        Fh, Fl = (np.asarray(v) for v in s.residual(vh, vl))
        return Fh[8 * M:12 * M], Fl[8 * M:12 * M]

    def invert(self, u, zh, zl, th, tl, dh, dl):
        """Newton on the 2M horizon data (dh, dl: m_h then a_h) so that the horizon series at u matches the target
        (th, tl) in the rung rows; damped by halving when the max norm of those rows does not fall (a trial whose
        series is not finite is rejected). Returns the data, the history and whether it converged (step below 1e-30
        times the data's scale, or the residual at its floor: no decrease by 2 after a step below 1e-24, or no decrease
        by 2, or no trial accepted, with the residual below floor_tol = 1e-29 (1 - u)^-(n-1), with floor_rule 'rel'
        only: amendment 9, withdrawn, notes/RESEED.md)."""
        M, a = self.M, self.aoff
        s = self.rung_solver(u)
        rows = self.rung_rows()
        cols = np.r_[a["mh"]:a["mh"] + 2 * M]
        base_h = np.zeros(a["L"]); base_l = np.zeros(a["L"])
        base_h[:2 * M], base_l[:2 * M] = zh[:2 * M], zl[:2 * M]                     # centre data (block 0 not used)
        base_h[a["node"](0):a["node"](1)] = zh[self.off["node"](0):self.off["node"](1)]
        base_l[a["node"](0):a["node"](1)] = zl[self.off["node"](0):self.off["node"](1)]
        base_h[a["node"](1):a["node"](1) + 4 * M], base_l[a["node"](1):a["node"](1) + 4 * M] = th, tl
        base_h[-1], base_l[-1] = zh[-1], zl[-1]

        def put(ch, cl):
            vh, vl = base_h.copy(), base_l.copy()
            vh[cols], vl[cols] = ch, cl
            return vh, vl

        def res(ch, cl):
            Fh, Fl = (np.asarray(v) for v in s.residual(*put(ch, cl)))
            return Fh[rows], Fl[rows]

        hist = []
        Rh, Rl = res(dh, dl)
        nR = float(np.max(np.abs(Rh + Rl)))
        if not np.isfinite(nR):
            return dh, dl, [dict(it=0, R=nR)], False
        conv = False
        small_prev = None
        floor_tol = 1e-29 * (1.0 - u) ** -(float(self.D) - 3.0) if self.floor_rule == "rel" else -1.0
        stall = self.floor_rule == "stall"
        nd_prev = None
        for it in range(self.it_max):
            Jh, Jl = (np.asarray(v) for v in s.jacobian(*put(dh, dl)))
            Ah = np.ascontiguousarray(Jh[np.ix_(rows, cols)]); Al = np.ascontiguousarray(Jl[np.ix_(rows, cols)])
            if len(rows) == len(cols):
                xh, xl = dd_solve(Ah, Al, np.ascontiguousarray(-Rh), np.ascontiguousarray(-Rl))
            else:
                # least squares through the augmented system [[I, A], [A^T, 0]] [r; x] = [-R; 0]
                nr, nc = len(rows), len(cols)
                Bh = np.zeros((nr + nc, nr + nc)); Bl = np.zeros_like(Bh)
                Bh[:nr, :nr] = np.eye(nr); Bh[:nr, nr:] = Ah; Bl[:nr, nr:] = Al; Bh[nr:, :nr] = Ah.T; Bl[nr:, :nr] = Al.T
                yh, yl = dd_solve(Bh, Bl, np.r_[-Rh, np.zeros(nc)], np.r_[-Rl, np.zeros(nc)])
                xh, xl = np.asarray(yh)[nr:], np.asarray(yl)[nr:]
            xh, xl = np.asarray(xh), np.asarray(xl)
            nd = float(np.max(np.abs(xh + xl)))
            lam, ok = 1.0, False
            for _ in range(12):
                th_, tl_ = dd_add(dh, dl, lam * xh, lam * xl)
                Th, Tl = res(th_, tl_)
                nT = float(np.max(np.abs(Th + Tl)))
                if np.isfinite(nT) and (nT < nR or nd < 1e-24):
                    ok = True
                    break
                lam *= 0.5
            hist.append(dict(it=it, R=nR, step=nd, lam=lam if ok else 0.0, R_new=nT if ok else None))
            if not ok:
                conv = nR <= floor_tol          # no trial lowers a residual already at the floor
                if stall and nd_prev is not None and nd > 0.1 * nd_prev:
                    conv = True                 # 'stall': no trial lowers the residual, and the step stopped shrinking
                break
            dh, dl, Rh, Rl = np.asarray(th_), np.asarray(tl_), Th, Tl
            if nd * lam < 1e-30 * max(1.0, float(np.max(np.abs(dh)))):
                nR = nT
                conv = True
                break
            if nd < 1e-24:
                if small_prev is not None and nT > 0.5 * small_prev:
                    nR = nT
                    conv = True              # the floor: the residual no longer falls
                    break
                small_prev = nT
            if nT <= floor_tol and nT > 0.5 * nR:
                nR = nT
                conv = True                  # at the floor and no longer halving
                break
            if stall and nd_prev is not None and nd > 0.1 * nd_prev and nT > 0.5 * nR:
                nR = nT
                conv = True                  # 'stall': the residual no longer halves and the step no longer shrinks
                break
            nd_prev = nd
            nR = nT
        hist.append(dict(it=len(hist), R=nR))
        return dh, dl, hist, conv

    def invert_q(self, z4, d4, it_max=16, refactor_max=3, R_stop=1e-34, pre="am", take_all=True, rise_max=1e12,
                 sweeps=3):
        """the u0 rung in quad-double (polish_q), run again from d4 with the block steps of pre repeated while it does
        not converge: pre once, then twice, up to sweeps times. Past D 209 one pass of the a_h and m_h steps leaves the
        iterate outside Newton's basin, and a second pass brings it in (#446). An iteration that converges on its
        first run is unchanged, and sweeps=1 never repeats. Returns the last run's d4, history and whether it
        converged."""
        for k in range(1, max(1, sweeps) + 1):
            d, hist, conv = self.polish_q(z4, d4, it_max, refactor_max, R_stop, pre * k, take_all, rise_max)
            if conv or not pre:
                break
        return d, hist, conv

    def polish_q(self, z4, d4, it_max=16, refactor_max=3, R_stop=1e-34, pre="am", take_all=True, rise_max=1e12):
        """the u0 rung in quad-double: Newton on the 2M horizon data d4 (four parts) so that the horizon series at u0
        matches the last node of z4 (four parts) in the rung rows. The residual is Solver.residual_q and the step solves
        the rung rows' block of the quad-double Jacobian (Solver.jacobian_q) with multidouble's QDLu: the double-double
        Jacobian's own rounding, amplified by the mass mode, stops a double-double-preconditioned iteration near
        1e-19 at D 128. The Jacobian is factored at the start and again, at most refactor_max times, where a step does
        not halve the residual. Stops below R_stop, when a step after the last refactoring does not halve the residual
        (its floor, about eps_qd G), or after it_max steps. Past D ~ 150 the iteration with one factoring needs about 12
        steps from the double-double rung and stalls near 1e-14 at some points (notes/QUAD_DOUBLE.md 13.26);
        pre: least-squares steps before Newton in the block's a_h ("a") then m_h ("m") columns alone; take_all: Newton
        steps refactored at every step and taken whether or not they lower |R|, the best iterate kept, instead of the
        rule above (refactor_max), and stopped once |R| exceeds rise_max times the best (the first Newton step after
        the block steps can raise it by 1e6 or more before it falls quadratically). Past D 160 neither plain Newton nor that rule converges from the double-double
        rung, and the a_h and m_h steps followed by Newton do (the first Newton step may raise |R| before it falls
        quadratically; notes/QUAD_DOUBLE.md 13.27). pre="", take_all=False is the rule of 13.26, and with it_max=6,
        refactor_max=0 the earlier one. Returns d4, the history and whether it converged (at least one step taken and
        the residual below 1e-24)."""
        M, a, o = self.M, self.aoff, self.off
        u = self.us[-1]
        s = self.rung_solver(u)
        rows = self.rung_rows()
        cols = np.r_[a["mh"]:a["mh"] + 2 * M]
        L = a["L"]
        base = [np.zeros(L) for _ in range(4)]
        n0, n1 = o["node"](0), o["node"](1)
        nk = o["node"](self.K - 1)
        for q in range(4):
            zq = np.asarray(z4[q], dtype=float)
            base[q][:2 * M] = zq[:2 * M]
            base[q][a["node"](0):a["node"](1)] = zq[n0:n1]
            base[q][a["node"](1):a["node"](1) + 4 * M] = zq[nk:nk + 4 * M]
            base[q][-1] = zq[-1]

        def put(d):
            v = [b.copy() for b in base]
            for q in range(4):
                v[q][cols] = d[q]
            return tuple(np.ascontiguousarray(x) for x in v)

        def res(d):
            F = s.residual_q(*put(d))
            return [np.asarray(F[q])[rows] for q in range(4)]

        d = [np.ascontiguousarray(np.asarray(x, dtype=float)) for x in d4]
        R = res(d)
        nR = float(np.max(np.abs(R[0] + R[1])))
        hist = [dict(it=0, R=nR)]
        if not np.isfinite(nR):
            return d, hist, False
        def block(d):
            J = s.jacobian_q(*put(d))
            return [np.asarray(J[q])[np.ix_(rows, cols)] for q in range(4)]           # the rung block

        def factor(d):
            return QDLu(*block(d))

        B = block(d)
        steps, refactors = 0, 0
        for part in pre:
            lo, hi = (M, 2 * M) if part == "a" else (0, M)
            dn = [np.ascontiguousarray(np.asarray(p)) for p in qd_axpy(1.0, *cols_least_squares(B, R, lo, hi), *d)]
            Rn = res(dn)
            nn = float(np.max(np.abs(Rn[0] + Rn[1])))
            hist.append(dict(it=len(hist), R=nn, pre=part, refactors=0))
            if np.isfinite(nn) and nn < nR:
                d, R, nR = dn, Rn, nn
                steps += 1
                B = block(d)
        if take_all:
            best = (nR, d)
            for it in range(it_max):
                if nR < R_stop:
                    break
                x = QDLu(*B).solve(*[-np.asarray(r) for r in R])                 # Newton, the block at d
                dn = [np.ascontiguousarray(np.asarray(p)) for p in qd_axpy(1.0, *x, *d)]
                Rn = res(dn)
                nn = float(np.max(np.abs(Rn[0] + Rn[1])))
                hist.append(dict(it=len(hist), R=nn, refactors=refactors))
                if not np.isfinite(nn) or nn > rise_max * best[0]:
                    break                                   # diverging: keep the best iterate
                d, R, nR = dn, Rn, nn
                steps += 1
                if nR < best[0]:
                    best = (nR, d)
                B = block(d)
                refactors += 1
            nR, d = best
        else:
            lu = QDLu(*B)
            for it in range(it_max):
                x = lu.solve(*[-np.asarray(r) for r in R])                        # J_block x = -R in quad-double
                dn = [np.ascontiguousarray(np.asarray(p)) for p in qd_axpy(1.0, *x, *d)]
                Rn = res(dn)
                nn = float(np.max(np.abs(Rn[0] + Rn[1])))
                hist.append(dict(it=it + 1, R=nn, step=float(np.max(np.abs(np.asarray(x[0])))), refactors=refactors))
                lower = bool(np.isfinite(nn) and nn < nR)
                halving = lower and nn <= 0.5 * nR
                if lower:
                    d, R, nR = dn, Rn, nn                   # a step that does not lower the residual is not taken
                    steps += 1
                if nR < R_stop:
                    break
                if not halving:
                    if refactors >= refactor_max:
                        break                               # no longer halving after the last factoring: the floor
                    lu = factor(d)                          # the Jacobian at the current data
                    refactors += 1
        hist.append(dict(it=len(hist), R=nR, refactors=refactors))     # the accepted residual, as invert() ends
        return d, hist, steps > 0 and nR < 1e-24

    # ------------------------------------------------------------------ the re-seed
    def refine_o(self, z8, steps=None, mutate=""):
        """the u0 rung's horizon data in octo-double: z8 (eight parts) with its 2M horizon data moved by steps Newton
        corrections. The residual is the rung rows of Solver.residual_o at all eight parts, and each step solves the
        rung block of the quad-double Jacobian at the point's quad-double rounding (QDLu, factored once). The
        quad-double polish (polish_q) leaves the data eps_qd from those the node asks for, which the mass mode
        amplifies to about eps_qd G in the corrector's residual, 1e-31 at D 216 and 1e-29 at D 226 (#461); a
        correction lowers that remainder by about eps_qd times the block's condition. The LU is factored again at the
        point reached when a correction lowers |R| by less than OD_REFACTOR, and each LU gets the requested number of
        corrections (at most OD_EXTRA more in all, none when steps is 0). Returns (z8', hist): hist is
        the rung's max |R| before each correction and after the last. A residual that is not finite returns the
        point as it stands. mutate 'sign' applies each correction with the wrong sign (the control of
        octo_corrector.py horizon); 'badlu' factors the first LU with the block's first column times 1.01, which
        the refactoring must recover from, and 'badlu_keep' keeps that LU throughout (its control)."""
        from multidouble import od_axpy
        steps = self.od if steps is None else int(steps)
        M, a, o = self.M, self.aoff, self.off
        s = self.rung_solver(self.us[-1])
        rows = self.rung_rows()
        cols = np.r_[a["mh"]:a["mh"] + 2 * M]
        n0, n1, nk = o["node"](0), o["node"](1), o["node"](self.K - 1)
        sl_ = slice(o["mh"], o["mh"] + 2 * M)
        z8 = [np.array(p, dtype=float) for p in z8]
        # the two-node layout of the rung: the centre data, node 0, the last node as node 1, and Delta
        base = [np.zeros(a["L"]) for _ in range(8)]
        for q in range(8):
            base[q][:2 * M] = z8[q][:2 * M]
            base[q][a["node"](0):a["node"](1)] = z8[q][n0:n1]
            base[q][a["node"](1):a["node"](1) + 4 * M] = z8[q][nk:nk + 4 * M]
            base[q][-1] = z8[q][-1]

        def put(d, parts):
            v = [b.copy() for b in base[:parts]]
            for q in range(parts):
                v[q][cols] = d[q]
            return [np.ascontiguousarray(x) for x in v]

        d = [np.ascontiguousarray(z8[q][sl_]) for q in range(8)]
        zero = np.zeros(2 * M)
        hist, lu, since = [], None, 0
        # Each LU gets the requested number of corrections (since: those made with the current one), at most OD_EXTRA
        # beyond steps in all, and none when steps is 0. A correction that lowers |R| by less than OD_REFACTOR
        # refactors the LU at the point reached, above the floor OD_FLOOR. At some points the first correction from the once-factored LU raised
        # |R| 200 times and the next ones gained 1e-8 each, ending near 1e-40 where the others end near 1e-76 (#461,
        # a 0.0625 at D 242.48).
        last = steps + (OD_EXTRA if steps > 0 else 0)
        for k in range(last + 1):
            F = s.residual_o(put(d, 8))
            R = [np.ascontiguousarray(np.asarray(F[q])[rows]) for q in range(8)]
            hist.append(float(np.max(np.abs(R[0] + R[1]))))
            if k == last or not np.isfinite(hist[-1]) or since >= steps:
                break
            if lu is None or (k > 0 and hist[-1] > OD_REFACTOR * hist[-2] and hist[-1] > OD_FLOOR
                                and mutate != "badlu_keep"):
                J = s.jacobian_q(*put(d, 4))
                blk = [np.array(np.asarray(J[q])[np.ix_(rows, cols)]) for q in range(4)]
                if lu is None and mutate in ("badlu", "badlu_keep"):
                    for q in range(4):
                        blk[q][:, 0] *= 1.01         # the plant: a first LU whose chord steps gain only about 1e-2
                lu, since = QDLu(*[np.ascontiguousarray(b) for b in blk]), 0
            x = [np.ascontiguousarray(np.asarray(p)) for p in lu.solve(*[-r for r in R[:4]])]
            d = [np.ascontiguousarray(np.asarray(p))
                 for p in od_axpy(-1.0 if mutate == "sign" else 1.0, *x, *([zero] * 4), *d)]
            since += 1
        for q in range(8):
            z8[q][sl_] = d[q]
        return tuple(np.ascontiguousarray(p) for p in z8), hist

    def reseed(self, z, verbose=False):
        """z: a state of two or four parts (hi, lo[, z2, z3]). Returns (z', diag): z' has the same parts with m_h and a_h
        replaced (their lower parts zero), diag the per-rung records; diag['ok'] is False when a rung's Newton did not
        converge or a target is not finite (z' is then z unchanged)."""
        t0 = time.time()
        zh, zl = np.asarray(z[0], dtype=float), np.asarray(z[1], dtype=float)
        M, o = self.M, self.off
        pre = []
        if self.qd and len(z) == 4 and self.qd_start == "pred":
            # the quad-double polish from the point's own four-part horizon data (#693); the ladder below only if it fails
            sl_ = slice(o["mh"], o["mh"] + 2 * M)
            tq = time.time()
            d4, hq, cq = self.invert_q(z, tuple(np.ascontiguousarray(np.asarray(p, dtype=float)[sl_]) for p in z))
            pre = [dict(u=self.us[-1], qd=True, start="pred", conv=cq, its=len(hq) - 1, R0=hq[0]["R"], R=hq[-1]["R"],
                        seconds=round(time.time() - tq, 2))]
            if verbose:
                print(f"    rung u0 in quad-double from the point's own data: {len(hq) - 1} its, |R| {hq[0]['R']:.2e} -> "
                      f"{hq[-1]['R']:.2e}, {'converged' if cq else 'NOT converged, the ladder follows'}, "
                      f"{time.time() - tq:.1f} s", flush=True)
            if cq:
                out = [np.array(p, dtype=float) for p in z]
                for q in range(4):
                    out[q][sl_] = d4[q]
                return tuple(np.ascontiguousarray(p) for p in out), dict(us=[self.us[-1]], dropped=[], plan=None, rungs=pre,
                                                                        ok=True, rows=self.rows, levels=self.levels,
                                                                        mutate=self.mutate, start="pred",
                                                                        seconds=round(time.time() - t0, 2))
        tg, info = self.targets(zh, zl)
        self.last_targets = tg
        us = [u for u in self.us if u in tg]
        diag = dict(us=us, dropped=[u for u in self.us if u not in tg], plan=info, rungs=list(pre), ok=True, rows=self.rows,
                    levels=self.levels, mutate=self.mutate)
        dh = np.array(zh[o["mh"]:o["mh"] + 2 * M]); dl = np.array(zl[o["mh"]:o["mh"] + 2 * M])
        d0 = dh + dl
        for u in us:
            th, tl = tg[u]
            if not np.all(np.isfinite(th)):
                diag["ok"] = False
                diag["rungs"].append(dict(u=u, finite=False))
                break
            t1 = time.time()
            dh, dl, hist, conv = self.invert(u, zh, zl, th, tl, dh, dl)
            dv = dh + dl
            rec = dict(u=u, conv=conv, its=len(hist) - 1, R0=hist[0]["R"], R=hist[-1]["R"], seconds=round(time.time() - t1, 2),
                       hist=[dict(R=h_.get("R"), step=h_.get("step"), lam=h_.get("lam")) for h_ in hist],
                       moved_mh=float(np.max(np.abs(dv[:M] - d0[:M]))), moved_ah=float(np.max(np.abs(dv[M:] - d0[M:]))))
            diag["rungs"].append(rec)
            if verbose:
                print(f"    rung u {u:.6f}: {rec['its']} its, |R| {rec['R0']:.2e} -> {rec['R']:.2e}, "
                      f"{'converged' if conv else 'NOT converged'}, data moved m_h {rec['moved_mh']:.1e} a_h {rec['moved_ah']:.1e}, "
                      f"{rec['seconds']} s", flush=True)
            if not conv:
                diag["ok"] = False
                break
        sl_ = slice(o["mh"], o["mh"] + 2 * M)
        if self.qd and len(z) == 4:
            # the ladder's last (u0) rung may stop at its double-double floor; the quad-double polish decides
            if not diag["ok"] and diag["rungs"] and diag["rungs"][-1].get("u") == self.us[-1] and np.all(np.isfinite(dh)):
                diag["ok"] = True
            if diag["ok"]:
                tq = time.time()
                d4, hq, cq = self.invert_q(z, (dh, dl, np.zeros(2 * M), np.zeros(2 * M)))
                diag["rungs"].append(dict(u=self.us[-1], qd=True, conv=cq, its=len(hq) - 1, R0=hq[0]["R"], R=hq[-1]["R"],
                                          seconds=round(time.time() - tq, 2)))
                if verbose:
                    print(f"    rung u0 in quad-double: {len(hq) - 1} its, |R| {hq[0]['R']:.2e} -> {hq[-1]['R']:.2e}, "
                          f"{'converged' if cq else 'NOT converged'}, {time.time() - tq:.1f} s", flush=True)
                diag["ok"] = cq
                if cq:
                    diag["seconds"] = round(time.time() - t0, 2)
                    out = [np.array(p, dtype=float) for p in z]
                    for q in range(4):
                        out[q][sl_] = d4[q]
                    return tuple(np.ascontiguousarray(p) for p in out), diag
        diag["seconds"] = round(time.time() - t0, 2)
        if not diag["ok"]:
            return tuple(np.array(p) for p in z), diag
        out = [np.array(p, dtype=float) for p in z]
        out[0][sl_], out[1][sl_] = dh, dl
        for p in out[2:]:
            p[sl_] = 0.0
        return tuple(np.ascontiguousarray(p) for p in out), diag


    def project(self, z):
        """z (four parts) with its horizon data re-inverted at u0 only, from its own data (a warm start for points near
        the regular family: the corrector's iterates); the full ladder (reseed) when that Newton does not converge.
        Returns (z', how): how is 'u0', 'ladder' or 'failed' (z' = z)."""
        zh, zl = np.asarray(z[0], dtype=float), np.asarray(z[1], dtype=float)
        M, o = self.M, self.off
        sl_ = slice(o["mh"], o["mh"] + 2 * M)
        u0 = self.us[-1]
        nh = zh[o["node"](self.K - 1):o["node"](self.K - 1) + 4 * M]
        nl = zl[o["node"](self.K - 1):o["node"](self.K - 1) + 4 * M]
        dh, dl, hist, conv = self.invert(u0, zh, zl, np.array(nh), np.array(nl), np.array(zh[sl_]), np.array(zl[sl_]))
        # last_project: what each stage of this call did, for the log of a failed projection (#743); no effect on z'
        self.last_project = lp = dict(u0=dict(conv=bool(conv), its=len(hist) - 1, R0=hist[0]["R"], R=hist[-1]["R"],
                                              finite=bool(np.all(np.isfinite(dh)))))
        how = "u0"
        if self.qd and len(z) == 4 and np.all(np.isfinite(dh)):
            # the double-double u0 rung as the start (converged or at its floor), then the quad-double polish
            d4, hq, cq = self.invert_q(z, (dh, dl, np.zeros(2 * M), np.zeros(2 * M)))
            lp["u0q"] = dict(conv=bool(cq), its=len(hq) - 1, R0=hq[0]["R"], R=hq[-1]["R"])
            if cq:
                out = [np.array(p, dtype=float) for p in z]
                for q in range(4):
                    out[q][sl_] = d4[q]
                return tuple(np.ascontiguousarray(p) for p in out), "u0q"
            conv = False
        if not conv:
            zr, diag = self.reseed(z)
            last = {k: v for k, v in (diag["rungs"][-1] if diag["rungs"] else {}).items() if k != "hist"}
            lp["ladder"] = dict(ok=bool(diag["ok"]), rungs=len(diag["rungs"]), dropped=list(diag.get("dropped", [])),
                                last=last)
            if not diag["ok"]:
                return tuple(np.ascontiguousarray(np.asarray(p, dtype=float)) for p in z), "failed"
            return tuple(np.ascontiguousarray(np.asarray(p, dtype=float)) for p in zr), "ladder"
        out = [np.array(p, dtype=float) for p in z]
        out[0][sl_], out[1][sl_] = dh, dl
        for p in out[2:]:
            p[sl_] = 0.0
        return tuple(np.ascontiguousarray(p) for p in out), how


def why_failed(nf, lp):
    """one line on a failed projection: the point's pinned max |F| nf and each stage of Reseeder.project (lp, its
    last_project), so that the stage that failed and its residuals can be read (#743)"""
    if not lp:
        return f"point max|F| {nf:.2e}; no stage record"

    def stage(d):
        return f"{d['its']} its, |R| {d['R0']:.2e} -> {d['R']:.2e}, {'converged' if d['conv'] else 'NOT converged'}"
    out = [f"point max|F| {nf:.2e}", "u0 rung " + stage(lp["u0"]) + ("" if lp["u0"]["finite"] else ", data not finite")]
    if "u0q" in lp:
        out.append("u0 quad-double " + stage(lp["u0q"]))
    if "ladder" in lp:
        lad, r = lp["ladder"], lp["ladder"]["last"]
        if r.get("finite") is False:
            at = f"rung u {r['u']:.6f}: target not finite"
        elif r:
            at = f"rung u {r['u']:.6f}{' quad-double' if r.get('qd') else ''} " + stage(r)
        else:
            at = "no rung"
        out.append(f"ladder {'ok' if lad['ok'] else 'FAILED'} after {lad['rungs']} rungs, last {at}"
                   + (f", dropped {lad['dropped']}" if lad["dropped"] else ""))
    return "; ".join(out)


class StartNotProjected(Exception):
    """ElimQ: the projection of the corrector's start failed (its counts are the argument)"""


class ElimQ:
    """The corrector on the system with the horizon data eliminated (notes/RESEED.md 3.3, the arm elim): a wrapper of a
    pinned quad-double solver (pinning.PinnedQ) for hpkernel.driver.newton_qd. residual_q and newton_q project the point
    first (Reseeder.project: the horizon data re-inverted from the point's last node); newton_q returns the step from
    the given point, the projection's displacement included, so newton_qd's trial points z + lam dz are steps of the
    eliminated system. counts: the projections by kind. switch: a point whose pinned quad-double max |F| is at most
    switch is not projected (plain Newton finishes): the projection is double-double, so it would hold every point at
    the double-double floor phi (the first jump run stalled at max |F| 3.7e-18 this way, notes/RESEED.md 3.3).
    start_F: a start (the first point projected) whose projection fails at max |F| above start_F raises
    StartNotProjected (None: never); no such attempt of the chain logs to D 188 converged (notes/QUAD_DOUBLE.md 13.28).
    """

    def __init__(self, sq, rs, switch=1e-15, start_F=1e-6, log=False):
        self.sq, self.rs, self.switch, self.start_F = sq, rs, switch, start_F
        self.counts = dict(u0=0, u0q=0, ladder=0, failed=0, skipped=0)
        self.first = True
        self.log = log                  # log=True prints one line per failed projection (why_failed, #743)

    def __getattr__(self, name):
        return getattr(self.sq, name)

    def _proj(self, z):
        first, self.first = self.first, False
        nf = qmax(self.sq.residual_q(*z))
        if nf <= self.switch:
            self.counts["skipped"] += 1
            return z
        zp, how = self.rs.project(z)
        self.counts[how] += 1
        if how == "failed" and self.log:
            print(f"      projection failed ({self.counts['failed']}): "
                  + why_failed(nf, getattr(self.rs, "last_project", None)), flush=True)
        if how == "failed" and first and self.start_F is not None and nf > self.start_F:
            # an unprojected start keeps its horizon data off by up to the mass-mode gain, so every trial's
            # projection fails too, at minutes apiece past D 185 (notes/QUAD_DOUBLE.md 13.28)
            raise StartNotProjected(dict(self.counts))
        return zp

    def residual_q(self, z0, z1, z2, z3, mutate=0):
        return self.sq.residual_q(*self._proj((z0, z1, z2, z3)), mutate=mutate)

    def newton_q(self, z0, z1, z2, z3, mutate=0):
        from hpkernel.driver import qd_axpy
        z = tuple(np.ascontiguousarray(np.asarray(p, dtype=float)) for p in (z0, z1, z2, z3))
        zp = self._proj(z)
        st = self.sq.newton_q(*zp, mutate=mutate)
        disp = qd_axpy(-1.0, *z, *zp)                                  # z' - z
        d = qd_axpy(1.0, *(np.ascontiguousarray(np.asarray(p)) for p in disp), *(np.ascontiguousarray(np.asarray(p)) for p in st[:4]))
        return (*d, *st[4:])


# ------------------------------------------------------------------ the unit check
def pinned(meta, D, z, threads=2):
    """the chain's solver at D (kfreeze from meta, segs planned at z at 1e-32 as anm_run.py plans a prediction) and its
    pinned quad-double wrapper, with the held harmonics at the state's own values (a converged chain state holds them
    at the trend's values at its D)"""
    from pinning import PinnedQ, PinnedSolver, band_coeffs, band_field
    M = int(meta["M"])
    s = make_solver(D, M, nodes=meta["nodes"], kt=meta.get("kt", 36), kc=meta.get("kc", 146), ks=meta.get("ks", 130),
                    threads=threads, kfreeze=meta["kfreeze"])
    s.set_segs(s.plan(np.asarray(z[0]), np.asarray(z[1]), 1e-32, 0.05))
    ks_held = [k for k in range(int(meta["kfreeze"]) + 1, 14) if k % 2 == 1]
    hh, hl = band_field(band_coeffs(np.asarray(z[0])[:M], np.asarray(z[1])[:M], M, ks_held), ks_held, M)
    sp = PinnedSolver(s, M, hh, hl)
    return s, sp, PinnedQ(s, sp.ch, sp.cl)


def parse(argv, opt):
    for k in list(opt):
        f = "--" + k
        if f in argv:
            i = argv.index(f)
            opt[k] = type(opt[k])(argv[i + 1])
    return opt


def check(argv):
    """notes/RESEED.md 3.1 and 3.2 on a converged state (the docstring); returns the exit code"""
    opt = parse(argv, dict(threads=2, levels=5, rows="ma", mutate="none", out="", delta_h=1e-6, seed=1, plan_tol=1e-32, floor_rule="stall",
                           qd=0))
    path = argv[0]
    if opt["out"]:
        from raw_out import require_raw
        require_raw(opt["out"])
    z4, meta = load_q(path)
    D = mp.mpf(meta["D_str"])
    M, K = int(meta["M"]), len(meta["nodes"])
    n = float(D) - 2.0
    G = float(meta["nodes"][-1]) ** -(n - 1)                # the mass mode's gain from the horizon to the last node
    o = offsets(M, K)
    s, sp, sq = pinned(meta, D, z4, threads=opt["threads"])
    F_truth = qmax(sq.residual_q(*z4))
    Fh, Fl = sp.residual(z4[0], z4[1])
    phi = float(np.linalg.norm(np.asarray(Fh) + np.asarray(Fl)))
    # the start: the truth with its horizon data off by delta_h (relative, random signs and sizes 0.5-1)
    rng = np.random.default_rng(opt["seed"])
    sl_ = slice(o["mh"], o["mh"] + 2 * M)
    zs = [np.array(p) for p in z4]
    r = rng.uniform(0.5, 1.0, 2 * M) * rng.choice([-1.0, 1.0], 2 * M)
    zs[0][sl_] = zs[0][sl_] * (1 + opt["delta_h"] * r)
    zs[1][sl_] = 0.0; zs[2][sl_] = 0.0; zs[3][sl_] = 0.0
    rs = Reseeder(meta, D, threads=opt["threads"], levels=opt["levels"], rows=opt["rows"], mutate=opt["mutate"],
                  plan_tol=opt["plan_tol"], floor_rule=opt["floor_rule"], qd=bool(opt["qd"]))
    print(f"check {path}: D {mp.nstr(D, 12)}, G {G:.2e}, truth max|F| (pinned, quad-double) {F_truth:.2e}, double-double "
          f"|F|_2 phi {phi:.2e}; start: horizon data off by {opt['delta_h']:.0e}; rungs {[round(u, 6) for u in rs.us]}, rows "
          f"{opt['rows']}, mutate {opt['mutate']}, plan_tol {opt['plan_tol']:.0e}", flush=True)
    zr, diag = rs.reseed(zs, verbose=True)
    # 3.2: the targets (the truth's node integrated) against the truth's horizon series at each rung below u0
    th_d, tl_d = np.asarray(z4[0][sl_]), np.asarray(z4[1][sl_])
    stepper = []
    for p in diag["plan"]:
        u = p["u"]
        rec = dict(p)
        if u in rs.last_targets:
            Hh, Hl = rs.series_at(u, np.asarray(z4[0]), np.asarray(z4[1]), th_d, tl_d)
            Th, Tl = rs.last_targets[u]
            rec["vs_truth_series"] = float(np.max(np.abs((Th - Hh) + (Tl - Hl))))
        stepper.append(rec)
        print(f"    plan to x {p['x']:.6f}: {p['steps']} steps (smallest {p['h_min']:.1e}), max (x - c) {p['max_x_minus_c']:+.2e}, "
              f"stepper error estimate {p['err_est']:.1e}, against the truth's series {rec.get('vs_truth_series', float('nan')):.1e}",
              flush=True)
    # 3.2b (added after 3.2 failed as written, notes/RESEED.md 3.2): a node planted on the regular family, the truth's
    # series summed at u_s = u0 / 2, integrated to the rungs below and compared with the series there
    planted = []
    inner = sorted([u for u in rs.us if u < rs.us[-1]], reverse=True)
    if len(inner) >= 2:
        u_s = inner[0]
        Nh, Nl = rs.series_at(u_s, np.asarray(z4[0]), np.asarray(z4[1]), th_d, tl_d)
        xs_to = [1.0 - u for u in inner[1:]]
        T, info = rs.integrate(np.asarray(z4[0]), np.asarray(z4[1]), Nh, Nl, xs_to, rs.plan_tol, rs.hmax, rs.kt, x_start=1.0 - u_s)
        T2, _ = rs.integrate(np.asarray(z4[0]), np.asarray(z4[1]), Nh, Nl, xs_to, rs.check_tol, rs.hmax / 2, rs.check_kt, x_start=1.0 - u_s)
        for u, t, t2, p in zip(inner[1:], T, T2, info):
            Hh, Hl = rs.series_at(u, np.asarray(z4[0]), np.asarray(z4[1]), th_d, tl_d)
            d = (t[0] - Hh) + (t[1] - Hl)
            rec = dict(u=u, steps=p["steps"], max_x_minus_c=p["max_x_minus_c"], vs_series=float(np.max(np.abs(d))),
                       vs_series_pmac=[float(np.max(np.abs(d[q * M:(q + 1) * M]))) for q in range(4)],
                       err_est=float(np.max(np.abs((t[0] - t2[0]) + (t[1] - t2[1])))))
            planted.append(rec)
            print(f"    3.2b: node on the family at u {u_s:.4f} integrated to u {u:.6f}: {rec['steps']} steps, against the series "
                  f"{rec['vs_series']:.1e} (p {rec['vs_series_pmac'][0]:.0e} m {rec['vs_series_pmac'][1]:.0e} a {rec['vs_series_pmac'][2]:.0e} "
                  f"c {rec['vs_series_pmac'][3]:.0e}), error estimate {rec['err_est']:.1e}", flush=True)
    e = (np.asarray(zr[0][sl_]) - np.asarray(z4[0][sl_])) + (np.asarray(zr[1][sl_]) - np.asarray(z4[1][sl_])) - np.asarray(z4[2][sl_])
    err_mh, err_ah = float(np.max(np.abs(e[:M]))), float(np.max(np.abs(e[M:])))
    Fq = sq.residual_q(*zr)
    Fv = np.asarray(Fq[0]) + np.asarray(Fq[1])
    Fmax = float(np.max(np.abs(Fv)))
    hb = K * 4 * M
    Fh_rows = [float(np.max(np.abs(Fv[hb + q * M:hb + (q + 1) * M]))) for q in range(4)]
    # the criteria of notes/RESEED.md 3.1 and 3.2, written before the first run
    crit = dict(ok=bool(diag["ok"]) and not diag["dropped"], err_ah=err_ah <= 10 * phi / G, err_mh=err_mh <= 20 * phi,
                F=Fmax <= 10 * phi)
    if opt["qd"]:
        # --qd 1 (the quad-double u0 rung, notes/QUAD_DOUBLE.md 13.22, written before its first run): the re-seeded state's
        # pinned quad-double residual within 100 times the larger of the truth's own and the quad-double scale 2^-209 G
        bound_q = 100 * max(F_truth, 2.0 ** -209 * G)
        crit = dict(ok=bool(diag["ok"]) and not diag["dropped"], F_qd=Fmax <= bound_q)
        print(f"  quad-double mode: pinned max|F| {Fmax:.2e} against {bound_q:.1e} (truth {F_truth:.1e}, 2^-209 G {2.0 ** -209 * G:.1e})",
              flush=True)
    crit_step = dict(agree=all(r.get("vs_truth_series", np.inf) <= 1e-24 for r in stepper) and len(stepper) == len(rs.us) - 1,
                     err_est=all(r["err_est"] <= 1e-28 for r in stepper), sonic=all(r["max_x_minus_c"] < 0 for r in stepper))
    crit_planted = dict(agree=bool(planted) and all(r["vs_series"] <= 1e-24 for r in planted),
                        err_est=all(r["err_est"] <= 1e-28 for r in planted), sonic=all(r["max_x_minus_c"] < 0 for r in planted))
    passed, passed_step, passed_planted = all(crit.values()), all(crit_step.values()), all(crit_planted.values())
    rec = dict(test="unit", state=path, D=float(D), D_str=meta["D_str"], G=G, phi=phi, mutate=opt["mutate"], rows=opt["rows"], floor_rule=opt["floor_rule"],
               levels=opt["levels"], plan_tol=opt["plan_tol"], delta_h=opt["delta_h"], seed=opt["seed"], F_truth=F_truth, err_mh=err_mh,
               err_ah=err_ah, F_max=Fmax, F_horizon_pmac=Fh_rows, crit=crit, passed=passed, stepper=stepper, crit_stepper=crit_step,
               passed_stepper=passed_step, planted=planted, crit_planted=crit_planted, passed_planted=passed_planted, rungs=diag["rungs"], dropped=diag["dropped"], seconds=diag["seconds"])
    print(f"  re-seeded: |m_h - truth| {err_mh:.2e} (bound {20 * phi:.1e}), |a_h - truth| {err_ah:.2e} (bound {10 * phi / G:.1e}); "
          f"pinned quad-double max|F| {Fmax:.2e} (bound {10 * phi:.1e}; horizon rows p {Fh_rows[0]:.1e}, m {Fh_rows[1]:.1e}, "
          f"a {Fh_rows[2]:.1e}, c {Fh_rows[3]:.1e}); {diag['seconds']} s", flush=True)
    print(f"  3.1 {crit}: {'PASS' if passed else 'FAIL'}", flush=True)
    print(f"  3.2 {crit_step}: {'PASS' if passed_step else 'FAIL'}", flush=True)
    print(f"  3.2b {crit_planted}: {'PASS' if passed_planted else 'FAIL'}", flush=True)
    if opt["out"]:
        with open(opt["out"], "a") as f:
            f.write(json.dumps(rec) + "\n")
    return 0 if (passed and passed_planted) else 1


def jac(argv):
    """notes/RESEED.md 2.1: singular values of the horizon block's rows (by groups) in the m_h, a_h columns"""
    z4, meta = load_q(argv[0])
    D = mp.mpf(meta["D_str"])
    M, K = int(meta["M"]), len(meta["nodes"])
    o = offsets(M, K)
    s, sp, sq = pinned(meta, D, z4, threads=2)
    print(f"{argv[0]}: D {mp.nstr(D, 12)}, G = x_K-1^-(n-1) {float(meta['nodes'][-1]) ** -(float(D) - 3):.3e}")
    # the chain's kernel (the p rows projected by the freeze: rank 4 at kfreeze 3) and the same without the freeze
    s0 = make_solver(D, M, nodes=meta["nodes"], kt=meta.get("kt", 36), kc=meta.get("kc", 146), ks=meta.get("ks", 130), threads=2)
    s0.set_segs(s.get_segs())
    for label, sx in (("freeze (the chain's)", s), ("no freeze", s0)):
        Jh, Jl = (np.asarray(v) for v in sx.jacobian(z4[0], z4[1]))
        J = Jh + Jl
        r0 = K * 4 * M
        H = J[r0:r0 + 4 * M, o["mh"]:o["mh"] + 2 * M]
        print(f" {label}:")
        jac_groups(H, M)
    return 0


def jac_groups(H, M):
    for name, rr in (("all", np.r_[0:4 * M]), ("p,m", np.r_[0:2 * M]), ("a,c", np.r_[2 * M:4 * M]), ("m,a", np.r_[M:3 * M]),
                     ("m,c", np.r_[M:2 * M, 3 * M:4 * M])):
        U, S, Vt = np.linalg.svd(H[rr])
        mh = np.linalg.norm(Vt[:, :M], axis=1)
        print(f"  rows {name}: singular values {S[0]:.3e} .. {S[-1]:.3e}; the a_h-dominated ones {S[mh < 0.5].min() if (mh < 0.5).any() else 0:.3e}"
              f"..{S[mh < 0.5].max() if (mh < 0.5).any() else 0:.3e}, the m_h-dominated ones {S[mh >= 0.5].min():.3e}..{S[mh >= 0.5].max():.3e}")


def checkq(argv):
    """the re-seed with its quad-double polish (reseed, qd) on a stored prediction (anm_run.py --save_pred 1, at its D):
    it must converge, the u0 rung's quad-double residual below 1e-24 (notes/QUAD_DOUBLE.md 13.26); exit 1 otherwise.
    The control --legacy 1 (the polish with it_max 6 and one factoring, the earlier rule) must fail on a prediction
    where that rule stopped short, as T118's at D 152.93 did. The control --sweeps 1 (one pass of the block steps, never
    repeated) must fail on a prediction past D 210, as I215-win's d6 prediction at D 215.0 does (#446)."""
    opt = parse(argv, dict(threads=4, legacy=0, rule136=0, sweeps=0))
    z4, meta = load_q(argv[0])
    D = mp.mpf(meta["D_str"])
    rs = Reseeder(meta, D, threads=opt["threads"], qd=True)
    if opt["legacy"] or opt["rule136"]:
        inv = rs.invert_q
        kw0 = dict(pre="", take_all=False) if opt["rule136"] else dict(pre="", take_all=False, it_max=6, refactor_max=0)
        rs.invert_q = lambda z, d, **kw: inv(z, d, **kw0)
    elif opt["sweeps"]:
        inv = rs.invert_q
        rs.invert_q = lambda z, d, **kw: inv(z, d, **dict(kw, sweeps=opt["sweeps"]))
    t0 = time.time()
    _, diag = rs.reseed(z4, verbose=True)
    q = [r for r in diag["rungs"] if r.get("qd")]
    ok = bool(diag["ok"]) and bool(q) and q[-1]["R"] < 1e-24
    rule = 'legacy rule' if opt['legacy'] else ('13.26 rule' if opt['rule136'] else 'a_h, m_h steps and Newton')
    print(f"checkq {argv[0]}: D {mp.nstr(D, 12)}, {rule}: re-seed "
          f"{'ok' if diag['ok'] else 'FAILED'}, u0 quad-double |R| {q[-1]['R'] if q else float('nan'):.2e}, "
          f"{time.time() - t0:.0f} s: {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


def truthq(argv):
    """planted truth for the quad-double polish (#446): a converged chain state (its first four parts, load_q) with its
    horizon data rounded to double-double and moved by --delta (relative, random signs), as the double-double ladder
    leaves them. The re-seed must converge and return the state's own a_h and m_h, each within 1e-3 delta of them in
    the max norm relative to the block's largest entry; exit 1 otherwise. The control --sweeps 1 (the block steps never
    repeated) must fail on a state past D 210: the planted error is then left where it was. A moved node is no control
    here, because the horizon data follow the node only through the mass mode's inverse gain, about 1e-33 at D 215."""
    opt = parse(argv, dict(threads=4, sweeps=0, delta=1e-32, seed=1))
    z4, meta = load_q(argv[0])
    D = mp.mpf(meta["D_str"])
    rs = Reseeder(meta, D, threads=opt["threads"], qd=True)
    if opt["sweeps"]:
        inv = rs.invert_q
        rs.invert_q = lambda z, d, **kw: inv(z, d, **dict(kw, sweeps=opt["sweeps"]))
    M, o = rs.M, rs.off
    sl_ = slice(o["mh"], o["mh"] + 2 * M)
    truth = [np.ascontiguousarray(np.asarray(p, dtype=float)[sl_]) for p in z4]
    start = [np.array(p, dtype=float) for p in z4]
    start[1][sl_] += opt["delta"] * start[0][sl_] * np.random.default_rng(opt["seed"]).choice([-1.0, 1.0], 2 * M)
    start[2][sl_] = 0.0
    start[3][sl_] = 0.0
    t0 = time.time()
    zr, diag = rs.reseed(tuple(np.ascontiguousarray(p) for p in start), verbose=True)
    got = [np.ascontiguousarray(np.asarray(p, dtype=float)[sl_]) for p in zr]
    d = qd_axpy(-1.0, *truth, *got)
    d = np.abs(np.asarray(d[0]) + np.asarray(d[1]))
    em, ea = float(np.max(d[:M]) / np.max(np.abs(truth[0][:M]))), float(np.max(d[M:]) / np.max(np.abs(truth[0][M:])))
    ok = bool(diag["ok"]) and max(em, ea) <= 1e-3 * opt["delta"]
    print(f"truthq {argv[0]}: D {mp.nstr(D, 12)}, planted {opt['delta']:.0e}: re-seed "
          f"{'ok' if diag['ok'] else 'FAILED'}, error left in m_h {em:.2e}, in a_h {ea:.2e} (limit {1e-3 * opt['delta']:.0e}), "
          f"{time.time() - t0:.0f} s: {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


def checkstart(argv):
    """the corrector of anm_run.py --reseed 2 (ElimQ around the pinned quad-double solver, newton_qd with the chain's
    tol 1e-28 and 12 steps) on a stored prediction (anm_run.py --save_pred 1), after its re-seed: it must end,
    converged or stopped by StartNotProjected, within --budget seconds; exit 1 otherwise. The control --start_F 0 (no
    stop) must fail on a prediction whose start projection fails, as T129's at D 189.02 (notes/QUAD_DOUBLE.md 13.28).
    The budget is a SIGALRM (macOS, Linux)."""
    import signal
    from hpkernel.driver import newton_qd
    opt = parse(argv, dict(threads=4, start_F=1e-6, budget=900))
    z4, meta = load_q(argv[0])
    D = mp.mpf(meta["D_str"])
    _, _, sq = pinned(meta, D, z4, threads=opt["threads"])
    rs = Reseeder(meta, D, threads=opt["threads"], qd=True)
    t0 = time.time()
    zr, diag = rs.reseed(z4)
    el = ElimQ(sq, rs, start_F=opt["start_F"] if opt["start_F"] > 0 else None)

    class Budget(Exception):
        pass

    def over(*_):
        raise Budget()

    signal.signal(signal.SIGALRM, over)
    signal.alarm(int(opt["budget"]))
    try:
        _, hist, conv = newton_qd(el, zr, tol=1e-28, it_max=12, verbose=True)
        how = f"{'converged' if conv else 'not converged'} after {len(hist)} steps"
    except StartNotProjected:
        how = "stopped, the start's projection failed"
    except Budget:
        how = f"over the budget of {opt['budget']} s"
    finally:
        signal.alarm(0)
    ok = not how.startswith("over")
    print(f"checkstart {argv[0]}: D {mp.nstr(D, 12)}, start_F {opt['start_F']:g}: re-seed {'ok' if diag['ok'] else 'FAILED'}, "
          f"corrector {how}, projections {el.counts}, {time.time() - t0:.0f} s: {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


def projlog(argv):
    """the log line of a failed projection (#743) on stubs, no kernel run: a re-seeder whose u0 rung, quad-double
    polish and ladder all fail, twice. Pass: two lines, numbered 1 and 2, each naming the point's max|F|, the three
    stages' residuals and the ladder's last rung. --mutate=silent (log off) and --mutate=stage (the ladder's record
    dropped) must fail. Exit 0 on pass."""
    import contextlib
    import io
    mutate = next((a.split("=", 1)[1] for a in argv if a.startswith("--mutate=")), "none")

    class RS:
        def project(self, z):
            self.last_project = dict(u0=dict(conv=False, its=30, R0=1.0e-20, R=3.0e-22, finite=True),
                                     u0q=dict(conv=False, its=16, R0=3.0e-22, R=2.0e-40),
                                     ladder=dict(ok=False, rungs=4, dropped=[],
                                                 last=dict(u=0.0123, conv=False, its=30, R0=1.0e-18, R=5.0e-21)))
            if mutate == "stage":
                del self.last_project["ladder"]
            return z, "failed"

    class SQ:
        def residual_q(self, *z):
            return np.array([1.5e-20]), np.zeros(1), np.zeros(1), np.zeros(1)

    el = ElimQ(SQ(), RS(), switch=0.0, start_F=None, log=mutate != "silent")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        for _ in range(2):
            el._proj(tuple(np.zeros(1) for _ in range(4)))
    lines = buf.getvalue().splitlines()
    want = ["point max|F| 1.50e-20", "u0 rung 30 its, |R| 1.00e-20 -> 3.00e-22, NOT converged",
            "u0 quad-double 16 its, |R| 3.00e-22 -> 2.00e-40, NOT converged",
            "ladder FAILED after 4 rungs, last rung u 0.012300 30 its, |R| 1.00e-18 -> 5.00e-21, NOT converged"]
    ok = (len(lines) == 2 and el.counts["failed"] == 2
          and all(f"projection failed ({k + 1})" in ln and all(w in ln for w in want) for k, ln in enumerate(lines)))
    for ln in lines:
        print(ln)
    print(f"projlog: {'PASS' if ok else 'FAIL'} ({len(lines)} lines, failed count {el.counts['failed']}"
          + (f", mutate={mutate}" if mutate != "none" else "") + ")", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "projlog":
        sys.exit(projlog(sys.argv[2:]))
    if len(sys.argv) > 2 and sys.argv[1] == "check":
        sys.exit(check(sys.argv[2:]))
    if len(sys.argv) > 2 and sys.argv[1] == "checkq":
        sys.exit(checkq(sys.argv[2:]))
    if len(sys.argv) > 2 and sys.argv[1] == "truthq":
        sys.exit(truthq(sys.argv[2:]))
    if len(sys.argv) > 2 and sys.argv[1] == "checkstart":
        sys.exit(checkstart(sys.argv[2:]))
    if len(sys.argv) > 2 and sys.argv[1] == "jac":
        sys.exit(jac(sys.argv[2:]))
    print(__doc__)
    sys.exit(2)

#!/usr/bin/env python3
"""Validated integration runs for the M-harmonic certificate (proof/A_M.md section 3), in ball arithmetic.

The integrator is validated_int.Lohner (the QR-Lohner method of proof/A_certify.md section 9.1), unchanged in its
mathematics, with three choices made here:
  * the a-priori enclosures of lemma6_common.py (a common inflation floor; containment still verified), as on the
    lower path of Theorem A(M = 1);
  * the Taylor coefficients by Picard iteration with growing series length (iteration k carries the series to length
    k + 2; coefficients 0..N are the same numbers as validated_int.taylor_Y's, at about a third of the cost);
  * the system is certify_M_sys.SysM (any odd M) and the path legs certify_M_sys.LegM (start ray angle a parameter).

Two kinds of run:
  * point run (C^1 along one trajectory): the set of y+(xi; p^) and the M + 1 parameter-derivative columns
    dy+/dp at p^, each a Lohner set; no parameter box. Gives g(p^) and Dg(p^) with tiny radii. Used for the (non-
    rigorous) refinement of the centre and for the chart and the preconditioner of the Krawczyk test.
  * box run (C^1 over a parameter parallelepiped P = {p^ + T x : x in X}, X a box centred at 0): the set of the
    centre trajectory y+(xi; p^) and the derivative columns dy+/dx = dy+/dp T over P (validated_int's centre_set mode:
    the state set over P is rebuilt at every step as y(t; p^) + V(t; X) x, mean value theorem).

Every run starts from start data (values and derivative columns at xi0), passed in by the caller (certify_M.py), and
saves its final Lohner sets exactly (Q matrices as float hex, balls as decimal strings with radius).
"""
import json
import os
import sys
import time

import numpy as np
from flint import acb, arb, acb_mat, arb_mat, acb_series, ctx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import validated_int as VI  # noqa: E402
import lemma6_common  # noqa: E402
import certify_M_sys as CS  # noqa: E402

T0 = time.time()


def taylor_Y(sys_, leg, t0, Y0, N):
    """validated_int.taylor_Y with growing series length: after Picard iteration k the coefficients 0..k are final,
    so iteration k only needs length k + 2. Returns the same (series list, context) as validated_int.taylor_Y."""
    L = N + 1
    ctx.cap = L
    xs, dxs = leg.series(t0, L)
    nd = sys_.nd
    Ys = [acb_series([y], prec=L) for y in Y0]
    for it in range(N + 1):
        Lk = min(it + 2, L)
        ctx.cap = Lk
        xk = acb_series(xs.coeffs()[:Lk], prec=Lk)
        dxk = acb_series(dxs.coeffs()[:Lk], prec=Lk)
        ixk = 1 / xk
        zk = acb_series([0], prec=Lk)
        Yk = [acb_series(y.coeffs()[:Lk], prec=Lk) for y in Ys]
        F, _ = sys_.rhs(Yk, xk, ixk, zk)
        new = [acb_series([Y0[i]], prec=Lk) + (F[i] * dxk).integral() for i in range(nd)]
        Ys = new + Ys[nd:]
    ctx.cap = L
    ixs = 1 / xs
    zero = acb_series([0], prec=L)
    Ys = [acb_series(y.coeffs()[:L], prec=L) for y in Ys[:nd]] + [acb_series([y], prec=L) for y in Y0[nd:]]
    F, cache = sys_.rhs(Ys, xs, ixs, zero)
    return Ys, (cache, xs, ixs, dxs, zero)


def _trunc(x, L):
    if isinstance(x, acb_series):
        return acb_series(x.coeffs()[:L], prec=L)
    return x


def truncate_ctx(ctxv, L):
    """the context of taylor_Y (the value-level cache and the path series) truncated to length L: the same
    coefficients 0..L-1 that taylor_Y(..., L - 1) would give (they do not depend on the higher ones)"""
    cache, xs, ixs, dxs, zero = ctxv
    cache = tuple({k: _trunc(v, L) for k, v in c.items()} if isinstance(c, dict) else _trunc(c, L) for c in cache)
    return cache, _trunc(xs, L), _trunc(ixs, L), _trunc(dxs, L), _trunc(zero, L)


def taylor_V(sys_, ctxv, N):
    """validated_int.taylor_V with the columns of A(tau) = F_Y(Y(tau)) xi'(tau) from SysM.lin_unit (the same
    linearization, products with zero series skipped); the coefficient recursion is the same:
    V_{k+1} = (1/(k+1)) sum_{j <= k} A_{k-j} V_j, V_0 = I."""
    cache, xs, ixs, dxs, zero = ctxv
    L = N + 1
    ctx.cap = L
    n, nd = sys_.n, sys_.nd
    Acols = []
    for j in range(n):
        F = sys_.lin_unit(cache, j, xs, ixs, zero)
        Acols.append([(F[i] * dxs).coeffs() for i in range(nd)])
    Ak = []
    for k in range(N + 1):
        Mk = acb_mat(n, n)
        for j in range(n):
            col = Acols[j]
            for i in range(nd):
                cs = col[i]
                if k < len(cs):
                    Mk[i, j] = cs[k]
        Ak.append(Mk)
    V = [acb_mat([[1 if i == j else 0 for j in range(n)] for i in range(n)])]
    for k in range(N):
        acc = Ak[k] * V[0]
        for j in range(1, k + 1):
            acc = acc + Ak[k - j] * V[j]
        V.append(acc * acb(arb(1) / (k + 1)))
    return V


class PilotStop(Exception):
    """raised when a pilot run reaches its step budget (LohnerM.max_steps)"""


class LohnerM(VI.Lohner):
    """validated_int.Lohner with the fast taylor_Y above and lemma6_common's a-priori enclosures; the step is
    validated_int.Lohner.step line by line otherwise, except that the step Jacobian may use its own Taylor order
    NJ <= N: J = sum_{k <= NJ} V_k(X) h^k + h^(NJ+1) Phi_{NJ+1}(Y~) V~. That remainder formula holds for any order
    (integral form of the remainder: V^(NJ+1)(s) = (NJ+1)! Phi_{NJ+1}(Y(s)) V(s), Y(s) in Y~, V(s) in V~), so J is an
    enclosure for every NJ; a lower NJ only widens it by about (h lambda)^(NJ+1), which is far below the width that
    the parameter box gives it. NJ = N is validated_int's choice."""

    NJ = None

    def step(self, leg, t0, h):
        try:
            return self._step(leg, t0, h)
        except (ZeroDivisionError, ValueError, OverflowError, FloatingPointError):
            return False

    def _step(self, leg, t0, h):
        s, N, n = self.s, self.N, self.s.n
        X = self.box()
        Yt, T = lemma6_common.apriori(s, leg, t0, h, X)
        if Yt is None:
            return False
        Vt = lemma6_common.apriori_V(s, leg, T, Yt, h)
        if Vt is None:
            return False
        hh = acb(h)
        hN1 = hh ** (N + 1)
        Yc, _ = taylor_Y(s, leg, acb(t0), self.c, N)
        Yr, ctr = taylor_Y(s, leg, T, Yt, N + 1)
        nd = s.nd
        phic = [VI.evalpoly(Yc[i], hh, N) + hN1 * VI.coef(Yr[i], N + 1) for i in range(nd)] + self.c[nd:]
        NJ = self.NJ or N
        _, ctxX = taylor_Y(s, leg, acb(t0), X, NJ)
        Vk = taylor_V(s, ctxX, NJ)
        ctrJ = truncate_ctx(ctr, NJ + 2) if NJ < N else ctr
        VR = taylor_V(s, ctrJ, NJ + 1)
        J = Vk[NJ]
        for k in range(NJ - 1, -1, -1):
            J = J * hh + Vk[k]
        J = J + hh ** (NJ + 1) * (VR[NJ + 1] * Vt)
        JR = VI.realify(J)
        cnew = [acb(v.mid()) for v in phic]
        e = VI.stack(acb_mat([[phic[i] - cnew[i]] for i in range(n)]))
        A = JR * self.Q
        Qn = VI.from_np_r(VI.qr_pivot(VI.to_np_r(A), [float(abs(x).upper()) for x in self.q]))
        Qi = Qn.inv()
        qn = (Qi * A) * arb_mat([[x] for x in self.q]) + Qi * e
        if self.var is not None:
            cV, QV, qV = self.var
            m = cV.ncols()
            JC = J * cV
            cVn = acb_mat([[acb(JC[i, j].mid()) for j in range(m)] for i in range(n)])
            AV = JR * QV
            QVn = VI.from_np_r(VI.qr_pivot(VI.to_np_r(AV), [max(float(abs(qV[i, j]).upper()) for j in range(m))
                                                             for i in range(2 * n)]))
            QVi = QVn.inv()
            self.var = (cVn, QVn, (QVi * AV) * qV + QVi * VI.stack(JC - cVn))
        self.c, self.Q, self.q = cnew, Qn, [qn[i, 0] for i in range(2 * n)]
        return True

    def run(self, leg, h0, verbose=False, log=None, t_start=0.0, steps_start=0, ckpt=None):
        """validated_int.Lohner.run (t: 0 -> 1 on the grid 2^-30 Z) with optional progress lines to a log; may start
        at t_start (a resumed run) and calls ckpt(t, h, steps) about every CKPT_SEC seconds"""
        q = 2.0 ** -30
        hmax = getattr(self, "hmax", 1.0)
        t, h, steps = t_start, min(h0, hmax), steps_start
        last = time.time()
        while t < 1.0:
            if ckpt is not None and time.time() - last > CKPT_SEC:
                ckpt(t, h, steps)
                last = time.time()
            if getattr(self, "max_steps", None) is not None and getattr(self, "total_steps", 0) >= self.max_steps:
                raise PilotStop(f"{leg.kind} t = {t:.6f}")
            hs = min(max(round(h / q), 1) * q, 1.0 - t)
            if not self.step(leg, t, hs):
                h = hs / 2
                if h < 1e-9:
                    raise RuntimeError(f"step size underflow at t = {t} on the {leg.kind} leg")
                continue
            t = t + hs
            steps += 1
            self.total_steps = getattr(self, "total_steps", 0) + 1
            h = min(hs * 1.25, hmax)
            if (verbose or log) and steps % 25 == 0:
                X = self.box()
                rad = max(max(float(x.real.rad()), float(x.imag.rad())) for x in X[:self.s.nd])
                vr = 0.0
                if self.var is not None:
                    V = self.var_box()
                    vr = max(VI.rad_of(V[i, j]) for i in range(self.s.nd) for j in range(V.ncols()))
                msg = (f"   {leg.kind} t {t:.6f} h {hs:.2e} steps {steps}  state radius {rad:.2e}  "
                       f"derivative radius {vr:.2e}  [{time.time() - T0:.0f} s]")
                if verbose:
                    print(msg, flush=True)
                if log:
                    with open(log, "a") as fh:
                        fh.write(msg + "\n")
        return steps


# ---------------------------------------------------------------- numerics (NOT rigorous): for the centre refinement

NUM_RATE = 4.0       # numerics: steps with (row-sum norm of A_0) h <= NUM_RATE


def numerics_run(sys_, y0, w, cols0, path, N=24, tol=1e-40, log=None):
    """plain high-precision Taylor integration (midpoints only, no enclosure): y and the derivative columns along the
    path. y0: nd acb (midpoints used); w: arb; cols0: m columns of nd + 1 entries (the last the w-component).
    Step size from the last two Taylor coefficients: |Y_k| h^k <= tol (k = N - 1, N), in units of the leg parameter.
    Used only to choose the centre p^ of the box (any centre is allowed; the test is what is rigorous)."""
    n, nd = sys_.n, sys_.nd
    y = [acb(v.mid()) for v in y0] + [acb(arb(w).mid())]
    m = len(cols0)
    W = acb_mat([[acb(cols0[j][i].mid()) for j in range(m)] for i in range(n)])
    steps = []
    for leg in CS.legs_for(path):
        t, ns = 0.0, 0
        while t < 1.0:
            Ys, ctxs = taylor_Y(sys_, leg, acb(t), y, N)
            cs = [Y.coeffs() for Y in Ys[:nd]]

            def cmax(k):
                return max(float(abs(c[k]).mid()) if k < len(c) else 0.0 for c in cs) + 1e-300
            ymax = max(1.0, max(float(abs(c[0]).mid()) for c in cs))
            h = min((tol * ymax / cmax(N)) ** (1.0 / N), (tol * ymax / cmax(N - 1)) ** (1.0 / (N - 1)), 1.0 - t)
            Vk, A0 = taylor_V_cols(sys_, ctxs, N, W, with_A0=True)
            h = min(h, NUM_RATE / A0)             # the linearization's rate (row-sum norm): growing modes are resolved
            hh = acb(h)
            y = [acb(VI.evalpoly(Ys[i], hh, N).mid()) for i in range(nd)] + y[nd:]
            Wn = Vk[N]
            for k in range(N - 1, -1, -1):
                Wn = Wn * hh + Vk[k]
            W = acb_mat([[acb(Wn[i, j].mid()) for j in range(m)] for i in range(n)])
            t += h
            ns += 1
            if log and ns % 50 == 0:
                with open(log, "a") as fh:
                    fh.write(f"   numerics {leg.kind} t {t:.6f} h {h:.2e} steps {ns} [{time.time() - T0:.0f} s]\n")
        steps.append(ns)
    return y, W, steps


def taylor_V_cols(sys_, ctxv, N, W0m, with_A0=False):
    """Taylor coefficients of W(tau) = Phi(tau) W0 (n x m), W' = A(tau) W: W_{k+1} = (1/(k+1)) sum_j A_{k-j} W_j.
    with_A0: also return the max row sum of |A_0| (the local rate of the linearization, per unit of the leg)."""
    cache, xs, ixs, dxs, zero = ctxv
    L = N + 1
    ctx.cap = L
    n, nd = sys_.n, sys_.nd
    Acols = []
    for j in range(n):
        F = sys_.lin_unit(cache, j, xs, ixs, zero)
        Acols.append([(F[i] * dxs).coeffs() for i in range(nd)])
    Ak = []
    for k in range(N + 1):
        Mk = acb_mat(n, n)
        for j in range(n):
            col = Acols[j]
            for i in range(nd):
                cs = col[i]
                if k < len(cs):
                    Mk[i, j] = cs[k]
        Ak.append(Mk)
    Wk = [W0m]
    for k in range(N):
        acc = Ak[k] * Wk[0]
        for j in range(1, k + 1):
            acc = acc + Ak[k - j] * Wk[j]
        Wk.append(acc * acb(arb(1) / (k + 1)))
    if with_A0:
        A0 = max(sum(float(abs(Ak[0][i, j]).mid()) for j in range(n)) for i in range(n)) + 1e-300
        return Wk, A0
    return Wk


# ---------------------------------------------------------------- start sets

def start_set(sys_, vals_c, w_c, cols=None, wcol=None, dx=None, N=24, NJ=None):
    """Lohner set at xi0.
    vals_c: enclosures of y+(xi0; p^) (nd acb); w_c: w at p^ (arb).
    cols: derivative columns over the parameter set (list of m lists of nd acb), wcol: their w-components (m arb).
    dx: None (point run: the columns are along the single trajectory p^) or the m parameter offsets (acb balls) of the
    box (box run: centre_set mode, the state set is c + Q q + V dx)."""
    n, nd = sys_.n, sys_.nd
    c = [acb(v.mid()) for v in vals_c] + [acb(w_c.mid())]
    q = [vals_c[i] - c[i] for i in range(nd)] + [acb(w_c) - c[nd]]
    Q = acb_mat([[1 if i == j else 0 for j in range(n)] for i in range(n)])
    var = None
    if cols is not None:
        m = len(cols)
        V0 = acb_mat(n, m)
        for j in range(m):
            for i in range(nd):
                V0[i, j] = cols[j][i]
            V0[nd, j] = acb(wcol[j])
        cV = acb_mat([[acb(V0[i, j].mid()) for j in range(m)] for i in range(n)])
        var = (cV, Q, V0 - cV)
    lo = LohnerM(sys_, c, Q, q, N, var)
    lo.NJ = NJ
    if dx is not None:
        lo.dx = list(dx)
    return lo


CKPT_SEC = 600.0      # checkpoint interval of long runs (wall-clock seconds)


def save_ckpt(path_file, key, lo, k, t, h, steps_done, steps_leg):
    """the run's state (the Lohner sets exactly, the leg, t on the 2^-30 grid, the step size, the step counts),
    written atomically"""
    d = {"key": key, "leg": k, "t": float(t).hex(), "h": float(h).hex(), "steps_done": steps_done,
         "steps_leg": steps_leg, "total_steps": getattr(lo, "total_steps", 0), "sets": dump_lohner(lo),
         "prec": ctx.prec, "time": time.time()}
    tmp = path_file + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(d, fh)
    os.replace(tmp, path_file)


def integrate(lo, path, h0=1e-3, verbose=False, log=None, keep_rho=False, ckpt=None, key=None):
    """run the legs of path = (r0, phi0, rho, xi1); returns the step counts per leg (and the state at the end of
    the arc if keep_rho). With ckpt (a file name) the run saves its state about every CKPT_SEC seconds and at the end
    of every leg, and resumes from that file if it exists and was written by a run with the same key (the run's
    parameters); a resumed run continues from exactly the saved Lohner sets, so its result is a valid enclosure."""
    steps, at_rho = [], None
    k0, t0_, h_, s0 = 0, 0.0, h0, 0
    if ckpt and os.path.exists(ckpt):
        with open(ckpt) as fh:
            d = json.load(fh)
        if d["key"] != key:
            raise RuntimeError(f"checkpoint {ckpt} belongs to another run (key mismatch); remove it to start afresh")
        lo.c, lo.Q, lo.q, lo.var = load_lohner(d["sets"])
        lo.total_steps = d["total_steps"]
        k0, t0_, h_, s0, steps = d["leg"], float.fromhex(d["t"]), float.fromhex(d["h"]), d["steps_leg"], d["steps_done"]
        if log:
            with open(log, "a") as fh:
                fh.write(f"resumed from {ckpt}: leg {k0}, t {t0_:.6f}, {lo.total_steps} steps done\n")
    for k, leg in enumerate(CS.legs_for(path)):
        if k < k0:
            continue
        t1 = time.time()
        cb = None
        if ckpt:
            def cb(t, h, n, k=k):
                save_ckpt(ckpt, key, lo, k, t, h, steps, n)
        ts, ss, hh = (t0_, s0, h_) if k == k0 else (0.0, 0, h0)
        steps.append(lo.run(leg, hh, verbose, log, t_start=ts, steps_start=ss, ckpt=cb))
        if log:
            with open(log, "a") as fh:
                fh.write(f"leg {leg.kind} done: {steps[-1]} steps, {time.time() - t1:.0f} s\n")
        if ckpt:
            save_ckpt(ckpt, key, lo, k + 1, 0.0, h0, steps, 0)
        if k == 1 and keep_rho:
            at_rho = lo.box()
    return steps, at_rho


# ---------------------------------------------------------------- serialization (exact)

def s_arb(x, digits=None):
    digits = digits or max(30, int(ctx.prec / 3.3) + 10)
    return x.str(digits, radius=True, more=True)


def s_acb(z):
    return [s_arb(z.real), s_arb(z.imag)]


def p_acb(v):
    return acb(arb(v[0]), arb(v[1]))


def s_mat_float(Mr):
    """real arb_mat of exact floats -> list of float.hex strings (row major)"""
    return [[float(Mr[i, j].mid()).hex() for j in range(Mr.ncols())] for i in range(Mr.nrows())]


def p_mat_float(L):
    return arb_mat([[arb(float.fromhex(x)) for x in row] for row in L])


def check_exact_float(Mr):
    """the Lohner Q matrices are exact binary floats (from_np_r); verify before saving them as float hex"""
    for i in range(Mr.nrows()):
        for j in range(Mr.ncols()):
            x = Mr[i, j]
            if x.rad() != 0 or arb(float(x.mid())) != x:
                return False
    return True


def dump_lohner(lo):
    """the final Lohner sets: centre set (c, Q, q) and, if present, the derivative sets (cV, QV, qV)"""
    assert check_exact_float(lo.Q)
    out = {"c": [s_acb(z) for z in lo.c], "Q": s_mat_float(lo.Q), "q": [s_arb(x) for x in lo.q]}
    if lo.var is not None:
        cV, QV, qV = lo.var
        assert check_exact_float(QV)
        out["cV"] = [[s_acb(cV[i, j]) for j in range(cV.ncols())] for i in range(cV.nrows())]
        out["QV"] = s_mat_float(QV)
        out["qV"] = [[s_arb(qV[i, j]) for j in range(qV.ncols())] for i in range(qV.nrows())]
    if lo.dx is not None:
        out["dx"] = [s_acb(z) for z in lo.dx]
    return out


def load_lohner(d):
    """inverse of dump_lohner: (c list of acb, Q real arb_mat, q list of arb, var or None)"""
    c = [p_acb(v) for v in d["c"]]
    Q = p_mat_float(d["Q"])
    q = [arb(x) for x in d["q"]]
    var = None
    if "cV" in d:
        cV = acb_mat([[p_acb(v) for v in row] for row in d["cV"]])
        QV = p_mat_float(d["QV"])
        qV = arb_mat([[arb(x) for x in row] for row in d["qV"]])
        var = (cV, QV, qV)
    return c, Q, q, var

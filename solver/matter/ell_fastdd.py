#!/usr/bin/env python3
"""Block 12 (horizon) of ell_ddm.DDEll.mode_matrix in double-double by the stage-matrix variational map (#1025).

ell_ddm carries block 12's nf tangents as dual numbers: every double-double product of the linearised right side acts on
the nf tangent rows, with 12 Ozaki products against the tau operators per RK4 stage. Here the horizon series and its
evaluation at 1 - u0 are ell_ddm's own (DDEll.ssh_series on dual numbers, unchanged), the base state is integrated
with ell_ddm's value arithmetic (the same operations in the same order, so the same base trajectory), and each stage's
linearised map on coefficient rows is formed as a double-double (blk, blk) matrix
    J[g rows, f cols] = (B_g * coef_fg) @ S_f        (B_g, S_f ell_ddm's exact tau operators; coef_fg (1, N))
and applied to the tangents as one Ozaki product dc @ J (ell_ddm.Op). In exact arithmetic this equals ell_ddm's
((dc_g @ B_g) * coef_fg) @ S_f summed over g; the products with the base state's coefficients are formed on the n_g rows
of B_g instead of the nf tangent rows. The coefficients are those of ell_fastm._base (lambda in the tangents' tau
derivative), in double-double.

    python ell_fastdd.py check STUB.jsonl STATE.npy D LAM [--mutate=nolam|jround] [--hybrid=1]
        block 12 against DDEll.mode_matrix(only=[12]): max |difference| / max |block| <= 1e-20 (float64 rounding of this
        block is about 1e-10, ell_fastm), with both timings; --hybrid=1 also compares the hybrid determinant (float64
        blocks 0-11 by ell_fastm, this block 12 against ell_ddm's): the same dd_logdet sign and ln|det| within 1e-8.
        Exit 0 only if every comparison passes. Controls: --mutate=nolam (lambda dropped from the tangent coefficients)
        and --mutate=jround (each stage's J rounded to float64) must fail.
    python ell_fastdd.py selftest [--mutate=nolam|jround]
        on a k_c 15 state (ell_fastm._toy_state): the d/dtau basis rows against DDEll's Dspec + phys (1e-28), and block 12
        against DDEll's at lambda 0 and 0.9 (1e-20). Exit 0 only if every comparison passes."""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ddnp as dd                                    # noqa: E402
import ell_ddm                                       # noqa: E402
from ell_ddm import DT, Op, _eval_series             # noqa: E402

MUTATE = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--mutate=")), "")

# per output field: (input field, coefficient key) and the d/dtau term's (input field, key); fields p, m, a, c, psi = 0..4
_TERMS = ((((0, "pP"), (1, "pM"), (2, "pA"), (3, "pC"), (4, "pQ")), (0, "pD")),
          (((0, "mP"), (1, "mM"), (2, "mA"), (3, "mC"), (4, "mQ")), (1, "mD")),
          (((2, "aA"), (0, "aP"), (1, "aM"), (4, "aQ")), None),
          (((3, "cC"), (2, "cA"), (4, "cQ")), None),
          (((0, "hx"), (1, "hx")), None))


class DDJac:
    """block 12 of DDEll E by double-double stage matrices"""

    def __init__(self, E):
        S = E.S
        self.E, self.S = E, S
        no, ne = S.n_odd, S.n_even
        self.cut = np.cumsum([0, no, no, ne, ne, no])
        Bo, Be = E.B["odd"].W, E.B["even"].W
        mo = len(S.kodd)
        k = np.asarray(S.kodd, float)[:, None]
        # d/dtheta of the odd basis rows (DDEll._dcoef: (re, im) -> (-k im, k re)): re_k row -> k B[im_k], im_k -> -k B[re_k]
        top = dd.mul_d(dd.R(Bo.h[mo:], Bo.l[mo:]), k)
        bot = dd.mul_d(dd.R(-Bo.h[:mo], -Bo.l[:mo]), k)
        self.BDo = dd.R(np.concatenate([top.h, bot.h]), np.concatenate([top.l, bot.l]))
        self.Bg = (Bo, Bo, Be, Be, Bo)
        self.Sf = (E.Sp["odd"], E.Sp["odd"], E.Sp["even"], E.Sp["even"], E.Sp["odd"])

    def stage(self, x, s, dc, lam):
        """the base right side (DDEll.rhs on values, as a list of DT) and the tangents' right side dc @ J (dd.R)"""
        E, S = self.E, self.S
        n, L = S.n, S.L
        p, m, a, c, q = (E.phys(s[f], E.PAR[f]) for f in range(5))
        Dp = E.phys(E.Dspec(s[0], "odd"), "odd")
        Dmm = E.phys(E.Dspec(s[1], "odd"), "odd")
        a2 = a * a
        g = a2 * q * q
        cx = c * ((n - 1) * (a2 - 1.0) - (2 * L) * g) / x
        ax = a * ((p * p + m * m) + (2 * L) * g + (n - 1) * (1.0 - a2)) / (2 * x)
        qx = (p + m) / (2 * x)
        pc = n * c * (p + m) / (2 * x)
        lt = L * c * a2 * q / x
        px = (-Dp + cx * p - c * p / x + pc - lt) / (x - c)
        mx = (-Dmm - cx * m + c * m / x - pc + lt) / (x + c)
        base = [E.spec(px, "odd"), E.spec(mx, "odd"), E.spec(ax, "even"), E.spec(cx, "even"), E.spec(qx, "odd")]
        # the linearisation's coefficients (ell_fastm._base, in double-double)
        qq = q * q
        W = ((n - 1) * (a2 - 1.0) - (2 * L) * g) / x
        U = ((p * p + m * m) + (2 * L) * g + (n - 1) * (1.0 - a2)) / (2 * x)
        ip = 1.0 / (x - c)
        im = 1.0 / (x + c)
        wA = (a * (2.0 * (n - 1)) - a * qq * (4.0 * L)) / x
        wQ = a2 * q * (-4.0 * L) / x
        cxA, cxQ = c * wA, c * wQ
        pcC, pcPM = (p + m) * n / (2 * x), c * n / (2 * x)            # divisions in double-double (x is a double)
        ltC, ltA, ltQ = a2 * q * L / x, c * a * q * (2.0 * L) / x, c * a2 * L / x
        hx = DT(dd.div(dd.R(np.ones((1, S.N))), dd.R(np.full((1, S.N), 2 * x))))
        lt_ = 0.0 if MUTATE == "nolam" else float(lam)
        cf = dict(pP=(cx - c / x + pcPM - lt_) * ip, pM=pcPM * ip, pA=(cxA * p - ltA) * ip,
                  pC=(W * p - p / x + pcC - ltC + px) * ip, pQ=(cxQ * p - ltQ) * ip, pD=ip * (-E.om),
                  mM=(-cx + c / x - pcPM - lt_) * im, mP=(-pcPM) * im, mA=(ltA - cxA * m) * im,
                  mC=(m / x - W * m - pcC + ltC - mx) * im, mQ=(ltQ - cxQ * m) * im, mD=im * (-E.om),
                  aA=U + a * (a * qq * (2.0 * L) - a * (n - 1.0)) / x, aP=a * p / x, aM=a * m / x,
                  aQ=a * a2 * q * (2.0 * L) / x, cC=W, cA=cxA, cQ=cxQ, hx=hx)
        k = self.cut
        blk = k[-1]
        N = S.N
        Jh = np.empty((blk, blk))
        Jl = np.empty((blk, blk))
        for f, (terms, dterm) in enumerate(_TERMS):
            Yh = np.zeros((blk, N))
            Yl = np.zeros((blk, N))
            for gi, key in terms:
                Bg = self.Bg[gi]
                y = dd.mul(Bg, cf[key].v)
                if dterm is not None and dterm[0] == gi:
                    y = dd.add(y, dd.mul(self.BDo, cf[dterm[1]].v))
                Yh[k[gi]:k[gi + 1]] = y.h
                Yl[k[gi]:k[gi + 1]] = y.l
            Jf = self.Sf[f].rmul(dd.R(Yh, Yl))
            Jh[:, k[f]:k[f + 1]] = Jf.h
            Jl[:, k[f]:k[f + 1]] = Jf.l
        if MUTATE == "jround":
            Jl[:] = 0.0
        return base, Op(dd.R(Jh, Jl)).rmul(dc)

    def integrate_u(self, s, dc, xa, xb, steps, lam):
        """DDEll.integrate_u (its _rk4) on the base values and the tangent rows dc"""
        ya, yb = np.log(1 - xa), np.log(1 - xb)
        h = (yb - ya) / steps
        h6 = dd.div(dd.R(np.array(h)), dd.R(np.array(6.0)))

        def f(y, st, t):
            e = float(np.exp(y))
            b, dk = self.stage(1 - e, st, t, lam)
            return [r * (-e) for r in b], dd.mul_d(dk, -e)
        ax = self.E._axpy
        y = ya
        for _ in range(steps):
            k1, d1 = f(y, s, dc)
            k2, d2 = f(y + h / 2, ax(s, k1, h / 2), dd.add(dc, dd.mul_d(d1, h / 2)))
            k3, d3 = f(y + h / 2, ax(s, k2, h / 2), dd.add(dc, dd.mul_d(d2, h / 2)))
            k4, d4 = f(y + h, ax(s, k3, h), dd.add(dc, dd.mul_d(d3, h)))
            inc = [a + b * 2.0 + c * 2.0 + d for a, b, c, d in zip(k1, k2, k3, k4)]
            s = [si + ii * h6 for si, ii in zip(s, inc)]
            dinc = dd.add(dd.add(dd.add(d1, dd.mul_d(d2, 2.0)), dd.mul_d(d3, 2.0)), d4)
            dc = dd.add(dc, dd.mul(dinc, h6))
            y += h
        return s, dc

    def block12(self, z, lam):
        """(nf, blk) tangent rows of block 12 (the transpose of its M block), as DDEll.block(12)"""
        E, S = self.E, self.S
        no, ne = S.n_odd, S.n_even
        blocks, offs = S._block_list()
        c0, nf, fun, tgt = blocks[-1]
        E.om = 2 * np.pi / z[-1]
        E.lam = float(lam)
        try:
            mS, aS, qS = E._inputs(z[c0:c0 + nf], [no, ne, no], ["odd", "even", "odd"])
            Sx = E.ssh_series(E.phys(mS, "odd"), E.phys(aS, "even"), E.phys(qS, "odd"))
            st = [E.spec(_eval_series(Sf, S.u0), E.PAR[f]) for f, Sf in enumerate(Sx)]
        finally:
            E.lam = 0.0
        s = [DT(x.v) for x in st]
        dc = dd.R(np.concatenate([x.t.h for x in st], axis=1), np.concatenate([x.t.l for x in st], axis=1))
        if S.long:
            return dc
        E.lam = 0.0
        _, dc = self.integrate_u(s, dc, 1 - S.u0, S.nodes[-1], S._steps(1 - S.u0, S.nodes[-1]), lam)
        return dc

    def mode_matrix(self, z, lam):
        """DDEll.mode_matrix(z, lam, only=[12]) by this block 12"""
        S = self.S
        sizes, K, L = S.layout()
        nv = L - 1
        Mh = np.zeros((nv, nv))
        Ml = np.zeros((nv, nv))
        blocks, offs = S._block_list()
        blk = 3 * S.n_odd + 2 * S.n_even
        node0 = offs[2]
        c0, nf, fun, tgt = blocks[-1]
        b = len(blocks) - 1
        dc = self.block12(z, lam)
        r0 = b * blk
        Mh[r0:r0 + blk, c0:c0 + nf] = dc.h.T
        Ml[r0:r0 + blk, c0:c0 + nf] = dc.l.T
        t0_ = node0 + blk * tgt
        Mh[r0:r0 + blk, t0_:t0_ + blk] -= np.eye(blk)
        return dd.R(Mh, Ml)


def block_slice(S):
    blocks, offs = S._block_list()
    blk = 3 * S.n_odd + 2 * S.n_even
    c0, nf, fun, tgt = blocks[-1]
    r0 = (len(blocks) - 1) * blk
    return (slice(r0, r0 + blk), slice(c0, c0 + nf))


def rel_dd(A, B, sl):
    """max |A - B| / max |B| over the slice, the difference formed from both parts"""
    d = (A.h[sl] - B.h[sl]) + (A.l[sl] - B.l[sl])
    return float(np.max(np.abs(d)) / np.max(np.abs(B.h[sl])))


def selftest():
    import ell_fastm
    S, z = ell_fastm._toy_state()
    E = ell_ddm.DDEll(S)
    Jc = DDJac(E)
    ok = True
    rng = np.random.default_rng(4)
    c = dd.R(rng.standard_normal((3, S.n_odd)))
    E.om = 1.0
    ref = E.phys(E.Dspec(DT(c), "odd"), "odd").v
    got = Op(Jc.BDo).rmul(c)
    e = float(np.max(np.abs((got.h - ref.h) + (got.l - ref.l))) / np.max(np.abs(ref.h)))
    g = e <= 1e-28
    ok &= g
    print(f"d/dtau basis rows against DDEll's Dspec + phys: {e:.2e} (<= 1e-28) {'ok' if g else 'FAIL'}", flush=True)
    sl = block_slice(S)
    for lam in (0.0, 0.9):
        with np.errstate(all="ignore"):
            A = Jc.mode_matrix(z, lam)
            B = E.mode_matrix(z, lam, only=[12])
        r = rel_dd(A, B, sl)
        g = r <= 1e-20
        ok &= g
        print(f"block 12 at lambda {lam}: max |fast - DDEll| / max |DDEll| = {r:.2e} (<= 1e-20) {'ok' if g else 'FAIL'}",
              flush=True)
    print(f"selftest (mutate {MUTATE or 'none'}): {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)


def checkcmd(argv):
    import ell_fastm
    import multidouble as md
    args = [a for a in argv if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in argv if a.startswith("--") and "=" in a)
    recf, st, D, lam = args[0], args[1], float(args[2]), float(args[3])
    S, z, rec, same, msg = ell_fastm.load_system(recf, st, D)
    print(msg, flush=True)
    if not same:
        sys.exit(3)
    E = ell_ddm.DDEll(S)
    Jc = DDJac(E)
    with np.errstate(all="ignore"):
        t0 = time.time()
        A = Jc.mode_matrix(z, lam)
        ta = time.time() - t0
        t0 = time.time()
        B = E.mode_matrix(z, lam, only=[12])
        tb = time.time() - t0
    sl = block_slice(S)
    r = rel_dd(A, B, sl)
    ok = r <= 1e-20
    print(f"block 12 at lambda {lam!r} (mutate {MUTATE or 'none'}): max |fast - DDEll| / max |DDEll| = {r:.3e} (<= 1e-20) "
          f"{'ok' if ok else 'FAIL'}; times: fast {ta:.1f} s, DDEll {tb:.1f} s, ratio {tb / ta:.2f}", flush=True)
    res = dict(D=D, kc=rec["kc"], lam=lam, mutate=MUTATE, rel=r, t_fast=ta, t_ddell=tb)
    if opts.get("hybrid") == "1":
        with np.errstate(all="ignore"):
            Mf = ell_fastm.mode_matrix_fast(S, z, lam)
        out = []
        for X in (A, B):
            H, Hl = Mf.copy(), np.zeros_like(Mf)
            H[sl] = X.h[sl]
            Hl[sl] = X.l[sl]
            out.append(md.dd_logdet(H, Hl, True)[:2])
        g = out[0][0] == out[1][0] and abs(out[0][1] - out[1][1]) <= 1e-8
        ok &= g
        print(f"hybrid: with this block 12 {out[0][0]:+.0f} {out[0][1]:.12f}, with DDEll's {out[1][0]:+.0f} {out[1][1]:.12f}, "
              f"|d ln| {abs(out[0][1] - out[1][1]):.2e} (<= 1e-8) {'ok' if g else 'FAIL'}", flush=True)
        res.update(hyb=[[float(v) for v in o] for o in out])
    if "out" in opts:
        open(opts["out"], "w").write(json.dumps(dict(res, ok=bool(ok))) + "\n")
    print(f"check: {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "check":
        checkcmd(sys.argv[2:])
    elif cmd == "selftest":
        selftest()
    else:
        print(__doc__)
        sys.exit(2)

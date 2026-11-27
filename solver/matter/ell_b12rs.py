#!/usr/bin/env python3
"""Block 12 (horizon) of ell_ddm.DDEll.mode_matrix in Rust: the ellrs module (solver/matter/ellrs, #1102).

ellrs runs DDEll.block(12, X) operation for operation: the parity-restricted Fourier state, dual-number tangents,
ddnp.py's double-double algorithms (on multidouble's f64 lanes), ell_ddm.Op's Ozaki products, the horizon series with its
periodic solves, and the frozen-step RK4 of the horizon segment. This driver hands it what DDEll computes in Python: the
tau operators and their Ozaki slices (from a DDEll), the RK4 abscissae exp(y) by numpy, and the periodic solves' f64 LU
by scipy (lu="scipy", the default, called back from Rust), so that on one machine the block is DDEll's bit for bit.
lu="native" factors in Rust instead (partial pivoting in another order: the solves' last f64 bits differ, then the
double-double refinement).

    python ell_b12rs.py check STUB.jsonl STATE.npy D LAMS [--mutate=nolam|harm] [--lu=scipy|native] [--threads=T]
                       [--bar=1e-28] [--out=OUT.jsonl]
        Rust's block 12 against DDEll.mode_matrix(only=[12]) at each lambda of LAMS (comma-separated): the number of
        entries equal bit for bit (hi and lo parts), the largest |difference| / |entry| over the entries (the
        difference formed from both parts), and the largest |difference| / max |block|; both timings, in one process.
        PASS when the per-entry relative difference is at most --bar at every lambda (exit 0, else 1; 3 when the state
        does not reproduce its stub record). Controls (each must FAIL): --mutate=nolam (Rust's tangents without lambda)
        and --mutate=harm (the top odd harmonic dropped from Rust's tau operators).
    python ell_b12rs.py selftest [--mutate=nolam|harm]
        on ell_fastm's k_c 15 toy state: each operator's Ozaki product against ell_ddm.Op.rmul (bit for bit), and block
        12 against DDEll's at lambda 0 and 0.9 (bit for bit, with lu="scipy"). Exit 0 only if every comparison passes.

Needs ellrs importable (solver/matter/ellrs/build.sh, or PYTHONPATH at its site folder)."""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ddnp as dd                                    # noqa: E402

MUTATE = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--mutate=")), "")


def rk_grid(S):
    """the horizon segment's RK4 in y = ln(1 - x) as DDEll.integrate_u steps it: h and, per step, the floats
    exp(y), exp(y + h/2), exp(y + h) (numpy's exp, the reference's abscissae)"""
    xa, xb = 1 - S.u0, S.nodes[-1]
    steps = S._steps(xa, xb)
    if steps is None:
        raise ValueError("ell_b12rs needs frozen RK step counts (S.freeze_from)")
    ya, yb = np.log(1 - xa), np.log(1 - xb)
    h = (yb - ya) / steps
    y = ya
    e = np.empty((steps, 3))
    for i in range(steps):
        e[i] = float(np.exp(y)), float(np.exp(y + h / 2)), float(np.exp(y + h))
        y += h
    return float(h), e


def scipy_lu(om, Dm_h):
    """DDEll._solve's f64 LU (scipy) as a callable for ellrs: diag -> (R -> solves per row of R)"""
    import scipy.linalg as sla
    DmT = om * Dm_h.T

    def factor(diag):
        lu = sla.lu_factor(DmT + np.diag(diag))
        return lambda R: np.ascontiguousarray(sla.lu_solve(lu, R.T).T)
    return factor


class RsB12:
    """block 12 of DDEll E (its operators and slices) in Rust"""

    def __init__(self, S, E=None, threads=0, mutate=None):
        import ellrs
        import ell_ddm
        if S.long:
            raise ValueError("ell_b12rs covers the horizon segment's RK4 (S.long is False)")
        self.S = S
        self.E = E if E is not None else ell_ddm.DDEll(S)
        if ell_ddm.NSLICE != 5:
            raise ValueError("ellrs has 5 Ozaki slices; ell_ddm runs with %d" % ell_ddm.NSLICE)
        E = self.E
        sl = lambda op: [np.ascontiguousarray(s) for s in op.sl]
        self.R = ellrs.Block12(float(S.n), float(S.L), int(S.N), [float(k) for k in S.kodd], int(S.Ks), float(S.u0),
                               sl(E.B["odd"]), sl(E.B["even"]), sl(E.Sp["odd"]), sl(E.Sp["even"]), sl(E.Dm),
                               np.ascontiguousarray(E.Dm_h), threads=threads,
                               mutate=MUTATE if mutate is None else mutate)
        self.h, self.e = rk_grid(S)
        blocks, offs = S._block_list()
        self.c0, self.nf, _, self.tgt = blocks[-1]
        self.blk = 3 * S.n_odd + 2 * S.n_even
        self.r0 = (len(blocks) - 1) * self.blk

    def block12(self, z, lam, lu="scipy", chunks=0):
        """(nf, blk) tangent rows of block 12 (the transpose of its M block) as dd.R, and the end state's value row"""
        om = 2 * np.pi / z[-1]
        f = scipy_lu(om, self.E.Dm_h) if lu == "scipy" else None
        th, tl, vh, vl = self.R.block(np.ascontiguousarray(z[self.c0:self.c0 + self.nf]), float(om), float(lam), self.h,
                                      self.e, f, chunks)
        return dd.R(th, tl), dd.R(vh, vl)

    def mode_matrix(self, z, lam, lu="scipy"):
        """DDEll.mode_matrix(z, lam, only=[12]) by this block 12"""
        S = self.S
        nv = S.layout()[2] - 1
        Mh = np.zeros((nv, nv))
        Ml = np.zeros((nv, nv))
        t, _ = self.block12(z, lam, lu)
        r0, c0, blk = self.r0, self.c0, self.blk
        Mh[r0:r0 + blk, c0:c0 + self.nf] = t.h.T
        Ml[r0:r0 + blk, c0:c0 + self.nf] = t.l.T
        t0_ = S._block_list()[1][2] + blk * self.tgt
        Mh[r0:r0 + blk, t0_:t0_ + blk] -= np.eye(blk)
        return dd.R(Mh, Ml)


def compare(A, B):
    """A against the reference B (dd.R of one shape): entries equal bit for bit (both parts), the largest |A - B| /
    |B| (an entry with B = 0 and A != 0 counts as inf), the largest |A - B| / max |B|"""
    d = np.abs((A.h - B.h) + (A.l - B.l))
    same = int(np.sum((A.h == B.h) & (A.l == B.l)))
    ref = np.abs(B.h)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(ref > 0, d / np.where(ref > 0, ref, 1.0), np.where(d > 0, np.inf, 0.0))
    return dict(same=same, n=int(A.h.size), rel_entry=float(np.max(rel)), rel_max=float(np.max(d) / np.max(ref)))


def block_of(Md, S):
    """the (nf, blk) tangent rows inside a mode matrix (dd.R)"""
    blocks, offs = S._block_list()
    c0, nf, _, _ = blocks[-1]
    blk = 3 * S.n_odd + 2 * S.n_even
    r0 = (len(blocks) - 1) * blk
    return dd.R(Md.h[r0:r0 + blk, c0:c0 + nf].T.copy(), Md.l[r0:r0 + blk, c0:c0 + nf].T.copy())


def checkcmd(argv):
    import ell_ddm
    from ell_fastm import load_system
    args = [a for a in argv if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in argv if a.startswith("--") and "=" in a)
    recf, st, D = args[0], args[1], float(args[2])
    lams = [float(v) for v in args[3].split(",")]
    bar = float(opts.get("bar", 1e-28))
    lu = opts.get("lu", "scipy")
    S, z, rec, same, msg = load_system(recf, st, D)
    print(msg, flush=True)
    if not same:
        sys.exit(3)
    t0 = time.time()
    E = ell_ddm.DDEll(S)
    R = RsB12(S, E, threads=int(opts.get("threads", 0)))
    import ellrs
    print(f"N {S.N}, n_odd {S.n_odd}, n_even {S.n_even}, horizon steps {len(R.e)}, Ks {S.Ks}, u0 {S.u0}; operators "
          f"{time.time() - t0:.1f} s; ellrs {ellrs.isa()}, {R.R.threads} threads, lu {lu}, mutate {MUTATE or 'none'}",
          flush=True)
    ok = True
    for lam in lams:
        with np.errstate(all="ignore"):
            t0 = time.time()
            Tr, _ = R.block12(z, lam, lu)
            tr = time.time() - t0
            t0 = time.time()
            Tp = block_of(E.mode_matrix(z, lam, only=[12]), S)
            tp = time.time() - t0
        c = compare(Tr, Tp)
        good = c["rel_entry"] <= bar
        ok &= good
        print(f"lambda {lam!r}: {c['same']} of {c['n']} entries bit for bit; max |d|/|entry| {c['rel_entry']:.3e}, "
              f"max |d|/max|block| {c['rel_max']:.3e} (bar {bar:g} per entry): {'PASS' if good else 'FAIL'}; "
              f"Rust {tr:.2f} s, DDEll {tp:.1f} s, ratio {tp / tr:.1f}", flush=True)
        if "out" in opts:
            with open(opts["out"], "a") as fo:
                fo.write(json.dumps(dict(D=D, kc=rec["kc"], lam=lam, state=os.path.basename(st), mutate=MUTATE or None,
                                         lu=lu, threads=R.R.threads, isa=ellrs.isa(), **c, bar=bar,
                                         verdict="PASS" if good else "FAIL", sec_rust=tr, sec_ddell=tp)) + "\n")
    print(f"check: {'PASS' if ok else 'FAIL'}" + (f" [mutate {MUTATE}]" if MUTATE else ""), flush=True)
    sys.exit(0 if ok else 1)


def selftest():
    import ell_ddm
    import ell_fastm
    S, z = ell_fastm._toy_state()
    E = ell_ddm.DDEll(S)
    R = RsB12(S, E)
    ok = True
    g = np.random.default_rng(3)
    for name, op in (("bodd", E.B["odd"]), ("beven", E.B["even"]), ("sodd", E.Sp["odd"]), ("seven", E.Sp["even"]),
                     ("dm", E.Dm)):
        X = dd.R(g.standard_normal((9, op.n)) * np.exp2(g.integers(-30, 30, (9, 1))))
        X = dd.add(X, dd.R(X.h * 1e-17 * g.standard_normal(X.shape)))
        ref = op.rmul(X)
        h, l = R.R.ozaki(name, X.h, X.l)
        c = compare(dd.R(h, l), ref)
        good = c["same"] == c["n"]
        ok &= good
        print(f"Ozaki {name}: {c['same']} of {c['n']} entries bit for bit {'ok' if good else 'FAIL'}", flush=True)
    for lam in (0.0, 0.9):
        with np.errstate(all="ignore"):
            Tr, _ = R.block12(z, lam)
            Tp = block_of(E.mode_matrix(z, lam, only=[12]), S)
        c = compare(Tr, Tp)
        good = c["same"] == c["n"]
        ok &= good
        print(f"block 12 at lambda {lam}: {c['same']} of {c['n']} entries bit for bit, max |d|/|entry| "
              f"{c['rel_entry']:.2e} {'ok' if good else 'FAIL'}", flush=True)
    print(f"selftest (mutate {MUTATE or 'none'}): {'PASS' if ok else 'FAIL'}", flush=True)
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

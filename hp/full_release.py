#!/usr/bin/env python3
"""Release the held harmonics: the quad-double Newton (newton_qd, plain) of the full system (kfreeze 13, every p1
harmonic solved) from a pinned chain state, as T75's protocol checks its pinned states (README: "Delta after one
full-system Newton step"; notes/QUAD_DOUBLE.md 12, the full-system convergence of T75's states). Prints the
full-system residual at the pinned state, each step, Delta before and after with the first step's dDelta, and the
1/D series fitted on D <= 67.77 (dd_fits_mp, degree 20) at that D for reference. --kfreeze K (default 13) frees the
harmonics above K only; --kfreeze 3 reproduces the chain's own system, whose residual at its state must be at the floor
(the control). --octo 1 runs the Newton in octo-double instead (octo_corrector.newton_od on the kernel's newton_o,
M = 32 only), whose LU resolves the mass mode's weak direction past D ~ 185 where the quad-double one does not
(notes/OCTO_DOUBLE.md 5); the state's parts are promoted to eight. The test (notes/QUAD_DOUBLE.md 13.29): at 79.77 the
released Delta must land on the series, about 5.0e-8 above the chain, where the quad-double release stalled.
--junk w (with --octo 1) freezes the junk among the w weakest directions of the first Jacobian (no Delta content;
octo_corrector.junk_subspace_o), as newton_qd's junk mode does: whether the release then converges is the test of
notes/QUAD_DOUBLE.md 13.30.
    cd hp && python3 full_release.py RAWDIR STATE [STATE ...] [--kfreeze 13] [--threads 2] [--it 10] [--octo 0] [--junk 0]
solver() and release() are the step for one state, which release_planted.py runs on planted states.
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402
from hpkernel.driver import make_solver, newton_qd, load_q, qmax  # noqa: E402
from pinning import PinnedQ, PinnedSolver, band_coeffs, band_field  # noqa: E402
import dd_fits_mp as F  # noqa: E402
mp.mp.dps = 60


class FullO:
    """the full system's octo-double Newton, as newton_od calls it: the kernel's newton_o and residual_o, no offset"""

    def __init__(self, s):
        self.s = s

    def __getattr__(self, name):
        return getattr(self.s, name)

    def residual_o(self, z, mutate=0):
        from octo_corrector import o8
        return o8(self.s.residual_o(list(o8(z)), mutate=mutate))

    def newton_o(self, z, mutate=0):
        from octo_corrector import o8
        d, fh, fl, pmin, pmax = self.s.newton_o(list(o8(z)), mutate=mutate)
        return o8(d), fh, fl, pmin, pmax


_series = []


def series(D):
    """the 1/D series of degree 20 fitted on the double-double points 10 <= D <= 67.77, at D"""
    if not _series:
        pts, _ = F.load_points()
        Ds = F.clean(pts, sorted(d for d in pts if 10 <= d <= 67.77), verbose=False)
        _series.append(F.fit(pts, Ds, "1/D", 20)[0])
    return _series[0](D)


def held_ks(kf):
    """the held harmonics of a chain state with kfreeze kf: p1's odd ones in (kf, 13]"""
    return [k for k in range(kf + 1, 14) if k % 2 == 1]


def solver(m, z, kfreeze, threads, held=None, segs=None):
    """(solver, segs, D) at the state's D with kfreeze. At the state's own kfreeze it is the chain's own system, its
    frozen rows reading coefficient - held value, with held ((a_k, b_k) per held harmonic) the state's own unless given.
    segs: a plan to reuse (default: planned at z)"""
    D, M = (mp.mpf(m["D_str"]) if "D_str" in m else mp.mpf(float(m["D"]))), int(m["M"])
    s = make_solver(D, M, nodes=list(m["nodes"]), kt=m.get("kt", 36), kc=m.get("kc", 146), ks=m.get("ks", 130), threads=threads,
                    kfreeze=kfreeze)
    if segs is None:
        segs = s.plan(np.asarray(z[0]), np.asarray(z[1]), 1e-32, 0.05)
    s.set_segs(segs)
    kf_state = int(m.get("kfreeze", 3))
    if kfreeze == kf_state:
        ks_held = held_ks(kf_state)
        if held is None:
            held = band_coeffs(np.asarray(z[0])[:M], np.asarray(z[1])[:M], M, ks_held)
        hh, hl = band_field(held, ks_held, M)
        ps = PinnedSolver(s, M, hh, hl)
        s = PinnedQ(s, ps.ch, ps.cl)
    return s, segs, D


def release(st, z, m, s, D, kfreeze, octo, junk, it):
    """the Newton of solver s from the four-part state z, printed as below; returns (Delta before, after, the first
    step's dDelta, converged)"""
    M = int(m["M"])
    d0 = sum(mp.mpf(float(p[-1])) for p in z)
    if octo:
        from octo_corrector import PinnedO, newton_od, o8_of_q, omax
        so = PinnedO(s.s, s.ch, s.cl) if isinstance(s, PinnedQ) else FullO(s)
        z8 = o8_of_q(z)
        F0 = so.residual_o(z8)
        print(f"{st}: D {mp.nstr(D, 12)}, M {M}, kfreeze {kfreeze}, octo-double: residual at the pinned state "
              f"{omax(F0[:2]):.2e}, Delta {mp.nstr(d0, 17)}", flush=True)
        z2, hist, conv = newton_od(so, z8, tol=1e-50, it_max=it, verbose=True, junk_w=junk)
    else:
        F0 = s.residual_q(*z)
        print(f"{st}: D {mp.nstr(D, 12)}, M {M}, kfreeze {kfreeze}: residual at the pinned state {qmax(F0[:2]):.2e}, Delta {mp.nstr(d0, 17)}", flush=True)
        z2, hist, conv = newton_qd(s, z, tol=1e-40, it_max=it, verbose=True)
    d1 = sum(mp.mpf(float(p[-1])) for p in z2)
    ser = series(D)
    first = hist[0].get("dDelta") if hist else None
    print(f"  {'converged' if conv else 'not converged'} in {len(hist)} steps; Delta {mp.nstr(d1, 17)}, moved {float(d1 - d0):+.3e}"
          f" (first step's dDelta {first if first is None else format(first, '+.3e')}); series 1/D 20 {mp.nstr(ser, 17)}:"
          f" pinned - series {float(d0 - ser):+.3e}, released - series {float(d1 - ser):+.3e}", flush=True)
    return d0, d1, first, conv


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opt = dict(kfreeze=13, threads=2, it=10, octo=0, junk=0)
    for k in list(opt):
        if f"--{k}" in sys.argv:
            opt[k] = int(sys.argv[sys.argv.index(f"--{k}") + 1])
            args = [a for a in args if a != str(opt[k])]
    raw, states = args[0], args[1:]
    for st in states:
        z, m = load_q(os.path.join(raw, st))
        s, _, D = solver(m, z, opt["kfreeze"], opt["threads"])
        release(st, z, m, s, D, opt["kfreeze"], opt["octo"], opt["junk"], opt["it"])

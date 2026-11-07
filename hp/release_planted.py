#!/usr/bin/env python3
"""Planted test of full_release.py's step (notes/QUAD_DOUBLE.md 13.29): does the junk-frozen full-system step return a
small known bias of the held harmonics? From a chain state, the chain's own system is solved again with the cosine of
one held harmonic (5, the one Delta is most sensitive to, notes/HELD_HARMONICS.md) moved by delta, by the quad-double
Newton on the pinned system as the chain's corrector does. That moves the pinned Delta by b = Delta(delta) - Delta(0),
the planted bias (s_5a delta to first order: s_5a is 9.2e-3 at D 79.77 and 1.8e-2 at 200.27). The full system does
not hold the harmonics, so it is the same for both states, and the step from the moved state must add b less:
r = dDelta(delta) - dDelta(0) = -b. Pass, set before the run: |r / (-b) - 1| <= 0.02 for every delta, the step's
agreement on the trend-held chain (13.29). The pinned Newton's residual after its first step is quadratic in delta
(about 3e8 delta^2 at D 80, more at 100), so the move is made in --sub steps of at most that size, each solved from the
last (continuation), with --ls the quad-double Newton's acceptance rule (newton_qd: "dz", its default, stalls at
1e-10 at D 80; "F", backtracking on |F|). --pin od solves the pinned system in octo-double instead (PinnedO,
newton_od to --tol_od), where the quad-double LU fails (D 200: |dz| = 100 delta). --pin_junk W freezes the junk among
the W weakest directions in the pinned solve (newton_qd's junk mode, newton_od's junk_w: no Delta content), as the
release does; from D 150 the plain pinned Newton rejects its first step from a move of 1e-11. Every solve uses the plan
of the state itself. Each release prints
full_release.py's lines, the one of delta 0 being full_release.py's own run on the state. --mutate 1 (the control,
which must fail) takes r from the unmoved state's step instead of the moved one's.

    cd hp && python3 release_planted.py RAWDIR STATE [--delta 3e-9,1e-9,1e-10,1e-11] [--harm 5] [--threads 2] [--it 6]
        [--junk 4] [--octo 1] [--qd_it 12] [--tol_qd 1e-28] [--sub 0] [--ls dz] [--pin qd] [--pin_junk 0] [--tol_od 1e-40]
        [--verbose 0] [--mutate 0]

Prints one "result" line per delta (JSON). Exits 1 when a delta fails or a pinned solve does not converge.
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402
from hpkernel.driver import load_q, newton_qd  # noqa: E402
from pinning import band_coeffs  # noqa: E402
import full_release as R  # noqa: E402

TOL_RATIO = 0.02


def pin_solve(sp, z, opt):
    """the pinned system's Newton from the four-part z: quad-double (newton_qd, acceptance --ls) or octo-double
    (--pin od: PinnedO, newton_od), returning four parts"""
    if opt["pin"] == "od":
        from octo_corrector import PinnedO, newton_od, o8_of_q, q4_of_o
        z8, hist, conv = newton_od(PinnedO(sp.s, sp.ch, sp.cl), o8_of_q(z), tol=opt["tol_od"], it_max=opt["qd_it"],
                                   verbose=bool(opt["verbose"]), junk_w=opt["pin_junk"])
        return q4_of_o(z8), hist, conv
    zz, hist, conv = newton_qd(sp, z, tol=opt["tol_qd"], it_max=opt["qd_it"], verbose=bool(opt["verbose"]), ls=opt["ls"],
                               w=opt["pin_junk"], junk=opt["pin_junk"] > 0)
    return tuple(np.ascontiguousarray(np.asarray(p)) for p in zz), hist, conv


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    opt = dict(delta="3e-9,1e-9,1e-10,1e-11", harm=5, threads=2, it=6, junk=4, octo=1, qd_it=12, tol_qd=1e-28, sub=0.0,
               ls="dz", pin="qd", pin_junk=0, tol_od=1e-40, verbose=0, mutate=0)
    for k in list(opt):
        if f"--{k}" in argv:
            v = argv[argv.index(f"--{k}") + 1]
            opt[k] = type(opt[k])(float(v)) if isinstance(opt[k], int) else type(opt[k])(v)
            args = [a for a in args if a != v]
    if len(args) != 2:
        sys.exit(__doc__)
    raw, st = args
    deltas = [float(x) for x in opt["delta"].split(",")]
    z, m = load_q(os.path.join(raw, st))
    M, kf = int(m["M"]), int(m.get("kfreeze", 3))
    ks_held = R.held_ks(kf)
    if opt["harm"] not in ks_held:
        sys.exit(f"harmonic {opt['harm']} is not held at kfreeze {kf} ({ks_held})")
    i = ks_held.index(opt["harm"])
    held0 = band_coeffs(np.asarray(z[0])[:M], np.asarray(z[1])[:M], M, ks_held)
    print(f"{st}: held harmonic {opt['harm']} cosine {held0[i][0]:+.17e}; deltas {deltas}; mutate {opt['mutate']}", flush=True)
    full = lambda zz, segs: R.solver(m, zz, 13, opt["threads"], segs=segs)  # noqa: E731
    s13, segs, D = full(z, None)
    _, _, step0, _ = R.release(st, z, m, s13, D, 13, opt["octo"], opt["junk"], opt["it"])
    pin0 = sum(mp.mpf(float(p[-1])) for p in z)
    ok = step0 is not None
    for dl in deltas:
        n = max(1, int(np.ceil(abs(dl) / opt["sub"] - 1e-9))) if opt["sub"] > 0 else 1
        zd, conv = z, True
        print(f"pinned solve ({opt['pin']}, ls {opt['ls']}), harmonic {opt['harm']} cosine moved by {dl:+.6e} in {n} steps:",
              flush=True)
        for j in range(1, n + 1):
            held = [list(c) for c in held0]
            held[i][0] = float(held0[i][0] + dl * j / n)
            dl_true = held[i][0] - held0[i][0]                           # the float64 move actually made
            sp, _, _ = R.solver(m, z, kf, opt["threads"], held=[tuple(c) for c in held], segs=segs)
            zd, hist, conv = pin_solve(sp, zd, opt)
            print(f"  move {j}/{n}, {dl_true:+.6e}: {len(hist)} Newton steps, |F| {hist[-1]['F'] if hist else float('nan'):.2e}, "
                  f"{'converged' if conv else 'not converged'}", flush=True)
            if not conv:
                break
        if not conv:
            print("  the pinned solve did not converge", flush=True)
            ok = False
            continue
        b = sum(mp.mpf(float(p[-1])) for p in zd) - pin0                 # the planted bias
        if opt["mutate"]:
            step = step0                                                 # the control: the unmoved state's step
        else:
            sd, _, _ = full(zd, segs)
            _, _, step, _ = R.release(f"{st} moved {dl_true:+.1e}", zd, m, sd, D, 13, opt["octo"], opt["junk"], opt["it"])
        if step is None:
            ok = False
            continue
        r = step - step0
        ratio = r / float(-b)
        good = abs(ratio - 1) <= TOL_RATIO
        ok &= good
        print("result " + json.dumps(dict(state=st, D=float(D), harm=opt["harm"], delta=dl_true, bias=float(b),
                                         s=float(b) / dl_true, step0=step0, step=step, response=r, ratio=ratio,
                                         mutate=opt["mutate"], pass_=good)), flush=True)
        print(f"  planted bias {float(b):+.4e} (s {float(b) / dl_true:.4e}); step {step:+.4e} - step at 0 {step0:+.4e} = "
              f"{r:+.4e}; ratio to -bias {ratio:.4f}: {'pass' if good else 'FAIL'}", flush=True)
    print(f"{'pass' if ok else 'FAIL'}: {len(deltas)} deltas at D {mp.nstr(D, 12)}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

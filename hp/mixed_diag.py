#!/usr/bin/env python3
"""Why the f64 Jacobian agrees with the double-double one to ~1e-7 at M >= 48 but to ~1e-14 at M = 16
(notes/MIXED_PRECISION.md, "Jacobian agreement"). Each section compares Solver.jacobian_f64 with the hi part of
Solver.jacobian at the D = 3.7 states (solver settings from the state's meta), per column (max |dJ| / max |J|):

  steps     M = 48, Taylor plans with hmax = 0.05 (the saved states' plan) and 0.01, for the all-f64 lanes (f64x1) and
            a double-double base trajectory (dd1): the error per residual block (centre, segments, horizon);
  harmonics M = 48, plan hmax = 0.05: for three columns of segment 11 (node 10 p lane 8, p lane 30, a lane 5), the
            error and the column itself by tau-harmonic, in the p and a output rows;
  M         M = 16 and 32 (the M = 48 state resampled) and 48, 64, 80 (their own states), f64x4 and dd4;
  orders    M = 48: the horizon block's error with the horizon series order ks lowered from the state's 250, and the
            centre block's with the centre order kc lowered from 100 (the f64 tangents need far fewer orders; the
            f64 lanes stop the series by themselves once the tangent terms are below f64 rounding, multidouble's TangentTail).

    cd hp && python3 mixed_diag.py [steps] [harmonics] [M] [orders]      (all sections by default; ~3 min on 2 cores)
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from hpkernel.driver import load, make_solver, resample_z  # noqa: E402

STATE = lambda M: os.path.join(HERE, "data", "lowD_states", f"lowDv_D3.7000_M{M}.npz")  # noqa: E731


def setup(M, hmax=0.05, src=None, **kw):
    zh, zl, m = load(STATE(src or M))
    K = len(m["nodes"])
    if (src or M) != M:
        zh, zl = resample_z(zh + zl, m["M"], M, K)
    args = dict(nodes=m["nodes"], kt=m["kt"], kc=m["kc"], ks=m["ks"], threads=2)
    args.update(kw)
    s = make_solver(m["D"], M, **args)
    s.set_segs(s.plan(zh, zl, 1e-30, hmax))
    return s, np.asarray(zh), np.asarray(zl), m, K


def per_block(Jh, Jf, M, K):
    out = []
    for b in range(K + 1):
        rows = slice(b * 4 * M, (b + 1) * 4 * M)
        cm = np.abs(Jh[rows]).max(axis=0)
        sel = cm > 0
        out.append((np.abs(Jf[rows] - Jh[rows]).max(axis=0)[sel] / cm[sel]).max())
    return out


def percol(Jh, Jf):
    cm = np.abs(Jh).max(axis=0)
    return float((np.abs(Jf - Jh).max(axis=0) / np.where(cm > 0, cm, 1)).max())


def steps():
    print("== steps: per-column error by residual block (centre, segments 1..12, horizon), M = 48")
    for hmax in (0.05, 0.01):
        s, zh, zl, m, K = setup(48, hmax)
        segs = s.get_segs()
        Jh = np.asarray(s.jacobian(zh, zl)[0])
        print(f" plan hmax {hmax}: steps per segment {[len(g) - 1 for g in segs]}, largest step per segment "
              f"{[round(float(max(np.diff(g))), 4) for g in segs]}")
        for v, kw in (("f64x1", dict(tangents=1)), ("dd1", dict(tangents=1, dd_base=True))):
            Jf = np.asarray(s.jacobian_f64(zh, zl, 0, **kw))
            pb = per_block(Jh, Jf, 48, K)
            print(f"  {v:6s}: max {max(pb):.1e}; by block " + " ".join(f"{x:.0e}" for x in pb), flush=True)


def harmonics():
    M = 48
    print("== harmonics: M = 48, plan hmax 0.05, segment 11 (node 10 -> node 11) rows; |coefficient| by harmonic")
    s, zh, zl, m, K = setup(M)
    Jh = np.asarray(s.jacobian(zh, zl)[0])
    Jf = np.asarray(s.jacobian_f64(zh, zl, 0, tangents=1))
    b = 11
    rows = slice(b * 4 * M, (b + 1) * 4 * M)
    o = 2 * M + 4 * M * (b - 1)
    for c, name in ((o + 8, "p lane 8"), (o + 30, "p lane 30"), (o + 2 * M + 5, "a lane 5")):
        col, d = Jh[rows, c], (Jf - Jh)[rows, c]
        print(f" column {c} (node 10 {name}): max |J| {np.abs(col).max():.1e}; max |dJ| in p, m, a, c rows: "
              + ", ".join(f"{np.abs(d[q * M:(q + 1) * M]).max():.1e}" for q in range(4)))
        da, ja = d[2 * M:3 * M], col[2 * M:3 * M]
        dp, jp = d[:M], col[:M]
        ev = lambda f: np.abs(np.fft.rfft(np.r_[f, f]))[::2][:25] / (2 * M)  # noqa: E731  even harmonics 0..48
        od = lambda f: np.abs(np.fft.rfft(np.r_[f, -f]))[1::2][:24] / (2 * M)  # noqa: E731  odd harmonics 1..47
        print("   a rows, error, harmonics 0, 2, .., 48: " + " ".join(f"{x:.0e}" for x in ev(da)))
        print("   a rows, J,     harmonics 0, 2, .., 48: " + " ".join(f"{x:.0e}" for x in ev(ja)))
        print("   p rows, error, harmonics 1, 3, .., 47: " + " ".join(f"{x:.0e}" for x in od(dp)))
        print("   p rows, J,     harmonics 1, 3, .., 47: " + " ".join(f"{x:.0e}" for x in od(jp)), flush=True)


def by_M():
    print("== M: per-column error and seconds (dd Jacobian / f64x4 / dd4), 2 threads")
    for M, src in ((16, 48), (32, 48), (48, 48), (64, 64), (80, 80)):
        s, zh, zl, m, K = setup(M, src=src)
        t = time.time(); Jh = np.asarray(s.jacobian(zh, zl)[0]); td = time.time() - t
        t = time.time(); J4 = np.asarray(s.jacobian_f64(zh, zl, 0, tangents=4)); t4 = time.time() - t
        t = time.time(); Jd = np.asarray(s.jacobian_f64(zh, zl, 0, tangents=4, dd_base=True)); tdd = time.time() - t
        res = " (resampled from M = 48)" if src != M else ""
        print(f" M {M}{res}: f64x4 {percol(Jh, J4):.1e}, dd4 {percol(Jh, Jd):.1e}; {td:.1f} / {t4:.2f} / {tdd:.2f} s",
              flush=True)


def orders():
    print("== orders: M = 48, plan hmax 0.05; error of the horizon block with ks lowered, of the centre block with kc "
          "lowered (f64x1)")
    s, zh, zl, m, K = setup(48)
    segs = s.get_segs()
    Jh = np.asarray(s.jacobian(zh, zl)[0])
    hz, ce = slice(K * 4 * 48, (K + 1) * 4 * 48), slice(0, 4 * 48)
    for ks in (250, 120, 80, 60, 40, 30):
        s2, *_ = setup(48, ks=ks)
        s2.set_segs(segs)
        Jf = np.asarray(s2.jacobian_f64(zh, zl, 0, tangents=1))
        print(f"  ks {ks:3d}: horizon block {percol(Jh[hz], Jf[hz]):.1e}", flush=True)
    for kc in (100, 40, 30, 25, 20):
        s2, *_ = setup(48, kc=kc)
        s2.set_segs(segs)
        Jf = np.asarray(s2.jacobian_f64(zh, zl, 0, tangents=1))
        print(f"  kc {kc:3d}: centre block {percol(Jh[ce], Jf[ce]):.1e}", flush=True)


def main():
    secs = [a for a in sys.argv[1:]] or ["steps", "harmonics", "M", "orders"]
    for a in secs:
        {"steps": steps, "harmonics": harmonics, "M": by_M, "orders": orders}[a]()


if __name__ == "__main__":
    main()

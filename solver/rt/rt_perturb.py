"""An RT state moved off its root by a planted, seeded perturbation, for testing what sets rt_sigma.py's bounds there.

    python3 solver/rt/rt_perturb.py STATE.npz OUT.npz --delta 1e-19 [--seed 977] [--along OTHER.npz
        [--along2 OTHER2.npz --mode sum|orth]]
    python3 solver/rt/rt_perturb.py --selftest [--along | --orth]

OUT holds STATE's keys with z = zh + zl moved by delta max|z| r, where r has independent normal entries
(numpy default_rng(seed)) scaled to max|r| = 1, every entry the scalar mu included. The sum is formed in double-double
(Knuth's two-sum), so a delta far below the double ulp is kept in zl. Z, the f64 copy, is set to the new zh. Printed:
max|z| and the realised max|z' - z| / max|z| in double-double. --along OTHER.npz takes r along OTHER's z minus
STATE's z (formed in double-double, then scaled to max|r| = 1) in place of the random one, so the move runs along the
line through two root samples. --selftest moves a planted dd vector by 1e-25 and checks the realised move to 1e-6 of
itself; its control (--selftest --mutate) drops zl's part of the move and must fail (exit 1). --selftest --along does
the same for a move along a planted second sample, and also checks that the move is parallel to the planted
difference (cosine above 1 - 1e-6). With --along2, d1 = OTHER - z and d2 = OTHER2 - z: --mode sum takes
r = d1/max|d1| + d2/max|d2| (a direction inside the plane of the three samples), --mode orth the part of d2
orthogonal to d1 in the 2-norm. --selftest --orth checks that the orth move's cosine with d1 is below 1e-9 and that it
lies in the plane (its part outside d1 and d2 below 1e-9 of it).
"""
import argparse
import sys

import numpy as np


def two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def dd_add(ah, al, bh, bl):
    """(ah + al) + (bh + bl) in double-double, elementwise"""
    s, e = two_sum(ah, bh)
    e = e + al + bl
    return two_sum(s, e)


def diff(other, zh, zl):
    """other - z, formed in double-double and rounded to double"""
    dh, dl = dd_add(other[0], other[1], -zh, -zl)
    return dh + dl


def move(zh, zl, delta, seed, along=None, along2=None, mode="sum"):
    """z + delta max|z| r in double-double; the move itself is a double vector. along: (oh, ol), a second sample;
    r is then (o - z) / max|o - z| instead of a seeded random vector. along2 with mode: the sum of the two unit
    differences, or the part of the second orthogonal to the first"""
    if along is None:
        r = np.random.default_rng(seed).normal(size=zh.shape)
    else:
        r = diff(along, zh, zl)
        if along2 is not None:
            d1, d2 = r, diff(along2, zh, zl)
            if mode == "sum":
                r = d1 / np.abs(d1).max() + d2 / np.abs(d2).max()
            elif mode == "orth":
                r = d2 - (d2 @ d1) / (d1 @ d1) * d1
            else:
                raise ValueError(f"mode {mode}")
    r = r / np.abs(r).max()
    p = delta * float(np.abs(zh + zl).max()) * r
    return dd_add(zh, zl, p, np.zeros_like(p))


def realised(zh, zl, nh, nl):
    """max|z' - z| / max|z| in double-double"""
    dh, dl = dd_add(nh, nl, -zh, -zl)
    return float(np.abs(dh + dl).max()) / float(np.abs(zh + zl).max())


def selftest_orth():
    """the orth move: orthogonal to d1, inside the plane of d1 and d2"""
    rng = np.random.default_rng(6)
    zh = rng.normal(size=4000)
    zl = zh * 2.0 ** -60 * rng.uniform(-1, 1, size=4000)
    r1, r2 = rng.normal(size=4000), rng.normal(size=4000) + 0.5 * rng.normal(size=4000)
    s1 = dd_add(zh, zl, 3e-21 * r1, np.zeros(4000))
    s2 = dd_add(zh, zl, 3e-21 * (0.2 * r1 + r2), np.zeros(4000))
    nh, nl = move(zh, zl, 1e-25, 977, along=s1, along2=s2, mode="orth")
    dh, dl = dd_add(nh, nl, -zh, -zl)
    d = dh + dl
    d1, d2 = diff(s1, zh, zl), diff(s2, zh, zl)
    cos1 = abs(float(d @ d1)) / (np.linalg.norm(d) * np.linalg.norm(d1))
    B = np.linalg.qr(np.c_[d1, d2])[0]
    out = float(np.linalg.norm(d - B @ (B.T @ d)) / np.linalg.norm(d))
    ok = cos1 < 1e-9 and out < 1e-9 and abs(realised(zh, zl, nh, nl) - 1e-25) <= 1e-6 * 1e-25
    print(f"selftest --orth: cosine with d1 {cos1:.1e}, part outside the plane {out:.1e}: {'PASS' if ok else 'FAIL'}")
    return ok


def selftest(mutate, along=False):
    rng = np.random.default_rng(5)
    zh = rng.normal(size=4000)
    zl = zh * 2.0 ** -60 * rng.uniform(-1, 1, size=4000)
    other, cos = None, 1.0
    if along:
        r0 = rng.normal(size=4000)
        oh, ol = dd_add(zh, zl, 3e-21 * r0, np.zeros(4000))     # a planted second sample, 3e-21 away
        other = (oh, ol)
    nh, nl = move(zh, zl, 1e-25, 977, along=other)
    if mutate:
        nl = zl.copy()          # the control: the move's low part dropped, as a double-only sum would do
    got = realised(zh, zl, nh, nl)
    ok = abs(got - 1e-25) <= 1e-6 * 1e-25
    if along:
        dh, dl = dd_add(nh, nl, -zh, -zl)
        d = dh + dl
        cos = float(d @ r0 / (np.linalg.norm(d) * np.linalg.norm(r0))) if np.linalg.norm(d) > 0 else 0.0
        ok = ok and cos > 1 - 1e-6
    print(f"selftest{' --along' if along else ''}{' --mutate' if mutate else ''}: planted move 1e-25, realised "
          f"{got:.6e}" + (f", cosine with the planted difference {cos:.9f}" if along else "") + f": {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("state", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--delta", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=977)
    ap.add_argument("--along", nargs="?", const="", default=None, help="OTHER.npz: move along OTHER's z minus STATE's")
    ap.add_argument("--along2", default=None, help="OTHER2.npz, with --along and --mode")
    ap.add_argument("--mode", default="sum", choices=["sum", "orth"])
    ap.add_argument("--orth", action="store_true", help="--selftest of the orth move")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mutate", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        if a.orth:
            return 0 if selftest_orth() else 1
        return 0 if selftest(a.mutate, along=a.along is not None) else 1
    d = np.load(a.state)
    out = {k: d[k] for k in d.files}
    zh, zl = np.asarray(d["zh"], dtype=float), np.asarray(d["zl"], dtype=float)
    other = None
    if a.along:
        o = np.load(a.along)
        other = (np.asarray(o["zh"], dtype=float), np.asarray(o["zl"], dtype=float))
        if other[0].shape != zh.shape:
            raise SystemExit(f"--along: {a.along} has {other[0].size} entries, {a.state} {zh.size}")
    other2 = None
    if a.along2:
        o2 = np.load(a.along2)
        other2 = (np.asarray(o2["zh"], dtype=float), np.asarray(o2["zl"], dtype=float))
    nh, nl = move(zh, zl, a.delta, a.seed, along=other, along2=other2, mode=a.mode)
    out["zh"], out["zl"] = nh, nl
    if "Z" in out:
        out["Z"] = nh.copy()
    np.savez(a.out, **out)
    how = f"along {a.along}, max|other - z| / max|z| {realised(zh, zl, *other):.6e}" if other else f"seed {a.seed}"
    if other2 is not None:
        how += f"; with {a.along2} ({a.mode})"
    print(f"{a.state} -> {a.out}: max|z| {float(np.abs(zh + zl).max()):.6e}, delta {a.delta:g}, {how}, "
          f"realised max|z' - z| / max|z| {realised(zh, zl, nh, nl):.6e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

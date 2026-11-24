#!/usr/bin/env python3
"""#1078: a root of the matched mode determinant D_m(lam) (hp/mode_matched.py) inside a given bracket [A, B].

    python3 mode_matched_bracket.py root OUT.jsonl STATE.npz A B [--seed=SCAN.jsonl] [--xtol=1e-11] [--threads=6]
                                    [--hmax=0.05] [--match=valley|N]
    python3 mode_matched_bracket.py selftest [--mutate=shift]

`mode_matched.py root` searches from a guess with lowD_gamma.find_lam's fixed widths (1e-4 ... 1e-1). A sign scan has
already found a bracket whose ends differ in sign, so here brentq runs on that bracket directly. The ends' values can
be read from the scan's own rows (`--seed`): rows of kind "det" for D_m (which "m") with the same state, M, match node,
hmax and plan length, at exactly lam = A or B; mode_det is deterministic on one build, so they are the values a new
evaluation would give.

ln|D_m| grows by about 160 per unit lam at D 3.044375 (#1083), so sign * |D_m| varies by e^16 over a bracket of 0.1.
brentq runs on g(lam) = sign(lam) exp(ln|D_m(lam)| - ln|D_m(A)| - s (lam - A)), with s the bracket's secant slope of
ln|D_m|: a positive factor, so g has D_m's zeros and signs.

Verdict (exit 1 on FAIL): the signs of every evaluated point, sorted by lam, change exactly once; each evaluation's
double-double LU pivot ratio is above 1e-20; the root lies in [A, B].

`selftest` runs the same search on a planted function with D_m's shape, -exp(162 lam + 0.3 sin 7 lam) (lam - r), with r
known, and passes when the root is r to 1e-11 and one sign change is seen. Its control `--mutate=shift` plants the root
at r + 1e-6 and checks against r: it must FAIL (exit 1).

References: R. P. Brent, Algorithms for Minimization without Derivatives (Prentice-Hall, 1973), ch. 4 (scipy's brentq).
"""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PIVOT_BAR = 1e-20


def bracket_root(ev, a, b, xtol):
    """root of the sign/log function ev(lam) -> (sign, ln) in [a, b] by brentq on the detrended g; returns
    (root, points) with points the sorted (lam, sign) of every evaluation"""
    from scipy.optimize import brentq
    (sa, la), (sb, lb) = ev(a), ev(b)
    if sa == sb:
        raise ValueError(f"no sign change in the bracket: sign {sa:+.0f} at {a!r} and {sb:+.0f} at {b!r}")
    s = (lb - la) / (b - a)
    seen = {a: sa, b: sb}

    def g(lam):
        sg, ln = ev(lam)
        seen[lam] = sg
        return sg * math.exp(ln - la - s * (lam - a))

    r = brentq(g, a, b, xtol=xtol, rtol=4 * sys.float_info.epsilon, maxiter=100)
    return r, sorted(seen.items())


def sign_changes(points):
    sg = [s for _, s in points if s != 0]
    return sum(1 for i in range(len(sg) - 1) if sg[i] != sg[i + 1])


def seed_rows(path, info):
    """(lam -> (sign, ln, row)) from det rows of D_m in a jsonl, on the same state, M, match node, hmax and plan"""
    out = {}
    if not path:
        return out
    keys = ("state", "M", "m", "hmax", "steps_m")
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("kind") != "det" or r.get("which") != "m":
                continue
            if all(r.get(k) == info[k] for k in keys):
                out[r["lam"]] = (r["sign"], r["ln"], r)
    return out


def cmd_root(args, o):
    import mode_matched as MM
    out, path, a, b = args[0], args[1], float(args[2]), float(args[3])
    xtol = float(o.get("xtol", 1e-11))
    w = MM.writer(out)
    c = MM.Case(path, o, need=("out", "m"))
    info = c.info()
    seeds = seed_rows(o.get("seed"), info)
    rows = {}

    def ev(lam):
        lam = float(lam)
        if lam not in rows:
            if lam in seeds:
                r = dict(seeds[lam][2])
                r["seeded_from"] = os.path.basename(o["seed"])
                c.log(f"    m   lam {lam!r}: sign {r['sign']:+.0f} ln|det| {r['ln']:.9f} (seeded from the scan)")
            else:
                r = c.det("m", lam)
                r = dict(kind="bracket_det", **info, **r)
            rows[lam] = r
            w(dict(r, kind="bracket_det"))
        return rows[lam]["sign"], rows[lam]["ln"]

    t = time.time()
    err = None
    try:
        root, pts = bracket_root(ev, a, b, xtol)
    except ValueError as e:
        root, pts, err = None, sorted((l, r["sign"]) for l, r in rows.items()), str(e)
    nch = sign_changes(pts)
    ratio = min(r["pmin"] / r["pmax"] for r in rows.values())
    inside = root is not None and a <= root <= b
    ok = root is not None and nch == 1 and ratio > PIVOT_BAR and inside
    v = dict(kind="verdict", test="bracket_root", **info, bracket=[a, b], xtol=xtol, lam=root,
             gamma=(1 / root if root else None), n_signchanges=nch, n_det=len(rows),
             n_seeded=sum(1 for r in rows.values() if "seeded_from" in r), min_pivot_ratio=ratio,
             points=pts, error=err, sec=round(time.time() - t, 1), passed=ok)
    w(v)
    c.log(f"bracket root of D_m in [{a!r}, {b!r}]: lam {root!r} (gamma {v['gamma']!r}); sign changes {nch} over "
          f"{len(pts)} points, smallest pivot ratio {ratio:.1e} (> {PIVOT_BAR:.0e}), inside {inside}"
          + (f", error: {err}" if err else "") + f": {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def cmd_selftest(args, o):
    mut = o.get("mutate")
    r = 6.4123456789012345
    rp = r + (1e-6 if mut == "shift" else 0.0)
    n = [0]

    def ev(lam):
        n[0] += 1
        v = lam - rp
        return (-1.0 if v > 0 else (1.0 if v < 0 else 0.0)), 162 * lam + 0.3 * math.sin(7 * lam) + math.log(abs(v) or 1e-300)

    root, pts = bracket_root(ev, 6.25, 6.5, 1e-11)
    nch = sign_changes(pts)
    err = abs(root - r)
    ok = err <= 1e-11 and nch == 1
    print(f"selftest{' (control ' + mut + ')' if mut else ''}: planted root {r!r}"
          f"{' (planted at ' + repr(rp) + ')' if mut else ''}, found {root!r}, |error| {err:.1e} (<= 1e-11), "
          f"{n[0]} evaluations, {nch} sign change(s): {'PASS' if ok else 'FAIL'}" + (" (must FAIL)" if mut else ""),
          flush=True)
    return 0 if ok else 1


def main():
    import mode_matched as MM
    from raw_out import pop_allow, require_raw
    argv = sys.argv[1:]
    allow = pop_allow(argv)
    args, o = MM.opts_args(argv)
    if not args or args[0] not in ("root", "selftest"):
        print(__doc__)
        return 2
    if args[0] == "selftest":
        return cmd_selftest(args[1:], o)
    require_raw(args[1], allow=allow)
    return cmd_root(args[1:], o)


if __name__ == "__main__":
    sys.exit(main())

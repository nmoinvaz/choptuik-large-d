#!/usr/bin/env python3
"""Locate the zeros of g found by a count, from the count's own samples (no new determinants).

    python3 modecount_moment.py TAG STATE.npz [BOX ...] [--data=DIR] [--out=FILE] [--mutate=nok]

BOX: B1b, B1c (default) or U. TAG names the count: DIR/modecount_results.jsonl (its last record with that tag) and
DIR/modecount_<TAG>_evals.jsonl (DIR defaults to hp/data). STATE is the state the count ran on; the candidate poles of g
are recomputed from it as the count did, and ln g is checked against the samples the record stores.

Power sums (the method of Delves and Lyness, Math. Comp. 21, 543 (1967), doi:10.1090/S0025-5718-1967-0228165-4). On a
closed contour C without zeros of g, with W its winding and c a centre,

    s_k = sum_j (z_j - c)^k = (1 / 2 pi i) closed int (lam - c)^k g'/g d lam
        = W (lam_0 - c)^k - (k / 2 pi i) closed int (lam - c)^(k - 1) ln g d lam,      k = 1 .. W,

by integration by parts, with ln g continued along C from lam_0 (it ends at ln g(lam_0) + 2 pi i W). Newton's
identities, k e_k = sum_{i = 1..k} (-1)^(i - 1) e_(k - i) s_i, give the monic polynomial prod_j (x - (z_j - c)), and its
roots plus c are the zeros. For k = 1 this is the first moment, sum z_j = W lam_0 - (1 / 2 pi i) closed int ln g d lam.

Contours, each counter-clockwise:
  half   the upper half of B1b (V2, H2, V1 and the real axis from a1 to a2) or of B1c (R1, H1, V2 and the real axis
         from a2 to aR). c is the half box's centre. The real-axis part uses the count's real-axis samples, which are
         0.5 apart and not refined.
  sym    the whole box B1b or B1c. Its lower half is the mirror image of the upper (det M(conj lam) = conj det M(lam),
         and g likewise), so no real-axis sample is used. Its zeros are the z_j of the upper half and their
         conjugates, so the polynomial has degree 2 W_half and real coefficients; c is the box's centre (real).
  strip  U: H3, H2, H1 reversed, R2, T, L2 reversed; c is its centre.
Quadrature, on each straight piece of the contour: the trapezoid rule, and the rule that integrates the cubic through
the four samples around each interval. The cubic rule's roots are reported. The quadrature error estimate of a root is
its distance from the trapezoid rule's root (an estimate, not a bound; hp/modecount_moment_planted.py tests it against
planted zeros). --mutate=nok drops the factor k of the integration by parts (the planted test's control).
"""
import itertools
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DATA = os.path.join(HERE, "data")
BOXES = ("B1b", "B1c", "U")


def load_count(data, tag):
    """the count's last record with this tag and its evaluation cache, {(re, im): ln det M}"""
    with open(os.path.join(data, "modecount_results.jsonl")) as f:
        r = [x for x in (json.loads(l) for l in f) if x["tag"] == tag][-1]
    ev = {}
    with open(os.path.join(data, f"modecount_{tag}_evals.jsonl")) as f:
        for line in f:
            e = json.loads(line)
            ev[(e["re"], e["im"])] = complex(e["lnabs"], e["arg"])
    return r, ev


def factored(poles, r):
    """the candidate poles the count multiplied out (modecount_wind.count: |Im| < y2 + 3 omega)"""
    return np.array([p for p in poles if abs(p.imag) < r["y2"] + 3 * r["omega"]], complex)


def poles_from_state(statefile, r):
    """the candidate poles of det M as modecount_common.horizon_poles gives them, from the saved state"""
    from hpkernel.driver import load, dmat_odd
    zh, zl, m = load(statefile)
    M, K, D = m["M"], len(m["nodes"]), m["D"]
    n = D - 2
    ah = (np.asarray(zh) + np.asarray(zl))[3 * M + 4 * M * K:4 * M + 4 * M * K]
    c1 = (n - 1) * (1 - ah ** 2)
    dh, dl = dmat_odd(M)
    Dm = (np.asarray(dh) + np.asarray(dl)) * r["omega"]
    poles = []
    for kk in range(m.get("ks", 100) + 1):
        poles += [-complex(e) for e in np.linalg.eigvals(Dm + np.diag((c1 + 1) * (kk + 1) - n / 2))]
    return factored(poles, r)


class LnG:
    """ln g at the count's sample points: ln det M (cache) + ln prod (lam - p_j) - ln prod (lam - z_k) - p(lam)"""

    def __init__(self, r, ev, Q):
        self.keys = np.array([complex(*k) for k in ev])
        self.vals = list(ev.values())
        self.Q, self.K = np.asarray(Q, complex), np.array([complex(*z) for z in r["expected_zeros"]])
        self.pc = r["trend"]

    def __call__(self, lam):
        i = int(np.argmin(np.abs(self.keys - lam)))
        assert abs(self.keys[i] - lam) < 1e-9, (lam, self.keys[i])
        return (self.vals[i] + complex(np.sum(np.log(lam - self.Q)) - np.sum(np.log(lam - self.K)))
                - complex(np.polyval(self.pc, lam)))


def edge_points(r, name, rev=False):
    e = r["edges"][name]
    z0, z1 = complex(*e["z0"]), complex(*e["z1"])
    p = [z0 + s[0] * (z1 - z0) for s in e["samples"]]
    return p[::-1] if rev else p


def real_points(r, a, b):
    return sorted((complex(x, 0) for x, _, _ in r["real_axis"] if a - 1e-12 <= x <= b + 1e-12), key=lambda z: z.real)


def stored_check(r, lng, names):
    """largest difference between ln g recomputed here and the record's stored samples (ln|g|, wrapped arg g)"""
    worst = 0.0
    for name in names:
        e = r["edges"][name]
        z0, z1 = complex(*e["z0"]), complex(*e["z1"])
        for s in e["samples"]:
            v = lng(z0 + s[0] * (z1 - z0))
            worst = max(worst, abs(v.real - s[1]), abs(math.remainder(v.imag - s[2], 2 * math.pi)))
    return worst


def contour(r, box, mode, lng):
    """the closed contour as straight pieces [(points, ln g values)], counter-clockwise, and the centre c"""
    if box == "U":
        spec = [("H3", True), ("H2", True), ("H1", True), ("R2", False), ("T", False), ("L2", True)]
        rect = (r["aL"], r["aR"], r["y1"], r["y2"])
        real = None
    elif box == "B1b":
        spec = [("V2", False), ("H2", False), ("V1", True)]
        rect, real = (r["a1"], r["a2"], 0.0, r["y1"]), (r["a1"], r["a2"])
    elif box == "B1c":
        spec = [("R1", False), ("H1", False), ("V2", True)]
        rect, real = (r["a2"], r["aR"], 0.0, r["y1"]), (r["a2"], r["aR"])
    else:
        raise SystemExit(f"box {box}: one of {', '.join(BOXES)}")
    pieces = []
    for name, rev in spec:
        p = edge_points(r, name, rev)
        pieces.append((p, [lng(z) for z in p]))
    if box == "U":
        c = complex(0.5 * (rect[0] + rect[1]), 0.5 * (rect[2] + rect[3]))
    elif mode == "half":
        p = real_points(r, *real)
        pieces.append((p, [lng(z) for z in p]))
        c = complex(0.5 * (rect[0] + rect[1]), 0.5 * rect[3])
    elif mode == "sym":
        # the lower half: the mirror images of the upper pieces, in reverse order and each reversed
        pieces += [([z.conjugate() for z in p[::-1]], [v.conjugate() for v in vals[::-1]]) for p, vals in pieces[::-1]]
        c = complex(0.5 * (rect[0] + rect[1]), 0.0)
    else:
        raise SystemExit("mode: half or sym")
    assert abs(pieces[-1][0][-1] - pieces[0][0][0]) < 1e-9, (pieces[0][0][0], pieces[-1][0][-1])
    for (p, _), (q, _) in zip(pieces[:-1], pieces[1:]):
        assert abs(p[-1] - q[0]) < 1e-9, (p[-1], q[0])
    return pieces, c


def unwrap(pieces):
    """continue ln g along the contour: each phase within pi of the one before (the count's steps are <= pi/4)"""
    out, prev = [], None
    for p, vals in pieces:
        u = []
        for v in vals:
            if prev is not None:
                v = complex(v.real, prev.imag + math.remainder(v.imag - prev.imag, 2 * math.pi))
            u.append(v)
            prev = v
        out.append((np.array(p, complex), np.array(u, complex)))
    return out


def piece_integral(p, f, rule):
    """integral of f d lam along a straight piece sampled at p (any spacing): trapezoid or local cubic"""
    s = np.abs(p - p[0])
    L = s[-1]
    u = (p[-1] - p[0]) / L
    if rule == "trap" or len(p) == 2:
        return u * np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(s))
    m = len(p) - 1
    nn = min(4, m + 1)
    tot = 0j
    for i in range(m):
        j0 = min(max(i - 1, 0), m + 1 - nn)
        h = s[i + 1] - s[i]
        x = (s[j0:j0 + nn] - s[i]) / h
        # weights of the interpolant through these nodes, integrated over (0, 1): V^T w = (1, 1/2, 1/3, ...)
        V = np.vander(x, nn, increasing=True)
        w = np.linalg.solve(V.T, 1.0 / np.arange(1, nn + 1))
        tot += h * np.dot(w, f[j0:j0 + nn])
    return u * tot


def power_sums(cont, c, kmax, rule, mutate=None):
    """W and s_k = sum (z_j - c)^k, k = 1 .. kmax, from the unwrapped contour values"""
    lam0 = cont[0][0][0]
    dL = cont[-1][1][-1] - cont[0][1][0]          # 2 pi i W, to rounding
    W = dL.imag / (2 * math.pi)
    s = []
    for k in range(1, kmax + 1):
        I = sum(piece_integral(p, (p - c) ** (k - 1) * v, rule) for p, v in cont)
        fac = 1 if mutate == "nok" else k
        s.append((lam0 - c) ** k * dL / (2j * math.pi) - fac * I / (2j * math.pi))
    return W, s


def newton_roots(s, c):
    """roots of prod (x - (z_j - c)) from its power sums s_1..s_N (Newton's identities), shifted back by c"""
    N = len(s)
    if N == 0:
        return np.array([], complex)
    e = [1.0 + 0j]
    for k in range(1, N + 1):
        e.append(sum((-1) ** (i - 1) * e[k - i] * s[i - 1] for i in range(1, k + 1)) / k)
    coef = [(-1) ** k * e[k] for k in range(N + 1)]
    return np.roots(coef) + c


def match(a, b):
    """the pairing of the points b with the points a (same number) that minimises the largest distance (greedy, nearest
    first, above 8 points)"""
    if len(b) > 8:
        rest, out = list(b), []
        for z in a:
            j = int(np.argmin([abs(z - w) for w in rest]))
            out.append(rest.pop(j))
        return out
    best = None
    for perm in itertools.permutations(range(len(b))):
        d = max([abs(a[i] - b[j]) for i, j in enumerate(perm)] + [0.0])
        if best is None or d < best[0]:
            best = (d, perm)
    return [b[j] for j in best[1]]


def locate(r, ev, Q, box, mode, mutate=None):
    """the zeros of g inside one contour, by both quadrature rules; a dict for the record"""
    lng = LnG(r, ev, Q)
    pieces, c = contour(r, box, mode, lng)
    cont = unwrap(pieces)
    W, _ = power_sums(cont, c, 0, "trap")
    N = int(round(W))
    out = dict(box=box, mode=mode, centre=[c.real, c.imag], winding=W, samples=sum(len(p) for p, _ in cont))
    if abs(W - N) > 1e-6 or N < 0 or N > 12:
        out.update(ok=False, why=f"winding {W:.6f} is not a small non-negative integer")
        return out
    roots = {}
    for rule in ("cubic", "trap"):
        _, s = power_sums(cont, c, N, rule, mutate)
        roots[rule] = newton_roots(s, c)
        if rule == "cubic":
            out["power_sums"] = [[x.real, x.imag] for x in s]
    zc = list(roots["cubic"])
    zt = match(zc, list(roots["trap"]))
    est = [abs(a - b) for a, b in zip(zc, zt)]
    order = sorted(range(N), key=lambda i: (zc[i].real, zc[i].imag))
    out.update(ok=True, N=N, roots=[[zc[i].real, zc[i].imag] for i in order],
               roots_trap=[[zt[i].real, zt[i].imag] for i in order], est=[est[i] for i in order])
    if mode == "sym":
        # the polynomial's coefficients are real (the contour is its own mirror image), so the roots pair up
        out["upper"] = [i for i, j in enumerate(order) if zc[j].imag > 0]
    return out


def show(tag, x):
    head = f"{tag} {x['box']} ({x['mode']}): {x['samples']} samples, winding of g {x['winding']:+.6f}"
    if not x["ok"]:
        return head + f"; NOT LOCATED: {x['why']}"
    lines = [head + f", {x['N']} zeros of g inside (centre {x['centre'][0]:.4f} {x['centre'][1]:+.4f}i)"]
    for (re, im), e in zip(x["roots"], x["est"]):
        lines.append(f"    {re:9.5f} {im:+9.5f}i   quadrature error estimate {e:.1e}")
    return "\n".join(lines)


def main():
    from modecount_wind import parse
    args, opts = parse(sys.argv[1:])
    tag, statefile = args[0], args[1]
    boxes = args[2:] or ["B1c"]
    data = opts.get("data", DATA)
    r, ev = load_count(data, tag)
    Q = poles_from_state(statefile, r)
    if len(Q) != r["n_candidate_poles_factored"]:
        raise SystemExit(f"{len(Q)} candidate poles from {statefile}, the count factored "
                         f"{r['n_candidate_poles_factored']}: not the count's state")
    lng = LnG(r, ev, Q)
    names = sorted({n for b in boxes for n in {"B1b": ("V1", "V2", "H2"), "B1c": ("R1", "H1", "V2"),
                                                "U": ("H1", "H2", "H3", "R2", "T", "L2")}[b]})
    chk = stored_check(r, lng, names)
    print(f"{tag}: {len(ev)} cached determinants, {len(Q)} candidate poles; ln g recomputed against the record's "
          f"stored samples on {', '.join(names)}: largest difference {chk:.1e}")
    if chk > 1e-4:
        raise SystemExit("ln g does not reproduce the record's samples: wrong state, cache or record")
    res = []
    for b in boxes:
        for mode in (("strip",) if b == "U" else ("half", "sym")):
            x = locate(r, ev, Q, b, mode, opts.get("mutate"))
            print(show(tag, x), flush=True)
            res.append(dict(x, tag=tag, data=os.path.abspath(data), state=os.path.basename(statefile),
                            stored_check=chk, mutate=opts.get("mutate")))
    if "out" in opts:
        with open(opts["out"], "a") as fo:
            for x in res:
                fo.write(json.dumps(x) + "\n")
    sys.exit(0 if all(x["ok"] for x in res) else 1)


if __name__ == "__main__":
    main()

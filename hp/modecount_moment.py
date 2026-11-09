#!/usr/bin/env python3
"""Locate the zeros of g found by a count, from the count's own samples (no new determinants).

    python3 modecount_moment.py TAG STATE.npz [BOX ...] [--data=DIR] [--out=FILE] [--mutate=nok]

BOX: any box of the count, B0, B1a, B1b, B1c (default), B1, full, U0 or U. TAG names the count: DIR/modecount_results.jsonl (its last record with that tag) and
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
Quadrature, on each straight piece of the contour: on each interval the cubic through four neighbouring samples,
integrated exactly, with the central stencil (samples i - 1 .. i + 2) and the two shifted ones (i - 2 .. i + 1 and
i .. i + 3). Reported, for B1b and B1c: the zeros in the upper half from the sym contour's central stencil. Their
spread is the largest distance to the same zero by the other stencils of sym and the three stencils of half; for U,
by the other stencils. The spread is an estimator range over quadrature choices, not a bound. Its first version, the
distance to the trapezoid rule's root, failed the planted test as too pessimistic, by 10 to 100 times
(hp/modecount_moment_planted.py, which also tests this one). --mutate=nok drops the factor k of the integration by
parts (the planted test's control).
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
# the boxes of modecount_wind.count: edges (name, reversed) counter-clockwise (for a symmetric box its upper half,
# from the real axis at the right back to it at the left), the left and right Re, and the kind
BOXES = {
    "B0": ([("L1", False), ("H0", False), ("V0", True)], "aLL", "aL", "sym"),
    "B1a": ([("V1", False), ("H3", False), ("L1", True)], "aL", "a1", "sym"),
    "B1b": ([("V2", False), ("H2", False), ("V1", True)], "a1", "a2", "sym"),
    "B1c": ([("R1", False), ("H1", False), ("V2", True)], "a2", "aR", "sym"),
    "B1": ([("R1", False), ("H1", False), ("H2", False), ("H3", False), ("L1", True)], "aL", "aR", "sym"),
    "full": ([("R1", False), ("R2", False), ("T", False), ("L2", True), ("L1", True)], "aL", "aR", "sym"),
    "U0": ([("H0", True), ("L2", False), ("T0", False), ("V0b", True)], "aLL", "aL", "strip"),
    "U": ([("H3", True), ("H2", True), ("H1", True), ("R2", False), ("T", False), ("L2", True)], "aL", "aR", "strip"),
}


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


def box_spec(r, box):
    """the box's edges (name, reversed) counter-clockwise, its rectangle (upper half for the symmetric boxes), its
    kind, and the real-axis interval of the half contour (None: the count has no real-axis samples there)"""
    if box not in BOXES:
        raise SystemExit(f"box {box}: one of {', '.join(BOXES)}")
    spec, left, right, kind = BOXES[box]
    lo, hi = (r["y1"], r["y2"]) if kind == "strip" else (0.0, r["y2"] if box == "full" else r["y1"])
    a, b = r[left], r[right]
    real = None
    if kind == "sym":
        xs = [x for x, _, _ in r["real_axis"]]
        if any(abs(x - a) < 1e-12 for x in xs) and any(abs(x - b) < 1e-12 for x in xs):
            real = (a, b)
    return spec, (a, b, lo, hi), kind, real


def contour(r, box, mode, lng):
    """the closed contour as straight pieces [(points, ln g values)], counter-clockwise, and the centre c"""
    spec, rect, kind, real = box_spec(r, box)
    if (mode == "strip") != (kind == "strip") or (mode == "half" and real is None):
        raise SystemExit(f"box {box}: no {mode} contour")
    pieces = []
    for name, rev in spec:
        p = edge_points(r, name, rev)
        pieces.append((p, [lng(z) for z in p]))
    if kind == "strip":
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


def piece_integral(p, f, bias=0):
    """integral of f d lam along a straight piece sampled at p (any spacing): on each interval (s_i, s_i+1) the cubic
    through the samples i - 1 + bias .. i + 2 + bias (moved inside the piece at its ends), integrated exactly"""
    s = np.abs(p - p[0])
    u = (p[-1] - p[0]) / s[-1]
    m = len(p) - 1
    nn = min(4, m + 1)
    tot = 0j
    for i in range(m):
        j0 = min(max(i - 1 + bias, 0), m + 1 - nn)
        h = s[i + 1] - s[i]
        x = (s[j0:j0 + nn] - s[i]) / h
        # weights of the interpolant through these nodes, integrated over (0, 1): V^T w = (1, 1/2, 1/3, ...)
        V = np.vander(x, nn, increasing=True)
        w = np.linalg.solve(V.T, 1.0 / np.arange(1, nn + 1))
        tot += h * np.dot(w, f[j0:j0 + nn])
    return u * tot


def power_sums(cont, c, kmax, bias=0, mutate=None):
    """W and s_k = sum (z_j - c)^k, k = 1 .. kmax, from the unwrapped contour values"""
    lam0 = cont[0][0][0]
    dL = cont[-1][1][-1] - cont[0][1][0]          # 2 pi i W, to rounding
    W = dL.imag / (2 * math.pi)
    s = []
    for k in range(1, kmax + 1):
        I = sum(piece_integral(p, (p - c) ** (k - 1) * v, bias) for p, v in cont)
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


def solve(r, lng, box, mode, mutate=None):
    """the zeros of g inside one contour by the three cubic stencils (bias 0, -1, +1): W, N, {bias: roots}, info"""
    pieces, c = contour(r, box, mode, lng)
    cont = unwrap(pieces)
    W, _ = power_sums(cont, c, 0)
    N = int(round(W))
    info = dict(centre=[c.real, c.imag], winding=W, samples=sum(len(p) for p, _ in cont))
    if abs(W - N) > 1e-6 or N < 0 or N > 12:
        return W, None, None, info
    roots = {}
    for bias in (0, -1, 1):
        _, s = power_sums(cont, c, N, bias, mutate)
        roots[bias] = list(newton_roots(s, c))
        if bias == 0:
            info["power_sums"] = [[x.real, x.imag] for x in s]
    return W, N, roots, info


def locate(r, ev, Q, box, mutate=None):
    """the zeros of g in one box (symmetric boxes: those in the upper half; U, U0: those in the strip), a dict for the
    record. Reported: the central-stencil roots of the mirrored whole box (sym; for U and U0 the strip). spread: the
    largest distance from a reported root to the same zero by the other two stencils and, where the count has real-axis
    samples at both ends of the box (not B0), by the half contour's three."""
    lng = LnG(r, ev, Q)
    out = dict(box=box)
    _, _, kind, real = box_spec(r, box)
    strip = kind == "strip"
    modes = ("strip",) if strip else ("sym", "half") if real else ("sym",)
    sol = {m: solve(r, lng, box, m, mutate) for m in modes}
    for m in modes:
        W, N, roots, info = sol[m]
        out[m] = dict(info, N=N, roots={str(b): [[z.real, z.imag] for z in zs] for b, zs in (roots or {}).items()})
    W, N, roots, _ = sol[modes[0]]
    if N is None:
        out.update(ok=False, why=f"winding {W:.6f} of the {modes[0]} contour is not a small non-negative integer")
        return out
    rep = roots[0] if strip else [z for z in roots[0] if z.imag > 0]
    others = [roots[-1], roots[1]]
    if not strip:
        # the sym polynomial has real coefficients (the contour is its own mirror image): its roots pair up
        if N % 2 or len(rep) != N // 2:
            out.update(ok=False, why=f"the sym contour's {N} roots do not pair into {N // 2} conjugate pairs")
            return out
        others = [[z for z in o if z.imag > 0] for o in others]
        if "half" in sol:
            Wh, Nh, rh, _ = sol["half"]
            if Nh != N // 2:
                out.update(ok=False, why=f"winding {Wh:.6f} of the half contour is not half of {W:.6f}")
                return out
            others += [rh[0], rh[-1], rh[1]]
        if any(len(o) != len(rep) for o in others):
            out.update(ok=False, why="a stencil's sym roots do not pair up")
            return out
    spread = [0.0] * len(rep)
    for o in others:
        for i, z in enumerate(match(rep, o)):
            spread[i] = max(spread[i], abs(z - rep[i]))
    order = sorted(range(len(rep)), key=lambda i: (rep[i].real, rep[i].imag))
    out.update(ok=True, N=len(rep), roots=[[rep[i].real, rep[i].imag] for i in order],
               spread=[spread[i] for i in order])
    if "half" in sol:
        out["roots_half"] = [[z.real, z.imag] for z in match([rep[i] for i in order], rh[0])]
    return out


def show(tag, x):
    m = "strip" if "strip" in x else "sym"
    head = f"{tag} {x['box']}: winding of g {x[m]['winding']:+.6f} ({m}, {x[m]['samples']} samples)"
    if "half" in x:
        head += f", {x['half']['winding']:+.6f} (half, {x['half']['samples']} samples)"
    if not x["ok"]:
        return head + f"; NOT LOCATED: {x['why']}"
    lines = [head + f"; {x['N']} zeros of g" + ("" if m == "strip" else " in the upper half")]
    for i, ((re, im), e) in enumerate(zip(x["roots"], x["spread"])):
        h = "" if "roots_half" not in x else f"   (half contour {x['roots_half'][i][0]:.5f} {x['roots_half'][i][1]:+.5f}i)"
        lines.append(f"    {re:9.5f} {im:+9.5f}i   spread {e:.1e}{h}")
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
    names = sorted({n for b in boxes for n, _ in box_spec(r, b)[0]})
    chk = stored_check(r, lng, names)
    print(f"{tag}: {len(ev)} cached determinants, {len(Q)} candidate poles; ln g recomputed against the record's "
          f"stored samples on {', '.join(names)}: largest difference {chk:.1e}")
    if chk > 1e-4:
        raise SystemExit("ln g does not reproduce the record's samples: wrong state, cache or record")
    res = []
    for b in boxes:
        x = locate(r, ev, Q, b, opts.get("mutate"))
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

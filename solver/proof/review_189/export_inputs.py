#!/usr/bin/env python3
"""Inputs of capd_m5 from the stored certificate files (#189 review). Exact rational arithmetic, no python-flint.

    export_inputs.py start6b1 <ray.json> <out> [--perturb=i:delta]
        the C^1 start set of the ray run at s0 = 405 e^{-3 i pi/4} (ray.json "sets"["0"]): c + Q q + V dx with
        V = cV + (QV qV) (complex columns, one per parameter) and dx = +-1.01e-18, written as the doubleton
        x + C r0: x = the box of c + Q q, C = V (real coordinates, 39 x 6), r0 = dx
    export_inputs.py path6b1 <ray.json> <row.json> <out> [--legs=K]
        the ray legs |u| = rho_{j-1} -> rho_j (tags ray1 .. ray17), then the arc of the row with a stop at each disc
        time t_k (tags arc<j>_<k>); --legs=K keeps the first K ray legs only (no arc)
    export_inputs.py start6a <box.json> <out> --wf=<decimal> [--h=<decimal>] [--perturb=i:delta]
        lemma 6a's backward start: the hull of y (the box file's yX_xi1) and sigma(y) (mu -> -mu with conjugation), and
        w = W0 (wf +- h), W0 = 2 pi / 1.64389375
    export_inputs.py path6a <out> [--xi1=0.3] [--xia=0.067]
Real layout: Re y_k, Im y_k at 2k, 2k + 1 (k = 0..18, the stored order), w at 38. --perturb adds delta to coordinate i
of the box (the start mutation).
"""
import json
import sys
from decimal import Decimal
from fractions import Fraction

import stored as S

DIM = 39
# pi to 60 digits (lower bound) and an upper bound 1e-59 above it
PI_LO = Fraction("3.14159265358979323846264338327950288419716939937510582097494")
PI_HI = PI_LO + Fraction(1, 10 ** 59)
W0 = (2 * PI_LO * 160000 / 263023, 2 * PI_HI * 160000 / 263023)    # 1.64389375 = 263023 / 160000


def opts_of(argv):
    return {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in argv if a.startswith("--")}


def perturb(x, spec):
    if not spec:
        return x
    i, d = spec.split(":")
    i, d = int(i), Fraction(d)
    x = list(x)
    x[i] = (x[i][0] + d, x[i][1] + d)
    return x


def start6b1(ray, out, opts):
    with open(ray) as fh:
        d = json.load(fh)
    st = d["sets"]["0"]
    n = len(st["c"])                      # 21 complex: 19 states, w, A
    m = len(st["dx"])                     # 6 parameters
    Q = [[S.arb_str(v) for v in row] for row in st["Q"]]
    q = [S.arb_str(v) for v in st["q"]]
    QV = [[S.arb_str(v) for v in row] for row in st["QV"]]
    qV = [[S.arb_str(v) for v in row] for row in st["qV"]]
    c = [(S.arb_str(re), S.arb_str(im)) for re, im in st["c"]]
    cV = [[(S.arb_str(re), S.arb_str(im)) for re, im in row] for row in st["cV"]]
    dx = [(S.arb_str(re), S.arb_str(im)) for re, im in st["dx"]]
    for re, im in dx:
        assert im == (0, 0), "dx must be real"

    def matvec(A, v):
        res = []
        for row in A:
            acc = (Fraction(0), Fraction(0))
            for a, b in zip(row, v):
                if a != (0, 0) and b != (0, 0):
                    acc = S.iadd(acc, S.imul(a, b))
            res.append(acc)
        return res
    Qq = matvec(Q, q)                     # real 2n: [Re 0..n-1, Im 0..n-1]
    # V = cV + unstack(QV qV): column j
    Vc = []
    for j in range(m):
        col = matvec(QV, [qV[i][j] for i in range(2 * n)])
        Vc.append([(S.iadd(cV[i][j][0], col[i]), S.iadd(cV[i][j][1], col[n + i])) for i in range(n)])
    x = []
    for k in range(19):
        x.append(S.iadd(c[k][0], Qq[k]))
        x.append(S.iadd(c[k][1], Qq[n + k]))
    w_im = S.iadd(c[19][1], Qq[n + 19])
    assert w_im[0] <= 0 <= w_im[1] and S.rad(w_im) < Fraction(1, 10 ** 60), "w must be real"
    x.append(S.iadd(c[19][0], Qq[19]))
    x = perturb(x, opts.get("perturb"))
    with open(out, "w") as fh:
        fh.write(f"# start6b1 from {ray} sets[0]; perturb {opts.get('perturb')}\n")
        for i, v in enumerate(x):
            fh.write(f"x {i} {S.iv_line(v)}\n")
        for j in range(m):
            for k in range(19):
                re, im = Vc[j][k]
                if re != (0, 0):
                    fh.write(f"C {2 * k} {j} {S.iv_line(re)}\n")
                if im != (0, 0):
                    fh.write(f"C {2 * k + 1} {j} {S.iv_line(im)}\n")
            re_w, im_w = Vc[j][19]
            assert im_w[0] <= 0 <= im_w[1] and S.rad(im_w) < Fraction(1, 10 ** 60)
            if re_w != (0, 0):
                fh.write(f"C 38 {j} {S.iv_line(re_w)}\n")
        for j in range(m):
            fh.write(f"r0 {j} {S.iv_line(dx[j][0])}\n")
    rmax = max(float(S.rad(v)) for v in x[:38])
    print(f"start6b1: centre box max radius {rmax:.3e}, {m} parameter columns, |dx| <= {float(dx[0][0][1]):.3e}")


def path6b1(ray, row, out, opts):
    with open(ray) as fh:
        d = json.load(fh)
    with open(row) as fh:
        r = json.load(fh)
    rho = d["rho"]
    K = int(opts.get("legs", str(len(rho) - 1)))
    j = int(r["j"])
    assert abs(float(r["rho"]) - rho[j]) < 1e-12
    lines = []
    for i in range(1, K + 1):
        lines.append(f"ray {rho[i - 1]!r} {rho[i]!r} ray{i}")
    if K == len(rho) - 1:
        ts = r["t"]
        # the exact binary value of each stop time (on the 2^-30 grid), as a finite decimal
        lines.append(f"arc {rho[j]!r} " + " ".join(f"{Decimal(float(t))} arc{j}_{k}" for k, t in enumerate(ts)))
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"path6b1: {K} ray legs" + (f", arc of row {j} at |u| = {rho[j]} with {len(r['t'])} stops" if K == len(rho) - 1
                                      else ""))


def sigma_index():
    odd = list(range(-5, 6, 2))
    even = list(range(-6, 7, 2))
    idx = [odd.index(-m) for m in odd] + [6 + odd.index(-m) for m in odd] + [12 + even.index(-n) for n in even]
    return idx


def start6a(box, out, opts):
    with open(box) as fh:
        d = json.load(fh)
    y = [(S.any_real(re), S.any_real(im)) for re, im in d["yX_xi1"]]
    assert len(y) == 19
    sig = sigma_index()
    x = []
    for k in range(19):
        re_s, im_s = y[sig[k]]
        ys = (re_s, (-im_s[1], -im_s[0]))          # conj(y_sigma(k))
        x.append(S.ihull(y[k][0], ys[0]))
        x.append(S.ihull(y[k][1], ys[1]))
    wf = Fraction(opts["wf"])
    h = Fraction(opts.get("h", d.get("h", "0")))
    x.append(S.imul(W0, (wf - h, wf + h)))
    x = perturb(x, opts.get("perturb"))
    with open(out, "w") as fh:
        fh.write(f"# start6a from {box}: hull of y and sigma(y); w = W0 ({opts['wf']} +- {h}); perturb "
                 f"{opts.get('perturb')}\n")
        for i, v in enumerate(x):
            fh.write(f"x {i} {S.iv_line(v)}\n")
    print(f"start6a: hull max radius {max(float(S.rad(v)) for v in x[:38]):.3e}, w radius {float(S.rad(x[38])):.3e}")


def path6a(out, opts):
    with open(out, "w") as fh:
        fh.write(f"real {opts.get('xi1', '0.3')} {opts.get('xia', '0.067')} xia\n")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = opts_of(sys.argv[1:])
    cmd = args[0]
    if cmd == "start6b1":
        start6b1(args[1], args[2], opts)
    elif cmd == "path6b1":
        path6b1(args[1], args[2], args[3], opts)
    elif cmd == "start6a":
        start6a(args[1], args[2], opts)
    elif cmd == "path6a":
        path6a(args[1], opts)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

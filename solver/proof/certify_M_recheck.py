#!/usr/bin/env python3
"""Independent re-check of the Krawczyk assembly of certify_M.py (proof/A_M.md section 6), from a saved box run.

Shares with certify_M.py only the saved data (the final Lohner sets of the box run, the tilt C, the widths) and the
statement of the test; everything else is written anew here, in mpmath interval arithmetic (mpmath.iv), not
python-flint:
  * the jump functionals J_nu = D_nu - conj(D_-nu) on the real coordinates (Re y, Im y), built from the state layout
    (D on the odd harmonics, then S, then C, then w);
  * the enclosures l (c + Q q) and l (cV_k + QV qV_k), with the preconditioner applied to l before the interval
    products, as certify_M.py does (the leakage must cancel in the point matrix);
  * the preconditioner Y: the inverse of the midpoint matrix of the column enclosures, computed in mpmath at 120
    digits (any Y is allowed);
  * K(X) = -Y g(0) + (I - Y Dgt(X)) X and the inclusion test, and the zero enclosure p^ + T K(X).

    python3 certify_M_recheck.py data/A_M_box_M3_1e-9.json [--mutate]
Exit status 0 if K(X) is in the interior of X, 1 otherwise. --mutate flips the sign of the jump
(D_nu + conj D_-nu): the test must then fail.
"""
import json
import sys

import mpmath
from mpmath import iv, mp

iv.prec = 400
mp.dps = 120


def ball(s):
    """an arb string 'm +/- r' (or 'm') -> iv interval containing it"""
    s = s.strip()
    if s.startswith("["):
        s = s[1:-1]
    if "+/-" in s:
        m, r = s.split("+/-")
        m = iv.mpf(m.strip()) if m.strip() else iv.mpf(0)
        r = iv.mpf(r.strip())
        return iv.mpf([(m - r).a, (m + r).b])
    return iv.mpf(s)


def exact(x):
    """a float hex string -> exact interval"""
    return iv.mpf(float.fromhex(x))


def layout(M):
    odd = [mu for mu in range(-M, M + 1) if mu % 2]
    even = [nu for nu in range(-(M + 1), M + 2) if nu % 2 == 0]
    nd = 2 * len(odd) + len(even)
    return odd, even, nd, nd + 1


def main(path, mutate=False):
    bx = json.load(open(path))
    M = bx["M"]
    m = M + 1
    odd, even, nd, n = layout(M)
    S = bx["sets"]
    # real coordinates z = (Re y_0..y_{n-1}, Im y_0..y_{n-1})
    c = [ball(v[0]) for v in S["c"]] + [ball(v[1]) for v in S["c"]]
    Q = [[exact(x) for x in row] for row in S["Q"]]
    q = [ball(x) for x in S["q"]]
    cV = [[ball(S["cV"][i][j][0]) for j in range(m)] for i in range(n)] + \
         [[ball(S["cV"][i][j][1]) for j in range(m)] for i in range(n)]
    QV = [[exact(x) for x in row] for row in S["QV"]]
    qV = [[ball(x) for x in row] for row in S["qV"]]
    # the jump functionals
    ell = []
    for nu in range(1, M + 1, 2):
        ip, im = odd.index(nu), odd.index(-nu)            # D occupies the first len(odd) slots
        re = [0] * (2 * n)
        imr = [0] * (2 * n)
        re[ip], re[im] = 1, (1 if mutate else -1)          # mutate: D_nu + conj D_-nu (must fail)
        imr[n + ip], imr[n + im] = 1, (-1 if mutate else 1)
        ell += [re, imr]
    # midpoints of the column enclosures -> Y (mpmath, 120 digits)
    def mid(x):
        return mp.mpf(x.mid.a)
    cols = []
    for k in range(m):
        col = []
        for r in ell:
            acc = mp.mpf(0)
            for i, e in enumerate(r):
                if e:
                    acc += e * mp.mpf(mid(cV[i][k]))
            col.append(acc)
        cols.append(col)
    D = mp.matrix([[cols[k][i] for k in range(m)] for i in range(m)])
    Ymp = D ** -1
    Y = [[iv.mpf(Ymp[i, j]) for j in range(m)] for i in range(m)]
    Yl = [[sum((Y[i][r] * ell[r][j] for r in range(m) if ell[r][j]), iv.mpf(0)) for j in range(2 * n)]
          for i in range(m)]
    # Y l Q and Y l QV (point-like matrices, computed in intervals)
    def mul_rows(A, B):
        rows, inner, cols_ = len(A), len(B), len(B[0])
        out = []
        for i in range(rows):
            nz = [t for t in range(inner) if A[i][t] != 0]
            out.append([sum((A[i][t] * B[t][j] for t in nz), iv.mpf(0)) for j in range(cols_)])
        return out
    YlQ = mul_rows(Yl, Q)
    YlQV = mul_rows(Yl, QV)
    Yg = [sum((Yl[i][j] * c[j] for j in range(2 * n)), iv.mpf(0)) + sum((YlQ[i][j] * q[j] for j in range(2 * n)), iv.mpf(0))
          for i in range(m)]
    YJ = [[sum((Yl[i][j] * cV[j][k] for j in range(2 * n)), iv.mpf(0)) +
           sum((YlQV[i][j] * qV[j][k] for j in range(2 * n)), iv.mpf(0)) for k in range(m)] for i in range(m)]
    h = iv.mpf(bx["h"])
    ws = bx["w"]
    widths = [h, h] + [iv.mpf(ws[(i // 2) % len(ws)]) for i in range(M - 1)]
    K = []
    for i in range(m):
        t = -Yg[i]
        for j in range(m):
            t = t + ((1 if i == j else 0) - YJ[i][j]) * iv.mpf([-widths[j].b, widths[j].b])
        K.append(t)
    ok = all(K[i].a > -widths[i].a and K[i].b < widths[i].a for i in range(m))
    rel = [float((mid(widths[i]) - max(abs(mp.mpf(K[i].a.a)), abs(mp.mpf(K[i].b.a)))) / mid(widths[i])) for i in range(m)]
    print(f"recheck (mpmath intervals){' [mutated: jump sign]' if mutate else ''} {path}: K(X) in int X: {ok}; relative margins " +
          ", ".join(f"{x:+.3f}" for x in rel))
    # the zero in p-coordinates
    C = [[iv.mpf(ball(x)) for x in r] for r in bx["C"]]
    # certify_M's centre is the 256-bit binary midpoint of these decimals: widen by 2^-250 relative
    p = [iv.mpf(x) * iv.mpf([1 - mpmath.mpf(2) ** -250, 1 + mpmath.mpf(2) ** -250]) for x in bx["p"]]
    names = ["A", "w/w_inf"] + [f"{b}{nu}" for nu in range(3, M + 1, 2) for b in ("B", "C")]
    for i in range(m):
        z = p[i] + K[i] if i < 2 else p[i] + C[i - 2][0] * K[0] + C[i - 2][1] * K[1] + K[i]
        print(f"   {names[i]:8s} in [{mpmath.nstr(mp.mpf(z.a.a), 17)}, {mpmath.nstr(mp.mpf(z.b.a), 17)}]")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1], "--mutate" in sys.argv) else 1)

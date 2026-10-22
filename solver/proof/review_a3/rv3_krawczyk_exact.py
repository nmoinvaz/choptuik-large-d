#!/usr/bin/env python3
"""Review A3, check K1: the Krawczyk inclusion of the M-harmonic point certificate, recomputed from the saved box run
in EXACT RATIONAL interval arithmetic (Python fractions; no python-flint, no mpmath, no code of certify_M*.py).

What is shared with the certificate: only the saved JSON (the final Lohner sets c + Q q of the centre trajectory and
cV + QV qV of the tilted derivative columns, the tilt C, the centre strings p, the widths h, w). Everything else is
written here from the statement of the test in proof/A_M.md section 2:

  * the state layout: D on the odd harmonics -M..M, then S, then C on the even harmonics -(M+1)..M+1, then w;
    real coordinates (Re y_0..y_{n-1}, Im y_0..y_{n-1});
  * the jump functionals l: Re J_nu = Re D_nu - Re D_-nu, Im J_nu = Im D_nu + Im D_-nu (J_nu = D_nu - conj D_-nu);
  * a preconditioner Y of my own: the exact rational inverse of a rounded midpoint of the column enclosures;
  * g(0) in (Y l) c + ((Y l) Q) q and Dgt(X) in (Y l) cV + ((Y l) QV) qV, every product exact;
  * K(X) = -Y g(0) + (I - Y Dgt(X)) X on the NOMINAL box X = [-h, h]^2 x [-w, w]^(M-1) (a sub-box of the run's box,
    whose radii are mag-rounded upward from h and w, so Dgt(X_run) also encloses the Jacobian over it);
  * the zero in p = p^ + T K(X), with p^ within 2^-255 relative of the decimal strings (exact_p rounds them at 256
    bits) and T's entries the saved balls.

Every ball string "[m +/- r]" is read as the rational interval [m - r, m + r] (exact; arb's printed radius already
covers the decimal conversion of the midpoint). Floats (Q matrices, saved as float.hex) are exact rationals.

    python3 rv3_krawczyk_exact.py ../data/A_M_box_M3_1e-10.json [--mutate=sign|noconj|shift|widen]
Exit 0 iff K(X) lies in the interior of X (and, without --mutate, the zero enclosure lies inside the one quoted in
proof/A_M.md section 6.2). Mutations must give exit 1: 'sign' (D_nu + conj D_-nu), 'noconj' (D_nu - D_-nu),
'shift' (g(0) replaced by g(0) + Dgt_mid (2h e_A), a zero planted outside), 'widen' (g(0) widened by 1e3 h).
"""
import json
import sys
from fractions import Fraction as Fr

# ------------------------------------------------------------------ rational intervals


class Iv:
    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi=None):
        self.lo = Fr(lo)
        self.hi = Fr(lo) if hi is None else Fr(hi)
        assert self.lo <= self.hi

    def __add__(self, o):
        o = o if isinstance(o, Iv) else Iv(o)
        return Iv(self.lo + o.lo, self.hi + o.hi)

    __radd__ = __add__

    def __neg__(self):
        return Iv(-self.hi, -self.lo)

    def __sub__(self, o):
        return self + (-(o if isinstance(o, Iv) else Iv(o)))

    def __mul__(self, o):
        if not isinstance(o, Iv):
            o = Fr(o)
            return Iv(self.lo * o, self.hi * o) if o >= 0 else Iv(self.hi * o, self.lo * o)
        ps = [self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi]
        return Iv(min(ps), max(ps))

    __rmul__ = __mul__

    def mag(self):
        return max(abs(self.lo), abs(self.hi))

    def mid(self):
        return (self.lo + self.hi) / 2

    def __repr__(self):
        return f"[{float(self.lo):.17g}, {float(self.hi):.17g}]"


def parse_ball(s):
    """arb string -> exact rational interval"""
    s = s.strip()
    if s == "0":
        return Iv(0)
    if s.startswith("["):
        s = s[1:-1].strip()
    if "+/-" in s:
        m, r = s.split("+/-")
        m = Fr(m.strip()) if m.strip() else Fr(0)
        r = Fr(r.strip())
        assert r >= 0
        return Iv(m - r, m + r)
    return Iv(Fr(s))


def parse_hex(x):
    return Fr(float.fromhex(x))


def round_fr(x, bits=200):
    """a nearby rational with denominator 2^bits (a choice: only used to build the preconditioner)"""
    return Fr(round(x * 2 ** bits), 2 ** bits)


def inv_exact(A):
    """exact inverse of a square Fraction matrix (Gauss-Jordan)"""
    n = len(A)
    M = [list(A[i]) + [Fr(1 if i == j else 0) for j in range(n)] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if M[piv][col] == 0:
            raise ZeroDivisionError("singular")
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        M[col] = [x / pv for x in M[col]]
        for r in range(n):
            if r != col and M[r][col] != 0:
                f = M[r][col]
                M[r] = [a - f * b for a, b in zip(M[r], M[col])]
    return [row[n:] for row in M]


# ------------------------------------------------------------------ the test

QUOTED_M3 = {  # proof/A_M.md section 6.2, the certified zero (endpoints rounded outward)
    0: ("2.72647588563032", "2.72647588563144"),
    1: ("1.01560309485743", "1.01560309485804"),
    2: ("0.144419478469924", "0.144419478470230"),
    3: ("0.0110874258881079", "0.0110874258881255"),
}


def main(path, mutate=None):
    bx = json.load(open(path))
    M = bx["M"]
    m = M + 1
    odd = [mu for mu in range(-M, M + 1) if mu % 2]
    even = [nu for nu in range(-(M + 1), M + 2) if nu % 2 == 0]
    nd = 2 * len(odd) + len(even)
    n = nd + 1
    S = bx["sets"]
    assert len(S["c"]) == n and len(S["Q"]) == 2 * n and len(S["q"]) == 2 * n
    # state: real coordinates
    c = [parse_ball(z[0]) for z in S["c"]] + [parse_ball(z[1]) for z in S["c"]]
    Q = [[parse_hex(x) for x in row] for row in S["Q"]]
    q = [parse_ball(x) for x in S["q"]]
    cV = [[parse_ball(S["cV"][i][k][0]) for k in range(m)] for i in range(n)] + \
         [[parse_ball(S["cV"][i][k][1]) for k in range(m)] for i in range(n)]
    QV = [[parse_hex(x) for x in row] for row in S["QV"]]
    qV = [[parse_ball(x) for x in row] for row in S["qV"]]
    # jump functionals (rows: Re J_1, Im J_1, Re J_3, Im J_3, ...)
    idxD = {mu: i for i, mu in enumerate(odd)}          # D_mu is component i
    ell = []
    for nu in range(1, M + 1, 2):
        re = [Fr(0)] * (2 * n)
        im = [Fr(0)] * (2 * n)
        a, b, cc, d = 1, -1, 1, 1
        if mutate == "sign":
            a, b, cc, d = 1, 1, 1, -1
        if mutate == "noconj":
            a, b, cc, d = 1, -1, 1, -1
        re[idxD[nu]], re[idxD[-nu]] = Fr(a), Fr(b)
        im[n + idxD[nu]], im[n + idxD[-nu]] = Fr(cc), Fr(d)
        ell += [re, im]

    def matmul(A, B):
        return [[sum((A[i][k] * B[k][j] for k in range(len(B)) if A[i][k] != 0), Fr(0)) for j in range(len(B[0]))]
                for i in range(len(A))]

    def apply_iv(R, cvec, Qm, qvec):
        """R (c + Q q) for exact rational R, Q; interval c, q: (R c) + (R Q) q"""
        RQ = matmul(R, Qm)
        out = []
        for i in range(len(R)):
            acc = Iv(0)
            for k in range(len(cvec)):
                if R[i][k] != 0:
                    acc = acc + cvec[k] * R[i][k]
            for k in range(len(qvec)):
                if RQ[i][k] != 0:
                    acc = acc + qvec[k] * RQ[i][k]
            out.append(acc)
        return out

    def cols(R):
        """R (cV_k + QV qV_k), k = 0..m-1 -> matrix [row][k] of intervals"""
        res = [[None] * m for _ in range(len(R))]
        for k in range(m):
            v = apply_iv(R, [cV[i][k] for i in range(2 * n)], QV, [qV[i][k] for i in range(2 * n)])
            for i in range(len(R)):
                res[i][k] = v[i]
        return res

    # unpreconditioned Dgt(X), only to build Y
    D0 = cols(ell)
    Dmid = [[round_fr(D0[i][j].mid()) for j in range(m)] for i in range(m)]
    Y = inv_exact(Dmid)
    Yl = matmul(Y, ell)
    Yg = apply_iv(Yl, c, Q, q)
    YJ = cols(Yl)
    h = Fr(bx["h"])
    ws = [Fr(x) for x in bx["w"]]
    widths = [h, h] + [ws[(i // 2) % len(ws)] for i in range(M - 1)]
    if mutate == "shift":           # g -> g + Dmid (2h e_A): the zero moves to about -2h e_A, outside X
        d = [Fr(0)] * m
        d[0] = 2 * h
        Dd = [sum((Dmid[i][j] * d[j] for j in range(m)), Fr(0)) for i in range(m)]
        YDd = [sum((Y[i][j] * Dd[j] for j in range(m)), Fr(0)) for i in range(m)]
        Yg = [Yg[i] + YDd[i] for i in range(m)]
    if mutate == "widen":           # g(0) widened by 1e3 h in the outer components (raw units)
        wid = 1000 * h
        gw = [Iv(-wid, wid) if i < 2 else Iv(0) for i in range(m)]
        Yg = [Yg[i] + sum((gw[j] * Y[i][j] for j in range(m)), Iv(0)) for i in range(m)]
    K = []
    for i in range(m):
        t = -Yg[i]
        for j in range(m):
            a = (Iv(1) if i == j else Iv(0)) - YJ[i][j]
            r = a.mag() * widths[j]
            t = t + Iv(-r, r)
        K.append(t)
    inside = [K[i].mag() < widths[i] for i in range(m)]
    ok = all(inside)
    rel = [float((widths[i] - K[i].mag()) / widths[i]) for i in range(m)]
    print(f"rv3 K1 (exact rational intervals), {path}{' [mutation ' + mutate + ']' if mutate else ''}")
    print(f"   M = {M}, n = {n}, h = {bx['h']}, w = {bx['w']}; path {bx['path']}, steps {bx['steps']}")
    print(f"   start data: {bx['start']['start'][:60]}...; rigorous flag {bx['start'].get('rigorous')}")
    print(f"   |g(0)| (raw, max over rows) <= {max(float(x.mag()) for x in apply_iv(ell, c, Q, q)):.3e}; "
          f"max radius of (Y Dgt(X))_ij {max(float((YJ[i][j].hi - YJ[i][j].lo) / 2) for i in range(m) for j in range(m)):.3e}")
    print("   Y Dgt(X), midpoints: " + "; ".join(", ".join(f"{float(YJ[i][j].mid()):+.2e}" for j in range(m))
                                          for i in range(m)))
    print("   K(X) in int X per coordinate: " + ", ".join(str(b) for b in inside))
    print("   relative margins (w_i - |K_i|)/w_i: " + ", ".join(f"{r:+.4f}" for r in rel))
    # the zero in p coordinates
    T = [[Iv(1 if i == j else 0) for j in range(m)] for i in range(m)]
    for i, row in enumerate(bx["C"]):
        for j in range(2):
            T[2 + i][j] = parse_ball(row[j])
    eps = Fr(1, 2 ** 255)
    phat = []
    for x in bx["p"]:
        v = Fr(x)
        phat.append(Iv(v - abs(v) * eps, v + abs(v) * eps))
    pstar = []
    for i in range(m):
        acc = phat[i]
        for j in range(m):
            acc = acc + T[i][j] * K[j]
        pstar.append(acc)
    names = ["A", "w/w_inf"] + [f"{b}{nu}" for nu in range(3, M + 1, 2) for b in ("B", "C")]
    for i, nm in enumerate(names):
        print(f"   {nm:8s} in [{float(pstar[i].lo):.16g}, {float(pstar[i].hi):.16g}]")
    status = 0 if ok else 1
    if ok and mutate is None and M == 3 and "1e-10" in path:
        within = all(Fr(QUOTED_M3[i][0]) <= pstar[i].lo and pstar[i].hi <= Fr(QUOTED_M3[i][1]) for i in range(m))
        print(f"   inside the enclosure quoted in A_M.md 6.2: {within}")
        status = 0 if within else 1
    print("VERDICT: " + ("K(X) in int X: exactly one zero, non-singular Jacobian" if ok else "K(X) NOT in int X"))
    return status


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    mut = None
    for a in sys.argv[1:]:
        if a.startswith("--mutate="):
            mut = a.split("=", 1)[1]
    sys.exit(main(args[0], mut))

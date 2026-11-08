#!/usr/bin/env python3
"""Independent check of lemma 6b1(5): the disc lemma and the covering conditions C1-C8, in mpmath interval arithmetic
(review of #39). Nothing here imports python-flint or the code under review; the inputs are files.

The system (proof/A_lemmas_M.md section 1), rewritten by me in u = 1/z (d/du = -z^2 d/dz), state as stored by the
runs, [P_mu (odd mu ascending), V_mu, kappa (C_nu - delta_nu0) (even nu ascending)], kappa = 100:
    P_u = -z W / 2,        W = V + [(C - 1) Q],  Q = V + (1 + s) P,  s = i mu w
    V_u = -V/2 - ([(C - 1) Q] + [(C - 1) W])/2 - z (-(2 + s) Q - (1 + s) W + [[C E] P]) / 2,   E = [P P]/2
    C_u = -z^2 [C E] / 2
Each bracket is a product projected onto its result's harmonics (|mu| <= M odd, |nu| <= M + 1 even); the triple product
is taken as [[C E] P], the nesting of the code under review (the statement's [C E P] does not fix it).

Disc lemma (my implementation, same claim as lemma6b1_disc_M.certify): for every exact centre c in the disc's centre
interval, every y0 in the box Y0 and every w in the w interval, the solution with y(c) = y0 is analytic on |u - c| < r.
  1. p: the Taylor polynomial of degree n at c, by the coefficient recursion p_{k+1} = [F(p)]_k / (k + 1) (lazy
     series, each coefficient computed once).
  2. The defect d = F(c + s, p(s)) - p'(s), as a Taylor model of order n in s on |s| <= r: polynomial part plus
     s^(n+1) E(s), |E| <= e. Products truncate at n and bound the dropped part by the triangle inequality;
     z = 1/(c + s) is its Taylor polynomial plus s^(n+1) (-1)^(n+1) / (c^(n+1) (c + s)). The coefficients below n
     must contain 0 (else the polynomial is not the Taylor polynomial of this F), and
     sup |d_i(s)| / |s|^n <= Delta_i = |d_{i,n}| + r e_i.
  3. T(R) = int_0^tau [d + F(p + R) - F(p)] on R = O(tau^(n+1)), norm max_i om_i sup |R_i| / |tau|^(n+1):
     ||T(0)|| <= F0 = max_i om_i Delta_i / (n + 1), Lipschitz Lam = r/(n + 2) max_i sum_j om_i L_ij / om_j, with
     L_ij >= sup |dF_i/dy_j| over the rectangle box B_j = Y0_j + (Msup_j + b_j)(+-1 +- i), Msup_j = sum_k |p_jk| r^k,
     z in 1/c + (+-1 +- i) r/(|c|(|c| - r)) (forward-mode derivatives on intervals). With rho = 2 F0/(1 - Lam):
     F0 + Lam rho <= rho, and rho r^(n+1)/om_j <= b_j keeps p + R in B. Weights om: the Perron vector of L (a choice).
Covering (C1-C8): the conditions of proof/A_certify.md 12.3 on the exact numbers of the design (re-implemented here,
compared with the stored t lists), with the tiles glued row by row; see `cover`.

    python3 check_6b1.py discs --inputs=<dir> --rows=17,16 --out=<dir> [--mutate=sign] [--discs=0,5] [--r_scale=1.01]
    python3 check_6b1.py cover --runs=<stored run dir> --mine=<dir of disc verdicts> [--json=f] [--mutate=drop:9,5 |
        radius:9 | g0]
Exit status 0 only if every requested disc certifies (discs) or every condition holds (cover).
"""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import civ as C  # noqa: E402
from civ import (Z0, CZ, ri, radd, rsub, rmul, rdiv, rup, rlt, rle, rgt, rmaxup, cadd, csub, cmul, cmulr, cinv,  # noqa
                 cabsup, cabslow, cinflate, ccontains0, cfrom_me, cexpi, rpi, ri_dec)
from mpmath.libmp import fzero, to_float, from_float, mpf_lt  # noqa: E402

KAPPA = 100
ONE = (ri(1), Z0)
HALF_R = (C.from_man_exp(1, -1), C.from_man_exp(1, -1))
HALF = (HALF_R, Z0)
MHALF = (C.rneg(HALF_R), Z0)


def harmonics(M):
    return [m for m in range(-M, M + 1) if m % 2], [n for n in range(-(M + 1), M + 2) if n % 2 == 0]


# ------------------------------------------------------------------ the right-hand side, once, over a ring R

def rhs_generic(M, Y, z, z2, w, R, mutate=None):
    """the nd components of the u-system for ring elements Y (state), z = 1/u and z2 = z^2; w a complex interval.
    R provides lin([(coef, X), ...], const) and pairs([(X, Y), ...]) = sum X*Y."""
    odd, even = harmonics(M)
    no, ne = len(odd), len(even)
    P = {m: Y[i] for i, m in enumerate(odd)}
    V = {m: Y[no + i] for i, m in enumerate(odd)}
    Ct = {n: Y[2 * no + i] for i, n in enumerate(even)}
    ik = (rdiv(ri(1), ri(KAPPA)), Z0)
    s = {m: cmul((Z0, ri(m)), w) for m in odd}
    s1 = {m: cadd(ONE, s[m]) for m in odd}
    s2 = {m: cadd((ri(2), Z0), s[m]) for m in odd}
    Cm1 = {n: R.lin([(ik, Ct[n])]) for n in even}
    Cf = {n: R.lin([(ONE, Cm1[n])], ONE if n == 0 else None) for n in even}
    Q = {m: R.lin([(ONE, V[m]), (s1[m], P[m])]) for m in odd}

    def pr(X, Yd, out):
        return {k: R.pairs([(X[a], Yd[k - a]) for a in sorted(X) if (k - a) in Yd]) for k in out}
    CQ = pr(Cm1, Q, odd)
    W = {m: R.lin([(ONE, V[m]), (ONE, CQ[m])]) for m in odd}
    CW = pr(Cm1, W, odd)
    PP = pr(P, P, even)
    E = {k: R.lin([(HALF, PP[k])]) for k in even}
    CE = pr(Cf, E, even)
    CEP = pr(CE, P, odd)
    G = {m: R.lin([(C.cneg(s2[m]), Q[m]), (C.cneg(s1[m]), W[m]), (ONE, CEP[m])]) for m in odd}
    zW = {m: R.pairs([(z, W[m])]) for m in odd}
    zG = {m: R.pairs([(z, G[m])]) for m in odd}
    z2CE = {n: R.pairs([(z2, CE[n])]) for n in even}
    vh = HALF if mutate == "sign" else MHALF          # MUTATION sign: V_u = +V/2 + ...
    dP = [R.lin([(MHALF, zW[m])]) for m in odd]
    dV = [R.lin([(vh, V[m]), (MHALF, CQ[m]), (MHALF, CW[m]), (MHALF, zG[m])]) for m in odd]
    mk2 = (C.rneg(ri(KAPPA // 2)), Z0)
    dC = [R.lin([(mk2, z2CE[n])]) for n in even]
    return dP + dV + dC


# ------------------------------------------------------------------ ring 1: lazy power series (the recursion)

class LZ:
    __slots__ = ("kind", "a", "b", "memo")

    def __init__(self, kind, a=None, b=None):
        self.kind, self.a, self.b, self.memo = kind, a, b, []

    def get(self, k):
        memo = self.memo
        while len(memo) <= k:
            memo.append(self._coef(len(memo)))
        return memo[k]

    def _coef(self, k):
        p = C.PREC
        if self.kind == "lin":
            terms, const = self.a, self.b
            acc = const if (k == 0 and const is not None) else CZ
            for c, X in terms:
                acc = cadd(acc, cmul(c, X.get(k)))
            return acc
        if self.kind == "pairs":
            from mpmath.libmp.libmpi import mpi_add, mpi_sub, mpi_mul
            ar, ai = Z0, Z0
            for X, Yv in self.a:
                for i in range(k + 1):
                    xr, xi = X.get(i)
                    yr, yi = Yv.get(k - i)
                    ar = mpi_add(ar, mpi_sub(mpi_mul(xr, yr, p), mpi_mul(xi, yi, p), p), p)
                    ai = mpi_add(ai, mpi_add(mpi_mul(xr, yi, p), mpi_mul(xi, yr, p), p), p)
            return (ar, ai)
        if self.kind == "state":
            y0, F = self.a, self.b
            if k == 0:
                return y0
            return cmulr(F[0].get(k - 1), rdiv(ri(1), ri(k)))
        if self.kind == "z":                      # z_k = (-1)^k / c^(k+1)
            z0 = self.a
            if k == 0:
                return z0
            return C.cneg(cmul(self.get(k - 1), z0))
        raise ValueError(self.kind)


class LZRing:
    @staticmethod
    def lin(terms, const=None):
        return LZ("lin", terms, const)

    @staticmethod
    def pairs(prs):
        return LZ("pairs", prs)


# ------------------------------------------------------------------ ring 2: Taylor models of order N on |s| <= r

class TMRing:
    """an element: (coefs[0..N], e) with f(s) = sum coefs_k s^k + s^(N+1) E(s), |E(s)| <= e (point interval) on
    |s| <= r"""

    def __init__(self, N, r):
        self.N, self.r = N, r
        self.rp = [ri(1)]
        for _ in range(2 * N + 2):
            self.rp.append(rmul(self.rp[-1], r))
        self.rN1 = self.rp[N + 1]

    def scaled(self, X):
        """|X_k| r^k (point upper bounds), cached on the element"""
        if len(X) == 3:
            return X[2]
        a = [rup(rmul(cabsup(c), self.rp[k])) for k, c in enumerate(X[0])]
        X.append(a)
        return a

    def norm(self, X):
        a = self.scaled(X)
        acc = Z0
        for v in a:
            acc = radd(acc, v)
        return rup(acc)

    def lin(self, terms, const=None):
        N = self.N
        coefs = [CZ] * (N + 1)
        if const is not None:
            coefs[0] = const
        e = Z0
        for c, X in terms:
            for k in range(N + 1):
                coefs[k] = cadd(coefs[k], cmul(c, X[0][k]))
            e = radd(e, rmul(cabsup(c), X[1]))
        return [coefs, rup(e)]

    def pairs(self, prs):
        from mpmath.libmp.libmpi import mpi_add, mpi_sub, mpi_mul
        N, p = self.N, C.PREC
        cr = [Z0] * (N + 1)
        cim = [Z0] * (N + 1)
        e = Z0
        for X, Yv in prs:
            xs, ys = X[0], Yv[0]
            for k in range(N + 1):
                ar, ai = cr[k], cim[k]
                for i in range(k + 1):
                    xr, xi = xs[i]
                    yr, yi = ys[k - i]
                    ar = mpi_add(ar, mpi_sub(mpi_mul(xr, yr, p), mpi_mul(xi, yi, p), p), p)
                    ai = mpi_add(ai, mpi_add(mpi_mul(xr, yi, p), mpi_mul(xi, yr, p), p), p)
                cr[k], cim[k] = ar, ai
            # the dropped part of the product of the polynomial parts: sum_{i + j > N} |X_i| |Y_j| r^(i+j-N-1)
            a, b = self.scaled(X), self.scaled(Yv)
            suf = [Z0] * (N + 2)
            for j in range(N, -1, -1):
                suf[j] = rup(radd(suf[j + 1], b[j]))
            hi = Z0
            for i in range(1, N + 1):
                hi = radd(hi, rmul(a[i], suf[N + 1 - i]))
            hi = rdiv(rup(hi), self.rN1)
            nX, nY = self.norm(X), self.norm(Yv)
            ex, ey = X[1], Yv[1]
            e = radd(e, radd(radd(hi, rmul(ex, nY)), radd(rmul(ey, nX), rmul(self.rN1, rmul(ex, ey)))))
        return [list(zip(cr, cim)), rup(e)]


# ------------------------------------------------------------------ ring 3: forward-mode derivatives (vector tangents)

class DRing:
    def __init__(self, nd):
        self.nd = nd

    def lin(self, terms, const=None):
        v = const if const is not None else CZ
        d = [CZ] * self.nd
        for c, X in terms:
            v = cadd(v, cmul(c, X[0]))
            for j in range(self.nd):
                if X[1][j] is not CZ:
                    d[j] = cadd(d[j], cmul(c, X[1][j]))
        return (v, d)

    def pairs(self, prs):
        v = CZ
        d = [CZ] * self.nd
        for X, Yv in prs:
            v = cadd(v, cmul(X[0], Yv[0]))
            for j in range(self.nd):
                if X[1][j] is not CZ:
                    d[j] = cadd(d[j], cmul(X[1][j], Yv[0]))
                if Yv[1][j] is not CZ:
                    d[j] = cadd(d[j], cmul(X[0], Yv[1][j]))
        return (v, d)


# ------------------------------------------------------------------ the disc lemma

def perron_weights(Lf):
    """weights om_i = 1/u_i, u the Perron vector of L + 1e-12 (floats; any positive weights are valid)"""
    nd = len(Lf)
    A = [[Lf[i][j] + 1e-12 for j in range(nd)] for i in range(nd)]
    u = [1.0] * nd
    for _ in range(500):
        v = [sum(A[i][j] * u[j] for j in range(nd)) for i in range(nd)]
        s = max(v)
        u = [x / s for x in v]
    m = max(u)
    return [1.0 / (x + 1e-6 * m) for x in u]


def certify_disc(M, c, Y0, w, r, n, mutate=None, verbose=False):
    """the disc lemma above; c, w complex intervals, Y0 list of complex intervals, r real interval, n int.
    Returns a dict (ok, Lam, F0, Delta, Msup, low_ok, ...)."""
    nd = len(Y0)
    t0 = time.time()
    absc = cabslow(c)
    if not rlt(r, absc):
        return dict(ok=False, why="disc reaches u = 0")
    # 1. Taylor polynomial (lazy recursion)
    zL = LZ("z", cinv(c))
    z2L = LZRing.pairs([(zL, zL)])
    states = [LZ("state", Y0[i], [None]) for i in range(nd)]
    F = rhs_generic(M, states, zL, z2L, w, LZRing)
    for i in range(nd):
        states[i].b[0] = F[i]
    for k in range(n + 1):
        for i in range(nd):
            states[i].get(k)
    p = [[states[i].get(k) for k in range(n + 1)] for i in range(nd)]
    t1 = time.time()
    # 2. defect as a Taylor model of order N = n
    N = n
    TR = TMRing(N, r)
    zc = [zL.get(k) for k in range(N + 1)]
    # |z - z_N(s)| = |s|^(N+1) / (|c|^(N+1) |c + s|) <= |s|^(N+1) / (|c|^(N+1) (|c| - r))
    ez = rup(rdiv(ri(1), rmul(C.mpi_pow_int(absc, N + 1, C.PREC), rsub(absc, r))))
    zT = [zc, ez]
    z2T = TR.pairs([(zT, zT)])
    pT = [[p[i], Z0] for i in range(nd)]
    FT = rhs_generic(M, pT, zT, z2T, w, TR, mutate=mutate)
    low_ok = True
    Delta = []
    for i in range(nd):
        d = [csub(FT[i][0][k], cmulr(p[i][k + 1], ri(k + 1)) if k < n else CZ) for k in range(N + 1)]
        for k in range(n):
            if not ccontains0(d[k]):
                low_ok = False
        Delta.append(rup(radd(cabsup(d[n]), rmul(r, FT[i][1]))))
    t2 = time.time()
    # sup of |p_j(s) - y0_j| on the disc
    Msup = []
    for i in range(nd):
        acc = Z0
        for k in range(1, n + 1):
            acc = radd(acc, rmul(cabsup(p[i][k]), TR.rp[k]))
        Msup.append(rup(acc))
    # 3. contraction
    rz = rup(rdiv(r, rmul(absc, rsub(absc, r))))
    Zs = cinflate(cinv(c), rz[1])
    Z2s = cmul(Zs, Zs)
    DR = DRing(nd)
    b = [(from_float(1e-30), from_float(1e-30))] * nd
    res = None
    for it in range(6):
        Bx = [cinflate(Y0[i], radd(Msup[i], b[i])[1]) for i in range(nd)]
        Yd = [(Bx[i], [ONE if j == i else CZ for j in range(nd)]) for i in range(nd)]
        Fd = rhs_generic(M, Yd, (Zs, [CZ] * nd), (Z2s, [CZ] * nd), w, DR)
        L = [[cabsup(Fd[i][1][j]) for j in range(nd)] for i in range(nd)]
        om = perron_weights([[to_float(L[i][j][1]) for j in range(nd)] for i in range(nd)])
        omi = [(from_float(x), from_float(x)) for x in om]
        rows = []
        for i in range(nd):
            acc = Z0
            for j in range(nd):
                acc = radd(acc, rdiv(rmul(omi[i], L[i][j]), omi[j]))
            rows.append(acc)
        Lam = rup(rmul(rdiv(r, ri(n + 2)), rmaxup(rows)))
        F0 = rmaxup([rdiv(rmul(omi[i], Delta[i]), ri(n + 1)) for i in range(nd)])
        if not rlt(Lam, ri(1)):
            res = dict(ok=False, why="no contraction", Lam=Lam, F0=F0, it=it)
            break
        rho = radd(rup(rdiv(rmul(ri(2), F0), rsub(ri(1), Lam))), (from_float(1e-300), from_float(1e-300)))
        selfmap = rle(radd(F0, rmul(Lam, rho)), rho)
        rn1 = TR.rp[n + 1]
        breq = [rdiv(rmul(rho, rn1), omi[j]) for j in range(nd)]
        consistent = all(rle(breq[j], b[j]) for j in range(nd))
        res = dict(ok=bool(selfmap and consistent and low_ok), Lam=Lam, F0=F0, rho=rho, low_ok=low_ok,
                   selfmap=selfmap, consistent=consistent, it=it,
                   sup=max(to_float(radd(Msup[j], b[j])[1]) for j in range(nd)),
                   breq=max(to_float(x[1]) for x in breq))
        if consistent:
            break
        b = [rup(rmul(ri(2), breq[j])) for j in range(nd)]
    res["sec"] = (t1 - t0, t2 - t1, time.time() - t2)
    res["Lam_f"] = to_float(res["Lam"][1])
    res["F0_f"] = to_float(res["F0"][1])
    for key in ("Lam", "F0", "rho"):
        res.pop(key, None)
    return res


# ------------------------------------------------------------------ the design (re-implemented)

CONF5 = dict(rho0=405, u_in=25, u_out=260, g0="402")
GRID = 2.0 ** -30


def disc_params(rho):
    if rho >= 100:
        return 24, 70
    if rho >= 50:
        return 16, 60
    if rho >= 32:
        return 10, 50
    return 6, 50


def design():
    """the M = 5 rows (floats, as the runs chose them; every condition is re-checked on these exact numbers)"""
    rows = []
    rho = float(CONF5["rho0"])
    pi34 = 3 * math.pi / 4
    while True:
        r, n = disc_params(rho)
        dl = 0.6 * r
        ca = 1 - ((0.92 * r) ** 2 - dl ** 2) / (2 * rho * (rho + dl))
        K = math.ceil(pi34 / (1.6 * math.acos(ca)))
        alpha = 0.6 * (pi34 / K)
        rows.append(dict(j=len(rows), rho=rho, r=r, n=n, a=rho - dl, b=rho + dl, K=K, alpha=alpha,
                         t=[round(k / K / GRID) * GRID for k in range(K + 1)]))
        if rho - dl < CONF5["u_in"] - 0.5:
            break
        rn = rho - dl
        for _ in range(50):
            dn = 0.6 * disc_params(rn)[0]
            rn_new = (rho - dl) + 0.15 * min(dl, dn) - dn
            if abs(rn_new - rn) < 1e-9:
                break
            rn = rn_new
        rho = round(rn, 3)
    return rows


def phi_of(t):
    """arg u at arc parameter t: -(3 pi/4)(1 - t)"""
    p34 = rmul(rdiv(ri(3), ri(4)), rpi())
    tt = (from_float(t), from_float(t))
    return C.rneg(rmul(p34, rsub(ri(1), tt)))


def centre(rho, t):
    """rho e^{i phi(t)}, rho the decimal the runs used"""
    return cmulr(cexpi(phi_of(t)), ri_dec(repr(rho)))


# ------------------------------------------------------------------ command: discs

def cmd_discs(opts):
    C.setprec(int(opts.get("prec", "128")))
    M = 5
    rows = design()
    out = opts["out"]
    os.makedirs(out, exist_ok=True)
    mutate = opts.get("mutate")
    rscale = opts.get("r_scale")
    allok = True
    for j in [int(v) for v in opts["rows"].split(",")]:
        fin = os.path.join(opts["inputs"], f"row_{j:02d}.json")
        if "wait" in opts:
            # the inputs are written row by row by dump_inputs.py; a lock lets two processes share one row list
            while not os.path.exists(fin):
                time.sleep(60)
            try:
                os.close(os.open(os.path.join(out, f"row_{j:02d}.lock"), os.O_CREAT | os.O_EXCL))
            except FileExistsError:
                continue
        with open(fin) as fh:
            inp = json.load(fh)
        rw = rows[j]
        assert inp["t"] == rw["t"] and inp["rho"] == rw["rho"] and inp["r"] == rw["r"] and inp["n"] == rw["n"], \
            f"row {j}: inputs made with another design"
        sel = [int(v) for v in opts["discs"].split(",")] if "discs" in opts else range(len(inp["discs"]))
        rec = {"j": j, "rho": rw["rho"], "r": rw["r"], "n": rw["n"], "mutate": mutate, "r_scale": rscale,
               "prec": C.PREC, "discs": []}
        for k in sel:
            dk = inp["discs"][k]
            c_mine = centre(rw["rho"], rw["t"][k])
            c_orig = cfrom_me(dk["c"])
            # the stored centre ball and mine must overlap (both contain the exact centre)
            ov = all(not (mpf_lt(c_mine[q][1], c_orig[q][0]) or mpf_lt(c_orig[q][1], c_mine[q][0])) for q in (0, 1))
            Y0 = [cfrom_me(y) for y in dk["Y0"]]
            w = cfrom_me(dk["w"])
            r = ri(rw["r"]) if rscale is None else rmul(ri(rw["r"]), ri_dec(rscale))
            t0 = time.time()
            res = certify_disc(M, c_mine, Y0, w, r, rw["n"], mutate=mutate)
            res.update(k=k, centre_overlap=ov, orig_ok=dk["orig"]["ok"], orig_Lam=dk["orig"]["Lam"],
                       wall=time.time() - t0)
            res["ok"] = bool(res["ok"] and ov)
            allok = allok and res["ok"]
            rec["discs"].append(res)
            print(f"row {j} disc {k}/{len(inp['discs']) - 1}: ok {res['ok']}, Lam {res['Lam_f']:.6f} (orig "
                  f"{dk['orig']['Lam']:.6f}, ok {dk['orig']['ok']}), F0 {res['F0_f']:.2e}, low coefficients 0: "
                  f"{res.get('low_ok')}, sup {res.get('sup', float('nan')):.4g}, it {res.get('it')}, "
                  f"{res['wall']:.1f} s" + (f" [{res.get('why')}]" if res.get("why") else ""), flush=True)
        tag = "" if (mutate is None and rscale is None) else f"_{mutate or ''}{'r' + rscale if rscale else ''}"
        with open(os.path.join(out, f"row_{j:02d}{tag}.json"), "w") as fh:
            json.dump(rec, fh, indent=1)
    return 0 if allok else 3


# ------------------------------------------------------------------ command: cover (C1-C8)

def imax(x, y):
    """an interval containing max(x*, y*) for x* in x, y* in y"""
    return (x[0] if not mpf_lt(x[0], y[0]) else y[0], x[1] if not mpf_lt(x[1], y[1]) else y[1])


def imin(x, y):
    return (x[0] if mpf_lt(x[0], y[0]) else y[0], x[1] if mpf_lt(x[1], y[1]) else y[1])


class Rect:
    """open polar rectangle {a < |u| < b, t1 < arg u < t2}; bounds are intervals containing the exact bounds, b None
    for infinity"""

    def __init__(self, a, b, t1, t2, name):
        self.a, self.b, self.t1, self.t2, self.name = a, b, t1, t2, name

    def meet(self, o):
        b = o.b if self.b is None else (self.b if o.b is None else imin(self.b, o.b))
        return Rect(imax(self.a, o.a), b, imax(self.t1, o.t1), imin(self.t2, o.t2), "")

    def status(self):
        """True: proven nonempty; False: proven empty; None: undecided"""
        if (self.b is not None and rle(self.b, self.a)) or rle(self.t2, self.t1):
            return False
        if (self.b is None or rlt(self.a, self.b)) and rlt(self.t1, self.t2):
            return True
        return None

    def has(self, rho, th):
        return rlt(self.a, rho) and (self.b is None or rlt(rho, self.b)) and rlt(self.t1, th) and rlt(th, self.t2)


def cmd_cover(opts):
    C.setprec(256)
    rows = design()
    fails = []

    def need(cond, msg):
        if not cond:
            fails.append(msg)
        return cond
    mut = opts.get("mutate", "")
    drop = tuple(int(v) for v in mut.split(":")[1].split(",")) if mut.startswith("drop:") else None
    rmut = int(mut.split(":")[1]) if mut.startswith("radius:") else None
    g0s = CONF5["g0"]
    g0 = ri_dec(g0s)
    if mut == "g0":
        g0 = rmul(g0, ri_dec("1.01"))                       # MUTATION g0: G0's radius x 1.01 (402 -> 406.02)
    # C1: every run present, made with this design and this X, and every disc certified (by this review's lemma)
    with open(os.path.join(opts["runs"], "ray.json")) as fh:
        ray = json.load(fh)
    need(ray["h"] == "1e-18" and ray["rho"] == [rw["rho"] for rw in rows] and ray.get("c1"), "C1: ray.json design/box")
    if "x" in opts:
        xs = opts["x"].split(",")
        need(len(xs) == 6 and all(a[:38] == b[:38] for a, b in zip(ray["x"], xs)), "C1: ray.json centre is not p^")
    need(all(str(j) in ray["sets"] for j in range(len(rows))), "C1: ray states missing")
    ndisc = 0
    for rw in rows:
        j = rw["j"]
        f = os.path.join(opts["runs"], f"row_{j:02d}.json")
        if not need(os.path.exists(f), f"C1: row {j} missing"):
            continue
        with open(f) as fh:
            run = json.load(fh)
        rr = run["r"] * 1.01 if j == rmut else run["r"]      # MUTATION radius:j: the stored row's r x 1.01
        need(run["h"] == "1e-18" and run["rho"] == rw["rho"] and rr == rw["r"] and run["n"] == rw["n"],
             f"C1: row {j}: stored run made with another design (rho, r, n or h)")
        need([d["t"] for d in run["discs"]] == rw["t"], f"C1: row {j}: centres differ from the design")
        mf = os.path.join(opts["mine"], f"row_{j:02d}.json")
        if need(os.path.exists(mf), f"C1: row {j}: no disc verdicts of this review"):
            with open(mf) as fh:
                mine = json.load(fh)
            need(mine["r"] == rw["r"] and mine["n"] == rw["n"] and mine["rho"] == rw["rho"] and mine["mutate"] is None
                 and mine["r_scale"] is None, f"C1: row {j}: verdicts for another disc set")
            ks = [d["k"] for d in mine["discs"]]
            need(ks == list(range(len(rw["t"]))), f"C1: row {j}: verdicts not for every disc")
            for d in mine["discs"]:
                ndisc += 1
                need(d["ok"], f"C1: disc ({j}, {d['k']}) not certified by this review")
    pi = rpi()
    p34 = rmul(rdiv(ri(3), ri(4)), pi)
    R = [ri_dec(repr(rw["rho"])) for rw in rows]
    A = [(from_float(rw["a"]),) * 2 for rw in rows]
    B = [(from_float(rw["b"]),) * 2 for rw in rows]
    Al = [(from_float(rw["alpha"]),) * 2 for rw in rows]
    Rd = [ri(rw["r"]) for rw in rows]
    Ph = [[phi_of(t) for t in rw["t"]] for rw in rows]
    for j, rw in enumerate(rows):
        need(rw["t"][0] == 0.0 and rw["t"][-1] == 1.0, f"C4: row {j}: end centres")
        need(rlt(A[j], R[j]) and rlt(R[j], B[j]), f"row {j}: centre outside its band")
        need(rgt(Al[j], Z0) and rlt(Al[j], ri(1)), f"row {j}: tile half-angle")
        cosA = C.rcos_sin(Al[j])[0]
        for k in range(len(Ph[j])):
            # C2: the closed tile lies in the open disc; the farthest points are the corners (convexity in rho)
            for rr in (A[j], B[j]):
                d2 = rsub(radd(rmul(rr, rr), rmul(R[j], R[j])), rmul(rmul(ri(2), rmul(rr, R[j])), cosA))
                need(rlt(d2, rmul(Rd[j], Rd[j])), f"C2: tile ({j}, {k}) not inside its disc")
            if k:
                dph = rsub(Ph[j][k], Ph[j][k - 1])
                # C3: the half arcs to the midpoint lie in their discs (chord 2 rho sin(dphi/4))
                ch = rmul(rmul(ri(2), R[j]), C.rcos_sin(rdiv(dph, ri(4)))[1])
                need(rgt(dph, Z0) and rlt(ch, Rd[j]), f"C3: arc halves ({j}, {k})")
                # C4: consecutive tiles overlap, non-consecutive do not
                need(rlt(rdiv(dph, ri(2)), Al[j]) and rlt(Al[j], dph), f"C4: tiles ({j}, {k - 1}), ({j}, {k})")
        # C4: the first tile reaches below -3pi/4, the last above 0
        need(rlt(rsub(Ph[j][0], Al[j]), C.rneg(p34)) and rgt(radd(Ph[j][-1], Al[j]), Z0), f"C4: row {j} ends")
        if j:
            rq = rdiv(radd(A[j - 1], B[j]), ri(2))
            # C5 and C8: bands overlap, the ray point q between the centres and in both bands, segments in the discs
            need(rlt(A[j - 1], B[j]), f"C8: bands {j - 1}, {j} do not overlap")
            need(rlt(R[j], rq) and rlt(rq, R[j - 1]), f"C5: q between rows {j - 1}, {j}")
            need(rlt(rsub(R[j - 1], rq), Rd[j - 1]) and rlt(rsub(rq, R[j]), Rd[j]), f"C5: ray segments {j - 1}, {j}")
            need(rlt(A[j], rq) and rlt(rq, B[j]) and rlt(A[j - 1], rq) and rlt(rq, B[j - 1]), f"C5: q in both bands")
    # C8: the bands cover [u_in, u_out]; also reported: they reach G0's radius g0 (what the M = 5 domain needs)
    need(rgt(B[0], ri(CONF5["u_out"])) and rlt(A[-1], ri(CONF5["u_in"])), "C8: bands do not cover [25, 260]")
    reach_g0 = bool(rgt(B[0], g0))
    # C6: s0 in G0
    need(rgt(R[0], g0), "C6: s0 = rho_0 e^{-3 i pi/4} not in G0")
    # C7: gluing, row by row from the outside, each row from the ray to the axis
    G0 = Rect(g0, None, rsub(C.rneg(p34), ri_dec("0.4")), Z0, "G0")
    done = [G0]
    npieces = nlinks = 0
    for j, rw in enumerate(rows):
        for k in range(len(Ph[j])):
            if drop is not None and (j, k) == drop:
                continue
            T = Rect(A[j], B[j], rsub(Ph[j][k], Al[j]), radd(Ph[j][k], Al[j]), f"T{j},{k}")
            pieces = []
            for X in done:
                Pm = T.meet(X)
                st = Pm.status()
                if st is False:
                    continue
                pieces.append((X.name, Pm, st))
            npieces += len(pieces)
            if k:
                qn, qr, qt = f"T{j},{k - 1}", R[j], rdiv(radd(Ph[j][k - 1], Ph[j][k]), ri(2))
            elif j:
                qn, qr, qt = f"T{j - 1},0", rdiv(radd(A[j - 1], B[j]), ri(2)), C.rneg(p34)
            else:
                qn, qr, qt = "G0", R[0], C.rneg(p34)
            qp = [Pm for (nm, Pm, st) in pieces if nm == qn]
            need(len(qp) == 1 and qp[0].has(qr, qt), f"C7: tile ({j}, {k}): agreement point not in the meeting set")
            if not need(all(st is True for (_, _, st) in pieces), f"C7: tile ({j}, {k}): undecided piece"):
                done.append(T)
                continue
            names = [nm for (nm, _, _) in pieces]
            start = names.index(qn) if qn in names else 0
            seen, stack = set(), [start]
            while stack:
                i = stack.pop()
                if i in seen:
                    continue
                seen.add(i)
                for l in range(len(pieces)):
                    if l not in seen and pieces[i][1].meet(pieces[l][1]).status() is True:
                        stack.append(l)
                        nlinks += 1
            need(len(seen) == len(pieces), f"C7: tile ({j}, {k}): meeting set not connected")
            done.append(T)
    if drop is not None:
        jj, kk = drop
        need(any(X.has(R[jj], Ph[jj][kk]) for X in done), f"C8: point at tile ({jj}, {kk}) not covered")
    ok = not fails
    out = dict(ok=ok, nfails=len(fails), fails=fails[:50], tiles=len(done) - 1, discs_certified=ndisc,
               pieces=npieces, links=nlinks, bands_reach_g0=reach_g0, mutate=mut or None)
    print(f"tiles glued {len(done) - 1}, meeting-set pieces {npieces}, links {nlinks}; disc verdicts read {ndisc}; "
          f"bands reach g0 = {g0s}: {reach_g0}")
    print(f"LEMMA 6b1(5) COVER CHECK (review): {'PASS' if ok else 'FAIL'}"
          + ("" if ok else f" ({len(fails)} failures: {fails[:4]})") + (f"   [MUTATION {mut}]" if mut else ""))
    if "json" in opts:
        with open(opts["json"], "w") as fh:
            json.dump(out, fh, indent=1)
    return 0 if ok else 3


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
    cmd = args[0] if args else "discs"
    if cmd == "discs":
        return cmd_discs(opts)
    if cmd == "cover":
        return cmd_cover(opts)
    raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    sys.exit(main())

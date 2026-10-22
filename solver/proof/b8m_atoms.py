#!/usr/bin/env python3
"""Lemma B8 (proof/B8_constants.md, section 3): weighted bounds of the known (truncated-series) quantities of the
M = 1 model (p, m on {+-1}; b, c on {0, +-2}; b8m_model.py), in ball arithmetic over a complex parameter polydisc.

Norms: l1 over harmonics for vectors (odd O = {-1, 1}, even E = {-2, 0, 2}); for operators on odd vectors the induced
l1 norm (maximum column sum).  For a vector (or a matrix) of bivariate polynomials the descriptor is the table
c_{jm} = sum over components (and the maximum over columns) of |coefficient of eps^j x^m|, with m0 = min m, w = max(m - 2j);
Lemma W gives |p(x)|_1 <= theta^m0 X^w sum_j (sum_m c_jm) rho^j.

Quantities (K = truncation order; Z^K = sum_{k<=K} eps^k Z_k from b8m_formal.py):
  vectors  V, S, S/x^2, beta, beta/x^2, c - 1, cb = [c beta], cb/x^2, Y = V + S, Dl (truncated Delta series), each with
           x f' and x (x f')';
  operators C^K (from c - 1), Gamma = (C^K^2 - I)/(x^2 - 1) (exact division), det = det C^K, and the order-1 transport
           eta~ = eta/x^2,  eta = A^K - (n-1) I + (n-1) Gamma (section 4), as Neta = det * eta~ (polynomial);
  residuals tau_D = x[cS'] - x^2 Dl' - x Om Dl + [[c beta] S] + (n-1)[c S]   (O(eps^K)),
           tau_v = [c Dl] - x (V + S) (O(eps^(K+1))), tau_b = x beta' + (n-1) beta + [beta beta] - (1-eps)(H + eps'[beta H])
           (O(eps^K), H = ([SS] + [Dl Dl])/2), rho4 = x c' - [c beta] (O(eps^(K+1))), with x f' and x (x f')'.
Structural zeros (Lemma F: parity, vanishing order at x = 0, eps-order of the residuals) are imposed after checking at
the centre that the removed entries are rounding noise.

    python3 b8m_atoms.py [K] [rA] [rW]
"""
import sys
from flint import acb, arb, acb_poly, ctx
import b8_formal as BF          # bivariate polynomial helpers ({j: acb_poly})
import b8m_formal as MF

X1 = acb_poly([0, 1])
O, E = [-1, 1], [-2, 0, 2]


def comp_bp(series, comp, K):
    return BF.bp({k: series[k][comp] for k in range(min(K + 1, len(series))) if comp in series[k]})


def vprod(A, B, keep):
    out = {k: {} for k in keep}
    for i, a in A.items():
        for j, b in B.items():
            if i + j in out:
                out[i + j] = BF.badd(out[i + j], BF.bmul(a, b))
    return out


def vadd(*vs):
    out = {}
    for v in vs:
        for k, p in v.items():
            out[k] = BF.badd(out.get(k, {}), p)
    return out


def vscale(v, s):
    return {k: BF.bscale(p, s) for k, p in v.items()}


def vxd(v):
    return {k: BF.bx(BF.bdx(p)) for k, p in v.items()}


def vshift(v, j):
    return {k: BF.bshift(p, j) for k, p in v.items()}


def vx(v, j=1):
    return {k: BF.bx(p, j) for k, p in v.items()}


def vdivx2(v):
    out = {}
    for k, p in v.items():
        q = {}
        for j, pp in p.items():
            cs = list(pp.coeffs())
            for m in (0, 1):
                if m < len(cs) and not cs[m].contains(0):
                    raise ValueError("divide by x^2: not divisible")
            q[j] = acb_poly(cs[2:]) if len(cs) > 2 else acb_poly([])
        out[k] = BF.bp(q)
    return out


def struct_vec(v, jmin=None, m0=0, parity=None):
    out, rem = {}, arb(0)
    for k, p in v.items():
        q = {}
        for j, pp in p.items():
            cs = []
            for m, cf in enumerate(pp.coeffs()):
                drop = (jmin is not None and j < jmin) or m < m0 or (parity is not None and m % 2 != parity)
                if drop:
                    u = arb(abs(cf).upper())
                    rem = u if u > rem else rem
                    cs.append(acb(0))
                else:
                    cs.append(cf)
            qq = acb_poly(cs)
            if any(not (c == 0) for c in cs):
                q[j] = qq
        out[k] = q
    return out, rem


def desc_table(entries):
    """descriptor of the l1 sum of a list of bivariate polynomials: coefficient table summed over the entries"""
    coef = {}
    for p in entries:
        for j, q in p.items():
            for m, cf in enumerate(q.coeffs()):
                if not (cf == 0):
                    coef[(j, m)] = arb((coef.get((j, m), arb(0)) + abs(cf)).upper())
    if not coef:
        return None
    m0 = min(m for (_, m) in coef)
    w = max(m - 2 * j for (j, m) in coef)
    jm = min(j for (j, _) in coef)
    return dict(m0=m0, w=w, coef=coef, jmin=jm)


def desc_max(ds):
    ds = [d for d in ds if d is not None]
    coef = {}
    for d in ds:
        for k, v in d["coef"].items():
            coef[k] = v if k not in coef or v > coef[k] else coef[k]
    m0 = min(m for (_, m) in coef)
    w = max(m - 2 * j for (j, m) in coef)
    return dict(m0=m0, w=w, coef=coef, jmin=min(j for (j, _) in coef))


def vdesc(v):
    return desc_table(list(v.values()))


def mdesc(Mx):
    """induced l1 norm: maximum over columns of the column sums"""
    cols = sorted({c for (_, c) in Mx})
    return desc_max([desc_table([Mx[(r, c)] for (r, c2) in Mx if c2 == c]) for c in cols])


def mat_mul(A, B):
    out = {}
    for (r, k1), a in A.items():
        for (k2, c), b in B.items():
            if k1 == k2:
                out[(r, c)] = BF.badd(out.get((r, c), {}), BF.bmul(a, b))
    return out


def mat_add(*Ms):
    out = {}
    for Mx in Ms:
        for k, p in Mx.items():
            out[k] = BF.badd(out.get(k, {}), p)
    return out


def mat_scale(Mx, s):
    return {k: BF.bscale(p, s) for k, p in Mx.items()}


def mult_op(g):
    """operator f -> [g f] on O for an even-harmonic vector g"""
    return {(mu, nu): g.get(mu - nu, {}) for mu in O for nu in O}


def atoms(K, rA="0", rW="0", prec=256, verbose=False):
    ctx.prec = prec
    F, om = MF.params(rA, rW, M=1)
    fs = MF.formal(K + 1, F, om, M=1)          # K+1 orders computed; truncation at K
    V = {m: comp_bp(fs["V"], m, K) for m in O}
    S = {m: comp_bp(fs["S"], m, K) for m in O}
    Dl = {m: comp_bp(fs["Dl"], m, K) for m in O}
    be = {k: comp_bp(fs["be"], k, K) for k in E}
    c = {k: comp_bp(fs["c"], k, K) for k in E}
    one = BF.bconst(acb(1))
    cm1 = dict(c)
    cm1[0] = BF.badd(c[0], BF.bscale(one, -1))
    Om = {m: acb(0, m) * om for m in O}
    D = {}
    rems = {}

    def put(name, v, jmin=None, m0=0, parity=0, deriv=2, isvec=True):
        vv, r = struct_vec(v, jmin, m0, parity)
        rems[name] = r
        D[name] = vdesc(vv)
        d1, r1 = struct_vec(vxd(vv), jmin, max(m0, 1) if m0 > 0 else 0, parity)
        D["x" + name + "'"] = vdesc(d1)
        if deriv >= 2:
            d2, _ = struct_vec(vxd(d1), jmin, max(m0, 1) if m0 > 0 else 0, parity)
            D["x(x" + name + "')'"] = vdesc(d2)
        return vv

    put("V", V, None, 0, 0)
    S = put("S", S, 1, 2, 0)
    put("S/x2", vdivx2(S), 1, 0, 0)
    be = put("beta", be, 1, 2, 0)
    put("beta/x2", vdivx2(be), 1, 0, 0)
    cm1 = put("c-1", cm1, 1, 0, 0)
    cb = vprod(c, be, E)
    cb = put("cb", cb, 1, 2, 0)
    put("cb/x2", vdivx2(cb), 1, 0, 0)
    Y = vadd(V, S)
    put("Y", Y, None, 0, 0)
    put("Dl", Dl, None, 1, 1)
    # operators
    C = mat_add(mult_op(c))
    I2 = {(mu, nu): (one if mu == nu else {}) for mu in O for nu in O}
    C2mI = mat_add(mat_mul(C, C), mat_scale(I2, -1))
    Gam = {}
    for k, p in C2mI.items():
        q = {}
        for j, pp in p.items():
            qq, r = divmod(pp, acb_poly([-1, 0, 1]))
            for cf in r.coeffs():
                if not cf.contains(0):
                    raise ValueError("Gamma: C^2 - I not divisible by x^2 - 1")
            q[j] = qq
        Gam[k] = BF.bp(q)
    Gam = {k: struct_vec({0: v}, 1, 0, 0)[0][0] for k, v in Gam.items()}
    D["Gamma"] = mdesc(Gam)
    xG = {k: BF.bx(BF.bdx(v)) for k, v in Gam.items()}
    D["xGamma'"] = mdesc(xG)
    D["x(xGamma')'"] = mdesc({k: BF.bx(BF.bdx(v)) for k, v in xG.items()})
    det = BF.badd(BF.bmul(C[(1, 1)], C[(-1, -1)]), BF.bscale(BF.bmul(C[(1, -1)], C[(-1, 1)]), -1))
    detm1 = struct_vec({0: BF.badd(det, BF.bscale(one, -1))}, 1, 0, 0)[0][0]
    D["det-1"] = desc_table([detm1])
    D["xdet'"] = desc_table([BF.bx(BF.bdx(detm1))])
    D["x(xdet')'"] = desc_table([BF.bx(BF.bdx(BF.bx(BF.bdx(detm1))))])
    adj = {(1, 1): C[(-1, -1)], (-1, -1): C[(1, 1)], (1, -1): BF.bscale(C[(1, -1)], -1),
           (-1, 1): BF.bscale(C[(-1, 1)], -1)}
    # Neta = det * eta~ = (n-1) det Gamma + det C M_{cb/x^2} - det (2 Om + I) - ([C, Om] - M_cb) adj
    nm1 = BF.badd(BF.bconst(acb(1), -1), BF.bscale(one, -1))          # n - 1 = eps^-1 - 1
    cbx2 = vdivx2(cb)
    Mcb, Mcbx2 = mult_op(cb), mult_op(cbx2)
    OmM = {(mu, nu): (BF.bconst(Om[mu]) if mu == nu else {}) for mu in O for nu in O}
    detI = {(mu, nu): (det if mu == nu else {}) for mu in O for nu in O}
    comm = mat_add(mat_mul(C, OmM), mat_scale(mat_mul(OmM, C), -1))
    t1 = {k: BF.bmul(BF.bmul(nm1, det), v) for k, v in Gam.items()}
    t2 = {k: BF.bmul(det, v) for k, v in mat_mul(C, Mcbx2).items()}
    t3 = mat_scale(mat_mul(detI, mat_add(mat_scale(OmM, 2), I2)), -1)
    t4 = mat_scale(mat_mul(mat_add(comm, mat_scale(Mcb, -1)), adj), -1)
    Neta = mat_add(t1, t2, t3, t4)
    Neta = {k: struct_vec({0: v}, 0, 0, 0)[0][0] for k, v in Neta.items()}
    D["Neta"] = mdesc(Neta)
    xN = {k: BF.bx(BF.bdx(v)) for k, v in Neta.items()}
    D["xNeta'"] = mdesc(xN)
    D["x(xNeta')'"] = mdesc({k: BF.bx(BF.bdx(v)) for k, v in xN.items()})
    # residual vectors
    epsP = BF.bconst(acb(1), 1)
    Sx = {m: BF.bdx(S[m]) for m in O}
    cSx = vprod(c, Sx, O)
    cbS = vprod(cb, S, O)
    cS = vprod(c, S, O)
    tauD = vadd(vx(cSx), vscale(vx({m: BF.bdx(Dl[m]) for m in O}, 2), -1),
                {m: BF.bscale(BF.bx(Dl[m]), -Om[m]) for m in O}, cbS,
                {m: BF.bmul(nm1, cS[m]) for m in O})
    put("tauD", tauD, K, 2, 0)
    tauv = vadd(vprod(c, Dl, O), vscale(vx(Y), -1))
    put("tauv", tauv, K + 1, 3, 1)
    H = vscale(vadd(vprod(S, S, E), vprod(Dl, Dl, E)), acb("0.5"))
    bH = vprod(be, H, E)
    taub = vadd(vxd(be), {k: BF.bmul(nm1, be[k]) for k in E}, vprod(be, be, E),
                vscale(H, -1), {k: BF.bmul(epsP, H[k]) for k in E}, vscale({k: BF.bmul(epsP, bH[k]) for k in E}, -1))
    put("taub", taub, K, 2, 0)
    rho4 = vadd(vxd(c), vscale(cb, -1))
    put("rho4", rho4, K + 1, 2, 0)
    consts = dict(absw=arb(abs(om).upper()), absf=arb(max(abs(F[m]) for m in O).upper()))
    if verbose:
        print("removed structural entries (should be rounding noise at a point):",
              {k: float(v) for k, v in rems.items()})
    return D, consts


def wv(d, rho):
    return sum((c * rho ** j for (j, m), c in d["coef"].items()), arb(0))


if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    rA = sys.argv[2] if len(sys.argv) > 2 else "0"
    rW = sys.argv[3] if len(sys.argv) > 3 else "0"
    D, C = atoms(K, rA, rW, verbose=True)
    print(f"K = {K}, polydisc rA = {rA}, rW = {rW}; |omega| <= {float(C['absw']):.6f}")
    for k, d in D.items():
        if d is None:
            print(f"  {k:12s} (zero)")
            continue
        js = {}
        for (j, m), v in d["coef"].items():
            js[j] = js.get(j, arb(0)) + v
        ns = sorted(js.items())
        print(f"  {k:12s} m0 {d['m0']}, w {d['w']:+d}, j {ns[0][0]}..{ns[-1][0]}: "
              + ", ".join(f"[{j}] {float(v):.4g}" for j, v in ns[:4]))

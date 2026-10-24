#!/usr/bin/env python3
"""P2 review: an independent formal 1/n series of the M = 1 model, derived directly from the ORIGINAL equations of
proof/B8_constants.md section 1.1 (p, m, b, c), not from B8's reduced system (b8m_model.py) or recursion
(b8m_formal.py). mpmath complex arithmetic at the box centre (A*, omega* = omega_inf w*), 60 digits.

Derivation (P2_review.md, item LC-formal). With p = e^{1/2} P, m = e^{1/2} M (e = 1/n), S = P + M, Dl = P - M,
beta = (n - 1)(b - 1), Om = diag(i mu omega), every product projected onto its harmonic set:
    (p) - (m):  e x^2 Dl' - e x [c S'] = -e x Om Dl + e [[c beta] S] + (1 - e) [c S]
    (p) + (m):  x^2 S' - x [c Dl'] = -x Om S + [[c beta] Dl] - [c Dl]
    (b):        e x beta' + (1 - e) beta + e [beta beta] = e (1 - e) H + e^2 [beta H],   H = ([S S] + [Dl Dl])/2
    (c):        x c' = [c beta]
Order by order (Z = sum e^k Z_k): S_0 = 0, beta_0 = 0, c_0 = 1, Dl_0 = 2 F x; for k >= 1 the first equation gives S_k,
the third beta_k, the fourth c_k (gauge c_k(1) = 0, the light cone at x = 1 to all orders), the second Dl_k from
(1 - x d/dx) Dl_k = RHS (the x^1 coefficient of Dl_k is 0: the profile Dl'(0) = 2F is fixed; RHS has no x^1 term,
checked), and V_k = ([c Dl]_k - x S_k)/x.

Outputs compared with b8m_atoms_K4.log (rA = rW = 0 run) and with the notes: the coefficient sums (descriptors) of V,
S, beta, c - 1, Y, Dl, Gamma, det - 1, det eta~, tau_D, tau_v, tau_b, rho4; |c_k(0)|_1 for k = 5, 6, 7 (B_chain.md
section 4: 1.227e5, 1.52e6, 1.95e7); the exact rho4 part of the gauge offset, sum |rho4_jm| n0^(5-j)/m (B_chain.md
13.4: 1.90e5 over the box).

    python3 p2_formal.py [--K=4] [--order=8] [--mutate=sign]
    mutation sign: the Om term of the (p) - (m) equation with the wrong sign; the descriptor comparison must fail.
"""
import json
import re
import sys

from mpmath import mp, mpc, mpf, pi, fabs

mp.dps = 60
OPTS = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
MUT = OPTS.get("mutate")
ODD, EVEN = (-1, 1), (-2, 0, 2)


# ---------------------------------------------------------------- polynomials in x: dict power -> mpc
def padd(*ps):
    out = {}
    for p in ps:
        for m, v in p.items():
            out[m] = out.get(m, 0) + v
    return out


def pscale(p, s):
    return {m: v * s for m, v in p.items()}


def pmul(p, q):
    out = {}
    for a, u in p.items():
        for b, v in q.items():
            out[a + b] = out.get(a + b, 0) + u * v
    return out


def pder(p):
    return {m - 1: v * m for m, v in p.items() if m != 0}


def px(p, j=1):
    return {m + j: v for m, v in p.items()}


def pdivx(p):
    if abs(p.get(0, 0)) > mpf(10) ** (-40):
        raise ValueError("not divisible by x")
    return {m - 1: v for m, v in p.items() if m != 0}


def peval(p, x):
    return sum((v * x ** m for m, v in p.items()), mpc(0))


# ---------------------------------------------------------------- harmonic vectors of polynomials
def vadd(*vs):
    out = {}
    for v in vs:
        for h, p in v.items():
            out[h] = padd(out.get(h, {}), p)
    return out


def vscale(v, s):
    return {h: pscale(p, s) for h, p in v.items()}


def proj(f, g, keep):
    """[f g] projected onto the harmonic set keep"""
    out = {h: {} for h in keep}
    for a, p in f.items():
        for b, q in g.items():
            if a + b in out:
                out[a + b] = padd(out[a + b], pmul(p, q))
    return out


def vder(v):
    return {h: pder(p) for h, p in v.items()}


def vx(v, j=1):
    return {h: px(p, j) for h, p in v.items()}


def zero(keep):
    return {h: {} for h in keep}


def series_coef(ser, k, keep):
    return ser[k] if 0 <= k < len(ser) else zero(keep)


def sprod(f, g, k, keep):
    """the e^k coefficient of [f g] for series f, g (lists of harmonic vectors)"""
    out = zero(keep)
    for i in range(k + 1):
        if i < len(f) and k - i < len(g):
            out = vadd(out, proj(f[i], g[k - i], keep))
    return out


def formal(order, A, w):
    F = {1: A / 2, -1: A / 2}                      # B8's normalization F_{+1} = F_{-1} = A/2
    Om = {m: mpc(0, m) * w for m in ODD}
    sgn = -1 if MUT == "sign" else 1
    S = [zero(ODD)]
    Dl = [{m: {1: 2 * F[m]} for m in ODD}]
    be = [zero(EVEN)]
    c = [{0: {0: mpc(1)}, 2: {}, -2: {}}]
    H = []
    for k in range(1, order + 1):
        cb = [sprod(c, be, j, EVEN) for j in range(k)]
        # (p) - (m), order k: S_k = x^2 Dl'_{k-1} - x[c S']_{k-1} + x Om Dl_{k-1} - [[c beta] S]_{k-1} + [c S]_{k-1}
        #                            - sum_{j=1}^{k-1} [c_j S_{k-j}]
        Sx = [vder(s) for s in S]
        t = vadd(vx(vder(Dl[k - 1]), 2), vscale(vx(sprod(c, Sx, k - 1, ODD)), -1),
                 {m: pscale(px(Dl[k - 1][m]), sgn * Om[m]) for m in ODD},
                 vscale(sprod(cb, S, k - 1, ODD), -1), sprod(c, S, k - 1, ODD))
        for j in range(1, k):
            t = vadd(t, vscale(proj(c[j], S[k - j], ODD), -1))
        S.append(t)
        # H_{k-1}
        H.append(vscale(vadd(sprod(S, S, k - 1, EVEN), sprod(Dl, Dl, k - 1, EVEN)), mpf("0.5")))
        # (b), order k: beta_k = beta_{k-1} - x beta'_{k-1} - [beta beta]_{k-1} + H_{k-1} - H_{k-2} + [beta H]_{k-2}
        bk = vadd(be[k - 1], vscale(vx(vder(be[k - 1])), -1), vscale(sprod(be, be, k - 1, EVEN), -1), H[k - 1])
        if k >= 2:
            bk = vadd(bk, vscale(H[k - 2], -1), sprod(be, H, k - 2, EVEN))
        be.append(bk)
        # (c), order k: x c_k' = [c beta]_k, c_k(1) = 0
        cbk = sprod(c, be, k, EVEN)
        ck = {}
        for h in EVEN:
            p = cbk[h]
            if abs(p.get(0, 0)) > mpf(10) ** (-40):
                raise ValueError("[c beta] has a constant term")
            q = {m: v / m for m, v in p.items() if m != 0}
            q[0] = -sum(q.values(), mpc(0))
            ck[h] = q
        c.append(ck)
        # (p) + (m), order k: (1 - x d/dx) Dl_k = -x^2 S_k' - x Om S_k + [[c beta] Dl]_k
        #                                         - sum_{j>=1} ([c_j Dl_{k-j}] - x [c_j Dl'_{k-j}])
        cbfull = [sprod(c, be, j, EVEN) for j in range(k + 1)]
        rhs = vadd(vscale(vx(vder(S[k]), 2), -1), {m: pscale(px(S[k][m]), -Om[m]) for m in ODD},
                   sprod(cbfull, Dl + [zero(ODD)], k, ODD))
        for j in range(1, k + 1):
            rhs = vadd(rhs, vscale(proj(c[j], Dl[k - j], ODD), -1), vx(proj(c[j], vder(Dl[k - j]), ODD)))
        dk = {}
        for m in ODD:
            if abs(rhs[m].get(1, 0)) > mpf(10) ** (-30):
                raise ValueError(f"solvability: x^1 coefficient {rhs[m].get(1)} at order {k}")
            dk[m] = {p: v / (1 - p) for p, v in rhs[m].items() if p != 1}
        Dl.append(dk)
    V = []
    for k in range(order + 1):
        cD = sprod(c, Dl, k, ODD)
        V.append({m: pdivx(padd(cD[m], pscale(px(S[k][m]), -1))) for m in ODD})
    return dict(S=S, Dl=Dl, be=be, c=c, V=V, F=F, Om=Om)


# ---------------------------------------------------------------- bivariate (eps, x) objects: dict (j, m) -> mpc
def trunc_vec(ser, K, keep):
    """sum_{k <= K} e^k Z_k as {h: {(j, m): coef}}"""
    out = {h: {} for h in keep}
    for k in range(min(K + 1, len(ser))):
        for h, p in ser[k].items():
            for m, v in p.items():
                out[h][(k, m)] = out[h].get((k, m), 0) + v
    return out


def b_add(*ps):
    out = {}
    for p in ps:
        for key, v in p.items():
            out[key] = out.get(key, 0) + v
    return out


def b_mul(p, q):
    out = {}
    for (j1, m1), u in p.items():
        for (j2, m2), v in q.items():
            key = (j1 + j2, m1 + m2)
            out[key] = out.get(key, 0) + u * v
    return out


def b_scale(p, s):
    return {k: v * s for k, v in p.items()}


def b_xd(p):
    """x d/dx"""
    return {(j, m): v * m for (j, m), v in p.items() if m != 0}


def b_x(p, i=1):
    return {(j, m + i): v for (j, m), v in p.items()}


def b_eps(p, i=1):
    return {(j + i, m): v for (j, m), v in p.items()}


def bv_proj(f, g, keep):
    out = {h: {} for h in keep}
    for a, p in f.items():
        for b, q in g.items():
            if a + b in out:
                out[a + b] = b_add(out[a + b], b_mul(p, q))
    return out


def bv_add(*vs):
    out = {}
    for v in vs:
        for h, p in v.items():
            out[h] = b_add(out.get(h, {}), p)
    return out


def bv_scale(v, s):
    return {h: b_scale(p, s) for h, p in v.items()}


NM1 = {(-1, 0): mpc(1), (0, 0): mpc(-1)}          # n - 1 = e^-1 - 1


def jsums(entries):
    """descriptor j-sums: sum over entries and over m of |coef(e^j x^m)|, entries cleaned at 1e-40"""
    out = {}
    for p in entries:
        for (j, m), v in p.items():
            if fabs(v) > mpf(10) ** (-40):
                out[j] = out.get(j, 0) + fabs(v)
    return out


def op_jsums(Mx):
    """induced l1 norm descriptor: per (j, m) the max over columns of the column sums, then summed over m"""
    cols = {}
    for (r, cc), p in Mx.items():
        for key, v in p.items():
            if fabs(v) > mpf(10) ** (-40):
                cols.setdefault(cc, {})
                cols[cc][key] = cols[cc].get(key, 0) + fabs(v)
    mx = {}
    for cc, d in cols.items():
        for key, v in d.items():
            mx[key] = max(mx.get(key, 0), v)
    out = {}
    for (j, m), v in mx.items():
        out[j] = out.get(j, 0) + v
    return out


def mult_op(g):
    return {(mu, nu): g.get(mu - nu, {}) for mu in ODD for nu in ODD}


def mat_mul(A, B):
    out = {}
    for (r, k1), a in A.items():
        for (k2, cc), b in B.items():
            if k1 == k2:
                out[(r, cc)] = b_add(out.get((r, cc), {}), b_mul(a, b))
    return out


def mat_add(*Ms):
    out = {}
    for Mx in Ms:
        for k, p in Mx.items():
            out[k] = b_add(out.get(k, {}), p)
    return out


def mat_scale(Mx, s):
    return {k: b_scale(p, s) for k, p in Mx.items()}


def divide_x2m1(p):
    """exact division of a bivariate polynomial by x^2 - 1 (in x, for each j); returns quotient, max |remainder|"""
    out = {}
    js = sorted({j for j, _ in p})
    rem = mpf(0)
    for j in js:
        coefs = {m: v for (jj, m), v in p.items() if jj == j}
        if not coefs:
            continue
        deg = max(coefs)
        a = [coefs.get(m, mpc(0)) for m in range(deg + 1)]
        q = [mpc(0)] * max(deg - 1, 0)
        for m in range(deg, 1, -1):          # divide by x^2 - 1
            qm = a[m]
            q[m - 2] = qm
            a[m] -= qm
            a[m - 2] += qm
        rem = max(rem, fabs(a[0]), fabs(a[1]) if deg >= 1 else 0)
        for m, v in enumerate(q):
            if v != 0:
                out[(j, m)] = v
    return out, rem


def run(K, order, A, w, verbose=True):
    fs = formal(order, A, w)
    S, Dl, be, c, V, F, Om = (fs[k] for k in ("S", "Dl", "be", "c", "V", "F", "Om"))
    out = {}
    # c_k(0) (the formal light-cone gauge offset terms)
    ck0 = {k: sum(fabs(c[k][h].get(0, 0)) for h in EVEN) for k in range(1, order + 1)}
    out["ck0"] = {k: float(v) for k, v in ck0.items()}
    # truncated series Z^K as bivariate objects
    St, Dt, bt, ct, Vt = (trunc_vec(s, K, keep) for s, keep in ((S, ODD), (Dl, ODD), (be, EVEN), (c, EVEN), (V, ODD)))
    Yt = bv_add(Vt, St)
    cm1 = {h: dict(p) for h, p in ct.items()}
    cm1[0] = b_add(cm1[0], {(0, 0): mpc(-1)})
    Omb = {m: {(0, 0): Om[m]} for m in ODD}
    cb = bv_proj(ct, bt, EVEN)
    # residuals (B8_constants.md section 4, at Z^K)
    tauD = bv_add(bv_proj(ct, {m: b_xd(p) for m, p in St.items()}, ODD),
                  {m: b_scale(b_x(b_xd(Dt[m])), -1) for m in ODD},      # -x^2 Dl' = -x (x Dl')
                  {m: b_scale(b_x(Dt[m]), -Om[m]) for m in ODD},
                  bv_proj(cb, St, ODD),
                  {m: b_mul(NM1, bv_proj(ct, St, ODD)[m]) for m in ODD})
    tauv = bv_add(bv_proj(ct, Dt, ODD), {m: b_scale(b_x(Yt[m]), -1) for m in ODD})
    Hb = bv_scale(bv_add(bv_proj(St, St, EVEN), bv_proj(Dt, Dt, EVEN)), mpf("0.5"))
    bH = bv_proj(bt, Hb, EVEN)
    taub = bv_add({h: b_xd(p) for h, p in bt.items()}, {h: b_mul(NM1, p) for h, p in bt.items()},
                  bv_proj(bt, bt, EVEN), bv_scale(Hb, -1), {h: b_eps(p) for h, p in Hb.items()},
                  {h: b_scale(b_eps(p), -1) for h, p in bH.items()})
    rho4 = bv_add({h: b_xd(p) for h, p in ct.items()}, bv_scale(cb, -1))
    # operators
    Cop = mult_op(ct)
    I2 = {(mu, nu): ({(0, 0): mpc(1)} if mu == nu else {}) for mu in ODD for nu in ODD}
    C2mI = mat_add(mat_mul(Cop, Cop), mat_scale(I2, -1))
    Gam, remG = {}, mpf(0)
    for key, p in C2mI.items():
        Gam[key], r = divide_x2m1(p)
        remG = max(remG, r)
    det = b_add(b_mul(Cop[(1, 1)], Cop[(-1, -1)]), b_scale(b_mul(Cop[(1, -1)], Cop[(-1, 1)]), -1))
    detm1 = b_add(det, {(0, 0): mpc(-1)})
    adj = {(1, 1): Cop[(-1, -1)], (-1, -1): Cop[(1, 1)], (1, -1): b_scale(Cop[(1, -1)], -1),
           (-1, 1): b_scale(Cop[(-1, 1)], -1)}
    # det eta~ = (n-1) det Gamma + det C M_{cb}/x^2 - det (Om + I) + (M_cb - C Om) adj   (derived independently)
    cbx2 = {h: {(j, m - 2): v for (j, m), v in p.items()} for h, p in cb.items()}
    if any(m < 0 for p in cbx2.values() for (_, m) in p):
        raise ValueError("[c beta] not divisible by x^2")
    Mcb, Mcbx2 = mult_op(cb), mult_op(cbx2)
    OmM = {(mu, nu): ({(0, 0): Om[mu]} if mu == nu else {}) for mu in ODD for nu in ODD}
    detI = {(mu, nu): (det if mu == nu else {}) for mu in ODD for nu in ODD}
    t1 = {k: b_mul(b_mul(NM1, det), v) for k, v in Gam.items()}
    t2 = {k: b_mul(det, v) for k, v in mat_mul(Cop, Mcbx2).items()}
    t3 = mat_scale(mat_mul(detI, mat_add(OmM, I2)), -1)
    t4 = mat_mul(mat_add(Mcb, mat_scale(mat_mul(Cop, OmM), -1)), adj)
    Neta = mat_add(t1, t2, t3, t4)
    out["jsums"] = {
        "V": jsums(Vt.values()), "S": jsums(St.values()), "beta": jsums(bt.values()), "c-1": jsums(cm1.values()),
        "Y": jsums(Yt.values()), "Dl": jsums(Dt.values()), "Gamma": op_jsums(Gam), "det-1": jsums([detm1]),
        "Neta": op_jsums(Neta), "tauD": jsums(tauD.values()), "tauv": jsums(tauv.values()),
        "taub": jsums(taub.values()), "rho4": jsums(rho4.values())}
    out["remGamma"] = float(remG)
    # lowest eps-orders of the residuals (must be K, K+1, K, K+1)
    out["minj"] = {nm: min(j for j, v in out["jsums"][nm].items() if v > mpf(10) ** (-30))
                   for nm in ("tauD", "tauv", "taub", "rho4")}
    # the exact rho4 part of the gauge offset: sum |rho4_jm| n0^(K+1-j)/m, at n0 = e^73.89
    n0 = mp.exp(mpf("73.89"))
    r4 = mpf(0)
    for h, p in rho4.items():
        for (j, m), v in p.items():
            if fabs(v) > mpf(10) ** (-40):
                if m < 1:
                    raise ValueError("rho4 has a constant term")
                r4 += fabs(v) * n0 ** (K + 1 - j) / m
    out["r4"] = float(r4)
    return out, fs


def compare_with_atoms(js):
    """parse b8m_atoms_K4.log (first block: rA = rW = 0) and compare the printed j-sums (4 significant digits)"""
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "b8m_atoms_K4.log")
    ref = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("K = 4") and ref:
                break
            m = re.match(r"^\s+(\S+)\s+m0 .*?: (.*)$", line)
            if m:
                ref[m.group(1)] = {int(a): float(b) for a, b in re.findall(r"\[(\d+)\] ([0-9.e+-]+)", m.group(2))}
    worst = 0.0
    rows = []
    for nm, d in js.items():
        if nm not in ref:
            continue
        for j, v in ref[nm].items():
            mine = float(d.get(j, 0))
            rel = abs(mine - v) / v
            worst = max(worst, rel)
            rows.append((nm, j, v, mine, rel))
    return worst, rows


def main():
    K = int(OPTS.get("K", "4"))
    order = int(OPTS.get("order", "8"))
    A = mpf("2.5421480475")
    w = 2 * pi / mpf("1.64389375") * mpf("0.7769940136")
    out, fs = run(K, order, A, w)
    print(f"independent formal series of the M = 1 model to order {order} (centre A = {A}, omega = {mp.nstr(w, 10)})")
    print("  |c_k(0)|_1 (light-cone gauge c_k(1) = 0): " + ", ".join(f"k={k}: {v:.4e}" for k, v in out["ck0"].items()))
    print(f"  Gamma = (C^2 - I)/(x^2 - 1): division remainder {out['remGamma']:.1e}")
    print(f"  lowest eps-orders of tau_D, tau_v, tau_b, rho4: {out['minj']} (expected K, K+1, K, K+1)")
    print(f"  exact rho4 part of the gauge offset at n0 = e^73.89: {out['r4']:.6e} (b8m_lc_K4.log, over the box: 1.901089e5)")
    worst, rows = compare_with_atoms(out["jsums"])
    for nm, j, v, mine, rel in rows:
        if rel > 5e-4:
            print(f"   MISMATCH {nm} j={j}: atoms log {v:.4g}, here {mine:.6g} (rel {rel:.1e})")
    print(f"  descriptor j-sums vs b8m_atoms_K4.log (rA = 0; {len(rows)} printed values, 4 digits): worst relative "
          f"difference {worst:.2e}")
    ok = (worst < 5e-4 and out["minj"] == {"tauD": K, "tauv": K + 1, "taub": K, "rho4": K + 1}
          and abs(out["ck0"][5] / 1.227e5 - 1) < 1e-3)
    res = dict(K=K, order=order, ck0=out["ck0"], r4=out["r4"], minj=out["minj"], worst_rel=worst, ok=ok,
               mutate=MUT)
    with open(OPTS.get("json", "p2_formal.json") if not MUT else "p2_formal_mut.json", "w") as fh:
        json.dump(res, fh, indent=1)
    if MUT:
        print(f"MUTATION {MUT}: the comparison must FAIL: {'caught' if not ok else 'NOT caught'}")
        return 0 if not ok else 3
    print(f"=> {'AGREES' if ok else 'DISAGREES'} with B8's formal series and atoms")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

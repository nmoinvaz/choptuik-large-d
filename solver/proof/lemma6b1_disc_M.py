"""Lemma D (the validated Taylor disc, proof/A_certify.md section 12.2) for the M-harmonic system. Rigorous (arb).

The lemma and its proof are those of lemma6b1_disc.py verbatim; only the system changes: lemma6_usys_M.SysUM (the
variables [P_mu, V_mu, kappa (C_nu - delta_nu0)], nd = 2(M + 1) + (M + 2) components), with the right-hand side
lemma6_usys_M.rhs_gen_M (checked against SysUM.rhs by lemma6_usys_M.py, test (f)) and the Jacobian SysUM.lin (test (e)).
Claim checked by certify(M, c, Y0, w, r, n): for every exact centre c' in the ball c, every y0 in the box Y0 and every w
in the ball w, the solution with y(c') = y0 extends analytically to the open disc |u - c'| < r, with |y_j - y0_j| <= M_j.

    python3 lemma6b1_disc_M.py      self-test: M = 1 reproduces lemma6b1_disc.certify on two discs (same Lam, F0, sup)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from flint import acb, arb, acb_poly, acb_series, ctx  # noqa: E402
import lemma6_usys_M as UM  # noqa: E402
from lemma6b1_disc import box_add, up, perron_weights  # noqa: E402


def taylor(M, c, Y0, w, n):
    L = n + 1
    ctx.cap = L
    zs = 1 / acb_series([c, 1], prec=L)
    zero = acb_series([0], prec=L)
    nd = len(Y0)
    Ys = [acb_series([y], prec=L) for y in Y0]
    for _ in range(n):
        F = UM.rhs_gen_M(M, Ys, zs, w, zero)
        Ys = [acb_series([Y0[i]], prec=L) + F[i].integral() for i in range(nd)]
    out = []
    for s_ in Ys:
        cs = s_.coeffs()
        out.append(list(cs) + [acb(0)] * (L - len(cs)))
    return out


def certify(M, c, Y0, w, r, n=40, J=None, verbose=False, mutate=None):
    """lemma6b1_disc.certify for M harmonics (same steps, same checks). mutate='sign': the fast term with the wrong sign
    in the defect (a wrong system): the certificate must fail."""
    r = arb(r)
    J = J or n + 8
    nd = len(Y0)
    absc = arb(abs(c).lower())
    if not absc > r:
        raise ValueError("disc reaches u = 0")
    cf = taylor(M, c, Y0, w, n)
    p = [acb_poly(cf[i][:n + 1]) for i in range(nd)]
    ctx.cap = J + 1
    zser = (1 / acb_series([c, 1], prec=J + 1)).coeffs()
    ez = arb(1) / (absc ** (J + 2) * (1 - r / absc))
    zpoly = acb_poly(list(zser) + [acb(0)] * (J + 1 - len(zser)) + [acb(arb(0, ez), arb(0, ez))])
    zero = acb_poly([0])
    F = UM.rhs_gen_M(M, p, zpoly, w, zero, mutate="sign" if mutate == "sign" else None)
    low_ok = True
    Delta = []
    for i in range(nd):
        d = F[i] - p[i].derivative()
        ds = d.coeffs()
        for k in range(min(n, len(ds))):
            if not ds[k].contains(0):
                low_ok = False
        acc, rp = arb(0), arb(1)
        for k in range(n, len(ds)):
            acc = acc + up(ds[k]) * rp
            rp = rp * r
        Delta.append(acc)
    Msup = []
    for i in range(nd):
        acc, rp = arb(0), arb(1)
        for k in range(1, n + 1):
            rp = rp * r
            acc = acc + up(cf[i][k]) * rp
        Msup.append(acc)
    rz = r / (absc * (absc - r))
    Z = box_add(acb(1) / c, rz.upper())
    sysu = UM.SysUM(M)
    b = [arb("1e-30")] * nd
    res = None
    for it in range(6):
        B = [box_add(Y0[i], (Msup[i] + b[i]).upper()) for i in range(nd)] + [w, acb(0)]
        _, cache = sysu.rhs(B, acb(1) / Z, Z, acb(0))
        L = [[None] * nd for _ in range(nd)]
        for j in range(nd):
            e = [acb(1) if i == j else acb(0) for i in range(nd + 2)]
            col = sysu.lin(cache, e, acb(1) / Z, Z, acb(0))
            for i in range(nd):
                L[i][j] = up(col[i])
        om = perron_weights(L)
        rowsum = [sum(arb(om[i]) * L[i][j] / arb(om[j]) for j in range(nd)) for i in range(nd)]
        Lam = r / (n + 2) * arb(max(x.upper() for x in rowsum))
        F0 = arb(max((arb(om[i]) * Delta[i] / (n + 1)).upper() for i in range(nd)))
        if not Lam < 1:
            res = dict(ok=False, Lam=Lam, F0=F0, why="no contraction")
            break
        rho = arb((2 * F0 / (1 - Lam)).upper()) + arb("1e-300")
        selfmap = F0 + Lam * rho
        breq = [rho * r ** (n + 1) / arb(om[j]) for j in range(nd)]
        consistent = all(breq[j] <= b[j] for j in range(nd))
        res = dict(ok=bool(Lam < 1) and bool(selfmap <= rho) and consistent and low_ok, Lam=Lam, F0=F0, rho=rho,
                   b=b, breq=breq, low_ok=low_ok, Msup=[Msup[j] + b[j] for j in range(nd)], Delta=Delta, om=om, it=it)
        if consistent:
            break
        b = [arb((2 * breq[j]).upper()) for j in range(nd)]
    if verbose:
        print(f"disc |u - c| < {r.str(4)} at c = {acb(c.mid()).str(6)} (M = {M}): n = {n}, Lam = "
              f"{float(res['Lam'].upper()):.3f}, F0 = {float(res['F0'].upper()):.2e}, ok = {res['ok']}"
              + (f", sup|y - y0| <= {max(float(m.upper()) for m in res['Msup']):.3g}" if 'Msup' in res
                 else f" ({res.get('why')})"))
    return res


def selftest():
    """M = 1: certify(1, ...) against lemma6b1_disc.certify on discs at the M = 1 data (lemma6b1_runs/ray.json)"""
    import json
    import lemma6b1_disc as LD
    ctx.prec = 128
    path = os.path.join(HERE, "lemma6b1_runs", "ray.json")
    with open(path) as fh:
        ray = json.load(fh)
    worst = 0.0
    ok = True
    for j, r, n in (("6", "24", 70), ("11", "10", 40)):
        H = [acb(arb(p[0]), arb(p[1])) for p in ray["starts"][j]]
        rho = ray["rho"][int(j)]
        c = acb(arb(str(rho))) * (acb(0, -1) * 3 * arb.pi() / 4).exp()
        r1 = LD.certify(c, H[:7], H[7], r, n)
        r2 = certify(1, c, H[:7], H[7], r, n)
        d = max(abs(float(r1[key].upper()) - float(r2[key].upper())) / float(r1[key].upper()) for key in ("Lam", "F0"))
        worst = max(worst, d)
        ok = ok and r1["ok"] == r2["ok"] and r1["ok"]
        print(f"M = 1 disc at |u| = {rho} on the start ray, r = {r}, n = {n}: lemma6b1_disc Lam "
              f"{float(r1['Lam'].upper()):.6f} ok {r1['ok']}; lemma6b1_disc_M Lam {float(r2['Lam'].upper()):.6f} ok "
              f"{r2['ok']}; relative difference of Lam, F0 {d:.1e}")
    good = ok and worst < 1e-10
    print(f"M = 1 REPRODUCES lemma6b1_disc: {good}")
    return good


if __name__ == "__main__":
    sys.exit(0 if selftest() else 3)

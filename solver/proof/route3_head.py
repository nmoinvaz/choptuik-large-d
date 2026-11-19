#!/usr/bin/env python3
"""P6 route 3: the head j < j0 = max(2 nu, 10) of the Darboux sum (notes/P6_route3.md section 11, #1015).

route3_bound.py certifies |DR_j / K_nu| <= C_mu |p| nu / j^2 for j >= j0 in the one-A^2-step model (section 10). The
normalised multiplier is R_0 + sum_{j >= 0} DR_j / K_nu, so the head is the finite sum over 0 <= j < j0, about 2 nu
terms. This script measures it on the exact recursion (route3_cancel.dr_builder), one perturbation of label
mu in {nu - 2, nu, nu + 2} at a time, at the physical w.

    python3 route3_head.py --scope [--numax=21]   for odd nu = 3..numax and each perturbation: the largest
                                                 j^2 |DR_j/K_nu| / (nu |p|) over the head (against C_mu), the head's
                                                 sum of |DR_j/K_nu| / |p| (the finite-sum bound) and its signed value,
                                                 and the growth in nu of that sum against e^(-0.736 nu)
"""
import math
import sys

from flint import acb, arb, ctx

import route3_cancel as R

MUT = R.MUT
PERTS = {"m": -2, "0": 0, "p": 2}
C_TAIL = {"m": 28.449, "0": 56.874, "p": 4.7661e6}           # section 10.2, kappa = 2, J0 = 10
SIGMA = 0.736                                                # (star)'s rate, B_tail_M7.md 8.4


def opt(name, default):
    return type(default)(next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default))


def j0_of(nu):
    return max(2 * nu, 10)


def head_terms(nu, pk, prec=256, extra=4):
    """|DR_j / K_nu| / |p| for 0 <= j < j0 + extra, and the signed head sum / p, from the exact recursion"""
    ctx.prec = prec
    wv = acb(2 * arb.pi() / arb("1.64389375"))
    amp = acb("0.7", "0.3")
    sh = PERTS[pk]
    labels = [(1, acb(0, -1), 0), (-1, acb(0, 1), 0), (nu + sh, amp, 1)]
    if sh != 0:
        labels.append((nu, acb(0), 1))
    j0 = j0_of(nu)
    J = j0 + extra + 2
    DR, T, TN = R.dr_builder(labels, nu, J, wv, 1, "psi" if "psi" in MUT else "")
    vals, signed = [], acb(0)
    for jv in range(j0 + extra):
        v = DR(jv) / amp
        rel = float(v.rad()) / max(float(abs(v).mid()), 1e-300)
        if rel > 1e-6:
            raise SystemExit(f"precision lost at nu {nu}, {pk}, j {jv}: relative radius {rel:.1e}")
        vals.append(v)
        if jv < j0:
            signed += v
    return vals, signed, j0


def majorants(nu, pk, prec=256):
    """termwise (finite-sum) bounds of |DR_j / K_nu| / |p| for 0 <= j < j0, from the Darboux form
    DR_j/K = -b_j / T^nu_{j+1} + sum_r E_r d0_r (rho^r_{j+1} - rho^r_j) + psi term, b_j = c_j Ss_j + Sd_{j+1},
    with the triangle inequality on every convolution term (order3's sums in absolute value):
      exact    every |T| exact;
      reduced  route3_bound's B3: |T^(+-1)_l| = t_l, |T^mu_l| -> P_mu t_l, |T^nu_(j+1)| -> t_(j+1) (t = |T^1|), and
               |rho^r| <= 1, or P for r = nu + 2 (B2); prefactors exact in nu.
    Returns two lists over j."""
    ctx.prec = prec
    wv = acb(2 * arb.pi() / arb("1.64389375"))
    W = wv.real
    Iv = acb(0, 1)
    sh = PERTS[pk]
    mu = nu + sh
    j0 = j0_of(nu)
    L = j0 + 3
    labs = [(1, acb(0, -1), 0), (-1, acb(0, 1), 0), (mu, acb(1), 1)]
    T = {}
    for lam in {1, -1, mu, nu}:
        a, b = (1 + Iv * lam * wv) / 2, (2 + Iv * lam * wv) / 2
        tt = [acb(1)]
        for l in range(L):
            tt.append(tt[-1] * (l + a) * (l + b) / (l + 1))
        T[lam] = [abs(x) for x in tt]
    t = T[1]
    pi = arb.pi()
    P = (arb(nu) * (pi * (nu + 2) * W).sinh() / ((nu + 2) * (pi * nu * W).sinh())).sqrt() if sh == 2 else arb(1)

    def mag(lam, l, red):
        if not red:
            return T[lam][l]
        return t[l] * (P if lam == mu else 1)

    out = {False: [], True: []}
    for red in (False, True):
        d = {lam: [abs(d0) * mag(lam, l, red) for l in range(L)] for lam, d0, _ in labs}
        s = {lam: [abs(2 * l + 1 + Iv * lam * wv) * d[lam][l] for l in range(L)] for lam, _, _ in labs}
        lapse = {}
        for lp, _, gp in labs:
            for lq, _, gq in labs:
                if gp + gq > 1:
                    continue
                key = (lp + lq, gp + gq)
                acc = lapse.setdefault(key, [arb(0)] * L)
                for m in range(1, L):
                    acc[m] += sum((d[lp][l] * d[lq][m - 1 - l] for l in range(m)), arb(0)) / (8 * m)
        chans = [(lapse[(nu - lr, 1 - gr)], lr) for lr, _, gr in labs if (nu - lr, 1 - gr) in lapse]
        # de-mixing and psi constants (exact values; only their size enters)
        F2 = lambda h, gm: sum((d0p * d0q / 4 for lp, d0p, gp in labs for lq, d0q, gq in labs
                                if lp + lq == h and gp + gq == gm), acb(0))
        aN, bN = (1 + Iv * nu * wv) / 2, (2 + Iv * nu * wv) / 2
        for jv in range(j0):
            cj = abs(2 * jv + 2 + Iv * nu * wv) / (4 * (jv + 1))
            ss = sum((g[i] * s[lr][jv - i] for g, lr in chans for i in range(1, jv + 1)), arb(0))
            sd = sum((g[i] * d[lr][jv + 1 - i] for g, lr in chans for i in range(1, jv + 2)), arb(0))
            den = t[jv + 1] if red else T[nu][jv + 1]
            v = (cj * ss + sd) / den
            for lr, d0r, gr in labs:
                h = nu - lr
                if h != 0:
                    E = abs(F2(h, 1 - gr)) / (abs(h) * W)
                    if red:
                        rho = P if lr == nu + 2 else arb(1)
                        v += E * abs(d0r) * 2 * rho
                    else:
                        v += E * abs(d0r) * (T[lr][jv + 1] / T[nu][jv + 1] + T[lr][jv] / T[nu][jv])
                else:
                    v += abs(F2(0, 1 - gr)) / 2 * abs(d0r) * (1 / abs(aN + jv) + 1 / abs(bN + jv))
            out[red].append(float(v.upper()))
    return out[False], out[True]


def scope(numax):
    rows = []
    for nu in range(3, numax + 1, 2):
        for pk in PERTS:
            vals, signed, j0 = head_terms(nu, pk)
            a = [float(abs(v).mid()) for v in vals]
            q = [(jv * jv * a[jv] / nu, jv) for jv in range(1, j0)]
            qmax, qat = max(q)
            qtail = max(jv * jv * a[jv] / nu for jv in range(j0, len(a)))
            Habs = sum(a[:j0])
            mx, mr = majorants(nu, pk)
            if any(mx[k] < a[k] * (1 - 1e-12) for k in range(j0)):
                raise SystemExit(f"a termwise majorant is below the exact term at nu {nu}, {pk}")
            rows.append(dict(nu=nu, pk=pk, j0=j0, a0=a[0], qmax=qmax, qat=qat, qtail=qtail, Habs=Habs,
                             Hsig=float(abs(signed).mid()), amax=max(a[:j0]), amaxat=max(range(j0), key=lambda k: a[k]),
                             Mx=sum(mx), Mr=sum(mr)))
            print(f"nu {nu:2d} {pk}: j0 {j0:2d}  |DR_0| {a[0]:.3e}  max_j j^2|DR|/nu {qmax:.3e} at j {qat:2d} "
                  f"(C_mu {C_TAIL[pk]:.4g}; first tail j's {qtail:.3e})  head sum |.| {Habs:.4e}, signed {rows[-1]['Hsig']:.4e}"
                  f"  max term {rows[-1]['amax']:.3e} at j {rows[-1]['amaxat']}", flush=True)
    print()
    print("| nu | perturbation | j0 | max j^2|DR_j/K|/(nu|p|), head | at j | / C_mu | head sum of |DR_j/K|/|p| | signed head | "
          "largest term, at j | termwise, exact T | termwise, reduced (B2, B3) | e^(-0.736 nu) x reduced |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['nu']} | {r['pk']} | {r['j0']} | {r['qmax']:.4g} | {r['qat']} | {r['qmax'] / C_TAIL[r['pk']]:.2e} | "
              f"{r['Habs']:.4g} | {r['Hsig']:.4g} | {r['amax']:.3g}, {r['amaxat']} | {r['Mx']:.4g} | {r['Mr']:.4g} | "
              f"{math.exp(-SIGMA * r['nu']) * r['Mr']:.3g} |")
    # growth of the head sum: slopes of log H against log nu and against nu, over nu >= 9
    for pk, key in [(p_, k_) for k_ in ("Habs", "Mx", "Mr") for p_ in PERTS]:
        pts = [(r["nu"], r[key]) for r in rows if r["pk"] == pk and r["nu"] >= 9]
        if len(pts) >= 3:
            xs = [math.log(n) for n, _ in pts]
            ys = [math.log(h) for _, h in pts]
            zs = [n for n, _ in pts]
            sl = lambda X, Y: (len(X) * sum(x * y for x, y in zip(X, Y)) - sum(X) * sum(Y)) / (len(X) * sum(x * x for x in X) - sum(X) ** 2)
            print(f"{key}, perturbation {pk}: ~ nu^{sl(xs, ys):.2f} (log-log slope), rate {sl(zs, ys):+.3f} per unit nu "
                  f"(against sigma = {SIGMA}), over nu = {pts[0][0]}..{pts[-1][0]}")
    return rows


if __name__ == "__main__":
    if "--scope" in sys.argv:
        scope(opt("numax", 21))
        sys.exit(0)
    print(__doc__)
    sys.exit(2)

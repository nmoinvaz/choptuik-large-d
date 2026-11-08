#!/usr/bin/env python3
"""Independent check of lemma 6b1(5)'s start data at s0 = 405 e^{-3 i pi/4} (#189 review). mpmath only, NOT rigorous.

The formal series of the (P, V, C) system of proof/A_lemmas_M.md section 1 (in z = xi^2), by its own order-by-order
recursion written here from those equations:
    2 z P' = V + [(C-1)Q],   2 z^2 V' = V + [(C-1)Q] + [(C-1)W] + z (-(2+s)Q - (1+s)W + [[C E] P]),   C' = [C E]/2,
    Q = V + (1+s)P, W = V + [(C-1)Q], E = [P P]/2, s = i mu w, w = W0 x_1,
with P(0) = 2 F_mu (F from the profile parameters: F_{+-1} = -+ i A/2, F_{+-nu} = -+ i B_nu/2 + C_nu/2), V(0) = 0,
C(0) = 1 (on nu = 0). At order j: c_j = [C E]_{j-1} / (2 j); v_j = 2 (j-1) v_{j-1} - [(C-1)Q]_j - [(C-1)W]_j
+ (2+s) q_{j-1} + (1+s) w_{j-1} - [[C E] P]_{j-1}; p_j = (v_j + [(C-1)Q]_j) / (2 j).
Compared with the stored centre of the C^1 start set (ray.json sets["0"] c, in the SysUM variables P, V,
kappa (C - delta)) and its derivative columns cV (by central differences of the series in each parameter):
    * the truncation N of the stored value is found as the N whose Y_N(z0) matches c best;
    * |Y_N - Y_N'|(z0) for N' > N, the size of the true remainder eta (the series at z0 = 1/405 is far from its
      optimal truncation), against the stored eta bound (the centre radius q);
    * the derivative columns against cV, within the stored column radii qV.
"""
import json
import sys

import mpmath as mp

mp.mp.dps = 110
ODD = list(range(-5, 6, 2))
EVEN = list(range(-6, 7, 2))
KAPPA = 100


def proj(X, Y, out):
    """[X Y]_k for k in out; X, Y dicts harmonic -> mpc"""
    r = {}
    for k in out:
        acc = mp.mpc(0)
        for a, xa in X.items():
            yb = Y.get(k - a)
            if yb is not None:
                acc += xa * yb
        r[k] = acc
    return r


def series(x, J):
    A, wf = x[0], x[1]
    W0 = 2 * mp.pi / mp.mpf("1.64389375")
    w = W0 * wf
    F = {1: -1j * A / 2, -1: 1j * A / 2}
    for i, nu in enumerate((3, 5)):
        B, C = x[2 + 2 * i], x[3 + 2 * i]
        F[nu] = -1j * B / 2 + C / 2
        F[-nu] = 1j * B / 2 + C / 2
    s = {m: 1j * m * w for m in ODD}
    p = [{m: 2 * F[m] for m in ODD}]
    v = [{m: mp.mpc(0) for m in ODD}]
    c = [{n: mp.mpc(1 if n == 0 else 0) for n in EVEN}]
    q = [{m: v[0][m] + (1 + s[m]) * p[0][m] for m in ODD}]
    wv = [{m: v[0][m] for m in ODD}]           # W = V + [(C-1)Q]; (C-1)_0 = 0
    E = []                                      # E_j = [P P]/2 at order j
    CE = []                                     # [C E]_j
    CEP = []                                    # [[C E] P]_j

    def cm1(i):
        return {n: c[i][n] - (1 if (n == 0 and i == 0) else 0) for n in EVEN}

    def conv_series(Xs, Ys, j, out, start=0):
        """order j of the projected product of two series (lists of dicts), sum over i >= start"""
        acc = {k: mp.mpc(0) for k in out}
        for i in range(start, j + 1):
            t = proj(Xs[i], Ys[j - i], out)
            for k in out:
                acc[k] += t[k]
        return acc
    for j in range(1, J):
        # orders j - 1 of E, CE, CEP (need p, c up to j - 1)
        jj = j - 1
        Ej = conv_series(p, p, jj, EVEN)
        E.append({k: e / 2 for k, e in Ej.items()})
        CE.append(conv_series(c, E, jj, EVEN))
        CEP.append(conv_series(CE, p, jj, ODD))
        c.append({n: CE[jj][n] / (2 * j) for n in EVEN})
        Cm = [cm1(i) for i in range(j + 1)]
        # [(C-1)Q]_j and [(C-1)W]_j: sum_{i=1..j} (C-1)_i q_{j-i}
        CQj = {m: mp.mpc(0) for m in ODD}
        CWj = {m: mp.mpc(0) for m in ODD}
        for i in range(1, j + 1):
            t1 = proj(Cm[i], q[j - i], ODD)
            t2 = proj(Cm[i], wv[j - i], ODD)
            for m in ODD:
                CQj[m] += t1[m]
                CWj[m] += t2[m]
        vj = {m: 2 * (j - 1) * v[j - 1][m] - CQj[m] - CWj[m] + (2 + s[m]) * q[j - 1][m] + (1 + s[m]) * wv[j - 1][m]
              - CEP[jj][m] for m in ODD}
        pj = {m: (vj[m] + CQj[m]) / (2 * j) for m in ODD}
        v.append(vj)
        p.append(pj)
        q.append({m: vj[m] + (1 + s[m]) * pj[m] for m in ODD})
        wv.append({m: vj[m] + CQj[m] for m in ODD})
    return p, v, c, w


def partial_sums(x, J, z0):
    """[P_mu, V_mu, kappa (C_nu - delta)] of Y_N at z0 for every N <= J (Y_N: the first N terms)"""
    p, v, c, _ = series(x, J)
    out, acc = {}, [mp.mpc(0)] * 19
    zj = mp.mpc(1)
    for j in range(J):
        terms = [p[j][m] for m in ODD] + [v[j][m] for m in ODD] + [KAPPA * c[j][n] for n in EVEN]
        acc = [a + t * zj for a, t in zip(acc, terms)]
        zj *= z0
        out[j + 1] = [a - (KAPPA if (i == 12 + 3) else 0) for i, a in enumerate(acc)]
    return out


def state(x, N, z0):
    return partial_sums(x, N, z0)[N]


def arbmid(s):
    s = s.strip()
    if s.startswith("["):
        b = s[1:-1]
        if b.strip().startswith("+/-"):
            return mp.mpf(0), mp.mpf(b.strip()[3:])
        m, r = b.split("+/-")
        return mp.mpf(m), mp.mpf(r)
    return mp.mpf(s), mp.mpf(0)


def main():
    ray = sys.argv[1]
    with open(ray) as fh:
        d = json.load(fh)
    x = [mp.mpf(v) for v in d["x"]]
    st = d["sets"]["0"]
    c = [mp.mpc(arbmid(re)[0], arbmid(im)[0]) for re, im in st["c"][:19]]
    qr = max(max(arbmid(v)[1] for v in st["q"]), mp.mpf(0))
    u0 = mp.mpf(d["rho"][0]) * mp.expj(-3 * mp.pi / 4)
    z0 = 1 / u0
    print(f"start point u0 = {d['rho'][0]} e^(-3 i pi/4), z0 = 1/u0; stored centre radius (eta bound) {mp.nstr(qr, 3)}")
    sums = partial_sums(x, 260, z0)
    best = None
    for N in range(40, 180):
        dev = max(abs(a - b) for a, b in zip(sums[N], c))
        if best is None or dev < best[1]:
            best = (N, dev)
    N, dev = best
    yN = sums[N]
    print(f"best-matching truncation N = {N}: max |Y_N(z0) - c| = {mp.nstr(dev, 3)}")
    tail = []
    for Np in (N + 10, N + 40, N + 80):
        tail.append((Np, max(abs(a - b) for a, b in zip(sums[Np], yN))))
    for Np, t in tail:
        print(f"  |Y_{Np}(z0) - Y_{N}(z0)| = {mp.nstr(t, 3)} (the remainder eta is about this size; bound {mp.nstr(qr, 3)})")
    ok_val = dev < mp.mpf("1e-70")
    ok_eta = all(t <= qr for _, t in tail)
    # derivative columns: central differences of Y_N in each parameter, step 1e-30
    hd = mp.mpf("1e-30")
    cV = [[mp.mpc(arbmid(re)[0], arbmid(im)[0]) for re, im in row] for row in st["cV"][:19]]
    qV = st["qV"]
    worst = mp.mpf(0)
    for jpar in range(6):
        xp, xm = list(x), list(x)
        xp[jpar] += hd
        xm[jpar] -= hd
        yp, ym = state(xp, N, z0), state(xm, N, z0)
        for i in range(19):
            dnum = (yp[i] - ym[i]) / (2 * hd)
            rr = max(arbmid(qV[i][jpar])[1], arbmid(qV[21 + i][jpar])[1])
            dev_i = abs(dnum - cV[i][jpar])
            worst = max(worst, dev_i / rr if rr > 0 else (0 if dev_i < 1e-40 else mp.inf))
    print(f"derivative columns: max |dY_N/dx (central difference) - cV| / (stored column radius) = {mp.nstr(worst, 3)}")
    ok_der = worst <= 1
    print(f"SERIES CHECK (numerical evidence, not rigorous): value {ok_val}, remainder within the eta bound {ok_eta}, "
          f"derivatives within the column radii {ok_der}")
    return 0 if (ok_val and ok_eta and ok_der) else 3


if __name__ == "__main__":
    sys.exit(main())

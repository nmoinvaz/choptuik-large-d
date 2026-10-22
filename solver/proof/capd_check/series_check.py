#!/usr/bin/env python3
"""Sanity check of the exported start data against an independent evaluation of the formal series (mpmath).

The M = 1 inner system of proof/A_certify.md section 2, with y = D, sigma = S, c = C on the harmonics mu = +-1
(y, sigma) and nu = 0, +-2 (c), every product projected onto its result's harmonics:

    xi^2 D_mu' + i mu w xi D_mu = [C S]_mu
    xi^2 S_mu' + i mu w xi S_mu = [C (xi D')]_mu + [(C E) D]_mu - [C D]_mu,     E = [D D]/2
    xi C_nu' = [C E]_nu

Inserting D = sum_j d_j xi^(2j+1), S = sum_j s_j xi^(2j+2), C = sum_j g_j xi^(2j), with g_0 = (0, 1, 0) and
d_0 = 2F, F_{+1} = -iA/2, F_{-1} = +iA/2 (the convention of fixed_point.setup), and matching powers of xi:

    xi^(2j+2), first equation:   (2j + 1 + i mu w) d_j = sum_{k+l=j} [g_k s_l]_mu
    xi^(2j+1), second equation:  (2j + i mu w) s_{j-1} = sum_{k+l=j} 2l [g_k d_l]_mu + sum_{m+l=j-1} [(gE)_m d_l]_mu
    xi^(2j),   third equation:   2j g_j = (gE)_{j-1},     e_m = (1/2) sum_{a+b=m} [d_a d_b],  (gE)_m = sum_{k+n=m} [g_k e_n]

so, for j >= 1: g_j from the third; d_j from the second (its l = j term is 2j d_j, since g_0 is the identity on odd
harmonics); s_j from the first. This is written from the equations, not from series_arb.py or fixed_point.py.

The truncated sums (j < N = 53, the truncation fixed_point uses at |xi0| = 0.1) are evaluated at xi0 = 0.1 e^{3 i pi/8}
together with their A- and w/w_inf-derivatives (central differences at 60 digits), and compared with
data/start_point.txt. The start data are Y_N + eta with the same truncation and |eta| <= 5e-32, so the series must lie
inside the exported intervals: the double ones (half-widths about 1e-17), the double-double ones (about 1e-33, the
eta ball of step (i)), and, for the derivatives, the 1.6e-9 / 3.9e-9 Cauchy radii.

    python3 series_check.py
"""
import os

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
mp.mp.dps = 60
ODD, EVEN = (-1, 1), (-2, 0, 2)
W0 = 2 * mp.pi / mp.mpf("1.64389375")
N = 53


def proj(a, b, keep):
    """restricted convolution of harmonic dicts a, b, kept on the harmonics in keep"""
    out = {k: mp.mpc(0) for k in keep}
    for i, x in a.items():
        for j, y in b.items():
            if i + j in out:
                out[i + j] += x * y
    return out


def series(A, wf, J=N):
    w = W0 * wf
    I = mp.mpc(0, 1)
    d = [{1: -I * A, -1: I * A}]
    g = [{-2: mp.mpc(0), 0: mp.mpc(1), 2: mp.mpc(0)}]
    s = [{mu: (1 + I * mu * w) * d[0][mu] for mu in ODD}]
    e = [{nu: v / 2 for nu, v in proj(d[0], d[0], EVEN).items()}]
    gE = [proj(g[0], e[0], EVEN)]
    for j in range(1, J):
        g.append({nu: gE[j - 1][nu] / (2 * j) for nu in EVEN})
        rhs = {mu: mp.mpc(0) for mu in ODD}
        for k in range(1, j + 1):                      # sum_{k+l=j, k>=1} 2l [g_k d_l]
            t = proj(g[k], d[j - k], ODD)
            for mu in ODD:
                rhs[mu] += 2 * (j - k) * t[mu]
        for m in range(0, j):                          # sum_{m+l=j-1} [(gE)_m d_l]
            t = proj(gE[m], d[j - 1 - m], ODD)
            for mu in ODD:
                rhs[mu] += t[mu]
        d.append({mu: ((2 * j + I * mu * w) * s[j - 1][mu] - rhs[mu]) / (2 * j) for mu in ODD})
        sj = {}
        for mu in ODD:
            acc = (2 * j + 1 + I * mu * w) * d[j][mu]
            for k in range(1, j + 1):
                acc -= proj(g[k], s[j - k], ODD)[mu]
            sj[mu] = acc
        s.append(sj)
        ej = {nu: mp.mpc(0) for nu in EVEN}
        for a in range(j + 1):
            t = proj(d[a], d[j - a], EVEN)
            for nu in EVEN:
                ej[nu] += t[nu] / 2
        e.append(ej)
        gEj = {nu: mp.mpc(0) for nu in EVEN}
        for k in range(j + 1):
            t = proj(g[k], e[j - k], EVEN)
            for nu in EVEN:
                gEj[nu] += t[nu]
        gE.append(gEj)
    return d, s, g


def state(A, wf, xi):
    d, s, g = series(A, wf)
    z = xi * xi
    D = [sum(d[j][mu] * z ** j for j in range(N)) * xi for mu in ODD]
    S = [sum(s[j][mu] * z ** j for j in range(N)) * z for mu in ODD]
    C = [sum(g[j][nu] * z ** j for j in range(N)) for nu in EVEN]
    return D + S + C


def read_start(fn):
    rows = {}
    for ln in open(fn):
        p = ln.split("#")[0].split()
        if len(p) == 6 and p[0] in ("Y", "dA", "dw"):
            lo_hi = [float.fromhex(x) for x in p[2:]]
            rows.setdefault(p[0], {})[int(p[1])] = lo_hi
        elif len(p) == 10 and p[0] == "Y2":       # double-double bounds
            v = [mp.mpf(float.fromhex(x)) for x in p[2:]]
            rows.setdefault("Y2", {})[int(p[1])] = [v[0] + v[1], v[2] + v[3], v[4] + v[5], v[6] + v[7]]
    return rows


def dist(z, iv):
    """distance of the complex number z from the rectangle iv = (re_lo, re_hi, im_lo, im_hi), and its half-width"""
    rl, rh, il, ih = (mp.mpf(x) for x in iv)
    dr = max(rl - z.real, z.real - rh, 0)
    di = max(il - z.imag, z.imag - ih, 0)
    return max(dr, di), max(rh - rl, ih - il) / 2


if __name__ == "__main__":
    A, wf = mp.mpf("2.5421480475"), mp.mpf("0.7769940136")
    xi0 = mp.mpf("0.1") * mp.expjpi(mp.mpf(3) / 8)
    d, s, g = series(A, wf)
    print(f"|d_j| xi0^(2j): j = 0: {float(abs(d[0][1]) * 0.1):.2e}, j = {N - 1}: "
          f"{float(max(abs(d[N - 1][mu]) for mu in ODD) * mp.mpf('0.01') ** (N - 1) * 0.1):.2e}")
    Y = state(A, wf, xi0)
    eps = mp.mpf("1e-20")
    dYA = [(p - m) / (2 * eps) for p, m in zip(state(A + eps, wf, xi0), state(A - eps, wf, xi0))]
    dYw = [(p - m) / (2 * eps) for p, m in zip(state(A, wf + eps, xi0), state(A, wf - eps, xi0))]
    st = read_start(os.path.join(HERE, "data", "start_point.txt"))
    names = ["D-1", "D+1", "S-1", "S+1", "C-2", "C0", "C+2"]
    worst = {"Y": 0, "Y2": 0, "dA": 0, "dw": 0}
    for tag, vals in (("Y", Y), ("Y2", Y), ("dA", dYA), ("dw", dYw)):
        for k, v in enumerate(vals):
            dd, hw = dist(v, st[tag][k])
            worst[tag] = max(worst[tag], float(dd))
            print(f"{tag:2s} {names[k]:4s} series {mp.nstr(v.real, 17):>22s} {mp.nstr(v.imag, 17):>22s}i   "
                  f"outside the exported interval by {float(dd):.1e} (half-width {float(hw):.1e})")
    print("largest distance outside the exported intervals: values {Y:.1e} (double-double bounds {Y2:.1e}), "
          "d/dA {dA:.1e}, d/dwf {dw:.1e}".format(**worst))
    print("PASS" if max(worst.values()) == 0 else "MISMATCH")

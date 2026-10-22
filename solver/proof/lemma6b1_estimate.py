#!/usr/bin/env python3
"""Lemma 6b_1, route assessment (proof/A_certify.md section 12.1). Double precision (scipy), NOT rigorous.

1. y+ on the annulus 25 <= |u| <= 260 (u = 1/xi^2), from the formal series at |u| = 260 on the start ray, along the
   ray and the arcs: the sizes of P, V, Q = V + (1 + s) P, W = V + [(C - 1) Q], C - 1 (variables of lemma6_usys).
2. Route 1 (a start-value contraction built on the linear Kummer flow, nonlinear terms as the perturbation): the loop
   P -> C -> V -> P of the linearized nonlinear part, in the natural norms (|delta C| <= K/|u|, |delta V| <= K/|u|),
       g(rho) = max (|Q| + |W|)/2 * (sqrt 2 / a) * max |P| / (2 rho),   a = sqrt(2)/4,
   from C_u = -z^2 [C E]/2 (d/dP ~ z^2 |P|, integrated from infinity: |P|/|u|), the C-coupling -([(C-1)Q] + [(C-1)W])/2
   of V_u (d/dC ~ (Q + W)/2) through the fast kernel (1/a, and |t| >= |u|/sqrt 2 on its path), and the linear coupling
   P_u = -z V/2 (integrated from infinity: 1/(2|u|)). A contraction needs g < 1 at least.
3. The branch point of y+ (review R6): bisection on which side arcs and inward rays pass it.

    python3 lemma6b1_estimate.py
"""
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from flint import acb, arb, ctx  # noqa: E402
import validated_int as VI  # noqa: E402
import lemma6_usys as U  # noqa: E402

ctx.prec = 256
A = arb("2.5421480475")
WF = arb("0.7769940136")
W = VI.W0 * WF
WC = complex(W.mid())
KAP = U.KAPPA
ODD, EVEN = (-1, 1), (-2, 0, 2)
E3 = np.exp(-3j * np.pi / 4)


def prn(X, Y, out):
    return {k: sum(X[a] * Y[k - a] for a in X if (k - a) in Y) for k in out}


def parts(u, y):
    P = {-1: y[0], 1: y[1]}
    V = {-1: y[2], 1: y[3]}
    Cm1 = {n: y[4 + i] / KAP for i, n in enumerate(EVEN)}
    C = {n: Cm1[n] + (1 if n == 0 else 0) for n in EVEN}
    s = {m: 1j * m * WC for m in ODD}
    Q = {m: V[m] + (1 + s[m]) * P[m] for m in ODD}
    CQ = prn(Cm1, Q, ODD)
    Wd = {m: V[m] + CQ[m] for m in ODD}
    return P, V, C, Cm1, s, Q, CQ, Wd


def F(u, y):
    P, V, C, Cm1, s, Q, CQ, Wd = parts(u, y)
    z = 1 / u
    CW = prn(Cm1, Wd, ODD)
    Eh = {k: v / 2 for k, v in prn(P, P, EVEN).items()}
    CE = prn(C, Eh, EVEN)
    CEP = prn(CE, P, ODD)
    dP = [-z * (V[m] + CQ[m]) / 2 for m in ODD]
    dV = [-V[m] / 2 - (CQ[m] + CW[m]) / 2 - z * (-(2 + s[m]) * Q[m] - (1 + s[m]) * Wd[m] + CEP[m]) / 2 for m in ODD]
    dC = [-z * z * CE[n] / 2 * KAP for n in EVEN]
    return np.array(dP + dV + dC)


def start(rho=260):
    xi0 = acb(arb(rho) ** arb(-0.5)) * (VI.I * VI.PHI0).exp()
    r0 = (1 / arb(rho)).sqrt()
    n = VI.optimal_n(A, W, r0)
    Y = VI.series_value(acb(A), acb(W), r0, n)
    return np.array([complex(p.mid()) for p in U.dsc_to_pvc(Y, xi0, acb(W))])


def integrate(path, y0, t0, t1):
    def f(t, yr):
        u, du = path(t)
        d = F(u, yr[:7] + 1j * yr[7:]) * du
        return np.concatenate([d.real, d.imag])
    return solve_ivp(f, (t0, t1), np.concatenate([y0.real, y0.imag]), method="DOP853", rtol=1e-12, atol=1e-14,
                     dense_output=True)


def ray(t):
    return t * E3, E3


def arc(rho):
    return lambda t: (rho * np.exp(1j * t), 1j * rho * np.exp(1j * t))


def val(sol, t):
    y = sol.sol(t)
    return y[:7] + 1j * y[7:]


def main():
    sr = integrate(ray, start(260), 260, 5)
    print("1-2. y+ on arcs |u| = rho (max over the arc), and the route-1 loop gain g (natural norms):")
    a = 2 ** 0.5 / 4
    for rho in (260, 150, 100, 60, 40, 25):
        sa = integrate(arc(rho), val(sr, rho), -3 * np.pi / 4, 0)
        mP = mV = mQ = mQW = mC = 0
        for t in np.linspace(-3 * np.pi / 4, 0, 61):
            P, V, C, Cm1, s, Q, CQ, Wd = parts(rho * np.exp(1j * t), val(sa, t))
            mP = max(mP, max(abs(P[m]) for m in ODD))
            mV = max(mV, max(abs(V[m]) for m in ODD))
            mQ = max(mQ, max(abs(Q[m]) for m in ODD))
            mQW = max(mQW, max((abs(Q[m]) + abs(Wd[m])) / 2 for m in ODD))
            mC = max(mC, abs(Cm1[0]))
        g = mQW * (2 ** 0.5 / a) * mP / (2 * rho)
        print(f"  |u| = {rho:3d} (|xi| = {rho ** -0.5:.4f}): max|P| {mP:.2f}, max|V| {mV:.2f}, max|Q| {mQ:.2f}, "
              f"max|C_0 - 1| {mC:.3f};  g = {g:.2f}")

    def side(rho):
        return val(integrate(arc(rho), val(sr, rho), -3 * np.pi / 4, 0.0), 0.0)[1].real > 0
    lo, hi = 9.5, 10.0
    for _ in range(30):
        mid = (lo + hi) / 2
        hi, lo = (mid, lo) if side(mid) else (hi, mid)
    rs = (lo + hi) / 2
    s25 = integrate(arc(25.0), val(sr, 25.0), -3 * np.pi / 4, 0.0)

    def ray_end(th):
        e = np.exp(1j * th)
        return val(integrate(lambda t: (t * e, e), val(s25, th), 25.0, 5.0), 5.0)[1]
    ta, tb = -2.3, -2.1
    va, vb = ray_end(ta), ray_end(tb)
    for _ in range(30):
        m = (ta + tb) / 2
        vm = ray_end(m)
        ta, tb = (m, tb) if abs(vm - va) < abs(vm - vb) else (ta, m)
    th = (ta + tb) / 2
    us = rs * np.exp(1j * th)
    print(f"3. branch point of y+: u* = {rs:.4f} e^({th:.4f} i), xi* = {rs ** -0.5:.4f} e^({-th / 2:.4f} i); distance "
          f"from 14 e^(-3 i pi/4): {abs(us - 14 * E3):.3f}")


if __name__ == "__main__":
    main()

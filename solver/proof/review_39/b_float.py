#!/usr/bin/env python3
"""A double-precision control of lemma 6a's stored backward enclosure B (#39 review; NOT rigorous).

B encloses d(state at xi_a)/d(state at xi_1) of the flow along the real axis, in the variables [D_mu, S_mu, C_nu]. Here
the same flow is integrated by a second route: the (P, V, C) system in u = 1/xi^2 (proof/A_lemmas_M.md section 1,
transcribed again in numpy, with forward-mode derivatives), from u_1 = 1/0.3^2 to u_a = 1/0.067^2 with DOP853, the
variational matrix alongside, then the chain rule of D = xi P, S = xi^2 (V + (1 + s) P), C = 1 + C~/kappa at both
ends. The start is the midpoint of the stored y+(xi_1) box. Reported: the largest |B_float - mid(B)| over B's radius,
with w from the stored zero x^ (what the run used) and from the certificate's centre p^.

    python3 b_float.py --B=B.json --box=A_M2_6a.json [--mutate=w]
        --mutate=w: wf + 1e-6 in both runs, a control that must put B_float outside B
"""
import json
import sys

import numpy as np
from scipy.integrate import solve_ivp

M = 5
ODD = [m for m in range(-M, M + 1) if m % 2]
EVEN = [n for n in range(-(M + 1), M + 2) if n % 2 == 0]
NO, NE = len(ODD), len(EVEN)
ND = 2 * NO + NE
KAPPA = 100.0
W0 = 2 * np.pi / 1.64389375


def pr(X, Y, out):
    """[X Y] on the harmonics out; elements are dual arrays [value, d/dy_1 .. d/dy_ND]"""
    r = {}
    for k in out:
        acc = np.zeros(ND + 1, complex)
        for a, xa in X.items():
            yb = Y.get(k - a)
            if yb is not None:
                acc = acc + xa[0] * yb
                acc[1:] += yb[0] * xa[1:]
        r[k] = acc
    return r


def rhs_dual(y, u, w, sign=-0.5):
    """F(u, y) and its Jacobian in y, state [P_mu, V_mu, kappa (C_nu - delta)]"""
    z = 1.0 / u
    Y = []
    for i in range(ND):
        v = np.zeros(ND + 1, complex)
        v[0], v[1 + i] = y[i], 1.0
        Y.append(v)
    P = {m: Y[i] for i, m in enumerate(ODD)}
    V = {m: Y[NO + i] for i, m in enumerate(ODD)}
    Cm1 = {n: Y[2 * NO + i] / KAPPA for i, n in enumerate(EVEN)}
    Cf = {n: Cm1[n] + (np.eye(ND + 1)[0] if n == 0 else 0) for n in EVEN}
    s = {m: 1j * m * w for m in ODD}
    Q = {m: V[m] + (1 + s[m]) * P[m] for m in ODD}
    CQ = pr(Cm1, Q, ODD)
    W = {m: V[m] + CQ[m] for m in ODD}
    CW = pr(Cm1, W, ODD)
    E = {k: v / 2 for k, v in pr(P, P, EVEN).items()}
    CE = pr(Cf, E, EVEN)
    CEP = pr(CE, P, ODD)
    dP = [-z * W[m] / 2 for m in ODD]
    dV = [sign * V[m] - (CQ[m] + CW[m]) / 2 - z * (-(2 + s[m]) * Q[m] - (1 + s[m]) * W[m] + CEP[m]) / 2 for m in ODD]
    dC = [-KAPPA * z * z * CE[n] / 2 for n in EVEN]
    Fd = np.array(dP + dV + dC)
    return Fd[:, 0], Fd[:, 1:]


def dsc_to_pvc(Y, xi, w):
    D, S, Cc = Y[:NO], Y[NO:2 * NO], Y[2 * NO:]
    P = D / xi
    V = S / xi ** 2 - (1 + 1j * np.array(ODD) * w) * P
    Ct = KAPPA * (Cc - np.array([1.0 if n == 0 else 0.0 for n in EVEN]))
    return np.concatenate([P, V, Ct])


def jac_pvc_to_dsc(xi, w):
    """d(D, S, C)/d(P, V, C~) at fixed xi"""
    J = np.zeros((ND, ND), complex)
    for r, m in enumerate(ODD):
        J[r, r] = xi
        J[NO + r, NO + r] = xi ** 2
        J[NO + r, r] = xi ** 2 * (1 + 1j * m * w)
    for r in range(NE):
        J[2 * NO + r, 2 * NO + r] = 1 / KAPPA
    return J


def flow_B(y1_dsc, xi1, xia, w, sign=-0.5):
    y1 = dsc_to_pvc(y1_dsc, xi1, w)

    def fun(u, Y):
        y = Y[:ND]
        Phi = Y[ND:].reshape(ND, ND)
        F, J = rhs_dual(y, u, w, sign)
        return np.concatenate([F, (J @ Phi).ravel()])
    Y0 = np.concatenate([y1, np.eye(ND, dtype=complex).ravel()])
    sol = solve_ivp(fun, (1 / xi1 ** 2, 1 / xia ** 2), Y0, method="DOP853", rtol=1e-12, atol=1e-14)
    Phi = sol.y[ND:, -1].reshape(ND, ND)
    Jin_inv = np.linalg.inv(jac_pvc_to_dsc(xi1, w))
    return jac_pvc_to_dsc(xia, w) @ Phi @ Jin_inv, sol.t.size


def mid(p):
    return complex(float(p[0][0]), float(p[1][0]))


def rad(p):
    return max(float(p[0][1]), float(p[1][1]))


def main():
    opts = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
    with open(opts["B"]) as fh:
        sb = json.load(fh)
    with open(opts["box"]) as fh:
        bx = json.load(fh)
    xi1, xia = float(sb["xi1"]), float(sb["xia"])
    Bm = np.array([[mid(sb["B"][i][j]) for j in range(ND)] for i in range(ND)])
    Br = np.array([[rad(sb["B"][i][j]) for j in range(ND)] for i in range(ND)])

    def pmid(s):
        a, b = s.strip("[]").split("+/-")
        return float(a)
    y1 = np.array([complex(pmid(p[0]), pmid(p[1])) for p in bx["yX_xi1"]])
    sign = -0.5
    dw = 1e-6 if opts.get("mutate") == "w" else 0.0          # MUTATION w: wf + 1e-6
    res = {}
    for name, wf in (("xhat 0.999228404", 0.999228404), ("phat " + bx["p"][1][:20], float(bx["p"][1]))):
        B, nst = flow_B(y1, xi1, xia, W0 * (wf + dw), sign)
        dev = np.abs(B - Bm) / Br
        res[name] = B
        print(f"w from {name}: {nst} DOP853 points; max |B_float - mid(B)| / rad(B) = {dev.max():.3g} "
              f"(entry {np.unravel_index(dev.argmax(), dev.shape)}), max |B_float - mid(B)| = "
              f"{np.abs(B - Bm).max():.3g}, max rad(B) = {Br.max():.3g}"
              + ("   [MUTATION w: must exceed 1]" if dw else "   (inside B: must be below 1)"))
    a, b = res.values()
    print(f"B_float(p^) - B_float(x^): max {np.abs(a - b).max():.3g}, max over rad(B) {(np.abs(a - b) / Br).max():.3g}")
    worst = max((np.abs(B - Bm) / Br).max() for B in res.values())
    if dw:
        return 0 if min((np.abs(B - Bm) / Br).max() for B in res.values()) > 1 else 3
    return 0 if worst < 1 else 3


if __name__ == "__main__":
    sys.exit(main())

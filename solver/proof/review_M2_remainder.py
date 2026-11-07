#!/usr/bin/env python3
"""Step-halving control for the remainders of the C^2 step (review of proof/A_M2.md, proof/A_M2_review.md).

The C^2 self-test and the control D2 compare midpoints with finite differences. Neither can see a remainder term that
is missing or too small: a remainder moves the midpoint by far less than the radii of the finite differences. This
check can. From one C^2 set it takes one LohnerM2 step of size h, and K steps of size h / K, and requires every
enclosure of the two results to overlap (both contain the true flow if the step is valid): the centre box, V0 and
W. With the Taylor orders lowered so that the truncation error of the big step is far above the rounding, a step whose
remainder is dropped no longer contains the true flow, and the overlap fails.

Mutations (applied to every step, by wrapping functions of certify_M2_lohner; the file itself is not changed):
  psiR   the Lagrange remainder of H (the second variation) dropped: the coefficient NJ + 1 of the remainder series
         set to 0 (the call psi_coeffs(..., NJ + 1))
  remJ   the remainder of the step Jacobian dropped (the coefficient NJ + 1 of taylor_VA's remainder call set to 0)
  both   the two together

    python3 review_M2_remainder.py [--mutate=psiR|remJ|both] [--h=2^-8] [--K=8] [--NJ=8] [--M=1|3]
Exit status 0 if every enclosure overlaps, 1 otherwise.
"""
import sys

import numpy as np
from flint import acb, arb, acb_mat, ctx

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import certify_M_sys as CS  # noqa: E402
import certify_M2_lohner as C2  # noqa: E402

MUT = set()
NJ_RUN = [None]
_psi, _tva = C2.psi_coeffs, C2.taylor_VA


def psi_wrap(Ak, beta, Psi0, K):
    out = _psi(Ak, beta, Psi0, K)
    if "psiR" in MUT and K == NJ_RUN[0] + 1:
        out[K] = acb_mat(out[K].nrows(), out[K].ncols())
    return out


def tva_wrap(s, ctxv, N):
    V, A = _tva(s, ctxv, N)
    if "remJ" in MUT and N == NJ_RUN[0] + 1:
        V[N] = acb_mat(V[N].nrows(), V[N].ncols())
    return V, A


C2.psi_coeffs, C2.taylor_VA = psi_wrap, tva_wrap


def start(M, NJ, box):
    rng = np.random.default_rng(7)
    ctx.prec = 256
    s = CS.SysM(M)
    if M == 1:
        phi0, r0 = 3 * arb.pi() / 8, "0.1"
        p = [arb("2.5421480475"), arb("0.7769940136")]
    else:
        phi0, r0 = arb(5) / 16 * arb.pi(), "0.0741"
        p = [arb("2.72647588563088"), arb("1.0156030948577"), arb("0.144419478470077"), arb("0.0110874258881167")]
    leg = CS.LegM("ray", r0, "0.2", phi0)
    m = M + 1
    xi0, vals, ders = CS.start_series(p, M, r0, phi0, 60 if M == 3 else 40)
    W0 = CS.W0()
    c = [acb(v.mid()) for v in vals]
    cols = [[acb(v.mid()) for v in ders[k]] for k in range(m)]
    wcol = [W0 if k == 1 else arb(0) for k in range(m)]
    P = m * (m + 1) // 2
    Wc = [[acb(complex(*(1e-2 * rng.normal(size=2)))) for _ in range(s.nd)] for _ in range(P)]
    pdx = [acb(arb(0, box)) for _ in range(m)]
    N = 24 if M == 1 else 20
    lo = C2.start_set2(s, c, W0 * p[1], cols, wcol, Wc, pdx, N, NJ)
    return s, leg, lo


def sets(lo):
    X = lo.box()
    return {"centre": [X[i] for i in range(lo.s.nd)],
            "V0": [lo.var_box()[i, j] for i in range(lo.s.nd) for j in range(lo.m)],
            "W": [lo.W_box()[i, p] for i in range(lo.s.nd) for p in range(len(lo.pairs))]}


def sep(a, b):
    """largest |mid a - mid b| / (rad a + rad b) over real and imaginary parts (overlap iff <= 1)"""
    w = 0.0
    for x, y in ((a.real, b.real), (a.imag, b.imag)):
        d = abs(x.mid() - y.mid())
        r = x.rad() + y.rad()
        w = max(w, float((d / r).mid()) if r > 0 else (0.0 if d == 0 else float("inf")))
    return w


def main():
    o = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else True) for a in sys.argv[1:]
         if a.startswith("--")}
    for mu in (o.get("mutate") or "").split(","):
        if mu == "both":
            MUT.update({"psiR", "remJ"})
        elif mu:
            MUT.add(mu)
    M = int(o.get("M", "1"))
    NJ = int(o.get("NJ", "8"))
    K = int(o.get("K", "8"))
    hstr = o.get("h", "2^-8")
    h = 2.0 ** int(hstr.split("^")[1]) if "^" in hstr else float(hstr)
    box = float(o.get("box", "1e-12"))
    NJ_RUN[0] = NJ
    s, leg, lo1 = start(M, NJ, box)
    _, _, loK = start(M, NJ, box)
    ok1 = lo1.step(leg, 0.0, h)
    okK = all(loK.step(leg, k * h / K, h / K) for k in range(K))
    if not (ok1 and okK):
        print(f"a step failed (big {ok1}, small {okK}): choose a smaller h")
        return 2
    A, B = sets(lo1), sets(loK)
    allok = True
    print(f"M = {M}, NJ = {NJ}, one step h = {h:.3e} against {K} steps h / {K}, box half-width {box:g}; "
          f"mutations {sorted(MUT) or 'none'}")
    for key in ("centre", "V0", "W"):
        ws = [sep(a, b) for a, b in zip(A[key], B[key])]
        ra = max(max(float(z.real.rad()), float(z.imag.rad())) for z in A[key])
        rb = max(max(float(z.real.rad()), float(z.imag.rad())) for z in B[key])
        ok = max(ws) <= 1
        allok &= ok
        print(f"   {key:6s}: overlap {ok}; largest |mid difference| / (sum of radii) {max(ws):.2e}; radii: one step "
              f"{ra:.2e}, {K} steps {rb:.2e}")
    print("OVERLAP " + ("ALL" if allok else "FAILS"))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())

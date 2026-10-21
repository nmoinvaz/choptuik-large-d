#!/usr/bin/env python3
"""Layer A of PROOF.md, prototype for the one-harmonic (M = 1) inner problem: what is certified so far.

  1. RIGOROUS  linear (no-gravity) Stokes normalization: the jump of the two lateral solutions of the linear inner
     problem equals 2 pi i K'(s) f(xi1) 2F_1 exactly (the normalization lateral_np.py divides by), as a ball, from arb's
     rigorous Kummer U on both sides of its cut (lin_check.py).
  2. RIGOROUS  formal series of the truncated M = 1 system in ball arithmetic (series_arb.py; A, omega may be boxes) and
     the residual polynomial of its optimal truncation, with a rigorous bound on |xi| <= r0 (residual_poly.py): this is
     the forcing M_0 of the fixed-point lemma that will enclose the lateral solution at the start point.
  3. NOT RIGOROUS (dry run)  Poincare-Miranda test on a preconditioned box around the zero with the double-precision
     lateral solver (pm_dryrun.py): the enclosure accuracy the certified jump must reach.
Missing for a certificate: the contraction constant of the fixed-point lemma (step i) and validated integration of the
lateral solution from the start point to the real axis (step ii). See proof/A_certify.md.

    python3 certify_M1.py [--pm]
"""
import sys
from flint import arb, ctx
import lin_check
import residual_poly as RP
from series_arb import SeriesArb
from flint import acb


def main():
    ctx.prec = 256
    W0 = 2 * arb.pi() / arb("1.64389375")
    print("1. linear Stokes normalization (rigorous):")
    for xi1 in (arb(1) / 5, arb(3) / 10):
        for wf in (arb(1), arb("0.7769940136")):
            *_, J, ref = lin_check.jump_exact(W0 * wf, xi1)
            print(f"   xi1 {float(xi1.mid()):.2f}, w/w_inf {float(wf.mid()):.4f}: J / (2 pi i K' f 2F_1) = {(J / ref).str(20)}")
    print("2. truncated formal series and its residual (rigorous), M = 1 at the zero:")
    ctx.prec = 320
    A, wf, r0 = arb("2.5421480475"), arb("0.7769940136"), "0.1"
    sa = SeriesArb({1: acb(0, -A / 2), -1: acb(0, A / 2)}, W0 * wf, prec=320)
    J = int(1 / (2 * float(r0) ** 2)) + 30
    d, s, g = sa.series(J)
    x2 = arb(r0) ** 2
    mags = [max(abs(d[j][mu]) for mu in sa.odd) * x2 ** j for j in range(J)]
    n = min(range(1, J), key=lambda j: float(mags[j].mid()))
    r1, r2, r3 = RP.residual_poly(sa, d, s, g, n)
    worst = max((RP.weighted_bound(p, r0)[1] for dct in (r1, r2, r3) for p in dct.values()), key=lambda a: float(a.upper()))
    kmin = min(RP.weighted_bound(p, r0)[0] for dct in (r1, r2, r3) for p in dct.values())
    print(f"   N = {n} terms; |R(xi)| <= sum_k |rho_k| |xi|^k <= {float(worst.upper()):.2e} on |xi| <= {r0}; "
          f"nonzero coefficients from xi^{kmin}")
    if "--pm" in sys.argv:
        print("3. Poincare-Miranda dry run (double precision, not rigorous):")
        import pm_dryrun
        pm_dryrun.main()


if __name__ == "__main__":
    main()

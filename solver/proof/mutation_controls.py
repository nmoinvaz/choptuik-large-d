#!/usr/bin/env python3
"""Mutation controls for the M = 1 Krawczyk test (proof/A_certify.md §9.5).

The certificate's last step (krawczyk.py, assembled by validated_int.py) has so far only been seen to pass. A check
that has never been seen to fail is untested (the "planted-truth" discipline, notes/BOOTLOOPS.md): here the function
itself is mutated, the full assembly is rerun from the saved certificate runs (validated_int_krawczyk_1e-9.json), and
each verdict is compared with the one written down below before the controls were first run.

The controls, with their expected verdicts (frozen):

    C0  clean twin: the unmodified assembly                                   PASS, the recorded margins
    C1  jump sign flipped: J = D+_{+1} + conj(D+_{-1})                         FAIL (no zero near x^)
    C2  conjugation dropped: J = D+_{+1} - D+_{-1}                             FAIL (not the jump)
    C3  equivalent form: J = D+_{-1} - conj(D+_{+1}) = -conj(J_D)              PASS, same zero (K overlaps C0's K)
    C4  normalization scaled by 2 (N -> 2N)                                    PASS, K identical to C0's
    C5  planted zero moved out of the box: g -> g - L d, d = (2h, 0)           FAIL
    C6  planted zero moved inside the box: g -> g - L d, d = (h/2, -3h/10)     PASS, K contains C0's zero + d and
                                                                              excludes C0's zero + d + (h/4, 0)
    C7  point enclosure degraded: radius of D+_{+1}(xi1) widened by 1e-9       FAIL (enclosure too wide)

L is the exact (binary) midpoint of J(X), so g - L d is a function whose zero is moved by d up to the second-order
remainder; J(X) is unchanged because L d is constant. C5 and C6 are planted truths: the answer is known before the run.
C3 and C4 are nulls: reformulations with the same zero set must give the same answer. A Jacobian-only mutation is not a
meaningful control (a wrong J(X) can make the test pass or fail arbitrarily, since its soundness assumes J(X) encloses
the true derivative), so every control here changes the function or its enclosure.

    python3 mutation_controls.py        # seconds; exit status 0 only if every verdict matches its expectation
"""
import json
import os
import sys

from flint import acb, arb, ctx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import krawczyk as KR  # noqa: E402

RUNS = os.path.join(HERE, "validated_int_krawczyk_1e-9.json")


def parse_acb(p):
    return acb(arb(p[0]), arb(p[1]))


def assemble(d, jump=None, nscale=1, plant=None, widen=None):
    """K(X) with an optional mutation. jump(Dp1, Dm1) -> J; nscale multiplies N; plant = (dA, dwf) moves the zero by
    subtracting L d (L the exact midpoint of J(X)); widen adds a radius to D+_{+1}(xi1)."""
    ctx.prec = 320
    As, wfs, h, r1 = d["A"], d["wf"], d["h"], d["xi1"]
    xhat = [arb(As), arb(wfs)]
    Dm1, Dp1 = parse_acb(d["point"]["Dm1"]), parse_acb(d["point"]["Dp1"])
    dDm1 = [parse_acb(p) for p in d["box"]["dDm1"]]
    dDp1 = [parse_acb(p) for p in d["box"]["dDp1"]]
    if widen is not None:
        Dp1 = acb(arb(Dp1.real.mid(), Dp1.real.rad() + widen), arb(Dp1.imag.mid(), Dp1.imag.rad() + widen))
    N, xs = KR.normalization(xhat, r1, prec=320)
    N = N * nscale
    c = xs / N
    jf = jump or (lambda P, M: P - M.conjugate())
    gv = jf(Dp1, Dm1) * c
    # derivative of the jump: jf is linear in (P, conj-able M) for every mutation used, so apply it to the columns
    dJ = [jf(dDp1[k], dDm1[k]) * c for k in range(2)]
    g_hat = [gv.real, gv.imag]
    JX = [[dJ[0].real, dJ[1].real], [dJ[0].imag, dJ[1].imag]]
    if plant is not None:
        L = [[arb(JX[i][j].mid()) for j in range(2)] for i in range(2)]
        dv = [arb(plant[0]), arb(plant[1])]
        g_hat = [g_hat[i] - (L[i][0] * dv[0] + L[i][1] * dv[1]) for i in range(2)]
    ok, K, margins = KR.krawczyk(xhat, g_hat, JX, [arb(h), arb(h)])
    return ok, K, margins


def main():
    d = json.load(open(RUNS))
    h = float(d["h"])
    ok0, K0, m0 = assemble(d)
    dplant = (h / 2, -3 * h / 10)
    controls = [
        ("C0 clean twin", dict(), True),
        ("C1 jump sign flipped", dict(jump=lambda P, M: P + M.conjugate()), False),
        ("C2 conjugation dropped", dict(jump=lambda P, M: P - M), False),
        ("C3 equivalent form -conj(J)", dict(jump=lambda P, M: M - P.conjugate()), True),
        ("C4 normalization x2", dict(nscale=2), True),
        ("C5 zero planted outside (2h, 0)", dict(plant=(2 * h, 0.0)), False),
        ("C6 zero planted inside (h/2, -3h/10)", dict(plant=dplant), True),
        ("C7 point enclosure widened by 1e-9", dict(widen=1e-9), False),
    ]
    declared = 8
    bad = 0
    for name, kw, expect in controls:
        ok, K, margins = assemble(d, **kw)
        extra = ""
        good = ok == expect
        if name.startswith("C3"):
            same = all(K[i].overlaps(K0[i]) for i in range(2))
            extra = f"; K overlaps C0's K: {same}"
            good = good and same
        if name.startswith("C4"):
            same = all(K[i].str(30, radius=True) == K0[i].str(30, radius=True) for i in range(2))
            extra = f"; K identical to C0's to 30 digits: {same}"
            good = good and same
        if name.startswith("C6"):
            moved = [K0[i] + arb(dplant[i]) for i in range(2)]
            rec = all(K[i].overlaps(moved[i]) for i in range(2))
            wrong = [moved[0] + arb(h / 4), moved[1]]               # the recovery check must reject a wrong answer
            rej = not all(K[i].overlaps(wrong[i]) for i in range(2))
            extra = f"; K overlaps C0's zero + d: {rec}; rejects zero + d + (h/4, 0): {rej}"
            good = good and rec and rej
        bad += not good
        print(f"{name:40s} test {'PASS' if ok else 'FAIL'} (expected {'PASS' if expect else 'FAIL'}); "
              f"margins {margins[0]:+.3e}, {margins[1]:+.3e}{extra}  -> {'as expected' if good else 'UNEXPECTED'}")
    n = len(controls)
    assert n == declared, f"ran {n} controls, declared {declared}"
    print(f"{n - bad} of {n} controls behaved as expected (box half-width h = {d['h']}, xi1 = {d['xi1']})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

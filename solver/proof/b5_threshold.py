#!/usr/bin/env python3
"""Lemma B5(1) (proof/B5_inner.md, sections 5.2 and 7): numbers derived from the saved certificate
b5_bounds_k0.3_Z0.0085.json and the leading data of b5_data.py (b5_data.log).

1. The bound x for the leading data delta0 = (0, 76.56, 3.23125, 6.46251) of (H8). Super-solutions add: if
   b_i + L x_i < x_i for each right-hand side, then sum_i c_i b_i + L (sum c_i x_i) < sum c_i x_i for c_i >= 0, so
   x = x^forcing + sum_B delta_B x^(B) is a super-solution for b = forcing + delta (each x^(B) certified in
   b5_bounds.py). The sums are taken in arb (upper bounds), per harmonic, then maximized over harmonics.

2. An ESTIMATE, not certified, of the threshold of the chain B5(1) + B5(2) for M = 1 (section 7). SUPERSEDED by
   b5_jump.py (proof/B_chain.md section 3): the model below uses the exponent <F^2> for the fast mode, where the
   Phi-component's fast mode is e^{-1/(2z)} z^G with Re G = <F^2> - 3/2, and it omits the weight m_Phi = |z|. The relative error of
   the jump at u* = kappa1/ln n is modelled as
        x_Phi n^{-1 + 1/(2 kappa1)} (ln n / kappa1)^{<F^2>},   <F^2> = A^2/2,
   and required to be <= MARGIN (default 1e-7, the size asked for by the implicit function theorem on the certified box
   with the Cauchy factor 1e6). For each kappa1 the smallest ln n with this is found by bisection; the chain needs
   ln n >= max(that, kappa1/(Z cos kappa)) (the second is B5(1)'s domain condition). Scanned over kappa1.

Check of 1.: the same superposition for the reference data delta_ref = (1, 200, 10, 20) must reproduce, to 2 % in every
harmonic, the super-solution that b5_bounds.py certified directly for delta_ref (the solves differ only by their margins).

    python3 b5_threshold.py [--margin=1e-7]
    python3 b5_threshold.py --mutate=sum      drop the forcing term from the superposition: the check must fail
"""
import json
import math
import os
import sys

from flint import arb

HERE = os.path.dirname(os.path.abspath(__file__))
OPTS = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
MUT = OPTS.get("mutate")
A_STAR = "2.5421480475"                      # b5_numeric.A_STAR (the centre of Theorem A(M = 1)'s box)
DELTA0 = {"P": "0", "F": "76.56", "C": "3.23125", "v": "6.46251"}     # b5_data.log (leading data, upper values)


def superpose(ss, delta):
    x = {}
    for c in ss["forcing"]["x"]:
        s = arb(0) if MUT == "sum" else arb(ss["forcing"]["x"][c])
        for blk in "PFCv":
            s += arb(str(delta[blk])) * arb(ss[f"unit_{blk}"]["x"][c])
        x[c] = float(s.upper())
    return x


def leading_data_bound(ref):
    ss = ref["supersolution"]
    x = superpose(ss, DELTA0)
    blocks = {blk: max(v for c, v in x.items() if c[0] == blk) for blk in "PFCv"}
    # check: the superposition for delta_ref vs the super-solution certified directly for delta_ref
    xr = superpose(ss, ref["delta_ref"])
    dev = max(abs(xr[c] - v) / v for c, v in ss["data"]["x"].items())
    return blocks, dev


def ln_n_b52(kappa1, xphi, f2, margin):
    """smallest L = ln n with ln xphi - (1 - 1/(2 kappa1)) L + f2 ln(L/kappa1) <= ln margin (bisection; the left side
    decreases in L once L > f2/(1 - 1/(2 kappa1)))"""
    a = 1 - 1 / (2 * kappa1)
    g = lambda L: math.log(xphi) - a * L + f2 * math.log(L / kappa1) - math.log(margin)
    lo = max(f2 / a, kappa1) * 1.0001
    hi = lo
    while g(hi) > 0:
        hi *= 2
    if g(lo) <= 0:
        return lo
    for _ in range(200):
        mid = (lo + hi) / 2
        if g(mid) > 0:
            lo = mid
        else:
            hi = mid
    return hi


if __name__ == "__main__":
    with open(os.path.join(HERE, "b5_bounds_k0.3_Z0.0085.json")) as fh:
        ref = json.load(fh)[0]
    blocks, dev = leading_data_bound(ref)
    ok = dev < 0.02
    print("1. leading data delta0 = (0, 76.56, 3.23125, 6.46251): x = forcing + sum delta_B x^(B), max over harmonics: "
          + ", ".join(f"x_{b} {v:.4g}" for b, v in blocks.items())
          + (f"   [MUTATION {MUT}]" if MUT else ""))
    print(f"   check: superposition for delta_ref vs the directly certified x(delta_ref), worst relative difference "
          f"{dev:.2e} (< 0.02): {'PASS' if ok else 'FAIL'}")
    margin = float(OPTS.get("margin", "1e-7"))
    Z, kap = ref["Z"], ref["kappa"]
    k1ref = 0.6
    for Zc, lab in ((Z, "reference run"), (0.009, "Z = 0.009, certified in b5_bounds_scan.log")):
        L0 = k1ref / (Zc * math.cos(kap))
        print(f"   n0 from B5(1)'s domain condition (kappa1 = {k1ref}, {lab}): ln n0 = {L0:.4f}, "
              f"n0 = {math.exp(L0):.3g} = 10^{L0 / math.log(10):.2f}")
    f2 = float(arb(A_STAR) ** 2 / 2)
    xd = max(ref["supersolution"]["data"]["x"][c] for c in ("F-1", "F+1"))
    print(f"2. ESTIMATE (not certified): jump error x_Phi n^(-1+1/(2 kappa1)) (ln n/kappa1)^<F^2> <= {margin:g}, "
          f"<F^2> = A^2/2 = {f2:.5f}, Z = {Z}, kappa = {kap}")
    best = {}
    for tag, xphi in (("delta_ref", xd), ("leading data", blocks["F"])):
        print(f"   x_Phi = {xphi:.4g} ({tag}):")
        rows = []
        for k1 in [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2]:
            Lb = ln_n_b52(k1, xphi, f2, margin)
            Ld = k1 / (Z * math.cos(kap))
            rows.append((max(Lb, Ld), k1, Lb, Ld))
            print(f"     kappa1 = {k1:4.2f}: B5(2) needs ln n >= {Lb:7.2f} [(1 - 1/(2 kappa1)) ln n = "
                  f"{(1 - 1 / (2 * k1)) * Lb:6.2f}], B5(1) needs ln n >= {Ld:7.2f};  chain: ln n >= {max(Lb, Ld):7.2f}")
        best[tag] = min(rows)
        print(f"     best on this grid: kappa1 = {best[tag][1]}, ln n >= {best[tag][0]:.2f}")
    if MUT:
        print(f"MUTATION {MUT}: the check must FAIL: {'caught' if not ok else 'NOT caught'}")
        sys.exit(0 if not ok else 3)
    sys.exit(0 if ok else 1)

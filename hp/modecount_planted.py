#!/usr/bin/env python3
"""Planted-truth test of the argument-principle count (hp/modecount_wind.py, function count) on a synthetic det M.

    python3 modecount_planted.py            (exits 1 if any case fails; about 10 s)

The synthetic determinant has the structure found for the DSS problem at D = 30 (omega, lam_g, lam_phys, the rows of
horizon poles on Im = +-omega, +-3 omega, +-5 omega, exp(57 lam + 0.3 lam^2) growth, copies 2 i k omega of the zeros),
and each case plants extra zeros at known places. The count runs through the same code path as on the real det M
(same geometry, trend, factors, sampling and bookkeeping), with an evaluation cache in a temporary directory. Pass
criteria, fixed before running: in every box, wind(g) equals the number of planted extra zeros inside it (counting the
conjugates for the boxes symmetric about the real axis), and every phase step is at most pi/4.
"""
import cmath
import math
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import modecount_wind as W  # noqa: E402

OM, LG, LP, N = 3.1987, 0.8265, 2.0797, 28.07
BASE_ZEROS = [z + 2j * k * OM for z in (0.0, LP, LG) for k in range(-4, 5)]
POLES = [N / 2 - LG * (kk + 1) + s * 1j * OM for kk in range(101) for s in (1, -1, 3, -3, 5, -5)]
CASES = {
    "none": [],
    "B1c, complex pair": [4.0 + 0.4j],
    "B1c, real": [3.3 + 0j],
    "U, other-sector real mode": [1.9 + 1j * OM],
    "U, near the top edge": [5.0 + 4.6j],
    "B0, slowly growing pair": [0.07 + 0.5j],
    "B1b, second mode next to lam_phys": [2.4 + 0.2j],
    "B1a, near an edge (0.05 from Re = aL)": [0.35 + 1.0j],
}


def det_for(extra):
    zeros = BASE_ZEROS + extra + [z.conjugate() for z in extra if z.imag != 0]

    def det(re, im):
        lam = complex(re, im)
        v = 57 * lam - 700 + 0.3 * lam * lam
        v += sum(cmath.log(lam - z) for z in zeros)
        v -= sum(cmath.log(lam - p) for p in POLES)
        return v.real, math.remainder(v.imag, 2 * math.pi), 0.0, 0.0
    return det


def expected(rect, extra):
    r0, r1, i0, i1 = rect
    pts = extra + [z.conjugate() for z in extra if z.imag != 0]
    return sum(1 for z in pts if r0 < z.real < r1 and i0 < z.imag < i1)


def main():
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        W.DATA = tmp
        for i, (name, extra) in enumerate(CASES.items()):
            info = dict(D=N + 2, M=0, omega=OM, lam_g=LG, lam_f=LP)
            rec = W.count(det_for(extra), info, POLES, {}, lambda m: None, f"planted{i}", os.path.join(tmp, "res.jsonl"))
            bad = []
            for b, x in rec["boxes"].items():
                want = expected(x["rect"], extra)
                if x["winding_g_int"] != want or abs(x["winding_g"] - want) > 1e-6:
                    bad.append(f"{b}: wind(g) {x['winding_g']:+.3f}, planted {want}")
            mx = max(e["max_step"] for e in rec["edges"].values())
            if mx > math.pi / 4 + 1e-12:
                bad.append(f"max phase step {mx:.3f} > pi/4")
            ok &= not bad
            got = ", ".join(f"{b} {x['winding_g_int']}" for b, x in rec["boxes"].items())
            print(f"{'PASS' if not bad else 'FAIL'} {name}: {got}; {rec['evaluations']} evaluations, max step {mx:.3f}"
                  + ("" if not bad else "  <- " + "; ".join(bad)), flush=True)
    print("all cases pass" if ok else "SOME CASES FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

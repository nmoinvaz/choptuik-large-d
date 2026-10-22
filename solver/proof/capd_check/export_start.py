#!/usr/bin/env python3
"""Export the step-(i) start data for capd_check as outward-rounded double intervals (exact hex floats).

This is the only place where capd_check touches the Python proof code. It calls

    fixed_point.start_data   (step (i), proof/A_certify.md 4.4-4.5): y+(xi0) at xi0 = 0.1 e^{3 i pi/8} and its
                             derivatives d/dA, d/d(w/w_inf), rigorous, over a real parameter box
    krawczyk.normalization   (rigorous kappa = xi1^(-A^2) / N at the centre, used only to print g = kappa J_D)

and writes the results to data/. Nothing of validated_int.py is used. Every arb ball [m +- r] is written as the pair
of doubles (lo, hi) with lo <= m - r and m + r <= hi, checked exactly in arb.

    python3 export_start.py              # data/start_point.txt, data/start_box_1e-9.txt, data/start_box_1e-8.txt
    python3 export_start.py 1e-7 1e-6    # more boxes (data/ also holds 1e-7, 1e-6, 1e-5)

The box files are computed over x^ +- h (1 + 2^-16) with h the double nearest 1e-9 or 1e-8, slightly more than the box
capd_check tests (x^ +- h, with x^ itself only known to a double interval), so that every parameter capd_check
uses lies inside the box the derivatives are valid for. capd_check checks this margin.
"""
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from flint import arb, acb, ctx  # noqa: E402

import fixed_point as FP  # noqa: E402
import krawczyk as KR  # noqa: E402

XHAT = ("2.5421480475", "0.7769940136")
R0, XI1, CAUCHY = "0.1", "0.3", "1e-3"
NAMES = ["D-1", "D+1", "S-1", "S+1", "C-2", "C0", "C+2"]


def dbl_bounds(x):
    """doubles lo <= x.lower(), hi >= x.upper(), checked exactly"""
    lo, hi = x.lower(), x.upper()
    a, b = float(lo), float(hi)
    while not arb(a) <= lo:
        a = math.nextafter(a, -math.inf)
    while not arb(b) >= hi:
        b = math.nextafter(b, math.inf)
    return a, b


def dd_bounds(x):
    """double-double bounds: lo = a0 + a1 <= x.lower(), hi = b0 + b1 >= x.upper(), checked exactly (about 1e-32
    relative), for the multiprecision point run"""
    lo, hi = x.lower(), x.upper()
    a0 = float(lo)
    a1 = float(lo - arb(a0))
    while not arb(a0) + arb(a1) <= lo:
        a1 = math.nextafter(a1, -math.inf)
    b0 = float(hi)
    b1 = float(hi - arb(b0))
    while not arb(b0) + arb(b1) >= hi:
        b1 = math.nextafter(b1, math.inf)
    return a0, a1, b0, b1


def line2(tag, k, z):
    """tag k  re_lo (2 doubles) re_hi (2) im_lo (2) im_hi (2), hex"""
    r = dd_bounds(z.real)
    i = dd_bounds(z.imag)
    return f"{tag} {k} " + " ".join(v.hex() for v in r + i)


def line(tag, k, z):
    """tag k re_lo re_hi im_lo im_hi (hex), plus a readable comment"""
    rl, rh = dbl_bounds(z.real)
    il, ih = dbl_bounds(z.imag)
    return (f"{tag} {k} {rl.hex()} {rh.hex()} {il.hex()} {ih.hex()}   "
            f"# {z.real.mid().str(17)} {z.imag.mid().str(17)}i, radii {float(z.real.rad()):.2e} {float(z.imag.rad()):.2e}")


def real_line(tag, x):
    lo, hi = dbl_bounds(x)
    return f"{tag} {lo.hex()} {hi.hex()}   # {x.mid().str(20)} +- {float(x.rad()):.2e}"


def export(fname, hd=None):
    ctx.prec = 320
    Ah, wh = arb(XHAT[0]), arb(XHAT[1])
    t0 = time.time()
    _, vals, dAs, dws, eta = FP.start_data(Ah, wh, R0, +1, arb(CAUCHY))
    out = [f"# capd_check start data: fixed_point.start_data(A, wf, r0={R0}, theta_sign=+1, cauchy={CAUCHY})",
           "# state order [D-1, D+1, S-1, S+1, C-2, C0, C+2] at xi0 = 0.1 e^{3 i pi/8}; wf = w/w_inf",
           "# lines: Y/dA/dw k re_lo re_hi im_lo im_hi (hex doubles, outward), comment = midpoint and radii",
           "# Y2 lines: the same as double-double bounds lo = lo0 + lo1, hi = hi0 + hi1 (re lo0 lo1 hi0 hi1, im ...)",
           real_line("xhatA", Ah), real_line("xhatW", wh),
           "xhatA2 " + " ".join(v.hex() for v in dd_bounds(Ah)),
           "xhatW2 " + " ".join(v.hex() for v in dd_bounds(wh)),
           f"# eta: largest component bound at the centre {float(eta[0].upper()):.2e} (fixed_point.start_data's return value)"]
    out += [line("Y", k, v) for k, v in enumerate(vals)]
    out += [line2("Y2", k, v) for k, v in enumerate(vals)]
    if hd is None:          # the point: derivatives at x^ itself (the C^1 data at a point)
        out += [line("dA", k, v) for k, v in enumerate(dAs)]
        out += [line("dw", k, v) for k, v in enumerate(dws)]
        N, xs = KR.normalization(XHAT, XI1, prec=320)
        kap = xs / N
        out.append(line("kappa", 0, kap))
        out.append(f"# kappa = xi1^(-A^2) / N(x^) at xi1 = {XI1} from krawczyk.normalization; |kappa| = "
                   f"{float(abs(kap).mid()):.6f}")
    else:
        rX = arb(hd) * (1 + arb(2) ** -16)
        A, wf = Ah + arb(0, rX), wh + arb(0, rX)
        _, valsX, dAX, dwX, etaX = FP.start_data(A, wf, R0, +1, arb(CAUCHY))
        out.append(f"h {hd.hex()}   # the box half-width capd_check tests, the double nearest {hd:.0e}")
        out.append(real_line("hX", rX) + "   (start data valid over x^ +- hX)")
        out += [line("dA", k, v) for k, v in enumerate(dAX)]
        out += [line("dw", k, v) for k, v in enumerate(dwX)]
    out.append(f"# computed in {time.time() - t0:.0f} s")
    with open(os.path.join(HERE, "data", fname), "w") as f:
        f.write("\n".join(out) + "\n")
    print("wrote", fname, f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    which = sys.argv[1:] or ["point", "1e-9", "1e-8"]
    for w in which:
        if w == "point":
            export("start_point.txt")
        else:
            export(f"start_box_{w}.txt", float(w))

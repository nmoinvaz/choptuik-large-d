"""chi, g estimators at x = 0.5, 0.6 from the exact pole sum (no cancellation) vs the direct series."""
import os
import sys
import mpmath as mp
sys.argv = ["x", "122"]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "largeD_linear_late.py")).read().split('if __name__')[0])
def res_c(J, z):
    return ((-1) ** J / mp.factorial(J) * mp.rf(a, J + 1) * mp.rf(b, J + 1) / mp.factorial(J + 1) * z ** (J + 1)
            * mp.hyp2f1(a + J + 1, b + J + 1, J + 2, z))
for xs in ("0.5", "0.6"):
    x = mp.mpf(xs); z = x * x
    R = [res_c(J, z) for J in range(0, 1400)]
    yd = lambda k: 2 * x * (-1) ** (k - 1) * mp.fsum(R[J] * (2 * J + 1) ** (k - 1) for J in range(len(R)))
    Phi = mp.log(x) - mp.log(1 - z) / 2
    for k in (100, 110, 118):
        for name, f in (("exact", yd), ("direct", lambda k: yk(k, x))):
            y0, y1, y2 = f(k), f(k + 1), f(k + 2)
            r0, r1 = y1 / y0, y2 / y1
            chi = 1 / (r1 - r0); g = k - chi * r0
            print(f"x {xs} k {k} {name:6s}: chi {mp.nstr(chi, 7)}  (Phi - i pi/2 = {mp.nstr(Phi - 1j*mp.pi/2, 7)})  g {mp.nstr(g, 5)}"
                  + (f"   direct/exact - 1 = {mp.nstr(abs(yk(k, x)/yd(k) - 1), 2)}" if name == "direct" else ""), flush=True)

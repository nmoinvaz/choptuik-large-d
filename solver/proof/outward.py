"""Decimal strings of arb endpoints rounded outward (AGENTS.md: quote enclosures with endpoints rounded outward).

arb's .str(n, radius=False) rounds to nearest, so a printed upper bound can fall below the value it bounds (#1063,
finding 1). up_str gives x.upper() rounded up to n significant digits, lo_str gives x.lower() rounded down.

    python3 outward.py                    self-test: planted values, a carry to the next power of ten; exit 0 iff all pass
    python3 outward.py --mutate=nearest   control: arb's own round-to-nearest strings in place of these; must fail, exit 1
"""
import math
import sys
from fractions import Fraction

from flint import arb


def _exact(x):
    """the exact rational value of an arb of radius zero"""
    m, e = x.man_exp()
    return Fraction(int(m)) * Fraction(2) ** int(e)


def _sig(fr, n, toward_plus):
    """fr rounded to n significant digits, toward +infinity if toward_plus, else toward -infinity"""
    if fr == 0:
        return "0"
    neg = fr < 0
    a = -fr if neg else fr
    k = len(str(a.numerator)) - len(str(a.denominator))     # within 1 of the exponent, no float range limit
    while Fraction(10) ** k > a:                           # exact leading-digit exponent, 10^k <= a < 10^(k+1)
        k -= 1
    while Fraction(10) ** (k + 1) <= a:
        k += 1
    v = a * Fraction(10) ** (n - 1 - k)
    away = toward_plus != neg                                # magnitude rounds up for a positive upper end
    q = math.ceil(v) if away else math.floor(v)
    if q == 10 ** n:                                         # carry: 9.99..9 up to 10.0..0
        q, k = q // 10, k + 1
    d = str(q)
    sign = "-" if neg else ""
    if -4 <= k < 6:
        p = n - 1 - k                                        # digits after the point
        if p <= 0:
            return sign + d + "0" * (-p)
        s = d.rjust(p + 1, "0")
        return sign + s[:-p] + "." + s[-p:]
    return sign + d[0] + ("." + d[1:] if n > 1 else "") + f"e{k:+d}"


def up_str(x, n):
    """x's upper end, rounded up to n significant digits"""
    return _sig(_exact(x.upper()), n, True)


def lo_str(x, n):
    """x's lower end, rounded down to n significant digits"""
    return _sig(_exact(x.lower()), n, False)


def _selftest():
    cases = [(arb("0.0110124300"), 5, "0.011013", "0.011012"), (arb("28.4491411"), 5, "28.450", "28.449"),
             (arb("4766104.06"), 5, "4.7662e+6", "4.7661e+6"), (arb("9.99995"), 5, "10.000", "9.9999"),
             (arb("-2.5"), 2, "-2.5", "-2.5"), (arb("-2.51"), 2, "-2.5", "-2.6"), (arb("3.3072783"), 5, "3.3073", "3.3072"),
             (arb("0.000012345678"), 3, "1.24e-5", "1.23e-5"), (arb("1.2e-55"), 2, "1.3e-55", "1.1e-55"),
             (arb("163995.2"), 6, "163996", "163995"), (arb(100), 3, "100", "100"),
             (arb("0.000123456"), 3, "0.000124", "0.000123"), (arb("1.5e-400"), 2, "1.6e-400", "1.4e-400"),
             (arb("2.5e400"), 2, "2.6e+400", "2.4e+400")]
    nearest_mut = "--mutate=nearest" in sys.argv                 # control: arb's round-to-nearest must fail the test
    bad = 0
    for x, n, want_up, want_lo in cases:
        if nearest_mut:
            u, lo = x.upper().str(n, radius=False), x.lower().str(n, radius=False)
        else:
            u, lo = up_str(x, n), lo_str(x, n)
        ok = u == want_up and lo == want_lo and arb(u) >= x.upper() and arb(lo) <= x.lower()
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {x.str(12)}: up {u} (want {want_up}), lo {lo} (want {want_lo})")
    nearest = arb("0.0110124300").str(5, radius=False)
    print(f"control: arb's own str gives {nearest} for 0.01101243, below it: "
          f"{'yes, as expected' if arb(nearest) < arb('0.0110124300') else 'NO'}")
    print("outward self-test:", "PASS" if bad == 0 else f"FAIL ({bad})")
    return bad == 0


if __name__ == "__main__":
    sys.exit(0 if _selftest() else 1)

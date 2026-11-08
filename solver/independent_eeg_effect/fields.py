"""Printed NLO and NNLO of arXiv:2611.07007v1, supplement (S40)-(S51), and the corrections claimed in #176.

b[k] is the k-th tau-derivative of beta~. Plain beta in (S43)-(S51) is read as beta~.
Transcribed from main.tex of the arXiv e-print 2611.07007v1, not from other code in this repository.
"""
import sympy as sp

x = sp.Symbol('x', real=True)
b = sp.symbols('b0:7', real=True)
R = sp.Rational


def printed():
    b0, b1, b2, b3 = b[:4]
    f1 = -(1 - x**2) / 2 * b0**2
    Om1 = b0**2
    Pi1 = b0
    Psi1 = b1 + b0
    # (S40)-(S43)
    f2 = R(1, 8) * (1 - x**2) * b0 * (4 * (1 - x**2) * b0 + (3 + x**2) * b0**3
                                       - 2 * (1 + x**2) * (b2 + 3 * b1))
    Om2 = b0 * (x**2 * (b2 + 3 * b1 - b0**3) - 2 * (1 - x**2) * b0)
    Pi2 = x**2 / 2 * (2 * b0 + b2 + 3 * b1 - b0**3)
    Psi2 = R(1, 2) * (b0**3 + b0**2 * b1
                      + x**2 * (-4 * b0**3 - 4 * b0**2 * b1 + 6 * b0 + 11 * b1 + 6 * b2 + b3))
    # (S46)
    Pi3 = (x**2 / 6 * (24 * b0**3 + 15 * b0**2 * b1 + 3 * b0**2 * b2)
           + x**4 / 6 * (3 * b0**5 - 16 * b0**3 - 20 * b0**2 * b1 - 4 * b0**2 * b2 + 12 * b0 + 22 * b1
                         + 12 * b2 + 2 * b3))
    # (S45) without its 2 beta Pi3 term.
    Om3_rest = (x**4 / 4 * (b0**6 - 4 * b0**4 - 6 * b0**3 * b1 - 2 * b0**3 * b2 + 4 * b0**2 + 12 * b0 * b1
                            + 4 * b0 * b2 + 9 * b1**2 + 6 * b1 * b2 + b2**2)
                + x**2 * (4 * b0**4 - 7 * b0**2 - 10 * b0 * b1 - 4 * b0 * b2 + b1**2))
    # (S48)-(S51)
    c0 = (-R(17, 48) * b0**6 - R(79, 36) * b0**4 + R(35, 72) * b0**3 * b1 + R(13, 72) * b0**3 * b2
          + R(11, 12) * b0**2 + R(7, 9) * b0 * b1 + R(1, 6) * b0 * b2 - R(1, 9) * b0 * b3
          - R(5, 8) * b1**2 - R(1, 4) * b1 * b2 - R(1, 24) * b2**2)
    c2 = R(3, 16) * b0**6 + R(3, 4) * b0**4 - R(3, 8) * b0**3 * b1 - R(1, 8) * b0**3 * b2
    c4 = (R(1, 16) * b0**6 + R(9, 4) * b0**4 + R(7, 8) * b0**3 * b1 + R(1, 8) * b0**3 * b2
          - R(7, 4) * b0**2 - R(5, 2) * b0 * b1 - b0 * b2 + R(1, 4) * b1**2)
    c6 = (R(5, 48) * b0**6 - R(29, 36) * b0**4 - R(71, 72) * b0**3 * b1 - R(13, 72) * b0**3 * b2
          + R(5, 6) * b0**2 + R(31, 18) * b0 * b1 + R(5, 6) * b0 * b2 + R(1, 9) * b0 * b3
          + R(3, 8) * b1**2 + R(1, 4) * b1 * b2 + R(1, 24) * b2**2)
    # (S44)
    f3 = c0 + x**2 * c2 + x**4 * c4 + x**6 * c6
    return dict(f1=f1, f2=f2, f3=f3, Om1=Om1, Om2=Om2, Om3_rest=Om3_rest, Pi1=Pi1, Pi2=Pi2, Pi3=Pi3,
                Psi1=Psi1, Psi2=Psi2)


def corrections():
    """The claimed dPi3 and df3, with corrected = printed + delta."""
    b0, b1, b2, b3, b4 = b[:5]
    dPi3 = (-b0 * x**2 * (4 * b0**2 + b0 * b1 - b1**2) / 2
            - x**4 * (3 * b0**5 + 8 * b0**3 + 40 * b0**2 * b1 + 8 * b0**2 * b2 + 24 * b0 * b1**2 - 24 * b0
                      - 62 * b1 - 57 * b2 - 22 * b3 - 3 * b4) / 24)
    d0 = (b0**6 / 24 + R(10, 9) * b0**4 + R(29, 36) * b0**3 * b1 + b0**3 * b2 / 9 + b0**2 * b1**2 / 12
          - b0**2 / 3 - R(31, 36) * b0 * b1 - R(19, 24) * b0 * b2 - R(11, 36) * b0 * b3 - b0 * b4 / 24)
    d4 = -b0**4 - b0**3 * b1 / 4 + b0**2 * b1**2 / 4
    d6 = (-b0**6 / 24 - b0**4 / 9 - R(5, 9) * b0**3 * b1 - b0**3 * b2 / 9 - b0**2 * b1**2 / 3 + b0**2 / 3
          + R(31, 36) * b0 * b1 + R(19, 24) * b0 * b2 + R(11, 36) * b0 * b3 + b0 * b4 / 24)
    df3 = d0 + x**4 * d4 + x**6 * d6
    return dict(dPi3=dPi3, df3=df3, d0=d0)

"""Shared helper for the lemma6_* scripts (proof/A_certify.md section 11).

validated_int.apriori and apriori_V find an a-priori enclosure by interval Picard iteration, inflating every real and
imaginary radius by a factor 1.5 plus 1e-60 until containment holds. On the real axis some components are nearly real
(C_0, and the imaginary parts of the derivatives), so their imaginary radius starts near 1e-15 while the wrapping from
the other components feeds it at the scale h |F|; 30 multiplications by 1.5 never catch up, and the step size collapses
to about 1e-6 (the real leg 0.2 -> 0.3 would take 10^6 steps). The versions here add to every radius a common floor,
1e-2 times the largest radius of the vector (or matrix). That is only a different initial guess for the iteration:
containment is still verified with `contains` exactly as in validated_int.py, so every enclosure stays rigorous.
Two step-control rules (choices, not bounds) come with it: a step whose a-priori box grows by more than MAXREL times
(1 + |state|) is refused (otherwise the Taylor remainder over a huge box overflows the floating-point QR), and a step
whose floating-point QR fails is reported as failed instead of raising (the set is only replaced at the end of a
successful step). validated_int.py itself is not changed; install() swaps these functions inside the imported module.
"""
from flint import acb, arb, acb_mat

import validated_int as VI


def _infl(z, fac, add):
    re, im = z.real, z.imag
    return acb(arb(re.mid(), re.rad() * fac + add), arb(im.mid(), im.rad() * fac + add))


def _floor(zs):
    m = 0.0
    for z in zs:
        m = max(m, float(z.real.rad()), float(z.imag.rad()))
    return arb(1e-2 * m + 1e-60)


def apriori(sys_, leg, t0, h, Xbox):
    """as validated_int.apriori; the inflation adds a common floor (1e-2 x the largest radius)"""
    T = acb(t0) + acb(arb(h / 2, h / 2))
    xi = leg.xi(T)
    dxi = leg.dxi(xi)
    ixi = 1 / xi
    H = acb(arb(h / 2, h / 2))
    zero = acb(0)
    nd = sys_.nd
    F, _ = sys_.rhs(Xbox, xi, ixi, zero)
    Yt = [Xbox[i] + H * F[i] * dxi for i in range(nd)]
    fl = _floor(Yt)
    Yt = [_infl(y, 2.0, fl) for y in Yt] + Xbox[nd:]
    scale = 1 + max(float(abs(x).upper()) for x in Xbox[:nd])
    for it in range(30):
        F, _ = sys_.rhs(Yt, xi, ixi, zero)
        Z = [Xbox[i] + H * F[i] * dxi for i in range(nd)] + Xbox[nd:]
        if all(Yt[i].contains(Z[i]) for i in range(nd)):
            # step control (not a bound): refuse steps whose a-priori box is large, as the remainder would be useless
            grow = max(float(Z[i].real.rad()) + float(Z[i].imag.rad())
                       - float(Xbox[i].real.rad()) - float(Xbox[i].imag.rad()) for i in range(nd))
            if grow > MAXREL * scale:
                return None, T
            return Z, T
        U = [Yt[i].union(Z[i]) for i in range(nd)]
        fl = _floor(U)
        Yt = [_infl(u, 1.5, fl) for u in U] + Xbox[nd:]
    return None, T


def apriori_V(sys_, leg, T, Yt, h):
    """as validated_int.apriori_V; the inflation adds a common floor (1e-2 x the largest entry radius)"""
    n = sys_.n
    xi = leg.xi(T)
    dxi = leg.dxi(xi)
    ixi = 1 / xi
    zero = acb(0)
    _, cache = sys_.rhs(Yt, xi, ixi, zero)
    Fy = acb_mat(n, n)
    for j in range(n):
        e = [acb(1) if i == j else acb(0) for i in range(n)]
        col = sys_.lin(cache, e, xi, ixi, zero)
        for i in range(sys_.nd):
            Fy[i, j] = col[i] * dxi
    H = acb(arb(h / 2, h / 2))
    Id = acb_mat([[1 if i == j else 0 for j in range(n)] for i in range(n)])
    V = Id + H * (Fy * Id)
    fl = _floor([V[i, j] for i in range(n) for j in range(n)])
    V = acb_mat([[_infl(V[i, j], 2.0, fl) for j in range(n)] for i in range(n)])
    for it in range(30):
        Z = Id + H * (Fy * V)
        if all(V[i, j].contains(Z[i, j]) for i in range(n) for j in range(n)):
            return Z
        U = [[V[i, j].union(Z[i, j]) for j in range(n)] for i in range(n)]
        fl = _floor([u for row in U for u in row])
        V = acb_mat([[_infl(U[i][j], 1.5, fl) for j in range(n)] for i in range(n)])
    return None


MAXREL = 0.05
_orig_step = VI.Lohner.step


def _safe_step(self, leg, t0, h):
    """validated_int.Lohner.step, with a failed step (instead of an exception) when the enclosures are so wide that the
    floating-point QR of the Lohner update overflows; the set is only replaced at the end of a successful step"""
    try:
        return _orig_step(self, leg, t0, h)
    except (ZeroDivisionError, ValueError, OverflowError, FloatingPointError):
        return False


def install():
    VI.apriori = apriori
    VI.apriori_V = apriori_V
    VI.Lohner.step = _safe_step

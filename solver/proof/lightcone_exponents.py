"""Light-cone exponents of the M-harmonic model (Lemma B6, non-resonance).

Near x = 1 the transport of the fast amplitude (Proposition 3) reduces to t A_t = (-omega d/dtheta - 1 + F(theta)^2) A,
t = 1 - x, so the light-cone exponents (minus n/2) are the eigenvalues of the operator T = -omega d/dtheta - 1 + F^2 on
the odd harmonics. Untruncated, the gauge e^{Pi}, Pi = omega^-1 int (F^2 - <F^2>), diagonalizes T exactly: the
eigenvalues are -1 + <F^2> - i mu omega, mu odd, pairwise differing by non-zero imaginary amounts, so no two differ by an
integer (non-resonance). In the M-harmonic model T is replaced by its Galerkin truncation P_M T P_M, which breaks the
gauge step; this script computes the truncated eigenvalues, their distance from the untruncated ones, and the smallest
distance of any pairwise difference from the integers.

    python3 lightcone_exponents.py
"""
import numpy as np

W0 = 2 * np.pi / 1.64389375


def profile(A, r3=0.054883 + 0.005414j, r5=0.003092 + 0.000541j):
    """F = 2 Re[F1 e^{i th} + F3 e^{3 i th} + F5 e^{5 i th}], F1 = A/(2i) (the M = 5 zero of hier.py)"""
    F1 = A / 2j
    return {1: F1, 3: F1 * r3, 5: F1 * r5, -1: np.conj(F1), -3: np.conj(F1 * r3), -5: np.conj(F1 * r5)}


def F2_coeffs(Fc):
    out = {}
    for a, fa in Fc.items():
        for b, fb in Fc.items():
            out[a + b] = out.get(a + b, 0) + fa * fb
    return out


def truncated_T(M, Fc, w):
    mus = [m for m in range(-M, M + 1) if m % 2]
    F2 = F2_coeffs(Fc)
    T = np.zeros((len(mus), len(mus)), complex)
    for i, mu in enumerate(mus):
        T[i, i] += -1j * mu * w - 1
        for j, nu in enumerate(mus):
            T[i, j] += F2.get(mu - nu, 0)
    return mus, T


for A, wf in ((2.7186, 0.99923), (2.72, 1.0)):
    Fc = profile(A)
    meanF2 = F2_coeffs(Fc)[0].real
    print(f"A {A}  w/w_inf {wf}  <F^2> = {meanF2:.6f}")
    for M in (1, 3, 5, 7, 9, 15, 25):
        mus, T = truncated_T(M, Fc, wf * W0)
        ev = np.linalg.eigvals(T)
        ev = ev[np.argsort(ev.imag)]
        exact = np.array(sorted([-1 + meanF2 - 1j * mu * wf * W0 for mu in mus], key=lambda z: z.imag))
        dev = np.abs(ev - exact)
        diffs = [ev[i] - ev[j] for i in range(len(ev)) for j in range(len(ev)) if i != j]
        res = min(abs(d - round(d.real)) if abs(d.imag) < 0.5 else abs(d.imag) for d in diffs) if diffs else np.inf
        print(f"   M {M:2d}: max |eig - (<F^2> - 1 - i mu w)| = {dev.max():.3e} (central harmonics {dev[len(dev)//2]:.1e});"
              f"  min distance of eigenvalue differences from Z: {res:.3f};  min |Im diff| {min(abs(d.imag) for d in diffs) if diffs else np.inf:.3f};"
              f"  max |Re diff| {max(abs(d.real) for d in diffs) if diffs else 0:.3f}")

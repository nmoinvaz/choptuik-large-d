"""Global Chebyshev collocation of the free self-similar wave against n: does a fixed degree N see the horizon (#1089)?

The regular-at-centre self-similar wave e^{lambda tau} f(x) of the flat free field in n + 2 dimensions is
f = 2F1(a, b; c; z), z = x^2, a = lambda/2, b = (lambda + 1)/2, c = (n + 1)/2 (notes/LARGE_D_BASICS.md 3.4). Its exponents
are 0 and (1 - n)/2 at z = 0 and 0 and sigma = n/2 - lambda at z = 1. A wave regular at both ends exists only where
B = Gamma(c)Gamma(a + b - c)/(Gamma(a)Gamma(b)) = 0, never for lambda = i j omega ("a free wave cannot echo").

This script collocates z(1 - z) f'' + (c - (a + b + 1) z) f' - a b f = 0 with a polynomial of degree N in z at the N + 1
Chebyshev-Lobatto points of [0, 1], with no boundary rows (as RT's frequency-space scheme has none), and prints the
smallest singular value of the row-equilibrated matrix against n at fixed N. A small value means the polynomial space
nearly holds a wave regular at both ends, i.e. the discretization cannot see the horizon condition (B) the selection
needs. The model of cheb_branch_tail.py predicted the smallest value near Re sigma ~ 0.75 N and a recovery beyond. This
script refutes that: the ratio falls like n^-(N+1) for n >> N with no recovery, because the centre-regular wave is a
polynomial of degree N up to its Taylor tail, whose terms fall like n^-k, and the horizon information lives in that tail
(the fast mode varies on the scale x/n). The floor near 1e-85 is the working precision.

Usage: python3 free_field_cheb.py [--N=40] [--j=1] [--mutate=planted]
  --mutate=planted: control; lambda = -2 (a = -1), where 2F1 is a polynomial of degree 1, regular at both ends. The
  matrix must then be singular to working precision (sigma_min/sigma_max < 1e-40), else exit 1.
"""
import sys

import mpmath as mp

mp.mp.dps = 80


def cheb_nodes(N):
    return [(1 - mp.cos(mp.pi * k / N)) / 2 for k in range(N + 1)]  # Lobatto points mapped to [0, 1]


def T_and_derivs(m, z):
    """T_m(s), dT/dz, d2T/dz2 at s = 2z - 1, by the trigonometric/derivative recurrences."""
    s = 2 * z - 1
    T = [mp.mpf(1), s]
    dT = [mp.mpf(0), mp.mpf(1)]
    d2T = [mp.mpf(0), mp.mpf(0)]
    for k in range(2, m + 1):
        T.append(2 * s * T[k - 1] - T[k - 2])
        dT.append(2 * T[k - 1] + 2 * s * dT[k - 1] - dT[k - 2])
        d2T.append(4 * dT[k - 1] + 2 * s * d2T[k - 1] - d2T[k - 2])
    return T, [2 * v for v in dT], [4 * v for v in d2T]  # chain rule ds/dz = 2


def smin_ratio(n, lam, N):
    a, b, c = lam / 2, (lam + 1) / 2, mp.mpf(n + 1) / 2
    A = mp.matrix(N + 1, N + 1)
    for i, z in enumerate(cheb_nodes(N)):
        T, dT, d2T = T_and_derivs(N, z)
        row = [z * (1 - z) * d2T[m] + (c - (a + b + 1) * z) * dT[m] - a * b * T[m] for m in range(N + 1)]
        scale = max(abs(v) for v in row)
        for m in range(N + 1):
            A[i, m] = row[m] / scale
    s = mp.svd_c(A, compute_uv=False)
    sv = sorted(abs(s[k]) for k in range(N + 1))
    return sv[0] / sv[-1]


def main():
    N, j = 40, 1
    for arg in sys.argv[1:]:
        if arg.startswith('--N='):
            N = int(arg[4:])
        if arg.startswith('--j='):
            j = int(arg[4:])
    omega = mp.mpf('3.822')
    if '--mutate=planted' in sys.argv:
        r = smin_ratio(60, mp.mpf(-2), N)
        print(f"planted lambda = -2, n 60, N {N}: sigma_min/sigma_max = {mp.nstr(r, 4)}")
        ok = r < mp.mpf(10) ** -40
        print("control " + ("as expected (singular)" if ok else "FAILED"))
        sys.exit(0 if ok else 1)
    lam = mp.mpc(0, j * omega)
    print(f"N = {N}, lambda = i {j} omega (omega {omega}); Re sigma = n/2; working precision 1e-{mp.mp.dps}")
    print(f"{'n':>6} {'Re sigma':>9} {'sigma_min/sigma_max':>21} {'local slope d log/d log n':>27}")
    prev = None
    for n in (10, 20, 30, 40, 50, 60, 80, 100, 140, 200, 400, 1000, 2000):
        r = smin_ratio(n, lam, N)
        slope = '' if prev is None else mp.nstr(mp.log(r / prev[1]) / mp.log(mp.mpf(n) / prev[0]), 4)
        print(f"{n:6d} {n / 2:9.1f} {mp.nstr(r, 4):>21} {slope:>27}")
        prev = (n, r)


if __name__ == '__main__':
    main()

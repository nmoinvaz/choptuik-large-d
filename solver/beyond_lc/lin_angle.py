"""Linear (c = b = 1) self-similar wave problem, one harmonic e^{i k theta}: how nearly parallel are the
centre-regular and the light-cone-analytic solutions at a matching point x_m?

The normalized Wronskian |p1 m2 - p2 m1| / (|(p1,m1)| |(p2,m2)|) is the sine of the angle between the two solution
vectors; it is the singular value with which the standard two-sided matching (the hp/ solver's layout) sees the
light-cone obstruction of harmonic k. Exact closed forms (Proposition 2 of the README): y = p - m,
y_reg = x 2F1(a, b; kappa; x^2), y_an = x 2F1(a, b; a + b - kappa + 1; 1 - x^2), a = (1+s)/2, b = (2+s)/2,
kappa = (n+1)/2, s = i k omega, p + m = (x(1-x^2) y' - (1 + s x^2) y) / (x (s - n + 1)).

Usage: python3 lin_angle.py
"""
import mpmath as mp

mp.mp.dps = 40
W = 2 * mp.pi / mp.mpf('1.6438937')


def pm_from(y, yp, x, s, n):
    sig = (x * (1 - x**2) * yp - (1 + s * x**2) * y) / (x * (s - n + 1))
    return ((sig + y) / 2, (sig - y) / 2)


def angle(n, k, xm, omega=W):
    s = 1j * k * omega
    a = (1 + s) / 2
    b = (2 + s) / 2
    kap = mp.mpf(n + 1) / 2
    z = xm**2
    w = mp.hyp2f1(a, b, kap, z)
    wp = a * b / kap * mp.hyp2f1(a + 1, b + 1, kap + 1, z)
    yr, yrp = xm * w, w + 2 * z * wp
    c2 = a + b - kap + 1
    w = mp.hyp2f1(a, b, c2, 1 - z)
    wp = -a * b / c2 * mp.hyp2f1(a + 1, b + 1, c2 + 1, 1 - z)
    ya, yap = xm * w, w + 2 * z * wp
    p1, m1 = pm_from(yr, yrp, xm, s, n)
    p2, m2 = pm_from(ya, yap, xm, s, n)
    det = p1 * m2 - p2 * m1
    n1 = mp.sqrt(abs(p1)**2 + abs(m1)**2)
    n2 = mp.sqrt(abs(p2)**2 + abs(m2)**2)
    return abs(det) / (n1 * n2)


if __name__ == '__main__':
    ns = [19, 29, 39, 49, 59, 99, 199]
    for xm in ['0.5', '0.7071067811865475', '0.8', '0.9']:
        print('x_m =', xm, ' columns n =', ns)
        for k in [1, 3, 5, 7, 9, 13]:
            row = [f'{float(angle(n, k, mp.mpf(xm))):.1e}' for n in ns]
            print(f'  k={k:2d}', ' '.join(row), flush=True)

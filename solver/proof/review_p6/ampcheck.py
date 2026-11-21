"""PC1 of notes/P6_route3_review.md: the review's model against amp_flint.py's own passes and S_at.

amp_flint.Amp runs untuned. The perturbation is injected at its natural order k_mu = |mu| with add_linear (d0 = 2f),
and the multiplier sequence of harmonic nu is read at order k_mu + 2, the order where the perturbation first acts on
nu through one A^2 step. S_at's own late-order sequence y_j = base_j [z N]_k is captured by replacing mpmath.qr_solve
for the call. The part linear in the perturbation is (y(+eps) - y(-eps)) / (2 eps), eps = 2^-60.

    python3 ampcheck.py                     PASS: relative difference <= 1e-30 at nu = 3, 5, each mu, j <= 40
    python3 ampcheck.py --mutate=droppair   M6, one ordered lapse pair dropped from the model: FAIL, exit 1
"""
import os
import sys
import mpmath

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'inner_amp'))
sys.path.insert(0, HERE)
import amp_flint                                   # noqa: E402
from flint import acb, arb                        # noqa: E402
import model                                      # noqa: E402

J = 40
BAR = mpmath.mpf('1e-30')


def amp_sequence(nu, mu, f):
    km = abs(mu)
    K = km + 2
    A = amp_flint.Amp(1.0, K, J, prec=320, dps=80)
    A.jlo = 0                                      # Amp treats jlo=0 as "default", (2J)//5
    A.P = {}
    for k in range(1, K + 1):
        if k % 2 == 0:
            A.pass_even(k)
        else:
            A.pass_odd(k)
            if k == km:
                A.add_linear(k, mu, f)
            A.P[("d", k)] = A.poly(A.d[k])
            A.P[("s", k)] = A.poly(A.s[k])
    got = {}
    orig = mpmath.qr_solve

    def grab(Am, y):
        got['y'] = [y[i] for i in range(y.rows)]
        return mpmath.matrix(Am.cols, 1), mpmath.mpf(0)
    mpmath.qr_solve = grab
    try:
        A.S_at(nu, K)
    finally:
        mpmath.qr_solve = orig
    return got['y']


def main():
    if '--mutate=droppair' in sys.argv:
        model.MUTATE.add('droppair')
    eps = mpmath.mpf(2) ** -60
    worst = 0
    for nu in (3, 5):
        for mu in (nu - 2, nu, nu + 2):
            yp = amp_sequence(nu, mu, acb(arb(eps / 2)))
            ym = amp_sequence(nu, mu, acb(arb(-eps / 2)))
            mpmath.mp.prec = 320
            s_ = 1j * nu * 2 * mpmath.pi / mpmath.mpf("1.64389375")
            K = mpmath.rgamma((1 + s_) / 2) * mpmath.rgamma((2 + s_) / 2)
            amp = [(a - b) / (2 * eps) / K for a, b in zip(yp, ym)]
            lit = model.literal(nu, mu, J)
            C, w = model.ctx('mp', 320)
            D = model.darboux(nu, mu, J, C, w)
            R0 = lit[0]
            dar = [R0]
            for x in D:
                dar.append(dar[-1] + x)
            scale = max(abs(x) for x in amp)
            e1 = max(abs(a - b) for a, b in zip(amp, lit)) / scale
            e2 = max(abs(a - b) for a, b in zip(amp, dar)) / scale
            e3 = abs(amp[0] - R0) / scale
            worst = max(worst, e1, e2)
            print(f"nu {nu} mu {mu:2d} (inject at order {abs(mu)}, read order {abs(mu) + 2}): max|R| {mpmath.nstr(scale, 6)}"
                  f"  literal {mpmath.nstr(e1, 3)}  darboux {mpmath.nstr(e2, 3)}  R_0 {mpmath.nstr(e3, 3)}", flush=True)
    ok = worst <= BAR
    print(f"PC1 {'PASS' if ok else 'FAIL'}: worst relative difference {mpmath.nstr(worst, 3)} (bar {mpmath.nstr(BAR, 1)})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

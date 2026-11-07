"""(a) Hermite level nu = 2 at large D: a non-monotone branch with alpha near 4 (z < 2).
(b) D = 3, alpha(z) at large z: return toward 2 (oscillation about the singular solution 2/x^2,
exponents lambda = (-1 +- i sqrt 7)/2 at D = 3), first crossing of alpha = 2 after z = 6."""
import sys, math, time, flint
from flint import arb
import sp_bvp2 as B
flint.ctx.prec = 110
tol = arb(2)**(-90)
if '--hermite' in sys.argv:
    for D in ('64',):
        Dn = float(D); xm = math.sqrt(Dn)
        P = B.Prob(D, '%.2f' % (xm + 12), '%.4f' % xm, 28)
        for zs in ('1', '1.5'):
            z = arb(zs)
            try:
                al, A = B.alpha_of_z(P, z, P.A_guess(z, arb('4.05')), arb('4.05'), tol)
                print('D=%s z=%s alpha=%s (alpha-4)*D=%s' % (D, zs, al.str(15, radius=False), ((al - 4)*arb(D)).str(6, radius=False)), flush=True)
            except Exception as ex:
                print('D=%s z=%s failed %s' % (D, zs, ex))
P = B.Prob('3', '13', '2', 24)
A, al = arb('1.7'), arb('2.07')
prev = None
for zs in ('800', '1600', '3200', '6400', '12800', '25600', '51200', '102400', '204800'):
    t = time.time()
    al, A = B.alpha_of_z(P, arb(zs), A, al, tol)
    print('D=3 z=%s alpha-2=%s [%.1fs]' % (zs, (al - 2).str(10, radius=False), time.time() - t), flush=True)

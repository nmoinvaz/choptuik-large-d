"""Large-d limit of the explicit Yang-Mills light-cone roots z_pm(d) = (1 - alpha/(beta+1))^2 of [BGW26, eq. 25]
(transcribed through a summarizing fetch of arXiv:2602.00345). Prints d (1 - z_pm) and the predicted limits
4 alpha_inf / (2/3 +- 1/sqrt3), with alpha_inf = 2 (1 +- 1/sqrt3). Control: the d = 11, 13 values of z_pm must be
roots of the printed quadratic factors 225 z^2 - 114 z + 1 and 121 z^2 - 77 z + 4.
"""
import sys
import mpmath as mp
mp.mp.dps = 40
def z(d, sgn):
    al = 2 + sgn*2*mp.sqrt(mp.mpf(d-4)/(3*(d-2)))
    be = mp.mpf(2)/3*(d-4) + sgn*mp.sqrt(3*(d-2)*(d-4))/3
    return (1 - al/(be+1))**2
ok = True
for d, q in ((11, (225, -114, 1)), (13, (121, -77, 4))):
    for sgn in (1, -1):
        r = q[0]*z(d, sgn)**2 + q[1]*z(d, sgn) + q[2]
        print(f'd={d} sign={sgn:+d} z={mp.nstr(z(d, sgn), 12)} quadratic residual={mp.nstr(r, 3)}')
        ok &= abs(r) < mp.mpf(10)**-30
for sgn in (1, -1):
    lim = 4*(1 + sgn/mp.sqrt(3))/(mp.mpf(2)/3 + sgn/mp.sqrt(3))
    vals = [mp.nstr(d*(1 - z(d, sgn)), 10) for d in (101, 1001, 10001, 100001)]
    print(f'sign={sgn:+d} d(1-z) at d=1e2..1e5:', vals, ' limit', mp.nstr(lim, 10))
print('AS EXPECTED' if ok else 'UNEXPECTED'); sys.exit(0 if ok else 1)

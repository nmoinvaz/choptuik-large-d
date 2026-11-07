import sys, time
sys.path.insert(0, sys.argv[1])
from flint import arb, ctx
from lp_core import *
ctx.prec = 128
for br in ['LP', 'Hunter']:
    r, w, (f1, g1) = sonic_series(arb('2.3'), 30, br)
    print(br, 'order-1 residuals', f1.mid().str(5), g1.mid().str(5))
    print('  |r_n|^(1/n):', [round(abs(float(r[k].mid()))**(1.0/k), 3) for k in (5, 10, 20, 29)],
          ' |w_n|^(1/n):', [round(abs(float(w[k].mid()))**(1.0/k), 3) for k in (5, 10, 20, 29)])
# wrong branch control: perturb w1, order-1 residual must be nonzero
r, w, (f1, g1) = sonic_series(arb('2.3'), 6, 'LP', d1=arb('0.01'))
print('mutated w1 residuals', f1.mid().str(5), g1.mid().str(5))
# exact solutions as checks of the integrator: SIS rho=1/y^2, w=1 ; Friedman 1/3,1/3
t0 = time.time()
rr, ww, st = integrate(arb('0.5'), arb(4), arb(1), arb('7.0'), n=32)
print('SIS: rho(7)*49 - 1 =', (rr * 49 - 1).mid().str(5), ' w-1 =', (ww - 1).mid().str(5), 'steps', st, round(time.time()-t0,2),'s')
rc, wc = origin_series(arb('0.8'), 24)
print('origin series rho:', [c.mid().str(8) for c in rc[:6]])
print('origin series w  :', [c.mid().str(8) for c in wc[:6]])

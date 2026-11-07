"""Scan f(sigma) = det[v_centre, v_sonic](y_m) on a grid; sign changes with small |f| mark eigenvalues.
usage: mode_scan.py <scriptdir> <branch> <y*> <rho0> <ym> <ya> <ds> <slo> <shi> <N>"""
import sys
sd, br, ysv, rv, ymv, yav, dsv, lo, hi, N = sys.argv[1:11]
sys.argv = ['pert.py', sd, '96', '24', br, ysv, rv, ymv, yav, dsv, '2', '3']
src = open(sd + '/pert.py').read().split('t0 = time.time()')[0]
exec(src)
lo, hi, N = float(lo), float(hi), int(N)
prev = None
for i in range(N + 1):
    s = lo + (hi - lo) * i / N
    try:
        v = float(f(arb(s))[0].mid())
    except Exception as e:
        v = float('nan')
    flag = ''
    if prev is not None and v * prev < 0:
        flag = '  <-- sign change'
    print('sigma %.4f f %+.4e%s' % (s, v, flag), flush=True)
    prev = v

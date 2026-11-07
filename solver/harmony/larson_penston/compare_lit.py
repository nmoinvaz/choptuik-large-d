"""Convert our values to Maeda-Harada's (Q0 = ln(4 pi G t^2 rho_c) = ln(2 rho0), z_s = -1/y*) and Shu's (A = 2C);
Richardson-extrapolate the far-field constants from lp_far.txt."""
import mpmath as mp, re, sys
mp.mp.dps = 50
vals = {'LP': ('2.341117280580046198337387733076470394181338481704335123', '0.83290805246924694807822742611551348265753074077239511'),
        'HunterA': ('0.73907275734887182806408141883656333460782795', '865.24391001322791598365448994145306652275')}
lit = {'LP': (0.5101, -0.4271), 'HunterA': (7.45616, -1.35305)}
for k, (ys, r0) in vals.items():
    ys, r0 = mp.mpf(ys), mp.mpf(r0)
    Q0, zs = mp.log(2 * r0), -1 / ys
    print(k, 'alpha(0) = 2 rho0 =', mp.nstr(2 * r0, 20), ' Q0 =', mp.nstr(Q0, 12), '(MH01: %s)' % lit[k][0], ' z_s =', mp.nstr(zs, 12), '(MH01: %s)' % lit[k][1])
rows = [l for l in open(sys.argv[1]) if l.startswith('y=')]
ys_, C_, U_ = [], [], []
for l in rows:
    m = re.match(r'y=1e(\d+) rho\*y\^2 = (\S+)\s+u=y\(w-1\) = (\S+)', l)
    ys_.append(mp.mpf(10) ** int(m.group(1))); C_.append(mp.mpf(m.group(2))); U_.append(mp.mpf(m.group(3)))
# C converges as 1/y^2, u as 1/y (ratios of successive differences 100 and 10)
c6, c7, c8 = C_[-3:]; u6, u7, u8 = U_[-3:]
print('diff ratios C: %s  u: %s' % (mp.nstr((c7 - c6) / (c8 - c7), 6), mp.nstr((u7 - u6) / (u8 - u7), 6)))
C = c8 + (c8 - c7) / 99; U = u8 + (u8 - u7) / 9
print('C = lim rho y^2 =', mp.nstr(C, 17), '  A = 2C =', mp.nstr(2 * C, 17), '  u_inf =', mp.nstr(U, 12))
print('spread from y=1e7 extrapolation: C %s, u %s' % (mp.nstr(abs(C - (c7 + (c7 - c6) / 99)), 3), mp.nstr(abs(U - (u7 + (u7 - u6) / 9)), 3)))

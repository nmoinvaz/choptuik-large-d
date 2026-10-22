#!/usr/bin/env python3
"""Scans for SIGMA.md: CSS solutions of the self-gravitating hedgehog in D = n + 2 and their growing modes.

    python3 sigma_runs.py d4            # D = 4: f_0 and f_1 against eta, lambda_1, gauge mode, mode count, horizons
    python3 sigma_runs.py alpha         # D = 4: eta where f_0 / f_1 first get an apparent horizon outside the SSH
    python3 sigma_runs.py dscan ETA     # f_1 continued in D at fixed eta (ETA = 0 is flat space)
    python3 sigma_runs.py hscan ETAHAT  # f_1 continued in D at fixed eta_hat = eta/(D-3)

Output: solver/matter/data/sigma_*.jsonl (one JSON object per point).  Run under nice -n 10, one process.
"""
import os
import sys
import json
import time
import numpy as np
from sigma_css import CSS, count_modes, resonances

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
os.makedirs(DATA, exist_ok=True)


def emit(fname, rec):
    with open(os.path.join(DATA, fname), 'a') as f:
        f.write(json.dumps(rec) + '\n')
    print(json.dumps(rec), flush=True)


def point(D, eta, z0, lam0=None, count=False, horizon=True, branch='f1'):
    t0 = time.time()
    S = CSS(D, eta)
    z = S.solve(z0)
    rec = dict(branch=branch, D=D, n=D - 2, eta=eta, q=z[0], c0=z[1], U1=S.U1, V1=S.V1, F=S.F, cond=S.cond,
               x0=S.x0, d1=S.d1, R=S.sR)
    if lam0 is not None:
        lam = S.find_mode(lam0)
        rec['lam'] = lam.real
        rec['lam_im'] = lam.imag
        rec['gamma'] = 1 / lam.real
        rec['RM_rel'], rec['sv_rel'] = S.mode_check(lam.real)
        g = S.find_mode(1.02)
        rec['gauge'] = g.real
    if count:
        nz, wind, poles = count_modes(S)
        rec['modes_in_box'] = nz
        rec['winding'] = wind
        rec['poles'] = poles
    if horizon:
        rec['xB'] = S.beyond()
    rec['sec'] = time.time() - t0
    return S, z, rec


ETAS4 = [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.2, 0.25, 0.3,
         0.35, 0.4, 0.42, 0.44]


def run_d4():
    out = 'sigma_d4.jsonl'
    # f_1 from flat space, then eta upward
    z = (21.75741, 1.0, -0.30566)
    lam = 6.3336
    for eta in ETAS4:
        try:
            S, z, rec = point(4.0, eta, z, lam0=lam, count=(eta in (0.0, 0.1, 0.14, 0.17, 0.3)))
        except Exception as e:  # noqa
            emit(out, dict(branch='f1', D=4.0, eta=eta, error=repr(e)))
            break
        lam = rec['lam']
        emit(out, rec)
    run_d4_f0()


def run_d4_f0():
    out = 'sigma_d4.jsonl'
    z = (2.0, 1.0, 1.0)
    for eta in ETAS4:
        try:
            S, z, rec = point(4.0, eta, z, lam0=None, count=(eta in (0.0, 0.1, 0.3)), branch='f0')
            rec['gauge'] = S.find_mode(1.02).real
        except Exception as e:  # noqa
            emit(out, dict(branch='f0', D=4.0, eta=eta, error=repr(e)))
            break
        emit(out, rec)


def bisect_horizon(D, z, eta_lo, eta_hi, branch, tol=1e-5):
    """eta at which the branch's solution first has an apparent horizon outside the SSH (x_B finite)."""
    S = CSS(D, eta_lo); z_lo = S.solve(z); h_lo = S.beyond()
    S = CSS(D, eta_hi); z_hi = S.solve(z_lo); h_hi = S.beyond()
    print(branch, eta_lo, h_lo, eta_hi, h_hi, flush=True)
    assert (h_lo is None) != (h_hi is None)
    while eta_hi - eta_lo > tol:
        em = 0.5 * (eta_lo + eta_hi)
        S = CSS(D, em); zm = S.solve(z_lo); hm = S.beyond()
        print('   ', em, hm, flush=True)
        if (hm is None) == (h_lo is None):
            eta_lo, z_lo, h_lo = em, zm, hm
        else:
            eta_hi, z_hi, h_hi = em, zm, hm
    return 0.5 * (eta_lo + eta_hi)


def run_alpha():
    out = 'sigma_d4_horizon.jsonl'
    a0 = bisect_horizon(4.0, (2.0, 1.0, 1.0), 0.05, 0.1, 'f0')
    emit(out, dict(branch='f0', D=4.0, eta_horizon=a0, ref='Bizon-Wasserman alpha_0 ~ 0.0688'))
    a1 = bisect_horizon(4.0, (19.0, 1.0, -0.34), 0.12, 0.2, 'f1')
    emit(out, dict(branch='f1', D=4.0, eta_horizon=a1, ref='Bizon-Wasserman alpha_1 ~ 0.1518'))


def continue_D(eta_of_D, Ds, out, tag, z_start, lam_start, D_start=4.0, count_every=4):
    """Continue f_1 in D from the D = 4 solution.  At D = 4 the SSH datum is U'(1) (U(1) = pi/2); above, U(1)."""
    S = CSS(D_start, eta_of_D(D_start))
    z = S.solve(z_start)
    lam = S.find_mode(lam_start).real
    n0 = D_start - 2
    a2 = (n0 - 1) / (n0 - 1 - 2 * eta_of_D(D_start))
    hist = []
    for i, D in enumerate(Ds):
        n = D - 2
        if not hist:          # leave the resonant parametrisation at D = 4: U(1) = pi/2 - (n-2) U'(1)/(n a^2)
            zg = (z[0], z[1], np.pi / 2 - (n - 2) * z[2] / (n * a2))
        elif len(hist) == 1:
            zg = hist[-1][1]
        else:
            (D1, z1, l1), (D2, z2, l2) = hist[-2], hist[-1]
            zg = z2 + (np.array(z2) - np.array(z1)) * (D - D2) / (D2 - D1)
            lam = l2 + (l2 - l1) * (D - D2) / (D2 - D1)
        try:
            S, z, rec = point(D, eta_of_D(D), zg, lam0=lam, count=(i % count_every == 0), horizon=True)
        except Exception as e:  # noqa
            emit(out, dict(tag=tag, D=D, eta=eta_of_D(D), error=repr(e)))
            break
        rec['tag'] = tag
        rec['poles_near'] = resonances(n, rec['lam'] - 0.5, rec['lam'] + 0.5)
        emit(out, rec)
        if not np.isfinite(rec['lam']) or rec['F'] > 1e-8:
            break
        hist.append((D, z, rec['lam']))
    return hist


def follow_path(nodes, h, out, tag, emit_every=1, count_every=8, z_start=(21.75741, 1.0, -0.30566), lam_start=6.3336):
    """Follow f_1 along a piecewise-linear path in (D, eta) starting at D = 4, eta = 0 (flat f_1).  Even-n D are
    shifted by 1e-3 (resonant SSH series).  Steps of length ~h in D (or in eta along a fixed-D leg)."""
    pts = []
    for (D0, e0), (D1, e1) in zip(nodes[:-1], nodes[1:]):
        L = max(abs(D1 - D0), abs(e1 - e0) * 4)
        m = max(1, int(round(L / h)))
        for k in range(1, m + 1):
            pts.append((D0 + (D1 - D0) * k / m, e0 + (e1 - e0) * k / m))
    S = CSS(4.0, 0.0)
    z = S.solve(z_start)
    lam = S.find_mode(lam_start).real
    hist = [(4.0, 0.0, np.array(z), lam)]
    z0b = np.array([2.0, 1.0, 1.0])            # f_0, followed along the same path (its horizon decides the role of f_1)
    f0_ok = True
    for i, (D, eta) in enumerate(pts):
        Dr = D
        if abs(D - round(D)) < 1e-9 and round(D) % 2 == 0 and round(D) != 4:
            Dr = D + 1e-3
        n = Dr - 2
        prevD, preve, pz, pl = hist[-1]
        if abs(prevD - 4.0) < 1e-12 and abs(Dr - 4.0) > 1e-12:     # leave the D = 4 parametrisation (U(1) = pi/2)
            a2 = 1.0 / (1 - 2 * preve)
            zg = np.array([pz[0], pz[1], np.pi / 2 - (n - 2) * pz[2] / (n * a2)])
            lg = pl
        elif len(hist) >= 2 and (abs(hist[-2][0] - 4.0) > 1e-12 or abs(Dr - 4.0) < 1e-12):
            D1_, e1_, z1, l1 = hist[-2]
            d0 = np.hypot(prevD - D1_, preve - e1_)
            d1 = np.hypot(Dr - prevD, eta - preve)
            r = d1 / d0 if d0 > 0 else 0.0
            zg = pz + (pz - z1) * r
            lg = pl + (pl - l1) * r
        else:
            zg, lg = pz, pl
        try:
            want = (i % emit_every == 0) or i == len(pts) - 1
            S = CSS(Dr, eta)
            z = S.solve(zg)
            if S.F > 1e-8:
                z = S.solve(pz)
            lam = S.find_mode(lg).real
            if want:
                rec = dict(tag=tag, branch='f1', D=Dr, n=n, eta=eta, etahat=eta / (n - 1), q=z[0], c0=z[1], U1=S.U1,
                           V1=S.V1, F=S.F, cond=S.cond, lam=lam, gamma=1 / lam, A=z[0] * np.sqrt(n))
                rec['RM_rel'], rec['sv_rel'] = S.mode_check(lam)
                rec['gauge'] = S.find_mode(1.02).real
                if i % count_every == 0:
                    rec['modes_in_box'], rec['winding'], rec['poles'] = count_modes(S)
                rec['xB'] = S.beyond()
            if f0_ok:
                try:
                    pD = hist[-1][0]
                    if abs(pD - 4.0) < 1e-12 and abs(Dr - 4.0) > 1e-12:
                        a2b = 1.0 / (1 - 2 * hist[-1][1])
                        z0b = np.array([z0b[0], z0b[1], np.pi / 2 - (n - 2) * z0b[2] / (n * a2b)])
                    S0 = CSS(Dr, eta)
                    z0b = S0.solve(z0b)
                    if want:
                        rec['f0_q'], rec['f0_c0'], rec['f0_U1'], rec['f0_F'] = z0b[0], z0b[1], S0.U1, S0.F
                        rec['f0_xB'] = S0.beyond()
                    if S0.F > 1e-8:
                        f0_ok = False
                except Exception as e:  # noqa
                    f0_ok = False
                    if want:
                        rec['f0_error'] = repr(e)
            if want:
                emit(out, rec)
        except Exception as e:  # noqa
            emit(out, dict(tag=tag, D=Dr, eta=eta, error=repr(e)))
            break
        hist.append((Dr, eta, np.array(z), lam))
    return hist


def horizon_threshold(D, z, eta0, step, branch, out, xmax=1e8, tol=2e-5, maxsteps=200):
    """Starting from the solution z at (D, eta0), step eta (sign of step) until the apparent-horizon status changes,
    then bisect.  Records every (eta, x_B) visited (for the ln x_B = K/(eta - alpha) + c extrapolation)."""
    pts = []
    S = CSS(D, eta0); z = S.solve(z); h = S.beyond(xmax); pts.append((eta0, h))
    e, zprev, hprev = eta0, z, h
    for _ in range(maxsteps):
        en = e + step
        if en <= 0 or 2 * en >= (D - 3) * 0.999:
            break
        S = CSS(D, en); zn = S.solve(zprev)
        if S.F > 1e-8:
            break
        hn = S.beyond(xmax); pts.append((en, hn))
        if (hn is None) != (hprev is None):
            lo, hi, zlo = (e, en, zprev)
            hlo = hprev
            while abs(hi - lo) > tol:
                m = 0.5 * (lo + hi)
                S = CSS(D, m); zm = S.solve(zlo); hm = S.beyond(xmax); pts.append((m, hm))
                if (hm is None) == (hlo is None):
                    lo, zlo = m, zm
                else:
                    hi = m
            rec = dict(branch=branch, D=D, eta_horizon=0.5 * (lo + hi), xmax=xmax,
                       pts=[(p, None if q is None else q) for p, q in pts])
            emit(out, rec)
            return rec
        e, zprev, hprev = en, zn, hn
    rec = dict(branch=branch, D=D, eta_horizon=None, pts=pts)
    emit(out, rec)
    return rec


def f0_scan(eta, Ds, out):
    """The ground state f_0 continued in D at fixed eta (start: D = 4); amplitude A = sqrt(n) U'(0) for comparison with
    the D = infinity inner problem (sigma_inner.py)."""
    S = CSS(4.0, 0.0)
    z = S.solve((2.0, 1.0, 1.0))
    for e in np.linspace(0, eta, 6)[1:]:
        S = CSS(4.0, e)
        z = S.solve(z)
    prev = (4.0, np.array(z))
    hist = []
    for D in Ds:
        n = D - 2
        if prev[0] == 4.0:
            a2 = 1.0 / (1 - 2 * eta)
            zg = np.array([z[0], z[1], np.pi / 2 - (n - 2) * z[2] / (n * a2)])
        elif len(hist) >= 2:
            (D1, z1), (D2, z2) = hist[-2], hist[-1]
            zg = z2 + (z2 - z1) * (D - D2) / (D2 - D1)
        else:
            zg = prev[1]
        try:
            S = CSS(D, eta)
            z = S.solve(zg)
        except Exception as ex:  # noqa
            emit(out, dict(branch='f0', D=D, eta=eta, error=repr(ex)))
            break
        rec = dict(branch='f0', D=D, n=n, eta=eta, q=z[0], c0=z[1], U1=S.U1, V1=S.V1, F=S.F, cond=S.cond,
                   A=z[0] * np.sqrt(n), xB=S.beyond())
        emit(out, rec)
        if S.F > 1e-8:
            break
        prev = (D, np.array(z))
        hist.append(prev)


def follow_from(D0, eta, z0, lam0, D1, step, out, tag, count_every=6, eta_of_D=None):
    """Continue f_1 in D from a solution at D0 (non-resonant parametrisation, s1 = U(1)); eta fixed, or eta_of_D."""
    etaf = eta_of_D or (lambda D: eta)
    S = CSS(D0, etaf(D0))
    z = S.solve(z0)
    lam = S.find_mode(lam0).real
    hist = [(D0, np.array(z), lam)]
    Ds = list(np.arange(D0 + step, D1 + 1e-9, step)) if step > 0 else list(np.arange(D0 + step, D1 - 1e-9, step))
    for i, D in enumerate(Ds):
        Dr = D + 1e-3 if abs(D - round(D)) < 2e-3 and round(D) % 2 == 0 else D
        if len(hist) >= 2:
            (Da, za, la), (Db, zb, lb) = hist[-2], hist[-1]
            r = (Dr - Db) / (Db - Da)
            zg, lg = zb + (zb - za) * r, lb + (lb - la) * r
        else:
            zg, lg = hist[-1][1], hist[-1][2]
        try:
            S = CSS(Dr, etaf(Dr))
            z = S.solve(zg)
            if S.F > 1e-8:
                z = S.solve(hist[-1][1])
            lam = S.find_mode(lg).real
            rec = dict(tag=tag, branch='f1', D=Dr, n=Dr - 2, eta=etaf(Dr), q=z[0], c0=z[1], U1=S.U1, V1=S.V1, F=S.F,
                       cond=S.cond, lam=lam, gamma=1 / lam, A=z[0] * np.sqrt(Dr - 2))
            rec['RM_rel'], rec['sv_rel'] = S.mode_check(lam)
            rec['gauge'] = S.find_mode(1.02).real
            if i % count_every == 0:
                rec['modes_in_box'], rec['winding'], rec['poles'] = count_modes(S)
            rec['xB'] = S.beyond()
            emit(out, rec)
        except Exception as e:  # noqa
            emit(out, dict(tag=tag, D=Dr, eta=etaf(Dr), error=repr(e)))
            break
        if S.F > 1e-8:
            break
        hist.append((Dr, np.array(z), lam))
    return hist


if __name__ == '__main__':
    what = sys.argv[1]
    if what == 'from':
        # python3 sigma_runs.py from NAME D0 ETA q c0 s1 lam0 D1 step
        name = sys.argv[2]
        D0, eta = float(sys.argv[3]), float(sys.argv[4])
        z0 = tuple(float(v) for v in sys.argv[5:8])
        lam0, D1, step = float(sys.argv[8]), float(sys.argv[9]), float(sys.argv[10])
        follow_from(D0, eta, z0, lam0, D1, step, 'sigma_from_%s.jsonl' % name, name)
    elif what == 'f0scan':
        # python3 sigma_runs.py f0scan ETA DMAX STEP [DSTART]  (a DSTART off the integers avoids the resonant SSH series)
        eta, Dmax, step = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
        D0 = float(sys.argv[5]) if len(sys.argv) > 5 else 4.0 + step
        Ds = [D + 1e-3 if abs(D - round(D)) < 1e-9 and round(D) % 2 == 0 else D
              for D in np.round(np.arange(D0, Dmax + 1e-9, step), 6)]
        f0_scan(eta, Ds, 'sigma_f0scan_eta%.4f.jsonl' % eta)
    elif what == 'hthr':
        # python3 sigma_runs.py hthr D branch q c0 s1 eta0 step
        D, br = float(sys.argv[2]), sys.argv[3]
        z = tuple(float(v) for v in sys.argv[4:7])
        horizon_threshold(D, z, float(sys.argv[7]), float(sys.argv[8]), br, 'sigma_horizon.jsonl')
    elif what == 'path':
        # python3 sigma_runs.py path NAME h D1 eta1 D2 eta2 ...   (starts at D = 4, eta = 0)
        name, h = sys.argv[2], float(sys.argv[3])
        vals = [float(v) for v in sys.argv[4:]]
        nodes = [(4.0, 0.0)] + list(zip(vals[0::2], vals[1::2]))
        follow_path(nodes, h, 'sigma_path_%s.jsonl' % name, name)
    elif what == 'd4':
        run_d4()
    elif what == 'd4f0':
        run_d4_f0()
    elif what == 'alpha':
        run_alpha()
    elif what == 'dscan':
        eta = float(sys.argv[2])
        Dmax = float(sys.argv[3]) if len(sys.argv) > 3 else 12.0
        step = float(sys.argv[4]) if len(sys.argv) > 4 else 0.125
        Ds = list(np.round(np.arange(4.0 + step, Dmax + 1e-9, step), 6))
        Ds = [D + 1e-3 if abs(D - round(D)) < 1e-9 and round(D) % 2 == 0 else D for D in Ds]   # even n: resonant SSH
        z0 = (21.75741, 1.0, -0.30566)
        S = CSS(4.0, 0.0); z0 = S.solve(z0)
        for e in np.linspace(0, eta, 6)[1:]:
            S = CSS(4.0, e); z0 = S.solve(z0)
        lam0 = S.find_mode(6.0 if eta < 0.1 else 5.4).real
        continue_D(lambda D: eta, Ds, 'sigma_dscan_eta%.4f.jsonl' % eta, 'eta=%g' % eta, z0, lam0)
    elif what == 'hscan':
        eh = float(sys.argv[2])
        Dmax = float(sys.argv[3]) if len(sys.argv) > 3 else 12.0
        step = float(sys.argv[4]) if len(sys.argv) > 4 else 0.125
        Ds = list(np.round(np.arange(4.0 + step, Dmax + 1e-9, step), 6))
        Ds = [D + 1e-3 if abs(D - round(D)) < 1e-9 and round(D) % 2 == 0 else D for D in Ds]
        z0 = (21.75741, 1.0, -0.30566)
        S = CSS(4.0, 0.0); z0 = S.solve(z0)
        for e in np.linspace(0, eh, 6)[1:]:
            S = CSS(4.0, e); z0 = S.solve(z0)
        lam0 = S.find_mode(6.0 if eh < 0.1 else 5.4).real
        continue_D(lambda D: eh * (D - 3), Ds, 'sigma_hscan_eh%.4f.jsonl' % eh, 'etahat=%g' % eh, z0, lam0)

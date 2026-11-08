#!/usr/bin/env python3
"""The free complex scalar's CSS (Hirschmann-Eardley, kappa = 0) in D: its growth rates as zeros of det M(lambda)
(css_pert.py, complex-path shooting), followed in D and counted by winding numbers (matter/COMPLEX_CRITICAL.md).

    python3 complex_css.py root D LAM [--R=0.85,0.6]                  secant root of det M from LAM, at each radius
    python3 complex_css.py follow DLIST LAM1[,LAM2..] [--R=0.85,0.6] [--out=F]
                                                                     each root followed through the backgrounds DLIST
    python3 complex_css.py track D0 D1 LAM1[,LAM2..] [--h=0.1] [--hmax=1] [--spread=1e-8] [--out=F]
                                                                     background and roots continued in D (adaptive)
    python3 complex_css.py count D RE0 RE1 IM0 IM1 [--R=0.85] [--plant=Z] [--mutate=M] [--K=12] [--out=F]
                                                                     zeros of det M in RE0 < Re < RE1, IM0 < Im < IM1
    python3 complex_css.py boxes D LAMOSC [--R=0.85,0.6] [--names=A,B,C,S] [--plant=Z] [--mutate=M] [--out=F]
                                                                     the standard boxes A, B, C, S at each radius
    python3 complex_css.py selftest                                  the winding code on a synthetic function

Backgrounds: data/css_bg_k0.jsonl (css_lateral.py cont 0 ...), --bg for another file.  Growth rates are zeros of
det M for every circle radius R = RFRAC x_h; the jump method's artefacts move with R (matter/CSS.md section 6), so a
root is accepted as a growth rate when the secant iteration from it converges to the same lambda at both radii.

Controls (AGENTS.md "every check must be able to fail"):
  --plant=Z    multiplies det M by (lambda - Z): a box holding Z must count one more.
  --mutate=nometric   the perturbation's metric (db, du) is not fed back into the wave equations (a test field on the
               frozen CSS); --mutate=hamsign flips the sign of the Hamiltonian constraint's matter source.  Either
               must change the count of a box holding a growth rate.
  selftest     a synthetic det with known zeros (one 0.02 inside an edge) through the same winding code; with one
               sample per edge and no refinement it must fail (exit 1 if the control does not fire).
"""
import json
import os
import sys
import time

import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import css_lateral as cl  # noqa: E402
import css_pert as cp  # noqa: E402

BG = os.path.join(HERE, "data", "css_bg_k0.jsonl")


def build_mut(mutate):
    """css_pert.build with one deliberate change (the controls); mutate None gives css_pert.build's equations."""
    x, lam, om, n, kap, eps = sp.symbols('x lam om n kap eps')
    f, g, q, qg, f2, g2, b0, b1, u0, P0, Pg0, P0x, Pg0x = sp.symbols('f g q qg f2 g2 b0 b1 u0 P0 Pg0 P0x Pg0x')
    A, A1, B, B1, db, du = sp.symbols('A A1 B B1 db du')
    unk = sp.symbols('Pa Pb db1 du1 Pa1 Pb1 A2 B2')
    Pa, Pb, db1, du1, Pa1, Pb1, A2, B2 = unk
    I = sp.I
    met = 0 if mutate == "nometric" else 1
    hs = -1 if mutate == "hamsign" else 1
    psi, psib = f + eps * A, g + eps * B
    psx, psbx = q + eps * A1, qg + eps * B1
    psxx, psbxx = f2 + eps * A2, g2 + eps * B2
    P, Pbar = P0 + eps * Pa, Pg0 + eps * Pb
    Px, Pbx = P0x + eps * Pa1, Pg0x + eps * Pb1
    u, b, bx = u0 + met * eps * du, b0 + met * eps * db, b1 + met * eps * db1
    rho = 1 - kap * psi * psib
    h = 1 / rho ** 2
    lin = lambda e: sp.diff(e, eps).subs(eps, 0)
    eqs = []
    e1 = b * P + I * om * psi - x * psx
    e1b = b * Pbar - I * om * psib - x * psbx
    eqs.append(lin(e1) - lam * A)
    eqs.append(lin(e1b) - lam * B)
    eqs.append(lin(bx * P + b * Px + I * om * psx - psx - x * psxx) - lam * A1)
    eqs.append(lin(bx * Pbar + b * Pbx - I * om * psbx - psbx - x * psbxx) - lam * B1)
    e2 = (I * om - 1) * P - x * Px + b * psxx + (n * b / x + bx) * psx + b * (2 * kap * psib / rho) * (psx ** 2 - P ** 2)
    e2b = (-I * om - 1) * Pbar - x * Pbx + b * psbxx + (n * b / x + bx) * psbx + b * (2 * kap * psi / rho) * (psbx ** 2 - Pbar ** 2)
    eqs.append(lin(e2) - lam * Pa)
    eqs.append(lin(e2b) - lam * Pb)
    # the db, du rows stay in the system (so the matrix is the same size); with nometric they are not fed back
    ub, bb = u0 + eps * du, b0 + eps * db
    eqs.append(lin((n - 1) * bb * ub / x) - db1)
    ham = (1 + ub) * (hs * (2 * x / n) * h * (P * Pbar + psx * psbx) - (n - 1) * ub / x)
    eqs.append(lin(ham) - du1)
    Mx, rhs = sp.linear_eq_to_matrix(eqs, list(unk))
    args = (x, lam, om, n, kap, f, g, q, qg, f2, g2, b0, b1, u0, P0, Pg0, P0x, Pg0x, A, A1, B, B1, db, du)
    return sp.lambdify(args, Mx, 'numpy'), sp.lambdify(args, rhs, 'numpy')


def set_mutation(mutate):
    if mutate:
        cp._FM, cp._FR = build_mut(mutate)
        cp._FMOM = None
    else:
        cp.funcs()


def load_bg(path=BG):
    return {r["D"]: r for r in (json.loads(l) for l in open(path))}


class Det:
    """det M(lambda) for one background, one radius, with an optional planted factor; evaluations cached."""

    def __init__(self, r, Rf, plant=None):
        self.r, self.Rf, self.plant = r, Rf, plant
        self.n = r["D"] - 2
        self.xh = r["X"]
        self.cache = {}
        self.nev = 0

    def __call__(self, lam):
        key = (round(lam.real, 14), round(lam.imag, 14))
        if key not in self.cache:
            r = self.r
            d = cp.detM(self.n, r["kappa"], r["omega"], r["f0"], lam, self.xh, self.Rf * self.xh)
            if self.plant is not None:
                d *= (lam - self.plant)
            self.cache[key] = d
            self.nev += 1
        return self.cache[key]


def winding(F, corners, K=12, tol=0.5, maxdepth=30):
    """(1/2 pi) x the change of arg F around the closed polygon `corners` (counterclockwise), tracked on K points per
    edge, every interval bisected until its phase step is below tol rad and its two halves agree with the whole.
    Returns (winding, largest step, number of intervals)."""
    total, big, nint = 0.0, 0.0, 0
    pts = list(corners) + [corners[0]]
    for a, c in zip(pts[:-1], pts[1:]):
        def val(t):
            return F(a + (c - a) * t)

        def incr(t0, t1, depth=0):
            nonlocal big, nint
            d = np.angle(val(t1) / val(t0))
            tm = 0.5 * (t0 + t1)
            d1 = np.angle(val(tm) / val(t0))
            d2 = np.angle(val(t1) / val(tm))
            if (abs(d1 + d2 - d) > 1e-6 or abs(d) > tol) and depth < maxdepth:
                return incr(t0, tm, depth + 1) + incr(tm, t1, depth + 1)
            big = max(big, abs(d1 + d2))
            nint += 1
            return d1 + d2
        ts = np.linspace(0, 1, K + 1)
        for t0, t1 in zip(ts[:-1], ts[1:]):
            total += incr(t0, t1)
    return total / (2 * np.pi), big, nint


def detrended(det, bx):
    """det M exp(-q), q the quadratic in mu = lam - (box centre) whose real part fits ln|det M| at the box's corners
    and edge midpoints (5 real parameters, 8 points): exp(-q) has no zeros, so the winding is unchanged, but most of
    det M's phase drift along the edges (its growth in lam) goes, and with it most of the bisections."""
    re0, re1, im0, im1 = bx
    c = complex(0.5 * (re0 + re1), 0.5 * (im0 + im1))
    pts = [complex(a, b) for a in (re0, 0.5 * (re0 + re1), re1) for b in (im0, 0.5 * (im0 + im1), im1)
           if not (a == 0.5 * (re0 + re1) and b == 0.5 * (im0 + im1))]
    mu = np.array(pts) - c
    A = np.stack([np.ones(len(mu)), mu.real, -mu.imag, (mu * mu).real, -(mu * mu).imag], axis=1)
    y = np.array([np.log(abs(det(p))) for p in pts])
    k = np.linalg.lstsq(A, y, rcond=None)[0]
    c0, c1, c2 = complex(k[0], 0), complex(k[1], k[2]), complex(k[3], k[4])

    def F(lam):
        m = lam - c
        return np.exp(np.log(det(lam)) - (c0 + c1 * m + c2 * m * m))
    F.trend = [k.tolist()]
    return F


def box(re0, re1, im0, im1):
    return [complex(re0, im0), complex(re1, im0), complex(re1, im1), complex(re0, im1)]


def opts_of(argv):
    pos, o = [], {}
    for a in argv:
        if a.startswith("--"):
            k, _, v = a[2:].partition("=")
            o[k] = v
        else:
            pos.append(a)
    return pos, o


def secant(det, lam0, tol=1e-11, maxit=40):
    l0 = complex(lam0)
    l1 = l0 * (1 + 1e-4) + 1e-5
    d0, d1 = det(l0), det(l1)
    for _ in range(maxit):
        l2 = l1 - d1 * (l1 - l0) / (d1 - d0)
        if abs(l2 - l1) < tol * max(1, abs(l2)):
            return l2, True
        l0, d0 = l1, d1
        l1, d1 = l2, det(l2)
    return l2, False


def main():
    pos, o = opts_of(sys.argv[1:])
    bg = load_bg(o.get("bg", BG))
    mutate = o.get("mutate") or None
    set_mutation(mutate)
    out = o.get("out")

    def emit(rec):
        print(json.dumps(rec), flush=True)
        if out:
            with open(out, "a") as fo:
                fo.write(json.dumps(rec) + "\n")

    if pos[0] == "selftest":
        zs = [0.3 + 0.4j, 0.7 - 0.2j, 0.98 + 0.5j, 1.02 + 0.1j]   # three inside the unit box, one 0.02 outside

        def F(l):
            return np.exp(30 * l + 2j * l * l) * np.prod([l - z for z in zs])
        Fd = detrended(F, (0, 1, -0.6, 0.6))           # the path the counts take
        w, big, ni = winding(Fd, box(0, 1, -0.6, 0.6), K=6)
        wc, bigc, nic = winding(F, box(0, 1, -0.6, 0.6), K=1, tol=10.0, maxdepth=0)
        ok = abs(w - 3) < 1e-6 and big < 0.5
        ctl = abs(wc - 3) > 1e-6
        print("selftest: winding %.6f (expect 3, largest step %.2f, %d intervals); control (no refinement) %.6f"
              % (w, big, ni, wc))
        print("selftest %s, control %s" % ("PASS" if ok else "FAIL", "fires" if ctl else "DOES NOT FIRE"))
        sys.exit(0 if ok and ctl else 1)

    if pos[0] == "root":
        D = float(pos[1])
        lam0 = complex(pos[2].replace(" ", ""))
        for Rf in [float(s) for s in o.get("R", "0.85,0.6").split(",")]:
            det = Det(bg[D], Rf)
            t = time.time()
            lam, ok = secant(det, lam0)
            emit(dict(kind="root", D=D, R_over_xh=Rf, lam=[lam.real, lam.imag], converged=ok, start=[lam0.real, lam0.imag],
                      evaluations=det.nev, seconds=round(time.time() - t, 1), mutate=mutate))
        return

    if pos[0] == "follow":
        Ds = [float(s) for s in pos[1].split(",")]
        lams = [complex(s.replace(" ", "")) for s in pos[2].split(",")]
        Rfs = [float(s) for s in o.get("R", "0.85,0.6").split(",")]
        hist = {i: [] for i in range(len(lams))}
        for D in Ds:
            r = bg[D]
            for i, l0 in enumerate(lams):
                h = hist[i]
                if len(h) >= 2:
                    (Da, la), (Db, lb) = h[-2], h[-1]
                    guess = lb + (lb - la) * (D - Db) / (Db - Da)
                elif h:
                    guess = h[-1][1]
                else:
                    guess = l0
                res = []
                for Rf in Rfs:
                    det = Det(r, Rf)
                    lam, ok = secant(det, guess)
                    res.append((Rf, lam, ok, det.nev))
                lam = res[0][1]
                spread = max(abs(x[1] - lam) for x in res)
                h.append((D, lam))
                emit(dict(kind="follow", D=D, mode=i, guess=[guess.real, guess.imag], lam=[lam.real, lam.imag],
                          radii=[[x[0], x[1].real, x[1].imag, x[2]] for x in res], radius_spread=spread,
                          omega=r["omega"], resonance_Im=r["kappa_s"] * r["mu"][1], mutate=mutate))
        return

    if pos[0] == "track":
        # complex_css.py track D0 D1 LAM1[,LAM2..] [--h=0.1] [--hmax=1] [--spread=1e-8] [--out=F]: the background continued in D by
        # Newton on its jump (css_lateral.newton), each root followed by secant at two radii; a step is accepted when
        # the background converges, every root converges at both radii to the same lambda (--spread) and lies within
        # 0.02 (1 + |lambda|) of its extrapolation; otherwise the step is halved.
        D0, D1 = float(pos[1]), float(pos[2])
        lams = [complex(s.replace(" ", "")) for s in pos[3].split(",")]
        h0 = float(o.get("h", "0.1"))
        hmax = float(o.get("hmax", "1.0"))
        Rfs = [float(s) for s in o.get("R", "0.85,0.6").split(",")]
        smax = float(o.get("spread", "1e-8"))
        r = bg[D0]
        kap = r["kappa"]
        bh = [(D0, r["omega"], r["f0"])]
        lh = [[(D0, l)] for l in lams]
        D, h = D0, h0
        sgn = 1 if D1 > D0 else -1

        def extrap(hist, Dn):
            if len(hist) >= 3:
                Ds = np.array([q[0] for q in hist[-3:]])
                V = np.array([q[1] for q in hist[-3:]])
                c = np.polyfit(Ds - Ds[-1], V, 2)
                return np.polyval(c, Dn - Ds[-1])
            if len(hist) == 2:
                (Da, va), (Db, vb) = hist
                return vb + (vb - va) * (Dn - Db) / (Db - Da)
            return hist[-1][1]
        while sgn * (D1 - D) > 1e-12:
            Dn = D + sgn * min(h, abs(D1 - D))
            om_g = float(np.real(extrap([(q[0], q[1]) for q in bh], Dn)))
            f_g = float(np.real(extrap([(q[0], q[2]) for q in bh], Dn)))
            ok = True
            try:
                xg = cl.horizon_estimate(Dn - 2, kap, om_g, f_g)
                p, aJ, _ = cl.newton(Dn - 2, kap, om_g, f_g, 0.85 * xg, verbose=False)
                ok = aJ < 1e-10 and abs(p[0] - om_g) < 0.05 * abs(om_g) + 0.02
                xh = cl.horizon_estimate(Dn - 2, kap, p[0], p[1])
            except Exception:
                ok = False
            new = []
            if ok:
                rn = dict(D=Dn, kappa=kap, omega=p[0], f0=p[1], X=xh)
                for i, hist in enumerate(lh):
                    g = complex(extrap(hist, Dn))
                    res = []
                    for Rf in Rfs:
                        det = Det(rn, Rf)
                        lam, conv = secant(det, g)
                        res.append((Rf, lam, conv))
                    lam = res[0][1]
                    spread = max(abs(q[1] - lam) for q in res)
                    good = all(q[2] for q in res) and spread < smax and abs(lam - g) < 0.02 * (1 + abs(lam))
                    if not good:
                        ok = False
                        break
                    new.append((lam, spread, g))
            if not ok:
                h *= 0.5
                if h < 1e-3:
                    emit(dict(kind="track_stop", D=D, reason="step below 1e-3", mutate=mutate))
                    break
                continue
            bh.append((Dn, p[0], p[1]))
            for i, (lam, spread, g) in enumerate(new):
                lh[i].append((Dn, lam))
                emit(dict(kind="track", D=Dn, mode=i, lam=[lam.real, lam.imag], guess=[g.real, g.imag],
                          radius_spread=spread, omega=p[0], f0=p[1], X=xh, absJ=aJ, mutate=mutate))
            D = Dn
            h = min(hmax, 1.5 * h)
        return

    if pos[0] == "count":
        D = float(pos[1])
        re0, re1, im0, im1 = [float(s) for s in pos[2:6]]
        Rf = float(o.get("R", "0.85"))
        plant = complex(o["plant"].replace(" ", "")) if "plant" in o else None
        K = int(o.get("K", "6"))
        r = bg[D]
        det = Det(r, Rf, plant)
        t = time.time()
        w, big, ni = winding(detrended(det, (re0, re1, im0, im1)), box(re0, re1, im0, im1), K=K)
        emit(dict(kind="count", D=D, box=[re0, re1, im0, im1], R_over_xh=Rf, winding=w, largest_step=big,
                  intervals=ni, evaluations=det.nev, seconds=round(time.time() - t, 1), plant=None if plant is None
                  else [plant.real, plant.imag], mutate=mutate, omega=r["omega"],
                  resonance_Im=r["kappa_s"] * r["mu"][1]))
        return

    if pos[0] == "locate":
        # complex_css.py locate D RE0 RE1 IM0 IM1 [--w=0.5] [--R=0.85,0.6] [--out=F]
        # the zeros in a box, located and tested against the second radius: the box is cut into sub-boxes of width w
        # in Re; each is counted at the first radius; in a sub-box with k >= 1 zeros the secant iteration starts from a
        # 4 x 3 grid and keeps the distinct converged roots inside it; each root is then refined at every other radius.
        # A growth rate converges to the same lambda at every radius; an artefact of the jump method moves (CSS.md 6).
        D = float(pos[1])
        re0, re1, im0, im1 = [float(s) for s in pos[2:6]]
        w = float(o.get("w", "0.5"))
        Rfs = [float(s) for s in o.get("R", "0.85,0.6").split(",")]
        r = bg[D]
        det1 = Det(r, Rfs[0])
        a = re0
        while a < re1 - 1e-12:
            b = min(a + w, re1)
            bx = (a, b, im0, im1)
            wnd, big, ni = winding(detrended(det1, bx), box(*bx), K=4)
            k = int(round(wnd))
            roots = []
            if k >= 1:
                for sa in np.linspace(a, b, 6)[1:-1]:
                    for sb in np.linspace(im0, im1, 5)[1:-1]:
                        lam, ok = secant(det1, complex(sa, sb))
                        if ok and a < lam.real < b and im0 < lam.imag < im1 and all(abs(lam - q) > 1e-6 for q in roots):
                            roots.append(lam)
            other = []
            for lam in roots:
                moved = []
                for Rf in Rfs[1:]:
                    l2, ok2 = secant(Det(r, Rf), lam)
                    moved.append([Rf, l2.real, l2.imag, bool(ok2), abs(l2 - lam)])
                other.append(moved)
            emit(dict(kind="locate", D=D, subbox=list(bx), R_over_xh=Rfs[0], winding=wnd, largest_step=big,
                      roots=[[q.real, q.imag] for q in roots], at_other_radii=other, omega=r["omega"],
                      mutate=mutate))
            a = b
        return

    if pos[0] == "boxes":
        # complex_css.py boxes D LAMOSC [--R=0.85,0.6] [--names=A,B,C,S] [--plant=Z] [--mutate=M] [--out=F]
        # the standard boxes of matter/COMPLEX_CRITICAL.md section 2 (omega from the background, eps = 0.25):
        #   A  0.05 < Re < 0.9, |Im| < omega - eps            expected: none
        #   B  0.9 < Re < RE1, |Im| < omega - eps             expected: lambda = 1 and lambda_phys
        #   C  0.05 < Re < RE1, omega + eps < Im < Im(osc) + 1.5   expected: the oscillatory mode
        #   S  0.05 < Re < RE1, omega - eps < Im < omega + eps  (the resonance line: the jump method's artefacts)
        # RE1 = 5.  The oscillatory mode is first refined by secant from LAMOSC.
        D = float(pos[1])
        r = bg[D]
        om = r["omega"]
        lo = complex(pos[2].replace(" ", ""))
        lo, ok = secant(Det(r, 0.85), lo)
        eps, re1 = 0.25, 5.0
        top = lo.imag + 1.5
        allb = {"A": (0.05, 0.9, -(om - eps), om - eps), "B": (0.9, re1, -(om - eps), om - eps),
                "C": (0.05, re1, om + eps, top), "S": (0.05, re1, om - eps, om + eps)}
        names = o.get("names", "A,B,C,S").split(",")
        plant = complex(o["plant"].replace(" ", "")) if "plant" in o else None
        for Rf in [float(s) for s in o.get("R", "0.85,0.6").split(",")]:
            for nm in names:
                bx = allb[nm]
                det = Det(r, Rf, plant)
                t = time.time()
                F = detrended(det, bx)
                w, big, ni = winding(F, box(*bx), K=int(o.get("K", "6")))
                emit(dict(kind="box", D=D, box_name=nm, box=list(bx), R_over_xh=Rf, winding=w, largest_step=big,
                          intervals=ni, evaluations=det.nev, seconds=round(time.time() - t, 1),
                          lam_osc=[lo.real, lo.imag], lam_osc_converged=ok, omega=om,
                          plant=None if plant is None else [plant.real, plant.imag], mutate=mutate))
        return
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Review A3, check K4: the step-(i) start values of the M = 3 certificate (fixed_point_M.start_data_M at the centre
p^, ray arg xi = 5pi/16, r0 = 0.0741, diag splitting) against an independent high-precision solution: the analogue
of the M = 1 review's R5 (proof/A_review.md section 2f).

Reference (nothing shared with fixed_point_M.py, series_arb.py or certify_M*.py):
  * the formal series D = sum d_j xi^(2j+1), S = sum s_j xi^(2j+2), C = sum g_j xi^(2j), derived here by hand from
    the equations of solver/inner_stokes/inner_np.py's docstring (d_0 = 2F, g_0 = delta_{nu 0}):
        s_j        = (2j + 1 + i mu w) d_j - sum_{i=1..j} [g_i s_{j-i}]
        e_m        = sum_{i=0..m-1} [d_i d_{m-1-i}] / 2,   ce_m = sum_{i=0..m-1} [g_i e_{m-i}],   g_m = ce_m / (2m)
        2(j+1) d_{j+1} = (2j + 2 + i mu w) s_j - sum_{i=1..j+1} 2(j+1-i) [g_i d_{j+1-i}] - sum_{m=1..j+1} [ce_m d_{j+1-m}]
    in mpmath at 110 digits, truncated at its smallest term at |xi| = 0.05 (the truncation error there is about the
    smallest term, ~1e-80; it is not a rigorous bound, which is why this is a check and not a proof);
  * mpmath.odefun (Taylor method, 75 digits) along the ray from 0.05 e^{5 i pi/16} out to 0.0741 e^{5 i pi/16}, on a
    right-hand side written here (the fast modes decay outward on this ray, by about e^-40 over this stretch).
Pass: every component of the reference lies in the start-value ball of fixed_point_M (radius ~1e-50). Control: the
reference recomputed with A moved by 1e-40 must fall outside (|dy/dA| is of order one).

    python3 rv3_start_value.py        (a few minutes)
"""
import os
import sys
import time

import mpmath
from mpmath import mp, mpc, mpf

HERE = os.path.dirname(os.path.abspath(__file__))
M = 3
ODD = [m for m in range(-M, M + 1) if m % 2]
EVEN = [v for v in range(-(M + 1), M + 2) if v % 2 == 0]


def proj(X, Y, out):
    return {k: sum((X[a] * Y[k - a] for a in X if (k - a) in Y), mpc(0)) for k in out}


def series(Fh, w, J):
    I = mpc(0, 1)
    d = [{m: 2 * Fh[m] for m in ODD}]
    s, g = [], [{v: mpc(1 if v == 0 else 0) for v in EVEN}]
    e = [{v: mpc(0) for v in EVEN}]
    ce = [{v: mpc(0) for v in EVEN}]
    for j in range(J):
        sj = {m: (2 * j + 1 + I * m * w) * d[j][m] for m in ODD}
        for i in range(1, j + 1):
            t = proj(g[i], s[j - i], ODD)
            for m in ODD:
                sj[m] -= t[m]
        s.append(sj)
        mm = j + 1
        em = {v: mpc(0) for v in EVEN}
        for i in range(mm):
            t = proj(d[i], d[mm - 1 - i], EVEN)
            for v in EVEN:
                em[v] += t[v] / 2
        e.append(em)
        cm = {v: mpc(0) for v in EVEN}
        for i in range(mm):
            t = proj(g[i], e[mm - i], EVEN)
            for v in EVEN:
                cm[v] += t[v]
        ce.append(cm)
        g.append({v: cm[v] / (2 * mm) for v in EVEN})
        num = {m: (2 * j + 2 + I * m * w) * s[j][m] for m in ODD}
        for i in range(1, j + 2):
            t = proj(g[i], d[j + 1 - i], ODD)
            for m in ODD:
                num[m] -= 2 * (j + 1 - i) * t[m]
        for k in range(1, j + 2):
            t = proj(ce[k], d[j + 1 - k], ODD)
            for m in ODD:
                num[m] -= t[m]
        d.append({m: num[m] / (2 * j + 2) for m in ODD})
    return d, s, g


def F_ref(Y, xi, w):
    no = len(ODD)
    D = {m: Y[i] for i, m in enumerate(ODD)}
    S = {m: Y[no + i] for i, m in enumerate(ODD)}
    C = {v: Y[2 * no + i] for i, v in enumerate(EVEN)}
    I = mpc(0, 1)
    CS_ = proj(C, S, ODD)
    dD = {m: (CS_[m] - I * m * w * xi * D[m]) / (xi * xi) for m in ODD}
    E = {k: v / 2 for k, v in proj(D, D, EVEN).items()}
    CE = proj(C, E, EVEN)
    xdD = {m: xi * dD[m] for m in ODD}
    a, b, c = proj(C, xdD, ODD), proj(CE, D, ODD), proj(C, D, ODD)
    dS = {m: (a[m] + b[m] - c[m] - I * m * w * xi * S[m]) / (xi * xi) for m in ODD}
    dC = {v: CE[v] / xi for v in EVEN}
    return [dD[m] for m in ODD] + [dS[m] for m in ODD] + [dC[v] for v in EVEN]


def reference(p, rs="0.05", r0="0.0741"):
    mp.dps = 110
    A, wf, B3, C3 = [mpf(x) for x in p]
    w = 2 * mp.pi / mpf("1.64389375") * wf
    I = mpc(0, 1)
    Fh = {1: A / (2 * I), -1: -A / (2 * I), 3: B3 / (2 * I) + C3 / 2, -3: -B3 / (2 * I) + C3 / 2}
    phi = 5 * mp.pi / 16
    u = mpmath.exp(I * phi)
    xs = mpf(rs) * u
    J = int(1 / (2 * mpf(rs) ** 2)) + 60
    d, s, g = series(Fh, w, J)
    x2 = xs * xs
    mags = [max(abs(d[j][m]) for m in ODD) * abs(x2) ** j for j in range(J)]
    jt = min(range(5, J), key=lambda j: mags[j])
    D = {m: sum(d[j][m] * xs * x2 ** j for j in range(jt)) for m in ODD}
    S = {m: sum(s[j][m] * x2 * x2 ** j for j in range(jt)) for m in ODD}
    C = {v: sum(g[j][v] * x2 ** j for j in range(jt)) for v in EVEN}
    y0 = [D[m] for m in ODD] + [S[m] for m in ODD] + [C[v] for v in EVEN]
    smallest = mags[jt]
    mp.dps = 75
    n = len(y0)
    ra, rb = mpf(rs), mpf(r0)

    def f(t, z):
        Y = [mpc(z[i], z[n + i]) for i in range(n)]
        xi = (ra + (rb - ra) * t) * u
        out = [v * (rb - ra) * u for v in F_ref(Y, xi, w)]
        return [v.real for v in out] + [v.imag for v in out]
    sol = mpmath.odefun(f, 0, [+v.real for v in y0] + [+v.imag for v in y0])
    z = sol(1)
    return [mpc(z[i], z[n + i]) for i in range(n)], jt, smallest


def main():
    import json
    sys.path.insert(0, os.path.join(HERE, ".."))
    from flint import arb, acb, ctx
    import fixed_point_M as FPM           # the code under test
    bx = json.load(open(os.path.join(HERE, "..", "data", "A_M_box_M3_1e-10.json")))
    pstr = bx["p"]
    ctx.prec = 256
    pc = [arb(arb(x).mid()) for x in pstr]  # the run's exact centre (certify_M.exact_p)
    ctx.prec = 320
    t0 = time.time()
    _, vals, _, _ = FPM.start_data_M([arb(x.mid()) for x in pc], M, "0.0741", "5pi/16", +1, "1e-3", diag=True)
    print(f"K4: fixed_point_M start values at p^ ({time.time() - t0:.0f} s): max radius "
          f"{max(float(v.rad()) for v in vals):.2e}", flush=True)
    ok = True
    for label, shift in (("clean", 0), ("control: A + 1e-40", 1)):
        t0 = time.time()
        p = list(pstr)
        mp.dps = 110
        if shift:
            p[0] = mpmath.nstr(mpf(p[0]) + mpf("1e-40"), 80)
        ref, jt, small = reference(p)
        ctx.prec = 320
        inside, worst = True, 0.0
        for i, v in enumerate(ref):
            re = arb(mpmath.nstr(v.real, 72))
            im = arb(mpmath.nstr(v.imag, 72))
            dre = float(abs(vals[i].real.mid() - re))
            dim = float(abs(vals[i].imag.mid() - im))
            worst = max(worst, dre, dim)
            inside &= bool(vals[i].real.contains(re)) and bool(vals[i].imag.contains(im))
        print(f"   {label}: series at 0.05 to {jt} terms (smallest term {mpmath.nstr(small, 3)}), integrated to 0.0741 "
              f"({time.time() - t0:.0f} s): inside every start-value ball {inside} (largest |mid - ref| {worst:.1e})",
              flush=True)
        ok &= inside if not shift else not inside
    print("K4: " + ("PASS (reference inside, control outside)" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Lower-precision factorizations of the Newton matrix, the residual kept in full precision (an inexact Newton): where
they are valid and what they would save (notes/LARGE_D_WALL.md, "Lower-precision Newton matrices").

With a Newton matrix M in place of the Jacobian J, the error goes to e' = M^-1 (M - J) e plus second order, so the
iteration converges linearly, to the residual's own accuracy, at the rate rho, the spectral radius of M^-1 (M - J),
while rho < 1. For M the factors of J rounded to unit roundoff u and factored there (multisolver's Lowered, hp's
newton_mixed LU), rho is about u cond(J), with cond(J) = || |J^-1| |J| ||, the condition number that no row scaling
changes (Skeel, J. ACM 26 (1979) 494-526); the ratio of the LU's largest and smallest pivots does change with the row
scaling and can overstate it by many orders. The law for refinement with a low-precision LU is Carson and Higham's
(SIAM J. Sci. Comput. 40 (2018) A817-A847: convergence needs u_f kappa below about 1). For M formed in the lower
precision (its own tangents), M - J also carries the formation's error, which the mass mode amplifies at large D.

    python3 lowered_lu.py scope RAW
    python3 lowered_lu.py planted [--n 8] [--hi 400] [--lo 106] [--inside -2.5] [--outside 1.5] [--it_max 200]
        [--seed 1] [--mutate lowres|exactlu]
    python3 lowered_lu.py measure RAW STATE [--threads 10] [--iters 6] [--seed 1] [--variants oqdfQ]
        [--mutate scale] [--out FILE.json]
    python3 lowered_lu.py measure --planted K [--n 64] [--form 1e-30] [--seed 1] [--mutate scale]

scope: the table of the notes from stored logs of the raw-data repository RAW (the chain #966, #1008's arclength
corrector, low D T82/T100) and #502's equilibrated singular values: pivot ratios, u times them, the equilibrated
condition number at D 242.36, and the seconds of a Jacobian, a trial and a base. Reads only; no kernel.

planted: the law on a planted problem in mpmath, no kernel (its docstring). Exit 1 when a criterion fails; each
--mutate must make it fail.

measure: on a Linux runner with the built module, at a stored eight-part state (octo_corrector.truth_system: the
chain's own pinned system), rho of each Newton matrix by power iteration on T e = e - M^-1 (J e), J the exact
octo-double Jacobian (newton_o, keep_jacobian): o the octo-double LU of J (the control), q J rounded to quad-double
and factored there, d to double-double, f to f64, Q the quad-double Jacobian (jacobian_q, its own tangents) in
quad-double. Exit 1 when the control's rho exceeds 1e-45; --mutate scale moves one entry of the control's matrix by
1e-40 of itself and must make it fail. --planted K runs the same code on a planted J (planted_system: condition
number 10^K, rows and columns scaled, a quad-double "formed" Jacobian with a relative error --form) with no kernel,
and also requires rho_q within [1e-4, 1e2] of u_qd 10^K and rho_Q within that of form 10^K (at least 1 when that
exceeds 1e10).
"""
import json
import math
import os
import re
import statistics as st
import sys
import time

import numpy as np

U_OD, U_QD, U_DD, U_F64 = 2.0 ** -418, 2.0 ** -209, 2.0 ** -104, 2.0 ** -53     # contraction per unit cond


def _parse(argv, opt):
    unknown = [a for a in argv if a.startswith("--") and a[2:] not in opt]
    if unknown:
        raise SystemExit(f"lowered_lu.py: unknown options {unknown}; known: {sorted(opt)}")
    for k in list(opt):
        if "--" + k in argv:
            opt[k] = type(opt[k])(argv[argv.index("--" + k) + 1])
    return opt


# ------------------------------------------------------------------ scope: the stored logs
NUM = r"[-+]?[\d.]+(?:e[-+]?\d+)?"


def corrector_rows(path):
    """each Newton iteration line of an anm_run log ('on k: |F| ... lam L, ... piv a..b, T s'), with the seconds of its
    trials and of a strong chord before it: an iteration of the natural test forms one Jacobian (newton_o) at its
    accepted trial, so T less those is that Jacobian's seconds"""
    rows, trials, chord = [], 0.0, 0.0
    for line in open(path, errors="replace"):
        line = line.rstrip()
        m = re.search(rf"on \d+ trial lam {NUM}: .*, ({NUM}) s$", line)
        if m:
            trials += float(m.group(1))
            continue
        m = re.search(rf"on 0: strong chord .*: (?:accepted|refused), ({NUM}) s", line)
        if m:
            chord = float(m.group(1))
            continue
        m = re.search(rf"^\s+on \d+: \|F\| .* lam ({NUM}), .*piv ({NUM})\.\.({NUM}), ({NUM}) s", line)
        if m:
            T = float(m.group(4))
            rows.append(dict(pmin=float(m.group(2)), pmax=float(m.group(3)), T=T, trials=trials, chord=chord,
                             jac=T - trials - chord if trials > 0 and chord == 0 else None))
            trials, chord = 0.0, 0.0
    return rows


def chain_bases(path):
    """the bases of an anm_run log: (D, pmin, pmax, series s) from 'step k: D x, pivots a..b, ..., S s', and the
    attempts' re-seed and total seconds"""
    bases, reseed, attempt = [], [], []
    for line in open(path, errors="replace"):
        m = re.search(rf"^step \d+: D ({NUM}), pivots ({NUM})\.\.({NUM}), .*, ({NUM}) s$", line.rstrip())
        if m:
            bases.append(tuple(float(m.group(i)) for i in range(1, 5)))
        m = re.search(rf"re-seeded \(ok\), ({NUM}) s", line)
        if m:
            reseed.append(float(m.group(1)))
        m = re.search(rf"^\s+a {NUM} -> D .*, ({NUM}) s$", line.rstrip())
        if m:
            attempt.append(float(m.group(1)))
    return bases, reseed, attempt


def lowd_rows(path):
    """the low-D Newton calls of lowD_highM logs: 'newton k: |F| f ... piv a..b  t T s'"""
    out = []
    for line in open(path, errors="replace"):
        m = re.search(rf"newton \d+: \|F\| ({NUM}) .*piv ({NUM})\.\.({NUM}) +t ({NUM})s", line)
        if m:
            out.append(tuple(float(m.group(i)) for i in range(1, 5)))
    return out


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else float("nan")


def ruiz(A, tol=1e-12, it_max=200):
    """Ruiz's equilibration of |A| in the max norm (every row and column of max norm 1): the scalings r, c and the
    iterations, as hp/near_null_eq.py's first stage (#502)"""
    A = np.abs(A)
    n = A.shape[0]
    r, c = np.ones(n), np.ones(n)
    for it in range(it_max):
        B = r[:, None] * A * c[None, :]
        rm, cm = B.max(1), B.max(0)
        if max(np.abs(1 - rm).max(), np.abs(1 - cm).max()) < tol:
            return r, c, it
        r, c = r / np.sqrt(rm), c / np.sqrt(cm)
    return r, c, it_max


def scope(argv):
    raw = argv[0]
    chain = os.path.join(raw, "I966-dh370", "ks_chain3b")
    print(f"u: quad-double 2^-209 = {U_QD:.2e}, double-double 2^-104 = {U_DD:.2e}, f64 2^-53 = {U_F64:.2e}")
    print("\nthe chain (#966, dh370, 10 threads, ks 293; raw I966-dh370/ks_chain3b): the corrector's LUs")
    print("| leg | D of the bases | pivot ratio | u_qd x ratio | Jacobian s | trial s | series s | re-seed s | attempt s |")
    print("|---|---|---|---|---|---|---|---|---|")
    fitD, fitR, fitP = [], [], []
    for leg in sorted(d for d in os.listdir(chain) if d.startswith("leg")):
        p = os.path.join(chain, leg, "anm_run.log")
        if not os.path.exists(p):
            continue
        rows = corrector_rows(p)
        bases, reseed, attempt = chain_bases(p)
        if not rows or not bases:
            continue
        ratio = [r["pmax"] / r["pmin"] for r in rows]
        for D, pmin, pmax, _ in bases:
            fitD.append(D)
            fitR.append(math.log10(pmax / pmin))
            fitP.append(math.log10(pmin))
        Ds = [b[0] for b in bases]
        print(f"| {leg[:5]} | {min(Ds):.2f}-{max(Ds):.2f} | {min(ratio):.1e}-{max(ratio):.1e} | "
              f"{U_QD * min(ratio):.1e}-{U_QD * max(ratio):.1e} | {med(r['jac'] for r in rows):.1f} | "
              f"{med(r['trials'] for r in rows if r['trials'] > 0):.1f} | {med(b[3] for b in bases):.0f} | "
              f"{med(reseed):.0f} | {med(attempt):.0f} |")
    sR, iR = np.polyfit(fitD, fitR, 1)
    sP, iP = np.polyfit(fitD, fitP, 1)
    print(f"\nbase LUs, {len(fitD)} bases D {min(fitD):.2f}-{max(fitD):.2f}: log10(pivot ratio) rises {sR:.3f} per unit D "
          f"(a factor {10 ** sR:.2f}), log10(smallest pivot) {sP:+.3f} per unit D (least squares)")

    # #502: the equilibrated system at D 242.3642 (layout 0.3-0.7, L 769, the chain's), its sigma_1 from the octo-double LU
    eq = os.path.join(raw, "I502-mac")
    rec = json.load(open(os.path.join(eq, "L07_plant7.json")))
    s1 = rec["methods"]["Ruiz"]["vectors"][0]["sigma"]
    J0 = np.load(os.path.join(eq, "explore", "J0_L07.npz"))["J0"]
    r, c, its = ruiz(J0)
    smax = float(np.linalg.norm(r[:, None] * J0 * c[None, :], 2))
    k = smax / s1
    print(f"\n#502 (raw I502-mac): D {rec['D']:.4f}, L {rec['L']}, raw pivots {rec['pivots'][0]:.2e}..{rec['pivots'][1]:.2e} "
          f"(ratio {rec['pivots'][1] / rec['pivots'][0]:.1e}); Ruiz scaling in {its} iterations, D_r {r.min():.2e}..{r.max():.2e} "
          f"(#502: {rec['methods']['Ruiz']['D_r'][0]:.2e}..{rec['methods']['Ruiz']['D_r'][1]:.2e}); "
          f"sigma_max {smax:.3f} (f64 leading part), sigma_1 {s1:.3e} (#502, octo-double LU)")
    print(f"  kappa_eq = {k:.2e}: u_qd kappa {U_QD * k:.1e}, u_dd kappa {U_DD * k:.1e}, u_f64 kappa {U_F64 * k:.1e}")
    print("  u_qd kappa_eq carried to the chain's bases, if kappa_eq grows as the pivot ratio / as 1 / smallest pivot:")
    for D in (245.0, 250.71, 252.89, 255.0, 256.53, 257.15):
        print(f"    D {D:.2f}: {U_QD * k * 10 ** (sR * (D - rec['D'])):.1e} / {U_QD * k * 10 ** (-sP * (D - rec['D'])):.1e}")
    one = -math.log10(U_QD * k)
    print(f"    u_qd kappa_eq = 1 at D {rec['D'] + one / sR:.1f} / {rec['D'] - one / sP:.1f}")

    print("\n#1008's arclength corrector (macm3, 6 threads, ks 135, a kernel from before #921): raw I1008-macm3")
    for f in ("ab1.log", "run1.log"):
        p = os.path.join(raw, "I1008-macm3", f)
        if not os.path.exists(p):
            continue
        rows = corrector_rows(p)
        ratio = [r["pmax"] / r["pmin"] for r in rows]
        tot = [float(m.group(1)) for m in re.finditer(rf"corrector: \d+ steps, .*, ({NUM}) s;", open(p).read())]
        jac, tri = med(r["jac"] for r in rows), med(r["trials"] for r in rows if r["trials"] > 0)
        print(f"  {f}: {len(rows)} corrections, pivot ratio {min(ratio):.1e}-{max(ratio):.1e} (u_qd x {U_QD * min(ratio):.0e}-"
              f"{U_QD * max(ratio):.0e}), Jacobian {jac:.1f} s, trial {tri:.1f} s, correctors {', '.join(f'{t:.0f}' for t in tot)} s")

    print("\nlow D (lowD_highM.py, newton_mixed: double-double residual, f64 Jacobian and LU): raw T82-win, T100-win2")
    for f in ("T82-win/down25.log", "T100-win2/down25e.log"):
        p = os.path.join(raw, f)
        if not os.path.exists(p):
            continue
        rows = lowd_rows(p)
        ratio = [b / a for _, a, b, _ in rows]
        print(f"  {f}: {len(rows)} calls, pivot ratio {min(ratio):.1e}-{max(ratio):.1e} (u_f64 x {U_F64 * min(ratio):.0e}-"
              f"{U_F64 * max(ratio):.0e}), call {med(t for *_, t in rows):.0f} s; |F| per call "
              + " ".join(f"{F:.1e}" for F, *_ in rows[:12]))
    return 0


# ------------------------------------------------------------------ planted: the law on a known problem
def planted(argv):
    """The inexact Newton on F(z) = R U [s (y + g y^2)] - b, y = V^T C z, with U, V random orthogonal, s singular values
    falling geometrically from 1 to 1/kappa0, rows scaled by R = 10^U(-12, 12) and unknowns by C = 10^U(-4, 4), and
    g = 0.1, so that J(z) = R U diag(s (1 + 2 g y)) V^T C keeps the singular structure of the linear part; z* random,
    b = F's linear-plus-quadratic part at z*. The residual is formed at --hi bits; the Newton matrix is J(z) formed at
    --hi bits, rounded to --lo bits and factored there by LU with partial pivoting (the kernel's pivoting, not mpmath's
    scaled one), each solve rounding F to --lo bits: multisolver's Lowered at a precision of u = 2^-lo. kappa0 is set
    so that log10(u kappa0) is --inside (-2.5) or --outside (1.5). From z* moved by 1e-10 (in C z), the damped
    iteration (lam = 1, 1/2, ..., 2^-8, a trial accepted when its row-scaled max |F| falls) runs to --it_max steps.

    Pass, fixed before the first run: (1) inside converges, its forward error (max |C (z - z*)| / max |C z*|) at most
    1e3 2^-hi kappa_eq, with the observed rate rho at most 0.1; (2) outside does not converge, and its verdict (refused:
    no trial accepted; stalled: max |F| falls by less than 2 over 10 accepted steps; diverged; cap) is printed; (3)
    inside's u times the LU's pivot ratio is at least 10, so the pivot ratio alone would have refused it; (4) rho
    inside within [1e-3, 1e2] of u kappa_eq, kappa_eq the 2-norm condition number of J(z*) Ruiz-equilibrated.
    Controls, each of which must make it fail: --mutate lowres forms the residual at --lo bits too (inside then stops
    near u kappa_eq, (1)); --mutate exactlu factors at --hi bits (outside then converges, (2))."""
    import mpmath as mp
    opt = _parse(argv, dict(n=8, hi=400, lo=106, inside=-2.5, outside=1.5, it_max=200, seed=1, mutate="", sweep=1))
    n, hi, lo = opt["n"], opt["hi"], opt["lo"]
    if opt["mutate"] not in ("", "lowres", "exactlu"):
        raise SystemExit(f"--mutate {opt['mutate']}: lowres or exactlu")
    u = 2.0 ** -lo
    rng = np.random.default_rng(opt["seed"])

    def orth(n):
        with mp.workprec(hi):
            Q, _ = mp.qr(mp.matrix(rng.standard_normal((n, n)).tolist()))
            return Q

    U0, V0 = orth(n), orth(n)
    rsc = [mp.mpf(10) ** int(x) for x in rng.integers(-12, 13, n)]
    csc = [mp.mpf(10) ** int(x) for x in rng.integers(-4, 5, n)]
    w = rng.uniform(-1, 1, n)
    d0 = rng.standard_normal(n)

    def problem(t):
        with mp.workprec(hi):
            k0 = mp.mpf(10) ** t / mp.mpf(u)
            s = [k0 ** (-mp.mpf(i) / (n - 1)) for i in range(n)]
            zs = [mp.mpf(w[j]) / csc[j] for j in range(n)]
            return dict(k0=k0, s=s, zs=zs, b=part(s, zs, hi))

    g = mp.mpf("0.1")

    def part(s, z, prec):
        """R U [s (y + g y^2)] at prec bits"""
        with mp.workprec(prec):
            y = [mp.fsum(V0[i, j] * csc[i] * z[i] for i in range(n)) for j in range(n)]
            v = [s[j] * (y[j] + g * y[j] ** 2) for j in range(n)]
            return [rsc[i] * mp.fsum(U0[i, j] * v[j] for j in range(n)) for i in range(n)]

    def resid(P, z):
        prec = lo if opt["mutate"] == "lowres" else hi
        with mp.workprec(prec):
            F = part(P["s"], [+x for x in z], prec)
            return [F[i] - P["b"][i] for i in range(n)]

    def jac(P, z):
        with mp.workprec(hi):
            y = [mp.fsum(V0[i, j] * csc[i] * z[i] for i in range(n)) for j in range(n)]
            dg = [P["s"][j] * (1 + 2 * g * y[j]) for j in range(n)]
            return [[rsc[i] * mp.fsum(U0[i, k] * dg[k] * V0[j, k] for k in range(n)) * csc[j] for j in range(n)]
                    for i in range(n)]

    def lu(A, prec):
        """LU with partial pivoting (the largest |a_ik| in the column) at prec bits of A rounded there: (LU, perm,
        pivots)"""
        with mp.workprec(prec):
            a = [[+x for x in row] for row in A]
            perm = list(range(n))
            for k in range(n):
                p = max(range(k, n), key=lambda i: abs(a[i][k]))
                a[k], a[p], perm[k], perm[p] = a[p], a[k], perm[p], perm[k]
                for i in range(k + 1, n):
                    a[i][k] = a[i][k] / a[k][k]
                    for j in range(k + 1, n):
                        a[i][j] -= a[i][k] * a[k][j]
            return a, perm, [abs(a[k][k]) for k in range(n)]

    def solve_neg(f, F, prec):
        """-J^-1 F by the factors f at prec bits, F rounded there (negated inside the context: an mpf operation
        outside it rounds to the default 53 bits)"""
        a, perm, _ = f
        with mp.workprec(prec):
            x = [-F[perm[i]] for i in range(n)]
            for i in range(n):
                x[i] -= mp.fsum(a[i][j] * x[j] for j in range(i))
            for i in reversed(range(n)):
                x[i] = (x[i] - mp.fsum(a[i][j] * x[j] for j in range(i + 1, n))) / a[i][i]
            return x

    def kappa_eq(A):
        """the 2-norm condition number of A Ruiz-equilibrated, at hi bits"""
        with mp.workprec(hi):
            r, c = [mp.mpf(1)] * n, [mp.mpf(1)] * n
            for _ in range(60):
                B = [[abs(r[i] * A[i][j] * c[j]) for j in range(n)] for i in range(n)]
                rm = [max(B[i]) for i in range(n)]
                r = [r[i] / mp.sqrt(rm[i]) for i in range(n)]
                B = [[abs(r[i] * A[i][j] * c[j]) for j in range(n)] for i in range(n)]
                cm = [max(B[i][j] for i in range(n)) for j in range(n)]
                c = [c[j] / mp.sqrt(cm[j]) for j in range(n)]
            S = mp.svd_r(mp.matrix([[r[i] * A[i][j] * c[j] for j in range(n)] for i in range(n)]), compute_uv=False)
            return max(S) / min(S)

    def ferr(P, z):
        return float(max(abs(csc[j] * (z[j] - P["zs"][j])) for j in range(n)) / max(abs(csc[j] * P["zs"][j]) for j in range(n)))

    def fnorm(F):
        return float(max(abs(F[i] / rsc[i]) for i in range(n)))

    def run(t, damped=True, steps=None):
        P = problem(t)
        with mp.workprec(hi):
            z = [P["zs"][j] + mp.mpf(1e-10) * mp.mpf(d0[j]) / csc[j] for j in range(n)]
        J = jac(P, P["zs"])
        ke = kappa_eq(J)
        piv = lu(J, lo)[2]
        prat = float(max(piv) / min(piv))
        floor = float(1e3 * mp.mpf(2) ** -hi * ke)
        F = resid(P, z)
        errs, verdict, acc = [ferr(P, z)], "cap", [fnorm(F)]
        for it in range(steps or opt["it_max"]):
            if errs[-1] <= floor:
                verdict = "converged"
                break
            fac = lu(jac(P, z), hi if opt["mutate"] == "exactlu" else lo)
            dz = solve_neg(fac, F, hi if opt["mutate"] == "exactlu" else lo)
            lam, ok = 1.0, False
            for _ in range(9 if damped else 1):
                with mp.workprec(hi):
                    zt = [z[j] + lam * dz[j] for j in range(n)]
                Ft = resid(P, zt)
                if not damped or (all(mp.isfinite(x) for x in Ft) and fnorm(Ft) < fnorm(F)):
                    ok = True
                    break
                lam *= 0.5
            if not ok:
                verdict = "refused"
                break
            z, F = zt, Ft
            errs.append(ferr(P, z))
            acc.append(fnorm(F))
            if not math.isfinite(errs[-1]) or errs[-1] > 1.0:
                verdict = "diverged"
                break
            if damped and len(acc) > 10 and acc[-1] > 0.5 * acc[-11]:
                verdict = "stalled"
                break
        lin = [errs[i + 1] / errs[i] for i in range(len(errs) - 1) if errs[i + 1] > 1e3 * floor and errs[i] > 0]
        rho = math.exp(st.mean(math.log(x) for x in lin)) if lin else float("nan")
        return dict(t=t, kappa0=float(P["k0"]), kappa_eq=float(ke), pivot_ratio=prat, floor=floor, steps=len(errs) - 1,
                    err=errs[-1], verdict=verdict, rho=rho, F=acc[-1])

    print(f"planted inexact Newton: n {n}, residual at {hi} bits, Newton matrix at {lo} bits (u {u:.2e}); "
          f"mutate '{opt['mutate']}'", flush=True)
    res = {}
    for name in ("inside", "outside"):
        r = res[name] = run(opt[name])
        print(f"  {name}: log10(u kappa0) {r['t']:+.1f}; kappa_eq {r['kappa_eq']:.2e} (u kappa_eq {u * r['kappa_eq']:.1e}), "
              f"LU pivot ratio {r['pivot_ratio']:.1e} (u x {u * r['pivot_ratio']:.1e}); {r['verdict']} after {r['steps']} steps, "
              f"forward error {r['err']:.2e} (floor {r['floor']:.1e}), rho {r['rho']:.2e}", flush=True)
    if opt["sweep"]:
        print("  sweep, undamped, 8 steps: log10(u kappa0), u kappa_eq, rho")
        for t in (-4.0, -3.0, -2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0):
            r = run(t, damped=False, steps=8)
            print(f"    {t:+.1f}  {u * r['kappa_eq']:.1e}  {r['rho']:.2e}", flush=True)
    i, o = res["inside"], res["outside"]
    law = i["rho"] / (u * i["kappa_eq"]) if i["rho"] == i["rho"] else float("nan")
    checks = [("(1) inside converges to its floor with rho <= 0.1", i["verdict"] == "converged" and i["rho"] <= 0.1),
              ("(2) outside does not converge", o["verdict"] != "converged"),
              ("(3) inside's u x pivot ratio >= 10", u * i["pivot_ratio"] >= 10),
              (f"(4) inside rho / (u kappa_eq) = {law:.2e} in [1e-3, 1e2]", 1e-3 <= law <= 1e2)]
    for text, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}: {text}", flush=True)
    ok = all(c for _, c in checks)
    print(f"  {'PASS' if ok else 'FAIL'}: planted{' (--mutate ' + opt['mutate'] + ')' if opt['mutate'] else ''}; "
          f"outside's verdict: {o['verdict']}", flush=True)
    return 0 if ok else 1


# ------------------------------------------------------------------ measure: rho at a stored state (Linux runner)
def o8(z):
    """eight parts as contiguous float64 arrays (octo_corrector.o8)"""
    return tuple(np.ascontiguousarray(np.asarray(p, dtype=float)) for p in z)


def q4_of_o(zo):
    """an octo-double vector rounded to quad-double (octo_corrector.q4_of_o)"""
    from multidouble import qd_add_dd
    zo = o8(zo)
    tail = zo[4] + (zo[5] + (zo[6] + zo[7]))
    return tuple(np.ascontiguousarray(np.asarray(p)) for p in qd_add_dd(zo[0], zo[1], zo[2], zo[3], tail, np.zeros(len(tail))))


class _Planted:
    """measure --planted's stand-in for the kernel's Solver: newton_o returns a random step, jacobian_o_kept the
    planted J in eight parts, jacobian_q J times (1 + form r) in four parts, r uniform in [-1, 1] (a formation
    error of relative size form)"""

    def __init__(self, J8, Jq, dz):
        self.J8, self.Jq, self.dz = J8, Jq, dz

    def newton_o(self, z, keep_jacobian=False):
        zero = np.zeros(len(self.dz[0]))
        return self.dz, zero, zero, float("nan"), float("nan")

    def jacobian_o_kept(self, z):
        return self.J8

    def jacobian_q(self, *z):
        return self.Jq


def planted_system(K, n, seed, form):
    """J = R U diag(s) V^T C at 600 bits, s from 1 down to 10^-K geometrically, U and V random orthogonal, rows scaled by
    10^U(-12, 12) and columns by 10^U(-4, 4), split into eight float64 parts (and four for jacobian_q, with the
    formation error); (z, meta, solver, info) as truth_system's, z zero"""
    import mpmath as mp
    rng = np.random.default_rng(seed)
    with mp.workprec(600):
        U = mp.qr(mp.matrix(rng.standard_normal((n, n)).tolist()))[0]
        V = mp.qr(mp.matrix(rng.standard_normal((n, n)).tolist()))[0]
        sv = [mp.mpf(10) ** (-mp.mpf(K) * i / (n - 1)) for i in range(n)]
        R = [mp.mpf(10) ** int(x) for x in rng.integers(-12, 13, n)]
        C = [mp.mpf(10) ** int(x) for x in rng.integers(-4, 5, n)]
        fr = rng.uniform(-1, 1, (n, n))
        J8, Jq = [np.zeros((n, n)) for _ in range(8)], [np.zeros((n, n)) for _ in range(4)]
        for i in range(n):
            for j in range(n):
                v = R[i] * mp.fsum(U[i, k] * sv[k] * V[j, k] for k in range(n)) * C[j]
                w = v * (1 + mp.mpf(form) * mp.mpf(fr[i, j]))
                for q in range(8):
                    J8[q][i, j] = float(v)
                    v -= mp.mpf(J8[q][i, j])
                for q in range(4):
                    Jq[q][i, j] = float(w)
                    w -= mp.mpf(Jq[q][i, j])
    dz = o8([rng.standard_normal(n) / np.array([float(c) for c in C])] + [np.zeros(n)] * 7)
    meta = dict(D_str="planted", ks=None, nodes=[])
    return o8([np.zeros(n)] * 8), meta, _Planted(J8, Jq, dz), dict(planted=K, form=form)


def measure(argv):
    """rho of each Newton matrix at RAW/STATE by power iteration (the module's docstring). The norm is the max norm of
    D_c^-1 e, D_c the Ruiz column scaling of J's leading part, so that every unknown counts on the equilibrated
    scale. Two starts: D_c times a random normal vector (--seed), and the octo-double Newton step at the state. rho is
    the larger of the two starts' last ratios; the effective condition number rho / u is printed with it. Pass
    (exit 0): the control o has rho at most 1e-45 at both starts. Classification: valid at rho <= 0.1, marginal
    below 1, not valid at 1 or more."""
    import multidouble as md
    pos = [a for i, a in enumerate(argv) if not a.startswith("--") and (i == 0 or not argv[i - 1].startswith("--"))]
    opt = _parse(argv, dict(threads=10, iters=6, seed=1, variants="oqdfQ", mutate="", out="", planted=-1.0, n=64,
                            form=1e-30))
    if opt["mutate"] not in ("", "scale"):
        raise SystemExit(f"--mutate {opt['mutate']}: scale")
    t0 = time.time()
    if opt["planted"] >= 0:
        z, meta, s, info = planted_system(opt["planted"], opt["n"], opt["seed"], opt["form"])
        path = f"planted, kappa0 1e{opt['planted']:g}, n {opt['n']}"
    else:
        from octo_corrector import truth_system
        raw, path = pos[0], pos[1]
        z, meta, (s, sp, sq, so), info = truth_system(raw, path, threads=opt["threads"])
    z = o8(z)
    L = len(z[0])
    zero = np.zeros(L)
    print(f"{path}: D {meta['D_str'][:16]}, L {L}, ks {meta.get('ks')}, nodes {meta['nodes']}; system {info}", flush=True)
    rec = dict(state=path, D_str=meta["D_str"], ks=meta.get("ks"), L=L, mutate=opt["mutate"], seconds={}, rho={})
    t = time.time()
    keep = hasattr(s, "jacobian_o_kept")
    st0 = s.newton_o(list(z), keep_jacobian=True) if keep else s.newton_o(list(z))
    rec["seconds"]["newton_o"] = time.time() - t
    if keep:
        J8 = s.jacobian_o_kept(list(z))
    else:
        print("  this hpkernel has no jacobian_o_kept: the Jacobian is formed again", flush=True)
        t = time.time()
        J8 = s.jacobian_o(list(z))
        rec["seconds"]["jacobian_o"] = time.time() - t
    J8 = [np.ascontiguousarray(np.asarray(p)) for p in J8]
    dz = o8(st0[0])
    rec["pivots_o"] = [st0[3], st0[4]]
    print(f"  newton_o {rec['seconds']['newton_o']:.1f} s (the octo-double Jacobian and LU), pivots {st0[3]:.2e}..{st0[4]:.2e}",
          flush=True)
    r, c, its = ruiz(J8[0])
    JT = [np.ascontiguousarray(p.T) for p in J8]

    def jv(e):
        """J e in octo-double for an f64 vector e, column by column (od_axpy)"""
        y = [zero] * 8
        for j in np.nonzero(e)[0]:
            y = md.od_axpy(float(e[j]), *[JT[q][j] for q in range(8)], *y)
        return o8(y)

    def size(v):
        return float(np.abs(np.asarray(v) / c).max())

    Ms = {}
    if "o" in opt["variants"]:
        A = [p.copy() for p in J8]
        if opt["mutate"] == "scale":
            j0 = L - 1
            i0 = int(np.argmax(np.abs(J8[0][:, j0]) * r))
            a8 = [np.array([A[q][i0, j0]]) for q in range(8)]
            e = md.od_axpy(1e-40, *a8, *a8)
            for q in range(8):
                A[q][i0, j0] = float(e[q][0])
            print(f"  --mutate scale: the control's entry ({i0}, {j0}) moved by 1e-40 of itself", flush=True)
        t = time.time()
        lu = md.ODLu(*A)
        rec["seconds"]["factor_o"] = time.time() - t
        Ms["o"] = (lambda y, lu=lu: o8(lu.solve(*y)), U_OD, "octo-double LU of J (control)")
    if "q" in opt["variants"]:
        t = time.time()
        lq = md.QDLu(*J8[:4])
        rec["seconds"]["factor_q"] = time.time() - t
        Ms["q"] = (lambda y, lq=lq: o8(tuple(lq.solve(*y[:4])) + (zero,) * 4), U_QD, "J rounded to quad-double, QDLu")
    if "d" in opt["variants"]:
        Ms["d"] = (lambda y: o8(tuple(md.dd_solve(J8[0], J8[1], y[0], y[1])) + (zero,) * 6), U_DD,
                   "J rounded to double-double, dd_solve")
    if "f" in opt["variants"]:
        Ms["f"] = (lambda y: o8((np.linalg.solve(J8[0], np.asarray(y[0]) + np.asarray(y[1])),) + (zero,) * 7), U_F64,
                   "J rounded to f64, LAPACK")
    if "Q" in opt["variants"]:
        t = time.time()
        Jq = [np.ascontiguousarray(np.asarray(p)) for p in s.jacobian_q(*q4_of_o(z))]
        rec["seconds"]["jacobian_q"] = time.time() - t
        t = time.time()
        lQ = md.QDLu(*Jq)
        rec["seconds"]["factor_Q"] = time.time() - t
        Ms["Q"] = (lambda y, lQ=lQ: o8(tuple(lQ.solve(*y[:4])) + (zero,) * 4), U_QD, "the quad-double Jacobian, QDLu")
    print("  seconds " + ", ".join(f"{k} {v:.1f}" for k, v in rec["seconds"].items()), flush=True)
    rng = np.random.default_rng(opt["seed"])
    starts = dict(random=c * rng.standard_normal(L), step=np.asarray(dz[0]) + np.asarray(dz[1]))
    ok = True
    for key, (solve, u, what) in Ms.items():
        rhos = {}
        for sname, e in starts.items():
            if not size(e) > 0:
                rhos[sname] = [float("nan")]           # a zero step: this start is skipped
                continue
            e = e / size(e)
            hist = []
            for _ in range(opt["iters"]):
                x = solve(jv(e))
                d = md.od_axpy(-1.0, e, *([zero] * 7), *x)        # M^-1 J e - e, in octo-double
                d0 = np.asarray(d[0]) + np.asarray(d[1])
                hist.append(size(d0) / size(e))
                if not np.isfinite(hist[-1]) or hist[-1] == 0:
                    break
                e = d0 / size(d0)
            rhos[sname] = hist
        rho = max((h[-1] for h in rhos.values() if h[-1] == h[-1]), default=float("nan"))
        cls = "valid" if rho <= 0.1 else ("marginal" if rho < 1 else "not valid")
        rec["rho"][key] = dict(what=what, rho=rho, kappa_eff=rho / u, hist=rhos, cls=cls)
        print(f"  {key} ({what}): rho {rho:.2e} ({cls}), rho / u {rho / u:.2e}; random "
              + " ".join(f"{x:.1e}" for x in rhos["random"]) + "; step " + " ".join(f"{x:.1e}" for x in rhos["step"]), flush=True)
        if key == "o" and not rho <= 1e-45:
            ok = False
    if opt["planted"] >= 0:
        # the planted law: rho of the quad-double factorization within [1e-4, 1e2] of u_qd kappa0, and of the
        # quad-double "formed" Jacobian within the same of form kappa0 when that is below 1e10 (else at least 1)
        k0 = 10.0 ** opt["planted"]
        lq = rec["rho"].get("q", {}).get("rho", float("nan")) / (U_QD * k0)
        fQ = opt["form"] * k0
        lQ = rec["rho"].get("Q", {}).get("rho", float("nan"))
        okQ = (1e-4 <= lQ / fQ <= 1e2) if fQ < 1e10 else lQ >= 1
        print(f"  planted law: rho_q / (u_qd kappa0) = {lq:.2e} (pass in [1e-4, 1e2]); rho_Q {lQ:.2e} against form kappa0 "
              f"{fQ:.1e} ({'pass' if okQ else 'FAIL'})", flush=True)
        ok = ok and 1e-4 <= lq <= 1e2 and okQ
    rec["ok"] = ok
    rec["seconds"]["total"] = time.time() - t0
    if opt["out"]:
        with open(opt["out"], "w") as f:
            f.write(json.dumps(rec, default=float) + "\n")
    print(f"  {'PASS' if ok else 'FAIL'}: the control's rho at most 1e-45{' (--mutate scale)' if opt['mutate'] else ''}; "
          f"{rec['seconds']['total']:.0f} s", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("scope", "planted", "measure"):
        sys.exit(__doc__)
    sys.exit({"scope": scope, "planted": planted, "measure": measure}[sys.argv[1]](sys.argv[2:]))

#!/usr/bin/env python3
"""Eq. (16)'s parameters as series in A^2 from the D = infinity amplitude expansion (#390; pre-registered on the issue).

The amplitude expansion of the inner problem (solver/inner_amp, `f` of sigF_*.json) gives
F = A sin(theta) + sum_{nu >= 3} f_nu(A) e^{i nu theta} + c.c., each f_nu^(k) fixed by S_nu = 0 at order A^k. With
F_1 = -iA/2,
    w(A) = F_3/F_1 = 2i f_3(A)/A = sum_{m >= 1} q_m x^m,   x = A^2,   q_m = 2i f[(2m+1)_3],
which is EEG's q e^{i delta} at A_* = sqrt2 A (notes/EEG_PROFILE.md section 3.1). The higher ratios
r_nu = F_{nu+2}/F_nu are series in x by series division. A sum's truncation estimate (pre-registered): coefficient m is
trusted when the two J agree to 1e-3 relative and |q_m/q_{m-1}| is within a factor 1.5 of the median ratio over
m = 5..15; M_t is the last m with every coefficient up to it trusted; S = sum_{m <= M_t} q_m x^m and
T = |t_{M_t}| rho/(1 - rho), rho the median |t_m/t_{m-1}| over the last 6 trusted terms.

    cd solver/eeg_profile
    python3 w_series.py planted [--mutate=index|phase]   planted series through the same code (must pass; a mutation
                                                         must make it fail)
    python3 w_series.py run [--A=2.72] [--out=FILE]       the series, their sums and the comparisons
Exit codes: planted 0 when recovered (1 otherwise); run 0 when comparison 2 (the tuned-profile value) passes, 1 if
not. Comparison 1 (the issue's 4-digit target) and 3 (M = 7) are printed with their verdicts but do not set it.
"""
import json
import math
import os
import sys

import mpmath as mp

mp.mp.dps = 50
HERE = os.path.dirname(os.path.abspath(__file__))
AMP = os.path.join(HERE, "..", "inner_amp")
F1500 = os.path.join(AMP, "scan45J1500", "sigF_w1.0000_K45_J1500_p1280.json")
F1250 = os.path.join(AMP, "scan45J1250", "sigF_w1.0000_K45_J1250_p1280.json")
K43 = {w: os.path.join(AMP, "scan43J1500", "sigF_w%.4f_K43_J1500_p1280.json" % w) for w in (0.99, 1.0, 1.01)}
args = dict(a.lstrip("-").split("=", 1) if "=" in a else (a.lstrip("-"), "1") for a in sys.argv[2:])
MUT = args.get("mutate")
OMEGA_INF = 2 * mp.pi / mp.mpf("1.64389375")


def load_f(path):
    d = json.load(open(path))
    return {tuple(int(v) for v in k.split("_")): mp.mpc(*v) for k, v in d["f"].items()}


def harmonic(f, nu):
    """coefficients a_j of F_nu/F_1 = sum_j a_j x^((nu-1)/2 + j); F_1 = -iA/2 (mutate=phase: A/2)"""
    c = mp.mpf(2) if MUT == "phase" else 2j
    off = 2 if MUT == "index" else 0
    out = []
    j = 0
    while (nu + 2 * j + off, nu) in f:
        out.append(c * f[(nu + 2 * j + off, nu)])
        j += 1
    return out


def w_coeffs(f):
    """q_m, m = 1..: w = F_3/F_1 = sum q_m x^m"""
    return harmonic(f, 3)


def ratio_coeffs(f, nu):
    """r_nu = F_{nu+2}/F_nu = x * (sum b_j x^j)/(sum a_j x^j): coefficients of x^m, m = 1.."""
    a, b = harmonic(f, nu), harmonic(f, nu + 2)
    n = min(len(a), len(b))
    c = []
    for m in range(n):                          # series division b / a
        c.append((b[m] - mp.fsum(c[j] * a[m - j] for j in range(m))) / a[0])
    return c


def trusted_range(q, q2):
    """M_t (1-based count of leading trusted coefficients) under the pre-registered rule"""
    ratios = [abs(q[m] / q[m - 1]) for m in range(1, len(q))]          # ratio for coefficient index m (0-based)
    med = sorted(ratios[3:14])[len(ratios[3:14]) // 2]                  # coefficients m = 5..15 (1-based)
    Mt = 0
    for m in range(len(q)):
        jok = abs(q[m] - q2[m]) <= 1e-3 * abs(q[m]) if m < len(q2) else False
        rok = True if m < 4 else (med / 1.5 <= ratios[m - 1] <= med * 1.5)
        if not (jok and rok):
            break
        Mt = m + 1
    return Mt, float(med)


def sum_T(q, x, Mt):
    t = [q[m] * x ** (m + 1) for m in range(len(q))]
    S = mp.fsum(t[:Mt])
    rr = [abs(t[m] / t[m - 1]) for m in range(Mt - 6, Mt)]
    rho = sorted(rr)[len(rr) // 2]
    T = abs(t[Mt - 1]) * rho / (1 - rho)
    return S, float(T), float(rho), mp.fsum(t[Mt:])


def cnum(z, d=7):
    return "%.*f%+.*fi" % (d, float(mp.re(z)), d, float(mp.im(z)))


# ---- planted ---------------------------------------------------------------------------------------------------
def planted_f(q1, p1, version):
    """f dict of a planted profile: w_p = q1 x (1 + x e^{0.4i}/31.25)^(-1/2),
    r_3 = u_p = p1 x (1 + x e^{0.3i}/30)^(-1/3);
    F_5/F_1 = w_p u_p. version 1 perturbs the trusted coefficients by 1e-12 relative (a second J); the last three
    coefficients of each harmonic carry a 1e-14 floor that differs between versions"""
    n3 = 22
    zw, zu = mp.expj(0.4) / mp.mpf("31.25"), mp.expj(0.3) / mp.mpf(30)
    qw = [q1 * mp.binomial(-0.5, j) * zw ** j for j in range(n3)]               # coefficient of x^(j+1)
    qu = [p1 * mp.binomial(-mp.mpf(1) / 3, j) * zu ** j for j in range(n3)]
    g = [mp.fsum(qw[i] * qu[j - 1 - i] for i in range(j)) for j in range(1, n3)]  # w_p u_p: coefficient of x^(j+1)
    seed = [0.137, 0.571, 0.912, 0.318, 0.744, 0.266]
    x = mp.mpf("2.72") ** 2
    Sw = q1 * x * (1 + x * zw) ** mp.mpf(-0.5)
    f = {}
    for j in range(n3):
        v = qw[j]
        if version == 1 and j < n3 - 3:
            v = v * (1 + mp.mpf("1e-12") * mp.expj(7 * j))
        if j >= n3 - 3:
            v = v + mp.mpf("1e-14") * abs(Sw) / x ** (j + 1) * mp.expj(2 * math.pi * seed[(j + 3 * version) % 6])
        f[(2 * j + 3, 3)] = v / 2j
    for j in range(len(g)):
        v = g[j]
        if version == 1 and j < len(g) - 3:
            v = v * (1 + mp.mpf("1e-12") * mp.expj(5 * j))
        if j >= len(g) - 3:
            floor = mp.mpf("1e-14") * abs(Sw) * abs(qu[0]) / x ** (j + 2)
            v = v + floor * mp.expj(2 * math.pi * seed[(j + version) % 6])
        f[(2 * j + 5, 5)] = v / 2j
    return f, Sw, p1 * x * (1 + x * zu) ** (-mp.mpf(1) / 3)


def cmd_planted():
    fd = load_f(F1500)
    q1 = 2j * fd[(3, 3)]
    p1 = (2j * fd[(5, 5)]) / q1
    fa, Sw, Su = planted_f(q1, p1, 0)
    fb, _, _ = planted_f(q1, p1, 1)
    x = mp.mpf("2.72") ** 2
    ok = True
    for name, ca, cb, truth in (("w = F3/F1", w_coeffs(fa), w_coeffs(fb), Sw),
                                ("r_3 = F5/F3", ratio_coeffs(fa, 3), ratio_coeffs(fb, 3), Su)):
        Mt, med = trusted_range(ca, cb)
        S, T, rho, untr = sum_T(ca, x, Mt)
        err = abs(S - truth)
        good = err <= T and T <= 1e-8 * abs(truth)
        ok &= good
        print("planted %-12s: M_t %d of %d (median ratio %.4f), S %s, truth %s, |error| %.2e, T %.2e (rho %.3f), "
              "untrusted terms %.1e: %s" % (name, Mt, len(ca), med, cnum(S, 10), cnum(truth, 10), err, T, rho,
                                             abs(untr), "PASS" if good else "FAIL"))
    print("planted%s: %s" % (" (mutate=%s)" % MUT if MUT else "", "PASS" if ok else "FAIL"))
    return ok


# ---- run --------------------------------------------------------------------------------------------------------
def sum_at(f, f2, A, coeffs=w_coeffs):
    x = mp.mpf(A) ** 2
    q, q2 = coeffs(f), coeffs(f2)
    Mt, med = trusted_range(q, q2)
    S, T, rho, untr = sum_T(q, x, Mt)
    S2, _, _, _ = sum_T(q2, x, Mt)
    return dict(q=q, q2=q2, Mt=Mt, med=med, S=S, T=T, rho=rho, untrusted=untr, Jdiff=abs(S - S2))


def cmd_run():
    A = mp.mpf(args.get("A", "2.72"))
    fa, fb = load_f(F1500), load_f(F1250)
    x = A ** 2
    out = {"A": str(A)}
    r = sum_at(fa, fb, A)
    q = r["q"]
    print("w(A) = F3/F1 = sum q_m x^m, x = A^2 (K 45, omega_inf, J 1500; J 1250 for the second column)")
    print("  m  q_m (J 1500)                               |q_m(1250)/q_m(1500) - 1|  |q_m/q_{m-1}|  term at A=%s" % A)
    for m in range(len(q)):
        jd = abs(r["q2"][m] / q[m] - 1) if m < len(r["q2"]) else float("nan")
        rat = abs(q[m] / q[m - 1]) if m else float("nan")
        print("  %2d %+.10e %+.10ei   %.1e   %.4f   %.2e%s" % (m + 1, float(q[m].real), float(q[m].imag), jd, rat,
                                                              float(abs(q[m] * x ** (m + 1))),
                                                              "" if m < r["Mt"] else "  (untrusted)"))
    S, T = r["S"], r["T"]
    print("\nS = w(%s) = %s = %.9f e^{%.7f i}; M_t = %d, T = %.2e (rho %.3f, %.1e relative); untrusted terms %.1e; "
          "J 1250 - 1500: %.1e" % (A, cnum(S, 10), float(abs(S)), float(mp.arg(S)), r["Mt"], T, r["rho"],
                                   T / float(abs(S)), float(abs(r["untrusted"])), float(r["Jdiff"])))
    out["w"] = {"q": [[str(v.real), str(v.imag)] for v in q], "Mt": r["Mt"], "S": [str(S.real), str(S.imag)],
                "abs": float(abs(S)), "arg": float(mp.arg(S)), "T": T, "Jdiff": float(r["Jdiff"])}
    # partial sums: the leading order and its corrections
    print("partial sums: " + "; ".join("to x^%d: %.7f e^{%.6f i}" % (n, float(abs(ps)), float(mp.arg(ps)))
                                       for n, ps in ((n, mp.fsum(q[m] * x ** (m + 1) for m in range(n)))
                                                     for n in (1, 2, 3, 4, 6, 10))))
    # comparisons
    absS, argS = float(abs(S)), float(mp.arg(S))
    c1 = abs(absS - 0.05509) <= 5e-6 + T and abs(argS - 0.0988) <= 5e-5 + T / absS
    c2 = abs(absS - 0.055085) <= 5e-7 + T and abs(argS - 0.09879) <= 5e-6 + T / absS
    print("\ncomparison 1, the issue's 0.05509 e^{0.0988i}: |S| - 0.05509 = %+.2e (bound %.2e), arg - 0.0988 = %+.2e "
          "(bound %.2e): %s" % (absS - 0.05509, 5e-6 + T, argS - 0.0988, 5e-5 + T / absS, "PASS" if c1 else "FAIL"))
    print("comparison 2, tuned_profile_A2.72.txt's 0.055085 e^{0.09879i}: |S| - 0.055085 = %+.2e (bound %.2e), "
          "arg - 0.09879 = %+.2e (bound %.2e): %s" % (absS - 0.055085, 5e-7 + T, argS - 0.09879, 5e-6 + T / absS,
                                                      "PASS" if c2 else "FAIL"))
    # 3. M = 7 at its A, with the omega offset from the K 43 files
    A7, wf7 = mp.mpf("2.719213648"), mp.mpf("1.000019347")
    r7 = sum_at(fa, fb, A7)
    k43 = {w: load_f(p) for w, p in K43.items()}
    s43 = {w: sum_T(w_coeffs(k43[w]), A7 ** 2, 19)[0] for w in k43}
    dS = (s43[1.01] - s43[0.99]) / mp.mpf("0.02") * (wf7 - 1)
    S7 = r7["S"] + dS
    m7 = mp.mpc("0.054787", "0.005432")
    tol = 5e-7 + r7["T"]
    c3 = abs(S7.real - m7.real) <= tol and abs(S7.imag - m7.imag) <= tol
    print("comparison 3 (informational), M = 7 at A %s, omega/omega_inf %s: series %s (omega shift %s), M = 7 %s; "
          "difference %s against %.1e per component (M = 7's truncation not estimated): %s" % (
              A7, wf7, cnum(S7, 7), cnum(dS, 9), cnum(m7, 6), cnum(S7 - m7, 7), tol, "within" if c3 else "outside"))
    # leading order and the paper
    q1 = q[0]
    om = OMEGA_INF
    readme = 8 * om ** 2 - 4j * om - 1.3
    print("\nleading order: q_1 = %s, 1/q_1 = %s; README 8 omega^2 - 4i omega - 1.3 = %s (difference %s)" % (
        cnum(q1, 9), cnum(1 / q1, 5), cnum(readme, 5), cnum(1 / q1 - readme, 5)))
    print("  w ~ q_1 x at A = %s: %.6f e^{%.5f i};  the paper's fitted q e^{i delta} = 0.0551 e^{0.0985i}" % (
        A, float(abs(q1 * x)), float(mp.arg(q1 * x))))
    # higher ratios
    for nu in (3, 5):
        rr = sum_at(fa, fb, A, coeffs=lambda f, nu=nu: ratio_coeffs(f, nu))
        c = rr["q"]
        print("r_%d = F%d/F%d = sum c_m x^m: c_1..c_4 = %s; M_t %d of %d; at A: %.9f e^{%.7f i}, T %.1e, "
              "J diff %.1e" % (
            nu, nu + 2, nu, ", ".join(cnum(v, 8) for v in c[:4]), rr["Mt"], len(c), float(abs(rr["S"])),
            float(mp.arg(rr["S"])), rr["T"], float(rr["Jdiff"])))
        out["r_%d" % nu] = {"c": [[str(v.real), str(v.imag)] for v in c], "Mt": rr["Mt"], "abs": float(abs(rr["S"])),
                            "arg": float(mp.arg(rr["S"])), "T": rr["T"]}
    out.update(c1=c1, c2=c2, c3=c3)
    if "out" in args:
        json.dump(out, open(args["out"], "w"), indent=1)
    return c2


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "planted":
        sys.exit(0 if cmd_planted() else 1)
    if cmd == "run":
        sys.exit(0 if cmd_run() else 1)
    raise SystemExit(__doc__)

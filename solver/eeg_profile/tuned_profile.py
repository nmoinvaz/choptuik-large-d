#!/usr/bin/env python3
"""The D = infinity tuned echo profile of the amplitude expansion against the geometric model of Ecker, Ecker and
Grumiller, arXiv:2611.07007v1, Eq. (16) (notes/EEG_PROFILE.md section 3).

Input: the tuned higher harmonics of the inner problem's amplitude expansion at omega_inf (solver/inner_amp, the
`f` field of sigF_*.json: f[k_nu] is the coefficient of A^k in the e^{i nu theta} harmonic of
F = A sin(theta) + sum_{nu >= 3} f_nu(A) e^{i nu theta} + c.c., tuned so that S_nu = 0 for nu >= 3). With F_1 = -iA/2,
F_nu = sum_k f[k_nu] A^k is summed at the chosen A.

Map to the paper's variables (notes/EEG_PROFILE.md section 2.1): bt = sqrt(2) F at D = infinity, theta_EEG = theta - pi/2,
so the paper's e^{i nu theta} coefficient is C_nu = i^nu F_nu (C_1 = A/2 > 0) and its complex ratio
w = q e^{i delta} = -C_{nu+2}/C_nu equals r_nu = F_{nu+2}/F_nu of this repository's gauge.

Printed: the harmonics and ratios r_nu, the local decay rate (1/2) ln(1/|r_nu|), the paper's fit (its Eqs. (S61)-(S62):
A_* tied to <bt^2>, chi^2 over the first four harmonics) and free fits, the deviation of each harmonic from the
geometric law, and the late-ratio limit, with the spread between two J (late-order fit resolutions) of the same K.

    python3 tuned_profile.py [--A 2.72] [JSON ...]     default JSON: scan45J1250 and scan45J1500 (K = 45, omega_inf)
    python3 tuned_profile.py --plant                   planted truth: an exact geometric profile through the same path;
                                                       the fit must return (q, delta) to 1e-10 and zero drift
    python3 tuned_profile.py --plant --mutate=drift    planted r_nu = w (1 + 0.3/nu): the geometric test must FAIL
    python3 tuned_profile.py --plant --mutate=sign_q   fit with the ratio +w (1 - q e^{..}): recovery must FAIL

Exit status 1 when a planted check fails (or, under --mutate, when it fails as it must).
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
AMP = os.path.join(HERE, "..", "inner_amp")
DEFAULT = [os.path.join(AMP, "scan45J1250", "sigF_w1.0000_K45_J1250_p1280.json"),
           os.path.join(AMP, "scan45J1500", "sigF_w1.0000_K45_J1500_p1280.json")]
Q_EEG, D_EEG, A_EEG = 0.0551, 0.0985, 3.8455      # arXiv:2611.07007v1 Eq. (16) and the text after it

ap = argparse.ArgumentParser()
ap.add_argument("json", nargs="*", default=None)
ap.add_argument("--A", type=float, default=2.72)
ap.add_argument("--plant", action="store_true")
ap.add_argument("--mutate", default=None, choices=[None, "drift", "sign_q"])
ap.add_argument("--numax", type=int, default=25)
args = ap.parse_args()
SGN = -1.0 if args.mutate == "sign_q" else 1.0     # model ratio -SGN*w between successive paper-gauge harmonics


def load_f(path):
    d = json.load(open(path))
    f = {}
    for key, (re_, im_) in d["f"].items():
        k, nu = map(int, key.split("_"))
        f[(k, nu)] = complex(re_, im_)
    return d["K"], d["J"], f


def harmonics(f, K, A, numax):
    """F_nu (nu odd, 1..numax) at amplitude A, the number of A-terms and the size of the last term relative to F_nu."""
    F = {1: -0.5j * A}
    last = {1: 0.0}
    nterms = {1: 1}
    for nu in range(3, numax + 1, 2):
        terms = [f.get((k, nu), 0) * A ** k for k in range(nu, K + 1, 2)]
        F[nu] = sum(terms)
        last[nu] = abs(terms[-1]) / abs(F[nu]) if terms else float("nan")
        nterms[nu] = len(terms)
    return F, last, nterms


def planted_f(A, K, numax, drift):
    """f[k_nu] that sum, at amplitude A, to an exact geometric profile r_nu = w (or w (1 + 0.3/nu) under drift)."""
    w = Q_EEG * np.exp(1j * D_EEG)
    f, Fnu = {}, -0.5j * A
    for nu in range(1, numax - 1, 2):
        Fnu = Fnu * w * ((1 + 0.3 / nu) if drift else 1)
        f[(nu + 2, nu + 2)] = Fnu / A ** (nu + 2)
    return f


def paper_coeffs(F, nus):
    """c_n - i s_n of bt = sqrt(2) F in the paper's gauge (Eq. (S53)), n = (nu - 1)/2: 2 sqrt(2) i^nu F_nu."""
    return np.array([2 * math.sqrt(2) * (1j ** nu) * F[nu] for nu in nus])


def model(Astar, q, dl, n):
    return Astar * (-SGN * q) ** n * np.exp(1j * n * dl)


def fit_paper(b, N0, nfit=4):
    """The paper's procedure (S61)-(S62): A_* = sqrt(2 N0 (1 - q^2)), least squares over the first nfit harmonics."""
    n = np.arange(nfit)
    res = lambda p: np.r_[(b[:nfit] - model(math.sqrt(2 * N0 * (1 - p[0] ** 2)), p[0], p[1], n)).real,
                          (b[:nfit] - model(math.sqrt(2 * N0 * (1 - p[0] ** 2)), p[0], p[1], n)).imag]
    r = least_squares(res, [0.05, 0.1], xtol=1e-15, ftol=1e-15, gtol=1e-15)
    q, dl = r.x
    return math.sqrt(2 * N0 * (1 - q ** 2)), q, dl, math.sqrt(np.sum(r.fun ** 2))


def fit_free(b, nfit):
    n = np.arange(nfit)
    wt = 1 / np.abs(b[:nfit])                                  # relative residuals, every harmonic counts
    res = lambda p: np.r_[((b[:nfit] - model(p[0], p[1], p[2], n)) * wt).real,
                          ((b[:nfit] - model(p[0], p[1], p[2], n)) * wt).imag]
    r = least_squares(res, [3.8, 0.05, 0.1], xtol=1e-15, ftol=1e-15, gtol=1e-15)
    return (*r.x, math.sqrt(np.sum(r.fun ** 2) / nfit))


def analyse(F, last, nterms, label, numax):
    nus = list(range(1, numax + 1, 2))
    r = {nu: F[nu + 2] / F[nu] for nu in nus[:-1]}
    b = paper_coeffs(F, nus)
    N0 = 0.5 * float(np.sum(np.abs(b) ** 2))                 # <bt^2> = (1/2) sum (c_n^2 + s_n^2)
    print(f"\n== {label}")
    print(f"<F^2> = {N0 / 2:.6f}   <bt^2> = N0 = {N0:.6f}   (paper: N0 = (Delta_1 - 4 pi mu_1)/(4 pi mu_0) = 7.4158)")
    print(" nu  |F_nu/F_1|     arg(F_nu/F_1)  A-terms last/|F|   r_nu=F_{nu+2}/F_nu  |r_nu|    arg r_nu  rate (1/2)ln(1/|r|)")
    for nu in nus[:-1]:
        x = F[nu] / F[1]
        print(f"{nu:3d}  {abs(x):.6e}  {np.angle(x):+.5f}      {nterms[nu]:3d}   {last[nu]:.1e}    "
              f"{r[nu].real:+.6f}{r[nu].imag:+.6f}i  {abs(r[nu]):.6f}  {np.angle(r[nu]):+.5f}  {0.5 * math.log(1 / abs(r[nu])):.5f}")
    Ap, qp, dp, rp = fit_paper(b, N0)
    print(f"paper's fit (S61)-(S62), n = 0..3: A_* = {Ap:.5f}  q = {qp:.6f}  delta = {dp:.6f}  (residual {rp:.2e});"
          f"  A_*/sqrt2 = {Ap / math.sqrt(2):.5f}")
    fits = {"paper": (Ap, qp, dp)}
    for nfit in (4, 8):
        Af, qf, df, rf = fit_free(b, nfit)
        fits[f"free{nfit}"] = (Af, qf, df)
        print(f"free fit n = 0..{nfit - 1} (relative residuals): A_* = {Af:.5f}  q = {qf:.6f}  delta = {df:.6f}"
              f"  (rms relative residual {rf:.2e})")
    print("deviation of each harmonic from the paper's model (q, delta, A_* as printed in arXiv:2611.07007) and from the"
          " model through harmonics 1 and 3 (w = r_1):")
    w1 = r[1]
    print(" nu   |C/C_EEG|-1   arg(C/C_EEG)    |C/C_w1|-1   arg(C/C_w1)")
    for i, nu in enumerate(nus[:-1]):
        m_eeg = model(A_EEG, Q_EEG, D_EEG, i)
        m_w1 = b[0] * (-w1) ** i
        print(f"{nu:3d}   {abs(b[i] / m_eeg) - 1:+.4e}   {np.angle(b[i] / m_eeg):+.5f}      "
              f"{abs(b[i] / m_w1) - 1:+.4e}   {np.angle(b[i] / m_w1):+.5f}")
    return r, fits, N0


def late_limit(r, nus):
    """|r_nu| = rho (1 + b/nu + c/nu^2) and arg r_nu = a0 + a1/nu on the window nus (least squares)."""
    x = 1 / np.array(nus, float)
    y = np.array([abs(r[nu]) for nu in nus])
    ya = np.array([np.angle(r[nu]) for nu in nus])
    c = np.polyfit(x, y, 2)
    ca = np.polyfit(x, ya, 1)
    return c[-1], c[-2] / c[-1], ca[-1]


paths = args.json or DEFAULT
if args.plant:
    K, numax = 45, args.numax
    runs = [(f"planted geometric profile{' with 1/nu drift' if args.mutate == 'drift' else ''}", K,
             planted_f(args.A, K, numax + 2, args.mutate == "drift"))]
else:
    runs = []
    for p in paths:
        K, J, f = load_f(p)
        runs.append((f"{os.path.relpath(p, HERE)}  (K {K}, J {J}), A = {args.A}", K, f))

out = []
for label, K, f in runs:
    F, last, nterms = harmonics(f, K, args.A, args.numax + 2)
    r, fits, N0 = analyse(F, last, nterms, label, args.numax + 2)
    out.append((F, r, fits))
    for win in ((9, 11, 13, 15, 17), (11, 13, 15, 17, 19), (13, 15, 17, 19, 21)):
        if max(win) in r:
            rho, b1, a0 = late_limit(r, win)
            print(f"late-ratio fit on nu = {win[0]}..{win[-1]}: |r_inf| = {rho:.5f} (1 + {b1:+.3f}/nu ...), "
                  f"arg r_inf = {a0:+.4f}, sigma_inf = (1/2) ln(1/|r_inf|) = {0.5 * math.log(1 / rho):.4f}")
    # poles of the model and nearest singularity of the profile, in the paper's gauge: Re t = (pi - arg w)/2
    rinf = r[19] if 19 in r else r[max(r)]
    print(f"nearest singularity, paper's gauge: model with w = r_1: t = {(math.pi - np.angle(r[1])) / 2:.4f} -/+ "
          f"{0.5 * math.log(1 / abs(r[1])):.4f} i;  paper's (q, delta): {(math.pi - D_EEG) / 2:.4f} -/+ "
          f"{0.5 * math.log(1 / Q_EEG):.4f} i;  profile (r_19): {(math.pi - np.angle(rinf)) / 2:.4f} -/+ "
          f"{0.5 * math.log(1 / abs(rinf)):.4f} i")
    # approach of r_nu to its limit: power law |r_nu - r_19| ~ nu^-p on nu = 5..13
    nn = [nu for nu in (5, 7, 9, 11, 13) if nu in r]
    if len(nn) == 5 and not args.plant:
        dv = [abs(r[nu] - rinf) for nu in nn]
        p = -np.polyfit(np.log(nn), np.log(dv), 1)[0]
        print("|r_nu - r_19| for nu = 5..13: " + ", ".join(f"{x:.2e}" for x in dv) + f"  (power-law slope {p:.2f})")
    # one pole pair with a zero: e^{it} (1 + b e^{2it}) / (1 + w e^{2it}), Pade [1/1] in e^{2it}, exact on nu = 1, 3, 5
    wv, bv = r[3], r[3] - r[1]
    nus = sorted(F)
    print(f"[1/1] model through nu = 1, 3, 5: w = {wv:.6f} (|w| {abs(wv):.6f}, arg {np.angle(wv):+.5f}), "
          f"b = {bv:.6f} (|b| {abs(bv):.2e})")
    dev = []
    for nu in nus:
        if nu < 7 or nu > 19:
            continue
        Fm = F[3] * wv ** ((nu - 3) // 2)
        dev.append(f"nu {nu}: {abs(F[nu] / Fm) - 1:+.2e}")
    print("  deviation |F/F_model| - 1, " + ", ".join(dev))

if len(out) == 2 and not args.plant:
    print("\nspread between the two inputs (different J), |r_nu| and arg r_nu:")
    for nu in sorted(out[0][1]):
        a, b2 = out[0][1][nu], out[1][1][nu]
        print(f" nu {nu:2d}: d|r| = {abs(a) - abs(b2):+.2e}  d arg = {np.angle(a) - np.angle(b2):+.2e}")

if args.plant:
    F, r, fits = out[0]
    _, qp, dp = fits["paper"]
    ok_fit = abs(qp - Q_EEG) < 1e-10 and abs(dp - D_EEG) < 1e-10
    drift = max(abs(r[nu] / r[1] - 1) for nu in r)
    ok_geo = drift < 1e-10
    print(f"\nplanted: recovered q - q0 = {qp - Q_EEG:+.2e}, delta - delta0 = {dp - D_EEG:+.2e}  -> {'PASS' if ok_fit else 'FAIL'}")
    print(f"planted: max_nu |r_nu/r_1 - 1| = {drift:.2e} (geometric law)  -> {'PASS' if ok_geo else 'FAIL'}")
    sys.exit(0 if (ok_fit and ok_geo) else 1)

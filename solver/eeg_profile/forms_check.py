#!/usr/bin/env python3
"""Equivalent forms of the large-D echo-profile model of Ecker, Ecker and Grumiller, arXiv:2611.07007v1, Eq. (16),

    bt(theta) = A Re[ e^{i theta} / (1 + q e^{i(2 theta + delta)}) ],     w = q e^{i delta},  0 <= q < 1,

checked symbolically (sympy, exact rational identities in z = e^{i theta}, e = e^{i delta}) and numerically (mpmath,
50 digits). The forms (notes/EEG_PROFILE.md section 1):

  R   real closed form       A (cos t + q cos(t + d)) / (1 + q^2 + 2 q cos(2t + d))
  F   Fourier series         A sum_k (-q)^k cos((2k+1) t + k d), remainder after N terms <= A q^{N+1}/(1 - q)
  H   reciprocal oscillator  A Re[1 / (e^{-it} + w e^{it})], with h = e^{-it} + w e^{it} solving h'' + h = 0
  E   ellipse (shifted)      A [(1+q) cos p cos(d/2) + (1-q) sin p sin(d/2)] / [(1+q)^2 cos^2 p + (1-q)^2 sin^2 p],
                             p = t + d/2, i.e. A Re[e^{-i d/2} G_q(p)], G_q = 1/(e^{-ip} + q e^{ip})
  M   AM-FM carrier          A cos(t - chi) / sqrt(1 + q^2 + 2 q cos(2t + d)), chi = arg(1 + w e^{2it})
  P   Poisson kernel         A (cos t + q cos(t + d)) P_q(2t + d + pi) / (1 - q^2), P_q = (1-q^2)/(1 - 2q cos + q^2)
  PF  partial fractions      e^{it}/(1 + w e^{2it}) = [1/(1 - a z) - 1/(1 + a z)] / (2a), a = i sqrt(w)
  ODE linear                 (d^2 + 1)[(1 + q^2 + 2 q cos(2t + d)) bt] = 0; nonlinear g g'' = 2 g'^2 + g^2 for g = bt + i bt~
  poles                      t = (pi - d)/2 + k pi +- (i/2) ln(1/q), simple, residue (A i/4) e^{i t_p} for the lower ones
  zeros, extrema             zeros at t = pi/2 - arg(1 + w) + k pi; one maximum and one minimum per period while
                             q < q_c(d), with q_c(0) = 3 - 2 sqrt 2
  norm                       <bt^2> = A^2 / (2 (1 - q^2)) for every d
  symmetries                 (q, d) ~ (-q, d + pi); bt(-t; q, d) = bt(t; q, -d); d is not removable by a shift of t
  INT antiderivative         int bt dt = A Re[-i w^{-1/2} arctan(w^{1/2} e^{it})]; at d = 0, A artanh(a sin t)/(2 sqrt q),
                             a = 2 sqrt(q)/(1 + q) (the form of Rozali and Way's oscillaton at a fixed radius)
  NLO                        Eq. (20) of arXiv:2601.14358 evaluated on Eq. (16) (control: their Eq. (21) gives 1)

    python3 forms_check.py                 # every check passes, exit 0
    python3 forms_check.py --mutate=X      # X in sign_q, fourier_phase, pole_dist, norm, recip, ellipse,
                                           atanh: must FAIL

Exit status 0 only if every check holds (and, with --mutate, 1 when the mutated check fails, as it must).
"""
import sys

import mpmath as mp
import sympy as sp

mut = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--mutate=")), None)
MUTS = ("sign_q", "fourier_phase", "pole_dist", "norm", "recip", "ellipse", "atanh")
if mut is not None and mut not in MUTS:
    sys.exit(f"unknown mutation {mut}; one of {MUTS}")

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


# ---------------------------------------------------------------- symbolic: rational identities on the unit circle
z, e, q, A, a = sp.symbols("z e q A a", nonzero=True)
w = q * e
cj = {z: 1 / z, e: 1 / e}                       # conjugation on |z| = |e| = 1, q and A real
cos_t, sin_t = (z + 1 / z) / 2, (z - 1 / z) / (2 * sp.I)
cos_d, sin_d = (e + 1 / e) / 2, (e - 1 / e) / (2 * sp.I)
cos_td = (z * e + 1 / (z * e)) / 2               # cos(t + d)
cos_2td = (z ** 2 * e + 1 / (z ** 2 * e)) / 2    # cos(2t + d)

g = z / (1 + w * z ** 2)                         # Eq. (16) inside Re[]
bt = A * (g + g.subs(cj, simultaneous=True)) / 2  # A Re[g]

# R: the real closed form
sq = -1 if mut == "sign_q" else 1
R = A * (cos_t + sq * q * cos_td) / (1 + q ** 2 + 2 * sq * q * cos_2td)
check("R  real closed form = A Re[g]", sp.cancel(sp.together(bt - R)) == 0)

# F: partial sums of the Fourier series; the remainder is exact
N = 6
ph = -1 if mut == "fourier_phase" else 1
SN = sum(A * (-q) ** k * (z ** (2 * k + 1) * e ** (ph * k) + z ** (-(2 * k + 1)) * e ** (-ph * k)) / 2
         for k in range(N + 1))
rem = A * (z * (-w * z ** 2) ** (N + 1) / (1 + w * z ** 2))
rem = (rem + rem.subs(cj, simultaneous=True)) / 2
check(f"F  Fourier partial sum (N = {N}) + exact remainder = A Re[g]", sp.cancel(sp.together(SN + rem - bt)) == 0)

# H: reciprocal oscillator
h = 1 / z + w * z
hh = 1 / z + (w * z if mut != "recip" else -w * z)
check("H  g = 1/(e^{-it} + w e^{it})", sp.cancel(g - 1 / hh) == 0)
t = sp.symbols("t", real=True)
qq, dd = sp.symbols("q delta", real=True)
ht = sp.exp(-sp.I * t) + qq * sp.exp(sp.I * dd) * sp.exp(sp.I * t)
check("H  h'' + h = 0", sp.simplify(sp.diff(ht, t, 2) + ht) == 0)
gt = 1 / ht
check("H  g g'' = 2 g'^2 + g^2 (autonomous, nonlinear)",
      sp.simplify(gt * sp.diff(gt, t, 2) - 2 * sp.diff(gt, t) ** 2 - gt ** 2) == 0)

# E: ellipse / shifted form, p = t + d/2, written with s = e^{i d/2} (e = s^2) and u = e^{i p} = z s
s, u = sp.symbols("s u", nonzero=True)
cos_p, sin_p = (u + 1 / u) / 2, (u - 1 / u) / (2 * sp.I)
cos_h, sin_h = (s + 1 / s) / 2, (s - 1 / s) / (2 * sp.I)
eh = -1 if mut == "ellipse" else 1
E = A * ((1 + q) * cos_p * cos_h + eh * (1 - q) * sin_p * sin_h) / ((1 + q) ** 2 * cos_p ** 2 + (1 - q) ** 2 * sin_p ** 2)
bt_s = bt.subs({e: s ** 2}).subs({z: u / s})
check("E  ellipse form in p = t + d/2", sp.cancel(sp.together(bt_s - E)) == 0)
Gq = u / (1 + q * u ** 2)
Hil = A * (Gq / s + (Gq / s).subs({u: 1 / u, s: 1 / s}, simultaneous=True)) / 2
check("E  bt = A Re[e^{-i d/2} G_q(t + d/2)]", sp.cancel(sp.together(bt_s - Hil)) == 0)

# P: Poisson kernel
Pk = lambda rho, cphi: (1 - rho ** 2) / (1 - 2 * rho * cphi + rho ** 2)
P = A * (cos_t + q * cos_td) * Pk(q, -cos_2td) / (1 - q ** 2)   # cos(2t + d + pi) = -cos(2t + d)
check("P  Poisson-kernel form", sp.cancel(sp.together(bt - P)) == 0)

# PF: partial fractions, w = -a^2
PF = (1 / (1 - a * z) - 1 / (1 + a * z)) / (2 * a)
check("PF partial fractions with a = i sqrt(w)", sp.cancel(PF - z / (1 - a ** 2 * z ** 2)) == 0)

# ODE: (d^2 + 1)[Dn bt] = 0 in real variables
Ar = sp.symbols("A_r", positive=True)
btr = Ar * (sp.cos(t) + qq * sp.cos(t + dd)) / (1 + qq ** 2 + 2 * qq * sp.cos(2 * t + dd))
Dn = 1 + qq ** 2 + 2 * qq * sp.cos(2 * t + dd)
check("ODE (d^2 + 1)[(1 + q^2 + 2q cos(2t + d)) bt] = 0", sp.simplify(sp.diff(Dn * btr, t, 2) + Dn * btr) == 0)

# symmetries
btr_neg = btr.subs({qq: -qq, dd: dd + sp.pi})
check("SYM (q, d) -> (-q, d + pi) leaves bt unchanged", sp.simplify(sp.expand_trig(btr_neg - btr)) == 0)
check("SYM bt(-t; q, d) = bt(t; q, -d)", sp.simplify(sp.expand_trig(btr.subs(t, -t) - btr.subs(dd, -dd))) == 0)

# ---------------------------------------------------------------- numerics, 50 digits, at the paper's parameters
mp.mp.dps = 50
QE, DE, AE = mp.mpf("0.0551"), mp.mpf("0.0985"), mp.mpf("3.8455")


def btn(th, Q=QE, Dl=DE, Am=AE):
    return Am * mp.re(mp.exp(1j * th) / (1 + Q * mp.exp(1j * (2 * th + Dl))))


def bt_c(th, Q=QE, Dl=DE, Am=AE):
    """Analytic continuation of bt off the real axis: A/2 [g(t) + conj(g)(t)]."""
    wq = Q * mp.exp(1j * Dl)
    return Am / 2 * (1 / (mp.exp(-1j * th) + wq * mp.exp(1j * th))
                     + 1 / (mp.exp(1j * th) + mp.conj(wq) * mp.exp(-1j * th)))


ths = [mp.mpf(k) * 2 * mp.pi / 37 + mp.mpf("0.013") for k in range(37)]
fs = [btn(x) for x in ths]
err = max(abs(sum(AE * (-QE) ** k * mp.cos((2 * k + 1) * x + k * DE) for k in range(60)) - f) for x, f in zip(ths, fs))
check("F  60-term series vs closed form (num.)", err < max(AE * QE ** 60 / (1 - QE), mp.mpf(10) ** -45),
      f"max err {mp.nstr(err, 3)}")
chi = lambda x: mp.arg(1 + QE * mp.exp(1j * (2 * x + DE)))
err = max(abs(AE * mp.cos(x - chi(x)) / mp.sqrt(1 + QE ** 2 + 2 * QE * mp.cos(2 * x + DE)) - f) for x, f in zip(ths, fs))
check("M  AM-FM carrier form (num.)", err < mp.mpf(10) ** -45, f"max err {mp.nstr(err, 3)}")
err = max(abs(bt_c(x) - f) for x, f in zip(ths, fs))
check("   continuation equals bt on the real axis (num.)", err < mp.mpf(10) ** -45, f"max err {mp.nstr(err, 3)}")

# poles: t_p = (pi - d)/2 + k pi -/+ i sigma, sigma = (1/2) ln(1/q)
sig = mp.log(1 / QE) * (1 if mut == "pole_dist" else mp.mpf(1) / 2)
for k in (0, 1):
    for sgn in (-1, 1):
        tp = (mp.pi - DE) / 2 + k * mp.pi + sgn * 1j * sig
        eps = mp.mpf(10) ** -20
        res = eps * bt_c(tp + eps)
        exp_res = (AE * 1j / 4) * mp.exp(1j * tp) if sgn < 0 else -(AE * 1j / 4) * mp.exp(-1j * tp)
        ok = abs(res - exp_res) < mp.mpf(10) ** -15 * abs(exp_res)
        check(f"POLE t = (pi-d)/2 + {k} pi {'-' if sgn < 0 else '+'} i sigma: residue", ok,
              f"res {mp.nstr(res, 8)}  formula {mp.nstr(exp_res, 8)}")
check("POLE sigma = (1/2) ln(1/q) (paper's q)", abs(sig - mp.mpf("1.44930")) < 1e-4, f"sigma = {mp.nstr(sig, 6)}")

# zeros and extrema per period
grid = [mp.mpf(k) * 2 * mp.pi / 4000 + mp.mpf("1.234e-4") for k in range(4000)]   # offset: no grid point on a zero


def count_sign_changes(f):
    v = [f(x) for x in grid]
    return sum(1 for i in range(len(v)) if v[i - 1] * v[i] < 0)


th0 = mp.pi / 2 - mp.arg(1 + QE * mp.exp(1j * DE))
check("ZERO bt(pi/2 - arg(1 + w)) = 0, two zeros per period",
      abs(btn(th0)) < mp.mpf(10) ** -45 and abs(btn(th0 + mp.pi)) < mp.mpf(10) ** -45 and count_sign_changes(btn) == 2,
      f"theta_0 = {mp.nstr(th0, 10)}")


def n_extrema(Q, Dl):
    def d1(x):                                   # d/dt Re[g] = Re[i z (1 - w z^2) / (1 + w z^2)^2]
        zz, wq = mp.exp(1j * x), Q * mp.exp(1j * Dl)
        return mp.re(1j * zz * (1 - wq * zz ** 2) / (1 + wq * zz ** 2) ** 2)
    return count_sign_changes(d1)


qc0 = 3 - 2 * mp.sqrt(2)
check("EXT 2 extrema per period at the paper's (q, d)", n_extrema(QE, DE) == 2)
check("EXT q_c(0) = 3 - 2 sqrt 2: 2 extrema below, 6 above",
      n_extrema(qc0 - mp.mpf("0.002"), 0) == 2 and n_extrema(qc0 + mp.mpf("0.002"), 0) == 6, f"q_c = {mp.nstr(qc0, 8)}")

# norm (Parseval) at several delta
nf = (lambda Q: 2 * (1 + Q ** 2)) if mut == "norm" else (lambda Q: 2 * (1 - Q ** 2))
ok = True
for Dl in (0, DE, 1, 2.5):
    I = mp.quad(lambda x: btn(x, QE, Dl) ** 2, [0, mp.pi / 2, mp.pi, 3 * mp.pi / 2, 2 * mp.pi]) / (2 * mp.pi)
    ok = ok and abs(I - AE ** 2 / nf(QE)) < mp.mpf(10) ** -30
check("NORM <bt^2> = A^2/(2(1 - q^2)) for d = 0, 0.0985, 1, 2.5", ok, f"= {mp.nstr(AE ** 2 / nf(QE), 10)}")

# delta is not a shift: the best shift of the d = 0 profile leaves a residual of order A sin(d/2) |Hilbert part|
# (mean square from the exact Fourier coefficients A/2 (-q)^k e^{i k d} of harmonic 2k+1, checked by F above)
mp.mp.dps = 20
ms = lambda sh: sum(2 * (AE / 2) ** 2 * QE ** (2 * k) * abs(mp.exp(1j * (k * DE + (2 * k + 1) * sh)) - 1) ** 2
                    for k in range(40))
best = mp.findroot(lambda sh: mp.diff(ms, sh), -DE / 2)
rms = mp.sqrt(ms(best))
check("SYM d is not removable by a shift (best-shift rms residual > 0)", rms > mp.mpf("1e-4"),
      f"best shift {mp.nstr(best, 6)} (-d/2 = {mp.nstr(-DE / 2, 6)}), rms {mp.nstr(rms, 4)}")

# INT: the antiderivative (the paper's Eq. (12): the central scalar field is the tau-integral of bt at leading order)
mp.mp.dps = 50
check("INT d/dt[-i w^{-1/2} arctan(w^{1/2} e^{it})] = e^{it}/(1 + w e^{2it})",
      sp.simplify(sp.diff(-sp.I / sp.sqrt(qq * sp.exp(sp.I * dd)) * sp.atan(sp.sqrt(qq * sp.exp(sp.I * dd))
                                                                           * sp.exp(sp.I * t)), t)
                  - sp.exp(sp.I * t) / (1 + qq * sp.exp(sp.I * dd) * sp.exp(2 * sp.I * t))) == 0)
aa = 2 * mp.sqrt(QE) / ((1 - QE) if mut == "atanh" else (1 + QE))
err = max(abs(mp.re(-1j / mp.sqrt(QE) * mp.atan(mp.sqrt(QE) * mp.exp(1j * x)))
              - mp.atanh(aa * mp.sin(x)) / (2 * mp.sqrt(QE))) for x in ths)
check("INT d = 0: integral of Re[g] = artanh(a sin t)/(2 sqrt q), a = 2 sqrt(q)/(1 + q)", err < mp.mpf(10) ** -45,
      f"max err {mp.nstr(err, 3)}")

# NLO condition of arXiv:2601.14358, Eq. (20): Delta = |b''|/(3|b'|) at b = 0, b of unit period (b(u) = bt(2 pi u)).
# Control: their Eq. (21), b = cos(2 pi u) + sin(6 pi u)/15.9476, was built for Delta = 1.


def nlo_delta(b, u0):
    u = mp.findroot(b, u0)
    return abs(mp.diff(b, u, 2)) / (3 * abs(mp.diff(b, u, 1))), u


d21, _ = nlo_delta(lambda u: mp.cos(2 * mp.pi * u) + mp.sin(6 * mp.pi * u) / mp.mpf("15.9476"), mp.mpf("0.25"))
d16, u16 = nlo_delta(lambda u: btn(2 * mp.pi * u), mp.mpf("0.25"))
check("NLO control: Eq. (21) of arXiv:2601.14358 gives Delta = 1", abs(d21 - 1) < 1e-4, f"Delta = {mp.nstr(d21, 7)}")
# at a zero of N = cos t + q cos(t + d), N'' = -N = 0, so bt''/bt' = -2 Dn'/Dn there, Dn' = -4 q sin(2t + d)
t0 = 2 * mp.pi * u16
pred = 2 * mp.pi / 3 * 2 * abs(4 * QE * mp.sin(2 * t0 + DE)) / (1 + QE ** 2 + 2 * QE * mp.cos(2 * t0 + DE))
check("NLO Eq. (16) at the paper's (q, d): Delta = (2 pi/3) |2 Dn'/Dn| at the zero", abs(d16 - pred) < mp.mpf(10) ** -30,
      f"Delta = {mp.nstr(d16, 6)} (formula {mp.nstr(pred, 6)}; zero at theta = {mp.nstr(t0, 8)}), not 1.6439")

bad = [n for n, ok in results if not ok]
print(f"\n{len(results) - len(bad)}/{len(results)} checks pass" + (f"; FAILED: {bad}" if bad else ""))
sys.exit(1 if bad else 0)

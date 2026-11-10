# EEG_PROFILE: the large-D echo-profile model of arXiv:2611.07007, Eq. (16)

**Verdict.** Eq. (16) of Ecker, Ecker and Grumiller (EEG, arXiv:2611.07007v1, 4 Nov 2026) is the simplest profile with
a geometric harmonic law. With one complex number $`w = qe^{i\delta}`$ in the unit disk it reads

```math
\tilde\beta(\theta) = A_*\,\mathrm{Re}\,\frac{1}{e^{-i\theta} + w\,e^{i\theta}} ,
```

the real part of the reciprocal of a harmonic oscillator (an ellipse in the complex plane). It has one simple pole pair
per half period, at $`\theta = (\pi-\delta)/2 \pm \tfrac{i}{2}\ln(1/q)`$, so its strip of analyticity and its decay
rate per harmonic are the same number, 1.4493 at the paper's $`q`$. The phase $`\delta`$ cannot be removed by a shift
of $`\theta`$. It is the echo's asymmetry under time reversal, and the sign of $`q`$ is redundant with it.

Eq. (16) is the repository's own "leading structure" form (README, math paper "The shape of the echo"). Mapped to our
variables, EEG's fitted $`(q, \delta) = (0.0551, 0.0985)`$ is the first ratio $`F_3/F_1`$ of the $`D = \infty`$ tuned
profile at A = 2.72, $`0.05508\,e^{0.0988i}`$. That profile comes from the inner equations alone, while EEG extrapolated
solutions to D = 200, so the two share no code. Applied to our tuned profile, EEG's own fitting procedure returns
$`(0.05510, 0.09866)`$ (*numerical evidence*).

Our tuned profile is not exactly geometric. The successive ratios grow from 0.0551 to a limit of 0.05745, and their
phase falls from 0.0988 to 0.0726. With the paper's parameters, Eq. (16) is 3.8% low at $`|F_5|`$, 8.1% at $`|F_7|`$
and 39% at $`|F_{19}|`$. It also puts the nearest singularity at $`|\mathrm{Im}\,\theta| = 1.449`$ instead of the
profile's 1.428. One extra parameter, a zero in the numerator, brings every harmonic through $`\nu = 19`$ within 3%.

Scripts: [`solver/eeg_profile/`](../solver/eeg_profile/); their outputs are in `solver/eeg_profile/data/`. Status words
are as in AGENTS.md.

## 1. The model in the paper

From the HTML of v1, read 2026-11-06, main text Sec. V and supplement VII.4 (S52)–(S62):

- **The function.** $`\tilde\beta(\tau)`$ is the free function of their large-D solution, rescaled by their Eq. (10),
  $`\beta = \sqrt\epsilon\,\tilde\beta`$ with $`\epsilon = 1/(D-1)`$. At the centre,
  $`\Pi = \sqrt\epsilon\,\tilde\beta`$ (their Eq. (14)), and (S52) defines
  $`\tilde\beta_D(\theta) = \Pi_D(\Delta\theta/2\pi, 0)/\sqrt\epsilon`$. Here $`\tau`$ is their GG time, and the
  leading-order central scalar field is $`\phi = \oint\tilde\beta\,d\tau`$ (their Eq. (12)).
- **The phase.** $`\theta = 2\pi\tau/\Delta_0 = \tau/(2\mu_0)`$, fixed by their fits $`\Delta_0 \approx 1.6439`$ and
  $`\mu_0 \approx 0.130818`$. The time origin is fixed so that the fundamental is a pure $`\cos\theta`$
  ($`c_0 > 0`$, $`s_0 = 0`$).
- **Amplitude.** $`A_* = [2(1-q^2)\mathcal N_0]^{1/2} \approx 3.8455`$, where
  $`\mathcal N_0 = (\Delta_1 - 4\pi\mu_1)/(4\pi\mu_0) = \langle\tilde\beta^2\rangle`$ (their Eq. (15), (S59)–(S61)).
  It is therefore not fitted. Their printed $`\Delta_1 = 10.37`$, $`\mu_1 = -0.1449`$ give $`\mathcal N_0 = 7.4158`$,
  while $`A_* = 3.8455`$ with $`q = 0.0551`$ implies 7.4165, so they used more digits than they print.
- **$`q`$ and $`\delta`$** are the only fitted parameters, $`q \approx 0.0551`$ and $`\delta \approx 0.0985`$. They are
  found by least squares on the first four Fourier coefficients of the profile extrapolated in $`\epsilon`$ coefficient
  by coefficient ((S55), (S62)). The text calls $`q`$ "a damping constant" and $`\delta`$ "a phase", and quotes
  $`\alpha = -\ln q \approx 2.90`$ per $`n`$, the decay over a pair of harmonics.
- **Motivation and use.** The amplitudes of the extrapolated coefficients fall "approximately geometrically",
  $`A_n \simeq A_*q^n`$, and their phases advance linearly, $`\varphi_n \simeq n(\pi+\delta)`$ ((S57)–(S58)). The
  supplement itself gives the Fourier series and the real closed form (S58), and the norm
  $`A_*^2/(2(1-q^2))`$ (S60). The model is put into their NNLO expressions for $`f, \Omega, \Pi, \Psi`$ and compared
  with the D = 200 solution (their Fig. 4) and with D = 25–200 (Fig. 5, relative rms deviations). It replaces the
  two-mode model of their earlier paper, Eq. (21) of arXiv:2601.14358. The outlook asks for "a derivation of the
  large-D approximation (16) and the fitting parameters therein".

## 2. The forms

Write $`z = e^{i\theta}`$, $`w = qe^{i\delta}`$ and $`D_q(\theta) = 1 + q^2 + 2q\cos(2\theta+\delta) = |1 + wz^2|^2`$.
Each line below is an identity checked by `forms_check.py`, symbolically (sympy, as rational identities in $`z`$ and
$`e^{i\delta}`$ on the unit circle, or in real variables) or numerically (mpmath at 50 digits). Where a form has its own
mutation, which must break its check, the mutation is named.

**(R) Real closed form** (*checked symbolically*; mutation `sign_q`):

```math
\tilde\beta = A_*\,\frac{\cos\theta + q\cos(\theta+\delta)}{1 + q^2 + 2q\cos(2\theta+\delta)} .
```

The numerator is a pure first harmonic, $`|1+w|\cos(\theta + \arg(1+w))`$, and the denominator is positive for
$`q < 1`$. So $`\tilde\beta`$ is a first harmonic divided by a positive trigonometric polynomial of degree 2.

**(F) Fourier series** (*checked symbolically*, the partial sum plus its exact remainder for N = 6; *checked
numerically*, 60 terms; mutation `fourier_phase`):

```math
\tilde\beta = A_*\sum_{k\ge0}(-q)^k\cos\big((2k+1)\theta + k\delta\big),\qquad
C_{2k+1} = \tfrac{A_*}{2}(-w)^k .
```

It has odd harmonics only, and every successive pair has the same complex ratio, $`C_{\nu+2}/C_\nu = -w`$. After N
terms the remainder is $`\mathrm{Re}[A_*z(-wz^2)^{N+1}/(1+wz^2)]`$, at most $`A_*q^{N+1}/(1-q)`$ on the real axis. The
series converges for $`q < 1`$ and, off the axis, in the strip $`|\mathrm{Im}\,\theta| < \tfrac12\ln(1/q)`$. At
$`q = 1`$ the poles reach the real axis. For $`q > 1`$ the closed form is $`1/q`$ times a profile with ratio $`1/q`$ and
the rotated amplitude $`e^{i\delta}`$ (*proved* by hand), so $`0 \le q < 1`$ covers the family.

**(H) Reciprocal oscillator** (*checked symbolically*; mutation `recip`):

```math
\tilde\beta = A_*\,\mathrm{Re}\,g,\qquad g = \frac{1}{h},\qquad h = e^{-i\theta} + w\,e^{i\theta},\qquad h'' + h = 0 .
```

The curve $`h(\theta)`$ is an ellipse with semi-axes $`1 \pm q`$, its major axis along $`e^{i\delta/2}`$, and $`g`$ is
its inversion. Equivalently $`g`$ solves the autonomous equation $`g g'' = 2g'^2 + g^2`$ (*checked symbolically*).

**(ODE) A linear equation for the real profile** (*checked symbolically*):

```math
\Big(\frac{d^2}{d\theta^2} + 1\Big)\big[D_q(\theta)\,\tilde\beta\big] = 0 .
```

Its solutions are $`(a\cos\theta + b\sin\theta)/D_q`$, which is $`\mathrm{Re}[c\,z/(1+wz^2)]`$ for a complex
$`c`$. All of them have the ratio $`-w`$, so for given $`w`$ the family has two real parameters.

**(E) The shifted, or ellipse, form** (*checked symbolically*; mutation `ellipse`). With
$`\varphi = \theta + \delta/2`$,

```math
\tilde\beta = A_*\,\frac{(1+q)\cos\varphi\cos\tfrac\delta2 + (1-q)\sin\varphi\sin\tfrac\delta2}
{(1+q)^2\cos^2\varphi + (1-q)^2\sin^2\varphi}
= A_*\,\mathrm{Re}\big[e^{-i\delta/2}\,G_q(\varphi)\big],\qquad G_q = \frac{1}{e^{-i\varphi} + q\,e^{i\varphi}} .
```

The shift removes $`\delta`$ from the denominator, but not from the profile. $`\mathrm{Im}\,G_q`$ is the conjugate
function (Hilbert transform) of $`\mathrm{Re}\,G_q`$. So the profile with phase $`\delta`$ is the $`\delta = 0`$ profile
shifted by $`\delta/2`$, with every positive-frequency harmonic rotated by $`-\delta/2`$. A shift cannot do that
rotation, since it rotates harmonic $`\nu`$ by an angle proportional to $`\nu`$.

**(M) An AM–FM carrier** (*checked numerically*):
$`\tilde\beta = A_*\cos(\theta - \chi)/\sqrt{D_q}`$ with $`\chi = \arg(1 + we^{2i\theta})`$, $`|\chi| \le \arcsin q`$.
The fundamental is modulated in amplitude, by a depth of about $`q`$, and in phase, by at most $`\arcsin q = 0.0551`$,
both at twice its frequency.

**(P) Poisson kernel** (*checked symbolically*): $`1/D_q = P_q(2\theta+\delta+\pi)/(1-q^2)`$ with
$`P_\rho(\phi) = (1-\rho^2)/(1 - 2\rho\cos\phi + \rho^2)`$, so
$`\tilde\beta = A_*\,[\cos\theta + q\cos(\theta+\delta)]\,P_q(2\theta+\delta+\pi)/(1-q^2)`$.

**(PF) Partial fractions in z** (*checked symbolically*):
$`z/(1+wz^2) = [1/(1-az) - 1/(1+az)]/(2a)`$ with $`a = i\sqrt w`$. This is the odd part of one geometric series in
$`az`$.

**(SEC) A secant, and its pole series** (*checked numerically*, raw `I221-mac/identity_check.py`, #221). With
$`\sigma = \tfrac12\ln(1/q)`$ and $`\varphi = \theta + \delta/2`$, $`\tilde\beta = (A_*/2\sqrt q)\,\mathrm{Re}[e^{-i\delta/2}\sec(\varphi + i\sigma)]`$.
Its Mittag-Leffler series is an alternating sum of Lorentzians of half-width σ centred at $`(k+\tfrac12)\pi - \delta/2`$,
dispersive at δ = 0 and turned toward absorptive by δ/2. Pollnau and Eichhorn sum same-sign absorptive Lorentzians of
the same width to the Airy distribution (Prog. Quantum Electron. 72, 100255 (2020), Eqs. (99)–(108)); the alternation
here comes from the odd harmonics.

**(INT) The antiderivative**, which by their Eq. (12) is the leading-order central scalar field up to the factor
$`\Delta_0/2\pi`$ (*checked symbolically*; at $`\delta = 0`$ *numerically*, mutation `atanh`):

```math
\int\tilde\beta\,d\theta = A_*\,\mathrm{Re}\Big[-\frac{i}{\sqrt w}\arctan\big(\sqrt w\,e^{i\theta}\big)\Big],\qquad
\delta = 0:\ \ \frac{A_*}{2\sqrt q}\,\mathrm{artanh}\Big(\frac{2\sqrt q}{1+q}\sin\theta\Big).
```

At $`\delta = 0`$ this is the functional form of Rozali and Way's oscillaton at a fixed radius, their Eq. (10),
$`\varphi_0 = \tanh^{-1}(\epsilon e^{-\omega^2r^2/2}\sin(\omega\tau - \tau_0))`$ (section 5), with
$`\epsilon e^{-\omega^2r^2/2} = 2\sqrt q/(1+q)`$. This is an identity between elementary functions. The oscillaton is a
single-mode solution of a different problem, and nothing here relates the two physically.

**Singularities** (*checked numerically*, residues at 15 digits; mutation `pole_dist`). Continued off the real axis,
$`\tilde\beta = \tfrac{A_*}2[g(\theta) + 1/(e^{i\theta} + \bar w e^{-i\theta})]`$. It has simple poles only, at

```math
\theta_p = \frac{\pi-\delta}{2} + k\pi \mp \frac{i}{2}\ln\frac1q ,\qquad
\mathrm{Res} = \pm\frac{iA_*}{4}\,e^{\pm i\theta_p}\quad(\text{lower/upper}),\qquad |\mathrm{Res}| = \frac{A_*}{4\sqrt q} .
```

That is one pole pair per half period, placed above and below the profile's zeros and shifted from them by
$`-\delta/2`$. The distance from the real axis, $`\sigma = \tfrac12\ln(1/q) = 1.4493`$, equals the decay rate per
harmonic, $`\tfrac12\ln|C_\nu/C_{\nu+2}|`$, exactly and for every $`\nu`$. Conversely, a ratio that is constant from the
first harmonic on gives exactly this profile.

**Zeros, extrema, norm** (*checked numerically*; mutation `norm`). The zeros are at
$`\theta_0 = \pi/2 - \arg(1+w) + k\pi`$ (1.565659 at the paper's values), two per period. While $`q < q_c(\delta)`$
there is one maximum and one minimum per period. At $`\delta = 0`$, $`q_c = 3 - 2\sqrt2 = 0.1716`$, above which the
maximum splits into two humps. The paper's $`q`$ is a third of that. At $`\delta = 0`$ the profile is $`\cos\theta`$
times the Airy distribution of a Fabry–Pérot cavity, $`A_*(1+q)/(1-q)^2\cdot\cos\theta/(1 + F\cos^2\theta)`$, with
coefficient of finesse $`F = 4q/(1-q)^2`$. So $`q_c`$ is $`F = 1`$, the reflectivity below which the Airy peaks have no
full width at half maximum (Pollnau and Eichhorn, Prog. Quantum Electron. 72, 100255 (2020), Eq. (118); checked by
raw `I198-mac/identity_check.py`, #198). The coincidence holds at δ = 0 only: at δ = 0.0985, $`q_c = 0.21328`$, while
the Airy factor's half-maximum breakdown stays at $`3 - 2\sqrt2`$ for every δ (#221). The norm is
$`\langle\tilde\beta^2\rangle = A_*^2/(2(1-q^2))`$ for every $`\delta`$ (their (S60)).

**The parameters.**

- $`(q, \delta) \to (-q, \delta+\pi)`$ leaves $`\tilde\beta`$ unchanged (*checked symbolically*), so the model is one
  amplitude $`A_*`$ and one complex ratio $`w`$ in the unit disk.
- $`\tilde\beta(-\theta; q, \delta) = \tilde\beta(\theta; q, -\delta)`$ (*checked symbolically*): time reversal flips
  $`\delta`$.
- No shift of $`\theta`$ removes $`\delta`$. In the form $`\mathrm{Re}[c\,z/(1+wz^2)]`$ a shift by $`s`$ maps
  $`(c, w) \to (ce^{is}, we^{2is})`$, and the invariant is $`\arg w - 2\arg c`$. It equals $`\delta`$ in the paper's
  gauge ($`c`$ real), and in terms of any harmonics it is $`\delta = \arg(C_3\bar C_1^3) - \pi`$. The best shift of the
  $`\delta = 0`$ profile onto the $`\delta = 0.0985`$ one leaves an rms residual of 0.0146, 0.54% of the profile's rms,
  2.72 (*checked numerically*).
- Meaning: $`A_*`$ is the amplitude of the fundamental. $`q = |C_{\nu+2}/C_\nu|`$, with $`\tfrac12\ln(1/q)`$ the width
  of the strip of analyticity. $`\delta`$ is the time asymmetry of the echo, zero exactly when the profile is even about
  its maximum.

**The earlier NLO condition.** Eq. (20) of arXiv:2601.14358, $`\Delta = |\beta''|/(3|\beta'|)`$ at $`\beta = 0`$ for
$`\beta`$ of unit period, gives 1.0000 on that paper's Eq. (21), the control. On Eq. (16) it gives $`\Delta = 0.0911`$
(*checked numerically*, and against the closed form $`(2\pi/3)\,|2D_q'/D_q|`$ at the zero, which holds because
$`N'' = -N`$ vanishes there), not $`\Delta_0 = 1.6439`$. The new paper does not use that condition (its Eq. (13) has
$`f = 1 - \epsilon(1-x^2)\tilde\beta^2/2 \le 1`$ at this order). The repository had already found that the condition
selects nothing (physics paper Sec. eeg; DINF_ANALYTIC.md §3).

## 3. The comparison with this repository

### 3.1 The map

The repository's E1–E4 are EEG's (3a)–(3d) under $`y = p - m = \sqrt2\,x\,\Pi`$ and $`p + m = \sqrt2\,x^2\Psi`$, with
the same $`\tau`$ (solver/supercrit/eqmap.py, *checked symbolically*; SUPERCRITICAL.md §1). So
$`\partial_xp(0) = \Pi(\tau, 0)/\sqrt2`$ and, with $`F = \sqrt n\,\partial_xp(0)`$,
$`\tilde\beta = \sqrt2\,\sqrt{(n+1)/n}\,F`$, which is $`\sqrt2\,F`$ at $`D = \infty`$. Our gauge has
$`F = A\sin\theta + \dots`$ ($`F_1 = -iA/2`$), and $`\theta_{\rm EEG} = \theta - \pi/2`$ turns it into theirs. Then
$`C_\nu = i^\nu F_\nu`$ and

```math
w = q\,e^{i\delta} = -\frac{C_{\nu+2}}{C_\nu} = \frac{F_{\nu+2}}{F_\nu} \equiv r_\nu ,\qquad A_* = \sqrt2\,A .
```

The README's $`F \approx 2\,\mathrm{Re}[F_1e^{i\theta}/(1 - qA^2e^{2i\theta})]`$ becomes Eq. (16) under this map, with
$`w = qA^2`$.

### 3.2 What the repository already says

- README, "Leading structure and a prediction" (*conjectured*). The tuned ratios are $`f_{\nu+2}/f_\nu \approx qA^2`$,
  with $`q = 0.00850 + 0.00114i`$, $`0.00851 + 0.00085i`$, $`0.00852 + 0.00082i`$ for $`\nu = 1, 3, 5`$ at leading
  order in $`A^2`$. That gives the form above, with singularities at $`\tfrac12\ln(1/|q|A^2) \approx 1.4`$, so F∞ is
  not entire.
- Math paper, Gap (b) and "The shape of the echo": $`q_\infty = 0.008524 + 0.000808i`$, $`|q|A^2 = 0.0633`$,
  "independent of $`\nu`$ to six digits" at leading order. Summed in $`A`$, the decay is 0.0574 per pair, and $`F`$ is
  analytic on $`|\mathrm{Im}\,\theta| < \sigma_F = 1.428`$. Physics paper Sec. V: 0.057 per pair, 1.43 per harmonic,
  $`|\mathrm{Im}\,\theta| \approx 1.4`$.
- DINF_ANALYTIC.md §5 [EEG26b] already identified Eq. (16) as "the geometric form of the math paper's tuned profile".

At leading order the moduli of those $`q_\nu`$ agree to 0.3%, but their phases do not: arg $`q_1 = 0.133`$ against
0.0996, 0.0960 and $`\to 0.0945`$. The first ratio is already special at leading order.

### 3.3 The tuned $`D = \infty`$ profile against Eq. (16)

Input: the tuned harmonics $`f_{k,\nu}`$ of the amplitude expansion at $`\omega_\infty`$, K = 45
(`solver/inner_amp/scan45J1250/`, `scan45J1500/`), summed at $`A`$ (`tuned_profile.py`). The two J agree to
$`10^{-11}`$ in every ratio. The A series has 22 terms for $`\nu = 3`$ and 14 for $`\nu = 19`$, whose last term is
$`2.5\times10^{-6}`$ of the sum. So $`\nu \le 19`$ is resolved, and $`\nu \ge 21`$ is truncation-limited (last term
$`2\times10^{-5}`$ to $`7\times10^{-4}`$).

The ratios at A = 2.72 (`data/tuned_profile_A2.72.txt`). The last column is the deviation of $`|C_\nu|`$ from Eq. (16)
with the paper's printed $`(A_*, q, \delta)`$. Its 3.0×10⁻⁴ at $`\nu = 1`$ is $`\sqrt2\cdot2.72/3.8455 - 1`$.

| $`\nu`$ | $`\lvert r_\nu\rvert`$ | arg $`r_\nu`$ | rate $`\tfrac12\ln(1/\lvert r_\nu\rvert)`$ | $`\lvert F_\nu/F_1\rvert`$ | $`\lvert C_\nu\rvert`$ against Eq. (16) |
|---|---|---|---|---|---|
| 1 | 0.055085 | 0.09879 | 1.44944 | 1 | +0.03% |
| 3 | 0.057203 | 0.07881 | 1.43057 | 0.055085 | +0.002% |
| 5 | 0.057392 | 0.07483 | 1.42893 | 3.1510×10⁻³ | +3.8% |
| 7 | 0.057431 | 0.07371 | 1.42859 | 1.8084×10⁻⁴ | +8.1% |
| 9 | 0.057443 | 0.07328 | 1.42848 | 1.0386×10⁻⁵ | +12.7% |
| 13 | 0.057451 | 0.07298 | 1.42841 | 3.4274×10⁻⁸ | +22.5% |
| 19 | 0.057453 | 0.07287 | 1.42839 | 6.4994×10⁻¹² | +38.9% |
| limit | 0.05745 | 0.0726 | 1.4284 | | |

The limit is a fit $`|r_\nu| = \rho(1 + b/\nu + c/\nu^2)`$ on $`\nu`$ = 9–17, 11–19 and 13–21, which agree to the
digits shown. Its $`b = 0.002`$–0.003 means a simple pole, since a pole of order $`m`$ gives $`b = 2(m-1)`$. The ratio
approaches its limit like $`\nu^{-3.0}`$ ($`|r_\nu - r_{19}|`$ for $`\nu`$ = 5–13). At A = 2.71918, the paper's
$`A_*/\sqrt2`$, every ratio moves by about $`3\times10^{-5}`$ (`data/tuned_profile_A2.71918.txt`).

Fits of the model to the tuned profile:

| fit | $`A_*`$ | $`q`$ | $`\delta`$ |
|---|---|---|---|
| arXiv:2611.07007, extrapolated finite-D profile | 3.8455 | 0.0551 | 0.0985 |
| the paper's procedure (S61)–(S62) on our profile, A = 2.72 | 3.84666 | 0.055097 | 0.098660 |
| the same at A = 2.71918 | 3.84550 | 0.055069 | 0.098679 |
| free fit, harmonics 1–7, relative residuals | 3.79969 | 0.056619 | 0.086547 |
| free fit, harmonics 1–15, relative residuals | 3.74736 | 0.057205 | 0.079772 |

The paper's procedure weights absolute coefficients, so it is set by $`C_1`$ and $`C_3`$ and returns the first ratio.
Ours and theirs agree to $`3\times10^{-5}`$ in $`q`$ and $`2\times10^{-4}`$ in $`\delta`$. A fit that gives the higher
harmonics weight moves toward the late ratio, and that dependence on the window is the sign of a non-geometric profile.

**Singularities** (paper's gauge). Eq. (16) with the paper's values has them at $`1.5215 \mp 1.4493i`$. The tuned
profile's nearest ones, from $`r_{19}`$, are at $`1.5344 \mp 1.4284i`$. They are $`0.021`$ closer to the axis and
shifted by $`0.013`$ along it.

**One parameter more.** The Padé [1/1] form in $`e^{2i\theta}`$,
$`\tilde\beta = A_*\,\mathrm{Re}[e^{i\theta}(1 + be^{2i\theta})/(1 + we^{2i\theta})]`$, keeps one pole pair and adds a
zero. Made exact on $`\nu = 1, 3, 5`$ ($`w = r_3 = 0.057203\,e^{0.0788i}`$, $`b = r_3 - r_1`$,
$`|b| = 2.4\times10^{-3}`$), it misses $`|F_7|`$ by 0.33%, $`|F_{13}|`$ by 1.6% and $`|F_{19}|`$ by 2.9%, against 8.1%,
22% and 39% for Eq. (16). The tuned profile is thus a geometric tail with ratio $`r_\infty`$, plus a first ratio about
4% smaller in modulus and 0.026 larger in phase.

*Status:* **numerical evidence.** The tuned profile assumes the selection rule (Conjecture 1 of the math paper) at A =
2.72 and $`\omega_\infty`$.

### 3.4 The finite-D profiles

`finiteD_ratios.py` takes the states in hp/data/states (D = 12–60, M = 32–48 harmonics in the half period), gauge-fixes
them to $`F_1 = -i|F_1|`$, and applies the same fit (`data/finiteD_ratios.txt`):

| D | $`\lvert r_1\rvert`$, arg | $`\lvert r_3\rvert`$, arg | $`\lvert r_5\rvert`$, arg | paper's fit $`q`$, $`\delta`$ |
|---|---|---|---|---|
| 20.04 | 0.1304, −0.0482 | 0.1862, +0.0093 | 0.2086, +0.0236 | 0.1324, −0.0454 |
| 30.07 | 0.1054, −0.0185 | 0.1473, +0.0118 | 0.1654, +0.0254 | 0.1063, −0.0176 |
| 50.01 | 0.0853, +0.0110 | 0.1143, +0.0137 | 0.1284, +0.0254 | 0.0858, +0.0111 |
| 60.00 | 0.0803, +0.0204 | 0.1055, +0.0154 | 0.1184, +0.0258 | 0.0806, +0.0204 |
| ∞, degree 1, 2, 3 in 1/n | 0.0579, 0.0552, 0.0549; +0.049, +0.066, +0.081 | 0.0701, 0.0615, 0.0589 | 0.0790, 0.0685, 0.0646 | 0.0575, 0.0552, 0.0550; +0.047, +0.066, +0.080 |
| ∞, tuned profile | 0.0551, +0.0988 | 0.0572, +0.0788 | 0.0574, +0.0748 | 0.0551, +0.0987 |

The ∞ rows are a spread over fit degree, not error bars. At finite D the profile is far from geometric. At D = 60 the
ratios grow by 56% from $`r_1`$ to $`r_7`$, against 4% at $`D = \infty`$, and the phase $`\delta(D)`$ changes sign
near D = 41. The moduli of $`r_1`$ and the paper's fit extrapolate to 0.0549–0.0552 for degrees 2–3, on the tuned
0.0551 and the paper's 0.0551. The higher ratios are still falling with the degree (README: "not fully converged").

**A check that fails, reported.** The criterion, set before the first run, is that the degree-2 and degree-3
extrapolations of arg $`r_1`$ lie within 0.01 of the tuned 0.0988. They give 0.066 and 0.081, so the script exits 1. The
orientation half of the check passes: both are positive, and its control, $`\theta \to -\theta`$ (`--mutate=reverse`),
makes them negative. The phase changes by 0.13 over D = 12–60 and is still moving at D = 60, so D ≤ 60 does not fix it
to 0.01. EEG's value from D ≤ 200, 0.0985, lies on the tuned value. The likely cause is that the extrapolation has not
converged. That is not shown, and the threshold has not been loosened.

### 3.5 The amplitude

$`A_*/\sqrt2 = 2.71918`$ against the selected zero of the amplitude expansion, $`A_0 = 2.720`$ at twelve orders, and the
M = 7 zero, $`A = 2.71922`$ (math paper, "Seven harmonics"; proof/B_tail_M7.md). The paper's $`A_*`$ comes from
$`\mathcal N_0`$, whose printed inputs ($`\Delta_1 = 10.37`$ to four digits) fix $`A`$ only to about $`5\times10^{-4}`$.
The agreement within $`4\times10^{-5}`$ is therefore partly luck. At $`A = 2.71918`$ the tuned profile has
$`\langle\tilde\beta^2\rangle = 2\langle F^2\rangle = 7.4164`$, against the paper's $`\mathcal N_0 = 7.4158`$ and the
model's $`A_*^2/(2(1-q^2)) = 7.4165`$.

#### 3.5.1 The states' D = ∞ amplitude against the selection's (agent-comms#54)

The finite-D states extrapolate to A∞ = 2.71920535. The M = 7 zero, 2.719219, lies 1.4×10⁻⁵ above it. The zero's limit
in M lies 1.2–6.5×10⁻⁵ below it, depending on how the M sequence of A continues. In units of the pre-registered spread
over fit choices, s = 4.7×10⁻⁵, the gaps to the grid's median are +0.28 s for the zero and −0.74 s for its geometric
limit. So the two agree. The comparison is limited by the selection's amplitude, not by the states. The M = 7 zero
carries ±2×10⁻⁵ of noise, and its truncation tail in A is larger still, while the states fix A∞ to about 10⁻⁷. *Status:*
the states' A∞ is **numerical evidence**. A₇, its limit and the gaps are **conditional** on the selection rule
(Conjecture 1).

**Method** (`solver/inner_pert/ainf_from_states.py`). A_n = 2|c₁(n)| is the fundamental amplitude of
F_n = β̃/√(2(1 + 1/n)) at each of the 63 states of D 15–215. The map is the one of §3.1, in the gauge F₁ = −iA/2
(`nnlo_closed_form.profile`). A∞ is the intercept of a least-squares fit of A_n in powers 1 to kmax of 1/n, over the
states with D ≥ dlo. This is `fit_K` with a free intercept, #42's "F fitted too". The grid has kmax 5–9, dlo 15, 20 and
30, and the test states 40.02, 60.00, 100 and 200 held out or included, which makes 30 choices. The criteria were posted
on the issue before any run. The pre-registered central value is the median over the grid, 2.7192060. The spread
s = max |A∞ − median| = 4.7×10⁻⁵ is a spread over fit choices, not an error bar. The kmax 6, D ≥ 20 choice with the test
states held out reproduces #42's 2.7192088. #42's nine choices reproduce its range 2.7192055–2.7192530, to 4×10⁻⁸ (R0).

| kmax | D ≥ 15 | D ≥ 20 | D ≥ 30 |
|---|---|---|---|
| 5 | 2.719251064 | 2.719221520 | 2.719209496 |
| 6 | 2.719215490 | 2.719208812 | 2.719206017 |
| 7 | 2.719207794 | 2.719206136 | 2.719205460 |
| 8 | 2.719205970 | 2.719205511 | 2.719205356 |
| 9 | 2.719205494 | 2.719205375 | 2.719205338 |

The table uses all states. Holding out the test states changes an entry by at most 1.9×10⁻⁶, at kmax 5 and D ≥ 15. The
fits converge with the degree and the low end of D. The spread s is set by the low degrees at low D, which carry the
truncation of the 1/n series. Chosen after the run, from the planted calibration below, the converged choices are kmax
9–12 at D ≥ 20 and 30 and kmax 8 at D ≥ 30 and 40, with all states. They give 2.719205332–2.719205375, which is the A∞
quoted above. Their half-range, 2×10⁻⁸, is a spread over fit choices.

**Planted truth, run first.** Each state's central Π is replaced by the tuned profile scaled to a known A_pl(n), at a
known time shift, and passed through the same code. Every family has A∞ = 2.7180, A₁ = 14.52 and A₂ = −33 (#42's fit),
with a non-terminating tail. In `alt12` the tail is A₂t²/(1 + 12t), with t = 1/n. In `cpx7` it is
A₂t²/(1 − 7t + 49t²). These two are sized so that their low-degree offsets match the real grid's. `alt20` is harsher
than the data. The fit's errors:

| family | median | kmax 5, D ≥ 15 | kmax 6, D ≥ 20 | kmax 9, D ≥ 30 | converged choices | criterion |
|---|---|---|---|---|---|---|
| alt12 | +5.9×10⁻⁷ | +4.5×10⁻⁵ | +3.2×10⁻⁶ | +2.5×10⁻⁹ | ≤ 2.7×10⁻⁸ | P1, median ≤ 2×10⁻⁶: pass |
| cpx7 | −5.8×10⁻¹⁰ | +6.8×10⁻⁵ | −3.6×10⁻⁶ | −5.8×10⁻¹⁰ | ≤ 1.6×10⁻⁸ | P1: pass |
| alt20 | +3.8×10⁻⁶ | +1.2×10⁻⁴ | +1.4×10⁻⁵ | +4.6×10⁻⁸ | ≤ 3.6×10⁻⁷ | information (misses 2×10⁻⁶) |

`alt12` reproduces the real grid's pattern. Measured from the kmax 9, D ≥ 30 entry, the real offsets are 4.8×10⁻⁵ and
3.5×10⁻⁶ at the first two choices, against 4.5×10⁻⁵ and 3.2×10⁻⁶ planted. On `alt12` the median is 5.9×10⁻⁷ high,
which is why the converged choices are quoted. With Gaussian noise of the real fit's residual added (8.9×10⁻¹⁰, 20
seeds), the median's error is 5.9×10⁻⁷ for `alt12` and 2.0×10⁻⁸ for `cpx7` (P3, pass). The control `--mutate=offset`
compares the planted runs against the truth plus 5×10⁻⁶, and P1 then fails, as it must.

**The data.** The reference fit (kmax 8, D ≥ 20, all states) leaves an rms residual of 8.9×10⁻¹⁰ in A_n. The mean
residual of each group of states (D ≤ 60, 73, 80–170, 180–215) is below 2.3×10⁻¹⁰. The held-harmonic chains above D 73
are not offset from the trend below, since a step fitted at D 76.5 comes out at −4.5×10⁻⁹. With the T140 chains d6 and
d10 in place of the main chain at D 100, 150 and 180, A_n moves by at most 7.7×10⁻¹⁰ and A∞ by 2.5×10⁻⁹. One factor at a
time around the reference, A∞ moves as follows.

| change | A∞ moves by |
|---|---|
| variable 1/(D − 1) in place of 1/n | −8.7×10⁻⁸ |
| prefactor √2 in place of √(2(1 + 1/n)) | −6.5×10⁻⁹ |
| D ≥ 40 | −1.7×10⁻⁷ |
| kmax 10, 11, 12 | −1.7×10⁻⁷, −1.8×10⁻⁷, −1.8×10⁻⁷ |
| kmax 4 | +8.6×10⁻⁵ |

**Controls.** All of them behave (`controls`, exit 0).

| control | criterion | measured |
|---|---|---|
| C, every A_n + 1×10⁻⁵ in the field Π(τ, 0) | every choice moves by 1×10⁻⁵ within 20% | 1.000×10⁻⁵, relative deviation 5.9×10⁻⁹ |
| slope, A_n + 1×10⁻³/n | every choice moves by ≤ 1×10⁻⁸ | 6.4×10⁻¹⁴ |
| step, A_n + 1×10⁻⁵ for D ≥ 76.5 | the step fit recovers it within 20% | relative deviation 1×10⁻¹⁰ |
| each criterion on another mutation (C on step, slope on shift, step on slope) | must fail | all three fail |

Under the step mutation the plain grid moves by −2.2×10⁻⁵ to +1.4×10⁻⁴. A chain offset of 10⁻⁵ would therefore show,
and none is there.

**The selection's A** (conditional). The affine fit of the ten M = 7 logs (`m7_outer.py`'s fit) gives
A₇ = 2.719219254 and ω/ω∞ = 1.000021738. This reproduces proof/B_tail_M7.md's 2.719219 and 1.000022 (R1). Its noise is
±2×10⁻⁵, a noise estimate from the scatter of repeated solves, not an enclosure. The notes give a truncation error for ω
only, the M > 7 tail of about 4×10⁻⁵ (B_tail_M7.md §3). For A it is computed here from the same sequence and estimator
(`trunc_seq_M7.py`). The A sequence is 2.542148, 2.726476, 2.718579 and 2.719219 for M = 1, 3, 5 and 7. Its ratios per
pair are −0.043 and −0.081. They have not settled, unlike ω's −0.069 and −0.048. The same code reproduces ω's limit,
0.999985 (R2). The limit of A depends on the ratios past M = 7.

| ratios past M = 7 | limit of A | tail past M = 7 | limit minus the states' A∞ |
|---|---|---|---|
| held at −0.081 (the geometric estimate L_A) | 2.719171 | −4.8×10⁻⁵ | −3.4×10⁻⁵ |
| back to −0.043 | 2.719193 | −2.6×10⁻⁵ | −1.2×10⁻⁵ |
| ω's −0.048 | 2.719190 | −3.0×10⁻⁵ | −1.6×10⁻⁵ |
| the drift continued, \|ratio\| ≤ 1/4 | 2.719141 | −7.9×10⁻⁵ | −6.5×10⁻⁵ |
| −1/4 throughout | 2.719091 | −1.3×10⁻⁴ | −1.1×10⁻⁴ |

The estimator's error formula, |A₇ − L_A|(|e| + 0.05)/(1 − |κ|) = 4.9×10⁻⁵, uses the measured drift e = 0.89 of the
ratio. That is far outside the ±0.1 at which `trunc_seq_M7.py --planted` tested it. It covers the first four scenarios,
not the last.

**The gap** (conditional). Each gap is the selection's value minus the grid's median, 2.7192060, with s = 4.7×10⁻⁵. The
combined scale is √(s² + noise² + tail²). The tail is |L_A − A₇| for A₇ and the error formula for L_A.

| selection's A | gap | in units of s | in units of the combined scale |
|---|---|---|---|
| A₇ = 2.719219 | +1.3×10⁻⁵ | +0.28 | +0.19 |
| L_A = 2.719171 | −3.5×10⁻⁵ | −0.74 | −0.49 |

The two agree under the criterion posted before the runs, which asks for |gap| ≤ 2 × the combined scale. Against the
converged A∞ the gaps are +1.4×10⁻⁵ for A₇ and those of the scenario table. The states' value lies between A₅ and A₇,
where an alternating sequence's limit must lie. It is the limit if the next ratio of the A sequence is −0.021. With A₇
moved by its noise, that ratio lies between −0.053 and +0.011. This range includes ω's −0.048 but not A's last ratio,
−0.081. The states' A∞ is 2.5×10⁻⁵ above the tuned profile's 2.71918, which is EEG's A*/√2. Times √2 the grid's median
is 3.845538, which rounds to EEG's printed 3.8455.

**What it shares and what it does not show.** The states' extrapolation and the M = 7 selection share no code and no
method. One fits the finite-D solutions of the field equations, and the other solves the inner problem's matching
conditions with seven harmonics. They share the definition of A, F = √n ∂ₓp(0) with F₁ = −iA/2 (§3.1). The
extrapolation shares `profile` and `fit_K` with #42, whose number R0 reproduces. To test Conjecture 1's amplitude at
the 10⁻⁵ level, the selection's A must be known to 10⁻⁵. M = 9 would shrink the tail by about one more ratio per pair,
and the noise would have to fall below the present ±2×10⁻⁵.

**Discrepancies (reported, not fixed).**

- SECOND_ORDER.md §1.2, "Cross-checks", says the states extrapolate "to an amplitude about 1×10⁻⁵ above the paper's".
  For #42's 2.719209 the difference from 2.71918 is 2.9×10⁻⁵ absolute, which is 1.1×10⁻⁵ relative. For the converged
  A∞ it is 2.5×10⁻⁵ absolute.
- #42's 2.719209 is its kmax 6, D ≥ 20 choice. The higher degrees converge 3.5×10⁻⁶ lower. #42's range
  2.7192055–2.7192530 measures the low degrees' truncation, not an uncertainty of the extrapolation.
- §3.5 above and the math paper's "Seven harmonics" quote A = 2.71922 without a truncation error in A. Its noise,
  ±2×10⁻⁵, is smaller than the A tail past M = 7, which is 2.6–7.9×10⁻⁵ under the first four scenarios.

**Proposed text** (for the coordinator, SECOND_ORDER.md §1.2's first cross-check):

> With the intercept fitted too, the states extrapolate to A = 2.71920535. The fits of degree 8–12 in 1/n that have
> converged span 2.719205332–2.719205375, and the 30 fit choices of EEG_PROFILE.md 3.5.1 span 2.7192053–2.7192530, the
> spread of the low degrees. That is 2.5×10⁻⁵ above the tuned profile's 2.71918, which is EEG's A*/√2. The M = 7 zero,
> 2.719219 ± 2×10⁻⁵, lies 1.4×10⁻⁵ above it, and its limit in M lies 1.2–6.5×10⁻⁵ below it, depending on the ratio of
> the A sequence past M = 7 (*conditional*).

**Reproduction.** From `solver/inner_pert/`, with the raw-data repository cloned beside the checkout, each command takes
a few seconds. `data/ainf_from_states.json` is the summary these numbers come from.

```
F=../../../choptuik-large-d-raw/I126-mac
python3 ainf_from_states.py planted $F/fields/fields_D*.npz --summary=data/ainf_from_states.json     # P1: exit 0
python3 ainf_from_states.py planted $F/fields/fields_D*.npz --family=alt12,cpx7 --mutate=offset \
    --summary=data/ainf_from_states.json                                    # Pm: P1 fails, exit 0
python3 ainf_from_states.py states $F/fields/fields_D*.npz --chains=$F/fields_t140_d6,$F/fields_t140_d10 \
    --summary=data/ainf_from_states.json                                    # R0, R1, R2: exit 0
python3 ainf_from_states.py controls $F/fields/fields_D*.npz --summary=data/ainf_from_states.json    # exit 0
python3 ainf_from_states.py planted $F/fields/fields_D*.npz --family=alt12,cpx7 --noise=8.85e-10 \
    --summary=data/ainf_from_states.json                                    # P3: exit 0
python3 ainf_from_states.py selection                                       # R1, R2: exit 0
```

Raw output: folder `AC54-mac/` of the raw-data repository, at its commit 7b3077a8 (logs, JSON, the commands as run in
`run_all.cmd`, exits in `exits.txt`).

## 4. What this means

- Eq. (16) is the one-pole-pair (geometric) truncation of the $`D = \infty`$ echo. Its $`w`$ is the first harmonic
  ratio, which two independent routes give to 3×10⁻⁵ in modulus and 2×10⁻⁴ in phase: the finite-D solutions to D = 200
  extrapolated (EEG), and the inner problem's selection rule (here). What the two share is the physical model. They
  share no code, no method, and no finite-D input on our side.
- The model's analyticity strip, 1.449, is set by the first ratio. The true strip, 1.428 in the math paper, is set by
  the late ratio, 0.0575 per pair. Hypothesis (H1)'s $`\sigma_F \approx 1.4`$ is unaffected.
- The derivation the paper asks for at least has a target. $`w`$ is $`r_1 = f_3(A)/F_1`$, fixed by $`S_3 = 0`$ and,
  through the A series, by the higher conditions. At leading order it is $`q_1A^2`$, with roughly
  $`1/q \approx 8\omega^2 - 4i\omega - 1.3`$ (README). That is a nonlinear Stokes condition, not a closed form. Its
  phase $`\delta`$ comes from the complex tuning ratio, so the Stokes conditions set the echo's time asymmetry.

## 5. Literature

- **Rozali, Way**, JHEP 11 (2018) 106, arXiv:1807.10283 (verified in LIT_REVIEW.md). Read now in full text (ar5iv HTML,
  Sec. 2.3): Eq. (10) is $`\varphi_0 = \tanh^{-1}(\epsilon e^{-\omega^2r^2/2}\sin(\omega\tau - \tau_0))`$, a single
  frequency with a sharp threshold at $`\epsilon = 1`$. Sec. 3.5 has the AdS analogue. There is no geometric or rational
  profile of the Eq. (16) type. The functional coincidence of §2 (INT) is noted there, and nothing more.
- **Ecker, Ecker, Grumiller**, PRL 136, 191401 (2026), arXiv:2601.14358 (verified in REF_CHECK.md). Read now (HTML v1,
  Secs. VII–VIII): Eq. (20) is the NLO condition, and Eq. (21) is the two-mode model
  $`\beta = \cos 2\pi\tau + \sin(6\pi\tau)/A`$, $`A \approx 15.9476`$. In the present notation that is a two-term
  truncation with $`|w| = 1/A = 0.0627`$ and $`\delta = \pi/2`$. No geometric series.
- **Ecker, Ecker, Grumiller**, arXiv:2611.07007v1 (verified, arXiv listing; LIT_REVIEW.md): the source of Eq. (16),
  which states the series, the closed form and the norm itself ((S58), (S60)). The poles, the reciprocal-oscillator and
  ellipse forms, the ODE, the antiderivative and the $`(q,\delta)`$ reductions are not in it.
- **Price, Pullin**, gr-qc/9601009, PRD 54, 3792 (1996) (as in LIT_REVIEW.md). This is a flat-space matching in
  $`D = 4`$. The full text was not reread for this note, so whether it contains a profile of this form is
  *(unverified)*.
- The Poisson kernel, the geometric series and the link between pole distance and Fourier decay are textbook facts.
  Nothing in §2 is claimed as new.

## 6. Reproduction

From the repository root, with numpy, scipy, mpmath and sympy. `finiteD_ratios.py` also needs hpkernel (hp/build.sh).

```
python3 solver/eeg_profile/forms_check.py                 # 29/29 checks pass, exit 0
python3 solver/eeg_profile/forms_check.py --mutate=sign_q # (and fourier_phase, pole_dist, norm, recip, ellipse, atanh) exit 1
python3 solver/eeg_profile/tuned_profile.py               # section 3.3 at A = 2.72, both J
python3 solver/eeg_profile/tuned_profile.py --A 2.71918   # at the paper's A_*/sqrt 2
python3 solver/eeg_profile/tuned_profile.py --plant       # planted geometric profile, recovered to 1e-16, exit 0
python3 solver/eeg_profile/tuned_profile.py --plant --mutate=drift   # exit 1 (and --mutate=sign_q)
python3 solver/eeg_profile/finiteD_ratios.py              # section 3.4; exits 1, the agreement check fails (3.4)
python3 solver/eeg_profile/mutation_controls.py           # every control above, "all controls behave", exit 0
```

## 7. Proposed text (for the coordinator)

For the physics paper Sec. V, after "a rate of 1.43 per harmonic":

> Ecker, Ecker and Grumiller model the extrapolated $`D \le 200`$ profile by $`A_*\mathrm{Re}[e^{i\theta}/(1 +
> qe^{i(2\theta+\delta)})]`$ with fitted $`q = 0.0551`$, $`\delta = 0.0985`$ [EEG26b], our leading structure with a
> single pole pair. Its ratio is the first tuned ratio, $`F_3/F_1 = 0.0551\,e^{0.0988i}`$, while the later ratios rise
> to $`0.0575\,e^{0.073i}`$, which sets the singularities at $`|\mathrm{Im}\,\theta| = 1.428`$ rather than the model's
> 1.449.

## 8. Rational profiles beyond Eq. (16) (issue #133)

**Verdict.** The family $`\tilde\beta = A_*\,\mathrm{Re}[e^{i\theta}P_L(e^{2i\theta})/Q_N(e^{2i\theta})]`$ reaches
10⁻⁶ at every $`\nu \le 19`$ first at $`L + N = 7`$, for [4/3], [3/4] and [2/5] fitted by least squares or minimax.
That is 8 complex parameters for 10 complex harmonics. Padé matching reaches it for no $`L + N \le 8`$ on the data as
stored: [4/4] is off by 1.15×10⁻⁶ at $`\nu = 19`$, where the data's own truncation is 5.4×10⁻⁷. The error falls by about
a factor 7 per added complex parameter, so the family converges, but it does not compress the profile at 10⁻⁶
(*numerical evidence*).

Every fit with $`L + N \ge 5`$ puts its nearest pole on the profile's singularity, 1.5344 ∓ 1.4284i in the paper's
gauge, and the fits that reach 10⁻⁶ agree on it to 2×10⁻⁵. The other poles are not a second singularity. They pair
with zeros, carry small residues and move toward the nearest pole as the order grows, as Padé poles do on the cut of
a branch point. The ratios also approach their limit as a power of $`\nu`$, which no single further pole gives. So the
nearest singularity is a simple pole with a weaker branch at the same point, likely logarithmic, rather than an
isolated pole (*numerical evidence*). The issue's reading of the $`\nu^{-3}`$ law, "a simple nearest pole pair", holds
for the leading term only.

Scripts: [`solver/eeg_profile/rational_fit.py`](../solver/eeg_profile/rational_fit.py) and
[`rational_controls.py`](../solver/eeg_profile/rational_controls.py). Raw output: folder `I133-mac/` of the raw-data
repository.

### 8.1 The family and the fits

In $`u = e^{2i\theta}`$ (paper's gauge) the model is $`G(u) = P_L(u)/Q_N(u)`$ with $`Q_N(0) = 1`$. Its Taylor
coefficients are the harmonic ratios $`c_k = C_{2k+1}/C_1`$, with $`C_\nu = i^\nu F_\nu`$ (§3.1). Eq. (16) is [0/1],
$`G = 1/(1 + wu)`$, and §3.3's one-zero form is [1/1]. A fit has $`L + N + 1`$ complex parameters, among them the
amplitude $`p_0 = A_*e^{i\varphi}/2C_1`$. The phase $`\varphi`$ is fitted. It is at the level of the fit's error,
1.9×10⁻² for Eq. (16) by least squares and at most 2.8×10⁻⁷ for the fits that reach 10⁻⁶.

The fits use $`k = 0, \dots, 9`$, that is $`\nu \le 19`$, at 60 digits:

- **Padé**, exact on $`c_0, \dots, c_{L+N}`$ [Padé 1892]. With $`L + N \le 8`$, at least $`\nu = 19`$ is out of
  sample.
- **Least squares** of the relative residuals $`m_k/c_k - 1`$ with every weight 1, so each harmonic counts alike.
  It runs Levenberg–Marquardt from the Padé fit [Levenberg 1944; Marquardt 1963].
- **Minimax**: the largest relative residual is minimized by Lawson's reweighted least squares [Lawson 1961], started
  from the least-squares fit. It keeps the best iterate, so its column is an upper bound on the true minimax error.

The error is $`e_k = |m_k/c_k - 1|`$, complex, so it bounds the modulus and the phase together. §3.3 quotes moduli
only. Its 2.9% for [1/1] at $`\nu = 19`$ is 2.83% here as a modulus error and 4.7% as a complex one.

### 8.2 The data's precision

The f coefficients are doubles, and the A series stops at K = 45. The remainder of each harmonic's series is
estimated as the geometric tail of its last two terms. That is an estimator, not a bound:

| $`\nu`$ | 11 | 13 | 15 | 17 | 19 | 21 | 23 | 25 |
|---|---|---|---|---|---|---|---|---|
| remainder estimate, relative | 1.1×10⁻¹⁰ | 1.0×10⁻⁹ | 9.0×10⁻⁹ | 7.2×10⁻⁸ | 5.4×10⁻⁷ | 3.8×10⁻⁶ | 2.6×10⁻⁵ | 1.6×10⁻⁴ |

So at $`\nu = 19`$ the 10⁻⁶ target is within a factor 2 of the data's resolution. `--tail` adds the estimate to the
data, and every verdict below is checked both ways.

Between the two J (1250, 1500), no maximum error changes by more than 4×10⁻⁶ of itself and no nearest pole by more
than 4×10⁻¹¹ (Padé and least squares). The harmonics reproduce `data/tuned_profile_A2.72.txt` to its printed digits
(check D: 2.9×10⁻⁷ in modulus, 4.7×10⁻⁶ in argument).

### 8.3 Planted truth and the controls

The criteria are in the script's docstring, committed (e973bb9) before the first run. The planted profiles enter as
f coefficients of the A series, rounded to doubles, and go through the same loader, gauge map and fits.

| run | criterion | result |
|---|---|---|
| planted [2/3]: poles 1.5344 − 1.4284i, 1.47 − 1.72i, 1.68 − 1.95i | P1: no $`L + N \le 4`$ reaches 10⁻⁶, and at 5 only (2, 3) does. P2: its error ≤ 10⁻¹². P3: poles and zeros recovered to 10⁻⁸ | pass. Error 2.3×10⁻¹⁵ (Padé), 5.6×10⁻¹⁷ (least squares). Poles recovered to 3.3×10⁻¹⁴, zeros to 6.1×10⁻¹⁴ |
| the same, `--mutate=gauge` ($`C_\nu = F_\nu`$) | P3 must fail | fails, poles off by 1.5. P1 and P2 pass, since a gauge only rotates $`u`$ |
| the same, `--mutate=drop` ($`\nu = 11`$ dropped) | P1 or P2 must fail | all fail. The [2/3] error is 3.0×10⁹ (Padé), 18 (least squares) |
| planted pole with a logarithm (`--plant=branch`) | marching | pass. $`\theta_2`$ moves 0.293 inward from [3/3] to [4/4] |
| planted meromorphic profile, six poles (`--plant=poles`) | not marching, and converged ($`\lvert\theta_2(4) - \theta_2(3)\rvert \lt 0.02`$, within 0.02 of the planted pole) | **fails, reported.** Not marching passes. Converged fails: the step is 0.025, though $`\theta_2(4)`$ lies 0.002 from the planted pole |
| real data, check S | nearest pole within 10⁻³ of 1.5344 − 1.4284i, every fit with $`L + N \ge 5`$ | pass, 78 of 78. `--mutate=gauge` fails it |
| real data, check D | the data file to its digits | pass. `--mutate=A` (A = 2.7203) fails it |

The pole-march test takes $`\theta_2(n)`$, the pole second nearest the real axis in the Padé [n/n] for n = 2, 3, 4.
It calls the sequence *marching* when $`\lvert\mathrm{Im}\,\theta_2\rvert`$ falls at each step and the last step
exceeds 0.05, and *converged* when the last step is below 0.02.

- **The converged half fails its meromorphic plant as stated.** The plant's further poles lie only 0.3 apart in
  $`\mathrm{Im}\,\theta`$, so four orders do not settle the second one. The threshold has not been loosened. As
  written, the test certifies marching but not convergence.
- **The marching half separates the two plants.** The branch plant marches. The meromorphic plant's second pole moves
  outward, toward its planted position.
- **Check D first passed for the wrong reason.** The data file begins with a blank line, so the inputs were paired
  one block off and the first input was compared with nothing. Its mutation passed, which exposed this, and ed08568
  fixed the pairing.

`rational_controls.py` runs every row above and prints "all controls behave". mutation_controls.py does not call it
yet. That is a proposal for the coordinator.

### 8.4 The maximum error against (L, N)

Least squares, $`\max_{\nu \le 19} e_\nu`$, rows L, columns N:

| L \ N | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| 0 | 3.50×10⁻² | 6.63×10⁻³ | 1.87×10⁻³ | 6.40×10⁻⁴ | 2.40×10⁻⁴ | 8.98×10⁻⁵ | 3.86×10⁻⁵ | 1.62×10⁻⁵ |
| 1 | 5.19×10⁻³ | 7.82×10⁻⁴ | 1.54×10⁻⁴ | 3.53×10⁻⁵ | 9.08×10⁻⁶ | 2.43×10⁻⁶ | 5.82×10⁻⁷ | |
| 2 | 1.45×10⁻³ | 1.19×10⁻⁴ | 1.60×10⁻⁵ | 2.75×10⁻⁶ | 5.08×10⁻⁷ | 1.29×10⁻⁷ | | |
| 3 | 5.13×10⁻⁴ | 2.37×10⁻⁵ | 2.39×10⁻⁶ | 2.80×10⁻⁷ | 6.62×10⁻⁸ | | | |
| 4 | 1.98×10⁻⁴ | 5.99×10⁻⁶ | 3.99×10⁻⁷ | 6.18×10⁻⁸ | | | | |
| 5 | 7.61×10⁻⁵ | 1.67×10⁻⁶ | 1.03×10⁻⁷ | | | | | |
| 6 | 2.85×10⁻⁵ | 4.37×10⁻⁷ | | | | | | |
| 7 | 1.27×10⁻⁵ | | | | | | | |

The best family at each total degree for the three fits (the full Padé and minimax tables are in the log):

| L + N | complex parameters | Padé | least squares | minimax |
|---|---|---|---|---|
| 1 | 2 | [0/1] 3.28×10⁻¹ | [0/1] 3.50×10⁻² | [0/1] 2.16×10⁻² |
| 2 | 3 | [1/1] 4.69×10⁻² | [1/1] 5.19×10⁻³ | [1/1] 3.37×10⁻³ |
| 3 | 4 | [1/2] 7.75×10⁻³ | [1/2] 7.82×10⁻⁴ | [1/2] 5.17×10⁻⁴ |
| 4 | 5 | [2/2] 1.66×10⁻³ | [2/2] 1.19×10⁻⁴ | [2/2] 9.26×10⁻⁵ |
| 5 | 6 | [2/3] 3.12×10⁻⁴ | [2/3] 1.60×10⁻⁵ | [2/3] 1.22×10⁻⁵ |
| 6 | 7 | [3/3] 5.27×10⁻⁵ | [3/3] 2.39×10⁻⁶ | [3/3] 1.78×10⁻⁶ |
| 7 | 8 | [3/4] 5.82×10⁻⁶ | [3/4] 2.80×10⁻⁷ | [3/4] 2.17×10⁻⁷ |
| 8 | 9 | [4/4] 1.15×10⁻⁶ | [4/4] 6.18×10⁻⁸ | [3/5] 4.97×10⁻⁸ |

- **The smallest family.** At $`L + N = 7`$, least squares and minimax reach 10⁻⁶ with [3/4] (2.80×10⁻⁷, 2.17×10⁻⁷),
  [4/3] (3.99×10⁻⁷, 2.96×10⁻⁷) and [2/5] (5.08×10⁻⁷, 3.78×10⁻⁷). [5/2] and [1/6] miss it. At $`L + N = 6`$ nothing
  reaches it, and the best is minimax [3/3] at 1.78×10⁻⁶.
- **The verdict does not depend on the data's tail.** With `--tail` the smallest family is again at $`L + N = 7`$,
  with the same three members ([3/4] by least squares 2.25×10⁻⁷), and $`L + N = 6`$ still misses (minimax [3/3]
  1.70×10⁻⁶). Padé then reaches 10⁻⁶ at $`L + N = 8`$ with [4/4] (3.13×10⁻⁷), [3/5], [5/3] and [2/6]. At A = 2.71918,
  the paper's $`A_*/\sqrt2`$, the verdicts are unchanged.
- **Out of sample.** The passing fits predict the stored $`\nu = 21`$ within 4.0×10⁻⁶ and $`\nu`$ = 23, 25 within
  3.2×10⁻⁵ and 1.5×10⁻⁴, the size of those harmonics' remainder estimates. With the tail added the residuals shrink:
  [4/4] by least squares is off by 4.0×10⁻⁷, 1.5×10⁻⁶ and 5.0×10⁻⁶ at $`\nu`$ = 21, 23, 25. That is consistent with
  the tail estimate. It is not an independent check, since the tail also enters the fitted $`\nu \le 19`$.
- **The rate.** Along the best family, each added complex parameter lowers the least-squares error by a factor of
  6.6–8.5 ($`L + N`$ = 1–7). With a single pole the gain is slower: [L/1] falls by a factor of 2.2–3.6 per added L,
  to 1.27×10⁻⁵ at [7/1].
- **Eq. (16) and its one-zero form.** Fitted by least squares, [0/1] is within 3.5% of every harmonic. As a Padé fit
  with $`w = r_1`$ its modulus error at $`\nu = 19`$ is 28% (33% with the phase). That is §3.3's 39% measured the
  other way, $`|m|/|c|`$ instead of $`|c|/|m|`$. [1/1] by least squares is within 5.2×10⁻³, nine times better than
  Padé's 4.7×10⁻².

### 8.5 The poles

Least-squares fits, in $`\theta`$ (paper's gauge). The residue is relative to the nearest pole's, and "zero at" is
the distance to the nearest zero.

| fit | max error | nearest pole | second pole (residue; zero at) | third pole (residue; zero at) |
|---|---|---|---|---|
| [2/2] | 1.19×10⁻⁴ | 1.53431 − 1.42842i | 1.59177 − 1.99185i (0.057; 0.035) | |
| [3/3] | 2.39×10⁻⁶ | 1.53437 − 1.42839i | 1.55443 − 1.72609i (0.014; 0.0045) | 1.61405 − 2.32282i (0.084; 0.083) |
| [4/3] | 3.99×10⁻⁷ | 1.53438 − 1.42839i | 1.54965 − 1.67595i (0.0097; 0.0025) | 1.58293 − 2.13143i (0.052; 0.042) |
| [3/4] | 2.80×10⁻⁷ | 1.53438 − 1.42839i | 1.55241 − 1.66439i (0.0087; 0.0021) | 1.59145 − 2.07882i (0.043; 0.032) |
| [4/4] | 6.18×10⁻⁸ | 1.53439 − 1.42839i | 1.56393 − 1.61076i (0.0052; 0.0010) | 1.59349 − 1.92134i (0.025; 0.014) |

Farther out, [3/4] has a fourth pole at 1.73326 − 3.01063i, and [2/5] has two at 0.67032 − 3.17169i and 2.63067 −
3.16372i, with residues of 0.11–0.17. They lie 1.6–1.7 farther from the axis, and they are not the same from one fit
to the next.

- **The nearest pole.** Every one of the 78 fits with $`L + N \ge 5`$ (all three methods, A = 2.72) puts it at
  1.53412–1.53439 − (1.42839–1.42848)i. The 18 that reach 10⁻⁶ put it at 1.53437–1.53439 − 1.42839i. That is the
  profile's 1.5344 ∓ 1.4284i (§3.3, from $`r_{19}`$), and its 1.4284 is §3.3's late decay rate, 0.05745 per pair. In
  the repository's gauge, $`\theta = \theta_{\rm EEG} + \pi/2`$, it is 3.1049–3.1052 ∓ 1.4284i. At A = 2.71918 its
  imaginary part moves to −1.42866 to −1.42875.
- **The further poles march.** The second pole of the Padé [n/n] moves from 1.60680 − 2.11295i (n = 2) to
  1.55974 − 1.77480i (n = 3) and 1.54256 − 1.64552i (n = 4). The last step is 0.130, so it is marching, like the
  branch plant (0.293) and unlike the meromorphic plant, whose pole moves outward and lands within 0.002 of its
  planted position. Least squares shows the same: the second pole's $`\lvert\mathrm{Im}\,\theta\rvert`$ is 1.99,
  1.73, 1.66–1.68 and 1.61 at $`L + N`$ = 4, 6, 7, 8.
- **The further poles pair with zeros.** Each has a zero 0.001–0.08 away, nearer for the poles nearer the singularity,
  and a residue of 0.5–8% of the nearest pole's. They line up along $`\mathrm{Re}\,\theta \approx 1.55`$–1.61,
  outward from the singularity. That is how Padé approximants represent a cut from a branch point [Stahl 1997]. For a
  function meromorphic in a larger disk their poles converge to its poles instead [de Montessus de Ballore 1902].
- **The ratios approach their limit as a power.** Let $`r_k = c_{k+1}/c_k`$, so $`r_k`$ is §3.3's $`r_\nu`$ at
  $`\nu = 2k + 1`$. With the tail added, $`(k + 1.5)^4\,\lvert r_{k+1} - r_k\rvert`$ is 0.01124 for $`\nu`$ = 11–17 to
  four digits, and 0.01154 at $`\nu = 3`$. As stored, the last two values are 0.01142 and 0.00926, an effect of the
  truncation at $`\nu`$ = 19 and 21. A single further pole would make $`\lvert r_{k+1} - r_k\rvert`$ a geometric
  sequence $`\rho^k`$. The second difference of $`\ln[(k + 1.5)^4\rho^k]`$ is about $`-4/(k + 1.5)^2`$ whatever
  $`\rho`$ is, near −0.07 here. The data are constant to the four printed digits, so theirs is below 2×10⁻³.
- **What kind of branch.** A $`\nu^{-3}`$ approach of the ratio is a $`\nu^{-2}`$ correction to the pole's
  coefficients. A term $`(\theta - \theta_0)\ln(\theta - \theta_0)`$ beside the pole gives exactly that: the
  coefficients of $`(1 - x)\ln(1 - x)`$ are $`1/(k(k-1))`$. Ten harmonics do not exclude an algebraic
  $`(\theta - \theta_0)^\alpha`$ with $`\alpha`$ near 1.

*Status:* **numerical evidence**, conditional on the selection rule as in §3.3. The fits and their controls are
reproducible. The branch point is read from ten harmonics, one Padé signature and one power law, and no analysis of
the inner equations near the singularity has confirmed it.

### 8.6 What it means for #137 and #138

**#137, the mapped Fourier basis.**

- **Where to tie the map.** The map can be tied to 1.5344 ∓ 1.4284i in the paper's gauge, one singularity pair per
  half period at $`\mathrm{Re}\,\theta = 1.5344 + k\pi`$. In the repository's gauge that is 3.1052 ∓ 1.4284i. All
  the fits agree on the location to 3×10⁻⁴, so #137 can take it from here at $`D = \infty`$. At finite D it must
  come from the states (#136, #93).
- **A conformal map, not a pole subtraction.** Subtracting the pole does not widen the strip, since the branch point
  stays at the same distance. A conformal map needs only the location. The fits find nothing closer to the axis, and
  the further structure lies outward on the same vertical line.
- **The control.** The control #137 asks for, a map tied to the wrong poles, can be the pole moved by
  $`\pi/2`$. That is what this note's gauge mutation does.

**#138, the rational trial solution.**

- **The [L/1] route gains slowly.** By least squares [1/1] reproduces the harmonics to 5.2×10⁻³, and each further L
  gains a factor of 2.2–3.6, reaching 1.3×10⁻⁵ at [7/1]. With one pole the family has to imitate a branch point.
- **Grow the family diagonally.** Adding poles and zeros in turn ([1/2], [2/2], [2/3], ...) gains about a factor 7
  per complex parameter. The natural next form has the logarithm built in, a pole plus
  $`(\theta - \theta_0)\ln(\theta - \theta_0)`$. It is not tested here.
- **How far a few parameters go.** By least squares [1/1] and [1/2] match the profile to 5.2×10⁻³ and 7.8×10⁻⁴
  (4.7×10⁻² and 7.8×10⁻³ as Padé). So they can return ω∞ only as precisely as the selection is sensitive to the
  profile at that level.
- **Starting values.** Every fit's P and Q, to 30 digits, is in `I133-mac/real_A2.72.json`.

### 8.7 Reproduction

From `solver/eeg_profile`, with mpmath. Each real-data run takes about a minute on one core.

```
python3 rational_fit.py                        # 8.4-8.5 at A = 2.72, both J; checks S and D pass, exit 0
python3 rational_fit.py --tail                 # the same with the A series' tail estimate added
python3 rational_fit.py --A 2.71918            # at the paper's A_*/sqrt 2 (check D skipped)
python3 rational_fit.py --plant=rational       # planted [2/3]: P1-P3 pass, exit 0
python3 rational_fit.py --plant=rational --mutate=gauge   # P3 fails, exit 1 (and --mutate=drop)
python3 rational_fit.py --plant=branch         # marching, exit 0
python3 rational_fit.py --plant=poles          # the converged half fails as stated (8.3), exit 1
python3 rational_controls.py                   # every row of 8.3, "all controls behave", exit 0
```

### 8.8 Proposed text (for the coordinator)

For the math paper's "The shape of the echo", after the model of [EEG26b]:

> Eq. (16) of [EEG26b] is the first member of the rational family
> $`A_*\mathrm{Re}[e^{i\theta}P_L(e^{2i\theta})/Q_N(e^{2i\theta})]`$. Fitted to the tuned harmonics through
> $`\nu = 19`$, the family's error falls by about a factor 7 per complex parameter, from 3.5% for Eq. (16) to
> $`3\times10^{-7}`$ with 8 parameters for 10 harmonics. Every fit puts the nearest singularity at
> $`1.5344 \mp 1.4284i`$. The further poles pair with zeros and move toward it as the order grows, and the harmonic
> ratios approach their limit like $`\nu^{-3}`$. Both signs point to a pole with a weaker branch at the same point,
> not an isolated pole.

References for this section, each checked against the publisher's or a catalog record:

- H. Padé, Ann. Sci. Éc. Norm. Supér. (3) 9 (1892) S3–S93, doi:10.24033/asens.378.
- K. Levenberg, Q. Appl. Math. 2 (1944) 164–168, doi:10.1090/qam/10666.
- D. W. Marquardt, J. SIAM 11 (1963) 431–441, doi:10.1137/0111030.
- C. L. Lawson, *Contributions to the theory of linear least maximum approximation*, PhD thesis, UCLA (1961), from the
  Mathematics Genealogy Project record.
- H. Stahl, J. Approx. Theory 91 (1997) 139–204, doi:10.1006/jath.1997.3141.
- R. de Montessus de Ballore, Bull. Soc. Math. France 30 (1902) 28–36, doi:10.24033/bsmf.653.

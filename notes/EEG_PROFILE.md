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
profile at A = 2.72, $`0.05509\,e^{0.0988i}`$. That profile comes from the inner equations alone, while EEG extrapolated
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
maximum splits into two humps. The paper's $`q`$ is a third of that. The norm is
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

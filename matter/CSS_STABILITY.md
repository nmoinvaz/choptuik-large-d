# The κ-model CSS's oscillatory mode: its D = ∞ limit and the change of stability κ_c(D)

Issues #376 and #377. The phase-rotating CSS solution of the Hirschmann–Eardley κ-model (matter/CSS.md) has, besides
its relevant mode λ_phys, an oscillatory pair λ_osc, λ̄_osc that is unstable for small κ. At κ = 0 the pair stays
unstable up to D = 80 (matter/COMPLEX_CRITICAL.md §2). In D = 4 it crosses to Re λ < 0 at κ ≈ 0.0754 [HE97], above
which the CSS has a single unstable mode. This page computes the pair at D = ∞ (§1, #376) and follows it in κ at fixed
D to κ_c(D), where Re λ_osc = 0 (§2, #377).

Conventions as in matter/CSS.md: D = n + 2, τ = −ln(−t), perturbations ∝ $`e^{\lambda\tau}`$, γ = 1/λ (radius
exponent), and the field rotates as $`F\propto(-t)^{i\omega}`$.

**Result in brief.**

- PENDING

## 1. The D = ∞ spectrum of the inner CSS solution (#376)

### 1.1 The perturbation problem

The D = ∞ inner problem is the autonomous heat-type system of matter/CSS.md §8 in s = r²/(2n) and t. With
τ = −ln(−t), ξ = √(2s)/(−t), z = ξ², θ = ξ∂_ξ and X = ξ∂_ξ + ∂_τ (so that ∂_t = e^τ X and ∂_s = e^{2τ}ξ⁻¹∂_ξ) it reads

```math
c^2\,\theta F = z\Big[(X^2 + X)F - \frac{Xc}{c}\,XF + \frac{2\kappa\bar F}{\rho}\,(XF)^2\Big], \qquad
c\,\theta c = 2z\,h\,(XF)(X\bar F), \qquad \rho = 1-\kappa F\bar F,\ h = \rho^{-2} .
```

For the CSS solution F = $`e^{-i\omega\tau}`$f(ξ), c = b(ξ) this is the inner ODE of `css_inner.py`. A perturbation is

```math
F = e^{-i\omega\tau}\big(f + \epsilon e^{\lambda\tau}A\big), \qquad
\bar F = e^{i\omega\tau}\big(g + \epsilon e^{\lambda\tau}B\big), \qquad c = b + \epsilon e^{\lambda\tau}\beta ,
```

with A and B independent (the real system, complexified), so X acts on A, B, β as θ + λ − iω, θ + λ + iω, θ + λ.
The data at the irregular singular point ξ = 0 are A(0) and B(0). The lapse there is fixed by the matching (GAMMA.md §4),
β(0) = 0. This gauge removes the time reparametrizations t → t + η(t): with η ∝ (−t)^{1−λ} they are perturbations
with β(0) = λ − 1, so only the λ = 1 one survives, as the time translation. As for the background, the regular formal
series in z diverges, with the fast mode $`e^{-b^2/(2\xi^2)}`$, and a growth rate is a λ at which a regular perturbation
has zero Stokes jump (GAMMA.md §5).

**The determinant.** The upper and lower lateral solutions are computed as in `css_inner.py`: the optimally truncated
series at $`\xi_0 = r_0e^{\pm3i\pi/8}`$, where the fast mode is exponentially large, integrated along the ray to |ξ| = r₁ and
along the arc to the real axis. Both sides are integrated, background and perturbation together, so λ may be complex.
The jumps of A and B at ξ = r₁ are linear in the data, J = M(λ)(A(0), B(0)), and the growth rates are the zeros of
det M at every setting (r₀, r₁). A zero that moves with the setting is an artefact: det M is the product of the
determinant of the Stokes multipliers and that of the fast modes' (A, B) values at r₁, and the latter can vanish.
Settings: A = (0.1, 0.3), B = (0.1, 0.25), C = (0.09, 0.3); r₁ stays below the background's movable singularity at
|ξ| ≈ 0.335 (matter/STRUCTURE.md §7).

**Exact modes.** The symmetries of the inner system (matter/CSS.md §8) give five regular perturbations with zero jump:

| λ | symmetry | (A, B, β) |
|---|---|---|
| 0 | U(1) phase | (if, −ig, 0) |
| 1 | t-translation | (θf − iωf, θg + iωg, θb) |
| 2 | s-translation with the gauge term that restores β(0) = 0 | (f′/ξ − 2b₁(θf − iωf), g′/ξ − 2b₁(θg + iωg), b′/ξ − 2b₁(θb + b)), b = 1 + b₁ξ² + … |
| ±iω | target isometries | (1, −κg², 0) at +iω, (−κf², 1, 0) at −iω |

The λ = 2 row is GAMMA.md §4's construction for the CSS, where the gauge function P is a constant.

### 1.2 Checks and controls

`css_inner_pert.py` shares with `css_inner.py` the background series and the inner ODE, and nothing with the
finite-D codes (`css_pert.py`, the collocation) but the derivation. Each check below was seen to fail on a broken input.

| check | result | control, and its result |
|---|---|---|
| the linearized rhs is the ε-derivative of the (s, t) inner PDE (`css_inner_pert_eqs.py`, sympy, 8 random complex points, κ = 0–0.96, complex λ) | largest residual 2×10⁻¹⁵, relative to the largest term | `--mutate=hamsign` (sign of the β equation's source) and `nometric` (β not fed back): residuals 2.9 and 1.1, rejected |
| background part of the rhs against `css_inner.rhs` | 3.6×10⁻¹⁵ | — |
| the series recursion reproduces the five exact modes' series (60 coefficients, κ = 0) | relative 1.4×10⁻²⁷ (λ = 0), 2.2×10⁻²⁷ (1), 4.4×10⁻²⁷ (2), 0 (±iω) | the same data at λ + 0.01: relative 6×10⁹ to 4×10⁹³, rejected (`css_inner_pert.py check`) |
| M(λ) is singular at λ = 0, 1, 2 (smallest / largest singular value, setting A) | 1.3×10⁻¹³, 2.2×10⁻¹⁴, 5.2×10⁻¹³ | 0.072 at λ = 1.5 |
| the same with the mutations | — | nometric: 1.0, 1.0 at λ = 1, 2 (smallest = largest, 0.38 and 3.0); hamsign: 0.32, 0.55 |
| the secant from λ = 2.1 | 2 − 1.4×10⁻¹² (A), 2 − 5.9×10⁻¹³ (B), 2 − 1.4×10⁻¹² (C) | nometric: runs to λ = 9×10⁻⁷ (not converged; the test field's phase modes); hamsign: runs to λ = 0 |

**The control of the issue passes: the inner problem's exact λ = 2 mode is found**, as a zero of det M at all three
settings to 1.4×10⁻¹², with the series of the symmetry construction reproduced by the recursion to 4×10⁻²⁷. The secant
from 1.1 finds λ = 1 to 2×10⁻¹² (A, B). For κ = 0 the shift modes A = 1, B = 1 are constants, and det M(±iω) is 0 to
the last bit.

### 1.3 λ_osc(∞) against the finite-D values

The secant on det M from the extrapolated 0.536 + 7.93i (`css_inner_pert.py root 0 0.536+7.93j`, raw
`I376-mac/roots_k0.jsonl`):

| setting | λ_osc(∞) | Im λ_osc / ω∞ |
|---|---|---|
| A (r₀ 0.1, r₁ 0.3) | 0.5365223424 + 7.9320241697i | 1.9934870461 |
| B (0.1, 0.25) | 0.5365223426 + 7.9320241699i | 1.9934870461 |
| C (0.09, 0.3) | 0.5365223424 + 7.9320241697i | 1.9934870461 |
| A, integrator rtol 10⁻¹³ | 0.5365223423 + 7.9320241697i | 1.9934870461 |
| A, series at 60 digits | 0.5365223422 + 7.9320241696i | 1.9934870461 |

The largest difference between the rows is 2×10⁻¹⁰, a spread over settings of one method, not an error bound.
**At D = ∞ the κ = 0 CSS keeps its oscillatory unstable pair, λ_osc(∞) = 0.5365223424 ± 7.9320241697i, and
Im λ_osc/ω∞ = 1.9934870, not 2** (2ω∞ − Im λ_osc = 0.0259, against the spread 2×10⁻¹⁰).

Against the 1/n fits of the finite-D track (`complex_osc_fit.py` on raw `I95-mac/css/track_k0*.jsonl`, degree K = 3, 4
on D ≥ 15, 20, 30):

| | fits (spread over the six choices) | D = ∞ inner | inside |
|---|---|---|---|
| Re λ_osc | 0.53642930–0.53653455 | 0.53652234 | yes |
| Im λ_osc / ω | 1.99348767–1.99349871 | 1.99348705 | no, 6.2×10⁻⁷ below the lowest fit |

The Im/ω fits fall monotonically toward the inner value as D_min and K grow (1.99349871 at D ≥ 15, K = 3 to 1.99348767 at
D ≥ 30, K = 4). The finite-D values converge to the inner one at the rate of an ordinary 1/n series:
n(Re λ_osc(D) − Re λ_osc(∞)) = −0.98, −1.02, −1.09, −1.13 and n(Im/ω(D) − Im/ω(∞)) = −0.565, −0.582, −0.606, −0.617 at
D = 15.3, 20.3, 39.2, 80. Fits with the limit fixed at the inner value, in powers 1/n … 1/n^{K+1}, have rms residuals 1
to 3.5 times smaller than the free fits with the same number of parameters (Re: 1.1×10⁻⁶ against 3.1×10⁻⁶ at D ≥ 15,
K = 3; Im/ω: 2.7×10⁻⁹ against 7.8×10⁻⁹ at D ≥ 20, K = 4), with the 1/n coefficients −1.178 (Re) and −0.6273 (Im/ω).
So the issue's acceptance holds for Re λ_osc and fails narrowly for Im/ω: the inner value lies outside the fits' range
by 6% of its width. As for ω∞ in matter/CSS.md §7, the narrow spread over fit choices understates the extrapolation
error (§3).

### 1.4 Box counts

Winding numbers of det M (`css_inner_pert.py count`, the detrended argument principle of `complex_css.py`, every phase
step ≤ 0.5 rad after bisection; raw `I376-mac/counts_k0.jsonl`), in matter/COMPLEX_CRITICAL.md §2's boxes with
ω = ω∞ = 3.97897 and ε = ¼:

| box | region | expected | setting A | setting B | determinants (A, B) |
|---|---|---|---|---|---|
| A | 0.05 < Re λ < 0.9, \|Im λ\| < ω − ε | none | 0 | 0 | 160, 156 |
| B | 0.9 < Re λ < 5, \|Im λ\| < ω − ε | λ = 1, λ = 2 | 2 | 2 | 132, 136 |
| C | 0.05 < Re λ < 5, ω + ε < Im λ < Im λ_osc + 1.5 = 9.432 | λ_osc | 1 | 1 | 74, 72 |
| S | 0.05 < Re λ < 5, ω − ε < Im λ < ω + ε | none | 0 | 0 | 68, 68 |

Controls, each seen to fire (setting A): a zero planted at 1 + 6i makes C count 2; `--mutate=nometric` makes B count 0.
At finite D the strip S holds the jump method's resonance artefacts (6 zeros at D = 4, none at D ≥ 40); at D = ∞ it
holds none at either setting, as expected, since the inner problem has no regular singular point. Zeros come in pairs
λ, λ̄ by the conjugation symmetry of the real perturbation system, so C covers the lower half plane as well (checked
numerically in `I376-mac/conj_k0.jsonl`, PENDING).

**So in these boxes the D = ∞ κ = 0 CSS has λ = 1 (gauge), λ = 2 (the relevant mode, exact) and the pair λ_osc, λ̄_osc,
and no other growth rate: three unstable directions, as at every finite D from 4 to 80.** Its instability is not a
finite-D effect. Not covered: growth rates with Re λ > 5, |Im λ| > 9.43, or 0 < Re λ < 0.05, and the status of the
selection rule itself (Conjecture 1), which this computation assumes, as GAMMA.md §5 does for λ = 2.

## 2. The change of stability κ_c(D) (#377)

### 2.1 Method

`css_kappa_crit.py path D …` starts from the κ = 0 background of `data/css_bg_k0.jsonl` and the κ = 0 roots of
matter/COMPLEX_CRITICAL.md §4. It steps in κ: the background by Newton on its horizon jump (`css_lateral.newton`), then
λ_osc and λ_phys by secant on `css_pert.py`'s det M(λ) at the circle radii R = 0.85 x_h and 0.6 x_h. A step is accepted
when |J| < 10⁻¹⁰ and each root converges at both radii to within 10⁻⁷ and near its extrapolation; otherwise the step is
halved. The path stops at the first κ where Re λ_osc < 0 at both radii. `crit` then refines the zero of Re λ_osc(κ)
by regula falsi (Illinois) in that bracket, at each radius separately, each evaluation a new background and a new root.
The crossing is accepted only as a sign change at both radii, since the jump method's artefacts move with R.

### 2.2 D = 4 against Hirschmann and Eardley

| quantity | R = 0.85 x_h | R = 0.6 x_h |
|---|---|---|
| Re λ_osc at κ = 0.07125 / 0.12125 | +0.0172251 / −0.1894496 | +0.0172251 / −0.1894496 |
| κ_c(4) | 0.0757329315 | 0.0757329315 |
| λ_osc at κ_c | ±3.6888136i (ω = 1.8562917, Im λ_osc/ω = 1.98720) | the same |
| γ = 1/λ_phys at κ_c | 0.3750460 | 0.3750460 |

The two radii agree to 5×10⁻¹² in κ_c. The collocation of matter/CSS.md §2 (`css_scan.py`, N = 40 and 56), which
shares only the ODE with the jump method, places the crossing at the same κ:

| κ | Re λ_osc (collocation) | Im λ_osc | unstable directions |
|---|---|---|---|
| 0.0754 | +0.0012878 | ±3.6880113 | 3 |
| 0.0757329315 | +7×10⁻⁸ (zero within its error, about 10⁻⁷ at D = 4) | ±3.6888136 | 1 (and the marginal pair) |
| 0.076 | −0.0010341 | ±3.6894573 | 1 |

**κ_c(4) = 0.075733, not HE97's 0.0754**: at κ = 0.0754 the oscillatory pair still grows, with Re λ_osc = +0.0012878
(collocation) and PENDING (jump method, both radii).
The model and its normalization are HE97's: the action is their eq. (4), and the D = 4 values at three κ are
reproduced, γ = 0.3871055 (κ = 0; HE97 Table 1 0.3871), 0.2641066 (κ = 1; 0.2641) and 0.1468992 (κ = 10, jump method at
both radii and collocation; 0.1469). The background continued in κ from 0 to 1 at D = 4 lands on HHS96's axion–dilaton
solution, ω = 1.1769527242 and |f(0)| = 0.8925554116 (matter/CSS.md §4). And the κ = 0 pair agrees with HE95b within
their error bars (matter/COMPLEX_CRITICAL.md §2). HE97's Table 1 entry "γ ≳ 0.373" at "κ ≲ 0.0754" holds as an
inequality: γ = 0.375046 at κ_c. Under their map ω_BD = −3/2 + 1/(8κ), κ_c(4) is ω_BD = 0.1505 (their 0.158). §3
records the discrepancy.

### 2.3 κ_c(D)

`css_kappa_crit.py path` and `crit` at D = 4, 6, 10, 20, 40 (raw `I377-mac/path_D*.jsonl`, `crit.jsonl`). The radius
spread is the difference between the κ_c found at the two radii, a spread between two settings of one method:

| D | κ_c(D) | radius spread | Re λ_osc at the bracket (κ = 0.07125 / 0.12125, both radii) | Im λ_osc at κ_c | ω at κ_c | Im λ_osc / ω | γ at κ_c |
|---|---|---|---|---|---|---|---|
| 4 | 0.0757329315 | 5×10⁻¹² | +0.017225 / −0.189450 | 3.6888136 | 1.8562917 | 1.98720 | 0.375046 |
| 6 | 0.0895073544 | 2×10⁻¹¹ | +0.075270 / −0.135060 | 4.2052783 | 2.1089666 | 1.99400 | 0.432606 |
| 10 | 0.0993081158 | 1×10⁻¹⁰ | +0.121694 / −0.095955 | 5.0272090 | 2.5137611 | 1.99988 | 0.457452 |
| 20 | 0.1065515774 | 2×10⁻¹¹ | +0.157997 / −0.065500 | 6.0137108 | 3.0006999 | 2.00410 | 0.474692 |
| 40 | 0.1103183901 | 6×10⁻¹⁰ | +0.177239 / −0.049124 | 6.7339595 | 3.3571176 | 2.00588 | 0.485261 |

At every D the sign change of Re λ_osc shows at both radii, with the same values to 10⁻⁸. **κ_c(D) rises
monotonically with D and levels off**: polynomial fits in 1/n (n = D − 2) of degree 2–3 on D ≥ 4, 6, 10 give
κ_c(∞) = 0.1129–0.1142 (a spread over fit choices on three to five points, not an error bound). So the band of κ where
the CSS has three unstable directions widens with D, from κ < 0.0757 at D = 4 to κ < 0.110 at D = 40, and the
axion–dilaton (κ = 1) stays far on the stable side. A CSS solution with κ in between (0.076 < κ < 0.110) is critical
in D = 4 and not in D = 40. γ at κ_c rises with D toward 1/2 as for every κ.

PENDING: the D = ∞ path (§2.4).

## 3. Discrepancies

- **HE97's κ_c(4) = 0.0754 is not reproduced.** Two independent methods here, complex-path shooting at two circle radii
  and the collocation, give κ_c(4) = 0.0757329315, 3.3×10⁻⁴ above it (0.44%). At κ = 0.0754 the oscillatory pair still
  grows, Re λ_osc = +0.0013, so the issue's acceptance "κ_c(4) reproduces 0.0754 to their digits" fails. The model and
  its κ are HE97's (the same action; γ at κ = 0, 1, 10 reproduced to their digits, §2.2), so the difference is not one
  of normalization. HE97 give no error estimate for 0.0754 and no σ of the oscillatory mode for κ > 0, so the cause
  is not known. Their horizon start, a second-order Taylor expansion (their §V), is not it by itself: at D = 4 the
  horizon exponent has Re μ ≈ 1.3–1.7 < 2, where such a start does select the analytic solution (matter/CSS.md §3), and
  their γ at κ = 0, 1, 10 agree with ours. Near the crossing dRe λ_osc/dκ ≈ −3.9, so their value corresponds to an
  error of 1.3×10⁻³ in Re λ_osc. LC96's evolutions put the switch near κ ≈ 0.083 (matter/LITERATURE.md §5.2), on the
  other side of 0.0757.
- **Im λ_osc/ω∞ at κ = 0** lies 6.2×10⁻⁷ below the spread of the six 1/n fits of the finite-D track (§1.3), so the
  acceptance of #376 holds for Re λ_osc and fails narrowly for Im/ω. The fits fall toward the inner value as D_min and
  K grow, and the finite-D values converge to it like 1/n, so this reads as the spread over fit choices understating
  the extrapolation error, as for ω∞ in matter/CSS.md §7.

## 4. Proposed text for matter/CSS.md, matter/COMPLEX_CRITICAL.md and the README

PENDING

## 5. Proposed notes/LIT_REVIEW.md entries

- **[HE97] E. W. Hirschmann, D. M. Eardley, "Criticality and bifurcation in the gravitational collapse of a self-coupled
  scalar field", PRD 56, 4696–4705 (1997), arXiv:gr-qc/9511052** (read 2026-11-14: the arXiv TeX source, whole; the
  record checked on INSPIRE and Crossref). Contains: the κ-model action (their eq. 4, the one of matter/CSS.md §1), its
  D = 4 CSS solutions and the linear perturbations, with det A(σ) counted by the argument principle around contours in
  the lower half σ plane and its zeros found by secant; the four gauge modes σ = 0, −i, ±ω (λ = 0, 1, ±iω here); the
  statement that "additional modes … approach the real axis as κ decreases" and that the first crossing "occurs at
  κ ≈ 0.0754" ("Hopf-type bifurcation"); Table 1: γ = 0.1469 (κ = 10), 0.2641 (1), "≳ 0.373" at "≲ 0.0754", 0.3871 (0),
  ω_BD = −3/2 + 1/(8κ); more unstable modes by κ ≈ −0.28. Does not contain: the oscillatory mode's σ at the crossing or
  for any κ > 0, an error estimate for 0.0754, any dimension other than 4.
- **[L98] S. L. Liebling, "Multiply unstable black hole critical solutions", PRD 58, 084015 (1998), arXiv:gr-qc/9805043**
  (abstract only, 2026-11-14): evolutions of the same model in which a family of data finds the CSS as its critical
  solution in the region where the DSS is the attractor. A search (2026-11-14) found no later computation of κ_c(4)
  besides HE97's; the review literature quotes 0.0754.

## 6. Scripts, data and how to reproduce

All in `solver/matter/`, run single-process under `nice -n 10` on the Mac (Apple M5); times for one process there, with
the machine shared. Raw output: [choptuik-large-d-raw](https://github.com/nmoinvaz/choptuik-large-d-raw), folders
`I376-mac/` and `I377-mac/` (the queue scripts that ran are in them, with their logs).

| script | what it does | cost |
|---|---|---|
| `css_inner_pert.py check / det / root / count / kpath / kcrit` | the D = ∞ inner CSS perturbations: series, lateral solutions, det M(λ), secant roots at settings A, B, C, winding counts (`--plant`, `--mutate=nometric\|hamsign`), the root followed in κ and κ_c(∞) | 1.5 s per determinant (κ = 0, 5.5 s to build a background); 1 min per root at three settings; 2.5–5 min per box |
| `css_inner_pert_eqs.py [--mutate=…]` | sympy: the linearized rhs is the ε-derivative of the inner PDE in (s, t) | 20 s |
| `css_kappa_crit.py path / crit / bgpath / selftest` | finite D: the background and λ_osc, λ_phys followed in κ at two radii; κ_c refined at each radius; the background alone along κ; the crossing finder on a planted function | 10–60 s per κ step; 1–4 min per κ_c |

To reproduce from the repository root (the second and third lines need the venv of SETUP.md):

```
python3 solver/matter/css_inner_pert_eqs.py                                  # worst residual 2e-15, exit 0
python3 solver/matter/css_inner_pert.py check 0                              # check PASS
python3 solver/matter/css_inner_pert.py root 0 2.1 --set=A                   # lambda = 2 - 1.4e-12
python3 solver/matter/css_inner_pert.py root 0 0.536+7.93j --set=A           # 0.5365223424 + 7.9320241697i
python3 solver/matter/css_inner_pert.py count 0 0.05 5 4.2289695 9.432 --set=A   # winding 1
python3 solver/matter/css_kappa_crit.py selftest
python3 solver/matter/css_kappa_crit.py path 4 0.2586698339+3.5022487970j 2.5832751746 --k1=1 --out=/tmp/p4.jsonl
python3 solver/matter/css_kappa_crit.py crit /tmp/p4.jsonl 4                 # kappa_c 0.0757329315 at both radii
python3 solver/matter/css_scan.py 0.0754 4 --guess=1.856291661723,4.797192767370,1.231464079551 --N 40,56
                                                                             # Re lambda_osc = +0.0012877524
```

## 7. Recommended tasks

PENDING

## References

- [HE95b] E. W. Hirschmann, D. M. Eardley, PRD 52, 5850 (1995), gr-qc/9506078.
- [HE97] E. W. Hirschmann, D. M. Eardley, "Criticality and bifurcation in the gravitational collapse of a self-coupled
  scalar field", Phys. Rev. D 56, 4696–4705 (1997), doi:10.1103/PhysRevD.56.4696, arXiv:gr-qc/9511052 (verified
  2026-11-14 against INSPIRE's record and Crossref's for the DOI: title, authors, volume, issue 8, pages 4696–4705,
  published 1997-10-15).

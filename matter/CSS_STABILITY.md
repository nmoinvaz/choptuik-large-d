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

PENDING

## 3. Discrepancies

PENDING

## 4. Proposed text for matter/CSS.md, matter/COMPLEX_CRITICAL.md and the README

PENDING

## 5. Proposed notes/LIT_REVIEW.md entry

PENDING

## 6. Scripts, data and how to reproduce

PENDING

## 7. Recommended tasks

PENDING

## References

- [HE95b] E. W. Hirschmann, D. M. Eardley, PRD 52, 5850 (1995), gr-qc/9506078.
- [HE97] E. W. Hirschmann, D. M. Eardley, "Criticality and bifurcation in the gravitational collapse of a self-coupled
  scalar field", Phys. Rev. D 56, 4696–4705 (1997), doi:10.1103/PhysRevD.56.4696, arXiv:gr-qc/9511052 (verified
  2026-11-14 against INSPIRE's record and Crossref's for the DOI: title, authors, volume, issue 8, pages 4696–4705,
  published 1997-10-15).

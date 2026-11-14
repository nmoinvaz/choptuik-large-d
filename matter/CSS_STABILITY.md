# The κ-model CSS's oscillatory mode: its D = ∞ limit and the change of stability κ_c(D)

Issues #376 and #377. The phase-rotating CSS solution of the Hirschmann–Eardley κ-model (matter/CSS.md) has, besides
its relevant mode λ_phys, an oscillatory pair λ_osc, λ̄_osc that is unstable for small κ. At κ = 0 the pair stays
unstable up to D = 80 (matter/COMPLEX_CRITICAL.md §2). In D = 4 Hirschmann and Eardley found that it turns stable at a
finite κ, above which the CSS has a single unstable mode [HE97]. This page computes the pair at D = ∞ (§1, #376) and
follows it in κ at fixed D to κ_c(D), where Re λ_osc = 0 (§2, #377).

Conventions as in matter/CSS.md: D = n + 2, τ = −ln(−t), perturbations ∝ $`e^{\lambda\tau}`$, γ = 1/λ (radius
exponent), and the field rotates as $`F\propto(-t)^{i\omega}`$.

**Result in brief.**

- **D = ∞, κ = 0.** The perturbation problem of the inner CSS solution is set up as the background's: the growth rates
  are the λ at which a regular perturbation has a zero Stokes jump. Its exact λ = 2 mode is found, to 1.4×10⁻¹², and so
  are λ = 0, 1 and ±iω. **The oscillatory pair survives at D = ∞: λ_osc(∞) = 0.5365223424 ± 7.9320241697i**, the same
  to 2×10⁻¹⁰ over three matching settings, with **Im λ_osc/ω∞ = 1.99349, not 2**. Box counts find λ = 1, 2 and the
  pair, and nothing else, in 0.05 < Re λ < 5, |Im λ| < 9.43. So the κ = 0 CSS has three unstable directions at D = ∞,
  as at every D from 4 to 80.
- **Against finite D.** The 1/n fits of the finite-D track give Re λ_osc → 0.53643–0.53653, which holds the inner
  0.53652, and Im λ_osc/ω → 1.9934877–1.9934987, which the inner 1.9934870 misses by 6×10⁻⁷ (§3). The finite-D values
  converge to the inner ones like 1/n.
- **κ_c(D).** Re λ_osc changes sign, at both circle radii, at κ_c = 0.0757329, 0.0895074, 0.0993081, 0.1065516 and
  0.1103184 for D = 4, 6, 10, 20 and 40, and the inner problem gives κ_c(∞) = 0.1143406. κ_c(D) rises monotonically
  and converges to κ_c(∞) like 1/n. So the band κ > κ_c(D), where the oscillatory pair is stable and the CSS can be
  critical, narrows with D but stays open at D = ∞, with the axion–dilaton (κ = 1) far inside it. Whether other modes
  turn unstable at intermediate κ was not counted here; at κ = 1 the CSS is critical to D = 80 (matter/CSS.md §6).
- **D = 4.** Two independent methods, complex-path shooting at two circle radii and the collocation, give
  κ_c(4) = 0.075733 (§2.2).
- **D = ∞, κ = 1.** For the axion–dilaton the counts find λ = 1 and λ = 2 and no other growth rate in
  0.05 < Re λ < 8, |Im λ| < 3ω∞, so its relevant mode is unique there at D = ∞ (§1.6).

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

with A and B independent (the real system, complexified), so X acts on A, B, β as θ + λ − iω, θ + λ + iω, θ + λ. The
data at the irregular singular point ξ = 0 are A(0) and B(0). The lapse there is fixed by the matching (GAMMA.md §4),
β(0) = 0. This gauge removes the time reparametrizations t → t + η(t): with η ∝ (−t)^{1−λ} they are perturbations with
β(0) = λ − 1, so only the λ = 1 one survives, as the time translation. As for the background, the regular formal series
in z diverges, with the fast mode $`e^{-b^2/(2\xi^2)}`$, and a growth rate is a λ at which a regular perturbation has
zero Stokes jump (GAMMA.md §5).

**The determinant.** The upper and lower lateral solutions are computed as in `css_inner.py`: the optimally truncated
series at $`\xi_0 = r_0e^{\pm3i\pi/8}`$, where the fast mode is exponentially large, integrated along the ray to |ξ| =
r₁ and along the arc to the real axis. Both sides are integrated, background and perturbation together, so λ may be
complex. The jumps of A and B at ξ = r₁ are linear in the data, J = M(λ)(A(0), B(0)), and the growth rates are the zeros
of det M at every setting (r₀, r₁). A zero that moves with the setting is an artefact: det M is the product of the
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
| the series recursion reproduces the five exact modes' series (60 coefficients, κ = 0) | relative 1.4×10⁻²⁷ (λ = 0), 2.2×10⁻²⁷ (1), 4.4×10⁻²⁷ (2), 0 (±iω) | the same data at λ + 0.01: relative 6×10⁹ to 1×10¹³, rejected (`css_inner_pert.py check`) |
| M(λ) is singular at λ = 0, 1, 2 (smallest / largest singular value, setting A) | 1.3×10⁻¹³, 2.2×10⁻¹⁴, 5.2×10⁻¹³ | 0.072 at λ = 1.5 |
| the same two checks at κ = 0.5 (`check 0.5`), where the ±iω modes have B = −κg² ≠ 0 | series ≤ 2.6×10⁻²⁷; M singular to ≤ 8.4×10⁻¹³ at λ = 0, 1, 2 | 0.11 at λ = 1.5; series at λ + 0.01 miss by 3×10⁹ or more |
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

The Im/ω fits fall monotonically toward the inner value as D_min and K grow (1.99349871 at D ≥ 15, K = 3 to 1.99348767
at D ≥ 30, K = 4). The finite-D values converge to the inner one at the rate of an ordinary 1/n series: n(Re λ_osc(D) −
Re λ_osc(∞)) = −0.98, −1.02, −1.09, −1.13 and n(Im/ω(D) − Im/ω(∞)) = −0.565, −0.582, −0.606, −0.617 at D = 15.3, 20.3,
39.2, 80. Fits with the limit fixed at the inner value, in powers 1/n … 1/n^{K+1}, have rms residuals 1 to 3.5 times
smaller than the free fits with the same number of parameters (Re: 1.1×10⁻⁶ against 3.1×10⁻⁶ at D ≥ 15, K = 3; Im/ω:
2.7×10⁻⁹ against 7.8×10⁻⁹ at D ≥ 20, K = 4), with the 1/n coefficients −1.178 (Re) and −0.6273 (Im/ω). So the issue's
acceptance holds for Re λ_osc and fails narrowly for Im/ω: the inner value lies outside the fits' range by 6% of its
width. As for ω∞ in matter/CSS.md §7, the narrow spread over fit choices understates the extrapolation error (§3).

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
numerically: det M(λ̄) = conj det M(λ) to 1.3×10⁻¹⁰ relative at 1.3 ± 2.1i and 0.7 ± 6.5i, and the secant from
0.536 − 7.93i converges to 0.5365223424 − 7.9320241698i at settings A and B; raw `I376-mac/conj_k0.jsonl`).

**So in these boxes the D = ∞ κ = 0 CSS has λ = 1 (gauge), λ = 2 (the relevant mode, exact) and the pair λ_osc, λ̄_osc,
and no other growth rate: three unstable directions, as at every finite D from 4 to 80.** Its instability is not a
finite-D effect. Not covered: growth rates with Re λ > 5, |Im λ| > 9.43, or 0 < Re λ < 0.05, and the status of the
selection rule itself (Conjecture 1), which this computation assumes, as GAMMA.md §5 does for λ = 2.

### 1.5 A second implementation (#771)

`solver/matter/css_inf_jump.py` finds the same growth rates as zeros of a 2×2 Stokes-jump determinant, not of det M.
sympy derives the perturbation ODEs from matter/CSS.md §8's PDE (F and F̄ independent, so λ may be complex; w(0) = 0
as the gauge). The regular formal series are solved order by order, the lateral solutions run from ξ₀ = r₀e^{±iα}
along a ray and an arc to r₁, and the secant finds the zeros of the jumps' determinant. It shares with
`css_inner_pert.py` only the PDE, CSS.md's (|f(0)|, ω) as Newton's start, and the lateral-solution idea of
`struct_css.py`.

| check (r₀ 0.1, r₁ 0.3, α 3π/8, rtol 10⁻¹³) | result | bound |
|---|---|---|
| the exact λ = 2 | 2 − 5.3×10⁻¹⁴ | 10⁻¹⁰ |
| λ_osc(∞) | 0.5365223423982 + 7.9320241696692i, 3.1×10⁻¹¹ from §1.3 | 10⁻⁸ |
| spread over six settings (r₀ 0.09–0.11, r₁ 0.25–0.3, α 0.3π or 3π/8) | λ = 2 within 1.4×10⁻¹², λ_osc within 1.2×10⁻¹⁰ | 10⁻¹⁰, 10⁻⁸ |
| a planted ₂F₀(a, 0.7;; 2ξ²) through the same series, paths and secant | zeros a = −1, −2 to 1×10⁻¹⁴ and 2×10⁻¹³ | 10⁻¹² |
| its control, one coefficient × 1.001 | zeros off by 1.2×10⁻⁴ and 5.9×10⁻⁴, exit 1 | must fail |

The quoted ± is the spread over one method's settings. At r₁ = 0.32, inside the movable singularity at |ξ| ≈ 0.335
(matter/STRUCTURE.md §7), λ = 2 moves by 3.4×10⁻¹⁰, so r₁ stays at 0.3 or below. κ_c(∞) was not recomputed this way.
Raw `I771-mac/` at 3a94ada8.

**The count by the second method (#781).** `solver/matter/css_inf_count.py` winds D(λ) around the full boundary of
0.05 < Re λ < 5, |Im λ| < 9, with steps from 0.25 bisected until each phase step is at most 0.5 rad. D is entire in
λ, so its winding is its number of zeros. The conjugation symmetry is not assumed, only checked.

| setting (r₀, r₁, α) | winding | with a pair planted at 3.1 ± 5i | largest step | determinants | D(λ̄) = conj D(λ) |
|---|---|---|---|---|---|
| A (0.1, 0.3, 3π/8) | 4.000000000000003 | 5.999999999999997 | 0.4985 rad | 464 | 1.1×10⁻⁹ (231 pairs) |
| B (0.11, 0.25, 0.3π) | 3.9999999999999956 | 6.0000000000000036 | 0.4997 rad | 476 | 7.7×10⁻⁹ (237 pairs) |

**So the second method also finds four zeros in the box: λ = 1, λ = 2 and the pair λ_osc, λ̄_osc.** The counter's
selftest uses a synthetic D with e^{4λ} growth and six zeros inside, three of them 0.02 to 0.05 outside its edges. It
counts 6, and 8 with the plant. With one sample per edge (the control) it counts 0.

The exact mode λ = iω lies 0.05 outside the left edge. It is a simple zero of D, since |D(iω + ε)|/|ε| tends to
3.39×10⁴ from three directions. #771's evaluation of D(iω) had not finished. At iω the data (1, 0) is the exact shift
mode, so v and w are rounding noise, and DOP853's absolute tolerance of 10⁻¹⁹ makes it resolve that noise. With the
absolute tolerance at 10⁻¹⁶ for that point, |D(iω)| = 7.8×10⁻⁸. Raw `I781-mac/`.

**The strip −0.05 < Re λ < 0.05, |Im λ| < 9 (#791)** holds exactly λ = 0 and ±iω, by the same count. It winds
2.9999999999999987 at setting A and 2.999999999999998 at B. A pair planted at 0.02 ± 6i makes it 5 at both, the largest
step is 0.4978 and 0.4961 rad, and D(λ̄) = conj D(λ) holds to 1.6×10⁻⁹ and 1.0×10⁻⁸ on the contour.

The three zeros are located by the secant:
- the one near 0 to 2×10⁻¹³ from λ = 0, with the default tolerance;
- the ones near ±iω to 2×10⁻¹¹ (A) and 3×10⁻¹⁰ (B) from ±iω, which is D's noise floor there. These use the absolute
  tolerance of 10⁻¹⁶ (`--atol`), and only these do.

So no growth rate with 0 ≤ Re λ < 0.05 hides in |Im λ| < 9, and §1.4's "not covered" no longer includes that strip.
Raw `I791-mac/`.

### 1.6 The axion–dilaton at D = ∞ (#770)

The same counts at κ = 1, the axion–dilaton, whose inner solution has f(0) = 0.4048768438 and ω∞ = 3.1998939020
(`data/css_inner_kpath.jsonl`). Raw `I770-mac/`.

**The exact modes.** `check 1` passes: the recursion reproduces the five exact modes' series to 8×10⁻²⁴ (3.5×10⁹ or more
at λ + 0.01), and M is singular at λ = 0, 1, 2 to 9.9×10⁻¹³ (0.149 at λ = 1.5). The secant finds λ = 2 to 1.2×10⁻¹² and
λ = 1 to 6.7×10⁻¹³ at each of settings A, B and C, so r₁ = 0.3 still serves at κ = 1. det M(λ̄) = conj det M(λ) to
2.0×10⁻¹² relative at 1.3 ± 2.1i and 0.7 ± 6.5i at settings A and B, so the upper-half boxes S and C cover the lower
half as well.

**The counts**, in §1.4's boxes with ω = ω∞(κ = 1) and ε = ¼. Box C reaches up to 3ω∞ = 9.600, since the pair λ_osc is
damped by κ = 0.9 (§2.4). W is the wider box of #770:

| box | region | expected | setting A | setting B | determinants (A, B) |
|---|---|---|---|---|---|
| A | 0.05 < Re λ < 0.9, \|Im λ\| < ω − ε | none | 0 | 0 | 124, 124 |
| B | 0.9 < Re λ < 5, \|Im λ\| < ω − ε | λ = 1, λ = 2 | 2 | 2 | 132, 132 |
| S | 0.05 < Re λ < 5, ω − ε < Im λ < ω + ε | none | 0 | 0 | 68, 68 |
| C | 0.05 < Re λ < 5, ω + ε < Im λ < 3ω | none | 0 | 0 | 52, 52 |
| W, 24 starting samples per edge | 0.05 < Re λ < 8, \|Im λ\| < 3ω | λ = 1, λ = 2 | 4 | 2 | 480, 408 |

Controls, each seen to fire (setting A): a zero planted at 3 + 1i makes B count 3, and `--mutate=nometric` makes it
count 1. Box C counts 0 again with 24 starting samples per edge (192 determinants).

**W's two extra zeros at setting A are an artefact of the jump at r₁ = 0.3.** The strip 5 < Re λ < 8 of W counts 0 about
the real axis and 1 in 2.95 < Im λ < 3ω at setting A. `css_inner_pert.py locate` halves that box while it winds and
finds the zero at λ = 7.7410972542 + 3.7686870595i, at setting A and at C (r₀ 0.09, the same r₁), which agree to
1×10⁻¹³. At setting B (r₁ = 0.25) the secant from it does not converge, and its box 6.5 < Re λ < 8,
2.95 < Im λ < 4.61 counts 0. With its conjugate it accounts for W's 4 at A against 2 at B. `locate`'s control, a zero
planted at 2.3 + 6.1i in a box free of zeros, is found there exactly at settings A and B, where M itself is not
singular (smallest/largest singular value 1.4×10⁻⁶).

**So in 0.05 < Re λ < 8, |Im λ| < 3ω∞ the D = ∞ axion–dilaton CSS has λ = 1 (gauge) and λ = 2 (the relevant mode) and
no other growth rate: its unstable mode is unique there.** Not covered: growth rates with Re λ > 8, |Im λ| > 9.60, or
0 < Re λ < 0.05, and the selection rule itself (Conjecture 1), which this computation assumes.

**The count can miss whole turns.** With `count`'s default of 6 starting samples per edge, W counted 0 at both settings
(344 and 304 determinants), against 4 and 2 with 24. The winding routine bisects an interval until its phase step is
below 0.5 rad and its halves agree, and an interval over whose halves the phase turns by whole multiples of 2π passes
that test. The tiles A, B, S, C, with edges up to 6.15 long, agree with W at 24 samples, and #781's count at κ = 0 (from
steps of 0.25) agrees with §1.4's tiles, so the counts quoted on this page stand. #781's counter shows the same
blindness in its control: with one sample per edge it counts 0.

## 2. The change of stability κ_c(D) (#377)

### 2.1 Method

`css_kappa_crit.py path D …` starts from the κ = 0 background of `data/css_bg_k0.jsonl` and the κ = 0 roots of
matter/COMPLEX_CRITICAL.md §4. It steps in κ: the background by Newton on its horizon jump (`css_lateral.newton`), then
λ_osc and λ_phys by secant on `css_pert.py`'s det M(λ) at the circle radii R = 0.85 x_h and 0.6 x_h. A step is accepted
when |J| < 10⁻¹⁰ and each root converges at both radii to within 10⁻⁷ and near its extrapolation; otherwise the step is
halved. The path stops at the first κ where Re λ_osc < 0 at both radii. `crit` then refines the zero of Re λ_osc(κ)
by regula falsi (Illinois) in that bracket, at each radius separately, each evaluation a new background and a new root.
The crossing is accepted only as a sign change at both radii, since the jump method's artefacts move with R.

### 2.2 D = 4

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
| 0.07125 | +0.0172252 | ±3.6779905 | 3 |
| 0.0757329315 | +7×10⁻⁸ (zero within its error, about 10⁻⁷ at D = 4) | ±3.6888136 | 1 (and the marginal pair) |
| 0.076 | −0.0010341 | ±3.6894573 | 1 |

**κ_c(4) = 0.0757329315**, the same by both methods. At the bracket's ends the jump method and the collocation give
Re λ_osc = +0.0172251000 and +0.0172251644 at κ = 0.07125 (raw `I377-mac/path_D4.jsonl`, `scan_D4_lo.jsonl`), and
−0.0010341099 and −0.0010341192 at κ = 0.076 (the jump method's radii differ by 2×10⁻¹¹; raw `I377-mac/at_D4.jsonl`,
`scan_D4.jsonl`).

The model and its normalization are HE97's: the action is their eq. (4), and γ at three κ matches their Table 1,
γ = 0.3871055 (κ = 0; HE97 0.3871), 0.2641066 (κ = 1; 0.2641) and 0.1468992 (κ = 10, jump method at both radii and
collocation; 0.1469). The background continued in κ from 0 to 1 at D = 4 lands on HHS96's axion–dilaton solution,
ω = 1.1769527242 and |f(0)| = 0.8925554116 (matter/CSS.md §4). And the κ = 0 pair agrees with HE95b within their error
bars (matter/COMPLEX_CRITICAL.md §2).

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
the oscillatory pair is unstable widens with D, from κ < 0.0757 at D = 4 to κ < 0.110 at D = 40, and the
axion–dilaton (κ = 1) stays far on the stable side. For κ between 0.0757 and 0.1103 the pair is stable at D = 4 and
unstable at D = 40. γ at κ_c rises with D toward 1/2 as for every κ.

### 2.4 κ_c at D = ∞

`css_inner_pert.py kpath 0 1 0.05` follows λ_osc of the inner solution in κ at settings A and B, each inner background
re-solved by `css_inner.py`, and `kcrit` refines the crossing at each setting (raw `I377-mac/kpath_inf.jsonl`,
`kcrit_inf.jsonl`):

| κ | ω∞(κ) | λ_osc(∞) | Im λ_osc / ω∞ | setting spread |
|---|---|---|---|---|
| 0 | 3.9789695 | 0.5365223 + 7.9320242i | 1.99349 | 1×10⁻¹⁰ |
| 0.05 | 3.9256528 | 0.2972904 + 7.8553060i | 2.00102 | 1×10⁻¹⁰ |
| 0.1 | 3.8739534 | 0.0653125 + 7.7716617i | 2.00613 | 9×10⁻¹¹ |
| **0.1143406** | 3.8594315 | ±7.7465455i | 2.00717 | κ_c: 9×10⁻¹² |
| 0.15 | 3.8239189 | −0.1602438 + 7.6821182i | 2.00896 | 9×10⁻¹¹ |
| 0.2 | 3.7755639 | −0.3802052 + 7.5874227i | 2.00961 | 6×10⁻¹¹ |
| 0.4 | 3.5985546 | −1.2210364 + 7.1648785i | 1.99104 | 1×10⁻¹⁰ |
| 0.6 | 3.4457917 | −2.0480255 + 6.6713159i | 1.93608 | 5×10⁻¹⁰ |
| 0.9 | 3.2549120 | −3.6391186 + 5.6869460i | 1.74719 | 1×10⁻⁸ |

**κ_c(∞) = 0.1143405667** (settings A and B agree to 9×10⁻¹²), with γ∞ = 1/2 there as at every κ. Beyond κ = 0.9 the
path is not resolved: at κ = 0.95 setting A's secant did not converge, and at κ = 1 the two settings end 12 apart, so
λ_osc(∞) of the axion–dilaton is not given here (only that it is damped, Re λ_osc < −3.6 at κ = 0.9).

The finite-D values approach κ_c(∞) like an ordinary 1/n series: n(κ_c(∞) − κ_c(D)) = 0.077, 0.099, 0.120, 0.140,
0.153 at D = 4, 6, 10, 20, 40. Fits of κ_c(D) − κ_c(∞) in powers 1/n … 1/n^K have rms residuals 2 to 4 times smaller
than free fits with the same number of parameters (7.1×10⁻⁴ against 1.9×10⁻³ with two parameters on D ≥ 4, 2.5×10⁻⁵
against 8.1×10⁻⁵ with three on D ≥ 6), with the 1/n coefficient −0.13 to −0.16. The free fits of degree 2–3 put the
limit at 0.1129–0.1142, just below the inner value, as for λ_osc in §1.3. Im λ_osc/ω at the crossing runs 1.98720,
1.99400, 1.99988, 2.00410, 2.00588 at D = 4–40 toward the inner 2.00717.

## 3. Discrepancies

- **Im λ_osc/ω∞ at κ = 0** lies 6.2×10⁻⁷ below the spread of the six 1/n fits of the finite-D track (§1.3), so the
  acceptance of #376 holds for Re λ_osc and fails narrowly for Im/ω. The fits fall toward the inner value as D_min and
  K grow, and the finite-D values converge to it like 1/n, so this reads as the spread over fit choices understating
  the extrapolation error, as for ω∞ in matter/CSS.md §7.

## 4. Proposed text for matter/CSS.md, matter/COMPLEX_CRITICAL.md, matter/STRUCTURE.md and the README

For matter/CSS.md §11, replacing the bullet "**The λ = 2 mode of the CSS inner solution was not constructed
explicitly.**":

> - **The λ = 2 mode of the CSS inner solution** is constructed (the s-translate plus the gauge term with constant P)
>   and found as a zero of the inner perturbation determinant, to 1.4×10⁻¹² at κ = 0, with the other exact modes
>   λ = 0, 1, ±iω ([CSS_STABILITY.md](CSS_STABILITY.md) §1). It inherits the status of the selection rule
>   (Conjecture 1).

For matter/CSS.md §11, the bullet "**Other κ.**", one sentence after its first: "The CSS's oscillatory pair crosses
to Re λ < 0 at κ_c(D) = 0.0757, 0.0895, 0.0993, 0.1066, 0.1103 at D = 4, 6, 10, 20, 40 and 0.1143 at D = ∞
([CSS_STABILITY.md](CSS_STABILITY.md) §2), so the CSS can be critical only for κ > κ_c(D)."

For matter/COMPLEX_CRITICAL.md §4, "Status", replacing "Open: the 1/n fits give Im λ_osc / ω → 1.9935 with no visible
spread, not 2; whether the limit is 2 with a slower correction (a log, or a non-analytic term) is not decided by
D ≤ 80.": "The D = ∞ inner problem has the pair at λ_osc = 0.5365223 ± 7.9320242i, Im λ_osc/ω∞ = 1.99349, and no other
growth rate besides λ = 1, 2 in its boxes ([CSS_STABILITY.md](CSS_STABILITY.md) §1): the CSS is not critical at D = ∞
either, and the limit of Im λ_osc/ω is not 2." The proposed text of its §6 for matter/CSS.md's κ = 0 bullet can end
with the same clause.

For matter/STRUCTURE.md §7, replacing "while it is for κ ≳ 0.075)": "while it can be for κ > κ_c(D), where its
oscillatory pair is stable: κ_c = 0.0757 at D = 4, rising to 0.114 at D = ∞ (matter/CSS_STABILITY.md))".

For the README, in "What is known about the selection, and what is open", one sentence: "For the Hirschmann–Eardley
κ-model the phase-rotating CSS solution can be critical only above a coupling κ_c(D), where its oscillatory pair of
modes turns stable [^HE97]: κ_c = 0.0757 at D = 4, rising to 0.110 at D = 40 and to 0.114 at D = ∞, where the free
complex scalar's CSS keeps the pair, λ = 0.537 ± 7.932i (matter/CSS_STABILITY.md)." Machine-readable sources for
`check_numbers.py`: raw `I377-mac/crit.jsonl` and `kcrit_inf.jsonl`, `I376-mac/roots_k0.jsonl`.

## 5. Proposed notes/LIT_REVIEW.md entries

- **[HE97] E. W. Hirschmann, D. M. Eardley, "Criticality and bifurcation in the gravitational collapse of a self-coupled
  scalar field", PRD 56, 4696–4705 (1997), arXiv:gr-qc/9511052** (read 2026-11-14: the arXiv TeX source, whole; the
  record checked on INSPIRE and Crossref). Contains: the κ-model action (their eq. 4, the one of matter/CSS.md §1), its
  D = 4 CSS solutions and the linear perturbations, with det A(σ) counted by the argument principle around contours in
  the lower half σ plane and its zeros found by secant; the four gauge modes σ = 0, −i, ±ω (λ = 0, 1, ±iω here); the
  statement that "additional modes … approach the real axis as κ decreases" and that the first crossing "occurs at
  κ ≈ 0.0754" ("Hopf-type bifurcation"); Table 1: γ = 0.1469 (κ = 10), 0.2641 (1), "≳ 0.373" at "≲ 0.0754", 0.3871 (0),
  ω_BD = −3/2 + 1/(8κ); more unstable modes by κ ≈ −0.28. Does not contain: the oscillatory mode's σ at the crossing or
  for any κ > 0, any dimension other than 4.
- **[L98] S. L. Liebling, "Multiply unstable black hole critical solutions", PRD 58, 084015 (1998),
  arXiv:gr-qc/9805043** (abstract only, 2026-11-14): evolutions of the same model in which a family of data finds the
  CSS as its critical solution in the region where the DSS is the attractor.

## 6. Scripts, data and how to reproduce

All in `solver/matter/`, run single-process under `nice -n 10` on the Mac (Apple M5); times for one process there, with
the machine shared. Raw output: [choptuik-large-d-raw](https://github.com/nmoinvaz/choptuik-large-d-raw), folders
`I376-mac/`, `I377-mac/` and `I770-mac/` (the queue scripts that ran are in them, with their logs).

| script | what it does | cost |
|---|---|---|
| `css_inner_pert.py check / det / root / count / kpath / kcrit / locate` | the D = ∞ inner CSS perturbations: series, lateral solutions, det M(λ), secant roots at settings A, B, C, winding counts (`--plant`, `--mutate=nometric\|hamsign`, `--K` starting samples per edge), the root followed in κ and κ_c(∞), the zeros in a box refined at every setting | 1.5 s per determinant (κ = 0, 5.5 s to build a background); 1 min per root at three settings; 2.5–5 min per box |
| `css_inner_pert_eqs.py [--mutate=…]` | sympy: the linearized rhs is the ε-derivative of the inner PDE in (s, t) | 20 s |
| `css_kappa_crit.py path / crit / bgpath / selftest` | finite D: the background and λ_osc, λ_phys followed in κ at two radii; κ_c refined at each radius; the background alone along κ; the crossing finder on a planted function | 10–60 s per κ step; 1–4 min per κ_c |

To reproduce from the repository root, with the venv of SETUP.md:

```
python3 solver/matter/css_inner_pert_eqs.py                                  # worst residual 2e-15, exit 0
python3 solver/matter/css_inner_pert.py check 0                              # check PASS
python3 solver/matter/css_inner_pert.py root 0 2.1 --set=A                   # lambda = 2 - 1.4e-12
python3 solver/matter/css_inner_pert.py root 0 0.536+7.93j --set=A           # 0.5365223424 + 7.9320241697i
python3 solver/matter/css_inner_pert.py count 0 0.05 5 4.2289695 9.432 --set=A   # winding 1
python3 solver/matter/css_inner_pert.py count 1 0.9 5 -2.9498939 2.9498939 --set=A   # winding 2
python3 solver/matter/css_inner_pert.py count 1 0.05 8 -9.5996817 9.5996817 --set=B --K=24   # winding 2
python3 solver/matter/css_inner_pert.py locate 1 5 8 2.9498939 6.2747878 --set=A,B,C --w=1.7
                                                      # 7.7410972542 + 3.7686870595i at A and C, not converged at B
python3 solver/matter/css_kappa_crit.py selftest
python3 solver/matter/css_kappa_crit.py path 4 0.2586698339+3.5022487970j 2.5832751746 --k1=1 --out=/tmp/p4.jsonl
python3 solver/matter/css_kappa_crit.py crit /tmp/p4.jsonl 4                 # kappa_c 0.0757329315 at both radii
python3 solver/matter/css_scan.py 0.07125 4 --guess=1.856291661723,4.797192767370,1.231464079551 --N 40,56
                                                                             # Re lambda_osc = +0.0172251644
```

## 7. Recommended tasks

1. **Uniqueness of the axion–dilaton CSS's unstable mode at D = ∞** (done, #770, §1.6). What:
   `css_inner_pert.py count 1 …` in the boxes A, B, C, S with ω∞(κ = 1) = 3.19989, and a wider box (Re λ < 8, |Im λ| <
   3ω∞), at settings A and B. Why: matter/CSS.md §11 lists uniqueness at D = ∞ as open, and this code makes it a count
   of a few hours. Acceptance (can fail): B counts exactly 2 (λ = 1, 2) at both settings; every other box counts 0, or
   its zeros are located and their Re λ reported; a planted zero adds one and `--mutate=nometric` changes B. Depends on:
   #376.
2. **A second implementation of the D = ∞ perturbation determinant** (done, #771, §1.5). What: the inner perturbation
   problem in matter/STRUCTURE.md §7's variables (y, σ, c), as in `struct_css.py`, or a Chebyshev discretization of the
   inner ODE on a ray, sharing no code with `css_inner_pert.py`. Why: AGENTS.md asks for independent checks; the present
   cross-check against finite D is an extrapolation. Acceptance (can fail): λ = 2 found to 10⁻¹⁰, λ_osc(∞, κ = 0) and
   κ_c(∞) reproduced to 10⁻⁸. Depends on: #376.
3. **λ_osc at D = ∞ for κ ≥ 0.95.** What: `css_inner_pert.py kpath 0.9 1 0.01` at settings A, B and C, and a box
   count around the result at κ = 1. Why: the path of §2.4 is unresolved there (the settings end 12 apart at κ = 1), so
   the axion–dilaton's damped pair at D = ∞ is not known. Acceptance (can fail): the settings agree to 10⁻⁸ at every
   κ, and the box count around λ_osc(κ = 1) is 1 at two settings. Depends on: #377.

## References

- [HE95b] E. W. Hirschmann, D. M. Eardley, PRD 52, 5850 (1995), gr-qc/9506078.
- [HE97] E. W. Hirschmann, D. M. Eardley, "Criticality and bifurcation in the gravitational collapse of a self-coupled
  scalar field", Phys. Rev. D 56, 4696–4705 (1997), doi:10.1103/PhysRevD.56.4696, arXiv:gr-qc/9511052 (verified
  2026-11-14 against INSPIRE's record and Crossref's for the DOI: title, authors, volume, issue 8, pages 4696–4705,
  published 1997-10-15).

# Mode count: how many unstable modes does the DSS solution have?

**Status: work in progress** (counts at D = 61 and 45 done; D = 30, the M = 16 resolution checks, the f64 count at
D = 4, and D = 16.8 and 8.25 running). This file is completed when they are in.

The referee objection (notes/REFEREE_SIM.md, row 2): the growth rate λ is a local root of det M(λ) near its predicted
value (`hp/gamma_dd.py`, `find_lam`), so a second unstable mode would have gone unnoticed, and then 1/λ would not be the
critical exponent. Here the zeros of det M(λ) are counted by the argument principle in boxes of the right half plane
that cover 0 < Re λ < 6 and both symmetry sectors of the perturbations.

## 1. What det M(λ) is, and which of its zeros and poles are known in advance

The growth rate comes from the growth-mode matrix M(λ) of `hp/` (README, "Exact Jacobian and growth-mode matrix"):
the exact Jacobian of the multiple-shooting residual for perturbations $`\propto e^{\lambda\tau}`$, periodic in τ, with Δ
held fixed and the phase row dropped. Its unknowns are the centre data, the node states and the horizon data on the
half-period grid of M points in τ, with p, m odd and a, c even under τ → τ + Δ/2 (the "folded" layout). det M(λ) = 0
exactly when the linearised equations have a solution $`e^{\lambda\tau}`$ × (periodic, regular at the centre and at
the past light cone x = 1, with c(τ, 1) = 1). So its zeros with Re λ > 0 are the unstable modes of this formulation.
Not all of them are physical, and det M also has poles; both have to be known before a count means anything.

**Both symmetry sectors are in one period.** Multiplying a perturbation by $`e^{i\omega\tau}`$ (ω = 2π/Δ) swaps its
parity under the half-period shift, so a perturbation of the other sector (p, m even, a, c odd) with growth rate λ is a
folded perturbation with growth rate λ + iω. Multiplying by $`e^{2i\omega\tau}`$ keeps the parity, so the zeros of the
folded det M repeat with period 2iω (exactly in the continuum, up to the highest harmonics in the discretisation). The
strip −ω/2 < Im λ < 3ω/2 therefore holds every growth rate of both sectors exactly once: real growth rates of the
symmetric sector on Im λ = 0, those of the other sector on Im λ = ω, complex ones anywhere. The boxes below cover
|Im λ| < 3ω/2, which is that strip together with the mirror image of its upper part.

**λ = 0** is the translation in τ (∂τ of the background solves the λ = 0 equations). It is marginal, not unstable;
the strip B0 below contains it.

**λ_g = 1 + ⟨c₁⟩ is a residual gauge mode.** The usual gauge mode of critical collapse, λ = 1 (a shift of the
accumulation time t*), is not a zero here: the formulation pins the past light cone of the singularity to x = 1 by
c(τ, 1) = 1, and a shift of t* moves the light cone. What remains is a reparametrisation of time that keeps c = 1 on
x = 1. In polar–areal coordinates t → t + ε h(t) leaves r and the fields p, m, a unchanged as functions on spacetime
and rescales the lapse, α → α(1 − ε h′). With $`h = e^{(\lambda-1)\tau}P(\tau)`$ (τ = −ln(−t), x = r/(−t)) the change at
fixed (τ, x) is

```math
\delta f = -\varepsilon\, e^{\lambda\tau} P\,(\partial_\tau + x\partial_x) f \quad (f = p, m, a),\qquad
\delta c = -\varepsilon\, e^{\lambda\tau}\Big[P\,(\partial_\tau + x\partial_x) c + \big((\lambda-1)P + P'\big)\,c\Big].
```

At x = 1 one has c = 1, ∂τc = 0 and, from $`x c_x = (n-1)(a^2-1)c`$, $`\partial_x c = -c_1(\tau)`$ with
$`c_1 = (n-1)(1-a_h^2)`$, a_h = a(τ, 1). So δc(τ, 1) = 0 is $`P' = (1 + c_1 - \lambda)P`$. The perturbation lies in the
folded layout when P is Δ/2-periodic, which happens exactly for

```math
\lambda = \lambda_g + 2ik\omega,\qquad \lambda_g = 1 + \langle c_1\rangle = 1 - (n-1)\,\big\langle a_h^2 - 1\big\rangle ,
```

with ⟨·⟩ the average over τ and k any integer. (With P antiperiodic one gets λ_g + iω in the other sector, but
$`e^{(\lambda_g + i\omega)\tau}P_{\rm anti} = e^{\lambda_g\tau}P_{\rm per}`$: it is the same perturbation, so the other sector
adds no gauge zero; the counts below confirm this.) The perturbation is a change of coordinates, regular at the centre
and at x = 1, so it is a zero of det M that changes nothing physical. The real zero of det M sits at 1 + ⟨c₁⟩ to
1.1×10⁻¹⁶ at D = 30.07 (no freeze) and to 4.9×10⁻¹¹ at D = 61.03 (kfreeze 13) (`hp/modecount_check.py`,
`hp/data/modecount_check.jsonl`). λ_g runs from 0.66 at D = 8.25 to 0.90 at D = 61. It is the real sign change of det M
between 0.5 and 1 that a scan of the real axis finds besides λ_phys.

**Poles.** λ enters M through τ-derivatives, polynomially in the centre series and the Taylor steps, and through the
horizon series, whose order-kk tangent solves $`(\omega D + \mathrm{diag}\,\kappa_{kk} + \lambda)\,\delta p_{kk} = r_{kk}`$ with
$`\kappa_{kk} = (1+c_1)(kk+1) - n/2`$ (`hp/src/model.rs`, `horizon_lus` with `shift` = λ). So det M is meromorphic, and
its only possible poles are the λ where one of these M × M matrices is singular, kk = 0, …, ks (ks = 100):

```math
\lambda_{kk,j} = \frac{n}{2} - \lambda_g\,(kk+1) - i(2j+1)\,\omega
```

(the code uses the matrices' eigenvalues themselves). These are resonances of the horizon expansion: the exponent of
a perturbation's singular branch at the light cone, $`\sigma = (n/2 - \lambda)/\lambda_g - 1`$, is an integer there. They
lie on Im λ = ±ω, ±3ω, …, spaced by λ_g in Re λ. Each is at most a simple pole of det M: the eigenvalues are simple, so
near one of them one horizon coefficient, and with it the horizon block of M, has a rank-one singular part. At D = 61
there are 7 of them in 0.3 < Re λ < 6.6 on each of the lines Im = ±ω.

## 2. Method

**det M at complex λ.** `Solver.mode_det(zh, zl, lam_hi, lam_lo)` of `hp/` takes λ as a real double-double: its
fourth argument is the low part of λ, not Im λ, and complex λ is not supported. `hp/modecount_cx/` is a separate crate
(built by `hp/modecount_build.sh` into its own build directory; hpkernel is not modified or reinstalled) that links the
hp kernel as a library and repeats the λ-dependent parts of `hp/src/model.rs` with a complex tangent: a lane type with
a real value and a complex tangent (the background is real, so every dual-number rule acts on the real and imaginary
parts separately), d/dτ → ωD + λ on the tangent, the horizon solves (A + λ) as the real 2M × 2M system
[[A + Re λ, −Im λ], [Im λ, A + Re λ]], and the same kfreeze-13 rows. ln|det M| and arg det M come from a complex
double-double LU with partial pivoting. Checks (`hp/modecount_check.py`, `hp/data/modecount_check.jsonl`):

| check | D = 30.07 (M 32, no freeze) | D = 61.03 (M 32, kfreeze 13) |
|---|---|---|
| M(1.7), entry by entry, against hpkernel's `mode_matrix` | identical (largest difference 0) | identical (largest difference 0) |
| ln\|det M\| at real λ against hpkernel's `mode_det` | equal to all printed digits at λ = 0.5, 1.7, 4 | 4×10⁻¹⁰ apart at λ = 1.7: the two double-double LUs round differently and max \|M\| = 2×10⁹ |
| sign of det M at real λ | arg det = 0 or π, matching | matching |
| analyticity: d ln det M / dλ along h and along ih (h = 10⁻⁵) | agree to 9×10⁻⁷ and 3×10⁻⁷ at λ = 1.3 and 3.0, i.e. to O(h) | not run |
| real zero near 1 + ⟨c₁⟩ | 1.1×10⁻¹⁶ from it | 4.9×10⁻¹¹ from it |
| complex LU against numpy's `slogdet` (random matrices, n = 7–129) | 8×10⁻¹³ | |

**The f64 solver at D = 4.** The double-double states start at D = 8.25. At D = 4 the f64 solver's state
`solver/data/neB_kc39_D4.0000.npy` (kc 39, nodes 0.02–0.6, horizon series Ks = 60) is used with the Python reference
(`solver/`, `DSSSpec`: Fourier in τ, RK4 in x; the same equations, gauge and boundary conditions). Its mode matrix
(`DSSSpec.mode_matrix`: central differences with ε = 10⁻⁶, Δ fixed, no phase row) is evaluated with complex λ
throughout (`hp/modecount_f64.py`: `solver/fast.py`'s matrix forms of d/dτ, the truncation and the periodic solves are
complex-safe, and `phys`/`spec` are applied to the real and imaginary parts), and det M from numpy's complex LU. It is
checked at real λ against the Rust f64 kernel's exact (dual-number) mode matrix. Its periodic solves run on the full τ
grid, so besides the resonances above there are full-grid resonances with Im λ an even multiple of ω, which the odd
right-hand sides do not excite; they are kept in the candidate list, which keeps the bound below valid.

**The function whose phase is followed.** On a closed contour C the winding number of det M is (zeros − poles)
inside C. Rather than det M, the phase of

```math
g(\lambda) = \det M(\lambda)\;\frac{\prod_j (\lambda - p_j)}{\prod_k (\lambda - z_k)}\;e^{-P_2(\lambda)}
```

is followed, with p_j all candidate poles with |Im λ| < 4.5ω (every kk = 0, …, ks), z_k the expected zeros 0, λ_g and
λ_phys with their copies 2ikω, and P₂ a real quadratic fitted to ln|det M ∏(λ − p_j)/∏(λ − z_k)| at the 15–17 sample
points on the real axis. The three factors are known exactly and the exponential has no zeros, so for every box

```math
\mathrm{wind}_C(\det M) = \mathrm{wind}_C(g) + N_{\rm exp}(C) - N_{\rm cand}(C),\qquad
Z(C) \le \mathrm{wind}_C(g) + N_{\rm exp}(C),
```

where Z(C) is the number of zeros of det M inside C, N_exp and N_cand the numbers of expected zeros and candidate poles
inside. The inequality holds because every pole of det M is a candidate and simple (equality when every candidate
inside is an actual pole). If det M has no zeros besides the expected ones and every candidate is a pole, wind(g) = 0
for every box. A further zero shows as wind(g) ≥ 1. A candidate that is not a pole shows as wind(g) ≥ 1 too, so it can
only make the bound weaker, never hide a zero. det M itself grows like $`e^{33\lambda}`$ (D = 61) to $`e^{57\lambda}`$ (D = 30).
Dividing that and the known zeros and poles out makes g nearly constant (|ln g| < 1.5 on every edge at D = 61), which
is what keeps the cost near 100 determinants per state; it does not decide the count.

**Contour network** (`hp/modecount_wind.py`): vertical edges at Re λ = aLL ≈ −0.1, aL = 0.3 and aR ≈ 6–6.6 from
Im λ = 0 to y2 = 3ω/2, and at a1 (between λ_g and λ_phys) and a2 = λ_phys + 0.6 from Im λ = 0 to y1 = ω/2; horizontal
edges at Im λ = y1 and y2 from aR to aLL. aLL, aL, a1 and aR are chosen within their ranges as far as possible from the
real parts of the candidate poles and expected zeros. det M is real on the real axis and $`\det M(\bar\lambda) = \overline{\det M(\lambda)}`$,
so a box symmetric about the real axis has winding (1/π) × (phase change along its upper half); the strips
y1 < Im λ < y2 are closed contours of their own (their mirror images hold the conjugate zeros).

| box | region | zeros expected | wind(g) expected |
|---|---|---|---|
| B0 | aLL < Re < aL, \|Im\| < ω/2 | λ = 0 | 0 |
| U0 | aLL < Re < aL, ω/2 < Im < 3ω/2 | none | 0 |
| B1a | aL < Re < a1, \|Im\| < ω/2 | λ_g | 0 |
| B1b | a1 < Re < a2, \|Im\| < ω/2 | λ_phys | 0 |
| B1c | a2 < Re < aR, \|Im\| < ω/2 | none | 0 |
| B1 | aL < Re < aR, \|Im\| < ω/2 | λ_g, λ_phys | 0 |
| U | aL < Re < aR, ω/2 < Im < 3ω/2 | none | 0 |
| full | aL < Re < aR, \|Im\| < 3ω/2 | λ_g, λ_phys | 0 |

B1b (λ_phys alone), B1c (no zero) and B1a (λ_g alone) are the deliberate checks of item 4: the windings of det M must
come out 1, 0 and 1. U is a stronger one: its winding of det M must come out as minus the number of candidate poles in
it (−7 at D = 61). B0 and U0 together cover aLL < Re λ < 0.3, so slowly growing modes, 0 < Re λ < 0.3, are counted too.

**Sampling.** Each edge starts on a grid of spacing 0.4 and every interval is bisected until the phase change of g
across it is at most π/4. Then every interval whose phase change exceeds π/8 is bisected once more and the two halves
are checked to add up to the whole (a hidden turn of 2π would show as a mismatch; there were none). Note that the sum of
wrapped phase steps around a closed contour is always a multiple of 2π, so windings coming out as integers prove
nothing; what makes them reliable is that every step is well below π. The records give, per edge and box, the largest
phase step of g, the largest phase step of det M itself between the same samples (up to 3 rad, because of its
exponential growth, which is removed exactly and is therefore harmless), the range of ln|g| on the boundary (how close
the boundary comes to an unaccounted zero or pole of g), and the distances from the boundary to the nearest candidate
pole and expected zero.

## 3. Results so far

Windings: wind(g) [wind(det M) = wind(g) + expected - candidate poles]; zeros of det M <= wind(g) + expected.

| D | solver | λ_g | λ_phys | B0 | U0 | B1a | B1b | B1c | B1 | U | full |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 45.0507 | dd, M 32, kf 13 | 0.871366 | 2.056739 | -0.000 [+1] | +0.000 [+0] | +0.000 [+1] | +0.000 [+1] | +0.000 [+0] | +0.000 [+2] | -0.000 [-7] | +0.000 [-12] |
| 61.0264 | dd, M 32, kf 13 | 0.899231 | 2.043533 | +0.000 [+1] | -0.000 [+0] | +0.000 [+1] | +0.000 [+1] | +0.000 [+0] | +0.000 [+2] | -1.000 [-7] | -2.000 [-12] |

| D | solver | expected zeros / candidate poles inside, per box | zeros of det M in the full box (bound) |
|---|---|---|---|
| 45.0507 | dd, M 32, kf 13 | B0 1/0, U0 0/0, B1a 1/0, B1b 1/0, B1c 0/0, B1 2/0, U 0/7, full 2/14 | ≤ 2 (+ B0: ≤ 1, U0: ≤ 0) |
| 61.0264 | dd, M 32, kf 13 | B0 1/0, U0 0/0, B1a 1/0, B1b 1/0, B1c 0/0, B1 2/0, U 1/7, full 4/14 | ≤ 2 (+ B0: ≤ 1, U0: ≤ 0) |

| D | solver | aLL, aL, a1, a2, aR | y1, y2 | evaluations | s / eval | max phase step of g (rad) | max step of det M (rad) | ln\|g\| on the boundary | nearest candidate pole / expected zero to an edge | real-axis sign changes |
|---|---|---|---|---|---|---|---|---|---|---|
| 45.0507 | dd, M 32, kf 13 | -0.130, 0.300, 1.221, 2.657, 6.277 | 1.687, 5.061 | 104 | 100 | 0.379 | 3.08 | -1.408 … 0.733 | 0.129 / 0.130 | (0.50, 1.00), (2.00, 2.50) |
| 61.0264 | dd, M 32, kf 13 | -0.080, 0.300, 1.269, 2.644, 6.585 | 1.738, 5.215 | 125 | 39 | 0.392 | 3.02 | -1.294 … 0.080 | 0.081 / 0.080 | (0.50, 1.00), (2.00, 2.50) |

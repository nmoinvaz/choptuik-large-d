# Mode count: how many unstable modes does the DSS solution have?

The referee objection (notes/REFEREE_SIM.md, row 2): the growth rate λ is a local root of det M(λ) near its predicted
value (`hp/gamma_dd.py`, `find_lam`), so a second unstable mode would have gone unnoticed, and then 1/λ would not be
the critical exponent. Here the zeros of det M(λ) are counted by the argument principle in boxes of the complex λ plane
that cover at least −0.08 < Re λ < 6.18 and both symmetry sectors of the perturbations.

**Verdict.** At D = 30.07, 45.05 and 61.03 (double-double states at the production resolution, M = 32 points in τ),
det M has exactly three zeros in −0.08 < Re λ < 6.18 (the boxes reach −0.15 and 6.6 at some D), |Im λ| < 3ω/2:

- λ = 0, the translation in τ (marginal);
- λ_g = 1 + ⟨c₁⟩ (0.826, 0.871, 0.899), a residual gauge mode of this formulation: a change of the time coordinate
  that changes nothing physical (section 1);
- λ_phys (2.0797, 2.0567, 2.0435), the known growth rate.

So **at these D the solution has one physical unstable mode among the perturbations the layout resolves**, and
γ = 1/λ_phys. The deliberate checks behave as they should. The box holding λ_phys alone counts 1, the box next to it
counts 0, and the gauge box counts 1. The box holding seven poles of det M winds −7, and zeros planted in a synthetic
det M are all found.

One item is open. At a second resolution, M = 48 at D = 30, the count is the same except for an **extra conjugate
pair at 4.267 ± 1.324i**. The pair is absent at M = 32, and absent at M = 48 when the centre data's τ-harmonics above
13 are pinned (the cut the production runs use from D = 35 on). Without the cut, the M = 48 background converges only
to |F| = 9×10⁻¹⁸. Everything points to an artifact of the solver's weakly coupled high-harmonic directions, but a third
resolution is needed to close it (section 5). At D = 4 (f64 solver, real λ only) det M changes sign only at
λ_g = 0.6139 and λ_phys = 2.6741 on 0.02 < λ < 6.47.

What is not covered: Re λ > 6.6, perturbations whose τ-harmonics lie above what the layout holds (above 31, and for the
centre data above 13 from D = 35 on), complex λ at D < 30, and resolutions other than M = 32 except at D = 30. The
runs proposed for the local machine are in section 7.

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
strip −ω/2 < Im λ < 3ω/2 therefore holds every growth rate of both sectors exactly once (among the perturbations the
layout resolves, section 6): real growth rates of the
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
$`e^{(\lambda_g + i\omega)\tau}P_{\rm anti} = e^{\lambda_g\tau}P_{\rm per}`$: it is the same perturbation, so the
other sector adds no gauge zero; the counts below confirm this.) The perturbation is a change of coordinates, regular
at the centre and at x = 1, so it is a zero of det M that changes nothing physical. The real zero of det M sits
at 1 + ⟨c₁⟩ to 1.1×10⁻¹⁶ at D = 30.07 (no freeze) and to 4.9×10⁻¹¹ at D = 61.03 (kfreeze 13) (`hp/modecount_check.py`,
`hp/data/modecount_check.jsonl`). λ_g runs from 0.61 at D = 4 to 0.90 at D = 61. It is the real sign change of det M
between 0.5 and 1 that a scan of the real axis finds besides λ_phys.

**Poles.** λ enters M through τ-derivatives, polynomially in the centre series and the Taylor steps, and through the
horizon series, whose order-kk tangent solves
$`(\omega D + \mathrm{diag}\,\kappa_{kk} + \lambda)\,\delta p_{kk} = r_{kk}`$ with
$`\kappa_{kk} = (1+c_1)(kk+1) - n/2`$ (`hp/src/model.rs`, `horizon_lus` with `shift` = λ). So det M is meromorphic,
and its only possible poles are the λ where one of these M × M matrices is singular, kk = 0, …, ks (ks = 100):

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

**D = 4 with the f64 solver, real λ only.** The double-double states start at D = 8.25. At D = 4 the f64 solver's
state `solver/data/neB_kc39_D4.0000.npy` (kc 39, nodes 0.02–0.6, horizon series Ks = 60) is used with the Rust f64
kernel's exact (dual-number) mode matrix, which takes real λ only (`hp/modecount_f64.py --realscan`): the sign of
det M on a grid of step 0.05 over 0.02 < λ < 6.47, each sign change refined by brentq. This covers the symmetric
sector's real growth rates only. (A complex-λ version through the Python reference's central-difference mode matrix,
`hp/modecount_f64.py` without `--realscan`, failed its real-λ check against the Rust matrix, max |ΔM|/max |M| = 0.89 at
λ = 0.3, cause not identified, and took 430 s per matrix; it is not used.) The f64 code solves the horizon series'
periodic problems on the full τ grid, so besides the resonances above it has full-grid resonances with Im λ an even
multiple of ω, which odd right-hand sides do not excite; one of them, at λ = 0.386, lies on the scanned interval.

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

where Z(C) is the number of zeros of det M inside C, N_exp and N_cand the numbers of expected zeros and candidate
poles inside. The inequality holds because every pole of det M is a candidate and simple (equality when every
candidate inside is an actual pole). If det M has no zeros besides the expected ones and every candidate is a pole,
wind(g) = 0 for every box. A further zero shows as wind(g) ≥ 1. A candidate that is not a pole shows as wind(g) ≥ 1
too, so it can only make the bound weaker, never hide a zero. Along the real axis det M grows like $`e^{33\lambda}`$
(D = 45, 61) to $`e^{57\lambda}`$ (D = 30). Dividing that and the known zeros and poles out makes g nearly constant
(|ln g| < 1.5 on every edge at D = 61), which is what keeps the cost near 100 determinants per state; it does not
decide the count.

**Contour network** (`hp/modecount_wind.py`): vertical edges at Re λ = aLL ≈ −0.1, aL = 0.3 and aR ≈ 6–6.6 from
Im λ = 0 to y2 = 3ω/2, and at a1 (between λ_g and λ_phys) and a2 = λ_phys + 0.6 from Im λ = 0 to y1 = ω/2; horizontal edges
at Im λ = y1 and y2 from aR to aLL. aLL, aL, a1 and aR are chosen within their ranges as far as possible from the real
parts of the candidate poles and expected zeros. det M is real on the real axis and
$`\det M(\bar\lambda) = \overline{\det M(\lambda)}`$, so a box symmetric about the real axis has winding (1/π) ×
(phase change along its upper half); the strips y1 < Im λ < y2 are closed contours of their own (their mirror images
hold the conjugate zeros).

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
it (−7 at D = 30, 45 and 61). B0 and U0 together cover aLL < Re λ < 0.3, so slowly growing modes, 0 < Re λ < 0.3, are
counted too.

**Sampling.** Each edge starts on a grid of spacing 0.4 (0.6 in the M = 48 counts) and every interval is bisected
until the phase change of g across it is at most π/4. Then every interval whose phase change exceeds π/8 is bisected
once more and the two halves are checked to add up to the whole (a hidden turn of 2π would show as a mismatch; there
were none). Note that the sum of wrapped phase steps around a closed contour is always a multiple of 2π, so windings
coming out as integers prove nothing; what makes them reliable is that every step is well below π. The records give,
per edge and box, the largest phase step of g, the largest phase step of det M itself between the same samples (up to
3 rad, because of its exponential growth, which is removed exactly and is therefore harmless), the range of ln|g| on
the boundary (how close the boundary comes to an unaccounted zero or pole of g), and the distances from the boundary
to the nearest candidate pole and expected zero.

## 3. Counts

Per state: wind(g) for every box (expected 0), and in brackets the winding of det M itself, wind(g) + (expected zeros
inside) − (candidate poles inside). Every production-resolution count has wind(g) = 0 in every box; the M = 48 count
at D = 30 (no freeze) is the one exception, in B1c (section 5).

| D | layout | ω | λ_g = 1 + ⟨c₁⟩ | λ_phys | B0 | U0 | B1a | B1b | B1c | B1 | U | full |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30.0688 | dd, M 32 | 3.198742 | 0.826484 | 2.079748 | 0 [+1] | 0 [−1] | 0 [+1] | 0 [+1] | 0 [0] | 0 [+2] | 0 [−7] | 0 [−12] |
| 45.0507 | dd, M 32, kfreeze 13 | 3.373741 | 0.871366 | 2.056739 | 0 [+1] | 0 [0] | 0 [+1] | 0 [+1] | 0 [0] | 0 [+2] | 0 [−7] | 0 [−12] |
| 61.0264 | dd, M 32, kfreeze 13 | 3.476755 | 0.899231 | 2.043533 | 0 [+1] | 0 [0] | 0 [+1] | 0 [+1] | 0 [0] | 0 [+2] | 0 [−7] | 0 [−12] |
| 30.0688 | dd, M 48, no freeze (\|F\| 9×10⁻¹⁸) | 3.198742 | 0.826484 | 2.079748 | 0 [+1] | 0 [−1] | 0 [+1] | 0 [+1] | **2 [+2]** | 2 [+4] | 0 [−7] | 2 [−10] |

What the M = 32 windings say, box by box (the same at all three D):

- **B0** (aLL < Re λ < 0.3, |Im λ| < ω/2): one zero, λ = 0 (τ translation). No mode with 0 < Re λ < 0.3.
- **U0** (aLL < Re λ < 0.3, ω/2 < Im λ < 3ω/2): no zero. (At D = 30 one candidate pole, −0.016 + iω, lies inside,
  so det M winds −1 there.)
- **B1a** (0.3 < Re λ < a1): one zero, λ_g, the gauge mode. In particular λ = 1 is not a zero.
- **B1b** (a1 < Re λ < λ_phys + 0.6): one zero, λ_phys. **B1c** (λ_phys + 0.6 < Re λ < aR ≈ 6.2–6.6): none. These are
  the deliberate checks: the box holding the known growth rate counts 1, the box next to it counts 0, and so does the
  gauge box B1a, with the same code and the same sampling.
- **U** (0.3 < Re λ < aR, ω/2 < Im λ < 3ω/2): no zero. det M winds −7 around it, and there are 7 candidate poles inside
  (the resonances kk with Re between 0.3 and aR on Im = ω; at D = 61 at Re = 0.738, 1.637, …, 6.133). Since a winding of
  −7 needs at least 7 poles, every one of the 7 candidates is a simple pole and the box holds no zero. The count is
  therefore sensitive to individual poles one at a time, not only to the known zeros.
- **full** (0.3 < Re λ < aR, |Im λ| < 3ω/2): det M winds −12 = 2 − 14: the two zeros λ_g and λ_phys and the 14 poles
  on Im = ±ω, nothing else.

So in −0.08 … −0.15 < Re λ < 6.2 … 6.6 and |Im λ| < 3ω/2, which holds every growth rate of both symmetry sectors
(section 1), det M has exactly three zeros at each of D = 30, 45, 61: λ = 0 (τ translation, marginal), λ_g (residual
gauge, a coordinate change) and λ_phys. The real-axis scan of each run shows the same: det M changes sign only in
(0.5, 1), at λ_g, and in (2, 2.5), at λ_phys.

**Sampling quality** (`hp/modecount_table.py`; per edge in `hp/data/modecount_results.jsonl`, all samples in
`hp/data/modecount_<D>_M<M>_evals.jsonl`):

| D | determinants | largest phase step of g | largest phase step of det M itself | ln\|g\| on all edges | sample spacing | halving checks (mismatches) | nearest candidate pole to an edge |
|---|---|---|---|---|---|---|---|
| 30.0688 | 98 | 0.36 rad | 2.60 rad | −1.23 … 0.69 | 0.20–0.40 | 4 (0) | 0.134 |
| 45.0507 | 104 | 0.38 rad | 3.08 rad | −1.41 … 0.73 | 0.19–0.40 | 3 (0) | 0.129 |
| 61.0264 | 125 | 0.37 rad | 2.26 rad | −1.46 … 0.75 | 0.19–0.39 | 4 (0) | 0.081 |
| 30.0688, M 48 | 148 | 0.50 rad | 2.72 rad | −0.73 … 6.17 | 0.002–0.53 | 50 (0) | 0.134 |

In the M = 32 counts the largest phase step of g is below π/8 = 0.39 on every edge, against the tolerance π/4. The
larger steps of det M itself come from its growth along the real direction, ln|det M| rising by about 33 (D = 45, 61)
to 57 (D = 30) per unit of Re λ; the matching phase is removed exactly by P₂. In the M = 32 counts ln|g| stays within
−1.5 … 0.8 on every edge: no edge passes near an unaccounted zero or pole of g (one at distance d from an edge would pull ln|g| down by
about |ln d| there). The determinant itself is accurate to about 10⁻¹⁰ relative at D = 61 (the two LUs above), far
below what moves a phase. The windings, by construction integers, are reliable because every step is far below π.

At D = 61 the first count was run with λ_g ± iω wrongly in the list of expected zeros. It gave wind(g) = −1 in U and
−2 in the full box, and with that bookkeeping the same windings of det M (−7, −12). Recounted from the same samples
with the corrected list (`--v2` in `hp/modecount_jobs.sh`), every wind(g) is 0. This is the observation that showed
that the other sector has no separate gauge zero.

## 4. Checks that the count can fail

- **Known zero in, known zero out (item 4 of the task).** B1b holds λ_phys alone and B1c, next to it, nothing: their
  windings of det M are 1 and 0 at every D (section 3). B1a holds the gauge zero alone and counts 1. U holds 7 poles
  and no zero and winds −7, so single poles are resolved too.
- **Planted zeros** (`hp/modecount_planted.py`, `hp/data/modecount_planted.log`). A synthetic det M with the D = 30
  structure (ω, λ_g, λ_phys, the rows of horizon poles, growth $`e^{57\lambda + 0.3\lambda^2}`$, the 2iω copies) goes
  through the same code (geometry, factors, trend, sampling, bookkeeping). With nothing planted every wind(g) is 0;
  with an extra zero (or pair) planted in B1c (complex and real), in U (an other-sector real mode at 1.9 + iω, and one
  0.2 below the top edge), in B0 (a slowly growing pair at 0.07 ± 0.5i), in B1b next to λ_phys (2.4 ± 0.2i) and in
  B1a 0.05 from an edge, every box counts exactly what was planted in it. Pass criteria were fixed in the script before
  it was run; it exits non-zero on a failure.
- **Bookkeeping error caught.** The first D = 61 count listed λ_g ± iω as expected zeros. wind(g) came out −1 in U,
  not 0: the count exposed the wrong expectation, which the argument in section 1 then explained (the other sector's
  gauge mode is the same perturbation as λ_g).
- **The kernel at complex λ** (section 2): bitwise agreement with hpkernel at real λ, the Cauchy–Riemann test, and the
  gauge zero at 1 + ⟨c₁⟩ to 10⁻¹⁶.

## 5. Second resolution, and other D

**M = 48 at D = 30, without the freeze.** The M = 32 state resampled to M = 48 τ points converges only to
|F| = 9×10⁻¹⁸ (Newton stalls, Levenberg–Marquardt stops there). The count (`hp/modecount_wind.py
hp/data/states/up6_D30.0688.npz --M=48 --h0=0.6 --fmax=1e-15`, 148 determinants) agrees with M = 32 in every box but
one: B0 1 (λ = 0), B1a 1 (λ_g), B1b 1 (λ_phys), U 0 zeros (det M winds −7), U0 0 zeros, but **B1c holds an extra
conjugate pair**, wind(g) = 2. The real-axis signs are unchanged (sign changes only at λ_g and λ_phys). The count's
own samples place the upper zero at 4.267 + 1.324i (first moment of the argument principle over the upper half of
B1c, `hp/modecount_moment.py`; the same computation gives 0 for the M = 32 count and for U). A small circle confirms
it: det M winds +1 on |λ − (4.267 + 1.324i)| = 0.35 at M = 48 (`hp/modecount_circle.py`, 24 points, largest step
0.37 rad). Along the top edge Im λ = 3ω/2 there is a further zero within about 0.01 of the edge near Re λ = 3.30,
just outside the box (it does not enter U, which counts 0).

**The same with the kfreeze-13 cut.** Resampled to M = 48 with the centre data's τ-harmonics above 13 pinned (the cut
the production runs use from D = 35 on), the D = 30 state converges to |F| = 8×10⁻²⁸, and det M winds **0** on the
same circle (16 points, largest step 0.05 rad: g is flat there).

**Reading.** The extra pair is present only at M = 48 and only without the cut. It is absent at M = 32, whose count
covered the same region with steps of g below 0.36 rad. It disappears when the centre harmonics 15–47 are pinned,
which is also what lets the M = 48 background converge. These are the solver's weak ("junk") directions (README,
"The weak directions"): a ladder of high τ-harmonics of the centre data, outgoing waves near the centre that barely
reach the matching point, with singular values falling like $`e^{-0.7k}`$, which the production runs pin from D = 35
on. With M = 48 and no cut, harmonics 33–47 add more such directions, the background is undetermined in them at the
10⁻¹⁷ level, and det M picks up a zero there. The data support reading the pair as an artifact of that unconverged,
weakly coupled sector, not as an unstable mode.

That reading has a limit, and it is a limit of every finite-M count, not only of this one. A box of height 2ω holds
every growth rate only for perturbations the layout can represent once shifted into the strip, i.e. whose
eigenfunction lives in |k| ≤ M − 1. A perturbation that oscillates much faster in τ than the background, with its
weight near harmonics 30–50 after the shift, is not resolved at M = 32. It is also removed by the cut, so the cut test
does not separate such a mode from an artifact. What would separate them is a third resolution: at M = 64, a genuine
mode stays at the same λ while an artifact of the truncation moves or multiplies. That run is proposed in section 7.
So the statement this count supports is: **no extra unstable mode among perturbations with τ-harmonics up to about
30** (both sectors, 0 < Re λ < 6), with the M = 48 pair as the one open item.

**M = 48 at D = 61, with the cut (as in the production run).** The resampled state reaches only |F| = 2.7×10⁻¹⁵ after
Levenberg–Marquardt (kept in `hp/data/modecount_state_D61.0264_M48_kf13.npz`), above the 10⁻¹⁵ accepted for a
count, so no count was run on it. Continuing the convergence and the count is local task 6.

**D = 4, f64 solver, real λ.** On 0.02 < λ < 6.47 (130 determinants of the Rust kernel's exact mode matrix, step 0.05)
det M changes sign exactly twice: at λ = 0.6138559 (λ_g = 1 + ⟨c₁⟩ = 0.6138550 from the state's a_h; in the f64 RK4
discretisation the discrete zero sits 9×10⁻⁷ away) and at λ = 2.6740838 (λ_phys; γ = 0.3739599 from this kc 39,
x_K 0.6 state, against the quoted 0.3739610, which is step-converged). There is no sign change at the full-grid
resonance λ = 0.386, so it is not a pole, as expected. This covers only the symmetric sector's real growth rates at
D = 4; a zero of even multiplicity would not change the sign (`hp/data/modecount_check.jsonl`, record "realscan").

**Lower D with complex λ** (D = 16.76 and 8.25 have states in `hp/data/states`; D = 4 needs a double-double state
first, `hp/modecount_lowD.py`) is proposed for the local machine (section 7).

## 6. Coverage and limits

- **Region.** −0.15 … −0.08 < Re λ < 6.2 … 6.6 and |Im λ| < 3ω/2. By the 2iω periodicity of the folded layout (exact
  in the continuum) this holds every growth rate of both symmetry sectors with Re λ in that range. Not covered:
  Re λ > 6.6. The physical growth rate is near 2 and tends to 2 as D → ∞; nothing here rules out a mode with
  Re λ > 6.6, but nothing points to one either (ln|g| is flat out to the right edge).
- **Dimensions.** Complex-λ counts at D = 30.07, 45.05 and 61.03; at D = 4 a real-axis scan only. D = 16.76 and 8.25
  have states and are local tasks 1 and 2; D = 4 with complex λ is task 3.
- **Harmonics.** A count covers the perturbations the layout represents once their growth rate is shifted into the
  strip: τ-harmonics up to 31 at M = 32, and for the centre data only up to 13 from D = 35 on (the cut is part of the
  production formulation there). A perturbation oscillating much faster in τ than the background is outside every
  finite-M count; the M = 48 pair at D = 30 sits exactly in that sector (section 5).
- **Discretisation.** A count is of the zeros of the discretised det M. At M = 32 no zero beyond the expected ones
  appeared at any D. At M = 48 (D = 30, no cut) one pair appeared, absent at M = 32 and with the cut (section 5).
  The gauge zero sits at 1 + ⟨c₁⟩ to 10⁻¹⁶ (D = 30) and 5×10⁻¹¹ (D = 61); λ_phys agrees with `hp/gamma_dd.py`.
- **Sampling.** Steps of g at most 0.38 rad (M = 32) and 0.50 rad (M = 48) against the π/4 tolerance. A zero within
  about 0.2 of an edge, or a zero–pole pair near an edge, would show as a large step or a dip of ln|g|, and none did
  (ln|g| ≥ −1.5). Two zeros of g hidden between the same two samples (about 0.4 apart) and adding a full turn would be
  missed; the halving checks found no such case where steps exceeded π/8, but not every interval was halved.
- **Accuracy of det M.** Double-double throughout; the complex and real LUs agree to 4×10⁻¹⁰ in ln|det M| at D = 61,
  where max |M| = 2×10⁹. Phases are therefore good to far better than 10⁻⁶ rad on every edge.
- **What the gauge zero means.** λ_g is a coordinate mode of the formulation that pins the light cone to x = 1 (section
  1); it is not a perturbation of the spacetime, so it does not enter the critical exponent. Its value matches the
  analytic 1 + ⟨c₁⟩, and its real sign change is the only one besides λ_phys.
- **What is not addressed.** Non-spherical perturbations; competition with other self-similar solutions (the CSS
  solutions of Clark and Pimentel); evolutions above D = 14. The count shows only that the DSS branch has one physical
  unstable spherical mode in this region at these D.

## 7. Not done here: proposed for the local machine

The cloud machine is shared; each of these needs one to a few hours on one thread (estimates are for an unloaded cloud
core; AGENTS.md puts the local machine at about 4× slower with 6 busy threads). One thread per job; the jobs are
independent. Commands run from the repository root (`hp/modecount_local.sh` holds them); each count is resumable
(evaluations in `hp/data/modecount_<tag>_evals.jsonl`, re-converged states in `hp/data/modecount_state_*.npz`). In
every count, wind(g) ≠ 0 in a box is a finding: stop, locate it (`hp/modecount_moment.py TAG STATE`), test it on a
circle at another M (`hp/modecount_circle.py`), and report.

| task | command | cost (one thread, unloaded) | stop condition | push |
|---|---|---|---|---|
| 1. D = 16.76, M 32 | `bash hp/modecount_local.sh 1` | ~80 determinants of N = 1152, ~45 s each on a cloud core | as below | `hp/data/modecount_D16.7600_M32*`, the results line |
| 2. D = 8.25, M 32 | `bash hp/modecount_local.sh 2` | convergence, then ~80 determinants of N = 1408, ~70 s each | setup raises if \|F\| > 10⁻¹⁵ | `hp/data/modecount_D8.2500_M32*`, the saved state |
| 3. D = 4, double-double | `bash hp/modecount_local.sh 3` | seeding: Newton at N = 1281 then 1921; count: ~80 determinants of N = 1920, ~2–3 min each | pilot: \|F\| < 10⁻²⁰ at M 32 and Δ within 10⁻⁶ of 3.4454555, else stop | the state, `hp/data/modecount_D4.0000_M48*` |
| 4. the D = 30 pair at M 64 | `bash hp/modecount_local.sh 4` | two convergences at N = 1536, ~20 determinants each, ~2 min each | if the pair stays at the same λ at M 64 without the cut, stop and report | `hp/data/modecount_circles.jsonl`, states |
| 5. D = 30, M 48 with the cut, full count | `bash hp/modecount_local.sh 5` | ~80 determinants, ~45 s each (state already converged) | wind(g) ≠ 0 in any box | `hp/data/modecount_D30.0688_M48_kf13*` |
| 6. D = 61, M 48 with the cut | `bash hp/modecount_local.sh 6` | convergence from \|F\| 2.7×10⁻¹⁵, then ~80 determinants | setup raises if \|F\| > 10⁻¹⁵ | `hp/data/modecount_D61.0264_M48*` |
| 7. D = 45, M 48 with the cut | `bash hp/modecount_local.sh 7` | as 6 | as 6 | `hp/data/modecount_D45.0507_M48*` |
| 8. D = 61, Im λ up to 3.5ω | `bash hp/modecount_local.sh 8` | ~60 new determinants (shares the M 32 cache) | expected wind(g) = 0 with the 2iω copies of 0, λ_g, λ_phys inside | results line, cache |
| 9. D = 61, Re λ up to 12 | `bash hp/modecount_local.sh 9` | ~50 new determinants | wind(g) ≠ 0 | results line, cache |

Order: 4 first (it decides the open item), then 1, 2, 5; 3 after its pilot (the seeding alone, checked before the count
starts).

## 8. Files and how to reproduce

| file | what |
|---|---|
| `hp/modecount_cx/` | the crate: det M at complex λ on top of the hp kernel (`src/lib.rs`); build with `hp/modecount_build.sh` |
| `hp/modecount_common.py` | loading a state, rebuilding hpkernel's and the crate's layouts, λ_g, the candidate poles |
| `hp/modecount_check.py` | the kernel checks of section 2 → `hp/data/modecount_check.jsonl` |
| `hp/modecount_wind.py` | the count → `hp/data/modecount_results.jsonl`, logs `hp/data/modecount_<D>_M<M>.log` |
| `hp/modecount_planted.py` | the planted-zero test → `hp/data/modecount_planted.log` |
| `hp/modecount_f64.py` | D = 4: `--realscan` (Rust f64, real λ); the complex central-difference path is not used |
| `hp/modecount_table.py` | the tables of section 3 from the results file |
| `hp/modecount_circle.py` | winding of det M on small circles, with a local linear detrending → `hp/data/modecount_circles.jsonl` |
| `hp/modecount_moment.py` | locates the zeros inside a box from a count's own samples (first moment) |
| `hp/modecount_local.sh` | the runs proposed for the local machine (section 7) |
| `hp/modecount_lowD.py` | a double-double state at low D from an f64 state (for the local D = 4 task) |
| `hp/modecount_jobs.sh`, `hp/modecount_queue.sh` | the job list as run here (resumable, commits after each job) |
| `hp/data/modecount_*` | evaluation caches, logs, results, re-converged M = 48 states |

To recompute a count from its cached determinants (no new determinants; a few minutes, since the state is first
re-converged in the rebuilt layout): `python3 hp/modecount_wind.py
hp/data/states/up6_D30.0688.npz --lam=2.079747703853` (likewise `hp/data/states/up7_D45.0507.npz` and
`hp/data/seeds/local/up23_D61.0264.npz --lam=2.043532530673195`), then `python3 hp/modecount_table.py`. Without the
caches each count takes 100–125 determinants, 20–60 s each on one core.

Two attempts are kept for the record and not used. The M = 16 run at D = 61 (`hp/data/modecount_D61.0264_M16*`): the
state resampled to M = 16 did not converge (|F| = 1.3×10⁻⁹ after Levenberg–Marquardt), and on it det M had an
additional real sign change between 4 and 4.5, which cannot be interpreted on an unconverged background;
`modecount_common.setup` now refuses such a background. The complex-λ f64 check at D = 4 (in
`hp/data/modecount_queue.log`) failed against the Rust kernel, as described in section 2.

## 9. Proposed README text

For the bullet "**Assumed, not checked: the branch stays critical.**" in "What is known about the selection, and what is
open":

> **Checked at D = 30, 45 and 61: one physical unstable mode.** The zeros of the growth-mode determinant were counted
> by the argument principle in −0.08 < Re λ < 6.18 and a full period in Im λ, which holds the growth rates of both
> symmetry sectors (notes/MODE_COUNT.md). Besides λ_phys there are only the τ translation (λ = 0) and a gauge mode of
> the formulation at λ = 1 + ⟨c₁⟩ (a change of time coordinate that keeps the light cone at x = 1). Open: a second
> resolution (M = 48) shows one extra pair at D = 30, absent at M = 32 and with the high-harmonic cut, and taken as a
> truncation artifact pending M = 64; lower D with complex λ; Re λ > 6.6; non-spherical perturbations; competition with
> CSS solutions (Clark and Pimentel).


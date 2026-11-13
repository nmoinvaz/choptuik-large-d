# The Borel geometry of S₁ from 23 coefficients: the far end (#547), and chord against arc (#548)

Status of everything here: numerical evidence. Nothing is rigorous. It bears on the type bounds of routes A3 and P5,
which follow from the farthest singularity of the Borel transform (proof/D_entire.md sections 1 and 7). Sections 1–4
are #547, sections 5–6 are #548, and section 7 is #670.

- **#547.** A calibrated estimator recovers the far end of smooth-density plants within 0.009, but it fails #547's
  acceptance. On the real series two controls fire, so no far end is quoted (§4).
- **#548.** No statistic of σ₀–σ₂₂ can tell a chord from an arc at the measured errors. A function with only two branch
  points reproduces the arc plant's 23 coefficients within those errors (§6).
- **#670.** More coefficients would not separate them either. The smallest such twin of the arc plant has 19 terms at
  23 coefficients and 19–20 at 31, with the errors of the extra orders extrapolated, so its size does not grow with
  the coefficient count (§7).

## 1. Question

B(t) = Σ k!σ_k t^(−k−1). Its singular support was placed by D_entire §1 and §7.1 on a curve from P0 ≈ 0.749+0.155i
to κ⋆ = 1.279+1.081i (|κ⋆| = 1.675), with κ⋆ from the ratio law over k = 6–11. #541 found that from k = 8 the real
[k/k] approximants have a pole past κ⋆, at |t| = 1.715, 1.762 and 1.806 for k = 8, 9 and 10, and that the ratio law over
k = 12–21 gives |κ| = 1.833. On its plants that window overshot κ⋆ by 0.10–0.14, so neither number located the far end.

## 2. Method

### 2.1 Estimator

`solver/inner_amp/pade_farend.py`. In u = 1/t, f(u) = Σ a_k u^k with a_k = k!σ_k. The far end is the singularity of f
nearest u = 0.

The estimator (DA2) uses second-order inhomogeneous differential approximants, Q₂f″ + Q₁f′ + Q₀f = P, with polynomial
Q_j and P matched to the series (Guttmann and Joyce 1972; Hunter and Baker 1979). The singular points of an approximant
are the zeros of Q₂. The ensemble is fixed:

- the degrees (d₀, d₁, d₂) are (d, d, d) and (d − 1, d − 1, d) for d = 4, 5, 6;
- the degree of P is chosen so that the conditions use a₀…a₂₂, or a₀…a₂₀.

That gives 11 members. Each member's estimate is its farthest zero t = 1/u of Q₂ with 1.2 < |t| < 3 and Im t > 0. The
estimate is the componentwise median over the members.

Why it can work where the ratio law does not: near its far end such a Borel transform is A(t)(κ − t)^γ + H(t), with A
and H analytic there. A ratio fit must absorb the 1/k corrections that A's variation and the end exponent produce. A
differential approximant carries A and H in its polynomial coefficients. For a chord density with a semi-classical
weight, such as (1 − s)^ex e^{cs}, the relation is exact (selftest 1).

The ensemble was chosen on five development plants (D1–D5 in 2.2), from a scan of first- and second-order
approximants of degree 1–8. First-order approximants and the ratio law are reported as context only.

### 2.2 Plants

All are continuous Laplace integrals, built as in `pade_geometry.py` (#541) and notes/PLANTED_SELECTION.md 2.2. The
density (1 − s)^ex Σ c_m T_m(2s − 1) lies on a path from P0 to P1, times the planted zero factor. Its c_m are fitted to
the real σ₀–σ₁₂ of the node. The far end P1 is κ⋆ moved radially to the modulus R.

- **Kinds.** chord2 (two branch points); arc (the cubic arc, smooth density); arcjump, arcmid and arcsteps (steps on the
  arc, so log branch points along it).
- **Development set**, used to choose the ensemble:
  - D1 chord2, D2 arcjump, at κ⋆;
  - D3 chord2 at R = 1.80, D4 chord2 at R = 1.85, D5 arc at R = 1.85.
- **Validation set**, first computed in the frozen run:
  - arc, arcmid and arcsteps at κ⋆;
  - chord2, arc and arcjump at R = |κ⋆| − 0.05, |κ⋆| + 0.05, 1.75 and 1.90;
  - arc and arcjump at 1.80, and arcjump at 1.85;
  - chord2 and arcjump at κ⋆ at ω/ω∞ = 0.99 and 1.01, and chord2 at R = 1.85 at both;
  - chord2 with the end exponent ex = β − 1 ± 1, at κ⋆ and at R = 1.85;
  - chord2 and arc "stretched" to R = 1.85: the density fitted at κ⋆, the path stretched, with no refit.
- All are at ω/ω∞ = 1.00 unless stated.

The plants fitted with a far end past κ⋆ also match the real σ₀–σ₁₂, with largest relative residuals of order 10⁻⁴
(3.6×10⁻⁵ at κ⋆). Their low orders therefore still grow like |κ⋆|^k, and the far end barely shows at k ≤ 22. They are
the hard case.

### 2.3 Error bars

None of these is a bound.

- **noise**: the largest shift of the estimate over 10 seeds of relative complex Gaussian noise. Its size per order is
  the measured plateau error (`sigma_err_fit.py`, as in #541 `--noise=measured`): 3×10⁻²⁶ at k = 1, up to
  1.4×10⁻¹² at k = 22.
- **spread**: the median absolute deviation of the 11 members from the median.
- **calibration**: the largest error of the estimate on the plants.
- **J**: the difference between J 1250 and J 1500 at ω/ω∞ = 1.00.

## 3. Frozen criteria

These are written into the script's `FROZEN` dict and here before the validation plants and before any real run of the
script. The development plants were run with the same estimator before the freeze.

| check | criterion |
|---|---|
| C1 recovery | every plant: \|estimate − P1\| ≤ 0.02 |
| C2 response | every pair whose far ends differ by 0.05 (κ⋆ against κ⋆ ± 0.05 for chord2, arc and arcjump; 1.80 against 1.85 for the same three): the estimates differ by more than 0.03 |
| C3 noise | every plant: noise ≤ 0.01 |
| B1 baseline | the ratio law over k = 12–21 misses every plant with P1 = κ⋆ by more than 0.05, which checks that these plants behave as #541's |
| real range | the real estimate counts as calibrated only if its modulus lies in the planted range, 1.62–1.90 |
| mutation | `--mutate=scale1.03` (σ_k × 1.03^k scales the Borel plane by 1.03) must move the real estimate to 1.03 times itself, within 10⁻⁶, and by more than 0.03. `--mutate=sigma7` (σ₇ × 1.01, which moves no singularity) is reported, not judged |

The real far end is quoted at ω/ω∞ = 1.00 (J 1250), with the calibration, noise, spread and J listed apart, and the
other two nodes reported.

The selftest (`python3 pade_farend.py selftest`) checks three things, each with a mutation that fires
(`--mutate=selftest_sigma18`, `selftest_ensemble`, `selftest_moment`):

1. One member reproduces the end of a semi-classical chord density exactly, with error 0 against a bound of 10⁻¹⁰.
2. The ensemble recovers the end of a density outside that class, (1 + sin 4s/2 + 0.3i cos 3s)(1 − s)^ex, to 3.7×10⁻⁵.
3. The moved-end plants' moments agree with `mpmath.quad` to 6×10⁻⁶¹.

References: A. J. Guttmann and G. S. Joyce, J. Phys. A 5, L81 (1972), doi:10.1088/0305-4470/5/9/001 (differential
approximants); D. L. Hunter and G. A. Baker, Phys. Rev. B 19, 3808 (1979), doi:10.1103/PhysRevB.19.3808 (the
inhomogeneous form). Both entries were checked against the publisher's record through Scilit and the University of
Melbourne's listing, not against the full text.

## 4. Results

All runs used one core under `nice -n 10`, 2026-11-13 07:59–08:12 UTC, with `pade_farend.py` at e3822c28. The records
are in the raw-data repository's `I547-mac/` (cce7a42).

### 4.1 Plants: C1, C2 and C3 fail

The estimator recovers smooth densities, and misses densities with branch points along the arc. Errors are |estimate −
P1| in t units. "noise" and "spread" are as in 2.3.

| family | plants | largest error | noise |
|---|---|---|---|
| family | plants | largest error | noise |
|---|---|---|---|
| chord2 and arc, far end 1.62–1.90, three nodes; chord2 also with exponent β − 2 at κ⋆ and β | 21 | 0.009 | 0.003–0.024 |
| chord2 and arc stretched to 1.85 (low orders not refitted) | 2 | 0.001 | 0.004–0.010 |
| arcmid and arcsteps at κ⋆ | 2 | 0.012 | 0.013–0.027 |
| arcjump, far end 1.62–1.90, three nodes | 9 | 0.042 | 0.02–0.36 |
| chord2 at 1.85 with the end exponent β − 2 | 1 | 0.063 | 0.066 |

On arcjump the estimate's modulus falls short of the far end by 0.02–0.04 once the end is past κ⋆. A likely cause, not
tested here, is the log branch point of the step at s = 0.75, which competes with the far end at these orders.

| check | result | verdict |
|---|---|---|
| C1, every plant within 0.02 | 7 of 35 fail: six arcjump plants and chord2 at 1.85 with exponent β − 2 | BAD |
| C2, an end moved 0.05 moves the estimate > 0.03 | 0.047–0.051 for chord2 and arc, but 0.015 for arcjump κ⋆ → κ⋆ + 0.05 | BAD |
| C3, noise ≤ 0.01 | above 0.01 on 24 of 35 plants, up to 0.36 (arcjump) | BAD |
| B1, the ratio law 12–21 misses the κ⋆ plants by > 0.05 | 0.10–0.14, as in #541 | OK |

The ratio law over k = 12–21 misses the plants with far ends at 1.75–1.90 by 0.02–0.36, so its value of 1.833 on the
real series does not locate the far end either.

### 4.2 Real series, frozen run: two controls fire

| | ω/ω∞ 0.99 | 1.00 | 1.01 | 1.00, J 1500 |
|---|---|---|---|---|
| DA2 median | 2.201+1.328i | 2.228+1.435i | 2.303+1.349i | 2.222+1.371i |
| modulus | 2.570 | 2.650 | 2.669 | 2.610 |
| spread | 0.27 | 0.24 | 0.18 | 0.28 |
| noise | 0.15 | 0.19 | 0.13 | |

| control | bound | result | verdict |
|---|---|---|---|
| real range | modulus in 1.62–1.90 | 2.65 | BAD, fires |
| `--mutate=scale1.03` | equal to 1.03 × the estimate within 10⁻⁶ | off by 0.11 | BAD, fires |
| `--mutate=sigma7`, reported only | | the estimate moves 0.25 | WATCH |

Every member's farthest zero lies at |t| = 2.4–3.0, against the window edge at 3. Scaling by 1.03 pushes some members'
zeros past the edge, and those members then report another zero, which is why the scale control fails. By the frozen
rules no far end is quoted, and the work on the real quote stopped here.

### 4.3 Post-freeze diagnostics, labeled so

`pade_farend_diag.py` prints every zero of Q₂ with 1.65 < |t| < 6, for each member. Zeros inside |t| = 1.65, the
arc's interior, are not printed, because #548 had not yet been frozen.

- **On the plants, the members agree.** chord2 at κ⋆ puts 11 of 11 members within 0.05 of the far end (median absolute
  deviation 0.0004). chord2 at 1.85 puts 10 of 11 there (0.0016), and arcjump 6 of 11 (0.015).
- **On the real series, no six members agree within 0.05 anywhere past |t| = 1.65.** At ω/ω∞ = 1.00 every member has
  a zero in 1.98–2.55 + 1.0–1.64i, scattered by 0.1–0.3. Five of 11 have one at 1.53–1.95 − 0.65–0.85i, in the lower
  half plane. None has a zero within 0.17 of #541's far Padé poles (1.355–1.457 + 1.05–1.07i). J 1500 reproduces 7 of
  the 11 members' farthest zeros within 0.03 and moves the other four by 0.1 or more. With a₂₂ dropped, the farthest
  zeros stay in 2.06–2.56 + 1.18–1.59i.

### 4.4 What this means

- The 23 coefficients do not locate the far end by this method. The estimator meets #547's acceptance only on plants
  with a smooth density, and the real series behaves like none of the plants.
- **On the three plants tested (D1, D2, D4), at least 6 of the 11 DA2 members agree on the far end within 0.05. On the
  real series no 6 agree anywhere past |t| = 1.65.**
  This is a second structural difference between the real series and the plants, after #541's (the real Padé poles
  reach the chord sooner than any plant).
- So the evidence that the far end lies past |κ⋆| = 1.675 is still that of #541: the far Padé poles and the drift of
  the ratio law. #547 neither confirms nor refutes it, and the type in routes A3 and P5 stays uncalibrated.

### 4.5 Proposed text for proof/D_entire.md

As a bullet at the end of 7.4, after "Together, these are numerical evidence ...":

```diff
 - **Together, these are numerical evidence** that the far end of the singular support, and with it the type, lies
   past |κ⋆| = 1.675.
+- **A calibrated estimator does not locate it (#547).** Second-order differential approximants
+  (`solver/inner_amp/pade_farend.py`, notes/PADE_FAR_END.md) recover the far end of every smooth-density plant within
+  0.009, for far ends from 1.62 to 1.90. They miss plants with steps on the arc by up to 0.042 and one exponent
+  variant by 0.063, and their noise spread reaches 0.36, so #547's acceptance fails. On three plants (chord2 at κ⋆ and
+  at 1.85, arcjump), at least 6 of the 11 approximants agree on the far end within 0.05. On the real series no 6 agree
+  anywhere past |t| = 1.65, and
+  none has a zero within 0.17 of the far poles above. The far end is therefore not located by σ₀–σ₂₂, and the type
+  stays uncalibrated.
```

## 5. Chord against arc at k ≤ 11 (#548): method and frozen criteria

`solver/inner_amp/pade_chord_arc.py`. #541's arc fraction R_k separated its plants chord2 (two branch points) and
arcjump (log branch points along the arc) by only 0.11–0.15. The three candidates below use the same 23 coefficients
and #541's chord frame: s along P0 → κ⋆, d off the chord, and the arc's offset profile d_arc(s).

- **H_DA** is the fraction of the 11 DA2 members of §2.1 that have a zero of Q₂ on the middle of the arc, meaning
  0.3 ≤ s ≤ 0.7 and |d − d_arc(s)| ≤ 0.04. There the arc lies 0.09–0.11 off the chord, so such a zero is nearer the arc
  than the chord.
- **H_HP** is the same fraction for the discriminant zeros of quadratic Hermite–Padé approximants
  p₀ + p₁f + p₂f² = O(uⁿ) (Shafer 1974). The degrees are (m, m, m) and (m − 1, m, m) for m = 6, 7, with n ≤ 23.
- **R_w** is #541's arc fraction pooled over k = 6–11, with each pole weighted by the modulus of its residue.

Each is near 0 for a chord and near 1 for singularities on the arc. The noise is that of §2.3, the measured per-order
errors, over 10 seeds.

H_DA was looked at on chord2 and arcjump at the three nodes, with this noise, while it was being designed. H_HP and R_w
were not.

| check | criterion |
|---|---|
| separation | per node and per statistic: S = X(arcjump) − X(chord2) ≥ 0.3 and S ≥ 3ν, with ν the larger noise spread of X on the two plants |
| pass | the separation holds at all three nodes, and the mirror mutation fires |
| mutation | `--mutate=mirror` reflects the arc's profile across the chord, and must give S < 0.3 at every node. `--mutate=sigma7` (σ₇ × 1.01) is reported, not judged |
| real run | only for a statistic that passes. Otherwise the task reports the bound |
| bound | the noise-scale study: the largest factor F among 1, 0.1, …, 10⁻⁴ by which the measured errors must be multiplied for the separation to hold at all three nodes |

The selftest (`python3 pade_chord_arc.py selftest`, mutations `selftest_hp` and `selftest_step`) checks two things:

1. The Hermite–Padé discriminant of ((1 − k₁u)(1 − k₂u))^{1/2} e^{u/5} has zeros at both branch points, to 1.8×10⁻¹².
2. A planted density step on the arc at s = 0.5–0.6, added to chord2, raises H_DA from 0.00 to 0.55.

## 6. Results (#548)

All runs used one core under `nice -n 10`, 2026-11-13 08:17–09:00 UTC, with `pade_chord_arc.py` at 6566f429. The
records are in the raw-data repository's `I548-mac/` (edcf1b1). The twin stage of 6.3 was added after the plants run.

### 6.1 Plants: no statistic separates at the measured errors

Each entry is the clean value, with the noise spread in parentheses.

| | H_DA | H_HP | R_w |
|---|---|---|---|
| chord2, ω/ω∞ 0.99 / 1.00 / 1.01 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.25 | 0.66 / 0.73 / 0.81 |
| arcjump, 0.99 | 0.55 (0.36) | 0.75 (0.75) | 0.71 (0.09) |
| arcjump, 1.00 | 0.73 (0.64) | 1.00 (1.00) | 0.73 (0.10) |
| arcjump, 1.01 | 0.64 (0.64) | 0.50 (0.50) | 0.82 (0.07) |
| context at 1.00: arcmid, arcsteps, arc | 0.91, 0.36, 0.00 | 1.00, 0.75, 0.00 | 0.83, 0.76, 0.70 |

| check | H_DA | H_HP | R_w |
|---|---|---|---|
| clean separation, S ≥ 0.3 at every node | 0.55–0.73, OK | 0.25–1.00, BAD at 1.01 | ≤ 0.05, BAD |
| S ≥ 3ν at the measured errors | ν = 0.36–0.64, BAD | ν = 0.50–1.00, BAD | ν = 0.09–0.10, BAD |
| mirror mutation, S < 0.3 at every node | 0 at every node, fires | S = 0.50 at 1.01, does not fire | S ≤ 0.05 |
| σ₇ × 1.01, reported | both plants at 0.09–0.18 | 0–0.5 | 1.8–2.2 |

**Without noise, the singular points of the differential approximants tell the plants apart.** H_DA puts arcjump at
0.55–0.73 and chord2 at 0. #541's pole statistic, pooled and weighted by residue, does not: R_w is 0.66–0.82 on both.
At the measured errors no statistic passes, so by the frozen rule there is no real run.

### 6.2 The noise-scale bound

The measured errors were multiplied by a factor F before the noise was drawn.

| F | H_DA: S / ν at 0.99, 1.00, 1.01 | separates at all three nodes |
|---|---|---|
| 1 | 0.55/0.36, 0.73/0.64, 0.64/0.64 | no |
| 0.1 | 0.55/0.36, 0.73/0.64, 0.64/0.45 | no |
| 0.01 | 0.55/0.27, 0.73/0.36, 0.64/0.45 | no |
| 10⁻³ | 0.55/0.18, 0.73/0.36, 0.64/0.09 | no (fails at 1.00) |
| 10⁻⁴ | 0.55/0.18, 0.73/0.27, 0.64/0.09 | no (fails at 1.00) |

H_HP and R_w separate at no F either (`bound.json`). The spread of H_DA falls far more slowly than the noise. That
points to members whose zeros lie near the 0.04 tolerance, where a perturbation of 10⁻¹⁶ relative moves them across
it; this reading was not checked. Either way, no statistic tried here would separate the plants even with σ₁₃–σ₂₂ 10⁴
times more accurate.

### 6.3 Why no statistic can: a two-branch-point twin of arcjump

The chord twin has a density (1 − s)^ex Σ_{m<M} c_m T_m(2s − 1) on the straight chord P0 → κ⋆, with the plants'
planted zero factor. It is a function with exactly two branch points. It is fitted to arcjump's σ₀–σ₂₂ by least
squares with weights 1/err_k, and χ = max_k |σ_k(twin) − σ_k(arcjump)| / err_k. Here err_k is the measured absolute
error, 3×10⁻²⁶ relative at k = 1 up to 1.4×10⁻¹² at k = 22, with a floor of 10⁻³⁰ at k = 0. `--mutate=twin_noshift`
fits it to chord2 instead, and must reproduce it at M = 6; it does, with χ ≤ 10⁻³⁴.

| M | 6 | 12 | 16 | 18 | 20 | 22 |
|---|---|---|---|---|---|---|
| χ at ω/ω∞ 1.00 | 5×10¹⁷ | 7×10⁷ | 1.3×10³ | 3.1 | 1.5×10⁻³ | 2.5×10⁻⁷ |
| largest \|c_m\| | 53 | 118 | 652 | 1150 | 1560 | 1970 |
| largest \|density\| on the chord | 3.1 | 4.3 | 6.2 | 7.1 | 8.8 | 10.2 |

ω/ω∞ = 0.99 and 1.01 give the same picture: χ = 1.7×10⁻³ and 2.1×10⁻³ at M = 20. For comparison, chord2 has M = 6,
largest |c_m| = 28 and density at most 2.9.

**At the measured errors, a function with only two branch points reproduces all 23 coefficients of the arc plant.**
Its density has 20 terms, and its peak is 2.5–3.5 times chord2's. Any statistic of σ₀–σ₂₂ therefore gives the twin
the value it gives arcjump, to within the noise. So no discriminator at this order exists, unless a prior limits the
density to fewer than 20 Chebyshev terms; at 18 terms χ is already only 2.6–3.1. The separation that H_DA shows on clean
plants rests on such a prior, since the plants have 6 terms.

### 6.4 Proposed text for proof/D_entire.md

In 7.4, replacing the bullet "Twenty-three coefficients do not separate the plants":

```diff
-- **Twenty-three coefficients do not separate the plants.** Both fall from about 1 toward the chord, at nearly the
-  same rate. The frozen separation test failed at every node, so #541's frozen verdict is indecisive.
+- **Twenty-three coefficients cannot separate a chord from an arc (#548).** Both plants fall from about 1 toward the
+  chord, at nearly the same rate, and #541's frozen verdict is indecisive. Singular points of differential approximants
+  do separate the clean plants (0.55–0.73), but at the measured errors their noise spread is as large as that, and
+  stays so with the errors 10⁴ times smaller. A function with exactly two branch points, a 20-term density on the
+  chord P0 → κ⋆ peaking at 2.5–3.5 times chord2's, reproduces the arc plant's σ₀–σ₂₂ within the measured errors at
+  every order (notes/PADE_FAR_END.md §6). So no statistic of these coefficients can tell them apart without a prior on
+  the density.
```

## 7. Twin size against the number of coefficients (#670)

`solver/inner_amp/pade_twin_count.py`. The question is whether more coefficients of S₁, such as #549's σ₂₃–σ₂₄,
could separate a chord from an arc. #548's chord twin is refitted to n = 23, 25, 27 and 31 coefficients of arcjump,
the arc plant of §6, and the smallest twin size is found at each n.

### 7.1 Method

- **Coefficients.** The plants are known functions, so their σ₀–σ₃₀ are computed from them, by the tanh-sinh moments of
  §2.2 kept to order 31. Halving the step changes none of them by more than 8.7×10⁻⁶⁰ relative.
- **Errors.** err_k = F rel_k |σ_k|. For k ≤ 22, rel_k is the measured error of §2.3, with a floor of 10⁻³⁰. For
  k ≥ 23 it is an estimate: log₁₀ rel_k = a + bk, fitted on k = 18–22 at each node. The slope is 0.157–0.158 dex per
  order at the three nodes, with residuals at most 0.008 dex, and it gives rel₃₀ = 2.6×10⁻¹¹ at ω/ω∞ = 1.00. The
  measured errors do not depend on K at a given order (the J 1250 files with K = 41, 43 and 45 agree), so the model
  assumes that #549's runs would keep J 1250's errors. F = 1 is the model, and F = 0.1 is every error 10 times smaller,
  which is what more precise runs would buy. F = 0.01 is context: J 1500's errors at k = 16–20 are 10^2.3–10^2.7 times
  smaller than J 1250's.
- **Twin.** As in §6.3: the density (1 − s)^ex Σ_{m<M} c_m T_m(2s − 1) on the chord P0 → κ⋆, times the planted zero
  factor, fitted by least squares with weights 1/err_k. Its moments are exact, from Beta functions, rather than
  quadrature, so the same-class control also tests the plants' extra orders. The fit is a Householder QR at 150 digits.
- **Sizes.** χ = max_k |σ_k(twin) − σ_k|/err_k as in §6.3, and l2 is the root sum of squares of the same ratios.
  - M_min is the smallest M at which the least-squares twin has χ ≤ 1. It is an upper bound on the size of the smallest
    twin.
  - M_lo is the smallest M with l2 ≤ √n. It is a lower bound, since a twin with χ ≤ 1 has l2 ≤ √n and the least-squares
    twin minimizes l2 among twins of its size.
- **Visibility, context.** d_m is the weighted distance of the column of T_m from the span of the lower columns. A term
  of size |c_m| moves σ₀…σ_{n−1} by |c_m| d_m error units once the lower terms are refitted. It depends on the basis and
  the errors only.

### 7.2 Frozen criteria

Written into the script's `FROZEN` dict before its first selftest, and here before the run stage was first executed.

| check | criterion |
|---|---|
| C0 same-class control | chord2's own twin at M = 6 has χ ≤ 10⁻³ at every n, node and F |
| C1 precision | at M_min − 1 and M_min, χ at 250 digits agrees with 150 digits within 10⁻³ relative |
| C2 reproduction | at n = 23 and F = 1, χ at M = 18, 20 and 22 is within a factor 2 of #548's values; a failure is reported as a discrepancy |
| near n − 3 | at F = 1 and every node, M_min ≥ n − 4 at n = 27 and 31 |
| saturates | at F = 1 and every node, M_min(31) ≤ M_min(23) + 2 |
| verdict on #549 | if M stays near n − 3 or saturates, at F = 1 and at F = 0.1, more orders cannot separate a chord from an arc without a prior on the density's size, and the prior they need moves by at most M_min(25) − M_min(23) terms. Otherwise the gap n − M_min is reported and the verdict is open |
| context | the plant arc (smooth density on the arc, ω/ω∞ = 1.00), F = 0.01 and d_m are reported, not judged |

`run` exits 1 when C0 or C1 fails at F = 1 or 0.1. `run --mutate=moment` multiplies one exact moment, of T₀ at order
26, by 1 + 10⁻⁸, and must make C0 fail at n = 27 and 31 but not at 23 and 25.

The selftest (`python3 pade_twin_count.py --selftest`) checks six things, and each fires under a mutation:

1. The exact chord moments agree with the plant's quadrature to 2.3×10⁻⁴³ for m, k < 31 (`--mutate=moment` gives
   10⁻⁸).
2. The plants' σ₀–σ₃₀ change by at most 8.7×10⁻⁶⁰ when the quadrature step is halved (`--mutate=quad` compares
   against step 1/8 and gives 2.4×10⁻⁷).
3. The tail fit of the error model has residuals at most 0.05 dex (`--mutate=errtail`, rel₂₀ × 3, gives 0.39).
4. C0 at n = 31 gives χ = 5×10⁻³⁷ (`--mutate=moment` gives 930).
5. arcjump's twin at n = 31, M = 27 has the same χ at 150 and 250 digits (`--mutate=moment`, applied at 150
   digits only, gives a relative difference of 18).
6. Planted chord densities with c_m = 0.8^m e^{im} and M_true = 8, 12 and 16 terms are found at their size, at n = 23
   and 31 and F = 1 and 0.1: l2 > √n at M_true − 1 and χ ≤ 1 at M_true (`--mutate=planted` drops the top term).

Seen before the freeze, and disclosed:

- The selftest's precision check fits arcjump at n = 31, M = 27, and printed χ = 1.0×10⁻¹⁰, so M_min(31) ≤ 27 was
  known.
- The planted sizes were first 8, 16 and 24. M_true = 24 failed: its top term, 0.8²³ T₂₃, moves σ₀–σ₃₀ by only
  3.7×10⁻⁷ error units once the other 23 terms are refitted, so a 23-term density reproduces it. That is a property of
  the data, not a fault of the finder, and the sizes became 8, 12 and 16, whose top terms are visible by factors of
  500 or more.
- d_m was computed with chord2's weights at ω/ω∞ = 1.00 and F = 1. A unit term is visible (d_m ≥ 1) up to m = 17 at
  n = 23, and up to m = 18 at n = 27 and n = 31.

### 7.3 Results

All runs used one core under `nice -n 10`, 2026-11-13 09:20–09:33 UTC, with `pade_twin_count.py` at e6845dfb. The
records are in the raw-data repository's `I670-mac/` (95ad391).

| check | result | verdict |
|---|---|---|
| selftest, and its mutations `moment`, `planted`, `errtail`, `quad` | exit 0, and 1, 1, 1, 1 | OK, all fire |
| C0, chord2's own twin at M = 6, every n, node and F | χ ≤ 1.5×10⁻³⁴ | OK |
| C1, 150 against 250 digits at M_min − 1 and M_min | the same χ in double precision, every row | OK |
| C2, #548's χ at M = 18, 20, 22, n = 23 | equal to 4 digits at all three nodes | OK |
| `run --mutate=moment` | C0 gives χ = 540–1000 at n = 27 and 31 with F = 1 (×10, ×100 at F = 0.1, 0.01), and at most 1.5×10⁻³⁴ at n = 23 and 25; exit 1 | fires |
| near n − 3, M_min ≥ n − 4 at n = 27, 31 | M_min = 19–20, so n − M_min = 7–11 | fails |
| saturates, M_min(31) ≤ M_min(23) + 2 | +1 at every node at F = 1; +1, +2, +2 at F = 0.1 | holds |

The twin's size is given as M_lo–M_min, the bracket of 7.1 that holds the smallest twin, or as one number when they are
equal.

| F | ω/ω∞ | n = 23 | 25 | 27 | 31 |
|---|---|---|---|---|---|
| 1 | 0.99 | 19 | 19 | 19–20 | 20 |
| 1 | 1.00 | 19 | 19 | 19 | 20 |
| 1 | 1.01 | 19 | 19 | 19–20 | 19–20 |
| 0.1 | 0.99 | 19–20 | 20 | 20 | 20–21 |
| 0.1 | 1.00 | 19 | 20 | 20 | 20–21 |
| 0.1 | 1.01 | 19 | 20 | 20 | 20–21 |
| 0.01, context | each node | 20 | 20–21 | 21 | 21 |
| | n − 3 | 20 | 22 | 24 | 28 |

χ of the least-squares twin at ω/ω∞ = 1.00 and F = 1, by size M:

| M | 18 | 19 | 20 | 21 | 22 |
|---|---|---|---|---|---|
| n = 23 | 3.1 | 0.095 | 1.5×10⁻³ | 2.3×10⁻⁵ | 2.4×10⁻⁷ |
| n = 25 | 9.3 | 0.39 | 0.011 | 2.2×10⁻⁴ | 5.5×10⁻⁶ |
| n = 27 | 20 | 0.99 | 0.040 | 1.4×10⁻³ | 3.7×10⁻⁵ |
| n = 31 | 54 | 2.7 | 0.16 | 5.9×10⁻³ | 6.0×10⁻⁴ |

- **The twins are large.** At M_min the twin's density peaks at 7.0–11.1 and its largest |c_m| is 946–1718, over every
  n, node and F. chord2 peaks at 2.9 with |c_m| ≤ 28. These are the least-squares twins, which do not minimize size.
- **The plant arc, context.** Its density is smooth on the arc, so it has two branch points only. Its twin needs 16
  terms at F = 1 (M_lo = 15 at n = 23), 16–17 at F = 0.1 and 17 at F = 0.01, at every n. That twin peaks at 2.7 with
  |c_m| ≤ 28, the size of chord2.
- **Visibility.** With arcjump's weights at F = 1, a unit term is visible for m ≤ 17 at n = 23 and m ≤ 18 at n = 25–31,
  at every node. A term of size 10³ is visible for m ≤ 19, 20, 20 and 21 at n = 23, 25, 27 and 31. Each factor 10 in F
  adds about one term.
- **Error model.** rel₃₀ is 2.4×10⁻¹¹, 2.6×10⁻¹¹ and 2.9×10⁻¹¹ at the three nodes. The tail's slope falls slowly with
  k, and a quadratic fit on k = 18–22 would put rel₃₀ 2.5 times lower. That lies inside the F = 0.1 variant.

### 7.4 What this means

- **The twin's size does not grow as n − 3, it saturates.** From 23 to 31 coefficients the smallest chord twin of
  arcjump grows from 19 terms to 19–20, at every node. So a chord density with a fixed number of terms reproduces
  σ₂₃–σ₃₀ as well, within the extrapolated errors, and the extra orders add no constraint it cannot meet. With every
  error 10 and 100 times smaller, the twin at n = 31 has 20–21 and 21 terms.
- **Why, in terms of the basis.** Beyond about 18 Chebyshev terms, a unit term moves σ₀…σ_{n−1} by less than their
  errors once the lower terms are refitted, and 8 more orders raise that count by only one. Each new order is less
  precise than the last, by 0.157 dex, and what it says about the density overlaps what the lower orders say. A
  likely reading, not tested here, is that the k-th coefficient weights the density near the far end, where the
  factor (1 − s)^ex, with Re ex ≈ 3.2, suppresses it.
- **Verdict on #549, by the frozen rule.** M saturates at F = 1 and at F = 0.1. So more coefficients cannot separate a
  chord from an arc without a prior on the density. For σ₂₃–σ₂₄ (n = 25), the prior they would need moves by
  M_min(25) − M_min(23) = 0 terms at F = 1, and by 0–1 at F = 0.1. #549 is not worth running for this purpose. Its own
  aim, fixing the k = 11 Padé poles, is not judged here.
- **What a prior would need.** A prior on the number of terms has to exclude 19–21 terms, whatever n. A prior on the
  density's size might do better: arcjump's least-squares twins have coefficients 34–61 times chord2's, while the
  twin of the smooth plant arc is the size of chord2. Whether a smaller twin of arcjump exists is open, since the least
  squares do not minimize size.

### 7.5 A discrepancy with 6.3

#548 scanned even M only. With every M, a 19-term twin already reproduces σ₀–σ₂₂ at n = 23, with χ = 0.08–0.11 at the
three nodes, and M_lo = 19, so the smallest twin has exactly 19 terms. §6.3's "unless a prior limits the density to
fewer than 20 Chebyshev terms" should read 19. Its quoted values at M = 18, 20 and 22 are reproduced (C2). This is
reported, not changed.

### 7.6 Proposed text for proof/D_entire.md

In 7.4, after the bullet "Twenty-three coefficients cannot separate a chord from an arc (#548)":

```diff
+- **More coefficients would not separate them either (#670).** Refitted to arcjump's σ₀–σ₃₀, with the errors
+  of σ₂₃–σ₃₀ extrapolated from the measured ones (an estimate), the smallest two-branch-point twin grows only from 19
+  terms at 23 coefficients to 19–20 at 31, and to 20–21 with every error 10 times smaller (notes/PADE_FAR_END.md §7).
+  A twin of fixed size reproduces the extra orders, so σ₂₃–σ₂₄ (#549) cannot separate a chord from an arc without a
+  prior on the density.
```

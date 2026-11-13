# The far end of S₁'s Borel singular support (#547)

Status of everything here: numerical evidence. Nothing is rigorous. It bears on the type bounds of routes A3 and P5,
which follow from the farthest singularity of the Borel transform (proof/D_entire.md sections 1 and 7).

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

# Chord against arc at k ≤ 11 (#548)

## 5. Method and frozen criteria

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
| σ₇ × 1.01, reported | both plants fall to 0.09–0.18 | 0–0.5 | about 2 |

**Without noise, the singular points of the differential approximants tell the plants apart.** H_DA puts arcjump at
0.55–0.73 and chord2 at 0. #541's pole statistic, pooled and weighted by residue, does not: R_w is about 0.7 on both.
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

H_HP and R_w separate at no F either (`bound.json`). The spread of H_DA falls far more slowly than the noise. A few
members have zeros right at the 0.04 tolerance, so even a perturbation of 10⁻¹⁶ relative moves them across it. So no
statistic tried here would separate the plants even with σ₁₃–σ₂₂ 10⁴ times more accurate.

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
Its density has 20 terms and a peak about three times chord2's. Any statistic of σ₀–σ₂₂ therefore gives the twin the
value it gives arcjump, to within the noise. So no discriminator at this order exists, unless a prior limits the
density to fewer than about 18 Chebyshev terms. The separation that H_DA shows on clean plants comes from such a
prior: the plants have 6 terms.

### 6.4 Proposed text for proof/D_entire.md

In 7.4, replacing the bullet "Twenty-three coefficients do not separate the plants":

```diff
-- **Twenty-three coefficients do not separate the plants.** Both fall from about 1 toward the chord, at nearly the
-  same rate. The frozen separation test failed at every node, so #541's frozen verdict is indecisive.
+- **Twenty-three coefficients cannot separate a chord from an arc (#548).** Both plants fall from about 1 toward the
+  chord, at nearly the same rate, and #541's frozen verdict is indecisive. Singular points of differential approximants
+  do separate the clean plants (0.55–0.73), but at the measured errors their noise spread is as large as that, and
+  stays so with the errors 10⁴ times smaller. A function with exactly two branch points, a 20-term density on the
+  chord P0 → κ⋆ peaking at three times chord2's, reproduces the arc plant's σ₀–σ₂₂ within the measured errors at every
+  order (notes/PADE_FAR_END.md §6), so no statistic of these coefficients can tell them apart without a prior on the
+  density.
```

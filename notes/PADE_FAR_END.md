# The Borel geometry of S₁ from 23 coefficients: the far end (#547), and chord against arc (#548)

Status of everything here: numerical evidence. Nothing is rigorous. It bears on the type bounds of routes A3 and P5,
which follow from the farthest singularity of the Borel transform (proof/D_entire.md sections 1 and 7). Sections 1–4
are #547, sections 5–6 are #548, section 7 is #670, section 8 is #669, section 9 is #672, and section 10 is #683.

- **#547.** A calibrated estimator recovers the far end of smooth-density plants within 0.009, but it fails #547's
  acceptance. On the real series two controls fire, so no far end is quoted (§4).
- **#548.** No statistic of σ₀–σ₂₂ can tell a chord from an arc at the measured errors. A function with only two branch
  points reproduces the arc plant's 23 coefficients within those errors (§6).
- **#670.** More coefficients would not separate them either. The smallest such twin of the arc plant has 19 terms at
  23 coefficients and 19–20 at 31, with the errors of the extra orders extrapolated, so its size does not grow with
  the coefficient count (§7).
- **#669.** The bound holds both ways: a function with log branch points along the arc reproduces the chord plant with
  18 terms. No prior on the coefficient norm separates them either, since the arc plant's smallest chord twin is no
  larger than the chord plant's own coefficients. A control fired on the first real scan, so the real series is not
  done (§8).
- **#672.** No twin class, chord or arc, reproduces the real series with fewer than 23 terms, at any node, while each
  plant has a twin of 14–19 terms in another class. No prior on the peak density separates a chord from an arc either:
  the arc plant's smallest chord twin peaks at only 1.4–1.8 times the chord plant's, and this measure passes the
  calibration that the coefficient norm failed (§9).
- **#683.** Every script now reads the real σ_k from their strings at full precision. No quoted real-series number
  moves. The strings' 20 digits are coarser than the measured errors at k = 1–8, but that rounding moves the Padé
  poles, R_k and DA2 zeros by at most 4.4×10⁻⁶, and the real twins' χ by at most 2.3×10⁻⁶ error units (§10).

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
density to fewer than 19 Chebyshev terms (a 19-term twin fits, χ = 0.08–0.11, §7; #548 scanned only even M); at 18
terms χ is already only 2.6–3.1. The separation that H_DA shows on clean plants rests on such a prior, since the
plants have 6 terms.

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

## 8. Twins both ways, twins of the real series, and the smallest twin by size (#669)

`solver/inner_amp/pade_twin_size.py`. #548 and #670 showed that a chord twin reproduces the arc plant. Here the twin
runs the other way and on the real σ₀–σ₂₂ (part 1). Then the twin is minimised by its size rather than by its number
of terms (part 2), to test whether a prior on the density's size could separate a chord from an arc.

### 8.1 Method

- **Twin classes.** Each twin is the planted zero factor times a Laplace integral with density
  w(s) (1 − s)^ex Σ_{m<M} c_m T_m(2s − 1) along a path, with ex = β − 1 as in the plants.
  - *chord*: the chord P0 → κ⋆, w = 1. Exactly two branch points, the class of chord2.
  - *arc*: the cubic arc of §2.2, with arcjump's step weights w = 1, 2, 1, 2 on the quarters of s. Each step is a log
    branch point on the arc, so this is the class of arcjump.
  - *smooth*, context: the same arc with w = 1, the class of the plant arc. Its density continues off the arc, so it
    has two branch points only.
- **Exact moments.** On the arc, γ^k γ′ T_m(2s − 1) is a polynomial in u = 1 − s, and ∫ u^(ex+j) du is elementary on
  each piece, so the arc moments are exact at the working precision. The chord moments are the Beta-function moments of
  §7.1. Neither basis shares the plants' tanh-sinh quadrature.
- **Part 1, the twin scan.** As §7.1: the least-squares twin with weights 1/err_k, χ = max_k |Δσ_k|/err_k, M_min the
  smallest M with χ ≤ 1 (an upper bound on the smallest twin), M_lo the smallest M with l2 ≤ √n (a lower bound). The
  errors are the measured ones, err_k = rel_k |σ_k| (§2.3, floor 10⁻³⁰), at n = 23. The fit is at 150 digits and
  checked at 250.
- **Part 2, the size.** The size of a chord twin is N = ‖c‖₂ = (Σ_m |c_m|²)^{1/2}, the norm of its Chebyshev
  coefficients, with M = 40 terms. This measure was chosen before the runs; the peak density is reported as context.
  - With A the twin's columns and b the target, rows weighted by 1/err_k, the residual is ρ = Ac − b. For M ≥ n the
    least-norm c with residual ρ is A^H (AA^H)^{−1} (b + ρ). So the smallest twin with χ ≤ 1 solves
    N² = min (b + ρ)^H (AA^H)^{−1} (b + ρ) over the polydisc |ρ_k| ≤ 1. That is a convex quadratic problem in n complex
    unknowns.
  - It is solved by a log-barrier Newton method (Boyd and Vandenberghe 2004, ch. 11), at 150 digits. N_hi is the norm
    of an explicit twin, whose χ is recomputed from its coefficients at 250 digits. N_lo is the square root of the
    Lagrange dual at the barrier's multipliers, which bounds the minimum from below for any multipliers. So the
    smallest twin by size lies in [N_lo, N_hi].
  - The errors are #670's model: measured for k ≤ 22, extrapolated beyond (§7.1), F = 1. The targets are chord2,
    arcjump and arc, at n = 23 and 31, at every node.
  - Context: the ridge route the issue suggested (Hoerl and Kennard 1970), c(λ) minimising ‖Ac − b‖² + λ‖c‖², whose
    smallest ‖c‖ with χ ≤ 1 on the path is an upper bound; N at M = 50 at ω/ω∞ = 1.00; the peak density and the
    largest |c_m| of each smallest twin.

References: S. Boyd and L. Vandenberghe, *Convex Optimization* (Cambridge University Press, 2004), ch. 11 (barrier
method) and ch. 5 (duality); A. E. Hoerl and R. W. Kennard, Technometrics 12, 55 (1970),
doi:10.1080/00401706.1970.10488634 (ridge regression). Both were checked against a publisher record or the book's
contents through a web search, not against the full text.

### 8.2 Frozen criteria

Written into the script's `FROZEN` dict, and here, before the `twins`, `size`, `twins_real` and `size_real` stages were
first run.

| check | criterion |
|---|---|
| C0 same class | at n = 23, M = 6 and every node: chord2 in the chord basis, arcjump in the arc basis and the plant arc in the smooth basis each have χ ≤ 10⁻³⁰ |
| C1 precision | at M_min − 1 and M_min of every scan, χ at 250 digits agrees with 150 digits within 10⁻³ relative |
| C2 reproduction | the chord twin of arcjump has M_min = 19 at every node (#670); a failure is reported as a discrepancy |
| both ways | the arc twin of chord2 counts as a twin, not an interpolant, if M_min ≤ n − 1 = 22; M_min and the gap n − M_min are reported per node |
| twin mutations | `--mutate=arccol` (the arc basis' column m = 0 at order 20 times 1 + 10⁻⁸) must make C0 fail for the arc basis at every node, and `--mutate=chordcol` for the chord basis; each run exits 1 |
| twins_real | only if C0 and C1 pass on the plants. Per node, M_min and M_lo of the chord, arc and smooth twins of the real σ₀–σ₂₂ (J 1250). If C1 fires on real data, the work stops. A class counts as preferred by term count only if its M_min is smaller by at least 3 at every node; J 1500 at ω/ω∞ = 1.00 is context |
| S_gap | N_hi/N_lo − 1 ≤ 10⁻⁴ for every target, n and node |
| S_valid | the explicit twin's χ recomputed at 250 digits with the 250-digit basis is at most 1 + 10⁻⁶, and N_lo recomputed at 250 digits with the same multipliers agrees within 10⁻⁶ relative |
| S0 calibration | chord2's smallest twin lies within 10% of chord2's own size N_own (its own 6 coefficients): N_lo ≥ 0.9 N_own and N_hi ≤ 1.1 N_own, at n = 23 and 31 and every node |
| size mutation | `--mutate=scale` (chord2's σ_k × 1.25 with its errors kept) must make S0 fail at every n and node; the run exits 1 |
| separation | at n = 31: ρ_hi = N_hi(arcjump)/N_lo(chord2) and ρ_lo = N_lo(arcjump)/N_hi(chord2). If ρ_hi ≤ 3 at every node, no size prior separates them, and that is the answer. If ρ_lo > 3 at every node, a size prior of factor 3 separates them. Otherwise the verdict is mixed. n = 23 is reported the same way |
| size_real | only if S_gap, S_valid and S0 pass and ρ_lo > 3 at every node at n = 23: the real series' smallest chord twin at n = 23, with ρ_real = N(real)/N(chord2) per node read against 3. Otherwise there is no real size run |
| context | the ridge route, N at M = 50, the peak density and largest \|c_m\| of each smallest twin, and the plant arc's N, reported and not judged |

The selftest (`python3 pade_twin_size.py --selftest`) checks four things. Its mutations `upoly` (dγ/ds × (1 + 10⁻¹⁰)
in the u-polynomial) and `radius` (the solver's polydisc radius 1.05) each make it fail.

1. The u-polynomial moments on the chord agree with the Beta moments of §7.1 to 5×10⁻¹⁰⁰ for m < 50, k < 31
   (`upoly` gives 10⁻¹⁰).
2. The exact stepped-arc moments agree with the plant's tanh-sinh moments to 1.5×10⁻⁵⁸ for m, k < 23 (`upoly` gives
   3×10⁻¹⁰).
3. On a planted polydisc problem with a closed-form answer, Q⋆ = 5.5625, and AA^H's eigenvalues spread over
   10⁻⁴⁰–10⁴⁰, the barrier and the dual both return Q⋆ to 1.4×10⁻¹¹, and the twin has χ ≤ 1 (`radius` gives 4.84).
4. A planted 8-term chord density (c_m = 0.8^m e^{im}, #670's) has a smallest twin no larger than itself, with the
   bracket closed to 5×10⁻¹² and χ ≤ 1 at 250 digits.

Seen before the freeze, and disclosed. These are development runs on the planted chord densities of #670 only, not on
the plants judged here, at ω/ω∞ = 1.00.

- The 8- and 12-term densities' smallest twins are 0.962–0.966 times their own norms, at n = 23 and 31 and M = n, 40,
  50 and 60, with 11–12 of the n constraints active.
- M = 40 and M = 60 agree to 1×10⁻⁵, which fixed M = 40.
- With the radius 1.05 instead of 1 (the selftest's mutation), the 8-term density's smallest twin shrinks by only
  3×10⁻⁵. So the size reflects what the data determine, not the error level.
- The ridge route came within 3×10⁻⁴ of the exact minimum.

### 8.3 Results

All runs used one process under `nice -n 10`, single-threaded, 2026-11-13 10:10–10:33 UTC, with `pade_twin_size.py`
at f2926c0c. The records are in the raw-data repository's `I669-mac/` (31f549f6).

| check | result | verdict |
|---|---|---|
| selftest, and its mutations `upoly`, `radius` | exit 0, and 1, 1 | OK, both fire |
| C0, own class at M = 6 | χ = 2.5×10⁻³⁷ to 1.5×10⁻³⁶ for all three classes at every node | OK |
| `--mutate=arccol`, `chordcol` | χ = 1.6–2.9×10³ (arc) and 0.9–1.1×10⁴ (chord) for the mutated basis at every node, the others unchanged; exit 1 | both fire |
| C1 on the plants | 150 and 250 digits give the same χ on every scan | OK |
| C2, the chord twin of arcjump | M_min = 19 at every node, as #670 | OK |
| both ways, M_min ≤ 22 | 18 at every node | holds |
| C1 on the real series | fired at the first real scan (8.4) | BAD, the real stage stopped |
| S_gap, S_valid | gaps at most 2.5×10⁻¹¹; χ at 250 digits below 1 by 1.8×10⁻¹¹ or more, and N_lo unchanged | OK |
| S0, chord2's smallest twin within 10% of its own size | N/N_own = 0.56–0.60 at every n and node | BAD |
| `--mutate=scale` | N/N_own = 0.70–0.75, S0 fails; it fails without the mutation too, so this shows nothing | uninformative |
| separation at n = 31 | ρ = 0.99–1.59 | no size prior separates |

#### The twin both ways (n = 23)

M_min and M_lo agree in every scan, so each number is the exact smallest size. The largest |c_m| and the peak density
are those of the least-squares twin at M_min.

| twin | M_min at ω/ω∞ 0.99, 1.00, 1.01 | largest \|c_m\| | peak density |
|---|---|---|---|
| stepped-arc twin of chord2 | 18, 18, 18 | 49–64 | 2.66–2.71 |
| smooth-arc twin of chord2 (context) | 14, 14, 14 | 19–22 | 2.36–2.38 |
| chord twin of arcjump | 19, 19, 19 | 1160–1520 | 7.0–9.5 |
| chord2 itself | 6 | 27–30 | 2.87–3.00 |

#### The smallest chord twin by coefficient norm (M = 40, F = 1)

N is given to four digits. N_lo and N_hi agree to 2.5×10⁻¹¹ in every case, so N is the exact minimum.

| ω/ω∞ | n | chord2's own N_own | chord2 | arcjump | arc | ρ = arcjump / chord2 |
|---|---|---|---|---|---|---|
| 0.99 | 23 | 43.83 | 25.04 | 36.04 | 27.41 | 1.44 |
| 0.99 | 31 | 43.83 | 26.27 | 41.77 | 28.59 | 1.59 |
| 1.00 | 23 | 45.94 | 26.00 | 31.11 | 27.69 | 1.20 |
| 1.00 | 31 | 45.94 | 27.24 | 35.22 | 28.85 | 1.29 |
| 1.01 | 23 | 48.93 | 27.31 | 25.46 | 27.95 | 0.93 |
| 1.01 | 31 | 48.93 | 28.60 | 28.30 | 29.13 | 0.99 |

The same twins, context:

| ω/ω∞ | n | peak density: chord2 own / chord2 / arcjump / arc | largest \|c_m\|: chord2 own / chord2 / arcjump |
|---|---|---|---|
| 0.99 | 23 | 2.87 / 3.75 / 10.91 / 3.18 | 26.6 / 11.5 / 14.9 |
| 0.99 | 31 | 2.87 / 3.30 / 12.34 / 3.80 | 26.6 / 12.3 / 17.3 |
| 1.00 | 23 | 2.92 / 3.63 / 9.22 / 3.09 | 27.8 / 11.9 / 12.2 |
| 1.00 | 31 | 2.92 / 3.33 / 10.52 / 3.92 | 27.8 / 12.6 / 14.1 |
| 1.01 | 23 | 3.00 / 3.60 / 7.89 / 3.17 | 29.5 / 12.4 / 9.2 |
| 1.01 | 31 | 3.00 / 3.33 / 8.67 / 3.83 | 29.5 / 13.1 / 10.4 |

- 12–15 of the n residual constraints are active at each minimum.
- M = 50 changes N by at most 5×10⁻⁵ at ω/ω∞ = 1.00.
- The ridge route's best twin is 0.4–0.6% above the minimum for chord2 and arc, and 1.2–2.0% above it for arcjump.

### 8.4 The real series: C1 fired, so the real stages stopped

The first real scan was the chord twin of the real σ₀–σ₂₂ at ω/ω∞ = 0.99. It has M_min = 23 = n. At M = 22, χ is
9.7×10⁴, the same at both precisions. At M = 23 the least-squares twin is the interpolant, so χ is rounding noise, at
2.2×10⁻¹⁰⁷ with 150 digits and 2.6×10⁻²⁰⁷ with 250. C1 compares those two relatively, so it fired, and by the frozen
rule the real stage stopped there. No other real scan was run, and the real size stage was not run either.

The cause was read from `twins_real.json`, without a new run. C1's relative test means nothing when χ is zero to
working precision, and that happens only when M_min = n. That is a defect of the criterion, not a loss of precision.
The criterion was not changed and the stage was not rerun. One possible amendment, for the coordinator to decide: at
M = n, require χ ≤ 10⁻³⁰ at both precisions instead.

The scan's other rows are unaffected, since C1 holds at M = 22. **No chord twin with fewer than 23 terms reproduces the
real σ₀–σ₂₂ at ω/ω∞ = 0.99**, while the arc plant needs 19. At 20, 21 and 22 terms χ is 1.4×10⁷, 1.3×10⁶ and 9.7×10⁴.
The 23-term interpolant has largest |c_m| = 1.4×10¹⁵. That is one node and one class only, and it is not yet a result.

### 8.5 What this means

- **The term-count bound holds both ways.** A function with log branch points along the arc reproduces chord2, which
  has two branch points, with 18 terms. A chord function reproduces arcjump with 19. So each class mimics the other
  with 18–19 terms at the measured errors.
- **The least-squares twins differ in size, but the smallest twins do not.** The least-squares chord twin of arcjump
  has coefficients 39–57 times chord2's and a peak density 2.3–3.3 times chord2's. The stepped-arc twin of chord2 has
  coefficients about twice chord2's and a peak density below chord2's. Minimised exactly, though, the smallest chord
  twin of arcjump has a coefficient norm 0.93–1.59 times chord2's smallest, and its largest |c_m| is 9–17. So the large
  coefficients of the least-squares twins of §6.3 and §7.3 lie in directions the data do not see.
- **The coefficient norm fails the calibration.** chord2's own six coefficients have a norm 1.7–1.8 times its smallest
  twin's, so S0 fails at every n and node, and the coefficient norm does not measure the density's own size. The
  smallest twins lower the coefficients by raising the density: their peak is 3.3–3.8, against chord2's 2.9–3.0. The
  norm weights the polynomial factor, which can be large near the far end, where (1 − s)^ex hides it.
- **No coefficient-norm prior separates a chord from an arc, by the frozen rule.** ρ is at most 1.59 at n = 31. This
  verdict does not depend on S0. arcjump's smallest chord twin, at 25–42, is smaller than chord2's own coefficient
  vector, at 44–49, at every n and node. So a prior on ‖c‖ loose enough to admit chord2 admits a chord twin of the arc
  plant. The smooth plant arc's smallest twin is 1.02–1.10 times chord2's.
- **The peak density is still open.** The coefficient-norm minimisers of arcjump peak at 7.9–12.3. The least-squares
  twins of #670 at n = 31 peak at 7.5–11.0, which is 2.5–3.8 times chord2's own peak. So at n = 31 a twin whose peak is
  at most 3 times chord2's is known only at ω/ω∞ = 1.01. The smallest peak is also a convex problem, and it was not
  computed. A peak-density prior could therefore separate them by a factor of at most 2.5–3.8 at n = 31.
- **Status.** Everything here is numerical evidence on plants. The real series' only scan so far says that no chord
  twin of fewer than 23 terms fits it at ω/ω∞ = 0.99 (8.4).

### 8.6 Proposed text for proof/D_entire.md

In 7.4, after the bullet "More coefficients would not separate them either (#670)":

```diff
+- **Nor would a prior on the coefficients' size (#669).** Minimised exactly over the twins with χ ≤ 1, the smallest
+  chord twin of the arc plant has a coefficient norm 0.93–1.59 times chord2's smallest, at 23 and 31 coefficients.
+  It is smaller than chord2's own coefficient vector at every node (notes/PADE_FAR_END.md §8). That measure fails its
+  calibration, since chord2's own coefficients have 1.7–1.8 times its smallest twin's norm, but the comparison with
+  chord2's own coefficients does not need it. The other way round, a density on the arc with log branch points along
+  it reproduces chord2 with 18 terms. Whether a prior on the peak density separates them is open, since the known
+  chord twins of the arc plant peak at 2.3–3.8 times chord2's density.
```

## 9. The real twins with C1 amended at M = n, and the smallest chord twin by peak density (#672)

Part 1 reruns §8's twin scans on the real σ₀–σ₂₂, which stopped on C1 (§8.4). Part 2 minimises the chord twin's peak
density instead of its coefficient norm.

### 9.1 The amendment of C1 (frozen)

Written into `pade_twin_size.py`'s `FROZEN` (`C1_amended_672`, `mutation_C1_672`) and here before `twins` and
`twins_real` were rerun. It is the coordinator's amendment, and no other criterion of §8.2 changes.

| check | criterion |
|---|---|
| C1 below n | unchanged: at M_min − 1 and M_min, χ at 250 digits agrees with 150 digits within 10⁻³ relative |
| C1 at M = n | χ ≤ 10⁻³⁰ at both 150 and 250 digits |
| a singular check | a 250-digit fit that mpmath finds numerically singular counts as χ = ∞, so C1 fails |
| mutation | `twins_real --mutate=c1dup` replaces column n − 1 of the 250-digit bases by column n − 2. It must make the amended C1 fire on every scan with M_min = n and leave the scans with M_min < n passing. The run judges C1 only, does not stop when it fires, and exits 1 |

Why the amendment. At M = n the least-squares twin is the interpolant, so χ is rounding noise at each precision
(2.2×10⁻¹⁰⁷ and 2.6×10⁻²⁰⁷ in §8.4), and their relative difference measures nothing. What can still go wrong at M = n
is a fit that does not interpolate, and the absolute bound catches that.

Why this mutation. A column multiplied by 1 + 10⁻⁸, as `arccol` does, leaves the 250-digit system square and
nonsingular, so the fit still interpolates at M = n and χ stays at rounding noise. Only a basis that cannot interpolate
can make the amended C1 fire. A duplicated last column is such a basis, and it leaves every row below M = n unchanged.
Before the freeze, this mechanism was tried on the plant arcjump at ω/ω∞ = 0.99 in the chord basis: χ at M = 22 stayed
2.7×10⁻⁷, and at M = 23 mpmath's QR reported the matrix numerically singular. Perturbing the last column toward its
neighbour by 10⁻¹⁵⁰ to 10⁻²⁴⁵ did the same.

### 9.2 Results, part 1: no class reproduces the real series with fewer than 23 terms

All runs used one process under `nice -n 10`, single-threaded, 2026-11-13 10:51–10:57 UTC, with `pade_twin_size.py` at
2f6e3d6d (the diagnostic at bc8bf51d). The records are in the raw-data repository's `I672-mac/`.

| check | result | verdict |
|---|---|---|
| `twins`, the plants rerun | the same M_min, M_lo and χ as §8.3 in every scan; C0, C1 and C2 pass; exit 0 | OK |
| `twins_real`, amended C1 | every scan has M_min = n = 23. Below n the two precisions give the same χ; at M = n, χ is 3×10⁻¹⁰⁸ to 2×10⁻¹⁰⁷ at 150 digits and 1×10⁻²⁰⁸ to 3×10⁻²⁰⁷ at 250; exit 0 | OK |
| `twins_real --mutate=c1dup` | the amended C1 fires at M = n in all 9 scans (the 250-digit fit is singular), and the M = 22 rows still agree; exit 1 | fires |

χ of the least-squares twin of the real σ₀–σ₂₂ (J 1250) at 22 terms. M_min = M_lo = 23 in every case, so no twin with
fewer than 23 terms exists in any class, by the bound of §7.1.

| ω/ω∞ | chord | stepped arc | smooth arc |
|---|---|---|---|
| 0.99 | 9.7×10⁴ | 1.24×10⁵ | 1.27×10⁵ |
| 1.00 | 9.1×10⁴ | 1.15×10⁵ | 1.18×10⁵ |
| 1.01 | 8.6×10⁴ | 1.06×10⁵ | 1.08×10⁵ |
| 1.00, J 1500 (context) | 9.1×10⁴ | 1.15×10⁵ | not run |

- **No class reproduces the real series with fewer terms than coefficients.** The chord, the stepped arc and the
  smooth arc each need all 23 terms, at every node, where the plants' twins need 14–19 (§8.3). By the frozen rule
  neither the chord nor the arc is preferred by term count. The chord fits best at 22 terms, by a factor 1.2–1.3, at
  every node.
- **Where the misfit sits (post-freeze diagnostic, `twins_jdiag`, ω/ω∞ = 1.00).** At 22 terms the residual is below
  one error unit for k ≤ 10 and 38–46 units at k = 11. It peaks at k = 17 (0.9–1.2×10⁵ units) and falls to 130–180
  units at k = 22. The three classes have the same profile.
- **What the twins share.** Every class here has the plants' end points P0 and κ⋆, their exponent β − 1 and their
  zero factor, which were matched to σ₀–σ₁₂ (§2.2). So the result says that no density of up to 22 terms with those
  end points and that exponent fits σ₁₁–σ₂₂ within the measured errors. It does not say which ingredient fails. One
  candidate is the far end, which §4 and D_entire 7.4 suggest may lie past κ⋆.

#### Discrepancies found on the way (reported, not fixed)

- **The real series is held at 53 bits.** `pade_twin_count.real_sig()` reads the J 1250 files at the precision in
  force when it is first called, which is mpmath's default 53 bits in `twins_real` (and in #669's run). The values are
  then off their strings by up to 7.5×10⁻¹⁷ relative, which is 10⁹ times the measured error at k = 1. Read at 150
  digits instead, the chord, arc and smooth scans give the same M_min, and χ at M = 18–22 agrees to four digits
  (`twins_jdiag`). So the result above does not depend on it.
- **The files' strings are coarser than their errors at low order.** Re σ_k has 20 significant digits for k = 1–8,
  a rounding of up to 5×10⁻²⁰ relative, while the measured errors are 3.3×10⁻²⁶ to 2.6×10⁻²¹ there. The fits treat the
  strings as exact. Since the misfit sits at k ≥ 11, this does not move the twin scans either.
- **The J 1500 line of `twins_real` is an artifact.** It compares J 1500 read at 60 digits with J 1250 held at 53 bits,
  and so it reports χ = 1.7×10⁹, at k = 1. Read at 150 digits, the two files agree exactly at k ≤ 9, and they differ by
  at most 1.0 error unit, at k = 16 (`twins_jdiag`). With err_k widened to the J difference, every class still needs
  23 terms.

### 9.3 Part 2: method

`solver/inner_amp/pade_twin_peak.py`. The twin is §8.1's chord twin with M = 40 terms and #670's errors (F = 1). Its
size is now the peak density on #548's grid, P(c) = max_j |φ(s_j)| with φ(s) = (1 − s)^ex Σ_m c_m T_m(2s − 1) and
s_j = j/200, j < 200.

- **The problem.** With A and b as in §8.1 and Φ the matrix of φ on the grid, P⋆ = min over c of max_j |(Φc)_j|
  subject to |(Ac − b)_k| ≤ 1 for every k. Each modulus of a complex number is a second-order cone, so this is a
  convex cone program.
- **Coordinates.** A = U S V by mpmath's SVD at 150 digits. Along a singular direction with s_i ≥ 1 the unknown is the
  residual it causes, along one with s_i < 1 and along the null directions it is the coefficient itself. In these
  unknowns every entry of the problem is at most of order one, so it is solved in double precision. The start is the
  ridge twin with ‖residual‖₂ = 1/2.
- **Solver.** A log-barrier method (Boyd and Vandenberghe 2004, ch. 11) with damped Newton steps, after the unknowns
  are whitened by numpy's SVD. Directions below 10⁻¹³ of the largest singular value, which move neither the density
  nor a residual, are dropped. The polydisc radius is 1 − 10⁻⁹, so the explicit twin has χ < 1.
- **The bracket.** P_hi is the peak of an explicit twin, rebuilt at 150 digits, whose χ is recomputed from its
  coefficients at 150 digits and, with the 250-digit basis, at 250. P_lo is a dual bound (ibid. ch. 5). For any λ with
  ‖λ‖₁ = 1 and Φ^H λ = A^H μ, every twin has max_j |(Φc)_j| ≥ Re μ^H (b + r) ≥ Re μ^H b − ‖μ‖₁. λ comes from the
  barrier's last three centred stages. It is projected at 150 digits so that μ exists exactly, removing its part along
  the null directions and along those with s_i below a threshold (10⁻² to 10⁻¹², or none). Every such bound is valid,
  and P_lo is the largest.
- **Context.** A grid of 200 points leaves room between its points. The minimiser can bulge between s = 0 and 0.005,
  and its polynomial factor can grow near s = 1, where (1 − s)^ex hides it. So the same minimisation is repeated on
  s = j/1000, and each twin's peak is also given on s = j/2000. Neither is judged.

### 9.4 Part 2: frozen criteria

Written into `pade_twin_peak.py`'s `FROZEN` and here before the `peak` and `peak_real` stages were first run.

| check | criterion |
|---|---|
| planted (selftest) | (1) a separable problem hidden by a random unitary rotation, with weights 10⁻³⁰–10³⁰, P⋆ = max_k (\|b_k\| − 1)/s_k = 3.1; (2) the Chebyshev problem on a grid holding the 11 extrema of T₁₀, one constraint \|S 2⁹ c₁₀ − S\| ≤ 1 with S = 100, P⋆ = (1 − 1/S) 2⁻⁹. Each needs P_hi and P_lo within 10⁻⁶ of P⋆, each on its side, and χ ≤ 1. (3) #670's planted 8-term chord density at ω/ω∞ = 1.00, n = 23: P_lo ≤ P_hi ≤ its own peak, P_hi/P_lo − 1 ≤ 0.01, χ ≤ 1 + 10⁻⁶ at 250 digits |
| selftest mutations | `radius` (polydisc radius 1.05), `noproj` (λ not projected) and `dual` (the bound's −‖μ‖₁ dropped) each make the selftest fail |
| P_gap | P_hi/P_lo − 1 ≤ 0.01 for every target, n and node |
| P_valid | P_lo ≤ P_hi (1 + 10⁻¹²); the explicit twin's χ at 250 digits ≤ 1 + 10⁻⁶; P_lo recomputed at 250 digits from the same λ within 10⁻⁶ relative |
| P0 calibration | chord2's smallest peak lies within 10% of its own peak P_own (its 6-term density on the same grid): P_lo ≥ 0.9 P_own and P_hi ≤ 1.1 P_own, at n = 23 and 31 and every node |
| P0 mutation | `peak --mutate=scale` (chord2's σ_k × 0.75, its errors kept) must make P0 fail at every n and node; the run exits 1 |
| separation | §8.2's factor-3 rule on the peak. At n = 31, ρ_hi = P_hi(arcjump)/P_lo(chord2) and ρ_lo = P_lo(arcjump)/P_hi(chord2). If ρ_hi ≤ 3 at every node, no peak prior separates them, and that is the answer. If ρ_lo > 3 at every node, a peak prior of factor 3 separates them. Otherwise the verdict is mixed. n = 23 is reported the same way |
| peak_real | only if P_gap, P_valid and P0 pass and ρ_lo > 3 at every node at n = 23: the real series' smallest peak chord twin at n = 23, with ρ_real = P(real)/P(chord2) read against 3. Otherwise there is no real run |
| context | the grid s = j/1000 with the same rule and chord2's ratio to its own peak there; each twin's peak on s = j/2000, its coefficient norm and largest \|c_m\|; the active constraints; the plant arc; arcjump against chord2's own peak; M = 50 at ω/ω∞ = 1.00. Reported, not judged |

`scale` fires by construction, since chord2's own density is a twin of itself: the scaled target's smallest peak is
at most 0.75 P_own. It shows that the comparison can fail, not that P0 is sensitive.

Seen before the freeze, and disclosed. These are development runs on the planted problems and on #670's 8-term chord
density only, not on chord2, arcjump or arc.

- On the 8-term density at n = 23 the smallest peak is 3.349 against its own 3.721, a ratio of 0.90. Its twin has
  ‖c‖ = 2.3×10⁷, and its peak on s = j/2000 is 5.53, at s = 0.0015, between the first two grid points. That is why the
  finer grid is reported as context.
- The first version of the solver used the least-norm interpolant as its start and stopped on its Newton step limit,
  which gave a bracket of 41% on that density. The ridge start, the whitening and the damped steps fixed both.
- `noproj` passes the two closed-form problems, whose duals are accurate without the projection. It fails only on the
  8-term density, where its "bound" exceeds the explicit twin's peak.

### 9.5 Results, part 2

All runs used one process under `nice -n 10`, single-threaded, 2026-11-13 11:18–11:47 UTC, with `pade_twin_peak.py`
at 15a53292. The records are in the raw-data repository's `I672-mac/` (13ae7a4).

| check | result | verdict |
|---|---|---|
| selftest | separable problem within 6.8×10⁻¹⁰, Chebyshev problem within 2.0×10⁻⁸, the 8-term density bracketed to 1.9×10⁻⁴; exit 0 | OK |
| selftest `--mutate=radius`, `noproj`, `dual` | exit 1, 1, 1 | all fire |
| P_gap | P_hi/P_lo − 1 is 3.4×10⁻⁵ to 6.8×10⁻³, largest for arcjump at n = 31 | OK |
| P_valid | P_lo < P_hi everywhere; χ at 250 digits at most 0.999999999; P_lo at 250 digits equal to 150 in double precision | OK |
| P0, chord2's smallest peak against its own | 0.922–0.929 at every n and node | OK |
| `peak --mutate=scale` | P/P_own = 0.691–0.697, P0 fails at every n and node; exit 1 | fires |
| separation at n = 31 | ρ = 1.45–1.82 | no peak prior separates |
| `peak_real` | refused by its frozen rule, since ρ_lo ≤ 3 at n = 23; exit 1 | not run |

The smallest peak density of the chord twin on s = j/200, as the bracket [P_lo, P_hi] rounded to three decimals. ρ is
arcjump's smallest against chord2's smallest, from ρ_lo to ρ_hi.

| ω/ω∞ | n | chord2's own peak | chord2 | arcjump | arc | chord2 / own | ρ | arcjump / chord2's own |
|---|---|---|---|---|---|---|---|---|
| 0.99 | 23 | 2.870 | 2.654 | 4.411–4.413 | 2.460–2.461 | 0.925 | 1.662–1.663 | 1.54 |
| 0.99 | 31 | 2.870 | 2.667 | 4.810–4.843 | 2.467 | 0.929 | 1.804–1.816 | 1.69 |
| 1.00 | 23 | 2.922 | 2.698–2.699 | 4.088–4.089 | 2.529 | 0.923 | 1.515 | 1.40 |
| 1.00 | 31 | 2.922 | 2.711 | 4.392–4.404 | 2.537–2.538 | 0.928 | 1.620–1.625 | 1.51 |
| 1.01 | 23 | 2.998 | 2.763 | 3.830–3.831 | 2.647 | 0.922 | 1.386–1.387 | 1.28 |
| 1.01 | 31 | 2.998 | 2.776 | 4.023–4.045 | 2.658 | 0.926 | 1.449–1.457 | 1.35 |

Context, not judged:

- **The finer grid.** On s = j/1000 every minimum is within 0.2% of the one above, chord2's ratio to its own peak is
  again 0.92–0.93, and ρ at n = 31 is 1.45–1.82. The twins found on s = j/200 do bulge between its points, mostly at
  s = 0.0125: on s = j/2000 they peak 7–25% above their grid peak, and 56% above it for arc at ω/ω∞ = 1.01, n = 31.
  The twins found on s = j/1000 stay within 2.7% of theirs.
- **Their coefficients are huge.** ‖c‖ is 5×10⁶ to 2×10⁷ for every smallest-peak twin. These coefficients make a
  polynomial that is large only near s = 1, where (1 − s)^ex hides it from both the grid and the moments. So the peak
  and the coefficient norm of §8 measure different things.
- **M = 50 at ω/ω∞ = 1.00** lowers chord2's and arc's smallest peaks by 0.16–0.28%, and arcjump's by 1.7%.
- **Active constraints.** 31–38 grid points and 10–14 of the n residuals are active at each minimum.

### 9.6 What this means

- **No twin class reproduces the real series with fewer terms than coefficients.** The chord, the stepped arc and the
  smooth arc each need all 23 terms at every node, while each plant has a twin of 14–19 terms in another class. The
  classes share the plants' end points, exponent and zero factor, so the real σ₁₁–σ₂₂ are not of that form at the
  measured errors. Which ingredient fails is not determined here. By term count, neither the chord nor the arc is
  preferred.
- **The peak density passes the calibration that the coefficient norm failed.** chord2's smallest twin peaks at
  0.92–0.93 of chord2's own density, so at 23 or 31 coefficients no chord twin of chord2 peaks more than 8% below
  chord2 itself. By coefficient norm the smallest twin was 40–44% below chord2's own (§8.5).
- **No prior on the peak separates a chord from an arc, by the frozen rule.** The arc plant's smallest chord twin
  peaks at 1.45–1.82 times chord2's smallest at n = 31, and 1.39–1.66 at n = 23. Against chord2's own density it is
  1.28–1.69. So a prior would have to cap the peak within a factor 1.3–1.7 of the true density's, which supposes the
  density's size known to that accuracy. With 8 more coefficients the margin grows by 5–9%, with their errors
  extrapolated (§7.1).
- **A smooth density on the arc has a chord twin smaller than chord2.** The plant arc's smallest chord twin peaks at
  0.92–0.96 times chord2's smallest, and 0.86–0.89 times chord2's own density.
- **Status.** Numerical evidence on plants, with exact brackets for each minimum (within 0.7%) at the measured errors
  and #670's extrapolated ones. The real series was not run in part 2, by the frozen rule.

### 9.7 Proposed text for proof/D_entire.md

In 7.4, replacing the last three sentences of the bullet "Nor would a prior on the coefficients' size (#669)" and
adding two bullets after it:

```diff
   chord2's own coefficients does not need it. The other way round, a density on the arc with log branch points along
-  it reproduces chord2 with 18 terms. Whether a prior on the peak density separates them is open, since the known
-  chord twins of the arc plant peak at 2.3–3.8 times chord2's density. On the real series at ω/ω∞ = 0.99, no chord
-  twin with fewer than 23 terms reproduces σ₀–σ₂₂ (χ 9.7×10⁴ at 22 terms), where the arc plant needs 19; one node,
-  and the real stage stopped there on a control (§8).
+  it reproduces chord2 with 18 terms.
+- **Nor would a prior on the peak density (#672).** Minimised exactly on #548's grid, chord2's smallest chord twin
+  peaks at 0.92–0.93 times chord2's own density, so this measure passes the calibration that the norm failed. The arc
+  plant's smallest chord twin peaks at only 1.45–1.82 times chord2's at 31 coefficients and 1.39–1.66 at 23, so by
+  the factor-3 rule no peak prior separates them (notes/PADE_FAR_END.md §9). A grid five times finer moves these
+  minima by at most 0.2%.
+- **The real series has no short twin in any of these classes (#672).** No chord, stepped-arc or smooth-arc density
+  with the plants' end points P0 and κ⋆ and their exponent reproduces the real σ₀–σ₂₂ with fewer than 23 terms, at any
+  node (χ 0.9–1.3×10⁵ at 22 terms), while each plant has a twin of 14–19 terms in another class. The misfit sits at
+  σ₁₁–σ₂₂. Which ingredient fails, the far end or the exponent, is not determined (notes/PADE_FAR_END.md §9.2).
```

## 10. Precision of the real inputs (#683)

The scripts of sections 1–9 read the real σ₀–σ₂₂ from the strings (`sigma_str`) of the scan files, which carry 20
significant digits. #672 found that `pade_twin_count.real_sig()` held them at mpmath's default 53 bits (§9.2).
`solver/inner_amp/sigma_load.py` now parses them at 300 digits for every caller, inside `mp.workdps` and never through a
float. `solver/inner_amp/pade_precision.py` measures what the earlier loading did, and what the strings' own rounding
leaves. All runs used one process under `nice -n 10`, single-threaded, 2026-11-13 12:13–12:28 UTC. The records are in
the raw-data repository's `I683-mac/` (4782261).

### 10.1 The callers

Each row is a stage that loaded the real series, with the precision its strings were parsed at before #683. Every row
now parses at 300 digits.

| entry point | stages (task) | parsed at | what the values feed |
|---|---|---|---|
| `pade_geometry.load_scan` | `pade_geometry.py plants` (#541) | 53 bits | the plants' template σ₀–σ₁₂ |
| | `pade_geometry.py real` and `selftest` (#541) | 60 digits | the real [k/k] poles at 120 digits; the selftest's template |
| | `pade_farend.py`, every stage (#547), and `pade_farend_diag.py` | 100 digits | the plants' template; the real DA2 at 100 digits |
| | `pade_chord_arc.py`, every stage (#548) | 100 digits | the plants' template |
| | `pade_twin_size.py twins_real`, its J 1500 context line (#669, #672) | 60 digits | J 1500 against J 1250 |
| | `twins_jdiag` and `peak_real` (#672) | 150 digits, explicitly | a diagnostic; a gated stage, not run |
| `pade_twin_count.real_sig` | `pade_twin_count.py` (#670), `pade_twin_size.py twins` and `size` (#669, #672), `pade_twin_peak.py` (#672) | 53 bits | the plants' template |
| | `pade_twin_size.py twins_real` (#669, #672) | 53 bits | the targets of the real twin fits at 150 and 250 digits |
| `planted_plants.load_real` | the P-ARC families (#541's `repo-arc` and `repo-chord`; notes/PLANTED_SELECTION.md) | 50 digits | their template |
| `sigma_err_fit.recompute`, through `pade_geometry.noise_eps` | every stage with the measured errors | 40 digits | \|σ_k\| as a double, in err_k only; not switched |

`twins_jdiag` keeps its 53-bit reading, now explicit (`sigma_load.load_scan(..., prec=53)`), since it documents #672's
values. The other loaders of the scan files lie outside these scripts and were not switched. They parse at their
module's 30–80 digits, above the strings' 20 (`solver/proof/exact_sigma.py --borel`, `entire_cert.py`,
`homega_interp.py`, `feasibility.py`, `zero_selection.py`, `planted_estimators.py`, `solver/plot_selection.py`,
`solver/harmony/stokes/`, `solver/independent_second_zero/`), or they use doubles by design (`trunc_seq_M7.py`,
`fit_arc.py`, `check_numbers.py`).

### 10.2 The load check

`python3 sigma_load.py --check` calls each entry point at mpmath's default 53 bits, as a script's first load does. Every
nonzero component of every loaded σ_k must equal its string, parsed again at 600 digits, within 10⁻¹⁰⁰ relative. It
covers the J 1250 files at the three nodes, the J 1500 file and the `scan25phys` files of the P-ARC families.

| run | largest relative error | verdict |
|---|---|---|
| `--check` | 6.0×10⁻³⁰² to 8.6×10⁻³⁰² on every path; exit 0 | OK |
| `--check --mutate=53bit`, the loader parsing at 53 bits | 6.9×10⁻¹⁷ to 1.1×10⁻¹⁶ on every path; exit 1 | fires |

### 10.3 What a 53-bit template did to the plants

The plants' c_m are a least-squares fit in double precision to the real σ₀–σ₁₂ (§2.2), so the template's precision
enters through the rounding of the fit's inputs. `pade_precision.py plants` refits chord2, arcjump and arc
(`pade_farend.FarPlant`) and chord2 and arcjump (`pade_geometry.Plant`) at each node, with the template parsed at 53
bits, 60 digits and 100 digits, against 300 digits.

| template | c_m that differ | largest change of c_m | of σ₀–σ₃₀ | of σ₁–σ₂₂, in error units |
|---|---|---|---|---|
| 60 or 100 digits | 0 of 6, every plant and node | 0 | 0 | 0 |
| 53 bits | 6 of 6, every plant and node | 1.0×10⁻¹³ to 3.4×10⁻¹² relative | 2.5×10⁻¹⁴ to 3.8×10⁻¹³ relative | 1.3–5.1×10¹⁰ |

So the plants of #547 and #548 are unchanged, and those of #541, #669, #670 and #672 change at about 10⁻¹³ relative.
Before #683, `pade_twin_count.py`'s chord2 was therefore a different function from `pade_farend.py`'s, by 10¹⁰ error
units at low order. Now they are the same. Both are plants of the same family, so this is a change of plant, not an
error. What it does downstream is in 10.4.

### 10.4 Reruns at full precision

Each stage below was rerun with its own script and compared, number by number, with its stored record
(`pade_precision.py compare`, `I683-mac/compare/`). At M = n the twin is the interpolant and its χ is rounding noise,
so those rows and C1's ratios there are left out (`--skip`).

| stage | record | numbers that differ | largest relative change |
|---|---|---|---|
| `pade_farend.py real` | `I547-mac/real.json` | 0 of 305 | 0 |
| `pade_farend_diag.py` | `I547-mac/diag_post_freeze.json` | 0 of 675 | 0 |
| `pade_geometry.py real`, frozen and `--noise=measured` | `I541-mac/real*.json` | 39 of 1736, all of them distances to the plants in the verdict | 6.8×10⁻¹¹ |
| `pade_geometry.py plants`, frozen and `--noise=measured` | `I541-mac/plants*.json` | 3294 of 4367, and 3682 of 4808 | 6.6×10⁻⁹ (a pole offset of 5.6×10⁻⁵); 1.5×10⁻⁶ for a noise spread |
| `pade_twin_size.py twins`, the plants | `I672-mac/twins_plants.json` | 800 | 3.6×10⁻¹² |
| `pade_twin_size.py twins_real` | `I672-mac/twins_real.json` | 783 | 5.1×10⁻⁸ |
| its J 1500 context line | the same | | χ 1.7×10⁹ → 1.01, at k = 16 |

Every verdict matches its record. `pade_geometry.py plants` and `real` exit 1 on their frozen verdicts, which are
#541's (T1–T4 fail on the plants, and the real series converges at no node), and the other stages exit 0. The real
Padé poles, R_k, bows, ratio-law κ and noise spreads of #541, and every number of #547's real and diagnostic runs, are
bit-identical in double precision, since those stages parsed at 60 and 100 digits. The J 1500 line now gives what
§9.2's diagnostic found, at most 1.0 error unit, at k = 16.

**No quoted real-series number moves by its last digit.**

| quoted number | where | quoted | rerun |
|---|---|---|---|
| R_k of the real series at ω∞, k = 6–11 | D_entire 7.4 | 1.000, 0.826, 0.701, 0.405, −0.057, 0.33 | bit-identical |
| the real [6/6] bow | D_entire 7.4 | 0.116 | bit-identical |
| the poles past κ⋆ at ω∞, k = 8, 9, 10 | D_entire 7.4, §1 | 1.355 + 1.051i, 1.407 + 1.061i, 1.457 + 1.069i (\|t\| 1.715, 1.762, 1.806) | bit-identical |
| the ratio law over k = 12–21 | D_entire 7.4, §1 | 1.457 + 1.112i, \|κ\| = 1.833 | bit-identical |
| the DA2 median, modulus, spread and noise, per node and J 1500 | §4.2 | at ω∞ 2.228 + 1.435i, 2.650, 0.24, 0.19 | bit-identical |
| the DA2 zeros past \|t\| = 1.65 | §4.3, D_entire 7.4 | no 6 members agree; none within 0.17 of the far poles | bit-identical |
| M_min = M_lo of the real chord, arc and smooth twins | §9.2, D_entire 7.4 | 23 at every node | 23 at every node |
| their χ at 22 terms | §9.2, D_entire 7.4 | 9.7×10⁴ to 1.27×10⁵ (0.9–1.3×10⁵) | the same to 7 digits |
| the J 1500 twins' χ at 22 terms | §9.2 | 9.1×10⁴ and 1.15×10⁵ | the same |

The plant stages of #548 are unchanged exactly, since their plants are (10.3). Those of #669, #670 and #672 part 2
(`size`, `run`, `peak`) were not rerun. On the same plants, `twins` moved by at most 3.6×10⁻¹² relative, so their
quoted three- and four-digit numbers are not expected to move. That is an inference, not a rerun.

### 10.5 The floor that the 20-digit strings put on each statistic

`amp_flint_par.py` writes σ_k with `mp.nstr(z, 20)`, and its log prints 15 digits. The raw-data repository holds those
two for every J 1250 and J 1500 run at these nodes (`T99-win2/`, `T102-mac/`, `T103-mac/`; the repository's wf 1.00
J 1250 file is T99-win2's, whose strings equal T102-mac's). The certified σ₁ of D_entire 6.5 is an enclosure of
half-width 3.5×10⁻⁵.

Files with more digits exist, from `scan_omega.py`, which writes 25 (`pade_precision.py strings`). Of the 78 files of
`solver/inner_amp/` with σ_k at these nodes, 27 have 25 digits, all at lower J. The J 1000 `--fit phys` scan
(`scan27J1000`, 640 bits, σ₀–σ₁₃) is the closest. It agrees with the J 1250 strings within their rounding at
k = 1–7 (0.02–0.72 rounding units) and departs from them by 69–1900 units at k = 8. The J 750 scans agree at k ≤ 3,
not at every node at k = 4, and depart by 15–260 units at k = 5–6. Nothing measures the J 1000 to J 1250 difference
below the rounding, and at k = 8 it is already 69–1900 times the rounding, so those extra digits are not the J 1250
values' and were not used. The strings are therefore the floor. Their rounding is half a unit
in the 20th digit (`sigma_load.py --rounding`). Against the measured errors, all relative:

| ω/ω∞ | measured error, k = 1–8 | string rounding, k = 1–8 | rounding over error, k = 1, …, 8 | k = 9 |
|---|---|---|---|---|
| 0.99 | 3.0×10⁻²⁶ to 5.5×10⁻²¹ | 7.8×10⁻²¹ to 4.5×10⁻²⁰ | 2.6×10⁵, 2.8×10⁵, 4.0×10⁵, 1.1×10⁵, 870, 940, 24, 3.4 | 0.089 |
| 1.00 | 3.3×10⁻²⁶ to 2.6×10⁻²¹ | 7.8×10⁻²¹ to 4.5×10⁻²⁰ | 2.3×10⁵, 2.5×10⁵, 3.6×10⁵, 1.4×10⁵, 780, 2100, 22, 7.1 | 0.079 |
| 1.01 | 3.7×10⁻²⁶ to 2.6×10⁻²¹ | 7.7×10⁻²¹ to 4.6×10⁻²⁰ | 2.1×10⁵, 2.3×10⁵, 3.2×10⁵, 1.2×10⁵, 1700, 2200, 14, 7.4 | 0.070 |

The rounding exceeds the measured error at k = 1–8 at every node, and from k = 9 on it is below it. The J 1500 file
rounds the same way. It equals J 1250 exactly at k ≤ 9, because both round to the same 20 digits there, so the J
difference says nothing about those orders.

The floor of a statistic is its first-order worst case over the box the rounding leaves: each component of σ₁–σ₂₂
anywhere within half a unit of its 20th digit (σ₀ = 1 by normalisation). `pade_precision.py floor` takes one finite
difference per component at the working precision and sums the moduli. It is a first-order bound, not an enclosure.
Its control evaluates the corner of the box that maximises Re t of the pole past κ⋆ at k = 10, and the one for R₁₀, at
ω∞. The actual shifts, 1.3624×10⁻⁸ and 2.7579×10⁻⁷, equal the bounds to five digits, and `--mutate=corner`, which
doubles the corner, gives the ratio 2.00000 and fails.

| statistic | floor over the three nodes and J 1500 | quoted to | noise spread at the measured errors |
|---|---|---|---|
| [k/k] geometry poles, k = 6–9 | ≤ 2.1×10⁻⁹ | 3 decimals | |
| the same, k = 10 and 11 | ≤ 9.5×10⁻⁸ and ≤ 9.5×10⁻⁷ | 3 decimals | |
| the poles past κ⋆, k = 8, 9, 10 | ≤ 1.2×10⁻¹⁰, 1.1×10⁻⁹ and 2.2×10⁻⁸ | 3 decimals | |
| their pole–zero gaps | ≤ 9.4×10⁻¹² | 1 digit | |
| R_k, k = 6–9 | ≤ 1.2×10⁻⁸ | 3 decimals | |
| R₁₀ and R₁₁ | ≤ 3.3×10⁻⁷ and ≤ 3.5×10⁻⁶ | 3 and 2 decimals | 0.05 and 0.47 at ω∞ |
| the bow at k = 6 | ≤ 1.9×10⁻¹² | 3 decimals | |
| each DA2 member's farthest zero, and so the median | ≤ 4.4×10⁻⁶ | 3 decimals | 0.13–0.19 |
| χ of the real chord, arc and smooth twins at 22 terms | ≤ 2.3×10⁻⁶ error units, largest at k = 16–18 | 2–3 digits, of about 10⁵ | |

`pade_precision.py floor_twins` gives the twins' row. The residual of a least-squares twin is linear in its target,
r = (I − QQᵀ)Wσ with Q from mp.qr, so its shift over the box is exact to first order. Two controls pass at every node
and class. χ from Q equals `tc.lsq`'s, with a relative difference of 0 in double precision. A refit by `tc.lsq` with Re
σ₃ moved by 10⁶ half units matches Q's prediction to 10⁻¹⁰⁴. `--mutate=proj`, which drops Q's last column, makes both
fail, by factors of 2.4–9.3 and 10⁹–10¹⁰, and exits 1. With err_k widened to max(err_k, rounding_k), every class still
has M_min = M_lo = 23 at every node, and χ at 22 terms is unchanged at three digits.

### 10.6 What this means

- **The 53-bit loading moved no quoted number.** It reached the real series only through `twins_real`, whose χ it
  moved by 5×10⁻⁸ relative, and through the J 1500 context line, which §9.2 had already flagged as an artifact. Its
  larger effect was on the plants. It changed them by 10⁻¹³ relative, at 10¹⁰ error units, and their statistics by at
  most 7×10⁻⁹ relative (1.5×10⁻⁶ for a noise spread).
- **The strings are coarser than the errors at k = 1–8, and no statistic here notices.** The measured errors at
  k = 1–4 lie 10⁵ times below the rounding, so the true input error at k = 1–8 is the rounding, and a fit that weights
  those orders by the measured errors treats up to 4×10⁵ error units of rounding as signal. But every statistic here is
  set by the higher orders. Over the whole box the Padé poles and DA2 zeros move by at most 4.4×10⁻⁶ in t, R_k by at
  most 3.5×10⁻⁶, and the real twins' χ by at most 2.3×10⁻⁶ error units, and the twin verdict survives errors widened to
  the rounding. A statistic that leaned on σ₁–σ₈ at their measured errors would need more digits. None of those quoted
  here does.
- **More digits would need a rerun.** `amp_flint_par.py` computes σ_k at 1280 bits and keeps 20 digits. A rerun that
  writes them in full would lift the floor, and nothing quoted here needs it.

### 10.7 Proposed text for proof/D_entire.md

No number of 7.4 changes. One sentence would record the inputs' precision, at the end of 7.4's first paragraph:

```diff
 `solver/inner_amp/pade_geometry.py` computes the [k/k] poles for k = 6–11 from σ₀–σ₂₂ (`scan45J1250`, 120 digits). It
 does the same for continuous plants matched to σ₀–σ₁₂ to the conventions of notes/PLANTED_SELECTION.md 2.2.
+The files give σ_k to 20 digits, coarser than their measured errors at k ≤ 8. That rounding moves the poles, R_k and
+the differential approximants' zeros by less than 10⁻⁵, and the twins' χ by less than 10⁻⁵ error units
+(notes/PADE_FAR_END.md §10).
```

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

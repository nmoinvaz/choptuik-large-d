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

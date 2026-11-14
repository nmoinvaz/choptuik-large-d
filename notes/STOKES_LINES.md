# STOKES_LINES: do the zeros of S₁ follow a two-endpoint law in the amplitude down to the physical one? (#357)

The open test of DINF_ANALYTIC.md §5a: "whether S₁ has a two-Stokes-line form in some large parameter (the harmonic
index, or the amplitude), with zeros that continue to the physical one." This note fixes the test before any
computation (§1–§4, committed first). The results follow in §5.

## 1. The large parameter and the law

The large parameter is the amplitude, x = A². The harmonic index is not tested, since no multipliers of high harmonics
exist in the form the test needs. S₁ is entire in x of order 1 (DINF_ANALYTIC.md §2.3), a Laplace integral of its Borel
transform over an arc from a near endpoint ζ ≈ 0.75 + 0.16i to a far endpoint κ. If each endpoint contributes one
algebraic term, then for large |x|

```math
S_1(x) \simeq C_\kappa\,e^{\kappa x}x^{-a} + C_\zeta\,e^{\zeta x}x^{-b},
```

and the zeros satisfy the two-endpoint law

```math
\Lambda\,x_j + q\,\ln x_j = L + 2\pi i\,j, \qquad \Lambda = \kappa - \zeta,\quad q = b - a,
```

with three complex constants (Λ, q, L) and the principal logarithm. HARMONY.md item 1 and
`solver/harmony/stokes/ratio_law_zeros.py` (b) used this law with Λ and q taken from endpoint estimates, to predict the
second zero from the first, with a miss of 0.13–1.27 depending on those estimates. Here the constants are fitted on
zeros of larger modulus, and the law is asked to predict the physical zero.

## 2. The question, as a test

**H.** At ω = ω∞, the zeros x₁, x₂, x₃ of the arm defined below determine (Λ, q, L), and the law at j = 0 returns
the physical zero x₀ = 7.39408 (A₀ = 2.7192).

- **Arm rule** (applied identically to every function, exact or summed):
  - x₁ is the zero with 10 < Re x < 16 nearest the real axis;
  - x₀ is the zero with 4 < Re x < Re x₁ − 2 nearest x₁ − 5.4;
  - x₂ is the zero with Re x > Re x₁ + 2 nearest 2x₁ − x₀;
  - x₃ is the zero with Re x > Re x₂ + 2 nearest 2x₂ − x₁.
- **Fit.** (Λ, q, L) solve the law exactly at j = 1, 2, 3 (a 3×3 complex linear system). The prediction x̂₀ solves
  Λx + q ln x = L by Newton from x₁ − (x₂ − x₁).
- **Summation.** A function given by σ₀…σ_K is summed by Borel–Padé [m/(K − m)], with m from ⌈0.35K⌉ to ⌊K/2⌋ +
  1, and its zeros in |x| < 30 are found by `zerolib.zeros_in_disc` (h = 0.2). A sum enters only if the arm rule
  finds all four zeros in it. x̂₀ is the median over the sums that enter, and its spread is their range.
- **Error.** e = |x̂₀ − x₀|, where x₀ is the exact zero for a planted function and 7.39408 for the data.

## 3. Planted functions and controls (fixed now)

- **P1, realistic.** The arc family of #214 (`solver/independent_second_zero/arc_family.py`), fitted to the data's
  σ₁…σ₂₂ at ω/ω∞ = 1.09–1.11, taken at w = 1.10. Its curved arc, non-Beta density and square-root branch point make it
  the closest planted function the project has. Exact coefficients to n = 200 at 160 digits; exact zeros from that
  series.
- **P2, two plain endpoints.** The straight segment from ζ = 0.749 + 0.155i to κ = 1.703 + 1.127i with density
  (t − ζ)^{α−1}(κ − t)^{β−2}(1 + g(t − ζ)), α = 1.5 + 0.5i, β = 9.74 − 5.81i (the ratio law of real_zeros.py at
  k = 12–21), g = 0.3 − 0.2i. Exact moments by quadrature at 60 digits; exact zeros by Newton on the quadrature.
- **Noise.** Each coefficient gets complex relative noise of 10⁻¹⁴ (fixed seed), the J truncation level of the data
  (#213).
- **Control C (must fire).** P1 times (x − x₀ − d)/(x − x₀), with d = 1.0, which moves its physical zero by 1.0 and
  changes the large-|x| behaviour only at O(1/x). The test must report that the displaced zero is not on the law's
  sequence: |x̂₀ − (x₀ + d)| > 3e₁, where e₁ is P1's error at the same K.

## 4. Criteria (fixed now)

1. **Decisive at K.** The test is decisive at K when P1 and P2 both give e ≤ 0.15 (a fifth of the control's d, with
   the factor 3 of item 3 still below d/2), and control C fires. Otherwise the verdict at K is "inconclusive", and the
   planted runs at K = 30 and 40 (exact coefficients are free) give the K at which it becomes decisive.
2. **The data.** Run only at a decisive K. H holds if e_data ≤ 3 max(e₁, e₂, spread_data). H fails if e_data exceeds
   that bound.
3. **The data available.** σ₀…σ₂₂ at ω∞ (K = 22: K 45 J 1500 1280 bits,
   `solver/inner_amp/scan45J1500/sigF_w1.0000_K45_J1500_p1280.json`). If the test needs K > 22, I stop and report the
   amplitude run it needs, with its command and an estimate, since it exceeds the Mac's share.
4. **Seen before fixing these criteria.** The Borel–Padé zeros of the data in |x| < 30 (`real_zeros_R30.json` of
   `harmony-mac/stokes/`). By the arm rule they give x₁ = 12.75 + 1.12i and x₂ = 16.81 + 1.78i in [10/12] and [11/11],
   but 17.07 + 1.34i in [8/14]. The candidates for x₃ differ by more than 1 between sums. The test has not been run on
   them.

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

## 5. Results (2026-11-14)

**Verdict: inconclusive. The test as fixed cannot decide, at any number of coefficients, so the data run was not
made (§4, item 2).** Its limit is not the 23 coefficients at hand. With exact zeros, the two-endpoint law fitted on
x₁–x₃ misses the realistic planted function's x₀ by 0.50, against the fixed bound of 0.15. The physical zero lies where
the law's corrections are still large. Both planted functions do follow the law asymptotically: its fitted Λ tends to
their true endpoint difference as the fit moves out. Their x₀ is the j = 0 member of the sequence, at 0.08–0.6 from
the law's prediction against a spacing of about 5.5. But the law is not accurate enough there to tell a zero on the
sequence from one displaced by 1.0, which is what the test asks of it. Script: `solver/harmony/stokes/stokes_lines.py`;
logs in the raw-data repository's `I357-mac/`.

### 5.1 The planted runs (fixed in §3–§4)

The law's prediction x̂₀ is the median over the Borel–Padé sums; e = |x̂₀ − x₀|; the control is P1 with x₀ moved by
d = 1.0, and it fires when |x̂₀ − (x₀ + d)| > 3e₁.

| K | e₁ (P1, arc) | spread₁ | e₂ (P2, segment) | spread₂ | control | decisive |
|---|---|---|---|---|---|---|
| exact zeros | 0.504 | — | 0.077 | — | — | — |
| 22 | 3.46 | 4.53 | 1.38 | 0.89 | 2.85, does not fire | no |
| 30 | 0.481 | 0.98 | 0.017 | 0.088 | 0.881, does not fire | no |
| 40 | 0.501 | 0.018 | 0.076 | 0.0012 | 0.580, does not fire | no |

- **K = 22.** Every sum places x₀ and x₁ to three decimals. For P1, x₂ is off by 0.03–2.1 and x₃ by 1.7–5.3, and the
  3×3 fit turns that into |q| of 21–98 against the exact 8.0. The data's sums show the same pattern (§4, item 4).
- **K = 40.** The sums reproduce the exact arm and the exact-zero prediction (P1: x̂₀ = 8.209 − 1.000i with spread
  0.018, against 8.210 − 1.004i from exact zeros). The remaining error is the law's own.
- **The checks of P2's two routes.** Its fifth moment by quadrature, and S(9 + 2i) by Kummer functions, agree with the
  series to 1.5×10⁻⁵¹ and 2.3×10⁻⁵¹.

### 5.2 Post hoc: the law's error from exact zeros (not part of the fixed test)

`stokes_lines.py explore`, on the exact arms. The arm in |x| < 30 has five zeros, x₀ to x₄.

| fit | P1: e at x₀ | P1: Λ | P2: e at x₀ | P2: Λ |
|---|---|---|---|---|
| law on x₁–x₃ | 0.504 | 0.932 + 1.357i | 0.077 | 0.951 + 1.025i |
| law on x₂–x₄ | 0.644 | 0.920 + 1.333i | 0.160 | 0.955 + 1.006i |
| law + c/x on x₁–x₄ | 0.359 | 0.887 + 1.301i | 0.016 | 0.944 + 0.974i |
| true endpoint difference | | 0.914 + 1.325i | | 0.954 + 0.972i |

- **The law's form is right.** Λ from x₂–x₄ is 0.7% from P1's true zp − za, closer than from x₁–x₃. So the zeros are
  those of two endpoint contributions, as the hypothesis assumes.
- **Extrapolating down is not.** Fitting farther out makes the prediction at x₀ worse (0.64 for P1), since x₀ is then
  farther from the fitted zeros. A 1/x term helps (0.36 for P1), but to reach 0.15 the law needs more correction terms,
  and each term needs one more zero.
- **The control machinery can fire.** With d = 3 at K = 40 (run post hoc; the fixed d is 1.0) the control gives
  |x̂₀ − (x₀ + d)| = 2.55, against 3e₁ = 1.50, so it fires.

### 5.3 What a decisive test would need

1. **A law with corrections, tested on planted zeros first.** The two-endpoint asymptotic series to O(x⁻²) or more,
   one unknown per order, fitted on P1's exact zeros out to |x| = X. The question is the X at which P1's x₀ comes out
   within 0.15 and the d = 1 control fires. That needs exact zeros of the arc family beyond |x| = 30, by the same
   200-term series at 160 digits (P1's coefficients grow like 2.2ⁿ, so |x| = 40 still converges, with more terms).
   Hours on one core.
2. **The coefficients the data would then need.** At K = 40 the sums reproduce the exact zeros to |x| ≈ 26 with
   spread 0.02. If the resolved radius grows about linearly with K, |x| = X needs K ≈ 1.5X: K ≈ 60 for X = 40. That is
   σ₀…σ₆₀, an amplitude run of order 121, against the K 45 run at ω∞ that took 59 893 s with 3 processes on win2
   (T99-win2/scan45J1500/w1.0000.log). By that step, it exceeds the Mac's share many times over, and it is worth
   making only if item 1 finds a finite X.

The harmonic index as the large parameter was not tested (§1).

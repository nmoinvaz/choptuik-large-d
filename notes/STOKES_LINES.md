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

## 6. The law with 1/x corrections, on planted zeros (#773)

**On exact zeros, the corrected law decides at X = 35. Its order is 3, fitted on the six arm zeros x₁…x₆
(|x| ≤ 34.6).** There P1 gives e = 0.012, P2 gives e = 0.002, and the d = 1 control fires. The law cannot yet be
applied to the data, though, because it amplifies errors in the zeros it is fitted on by about 10⁴. Script:
`solver/harmony/stokes/stokes_series.py`. Raw: `I773-mac/` of the raw-data repository (ffb1dd8d). The X and the order
were pre-registered on #773 before any zero beyond |x| = 30 was computed.

**The law of order N** adds the 1/x series of the two endpoint contributions to §1's law:

```math
\Lambda x_j + q\ln x_j - L + \sum_{k=1}^{N} c_k x_j^{-k} = 2\pi i\,j .
```

The c_k are the coefficients of the logarithm of the ratio of the two endpoints' series. The law has N + 3 unknowns
and is fitted exactly on x₁…x_{N+3}. x̂₀ solves it at j = 0, as in §2.

The order at each X is N(X) = n(X) − 3, where n(X) is the number of P1's arm zeros within |x| ≤ X.

**Zeros.** They come from the 200-term series at 160 digits, searched in |x| ≤ 46 with Re x ≥ 2, in arb at 640 bits.
- In every function, the number of zeros found equals the winding on that region's boundary.
- The arm zeros move by at most 6×10⁻¹⁷ at 300 terms.
- At x₇ (|x| = 38.6), quadrature of the arc integral (P1) and Kummer functions (P2) confirm the zero to 6×10⁻¹⁷
  in |S|/|S′|.
- The fitter recovers x₀ from an arm generated by a known order-3 law to 5×10⁻⁵⁷. With its ln x column removed, it
  misses by 0.43.

| X | N | e₁ (P1) | e₂ (P2) | control: \|x̂₀ − (x₀ + 1)\| against 3e₁ | decisive |
|---|---|---|---|---|---|
| 30 | 1 | 0.359 | 0.016 | 0.77 against 1.08, does not fire | no |
| 35 | 3 | 0.012 | 0.0020 | 1.01 against 0.035, fires | yes |
| 40 | 4 | 0.132 | 0.0010 | 0.87 against 0.40, fires | yes |

With d = 0, the control never fires.

- **Post hoc: the law's error is not monotone in its order for P1.** For N = 0…5, e is 0.50, 0.36, 0.22, 0.012, 0.13
  and 0.21. The arc family at w = 1.09 and 1.11 behaves the same way (e = 0.006 and 0.018 at N = 3). So the order has
  to be fixed at 3 on x₁…x₆, not taken as high as the zeros allow. P2's error falls at every order, to 0.0006 at
  N = 5.
- **The control adds nothing on exact zeros.** C's arm beyond x₀ is P1's, so it fires whenever e₁ < 1/4. It checks
  the code path only.
- **Post hoc: what the data would need.** At N = 3, |∂x̂₀/∂x_j| is 81, 698, 2230, 3330, 2370 and 646 for P1's
  x₁…x₆, 9.4×10³ in all. To keep x̂₀ within 0.15, the data's zeros out to |x| = 35 must therefore be accurate to about
  2×10⁻⁵. §5.1's sums at K = 40 give zeros to |x| ≈ 26 with a spread of 0.02. So §5.3's estimate of K ≈ 1.5X (52 here)
  does not settle the order of the amplitude run. The K that gives such zeros is to be measured on P1's sums first.
- **A pitfall of `stokes_lines.displaced`.** For zeros at large |x|, C needs x₀ far beyond the series' 160 digits. A
  60-digit x₀ left a term ε Σ xⁿ/x₀ⁿ⁺¹ with ε = S(x₀), which gave C 46 zeros in the region instead of 37. The
  winding check caught it, and stokes_series.py refines x₀ to 320 digits. §5's runs used C only through σ₀…σ₄₀,
  and whether the 60-digit x₀ mattered there was not checked.

## 7. The Borel–Padé order the corrected law needs (#788)

**At the data's coefficient noise of 10⁻¹⁴, no K ≤ 100 suffices.** At K = 100 every Borel–Padé sum of P1 gives
e₁ = 2.13 against the bound of 0.15, and the control C does not fire at any K. Script:
`solver/harmony/stokes/stokes_bp.py`. Raw: `I788-mac/` (d2ff76f8). The test was pre-registered on #788 before any
sum was computed.

**Method.** P1's σ₀…σ_K, and C's, carry 10⁻¹⁴ relative noise (§3). They are summed by Borel–Padé
[m/(K − m)] over §2's range of m. Each sum's zeros in |x| ≤ 42, Re x ≥ 2 are found in arb and checked by the argument
principle. The arm must reach x₆, and §6's order-3 law on x₁…x₆ gives x̂₀. K is decisive when every P1 sum that
enters has e₁ ≤ 0.15 and every C sum that enters has |x̂₀ − (x₀ + 1)| > 3 max e₁.

| K | P1 sums entered | e₁ | median error of x₅, x₆ | decisive |
|---|---|---|---|---|
| 40 | 5 of 8 | 5.1–6.1 | 5.1, 12 | no (the control, as required) |
| 60 | 10 of 11 | 4.2–5.2 | 8×10⁻³, 0.42 | no |
| 80 | 12 of 14 | 1.2–2.9 | 1.2×10⁻⁴, 1.5×10⁻² | no |
| 100 | 16 of 17 | 2.13 | 2.0×10⁻⁴, 2.4×10⁻² | no |

- **Post hoc: the noise sets the floor, not K.** From K = 80 to 100 the outer zeros stop improving. At K = 100 their
  errors scale with the noise: x₆ is off by 2.5×10⁻⁸ at 10⁻²⁰, by 2.5×10⁻⁵ at 10⁻¹⁷ and by 2.4×10⁻² at 10⁻¹⁴, the same
  in every sum. With exact coefficients, the sums reproduce the exact zeros to 10⁻²⁴ and e₁ = 0.0117, §6's value.
- **Post hoc: what a data test would need.** At noise 10⁻¹⁷ every K = 100 sum gives e₁ ≈ 0.011. K = 80 does not:
  its e₁ runs from 0.06 to 5.3 across the sums. So σ₀…σ₁₀₀ would be needed to about 10⁻¹⁷, a thousand times below the
  J truncation of the data in hand (#213), from an amplitude run of order about 200. By §5.3's cost, that is far beyond
  the runs made so far.

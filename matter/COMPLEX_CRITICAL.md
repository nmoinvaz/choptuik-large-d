# Which solution is critical for a complex scalar at large D: the real DSS or the CSS

Issue #95. For a free complex (two-component) massless scalar there are two self-similar candidates for the critical
solution: the real scalar's DSS solution (Choptuik's, with the field rotated into any fixed direction of the complex
plane), and Hirschmann and Eardley's phase-rotating CSS solution (matter/CSS.md, κ = 0). A critical solution has
exactly one unstable mode. This page counts the unstable modes of both in D.

Conventions as in matter/CSS.md: D = n + 2, τ = −ln(−t), perturbations ∝ $`e^{\lambda\tau}`$, γ = 1/λ (radius exponent).

**Result in brief.** [TBD]

## 1. The two counts

[TBD]

## 2. The CSS: its second unstable mode in D

**D = 4 against Hirschmann and Eardley.** Read (WebSearch, then WebFetch of the arXiv listing, of the ar5iv rendering of
arXiv v3 and of the v3 PDF's first page, 2026-11-08): HE95b write perturbations as $`e^{-i\sigma\tau}`$ with
τ = log(−t), so that "Eigenvalues in the lower half plane Im σ < 0 belong to unstable (growing) normal modes", and
γ = −1/Im σ. Their Section VII ("Results"), eqs. (33)–(34): "The third solution also lies on the negative imaginary
axis, at σ = −2.58324i ± 0.00002i", and "The fourth and fifth solutions form a conjugate pair lying at
σ = ±(3.50224 ± 0.00001) − (0.25867 ± 0.00001)i", with "γ = 0.387111 ± 0.000003". Our τ is −log(−t), so λ = iσ.

| mode | HE95b (λ = iσ) | here, `complex_css.py root 4` (R = 0.85 x_h; R = 0.6 x_h) | difference / their error |
|---|---|---|---|
| oscillatory pair | 0.25867 ± 3.50224i (± 0.00001 each) | 0.2586698339 + 3.5022487970i (0.2586698339 + 3.5022487970i) | 0.0 in Re, 0.9 in Im |
| black-hole mode | 2.58324 ± 0.00002 | 2.5832751746 (2.5832751746) | 1.8 |

The radii agree to 2×10⁻¹¹ (oscillatory) and 6×10⁻¹³ (real). These are spreads between two settings of one method,
not error bounds; the collocation of matter/CSS.md §4 (an independent code) gives 0.2586698 ± 3.5022488i and 2.5832752.
**The oscillatory mode matches HE95b within their error bars.** The black-hole mode is 3.5×10⁻⁵ above theirs, 1.8 of
their quoted errors (§5).

**Method.** The growth rates of the κ = 0 CSS are the zeros of css_pert.py's det M(λ): complex-path shooting, the
perturbation's jump across the self-similarity horizon on circles of radius R = 0.85 x_h and 0.6 x_h (matter/CSS.md
§2). A growth rate is a zero at every R; the jump method's artefacts move with R and sit near the resonance line
Im λ = κ_s Im μ, which for κ = 0 is Im λ = ω (κ_s Im μ = ω exactly when κ = 0, §3 of CSS.md). `complex_css.py` drives
css_pert.py without changing it:

- `track` continues the background in D by Newton on its jump (css_lateral.newton) and follows each root by secant at
  both radii, with an adaptive step: a step is accepted when the background converges (|J| < 10⁻¹⁰), each root
  converges at both radii to the same λ (to 10⁻⁸ up to D = 24, 10⁻⁶ above) and lies within 0.02 (1 + |λ|) of its
  quadratic extrapolation. A first run with fixed steps of one in D (`follow`) lost the oscillatory root at D = 5,
  where the secant from the linear extrapolation ran to Re λ = −3.4 (kept in the raw folder, `css/follow_k0.jsonl`).
- `boxes` counts the zeros of det M by the argument principle in four boxes per D, at both radii (ε = 0.25):

| box | region | expected zeros |
|---|---|---|
| A | 0.05 < Re λ < 0.9, \|Im λ\| < ω − ε | none |
| B | 0.9 < Re λ < 5, \|Im λ\| < ω − ε | λ = 1 (time translation) and λ_phys |
| C | 0.05 < Re λ < 5, ω + ε < Im λ < Im λ_osc + 1.5 | λ_osc |
| S | 0.05 < Re λ < 5, ω − ε < Im λ < ω + ε | none (the resonance line: artefacts only) |

  By the conjugation symmetry of the real perturbation system, zeros come in pairs λ, λ̄, so C and S cover the lower
  half too. The exact modes λ = 0 (phase = homothety) and λ = ±iω (shifts F → F + c) lie on Re λ = 0, outside.
- Controls: `--plant=Z` multiplies det M by (λ − Z); `--mutate=nometric` drops the metric perturbation from the wave
  equations (a test field on the frozen CSS), which must change the counts of B and C.

[TBD: results]

## 3. The real DSS: the imaginary sector

**The sector's equations** (`complex_imag_eqs.py`, sympy, every residual 0; its `--mutate` control, n/2 → n in the p
equation, is rejected). Write the complex field as φ = φ_R + iεψ on the real DSS solution φ_R.

- The stress tensor Re(∂_aφ ∂_bφ̄) − ½ g_ab |∂φ|² has no term of order ε, so at linear order ψ does not act on the
  metric or on φ_R. It obeys □ψ = 0 on the frozen DSS metric, the field equation of φ being linear.
- □ψ computed from ds² = −α²dt² + a²dr² + r²dΩ_n² with explicit sphere coordinates (D = 4, 5) is the general-n form
  $`(\alpha a r^n)^{-1}[-(a r^n\psi_t/\alpha)_t + (\alpha r^n\psi_r/a)_r]`$.
- With c = α/a, Φ = ψ_r, Π = ψ_t/c, p = r(Φ + Π), m = r(Φ − Π) and (τ, x), the p and m equations of
  `hp/src/model.rs` are exactly −x² and x² times $`(-t)^2 r^{-n}`$ × (αa rⁿ □ψ), with G = x c_x. No a and no
  Einstein equation enter. On the background G = (n − 1) c (a² − 1) by the slicing equation, which is the form used.
- For ψ ∝ $`e^{\lambda\tau}`$ × (periodic) the equations are those of the issue: the p, m equations with a, c, G frozen
  and ∂_τ → ∂_τ + λ. They are homogeneous of degree one in (p, m), and at λ = 0 with the background's own (p, m) they
  are the background's p, m rows. So ψ = φ_R, the U(1) phase rotation, is an exact λ = 0 mode.
- Series: at the centre the order-k coefficients obey a pointwise 2 × 2 system with determinant −c₀²(k − 1)(k − 1 + n),
  so P₁ = −M₁ is the free datum; at the past light cone, x = 1 − u, the order-k p coefficient obeys
  $`(\partial_\tau + \lambda + (1 + c_1)(k + 1) - n/2)P_k = \ldots`$ and the m equation gives $`M_{k+1}`$ explicitly.

This is the test-field problem of notes/GHOST_ADMIXTURE.md, where an ordinary or a ghost second scalar obeys the same
equation.

**The determinant** (`complex_imag.py`). Multiple shooting in the folded layout of hp (p, m odd in θ, M points on
the half period). Unknowns: the centre datum P₁, the test field's (P, Q) at each node of the stored state, and Q at the
light cone; rows: the centre series (at x_s = min(node 0, 0.05)) propagated to node 0, each segment's propagator, and
the light-cone series at the last node. det M(λ) vanishes at the growth rates. It shares with hp and with
`hp/modecount_cx` (the crate behind GHOST_ADMIXTURE.md's counts) only the stored state (hi + lo parts) and the
equations above. The θ derivative (built from Fourier modes), the background's centre and light-cone series, the
background between the nodes (DOP853, rtol 10⁻¹³), the test field's series and propagators (DOP853, rtol 10⁻¹¹) and
the LU are separate code, in double precision. The background's light-cone series is computed in mpmath at 34 digits:
in double precision rounding excites the mass mode, whose coefficients grow like binomial(n − 2 + k, k), and at D = 60
the coefficients past order 15 are that mode (|c₁₀₀| ≈ 10²⁹); fed into the test field's series this left the
background's (p, m) a null vector of M(0) only to 3×10⁻⁴.

**Checks** (`complex_imag.py check`, records in the raw folder's `imag/checks.jsonl`): [TBD]

**Counting.** As in notes/MODE_COUNT.md §2: the phase of g(λ) = det M(λ) ∏(λ − p_j) e^{−q(λ)} is followed around a box,
with p_j the candidate poles (the eigenvalues of −(ω d_θ + κ_k), κ_k = (1 + c₁)(k + 1) − n/2, k = 0 … 100, on
Im λ = ±ω, ±3ω, …) within 2 of the box in Re λ and 2ω in Im λ, and q a real quadratic fitted to ln|g| at the box's
corners and edge midpoints (det M grows like $`e^{60\lambda}`$ at D = 20 and $`e^{300\lambda}`$ at D = 4). Each
pole is at most simple, so wind(g) is an upper bound on the number of zeros inside, with equality when every candidate
inside is a pole. Edges are sampled at spacing 0.25 and bisected until every phase step is at most π/4 and the two
halves of each step add up to it. Symmetric boxes use det M(λ̄) = conj det M(λ) (the background is real): their
winding is the phase change along the upper half divided by π. The boxes hold every growth rate of both θ-parity
sectors with −0.05 < Re λ < 6, by the 2iω periodicity of the folded layout (MODE_COUNT.md §1):

| box | region | expected |
|---|---|---|
| B0 | −0.05 < Re λ < 0.3, \|Im λ\| < ω/2 | λ = 0, the phase rotation |
| U0 | −0.05 < Re λ < 0.3, ω/2 < Im λ < 3ω/2 | none |
| FULL | 0.3 < Re λ < 6, \|Im λ\| < 3ω/2 | none |

Controls: `selftest` (a synthetic det with zeros at 0, 1.2 ± 0.7i, 5.98 ± 0.2i, 0.02 inside FULL's right edge, and 3 ± 9i
outside, with the growth $`e^{40\lambda + 1.5\lambda^2}`$, through count_box itself: counts 1, 0, 4, and 1, 0, 6 with a
pair planted at 2 ± i; with one sample per edge and no bisection FULL counts 2, so the control fires), `--plant`, and
two mutated equations that break the λ = 0 mode: `--mutate=coupling` (the p equation's (n/2) c (P + Q) times 1.1) and
`--mutate=noG` (G dropped).

## 4. Verdict

[TBD]

## 5. Discrepancies

[TBD]

## 6. Proposed text for matter/CSS.md, matter/STRUCTURE.md and the README

[TBD]

## 7. Proposed notes/LIT_REVIEW.md entries

- **[HE95b] E. W. Hirschmann, D. M. Eardley, "Critical exponents and stability at the black hole threshold for a
  complex scalar field", PRD 52, 5850 (1995), arXiv:gr-qc/9506078** (read 2026-11-08: the arXiv listing, the ar5iv
  rendering of v3, the v3 PDF's first page). Contains: the linear perturbations of their D = 4 CSS solution, five zeros
  of det A in the lower half σ plane (σ = 0 and −i gauge, the black-hole mode σ = −2.58324i ± 0.00002i, the pair
  σ = ±(3.50224 ± 0.00001) − (0.25867 ± 0.00001)i), γ = 0.387111 ± 0.000003, and the statement that the solution is
  "unstable via a growing oscillatory mode", so not critical. The listing's abstract gives γ ≈ 0.387106 instead
  (versions 1 and 2 were withdrawn). Does not contain: any dimension other than 4, κ ≠ 0 (that is HE97), or the real
  DSS solution's perturbations as a complex field.
- **Gundlach, PRD 55, 695 (1997), arXiv:gr-qc/9604019**, as quoted in notes/GHOST_ADMIXTURE.md through the review
  arXiv:gr-qc/9712084: of the Choptuik solution's perturbations as a free complex scalar "only one of them, purely real,
  has Re λ > 0". Not reread here; the body of the 1997 paper has still not been read in this repository, so the entry
  should stay marked as read through the review only.

## 8. Scripts, data and how to reproduce

[TBD]

## 9. Recommended tasks

1. **An independent imaginary-sector count at large D.** What: complex_imag.py's propagators and LU in double-double
   (or arb) so that D ≥ 30 has a second implementation besides hp/modecount_cx. Why: in double precision det M is
   rounding noise at D = 60, with or without the kfreeze-13 cut (§3), so the large-D counts rest on one code.
   Acceptance: at D = 30.07, 45.05 and 60.00 the phase of det M agrees between integration tolerances 10⁻¹² and 10⁻¹⁴
   to 10⁻³ rad on every sample; the counts of B0, U0 and FULL equal the crate's; a planted pair is counted and
   `--mutate=coupling` changes B0. Depends on: nothing.
2. **The CSS's oscillatory mode at D = ∞.** What: the perturbation spectrum of the κ = 0 inner CSS solution
   (css_inner.py), to see whether Im λ_osc/ω → 2 and Re λ_osc → a limit near 0.53. Why: the finite-D track (§2)
   suggests both; a limit would make the CSS's instability exact at D = ∞, not only up to D = 80. Acceptance: the
   inner problem's λ = 2 mode is found (its control), and λ_osc(∞) lies within the 1/n extrapolation's spread of the
   finite-D values. Depends on: nothing.
3. **The κ-model's stability change in D.** What: follow the oscillatory mode in κ at fixed D (D = 4, 6, 10, 20, 40)
   to the κ_c(D) where Re λ_osc = 0. Why: HE97 find κ_c ≈ 0.0754 at D = 4, above which the CSS is critical; matter/CSS.md
   finds the κ = 1 CSS critical up to D = 80, and κ = 0 is not critical at any D here, so κ_c(D) is between. Acceptance:
   κ_c(4) reproduces HE97 to their digits; the crossing is a sign change of Re λ_osc seen at both circle radii.
   Depends on: backgrounds at intermediate κ (css_lateral.py cont).

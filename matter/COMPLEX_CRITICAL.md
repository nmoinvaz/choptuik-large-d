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

**Counting.** [TBD]

## 4. Verdict

[TBD]

## 5. Discrepancies

[TBD]

## 6. Proposed text for matter/CSS.md, matter/STRUCTURE.md and the README

[TBD]

## 7. Proposed notes/LIT_REVIEW.md entries

[TBD]

## 8. Scripts, data and how to reproduce

[TBD]

## 9. Recommended tasks

[TBD]

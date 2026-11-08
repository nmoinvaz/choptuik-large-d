# Which solution is critical for a complex scalar at large D: the real DSS or the CSS

Issue #95. For a free complex (two-component) massless scalar there are two self-similar candidates for the critical
solution: the real scalar's DSS solution (Choptuik's, with the field rotated into any fixed direction of the complex
plane), and Hirschmann and Eardley's phase-rotating CSS solution (matter/CSS.md, κ = 0). A critical solution has
exactly one unstable mode. This page counts the unstable modes of both in D.

Conventions as in matter/CSS.md: D = n + 2, τ = −ln(−t), perturbations ∝ $`e^{\lambda\tau}`$, γ = 1/λ (radius exponent).

**Result in brief.** [TBD]

## 1. The two counts

A self-similar solution is critical, i.e. an attractor of codimension one within the threshold, when it has exactly
one physical unstable mode. Modes that are coordinate or symmetry transformations do not count: they move along the
family of equivalent solutions.

- **The real DSS as a complex field.** Its perturbations split into a real sector, δφ_R with the metric, which is the
  real scalar's problem (one physical unstable mode, λ_phys, at the D counted in notes/MODE_COUNT.md), and the
  imaginary sector δφ_I, which decouples at linear order (§3). The imaginary sector's exact mode λ = 0 is the U(1)
  rotation φ → e^{iα}φ: it moves the solution along the circle of rotated copies, marginal, not unstable. So the real
  DSS is critical for the complex field if and only if the imaginary sector has no growth rate with Re λ > 0.
- **The CSS.** Its symmetry modes are λ = 0 (the phase rotation, which on a CSS solution is the homothety t → kt),
  λ = 1 (t → t + ε) and λ = ±iω (the shifts F → F + c). Besides them it has the real mode λ_phys and, in D = 4, the
  oscillatory pair λ_osc, λ̄_osc (HE95b). A complex pair is a two-dimensional unstable manifold, so the CSS has three
  unstable directions in D = 4 and is not critical. It would become critical in D if the pair crossed to Re λ < 0.

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

**The oscillatory mode in D** (`complex_css.py track 4 80 …` then `track 24 80 …`, raw `css/track_k0.jsonl`,
`css/track_k0_b.jsonl`; this table is printed by `complex_table.py`). The radius spread is the difference between the
roots at R = 0.85 x_h and 0.6 x_h, a spread between two settings of one method.

| D | ω | λ_osc | Im λ_osc / ω | radius spread | λ_phys |
|---|---|---|---|---|---|
| 4 | 1.915445 | 0.25866983 + 3.50224880i | 1.8284 | 2e-11 | 2.5832751746 |
| 4.1000 | 1.922890 | 0.26548619 + 3.52348307i | 1.8324 | 3e-11 | 2.5420759349 |
| 4.8125 | 2.010121 | 0.30586077 + 3.73046691i | 1.8558 | 2e-11 | 2.3757546939 |
| 6.0781 | 2.186683 | 0.35406091 + 4.12059626i | 1.8844 | 1e-10 | 2.2653161011 |
| 8.0781 | 2.420035 | 0.39904816 + 4.62645238i | 1.9117 | 6e-10 | 2.1973951429 |
| 10.0781 | 2.599557 | 0.42599540 + 5.01267114i | 1.9283 | 5e-10 | 2.1635873275 |
| 15.3281 | 2.916317 | 0.46320361 + 5.68992169i | 1.9511 | 2e-09 | 2.1187091738 |
| 20.3281 | 3.104878 | 0.48085224 + 6.09087669i | 1.9617 | 7e-10 | 2.0961158712 |
| 29.1953 | 3.312619 | 0.49742283 + 6.53091086i | 1.9715 | 8e-09 | 2.0727746422 |
| 39.1953 | 3.452773 | 0.50721444 + 6.82684981i | 1.9772 | 6e-09 | 2.0574658506 |
| 50.1172 | 3.550802 | 0.51349657 + 7.03342138i | 1.9808 | 6e-09 | 2.0468389353 |
| 60.9902 | 3.617682 | 0.51754131 + 7.17416570i | 1.9831 | 4e-09 | 2.0395953299 |
| 70.9902 | 3.663022 | 0.52017954 + 7.26949585i | 1.9846 | 9e-09 | 2.0346793610 |
| 80.0000 | 3.695097 | 0.52199898 + 7.33689470i | 1.9856 | 6e-07 | 2.0311970219 |

**The pair never approaches Re λ = 0.** Its real part rises monotonically, from 0.2587 at D = 4 to 0.5220 at D = 80, and
Im λ_osc stays between ω and 2ω. A fit in powers of 1/n (`complex_osc_fit.py`, degrees 3–4, D ≥ 15–30) gives
Re λ_osc → 0.5364–0.5365 and Im λ_osc / ω → 1.9935 (spreads over fit choices; the fits recover a planted series
0.53 − 0.9/n + 3/n² + 10/n³ − 40/n⁵ to 1.2×10⁻⁵ and 2 − 0.6/n + 1/n² + 5/n⁵ to 1.5×10⁻⁶). λ_phys(κ = 0) falls from
2.5833 to 2.0312, toward 2, as for κ = 1 (matter/CSS.md §7). At D = 80 the background's ω from the track differs from
`data/css_bg_k0.jsonl`'s by 2×10⁻⁷, and the radius spread of λ_osc is 6×10⁻⁷: the selection's sensitivity |∂J/∂ω|
falls like e^{−0.2n} (CSS.md §3), so D = 80 is near the end of what double precision resolves.

[TBD: box counts]

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

**Checks** (`complex_imag.py check`, records in the raw folder's `imag/checks.jsonl`, printed by
`complex_table.py`). States: `lowDn_D4.0000_M48` (11 nodes 0.01 … 0.7, the layout GHOST_ADMIXTURE.md §4 found free of
the 9-node layout's artefacts; copied into the raw folder's `states/`), `hp/data/states/up_D10.0000` (M 48),
`up2_D20.0437` (M 32), `up13_D60.0020` (M 32, nodes 0.3 … 0.7).

| check | D = 4 | D = 10 | D = 20.04 | D = 60.00 |
|---|---|---|---|---|
| background integrated node to node vs the stored nodes (max abs difference) | 3.0×10⁻¹² | 1.1×10⁻¹³ | 2.8×10⁻¹⁵ | 9.1×10⁻¹⁵ |
| light-cone series (34 digits) at the last node vs the stored node | 4.4×10⁻¹⁶ | 2.2×10⁻¹⁶ | 4.4×10⁻¹⁶ | 4.4×10⁻¹⁶ |
| background (p, m) as a null vector of M(0): \|M(0) z\| / \|z\| | 1.2×10⁻¹² | 1.7×10⁻⁹ | 7.0×10⁻¹¹ | 1.5×10⁻³ |
| smallest and next singular value of M(0) (largest 3–5) | 1.2×10⁻¹⁴, 6.0×10⁻⁷ | 6.6×10⁻¹², 6.9×10⁻¹² | 8.5×10⁻¹⁷, 2.2×10⁻¹⁶ | 6.1×10⁻¹⁷, 2.3×10⁻¹⁶ |
| Cauchy–Riemann on ln det M at 1.3 + 0.4i, h = 10⁻³ (anti-analytic control) | 2.8×10⁻⁹ (2.0) | 2.3×10⁻⁵ (2.0) | [TBD] | 2.4 (1.2): fails |
| det M(λ̄) − conj det M(λ) in ln | 7×10⁻¹⁵ | 5×10⁻¹³ | 3×10⁻¹³ | 2×10⁻¹³ |
| θ-derivative matrix vs hpkernel's dmat_odd | 3.4×10⁻¹³ | 3.4×10⁻¹³ | 8.8×10⁻¹⁴ | 8.8×10⁻¹⁴ |

So the reconstruction of the background is exact to rounding at every D, but at D = 60 the determinant is not: ln|det M|
jumps by 1–3 between λ 10⁻⁵ apart, and its phase is noise (`imag/evals_D60.jsonl`, `imag/evals_D60_kf13.jsonl`). With
the centre data's harmonics above 13 pinned and the light-cone block's p rows projected on them (hp's kfreeze) the
noise is smaller at some λ but still O(1) near the real axis (the run was stopped, `imag/queue1.log`). The cause is the
conditioning of the matching at large D: the multiple-shooting propagators have singular values down to 5×10⁻⁹ per
segment at D = 60, and matching nearer the light cone is worse (from x = 0.7 to 0.9 the irregular branch u^σ is damped by
3×10⁻¹⁹). hp works in double-double there. At D = 4–20 the phase of det M agrees between integration tolerances 10⁻⁹,
10⁻¹¹ and 10⁻¹³ to 10⁻⁶ rad or better (development runs; the counts' own steps are below π/4). **D = 60 is therefore
counted with hp/modecount_cx (double-double), as in GHOST_ADMIXTURE.md**, and complex_imag.py is the independent
check at D = 4, 10, 20.

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

**Counts** (`complex_imag.py count`, raw `imag/counts.jsonl`; windings of g, which are integers to 10⁻¹³; every phase step
≤ π/4 after bisection, no bisection limit reached):

| D | state | B0 | U0 | FULL (candidate poles inside) | largest step | determinants |
|---|---|---|---|---|---|---|
| 4 | lowDn_D4.0000_M48 | 1 | 0 | 0 (2) | 0.73 rad | 162 |
| 10 | up_D10.0000, M 48 | 1 | 0 | 0 (10) | 0.72 rad | 186 |
| 20.04 | up2_D20.0437, M 32 | 1 | 0 | 0 (16) | 0.65 rad | 210 |
| 60.00 | up13_D60.0020, M 32, kfreeze 13: hp/modecount_cx, double-double | [TBD] | | [TBD] | | |

In every box wind(g) equals the number of expected zeros, so each box holds exactly its expected zeros and every
candidate pole inside FULL is a pole. **The imaginary sector has no growth rate with Re λ > −0.05 other than λ = 0, in
|Im λ| < 3ω/2 (both θ-parity sectors) and Re λ < 6, at D = 4, 10, 20 (and 60, [TBD]).**

**Controls**, each seen to fire:

| control | D | box | count without → with |
|---|---|---|---|
| pair planted at 3.1 ± 1.0i | 20.04 | FULL | 0 → 2 (B0 stays 1) |
| zero planted at 0.05 + 1.5i (with its conjugate) | 4 | U0 | 0 → 1 |
| `--mutate=coupling` | 20.04 | B0 | 1 → 2 |
| `--mutate=coupling` | 10 | B0 | 1 → 0 |
| `--mutate=noG` | 20.04 | B0 | 1 → 0 |
| `--mutate=noG` | 4 | B0 | 1 → 0 |
| selftest, one sample per edge | synthetic | FULL | 4 → 2 (must differ) |

**The λ = 0 mode located.** Secant on det M (`complex_imag.py root`) from 0.03 + 0.02i at D = 20.04 converges to
λ = 3.8×10⁻⁸ + 5.4×10⁻⁸i, i.e. to 0 within the determinant's noise (about 10⁻⁶ in ln det M). The null vector of M(0)
is the background's own (p, m) (the checks table above).

**Against hp/modecount_cx.** At D = 4 the same secant from −0.36 converges to −0.3651202704; GHOST_ADMIXTURE.md §4 has
−0.3651202372 (lowDn layout, M = 64) and −0.3651202796 (9-node layout, M = 48) from the crate, a decaying mode of the
imaginary sector. The two codes share the stored states and the equations, nothing else, and agree to 3×10⁻⁸ and
9×10⁻⁹. Their counts agree at the D both have counted: GHOST_ADMIXTURE.md §3 finds only λ = 0 at D = 8.25, 16.76,
30.07, 45.05 (two layouts each) and 67.01 (`hp/data/modecount_test_results.jsonl`).

## 4. Verdict

[TBD: table]

**The real DSS is the critical solution of the free complex scalar at every D checked, D = 4 to 67, and nothing in
the trends points to a change at larger D.** Its imaginary sector adds no unstable mode, only the marginal phase
rotation, so it keeps the real scalar's single unstable mode. The CSS has three unstable directions at every D from
4 to 80 (λ_phys and the oscillatory pair), and the pair moves away from Re λ = 0 as D grows (Re λ_osc → 0.536 by
extrapolation). Neither count changes with D, so there is no crossover dimension.

**What it means for the critical exponent.** At the threshold of black-hole formation in a one-parameter family of
complex-scalar data, the evolution approaches the real DSS rotated by some phase e^{iα} (α set by the data; the phase
mode is marginal, so α is a modulus, not a source of scaling). The scaling comes from λ_phys alone, so the complex
scalar's γ(D) is the real scalar's γ(D) = 1/λ_phys(D), with γ∞ = 1/2 (GAMMA.md), and the mass exponent is (D − 3)γ.
The CSS's exponent 1/λ_phys(κ = 0) (0.3871 at D = 4, 0.4923 at D = 80 from the table of §2) is not observed for
generic families, because reaching the CSS takes tuning three parameters. That the CSS's λ_phys also tends to 2 does
not make the two exponents equal at finite D: at D = 20 they are 1/λ = 0.477 (CSS, κ = 0) and [TBD] (DSS).

**Status.** *Numerical evidence*: winding counts in finite boxes (|Im λ| < 3ω/2 and Re λ < 6 for the imaginary
sector; Re λ < 5 and Im λ below Im λ_osc + 1.5 for the CSS), each passing its planted and mutated controls. Not covered:
growth rates outside the boxes, perturbations with τ-harmonics above the layout's (M = 32–48, and 13 at the centre at
D = 60), non-spherical perturbations, and D = ∞ itself.

## 5. Discrepancies

- **HE95b's black-hole mode.** The v3 text gives σ = −2.58324i ± 0.00002i and γ = 0.387111 ± 0.000003; the arXiv
  listing's abstract gives γ ≈ 0.387106. This page's λ_phys = 2.5832751746 (complex-path shooting, two radii
  agreeing to 6×10⁻¹³) and matter/CSS.md's collocation 2.5832752 give γ = 0.3871055, which matches the listing and is
  1.8 of HE95b's quoted errors from their text. matter/CSS.md §4 quotes "γ = 0.387106 [HE95b] (λ = 2.58327)", from the
  listing, while matter/LITERATURE.md §7.2 quotes 0.387111, from the text: the repository quotes both. The oscillatory
  pair, by contrast, agrees with the text within its errors (§2).
- **complex_imag.py at D = 60.** Not a disagreement but a limit: the independent double-precision determinant is
  rounding noise there (§3), so the D = 60 count rests on hp/modecount_cx alone.
- **Re λ_osc and Im λ_osc / ω at D = ∞.** The 1/n fits give Im λ_osc / ω → 1.9935 with no visible spread, not 2. Whether
  the limit is 2 with a slower correction (a log, or a non-analytic term) is not decided by D ≤ 80.

## 6. Proposed text for matter/CSS.md, matter/STRUCTURE.md and the README

For matter/CSS.md §11, replacing the bullet "**κ = 0 (free complex scalar)**":

> - **κ = 0 (free complex scalar)**: ω(D) and |f(0)|(D) were computed to D = 80 and extrapolated (§7). Its second,
>   oscillatory unstable mode (λ = 0.2587 ± 3.5022i at D = 4, HE95b's σ = ±3.50224 − 0.25867i) was followed to D = 80
>   in [COMPLEX_CRITICAL.md](COMPLEX_CRITICAL.md): Re λ_osc rises to 0.522 (0.536 extrapolated), so the κ = 0 CSS has
>   three unstable directions at every D and is never critical; the real DSS is (its imaginary sector adds no unstable
>   mode).

For matter/CSS.md §4, the row "κ = 0: λ (real)": "γ = 0.387106 [HE95b abstract on arXiv]; their text gives
σ = −2.58324i ± 0.00002i, γ = 0.387111 ± 0.000003", and the row "κ = 0: oscillatory mode": "σ = ±(3.50224 ± 0.00001)
− (0.25867 ± 0.00001)i [HE95b, Sec. VII], i.e. λ = 0.25867 ± 3.50224i".

For matter/STRUCTURE.md §7, after "(conjecture; in D = 4 the free complex scalar's CSS solution has more than one
unstable mode, so it is not critical for κ = 0, while it is for κ ≳ 0.075)": "The same holds at every D up to 80, where
its oscillatory pair has Re λ = 0.52 (matter/COMPLEX_CRITICAL.md); the free complex scalar's critical solution is the
real DSS, whose imaginary sector has no unstable mode."

For the README, in "What is known about the selection, and what is open", one sentence: "For a free complex scalar the
critical solution is the real DSS at every D checked (4–67): its imaginary sector, a free test field on the DSS
metric, has no growing mode, while Hirschmann and Eardley's CSS keeps a second, oscillatory unstable mode to D = 80
(matter/COMPLEX_CRITICAL.md)."

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

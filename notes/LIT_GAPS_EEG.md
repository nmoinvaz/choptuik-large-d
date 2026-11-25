# LIT_GAPS_EEG: the large-D papers of Ecker, Ecker and Grumiller and their inputs, read for the large-D quest

Issue #1089, slice 1 of three. This is a subagent of the coordinator on the Mac, 2026-11-25. Its first post and its
correction are the comments of 00:28 and 00:33 UTC on #1089. The papers are cited by arXiv identifier:

- Ecker, Ecker, Grumiller arXiv:2601.14358 (PRL 136, 191401; here **EEG**), arXiv:2611.07007v1 (**EEG26b**) and
  arXiv:2411.09233 (PRD 112, L021505).
- Ecker, Ecker, Grumiller, Jechtl arXiv:2602.10185 (SciPost Phys. 20, 176; **EEGJ**).
- Lai, Wen arXiv:2610.04960v2 (**LW**).
- Reiterer, Trubowitz arXiv:1203.3766 (CMP 368, 143; **RT**).

How they were read:

- EEG26b: the arXiv HTML, through WebFetch, equation by equation.
- EEG pages 2–4 and RT pages 1–13: the PDFs rendered to images.
- EEGJ: the HTML of v2, to its first 100,000 characters.
- LW: the HTML of v2.

Earlier readings are in notes/LIT_REVIEW.md (#802 addendum), notes/RT_FORMULATION.md (#91, #106), notes/EEG_PROFILE.md
and notes/SECOND_ORDER.md. This note points to them rather than repeating them. No text addressed to an AI agent was
found in any of the papers. EEG26b's footnote 2 and its Summary credit this project's D ≤ 61 work and γ → 1/2, which
bears on nothing below.

Scripts: [`solver/lit_gaps/`](../solver/lit_gaps/). Each prints in seconds, and each has a mutation control that fails.
Their outputs are quoted here and are reproduced by section 8, so there is no raw folder. Status words follow AGENTS.md.

## 0. Verdicts

| question | answer | status |
|---|---|---|
| Newer work by these authors past D 200? | None. EEG26b is still v1 (4 Nov 2026), and the 2026 author listing has no later paper on collapse | read, 2026-11-25 |
| Can a published scheme reach D 2000–10000 much faster than ours? | No. EEG26b's fixed 40 × 80 spectral scheme has an error in Δ that grows from D ≈ 68 (§4.1). A fixed polynomial degree N sees the horizon condition only at relative level n^−(N+1) (§4.2) | numerical evidence |
| EEG26b's explicit large-D model as a seed? | A D-independent profile such as Eq. (16), no: with our F (of which Eq. (16) is a two-parameter fit, EEG_PROFILE) Ω is off by 4.8% rms at D 200, falling like ε^0.88. Its corrected NNLO (S35)–(S51) with our profile's K₁, K₂, yes: 2.8×10⁻⁴ at D 200, falling like ε^3.02 (§3) | numerical evidence (SECOND_ORDER §1.2) |
| EEG26b's Table I from D 93 | 0.5–4.8×10⁻⁶ high in every row, against a stated check that "varying the number of Fourier or Chebyshev modes does not alter these digits" (§4.1) | discrepancy, numerical evidence |
| #1086's candidate cause, σ_R passing Chebyshev degree 80 near D 157 | Not supported: no step at 157 (best knot 104, step 1.8×10⁻⁷, below the rounding). A fixed degree's grip falls smoothly like a power of n, with no threshold (§4.1–4.2) | numerical evidence |
| One echo period for every D > 3 | Assumed (EEGJ Sec. 2) and seen in local continuation (EEGJ Sec. 4.1), not shown. Our M = 3 truncation at D = ∞ has two more certified zeros (§4.3) | open |
| EEG's NLO selection, Eq. (20) | EEG26b's own fitted profile violates it by 18× in Δ: applied to Eq. (16), it selects Δ = 0.0911 (§4.4) | checked numerically |
| D → ∞ values of EEG26b | Table I's ∞ entry 1.643911 is 1.7×10⁻⁵ above our Δ∞; its 4πμ₀ is 1.4×10⁻⁵ above. Eq. (8)'s "≳" is an equality in the limit (§4.5) | numerical evidence |
| Connections | EEG26b Eq. (3) = our identity; EEG's N₀ = ⟨β̃₀²⟩ = 7.4158 against our 7.4164; EEGJ's δ = our σ_R with ⟨ψ₋₀²⟩ = ḡ; LW's Floquet λ = 1 − ḡ = 4πμ/Δ (§5) | checked by hand / numerically |

## 1. What each paper contains

### 1.1 EEG26b, arXiv:2611.07007v1

**Formulation.** RT's gauge in D dimensions, Eq. (2):

    ds² = e^{−2ζ}( −(1 − ξ²)dt² − (2/μ) ξ dt dξ + dξ²/μ² + ξ²/(μ + Qξ²)² dΩ²_{D−2} ).

Its variables are ω₁, ω₂^±, ω₃^±, ω₄^± (Eq. (4)) and its first-order quadratic equations Ω_i^± = 0 (Eq. (5)). Both
are transcribed and checked in sympy against the covariant equations in notes/RT_FORMULATION.md §1–2, which also records
the normalization difference between Eq. (1) and Eq. (5). The echo period is Eq. (3):

    Δ = 4πμ / (1 − ⟨ψ₋₁²⟩) = 4πμ − ∫₀^{4π} ω₃⁻(t, 0) dt.

The SM's (S23) gives the same as Δ = 4πμ − ∫₀^{4π} ω₂⁻(t, 1) dt.

**Numerics** (section "Numerical setup", SM VII.1):

- The grid is Fourier in t on [0, 4π] times Chebyshev in ξ on the parity-extended [−1, 1] (Eq. (6)):
  g = Σ g_nm exp(int/2 + imθ), θ = arccos ξ.
- N_Fourier = 40 and N_Chebyshev = 80, which is 11,200 unknowns. No change of these with D is stated.
- Newton–Raphson from "a sufficiently close initial guess". "Typically, after four iterations the residual is brought
  down to ‖Ω⁺ᵢ‖∼10⁻²⁵".
- The arithmetic is GMP dyadic numbers. "Rounding only becomes necessary once the Jacobian is inverted at each Newton
  step, which we need to do with increasingly higher precision as D grows."
- Steps of ½ in D from D = 4, "taking linear extrapolations of the two previous converged results as initial guesses".
- The SM's introduction states: "The results for D>28 have sufficient accuracy so that we trust all displayed digits,
  and we checked that varying the number of Fourier or Chebyshev modes does not alter these digits." It names no mode
  numbers and no D.
- SM VII.1 gives the horizon exponent as (1 − x)^{(D−4+f(D))/(2−f(D))}, "approximately as D/2". It also credits RT's
  "judiciously chosen split of the spectrum into UV and IR modes".

**Large D.**

- Fits from μ = Σ₀⁴ μ_n εⁿ and Δ = Σ Δ_n εⁿ, ε = 1/(D − 1): Eq. (7) μ₀ ≈ 0.130818, μ₁ ≈ −0.1449; Eq. (9) Δ₀ ≈ 1.6439,
  Δ₁ ≈ 10.37.
- Eq. (8): lim Δ ≳ 1.6439, from Δ ≥ 4πμ, because the denominator of Eq. (3) is at most 1.
- Eq. (10): β = √ε β̃.
- Eqs. (11)–(14): the leading order, f = 1 − ε(1 − x²)β̃²/2 and Ω = εβ̃².
- Eq. (15): (Δ₁ − 4πμ₁)/(4πμ₀) = ⟨β̃²⟩.
- Eq. (16), the model: β̃ = A∗ Re[e^{iθ}/(1 + q e^{i(2θ+δ)})], with A∗ ≈ 3.8455, q ≈ 0.0551, δ ≈ 0.0985 and θ = 2πτ/Δ₀.
- The SM gives the GG-gauge equations (S36)–(S39), the NLO (S40)–(S43) and the NNLO (S44)–(S51).
- It constructs the model in (S52)–(S62): the profile's Fourier coefficients are extrapolated in ε coefficient by
  coefficient, and q and δ are fitted on the first four (S62). The forms are in EEG_PROFILE.md §1–2.

**Outlook.** Large-D numbers from first principles "will likely require to add instanton corrections of the form
e^{−1/ε}". Also, "it may be necessary to extend the solution beyond the SSH and to match with the Rozali–Way regime".

**Not there.**

- The precision used at any D.
- The Jacobian's conditioning.
- Which mode numbers the convergence check used, and at which D.
- A reason for stopping at D = 200, a run time or a cost.
- γ at any D > 5.5. γ → 1/2 is quoted from this project's work.
- Any mention of EEG's Eq. (20).
- Any error bar on Table I's ∞ entry, 1.643911.

### 1.2 EEG, arXiv:2601.14358 (read from the PDF, pages 2–4)

**Gauge and equations.**

- GG gauge, with Eq. (2): Ω = (D−3)(a²−1)/x², f = α/a, ε = 1/(D−1).
- Field equations (3a)–(3d). The convexity conditions (6) are 0 ≤ Ω and 0 ≤ x ≤ f ≤ 1.

**Leading order.**

- Ω_LO = Π²_LO (7).
- Ψ_LO = D₁Π_LO/f_LO (8).
- Π_LO = β/√(1 + β²x²) (10), with β(τ) free.
- f_LO = √((1 + β²x²)/(1 + β²)) (13), fixed by f(τ, 1) = 1 (12).
- "any periodic function β(τ) with any echoing period Δ solves our equations of motion" (page 3).

**Next order.**

- NLO solution (15)–(18).
- Eq. (19): f_LO + εf_NLO = 1 − ε(1 − x)β(β'' + 3β') + … near the SSH and a zero of β.
- **Eq. (20)**: "echoing period: Δ = |β''|/(3|β'|) at β = 0". Here β(τ/Δ) has period Δ, so β itself has period 1.
- The example **Eq. (21)**: β = cos 2πτ + sin(6πτ)/A, A ≈ 15.9476, chosen "to satisfy the condition (20) with Δ = 1".
- Its NNLO is "(barely) viable" from D = 52.
- Eq. (22): max f = 1 − (ε²/4)(1 − x⁴)β'(τ₀)² + O(ε³).
- At D = 300 the expansion "converges fast in the center and slowly at the SSH" (page 4).

**Not there.** Any numerics of the full equations. Any test of Eq. (20) against a solved profile.

### 1.3 EEGJ, arXiv:2602.10185v2

**Method.** Martín-García–Gundlach shooting (Sec. 4.1):

- N_τ = 512 Fourier modes for D ≳ 3.2.
- N_x = 8000 implicit Runge–Kutta steps for D ≥ 4.
- Centre Taylor series to O(x_L⁵).
- SSH Taylor series to O((1 − x_R)²) below D 4.8 and O((1 − x_R)³) to 5.5, with x_R = 1 − 10⁻².
- Matching at 0.1.
- Finite-difference Newton (10⁻¹⁰, damping 0.1), in double precision (C++, FFTW, LAPACK).

**Exponent.** App. B (`eq:appB7`, as recorded in LIT_REVIEW): δ = (2⟨ψ₋₀²⟩ + D − 4)/(2 − 2⟨ψ₋₀²⟩).

**Stop.** At D = 5.5, because the branches could no longer be told apart (Sec. 8.1, 8.3).

**Uniqueness** (§4.3 below):

- Sec. 2: "In practice, only a single such solution exists", which is then assumed for real D.
- Sec. 4.1: "for any D there exists a unique solution for a specific value of the echoing period Δ".

### 1.4 EEG, arXiv:2411.09233

The NEC-line angle is α = 2 arccot(D − 1). It rests on Taylor expandability at x = 0 and on v₁ having two simple zeros
per period (LIT_REVIEW's entry). Δ does not enter it. It offers nothing for speed. Its assumption of two zeros per
period is the "one echo per period" shape assumption that §4.3 discusses.

### 1.5 LW, arXiv:2610.04960v2

- It works in EEG's past patch with EEG's leading order.
- Eq. (4) is EEG's Eq. (21).
- The Polyakov auxiliary field has the monodromy C = 2∫₀¹ β²/(1 + β²) dτ (Eqs. (26)–(27)).
- SSH regularity fixes the Floquet exponent λ = ∫₀¹ dτ/(1 + β²) (Eqs. (34)–(35)), with C = 2(1 − λ) (Eq. (38)).
- On uniqueness: it "appears only after next-to-leading 1/D corrections are imposed and convexity conditions such as
  f≤1 are enforced" (Sec. II.B). This is EEG's Eq. (20), which §4.4 tests.
- It gives no numerics of the classical problem.

### 1.6 RT, arXiv:1203.3766

**Coordinates.** u_σ = −(1 + σξ) e^{−μτ}, σ = ±, so that Θ acts as τ → τ + 2π (Eq. (6)).

**The invariant μ.**

- e^{2πμ} = f′_u(0), the derivative at the light cone of the self-similarity map's action on any regular ingoing null
  coordinate (Sec. 1.4, Facts 1–3).
- μ is enclosed to 10⁻⁸⁰ (Eq. (3)), and K = Δ/2 likewise (Eq. (2)).

**Space.**

- Fourier–Chebyshev coefficients that decay like κ₁^−|m| κ₂^−|n|, with κ₁ = 65/64 and κ₂ = 5/4.
- So the functions are analytic on the strip |Im τ| < 2 log κ₁ times an ellipse, |ξ| < 41/40 on the real line (Sec.
  1.1).
- The principal part is "morally multiplication by im/2 + μ(n + 1)", and its inverse is O((m + n)⁻¹). It is diagonal in
  m and upper triangular in n.
- "It disposes of the geometric boundaries … in a seamless way, and one does not specify initial data anywhere."

**Linear algebra** (Sec. 1.3):

- 250 × 750 modes, so N = 654,375.
- The matrix X is a dense block of size about N/100, plus a band close to the identity whose width is "in the
  thousands", plus small entries.
- The approximate inverse is Gaussian elimination on the block, with the band, then a truncated Neumann series. It is
  checked by ‖1 − XY‖ < 1.
- Costs, on "~2010 equipment": the reference takes "a few dozen CPU hours", and the inverse and its estimates take "a
  few thousand CPU hours", in integer arithmetic only.

**Not there.** D ≠ 4.

## 2. Schemes and their cost against D (question 1)

| scheme | unknowns | arithmetic | horizon treatment | path in D | what grows with D | at D 2000 / 10000 |
|---|---|---|---|---|---|---|
| EEGJ shooting (§1.3) | 512 modes × shooting, 8000 steps | double | Taylor at 1 − 10⁻², order 2–3 | continuation | Taylor order must exceed δ ≈ (D − 4)/2 (their SM VII.1 and Sec. 8) | order above 1000 / 5000; stopped at 5.5 |
| EEG26b spectral (§1.1) | 40 × 80, 11,200 | GMP; the inverse's precision rises with D, value not stated | none, left to the basis | ½, linear predictor | the error in Δ, linear in D from 68 (§4.1); the grip, a power of n (§4.2) | 3,600 / 19,600 more steps from D 200; the linear extrapolation gives 6.4×10⁻⁵ / 3.3×10⁻⁴ in Δ, with no mechanism |
| RT, D = 4 proof (§1.6) | 654,375 | integer | none, analyticity in the κ-ellipse | — | — | — |
| ours, the chain (LARGE_D_BASICS §4) | 769 | octo-double | series, ks ≥ σ_R + 7 | ANM | digits ≈ 0.155(n − 1) + 0.0087n + the weak directions' gain; ks ≈ n/2 + 9.7 (§5) | about 330 / 1640 digits, plus the weak directions' gain; ks 1009 / 5009 |

The EEG26b row cannot be costed per step, because neither the precision nor the run time is stated. A dense LU at
L = 11,200 costs about 4.2 h at 512 bits on one core of this Mac (RT_FORMULATION §5(3)). RT's block-plus-Neumann
inverse would be far cheaper, but its block size at D 200 is not stated.

## 3. The explicit large-D model as a seed at any D (question 2)

**The formulas.** These are the GG-gauge fields of EEG (§1.2) and of EEG26b's SM. The map to our variables, checked
symbolically in notes/SECOND_ORDER.md (Summary), is f = c, x²Ω = (n − 1)(a² − 1), Π = (p − m)/(√2 x),
Ψ = (p + m)/(√2 x²), with ε = 1/(n + 1).

- **LO** (EEG Eqs. (7), (10), (13)), with β = √ε β̃:

      Π = β/√(1 + β²x²),   Ω = Π²,   f = √((1 + β²x²)/(1 + β²)),   Ψ = ε^{1/2}·ε(β̃′ + β̃)   ((S35b)).

  In our variables this is notes/LARGE_D_BASICS.md §3.2's slow manifold with K = β² = 2p₁².
- **NLO**: EEG26b (S40)–(S43), with plain β read as β̃ (#142's reading A). For example,
  Π₂ = (x²/2)(2β̃ + β̃″ + 3β̃′ − β̃³) and Ω₂ = β̃(x²(β̃″ + 3β̃′ − β̃³) − 2(1 − x²)β̃).
- **NNLO**: EEG26b (S44)–(S51) **corrected**. The printed Π₃ violates (S39) at ε^{5/2}, and Ω₃ and f₃ inherit the error
  (#142, `solver/independent_eeg_nnlo/REPORT.md`, which gives ΔΠ₃). The corrected terms are in SECOND_ORDER.md §1 and
  in `solver/inner_pert/nnlo_vs_solution.py` (`model_fields`, corrected, order 3).
- **Profile**: EEG26b's Eq. (16) is β̃ = A∗ Re[e^{iθ}/(1 + q e^{i(2θ+δ)})]. Its D-dependent replacement is
  SECOND_ORDER.md §1.2's
  β̃_D(θ) = √(2(1 + 1/n)) (F + K₁/(2n) + K₂/(2n²)), θ = 2πτ/Δ(n), Δ(n) = Δ∞ + Δ₁/n + Δ₂/n².
  The data are F, K₁, K₂ for every odd ν ≤ 31 in `solver/inner_pert/data/closed_form_K.json`, with Δ∞ = 1.643893746,
  Δ₁ = 10.3753 and Δ₂ = −49.1685.

**How good** (SECOND_ORDER.md §1.2's table; relative rms of Ω against the stored states, numerical evidence):

| profile in the corrected NNLO | D 40 | D 100 | D 200 | exponent in ε |
|---|---|---|---|---|
| a D-independent profile: our F, of which Eq. (16) is a two-parameter fit | 0.203 | 0.0923 | 0.0484 | 0.88 |
| F + K₁/2n | 0.0393 | 0.00339 | 7.1×10⁻⁴ | 2.46 |
| F + K₁/2n + K₂/2n² (the closed form) | 0.0388 | 0.00226 | 2.81×10⁻⁴ | 3.02 |

Extrapolated with ε^3.02 (`eeg_consistency.py`; a measured law carried past its range, not a bound):

| D | 245.64 | 263.6 | 400 | 2000 | 10000 |
|---|---|---|---|---|---|
| rms of Ω, relative | 1.5×10⁻⁴ | 1.2×10⁻⁴ | 3.4×10⁻⁵ | 2.7×10⁻⁷ | 2.1×10⁻⁹ |

**Verdict.** EEG26b's Eq. (16) is a poor seed at any D, because it has no 1/n terms. The corrected NNLO with our K₁ and
K₂ is a good seed in the sense of a smooth guess everywhere on 0 ≤ x ≤ 1. Each further order of the outer expansion
gains about 25ε (SECOND_ORDER.md §1.1: R/ε^p is 320, 6.0×10³ and 1.6×10⁵ for Ω at LO, NLO and NNLO).

Two limits:

1. The ladder of weak directions lies in the profile's high harmonics. There K₂ is the least determined part (±1.8 on
   the fundamental, 9 fit choices). At D 2000 that alone is about 2×10⁻⁷ of F₁ (1.8/(2n²)).
2. The seed is conditional on Conjecture 1 through F. Fitting F as well removes that, with an exponent of 3.05.

## 4. Gaps and errors (question 3)

### 4.1 EEG26b's Table I and its convergence statement

The table is compared with ours in `hp/data/delta_published_compare.csv`, using our dd, qd and chain rows. The shape of
d = Δ_EEG − Δ_ours on D 60–200 (`eeg_gap_shape.py`; 141 rows; rounding σ = 10⁻⁶/√12 = 2.9×10⁻⁷, a model of the six
decimals, not a bound):

| D window | mean d | slope per unit D |
|---|---|---|
| 60–100 | 4.4×10⁻⁷ | 1.8×10⁻⁸ |
| 100–140 | 1.6×10⁻⁶ | 3.5×10⁻⁸ |
| 140–170 | 2.9×10⁻⁶ | 3.0×10⁻⁸ |
| 170–200 | 4.0×10⁻⁶ | 3.2×10⁻⁸ |

- A straight line fits at χ²/dof 1.12, crossing zero at D 68.1.
- A broken line with a step, its knot scanned over 100–185, finds its best knot at 104, with a step of 1.8×10⁻⁷. That
  step is below the rounding.
- Control `--mutate=step`: a planted 10⁻⁶ step at D 157 is found at 155, with a step of 1.01×10⁻⁶.

**Findings.**

1. EEG26b says it checked its digits against the number of modes (§1.1). Yet every row from D 93 on lies more than half
   a unit of the sixth decimal above ours (the smallest is 5.2×10⁻⁷, at D 100), by up to 4.8 units at D 198. The check,
   whatever it was, missed an error that grows smoothly from D ≈ 68. LIT_REVIEW's #802 row says the SM has no
   convergence test in N. It has a statement of one, without numbers (proposed correction, §7).
2. #1086 (LARGE_D_BASICS §6.1) proposed as the cause "a Chebyshev degree fixed at 80 while σ_R ≈ n/2 + 2.6 passes 80
   near D 157". The data show no change at 157 (finding of the step scan). §4.2 explains why a fixed degree has no
   threshold there. The candidate stands only in the weaker form "a fixed degree whose grip weakens like a power of n".
3. RT_FORMULATION §12's collocation at EEG26b's degree (40 × 79) gives 0.64–1.35 of d at D 66–84, growing the same way.
   That remains the evidence that truncation at their resolution has the right size. Their Galerkin projection, path
   and root on the weak direction (RT_FORMULATION §10.3) were not run here.

### 4.2 What a fixed polynomial degree sees at large n

`free_field_cheb.py` collocates the flat free self-similar wave with a polynomial of degree N in z = x², at Chebyshev
points, with no boundary rows, as RT's scheme has none. The wave is z(1−z)f″ + (c − (a+b+1)z)f′ − abf = 0, with
a = λ/2, b = (λ+1)/2 and c = (n+1)/2. Its centre-regular solution is ₂F₁(a, b; c; z), and its horizon exponent is
σ = n/2 − λ. No wave with λ = ijω is regular at both ends (LARGE_D_BASICS §3.4).

The table gives σ_min/σ_max of the row-equilibrated matrix at λ = iω, ω = 3.822:

| n | 20 | 60 | 100 | 200 | 400 | 1000 | 2000 |
|---|---|---|---|---|---|---|---|
| N = 24 | 6.6×10⁻¹⁷ | 8.6×10⁻²⁸ | 3.1×10⁻³³ | 9.8×10⁻⁴¹ | 2.9×10⁻⁴⁸ | 3.3×10⁻⁵⁸ | 9.7×10⁻⁶⁶ |
| slope d ln/d ln n, N = 24 | −18.2 | −24.2 | −24.7 | −24.9 | −25.0 | −25.0 | −25.0 |
| N = 40 | 3.3×10⁻²¹ | 1.3×10⁻³⁷ | 2.6×10⁻⁴⁶ | 1.7×10⁻⁵⁸ | 8.6×10⁻⁷¹ | (floor) | (floor) |

The 10⁻⁸⁰ working precision floors the last two N = 40 entries. Control `--mutate=planted`: λ = −2, a polynomial wave
regular at both ends, gives 7×10⁻⁸⁵, i.e. singular.

**Reading** (numerical evidence on a linear model):

- σ_min/σ_max falls like n^−(N+1), smoothly through Re σ = N (n = 48 for N = 24, n = 80 for N = 40), with no recovery
  beyond.
- The reason: the centre-regular wave is a degree-N polynomial up to its Taylor tail, and the k-th term of that tail is
  O(n^−k). That tail is the outer 1/n expansion, which admits any Δ. The horizon information appears only where the
  Taylor coefficients turn algebraic, at k of order n, because the fast mode F varies on the scale x/n.
- **So, in this model, a polynomial discretization needs of order n/2 degrees of freedom across x to impose the
  selection with an O(1) grip.** That covers a horizon series (ks ≥ σ_R), a global Chebyshev degree, and Taylor steps
  resolving F (LARGE_D_BASICS §4).
- This supports RT_FORMULATION §5(2)'s verdict, with a cleaner law than its measured σ_min ∝ N^−p(D).

**Withdrawn.** My first post's item 2 rested on `cheb_branch_tail.py`'s model, the Chebyshev share of the bare branch
(1 − ξ)^σ. That share has its minimum near σ ≈ 0.75N and rises beyond, which suggested a recovery past σ = N. The
collocation above refutes the inference. The bare branch's representability is not what sets the grip. The script stays
as a record, and its docstring says so.

**Discrepancy, not resolved.** RT_FORMULATION §4's 24 × 21 RT grid lost only 2.2 orders of σ_min from D 20 to 100. The
free-field law at N = 21 would give about 15. Several things differ:

- The weakest direction found there is ω₃'s harmonic 0, a metric field, not the scalar's horizon direction.
- The states are mapped states with a residual of 10⁻⁷, not solutions.
- The variables and the projection differ.

Which of these explains the gap was not tested.

### 4.3 One echo period for every D > 3

EEGJ Sec. 2 assumes a single critical solution for real D ("In practice, only a single such solution exists"). Sec. 4.1
observes "a unique solution for a specific value of the echoing period Δ". That observation comes from shooting with
Newton along a continuation, which cannot see other roots. EEG (page 3) focuses on profiles with "a single zero in
the interval τ ∈ [0, Δ/2)", and 2411.09233 assumes that v₁ has two simple zeros per period (LIT_REVIEW).

Against this, our M = 3 inner selection at D = ∞ has two further certified simple zeros (#1076, #1079), at (A, ω/ω∞) =
(3.446, 0.3476) and (3.526, 0.2115). If they continue to solutions at finite D, those would have echo periods of about
2.88 and 4.73 Δ∞, and uniqueness would fail.

Untested:

- whether they persist as M grows;
- whether they persist at finite D;
- how many zeros their profiles have per half period.

None of this can mislead a jump near Δ ≈ 1.67.

### 4.4 EEG's NLO condition against EEG26b's own profile

`nlo_condition_check.py` evaluates R = (β″ + 3β′)/(3β′) at the zeros of a profile, with τ-derivatives. Eq. (20) sets
R = 0. It also gives the Δ that Eq. (20) would select for the profile's shape, Δ = −2πF_θθ/(3F_θ):

| profile | R at its zeros | Δ selected by Eq. (20) |
|---|---|---|
| EEG26b Eq. (16), q 0.0551, δ 0.0985 | 0.9446 | 0.0911 |
| our tuned D = ∞ profile F (SECOND_ORDER §1.2) | 0.9460 | 0.0887 |
| LW Eq. (4) = EEG Eq. (21), the control | 3.6×10⁻⁸ | 1.0 |
| the control with its third harmonic dropped (`--mutate=drop3`) | 1.0 | — |

So EEG26b's own fitted profile, applied to EEG's Eq. (20), selects a period 18 times shorter than EEG26b's Δ₀ = 1.6439.
EEG26b does not mention Eq. (20), and LW v2 still relies on it.

This agrees with the README ("Relation to Ecker, Ecker & Grumiller"):

- f ≤ 1 holds for every regular solution, so it cannot select.
- The double-double states give R = 0.98–1.06 at D 12–60.

Eq. (20) also explains EEG's conjecture Δ → 0. A profile nearly antisymmetric about its zeros has F_θθ ≈ 0 there, and
Eq. (20) then gives Δ ≈ 0. This is a reading, not a statement of theirs.

### 4.5 Other points that do not follow

- **Eq. (8) is not a bound in the limit.** ḡ = (n − 1)⟨a_h² − 1⟩ = N₀ε + O(ε²), so Δ∞ = 4πμ∞ exactly. The "≳" is only as
  good as μ₀.
- **μ₀ and the ∞ entry are biased high** (`eeg_consistency.py`):
  - 4π × 0.130818 = 1.6439075 is 1.37×10⁻⁵ above our Δ∞, which is 2.2 times the rounding of μ₀'s last digit (6.3×10⁻⁶).
  - Table I's ∞ entry, 1.643911, is 1.73×10⁻⁵ above. That is 1150 times our ±1.5×10⁻⁸, a spread over fit choices,
    conditional on analyticity in 1/D.
  - Both are consistent with a fit that inherits the table's bias, which grows with D (§4.1).
- **The printed NNLO.** EEG26b's Π₃ (S46) is wrong (#142), and β is printed for β̃ from (S43) on. This was known and is
  not new here.
- **The scalar's normalization.** EEG26b's Eq. (1) and Eq. (5) disagree (RT_FORMULATION §2). This was known; Δ is
  unaffected.
- **"Analytic DSS solutions".** EEG's title and Summary ("a family of exact DSS solutions (7)-(13) of Einstein gravity
  in the large-D expansion") present the family as solutions. Every order admits any β and any Δ, and EEG26b's outlook
  needs e^{−1/ε} terms for "large-D numbers from first principles". So the family is the outer expansion, and the
  selection lies beyond all its orders (README, Conjecture 1; SECOND_ORDER §6). This is a reading of their text, not an
  error in an equation.

## 5. Connections to our results (question 4)

- **Δ∞.**
  - EEG26b's Δ₀ ≈ 1.6439 and Eq. (8) agree with Δ∞ = 1.643893746 ± 1.5×10⁻⁸ in their printed digits.
  - Their extra digits (1.643911) do not (§4.5).
  - The μ₀ implied by ours is Δ∞/4π = 0.130816908.
- **The identity.** EEG26b's Eq. (3) is RT_FORMULATION §2's Δ = 4πμ/(1 − (n − 1)⟨a_h² − 1⟩), with ⟨ψ₋₁²⟩ = ḡ. That note
  already credits it as "our form of EEG's Eq. (3)".
- **The 1/n coefficient.**
  - EEG26b's Eq. (15) is the O(ε) term of the identity: N₀ = lim (D−1)ḡ = (Δ₁ − 4πμ₁)/(4πμ₀) = ⟨β̃₀²⟩.
  - Theirs is 7.41579. Ours, from the tuned profile (4Σ|F_ν|²), is 7.41643.
  - Our Δ₁ = 10.3753 then implies μ₁ = −0.14455, against their −0.1449.
  - Control `--mutate=norm` (a factor 2) fails the comparison.
- **The horizon exponent.**
  - σ_R = (n/2 − 1 + ḡ)/(1 − ḡ) (LARGE_D_BASICS §2) equals (D − 4 + 2ḡ)/(2 − 2ḡ).
  - That is EEGJ's δ with ⟨ψ₋₀²⟩ = ḡ, and EEG26b's (D − 4 + f(D))/(2 − f(D)) with f = 2ḡ.
  - With N₀ it is σ_R = n/2 − 1 + N₀/2 + O(1/n) = n/2 + 2.71. That gives 133.5 at D 263.6, against #1086's measured
    133.41.
- **LW's Floquet exponent.** At leading order, c_x(1) = ∂_x ln f(τ, 1) = Ω_LO(x = 1) = β²/(1 + β²), from EEG's
  Eqs. (3b), (7) and (10). So λ = ⟨1/(1 + β²)⟩ = 1 − ḡ = 4πμ/Δ. LW's exponent is RT's invariant in leading-order form
  (checked by hand).
- **EEG26b's outlook is our inner problem.**
  - "extend the solution beyond the SSH and … match with the Rozali–Way regime" is the D = ∞ inner problem at x ∼ √n
    (README, "The inner problem at large x").
  - Their "instanton corrections e^{−1/ε}" are the Stokes jump e^{−nΦ}, whose cancellation fixes Δ's coefficients
    order by order (SECOND_ORDER §6, a reading).
  - This records a correspondence of content and makes no claim of priority.
- **γ∞ = 1/2.**
  - EEGJ assume a single unstable mode (Sec. 2, 3.5).
  - EEG26b quote γ → 1/2 from this project, "not yet verified by human analysis".
  - Neither paper has the r²-shift mode of Proposition 5.

## 6. Ranked leads (question 5)

Every cost is on this Mac at nice 10, one process. The thing to beat is the chain's continuation at about 4 h per unit D
near D 260 (#1088).

| rank | lead | what it gives | test that can fail | cost |
|---|---|---|---|---|
| 1 | **The closed-form seed** (§3): EEG26b's corrected NNLO with F, K₁, K₂ and Δ(n), mapped to (a, c, p, m) by SECOND_ORDER's map | A guess at any D with no continuation: rms 3.4×10⁻⁵ at D 400, 2.7×10⁻⁷ at 2000, 2.1×10⁻⁹ at 10000 (ε^3.02 law, extrapolated) | Refit K₁, K₂ on states with D ≤ 200 only and predict the stored states at D 245.64 and 263.6 (`nnlo_closed_form.py compare --test=...`, after extracting their fields with `nnlo_vs_solution.py`). Pass if the rms of Ω is within 2× of 1.5×10⁻⁴ and 1.2×10⁻⁴. Then compare, at the same D, with Track A's rescaled-state guess, and take the better | 1–2 h (fields from the octo-double states, then one compare) |
| 2 | **Higher orders of the outer expansion** (EEG26b (S36)–(S39) order by order; each order is unique and affine, #142) | About 25ε per order: about 0.0125 at D 2000 and 0.0025 at 10000. Three more orders would bring the seed to about 2.7×10⁻⁷ × 0.0125³ ≈ 5×10⁻¹³ at D 2000, if the profile were exact | The residual exponent rises by 1 per order on the 63 stored states, under SECOND_ORDER §1.1's planted control, which must recover the planted order | 1–2 days (sympy; β̃ derivatives to order 2K) |
| 3 | **The profile's 1/n coefficients K₃…K₆**: already fitted and dropped (SECOND_ORDER §1.2). Hold them, with the D 226–263 octo-double states added to the fit. Those states' held harmonics differ between chains along near-null directions (SECOND_ORDER §1.1's C6 dropped D 150 and 180 for this), so fit the harmonics ν ≤ 3 from them and the rest from the full-system states | Removes the profile's limit (about 2×10⁻⁷ of F₁ at D 2000, from K₂'s ±1.8) | Fit choices kmax 5–7 and D_lo 15–30 must agree on K₃ to 10%, and a held-out D 263.6 profile must improve by ≥ 3× over K₁ + K₂ alone | 2–3 h |
| 4 | **ks and the digits from N₀ in advance**: σ_R = n/2 + 2.71 + O(1/n), ks = ⌈n/2 + 9.7⌉ | Track A's horizon order at any D without a probe: 209 at D 400, 1009 at 2000, 5009 at 10000 | σ_R from any new state must lie within 0.3 of n/2 + 2.71 at D ≥ 260 (0.1 at 263.6 now) | minutes |
| 5 | **Exact residual, inexact Jacobian inverse** (EEG26b's and RT's practice, §1.1, §1.6): the final accuracy is set by the residual's arithmetic and the inverse needs only enough precision to contract | Newton's per-step cost, where the Jacobian dominates | Already ours for dd (notes/MIXED_PRECISION.md: a whole solve 7.4–8.7× faster). For octo-double, test a qd Jacobian with an od residual at D 245 on the chain state: pass if Newton contracts to the od corrector's 10⁻⁴⁰ in ≤ 2× the steps | half a day, if hp exposes a qd Jacobian beside an od residual |
| — | **A fixed-degree global scheme (EEG26b's, or RT's at any fixed N)** | Not a route: its grip falls like n^−(N+1) (§4.2), and EEG26b's own error grows linearly from D 68 (§4.1) | — | — |

**What the literature does not offer.** No published formulation is D-uniform. EEG26b's outlook itself defers the
selection to e^{−1/ε} terms and to matching beyond the SSH, which is the project's inner problem. Leads 1–3 make Track
A's guess good at any D. None of them removes the need to resolve the fast mode with order n/2 degrees of freedom
(§4.2), or the 0.155n digits of the mass mode in our horizon block (LARGE_D_BASICS §4). Those remain Track B's problem.

## 7. Proposed corrections to our notes (for the coordinator)

- **notes/LIT_REVIEW.md, #802 addendum, EEG26b row, "Not there".** Replace "any convergence test in N_Fourier or
  N_Chebyshev at large D" with: "the mode numbers and the D of the stated check ('we checked that varying the number of
  Fourier or Chebyshev modes does not alter these digits', SM introduction, for D > 28)".
- **notes/LARGE_D_BASICS.md §6.1.** "EEG26b's Chebyshev degree 80 is fixed while σ_R ≈ n/2 passes 80 near D 157" is not
  borne out as a threshold. Their departure grows linearly from D ≈ 68 with no step at 157 (this note §4.1). A fixed
  degree's grip falls like n^−(N+1) with no threshold (§4.2).
- **notes/RT_FORMULATION.md §5(2).** Supported. The free-field collocation gives the law at fixed N, n^−(N+1).
  §4.2 above records the unexplained gap to §4's coarse-grid measurement.

## 8. Reproduction

From the repository root, with `/Users/nathan/venv-314/bin/python` (or any Python with numpy and mpmath):

```
python3 solver/lit_gaps/eeg_gap_shape.py hp/data/delta_published_compare.csv     # 4.1; exit 0
python3 solver/lit_gaps/eeg_gap_shape.py hp/data/delta_published_compare.csv --mutate=step   # finds the step, exit 0
python3 solver/lit_gaps/free_field_cheb.py --N=24                                  # 4.2, about 2 s
python3 solver/lit_gaps/free_field_cheb.py --N=40                                  # 4.2, about 6 s
python3 solver/lit_gaps/free_field_cheb.py --N=24 --mutate=planted                 # singular, exit 0
python3 solver/lit_gaps/nlo_condition_check.py                                     # 4.4; exit 0
python3 solver/lit_gaps/nlo_condition_check.py --mutate=drop3                      # control R = 1, exit 0
python3 solver/lit_gaps/eeg_consistency.py                                         # 4.5, 5, 6; exit 0
python3 solver/lit_gaps/eeg_consistency.py --mutate=norm                           # N0 disagrees, exit 1
python3 solver/lit_gaps/cheb_branch_tail.py                                        # the withdrawn model, a record
python3 solver/lit_gaps/cheb_branch_tail.py --mutate=formula                       # its control fails, exit 1
```

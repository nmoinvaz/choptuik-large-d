# Independent check of the write-up of arXiv:2611.07007v1 Eq. (16) (issue #165)

The checker's closing summary on #165, committed by the coordinator.


The write-up's mathematics is right. All five forms equal Eq. (16), and so do the ellipse, the poles, the ratio −w and the time asymmetry of δ. Its numbers reproduce from the data file, except one digit (0.05509 should be 0.05508).

Four statements need fixing. A* is not a fitted parameter. "None of these points is in the paper" overlooks the paper's own A_n ≃ A* qⁿ and φ_n ≃ n(π + δ), which together are the ratio −w. The sign in "w = F₃/F₁" rests on a time origin the write-up does not state. Form E is the scalar field of Eq. (12) only up to the factor 2μ₀.

**What ran.** `check.py` exit 0 (28 pass, 3 expected fails that document discrepancies), and `controls.py` exit 0 (5 planted profiles, then 15 mutations). Each run takes about 6 s on one core under `nice -n 10`, using under 0.3 GB. The paper was read in full: the v1 PDF, all 14 pages including (S1)–(S62) and Table I, with display equations checked against the HTML. Shared input: `solver/eeg_profile/data/tuned_profile_A2.72.txt` only.

- Code: `solver/independent_eeg_eq16/check.py`, `controls.py`, on `wip/i165` at a03671f.
- Raw: choptuik-large-d-raw `I165-mac/` at a11a092 (merged as a11a092).
- REPORT.md was not written, because the session's tooling refuses report files from subagents. Its content is this comment.

### Claims

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| 1 | Form A = Eq. (16) | confirmed | sympy on rational functions of z = e^{iθ}; mpmath at 25 real and complex points, max rel. diff 2e-51 (tol 1e-40) |
| 2 | h = e^{−iθ} + w e^{iθ} is an ellipse, semi-axes 1 ± q, h″ + h = 0 | confirmed | e^{−iδ/2}h = (1+q)cos φ − i(1−q)sin φ, φ = θ + δ/2 |
| 3 | Form B = Eq. (16) | confirmed | sympy, mpmath 4e-51 |
| 4 | Form C = Eq. (16) | confirmed | sympy by induction with the exact remainder, mpmath 5e-51 |
| 5 | Form D holds | confirmed | sympy, mpmath 3e-51 |
| 6 | Form E is an antiderivative | confirmed | sympy, mpmath derivative and definite integral 4e-51, either branch of √w |
| 7 | B and C are in (S57)–(S58) | confirmed | C, B and Eq. (16) are the three members of (S58); (S57) defines A_n, φ_n |
| 8 | A, D, E not in the paper | confirmed | no ellipse, no ODE for β̃, no closed-form integral; the only arctan is (S57) |
| 9 | Poles (π − δ)/2 + kπ ± (i/2) ln(1/q), simple, the only singularities | confirmed | denominator 2(q + uz²)(1 + quz²), u = e^{iδ}, numerator nonzero at its roots |
| 10 | The strip ½ ln(1/q) sets the convergence | confirmed | \|c_ν\| = A* q^{(ν−1)/2}; it is the paper's α/2 |
| 11 | Time reversal flips δ; no shift removes it | confirmed | β̃(−θ; δ) = β̃(θ; −δ); invariant arg(c₃ c̄₁³) = π + δ; min rms asymmetry over shifts 7.6e-3 A* |
| 12 | The one profile with all ratios −w | confirmed | ratios −w to 5e-40, converse by C; for a given A* and the paper's origin c₁ > 0 |
| 13 | "None of these points is in the paper, except that its supplement relates q to the decay" | refuted in part | main text after Eq. (16): e^{−αn}, α = −ln q. After (S57): A_n ≃ A* qⁿ and φ_n ≃ n(π + δ), which is claim 12's ratio −w |
| 14 | D = ∞ singularity close to the model's pole | confirmed | 1.5344 ± 1.4284i against 1.5215 ± 1.4493i |
| 15 | Not only a pole, a branch point there | confirmed as numerical evidence | leading term a pole (γ = −0.0005). The ratio drift is a power law, slope 4.09 (planted pole with branch: 3.80 for truth 4), not an exponential (planted second pole). Several further poles are not excluded |
| 16 | F₃/F₁ = 0.05509 e^{0.0988i} | refuted in the 4th digit | \|F₃/F₁\| = 0.05508456; the phase 0.098790 is right |
| 17 | w is F₃/F₁ | unclear | in the paper's frame (c₀ > 0, s₀ = 0) Eq. (16) has F₃/F₁ = −w; the data give +w. True if the data's origin is a quarter period from the paper's. Read in the paper's frame, the data would be q → −q, a different shape |
| 18 | A* = √2A | confirmed | header ⟨β̃²⟩ = 2⟨F²⟩, and (A²/2)Σ\|F_ν/F₁\|² = ⟨F²⟩ to 1e-7, so √2A is β̃'s fundamental. (S61) on the data gives 3.84666 = √2A |
| 19 | A = 2.720 from the selection rule | unclear | the file labels the profile by A = 2.72 and reproduces ⟨F²⟩ with exactly that value. Whether the rule outputs it is not in the file |
| 20 | Agreement with q, δ, A* | confirmed, with caveats | table below |
| 21 | One-zero form within 3% of every \|F_ν\|, ν ≤ 19 | confirmed | max 2.91% at ν 19 (A* = √2A), 2.94% with A* = 3.8455 |
| 22 | Rational forms converge, more slowly than for a pole | partly supported | the data rule out one or two poles at 1e-4. They cannot show convergence (10 harmonics, floor about 1e-5) or exclude three or more poles |

### The parameters

The ± are rounding half-widths of printed digits. The paper gives no fit uncertainties.

| quantity | data (own code) | paper | difference |
|---|---|---|---|
| \|F₃/F₁\| | 0.05508456 | q = 0.0551 ± 5e-5 | −1.5e-5 |
| arg F₃/F₁ | 0.098790 | δ = 0.0985 ± 5e-5 | +2.9e-4 (a different estimator) |
| (S62) fit on the data, q | 0.055097 | 0.0551 | −3e-6 |
| (S62) fit on the data, δ | 0.098664 | 0.0985 | +1.6e-4 |
| √2A | 3.84666 | A* = 3.8455 | +3.0e-4 relative |
| ⟨β̃²⟩ against N₀ (S59) | 7.420923 | 7.415787 from the printed μ₀, μ₁, Δ₁ | +6.9e-4 relative |
| (S61) from the printed inputs | | 3.84533, rounding range 3.84442–3.84623 | √2A is 4.3e-4 above the range |

**What fixes A*.** The main text after Eq. (16) and (S59)–(S61) fix it by the L² norm, A* = √(2N₀(1 − q²)), where N₀ = (Δ₁ − 4πμ₁)/(4πμ₀) comes from the fits (7) and (9). In the model A* is also the fundamental's amplitude, so √2A, the data's fundamental of β̃, is a fair comparison. The model-free comparison is N₀ against ⟨β̃²⟩, which differ by 6.9e-4. For δ, the like-for-like comparison is the (S62) fit applied to the data, not arg F₃/F₁.

### Eq. (16) against the data

| ν | paper's q, δ, A*: \|C/C₁₆\| − 1 | arg | w = F₃/F₁, A* = √2A | one-zero form |
|---|---|---|---|---|
| 5 | +3.82% | −0.0194 | +3.85% | 0 (fitted) |
| 7 | +8.14% | −0.0431 | +8.20% | +0.33% |
| 13 | +22.5% | −0.1185 | +22.7% | +1.58% |
| 19 | +38.9% | −0.1952 | +39.2% | +2.91% |

### Singularity and rational forms

The ratio moduli rise by D_ν = 3.90e-5, 1.23e-5, 4.98e-6, 2.35e-6, 1.27e-6, 6.93e-7 at ν = 5, …, 15. Against log ν, log D lies on a line of slope −4.09 (rms 0.020), but not against ν (rms 0.178). A pole with a weaker branch point at it, F_ν ∝ ρ^ν(1 + aν⁻²), gives slope −4. Only moduli were used, because the file prints phases to 5 decimals, the size of the late drift.

The table gives the maximum relative error of the Padé forms [L/M] in z = e^{2iθ} on the harmonics they were not built from, up to ν 19.

| L | [L/1] data | [L/2] data | [L/2] planted pole + branch | [L/2] planted two poles |
|---|---|---|---|---|
| 0 | 3.3e-1 | 5.8e-2 | 3.8e-2 | 1.0e-1 |
| 1 | 4.7e-2 | 7.8e-3 | 4.5e-3 | 5.2e-5 |
| 2 | 1.2e-2 | 1.6e-3 | 8.7e-4 | 4.4e-5 |
| 3 | 3.7e-3 | 4.5e-4 | 1.8e-4 | 7.2e-5 |
| 4 | 1.4e-3 | 6.3e-5 | 5.4e-5 | 3.0e-5 |

### Controls

| planted profile (data format, rounding, noise) | criterion | verdict |
|---|---|---|
| exact Eq. (16) | deviation < 1e-4; (S62) recovers q, δ | pass (3e-7; q, δ to 1e-6) |
| smooth drift | the exact criterion must fail | fails as required (0.24) |
| kink at ν 13 | the one-zero 3% criterion must fail | fails as required (5.0%) |
| pole with branch | power law, slope 3.5–4.5 | pass (3.80) |
| two poles | exponential, λ within 10% | pass (0.543 against 0.548) |

`exact` failed on its first run, because it was planted with A = 2.72 and compared with A* = 3.8455. It was replanted with A = A*/√2. The drift classifier was rewritten after a first look at the data, since 5-parameter complex fits could not reach the data's precision. Its planted criteria were then written before the planted and data runs.

All 15 mutations flip their target check, and each makes `check.py` exit 1.

| mutations | target |
|---|---|
| A_conj, B_sign, C_sign, D_delta, E_noi, E_sqrt | forms A–E, mpmath and sympy |
| ellipse_axes, ellipse_ode | the ellipse, h″ + h = 0 |
| pole_sign, ratio_plus | pole location, the ratio −w |
| delta0, treversal | no shift removes δ, time reversal |
| data_kink, data_A, data_frame | one-zero 3%, A = \|F₁\|, the (S62) fit |

### Clarity

A reader who knows the paper can follow the forms and the singularity argument. They cannot follow the parameter bullet without the repository's conventions.

1. "Their fitted parameters are q = 0.0551, δ = 0.0985 and A* = 3.8455." A* is not fitted. Suggest "Their parameters are q = 0.0551 and δ = 0.0985, fitted to four harmonics by (S62), and A* = 3.8455, fixed by the L² norm through (S59)–(S61)."
2. "B and C, are in the paper's supplement, (S57)–(S58)." Suggest "B and C are the second and first members of (S58)."
3. "semi-axes 1 ± q, since h'' + h = 0." The ODE gives an ellipse, and the moduli 1 and q give its axes. Suggest "h solves h'' + h = 0, so for real θ it traces an ellipse about the origin. Its semi-axes are 1 + q and 1 − q, the major one at angle δ/2."
4. "each pair in the same ratio −w" is ambiguous. Suggest "with β̃ = Re Σ c_ν e^{iνθ} and c₁ = A* > 0 as in the paper, each ratio c_{ν+2}/c_ν equals −w."
5. Form D, "A linear equation it solves." Suggest adding "Its solutions are (a cos θ + b sin θ)/(1 + q² + 2q cos(2θ+δ)), and Eq. (16) is the one with a = A*(1 + q cos δ), b = −A* q sin δ."
6. Form E, "the central scalar field of their Eq. (12)." Eq. (12) integrates over τ = 2μ₀θ. Suggest "which times 2μ₀ = Δ₀/2π is, up to a constant, the leading-order scalar field of their Eq. (12)."
7. "None of these points is in the paper, except that its supplement relates q to the decay of the harmonics." This is overstated. Suggest "The paper finds A_n ≃ A* qⁿ and φ_n ≃ n(π + δ) after (S57), which is the ratio −w below, and gives α = −ln q after Eq. (16). The other points are not in it."
8. "The analyticity strip, ½ ln(1/q)". Suggest "The strip's half-width, ½ ln(1/q) = α/2 ≈ 1.45".
9. "Its harmonics point to a branch point at the same place." Suggest "Its harmonics point to a weaker branch point at that singularity, since the ratios drift as ν⁻³, not exponentially (numerical evidence, ν ≤ 19)."
10. "It is the echo's time asymmetry." Suggest "It is the model's only time-asymmetric parameter, unless δ is a multiple of π."
11. "In this repository's variables, w is the first harmonic ratio F₃/F₁ ... and A* = √2 A." Neither the frame nor the √2 is stated. Suggest "Here F = β̃/√2 and A = |F₁|. Our time origin is a quarter period from the paper's, where F₃/F₁ = −w, so w = F₃/F₁ and A* = √2A." This holds only if the repository confirms the frame (recommended task 1).
12. "F₃/F₁ = 0.05509 e^{0.0988i}". It should read 0.05508. Also suggest adding "the paper's fit (S62) applied to our harmonics gives q = 0.05510, δ = 0.0987".
13. "The D = ∞ equations with our selection rule give ... A = 2.720." The write-up never defines the selection rule. Suggest one sentence on the rule and on where A comes from.
14. "Rational forms with more terms do converge to the profile, though more slowly than if its singularity were only a pole." Suggest "Over ten harmonics, Padé forms with two poles leave errors from 8e-3 down to 6e-5, where a profile with two poles would be reproduced exactly. Convergence itself is not shown."
15. The commands under "Checks" give no directory. Suggest naming the folder.

### What I read in the repository

- AGENTS.md and this issue.
- `solver/eeg_profile/data/tuned_profile_A2.72.txt` in full. I saw its derived lines (the paper's fit, the deviations, the [1/1] model and the singularity), but the code reads only the columns ν, |F_ν/F₁|, arg and truncation, and the header values A, ⟨F²⟩ and ⟨β̃²⟩. My numbers agree with those lines (the (S62) δ to 4e-6).
- A listing of the file names in `solver/` and the recent commit titles.
- Not opened: notes/EEG_PROFILE.md, notes/SECOND_ORDER.md, the rest of solver/eeg_profile/, solver/independent_eeg_nnlo/, any eeg_eq16_*.py, and the write-up's link.
- Shared with the work under test: the data file and the paper.

### Discrepancies (reported, not fixed)

1. |F₃/F₁| is 0.05508, not 0.05509.
2. "w = F₃/F₁" has the opposite sign of the paper's frame (S53, S57). The quarter-period map it needs is not stated, and the file cannot confirm it.
3. √2A = 3.84666 lies 4.3e-4 above the range that the paper's printed inputs allow for A* (3.84442–3.84623). The difference in N₀ is 6.9e-4.
4. The "not in the paper" sentence misses the paper's main-text α = −ln q and its supplement's φ_n ≃ n(π + δ).
5. The write-up calls A* fitted, and the paper says it is not.

### Recommended tasks

1. **State the harmonic frame of the D = ∞ data.** What: give arg F₁ in the repository's time origin, and compute the shift-invariant arg(F₃ F̄₁³). Why: the write-up's w = F₃/F₁ and its one-zero parameters assume a quarter-period map that is not recorded, and without that map the profile reads as q → −q. Acceptance: arg(F₃ F̄₁³) within 1e-3 of π + 0.0985 (it fails if near 0.0988), with a control that drops the map and must fail. Depends on #165.
2. **Record how A = 2.72 is chosen.** What: say in the data file whether A is the selection rule's output, with at least 6 digits, or an input. Why: the write-up calls it an output, and the 3e-4 A* comparison is at the level of A's 4-digit rounding (1.8e-4). Acceptance: the rule's A to 6 digits, and √2A against the paper's A* recomputed. It fails if |√2A/3.8455 − 1| > 1e-3. Depends on #165.
3. **Derive the √2 between F and the paper's β̃.** What: one sentence and a sympy check relating the repository's normalization to Eq. (1) (4πG = 1), (S25) and (S34). Why: a reader of the paper cannot check A* = √2A. Acceptance: the factor computed from both normalizations equals √2 (it fails on 1 or 2). Depends on #165.
4. **Export the D = ∞ harmonics at full precision and repeat the singularity analysis.** What: F_ν to at least 12 digits in both modulus and phase, to ν 25 with the truncation estimate, then rerun the drift and Padé analysis using the phases too. Why: the branch-point reading rests on 6 ratio differences of moduli only, and the 5-decimal phases are the floor. Acceptance: a power-law class with slope within 0.3 of 4 on ν = 5–21, with the planted two-pole control still classified exponential. It fails if the class changes. Depends on #165 and #133.
5. **Revise the write-up per the clarity list.** What: apply items 1–15 and rerun this checker. Why: items 1, 7, 11 and 12 are errors or overstatements a reader of the paper would catch. Acceptance: `check.py` exits 0 with the documented-discrepancy checks updated to pass, and the mutations still fire. Depends on tasks 1 and 2.


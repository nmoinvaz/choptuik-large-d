# Literature review of the novelty and priority claims

Review of 2026-10-21, following the `lit-review` protocol of the BootLoops skills repository, with the
`reading-contract` rules.

- **Scope.** The novelty and priority claims in [README.md](../README.md),
  [paper/physics/main.tex](../paper/physics/main.tex), [paper/math/main.tex](../paper/math/main.tex) and
  [PROOF.md](../PROOF.md), plus one claim in [GAMMA.md](../GAMMA.md) that backs Proposition 5.
- **No edits to the claims.** Wording is proposed in §2 for the integrator.
- **How the papers were read.** Every statement about a paper's content comes from a page opened in this session
  through the `WebFetch` tool, cited by URL.
- **What the quotes are.** That tool returns an extraction of the page, with verbatim quotes limited to about 120
  characters. The quotes below are as the tool returned them, marked as quotes. They were not byte-compared against the
  PDFs, which is the residual risk of this channel.
- **Labelled inferences.** Inferences of mine are marked *inferred*.
- **Not a fresh reader.** This review was not run in a context free of the authors' framing: I read the README, the
  paper and PROOF.md first.

## 0. The claims (pinned before the search)

| # | Claim | Exact wording and location |
|---|---|---|
| C1 | All-orders freedom of the 1/D expansion (Prop. 1) | README "Large D": "The all-orders statement (Proposition 1), [...] were not found in print." Paper abstract: "We show that the $`1/D`$ expansion of the self-similar field equations can be carried to all orders for any period and any time profile, so the period (and likewise $`\gamma`$) is selected beyond all orders in $`1/D`$." |
| C2 | Closed-form light-cone coefficient $`C(n,s)`$ (Prop. 2) | README: "the exact light-cone coefficient in every dimension (Proposition 2) [...] were not found in print"; after Prop. 2: "what Proposition 2 adds is the closed-form light-cone coefficient for every D and harmonic." |
| C3 | The Stokes reading (Props. 3–4 and the inner problem) | README: "the Stokes reading (Propositions 3–4 and the inner problem) were not found in print." Paper abstract: "which we identify as the Stokes multiplier of a mode born at large self-similar radius"; intro item 3: "a proof that the late orders of the $`1/D`$ series encode that coefficient as a Stokes multiplier". |
| C4 | An $`n`$-free inner problem whose Stokes multipliers select the period | Paper intro item 4: "An $`n`$-free inner problem whose Stokes multipliers, by a conjecture [...] select the period"; abstract: "the vanishing of the Stokes multipliers of an $`n`$-free inner problem". |
| C5 | Proposition 5 makes Rozali–Way's heuristic exact | README: "Proposition 5 turns Rozali and Way's heuristic for γ → 1/2 into an exact statement about the discretely self-similar solution." Paper: "the exact DSS form of the heuristic of Rozali and Way". Prop. 5(i): the inner equations "are the nonlinear heat equation". GAMMA.md §2: "No published work derives λ∞ = 2 for the DSS solution, or the 1/n coefficient." |
| C6 | Numerics: $`\Delta(D)`$ to $`D = 61`$ and a finite limit, against the published $`\Delta\to0`$ | Paper abstract: "$`\Delta(D)`$ [...] at 857 points $`8.25 \le D \le 61.03`$" (implicit priority above $`D = 14`$); "A power-law approach to $`\Delta = 0`$, as conjectured in the literature, is excluded by the data". README: "the published conjecture Δ → 0 as D → ∞ is not supported". |
| C7 | The D = 5 discrepancy with Ecker et al. is a resolution effect | README/paper: "a 6×10⁻⁵ difference at D = 5 that the published higher-resolution data trace to resolution". |
| C8 | The NLO condition of Ecker, Ecker & Grumiller does not select $`\Delta`$ | Paper Sec. eeg: "Within our framework this condition does not select $`\Delta`$"; README: "So Eq. (20) does not select Δ, and Proposition 1 stands." |
| C9 | What is new in the proof problem | PROOF.md Summary: "**What is genuinely new here** is that the unknown is a function of two variables (x, θ), periodic in θ, [...] Neither template has that." |
| C10 | First rigorous derivation of an echo period | PROOF.md Summary: "(A) is a substantial but standard computer-assisted proof; (B) is a research-level paper in exponential asymptotics; together they would be the first rigorous derivation of a critical-collapse echo period." |

(The quotations are copied from the source files; their LaTeX math is shown here in GitHub's math syntax.)

## 1. Closest works

Proximity grades follow the protocol: P2 shares a component; P3 is the same question with other tools, or a
neighbouring question with the same tools; P4 overlaps part of a claim; P5 contains it.

| Work | Grade | Read | What it contains that bears on the claims (verbatim quotes, with locators) |
|---|---|---|---|
| **M. Rozali, B. Way**, *Gravitating scalar stars in the large D limit*, JHEP 11 (2018) 106, arXiv:1807.10283 ([ar5iv](https://ar5iv.arxiv.org/html/1807.10283), [Springer](https://link.springer.com/doi/10.1007/JHEP11(2018)106)) | **P4** for C4 and C5 | Full text through Sec. 2.3 (the extraction stopped there; the later sections, including the AdS part, are unread) | **Setup.** The large-$`D`$ limit of the Einstein–scalar system with time rescaled ("Now we move to the time coordinate" $`\tau=\sqrt D\,t`$, Sec. 2.2) and $`r`$ not rescaled. **Leading order.** At leading order they get $`D`$-free equations, Eqs. (9a)–(9c), first order in $`r`$ and second order in $`\tau`$, e.g. $`\partial_r\varphi_0 = \frac{r}{\sqrt{1-\delta_0}}\partial_\tau\big(\frac{\partial_\tau\varphi_0}{\sqrt{1-\delta_0}}\big)`$ and $`\partial_r\delta_0 = -2r(\partial_\tau\varphi_0)^2`$ (transcribed by the tool). **Heat form.** Sec. 2.2, at Eq. (9c): "the scalar field equation resembles a heat equation with spatial and temporal coordinates swapped." **Exact solutions.** Sec. 2.3: "there is an exact analytic solution to the equations" (the oscillatons, Eq. (10)). **γ = 1/2.** The divergence where $`\epsilon^2e^{-\omega^2r^2}=1`$ gives $`r_{\rm div}=\sqrt{2(\epsilon-1)}`$ (Eq. (12)) and hence $`\gamma = 1/2`$, with the caveats "While this simple result is suggestive, it remains unclear whether" (Sec. 2.3) and "we therefore cannot use a mass as a measure for the critical exponent". **No DSS.** DSS and $`\Delta`$ are not discussed at large $`D`$. *Inferred:* their regime $`r = O(1)`$, $`\tau = O(1)`$ is the paper's inner region, since $`x = r/(-t) \sim \sqrt n`$ there. Written in $`s = r^2/2`$, their Eq. (9c) has the structure of the paper's Prop. 5(i), $`Y_s = c^{-1}\partial_t(c^{-1}\partial_t Y) - Y^3/4`$, up to a field redefinition I did not carry out. |
| **C. Ecker, F. Ecker, D. Grumiller**, PRL 136, 191401 (2026), arXiv:2601.14358 ([HTML v1](https://arxiv.org/html/2601.14358v1)) | **P4** for C1 and C8 | Full text (HTML) | **Expansion.** $`\epsilon = 1/(D-1)`$ (Eq. (2)); the scalar field scales as $`\psi = \epsilon^{-1/2}\sum\psi_n\epsilon^n`$ (Supplement Eq. (S3)). **Leading-order freedom.** "any periodic function $`\beta(\tau)`$ with any echoing period $`\Delta`$ solves our equations of motion" ("Properties of leading order solution"). **NLO selection.** Eq. (20) gives $`\Delta`$ as the ratio of $`\lvert\beta''\rvert`$ to $`3\lvert\beta'\rvert`$ at $`\beta = 0`$, derived from "If $`\beta''+3\beta'`$ does not vanish simultaneously with $`\beta`$" (then $`f\gt1`$). **Limits.** "Existing data [16, 17, 29] suggest $`\Delta\to0`$ as $`D\to\infty`$"; "Numerical and theoretical arguments suggest" $`\lim\gamma=\tfrac12`$, "hinting at a CSS limit [23]" (Summary). **Convergence.** "it would be important to understand whether the 1/D series converges"; "indicating that the large-D expansion is not convergent for $`D \lesssim 100`$" (Supplement). **Not there.** No Stokes phenomena or beyond-all-orders selection; no hypergeometric solution, since the leading-order solutions are elementary. |
| **C. Ecker, F. Ecker, D. Grumiller**, *Angle of null energy condition lines in critical spacetimes*, Phys. Rev. D 112, L021505 (2025), arXiv:2411.09233, doi:10.1103/vpyn-b2fn ([HTML v1](https://arxiv.org/html/2411.09233v1)) | P2: a local quantity of the same critical solution, no claim of ours overlaps | Full text (HTML v1), read through a summarizing fetch, 2026-11-04 | **Quantity.** The lines where the null energy condition saturates, UV/x² = 0, are the zero-curvature lines of R^(D) = −(D−2) e^{2τ} e^{−ω} UV/x², and they meet at the centre x = 0 at a time τ₀ where v₁(τ₀) = 0. **Result.** α = 2 arccot(D − 1) for every D > 3, with α = 2/(D − 1) + O(1/D³) at large D. It is derived locally from a quadratic Taylor expansion at x = 0, "we have no proof of Taylor-expandability around x=0 for arbitrary D>3", from v₁ periodic with zero average and two simple zeros per period. "the dependence on the initial data f₀ and v₀ drops out", and Δ and γ do not enter. **Check.** D = 4 only, read from a plot: "α ≈ 37° ≃ 0.64". **Bearing.** The angle follows from regularity at the centre and the zeros of v₁ being simple, both of which our solver imposes or would show directly. So computing it from our states tests little beyond those, and it is not an independent check of Δ(D). The open part is their assumptions at D ≠ 4, which our states could confirm (two simple zeros of v₁ per period, D 3.04–85) at no cost of new runs. |
| **C. Ecker, F. Ecker, D. Grumiller, T. Jechtl**, SciPost Phys. 20, 176 (2026), arXiv:2602.10185 ([PDF](https://arxiv.org/pdf/2602.10185), [v2 HTML](https://arxiv.org/html/2602.10185v2)) | P3 for C6 and C7 | Full text to about 88k characters; the reference list was not reachable | Values at Eqs. (69)–(71) and in the abstract (see REF_CHECK.md §4). Sec. 4.1: "The expansion order around the SSH has to be increased above D=4.8". Sec. 4.2: perturbations "cannot be computed any more with the current method" for $`4.81 \le D \le 5.38`$. Sec. 5.2: "compatible with the conjectures $`\Delta \to 0`$ both for $`D \to \infty`$ and for $`D \to 3^+`$". Sec. 5.3: "compatible with the conjecture" $`\lim\gamma = \tfrac12`$, "see for instance [12]". Ref. [12] could not be identified: the reference list did not come through. An aggregator ([pith.science](https://pith.science/paper/2602.10185)) lists Rozali–Way among the works it cites; unverified. |
| **S.-H. Lai, W.-Y. Wen**, *Polyakov backreaction in large D discretely self-similar collapse*, arXiv:2610.04960 (October 2026) ([HTML](https://arxiv.org/html/2610.04960)) | P3 for C1 and C8 | Full text (HTML) | Builds on EEG: "Following Ref. [12], we work in the past patch of the critical spacetime" (Sec. II.A). "the next-to-leading large-D consistency condition selects $`\Delta = 1`$" (Sec. II.A, unit normalisation). "Uniqueness of the echoing period is not a leading-order statement." (Sec. II.B). The period is "a modulus" fixed by subleading conditions (Sec. IV.C). It does not cite Rozali–Way or this project. **A 2026 paper relies on the NLO selection that C8 disputes.** |
| **C. R. Clark, G. L. Pimentel**, *Black hole critical collapse in infinite dimensions: continuous self-similar solutions*, JHEP 11 (2025) 017, arXiv:2504.10590 ([Springer](https://link.springer.com/article/10.1007/JHEP11(2025)017), [PDF](https://arxiv.org/pdf/2504.10590)) | P3 for C1, C5 and C6 | Full text (PDF extraction) | Large-$`D`$ matched asymptotics for the CSS critical solution, with $`\epsilon = 1/(D-2)`$ (Eq. (2.8)) and regions A/B/C in self-similar time. Footnote 1: "it has not yet been proven that there are no further restrictions from the 'matching' of these asymptotics". CSS-breaking mass exponent: "γ = 2(D−3)/(D−2) ≈ 2 − 2/D" (Eq. (1.5)), with $`M \sim (p-p^*)^\gamma`$ (Eq. (2.1)). The introduction cites [28] for the exponent possibly approaching 1/2. The DSS solution is "only known approximately and numerically" (footnote 5). There is no heat-equation form and no beyond-all-orders discussion. **Not cited in the paper or the README.** |
| **C.-H. Wu**, *Quantum critical collapse in the large D limit*, arXiv:2610.20705 (14 Oct 2026) ([abs](https://arxiv.org/abs/2610.20705)) | P2 | Abstract only | Quantum $`s`$-wave growth in CSS and DSS critical spacetimes at large $`D`$; no $`\Delta`$, $`\gamma`$ or Stokes content in the abstract. |
| **C. Gundlach, D. Hilditch, J. M. Martín-García**, *Critical Phenomena in Gravitational Collapse*, arXiv:2507.07636 ([HTML](https://arxiv.org/html/2507.07636v1)) | P2 | Full text (HTML) | No large-$`D`$ results and no limits of $`\Delta`$ or $`\gamma`$. It does not mention Rozali–Way, Clark–Pimentel or Ecker et al. Sec. IV.A.2: Kol "claims to obtain analytic estimates for γ and Δ in arbitrary dimension". Sec. III.C: the critical solution "was determined accurately in [43] by assuming analyticity at the centre of spherical symmetry and at the past light cone". No exponential asymptotics. |
| **B. Kol**, *Choptuik scaling and the merger transition*, hep-th/0502033, DOI 10.1088/1126-6708/2006/10/017 ([PDF v3](https://arxiv.org/pdf/hep-th/0502033v3)) | P2 | Full text (PDF extraction) | For the black-string merger system: Eq. (3.20) "$`\frac{2\pi}{\Delta} = \sqrt{\frac{10 - D}{D - 2}}`$"; Eq. (3.21) "$`\gamma = \frac{1}{4} \left( 1 + \sqrt{\frac{D - 10}{D - 2}} \right)`$" for $`D \ge 10`$. *Inferred (my arithmetic):* $`\gamma\to1/2`$ there too, as $`D\to\infty`$. There is no formula for Choptuik's $`\Delta(D)`$. |
| **R. Emparan, C. P. Herzog**, RMP 92, 045005 (2020) ([PDF](https://arxiv.org/pdf/2003.11394)) | P2 | Partial: the text extraction stops at Sec. V. **Sec. X.C "Critical collapse and Choptuik scaling" is FULL-TEXT-UNREAD**, so this grade is a ceiling. | Sec. III.F: "The breakdown is non-perturbative in 1/n"; "non-perturbative corrections = O(n^{-n})"; quasinormal modes as "non-perturbative, or trans-series, corrections". Sec. IV.E.1: "resemble (inhomogeneous) heat equations" (black-brane effective equations). The large-$`D`$ community already knows trans-series effects in $`1/D`$, and heat-like effective equations. |
| **R. Donninger**, J. Evol. Equ. 9, 511 (2009), arXiv:0809.5177 ([PDF](https://arxiv.org/pdf/0809.5177)) | P3 for C2 | Full text (PDF extraction) | "We consider the radial wave equation in similarity coordinates within the semigroup formalism"; "the long time behaviour of the solution is governed by the analytic eigenfunctions". This is 3+1 only, with explicit eigenfunctions $`(1-\rho)^{1-\lambda}-(1+\rho)^{1-\lambda}`$. It gives no connection coefficients and no general $`d`$. |
| Donninger, arXiv:0805.0520; Costin–Donninger–Glogić, arXiv:1604.00303; Donninger–Aichelburg, math-ph/0702025; Biernat–Donninger–Schörkhuber, arXiv:1707.09812 ([ar5iv](https://ar5iv.arxiv.org/html/0805.0520), [PDF](https://arxiv.org/pdf/1604.00303), [PDF](https://arxiv.org/pdf/math-ph/0702025v1), [ar5iv](https://ar5iv.arxiv.org/html/1707.09812)) | P1–P2 for C2 | Partial extractions | None gives the free-wave mode equation in $`d`$ dimensions as $`{}_2F_1`$ with a Gamma-function connection coefficient at the light cone. Costin–Donninger–Glogić reduce to Heun equations and state "the general connection problem for these equations is unresolved" (Sec. 4). |
| **R. H. Price, J. Pullin**, *Analytic approximations to the spacetime of a critical gravitational collapse*, gr-qc/9601009 ([ar5iv](https://ar5iv.labs.arxiv.org/html/gr-qc/9601009)) | P2 for C2 and C4 | Partial (truncated before the matching section) | Abstract: "the critical solution can be well approximated by a flat spacetime scalar field solution"; the matching "yields the right order of magnitude for the oscillations". This is an early flat-space (free-wave) reading of the echo. Cited in both papers. |
| **R. B. Paris**, *Asymptotics of the Gauss hypergeometric function with large parameters, I*, J. Classical Anal. 2 (2013) 183–203 ([PDF](https://files.ele-math.com/articles/jca-02-15.pdf)) | P2 for C3 (Prop. 4) | Full text (PDF extraction) | Large-$`\lambda`$ asymptotics of $`F(a+\epsilon_1\lambda, b+\epsilon_2\lambda; c+\epsilon_3\lambda; z)`$. The singularities in the $`\tau`$-plane occur "at $`\tau = 2\pi k i\varepsilon`$" and "$`\tau = \Psi(1/z)+2\pi k i`$" (Sec. 3.2), the analogue of Prop. 4's $`\tau_\ell`$, and there is a Stokes discussion in Sec. 4.2. Large $`c`$ alone is only quoted: "An exception is the case when only the parameter c is large" (Sec. 1). |
| N. M. Temme, *Large parameter cases of the Gauss hypergeometric function* (JCAM 2003) ([ResearchGate page](https://www.researchgate.net/publication/2524192_Large_Parameter_Cases_of_the_Gauss_Hypergeometric_Function)) | P1–P2 | Abstract only | Large $`a`$, $`b`$, $`c`$ cases; no statement on exponentially small terms visible. |
| **M. Reiterer, E. Trubowitz**, CMP 368, 143 (2019), arXiv:1203.3766 ([PDF](https://arxiv.org/pdf/1203.3766), [Springer](https://link.springer.com/article/10.1007/s00220-019-03413-8)) | **P3, contradicts C10 as worded** | Introduction and abstract | "We prove rigorously the existence of a real analytic solution"; "The proof is computer assisted." The introduction, after Fig. 1: "Here $`K \approx 1.72`$ and we prove", followed by Eq. (2), which encloses $`K`$ to 80 digits, "consistent with the numerical result 3.445452402(3) for 2K". **This is a rigorous enclosure of the $`D = 4`$ echo period.** Cited in PROOF.md but not in the paper. |
| T. Taves, G. Kunstatter, arXiv:1105.0878 ([PDF](https://arxiv.org/pdf/1105.0878)) | P2 for C6 | Full text (extraction) | Reliable $`\gamma`$ and echo period for $`D = 4`$–$`7`$; at $`D = 8`$ the values are tentative. "Since we are limited at the moment to D ≤ 7 we cannot make any definitive claims about the asympotic behaviour." Not cited. |
| J. Bland, PhD thesis (2007) ([PDF](https://mspace.lib.umanitoba.ca/bitstreams/319e409d-2cfd-46d5-93ca-2bc913dbbbfb/download)) | P2 for C6 | Abstract and front matter (OCR) | "the critical solution of the theory at the limit of large dimension is discretely self-similar with a period of 5/2". |

## 2. Verdicts, stated against interest

**C1, Prop. 1 (all-orders freedom). New with required engagement.**
- **Prior work.**
  - *EEG.* They have the same outer expansion and the same leading-order freedom, but conclude the opposite at
    next-to-leading order: Eq. (20) "fixes the period".
  - *Lai & Wen (Oct 2026).* They adopt EEG's NLO selection ("selects $`\Delta = 1`$").
  - *Clark & Pimentel.* They flag the open "matching" restrictions in their own CSS expansion.
- **Not found.** No work was found that proves all-orders freedom of the $`1/D`$ expansion for DSS solutions.
- **Required.** The claim stands only together with the Sec. eeg rebuttal and with the ansatz stated wherever the claim
  appears. The abstract states it without the ansatz.
- **Proposed abstract wording:** "Within the scaling measured on the solutions (integer powers of $`1/D`$ after an
  overall $`D^{-1/2}`$), the $`1/D`$ expansion of the self-similar field equations can be carried to all orders for any
  period and any time profile; the next-to-leading-order condition of Ref. [EEG] is satisfied for every period (Sec.
  eeg)."
- **Cite** Lai & Wen (arXiv:2610.04960) in Sec. eeg as a work that uses that condition.

**C2, Prop. 2 ($`C(n,s)`$ in closed form). New as far as this review can see, with required credit; scope the
sentence.**
- **Prior work.** Donninger (3+1) gives the analytic modes of the free radial wave equation in similarity coordinates.
  In $`D = 4`$ these already imply that no time-periodic mode is analytic at the light cone.
- **Not found.** No source in the search gives the $`d`$-dimensional hypergeometric reduction with the explicit
  connection coefficient.
- **Not novel in method.** The coefficient is a direct application of Gauss's connection formula (DLMF 15.8.4).
- **Search coverage.** The mathematical literature on self-similar blowup (Donninger–Schörkhuber, Glogić, Bizoń and
  coauthors) was searched by keyword but not exhaustively (gap statement), so the residual risk is real.
- **Proposed README wording:** "We did not find the explicit coefficient $`C(n,s)`$ for general $`D`$ in print; it
  follows from Gauss's connection formula once the mode equation is recognised as hypergeometric. In $`D = 4`$ the
  absence of analytic time-periodic modes follows from Donninger's analytic-mode analysis."

**C3, Stokes reading (Props. 3–4). New in this application, with credit; Prop. 4's mathematical content is a standard
computation and should be presented as an instance.**
- **Not found.** No critical-collapse or large-$`D`$ paper in the search uses Stokes multipliers or late-order
  (Dingle) analysis for the selection of $`\Delta`$:
  - EEG: none;
  - the 2025 review: none;
  - Clark–Pimentel: none.
- **What the large-D literature already has.** Emparan & Herzog Sec. III.F treat "non-perturbative, or trans-series,
  corrections" in $`1/n`$ for quasinormal modes. That should be credited as the large-$`D`$ precedent for trans-series
  effects.
- **Prop. 4.** It is a Watson/Darboux late-terms analysis of a Gauss function with one large parameter. Paris (2013)
  works out the analogous $`\tau`$-plane singularity structure for large-parameter $`{}_2F_1`$.
- **Proposed paper wording (intro item 3):** "...and, for the leading-order (hypergeometric) series, an explicit
  late-order analysis (Watson's lemma and Darboux's method) showing that its divergence encodes $`C(n,s)`$ as a Stokes
  multiplier."
- **Proposed README wording:** "were not found in print" -> "were not found in the critical-collapse or large-$`D`$
  literature searched (see notes/LIT_REVIEW.md)".

**C4, the $`n`$-free inner problem. At risk as worded: credit Rozali & Way.**
- **The overlap.** Rozali & Way derived the $`D`$-independent leading-order Einstein–scalar equations in the regime
  where time is rescaled by $`\sqrt D`$ (Eqs. (9a)–(9c)). They noted that "the scalar field equation resembles a heat
  equation with spatial and temporal coordinates swapped" and found exact (non-DSS) solutions of it.
- **Same regime.** *Inferred:* that regime is the paper's inner region $`x\sim\sqrt n`$.
- **What is new.** Posing the DSS problem there, its Stokes multipliers, and the selection conjecture.
- **What should not be presented as new.** The inner equations themselves, and the heat-equation form.
- **Proposed paper wording (Sec. inner, first paragraph):** "At $`x \sim \sqrt n`$ the expansion is not uniform; there
  the equations reduce to the $`n`$-free leading-order system of Rozali and Way [RW, Eqs. (9)], written here in
  self-similar variables."
- **Check to make.** Write out the variable map between $`(y,\sigma,c)`$ and $`(\varphi_0,\delta_0)`$, and verify it
  symbolically. I did not do this; it is the cheapest check that the identification is exact.

**C5, Prop. 5. New with required credit; "exact statement" overstated.**
- **Not found.** No source derives an exact $`\lambda = 2`$ mode for the DSS solution:
  - Rozali & Way: oscillaton heuristic, explicitly "suggestive";
  - Clark & Pimentel: CSS, with a mass exponent from CSS-breaking modes;
  - EEG and EEGJ: they report the conjecture;
  - Kol: a different system.
  GAMMA.md's "No published work derives λ∞ = 2 for the DSS solution, or the 1/n coefficient" survives as far as
  searched.
- **The heat-equation form (Prop. 5(i)).** It must credit Rozali & Way (see C4).
- **What is exact and what is conditional.** The mode is exact for the $`D = \infty`$ inner problem. $`\gamma_\infty =
  1/2`$ for the DSS solution follows only under Conjecture 1, uniqueness of the unstable mode and continuity of
  $`\lambda(D)`$. The paper body says this; the README sentence does not.
- **Proposed README wording:** "Proposition 5 gives an exact $`\lambda = 2`$ mode of the $`D = \infty`$ inner equations
  (the leading-order large-$`D`$ system of Rozali and Way, whose heat-like structure they noted), generated by the
  shift of $`r^2`$. Under Conjecture 1, uniqueness of the unstable mode and continuity of $`\lambda(D)`$, it gives
  $`\gamma_\infty = 1/2`$ for the discretely self-similar solution: the DSS counterpart of Rozali and Way's heuristic."
- **Clark–Pimentel in GAMMA.md.** Their exponent ($`2(D-3)/(D-2)`$ for mass, Eq. (1.5)) uses a mass convention and a
  CSS background. Reconcile it with the radius convention before citing it as support or contrast.

**C6, numerics to D = 61 and a finite limit. Supported as far as searched, with required credit.**
- **No computation above D = 14 found.** Prior numerics:
  - Garfinkle et al. ($`D=6`$);
  - Sorkin–Oren ($`D \le 14`$);
  - Bland et al. ($`3.5 \le D \le 14`$);
  - Taves–Kunstatter ($`D \le 7`$, 8 tentative);
  - Bland's thesis;
  - EEGJ ($`3.05 \lt D \lt 5.5`$).
- **Possible competitor.** EEG mention moderate-$`D`$ numerics "[25]" as work in progress. A per-author recency sweep
  of the Vienna group is owed (gap statement), so a near-simultaneous competitor cannot be excluded.
- **The contradiction claim.** "The published conjecture $`\Delta \to 0`$ is not supported" is a fair reading: EEG
  "suggest" it from "existing data", and EEGJ call their data "compatible with" it.
- **Attribution fix for $`\gamma \to 1/2`$.** The intro attributes this conjecture to EEG and EEGJ (REF_CHECK.md §4).
  Proposed wording: "Ecker, Ecker and Grumiller constructed analytic DSS solutions at large $`D`$ and, with [EEGJ],
  conjectured $`\Delta \to 0`$; $`\gamma \to 1/2`$ had been argued by Rozali and Way [RW] and is discussed in
  [EEG, EEGJ, CP]."
- **Add Clark & Pimentel** (JHEP 11 (2025) 017) to the introduction as the other large-$`D`$ critical-collapse
  analysis.

**C7, the D = 5 discrepancy.** This is not a novelty claim. The checkable parts are supported (REF_CHECK.md §4). The
repository comparison was not repeated here.

**C8, EEG Eq. (20) does not select $`\Delta`$.** This contradicts a published PRL. The paper's argument is
substantive:
- $`b \ge 1`$ holds globally;
- the truncated square;
- the data violate Eq. (20).

The verdict is unchanged, but the engagement is incomplete. Add Lai & Wen 2026, which builds on Eq. (20). Quote EEG's
own derivation step ("If $`\beta''+3\beta'`$ does not vanish simultaneously with $`\beta`$" then $`f\gt1`$), so that
readers see which inequality is at issue.

**C9, "genuinely new: two variables, periodic in θ". Survives as worded; low confidence.**
- **Scope.** The sentence is scoped to "neither template" (Xie–Tanveer; Baldomá–Capiński–Guardia–Seara).
- **Not read.** I read neither template's full text: metadata only (FULL-TEXT-UNREAD), so this verdict is a ceiling.
- **Nearest prior.** Lombardi's monograph (oscillatory component; metadata read) and Costin–Tanveer's PDE Borel
  summability (metadata read). PROOF.md itself names both, which is the right engagement.

**C10, "the first rigorous derivation of a critical-collapse echo period". Contradicted as worded.**
- **The contradicting work.** Reiterer & Trubowitz, CMP 368, 143 (2019) prove the existence of Choptuik's solution
  with a computer-assisted proof and enclose $`K`$ ($`\Delta = 2K`$) to 80 digits: "Here $`K \approx 1.72`$ and we
  prove" Eq. (2). That is a rigorous determination of the $`D = 4`$ echo period.
- **Proposed PROOF.md wording:** "together they would be the first rigorous derivation of the large-$`D`$ limit of the
  echo period from the $`D = \infty`$ equations (at $`D = 4`$, Reiterer and Trubowitz enclosed $`\Delta`$ rigorously
  by a computer-assisted existence proof)."
- **Also cite Reiterer–Trubowitz in the paper.** It is the computer-assisted-proof incumbent for this solution (see
  REFEREE_SIM.md).

## 3. Credit list, diffed against the bibliographies

These works were verified in this session and are missing from the paper's bibliography. Rozali & Way is cited, but
its overlap (C4, C5) is not acknowledged.

| Work | Why it belongs | Verified entry |
|---|---|---|
| Clark & Pimentel 2025 | the other large-$`D`$ critical-collapse analysis; EEG cite it for $`\gamma\to1/2`$ | C. R. Clark, G. L. Pimentel, *Black hole critical collapse in infinite dimensions: continuous self-similar solutions*, JHEP 11 (2025) 017, arXiv:2504.10590 ([Springer](https://link.springer.com/article/10.1007/JHEP11(2025)017)) |
| Lai & Wen 2026 | builds on EEG's NLO selection, which Sec. eeg disputes | S.-H. Lai, W.-Y. Wen, *Polyakov backreaction in large D discretely self-similar collapse*, arXiv:2610.04960 ([HTML](https://arxiv.org/html/2610.04960)); no journal-ref seen |
| Reiterer & Trubowitz 2019 | rigorous, computer-assisted results on Choptuik's solution, including $`\Delta`$ | Commun. Math. Phys. 368, 143–186 (2019), arXiv:1203.3766 ([Springer](https://link.springer.com/article/10.1007/s00220-019-03413-8)) |
| Taves & Kunstatter 2011 | earlier higher-$`D`$ numerics | T. Taves, G. Kunstatter, *Higher dimensional Choptuik scaling in Painlevé–Gullstrand coordinates*, arXiv:1105.0878 ([PDF](https://arxiv.org/pdf/1105.0878)). The journal is unverified: the [APS DOI page for PRD 84, 044034](https://link.aps.org/doi/10.1103/PhysRevD.84.044034) showed only "received 15 May 2011", not the title. |
| Price & Pullin 1996 | flat-space (free-wave) analytic approximation of the critical solution, a precursor of the leading-order picture | gr-qc/9601009 ([ar5iv](https://ar5iv.labs.arxiv.org/html/gr-qc/9601009)). Phys. Rev. D 54, 3792 (1996), from the arXiv DOI field and an institutional record (notes/PAPER_FIXES.md); now cited in both papers. |
| Kol 2006 | analytic $`\gamma(D)`$, $`\Delta(D)`$ for the merger analogue; cited by the 2025 review as analytic estimates in arbitrary $`D`$ | hep-th/0502033, DOI 10.1088/1126-6708/2006/10/017 ([abs](https://arxiv.org/abs/hep-th/0502033v3)) |
| Wu 2026 | DSS at large $`D`$ (quantum) | arXiv:2610.20705 ([abs](https://arxiv.org/abs/2610.20705)) |
| Baldomá, Capiński, Guardia & Seara 2023 | computer-assisted proof of a Stokes constant: the CAP precedent for the M = 1 certificate | J. Nonlinear Sci. 33 (2023), DOI 10.1007/s00332-022-09882-x ([recercat](https://www.recercat.cat/handle/2445/195109?show=full)) |
| Paris 2013 | large-parameter $`{}_2F_1`$ asymptotics with Stokes phenomena (Prop. 4 context) | J. Classical Anal. 2 (2013) 183–203 ([PDF](https://files.ele-math.com/articles/jca-02-15.pdf)) |

## 4. Synonym family (built during the search)

- large $`D`$ / large dimension / infinite dimensions / $`1/D`$ expansion;
- DSS / discretely self-similar / echoing / "critical spacetime crystal" (EEGJ's coinage) / scaling constant $`K`$
  ($`\Delta = 2K`$, Reiterer–Trubowitz);
- echo period / echoing period / self-similarity period;
- inner problem / oscillaton / oscillating soliton star (Rozali–Way);
- light cone / past light cone / self-similarity horizon (SSH);
- Stokes multiplier / Stokes constant / exponentially small / beyond all orders / non-perturbative / trans-series;
- similarity coordinates / self-similar coordinates / analytic modes;
- critical exponent / Choptuik exponent / scaling exponent / Lyapunov exponent.

"Self-similarity horizon" and "oscillaton" were found mid-review. They were not re-run as separate queries on every
engine (owed).

## 5. Search ledger

All searches ran on 2026-10-21 with the session's `WebSearch` tool (a US web index; some queries in "extended" mode).
The year range is unrestricted unless a year is named.

- **Large-D and priority queries.**
  - "large D limit Choptuik critical collapse echoing period 2026" (extended): found Lai–Wen, Wu and EEGJ v2.
  - "Rozali Way large D critical collapse cited by 2025 2026 discretely self-similar large dimension" (extended).
  - "\"large D\" critical collapse \"1/D expansion\" self-similar echoing period selection \"beyond all orders\" OR
    \"exponentially small\" OR \"Stokes\"" (extended): nothing beyond the known papers and this project's own
    repository.
  - "Emparan Suzuki Tanabe large D gravitational collapse scalar field critical phenomena".
  - "Ecker Ecker Grumiller large D discrete self-similar Physical Review Letters 2026 volume 136" (extended).
  - "pith.science 2601.14358 ... citations".
  - "Barak Kol Choptuik scaling merger transition analytic estimates gamma Delta arbitrary dimension".
  - "Emparan Herzog ... \"Critical collapse and Choptuik scaling\" ar5iv".
  - "Reiterer Trubowitz \"Choptuik's critical spacetime exists\" ...".
  - "\"3.445452402\" Choptuik echoing period Delta": found Price–Pullin and others.
- **Prop. 2 queries.**
  - "free wave equation similarity coordinates d dimensions mode solutions hypergeometric function analytic at light
    cone eigenvalues" (extended).
  - "\"self-similar\" scalar wave equation flat space \"hypergeometric\" \"past light cone\" discretely self-similar
    linear solutions not analytic" (extended).
  - "Costin Donninger Glogić mode stability self-similar wave maps higher dimensions hypergeometric connection
    coefficients".
  - "Donninger Schörkhuber free wave equation similarity coordinates d dimensions spectrum eigenvalues hypergeometric
    \"2F1\" self-similar blowup" (extended).
- **Prop. 4 queries.**
  - "Gauss hypergeometric function large c asymptotic expansion exponentially small Stokes phenomenon Paris"
    (extended).
  - "Olde Daalhuis hypergeometric function large parameter c resurgence late terms Borel".
- **Bibliographic searches** (about 45 queries, used for REF_CHECK.md, e.g. "Ecker Grumiller Jechtl critical spacetime
  crystal continuous dimension SciPost" and "Gundlach Hilditch Martín-García \"Critical phenomena in gravitational
  collapse\" Living Reviews in Relativity 2025"). Several of them also served as recall checks.
- **2026-11-08, issue #172 (the forms of arXiv:2611.07007's Eq. (16)).** `WebSearch` in standard mode unless marked,
  and the INSPIRE REST API through `WebFetch`. The queries are verbatim; what each surfaced is in the raw-data
  repository, `I172-mac/search_log.md` (commit 4e33763).
  - Critical collapse: "Choptuik critical solution Fourier coefficients odd harmonics decay echo profile analytic
    approximation"; "Hirschmann Eardley critical collapse complex scalar field discretely self-similar echoing time
    dependence Fourier analytic"; "critical collapse echoing \"Fourier coefficients\" \"decay exponentially\" complex
    tau singularity strip Choptuik spacetime analyticity harmonics geometric ratio"; "\"Poisson kernel\" critical
    collapse OR \"discretely self-similar\" OR echoing Fourier harmonics model".
  - Large D: "\"2611.07007\" critical collapse discretely self-similar large D"; "Emparan Suzuki Tanabe large D
    effective theory non-uniform black string profile analytic solution Fourier"; "Ecker Grumiller large D echo profile
    \"damping constant\" geometric Fourier coefficients model q delta 2026".
  - Echoes in other fields: "gravitational wave echoes exotic compact objects transfer function geometric series 1/(1 -
    R e^{2i omega}) template"; "Mark Zimmerman Du Chen \"A recipe for echoes from exotic compact objects\" transfer
    function geometric series Physical Review D 2017"; "Fabry-Perot field transmission geometric series multiple
    reflections 1/(1 - r1 r2 e^{-2i phi}) Airy distribution Ismail Optics Express 2016"; "Ismail Kores Geskus Pollnau
    \"Fabry-Pérot resonator: spectral line shapes, generic and related Airy distributions\" Optics Express 24 16366";
    "Fabry-Perot transmitted field amplitude \"t_1 t_2\" e^{-i\phi} \"1 - r_1 r_2 e^{-2i\phi}\" sum over round trips
    derivation arXiv"; "ccrma Julius Smith feedback comb filter amplitude response \"1 + a_M\" cos transfer function
    H(z) = b0/(1 + aM z^-M)".
  - Breathers, period doubling and log-periodicity: "sine-Gordon breather Fourier series in time odd harmonics geometric
    arctan sin"; "sine-Gordon breather Fourier expansion in time harmonics coefficients sech exact \"breather\"
    \"Fourier coefficients\" decay exponentially odd harmonics"; "Birnir McKean Weinstein \"rigidity of sine-Gordon
    breathers\" Fourier series time harmonics"; "Segur Kruskal nonexistence of small-amplitude breather solutions phi^4
    harmonics expansion exponentially small 1987 Physical Review Letters"; "Akhmediev breather analytic Fourier spectrum
    geometric decay harmonics closed form Dudley Genty Dias Kibler Akhmediev 2009 Optics Express"; "Akhmediev breather
    spectrum \"geometric\" sidebands amplitude ratio analytic \"triangular\" spectrum log scale modulation instability";
    "Feigenbaum onset spectrum of turbulence period doubling subharmonics amplitude ratio geometric 1979 Physics Letters
    A"; "Sornette discrete scale invariance log-periodic oscillations harmonics Fourier series amplitudes decay
    geometric \"log-periodic\" higher harmonics"; "\"discretely self-similar\" OR \"log-periodic\" profile \"geometric
    series\" harmonics closed form rational function Fourier \"single pole\" model".
  - Identities: "Gradshteyn Ryzhik 1.447 sum p^k sin kx = p sin x/(1 - 2p cos x + p^2) Fourier series"; "conjugate
    Poisson kernel Fourier series sum r^|n| sign(n) e^{in theta} = 2 r sin theta/(1 - 2 r cos theta + r^2) textbook";
    "encyclopediaofmath conjugate Poisson integral kernel \"conjugate Poisson kernel\" circle r sin / (1 - 2r cos +
    r^2)"; "Fourier series \"arctan\" \"2r sin\" \"1-r^2\" sum r^(2n+1) sin((2n+1)x)/(2n+1) identity"; "hippopede Booth
    lemniscate inverse of ellipse with respect to its centre MacTutor famous curves"; "Geer \"rational trigonometric
    approximations using Fourier series partial sums\" Journal of Scientific Computing 1995 Fourier-Padé"; "periodic
    function times trigonometric polynomial equals trigonometric polynomial Fourier coefficients linear recurrence
    constant coefficients rational generating function theorem"; "Koepf \"trigonometric holonomic\" Fourier coefficients
    recurrence linear differential equation arXiv"; "Sulem Sulem Frisch tracing complex singularities with spectral
    methods analyticity strip Fourier coefficients exponential decay rate 1983".
  - The forms themselves: "\"inverse of an ellipse\" Fourier series harmonics \"1/(e^{-i\theta}\" OR \"reciprocal of
    an ellipse\" complex parametrization geometric series odd harmonics"; "echo profile \"odd harmonics\" \"geometric\"
    amplitudes constant ratio \"phase\" self-similar periodic profile Fourier fit closed form \"1 + q^2 + 2q cos\""
    (extended); "Fourier series \"cos x/(1 + a cos 2x)\" OR \"cos θ/(1 + ε cos 2θ)\" odd harmonics geometric
    coefficients closed form"; "periodic waveform \"harmonics in geometric progression\" odd harmonics closed form sum
    \"1 - 2a cos 2\" signal \"only odd harmonics\" geometric".
  - INSPIRE: `q=arxiv 2611.07007` (recid 3223940, 0 citations); `q=arxiv 2601.14358` and `q=refersto:recid:3109820`
    (5 citing records); `q=arxiv gr-qc/0304070` and `q=refersto:recid:617314` (49); `q=arxiv 1807.10283` and
    `q=refersto:recid:1684230` (13). The API ignores the form `refersto:arxiv:<id>` and then returns an unfiltered list,
    so the recid form is the one to repeat.
- **2026-11-09, issue #198 (Fabry–Pérot and nonlinear cavities, Fratini et al.'s later work).** `WebSearch` in
  standard mode, and the INSPIRE REST API through `WebFetch`. The queries are verbatim in the raw-data repository,
  `I198-mac/search_log.md` (commit a3abcb5), with what each surfaced; shortened here.
  - The linear cavity: Ismail–Kores–Geskus–Pollnau's title with "arXiv"; "Fabry-Perot plane-parallel Gaussian beam
    transmission sum over round trips diffraction walk-off amplitude power law Airy function modified"; the
    Gires–Tournois all-pass formula; "\"generic Airy distribution\" Fabry-Perot \"Lorentzian finesse\" Pollnau"; the
    Wikipedia article's mode-profile and frequency-dependent-reflectivity sections; the Optica page of Opt. Express 24,
    16366 (allowed domains); Marques's angular Airy function; Madsen–Zhao's book with "z-transform ARMA poles zeros
    coupled cavities"; "multilayer thin film stack equal optical thickness layers transmission coefficient all-pole
    z-transform digital filter analogy" (nothing on thin films).
  - Nonlinear cavities: the titles of Ikeda (1979), Coen–Haelterman (1997), Hansson–Wabnitz (2015), Lugiato–Lefever
    (1987), Haelterman–Trillo–Wabnitz (1992), Ikeda–Daido–Akimoto (1980) and Felber–Marburger (1978); "Parra-Rivas
    Gomila Gelens Lugiato-Lefever numerical continuation AUTO bifurcation diagram ... period-doubling arXiv";
    Chembo–Yu's modal expansion; "Gomila Scroggie Firth ... saddle-loop homoclinic period diverges logarithmically
    unstable cavity soliton separatrix"; "optical bistability middle branch unstable saddle switching time
    logarithmic divergence critical slowing down" (no logarithmic law); "Hill's method Floquet-Fourier-Hill
    Lugiato-Lefever stability" (no LLE paper with Hill's method); Deconinck–Kutz (2006) with "spurious eigenvalues
    convergence"; "Kerr microresonator breather soliton periodic orbit numerical continuation Floquet multipliers Newton
    shooting harmonic balance"; "Ikeda map period-two state fixed point of the twice-iterated map ... Floquet
    stability fiber ring cavity".
  - Fratini et al.: "Filippo Fratini later work nonlinear Fabry-Perot quantum mirror rectification ... Google Scholar"
    (no Google Scholar page surfaced).
  - INSPIRE: `q=arxiv 1410.5972` (recid 3025790, 57 citations); `q=refersto:recid:3025790` (57 records);
    `q=a F.Fratini.1 or a Fratini, Filippo` (17); `q=refersto:recid:3027964` (21).

## 6. Read ledger

- **Full text, through WebFetch extraction:**
  - Rozali–Way (Secs. 1–2.3 only);
  - EEG (HTML v1);
  - EEGJ (up to Sec. 6; no reference list), with spot checks on v2;
  - Lai–Wen;
  - Clark–Pimentel;
  - the 2025 review (HTML);
  - Kol (v3 PDF);
  - Donninger 0809.5177;
  - Costin–Donninger–Glogić 1604.00303;
  - Donninger–Aichelburg;
  - Paris 2013;
  - Taves–Kunstatter;
  - Martín-García–Gundlach 2003 (for Eq. (58));
  - Xie 1610.00076 (theorems);
  - Bland's thesis (abstract and front matter, OCR).
- **Partial:**
  - Emparan–Herzog (to Sec. V);
  - Donninger 0805.0520;
  - Biernat–Donninger–Schörkhuber;
  - Price–Pullin (truncated);
  - Reiterer–Trubowitz (abstract and introduction).
- **Abstract or metadata only:** Wu 2026; Temme; Bland–Kunstatter; Sorkin–Oren; Garfinkle et al.; all PROOF.md
  precedents except Xie 1610.00076 and Reiterer–Trubowitz.
- **FULL-TEXT-UNREAD**, grades are ceilings:
  - Emparan–Herzog Sec. X.C (critical collapse);
  - Rozali–Way beyond Sec. 2.3;
  - the reference lists of EEG and EEGJ, so EEG's [16, 17, 25, 29] and EEGJ's [12] are unidentified;
  - Olde Daalhuis's large-parameter hypergeometric papers;
  - Lombardi; Xie–Tanveer; Tanveer–Xie; Baldomá et al. (both); Costin's works; Dingle.
- **2026-11-08, issue #172** (through `WebFetch`; quotes and URLs in the raw-data repository,
  `I172-mac/read_log.md`, commit 4e33763):
  - Full extraction: Price–Pullin (ar5iv); Lai–Wen; Emparan–Suzuki–Tanabe 1506.06772; Rozali–Way Sec. 2.3; EEG
    2601.14358 (HTML v1); EEG 2611.07007v1 (HTML, Eq. (16) and (S57)–(S58)); Fratini et al. 1410.5972 (with the
    supplement); Lu–Cuevas-Maraver–Kevrekidis 1406.3082; Ryazanov 1711.02717; DLMF §1.15, §18.12 and §4.24; the two
    CCRMA feedback-comb pages of Smith's *Physical Audio Signal Processing*; Coffman's lemniscates page; MathWorld,
    *Hippopede*; Wikipedia, *Fabry–Pérot interferometer* (a secondary source).
  - Partial: Martín-García–Gundlach 2003 (characters 0–100000 of 143647); Gundlach 1997 (0–100000 of 130156);
    Reiterer–Trubowitz (0–100000 of 118188); EEGJ v2 (0–100000 of 214015; App. C.3 and Fig. 22 unread); the 2025
    review (0–200000 of 317589); Wu 2610.20705 (0–200000 of 209108); Testa–Pani (0–100000 of 121374); Mark et al.
    (0–100000 of 137091); Fodor 1911.03340 (0–100000 and 190000–290000 of 291274); Benoit–Brisebarre–Salvy
    2604.26944 (to Sec. 7.2).
  - Abstract or record only: Hirschmann–Eardley gr-qc/9412066 (ar5iv conversion failed); Wu 2606.17135; Dudley et al.
    0910.1930; Geer's ICASE report 93-68 (NTRS); arXiv journal-ref pages of Testa–Pani, Mark et al., Fratini et al.,
    Emparan–Suzuki–Tanabe and Lu et al.
  - Failed (FULL-TEXT-UNREAD): PDFs that came back as binary (Reiterer–Trubowitz, Dudley et al., a course copy of
    Gradshteyn–Ryzhik's series tables); Optica pages (loading placeholder); HAL and PMC (bot checks); MacTutor
    *Hippopede* (404); the Springer page of Koepf–Nana Chiadjeu (login); arXiv:2312.01320 (no HTML).
- **2026-11-09, issue #198** (through `WebFetch`; PDFs that came back as binary were extracted with pypdf from the copy
  the tool saved; quotes and URLs in the raw-data repository, `I198-mac/read_log.md`, commit a3abcb5):
  - Full extraction: Pollnau–Eichhorn 2020 (KITopen PDF; Secs. 3.5–4.3 closely, the reference list); Sánchez-Soto–
    Monzón–Leuchs 1610.06163 (ar5iv); Marques–Guggenheim–Munro 2021 (UCL PDF, 7 pages); Hansson–Wabnitz 1503.03274
    (ar5iv, Secs. II–V); Conforti et al. 1405.6041 (ar5iv); Parra-Rivas et al. 2010.03375 (ar5iv, 0–193242 in two
    fetches); Gomila–Matías–Colet nlin/0411047 (ar5iv); Maitland et al. 2004.14095 (ar5iv); Fratini et al. 1410.5972
    (ar5iv, with the supplement); Fratini–Ghobadi 1510.01968, Mascarenhas et al. 1510.01472 and Almeida–Werlang–Valente
    1902.10181 (ar5iv).
  - Partial: Wikipedia, *Fabry–Pérot interferometer*, Secs. 3.5, 3.8 and 3.9 (a secondary source).
  - Abstract or list only: Chembo–Menyuk 1210.8210; Roy 1611.05050; Dai et al. 1510.04494; Manasi–Roy 1712.04474;
    Zheng et al. 1605.01135; Fratini's publication page; the arXiv pages of Hansson–Wabnitz, Conforti et al.,
    Parra-Rivas et al. and Gomila et al. (journal fields); the INSPIRE records and citation lists.
  - Records only: Ismail et al. 2016 (search listing of the Optica page); Ikeda 1979 (a library record); Ikeda–Daido–
    Akimoto 1980; Lugiato–Lefever 1987; Coen–Haelterman 1997; Haelterman–Trillo–Wabnitz 1992; Felber–Marburger 1978;
    Chembo–Yu 2010; Madsen–Zhao 1999.
  - Failed (FULL-TEXT-UNREAD): the Optica pages of Ismail et al. (loading placeholder) and its scispace copy (empty);
    the UCL "pp" mirror (403); scilit (403); a Czech library record (bot check); a Kyoto repository PDF (empty).

## 7. Citation chains

- **Backward.**
  - EEG cites [23] Clark–Pimentel for the CSS limit. Clark–Pimentel cite [28] for $`\gamma \to 1/2`$ (not identified)
    and Soda–Hirata [34] for the CSS-preserving exponent.
  - EEGJ cite [7] (EEG), [12] (the $`\gamma`$ conjecture; per the pith.science aggregator their list includes
    Rozali–Way, unverified), [17] Bland et al., [18] Bland's thesis, [33] Sorkin–Oren.
  - The 2025 review leads to Kol, Taves–Kunstatter and Porto Veronese–Gundlach ($`D = 9`$, not read).
- **Forward.**
  - EEG is cited by Lai–Wen (arXiv:2610.04960).
  - EEGJ is cited by Wu (arXiv:2610.20705), per pith.science.
  - No INSPIRE cited-by listing could be read (INSPIRE was blocked to direct fetch and search did not surface its
    citation pages), so the forward chains rest on keyword searches and one aggregator.

## 8. Gap statement, rounds, dryness

**Not searched, or searched with a dead channel:**
- INSPIRE, arXiv listing API, CrossRef, Semantic Scholar and OpenAlex were blocked to direct access (proxy 403).
  INSPIRE "refersto" cited-by searches, the most important forward-chain tool for this field, never ran.
- Google Scholar was not queried (no lawful automated route here).
- Per-author recency sweeps through personal or group pages were not done for: Ecker (both), Grumiller, Jechtl,
  Rozali, Way, Clark, Pimentel, Emparan, Herzog, Lai, Wen, Wu, Donninger, Glogić, Schörkhuber. That is the
  highest-priority owed round. EEG's "[25] work in progress" on moderate-$`D`$ numerics is the specific risk for C6.
- The mathematical self-similar-blowup literature (Donninger–Schörkhuber, Glogić–Schörkhuber, Bizoń et al.) was
  searched for Prop. 2 by keyword only.
- The exponential-asymptotics literature on large-parameter hypergeometric functions (Olde Daalhuis; Paris II) was
  searched for Prop. 4 by keyword only.
- No dissertations beyond Bland's, no non-English sources, no journal tables of contents.

**Rounds:**
1. The named papers (Rozali–Way, EEG, EEGJ, the review, Emparan–Herzog). Added the P4 overlap with Rozali–Way and
   EEG's NLO details.
2. Recency and forward chains for 2025–2026. Added Lai–Wen (P3), Clark–Pimentel (P3), Wu (P2) and EEGJ v2.
3. Synonyms and methods (similarity coordinates and hypergeometric functions; large-$`c`$ $`{}_2F_1`$; Stokes or
   trans-series at large $`D`$). Added Donninger and Paris (P2–P3) and the Emparan–Herzog trans-series section (P2).
4. Mined from the review, and priority checks. Added Kol, Taves–Kunstatter and Price–Pullin (P2), and
   Reiterer–Trubowitz (P3, decisive for C10).

**Not iterated to dry.** Round 4 still added a P3 work, and the INSPIRE and per-author channels are dead or owed. A
fifth round should run the per-author sweeps and INSPIRE cited-by on Rozali–Way, EEG, EEGJ and Clark–Pimentel (with
INSPIRE access), and re-run "oscillaton", "self-similarity horizon" and "critical spacetime crystal" as queries.

## Addendum (2026-10-21): earlier formulas for Δ(D) and γ(D)

Question: was the rational formula in 1/(D−2) for Δ(D) and γ(D) (README "The formula") published before? Abstracts and
full texts checked:
- **Bland et al.**, CQG 22, 5355 (2005), gr-qc/0507088 (abstract): an exponential fit for γ only,
  "γ ∼ 0.466(1 − e^{−0.408 D})" over 3.5 ≤ D ≤ 14, with a large-D value of 0.466, not 1/2. No formula for Δ.
- **Sorkin & Oren**, PRD 71, 124005 (2005), hep-th/0502034 (abstract): no formula; the trends "seem to indicate that
  gamma reaches a maximum and Delta a minimum value around 11 <= D <= 13". The double-double data show neither: Δ
  decreases and γ increases monotonically to D = 61.
- **Bland & Kunstatter**, PRD 75, 101501 (2007), hep-th/0702226 (abstract): γ at D = 5 only. **Bland**'s thesis (2007)
  conjectured Δ∞ = 5/2 (as already recorded).
- **Taves & Kunstatter**, arXiv:1105.0878 (full text): values for D = 4–8 only, no formula.
- **Ecker, Ecker, Grumiller & Jechtl**, SciPost Phys. 20, 176 (2026), arXiv:2602.10185 (abstract): numerical values for
  3.05 ≤ D ≤ 5.5, analytic expansions in 1/D and in D − 3 suggesting Δ, γ → 0 as D → 3⁺, and the conjectures Δ → 0,
  γ → 1/2 as D → ∞; no fitting formula.
- **Ecker, Ecker & Grumiller**, "Critical collapse remains discretely self-similar at large D", arXiv:2611.07007v1
  (4 Nov 2026; verified, arXiv listing; HTML read 2026-11-06): computed Δ for D ≤ 200 (supplemental Table 1, six
  decimals, no stated uncertainty), the fit Δ ≈ 1.6439 + 10.37/D, Δ(∞) = 1.643911 from a fit in 1/(D − 1), and an
  explicit large-D model profile; no analytic value of Δ∞. It argues that Δ stays finite as D → ∞. Its values are
  compared with ours in hp/data/delta_published_compare.csv (hp/compare_published.py).
- **Kol**, hep-th/0502033 (full text, already recorded above): 2π/Δ = √((10 − D)/(D − 2)) and a formula for γ, but for
  the black-string merger, not for scalar-field collapse.

Verdict: no earlier formula for Δ(D) was found, and the only earlier fit for γ(D) (Bland et al.'s exponential) has a
different form and limit. The rational formula is an empirical fit, a ratio of polynomials with 11 fitted coefficients
for Δ and 9 for γ. Rational (Padé-type) fitting is a standard method, so what is new is the fitted function and its
validation (precision across D ≥ 4, held-out agreement with the double-double points to D = 61, and the correct limits),
not a law derived from the equations.

## Addendum (2026-10-21): prior art for the techniques in PARALLEL.md

Question: is any technique in PARALLEL.md new? PARALLEL.md itself claims no novelty. Sources read (abstracts or the
relevant section), not recalled:

| PARALLEL.md item | Closest prior work read | What it contains |
|---|---|---|
| §1 affine maps (A, b) compose associatively, so a recurrence is a prefix scan | Blelloch, *Prefix sums and their applications* (1990), via Heinsen, arXiv:2311.06281, §2–3; ScanWeaver, arXiv:2606.00601, eq. 7 | first-order linear recurrences as scans; the operator (A₂A₁, A₂b₁ + b₂); the closed form x_t = (Πa)(x₀ + Σ b_i/Πa) |
| §1 `crc32_combine` | zlib man page crc32_combine(3) | the CRC of a concatenation from crc1, crc2 and len2 only |
| §2 Newton step on a block-bidiagonal system is an affine recurrence over segments, solved by a scan; long Jacobian products unstable | Lim et al., DEER, ICLR 2024 (arXiv:2309.12252), as described in Gonzalez et al., arXiv:2407.19115, §3.1, §3.3 | the same structure for nonlinear RNNs: Newton, block-bidiagonal Jacobian, parallel associative scan, instability when the per-step Jacobians have eigenvalues above 1 |
| §2 condensing the multiple-shooting system | *Block structured quadratic programming for the direct multiple shooting method for optimal control*, optimization-online 2009/09/2395 (authors not checked) | the classical condensing of block-sparse multiple-shooting systems to small dense ones |
| §2 stable structured alternatives | Varah, *Alternate row and column elimination* (UBC TR 74-03; journal version doi:10.1137/0713008, not opened); Wright, SIAM J. Sci. Stat. Comput. 13 (1992), already cited | pivoted elimination for bordered block systems of BVPs with no fill-in, as stable as Gaussian elimination; stable parallel BVP elimination |
| §3 Jacobian columns by forward-mode tangents; vector mode sharing the primal | standard AD (Griewank–Walther; survey arXiv:1502.05767) | forward and vector forward mode |
| §4 fit c₀ as the first row of a precomputed pseudo-inverse | Savitzky–Golay filter (1964), Wikipedia summary | least-squares coefficients from a fixed design matrix, computed once and applied as a dot product |
| §4 Kronecker substitution | Harvey, arXiv:0712.4046 | credits Kronecker (1882) and Schönhage |
| §4 truncated products | Giorgi–Grenet–Roche, arXiv:1902.02967, Def. 3.1 | short product; cites Mulders (2000) and Hanrot–Zimmermann (2004) |
| §4 relaxed multiplication | van der Hoeven (texmacs.org/joris/relax), already cited | online power-series products |
| §6 parallel continuation | arXiv:1312.3491 | parallel step-length adaptation in pseudo-arclength continuation (several predictor–corrector chains at once); not Parareal in the parameter |

Verdict: every technique in PARALLEL.md has a named precedent. Not found in this search, and not a technique in any
case: the CRC-32 framing of multiple shooting and of the series (an exposition), the instances specific to this code
(the summation-factor closed form with T_j = (α)_j(β)_j/j!, the 193-unknown condensed system, the 1.9% nonzero LU work
after pivoting), the measured speedups, and the late-order fit-basis finding of §4 (every neighbour family |ν′| ≤ k is
needed). That last one is a numerics result about this computation; the general principle, that late orders carry every
singularity's contribution, is standard in resurgence (Dingle; Berry–Howls).

PARALLEL.md was removed afterwards (2026-10-21). Its measurements are in the README (Performance) and in the docstrings
of `solver/inner_amp/amp_flint_par.py` and `hp/src/linalg.rs`; it is in the git history.

- 2026-11-04: **Ecker, Ecker, Grumiller & Jechtl**, arXiv:2602.10185v1, the arXiv HTML, Sections 6.2 ("Small-(D−3)
  expansion": 6.2.1 Fuchsian system, 6.2.2 scaling regime) and 8 read (via fetch; Appendix D, their table, lies past
  the fetched text). Their reading of the limit: a fit to their Fig. 16 data, Δ ≈ 3.88 (D − 3)^0.15 (Eq. 110), with α
  ≈ 0.15 taken as a lower bound; a scaling argument (x ~ ε^δ, e^ω ~ 1/ε, Δ ~ ε^(δ−β), Eqs. 111–115) gives α ≈ 0.3; the
  leading Fuchsian analysis gives no power law and says the limit D → 3 does not exist as stated; γ → 0 faster than Δ
  (Section 5.3), so Δ/(2γ) diverges. Values quoted in the text: D = 3.05: Δ ≈ 2.48, γ ≈ 0.16. They cite Bland's thesis
  (2007) for D = 3.02: Δ = 0.1379 ± 0.0042, "suggesting a faster falloff" (the thesis itself not read for it).
  Recorded in `hp/data/published_lowD_limits.json` for `hp/plot_lowD_limits.py`. Their numerical values are already
  `hp/data/published_delta.csv` (their repository's DeltaList.dat, 3.046 ≤ D ≤ 5.5).

## Addendum (2026-11-08, issue #172): the rewritten forms of arXiv:2611.07007's Eq. (16)

**Question.** Do the rewritten forms of Eq. (16) of Ecker, Ecker and Grumiller (arXiv:2611.07007v1, here EEG26b)
appear elsewhere, as the same function or as the same idea applied to an echo, a self-similar profile or another
periodic profile? The forms are defined in notes/EEG_PROFILE.md §2, whose labels are given in brackets:
- **A**, the reciprocal of an ellipse, 1/(e^{−iθ} + w e^{iθ}) with w = q e^{iδ} (§2 (H));
- **B**, the real form (cos θ + q cos(θ+δ))/(1 + q² + 2q cos(2θ+δ)) (§2 (R));
- **C**, the odd-harmonic Fourier series with the single complex ratio −w (§2 (F));
- **D**, the linear equation (d²/dθ² + 1)[(1 + q² + 2q cos(2θ+δ)) β̃] = 0 (§2 (ODE));
- **E**, the arctan closed form of the antiderivative (§2 (INT)).

**How.**
- **Searches.** 35 web queries and 7 INSPIRE queries, all on 2026-11-08, listed verbatim in §5.
- **Reads.** 32 sources opened, some in part and some as abstracts, and about 15 fetches that failed, listed in §6.
  Quotes and URLs are in the raw-data repository, folder `I172-mac/`, commit 4e33763.
- **Inferences.** Statements of content come from pages opened in this session. Inferences are marked *inferred*.
- **Identity check.** Four identities that tie the forms to the sources are *inferred* and *checked numerically* by
  `I172-mac/identity_check.py` (30 digits, 41 points, exit 0). Its sign mutation makes all four fail (exit 1).
- **Scope.** No numbers from the D = ∞ tuned profile are used.
- **Verdict classes**, fixed before the search: found, with the source and its equation; found as a general identity
  only; not found in the sources searched.

### Closest works

Grades as in §1, against the statement "this form appears elsewhere".

| Work | Bears on | Grade | What it contains | What it does not contain |
|---|---|---|---|---|
| **EEG26b**, arXiv:2611.07007v1 (INSPIRE recid 3223940), HTML v1 | B, C | **P5** (the source) | Eq. (16), "β̃(θ) = A∗ Re[ e^{iθ} / (1 + q e^{i(2θ+δ)}) ]"; (S58), the series "A∗ Σ_{n≥0} (−q)^n cos[(2n+1)θ + nδ]" and the real form; after Eq. (16), amplitudes "exponentially suppressed as e^{-αn}, where α = −ln q ≈ 2.90". | The reciprocal form A, an equation like D, an antiderivative like E (arctan appears only in the phase definition (S57)), a citation for the geometric model, the Poisson kernel. This agrees with #165. |
| **J. M. Martín-García, C. Gundlach**, PRD 68, 024011 (2003), gr-qc/0304070 (verified in REF_CHECK.md §1), ar5iv, characters 0–100000 of 143647 | B, C | **P3**: the same idea applied to the echo, at D = 4 | Sec. IV.1.1: metric fields have only even k, matter fields only odd k. Sec. IV.2: the Fourier coefficients in τ decay exponentially, "the decay exponent a ranges from 0.28 for f_c at the center to 2.04 for U_p at the past lightcone"; the phases of the high modes "tend to a linear dependence on k at a given point x". So the high harmonics have a nearly constant complex ratio (*inferred*). Then "the high frequency behaviour of our fields Z is similar to that of the function" Σ_k e^{−a\|k\|} e^{4πikτ/Δ} = sinh a/(cosh a − cos(4πτ/Δ)), Eq. (59), the Poisson kernel with r = e^{−a}. | A fit. The model has a real ratio, sums all k, and has no first-harmonic numerator and no phase parameter. The forms A, D and E are absent. "We have not found an explanation for this behavior, but it seems to be of dynamical origin." |
| **M. Rozali, B. Way**, JHEP 11 (2018) 106, arXiv:1807.10283 (verified), ar5iv Sec. 2.3 | E at δ = 0; B at δ = 0 | **P3** for E, P2 for B | Eq. (10), φ₀ = tanh⁻¹(ε e^{−ω²r²/2} sin(ωτ − τ₀)), and δ₀ = ε²e^{−ω²r²}cos²(ωτ−τ₀)/(1 − ε²e^{−ω²r²}sin²(ωτ−τ₀)). At fixed r, φ₀ is form E at δ = 0 up to a constant (already in EEG_PROFILE.md §2 (INT)). *Inferred* and checked: ∂_τφ₀ is ω·2√q times form B at δ = 0 (with A* = 1), where ε e^{−ω²r²/2} = 2√q/(1+q). | ∂_τφ₀ written out, a harmonic expansion ("Since this oscillation is a single frequency, this solution describes the large D version of an oscillaton"), δ ≠ 0, self-similarity. |
| **Fabry–Pérot transmitted field**: Wikipedia, *Fabry–Pérot interferometer* (secondary; it cites O. Svelto, *Principles of Lasers*, 5th ed. (2010), ch. 4.5.1, *(unverified)*); F. Fratini et al., PRL 113, 243601 (2014), arXiv:1410.5972, supplement Eq. (S.2) | A, C | **P3**: the same function, in another variable | The round-trip sum E_trans,1 = −E_inc t1 t2 e^(−iφ), E_trans,m+1 = E_trans,m r1 r2 e^(−i2φ), so E_trans/E_inc = −t1 t2 e^(−iφ)/(1 − r1 r2 e^(−i2φ)) (Wikipedia). Fratini et al. write the same sum with a complex round-trip ratio √(R1R2) e^{2ikL}e^{i(θ1+θ2)}. *Inferred* and checked: with θ = −φ and w = −r1r2, this is Eq. (16)'s complex function up to a constant, and the round-trip sum is series C in φ. | A time profile. φ is the single-pass optical phase, a frequency variable. The real part is not the observable, and the ellipse reading, D and E are absent. |
| **Sine-Gordon breather**, in N. Lu, J. Cuevas-Maraver, P. G. Kevrekidis, J. Phys. A 47, 455101 (2014), arXiv:1406.3082, Eq. (2) | E at δ = π; B at δ = π | P2 | b_o(x,t) = 4 arctan[σ sin(at)/(a cosh σx)], σ = √(1−a²). *Inferred* and checked: at x = 0 this is form E at δ = π up to a constant, with σ/a = 2√q/(1−q), and its time derivative is 4a·2√q times form B at δ = π (with A* = 1). | A harmonic series in closed form (their Appendix Eq. (17) is a numerical Fourier expansion with k_m = 15), δ ∉ {0, π}, an echo. |
| **Gravitational-wave echo templates**: Z. Mark, A. Zimmerman, S. M. Du, Y. Chen, PRD 96, 084002 (2017), arXiv:1706.06155, Eqs. (20), (29); A. Testa, P. Pani, PRD 98, 044018 (2018), arXiv:1806.04253, Eqs. (10), (37), (38) | A, C | P2: the summation idea applied to echoes, a different function | The transfer function 𝒦 = 𝒯_BH ℛ e^{−2iωx₀}/(1 − ℛ_BH ℛ e^{−2iωx₀}) as a geometric sum over echoes, the n-th carrying (ℛ_BH ℛ)^{n−1}; its poles are the object's quasinormal modes. | Periodicity in ω (the reflectivities depend on ω), the odd-only structure, a real part, an arctan. |
| **Feedback comb filter**: J. O. Smith III, *Physical Audio Signal Processing* (W3K, 2010), the CCRMA pages "Feedback Comb Filters" and "Feedback Comb Filter Amplitude Response" | C | P2 | "a computational physical model of a series of echoes, exponentially decaying and uniformly spaced in time"; H(z) = 1/(1 − g z^{−M}); the variant y(n) = b_M x(n−M) − a_M y(n−M). | Eq. (16)'s complex function. *Inferred*: it is z^{−1}/(1 + w z^{−2}) at z = e^{−iθ}, a loop with a one-step forward delay and a two-step feedback, which the pages do not write. |
| **Odd harmonics of the Choptuik solution**: C. Gundlach, PRD 55, 695 (1997), Sec. II.3; Martín-García–Gundlach 2003, Sec. IV.1.1; M. Reiterer, E. Trubowitz, CMP 368, 143 (2019), Secs. 1.1, 3; EEGJ, SciPost Phys. 20, 176 (2026), Sec. 4.1; C. Ecker, F. Ecker, D. Grumiller, PRL 136, 191401 (2026), HTML v1 (all verified in REF_CHECK.md) | C (its odd-harmonic part) | P2 | Gundlach: "X± only of odd frequency terms", a "reflection-type symmetry". Reiterer–Trubowitz: the antiperiodic class has "v_mn = 0 for all even m", with coefficients "𝒪((κ1)^−\|m\|(κ2)^−\|n\|)" and κ1 = 65/64, a rigorous bound. Ecker et al. (PRL): "the frequencies of higher modes are odd integer multiples of the fundamental frequency", from finite-D numerics. | A constant complex ratio from the first harmonic on, or a closed form. |
| **The identities**: DLMF 1.15.12, 18.12.7–18.12.10 and 4.24.3; V. Ryazanov, arXiv:1711.02717, Eqs. (3.1), (4.1); A. Coffman, *Lemniscates* (web page); J. F. Geer, ICASE Report 93-68 (1993) | A, B, C, D, E as general identities | P2 | 1.15.12: P(r,θ) = (1 − r²)/(1 − 2r cos θ + r²) = Σ r^{\|n\|} e^{inθ}. 18.12.7–18.12.10: the Chebyshev generating functions, the geometric Fourier series at x = cos θ. 4.24.3: arctan z = z − z³/3 + z⁵/5 − ⋯. Ryazanov: P_r and the conjugate kernel Q_r(Θ) = 2r sin Θ/(1 − 2r cos Θ + r²). Coffman: the elliptic lemniscate of Booth "is the image of an ellipse under inversion with respect to its center". Geer: an approximation that "is the quotient of a trigonometric polynomial of degree N" and one of degree M, matched to Fourier coefficients. *Inferred* and checked: B = ½[(1 + P_q(φ)) cos θ − Q_q(φ) sin θ] with φ = 2θ + δ + π. | Any application to an echo or a self-similar profile. |

Negative reads, which contain none of the forms:
- Price–Pullin, gr-qc/9601009: one frequency, "If, in fact, we take F~ to be a sinusoidal oscillation of amplitude
  unity".
- Lai–Wen, arXiv:2610.04960, Eq. (4), and EEG's PRL, Eq. (21): the two-mode model cos 2πτ + sin(6πτ)/15.9476.
- Wu, arXiv:2610.20705: "Any smooth periodic β(T) gives a leading-order DSS solution".
- Emparan–Suzuki–Tanabe, PRL 115, 091102 (2015), arXiv:1506.06772: the only analytic profile is a Gaussian blob, Eq.
  (22).
- EEGJ, the 2025 review, and Fodor's oscillon review (arXiv:1911.03340), in the parts read.
- Abstract only: Hirschmann–Eardley, PRD 51, 4198 (1995) (continuously self-similar); Wu, PRD 114, 044092 (2026).

### Verdicts per form

**A, the reciprocal of an ellipse. Found as the same complex function in optics. The ellipse reading is found as a
general identity only.**
- The complex function e^{iθ}/(1 + w e^{2iθ}) = 1/(e^{−iθ} + w e^{iθ}) is the Fabry–Pérot transmitted field, summed
  over round trips, with θ = −φ and w = −r1r2 (Wikipedia, citing Svelto *(unverified)*; Fratini et al., supplement Eq.
  (S.2)). Its variable there is the optical phase, not time.
- EEG26b write the ratio, not the reciprocal.
- The inverse of a centred ellipse is Booth's elliptic lemniscate (Coffman). *Inferred*: since 1/h = h̄/|h|², the curve
  of g = 1/h is that lemniscate reflected in the real axis.
- Not found in the sources searched: the reciprocal of an ellipse used for an echo or a self-similar profile.

**B, the real form. Found in EEG26b (S58). Elsewhere it is found only at δ = 0 and δ = π, implicitly, and as a
general identity.**
- At δ = 0 it is, up to a constant, the time derivative of Rozali and Way's oscillaton at a fixed radius. At δ = π it
  is, up to a constant, the time derivative of the sine-Gordon breather at its centre. Both are *inferred*, since the
  sources write the field, not its derivative.
- Its denominator is the Poisson kernel's (DLMF 1.15.12; EEG_PROFILE.md §2 (P)), and B is a combination of P_q and Q_q
  (*inferred*, table above).
- The closest work in critical collapse is Martín-García and Gundlach's Eq. (59), a Poisson-kernel sum offered as a
  model of the D = 4 echo's high harmonics. It is not fitted, and its ratio is real.
- Not found in the sources searched: B with a general δ outside EEG26b.

**C, the odd-harmonic series with ratio −w. Found in EEG26b (S58). Its parts are older.**
- Only odd harmonics in the scalar field of the Choptuik solution: Gundlach (1997), Sec. II.3, and the later works in
  the table.
- Exponential decay of the echo's harmonics with phases linear in k, so a nearly constant complex ratio at high k:
  Martín-García and Gundlach (2003), Sec. IV.2, observed at D = 4 and not explained there. EEG26b cite no source for
  their geometric law.
- The same series appears as a sum of echoes in the Fabry–Pérot round trips, in the optical phase. It is also the
  frequency response of a delayed feedback comb (*inferred*).
- As an identity, it is the geometric series (DLMF 1.15.12, 18.12.7–18.12.10).
- Not found in the sources searched: a profile, outside EEG26b, whose odd harmonics have one complex ratio from the
  first harmonic on.

**D, the linear equation. Not found in the sources searched. Found as a general method only.**
- The equation says that the profile times a positive trigonometric polynomial of degree 2 is a first harmonic.
- Two general methods lie behind it. One is the Fourier–Padé form, a quotient of trigonometric polynomials matched to
  the Fourier coefficients (Geer 1993). The other turns a linear ODE with trigonometric coefficients into a recurrence
  for the Fourier coefficients (Koepf and Nana Chiadjeu, Math. Comput. Sci. 9 (2015) 365–389 *(unverified)*, not read).
- No source searched writes a differential equation of this kind for an echo or another periodic profile.

**E, the arctan integral. Found as a general identity, and at δ = 0 and δ = π. Not found for general δ.**
- The identity is the termwise integral of series C with the arctan series (DLMF 4.24.3).
- At δ = 0 it is Rozali and Way's oscillaton at fixed r, Eq. (10), with ε e^{−ω²r²/2} = 2√q/(1+q).
- At δ = π it is the sine-Gordon breather at its centre, 4 arctan(k sin at), with k = 2√q/(1−q) (*inferred*).
- Both are exact single-mode solutions of other problems, not echo profiles.
- It is not in EEG26b (#165).

### Citation chains (INSPIRE, 2026-11-08)

- **Forward from EEG26b** (recid 3223940): no citations yet.
- **Forward from EEG's PRL** (recid 3109820): 5 citing records, Lai–Wen, EEG26b, EEGJ and Wu's two papers. The four
  besides EEG26b were read in whole, in part or in abstract, and contain no form.
- **Forward from Martín-García–Gundlach 2003** (recid 617314): 49 titles. None but EEG's PRL suggests an analytic
  model of the time dependence. This rests on titles only.
- **Forward from Rozali–Way** (recid 1684230): 13 titles. Of these, Fodor's review was read in part.
- **Backward from EEG26b**: no source is cited for the geometric model.

### Gap statement

- **Textbooks not opened.** Gradshteyn–Ryzhik 1.447–1.448 (the entry numbers come from search summaries only,
  *(unverified)*); Born–Wolf, Hecht, Svelto and Siegman for the Fabry–Pérot sum; Zygmund or Katznelson for the
  conjugate kernel.
- **PDF-only sources not read.** The fetch tool returned PDFs as binary, and no PDF text extractor was available. So
  the analytic spectra of Akhmediev breathers (Dudley et al. 2009; Akhmediev et al. 2011), the lecture notes, and
  Feigenbaum's period-doubling spectrum were not read. The Akhmediev-breather spectrum is a likely further instance of
  a geometric harmonic law from a rational periodic profile *(unverified)*.
- **Partly read.** Martín-García–Gundlach 2003 beyond character 100000; the body of Hirschmann–Eardley; the account of
  Koepf–Nana Chiadjeu by Benoit, Brisebarre and Salvy (arXiv:2604.26944); Sulem–Sulem–Frisch (the pole-to-decay method
  in fluids, background only).
- **Channels.** Forward chains were run on INSPIRE for four records only. There were no per-author sweeps, no Google
  Scholar queries and no non-English sources.
- **Rounds.** Round 1 covered the named areas. Round 2 covered synonyms, identities and forward chains, and added the
  Fabry–Pérot, comb-filter and breather instances, so the search is not dry.

### Discrepancies and proposed text (for the coordinator)

- **Price–Pullin.** notes/EEG_PROFILE.md §5 marks its content *(unverified)*. It has now been read (ar5iv, full
  extraction), and it keeps a single sinusoid. Proposed: "The full text (ar5iv) keeps one frequency (their Eq. (43) and
  Sec. IV) and has no profile of this form."
- **Credit for the geometric law.** Martín-García and Gundlach found at D = 4 that the echo's Fourier coefficients decay
  exponentially in τ with phases linear in k, and compared them with a Poisson-kernel sum. EEG26b do not cite this.
  Where the repository states its leading structure (README, "Leading structure and a prediction"; the math paper,
  "The shape of the echo"), proposed: "At D = 4, Martín-García and Gundlach found the Fourier coefficients of the
  critical solution to decay exponentially in τ with phases linear in k, and compared them with the sum
  Σ e^{−a|k|} e^{4πikτ/Δ} [MG03, Sec. IV.2, Eq. (59)]."
- **No novelty sentence** was added by this task. The strongest statement made is "not found in the sources searched".

## Addendum (2026-11-09, issue #198): Fabry–Pérot cavities, nonlinear cavities and Fratini et al.'s later work

**Question.** #172 found that Eq. (16) of EEG26b, A* Re[e^{iθ}/(1 + w e^{2iθ})], is, up to a factor, the field a
Fabry–Pérot cavity transmits, e^{−iφ}/(1 − r₁r₂e^{−2iφ}) with θ = −φ and w = −r₁r₂. Issue #198 asks whether optics has
results or methods this project can use:
- **Q1, the linear cavity.** Do its line shapes and their deformations (losses, dispersion, extra zeros and poles)
  match #133's rational extensions (notes/EEG_PROFILE.md §8) or the branch point at the profile's pole (§3.3, §8.5)?
- **Q2, nonlinear cavities.** Kerr cavities, the Ikeda round-trip map and the Lugiato–Lefever equation (LLE). Do their
  methods give tools for a fixed point of a one-period map with one unstable direction: Floquet stability,
  continuation, harmonic balance, period doubling?
- **Q3, Fratini et al.'s later work.** Nonlinear transport or rectification models whose stationary states are
  geometric sums with a nonlinear ratio.

**How.**
- **Searches.** 25 web queries and 4 INSPIRE queries, all on 2026-11-09, listed in §5.
- **Reads.** 19 sources opened, 13 in full or in part and 6 as abstracts or publication lists, and the bibliographic
  records of 9 more, listed in §6. Quotes and URLs are in the raw-data repository, folder `I198-mac/`, commit
  a3abcb5.
- **PDFs.** Where `WebFetch` returned a PDF as binary, the copy it saved was extracted with pypdf 7.0.0 in a scratch
  environment. That opened Pollnau–Eichhorn (2020) and Marques et al. (2021).
- **Identities.** Four identities and two asymptotic laws that tie the sources to Eq. (16) are *inferred* and
  *checked numerically* by `I198-mac/identity_check.py` (mpmath, 30 digits, exit 0). Its mutation makes all six fail
  (exit 1).
- **Numbers.** EEG26b's q = 0.0551 comes from their extrapolation of finite-D solutions. The tuned D = ∞ profile's
  numbers (its singularity at 1.5344 ∓ 1.4284i, its line of Padé poles, the k^{−4} law of its ratio differences) assume
  the selection rule, Conjecture 1 of the math paper, and are marked *conditional*.
- **Verdict classes**, fixed before the search: a usable result or method, named with its equation; an analogy only;
  nothing found in the sources searched.

### Closest works

| Work | Bears on | What it contains | What it does not contain |
|---|---|---|---|
| **M. Pollnau, M. Eichhorn**, Prog. Quantum Electron. 72, 100255 (2020), KITopen PDF, Secs. 3.5–4.3 | Q1 | Eq. (99): each longitudinal mode profile is a Lorentzian, written as a pole pair "1/(δ − i(2ϕ − 2πq)) + 1/(δ + i(2ϕ − 2πq))". Summed with Eq. (100), Σ 1/(x − q) = π cot πx, they give the Airy distribution exactly, Eq. (108), with e^{−δ} = √(R₁R₂). Eq. (114): the Lorentzian linewidth equals the free spectral range at R₁R₂ = e^{−2π}. Eq. (118): the Airy linewidth (FWHM) is undefined below R₁R₂ = 17 − 12√2 ≈ 0.02944. With frequency-dependent reflectivities "the mode profiles are not Lorentzian-shaped anymore" and the Airy distributions "become distorted" (citing Ismail et al.). Propagation loss multiplies R₁R₂ by e^{−2α ℓ} (Eqs. (111)–(112)). | A functional form of a frequency-dependent reflectivity, poles off the round-trip lattice, zeros, branch points, a nonlinear cavity. |
| **N. Ismail, C. C. Kores, D. Geskus, M. Pollnau**, Opt. Express 24, 16366 (2016) | Q1 | The source the issue names. Known here only through Pollnau–Eichhorn and Wikipedia's *Fabry–Pérot interferometer*, Secs. 3.5 and 3.8 (secondary). | FULL-TEXT-UNREAD: Optica returned a loading page. |
| **L. L. Sánchez-Soto, J. J. Monzón, G. Leuchs**, Eur. J. Phys. 37, 064001 (2016), arXiv:1610.06163 | Q1 | Eq. (2.1), the round-trip sum; Eq. (2.13), the self-consistent field A(0) = A(0)r²e^{−2iφ} + t′A_in; "its poles and zeros provide full information of the response of the system"; Eq. (2.16), the impulse response, a pulse train decaying by r² per round trip; Eq. (3.7), the Airy distribution as (1−r²)/(1+r²) Σ_m (Γ/2)/((φ − mπ)² + (Γ/2)²), Γ = ln(1/r²). | Losses ("we assume that such a mirror is lossless"), coupled cavities, nonlinear effects. The poles are not listed. |
| **D. M. Marques, J. A. Guggenheim, P. R. T. Munro**, Opt. Express 29, 24144 (2021) | Q1 (the branch point) | Eq. (1): transmission √(1−R₁)√(1−R₂)e^{iφ/2}/(1 − √(R₁R₂)e^{iφ}), and reflection a ratio of first-degree expressions in e^{iφ} (a zero and a pole), φ = 4πNh cos θ/λ. Eq. (3), the angular Airy function Ê_fp(θ, ϕ) = Ê_i(θ, ϕ)A(θ): each plane-wave component of a beam has its own round-trip phase. It explains "the asymmetry, broadening and visibility reduction" of focused Gaussian beams; the asymmetry arises because only on one side of resonance do further components "enter the filtering condition". | The analytic structure of the averaged line, a closed form, a law for the amplitude of the m-th round trip. |
| **T. Hansson, S. Wabnitz**, JOSA B 32, 1259 (2015), arXiv:1503.03274 | Q2, Q3 | Eqs. (1)–(2), the Ikeda map. Eq. (4), its fixed point E = ρe^{iφ}E + √θE_in with φ = δ₀ − γL_eff\|E\|², so the stationary field is the round-trip geometric sum with a ratio that depends on the field (*inferred*). Eq. (5), the Airy function of the nonlinear phase; "Either one or three stationary states" for small rotation, more at larger. Sec. IV: the one-round-trip matrix Q, unstable "whenever the modulus of an eigenvalue of Q is larger than unity"; Eq. (11), q± = √(1−θ)(p ± √(p² − 1)), where q₋ "gives rise to a period doubling instability". The LLE is the round-trip average of the map and misses these states and the period doubling. | Newton iteration or continuation of the fixed points (they come from Eq. (5) and a graphical construction), harmonic balance. |
| **M. Conforti, A. Mussot, A. Kudlinski, S. Trillo**, Opt. Lett. 39, 4200 (2014), arXiv:1405.6041 | Q2 | The 2×2 one-round-trip matrix from two initial conditions; its eigenvalues are the Floquet multipliers, unstable for \|λ\| > 1 with gain ln\|λ\|/Λ; "MI if \|Δ\|>(1+W)", Δ the Floquet discriminant and W the Wronskian; a pulse train "reproduced identical after two round-trips". | Continuation, harmonic balance, Hill's method. |
| **D. Gomila, M. A. Matías, P. Colet**, PRL 94, 063905 (2005), arXiv:nlin/0411047 | Q2 | In the 2D LLE, the middle-branch localized structure "has a single unstable direction". Near the saddle-loop bifurcation the period obeys T ∝ −(1/λ₁) ln(θ_c − θ), λ₁ "the unstable eigenvalue of the saddle point": 1/λ₁ = 5.65 against a fitted slope of 5.60. | How λ₁ was computed (deferred to an earlier paper). |
| **P. Parra-Rivas, E. Knobloch, L. Gelens, D. Gomila**, arXiv:2010.03375 (review) | Q2 | LLE stationary states by "numerical path continuation" with "a predictor-corrector approach"; stability from "the eigenvalues of the Jacobian matrix obtained from L after spatial discretization"; breathers born at a Hopf bifurcation, then "a period-doubling bifurcation (PD)", chaos, and a boundary crisis with "the infinite-dimensional saddle". | Continuation of the periodic orbits, their Floquet multipliers, the software. |
| **C. Maitland, M. Conforti, A. Mussot, F. Biancalana**, Phys. Rev. Research 2, 043195 (2020), arXiv:2004.14095 | Q2 | Two fibre loops, "the output of each is fed into the other". One-round-trip fixed points; a 4×4 Floquet matrix with eigenvalue pairs ±λ₁, ±λ₂; period-two and period-four trains in simulations. | A reduction by the swap symmetry (a half map composed with the swap); their period doubling is "outside the scope". |
| **F. Fratini et al.**, PRL 113, 243601 (2014), arXiv:1410.5972 | Q3 | (S.2), the round-trip sum with the ratio √(R₁R₂)e^{2ikL}e^{i(θ₁+θ₂)}; Eq. (1), the reflectance of an emitter falls with the power p at it; (S.7), the powers at the emitters "appear both at the right- and the left-hand sides", solved numerically. So the stationary state is a geometric sum whose ratio depends on the field it sums. | Multiple solutions, bistability, any stability analysis. |
| **Its successors**: Fratini–Ghobadi, PRA 93, 023818 (2016); Mascarenhas–Santos–Auffèves–Gerace, PRA 93, 043821 (2016); Almeida–Werlang–Valente, JOSA B 36, 3357 (2019) | Q3 | Full-quantum models: a master equation with Bloch–Langevin equations; a quantum stochastic master equation; a single-excitation Schrödinger equation with a transfer-matrix limit. "the quantumness reduces the nonlinearity of the system"; "the Fabry-Perot approach overestimates the diode efficiency". | A round-trip sum, a self-consistent geometric series, bistability, a stability analysis. |

### The identities

*Inferred*, and *checked numerically* by `I198-mac/identity_check.py`:
1. **The δ = 0 profile is a carrier times the Airy distribution.** With 𝒜(φ) = 1/(1 + F sin²φ) and the coefficient of
   finesse F = 4q/(1−q)² (Sánchez-Soto et al., Eqs. (3.1)–(3.2), with r² = q),
   β̃ = A*(1+q)/(1−q)² · cos θ · 𝒜(θ + π/2). For any δ, Re(1/h) = Re(h̄)/|h|², so the profile is a first harmonic
   times 1/|h|², which is EEG_PROFILE.md §2 (R) in the cavity's terms. #172 already matched that denominator to the
   Poisson kernel.
2. **The hump threshold is F = 1.** The δ = 0 profile has a single maximum per half period exactly when F < 1, so
   EEG_PROFILE.md §2's q_c = 3 − 2√2 is F = 1. The same condition gives Pollnau–Eichhorn's Eq. (118):
   (3 − 2√2)² = 17 − 12√2 is the reflectivity product below which the Airy peaks have no full width at half maximum.
3. **The Lorentzian half-width is the pole distance.** In Eq. (3.7) of Sánchez-Soto et al., Γ/2 = ½ ln(1/q) is
   Eq. (16)'s σ, 1.4493 at the paper's q. The cavity's "Lorentzian linewidth equals the round-trip decay rate" is
   EEG_PROFILE.md §2's "strip of analyticity equals the decay rate per harmonic".
4. **Eq. (114) is 2σ = π**, at q = e^{−π} = 0.0432.
5. **A pole smeared over a spread of round-trip phase.** A paraxial Gaussian beam in two transverse dimensions has a
   density of phase deficits that is finite at zero deficit (*inferred* from Marques et al., Eqs. (2)–(3)). That gives
   coefficients w^k[1 + b s₀/(1 + 2iks₀)], a logarithm at the pole, and ratio differences |r_{k+1} − r_k| falling like
   k^{−3}.
6. **A pole smeared over a spread of round-trip loss**, with a density that vanishes linearly at the pole's own loss,
   gives w^k[1 + b/((k+1)(k+2))] and differences falling like k^{−4}. That is the law EEG_PROFILE.md §8.5 reads from
   the tuned profile (*conditional*), and the pole with a (θ − θ₀)ln(θ − θ₀) term proposed there. A phase spread with
   a linear edge also gives k^{−4}.

At EEG26b's q = 0.0551 the echo is a low-finesse cavity: F = 0.247 < 1, so in Pollnau–Eichhorn's terms its Airy
linewidth is undefined, and its Lorentzian finesse π/(2σ) = 1.084 lies just above their Eq. (114) threshold.

### Verdicts

**Q1, the linear cavity: an analogy only.**
- **Line shapes.** The Airy distribution and its Lorentzian (pole-pair) decomposition are the cavity's version of
  facts EEG_PROFILE.md §2 already states: the Poisson-kernel denominator, one pole pair per half period, and the pole
  distance equal to the decay rate. Identities 1–4 rename those facts and add no tool. Per Koppelmann, as quoted by
  Pollnau–Eichhorn, the exact decomposition goes back to Bayer-Helms, Z. Angew. Phys. 15, 330 (1963) *(unverified)*.
- **Losses** only multiply the round-trip ratio by e^{−2α ℓ}, so the line keeps Eq. (16)'s [0/1] form.
- **Extra zeros and poles.** The field a Fabry–Pérot reflects has one zero and one pole in the round-trip
  exponential (Marques et al., Eq. (1)). Apart from the single-pass factor e^{iθ}, that is the structure of
  EEG_PROFILE.md §3.3's one-zero [1/1] form (*inferred*). Cavities with more mirrors or stages give rational functions
  of e^{2iφ} of higher degree, which Madsen and Zhao's book organizes as moving-average, autoregressive and ARMA
  architectures (*inferred* from its table of contents only). So #133's [L/N] family is the class of multi-stage
  cavity transfer functions. No source gives a reason for particular L, N or coefficients.
- **Dispersion.** A frequency-dependent reflectivity distorts the mode profiles and the Airy distributions
  (Pollnau–Eichhorn; Wikipedia, Sec. 3.8), but no source read gives a form for it or for its poles.
- **The branch point.** The one optical mechanism found that turns a pole into a branch point is an average over a
  spread of round-trip phases, the angular Airy function of Marques et al. (Eq. (3)). *Inferred*: a phase spread moves
  the pole along Re θ, so its cut runs parallel to the real axis, while a spread of round-trip loss moves it along
  Im θ. The tuned profile's further Padé poles lie along Re θ ≈ 1.55–1.61, outward in |Im θ| (EEG_PROFILE.md §8.5,
  *conditional*), which is the loss direction. The Gaussian beam's finite-edge density also gives the wrong power
  (identity 5, k^{−3} against the profile's k^{−4}). No source read treats a spread of round-trip loss. So optics
  supplies a mechanism of the right kind, a smeared pole, but not the one the profile shows.

**Q2, nonlinear cavities: an analogy only.** The methods exist, and the project already has their counterparts.
- **The fixed point.** The Ikeda map's stationary state (Hansson–Wabnitz, Eqs. (4)–(5)) is the round-trip geometric
  sum with a field-dependent ratio, solved as one algebraic equation for |E|², with one or three roots. The echo's
  ratio also depends on the amplitude at leading order (w ≈ qA², EEG_PROFILE.md §3.1), but it is set by the Stokes
  conditions, not by a closed equation for the sum (*inferred*).
- **Stability.** The cavity papers take the eigenvalues of the linearized one-round-trip map (Hansson–Wabnitz,
  Eq. (11); Conforti et al., |Δ| > 1 + W). Their background is homogeneous, so perturbations separate by sideband
  frequency and each matrix is 2×2. The echo's background varies in τ, so its harmonics couple. The project's
  det M(λ) with its winding count (notes/MODE_COUNT.md) is that Floquet analysis for a non-homogeneous fixed point
  (*inferred*), and no source read offers a replacement.
- **The one unstable direction.** Gomila et al.'s middle-branch structure "has a single unstable direction", and the
  time spent near it scales as −(1/λ₁) ln(θ_c − θ), measured at 5.60 against 1/λ₁ = 5.65. That is the mechanism behind
  γ = 1/λ in critical collapse [KoikeHaraAdachi1995], seen in a cavity.
- **Continuation.** Predictor–corrector continuation with the eigenvalues of the discretized Jacobian (Parra-Rivas et
  al.). The project's continuation, the asymptotic numerical method (notes/LARGE_D_LITERATURE.md), is of higher order.
- **Harmonic balance.** Kerr-comb models use a modal (Galerkin) expansion (Chembo–Yu, records only) or the LLE in
  space and time, which Chembo–Menyuk find "significantly more suitable" for wide combs (abstract). The project
  collocates in τ, which corresponds to the second choice (*inferred*).
- **Averaging.** The LLE is the round-trip average of the Ikeda map. It needs each round trip to change the field
  little and fails for an O(1) nonlinear phase per pass (Hansson–Wabnitz). With F = 0.247 the echo is far from that
  regime (*inferred*), so averaging gives no continuous-time model here.
- **Period doubling.** In the cavities it is a bifurcation: a Floquet multiplier crosses −1 (*inferred* for
  Hansson–Wabnitz, who tie it to q₋ without naming the sign; the breathers of Parra-Rivas et al.). The echo's
  odd-harmonic structure, F(θ + π) = −F(θ), is a symmetry instead. Its cavity analogue would be a fixed point of a
  half-period map composed with a sign flip. The coupled-loop resonator of Maitland et al. has eigenvalue pairs ±λ
  but makes no such reduction. The split of cavity instabilities into
  period-one (q₊) and period-two (q₋) matches MODE_COUNT.md's two symmetry sectors of the perturbations (*inferred*).

**Q3, Fratini et al.'s later work: nothing found in the sources searched.**
- Within this line, Fratini et al. (2014) is the only source read whose stationary state is a geometric sum with a
  nonlinear ratio, (S.2) and (S.7). It is solved numerically, with no count of solutions and no stability analysis.
- The line's later papers replace the round-trip sum by full-quantum models: Fratini–Ghobadi (2016), Mascarenhas et
  al. (2016) and Almeida et al. (2019) in full, and Dai et al. (2015), Roy (2017), Manasi–Roy (2018) and Zheng et al.
  in abstract. None contains a round-trip sum or a self-consistent geometric series.
- The classical antecedents with the same structure are the Ikeda fixed point (Q2) and the Kerr-filled Fabry–Pérot
  of Felber and Marburger, PRA 17, 335 (1978), whose field is solved "exactly ... in terms of elliptic functions"
  according to a search summary of its abstract. It was not read.

### Citation chains (INSPIRE, 2026-11-09)

- **Forward from Fratini et al.** (recid 3025790): 57 citing records, listed in `I198-mac/fratini_citations.md`. Six
  were opened (three in full, three as abstracts). The other 51 rest on titles, which are about waveguide QED and
  nonreciprocity.
- **Forward from Fratini–Ghobadi** (recid 3027964): 21 records, 6 of them not in the first list; Zheng et al. was
  opened as an abstract.
- **Fratini's own later work**: 17 INSPIRE records. After 2014 this line continues only in Fratini–Ghobadi. His web
  page adds an optical transistor with a quantum dot in a photonic wire, PRB 97, 201106(R) (2018), not opened.
- **Backward from Pollnau–Eichhorn**: Ismail et al. [2], Koppelmann [51] and Bayer-Helms [52] *(unverified)*.
- **Backward from Marques et al.**: La Penna et al., Opt. Commun. 162, 267 (1999); Nichelatti–Salvetti, Appl. Opt.
  34, 4703 (1995); Lee–Hahn–Lee, JOSA A 19, 973 (2002), on finite beams in plane-parallel cavities. Not opened, all
  *(unverified)*.

### References checked

| Entry | Checked against | Verdict |
|---|---|---|
| Pollnau, Eichhorn, Prog. Quantum Electron. 72, 100255 (2020), doi:10.1016/j.pquantelec.2020.100255 | the publisher's PDF on KITopen (running footer) and the Surrey repository record | OK |
| Ismail, Kores, Geskus, Pollnau, Opt. Express 24(15), 16366–16389 (2016), doi:10.1364/OE.24.016366 | the search listing of the Optica abstract page (the page itself did not load) and Pollnau–Eichhorn's reference [2] | PARTIAL |
| Sánchez-Soto, Monzón, Leuchs, Eur. J. Phys. 37, 064001 (2016), doi:10.1088/0143-0807/37/6/064001 | the arXiv page of 1610.06163 (journal-ref, DOI) | OK |
| Marques, Guggenheim, Munro, Opt. Express 29(15), 24144–24150 (2021), doi:10.1364/OE.431362 | the typeset PDF on UCL Discovery (header) | OK |
| Hansson, Wabnitz, JOSA B 32(7), 1259–1266 (2015), doi:10.1364/JOSAB.32.001259 | the arXiv page (related DOI) and the Chalmers record (volume, issue, pages) | OK |
| Conforti, Mussot, Kudlinski, Trillo, Opt. Lett. 39, 4200 (2014), doi:10.1364/OL.39.004200 | the arXiv page of 1405.6041 (related DOI); volume and page read from the DOI | PARTIAL |
| Gomila, Matías, Colet, PRL 94, 063905 (2005), doi:10.1103/PhysRevLett.94.063905 | the arXiv page of nlin/0411047 (journal-ref, DOI) | OK |
| Parra-Rivas, Knobloch, Gelens, Gomila, arXiv:2010.03375 | the arXiv page (no journal-ref); IMA J. Appl. Math. per a search listing *(unverified)* | PARTIAL |
| Maitland, Conforti, Mussot, Biancalana, Phys. Rev. Research 2, 043195 (2020) | the arXiv page of 2004.14095 (journal-ref, DOI) | OK |
| Chembo, Menyuk, PRA 87, 053852 (2013), arXiv:1210.8210 | the arXiv page (related DOI 10.1103/PhysRevA.87.053852); the year read from the volume | PARTIAL |
| Fratini et al., PRL 113, 243601 (2014); Fratini, Ghobadi, PRA 93, 023818 (2016); Mascarenhas et al., PRA 93, 043821 (2016); Almeida, Werlang, Valente, JOSA B 36, 3357 (2019); Dai et al., PRA 92, 063848 (2015); Roy, PRA 96, 033838 (2017); Manasi, Roy, PRA 98, 023802 (2018) | INSPIRE records (recids in `I198-mac/fratini_citations.md`); arXiv pages for the last three | OK |
| Ikeda, Opt. Commun. 30, 257–261 (1979); Ikeda, Daido, Akimoto, PRL 45, 709–712 (1980); Lugiato, Lefever, PRL 58, 2209–2211 (1987); Coen, Haelterman, PRL 79, 4139–4142 (1997); Haelterman, Trillo, Wabnitz, Opt. Commun. 91, 401–407 (1992); Felber, Marburger, PRA 17, 335–342 (1978); Chembo, Yu, PRA 82, 033801 (2010); Madsen, Zhao, *Optical Filter Design and Analysis* (Wiley, 1999) | search listings and library records only | UNVERIFIABLE here, cited as not read |

### Gap statement

- **Not read.** Ismail et al. (2016), the source the issue names (Optica returned a loading page). The classical
  papers on nonlinear cavities: Ikeda (1979), Ikeda–Daido–Akimoto (1980), Felber–Marburger (1978), Lugiato–Lefever
  (1987), Haelterman–Trillo–Wabnitz (1992), Coen–Haelterman (1997). Madsen–Zhao's book. The textbooks (Born–Wolf,
  Siegman, Svelto, Vaughan). The finite-beam papers behind Marques et al.
- **Not from the cavity sources.** Hill's method and its convergence theory (Deconinck–Kutz 2006; Curtis–Deconinck;
  Johnson–Zumbrun) surfaced in a search but in no cavity source read, so they were not opened. They bear on
  MODE_COUNT.md's extra pair at M = 48 more than on this issue.
- **Channels.** Google Scholar's cited-by list was not reached, so the forward chains rest on INSPIRE, which may miss
  optics journals. Of the 57 records citing Fratini et al., 51 rest on titles.
- **Rounds.** Round 1 covered the three named topics. Round 2 followed the INSPIRE chains and the angular-Airy and
  excitability leads. The search is not dry: the finite-beam papers and any optical treatment of a spread of
  round-trip loss remain open.

### Discrepancies and proposed text (for the coordinator)

- **No discrepancy** with the repository's text was found.
- **A pointer.** #172 recorded Wikipedia's citation of Svelto's *Principles of Lasers*, 5th ed., as "ch. 4.5.1
  *(unverified)*". Pollnau–Eichhorn cite the same edition as "pp. 169–171. Ch. 5.3" for the Airy linewidth. They may
  cite different passages, and neither was checked against the book.
- **Proposed text** for notes/EEG_PROFILE.md §2, after the hump threshold: "At δ = 0 the profile is cos θ times the
  Airy distribution of a Fabry–Pérot cavity, A*(1+q)/(1−q)² · cos θ/(1 + F cos²θ), with coefficient of finesse
  F = 4q/(1−q)². The threshold q_c = 3 − 2√2 is F = 1, the reflectivity below which the Airy peaks have no full width
  at half maximum [Pollnau–Eichhorn 2020, Eq. (118)]." Its check is `I198-mac/identity_check.py`.
- **No novelty sentence** was added by this task. The strongest statement made is "not found in the sources searched".

### Recommended tasks (for the coordinator to file)

1. **The direction of the echo's branch cut: a planted test with smeared poles.**
   - *What.* Add two plants to `solver/eeg_profile/rational_fit.py`'s planted-truth path: a pole smeared over a spread
     of round-trip loss and one smeared over a spread of round-trip phase, both with a linearly vanishing edge
     (identities 5–6). Run the Padé pole march of EEG_PROFILE.md §8.3 on both, then on the tuned data.
   - *Why.* The one optical mechanism for a branch point at a pole (Q1) predicts the direction of the cut. The test
     says which class of smeared pole the profile belongs to, for #150.
   - *Acceptance.* On the loss plant the second Padé pole marches with |ΔRe θ₂| < 0.3 |ΔIm θ₂| (n = 2 to 4), and on
     the phase plant with |ΔIm θ₂| < 0.3 |ΔRe θ₂|. The test fails if both plants march the same way, and the
     real-data verdict is then void. Report the tuned profile's direction, *conditional* on Conjecture 1.
   - *Depends on.* #133, #198. A few minutes on one core.
2. **Read the cavity sources that did not open.**
   - *What.* Ismail et al. (2016), Felber–Marburger (1978), Ikeda (1979) and Coen–Haelterman (1997), through a channel
     that renders Optica and APS pages, logged in §6.
   - *Why.* The Q1 and Q2 verdicts rest on secondary accounts of these four.
   - *Acceptance.* Each is read in full and its content recorded. The task fails, and says so, if one of them
     contains a closed form for a frequency-dependent reflectivity, a branch point at a cavity pole, or a stability
     method for a non-homogeneous fixed point that the verdicts above miss.
   - *Depends on.* #198.

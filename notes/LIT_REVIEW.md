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
| **C. Gundlach, D. Hilditch, J. M. Martín-García**, *Critical Phenomena in Gravitational Collapse*, arXiv:2507.07636 ([HTML](https://arxiv.org/html/2507.07636v1)) | P2 | Full text (HTML); read again in full from the arXiv source in the addendum of issue #380, which supersedes this row | No large-$`D`$ results and no limits of $`\Delta`$ or $`\gamma`$. It does not mention Rozali–Way, Clark–Pimentel or Ecker et al. Sec. IV.A.2: Kol "claims to obtain analytic estimates for γ and Δ in arbitrary dimension". Sec. III.C: the critical solution "was determined accurately in [43] by assuming analyticity at the centre of spherical symmetry and at the past light cone". No exponential asymptotics. |
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
  repository, `I172-mac/search_log.md` (commit 46443de).
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
  `I198-mac/search_log.md` (commit 92526ac), with what each surfaced; shortened here.
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
- **2026-11-09, issue #200 (the superconducting threshold, mean-field exponents, RG limit cycles and viscosity).**
  `WebSearch` in standard mode, and the INSPIRE REST API through `WebFetch`. The queries are verbatim in the raw-data
  repository, `I200-mac/search_log.md` (commit 61bc23f), with what each surfaced; shortened here.
  - Q1: Mariño–Reis's title with "BCS gap non-perturbative perturbative series ground state energy"; the titles of BCS
    (1957) and Cooper (1956) with "pdf" (records only); "\"beyond all orders\" OR \"exponential asymptotics\"
    superconductivity gap equation Stokes phenomenon trans-series BCS essential singularity resummation" (nothing on
    the gap equation itself).
  - Q2 and Q3: Thouless's 1960 title with "pair susceptibility divergence T-matrix pole" (secondary sources only);
    "Vollhardt review infinite dimensions mean-field theory exact \"1/sqrt(d)\" hopping scaling Metzner Vollhardt 1989
    Ising model Weiss mean field exact d to infinity arXiv"; "upper critical dimension mean-field exponents exact above
    d=4 Ginzburg criterion large-N saddle point exponent 1/2 order parameter beta=1/2 why"; "Penrose criterion plasma
    instability argument principle winding number Nyquist count unstable modes dielectric function zeros upper half
    plane" (lecture notes only).
  - Q4: "holographic superconductor large D limit 1/D expansion critical temperature analytic"; "dynamical formation
    of hairy black hole holographic superconductor critical behavior threshold scaling law Choptuik-like time evolution
    charged scalar AdS" (no threshold-scaling study).
  - Q5: "\"Russian doll\" renormalization group superconductivity LeClair Roman Sierra limit cycle BCS".
  - Q6: "critical collapse viscous fluid black hole threshold self-similar bulk viscosity shear viscosity critical
    exponent" (nothing on viscous critical collapse); "Emparan Suzuki Tanabe large D effective theory black brane
    viscous fluid shear viscosity eta/s = 1/4pi \"large D\" hydrodynamics"; "holographic superfluids Landau criterion
    quasinormal mode crosses real axis instability critical velocity Amado Arean Jimenez-Alba Landsteiner"; "Landau
    criterion superfluidity critical velocity onset of dissipation review Leggett \"Superfluidity\" Reviews of Modern
    Physics 1999" (Leggett's review did not surface).
  - INSPIRE: `q=t viscous and t collapse and (t critical or t self-similar or t threshold)` (1 record);
    `q=(t viscosity or t viscous or t dissipative or t "Israel-Stewart") and (t "critical collapse" or t "critical
    phenomena" or t "gravitational collapse")` (25); `q=refersto:recid:371808 and (t viscous or t viscosity or t
    dissipative or t heat or t kinetic or t Vlasov or t collisionless or t anisotropic)` (2); `q=(t Choptuik or t
    "critical collapse" or t "critical gravitational collapse" or t "critical phenomena") and (t superconductor or t
    superconducting or t superfluid or t hairy or t hair)` (2); `q=(t "limit cycle" or t "limit cycles" or t "discrete
    scale invariance" or t Efimov or t "Russian doll") and (t superconductor or t superconductivity or t BCS or t
    "Cooper pair" or t "Cooper pairs")` (5); `q=refersto:recid:1267488` (45); `q=refersto:recid:1684230` (13);
    `q=refersto:recid:1736509` (38); `q=(t "large D" or t "large dimension" or t "infinite dimensions" or t "1/D
    expansion") and (t collapse or t "horizon formation" or t "mean field" or t threshold or t critical)` (7); and three
    batches of `q=arxiv <id> or ...` for the bibliographic records.
- **2026-11-09, issue #221 (Pollnau and Eichhorn's resonator work).** `WebSearch` in standard mode, and the OpenAlex,
  Europe PMC, Unpaywall and INSPIRE REST APIs through `WebFetch`. The queries are verbatim in the raw-data repository,
  `I221-mac/search_log.md` (commit 4ecb5c2), with what each surfaced; shortened here.
  - Part II and later work: `Pollnau Eichhorn "Spectral coherence, Part II"`, the same with "factor of two threshold"
    and with "Progress in Quantum Electronics 2021 2022 2023", and "Markus Pollnau laser linewidth 2021 OR ... 2025"
    (Part I only, each time).
  - Ismail et al.: the full title with "pdf"; the title with the Twente and KTH repository domains; the SPIE companion
    paper's title. Europe PMC `DOI:10.1364/OE.24.016366` (PMID 27464090); Unpaywall (the KTH DiVA copy).
  - Line shapes and thresholds: "Fabry-Perot resonator transmission \"branch point\" OR \"branch cut\" pole complex
    frequency frequency-dependent mirror reflectivity analytic continuation line shape" (nothing on branch points);
    "laser threshold pole crosses real axis complex frequency \"gain equals loss\" ... Pollnau spectral-coherence
    factor" (the S-matrix-pole picture of threshold, not opened); the title and authors of Vujnović et al. (no arXiv
    copy).
  - OpenAlex: Part I (W3010732843) and `filter=cites:` (43 records); Ismail et al. (W2461253794, 305 citations) and
    `filter=cites:` with a title search for pole, singularity, analytic, Lorentzian, Airy, line shape, Fano, zero,
    threshold or branch (16 records); Pollnau's works since 2017 (60) and with "spectral coherence" in the title (10,
    Part I the only one of this line); Eichhorn's works since 2016 with resonator or linewidth words (20).
  - INSPIRE: `q=a Pollnau` (0 records); the `refersto:doi:` form is ignored by the API (an unfiltered list), so INSPIRE
    gives no chain for this line of work.

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
  `I172-mac/read_log.md`, commit 46443de):
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
  the tool saved; quotes and URLs in the raw-data repository, `I198-mac/read_log.md`, commit 92526ac):
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
- **2026-11-09, issue #200** (through `WebFetch`; Kol's PDF, which came back as binary, was extracted with pypdf from
  the copy the tool saved; quotes and URLs in the raw-data repository, `I200-mac/read_log.md`, commit 61bc23f):
  - Full extraction: Mariño–Reis 1905.09569; Polchinski hep-th/9210046; LeClair–Román–Sierra cond-mat/0211338;
    Kaplan–Lee–Son–Stephanov 0905.4752; Kol hep-th/0502033v3 (PDF); Hartnoll–Herzog–Horowitz 0803.3295; Emparan–Tanabe
    1312.1108; Hartnoll–Horowitz–Kruthoff–Santos 2008.12786; Camps–Emparan–Haddad 1003.3636; Rozali–Way from Sec. 3 to
    the end; Amado et al. 1307.8100; Koike–Hara–Adachi gr-qc/9503007; Maison gr-qc/9504008; Evans–Coleman
    gr-qc/9402041; Kovtun–Son–Starinets hep-th/0405231; Marsiglio–Beach–Gooding 1203.5122; Emparan–Herzog 2003.11394
    (characters 0–300000 of 302723, Sec. X.3 included).
  - Partial: Braaten–Hammer (0–200000 of 248219); Hara–Koike–Adachi gr-qc/9607010 (0–100000 of 216312); Larkin–Varlamov
    (0–200000 of 224656); Vollhardt 2210.06072 (0–100000 of 172147); Emparan–Izumi–Luna–Suzuki–Tanabe 1602.05752
    (0–100000 of 117569); Emparan–Rafecas-Ventosa–Way 2512.14186 (0–100000 of 197383); Yuwen et al. 2606.26532
    (0–100000 of 122393); Schmitt 1404.1284 (0–100000 of 244179); the 2025 review 2507.07636 (0–100000, Secs. II.5–II.7
    and a keyword search); Wikipedia, *Ginzburg criterion* (a secondary source).
  - Abstract or record only: Georges–Kotliar–Krauth–Rozenberg cond-mat/9510091; Glazek–Wilson hep-th/0203088;
    Mariño–Reis 2010.16174; Müller–Schäfer 1710.00136; Rein–Rendall–Schaeffer gr-qc/9804040; Olabarrieta–Choptuik
    gr-qc/0107076; Chavanis 1002.0291; Silveira, EPL 149, 49001 (2025) (INSPIRE record).
  - Failed (FULL-TEXT-UNREAD): Cooper (1956), a scan that returned 404; BCS (1957) and Thouless (1960), no full text
    surfaced; Beekman–Rademaker–van Wezel 1909.01820, whose Chapter 5 (Landau theory) the extraction did not reach.
- **2026-11-09, issue #221** (PDFs that `WebFetch` returned as binary were extracted page by page with macOS PDFKit
  from the copy the tool saved; one page was read as text through Claude in Chrome; quotes and URLs in the raw-data
  repository, `I221-mac/read_log.md`, commit 4ecb5c2):
  - Full extraction: Pollnau–Eichhorn 2020 (KITopen PDF, all 67 pages, Secs. 2–4 and 6–10 closely); Ismail–Kores–
    Geskus–Pollnau 2016 (the published version in KTH DiVA, all 25 pages).
  - Partial: Vujnović–Kralj–Karuža, Photonics 13, 721 (2026), Sections 1–6 (characters 0–50000 of 80809, through
    Chrome; Appendices A–D and the references unread).
  - Abstract or record only: Eichhorn, Appl. Phys. B 128 (2022); He–Lu–Zhao, Opt. Express 27, 17876 (2019); Atvars,
    JOSA B 38, 3116 (2021); Zhou et al., JLT 43, 1007; Pollnau, Opt. Lett. 43, 5033 (2018); Kores et al., IEEE
    Photonics J. (2018) and CLEO/Europe (2019); Yeung–Pollnau, SPIE (2020); Pollnau, ICTON (2019) (OpenAlex records);
    Ismail et al., SPIE 10090 (2017) (search listings).
  - Failed (FULL-TEXT-UNREAD): the scispace copy of Ismail et al. (empty) and its Optica pages (loading placeholder);
    the SPIE page of the 2017 paper (empty); MDPI through `WebFetch` (HTTP 403, read through Chrome instead); the
    Pollnau–Ismail book chapter (2018), of which only a university proxy mirror surfaced, not used.

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
  Quotes and URLs are in the raw-data repository, folder `I172-mac/`, commit 46443de.
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
  92526ac.
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

## Addendum (2026-11-09, issue #200): the superconducting threshold, mean-field exponents, RG limit cycles and viscosity

**Question.** Issue #200 asks whether critical collapse at large D shares mathematical structure with the
superconducting threshold and with viscosity, read toward the selection rule (Conjecture 1 of the math paper), γ∞ = 1/2
and the mode count:
- **Q1, results beyond all orders.** The BCS gap Δ ∝ ω_D e^{−1/(N(0)V)} beside the period, which Proposition 1 leaves
  free to every order in 1/D. Do the methods that resolve the BCS essential singularity apply to the selection rule?
- **Q2, mean-field exponents.** γ → 1/2 at D = ∞ (Proposition 5) beside the exponents 1/2 of Ginzburg–Landau and
  mean-field theory.
- **Q3, the threshold as one unstable mode.** The Cooper or Thouless criterion beside the critical solution's one
  unstable mode: how each counts its modes, and the growth rate near threshold.
- **Q4, holographic superconductors at large D.** Methods for the 1/D expansion of a threshold.
- **Q5, discrete scale invariance.** Efimov physics and RG limit cycles, and whether superconductivity uses them.
- **Q6, viscosity.** (a) Superfluidity. (b) Large-D horizons as viscous fluids, η/s = 1/4π, and whether that effective
  theory describes the threshold or the black hole formed past it. (c) Whether viscosity changes the perfect-fluid
  critical solution or its universality class. Viscous fingering, already PROOF.md's template, was not re-read.

**How.**
- **Searches.** 15 web queries and 13 INSPIRE queries, all on 2026-11-09, listed in §5.
- **Reads.** 36 sources opened: 17 in full, 10 in part (one of them secondary), 8 as abstracts or records, and one
  whose relevant chapter did not come through. Three more were not found. All are listed in §6, with quotes and URLs in
  the raw-data repository, folder `I200-mac/`, commit 61bc23f.
- **Identities.** Six relations that tie the sources to the project are *inferred*, two of them by checking a source's
  equations against each other. They are *checked numerically or symbolically* by `I200-mac/identity_check.py`
  (mpmath and sympy, exit 0), and its mutation makes all six fail (exit 1).
- **Numbers.** γ∞ = 1/2 to about 5×10⁻⁷ and λ = 2 + 3.0324/n are numerical evidence from the double-double γ (README,
  "The formula"). That λ = 2 is a growth rate of the D = ∞ critical solution is *conditional* on Conjecture 1
  (Proposition 5(iii) and its corollary). No number from the selection rule itself is used.
- **Verdict classes**, fixed before the search: a shared mathematical structure, named with the equations on both
  sides; an analogy only; nothing found in the sources searched.

### Closest works

| Work | Bears on | What it contains | What it does not contain |
|---|---|---|---|
| **M. Mariño, T. Reis**, *Resurgence for superconductors*, J. Stat. Mech. (2019), arXiv:1905.09569; and *Three roads to the energy gap*, PRB 106, 125142 (2022), arXiv:2010.16174 (abstract) | Q1 | The ground-state energy's series in the coupling γ is "factorially divergent and non-Borel summable", with its first Borel singularity at the strength A of Δ² ~ e^{−A/γ} (Eq. 1.1). In the Gaudin–Yang model c_k ~ −(1/π)(π²)^{−k+1}Γ(k−1) (Eq. 4.6), "precisely twice the strength of the exponentially small term in the energy gap"; the energy is π²/12 − γ/2 − 2π²e^{−π²/γ} (Eq. 2.45). "the Borel singularity is of the renormalon type", from ring diagrams and the logarithm of the Lindhard function, and they argue it is not instanton-driven. A trans-series with C₁ = ±i (Eq. 4.18), which "plays the rôle of a Stokes parameter". The 2022 paper gets the gap from the Bethe ansatz, from resurgence and from the RG: "The three methods lead to the same answer." | A parameter selected by a Stokes multiplier; the critical temperature; the Thouless criterion. |
| **J. Polchinski**, *Effective field theory and the Fermi surface*, TASI 92, hep-th/9210046 | Q1, Q2 | E∂_E V = NV² + O(V³) (Eq. 28), V(E) = V/(1 + NV ln(E₀/E)) (Eq. 29), which blows up at E_c = E₁e^{−1/(λ−μ*)} (Eq. 32). "The gap and the critical temperature are indeed of the form (32)", but the RG "does not get the O(1) coefficient". The effective potential "sums up the 'cactus' graphs, the same as in large-N O(N) models and mean field theory". | An exponentially small selection condition. |
| **A. Larkin, A. Varlamov**, *Fluctuation phenomena in superconductors*, cond-mat/0109177 (part); **F. Marsiglio, K. S. D. Beach, R. J. Gooding**, Can. J. Phys. 90, 889 (2012), arXiv:1203.5122 | Q3, Q2 | Fluctuation Cooper pairs live τ_GL = πħ/[8k_B(T − T_c)] (Eq. 2). The TDGL propagator is L(p, Ω) = (−iγ_GLΩ + ε_p)^{−1} (Eq. 106) with ε_p = αT_c(ε + ξ²p²) (Eq. 108); ν = 1/2 in the GL region, 1/2 + ε/10 near d = 4 (ε = 4 − d) and 3/5 at d = 3. Thouless "tracked the BCS instability from above (in temperature) by monitoring the two particle propagator", g₂ = χ₀/(1 − \|U\|χ₀) (Eq. 1): "a single pole passes from the lower half-plane to the upper-half-plane in complex frequency". | A count of unstable modes beyond the first crossing; the microscopic TDGL (Larkin–Varlamov Sec. 5, not reached). Thouless (1960) itself was not read. |
| **T. Koike, T. Hara, S. Adachi**, PRL 74, 5170 (1995); **T. Hara, T. Koike, S. Adachi**, gr-qc/9607010 (part); **D. Maison**, PLB 366, 82 (1996) | Q3, Q5, Q6c | The critical solution as an RG fixed point, and for DSS "replacing “fixed point” with “limit cycle”". β_BH = β/κ with κ "the unique positive eigenvalue"; "if U∗ has a unique relevant mode, then (S1) and (U1) hold". For the radiation fluid κ ≃ 2.81055255 and β ≃ 0.35580192, with "no other relevant (or marginal) eigenmodes in the range 0≤Re κ≤15, \|Im κ\|≤14" (method not stated). Maison: "the critical index depends rather strongly on the value of k, hence it is not universal", 0.1143 to 0.8157 for k = 0.01 to 0.888, uniqueness of the mode assumed. | Mean field, Landau theory, BCS, viscosity; a symmetry behind the relevant mode. |
| **S. A. Hartnoll, C. P. Herzog, G. T. Horowitz**, PRL 101, 031601 (2008), arXiv:0803.3295 | Q2, Q4 | ⟨𝒪⟩ ∝ (1 − T/T_c)^{1/2} (Eqs. 11–12), "a square root behaviour that is typical of second order phase transitions". Continuous transitions in 2+1 dimensions "are only possible ... in the large N limit (i.e. the classical gravity limit of our model), where fluctuations are suppressed." | A zero-mode criterion for T_c, which they read off the nonlinear solutions. |
| **R. Emparan, K. Tanabe**, *Holographic superconductivity in the large D expansion*, JHEP 01 (2014) 145, arXiv:1312.1108 | Q4 | Probe limit, n = D − 1, m̂ = m/n, 𝖱 = (r/r₀)ⁿ. A hypergeometric near zone (Eq. 2.27) and a Bessel far zone (Eq. 2.30), matched to Ai(−δμ̂) = 0 (Eq. 2.39), a zero-energy bound state in a triangular well. T_c in closed form with the term a₁(λ/2 − n/4)^{1/3} (Eq. 2.42), "better than 15%" at n = 4. Exponentially small terms in n are dropped; "we cannot determine the critical exponent at this level of approximation". | A beyond-all-orders effect; the nonlinear condensate. |
| **R. Emparan, C. P. Herzog**, RMP 92, 045005 (2020), arXiv:2003.11394, Secs. I, III.6, IV.4.5, V, VII.2, X.1–X.3 | Q2, Q4, Q6b | Large D as "mean field-like": "when the coordination number of a lattice diverges, one recovers a mean field theory". X.1: holographic "hair condenses just outside the near-horizon zone", so the large-D gains are "meager". X.3, unread before: γ is "well approximated by" ½(1 − 1/D), or ½(1 − 1/(D+1)) at larger D, from the data of Sorkin–Oren and Bland et al.; Rozali–Way's solutions are "not self-similar, and it is not a collapse"; critical collapse calls for a large-D approach "going beyond the scope of the effective membrane theory"; the period's D dependence "is less clear". | Δ(D); a selection mechanism. |
| **D. Vollhardt**, *Why calculate in infinite dimensions?*, Jülich lecture notes (2022), arXiv:2210.06072 (part); **A. Georges, G. Kotliar, W. Krauth, M. J. Rozenberg**, RMP 68, 13 (1996) (abstract) | Q2 | J = J*/Z (Eq. 11). The local field's fluctuation is a sum of Z terms over Z, so it falls like 1/√Z (Eq. 12), and "The Weiss mean-field theory is seen to become exact in the limit of infinite coordination number Z". Fermions need t = t*/√d (Eq. 27). DMFT "is exact in the limit of large lattice coordination (or infinite spatial dimensions)". | Critical exponents at d = ∞, in the parts read. |
| **A. LeClair, J. M. Román, G. Sierra**, PRB 69, 020505 (2004), cond-mat/0211338; **E. Braaten, H.-W. Hammer**, Phys. Rep. 428, 259 (2006) (part); **S. D. Glazek, K. G. Wilson**, PRL 89, 230401 (2002) (abstract) | Q5, Q1 | A reduced BCS model with a time-reversal-breaking coupling θ: dg/ds = g² + θ² (Eq. 15), g = θ tan[θs + tan⁻¹(g₀/θ)] (Eq. 16), period π/θ (Eq. 17), and a tower of gaps Δ_n = ω/sinh(t₀ + nπ/θ) (Eq. 10). θ → 0 gives Δ₀ ∼ 2ωe^{−1/g}, "recovering the standard BCS result"; the regime "reduces at one loop to the Kosterlitz-Thouless flow". Braaten–Hammer: "one of the signatures of an RG limit cycle is discrete scale invariance"; Efimov's s₀ ≈ 1.00624 from Eq. (35), "so eπ/s0≈22.7"; LeClair et al.'s "gap equation ... has multiple solutions related by a discrete scaling symmetry". | Gravity; a period fixed by a nonlinear selection. |
| **D. B. Kaplan, J.-W. Lee, D. T. Son, M. A. Stephanov**, *Conformality lost*, PRD 80, 125005 (2009); **B. Kol**, JHEP 10 (2006) 017, hep-th/0502033v3 | Q5, Q2 | Fixed points g± = g* ± √(α − α*) (Eq. 3) merge and turn complex: one side has Λ_IR/Λ_UV ≃ e^{−π/√(α*−α)} (Eq. 4), the other a limit cycle. The 1/r² potential's ν± = −(d−2)/2 ± √(α − α*) (Eq. 12) "can explain the Efimov effect", and the merger is tied to violating the BF bound. Kol: the double cone's perturbation ε = ρ^{s±}, s± = ((D−2)/2)(−1 ± i√(8/(D−2) − 1)) (Eq. 2.5), real for D ≥ 10; −1/s = γ(1 ± 2πi/Δ) (Eq. 3.16) gives 2π/Δ = √((10−D)/(D−2)) (Eq. 3.20) and γ = ¼(1 + √((D−10)/(D−2))) (Eq. 3.21). | Kaplan et al.: BCS, critical collapse. Kol: the scalar field's Δ, which Sorkin–Oren found (3.20) does not estimate. |
| **J. Camps, R. Emparan, N. Haddad**, JHEP 05 (2010) 042; **R. Emparan, K. Izumi, R. Luna, R. Suzuki, K. Tanabe**, JHEP 06 (2016) 117 (part); **R. Emparan, J. Rafecas-Ventosa, B. Way**, JHEP 04 (2026) 034 (part); **P. Kovtun, D. T. Son, A. O. Starinets**, PRL 94, 111601 (2005) | Q6b | η = s/4π, ζ = 2η(1/p − c_s²), c_s² = −1/(n+1) (Eqs. 1.3–1.4); "Sound waves in the effective fluid are unstable", and at large n Ω̃ = k̃(1 − k̃/(4πT)), "the exact limiting relation valid for all wavenumbers". The large-D brane is a compressible viscous fluid with η = ρ and s = 4πρ (Eqs. 2.18–2.21); it "reconstructs only a region of radial extent ∼1/D above the horizon", with vacuum and gauge charges only. The general equations "form a parabolic system with a well-posed initial value problem", and "Time-dependent horizon fluctuations with amplitudes of O(D⁰) lie outside the regime of applicability". η/s = ħ/4πk_B for theories with gravity duals, from η = σ_abs(0)/16πG (Eq. 7). | Horizon formation, collapse, scalar matter, critical phenomena. |
| **M. Rozali, B. Way**, JHEP 11 (2018) 106, from Sec. 3 to the end (unread before) | Q6b, Q2 | "While black hole collapse does not occur in the large D limit"; "this horizonless large D limit can be joined with the well-developed large D limit of black holes", proposed, not done; their time scaling "has not yet appeared in any large D study of black holes"; "The metric we have chosen cannot have the continuous or discrete self-similarity"; "These equations are parabolic in character". | A construction of the critical solution. |
| **I. Amado et al.**, *Holographic superfluids and the Landau criterion*, JHEP 02 (2014) 063; **A. Schmitt**, *Introduction to superfluidity*, Lect. Notes Phys. 888 (2015) (part) | Q6a | The Landau criterion v_c = min_p ε_p/p (Schmitt, Eq. 3; Amado et al., Eq. 2) is an energetic threshold. In holographic superfluids the dynamical instability, a QNM leaving the lower half plane, appears beyond it and is "strongest at finite wavelength"; the QNMs come from "the determinant method". Only the superfluid component flows through a thin tube, and "only the normal fluid carries entropy". | A threshold with one unstable mode at a finite rate; gravitational collapse. |
| **C. R. Evans, J. S. Coleman**, PRL 72, 1782 (1994); **C. Gundlach, D. Hilditch, J. M. Martín-García**, arXiv:2507.07636, Secs. II.5–II.7; **Z.-Y. Yuwen, C. Joana, S.-J. Wang, R.-G. Cai**, PRD 114, L081309 (2026) (part); Rein–Rendall–Schaeffer, PRD 58, 044007 (1998), and Olabarrieta–Choptuik, PRD 65, 024007 (2002) (abstracts) | Q6c | Evans–Coleman: a perfect fluid, with n "an eigenvalue of the nonlinear system" and β = 0.36. The review: dimensionful parameters "become irrelevant", dimensionless ones such as k "do change the universality class", and "a fixed point of the renormalisation group cannot have a preferred length scale". Yuwen et al.: Eckart bulk viscosity Π = −ϵρ^{1/2}Θ, "ϵ is the dimensionless viscosity strength"; fitted γ at w = 1/3 of 0.37 and 0.28 (two profiles) at ϵ = 0 and 0.34 and 0.26 at ϵ = 0.01 (Table 3), "the γ parameter remains at a similar order of magnitude"; shear neglected. Collisionless matter: "a discontinuity in the mass of the black hole", and "the critical behavior in this model is Type I". | A viscous critical solution, its modes or its exponent. |

Negative reads:
- Hartnoll–Horowitz–Kruthoff–Santos, SciPost Phys. 10, 009 (2021): inside the hairy black hole the amplitude
  c₅ = A sin(B/(1−T/T_c) + C) (Eq. 25), so the temperatures at which the Kasner inversions diverge accumulate at T_c
  like 1/n (*inferred*), not geometrically. No DSS or limit cycle.
- Abstracts or records: Müller–Schäfer (viscous matter falling into an existing black hole); Silveira, EPL 149, 49001
  (2025) (Jeans-type collapse with shear viscosity); Chavanis, EPJB 85, 229 (2012) (the Nyquist method for
  self-gravitating systems).
- Wikipedia, *Ginzburg criterion* (secondary): the upper critical dimension "comes out to be 4".

### The identities

*Inferred*, and *checked* by `I200-mac/identity_check.py`:
1. **Kol's merger is a fixed-point merger.** His Eq. (2.5) is the root pair of s² + (D−2)s + 2(D−2) = 0. That is
   Kaplan et al.'s Eq. (12) with d = D, α = −2(D−2) and α* = −(D−2)²/4, so α − α* = (D−2)(D−10)/4 and the merger sits
   at D* = 10. His Eqs. (3.19)–(3.21) follow from Eq. (3.16).
2. **Kol's merger also has γ → 1/2.** λ = −s₊ = 2 + 4/n + 16/n² + O(n⁻³), with n = D − 2. At D = ∞ the indicial
   equation, divided by n, becomes first order and gives s = −2. The 1/n coefficient is 4, against 3.0324 here.
3. **Emparan–Herzog's fits miss the computed γ above D = 20.** Against README's Values, ½(1 − 1/D) and
   ½(1 − 1/(D+1)) miss by at least 1.3×10⁻³ at D = 20, 30, 40, 50 and 60. Their large-D deficit 1/(2D) is 0.66 of the
   computed 0.758/(D − 2).
4. **The Russian-doll flow is a limit cycle.** Integrating dg/ds = g² + θ² gives poles spaced by π/θ; Eq. (16)
   solves Eq. (15); the gap ratio tends to e^{−π/θ}; and θ → 0 gives Δ₀ → 2ωe^{−1/g}.
5. **The superconducting rate passes through zero at T_c.** The pole of Larkin–Varlamov's L(0, Ω) lies at
   Ω = −i(8/π)(T − T_c), the inverse of their τ_GL. Below T_c the p = 0 mode grows at (8/π)(T_c − T).
6. **The gravity-free inner problem is the massless φ⁴ gradient flow.** With c = 1 the inner equations of Proposition
   5(i) give Y_s = Y_tt − Y³/4 = −δF/δY, with F = ∫(Y_t²/2 + Y⁴/16) dt and s the flow time. That is the real
   Ginzburg–Landau equation at T = T_c in one "space" dimension, t. Under (t, s, Y) → (at, a²s, Y/a) a mass term −rY
   needs r → r/a², the weight 2 of the s-shift. With gravity, c_s = cY²/4 couples c back in.

### Verdicts

**Q1, results beyond all orders: an analogy only.** The two share resurgence as a language, which the project already
uses, and differ in mechanism.
- **What both share.** The perturbative series cannot see the answer. The BCS energy's series is that of the normal
  state, and the condensation energy −2π²e^{−π²/γ} lies beyond it (Mariño–Reis, Eq. 2.45; *inferred* reading). Here
  every period has a formal 1/D solution (Proposition 1).
- **What lies beyond all orders.** In BCS it is the answer itself, the gap. Here the answer Δ is O(1) and, if
  analytic in 1/D, has an ordinary expansion. The exponentially small object is the condition that selects it, the
  Stokes multiplier of a fast mode (Propositions 2–4).
- **Where the exponential comes from.** BCS's e^{−1/g} comes from the logarithmic flow of a marginal coupling, which
  reaches a pole after ln(E₀/E) = 1/g (Polchinski, Eq. 29), and its Borel singularity is "of the renormalon type". The
  light-cone obstruction is a WKB exponent nΦ(x) of a fast mode (Proposition 3), an instanton-like exponent
  (*inferred* classification), the kind Mariño–Reis argue against for BCS.
- **The methods.** Of the three roads to the gap (Bethe ansatz, resurgence, RG), only resurgence has a counterpart
  here: the late orders of the 1/D series encode C(n,s) as a Stokes multiplier (Propositions 3–4). The gap equation
  is a self-consistency for a scale with a closed-form solution (Δ = ω/sinh(1/g) at θ = 0 in LeClair et al., Eqs.
  10–11), while the selection condition is a connection problem with no closed form. The Cooper instability is a pole
  at an exponentially small binding energy, not a zero of a multiplier that fixes an O(1) parameter.
- No source read gives a tool for Conjecture 1 beyond the Borel summation and resurgence that PROOF.md and
  proof/B3_borel.md already take from exponential asymptotics.

**Q2, mean-field exponents: an analogy only, through a weight 2 under parabolic scaling. Within gravity, a candidate
shared structure with Kol's merger.**
- **The common weight.** Both exponents are the inverse of a weight 2. In Ginzburg–Landau the control parameter adds
  to the squared wavenumber, ε_p = αT_c(ε + ξ²p²) (Larkin–Varlamov, Eq. 108), so the correlation length goes like
  ε^{−1/2}. In Proposition 5 the unstable mode is the shift of s = r²/(2n), so r_BH² ∝ p − p*. Identity 6 makes the
  parallel exact for the inner problem without gravity, the massless φ⁴ flow, where a mass and the s-shift both have
  weight 2.
- **Different mechanisms.** The mean-field 1/2 holds for every d above 4 and fails below it by the Ginzburg criterion
  (Wikipedia; Larkin–Varlamov give ν = 1/2 + ε/10 with ε = 4 − d, and 3/5 at d = 3). γ(D) moves at every finite D,
  γ = 1/2 − 0.758/n + … (numerical evidence), with no upper critical dimension on 3.04 ≤ D ≤ 61. Its 1/2 comes from an
  exact symmetry of the D = ∞ equations, broken at O(1/n), and is a growth rate of the critical solution only given
  Conjecture 1 (*conditional*).
- **Not a saddle-point limit.** Mean-field exactness at infinite coordination is a central-limit cancellation of the
  local field's fluctuations (Vollhardt, Eq. 12). HHH's square root comes from the classical bulk, "where fluctuations
  are suppressed". Critical collapse is classical at every D, yet γ(4) = 0.374. Emparan–Herzog call large D "mean
  field-like" by the coordination analogy, but the inner limit comes from the dominance of the radial term (*inferred*),
  which makes the equations "parabolic in character" (Rozali–Way; Proposition 5(i)).
- **Within gravity.** Kol's merger has γ → 1/2 by the same first-order limit, s² + ns + 2n = 0 tending to s = −2
  (identity 2), with a different 1/n coefficient, 4 against 3.0324. No source says whether its −2 is the weight of a
  symmetry, as Proposition 5's 2 is. Recommended task 1 tests it.

**Q3, the threshold as one unstable mode: an analogy only. The two thresholds are different kinds of instability.**
- **The superconducting threshold** is a bifurcation of the normal state. "a single pole passes from the lower
  half-plane to the upper-half-plane" (Marsiglio et al.), and its rate passes through zero, (8/π)(T − T_c) (identity 5),
  so τ_GL diverges at T_c. Below T_c the unstable modes form a band in p, wherever ε + ξ²p² < 0, most unstable at p = 0
  (*inferred* from Eq. 108).
- **The collapse threshold** is the codimension-one stable manifold of a saddle that has one unstable mode at a finite
  rate (Koike–Hara–Adachi; Hara–Koike–Adachi, Claim 1). p − p* sets the mode's amplitude, not its rate, and γ = 1/λ.
  The statistical-mechanics counterpart is an RG fixed point's single relevant direction, the reading those papers
  use, not the Thouless instability.
- **Counting.** The Thouless criterion detects the first crossing by a real sign change at one point,
  1 − |U|χ₀(0, 0) (Marsiglio et al., Eq. 1). It does not count. In critical collapse, Koike–Hara–Adachi report no
  further mode in 0 ≤ Re κ ≤ 15, |Im κ| ≤ 14 without stating the method, and Maison assumes uniqueness. Holographic
  QNM studies use "the determinant method" (Amado et al.) without a winding count. MODE_COUNT.md's argument-principle
  count of det M(λ) has two precedents in critical collapse. Hirschmann and Eardley count the zeros of their shooting
  determinant around a rectangle for the complex-scalar CSS solution (PRD 52, 5850 (1995), Secs. VI–VII), and Gundlach
  counts the zeros minus the poles of his for the D = 4 Choptuik solution (PRD 55, 695 (1997), App. E); the reading is
  the addendum of issue #619.
- **The rate near threshold** vanishes at a bifurcation (critical slowing down) and stays finite at a saddle, λ = 2 +
  O(1/n) here. #198's cavity saddle (Gomila et al.) remains the closer analogue.

**Q4, holographic superconductors at large D: an analogy only. Their threshold is visible order by order in 1/n.**
- **The method.** Emparan–Tanabe find T_c by near–far matching of a linear problem, a hypergeometric near zone joined
  to a Bessel far zone at a turning point, Ai(−δμ̂) = 0 (Eq. 2.39). The threshold has a 1/n expansion, with powers
  n^{1/3} from the Airy zero (Eq. 2.42), and the exponentially small terms are dropped. That is the opposite of
  Proposition 1, where the period is invisible at every order.
- **Tools already in the project.** Proposition 2 is also a hypergeometric connection problem, at the light cone,
  solved by Gauss's formula (*inferred* comparison). The holographic condensate itself is left to numerics, since the
  hair "condenses just outside the near-horizon zone" (Emparan–Herzog X.1), and the critical exponent is not obtained at
  large D (Emparan–Tanabe).
- **One difference worth a test.** A turning point makes fractional powers of n. The project's analyticity tests added
  D^{−1/2}, D^{−3/2}, ln D/D, ln D/D² and e^{−D/4} terms and series in n^{−1/2} (README, "Is the period analytic in
  1/D?"), not n^{−1/3} or n^{−2/3}. Recommended task 2.

**Q5, discrete scale invariance: found in superconductivity as an RG limit cycle. A shared structure with Kol's merger,
not with the Choptuik solution.**
- **In superconductivity.** The Russian-doll BCS model has an RG limit cycle, dg/ds = g² + θ², with period π/θ and
  gaps in the ratio e^{−π/θ} (LeClair et al.; identity 4). Its mechanism is that of conformality lost, which explains
  the Efimov effect: a pair of fixed points pushed into the complex plane, g = ±iθ here and g* ± √(α − α*) in Kaplan et
  al. (*inferred* from the two beta functions, which have the same form with θ² = α* − α). As θ → 0
  the period diverges and BCS's single gap e^{−1/g} returns. *Inferred*: BCS's marginal flow dg/ds = g² is the merger
  point θ = 0 of that family.
- **Kol's double cone** has exactly this structure, with D in the role of α. Identity 1 maps his Eq. (2.5) onto
  Kaplan et al.'s Eq. (12), α − α* = (D − 2)(D − 10)/4, and his 2π/Δ = √((10 − D)/(D − 2)) is the cycle's period. This
  is a shared mathematical structure, named with the equations on both sides, but it belongs to the linear
  perturbations of the merger system.
- **The Choptuik solution** is a limit cycle of the self-similarity RG (Hara–Koike–Adachi; the 2025 review, II.5), but
  its period is the eigenvalue of a nonlinear problem, as Evans–Coleman's n is "an eigenvalue of the nonlinear system",
  and at large D it is selected beyond all orders (Proposition 1). No complex pair of fixed points is known for it. A
  merger would make Δ diverge like (D − D*)^{−1/2}. The computed Δ(D) is finite on 3.04 ≤ D ≤ 78, with its maximum at
  D = 3.755725 (README), so no merger of that kind occurs in the computed range (*inferred*).
- The superconductivity sources read do not use Efimov physics. One title joins them (*Efimov trimers and Cooper
  pairs*, NJP 21, 073051 (2019), arXiv:1904.01939), not opened.
- **Follow-up.** For the Choptuik solution, none. Recommended task 1 rederives identity 1 from Kol's ansatz.

**Q6a, superfluidity: nothing found in the sources searched beyond the analogy of a threshold.**
- The Landau criterion is an energetic threshold, ε_p + p·v_s < 0 (Schmitt, Eqs. 2–3). In holographic superfluids the
  dynamical instability follows beyond it, at finite wavelength (Amado et al.).
- Zero viscosity belongs to the superfluid component, while the normal component carries the entropy and a finite shear
  viscosity (Schmitt; Kovtun et al.).
- Neither the onset of superfluidity nor the critical velocity is a codimension-one saddle with one unstable mode, and
  no source read ties either to critical collapse.

**Q6b, large-D horizons as viscous fluids: by its own stated limits the effective theory cannot describe the
threshold. An analogy only for the black hole past it.**
- **Scope.** The theory needs an existing horizon, a near zone of thickness 1/D, horizon gradients of order 1/√D and
  fluctuations of amplitude O(1/D): "Time-dependent horizon fluctuations with amplitudes of O(D⁰) lie outside the regime
  of applicability" (Emparan–Rafecas-Ventosa–Way). Emparan–Herzog X.3 place critical collapse "beyond the scope of the
  effective membrane theory", and Rozali–Way leave the join of the horizonless and black-hole limits open.
- **The threshold.** The critical solution has no horizon, and past the threshold the horizon grows from zero radius,
  an O(1) change (*inferred*). The scalar field that forms and feeds it is outside the theory, which carries vacuum and
  gauge charges only (Emparan et al. 2016).
- **Where viscosity sets a large-D threshold.** It sets the Gregory–Laflamme band edge, from Ω̃ = k̃(1 − k̃/(4πT)) with
  η/s = 1/4π (Camps–Emparan–Haddad), k_GL = 1 − 1/(2n) + … in Emparan–Herzog Sec. V. That threshold is a mode crossing
  zero at k_GL, a bifurcation like the superconducting one of Q3, not a saddle with one unstable mode.
- **Parabolic equations on both sides.** The membrane equations are parabolic in real time (Emparan–Rafecas-Ventosa–
  Way), and the inner problem is parabolic in s = r²/(2n) (Proposition 5(i)). No source maps one onto the other.

**Q6c, viscosity and the perfect-fluid critical solution: nothing found in the sources searched.**
- No source read constructs a viscous critical solution or its modes. INSPIRE's 25 records on viscous or dissipative
  gravitational collapse are, by title, dissipative-collapse models and Jeans-type clouds, not thresholds.
- The one threshold study with viscosity, Yuwen et al. (2026, primordial black holes, Eckart bulk viscosity), fits γ
  with and without it. Its two profiles give 0.37 and 0.28 at ϵ = 0 and w = 1/3, where universality and
  Koike–Hara–Adachi's 0.3558 predict one value. So its fits cannot resolve a change of γ by ϵ = 0.01 (*inferred*).
- **The review's rule sorts the cases** (*inferred* application of Gundlach–Hilditch–Martín-García, II.6–II.7). Take
  a viscosity coefficient ∝ ρ^q. On a self-similar background of scale L, ρ ∝ L^{−2} (G = c = 1), so the viscous stress
  over the pressure scales like ϵL^{1−2q}.
  - q = 1/2, Yuwen et al.'s choice, is scale-free. ϵ is then a dimensionless parameter like k, which can change the
    universality class, as k does over γ = 0.11–0.82 (Maison).
  - q < 1/2 is irrelevant on small scales and keeps the perfect-fluid class.
  - q > 1/2 grows on small scales and breaks self-similarity there. η = s/4π for a conformal fluid in four dimensions
    has q = 3/4 once ħ sets the entropy (*inferred*), so it would matter only near the scale where hydrodynamics itself
    fails.
- At the other end of the mean free path, collisionless matter is type I with a mass gap (Rein–Rendall–Schaeffer;
  Olabarrieta–Choptuik), so the class does change between zero and infinite mean free path. None of this bears on the
  scalar field, which has no viscosity.

### Citation chains (INSPIRE, 2026-11-09)

- **Forward from Emparan–Tanabe** (recid 1267488): 45 records. By title, only arXiv:1502.03616 and 2001.08358 continue
  holographic superconductors at large D. Not opened.
- **Forward from Rozali–Way** (recid 1684230): 13 records, #172's list plus Emparan–Rafecas-Ventosa–Way (opened).
- **Forward from Mariño–Reis** (recid 1736509): 38 records, none about gravity, selection or Stokes multipliers by
  title. *Three roads to the energy gap* was opened as an abstract.
- **Forward from Evans–Coleman** (recid 371808, 365 citations), restricted to viscous, dissipative, kinetic or
  collisionless titles: 2 records, both read.
- **Backward.** Braaten–Hammer cite Wilson (1971), Glazek–Wilson and LeClair et al. for limit cycles. LeClair et al.
  cite Bedaque–Hammer–van Kolck and Bernard–LeClair. Kol cites Sorkin–Oren for the test of his Eq. (3.20).

### References checked

| Entry | Checked against | Verdict |
|---|---|---|
| Mariño, Reis, *Resurgence for superconductors*, J. Stat. Mech. (2019), doi:10.1088/1742-5468/ab4802 | the arXiv page (related DOI) and INSPIRE recid 1736509 (DOI, no journal fields); the article number 123102 from a search summary only | PARTIAL: article number *(unverified)* |
| Mariño, Reis, PRB 106, 125142 (2022), arXiv:2010.16174 | the arXiv page (related DOI 10.1103/PhysRevB.106.125142) | OK |
| Polchinski, TASI 92, pp. 235–276, hep-th/9210046 | INSPIRE recid 339181 | OK |
| Larkin, Varlamov, cond-mat/0109177, a chapter of *Handbook on Superconductivity: Conventional and Unconventional Superconductors* (Springer, 2002) | the arXiv comments field, where the first editor's name is a broken link; no INSPIRE record | PARTIAL: editors and pages *(unverified)* |
| Marsiglio, Beach, Gooding, Can. J. Phys. 90, 889 (2012), arXiv:1203.5122 | the arXiv journal-ref; the page's related DOI points to an IEEE record, so no DOI is given | PARTIAL |
| Koike, Hara, Adachi, PRL 74, 5170–5173 (1995), doi:10.1103/PhysRevLett.74.5170; Hara, Koike, Adachi, gr-qc/9607010 | INSPIRE recids 393236 and 420411 (the second without a journal) | OK; the second as a preprint |
| Maison, PLB 366, 82–84 (1996), doi:10.1016/0370-2693(95)01381-4; Evans, Coleman, PRL 72, 1782–1785 (1994), doi:10.1103/PhysRevLett.72.1782 | INSPIRE recids 394036 and 371808 | OK |
| Hartnoll, Herzog, Horowitz, PRL 101, 031601 (2008); Emparan, Tanabe, JHEP 01 (2014) 145; Hartnoll, Horowitz, Kruthoff, Santos, SciPost Phys. 10, 009 (2021) | INSPIRE recids 781888, 1267488 and 1814043 (DOIs) | OK |
| Vollhardt, in *Dynamical Mean-Field Theory of Correlated Electrons*, Modeling and Simulation vol. 12 (Forschungszentrum Jülich, 2022), arXiv:2210.06072 | INSPIRE recid 2164807 | OK |
| Georges, Kotliar, Krauth, Rozenberg, RMP 68, 13–125 (1996), doi:10.1103/RevModPhys.68.13 | INSPIRE recid 439061, whose published title, *Dynamical mean-field theory of strongly correlated fermion systems and the limit of infinite dimensions*, differs from the arXiv title | OK |
| LeClair, Román, Sierra, PRB 69, 020505 (2004); Braaten, Hammer, Phys. Rep. 428, 259–390 (2006), doi:10.1016/j.physrep.2006.03.001; Glazek, Wilson, PRL 89, 230401 (2002), erratum PRL 92, 139901 (2004) | INSPIRE recids 602538, 662213 and 583936; the arXiv pages of the last two | OK |
| Kaplan, Lee, Son, Stephanov, PRD 80, 125005 (2009), doi:10.1103/PhysRevD.80.125005; Kol, JHEP 10 (2006) 017, doi:10.1088/1126-6708/2006/10/017 | INSPIRE recids 821642 and 676109 | OK |
| Camps, Emparan, Haddad, JHEP 05 (2010) 042; Emparan, Izumi, Luna, Suzuki, Tanabe, JHEP 06 (2016) 117; Emparan, Rafecas-Ventosa, Way, JHEP 04 (2026) 034 | INSPIRE recids 849250, 1422195 and 3093279 (DOIs) | OK |
| Kovtun, Son, Starinets, PRL 94, 111601 (2005), doi:10.1103/PhysRevLett.94.111601; Amado et al., JHEP 02 (2014) 063; Schmitt, Lect. Notes Phys. 888, 1–155 (2015), doi:10.1007/978-3-319-07947-9 | INSPIRE recids 651094, 1245296 and 1288720 | OK |
| Yuwen, Joana, Wang, Cai, PRD 114, L081309 (2026), doi:10.1103/x8p6-4zwv; Rein, Rendall, Schaeffer, PRD 58, 044007 (1998); Olabarrieta, Choptuik, PRD 65, 024007 (2002) | INSPIRE recids 3173008, 469333 and 560433, and the arXiv pages | OK |
| Müller, Schäfer, EPJA 54, 30 (2018); Silveira, EPL 149, 49001 (2025), doi:10.1209/0295-5075/adae2f; Chavanis, EPJB 85, 229 (2012) | INSPIRE recids 1628075 and 2896111; the arXiv page of 1002.0291 | OK |
| Cooper, PR 104, 1189–1190 (1956); Bardeen, Cooper, Schrieffer, PR 108, 1175–1204 (1957); Thouless, Ann. Phys. 10, 553 (1960); Metzner, Vollhardt (1989); Ginzburg (1961) | search listings and the citing sources (Marsiglio et al. for Thouless; Wikipedia for Ginzburg) | UNVERIFIABLE here, cited as not read |

### Gap statement

- **Not read.** Cooper (1956), BCS (1957), Thouless (1960), Metzner–Vollhardt (1989), Ginzburg (1961) and a textbook
  account of Landau theory. The large-D holographic-superconductor follow-ups (arXiv:1502.03616, 2001.08358). The
  Efimov-and-Cooper-pairs paper (arXiv:1904.01939). Penrose's criterion (1960), which counts unstable plasma modes by
  the argument principle and surfaced only through lecture notes.
- **Partly read.** Larkin–Varlamov Sec. 5, Hara–Koike–Adachi's DSS section, Braaten–Hammer Secs. 5–8, Beekman et al.'s
  Chapter 5, and Yuwen et al. and the 2025 review beyond character 100000.
- **Channels.** INSPIRE misses condensed-matter journals, so the forward chains of Larkin–Varlamov, Marsiglio et al.
  and Schmitt were not run. There were no Google Scholar queries and no per-author sweeps.
- **Rounds.** Round 1 covered the six questions. Round 2 followed the INSPIRE chains and the mode-count lead, and added
  Marsiglio et al., Emparan–Rafecas-Ventosa–Way and the "three roads", so the search is not dry.

### Discrepancies and proposed text (for the coordinator)

- **Emparan–Herzog's fit of γ(D).** Their Sec. X.3 fits the D ≤ 14 values of Sorkin–Oren and Bland et al. by
  γ ≈ ½(1 − 1/D), or ½(1 − 1/(D+1)) at larger D. Against the double-double values both miss by 1.3×10⁻³ to 3.1×10⁻³ at
  D = 20–60, and their large-D deficit 1/(2D) is 0.66 of the computed asymptotic 0.758/(D − 2) (identity 3). This is an
  approximate published fit that the computed values leave by more than 10⁻³ from D = 20 on, not a disagreement
  with the repository's text. Proposed text for
  the README's "Earlier formulas", after the Sorkin–Oren sentence: "Emparan and Herzog's review fits the published
  values by γ ≈ ½(1 − 1/D) [RMP 92, 045005 (2020), Sec. X.3]; at D = 20–60 it misses the values here by 1.3–3.1×10⁻³,
  and its large-D deficit, 1/(2D), is two-thirds of the computed 0.758/(D − 2)." Its check is
  `I200-mac/identity_check.py`, item 3.
- **FULL-TEXT-UNREAD marks.** §1 and §6 mark Emparan–Herzog Sec. X.C and Rozali–Way beyond Sec. 2.3 as unread. Both are
  now read (X.C is Sec. X.3 in the arXiv numbering), and neither changes a grade. Emparan–Herzog stay P2, with no
  Stokes reading, no selection and no Δ(D). Rozali–Way stay P4 for C4 and C5: "The metric we have chosen cannot have
  the continuous or discrete self-similarity".
- **Yuwen et al.'s γ** at ϵ = 0 depends on the initial profile (0.37 and 0.28 at w = 1/3), where the radiation-fluid
  exponent is 0.3558 (Koike–Hara–Adachi; Maison). This is an inconsistency inside a source, recorded because it limits
  the Q6c verdict.
- **No novelty sentence** was added by this task. The strongest statement made is "not found in the sources searched".

### Recommended tasks (for the coordinator to file)

1. **Kol's merger at D = ∞: is its λ = 2 the weight of a symmetry?**
   - *What.* From Kol's ansatz (his Eqs. (2.3)–(2.4)), derive the double cone's perturbation equation in sympy, take
     n = D − 2 → ∞, and test whether the leading mode ε ∝ ρ^{−2} is generated by a symmetry of the D = ∞ equations, as
     Proposition 5's mode is by the shift of r².
   - *Why.* If it is, two gravitational thresholds share the mechanism behind γ∞ = 1/2 (identity 2), which bears on
     how robust Proposition 5's corollary is. If not, their common 2 is only the first-order limit of an indicial
     equation.
   - *Acceptance.* The derived indicial equation must be s² + (D−2)s + 2(D−2) = 0, reproducing Kol's Eq. (2.5) at
     three finite D to 10⁻²⁰ and so identity 1 from the ansatz rather than from his result; if not, the task stops. Then
     a yes or no on the symmetry, with the generator written out if yes, and the expansion λ = 2 + 4/n + 16/n² recovered
     from the derived equation.
   - *Depends on.* #200. Minutes on one core.
2. **Turning-point powers in the analyticity tests of Δ(D) and γ(D).**
   - *What.* Add n^{−1/3} and n^{−2/3} terms, one at a time, to the 1/D fits of README, "Is the period analytic in
     1/D?", through the planted-truth path first (`hp/planted_dinf.py`), then `hp/dd_fits_mp.py --mixed`.
   - *Why.* Emparan–Tanabe's large-D threshold carries n^{1/3} from an Airy zero at a turning point (their Eq. 2.42),
     and the tests so far tried other non-analytic terms only.
   - *Acceptance.* On planted data with an n^{−2/3} term of the size the real fits allow, the fit recovers its
     coefficient within 30% and shifts Δ∞ by the planted amount; if not, the real-data verdict is void. Then report the
     real-data coefficient and the shift of Δ∞, with the spread over fit choices named as such.
   - *Depends on.* #200. Under an hour on one core.
3. **Read the primary sources that did not open.**
   - *What.* Cooper (1956), BCS (1957), Thouless (1960), Metzner–Vollhardt (1989) and Penrose (1960), logged in §6.
   - *Why.* Q1 and Q3 rest on later accounts of the first four (Mariño–Reis, Polchinski, Marsiglio et al.). Penrose's
     argument-principle count, once read, would be the citation for MODE_COUNT.md's winding method.
   - *Acceptance.* Each is read in full and recorded. The task fails, and says so, if one of them changes a verdict
     above.
   - *Depends on.* #200.

## Addendum (2026-11-09, issue #221): Pollnau and Eichhorn's resonator work

**Question.** #198 found that Eq. (16) of EEG26b is, at δ = 0, a cavity's Airy distribution times cos θ, and that its
hump threshold q_c = 3 − 2√2 is the point where Pollnau and Eichhorn's Airy linewidth breaks down, their Eq. (118).
Issue #221 asks what else their line of work shares with this project:
- **Q1, line shapes and singularities.** Their split of the Airy distribution into Lorentzian pole pairs, Eqs.
  (99)–(108), and the breakdown at R₁R₂ = 17 − 12√2, against the profile's pole pairs, its decay rate as a pole
  distance and its hump threshold. Do they treat deformed line shapes, extra zeros or poles, or a branch point near a
  pole (#133, #216)?
- **Q2, thresholds.** A laser at threshold beside the critical solution's single unstable mode with γ = 1/λ
  (notes/MODE_COUNT.md) and #200's superconducting threshold.
- **Q3, method.** Their rederivation of the Schawlow–Townes linewidth from classical physics, beside this project's
  derivations.
- **Q4, their later work.** Part II and later parts, Ismail et al. (2016), their other resonator and threshold papers,
  and the works that cite them.

**How.**
- **Searches.** 11 web queries and 12 queries of the OpenAlex, Europe PMC, Unpaywall and INSPIRE APIs, all on
  2026-11-09, listed in §5.
- **Reads.** Pollnau–Eichhorn (2020) and Ismail et al. (2016) in full, Vujnović et al. (2026) through Sec. 6, and the
  records or abstracts of 10 more, listed in §6. Quotes and URLs are in the raw-data repository, folder `I221-mac/`,
  commit 4ecb5c2.
- **PDFs.** The two full texts came back from `WebFetch` as binary and were extracted page by page with macOS PDFKit.
  Ismail et al., which #198 could not open, is the published version deposited in KTH DiVA, found through Unpaywall.
- **Identities.** Six relations that tie the sources to Eq. (16) are *inferred* and *checked numerically* by
  `I221-mac/identity_check.py` (mpmath, 30 digits, exit 0). Its mutation makes all six fail (exit 1).
- **Numbers.** EEG26b's (q, δ) = (0.0551, 0.0985) come from their extrapolation of finite-D solutions. The tuned D = ∞
  profile's numbers (its first ratios, the [1/1] form of EEG_PROFILE.md §3.3) assume the selection rule, Conjecture 1
  of the math paper, and are marked *conditional*.
- **Verdict classes**, fixed before the search: a shared structure, named with the equations on both sides; an
  analogy only; nothing found in the sources searched.

### Closest works

| Work | Bears on | What it contains | What it does not contain |
|---|---|---|---|
| **M. Pollnau, M. Eichhorn**, Prog. Quantum Electron. 72, 100255 (2020), KITopen PDF, read in full | Q1–Q3 | Eq. (62): the Airy distribution integrated over one free spectral range is 1/(1 − R₁R₂e^{−2α ℓ}). Eq. (91): the reflected field (r₁ − r₂e^{−αℓ}e^{−2iϕ})/(1 − r₁r₂e^{−αℓ}e^{−2iϕ}), the direct reflection plus the cavity's leakage. Eqs. (99)–(108): the mode profile 2δ/(δ² + (2ϕ − 2πq)²) as a pole pair, summed with Σ 1/(x − q) = π cot πx to the Airy distribution, "over the intensity mode profiles ... rather than over the electric fields". Eqs. (113)–(119): Lorentzian and Airy finesse; Fig. 19, the Airy linewidth jumps to infinity "When the minimum of the Airy distribution crosses half its maximum". Eqs. (142)–(146): gain lowers the decay rate, 1/τ_L = 1/τ_c − cg, and Λ = 1/(1 − cgτ_c) diverges at threshold. Eq. (156): in a cw laser "the gain ... is smaller than the losses", by the spontaneous rate. Sec. 7.5: the four approximations that take their linewidth to Schawlow–Townes. | The word "pole" (the text never uses it), complex frequency, zeros beyond Eq. (94), branch points, a functional form of a frequency-dependent reflectivity, a growth law or build-up time above threshold. |
| **N. Ismail, C. C. Kores, D. Geskus, M. Pollnau**, Opt. Express 24, 16366 (2016), KTH DiVA, read in full | Q1 | Secs. 2–4: the lossless version of the above, with the cot sum in Appendix C. Sec. 5: reflectivities as "general functions of frequency, Ri(ν)", so τ_c(ν) becomes local, "thereby losing its meaning as the FWHM linewidth"; a numerical example with a multilayer mirror (98% at 300 THz, shown only in Fig. 9), where the nearest mode profile and the Airy distribution "become strongly asymmetric". The truncated mode sum converges like M⁻¹; per-mode Lorentzians with frozen widths leave an error that "would converge to a constant value". | A closed form for Ri(ν), complex frequency, poles, zeros, branch points. |
| **V. Vujnović, N. Kralj, M. Karuža**, Photonics 13, 721 (2026), Secs. 1–6 | Q1, Q4 | Cites Ismail et al. Poles of a cavity with internal reflectors as zeros of a reduced denominator, Eqs. (23)–(25); a one-pole form with a "self-energy", Eq. (29); a weak reflector's first-order complex pole shift, δω̃ = i(c/2L)ζ, whose imaginary and real parts are the frequency pulling and the linewidth change, Eqs. (39)–(41); the three-mirror denominator, Eq. (54); pole tracking by continuation, Eq. (56), which "applies to a simple pole". | Zeros of the response, branch points, non-geometric harmonic laws. Its Appendix D on non-simple poles was not read. |
| **Pollnau's other work since 2017** (60 OpenAlex records) and **Eichhorn's** (20) | Q4 | Counter-propagating modes ±q as independent modes (Opt. Lett. 43, 5033 (2018)); distributed phase shifts in distributed-feedback resonators (2018–2019); exact fields in Bragg mirrors by a recursive circulating-field method (2020); laser photon statistics with a beam-splitter output coupler (Eichhorn, Appl. Phys. B 128 (2022)). Abstracts only. | Part II, line-shape singularities, threshold laws. |

### The identities

*Inferred*, and *checked numerically* by `I221-mac/identity_check.py`; σ = ½ ln(1/q) and φ = θ + δ/2:
1. **Eq. (16) is the field-level Mittag-Leffler sum.** β̃ = (A*/2√q) Re[e^{−iδ/2} sec(φ + iσ)]. Expanded over the
   poles of sec, it is a sum over k of (−1)^{k+1}[cos(δ/2)(φ − c_k) − sin(δ/2)σ]/((φ − c_k)² + σ²), with
   c_k = (k + ½)π. So each pole pair is a Lorentzian of half-width σ, dispersive at δ = 0 and turned toward an
   absorptive one by the angle δ/2, and successive pairs alternate in sign. Pollnau–Eichhorn's Eq. (108) sums
   same-sign absorptive Lorentzians of the same half-width (their δ/2 is σ) through the cot series. The alternating
   series is the csc series, Σ(−1)^q/(x − q) = π/sin πx, and it describes the transmitted field itself:
   t₁t₂e^{−iϕ}/(1 − ρe^{−2iϕ}) = (t₁t₂/2i√ρ) Σ_q (−1)^q/(ϕ − πq − iσ). The coherent sum has the Airy shape as well,
   |Σ_q (−1)^q/(ϕ − πq − iσ)|² = (2σ/sinh 2σ) Σ_q 1/((ϕ − πq)² + σ²). So the cross terms between modes do not vanish
   but add the constant factor sinh(2σ)/2σ, 1 + O(σ²) at high finesse and 3.12 at EEG26b's q.
2. **The norm is the integrated Airy distribution.** ⟨β̃²⟩ = (A*²/2) A_circ,FSR with R₁R₂e^{−2αℓ} = q², which by their
   Eq. (62) is A*²/(2(1 − q²)), EEG26b's (S60), for every δ. EEG26b's A* = [2(1 − q²)𝒩₀]^{1/2} is this relation.
3. **Reading check of their Eqs. (114), (117) and (118).** Eq. (117) gives Δν_Airy/Δν_c = 1.001, 1.022, 1.132, 1.717 at
   R₁ = R₂ = 0.9, 0.6, 0.32, 0.172, the values of their Fig. 12, and the two thresholds are e^{−2π} = 0.001867
   (R = 4.32%) and 17 − 12√2 = 0.02944 (R = 17.2%).
4. **The hump threshold is the Airy breakdown only at δ = 0.** The Airy factor 1/D_q has min/max ((1 − q)/(1 + q))² for
   every δ, so it crosses ½, their Fig. 19 criterion, at q = 3 − 2√2 whatever δ is. The profile's second local maximum
   per half period appears where dβ̃/dθ has a double zero: at q_c = 3 − 2√2 for δ = 0, and at q_c = 0.21328, θ =
   −0.3565 (paper's gauge), for δ = 0.0985. The two criteria differ (half maximum against a turning point of cos θ
   times the Airy factor), and they meet at F = 1 only when δ = 0. EEG26b's q is 0.26 of q_c(0.0985).
5. **The one-zero [1/1] form is their two-path reflection, Eq. (91).** (1 + bu)/(1 + wu) = b/w + (1 − b/w)/(1 + wu), a
   direct term of weight b/w plus a round-trip train. Their reflection has b/w = 1/R₁ ≥ 1, the direct reflection
   dominating. The tuned profile's [1/1] (EEG_PROFILE.md §3.3: w = r₃, b = r₃ − r₁) has |b/w| = 0.0419 and
   arg(b/w) = −0.477 (*conditional*). So its zero is not a passive reflector's: in these terms the profile is a
   transmission train with a weak direct admixture, which is §3.3's "geometric tail ... plus a first ratio about 4%
   smaller" read as two paths.
6. **Gain moves the pole.** With the round-trip amplitude ρ_L = ρ_c e^{gℓ}, their spectral-coherence factor
   Λ = 1/(1 − cgτ_c) is σ_c/σ_L, the ratio of the passive and active pole distances, and their threshold, Λ → ∞, is
   ρ_L → 1, that is q → 1 in Eq. (16), where its poles reach the real axis.

At EEG26b's q the echo is a low-finesse cavity on both of their measures: F = 0.247 < 1, so the Airy linewidth is
undefined, and the Lorentzian finesse π/2σ = 1.084 is just above their Eq. (114) limit (#198).

### Verdicts

**Q1, line shapes and singularities: a shared structure for the undeformed line, the Mittag-Leffler expansion over one
pole pair per round-trip period. Nothing found for deformed lines, zeros or branch points.**
- **The shared structure.** Their Eqs. (99)–(101) and (107)–(108) on one side, and identity 1 on the other, are the
  same partial-fraction expansion over the lattice of longitudinal modes, at two levels. Theirs is the intensity, same
  sign, cot. Eq. (16) is a field, alternating, csc or sec, because the echo is antiperiodic (odd harmonics). The
  half-width of every term is σ on both sides, the decay rate per harmonic. The echo's δ, its time asymmetry, is the
  dispersive-to-absorptive mixing angle δ/2 of every term, which their absorptive Lorentzians do not have. Identity 2
  (the norm) and identity 4 (the threshold) are corollaries.
- **The threshold correspondence holds at δ = 0 only.** #198's statement, which EEG_PROFILE.md §2 makes at δ = 0, is
  right as stated. At EEG26b's δ the hump threshold moves to 0.21328 while the Airy breakdown stays at 0.17157
  (identity 4), so the coincidence is one of the δ = 0 family, not of the cavity.
- **Deformed line shapes.** Ismail et al., Sec. 5, the source #198 could not open, treats frequency-dependent
  reflectivity only numerically, with a local decay time τ_c(ν). It gives no form for Ri(ν) and no poles, zeros or
  branch points; Pollnau–Eichhorn's p. 24 repeats its conclusion. Propagation loss again only rescales the round-trip
  ratio, R₁R₂ → R₁R₂e^{−2αℓ}, so the line keeps Eq. (16)'s [0/1] form.
- **Extra zeros.** Their reflected field, Eq. (91), is the one-zero form, and identity 5 places the tuned profile
  outside the passive range (*conditional*). Extra poles come from internal reflectors (Vujnović et al., Eq. (54)), a
  rational function of higher degree, which #198 already matched to #133's [L/N] family. With meromorphic
  reflectivities such cavities stay meromorphic, so they cannot make the branch point of #133 and #216 (*inferred*).
- **A branch point near a pole.** None of the sources read has one. Vujnović et al.'s pole continuation, Eq. (56), is
  stated for simple poles, and its non-simple case (Appendix D) was not read.

**Q2, thresholds: an analogy only. The laser threshold is a bifurcation, like #200's superconducting threshold.**
- **The laser threshold** is the active mode's pole reaching the real axis (identity 6). Below it Λ is finite. Their
  cw laser never reaches it, since the spontaneous rate keeps the gain below the losses (Eq. (156)). Above it the
  mode grows, Λ < 0, a regime they call "temporally unsustainable" and treat only as transient (Sec. 7.3). Its rate
  passes through zero at threshold, as the superconducting rate does at T_c (#200, identity 5).
- **The collapse threshold** is the stable manifold of a saddle whose one unstable mode keeps a finite rate, λ = 2.08 at
  D = 30 (MODE_COUNT.md) and λ → 2 as D → ∞ (numerical evidence; a growth rate of the critical solution only given
  Conjecture 1). p − p* sets the mode's amplitude, not its rate. No source read gives the laser counterpart of
  γ = 1/λ, a build-up time ∝ ln(1/|p − p*|)/λ: Pollnau–Eichhorn have no build-up law, and #198's cavity saddle of
  Gomila et al. remains the closest optical analogue.
- **In Eq. (16)'s terms** the threshold would be q → 1. EEG26b's q = 0.055 is far below it, so nothing in the echo is
  near a threshold of this kind (*inferred*). The single unstable mode and the profile's pole pairs are different
  objects: one is a growth rate in τ, the other a singularity of the profile in complex θ.
- Their Sec. 6.2 recalls Blaquière's comparison of the maser with "a nuclear reactor in a subcritical, critical, or
  supercritical state". That shares the vocabulary of critical collapse, and the source was not read.

**Q3, method: an analogy only.**
- **Their move.** A linewidth attributed to quantum noise is rederived as the decay rate of the active mode, a pole
  distance, from a classical rate balance. The Schawlow–Townes formula is then recovered by four named approximations.
  Earlier, a known discrepancy (the Airy linewidth exceeding the Lorentzian one) is explained as the overlap of
  neighbouring modes in their sum, not as an approximation (Sec. 3.5, against Svelto).
- **The parallel here.** EEG_PROFILE.md §2 reads EEG26b's fitted "damping constant" q as a pole distance: the strip
  of analyticity equals the decay rate per harmonic. That is the same move, a fitted rate explained by the location of
  a pole. They verify the Airy sum numerically (31 modes in their Fig. 14, 6 × 10⁶ in Ismail et al.), without
  planted truth or a mutation.
- **No tool to adopt.** Their physics claims about the phase of stimulated emission and about quantum theories of
  the linewidth are outside this project. Responses to them were not read, so this review takes no position on them.

**Q4, their later work: Part II not found. The closest later work is a pole-based treatment of composite cavities,
with nothing on branch points.**
- **Part II.** Pollnau–Eichhorn promise it twice (Sec. 2.2 on counter-propagating modes; Sec. 10 on the factor of two
  near threshold). Four web searches and Pollnau's OpenAlex record (works with "spectral coherence" in the title)
  find only Part I.
- **Their own later work** (abstracts only) moves to distributed-feedback resonators and Bragg mirrors, mode counting
  in a Fabry–Pérot resonator, laser photon statistics, and a statistical principle for bosons. None is about the
  analytic structure of a line shape or about a threshold law.
- **Citing works.** OpenAlex lists 43 for Part I and 305 for Ismail et al., 16 of them with a line-shape word in the
  title. Vujnović et al. (2026) is the closest. It gives the first-order complex pole shift of a weak internal
  reflector and continuation of the poles. The rest are reflection Airy distributions, Fano devices, analytic
  (Hilbert-transform) spectra and laser engineering, by title or abstract.

### Citation chains (OpenAlex, 2026-11-09)

- **Forward from Pollnau–Eichhorn** (W3010732843): 43 records, listed in `I221-mac/search_log.md`. None was opened in
  full. By title, the nearest are Atvars (2021, abstract read), Khurgin–Noginov (2021) on the laser threshold, and a
  degenerate coherent perfect absorber (Science, 2022), which are titles only.
- **Forward from Ismail et al.** (W2461253794): 305 records, filtered by title to 16. Vujnović et al. was read through
  Sec. 6, and He–Lu–Zhao (2019) and Zhou et al. (JLT 43) as abstracts.
- **Their own works**: Pollnau (A5031372095), 60 records since 2017; Eichhorn (A5060333217), 20 resonator or linewidth
  records since 2016.
- **INSPIRE** has no record of this line of work (`a Pollnau`: 0), so no INSPIRE chain exists.
- **Backward from Pollnau–Eichhorn** for the decomposition: Koppelmann [51] and Bayer-Helms [52], whose exactness they
  quote, and Remmert [54] for the cot series, all *(unverified)*. Suter–Dietiker, Appl. Opt. 53, 7004 (2014), proposed
  the Lorentzian finesse earlier [58] *(unverified)*.

### References checked

| Entry | Checked against | Verdict |
|---|---|---|
| Pollnau, Eichhorn, Prog. Quantum Electron. 72, 100255 (2020), doi:10.1016/j.pquantelec.2020.100255 | the KITopen record and the PDF's running footer; OpenAlex W3010732843 | OK. The PDF says "Available online 9 March 2020", and the KITopen record "published online 11.08.2020" |
| Ismail, Kores, Geskus, Pollnau, Opt. Express 24(15), 16366–16389 (2016), doi:10.1364/OE.24.016366 | the published PDF's header in KTH DiVA (diva2:948682) and Europe PMC (PMID 27464090) | OK (upgrades #198's PARTIAL) |
| Vujnović, Kralj, Karuža, Photonics 13(8), 721 (2026), doi:10.3390/photonics13080721 | the MDPI article page and OpenAlex | OK |
| Ismail, Kores, Geskus, Pollnau, Proc. SPIE 10090, 1009011 (2017), doi:10.1117/12.2252740 | OpenAlex (title, DOI) and search listings; the SPIE page was empty | PARTIAL |
| Eichhorn, Appl. Phys. B 128(10) (2022), doi:10.1007/s00340-022-07897-6; He, Lu, Zhao, Opt. Express 27(13), 17876 (2019), doi:10.1364/OE.27.017876; Atvars, JOSA B 38(10), 3116 (2021), doi:10.1364/JOSAB.419993; Zhou, Fu, Chen, Tsai, Jun, J. Lightwave Technol. 43(3), 1007–1016, doi:10.1109/JLT.2024.3485574; Pollnau, Opt. Lett. 43, 5033–5036 (2018), doi:10.1364/OL.43.005033 | OpenAlex records (authors, volume, issue, pages, DOI); Opt. Lett. also Pollnau–Eichhorn's ref. [1] | PARTIAL: Eichhorn's article number and Zhou et al.'s year *(unverified)* |
| Koppelmann, Progress in Optics 7 (1969); Bayer-Helms, Z. Angew. Phys. 15, 330–338 (1963); Remmert, *Theory of Complex Functions* (Springer, 1998); Suter, Dietiker, Appl. Opt. 53, 7004 (2014); Blaquière, C. R. Acad. Sci. 255, 3141 (1962) | Pollnau–Eichhorn's and Ismail et al.'s reference lists only | UNVERIFIABLE here, cited as not read |

### Gap statement

- **Not read.** Part II, which was not found. The appendices and references of Vujnović et al. The SPIE paper of 2017
  and the Pollnau–Ismail chapter of 2018, which repeat the 2016 paper by their abstracts and titles. The S-matrix-pole
  treatments of the laser threshold that surfaced in a search (Stone's group; Beenakker, arXiv:0904.1432). Koppelmann
  and Bayer-Helms. Responses to Pollnau–Eichhorn's claims about quantum theories of the linewidth.
- **Channels.** OpenAlex for the forward chains, since INSPIRE does not index this line. Google Scholar was not reached.
  Of the 43 and 305 citing records, 4 were opened, one in part and three as abstracts; the others rest on titles.
- **Rounds.** Round 1 covered the four questions and the two named papers. Round 2 followed the OpenAlex chains and
  read the closest citing work. The search is not dry for Q1's branch point, since no optical source on a pole
  merged with a logarithmic branch has surfaced in #198 or here.

### Discrepancies and proposed text (for the coordinator)

- **No discrepancy** with the repository's text was found. The issue's phrase "its hump threshold q_c = 3 − 2√2, which
  is the same point" holds at δ = 0, as EEG_PROFILE.md §2 states it; at EEG26b's δ the two separate (identity 4).
- **#198's Svelto pointer resolved.** #198 noted that #172 recorded Wikipedia's citation of Svelto's *Principles of
  Lasers*, 5th ed., as ch. 4.5.1, and that Pollnau–Eichhorn cite it as ch. 5.3. They cite both: [18], ch. 4.5.1,
  pp. 142–146, for the round-trip-decay derivation and the approximate Airy linewidth, and [19], ch. 5.3,
  pp. 169–171, for Svelto's account of the Airy–Lorentzian difference. Neither passage was checked against the book.
- **Ismail et al.** is now read, and #198's Q1 verdict stands: no closed form for a frequency-dependent reflectivity,
  and no poles, zeros or branch points.
- **Proposed text** for notes/EEG_PROFILE.md §2, after (PF): "**(SEC) A secant, and its pole series** (*checked
  numerically*). With σ = ½ ln(1/q) and φ = θ + δ/2, β̃ = (A*/2√q) Re[e^{−iδ/2} sec(φ + iσ)]. Its Mittag-Leffler
  series is an alternating sum of Lorentzians of half-width σ centred at (k + ½)π − δ/2, each dispersive at δ = 0 and
  turned toward an absorptive one by δ/2. Pollnau and Eichhorn sum same-sign absorptive Lorentzians of the same width
  to the Airy distribution [Prog. Quantum Electron. 72, 100255 (2020), Eqs. (99)–(108)]; the alternation here comes
  from the odd harmonics." And after the hump threshold: "At δ = 0.0985, q_c = 0.21328; the Airy factor's half-maximum
  breakdown stays at 3 − 2√2 for every δ." Both checks are raw `I221-mac/identity_check.py`, items 1 and 4.
- **No novelty sentence** was added by this task. The strongest statement made is "not found in the sources searched".

### Recommended tasks (for the coordinator to file)

1. **Put the pole-series and threshold identities into `forms_check.py`.**
   - *What.* Add identity 1 (the secant form and its alternating pole series), identity 2 (the norm as the integrated
     Airy distribution) and identity 4 (q_c(δ) against the δ-independent Airy breakdown) to
     `solver/eeg_profile/forms_check.py`, each with its own mutation, and q_c(0.0985) to EEG_PROFILE.md §2.
   - *Why.* They are the shared structure of Q1, and §2 is where a reader of the profile looks for them.
   - *Acceptance.* forms_check.py passes, and each new mutation (the sign alternation dropped, R₁R₂ = q, δ set to 0)
     makes its check fail. q_c(0.0985) from the double-zero solve and from a direct count of extrema on a grid agree
     to 10⁻⁵, or the task stops.
   - *Depends on.* #221. Minutes on one core.
2. **Subtract the pole series from the tuned profile and measure what remains.**
   - *What.* Fit the nearest pole pair's location and residue, subtract its full Mittag-Leffler series (identity 1,
     with the fitted σ, δ and residue) from the tuned harmonics, and fit the remainder's harmonics to
     w_∞^k k^{−p}(a + b ln k). Run it first on `rational_fit.py`'s branch plant and on its meromorphic plant.
   - *Why.* EEG_PROFILE.md §8.5 reads a pole with a weaker logarithmic branch at the same point from a ν⁻³ law of the
     ratios. Subtracting the pole series leaves the branch alone, so its exponent and coefficient can be measured
     directly, for #150 and #216.
   - *Acceptance.* On the branch plant the remainder's planted exponent and coefficient come back within 5%. On the
     meromorphic plant the remainder falls geometrically, with no power law. If either fails, the real-data numbers
     are void. Then report p and a for the tuned profile, *conditional* on Conjecture 1, with the spread over the
     fitted harmonic windows named as such.
   - *Depends on.* #133, #216, #221. Under an hour on one core.
3. **Read what did not open.**
   - *What.* Vujnović et al.'s Appendices A–D (non-simple poles) and references; the SPIE paper of 2017 and the
     Pollnau–Ismail chapter of 2018; the S-matrix-pole papers on the laser threshold that surfaced (Stone's group;
     Beenakker, arXiv:0904.1432), logged in §6.
   - *Why.* Q1's branch-point verdict and Q2's threshold verdict rest partly on sources read through abstracts.
   - *Acceptance.* Each is read and recorded. The task fails, and says so, if one of them contains a branch point at a
     cavity pole, a closed form for a frequency-dependent reflectivity, or a growth or build-up law at a laser
     threshold that the verdicts above miss.
   - *Depends on.* #221.

## Addendum (2026-11-12, issue #527): Newton's method at singular and nearly singular roots, against #499's wall

**Question.** What does the prior work on Newton's method at singular and nearly singular roots, and on the
Lyapunov–Schmidt reduction of a system to its weak subspace, say about #499's findings at the D 242.5 wall, and which
corrector does it support? Notation as in #499: J = F′(z0) at the corrector's start z0, V and U its four weak right and
left singular vectors, X the strong complement of span(V), B = UᵀJV (singular values σ₁…σ₄), Q = UᵀF″(V·, V·) and
T = UᵀF‴(V·, V·, V·) the second and third derivatives projected on the weak left directions, and a the weak
coefficients of Newton's first move. The findings are numerical evidence (#499, comments of 2026-11-11 17:53 UTC to
2026-11-12 00:34 UTC):
- **F1.** Newton's first move lies on span(V), share 1.0 with 8.5e-4 of it elsewhere; σ₁…σ₄ = 2.5e-27…7.9e-26 at
  D 242.36 and ‖J‖ ≈ 3e34.
- **F2.** Q is below the differences' resolution (pure truncation, Q(h) = 6.3e60 h², bounded by 6e12 at h = 1e-24),
  while T ≈ 1.6e33 agrees across h to 3e-18.
- **F3.** The cubic reduced system B(c − a) + T(c, c, c)/6 = 0 has one real root near the start, at |c| = 0.45–0.54 |a|.
  It solves the actual weak residual on z0 + Vc to under 1%; the other three roots are artefacts of the truncation.
- **F4.** With the weak part placed on that root, newton_od is still refused. The strong-row residual that the weak move
  leaves, ½F″(Vc*, Vc*) ≈ 2e-26, is the suspected coupling.

**How.**
- **Searches.** 24 web searches on 2026-11-12, and record queries of the Crossref, zbMATH Open and Semantic Scholar APIs
  (the last gave no abstracts).
- **Read in full.** Griewank's thesis (1980), Decker, Keller and Kelley (1983, the published PDF in CaltechAUTHORS),
  Deuflhard (1974, the GDZ scan), Keller's TIFR lectures (the Internet Archive's copy of 2025-06-24; the TIFR link
  returns 404), Kuznetsov (1999, §1–5, CWI repository), Dallas and Pollock (2023, arXiv v4), and Fischer, Izmailov and
  Solodov (2019, the preprint of 2017-09-04), and §4.3 of the p-regularity survey arXiv:2501.07671.
- **Records or reviews only.** The SIAM, AMS, Springer, Elsevier and DTIC pages refused the fetch (403 or a login
  redirect). The other works below are recorded from Crossref's deposited metadata, zbMATH reviews and what the full
  texts above say about them, marked as such.
- **Text channel.** The PDFs were downloaded with curl and their text extracted with macOS PDFKit, or with the Vision
  framework's OCR for scans (Deuflhard 1974; DKK 1983's theorem statements, read in both extractions). Quotes are from
  that text and were not compared with the page images. Obvious OCR errors were corrected by hand ("newe" to "new"),
  and others may remain.
- **Inferences** are marked *inferred*. Every theorem read is proved for an exactly singular F′(x*); none treats a root
  with a nonzero smallest singular value, except Decker and Kelley (1985, review only) and remarks in Griewank's thesis.

### Closest works

| Work | Bears on | What it contains | What it does not contain |
|---|---|---|---|
| **A. O. Griewank**, *Analysis and modification of Newton's method at singularities*, PhD thesis, Australian National University (June 1980), read in full | F1–F4, design 1 | **Order and regularity.** A singularity has order k when the derivatives projected on the singular equations, P∇^q f(x*), vanish for q ≤ k and P∇^{k+1}f(x*) does not (Thm 1.6 (ii)); it is regular when B(t) = (1/k!)P∇^{k+1}f(x*)t^k restricted to the null space N is nonsingular for some t ∈ N (§2.1). **Rates.** From a star-like domain, ρ_{j+1}/ρ_j → k/(k+1) (Lemma 2.2), and the residual falls by k/(k+1) (Lemma 3.2). **Line searches.** "the usual line search conditions of stabilised Newton methods (e.g. Goldstein test [23]) will always be met by the full Newton step during the final approach to a regular singularity. However it was found in [24] that such modifications can slow down the iteration considerably before the final pattern has been established" (p. 94). **Corrected steps.** Reddien's corrected step leads "to a point x′ much closer to the solution x* than x_{j+1} = g(x_j) but ... the subsequent Newton step from x′ tended to be disadvantageous. In our framework this means that x′ can lie outside W and may even belong to the singular set" (p. 96). **Acceleration.** The two-point iteration y_{i+1} = λg(g(y_i)) − (λ−1)g(y_i), λ ∈ (1, k+1), is Q-linear, and Q-superlinear as λ_j → k+1 (Thm 3.5, Cor. 3.6); at k = 2 "with fixed λ=2.8 converges quite nicely with Q-factor ≈ 0.044" on his test problem (p. 112). **Nearly singular.** "Once the residual ‖f‖ is of the same magnitude as the smallest singular value of the Jacobian at the solution, Newton's method is naturally superior" (p. 103). **Bordering.** "At higher order singularities the condition (4.60) cannot be satisfied so that the Jacobian of the overdetermined system f = 0, Z = 0 has at x* linearly dependent columns" (p. 150). | Irregular singularities ("not considered in this thesis", Preface); a theory for nonzero σ at the root; the strong rows' curvature acting back on the weak rows; an acceptance test like newton_od's. |
| **D. W. Decker, H. B. Keller, C. T. Kelley** (DKK), *Convergence rates for Newton's method at singular points*, SIAM J. Numer. Anal. 20 (1983) 296–314, read in full | F1, F2, F4, design 1 | "the convergence of the Newton iterates to the root is linear rather than quadratic" (abstract). **Blocks.** With E = N ⊕ X, F′(x) = A + B + C + D (2.3)–(2.4); F′(x) is invertible iff the Schur complement D̂ = D − CA⁻¹B is invertible on N (Thm 2.6). The orders a, b, c, d of the blocks' first Taylor terms, barred when every argument lies in N (3.3)–(3.5), fix the leading weak operator: H = D̄_d̄ if d̄ < b̄ + c̄, D̄_d̄ − C̄Â⁻¹B̄ if d̄ = b̄ + c̄, −C̄Â⁻¹B̄ if d̄ > b̄ + c̄ (3.10). Regions W(ρ, θ, m) = {‖P_X x̃‖ ≤ θ‖P_N x̃‖^m} (3.1). **Rates.** Thm 5.1 (d = 1): the N-error ratio tends to 1/2. Thm 5.9: if d = d̄ ≤ c and D_d is nonsingular for all P_N x̃ ≠ 0, Newton converges from W(ρ, θ, 1) with N-error ratio d/(d+1) (5.11); Example 2 (d = 2) tabulates R_i → 0.6667. **Acceleration.** Thm 7.4 (two Newton steps, then a step of d+1 times the Newton correction; assumes d = d̄ ≤ c, b ≥ min(2, d), (7.5)) and Thm 7.27 (y = x − F′(x)⁻¹F(x), x⁺ = y − (d+1)F′(y)⁻¹F(y); assumes b ≥ 2, c ≥ d = d̄ ≥ 2, (7.5)): quadratic convergence of the N-error. | Nearly singular roots; a rate theorem for d̄ = b̄ + c̄; line searches. |
| **G. W. Reddien**, *On Newton's method for singular problems*, SIAM J. Numer. Anal. 15 (1978) 993–996 | F2 | Not read. Griewank (p. iii): convergence "from within the intersection of some cone with some ball, provided x* is a regular first order singularity". DKK: "convergence, with overall linear rate 1/2, for the first problem was known previously [14]". | (unread) |
| **G. W. Reddien**, *Newton's method and high order singularities*, Comput. Math. Appl. 5 (1979) 79–86 | F2 | Not read. DKK (p. 305): a result like Thm 5.9 "was derived in Reddien [15] but under the additional restrictions that dim (N) = 1", with a lower bound on F^{(p+1)} and the X-order "(p+1) reduced to 2". | (unread) |
| **D. W. Decker, C. T. Kelley**, *Newton's method at singular points. I*, SIAM J. Numer. Anal. 17 (1980) 66–70; *II*, 17 (1980) 465–471; *Convergence acceleration for Newton's method at singular points*, 19 (1982) 219–229 | F2, design 1 | Not read. DKK: its Thm 5.1 (rate 1/2) is "a result from [2]" (I); II treats d = b = c = 1 with d̄ = 2 in W(ρ, θ, 2) (p. 302); Thm 7.4 generalizes "a modified scheme of [6]" (Griewank's thesis). | (unread) |
| **D. W. Decker, C. T. Kelley**, *Expanded convergence domains for Newton's method at nearly singular roots*, SIAM J. Sci. Stat. Comput. 6 (1985) 951–966 | F1, F3 | zbMATH review (B. Döring) only: Newton "when the derivative of F is nearly singular at a zero"; "there is a larger than expected domain of attraction for the Newton iterates", "but that convergence appears to be only linear in most of this domain"; "A modification of Newton's method is given that improves this slow convergence." | Whether its class allows a vanishing quadratic term, and the modification itself (unread). |
| **A. Griewank**, *On solving nonlinear equations with simple singularities or nearly singular solutions*, SIAM Rev. 27 (1985) 537–563 | design 1 | zbMATH review (E. L. Allgower) only: Newton near a regular singular zero of order k, extrapolation, overrelaxation, bordering ("The bordering method has principally been used for simple singularities") and quadratic models after Frank and Schnabel, tested on a tubular reactor problem. | (unread) |
| **A. Griewank, M. R. Osborne**, *Analysis of Newton's method at irregular singularities*, SIAM J. Numer. Anal. 20 (1983) 747–773 | F2 | Record only (zbMATH keywords "irregular singularities", "divergence", "chaotical convergence behaviour"). Griewank's thesis: such cases "lead to very irregular behaviour of the Newton iteration even if the singularity is balanced" (p. 151). | (unread) |
| **M. Dallas, S. Pollock**, *Newton–Anderson at singular points*, Int. J. Numer. Anal. Model. 20 (2023) 667–692, arXiv:2207.12334v4, read in full | F2 | Restates DKK's setting with D̂_N(x) = P_N f″(x*)(P_N(x − x*), P_N ·) and the error expansion e_k + w_{k+1} = ½P_N e_k + T_k P_R e_k + O(‖e_k‖²) (3); D̂_N "fails to be invertible for all P_N(x−x*) ≠ 0 if dim N > 1 and odd". §7: the order d is the least d with P_N f^{(d+1)}(x*)(x − x*)^d ≠ 0, and P_N(e_k + w_{k+1}) = d/(d+1) P_N e_k + … (Lemma 7.1); depth-one Anderson acceleration with γ-safeguarding converges with a factor below d/(d+1) when dim N = 1 (Thm 7.1). | High order with dim N > 1; nearly singular roots. |
| **R. B. Schnabel, P. D. Frank**, *Tensor methods for nonlinear equations*, SIAM J. Numer. Anal. 21 (1984) 815–843; **D. Feng, P. D. Frank, R. B. Schnabel**, *Local convergence analysis of tensor methods for nonlinear equations*, Math. Program. 62 (1993) 427–459 | designs | Schnabel–Frank not read (SIAM and DTIC refused). Feng–Frank–Schnabel, zbMATH review (Z. Wang) only: on "problems where the Jacobian at the root has a null space of rank one", the iterates converge "two or three-step Q-superlinearly to the solution with Q-order 3/2", "while standard methods converge linearly with constant converging to 1/2". | A null space of dimension above one; a vanishing second-order term on it. |
| **P. Deuflhard**, *A modified Newton method for the solution of ill-conditioned systems of nonlinear equations with application to multiple shooting*, Numer. Math. 22 (1974) 289–315, GDZ scan read in full | F4, design 2 | The damped Newton method "is extended in such a way as to apply to the solution of ill-conditioned systems of nonlinear equations, i.e. systems having a "nearly singular" Jacobian at some iterate" (abstract). The natural level function, scaled by J(x_k)⁻¹ (2.8), gives the monotonicity test on the simplified Newton correction with the old Jacobian (4.1)–(4.2). Rank strategy (4.7): when the test fails for every λ_k in [λ_min, 1], "reduce the pseudo-rank of the Jacobian by one, compute the new correction Δx_k from (2.16), and try once more"; the reduced method is Newton for "the reduced mapping PF(Px)", the dropped components frozen (4.8). (4.10): "Start with initial pseudo-rank of the Jacobian less than maximum rank. In the course of the iteration increase the rank gradually." | Convergence to a root whose dropped equations must also be solved; rates at singular roots; a test with the new Jacobian. |
| **P. Deuflhard**, *Newton Methods for Nonlinear Problems: Affine Invariance and Adaptive Algorithms*, Springer Series in Computational Mathematics 35 (2004; 2011 printing) | F4 | Not read. Crossref lists the chapters "Systems of Equations: Local Newton Methods" (pp. 45–107), "Systems of Equations: Global Newton Methods" (109–172) and "Least Squares Problems: Gauss-Newton Methods" (173–231). ZIB's page of the book's codes: NLEQ_ERR implements §3.3.3, a global Newton method with an error-oriented criterion and an adaptive trust region; NLEQ2 has a rank strategy "slightly different from Section 3.3.3", and its header reads "Damped affine invariant Newton method with rank-strategy", with a QR decomposition "with subcondition estimation, rank decision and computation of the rank-deficient pseudoinverse". | (the text, including its restricted monotonicity test and its treatment of rank deficiency, is unread) |
| **H. B. Keller**, *Lectures on Numerical Methods in Bifurcation Problems*, TIFR Lectures on Mathematics and Physics 79 (Springer, 1987), read in full | F2, F3, design 2 | Ch. 1, Example 2: −u³ + λ₁u + λ₂ = 0 has three real roots on one side of the fold λ₁³/27 = λ₂²/4 (1.17), a pitchfork at λ₂ = 0, and "The fold has a cusp at the origin". §5.12: the algebraic bifurcation equations a₁₁α² + 2a₁₂αβ + a₂₂β² = 0, a₁₁ = ψᵀG_uu φφ. §5.13: near a simple bifurcation point "there is a full conical neighbourhood ... in which F_X(Y,s) is nonsingular", and Newton converges from it. Method III (Lyapunov–Schmidt): u = u₁(σ) + ε(φ + v), ψᵀv = 0, with a scalar equation h(s, ε, v) = 0 for the left-null component (5.30). §4.11: the bordering algorithm, valid when A and the bordered matrix are nonsingular, and a variant for singular A. Method V (perturbed bifurcation) and §5.27 (fold following for −ξ³ + λξ + τ = 0). | Newton's rates at singular points; reduced equations with a vanishing quadratic term beyond the scalar examples. |
| **Yu. A. Kuznetsov**, *Numerical normalization techniques for all codim 2 bifurcations of equilibria in ODE's*, SIAM J. Numer. Anal. 36 (1999) 1104–1124, §1–5 read | F2, F4, design 2 | §5, the cusp: a simple zero eigenvalue with Aq = 0, Aᵀp = 0, ⟨p, q⟩ = 1, and the reduced equation ẇ = bw² + cw³ with b = ½⟨p, B(q, q)⟩, B and C the second and third derivatives. When b = 0, "c = (1/6)⟨p, C(q,q,q) − 3B(q, A^INV B(q,q))⟩", where A^INV B(q, q) is found "by solving the nonsingular (n+1)-dimensional bordered system" (the idea credited to Keller, 1977). | A null space of dimension above one; a nearly singular A; an iteration (the coefficients are computed at a known point). |
| **A. Fischer, A. F. Izmailov, M. V. Solodov**, *Local attractors of Newton-type methods for constrained equations and complementarity problems with nonisolated solutions*, J. Optim. Theory Appl. 180 (2019) 140–169, preprint read | F2 | 2-regularity in a direction v: "the p×p-matrix Φ′(ū)+ΠΦ′′(ū)[v] is nonsingular", equivalently "there exists no nonzero u∈ker Φ′(ū) such that Φ′′(ū)[v, u] ∈im Φ′(ū)"; under it Newton-type methods have a star-like domain of attraction. Same line, records only: Izmailov, Kurennoy, Solodov, Math. Program. 167 (2018) 355–379; Fischer, Izmailov, Solodov, *Unit stepsize for the Newton method close to critical solutions*, Math. Program. 187 (2021) 697–721, whose search listing says it couples Newton "with a linesearch technique for globalization that locally accepts unit stepsize". | Any result without 2-regularity, which Q = 0 excludes in every direction (*inferred*). |
| **E. Bednarczuk, O. Brezhneva, K. Leśniewski, A. Prusińska, A. Tret'yakov**, *Towards nonlinearity. The p-regularity theory. Applications and developments*, arXiv:2501.07671v1 (2025), §4.3 read | designs | §4.3.2: the p-factor Newton method x_{k+1} = x_k − (F′ + P₁F″h + … + P_{p−1}F^{(p)}[h]^{p−1})⁻¹(F + P₁F′h + … + P_{p−1}F^{(p−1)}[h]^{p−1}), with projectors built from Im F′(x̄) and a direction h for which the p-factor matrix is nonsingular; quadratic convergence from a ball (Thm 4.16, quoted from Szczepanik, Prusińska and Tret'yakov 2012, not read). | Nearly singular roots: the projectors are those of an exactly singular F′(x̄). |

### #499's findings against the literature

| Finding | Prior work that predicts or explains it | Not predicted by any source read |
|---|---|---|
| **F1**, the first move on span(V) | **Explained.** At a singular root the error along N falls linearly while the X-error falls at higher order (DKK §1, (5.18)–(5.19)); after the first steps "the iterates are essentially confined to the one-dimensional nullspace" (Griewank, p. 149, of his case dim N = 1). At a nearly singular start the weak part of a Newton correction is the weak residual over σ (*inferred*, SVD algebra), and Deuflhard (1974, §1) names rounding at a nearly singular Jacobian as what forces tiny damping factors. | The move's size, a rounding-level weak residual \|Ba\| ≈ 1e-55 over σ (*inferred* arithmetic), and win2's floor of 1.3e-30 to 2e-30 for the first move at any a. |
| **F2**, Q unresolved, T ≈ 1.6e33 | **Classified, not predicted.** A vanishing projected second derivative with a nonvanishing third is a singularity of order 2 (Griewank, Thm 1.6), d̄ = 2 (DKK (3.5)), a high-order singularity (Reddien 1979) and a root of order d = 2 (Dallas–Pollock §7). In the reduced equation it is the cusp, b = ½⟨p, B(q, q)⟩ = 0 (Kuznetsov §5; Keller, Ch. 1). It rules out 2-regularity in every direction (Fischer–Izmailov–Solodov's definition), so the theory of critical solutions does not apply. At an exactly singular root of this order the N-error falls by d/(d+1) = 2/3 per step (DKK Thm 5.9 and Example 2; Griewank, Lemma 3.2), not by the 1/2 of a first-order root. | Why Q vanishes here. Also, the reduced cubic coefficient is not T alone (next section). |
| **F3**, one real root at about half the move | **Partly.** In one dimension B(c − a) + Tc³/6 = 0 is Keller's −ξ³ + λξ + τ = 0 with λ = −6B/T and τ = 6Ba/T (*inferred*). It has three real roots only when 4λ³ > 27τ² (Keller (1.17)), which needs λ > 0, that is B and T of opposite sign. So one real root is the same-sign side of the cusp, and the pitchfork that #499 considered and dropped is the other side (*inferred*, scalar model). Newton's step from c = 0, the reduced cubic's inflection point where the weak Jacobian is smallest (B), overshoots the root (*inferred*). With κ = \|T\|\|a\|²/(6\|B\|) = 1.6–3.5 from #499's numbers, κx³ + x − 1 = 0 gives c*/a = 0.52–0.62, against the measured 0.45–0.54 (*inferred*). | The four-dimensional tensor case; a start at the reduced cubic's inflection point (every convergence region read is a cone or star about the root). |
| **F4**, refused after the weak-first step | **Explained in kind.** (i) Griewank's account of Reddien's corrected steps (p. 96) is the same observation: a step that lands much closer to the root along N is followed by a poor Newton step, because the point "can lie outside W". The weak-first point has its N-error removed and its X-error (the strong residual of 2e-26 over the strong singular values) left, so it lies outside every region W(ρ, θ, m) of DKK (3.1), and no theorem read covers the next step (*inferred*). (ii) The coupling has a name: the term C̄Â⁻¹B̄ of DKK's leading weak operator (3.10), and −3⟨p, B(q, A^INV B(q, q))⟩ in Kuznetsov's cusp coefficient. F″(v, v) ≈ 1e33 in the strong rows makes b̄ = 1, so this term is absent only if the mixed weak-left curvature UᵀF″(v, x), with v ∈ span(V) and x ∈ X, vanishes (*inferred* from (3.10) and Kuznetsov §5). #499's T and its root check use the affine subspace z0 + Vc, which does not see it. (iii) Griewank (Lemma 3.2): line searches accept the full step in the final approach to a regular singularity, and slow the iteration before that pattern sets in. | The size of the coupling term. newton_od's acceptance test (the correction at the trial with its own Jacobian) is analysed in no source read; Deuflhard (1974) analyses the old-Jacobian test. |

### What the theorems assume, against #499's B, Q and T

- **Exact singularity.** Every rate theorem read assumes F′(x*) singular. Here σ₁ = 2.5e-27 at z0, and the weak block's
  Jacobian at the root is about ½T(c*, c*, ·) ≈ 1.6e-26 (#499, 00:34 UTC). The singular theory can describe the
  iteration only where the cubic term dominates the linear one, |c| ≫ √(2σ/|T|) ≈ 1.8e-30 for σ₁ and 1.0e-29 for σ₄
  (*inferred*). Below that the reduced root is regular and Newton is quadratic, as Griewank says for nearly singular
  roots (p. 103). #499's first moves, |a| = 0.8–1.3e-29, lie at this crossover, where neither regime's theory holds
  (*inferred*; σ is from D 242.36 and T from D 242.5146, which changes these scales by about 10%). The accepted
  attempt's four linear steps before quadratic convergence fit that picture, but #499 showed the factor per step to be
  the line search's λ.
- **Q.** Order 2 needs the whole weak-left second derivative to vanish: Griewank's order k = 2 needs P∇²f(x*) = 0, and
  DKK's d = 2 needs P_N F″(x*)(x̃, P_N ·) = 0 for every x̃, so the mixed block UᵀF″(x, v), x ∈ X, as well. #499 resolved
  only the weak–weak block Q. Its bound, 6e12 at h = 1e-24, is far above the size at which Q would change the reduced
  equation at the root's scale: ½Q(c, c) is below Bc and below T(c, c, c)/6 at |c| ≈ 5e-30 only if |Q| ≪ 1e3
  (*inferred*). The root check of F3 bounds Q along one direction only.
- **T.** Regularity needs ½T(t, t, ·) nonsingular on span(V) for some direction t (Griewank) or every direction (DKK
  Thm 5.9). This is unmeasured; the single real root is consistent with it and does not show it.
- **B, and the strong rows.** F″(v, v) ≈ 1.9e33 lies in the strong rows (#499, 17:53 UTC, which stands), so DKK's b and
  b̄ are 1. The acceleration theorems need b ≥ 2.

### Corrector designs from the literature

**Design 1: overrelaxed Newton at an order-2 singular root** (DKK Thms 7.4 and 7.27; Griewank Thm 3.5, Cor. 3.6).
- *Iteration.* y = x − F′(x)⁻¹F(x), x⁺ = y − 3F′(y)⁻¹F(y) (DKK with d = 2), or y⁺ = λg(g(y)) − (λ − 1)g(y) with
  λ → 3, g the Newton map (Griewank with k = 2).
- *Claim, proved there.* Quadratic convergence of the N-error (DKK (7.29)); Q-linear convergence whose factor falls as
  λ → k+1, superlinear for λ_j → k+1 (Griewank), with a Q-factor of 0.044 at λ = 2.8 on his k = 2 test problem.
- *Assumptions against #499.* **Exact singularity: fails** (B is not zero), and an overrelaxation by 3 overshoots once
  |c| reaches the crossover above (*inferred*). Griewank advises reverting to Newton "when the rounding errors become
  significant or the solution turns out to be only nearly singular" (p. 102). **b ≥ 2: fails**, since F″(v, v) lies in
  the strong rows. Q = 0 with the mixed block: unmeasured. T(t, t, ·) nonsingular for every t: unmeasured. A start in a
  cone W(ρ, θ, 1) about the root: #499's starts lie at the reduced cubic's inflection point instead (*inferred*).
- *Verdict.* Not applicable as stated: two assumptions fail on #499's data.

**Design 2: a Lyapunov–Schmidt corrector** (Keller's Method III and bordering algorithm; DKK Thm 2.6; Kuznetsov §5;
Deuflhard's (4.8) and (4.10)).
- *Iteration.* For given weak coordinates c, solve the strong rows for w ∈ X with c held, which is Newton on
  Deuflhard's reduced mapping with the weak components frozen. Then take a Newton step on the four-dimensional reduced
  map g(c) = UᵀF(z0 + Vc + w(c)). Its Jacobian is the Schur complement D − CA⁻¹B on span(V) (DKK Thm 2.6), formed by a
  bordered solve (Keller §4.11; Kuznetsov (4.6)). g is evaluated with the full F, so no Taylor model of it is needed.
  With Q = 0 its cubic coefficient is T − 3S, S(c, c, c) = UᵀF″(Vc, A^INV P_X F″(Vc, Vc)) (Kuznetsov §5, extended from
  one null vector to four, *inferred*). Deuflhard's rule (4.10), to start at a reduced pseudo-rank and raise it, orders
  the work the same way: strong rows first, weak rows after.
- *Claim.* Each part is quadratic where Newton–Kantorovich applies (Keller §2.23–2.24): the strong solve when the
  theorem's bound holds for A = P_X J P_X on X, whose smallest singular value is σ₅, and the reduced step near a root
  where g′(c*) is nonsingular. No source read proves convergence of the nested iteration as a whole (*inferred*).
- *Assumptions against #499.* **A gap after σ₄ = 7.9e-26:** σ₅ is not in #499, unmeasured. **g′(c*) nonsingular:** its
  T part is about 1.6e-26, six times σ₁, so the reduced root is regular unless the term 3S cancels it (*inferred*;
  S unmeasured). **Q:** g needs no Q, since it is evaluated directly. **Accuracy:** g must resolve |Ba| ≈ 1e-55 while
  the strong rows carry 1e-26 during the inner solve, well within octo-double's 1e-125 relative (*inferred*).
  **Start:** from c = 0 the reduced step meets the inflection again (g′ = B); from c = a, Newton's own first move,
  g′ is 20–50 times σ₁ (*inferred*).
- *Verdict.* No measured quantity contradicts it, and two assumptions are unmeasured. It differs from #499's weak-first
  step in one point: the weak equation is solved on z0 + Vc + w(c), with the strong rows solved, not on z0 + Vc. That
  is where the term S acts.

**Not candidates.** Tensor methods: Feng, Frank and Schnabel analyse a rank-one null space at a first-order root (their
baseline rate is 1/2), while #499's weak subspace is four-dimensional with a vanishing quadratic term, which a
second-order tensor model cannot represent (*inferred*). Bordering with a singularity condition fails at higher-order
singularities by Griewank's own statement (p. 150) and imposes an exact singularity that the root does not have. The
p-factor Newton method with p = 3 assumes an exactly singular F′(x̄) and its image; it was not checked further.

### References checked

| Entry | Checked against | Verdict |
|---|---|---|
| Griewank, PhD thesis, ANU (1980) | the PDF's title page (on R. P. Brent's ANU page, not the university repository) | OK |
| Decker, Keller, Kelley, SIAM J. Numer. Anal. 20(2), 296–314 (1983), doi:10.1137/0720020 | Crossref; the published first page (CaltechAUTHORS record zs4er-99k51) | OK |
| Reddien, SIAM J. Numer. Anal. 15(5), 993–996 (1978), doi:10.1137/0715064 | Crossref; zbMATH 0397.65042 | OK (bibliographic) |
| Reddien, Comput. Math. Appl. 5(2), 79–86 (1979), doi:10.1016/0898-1221(79)90061-0 | Crossref; zbMATH 0436.65032 | OK (bibliographic); DKK's list dates it 1980 |
| Decker, Kelley: 17(1), 66–70 (1980), doi:10.1137/0717009; 17(3), 465–471 (1980), doi:10.1137/0717039; 19(1), 219–229 (1982), doi:10.1137/0719012 | Crossref; zbMATH 0428.65037 for part I | OK (bibliographic) |
| Decker, Kelley, SIAM J. Sci. Stat. Comput. 6(4), 951–966 (1985), doi:10.1137/0906064 | Crossref; zbMATH 0597.65057 | OK (bibliographic) |
| Griewank, SIAM Rev. 27(4), 537–563 (1985), doi:10.1137/1027141 | Crossref; zbMATH 0598.65026 | OK (bibliographic) |
| Griewank, Osborne, SIAM J. Numer. Anal. 20(4), 747–773 (1983), doi:10.1137/0720050 | Crossref; zbMATH 0525.65025 | OK (bibliographic) |
| Dallas, Pollock, Int. J. Numer. Anal. Model. 20(5), 667–692 (2023), doi:10.4208/ijnam2023-1029 | Crossref; the arXiv page (journal reference) | OK |
| Schnabel, Frank, SIAM J. Numer. Anal. 21(5), 815–843 (1984), doi:10.1137/0721054 | Crossref; zbMATH 0562.65029 | OK (bibliographic) |
| Feng, Frank, Schnabel, Math. Program. 62, 427–459 (1993), doi:10.1007/BF01585177 | Crossref; zbMATH 0796.65069 (review through the MaRDI portal) | OK; MaRDI's date of 13 October 1994 disagrees, the deposited metadata is used |
| Deuflhard, Numer. Math. 22, 289–315 (1974), doi:10.1007/BF01406969 | the scan's first page, "Numer. Math. 22, 289-315 (1974)"; EuDML 132271 gives 289–316 | OK |
| Deuflhard, *Newton Methods for Nonlinear Problems*, Springer Ser. Comput. Math. 35 (2004); 2011 printing doi:10.1007/978-3-642-23899-4 | Crossref (the 2011 printing and its chapters); ZIB's NLEQ2 header for the 2004 edition | PARTIAL: the 2004 edition's DOI *(unverified)* |
| Keller, *Lectures on Numerical Methods in Bifurcation Problems*, TIFR Lectures on Math. and Phys. 79, Springer (1987) | zbMATH 0656.65063 (1987, IV+160 pp., ISBN 3-540-18367-1); the PDF's front matter (©TIFR 1986, ISBN 3-540-20228-5, 140 pages) | OK; MINOR: the two ISBNs and page counts differ |
| Kuznetsov, SIAM J. Numer. Anal. 36(4), 1104–1124 (1999), doi:10.1137/S0036142998335005 | Crossref; the CWI copy's first page | OK |
| Fischer, Izmailov, Solodov, J. Optim. Theory Appl. 180(1), 140–169 (2019), doi:10.1007/s10957-018-1297-2; Izmailov, Kurennoy, Solodov, Math. Program. 167(2), 355–379 (2018), doi:10.1007/s10107-017-1128-5; Fischer, Izmailov, Solodov, Math. Program. 187, 697–721 (2021), doi:10.1007/s10107-020-01496-z | Crossref; zbMATH 1470.65102 for 2021 | OK (bibliographic); the 2019 paper's content is from its 2017 preprint |
| Bednarczuk et al., arXiv:2501.07671v1 (2025) | the arXiv PDF's header | OK (preprint); the source of its Thm 4.16, Szczepanik, Prusińska, Tret'yakov (Schedae Informaticae, 2012), *(unverified)* |

### Gap statement

- **Not read.** Reddien 1978 and 1979; Decker and Kelley 1980 (I, II), 1982 and 1985; Griewank 1985; Griewank and
  Osborne 1981 and 1983; Kelley and Suresh 1983 and Kelley 1986 (records found, not entered above); Schnabel and Frank
  1984; Feng, Frank and Schnabel 1993; Fischer, Izmailov and Solodov 2021; Deuflhard's book; Allgower and Georg's
  *Introduction to Numerical Continuation Methods* (SIAM Classics 45, 2003, record only). Reviews and the full texts'
  accounts of them stand in, which is the residual risk.
- **The closest unread work** is Decker and Kelley (1985), on nearly singular roots with a modified Newton method. Its
  class and its modification could change design 2's standing.
- **Rounds.** Round 1 covered the issue's list. Round 2 went forward from DKK and Dallas–Pollock (Reddien 1979, Kelley
  1986, critical solutions, p-regularity) and to numerical normal forms (Kuznetsov 1999) for the reduced cubic
  coefficient. The search is not dry: no source read treats a nearly singular root whose reduced quadratic term
  vanishes.
- **Update (#531).** Schnabel and Frank (as their 1983 report) and Fischer, Izmailov and Solodov (2021) are now read,
  and the other works of this list are recorded as not obtained, in the addendum of issue #531 below.

### Discrepancies and proposed text (for the coordinator)

- **#499's T is not the reduced cubic coefficient.** With Q = 0 the reduced equation's cubic coefficient is T − 3S
  (Kuznetsov §5; DKK (3.10)). #499's T, its h-independence and the root check hold for the map c ↦ UᵀF(z0 + Vc) on the
  affine subspace. They do not yet show the root of the reduced equation, where the strong rows are solved. Reported,
  not changed.
- **Rate 1/2 or 2/3.** Issue #527's "linear convergence at rate 1/2 near a double root" is the first-order case, Q ≠ 0.
  With Q = 0 the classical rate is 2/3 (DKK, Example 2). #499 found the factor 1/2 to be the line search's λ, so neither
  classical rate has been observed.
- **Q's bound.** #499's reading that the weak nonlinearity is cubic rests, away from the root's direction, on the
  h-scan, whose bound on Q (6e12) is about 6 × 10⁹ times the 1e3 at which Q would change the reduced equation at
  |c| ≈ 5e-30 (*inferred*).
- **Citation details.** DKK's list dates Reddien's high-order paper 1980 (Crossref and zbMATH: 1979). MaRDI dates
  Feng–Frank–Schnabel 1994 (Crossref and zbMATH: 1993). Keller's lectures carry two ISBNs.
- **REF_CHECK.md, Deuflhard 1974.** Its content is now read from the scan: the natural monotonicity test with the old
  Jacobian, (4.1)–(4.2), and the rank strategy, (4.7) and (4.10). Proposed: replace "the content ... from the abstract
  only" in that row with a pointer to this addendum.
- **Proposed text** for the wall's write-up: "A vanishing Q = UᵀF″ with T = UᵀF‴ ≠ 0 makes the weak singularity one of
  order 2 [Griewank 1980; Decker, Keller and Kelley 1983], whose reduced equation is the cusp normal form [Keller 1987,
  Ch. 1; Kuznetsov 1999, §5]. At an exactly singular root of this order Newton's error along the null space falls by
  2/3 per step. The accelerations proved for it assume that the strong rows of F″ vanish along the null space [Decker,
  Keller and Kelley 1983, Thms 7.4 and 7.27], which F″(v, v) ≈ 1e33 rules out here."
- **No novelty sentence** was added by this task.

### Recommended tasks (for the coordinator to file)

1. **Measure the composite term of the reduced cubic.**
   - *What.* At the two D 242.5146 starts of #499 (accepted a 0.0078125, refused a 0.03125), form
     w_jk = A^INV P_X F″(v_j, v_k) by a bordered solve, then S = UᵀF″(v_i, w_jk) by central differences checked
     between h and h/10, symmetrized, and compare 3‖S‖ with ‖T‖. Evaluate the reduced map
     g(c) = UᵀF(z0 + Vc + w(c)), w(c) from a strong Newton solve at fixed c, at c = a and at #499's root c*.
   - *Why.* The literature's reduced cubic coefficient is T − 3S, and #499's measurements see only T. This decides
     whether the weak-first refusal is the coupling (F4).
   - *Acceptance.* S agrees between h and h/10 to 1e-2, or is bounded below its truncation. Control: w replaced by 2w
     must change g(c*) by the amount S predicts, to 1e-2. If 3‖S‖ < 1e-2‖T‖ and |g(c*)| < 1e-2|Ba|, the coupling is
     not this term and the refusal needs another cause; otherwise the cubic with T − 3S has a root c** with
     |g(c**)| < 1e-2|Ba|, or the task reports that it has none.
   - *Depends on.* #499 (`hp/dz0_weak.py`, `hp/weak_corrector.py`).
2. **A/B the Lyapunov–Schmidt corrector (design 2).**
   - *What.* Strong solve with span(V) frozen, then Newton on the four-dimensional g(c) from c = a, on win2's refused
     start D 242.5146 → 242.5459 (a 0.03125), with newton_od on the same system as the control.
   - *Why.* It is the design whose assumptions the data do not contradict.
   - *Acceptance.* First a planted system with a known cubic reduced root (planted σ, T and S, the same code path)
     converges to that root to 1e-3 relative, and its mutation, w set to 0, must fail to reach it. Then the real start
     converges below 1e-60 where newton_od is refused, or the task reports where it stops.
   - *Depends on.* Task 1, and σ₅ from task 3.
3. **Check the order-2 assumptions on the measured tensors.**
   - *What.* (a) σ₅ against σ₄ at z0. (b) The minimum over 1000 random unit t ∈ span(V) of σ_min/σ_max of the 4×4
     matrix ½T(t, t, ·). (c) Q at h down to 1e-29, where the truncation 6.3e60 h² falls below 1e3, and the mixed block
     UᵀF″(x, v) for a few strong directions x (task 1 gives the one that matters).
   - *Why.* They decide whether the weak singularity is regular of order 2 in Griewank's and DKK's sense, so that
     their rates apply away from the crossover, or irregular (Griewank and Osborne 1983).
   - *Acceptance.* A planted tensor with a known singular direction must show the zero in (b). (c) passes when Q(h) and
     Q(h/10) agree to 1e-2 or the truncation constant c h² stays below 1e3.
   - *Depends on.* #499.
4. **Read what did not open.**
   - *What.* Decker and Kelley (1985), Griewank (1985), Reddien (1979), Schnabel and Frank (1984), Deuflhard's book
     §3.3, Fischer, Izmailov and Solodov (2021), Allgower and Georg ch. 8.
   - *Why.* Design 2's standing and the F4 verdict rest partly on reviews.
   - *Acceptance.* Each is read and recorded. The task fails, and says so, if one of them analyses a corrector for a
     nearly singular root whose reduced quadratic term vanishes that the designs above miss.
   - *Depends on.* #527.

## Addendum (2026-11-12, issue #531): the near-singular Newton works #527 could not open

**Question.** What do the seven works that #527 saw only through records and reviews prove or propose for Newton's
method at or near a singular root, and does any of them analyse a corrector for a nearly singular root whose reduced
quadratic term vanishes, which #527's designs miss? Notation as in #527's addendum above (J, V, U, X, B, Q, T, a, S).
Since #527, #529 resolved Q at h = 10⁻³¹ (largest entry 5.4×10⁴ and 2.75×10⁴) and found σ₅ at least 1.2×10⁶ above σ₄
(notes/LARGE_D_WALL.md, "The octo-double wall near D 232–242").

**How.**
- **Searches.** 21 web searches on 2026-11-12, and record queries of Crossref, OpenAlex, Semantic Scholar and zbMATH
  Open (whose API returns the full review texts).
- **Read.** Fischer, Izmailov and Solodov (2021), the published text on Solodov's page (pages.cs.wisc.edu), in full.
  Schnabel and Frank as their report CU-CS-243-83 (April 1983, CU Scholar record np193b00z), in full; it is the
  journal paper's preprint, and the journal text was not compared. Standing in for Allgower and Georg's ch. 8: their
  *Numerical path following* (Handbook of Numerical Analysis 5, 1997), preprint of August 1994 from Georg's page
  (Wayback copy of 2010-06-16), §8.3 and §24. For its account of Griewank (1985), the passages of Mannel,
  arXiv:2111.12393v4 (2022), that cite it.
- **Not obtained.** Decker and Kelley (1985), Griewank (1985), Reddien (1979), Deuflhard's book and Allgower and
  Georg's book. SIAM and AMS returned 403, ScienceDirect a bot check (OpenAlex lists Reddien 1979 as free to read
  there), CORE 403, scholar.archive.org a session check, fatcat timed out, and DTIC showed a maintenance page. Kelley's
  NCSU publication list (Wayback copy of 2024-04-19) links no PDFs, and Georg's page offers only the book's programs.
  Copies of Deuflhard's chapter 3 and of Allgower and Georg's book appeared in search results only on a university
  library proxy and on a shadow-library mirror, and were not used.
- **Text channel.** PDFKit for the two typeset PDFs, Vision OCR for the report's scan. Quotes are from that text, not
  compared with the page images.

### The seven works

| Work | Read | Bears on | What it contains | What it does not contain |
|---|---|---|---|---|
| **D. W. Decker, C. T. Kelley**, *Expanded convergence domains for Newton's method at nearly singular roots*, SIAM J. Sci. Stat. Comput. 6 (1985) 951–966 | not obtained | F1, F3, design 2 | zbMATH 0597.65057 (B. Döring), whole review: for F from a Banach space into itself, "Newton's method for the approximate solution of the equation (*) F(x)=0 when the derivative of F is nearly singular at a zero of (*). They show that there is a larger than expected domain of attraction for the Newton iterates but that convergence appears to be only linear in most of this domain. A modification of Newton's method is given that improves this slow convergence. Two examples are considered: A 2×2 system and a discretized version of the Chandrasekhar H-equation." No text read describes it further: the "[DK85]" of Mannel (arXiv:2111.12393) is their Broyden paper, SIAM J. Numer. Anal. 22 (1985) 566–574. | Unknown: its class of near singularity, whether the reduced quadratic term may vanish, the modification, and any rate. |
| **A. Griewank**, *On solving nonlinear equations with simple singularities or nearly singular solutions*, SIAM Rev. 27 (1985) 537–563 | not obtained | F2, F4, designs 1 and 2 | zbMATH 0598.65026 (E. L. Allgower), whole review: a regular singular zero x* of order k; "a theorem is given characterizing the behavior of Newton's method near x* if the order of the singularity is k", which "makes it possible to empirically estimate k", motivating Richardson extrapolation; overrelaxation; bordering, "f(x)−λr=0 where rank (∇f,r)=n near x* and an appropriate normalizing equation (2) γ(x)=0. Since the singularity is thereby unfolded, one has that Newton's method converges locally quadratically to any zero-point of the resulting bordered map F(x,λ)", "principally used for simple singularities (although extensions to higher singularities can be formulated too)"; quadratic models h(s) = f(x) + Df(x)s + r sᵀBs after Schnabel and Frank; tests on an 80-point tubular reactor, where "If a regular solution is indeed present, then the bordering method yields very rapid convergence." Read texts' accounts: Newton converges along a single null direction at ratio 1/2 ([8, Thm 2.1] in Fischer, Izmailov and Solodov 2021), the doubled Newton step converges at ratio 1/4 ([8, Thm 4.1], there), a "simple singularity" is a regular first-order one with dim N = 1 ([Gri85, (2.7)] in Mannel), and its 1983 preprint's quadratic model is "locally 2-step q-superlinearly convergent" at rank n−1 using "information about the singularity in F′(x*) that would not be available to general purpose nonlinear equations solvers" (Schnabel and Frank, §7). | Visible only in outline: its treatment of nearly singular solutions, bordering for dim N > 1 or order 2, and the strong rows' curvature. |
| **G. W. Reddien**, *Newton's method and high order singularities*, Comput. Math. Appl. 5 (1979) 79–86 | not obtained | F2 | zbMATH 0436.65032 has no review. Only DKK's and Griewank's accounts (#527's row above). Schnabel and Frank's list also dates it 1980. | (unread) |
| **R. B. Schnabel, P. D. Frank**, *Tensor methods for nonlinear equations*, SIAM J. Numer. Anal. 21 (1984) 815–843 | report CU-CS-243-83, in full | designs | Abstract: "general purpose methods intended especially for problems where the Jacobian matrix at the solution is singular or ill-conditioned". **Model.** M_T(x_c + d) = F(x_c) + F′(x_c)d + ½Σ_{k≤p} a_k(s_kᵀd)² (3.13), the term of least Frobenius norm that interpolates F at p ≤ √n past iterates (3.3), (3.5), Thm 3.2. **Solve.** Algorithm 4.1 reduces the model to q quadratic equations in p unknowns and n − q linear ones; "when the Jacobian is singular, the effect of the tensor algorithm usually is to isolate this rank deficiency as the linear term in a small system of quadratic equations" (§4). A line search on ‖F‖₂, with the Newton or a Levenberg–Marquardt step as fallback (Alg. 5.2). **Tests.** At rank n−1 Newton's ratio tends to 0.5 and the tensor method's last ratio is about 0.01 (Table 6.6); the gains are smaller at rank n−2 (Tables 6.3, 6.4). **Limits stated.** "It would be very helpful to obtain local convergence results for our tensor algorithm applied to singular problems" (§7); for a null space of dimension m > 1 the term must carry "((m²+m)/2) n-vectors of information" and "our method does not seem to interpolate enough information" (§7); for minimization at a singular Hessian "an approximation not to the third but to the fourth derivative matrix is necessary to speed convergence. This is because the projection of the third derivative onto the null space of the Hessian must be zero at such a minimizer" (§7). | Any convergence proof; nearly singular roots in theory; a third-order term; curvature between span(s_k) and its complement, since the term depends on d only through the s_kᵀd. |
| **P. Deuflhard**, *Newton Methods for Nonlinear Problems: Affine Invariance and Adaptive Algorithms*, Springer Ser. Comput. Math. 35 (2004), §3.3 | not obtained | F4 | zbMATH 1056.65051 (W. C. Rheinboldt): affine covariant, contravariant and conjugate algorithms "controlled by error norms, residual norms, and energy norms", globalization including "Newton's method with damping strategies", and "augmented systems for determining selected critical points". Table of contents (the publisher's pages, GBV copy): §3.3 Error Oriented Descent, p. 134, with 3.3.1 General level functions, 3.3.2 Natural level function (138), 3.3.3 Adaptive trust region strategies (145), 3.3.4 Inexact Newton-ERR methods (152); "Adaptive rank strategies" are §4.3.5 (p. 215), under Gauss–Newton methods; §5.3 Computation of Simple Bifurcations (p. 263). | (the text, so its monotonicity tests and its treatment of rank deficiency stay unread) |
| **A. Fischer, A. F. Izmailov, M. V. Solodov**, *Unit stepsize for the Newton method close to critical solutions*, Math. Program. 187 (2021) 697–721 | in full | F2, F4, designs 1 and 2 | **Setting.** Φ twice differentiable with Φ″ Lipschitz at a singular, possibly nonisolated, solution ū, and 2-regular in some v̄ ∈ ker Φ′(ū): B(v) = ΠΦ″(ū)[v] restricted to ker Φ′(ū) is nonsingular, Π the projector onto (im Φ′(ū))⊥. **Line search.** Algorithm 1 backtracks on ‖Φ(u + αv)‖ ≤ (1 − σα)‖Φ(u)‖ (12). Thm 1: from a starlike set U whose excluded directions are those where 2-regularity fails in v or π(v), with σ ∈ (0, 3/4), the unit step is accepted for all large k and the iterates converge at ratio 1/2 along a null direction. **Where full steps are refused.** Lemma 2: the unit step is accepted where ‖u₁ − ū₁‖ ≥ Γ̄‖u₂ − ū₂‖² (any σ < 1) or ‖u₁ − ū₁‖ ≤ γ̄‖u₂ − ū₂‖² (σ < 3/4), u₁ the component on (ker)⊥ and u₂ on ker; between lies the "troublesome" set W (29), with points arbitrarily close to ū, which the iterates leave after finitely many steps (Lemma 5). Example 3 enters W in one step, "caused solely by the presence of the mixed term u₁u₂ in the second component", and "entering W can be avoided if, say, the restriction of ΠΦ″(ū)[u⁰] on (ker Φ′(ū))⊥ is identically zero". **One step from a strong start.** Lemma 7: from u⁰ = ū + tv, "u⁰ + v⁰ = ½tπ(v) + O(t²)" (37), with π(v) = v₂ + B(v)⁻¹ΠΦ″(ū)[v, v₁] (6); Lemma 8: if v ∉ ker the unit step is accepted for small t. Both assume 2-regularity in v only (and π(v) ≠ 0 for Lemma 7). Lemma 1 (4): the (ker)⊥ error after a step is O(‖ΠΦ″(ū)[u − ū, u₁ − ū₁]‖) + O(‖u − ū‖³). **Q = 0.** Example 5, Φ = (u₁(u₁² + u₂), u₂(1 + u₂)), "is not 2-regular at ū in any direction from ker Φ′(ū)"; Newton converged in all 100 runs, the trailing full steps making up 94.4–100% of each run (Table 1), and extrapolation gave "no accelerating effect". | Nearly singular roots; any theorem without 2-regularity; higher-order singularities (Example 5 is numerical only); an acceptance test other than the residual norm. |
| **E. L. Allgower, K. Georg**, *Numerical Continuation Methods: An Introduction*, Springer Ser. Comput. Math. 13 (1990), ch. 8 "Detection of Bifurcation Points Along a Curve", pp. 75–90 | not obtained; their 1997 chapter read in its place | F4, design 2 | From *Numerical path following* (1994 preprint), whose "philosophy and notation ... will be that of our book": a simple bifurcation point has dim ker H′(ū) = 2 and ē*H″(ū) on that kernel with one positive and one negative eigenvalue (24.1). The proof of Thm 24.1 is a Lyapunov–Schmidt reduction: the strong rows H₂(u₁, u₂) = 0 give u₂ = φ(u₁) by the implicit function theorem (24.3), and b(u₁) = H₁(u₁, φ(u₁)) = 0 "is called the bifurcation equation" (24.4), with φ′(ū₁) = 0 and b″(ū₁) = ∂₁²H₁(ū). The augmented determinant changes sign there (Thm 24.2). "Keller [1977] has shown that for sufficiently small steplength h, the predictor point will fall into the cone of attraction of the Newton corrector ... See also Allgower and Georg [1990, Lemma (8.1.16)]", and "it is necessary to obtain a good approximation of the Jacobian at least once at the predictor point since otherwise the local convergence of the Newton corrector iterations cannot be guaranteed when jumping over a simple bifurcation point. This is explained in Allgower and Georg [1990, p. 84]." Branching by perturbation, H(u) = d (§24.1), and Newton on a minimally extended system whose 2×2 matrix is M_ij = ψ*H″(u)[v_i, v_j] (Algorithm 24.1). | Ch. 8 itself; a singular b″ (a vanishing reduced quadratic term); rates; nearly singular roots other than the perturbed equation H(u) = d. |

### Against #530 and #499's F4

- **Verdict.** No work read analyses a corrector for a nearly singular root whose reduced quadratic term vanishes, so
  the task's fail condition is not met by any text read. Fischer, Izmailov and Solodov's theorem and Allgower and
  Georg's bifurcation algorithm both need a nonsingular reduced second derivative (2-regularity, and b″ nonsingular),
  and Schnabel and Frank prove nothing. The check is incomplete: Decker and Kelley (1985) and Griewank (1985), the two
  works whose titles name nearly singular roots, are not obtained, and their reviews leave this case open.
- **F4's feedback, at an exactly singular root.** Fischer, Izmailov and Solodov's (37) specializes, for a start error tx
  with x in X, to π(x) = B(x)⁻¹ΠΦ″(ū)[x, x], which lies in ker Φ′(ū) (*inferred*). So a Newton step from a point with no
  weak error removes the strong error and makes a weak one of the same order t, through their B(x), which in #499's
  notation is the weak-left curvature with one strong argument, the 4×4 matrix UᵀF″(x, V·). That is #499's weak-first
  observation (the next step lies on the weak directions), and their Example 3 names the same mixed term as what sends
  an iterate into the set where the residual test refuses full steps. Lemmas 7 and 8 assume 2-regularity in the start
  direction only, which for x in X is the nonsingularity of that mixed block, not of Q (*inferred*). Neither covers a
  nonzero σ or newton_od's test.
- **Design 2.** Allgower and Georg's bifurcation equation b(u₁) = H₁(u₁, φ(u₁)) is design 2's reduced map g(c), with
  the strong rows solved for fixed weak coordinates. Because c is held during the strong solve, the regeneration in
  (37) does not act inside it; it enters only through g, where it is the term S (*inferred*). Schnabel and Frank's
  Algorithm 4.1 orders the solve the same way inside a model, but its second-order term has no curvature between
  span(s_k) and the rest, so it cannot carry S, and with dim N = 4 a quadratic term needs 10 n-vectors and a cubic one
  20 (*inferred*, their count (m² + m)/2 extended to third order). Their remark on minimizers, that a vanishing
  projected derivative calls for the next one, gives the same reason a second-order model cannot capture T
  (*inferred*, by analogy).
- **Design 1.** Unchanged: the accelerations need the unit step to be accepted (Fischer, Izmailov and Solodov, §3), and
  on their Example 5, where 2-regularity fails, extrapolation gave no gain.
- **2-regularity is open again (discrepancy, reported).** #527's F2 row says that Q = 0 "rules out 2-regularity in every
  direction". #529 has since resolved Q, so whether the 4×4 matrix Q(t, ·) is nonsingular for some t in span(V) is open
  and measurable. If it is, the exactly singular theory to compare with is the first-order one of Fischer, Izmailov
  and Solodov (ratio 1/2, unit steps for σ < 3/4), not order 2 (*inferred*). Not changed in #527's text.

### References checked

| Entry | Checked against | Verdict |
|---|---|---|
| Decker, Kelley, SIAM J. Sci. Stat. Comput. 6(4), 951–966 (1985), doi:10.1137/0906064 | Crossref; zbMATH 0597.65057; Kelley's publication list (Wayback 2024-04-19) | OK (bibliographic) |
| Griewank, SIAM Rev. 27(4), 537–563 (1985), doi:10.1137/1027141 | Crossref; zbMATH 0598.65026; reference [8] of Fischer, Izmailov, Solodov (2021) | OK (bibliographic); Schnabel and Frank cite its 1983 Southern Methodist University preprint, not seen |
| Reddien, Comput. Math. Appl. 5(2), 79–86 (1979), doi:10.1016/0898-1221(79)90061-0 | Crossref; zbMATH 0436.65032 | OK (bibliographic); MINOR: Schnabel and Frank's list dates it 1980, as DKK's does |
| Schnabel, Frank, SIAM J. Numer. Anal. 21(5), 815–843 (1984), doi:10.1137/0721054; report CU-CS-243-83 (April 1983) | Crossref; the report's title page (CU Scholar np193b00z) | OK; content from the report |
| Deuflhard, *Newton Methods for Nonlinear Problems*, Springer Ser. Comput. Math. 35 (2004), xii+424 pp., ISBN 3-540-21099-7 | zbMATH 1056.65051; the table of contents (GBV) | OK (bibliographic); neither zbMATH nor Crossref gives the 2004 edition a DOI (the 2011 printing has doi:10.1007/978-3-642-23899-4) |
| Fischer, Izmailov, Solodov, Math. Program. 187(1–2), 697–721 (2021; online 2020-04-07), doi:10.1007/s10107-020-01496-z | Crossref; the article's first page | OK |
| Allgower, Georg, *Numerical Continuation Methods: An Introduction*, Springer Ser. Comput. Math. 13 (1990), doi:10.1007/978-3-642-61257-2, ch. 8 doi:10.1007/978-3-642-61257-2_8; SIAM Classics in Applied Mathematics 45 (2003), doi:10.1137/1.9780898719154, ch. 8 doi:10.1137/1.9780898719154.ch8, pp. 75–90 | Crossref; Georg's publications page (Wayback) for the series number | OK (bibliographic) |
| Allgower, Georg, *Numerical path following*, Handbook of Numerical Analysis 5 (1997) 3–207, doi:10.1016/S1570-8659(97)80002-6 | Crossref; Georg's publications page; the preprint's title page (August 1994) | OK; content from the 1994 preprint |

### Gap statement

- **Not obtained.** Decker and Kelley (1985), Griewank (1985), Reddien (1979), Deuflhard's book and Allgower and
  Georg's ch. 8. Closing them needs library access.
- **The residual risk** is the first two. Decker and Kelley give "a modification of Newton's method" at nearly singular
  roots, and Griewank's bordering "unfolds" the singularity, converging quadratically to a regular solution. Either
  could be a corrector of #530's kind, or rule one out.
- **No novelty sentence** was added by this task.

### Recommended tasks (for the coordinator to file)

1. **2-regularity of Q and of the mixed block at the D 242.5146 starts.**
   - *What.* From #529's Q at h = 10⁻³¹, the minimum over 1000 random unit t in span(V) of σ_min/σ_max of the 4×4
     matrix Q(t, ·), and the same ratio for the 4×4 matrix UᵀF″(x, V·) at the strong directions x of #529 (c).
   - *Why.* Q(t, ·) decides whether the exactly singular comparison is Fischer, Izmailov and Solodov's first-order
     theory or DKK's order 2, and UᵀF″(x, V·) whether their (37) applies to the weak-first step.
   - *Acceptance.* A planted tensor with a known singular direction must show the zero below 10⁻¹⁰, and the verdict
     must agree between h and h/10.
   - *Depends on.* #529.
2. **Plant F4's mechanism in #530's test system.**
   - *What.* Add to #530's planted system a mixed term in the weak rows, as in Fischer, Izmailov and Solodov's
     Example 3, beside the planted σ, Q, T and the strong-row F″(v, v).
   - *Why.* (37) predicts the weak error regenerated after a weak-first step, ½tπ(x), so the planted run tests the
     account of F4 before the real start does.
   - *Acceptance.* With σ set to 0 and the mixed term on, the weak error after one Newton step from the weak-first
     point matches ½tπ(x) to 10⁻². With the term set to 0, it falls below 10⁻² of that. If either check fails, the
     mixed-term account of F4 fails. Then the planted σ is restored and newton_od's verdict is reported.
   - *Depends on.* #528, #530.
3. **Read Decker and Kelley (1985) and Griewank (1985) through library access.**
   - *What.* Nathan obtains the two papers (and, if possible, Reddien 1979, Deuflhard §3.3 and Allgower and Georg
     ch. 8), and a subagent records them here.
   - *Acceptance.* As #531's: the task fails, and says so, if one of them analyses a corrector for a nearly singular
     root whose reduced quadratic term vanishes that #527's designs miss.
   - *Depends on.* #531.

## Addendum (2026-11-12, issue #380): the 2025 critical-phenomena review, read in full

**Question.** What does Gundlach, Hilditch and Martín-García's 2025 review contain, and not contain, on the topics of
the claims C1–C10 of §0 and on our other results? §1's row for it was read through `WebFetch` extractions, and the
#200 addendum read its first 100000 characters and a keyword search. The statement-by-statement comparison with our
values is
[REVIEW_GHMG25.md](REVIEW_GHMG25.md), which is the canonical place for it; this entry records what was read.

**How.**
- **Read in full.** The arXiv source of v1 (`critphen_lrr_maintext.tex`, all 5122 lines, and the `.bbl`), downloaded
  with curl on 2026-11-12. Quotes are verbatim from the LaTeX source, with math rewritten in Unicode, so this channel
  has no extraction limit, unlike the `WebFetch` quotes of §1.
- **Abstracts read** (arXiv pages): Garfinkle–Cutler–Duncan 1999, Birukou et al. 2002, Husain et al. 2003,
  Bland–Kunstatter 2007, Gundlach 1998, Martín-García–Gundlach 1999, Brady–Ottewill 1998, Berczi–Saffin–Zhou 2022.
- **Source read in part:** Porto Veronese–Gundlach 2022 (arXiv source), for its D = 5 values and its 8+1 appendix.
- **Inferences** are marked *inferred*.

### Closest works

| Work | Bears on | What it contains | What it does not contain |
|---|---|---|---|
| **C. Gundlach, D. Hilditch, J. M. Martín-García**, *Critical Phenomena in Gravitational Collapse*, arXiv:2507.07636v1 (2025), read in full | C6, C9, C10; the mode count; matter/ | **D = 4.** "Δ = 3.445452402(3) from a numerical construction of the critical solution based on exact self-similarity and analyticity", which "agrees with the leading digits" of Reiterer and Trubowitz's (III B). A critical solution has "exactly one λᵢ with positive real part (in fact it is purely real)", and the λ = 1 time shift is unphysical (II C); all non-spherical modes of the Choptuik solution decay (V A 1). **Other D.** γ ≃ 0.424 at D = 6, γ = 0.4131 ± 0.0001 at D = 5, Birukou et al.'s 0.36, 0.41, 0.44 and 0.52, Kol's estimates, and work "in noninteger dimension up to 14" and in 8+1 (IV A 2); 2+1 with Λ < 0 (IV A 1). **Proofs.** Reiterer–Trubowitz's existence proof, "the only DSS one so far" (VII C), from a Fourier × Chebyshev approximation in rational arithmetic, analytic "from Γ through to somewhat to the future of C₀⁻" (VII D 7); CSS Einstein–scalar solutions are only $`C^{1,\epsilon}`$ at the past light cone (VII D 3). **Surface gravity.** ν_SH,Choptuik ≃ 0.6138 from Reiterer–Trubowitz's μ and Δ (VII G, Eq. (125)). **Counting.** Gundlach 1998 counted the fluid solution's modes "using numerical contour integration in the complex plane of potential mode frequencies" (VI E). **Matter.** The κ-model (critical CSS for κ > 0.0754, three unstable modes below), the S³ σ-model's thresholds and its CSS–DSS transition, the ℓ-multiplet, Yang–Mills, Bianchi IX, charge (δ ≃ 0.88) and ghosts (IV A 3–5, V A 2, III E, IV D). | Large D in any form: Rozali–Way, Emparan–Herzog and Clark–Pimentel are not cited, and Ecker et al.'s papers appeared after it. Δ at any D ≠ 4. The limit D → 3⁺. Any formula or mechanism for Δ; Stokes phenomena or exponential asymptotics. Mode counts at D ≠ 4. Ghost or wormhole thresholds. |

### Verdicts on the claims of §0

- **C1–C5, C8.** The review has nothing on large D, so it neither supports nor contradicts them, and §2's verdicts
  stand. C3's "Not found" list already counted it (as "the 2025 review: none"); the full reading confirms it.
- **C6.** No Δ(D) is quoted for D ≠ 4, and no limit of Δ or γ. The higher-D work it cites (Sorkin–Oren, Bland et al.,
  Taves–Kunstatter, Porto Veronese–Gundlach) stops at D = 14. C6's implicit priority above D = 14 stands as far as the
  review goes.
- **C7.** The review predates Ecker et al.'s D = 5 value, so it is silent.
- **C9.** Survives as worded, since the sentence is scoped to the two Stokes-constant templates. Reiterer–Trubowitz's
  unknown is also a function of (τ, x), periodic in τ (VII D 7), in a computer-assisted proof without Stokes
  multipliers. Proposed: name them in PROOF.md's sentence as the two-variable precedent, so its scope reads plainly.
- **C10.** The review confirms §2's verdict: Reiterer–Trubowitz's proof is the existence proof of the D = 4 DSS
  solution, with Δ enclosed. PROOF.md's wording already follows that verdict.

### References checked

| Entry | Checked against | Verdict |
|---|---|---|
| Gundlach, Hilditch, Martín-García, arXiv:2507.07636 (2025) | the arXiv abstract page (v1 only, 10 July 2025, comment "Major update to the living review"); INSPIRE, 2026-11-12 | OK as a preprint; no journal publication listed |
| Birukou, Husain, Kunstatter, Vaz, Olivier, PRD 65, 104036 (2002), gr-qc/0201026 | INSPIRE | OK |
| Porto Veronese, Gundlach, PRD 106, 104044 (2022), arXiv:2209.08404 | INSPIRE | OK |
| Martín-García, Gundlach, PRD 59, 064031 (1999), gr-qc/9809059 | INSPIRE; the arXiv page | OK |
| Gundlach, PRD 57, R7075 (1998), gr-qc/9710066 | INSPIRE (DOI 10.1103/PhysRevD.57.R7075) | OK; the review's entry gives page 7075 |
| Gundlach, Martín-García, PRD 54, 7353 (1996), gr-qc/9606072 | INSPIRE | OK |
| Berczi, Saffin, Zhou, JHEP 02 (2022) 183, arXiv:2111.11400 | INSPIRE | OK |
| Brady, Ottewill, PRD 58, 024006 (1998), gr-qc/9804058 | INSPIRE | OK |
| Marouda, Cors, Rüter, Atteneder, Hilditch, PRD 109, 124042 (2024), arXiv:2402.06724 | INSPIRE | OK; the review cites the preprint only |
| Cors, Renkhoff, Rüter, Hilditch, Brügmann, PRD 108, 124021 (2023), arXiv:2308.01812 | INSPIRE (the APS DOI) | MINOR: INSPIRE's free-text field says 124035 |

### Gap statement

- **Not read.** The primary sources behind the review's numbers: Garfinkle–Cutler–Duncan 1999 and Birukou et al. 2002
  beyond their abstracts, Hirschmann–Eardley 1997, Bizoń–Wasserman 2000, Gundlach 1998's method, the axisymmetric
  complex-scalar papers. The review's statements stand in for them, which is the residual risk of REVIEW_GHMG25.md's
  rows 6, 9–11, 22, 24 and 29.
- **The published version.** The review invited feedback until October 2025 for "the version to be published", which
  was not available on 2026-11-12. A later version may change any row.

### Discrepancies and proposed text (for the coordinator)

- **This file's Q3 (addendum #200).** "MODE_COUNT.md's argument-principle count of det M(λ) has no counterpart in the
  sources read" is contradicted by the review's Sec. VI E (Gundlach 1998, contour integration). Reported, not changed;
  REVIEW_GHMG25.md D6 and its task R1. #619 read the 1998 paper, which has no such count, and found one in
  Hirschmann–Eardley 1995 and in Gundlach 1997 (addendum of issue #619).
- **§1's row for the review** is superseded by this addendum. Its content agrees with what was read here.
- The comparison's seven discrepancies, the citations to add and the proposed README text are in
  [REVIEW_GHMG25.md](REVIEW_GHMG25.md) §3, §5 and §7.
- **No novelty sentence** was added by this task.

### Recommended tasks (for the coordinator to file)

REVIEW_GHMG25.md §6 (R1–R5).

## Addendum (2026-11-13, issue #619): Gundlach 1998's mode count, and the determinant counts before it

**Question.** The 2025 review (Sec. VI E) says Gundlach 1998 counted the fluid critical solutions' modes "using
numerical contour integration in the complex plane of potential mode frequencies". Does it? If so, the Q3 sentence of
addendum #200, that MODE_COUNT.md's argument-principle count "has no counterpart in the sources read", is wrong
(REVIEW_GHMG25.md, D6).

**How.**
- **Sources.** The arXiv e-print sources, fetched with curl on 2026-11-13. Quotes are verbatim from the LaTeX with math
  in Unicode, and section numbers are RevTeX's.
  - Read in full: gr-qc/9710066 v1 (`codim.tex`, dated 12 October 1997) and v2 (`codim_prd.tex`, "revised 14 March
    1998"), diffed line by line.
  - Read in part: Gundlach 1997, gr-qc/9604019 (Sec. IV A and Apps. B and E). Gundlach 2002, gr-qc/9906124 v2 (the
    abstract and Secs. I, IV A, IV B and V). Hirschmann–Eardley 1995, gr-qc/9506078 v3 (Secs. VI and VII), whose ideas
    Gundlach 1997's App. E says it generalizes. The rest of each was searched for "contour", "winding", "residue",
    "pole" and "det".
  - The review's Secs. VI E and V B, and its `.bbl` entries `Gundlach_nonspherical` (PRD 57, 7075, gr-qc/9710066) and
    `Gundlach_critfluid2` (PRD 65, 084021, gr-qc/9906124), in its source.
- **Text addressed to an AI agent.** None in the four papers.
- **Sign conventions.** Gundlach 1997 puts the singularity at τ = −∞, so his growing modes have Re λ < 0.
  Hirschmann–Eardley write e^{−iστ}, so theirs lie in the lower half σ plane. Each value is quoted in its paper's
  convention.

### Closest works

| Work | Bears on | What it contains | What it does not contain |
|---|---|---|---|
| **C. Gundlach**, *Nonspherical perturbations of critical collapse and cosmic censorship*, PRD 57, R7075 (1998), gr-qc/9710066, v1 and v2 read in full | the review's VI E; MODE_COUNT.md | **Matter.** The perfect fluid p = ρ/3 only ("we have carried out the numerical calculations only for c_s² = 1/3"). **Method** (v2, the paragraph "Perturbation method"): "To answer this yes/no question, we write the perturbation equations as evolution equations in the time variable τ. [...] We evolve generic initial data for these equations for a sufficiently large interval of τ." Then "the mode with the largest Reλ takes over after a transition period, and both that λ and the corresponding f(x) can be simply read off from the late-time data", Koike, Hara and Adachi's method. v1 says "In this Letter, we do not determine all perturbation eigenvalues individually". **What was evolved.** l = 0 as a check ("γ ≃ 0.36"), and the odd and even l ≥ 1 perturbations in Gerlach and Sengupta's gauge-invariant variables, one (l, m) at a time, by a first-order upwind scheme along the characteristics (paragraph "Numerical method"). The even l = 1 perturbations are set aside as "pure gauge". A geometric-optics argument covers the radial frequencies the grid does not resolve, and large l. **Result.** "We find that the only growing mode is the previously known spherical one." | A contour integral, a determinant, or a count of modes in a region of the λ plane. The method finds only the dominant mode of each sector. Any growth rate for l ≥ 1, any table or figure of rates (its one figure is the numerical domain), the list of the l evolved, and any c_s² ≠ 1/3. Gundlach 2002 (Sec. I) corrects two of its assumptions, "that the l = 1 axial perturbations obey the same type of equation as the l ≥ 2 ones, and that the l = 1 polar perturbations are pure gauge", without changing its conclusion. |
| **E. W. Hirschmann, D. M. Eardley**, *Critical exponents and stability at the black hole threshold for a complex scalar field*, PRD 52, 5850 (1995), gr-qc/9506078 v3, Secs. VI–VII | MODE_COUNT.md's method; matter/COMPLEX_CRITICAL.md | **The determinant** (Sec. VI). Two-point shooting to a midpoint gives "A(σ) X = 0 where A(σ) is a 6×6 complex matrix", and the modes are the zeros of det A(σ). A is analytic in σ, with "no singularities in the closed lower half σ plane". **The count.** "there is a theorem for counting the number N_C of zeros of det A within any closed contour C [...]: Δ_C Arg det A = 2π N_C", "a result similar to Levinson's theorem". Cauchy–Riemann spot checks agree to about 10⁻⁶. **Coverage** (Sec. VII). "a large rectangular contour C bounded by 0 ≤ Re σ ≤ 25, 0 ≥ Im σ ≥ −10 (detouring around the three zeros on the imaginary axis)", where "det A(σ) ∼ exp(τ∞σ), τ∞ ≈ 6i". **Result.** Five zeros in the lower half plane for the D = 4 complex-scalar CSS solution: the two gauge modes at σ = 0 and −i, "at the expected places", the black-hole mode σ = −2.58324i, and the pair σ = ±3.50224 − 0.25867i. | Poles (A has none there). A planted zero or a box that must count a known number, beyond the two gauge modes. D ≠ 4, non-spherical perturbations, DSS solutions. |
| **C. Gundlach**, *Understanding critical collapse of a scalar field*, PRD 55, 695 (1997), gr-qc/9604019, Sec. IV A and Apps. B and E | MODE_COUNT.md's method and its poles | **The determinant** (App. E). Shooting from both ends to a midpoint gives "a complex square matrix d(mismatch at the midpoint) / d(free parameters at the endpoints)", A(λ): "We therefore have to search for zeros of det A(λ_i) in the complex plane". Holomorphy and det A(λ*) = det A(λ)* are checked numerically, to relative errors 10⁻⁹ and 10⁻¹⁰. **The count.** "Because det A(λ_i) is holomorphic, we can use the well-known contour integral ∫A′/A dλ_i = 2πi (N_z − N_p) to count the number of zeroes minus the number of poles within the contour." Both codes "generalize ideas already used in similar calculations", Hirschmann and Eardley's. **Poles** (App. B). The light-cone conditions have no periodic solution where an average coefficient vanishes. That gives λ = −1 for perturbations of the background's symmetry and λ ≃ −0.385, "the average of (1+ξ₀′)(1−a₀²)", for the opposite one. **What was counted.** The spherical perturbations of the D = 4 Choptuik solution, both symmetry sectors, in 0 ≤ Im λ < π/Δ (Sec. IV A). **Result** (Sec. IV A). One growing mode, real, "λ₁ = −2.674 ± 0.009", so γ = 0.374 ± 0.001; none of the opposite symmetry; the τ translation at λ = 0. | The extent of its contours. No box or region is stated beyond "the left half plane" of the result. Any deliberate check of the count. Non-spherical perturbations, other matter and D ≠ 4. App. E prints the strip as "0 ≤ Im λ_i < Δ/π", where Sec. IV A has π/Δ. |
| **C. Gundlach**, *Critical gravitational collapse of a perfect fluid: nonspherical perturbations*, PRD 65, 084021 (2002), gr-qc/9906124 v2, read in part | the review's VI E and V B | The fluid for 0 < κ < 1 and l ≤ 5. **Methods** (Sec. IV B). Of the three it names, 1b) is the determinant: "One therefore looks for zeros of the complex-analytic function det A(λ). This can be done by contour integrals [Gundlach 1997]." It uses the other two. "We first implemented the Lyapunov method, method 2)", which "worked well for the κ = 1/3 fluid [Gundlach 1998]", and then "both method 3a) and 3b)", the eigenvalues of the evolution operator discretized in x, with four difference schemes, convergence tests and independent residual evaluation. **Results** (abstract). One growing spherical mode for every κ. No growing non-spherical mode for 1/9 < κ ≲ 0.49, a growing l = 1 axial mode for κ < 1/9, and several growing non-spherical modes for κ ≳ 0.49. | A contour integration. Method 1b) is described, with Gundlach 1997 as its reference, and not used. "contour" occurs once in the source. |

### Verdicts

- **Gundlach 1998 counts no modes by contour integration.** It decides whether each (l, parity) sector has a growing
  mode by evolving generic data and reading off the dominant one. So the review's VI E sentence is not supported by the
  paper it cites, and the task's stop condition holds: the paper counts modes some other way (REVIEW_GHMG25.md, D6).
  Its successor, Gundlach 2002, names contour integration and uses other methods.
- **The determinant counts are for scalar fields.** Hirschmann–Eardley 1995 count the zeros of their shooting
  determinant by its phase wrap around a stated rectangle, for the complex-scalar CSS solution. Gundlach 1997 counts
  the zeros minus the poles of his, for the Choptuik solution. That the review's sentence conflates them with the fluid
  papers is *inferred*.
- **The Q3 sentence of addendum #200 is contradicted.** Both scalar-field counts are MODE_COUNT.md's method in outline:
  a shooting mismatch matrix whose determinant is holomorphic and conjugate-symmetric in λ, vanishes at the growth
  rates, and is counted by the argument principle. Gundlach 1997 also accounts for poles traced to the light-cone
  conditions, and Hirschmann–Eardley record the exponential growth of det A that MODE_COUNT.md divides out.
  MODE_COUNT.md adds stated boxes, planted and deliberate checks, multiple shooting with an exact Jacobian, and D from 4
  to 61 (*inferred* comparison; the codes share no machinery).
- **One pole is the same** (*inferred*, arithmetic only, not recomputed). Gundlach 1997's opposite-symmetry pole,
  λ ≃ −0.385 in his convention, sits where MODE_COUNT.md §1's kk = 0 horizon resonance does at D = 4,
  n/2 − λ_g = 1 − 0.6139 = 0.386, on the other sector's real axis (Im λ = ω in the folded layout). The f64 code's
  resonance at λ = 0.386 (MODE_COUNT.md §2) is the same. Both come from a solvability condition at the light cone, and
  in our sign of λ both are minus the average of 1 − a² there (his weighted by 1 + ξ₀′). His other pole, λ = −1,
  would sit at λ = 1 here, where the D = 4 real scan finds no sign change of det M (only λ_g and λ_phys change its
  sign on 0.02 < λ < 6.47, MODE_COUNT.md).

### References checked

In [REF_CHECK.md](REF_CHECK.md), addendum of issue #619. All four are OK. The 1998 paper's DOI is
10.1103/PhysRevD.57.R7075, and the form without the R, which arXiv's DOI field and the review's entry give, does not
resolve.

### Gap statement

- **Not read.** The published APS texts (v2 of the 1998 paper is the version "accepted for publication"). The parts of
  Gundlach 1997, Gundlach 2002 and Hirschmann–Eardley 1995 not named above, which were searched by keyword only.
  Koike–Hara–Adachi's PRD 59, 104008 (1999), the "Lyapunov method" of Gundlach 2002.
- **Not searched.** Determinant winding counts outside critical collapse (self-similar blowup, quasinormal modes)
  before 1995; Hirschmann–Eardley cite Levinson's theorem only as a similar result.

### Discrepancies and proposed text (for the coordinator)

- **The review against its source** (REVIEW_GHMG25.md, D6). VI E credits Gundlach 1998 with a contour-integration
  count of the fluid solutions' spherical and non-spherical modes. That paper evolves generic data, and its successor
  uses the Lyapunov method and the eigenvalues of the discretized operator. V B also cites Gundlach 1998 for "a mode
  ansatz" and for the ranges 0.49 ≲ k < 1, 1/9 < k ≲ 0.49 and 0 < k < 1/9, which are Gundlach 2002's results, while
  1998 has κ = 1/3 only. The review appears to cite 1998 where 2002 is meant (*inferred*).
- **This file's Q3 sentence** (addendum #200) is noted where it stands, not rewritten. Proposed replacement:
  > MODE_COUNT.md's argument-principle count of det M(λ) has two precedents in critical collapse. Hirschmann and
  > Eardley count the zeros of their shooting determinant around a rectangle for the complex-scalar CSS solution (PRD
  > 52, 5850 (1995), Secs. VI–VII), and Gundlach counts the zeros minus the poles of his for the D = 4 Choptuik
  > solution (PRD 55, 695 (1997), App. E).
- **MODE_COUNT.md §2**, at "The function whose phase is followed", a credit for the method (AGENTS.md, standard
  methods). Proposed:
  > Counting the zeros of a shooting determinant by its winding goes back to Hirschmann and Eardley (PRD 52, 5850
  > (1995), Sec. VI) and Gundlach (PRD 55, 695 (1997), App. E), whose poles also come from the light-cone conditions
  > (his App. B).
- **The README and the physics paper** name the argument-principle count (README, "Checked at D = 30 and 45";
  `paper/physics/main.tex`, "One unstable mode") without a reference for the method. Gundlach 1997 is already README
  [^3] and `\bibitem{Gundlach1997}`, and Hirschmann–Eardley 1995 is cited only in matter/. Proposed: cite both where the
  count is named.
- **No novelty sentence** was added by this task.

### Recommended tasks (for the coordinator to file)

1. **Credit the determinant count where it is used.**
   - *What.* Apply the proposed text above to this file's Q3 sentence and to MODE_COUNT.md §2, and add the two
     references where the README and `paper/physics/main.tex` name the count.
   - *Why.* The Q3 sentence is contradicted (D6), and AGENTS.md asks for the original work where a method is named.
   - *Acceptance.* `awk '/^\*\*Q3/,/^\*\*Q4/' notes/LIT_REVIEW.md | grep -c 'has no counterpart'` prints 0 (1 today),
     and `grep -c Hirschmann notes/MODE_COUNT.md` and `grep -c Gundlach notes/MODE_COUNT.md` each print at least 1 (0
     today). `python3 check_numbers.py --selftest` passes on both of its lines.
   - *Depends on.* #619. Minutes.

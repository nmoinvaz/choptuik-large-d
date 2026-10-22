# Literature review of the novelty and priority claims

Review of 2026-10-21, following the `lit-review` protocol of the BootLoops skills repository, with the
`reading-contract` rules.

- **Scope.** The novelty and priority claims in [README.md](../README.md), [paper/main.tex](../paper/main.tex) and
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
| **C. Ecker, F. Ecker, D. Grumiller, T. Jechtl**, SciPost Phys. 20, 176 (2026), arXiv:2602.10185 ([PDF](https://arxiv.org/pdf/2602.10185), [v2 HTML](https://arxiv.org/html/2602.10185v2)) | P3 for C6 and C7 | Full text to about 88k characters; the reference list was not reachable | Values at Eqs. (69)–(71) and in the abstract (see REF_CHECK.md §4). Sec. 4.1: "The expansion order around the SSH has to be increased above D=4.8". Sec. 4.2: perturbations "cannot be computed any more with the current method" for $`4.81 \le D \le 5.38`$. Sec. 5.2: "compatible with the conjectures $`\Delta \to 0`$ both for $`D \to \infty`$ and for $`D \to 3^+`$". Sec. 5.3: "compatible with the conjecture" $`\lim\gamma = \tfrac12`$, "see for instance [12]". Ref. [12] could not be identified: the reference list did not come through. An aggregator ([pith.science](https://pith.science/paper/2602.10185)) lists Rozali–Way among the works it cites; unverified. |
| **S.-H. Lai, W.-Y. Wen**, *Polyakov backreaction in large D discretely self-similar collapse*, arXiv:2610.04960 (October 2026) ([HTML](https://arxiv.org/html/2610.04960)) | P3 for C1 and C8 | Full text (HTML) | Builds on EEG: "Following Ref. [12], we work in the past patch of the critical spacetime" (Sec. II.A). "the next-to-leading large-D consistency condition selects $`\Delta = 1`$" (Sec. II.A, unit normalisation). "Uniqueness of the echoing period is not a leading-order statement." (Sec. II.B). The period is "a modulus" fixed by subleading conditions (Sec. IV.C). It does not cite Rozali–Way or this project. **A 2026 paper relies on the NLO selection that C8 disputes.** |
| **C. R. Clark, G. L. Pimentel**, *Black hole critical collapse in infinite dimensions: continuous self-similar solutions*, JHEP 11 (2025) 017, arXiv:2504.10590 ([Springer](https://link.springer.com/article/10.1007/JHEP11(2025)017), [PDF](https://arxiv.org/pdf/2504.10590)) | P3 for C1, C5 and C6 | Full text (PDF extraction) | Large-$`D`$ matched asymptotics for the CSS critical solution, with $`\epsilon = 1/(D-2)`$ (Eq. (2.8)) and regions A/B/C in self-similar time. Footnote 1: "it has not yet been proven that there are no further restrictions from the 'matching' of these asymptotics". CSS-breaking mass exponent: "γ = 2(D−3)/(D−2) ≈ 2 − 2/D" (Eq. (1.5)), with $`M \sim (p-p^*)^\gamma`$ (Eq. (2.1)). The introduction cites [28] for the exponent possibly approaching 1/2. The DSS solution is "only known approximately and numerically" (footnote 5). There is no heat-equation form and no beyond-all-orders discussion. **Not cited in the paper or the README.** |
| **C.-H. Wu**, *Quantum critical collapse in the large D limit*, arXiv:2610.20705 (14 Oct 2026) ([abs](https://arxiv.org/abs/2610.20705)) | P2 | Abstract only | Quantum $`s`$-wave growth in CSS and DSS critical spacetimes at large $`D`$; no $`\Delta`$, $`\gamma`$ or Stokes content in the abstract. |
| **C. Gundlach, D. Hilditch, J. M. Martín-García**, *Critical Phenomena in Gravitational Collapse*, arXiv:2507.07636 ([HTML](https://arxiv.org/html/2507.07636v1)) | P2 | Full text (HTML) | No large-$`D`$ results and no limits of $`\Delta`$ or $`\gamma`$. It does not mention Rozali–Way, Clark–Pimentel or Ecker et al. Sec. IV.A.2: Kol "claims to obtain analytic estimates for γ and Δ in arbitrary dimension". Sec. III.C: the critical solution "was determined accurately in [43] by assuming analyticity at the centre of spherical symmetry and at the past light cone". No exponential asymptotics. |
| **B. Kol**, *Choptuik scaling and the merger transition*, hep-th/0502033, DOI 10.1088/1126-6708/2006/10/017 ([PDF v3](https://arxiv.org/pdf/hep-th/0502033v3)) | P2 | Full text (PDF extraction) | For the black-string merger system: Eq. (3.20) "$`\frac{2\pi}{\Delta} = \sqrt{\frac{10 - D}{D - 2}}`$"; Eq. (3.21) "$`\gamma = \frac{1}{4} \left( 1 + \sqrt{\frac{D - 10}{D - 2}} \right)`$" for $`D \ge 10`$. *Inferred (my arithmetic):* $`\gamma\to1/2`$ there too, as $`D\to\infty`$. There is no formula for Choptuik's $`\Delta(D)`$. |
| **R. Emparan, C. P. Herzog**, RMP 92, 045005 (2020) ([PDF](https://arxiv.org/pdf/2003.11394)) | P2 | Partial: the text extraction stops at Sec. V. **Sec. X.C "Critical collapse and Choptuik scaling" is FULL-TEXT-UNREAD**, so this grade is a ceiling. | Sec. III.F: "The breakdown is non-perturbative in 1/n"; "non-perturbative corrections = O(n^{-n})"; quasinormal modes as "non-perturbative, or trans-series, corrections". Sec. IV.E.1: "resemble (inhomogeneous) heat equations" (black-brane effective equations). The large-$`D`$ community already knows trans-series effects in $`1/D`$, and heat-like effective equations. |
| **R. Donninger**, J. Evol. Equ. 9, 511 (2009), arXiv:0809.5177 ([PDF](https://arxiv.org/pdf/0809.5177)) | P3 for C2 | Full text (PDF extraction) | "We consider the radial wave equation in similarity coordinates within the semigroup formalism"; "the long time behaviour of the solution is governed by the analytic eigenfunctions". This is 3+1 only, with explicit eigenfunctions $`(1-\rho)^{1-\lambda}-(1+\rho)^{1-\lambda}`$. It gives no connection coefficients and no general $`d`$. |
| Donninger, arXiv:0805.0520; Costin–Donninger–Glogić, arXiv:1604.00303; Donninger–Aichelburg, math-ph/0702025; Biernat–Donninger–Schörkhuber, arXiv:1707.09812 ([ar5iv](https://ar5iv.arxiv.org/html/0805.0520), [PDF](https://arxiv.org/pdf/1604.00303), [PDF](https://arxiv.org/pdf/math-ph/0702025v1), [ar5iv](https://ar5iv.arxiv.org/html/1707.09812)) | P1–P2 for C2 | Partial extractions | None gives the free-wave mode equation in $`d`$ dimensions as $`{}_2F_1`$ with a Gamma-function connection coefficient at the light cone. Costin–Donninger–Glogić reduce to Heun equations and state "the general connection problem for these equations is unresolved" (Sec. 4). |
| **R. H. Price, J. Pullin**, *Analytic approximations to the spacetime of a critical gravitational collapse*, gr-qc/9601009 ([ar5iv](https://ar5iv.labs.arxiv.org/html/gr-qc/9601009)) | P2 for C2 and C4 | Partial (truncated before the matching section) | Abstract: "the critical solution can be well approximated by a flat spacetime scalar field solution"; the matching "yields the right order of magnitude for the oscillations". This is an early flat-space (free-wave) reading of the echo. Not cited. |
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
| Price & Pullin 1996 | flat-space (free-wave) analytic approximation of the critical solution, a precursor of the leading-order picture | gr-qc/9601009 ([ar5iv](https://ar5iv.labs.arxiv.org/html/gr-qc/9601009)). The journal appears in an APS search listing as PRD 54, 3792; that page was not fetched. |
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
- **Kol**, hep-th/0502033 (full text, already recorded above): 2π/Δ = √((10 − D)/(D − 2)) and a formula for γ, but for
  the black-string merger, not for scalar-field collapse.

Verdict: no earlier formula for Δ(D) was found, and the only earlier fit for γ(D) (Bland et al.'s exponential) has a
different form and limit. The rational formula is an empirical fit, a ratio of polynomials with 11 fitted coefficients for Δ and 9 for γ.
Rational (Padé-type) fitting is a standard method, so what is new is the fitted function and its validation (precision
across D ≥ 4, held-out agreement with the double-double points to D = 61, and the correct limits), not a law derived from
the equations.

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

Verdict: every technique in PARALLEL.md has a named precedent. Not found in this search, and not a technique in any case:
the CRC-32 framing of multiple shooting and of the series (an exposition), the instances specific to this code (the
summation-factor closed form with T_j = (α)_j(β)_j/j!, the 193-unknown condensed system, the 1.9% nonzero LU work after
pivoting), the measured speedups, and the late-order fit-basis finding of §4 (every neighbour family |ν′| ≤ k is needed).
That last one is a numerics result about this computation; the general principle, that late orders carry every
singularity's contribution, is standard in resurgence (Dingle; Berry–Howls).

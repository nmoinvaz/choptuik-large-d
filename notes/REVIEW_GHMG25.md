# REVIEW_GHMG25: our results against the 2025 critical-phenomena review (#380, part 1)

The review is C. Gundlach, D. Hilditch and J. M. Martín-García, *Critical Phenomena in Gravitational Collapse*,
arXiv:2507.07636v1 (10 July 2025; 51 pages; gr-qc, math.AP), the update of the 2007 Living Review (lrr-2007-5). This
note sets every statement of it that overlaps our work against our value or result. Part 2 of #380 (its PDE-blowup and
naked-singularity sections as a map for other fields) is not covered here.

- **Version.** v1 is the only arXiv version (abstract page fetched 2026-11-12). Its arXiv comment says that feedback
  "received by October 2025 will be considered in the version to be published". INSPIRE lists no journal publication
  on 2026-11-12, so the published text may differ from the one read here.
- **How it was read.** The arXiv source (`critphen_lrr_maintext.tex`, 5122 lines, and its `.bbl`), every line, on
  2026-11-12. Quotes below are verbatim from that source, with LaTeX math rewritten in Unicode. Section numbers follow
  RevTeX's style (IV A 2, which the arXiv HTML renders as IV.1.2). Equation numbers are those of the arXiv HTML
  rendering, matched to their tags.
- **Text addressed to an AI agent.** None was found in the source.
- **Primary sources.** A statement credited to a cited paper was read only as the review states it, unless the row says
  otherwise. The abstracts of eight cited papers were read, to verify the entries of §5.

## Verdict

- **35 statements overlap.** 15 agree, 13 are consistent (compatible, but not the same quantity, precision or reading),
  and 7 differ (§3). None of the 7 bears on a headline number of the README.
- **The review is silent on large D.** It has no Δ for any D ≠ 4, nothing on D → ∞ or D → 3⁺, and no mechanism that
  selects Δ. So Δ(D) above D = 4, Δ∞, γ∞ = 1/2, the formula, Propositions 1–5, the selection rule (Conjecture 1) and
  Theorems A and B have nothing in it to agree or disagree with.
- **The closest agreements.** Δ(4), with Reiterer and Trubowitz's digits. The surface gravity of the similarity
  horizon, ν_SH ≃ 0.6138, which at D = 4 is exactly our residual gauge rate λ_g. The single unstable mode at D = 4. The
  κ = 0 complex scalar, whose critical solution is the real DSS while the phase-rotating CSS has three unstable modes.
- **Seven citations to add** (§5). One corrects our own notes: notes/LIT_REVIEW.md says the argument-principle mode
  count has no counterpart in the sources it read. The review names Gundlach 1998, which has no such count (#619), but
  Hirschmann–Eardley 1995 and Gundlach 1997 do (D6).

## 1. What the review contains on our topics, and what it does not

Contains:

- the real scalar in D = 4, with Δ(4), the echo count and the mode picture (II A, II C, III B, III C);
- the scalar in other dimensions (IV A 1, 2+1; IV A 2, γ in D = 4, 5, 6 and work in non-integer D up to 14 and in 8+1);
- other matter (IV A 3–5, V A 2 and Table I): σ-models, the complex scalar, the ℓ-multiplet, Yang–Mills, Bianchi IX;
- charge, quantum effects and ghosts (III E, IV D) and toy models (IV E);
- non-spherical perturbations and axisymmetric evolutions (V A 1);
- a sentence crediting Gundlach 1998 with a mode count by contour integration (VI E), which that paper does not
  contain (D6);
- Reiterer and Trubowitz's existence proof, the regularity of CSS against DSS solutions, and the surface gravity ν of
  the self-similarity and Cauchy horizons (VII C, VII D 3, VII D 7, VII G, VII H).

Does not contain:

- large D in any form. Rozali–Way, Emparan–Herzog and Clark–Pimentel are not cited, and Ecker et al.'s papers appeared
  after it;
- Δ for the real scalar at any D ≠ 4 (only γ values, at D = 4, 5 and 6);
- the limit D → 3⁺ of the asymptotically flat family;
- any formula or mechanism for Δ, Stokes phenomena or exponential asymptotics;
- mode counts at D ≠ 4;
- ghost or wormhole thresholds. "Ghost" appears once (IV D), "wormhole" once, for a membrane toy model (IV E), and
  "negative mass" for the DSS continuations beyond the Cauchy horizon (III C).

## 2. The comparison table

Verdicts: **agree** (within the stated uncertainty, or to every digit quoted), **consistent** (no conflict, but not the
same quantity, precision or reading), **differs** (a discrepancy, §3). Our status words are AGENTS.md's.

| # | The review says | Where | Its source | Ours | Where here | Verdict |
|---|---|---|---|---|---|---|
| 1 | "Δ = 3.445452402(3) from a numerical construction of the critical solution based on exact self-similarity and analyticity" | III B | Martín-García & Gundlach 2003 | 3.4454524022278213499715 (M = 192), with the change from M = 160, 1.1×10⁻²², as its error (an estimator) | README, Relation to published work | agree |
| 2 | That value "agrees with the leading digits of the same quantity as computed in the existence proof of [Reiterer–Trubowitz]" | III B | Reiterer & Trubowitz 2019 | our M = 192 Δ is 3.1×10⁻²⁶ from their 2K, whose radius is 2×10⁻⁸⁰ (a rigorous enclosure); μ is 7.7×10⁻²⁸ from theirs | notes/RT_FORMULATION.md §2, `solver/rt/mu_check.py`, #372 | agree |
| 3 | ν_SH,Choptuik ≃ 0.6138..., from Reiterer–Trubowitz's μ and Δ with ν = 4πμ/Δ, Eq. (125); in spherical symmetry ν − 2 = −⟨a₀²⟩ on the horizon, Eq. (119), read as the horizon's surface gravity | VII G | Reiterer & Trubowitz 2019; the review's derivation | our residual gauge rate is λ_g = 1 − (n − 1)⟨a_h² − 1⟩, at n = 2 the form 2 − ⟨a_h²⟩ of the review's Eq. (119). Our map gives λ_g = 4πμ/Δ, and our μ is 7.7×10⁻²⁸ from theirs (row 2), so at D = 4 λ_g is the review's ν_SH to that precision. From the constants in `mu_check.py`, 4πμ/(2K) = 0.6138552748 (*inferred*, arithmetic); an f64 state gives λ_g = 0.6138550 | notes/MODE_COUNT.md §1 and §5; notes/RT_FORMULATION.md §2 | agree |
| 4 | "Any self-similar solution admits an unstable perturbation with λ = 1 that corresponds to an infinitesimal time shift of the background solution, and so is unphysical." | II C | — | λ = 1 is not a zero of our det M: the light cone is pinned to x = 1, and the residual gauge zero sits at λ_g instead. λ = 1 is a zero of the CSS counts and an exact mode of the inner problem | notes/MODE_COUNT.md §1; matter/COMPLEX_CRITICAL.md §2; PROOF.md §2 | consistent |
| 5 | A critical solution has "exactly one λᵢ with positive real part (in fact it is purely real)"; a mode ansatz finds "a single unstable mode"; all non-spherical modes decay | II A, II C, V A 1 | Martín-García & Gundlach 1999 (non-spherical) | one physical unstable spherical mode, real, at D = 4 (complex λ, M = 48), 8.25, 16.76, 30.07, 45.05, 61.03 (numerical evidence; M = 48 extras at D ≥ 54 open). Non-spherical perturbations: not computed, listed as open | notes/MODE_COUNT.md; README, Limits and open problems | agree at D = 4 (spherical); citation to add for the non-spherical case |
| 6 | "a similar count was carried out for the spectrum of (spherical and non-spherical) linear perturbations of the fluid collapse critical solutions, using numerical contour integration in the complex plane of potential mode frequencies" | VI E | Gundlach 1998 | notes/LIT_REVIEW.md (addendum #200, Q3) says the argument-principle count of det M(λ) "has no counterpart in the sources read". Gundlach 1998, read by #619, evolves generic data in τ and reads off the dominant mode, with no contour integral. The determinant counts are Hirschmann–Eardley 1995 and Gundlach 1997, for scalar fields, and they contradict our sentence | notes/LIT_REVIEW.md, addenda #200 and #619 | differs (D6: the review against its source; our sentence against two earlier papers) |
| 7 | "γ = 0.4131 ± 0.0001" at D = 5 | IV A 2 | Bland & Kunstatter 2007 | γ(5) = 0.413126936 (double-double, M = 48) | README, Values | agree |
| 8 | the holographic prediction "γ = 0.409552" at D = 5, which motivated #7 | IV A 2 | Álvarez-Gaumé et al. 2007 | 0.413126936, 3.6×10⁻³ above the prediction, which #7 already excludes | README, Values | agree (with #7) |
| 9 | D = 6: "mass scaling with γ ≃ 0.424" | IV A 2 | Garfinkle, Cutler & Duncan 1999 | γ(6) = 0.430079481, 0.0061 above; no uncertainty is quoted | README, Values | differs (D1) |
| 10 | Birukou et al. "confirm known results in 3+1 (γ ≃ 0.36) and 5+1 (γ ≃ 0.44)" and find in 4+1 "γ ≃ 0.41 for one family" | IV A 2 | Birukou et al. 2002; Husain et al. 2003 | γ = 0.37396072, 0.430079481, 0.413126936: differences −0.014, +0.010, −0.003, at the 1–4% level of those evolutions | README, Values | consistent |
| 11 | in 4+1, "γ ≃ 0.52 for another" family | IV A 2 | Birukou et al. 2002 | 0.413126936 for the universal exponent; 0.52 is 0.107 above | README, Values | differs (D2) |
| 12 | "mass scaling with γ" in 4+1 and 5+1 | IV A 2 | as #9–11 | the mass scales with (D − 3)γ; the quoted numbers match our γ, the exponent of the length scale, so they are read as the scaling of $`M^{1/(D-3)}`$ (*inferred*) | README, The problem | consistent (a wording point) |
| 13 | "a numerical determination of γ and Δ for the spherical massless scalar field in noninteger dimension up to 14", and "[Porto Veronese–Gundlach] for results in 8+1" | IV A 2 | Sorkin–Oren 2005; Bland et al. 2005; Taves–Kunstatter 2011; Porto Veronese & Gundlach 2022 | README [^4] cites the first three for 3.5 ≤ D ≤ 14, not Porto Veronese–Gundlach. Their 8+1 results are figures only (read in their arXiv source, the scalar-field appendix); their D = 5 values, γ_Ψ ≃ 0.413–0.415 and Δ_Ψ ≃ 1.6, are our γ(5) and, it seems, our Δ(5)/2 = 1.611 (*inferred*: a half-period reading, not checked) | README [^4] | agree; citation to add |
| 14 | Kol "claims to obtain analytic estimates for γ and Δ in arbitrary dimension", which "motivated" #13 | IV A 2 | Kol 2006 | README: Kol's formula "is for the black-string merger, not for scalar collapse" (Kol's text read, notes/LIT_REVIEW.md §1) | README, Relation to published work | consistent (the README is sharper than the review) |
| 15 | the echo count "N ≃ Δ⁻¹ γ ln\|p − p∗\| + constant", Eq. (19); γ = 1/λ₀, Eq. (15) | II C | — | "about γ ln(1/\|p − p*\|)/Δ echoes"; γ = 1/λ | README, What the numbers mean physically | agree |
| 16 | 2+1 with Λ < 0: "a universal CSS critical solution, and γ = 1.20 ± 0.05"; a curvature exponent 7/8 and mass exponent 8/23 from the glued Garfinkle solution | IV A 1 | Pretorius & Choptuik 2000; Jałmużna, Gundlach & Chmaj 2015 | the D → 3⁺ limit of the asymptotically flat DSS family is open (finite Δ(3) or Δ → 0); notes/LOW_D.md §6 records that nothing ties the 2+1 CSS solution to that limit | README, Below D = 4; notes/LOW_D.md §6 | consistent (no common claim) |
| 17 | Reiterer–Trubowitz prove "that the Choptuik solution exists as an analytic solution from Γ through to somewhat to the future of C₀⁻", from a Fourier (τ) × Chebyshev (x) approximation in rational arithmetic, a quadratic formulation and l¹ norms; "the only DSS one so far" | VII C, VII D 7 | Reiterer & Trubowitz 2019 | README [^10]: "A computer-assisted proof shows that the D = 4 solution exists". PROOF.md's table of precedents: existence "near the past light cone", a narrower phrase than the review's. Our certificates are of a D = ∞ inner model, not existence at a finite D | README; PROOF.md, precedents; proof/B_chain.md §12.1 | agree (PROOF.md's wording could be sharpened, §7) |
| 18 | Δ is selected numerically by analyticity at the centre and at the past light cone; "A priori, there could also be more than one type II critical solution for m = 0, although this has not been observed." | III B, VII D 7, II F | Gundlach 1997; Martín-García & Gundlach 2003 | the same boundary-value problem (PROOF.md: "the eigenvalue is selected by analyticity at the light cone"). notes/HARMONY.md has a second zero of the D = ∞ selection function at ω/ω∞ ≈ 1.0989 (Δ ≈ 1.496), conditional on Conjecture 1, its stability unknown | PROOF.md; notes/HARMONY.md | consistent |
| 19 | "a CSS ansatz for the spherical Einstein scalar system only gives solution that are not smooth across the similarity horizon, whereas a DSS ansatz gives rise to a solution that is analytic"; Christodoulou's CSS solutions are $`C^{1,\epsilon}`$ at the past light cone, and perturbations of low Hölder regularity "would not form from smooth initial data" | VII D 3, VII D 7, VII H | Christodoulou 1994; Singh 2024 | README lists a CSS solution (Clark–Pimentel) taking over the threshold as not excluded; matter/CSS.md §9 finds that solution has no regular centre and no analytic light cone | README, Limits and open problems; matter/CSS.md §9 | consistent; citation to add (D = 4 statement only) |
| 20 | "the single unstable perturbation of the Choptuik solution is analytic, like the solution itself" | VII H | — | our modes are computed in the class regular at the centre and at x = 1 | notes/MODE_COUNT.md §1 | consistent |
| 21 | κ = 0 (free complex scalar): critical solution DSS (Table I); the Hirschmann–Eardley CSS "is critical for κ > 0.0754, but has three unstable modes for κ < 0.0754" | IV A 3, Table I | Hirschmann & Eardley 1997; Gundlach 1997 | the real DSS is critical at D = 4–67 (one unstable mode; none from the imaginary sector); the κ = 0 CSS has three unstable directions at D = 4–80 (λ_phys = 2.5832751746 and 0.25866983 ± 3.50224880i at D = 4); numerical evidence | matter/COMPLEX_CRITICAL.md §4 (#95) | agree |
| 22 | the CSS–DSS switch at κ = 0.0754 | IV A 3 | Hirschmann & Eardley 1997 | the Hopf point at κ_c = 0.075733 (D = 4) | notes/HARMONY.md, "In the literature, reported, not resolved"; #377 | differs (D3) |
| 23 | Hirschmann and Eardley "conjectured that the transition is a Hopf bifurcation, such that the DSS cycle smoothly shrinks with growing κ, collapsing onto the CSS solution at the transition and then disappearing while the echoing period Δ remains finite" (unchanged from 2007) | IV A 3 | Hirschmann & Eardley 1997 | the cycle born at the Hopf point has a period 0.7–1.1% from Choptuik's Δ/2 at D = 4–6, but "It is not Choptuik's cycle, which exists at every κ" | notes/HARMONY.md item 6; notes/ECHO_ANALOGUES.md §B | differs (D4) |
| 24 | complex scalar in axisymmetry: "the critical solution is again the real Choptuik solution (up to a constant complex phase)" | V A 1 | Marouda et al. 2024; Cors et al. 2023 | the real DSS is the critical solution of the free complex scalar (spherical counts) | matter/COMPLEX_CRITICAL.md §4 | agree; citation to add |
| 25 | in scalar electrodynamics the real critical solution keeps one growing mode; charge scales with δ ≃ 0.88, Eq. (39); δ ≥ 2γ, Eq. (40) | III E | Gundlach & Martín-García 1996; Hod & Piran 1997 | with δ = (D − 2 − μ)γ (Mittal, Gundlach & Martel 2026, as quoted in matter/LITERATURE.md §8), μ(4) = −0.3651202372 (the slowest decaying real test-field rate found; Gundlach and Martín-García's −0.362 ± 0.012) and γ(4) = 0.37396072, δ(4) = 0.8845 > 2γ = 0.7479 (*inferred*, arithmetic). The same formula gives 1.4249 at D = 5 from their own μ = −0.448 and γ = 0.41325, against their 1.425 | notes/GHOST_ADMIXTURE.md §4; matter/LITERATURE.md §8 | agree |
| 26 | a semiclassical model with massive ghost scalars shows type II scaling with γ ≃ 0.38, "but it is not clear if and why the ghost fields remain stable" | IV D | Berczi, Saffin & Zhou 2022 | a massless second scalar, ghost or ordinary, adds no growing mode at linear order: D = 4 near the imaginary axis (two resolutions, one layout), D = 8.25–45 in boxes (numerical evidence). By II F the mass is irrelevant on small scales | notes/GHOST_ADMIXTURE.md; README, What the numbers mean physically | consistent (bears on the question at linear order); citation to add |
| 27 | quantum effects add an unstable mode with λ = 2 (Brady–Ottewill); in finite-dimensional toy models "the critical exponent γ, suitably defined, is 1/2", and in some 1+1 models "for kinematical reasons" | IV D, IV E | Brady & Ottewill 1998; others | λ∞ = 2 (γ∞ = 1/2) from the r²-shift symmetry of the D = ∞ inner equations, with λ_phys > 2 at every computed D | README, Proposition 5; GAMMA.md | consistent (same values, different mechanisms) |
| 28 | S³ σ-model: the n-th CSS solution is regular to the future light cone only for η < η_n, "η₁ ≃ 0.1518" | IV A 3 | Bizoń & Wasserman 2000 | α₁ = 0.1518–0.1519 (a range over fits of the horizon's divergence) | matter/SIGMA.md §4 | agree |
| 29 | the same, "η₀ ≃ 0.0688" | IV A 3 | Bizoń & Wasserman 2000 | α₀ = 0.0686–0.0687 (the same kind of range) | matter/SIGMA.md §4 | differs (D5, minor) |
| 30 | the CSS–DSS transition at "η_c ≃ 0.170", with "Δ ≃ −(2/λ) log(η − η_c) + b", λ the CSS Lyapunov exponent | IV A 3 | Aichelburg, Bizoń & Tabor 2006; Lechner et al. 2002 | η_c = 0.1702; the fitted slope 0.359 rises toward 2/λ_CSS = 0.389 as the window narrows | matter/SIGMA_DSS.md §4 | agree |
| 31 | for η ≳ 0.2, "The period Δ depends on η, monotonically decreasing towards an asymptotic value for η → ∞" | IV A 3 | Husa et al. 2000; Lechner 2001 | Δ falls from 1.276 (η = 0.18) to 0.459956 (η = 5), then 0.459963 (10) and 0.460021 (∞): non-monotonic by 6.5×10⁻⁵, below the k_c = 15 truncation (1.6×10⁻⁴ at η = ∞). The η → ∞ limit is the ℓ = 1 multiplet (our identification; the review does not name it) | matter/SIGMA_DSS.md §1, §4 | consistent (WATCH; a recommended task) |
| 32 | the ℓ-multiplet: DSS, "Δ and mass scaling exponent γ both decrease approximately exponentially with l" | V A 2 | Olabarrieta et al. 2007 | at D = 4, ℓ = 0, 1, 2: Δ = 3.4455, 0.4598611, 0.1193 (the last from an evolution); γ = 0.374, 0.118900 | matter/ELL.md; README, Other matter | consistent |
| 33 | EYM (magnetic): the DSS solution of the ansatz w = 1 − rS "has no non-gravitating counterpart"; in the general SU(2) system, "evidence against the existence of a universal critical solution" | IV A 4 | Gundlach 1997; Maliborski & Rinne 2018 | the same reading of the magnetic case (matter/LITERATURE.md §4.1 cites the review); the README's Yang–Mills row is for the magnetic ansatz | matter/LITERATURE.md §4.1; README, Other matter | agree (with a caveat for the general case) |
| 34 | Bianchi IX in 4+1: "Δ ≃ 0.49 and γ ≃ 0.3289" | IV A 5 | Bizoń, Chmaj & Schmidt 2005 | matter/LITERATURE.md §6.1 reads Δ ≈ 0.47 in that paper and in Porto Veronese–Gundlach ("the 2025 review quotes 0.49"); not computed here | matter/LITERATURE.md §6.1 | differs (D7, the review against its source) |
| 35 | dimensionless parameters that change the universality class include "k in the perfect fluid equation", "the κ of the 2-dimensional sigma model" | II F | — | matter/STRUCTURE.md predicts γ∞ = 1/2 for spherically symmetric scalars with any target geometry; the review's statement is at D = 4, where γ does depend on κ | matter/STRUCTURE.md; matter/LITERATURE.md §10 | consistent |

## 3. Discrepancies (reported, not resolved)

- **D1. γ(6).** The review quotes γ ≃ 0.424 (Garfinkle, Cutler & Duncan 1999), with no uncertainty. Ours is
  0.430079481, quoted to 10⁻⁹ (double-double, M = 48), 0.0061 (1.4%) higher. The 1999 value is an evolution fit, and
  the paper itself was not read here.
- **D2. A second exponent in 4+1.** The review quotes Birukou et al.'s γ ≃ 0.52 for a second family at D = 5, without
  comment. Ours, and Bland and Kunstatter's 0.4131 ± 0.0001, are 0.4131; a family-dependent exponent conflicts with
  universality. The 2002 paper was not read.
- **D3. κ_c.** The review states the CSS is critical for κ > 0.0754 (Hirschmann & Eardley 1997, no uncertainty).
  notes/HARMONY.md finds the Hopf point at κ_c = 0.075733 at D = 4 (numerical evidence: two settings agree, the
  κ = 0 mode reproduced to 10⁻¹¹), 3.3×10⁻⁴ higher. HARMONY.md already lists it against Hirschmann and Eardley; the
  review repeats their value. #377 (κ_c(D)) is the task that would settle it.
- **D4. Hirschmann and Eardley's Hopf picture.** The review keeps the 2007 sentence that the DSS cycle "smoothly
  shrinks with growing κ, collapsing onto the CSS solution at the transition". notes/HARMONY.md finds that the cycle
  born at the Hopf point is not Choptuik's, which exists at every κ. So the Choptuik cycle, critical at κ = 0, does not
  shrink onto the CSS; which DSS is critical just below κ_c is not settled by HARMONY.md's data (*inferred*). HARMONY.md
  records the conflict against the 2007 review; it stands against the 2025 one.
- **D5. η₀.** The review quotes η₀ ≃ 0.0688 (Bizoń & Wasserman 2000). matter/SIGMA.md gives 0.0686–0.0687, a range
  over fits of ln x_B = K/(η − α) + c, so 1–2 units of the last digit lower. The coupling where x_B first stays below
  10⁸, 0.06923, lies above both, so the difference may be the extrapolation's.
- **D6. The review against its source, and our notes against two earlier papers.** The review (VI E) says Gundlach
  1998 counted the fluid critical solution's modes, spherical and non-spherical, "using numerical contour integration
  in the complex plane". #619 read that paper's source, both versions. It evolves generic perturbation data, for
  p = ρ/3 only, and reads off the dominant mode: "To answer this yes/no question, we write the perturbation equations as
  evolution equations in the time variable τ" (paragraph "Perturbation method"). It has no contour integral and counts
  nothing. Its successor, Gundlach 2002 (Sec. IV B), names contour integrals of det A(λ) as one method, citing Gundlach
  1997, and uses the Lyapunov method and the eigenvalues of the discretized operator. The review's V B also credits
  Gundlach 1998 with 2002's ranges of k. Separately, notes/LIT_REVIEW.md (addendum #200, Q3) says that
  "MODE_COUNT.md's argument-principle count of det M(λ) has no counterpart in the sources read", and two papers
  contradict it. Hirschmann–Eardley 1995 (Secs. VI–VII) and Gundlach 1997 (App. E) count the zeros of a shooting
  determinant by the argument principle, for scalar-field critical solutions. The sentence is noted, not rewritten;
  the reading and the proposed text are in notes/LIT_REVIEW.md, addendum #619.
- **D7. The review against its source.** For Bianchi IX in 4+1 the review quotes Δ ≃ 0.49, where matter/LITERATURE.md
  §6.1 reads 0.47 in Bizoń, Chmaj & Schmidt 2005 and in Porto Veronese–Gundlach 2022 (whose 4+1 value Δ_B ≃ 0.47 was
  read here in their source). This is not our computation.

## 4. Existence proofs, and what a selection rule would add

- **What the review records.** Two critical solutions have rigorous existence proofs: Bizoń and Wasserman's CSS
  σ-model solution ("the first critical solution whose existence has been proved rigorously, and the only CSS one so
  far") and the Choptuik solution by Reiterer and Trubowitz ("the only DSS one so far"), VII C. The latter encloses Δ(4)
  and proves analyticity from the centre to slightly beyond the past light cone (VII D 7). Neither proof characterizes
  Δ beyond enclosing it, and the review states no uniqueness result for the critical solution (II F, row 18).
- **Its open problems that touch ours.**
  - A proof that the Choptuik solution is a naked singularity, analytic up to but not across its Cauchy horizon (VII
    D 7): "It might be possible to give a brute force proof using the method of [Reiterer–Trubowitz]". Our solvers stop
    at x = 1, so nothing here bears on it.
  - Mode stability, the "hard part" of a stability proof, which Donninger calls "probably hard to implement if the
    solution is not known in closed form" (VI A, quoting arXiv:2310.12016). Our winding counts (row 5) are numerical
    evidence for the mode count, not a proof.
  - Uniqueness of the type II critical solution (II F, row 18).
  - The completeness of the modes and the stability of the Cauchy horizon (III D), and a norm on the phase space (II E).
    Nothing here addresses them.
- **What Conjecture 1 would add.** An equation for Δ∞ (and so for the limit of Δ(D)) from the D = ∞ inner problem
  alone, which the review does not have in any form. It would not give existence at a finite D, which Reiterer and
  Trubowitz's method does (PROOF.md, layer C, proposes it at one large D). Theorem B(1) adds uniqueness of the selected
  parameter within a neighbourhood of the formal series, for a one-harmonic model at n ≥ e^{89.75} only.

## 5. Citations to add

Each entry was checked on INSPIRE or arXiv on 2026-11-12. Where only the review's statement was read, the row says so.

1. **MODE_COUNT.md §1, at λ_g.** The review, Sec. VII G, Eqs. (119) and (125): at D = 4 our λ_g is the review's ν_SH,
   the surface gravity of the self-similarity horizon, and its Sec. II C for the λ = 1 time-shift mode.
2. **MODE_COUNT.md §2 and LIT_REVIEW.md (Q3).** E. W. Hirschmann and D. M. Eardley, *Critical exponents and stability
   at the black hole threshold for a complex scalar field*, Phys. Rev. D 52, 5850 (1995), arXiv:gr-qc/9506078, Secs.
   VI–VII; and C. Gundlach, *Understanding critical collapse of a scalar field*, Phys. Rev. D 55, 695 (1997),
   arXiv:gr-qc/9604019, App. E. Both count the zeros of a shooting determinant by the argument principle (read by #619;
   checked on INSPIRE, arXiv and doi.org on 2026-11-13). Not Gundlach 1998, PRD 57, R7075, which the review names: it
   counts by time evolution (D6).
3. **README, the open item "Non-spherical perturbations".** J. M. Martín-García and C. Gundlach, *All nonspherical
   perturbations of the Choptuik spacetime decay*, Phys. Rev. D 59, 064031 (1999), arXiv:gr-qc/9809059 (abstract read):
   at D = 4 the question is answered; it stays open at D ≠ 4.
4. **README [^4].** M. Birukou, V. Husain, G. Kunstatter, E. Vaz and M. Olivier, *Scalar field collapse in any
   dimension*, Phys. Rev. D 65, 104036 (2002), arXiv:gr-qc/0201026 (the first D = 5 evolutions); and B. Porto Veronese
   and C. Gundlach, Phys. Rev. D 106, 104044 (2022), arXiv:2209.08404 (the scalar in 8+1, and a critique of Bland et
   al.'s formulation).
5. **matter/COMPLEX_CRITICAL.md §4.** K. Marouda, D. Cors, H. R. Rüter, F. Atteneder and D. Hilditch, *Twist-free
   axisymmetric critical collapse of a complex scalar field*, Phys. Rev. D 109, 124042 (2024), arXiv:2402.06724; and
   D. Cors, S. Renkhoff, H. R. Rüter, D. Hilditch and B. Brügmann, *Formulation improvements for critical collapse
   simulations*, Phys. Rev. D 108, 124021 (2023), arXiv:2308.01812 (the APS DOI on INSPIRE; its free-text field says
   124035). Their axisymmetric complex-scalar evolutions approach the real Choptuik solution (the review's statement;
   neither paper read here).
6. **README, the open item on a CSS solution taking over.** The review, Secs. VII D 3 and VII H: in D = 4 the
   continuously self-similar Einstein–scalar solutions are only $`C^{1,\epsilon}`$ at the past light cone.
7. **notes/GHOST_ADMIXTURE.md.** B. Berczi, P. M. Saffin and S.-Y. Zhou, *Gravitational collapse of quantum fields and
   Choptuik scaling*, JHEP 02 (2022) 183, arXiv:2111.11400 (abstract read): their model has massive ghost fields, whose
   stability the review calls unclear (Sec. IV D).

Also, not a missing citation: PROOF.md's novelty sentence C9 ("the unknown is a function of two variables (x, θ),
periodic in θ [...] Neither template has that") is scoped to the two Stokes-constant templates and survives as worded.
Reiterer and Trubowitz's unknown is also a function of (τ, x), periodic in τ (VII D 7), with no Stokes multipliers, so
naming them there would make the sentence's scope plain (notes/LIT_REVIEW.md, addendum #380).

## 6. Recommended tasks (for the coordinator to file)

- **R1. Read Gundlach 1998's mode count.**
  - *What.* Read Phys. Rev. D 57, R7075 (1998) and record how it counted the fluid solution's modes. Correct
    notes/LIT_REVIEW.md's "no counterpart" sentence if it is a contour-integration count.
  - *Why.* D6: the review names it as a precedent for MODE_COUNT.md's method.
  - *Acceptance.* The method passage is quoted with its locator. The task fails, and says so, if the paper counts modes
    some other way, which would make the review's statement a discrepancy to report.
  - *Depends on.* #380. About 1 hour.
  - *Done* by #619. The paper counts modes by time evolution, so the review's statement is the discrepancy (D6).
- **R2. A table of the published γ(D) and Δ(D) below D = 14.**
  - *What.* Add the values the review cites (Garfinkle–Cutler–Duncan, Birukou et al., Bland–Kunstatter, Sorkin–Oren,
    Bland et al.'s Table I, Taves–Kunstatter, Porto Veronese–Gundlach), each with its convention (mass or length
    exponent, full or half period), to a `gamma_published_compare.csv` beside `hp/data/delta_published_compare.csv`,
    which now holds only EEG, EEGJ and MGG2003, with a `check_numbers.py` check.
  - *Why.* D1, D2 and row 13 rest on values read through the review; the half-period reading of row 13 is unchecked.
  - *Acceptance.* Every value is copied from its paper with a locator. The check fails when a value is moved by one unit
    of its last digit.
  - *Depends on.* #380. 2–3 hours.
- **R3. The charge exponent δ(D) from the test-field rate.**
  - *What.* Compute μ(D), the slowest decaying test-field rate on the DSS (the GHOST_ADMIXTURE.md machinery), at D = 4,
    5, 6 and 8, and δ(D) = (D − 2 − μ)γ.
  - *Why.* Row 25 rests on one rate and on arithmetic; Mittal, Gundlach and Martel's D = 5 values (μ = −0.448,
    δ_Q = 1.425, arXiv:2609.17207) are a direct check.
  - *Acceptance.* μ(4) reproduces −0.36512 in two layouts. δ(5) matches 1.425 to their digits or is reported as a
    discrepancy. A mutation of the formula (D − 2 → D − 3) fails the D = 5 check.
  - *Depends on.* #380. Half a day to a day.
- **R4. ν_SH(D), the surface gravity of the self-similarity horizon.**
  - *What.* Tabulate λ_g(D) = 4πμ/Δ = 1 − (n − 1)⟨a_h² − 1⟩ on the stored states from D = 4 to the chain's frontier,
    identify it with the review's ν_SH (VII G) and fit its D → ∞ behaviour.
  - *Why.* Row 3. The review ties ν to the blue-shift instability and to the regularity of Christodoulou-type
    perturbations (VII D 4, "if one replaces k² by 1 − ν").
  - *Acceptance.* At D = 4 it reproduces 4πμ/(2K) from Reiterer and Trubowitz's constants to 10⁻²⁰ (`mu_check.py`'s
    state). The identity λ_g = 4πμ/Δ holds to the states' precision at three D. A mutation (n in place of n − 1) fails.
  - *Depends on.* #380. About 2 hours.
- **R5. Is Δ(η) of the σ-model monotonic at large η?**
  - *What.* Recompute Δ at D = 4 for η = 2, 5, 10, 20 and ∞ at k_c = 23 and 39.
  - *Why.* Row 31: the k_c = 15 table is non-monotonic by 6.5×10⁻⁵, below its own truncation error.
  - *Acceptance.* The sign of Δ(5) − Δ(∞) is the same at k_c = 23 and 39, and both differences exceed their change
    between the two resolutions; otherwise the question stays open and the task says so.
  - *Depends on.* #380. Half a day.

## 7. Proposed text for the README

For the coordinator, who edits the README. Each footnote entry is verified in §5.

- **Prior work, the Reiterer–Trubowitz bullet** (with #372's check):
  > A computer-assisted proof shows that the D = 4 solution exists and encloses its Δ to 80 digits[^10].
- **[^4]**, appended:
  > M. Birukou, V. Husain, G. Kunstatter, E. Vaz and M. Olivier, PRD 65, 104036 (2002); B. Porto Veronese and
  > C. Gundlach, PRD 106, 104044 (2022), [arXiv:2209.08404](https://arxiv.org/abs/2209.08404).
- **Limits and open problems, the open list:**
  > - Non-spherical perturbations at D ≠ 4. At D = 4 all of them decay (Martín-García and Gundlach[^n]).
  >
  > [^n]: J. M. Martín-García and C. Gundlach, PRD 59, 064031 (1999), [gr-qc/9809059](https://arxiv.org/abs/gr-qc/9809059).
- **The same list, the item on a CSS solution:**
  > No collapse has been evolved above D = 14, so a continuously self-similar solution (Clark and Pimentel, ...) taking
  > over the threshold is not excluded. In D = 4 the continuously self-similar Einstein–scalar solutions are not smooth
  > at the past light cone (the review[^9], Secs. VII D 3 and VII H).
- **Relation to published work, the Kol bullet** (row 14), optionally:
  > Kol's 2π/Δ = √((10 − D)/(D − 2)) (2005) is for the black-string merger, not for scalar collapse, though it motivated
  > the computations in non-integer D[^4].
- **PROOF.md's table of precedents** (not the README): "near the past light cone" → "from the centre to slightly
  beyond the past light cone".

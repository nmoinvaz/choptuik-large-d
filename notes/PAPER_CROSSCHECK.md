# Cross-check of the physics and mathematics papers

Review of 2026-10-21 of the two papers split from the combined draft:

- **physics**: [paper/physics/main.tex](../paper/physics/main.tex) (dss_physics.pdf, 14 pages);
- **math**: [paper/math/main.tex](../paper/math/main.tex) (dss_math.pdf, 30 pages);
- **old**: the combined draft [paper/main.tex](../paper/main.tex).

References below are `file:line`, with `phys:` for paper/physics/main.tex, `math:` for paper/math/main.tex and `old:`
for paper/main.tex. No paper was edited. Every fix is a proposal for the coordinator.

## How the checks were made

- **Read in full.** AGENTS.md, both papers, the old draft, the README sections they draw on (Results, Evidence, Large D,
  Other matter, How this work was done), PROOF.md, proof/A_certify.md §11.7 and §12.4–12.7, proof/B_tail.md §1–2,
  proof/D_entire.md §6, notes/LIT_REVIEW.md, notes/REF_CHECK.md, the summary and ranked issues of
  notes/NUMERICS_REVIEW.md, and the relevant parts of matter/CSS.md, SIGMA.md, ELL.md and LITERATURE.md.
- **Numbers.** A throwaway script (session scratch area, not committed) pulled every number from the body of each paper
  (preamble and bibliography excluded; `a\times10^{b}` and `10^{-b}` normalised) and searched a corpus of 857 text and
  data files (README, GAMMA, PROOF, proof/, matter/, notes/, solver/, hp/ including hp/data; paper/ excluded).
  - Every number of three or more significant digits in either paper is in a repository file. The only two misses are
    formatting: phys:373 "514 332" (README writes "514 332" with a space) and math:1815, where σ₆ is written as
    (−2.785… − 10.776628405i)×10⁻⁴ (README: −2.785…×10⁻⁴ − 1.0776628405×10⁻³i).
  - The 76 numbers that appear in both papers agree digit for digit. The disagreements found are between *different
    numbers quoted for the same quantity*; they are found by reading and are listed below.
  - Context was checked by hand for every number the physics paper adds over the old draft.
  - `python3 check_numbers.py` passes (39 of 39), but its `DOCS` still reads paper/main.tex, so **no check ties a number
    in either new paper to its source** (item C9).
- **Builds.** Both papers built once each in a scratch copy (two pdflatex passes; the tracked PDFs were not touched):
  14 and 30 pages, no undefined references or citations, no "??" in the text.

## Findings, ranked

Severity order as requested: contradiction > wrong status > missing credit > number mismatch > lost content >
duplication > cosmetic. Within a tier, the most consequential first.

### 1. Contradictions

**X1. The math paper cites its companion under a title that is not the physics paper's.**
- math:2095–2096: "Choptuik's critical collapse in $D$ dimensions: the echo period and critical exponent to $D = 61$".
- phys:20–21, the actual `\title`: "Choptuik's echo period and critical exponent at large dimension: evidence for
  $\Delta_\infty = 1.64389375$ and an $r^2$-shift symmetry behind $\gamma_\infty = 1/2$".
- The physics paper cites the math paper with its exact title (phys:831–832 against math:51–52).
- *Fix:* copy the physics `\title` into `\bibitem{companion}`. The physics title contains the number 1.64389375, so it
  must change with the Δ∞ decision below (P1); a title without the number would avoid that coupling.

**X2. The math paper says the companion's solvers compute exactly the M-harmonic (Galerkin) model; they do not.**
- math:856–863: the M-harmonic model is the equations "with $P_M$ applied to every right-hand side", and "The solvers
  of~\cite{companion} compute exactly such truncations, so statements about the $M$-harmonic model are statements about
  the systems actually solved".
- phys:134–141 and README:516–517: the double-double solver, which gives every large-D number, uses "collocation in τ on
  the half period … with no spectral truncation of the products". The f64 solver works on a Fourier grid with a cut-off
  and dealiasing (solver/CONTEXT.md:14, "kc = 23, N = 72"), which is close to, but not, $P_M$ applied to every product
  (the equations have cubic terms).
- *Fix (math:862):* "The solvers of~\cite{companion} use a Fourier truncation (double precision) or collocation
  (double-double) in τ; both converge to the same solutions as the $M$-harmonic models as $M$ grows, so the
  $M$-harmonic model is a model of the computed solutions, not the identical discrete system."

**X3. The physics paper calls the fast WKB mode exponentially small everywhere inside the light cone.**
- phys:497–498: "The non-analytic piece is a fast WKB mode, $e^{-n\Phi(x)}$ with $\Phi = \ln x - \tfrac12\ln(1-x^2)$,
  exponentially small at fixed $x < 1$."
- $\Phi < 0$ for $x < 1/\sqrt2$, so $e^{-n\Phi}$ is exponentially *large* there. math:1064–1065 (Lemma B8.2: "near the
  centre, where $e^{-n\Phi}$ is exponentially large") and README:993 say so.
- *Fix:* "…, which is exponentially small near the light cone ($x > 1/\sqrt2$) and is the singular branch $x^{2-n}$ at
  the centre." What is smaller than every power of $1/n$ at every fixed $x < 1$ is the branch $(1-x^2)^{n/2-1-s}$ alone
  (math:392–393, README:891).

### 2. Wrong or inconsistent status

**S1. The physics paper's disclosure claims checks that were made on the old draft, not on it.**
- phys:795–797: "an independent agent refereed the write-up; … and every citation and novelty claim was checked against
  the published record."
- notes/REFEREE_SIM.md:3 audits paper/main.tex (the old draft). notes/REF_CHECK.md:6–8 covers the old draft's 18
  entries, the README and PROOF.md. The physics paper's added matter references (phys:844–862) are recorded only in
  matter/LITERATURE.md ("checked by web search", LITERATURE.md:10–12). Two of its fields are recorded as unverified and
  are not marked (M6).
- The math paper's section is scoped correctly: references "taken over from the repository's reference audit" versus
  "added for this paper", with the unconfirmed ones marked (math:2081–2084). It makes no referee claim.
- *Fix (phys:795–797):* "an independent agent refereed the combined draft from which this paper was split; the
  citations taken over from it were checked against publisher or arXiv records, and those added here (other matter)
  against catalogue records, with unconfirmed fields marked."

**S2. Lemma σ₁ is "certified", but no independent review of its code is recorded, and the math paper's list of
unreviewed parts omits it.**
- math:157 (status table: "certified (computer-assisted)"), math:1891–1906 (the lemma), math:2084–2085 ("The parts of
  the proof not independently re-checked are listed in Section~\ref{sec:thmA}"; that section does not mention σ₁).
- proof/D_entire.md §6.1–6.6 records only that `exact_sigma.py`'s recursion is "independent of `amp_flint`" and agrees
  with the fitted values; no review. AGENTS.md "How results are checked": new certificate code gets an independent
  review before its claim is relied on.
- The lemma is not used by Theorem A(M = 1), so this is a labelling problem, not a gap in the theorem.
- *Fix:* the status entry "certified (computer-assisted, `exact_sigma.py`); not independently reviewed"; add σ₁ to the
  list in math:2084–2085.

**S3. The physics paper states Proposition 5(iii) without its condition.**
- phys:551–553: "its fast-mode Stokes multipliers vanish whenever the background's do~\cite{math}".
- math:763–767 and README:1479: "*Wherever the fast-type Stokes multipliers of the background are defined (by Borel
  summation; gap (a))*, the multipliers of this perturbation … vanish when those vanish."
- *Fix:* "…vanish whenever the background's do, wherever these multipliers are defined (by Borel summation, which is not
  yet proved in general~\cite{math})".

**S4. The physics paper extends Proposition 4 from the gravity-free series to "the $1/n$ series".**
- phys:503–505: "The late orders of the $1/n$ series locate its origin~\cite{Dingle1973,math}: $C(n,s)$ is the Stokes
  multiplier of a mode born at infinite self-similar radius."
- math:465–476: Proposition 4 is proved for $y = x\,{}_2F_1$, the leading-order (gravity-free) series. The nonlinear
  outer late orders are dominated by $x$-independent singularities of modulus about 0.13 (math:705–708, gap (a);
  README:1628–1640).
- *Fix:* "The late orders of the leading-order (gravity-free) $1/n$ series locate its origin".

**S5. The physics abstract gives two of the three hypotheses behind $\gamma_\infty = 1/2$.**
- phys:39–41: "which explains $\gamma_\infty = 1/2$ given a conjectured selection rule and the uniqueness of the unstable
  mode".
- The body (phys:561–563) and math's Corollary (math:822–826) also require continuity of $\lambda(D)$ at $1/n = 0$; the
  math abstract (math:70–71) says "under the conjecture and two further hypotheses".
- Otherwise $\gamma_\infty = 1/2$ is conditional in the same way, with the same three hypotheses, in both papers
  (phys:561–563; math:822–826; also GAMMA.md:166).
- *Fix (phys:40):* "given a conjectured selection rule, the uniqueness of the unstable mode and continuity in $1/D$".

Statuses that were checked and **agree** across the papers and with the sources:
- **Propositions 1–5, Conjecture 1, Corollary.** The physics paper describes each with the math paper's status (the
  ansatz for Proposition 1 at phys:485; "conjectured" for the rule at phys:572–573; "conditional" for γ∞ at phys:561).
- **Theorem B(M).** Conditional on two contractions (math:154, 1237–1240); the physics paper says only "the points where a
  proof is still missing" (phys:577–578).
- **Theorem A(M = 1).** It matches proof/A_certify.md §12.6 in both papers.
  - math:1721–1736 is §12.6 verbatim (same box, same zero enclosure, same three bullets; Braaksma is "the one fact taken
    from the literature").
  - phys:607–612 states the same strength in words (one-harmonic truncation, exactly one zero in a small box, simple,
    conditional on Braaksma's summability theorem, computer-assisted).
  - math:1757–1781 lists correctly what is not independently re-checked (Lemmas S, 6b₀, 6a; no second implementation of
    the disc covering), as A_certify.md §11.7 and §12.7 do.
- **Numerical evidence.** $\Delta_\infty = 1.6439 \pm 0.0002$ and $\omega^*/\omega_\infty = 0.9999$–$1.0001$ are called
  numerical evidence, with "a range of estimators", in both (phys:41, 596–599; math:74, 158, 2028).

### 3. Missing credit and unsupported novelty wording

**M1. Proposition 1 now lives in a paper that does not engage with the published claim it contradicts.**
- math:164–166 and math:326 credit Ecker, Ecker and Grumiller (EEG) with the leading-order freedom, and math:166 says
  "Proposition~\ref{prop:1} extends it to all orders".
- EEG take their next-to-leading-order condition (their Eq. (20)) to fix the period, and Lai and Wen adopt it. The
  rebuttal is only in the physics paper (phys:638–656), and the math paper cites neither the rebuttal nor Lai–Wen.
- notes/LIT_REVIEW.md:62–75 (claim C1, "New with required engagement"): the claim "stands only together with the
  Sec. eeg rebuttal".
- *Fix:* one sentence after Proposition 1 or in "Prior work": "Ecker, Ecker and Grumiller derive a next-to-leading-order
  condition that they take to fix the period, as do Lai and Wen~\cite{LaiWen2026}; the companion paper shows that it
  holds for every period~\cite{companion}." Add the LaiWen2026 entry (phys:842–843).

**M2. Unsupported novelty sentence in the math paper.**
- math:108–109: "The new feature is that the unknown depends on two variables, … so that the inner problem is an infinite
  system coupled through harmonics; neither of the templates we follow has that."
- notes/LIT_REVIEW.md:172–176 (C9): "I read neither template's full text: metadata only (FULL-TEXT-UNREAD), so this
  verdict is a ceiling", low confidence. AGENTS.md "Novelty claims" requires the closest prior work to be read first.
  The math paper's own disclosure (math:2081–2083) says the novelty claims were checked against prior work "read rather
  than recalled".
- *Fix:* read Xie–Tanveer and Baldomá–Capiński–Guardia–Seara and record them in LIT_REVIEW.md, or reword without
  novelty: "Unlike in those problems, the unknown here depends on two variables …".

**M3. The physics paper does not credit Gundlach for the eigenvalue formulation; the math paper does.**
- phys:131–132: "a periodic boundary-value problem with $\Delta$ as an eigenvalue", no citation.
- math:87–88, 223: \cite{Gundlach1995} for the same formulation (PROOF.md:118; REF_CHECK.md:124 verified the entry).
- *Fix:* add \cite{Gundlach1995} at phys:132, with the math paper's entry (math:2164–2165).

**M4. The physics paper names Braaksma's theorem without citing it.**
- phys:609: "conditional on Braaksma's summability theorem", no \cite.
- math:648, 1165, 2114–2115 cite Braaksma1992. AGENTS.md: "Every assumed fact (Braaksma's theorem) is named", and
  credit is given with \cite where used.
- *Fix:* add the Braaksma1992 entry to the physics bibliography.

**M5. A named numerical method without its citation in the physics paper.**
- phys:621 (Fig. selection caption) names "Padé ranges"; math:1153 cites Pade1892 for the same method.
- AGENTS.md "Standard methods": cite the original work where a method is named.
- *Fix:* cite Pade1892 in the caption, or point to the math paper.

**M6. Unverified bibliographic fields left unmarked in the physics paper.**
- phys:815–816, Bland et al. 2005, page 5355: notes/REF_CHECK.md:69, "PARTIAL … Page 5355 not verified".
- phys:859–860, Biernat–Bizoń–Maliborski 2017: matter/LITERATURE.md:626, "(page number not checked)". The entry gives no
  page, so it is not wrong, only incomplete.
- AGENTS.md "References" asks for *(unverified)*; the math paper follows this (math:2123, 2184).
- *Fix:* mark Bland et al.'s page *(unverified)*.

**M7. Outstanding credits from LIT_REVIEW.md §3 that are in neither paper.** These were missing from the old draft
too, so the split did not cause them.
- **Price & Pullin (1996), gr-qc/9601009:** an early flat-space (free-wave) approximation of the critical solution
  (LIT_REVIEW.md:53, 199). Both papers rest on that picture: phys:445–447 ("a free scalar wave on flat space") and
  math:348.
- **Kol (2006):** analytic Δ(D), γ(D) for the black-string merger (LIT_REVIEW.md:49, 200). phys:195–197 says "no earlier
  formula for $\Delta(D)$ was found"; README:447–448 adds that Kol's formula is for a different system.
- **Wu (2026), arXiv:2610.20705:** DSS at large D, quantum (LIT_REVIEW.md:47, 201).
- *Fix:* cite Price–Pullin where the free-wave reading is introduced in both papers; add Kol to the "no earlier
  formula" sentence in the physics paper, as the README does.

Credits that were checked and **agree** across the papers (same result, same people):
- Choptuik; Koike–Hara–Adachi (γ = 1/λ); Martín-García–Gundlach (Δ(4)).
- Rozali–Way: the leading-order equations, their parabolic character with space and time swapped, and γ = 1/2 from the
  divergence of their oscillating solutions (phys:67–69, 513, 525–528; math:173–175, 572–574, 776–783).
- EEG: the leading-order freedom (phys:71–72, 486–488; math:164–166, 326).
- Clark–Pimentel: CSS solutions at large D by matched asymptotics.
- Donninger: the D = 4 case of Proposition 2 (phys:479–480; math:166–167).
- Reiterer–Trubowitz: computer-assisted existence at D = 4.
- Baldomá et al.: the template of the certificate (phys:609; math:131, 1331).
- Dingle and Emparan–Herzog.
- The physics-only credits are consistent with the README and the source notes: Ecker et al. (D < 5.5), Bland et al.,
  Bland, Bland–Kunstatter, Sorkin–Oren, Lai–Wen, and the matter-model papers.
- The physics wording on γ → 1/2 (phys:72–73, 414–416) follows LIT_REVIEW.md C6 (line 152–155): "argued by Rozali and
  Way and discussed in [CP, EEG, EEGJ]".

Shared bibliography entries, diffed field by field (Choptuik1993, KoikeHaraAdachi1995, MartinGarciaGundlach2003,
GundlachReview2025, EmparanHerzog2020, RozaliWay2018, ClarkPimentel2025, EckerEckerGrumiller2026, DLMF, Donninger,
Dingle1973, Baldomá et al. 2023, ReitererTrubowitz2019, Wengert1964, Dekker1971):
- **No conflicting field.** The math entries add titles and last pages (Dekker 224–242, Wengert 463–464,
  Reiterer–Trubowitz 143–186, Donninger 511–523), all as verified in REF_CHECK.md (lines 75, 123 and the method
  addendum).
- **Keys differ but the data agree:** Donninger2008 (phys) vs Donninger2009 (math); BaldomaEtAl2023 vs
  BaldomaCapinskiGuardiaSeara2023.

**Novelty wording that was checked and is supported:**
- phys:195–196 ("no earlier formula for $\Delta(D)$ was found"): LIT_REVIEW.md addendum.
- phys:533–535 and math:175–176 (what is new over Rozali–Way): LIT_REVIEW.md C4–C5.
- math:169 ("we did not find it for general $D$ in the literature we searched"): C2, with the proposed wording.
- phys:414 ("The value $1/2$ itself is not new").
- math:1275 ("the genuinely new part") means new work still to be done, not a priority claim. It could read "the part
  with no published template".

### 4. Number mismatches

**N1. The physics paper's range for $|C|$ is wrong.**
- phys:481: "$|C| = 0.0121$–$0.0405$ for $n = 18$–$1000$".
- The values are 0.0121, **0.0114**, 0.0124, 0.0192 and 0.0405 at n = 18, 38, 58, 200 and 1000 (math:390–391;
  README:882; old:458). The minimum is 0.0114.
- *Fix:* "$0.0114$–$0.0405$".

**N2. Agreement of the matter selection-rule tests: three different statements of one number.**
- phys:714: "an agreement to $1.6\times10^{-4}$ and $1.3\times10^{-4}$".
- math:2040–2041: "agreement about $1.5\times10^{-4}$".
- The sources disagree with each other in the same way: matter/CSS.md:365 gives 1.6×10⁻⁴ (κ = 1) and 1.3×10⁻⁴ (κ = 0);
  CSS.md:40 gives 1–2×10⁻⁴; README:1578 gives about 1.5×10⁻⁴.
- From the rounded values printed in both papers (3.1998939 against 3.1994; 3.9789695 against 3.9784), a reader gets
  1.54×10⁻⁴ and 1.43×10⁻⁴. The 1.3 needs the unrounded extrapolation (about 3.97845).
- *Fix:* quote CSS.md §7 (1.6×10⁻⁴ and 1.3×10⁻⁴) in both papers, or "about 1.5×10⁻⁴" in both.

**N3. Harmonic decay of the tuned profile: 0.063 (physics) against 0.0574 (math).**
- phys:584–586: "The tuned ratios of successive harmonics are nearly the same for every harmonic,
  $|F_{\nu+2}/F_\nu| \approx 0.063$, so the decay rate per harmonic should level off near 1.4".
- math:1254–1256: the leading tuning ratio has $|q|A^2 = 0.0633$, but "the tuned profile decays by $0.0574$ per pair of
  harmonics, i.e. it is analytic on $|\mathrm{Im}\,\theta| < \sigma_F = 1.428$".
- proof/B_tail.md:31–33, 112–119: 0.0633 is the leading-order ratio $|q_\nu|A^2$. The tuned profile summed in A has
  $|F_\nu/F_{\nu-2}| = 0.0551$ (ν = 3) and 0.0572–0.0575 (ν ≥ 5). ½ln(1/0.0633) = 1.38 and ½ln(1/0.0574) = 1.43.
- *Fix (phys:585):* "the leading-order ratio $|q|A^2 \approx 0.063$ is the same for every harmonic, and summed in the
  amplitude the tuned profile decays by about 0.057 per pair, a rate of 1.43".

**N4. The physics paper gives two ranges for the double-precision solver.**
- phys:31 and phys:80: $4 \le D \le 40$. phys:134: $3.6 \le D \le 40$. README:49: "covers D = 3.6–40".
- *Fix:* one range. 3.6–40 is what the solver covers; the formula uses $D \ge 4$.

**P1. Pending decision (not a finding): every place Δ∞, ω∞ and their bar are quoted.** NUMERICS_REVIEW.md (Verdict,
F3) suggests 1.643893746. That would also move ω∞ = 2π/Δ∞ from 3.82213589 to 3.82213590.

*To update together with the decision:*
- physics:
  - phys:21, the **title** (and so the math bibitem of X1);
  - phys:34 (abstract); phys:84 (intro);
  - phys:248 (Values table, "∞" row);
  - phys:342 (Eq. dinf, Δ∞ and ω∞ with bars); phys:345 (bar wording); phys:348–350 (planted-truth sentence, "centre
    0.8–0.9×10⁻⁸ above … probably high by about 10⁻⁸");
  - phys:385 (candidate table, "none (free fit)" row; the "about 840 times" at phys:371 depends on the bar only);
  - phys:598 ("against 1.64389375").
- math:
  - math:61 (abstract); math:96 (intro);
  - math:231 (Eq. Dinf, Δ∞ and ω∞ with bars); math:233–234 (bar wording and "centre probably high by about 10⁻⁸");
  - math:2024 ("against 1.64389375").

*Not to update:*
- The math paper's **convention** $\omega_\infty := 2\pi/1.64389375$ is an exact input decimal of the certificates. If it
  changed, the certified boxes, stated in $\omega/\omega_\infty$, would no longer be the ones computed. These quotes are
  math:242, 1294, 1347, 1723 (Theorem A) and 1892 (Lemma σ₁). If the headline moves, add one sentence at math:242
  saying that the convention keeps the old decimal.
- Fit outputs and test inputs, which are not the quoted Δ∞:
  - phys:317–319: limits of individual fits in the snapshot table;
  - phys:361: the limit with a $c/\ln D$ term;
  - phys:490 and math:335: the test input $\omega_0 = 2\pi/1.64389$.

*Other physics-only numbers that NUMERICS_REVIEW.md disputes,* to decide at the same time (the math paper quotes none
of them):
- phys:144 and phys:175 (Table checks): pinning the harmonics above 13 "moves $\Delta$ by … $3\times10^{-17}$ at
  $D = 40$". The review measures +1.5, −1.1 and −1.2×10⁻¹⁵ at D = 40, 50 and 60 (summary, row 1i; F5).
- phys:146: "the estimated error per point is about $10^{-16}$ to $D = 60.1$". The review: as noise conservative, as error
  not, because of the cut's smooth 10⁻¹⁵ (row 2a; F5, F6).
- phys:176 and phys:792: "second machine … $2$–$4\times10^{-16}$". The review: 7×10⁻¹⁸ rms; the quoted figure is one ulp
  of the comparison (row 2c; F8).
- phys:339: "the limit ranges over $1.643893744$–$1.643893763$". The review: the top comes from a fit that the same rule
  rejects on cleaned data (F2).

### 5. Lost content

**L1. A dangling reference in the math paper.**
- math:448–449: "(over the interval above it multiplies it by about $40$)". No interval is given above: the integration
  range "from $x = 0.9$ to $0.72$" (old:483; README:956–957; phys:499) was dropped.
- *Fix:* "over $0.72 \le x \le 0.9$ it multiplies it by about 40".

**L2. In the old draft, but in neither new paper.** Most look deliberate; the second looks like a loss.

| old | content | still in | verdict |
|---|---|---|---|
| old:628–630 | the M = 3 zero compared with the shape extrapolated from the solutions ($A \approx 2.70$–$2.72$, $\lvert F_3/F_1\rvert \approx 0.050$–$0.058$, $\langle F^2\rangle \approx 3.65$–$3.7$) | README:1280–1282 | **loss**: math:1937–1938 gives the M = 3 zero without saying that it already has the measured shape. Propose restoring the clause in math:1938. |
| old:651–653 | the D = ∞ relation at each D's own amplitude tracks the measured $F_3/F_1$ at D = 20–60 to 0.004–0.006 (heuristic) | README:1408–1410 | deliberate trim (heuristic) |
| old:657–658 | a fit basis with only ν′ − ν = ±2, ±4, ±6 is good through σ₄ and fails from σ₆ on | README:1352–1355 | minor loss of a method caveat; math:1803 states the all-families rule without the reason |
| old:810 | "the discriminant $16\omega^2x^6F'^4$ is positive for every $F$ and $\omega$" (EEG section) | README:477 | minor loss; phys:651 keeps the perfect-square argument |
| old:483–485 | Prop. 3 check: "with the full $O(1/n)$ background the numbers are nearly the same" | README:963 | minor loss |
| old:668–669 | Padé residues 1.05–1.22; ratio estimates 2.744 − 0.026i and 2.748 − 0.013i | README:1363–1364 | deliberate trim |
| old:680–682 | Padé crossings by number of coefficients (0.981–0.991 at seven, … straddling at ten to twelve) | README:1387–1392 | deliberate trim (the bracket table is kept at math:1989–2006) |
| old:263–266 | Δ(5) from each code (3.221701945 ± 7×10⁻¹¹; 3.22170194461806, 3.22170194462672) | README:423–426 | deliberate trim (phys:271–273 keeps the agreement to 10⁻¹¹) |
| old:639 | M = 5 zero "steady to 8×10⁻⁶ in ω/ω∞ over the last steps" | README:1291–1292 | replaced by $\lvert r_1\rvert$ down to 8×10⁻⁷ (math:1943); fine |

Nothing else of substance in the old draft is missing. Every proposition, the conjecture with hypotheses (H1)–(H2),
gaps (a)–(c), the argument, every table and the six figures are in one of the two papers. The physics paper also
updates the old checks-table row: "the 94 points above $D = 60.1$" (phys:177) is right, against README:543; the old
draft's "32 points" (old:169) was a subset.

### 6. Duplication

Some overlap is unavoidable in companion papers. Under AGENTS.md "One canonical place per fact", each item below has a
proposed home; the other paper keeps one sentence and a pointer.

| # | passage | physics | math | proposed home |
|---|---|---|---|---|
| D1 | **the same two figures, with near-identical captions**: `selection.pdf` and `spectrum.pdf` | phys:616–626, 628–636 | math:2008–2017, 1963–1971 | `selection.pdf` in math (evidence for Conjecture 1); `spectrum.pdf` in physics (mostly data), cited from math:1960. Reusing a figure in two submissions needs a note at least. |
| D2 | numerical evidence for the selection rule: shape (0.0551, 0.00314 against 0.0549–0.0552, 0.0032–0.0034), period (0.9999–1.0001, bracket 0.99915–1.00011, $A_0 = 2.720$, Δ∞ = 1.6439 ± 0.0002), truncation sequence (−22%, +1.56%, −0.077%, fifteen- to twentyfold), M = 5 cross-check | phys:582–612 | math:1927–2025 | math; physics keeps the outcome in two sentences |
| D3 | the Rozali–Way identification (their r, τ, 1 − δ₀ with ρ, t, c²; Y = 2φ_t/c, S = Y_t/c; checked symbolically) | phys:525–528 | math:776–783 | math (it is the proof of Prop. 5(i)) |
| D4 | "The mode is physical, not gauge … $r_{\rm BH}^2 \propto p - p^*$ … DSS counterpart of Rozali and Way's estimate" | phys:555–560 | math:817–820 | physics (physical reading) |
| D5 | "At finite $D$ the outer region and the $1/n$ corrections … depend on $r$ explicitly … $2 + 3.0324/n$" | phys:564–567 | math:834–837 | either; the number in physics |
| D6 | Δ∞, ω∞ with bar and the bar's description | phys:341–350 | math:230–234 | physics; math quotes the value and cites (fewer places to update for P1) |
| D7 | the field equations and variable definitions | phys:120–132 | math:197–219 | both, since each paper must stand alone; fine |
| D8 | the measured scaling (3.480 → 2.962 → 2.72; ⟨F²⟩, n(1 − ⟨c(0)⟩) → 3.7; b − 1 to 3%) | phys:440–446, Table scaling | math:238–240, 246–248 | physics; math cites |
| D9 | matter tests of the rule (3.1998939, 3.9789695; σ-model 1.9225686) | phys:711–714, 735–737 | math:2037–2042 | physics; math keeps one sentence (and fixes N2) |
| D10 | $C(n,s)$ formula and values; the Weber ratio 1.00014 | phys:474–481, 531–533 | math:361–363, 389–391, 617–619 | math; physics needs the formula, but N1 shows the cost of two copies |

### 7. Cosmetic

- **C1. AI disclosure in the abstracts.** phys:43–45 names "Claude, Anthropic"; math:74–75 says "an AI model". Both
  "How this work was done" sections name Claude and Anthropic. Make the abstracts match.
- **C2. What Claude produced.** phys:788–789 lists "the code, the computations, the derivations and the text"; math:2074
  and README:1701 add "the proofs". The physics paper contains arguments (for example the $f \le 1$ argument at
  phys:644–646) and reports proved results. Use the same list in both.
- **C3. The physics paper never names the results it points to.** It cites "\cite{math}" for Propositions 1–5,
  Conjecture 1, the Corollary and Theorem A(M = 1) without their numbers (phys:477, 486, 504, 519, 548, 552, 573, 611).
  Write "\cite[Prop.~2]{math}" and so on; the README and the math paper use these names.
- **C4. Matter models in the math outlook.** math:2065–2068 calls other matter models "a natural test~\cite{companion}".
  The companion reports those tests (phys:658–762: the ℓ = 1 multiplet undecided to D = 9, the σ-model CSS branch
  ending at D = 8.54, the axion–dilaton γ → 1/2). Say so: "tested in~\cite{companion}".
- **C5. Δ → 0 wording.** phys:35: "a power-law approach to $\Delta = 0$ …, as conjectured in the literature". The
  literature conjectured Δ → 0 "with no rate" (phys:356). Write "an approach to $\Delta = 0$, as conjectured in the
  literature, is excluded if it is a power law …".
- **C6. Notation.**
  - Both papers use $s$ for $i\mu\omega$ (phys:469, math:351) and for $r^2/(2n)$ (phys:515, math:736).
  - The second heat-equation field is $S$ in the physics paper (phys:515–520) and $W$ in the math paper (math:736–743,
    where $S$ is the multiplier).
- **C7. Keys.** The Donninger bibkey says 2008 for a 2009 publication (phys:834; REF_CHECK.md:75 notes it).
- **C8. Physics build.** After the Makefile's two pdflatex passes the log still says "Label(s) may have changed. Rerun to
  get cross-references right". No reference is wrong in the text, but a third pass in paper/physics/Makefile would
  settle it.
- **C9. Tooling.** check_numbers.py `DOCS` (line 20–21) reads paper/main.tex, README.md, proof/A_certify.md and GAMMA.md
  only, so none of the new papers' numbers are tied to sources. The `dinf_quotes` and `omega_quotes` checks that guard Δ∞
  and ω∞ in the old draft should be pointed at both new papers before P1 is applied.
- **C10. A source inconsistency outside the papers.** The light-cone exponent spacing is "at least 7.27 apart in imaginary
  part" in PROOF.md:275, and $|\mathrm{Im}| \ge 6.9$ in proof/C_selection.md:399 and 884, which math:1082 follows.

## Authorship and disclosure

- **Authorship.** One author, Nathan Moinvaziri, "Independent researcher", nathan@nathanm.com, in both papers (phys:23–25;
  math:54–56). Claude and Anthropic are not listed as authors.
- **Disclosure.** Both papers have a "How this work was done" section (phys:785–801; math:2070–2089).
  - The two agree on the division of labour, that no human expert has reviewed the work, and "Corrections are welcome".
  - They differ only as in S1 (scope of the referee and reference checks), C1 and C2.

## Summary for the coordinator

| tier | items |
|---|---|
| contradiction | X1 (companion title), X2 (solvers ≠ M-harmonic model), X3 (fast mode "exponentially small") |
| status | S1 (physics disclosure claims old-draft checks), S2 (σ₁ "certified", unreviewed), S3 (Prop. 5(iii) condition), S4 (Prop. 4 scope), S5 (abstract hypotheses) |
| credit / novelty | M1 (EEG NLO and Lai–Wen absent from the math paper), M2 (math novelty sentence unread), M3 (Gundlach 1995), M4 (Braaksma), M5 (Padé), M6 (unverified fields), M7 (Price–Pullin, Kol, Wu) |
| numbers | N1 (\|C\| range), N2 (matter agreement), N3 (0.063 vs 0.0574), N4 (solver range); P1 (Δ∞ quotes: pending) |
| lost | L1 (math dangling interval), L2 (M = 3 shape comparison; minor trims) |
| duplication | D1–D10 (two shared figures; selection-rule evidence in both) |
| cosmetic | C1–C10 |

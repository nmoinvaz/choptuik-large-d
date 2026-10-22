# Fixes applied to the two papers from notes/PAPER_CROSSCHECK.md

Applied on 2026-10-22 to [paper/physics/main.tex](../paper/physics/main.tex) and [paper/math/main.tex](../paper/math/main.tex)
(and their Makefiles), following the coordinator's decisions. Passages were located by text, not by the cross-check's
line numbers. Not touched: Δ∞ and ω∞ (already updated), the numerics-review sentences (already fixed), and every number
that depends on `hp/dd_fits_mp.py`'s point loading ("823 cleaned points", the cleaning counts, "1.643893744 to
1.643893752").

## One line per finding

- **X1.** Math `\bibitem{companion}` now carries the physics paper's actual `\title` (with Δ∞ = 1.643893746).
- **X2.** Math, after Definition (M-harmonic model): the solvers are not such truncations (double precision: Fourier grid
  in τ with a cut-off; double-double: collocation in τ with no spectral truncation of the products), so the M-harmonic
  model is a model of the computed solutions, not the discrete system solved (README:522–523, solver/CONTEXT.md).
- **X3.** Physics: the fast mode is "exponentially small near the light cone (x > 1/√2) and is the singular branch
  x^{2−n} at the centre".
- **S1.** Physics "How this work was done" split into the same three parts as the math paper's (who did what; checks of
  the results; checks of the text). The text checks now say what was checked on what: a separate agent wrote a simulated
  referee report on the combined draft (REFEREE_SIM.md); that draft's citations and novelty claims were checked
  (REF_CHECK.md, LIT_REVIEW.md); the references added since were checked by web search with unconfirmed fields marked;
  after the split the numbers of both papers were cross-checked against the repository (PAPER_CROSSCHECK.md). Same
  sentence in both papers.
- **S2.** Math: status table "certified (computer-assisted, exact_sigma.py); not independently reviewed"; a sentence after
  Lemma σ₁ (separate implementation from the amplitude code, R_j agreement to 10⁻³⁹ for j ≤ 400, no review); Results
  item 7 says "not yet independently reviewed"; the disclosure lists σ₁ with Lemmas S, 6b₀, 6a, 6b₁ as not re-checked.
- **S3.** Physics: "…vanish whenever the background's do, wherever these multipliers are defined (by Borel summation,
  which is not yet proved in general) [math, Prop. 5(iii)]".
- **S4.** Physics: "The late orders of the leading-order (gravity-free) 1/n series locate its origin".
- **S5.** Physics abstract: "given a conjectured selection rule, the uniqueness of the unstable mode and continuity in 1/D".
- **M1.** Math: new Remark 3.1 after Proposition 1 (EEG's f ≤ 1 condition and their Eq. (20); Lai–Wen use it; the
  inequality holds for every regular solution, b ≥ 1 ⇒ c ≤ 1; the NLO term is the cross term of a square; the computed
  solutions violate the condition [companion]); "Prior work" points to it. LaiWen2026 entry added (physics entry and
  LIT_REVIEW.md §3).
- **M2.** Math intro: novelty sentence replaced by "Here the unknown depends on the radius and a periodic phase, and the
  inner problem is an infinite system coupled through harmonics." (no comparison with the unread templates).
- **M3.** Physics: \cite{Gundlach1995} at "a periodic boundary-value problem with Δ as an eigenvalue"; entry PRL 75, 3214
  (1995), gr-qc/9507054 (REF_CHECK.md §3, verified).
- **M4.** Physics: Braaksma1992 cited at "conditional on Braaksma's summability theorem"; entry Ann. Inst. Fourier 42, 517
  (1992), as in the math paper (REF_CHECK.md addendum).
- **M5.** Resolved by D1: the selection figure, whose caption named Padé ranges, is now only in the math paper, which
  cites Pade1892; the physics paper no longer names the method.
- **M6.** Physics: Bland et al. 2005 "5355 (2005) (page *unverified*)". Biernat–Bizoń–Maliborski 2017: added
  doi:10.1088/1361-6544/aa61ab (arXiv:1608.07707 abstract page, DOI field) and "(page *unverified*)"; the page number was
  not found on any page that could be fetched.
- **M7.** All three added, each where its content is used, after verification (below):
  - Price & Pullin 1996, both papers, where the free-wave (flat-space) reading is introduced (physics, scaling section;
    math, start of "The leading-order problem").
  - Kol 2006, physics, in the "no earlier formula for Δ(D)" sentence (his formula is for the black-string merger).
  - Wu 2026, physics introduction, with the other large-D critical-collapse work (content from its abstract, as
    LIT_REVIEW.md records).
- **N1.** Physics: |C| = 0.0114–0.0405 for n = 18–1000 (math Prop. 2 values), with \cite{math}.
- **N2.** Both papers quote 1.6×10⁻⁴ (κ = 1) and 1.3×10⁻⁴ (κ = 0), relative, from the unrounded extrapolations; the
  physics text says "before rounding". Source check and README discrepancy below.
- **N3.** Physics: "The leading-order ratio of successive tuned harmonics, |q|A² ≈ 0.063, is the same for every harmonic,
  and summed in the amplitude the tuned profile decays by about 0.057 per pair of harmonics, a rate of 1.43 per harmonic"
  (proof/B_tail.md). Math §9 now compares the measured ratios with the tuned 0.0574 as well as |q|A².
- **N4.** Physics abstract and introduction: the double-precision solver covers 3.6 ≤ D ≤ 40 (as Sec. 2 and README:55).
- **P1.** Already done on main (Δ∞ = 1.643893746 ± 1.5×10⁻⁸, ω∞ = 3.82213590); not touched.
- **L1.** Math Prop. 3 consequence (i): "for the profile above it multiplies it by about 40 over 0.72 ≤ x ≤ 0.9, the
  interval of the direct integration".
- **L2.** Math, three harmonics: restored "Its shape is close to that of the D = ∞ solution extrapolated from the data
  (A ≈ 2.70–2.72, |F₃/F₁| → 0.0549–0.0552, ⟨F²⟩ ≈ 3.7), while its frequency is 1.56% above ω∞, against 22% below with one
  harmonic". The |F₃/F₁| range is the one both papers already use, not the old draft's 0.050–0.058 (see discrepancies).
  The other L2 rows (deliberate trims) were not restored.
- **D1.** `selection.pdf` only in math, `spectrum.pdf` only in physics. Physics cites math for the selection-rule
  evidence; math says the harmonic-ratio extrapolations "are plotted in [companion]". No figure is now shared. Math
  Makefile no longer depends on spectrum.pdf.
- **D2.** Physics Sec. "The selection rule at D = ∞" reduced to the outcome (shape, period with Δ∞ = 1.6439 ± 0.0002, the
  decay rate, a one-sentence truncation remark) plus Theorem A's statement with its credits; the bracket, estimator
  families, truncation numbers and M = 5 cross-check live in math §9.
- **D3.** Physics keeps the identification with Rozali–Way in one sentence and points to the proof of Prop. 5(i) in math
  for the variable map.
- **D4.** Math reduced to one sentence ("The mode is physical, not gauge: s-translation shifts r², …") with a pointer to
  the companion for the physical reading.
- **D5.** Number in physics only: math says the 1/n coefficient "is measured in [companion]" and lists "that coefficient"
  as open; the outlook says "the coefficient of 1/n in λ".
- **D6.** Math §2.2 quotes Δ∞ = 1.643893746 and 2π/Δ∞ = 3.82213590 "to about 10⁻⁸ (an estimator range rather than a
  calibrated error; its construction is described there)"; the bar, its construction (48 fits, 0.8×10⁻⁸ shift) and
  Eq. (Dinf) are only in physics. `check_numbers.py` now has 51 checks instead of 53 (the two math ± quotes are gone).
- **D7.** No change (each paper needs the equations).
- **D8.** Math §2.2–2.3: dropped 3.480 → 2.962, "D = 12–60", "to about 3%", "→ 3.65" and the γ-bar; keeps F₁ → 2.72 and
  the common limit near 3.7 (both used later) with \cite{companion}.
- **D9.** Math "Other matter" reduced to one sentence with the N2 numbers and \cite{companion}.
- **D10.** Physics keeps the formula and the corrected |C| range with \cite{math}; the Weber ratio 1.00014 is dropped
  from physics (in math only).
- **C1.** Math abstract now has the physics abstract's sentence: "The work was done by an AI model (Claude, Anthropic)
  under the author's direction and has not been reviewed by a human expert."
- **C2.** Physics list is now "the code, the computations, the derivations, the proofs and the text", as in math.
- **C3.** Physics names the cited results: [math, Prop. 1], [Prop. 2], [Prop. 3], [Prop. 4], [Prop. 5(i)], [Prop. 5(ii)],
  [Prop. 5(iii)], [Cor. 1], [Conj. 1], [Thm. A]; "its Conjecture 1 and Theorem B"; "Conjecture 1 of [math]".
- **C4.** Math outlook: the companion "reports the tests": axion–dilaton γ → 1/2; ℓ = 1 undecided to D = 9; σ-model CSS
  branch ends at a fold near D = 8.54.
- **C5.** Physics abstract: "an approach to Δ = 0, as conjectured in the literature, is excluded by the data if it is a
  power law (exponent at least 0.01) and disfavoured if it is logarithmic".
- **C6.** Partly: the second heat field is W in both papers (physics S → W). The double use of s (iμω and r²/(2n)) is kept
  but stated explicitly where the second meaning is introduced in each paper; a rename was not made because both meanings
  are named objects in both papers and in the README (C(n,s), "s-translation", the matter table's "s-shift").
- **C7.** Physics key Donninger2008 → Donninger2009.
- **C8.** Physics Makefile runs pdflatex three times; so does the math Makefile, whose clean two-pass build also ended with
  "Rerun to get outlines right".
- **C9.** Nothing to do: check_numbers.py already reads both papers and its Δ∞/ω∞ checks cover them.
- **C10.** Math Lemma B6 text now matches `solver/proof/lightcone_exponents.py` (rerun): pairwise differences have
  |Re| < 0.02 and |Im| ≥ 6.89 (≥ 7.26 for M ≥ 3); proof of B6′: ρ < 0.02. Both sources are off (below).

## Source discrepancies for the coordinator

- **README:1601, "agreement about 1.5×10⁻⁴" (matter selection-rule test).** The data give two numbers. The medians in
  `solver/matter/data/css_fit_k1.json` and `css_fit_k0.json` (3.199388717, 3.978445470) against the inner zeros
  3.1998939020 and 3.9789695074 (matter/CSS.md §7) differ by 1.58×10⁻⁴ and 1.32×10⁻⁴ relative. matter/CSS.md §7 (1.6 and
  1.3×10⁻⁴) is right; README:1601 should quote both. matter/CSS.md:40 ("1–2×10⁻⁴") is consistent but loose. From the
  rounded 3.1994 and 3.9784 a reader gets 1.54 and 1.43×10⁻⁴, hence "before rounding" in the physics paper.
- **PROOF.md:275, "at least 7.27 apart in imaginary part".** `lightcone_exponents.py` gives 7.273 at (A, ω/ω∞) =
  (2.72, 1) but 7.267 at the M = 5 zero, and only for M ≥ 3; at M = 1 the minimum is 6.894 (6.899).
- **proof/C_selection.md:399 and 884, "|Im| ≥ 6.9 and |Re| ≤ 0.019".** The minimum |Im| is 6.894 (M = 1, at the M = 5
  zero) and the maximum |Re| is 0.01935–0.01939 (M ≥ 5), so both bounds are wrong in the last digit; "|Im| ≥ 6.89 (7.26
  for M ≥ 3), |Re| < 0.02" holds.
- **README:1304, "|F₃/F₁| ≈ 0.050–0.058" (M = 3 shape comparison).** `hp/largeD_Fratios.py` (recomputed from
  hp/data/fratios.npy) extrapolates |F₃/F₁| to 0.0579, 0.0552 and 0.0549 at degree 1, 2, 3; README:1361 quotes
  "0.0552–0.0549 (degree 1: 0.058)". No source for the lower end 0.050 was found.
- **matter/LITERATURE.md:626 (Biernat–Bizoń–Maliborski).** The DOI 10.1088/1361-6544/aa61ab can be added; the page is
  still unknown.
- **notes/LIT_REVIEW.md §3 (Price–Pullin).** The journal reference, recorded there as unverified, is now verified
  (below).

## Verification of the references added

| Entry | Checked against |
|---|---|
| R. H. Price, J. Pullin, Phys. Rev. D 54, 3792 (1996), gr-qc/9601009 | [arXiv abstract](https://arxiv.org/abs/gr-qc/9601009): title, authors, DOI 10.1103/PhysRevD.54.3792; [LSU repository record](https://repository.lsu.edu/physics_astronomy_pubs/4440): PRD 54(6), 3792–3799, 1996; the APS search listing gives the same title at [journals.aps.org](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.54.3792) (the page itself returned a client error) |
| B. Kol, *Choptuik scaling and the merger transition*, JHEP 10 (2006) 017, hep-th/0502033 | [arXiv abstract v3](https://arxiv.org/abs/hep-th/0502033v3): title, author, DOI 10.1088/1126-6708/2006/10/017; [HUJI research record](https://cris.huji.ac.il/en/publications/choptuik-scaling-and-the-merger-transition/): JHEP 2006, issue 10, article 017; IOPscience refused the fetch (robots) |
| C.-H. Wu, *Quantum critical collapse in the large D limit*, arXiv:2610.20705 (2026) | [arXiv abstract](https://arxiv.org/abs/2610.20705): title, author Chih-Hung Wu, v1 2026-10-14, no journal reference |
| Gundlach 1995, Braaksma 1992, Lai–Wen 2026 | reused: REF_CHECK.md §3, REF_CHECK.md addendum (math references), LIT_REVIEW.md §3 |

## Builds and checks

- Both papers build with their Makefiles from a clean directory (three pdflatex passes each): no errors, no undefined
  references or citations, no overfull boxes, no rerun warnings, no "??". Physics 14 pages (was 15), math 31 (was 30).
- `python3 check_numbers.py --selftest`: 51 of 51 checks pass, and each rejects a value shifted by one unit.

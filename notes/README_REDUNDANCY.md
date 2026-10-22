# README redundancy audit

An audit of README.md for facts, numbers and passages stated more than once, made on the README at commit 47c601b
(2004 lines). Line numbers below refer to that version. For each repeated item the table gives where it appears, the
place a reader looking for it would open (the canonical place, which keeps the full statement), and the action.

Result: 37 repeated items (below); 26 consolidated (merged or replaced by a pointer), 11 kept on purpose. README.md goes
from 2004 to 1974 lines. The main merges: the history paragraph absorbs the "What this builds on" list; The formula's
analyticity caveats move to "Is the period analytic in 1/D?"; the D = 4 and maximum-below-4 literature values live only
in "Relation to published work"; Proposition 5's opening absorbs "The critical exponent too", and "What is known …"
points to the Propositions instead of restating them; the two literature bullets on the large-D theory become one.

Ground rules used: the headline overview at the top (intro, history, the three questions, "In brief", the novelty list,
"How", "Headline results") may summarize; the body states each result once, in its own section, and points there from
elsewhere. Credits stay where the credited work is used (AGENTS.md: "credit prior equations and observations where they
are used, not only in the references"), so a credit at its place of use is not counted as a repeat; the overview
lists of credits are. No number, credit, caveat or link is dropped: where a passage is shortened, the full statement
is in its canonical place (the "Survives at" notes say where).

Actions: **keep** (left as is, with the reason), **pointer** (the repeat is replaced by a short pointer to the canonical
place), **merge** (two passages become one; text that is unique to either is moved, not dropped).

## Repetitions

| # | Item | Where it appears (section: line) | Canonical place | Action |
|---|---|---|---|---|
| 1 | History of the field vs "What this builds on" (Choptuik, γ = 1/λ, periodic eigenvalue problem, D ≤ 14 and near-D = 4 numerics, Rozali–Way's equations and γ = 1/2, CSS solutions, EEG's leading-order freedom and Δ → 0, existence proof, proof templates) | intro history paragraph: 13–18; "What this builds on" list: 33–40 | the history paragraph (13–18) | **merge**: one paragraph. The list's extra content moves into it: "which are the D = ∞ inner problem used here", "their heat-equation character", "the leading-order freedom of the period", the D = 4 existence proof [^10] and the proof templates [^11]. The list goes; the novelty list after it ("Not found in the literature searched") stays, introduced as "What is new". |
| 2 | Top-level summaries of the results: "In brief", the novelty list, "Headline results" | 28–31; 42–50; 60–88 | the overview (allowed to summarize) | **keep**: three granularities (four-line answer to the three questions; the novelty claims, which AGENTS.md ties to notes/LIT_REVIEW.md; the detailed headline). The body below no longer restates them (rows 3–24). |
| 3 | Δ∞ = 1.64389375 ± 1.5×10⁻⁸ (value and bar) | In brief 28; novelty list 43; headline 64–67; Results figure caption 141; The formula display 213–219; Values table 312; candidates table 386; Relation/Large D 453; Is the period analytic 648–656; Large D intro 689; inner problem 1416–1417 | derivation of the bar: Is the period analytic in 1/D? (648–656, 48 fits, `hp/dinf_budget.py`); the result: The formula display (213–219) | **keep** the overview, the display, the two tables (table rows), the one-clause uses at 453, 689, 1417, and the caption's value (check_numbers counts each `1.64389375 ± …` quote). The caption's other repeats go (rows 4, 6). |
| 4 | Δ∞ from the f64 points: 1.64395 ± 0.00008 | Results caption 141–142; The formula 235 | The formula (233–237) | **pointer**: dropped from the caption. |
| 5 | The range of the fit limits, 1.643893744 to 1.643893752 | The formula 233–234; Is the period analytic 650 | Is the period analytic (650–654) | **pointer**: The formula keeps "Δ∞ comes from fits of all the double-double points" and points to the section with the fits, residuals and bar. |
| 6 | γ → 1/2 − 0.758/n, γ∞ = 1/2 imposed in the fit, free double-double fits give 1/2 to about 5×10⁻⁷ (67 points) | headline 67; Results caption 140; The formula 210, 218, 221–231; Values 312 ("assumed"); Prop 5 1546–1548; Other matter table 1569; What is known 1639–1640 (λ∞ = 2 to about 10⁻⁶); γ figure caption 1662–1665 | The formula (221–231) for the numbers; Proposition 5 for the theory | **pointer** in the Results caption (keeps "γ → 1/2, imposed in the fit"); the What-is-known item moves into Proposition 5 (row 9). Others **keep** (display, table entries, Prop 5's own 1/n coefficient, figure caption). The checked text `0.758/n` stays at 1548 and 1583 (and in the display at 210). |
| 7 | Analyticity of Δ in 1/D: supported by the double-double points, not proven; why it is the natural default; why it is not proven; EEG's Δ → 0 | headline 65; The formula 239–248 and 252–256; Values 320–321; candidates 361; Relation/Large D 453–454; Is the period analytic 622, 639–646, 655–656; Prop 1 consequence 834; Prop 1 ansatz 845 | Is the period analytic in 1/D? | **merge**: The formula keeps the f64-era argument that goes with its extrapolation figure (alternative forms, their limits, the fan-out above D ≈ 100) and points to the analyticity section for the double-double test; its held-out numbers (5×10⁻¹³, 40–300 times better, no added non-analytic term helps) are already at 639–646. The "natural default" reasoning (no other scale; the singularity near D ≈ −9 to −11), "not proven" (exponentially small selection; EEG's Δ → 0) and "a proof needs the D → ∞ equations" move from 252–256 into 655–656, merged with the "not proven" sentence there. "The double-double points … favour the analytic form strongly" stays in The formula. One-clause conditionals elsewhere **keep**. |
| 8 | Proposition 5 summary: an exact λ = 2 mode from the r²-shift of the inner (Rozali–Way) heat equation explains γ∞ = 1/2; open: uniqueness of the mode, the 1/n coefficient 3.0324 | In brief 30; novelty list 49; headline 67–70; Contents 128; Large-D intro list 714–716; Relation/large-D theory 467–468; Prop 5 1449–1555 (open items 1548–1551); What is known "Why γ∞ = 1/2" 1641–1645; γ figure caption 1663 | Proposition 5 (with GAMMA.md) | **pointer**: the What-is-known bullet becomes a short pointer with the open items (merged with row 9). Overview, Contents, the section roadmap, the literature bullet and the figure caption **keep**. |
| 9 | The critical exponent is also selected beyond all orders; exact growth rates λ = 0 (θ-shift) and λ = 1 (time shift, the gauge mode) in the inner problem | Prop 5 opening 1451–1452; What is known "The critical exponent too" 1633–1640 | Proposition 5 (opening paragraph) | **merge**: the bullet's details (`expand_pert.py` through order 2, λ = 0 and λ = 1 with their scripts and precisions, the gauge mode 1 − ⟨G(1)⟩, λ∞ = 2 to about 10⁻⁶ from the data) move into Proposition 5's opening; the bullet merges with row 8's pointer. The link at 231 ("not fixed at any finite order", for γ) now points to Proposition 5. |
| 10 | Selection-rule bracket: ω between 0.99915 and 1.00011 ω∞ at twelve orders, best 0.9999–1.0001, amplitude 2.720, Δ∞ = 1.6439 ± 0.0002 | In brief 29; novelty list 48; headline 80–83; Large-D intro list 712–713; inner problem "Where that leaves it" 1409–1423; figure caption 1431–1437; What is known/Open 1648–1650 | The inner problem at large x, "Where that leaves it" | **keep** the overview, the section roadmap and the figure caption; **pointer** in What is known/Open (it keeps "within about 10⁻⁴ … with the measured shape" and points to the inner problem). |
| 11 | Truncation sequence: M = 1 (ω 22% below, Δ = 2.11571), M = 3 (1.56% above, Δ = 1.6186), M = 5 (Δ = 1.64516, 0.077% above, ω = 0.999228 ω∞); alternates, shrinks fifteen- to twentyfold per pair; Aitken 1.0003 | headline 76–79; inner problem 1274–1281, 1289–1298, 1299–1322 (sequence sentence 1308–1310); "Where that leaves it" 1413–1414; figure caption 1434–1435; What is known/Open 1648–1649 | The inner problem at large x, the M = 1, 3, 5 bullets | **pointer** in What is known/Open (row 10). **Keep** the headline, the figure caption and the one-clause cross-check at 1413–1414 (it carries the only explicit M = 1 ratio, 0.777, and compares estimators). The checked strings "Δ = 1.64516", "0.077% above", "A = 2.71858", "0.999228" are untouched. |
| 12 | The one-harmonic certificate (one component of the jump has exactly one simple zero in the 10⁻⁹ box), its independent review and CAPD re-check, and the two lemmas (first proved; second proved near ξ = 0; the annulus sweep remains) | In brief 30–31; novelty list 50; Contents 126–127; Conjecture details "Status of these gaps" 1153–1178; How this work was done 1717–1720 | status: the Conjecture details (1168–1178) and PROOF.md; verification: How this work was done | **pointer**: in the Conjecture details the review and CAPD re-check are named and point to How this work was done, which holds the details and the link to proof/A_review.md (the CAPD box ten times larger and §10 stay at the claim). Overview **keep**. |
| 13 | Rozali–Way: the inner equations are their leading-order large-D system; its heat-equation character; their estimate γ = 1/2 | history 16–17; "builds on" 36–37; headline 69; Relation/large-D theory 466–468; inner problem 1200–1201; Prop 5 1460, 1466–1467, proof 1501–1506, consequences 1539–1541; References 1678; [^6] | where used: the inner problem and Proposition 5; literature summary: Relation to published work | **merge** the two overview mentions (row 1). Place-of-use credits and the Relation bullet **keep**. |
| 14 | Other credits for the large-D theory: EEG's leading-order freedom; Donninger's 3+1 free wave equation (D = 4 case); Gauss's connection formula; Dingle late-order analysis | "builds on" 38, 40; Relation "The large-D theory" 464–468 and "Leading order" 469–471; Prop 1 789–790; Prop 2 900–902; Prop 4 1012, 1045 ([^14]) | where used (Props 1, 2, 4); literature summary: Relation to published work | **merge** the two Relation bullets ("The large-D theory" and "Leading order") into one (Donninger was named in both). Place-of-use credits **keep**. Overview: row 1. |
| 15 | The Δ → 0 conjecture and the evidence against it | history 18; questions 23–24; "builds on" 38; novelty list 43; headline 84–88; The formula 254–255; Relation/Large D 451–455; Relation to EEG 504–512 | Relation to published work ("Large D" bullet and the EEG subsection's rate tests) | **merge**: "builds on" into the history (row 1); The formula's mention moves into the merged "not proven" sentence of the analyticity section (row 7), which links to the EEG subsection. The rest **keep**. |
| 16 | D = 4 values against the literature: Δ = 3.445452402(3) (Martín-García & Gundlach 2003; see also Gundlach 1997), γ = 0.373961 (Ecker et al. 2026) | The formula 192–194; Relation to published work 430–432 | Relation to published work | **pointer**: The formula keeps its own values (3.4454524, 0.3739610) and points to Relation, which gains "see also Gundlach, PRD 55, 695 (1997)" and the journal reference that only The formula gave. |
| 17 | Maximum of Δ below D = 4: here D = 3.75570, Δ = 3.466774; Ecker et al. 2026 D = 3.755726, Δ = 3.466772; agreement 2×10⁻⁶ | The formula 199–203; Relation to published work 433–434 | computation: The formula; comparison: Relation to published work | **pointer**: The formula keeps the cubic, the quartic check and the rational formula's own peak, and states the 2×10⁻⁶ agreement with a pointer; Ecker et al.'s values (with their ±10⁻⁶ and Eq. (70)) are in Relation. |
| 18 | Large-D section intro paragraph vs its results list (Prop 1: all orders for any period and profile; Prop 2: hypergeometric, never analytic at the light cone) | intro paragraph 689–695; results list 699–703 | the results list (the section's roadmap), each Proposition for the full statement | **pointer**: the paragraph keeps the bold answer, its ansatz caveat and "selected by effects smaller than every power of 1/D"; the two sentences restating Props 1 and 2 go (they are the list's first two items). The headline bullet "Why the 1/D expansion cannot give it" **keep** (overview). |
| 19 | The ansatz of Proposition 1 is the measured scaling | 778–779; "Why the ansatz" 842–845; intro 691; consequence 835 | Proposition 1, after the expansion (778–779) | **merge**: the "Why the ansatz" paragraph joins the statement at 778–779. One-clause caveats at 691 and 835 **keep**. |
| 20 | Proposition 4's late-order form y_k ~ K x²(1−x²)^(−1−s) Γ(k+s+½)/χ^(k+s+½) with K | display before the statement 1032–1035; the statement 1049–1051 | Proposition 4 (the statement) | **pointer**: the earlier display is replaced by a sentence pointing to the statement (with χ = τ₀). |
| 21 | The earlier pole-sum route to Proposition 4 and its numerical checks | proof details 1080–1082; separate paragraph 1093–1095 | the proof's closing note (1080–1082) | **merge**: the check numbers (10⁻⁴⁹–10⁻⁹¹ at x = 0.3, 0.5, 0.6, k ≤ 60; 10⁻¹⁹ at x = 0.5, k = 30) move into the closing note; the separate paragraph goes. |
| 22 | The echo's shape predicted from the higher-harmonic conditions: F₃/F₁ ≈ 0.055, F₅/F₁ ≈ 0.003, against the extrapolated data 0.0549–0.0552 and 0.0032–0.0034; finite harmonic decay rate ≈ 1.4 (data 1.3–1.45) | headline 77; echo paragraph 353 (5.5%); Conjecture details "Evidence beyond the linear case" 1180–1182; M = 5 bullet 1304; amplitude expansion 1350–1356 and 1357–1365; Shape bullet 1424–1427; spectrum caption 1441–1447 | The inner problem at large x, the amplitude expansion bullets (1350–1365) | **pointer** in the Conjecture details (the claim stays, the numbers are at 1350–1365; its F₅/F₁ = 0.0031 is the Shape bullet's 0.00314 rounded); the Shape bullet's repeated data comparison becomes "data: above". Others **keep** (overview, figure text, the M = 5 cross-check). |
| 23 | F's harmonic decay rate levels off near 1.4: F∞ has singularities near Im θ ≈ 1.4, not entire | (H1) 1117; amplitude expansion 1357–1365; spectrum caption 1441–1447; What is known 1616–1619 | the amplitude expansion bullet (1357–1365) | **pointer**: What is known keeps its unique start (0.58 at D = 12 to 1.02 at D = 60, the first reading as ¼ ln n growth) and its conclusion (F∞ likely not entire), and points to the amplitude expansion for the rate 1.3–1.45 and the singularities near Im θ ≈ 1.4. |
| 24 | What is known/Open: restates Props 2 and 3 and the inner problem, and the M = 3, M = 5 and untruncated selected ω | 1646–1654 | the Propositions and the inner problem | **pointer**: the restated numbers go (rows 10, 11); "what remains" is kept word for word. |
| 25 | The double-double solver freezes the centre data's τ-harmonics above 13 from D = 35 | Evidence "No fixed-step error" 531–532; "Continuation in D" 536–537; "The weak directions" 599–601; checks table 556; spectrum caption 1446–1447 | Evidence, "The weak directions" (599–601) | **pointer**: the parenthetical at 531–532 goes ("Continuation in D" still mentions the freezing with "below"). Others **keep** (table row, a figure-specific note). |
| 26 | The points fit smooth curves to about 4×10⁻¹⁵ | headline 64; Evidence "How fast the selection weakens" 607–608; "How good the values are" 609 | "How good the values are" | **merge**: the two adjacent bullets' statements become one ("degree 18–20, all the way to D = 60.1" moves). Headline **keep**. |
| 27 | Why f64 stops working above D ≈ 26 | Evidence, double-double solver 518–521; Software, Double-double residual 1787–1788 | Evidence, The double-double solver | **pointer**: the Software subsection keeps one clause and links to Evidence. |
| 28 | Results figure caption vs The formula (fit range, held-out points, 6×10⁻⁸ against the double-double points) | caption 136–143; headline 62–63; The formula 173–190 | The formula | **keep** the figure description (it says what the curves and symbols are); only the limits are trimmed (rows 3, 4, 6). |
| 29 | "How this work was done" checks list vs the sections it summarizes (two solvers; two machines 2–4×10⁻¹⁶; two inner implementations; literature agreement; planted-truth test; check_numbers) | 1710–1726 vs checks table 557, inner problem 1254–1257, Relation 426–455, 658–665, Layout 1929–1930 | How this work was done (the record of checks); details in each section | **keep**: each item is a one-line summary with a pointer; it is the place a reader opens for "how was this checked". Gains the certificate's review link (row 12). |
| 30 | Authorship (Claude under Nathan Moinvaziri's direction; not reviewed by an expert) | "How" paragraph 56–58; How this work was done 1703–1706, 1728 | How this work was done | **keep**: the top is a one-sentence pointer. |
| 31 | Only the unstable mode's growth rate is computed, not the full spectrum | What the numbers mean 342–344; What is known 1655–1658; Prop 5 corollary 1543–1544 | What is known, "Assumed, not checked" | **keep**: one-clause caveats where the assumption is used. |
| 32 | Companion files (PARALLEL.md, AGENTS.md, PROOF.md, GAMMA.md) described twice | Contents 123–128; Layout 1895–1899 | Layout (file map) | **keep**: Contents is navigation. |
| 33 | γ unreliable above D ≈ 36 | The formula 197–199; Values 318 | The formula | **keep**: the Values note needs it for its source column. |
| 34 | Large-n form of C(n,s) | inner problem 1241–1252 (₂F₀ form); Weber form 1335–1342 | both | **keep**: the second is a different form, obtained by the duplication formula. |
| 35 | Points above D = 60.1 against the fit on D ≤ 60.1 | checks table 558 (32 points, 2.6×10⁻¹⁵); Evidence 612–613 (94 points, up to 3×10⁻¹⁵); Is the period analytic 626–627 (94 points, within 2.6×10⁻¹⁵) | — | **keep**: the statements differ (see "Inconsistencies noticed"). |
| 36 | The "Large D" bullet of "Relation to published work" vs the novelty list and the headline (finite Δ∞ if analytic; no turn toward 0 up to D = 61) | headline 84–88; novelty list 43; Relation 451–455 | Relation to published work | **keep**: the overview summarizes; Relation is the full statement. |
| 37 | Whether Δ∞ has a closed form is open; the D → ∞ equations are the way to find out | Closed forms, summary 367–368; its details 421–422 | the summary (visible text) | **merge**: "the D → ∞ limit of the equations is the way to find out" joins the summary; the details' repeat goes. |

Also looked at and kept: the echo paragraph's "31× per echo" (335–352) repeats the bullet above it (332) but is part of
a D-by-D progression (31×, 12×, 8.3×, 6.1×) that reads as one; the Large-D results list (697–716) is the section's
roadmap and only the paragraph above it was trimmed (row 18).

## check_numbers.py

No checked string was rephrased, and the checker was not touched. The quoted texts it looks for ("Δ is computed at 857
points from D = 8.25 to 61.03", "67 points, D = 8.25–61", "1.643893744–1.643893763", "3.82213589", "3.0324",
"0.758/n", "0.999228", "A = 2.71858", "Δ = 1.64516", "0.077% above") are all still present, and every quote of Δ∞ and
ω∞ with its bar is kept (the Results caption keeps its `1.64389375 ± 0.000000015`), so the number of checks is
unchanged: `python3 check_numbers.py --selftest` gives 34 of 34 pass and 34 of 34 reject a one-unit shift, and
"ties 34 headline numbers" in "How this work was done" stays true. `python3 solver/simplify_math.py README.md` keeps
406 expressions (407 before; the one removed is the |Im θ| ≈ 1.4 of row 23, which stays at its canonical place).

A mechanical check of the removed lines (every number, footnote reference, link target and script name in them still
occurs in the new README) passes. Footnote references fall from 28 to 20 because the "builds on" list repeated ten of
them; all fourteen footnotes are still referenced.

## Links

No heading was renamed, added or removed, so the Contents tree and the anchors other files use
(`#how-this-work-was-done`, `#large-d-what-the-1d-expansion-can-and-cannot-see`, `#how-the-solution-scales-with-d`,
`#proposition-5-the-unstable-mode-at-infinite-d-is-exact`, `#the-inner-problem-at-large-x`) are unchanged; all 79
internal links resolve. Two link targets inside the README changed: the γ "not fixed at any finite order" pointer now
goes to Proposition 5 (row 9), and the headline's "nine constants … are excluded" pointer now goes to "Closed forms
tested and excluded" (it pointed to "Evidence", which does not contain that test).

## Inconsistencies noticed (not changed)

These are outside a consolidation pass (they would change numbers or claims), so they are left for the authors:

- F₅/F₁ predicted by the higher-harmonic conditions is quoted as 0.0030 (amplitude expansion, line 1352), 0.0031
  (Conjecture details, 1181) and 0.00314 (Shape bullet, 1424); F₃/F₁ as 0.055 and 0.0551.
- The rational formula has eleven coefficients for Δ and nine for γ (line 173), but "Earlier formulas" (461–462) calls
  it "a ratio of polynomials with 13 coefficients".
- The checks table (558) says the 32 points at D = 60.95–61.026 agree with the degree-18 fit on D ≤ 60.1 to
  2.6×10⁻¹⁵; the analyticity section (626–627) says the 94 points above 60.1 lie within 2.6×10⁻¹⁵ of it, citing that
  table.
- The Layout says `hp/data/gamma_dd.jsonl` has "28 points D = 10-61" (1935), while the text uses 67 points
  (D = 8.25–61) from it and `hp/data/local/gamma_dd.jsonl` together.
- The Layout describes proof/A_review.md as "with the two lemmas still needed" (1918), while the Conjecture details
  (1174–1177) say the first lemma is proved and the second is proved near ξ = 0.
- What is known/Open lists "σ₈ and beyond across ω" as remaining work (1651), while σ₁–σ₁₂ have been computed across
  ω (1382–1384).
- The proof templates cited as [^11] are Baldomá–Capiński–Guardia–Seara and Xie–Tanveer; the Conjecture details
  (1159–1160) name "Baldomá–Fontich–Guardia–Seara".
- The selection conjecture is stated as "Conjecture (selection at D = ∞)" (1115) and referred to elsewhere as
  "Conjecture 1".
- The Large-D roadmap calls Proposition 4 "a derivation (not yet a full proof)" (707), while the section states it as a
  proposition with a proof ("The singulant, the exponent and the constant are proved", 1043, and the proof at
  1056–1084).
- The f64 points against the double-double values at D = 15.6–21: "5.6×10⁻¹¹" in The formula (189), "within
  5.5×10⁻¹¹" in Evidence (614).

# README reorganization plan

A proposal, not an edit. Line numbers refer to README.md at commit ea42b3e (2763 lines). The reader the plan serves is
a scientist (numerical relativity, critical phenomena, applied analysis) who wants the results, the methods, the
validation, the limits and the reproduction, not the project's operational history. Nothing is deleted: every moved
passage gets one canonical home in a notes file (part E), and every phrase that `check_numbers.py` pins is tracked
(part G).

## A. Diagnosis

The README is a lab notebook with an overview on top. It has 2763 lines (36 290 words), 30 figures, 22 footnotes plus a
22-entry reference list that repeats 18 of them, 81 internal links, 413 inline math expressions and 30 display fences,
25 `<details>` blocks that hide 1156 lines (42%), and 617 lines over 120 characters. By content, roughly: results and
their physics 560 lines (20%), theory and derivations 1100 (40%), validation 330 (12%), solver operations, software and
file layout 600 (22%), run chronology 170 (6%; run names T45–T101, machines, dates, log paths and commit hashes of the
raw-data repository, lines 274–285, 298–323, 798–882, 1878–1950). The scientific content is sound and complete, but a
reader meets the solver's stall history (829–882, five figures) before the analyticity test (1009), and the proof status
is split between 1535–1615 and 2245–2257. The overview (1–135) is itself 135 lines with 33 lines of headline bullets.

## B. Inventory

Audience: S scientific, R reproducer, I project-internal. Pins: phrases `check_numbers.py` requires verbatim (81 in all,
74 distinct; "+r" marks quotes matched by its two regex scans, see G). Links: anchors or section names other files use.

| Section (lines, n) | Contents | Aud. | Verdict | Pins | Links | Figs |
|---|---|---|---|---|---|---|
| Title, papers, intro, history, questions, in brief, new, how, headline, contents (1–135, 135) | overview; headline bullets 63–95; contents 97–134 | S | condense to 60: intro, questions, "what is new", results table (D §2) replaces 63–95 | 9 +2r (66, 67×2, 70, 71, 83×2, 85, 86) | REFEREE_SIM:210 "What is new here" (stale) | 0 |
| Results (136–169, 34) | delta.svg and echo_shapes.svg with captions | S | keep (30) | 3 +1r (146, 152, 163, 165) | — | 2 |
| The formula (170–446, 277) | formula, coefficients, tolerances; shorter forms 210–245; dd agreement 247–252; D = 4 and the maximum 254–267; low-D runs 269–331; √(D−3) fit and D → 3 limit 332–365; D → ∞ display 367–398; analyticity 400–412; power series 414–445 | S/I | split: formula + tolerances + limits (70); low D to its own section (D §5); 274–285, 298–323 to notes/LOW_D.md; 210–245, 414–445 to notes/FORMULA_FORMS.md | 10 +2r (208, 231, 248, 276, 280, 308, 359, 361–364, 376, 378) | LIT_REVIEW:332, REPRODUCTION:80, Layout 2582/2636, check docstrings "One formula", "Below D = 4" | 6 |
| Values (447–483, 37) | table and provenance | S | keep (30) | 5 +1r (452–455, 467, 472) | check docstring "Values" | 0 |
| What the numbers mean physically (484–524, 41) | physics.svg; ghost admixture 504–513; echo.svg 515–523 | S | keep 20; 504–513 to D §9 (one line) with notes/GHOST_ADMIXTURE.md; echo.svg merged with echo_shapes (F) | 0 | — | 2 |
| Closed forms tested and excluded (525–592, 68) | summary 527–538; tests, table, search 540–591 | S | condense to 12; 540–591 and candidates.svg to notes/CLOSED_FORMS.md | 1r (554) | REFEREE_SIM:99/205 (old name), BOOTLOOPS:176 quotes 537 | 1 |
| Relation to published work (593–651, 59) | point-by-point comparison | S | keep, 35, open (not `<details>`) | 0 | internal anchor ×6 | 0 |
| Relation to Ecker, Ecker & Grumiller (652–713, 62) | Eq. (20) argument; Δ → 0 rate tests 683–690; supercritical branch 693–712 | S | Eq. (20) in 12 lines under Prop 1; 683–690 to D §9; 693–712 and supercrit.svg to notes/SUPERCRITICAL.md | 0 | anchor used at 1076, 1265 | 1 |
| Evidence / The double-double solver (714–1008, 295) | how it works 726–796; benchmarks 798–816; spacing 818–827; stalls and walls 829–882; checks table 884–902; sample 904–929; what it showed 931–989; light_cone 991–998; error indicators 1000–1007 | R/I | condense to 60: 10-line method, checks table, error per point, one wall paragraph with large_d_dropoffs.svg; rest to notes/SOLVER_HP.md and notes/LARGE_D_WALL.md | 15 (772, 773, 850, 851×3, 854, 855, 858×3, 859, 969×3) | REF_CHECK:227 "Why the solver stalls"; NUMERICS_REVIEW:39; check docstring "Where the drop-offs are" | 9 |
| Is the period analytic in 1/D? (1009–1095, 87) | held-out table, bar, planted truth, transseries, data range | S | keep 50; 1037–1051 to 5 lines; 1088–1092 to notes | 5 (1059×2, 1083×2, 1085) | anchor from 2350 | 1 |
| Reproduce (1096–1108, 13) | reproduce.sh | R | keep, merged into D §11 | 0 | REPRODUCTION:117 | 0 |
| Large D intro (1109–1163, 55) | roadmap 1117–1137, notation, equations | S | keep 35 | 0 | anchor: PROOF.md:3, FIRST_ORDER.md:6, matter/STRUCTURE.md:4, matter/LITERATURE.md:6; LIT_REVIEW:22, SUPERCRITICAL:41, LARGE_D_LITERATURE:9 | 0 |
| How the solution scales with D (1164–1185, 22) | scaling table | S | keep 20 | 0 | anchor: matter/CSS.md:373 | 0 |
| Proposition 1 (1186–1266, 81) | statement, proof 1217–1245, check, consequence, comparison | S | statement + check (35); proof stays collapsed or moves (E) | 0 | novelty 1214–1215 (LIT_REVIEW C1) | 0 |
| Proposition 2 (1267–1330, 64) | statement, C(n,s), proof 1289–1313, check | S | 30; proof collapsed | 0 | novelty 1320–1322 (C2) | 0 |
| Proposition 3 (1331–1429, 99) | statement, proof 1358–1383, check table, consequences | S | 35; proof collapsed; check table 1393–1397 to details | 0 | — | 0 |
| Proposition 4 (1430–1616, 187) | late orders 1432–1457; statement; proof 1470–1500; what it says 1509–1523; Conjecture 1 1525–1533; argument and gaps 1535–1580; Theorems A(M = 1), A(M = 3) 1581–1609 | S | split: Prop 4 (35) and Conjecture 1 statement (12) stay; 1535–1615 become D §9's proof-status paragraph (30) pointing to PROOF.md | 0 | — | 0 |
| The inner problem at large x (1617–1989, 373) | equations 1619–1636; properties 1637–1675; C(n,s) 1677–1688; M-harmonic computations 1690–1770; amplitude expansion 1772–1790; details 1792–1955 (orders 13–21 log 1878–1950); three figures | S/I | 90: equations, Weber form, M-sequence table, shape prediction, one order-by-order table; 1637–1675, 1695–1770, 1817–1862 to notes/INNER_STOKES.md; 1878–1950 already in notes/PLANTED_SELECTION.md §7–13 | 29 (1747×3, 1756, 1895–1941) | anchor: GAMMA.md:74; BEYOND_LIGHT_CONE:478 names the bullet at 1808; PLANTED_SELECTION:644 | 3 |
| Proposition 5 (1990–2128, 139) | statement, proof 2041–2083, consequences, corollary, 1/n table 2098–2113, first_order.svg | S | 55; proof collapsed | 5 (2095, 2096, 2107–2109) | anchor: GAMMA.md:7 | 1 |
| Other matter at large D (2129–2209, 81) | table, matter.svg, four model paragraphs | S | 35: table, figure, 3 lines per model; details already in matter/*.md | 0 (2163 duplicates 2096) | novelty 2131–2133 | 1 |
| What is known about the selection, and what is open (2210–2294, 85) | eight bullets, gamma.svg, modecount.svg | S | becomes D §9 (40) | 0 (2260 duplicates 3.0324) | — | 2 |
| References (2295–2328, 34) | 22 entries, 18 also footnotes | S | merge with footnotes 2732–2763 into one References section | 0 | 595 points here | 0 |
| How this work was done (2329–2358, 30) | authorship, checks list | S | keep 25 | 0 | anchor: BOOTLOOPS:74; FORMAL.md:125 | 0 |
| Software: the Rust kernel + 8 subsections (2359–2714, 356) | kernel, backends, use, dd residual, Jacobian, build, performance, Layout 2520–2697, limits | R/I | 15 lines (two solvers, build, link); all of it to a new SOFTWARE.md | 0 | anchor used at 141; REPRODUCTION:144 | 0 |
| Raw data (2715–2731, 17) | raw-data repository | R | 8 lines in D §11 | 0 | — | 0 |
| Footnotes (2732–2763, 32) | [^1]–[^22] | S | keep, merged (above) | 0 | — | 0 |

## C. Facts stated more than once

Repeats the last audit left on purpose are not relisted unless they grew. Canonical place in bold.

| # | Fact | Statements (lines) | Canonical |
|---|---|---|---|
| 1 | Δ∞ = 1.643893746 ± 1.5×10⁻⁸ | 38, 46, 70, 152, 376, 467, 554, 631, 1064, 1111 | display **376**; derivation 1053–1070; table rows 467 keep; the rest one clause or a pointer |
| 2 | Selection bracket 0.999975–1.00005, Δ∞ = 1.64387 ± 0.00006 | 85–86, 1133–1134 (older 2×10⁻⁴), 1940–1941, 2265 (older 10⁻⁴) | **1939–1942**; 1133 and 2265 are out of date (twelve orders) |
| 3 | Truncation sequence M = 1, 3, 5, 7 (−22%, +1.56%, −0.077%, +0.0022%) | 83–84, 1716, 1737, 1747–1749, 1766–1768, 1867–1868, 1965–1966, 2118 | one table in **the inner problem** (new); 83 keeps one clause |
| 4 | F₃/F₁ ≈ 0.055, F₅/F₁ ≈ 0.003 (0.0030 / 0.0031 / 0.00309 / 0.00314 across lines) | 82, 522, 1611–1613, 1743–1744, 1803–1807, 1951 | **1801–1807**; pick one rounding |
| 5 | γ∞ = 1/2 to about 5×10⁻⁷; λ = 2 + 3.0324/n | 71, 383–390, 2095–2096, 2141, 2163, 2260–2261, 2282–2285 | numbers **383–390**; theory Prop 5 |
| 6 | u·κ = 1 at D = 59.9, 1.7 at 60.96, 168 at 67.74; pinned crossings 79/94/113/372 | 849–861, 877–879, 968–977 | **961–977**; the caption 848–861 repeats it with the same pinned numbers |
| 7 | Harmonics above 13 frozen from D = 35 | 779, 897, 940–945, 1004, 1987–1988, 2464 | **933–945** |
| 8 | Points above D = 60.1 against the fit on D ≤ 60.1 | 899, 983–984, 1015–1016 | **899** (checks table) |
| 9 | Point counts: 1199 / 857 / 342 at 67–68; "all 1218 points" at 813; 823 continuation points at 1056; Layout 2592–2594 | 67–68, 813, 1056, 2592 | **67–68**; 813 disagrees (1218 vs 1199) |
| 10 | Low-D continuation: D = 3.05 step 2.1×10⁻¹¹ | 276–277, 307–308 (both pinned) | one low-D section; drop one pin's README pointer (G) |
| 11 | τ modes needed toward D = 3 (M = 384 at 3.044; hundreds near 3) | 167–168, 272–277, 626–628 | **the low-D section** |
| 12 | Why f64 stops above D ≈ 26 | 718–721, 856–857, 2429 | **718–721** |
| 13 | What hp/ is and is for | 55–57, 718–724, 2359–2362, 2424–2434 | **718–724**; SOFTWARE.md for the software view |
| 14 | Only the unstable mode's growth rate is computed, not the full spectrum | 500–501, 2091–2092, 2269–2278 | **2269–2278** (the mode count now answers it at five D); 501 should point there |
| 15 | Figure → script → regeneration command | every caption (138–2293) and Layout 2639–2665 | **Layout** (SOFTWARE.md); captions name the script only |
| 16 | References vs footnotes | 2295–2328 vs 2732–2763 (18 shared) | **one References section** |
| 17 | Eq. (20) of EEG does not select Δ | 656–679, 1263–1265 | **652–681** (pointer at 1265 is fine) |
| 18 | gamma_dd.jsonl "28 points D = 10-61" vs 67 points | 2652 vs 385 | **385**; 2652 is stale (noted in the last audit, unchanged) |
| 19 | Proof status (Theorems A(M = 1), A(M = 3), B(1)) | 40–43, 1567–1609, 2245–2257, 2346–2349 | **one proof-status paragraph** (D §9); the headline 40–43 omits B(1) |
| 20 | Δ → 0 conjecture and the rate tests | 33, 46, 93–95, 629–633, 683–690, 1074–1076 | **683–690** |

## D. Proposed outline

Target 600–650 lines plus figures. "Feeds" names the current lines each section is built from.

| § | Section | Lines | Feeds |
|---|---|---|---|
| 0 | Title, one-paragraph abstract, the two papers | 12 | 1–12 |
| 1 | The problem, prior work, the three questions, what is new | 35 | 14–53 (history 21–28 and the novelty list 45–53 verbatim) |
| 2 | Main results: a table, one row per number, with status word and error-bar kind | 40 | 63–95, 367–381, 383–390, 1581–1609, 1747, 1765–1768, 1940–1942, 2091–2092, 2249–2251, 358–365, 608–620, 2269–2273 |
| 3 | The figures that carry the results: delta, echo_shapes, heldout, selection, gamma | 60 | 136–169, 1009–1035, 1957–1968, 2280–2285 |
| 4 | The formula and its domain of validity: display, coefficients, tolerance table, Values, the D → ∞ display, the analyticity caveat in one paragraph | 80 | 170–209, 247–267, 367–412 (412 → one sentence), 447–483 |
| 5 | Below D = 4 and the limit D → 3: the maximum, the √(D − 3) interpolation, the finite intercept against the power law, lowD_values and lowD_limits | 40 | 254–267, 271–273, 289–296, 332–365 |
| 6 | What the numbers mean physically | 20 | 484–502, 517–523 |
| 7 | Theory at large D: roadmap, equations and notation, scaling table, Propositions 1–5 as statements with one-line checks (proofs collapsed), Conjecture 1, the inner problem and the selection rule evaluated (M-sequence table, order-by-order table), the 1/n coefficients, other matter table | 150 | 1109–1185, 1186–1216, 1247–1265, 1267–1288, 1315–1329, 1331–1357, 1385–1428, 1430–1469, 1502–1533, 1617–1636, 1677–1693, 1772–1790, 1795–1816, 1863–1868, 1939–1953, 1980–1988, 1990–2040, 2085–2127, 2129–2156 |
| 8 | Validation and controls: checks table, error per point, held-out and planted-truth tests, bit-identity across backends and machines, two independent implementations, check_numbers, mutation controls | 60 | 729–736, 884–902, 931–989 (condensed), 1042–1051, 1053–1086, 1690–1701, 2336–2355 |
| 9 | Limits and open problems: the large-D wall (one paragraph, one figure), the analyticity assumption, the D → 3 limit, the proof status, uniqueness of the unstable mode and the mode count, other open items | 55 | 846–861, 1072–1076, 345–347, 683–690, 1535–1615 (status only), 2121–2127, 2210–2278, 504–513 |
| 10 | Relation to published work (open, not collapsed) and the Eq. (20) argument | 45 | 593–650, 652–681 |
| 11 | Reproduction: reproduce.sh, dependencies, build, data files, the raw-data repository | 30 | 1096–1108, 2476–2485, 2715–2731 |
| 12 | How this work was done | 25 | 2329–2358 |
| 13 | References (footnotes merged) | 45 | 2295–2328, 2732–2763 |

Total about 640 lines. Everything under `<details>` today that survives (proofs) stays collapsed; the 12 kept figures
are listed in F.

## E. Where the removed material goes

New files are marked (new); every other destination exists. "Proposed README text" sections already in several notes
(LOW_D §8, QUAD_DOUBLE §10, MODE_COUNT §9, FIRST_ORDER, HIGH_M §7) should be dropped or marked applied at the same time.

| Current lines | Content | Destination |
|---|---|---|
| 210–245, 414–445 | shorter 1/D forms per range; the earlier power series and its table | notes/FORMULA_FORMS.md (new); pin 231 moves (G) |
| 274–285, 298–323, 327–330 | low-D continuation narrative, node changes, lowD.svg, lowD_errors.svg | notes/LOW_D.md, new §11 "The continuation below D = 3.4"; pins 276, 280, 308 move |
| 540–591 | candidate tests, table, influence function, search; candidates.svg | notes/CLOSED_FORMS.md (new); the Δ∞ quote at 554 goes with it |
| 693–712 | supercritical branch, supercrit.svg | notes/SUPERCRITICAL.md (its summary section) |
| 726–796 | solver arithmetic, step control, Jacobian, mixed precision, τ grids, quad-double, continuation, kc | notes/SOLVER_HP.md (new); pins 772–773 move, or point at notes/QUAD_DOUBLE.md §12.1 (already in DOCS) |
| 798–827 | Newton variants bench, solve times, point spacing (3 figures) | notes/SOLVER_HP.md |
| 829–882 | why the steps collapse, drop-offs, horizon-block wall, u·κ (4 figures) | notes/LARGE_D_WALL.md (new), consolidating with notes/QUAD_DOUBLE.md §12.2–12.7 and §13 by pointers; pins 850–859, 969 move |
| 904–929 | sample of the computed points | notes/SOLVER_HP.md |
| 931–989 | the weak directions, how the selection weakens, why the solver stalls, how good the values are, the f64 points | README §8 keeps 15 lines (error per point, 978–989); the rest to notes/LARGE_D_WALL.md and notes/BEYOND_LIGHT_CONE.md pointers |
| 991–1007 | light_cone.svg, largeD_errors.svg | light_cone.svg to README §7 beside Prop 3 (option: notes/BEYOND_LIGHT_CONE.md); largeD_errors.svg to notes/SOLVER_HP.md |
| 1037–1051, 1088–1092 | transseries test; data range and run coordination | notes/DINF_FITS.md (new: the 48 fits, the bar, planted truth, transseries); README §8 keeps the verdicts |
| 1217–1245, 1289–1313, 1358–1383, 1470–1500, 2041–2083 | proofs of Propositions 1–5 | option A: stay collapsed (zero risk, +190 lines); option B: notes/LARGE_D_PROOFS.md (new) with a pointer to paper/math, which has them. The plan assumes A. |
| 1535–1580 | the argument for Conjecture 1 and gaps (a)–(c) | PROOF.md §1–2 (it is the roadmap); README §9 keeps the three gaps in one sentence each |
| 1637–1675 | inner problem properties (resummation, Kummer, Borel structure) | notes/INNER_STOKES.md (new) |
| 1695–1770 | M = 1, 3, 5, 7 computations in detail | notes/INNER_STOKES.md; README §7 keeps a 6-row table (M, ω/ω∞, A, Δ, offset) |
| 1817–1862 | harmonic-1 condition, σ values, crossing table, Padé | notes/INNER_STOKES.md (or notes/PLANTED_SELECTION.md §1 if the coordinator prefers one file) |
| 1878–1950 | orders 15–21, bullet by bullet | already notes/PLANTED_SELECTION.md §7–13; README §7 keeps a table (order, arg crossing, estimator range, note) |
| 1970–1978 | selection_zoom.svg | notes/PLANTED_SELECTION.md §7.7 |
| 2115–2119 | first_order.svg | notes/FIRST_ORDER.md (option: keep in §7, 5 lines) |
| 2158–2208 | other matter, four model paragraphs | matter/CSS.md, ELL.md, SIGMA.md, SIGMA_DSS.md (check each number is there first; the σ-model η values at 2200–2208 may not be) |
| 2212–2244 | divergence, weak directions, singularities, late orders | README §9 one line each; details already in notes/BEYOND_LIGHT_CONE.md, proof/B3_borel.md |
| 2245–2257 | layer B status | README §9 proof-status paragraph; proof/C_selection.md §13 is already canonical |
| 2359–2714 | Software, Backends, Use, dd residual, Jacobian, Build, Performance, Layout, Limits | SOFTWARE.md (new, top level); README §11 keeps the build lines 2479–2485 |
| 2715–2731 | raw data | README §11 (8 lines); rules stay in AGENTS.md |

## F. Figure inventory

| # | Line | File | Shows | Verdict | New § |
|---|---|---|---|---|---|
| 1 | 138 | solver/delta.svg | Δ(D), γ(D), formula, dd points to 73.3 | keep, first | 3 |
| 2 | 157 | hp/echo_shapes.svg | centre field over one period, 7 D | keep | 3 |
| 3 | 515 | solver/echo.svg | centre slope over several periods, 4 D | merge: same subject as #2; keep for the paper (paper/figures/echo.pdf), move to notes/SOLVER_HP.md or drop from README | — |
| 4 | 1018 | hp/heldout.svg | held-out prediction per model | keep | 3 / 8 |
| 5 | 1957 | solver/selection.svg | order-by-order bracket, truncation sequence, exponent | keep | 3 / 7 |
| 6 | 2280 | solver/gamma.svg | λ → 2 with free fits | keep | 3 / 7 |
| 7 | 486 | solver/physics.svg | e^Δ, fine-tuning per echo, field strength | keep | 6 |
| 8 | 287 | hp/lowD_values.svg | all dd values below 4 vs √(D − 3) formula, M steps | keep | 5 |
| 9 | 356 | hp/lowD_limits.svg | finite intercept vs power law, Bland point, T100 | keep | 5 / 9 |
| 10 | 1980 | solver/spectrum.svg | harmonic spectrum vs D = ∞ prediction | keep | 7 |
| 11 | 2150 | solver/matter.svg | γ(1/D) across matter models | keep | 7 |
| 12 | 846 | hp/large_d_dropoffs.svg | step collapse and u·κ with predicted crossings | keep as the one wall figure | 9 |
| 13 | 2287 | hp/modecount.svg | λ-plane count at D = 61 | keep (small) or notes/MODE_COUNT.md | 9 |
| 14 | 991 | hp/light_cone.svg | light cone, node rays, fast mode size | keep beside Prop 3 (option: move) | 7 |
| 15 | 2115 | solver/first_order.svg | 1/D corrections vs M | optional; else notes/FIRST_ORDER.md | 7 |
| 16 | 269 | hp/lowD.svg | low-D maximum, f64 bias, τ-mode cost | move | LOW_D.md |
| 17 | 325 | hp/lowD_errors.svg | last resolution step per D | move | LOW_D.md |
| 18 | 349 | hp/lowD_sqrt.svg | Δ vs s with the T100 band | move (superseded by #9) | LOW_D.md |
| 19 | 410 | solver/extrapolation.svg | f64-era forms fanning out above D = 100 | move (#4 carries the verdict) | FORMULA_FORMS.md |
| 20 | 571 | solver/candidates.svg | influence of forcing each constant | move | CLOSED_FORMS.md |
| 21 | 706 | solver/supercrit.svg | supercritical front and CSS checks | move | SUPERCRITICAL.md |
| 22 | 798 | hp/bench_variants.svg | Newton variants timing | move | SOLVER_HP.md |
| 23 | 807 | hp/solve_times.svg | hours/seconds per point by machine | move | SOLVER_HP.md |
| 24 | 818 | hp/point_spacing.svg | step in D per run | move | SOLVER_HP.md |
| 25 | 829 | hp/newton_stall.svg | Newton model along a weak direction | move | LARGE_D_WALL.md |
| 26 | 863 | hp/pole_basin.svg | horizon-series radius vs accuracy | move | LARGE_D_WALL.md |
| 27 | 873 | hp/stall_kappa.svg | u·κ and residual floor | move (#12 covers it) | LARGE_D_WALL.md |
| 28 | 1000 | hp/largeD_errors.svg | three error indicators vs D | move | SOLVER_HP.md or DINF_FITS.md |
| 29 | 1970 | solver/selection_zoom.svg | estimates as log distance from 1 | move | PLANTED_SELECTION.md |
| 30 | 1115 | — | (no figure in Large D intro) | — | — |

Kept: 12–15 figures in the order 1, 2, 4, 5, 6 (§3), 7 (§6), 8, 9 (§5), 14, 10, 15, 11 (§7), 12, 13 (§9). The seven
PDFs in paper/figures (delta, echo, gamma, heldout, physics, selection, spectrum) are all kept SVGs, so the paper build
is untouched; echo.svg must not be deleted even if it leaves the README.

## G. Risks and mechanics

**check_numbers.py.** 81 (phrase, "README.md") pins over 74 distinct phrases, plus two regex scans that check every
`1.643893746 ± …` (lines 70, 152, 376, 467, 554) and `3.82213590 ± …` (70, 378) quote and demand zero `1.64389375 ±`
quotes. The pins by section: overview 9, results figures 3, the formula 10, Values 5, dd solver 15, analyticity 5, inner
problem 29, Proposition 5 5. `DOCS` (check_numbers.py:20–22) lists the files a pin may name; a phrase that moves needs
(a) the text kept verbatim in the README, or (b) its pointer changed to the destination file and that file added to
`DOCS` in the same commit. 13 of the inner-problem pins (1895–1941) already have a twin pointer at
notes/PLANTED_SELECTION.md, so the README entry can be dropped when the bullets become a table, provided the table cell
keeps a number the check can still find or the check keeps its PLANTED_SELECTION pointer. The regex scans only read the
README, so moving the 554 quote out removes one item (allowed; no minimum count). Every stage ends with
`python3 check_numbers.py --selftest` (34 of 34 pass and 34 of 34 reject), a resolution check of the internal links
(81 today), and `python3 solver/simplify_math.py README.md` not above 413. The sentence at 2351 ("ties 34 headline
numbers") counts checks, not pins, and stays true. CI (`.github/workflows/ci.yml`) runs only cargo and the bindings
test, so these three checks are run by hand.

**Anchors and names.** Headings other files link to, to keep verbatim or to keep as `<a id="…"></a>` stubs (GitHub
honours them): `#how-this-work-was-done` (notes/BOOTLOOPS.md:74), `#large-d-what-the-1d-expansion-can-and-cannot-see`
(PROOF.md:3, notes/FIRST_ORDER.md:6, matter/STRUCTURE.md:4, matter/LITERATURE.md:6), `#how-the-solution-scales-with-d`
(matter/CSS.md:373), `#the-inner-problem-at-large-x` (GAMMA.md:74),
`#proposition-5-the-unstable-mode-at-infinite-d-is-exact` (GAMMA.md:7). Neither paper links a README anchor (only the
repository URL, physics 828, math 2246, 2258). Section names cited in prose: "Large D" (LIT_REVIEW:22, SUPERCRITICAL:41,
LARGE_D_LITERATURE:9), "The formula" (LIT_REVIEW:332, REPRODUCTION:80, Layout 2582, 2636), "Reproduce"
(REPRODUCTION:117), "Why the solver stalls" (REF_CHECK:227), "Leading structure and a prediction"
(BEYOND_LIGHT_CONE:478), "How this work was done" (proof/FORMAL.md:125), and the check_numbers docstrings ("One
formula", "Values", "Below D = 4", "What repeats with that period", "Where the drop-offs are", "The formula").
REFEREE_SIM:99, 205, 210 already cite names that no longer exist ("Candidate closed forms, all ruled out", "What is new
here"); update them in passing.

**Status-word audit.** Sentences that claim more than their status, or omit it:
- 38–39 "Δ tends to a finite limit, Δ∞ = 1.643893746" without the analyticity condition that 69–70 states; add it.
- 3 "from numerics to first principles": the first-principles value is a conjecture's output at 10⁻⁴; soften.
- 40–43 omits the certified theorem B(1) (2249–2257), which is the stronger claim; include it with its caveats
  ("certified, assuming Braaksma; second implementations of two lemmas not done; threshold n ≥ e^89.75").
- 861 "Δ(D) is smooth through every one of them": numerical evidence; say so.
- 1133–1134 and 2265 quote the twelve-order bracket (2×10⁻⁴, 10⁻⁴) while 1940 has the twenty-one-order one.
- 2141 "data: 1/2 to 5×10⁻⁷" and 71: a fit-spread estimator, not a bound; name the kind.
- 2166 "passes a second, truncation-free test": agreement 1.6×10⁻⁴, numerical evidence; say "agrees to".
- 1806 "the first number the D = ∞ selection rule has produced": chronology, reads as a priority claim; reword.
- 2008–2017 Proposition 5(i): proved by sympy (2044), so "checked symbolically" is the right word.
- 1457 "are proved" and 1127 "a proof" agree now; the last audit's inconsistency is gone.

**Novelty sentences to verify against notes/LIT_REVIEW.md before they survive.** 45–53 (C1–C6 recorded), 1214–1215
(C1), 1320–1322 (C2), 634–641 (the formula; LIT_REVIEW addendum), 2131–2133 "no large-D study of critical collapse for
any matter other than the real scalar" (matter/LITERATURE.md only; not in LIT_REVIEW's claim table, so record it or
drop "found no"), 1806 (above).

**Other risks.** (1) Moving text into a notes file that another agent owns (NUMERICS_REVIEW.md, A_review.md) would
change a review; use new files instead. (2) The Layout block (2520–2697) is the only map from figure to script; it must
land in SOFTWARE.md before captions lose their "Regenerate with" lines. (3) The redundancy audit's line numbers
(notes/README_REDUNDANCY.md) go stale; add a line there naming this plan and the commit. (4) Inconsistencies to settle,
not paper over: 813 "1218 points" vs 67 "1199"; 2652 "28 points"; F₅/F₁ roundings (C #4).

**Staged plan.** Each stage is one commit that leaves the three checks passing.
1. Move the operational blocks verbatim into notes/SOLVER_HP.md, notes/LARGE_D_WALL.md, notes/LOW_D.md §11,
   notes/CLOSED_FORMS.md, notes/FORMULA_FORMS.md, notes/INNER_STOKES.md and SOFTWARE.md, each replaced by a 1–3 line
   pointer; move the pins' pointers and extend `DOCS`; move the 16 figures with their captions. README ≈ 1700 lines.
2. Restructure the top and bottom: §0–§3 (results table with status words and error-bar kinds), §9 from "What is
   known" plus the scattered limits, §11 from Reproduce + build + raw data, §13 from References + footnotes; keep the
   five external headings or add id stubs; fix the Contents tree. README ≈ 1300.
3. Condense the theory: statements with one-line checks, proofs left collapsed, the M-sequence and order-by-order
   tables, Conjecture 1 and the proof-status paragraph. README ≈ 750.
4. Captions and figure order per F; the Values and formula sections per §4–§5; update REFEREE_SIM's stale names and
   the check_numbers docstrings that name sections. README ≈ 650.
5. Status-word and novelty pass (G), the inconsistency list, notes/README_REDUNDANCY.md addendum, and a clean-room run
   of every command the README still quotes (notes/REPRODUCTION.md protocol).

## H. The three biggest wins

1. Moving the solver operations (726–1008 less the checks table), the low-D run narrative (274–330), the order-by-order
   log (1878–1950) and Software/Layout (2359–2714) out: about 950 lines and 16 figures leave, nothing is lost.
2. A main-results table at the top (status word, error-bar kind, source) in place of the 33-line headline bullets
   (63–95), so the eleven numbers and what each one is can be read in one screen.
3. One "Limits and open problems" section that gathers the large-D wall, the analyticity assumption, the D → 3 limit,
   the proof status and the mode count, now spread over 345–347, 683–690, 846–882, 1072–1076, 1535–1615, 2121–2127
   and 2210–2278.

## Done (2026-11-04)

The five stages were applied as commits 6e12026 (stage 1), 96c7ab7 (stage 2), 58814a1 and ecf2b38 (stage 3), 955406a
(stage 4) and the stage-5 commit that carries this note, each leaving `check_numbers.py --selftest`, `check_links.py`
and `solver/simplify_math.py README.md` (308 expressions) passing. Choices made by the author: the proofs of
Propositions 1–5 stay in the README, collapsed; one new notes file (INNER_STOKES.md) holds the inner-problem
computations. Deviations from the plan: the headline bullets stay verbatim under the main-results table (their pinned
phrases), the order-by-order record keeps the twenty-first order's sentences after its table (six checks pin them), and
the four model paragraphs of "Other matter" sit under a details block rather than condensed, pending a check of their
numbers in matter/*.md. Still open from part G: the clean-room run of every command the README quotes
(notes/REPRODUCTION.md's protocol, a runner task), the long lines the moved notes inherited (a re-wrap must not touch
pinned phrases), and three pre-existing broken links inside quoted "proposed README text" blocks of
notes/SUPERCRITICAL.md and notes/PLANTED_SELECTION.md, which belong to a cleanup of those blocks.


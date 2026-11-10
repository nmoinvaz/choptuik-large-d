# HARMONY: what neighbouring fields and this project can do for each other (2026-11-07)

Six agents each compared the project with one field where a self-similar parameter is selected, as Δ is
([DINF_ANALYTIC.md](DINF_ANALYTIC.md) §5a). Each read its field's papers in both directions and ran small pilots. The
scripts are in `solver/harmony/<field>/`, and their outputs in the raw-data repository's `harmony-mac/<field>/`. Every
value below is numerical evidence: two precisions or settings agree, with the stated controls. None is an enclosure. No
one outside the project was contacted.

**In short.** No field gives a closed form for Δ∞, and the reasons are now explicit (items 2 and 3 below). Two ideas
are worth testing here. The first is a candidate second selected solution near ω/ω∞ ≈ 1.10. The second is existence
tests for the M = 5 certificate that need no derivative bounds. Going the other way, our tools gave constants that the
other fields know to 4–7 digits, to 25–55 digits, in minutes of CPU each.

## What they give us

| # | from | what | status |
|---|---|---|---|
| 1 | embedded solitons (van den Berg–King 2010; Baldomá–Guardia–Pelinovsky 2025) | zeros of S₁ arise where the two ends of its Borel arc interfere, as paired Stokes lines do; this predicts the spacing of zeros, not A₀ | a second zero of the Borel–Padé sums at A² ≈ 12.755 + 1.118i (ω∞), extrapolated real near ω/ω∞ ≈ 1.10 (A ≈ 3.68, Δ ≈ 1.49): a second selected solution, its crossing confirmed at ω/ω∞ ≈ 1.0989 by the K 45 test with both controls passing (#82, below) |
| 2 | self-gravitating Brownian particles (Sire–Chavanis 2002) | why their 1/D expansion selects an exponent and ours (Proposition 1) does not | both meet a Weber (Hermite) layer at x ~ √D. Theirs has a real index, and 1/Γ(−ν) vanishes at ν = 1 (α = 2). Ours has a complex index, and 1/Γ(1 + iμω) never vanishes. Proposed text below |
| 3 | implosion (Buckmaster–Cao-Labora–Gómez-Serrano) | their explicit resonance brackets | do not transfer: our characteristic exponents are complex, with min over j of \|Λ(ω∞) − j\| = 3.82 (μ = 1); DINF_ANALYTIC §5a corrected |
| 4 | implosion; Larson–Penston (Guo–Hadžić–Jang–Schrecker) | existence of the M = 5 zero by a sign or degree argument (Poincaré–Miranda on the box faces, or a winding number of S₁ over (A, ω)), with C⁰ enclosures only; or large validated Taylor discs in place of about 2000 Lohner steps | passed to the M = 5 builder (wip/m5-second-order), with a test that separates wrapping from curvature in T33's logs |
| 5 | Yang–Mills in odd d ≥ 11 (Bizoń–Glogić–Wasserman 2026) | self-similar profiles are roots of explicit polynomials in every odd d: an exact testbed for large-D selection and for a selected parameter's analyticity in 1/D | planted part run: d(1 − z±) → 5.0717968 and 18.928203, like 1/d; the rest is 1–2 days of code |
| 6 | DSS in explosions; Hirschmann–Eardley κ-model | a Hopf bifurcation in log-time from a continuously self-similar parent | at the κ-model's Hopf point the new cycle's period is within 0.7–1.1% of Choptuik's Δ/2 at D = 4–6 (table below). It is not Choptuik's cycle, which exists at every κ. The real scalar has no such parent (flat space has only real modes, Proposition 2) |

The Hopf points of the κ-model (`solver/harmony/dss_explosions/`; the D = 4, κ = 0 mode reproduced to 1e-11, and the
gauge mode to 3e-12):

| D | κ_c | Im λ | 2π/Im λ | Δ/2 (Choptuik) | ratio |
|---|---|---|---|---|---|
| 4 | 0.075733 | 3.6888136 | 1.703308 | 1.722726 | 0.98873 |
| 5 | 0.084306 | 3.9299236 | 1.598806 | 1.610851 | 0.99252 |
| 6 | 0.089507 | 4.2052783 | 1.494119 | 1.504742 | 0.99294 |

Also checked and of no use here: entropies and breather theorems of Ricci–harmonic flow (they do not apply on a
noncompact line, and cannot select ω); Sornette's log-periodic estimators (built for short noisy records).

**Proposed text for Proposition 1's discussion** (from item 2): "Large D selects an exponent order by order only when
the outer problem has a singular point at fixed x with an O(1) exponent, as in the Smoluchowski–Poisson case [SC02].
Here that exponent is O(n)."

**The second-zero test (2026-11-07).** T155 (win2) and T153 (macm3) computed σ₀…σ₁₆ at ω/ω∞ = 1.06, 1.08, 1.09, 1.10,
1.11, 1.12 and 1.14. `solver/harmony/stokes/second_zero_nodes.py RAW` tracks the zero x₁ of every Borel–Padé sum
[m/(16 − m)], m = 5…9, across the nodes (raw `harmony-mac/stokes/second_zero_nodes.log`, a711a2d). On the data Im x₁
changes sign in every sum, at ω/ω∞ = 1.0962–1.1001 (x₁ ≈ 13.4, A ≈ 3.66), so the criterion set in comms holds. Its two
controls, set at the same time, fail. With x₁ planted to cross at 1.095, the sums place the crossing at 1.0962, 1.1013
and, in [5/11], 1.0698, against a tolerance of 0.005. With the crossing planted at 1.15, [5/11] reports a false one at
1.1248, while the other sums correctly report none. So at 17 coefficients the method resolves a crossing to about
±0.006 at best, and [5/11] is unreliable. The candidate second selected solution near ω/ω∞ ≈ 1.10 (Δ ≈ 1.50) stays a
candidate. More coefficients at the bracketing nodes would settle it.

**The test at K 45 (#82, 2026-11-08).** win2 computed σ₀…σ₂₂ (K 45, J 1500, 1280 bits) at ω/ω∞ = 1.09, 1.10 and 1.11
(raw `I82-win2/`, fc926b6). `second_zero_nodes.py RAW --k45` tracks x₁ in the sums [m/(22 − m)], m = 8…14. Both
controls, set before the runs, now pass. The crossing planted at 1.095 is found at 1.0950 in every ring-free sum, and
the one planted at 1.15 shows no sign change in 1.09–1.11. On the data, all seven sums are ring-free and Im x₁ changes
sign in each, at ω/ω∞ = 1.0988–1.0989. Im x₁ is +0.105, −0.013 and −0.131 at the three nodes, linear to the printed
digits, so the interpolation between nodes does not limit the crossing; x₁ ≈ 13.465 − 0.013i at 1.10 (A ≈ 3.67). So S₁
has a second zero near ω/ω∞ = 1.0989, x₁ ≈ 13.46, and the D = ∞ inner problem a second selected solution there, with Δ =
Δ∞/1.0989 ≈ 1.496. This is numerical evidence from the Borel–Padé sums of the amplitude expansion, conditional on the
selection rule (Conjecture 1); the solution's stability is not known. The σ of the 1.10 node agree with macm3's K 33 J
1000 run at the same node to 20 digits through σ₅ and to 1.2×10⁻¹⁵ at σ₁₆ (#82). A discrepancy, now explained: #202
found the 1.10 node's distance from K 33 above both neighbours' at n = 6, 7, 15 and 16 (by up to 2.8×, at most 1.2×10⁻¹⁵
relative). It is the J 1000 truncation of the comparison files, not the K 45 node (#256; raw `I256-win2/`, b99c5e6).
Going from K 33 to K 39 changes σ₁…σ₁₅ by exactly 0 at all three nodes (σ_n needs the orders up to 2n + 1 only). At n =
16, |σ(J 1000) − σ(J 1500)| is 3.5×10⁻¹⁶, 1.2×10⁻¹⁵ and 4.4×10⁻¹⁶ at 1.09, 1.10 and 1.11, while #213's J 1250 on the new
firmware agrees with #82's J 1500 to 9.4×10⁻¹⁷, 4.9×10⁻¹⁷ and 5.2×10⁻¹⁷. #228's rerun of the 1.10 node on the new
firmware reproduces #82's log line by line through order 35. The worker count and the summation order move σ₁₆ of K 33
by at most 1.4×10⁻¹⁸ and leave σ₀–σ₁₅ unchanged (#229, #236, #247).

**An independent check (#203).** A second agent, given only the σ data, summed S₁ by a different route: the Laplace
transform F(t) = Σ n! σ_n tⁿ continued by quadratic Hermite–Padé approximants, then S₁ = Σ c̃_n xⁿ/n! to n = 220
(`solver/independent_second_zero/`). It finds the crossing at ω/ω∞ = 1.09884 (spread over approximant choices
1.09873–1.09894) and x₁ = 13.46562 − 0.01372i at 1.10, and its planted controls pass on ten seeds. The two methods share
only the σ coefficients and the pipeline that made them. The crossing needs σ_n to about 10⁻¹⁰ relative: 10⁻⁶ noise
moves it to 1.1008.

**The last coefficients against J (#213).** win2 recomputed the three nodes at J 1250 (K 45, 1280 bits; raw
`I213-win2/`, 6f7d14c). Over n = 18–22, max |σ_n(J 1250) − σ_n(J 1500)|/|σ_n| is 4.1×10⁻¹⁴ at ω/ω∞ = 1.09, 4.4×10⁻¹⁵ at
1.10 and 1.7×10⁻¹⁴ at 1.11, against the 10⁻¹⁰ the crossing needs. So the J truncation is not what limits the crossing.

**Controls on a family that is not Kummer (#214).** The planted controls of #203 used a Kummer family. #214 repeated
them on a Laplace integral over a curved arc with a non-Beta density, fitted to the three nodes' σ₁…σ₂₂ to 9×10⁻⁵ in log
(Kummer: 9×10⁻³), and planted by a shift in ω (`solver/independent_second_zero/arc_family.py`). Both controls pass on
ten seeds at 10⁻¹² noise (the crossing planted at 1.095 is found to 1.0×10⁻⁵, the one at 1.15 gives no sign change), and
both mutations fire. The spread rule fixed before the runs is not met: the real data's spread over approximants is up to
4.2 times the planted family's in Im x₁, outside the allowed 1/3–3. So the planted accuracy is not an error bar for the
crossing, whose uncertainty stays #203's spread over approximants, 1.09873–1.09894. The planted family reaches the real
data's spread only with about 3×10⁻¹² relative noise, while added noise of 10⁻¹² widens the real data's own spread about
sevenfold, so the real σ do not carry random error at that level. #213's J differences, about 10⁻¹⁴, are far below
3×10⁻¹², so the excess spread comes from the data's structure, not from J truncation errors in the σ (#233).

## What we can give them

| field | their open item | our pilot (agreement between runs) | next step, cost |
|---|---|---|---|
| wave maps, d = 3 (Bizoń–Chmaj–Tabor 2000) | threshold profile f₁ and its unstable mode known to 6–7 digits, mode count not proved | λ₁ = 6.33362532561945580975120280364780688460159137952 and f₁′(0) = 21.75741418187837883369727765489414698471872536097 (60 vs 80 digits, 3e-51) | certify f₁ and "one unstable mode" in a box: 3–5 days |
| Newtonian collapse (Larson–Penston, Hunter) | sonic points and the Newtonian critical exponent known to 4–5 digits; Hunter A not constructed rigorously | LP y* = 2.341117280580046198337387733076470394181338481704335123 (55 digits); Hunter A mode σ = 9.463694623877225257764499895497857251 and γ = 1/σ = 0.1056669767721546924500599639405601958 (35 digits) | certify Hunter A and its mode: 1–2 weeks; Yahil family in γ: a day |
| implosion (BCG 2025) | selected r⁽ⁿ⁾(γ) known only as brackets | r⁽³⁾(7/5) = 1.0794043933145569197958924 (25 digits; their figure's 1.079404), r⁽⁵⁾(7/5) = 1.1169976688685912479609 (22) | r⁽ⁿ⁾(γ) tables and the odd-n conjecture map: hours; a certified 10⁻²⁰ enclosure: days |
| self-gravitating Brownian particles (Chavanis–Sire 2004) | microcanonical α_max(D) known to 7 digits at D = 3 and to second order in 1/D | α_max(3) = 2.20973303559222629520713658705 (130 to 210 bits, 3e-36); the 1/D coefficients 1/2 and 11/16 confirmed, a₃ ≈ −0.2500, a₄ ≈ −3.80; Keller–Segel profiles at D = 3, z = 1814.357885871210126270700 and 167280.6751533805806219463 | α_max to D ≈ 500 and the series' radius: hours; profiles to D → 10 with mode counts: a day |
| embedded solitons (BGP 2025; Grimshaw) | inner Stokes constants known to about 4 digits | Λ = −19.968947358760960518270975287461846342544 (fifth-order KdV inner equation; Grimshaw's −19.97) and μ = 4.4985272071994845385699268397469577940511 (N = 500 vs 1000, 1e-79) | the vanishing parameters ε_n to 30 digits: a day of code |
| DSS in explosions (Oren–Sari, Yalinewich–Sari) | analyses linear in the amplitude; stability not done | none run | a nonlinear forced DSS solver: 2–4 days; complete spectra by winding counts: a day |

Each pilot's controls are in its scripts and logs. They include planted solutions recovered (the explicit wave map,
the singular isothermal sphere, the canonical Brownian profile, the resonances of BCG's review, Grimshaw's constant)
and mutations that make each check fail.

## Corrections to our own work that this found

- **paper/math, Borel density:** the endpoint exponent is β − 2, not β − 1 (fixed, 7550f49).
- **paper/math, σ₂₂ and the tail bound:** σ₂₂ exceeds the calibrated tail bound by 2.7%, which `entire_cert_K22.log`
  already showed (recorded, 7550f49).
- **DINF_ANALYTIC §5a, the resonance bracket:** it cannot work for C(n, s) (corrected, 907e0b3).
- **DINF_ANALYTIC §5a, details:** the details are corrected with this note. They are the author order of
  arXiv:1707.03792, the γ of r₃ = 6 − 2√6 (it is 5/3), the 10,000 Taylor pairs, and that 1.079404 is r⁽³⁾(7/5).
- **Open: SIGMA.md §5.** It says λ = 3 at d = 5 has no explanation. Biernat–Bizoń–Maliborski (arXiv:1608.07707) explain
  it as the reflected gauge mode. The identity is checked symbolically in `solver/harmony/magnetism/explicit_checks.py`.
- **Open: ECHO_ANALOGUES.md.**
  - Its test C.2 cannot pass on the flat D = ∞ background (Proposition 2).
  - Its failed shape law also fits a finite limit: 1 − P1 = 0.00227 + 0.254/D on the three states with D ≥ 40, and
    planted c/D data return 0. Three points cannot decide.
- **Open: the ratio law.** It drifts past k ≈ 12: |κ_est| is 1.70, 1.81 and 1.93 at k = 12, 16 and 21. So 1.675 is an
  effective type at intermediate k.

## In the literature, reported, not resolved

- Bizoń–Chmaj–Tabor print f₁′(0) = 21.757413; ours ends …414.
- Lou–Shen give the Larson–Penston sonic point as about 2.33; ours, and Maeda–Harada's, give 2.3411.
- Maeda–Harada's LP Q₀ = 0.5101; ours is 0.510315.
- Hirschmann–Eardley's κ_c ≈ 0.0754; ours is 0.075733 at D = 4.
- The 2007 review's statement that Hirschmann and Eardley's DSS cycle "smoothly shrinks" with κ conflicts with
  Choptuik's solution existing at every κ.
- Malomed et al.'s accumulation prefactor 3.27 against their fitted 2.8247.

## Decisions for Nathan

- Which demonstrations to finish, from the cost column above.
- Whether to share any result with its authors. That is a public action and his call. Drafts on request.

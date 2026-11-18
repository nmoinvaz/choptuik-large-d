# P7 scoping note: what Corollary C needs, after the RT root set at D 40.02

Queue item P7 of [PROOF_QUEUE.md](PROOF_QUEUE.md), issue #978. Written 2026-11-18 by Claude (Anthropic) under Nathan
Moinvaziri's direction. Not reviewed. It is a reading task. Nothing was run for it, and every number is quoted from the
file or issue named beside it. Status words are those of [AGENTS.md](../AGENTS.md).

## Summary

- A certified finite-D solution is necessary for Corollary C but not sufficient. Theorem B(1)'s uniqueness class exists
  only for n ≥ e^{89.75}, and only for the one-harmonic model, which the computed solutions do not solve. So C also
  needs Theorem B for the solvers' equations with an n₀ below a computable D (section 1, links L2 and L3).
- The RT system at D 40.02 imposes neither the light-cone condition nor the constraint. It fixes the phase by a row,
  has no scale freedom, and fixes the gauge by its variables and the centre by parity. Light-cone regularity is left to
  analyticity, which polynomials see only to about N^{−2 Re λ − 1} (section 3).
- The leading candidate for the two missing conditions is the scalar's horizon branch at its lowest harmonic. Its
  exponent is λ₄ = 21.12 at D 40, and N = 101 holds its k = ½ copy to 4.6×10⁻⁴⁴. One complex amplitude is two real
  conditions. The light-cone condition is the one that fixes Δ, so relaxing it moves μ, as the μ-led chords do. A
  constraint-violating family is the second candidate. Four of the tests that tell them apart read stored data only
  (section 3.4).
- Of ln n₀ = 89.75, the factor e^{1/(2u*)} at the matching point u* = 0.0081 contributes 61.73. The proof notes name
  four levers, and the largest moves the matching point outward (B5_inner.md §5.4). None is shown to reach ln n ≈ 5.5,
  where the chain is (section 4).
- The plan (section 5) starts with stage 0, read-only tests on the stored RT samples that take under an hour.

## 1. What Corollary C must state

[PROOF.md](../PROOF.md) §1 states Corollary C conditionally. If the critical solutions computed for D = 8.25–67.77 form
the branch of Theorem B, then ω∞ = ω*. Given Theorem B, that form has a one-line proof. P7 is the unconditional form.

**Corollary C (target).** There is a D_c, with n_c = D_c − 2 ≥ n₀, and a certified discretely self-similar solution z_c
of the D_c-dimensional equations, regular at the centre and analytic at the past light cone, such that z_c lies in
Theorem B's uniqueness class at n_c and agrees with the computed solution at D_c within the certificate's radius. Then
the computed solution at D_c lies on Theorem B's branch, whose frequency converges to the zero ω* of Theorem A(∞), and
Δ∞ = 2π/ω*.

| # | Link | Status | Where |
|---|---|---|---|
| L1 | Theorem A(∞): the untruncated selection condition has one simple zero in a box that contains the extrapolated ω∞ (a radius of 10⁻³ suffices, PROOF.md §6) | A(1), A(3) and A(5) **certified**, assuming Braaksma's theorem, in boxes of half-width 10⁻⁹ to 10⁻¹⁰ around ω/ω∞ = 0.7770, 1.0156 and 0.99923, none of which contains ω∞. A(∞) is open. The harmonic tail A.3 is a *conjecture* with *numerical evidence*, and route A3 is *conditional* on the coefficients and a type bound (P5) | PROOF.md §4; [B_tail_M7.md](../proof/B_tail_M7.md); [P5_scope.md](../proof/P5_scope.md) |
| L2 | Theorem B for the equations the solvers solve (every harmonic), with an explicit n₀ | B(1) **certified** for the one-harmonic model, n ≥ e^{89.75}, assuming Braaksma's theorem. B(M) **proved** for each fixed M given its lemmas, with no numerical n₀(M). Uniformity in M (gap (b)) is open, and P6 is queued | [B_chain.md](../proof/B_chain.md) §15; [C_selection.md](../proof/C_selection.md) §6, §13 |
| L3 | n₀ ≤ n_c for a D_c that a solver reaches | open. n₀ = e^{89.75} for M = 1, while the chain's states reach past D 250 ([CORRECTOR_BASIN.md](CORRECTOR_BASIN.md)) | section 4 |
| L4 | a certified solution z_c of the full equations at D_c | none in this repository. RT's D = 4 certificate is published (arXiv:1203.3766), and our D = 4 μ matches it to 7.7×10⁻²⁸ (*numerical evidence*, `solver/rt/mu_check.py`) | section 2 |
| L5 | z_c lies in Theorem B's uniqueness class at n_c | not started. For M = 1 the class is Lemma LC's 𝒦_h, a neighbourhood of the formal 1/n series in B8's weighted norms. No criterion that a computed state can be checked against is written | B_chain.md §§13, 15 |
| L6 | Theorem B's solutions form one branch in n, so that "the branch through z_c" is defined | not stated. B_chain.md §15 gives one p_n for each n, and continuity in n is not written | B_chain.md §15 |
| L7 | Braaksma's summability theorem | assumed (cited, its hypotheses checked against the paper) | C_selection.md §4.1 |

L1 to L3 are the theorem side and need no finite-D solution. L4 to L6 are P7's own. Two consequences follow.

- L4 alone does not close C. It gives an existence theorem at one D in the style of Reiterer and Trubowitz, and joining
  it to Theorem B needs L2, L3 and L5.
- PROOF.md §6's two alternatives are one. Its first ("a certified solution at D = 40–60 inside the basin of Theorem B's
  construction") needs that construction at n = 38–58. The construction exists only for n ≥ n₀, so the first
  alternative is the second ("an explicit n₀ and a certified solution at some D ≥ n₀ + 2").

## 2. The routes to a certified finite-D solution

| Route | What it has shown | Where it stops |
|---|---|---|
| RT prototype, `solver/rt/` ([RT_FORMULATION.md](RT_FORMULATION.md)) | the D-dimensional equations **checked symbolically** against the covariant ones (§2). On 32 × 101, Δ within 3.5×10⁻¹⁵, 3.3×10⁻¹⁵ and 2.1×10⁻¹⁴ of hp at D 40.02, 50.01 and 55.00 (§10.2, *numerical evidence*) | finer grids' roots are not unique (§10.3). Taylor rows of order 7 at the horizon do not restore uniqueness (§13.5). At D 40.02 the roots form a set of at least two dimensions at the floor (§14, #968, #969). A discrete system without an isolated root admits no isolated-root certificate |
| Octo-double chain states, `hp/` ([OCTO_DOUBLE.md](OCTO_DOUBLE.md) §5, [CORRECTOR_BASIN.md](CORRECTOR_BASIN.md)) | roots of the pinned M = 32 multiple-shooting system, to the octo-double corrector's tolerance of 10⁻⁴⁰ in pinned max \|F\| (AGENTS.md). The horizon series removes the horizon branch exactly (RT_FORMULATION.md §4) | no validated mode exists ([SOLVER_SURVEY.md](SOLVER_SURVEY.md) §4 item 1, not started). The states solve a pinned system with held harmonics (OCTO_DOUBLE.md §5.1, [RESEED.md](RESEED.md)), not the full truncated one. No bound exists on the τ tail above M = 32 or on the remainders of the centre series, the horizon series and the Taylor steps. The weak direction's smallest LU pivot falls as e^{−1.01 D} (CORRECTOR_BASIN.md) |
| Theorem A's boxes, with Theorem B(1) | n-free inner zeros **certified** at M = 1, 3 and 5. With B(1), solutions of the one-harmonic model **certified** for every n ≥ e^{89.75} | the model and the threshold. These are the D = ∞ end of a branch, not a finite-D solution of the full equations (L1, L2) |

The contrast with RT's own proof explains the first row. At D = 4 the horizon exponents are λ₂ = −0.41 and
λ₄ = 0.63 (RT_FORMULATION.md §13.1). Both branches are far from polynomials, so analyticity alone enforces the light
cone. At D 40 they are 5.32 and 21.12, and the second branch is nearly a polynomial (section 3).

## 3. The RT degeneracy

### 3.1 What a DSS problem of this form fixes

The unknowns are periodic in a logarithmic time, and the period Δ is an eigenvalue (Gundlach, PROOF.md §3).

1. Phase and scale. A time translation is free. A change of the spacetime scale is the same freedom, since
   (t, r) → (λt, λr) shifts the logarithmic time by −ln λ. One real condition fixes both.
2. Gauge. The time coordinate is free. Our gauge fixes it by c(1) = 1, the light cone at x = 1 for every τ. RT fix it
   by null coordinates on which the self-similarity acts linearly. RT Sec. 1.4 shows that such a coordinate is unique
   up to a constant factor when du ≠ 0 across the light cone, and that factor is the translation of item 1.
3. Centre regularity, which fixes half the data at the centre.
4. Light-cone regularity, the selection condition. It is one complex condition per harmonic of the singular branch
   (C_selection.md §6, C_{n,μ} = 0 for every odd μ), and it is what fixes Δ.
5. The constraints, which a first-order system must impose or propagate.

A shift of the accumulation time gives the λ = 1 mode (PROOF.md §2), which grows like e^τ. It is not periodic, so it is
no freedom of the periodic problem.

### 3.2 What the RT system fixes

`solver/rt/rt_model.py` has the unknowns ω₁ on ξ > 0 (odd in ξ), ω₂⁻, ω₃⁻ and ω₄⁻ on all N + 1 points, and μ. Its
rows are (1 + P)Ω₁⁺/ξ on ξ > 0, Ω₂,₃,₄⁺/ξ at every point, and the phase row. The system is square.

| Condition | How the RT system fixes it | Could it supply the two missing conditions? |
|---|---|---|
| 1, phase | the row Re(ω₄)₁,₀ = 0 (`phase_w`, cos(t/2) against the Chebyshev T₀ weights). The field rows leave one free direction by count, and the row fixes it | no, by count. Test T0b |
| 1, scale | none needed. μ is a geometric invariant, and ξ = 1 is the light cone by construction (RT_FORMULATION.md §1) | no |
| 2, gauge | built into the unknowns, unique up to the translation of item 1 when du ≠ 0 across the light cone (RT Sec. 1.4). That uniqueness rests on regularity at the light cone, as row 4 does. ω₃'s horizon branch, λ₃ = −1 − ik/μ, is unbounded (§13.1) | not separately. A gauge mode bounded at the horizon would be a bounded branch of row 4, and T0b covers the translation |
| 3, centre | parity, ω⁺ = −Pω⁻ and ω₁ odd, exact on the grid (N odd, so ξ = 0 is not a point). ω₁'s equation is characteristic at the centre (§13.1). The 1/ξ terms of Ω₂ and Ω₄ in `rt_model.py` give the odd parts of ω₂ and ω₄ the centre exponents −2 and −(D − 2), unbounded (a hand reading, not checked symbolically) | not expected. #423 found the rough x in ω₄ near ξ = 0 at D 50 and 55, so T1's vectors should be read at the centre too |
| 4, light cone | no row. At ξ = −1 the collocated equation is the only regularity condition (`rt_model.py` docstring). A bounded branch s^λ with s = 1 + ξ is excluded only by analyticity, which degree-N polynomials see to about N^{−2 Re λ − 1} (§4). At D 40, λ₂ = 5.32 (ω₂'s branch) and λ₄ = 21.12 (the scalar's, §13.1). `--horizon 7` excludes λ₂'s branch but not λ₄'s (§13.2) | yes, the leading candidate (H4) |
| 5, constraint | not imposed. Ω₅⁺ (RT's Ω₂♯) is left to RT Sec. 2.3's argument that it then holds (`rt_constraint.py`). Which regularity that argument uses is not recorded in this repository | possibly (HC) |
| discrete, Nyquist | the periodic fields' Nyquist mode has zero discrete t-derivative (`kp[Nt // 2] = 0`). It is a weak direction that RT's coefficient space does not have (§4) | unlikely, since the pair differences are smooth and at harmonic 0 (§10.3, §13.7). Test T0b |

### 3.3 What the D 40.02 data show, and what they do not

- The deepest direction is null at the root. At samples a, b and c (K = 0, 32 × 121) the median rough bounds are
  1.5×10⁻³⁵ to 2.6×10⁻³³ (§14). A random move of 10⁻²⁰ of max |z| lifts the bound to 5.5×10⁻²⁴ (#963). So a
  root-level bound reads the stored state's distance from the singular set, not J's gain at an exact root.
- J stays near-singular along two independent chords. Moves along a − c and b − c keep the bound at the root's level,
  10⁻¹⁰ to 10⁻¹² of the random moves' (#968, #969). The chords' cosine is 0.234.
- The second chord's own gain is bounded only by the residual floor. The samples are roots to max |F| ≤ 6.9×10⁻³¹ at
  192 bits (§14). For this quadratic residual F(a) − F(c) = J(m)(a − c) exactly, with m the midpoint (§10.3). So the
  residuals over the separation, about 1.7×10⁻²⁰ of max |z|, bound J's gain along each chord. That bound lies far below
  a random direction's gain and far above the deepest one. The data do not show whether the second direction is as
  deep as the first, and test T1 decides it.
- This is the local picture of a near-root set of at least two dimensions, as §14 reads it. An exact manifold reads the
  same as an isolated root whose two weakest gains and the matching second-order terms lie below the floor. Either one
  rules out an isolated-root certificate on this grid at this precision.

#972 (running) tells a tangent plane from two crossing curves. The tests below do not depend on its outcome.

### 3.4 Candidates, and the tests that tell them apart

**(H4)** The scalar's horizon branch at its lowest harmonic. ω₄ is antiperiodic (`kanti = i(k + ½)`), so its branch
has harmonics ±½, ±3/2 and so on, with no harmonic 0. The lowest pair carries one complex amplitude, which is two real
conditions. Three facts support it.

- N = 101 holds the k = ½ branch at D 40 to 4.6×10⁻⁴⁴ (§13.7), below every gain measured.
- Rough right-hand sides find the direction, and smooth ones miss it by a factor of about 10¹⁵ (#423). The docstring of
  `rt_sigma.py` expects this of a polynomial copy of a horizon branch.
- The light-cone condition fixes Δ (3.1, item 4). Relaxing it moves μ at first order, and the chords and the weak x are
  μ-led in the max norm (§14).

Against it, #423's prediction (H) failed. x was not at ξ = ±1, and the root-level bound did not follow λ₄ in D. The D
trend rests on root-level bounds, which #963 showed read the distance from the singular set (3.5). The location
criterion assumed that the copy sits at the horizon, while s^{21.12} vanishes there to order 21. So neither result
excludes (H4).

**(HC)** A constraint-violating family. If RT Sec. 2.3's argument uses analyticity at the light cone, a
constraint-violating solution with a bounded horizon branch is a near-root of the collocation. (HC) predicts that Ω₅⁺
changes at first order along the chords. (H4) predicts that it stays at its root level, since the relaxed solutions
are solutions of the full equations. #163's check found Ω₅⁺ 2.8×10⁻²¹ apart between #145's two 32 × 121 roots at D 50
(§13.7). It had no control move, so it does not decide.

**(H2)** λ₂'s branch as the second chord. At K = 0 the smooth-v bound is 5.9×10⁻¹⁸ at D 40.02 on 32 × 101 (§13.7),
below the chords' residual-floor bound, and the K = 7 rows exclude λ₂'s branch. Under (H2) there is one deep direction,
λ₂'s branch supplies the other chord, and the K = 7 samples' differences are parallel.

**(HP), (HN)** Phase or Nyquist. These predict that the chords project onto ∂_t z or onto the Nyquist modes.

| Test | What it reads | Data | Cost | Reading (to be fixed before the run) |
|---|---|---|---|---|
| T0a | the cosine of a₇ − c₇ and b₇ − c₇, the K = 7 samples of §14 (5.6–5.9×10⁻¹⁹ apart) | `I671-win/solves/k7_D40_*`, `I701-win/solves/k7b_D40_*`, `I951-win/solves/k7c_D40_*` | seconds | parallel: (H2). Not parallel: λ₂'s branch is not the second chord |
| T0b | the share of a − c and b − c along ∂_t z(c) (the phase orbit, μ part zero) and along the Nyquist modes | the K = 0 roots beside them | seconds | a large share: (HP) or (HN) |
| T0c | `rt_constraint.py` on a, b, c and on #963's 10⁻²⁰ random move (`rt_perturb.py --delta 1e-20` with #963's seed regenerates it). The change of Ω₅⁺ per unit move along each chord, against the random one | the same | minutes at 192 bits | comparable to the random move: (HC). Many orders smaller: not (HC) |
| T0d | `rt_branch_cheb.py` at λ₄ = 21.12 with ω = k/μ for k = ½, 3/2 and 5/2 at N = 121, then at the λ₄ of §13.1's table against D | none | seconds | the number of λ₄ pairs below the 10⁻³⁵ floor (two real directions per k), and the D at which the k = ½ pair crosses it on 32 × 121 |
| T1 | the count. `rt_sigma.py` with its rough x saved, then the singular values of the block of x at sample c | sample c | a few lines of code, then about 40 min on win (#969's runs) | the number of directions at the deep level. (H4) predicts T0d's count |
| T2 | the onset. #963's random move at D 30 on 32 × 121 (`I145-mac/cont/cont_32x121_D30.0000_32x121.npz`, root-level bound 1.35×10⁻³¹ in #423), and #968's chord move to a second D 30 root from another start | that state | one solve and three `rt_sigma.py` runs, about 3 h on win | (H4) predicts a null direction above T0d's crossing D and none below it |

Whichever candidate holds, a certificate needs the missing condition imposed explicitly. Under (HC) that means Ω₅⁺
rows. Under (H4) it means the light-cone condition to an order above λ₄, which is 22 at D 40. Taylor rows in the
Chebyshev basis cannot supply it, since a row of order n amplifies rounding by about N^{2n} (§13.2) and the order-30
pilot failed at D 55 (§13.4). A horizon patch with its own Taylor unknowns can, as hp's horizon series does.

### 3.5 A discrepancy (reported, not fixed)

RT_FORMULATION.md §13.8, second bullet, and #423's verdict take the D trend of the root-level rough bound as evidence
against λ₄'s branch. That trend is flat from D 40 to 50 on 32 × 101 and −0.57 decades per 10 in D on 32 × 121. #963
then showed that at D 40.02 the root-level bound reads the stored state's distance from the singular set, and §14
extends that reading to D 50 and 55. So those trends measure the solves' floors, and they neither exclude nor support
λ₄'s branch.

## 4. The gap in n

### 4.1 Where ln n₀ = 89.75 comes from (M = 1)

| Piece | Value | What sets it | Source |
|---|---|---|---|
| e^{1/(2u*)} at the fixed matching point u* = 0.0081 | e^{61.73} | u* = Z cos κ with Z = 0.0085, inside Z_max ≈ 0.009, the largest disc in z = x²/n on which B5(1)'s perturbative fixed point around the inner series closes. Its two limits are lemma S's coverage and the loops of the inner variational operator through the fast row | B_chain.md §3.2; [B5_inner.md](../proof/B5_inner.md) §5.4 |
| x_Φ = (39784, 38776), u*^{1−μ₊} = 33.84, ‖V⁻¹‖ ≤ 1.481 | with the row above, C_E = e^{76.91} (e^{76.94} for d₀ = 10⁷) | B5(1)'s super-solution for B8's (H8) data, δ_Φ = 79 to 122 | B_chain.md §§3.2, 4 |
| B7's margin | from ln C_E = 76.94 to ln n₀ = 89.75 | the degree condition on ∂X, X = x̂ ± 10⁻⁹. The multiplier on ∂X, bounded below through min \|J_D\| ≥ 2.63×10⁻¹¹ and ‖ℓ̃‖ ≤ 1.626×10⁻⁶, times 1 − ‖R‖, must exceed ‖V‖C_E/n | B_chain.md §8.3 |
| B8, (H8), B5(1), Lemma LC | n ≥ e^{73.89}, and B8 alone between e^{58} and e^{62} | ln n ≥ κ₁/(Z cos κ). B8's own limit is the far-region fast-kernel gain γ_F | B5_inner.md §5.3; [B8_constants.md](../proof/B8_constants.md) §§7.1–7.3 |

The structure is one competition. At u* the jump has relative size e^{−1/(2u*)}, and the matching error is O(1/n), so
n₀ is where C·e^{1/(2u*)}/n falls below B7's margin. The exponent 1/(2u*) dominates, and u* is set by how far from
the inner series B5(1) can be proved.

### 4.2 The levers the notes name

| Lever | Effect on ln n₀ | First step | Status |
|---|---|---|---|
| B5(1) on a fixed sector \|z\| ≤ Z_b with the exact inner variational flow, the jump evaluated at u* = Z_b | 61.73 becomes 1/(2Z_b), which is 12.5 at Z_b = 0.04, the reach of lemma 6b₁'s 178 discs (A_certify.md §12), and 5.6 at u = 0.09 (Theorem A's ξ₁ = 0.3). κ₁ disappears, and the threshold becomes polynomial in the constants (B5_inner.md §5.4) | the propagator on the disc covering by Cauchy estimates, as B5_inner.md §5.4 outlines ("the natural next step. It was not attempted here") | not started |
| S^{(1)}, the coefficient of 1/n | the error becomes O(e^{1/(2u*)}/n²), which roughly halves ln n₀ | a lemma-S-type fixed point for the inhomogeneous linearized inner problem, then its validated integration (B_chain.md §5) | not started |
| a larger Theorem A box | raises B7's margin with the box's half-width, up to B5(1)'s parameter box 𝒦 (a polydisc of 10⁻⁶ around X), beyond which B5(1) and B8 must be rerun | the second-order chart that certified A(5) ([A_M2.md](../proof/A_M2.md)), applied at M = 1 | not started |
| B8's γ_F | B8's own threshold, which binds once B5(1)'s falls below e^{62} | sharpen the arc part and the S-forcing on the far region (B8_constants.md §7.3) | not started |

The values 12.5 and 5.6 are 1/(2Z_b) evaluated by hand, and so is n ≈ 270 below.

### 4.3 Reading

With the first lever at its best, e^{1/(2u*)} is e^{5.6}, so n ≈ 270 before the other constants. Those constants are
not small. x_Φ is near 4×10⁴, B8's h is 10⁵ to 10⁸ (B8_constants.md §7.1, B_chain.md §4), and B7's margin remains. No
note estimates the combined threshold, and none claims a route to ln n₀ ≈ 5.5. B(1) is also not the theorem C needs
(L2). So L3 needs the first lever carried out and measured before any finite-D certificate is aimed at a particular D.

One different route is not examined in any note. In inner variables the exact equations are the inner problem plus
n⁻¹𝓠₁ + n⁻²𝓠₂ (C_selection.md §9, B5.0, **proved**). So at finite n the inner region is a perturbation polynomial in
1/n. A validated computation there over 1/n ∈ [0, 1/n_c], joined to B8 outside, might replace the asymptotic matching
in the inner region. Whether it can is a scoping question for after stage 5.

## 5. Staged plan

Costs are estimates. Times on win are from #969's runs.

| Stage | Goal | First runnable step | Cost | Decision it informs |
|---|---|---|---|---|
| 0 | read the RT degeneracy from stored data | T0a to T0d (3.4), with a read-only script in the manner of #972's `check_972.py` | under 1 h on one core, no solve | which candidate the chords carry, and whether T1 and T2 are needed |
| 1 | count the missing conditions at D 40.02 | T1 | a few lines in `rt_sigma.py`, then about 40 min on win | two deep directions with T0d predicting two: (H4) at D 40.02. Otherwise T0c's reading stands |
| 2 | the onset in D | T2 at D 30 | about 3 h on win | (H4) confirmed or refuted in D |
| 3 | impose the missing condition | (H4): a horizon patch on [−1, −1 + δ] with Taylor unknowns to an order above λ₄, joined to the Chebyshev bulk in `rt_model.py`. (HC): Ω₅⁺ rows. Then #963's random move and #968's chord move on the new root at D 40.02 | design 1–2 weeks (cloud agent), runs of hours | both moves lift the bound: the system has an isolated root, so go to stage 4 on it. Still a set: the RT route stops for P7, and stage 4 uses hp |
| 4 | a certificate plan at one D_c | a scoping note on a radii-polynomial test of stage 3's system with its τ and ξ tails (RT Secs. 1.3 and 5), or on hp's validated mode (SOLVER_SURVEY.md §4 item 1) at an unpinned state | 1–2 days of reading | which formulation to certify, and at which D_c (L4) |
| 5 | lower n₀ for M = 1 | B5(1) on \|z\| ≤ 0.04 with the exact inner flow from lemma 6b₁'s discs (B5_inner.md §5.4) | weeks, cloud agent | the new ln n₀, and whether L3 can close through B(M) |
| 6 | Theorem B beyond M = 1 | B(3) with constants, from the certified A(3) and B8's M-harmonic proof | months, after P6's scoping note | L2 |
| 7 | the identification statements | write L5's criterion (a computed state in the uniqueness class) and L6 (p_n continuous in n) | days, by hand | what stage 4's certificate must also prove |

Stages 0 to 3 run in order, each gated by the last. Stages 5 and 7 are independent of them. Stage 6 waits on P6.

## 6. Proposed text

For the P7 row of [PROOF_QUEUE.md](PROOF_QUEUE.md), Status column:

> scoping note written (2026-11-18, [notes/P7_scope.md](P7_scope.md)). Corollary C needs seven links (§1). A certified
> finite-D solution is necessary but not sufficient, since Theorem B(1)'s class exists only for n ≥ e^{89.75} and for
> the one-harmonic model. The RT system at D 40.02 imposes neither the light-cone condition nor the constraint. The
> scalar's horizon branch at its lowest harmonic, one complex amplitude, is the leading candidate for its two missing
> conditions. Next is stage 0, read-only tests on the stored samples.

For [PROOF.md](../PROOF.md) §6, in place of the second bullet's last sentence:

> A certified solution at one D is necessary but not sufficient. Theorem B's uniqueness class exists only for n ≥ n₀,
> so the identification needs an explicit n₀ ≤ D − 2 for the equations the solvers solve, and a certified solution at
> that D inside the class. For the one-harmonic model n₀ = e^{89.75}. The scoping is
> [notes/P7_scope.md](notes/P7_scope.md).

## 7. Discrepancies (reported, not fixed)

1. RT_FORMULATION.md §13.8, second bullet, and #423's verdict read a D trend into root-level rough bounds, which #963
   showed measure the distance from the singular set (3.5).
2. PROOF.md §6 gives two alternatives for identifying the computed branch, and they are the same condition (section 1).
3. PROOF.md §1 states Corollary C for D = 8.25–67.77. The large-D chain now reaches past D 250 (CORRECTOR_BASIN.md).
4. #163's constraint check (§13.7) compared two roots with no control move, so its finding that Ω₅⁺ does not tell
   them apart is not a test of (HC) (3.4).

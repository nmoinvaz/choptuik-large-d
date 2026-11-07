# Review of the M = 5 second-order certificate (A_M2.md)

An independent, adversarial review of [A_M2.md](A_M2.md) and its new code at commit b0ed77e of
`wip/m5-second-order`. Written by Claude (Anthropic) at Nathan Moinvaziri's request. The reviewer did not write the code
under review and had not seen it before. The review followed AGENTS.md, "How results are checked" and
"Computer-assisted proofs".

**Verdict.** The claim holds as stated, given Braaksma's theorem and the start-value lemma. I found no error in any
enclosure the proof needs: the second variational equation and its remainder, the start data's second derivatives,
the chart term, the mean-value Jacobian and the Krawczyk test are each what the theorem requires (§2). An independent
computation that shares no integration code with the certificate reproduces the second derivatives that set the
margins to 15 digits, deep inside the certified enclosures, and the third-order variation across the box is at most
1.5×10⁻³ of the radius the certificate allows for it (§5). The certified run reproduces bit for bit (§6).

The weaknesses are in the controls. At M = 5 the Krawczyk test passes with a wrong second derivative in three
different ways, and the control D2 catches only one of them: runs with the start values' second derivatives dropped,
or with the chart term dropped, pass both D2 and the Krawczyk test on the certified box (issue 1). No existing
control reaches the remainder terms of the C² step (issue 2). On the clean run these parts are confirmed by reading
and by this review's midpoint comparisons, not by the controls A_M2.md lists. None of the issues can change the
verdict.

Contents

- [1. The claim and the trusted code](#1-the-claim-and-the-trusted-code)
- [2. The mathematics against the code](#2-the-mathematics-against-the-code)
- [3. Issues](#3-issues)
- [4. Making it fail: mutations and what they reach](#4-making-it-fail-mutations-and-what-they-reach)
- [5. The independent computation](#5-the-independent-computation)
- [6. Reruns](#6-reruns)
- [7. What was checked and what was not](#7-what-was-checked-and-what-was-not)
- [8. Reproduce](#8-reproduce)

## 1. The claim and the trusted code

A_M2.md §4 and §6.4 claim, given Braaksma's theorem and the start-value lemma `fixed_point_M.start_data_M`:

- g, the D-jumps at ξ₁ = 0.3 of the M = 5 truncation, has exactly one zero in the curved parallelepiped
  p(X) = {p̂ + Tx + ½(0, Γ[x_o, x_o]) : x ∈ X}, X = [−10⁻¹⁰, 10⁻¹⁰]² × [−10⁻¹⁸, 10⁻¹⁸]² × [−10⁻¹⁹, 10⁻¹⁹]²;
- Dg is non-singular there;
- the zero has A ∈ [2.71857918921174, 2.71857918921175] and ω/ω∞ ∈ [0.999228403206567, 0.999228403206568] (and
  the four inner coordinates of §6.4).

The new trusted code is `certify_M2_lohner.py` (661 lines), `certify_M2.py` (749 lines) and `main2` of
`certify_M_recheck.py`. It rests on the reviewed `certify_M_lohner.py`, `certify_M_sys.py`, `validated_int.py` and
`lemma6_common.py` (A_review.md, A_M_review.md). `certify_M2_fd.py` is not trusted: it supplies the chart's Γ, which
can be anything, and the control D2.

## 2. The mathematics against the code

Each enclosure the proof needs, and where the code computes it. Line numbers are of b0ed77e.

**The C² Lohner set** (certify_M2_lohner.py:254–300). The claims (a)–(c) of the module docstring hold by induction
over the steps, if each step's enclosures are right.

- Y(X) = c + Qq + V₀x + ½W(X)[x, x] is Taylor's theorem with integral remainder, ∫₀¹(1 − s)ds = ½. `YX_box` sums the
  pairs j ≤ k with weight ½ on the diagonal and 1 off it, which is ½Σⱼₖ Wⱼₖxⱼxₖ for symmetric W. Products of the
  balls 0 ± wⱼ are written as products, never as powers (the python-flint pitfall).
- V(X) = V₀ + W(X)[·, x] is the mean value form of the first derivative, and `VX_box` sums over every k with the
  symmetric pair index. Correct.
- V₀ is the derivative along the centre trajectory only, so its step Jacobian J₀ may be enclosed over the centre set
  (:328–331). Its Taylor part uses the centre box X₀ and its remainder the a-priori sets of Y(X) ⊇ X₀. Correct.
- W is updated by the chain rule Wⱼₖ ↦ Dφ(y)Wⱼₖ + D²φ(y)[Vⱼ, Vₖ] at the same x, which is exact (no mean value step).
  The code encloses Dφ by JX over Y(X) (:332–334) and D²φ[Vⱼ, Vₖ] by H over Y(X) and the columns of V(X) (:336–341),
  then puts JX·W + H into Lohner form as validated_int does for V (:359–365). Taking a ∈ V(X)ⱼ and b ∈ V(X)ₖ
  independently only widens H.

**The second variational equation and its remainder** (:62–196, :315, :337–341).

- Ψ′ = AΨ + B[Φa, Φb] with A = ξ′F_y and B = ξ′F_yy is the derivative of D²φ_τ[a, b] = Ψ(τ), with Ψ(0) = 0. There is
  no factor ½ or 2. `secondvar` differentiates SysM.rhs twice by the product rule. The w-terms are present
  (aw·bD + bw·aD in dD and in dS). The self-test compares it with central differences of SysM.lin at random complex
  points, including w-directions, at M = 1, 3, 5 (3×10⁻⁴⁵, rerun). A missing or wrong term would show there.
- The Taylor part Σ_{l≤NJ} Ψ_l h^l uses the trajectory series from Y(X) at t₀ and Φ's series VkX, with v_j = Φa_j
  (pair_sources, :137–159), and the recursion Ψ_{l+1} = (Σᵢ A_{l−i}Ψᵢ + β_l)/(l + 1) (:162–172). The index ranges are
  right: β₀..β_{NJ−1} and A₀..A_{NJ−1} for Ψ₁..Ψ_NJ.
- The remainder h^{NJ+1}Ψ^{(NJ+1)}(s)/(NJ + 1)! uses the chain rule Ψ(s + σ) = Φ(s + σ, s)Ψ(s) +
  D²φ_{s+σ←s}(y(s))[Φ(s)a, Φ(s)b]. Its coefficient NJ + 1 in σ is exactly the recursion started from Ψ₀ = Ψ(s) with
  directions Φ(s)a, so the code runs it at the interval time T = t₀ + [0, h] from (Pt, Vt·V(X), Yt) (:339–341). The
  integral form of the remainder has a non-negative weight of mass one, so a box enclosure of the coefficient is valid
  for complex values.
- `apriori_Psi` (:199–233) checks [0, h](F_y(Yt)Pt + B) ⊂ Pt with `contains`, then returns that image. For a linear
  equation the Picard operator maps functions with values in Pt into themselves, and its iterates converge to the
  solution, so Ψ(τ) ∈ Pt and then Ψ(τ) ∈ image. Containment (not interior) suffices because the equation is linear.
  The inflation in `_infl_col` is only an initial guess.
- The ctx.cap settings of pair_sources do not leak into the next step: taylor_Y resets the cap before any series
  of the next step is formed, and the a-priori functions use no series.

**The start data's second derivatives** (certify_M2.py:114–224).

- D²Y_N[dⱼ, dₖ] by acb_series jets of length 3 of the series recursion over the hull of p(X) and over the directions
  as balls. The jet coefficient c₂ is ½D²[d, d]. The diagonal uses 2c₂ and the mixed pairs use the polarization
  c₂(a + b) − c₂(a) − c₂(b) = D²[a, b] (:196–202). Correct. series2 truncates at the lemma's N (j < N terms in both
  series_state and fixed_point_M.start_data_M), and the recursion of certify_M_sys.series_generic_M is line by line the
  same algebra as fixed_point_M.series_jet_M.
- D²η by Cauchy's estimate. For real x with |xᵢ − p̂ᵢ| ≤ hw and real u, v of max-norm ≤ ν, the points x + z₁u + z₂v with
  |z₁|, |z₂| < (c − hw)/(2ν) have real offsets below hw + (c − hw) = c and imaginary parts below c, so they lie in
  the lemma's complex box p̂ + c(1 + i) (fixed_point_M.py:450). The mixed Cauchy estimate on that bidisc gives
  |∂_{z₁}∂_{z₂}η| ≤ e/r², which is e(2ν/(c − hw))², with no factor 2. The same bound covers u = v.
- eₖ = c·minᵢ rad(∂y/∂pᵢ)ₖ (:172). start_data_M adds a ball of radius eC/c to both the real and imaginary part of every
  derivative (fixed_point_M.py:476–484), with eC the lemma's bound of |ηₖ| on the Cauchy box. So eₖ ≥ eC. The
  direction of the inequality is right.
- The chart term Dy⁺(p(x))·∂²p/∂xⱼ∂xₖ is Dy⁺·(0, Γⱼₖ) for j, k outer and 0 otherwise. `gam` maps (0,0), (0,1), (1,1)
  to Γ₀₀, Γ₀₁, Γ₁₁, which are the second derivatives of ½Γ[x_o, x_o] (:78–109, :209–211). The derivatives come from the
  lemma over the hull of p(X). Correct.
- ω enters only through w = W₀·wf, which is affine in x, so the w-row of W is 0 (start_set2 leaves it 0).

**The mean-value Jacobian and the Krawczyk operator** (certify_M2.py:278–363).

- Dg̃(x) = Dg̃(0) + ∫₀¹D²g̃(sx)[·, x]ds ∈ ℓV₀ + Σₖ ℓW(X)_{·,jk}Xₖ. `Mx` (:348) is I − Y(ℓV₀ + ΣₖℓW_{jk}Xₖ), entry by
  entry, with the symmetric pair index. No factor ½ belongs here, and there is none.
- J(X) is a box that contains Dg̃ on the convex X, so it contains every slope ∫₀¹Dg̃(y + s(x − y))ds. With
  K(X) = −Yg̃(0) + (I − YJ(X))X ⊂ int X, X centred at 0, g̃ has exactly one zero in X and every matrix in J(X) is
  non-singular (Krawczyk 1969; Neumaier 1990, as cited in A_M2.md §2). p(·) is a bijection of ℝ^{M+1} with unit lower-triangular
  derivative, so the statement transfers to p(X).
- Y is the exact midpoint of a 768-bit inverse of mid ℓV₀, and Yℓ is applied to the Lohner sets before any interval
  product (`apply_var`, `apply_set`). This is the Schur correction of A_M.md §2: Y's outer rows remove the leakage
  of the inner conditions, and nothing else is needed for the theorem, which holds for any Y.
- The box. The test's X and the run's box are built from the same decimal strings (`arb(0, arb("1e-10"))`); their
  radii are equal, 1.0000000014×10⁻¹⁰, 1.0000000022×10⁻¹⁸ and 1.0000000016×10⁻¹⁹ (checked). The interior test
  `abs(K) < widths` compares with the lower end of arb("1e-10"). So the zero lies strictly inside the nominal box and
  is unique in the slightly larger run box. The sub-box and θ-scaled tests reuse W(X), which is valid on every sub-box
  centred at 0.
- The quoted zero is the intersection of K(θX) (θ = 1.045×10⁻⁴) and K(X), both of which contain the unique zero.
  `outward` rounds the 60-digit endpoints outward past a 10⁻⁵⁵ relative nudge. The six intervals of A_M2.md §6.4
  match the rerun digit for digit.

## 3. Issues

Ranked by weight. None changes the verdict.

1. **D2 does not fire on two of its own mutations at M = 5** (medium; the controls). A_M2.md §5 expects `--mutate=w0`
   (the start values' second derivatives dropped) and `--mutate=chart` (the chart term dropped) to fail D2, but runs
   them only at M = 1 and M = 3. I ran both at M = 5 on the certified box (h = 10⁻¹⁰, w = 10⁻¹⁸, 10⁻¹⁹, 40 min each).
   Both pass D2, with worst ratios 0.16 and 0.21, and both pass the Krawczyk test, with margins of at least 0.650 and
   0.468. The mutations widen the shared frame QW of the W columns more than they move the midpoints. The inner rows'
   (outer, outer) radii grow by 10⁴ to 10⁵, and the (outer, inner) radii from 8×10⁻³ to 4×10² of the entries, so a
   containment test cannot see them. A midpoint form of D2, |FD − mid| ≤ 3·err + 10⁻¹⁰|FD| with the radius left out
   (`review_M2_d2mid.py`), separates them: 0.010 on the clean run (0.0084 on the clean h = 10⁻¹¹ run), 2.4×10⁴ (w0),
   2.5×10⁴ (chart) and 2.2×10¹¹ (lin2). The review's
   independent check separates them too (§5, §4). Recommendation: add the midpoint criterion to `d2check`, record the
   w0 and chart controls at M = 5, and say in §6.4 that D2 in its present form is not a control of the start data
   or the chart at M = 5.
2. **No control reaches the remainders of the C² step** (medium; the controls). The self-test and D2 compare
   midpoints, and a remainder moves a midpoint by far less than the radii they allow. With both remainders dropped
   (H's Lagrange term and the step Jacobian's remainder), `certify_M2_lohner.py selftest` still prints PASS. This
   review's step-halving check (`review_M2_remainder.py`: one step of size h against eight of size h/8, every
   enclosure must overlap) passes clean at M = 1 and M = 3, and fails by 1.1×10⁸ (M = 1) and 2.4×10⁸ (M = 3) with H's
   remainder dropped, and by 2.5×10³⁴ in V₀ with the Jacobian's remainder dropped. It must start from W = 0. With a
   nonzero start W the missing H remainder hides under JX's remainder (overlap ratio 2×10⁻²). Recommendation: add it
   to the self-test.
3. **Where D2 is sharp on the certified run** (low; the documentation). A_M2.md §6.5 says D2 rules out a wrong
   enclosure "on the runs where it is sharp". On the h = 10⁻¹⁰ run it is sharp in the blocks that set the inner
   margins, the inner rows against pairs with an inner index (radius 4.5×10⁻⁴ to 9×10⁻³ of the entries, deviations
   below 10⁻¹³ of them). It is blind in the (outer, outer) blocks and in the outer rows' (outer, inner) blocks (radius
   2 to 3×10⁸ times the entries), which contribute at most 2×10⁻⁶ to |K_i|/w_i. The inner margins (0.889) would
   absorb inner-inner second-order terms about nine times larger, so the Krawczyk pass is no evidence that W is right.
   §6.4 should say this per block.
4. **float() in the Cauchy radius rounds toward zero** (low; latent, no effect on this run). `start_data2` takes
   hw and ν as `float(abs(...).upper())` (certify_M2.py:160, :174), and float of an arb rounds toward zero (checked:
   float of the upper bound of 1/3 lies below it). Then (c − hw)/(2ν) could exceed the admissible bidisc radius by
   an ulp. In this run both are exact doubles (hw is a mag radius, ν = 1), and the lemma's box is 2.0×10⁻¹² wider
   than the lower end of c, so the bidisc lies in the box (`review_M2_misc.py`, whose mutation fails). Fix: keep the
   arb upper bounds.
5. **The start-data consistency check is overlap, documented as containment** (low; the documentation). A_M2.md §2
   says the run refuses to start unless Y_N(p̂) and DY_N(p̂)T₀ "lie in" the lemma's balls; the code tests `overlaps`
   (certify_M2.py:183–184) and says why. At M = 5 the balls have radius 10⁻⁹⁵, so the check cannot see a difference in
   the series below that. The proof needs series2's Y_N to be the lemma's Y_N exactly. That holds by reading (the
   same recursion, j < N terms in both) and the review's comparison of the two series at p̂ (1×10⁻⁷⁵ apart at 256
   bits), not by this check. Fix the wording.
6. **The recheck's printed intervals are not rounded outward** (low; the documentation). certify_M_recheck.py prints
   `mpmath.nstr(…, 17)`, rounded to nearest, and A_M2.md §6.3–6.4 quote them as intervals. They are comparisons; the
   certified intervals are certify_M2's outward-rounded ones. Mark them as approximate or round them outward.
7. **D2's finite differences share code with the certificate** (low; the documentation). certify_M2_fd.py is
   independent of the C² code, but it shares SysM, certify_M_lohner's Taylor integrator and the start series
   recursion `certify_M_sys.series_generic_M` (which series2 also uses), and d2check converts it to the second-order
   chart with the run's own ℓV₀. Record this in §5, as AGENTS.md asks.
8. **Note**: `export6a` and a resumed run use the box read back from its printed form (radius 1.01×10⁻¹⁰, rounded
   up), slightly larger than the run's box. Every set built from it contains the one built from the true box, so
   both stay valid. No action.
9. **Note**: A_M2.md and A_M.md say "the new code not reviewed"; the coordinator decides how to change that.

## 4. Making it fail: mutations and what they reach

Every mutation below was run by me, except lin2's box run (stored, `data/A_M2_box_M5_1e-11_mut_lin2.json`; its
Krawczyk test and D2 are in the author's logs, my independent comparison against it is new). "Independent" is the
review's comparison of §5, as the largest |here − midpoint| / radius; it fails when a value lies outside.

| mutation | run | Krawczyk test | D2 | midpoint D2 | independent | expected |
|---|---|---|---|---|---|---|
| none | M = 5, h = 10⁻¹⁰ | pass, 0.889 | pass, 2.6×10⁻⁷ | 0.010 | 7×10⁻¹², pass | pass |
| none | M = 5, h = 10⁻¹¹ | pass, 0.989 (author) | pass, 1.0×10⁻⁶ (author) | | 5×10⁻¹¹, pass | pass |
| lin2, a sign in F_yy | M = 5, h = 10⁻¹¹ | **pass**, 0.994 | fail, 1.4×10⁴ | 2.2×10¹¹ | 1.4×10⁴, fail | fail |
| w0, start D² dropped | M = 5, h = 10⁻¹⁰ | **pass**, 0.650 | **pass**, 0.16 | 2.4×10⁴ | 0.15, pass | fail |
| chart, chart term dropped | M = 5, h = 10⁻¹⁰ | **pass**, 0.468 | **pass**, 0.21 | 2.5×10⁴ | 0.21, pass | fail |

| mutation | check | result | expected |
|---|---|---|---|
| H's remainder dropped | step-halving, M = 1 and 3, start W = 0 | overlap fails, 1.1×10⁸ and 2.4×10⁸ | fail |
| the Jacobian's remainder dropped | step-halving, M = 1 | overlap fails in V₀, 2.5×10³⁴ | fail |
| both remainders dropped | `certify_M2_lohner.py selftest` | **PASS** | (fail) |
| none | step-halving, M = 1 and 3 | overlap, 6×10⁻⁴ and 2×10⁻³ | pass |
| lin2 | `certify_M2_lohner.py selftest` | FAIL (W step not contained) | fail |
| C1–C8, C6b | `certify_M2.py controls`, M = 5, h = 10⁻¹⁰ | all as expected | as A_M.md §5 |
| jump sign | `certify_M_recheck.py --mutate` | fail, exit 1 | fail |

So, besides the author's lin2, two more wrong second derivatives pass the Krawczyk test on the certified box, and
those two also pass D2 as it stands. A dropped remainder passes every existing control. What separates all of them
from the clean run is a comparison of midpoints with an independent value, or the step-halving overlap.

## 5. The independent computation

`solver/proof/review_M2_indep.py` recomputes Y D²g̃(0), Y Dg̃(0) and the third-order variation of D²g̃ along a set of
directions, at M = 5, with code written for this review.

- **System.** The M = 5 equations of certify_M_sys.py's docstring (and A_M_review.md §2), coded anew: projected
  products by a plain convolution, the grouping (CE)D kept.
- **Integrator.** A non-rigorous Taylor method of order 16 (Picard iteration on acb_series in the leg parameter, 256
  bits, midpoints only), on the path of A_M.md §1. The steps are chosen once at the centre, from the fast-mode rate
  h|ξ′|/|ξ|³ ≤ 1/2 and the last Taylor coefficients, and reused at every other parameter point. So the numerical flow
  is one analytic function of the parameters, and its derivatives are those of one fixed numerical map.
- **Start values.** The lemma's truncated series Y_N (fixed_point_M.series_jet_M, values only, N = 132). This is
  the lemma's own code, not certify_M_sys.series_generic_M, which the certificate's series2 and the control D2 use.
  The remainder η = y⁺ − Y_N is left out; the lemma bounds it by 2.5×10⁻²⁵ on its Cauchy box, which contains every
  point used here, and it is 8×10⁻⁹⁶ at p̂.
- **Derivatives without differences of derivatives.** The chart and g̃ extend holomorphically to complex x, with
  J_ν(z) = D_ν(p(z)) − conj(D_{−ν}(p(z̄))). Along a real direction u (scaled to the box half-widths), four complex
  points z_k = r iᵏ on a circle (r = 10⁻³) give the Taylor coefficients c_n of z ↦ J(zu) by Cauchy's formula,
  c_n = mean J(z_k)z_k⁻ⁿ, up to the aliased c_{n+4}r⁴. Then Dg̃[u] = c₁, D²g̃[u, u] = 2c₂, and D²g̃(tu)[u, u] =
  2c₂ + 6c₃t + 12c₄t² + …, which gives the variation of the second derivative across the box (t = ±1). No variational
  equation and no finite difference of derivatives is used. The mixed entries come from the directions eⱼ, eₖ and
  eⱼ + eₖ by polarization.
- **The certificate's side.** The final Lohner sets of `data/A_M2_box_M5_1e-10.json`, assembled anew in arb at 768
  bits: the jump rows, Y = the midpoint of (mid ℓV₀)⁻¹, Yℓ(cW + QW qW), Yℓ(cV + QV qV), Yℓ(c + Qq).

**What it shares with the certificate.** The definition of the system, the path, the jump and the chart, and their data
(p̂, C, Γ); the lemma's series code; python-flint; and the saved Lohner sets it compares with. It shares none of
certify_M_sys.SysM, certify_M_lohner, validated_int, certify_M2_lohner, certify_M2 and certify_M2_fd.

**Results** (14 directions: the six coordinate axes and eⱼ + eₖ for (0,1), (0,4), (2,3), (2,4), (2,5), (3,4), (3,5), (4,5),
each scaled to the box half-widths; 4 points each, 614 steps per run, about 40 s per run).

| directions, rows | n | \|here − mid\| / \|value\| | \|here − mid\| / radius | radius / \|value\| | variation / radius |
|---|---|---|---|---|---|
| inner, inner rows | 40 | 6×10⁻¹⁶ – 1.2×10⁻¹¹ | ≤ 7×10⁻¹² | 4.5×10⁻⁴ – 11 | ≤ 1.5×10⁻³ |
| inner, outer rows | 20 | 7×10⁻¹⁶ – 3×10⁻¹³ | ≤ 2×10⁻¹² | 4.6×10⁻³ – 1.6 | ≤ 1.8×10⁻⁴ |
| outer (A, ω), all rows | 18 | 4×10⁻¹³ – 5×10⁻⁸ | ≤ 2×10⁻¹⁴ | 40 – 1.3×10⁷ | ≤ 3×10⁻⁶ |
| outer + inner (e₀ + e₄), all rows | 6 | 3×10⁻¹⁶ – 5×10⁻¹⁵ | ≤ 2×10⁻¹² | 5.5×10⁻⁴ – 2.8×10⁻² | ≤ 1.2×10⁻³ |
| entries by polarization, inner rows, inner pairs | 24 | 5×10⁻¹⁶ – 3.5×10⁻¹⁵ | ≤ 7×10⁻¹² | | |
| entries by polarization, outer rows, inner pairs | 12 | 2×10⁻¹⁵ – 3×10⁻¹⁴ | ≤ 2×10⁻¹² | | |
| entries by polarization, a pair with an outer index | 12 | 2×10⁻¹⁵ – 1.1×10⁻⁷ | ≤ 1.1×10⁻¹¹ | | |

"Here" is Y D²g̃[u, u] from this computation, "mid" and "radius" are those of the certificate's enclosure Y ℓW(X)
along u, and the variation is (|6c₃| + |12c₄|), the change of D²g̃[u, u] from the centre to the box edge along u.
Every value, and both box edges along every u, lie in the enclosure, and Y Dg̃[u] lies in Y ℓV₀u in every row.

By polarization the mixed entries that set the inner margins, (Y D²g̃)_{i,jk} for i, j, k inner, agree with the
enclosure's midpoints to 5×10⁻¹⁶ – 3.5×10⁻¹⁵ relative, at most 7×10⁻¹² of its radius; for example
(Y D²g̃)_{2,45} = 1.095963322×10¹⁸ here and 1.095963322×10¹⁸ ± 4.6×10¹⁴ in the certificate. The inner rows' (outer,
outer) entries, which the chart removes, are between 6×10⁻¹³ and 2.7×10⁻¹¹ here, equal to the certificate's midpoints to
10⁻⁷ relative, against enclosure radii of 10⁻⁵ to 1.3×10⁻⁴. Without the chart term they would be of order Γ, about
0.1 to 1, so the chart term is right. The centre: Y g̃(0) is −1.6×10⁻³³ here (A row) inside the certificate's ±6.1×10⁻²⁴.

So two computations that share no integration code agree to 15 digits on the second derivatives that set the margins,
and the true variation of D²g̃ across the box along these 14 directions is at most 1.5×10⁻³ of the radius the
certificate gives it. Mixed third derivatives across directions not in the list are not measured.

**Accuracy and controls of the check itself.** With order 20 and half the step rate (933 steps instead of 614) the
deviations along e₄ + e₅ are reproduced to three digits, so the integrator's own error is below 10⁻³ of them; the
15-digit agreement is limited by the certificate's midpoints, not by this computation. The raw jump at p̂ changes from
1.6×10⁻²⁵ to 2×10⁻³¹ between the two step sequences, an error that is smooth in the parameters and does not reach the
derivatives. The check can fail: against the stored lin2-mutated run it reports values outside the enclosures by up
to 1.4×10⁴ radii (exit 1), and against the clean h = 10⁻¹¹ run it passes (null control).

## 6. Reruns

From `solver/proof/`, on the Mac, under `nice -n 10`, at most four processes.

| command (A_M2.md §7) | result | as A_M2.md |
|---|---|---|
| `certify_M2_lohner.py selftest` | PASS (secondvar 2.7×10⁻⁴⁵, H series 3.5×10⁻⁴⁹, W step 1.4×10⁻¹⁴) | yes |
| `certify_M2_lohner.py selftest --mutate=lin2` | FAIL (one step's W not contained, mismatch 0.20) | yes |
| `certify_M2.py box 5 … --h=1e-10 --w=1e-18,1e-19` | 2118, 863, 101 steps, 1984 s; final sets bit-identical to `data/A_M2_box_M5_1e-10.json` | yes |
| `certify_M2.py krawczyk --box=data/A_M2_box_M5_1e-10.json --shrink --hier` | pass, margins 0.996, 0.998, 0.889, 0.889, 0.912, 0.911; the six zero intervals of §6.4 digit for digit; hier (a), (b) hold | yes |
| `certify_M2.py controls --box=data/A_M2_box_M5_1e-10.json` | CONTROLS ALL AS EXPECTED, the log identical to the stored one | yes |
| `certify_M_recheck.py data/A_M2_box_M5_1e-10.json [--mutate]` | pass with the same margins [fail, exit 1], the log identical | yes |
| `certify_M2.py d2check --box=<the rerun> --fd=data/A_M2_fd_M5.json` | pass, worst ratio 2.64×10⁻⁷ | yes |

Also checked: the SHA-256 prefix of fixed_point_M.py is 589e5a810bc50289, as recorded; the trusted base files
(validated_int.py, lemma6_common.py, certify_M_lohner.py, certify_M_sys.py, certify_M.py, fixed_point_M.py) are
unchanged on the branch against main; the saved ball strings contain the balls when read back at 256 and 768 bits
(20 000 random balls, no failure; `review_M2_misc.py`, which also checks the box radii and the Cauchy bidisc, and whose
`--mutate` fails).

## 7. What was checked and what was not

Checked:

- the mathematics of §2 against every line of the new trusted code that builds an enclosure (certify_M2_lohner.py
  whole; certify_M2.py: the chart, the start data, the run, `assemble`, `krawczyk2_test`, `shrink2`, `outward` via
  certify_M.describe; certify_M_recheck.main2);
- the reruns of §6, the mutations of §4, the independent computation of §5;
- for the claim's box, that the run box and the test box are equal and that the interior test uses the nominal widths.

Not checked:

- the start-value lemma (fixed_point_M.py) and Braaksma's theorem, which the claim assumes; python-flint;
- the reviewed first-order code beyond its use here (it is unchanged);
- mixed third derivatives of g̃ along directions other than the 14 of §5, so the validity of the radius over the
  whole box rests on reading the code plus those 14 directions;
- the M = 1 and M = 3 runs of A_M2.md §6.1–6.3, the h = 10⁻⁹ run of §6.5 and the lemma-6a export, which I did not
  rerun; the finite differences of certify_M2_fd.py, which I used from `data/A_M2_fd_M5.json` without rerunning;
- the lin2-mutated box run, used from its stored file.

The logs of this review are not in the repository (the task allowed only this note and the review scripts); every
number above comes from the commands of §8.

## 8. Reproduce

From `solver/proof/`, with python-flint 0.9.0. The scratch paths are any directory.

```
python3 review_M2_remainder.py --h=2^-10 --box=1e-30 --wscale=0                  # overlap: ALL (exit 0)
python3 review_M2_remainder.py --h=2^-10 --box=1e-30 --wscale=0 --mutate=psiR    # FAILS, 1.1e8 (exit 1)
python3 review_M2_remainder.py --h=2^-10 --mutate=remJ                           # FAILS in V0, 2.5e34 (exit 1)
python3 review_M2_remainder.py --M=3 --h=2^-11 --box=1e-30 --wscale=0 [--mutate=psiR]   # ALL [FAILS, 2.4e8]
python3 review_M2_indep.py centre --out=S/indep_centre.json                      # 614 steps, 40 s
python3 review_M2_indep.py dir --steps=S/indep_centre.json --u=4,5 --out=S/indep_dir_4_5.json   # about 150 s each;
#   also --u=0, 1, 2, 3, 4, 5, and 0,1  0,4  2,3  2,4  2,5  3,4  3,5
python3 review_M2_indep.py compare --box=data/A_M2_box_M5_1e-10.json --centre=S/indep_centre.json S/indep_dir_*.json
#   COMPARE ALL CONTAINED (exit 0); with --box=data/A_M2_box_M5_1e-11_mut_lin2.json: NOT ALL CONTAINED (exit 1)
python3 review_M2_misc.py data/A_M2_box_M5_1e-10.json [--mutate]                # MISC OK [FAIL]
python3 review_M2_d2mid.py data/A_M2_box_M5_1e-10.json data/A_M2_fd_M5.json        # midpoint ratio 0.010 (exit 0)
python3 review_M2_d2mid.py data/A_M2_box_M5_1e-11_mut_lin2.json data/A_M2_fd_M5.json   # 2.2e11 (exit 1)
python3 certify_M2.py box 5 --chart=data/A_M2_chart_M5.json --h=1e-10 --w=1e-18,1e-19 --mutate=w0 --out=S/w0.json --log=S/w0.log
python3 certify_M2.py krawczyk --box=S/w0.json                                   # passes (0.650): issue 1
python3 certify_M2.py d2check --box=S/w0.json --fd=data/A_M2_fd_M5.json          # passes (0.16): issue 1
python3 review_M2_d2mid.py S/w0.json data/A_M2_fd_M5.json                          # 2.4e4 (exit 1)
#   the same with --mutate=chart: krawczyk passes (0.468), d2check passes (0.21), review_M2_d2mid 2.5e4 (exit 1)
```

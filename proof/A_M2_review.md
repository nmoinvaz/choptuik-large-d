# Review of the M = 5 second-order certificate (A_M2.md)

An independent, adversarial review of [A_M2.md](A_M2.md) and its new code at commit b0ed77e of
`wip/m5-second-order`. Written by Claude (Anthropic) at Nathan Moinvaziri's request. The reviewer did not write the code
under review and had not seen it before. The review followed AGENTS.md, "How results are checked" and
"Computer-assisted proofs".

**Verdict.** VERDICT_PLACEHOLDER

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

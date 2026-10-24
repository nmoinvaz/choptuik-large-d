# P2: independent review of Theorem B(1)

Queue item P2 of [notes/PROOF_QUEUE.md](../notes/PROOF_QUEUE.md). The reviewer is a fresh agent that had not seen how
P1 reached its results: it worked from the statements and proofs in [B_chain.md](B_chain.md) (§§1–3, 6–8, 13–15),
[C_selection.md](C_selection.md) §13.1, [B8_constants.md](B8_constants.md), [B5_inner.md](B5_inner.md) and the
certificate code. It did not read P1's transcripts or worktree. Written by Claude (Anthropic) under Nathan Moinvaziri's
direction; not checked by a mathematician. Status words as in [AGENTS.md](../AGENTS.md). Review code and logs are in
`solver/proof/review_p2/`.

Contents

- [0. Verdict](#0-verdict)
- [1. Method and what is shared](#1-method-and-what-is-shared)
- [2. Item 1: the hand arguments](#2-item-1-the-hand-arguments)
- [3. Item 2: second implementations](#3-item-2-second-implementations)
- [4. Item 3: audits of the remaining new certificates](#4-item-3-audits-of-the-remaining-new-certificates)
- [5. Item 4: the older code](#5-item-4-the-older-code)
- [6. Findings and proposed fixes](#6-findings-and-proposed-fixes)
- [7. Reproduce](#7-reproduce)

## 0. Verdict

**Theorem B(1) stands as stated** (B_chain.md §15: for every n ≥ e^{89.75} exactly one p_n in Theorem A's box gives a
light-cone-gauge, centre-regular solution analytic at x = 1, and |p_n − p*|_∞ ≤ e^{67.84}/n; Braaksma's theorem
assumed), **with corrections that change no quoted constant or threshold** (§6). No error that breaks the theorem was
found. Two items were not fully re-implemented: the bound algebra of Lemma LC's closure and Lemma J's secant
coefficients (§3.2, §3.3); their inputs, identities, final contraction checks and every hand argument were checked.

| item | what | verdict |
|---|---|---|
| 1a | Lemma LC: fixed-point formulation, bounded forms at x = 1, kernel on L | confirmed (identities re-derived and checked independently) |
| 1a | Lemma LC: majorant derivative bounds the Lipschitz constant | confirmed with corrections F1–F3 (the class 𝒦_h must say more; one bound uses the c-equation); no change in θ, h or d₀ (patched run) |
| 1a | Lemma LC: double-sector symmetry | confirmed |
| 1b | Lemma J: weighted-norm contraction, endpoint, a priori argument | confirmed (every Γ formula re-derived) |
| 1c | Lemma P | confirmed |
| 1d | §15 logic: (E), (H-J), B5(2), B7 | confirmed; the uniqueness class should be named in the statement (F4) |
| 1e | lemma L, B5(2), Lemma D′, (E) | confirmed with corrections F5, F6 (exponent directions; effect ≤ 2×10⁻⁴ on ln C_E) |
| 2 | B7 second implementation (`p2_b7.py`, mpmath intervals) | **confirmed**: n₀ = e^{89.75} (passes from e^{89.65}), C₇ = e^{67.838} |
| 2 | Lemma LC: second implementation | **partial**: formal series and atoms, Lemma R and the L forms, Perron θ re-implemented and confirmed (θ ≤ 0.7053, d₀'s ρ₄ part, c₅(0)); the closure's bound algebra not re-implemented |
| 2 | Lemma J: second implementation | **partial**: Perron θ = 0.1759 and every Γ formula confirmed; the secant coefficients ℓ (DQ) not re-implemented |
| 3 | `b5_label.py`, `b5_jump.py`, `b7_adjoint.py`, `b8m_bounds.py --d0` | audited: confirmed with F5, F6 |
| 4 | B5(1), B8, (H8) older code | B8's formal series and atoms reproduced independently; the rest not checked |

## 1. Method and what is shared

- **Hand arguments**: every step of B_chain.md §§2, 3, 6, 8, 13–15 re-derived on paper; the code that implements each
  step read against the derivation.
- **Second implementations** use mpmath (interval arithmetic `mpmath.iv`, exact rationals, 60–80-digit complex
  arithmetic) instead of python-flint, and import nothing from `solver/proof/` except `p2_lc_fix.py`, a labelled
  copy of `b8m_lc.py` used only to measure the proposed fixes (not an independent check).
- **Shared with the reviewed code** (by design, recorded as AGENTS.md asks):
  - `p2_b7.py` reads Theorem A's J_D(x̂) and J(X) (`lemma6_path_*.json`; Theorem A was reviewed in A_review.md),
    sup‖ℓ̃‖₁ (`b7_ift.json`), lemma L's ε_L, ‖V‖, ‖V⁻¹‖, Re λ (`b5_label.json`) and B5(1)'s super-solution x_Φ
    (`b5_jump_d0_1e7.json`). It recomputes C_E, the Cauchy bounds, the Krawczyk test, min_{∂X}|J_D|, ‖J(X)⁻¹‖, the
    degree condition and C₇.
  - `p2_perron.py` reads the saved matrices (`b8m_lc_lip.json`, `b8m_lcj.json`).
  - `p2_formal.py` and `p2_identities.py` share only the statements in the notes (the model of B8_constants.md §1.1,
    the identities of §4, the forms of B_chain.md §13.3) and the box centre (A*, ω*/ω∞) = (2.5421480475, 0.7769940136).
  - `p2_formal.py` compares with the printed `b8m_atoms_K4.log`.

## 2. Item 1: the hand arguments

### 2.1 Lemma LC (B_chain.md §13)

**Fixed-point formulation (§13.2): confirmed.** A fixed point satisfies (I2)–(I4) with zero defects (Lemma R holds for
any Z, and T_S, T_β, T_c, T_V are the variation-of-constants forms of those equations); T_c(U)(1) = d + ∫₀¹ = 0, so
c(1) = c^K(1) = 1; R_V(0) = [d 2F], R_S(0) = R̂_β(0) = 0. Conversely a solution in 𝒦_h with c(1) = 1 is a fixed point
(E(x, s) → 0 as s → 0 along P_x, and (s/x)^{n−1} → 0), which gives uniqueness. On L the integrand Λ(G₁ + G₂)/s equals
B8's ΛG₁/s + W₂ (xW₂ = ΛG₂), so T_S is one analytic path integral.

**Bounded forms near x = 1 (§13.3): confirmed**, and checked independently (`p2_identities.py`, T2–T3, 80 digits, at
random complex points with the truncated series of `p2_formal.py`): xW₂ = ΛG₂ with
G₂ = (A^K)⁻¹D^KD⁻¹(CM_hS + x²M_hC⁻¹Y) (residual 2×10⁻⁸¹) and q(ΛG₁ + xW₂) = ℓx⁻²(I + e)⁻¹ε₁A^K(G₁ + G₂)
(1.9×10⁻⁸¹). The bound |δD/(x − 1)| ≤ (2|c^K| + |R_c|) sup_{[1,x]}|R_c′| needs R_c(1) = 0 (see F1), and
(δD/(x − 1))′ = ∫₀¹ t δD″(1 + t(x − 1)) dt gives the ½ sup|δD″| used in `build_L`. The lower bound
‖D̃⁻¹‖⁻¹ ≥ |x + 1|(1 − ‖Γ‖) − ‖δD/(x − 1)‖ is the standard perturbation bound. Typo: §13.3 writes
u′ = (Λ/x)u − G′; it is u′ = (Λ/x)u + G′, which is what the formula for R_S below it uses.

**Kernel on L: confirmed.** Along s = 1 + re^{iφ}, d/dr ln|1 − s⁻²| ≥ 1/r − 1/|s + 1| − 2/|s| ≥ 1/r − c₁, and
(1/r − c₁) ≥ (1/r)(1 − c₁δ/2) for r ≤ δ/2, which gives the factor (n − 1)(1 − c₁δ/2)/2; |λ|‖H′‖ =
‖(I − Γ)⁻¹η‖/(|s||s + 1|r) ≤ η_L/r. So |E(x, s)| ≤ (|x − 1|/|s − 1|)^{N_L} with N_L as stated, and
∫|E||G′| ≤ sup|G′|·|x − 1|/(N_L − 1). The path P_{x′} to the radial projection x′ on Ac stays in 𝒟_n (at radius
1 ± δ/2 the arc down to arg x′ is outside the open disc), and B8's arc gain for targets at radius in [1 − δ/2, 1 + δ/2]
uses the right distance |s² − 1| ≥ (δ/2)(2|s|² + 2 − δ²/4)^{1/2} (the minimum of |s² − 1| on that arc is at the disc).

**Majorant derivative bounds the Lipschitz constant: confirmed for the operations, with three corrections to what the
class contains.** The argument (difference of a node ≤ directional derivative of its majorant at h; each candidate a sum
of term-wise suprema of non-negative-coefficient series in t, so convex, so slope ≤ secant) is correct for every
operation in `b8m_lc.evaluate`:
- sums, products (|fg − f̃g̃| ≤ |f||g − g̃| + |g̃||f − f̃|), inverses through 1/(lower bound − s(t)) (`c.lo`, `D.lo`,
  `lDt`), suprema over regions (linear in the coefficients), maxima over candidates (taken per candidate);
- the choices that look like minima (`arcp`, `lnK` in the kernel, the two options of `term_sup` on F) are fixed before
  h enters, so they do not break convexity;
- `SUMHYP` (Ra = N0 + Na in the Lipschitz runs) and `CONTAINS` correctly turn a perturbation of one band into a
  perturbation of the path regions that contain its points; holding D₀ fixed in the value columns and perturbing it in
  the derivative columns matches the norm's two kinds of components;
- ball mode keeps both secant runs exact (no outward rounding between them), so the upper end of the secant ball bounds
  the exact secant of the exact majorant, which is a valid majorant.

But the invariance and Lipschitz estimates are for a general U ∈ 𝒦_h, not only for solutions, and the class as defined
in §13.1 (value bounds per band and the first-derivative bounds D₀(h)) does not contain everything the code uses:
- F1: R_c(1) = 0 is used (δD/(x − 1) bounded on L) but is not part of 𝒦_h. Without it T is not even defined.
- F2: on L, G′ (needed for the integration by parts and for |xR_S′|) contains (δD/(x − 1))′, hence R_c″, which no
  class condition controls; the code substitutes the derivative of the c-equation's right-hand side, valid only when
  xR_c′ = F_c(U).
- F3: `build` bounds |xc′| by |c||β|, which is the c-equation xc′ = [cβ] (true for the solution, not for a class
  member); it enters the derivative bounds of C⁻¹ and D⁻¹.

All three are repairable without changing the numbers: F1 and F2 by adding the conditions to 𝒦_h (T preserves both:
T_c(U)(1) = 0, and x(xT_c(U)′)′ = xF_c(U)′ is bounded by the code's own Fc.d), F3 by using the class bound
|xc′| ≤ |xc^K′| + |xR_c′|. The patched copy `p2_lc_fix.py` measures it (§3.2): with F3 fixed the certificate closes
with the same ratios (≤ 0.990), the same d₀ = 3.586×10⁶ and θ = 0.7355 (Perron root 0.7051); with the R_c″ bound on L
multiplied by 10²⁰ nothing changes except h_Sp (5.97×10⁻⁷ → 7.02×10⁻⁷), because R_c″ enters only through factors
1/(N_L − 1) ≈ 2×10⁻³² and n₀⁻¹, and the derivative component of S on L feeds nothing back (its column of the Lipschitz
matrix is ≤ 10⁻⁴¹). So the extra norm component that F2 requires leaves θ unchanged.

**Double sector: confirmed.** On Σ_n ∪ Σ̄_n slit along (1, R′_n], T on the lower half uses only lower-half values
(and d, which is the integral over the common segment [0, 1]); the integrand is analytic off the slit, and for
x ∈ (0, 1) the upper and lower paths bound a region inside the domain, so T(U) is analytic across (0, 1). The
lower-half estimates are the upper ones for p̄ (the bounds are on moduli, the kernel λ = (n − 1)/(x(x² − 1)) has
real coefficients, and the box is conjugation-invariant). So T is a contraction with the same θ, its fixed point
restricts to each half's unique fixed point, and the two halves share c(0). For real p, σZ(x̄) is also a fixed point,
so Z is σ-symmetric. This is what makes Y⁺(1) = Y⁻(1) and Y⁺′(1) = Y⁻′(1), which (E) and Lemma J use.

**Consequences: confirmed.** |d|₁ ≤ 3.59×10⁶n⁻⁵ ≤ 10⁷n⁻⁵, so on 𝒟′_n the solution is B8's with normalization d(p)
(uniqueness of the centre-regular solution for given c(0)), and (H8), B5(1), B5(2) with the data of d₀ = 10⁷ apply.
`b8m_bounds.py --d0` adds exactly |[d 2F]|₁ ≤ 4 max|F_μ| d₀ to P_V and d₀ to P_c (§4), and the start of Lemma B8.b
still holds strictly because h_c > d₀; `b8m_h8.py --d0` passes the offset to `b8m_bounds` through the shared argv.

### 2.2 Lemma J (B_chain.md §14): confirmed

- *Equations and label.* With P = C⁻¹Y (B5's P = Δ/x) and Φ = xP′ = C⁻¹(xY′ − [xc′P]) (checked, T4), Φ⁺ = Φ⁻ at x*
  gives Λ(v_S + δG)(x*) = −Ωv_S + δCΦ⁻ + δ[xc′P] exactly, since δ(xV′) = Ωv_S and δ(xS′) = Λ(v_S + δG).
- *Γ.* Each entry re-derived: backward, ‖E(x, y)‖ ≤ exp(−(n − 1)(1 − h_H)(Φ(y) − Φ(x))) by the log-norm
  (μ(−λ(I + H′)) ≤ −λ + λ‖H′‖, λ > 0 on x > 1); with |Λ/y| ≤ (n − 1)Φ′(1 + h_H) and m(y)/m(x) = e^{(n−1)(Φ(y)−Φ(x))/2}
  the integral is ≤ (1 + h_H)/(1/2 − h_H); the endpoint term is ≤ e_j because 1/2 − h_H > 0. Forward,
  m(y)/y = m′(y)·2(y² − 1)/(n − 1) gives ∫₁^x m dy/y ≤ m(x)·2(x² − 1)/(n − 1), and (x² − 1)/ρ(x) ≤ n, so
  Γ_{V,S} = 2|ω|n/(n − 1) and Γ_{c,j} = 2n/(n − 1) sup(ℓ_{F_c,j}s_j); for β, ∫₁^x(y/x)^{n−1}dy/y ≤ 1/(n − 1) and the
  shapes are nondecreasing, so Γ_{β,j} = sup(ℓ_{ĝ,j}s_j/s_β)/(n − 1).
- *A priori argument.* On (1, 1 + δ/2] Ψ_out is bounded and v(1) = 0 for every slow component and the b-mode (Y⁺ and Y⁻
  agree at x = 1 by the double sector), so Ψ_out is the fixed point of the affine map with its actual endpoint value.
  The inhomogeneous term E(x, x₁)v_S(x₁) has finite weighted norm (its ratio to m tends to 0 as x → 1⁺ because
  1/2 − h_H > 0); the linear part contracts in the unweighted norm (θ₀) and in the weighted one (its Γ is dominated
  entrywise by the full Γ, which takes suprema over a set of bands containing L). So the Neumann series converges in
  both, and N(Ψ_out) < ∞. The final step can use the nonlinear differences directly: N(Ψ_out) ≤ θN(Ψ_out) in the
  Perron-weighted norm, so Ψ_out = 0. (For the Neumann series as a linear operator, |𝒢(x)w| ≤ Σℓ_j|w_j| holds for
  every w because the forward-differentiation bound is a bound on DG at every class point of the segment.)
- Minor: the docstring of `b8m_lcj.py` says s_b = 1/n; the code and the notes use s_β = ρ/n + ρ².

### 2.3 Lemma P (B_chain.md §8.2): confirmed

The Lohner step encloses the flow derivative over the box hull X_k of the current set (Taylor part over X_k, remainder
from the a priori enclosure over X_k), which is convex and contains y^±(ξ_k; p) and the segment between them, with the
parameters carried as constant states, so B̄_k ∈ [J_k] (dynamic block). y⁻(p) = σ(y⁺(p̄)) solves the same equation with
the same p (σ symmetry on the real axis), and the start hull covers both because X_ℂ is conjugation-invariant.
`acb_mat.inv` of a ball matrix encloses the inverse of every member, and the QR-Lohner product with the interval matrix
[J_k]^{−T} contains χ_{k+1}(p) = B̄_k^{−T}χ_k(p) for each p. The conversion to lemma L's variables,
ψ_S = u*𝒞⁻¹(ψ_Φ + Dψ_P − [ψ_C Q⁻]) with D = diag(1 + iμω), is the difference of 𝒞Q = Φ + DP (B5_inner.md Lemma 1 at
ε = 0); 𝒞 is enclosed over the inner enclosure at u*, which contains C⁺. One exponent in the remainder is in the wrong
direction (F6).

### 2.4 The logic of §15: confirmed

- (E) (§6): (⇐) glue Y⁺ on H⁺ and Y⁻ on H⁻ (continuous up to both real segments and equal there), removable singularity
  by boundedness; (⇒) identity theorem on H^± ∩ {|x − 1| < r}. Both need only (H-LC), which Lemma LC gives.
- (H-J) is Lemma J; with (E), "analytic at x = 1" ⇔ Ψ_out ≡ 0 ⇔ L_{u*}(Ψ_out) = 0.
- B5(2) is applied to the light-cone-gauge solution, whose normalization d(p) is holomorphic with |d|₁ ≤ 3.59×10⁶n⁻⁵;
  Ẽ(p) = Ẽ(p, d(p)) is holomorphic on 𝒦 and bounded by C_E/n pointwise, which is all the Cauchy estimates need.
- B7: selection zeros ⊂ zeros of F = J_D + Q; F has exactly one zero in X (Krawczyk), so at most one selection zero;
  existence by the homotopy g_t (deg g₀ = ±1 because k₊₁ has the single zero p* in X and DJ_D(p*) = ℓ̃DS̃(p*) is
  invertible, so Dk₊₁(p*) is; |g_t| ≥ (1 − ‖R‖)|k₊₁| − ‖V‖|Ẽ| on ∂X, using |k|_∞ = |k₊₁| for real p); for real p,
  [L]₋₁ = −conj([L]₊₁), so g₁ = 0 gives L = 0. Location: J_D(p_n) = −Q(p_n) and the mean value form with J(X).
  Monotonicity in n: Y in Krawczyk is the midpoint inverse of J(X), independent of n, so the test is inclusion-monotone
  in the widths |Q|, |DQ|; the degree condition is monotone in C_E/n.
- **"Exactly one p" follows**, among the solutions of Lemma LC. The uniqueness class 𝒦_h is named in the statement only
  as "(c(1) = 1, the solution of Lemma LC)" and in the closing remark. A light-cone-gauge solution outside 𝒦_h (far
  from the formal series) is not excluded, for any p. F4 proposes naming the class in the statement.

### 2.5 Lemma L, B5(2), Lemma D′: confirmed with corrections

- *Lemma L.* The Volterra/Grönwall argument, the identity for (J_F − 𝒢)/z (re-checked by the authors to 10⁻⁴⁹; the
  derivation from 𝒞 − I = z[Ct·] and 𝒢 = [Ct(0)·] − (D₂ + D)/2 is straightforward), and the identification with the flat
  estimate of lemma 6a are correct. In the C-term of g the exponents do not cancel: |z^{−λ_i}|·|ψ_C| with
  |ψ_C| ≤ 2e^{κ_C z}κ_P A z^{μ₋+3}e^{−1/(2z)} (the maximum over j of z^{Re λ_j + 1} for z < 1 is at the smallest
  Re λ, μ₋, not μ₊ as written) gives z^{2−δ}, not z². F5; the effect on ∫₀^{Z_L}g is below 10⁻⁸.
- *B5(2).* |(u*^{−Λ})_{ii}| = u*^{−Re λ_i} ≤ u*^{−μ₊} (u* < 1, so the largest Re λ gives the largest factor), not
  u*^{−μ₋}. F6: ln C_E increases by (μ₊ − μ₋)ln(1/u*) = 1.96×10⁻⁴ (76.93966 → 76.93986). The superposition of
  B5(1)'s super-solutions is valid: the super-solution inequality b + Lx < x is linear with L ≥ 0 fixed, and the data
  satisfy δ ≤ δ_ref, as `b5_jump.py` checks.
- *Lemma D′.* In `b8m_bounds.py` the outer radius enters only through ρ_max (the far region's box, N_F and the ray
  part of γ_F); every other bound is a term c X^aθ^bn^e evaluated at n₀ on boxes that do not move with n. Confirmed.

## 3. Item 2: second implementations

### 3.1 B7 (`p2_b7.py`): confirmed

Written from §8.3's statement in mpmath interval arithmetic (every decision on exact interval endpoints). It
recomputes C_E from B5(1)'s x_Φ (with the corrected exponent u*^{−μ₊}), the Cauchy bounds (r_P = 10⁻⁸ for X_ℂ,
r_C = (1 − 10⁻⁹)·10⁻⁶ for 𝒦), ‖J(X)⁻¹‖_∞ = 44.38 from an interval inverse, min_{∂X}|J_D| ≥ 2.693×10⁻¹¹ on 400 segments
per side (2.626×10⁻¹¹ with b7_ift's 40), the Krawczyk test with a fixed midpoint preconditioner and the degree
condition, on a 0.05 grid in ln n ∈ [74, 110].

| | b7_ift.py | p2_b7.py |
|---|---|---|
| C_E | e^{76.9397} (u*^{−μ₋}) | e^{76.9399} (u*^{−μ₊}) |
| first ln n where both tests pass | 89.75 (0.25 grid) | 89.65 (0.05 grid; 89.65 also with u*^{−μ₋} and 40 segments) |
| every later grid point passes | yes | yes |
| C₇ | 2.90×10²⁹ = e^{67.84} | 2.895×10²⁹ = e^{67.838} |

Mutations (`p2_b7.log`): J_D(x̂) + 5×10⁻¹¹(1 + i) and J_D without the conjugation of D₋₁ make every n fail; C_E·e⁵
moves the threshold to e^{94.65} (by 5.00).

### 3.2 Lemma LC: partial

Re-implemented independently and confirmed:
- *Formal series and atoms* (`p2_formal.py`): the 1/n series of the M = 1 model derived from the original p, m, b, c
  equations (not B8's reduced system), to order 8. The 52 descriptor sums printed in `b8m_atoms_K4.log` (V, S, β, c − 1,
  Y, Δ, Γ, det − 1, det·η̃, τ_D, τ_v, τ_b, ρ₄) agree to their printed precision (worst 3.9×10⁻⁴); the residual orders
  are (K, K + 1, K, K + 1). |c₅(0)|₁ = 1.2269×10⁵, |c₆(0)|₁ = 1.5164×10⁶, |c₇(0)|₁ = 1.9509×10⁷ (§4's formal offset);
  the exact ρ₄ part of d₀ at the box centre is 1.9002×10⁵ (1.9011×10⁵ over the box in `b8m_lc_K4.log`). Mutation: the
  wrong sign of the Ω term changes most descriptors by 60–96 %.
- *Identities* (`p2_identities.py`): T1, the reduction from the original equations to B8's reduced system (V, D, A, B,
  Δ = xC⁻¹Y), residual 3.5×10⁻⁸⁰; T2, Lemma R (I1)–(I4) with defect terms and both forms of q, ≤ 9.4×10⁻⁷⁷; T3, the L
  forms; T4, B5's Φ. Mutations (ρ₂ without its first term, q without (I + e)⁻¹, g₃ without the ½) are caught.
- *Contraction* (`p2_perron.py`, exact rationals): the saved 40 × 40 matrix has a Perron weight with θ = 0.70521
  (the certificate's θ = 0.7355 used λ = 0.755). Scaling the matrix by 1/0.7 makes it fail (caught).
- *Constants of the L region*, recomputed by hand from the formulas: c₁ = 2.5576, N_L = (n₀ − 1)(1 − c₁δ/2)/2 − η_L
  = 5.758×10³¹, and η_L = 6.17 is consistent with ‖det·η̃‖'s leading descriptor 11.30 (11.30·1.025²/((1 − ‖Γ‖)·0.975·1.975)
  = 6.17).

Not re-implemented: the bound algebra of the closure P̂(h) < h and of the 40 × 40 Lipschitz matrix (`b8_alg.py`,
`b8m_bounds.known`, `b8m_lc.evaluate`, the kernel constants). They were read line by line against Lemma R and §13
(§2.1), and the patched copy measures the fixes:

| run | closes, max P/h | d₀ | θ (Perron root) | h_{S,F} | h_Sp |
|---|---|---|---|---|---|
| `b8m_lc_K4.log` (certified) | yes, 0.990 | 3.586×10⁶ | 0.7355 (0.7052) | 4.121×10⁷ | 5.97×10⁻⁷ |
| F3 fixed (`p2_lc_fix.log`) | yes, 0.990 | 3.586×10⁶ | 0.7355 (0.7051) | 4.118×10⁷ | 5.97×10⁻⁷ |
| F3 fixed, R_c″ bound × 10²⁰ (`p2_lc_fix_rcpp.log`) | yes, 0.990 | 3.586×10⁶ | 0.7355 (0.7051) | 4.118×10⁷ | 7.02×10⁻⁷ |

### 3.3 Lemma J: partial

Confirmed: θ = 0.17593 from the saved Γ in exact arithmetic (and θ₀ ≤ 1.8×10⁻⁵ for Γ₀); every Γ formula (§2.2). The
contraction is robust: the only loops are S → V → S (5.94 × 1.40×10⁻³), S → c → S (0.109 × 0.024), the S self-coupling
(0.046) and the loops through β (≤ 5×10⁻³); the large β-row entries (Γ_{β,c} = 34) feed back only through
Γ_{S,β} = 2.2×10⁻⁵ and Γ_{c,β} = 1.5×10⁻⁴. Not re-implemented: the forward-differentiation coefficients ℓ (`DQ`); they
use the actual solutions R^± (so the c-equation is legitimately used there, unlike F3), and since the corrected LC
constants are componentwise no larger, `b8m_lcj.py`'s inputs remain valid.

## 4. Item 3: audits of the remaining new certificates

- **`b5_label.py`** (lemma L): rounding outward throughout (python-flint balls, upper ends); the cells cover (0, Z_L]
  with t^{−δ} ≤ t₁^{−δ} on [t₁, t₂] and the exact first cell; the coefficient sups are over the cells and 𝒦; the
  eigen-decomposition is closed form with the branch of the square root chosen away from the cut (ball arithmetic
  would fail visibly otherwise). Finding F5 (C-term exponent). Constants from certified sources only (lemma S over 𝒦).
- **`b5_jump.py`** (B5(2)): inputs from the B5(1) certificate JSON and the (H8) log, inflated by 10⁻¹² relative for
  the float round trip. Finding F6. Confirmed otherwise.
- **`b7_adjoint.py`** (Lemma P): §2.3. The adjoint block is the only change to validated_int's step; the double-precision
  check compares χ at the centre with a forward variational integration (relative 10⁻⁸) and its mutation (transpose)
  is caught. A second, rigorous implementation (a forward Lohner run or CAPD) was not done.
- **`b8m_bounds.py --d0`**: the two added terms are exactly those of B8_constants.md §8.4 (§2.1 above).
- **`b7_ift.py`**: re-implemented (§3.1). In `ell_bound` the bound on |ψ_C| uses u*^{μ₊+3}; it should be u*^{μ₋+3}
  (F6; effect ≤ 2×10⁻⁴ relative on a remainder that is 5 % of ‖ℓ̃‖).

## 5. Item 4: the older code

`b8m_formal.py` and `b8m_atoms.py` are reproduced by `p2_formal.py` (§3.2). Not checked: `b5_bounds.py`,
`b5_formulas.py` (B5(1)), `b8m_bounds.py` beyond `--d0` and Lemma D′, `b8m_h8.py`, lemma 6b₁'s disc covering, and the
M = 3 items of the queue.

## 6. Findings and proposed fixes

None changes a quoted constant or threshold; none breaks Theorem B(1).

1. **F1 (Lemma LC, §13.1): R_c(1) = 0 is not in the definition of 𝒦_h.** The bounded forms on L need it, and T
   preserves it. *Fix:* add "R_c(1) = 0" to 𝒦_h (a closed condition, so 𝒦_h stays complete).
2. **F2 (Lemma LC, §§13.1, 13.4): R_c″ on L is used but not controlled.** G′ on L contains (δD/(x − 1))′, hence R_c″;
   `build_L` bounds it by (|xF_c′| + |F_c|)/(1 − δ/2)², valid only when xR_c′ = F_c(U). *Fix:* add to 𝒦_h the bound
   |x(xR_c′)′| ≤ (that bound) on L, which T preserves (x(xT_c(U)′)′ = xF_c(U)′), and add it as a component of the norm
   N. It enters only the derivative bound of S on L (through 1/(N_L − 1) ≈ 2×10⁻³²) and the β bound on L (through
   (n − 1)⁻¹ ≤ n₀⁻¹), and the first of these feeds nothing back; `p2_lc_fix.py --rcpp=1e20` shows the closure and θ
   unchanged even with that bound multiplied by 10²⁰.
3. **F3 (`b8m_lc.py` `build`): |xc′| ≤ |c||β| uses the c-equation**, which a class member need not satisfy. *Fix:*
   `c.d = cK.d + Rc.d` (the class bound) and |xD′| ≤ 2|x|² + 2|c||xc′| with it (pass 2). Measured: closes with the same
   ratios, θ = 0.7355, d₀ unchanged, h_{S,F} 0.08 % smaller (`p2_lc_fix.log`). The same line in `b8m_bounds.py` is
   correct there, because B8's continuation argument is about the actual solution.
4. **F4 (§15 and C_selection.md §13.1): name the uniqueness class in the statement.** Proposed wording: "...exactly
   one p_n in X for which the light-cone-gauge solution of Lemma LC (the unique centre-regular solution with
   c(1) = 1 whose remainder lies in 𝒦_h, a neighbourhood of the formal series) is analytic at x = 1."
5. **F5 (lemma L, §2.3; `b5_label.py`): the C-term of g is 2γ_Cκ_Pe^{κ_C t}t^{2−δ}, not t².** The bound on ψ_C has
   z^{μ₋+3} (smallest Re λ), not z^{μ₊+3}. Effect: ∫₀^{Z_L}g grows by less than 10⁻⁸ (the C-term contributes 4×10⁻⁵
   in all, times Z_L^{−δ} − 1 ≤ 2×10⁻⁴); ε_L ≤ 0.3527 unchanged. *Fix:* `cubic` → ∫t^{2−δ} (or multiply by
   t₁^{−δ}), and μ₊ → μ₋ in the notes' bound on ψ_C.
6. **F6 (B5(2), §3.2; `b5_jump.py`; `b7_ift.ell_bound`): exponent directions.** In C_E, u*^{−Re λ_i} ≤ u*^{−μ₊}
   (code and notes have μ₋): ln C_E = 76.93986, not 76.93966; n₀ = e^{89.75} and C₇ = e^{67.84} unchanged (§3.1, which
   uses the corrected value). In `ell_bound`, `psC` should use μ₋ (`mu_lo`), not μ₊. *Fix:* swap `mu_lo` ↔ `mu_hi` in
   those two places and in the text of §3.2.
7. **F7 (typos).** §13.3: u′ = (Λ/x)u + G′. `b8m_lcj.py` docstring: s_b = ρ/n + ρ².

## 7. Reproduce

In `solver/proof/review_p2/` (each takes seconds, except the patched LC runs: about 6 minutes each):

```
python3 p2_b7.py                     # B7: n0, C7 (p2_b7.log, with the mutations below)
python3 p2_b7.py --mutate=JD         # caught (also --mutate=conj, --mutate=CE5)
python3 p2_perron.py                 # LC theta, J theta, theta0 in exact arithmetic (p2_perron.log)
python3 p2_perron.py --mutate=scale  # caught
python3 p2_formal.py                 # formal series, atoms, c_k(0), rho4 part of d0 (p2_formal.log)
python3 p2_formal.py --mutate=sign   # caught
python3 p2_identities.py             # reduction, Lemma R, L forms, Phi (p2_identities.log)
python3 p2_identities.py --mutate=rho2   # caught (also q, g3)
./run_lc_fix.sh                      # patched LC (F3), then with the R_c'' bound x 1e20 (p2_lc_fix*.log/.out)
```

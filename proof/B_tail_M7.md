# Gap (b) continued: the M = 7 zero and the tail lemma

This note continues [B_tail.md](B_tail.md) (gap (b) of Conjecture 1, [PROOF.md](../PROOF.md) §4 A.3 and stage 4 of §7). It
computes the seven-harmonic zero, updates the truncation-sequence analysis with it, states the tail lemma for the
jump-conditioned truncation precisely, proves the parts of it that can be proved by hand, estimates every constant from
the data, and names the one inequality that needs a new idea. Written by Claude (Anthropic) under Nathan Moinvaziri's
direction; not reviewed by a mathematician. Status words as in AGENTS.md: *proved* (by hand), *numerical evidence*,
*conjecture*, *conditional* (hypothesis named).

Scripts (all new; nothing existing was changed):

- `solver/inner_stokes/inner_nb.py`: a compiled (numba) backend for `inner_np.Inner`: the same series, right-hand side
  and DOP853 integrator (scipy's algorithm transcribed), 15–20× faster at M = 7.
- `solver/inner_stokes/hier_nb.py`: `hier.py` unchanged, run on that backend, with a `--pretune` start and a lazy
  residual (only the radii whose entries `hier.py` reads are integrated while a level approaches).
- `solver/proof/trunc_seq_M7.py`: the truncation sequence, ratios, coupling and limit estimates.
- `solver/proof/amp_trunc_hiermodel.py`: the amplitude expansion of `hier.py`'s own M-harmonic model (§4).

PLACEHOLDER-SUMMARY

## 1. Making M = 7 affordable

The README records that the first cold level-5 solve of M = 7 had not finished after 5 hours on the 8-thread machine.
Two causes, both in the cost of one residual, and two fixes that leave `hier.py`'s algorithm untouched.

- **Python overhead.** One M = 7 residual is one series (J ≈ 240 terms of a 25-component system) and four ray-plus-arc
  DOP853 integrations, ~10⁴ right-hand sides of ~20 µs of arithmetic each, wrapped in ~100 µs of `solve_ivp` and
  `np.convolve` overhead. `inner_nb.py` compiles the same series and the same right-hand side, and transcribes scipy's
  DOP853 (coefficients imported from scipy; initial step, error norm and step control as in `scipy/integrate/_ivp`).
  At M = 3 the two backends take exactly the same number of right-hand sides on every ray and arc (1706, 4322, 3290,
  1442) and agree to 10⁻¹⁴. Through `hier.demixed_multi`, at the M = 1, 3, 5 points and the tuned M = 7 point of
  `logs/hier_M7_fixA_run2.log`, every residual entry that `hier.py` reads differs from scipy's by at most 3.6 times scipy's
  own noise (its change when rtol goes from 10⁻¹² to 0.8×10⁻¹²); a sign flipped in the compiled right-hand side makes the
  check fail by 10¹⁴ (`inner_nb.py check 1 3 5 7`, `--mutate=sign`; logs `inner_nb_check*.log`). The entries that `hier.py`
  does not read (r₁, r₃ at the approach radii, where e^{1/(2ξ²)} ≈ 10²⁰ amplifies rounding) differ by up to 100%, as they
  must. Speed under the cloud load of this session (load average 16–22 on 2 cores): 1.5 s against 25 s per M = 7
  residual.
- **The arc at ξ = 0.2 while the top harmonic is untuned.** `hier.py` evaluates every residual at all radii from one
  integration. With F₇ off its tuned value by 1% (the amplitude-expansion seed), the harmonic-7 jump at ξ = 0.2 is large
  and nonlinear and the arc to the real axis there takes about 1.2×10⁶ right-hand sides (in either backend), against
  ~10⁴ near the tuned set. That is what made the first level-5 solve take hours. `hier_nb.py` does two things about it:
  `--pretune` solves S₇ = 0 alone first, integrating only to its approach and polish radii (15 s), and `LazyHier`, while
  a level ν ≥ 5 is in its approach mode, integrates only up to the largest radius whose entries `hier.py` can read (the
  approach radius of that level and the polish radii above it). The radii below that are `hier.py`'s own list, so the
  staged ray segments are the same and every entry read is bit-identical (`hier_nb.py selftest`, all four mode
  combinations; `--mutate=radius` fails by 290). The only visible difference is the "start residuals" line, which prints 0
  for the entries not integrated.

With both, the first cold level-5 solve at M = 7 took 362 s (229 residuals) under that load.

PLACEHOLDER-M7

## 4. Which truncation: hier's model is a Galerkin truncation too

B_tail.md §4 concludes that a tail lemma "cannot be a statement about Galerkin projections" because `amp_trunc.py`, the
amplitude recursion with every θ-function projected onto |μ| ≤ M, gives σ_{M+1} too large by 10⁴–10⁶. That conclusion
needs one correction, reported here and not fixed in B_tail.md:

- `hier.py`'s model 𝔐_M is itself a Galerkin truncation of the inner equations in θ (`inner_np.py`: "Galerkin-truncated in
  theta"), with the jump conditions imposed on the truncated system. `amp_trunc.py` imposes the same jump conditions
  (it tunes f₃ … f_M order by order). The two differ in one harmonic: `inner_np.py` keeps the even harmonics of c and E up
  to |μ| = M + 1, `amp_trunc.py` stops them at M − 1 (amp_flint's truncation |μ| ≤ H with H = M).
- Lemma O (§6) shows that this one harmonic is exactly what moves `amp_trunc.py`'s first change to σ_M: with the even
  harmonic M + 1 dropped the threshold is 2M + 2 − |h| instead of 2M + 4 − |h|. The 5×10⁻⁴ changes of σ₃ (M = 3) and σ₅
  (M = 5) in B_tail.md's table are that harmonic.
- `amp_trunc_hiermodel.py` runs the amplitude expansion of 𝔐_M itself (amp_flint's truncation with H = M + 1, which
  keeps the odd harmonics ≤ M and the even ones ≤ M + 1; its de-mixing loop needed a parity fix, documented in the
  script). Results at ω∞, J = 500, legacy fit with every family present:

PLACEHOLDER-AMP

So the distinction that matters for the tail lemma is not "jump-conditioned versus Galerkin" but which harmonics the
truncation keeps, and whether the increment is measured order by order or at the actual amplitude. The statement of §5
is for 𝔐_M, the model whose sequence is measured.

## 5. The tail lemma, stated

### 5.1 Objects

For odd M, 𝔐_M is the model `hier.py` solves (`inner_np.Inner` with every homotopy weight 1): y and σ carry the odd
harmonics |μ| ≤ M, c (and E = y²/2) the even harmonics |μ| ≤ M + 1, and every product is projected onto these spaces.
Its data are u = (A, ω, f₃, …, f_M) with F = A sin θ + Σ (B_ν sin νθ + C_ν cos νθ) and f_ν = F_ν.

- **Multipliers.** At fixed M the formal solution regular at ξ = 0 is 1-summable (Braaksma; [C_selection.md](C_selection.md)
  B4), and the difference of its lateral sums defines the label multipliers S_ν^{[M]}(u), |ν| ≤ M. Normalised:
  Ŝ_ν = S_ν/(2κ_ν), with κ_ν the linear (Weber) connection coefficient of label ν, so that Ŝ_ν = F_ν + (nonlinear terms)
  and |κ_ν| ≍ e^{π|ν|ω/2}.
- **Jump-conditioned truncation.** Hypothesis (H_T^M): the Jacobian ∂(Ŝ₃, …, Ŝ_M)/∂(f₃, …, f_M) is invertible on the box.
  Then the tuned set {Ŝ₃^{[M]} = ⋯ = Ŝ_M^{[M]} = 0} is a graph f = f^{[M]}(A, ω), and
  s_M(A, ω) = Ŝ₁^{[M]}(A, ω, f^{[M]}(A, ω)). Its zero is p_M = (A_M, ω_M), and c_M = ω_M/ω∞.
- **Increment.** δ_M = s_{M+2} − s_M, a function on the box: two different models, each on its own tuned set.
- **Untruncated.** s_∞, under hypothesis (H_∞): the tuned untruncated profile exists in X_σ and its harmonic-1
  multiplier is defined (gaps (a) and (b) at the level of existence; the amplitude expansion defines it order by order).
- **Norms** (B_tail.md §6). Profile: ‖f‖_σ = Σ_{ν odd ≥ 3} |f_ν| e^{σν} (the space X_σ, σ < σ_F = 1.428). Multipliers, in
  the dual norm: |Ŝ|*_σ = sup_ν |Ŝ_ν| e^{−σν}, i.e. |S_ν| with the weight e^{−σν}/|κ_ν| ≍ e^{−(σ + πω/2)ν}.

### 5.2 The statement

> **Lemma T (tail lemma; target, not proved).** Let B be the box of radius ε about p₇ in (A, ω/ω∞). There are
> σ ∈ (0, σ_F) and explicit constants Λ, Q, C_f, ρ_* with ρ_*² e^{2σ} ≤ 1/4 such that for every odd M ≥ 5 and every
> (A, ω) ∈ B:
>
> - (T1, profile) |f^{[M+2]}_{M+2}(A, ω)| ≤ C_f ρ_*^{M+1};
> - (T2, coupling) |δ_M(A, ω)| ≤ Λ e^{σ(M+2)} |f^{[M+2]}_{M+2}| + Q (e^{σ(M+2)} |f^{[M+2]}_{M+2}|)²;
> - (T3, consistency) s_M → s_∞ uniformly on B.
>
> Consequently |s_∞ − s₇| ≤ Σ_{M ≥ 7} |δ_M| on B (Lemma C), and Theorem A(∞) follows from a certified Theorem A(M = 7)
> on B whenever that sum is below σ_min(J₇) ε, with J₇ the outer Jacobian of s₇.

In ratio form, which is what the data measure:

> **(R)** sup_B |δ_{M+2}| ≤ κ sup_B |δ_M| for odd M ≥ 5, with κ ≤ 1/4.

(T1) and (T2) give (R) with κ = ρ_*² e^{2σ} (1 + O(Q e^{σ(M+2)} |f_{M+2}| / Λ)).

Three remarks fix what the statement is and is not.

1. Both s_M and s_{M+2} are jump-conditioned, each in its own model: no Galerkin projection of one tuned solution onto
   fewer harmonics appears, only two tuned problems side by side. (T2) is B_tail.md's second half, "|r₁^{(∞)} − r₁^{(M)}|
   ≤ L ‖(I − P_M)F‖_σ", written for consecutive truncations, so that every quantity in it is one `hier.py` computes.
2. Λ is the norm of the linearized coupling in the dual space X_σ*: the derivative of the tuned harmonic-1 multiplier with
   respect to the newly added harmonic pair, the lower harmonics re-tuned (identity H in §6 makes this exact).
3. The weights e^{π|ν|ω/2} do not appear in (T2). In `hier.py`'s raw residuals they appear as couplings such as
   ∂r₁/∂B₅ ≈ 10¹⁷; Lemma R shows that these are properties of the residual, not of the tuned problem.

What κ ≤ 1/4 asks of σ: with the measured profile decay ρ_*² = 0.0574 per pair, ρ_*² e^{2σ} ≤ 1/4 holds for
σ ≤ ½ ln(1/(4 × 0.0574)) = 0.736. So the coupling bound is needed in X_σ* for some σ ≤ 0.736, i.e. the linearized
coupling of harmonic ν may grow at most like e^{0.736ν}, against e^{πων/2} = e^{6.0ν} for the raw couplings.

## 6. What is proved

All by hand, under the hypotheses named; none uses a numerical value.

**Lemma R (the Schur-corrected residual is representation-independent).** Fix M and let r(u) ∈ ℝ^{M+1} be `hier.py`'s
raw residual (the de-mixed normalised jump at the radii it uses). Hypothesis (H_6a^M): r(u) = Φ_u(Ŝ(u)) with Φ_u a
C¹ map, Φ_u(0) = 0, T(u) = DΦ_u(0) invertible (the jump at those radii is fixed by the multipliers, vanishes exactly when
they all vanish, and is linear in them to first order; for M = 1 this is lemma 6a of [A_certify.md](A_certify.md) §11).
Write 1 for the harmonic-1 rows and H for the rows 3, …, M. Then:

1. On {r_H = 0}, r₁ = τ(Ŝ₁) with τ(0) = 0 and τ′(0) = T̃ = T₁₁ − T₁H T_HH⁻¹ T_H1, the Schur complement of T, which is
   invertible. So the zero `hier.py` finds (r₁ = 0 on r_H = 0) is the zero of s_M (Ŝ₁ = 0 on Ŝ_H = 0), exactly.
2. At that zero the outer Jacobians are related by D_p(r₁|_{r_H=0}) = T̃ (I − G h′)⁻¹ J_s, where J_s = D_p s_M,
   G = ∂_fŜ₁ (∂_fŜ_H)⁻¹ and h′ = −T_HH⁻¹ T_H1, whenever I − G h′ is invertible. So the zero is simple for one exactly when it is
   simple for the other.
3. The same identity holds for the Schur complement of hier's leakage Jacobian K = D r = T DŜ: the entries of T, such as
   ∂r₁/∂B₅ ≈ 10¹⁷ ≈ e^{π(5−1)ω/2} × (the finite-ξ de-mixing error), cancel exactly in it.

*Proof.* Assume also T_HH invertible (`hier.py` needs this to define its correction; its K_HH is invertible). Then
Φ_H(Ŝ₁, Ŝ_H) = 0 is solved by Ŝ_H = h(Ŝ₁) with h(0) = 0 and h′(0) = −T_HH⁻¹T_H1 (implicit function theorem), and
r₁ = Φ₁(Ŝ₁, h(Ŝ₁)) =: τ(Ŝ₁) with τ′(0) = T₁₁ + T₁H h′(0) = T̃, invertible since det T = det T_HH · det T̃. For 2,
differentiate Ŝ_H(p, f^r(p)) = h(Ŝ₁(p, f^r(p))) at the zero: with a = D_p(Ŝ₁ on {r_H = 0}), f^r′ = (∂_fŜ_H)⁻¹(h′a − ∂_pŜ_H), so
a = J_s + G h′ a. For 3, K's Schur complement is D_p of r₁ along the linearised set {K_H du = 0}, which at the zero is the
tangent of {r_H = 0}; apply 2. ∎

h′ is the leakage of label 1 *up* into the rows of higher labels, which carries e^{−π(μ−1)ω/2}, so I − G h′ ≈ I. Lemma R
is the precise form of the sentence "the weights multiply zero and cancel in the tuning" (B_tail.md §2): the ten to
seventeen orders of magnitude in `hier.py`'s K are a property of measuring multipliers through a finite-ξ jump, and they
leave neither the zero nor its Jacobian.

**Lemma O (order counting in hier's model).** Expand 𝔐_M in the amplitude, F = A sin θ + Σ_{ν ≥ 3} f_ν(A) e^{iνθ} + c.c.,
with every f_ν^{(k)}, 3 ≤ ν ≤ M, tuned order by order (Ŝ_ν^{(k)} = 0), and S₁/S₁^{(1)} = 1 + Σ σ_j^{[M]} x^j, x = A². Compare
with the untruncated expansion (σ_j, f_ν^{(k)}). Then every coefficient of 𝔐_M at order A^k and harmonic h agrees with the
untruncated one whenever k < 2M + 4 − |h|. In particular:

- σ_j^{[M]} = σ_j for j ≤ M, and f_ν^{(k),[M]} = f_ν^{(k)} for k ≤ 2M + 2 − ν;
- for the truncation of `amp_trunc.py` (every function at |μ| ≤ M, so the even harmonics stop at M − 1) the threshold is
  2M + 2 − |h| instead, and σ_M is the first coefficient that changes.

*Proof.* At order k only harmonics with |h| ≤ k, h ≡ k (mod 2), occur (true at k = 1; products add orders and harmonics; a
tuned term f_ν^{(k)} has |ν| ≤ k). Call (k, h) *different* if the order-k, harmonic-h coefficient (of y, σ, c, E, or of
the tuned F) differs between the two expansions. Claim: different ⇒ k ≥ τ(h) := 2M + 4 − |h|. A coefficient is different
for one of three reasons. (i) h lies outside 𝔐_M's space (odd |h| ≥ M + 2 or even |h| ≥ M + 3): then k ≥ |h| ≥ τ(h).
(ii) It is a sum of products one of whose factors, (k₁, h₁), is different: by induction k₁ ≥ τ(h₁), and the other
factors have total order k − k₁ ≥ |h − h₁| ≥ |h₁| − |h|, so k ≥ τ(h₁) + |h₁| − |h| = τ(h). (iii) It is a tuned term
f_ν^{(k)} whose order-k multiplier Ŝ_ν^{(k)} differs: that multiplier is a functional of the de-mixed late orders
Σ_μ [e^{−Π}]^{(k″)}_{ν−μ} d^{(k′)}_μ with k′ + k″ = k, and a different d^{(k′)}_μ needs k′ ≥ τ(μ) while [e^{−Π}]^{(k″)}_{ν−μ} ≠ 0
needs k″ ≥ |ν − μ|, so again k ≥ τ(ν) (the same holds for the ⟨F²⟩ series in the normalisation, harmonic 0). For h = 1,
k ≥ 2M + 3, i.e. relative order A^{2M+2}: σ_{M+1}. For `amp_trunc.py` the first dropped harmonic is the even M + 1, at
order M + 1, and the same induction with τ′(h) = 2M + 2 − |h| gives σ_M. ∎

(The multipliers here are the mathematical ones, defined order by order by lateral Borel sums; a late-order fit reads
them to its own accuracy, ~10⁻¹⁰.)

**Lemma L (the lowest-order increment is linear in the tail).** At the first order where δ_M can be non-zero,
δ_M = Δσ_{M+1} x^{M+1} + O(x^{M+2}) with Δσ_{M+1} = σ_{M+1} − σ_{M+1}^{[M]}, and Δσ_{M+1} is a real-linear functional of
the top-degree harmonic ±(M + 2) of the untruncated solution, y^{(M+2)}_{±(M+2)}(ξ) (= 2f^{(M+2)}_{M+2} times the regular
Kummer series plus the part forced by the lower top-degree harmonics).

*Proof.* In the induction of Lemma O, a coefficient at its threshold (k = τ(h), h ≠ 0) cannot contain two different
factors: different factors have k_i ≥ max(τ(h_i), |h_i|) ≥ M + 2, so two of them need k ≥ 2M + 4 > τ(h). So at threshold
the difference is linear in the threshold data, and descending from h = ±(M + 2), k = M + 2, every threshold
coefficient is a linear image of y^{(M+2)}_{±(M+2)}. (At h = 0 the threshold 2M + 4 allows the quadratic term
|f_{M+2}|² of ⟨F²⟩, which reaches harmonic 1 only at order 2M + 5.) ∎

**Lemma C (telescoping).** If (T3) holds and sup_B |δ_{M+2}| ≤ κ sup_B |δ_M| for odd M ≥ 5 with κ < 1, then
sup_B |s_∞ − s₇| ≤ κ/(1 − κ) · sup_B |δ₅|. *Proof:* s_∞ − s₇ = Σ_{j ≥ 1} δ_{5+2j}, and |δ_{5+2j}| ≤ κ^j sup|δ₅|. ∎

**Identity H (the increment as a coupling integral).** Call the harmonics of 𝔐_M old and the odd M + 2 and even
M + 3 new. Let 𝔐^λ_{M+2}, 0 ≤ λ ≤ 1, be 𝔐_{M+2} with two kinds of product terms multiplied by λ: in the equations of the
old harmonics, every term with a new factor (the back-reaction), and in the equations of the new harmonics, every term
with no new factor (the forcing by the old ones). Terms linear in the new harmonics with old coefficients (such as
c₀σ_{M+2}) and new–new terms keep weight 1. At λ = 0 the old harmonics obey 𝔐_M exactly and the new block is unforced,
so its tuning gives f_{M+2} = 0, y_new ≡ 0 and s^0 = s_M; at λ = 1, s^1 = s_{M+2}. If (H_T) holds along the path,

```math
\delta_M = \int_0^1 G(\lambda)\,d\lambda, \qquad G = \partial_\lambda \hat S_1 - \partial_f \hat S_1\,(\partial_f \hat S_H)^{-1}\,\partial_\lambda \hat S_H \quad (H = 3, \dots, M+2),
```

a Schur complement: the rate at which the tuned harmonic-1 multiplier changes as the coupling to the new pair is switched
on, every other label re-tuned. G(0) = 0: at λ = 0 the new harmonics vanish, so ∂_λ of the back-reaction terms and
∂Ŝ_old/∂f_{M+2} vanish there. So δ_M is first order in the forcing of the new pair and first order in its back-reaction.
This is the exact meaning of "the linearized coupling" below. *Proof:* chain rule and the implicit function theorem on
the tuned set. ∎ (`inner_np.Inner`'s homotopy weight is a different interpolation: it also multiplies the new harmonics'
linear terms, so at its λ = 0 the new block is degenerate rather than a decoupled tuned problem.)

PLACEHOLDER-CONST

## 8. The ratio bound: an attempt, and the inequality that needs a new idea

### 8.1 The mechanism the data show: a tuning cascade

PLACEHOLDER-CASCADE

### 8.2 Lowest order in the amplitude: what can be proved

By Lemma L, the lowest-order coefficient of the increment, Δσ_{M+1}, is linear in the top-degree harmonic M + 2. At top
degree (harmonic h at order A^h, no conjugates) the inner problem closes on the positive harmonics: it is the inner problem
for a holomorphic profile F⁺(w) = Σ_{ν > 0} f_ν^{(ν)} w^ν, w = A e^{iθ}, and its tuning conditions say that the top-degree
jump is a pure label-1 jump. Each order is then one *linear* forced Weber problem for one harmonic,

```math
y^{(\nu)}_\nu = 2 f^{(\nu)}_\nu\,u_\nu + y_{g_\nu}, \qquad \hat S_\nu\big[y^{(\nu)}_\nu\big] = f^{(\nu)}_\nu + \frac{S_\nu[y_{g_\nu}]}{2\kappa_\nu} = 0 ,
```

with u_ν the regular Kummer series (multiplier κ_ν by definition) and y_g the formal solution with zero datum and the
forcing g_ν built from lower top-degree harmonics. Two statements would give (T1) and (T2) at lowest order:

- (W) **single-step cancellation**: S_ν[y_g] = κ_ν P_ν[ĝ] with P_ν a functional of the Borel transform ĝ that does not
  contain κ_ν. *Sketch (not a proof):* in the Borel plane y_g = R_ν ∗ ĝ with R_ν the resolvent kernel of the Weber operator,
  whose only singularity at ζ = ½ has strength κ_ν; if ĝ is analytic on [0, ½] (true at top degree, where every label
  except 1 is tuned and label 1 enters harmonic ν only through e^{Π⁺}, as a separate family), the singular part of a
  convolution at ½ is the singular part of R_ν convolved with ĝ, so κ_ν factors out. Then f_ν^{(ν)} = −P_ν[ĝ_ν]/2: the
  weight cancels in every step (B_tail.md §2), exactly.
- (U) **uniformity**: |P_ν[ĝ]| ≤ C ‖g‖ in a norm that the forcing's harmonic structure controls, with C independent of ν.

(W) is classical in structure and should be provable with the explicit Kummer resolvent. (U) is where it stops: P_ν
integrates ĝ against a kernel that oscillates like ζ^{iνω/2} (the Weber exponent α_ν = (1 + iνω)/2), and a bound uniform in
ν needs either that oscillation (stationary phase, which gives decay in ν but with constants that depend on derivatives
of ĝ that themselves grow with the order) or analytic continuation of ĝ off the real ζ axis (which brings back a factor
e^{νω·angle}, see 8.4). Measured, the leading-order tuning ratio q_ν is ν-independent to six digits (B_tail.md §2), so (U)
holds in fact for the profile; no proof of it is known here.

PLACEHOLDER-LOWEST

### 8.3 Why the lowest order does not settle the ratio at A*

PLACEHOLDER-WHYNOT

### 8.4 The inequality that needs a new idea

> **(★) Uniform coupling bound.** For the tuned untruncated problem on B and every odd ν ≥ 7, the linearized coupling of
> identity H (the Schur complement G for the pair ν, integrated over λ) satisfies |G_ν| ≤ Λ e^{σν} |f_ν| with σ ≤ 0.736.

Everything else in Lemma T is either proved above (R, O, L, C, H), a contraction estimate of a standard kind (T1, in
X_σ, B_tail.md §6), or a consistency statement (T3) that follows from (★) and (T1). The data say (★) holds with room to
spare: the coupling per pair is g ≈ PLACEHOLDER-G, i.e. the effective σ is ½ ln g ≈ PLACEHOLDER-SIGEFF, while 0.736 is
needed. The naive bound, through the linear connection coefficients, is e^{πων/2} = e^{6.0ν}.

**Where the weights come from, as an angle.** On a ray ξ = |ξ| e^{iβ} the label harmonic is
e^{iνθ₀} = e^{iν(θ − 2ω ln|ξ|)} e^{2νωβ}. So:

- the linear weight κ_ν ≍ e^{πνω/2} is the label continued from the Stokes line (β = 0) to the anti-Stokes line
  β = π/4 (arg ξ² = π/2), where the fast modes are O(1): the lateral sums of `inner_np.py` start there and the arc back
  to the real axis is where the amplification happens;
- F's singularities reach the label at Im θ₀ = ±σ_F, i.e. β = ±σ_F/(2ω) = ±0.187 (B_tail.md §3);
- (★) with σ ≤ 0.736 asks for β ≤ 0.736/(2ω∞) = 0.096 (5.5°), half-way to the first singularity.

So the new idea needed is a representation of the tuned linear response along paths that stay within about 5° of the
Stokes line, with bounds uniform in the label harmonic. Every construction of the multipliers used so far (lateral
Borel–Laplace sums, the jump at finite ξ, the late-order fits) goes through the anti-Stokes direction or through
individual labels, and so meets e^{πνω/2} before the cancellation. Two candidates:

1. **The reality form of the condition.** For real F the lower lateral sum is the conjugate of the upper one on the real
   ξ axis, so the jump is 2i Im y⁺ and S ≡ 0 says that y⁺ is real on the Stokes line. The linearized coupling is then
   Im δy⁺ on the Stokes line, and δy⁺ is a Laplace integral along arg ζ = 0⁺ of a Borel transform whose growth near the
   real ζ axis is what has to be bounded. In the label variable that is a Costin–Tanveer strip-analytic Borel estimate
   (PROOF.md §3) with a strip of half-width 2ωβ ≈ 0.74 inside the analyticity strip 1.43.
2. **The cascade as a transfer operator.** If the downward step of 8.1 is, at large ν, a fixed operator acting on label
   space (as q_ν is a fixed ratio), its spectral radius is the coupling g; a proof of (★) is then a bound on one operator
   instead of on infinitely many Schur complements. This is the route suggested by the ν-independence of q_ν.

# P6 scoping note: uniformity in the number of harmonics

Queue item P6 of [PROOF_QUEUE.md](PROOF_QUEUE.md) is gap (b). The label weights e^{π|ν|ω/2} grow faster than the
profile's harmonics decay ([proof/B_tail.md](../proof/B_tail.md)). [proof/B_tail_M7.md](../proof/B_tail_M7.md) states
the tail lemma (Lemma T) for the jump-conditioned truncations and proves its algebraic parts. Its §8.4 names (★), a
coupling bound uniform in the harmonic, as the one inequality that needs a new idea. This note scopes the work on (★).
It covers the statements and what each lacks (§1), what the numbers say (§2), the candidate routes in the notes and in
the literature they cite (§3), and a staged plan (§4). It runs nothing heavy. Its only computation is
`trunc_seq_M7.py` on the newer M = 7 zeros (§7). Written 2026-11-18 by Claude (Anthropic) under Nathan Moinvaziri's
direction for issue #979; not reviewed. Status words as in AGENTS.md.

## Summary

- **(★) is still the open part, and no number contradicts it.** The coupling per pair is 1.21 at M = 5 and 0.85 at
  M = 7. That is an effective σ of 0.095 and −0.08, against the 0.736 that a ratio of ¼ needs (*numerical evidence*,
  §2.1).
- **No measured ratio lies inside (R)'s range.** (R) is stated for M ≥ 5, so its first instance is δ₇ against δ₅, which
  M = 9 measures. The measured κ₅ and κ₇ compare δ₃ with δ₁ and δ₅ with δ₃. They support (R) by extrapolation only, and
  the first coupling ratio wholly inside (★)'s range is also M = 9's (§2.2).
- **The M = 7 error bar is not the binding input.** Across rtol 10⁻¹² to 10⁻¹⁴ on x86 and aarch64, κ₇ stays in −0.0485
  to −0.0488 and the limit in 0.999985–0.999991 (§2.3). The zero has converged in rtol from 10⁻¹³ on, at the size of
  its noise estimates (#977). Radius independence (Lemma R) is untested at 2×10⁻⁵. It tests the numerics, not (★).
- **One step of B_tail_M7.md §5.2 does not follow.** (T1) and (T2) are upper bounds. They give a geometric bound on
  |δ_M|, not the ratio inequality (R) (§1.3). The telescoping still closes from (T1)–(T3), but it then needs Λ and C_f
  explicitly, in place of the measured sup|δ₅|.
- **Five candidate routes are named (§3). None has been checked against its source.** The notes read the key
  literature (Costin–Tanveer, Xie–Tanveer, Olde Daalhuis) at metadata level only (LIT_REVIEW.md §6).
- **M = 9 is blocked at level 5.** Its first level-5 solve stalls at |r₅| ≈ 43 with the floor rule off and at rtol
  10⁻¹² alike (#974), so the first test of (R) waits on a level-5 design change (§2.5).
- **First steps (§4).** A level-5 approach-radius pilot at M = 9 (#974's recommendation 1), then δ₇(p₇) = r₁^{[9]}(p₇)
  at the M = 7 zero. In parallel, check symbolically whether Lemma W's cancellations hold identically in ν.

## 1. The statements, their status, and what each needs

### 1.1 As stated

The objects are those of B_tail_M7.md §5.1. 𝔐_M is `hier.py`'s M-harmonic model, with odd harmonics |μ| ≤ M in y and
σ and even harmonics |μ| ≤ M + 1 in c and E. On its tuned set {Ŝ₃ = ⋯ = Ŝ_M = 0} the normalised harmonic-1 multiplier is
s_M(A, ω), with zero p_M. The increment δ_M = s_{M+2} − s_M compares two models, each on its own tuned set. The
profile lives in X_σ (Σ |f_ν| e^{σν} < ∞, σ < σ_F = 1.428) and the multipliers in its dual.

- **Lemma T** (§5.2; *conjecture*, the target). Let B be the box of radius ε about p₇. There are σ ∈ (0, σ_F) and
  explicit Λ, Q, C_f, ρ_* with ρ_*² e^{2σ} ≤ ¼ such that for every odd M ≥ 5 on B:
  - (T1) |f^{[M+2]}_{M+2}| ≤ C_f ρ_*^{M+1};
  - (T2) |δ_M| ≤ Λ e^{σ(M+2)} |f^{[M+2]}_{M+2}| + Q (e^{σ(M+2)} |f^{[M+2]}_{M+2}|)²;
  - (T3) s_M → s_∞ uniformly on B.

  Its consequence is Theorem A(∞) from a certified Theorem A(M = 7) on B, when Σ_{M≥7} |δ_M| < σ_min(J₇) ε.
- **(R)** (§5.2; *conjecture*). sup_B |δ_{M+2}| ≤ κ sup_B |δ_M| for odd M ≥ 5, with κ ≤ ¼.
- **(★)** (§8.4; *conjecture*). For every odd ν ≥ 7, |δ_{ν−2}| ≤ Λ e^{σν} |f^{[ν]}_ν| + (quadratic), with Λ independent
  of ν and σ ≤ 0.736. It is the linear part of (T2), uniformly in ν. The bar 0.736 = ½ ln(1/(4 × 0.0574)) is what
  κ ≤ ¼ needs at the measured profile decay ρ_*² = 0.0574. In angle it asks for paths within 5.5° of the Stokes line.
- **Lemma R** (§6; *proved* by hand, *conditional* on (H_6a^M) and T_HH invertible). The Schur-corrected residual of
  `hier.py` has exactly the zeros of s_M, and a zero is simple for one exactly when it is simple for the other. The raw
  couplings, such as ∂r₁/∂B₅ ≈ 10¹⁷, cancel exactly in the Schur complement. A consequence the runs test is that the
  zero does not depend on the matching radius ξ₁.

The rest of §6 is *proved*: Lemma O (order counting), Lemma L (the lowest-order increment is linear in the tail),
Lemma C (telescoping) and identity H (the increment as an integral of a Schur-complement coupling G(λ)). Lemma W (the
weight cancels exactly in every tuning step) is *proved* as algebra, given D_entire.md §6.1's normalisation.

### 1.2 What each needs that is not proved

| piece | status | what is missing | where |
|---|---|---|---|
| (T1), profile bound | not proved; numerical evidence ρ_*² = 0.0574 | a contraction for the tuning map in X_σ. The leading map has norm 0.0633 e^{2σ} (0.27 at σ = 0.736), and the A-series corrections (measured term ratio about 0.25) need rigorous bounds | B_tail.md §6; B_tail_M7.md §7 |
| (T2), linear part | *conjecture*: this is (★) | a Λ uniform in ν. The naive bound through the connection coefficients is e^{πων/2} = e^{6.0ν} | B_tail_M7.md §8.4 |
| (T2), quadratic part | not addressed | an explicit Q. Identity H makes G(0) = 0 and δ_M first order in forcing and back-reaction, so Q is the second-order remainder of ∫G dλ | B_tail_M7.md §6 |
| (T3), consistency | *conjecture* | (H_∞), that the untruncated tuned profile exists in X_σ and its harmonic-1 multiplier is defined. (T1)–(T2) make s_M Cauchy, so they give a limit, but not that the limit is s_∞ | B_tail_M7.md §5.1 |
| (H_T^M) for every M ≥ 9 | assumed per M | the tuning Jacobian invertible on B for every M. (T1)'s contraction would give it uniformly | B_tail_M7.md §5.1 |
| (H_6a^M) | *certified* for M = 1, 3, 5 on the boxes of the Theorem A(M) certificates (lemma 6a; at M = 5 a point box) | M = 7 and M = 9, and a box of radius 10⁻³ | A_lemmas_M.md §4 |
| Theorem A(M = 7) on B, and sup_B \|δ₅\| | not done. Theorem A(M = 5) is *certified* on a point box of radius about 10⁻¹⁸ | a certified M = 7 zero, and enclosures of s₅ and s₇ over a box of radius 10⁻³. The M = 5 box runs of A_M.md §7 failed, and the certificate that passed is a point certificate with a second-order chart. DINF_ANALYTIC.md §4.2 puts M = 7 at weeks of runs and engineering | PROOF.md §4; A_M2.md |
| r₁ against Ŝ₁ | not done at M = 7 | Lemma T is written in normalised multipliers, but the certificate's margin (3.9 for a 10⁻³ box) is in `hier.py`'s r₁. The factor τ′(0) = T̃ of Lemma R must be enclosed, as Lemma P linked the two Jacobians for M = 1 | B_tail_M7.md §6; B_chain.md §8 |

Lemma T serves layer A, the untruncated zero. Gap (b) has a second side in layer B. Theorem B's constants depend on M
([proof/C_selection.md](../proof/C_selection.md) §7), and [proof/B3_borel.md](../proof/B3_borel.md) §3 doubts that
the outer Borel sector (H-σ) stays open uniformly in M. That side is outside Lemma T and outside this note.

### 1.3 A step that does not follow

B_tail_M7.md §5.2 says "(T1) and (T2) give (R) with κ = ρ_*² e^{2σ} (1 + O(…))". Both are upper bounds. Together they
give

```math
\lvert\delta_M\rvert \le b_M := \Lambda C_f\,e^{\sigma(M+2)}\rho_*^{M+1} + Q\,\big(C_f\,e^{\sigma(M+2)}\rho_*^{M+1}\big)^2, \qquad \frac{b_{M+2}}{b_M} \le \rho_*^2 e^{2\sigma} \ \text{ when } \rho_*^2 e^{2\sigma} \le 1 .
```

That is a geometric bound on |δ_M|. It is not (R), which bounds one increment by the previous one and would need a
lower bound on |δ_M| to follow from these. The two forms close the argument differently.

- **Ratio form (Lemma C as written).** The tail is κ/(1 − κ) times sup_B |δ₅|, which would come from certified
  enclosures of s₅ and s₇ over B. Λ and C_f never need numbers. But (R) must be proved as a ratio.
- **Bound form.** The tail is Σ_{M≥7} b_M = b₇/(1 − ρ_*² e^{2σ}) at most. A proof of (★) delivers this form, but then Λ,
  C_f and Q are needed explicitly, in r₁ units, at M = 7.

The data measure ratios (§2), so they speak to the ratio form. A proof of (★) would deliver the bound form.

## 2. What the numbers say about (★)

### 2.1 The inputs

| quantity | value | what it measures | status | source |
|---|---|---|---|---|
| κ₅ = d₅/d₃ | −0.0686 | the zero shift from δ₃ over the one from δ₁ (M = 1 is pre-asymptotic) | numerical evidence | B_tail_M7.md §3 |
| κ₇ = d₇/d₅ | −0.0485 (±0.0013 from c₇'s ±2×10⁻⁵) | δ₅ over δ₃, read in ω | numerical evidence | B_tail_M7.md §3 |
| g_M = \|κ_M\|/\|F_M/F_{M−2}\| | 1.21 (M = 5), 0.85 (M = 7) | Λ e^{2σ} per pair, if Λ does not depend on ν | numerical evidence | B_tail_M7.md §3, §8.1 |
| effective σ = ½ ln g | 0.095, −0.08 | against σ ≤ 0.736 | arithmetic | B_tail_M7.md §8.1 |
| \|x²Δσ₄/Δσ₂\|, \|x²Δσ₆/Δσ₄\| | 0.035, 0.012 | the formal (order-by-order) analogues of κ₅ and κ₇, at x = A² = 7.4 | numerical evidence (fits) | B_tail_M7.md §4 |
| re-tuning of the lower harmonics by F₇ at p₅ | \|ΔF₃\| ≈ 0.28 \|F₇\|, \|ΔF₅\| ≈ 0.11 \|F₇\| | the coupling's structure: direct, not a nearest-neighbour cascade | numerical evidence | B_tail_M7.md §8.1 |
| r₁^{[7]}(p₅) = δ₅(p₅) in r₁ units | −4.49 − 5.26i (modulus 6.9) | δ₅ at one point, ξ₁ = 0.15 | numerical evidence | B_tail_M7.md §2 |
| A-sequence ratios | −0.043 (M = 5), −0.081 (M = 7) | the same increments, read in A | numerical evidence | notes/EEG_PROFILE.md |

What they say:

- Every ratio is inside ¼ by a factor of 3 or more, and the coupling has been below 1 since M = 7. That fits the
  simplest form of (★): Λ_ν of order 1 and roughly independent of ν, as the tuning ratio q_ν is to six digits (B_tail.md
  §2).
- The lowest-order ratios fall faster (0.035, 0.012) than the zero ratios (0.069, 0.049). B_tail_M7.md §8.3 shows why
  the first cannot bound the second. At x* = 7.4 each increment's series is far from its first terms. They are a
  formal check, not an input to a certificate.
- The A ratio grew from −0.043 to −0.081 while the ω ratio fell. (R) bounds the complex δ, while κ_M is read in one
  component of the zero shift. A reading, not established here, is that δ₅ and δ₃ point in different directions in r₁.
  Then neither component's ratio is the ratio of moduli that (R) needs. That ratio has not been measured at one radius
  (stage 1b of §4).

### 2.2 Which instance each number measures

- **(R)** is stated for odd M ≥ 5, so its first instance is sup|δ₇| ≤ κ sup|δ₅|, a statement about M = 9. κ₅ and κ₇
  are its instances at M = 1 and M = 3, outside the stated range. The data support (R) by extrapolating a two-term
  sequence whose first term is pre-asymptotic.
- **(★)** covers ν ≥ 7, i.e. δ₅, δ₇, and so on. Read with (T2) as an equality, g_M = (Λ_M/Λ_{M−2}) e^{2σ}. So g₇
  compares ν = 7 with ν = 5, across (★)'s lower edge. g₉ is the first ratio wholly inside its range.
- **The lowest order** follows the same indexing. x²Δσ₈/Δσ₆, from `hier.py`'s M = 7 model in the amplitude expansion,
  would be the first lowest-order instance of (R).

So M = 9 adds the first point inside both ranges.

### 2.3 The M = 7 error bar

The zero at ξ₁ = 0.15 by integrator tolerance and machine, with what `trunc_seq_M7.py` makes of it (§7):

| rtol | machine | zero (A, ω/ω∞) | estimate | κ₇ | coupling g₇ | limit L | tail above M = 7, in r₁ |
|---|---|---|---|---|---|---|---|
| 10⁻¹² | cloud (stored) | (2.71922, 1.000022) | ±2×10⁻⁵ quoted | −0.0485 | 0.847 | 0.999985 | 0.246 |
| 10⁻¹³ | win2, x86 (#975) | (2.719240, 1.000025) | 4.1×10⁻⁶ | −0.0487 | 0.851 | 0.999988 | 0.248 |
| 10⁻¹³ | rpi5, aarch64 (#965) | (2.719236, 1.000028) | 5.8×10⁻⁶ | −0.0488 | 0.854 | 0.999991 | 0.249 |
| 10⁻¹³ | rpi5, aarch64 (#974 run 4, 4 centre solves) | (2.719236, 1.000026) | 9.9×10⁻⁶ | −0.0487 | 0.852 | 0.999989 | 0.248 |
| 10⁻¹³ | win2 (#977 run 2, perturbed repeats) | (2.719243, 1.000026) | 4.6×10⁻⁶ | −0.0487 | 0.851 | 0.999989 | 0.248 |
| 10⁻¹⁴ | win2 (#976) | (2.719232, 1.000023) | 7.7×10⁻⁶ | −0.0485 | 0.848 | 0.999986 | 0.246 |
| 10⁻¹⁴ | win2 (#976 and #977 run 1, 3 centre solves) | (2.719234, 1.000025) | 9.1×10⁻⁶ | −0.0486 | 0.850 | 0.999988 | 0.247 |

The estimate column is `m7_outer.py --local`'s estimate (noise plus Jacobian terms) from repeated centre solves. It is
not an enclosure. Every row passes fit3_repeat's harmonics to `trunc_seq_M7.py`, since the newer runs' states were not
read here, so g₇ holds
|F₇/F₅| = 0.0572 fixed.

- Across the rows the zero moves by at most 2.3×10⁻⁵ in A and 6×10⁻⁶ in ω/ω∞. κ₇ moves by 3×10⁻⁴, under 1%. The tail
  above M = 7 stays near 0.25 in r₁, and the margin for a 10⁻³ box stays at 16.
- In A the ratio (A₇ − A₅)/(A₅ − A₃) runs from −0.081 to −0.084 over the rows. It is more sensitive, since 2×10⁻⁵ is 3%
  of A's increment of 6.4×10⁻⁴ (arithmetic on the table and B_tail_M7.md §3).
- Under one repeat rule the 10⁻¹³ and 10⁻¹⁴ zeros are 8.4×10⁻⁶ apart, against estimates of 4.6×10⁻⁶ and 9.1×10⁻⁶
  (#977). So the zero has converged in rtol from 10⁻¹³ on, at the size of these estimates, and the stored 10⁻¹² zero is
  low in A by 1.4×10⁻⁵ (against the 10⁻¹⁴ zero) to 2.3×10⁻⁵ (against the 10⁻¹³ zero). #977 notes that each estimate
  rests on 2 or 3 distinct centre solves.

For (★), the M = 7 zero is settled enough. Its open part is the radius check below.

### 2.4 The radius check (Lemma R)

| ξ₁ | runs | result |
|---|---|---|
| 0.14 | #964, #965, #974 run 3, #975 | zero (2.719332, 0.999994), estimate 1.2×10⁻³ (rpi5, #974). win2: no accepted zero |
| 0.15 | #965, #974 run 4, #975, #976 | the zeros of §2.3 |
| 0.16 | #964, #965 | no accepted zero. Level 3 floors at \|r₃\| 1.2–2.2×10⁻⁵ at rtol 10⁻¹³ |

At ξ₁ = 0.14 the zero differs from 0.15's by 9.6×10⁻⁵ in A, which is 0.08 of its estimate. No radius dependence shows,
and none at 2×10⁻⁵ can show at this noise. r₁'s noise follows level 3's floor, about 2×10⁶ times |r₃| at 0.14 and
7×10⁵ times at 0.15, and that floor is 6×10⁻⁷ to 2×10⁻⁶ at 0.14 against 3×10⁻⁸ at 0.15 (#974; shared note
1795041593464). Lemma R is proved, conditionally, so the check tests the numerics and (H_6a⁷), not (★). It does not
block P6.

### 2.5 What M = 9 would add, and what would test (★)

At M = 9, #965 found that level 5's first solve stalls at |r₅| = 42.7, and that `--floor` takes the stall as level 5's
noise floor. #974 reran it without `--floor` (rtol 10⁻¹³) and with it at rtol 10⁻¹². Level 5 stalls the same way in
both. Its approach solve at ξ 0.1188 stops at |g| ≈ 20 and its polish solve at |r₅| ≈ 43, while levels 9 and 7 converge
inside it. Level 3 never iterates (|r₃| 3.9×10⁵ and 5.2×10⁵). So the floor rule is not the cause, and M = 9 needs a
level-5 design change first. #974 also printed K by level. At M = 9 its blocks into r₃ reach 4.3×10¹⁹ (from level 5) to
3.6×10²² (from level 9), and every block is 8×10³ to 4×10⁴ times M = 7's. That growth alone does not explain the stall
(#974).

Once M = 9 converges, three things follow.

- **δ₇(p₇) = r₁^{[9]}(p₇).** This is one fixed-(A, ω) solve at the M = 7 zero. Since s₇(p₇) = 0, s₉(p₇) is the
  increment, as r₁^{[7]}(p₅) was for δ₅. B_tail_M7.md §3.1 expects "a few tenths", and 0.0485 × 6.9 = 0.34
  (arithmetic).
- **κ₉ and g₉**, from the M = 9 zero. κ₉ is the first instance of (R). (★) at σ = 0.736 allows g₉ up to
  e^{2 × 0.736} = 4.36, which is the same as |κ₉| ≤ ¼ at ρ_*² = 0.0574.
- **The re-tuning pattern.** |ΔF₃|, |ΔF₅| and |ΔF₇| in units of |F₉|, beside 0.28 and 0.11 of |F₇| at M = 7. It shows
  whether the response to the added pair repeats with ν, which is what route 2 of §3 needs.

What would count:

- **A test that can fail.** κ = ¼ allows |δ₇| up to ¼ × 6.9 ≈ 1.7 in r₁ at these points. A value above 1.7, with noise
  well below that, would put (R) with κ = ¼ in doubt at its first instance. It would not refute (R) strictly, since (R)
  compares sups over B and the two values sit at p₅ and p₇. A value near 0.34 continues the sequence.
- **The noise needed.** To read κ₉ to ±20% at the expected 0.34, r₁'s noise must be at most about 0.07. At M = 7 and
  ξ₁ = 0.15 it is 0.016 to 0.028 (half-range, #975, #976). M = 9's is unknown.
- **The lowest-order instance is not cheap.** If the lowest-order ratio stays near 0.012, |Δσ₈| ≈ 0.012 × 7.9×10⁻⁹ /
  7.4² ≈ 1.7×10⁻¹² (arithmetic). That is far below the agreement of `amp_trunc_hiermodel.py`'s legacy fits at J = 500
  (≤ 4.2×10⁻¹⁰ at σ₁–σ₅, B_tail_M7.md §4). The script takes only M, K, J and wf, so a finer fit is a code change and a
  pilot.

## 3. Candidate routes to (★)

These are named, not claimed. Each lists where the idea is in the notes, the literature the notes cite for it and how
far that was read (LIT_REVIEW.md §6), and what the route would need.

1. **The reality form and a strip-analytic Borel estimate.** B_tail_M7.md §8.4 gives it as candidate 1. PROOF.md §5
   states the geometric version as Tanveer's formulation. It continues the characteristics into complex θ and tracks
   where the label θ₀ meets F's singularities. For real F the jump is 2i Im y⁺ on the Stokes line. So (★) becomes a
   bound on Im δy⁺ there, a Laplace integral along arg ζ = 0⁺. In the label variable that is a Borel estimate on a strip
   of half-width about 0.74, inside the profile's 1.43.
   - *Literature.* Costin–Tanveer (Ann. IHP C 24 (2007) 795–823, and their Navier–Stokes paper arXiv:math/0612063) prove
     Borel summability for evolution PDEs with data analytic in a strip, in weighted Fourier norms (PROOF.md §3).
     Xie–Tanveer (ARMA 166 (2003)) continues into complex domains with weighted spaces. Both were read at metadata level
     only.
   - *Needs.* First, a check that the inner problem fits Costin–Tanveer's hypotheses. As written it is an ODE system in
     ξ coupled through θ, with the label carried along θ₀ = θ − 2ω ln ξ, not an evolution PDE in ℝ⁺ × ℂ^d. Second,
     lateral sums built along paths near the Stokes line. Every construction so far (`inner_np.py`) starts on the
     anti-Stokes line (β = π/4) and meets e^{πνω/2} on the arc back. Third, the location of F's singular strings at
     |ξ| = O(1), which has not been computed (B_tail.md §3, §7 item 3). F's singularities reach the label at
     β = ±0.187, and the path needs β ≤ 0.096, half-way to them.
2. **The coupling as one limit operator.** B_tail_M7.md §8.4 gives it as candidate 2. Suppose that for large ν the map
   from the new pair's data to the re-tuned low harmonics is one operator up to a shift in label space. Then (★)
   reduces to a bound on that operator, plus finitely many computed cases.
   - *Evidence.* q_ν is ν-independent to six digits (B_tail.md §2), and the leading tuning map is a weighted shift
     (B_tail.md §6).
   - *Literature.* No source in the notes treats this step. PROOF.md §3 names Reiterer–Trubowitz's split into high and
     low Fourier modes (CMP 368 (2019) 143; abstract and introduction read) as the model for the tail.
   - *Needs.* Identity H's Schur complement G_ν written so that its ν → ∞ limit can be taken, a proof that the limit
     exists and a bound on it, and an explicit rate for ν beyond the computed cases. The first data is the M = 9
     re-tuning pattern (§2.5).
3. **Lemma W with cancellations uniform in ν.** This is B_tail_M7.md §8.2, building on D_entire.md §6.1–6.2. The weight
   cancels exactly in each tuning step. But the Darboux sums converge only after two exact cancellations of their
   O(1/j) parts, shown so far for harmonic 1 at order 3. If those cancellations are identities in ν, a ν-uniform
   O(j⁻²) remainder gives the lowest-order, formal form of (★).
   - *Literature.* The step factors T^{(ν)}_j = (α_ν)_j (β_ν)_j / j! carry the large parameter α_ν = (1 + iνω)/2, so
     the tools are large-parameter asymptotics of Gauss and Kummer functions. Paris (J. Classical Anal. 2 (2013)
     183–203) was read in full for Proposition 4. Temme (JCAM 2003) was read as an abstract, and Olde Daalhuis's
     large-parameter papers were not read (LIT_REVIEW.md §6, §8).
   - *Needs.* The cancellation identities for general ν, then the remainder bound.
   - *Limit.* B_tail_M7.md §8.3 shows that the lowest order does not fix the ratio at x* = 7.4. This route gives a
     formal (R). A certificate also needs each increment's whole series, which is route 4's machinery applied to δ_M.
4. **A bypass through the amplitude series.** This is PROOF.md §4 route A3, [proof/D_entire.md](../proof/D_entire.md),
   and [proof/P5_scope.md](../proof/P5_scope.md) §1. Every order of the amplitude expansion is a trigonometric
   polynomial, so no harmonic truncation enters. Theorem A(∞) becomes *conditional* on (H_σ), (H_T) and (H_ω) instead of
   Lemma T, and the fixed-ω conditional certificate exists (PROOF_QUEUE.md P5).
   - *Needs.* The type bound (H_T), which is P5's new mathematics, and rigorous σ_k.
   - *What it does not do.* It does not prove (★). It does not remove gap (b) at the level of existence either, since
     the series-defined S₁ must still be shown to be the multiplier of an untruncated solution (DINF_ANALYTIC.md §4.3).
     It changes what P6 is for. If P5's route lands, (★) is needed for Theorem B in the strip-analytic setting
     (PROOF.md §7, stage 6), and not for Theorem A(∞).
5. **Late-order (Dingle) prefactors of the neighbour families.** This builds on B_tail.md §3. The family amplitudes fall
   by 10⁴–10⁸ per pair instead of growing by about 25. A formal late-order computation of their prefactors as functions
   of ν would give a formal (★) and predict Λ_ν.
   - *Literature.* Chapman–King–Adams (Proc. R. Soc. A 454 (1998) 2733–2755), already the formal backbone of
     Proposition 4 (PROOF.md §3).
   - *Needs.* The family prefactor in closed form for general ν. The route is formal, so it could guide routes 2 and 3
     but not replace them. B_tail.md §3 leaves one point open. The ν′ = ±3 amplitudes grow with k (10⁻¹⁶ to 10⁻⁸),
     and whether that is a real subleading term or fit contamination is not resolved.

Every route assumes Braaksma's theorem at fixed M (C_selection.md, Lemma B4(M)), as Theorem A(M) does. Costin (Duke
Math. J. 93 (1998)) gives its transseries form (PROOF.md §3). Neither gives uniformity in M.
Baldomá–Capiński–Guardia–Seara (arXiv:2206.13840) is the template for certifying a finite-M Stokes constant, which
serves Theorem A(M = 7). Lombardi's monograph is another instance of existence exactly when an exponentially small
coefficient vanishes (PROOF.md §3). Neither is a route to (★).

## 4. Staged plan

| stage | goal | first runnable step | cost | decision it informs |
|---|---|---|---|---|
| 0 | M = 9 converges | done in part: #974 shows level 5 stalls with the floor rule off and at rtol 10⁻¹². Next, #974's recommendation 1: level 5 alone at several approach radii and seeds of its harmonics | an M = 9 level-5 solve took 407 s (#974); a few radii and seeds, a few hours on rpi5 (estimate) | A radius or seed with level 5's first solve below 10⁻⁶: go to stage 1. None: M = 9 needs another design (the coordinator's), and (R)'s first test waits |
| 1 | the first instance of (R), and a coupling ratio inside (★) | M = 9 fixed-(A, ω) solves at p₇ (Z of #964, `hier_nb.py 9`), ξ₁ 0.15, rtol 10⁻¹³, with #976's perturbed repeats until two pass the bar. They give r₁^{[9]}(p₇), its noise, and ΔF₃, ΔF₅, ΔF₇ from the profile lines | about 46 min a solve on rpi5 (#965), 3 to 4 solves | \|r₁^{[9]}(p₇)\| against 0.34 (the sequence) and 1.7 (κ = ¼). A value above 1.7 with noise well below it means re-scoping before any proof work. The re-tuning pattern decides whether route 2 is worth stage 5 |
| 1b | the ratio of moduli in one normalisation | an M = 5 fixed-(A, ω) solve at p₃, ξ₁ 0.15, giving δ₃(p₃) in the units of \|δ₅(p₅)\| = 6.9 | one M = 5 solve, minutes | \|δ₅(p₅)\|/\|δ₃(p₃)\| beside κ₇ read in ω (−0.049) and in A (−0.081). It shows which number (R)'s κ is |
| 2 | the M = 9 zero | the outer fit of `m7_outer.py`, adapted to M = 9, over solves at p₇, p₇ + (2×10⁻⁴, 0) and p₇ + (0, 10⁻⁴) | 3 roles × 2 to 3 solves × 46 min, about 5 to 7 h on rpi5 | κ₉, and g₉ ≤ 4.36 (σ_eff ≤ 0.736). c₉ against c₇ + d₇κ₇ |
| 3 | route 3's cancellations, uniform in ν | generalize `_sym_tail` of `exact_sigma.py` (D_entire.md §6.2) to the first tuned order of harmonic ν, with ν symbolic in sympy | desk work and sympy, a few days (estimate); no long run | If the O(1/j) cancellations are identities in ν, write them as *checked symbolically* and go to the remainder bound. If they hold only for some ν, stop route 3 |
| 4 | route 1 against its literature | read Costin–Tanveer 2007 and math/0612063 in full, and record in LIT_REVIEW.md what they assume and prove, against the inner problem's form | reading, 1 to 2 days (estimate) | whether the inner problem is in their class, and if not, which hypothesis fails |
| 5 | route 2's limit operator | write identity H's G_ν in normalised units for general ν, and evaluate it at ν = 7 and 9 from stages 1 and 2 | unknown until stage 3 shows how G is computed | If a limit exists, bound it and its rate. If not, drop route 2 |
| 6 | Theorem A(M = 7) on B, with s₅ and s₇ enclosed | the A(M = 5) design (A_M2.md) scaled to M = 7 | weeks of runs and engineering (DINF_ANALYTIC.md §4.2) | only after a route to (★) has passed stages 3 to 5, and only if P5 has not made Lemma T unnecessary for Theorem A(∞) (route 4) |

Stages 0, 1 and 2 run in that order. Stage 1b can run at any time. Stages 3 and 4 are desk work beside them. Stage 5
waits on stages 1 to 3, and stage 6 comes last. The costs of stages 3 to 5 are estimates. Those of stages 0 to 2 come
from the timings in #965 and #974.

## 5. Proposed text (for the coordinator)

**PROOF_QUEUE.md**, the P6 row's Status cell, replacing "queued":

> scoping note written (2026-11-18, [notes/P6_scope.md](P6_scope.md)): (★) stays the open inequality, and no number
> contradicts it. The measured ratios κ₅ and κ₇ lie below (R)'s range (M ≥ 5), so M = 9 gives its first instance. The M
> = 7 error bar moves κ₇ by under 1%. Five routes are named (a strip-analytic Borel estimate, a limit operator,
> ν-uniform Lemma W cancellations, the amplitude-series bypass of P5, late-order family prefactors), and none has been
> checked against its source. M = 9 is blocked at level 5 (#974). Next: a level-5 approach-radius pilot at M = 9, then
> r₁^{[9]}(p₇) at the M = 7 zero, and route 3's symbolic check.

**proof/B_tail_M7.md** §5.2, replacing "(T1) and (T2) give (R) with κ = ρ_*² e^{2σ} (1 + O(Q e^{σ(M+2)} |f_{M+2}| /
Λ))." by:

> (T1) and (T2) give the geometric bound |δ_M| ≤ b_M := Λ C_f e^{σ(M+2)} ρ_*^{M+1} + Q (C_f e^{σ(M+2)} ρ_*^{M+1})², with
> b_{M+2}/b_M ≤ ρ_*² e^{2σ}. That is not (R), which bounds each increment by the previous one. With (T3) it gives
> |s_∞ − s₇| ≤ Σ_{M≥7} b_M directly, with Λ, C_f and Q needed explicitly.

**proof/B_tail_M7.md** §8.4, "a consistency statement (T3) that follows from (★) and (T1)" → "… that follows from (★),
(T1) and (H_∞)".

**proof/B_tail_M7.md** §9, the row of the M = 7 zero, append: "rtol 10⁻¹³ and 10⁻¹⁴ on x86 and aarch64 move it by at
most 2.3×10⁻⁵ in A and 6×10⁻⁶ in ω/ω∞ (#965, #975, #976, #977). Under one repeat rule the 10⁻¹³ and 10⁻¹⁴ zeros agree
within their noise estimates (#977), so the 10⁻¹² zero quoted here is low in A by 1.4–2.3×10⁻⁵. Four centre solves at ξ₁
= 0.15 give a noise estimate of 9.9×10⁻⁶ (#974). Radius independence is untested at 2×10⁻⁵. The ξ₁ = 0.14 zero's noise
estimate is 1.2×10⁻³ (#974), and ξ₁ = 0.16 gives no accepted zero (#964, #965)."

## 6. Discrepancies

1. B_tail_M7.md §5.2 derives (R) from (T1) and (T2), which are upper bounds and give only a geometric bound (§1.3).
2. B_tail_M7.md §8.4 says (T3) follows from (★) and (T1). That omits (H_∞), the hypothesis of §5.1 under which s_∞ is
   defined.
3. B_tail_M7.md §9 lists κ₅ = −0.069 and κ₇ = −0.049 as the numerical evidence for (R). Both lie outside (R)'s stated
   range M ≥ 5 (§2.2). That is not wrong, but the evidence is an extrapolation.
4. The quoted M = 7 zero, (2.71922, 1.000022) ± 2×10⁻⁵ at rtol 10⁻¹², is low in A by 1.4×10⁻⁵ to 2.3×10⁻⁵ against
   the rtol-converged zeros of #977, at or just beyond its own error bar. #975 and #977 already report this. It changes
   none of (★)'s inputs by more than 1% (§2.3).

## 7. Reproduce

From `solver/proof/`, with the harmonics of fit3_repeat in every line (B_tail_M7.md §10). Each prints the κ₇,
coupling, limit and tail of one row of §2.3.

```
X=0.148977988,0.014770713,0.008424206,0.001511189,0.000474205,0.00012168
python3 trunc_seq_M7.py --c7=1.000022,2.719219,$X          # rtol 1e-12 (stored): -0.0485, 0.847, L 0.999985, |r_1| 0.246
python3 trunc_seq_M7.py --c7=1.000025476,2.719240421,$X    # rtol 1e-13, win2 (#975): -0.0487, 0.851, L 0.999988, 0.248
python3 trunc_seq_M7.py --c7=1.000028,2.719236,$X          # rtol 1e-13, rpi5 (#965): -0.0488, 0.854, L 0.999991, 0.249
python3 trunc_seq_M7.py --c7=1.000026,2.719236,$X          # rtol 1e-13, rpi5 (#974): -0.0487, 0.852, L 0.999989, 0.248
python3 trunc_seq_M7.py --c7=1.000025865,2.719242554,$X    # rtol 1e-13, win2 (#977 run 2): -0.0487, 0.851, L 0.999989, 0.248
python3 trunc_seq_M7.py --c7=1.000022996,2.719232205,$X    # rtol 1e-14, win2 (#976): -0.0485, 0.848, L 0.999986, 0.246
python3 trunc_seq_M7.py --c7=1.000024742,2.719234224,$X    # rtol 1e-14, win2 (#976 + #977 run 1): -0.0486, 0.850, L 0.999988, 0.247
```

The remaining numbers derived here are arithmetic on quoted values, with the formula given where they appear: ½ ln g,
e^{2 × 0.736}, the A ratios, 0.0485 × 6.9, ¼ × 6.9, the Δσ₈ estimate, and 0.736/(2ω∞) = 0.096 rad (5.5°).

# Independent review of the M = 3 point certificate and Theorem A(3)

Task: an independent soundness review of the computer-assisted proof of Theorem A(M = 3) as assembled in
[A_lemmas_M.md](A_lemmas_M.md) §6–7, from the point certificate of [A_M.md](A_M.md) §6.2 and the lemmas of
A_lemmas_M.md, at commit 551113a. Reviewer: agent A3, which had not seen the work produced. Check scripts and their
logs: [`solver/proof/review_a3/`](../solver/proof/review_a3/). [A_review.md](A_review.md) (the M = 1 review) was read
as the template; nothing from it was taken on trust.

## Contents

- [Verdict](#verdict)
- [Summary of the checks](#summary-of-the-checks)
- [1. The point certificate](#1-the-point-certificate)
- [2. The model and the jump conditions](#2-the-model-and-the-jump-conditions)
- [3. Lemma 6a and the hypotheses H1–H3](#3-lemma-6a-and-the-hypotheses-h1h3)
- [4. Step (i) and the 6b₁ gluing check](#4-step-i-and-the-6b-gluing-check)
- [5. The mutation controls the notes cite](#5-the-mutation-controls-the-notes-cite)
- [6. Findings](#6-findings)
- [7. What I did not check](#7-what-i-did-not-check)
- [8. Reproduce](#8-reproduce)

## Verdict

**The point certificate is certified.** I found no error in any quantity that enters the Krawczyk test, and the test's
conclusion follows. What the code proves is the statement of A_M.md §6.2:

> Given Braaksma's theorem (for the identification of the start values with y⁺), the D-jump function
> g = (Re J₁, Im J₁, Re J₃, Im J₃) at ξ₁ = 0.3, with y⁺ continued from the ray arg ξ = 5π/16 (from |ξ| = 0.0741) to
> |ξ| = 0.2, along the arc |ξ| = 0.2 to the real axis and along the real axis to 0.3, has exactly one zero in
> P = p̂ + T·([−10⁻¹⁰, 10⁻¹⁰]² × [−4×10⁻¹⁶, 4×10⁻¹⁶]²), and its Jacobian is non-singular there.

An exact-rational re-assembly of the Krawczyk test from the saved run, sharing nothing with certify_M*.py, gives the
same verdict and margins (0.9944, 0.9970, 0.7777, 0.7775) and a zero enclosure inside the one quoted. Independent
references lie inside the new integrator's single-step enclosures at M = 3 on all three legs, and inside the
step-(i) start-value balls at the certificate's centre (to 2×10⁻⁶⁷ against radii of 5×10⁻⁶⁶).

**Theorem A(3) is certified given Braaksma's theorem, as A_lemmas_M.md §7 states, with one qualification about review
coverage.** The assembly in A_lemmas_M.md §6 is sound, and the hypotheses H1–H3 hold as exact inclusions (§3). I
reran lemma 6a on the certificate's own state box and reproduced ρ₆ₐ = 0.0413. Two links of the chain rest on code
that neither this review nor the M = 1 review has examined line by line:

- lemma S, lemma 6b₀ and the flat estimate (the majorant calculus of `fixed_point.py` with lemma6_sector's strict
  kernel bound);
- the 183 disc certificates of lemma 6b₁ (`lemma6b1_disc_M.py` and the Lohner runs behind them). I checked only the
  gluing check, its mutation, and that the box covers P.

A_certify.md §12.7 records the M = 1 versions of this code as not independently reviewed. I found nothing that
would make the theorem *conditional* or *not established*. By AGENTS.md's rule (new certificate code is reviewed
before its claim is relied on), those two links still need a review.

No finding below changes a number or a verdict. One proof-text gap needs a sentence (F3), and one check is weaker than
the notes say (F2). The rest are cosmetic.

## Summary of the checks

| check | how | verdict |
|---|---|---|
| every input of K(X) is an enclosure (no stray float, remainders bounded, JSON round trip) | reading certify_M.py, certify_M_lohner.py, certify_M_sys.py, validated_int.py, lemma6_common.py | OK |
| the Krawczyk operator, the box, the strict-interior test, the conclusion | reading; K1 | OK |
| the tilted box P = p̂ + T·X | reading; K1, K3 | OK (F5: T is a ball matrix) |
| K(X) ⊂ int X recomputed independently | K1: exact rational intervals, own preconditioner | same verdict, margins 0.9944, 0.9970, 0.7777, 0.7775 |
| K1's mutations (sign, conjugation dropped, zero planted outside, g widened) | K1 `--mutate=` | all fail, as they must |
| the new Lohner pieces at M = 3 (N = 20 with NJ = 12, growing Picard series, sparse linearization) | K2: one step on each leg vs mpmath, 60 digits, own right-hand side | contained, at both orders; remainder-removal controls fire |
| step-(i) start values at p̂ | K4: own series recursion + 75-digit integration | inside (2.1×10⁻⁶⁷ from the midpoints, radius 4.7×10⁻⁶⁶); control fires |
| start data over the box (same N, k, vanishing orders, Cauchy estimate) | reading fixed_point_M.start_data_M | OK; F2, F3 |
| the system is the inner_np.py model | reading both; selftest; K2 uses a right-hand side written from inner_np's docstring | OK |
| jump functionals and path | reading; K1 rebuilds ℓ from the layout | OK |
| H1, H2, H3 as inclusions | K3: exact rationals | hold (F4: the 6a ω-ball contains P's only by radius rounding) |
| lemma 6a code and its input B₁ | reading lemma6_jump_M.py; K3 recomputes B₁ ⊇ state set over P; rerun | OK; ρ₆ₐ reproduced |
| 6b₁ gluing check and its drop mutation | rerun | PASS / FAIL as stated |
| the notes' cheap mutation controls | rerun (§5) | all as stated |

## 1. The point certificate

### 1.1 What enters the Krawczyk test

The test (`certify_M.krawczyk_test`, certify_M.py:345–429) uses four things. Here is why each is an enclosure.

- **g̃(0) = g(p̂)**, as Yℓ(c + Qq) from the box run's centre-trajectory set (certify_M.py:369–374; `apply_set`,
  :204–210). That set starts at the step-(i) ball at p̂ exactly. The centre p̂ is the 256-bit binary rounding of the
  decimal strings (`exact_p`, :112–119), and the same function gives it in the run and in the test. `start_data`
  (:145–164) takes the values from `start_data_M` at p̂ and the derivative columns from the box call. The set is then
  propagated by the Lohner step, whose mean-value update needs q ∋ 0. That invariant holds: q starts as
  vals − mid(vals), w − mid(w), and is mapped by products with balls that contain 0.
- **Dg̃(X)**, as Yℓ(cV + QV qV) from the box run's tilted columns (:213–222, :375). They start as Σᵢ Tᵢₖ ∂y/∂pᵢ over the
  hull of P (:295–303), with exact jets plus Cauchy bounds. They evolve by v′ ∈ J cV + (J QV) qV, where J encloses the
  step Jacobian over the state set c + Qq + V·dx (validated_int.Lohner.box, the centre_set mode).
- **Y**, the exact midpoint matrix of a 768-bit inverse of mid Dg̃(X) (:370–372). Any real matrix is allowed.
- **X**, `arb(0, widths)` (:410–416). Its radius is the 30-bit mag rounded up from arb("1e-10"), that is
  1.0000000014×10⁻¹⁰. It is the same construction as the run's `dx` (:305), so Dg̃ was enclosed over exactly this X.
  The interior test `abs(K) < widths` (:417) is arb's certain comparison against the ball arb("1e-10"). So the zero lies
  strictly inside the nominal box, and it is unique in the run's slightly larger box. The quoted statement with nominal
  widths is correct.

**The Lohner step** is validated_int.Lohner's (reviewed in A_review.md §3), with three changes in
certify_M_lohner.LohnerM. I checked each.

- **NJ < N** (:147–155). J = Σ_{k≤NJ} V_k(X)h^k + h^{NJ+1}·Φ_{NJ+1}(T, Ỹ)·Ṽ. This holds for any order. The derivative
  satisfies Φ^{(k)}(τ, t₀)/k! = [Φ(·, τ)]_k Φ(τ, t₀), and the integral form of the remainder has a non-negative weight
  of total mass one. VR comes from the ball context (T, Ỹ) truncated to NJ + 2 coefficients. The truncation commutes
  with the series arithmetic, so the coefficients are the same as an order-(NJ + 1) computation.
- **Growing Picard series** (:40–64). After iteration k the coefficients 0..k + 1 are final, in ball arithmetic too:
  each is the same exact function of (t₀, Y₀) evaluated by interval extension.
- **`SysM.lin_unit`** (certify_M_sys.py:133–183) is self-tested against `lin` on every unit vector. `lin` is tested
  against central differences and against validated_int.Sys at M = 1. I reran the self-test: PASS. Its `coupling`
  mutation: FAIL.

The a-priori enclosures (lemma6_common.py:32–92) verify containment with `contains`; the inflation floor and MAXREL
are only step control. The legs (`LegM`, certify_M_sys.py:212–245) and their series match ξ(t) on the ray, the arc and
the real axis. Every t lies on 2⁻³⁰ℤ, and the arc ends exactly on the real axis, where exp(0) = 1.

**K2** (`review_a3/rv3_lohner_step.py`, log `rv3_lohner_step.log`) tests the M = 3 step directly. It takes one LohnerM
step on each leg (ray from the certificate's start value, arc and real leg from O(1) states), at the certificate's
orders (N, NJ) = (20, 12) and at (6, 4). The reference is mpmath.odefun at 60 digits, on a right-hand side and path
written from inner_np.py's docstring. Every state lies inside, 10⁻⁵⁰ to 10⁻¹² from the midpoints against radii
10⁻²⁸ to 10⁻⁸. Every finite-difference derivative column (a D direction, an S direction, and ω) lies inside. With the
trajectory remainder removed, the reference falls outside (ray, real). With the Jacobian remainder removed, a
reference column falls outside (ray, real).

**Serialization.** The saved sets are arb strings `[m +/- r]`, which contain the ball (the M = 1 review's R4 tested
this format on 20 000 balls), and Q matrices as float hex, checked exact before saving (`check_exact_float`). K1 reads
them as exact rational intervals.

### 1.2 The tilted box

The tilt C is read from the refinement JSON as 60-digit decimal strings, so T's lower-left entries are balls of radius
about 10⁻⁶¹ (certify_M.py:680, :803, :293). The start columns use these ball columns, the hull of P uses ball T, and
the Krawczyk test reads the run's columns directly. So every enclosure holds for each fixed matrix T* in those balls,
and the conclusion "exactly one zero in p̂ + T*X" holds for each such T*. The quoted zero enclosure p̂ + T·K(X) uses
the balls, so it contains p̂ + T*x* for all of them. This is sound, but "an explicit matrix T" (A_M.md:73) is explicit
only to 10⁻⁶¹ (F5).

The hull of P has half-widths 1.0×10⁻¹⁰, 1.0×10⁻¹⁰, 3.87×10⁻¹¹ and 2.51×10⁻¹² (K3), as the notes say. ω depends on
x₁ alone, because row 1 of T is e₁, and its column is W0·δₖ₁ (certify_M.py:160). A, B₃ and C₃ enter only the start
data.

### 1.3 The Krawczyk test and its conclusion

K(X) = −Y g̃(0) + (I − Y Dg̃(X)) X with X centred at 0, Dg̃(X) an interval matrix containing Dg̃(x) for every x ∈ X,
and Y a fixed real matrix. K(X) ⊂ int X implies that g̃ has exactly one zero in X and that every matrix in Dg̃(X), and
Y, is non-singular (Krawczyk 1969; Neumaier 1990, as cited in A_M.md §2). T is invertible (unit lower-triangular), so g has
exactly one zero in P and Dg is non-singular there. **Verdict: OK.**

**K1** (`review_a3/rv3_krawczyk_exact.py`) recomputes the test in exact rational interval arithmetic (Python
fractions), from the JSON only. It uses its own state layout, its own jump rows, and its own preconditioner, the exact
rational inverse of a 2⁻²⁰⁰-rounded midpoint matrix. On `data/A_M_box_M3_1e-10.json` it gives:

| | A | ω/ω∞ | B₃ | C₃ |
|---|---|---|---|---|
| relative margin (w − \|K\|)/w | 0.9944 | 0.9970 | 0.7777 | 0.7775 |
| zero, p̂ + T·K(X) | [2.72647588563032, 2.726475885631439] | [1.015603094857435, 1.015603094858032] | [0.1444194784699242, 0.1444194784702296] | [0.01108742588810796, 0.01108742588812546] |

The enclosure lies inside the one quoted in A_M.md §6.2, which the script checks. Mutations: `sign` (margins −7×10⁹,
−3.6×10⁹, …), `noconj`, `shift` (a zero planted 2h outside in A: −1.006) and `widen` all fail. On the h = 10⁻⁹ box the
inner rows fail (−87.7), and on the M = 1 box it passes (0.8629, 0.9759), both as the notes report.

## 2. The model and the jump conditions

**The system.** inner_np.py's docstring, and `Inner.rhs` (inner_np.py:104–117), are:

- ξ²D′ + iμωξD = [CS]_μ;
- ξ²S′ + iμωξS = [C ξD′]_μ + [(CE)D]_μ − [CD]_μ, with E = [DD]/2;
- ξC′ = [CE]_ν.

Here D and S live on the odd harmonics |μ| ≤ M, and C and E on the even harmonics |ν| ≤ M + 1. Each binary product is
projected, and (CE) is projected to the even set before it multiplies D. `SysM.rhs` (certify_M_sys.py:92–109) is the
same, term for term, grouping included. With λ = 1 the homotopy weights of inner_np are all 1. Three checks agree:

- the self-test against `Inner.rhs`, 4.4×10⁻¹⁶ (M = 3);
- `lemma6_usys_M.py`, all of (a)–(f);
- K2, whose reference right-hand side I wrote from the docstring. It agrees with LohnerM's SysM steps to 10⁻⁵⁰.

The profile F₁ = −iA/2 and F_ν = −iB_ν/2 + C_ν/2 is the same in inner_np.Fh_from, `CS.profile`, `FPM.fh_from` and K4.
ω∞ = 2π/1.64389375 everywhere. **Verdict: OK**; the theorem is about this truncation.

**The jump conditions.** J_ν = D_ν − conj(D_{−ν}) on the real coordinates: Re J = Re D_ν − Re D_{−ν},
Im J = Im D_ν + Im D_{−ν} (certify_M.py:177–201). This is inner_np's `D - conj(D[::-1])` and A_lemmas_M.md §4.1's
J_{D,μ}. The rows are ν = 1, 3, all (M + 1)/2 of them, as lemma 6a needs. The path's start ray 5π/16 and radius
0.0741 are inner_np's `r0_for` (the start check prints r0 = 0.0741). **Verdict: OK.**

## 3. Lemma 6a and the hypotheses H1–H3

**The code** (lemma6_jump_M.py) does what A_lemmas_M.md §4.2 says.

- It takes the hull of B₁ and σ(B₁), with σ as the index reflection plus conjugation (:62–82).
- It flows that hull backward along the real axis from 0.3 to 0.067 with identity columns in validated_int.Lohner, so
  B encloses DΦ over the hull.
- It changes variables exactly at ξ_a and splits G into 9 slow rows (4 p, 5 c) and 4 fast rows (v) against the 9
  u-columns (4 S, 5 C), so G_slow is square, (2M + 3) × (2M + 3).
- It computes ρ₆ₐ = Λ/(1 − Λ)·max_μ w_{v,μ}Σⱼ|H_μj|Z^{kⱼ−k}/wⱼ with the flat estimate's own Perron weights.

I re-derived the bound from b's weighted norm at z = Z and ε ≤ Λ/(1 − Λ)‖b‖; it is the formula in the code (:230–243).
σ maps real-axis solutions to solutions for real ω, which I checked on the equations. Only ω enters the backward flow;
A, B and C enter only the start data.

**The hypotheses** (K3, `review_a3/rv3_hypotheses.py`, exact rationals):

| | inclusion | margin |
|---|---|---|
| H1 | hull(P) ⊂ x̂ ± 10⁻⁶ (x̂ = `fixed_point_M.ZEROS[3]`) | hull within 4.7×10⁻¹⁰ of x̂ |
| H2 | hull(P) ⊂ x_b ± 1.1×10⁻⁹ (x_b recorded in `lemma6b1_runs_M3_cert/ray.json`; every row file carries h = 1.1e-9, and `row_run` asserts the ray's centre) | hull within 1.0×10⁻¹⁰ of x_b |
| H3a | ω-range of P ⊂ the 6a parameter ball | 1.08×10⁻¹⁹, from arb's radius rounding only (F4) |
| H3b | 6a's B₁ (`data/A_M_6a_M3_1e-10_in.json`) ⊇ the state set over P, which K3 recomputes from the box run's Lohner sets | holds in every component (state set radius 3.40×10⁻⁸, B₁ 3.42×10⁻⁸) |
| path | ray 5π/16 within 0.2 of 3π/8 (0.1963), r₀ = 0.0741 ≤ 0.1, arc 0.2, ξ₁ = 0.3 | |

`--mutate=shift` (the 6b₁ centre moved by 2×10⁻⁹) makes H2 fail.

**The rerun** of `lemma6_jump_M.py --M=3 --box=data/A_M_6a_M3_1e-10_in.json` (log
`review_a3/lemma6a_M3_cert_rerun.log`) gives RERUN_6A. The `partial` mutation on the certificate's own B₁, rather than the
pilot's (the notes' mutations used the pilot B₁, A_lemmas_M.md:801), gives PARTIAL_6A.

**The assembly** (A_lemmas_M.md §6) holds together.

- 6b₀ covers 0 < |ξ| ≤ 0.0624, and 6b₁ covers 0.062 ≤ |ξ| ≤ 0.2 by an open set. Together they cover Ω, so the path
  continuation equals y⁺ up to the end of the arc. On the real leg the continuation is along the real axis itself.
- 6a applies at every x ∈ P.
- The certificate gives one simple zero of J^path = J in P.
- The simplicity transfer uses DJ_D = 𝓛 ∘ DΣ. Given that factorization, a non-singular DJ_D already forces DΣ to be
  non-singular; the injectivity of 𝓛 is not needed for that step. I did not re-derive the factorization (A_certify.md
  §11.7).

One wording point: step 1 says the start point lies in the old sector lemma's domain "where the start value is y⁺".
The certificate's start value actually comes from Lemma FP(M) on the ray 5π/16 itself, which identifies it with y⁺
directly (by Braaksma). The conclusion is the same.

## 4. Step (i) and the 6b₁ gluing check

**The interface** `start_data_M` (fixed_point_M.py:439–504) uses the centre's N, k and vanishing orders for the real
box and for the Cauchy box, so η is one function throughout. The Cauchy estimate is applied to the holomorphic
η_q = η_v + (1 + s)η_p as a whole, with sup|1 + s| over the complex box. So the ω-dependence of s is accounted for.
The jets carry the profile and w = W0·x₁ exactly.

**The diag splitting** (fixed_point_M.py:161–194) uses the identity
(2 + s)Q + (1 + s)W = (3 + 2s)V + (2 + s)(1 + s)P + (1 + s)[(C − 1)Q], which I checked. The kernel is φ_μ(z)/φ_μ(t)
for φ_μ = e^{−1/(2z)}z^{−(3+2s)/2}, and the homogeneous solution grows like e^{a/|z|} toward 0 on the ray. See F3 for
the complex Cauchy box.

**K4** (`review_a3/rv3_start_value.py`) tests the start values. It uses a series recursion derived by hand from the
equations, in mpmath at 110 digits, truncated at its smallest term (10⁻¹¹⁶) at |ξ| = 0.05. It then integrates at 75
digits out to 0.0741 on the ray 5π/16. The result lies in every one of `start_data_M`'s 13 start-value balls at the
certificate's p̂, 2.1×10⁻⁶⁷ from the midpoints against radii up to 4.7×10⁻⁶⁶. With A moved by 10⁻⁴⁰ it falls outside
(6.1×10⁻⁴² from the midpoints). This is a planted-truth check, not a proof (the truncation error at 0.05 is estimated, not bounded). It
is independent of fixed_point_M.py, series_arb.py and certify_M*.py.

**Lemma 6b₁.** I reran only `lemma6b1_cover_M.py check` on the certificate's box: PASS, 183 tiles, 579 polar
rectangles, 461 links. With tiles (5, 6) dropped (`--drop=5,6`) it fails with exit 3 and three failures, as stated. I
also read how the box is set: `params`, `start_data_u` and `lohner_c1` put x_b ± h on every one of the M + 1
parameters (lemma6b1_cover_M.py:140–145, :223–313). The disc certificates themselves are not reviewed (§7).

## 5. The mutation controls the notes cite

Rerun by `review_a3/run_controls.sh` (log `run_controls.log`); every exit status is as stated.

| command | stated | got |
|---|---|---|
| `certify_M_sys.py selftest` / `--mutate=coupling` | PASS / FAIL | PASS / FAIL (rhs mismatch 3.8×10¹ at M = 1) |
| `certify_M.py krawczyk --box=…1e-10 --hier` / `--mutate=hier` | exit 0 / exit 1 | 0 / 1 |
| `certify_M.py controls --box=…1e-10` (C0–C8, C6b) | all as expected | all as expected, exit 0 |
| `certify_M_recheck.py …1e-10` / `--mutate` | pass / fail | 0 / 1 |
| `certify_M.py startcheck 3` / `--mutate` | contains / not | 0 / 1 |
| `certify_M.py crosscheck 3` / `--mutate` | AGREE / DISAGREE | 8.6×10⁻¹⁴ / 5.5×10⁻⁶ |
| `lemma6_usys_M.py` | all pass | all pass |
| `lemma6b1_cover_M.py check` / `--drop=5,6` | PASS / FAIL | PASS / FAIL (exit 3) |

Two of these are weaker than their names suggest. `startcheck` compares the step-(i) values with another truncation
of the same formal series at hier.py's zero, not at p̂. It does not test the η bound (K4 does). `crosscheck` compares
with inner_np at |ξ| = 0.2 on hier.py's path, without the real leg. The real leg is covered by K2 and by the agreement
of hier.py's zeros with the certified one.

## 6. Findings

None affects the certified statement or any quoted number.

- **F1 (cosmetic).** A_M.md:322–323 says the h = 10⁻¹⁰ run "was interrupted once by a restart and resumed from its
  checkpoint". `logs/A_M_box_M3_1e-10.log` shows one uninterrupted run: one header line, no "resumed from" line, and
  the timer rising monotonically from 154 s to 3357 s. The leg times (2061 + 1122 + 119 s) plus the start data account
  for the recorded 3392 s of one process. Either the sentence belongs to another run, or the log is incomplete. *Proposed:* drop the parenthesis, or point to the log that shows
  the resume. (A resume would be sound in any case: the checkpoint stores the Lohner sets as enclosing strings, and a
  ball-valued centre c is handled by the mean-value update.)
- **F2 (minor, a check weaker than stated).** fixed_point_M.py:279 checks `min(kmins) >= N - 1` over all rows. At the
  certificate's centre the orders are p, v: 95 = N and c: 94 = N − 1 (my run). So the p and v rows also drop their
  order-(N − 1) coefficient, on the evidence that the ball contains 0 at the centre. Over the box (`kmins_in`, :456 and
  :459) only containment of 0 is checked, which is nearly vacuous for wide balls. Those coefficients vanish identically,
  because the recursion solves the equations exactly through order N − 1; at M = 1 this pattern (N, N, N − 1) was proved
  exactly in GF(p) (A_review.md R1). At M ≥ 3 no code proves it. A_lemmas_M.md §2.1 lists "every coefficient … below
  its structural first nonzero order is a ball containing 0" as a hypothesis, which is exactly what the containment
  test does not establish over a box. *Proposed:* state the algebraic reason in §2.2, assert the per-row structural
  orders (p, v: N; c: N − 1), and add a GF(p) regression test as at M = 1.
- **F3 (minor, a proof-text gap).** A_lemmas_M.md:167 and fixed_point_M.py:23 say |(t/z)^{s_μ}| = 1 "for every label"
  on a ray. That holds for real ω. On the complex Cauchy box X + c(1 + i), whose sup|η| gives the certificate's
  derivative columns, ω is complex. There |(t/z)^{s_μ}| = (ϱ/r)^{−μ Im ω}, so the kernel bound
  |K_μ| ≤ e^{−a(1/ϱ−1/r)}(ϱ/r)^{3/2 − |μ Im ω|} ≤ e^{−a(1/ϱ−1/r)} needs |μ Im ω| ≤ 3/2. With c = 10⁻³ it is at most
  0.0115 at M = 3 and 0.019 at M = 5, so the conclusion stands with a wide margin. The code does not check it, and the
  text does not say it. The same applies to the old sector lemma's Cauchy box in 6b₁ (`start_data_u`, diag). The plain
  splitting is unaffected. *Proposed:* add that sentence to §2.2, and an assertion |μ|·W0·c ≤ 3/2 in `run_M` when
  diag is used with a complex box.
- **F4 (cosmetic).** The 6a input box is x ± 10⁻¹⁰ (`A_M_6a_M3_1e-10_in.json`), while P's outer radius is
  1.0000000014×10⁻¹⁰ (arb's mag rounding of 10⁻¹⁰). The ω-ball lemma6_jump_M.py actually builds,
  `arb(x, arb("1e-10"))` at 320 bits, has radius 1.0000000025×10⁻¹⁰, so it contains P's ω-range, by 1.1×10⁻¹⁹ (K3,
  and directly in arb). The inclusion is true, but by rounding, and lemma6_jump_M.py:153 does not check it: it checks
  only x ± h ⊂ x̂ ± 10⁻⁶. *Proposed:* export P's actual outer radius in `export6a`, and have lemma6_jump_M.py check that
  the ω-range of P lies in its ball (or run 6a with h = 1.1×10⁻¹⁰).
- **F5 (cosmetic).** "An explicit matrix T" (A_M.md:73, §6.2): T's tilt entries are 60-digit decimal balls of radius
  about 10⁻⁶¹. The proof holds for every T* in those balls (§1.2). *Proposed:* say so, or round C to binary before the
  run.
- **F6 (coverage).** Theorem A(3)'s trusted code includes M = 1 lemma code reused unchanged: lemma6_common.py,
  lemma6_sector.intbound_strict, lemma6b1_cover.check, and fixed_point.py's majorants beyond what R1, R2 and R7 checked.
  A_certify.md:1759 records the M = 1 lemma files as not independently reviewed, and A_review.md predates them. The
  notes' "trusted code" lists (A_lemmas_M.md §5.3, §7) should say this.
- **F7 (cosmetic).** In the A_lemmas_M.md §6 proof, step 1, the start value is attributed to the old sector lemma's
  domain; it comes from Lemma FP(M) on the ray 5π/16 (§3 above).

## 7. What I did not check

- **The box run itself.** I did not rerun it (55 minutes). Its rigour rests on the code reading above, on K2 (single
  steps), on the self-test, and on the agreement of the final enclosure with hier.py's and nested.py's zeros (to
  1.1×10⁻¹¹), which is consistent but not a proof.
- **Lemma S, lemma 6b₀ and the flat estimate.** Not the majorant computations (`lipschitz_M`, `intbound_strict`, the
  Perron weights beyond `wnorm`'s exact bound), and not the lemma-S domain geometry. I read only the diag identity and
  the kernel.
- **Lemma 6b₁.** Not the disc lemma (`lemma6b1_disc_M.certify`), not the C¹ row runs, not the planted-truth `c1check`
  (≈5 min), and not the geometry inside `lemma6b1_cover.check`. Only the check's verdict, its drop mutation, and the
  box inclusion.
- **Step (i).** Not the majorant calculus or the Lipschitz matrix of `fixed_point_M.run_M`, beyond K4 at the centre
  (the box and Cauchy-box bounds are not tested by K4) and the reading in §4.
- **The simplicity transfer** (A_certify.md §11.7, A_lemmas_M.md §4.2) and Braaksma's theorem's applicability on the
  ray 5π/16 (Lemma B4(M) of C_selection.md). These are mathematics I did not re-derive.
- **The remaining mutations of `mutations_M.py`** (fp_*, S_*, jump_swap, jump_coords, cover_sign, c1_*), and the
  M = 5 runs.

## 8. Reproduce

From `solver/proof/review_a3/` (times on the shared machine at load 1–2):

```
python3 rv3_krawczyk_exact.py ../data/A_M_box_M3_1e-10.json                 # K1, < 1 s; exit 0
python3 rv3_krawczyk_exact.py ../data/A_M_box_M3_1e-10.json --mutate=sign   # also noconj, shift, widen: exit 1
python3 rv3_hypotheses.py                                                   # K3, < 1 s; exit 0
python3 rv3_hypotheses.py --mutate=shift                                    # H2 fails; exit 1
python3 rv3_lohner_step.py                                                  # K2, ~5 min; exit 0
python3 rv3_start_value.py                                                  # K4, ~8 min; exit 0
sh run_controls.sh                                                          # section 5, ~6 min -> run_controls.log
```

From `solver/proof/`, the 6a rerun and its mutation (~18 min, then seconds):

```
python3 lemma6_jump_M.py --M=3 --box=data/A_M_6a_M3_1e-10_in.json --json=review_a3/lemma6a_M3_cert_rerun.json --save-B=B.json
python3 lemma6_jump_M.py --M=3 --box=data/A_M_6a_M3_1e-10_in.json --load-B=B.json --mutate=partial   # exit 0 = caught
```

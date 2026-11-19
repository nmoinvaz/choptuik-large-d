# P2's remaining items: a review plan, and Lemma J's θ₀ checked independently

Issue #988, queue item P2 of [PROOF_QUEUE.md](PROOF_QUEUE.md). P2 ([proof/P2_review.md](../proof/P2_review.md)) and
P2b ([proof/P2b_review.md](../proof/P2b_review.md)) left six certificate items without an independent check
(P2_review.md §5, P2b_review.md §8). This note plans a check of each (§1) and does the smallest (§2). Written
2026-11-19 by Claude (Anthropic) under Nathan Moinvaziri's direction. The reviewer, a helper session, had not worked
on Theorem B's certificates or read P1's transcripts. It read P2_review.md and P2b_review.md first. Not reviewed.
Status words as in [AGENTS.md](../AGENTS.md).

## 1. The six items

Sizes are line counts of the certificate scripts at main b081fbb2. Costs are estimates.

| # | Item | What it certifies, and where | The link that rests on it | Code | What an independent check needs | Cost |
|---|---|---|---|---|---|---|
| 1 | Lemma J's θ₀ | The a priori step of Lemma J. On (1, 1 + δ/2], with m = 1, the affine map with the endpoint value v_S(1 + δ/2) contracts (θ₀ = 5.8×10⁻⁵), so Ψ_out is a Neumann sum there and N(Ψ_out) < ∞. `b8m_lcj.py` (one block near its end), B_chain.md §14, last paragraph | (H-J), and through it Theorem B(1) (B_chain.md §15). Without it, T₀'s contraction does not reach Ψ_out | about 35 lines, on top of Lemma J's secant coefficients | the Γ⁰ entries re-derived from the statement, evaluated on secant coefficients that share no code with `b8m_lcj.py`. P2b's are such coefficients (P2b_review.md §4) | hours (done here, §2) |
| 2 | Lemma 6b₁'s discs, M = 1 | y⁺ analytic on the annular sector 0.0624 ≤ \|ξ\| ≤ 0.2, by 178 validated Taylor discs and their gluing. `lemma6b1_disc.py`, `lemma6b1_cover.py`, `lemma6b1_estimate.py`, A_certify.md §12 | Theorem A(M = 1): lemma 6b, that the continued y⁺ is the Borel sum. Everything in Theorem B(1) starts from Theorem A's box | 215 + 478 + 140 | (a) the covering's geometry and gluing in exact rationals, which is cheap; (b) a second disc certificate in mpmath intervals (Taylor coefficients and a remainder bound for the ODE on each disc), for every disc or a stated sample. A rerun on Windows (#889) reproduced the rows, but a rerun is not an independent check | (a) hours, (b) 2–4 days |
| 3 | (H8) | B8's continuation turned into B5(1)'s data on the inner arc \|x\| = n^γ, δ = (8.2×10⁻⁵, 79.03, 3.231, 6.463), and 122.31 for δ_Φ at d₀ = 10⁷. `b8m_h8.py`, B8_constants.md §8.1–8.3 | B5(1), so B5(2)'s C_E and B7, for n ≥ e^{73.89} (C_selection.md §13.1) | 339 | a reread of §8.2's three pieces by hand, then the conversion in mpmath intervals from B8's printed h. The input h is item 4's, so the check would share it; that is recorded, not avoided | 1–2 days |
| 4 | B8 beyond `--d0` | Lemma B8(1): the bootstrap P(h) < h on 𝒟′_n (K = 4, κ = 0.3, κ₁ = 0.6), closing for ln n₀ ≥ 62 and with the normalization offset for ln n₀ ≥ 73.89. `b8_alg.py`, `b8m_bounds.py`, `b8m_scan.py`, B8_constants.md §§3–7 | B8(1), (H8), B5(1) and the gauge range of Lemma LC (C_selection.md §13.1) | 227 + 402 + 84 | a second bootstrap in other arithmetic. P2b's LC majorant covers B8's estimates on the same bands in the light-cone gauge (P2b_review.md §8), and P2 reproduced the formal series and atoms (P2_review.md §5). What remains is B8's own closure with c(0) = c^K(0) + d. P2b's bound algebra could serve, with the sharing recorded | 3–5 days |
| 5 | B5(1) | The inner fixed point W = O(1/n) on the matching domain (κ = 0.3, Z = 0.0085), with the super-solutions x_Φ = (39784, 38776) that enter C_E. `b5_bounds.py`, `b5_formulas.py`, B5_inner.md §§3–5 | B5(1), and so B5(2) and B7. x_Φ is a factor of C_E, the dominant constant of n₀ (notes/P7_scope.md §4.1) | 692 + 288 | a reread of B5_inner.md §4's proof and a second super-solution certificate (b + Lx < x) in mpmath intervals. Its input is lemma S's enclosure over 𝒦 (A_certify.md §4.5), which would be shared | 4–7 days |
| 6 | The M = 3 items | Lemma S(3) and lemma 6b₀(3), which share the majorant code of `lemma6_sector_M.py` and the helpers of `fixed_point.py`, and the 183 discs of lemma 6b₁(3), in `lemma6b1_disc_M.py`, `lemma6b1_cover_M.py` and `lemma6b1_estimate_M.py` (A_lemmas_M.md §§3, 5). A_M_review.md checked the gluing, not the discs | Theorem A(M = 3). It is not on Theorem B(1)'s path. It anchors the truncation sequence that P6 uses | 189 + 520 (helpers), 141 + 598 + 207 | a second majorant implementation for lemma S (the start-value lemma's second implementation, #427, does not cover it), and the disc work of item 2 at M = 3 | 3–5 days |

**Order, by cost against the weight of the link.**
1. Item 1. It is the cheapest, and it is on Theorem B(1)'s path (done in §2).
2. Item 2(a), the covering's geometry. It costs hours and sits under the whole chain. Then item 2(b).
3. Item 3, (H8). It is moderate, and it sets the n ≥ e^{73.89} links.
4. Item 5, B5(1). It is the largest, and it carries the dominant constant.
5. Item 4, B8. It is large, and P2b's light-cone majorant already covers part of it.
6. Item 6. It is large and off Theorem B(1)'s path.

## 2. Item 1: Lemma J's θ₀, checked independently

### 2.1 Protocol, fixed before the run

- **Independence.** `b8m_lcj.py` is not opened before the numbers are committed. The Γ⁰ entries are derived here from
  B_chain.md §14's equations. The secant coefficients come from P2b's code (`review_p2b/p2b_lc.py` and
  `p2b_lcj.py`), which shares no code with the certificate (P2b_review.md §1). They are evaluated on P2b's L cell
  (1, 1.025] with every shape set to 1. One fact about the certificate was seen in a search before this protocol
  was written: a comment of `b8m_lcj.py` names the a priori norm "m = 1 on (1, 1 + delta/2], s_V = s_c = 1".
- **The derivation** (the docstring of `solver/proof/review_p2c/p2c_theta0.py`). Plain sup norms on I = (1, 1 + δ/2],
  with m = 1 and unit shapes.
  - S, backward, with B_chain.md §14's kernel bound: Γ⁰_{S,j} = (1 + h_H)/(1 − h_H) sup_I ℓ_{G,j}.
  - V: Γ⁰_{V,S} = |ω| ln(1 + δ/2).
  - c: Γ⁰_{c,j} = ln(1 + δ/2) sup_I ℓ_{F_c,j}.
  - β: Γ⁰_{β,j} = sup_I ℓ_{ĝ,j}/(n − 1).
  - The pseudo-component of δD/(x − 1) is added to ℓ_G and ℓ_ĝ, as P2b adds it.
- **Pass criteria.** θ₀ < 1 on the certificate's class constants (`--h=cert`) and on P2b's own (`--h=p2b_lc_cells.json`).
  The certificate's θ₀ = 5.8×10⁻⁵ is compared, not required. P2_review.md §3.3's θ₀ ≤ 1.8×10⁻⁵, from the saved Γ₀,
  is compared too. A difference larger than a factor 10 is located after the commit, by reading the certificate's
  block.
- **Controls.** `--planted`: the Perron certification on a rank-one matrix of known root, and the Γ⁰ assembly against
  hand values, must pass. `--planted --mutate=perron` must fail. `--mutate=nodecay`, the backward kernel times n₀
  (the certificate's own mutation in B_chain.md §14), must give θ₀ > 1 and exit 1.

### 2.2 Results

Not yet run.

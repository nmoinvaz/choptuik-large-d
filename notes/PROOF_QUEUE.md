# The queue of work toward a proof of the selection rule

The work that [PROOF.md](../PROOF.md) and [proof/C_selection.md](../proof/C_selection.md) §13 list as missing, put in
order and assigned. Status words are those of AGENTS.md. At most two cloud agents run at a time (one is on the low-D
push toward D = 3), so the cloud items run one after another; the local machine takes the long numerical runs through
comms.txt. The coordinator keeps this file current: when an item starts, finishes or changes.

| # | Item | Why it is needed | Where | Status |
|---|---|---|---|---|
| P1 | Close Theorem B(1), the selection theorem for the one-harmonic model | the whole chain from the inner zero to a solution exists for M = 1 except the links in C_selection.md §13.1 | cloud agent | **running** (from 2026-10-23) |
| P2 | Independent review and second implementations of the certificate code | no certificate of B5, B8 or (H8) has been reviewed; lemma 6b₁'s disc covering has not been re-implemented; at M = 3 the majorant code of lemmas S and 6b₀ and the 6b₁ discs are not reviewed | cloud agent that has not seen P1 | queued (next cloud slot) |
| P3 | Theorem A(M = 5) | the five-harmonic zero, the first truncation close to the real answer (0.999228); the point certificate fails with first-order enclosures | cloud agent | queued |
| P4 | σ₁₅ at ω/ω∞ = 0.99, 1.00, 1.01 | numerical evidence: tests the ring of fifteen zeros that σ₁₄ left open, and narrows the estimator range | local machine (T44, after T43) | queued |
| P5 | A rigorous type bound for S₁ (the entire-function route) | turns the amplitude series into a conditional Theorem A(∞): the zero certified assuming an explicit type bound ([proof/D_entire.md](../proof/D_entire.md)) | cloud agent | queued |
| P6 | Uniformity in the number of harmonics (gap (b)) | the step that needs a new idea: the Stokes weights e^{π\|ν\|ω/2} grow faster than the harmonics decay ([proof/B_tail.md](../proof/B_tail.md)) | cloud agent: a scoping note first | queued |
| P7 | Corollary C: the computed branch is the branch of the theorem | needs a certified finite-D solution and a threshold n₀ that a computable D can reach (Theorem B(1)'s certified threshold is n ≥ e^{62}–e^{74}, estimated ln n ≳ 105–285) | cloud agent: a scoping note first | queued |

## P1: Theorem B(1)

The missing links of C_selection.md §13.1, in this order (the computational ones first):

1. **B5(2)'s remainder.** Replace the estimate of `solver/proof/b5_threshold.py` by a certified bound on R and its
   derivative, and run B8 and (H8) at κ₁ = 0.85, where B5(1)'s own condition is ln n ≥ 104.68, so that the chain's
   threshold drops from the estimated ln n ≳ 285 to about 105.
2. **B8's normalization:** run the extension to c(0) = c^K(0) + O(n⁻⁵) that B8_constants.md §8.4 proves by hand.
3. **The 1/n rate:** part (1) with the S^{(1)} refinement of Section 10.3.
4. **B6:** write the block-diagonalization on [t₀, 1 − δ] and evaluate B6's constants for M = 1.
5. **The transport of the jump along x > 1:** settle whether conclusions (2) and (3) of Theorem B need it; if they do,
   write it and bound it.
6. **B7:** the implicit function theorem with explicit constants, giving Theorem B(1) with an explicit n₀.

Progress (wip/theoremB1):
- 2026-10-23, item 1 done (785c895): B5(2)'s remainder is certified for M = 1 (new lemma L, `b5_label.py`;
  `b5_jump.py`), |E| ≤ e^{76.91}/n for every n ≥ e^{73.89}. The evaluation point need not move with n, so the
  certified B5(1), B8 and (H8) already cover it (the κ₁ = 0.85 reruns confirm; this also gives the O(1/n) rate, item 3).
  `b5_threshold.py`'s error model used the wrong exponent (⟨F²⟩ = 3.23 instead of 1.73); the true error is about 10⁻⁵ of
  its estimate. A link the notes did not list: B7 needs the Jacobian of the multiplier itself, while Theorem A certifies
  the Jacobian of the D-jump conditions at ξ = 0.3; the forward propagation between them must be certified. Uncertified
  estimate of the final threshold: ln n₀ ≈ 88–90.

Done when C_selection.md §13.1 lists no missing link (Braaksma's theorem stays assumed) and the new certificate code
has mutation controls. The independent review is P2's.

## P2: review

A fresh agent, given the code and the statements but not P1's notes on how they were reached, re-derives or
re-implements: `b5_bounds.py` and `b5_formulas.py` (B5(1)); `b8_alg.py` and `b8m_*.py` (B8 and (H8)); lemma 6b₁'s
disc covering at M = 1 and M = 3; lemmas S and 6b₀'s majorant code at M = 3; and P1's new certificates once they
exist. Shared machinery between the two implementations is recorded (AGENTS.md: independent checks are independent).

## P3: Theorem A(M = 5)

proof/A_M.md §7: the next step is a design change, a second-order chart with C² enclosures, so that the inner widths
stay small; then lemma 6b₁ at M = 5 and the lemmas that turn a point certificate into the theorem (S, 6a, 6b₀),
already generalized to M harmonics.

## P4: σ₁₅

The T38 recipe with K 31 (σ₀…σ₁₅): a pilot at wf 1.00 with J 1000 and J 1250 side by side, the gate err15 ≤ 10⁻¹⁰
|σ₁₅| and σ₁₄ unchanged within its error, then wf 0.99 and 1.01. Then the K = 15 analysis of
notes/PLANTED_SELECTION.md §6 (its procedure is fixed in §6.0 and reruns with K = 15).

## P5, P6, P7: scoping notes first

Each starts with a short note before any long computation: the precise statement needed, the closest published
tools, what is computable now, and an estimate of the work. P7's note must face the threshold: Theorem B(1) is for the
one-harmonic model, and its n₀ is far beyond any D a solver reaches, so identifying the computed branch needs either a
much smaller certified n₀ or a different route.

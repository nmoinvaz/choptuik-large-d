# The queue of work toward a proof of the selection rule

The work that [PROOF.md](../PROOF.md) and [proof/C_selection.md](../proof/C_selection.md) §13 list as missing, put in
order and assigned. Status words are those of AGENTS.md. At most two cloud agents run at a time (one is on the low-D
push toward D = 3), so the cloud items run one after another; the local machine takes the long numerical runs through
comms.txt. The coordinator keeps this file current: when an item starts, finishes or changes.

| # | Item | Why it is needed | Where | Status |
|---|---|---|---|---|
| P1 | Close Theorem B(1), the selection theorem for the one-harmonic model | the whole chain from the inner zero to a solution exists for M = 1 except the links in C_selection.md §13.1 | cloud agent | **done** (2026-10-23; merged cba0f62 and 7d25496): with P1L, Theorem B(1) is certified for M = 1 for n ≥ e^{89.75}, assuming Braaksma's theorem; not yet reviewed (P2) |
| P1L | The light cone: gaps (a)–(c) of Theorem B(1) | P1 found that the remaining links rest on three open problems at x = 1: (a) control of the nonlinear solution in the half-discs near the light cone, (b) the exact light-cone gauge c(1) = 1, (c) a proof that the outer jump is fast-type; without them Theorem B(1) is conditional | the P1 agent | **done** (2026-10-23, merged 7d25496): Lemma LC (B8 into the light-cone half-disc in the exact gauge c(1) = 1, a contraction, θ = 0.7355, gauge offset ≤ 3.59×10⁶n⁻⁵) gives (a) and (b); Lemma J (θ = 0.176) gives (c); proof/B_chain.md §§13–15 |
| P2 | Independent review and second implementations of the certificate code | no certificate of B5, B8 or (H8) has been reviewed, nor P1's lemma L, B5(2), Lemma P, B7, Lemma LC and Lemma J (B_chain.md §10 lists the code and the hand arguments); lemma 6b₁'s disc covering has not been re-implemented; at M = 3 the majorant code of lemmas S and 6b₀ and the 6b₁ discs are not reviewed | cloud agent that has not seen P1 | **stopped by the user** (2026-10-23 17:28) after a first full draft, merged: Theorem B(1) stands as stated, with corrections F1–F7 that change no quoted constant or threshold (proof/P2_review.md §6; not yet applied to the notes and code); not done: second implementations of Lemma LC's closure bound algebra and of Lemma J's secant coefficients, and the older B5(1)/(H8) code and the M = 3 items |
| P3 | Theorem A(M = 5) | the five-harmonic zero, the first truncation close to the real answer (0.999228); the point certificate fails with first-order enclosures | cloud agent | scoping note written (2026-10-28, [proof/P3_scope.md](../proof/P3_scope.md)): the derivative radius grows 10⁸ on the real leg (fast directions), the harmonic-5 rows fail by 542 and the outer rows by 14; three designs. The M = 5 point run is done (T90, 2026-10-28, P3_scope.md §4): its derivative radius reaches 12 at ξ₁ against the box run's 3×10⁸, with the same growth along the path (10¹⁹) and a constant factor 10⁷ from the box, so the amplification is the enclosure's own and 12 already exceeds what Krawczyk tolerates; decision: design 2 (fast-direction-aware sets, cloud agent, two weeks, judged by the point run) first, design 3 beside it, design 1 only if the box runs then still fail with h |
| P4 | σ₁₅ at ω/ω∞ = 0.99, 1.00, 1.01 | numerical evidence: tests the ring of fifteen zeros that σ₁₄ left open, and narrows the estimator range | local machine (T44, after T43) | **done** (2026-10-25): σ₁₅ converged to 5×10⁻¹³ (T44, the Mac); the K = 15 analysis (notes/PLANTED_SELECTION.md §7) rejects every ring with zeros nearer than x₀, one ring with zeros just beyond x₀ survives (its zero inside the range), and the headline over K = 14, 15 is ω*/ω∞ = 0.999975–1.00005 (Δ∞ = 1.64387 ± 0.00006) |
| P8 | σ₁₆ at ω/ω∞ = 0.99, 1.00, 1.01 | numerical evidence: tests the one ring of zeros that survived σ₁₅ (zeros at 7.3 < \|x\| < 12, notes/PLANTED_SELECTION.md §7.9) and narrows the range 0.999975–1.00005 | the Mac (T83) | done (2026-10-27, §8): that ring is rejected by σ₁₆, two rings built from σ₀…σ₁₅ survive in turn, the K = 16 range is 0.999980–1.000014, the headline unchanged |
| P9 | σ₁₇ and σ₁₈ at ω/ω∞ = 0.99, 1.00, 1.01 | numerical evidence: tests the two rings that survived σ₁₆ (notes/PLANTED_SELECTION.md §8.4) and adds the K = 17, 18 ranges; the D-bracket question (§8.9) is fixed in §9.0 item 4 before the data (§5.3's rule as the primary reading, a second reading without D-bracket beside it) | the Mac (T86, T87) | σ₁₇ done (2026-10-28, §9.1–9.9): both rings rejected, no ring survives, K = 17 hull 0.999990–1.000009, headline unchanged. σ₁₈ done (2026-10-28, §10.1–10.9; §10.0 and its dry run were committed before the data): the two rings of the K = 17 generation rejected under both readings, no ring survives, K = 18 hull 0.999992–1.000004 inside the K = 17 range, headline unchanged, G below 1 for the first time (0.9999987), the projection favours ω* = 1 by 0.4×10⁻⁶ against a model error of 1.1×10⁻⁵ (not decisive); σ₁₉ (T88) tests C18-BP [3/15], [4/14], [5/13]: the nodes land 2026-10-28 about 23:00 UTC, but pilot (b) failed §11.0's agreement gate by 1.21 at σ₁₉ (20:55 UTC; a 1.3×10⁻¹¹ difference, nothing to the estimators), so the K = 19 stages wait for T91's attribution (J or bits, about 03:00 UTC 2026-10-29) and a dated amendment of item 1 |
| P5 | A rigorous type bound for S₁ (the entire-function route) | turns the amplitude series into a conditional Theorem A(∞): the zero certified assuming an explicit type bound ([proof/D_entire.md](../proof/D_entire.md)) | cloud agent | scoping note written (2026-10-28, [proof/P5_scope.md](../proof/P5_scope.md)): three named hypotheses (coefficients, type bound, ω-uniformity); at K = 17 the sharpened G-form tail is 2.3×10⁻⁶ against a Krawczyk need of 1.7×10⁻⁵ at radius 10⁻⁴, so the conditional certificate is step 1 (2–4 days); the bound calibrated at K = 12 holds to k = 17 at 0.91–0.92 of it, and σ₁₈ (T87 pilot) at 0.90 of the K = 17 bound. The fixed-ω certificate is done (`solver/proof/entire_cert_rig.py`, planted test and four mutations): at K = 17 and again at K = 18 one zero in \|x − x₀\| < 10⁻³ at each node, conditional on the coefficients and the type bound ((δ + tail)/min 0.13 and 0.11); the committed σ₁₈ is 0.90 of the K = 17 bound, σ₁₉ at ω∞ (T88 pilot (a), 2026-10-28) 0.90 of the K = 18 bound, and the k ≤ 12 calibration's ratio is 0.950 at k = 19 (0.934 at 18), rising 0.016 this order; the K = 19 recalibration predicts |σ₂₀| ≤ 2.06×10⁻¹⁷; the step across ω (H_ω) is next |
| P6 | Uniformity in the number of harmonics (gap (b)) | the step that needs a new idea: the Stokes weights e^{π\|ν\|ω/2} grow faster than the harmonics decay ([proof/B_tail.md](../proof/B_tail.md)) | cloud agent: a scoping note first | queued |
| P7 | Corollary C: the computed branch is the branch of the theorem | needs a certified finite-D solution and a threshold n₀ that a computable D can reach (Theorem B(1)'s threshold is n ≥ e^{89.75}, far beyond any D a solver reaches) | cloud agent: a scoping note first | queued |

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
- Item 6 done (bc236d1): Lemma P (a certified adjoint from the multiplier's Jacobian to Theorem A's) and B7 (the implicit
  function theorem by a Krawczyk test and a degree argument): Theorem B(1) for n ≥ e^{89.75}, conditional on (a)–(c)
  below, now stated as hypotheses (H-LC), (H-G), (H-J) in C_selection.md §13.1. Merged into main (cba0f62).
- Items 2–3 done (4c7c9d8): B8 and (H8) hold for |c(0) − c^K(0)| ≤ 10⁷ n⁻⁵ (`b8m_bounds.py --d0`); the rate is
  O(1/n) directly. Items 4–5 are harder than the notes said: three open gaps, (a) control of the nonlinear solution
  near the light cone x = 1 (B6′ is linear only), (b) the exact light-cone gauge, which needs (a), (c) a proof that
  the outer jump is fast-type (C_selection 10.3 asserted it), which also rests on (a). Now: item 6 (B7 and the
  certified link from the multiplier's Jacobian to Theorem A's), giving Theorem B(1) conditional on (a)–(c) with an
  explicit n₀; then (a)–(c) written up as the open hypotheses.

Done when C_selection.md §13.1 lists no missing link (Braaksma's theorem stays assumed) and the new certificate code
has mutation controls. The independent review is P2's.

Outcome (2026-10-23): done together with P1L. Theorem B(1) is certified for M = 1 (B_chain.md §15): for every
n ≥ e^{89.75} exactly one p_n in Theorem A's box gives a light-cone-gauge, centre-regular solution analytic at x = 1,
with |p_n − p*|_∞ ≤ e^{67.84}/n. It assumes Braaksma's theorem (through Theorem A and lemma S), and "the" solution is
unique within the class 𝒦_h of Lemma LC, a neighbourhood of the formal series. Items 4–5 (B6, the transport) turned out
not to be needed. Seven discrepancies with the earlier notes are recorded in B_chain.md §9.

## P1L: the light cone

In order: (a), then (b), then (c), since both later ones rest on (a). The light cone is a regular singular point of
the self-similar equations, so the natural tools are those for analytic nonlinear systems at a regular singular point
(Briot–Bouquet type theorems: an analytic solution exists and is unique under non-resonance, B6's gauge already
supplies the non-resonance), made explicit with majorant series for the Frobenius expansion at x = 1 and then validated
integration out to B8's domain, as B8 does elsewhere; Reiterer and Trubowitz's existence proof at D = 4 treats the same
light cone and is the closest precedent; Olver's error bounds at singularities cover the linear part. First a written
plan with the exact statements (P1's write-up of (a)–(c) in C_selection.md §13.1), then the certificates, with mutation
controls. Done when Theorem B(1) is unconditional apart from Braaksma's theorem.

Outcome (2026-10-23, cdc9aa4 and d1354fd): (a) and (b) as one lemma, Lemma LC (`b8m_lc.py`): B8 continued into the
light-cone half-disc in the exact gauge c(1) = 1, every D⁻¹ near x = 1 rewritten in bounded form, constants per radial
band; closure for n ≥ e^{73.89}, contraction θ = 0.7355 from a 40×40 Lipschitz matrix; gauge offset
|c(0) − c^K(0)|₁ ≤ 3.59×10⁶n⁻⁵, inside B8's certified 10⁷ range. (c) is Lemma J (`b8m_lcj.py`): a flat estimate on the
Stokes segment in a norm weighted by half the fast exponent, θ = 0.176. Mutation controls fire for every new
certificate; a numerical check at n = 10³ (`b8m_lc_numcheck.py`) passes. In the exact gauge the light cone helps: the
S-mode is attracting toward x = 1. Briot–Bouquet theory was not needed.

## P2: review

A fresh agent, given the code and the statements but not P1's notes on how they were reached, re-derives or
re-implements: `b5_bounds.py` and `b5_formulas.py` (B5(1)); `b8_alg.py` and `b8m_*.py` (B8 and (H8)); lemma 6b₁'s
disc covering at M = 1 and M = 3; lemmas S and 6b₀'s majorant code at M = 3; and P1's new certificates once they
exist: `b5_label.py`, `b5_jump.py`, `b8m_bounds.py --d0`, `b7_adjoint.py`, `b7_ift.py`, `b8m_lc.py`, `b8m_lcj.py`, and the
hand arguments of B_chain.md §10 (most of all, in Lemma LC, that the majorant's derivative bounds the Lipschitz
constant, and the double-sector symmetry). Shared machinery between the two implementations is recorded (AGENTS.md:
independent checks are independent).

## P3: Theorem A(M = 5)

proof/A_M.md §7: the next step is a design change, a second-order chart with C² enclosures, so that the inner widths
stay small; then lemma 6b₁ at M = 5 and the lemmas that turn a point certificate into the theorem (S, 6a, 6b₀),
already generalized to M harmonics.

## P4: σ₁₅

The T38 recipe with K 31 (σ₀…σ₁₅): a pilot at wf 1.00 with J 1000 and J 1250 side by side, the gate err15 ≤ 10⁻¹⁰
|σ₁₅| and σ₁₄ unchanged within its error, then wf 0.99 and 1.01. Then the K = 15 analysis of
notes/PLANTED_SELECTION.md §6 (its procedure is fixed in §6.0 and reruns with K = 15).

Done: the procedure, with the three changes §6's findings called for, was fixed in §7.0 before any K = 15 estimate, and
the results are §7.1–7.9.

## P5, P6, P7: scoping notes first

Each starts with a short note before any long computation: the precise statement needed, the closest published
tools, what is computable now, and an estimate of the work. P7's note must face the threshold: Theorem B(1) is for the
one-harmonic model, and its n₀ is far beyond any D a solver reaches, so identifying the computed branch needs either a
much smaller certified n₀ or a different route.

# Layer B documentation pass (2026-10-22)

Applies the proposed text of [proof/B5_inner.md](../proof/B5_inner.md) §9, [proof/B3_borel.md](../proof/B3_borel.md) §9
and [proof/B8_constants.md](../proof/B8_constants.md) §12, with two corrections throughout: the M-harmonic model keeps b
and c on the even harmonics |ν| ≤ M + 1, and hypothesis (H-σ) replaces H-a1/H-a2. Branch `wip/layer-b-docs`.

## Changes

**proof/C_selection.md**
- Header, §1: b and c on |ν| ≤ M + 1 (B5_inner.md §9 text); real-unknown count corrected to 8N + 2 (the old
  4N + 2(N − 1) + 2 matched neither truncation); a remark that B5(1) and B8 project every binary product, which agrees
  with projecting each right-hand side at M = 1 and not for M ≥ 3 (undecided, from B8_constants.md §11).
- Summary: the outer-problem bullet and the "genuinely open" bullet rewritten; the step table's status column updated.
- §2: Braaksma's hypotheses now "checked against the paper".
- §4.1: the inner system's c harmonics |ν| ≤ M + 1; the Braaksma parameter paragraph replaced (B3_borel.md §9, Lemma B4.P).
- §4.2: heading; "precise citation for the mixed case" replaced (B3 text); the H-a1/H-a2 block and the "if (H-a2)
  failed" paragraph replaced by the (H-σ) paragraph (B3 text).
- §7: H-a1/H-a2 bullets → (H-σ); routine items marked done where they are.
- §8: retitled; table, H-a1/H-a2 conclusions and the double-precision radius evidence removed; the superseded note of
  B3_borel.md §9 at the top, why the old check failed, the ball-arithmetic picture (with its truncation stated), the 2Φ
  candidate now by Lemma B3.7, and the b-mode remark kept.
- §9: B5.0 checked for the M + 1 model (b5_system.py).
- §10.3: B5(1) statement replaced (B5_inner.md §9 text), proof sketch replaced by a pointer; B5(2)'s error term changed
  from (ln n)² to (ln n)^{Re⟨F²⟩} (B5_inner.md §7) and the S^{(1)} refinement marked as not run.
- §10.4: B5 status replaced (B5_inner.md §9 text, plus the B5(2) estimate).
- §11: B8.2 wording (B3 text); "What remains in B8" replaced by "Lemma B8 with constants" (B8_constants.md §12 text) and
  the four departures of B8_constants.md §11 (R′_n, K = 4, light-cone disc, gauge).
- §13: new status table (canonical); new §13.1, the status of Theorem B(1), link by link with thresholds, and what is
  missing.
- References: Braaksma issue and DOI; Ramis–Sibuya 1994, Canalis-Durand 1993, Sokal 1980 added; Costin 1998 title
  corrected to the one verified in REF_CHECK §3.

**PROOF.md**: §2's "Everything nonlinear is open" → pointer to §4–5; B.2's "modulus about 0.13" obstruction replaced;
§5's status paragraph updated (B3 and B5 proposed text) and a four-row status table added (B5(1), B8 as proposed in
B8_constants.md §12, the outer Borel plane, and the Theorem B(1) summary row), pointing to C_selection.md §13.

**README.md**: "Where rigour fails" (a) rewritten without the 0.13 claim; "Status of these gaps" shortened to a pointer;
in "What is known about the selection, and what is open" the "(a singulant of modulus about 0.13)" parenthesis removed,
a sentence on the fixed-M recomputation added, and a new bullet on layer B and Theorem B(1). Inline-math count 406 → 405.

**paper/math/main.tex** (rebuilt with `make`: no errors, no undefined references, no overfull boxes): abstract and
Results item 5; Table 1's Theorem B row; "Where rigour fails" (a) and the paragraph after it; Section 7's introduction;
Definition of the M-model (|ν| ≤ M + 1, 8N + 2 unknowns, a paragraph on why and on the projection); Table "The lemmas of
Theorem B(M)"; "The cited lemmas" → "Lemmas B3 and B4" (B3 restated as proved, (H-σ) with Sokal, B4.P); "What remains:
two contractions" → "The two contractions" with (H8), the new B5(M), B8 with the M = 1 constants, and a new subsection
"Status of Theorem B(1)" (label sec:B1status); Outlook item 1; bibliography (Braaksma issue and DOI; CanalisDurand1993,
RamisSibuya1994, Sokal1980 added, Sokal flagged "volume and pages from a citing record").

**notes/REF_CHECK.md**: addendum with B3_borel.md's verified references.

`python3 check_numbers.py --selftest`: 62 of 62 pass, 62 of 62 reject a shifted value.

## Not applied, or applied differently

- In the README and the paper the H-a1/H-a2 history is not described ("Do not write about earlier versions of the work",
  AGENTS.md); (H-σ) is stated directly. C_selection.md keeps the history, since B3_borel.md refers to H-a1/H-a2 there.
- B3_borel.md's suggested README sentence was adapted: it now also says that summability is not needed at fixed M.
- The paper's B5 row follows B5_inner.md §9 but in the table's shorter form; the paper does not quote B5's δ_ref.
- Not touched (outside the named sections): the README's Layout section, which does not list B3_borel.md, B5_inner.md
  or B8_constants.md and still calls `outer_borel.py` a check for the selection theorem; the notes' own status tables
  (B5_inner.md §8 has (H8) "open", B3_borel.md §8 has B3's existence half "to be written with constants"; both since
  settled by B8_constants.md).

## Discrepancies found

1. **B3_borel.md uses a different model at M = 1.** `b3_outer_m1.py`, `b3_gevrey_certificate.py` and `b3_outer_M.py`
   keep b and c on |ν| ≤ M − 1 (θ-independent at M = 1) and project each equation once. Theorem A(M = 1), B5(1) and B8
   use |ν| ≤ M + 1 with binary projection. So B3.7's certified M = 1 constants, the certified 2Φ interval, Lemma B3.2's
   M = 1 reduction ("b and c have only the harmonic 0 when M = 1") and all of B3_borel.md §5's numerics are for another
   model than Theorem B(1). Not on the critical path, but those claims should be relabelled or rerun. Stated in
   C_selection.md §8, §13 and §13.1.
2. **κ₁.** B5(1), B8 and (H8) are certified at κ₁ = 0.6. The B5(2) threshold ln n ≳ 104.7 (B5_inner.md §7 and the task
   prompt) is at κ₁ = 0.85, where B8 and (H8) have not been run; at κ₁ = 0.6 `b5_threshold.log` gives ln n ≳ 284.75.
3. **H-a1/H-a2 labels.** B3_borel.md §1 restates them as "all singularities at |arg| ≥ 0.48π" and "2Φ absent";
   C_selection.md §4.2 defined them as "b-mode singularities off ℝ⁺" and "x-independent singularities not on ℝ⁺". Both
   are replaced, so this is harmless.
4. **B5(2)'s power of ln n**: (ln n)² in C_selection.md, (ln n)^{Re⟨F²⟩} in B5_inner.md §7 (applied).
5. **Consistent, recorded for the record**: B5's n₀(1) = max(e^{73.89}, n₈) with n₈ = e^{62} from B8; B5's leading
   δ_Φ = 76.56 against B8's certified 79.04 (B8 adds its remainder 2.46); profile phase conventions differ by θ₀ = π/2
   (immaterial for |W|, B8_constants.md §8.4); B8's R′_n and K = 4 against C_selection's R_n and K* = 3.
6. **Not between the notes, pre-existing**: C_selection.md §1's second remark says the repository's solvers compute such
   truncations; the paper (after its cross-check) says they do not. Left as is.

## A missing link found while writing §13.1

The error control of the outer transport of the jump along the Stokes line x > 1, from the light cone out to the inner
arc |x| = n^γ, is not written in C_selection.md, B5_inner.md or B8_constants.md. B6/B6′ cover x ∈ [½, 1), and
Proposition 3 is a leading-order statement. It enters the explicit factor κ_{n,μ} in C_{n,μ} = κ_{n,μ}[S_μ + R]. Whether
conclusions (2) and (3) of Theorem B need it, or only the equivalence "analytic at x = 1 ⇔ Y⁺ = Y⁻ on x > 1 ⇔ the jump's
label vanishes", is for the coordinator (or a mathematician) to decide. Listed as missing item 3 in C_selection.md §13.1.

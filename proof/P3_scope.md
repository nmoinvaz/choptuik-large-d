# P3 scoping note: Theorem A(M = 5), what failed and what to change

Queue item P3 of [notes/PROOF_QUEUE.md](../notes/PROOF_QUEUE.md): the point certificate for the five-harmonic inner
zero, the first truncation close to the real answer (ω/ω∞ = 0.999228). [A_M.md](A_M.md) §6.3 and §7 record that the
four box runs on the local machine failed and that the next step is a design change, not more runs. This note says
what the logs show, what the candidate designs are, and the one measurement that chooses between them. Written
2026-10-28 by Claude (Anthropic) under Nathan Moinvaziri's direction; not reviewed.

## 1. The statement needed

Theorem A(M = 5) as A_M.md §1 states it for M = 3: exactly one simple zero of g(p) = (J₁, J₃, J₅) at ξ₁ = 0.3 in an
explicit box around p̂ = (A, ω/ω∞, B₃, C₃, B₅, C₅) = (2.7185791892, 0.9992284032, 0.1492035749, 0.0147190525,
0.0084053469, 0.0014703452), given the M-harmonic start-value lemma and Braaksma's theorem; then lemmas S, 6a, 6b₀ and
6b₁ at M = 5 ([A_lemmas_M.md](A_lemmas_M.md), already written for general M) turn it into the statement about the
selection condition. The centre is ready (A_M.md §6.3: |g| ≤ 2.5×10⁻²⁹, quadratic convergence).

## 2. What the runs show

`solver/proof/logs/A_M_box_M5_1e-13.log` and `logs/A_M_krawczyk_M5_1e-13.log` (the one run that finished; h = 10⁻¹³,
w = (10⁻¹⁹, 10⁻²³)):

| where on the path | steps | state radius | derivative radius |
|---|---|---|---|
| arc, t = 0.73 of 1 | 725 | 2.5×10⁻¹³ | 1.9×10⁻² |
| arc, t = 0.84 | 800 | 3.0×10⁻¹² | 0.21 |
| end of the arc (ξ = 0.2) | 863 | 6.5×10⁻¹¹ | 4.8 |
| real leg, ξ = 0.25 | 50 | 8.1×10⁻⁸ | 8.6×10³ |
| end of the real leg (ξ₁ = 0.3) | 101 | 1.9×10⁻⁵ | 3.2×10⁸ |

The Krawczyk margins are −13.7, −11.8 (outer rows), +0.29, +0.33 (harmonic 3) and −542, −511 (harmonic 5), with
"max radius of Y Dg̃(X) 9.3×10⁷". The other three runs (h = 3×10⁻¹¹ and 10⁻¹² twice) stopped on the arc when the
derivative radius passed order one.

Reading. The excess is not the box: the derivative radius multiplies by 250 over the last fifth of the arc and by
10⁸ over the real leg, while the step size stays about 10⁻³ (arc) and 10⁻² (real). That is the fast directions. On
the real leg the harmonic-ν fast mode grows like e^{1/(2ξ²)} relative to the slow solution, 10³ from ξ = 0.2 to 0.3,
and at M = 5 the harmonic-5 components are 3×10¹⁴ per unit of B₅ at ξ₁ (A_M.md §2), so any excess the Lohner sets carry
in a fast direction is amplified by that factor, whatever the box. Two consequences in the test. The harmonic-5 rows
need w₅ above the outer columns' excess in the J₅ components divided by 3×10¹⁴, times h: with the measured excess that
is about 5×10⁻²¹, 500 times the run's w₅ (the −542). The outer rows fail by 14 because the outer columns' own excess in
the J₁ components, after the Schur correction, is 14h; this ratio is roughly independent of h (the excess is linear in
h, A_M.md §2), so no smaller h passes either. A box with w₅ = 5×10⁻²¹ would make the state set 3×10¹⁴ × w₅ ≈ 10⁻⁶
wide in the harmonic-5 coordinates, which a C¹ family integration tolerates (0.1%), so the inner rows are curable by w
alone if the outer rows were not failing.

## 3. The candidate designs

1. **Second-order chart and C² enclosures** (A_M.md §2's proposal). The state set over the box represented as
   y(p̂) + V x + ½ x·H·x with V a point column and H enclosed over the box, so the inner widths are third order in h
   and Dg̃(X) is enclosed in mean-value form. Removes the h-linear excess of the derivative columns, which is the outer
   rows' failure if that excess is box-generated. Cost: the second-order parameter jets in `certify_M_sys.py` (21
   columns at M = 5), their Lohner sets in `certify_M_lohner.py`, about 4× the step cost; a month of agent work with
   the planted tests (M = 1 must reproduce Theorem A(M = 1)'s box, M = 3 the §6.2 certificate).
2. **A fast-direction-aware set representation.** The Lohner sets' excess in the fast directions is what the real
   leg amplifies. Carrying the sets in the de-mixed basis (hier.py's e^{−Π} and K′ normalisations, a fixed linear map
   per step), or a QR-factored Lohner set that keeps the fast directions separate, would stop the slow-direction
   excess from being rotated into the fast ones. Cost: a change inside `validated_int.Lohner`'s set update, two
   weeks; tested against the M = 1 and M = 3 certificates.
3. **Pose the inner conditions at the end of the arc** (the mixed system of the centre refinement, A_M.md §6.3): J₃
   and J₅ at ρ = 0.2, J₁ at ξ₁ = 0.3. The harmonic-5 rows then see sensitivities of 3×10¹¹ instead of 3×10¹⁴ and no
   real-leg amplification, which cures the −542 directly; the outer rows are unchanged, so this does not cure the
   −14 alone. Needs lemma 6a(M = 5) at both radii (the vanishing D-jumps at one radius force the whole jump to
   vanish), which A_lemmas_M.md's 6a already does for M = 3 at ρ₆ₐ = 0.041. Cost: a week (the jump functional applied
   at the arc's end for the inner rows; the leakage L and the tilt recomputed for the mixed Jacobian).
4. **A higher-order a-priori enclosure.** Cuts the step count several-fold (A_M.md §3) and with it the per-step
   excess; a cost item, not a design fix, unless the excess is dominated by the number of steps.

## 4. The measurement that decides

A **point run** at M = 5 (h = 0: the C¹ run along the one trajectory at p̂, A_M.md §3), on the same path, reporting
the state and derivative radii at the same five places as the table above. It separates the two sources of excess:

- if the point run's derivative radius also reaches 10⁸ at ξ₁, the growth is the enclosure's own handling of the
  fast directions and the box plays no part: design 2 (with 3 for the inner rows), and design 1 would not help;
- if it stays small (10⁻⁵ or less at ξ₁, as the M = 3 point runs did), the growth is box-generated and design 1 is
  the fix, with 3 as the cheap complement.

Cost: one process, about 2000 steps at 1.5 s, 1 hour on an idle core of the Mac or 3–4 hours beside other jobs; no
new code (`certify_M.py point 5 --centre=data/A_M_refine_M5_d.json --use_out`, with the output and log paths of A_M.md
§8; the exact command is checked against §8 before it is queued). The second cheap measurement is the mixed-system
box run of design 3 at the failed run's (h, w), which the refinement's mixed mode almost provides, to see the outer
rows' margin without the harmonic-5 rows.

## 5. Order of work

| step | item | who | cost | decides |
|---|---|---|---|---|
| 1 | the M = 5 point run (§4) | the Mac, one thread, after T87's nodes free it | 1–4 h | whether the excess is the enclosure's or the box's |
| 2 | design 3 (inner rows at ρ = 0.2) | cloud agent | 1 week | the −542 rows; lemma 6a(5) at both radii |
| 3 | design 2 or 1, by step 1's answer | cloud agent | 2 weeks or a month | the outer rows' −14 |
| 4 | the four box runs again, then Krawczyk, controls, re-check, 6a export (A_M.md §7's recipe) | the Mac or Windows, 4 threads | 3–4 h | Theorem A(5) |

Theorem A(M = 5) matters because it is the first truncation whose zero lies within 10⁻³ of ω∞; it does not by itself
address uniformity in M (P6), and the entire-function route of [P5_scope.md](P5_scope.md) is the alternative that
does not need it.

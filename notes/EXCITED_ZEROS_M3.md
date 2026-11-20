# EXCITED_ZEROS_M3: proposed text for DIRECTION_REVIEW.md on #1021's M ≥ 5 tests of the excited M = 3 zeros

This file proposes a section for [DIRECTION_REVIEW.md](DIRECTION_REVIEW.md) §2.3 and does not edit that file. It
summarizes #1021's attempts to carry three excited M = 3 zeros (box 7's, box 16's and b11-15's +1) to M = 5 and M = 7.
Drafted 2026-11-20 by the Mac's xfchecks session for the coordinator. Raw data are in choptuik-large-d-raw
`I1021-mac/z7_m7/` (xi015c at a2ad786e; xi015d at bcdc5de8, d2365a9a, 9789745a and e1073c9f) and
`I1021-mac/z16_m7/rep3/` at 52359070.

## Proposed text (after §2.3's **First experiment.**)

**M ≥ 5 test of three excited zeros (#1021).** The adaptive count found −1 zeros of the M = 3 multiplier in box 7, at
(A, ω/ω∞) = (3.245479, 0.345852), located at ξ₁ = 0.17, and in box 16, at (3.553347, 0.413172), located at ξ₁ = 0.18,
and a +1 zero in boxes 11-15, at (3.526471, 0.211495), located at ξ₁ = 0.18. A zero
counts as converging when it survives at M = 5 and M = 7 under #1027's rule: |F₇/F₅| ≤ 0.2 at the M = 5 zero, with an M
5 → 7 shift smaller than the M 3 → 5 shift. The route is an outer (A, ω) Newton over fixed-(A, ω) hierarchical solves
(`walk1026b.py` at M = 5, `walk1027b.py` at M = 7, `hier_nb.py` at fc7dee4d). The criteria were fixed on #1048 before
the runs. Of ξ₁ = 0.15–0.25, only at 0.15 do both of the known zero's controls converge, so the box walks run at 0.15.
Each run's result counts only if both controls pass in that run.

- **Controls.** Both pass in all five counted runs. #1026's M = 5 walk ends +2.16×10⁻⁶ and +1.38×10⁻⁶ from the stored
  zero in A and ω/ω∞. #1027's M = 7 walk ends −1.43×10⁻⁶ and +6.06×10⁻⁷ from its zero, with |F₇/F₅| = 0.0573 (stored
  0.0574). The code is deterministic, and the five runs give the same zeros to the printed digits.
- **The starts all failed.** The first fixed-(A, ω) M = 5 solve at the zero's M = 3 place does not reach the solve bar
  |r₃|, |r₅| < 10⁻⁶. This holds for every start and with #1027's three perturbed repeats:

  | zero | start: the M = 3 harmonics | start: its best ξ₁ = 0.2 M = 5 attempt |
  |---|---|---|
  | box 7 | best \|r₃\| 4.8×10⁴ | best \|r₃\| 4.6×10² (the attempt had 2.8×10⁻⁶ at 0.2) |
  | box 16 | best \|r₃\| 2.3×10⁶ | best \|r₃\| 6.4×10³ (1.2×10⁻⁴ at 0.2) |
  | b11-15 | not run | best \|r₃\| 1.5×10⁶ (2.0×10⁶ at 0.2) |

  So none of the three has an M = 5 zero, and no M = 7 walk ran. b11-15's 0.2 attempt is the best of four: two
  repeats reached 3600 s, the cap per attempt set before the runs, and count as failed.
- **Cause for boxes 7 and 16: the double-precision floor at ξ₁ = 0.15.**
  - From the same start, the solve at ξ₁ = 0.2 reaches |r₃| 2.3×10⁻⁵ (box 7) and 5.5×10⁻⁴ (box 16).
  - At 0.15, hier's level-5 solves stop at their noise floor, |g| 3.2–5.2×10⁻² (box 7) and 0.14–0.68 (box 16). That is
    3×10⁴ to 7×10⁵ times the solve bar. At 0.2 the same stops sit at 7×10⁻⁸ to 2×10⁻⁶.
  - From 0.2 to 0.15, the largest entries d r₃/d x₃ and d r₅/d x₅ of hier's Jacobian K grow by 5.4×10⁵ and 6.2×10⁵ (box
    7), and by 3.4×10⁵ and 4.6×10⁴ (box 16).
  - The floor is rounding, not integration error. Cutting the integrator tolerance from 10⁻¹² to 10⁻¹³, 10⁻¹⁴ and 10⁻¹⁵
    leaves it at 9.9×10⁻³ to 9.4×10⁻² (box 7) and 7.8×10⁻² to 0.82 (box 16).
  - The walk's own residual is coarser still. At 0.15, K's largest entry d r₁/d x₃ is 1.0×10¹⁷ (box 7) and 4.9×10¹⁸
    (box 16). The spacing of doubles at B₃ is 1.1×10⁻¹⁶, so r₁ moves by about 11 and 5.4×10² per representable step of
    x₃. The walk's bar on |r₁| is 3.8×10⁻².
  - The known zero (A, ω/ω∞ ≈ 2.72, 1.00) converges at 0.15 in the same code, so the floor belongs to the boxes' points.
- **Cause for b11-15: no M = 5 solution at either radius.** Its solve fails at ξ₁ = 0.2 as well, so the floor above does
  not explain it.
  - At 0.2, from the same start, the level-5 solve stops at once at hier's noise floor, |g| 2.10×10⁵, while one double
    step of x₅ moves r₅ by only 1.5×10⁻⁵ (K's d r₅/d x₅ 1.38×10¹¹). So no Newton trial reduces |g| there, far above
    rounding. Level 3 then reaches |r₃| 3.4×10⁻² at harmonics far from the start (F₃/F₁ 0.2449 + 0.1035i, against
    0.2769 − 0.0005i), with |r₅| still 1.1×10⁵.
  - At 0.15 the level-5 solve stops at once at |g| 7.66×10⁵.
  - With integrator tolerance 10⁻¹⁴ (default 10⁻¹²), the first stop at each radius is unchanged.
  - The data do not say whether an M = 5 zero lies near this point.
- **Verdict: UNDECIDED** for all three (#1027's rule with no M = 5 zero; #1021 comments 7036844074, 7036555702,
  7039910892, 7040076131, 7043912653). The runs neither confirm nor rule out an excited zero. For boxes 7 and 16, double
  precision cannot resolve the solve bar at the one radius where the controls hold. For b11-15, no M = 5 solution was
  found at either radius, from its M = 3 harmonics (at 0.2) or from its best 0.2 attempt.

**Smallest change that would decide it.** Double-double arithmetic in `inner_nb.py`'s kernels (`conv`, `rhs`, `fpath`,
`dop853`, `series_nb`), with the harmonics x carried in double-double in hier's update. The Jacobian stays in double, as
iterative refinement. Every bar, the gate and #1027's rule stay as fixed on #1048. Three alternatives fail on the data
above:

- A tighter integrator tolerance: the floor does not move.
- Another radius: both controls pass only at 0.15 (#1048).
- A looser solve bar: that would change the criteria, and the walk's |r₁| would still sit above its bar, since one
  representable step of x₃ moves r₁ by more than the bar.

This change is for boxes 7 and 16. For b11-15 it may not be enough: its solve fails at ξ₁ = 0.2, where rounding is far
below the stop. Deciding b11-15 first needs an M = 5 start that converges at 0.2.

**Cost.** The M = 5 work is one to two agent-days of code and checks, with this pilot first:

- Double-double reproduces the double residuals at the known zero within the double floor.
- At box 16's 0.15 start (the larger floor), the level-5 solve reaches 10⁻⁶.
- Dropping the low parts (a `--mutate`) brings back the double floor.

The run time comes from measured parts and one assumption (`I1021-mac/z7_m7/xi015d/dd_bench/`):

- **Kernel, measured:** the double-double `conv` at the M = 5 truncation (L = 7) takes 5.2 times as long as complex128.
  Its error against an exact rational sum is 9.3×10⁻³³, against 1.1×10⁻¹⁶ in complex128.
- **Whole residual, assumed:** taking 5–20 times the double cost, and the 0.15 controls' 25 ms per residual and 220
  residuals per solve:
  - an M = 5 solve takes 0.5–1.8 min;
  - a walk of 8–25 solves takes 0.06–0.8 h per box.
- **M = 7:** if M = 7 needs the same change (it was not reached), that is 5.7–22.8 min per solve and 0.8–9.5 h per walk
  (79 ms per residual, 868 residuals per solve).

**Then:** rerun both boxes and both controls at 0.15 under #1048's criteria. A CONVERGES verdict would seed hp at D =
40, as §2.3 plans.

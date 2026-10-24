# How large a step in D Newton takes at D = 45–60 (T43)

The local machine's step-size test (`hp/step_test.py`, task T43, 2026-10-23; data, every run's records and the
summary in `hp/data/step_test/`, `summary.md`). Each run continues in D from a converged state at D0 = 45.05, 50.01,
55.04 or 60.00 with `hp/scan_hp.py` under an aggressive step controller (the step grows 1.5 times after every success
and halves on a failure), with the scan's own 7 states before D0 as predictor history, for up to 20 converged points,
1 hour or a span of 3 in D. The edge is the median of the last 8 attempted steps: the step Newton tolerates there.
Every converged point is checked against the stored continuation (`hp/data/delta_dd.csv`): all 21 runs that stepped
agree to at most 1.1×10⁻¹⁴, and the same report with the reference shifted by 10⁻⁹ fails all 21 (the control).

Variants: `base` (the scans' cubic predictor through 4 points); `pred6`, `pred8` (Lagrange predictors through 6 and 8
points); `pin11`, `pin9` (`--pin`); `node06` (matching node at 0.6 instead of 0.7); `kf11` (centre τ-harmonics frozen
above 11 instead of 13).

D per 100 Newton iterations over the whole run, with the edge in brackets:

| D0 | base | pred6 | pred8 | pin11 | pin9 | node06 |
|---|---|---|---|---|---|---|
| 45.05 | 0.484 (0.119) | 1.900 (0.458) | 2.754 (0.470) | 0.395 | 0.410 | 0.080 (0.030) |
| 50.01 | 0.263 (0.092) | 1.226 (0.346) | 2.658 (0.473) | 0.280 | 0.288 | 0 (no step) |
| 55.04 | 0.125 (0.046) | 0.844 (0.292) | 0.042 (0.0003) | 0.180 | 0.191 | 0 (no step) |
| 60.00 | 0.113 (0.034) | 0.023 (0.004) | 0.020 (4 of 14 steps) | 0.114 | 0.115 | 0 (no step) |

`kf11` never stepped: its start states' floors are 5×10⁻¹⁴ to 4×10⁻⁸, so freezing more harmonics is not an option.

**Findings.**

1. **base's edge falls with D:** 0.119, 0.092, 0.046 and 0.034 at D0 = 45, 50, 55 and 60, about e^{−0.084 D} or
   D^{−4.4}; four points do not separate the two. At D = 60 it is 7–17 times the 0.002–0.005 the scans used near
   D = 61, so those steps were set by the scans' caps, not by Newton.
2. **Higher-degree predictors win up to D = 55, then fail.** pred6 is 4–7 times faster than base at D0 = 45–55 and
   pred8 5.5–10 times at 45–50. pred8 at 55 and both at 60 fail the same way: their predictions have |F| of 10⁻¹³ to
   10⁻⁸, and Newton stalls at 10⁻²³ to 10⁻¹⁸, often within 2–10 times the tolerance after 25–29 iterations, even at
   steps of 2×10⁻⁴. The likely cause, from the local machine's analysis: extrapolating through 6–8 points amplifies the
   near-null noise that each converged state carries, and that noise grows with D. A predictor whose degree adapts
   (6–8 while Newton converges, back to 4 when it stalls) would keep the gain at D ≤ 55 and lose nothing at 60; not
   tried yet.
3. **The pins change nothing:** pin11 and pin9 are within 0.82–1.53 times base.
4. **Matching at 0.6 does not continue:** node06 converges at D0 = 45 (at a quarter of base's edge) and takes no step
   at 50, 55 or 60. The certification of the 0.7 solution in the 0.6 layout (notes/BEYOND_LIGHT_CONE.md §5.1) gets
   the 0.6 layout's smaller Δ bound without stepping there.

Phase 3 of the test, a run combining every option that beats base at 3 or more of the 4 D0, found only pred6, so no
combined run was made.

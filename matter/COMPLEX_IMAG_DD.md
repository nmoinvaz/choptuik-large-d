# The real DSS's imaginary sector at large D in double-double (#375)

matter/COMPLEX_CRITICAL.md §3 counts the zeros of the imaginary sector's det M(λ) with `complex_imag.py` (double
precision) at D = 4, 10, 20 and with hp/modecount_cx (double-double) at D = 60. This page puts the sensitive parts of
complex_imag.py's determinant in double-double, as a second implementation at large D. Script
`solver/matter/complex_imag_dd.py`, arithmetic `solver/matter/ddnp.py`, raw `I375-mac/` of the raw-data repository.

## 1. Why double fails at large D

At D = 60 and λ = 1.3 + 0.4i the double-double determinant has ln|det M| = −1450.79. The double code gives −1069.12,
a factor e^−382 too large. It is not noise around the right value. Each block of the matrix was swapped, one at a time,
from the double matrix into the double-double one (`diag/blocks_D60_ks200.log`):

| block | its entries off by | change of ln\|det M\| |
|---|---|---|
| propagator 0.3 → 0.4 | 3×10⁻¹⁵ | 4×10⁻⁴ |
| propagator 0.4 → 0.5 | 8×10⁻¹⁵ | 1.6 |
| propagator 0.5 → 0.6 | 7×10⁻¹⁵ | 85 |
| propagator 0.6 → 0.7 | 6×10⁻¹⁵ | 288 |
| light-cone block | 8×10⁻⁵ relative | 381 |

The cause is rounding, not truncation. Random relative changes of the double matrix's entries move ln det by about
10⁻² at 10⁻¹⁶ and 4×10⁻⁸ at 10⁻²⁰ (`diag/sens_D60.log`), while the double-double Taylor integration gives the same
det M from tolerance 10⁻¹² to 10⁻²⁴ (§3). The light-cone block in double loses about seven digits in its series
recursion already at D = 30.07 (7×10⁻¹⁰ relative), which puts the double det 0.05 rad off there
(`diag/blocks_D30.log`).

## 2. Method

- **Arithmetic.** Double-double on numpy arrays, written for this check (`ddnp.py`: Dekker's and Knuth's error-free
  transforms, Hida, Li and Bailey's operations). Its selftest compares every operation with mpmath at 60 digits
  (largest relative error 2.7×10⁻³², limit 10⁻³⁰); with the product's error term dropped (`--mutate=twoprod`) it
  fails at 10⁻¹⁶.
- **Propagators.** Background and test field are integrated together by a Taylor method of order 32 from each stored
  node (hi + lo of the hp state) to the next. The step makes the last two terms at most `--tol` of the first, per
  background field and per propagator column. The steps are chosen once per run, at λ = 6 + 1.5ωi (the far corner of
  the boxes), so that det M is analytic in λ.
- **Light-cone block.** complex_imag.TestField.horizon's recursion in double-double, on the background's 34-digit
  light-cone series, each order solved by an LU in double refined with double-double residuals (last correction below
  10⁻³⁰ of the solution).
- **Determinant.** An LU with partial pivoting in double-double, banded (lower and upper bandwidth 3M), and
  ln det M = Σ ln u_ii + iπ (row swaps). Its selftest reproduces mpmath's determinant of a banded matrix to 1.2×10⁻¹⁴
  in ln det, and without pivoting (the control) misses it by 2×10⁻².
- **What stays double.** The centre series and its propagation from x_s = 0.05 to the first node (DOP853, rtol 10⁻¹³).
  At D = 60 a random relative change of 10⁻¹⁵ in that block moves the phase by 10⁻⁶ rad, and rtol 10⁻¹¹ against 10⁻¹³
  by 7×10⁻⁸ rad (`diag/centre_D60.log`).
- **Shared with hp/modecount_cx:** the stored states and the equations of hp/src/model.rs. **Shared with
  complex_imag.py:** the double background used by the centre block, the centre series, the background's light-cone
  series, the candidate poles and `count_box`. The double-double arithmetic shares nothing with multidouble.

## 3. Checks

(`complex_imag_dd.py check`, tol 10⁻²⁰, `check_D30.jsonl`, `check2_D30.jsonl`, `check_D60.jsonl`)

| check | D = 30.07 | D = 60.00 |
|---|---|---|
| background from each stored node to the next | 5×10⁻¹⁷ (0.3 → 0.4), 6×10⁻¹⁹, 2×10⁻²⁰, 1×10⁻²¹ | 1×10⁻²⁴ … 1×10⁻²³ |
| the same at tol 10⁻²⁴, order 40 | unchanged, to 10⁻²⁶ | 3×10⁻²⁵ … 3×10⁻²⁹ |
| λ = 0: the background's (p, m) through the propagators | ≤ 3×10⁻¹⁷ | 0 |
| λ = 0: the light-cone block applied to m_h | 1×10⁻²⁰ | 9×10⁻²¹ |
| light-cone series at 0.65 carried to 0.7 by the integrator, against the series at 0.7 | 1×10⁻¹⁹ relative | 5×10⁻¹¹ relative (the series' tail at u = 0.35 is 10⁻⁸) |
| matching points every 0.02 against none (det M unchanged in exact arithmetic) | 2×10⁻¹¹ rad | 2×10⁻¹¹ rad |
| light-cone order 100 against 200 | | 1×10⁻¹⁰ rad |

At D = 60, λ = 1.3 + 0.4i, the phase of det M moves by 7×10⁻¹¹ rad between tolerances 10⁻¹² and 10⁻¹⁴, and by
1.1×10⁻¹⁰ rad over 10⁻¹² … 10⁻²⁴ (21 to 48 Taylor steps; `tolscan_D60_pilot.jsonl`).

## 4. Counts

(to be filled from the runner's raw folder)

## 5. Discrepancies

- **The stored state up6_D30.0688** is not consistent with the model equations on 0.3–0.4 at the level of its
  residual. The double-double background integrated from node 0.3 misses node 0.4 by 5.2×10⁻¹⁷, the same at two
  tolerances and two orders, while the crate's record of the state gives |F| = 6×10⁻²⁵. up13_D60.0020 agrees to
  3×10⁻²⁵. Its effect on det M was not measured. For scale, entries of the propagator 0.3 → 0.4 changed by
  1.4×10⁻¹⁴ move the phase by 4×10⁻¹⁰ rad at D = 30 (`diag/blocks_D30.log`).
- **complex_imag.py at D = 30.07** is 0.05 rad off in the phase of det M, from its light-cone block. Its counts at D = 4,
  10, 20 (COMPLEX_CRITICAL.md §3) are not affected by this page's numbers; whether its light-cone block is accurate
  there was not checked here.

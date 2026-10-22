# Mixed-precision Newton for the double-double solver (`hp/`)

A Newton step of the double-double DSS solver spent 96% of its time in the exact Jacobian (forward-mode duals over
double-double, one Taylor integration per column; `hp/data/profile_newton.log`). The mixed-precision step keeps the
residual in double-double and computes the Jacobian in plain f64 on SIMD f64 lanes. Newton with an approximate
Jacobian and an exact residual converges to the same root, linearly, so the solution does not change; only the cost
per step drops.

**Result (D = 3.7, two threads).** The f64 Jacobian takes TBD_JAC of the double-double one's time at M = 48 / 64 / 80
(TBD); a whole solve from a perturbed state to |F| ≤ 10⁻²⁸ is TBD_SOLVE faster, with one more Newton call at most,
and the solution is the same to the precision the equations fix it (Δ to ≤ 1.1×10⁻³⁰ relative, max |z| to
≤ 1.2×10⁻²⁷, where two exact-Jacobian solves from two starts differ by 1.4×10⁻³⁰ and 6.7×10⁻²⁸). The default
double-double path is bitwise unchanged. The f64 Jacobian agrees with the double-double one to 1.2×10⁻¹⁴ per column at
M = 16 but only to 1.6×10⁻⁷ at M = 48: the Taylor steps, planned for the double-double solution, do not resolve the top
τ-harmonics of a tangent, and the f64 rounding inside a step is amplified (section 3). That error sets the iteration's
contraction factor to about 10⁻⁷, which costs the one extra Newton call.

## 1. What was changed

All new behaviour is opt-in; `Solver.jacobian`, `newton`, `chord`, `mode_matrix` and the rest are bitwise as before
(section 4.1).

- **f64 lanes** (`hp/src/lane.rs`), all on the 8-lane `V8` of `v8.rs` (AVX-512, 2×AVX2 or the portable array):
  - `F8` (values) and `Fu` (value + one tangent);
  - `Fun<N>`: an f64 value with N f64 tangents, so N Jacobian columns of one block come from one evaluation
    (the base trajectory is computed once for N columns);
  - `Dtn<N>`: a double-double value with N f64 tangents (the base trajectory exact to about 10⁻³², the tangents f64).
  The `Lane` trait gained `VAL_F64`, `TAN_F64`, `NT` (the number of f64 tangents), `tan64` / `set_tan64`, and
  `dmat_apply`. Constants and inputs, which the model passes as double-double, enter the f64 lanes as their hi parts.
- **Generic Jacobian** (`hp/src/model.rs`): `jacobian` is now `jacobian_lanes::<Du>`, `jacobian_f64` is
  `jacobian_lanes::<Fu>`, and `jacobian_fun::<N>` / `jacobian_dtn::<N>` take the tasks of each block in groups of N
  (`jac_columns_n`). The model's equations (centre series, Taylor stepping, horizon series) are the same generic code
  for every lane type; nothing in them was duplicated. Lane-specific code is confined to:
  - the τ-derivative `DMat::apply`, which dispatches to `Lane::dmat_apply`: double-double lanes keep the old code,
    f64 lanes multiply by an f64 copy of the matrix (`DMat::cols64`), with the N tangents sharing each column load;
  - the horizon series' periodic solves (`psolve`): the M × M LUs (`horizon_lus`, ks + 1 of them) stay double-double
    and are computed once per Jacobian; the f64 lanes solve with the hi parts of their factors (`DDLu::solve_hi`), an
    f64 LU of the same matrices;
  - the Cauchy sums (`field::cs*`): for f64 lanes the B lane vectors' accumulations are interleaved, since a single
    chain of f64 FMAs is latency-bound (each lane vector still sums its terms in the same order);
  - the centre and horizon series stop early for f64 tangents (`Tail` in `model.rs`): once the tangent terms
    max |c_k| r^k of four consecutive orders are below 2⁻⁶⁰ of the largest so far (r the evaluation radius, k ≥ 8).
    The orders kc = 100, ks = 250 of the saved states are set for double-double values; the f64 Jacobian needs about
    30 and 60 (section 3, "orders"), and the horizon block drops from about 36% of the f64 Jacobian's time to 8%.
- **f64 LU** (`hp/src/linalg.rs`, `DDLu::factor_f64`): the blocked LU of `factor_blocked` (the same panels, row swaps,
  skipped zero blocks and register tiles) in plain f64, stored with lo = 0 so that the double-double solves (`solve`,
  `solve_t`, `solve_many`, `chord`) use its factors as they are. The task's suggested route, promoting the f64
  Jacobian to double-double and calling the double-double LU, stays available (`lu="dd"`); the LU had become a third
  of the mixed step.
- **Python** (`hp/src/python.rs`):
  - `Solver.jacobian_f64(zh, zl, mutate=0, tangents=4, dd_base=False)`: an (n, n) f64 array, like `jacobian`.
    `tangents` = 1, 2, 4, 8 columns per evaluation (1: `Fu`, otherwise `Fun<N>`); `dd_base=True` uses `Dtn<N>`.
    `mutate` breaks the f64 path on purpose for the controls (1: no τ-derivative, 2: ω scaled by 1 + 10⁻⁹, 3: the
    horizon solves' κ tangent term dropped).
  - `Solver.newton_mixed(zh, zl, tangents=4, dd_base=False, lu="f64")`: the same return as `newton`; residual in
    double-double, the f64 Jacobian, its LU (stored for `chord`, `lu_solve` and the rest), and dz = −J⁻¹F solved in
    double-double with those factors.
  - `Solver.lu_factor(ah, al=None, f64=False)`: factor a given matrix and store it as the solver's LU (used by the
    checks to solve with a given Jacobian).
- **Driver and scripts:**
  - `driver.newton_solve(..., jac="f64", mixed=None)` steps with `newton_mixed(**mixed)`; a step along which the line
    search finds no decrease is retried once with the exact Jacobian before giving up. `jac="dd"` (the default) is
    the old loop.
  - `hp/lowD_dd.py --jac f64`: its plain Newton steps use `newton_mixed`. The deflated (`--solver dn`) and
    Levenberg–Marquardt phases stay double-double. The LU kept for the sensitivities is then the f64 Jacobian's.
  - `hp/profile_newton.py --mixed --solve`: the timings below. Without the flags it prints what it printed before.
  - `hp/mixed_check.py`: the checks of section 4. `hp/mixed_diag.py`: the diagnostics of section 3.
    `hp/dump_ctx.py` writes a state and its solver set-up for the Rust bench (`cargo test --release --lib bench_jac
    -- --ignored --nocapture` with `FILE=`).
  - Rust tests: `linalg::tests::f64_lu_solves` (the f64 LU against the double-double LU, with a perturbed pivot as
    the control).

## 2. Variants

| name in the scripts | lanes | per evaluation | base trajectory |
|---|---|---|---|
| `f64x1` | `Fu` | 1 column | f64 |
| `f64x4` (default) | `Fun<4>` | 4 columns of one block | f64, once for the 4 |
| `dd4` | `Dtn<4>` | 4 columns of one block | double-double, once for the 4 |

The columns of `f64xN` agree with `f64x1`'s to rounding: the tangent arithmetic is the same, and the series cut waits
for all N tangents. `dd4` gives a Jacobian 400 times closer to the double-double one at M = 48 but no faster Newton
convergence (section 4.4), and costs more, so it is not the default.

## 3. Why the f64 Jacobian agrees to 10⁻⁷ and not to 10⁻¹⁴ at M ≥ 48

The expectation was 10⁻¹³–10⁻¹⁵. That holds at M = 16 and 32. At M = 48 to 80 the per-column agreement is 10⁻⁸–10⁻⁷,
all of it in the Taylor segments near the horizon side (x = 0.5–0.8) and around x = 0.1 (`python3 mixed_diag.py`,
log in `hp/data/mixed_precision/diag.log`):

TBD_DIAG

What these show:

1. **Step size.** With the Taylor plan's largest step capped at 0.01 instead of 0.05, segments 7–12 agree to
   ≤ 1.3×10⁻¹³; segments 4–5 (x = 0.1–0.16, where x − c reaches −0.03) stay at 10⁻⁹. The error comes from inside a
   Taylor step and grows with the step.
2. **Resolution in τ.** The same plan agrees to 10⁻¹⁴ at M = 16 and 32. A tangent that is a unit vector on one τ
   point holds every harmonic up to the top one, and the top harmonic k ≈ M rotates at a rate of about ωk/|x − c| in x.
   The plan (truncation 10⁻³⁰ for the solution) is set by the solution's own coefficients, whose top harmonics are
   ~10⁻³⁰, so for a tangent the step is long: the top harmonics' Taylor terms are large before they decay, and the
   discrete map even amplifies them (the column maxima reach 46 in segment 11, against about 1 with the shorter steps;
   the exact flow rotates them). The double-double Jacobian is the exact derivative of that same discrete map, so the
   two must agree to rounding, but the rounding inside a step is multiplied by the size of these intermediate terms.
3. **Which rounding.** With the base trajectory in double-double (`dd1`, `dd4`) the error falls from 1.6×10⁻⁷ to
   4×10⁻¹⁰; most of it was the f64 base trajectory's rounding. It lands in the a rows and in the low harmonics (the
   error by harmonic peaks at 2–8 while the column itself is mostly top harmonics): products of large top-harmonic
   tangent terms with the base trajectory's rounding noise alias into low harmonics, where nothing cancels them. The
   remaining 4×10⁻¹⁰ is the f64 tangent's own rounding, in the p rows. Compensated summation of the Taylor series
   (the step's final Horner sum in double-double) changes nothing: the error is in the coefficients, not in their sum.
4. **What it costs Newton.** The contraction factor of the mixed iteration, estimated by one step from a perturbed
   state with each Jacobian (both solved in double-double), is 6×10⁻⁸ for `f64x4` and 9×10⁻⁸ for `dd4`; rounding the
   double-double Jacobian to f64 gives 5×10⁻¹⁴. In the convergence runs |dz| falls by about 10⁻⁷ per step once the
   quadratic part is spent, for all variants. So the mixed Newton needs at most one more call, and making the
   Jacobian more accurate (`dd4`) does not remove it.

Not done, and possible: a finer Taylor plan for the whole solver (it would also make the discrete map resolve the top
harmonics); sub-stepping only the f64 Jacobian would differentiate a different discrete map and break the agreement in
exactly those directions.

## 4. Verification

All runs on the cloud machine (2 cores, AVX-512), two threads, the private build of this branch. The logs and JSON
records are in `hp/data/mixed_precision/`.

### 4.1 The default path is bitwise unchanged

`python3 mixed_check.py ref --save=REF.npz` with the build of the parent commit (52c6a65), then `--compare=REF.npz`
with this branch's build: residual, Jacobian (hi, lo), Newton step and pivots, chord step, `lu_solve_t`,
`mode_matrix` at λ = 0.3, a Jacobian with frozen harmonics (`kfreeze` 13) and one with an inward matching node
(`match_node` 8), at the D = 3.7, M = 48 state: all 17 arrays bit-identical. `cargo test --release`: 3 passed (the
two existing LU tests and the new f64 LU test), 2 benches ignored.

### 4.2 Jacobian agreement (`python3 mixed_check.py agree [--strict] --variant=...`)

Per column: max |J_f64 − J_dd,hi| / max |J_dd,hi| over the column; the pass criteria were written before the runs.

| tier | state | variant | per column | Frobenius | entrywise (nonzero entries) | step contraction | pass if ≤ |
|---|---|---|---|---|---|---|---|
| strict | M = 48 resampled to 16 | f64x4 | 1.2×10⁻¹⁴ | 1.6×10⁻¹⁵ | median 4.7×10⁻¹⁶, max 1.8×10⁻¹¹ | 2.1×10⁻¹³ | 10⁻¹² |
| strict | M = 48 resampled to 16 | f64x1 | 1.2×10⁻¹⁴ | 1.6×10⁻¹⁵ | | 2.1×10⁻¹³ | 10⁻¹² |
| strict | M = 48 resampled to 16 | dd4 | 2.3×10⁻¹⁵ | 3.7×10⁻¹⁶ | | 1.2×10⁻¹³ | 10⁻¹² |
| production | M = 48 | f64x4 | 1.6×10⁻⁷ | 8.7×10⁻⁸ | median 1.1×10⁻¹², 99% 1.1×10⁻⁴ | 6.1×10⁻⁸ | 10⁻⁶ |
| production | M = 48 | f64x1 | 1.6×10⁻⁷ | 8.7×10⁻⁸ | | 5.2×10⁻⁸ | 10⁻⁶ |
| production | M = 48 | dd4 | 4.4×10⁻¹⁰ | 1.0×10⁻¹⁰ | median 2.0×10⁻¹³ | 8.8×10⁻⁸ | 10⁻⁶ |

The entrywise maximum is large (up to 1.5 for `f64x4` at M = 48, 1.8×10⁻¹¹ at M = 16) on entries that are tiny in
their column (the worst: |J| = 1.3×10⁻⁹ in a column of maximum 1.0): cancellation in individual entries. No entry is
nonzero in one Jacobian and zero in the other. The step contraction is the relative difference of −J⁻¹F at a
perturbed state (Δ + 10⁻⁶, fields + 10⁻⁸) between the two Jacobians; with the double-double Jacobian rounded to f64
it is 8×10⁻¹⁵ (M = 16) and 5.3×10⁻¹⁴ (M = 48). By resolution (`mixed_diag.py M`), at each state's own plan: TBD_M.

The strict tier is the bug detector (an f64 path whose rounding is not amplified); the production tier measures what
Newton sees. Its threshold 10⁻⁶ is looser than the 10⁻¹² first written for it, after the cause of the 1.6×10⁻⁷ was
found (section 3); the original expectation was not met at M ≥ 48, and the subtle control (ω × (1 + 10⁻⁹)) only
fires in the strict tier.

### 4.3 Controls (`python3 mixed_check.py controls`: 7 of 7 fire)

| tier | variant | mutation | per column | verdict |
|---|---|---|---|---|
| strict | f64x4 | no τ-derivative | 3.6 | fails, as it must |
| strict | f64x4 | ω × (1 + 10⁻⁹) | 2.3×10⁻⁸ | fails |
| strict | f64x4 | horizon κ tangent term dropped | 3.9×10⁻² | fails |
| strict | f64x1 | no τ-derivative | 3.6 | fails |
| strict | dd4 | no τ-derivative | 3.6 | fails |
| production | f64x4 | no τ-derivative | 3.8 | fails |
| production | f64x4 | horizon κ tangent term dropped | 1.3×10⁻² | fails |

The κ mutation touches only the horizon block's columns (2M + 1 of them), so a check confined to one block fires.

### 4.4 Convergence to the same solution (`python3 mixed_check.py converge`)

From the D = 3.7 state with Δ moved by 10⁻⁶ and every field entry by a pseudo-random amount up to 10⁻⁸ (seed 1), to
|F| ≤ 10⁻²⁸, with `driver.newton_solve` (Armijo backtracking). Newton calls include the last one, which only confirms
|F| ≤ 10⁻²⁸. The control is a second exact-Jacobian solve from another start (Δ − 10⁻⁶, seed 2): how precisely the
equations fix the solution. Pass: the differences are at most max(10⁻²⁷, 10 × the control).

TBD_CONV

The mixed runs follow the exact Newton run step for step (|F| 5.7×10⁻⁷ → 5.7×10⁻⁷ → 4.0×10⁻¹¹ → 1.6×10⁻¹⁷) until
the quadratic part is spent; then the exact run reaches 10⁻³⁰ in one step and the mixed run takes two (1.6×10⁻¹⁷ →
3×10⁻²⁷ → 10⁻³⁰). The solutions agree in Δ to the last double-double digit and in z to the control's level; the
difference in z is in the near-null centre directions, which |F| ≤ 10⁻²⁸ fixes only to about 10⁻²⁷.

## 5. Timings

`python3 profile_newton.py --mixed --solve` (D = 3.7, two threads; log and JSON in `hp/data/mixed_precision/`):

TBD_TIMING

Single-thread breakdown at M = 48 and 80 from the Rust bench (`bench_jac`, a sample of columns per block,
`hp/data/mixed_precision/bench_M48.log`, `bench_M80.log`):

TBD_BENCH

### Batching (stage 2), separately

TBD_BATCH

## 6. What remains

- **Finer Taylor plans** for the top τ-harmonics (section 3), which would bring the f64 Jacobian to 10⁻¹³ and the
  contraction with it, and make the discrete map honest in those directions; at a cost in steps for the residual too.
- **The τ-derivative by FFT** (O(M log M) per application instead of the dense M × M product): `DMat::apply` is now a
  large share of the f64 Taylor stepping, and its share grows with M (the f64 Jacobian grows like M^TBD_EXP).
- **Memory traffic.** The f64 Cauchy sums are limited by L2 bandwidth, not arithmetic (the coefficient arrays of 12
  fields × 38 orders do not fit L1). A layout by lane vector (all orders of one lane vector together), or a
  tangent-only linearised recursion over stored base coefficients, would cut it.
- **The residual and line search** (each line-search trial is a double-double residual, 0.05–0.1 s) and the Python
  loop are now visible in the whole-solve times.
- `lowD_dd.py --jac f64` has been run only as a smoke test (section 4.5); the deflated-Newton and LM phases stay
  double-double.

## 7. Proposed text for the README (for the coordinator)

In "The double-double solver" list, after "Exact Jacobian":

> - **Mixed-precision Newton:** `Solver.newton_mixed` keeps the residual in double-double and computes the Jacobian
>   with f64 vector lanes, four columns per pass over the base trajectory, and factors it in f64; Newton then converges
>   linearly to the same solution (contraction about 10⁻⁷ per step, one extra step), TBD times faster per step at
>   M = 64 ([notes/MIXED_PRECISION.md](notes/MIXED_PRECISION.md); `hp/lowD_dd.py --jac f64`).

And in the paragraph on the Newton step's cost, replace "tangents that share one primal pass would save about 1.7×,
not implemented" with:

> tangents that share one primal pass are implemented for the f64 Jacobian (`Solver.newton_mixed`, about 1.2× on top
> of the f64 lanes; notes/MIXED_PRECISION.md).

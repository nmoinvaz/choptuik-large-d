# Mixed-precision Newton for the double-double solver (`hp/`)

A Newton step of the double-double DSS solver spent 96% of its time in the exact Jacobian (forward-mode duals over
double-double, one Taylor integration per column; `hp/data/profile_newton.log`). The mixed-precision step keeps the
residual in double-double and computes the Jacobian in plain f64 on SIMD f64 lanes. Newton with an approximate
Jacobian and an exact residual converges to the same root, linearly, so the solution does not change; only the cost
per step drops.

**Result** (D = 3.7, two threads on the cloud machine; section 5):

- The f64 Jacobian takes 1/10.5, 1/9.9 and 1/9.5 of the double-double Jacobian's time at M = 48, 64 and 80, and a
  mixed Newton step 1/9.8, 1/9.7 and 1/9.1 of a double-double step.
- A whole solve from a perturbed state to |F| ≤ 10⁻²⁸ is 8.7, 7.4 and 7.9 times faster, with the same number of
  Newton calls or one more.
- The solution is the same to the precision the equations fix it. Δ agrees to ≤ 1.4×10⁻³⁰ relative, max |z| to
  ≤ 2.3×10⁻²⁷, where two exact-Jacobian solves from two different starts differ by up to 1.4×10⁻³⁰ and 2.2×10⁻²⁷.
- The default double-double path is bitwise unchanged.

The f64 Jacobian agrees with the double-double one to 1.2×10⁻¹⁴ per column at M = 16, but only to 1.6×10⁻⁷ at M = 48,
against the 10⁻¹³–10⁻¹⁵ expected. The Taylor steps are planned for the double-double solution and do not resolve the
top τ-harmonics of a tangent, so the f64 rounding inside a step is amplified (section 3). The iteration's contraction
factor is then about 10⁻⁷ per step, which costs at most one extra Newton call.

## 1. What was changed

Everything new is opt-in. `Solver.jacobian`, `newton`, `chord`, `mode_matrix` and the rest give bitwise the same
results as before (section 4.1).

- **f64 lanes** (`hp/src/lane.rs`), all on the 8-lane `V8` of `v8.rs`, so AVX-512, 2×AVX2 or the portable array as
  before:
  - `F8` (values) and `Fu` (a value and one tangent);
  - `Fun<N>`: an f64 value with N f64 tangents. N Jacobian columns of one block come from one evaluation, and the
    base trajectory is computed once for the N of them;
  - `Dtn<N>`: a double-double value with N f64 tangents. The base trajectory is exact to about 10⁻³², the tangents
    are f64.

  The `Lane` trait gained `VAL_F64`, `TAN_F64`, `NT` (the number of f64 tangents), `tan64` / `set_tan64` and
  `dmat_apply`. Constants and inputs, which the model passes as double-double, enter the f64 lanes as their hi parts.
- **Generic Jacobian** (`hp/src/model.rs`): `jacobian` is now `jacobian_lanes::<Du>` and `jacobian_f64` is
  `jacobian_lanes::<Fu>`. `jacobian_fun::<N>` and `jacobian_dtn::<N>` take the tasks of each block in groups of N
  (`jac_columns_n`). The model's equations (centre series, Taylor stepping, horizon series) are the same generic code
  for every lane type; none of it is duplicated. The lane-specific code is confined to four places:
  - The τ-derivative `DMat::apply` now dispatches to `Lane::dmat_apply`. The double-double lanes keep the old code.
    The f64 lanes multiply by an f64 copy of the matrix (`DMat::cols64`), and the N tangents share each column load.
  - The horizon series' periodic solves (`psolve`). The M × M LUs (`horizon_lus`, ks + 1 of them) stay double-double
    and are computed once per Jacobian. The f64 lanes solve with the hi parts of their factors (`DDLu::solve_hi`),
    which is an f64 LU of the same matrices.
  - The Cauchy sums (`field::cs*`). For the f64 lanes the B lane vectors' accumulations are interleaved, because a
    single chain of f64 FMAs is latency-bound. Each lane vector still sums its terms in the same order.
  - The centre and horizon series stop early for f64 tangents (`Tail` in `model.rs`). They stop once the tangent
    terms max |c_k| r^k of four consecutive orders are all below 2⁻⁶⁰ of the largest so far (r the evaluation
    radius, k ≥ 8). The saved states' orders kc = 100 and ks = 250 are set for double-double values; the f64 tangents
    need 30–40 and 50–60 of them (section 3, "orders"). The horizon block is 35% of the double-double Jacobian's time
    and 8% of the f64 one's (`hp/data/mixed_precision/bench_M48.log`).
- **f64 LU** (`hp/src/linalg.rs`, `DDLu::factor_f64`): the blocked LU of `factor_blocked` in plain f64, with the same
  panels, row swaps, skipped zero blocks and register tiles. It is stored with lo = 0, so the double-double solves
  (`solve`, `solve_t`, `solve_many`, `chord`) use its factors as they are. The route the task suggested, promoting the
  f64 Jacobian to double-double and using the double-double LU, is kept as `lu="dd"`. That LU had become 30–40% of the
  mixed step, and the f64 LU is 2.5–3 times faster (section 5).
- **Python** (`hp/src/python.rs`):
  - `Solver.jacobian_f64(zh, zl, mutate=0, tangents=4, dd_base=False)` returns an (n, n) f64 array, like `jacobian`.
    `tangents` is the number of columns per evaluation (1, 2, 4 or 8; 1 uses `Fu`, the others `Fun<N>`), and
    `dd_base=True` uses `Dtn<N>`. `mutate` breaks the f64 path on purpose for the controls (1: no τ-derivative; 2: ω
    scaled by 1 + 10⁻⁹; 3: the horizon solves' κ tangent term dropped).
  - `Solver.newton_mixed(zh, zl, tangents=4, dd_base=False, lu="f64")` returns the same as `newton`. It computes the
    residual in double-double, the f64 Jacobian and its LU, and solves dz = −J⁻¹F in double-double with those factors.
    The LU is stored for `chord`, `lu_solve` and the rest.
  - `Solver.lu_factor(ah, al=None, f64=False)` factors a given matrix and stores it as the solver's LU. The checks use
    it to solve with a given Jacobian.
- **Driver** (`driver.newton_solve(..., jac="f64", mixed=None)`): the steps use `newton_mixed(**mixed)`, with two
  additions. A step along which the line search finds no decrease is retried once with the exact Jacobian before
  giving up. Once |F| < tol, the step computed at that point is still taken if |F| stays below tol (section 4.4 shows
  why). `jac="dd"`, the default, runs the old loop unchanged.
- **Scripts:**
  - `hp/lowD_dd.py --jac f64`: its plain Newton steps use `newton_mixed`. The deflated (`--solver dn`) and
    Levenberg–Marquardt phases stay double-double. The LU kept for the sensitivities is then the f64 Jacobian's.
  - `hp/profile_newton.py --mixed --solve`: the timings of section 5. Without the flags it prints what it printed
    before.
  - `hp/mixed_check.py` runs the checks of section 4 and `hp/mixed_diag.py` the diagnostics of section 3.
  - `hp/dump_ctx.py` writes a state and its solver set-up for the Rust bench: `FILE=... cargo test --release --lib
    bench_jac -- --ignored --nocapture`, run in `hp/` with `RUSTFLAGS="-C target-cpu=native"`.
- **Rust test:** `linalg::tests::f64_lu_solves` checks the f64 LU against the double-double LU, with one pivot scaled
  by 1 + 10⁻⁶ as the control.

## 2. Variants

| name in the scripts | lanes | per evaluation | base trajectory |
|---|---|---|---|
| `f64x1` | `Fu` | 1 column | f64 |
| `f64x4` (default) | `Fun<4>` | 4 columns of one block | f64, computed once for the 4 |
| `dd4` | `Dtn<4>` | 4 columns of one block | double-double, computed once for the 4 |

`f64xN` gives the same columns as `f64x1` to rounding. The tangent arithmetic is the same, and the series cut waits for
all N tangents. `dd4` gives a Jacobian 400 times closer to the double-double one at M = 48, but Newton converges no
faster with it (section 3) and it costs more, so it is not the default. A `+ddlu` suffix (`newton_mixed` only) factors
with the double-double LU.

## 3. Why the f64 Jacobian agrees to 10⁻⁷ and not to 10⁻¹⁴ at M ≥ 48

The expectation was 10⁻¹³–10⁻¹⁵, and it holds at M = 16 and 32. At M = 48 to 80 the per-column agreement is
3×10⁻⁸–1.6×10⁻⁷, all of it in the Taylor segments. The largest errors are near the horizon side (x = 0.5–0.8) and
around x = 0.1. `python3 mixed_diag.py` gives the following (log in `hp/data/mixed_precision/diag.log`; per-column error
max |dJ| / max |J|; segment k integrates from node k − 1 to node k):

| | centre | segment 4 (x = 0.05–0.1) | segment 5 (0.1–0.16) | segments 6–8 | segments 9–12 (0.4–0.8) | horizon |
|---|---|---|---|---|---|---|
| plan hmax 0.05, f64 base | 2×10⁻¹⁵ | 6×10⁻¹⁰ | 4×10⁻⁹ | ≤ 1×10⁻⁹ | 7×10⁻⁹–2×10⁻⁷ | 1×10⁻¹³ |
| plan hmax 0.05, double-double base | 2×10⁻¹⁵ | 4×10⁻¹¹ | 2×10⁻¹⁰ | ≤ 9×10⁻¹¹ | 2×10⁻¹⁰–4×10⁻¹⁰ | 1×10⁻¹³ |
| plan hmax 0.01, f64 base | 2×10⁻¹⁵ | 6×10⁻¹⁰ | 4×10⁻⁹ | ≤ 7×10⁻¹² | ≤ 1×10⁻¹³ | 1×10⁻¹³ |

| M (D = 3.7) | 16 (from 48) | 32 (from 48) | 48 | 64 | 80 |
|---|---|---|---|---|---|
| f64x4, per column | 1.2×10⁻¹⁴ | 3.3×10⁻¹⁴ | 1.6×10⁻⁷ | 3.3×10⁻⁸ | 7.7×10⁻⁸ |
| dd4, per column | 2.3×10⁻¹⁵ | 8.6×10⁻¹⁵ | 4.4×10⁻¹⁰ | 4.7×10⁻¹⁰ | 6.9×10⁻¹⁰ |

What these show:

1. **Step size.** With the plan's largest step capped at 0.01 instead of 0.05, segments 7–12 agree to ≤ 10⁻¹³. Segments
   4 and 5 stay at 6×10⁻¹⁰ and 4×10⁻⁹. There x − c reaches −0.03, and their steps (at most 0.0055 and 0.0093) were
   below the cap already. The error comes from inside a Taylor step and grows with the step.
2. **Resolution in τ.** With the same kind of plan, the agreement is 10⁻¹⁴ at M = 16 and 32. A tangent that is a unit
   vector on one τ point holds every harmonic up to the top one. The top harmonic k rotates at a rate of about
   ωk/|x − c| in x. The plan (truncation 10⁻³⁰ for the solution) is set by the solution's own coefficients, whose top
   harmonics are about 10⁻³⁰, so for a tangent the step is long. The top harmonics' Taylor terms grow large before
   they decay, and the discrete map even amplifies them, where the exact flow only rotates them. In segment 11 the p
   rows of the column for p at node 10, lane 8, reach 2–8 in harmonics 41–47, against 10⁻² in harmonics 1–31, and
   that column's maximum is 31. The double-double Jacobian is the exact derivative of that same discrete map, so the two
   Jacobians should agree to rounding, but the rounding inside a step is multiplied by these intermediate terms.
3. **Which rounding.** With the base trajectory in double-double the error falls from 1.6×10⁻⁷ to 4.4×10⁻¹⁰, so most
   of it was the f64 base trajectory's rounding. It sits in the a rows and in the low harmonics: the error peaks at
   harmonics 2–8, while the column itself is mostly top harmonics. Products of large top-harmonic tangent terms with
   the base trajectory's rounding noise alias into low harmonics, where nothing cancels them. The remaining 4×10⁻¹⁰ is
   the f64 tangent's own rounding, in the p rows. Summing each step's Taylor series in double-double (a development
   experiment, not kept) changed nothing: the error is in the coefficients, not in their sum.
4. **What it costs Newton.** One step from a perturbed state with each Jacobian, both solved in double-double, differs
   by 6×10⁻⁸ for `f64x4` and 9×10⁻⁸ for `dd4`, and by 5×10⁻¹⁴ when the double-double Jacobian is only rounded to f64
   (section 4.2). In the convergence runs |dz| falls by about 10⁻⁷ per step once the quadratic part is spent, for all
   variants. So the mixed Newton needs at most one more call, and the more accurate `dd4` Jacobian does not remove it.
   That points to the tangent's own rounding in the top harmonics, which `dd4` keeps, as what sets the contraction
   (an inference from these numbers, not tested separately).
5. **Orders.** The horizon block keeps its 1.3×10⁻¹³ agreement down to ks = 60 (5×10⁻¹⁰ at 40), and the centre
   block its 2×10⁻¹⁵ down to kc = 40 (10⁻¹³ at 30, 10⁻⁶ at 20). The adaptive cut (section 1) uses this.

Not done: a finer Taylor plan for the whole solver, which would make the discrete map resolve the top harmonics and
bring the f64 Jacobian to 10⁻¹³. Sub-stepping only the f64 Jacobian would differentiate a different discrete map and
break the agreement in exactly those directions.

## 4. Verification

All runs were on the cloud machine (2 cores, AVX-512) with two threads and a private build of this branch; the
commands run in `hp/`. The logs and JSON records are in `hp/data/mixed_precision/`.

### 4.1 The default path is bitwise unchanged

`python3 mixed_check.py ref --save=REF.npz` was run with the build of the parent commit (52c6a65), and
`--compare=REF.npz` with this branch's build (`ref_save_parent_build.log`, `ref_compare.log`). The 17 arrays compared,
at the D = 3.7, M = 48 state, are:

- the residual and the Jacobian (hi and lo);
- the Newton step and its pivots, the chord step after it, and `lu_solve_t`;
- `mode_matrix` at λ = 0.3;
- a Jacobian with frozen harmonics (`kfreeze` 13) and one with an inward matching node (`match_node` 8).

All 17 are bit-identical. `cargo test --release` passes 3 tests (the two existing LU tests and the new f64 LU test) and
ignores the 2 benches (`cargo_test.log`). `hp/proto/test_dd.py`, the existing check of the double-double operations
against mpmath, gives 1.2–1.6×10⁻³² as before (`proto_test_dd.log`).

### 4.2 Jacobian agreement (`python3 mixed_check.py agree [--strict] --variant=...`)

Per column means max |J_f64 − J_dd,hi| / max |J_dd,hi| over the column. The pass criteria were written before the
runs.

| tier | state | variant | per column | Frobenius | step | pass if ≤ |
|---|---|---|---|---|---|---|
| strict | M = 48 resampled to 16 | f64x4 | 1.2×10⁻¹⁴ | 1.6×10⁻¹⁵ | 2.1×10⁻¹³ | 10⁻¹² |
| strict | M = 48 resampled to 16 | f64x1 | 1.2×10⁻¹⁴ | 1.6×10⁻¹⁵ | 2.1×10⁻¹³ | 10⁻¹² |
| strict | M = 48 resampled to 16 | dd4 | 2.3×10⁻¹⁵ | 3.7×10⁻¹⁶ | 1.2×10⁻¹³ | 10⁻¹² |
| strict, `kfreeze` 13 | M = 48 resampled to 16 | f64x4 | 1.2×10⁻¹⁴ | 1.6×10⁻¹⁵ | 4.3×10⁻¹⁴ | 10⁻¹² |
| strict, `match_node` 8 | M = 48 resampled to 16 | f64x4 | 1.2×10⁻¹⁴ | 1.7×10⁻¹⁵ | 9.9×10⁻¹⁴ | 10⁻¹² |
| production | M = 48 | f64x4 | 1.6×10⁻⁷ | 8.7×10⁻⁸ | 6.1×10⁻⁸ | 10⁻⁶ |
| production | M = 48 | f64x1 | 1.6×10⁻⁷ | 8.7×10⁻⁸ | 5.2×10⁻⁸ | 10⁻⁶ |
| production | M = 48 | dd4 | 4.4×10⁻¹⁰ | 1.0×10⁻¹⁰ | 8.8×10⁻⁸ | 10⁻⁶ |
| production, `kfreeze` 13 | M = 48 | f64x4 | 1.6×10⁻⁷ | 8.7×10⁻⁸ | 5.0×10⁻⁷ | 10⁻⁶ |

"Step" is the relative difference of −J⁻¹F between the two Jacobians, with F the residual at a perturbed state
(Δ + 10⁻⁶, fields + 10⁻⁸) and both solved in double-double. It estimates the mixed iteration's contraction factor.
With the double-double Jacobian rounded to f64 it is 8×10⁻¹⁵ at M = 16 and 5.3×10⁻¹⁴ at M = 48.

Entrywise, on the nonzero entries, the median is 4.7×10⁻¹⁶ (M = 16) and 1.1×10⁻¹² (M = 48, f64x4). The maximum is
large (1.8×10⁻¹¹ at M = 16, 1.5 at M = 48) on entries that are tiny in their column; the worst at M = 48 is
|J| = 1.3×10⁻⁹ in a column of maximum 1.0. That is cancellation in single entries. No entry is nonzero in one Jacobian
and zero in the other.

The two tiers have different jobs. The strict tier is the bug detector: an f64 path whose rounding is not amplified.
The production tier measures what Newton sees. Its threshold of 10⁻⁶ replaced the 10⁻¹² first written for it, after the
cause of the 1.6×10⁻⁷ was found (section 3). So the original expectation was not met at M ≥ 48, and the subtle control
(ω × (1 + 10⁻⁹)) only fires in the strict tier.

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

The κ mutation changes only the horizon block's 2M + 1 columns, so the check also fires on an error confined to one
block. The f64 LU's control (a pivot scaled by 1 + 10⁻⁶) moves the solution by 4×10⁻⁸–7×10⁻⁷, against 2×10⁻¹⁵–1.4×10⁻¹⁴
without it.

### 4.4 Convergence to the same solution (`python3 mixed_check.py converge`)

Each run starts from the D = 3.7 state with Δ moved by 10⁻⁶ and every field entry by a pseudo-random amount up to 10⁻⁸
(seed 1), and runs `driver.newton_solve` (Armijo backtracking) to |F| ≤ 10⁻²⁸. The Newton calls include the last
one, which only confirms |F| ≤ 10⁻²⁸. The control is a second exact-Jacobian solve from another start (Δ − 10⁻⁶,
seed 2); it measures how precisely the equations fix the solution. Pass: the differences are at most
max(10⁻²⁷, 10 × the control).

| M | start | variant | Newton calls | final F | seconds | Δ difference | max \|z\| difference |
|---|---|---|---|---|---|---|---|
| 64 | its state, F 5.7×10⁻⁷ | exact (`newton`) | 5 | 9.6×10⁻³⁰ | 80.5 | | |
| 64 | | f64x4 | 6 | 6.1×10⁻³⁰ | 11.3 | 4.7×10⁻³¹ | 1.1×10⁻²⁷ |
| 64 | | f64x4+ddlu | 6 | 3.5×10⁻³⁰ | 13.0 | 3.0×10⁻³¹ | 9.1×10⁻²⁸ |
| 64 | | f64x1 | 6 | 6.9×10⁻³⁰ | 14.1 | 1.4×10⁻³⁰ | 6.9×10⁻²⁸ |
| 64 | | dd4 | 6 | 1.5×10⁻²⁹ | 17.4 | 9.6×10⁻³¹ | 1.0×10⁻²⁷ |
| 64 | | control (exact, second start) | 5 | 4.6×10⁻³⁰ | | 1.4×10⁻³⁰ | 6.7×10⁻²⁸ |
| 64 | M = 48 state resampled, F 7.4×10⁻⁴ | exact | 5 | 4.4×10⁻³⁰ | 218.5 | | |
| 64 | | f64x4 | 5 | 3.0×10⁻³⁰ | 33.5 | 8.3×10⁻³¹ | 2.3×10⁻²⁷ |
| 64 | | control | 5 | 4.6×10⁻³⁰ | | 1.3×10⁻³⁰ | 2.2×10⁻²⁷ |
| 80 | its state, F 7.5×10⁻⁶ | exact | 7 | 1.9×10⁻²⁹ | 209.4 | | |
| 80 | | f64x4 | 7 | 1.2×10⁻²⁹ | 27.7 | 3.6×10⁻³¹ | 7.3×10⁻²⁸ |
| 80 | | f64x4+ddlu | 7 | 5.1×10⁻³⁰ | 31.9 | 3.3×10⁻³¹ | 3.6×10⁻²⁸ |
| 80 | | control | 7 | 1.1×10⁻²⁹ | | 5.0×10⁻³¹ | 6.1×10⁻²⁸ |

All pass. The z differences relative to max |z| meet 10⁻²⁷ except at the resampled start, where the two
exact-Jacobian solves themselves differ by 2.2×10⁻²⁷. The solution's near-null centre directions are fixed only to
about that level by |F| ≈ 10⁻²⁹, so 10⁻²⁷ cannot be asked of z there, even between two exact runs.

At M = 64 the mixed runs follow the exact run step for step: |F| goes 5.7×10⁻⁷ → 5.7×10⁻⁷ → 4.0×10⁻¹¹ → 1.6×10⁻¹⁷.
Then the exact run reaches 10⁻²⁹ in one step, and the mixed run takes two (1.6×10⁻¹⁷ → 3.6×10⁻²⁷ → 7.6×10⁻³⁰).

**A discrepancy found and resolved.** The first M = 80 run stopped as soon as |F| ≤ 10⁻²⁸, as the exact Newton loop
does (`converge_M80_without_last_step.log`). Its z differed from the exact solution by 1.1×10⁻²⁶ (f64x4) and
1.6×10⁻²⁶ (+ddlu), against 6.1×10⁻²⁸ for the control, so the check failed. Δ agreed to 1.3×10⁻³⁰.

The cause is the linear convergence. The last mixed step leaves an error of about 10⁻⁷ times the previous step
(1.4×10⁻¹⁷ → about 10⁻²⁴), in the near-null directions, which |F| (3×10⁻³⁰) does not see. The step computed at the
converged point showed it: |dz| = 2.6×10⁻²⁵, against 1.7×10⁻²⁶ for the exact run. An exact run has no such error,
because its last step is quadratic.

The fix is in `newton_solve(jac="f64")`: at convergence it takes that already-computed step if |F| stays below tol,
which costs one residual. With it, M = 80 agrees to 7.3×10⁻²⁸. The exact Newton loop is unchanged.

### 4.5 `lowD_dd.py --jac f64`

This test is a continuation step from the D = 3.7, M = 48 state to D = 3.69, with the state's settings, run once
without and once with `--jac f64` (in `hp/`; each run writes its state to `hp/data/lowD_states/<tag>_D3.6900_M48.npz`,
which was not kept):

```
python3 lowD_dd.py OUT.jsonl --start data/lowD_states/lowDv_D3.7000_M48.npz --D 3.69 --Ms 48 --nodes 0.005,0.01,0.02,0.05,0.1,0.16,0.23,0.3,0.4,0.5,0.6,0.7,0.8 --kcs 100 --ks 250 --tol_step 1e-30 --tol 1e-28 --tag mixedsmoke_dd
python3 lowD_dd.py OUT.jsonl --start data/lowD_states/lowDv_D3.7000_M48.npz --D 3.69 --Ms 48 --nodes 0.005,0.01,0.02,0.05,0.1,0.16,0.23,0.3,0.4,0.5,0.6,0.7,0.8 --kcs 100 --ks 250 --tol_step 1e-30 --tol 1e-28 --tag mixedsmoke_f64 --jac f64
```

The logs and records are `lowD_dd_jac_dd.*` and `lowD_dd_jac_f64.*`. Both runs start from |F| = 7.4×10⁻³ and take 6
Newton calls, with the same |F| at each call down to 10⁻²³.

| | `--jac dd` | `--jac f64` |
|---|---|---|
| final F | 6.6×10⁻³⁰ | 4.0×10⁻³⁰ |
| Δ | 3.464558395113743 + 1.7408255559455386×10⁻¹⁷ | 3.464558395113743 + 1.740825555945719×10⁻¹⁷ |
| max dΔ/dF (from the stored LU) | 2.210161106761364 | 2.210161106758648 |
| seconds (with the sensitivities) | 93.4 | 12.7 |

The two Δ differ by 1.8×10⁻³¹ (5×10⁻³² relative). The sensitivities, computed with the f64 Jacobian's LU, differ by
1.2×10⁻¹².

The first attempt omitted `--nodes`, `--kcs` and `--ks`, so `lowD_dd.py` re-noded the start to its default nodes. From
that start (|F| 7.4×10⁻³, a nearly singular Jacobian with pivots up to 4.6×10³) neither `--jac dd` nor `--jac f64`
converged in 14 iterations (`lowD_dd_renoded_start_jac_*.log`). The two runs followed each other for three iterations
and then took different line-search paths. This is not a test of the mixed step. It records that far from the
solution, where the Jacobian is nearly singular, the two Jacobians' steps can differ visibly.

## 5. Timings

`python3 profile_newton.py --mixed --solve` (D = 3.7, two threads; `profile.log`, `profile.json`):

| M (N) | 48 (2689) | 64 (3585) | 80 (4481) |
|---|---|---|---|
| double-double Jacobian | 7.5 s | 16.8 s | 31.0 s |
| f64 Jacobian, f64x1 | 0.97 s (7.8×) | 2.28 s (7.4×) | 3.77 s (8.2×) |
| f64 Jacobian, f64x4 (default) | 0.72 s (10.5×) | 1.70 s (9.9×) | 3.26 s (9.5×) |
| f64 Jacobian, dd4 | 0.99 s (7.6×) | 3.39 s (5.0×) | 5.32 s (5.8×) |
| f64 LU of the f64 Jacobian | 0.14 s | 0.39 s | 0.50 s |
| double-double LU of it (from the `+ddlu` step) | 0.43 s | 1.14 s | 1.26 s |
| Newton step, double-double | 7.8 s | 19.5 s | 32.2 s |
| `newton_mixed` (f64x4, f64 LU) | 0.80 s (9.8×) | 2.01 s (9.7×) | 3.54 s (9.1×) |
| `newton_mixed` (f64x4, dd LU) | 1.09 s (7.2×) | 2.76 s (7.1×) | 4.30 s (7.5×) |
| whole solve to 10⁻²⁸, exact | 56.8 s (7 calls) | 92.2 s (5) | 218.9 s (7) |
| whole solve, `newton_mixed` | 6.5 s (7 calls, 8.7×) | 12.5 s (6, 7.4×) | 27.8 s (7, 7.9×) |
| whole solve, `newton_mixed` + dd LU | 7.4 s (7.7×) | 14.3 s (6.4×) | 31.9 s (6.9×) |

The times vary by about 10% between runs. An earlier run (`profile_earlier_run.log`, cut off at M = 80's whole solve,
before the last-step rule) gave 0.65, 1.61 and 3.05 s for f64x4 and 0.76, 1.82 and 3.32 s for `newton_mixed`. The
double-double LU line is the difference of the two `newton_mixed` steps plus the f64 LU.

Over M = 48–80 the double-double Jacobian grows like M^2.77, the f64x4 Jacobian like M^2.96 and `newton_mixed` like
M^2.93. In the mixed step the Jacobian is now 85–90% and the f64 LU 14–19% (the Jacobian's timing includes the copy
into a numpy array, which `newton_mixed` does not make). The whole-solve gain (7.4–8.7×) is below the per-step gain
(9.1–9.8×) because of the extra call at M = 64 and the double-double residuals of the line search and the last step.

Single-thread totals from the Rust bench (`bench_jac`, a sample of columns per block, extrapolated;
`bench_M48.log`, `bench_M80.log`):

| lanes | M = 48 | M = 80 |
|---|---|---|
| `Du` (double-double, the old Jacobian) | 11.26 s | 48.99 s |
| `Fu` (f64x1) | 1.34 s (8.4×) | 6.38 s (7.7×) |
| `Fun<2>` | 1.10 s (10.2×) | 5.46 s (9.0×) |
| `Fun<4>` (f64x4) | 1.11 s (10.1×) | 5.38 s (9.1×) |
| `Fun<8>` | 1.21 s (9.3×) | 7.21 s (6.8×) |
| `Dtn<1>` | 3.48 s (3.2×) | 17.26 s (2.8×) |
| `Dtn<4>` (dd4) | 1.63 s (6.9×) | 9.00 s (5.4×) |
| `Dtn<8>` | 1.72 s (6.5×) | 9.09 s (5.4×) |

The gain is about 10×, the order of magnitude expected, but it does not come from arithmetic alone. Per term of a
Cauchy sum the f64 lanes do 3 FMAs where the double-double lanes do about 30 operations, but the f64 loops are bound by
FMA latency and by L2 bandwidth: the coefficient arrays (12 fields × 38 orders × M points) do not fit in L1. The first
commit of this branch (444377d: f64 lanes with interleaved Cauchy sums, before the series cut) measured 5.5× at M = 48
(`bench_M48_444377d.log`); the series cut brought it to 8.4× and batching to 10×.

### Batching (stage 2), separately

Several tangents per base trajectory (`Fun<N>`, `Dtn<N>`) gain, at the same accuracy:

- With an f64 base (`Fu` → `Fun<4>`): 1.21× at M = 48 and 1.19× at M = 80 on one thread, and 1.35×, 1.34× and 1.16×
  at M = 48, 64 and 80 on two (f64x1 → f64x4 in the table above). It is small because the f64 base trajectory is
  cheap, and the τ-derivative becomes a product with N + 1 vectors sharing each matrix column, not a matrix–matrix
  product with large reuse.
- With a double-double base (`Dtn<1>` → `Dtn<4>`): 2.1× at M = 48 and 1.9× at M = 80. There the base trajectory is
  most of the work.

N = 8 is slower than N = 4 at M = 80: the coefficient arrays then exceed L2.

## 6. What remains

- **Finer Taylor plans** for the top τ-harmonics (section 3). They would bring the f64 Jacobian to 10⁻¹³, and the
  contraction with it, and make the discrete map resolve those directions. They would also cost the residual steps.
- **The τ-derivative by FFT**, O(M log M) per application instead of the dense M × M product. Its share of the f64
  Taylor stepping grows with M, and the f64 Jacobian grows like M^2.96 against M^2.77 for the double-double one.
- **Memory traffic.** The f64 Cauchy sums are limited by L2 bandwidth, not arithmetic. A layout by lane vector (all
  orders of one lane vector together), or a tangent-only linearised recursion over stored base coefficients, would
  cut it.
- **The rest of the step:** the f64 LU (now 14–19% of the mixed step), and the line search's double-double residuals
  and the Python loop, which show in the whole-solve times.
- **Untested paths:** f64 lanes with λ ≠ 0 (a mode matrix in f64) are implemented in `dtau` but not exposed or tested.
  The deflated-Newton and LM phases of `lowD_dd.py` stay double-double.

## 7. Proposed text for the README (for the coordinator)

In "The double-double solver" list, after "Exact Jacobian":

> - **Mixed-precision Newton:** `Solver.newton_mixed` keeps the residual in double-double, computes the Jacobian with
>   f64 vector lanes (four columns per pass over the base trajectory) and factors it in f64. Newton then converges
>   linearly to the same solution, with a contraction of about 10⁻⁷ per step and at most one extra step. A step is
>   about 9–10 times faster at M = 48–80, and a whole solve 7–9 times
>   ([notes/MIXED_PRECISION.md](notes/MIXED_PRECISION.md); `hp/lowD_dd.py --jac f64`).

In the Layout block under `hp/`, change two lines and add three:

```
├── src/lane.rs     value and forward-mode dual lanes: double-double, f64, and batched f64 tangents
├── src/linalg.rs   double-double LU, f64 LU, transpose solve, Gram matrix
├── mixed_check.py  checks of the mixed-precision Newton: Jacobian agreement, controls, convergence, bit-identity
├── mixed_diag.py   why the f64 Jacobian agrees to 1e-7 at M >= 48 (steps, harmonics, M, series orders)
├── profile_newton.py   time of a Newton step by part; --mixed --solve for the mixed-precision step
```

In the paragraph on the Newton step's cost, replace "tangents that share one primal pass would save about 1.7×, not
implemented" with:

> tangents that share one primal pass are implemented for the f64 Jacobian of the mixed-precision Newton step (about
> 1.2–1.35× on top of the f64 lanes; [notes/MIXED_PRECISION.md](notes/MIXED_PRECISION.md))

## 8. Large D (added by the coordinator, 2026-10-22)

Does the f64 Jacobian still steer where the Jacobian has directions 10¹⁷–10²⁴ times weaker than its strongest?
`hp/mixed_largeD.py` (log and JSON in `hp/data/mixed_precision/largeD.*`) takes the saved scan states at D = 8.25–60
(M = 32, or 48 at 8.25; from D = 40 with the p1 harmonics above 13 frozen, as the scan ran). At each state it
perturbs the fields by 10⁻¹² relative and compares Newton steps with the double-double step:

| D | σ_min/σ_max | f64 LU of the dd Jacobian's hi part | f64 Jacobian | f64 tangents on a dd base | Jacobian time: dd / f64 |
|---|---|---|---|---|---|
| 8.25 | 5×10⁻¹² | 5×10⁻⁷ | 2.6×10⁻³ | 6.8×10⁻³ | 8.8 / 1.5 s |
| 16.76 | 6×10⁻¹⁸ | 5×10⁻⁶ | 8.6×10⁻⁴ | 2.2×10⁻⁴ | 2.2 / 0.44 s |
| 20.04 | 2×10⁻²⁰ | 2×10⁻⁶ | 1.5×10⁻⁴ | 2.6×10⁻⁴ | 2.1 / 0.29 s |
| 25.08 | 7×10⁻²⁴ | 2×10⁻⁵ | 0.14–0.97 | 0.17 | 4.7 / 0.29 s |
| 30.07* | 4×10⁻²⁴ | 8×10⁻⁶ | 5.7×10⁻³ | 2.4×10⁻² | 6.5 / 0.33 s |
| 40.02 | 2×10⁻¹⁸ | 3×10⁻⁶ | 1.2–2.9×10⁻³ | 2.4×10⁻⁴–1.7×10⁻³ | 6.0 / 0.22 s |
| 45.05 | 2×10⁻¹⁸ | 1×10⁻⁷ | 2.6×10⁻³–1.4×10⁻² | 5.7×10⁻³–1.1×10⁻² | 6.3 / 0.25 s |
| 50.01 | 8×10⁻¹⁸ | 5×10⁻⁶ | 0.03–2.2 | 0.09–0.24 | 6.6 / 0.26 s |
| 55.04 | 3×10⁻¹⁸ | 2×10⁻⁶ | 1.2–2.8 | 0.47–0.69 | 6.1 / 0.19 s |
| 60.00 | 2×10⁻¹⁸ | 7×10⁻⁶ | 0.73–13 | 0.58–1.0 | 6.2 / 0.27 s |

The step columns give the relative difference from the double-double step. That difference is the contraction factor
of the mixed Newton iteration: below 1 it converges, gaining −log₁₀ of it in digits per step once the quadratic part is
spent. Ranges are over three draws of the random perturbation for the f64 Jacobian (`largeD_run1`, `largeD_run2`, `largeD`) and two for the dd base (`run1`, `largeD`). The Jacobian times are on 2 threads shared with
two other jobs. \*At D = 30.07 these solver settings leave |F| = 3×10⁻¹² at the saved state, so the scan used something
else there; the linear algebra is still representative.

What this shows:

1. Up to D ≈ 45 the f64 Jacobian steers: the step is within 0.02–1.4% of the exact one, except near D = 25, before the
   harmonics are frozen. Newton from the perturbed state converged to full accuracy with it at D = 8.25, 16.76 and
   20.04: 4, 6 and 8 iterations against 3, 5 and 6, in 5.9, 2.5 and 3.6 s against 29, 12 and 18 s. Above D = 20 the
   10⁻¹² perturbation was outside even the double-double plain Newton's basin; the scan uses deflated Newton there.
   So whole solves above 20 still need testing, with smaller perturbations or the deflated Newton.
2. At D = 50–60 it does not reliably steer: the contraction is 0.03 to 13 depending on the start.
3. The f64 factorization is not the problem: an f64 LU of the exact Jacobian's hi part gives the step to 10⁻⁷–10⁻⁵ at
   every D. The problem is the f64-built Jacobian's entries. Its errors are small against the largest entries
   (10⁻¹³ to 2×10⁻⁷ as D grows) but large against the small entries that carry the weak directions. A double-double
   base trajectory makes the entries 200–1000 times more accurate, yet the step error at D = 55–60 stays at 0.5–1.
4. The error lives in 2–4 directions: the singular values of X_f64 − X_dd over 48 right-hand sides fall by 10³–10⁴
   after the second to fourth. Correcting a handful of directions exactly would repair the step. Two ways:
   (a) a few Krylov (GMRES) steps with the f64 LU as preconditioner and exact double-double Jacobian-vector
   products, each costing about one dual pass per segment (a few % of a double-double Jacobian), which needs a JVP in
   the kernel; (b) the existing deflated Newton (`driver.deflated_newton`), solving the weak subspace's part in
   double-double and the rest in f64. Each would bring a large-D Newton step from about 6.5 s to under 1 s.
5. A cheaper step does not remove the climb's real cost above D ≈ 60: the steps in D shrink there, to 0.0015–0.005
   (22 h per unit of D). Faster steps would cut that time by the same factor if the number of steps stays the same.
6. Where the error lives (`hp/mixed_largeD_blocks.py`, log `hp/data/mixed_precision/largeD_blocks.log`). At
   D = 50, 55 and 60 (two perturbations each), the f64 Jacobian gets one block replaced by the exact one, first a
   block of 4M = 128 rows, then a block of 128 columns. Replacing the horizon block's rows (residual block 5 of 6: the
   matching to the horizon series) brings the step error from 0.13–3.4 down to 4×10⁻⁷–6×10⁻⁶, the floor of an f64
   LU. Replacing any other block of rows or columns changes nothing, or makes it worse. So the f64 errors that matter
   at large D are all in the horizon block. A likely cause, not yet checked: at large D the solution varies over a
   thin layer near the horizon, so the last segment and the horizon series carry large intermediate terms, as the top
   harmonics do at M ≥ 48 (section 3). The simplest fix is to compute the horizon block's Jacobian in double-double
   and the rest in f64.
7. The correction loop, prototyped (`hp/mixed_largeD_gmres.py`, log `hp/data/mixed_precision/largeD_gmres.log`).
   GMRES on J M⁻¹ (right preconditioning), with M the f64 LU of the f64-built Jacobian. As a stand-in for an exact
   Jacobian-vector product rounded to f64, it multiplies by the double-double Jacobian's hi part. At D = 50–60, a
   GMRES tolerance of 10⁻⁸ takes 4–5 products and brings the step error from 0.9–3.4 down to 3×10⁻⁷–5×10⁻⁶. Once,
   at D = 55, the 10⁻⁸ tolerance stopped at 4 products with the step still off by 6×10⁻⁴, and the other start needed
   27 products. The residual is not a safe stopping test here, because an error in a weak direction barely shows in
   the residual. The loop should stop on the change in the step instead, or on a fixed count followed by a check. A
   tolerance of 10⁻¹² is not reachable with an f64 operator: the floor is about 10⁻¹⁰ relative. At D = 20–45 it takes
   3 products. The plan is to build both fixes: the horizon block in double-double, and this loop on top, with an
   exact double-double Jacobian-vector product (one dual pass per segment) in the kernel.

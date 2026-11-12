# What of hp could move into multidouble (audit of 2026-11-04)

File and line references here are to hp at 9e63140, before #523 moved `qsolve.rs`, `osolve.rs`, `xsolve.rs` and
`block.rs` into multishoot's engine.

## Recommendation

Move one piece of Rust, the order-by-order solve of a jet lane in `model.rs` `psolve` (its two jet branches), into
multidouble's `FieldSolve` as `solve_lanes_jet` and `solve_lanes_jet_refine`. Add in the same change a guard that
makes `solve_lanes_refine_t` refuse a jet lane, which it now accepts and silently truncates to orders 0 and 1. Put two
small Python-package changes in the same release, a double-double Horner (`dd_horner`) and input coercion for every
array function. Release it as multidouble 2.1.0, a minor release. The new methods are additive, and the guard only
turns a silent truncation into a panic.

The cost is about one day in multidouble (code, docs, tests with controls, clippy on each backend, release) and half a
day in hp (bump, two branches replaced by calls, the jet hashes added to `highm_check.py ref`, the checks run). Both
figures are estimates. `hp/Cargo.toml`'s version and `hp/pyproject.toml`'s pinned commit both move to 2.1.0. The
bump's checks need hp's release module, so they run on the Mac or Windows, not the cloud.

The move cannot change a bit if the closures keep hp's operation order, which the proposed API leaves to the caller.
The existing bit-identity checks would not catch a mistake, though, because `highm_check.py ref` and `scaled` never
run a jet. So the first step, before any multidouble change, is to add `residual_jet` and `residual_jet_q` hashes to
`highm_check.py ref` and save them with the 2.0.0 build.

Everything else generic in hp is either too small to be worth a release (a few lane conveniences, listed below,
optional riders on the same release) or is duplication between hp's own scripts. That duplication belongs in hp's
`driver.py` and needs no multidouble release. The rest of hp is the model and stays.

No part of this is urgent. hp's jet code works and has tests (`qsolve_jet_*`, `hp/qjet_check.py`), and the runners use
the current build. The jet solve is worth moving when the next multidouble release is made anyway, or when
feat/qd-jet and feat/jet-all-sizes merge, whichever comes first.

## Scope

Read against hp on feat/qd-jet (76c1065, multidouble =2.0.0, Python package at 454b8ca), the change of
origin/feat/jet-all-sizes to `python.rs` (`residual_jet_sized` generic in B), hp/modecount_cx, and multidouble at tag
2.0.0 (454b8ca). Nothing was built, and no solver code was run (one state's metadata was read). Every bit-identity
statement below is by construction and unverified.

## Ranked candidates

| # | Candidate | Where | Generic or model | multidouble has | Verdict |
|---|---|---|---|---|---|
| 1 | Order-by-order jet solve, plus the jet guard | `model.rs` 491-514, 528-546 | generic | no. `FieldSolve` stops at `Du`/`Qu`, and `solve_lanes_refine_t` truncates jets | move, 2.1.0 |
| 2 | Double-double Horner at an f64 point | `anm_run.py` 99-105, and two f64-power variants | generic arithmetic | no | Python package, 2.1.0 |
| 3 | Coercion of array inputs to contiguous float64 | multidouble `python/pysrc/multidouble/__init__.py` | reverse item | only `dd_dot` coerces | Python package, 2.1.0 |
| 4 | `Fl::set_tan64`, unit-tangent seeding, `DD`/`QD` slice split and join | `model.rs` 182-183, 568, 896-923; `qsolve.rs` 175-197, 288; `python.rs` | generic lane machinery | the getter `Fl::tan64` only | optional rider on 2.1.0 |
| 5 | Levenberg-Marquardt step family | `python.rs` 1070-1115 | generic | `linalg::gram` only | leave |
| 6 | Complex-shifted field solve in real 2N form | `modecount_cx/src/lib.rs` 282-392 | generic | `clu_logdet` (determinant only) | leave |
| 7 | Tiny helpers: `recip_int`, `phase_row`/`phase_q`, `plan_path`'s step rule, the matrix assembly of `horizon_lus` | `model.rs` 165, 748, 416-445, 450-487; `qsolve.rs` 48 | generic, small | `DD::recip`, `DotAcc`, `QdAcc` | leave |
| - | Duplicates between hp's own scripts | `driver.py`, `anm_*.py`, `qjet_check.py`, `pinning.py`, `lowD_dd.py` | model-agnostic, Solver-specific | no | consolidate in `driver.py`, no release |

### 1. The order-by-order jet solve

**What it is.** `psolve` solves the horizon series' periodic systems (omega D + diag kappa) f = r. For a jet lane
(`Lane::NJ > 0`) the matrix depends on the jet variable, A(h) = A_0 + sum_j A_j h^j, and hp solves order by order,
A_0 f_k = r_k - sum_{j=1..k} A_j f_{k-j}. The double-double branch (`Jet<N>`, lines 528-546) solves each order with the
LU at the values. The quad-double branch (`QJet<N>`, lines 491-514) forms each right-hand side in quad-double and
refines each order's solve against A_0 (`solve_lanes_refine`, two steps, controls `QD_MUT_NOREFINE`/`QD_MUT_REFINE4`).
Only the operator (omega_j D + kappa_j) is the model's. The recurrence, the coefficient bookkeeping and the refinement
are lane machinery, the jet counterpart of `solve_lanes_refine_t`.

**Why it is generic.** Any series method that runs on a jet lane and has an implicit linear solve inside needs this
recurrence. multidouble's own docs already describe it (`Lane::VAL_QD`: "a jet's solve goes order by order";
`Lane::NJ`: "code with a branch for jets, such as an order-by-order solve, tests it") and leave it to the caller.

**The trap it closes.** `solve_lanes_refine_t` accepts a jet lane, but `Jet::from_vt` and `QJet::from_vt` set orders 0
and 1 only, so orders 2 and up come back zero without an error. hp is not affected because `psolve` branches on `NJ`
first. A guard `assert!(T::NJ == 0, ...)` matches the existing `assert!(!T::TANGENT)` of `solve_lanes_refine`.

**API in multidouble** (`field::FieldSolve`, implemented for `DDLu`, `self` the LU of A_0):

```rust
/// x(h) with A(h) x(h) = rhs(h) for a double-double jet lane (Lane::NJ > 0, panics otherwise): order by order,
/// x_k = A_0^-1 r_k, r_k = couple(k, x_0, ... couple(1, x_{k-1}, rhs_k)), `couple(j, x, r)` returning r - A_j x
/// formed by the caller, so the caller fixes the order of the operations
fn solve_lanes_jet<T: Lane, const B: usize>(
    &self,
    rhs: &Fl<T, B>,
    couple: impl FnMut(usize, &Fl<DV<T::S>, B>, Fl<DV<T::S>, B>) -> Fl<DV<T::S>, B>,
) -> Fl<T, B>;
/// the same with the coefficients in quad-double (Lane::coef_qd, set_coef_qd), each order's solve refined `steps`
/// times by solve_lanes_refine with `resid0(r, x)` = r - A_0 x in quad-double
fn solve_lanes_jet_refine<T: Lane, const B: usize>(
    &self,
    rhs: &Fl<T, B>,
    steps: usize,
    couple: impl FnMut(usize, &Fl<QV<T::S>, B>, Fl<QV<T::S>, B>) -> Fl<QV<T::S>, B>,
    resid0: impl FnMut(&Fl<QV<T::S>, B>, &Fl<QV<T::S>, B>) -> Fl<QV<T::S>, B>,
) -> Fl<T, B>;
```

The closure returns the updated right-hand side rather than A_j x. hp's branches compute (r - omega_j D x) - kappa_j x,
and r - (omega_j D x + kappa_j x) would round differently. hp's call sites would become

```rust
if T::NJ > 0 && T::VAL_QD {
    let om: Vec<QV> = (0..T::NJ).map(|j| omega.coef_qd(j)).collect();
    let k0 = kappa.coef_qd(0);
    return lu.solve_lanes_jet_refine(rhs, steps,
        |j, x, r| (r - ctx.d.apply(x).mul_t(om[j])) - kappa.coef_qd(j) * *x,
        |r, x| *r - (ctx.d.apply(x).mul_t(om[0]) + k0 * *x));
}
if T::NJ > 0 {
    return lu.solve_lanes_jet(rhs, |j, x, r| (r - ctx.d.apply::<DV>(x) * Fl([omega.coef(j); B])) - kappa.coef(j) * *x);
}
```

Two methods rather than one, because a double-double jet must keep its arithmetic in `DV`. Working in `QV` for both
would change the bits of `residual_jet`.

**Tests in multidouble** (tests/field.rs or tests/jet.rs), each accuracy check with a control that fails.

| Test | Pass | Control that must fail |
|---|---|---|
| Planted A(h) = A_0 + h A_1 + h^2 A_2 (random `DMat`s, A_0 diagonally dominant), planted x(h), rhs(h) by exact (`Dy`) products | `Jet<8>` x_k to double-double, `QJet<8>` to quad-double, relative to the planted ones | `couple` that skips j = k; `steps = 0` on `QJet` (stays at double-double) |
| `Jet<2>` against the same recurrence written out on `Du` lanes (x_v = A_0^-1 r_v, x_d = A_0^-1 (r_d - A_1 x_v), as hp's `Du` branch does) | bit for bit | a sign flipped in `couple` |
| The guard | `solve_lanes_refine_t` on `Jet<3>` panics (`should_panic`) | none needed (a panic test) |

**Bits in hp.** None change if the closures are written as above, since the operations and their order are hp's
current ones and Rust does not contract multiply-adds. The existing checks cannot confirm it. `highm_check.py ref` and
`scaled` hash the residual, the Taylor plan, the series, the dense Newton steps, the f64 Jacobian and `newton_mixed`,
but never call `residual_jet` or `residual_jet_q`. hp's jet tests (`qsolve_jet_sums_to_the_residual`,
`qsolve_jet_agrees_with_double_double_jet`, `qsolve_jet_resolves_sub_double_double_changes`) compare to tolerances
(1e-28 relative for two of them) and would miss a rounding-level change. The step that closes the gap is a `ref`
addition, hashes of `residual_jet` orders 0..7 on the state jet (z, v) with a fixed v at M = 32, 48 and 64 (the sizes
feat/qd-jet builds the jet for; with feat/jet-all-sizes also M = 8, 16 and 80) and of `residual_jet_q` at M = 32 and
48. They are saved with the 2.0.0 build before the bump and compared after it. Their control is `Ctx::mutate` 22
(`JET_MUT_KAPPA`), which must change the hashes.

**Gain for others.** It completes `Jet<N>`/`QJet<N>` for computations with a linear solve inside, which is the common
case of a Taylor (ANM) continuation, and it removes the truncation trap. multidouble's README names hp as its origin,
and this audit found no other user, so this gain is prospective.

**Effort.** In multidouble about 60-80 lines of code and docs and about 150 lines of tests. In hp, -30 lines in `psolve`
and about 15 lines in `highm_check.py`. The jet-all-sizes change is independent of it. With B generic, the jet solve
is instantiated at every M either way, so the move does not change the release build's memory.

### 2. A double-double Horner in the Python package

`anm_run.py` `series_at` (99-105) evaluates z_0 + sum_k z_k a^k by Horner in double-double, and its docstring explains
why. Float64 powers a**k carry about 1e-16 of themselves into the prediction unless a is a power of two, and
`predicted(..., powers=True)` keeps that variant as a control. Two other scripts evaluate the same series with float64
powers, `anm_step.py` 131-135 and `anm_lowd.py` `evaluate` (feat/jet-all-sizes). This is listed under discrepancies
below.

API: `dd_horner(ch, cl, a)`, ch and cl sequences (or (N + 1, n) arrays) of coefficient arrays, a an f64, returning the
(hi, lo) arrays of sum_k c_k a^k, each step `dd_add(c_k, dd_scale(p, a))` as `series_at` does now. It is arithmetic on
double-double arrays, like `dd_axpy`, so it belongs in multidouble's Python package rather than in `driver.py`.

Tests: against mpmath at 60 digits with random double-double coefficients and a = 0.4341, within (N + 2) 2^-104 of sum
|c_k a^k|; bit for bit against `series_at`'s loop. Control: the float64-power sum must fail the bound.

Bits: `anm_run.py` unchanged. `anm_step.py` and `anm_lowd.py` change, at about 1e-16 of their higher terms, by
design. Neither touches the kernel, so `highm_check.py` is unaffected. Effort: an hour or two.

### 3. Input coercion in the Python package (a reverse item)

multidouble's `qd_axpy`, `qd_add_dd`, `dd_op` and the elementwise four, `dd_matvec` and `dd_solve` raise unless every
array is a contiguous float64 array. `dd_dot` alone has a Python wrapper that calls `np.ascontiguousarray`. hp calls
`np.ascontiguousarray` in 48 places (`driver.py` 12, `anm_run.py` 9 with its `c4` lambdas, `pinning.py` 3,
`qjet_check.py`'s `c` and `q4`, and others). Not all of them are for multidouble. Some feed hp's own `Solver` methods,
which have the same contract (`python.rs` `slice`), so the multidouble change removes only part of the friction. hp
could add the same coercion in `driver.py` for its own methods.

Change: thin wrappers in `__init__.py` calling `np.ascontiguousarray(v, dtype=np.float64)`, as `dd_dot` does. Bits:
none, since the values are copied unchanged. Test: a strided view and an integer array give the bits of their
contiguous float64 copies. Effort: an hour.

### 4. Small lane conveniences (optional riders)

| Item | hp code | multidouble API | Bits, checks |
|---|---|---|---|
| Write f64 tangent k of a field | `model.rs` 182-183 (`dtau`), 568 (`psolve`), 902 and 911 (seeding in `jac_columns_n`) | `Fl::set_tan64(&mut self, k, &Fl<F8<S>, B>)` | data movement only; `ref` exercises `psolve`'s f64 branch and the seeding (`jacobian_f64`, `newton_mixed`) |
| Read f64 tangent k as a double-double field | `model.rs` 919-922 | already there: `s[q].tan64(k).val()` | identical (`F8::val` has lo = 0) |
| Unit tangent on point j | `model.rs` 896-904 and 976-985; `qsolve.rs` 183-196 | `Fl::unit_tangent(v, j)` per lane kind | data movement; `qsolve.rs`'s copy is covered only by the Rust tests |
| `[QD]` to four part arrays and back; `[DD]` to (hi, lo) and back | `python.rs` 256, 271, 902-904; `qsolve.rs` 224, 288, 309, 409; about 13 inline `DD` conversions in `python.rs` | `qd::to_parts`, `qd::from_parts`, `dd::split`, `dd::join` | data movement |

Each saves a few lines and none can change a bit. They are worth doing only in the same release as item 1.

### 5-7. Generic, but leave them in hp

- **`lm_steps`** (`python.rs` 1070-1115). For each mu it factors (G + mu^2 diag G), G = J^T J by `linalg::gram`. Only
  `lm_solve` uses it (`scan_hp.py --lm`, `lowD_dd.py --lm`), only on the dense path (M <= 128), and `ref` does not
  cover it. A move would be correct but would serve no current run.
- **The complex-shifted field solve** (`modecount_cx` `horizon_lus_c` and `psolve_c`'s 2M branch). It solves
  (A + lr + i li) f = r through the real form [[A + lr, -li], [li, A + lr]]. A `FieldSolve` method for a pair of
  fields would be generic, but its only user is the finished mode count.
- **`recip_int`** (`model.rs` 165, and a copy in `modecount_cx`) is `DD::from_f64(k as f64).recip()` bit for bit,
  since `DD::recip` is `DD::ONE / self`. Switching is cosmetic.
- **`phase_row`** sums w_i p1_i by sequential double-double adds, and `phase_q` uses `QdAcc`. A `Fl::dot` through
  `DotAcc` would change the residual's last row, which `ref` would flag. Nothing is gained.
- **`plan_path`'s step rule** (h from the last two Taylor coefficients, capped, with no sliver step) is generic, but
  it is ten lines tied to `TWork`.
- **The assembly in `horizon_lus`** (omega D + diag kappa_kk stored by columns) could be a `DMat::store_cols(scale,
  ld)`. That is six lines, and `modecount_cx` has its own 2M variant.

## Reverse: multidouble features hp reimplements or works around

| multidouble | hp | Action |
|---|---|---|
| `FieldSolve::solve_lanes_refine_t` truncates a jet silently | `psolve` branches on `NJ` before calling it | guard in multidouble (item 1) |
| No jet solve | `psolve`'s two jet branches | item 1 |
| `Fl::tan64` without a setter | hand loops of `set_tan64` | item 4 |
| `DD::recip` | `recip_int` (two copies) | optional, cosmetic |
| Python array functions demand contiguous float64 | about 48 `np.ascontiguousarray` wrappers | item 3 |
| `Fl::load_parts`/`store_parts` exist for fields, nothing for slices | inline `QD` part conversions | item 4 |
| `DDLu::factor` takes column-major input with a leading dimension | `python.rs` `lu_factor` transposes row-major input by hand, as multidouble's own `dd_solve` does | leave; a `DDLu::factor_rows` would serve both, but saves only a loop |

hp already uses multidouble's `gcr` (`driver.gmres_refine`), `dd_dot` (`delta_correction`), `StaircaseLu` (`block.rs`),
`TangentTail`, `pow2_near` and `solve_lanes_refine_t`, as the earlier moves intended.

## Duplicated helpers in hp's Python

| Helper | Copies | Differences | Home |
|---|---|---|---|
| Richardson (4 r2 - r1) / 3 with 1/3 in double-double (`THIRD`) | `anm_run.py` 90-96, `qjet_check.py` 45, 71-74 | none, line for line | `driver.py`; `qjet_check.py` already imports `anm_run.coefficients` |
| Evaluation of the ANM series | `anm_run.py` `series_at`, `anm_step.py` 131-135, `anm_lowd.py` `evaluate` | Horner in double-double against float64 powers | multidouble's `dd_horner` (item 2) |
| The ANM coefficient loop (residual_jet, solve, refinement) | `anm_step.py` 98-118, `anm_run.py` `coefficients`, `anm_lowd.py` `coefficients` | the solve (stored LU plus one `jvp` refinement, or `gmres_refine`) and the held offset | `driver.py`, with the solve as an argument |
| Lagrange weights | `driver.py` `predict` (420), `predict_q` (717), `pinning.py` `predict_dd` (108) | the same f64 weight loop, combined in f64, quad-double and double-double increments | one `lagrange_weights` in `driver.py`; the three combinations stay |
| Four-part helpers | `driver.py` `qparts`, `qmax`; `qjet_check.py` `q4`, `qsub`, `qmax`, `dd_as_q`; `anm_run.py` `c4` lambdas | the same | `driver.py` (`qparts`, `qmax`, a `qsub`); most go with item 3 |
| `load_state` | `anm_run.py`, `anm_step.py`, `jet_check.py` | `driver.load` without `astype(float)` | `driver.load` |
| A centre or horizon series summed at a point | `driver.renode` (twice), `lowD_dd.reintegrate`, and diagnostics such as `largeD_profiles.py` | unscaled centre coefficients in f64 Horner | `driver.py`, on the scaled series (`series(zh, zl, which, True)`, `series_scale`), as `renode`'s horizon branch and `lowD_dd.tails` already do |
| Weak and junk subspaces | `driver.py` `weak_subspace`/`junk_subspace` and their `_q` twins | the solves only | one implementation taking the two solves |
| Column-major to numpy | `python.rs` `to2` three times, `modecount_cx` `to2`; `freeze_pair` in both crates | none | one helper in `python.rs`, made `pub` for `modecount_cx` |

None of these needs multidouble except the series evaluation. They are hp-internal refactors of scripts, which the
kernel's bit-identity checks do not cover. The Richardson and Lagrange moves keep the operations, so the scripts'
outputs stay bit for bit.

## What stays in hp, and why

- **The model.** `Ctx` and its layout (offsets, `block_inputs`, `block_target`, `seg_inward`, `match_node`),
  `centre_series`, `taylor_coeffs`/`taylor_step`/`taylor_path`, `horizon_series` with `horizon_alpha`, `horizon_lus`'s
  kappa formula, `dtau` with the mode-matrix shift, `dim` and the jet controls. These are the field equations and the
  regular-centre and regular-horizon series of this ODE.
- **The non-jet branches of `psolve`.** They are the model's operator. Their generic part is already multidouble's
  (`solve_lanes`, `solve_lanes_hi`, `solve_lanes_refine_t`).
- **The Jacobian's layout and policy.** `jac_tasks`, `jac_column`, `jac_columns_n`, `jacobian_lanes`, `jacobian_block`,
  `jvp`, `qsolve`'s `jacobian_q` and `newton_q`, the freeze projection and the pinning (`pinning.py`), and the `HDD_*`
  and `TAIL_*` choices. They encode the residual's block structure and the large-D fixes.
- **Size dispatch.** `make_ctx!`, `qd_sizes!`, `residual_jet_sized` and `DENSE_M_MAX` are build-memory policy for this
  kernel.
- **The spectral grid.** `dmat_odd`, `plow_odd`, `phase_weights`, `band_coeffs`, `band_field`, `resample`,
  `resample_z` and `harmonics` are the half-grid folding of this model's parities.
- **Kept on purpose earlier** and not re-proposed here: the `Lu` enum, `qd_check.py` and `op_costs`.
- **The Newton drivers** (`newton_solve`, `deflated_newton`, `newton_qd`, `lm_solve`, `gmres_refine`'s wrapper). They
  are model-agnostic in form but are written against the `Solver` API and its stored-LU semantics.

One aside: `dtau`'s branch for f64 tangents with lam != 0 (`model.rs` 176-187) cannot be reached from Python.
`mode_matrix` sets lam only around `model::jacobian` (`Du` lanes), and no path sets it for `TAN_F64` lanes.

## Discrepancies found along the way (reported, not fixed)

1. `anm_step.py` (131-135) and `anm_lowd.py` (`evaluate`, feat/jet-all-sizes) sum the ANM series with float64 powers
   h**k. `anm_run.py` `series_at` names this as an error of about 1e-16 of each term and keeps it as a control. The
   error is of order 1e-16 |z_2| h^2, harmless for a predictor, but the three scripts evaluate one series three ways.
2. The unscaled centre series is summed in f64 at the first node in `driver.renode` (for a new node below the first
   old one) and in `lowD_dd.reintegrate`. `model.rs`'s own comment on `series_scale` says the unscaled coefficients
   overflow the f64 range near r = 1e-3 at 100 orders. T82-win's low-D states start at x = 0.000157 with kc = 100
   (their meta). There sigma = 2^-13, so an unscaled coefficient is its scaled one times 2^(13k), which leaves the f64
   range past k = 78 unless the scaled one is very small. These paths probably return inf on such states. Not run.
3. multidouble's `solve_lanes_refine_t` returns a jet's orders 2 and up as zero without an error (item 1). hp does not
   hit it.

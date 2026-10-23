# Larger τ resolutions for the double-double solver (`hp/`), and how far toward D = 3 they get

Status: work in progress (numbers to follow as the runs finish).

## 1. What changed

### 1.1 Sizes

The kernel's τ grid is M = 8B collocation points on the half period, with B a const generic (`Ctx<B>`, `Fl<T, B>`).
`hp/src/python.rs` now builds B = 20, 24, 32, 40 and 48 besides the old 1–16, so M = 160, 192, 256, 320 and 384 run.
Nothing else in the kernel assumed M ≤ 128: the fields, the τ-derivative matrix (`DMat`, M × M), the horizon LUs
(M × M, ks + 1 of them) and the Cauchy sums are all sized from B or allocated on the heap. On the Python side,
`driver.resample_z`, `dmat_odd` and `phase_weights` take any M.

Two things grow with M and were changed so they fit:

- **The horizon LUs** (ks + 1 double-double M × M factorizations, 0.6 GB at M = 384 and ks = 250) were cloned once per
  Jacobian when no mode-matrix shift was set (`lus_t = lus.clone()` in `model::jacobian_lanes`). They are now shared
  by reference; the shifted set is built only for a mode matrix (λ ≠ 0). This changes no number.
- **The worker threads' stacks**: the field arrays live on the stack (a state of `Dtn<8>` lanes at M = 384 is 120 kB),
  so the rayon pool is built with 256 MB stacks (`python.rs`; only the pages touched are used). This changes no
  number.

### 1.2 Block storage of the Newton system (`hp/src/block.rs`, `model::jacobian_block`)

The Jacobian of the multiple-shooting system (n = 4M(K + 1) + 1, K nodes) was built and factored as a dense n × n
array; `DDLu` skips its zero blocks in time, not in memory. At M = 256 and K = 13 (n = 14 337) a dense f64 Jacobian
is 1.6 GB and a double-double one 3.3 GB. The mixed-precision Newton step now has a block-storage path that never
forms the dense matrix:

- **Structure.** Residual block b (4M rows) depends on two neighbouring groups of unknowns and on Δ: block 0 on the
  centre data (p1, c0) and node 0, block b = 1…K−1 on nodes b−1 and b (whichever way its segment runs, so
  `match_node` layouts work), block K on node K−1 and the horizon data (m_h, a_h). The phase row depends on p1 only.
- **Δ.** Δ enters every block. The factorization works on an augmented system with one copy δ_b of Δ per residual
  block and link rows δ_{b−1} − δ_b = 0. With the column groups g_0 = (p1, c0), g_t = (δ_{t−1}, node t−1),
  g_{K+1} = (δ_K, m_h, a_h) and the row groups (phase, block 0), (link, block 1), …, (link, block K), row group s
  touches only g_s and g_{s+1}.
- **Staircase LU.** Gaussian elimination with partial pivoting then runs in steps: step s factors the columns of g_s
  (pivot candidates: the rows left over from step s−1, which are zero outside g_s, and row group s) and hands the
  rows it did not pivot on, now holding the Schur complement on g_{s+1}, to step s+1; the last step is square. This
  is exactly partial pivoting on the augmented matrix in this order (rows further down never hold a step's columns),
  so it has that algorithm's stability, and all fill stays inside the steps. Each step is the blocked right-looking
  LU of `DDLu::factor_f64`, restricted to the first np columns of a rectangular block (`block::lu_rect`).
- **Solves** are double-double with the f64 factors, as on the dense mixed path. With E the copy of Δ into every δ_b
  and S the selection of the non-link rows, J = S J_aug E, so J⁻¹ = P J_aug⁻¹ Sᵀ and J⁻ᵀ = S J_aug⁻ᵀ Pᵀ for any P
  with P E = I; P reads Δ from δ_0. `solve`, `solve_t` (the sensitivities ∂Δ/∂F) and `solve_many` are implemented.
- **The Jacobian** is built block by block (`model::jacobian_block`): for residual block b, the columns of its tasks
  come from the same `jac_columns_n` / `jac_column` as in `jacobian_lanes` (in parallel), so every entry equals the
  dense f64 Jacobian's; they are written with the −I of the target node, the Δ column (into δ_b) and, for block 0, the
  phase row straight into step b's array, and that step is factored before the next block is built. Peak memory is
  the factors so far plus one block.
- **Python.** `Solver.newton_mixed(..., block=None)` uses block storage when M > 128 and the dense path otherwise, so
  every result at M ≤ 128 is unchanged; `block=True` forces it at any M (tangents 1 or 4, `dd_base` with 4; f64 LU
  only). The stored LU (dense or block) serves `chord`, `lu_solve`, `lu_solve_t` and `lu_solve_many`.
  `Solver.lu_info()` gives the factors' size and, for block storage, the seconds of building and factoring.
  `block_mutate` and `jac_mutate` are the controls and the timing hook of `hp/highm_check.py`.
- **Not supported in block storage:** the frozen-harmonics layout (`kfreeze` couples the horizon block to p1) and
  the mode matrix (γ). Above M = 128 the dense paths (`jacobian`, `jacobian_f64`, `newton`, `lm_steps`,
  `mode_matrix`, `mode_det`, `lu_factor`, `newton_mixed(block=False)`) refuse with a message: the dense
  double-double Jacobian at M = 160 would be 1.3 GB. So the exact double-double Jacobian, the Levenberg–Marquardt
  fallback and γ stop at M = 128. In `driver.newton_solve(jac="f64")`, a step without decrease is retried with the
  exact Jacobian at M ≤ 128 as before, and above it with the `dd4` f64 Jacobian (f64 tangents on a double-double base
  trajectory) in block storage.

For later work on the Jacobian (a double-double horizon block, a double-double Jacobian-vector product): the block
path calls the per-block column functions (`jac_columns_n`, `jac_column`) and needs nothing else from
`jacobian_lanes`; a new kind of column only has to produce, per residual block, the 4M rows of each column task
(`jac_tasks`) as `jac_columns_n` does, and `jacobian_block`'s `fill` closure places them (input fields by
`Ctx::block_inputs`, the target by `Ctx::block_target`, Δ into the block's δ copy). `jacobian_lanes` itself changed
only in not cloning the horizon LUs.

### 1.3 Taylor plans of resampled states

A state resampled to a larger M needs 3.5–4 times more Taylor steps than the converged solution at that M (plan
tolerance 10⁻³⁰, hmax 0.05, D = 3.7: converged states 56, 72, 86 steps at M = 48, 64, 80; M = 48 resampled to 64,
272; 64 to 80, 311; 80 to 96, 346; 80 to 128, 366). The resampled state's new top harmonics come out of the products
as aliasing terms that the τ-derivative amplifies; near the solution they are small. A coarser plan blows up on the
resampled state (|F| = NaN at M = 128 on the 86-step plan of the M = 80 state), while the converged states keep
|F| = 4–7×10⁻³⁰ on their own plans from 56 up to 311 steps. So the solves here use two passes when the resolution
goes up: pass 1 on the resampled state's own plan to |F| ≤ 10⁻¹⁴, pass 2 on the plan of the pass-1 state to
10⁻²⁸; the continuation reuses the plans of converged states (1.5). The earlier records
(`hp/data/lowD_dd.jsonl`; the local machine's runs) planned from the resampled starts, so their plans are finer than
needed: that costs time, not accuracy.

### 1.4 The weakest directions at M ≥ 192 (D = 3.7)

At D = 3.7 the lift to M = 192 (from the converged M = 160 state, pass 1 on its 511-step plan) did not converge
with plain f64-Jacobian Newton: |F| 5.7×10⁻¹⁰ → 5.0×10⁻¹⁰ → 3.8×10⁻¹⁰ with line-search factors 0.125 and 0.25, steps
|dz| ≈ 10⁻⁷ (200 times |F|), and the largest LU pivot up from about 10 (M ≤ 160) to 1.9×10³
(`hp/data/lowD_highM/logs/lift_M192_plain_newton.log`). `python3 highm_diag.py jac` at that state (three f64 Jacobians that differ only in rounding, block storage;
`hp/data/lowD_highM/logs/diag_jac_M192.log`, and `diag_jac_M128.log` for M = 128) gave:

| | f64x4 | f64x1 | dd4 |
|---|---|---|---|
| Δ correction of the step | +6.9×10⁻¹³ | −3.6×10⁻¹³ | −1.1×10⁻¹³ |
| \|(J − J̃) dz\| (linear-model mismatch at t = 10⁻³) | 1.2×10⁻¹² | 1.2×10⁻¹² | 4.1×10⁻¹³ |
| \|F(z + dz)\| (|F(z)| = 3.75×10⁻¹⁰) | 1.8×10⁻⁸ | 3.7×10⁻⁹ | 2.2×10⁻⁹ |

The steps differ by 9–38% between the variants and the sensitivities ∂Δ/∂F by 6–50%, while at the converged M = 128
state the same comparison gives 2–3×10⁻⁷ and 10⁻⁵. Along each step the Jacobian is accurate to first order (the
mismatch is 0.1–0.3% of |F|); what differs is the component in the weakest directions, which only the rounding
fixes. Those components wreck the series (`python3 highm_diag.py radius`, `diag_radius.log`): at the M = 160 solution the horizon series has radius 0.28–0.30 in u
(u₀/r = 0.7, tail 2×10⁻⁴¹) and the centre series 0.017–0.020 (x₀/r = 0.3); at the M = 192 state after the damped steps
(|F| = 3.75×10⁻¹⁰; `hp/data/lowD_states/highm_p1_D3.7000_M192_stalled.npz`) they are 0.21 (u₀/r = 0.93, tail 3.7×10⁻¹¹) and 0.007 (x₀/r = 0.6–0.8), for every τ-harmonic. These are the
"junk" directions of `driver.deflated_newton` (top centre and horizon harmonics, coupled to the matching point only
exponentially weakly): at D = 3.7 the M = 192 grid is far beyond what Δ needs (its top harmonics are 10⁻¹³–10⁻¹⁵), so
they are weaker than the f64 Jacobian's rounding can resolve. dd4 (double-double base trajectory) does not help,
since its tangents are f64.

**Not only the rounding: the weak directions are nonlinear there.** The large-D agent ran its correction loop (GMRES
with exact double-double Jacobian–vector products, the f64 LU as preconditioner; `driver.newton_solve(jac="f64g")`,
branch `wip/largeD-mixed` at 127d606, notes/LARGE_D_MIXED.md §8) on the stalled state: the loop corrects the f64 step,
which was off by 0.3–4.7 of its own size, to 10⁻¹¹ in 12 products per step, but the exact steps are still cut back by
the line search (λ = 1/2, 1/4, 1, 1/8), and |F| stays at 1.4–2.0×10⁻¹⁰ over four Newton calls (stopped after
2 h 19 min under load 10–13). At |F| ≈ 2×10⁻¹⁰ an exact step of |dz| = 10⁻⁷–3×10⁻⁷ is already outside the range of
the quadratic model. So the stall comes from the nonlinearity of the weak directions, and only in part from the f64
Jacobian's rounding (which misplaces the step, and which the loop removes). That fits what cured it: on a node set
where both series are evaluated well inside their radii (below), the same lift took full steps and converged
quadratically.

**The cure was the node set, not the Newton step.** The start itself (the M = 160 solution resampled to 192, before
any step) already has series tails of 10⁻²⁰ (centre, first node 0.005) and 1.5×10⁻²² (horizon, last node 0.8), against
2×10⁻⁶⁰ and 2×10⁻⁴⁰ for the M = 160 solution it came from; filling the new harmonics by each field's geometric trend
(`lowD_dd.fill_tail`) changes that only to 2×10⁻²¹ (`python3 highm_diag.py start`, `diag_start.log`). With the first node
at 0.0025 the centre tail of that start is 3×10⁻⁵⁰ (8.5×10⁻⁵⁰ at M = 256), and with a node at 0.9 (u₀ = 0.1) the horizon
tail is 3×10⁻⁹⁷ (2×10⁻⁶⁸ at M = 256); with the last node at 0.8 the horizon series of the M = 256 start diverges
(tail 10⁷). On the 15 nodes 0.0025, 0.005, 0.01, …, 0.8, 0.9 the M = 192 lift took its first step at λ = 1 and
converged in two Newton calls (section 3). The node set has to follow M, as notes/LOW_D.md already found going from
M = 48 to 64; here the decisive quantity is u₀ against the horizon series' radius, which shrinks as the top harmonics
fill in. The deflated step with the junk directions frozen (`driver.deflated_newton(junk_only=True, jac="f64")`, now
usable with the mixed step and block storage) was started on the 13-node set and stopped unfinished once the node set
was found to be the cause.

`driver.renode`'s default Taylor step (0.004) blows up integrating toward a node at 0.9 at M = 160 (|F| 8×10⁹); with
0.001 the re-noded state has |F| 1.1×10⁻¹⁶ (the f64 evaluation of the centre series at the new first node). The scripts
here pass 0.001; `lowD_dd.py --auto_nodes` still uses the default (a discrepancy for the coordinator).

### 1.5 Plans for the continuation

The plan of a state is lean only if the state is a regular solution at the D of the equations it is planned with. A
state that is not (a predictor, a resampled state, even a converged state rounded to f64: the M = 48 state at D = 3.7
needs 56 steps with its lo parts and 115 without) is not regular at the sonic line x = c, where the p equation is
singular, and its Taylor coefficients there are large. `hp/lowD_highM.py` therefore reuses at each new D the plan of
the last converged double-double state at the same M, computed at that state's own D with the step tolerance times
--plan_margin (10⁻², about 14% more steps), and records, after convergence, how many steps the new solution's own plan
would have (steps_own_w / steps_own_n in the pair records). In the small-M dry runs (M = 48 / 64 / 80, D = 3.7 → 3.6)
this gave 63, 80, 78 and 94 steps where planning from the predictor gave 130–245; the values of Δ did not change.

### 1.6 The series orders (kc = 100, ks = 250) cost little and change nothing

The large-D agent found that a centre-series order of 420 made its double-double Jacobian 3 times slower for
bitwise-identical results. Here the orders are kc = 100 (centre) and ks = 250 (horizon), from the low-D states.
`python3 highm_diag.py orders STATE M --orders=... [--newton]` evaluates a converged state's residual on its own plan
with lower orders (`diag_orders_D3.3_M96.log`, `diag_orders_D3.3_M96_newton.log`; the pilot's D = 3.3, M = 96
state, 15 nodes):

| kc / ks | residual | centre / horizon tail | residual against 100 / 250 | Newton step |
|---|---|---|---|---|
| 100 / 250 | 0.51 s | 4×10⁻⁶⁴ / 2×10⁻⁹⁶ | | 77 s (load 3.2) |
| 80 / 200 | 0.46 s | 2×10⁻⁵¹ / 5×10⁻⁷⁷ | bitwise identical | |
| 60 / 150 | 0.43 s | 1.5×10⁻³⁸ / 6×10⁻⁵⁸ | bitwise identical | 75 s (load 3.6), bitwise identical step |
| 50 / 120 | 0.43 s | 3×10⁻³² / 4×10⁻⁴⁶ | differs by 3.9×10⁻³⁴ | |
| 40 / 100 | 0.38 s | 3×10⁻²⁵ / 6×10⁻³⁹ | differs by 8.8×10⁻²⁸ | |

So at low D the orders are above what the results need (60 / 150 gives the same bits at D = 3.3), but they cost
about 16% of a residual and 3% of a Newton step, whose time is in the Taylor steps; lowering them is not worth the
margin. The margin shrinks toward D = 3: the centre tail at kc = 100 grows from 10⁻¹¹⁰ (D = 3.7, M = 80) to 10⁻⁴⁹
(D = 3.25, M = 128) (section 5), and `--auto_nodes` moves the first node in when it passes --tail_max (10⁻³⁰).

## 2. Verification

All on the cloud machine (2 cores, AVX-512), private builds: main a1973f7 and this branch, both with
`RUSTFLAGS="-C target-cpu=native"` and the release profile of `hp/build.sh`. Logs in `hp/data/lowD_highM/logs/`.

### 2.1 Everything at M ≤ 128 is bit-identical to main

- `cd hp && python3 mixed_check.py ref --compare=REF.npz`, REF written by main's build (`ref --save`): the 17 outputs of
  the default double-double path at D = 3.7, M = 48 (residual, Jacobian, Newton step and pivots, chord, transpose
  solve, mode matrix at λ = 0.3, the frozen-harmonics and inward-matching Jacobians) are bit-identical
  (`ref_mixed_compare.log`).
- `cd hp && python3 highm_check.py ref --compare=data/lowD_highM/REF_highm_main_a1973f7.json` (the hashes written by
  main's build with `ref --save`, `ref_highm_save_main.log`): 24 outputs at each of M = 8, 16, 48, 64, 80, 96 and 128
  (the D = 3.7 state resampled to each M): residual, plan, centre and horizon series, the dense double-double Newton
  step with its pivots, the chord step and transpose solve after it, the dense f64 Jacobian, and `newton_mixed`'s
  step, residual, pivots, chord step and transpose solve. All 168 are bit-identical (`ref_highm_compare.log`). The
  outputs are compared as SHA-256 hashes of their float64 bytes, so the M = 128 Jacobians (0.4–0.8 GB) are not stored.

### 2.2 Rust tests

`cargo test --lib` (debug profile, `cargo_test_debug.log`): 5 pass, the 2 benches are ignored. The new
`block::tests::block_lu_matches_dense` builds random systems with the multiple-shooting structure (near-identity
propagators, outward and inward segments, a Δ column and a phase row; n = 101 to 1041) and compares the block LU's
solves (plain and transposed) with the double-double LU of the same f64 matrix, next to the dense f64 LU's error
(`cargo_test_block.log`):

| n (groups, chain) | block LU error | dense f64 LU error | pivot control | link control |
|---|---|---|---|---|
| 101 (10/20, 4) | 1.3×10⁻¹⁵ / 1.7×10⁻¹⁵ | 4.5×10⁻¹⁵ / 4.2×10⁻¹⁵ | 1.1×10⁻⁷ / 4.4×10⁻⁸ | 0.32 / 0.16 |
| 225 (16/32, 6) | 5.9×10⁻¹⁵ / 6.5×10⁻¹⁵ | 3.4×10⁻¹⁵ / 5.9×10⁻¹⁵ | 2.8×10⁻⁷ / 1.7×10⁻¹⁰ | 0.21 / 0.049 |
| 1041 (40/80, 12) | 1.4×10⁻¹³ / 1.1×10⁻¹³ | 1.9×10⁻¹⁴ / 1.2×10⁻¹⁴ | 2.5×10⁻⁹ / 1.4×10⁻⁹ | 1.7 / 0.91 |
| 673 (24/48, 13, inward from 8) | 5.3×10⁻¹⁵ / 3.5×10⁻¹⁵ | 3.9×10⁻¹⁵ / 5.0×10⁻¹⁵ | 4.5×10⁻⁷ / 1.1×10⁻⁷ | 0.65 / 0.41 |

(relative to the solution's max, solve / transposed solve). Pass: the block error at most 100 × the dense f64 error +
10⁻¹⁵; both controls (one pivot scaled by 1 + 10⁻⁶; the Δ copies untied) must fail that and do.
`block_solve_many_is_solve` checks that several right-hand sides give each row bitwise as single solves.

### 2.3 Block storage against the dense path (M ≤ 128)

`cd hp && python3 highm_check.py block --Ms=16,48 --variants=f64x4,f64x1,dd4` and `--Ms=80,128` (f64x4)
(`block_check_small.log`, `block_check_large.log`, JSON next to them): `newton_mixed` with block=True against
block=False at the D = 3.7 state resampled to each M, and at a perturbed state (Δ + 10⁻⁶, fields + 10⁻⁸). The residuals
are bit-identical (the same code). The step, the sensitivities ∂Δ/∂F (`lu_solve_t`) and a chord step after it are
compared relative to their max, next to the floor of the dense f64 LU itself (the same right-hand side solved with the
dense double-double LU of the same f64 Jacobian). Pass: every difference ≤ max(100 × floor, 10⁻¹²).

| M (N) | variants | step | ∂Δ/∂F | chord | floor (step, ∂Δ/∂F) | LU, block / dense |
|---|---|---|---|---|---|---|
| 16 (897) | f64x4, f64x1, dd4 | 4.6×10⁻¹⁵–1.5×10⁻¹⁴ | 1.6–4.9×10⁻¹⁵ | 7×10⁻¹⁵–1.8×10⁻¹⁴ | 2–10×10⁻¹⁴, 2–4×10⁻¹⁴ | 1.1 / 12 MB |
| 48 (2689) | f64x4, f64x1, dd4 | 4.1–8.2×10⁻¹⁴ | 1.9–3.7×10⁻¹⁴ | 5.1–7.9×10⁻¹⁴ | 3–7×10⁻¹⁴, 2–6×10⁻¹⁴ | 9.6 / 111 MB |
| 80 (4481) | f64x4 | 5.4×10⁻¹⁴, 1.2×10⁻¹³ | 1.1×10⁻¹³, 4.0×10⁻¹³ | 3.8×10⁻¹⁴, 4.1×10⁻¹³ | 4.8×10⁻¹⁴–1.1×10⁻¹³, 0.9–2.7×10⁻¹³ | 26 / 307 MB |
| 128 (7169) | f64x4 | 4.1×10⁻¹³, 1.2×10⁻¹¹ | 5.8×10⁻¹³, 1.6×10⁻¹¹ | 3.0–3.3×10⁻¹² | 2.8×10⁻¹³–7.7×10⁻¹², 1–12×10⁻¹² | 67 / 785 MB |

All pass, and the block path's steps agree with the dense path's as closely as the dense f64 LU agrees with an exact
solve of the same matrix. The block LU takes 8–12 times less memory and about the same time.

### 2.4 Controls

`python3 highm_check.py controls` (`block_controls.log`): at M = 48, one pivot of the middle step scaled by 1 + 10⁻⁶
changes the step by 4.8×10⁻⁸–1.5×10⁻⁷ (the limit is 4×10⁻¹²), and the Δ copies untied (each link row replaced by
δ_s = 0) by 0.21–0.44 and the sensitivities by 6.4×10²: both fire. The Rust test has the same two controls (2.2).

### 2.5 Convergence to the same solution

`python3 highm_check.py converge --M=80` (`converge_M80.log`): `driver.newton_solve(jac="f64")` from the D = 3.7 state
with Δ + 10⁻⁶ and fields + 10⁻⁸ (seed 1), with the dense and with the block path, 7 Newton calls each: Δ agrees to
1.3×10⁻³⁰ and max |z| to 5.0×10⁻²⁸ (relative), against 1.1×10⁻²⁷ between two dense solves from different starts (the
control); pass (≤ max(10⁻²⁷, 10 × the control)).

## 3. D = 3.7 from M = 80 to 256

From `hp/data/lowD_states/lowDv_D3.7000_M80.npz` (13 nodes 0.005…0.8, kc 100, ks 250, Taylor tolerance 10⁻³⁰,
hmax 0.05), each M lifted from the one below in two passes (`highm_check.py lift`; `stage1.sh`, `stage1_n15.sh`,
`pilot.sh`); f64-Jacobian Newton to |F| ≤ 10⁻²⁸ (2.5×10⁻³⁰ to 3.4×10⁻²⁹ reached). On the 15 nodes 0.0025…0.9 (section
1.4) the 80 and 96 states are the M = 80 state re-noded and lifted, the 128 and 160 states the 13-node ones re-noded.
`python3 lowD_highM_report.py --stage1`:

| M | nodes | Δ | change from the M before | Taylor steps (pass 1 → 2) | tails (centre, horizon) |
|---|---|---|---|---|---|
| 80 | 15 | 3.465200483544031475184929 | | 102 → 97 | 6×10⁻¹¹⁵, 9×10⁻¹⁴⁵ |
| 96 | 15 | 3.465200482305272998798531 | −1.24×10⁻⁹ | 395 → 123 | 2×10⁻¹¹¹, 3×10⁻¹³⁶ |
| 128 | 15 | 3.465200482271642183274714 | −3.36×10⁻¹¹ | 162 → 166 | 5×10⁻¹⁰⁰, 2×10⁻¹²³ |
| 160 | 15 | 3.465200482271638953641 | −3.23×10⁻¹⁵ | 203 → 256 | 1×10⁻⁸⁹, 5×10⁻¹¹⁵ |
| 192 | 15 | 3.465200482271638952016727 | −1.62×10⁻¹⁸ | 593 → 252 | 3×10⁻⁶⁶, 2×10⁻¹⁰⁸ |
| 128 | 13 | 3.465200482271642183274714 | | 366 → 191 | 1×10⁻⁶⁹, 9×10⁻⁴⁹ |
| 160 | 13 | 3.465200482271638953641 | −3.23×10⁻¹⁵ | 450 → 279 | 2×10⁻⁶⁰, 2×10⁻⁴⁰ |

- **The M differences shrink fast but not at one rate:** −1.24×10⁻⁹ (80 → 96, 16 modes), −3.36×10⁻¹¹, −3.23×10⁻¹⁵,
  −1.62×10⁻¹⁸ (32 modes each); successive ratios 0.027, 9.6×10⁻⁵, 5.0×10⁻⁴. So Δ(3.7) is converged in M to about
  10⁻¹⁸ at M = 160–192, with the M = 192 value taken as Δ(3.7) = 3.465200482271638952 and the last difference as its
  error estimate (an estimator, not a bound). The local machine's M = 96 value, 3.4652004823053, agrees with the M = 96
  here to all its digits (3.4652004823052730), and its M 80 → 96 difference, −1.2×10⁻⁹, with −1.24×10⁻⁹.
- **The node sets agree:** 13 against 15 nodes, 6.7×10⁻³¹ at M = 128 and 4.1×10⁻³⁰ at M = 160; the 15-node M = 80
  differs from the original 13-node M = 80 state by −5.4×10⁻³⁰.
- **The plans:** pass 1 runs on the resampled start's own plan (3–4 times the steps of the converged solution's),
  pass 2 on the plan of the pass-1 solution. Changing the plan moves Δ by 10⁻²⁴ (M = 128, 366 → 191 steps) to
  7×10⁻²² (M = 160, 450 → 279): the first Newton step of pass 2 corrects Δ by −1.5×10⁻²⁴ and +6.6×10⁻²² (logs).
- **The largest sensitivity |∂Δ/∂F|** (from the f64 LU) is 0.7–2.3 from M = 80 to 160 and 4.8×10² at M = 192, where
  the weakest directions are as weak as the f64 Jacobian's rounding (section 1.4): Δ is still fixed by |F| ≈ 10⁻²⁹ to
  about 5×10⁻²⁷.

## 4. Time and memory

**Load.** The cloud machine has 2 cores. During this work several other agents ran on it (load average 2–13); every
time below is wall time on 2 threads with the load average at the time recorded next to it, and times at load L > 2
are inflated by roughly L/2. They are measurements of this run, not of the kernel alone.

**One Newton step** (`newton_mixed`: double-double residual, f64 Jacobian, LU, solve) at converged D = 3.7 states
(`python3 lowD_highM_report.py --stage1`; dense storage at M ≤ 128, block above):

| M | nodes | Taylor steps | step | Jacobian | of which τ-derivative | LU | residual | LU MB | peak RSS MB | load |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 13 | 191 | 76 s | | | (dense) | 0.5 s | 785 (dense) | 987 | 3.0 |
| 160 | 13 | 279 | 100 s | 99 s | 31 s (31%) | 0.46 s | 0.6 s | 105 | 435 | 2.3 |
| 160 | 15 | 256 | 350 s | 347 s | 143 s (41%) | 1.2 s | 1.4 s | 121 | 542 | 4.8 |
| 192 | 15 | 252 | 984 s | 973 s | 231 s (24%) | 3.3 s | 4.9 s | 173 | 721 | 13.2 |

The τ-derivative's share is the Jacobian's time minus the same Jacobian built without it (`jac_mutate=1`). A profile at
M = 192 on a 500-step plan (load about 4) gave 561 s for the Jacobian, 46% of it the τ-derivative, 1.1 s for the
LU, 2.5 s for the residual and 535 MB peak RSS (`highm_check.py profile`; before the plans were made lean).

**Memory.** Block storage holds the factors in 105–121 MB at M = 160 against 1.3 GB for the dense f64 matrix (and
2.6 GB for a dense double-double one); the peak RSS of a whole lift stays under 0.7 GB from M = 80 to 192. The
largest single items at large M are the ks + 1 double-double horizon LUs (0.26 GB at M = 256, 0.6 GB at M = 384),
built once per residual and once per Jacobian.

**Where the time goes, and the τ-derivative.** The Jacobian is 98–99% of a Newton step at M ≥ 160 (the LU about
0.5%, the residual 0.4–0.6%). Within it the dense M × M τ-derivative (`DMat::apply`) is 24–46% at M = 160–192; the
rest is the Taylor recursions' Cauchy sums and field arithmetic, whose coefficient arrays (12 fields × 38 orders × M
points × 5 lanes for f64x4) no longer fit the 2 MB L2 cache above M ≈ 128. A real-FFT or DCT derivative,
O(M log M), would remove at most that 24–46%: a step 1.3–1.85 times faster even if the derivative were free, about
1.2–1.5 times with a realistic FFT. It was not done: it needs a new kernel build (35 minutes here under load, with the
container restarting every 30–60 minutes during this work), and the measurements say the larger costs are elsewhere:
the Taylor plans (section 1.3, 2–4 times), the Cauchy sums' memory traffic, and the number of Taylor steps, which
grows faster than M (converged plans: 56, 72, 86 steps at M = 48, 64, 80; 191 at 128; 252–279 at 160–192).

## 5. The pilot toward D = 3

### 5.1 How it runs

`hp/lowD_highM.py` (its docstring has the rules in full), started by `hp/data/lowD_highM/pilot.sh` under `nice -n 10`
on 2 threads; log `hp/data/lowD_highM/logs/pilot.log`, one record per pair in `pilot.jsonl`, one per solve in
`pilot_points.jsonl`, the continuation state in `pilot.jsonl.progress.json`; states `hp/data/lowD_states/lowDhm_*`
(15 nodes) and `lowDhm_x<first>-<last>_*` (after node changes). It resumes from the last converged point and from
the Newton checkpoints of an interrupted solve (it was restarted four times: once after a container restart, three
times by hand, 5.4). The 10-hour budget counts the pilot's running time across restarts. `python3 lowD_highM_report.py` prints the tables of this section.

- **Start:** D = 3.7 at M = 80 / 96 on the 15 nodes 0.0025, 0.005, 0.01, …, 0.8, 0.9 (section 1.4), the smallest pair
  that resolves D = 3.7 (80 → 96 is −1.24×10⁻⁹). A first attempt at 128 / 160 went astray in the weak directions
  from the first, constant predictor (`logs/pilot_attempt1_M128.log`).
- **Each D** (steps of 0.05): the working M_w and the next M_n, and after the first move up the previous working M as
  a helper M_c. The value is Δ(M_n), its error estimate |Δ(M_n) − Δ(M_w)| (an estimator from the next resolution, not
  a bound). f64-Jacobian Newton (`newton_mixed`, block storage above M = 128) to |F| ≤ 10⁻²⁸.
- **Starts:** the cubic predictor through the last four points at the same M gets worse as D falls (|F| 0.02–0.03 at
  D = 3.65–3.4, 0.27–0.5 at 3.25–3.2; its first Newton step is |dz| ≈ 3–30), and at M = 128 a predictor start went
  astray. So each resolution starts from the solution one rung below resampled (helper → M_w → M_n), with |F|
  10⁻⁴–10⁻⁶, and only the helper starts from its predictor.
- **Moving up:** when |Δ(M_n) − Δ(M_w)| > 10⁻⁸, or when a log-linear fit of the last three differences projects it
  above 10⁻⁸ two points ahead; the new M_n is solved at that D in two passes (section 1.3).
- **Plans:** reused from the last converged state at the same M (section 1.5); a start that does not fit the reused
  plan (|F| not finite or above 10⁻³ on it) is solved in two passes, pass 1 on its own plan.
- **Nodes:** besides `--auto_nodes` (a solution's series tail above 10⁻³⁰), the node set changes when a series tail
  of the next point's M_n start (the M_w solution resampled to M_n) passes 10⁻²⁰ (5.4); every resolution is then
  re-noded and re-solved at that D, and the three earlier states per resolution re-noded for the predictors.

### 5.4 What changed on the way down

- **D = 3.5, 3.35, 3.25: moving up.** 80 → 96 reached −1.27×10⁻⁸ at D = 3.5 and 96 → 128 −2.09×10⁻⁸ at 3.35; at 3.25
  128 → 160 was −9.8×10⁻¹⁰, but the last three differences (−7.3×10⁻¹¹, −1.9×10⁻¹⁰, −9.8×10⁻¹⁰) projected 2.7×10⁻⁸ at
  3.15, and the move up to 160 / 192 came then. (At D = 3.2 the helper's 128 → 160 was −6.3×10⁻⁹: the projection was
  right that 128 would not last beyond 3.2.)
- **D = 3.25: the centre series of the lifted start, and the first node.** The M = 128 solution at D = 3.25,
  resampled to M = 160, had |F| = 3.5×10⁻³, all in the centre block (the next blocks 10⁻⁵): its centre series, summed
  at the first node x₀ = 0.0025, had a tail of 10⁻² (`highm_diag.py start`), against 1.2×10⁻⁴⁹ for the M = 128
  solution itself and 3.8×10⁻⁸ for the same lift at D = 3.3, where |F| was 7.6×10⁻⁶. Newton from it went astray in the
  weak directions twice: on the reused 257-step plan (line-search factors 1/4, 1/128, 1/16, Δ moving 10⁻⁵ away from the
  M = 128 value) and, after the two-pass rule was added, on the start's own 629-step plan (1/4, 1/2), whose |F| was
  the same 3.5×10⁻³: the residual was not Taylor error. Both runs were stopped by hand
  (`lowDhm_D3.2500_M160_ckpt_astray.npz`, `lowDhm_p1_D3.2500_M160_ckpt_astray.npz`, not committed). With a new first
  node at 0.00125 the same start has centre tail 3×10⁻³² and |F| = 1.9×10⁻⁵
  (`logs/diag_start_D3.25_M160_x0.00125.log`), and the pilot, resumed with that node change at D = 3.3, solved
  D = 3.25 at 128 and 160 from lifted starts in 3 Newton calls each. The rule now in the driver (a lifted start's
  tail above 10⁻²⁰) would have fired at D = 3.3; at D = 3.2 it fired by itself (tail 2.7×10⁻¹⁵ for the M = 160
  solution resampled to 192; first node 0.00125 → 0.000625). So the centre series of a resampled state converges far
  worse than the solution's, and more so as D falls (why was not established; section 1.4 found the same for the
  horizon series at D = 3.7, M = 192). The node set
  agrees with the old one to 3×10⁻³² (M = 96), 3×10⁻³¹ (128) and 1.1×10⁻³⁰ (160) in Δ at D = 3.3.
- **The local machine's M 80 → 96 lifts** that "do not converge" below D = 3.45 (|F| 0.1 at D = 3.40 and 4×10⁴ at
  3.35 after resampling, comms.txt 2026-10-22 19:01) are, I think, the same effect on its 13 nodes with x₀ = 0.005:
  here, on x₀ = 0.0025, the M 80 → 96 lift at D = 3.40 started at |F| 1.5×10⁻⁴ and converged in 6 calls. At 3.35 it
  gave NaN on the reused plan and the predictor was used instead. The M = 80 solutions themselves agree with the local
  machine's to the 13 digits quoted (5.5); M = 80 does not resolve D ≤ 3.45 to 10⁻⁸ in Δ (80 → 96 = −6.6×10⁻⁸ at 3.45),
  as it says.
- **The horizon node.** The horizon series' radius in u = 1 − x shrinks as D falls: 0.29 at D = 3.7, 0.195 at 3.3 and
  0.155 at 3.2 (M = 160, all τ-harmonics alike; `logs/diag_radius_D3.3_D3.2_M160.log`), so with u₀ = 0.1 (last node
  0.9) the solutions' horizon tails grew from 10⁻¹³² (D = 3.65) to 4×10⁻⁴⁸ (3.2). Extrapolated, the radius reaches
  u₀ near D = 3.05. `lowD_dd.py`'s rule adds a node 0.1 further out only while the last is below 0.85; the pilot now
  adds one halfway to the horizon (0.95, then 0.975) when a solution's horizon tail passes 10⁻³⁰ or a lifted start's
  10⁻²⁰ (restart by hand at 06:26 to load it).
- **Two passes.** From D = 3.25 the reused plans no longer fit the lifted starts (|F| not finite on them), so most
  solves are two-pass: pass 1 on the start's own plan (550–780 steps, 2.5–3 times the converged solution's 190–270),
  pass 2 on the pass-1 state's own plan. This roughly doubles the time per point.
- **Series orders** (kc = 100, ks = 250, section 1.6): the solutions' centre tails stayed below 10⁻⁴⁹ (10⁻⁶³ after
  the node changes); the horizon tails are the ones that grow (above).

### 5.5 Against the local machine and the formula

- **The local machine** (comms.txt; 13 nodes 0.005…0.8 there, 15–17 nodes here): its M 80 → 96 changes of Δ,
  −1.2×10⁻⁹ (D = 3.70), −3.3×10⁻⁹ (3.65), −4.3×10⁻⁹ (3.60), −2.5×10⁻⁹ (3.55), −1.27×10⁻⁸ (3.50) and −6.6×10⁻⁸ (3.45),
  are −1.24×10⁻⁹, −3.34×10⁻⁹, −4.30×10⁻⁹, −2.51×10⁻⁹, −1.27×10⁻⁸ and −6.65×10⁻⁸ here. Its values agree to all the
  digits it quotes: M = 96 at D = 3.7 (3.4652004823053) and 3.55 (3.4404023464863), M = 80 at 3.40 (3.3662519179569)
  and 3.35 (3.3234621776975), within 3×10⁻¹⁴. Its M = 128 value at D = 3.6, 3.4527209746080, is 2.1×10⁻¹¹ below the
  M = 96 one here, its own M 96 → 128 change there. The node sets do not show at this level (section 3 and 5.4:
  10⁻³⁰).
- **The formula in s = √(D − 3)** (README, fitted to Ecker, Ecker, Grumiller & Jechtl's values, good to about
  10⁻⁵): formula − Δ is −4.7×10⁻⁶ to +6.5×10⁻⁶ over D = 3.65–3.35 and −7.6×10⁻⁶ to −1.3×10⁻⁶ over 3.3–3.175,
  changing sign along D as a fit's error does; the pilot's error estimates (≤ 2.1×10⁻⁸) are far below that, so these
  are the formula's own errors.

## 7. Proposed README text (not applied)

**(a)** In "The double-double solver (`hp/`)", after the bullet "Mixed-precision Newton (opt-in)":

> - **Larger τ grids:** M = 160, 192, 256, 320 and 384 collocation points besides 8–128. Above M = 128 the
>   mixed-precision step builds and factors its f64 Jacobian in block storage (a staircase LU over the
>   multiple-shooting blocks, `hp/src/block.rs`): 8–12 times less memory than the dense matrix (173 MB at M = 192), and
>   the same steps as the dense path to its rounding at M ≤ 128, where every other result is unchanged bit for bit. The
>   exact double-double Jacobian, the Levenberg–Marquardt fallback and γ stay at M ≤ 128. At D = 3.7, Δ changes by
>   −3.4×10⁻¹¹, −3.2×10⁻¹⁵ and −1.6×10⁻¹⁸ from M = 96 to 128, 160 and 192 ([notes/HIGH_M.md](notes/HIGH_M.md)).

**(b)** In the "Below D = 4" caption, replace "cross the double-double kernel's limit of 128 near D = 3.4, which is
where this computation stops; near D = 3 they run into the hundreds." by:

> cross 128 near D = 3.35, where the computed change of Δ from M = 96 to 128 is 2×10⁻⁸. The kernel now runs to
> M = 384; a pilot continuation in double-double reached D = D_LOW with M up to M_TOP (each value with the change from
> the next lower M as its error estimate, ERR_LOW at D_LOW), and the model, calibrated on it, puts D = 3.05 at
> M ≈ 320 and more than a day on two cores ([notes/HIGH_M.md](notes/HIGH_M.md)).

**(c)** In "A formula for 3 < D ≤ 4", replace "and match our own double-double values at D = 3.7–4, which the fit
never saw, to 4×10⁻⁶ and 10⁻⁷" by:

> and match our own double-double values at D = D_LOW–4 (Δ; γ at 3.7–4), which the fit never saw, to FORMULA_MAX
> (Δ) and 10⁻⁷ (γ)

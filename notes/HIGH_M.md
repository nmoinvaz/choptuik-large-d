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

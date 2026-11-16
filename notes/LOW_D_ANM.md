# Low D by the Taylor continuation in D: a design and a pilot (#927)

Written 2026-11-16 by a subagent on the Mac, branch `wip/sa-927`. Raw output in folder `I927-mac/` of the raw-data
repository. The question is #927's: can the asymptotic numerical method (ANM) of `hp/anm_run.py`, the exact Taylor
continuation in D[^cochelin], drive the low-D legs instead of `hp/lowD_highM.py`'s Lagrange predictor and Newton?

## Answer

PENDING

## 1. What already exists at low D

The low-D Taylor continuation was built and run on 2026-11-04/05 ([LOW_D.md](LOW_D.md) 14–16), and #927's text does
not mention it. Its pieces:

| file | what it does |
|---|---|
| `hp/anm_lowd.py` | `coefficients`: z₁..z_N at a converged state, each order's right-hand side −[F]_k from `Solver.residual_jet`, solved by `driver.gmres_refine` (the f64 LU of `Solver.newton_mixed` as preconditioner, exact double-double products by `Solver.jvp`, stopped at a change of 10⁻²⁰); `evaluate`, the partial sum in double-double; the control `JET_MUT_FREEZE_N` |
| `hp/lowD_taylor_leg.py` | the leg: per target, the series at the current base, the prediction planned on its own (`Solver.plan` from the predicted state), Newton with the f64 Jacobian, halvings |
| `hp/jet_lowd_check.py` | the first order link by link (jet orders 0 and 1, `jvp`, the stored states' tangent, the LU solve) |
| `hp/lowD_plan_probe.py` | where the planner freezes on predicted states |
| `hp/src/python.rs` `residual_jet_sized` | the double-double jet residual at every M the kernel builds (generic in the block size B = M/8), since a2f5d9a |

What they found (raw folders `T106-win2`, `T107-win2`, `T115-win2`, `T135-win`, `I798-win`):

| run | base | result |
|---|---|---|
| T106 D″, D‴ | D 3.045, M 320 and 384, N 4 | all orders clean (GMRES 4 products each); ratios \|z_(k−1)\|/\|z_k\| 0.0515, 0.0493, 0.0476 (M 320) and 0.0514, 0.0494, 0.0491 (M 384), about 1.1 (D − 3); prediction at 3.04375 with \|F\| 9.8×10⁻⁶ on its own plan, where the cubic had 6.6×10⁻⁴; Newton from it with full steps to T105's Δ within 2.2×10⁻³¹ (M 320) and 3.5×10⁻²⁹ (M 384) |
| T106 D′ | D 3.045, M 320, N 15 | orders 1–4 as above, then a jump of 10⁵ at order 5 and ratios 1.5–5.6×10⁻⁶ after: high orders polluted |
| T107, T115 | D 3.04375, M 320/384, N 4 then 3 | z₄ polluted at this base (ratio 2.7×10⁻⁷ at M 320); the plan of the predicted state at 3.0425 froze in the segment 0.01 → 0.02 at x 0.013725 (1.3 million steps without moving x) |
| T135, #798 | D 3.04375, M 384, N 3 | the plans of the partial sums n = 1–3 freeze for D ≤ 3.0431 (h ≤ −0.00065) and finish for D ≥ 3.0432; the base state itself plans to 3.04305; the frozen x converges with the order (0.013725 at 3.0425) |

So the ANM predictor is about 70–300 times closer to the branch than the cubic at the same step, Newton from it takes
full steps, and the series' radius is about D − 3 (0.044 at 3.044). It stopped on two things: the high orders are
polluted from order 4–5 near D 3.044, and every predicted state below D ≈ 3.0431 freezes the planner.

## 2. What anm_run.py assumes, against what the low-D legs need

Each row names where the assumption sits in the code, what the low-D legs have instead, and what a low-D driver does
about it. "Low D" here is `hp/lowD_highM.py` with `hp/lowD_dd.py` near D 3.04: M 320 and 384, 21 nodes from x 0.000157
to 0.975, L = 4M(K + 1) + 1 = 28161 and 33793 unknowns.

| # | anm_run.py's assumption | where | low D | for a low-D driver |
|---|---|---|---|---|
| 1 | M = 32 on the octo- and quad-double paths | `--jet_o` and `--corrector od` exit unless M = 32 (`anm_run.py` `main`); `residual_jet_o`, `series_o`, `newton_o`, `olu_solve` dispatch through `od_sizes!` (`hp/src/python.rs`, `Ctx<4>` only); `residual_jet_q`, `newton_q`, `qlu_solve` through `qd_sizes!` (M 32, 48) | M 320/384, block sizes B = 40/48 | only the double-double path runs: `Solver.residual_jet` (`residual_jet_sized`, generic in B) |
| 2 | a dense LU, one refinement per order | `coefficients` in `anm_run.py`: `s.lu_solve` after `sp.newton`, then `--refine` steps of one `s.jvp` | a dense double-double Jacobian would be 28161² × 16 B = 12.7 GB; Newton uses `Solver.newton_mixed`'s f64 Jacobian in block storage (577 MB at M 320) | the f64 block LU as a preconditioner and `driver.gmres_refine` with exact products, as `anm_lowd.coefficients` does (3–4 products per order) |
| 3 | pinning: p₁'s harmonics in (kfreeze, 13] held at a fit, their Taylor coefficients added to each order's right-hand side | `PinnedSolver`, `PinnedQ`, `PinnedO`, `HeldFit` (`hp/pinning.py`); `coefficients(held=...)`; `make_solver(kfreeze=...)` | every harmonic solved; block storage refuses kfreeze (`jacobian_block` in `python.rs`) | dropped, nothing replaces it |
| 4 | the horizon data re-seeded because their errors grow by G = 0.7^−(D−3) (10²⁰ at D 131) | `Reseeder`, `ElimQ` (`hp/reseed.py`), `ElimO` (`hp/octo_corrector.py`), `--horizon_alpha` | G ≈ 1 near D 3; the obstacle is instead the sonic line x = c inside the domain ([HIGH_M.md](HIGH_M.md) 1.5) | dropped; the sonic-line analogue of the re-seed is #926's projection |
| 5 | octo-double jets and corrector, since rounding grows like G² | `--jet_o`, `--corrector od` | double-double jets are clean to order 3–4 near D 3.044 and polluted after (section 1) | double-double; a quad-double jet at M 320 would need `qd_sizes!` for B 40/48 and a quad-double block LU, which multishoot does not have (days of Rust, a release, rebuilds), with no evidence that precision is the cause |
| 6 | D carried in mpmath (`D_str`), the step exact | `anm_run.py` (`mp.mpf(m["D_str"])`) | D is a float on a grid (3.04335); states store float D | float D; it moves Δ by about dΔ/dD × 4×10⁻¹⁶ = 3×10⁻¹⁵, below the 10⁻¹² checks |
| 7 | the step a = min(a_max, h_max), a_max = (ε \|z₁\|/\|z_N\|)^(1/(N−1)); a failed correction halves a (three times) | `anm_run.py` step loop | targets on a D grid shared by two resolutions (the pair M_w, M_n; err_est is their difference); a failed point inserts the midpoint | step = min(a_max, the next grid D); a series at each of M_w and M_n, or M_w's lifted with `driver.resample_z` |
| 8 | the plan is taken once, at the first base; each prediction is evaluated on it and planned anew only when its \|F\| ≤ 10⁻⁶ | `anm_run.py` attempt loop (`segs_n = segs`; `if Fp <= 1e-6: segs_n = sn.plan(...)`) | `lowD_highM.py` reuses the last converged state's plan (`plan_prev`); `lowD_taylor_leg.py` and `anm_lowd.py` plan from the predicted state, the step that froze (T107, T115, T135) | anm_run.py's rule: the base's plan for the series, the prediction and the corrector |
| 9 | the corrector: deflated Newton then plain steps (dnp), or quad/octo-double Newton with stall acceptance | `correct` in `anm_run.py`, `newton_od` | `lowD_dd.solve_point`: Newton with the f64 Jacobian, a checkpoint per step, `--stop_lam`, `--floor_ratio`, two passes | `solve_point` unchanged, started from the prediction on the fixed plan |
| 10 | states in eight parts with `D_str` | `save_o`, `load_o` | two parts with their plan (`driver.save`); `--auto_nodes` re-nodes mid-leg | two parts; a re-noding changes the unknowns, so the series is formed again after it |

[^cochelin]: B. Cochelin, Comput. Struct. 53, 1181–1192 (1994), as in [ALGORITHM.md](../ALGORITHM.md)'s footnote 2.

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

## 3. In what sense the series is regular

#927 calls the ANM predictor "the system's own series in D, regular by construction". That holds in one sense and
fails in another, and the difference decides the design.

- **On the base's plan it holds.** The residual on a fixed plan is a composition of Taylor steps between fixed points
  in x, so F(z, D) is smooth in both, and the jets give the Taylor coefficients of the solution z*(D) of that fixed-plan
  system. The partial sum P_N then has F(P_N, D) = O(h^(N+1)) on that plan, within the series' radius.
- **At the sonic line it fails.** Regularity at x = c, where the p equation is singular, is a property of the solution
  alone. A state off the branch by ε carries a singular part of about ε there, which is why a predictor's own plan has
  2–4 times the steps of a solution's ([HIGH_M.md](HIGH_M.md) 1.5). A truncated series is such a state, of every order.
  A model of the freeze, not shown: the planner keeps each step's truncation term at 10⁻³², so near a singular point of
  strength ε > 10⁻³² its steps shrink geometrically and never pass it. This fits T135's table, where every order froze
  at the same D and the frozen x converged with the order.

So the ANM predictor can help only where the prediction is never planned. anm_run.py already works that way (row 8).
`lowD_taylor_leg.py` did not, and the step that froze was its planning of the prediction, not the series. Whether the
partial sums are finite on fixed plans below D 3.0432, where the planner freezes and #343's Lagrange predictor read
NaN, is measured in section 4.

## 4. Measurements on the Mac (#927)

PENDING

## 5. The design: the series as a start inside lowD_highM.py

A new driver modelled on anm_run.py would have to rebuild what `lowD_highM.py` already does at low D (the pair of
resolutions, the move up in M, the re-noding, the halving, the checkpoints, the two passes). The smaller change keeps
that driver and adds the series as one more start for the working resolution M_w. Python only, no kernel change.

1. **`hp/lowD_dd.py` `solve_point`** returns its `Solver` with the converged state (about 5 lines). Its LU is already
   at the final z: the function refreshes it there for the sensitivities ("the LU stored in s must be at the final z").
   So the series needs no Jacobian of its own. In T106 the base's Jacobian was 2493 of the 5500 s that the base and four
   orders took at M 320.
2. **`hp/lowD_highM.py`**, `--predictor anm --anm_N 3` (the default stays Lagrange). After the M_w solve at D_l, keep
   its Solver. At the next target D_t, form z₁..z_N with `anm_lowd.coefficients` on that solve's plan, add the start
   ("anm", P_N(D_t)) to `starts_w`, and evaluate every start on the same plan, the one the jets used. `plan_prev`
   plans the last state anew at 10⁻³² instead. Taking the solve's own plan keeps the prediction's O(h^(N+1)) residual
   (section 3), and the plan check after convergence (`plan_check`) still reports whether it was lean. About 80 lines.
3. **M_n keeps its lift** of the M_w solution (`lifted`, then two passes). The lift's |F| is the resolution difference
   (1.3×10⁻⁵ at M 384, D 3.04335, #343), which a series at M_n would not improve, for another N orders' cost.
4. **The step** stays on lowD_highM.py's grid. The records carry the series' ratios and a_max = (ε |z₁|/|z_N|)^(1/(N−1))
   as anm_run.py computes it, so that `--dD` can follow them later. A failed point halves the step and reuses the
   series, as lowD_taylor_leg.py did.
5. **Checks that can fail:** the smoke test of T107 A at M 48 (D 4 → 3.99, 3.98); the control `--anm_mutate freeze`
   (`JET_MUT_FREEZE_N`), whose "anm" start must equal the last state, with the same |F|; and `--anm_mutate drop2`,
   whose start must lose its advantage over the Lagrange start (|F| up by about |z₂| h²).

**Cost.** In code about 85 lines and the three checks, under a day. In compute, per point at M 320 on win2, N = 3
orders cost about 3 × (335–642 s of jets + 148–234 s of solves) ≈ 30–45 min at 8 threads (T106), against Newton
calls of 25–40 min each (#343's leg 2). #343's D 3.04335 took 12 calls at M 320 from the Lagrange start (|F|
6.4×10⁻⁴), and T106's D″ took 5 from the series' start (|F| 9.8×10⁻⁶) on a fatter plan. If that ratio holds, the series
saves 4–7 calls (2–4 h) per point where both starts converge, a gain of about 2× at M_w and nothing at M_n.
Forming the series one order at a time, as `Solver.series_o` does in octo-double (a third of the cost at D 253, #914),
would need a double-double `series` for B 40/48 in `hp/src/python.rs`: a Rust change and a rebuild, not needed for a
first test.

## 6. The pilot at M 128

PENDING

## 7. What it decides

PENDING

## Files

- `hp/lowD_fixed_plan.py`: a stored series' partial sums and the Lagrange predictor at target D, on given plans,
  without planning from them; against converged states where they exist; `--mutate drop2`.
- `hp/anm_lowd_pilot.py`: one step (series, prediction and Newton on the base's plan) checked against lowD_highM.py's
  state at the target; `--mutate drop2|kappa|dshift`.
- Raw: `I927-mac/` of the raw-data repository (its README has the commands and the criteria as fixed before each run).

[^cochelin]: B. Cochelin, Comput. Struct. 53, 1181–1192 (1994), as in [ALGORITHM.md](../ALGORITHM.md)'s footnote 2.

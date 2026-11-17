# Low D by the Taylor continuation in D: a design and a pilot (#927)

Written 2026-11-16 by a subagent on the Mac, branch `wip/sa-927`. Raw output in folder `I927-mac/` of the raw-data
repository. The question is #927's: can the asymptotic numerical method (ANM) of `hp/anm_run.py`, the exact Taylor
continuation in D[^cochelin], drive the low-D legs instead of `hp/lowD_highM.py`'s Lagrange predictor and Newton?

## Answer

**The Taylor series does not carry the low-D legs past the point where they stall.** At D 3.04315, where #343's leg 2
met a NaN Lagrange predictor and stalled from its last state, the series' order-1 and order-3 sums from D 3.04375
(M 384) have a non-finite residual on both fixed plans, as the Lagrange predictor does (section 4.3). The order-3 sum
fails from 3.04325 on, in the segment x 0.01 → 0.02 where T135's planner froze. At 3.04335, where the leg converged, it is finite
and 7000 times closer to the solution than the Lagrange predictor. The obstacle is the regularity at the sonic line,
which no truncated series has (section 3), so the decision goes to #926's projection.

Where the starts are finite, the series is the better start. Section 5 puts it into `lowD_highM.py` as one more start,
about 85 lines of Python and no kernel change, for an estimated 2× fewer Newton calls at the working resolution. The
pilot of section 6 takes such a step at M 48 from D 3.7 to 3.695. Its prediction agrees with lowD_highM.py's converged
Δ to 8.7×10⁻¹⁷ and its corrected state to 3.1×10⁻³⁰, and the check fails under a dropped z₂, a wrong D derivative and
a shifted D. A low-D Taylor continuation
already existed (`hp/anm_lowd.py`, `hp/lowD_taylor_leg.py`, T106 and T107, 2026-11-04/05). Its legs froze in the
planner near D 3.0425 for the same reason, in the same segment of x.

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

So the ANM prediction's residual at 3.04375 was 67 times below the cubic's, Newton from it took full steps, and the
series' radius is about D − 3 (0.05 at 3.045). It stopped on two things: the high orders are polluted from order 4–5
near D 3.044, and every predicted state below D ≈ 3.0432 froze the planner.

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
| 8 | the plan is taken once, at the first base; each prediction is evaluated on it and planned anew only when its \|F\| ≤ 10⁻⁶ | `anm_run.py` attempt loop (`segs_n = segs`; `if Fp <= 1e-6: segs_n = sn.plan(...)`) | `lowD_highM.py` reuses the last converged state's plan (`plan_prev`); `lowD_taylor_leg.py` and `anm_lowd.py` plan from the predicted state, where T107, T115 and T135 froze | anm_run.py's rule, the base's plan for the series, the prediction and the corrector; it fails at about the same D (section 3) |
| 9 | the corrector: deflated Newton then plain steps (dnp), or quad/octo-double Newton with stall acceptance | `correct` in `anm_run.py`, `newton_od` | `lowD_dd.solve_point`: Newton with the f64 Jacobian, a checkpoint per step, `--stop_lam`, `--floor_ratio`, two passes | `solve_point` unchanged, started from the prediction on the fixed plan |
| 10 | states in eight parts with `D_str` | `save_o`, `load_o` | two parts with their plan (`driver.save`); `--auto_nodes` re-nodes mid-leg | two parts; a re-noding changes the unknowns, so the series is formed again after it |

## 3. In what sense the series is regular

#927 calls the ANM predictor "the system's own series in D, regular by construction". That holds in principle on a
fixed plan and fails in practice at the sonic line, below D ≈ 3.0433.

- **In principle, on a fixed plan.** The residual on a fixed plan is a composition of Taylor steps between fixed points
  in x, so F(z, D) is smooth near the base, and the jets give the Taylor coefficients of the fixed-plan system's
  solution z*(D). The partial sum P_N then has F(P_N, D) = O(h^(N+1)) on that plan, as long as F stays finite between
  P_N and z*(D).
- **In practice, at the sonic line.** Regularity at x = c, where the p equation is singular, is a property of the
  solution alone. A state off the branch by ε carries a singular part of about ε there, which is why a predictor's own
  plan has 2–4 times the steps of a solution's ([HIGH_M.md](HIGH_M.md) 1.5). A truncated series is such a state, of
  every order. Below D ≈ 3.0433 the measured consequence is that every predicted state's integration fails in the
  segment x 0.01 → 0.02, on a fixed plan (section 4.3, from h = −0.0005 with an order-3 error of about 10⁻⁵) as well as in the
  planner (T135 and #798, from h = −0.00065, at x near 0.0137 in the same segment).

The two onsets lie within 0.00015 of each other in D and in the same segment, so the obstacle is the predicted state,
not the planning of it. `lowD_taylor_leg.py` froze in the planner, and on the base's plan it would have read NaN at
about the same D. Keeping the base's plan, anm_run.py's rule (row 8), does not get past the obstacle. It only spares
the planning time above it.

## 4. Measurements on the Mac (#927)

All on the Mac under load (one process, 4 threads, nice 10), with `/Users/nathan/venv-314`'s hpkernel of 2026-11-08,
which has `residual_jet` at every M; the main checkout's `.venv` hpkernel has no jet residual.

**4.1 The series at D 3.175, M 128** (`I927-mac/series_D3.175_M128.log`). From lowD_highM.py's converged state on its
618-step plan: `newton_mixed` 257 s, then per order 37–60 s of jets and 12–20 s of GMRES (3–4 products, all
converged). The max-norm ratios |z_(k−1)|/|z_k| for k = 2–7 are 0.129, 0.185, 0.170, 0.220, 0.087 and 0.019. The
radius is about D − 3 here too, and the series is polluted from order 6, one order later than at D 3.045. A
step of h = 0.025 (h/R ≈ 0.15) cannot reach 10⁻¹² before the pollution, which is why the pilot's first design stepped
by 0.001 (section 6).

**4.2 The partial sums against converged states, M 384** (`I927-mac/fixed_plan/agree*`). T135's stored series (z₀..z₃
at T105's M 384 state at D 3.04375, on its stored 8319-step plan) against #343's converged M 384 states:

| target (h) | order 0 | order 1 | order 2 | order 3 | Lagrange through the earlier states |
|---|---|---|---|---|---|
| 3.04355 (−0.0002): max \|P − z\| | 2.83 | 1.13×10⁻² | 4.75×10⁻⁵ | 1.98×10⁻⁷ | 2.83 (one state) |
| 3.04355: Δ(P) − Δ | +1.62×10⁻³ | +2.21×10⁻⁶ | +4.22×10⁻⁹ | +1.03×10⁻¹¹ | +1.62×10⁻³ |
| 3.04335 (−0.0004): max \|P − z\| | 5.68 | 4.54×10⁻² | 3.82×10⁻⁴ | 3.18×10⁻⁶ | 2.28×10⁻² (two states) |
| 3.04335: Δ(P) − Δ | +3.24×10⁻³ | +8.86×10⁻⁶ | +3.38×10⁻⁸ | +1.65×10⁻¹⁰ | +4.44×10⁻⁶ |

At h = −0.0004 each order gains a factor of about 120 in the max norm and 200–370 in Δ, so the radius is about 0.05
in the state (D − 3 = 0.044) and 0.08–0.15 in Δ. At 3.04335 the order-3 sum lies 7000 times closer in the max norm, and
27000 times in Δ, than the Lagrange predictor through the same two D that #343's leg used there at M 320. The control
`--mutate drop2` (z₂ left out) moves orders 2 and 3 at 3.04335 to +8.86×10⁻⁶ and +8.83×10⁻⁶, the size of z₂'s term, as
fixed before the run.

**4.3 Residuals on fixed plans below D 3.0434** (`I927-mac/fixed_plan/`). The same series and #343's states, each
prediction's max |F| on fixed plans, with no planning from it. "Base" is the series' own plan (T105's state, 8319
steps). "3.04335" is the stored plan of #343's converged state there (8090 steps), the plan the leg would reuse.
lowD_highM.py re-plans that state at 10⁻³², which this stands in for.

| D (h) | prediction | \|F\| on the base's plan | \|F\| on 3.04335's plan |
|---|---|---|---|
| 3.04335 (−0.0004) | order 3 | 1.33×10⁻⁶ | |
| 3.04325 (−0.0005) | order 3 | nan (segment 0.01 → 0.02) | |
| 3.04315 (−0.0006) | order 3 | nan | nan |
| 3.04315 | order 1 | nan | nan |
| 3.04315 | Lagrange through 3.04375, 3.04355, 3.04335 | nan | nan |
| 3.04315 | order 0 (the series' base state, D 3.04375) | | 8.10×10⁻² |

The order-3 sum is finite at 3.04335, where the leg converged, and close to the solution there (4.2). At 3.04325 and
3.04315 it is not finite, and at 3.04315 neither is the order-1 sum or the Lagrange predictor, on either plan. The
Lagrange case reproduces at M 384 what #343's leg 2 met at M 320. In the two cases where the probe recorded where (the
order-3 sums at 3.04325 and 3.04315 on the base's plan), the non-finite rows are those of the segment x 0.01 → 0.02,
where T135's planner froze at x 0.0137. The series' base state itself, 0.0006 away in D and so about |z₁| h ≈ 8 off
in the max norm (an estimate), is finite at 3.04315. The farther state passes and the nearer ones fail, as in T135's
table, where the base state planned down to 3.04305. So accuracy does not help: a state near the branch below
D ≈ 3.0433 is not finite in that segment, whatever predicted it.

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
   above the obstacle (section 3; 1.1×10⁻¹³ against the base state's 3.7×10⁻³ in the pilot, section 6), and the plan
   check after convergence (`plan_check`) still reports whether it was lean. About 80 lines.
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
saves 4–7 calls (2–5 h) per point where both starts converge, a gain of about 2× at M_w and nothing at M_n.
Forming the series one order at a time, as `Solver.series_o` does in octo-double (a third of the cost at D 253, #914),
would need a double-double `series` for B 40/48 in `hp/src/python.rs`: a Rust change and a rebuild, not needed for a
first test.

## 6. The pilot

`hp/anm_lowd_pilot.py` is one step of the design's core, on the base's plan throughout. It plans the base at 10⁻³²
(lowD_highM.py's `plan_prev`), takes the f64 LU there, forms the series, evaluates the prediction on that plan and
corrects it with `newton_solve` (jac "f64", as `solve_point`). The check, fixed before the run (`I927-mac/README.md`
section 3), compares with lowD_highM.py's own converged state at the target. (A) Δ of the prediction within 10⁻¹² of
the reference. (B) The corrector converged, and its Δ within 10⁻¹². Three controls must each fail the check: z₂ left
out of the sum (`drop2`), the jets with κ's dependence on n frozen, a wrong D derivative (`kappa`), and the step taken
to D + 10⁻⁹ (`dshift`, which must also fail B).

The design put the pilot at M 128 near D 3.15, from lowD_highM.py's states there with steps of 0.001 (section 4.1
rules out 0.025). The Mac's load rose to 100–137 before any pilot run, and an M 128 Jacobian took 257 s at a load of
22, so it moved, by an amendment written before the runs, to the smallest M where lowD_highM.py has a stored pair at
low D: M 48/64 at D 3.7 (`lowDv_D3.7000_M48.npz`, 13 nodes 0.005 to 0.8). The references are a lowD_highM.py run from
that pair with `--dD 0.005`, and the step is 0.005 (h/R ≈ 0.007). The M 128 design stays runnable as written
(`run_pilot.sh` with `PAIR=128`).

**Result, step 1** (D 3.7 → 3.695, `I927-mac/pilot48/step1.log`): PASS. The base's plan has 63 steps and its LU's
pivots run from 1.8×10⁻³ to 4.9. All six orders' GMRES converged in 4 products, and the ratios |z_(k−1)|/|z_k| are
0.46, 0.54, 0.56, 0.75 and 0.84, so the radius is about D − 3 here too. Against lowD_highM.py's converged M 48 state at
3.695:

| order n | max \|P_n − z\| | Δ(P_n) − Δ |
|---|---|---|
| 0 | 2.60×10⁻¹ | +3.07×10⁻⁴ |
| 1 | 2.79×10⁻³ | +1.46×10⁻⁵ |
| 2 | 2.61×10⁻⁵ | +9.23×10⁻⁸ |
| 3 | 2.31×10⁻⁷ | +4.89×10⁻¹⁰ |
| 4 | 1.54×10⁻⁹ | +2.56×10⁻¹² |
| 5 | 9.14×10⁻¹² | +1.46×10⁻¹⁴ |
| 6 | 1.03×10⁻¹³ | +8.74×10⁻¹⁷ |

(A) holds at 8.7×10⁻¹⁷. The prediction's |F| on the fixed plan is 1.1×10⁻¹³, against 3.7×10⁻³ for the base state.
Newton from it took 3 calls with full steps to |F| 1.4×10⁻³⁰, and (B) holds at 3.1×10⁻³⁰. lowD_highM.py itself took
four Newton calls in its first pass and two in its second from its start of |F| 3.7×10⁻³.

**Controls**, each with exit 1:

| control | Δ(P₆) − Δ | (A) | (B) |
|---|---|---|---|
| `drop2`, z₂ left out of the sum | +1.45×10⁻⁵ | fails | not run |
| `kappa`, the jets with κ's n frozen | −2.31×10⁻² | fails | not run |
| `dshift`, the step to D + 10⁻⁹ | +6.43×10⁻¹¹ | fails | fails, +6.43×10⁻¹¹ after convergence |


## 7. What it decides

#927 asked whether to invest in the Taylor continuation at low D, or keep the Lagrange predictor with the sonic-line
projection of #926.

- **Keep the Lagrange predictor and do #926 first.** The series does not remove the NaN predictors at the sonic line.
  At D 3.04315 its sums are as non-finite as the Lagrange predictor on the plans the leg would use. Longer steps toward
  D = 3 need a start that is regular at x = c, whatever predicted it.
- **The series is worth adding after #926 works, and combines with it.** A projection maps a predicted state onto the
  sonic line's regularity conditions, so the closer the predicted state, the smaller the projection's change. The
  order-3 sum is 7000 times closer than the Lagrange predictor at 3.04335 (section 4.2). The change is section 5's,
  about a day of Python.
- **No quad- or octo-double machinery is needed at low D.** None of anm_run.py's large-D parts (pinning, the re-seed,
  octo-double, M = 32) addresses a low-D obstacle (section 2).

## Files

- `hp/lowD_fixed_plan.py`: a stored series' partial sums and the Lagrange predictor at target D, on given plans,
  without planning from them; against converged states where they exist; `--mutate drop2`.
- `hp/anm_lowd_pilot.py`: one step (series, prediction and Newton on the base's plan) checked against lowD_highM.py's
  state at the target; `--mutate drop2|kappa|dshift`.
- Raw: `I927-mac/` of the raw-data repository (its README has the commands and the criteria as fixed before each run).

[^cochelin]: B. Cochelin, Comput. Struct. 53, 1181–1192 (1994), as in [ALGORITHM.md](../ALGORITHM.md)'s footnote 2.

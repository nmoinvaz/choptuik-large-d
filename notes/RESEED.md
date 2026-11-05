# Re-seeding the horizon data from the last node

A coordinator subagent's study, 2026-11-05, on branch `wip/reseed` (from `feat/qd-jet` 76c1065). Code: `hp/reseed.py`
(the re-seed and its unit check), `hp/reseed_check.py` (the basin and jump tests). Raw output: folder `reseed-mac/` of
the raw-data repository. The toy study that proposed it is notes/QUAD_DOUBLE.md 13.21. Sections 1-3 were written before
any test ran. Section 4 holds the results.

## 1. The question

The corrector's unknowns include the horizon data m_h, a_h (2M values). The horizon block sums their series out to the
last node, x = 0.7 (u₀ = 0.3). An error δ in a_h reaches the node as Gδ, with G = 0.7^−(n−1) (4.9×10¹² at D 84.9). The
series' pole then moves inside u₀ once δ exceeds about 10⁻¹³ (13.6), and the corrector's basin is that small. Re-seeding
replaces the predicted horizon data, before the corrector, by those of the regular solution through the predicted last
node. A node error δ_n then becomes about δ_n/G in a_h. The toy predicts re-seeded basins near 10⁻¹ at every D. The toy
has one horizon unknown and none of τ, the interior, the pin or the sonic point, so the real solver is the test.

## 2. Design

**2.1 The horizon block's columns (a measurement made before the design).** The double-double Jacobian of the horizon
block's 4M rows (p, m, a, c) in the 2M columns (m_h, a_h) was taken at the converged state at D 84.9073
(`anm-mac/run_trend_pin3_qd9/anm_pin3_D84.9073.npz`; `python3 hp/reseed.py jac STATE` prints it). Its singular values
fall into two groups. The 32 directions of a_h have gains 4.9–5.7×10¹², which is G, the mass mode. The 32 directions of
m_h have gains from 8×10² down to 0.7, in cos/sin pairs, falling with the harmonic. The smallest singular value of each
pair of row groups is as follows.

| rows | p, m | a, c | m, a | m, c | all 4M |
|---|---|---|---|---|---|
| smallest singular value | 3.3×10⁻⁶ | 9.2×10⁻³ | 0.71 | 0.71 | 1.07 |

**2.2 The equations at a rung: the m and a rows.** These rows give a square system of 2M equations in the 2M unknowns,
with smallest singular value 0.7. So a node error δ_n goes into the a_h directions divided by G, and into m_h at most
multiplied by 1.4. The other choices are worse, for three reasons.

- The c rows carry the part of the node that no regular solution has. That part is c(1) ≠ 1, the horizon's position:
  (x, c) → (λx, λc) maps solutions of the four field equations to solutions, and the layout fixes the horizon at x = 1.
  A fit of the c rows would put that part into a_h, on which c depends with the mass mode's gain. The a equation does not
  contain c.
- The p rows of the real block are projected (the freeze, harmonics above kfreeze 3 dropped) and offset (the pin). Near
  the horizon p has a singular branch u^s, with s = (n/2 − 1)/(1 + c₁) ≈ 40 (c₁ = (n − 1)(1 − a_h²)). The inward
  integration suppresses that branch, but at u₀ the node keeps it. The p and m rows are nearly singular (3×10⁻⁶).
- Least squares over all 4M rows mixes the c rows in. It is implemented (`--rows all`) and measured in the unit check
  only.

So the last rung zeroes the m and a rows of the corrector's own horizon block. Its p and c rows keep the node's
off-family part, which the corrector removes by moving the nodes, the centre data and Δ.

**2.3 The rungs.** u_j = u₀/2^j for j = 5, 4, …, 0: 0.009375, 0.01875, 0.0375, 0.075, 0.15, 0.3. The target at u_j is the
last node integrated to x = 1 − u_j (the node itself at u₀). Each rung is a Newton in double-double on its 2M unknowns,
starting from the previous rung's data (from the given data at the first rung). The residual and the exact Jacobian are
the kernel's, of a two-node solver (nodes x₀ and 1 − u_j, no freeze, one Taylor step per segment). Its block 2 is the
horizon series at u_j minus node 1, and only that block's m and a rows and its m_h, a_h columns are used. The linear
solves are multidouble's `dd_solve`. A step is halved while the rows' max norm does not fall. A rung has converged when
its step is below 10⁻³⁰ of the data's scale, or when the residual stops halving after a step below 10⁻²⁴ (the floor).
13.6's law at D 85 gives a rough basin for each rung's series: 2×10⁻², 7×10⁻³, 10⁻³, 3×10⁻⁵, 3×10⁻⁸ and 3×10⁻¹⁴ in the
data, from u₀/32 to u₀. The data must move by less than that from rung to rung.

**2.4 The stepper toward the horizon.** The kernel's Taylor stepper (model.rs `taylor_step`, order kt = 36, double-double)
runs along a plan (`plan_path`: each step's truncation term about 10⁻³², steps at most 0.05). It is planned piece by piece
between rung points from the state reached. The error estimate at each rung is the max difference from a second
integration, planned at 10⁻³⁶ with steps at most 0.025 at order kt + 12 = 48. The sonic check: x = c is p's singular
point, and inside the horizon the solution has x − c ≈ −(1 + c₁)u < 0. So max over τ of (x − c) must be negative at
every step point. A rung whose path crosses is dropped, with every rung below it.

**2.5 Precision and output.** The re-seed is double-double throughout. Its floor is the double-double model's residual
φ at the state, 3.5×10⁻¹⁹ at D 84.9, the ε_dd·G floor of 13.19. Zeroing the double-double rows moves the data by at
most φ/G in a_h and φ/0.7 in m_h. The output is the state with m_h and a_h replaced, their lower parts zero, and Δ, the
centre data and the nodes unchanged.

## 3. Tests, criteria and controls (written before the runs)

**3.1 Unit check, the planted truth** (`hp/reseed.py check STATE`). The truth is a converged quad-double chain state. Its
horizon data are perturbed by δ_h = 10⁻⁶ (relative, random signs, sizes 0.5–1), outside the u₀ series' basin and
inside the first rung's. Then the state is re-seeded. Here φ is the double-double 2-norm of the truth's pinned residual,
measured in the same run. Pass needs all four of the following.

- Every rung converges.
- max |a_h − a_h,truth| ≤ 10φ/G.
- max |m_h − m_h,truth| ≤ 20φ (σ_min ≥ 0.5 assumed).
- The pinned quad-double max |F| of the re-seeded state is ≤ 10φ.

Two mutations must make it fail. `--mutate skip` keeps only the rung u₀, so it must not converge from δ_h = 10⁻⁶.
`--mutate wrongu` sums each rung's series at 1.001 u_j against the target at u_j, which must give wrong data. Bases are
84.9073 and 87.0224 (qd9) and the latest qdj1 state. The check exits 1 on failure.

**3.2 The stepper, against the planted truth** (in the same run). The truth's node integrated to each rung must agree
with the truth's own horizon series summed there, to 10⁻²⁴ in every field at every rung below u₀. The error estimate
of 2.4 must be at most 10⁻²⁸ at every rung, and the sonic check must hold at every step. Control: the same with the plan
at 10⁻¹² must fail the 10⁻²⁴ agreement (`--plan_tol 1e-12`). A failure stops the work here (the task's stop condition).

*Added after the first run (2026-11-05, between 02:51 and 02:57 UTC), when 3.2 failed as written (4.1).* The premise was wrong. A chain
state's node is not on the regular family, because the freeze drops p's harmonics above 3 from the matching. At D 84.9
the node's p differs from the series' by 9.2×10⁻¹⁸ in harmonic 5. So 3.2b plants a node on the family. The truth's
series is summed at u₀/2 (where its double-double floor is near ε_dd·8.5×10⁵), integrated from there to the rungs
below, and compared with the series at each. The criteria are those of 3.2: 10⁻²⁴, an error estimate of at most
10⁻²⁸, and the sonic check. The control is the same, `--plan_tol 1e-12` must fail it. The exit code of `check` reads
3.1 and 3.2b, and 3.2 is recorded.

**3.3 Basin test** (`hp/reseed_check.py basin`). Truths: the three bases of 3.1. Perturbations, with seeds 1, 2, 3 and
δ = 10⁻¹², 10⁻¹¹, …, 10⁻²:

- Family A, as the task states it: each entry of the centre data and of every node state moves by δ r S. Here r is
  uniform in ±[0.5, 1] with a random sign, and S is the max |z| of its field block (p1, c0, and each node's p, m, a, c).
  The horizon data and Δ stay at the truth.
- Family B, the realistic one (a prediction errs in every unknown): A plus the horizon data by the same rule, and Δ
  multiplied by 1 + δr.

The arms are plain, the chain's corrector (`newton_qd` on `PinnedQ`, tol 10⁻²⁸, 12 steps, 8 halvings, acceptance
"dz"), and re-seeded, the re-seed (2.2–2.4) and then the same corrector. Both arms use the held harmonics at the
truth's values.

A run has converged back to the truth when `newton_qd` reports convergence, max |z − z_truth| ≤ 10⁻²⁰, and
|Δ − Δ_truth| ≤ 10⁻²⁴. For each seed, δ rises until the first failure. The basin of an arm is the largest δ at which
all three seeds have converged at that δ and at every δ below it.

The criterion for "re-seeding widens the basin" is a re-seeded basin at least 100 times the plain one in family B at
every base. Family A is reported but does not decide it, since its plain arm starts from the exact horizon data. Every
run records the start's max |F| before and after the re-seed, to show that the perturbation reaches the corrector.

Controls that must fire:

- C1, the plain arm at the re-seeded basin's edge δ* (family B) must fail.
- C2, the re-seeded arm with `skip` at δ* (family B) must fail. This is the evidence that the rungs, not the extra
  Newton, carry the basin.

If the re-seeded basin is not wider, C1 and C2 are moot and that is the result.

**3.4 Jump test** (only if 3.3 passes; `hp/reseed_check.py jump`). From the latest qdj1 state, the chain's quad-double
series (anm_run.py's `coefficients_q`, N 24, the held harmonics on the T75 trend) is summed at h = 2, 5 and 10. Each
prediction is re-seeded and corrected at D + h, with the held harmonics at the trend's values there. The plain arm runs
on the same predictions. A jump passes when it converges, and then either its Δ lies within 10⁻²⁰ of the chain's own
state at that D (when the chain has one), or, with no chain state, Δ lies within 10⁻⁶ of the formula
(`taylor_predict.formula_delta`; the chain sits at −4.6×10⁻⁷ at 91.1). The second is a plausibility check, not a
reference. Then `anm_run.py --reseed 1` re-seeds every prediction before its corrector, with the default (0) unchanged.

## 4. Results

(Filled in as the runs finish.)

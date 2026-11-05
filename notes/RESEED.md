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

*Amendments of 2026-11-05 03:28 UTC, after the first runs at 84.9073 (4.2) and before the runs they govern.*

- The grid is extended downward, since both arms failed at δ = 10⁻¹² in family B. The plain arm now starts at 10⁻³⁰,
  the re-seeded one at 10⁻¹⁸, by decades. The rule (δ rises per seed until the first failure) and the criteria are
  unchanged.
- A third arm, *elim*, is added: the re-seed before the corrector and again inside it, at every point where the
  corrector takes a Newton step or tries one. That re-seed has the rung u₀ only, warm-started from the point's data, and
  falls back to the full ladder if that Newton fails. So the corrector works on the system with the horizon data
  eliminated, the toy's "elim" formulation. It is a wrapper of `PinnedQ` for the unchanged `newton_qd` (`ElimQ` in
  `hp/reseed.py`). The wrapper returns each step measured from the point the corrector holds, with the re-seed's
  displacement included. A run's result is re-seeded once more before it is compared with the truth. The criteria are
  those of the other arms. The claim "elimination widens the basin" needs an elim basin at least 100 times the plain
  one in family B.
- A first-step diagnostic (`reseed_check.py firststep`, 4.2) is recorded beside the basin runs. It perturbs one group
  of unknowns at a time and records how far the first Newton step departs from the exact correction.

*Amendment 2 of 2026-11-05 03:33 UTC.* At δ = 10⁻²⁸ the plain arm converged (max |F| ≤ 10⁻²⁸) to a point 1.4×10⁻²²
from the truth, with |Δ − Δ_truth| = 1.2×10⁻²³. That fails the criterion |Δ − Δ_truth| ≤ 10⁻²⁴, which was set before
the system's resolution was known. The first step from δ moves the point by 1.4×10³⁴ δ² along a direction where |F|
changes by only about 1.4×10⁻⁸ per unit (a near-null direction, σ ≈ 10⁻⁸). So a point converged to 10⁻²⁸ is fixed only
to about 10⁻²⁰ along it. The criterion stays as written ("back"). The ascent now continues while the corrector converges
(`--stop conv`), so each run records both measures, and both basins are reported: "back" by 3.3's criterion, and
"converged" when `newton_qd` meets its tolerance. Seed 1 of the plain arm continues from 10⁻²⁸, and seeds 2 and 3
restart from 10⁻³⁰.

**3.4 Jump test** (only if 3.3 passes; `hp/reseed_check.py jump`). From the latest qdj1 state, the chain's quad-double
series (anm_run.py's `coefficients_q`, N 24, the held harmonics on the T75 trend) is summed at h = 2, 5 and 10. Each
prediction is re-seeded and corrected at D + h, with the held harmonics at the trend's values there. The plain arm runs
on the same predictions. A jump passes when it converges, and then either its Δ lies within 10⁻²⁰ of the chain's own
state at that D (when the chain has one), or, with no chain state, Δ lies within 10⁻⁶ of the formula
(`taylor_predict.formula_delta`; the chain sits at −4.6×10⁻⁷ at 91.1). The second is a plausibility check, not a
reference. Then `anm_run.py --reseed 1` re-seeds every prediction before its corrector, with the default (0) unchanged.

*Amendment of 2026-11-05 03:30 UTC, before any jump ran.* The chain's records at 91–93 show the series' terms |z_k|h^k
still falling at k = 24 for h = 2 (to 3×10⁻¹⁸) and h = 5 (to 9×10⁻⁹). At h = 10 they grow, to 0.15 at k = 24, so there
the full sum is not a prediction. Each sum therefore stops at the order of its smallest term at h, which is all 24
orders at h = 2 and 5. The full sum's residual is recorded beside it. The arm elim (3.3's amendment) runs on the same
predictions.

*Amendment of 2026-11-05 03:47 UTC, before any jump ran.* The chain qdj1 stopped at 96.1184 (03:17 UTC), and no state past
it exists. So the jumps also run from qdj1's state at 91.1184, at h = 2 and 5, where the chain's own states at 93.1184
and 96.1184 are the truth (the chain stepped by exactly 1, so the D agree to 50 digits). They run before those from
96.1184 at h = 2, 5 and 10, which can be compared with the formula only.

## 4. Results

**4.1 The unit and stepper checks pass at three bases, and their controls fire** (raw-data repository
`reseed-mac/unit.jsonl` and `unit_*.log`; the re-seed takes 12–13 s on two threads). φ is the truth's double-double
residual, the floor of 2.5.

| base D | G | φ | rungs (Newton steps) | \|m_h − truth\| | \|a_h − truth\| | quad-double max \|F\| | 3.1 | 3.2b, worst |
|---|---|---|---|---|---|---|---|---|
| 84.9073 | 4.9×10¹² | 3.5×10⁻¹⁹ | 4, 2, 2, 2, 2, 3 | 5.7×10⁻³² | 1.7×10⁻³² | 1.7×10⁻¹⁹ | pass | 3.1×10⁻³¹ |
| 87.0224 | 1.0×10¹³ | 8.7×10⁻¹⁹ | 4, 2, 2, 2, 2, 3 | 1.0×10⁻³¹ | 1.3×10⁻³² | 2.7×10⁻¹⁹ | pass | 1.9×10⁻³¹ |
| 94.1184 | 1.3×10¹⁴ | 1.2×10⁻¹⁷ | 4, 2, 2, 2, 2, 3 | 4.8×10⁻³² | 1.6×10⁻³² | 4.3×10⁻¹⁸ | pass | 2.2×10⁻³¹ |

From horizon data off by 10⁻⁶ the re-seed returns the truth's to 10⁻³¹, far below the bounds of 3.1 (10φ/G and 20φ).
The quad-double residual it leaves is the double-double floor φ, mostly in the c rows. The first rung's Newton takes
4 steps from 2×10⁻⁶, and each later rung takes 2–3 from the last. The stepper's error estimate is 1.5–3×10⁻³¹ at every
rung. It takes 6 steps per piece, the smallest 1.0×10⁻³ near u₀/32. max (x − c) runs from −0.13 at x = 0.85 to
−8.1×10⁻³ at u₀/32, so no sonic crossing occurs (*numerical evidence*, on converged states).

The controls, at 84.9073:
- `skip` fails 3.1. Newton at u₀ from 10⁻⁶ runs 30 steps and ends at |R| = 3×10⁴³.
- `wrongu` fails 3.1. The rungs below u₀ converge (the series at 1.001u fits a nearby state), and the rung u₀ does not
  (|R| stays at 4.5×10⁻²).
- `--plan_tol 1e-12` fails 3.2b, at 1.9×10⁻¹⁵ against 10⁻²⁴, with its error estimate at the same level. 3.1 still
  passes there. The last rung's target is the node itself, so the intermediate targets need only lie inside the next
  rung's basin.
- `--rows all` (least squares over 4M rows) fails 3.1. a_h comes out 1.0×10⁻²⁰ off against the bound 7×10⁻³¹, and the
  residual is 3.9×10⁻¹⁸. The node's off-family part enters a_h, as 2.2 expected. This supports the m and a rows.

**3.2 as written fails, and the cause is the freeze, not the stepper.** At every rung below u₀ the integrated truth
differs from the truth's series by 6–9×10⁻¹⁹ (84.9), 0.8–1.0×10⁻¹⁸ (87.0) and 1.1–1.4×10⁻¹⁸ (94.1). The difference lies
in p (harmonic 5), m and c, and it hardly decays toward the horizon. The unfrozen quad-double residual of the truth's
horizon block locates it. Its m, a and c rows are 3–6×10⁻⁴⁷, while its p rows are 8.4×10⁻¹⁸, all in harmonics 5 and
above (9.2×10⁻¹⁸ in harmonic 5, falling by about 8 per harmonic). Those are the harmonics the freeze drops. So a chain
state's last node lies off the regular family by about 10⁻¹⁷ in p, and the integration carries that part inward. The
node planted on the family (3.2b) agrees with the series to 2–3×10⁻³¹.

**4.2 The first runs at 84.9073, and the first-step diagnostic** (`reseed-mac/basin/D84.9073_B_*.log`,
`firststep_D84.9073.jsonl`). In family B at δ = 10⁻¹² both arms fail. The plain start has max |F| = 0.6–1.0×10⁵. The
horizon data off by 10⁻¹² put the series' pole inside u₀, and the first Newton step of 10⁴ is refused at every halving
(3 seeds). The re-seeded start has max |F| = 3.7×10⁻¹² (seed 1), node level, with a_h within 5.6×10⁻¹⁶ of the truth.
But the first step is 2.7×10⁻⁹, 2700 times the error and almost all along Δ. The corrector then creeps (λ = 1/8–1/4)
and ends after 12 steps 1.5×10⁻⁹ from the truth. That matches 13.19's failed attempts.

The diagnostic perturbs one group at a time along seed 1's direction. It reports how far the first step departs from the
exact correction −e, as the ratio |dz + e|/|e| (small in the linear regime).

| group perturbed | plain, δ = 10⁻¹² | plain, 10⁻¹⁴ | re-seeded, 10⁻¹² | re-seeded, 10⁻¹⁴ |
|---|---|---|---|---|
| centre data | 4×10⁻¹⁴ | 4×10⁻¹⁶ | 4×10⁻¹⁴ | 4×10⁻¹⁶ |
| nodes | 1.2×10⁻² | 1.2×10⁻⁴ | 2.7×10³ | 27 |
| horizon data | 1.2×10¹⁶ (\|F\| 10⁵) | 2.4×10¹⁴ (\|F\| 0.1) | — | — |
| Δ | 1.8×10⁹ | 2.4×10⁹ | 4.6×10⁻⁸ | 4.6×10⁻¹⁰ |
| all (family B) | 1.2×10¹⁶ | 2.4×10¹⁴ | 2.6×10³ | 26 |

The reading (*numerical evidence* at one base, one direction):
- The centre and node errors are in the linear regime for the plain corrector. Node errors are second order with
  coefficient 1.2×10¹⁰, so they become nonlinear only near 10⁻¹¹. That may be the "about 10⁻¹⁰" basin of the chain.
- Horizon-data and Δ errors are not, at any δ tried. A Δ error alone of 10⁻¹⁴ gives a step 2.4×10⁹ times too large, the
  same ratio as at 10⁻¹², so this is not a second-order term. The Jacobian at a point whose data are not those of a
  regular solution through its node is far from the truth's, in the weak direction (pivot 3×10⁻⁷, with Δ content).
  13.19's jump study saw the same in the full system: 4×10⁻¹⁶ in Δ gave a step of 2.5×10⁻⁶.
- The re-seed puts the point back on that family. A Δ error is then linear (second order with coefficient 5×10⁴), and so
  is a data error, which the re-seed removes. That is the toy's mechanism, and it holds in the real solver.
- But the re-seed moves node-sized errors into the m_h directions of gain about 1 (the high harmonics, 2.1). There the
  equations' second-order term is amplified, in the form G δm²: the a equation is sourced by m², and the mass mode
  carries that to the node. Node errors become second order with coefficient 2.7×10¹⁵, so the first step is linear only
  below about 4×10⁻¹⁶. Those m_h directions are fixed by the node only to the node's own error, so a better inversion
  cannot remove this.

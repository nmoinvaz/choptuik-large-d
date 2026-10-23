# The mixed-precision Newton step at large D (`hp/`)

notes/MIXED_PRECISION.md §8 found that the mixed-precision Newton step (double-double residual, f64 Jacobian, f64 LU)
stops steering at D ≥ 50: its step differs from the double-double step by 0.03–13, the error comes from the f64
Jacobian's horizon block alone, and it lives in 2–4 directions. This note covers the two repairs built for it (a
horizon block in double-double, and a correction loop with exact Jacobian-vector products), the large-D solver with
them, and a benchmark for the local machine. Branch `wip/largeD-mixed`; all runs on the cloud machine (2 cores,
AVX-512), two threads, with private builds; every command runs in `hp/`.

**Summary.**

- **The Newton step is repaired.** Computing the horizon block's columns in double-double (`horizon_dd=4`, the only
  one of the candidate fixes that works) brings the step error at D = 50–60 from 0.13–3.4 to the floor of an f64
  factorization (3.5×10⁻⁷–1.7×10⁻⁵), for a Jacobian 1.7–2.7 times dearer. A correction loop (GMRES in GCR form,
  preconditioned by the f64 LU, with exact double-double Jacobian-vector products) brings it to ≤ 1.1×10⁻¹³ in 4–6
  products on the plain f64 LU and ≤ 5×10⁻¹⁶ in 3 on the horizon-fixed one, at 1/9–1/23 of an f64 Jacobian per
  product (§1–§3).
- **Continuation at large D works with the loop on the horizon-fixed LU (f64hg)**, which converged at every state and
  step tried, with Δ equal to the double-double run's to the level of that run's own linearised Δ correction (§4,
  §5). The plain f64 Jacobian fails from D = 40, the horizon fix alone (f64h) and the loop alone (f64g) fail at some
  states at D = 50–60. What limits them is not the step but the f64 LU's weakest directions: at D ≥ 40 its weak
  subspace has a principal angle of 90° to the exact one and its ∂Δ/∂F is off by up to 10⁵–10⁹ (§4, §7.4).
- **Speedups** (whole solves, same machine, interleaved; §5): 4–8× below D = 35 (f64 or f64g), 1.4–2.5× at D = 40–60
  (f64g, f64hg), against a double-double Newton with a sensible centre-series order. Against the double-double Newton
  as the scan runs it (kc = 420) a continuation point is 4–8 times faster with f64hg (§4). The scan's kc = 420 is about
  three times what the series needs and makes the double-double Jacobian three times slower for bitwise the same
  result (§7.2), the cheapest speedup of all.
- The M = 192 stall at D = 3.7 is not cured by exact steps: the loop corrects the f64 step (by 0.3–4.7 of its size),
  but the exact step's line search still cuts it to 1/2–1/8 (§8).

**Machine load.** The two cores were shared with the high-M agent's three jobs and several of this work's runs at
once; the load average was 6–14 during the measurements below. The accuracy numbers do not depend on it. The times do:
they are inflated by a factor of about 2–6 against an idle machine, and only ratios measured in the same run
(interleaved) are comparable. The benchmark (§5) interleaves the variants for that reason.

## 1. The horizon fix (`horizon_dd`)

**What was built.** `Ctx::f64_dd` (model.rs) holds bit flags that move parts of the f64 Jacobian to double-double;
`Solver.jacobian_f64(..., horizon_dd=FLAGS)` and `Solver.newton_mixed(..., horizon_dd=FLAGS)` set it for one call,
and 0 (the default) changes nothing (§6). The flags are the candidate fixes the task listed, plus two more:

| flag | what is double-double |
|---|---|
| 1 `HDD_SOLVE` | the horizon series' periodic solves (`psolve`), value and tangents, with the full double-double LU factors (the f64 path uses their hi parts) |
| 2 `HDD_NOTAIL` | nothing; the horizon series of the f64 tangents runs to ks instead of stopping at 2⁻⁶⁰ (`Tail`) |
| 8 `HDD_LASTSEG` | the last segment's columns (block K − 1): `Du` lanes, rounded to f64 |
| 16 `HDD_DTN` | the horizon block on a double-double base trajectory (`Dtn<4>`, f64 tangents); 17 adds the solves |
| 4 `HDD_BLOCK` | the horizon block's columns (m_h, a_h, Δ): `Du` lanes, i.e. the double-double Jacobian's columns, rounded to f64 |
| 32 `HDD_DU_CUT` | with 4: those columns' series stopped at double-double resolution (see §2) |

`jacobian_lanes` builds the tasks of the flagged blocks on their own lanes and the rest as before (groups of four
f64 tangents). The driver's `f64h` variant uses `HORIZON_DD = 4`.

**What was measured** (`python3 largeD_horizon.py --D=50,55,60,20,40,45`; `data/mixed_precision/largeD_horizon.jsonl`
and `.log`). At each saved scan state (kfreeze 13 from D = 35), perturbed by 10⁻¹² relative with the two seeds of
`mixed_largeD_blocks.py`, the Newton step with each variant against the double-double step (`Solver.newton`). Each
cell is the relative step difference with scipy's f64 LU (as in the prototypes) / with the kernel's f64 LU and
double-double solve (what `newton_mixed` does). "hi" is the floor of any f64 factorization: an f64 LU of the exact
Jacobian's hi part.

| D, seed | hi | 0 (f64) | 1 | 3 | 8 | 16 | 17 | 4 |
|---|---|---|---|---|---|---|---|---|
| 50.01, 1 | 1.8e-6 / 8.4e-7 | 0.13 | 0.12 | 0.12 | 0.13 | 0.18 | 0.10 | 6.1e-7 / 1.3e-6 |
| 50.01, 2 | 4.8e-6 / 1.3e-5 | 3.4 | 0.28 | 0.28 | 3.4 | 0.59 | 0.38 | 5.4e-6 / 1.7e-5 |
| 55.04, 1 | 1.4e-5 / 6.0e-6 | 3.4 | 2.7 | 2.7 | 3.4 | 11 | 10 | 5.7e-6 / 1.5e-5 |
| 55.04, 2 | 1.3e-5 / 3.7e-5 | 0.96 | 1.0 | 1.0 | 0.96 | 2.2 | 0.83 | 3.7e-6 / 1.3e-5 |
| 60.00, 1 | 5.4e-7 / 1.1e-6 | 2.2 | 0.79 | 0.79 | 2.2 | 0.062 | 0.27 | 3.5e-7 / 6.3e-7 |
| 60.00, 2 | 1.1e-6 / 1.2e-6 | 0.89 | 0.16 | 0.16 | 0.89 | 0.47 | 0.49 | 5.0e-6 / 1.3e-6 |
| 40.02, 1 | 2.7e-6 / 1.9e-6 | 4.7e-4 | 2.0e-4 | 2.0e-4 | 4.7e-4 | 4.5e-4 | 1.6e-4 | 5.0e-7 / 8.8e-7 |
| 40.02, 2 | 7.2e-7 / 1.8e-5 | 5.9e-3 | 2.0e-3 | 2.0e-3 | 5.9e-3 | 9.1e-4 | 1.7e-3 | 4.0e-6 / 1.1e-5 |
| 45.05, 1 | 4.5e-7 / 1.8e-7 | 9.5e-3 | 1.2e-2 | 1.2e-2 | 9.5e-3 | 2.0e-2 | 1.3e-2 | 1.7e-7 / 2.8e-7 |
| 45.05, 2 | 2.1e-7 / 2.6e-6 | 4.5e-3 | 1.6e-2 | 1.6e-2 | 4.5e-3 | 3.4e-2 | 9.0e-3 | 7.3e-6 / 2.3e-6 |
| 20.04, 1 | 6.7e-7 / 4.1e-7 | 2.9e-4 | 1.7e-4 | 1.7e-4 | 3.0e-4 | 1.0e-4 | 2.1e-5 | 1.5e-5 / 1.6e-5 |
| 20.04, 2 | 9.0e-7 / 5.4e-7 | 3.8e-4 | 3.2e-4 | 3.2e-4 | 4.0e-4 | 4.8e-4 | 2.0e-4 | 2.7e-5 / 2.7e-5 |

(Columns 0–17 agree between the two LUs to the digits shown; flag 2 alone gives column 0's numbers and 36 gives
column 4's, bitwise the same Jacobian.)

- Only the horizon block's columns in double-double (4) repair the step. The double-double solves (1) help at a few
  states but leave 0.1–2.7 at D = 50–60. The last segment (8) changes nothing, as the block-replacement test of
  MIXED_PRECISION §8 predicted. A double-double base trajectory (16), even with double-double solves (17), makes the
  horizon entries 1000 times more accurate (8×10⁻¹²–10⁻⁹ relative against 5×10⁻⁹–2×10⁻⁷) and the step no better: the
  step at D ≥ 50 needs those entries to about f64 rounding, so the tangent arithmetic itself has to be double-double.
- **Pass criterion (step error ≤ 10⁻⁵ at D = 50, 55, 60, both seeds): met with scipy's LU (3.5×10⁻⁷–5.7×10⁻⁶), not
  with the kernel's** (6.3×10⁻⁷–1.7×10⁻⁵; three of six cases at 1.3–1.7×10⁻⁵). The kernel's f64 LU of the *exact*
  Jacobian's hi part gives 8.4×10⁻⁷–3.7×10⁻⁵ at the same states, so the horizon-fixed Jacobian is at the floor of an
  f64 factorization, which is about 10⁻⁵ here; no change of the Jacobian can go below it. The correction loop does
  (§3).
- At D = 20 the fix gains only a factor 14–19 (2.9×10⁻⁴ → 1.5–2.7×10⁻⁵): there the f64 errors are not confined to
  the horizon block. At D = 40–45 it brings 5×10⁻⁴–10⁻² to 1.7×10⁻⁷–1.1×10⁻⁵.

**Cost.** In the same runs (load average about 8): f64 Jacobian 0.48–0.79 s, with the horizon block in double-double
1.18–1.80 s, the double-double Jacobian 12.9–21.0 s at D = 40–60 (M = 32, N = 769). So the fix adds about as much as the
whole f64 Jacobian costs (×2.2–2.7), and the fixed Jacobian is still 11–14 times cheaper than the double-double one.
The horizon block is 65 of 769 columns, but each is a double-double horizon series of ks = 100 orders with a tangent.
Stopping those series at double-double resolution (flag 36) gave the same columns bitwise and no saving.
`python3 largeD_mixed_check.py costs` (`data/mixed_precision/largeD_mixed/costs.json`, `.log`; medians of three
interleaved repeats, load average 11.5–13.5, the states' own kc):

| state | residual | dd Jacobian | f64 Jacobian | f64 + horizon in dd | f64 LU | product (full / cut) |
|---|---|---|---|---|---|---|
| D = 8.25, M = 48 | 0.088 s | 26.7 s | 4.40 s | 7.45 s | 0.40 s | 189 / 188 ms |
| 20.04, M = 32 | 0.038 s | 6.30 s | 0.84 s | 2.10 s | 0.077 s | 61 / 56 ms |
| 40.02 (kc 420) | 0.124 s | 19.1 s | 0.62 s | 1.66 s | 0.042 s | 306 / 67 ms |
| 50.01 (kc 420) | 0.119 s | 19.5 s | 0.64 s | 1.66 s | 0.029 s | 292 / 61 ms |
| 60.00 (kc 420) | 0.103 s | 19.0 s | 0.72 s | 1.70 s | 0.046 s | 292 / 64 ms |

The fix makes the f64 Jacobian 1.7–2.7 times dearer at every D. At D = 40–60 the horizon-fixed Jacobian is still
11–12 times cheaper than the double-double one with the scan's kc = 420, and about 3.5 times with a sensible kc (§7.2:
the dd Jacobian then takes about a third as long). One cut product costs 1/9–1/23 of an f64 Jacobian.

## 2. The exact Jacobian-vector product (`Solver.jvp`)

**What was built.** `model::jvp`: J v by one forward-mode pass per block with `Du` lanes (the double-double
Jacobian's lanes), the block's inputs and Δ carrying v's entries as their tangent; then the target node's −v, the
frozen horizon rows (Π_low(·) + Π_high v_p1) and the phase row, as the residual assembles them. The value part of the
same pass is the residual, so `Solver.jvp(zh, zl, vh, vl, with_residual=True)` also returns F (bitwise
`Solver.residual`). `mutate=1|2|3` breaks it for the controls (the Δ tangent scaled by 1 + 10⁻²⁰, the horizon block's
tangent dropped, the phase row dropped).

`cut=True` stops the centre and horizon series once their tangent terms fall below 2⁻¹¹⁰ of the largest, four orders
in a row (`HDD_DU_CUT`), instead of running to kc and ks. At large D the scan raised kc to 420, while the centre
series' terms at the first node fall to 10⁻¹³⁴ by then (D = 50; 10⁻⁵⁸ would be f64 level at order 57, 10⁻³³ at about
140), and the centre series was most of a product's cost: 133 ms at kc = 420, 49 ms at 210, 15–30 ms at 105, the
same on one thread as on two (D = 50). With the cut a product takes 54 ms instead of 182 ms at D = 50 and agrees with
the full one as closely as the full one agrees with the matrix (below). The correction loop uses `cut=True`.

**Check** (`python3 largeD_mixed_check.py jvp`; `data/mixed_precision/largeD_mixed/jvp_check.log`, `.json`). J v
against the double-double Jacobian (hi and lo) times v, summed in double-double (`dd_matvec`, itself checked against
exact rationals on five rows of the D = 20 Jacobian: 5.9×10⁻³³ of Σ|J_ij v_j|, and 2.4×10⁻⁵ with one term dropped as its
control; `dotcheck.log`), v random with lo parts. The measure is max_i |Jv − J·v|_i / (|J| |v|)_i.

| state | full product | cut product | J(v₁+v₂) − Jv₁ − Jv₂ (same measure) | limit | product time (cut) | f64 Jacobian / cut product |
|---|---|---|---|---|---|---|
| D = 3.7, M = 48 → 16 | 3.7×10⁻³¹ | 3.7×10⁻³¹ | | 10⁻²⁸ | 110 (64) ms | 7.5 |
| D = 3.7, M = 48 → 32 | 4.8×10⁻³⁰ | 4.8×10⁻³⁰ | | 10⁻²⁸ | 262 (247) ms | 15 |
| D = 20.04, M = 32 | 1.4×10⁻²⁸ | 1.4×10⁻²⁸ | 1.0×10⁻²⁸ | 10⁻²⁷ | 58 (57) ms | 8.7 |
| D = 50.01, M = 32, kfreeze 13 | 1.6×10⁻²⁶ | 1.6×10⁻²⁶ | 1.3×10⁻²⁶ | 10⁻²⁵ | 182 (54) ms | 8.7 |

The residual from the same pass is bitwise `Solver.residual`, and the full product of a unit vector is bitwise that
Jacobian column, at every state. The two share the model's equations (`block_state` on `Du` lanes); what the check
tests is the product's own part: the seeding of the tangents, Δ, the target node, the frozen rows, the phase row.

**A discrepancy, and what replaced the criterion.** The pass criterion written first was 10⁻²⁸ at every state. It
holds at the low-D states and fails at D = 50 (1.6×10⁻²⁶; 9.2×10⁻²⁷ in the first run). The cause is the same
amplification of rounding that makes the f64 Jacobian fail there (MIXED_PRECISION §3, §8), now at the double-double
level: the dual pass is not exactly linear in its tangent, and J(v₁ + v₂) − J v₁ − J v₂ is 1.3×10⁻²⁶ at D = 50 and
10⁻²⁸ at D = 20 in the same measure (2×10⁻³⁰ at D = 3.7, M = 32). The Jacobian's columns carry the same kind of error
(a tangent on two columns, a e_i + b e_j, differs from a J_i + b J_j by up to 3×10⁻²⁴ (D = 20) and 7×10⁻²³ (D = 50)
relative to |a J_i| + |b J_j|). So no two double-double evaluations of J v can agree to 10⁻²⁸ there. The check keeps the
10⁻²⁸ criterion at the low-D states (the strict tier) and, at D = 20 and 50, requires agreement within 10 times the
measured linearity defect at that state. For the correction loop the products only need f64 accuracy (§3).

**Controls** (`python3 largeD_mixed_check.py controls`; `jvp_controls.log`): each mutation at D = 50.01 and at
M = 16, 6 of 6 fire. The first run of the controls found that mutation 1 did not fire: `1.0 + 1e-20` is 1 in f64, so
the product was unchanged. It now adds 10⁻²⁰ v_Δ in double-double.

**Cost.** One product with the cut is 1/7.5–1/15 of an f64 Jacobian and 1/66–1/240 of a double-double one in the
table's runs.

## 3. The correction loop (`driver.gmres_refine`)

**What was built.** GMRES on J M⁻¹ (right preconditioning), M the f64 LU that the mixed step stored, the products J u
from `Solver.jvp(cut=True)`, written in its GCR form (Eisenstat, Elman and Schultz 1983; the same iterates as GMRES in
exact arithmetic). Each iteration takes u = M⁻¹ r, w = J u (one product), orthogonalizes w against the earlier w's
(modified Gram–Schmidt twice), applies the same coefficients to u, normalizes, and updates x += c u, r −= c w with
c = w·r. u, w, x and r are double-double and updated with the same f64 coefficients, so r stays the residual b − J x
to double-double rounding without recomputing it: an outer refinement loop with recomputed residuals is not needed.
The first iteration's u is M⁻¹ b, the plain mixed step.

**Stopping.** On the change of the step: the loop ends when an iteration changes x by at most `step_tol` |x| (default
10⁻¹⁰; `confirm` iterations in a row, default 1), or after `max_products` (default 30; 12 in the first benchmark run,
§5, which sent most large-D solves to exact steps near the floor; this check used 16). The residual is not used:
an error along a weak direction barely shows in it (MIXED_PRECISION §8, where a residual tolerance of 10⁻⁸ stopped
with the step still off by 6×10⁻⁴).

**Check** (`python3 largeD_mixed_check.py steps`; `data/mixed_precision/largeD_mixed/steps.jsonl`, `steps.log`). Same
states and perturbations as §1; the step error against the double-double step after every product:

| D, seed | f64 | f64g: products, error | errors after each product | f64h (kernel LU) | f64hg: products, error |
|---|---|---|---|---|---|
| 20.04, 1 | 2.9×10⁻⁴ | 4, 2.3×10⁻¹⁵ | 3e-4 1e-7 2e-11 2e-15 | 1.6×10⁻⁵ | 4, 2.6×10⁻¹⁶ |
| 20.04, 2 | 3.7×10⁻⁴ | 4, 6.1×10⁻¹⁶ | 4e-4 4e-8 5e-12 6e-16 | 2.7×10⁻⁵ | 4, 1.2×10⁻¹⁶ |
| 40.02, 1 | 4.7×10⁻⁴ | 4, 4.7×10⁻¹⁴ | 5e-4 7e-7 4e-11 5e-14 | 8.8×10⁻⁷ | 3, 4.2×10⁻¹⁷ |
| 40.02, 2 | 5.8×10⁻³ | 5, 5.7×10⁻¹⁷ | 6e-3 5e-6 1e-9 4e-13 3e-17 | 1.1×10⁻⁵ | 3, 2.9×10⁻¹⁶ |
| 45.05, 1 | 9.5×10⁻³ | 5, 7.0×10⁻¹⁵ | 9e-3 6e-6 8e-9 8e-12 7e-15 | 2.8×10⁻⁷ | 3, 4.8×10⁻¹⁶ |
| 45.05, 2 | 4.5×10⁻³ | 5, 2.4×10⁻¹⁴ | 5e-3 2e-4 2e-8 5e-12 2e-14 | 2.3×10⁻⁶ | 3, 7.5×10⁻¹⁷ |
| 50.01, 1 | 0.13 | 5, 1.1×10⁻¹³ | 1e-1 8e-3 2e-7 8e-11 1e-13 | 1.3×10⁻⁶ | 3, 2.2×10⁻¹⁷ |
| 50.01, 2 | 3.4 | 6, 1.1×10⁻¹⁶ | 3e0 1e-1 2e-6 3e-9 4e-13 9e-17 | 1.7×10⁻⁵ | 3, 3.7×10⁻¹⁶ |
| 55.04, 1 | 3.4 | 6, 8.8×10⁻¹⁵ | 3e0 2e0 6e-4 8e-8 5e-11 9e-15 | 1.5×10⁻⁵ | 3, 3.6×10⁻¹⁶ |
| 55.04, 2 | 0.96 | 6, 1.2×10⁻¹⁵ | 1e0 9e-1 6e-4 3e-7 4e-11 1e-15 | 1.3×10⁻⁵ | 3, 1.4×10⁻¹⁶ |
| 60.00, 1 | 2.2 | 6, 1.1×10⁻¹⁵ | 2e0 7e-1 2e-4 1e-8 2e-12 1e-15 | 6.3×10⁻⁷ | 3, 2.9×10⁻¹⁶ |
| 60.00, 2 | 0.89 | 6, 6.1×10⁻¹⁵ | 9e-1 6e-1 7e-4 1e-7 4e-12 6e-15 | 1.3×10⁻⁶ | 3, 3.1×10⁻¹⁶ |

- The loop on the plain f64 LU (f64g) takes 4 products at D = 20–40, 5 at 40–50, 6 at 50–60, and leaves the step
  within 1.1×10⁻¹³ of the double-double step everywhere; on the horizon-fixed LU (f64hg) 3 products (4 at D = 20) and
  ≤ 5×10⁻¹⁶. Pass (≤ 10⁻⁵) for both at every state. The run's exit code is 1 because its criterion also covered f64h
  alone, which is 1.3–2.7×10⁻⁵ at five states (the f64 LU floor of §1).
- The change of an iteration equals, to the digit, the error before it (the logged changes and errors line up shifted
  by one), so the stopping rule is conservative by one product: when it stops, the error is already about the product
  of the last two changes.
- The loop's time was 0.14–0.34 s per step under load, against 0.5–0.8 s for the f64 Jacobian and 13–21 s for the
  double-double one. The loop on the plain f64 Jacobian is cheaper than the horizon fix (§1: +0.7–1.0 s) and ends
  closer to the exact step.

**Transposed solves.** The loop corrects the forward solve only. With f64g the stored LU is the f64 Jacobian's, whose
horizon block is wrong at D ≥ 50, so J⁻ᵀ e_Δ (the sensitivities ∂Δ/∂F), the weak subspace of the deflated Newton
(inverse iteration with J⁻¹ and J⁻ᵀ) and `delta_correction` from that LU are as wrong as the step was. What the driver
does with f64g / f64hg: the linearised Δ correction −∂Δ/∂F·F is the Δ component of the refined step (the same
quantity for the exact J, so it is exact to the loop's tolerance); the weak subspace and the scan's per-block
sensitivities still come from the f64 LU. §4 measures how far they are from the double-double ones.

## 4. The large-D solver with these

**What was built** (`hp/python/hpkernel/driver.py`). `newton_step(s, zh, zl, jac, mixed=None, gmres=None)` makes one
Newton step with one of five Jacobian variants and returns the step, F, the pivots and the step's seconds by part
(`Solver.step_timing`: residual, Jacobian, LU, solves; plus the products' time and count):

| `jac` | step |
|---|---|
| `dd` | `Solver.newton`: the exact double-double Jacobian and LU (unchanged) |
| `f64` | `Solver.newton_mixed(**mixed)` (MIXED_PRECISION.md, unchanged) |
| `f64h` | `newton_mixed(horizon_dd=4)`: the horizon block in double-double (§1) |
| `f64g` | `f64`, then `gmres_refine` on its LU (§3) |
| `f64hg` | `f64h`, then `gmres_refine` on its LU |

`newton_solve(..., jac=..., mixed=..., gmres=...)` and `deflated_newton(..., jac=..., mixed=..., gmres=...)` take
all five; `scan_hp.py --jac VARIANT [--step_tol T] [--max_products N]` passes it on and records each point's time
split and products (`split` in its jsonl). The stored LU after a mixed step is the f64 LU, so chord steps,
`lu_solve_t` (the sensitivities ∂Δ/∂F), `weak_subspace` and `junk_subspace` use it.

- With the horizon fix the weak subspace and the sensitivities come from the f64 LU of an accurate Jacobian (step error
  10⁻⁵–10⁻⁷, §1); the comparison with the double-double LU is below.
- With the loop alone (f64g) the stored LU is the plain f64 Jacobian's, wrong at D ≥ 50 in a few directions. The Δ
  correction uses the refined step (§3); the deflated Newton's weak subspace and the scan's per-block sensitivities
  come from that LU. A transposed loop would need products with Jᵀ (reverse mode), which the kernel does not have.
- `dd_finish` (default on for f64h, f64g, f64hg): in the plain-Newton phase, once a mixed step is cut below λ = 1/4 or
  finds no decrease, or its loop ends without converging, the remaining steps are exact. `dd_last` (default off; the
  scan's `--dd_last`): once |F| < tol, one exact step (one double-double Jacobian), which also leaves the exact LU for
  the sensitivities. Both only up to M = 128 (the dense limit).

**Real continuation steps** (`python3 largeD_continuation.py --D=60,40,50`; one directory per variant under
`data/mixed_precision/largeD_cont/`, with `scan_hp.py`'s jsonl, log and states, and `report.json`). `scan_hp.py`
runs as the scan ran (kfreeze 13, deflated Newton w = 24, Lagrange predictor through the last four points, the scan's
tolerance and kc = 420, it_max 25, no LM), once per variant, from the same start and history to the same two targets.
The scan's predictor needs a history spaced like its steps, and the saved states are one unit of D apart: a cubic
through them misses by |F| = 0.04 (D = 40) and 0.35 (D = 50) per unit of step, 4×10⁻⁶ and 3.5×10⁻⁵ already for a step
of 10⁻⁴. So:

- D ≈ 60: the scan's own history, kept in `hp/data/seeds`: start 59.8221, history 59.7676, 59.7858, 59.8041 (run up12/
  up13), targets the scan's next two points 59.833359 and 59.847422, tol 3×10⁻²²;
- D = 40 and 50: a double-double ladder first, from the saved state with the three saved states below as history, to
  D₀ + 2×10⁻⁵, 10⁻⁴, 3×10⁻⁴, 10⁻³, 3×10⁻³, 0.01, 0.02, 0.03 (one `scan_hp.py` call per point); then the variants
  from D₀ + 0.03 with the ladder's three points before it as history, to D₀ + 0.06 and D₀ + 0.09. At D = 50 the
  ladder's step from 3×10⁻⁴ to 10⁻³ failed for the double-double Newton (|F| stuck at 4×10⁻¹¹, "step too small";
  `largeD_cont/D50.0075_first_ladder`), so the ladder continues there in ratios of 1.5 (5×10⁻⁴, 8×10⁻⁴, …, 0.0135,
  0.02, 0.03) and the history is 9×10⁻³, 0.0135, 0.02.

Results (`largeD_cont_report.log`; the times are each point's solve, without `scan_hp.py`'s re-convergence of the
start and history; double-double with the scan's kc = 420; load average 3.5–6 for these runs, except f64h at D = 40,
which ran earlier at 9–13):

| D (targets) | variant | points reached | Newton calls | seconds per point | ΔΔ against dd | dd's own dDelta_est |
|---|---|---|---|---|---|---|
| 59.83, 59.85 | dd | 2 of 2 | 7, 8 | 42.4, 48.2 | | −2.6×10⁻¹⁹, −4.0×10⁻¹⁸ |
| | f64 | 0 (10 halvings, "step too small") | | | | |
| | f64h | 6 points at steps of 0.001–0.004 (12 halvings), stopped before 59.847 | 6–9 | 3.6–6.1 | +2.2×10⁻¹⁶ at 59.833 | |
| | f64g | 1 (59.824; 11 halvings) | 8 | 15.9 | | |
| | **f64hg** | **2 of 2** | 8, 8 | **10.4, 10.7** (147, 146 products) | **+3.9×10⁻¹⁹, +2.6×10⁻¹⁸** | |
| | f64hg + exact last step | 2 of 2 | 8, 8 | 16.6, 17.0 | +3.9×10⁻¹⁹, +2.6×10⁻¹⁸ | |
| 40.08, 40.11 | dd | 2 of 2 | 6, 6 | 41.2, 36.8 | | −5.5×10⁻²², −2.8×10⁻²² |
| | f64 | 0 (10 halvings) | | | | |
| | f64h | 2 of 2 (exact finish) | 14, 7 | 114, 54 (load 12) | −4.9×10⁻²³, +6.9×10⁻²² | |
| | **f64g** | **2 of 2** | 6, 6 | **5.0, 5.5** (96 products each) | **−2.2×10⁻²², −5.2×10⁻²²** | |
| | f64hg | 2 of 2 | 6, 6 | 6.5, 6.5 (69, 67) | −4.9×10⁻²², −7.4×10⁻²² | |
| | f64g / f64hg + exact last step | 2 of 2 | 6, 6 | 11.4–12.2 | as without | |
| 50.07, 50.10 | dd | 2 of 2 | 8, 7 | 54.3, 48.5 | | −6.7×10⁻²¹, −7.9×10⁻²⁰ |
| | f64 | 0 | | | | |
| | f64h | 0 (8 halvings, "step too small") | | | | |
| | **f64g** | **2 of 2** | 8, 7 | **7.8, 6.5** (166, 145) | **−1.8×10⁻¹⁹, −1.2×10⁻¹⁹** | |
| | f64hg | 2 of 2 | 7, 8 | 7.1, 9.8 (108, 126) | −8.0×10⁻²⁰, −1.2×10⁻¹⁹ | |
| | f64g + exact last step | 2 of 2 | 8, 7 | 13.2, 13.0 | −7.1×10⁻²⁰, −2.7×10⁻¹⁹ | |
| | f64hg + exact last step | 2 of 2 | 7, 8 | 14.7, 15.4 | −8.0×10⁻²⁰, −1.2×10⁻¹⁹ | |

- **f64hg did every step at all three D**, 4–8 times faster per point than the double-double Newton as the scan ran
  (kc = 420); with a sensible kc the double-double point would take about a third (§7.2), and the gain is then about
  1.5–2.5×, as in the benchmark (§5). Its Δ agrees with the double-double run's to within that run's own linearised Δ
  correction, i.e. to the precision the tolerance fixes Δ. At D = 59.83 the scan's recorded values differ from both by
  2–5×10⁻¹⁸ (its predictor came from a longer history).
- f64g did every step at D = 40 and 50, as fast as f64hg or faster, but failed at 59.8 (in the deflated phase: its weak
  subspace comes from the plain f64 LU). f64h failed at 50 and 59.8 (and at 59.8, where it did converge on shorter
  steps, Δ was off by 2.2×10⁻¹⁶, 1000 times the double-double run's dDelta_est; its own dDelta_est, from the f64 LU, said
  −1.6×10⁻¹⁵). The plain f64 Jacobian failed everywhere.
- The exact last step (`--dd_last`) costs one double-double Jacobian (6–7 s at kc = 420) and changed Δ by at most
  1×10⁻¹⁹; what it buys is the stored LU: the record's ∂Δ/∂F is then the double-double one (8.1×10¹³ at D = 59.83,
  against 1.8×10⁸ from f64hg's f64 LU).

**Sensitivities and the weak subspace from the f64 LU** (the same report, at the double-double run's last point; each
variant's LU from one step there against the double-double LU's):

| D | f64 / f64g LU: per-block max ∂Δ/∂F, worst relative difference | f64h / f64hg LU | weak subspace (24 directions): sine of the largest principal angle |
|---|---|---|---|
| 40.11 | 4.7×10³ | 3.6×10³ | 1.0 (all variants) |
| 50.10 | 5.5×10⁵ | 5.3×10⁵ | 1.0 |
| 59.85 | 1.3×10⁹ | 3.2×10⁵ | 1.0 |

So with or without the horizon fix, the f64 LU's sensitivities and weakest directions are not usable at D ≥ 40: the
weakest singular values (10⁻¹⁸ of the largest) are below what an f64 factorization resolves. The horizon fix does not
change that; it repairs the step, whose right-hand side has little weight on those directions. What the variants rely
on instead: the loop's exact forward solves (the Δ correction from the refined step), and the exact last step when the
sensitivities are wanted. The deflated Newton still runs with the f64 LU's weak subspace; that it converges with
f64hg (and f64g at D ≤ 50) is an observation, not something the subspace guarantees.

## 5. The benchmark (`hp/bench_variants.py`)

**What one run is.** A state is perturbed, and the solve runs from there with one Jacobian variant to |F| < tol: plain
Newton (`newton_solve`) below D = 25, deflated Newton with w = 24 (`deflated_newton`, as `scan_hp.py`) from D = 25,
kfreeze 13 from D = 35 (and at the prepared D = 30 state). The Taylor plan is the reference state's, the same for every
variant; tol is 10 × the reference state's |F|, at least 10⁻²⁸. Recorded per (state, variant, repeat), one JSON line
as soon as it is done: converged, iterations, whole-solve seconds, the steps' seconds by part (residual, Jacobian, LU,
solves, products) summed and per step, the products, the start and final |F|, Δ and its difference from the dd run of
the same state and repeat, the load average before and after (os.getloadavg, else psutil's), the machine. A rerun
skips the lines present. At each state the variants run in turn, then the next repeat, so that load changes hit them
alike. `--report` prints the table of medians from a records file.

**States** (default `--states=3.7,8.25,16.76,20,30,40,45,50,55,60`): the saved scan state nearest each D
(hp/data/states; M = 48 at 8.25, 32 above), and two prepared ones in `hp/data/bench_variants/states/`
(`python3 bench_variants.py --prepare`): D = 3.7 at M = 80 (`lowDv_D3.7000_M80` confirmed by one double-double Newton
call, |F| 7.2×10⁻³⁰), and D = 30.07 frozen at kfreeze 13 and converged by the double-double deflated Newton to
9.2×10⁻²⁸ (the saved up6 state is a junk-accepted point, §7).

**The perturbation, and how it was chosen.** A random relative perturbation (every entry × (1 + ε ξ), as in
`mixed_largeD.py`) is no use at large D: the horizon block amplifies it. At D = 60, ε = 10⁻¹⁴ already gives
|F| = 2.4×10⁻⁵, plain Newton's first step is |dz| = 0.24 and it cannot decrease |F|; the double-double deflated Newton
did not converge in 10 steps from any ε in 10⁻⁸…10⁻¹⁴ at D = 60, and needed ε ≤ 10⁻¹⁴ at 55 and 10⁻¹³ at 50
(`data/bench_variants/calibrate_random.jsonl`). The bench therefore perturbs toward the neighbouring saved solution,
z + ε (z_nb − z), z_nb the saved state with the same nodes nearest above in D (below at 60), resampled to the state's
M (D = 3.7: lowDv_D4.0000_M64 to M = 80) and frozen like it: a smooth perturbation, like a continuation predictor's
error. `--calibrate` tries ε = 10⁻¹, 10⁻², … with the double-double variant (at most 10 Newton steps; a start with
|F| > 10⁻⁴ is not tried, since none converged), keeps the first ε that converges and uses a tenth of it
(`data/bench_variants/eps.json`, every try in `calibrate.jsonl`):

| state | largest ε that converged (start \|F\|, steps) | bench ε | neighbour |
|---|---|---|---|
| D = 3.7, M = 80 | | | lowDv_D4.0000_M64 |
| 8.25 | 10⁻⁴ (1.2×10⁻⁵, 5) | 10⁻⁵ | up_D9.0000 |
| 16.76 | 10⁻⁴ (8.0×10⁻⁶, 7) | 10⁻⁵ | up2_D17.0180 |
| 20.04 | 10⁻⁷ (5.5×10⁻⁸, 6) | 10⁻⁸ | up2_D21.0297 |
| 30.07 | 10⁻⁶ (6.1×10⁻⁶, 9) | 10⁻⁷ | up6_D31.0340 |
| 40.02 | 10⁻⁸ (1.0×10⁻⁶, 9) | 10⁻⁹ | up7_D41.0890 |
| 45.05 | 10⁻⁸ (3.9×10⁻⁶, 9) | 10⁻⁹ | up8_D46.0011 |
| 50.01 | 10⁻⁹ (1.8×10⁻⁶, 9) | 10⁻¹⁰ | up8_D51.0270 |
| 55.04 | 10⁻¹⁰ (8.1×10⁻⁷, 7) | 10⁻¹¹ | up11_D56.0117 |
| 60.00 | 10⁻¹⁰ (4.0×10⁻⁶, 9) | 10⁻¹¹ | up12_D59.0250 |

The basin is a start |F| of about 10⁻⁶ to 10⁻⁵ wherever the deflated Newton runs, and only 5.5×10⁻⁸ for the plain
Newton at D = 20. In z it is tiny at large D: a step of 10⁻¹⁰ toward a neighbour one unit of D away.

**The reference run here** (`python3 bench_variants.py --repeat=1 --threads=2 --out=data/bench_variants/bench_cloud.jsonl`,
log `bench_cloud.log`; 3 177 s in all at a load average of 9–13; the correction loop then had at most 12 products per
step). Whole-solve seconds, the speedup over dd at the same state, and the Newton calls (the last one confirms
|F| < tol):

| state (kc used / the state's) | dd | f64 | f64h | f64g | f64hg |
|---|---|---|---|---|---|
| D = 3.7, M = 80 (62 / 100) | 937 s, 4 | 138 s (6.8×), 5 | 515 s (1.8×), 5 | 123 s (7.7×), 4 | 414 s (2.3×), 4 |
| 8.25, M = 48 (47 / 60) | 68.2 s, 4 | 15.7 s (4.3×), 5 | 19.7 s (3.5×), 4 | 14.4 s (4.7×), 4 | 21.3 s (3.2×), 4 |
| 16.76 (77 / 80) | 21.7 s, 5 | 4.4 s (4.9×), 7 | 8.2 s (2.7×), 6 | 4.5 s (4.8×), 5 | 7.8 s (2.8×), 5 |
| 20.04 (76 / 80) | 16.2 s, 4 | 4.2 s (3.8×), 7 | 8.3 s (2.0×), 6 | 3.9 s (4.2×), 4 | 7.0 s (2.3×), 4 |
| 30.07, kfreeze 13 (217 / 414) | 38.9 s, 7 | 10.3 s (3.8×), 17 | 21.4 s (1.8×), 15 | 8.6 s (4.5×), 7 | 12.9 s (3.0×), 7 |
| 40.02 (174 / 420) | 24.2 s, 6 | fails | 22.3 s (1.1×), 8 | 20.3 s (1.2×), 7 | 17.5 s (1.4×), 6 |
| 45.05 (158 / 420) | 33.9 s, 9 | fails | 30.0 s (1.1×), 13 | 20.6 s (1.6×), 8 | 27.6 s (1.2×), 9 |
| 50.01 (149 / 420) | 26.5 s, 7 | fails | fails (25) | 17.3 s (1.5×), 7 | 27.2 s (1.0×), 9 |
| 55.04 (141 / 420) | 26.5 s, 7 | fails | 22.2 s (1.2×), 9 | 17.0 s (1.6×), 8 | 19.0 s (1.4×), 7 |
| 60.00 (134 / 420) | 22.5 s, 6 | fails | fails (25) | fails (25) | 27.3 s (0.8×), 9 |

With at most 30 products per step (now the default, `driver.GMRES`; `--max_products=30`, `bench_cloud_mp30.jsonl`):

| state | dd | f64g | f64hg |
|---|---|---|---|
| 40.02 | 27.2 s, 6 | 11.0 s (2.5×), 6; 96 products | 12.3 s (2.2×), 6; 70 |
| 45.05 | 36.5 s, 9 | 15.3 s (2.4×), 9; 168 | 19.6 s (1.9×), 9; 127 |
| 50.01 | 28.2 s, 7 | 16.7 s (1.7×), 8; 169 | 20.8 s (1.4×), 8; 124 |
| 55.04 | 25.5 s, 7 | 16.3 s (1.6×), 8; 200 | 16.1 s (1.6×), 7; 118 |
| 60.00 | 22.9 s, 6 | fails (|F| 3.5×10⁻⁹) | 16.2 s (1.4×), 7; 129 |

Every converged run's Δ agrees with the dd run's to the level of the dd run's own linearised Δ correction
(dDelta_est) or, below D = 25 where the plain Newton records none, well inside what the tolerance fixes
(|∂Δ/∂F| tol, 10⁻²⁶–2×10⁻²³): |Δ − Δ_dd| ≤ 6.4×10⁻³⁰ at D = 3.7 (2×10⁻³⁰ relative), ≤ 4.6×10⁻²⁹ at 8.25,
≤ 5.7×10⁻²⁸ at 16.76, ≤ 3.2×10⁻²⁷ at 20, ≤ 1.3×10⁻²⁴ at 30, and 8×10⁻²³ – 1.8×10⁻²¹ (40), 7×10⁻²² – 1.6×10⁻²⁰ (45),
3×10⁻²⁰ – 1.3×10⁻¹⁹ (50), 3×10⁻²⁰ – 5.9×10⁻¹⁹ (55), 2.0×10⁻¹⁸ – 4.8×10⁻¹⁸ (60), where the dd runs' dDelta_est is
4.7×10⁻²², −7.9×10⁻²¹, −1.0×10⁻¹⁹, −9.5×10⁻¹⁹ and −3.8×10⁻¹⁸.

What the table says:

- **Below D ≈ 35 the plain f64 Jacobian and the loop on it (f64g) are the fast variants**, 4–8× faster than dd; f64g
  needs as few or fewer Newton calls than f64. The horizon fix costs more than it saves there (the horizon block is
  a large part of the double-double work: 161 of 4 481 columns at M = 80 but each a 250-order series), and is not
  needed.
- **From D = 40 the plain f64 Jacobian fails** (it stalls at |F| = 10⁻¹⁷–10⁻¹⁹), as MIXED_PRECISION §8 predicted.
- **At D = 40–60 the gain over the double-double Newton is small, 1.4–2.5×,** once the double-double Jacobian uses a
  sensible centre-series order (§7.2; with the scan's kc = 420 the dd runs take 3 times longer and so would every
  exact step). The reason is the end of each solve: near the floor the residual lies in the weakest directions, which
  an f64 LU does not resolve, so the loop needs 15–30 products per step there or the exact finish takes over
  (`dd_finish`; with 12 products at most it took over in most large-D runs, whose Jacobian time is then mostly exact
  steps).
- **The robust variant at large D is f64hg**: it converged at every state. f64g fails at D = 60 and f64h at 50 and 60,
  in the deflated Newton (§4).

**Running it on the local machine** (Windows, i7-1185G7, 4 cores; from a checkout of `wip/largeD-mixed`):

```
cd hp
bash build.sh                          (or, in PowerShell: $env:RUSTFLAGS="-C target-cpu=native";
                                        maturin build --release --out dist;
                                        pip install --force-reinstall --no-deps (Get-Item dist\hpkernel-0.1.0-*.whl))
python bench_variants.py --repeat=3 --threads=4
python bench_variants.py --report --out=data\bench_variants\bench_<host>.jsonl
```

The records go to `data\bench_variants\bench_<host>.jsonl` (`--out` to change it); an interrupted run resumes where it
stopped when started again with the same arguments. The script lowers its own priority (`os.nice` on Linux; on
Windows through psutil, if installed, to below normal; `--no-nice` keeps it) and records the load average (psutil's
on Windows) before and after every run. `--states=50,55,60 --variants=dd,f64g,f64hg` restricts a run, `--dd_last`
adds the exact last step to the mixed variants, `--kc=meta` uses the states' own centre-series order (the scan's).
Nothing else is needed: the prepared states and `eps.json` are in the repository.

Expected run time: the reference run took 53 minutes for one repeat on two cloud cores at a load average of 9–13, two
thirds of it the D = 3.7, M = 80 state (its double-double solve alone 15.6 min). Idle, that machine would take about a
third of the time. On the laptop with `--threads=4`, plan for 20–60 minutes per repeat depending on what else runs,
i.e. 1–3 hours for `--repeat=3`; `--states=8.25,16.76,20,30,40,45,50,55,60` leaves out the M = 80 state and takes
about a third.

## 6. The default paths are unchanged

Everything new is opt-in: `horizon_dd=0`, `Solver.jvp` and `step_timing` are new, `jac="dd"` (the default of
`newton_solve`, `deflated_newton` and `scan_hp.py --jac`) runs the loops as before, and `jac="f64"` calls
`newton_mixed(**mixed)` as before. Checked:

- `python3 mixed_check.py ref --save=REF.npz` with a private build of main (e6c908a), then `--compare=REF.npz` with
  this branch's build: 34 arrays, all bit-identical (`data/mixed_precision/largeD_mixed/ref_save_main_build.log`,
  `ref_compare.log`). `ref` now also covers the mixed defaults (`jacobian_f64`, `newton_mixed`) at D = 3.7, M = 48, and
  residual, Jacobian, Newton step, `lu_solve_t`, `jacobian_f64` and `newton_mixed` at the D = 50.01 scan state with
  kfreeze 13; a reference file from before reports the arrays it lacks as not compared.
- `cargo test --release --lib`: 13 passed, 7 ignored (benches and diagnostics), among them two new tests
  (`hp/src/jvp_tests.rs`, on the synthetic M = 8 context of `qd_residual.rs`): `jvp_matches_dd_jacobian` (3.8×10⁻³⁰;
  with the Δ tangent × (1 + 10⁻²⁰) 3.7×10⁻²⁴, which must fail) and `hdd_block_rows_are_dd` (with `HDD_BLOCK` the
  horizon rows are bitwise the double-double Jacobian's hi parts and every other row bitwise the default f64 Jacobian's;
  with `HDD_LASTSEG` the horizon rows are not, the last segment's are) (`cargo_test.log`,
  `cargo_test_jvp_nocapture.log`).

## 7. Other findings

1. **The D = 30.07 state's settings are the scan's; the state is a junk-accepted point.** MIXED_PRECISION §8 notes
   that `mixed_largeD.py`'s settings leave |F| = 3×10⁻¹² there and concludes that the scan used something else. With
   those settings (nodes 0.3–0.7, kt 36, kc 414, ks 100, no kfreeze, plan(z, 10⁻³², 0.05)) the residual is
   3.3277×10⁻¹², and `hp/data/delta_dd.csv` records 3.328×10⁻¹² for that point (run up6). The same settings reproduce
   the recorded |F| at D = 29.05 (6.283×10⁻¹⁶), 25.08 (1.217×10⁻²⁸) and 40.02 (1.964×10⁻²⁶, kfreeze 13). Run up6
   (D = 26.9–35.2) ran with `--junk_only` and accepted points whose linearised Δ correction was below 10⁻¹⁴ at |F| up
   to 1.7×10⁻¹¹ (NUMERICS_REVIEW §1k). So the settings are right, and the footnote's inference is not: the state is
   not converged in its junk directions. The bench uses it frozen at kfreeze 13 and converged (§5).
2. **The scan's kc = 420 at large D costs a factor 3 in the double-double Jacobian for nothing.** At the D = 50 state
   the centre series' terms |c_k| x₀^k fall to 7×10⁻¹³⁴ by order 420 (7×10⁻⁵² by 160). With kc = 160 the residual and
   the double-double Jacobian are bitwise the same as with 420, and the Jacobian takes 5.8 s instead of 18.6 s, the
   residual 0.042 s instead of 0.126 s (same run, load average 13). The order at which the terms stay below 10⁻⁴⁵,
   plus 10, is 134 (D = 60), 149 (50), 174 (40) and 217 (30.07), and the residual is bitwise unchanged at each
   (`bench_variants.py --kc=auto`). The scan raised kc by 1.25 whenever its tail check saw a tail above 10⁻³³ (it
   reached 414 by D = 27, in the unfrozen junk-accepted run); it never lowers it. Lowering it is the cheapest speedup
   of the double-double path at large D and changes no number; proposed for the coordinator. It also matters for the
   comparisons here: the double-double times at large D in §4 (as the scan ran, kc = 420) are about 3 times what they
   need to be, while the f64 Jacobian stops its series by itself (`Tail`).
3. **Rounding amplification reaches the double-double Jacobian.** The dual pass is not exactly linear in its tangent
   at large D (§2): J(v₁ + v₂) − J v₁ − J v₂ is 10⁻²⁸ (D = 20) and 1.3×10⁻²⁶ (D = 50) of (|J| |v|), and a tangent on two
   columns differs from the sum of the two columns by up to 7×10⁻²³ of their size at D = 50. So the double-double
   Jacobian's columns are accurate to about 10⁻²³–10⁻²⁶ relative there, not 10⁻³². This is harmless for Newton, whose
   steps need far less, but it bounds any check of a double-double derivative at large D.
4. **An f64 LU's ∂Δ/∂F is wrong by orders of magnitude at D ≈ 60.** The scan's per-point max |∂Δ/∂F| from the
   double-double LU is 8.1–8.2×10¹³ at D = 59.83–59.85; from the f64 LU of the horizon-fixed Jacobian the same runs
   recorded 2.8×10⁷–1.0×10⁸. The weakest singular value there (about 2×10⁻¹⁸ of the largest, MIXED_PRECISION §8) is
   below what an f64 factorization resolves (2⁻⁵³ of the largest), so the transposed solve misses the direction that
   carries the sensitivity. The forward steps are still accurate where the residual has little weight on that
   direction (§1, §3), but the sensitivities, the Δ correction from the LU, and the weakest directions of the deflated
   Newton are not. See §4 for what the solver does about it.
5. **The plain Newton's basin at D = 20 is a start |F| of about 5×10⁻⁸** (from the calibration: 5.5×10⁻⁸ converged in
   6 steps, 5.5×10⁻⁷ failed its first line search), 100 times smaller than the deflated Newton's at D = 30–60.
6. **`scan_hp.py --predictor hermite` from a single state failed at D = 50**: the Euler step from the D = 50.01 state
   with the tangent of `driver.tangent` gave a non-finite Jacobian ("singular or non-finite pivot at column 0 (NaN)")
   at D = 50.0175 (`kfreeze 13`). Not investigated; the scan runs with the Lagrange predictor.

## 8. The M = 192 stall at D = 3.7 (for notes/HIGH_M.md §1.4)

The question from the coordinator: does f64 + the correction loop with exact products get the M = 192 lift at D = 3.7
to converge from the state where the plain f64-Jacobian Newton stalled
(`hp/data/lowD_states/highm_p1_D3.7000_M192_stalled.npz`, |F| = 3.75×10⁻¹⁰, its own 511-step plan, 13 nodes,
L = 10 753)? `python3 highm_gmres_check.py` (`newton_solve(jac="f64g")`, block storage, step_tol 10⁻¹⁰, at most 12
products; one Newton call at a time; log and records in `data/mixed_precision/largeD_mixed/M192_f64g.log`, `.json`):

| step | \|F\| before | loop: products, change of the step after each | line search | \|dz\| | ΔΔ |
|---|---|---|---|---|---|
| 0 | 3.75×10⁻¹⁰ | 12: 0.38, 0.040, 0.020, 1.3e-3, 4.8e-4, 4.1e-5, 4.7e-6, 1.8e-7, 4.1e-9, 1.1e-9, 5.1e-11 | λ = 1/2 | 7.9×10⁻⁸ | +2.6×10⁻¹³ |
| 1 | 1.87×10⁻¹⁰ | 12: 4.7, 0.028, 4.2e-3, 1.6e-3, 5.9e-5, 1.8e-5, 6.3e-7, 6.6e-8, 7.2e-9, 2.4e-10, 9.2e-12 | λ = 1/4 | 1.6×10⁻⁷ | +5.0×10⁻¹⁵ |
| 2 | 1.41×10⁻¹⁰ | 12: 0.32, 0.034, 4.5e-3, 1.5e-3, 1.2e-4, 1.4e-5, 4.0e-7, 6.6e-8, 1.7e-9, 1.9e-10, 5.1e-12 | λ = 1 | 1.3×10⁻⁷ | +3.3×10⁻¹² |
| 3 | 1.97×10⁻¹⁰ | 12: 0.27, 0.22, 0.096, 8.2e-3, 1.0e-3, 3.6e-5, 1.8e-6, 2.2e-7, 1.6e-8, 2.6e-10, 5.1e-12 | λ = 1/8 | 3.2×10⁻⁷ | −2.7×10⁻¹² |

(|F| is the largest entry; the line search works on the 2-norm, which fell at every step. The run was stopped after
four Newton calls, 2 h 19 min under load 10–13: the f64 Jacobian in block storage took 1 350–2 240 s per step, the 12
products 85–180 s.)

(The changes list starts at the second product; the first is 1 by definition.)

- The plain f64 step is far from the exact step at M = 192: the loop's first corrections are 0.38 and 4.7 of the
  step, so the high-M agent's diagnosis holds as far as it goes: the f64 Jacobian's rounding misplaces the step in a few
  directions, which the loop corrects (to 10⁻¹¹ in 12 products, 15 s each against 2 220 s for the f64 Jacobian in
  block storage under load 12).
- **Answer: no.** The exact steps do not converge either: the line search cuts them to 1/2, 1/4, 1 and 1/8, and |F|
  stays at 1.4–2.0×10⁻¹⁰ over four calls, while Δ moves by 10⁻¹² per step. The loop needed 12 products per step
  (it converged at the 12th each time, the change then 5×10⁻¹¹–9×10⁻¹²). At |F| ≈ 2×10⁻¹⁰ an exact Newton step of
  |dz| = 10⁻⁷–3×10⁻⁷ is outside the quadratic model's range, the same basin problem as the plain Newton's at large D
  (§5). So the stall at M = 192 is the weak directions' nonlinearity, and only in part the f64 Jacobian's rounding
  (which misplaces the step by 0.3–4.7 of its size, and which the loop removes). A deflated Newton (the step with the
  weakest directions removed) or a start closer to the solution is what it needs; with an f64 LU, though, the weakest
  directions it would remove are themselves not resolved (§7.4). Not tried here (a single measurement, as asked).

## 9. Merge with wip/high-m

This branch is wip/high-m (382d515) with main merged in, and wip/high-m's later commits merged when they appeared
(2b3b253, 985b137, 59eca1e; then 6ddbe6f, 45b8463, 052ed49, which changed no shared code). 2b3b253 added `jac="f64"` and `mixed` to `driver.deflated_newton`; this branch had added
`jac` (all five variants), `mixed` and `gmres` to the same function. The conflict was resolved by keeping their
signature and their `f64` path (the step from `s.newton_mixed(**mixed)`, the linearised Δ correction from the stored
LU's transpose solve, no extra step at convergence) and routing every variant through `newton_step`, which for `dd`
and `f64` makes exactly their calls. The new variants differ in two documented ways: at convergence they take the
already-computed step if |F| stays below tol (as `newton_solve` does for `f64`), and with the correction loop the Δ
correction is the refined step's Δ component. No Rust conflicted (wip/high-m changed no Rust after 382d515).

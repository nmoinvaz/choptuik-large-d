# The Taylor series in octo-double (2026-11-06)

The large-D chain builds its Taylor series in D with the quad-double jet (`anm_run.py --jet_q 1`,
notes/QUAD_DOUBLE.md 13.20). That series errs by about the quad-double unit times G², where G ≈ 0.7^−(D−3) is the
mass mode's gain, and near D 170–185 this reaches the chain's tolerance. This note adds the same series one level up,
in octo-double (eight `f64` parts, about 126 digits), on multidouble 4.0.0's octo-double jet (`OJet<N>`), dual (`Ou`)
and LU (`ODLu`). It is not used by the chain until the quad-double series gives out.

## 1. What was added

- `hp/src/osolve.rs`, `qsolve.rs` one level up. `residual_parts_o` and `residual_o` are the residual with `OV` lanes,
  `residual_jet_o` the jet in the dimension with `OJet<N>` lanes (its D-dependent constants in `OSeries<N>`),
  `jacobian_o` the exact Jacobian with `Ou` lanes, and `newton_o` the step through `ODLu`. States come in eight parts.
- `hp/src/model.rs` `psolve`: two branches for the octo-double lanes, ahead of the existing ones, which no existing lane
  reaches. An `OJet` is solved order by order through multidouble's `solve_lanes_jet_refine_od`. `OV` and `Ou` go
  through `solve_lanes_refine_t`. Each solve is refined to octo-double in `OD_REFINE` = 4 steps (5 with a tangent),
  twice the quad-double count, since each step gains about cond·2⁻¹⁰⁴.
- `hp/src/python.rs`: `Solver.residual_o`, `residual_jet_o`, `jacobian_o`, `newton_o` (keeps the LU) and `olu_solve`,
  each octo-double vector a sequence of eight part arrays. They are dispatched for M = 32 only (`Ctx<4>`) and
  `residual_jet_o` for N = 8, 24 and 32, to keep the release build's memory down.
- `hp/anm_run.py --jet_o 1`: the series z_1..z_N from `residual_jet_o` through the octo-double LU at the base, whose
  four quad-double parts are promoted to eight. The held rows are added in octo-double (`od_add_dd`), and the
  prediction is summed by Horner's rule in octo-double (`od_axpy`) and rounded to quad-double for the quad-double
  corrector. It needs `--corrector qd`, and with `--reseed` also `--reseed_q 1`. It replaces `--jet_q`, `--pred_q` and
  `--fix1`, and `--probe` does not apply.
- `hp/ojet_check.py`: the octo-double series against the quad-double one on a stored state.

## 2. Checks and their criteria (fixed before the first run)

**Rust tests**, `cargo test --release --lib osolve_` on the synthetic context of `qd_residual::synth`. Each test's doc
states its criterion and controls:

| test | pass | controls, which must fail |
|---|---|---|
| `osolve_agrees_with_quad_double` | residual, 24-order jet and Jacobian within 1e-58 of qsolve's, at a state exact in quad-double | |
| `osolve_jacobian_matches_differences` | Jacobian within 1e-78 of central differences of `residual_parts_o` at h = 2⁻¹⁴⁰ | horizon solves unrefined; the quad-double Jacobian (each > 1e3 × the error) |
| `osolve_refinement_converged` | `OD_REFINE` steps give the 8-step Jacobian to 1e-118 | |
| `osolve_newton_converges` | planted root from 1e-6: below 1e-100 after 5 steps, below 1e-118 after 6 | |
| `osolve_resolves_sub_quad_double_changes` | a change of 2⁻³⁰⁰: F(z* + d) − F(z*) = J d to 1e-6, one Newton step leaves < 2⁻²⁰ of it | the state read in its quad-double part (`OD_MUT_STATE_QD`) sees nothing (> 0.5) |
| `osolve_jet_sums_to_the_residual` | the 24-order jet summed against `residual_parts_o` at n + h: truncation order in [23, 25], error < 1e-28 at 2⁻⁷, 2⁻⁸ | the dimension frozen (`JET_MUT_FREEZE_N`, > 1e3 ×) |
| `osolve_jet_resolves_sub_quad_double_changes` | a change of 2⁻³⁰⁰ in v changes F_1 by J·2⁻³⁰⁰w to 1e-6 | v read in quad-double (> 0.5); solves unrefined (> 1e3 ×) |

The jet's sum check is limited by `residual_parts_o`'s D-dependent constants, which are formed in double-double where
the jet forms them in octo-double, as in qsolve's check.

**`hp/ojet_check.py` at D 128.56** (raw-data repository `anm-mac/pred_q/A2/anm_pin3_D128.5559.npz`, N = 8). The
quad-double series' own error is estimated independently of the octo-double code, by the same series built with the
horizon series in its alpha form (`horizon_alpha 1`, notes/LARGE_D_MIXED.md 10). That form is the same model with other
rounding. Per order k, d_oq(k) is the octo-double series' difference from the quad-double one, and d_qa(k) that of the
two quad-double forms. Pass: d_oq(k) ≤ 30 d_qa(k) for every k, and the median of d_oq / d_qa lies in [1/30, 30].
Controls, each must fail: the double-double series in place of the octo-double one (d_dq(k) > 30 d_qa(k) for some
k), and the octo-double z_1 with the dimension frozen (> 1e3 d_qa(1)).

**At T113's highest state** (`T113-win/chain/anm_pin3_D146.5559.npz`), `ojet_check.py --check 0 --N 24` records d_oq(k)
and d_qa(k) by order, which says where the quad-double series loses digits. There is no criterion.

**Bit identity of the existing paths**, with this build: `highm_check.py ref --compare=<raw>/anm-mac/md31/ref_old.json`,
`scaled --compare=<raw>/anm-mac/md31/scaled_old.json`, `jets --compare=<raw>/anm-mac/jets_inputs/jets_inputs.json`, and
`jets` with `--mutate=kappa`. Expected exits 0, 0, 0 and 1.

## 3. Results

**Rust tests: 7 of 7 pass** (raw-data repository `od-mac/osolve_tests.log`, with qsolve's 8 beside them, all on the
working tree of 2dcd7a5):

| test | measured | control |
|---|---|---|
| agreement with qsolve | residual 2.0e-64, jet 6.7e-63, Jacobian 3.9e-63 | |
| Jacobian vs central differences | 1.1e-83 | unrefined 6.6e-4, quad-double Jacobian 2.3e-63 |
| refinement converged | 4.3e-129 | |
| planted Newton from 1e-6 | 6.4e-8, 1.2e-14, 3.5e-28, 2.6e-55, 1.3e-109, 3.3e-128 | |
| change of 2⁻³⁰⁰ in the state | J d to 2.4e-40, Newton leaves 5.1e-38 of it | quad-double reading: 0.64, 2.6e25 |
| jet summed at n + h | 6.0e-27, 3.5e-34, 4.2e-39 at 2⁻⁶, 2⁻⁷, 2⁻⁸ (order 24.0) | frozen dimension 1.4e-3 |
| change of 2⁻³⁰⁰ in the jet's v | J d to 3.0e-38 | quad-double reading 0.94, unrefined 6.8e-2 |

On the first run the state control of `osolve_resolves_sub_quad_double_changes` did not fire (2.0e-33 where it must
exceed 0.5). Its planted root was the synthetic state itself, two parts per entry, so the change of 2⁻³⁰⁰ landed in the
third part, which quad-double holds. The root is now that state moved by 2⁻⁶⁰ along a direction with all eight parts in
use, which puts the change below the fourth part; the criteria are unchanged. The jet's version already used such a
direction, and its control fired on the first run.

**The release build** (`od-mac/build.log`, hp/build.sh under `/usr/bin/time -l`, the Mac at load 5–8): 31 min 11 s,
peak resident memory 12.7 GB. Two builds back to back from empty target directories at load 12–15, with
multidouble's octo-double branch: the build of main 37 min 0 s and 13.5 GB, the same with `residual_jet_o::<24, 4>` and
`jacobian_o::<4>` instantiated 42 min 7 s and 12.0 GB, and a dylib 1.7% larger. The octo-double code adds about a sixth
to the build's time on the Mac, and its peak memory did not rise measurably there.

**Bit identity of the existing paths** (`od-mac/highm_*.log`, this build against the stored references): `ref` 168 of
168 outputs bit-identical (exit 0), `scaled` PASS with its control firing (0), `jets` 28 of 28 (0), and the control
`jets --mutate=kappa` 28 of 28 different (1). The whole hp test suite passes (19 tests, the 2 benchmarks ignored).

**`hp/ojet_check.py` at D 128.56: PASS** (`od-mac/ojet_check_D128.log`, N = 8, `--alpha_o 1`, 2 threads at load
30–70). G = 2.8×10¹⁹, so eps_qd G² = 9.6×10⁻²⁵.

| k | \|z_k\| | d_oq | d_qa | d_dq | d_oa |
|---|---|---|---|---|---|
| 1 | 9.78e-4 | 2.14e-18 | 2.14e-18 | 8.56e2 | 1.14e-87 |
| 2 | 6.08e-6 | 5.19e-16 | 5.19e-16 | 2.40e5 | 3.65e-85 |
| 3 | 4.21e-8 | 5.48e-14 | 5.49e-14 | 4.71e7 | 3.16e-83 |
| 4 | 3.15e-10 | 3.68e-12 | 3.68e-12 | 1.38e10 | 2.74e-81 |
| 5 | 7.10e-12 | 5.98e-11 | 5.99e-11 | 2.01e12 | 3.71e-80 |
| 6 | 2.42e-12 | 5.30e-11 | 5.30e-11 | 3.41e13 | 4.18e-80 |
| 7 | 1.02e-12 | 3.07e-11 | 3.08e-11 | 1.49e14 | 2.05e-80 |
| 8 | 3.78e-13 | 1.79e-11 | 1.79e-11 | 4.90e14 | 1.49e-80 |

Every criterion holds: d_oq(k) ≤ 30 d_qa(k), the median ratio is 1.00, the double-double control fails by up to 10²⁵,
the frozen-dimension control by 1.0 against 2×10⁻¹⁸, and the octo-double series' two forms agree to 10⁻⁸⁰–10⁻⁸⁷. Times:
the octo-double LU 414 s and its 8-order series 676 s, against 36 s and 54 s in quad-double.

d_oq equals d_qa to three digits at every order. So nearly all of the two quad-double forms' difference is the default
form's error, and the alpha form's quad-double series lies much closer to the octo-double one. The chain runs the
default form (`horizon_alpha 0`, in `pred_q/A2` and T113). Its quad-double series is wrong by 2×10⁻¹⁸ relative at order
1 and by 6×10⁻¹¹ at order 5, 10⁶ to 10¹⁴ times eps_qd G². The horizon series' cancellation in a² − 1 and a − a³,
which the alpha form removes (notes/LARGE_D_MIXED.md 10), costs digits ahead of G². The model eps_qd G² therefore
understates the default form's error, and the octo-double series is needed sooner in that form. Version 2 of the
script also records d_oq1, the octo-double series against the alpha form's quad-double one.

**The `--jet_o` pilot** (`od-mac/pilot*_jet_*.log`), one step from T113's start (`pred_q/A2/anm_pin3_D128.5559.npz`)
with T113's options and `--jet_o 1` in place of `--jet_q 1 --pred_q 1`, N = 8 and 2 threads to keep within the thread
budget beside the T113 report. Pass, fixed before the run: the first attempt converges and the state is saved.

The first pilot failed. With `--eps 1e-14` the step was a_max = 0.2246, the prediction had |F| 2.23×10⁻³, and the
quad-double corrector could not start from it. The quad-double path (`--jet_q 1 --pred_q 1`) at the same options fails
with the same digits, 2.23×10⁻³, and at N = 24 with a = 0.225 it predicts 2.83×10⁻¹² and converges (`pilot_jet_q24`).
The series to N = 8 and to N = 12 agree bit for bit through order 8 (a separate script). So the failure is the
truncation at N = 8, not the octo-double code: the Jacobian's entries reach 1.9×10¹⁷ (the LU's pivot range), so a
prediction must be right to about 10⁻²⁹ for |F| near 10⁻¹², and |z_9| a⁹ ≈ 1.2×10⁻¹³ × 1.5×10⁻⁶ ≈ 2×10⁻¹⁹ there.
The chain's `--eps 1e-14` is set for N = 24, where the truncation at a = 1 is 6×10⁻²⁵. The second pilot keeps N = 8
and caps the step at 0.0125, where |z_9| a⁹ ≈ 10⁻³⁰.

The second pilot passes (`od-mac/pilot2_jet_o.log`). At a = 0.0125 the octo-double prediction has quad-double |F|
3.00×10⁻¹⁵, and the quad-double corrector converges in 4 steps to 1.2×10⁻⁴⁴ on its first attempt; the state
`pilot2_jet_o/anm_pin3_D128.5684.npz` is saved. Its series took 531 s on 2 threads. The quad-double path at the same
options (`pilot2_jet_q.log`, 53 s for its series) predicts the same |F|, 3.00×10⁻¹⁵, and converges to the same Δ to 17
digits. At D 128.56 and this step the quad-double series' rounding is not what limits the prediction.

**`hp/ojet_check.py --check 0` at T113's last state, D 148.74** (`od-mac/ojet_T113_D148.log`, N = 24, 2 threads). G =
3.8×10²², so eps_qd G² = 1.7×10⁻¹⁸. Relative errors of the quad-double series by order, taking the octo-double one as
the reference (d_oq the default form, d_oq1 the alpha form, d_qa the two forms' difference):

| k | \|z_k\| | d_oq | d_oq1 | d_qa |
|---|---|---|---|---|
| 1 | 7.75e-4 | 6.67e-13 | 3.00e-13 | 6.65e-13 |
| 2 | 4.13e-6 | 1.82e-10 | 3.11e-11 | 1.86e-10 |
| 3 | 2.45e-8 | 2.24e-8 | 1.13e-8 | 2.24e-8 |
| 4 | 1.58e-10 | 1.68e-6 | 2.75e-7 | 1.71e-6 |
| 5 | 8.02e-12 | 1.19e-5 | 6.80e-6 | 1.20e-5 |
| 6 | 3.29e-12 | 8.42e-6 | 1.29e-6 | 8.59e-6 |
| 8 | 4.75e-13 | 2.89e-6 | 4.04e-7 | 2.95e-6 |
| 10 | 4.29e-14 | 9.16e-7 | 1.14e-7 | 9.37e-7 |
| 12 | 2.65e-15 | 2.77e-7 | 2.93e-8 | 2.83e-7 |
| 16 | 4.01e-18 | 2.29e-8 | 2.55e-9 | 2.38e-8 |
| 20 | 2.26e-21 | 2.76e-9 | 3.05e-9 | 4.41e-9 |
| 24 | 5.91e-25 | 9.40e-9 | 1.75e-8 | 1.64e-8 |

The full table, all 24 orders and the double-double series' d_dq (above 10³ everywhere), is in the log. The quad-double
series keeps about 12 digits at order 1 and 5–6 at orders 4–9, the worst near order 5 (1.2×10⁻⁵). From D 128.56 to
148.74 its order-1 error grew 3.1×10⁵ times and its order-5 error 2×10⁵ times, where G² grew 1.8×10⁶, about G^1.7. The
alpha form is better by a factor 2–10 here, not by the margin D 128.56 suggested, so switching forms does not
postpone the need for octo-double by much. T113's last steps were 0.031 long (raw-data repository
`T113-win/README.md`); whether the series' lost digits set that step is not tested here. The octo-double series took
3108 s on 2 threads against 477 s for the quad-double one, and its LU 254 s against 31 s.

## 4. Running the chain with octo-double jets

With T113's options, `--jet_o 1` in place of `--jet_q 1 --pred_q 1` (M = 32; N up to 31):

    cd hp && python3 -u anm_run.py RAWDIR RAWDIR/TASK/chain --state STATE.npz --kfreeze 3 --N 24 --eps 1e-14
        --h_max 1 --steps 300 --D_max 200 --threads 4 --corrector qd --tol_qd 1e-28 --qd_it 12 --trend T75 --jet_o 1

Each base then costs about 7 times the quad-double series' time on this Mac, about 1 h on 2 threads at N = 24.

## 5. The corrector in octo-double (2026-11-06)

The chain's corrector was quad-double (`anm_run.py --corrector qd`: `newton_qd` on `pinning.PinnedQ`, inside
`reseed.ElimQ` under `--reseed 2`). It stopped T129's chain2 at D 193.77, where each of four halvings ended in "no trial
step accepted" at max |F| 1.0–1.1×10⁻²⁸ against `--tol_qd` 1e-28 (raw-data repository `T129-win2/chain2.log`). This
section finds why, and adds the same corrector one level up, `--corrector od`.

### 5.1 Why the quad-double Newton stops

Every test starts from stored converged chain states (`T129-win2/chain/` and `chain2/`), on the pinned system of
`reseed.pinned` (the state's own held values, the segments planned at the state). The scripts and logs are in the
raw-data repository's `od-corrector-mac/` (bcad91a).

**The residual is accurate.** `step0_res.py RAW PLAN_TOL STATES` evaluates the pinned residual in quad-double and in
octo-double at the state's four parts (`step0_res.log`). Their difference is the quad-double residual's own error.

| D | pinned max \|F\| | \|F_qd − F_od\| | with four refinement steps (mutate 8) |
|---|---|---|---|
| 180.52 | 2.83e-30 | 1.74e-36 | 1.74e-36 |
| 185.52 | 2.30e-29 | 1.20e-35 | 1.20e-35 |
| 188.02 | 9.75e-29 | 1.34e-35 | 1.34e-35 |
| 193.77 | 1.00e-28 | 9.69e-35 | 9.69e-35 |

The plan at 1e-40 (27–30 Taylor steps) and 1e-48 (44–48) gives the same values to three digits as the chain's 1e-32
(17–18). The octo-double residual with eight refinement steps differs from the default's by 1e-107 to 4e-105. So the
residual at these states is right to 10⁻³⁵ in quad-double, far below the floor, and the plan does not move it.

**A tighter plan does not lower the floor.** `step0_floor.py RAW PLAN_TOL STATES` is `qd_floor.py` (plain
`newton_qd`, tol 1e-45, it_max 5) with the plan's tolerance as an argument (`step0.log`; the 1e-32 column is
`qd_floor.log`, the same procedure). The entries are the start and the last max |F|. No step lowered max |F|
below the start in any run at D ≥ 185.

| D | plan 1e-32 | plan 1e-40 | plan 1e-48 |
|---|---|---|---|
| 175.52 | 6.57e-32 → 1.71e-37 | | |
| 180.52 | 2.83e-30 → 1.59e-36 | | |
| 185.52 | 2.30e-29 → 1.45e-22 | 2.30e-29 → 1.45e-24 | 2.30e-29 → 1.39e-24 |
| 188.02 | 9.75e-29 → 7.93e-16 | 9.75e-29 → 8.56e-16 | |
| 193.77 | 1.00e-28 → 3.07e-16 | 1.00e-28 → 1.92e-17 (stalled) | |

**The quad-double step is wrong in the weak direction.** `step0_step.py RAW 1e-32 anm_pin3_D188.0246.npz` takes one
Newton step from the state in quad-double (`PinnedQ.newton_q`), in quad-double with the horizon solves refined four
steps (mutate 8), and in octo-double (`PinnedO.newton_o`), and evaluates the octo-double residual after each full step:

| step | \|dz\| (all in Delta) | LU pivots | max \|F\| after the full step | \|dz − dz_od\| / \|dz_od\| |
|---|---|---|---|---|
| quad-double | 6.59e-20 | 6.4e-12 .. 1.8e26 | 5.14e-14 | 0.45 |
| quad-double, four refinements | 1.35e-19 | 2.4e-11 .. 1.8e26 | 3.76e-14 | 0.14 |
| octo-double | 1.19e-19 | 1.5e-10 .. 1.8e26 | 4.37e-14 | |

A full step raises max |F| by fourteen orders in all three, so that rise is the step's quadratic term (in the horizon
rows, below), not rounding; newton_qd's dz test is made for it. What differs is the step itself: the
quad-double LU's smallest pivot is 24 times the octo-double one's too small, and its step in Delta is wrong by 45%.
Four refinement steps of the horizon solves (QD_REFINE is 2) recover part of it. The quad-double LU's smallest pivot
falls from 3.7e-9 at D 175.52 to 1.0e-13 at 193.77 (`qd_floor.log`), 3.7×10⁴ times, while its largest grows
600 times. The octo-double LU's smallest pivot is 1.3e-9 at 180.52 (quad-double 1.2e-9), 1.5e-10 at 188.02 (6.4e-12)
and 2.5e-11 at 193.77 (1.0e-13; `check_a.log`, `step0.log`, `d_step.log`), so the two part ways between 180 and 188.
A reading consistent with this, not tested further, is that the quad-double Jacobian's error, of order eps_qd G²
(5×10⁻¹² at 188), approaches the weak direction's pivot (1.5×10⁻¹⁰ in octo-double), so the Newton iteration loses its
contraction in that direction. Neither the plan's truncation nor the held offset (formed exactly in
double-double, its rounding in the measured |F| about 10⁻³⁵) is the cause. In octo-double eps_od G² is about 10⁻⁷⁵.

**The octo-double step needs damping there too.** Plain `newton_od` from the same state (`step0_od.py`, stopped after
its first step, `step0_od188.log`) rejected the full and the half step by its dz test and took lam 1/4: |dz| 1.2e-19
→ 8.1e-20, max |F| 9.75e-29 → 2.73e-15, in 782 s (three octo-double Newton steps on 3 threads). So the state at D 188,
which the quad-double corrector left 1.2×10⁻¹⁹ off in Delta, lies outside the full step's basin in the weak direction
even with an exact step. At 175.52 and 180.52 the converged floor runs moved Delta by 2.3×10⁻²⁴ and 8.5×10⁻²² in all.
After the full octo-double step the residual sits in the horizon block's rows alone, 6.4e-15 to 4.4e-14 in its p, m, a
and c rows and at most 2.0e-41 in every other block (`rows188.py`, `rows188.log`; the quad-double step's is alike). It
is the horizon series' quadratic term, which the projection of ElimQ and ElimO removes, so the octo-double base polish
runs inside ElimO too.

### 5.2 What was added

- `hp/octo_corrector.py`. `PinnedO` is PinnedQ's offset on the octo-double methods: `residual_o` returns F − c in eight
  parts (`od_add_dd`), and `newton_o` returns dz + J⁻¹c (`olu_solve`, added by `od_axpy`) with F − c formed from
  `residual_o`, since the kernel's F in `newton_o` is rounded to double-double, 2⁻¹⁰⁶|c| ≈ 10⁻³⁵ on the held rows.
  `newton_od` is newton_qd's mode 'full' on it (the dz test, eight halvings), returning (z, hist, converged) the same
  way. `ElimO` is ElimQ for eight-part iterates: above the switch (pinned quad-double max |F| > 1e-15 at the point's
  quad-double rounding) the point is projected by `Reseeder.project` and its horizon data put back with parts 4–7 zero
  (`with_horizon`); below it plain octo-double Newton runs. A start whose projection fails above `start_F` raises
  `reseed.StartNotProjected`, as ElimQ. `save_o` writes save_q's keys and z4..z7, so `load_q` reads the first four
  parts; `load_o` reads eight (four-part files get zeros), `has_o` tells them apart. The helpers `o8`, `o8_of_q` and
  `q4_of_o` moved here from anm_run.py, which imports them.
- `hp/anm_run.py --corrector od --tol_od 1e-40` (`--qd_it` steps). Under `--jet_o 1` the start is the octo-double
  prediction `horner_o(Zo, a)` in all eight parts, its horizon data replaced by the re-seed's when that succeeds. The
  base polish starts from all eight stored parts, inside ElimO at the base's D under `--reseed 2` (the quad-double
  polish has no projection). A converged state is saved by `save_o` (meta `parts` 8), and under
  `--jet_o 1` the next base's series is taken at the eight parts (also on a restart from an eight-part file). The qd,
  dd and other paths are unchanged; `--accept_stall_qd` does not apply to od. M = 32 only, and `--reseed` needs
  `--reseed_q 1`.
- The checks, `octo_corrector.py planted` and `wall` (their docstrings).

Step 0 found no dependence on the segment plan's tolerance, so the corrector has no `--plan_tol` of its own (anm_run's
`--plan_tol` still plans the first base only, and each prediction is replanned at 1e-32).

### 5.3 Checks and their criteria (fixed before the runs)

- (a) Planted truth, `octo_corrector.py planted RAW T129-win2/chain/anm_pin3_D180.5246.npz`. The start is the
  converged state with its centre data and its nodes but the last moved by 10⁻¹² relative (random signs, sizes 0.5–1,
  seed 1), on the state's own pinned system. Pass: the octo-double corrector (ElimO, newton_od, tol 1e-40) converges
  with the octo-double pinned max |F| below 1e-40, and its Delta agrees with the quad-double corrector's from the same
  start (ElimQ, newton_qd to tol 1e-45 in 12 steps) to 10⁻³⁰. The task as set moved every node. A last node moved by
  10⁻¹² leaves the regular family by about G·10⁻¹² at the horizon, and the start's projection then failed (`probe_a.py`,
  `probe_a_last.log`: max |F| 3.62e-12, projection 'failed' in 249 s), so the last node stays at the truth and the
  projection succeeds (`probe_a.log`: 3.39e-12, 'u0q' in 88 s); `--last 1` moves it too.
- (b) The wall, `octo_corrector.py wall RAW T129-win2/chain2/pred_pin3_D194.2746.npz`: the stored prediction where
  chain2's quad-double corrector stalled at 1.12e-28 (Delta 1.6965746586979402 in `chain2/anm_run.jsonl`), re-seeded
  with the quad-double polish, then the octo-double corrector. Pass: its octo-double pinned max |F| below 1e-36.
- (c) The control, `planted ... --qd 0 --mutate_offset qd --Delta_ref <(a)'s quad-double Delta>`: PinnedO's J⁻¹c taken
  through the quad-double LU at the point's quad-double rounding. It must fail (a).
- (d) One chain step, `anm_run.py --corrector od --jet_o 1 --reseed 2 --reseed_q 1 --h_max 0.5 --steps 1` from
  `T129-win2/chain2/anm_pin3_D193.7746.npz` with T129's other options (N 24, eps 1e-14, trend T75), then a run with
  `--steps 0` from the eight-part state it saved. Pass: the step converges, and the restart's base polish converges.
- (e) Unchanged paths: one `--corrector qd` step (`--jet_o 1 --reseed 2 --reseed_q 1`, N 8, h_max 0.0625) from
  `T129-win2/chain/anm_pin3_D180.5246.npz` by the code of fc636e7 and by this change. Pass: the logs equal line by line
  with the times removed, and the records' Delta, |F| and step counts equal (`compare_e.py`).
- (f) `python3 check_numbers.py --selftest` passes.

### 5.4 Results

**(a) Planted truth: PASS.** The octo-double run (`check_a.log`) and the quad-double one (`--od 0 --Delta_ref`,
`check_a_qd.log`; the first run's quad-double half stopped on a bug since fixed) from the same start, max |F| 3.39e-12:

| step | octo-double max \|F\| | \|dz\| | quad-double max \|F\| | \|dz\| |
|---|---|---|---|---|
| 0 | 3.39e-12 → 5.92e-16 | 1.0e-12 | 3.39e-12 → 5.92e-16 | 1.0e-12 |
| 1 | 5.92e-16 → 1.63e-31 | 7.7e-16 | 5.92e-16 → 1.30e-26 | 7.7e-16 |
| 2 | 1.63e-31 → 7.61e-22 | 6.8e-24 | 1.30e-26 → 5.46e-30 | 3.2e-19 |
| 3 | 7.61e-22 → 6.86e-42 | 6.4e-34 | 5.46e-30 → 1.32e-20 | 1.0e-21 |
| 4 | 6.86e-42, converged | 6.4e-34 | 1.32e-20 → 5.30e-27 | 3.4e-26 |
| 5 | | | 5.30e-27 → 1.59e-33 | 2.2e-28 |
| 6–11 | | | 1.59e-33 → 1.69e-36, then 1.4e-36 to 5.8e-37, stopped after 12 | 4.7e-32 to 1.5e-42 |

The octo-double corrector converged in 5 steps (1559 s on 3 threads, 3 projections, all u0q) to 6.86e-42, below 1e-40.
Its Delta is 1.700531269818162204386381037834122353007, the quad-double one's 1.700531269818162204386381037834122235155,
1.18×10⁻³⁴ apart (criterion 10⁻³⁰). Step 2's rise to 7.6e-22 is the horizon rows' quadratic term again; the next step
removes it. The quad-double step already differs at D 180.5: its Delta step at step 0 is −2.93e-17 against −3.09e-17,
and from step 2 on it converges linearly to its floor near 1e-36. The two LUs' smallest pivots agree here (1.2e-9 and
1.3e-9). The octo-double Delta lies 8.5×10⁻²² from the stored state's, the move the quad-double floor run
(`qd_floor.log`) made from it.

**(b) The wall: PASS** (`check_b.log`). The re-seed of `pred_pin3_D194.2746.npz` succeeded in 83 s and left pinned
max |F| 5.16e-20 in octo-double. The octo-double corrector then converged in 3 steps, all full:

| step | max \|F\| | \|dz\| | dDelta | projections |
|---|---|---|---|---|
| 0 | 5.16e-20 → 1.34e-34 | 6.7e-19 → 9.7e-35 | +6.69e-19 | the trial projected (u0q) |
| 1 | 1.34e-34 → 1.84e-41 | 9.7e-35 → 4.2e-57 | −4.24e-35 | skipped |
| 2 | 1.84e-41, converged | | | |

629 s on 3 threads from the prediction, the re-seed included. The LU's pivots are 1.7e-11 .. 1.6e27. Delta is
1.69657465869794019449316000856225522831, and the quad-double stall's recorded Delta is 1.6965746586979402 (17 digits,
float64 of its first two parts, `chain2/anm_run.jsonl`), the same to those digits. The first step's trial carried the
horizon rows' quadratic term above ElimO's switch, and its projection removed it, which is the mechanism the plain
octo-double Newton at D 188 lacked.

**(c) The control fails (a), as it must** (`check_c.log`, exit 1). With J⁻¹c through the quad-double LU the first
step is |dz| 2.2e-6 instead of 1.0e-12. No trial of its nine was accepted, so the corrector stopped at the start's
3.39e-12, and its Delta lies 8.5×10⁻²² from (a)'s quad-double one, against 1e-40 and 10⁻³⁰. It took 4340 s, the
trial points' projections mostly failing (2 u0q, 3 ladder, 6 failed).

**(d) One chain step and a restart: PASS** (`run_d.sh`, `d_step.log`, `d_restart.log`). The base polish, inside ElimO
from chain2's four-part state at D 193.77, took three full steps, 1.00e-28 → 2.37e-34 → 8.93e-42, its first moving
Delta by −9.19e-19 (its trial projected once). The octo-double series at the polished base (2991 s on 3 threads) has
the record of chain2's last base to the printed digits: a_max 2.058, |z_N| 3.18e-25, ratios 246.4 220.9 106.5 3.4. The
step at a = 0.5 re-seeded in 84 s, and the corrector converged in three full steps, 1.36e-19 → 7.52e-35 → 3.20e-51,
in 518 s; the state `anm_pin3_D194.2746.npz` holds eight parts (`load_q` reads its first four, `has_o` is true). Its
Delta, 1.69657465869794022 − 2.4348e-17, differs from (b)'s, 1.69657465869794022 − 2.4350e-17, by about 2×10⁻²¹.
(b) holds the harmonics at the prediction's own coefficients rounded to float64, the chain at the trend's, and the
two predictions come from different bases (chain2's and the polished one).
The restart (`--steps 0`) from that state started at 5.96e-36, since its segments are planned at the state rather than
at the step's re-seeded prediction, and converged in two steps to 2.85e-58, moving Delta by −6.25e-29. So a replanned
segment plan moves the root by about 10⁻²⁸ in Delta at D 194.

**(e) Unchanged paths: PASS** (`run_e.sh before|after`, `compare_e.py`, `compare_e.log`). The code of fc636e7 and
this change ran the same `--corrector qd --jet_o 1 --reseed 2 --reseed_q 1` step from D 180.52 (base polish 2.83e-30,
the octo-double series at N 8 in 660 s, a = 0.0625, the corrector 8.69e-20 → 9.96e-31 in two steps, Delta
1.700511940714). Their 12 log lines are equal with the times removed, the records' D, |F|, Delta, step counts and
moves are equal, and the saved states are equal bit for bit, meta included. The pre-change copy needed
`solver/data/formula_coefficients.json` beside it, which its first run lacked.

**(f) `check_numbers.py --selftest`: PASS**, 1043 of 1043 checks pass and each rejects a value shifted by one unit
(`check_f.log`).

## 6. Running the chain with the octo-double corrector

T129's options with `--corrector od --tol_od 1e-40` in place of `--corrector qd --tol_qd 1e-28` (M = 32, `--jet_o 1`):

    cd hp && python3 -u anm_run.py RAWDIR RAWDIR/TASK/chain --state STATE.npz --kfreeze 3 --N 24 --eps 1e-14
        --h_max 0.5 --steps 300 --D_max 1000 --threads 18 --corrector od --tol_od 1e-40 --qd_it 12 --trend T75
        --jet_o 1 --reseed 2 --reseed_q 1

A quad-double state (four parts) as the start is polished in octo-double first. An octo-double Newton step took
145–360 s on 3 threads of the Mac in these runs, against 19–41 s for a quad-double one, nearly all of it the Jacobian
(`newton_o`'s timing, 355 s of 360 s at D 188 under load, `probe_time.py`, `probe_time.log`). From a prediction the
corrector took three steps in (b) and (d), 518–629 s with the re-seed.

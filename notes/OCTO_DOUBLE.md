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

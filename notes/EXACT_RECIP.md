# The horizon series' exact 1/k (#1043)

hp's horizon series multiplies by 1/k at every order, and hp forms that factor as `recip_int(k)`, 1/k rounded to
double-double, in every lane: the quad- and octo-double lanes widen it. #1035 found that this rounding is part of the
discrete problem hp solves. With an exact 1/k the stored states miss their nodes by 4.4×10⁻³⁰ at D 73 and by
8.4×10⁻⁷ at D 242.36 ([FACTORED_HORIZON.md](FACTORED_HORIZON.md) 6, item 9). The opt-in flag of this note gives the
series 1/k rounded once to each lane's own precision, so that re-solves can measure how much Δ depends on the rounding.
The re-solves are for a Linux runner (section 3). Status words are AGENTS.md's.

| question | answer | status |
|---|---|---|
| does each lane take 1/k rounded once to its precision, and the flag off nothing new | yes, on 15 lane types for k ≤ 4096, bit for bit | checked (`recip_tests`, `exact_recip_check.py recip`) |
| does the kernel's flag move the rows as #1035's transcription does | yes, to 10⁻⁵² to 10⁻³² of the move at four states, and for a planted 2⁻¹⁰⁶ at single orders | numerical evidence (2) |
| where the rows' miss comes from | 1/3: `recip_int(3)` alone gives all of it at the four states, D 251.88 included | numerical evidence (2) |
| Δ(exact 1/k) − Δ(recip_int) at D 73 | +1.743109×10⁻²⁵, above #1043's bound of 10⁻²⁵, which left out the solve's gain of 4.3×10¹⁷; a second solve path predicts it to 10⁻⁹; 0.7% of the quoted error | numerical evidence, dh610's release build (3.3) |
| the same at D 241.96 | +3.290002×10⁻¹³; 0.87% of the quoted error | numerical evidence, dh610's release build (3.3) |
| the same at D 251.88 | no value: the flag-on polish stops at its start, whose projection fails; without that bound its first Newton step is 4.56×10⁻² and no trial is accepted | dh610's release build (3.3, 3.4) |
| the same at D 246.44 and 249.47 | no value: both flag-on polishes stop at their start, whose projection fails from a miss of 3.29×10⁻⁶ and 9.02×10⁻⁶ (growing about 1.4 times per unit D, 2.02×10⁻⁵ at 251.88); so D 241.96's change is the last measured one | dh610's release build (3.5) |
| where the flag-on polish's basin ends | between D 241.96 and 242.86: at the four bases D 242.86–245.56 the flag-on start misses by 1.00×10⁻⁶ to 2.45×10⁻⁶ and stops; lifting the bound at D 246.44 and 249.47 does not converge either | dh610's release build (3.6) |
| the same at D 242.21 and 242.46 (flag-off states between the edges, polished with the flag on) | +3.443409×10⁻¹³ (0.91% of 3.8×10⁻¹¹) and +2.798345×10⁻¹³ (0.82% of 3.4×10⁻¹¹), each converged in 5 steps to 4.5×10⁻⁹³ and 2.2×10⁻⁹²; with D 241.96 the change stays near 3×10⁻¹³, under 1% of the quoted errors | dh610's release build (3.6, job 5) |
| does the first-order response predict Δ's change at large D | no: at D 241.96 it gives 11.5 times the measured change, and its `prow` control 5.1 times unknown 0's own; suspected cause, untested: second-order terms along the weak directions | numerical evidence, Mac release-dev build on dh610's bases (3.3) |

## 1. The flag

| layer | name |
|---|---|
| kernel | `Ctx::exact_recip` (0 off, `RECIP_EXACT` = 1 on) and `Ctx::recip_planted` in `hp/src/model.rs` |
| Python binding | `Solver.set_exact_recip(mode, planted=0)` (drops the stored LUs), `Solver.horizon_recip(k)` |
| driver | `make_solver(..., exact_recip=1)`, `reseed.Reseeder(..., exact_recip=1)` |
| chain | `anm_run.py --exact_recip 1`, threaded to every solver of the run and to the re-seeder; records and saved states carry it |

With the flag on, `mul_recip` multiplies by `recip_value(ctx, k)`. That value is 1/k by greedy round-to-nearest
(`recip_parts`): r₀ = RN(1/k), and each further part is the nearest f64 to 1/k less the parts before it, formed from
the exact integer remainder. A lane takes the first parts that it carries: eight in octo-double, four in quad-double,
two in double-double, and an f64 value the first. That prefix is 1/k rounded once to the lane's type. The quad- and
octo-double lanes multiply by their constant in their own arithmetic, coefficient by coefficient on a jet.

With the flag off, `mul_recip` returns `x.muls(recip_int(k))`, the expression the series used before.

`recip_int(k)` is 1/k's nearest double-double at every k from 1 to 4096 (`recip_tests`), so the flag changes nothing
in the double-double and f64 lanes. It changes the quad- and octo-double lanes, at every k that is not a power of two.

The flag covers the three products by 1/(k + 1) per order of the horizon series (c, a and m), in its three forms:
`horizon_series`, `horizon_series_at` and `series_split`'s split form. These factors keep their double-double values
in every lane, flag or not:
- the centre series: (n − 1)/k from `recip_int(k)`, 1/(k + n − 1) and the inverse determinant of its 2×2 solve;
- the Taylor recursion: 1/(k + 1), and 1/(x₀ (k + 1)) as the double-double 1/x₀ times `recip_int(k + 1)`;
- the constants n = D − 2, the folded d/dθ, 2π and the phase weights;
- hp/modecount_cx, which keeps its own double-double copy of `recip_int`.

The planted change `Ctx::recip_planted = k` multiplies the series' 1/k by 1 + 2⁻¹⁰⁶ at that k alone. It is formed
exactly in multidouble's dyadic arithmetic and rounded as the flag rounds 1/k. hp/exact_recip_check.py's planted
check uses it.

## 2. Checks

Every check has a control that must fail it. Criteria were fixed before the first run; the one change after it, the
order of the `miss` check's control, is described below its table.

`cargo test --profile release-dev --lib recip_` (hp/src/recip_tests.rs, on the Mac, 56 s; all four pass):

| test | what it checks | result | control |
|---|---|---|---|
| `recip_lanes_exact` | `recip_parts(k)` is 1/k's greedy expansion for k = 1..4096, checked exactly in multidouble's dyadic arithmetic (no part of its integer algorithm); on 15 lane types (f64, double-, quad- and octo-double values, duals and jets) the flag-on product of 1 and 1/k is its first parts bit for bit, other coefficients zero; flag off it is `recip_int(k)`'s | 0 failures at every k on every lane | `RECIP_MUT_ULP` (last part one ulp up) caught at every k on every lane; `recip_int` widened fails the greedy check at all 4083 k that are not powers of two; the flag-off comparison sees the flag-on quad- and octo-double 1/k at those 4083 k |
| `recip_planted_value` | the planted 1/k at k₀ = 1, 2, 3, 7, 130, 294, 627, 4095, both modes: c (1 + 2⁻¹⁰⁶) rounded greedily, the change 2⁻¹⁰⁶ c to 2⁻⁴⁰⁰ c, k₀ ± 1 unchanged | passes at all 16 | the unplanted parts fail the planted value's check at all 16 |
| `recip_residual_lanes` | on the synthetic state: flag on against off in octo-double, rows outside the horizon block bit for bit; quad- against octo-double with the flag on, rel ≤ 2⁻¹⁸⁰ | 0 rows differ outside, 32 of 32 differ inside (rel 6.1×10⁻³⁴); quad against octo 4.6×10⁻⁶⁴ | quad-double on against octo-double off: 6.1×10⁻³⁴ |
| `recip_series_forms` | flag on, 12 planted orders: the series by order (`horizon_series_at`) against `residual_jet` (`horizon_series`) bit for bit; the split form within 16 max(d_off, 2⁻⁴²⁰) of the default | 0 rows differ; split equals default bit for bit at every order | 384 rows differ against the flag-off `residual_jet`; split on against default off, 6.8×10⁹³ times the threshold |

The existing tests that run the horizon series, with the flag off, pass unchanged on the same build: `freeze_` (3),
`halpha_control_bitwise_on_rounding_only`, `mode_` (2), `oseries_by_order_bitwise`, `osolve_agrees_with_quad_double`,
`qsolve_agrees_with_double_double`, `qv_lane_blocks_agree` and `split_` (3). `ojet_lanes_bitwise` was stopped
unfinished after 6 min on the shared Mac and was not run.

`hp/exact_recip_check.py` on the Mac's release-dev build of this branch (`HP_PROFILE=release-dev`, NEON), from `hp/`,
the states read from the raw-data repository (`git show`), 288 bits for the transcription, 4 threads. Its logs are in
the session's scratch space, not in the raw-data repository, which this task was not to write; the runner's
`I1043-<machine>/` reproduces them (3.2). tol = 10⁻⁶ relative throughout.

| command | result | control |
|---|---|---|
| `recip` (k = 1..4096) | PASS: the flag-on eight parts are mpmath's greedy expansion at every k, the flag-off parts `factored_horizon.recip_int`'s | `--mutate=off`: FAIL at all 4083 k that are not powers of two |
| `miss` I461 D 242.36 | PASS: the rows move by 8.36×10⁻⁷ (c 8.4×10⁻⁷, a 4.2×10⁻⁷, p 1.4×10⁻⁷, m 1.3×10⁻⁷); kernel and transcription differ by 1.9×10⁻⁵² of it | `--mutate=planted` (k = 3): FAIL, 4.0 |
| `miss` I603 D 241.96 | PASS: 7.31×10⁻⁷, differ by 3.4×10⁻⁵² | FAIL, 4.0 |
| `miss` I966-dh370 leg 6 D 251.88 (ks 293) | PASS: 2.02×10⁻⁵, differ by 6.4×10⁻³³ | FAIL, 4.0 |
| `miss` T81 D 73 | PASS: 4.39×10⁻³⁰, differ by 2.8×10⁻⁵³ | FAIL, 4.0 |
| `plant` D 242.36 at K = 3, 5, 6, 7, 9, 10, 12, 15, 20, 30, 60, 100, 110, 120, 125, 129, 130, 131 | PASS at every K, worst 3.2×10⁻³² of the move (K = 30) | `--mutate=shift` at K = 30, 60, 100: FAIL, 2.9×10⁻⁵ to 3.1 |
| `plant` D 73 at K = 3, 5, 7, 30, 60, 100 | PASS, worst 3.8×10⁻¹⁵ of the move (K = 100, a move of 8.2×10⁻⁷³) | FAIL at every K, 0.87 to 3.5 |
| `plant` D 251.88 at K = 3, 5, 7, 30, 60, 100 | PASS, worst 5.6×10⁻²¹ (K = 30) | FAIL at every K, 1.1 to 4.8×10² |

The miss reproduces #1035's 8.4×10⁻⁷ at D 242.36, its 7×10⁻⁷ at D 241.96 and its 4.4×10⁻³⁰ at D 73, and the
flag-off rows reproduce the stored states' m, a and c rows (4.29×10⁻⁷³ at D 242.36, the last line of its corrector in
`I461-win2/crawlJ.log`).

How the planted change moves the rows (the transcription's max |ΔR| for 1/K times 1 + 2⁻¹⁰⁶):

| K | D 73 | D 242.36 (ks 130) | D 251.88 (ks 293) |
|---|---|---|---|
| 3 | 1.8×10⁻²⁹ | 3.3×10⁻⁶ | 8.1×10⁻⁵ |
| 5 | 1.5×10⁻³² | 1.2×10⁻¹⁰ | 2.7×10⁻⁹ |
| 7 | 2.5×10⁻³⁵ | 6.5×10⁻¹⁵ | 1.3×10⁻¹³ |
| 30 | 7.8×10⁻⁵⁰ | 1.0×10⁻³⁸ | 2.8×10⁻²⁷ |
| 60 | 3.0×10⁻⁶¹ | 8.8×10⁻²³ | 1.3×10⁻¹ |
| 100 | 8.2×10⁻⁷³ | 6.5×10⁻¹⁷ | 1.6×10⁷ |

- **The miss comes from 1/3.** `recip_int(k)`'s relative error is −2⁻¹⁰⁸ at k = 3, 5, 6, 7, 9 and 10, so 1/3 alone
  moves the rows by a quarter of the K = 3 entry: 8.4×10⁻⁷ at D 242.36 and 2.0×10⁻⁵ at D 251.88, the whole miss, and
  the k = 3 control reads exactly 4.0 of it at all four states. The low orders move the a and c rows, the mass
  mode's. The first controls tried, at k = 60 and at k = ks, moved the rows by 10⁻¹⁶ and 3×10⁻³³ of the miss and did
  not fire; k = 3 is the control now.
- **At ks 293 a single middle order has a gain of 10³⁹.** At D 251.88 the response at K = 100 is linear, 1.3×10³⁹
  per unit relative change of 1/K from 2⁻¹⁰⁶ down to 2⁻¹⁴⁰ (`exact_recip_check.py gain STATE --k 100`), and it is in
  the p rows. It is presumably the p series' transient past Re s (FACTORED_HORIZON.md 5.1, "Why the loop is slow"),
  which ks 293 runs about 165 orders past at D 251.88 and ks 130 a few at D 242.36; that reading was not tested.
  `recip_int(100)` alone, exact 1/k elsewhere, moves the rows by 1.0×10⁶. Yet `recip_int` at every order moves them
  by 2.0×10⁻⁵, and 1 + 2⁻¹⁰⁶ at every order by 0.055 against 9.8×10⁶ at orders 100..294: the orders' contributions
  cancel. Why `recip_int`'s pattern of
  errors cancels so closely was not found. A rounding at a middle order that is not shared by its neighbours, as a
  quad-double lane's 2⁻²¹² per operation is not, is not cancelled: 2⁻²¹² times 10³⁹ is 10⁻²⁵ in the rows, an estimate
  of the quad-double floor of the horizon block at ks 293.

## 3. Runner commands

For a Linux runner (dh610 or dh370). The re-solves and the checks run from `hp/` of a worktree at this branch's head,
with the raw-data repository cloned beside the checkout. Raw output goes to a task folder `I1043-<machine>/` of the
raw-data repository, made first with its README (AGENTS.md).

```sh
W=~/Source/choptuik-large-d-1043; RAW=~/Source/choptuik-large-d-raw; OUT=$RAW/I1043-<machine>
git -C ~/Source/choptuik-large-d fetch origin
git -C ~/Source/choptuik-large-d worktree add $W origin/feat/exact-recip-1043
cd $W && uv venv --seed --python 3.14 .venv && source .venv/bin/activate && uv pip install -r requirements.txt
PIP_NO_CACHE_DIR=1 python -m pip uninstall -y multidouble; hp/build.sh
cd hp
```

The build took 33 min and 24 GB on dh370 with the low-memory recipe (#935, raw `I935-dh370/README.md`):
`CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 hp/build.sh`.

### 3.1 The flag off is bit for bit the kernel before

The reference checks of a build on a new machine (AGENTS.md), with `OMP_NUM_THREADS=1`:

```sh
python3 highm_check.py ref --compare=$RAW/T81/ref_main.json --Ms=48,64,80                     # 72 outputs, 0 differ
python3 highm_check.py scaled --compare=$RAW/T81/scaled_main.json                               # 22 outputs, 0 differ
python3 highm_check.py jets --compare=$RAW/anm-mac/jets_inputs/jets_inputs.json                 # 28 outputs, 0 differ
python3 highm_check.py jets --compare=$RAW/anm-mac/jets_inputs/jets_inputs.json --mutate=kappa  # control: 28 differ, exit 1
```

The four took 15 min together on dh370 (#935). They do not run the octo-double path, which the chain uses. `bits` does,
against a build of main whose `hp/src` is main's at this branch's base (multidouble 5.1.0, multishoot 4.1.0, multisolver
0.2.2 since #1056), run in that build's venv first. dh610's runs of these checks (3.3) were at the earlier base
6f401e0b, with multidouble 5.0.0, multishoot 4.0.0 and multisolver 0.2.1. About 5 min each at 10 threads (one `newton_o`
and three orders of `series_o` at D 242.36):

```sh
python3 exact_recip_check.py bits $RAW/I461-win2/crawlJ/anm_pin3_D242.3642.npz --save=$OUT/bits242_main.json --threads 10     # main's build
python3 exact_recip_check.py bits $RAW/I461-win2/crawlJ/anm_pin3_D242.3642.npz --compare=$OUT/bits242_main.json --threads 10  # 0 of 6 differ
python3 exact_recip_check.py bits $RAW/I461-win2/crawlJ/anm_pin3_D242.3642.npz --compare=$OUT/bits242_main.json --threads 10 --mutate=on  # control: exit 1
```

### 3.2 The flag against #1035's transcription

The commands of section 2's Python table, with `--out $OUT/<name>.json` and a log each.

### 3.3 Re-solves at the stored D

A base polish of `anm_run.py` at the state's own D (`--steps 0 --save_base 1`) re-solves Δ, saving the polished state
as `base_pin<k>_D<D>.npz`. Each state is polished twice, with `--exact_recip 0` and `1`, by the same command
otherwise, so that both polishes share the plan, the held values and the corrector. The flag-off polish is the
baseline, not the stored state. The I461 and T81 states were made by builds before #523's exact freeze projector
(raw `I603-win2/README.md`, `T81/README.md`), and #1035's transcription reads their p rows on the harmonics
k <= kfreeze at 1.6×10⁻³⁴ and 2.6×10⁻³⁴, above their other rows (FACTORED_HORIZON.md 6, item 11), so the flag-off
polish moves them too. T81's state stores no plan, and both of its polishes plan it again in double-double, the
same plan for both.

**I461's D 242.36 state does not polish on the current kernel, flag off.** On the Mac's build its first octo-double
Newton step is |dz| 3.4×10⁻⁹, and its trials at λ = 1 and 1/2 read max|F| 3.6×10³¹ and 5.7×10²⁹, each projection
failing (stopped after those two trials, 2.5–3 min each). The step is of the size of the frozen rows' 1.6×10⁻³⁴ over
crawlJ's smallest pivot, 3.5×10⁻²⁵, which is 4.6×10⁻¹⁰ (a reading, not tested). The large-D re-solve near
D 242 therefore takes I603's D 241.96 state instead, which main's exact-projector kernel made (`I603-win2/README.md`):
#1035's check state, a row of hp/data/delta_all.csv (quoted error 3.8×10⁻¹¹), and converged on the current kernel
(its flag-off polish ends at once at max|F| 2.0×10⁻⁶⁵). Its flags are crawlJ's, which I603 ran
(`I603-win2/run_603.sh`):

```sh
for R in 0 1; do
  python3 -u anm_run.py $RAW $OUT/resolve241_r$R --state I603-win2/chain/anm_pin3_D241.9602.npz --kfreeze 3 \
    --held_fit data/held_fit.json --held_o 1 --N 24 --jet_o 1 --corrector od --tol_od 1e-60 --qd_it 12 --lam_stall 0.04 \
    --od_junk 0 --reseed 2 --reseed_q 1 --reseed_o 3 --threads 10 --steps 0 --save_base 1 --exact_recip $R \
    > $OUT/resolve241_r$R.log 2>&1
done
```

I461's own pair, as #1043 names it, is the same command with `--state I461-win2/crawlJ/anm_pin3_D242.3642.npz` and
`resolve242`; expect its flag-off polish not to be adopted.

**At D 241.96, Δ(exact 1/k) − Δ(recip_int) = +3.290002×10⁻¹³** on dh610's release build (raw
`I1054-dh610/delta241.json`): 0.87% of hp/data/delta_all.csv's quoted error there (3.8×10⁻¹¹, fit spread). Its
flag-off polish converged at once (70 s, max|F| 2.0×10⁻⁶⁵); its flag-on polish (2948 s) stalled at max|F| 6.8×10⁻⁵¹,
above `--tol_od` 10⁻⁶⁰, and saved its base.

The flag-on polish's first step moved Δ by +3.29×10⁻¹³ (`dDelta`) through the smallest pivot, 4.6×10⁻²⁵, as on
the Mac's unfinished release-dev pair. Its last steps moved Δ by 3×10⁻²⁹ each, so the stall does not touch the
change, which is 2×10¹² times the D 73 change.

D 251.88, I966-dh370's leg 6 state with that leg's corrector and re-seed flags (`I966-dh370/ks_chain3b.log`; the
state's ks is 293):

```sh
for R in 0 1; do
  python3 -u anm_run.py $RAW $OUT/resolve251_r$R --state I966-dh370/ks_chain3b/leg06_D251.2845_ks293/anm_pin3_D251.8782.npz \
    --kfreeze 3 --held_fit data/held_fit.json --held_o 1 --N 31 --jet_o 1 --corrector od --tol_od 1e-60 --qd_it 40 \
    --lam_stall 0.04 --od_junk 0 --reseed 2 --reseed_q 1 --reseed_o 3 --od_accept natural --od_strong 4 \
    --reseed_start pred --reseed_qd_bar 1e-21 --threads 10 --steps 0 --save_base 1 --exact_recip $R \
    > $OUT/resolve251_r$R.log 2>&1
done
```

**At D 251.88 the pair gave no value** (dh610, raw `I1054-dh610/resolve251_r1.log`). The flag-off polish converged (3857
s, 7 steps, max|F| 4.55×10⁻³⁵ to 7.2×10⁻⁸²). ElimO counted 10 failed projections there, the 7 it logged at trial points
of max|F| 1.9×10⁻¹⁵ to 9.5×10⁻¹⁵, which do not stop an attempt. The flag-on polish stopped at its start:
`the start's projection failed, attempt stopped`, after 320 s, with no step and no base, so `delta` found no flag-on
file and exited 1. In `octo_corrector.ElimO` a failed first projection stops the attempt when the start's pinned
quad-double max|F| exceeds `start_F`, 10⁻⁶, and with the flag on that start misses its rows by 2.0×10⁻⁵ (section 2's
`miss`), so the re-seed had to correct a miss twenty times `start_F` and did not. Why its projection failed was not read
(no `--log_proj 1`). The stopped polish printed neither "base polish saved" nor "not adopted" (dh610's report).

D 73, the control: T81's full-system state (kfreeze 13, so no harmonic is held), the octo-double corrector without
the re-seed:

```sh
for R in 0 1; do
  python3 -u anm_run.py $RAW $OUT/resolve73_r$R --state T81/conv_pin5_D73.0000_D73.000000.npz --kfreeze 13 --N 24 \
    --corrector od --tol_od 1e-60 --qd_it 12 --threads 10 --steps 0 --save_base 1 --exact_recip $R \
    > $OUT/resolve73_r$R.log 2>&1
done
```

Δ(exact 1/k) − Δ(recip_int), with the three rows of hp/data/delta_all.csv nearest in D and their quoted errors:

```sh
python3 exact_recip_check.py delta $OUT/resolve241_r0/base_pin3_D241.9602.npz $OUT/resolve241_r1/base_pin3_D241.9602.npz --out $OUT/delta241.json
python3 exact_recip_check.py delta $OUT/resolve251_r0/base_pin3_D251.8782.npz $OUT/resolve251_r1/base_pin3_D251.8782.npz --out $OUT/delta251.json
python3 exact_recip_check.py delta $OUT/resolve73_r0/base_pin13_D73.0000.npz $OUT/resolve73_r1/base_pin13_D73.0000.npz --out $OUT/delta73.json
```

Each log's last octo-double Newton step gives dDelta, the corrector's own first-order error of Δ at its floor; the
change is resolved where it exceeds both runs' |dDelta|. Pass criterion of the control, fixed before the runs: at D 73,
|Δ(exact) − Δ(recip_int)| < 10⁻²⁵ (#1043). A polish that is not adopted saves no base; its log says
"base polish not adopted", and that run is to be reported, not rerun with `--base_adopt 1`.

**The D 73 pair on the Mac fails that criterion.** The commands above, run on the Mac's release-dev build at 4 threads
(output in scratch with `--allow_unraw`), took 5.3 and 5.5 min. The flag-off polish took T81's state from max|F|
2.6×10⁻³⁴ to 2.3×10⁻⁸³ in three steps, the flag-on one from 4.4×10⁻³⁰ to 2.0×10⁻⁷³, both on the same plan; the last
steps' dDelta are 2.1×10⁻⁴⁹ and 1.6×10⁻⁴³. Δ(exact 1/k) − Δ(recip_int) = +1.74×10⁻²⁵, 1.7 times #1043's bound.
It is 0.7% of hp/data/delta_all.csv's quoted error at D 73.0 (2.4×10⁻²³, kernel), so it does not touch the quoted
value, but it is 4×10⁴ times the rows' miss, where #1043 expected about the miss (10⁻³⁰). dh610's release-build pair
(raw `I1054-dh610/delta73.json`) gives +1.743109×10⁻²⁵, and its two polished bases are bit for bit the Mac's.

**The control failed, and its cause is the bound's premise.** Under AGENTS.md the fired control stopped this point
until its cause was known. The bound took Δ's change to be of the size of the rows' miss and left out the solve's
gain. `exact_recip_check.py predict` tests that by a second solve path, with its criterion fixed before the run: at
the flag-off base, r = F_exact (the flag-on residual in octo-double) and J the flag-on octo-double Jacobian; Δ's row
of J⁻¹ comes from Jᵀy = e_Δ solved by FLINT's `arb_mat` at 600 bits, not by hp's LUs (J's f64 condition is 1.7×10¹⁸,
past the 10¹³ at which f64 stops); the prediction is −yᵀr, to pass within 1% of the measured change. On dh610's two
bases (Mac release-dev build, one process, 1–2 min a run):

| run | ‖y‖₁ | max\|r\| | ‖y‖₁ max\|r\| | predicted | measured | relative difference |
|---|---|---|---|---|---|---|
| Δ's row | 4.33×10¹⁷ | 4.39×10⁻³⁰ | 1.9×10⁻¹² | 1.743109×10⁻²⁵ | 1.743109×10⁻²⁵ | 1.1×10⁻⁹: PASS |
| control `--mutate=offr`, r flag off | 4.33×10¹⁷ | 2.26×10⁻⁸³ | 9.8×10⁻⁶⁶ | 5.2×10⁻⁸⁸ | 1.743109×10⁻²⁵ | 1.0: FAIL |
| control `--mutate=prow`, unknown 0's row | 1.91×10²⁴ | 4.39×10⁻³⁰ | 8.4×10⁻⁶ | 1.686769×10⁻¹⁸ | 1.743109×10⁻²⁵ | 9.7×10⁶: FAIL |

The first-order response of the solution to the miss accounts for the measured change to 10⁻⁹, through a gain of
4.3×10¹⁷ from the rows to Δ, so the 10⁻²⁵ bound was #1043's premise error, not a fault of the flag or the solve. The
`prow` control's prediction is unknown 0's own change between the bases to all seven digits printed, a further check
of the same linear response. The bound is the coordinator's to replace; a bound that counts the gain is ‖y‖₁ times
the miss, 1.9×10⁻¹² here.

**The first-order prediction does not hold at D 241.96, so it cannot stand in for the D 251.88 re-solve.** With held
harmonics (kfreeze 3) `predict` takes r = F_exact − F_recip_int at the flag-off base, in which the held offset
cancels. On dh610's two D 241.96 bases (Mac release-dev build, 5 min a run): J's f64 condition 4.2×10⁵⁹, ‖y‖₁
3.5×10⁹, max\|r\| 7.3×10⁻⁷; predicted −yᵀr = +3.77×10⁻¹² against the measured +3.290002×10⁻¹³, a factor 11.5: FAIL.
The `prow` control's prediction misses unknown 0's own change too, by a factor 5.1 (−6.7×10⁻¹³ against −1.3×10⁻¹³),
so the failure is not peculiar to Δ's row. Where the miss is
4.4×10⁻³⁰ (D 73) the linear response is exact to 10⁻⁹; at a miss of 7×10⁻⁷ through pivots of 4.6×10⁻²⁵ it is not.
Suspected cause, untested: the second-order term along the weak directions, which a step of that size reaches. The
D 251.88 prediction was therefore not run to its end.

Expected times, from the chain's own steps at these states (an estimate):

| run | per octo-double Newton step | re-seed | the two polishes |
|---|---|---|---|
| D 241.96, ks 130 | 16 s on win2 at 18 threads (`crawlJ.log`, D 242.36), about 30–40 s on dh370 | 24 s | 5–20 min |
| D 251.88, ks 293 | 350–460 s on dh370 at 10 threads; the strong chord adds up to 1400 s (leg 6's log) | 236 s | 1–3 h |
| D 73, ks 130 | as D 242.36 | none | 5–15 min |

### 3.4 D 251.88 without the start bound

`anm_run.py --od_start_F 0` sets no bound on ElimO's start, so the flag-on polish goes on from its unprojected start
instead of stopping there; the default, 10⁻⁶, is the bound as before. The run is section 3.3's R = 1 with that flag and
`--log_proj 1`, into a new folder, then `delta` against 3.3's flag-off base (dh610's `resolve251_r0`):

```sh
python3 -u anm_run.py $RAW $OUT/resolve251_r1b --state I966-dh370/ks_chain3b/leg06_D251.2845_ks293/anm_pin3_D251.8782.npz \
  --kfreeze 3 --held_fit data/held_fit.json --held_o 1 --N 31 --jet_o 1 --corrector od --tol_od 1e-60 --qd_it 40 \
  --lam_stall 0.04 --od_junk 0 --reseed 2 --reseed_q 1 --reseed_o 3 --od_accept natural --od_strong 4 \
  --reseed_start pred --reseed_qd_bar 1e-21 --threads 10 --steps 0 --save_base 1 --exact_recip 1 \
  --od_start_F 0 --log_proj 1 > $OUT/resolve251_r1b.log 2>&1
python3 exact_recip_check.py delta $OUT/resolve251_r0/base_pin3_D251.8782.npz \
  $OUT/resolve251_r1b/base_pin3_D251.8782.npz --out $OUT/delta251b.json > $OUT/delta251b.log 2>&1
```

Pass rule, fixed before the run: the log's `octo-double Newton: K steps, max|F| X` line has X ≤ 10⁻⁵⁰, and the line
`base polish saved as` follows. The result is then `delta`'s Δ(exact 1/k) − Δ(recip_int), with its share of
hp/data/delta_all.csv's quoted error at D 251.886, 4.7×10⁻⁷ (weak pivot), which `delta` prints among the three
nearest rows. Otherwise there is no value, and the result is the run's `projection failed (n)` lines: the start's
pinned max|F| and what each stage of each failed projection reached. Expected time: 1–3 h, as the flag-off polish
(3857 s on dh610).

The flag's checks, on the Mac's release-dev build at 4 threads (scratch output):
- The default is the code before it bit for bit: the D 241.96 flag-off polish of 3.3 at the parent commit c68a0d12
  and at 4204c9f9 saves the same base (all 9 entries, the meta's 16 keys) and logs the same steps, times left out.
  ElimO is in that path, its start skipped below the switch.
- At D 251.88 flag on, with `--qd_it 0` (no step) and `--save_base 0`: the default stops at the start, "the start's
  projection failed, attempt stopped" (23 min), and `--od_start_F 0` goes on past the same failed projection to
  "octo-double Newton: 0 steps" with no stop (60 min).

**The run fails the rule, and D 251.88 has no flag-on value** (dh610 at 4204c9f9, 1462 s, raw `I1054-dh610/` at
497035d8, `resolve251_r1b.log`). The start is no longer refused, and its projection fails as before (pinned max|F|
2.02×10⁻⁵; the u0 rung's |R| 1.03×10²⁶ unchanged, its quad-double polish 3.19×10¹⁸ to 3.67×10¹⁵, the ladder's eight
rungs). From that unprojected start the first Newton step has |dz| 4.56×10⁻², the strong chord is refused (θ inf),
and the full step and every halving to λ = 1/256 land at max|F| nan or 1.9×10¹³⁵ and above, each projection failing:
`no trial step accepted`, `octo-double Newton: 1 steps, max|F| 2.0e-05, not converged`, `base polish not adopted`.
`delta` then finds no flag-on base and exits 1. Why a miss of 2.0×10⁻⁵ gives a step of 4.6×10⁻² was not read.

dh610's recommended task (not opened here): the same pair, with the default start bound and `--log_proj 1`, on the
saved bases of `I966-dh370/ks_chain3b` between D 242 and 251.88, to find the first D where the flag-on polish stops
converging, with Δ's change and its share of the quoted error up to there.

### 3.5 Where the flag-on polish stops converging, between D 241.96 and 251.88

The question for #1043 is whether Δ's change reaches the quoted errors near D 240–260: +3.29×10⁻¹³ against 3.8×10⁻¹¹
at D 241.96, and 4.7×10⁻⁷ quoted at D 251.886, where the flag-on polish does not converge (3.4). The pair of 3.3 (flag
off, then on, the default start bound, `--log_proj 1`) runs at two of I966's saved bases between them, in the code and
build of 3.4's run (4204c9f9):

```sh
F="--kfreeze 3 --held_fit data/held_fit.json --held_o 1 --N 31 --jet_o 1 --corrector od --tol_od 1e-60 --qd_it 40 \
   --lam_stall 0.04 --od_junk 0 --reseed 2 --reseed_q 1 --reseed_o 3 --od_accept natural --od_strong 4 \
   --reseed_start pred --reseed_qd_bar 1e-21 --threads 10 --steps 0 --save_base 1 --log_proj 1"
for S in leg01_D243.8361_ks293/anm_pin3_D246.4399 leg02_D247.2801_ks293/anm_pin3_D249.4664; do
  T=${S##*_D}                                        # 246.4399, 249.4664
  for R in 0 1; do
    python3 -u anm_run.py $RAW $OUT/resolve${T}_r$R --state I966-dh370/ks_chain3b/$S.npz $F --exact_recip $R \
      > $OUT/resolve${T}_r$R.log 2>&1
  done
  python3 exact_recip_check.py delta $OUT/resolve${T}_r0/base_pin3_D$T.npz $OUT/resolve${T}_r1/base_pin3_D$T.npz \
    --out $OUT/delta$T.json > $OUT/delta$T.log 2>&1
done
```

The states are ks 293 and eight-part, with their stored plans; their legs ran with `--reseed_qd_bar` 10⁻¹⁵ (leg 1) and
10⁻¹³ (leg 2), and 3.3's 10⁻²¹ is kept here so that the three D take one recipe. The nearest rows of
hp/data/delta_all.csv: D 246.2616, quoted error 2.1×10⁻¹⁰ (weak pivot), and D 249.3927, 1.0×10⁻⁸ (weak pivot).

Reading, fixed before the runs. A polish converges when its log's `octo-double Newton: K steps, max|F| X` line has
X ≤ 10⁻⁵⁰ and `base polish saved as` follows.
- (a) All four converge: the result is the three changes (D 241.96, 246.44, 249.47) against D and against their rows'
  quoted errors, and the least-squares line of log10 |change| in D through them, evaluated at D 251.88. That value
  is an extrapolation, not a measurement, and is compared with 4.7×10⁻⁷.
- (b) A polish stops at its start or does not converge: the basin of the flag-on polish from a flag-off state ends
  below that D, and the change at the last D where both converged is the result.

Expected time: about 1 h per polish at ks 293 (dh610's D 251.88 pair: 3857 s flag off, 1462 s flag on), so 2–6 h for
the four (an estimate).

**Result: reading (b).** dh610 ran the pairs at 4204c9f9 on 2026-11-20 (PR #1054 comment at 19:19 UTC, raw
`I1054-dh610/` a58ca739). Both flag-off polishes converged (7.6×10⁻⁸³ and 1.8×10⁻⁸² in one step each). Both flag-on
polishes stopped at their start, "the start's projection failed, attempt stopped". Their pinned max|F| at the start is
3.29×10⁻⁶ (D 246.44) and 9.02×10⁻⁶ (D 249.47), above the 10⁻⁶ start bound, with the projection's u0 rung at |R|
2.3×10²³ and 8.3×10²⁴, not converged. The flag-on miss of the stored states grows about 1.4 times per unit D (log10
slope 0.145 between neighbours), which run back to D 241.96 gives about 7×10⁻⁷, just under the bound, where the
projection succeeded. So the basin of the flag-on polish from a flag-off state ends between D 241.96 and 246.44, and
the measured change stays +3.290002×10⁻¹³ at D 241.96 (0.87% of its quoted 3.8×10⁻¹¹). No line through the changes
could be fitted, as only one exists. Not shown: whether `--od_start_F 0` converges at these smaller misses (at D 251.88
it took no step), and where between D 241.96 and 246.44 the projection starts to fail.

### 3.6 The two follow-ups of 3.5: the lifted bound below D 251.88, and the basin's edge

Both run in main's code (a worktree of main from 394c9f2d on, which holds `--od_start_F`) with dh610's venv-i1054,
the only dh610 build with `Solver.set_exact_recip`. Main's kernel differs from venv-i1054's by #1055's `plan_auto`,
which these runs do not call, and #1056's dependency versions, which #1056's checks found bit for bit on the flag-off
paths; hp/python is unchanged since venv-i1054's build (5f488e2b), and main's run path imports no multidouble name
newer than that venv's package. `F` is 3.5's.

(1) `--od_start_F 0` at D 246.44 and 249.47, against 3.5's flag-off bases:

```sh
for S in leg01_D243.8361_ks293/anm_pin3_D246.4399 leg02_D247.2801_ks293/anm_pin3_D249.4664; do
  T=${S##*_D}
  python3 -u anm_run.py $RAW $OUT/resolve${T}_r1b --state I966-dh370/ks_chain3b/$S.npz $F --exact_recip 1 \
    --od_start_F 0 > $OUT/resolve${T}_r1b.log 2>&1
  python3 exact_recip_check.py delta $OUT/resolve${T}_r0/base_pin3_D$T.npz $OUT/resolve${T}_r1b/base_pin3_D$T.npz \
    --out $OUT/delta${T}b.json > $OUT/delta${T}b.log 2>&1
done
```

Reading, fixed before the runs, per D by 3.5's rule (X ≤ 10⁻⁵⁰ and `base polish saved as`):
- It converges: the result is the change, with its share of the nearest row's quoted error (D 246.2616: 2.1×10⁻¹⁰;
  D 249.3927: 1.0×10⁻⁸, both weak pivot). With both and D 241.96's, the least-squares line of log10 |change| in D
  is evaluated at D 251.88, as an extrapolation, against 4.7×10⁻⁷; with D 246.44 alone the line is through two
  points and has no spread, which is said with it.
- It does not: there is no value at that D, and the result is the log's first Newton step (`on 0:`, its |dz|) and
  its `projection failed (n)` lines, as at D 251.88 (3.4).

Expected time: about 25 min per D when no trial is accepted (D 251.88: 1462 s), and up to about 2 h when it
converges (an estimate).

(2) The default-bound pair (3.5's) at the four bases between D 241.96 and 246.44:

```sh
for S in leg00_D237.7132_ks291/anm_pin3_D242.8630 leg00_D237.7132_ks291/anm_pin3_D243.8361 \
         leg01_D243.8361_ks293/anm_pin3_D244.7177 leg01_D243.8361_ks293/anm_pin3_D245.5586; do
  T=${S##*_D}
  for R in 0 1; do
    python3 -u anm_run.py $RAW $OUT/resolve${T}_r$R --state I966-dh370/ks_chain3b/$S.npz $F --exact_recip $R \
      > $OUT/resolve${T}_r$R.log 2>&1
  done
  python3 exact_recip_check.py delta $OUT/resolve${T}_r0/base_pin3_D$T.npz $OUT/resolve${T}_r1/base_pin3_D$T.npz \
    --out $OUT/delta$T.json > $OUT/delta$T.log 2>&1
done
```

The first two are leg 0's states, at ks 291; the others are at ks 293. Reading, fixed before the runs, per D by 3.5's
rule:
- Both converge: the result is the change, with its share of the nearest row's quoted error (D 242.7053: 3.4×10⁻¹¹;
  243.6366: 2.9×10⁻¹¹; 244.6366: 3.8×10⁻¹¹; 245.6366: 4.0×10⁻¹¹, each a fit spread).
- The flag-on polish stops at its start: the result is its start's pinned max|F| (its `projection failed (1)` line).
- The flag-on polish starts but ends above 10⁻⁵⁰ (a stall, or the `--qd_it` cap): there is no value at that D, and the
  result is its steps and last max|F|. For the bracket it counts as not converging, as a stop at the start does.
- The flag-off polish does not converge: the stored state does not polish on this kernel, and the D gives no reading.

The basin's edge is then bracketed by the last D whose flag-on polish converges and the first where it stops. The
start misses of 3.5, carried back at 1.4 times per unit D, give 1.0×10⁻⁶ at D 242.86 and 1.4×10⁻⁶ at 243.84, so the
start bound would be passed between them, if the start's projection fails there. This is a prediction, not a
measurement. Both of those states are leg 0's, at ks 291; the ks changes to 293 between D 243.84 and 244.72.

Expected time: about 1 min per flag-off polish, about 6 min per flag-on polish that stops at its start, and about 1–3
h per one that converges (D 241.96's took 49 min at ks 130): 0.5–10 h for the four (an estimate).

**Results** (dh610, venv-i1054, 2026-11-20; PR #1054 comments 7051048042 and 7051557411, raw `I1054-dh610/` ae8e6740).
(1) With `--od_start_F 0`, neither flag-on polish converges, at D 246.44 or 249.47. The start's projection fails, the
polish takes one step, and it is not adopted. So lifting the bound gives no value there, and no line can be fitted.
(2) With the default bound, all four flag-off polishes converge, and all four flag-on ones stop at their start:

| D | ks | flag-on start's pinned max\|F\| |
|---|---|---|
| 242.86 | 291 | 1.00×10⁻⁶ |
| 243.84 | 291 | 1.38×10⁻⁶ |
| 244.72 | 293 | 1.85×10⁻⁶ |
| 245.56 | 293 | 2.45×10⁻⁶ |

The misses rise about 1.37 times per unit D, as 3.5's did, and the first, at D 242.86, lies just above the 10⁻⁶ bound,
as predicted. So the basin's edge is between D 241.96 and 242.86, and the measured change stays +3.290002×10⁻¹³ at D
241.96 alone. Not shown: whether the unbounded polish converges at D 242.86, where the miss is smallest (dh610's
recommended task).

**Job 4 and job 5** (dh610, PR #1054 comments 7053991012 and 7054633400, raw `I1054-dh610/` 0068b60f and 3a509542).
The unbounded polish at D 242.86 does not converge either. A flag-on continuation from the D 241.96 base (job 4, two
steps of 0.25 with crawlJ's Newton flags) failed at once: its base polish went from 6.8×10⁻⁵¹ to 1.6×10⁻²⁴ and its
first step did not converge, while the flag-off continuation took both steps. But polishing job 4's two flag-off
states with the flag on and 3.5's flags (job 5: the strong chord, the natural test, the pred re-seed start) converged
at both, giving +3.443409×10⁻¹³ at D 242.2102 and +2.798345×10⁻¹³ at D 242.4602. Job 4 and job 5 differ in their
Newton flags and their inputs, so job 4's failure is not yet traced to either; the likelier cause is its flags.

## 4. Discrepancies (reported, not fixed)

1. **The flag covers the horizon series alone, as #1043 asks, but 1/k is not the only double-double constant of the
   discrete problem in the octo-double lanes.** The centre series ((n − 1)/k, 1/(k + n − 1), the inverse determinant)
   and the Taylor recursion (1/(k + 1), 1/x₀) keep double-double factors in every lane (section 1), as do n, d/dθ, 2π
   and the phase weights. The rounding of n moves D, that of 2π moves the solved Δ, both by about 10⁻³² relative, and
   the phase weights set the gauge. The folded d/dθ and the Taylor and centre factors change the equations at about
   2⁻¹⁰⁶ relative, like `recip_int`. In the all-outward layout their errors do not pass through the mass mode's gain
   G between the horizon and the last node, which the horizon series' do. Their share of Δ was not measured.
2. **The issue's two large-D states are not rows of hp/data/delta_all.csv.** Its rows near them are I603's chain:
   D 241.96 and 242.71 (quoted errors 3.8×10⁻¹¹ and 3.4×10⁻¹¹, fit spreads) and D 251.886 (4.7×10⁻⁷, weak pivot).
   `delta` prints the three rows nearest in D. I603's own states at D 241.96 (#1035's check state) and 251.886 would put
   the change beside a row: `I603-win2/chain/anm_pin3_D241.9602.npz` and `anm_pin3_D251.8859.npz`, re-solved as in
   3.3 with crawlJ's flags (I603 ran them, `I603-win2/README.md`).
3. **The D 242.36 and D 73 states predate the kernel.** They were made before #523's exact freeze projector, so their
   stored Δ is not a solution of the current kernel's system with either 1/k (FACTORED_HORIZON.md 6, item 11). Section
   3.3 takes Δ(recip_int) from a flag-off polish for that reason. At D 73 that polish converges; at D 242.36 its first
   step, 3.4×10⁻⁹, blows up (3.3), so #1043's I461 state cannot give the change, and section 3.3 runs I603's D 241.96
   state in its place.
4. `recip_int(k)` is already 1/k's nearest double-double (k ≤ 4096). The rounding that #1035 found is the
   double-double rounding itself, not an error of multidouble's division.
5. **The D 73 control exceeds #1043's bound:** Δ moves by 1.743109×10⁻²⁵ against 10⁻²⁵, on the Mac's build and on
   dh610's release build (3.3). #1043 took the change to be of the rows' miss, 4.4×10⁻³⁰. Through the solve's gain,
   4.3×10¹⁷ from the rows to Δ, it is 4×10⁴ times that, and a second solve path predicts it to 10⁻⁹ (3.3). The bound
   was the premise's error.
6. **D 251.88 gives no value with section 3.3's command:** the flag-on polish stops at its start, because ElimO stops
   an attempt whose first projection fails above `start_F` (10⁻⁶), and the flag-on start misses by 2.0×10⁻⁵.
   `--od_start_F 0` removes that bound, but the corrector then accepts no step from the unprojected start (3.4). The
   first-order prediction is no substitute (it misses the D 241.96 change by a factor 11.5).
7. **A base polish stopped at its start says nothing:** with no `od_F` and no `z8`, `anm_run.py`'s adoption branch
   saves and prints nothing (dh610's report); the run then reads as if the stored state were polished.
8. **Only some tools read a state in its own 1/k.** `anm_run.py`, `arclength.py`, `reseed.py`'s state checks
   (`pinned` and the re-seeder take the state's saved `exact_recip`; 0 when the key is absent) and
   `exact_recip_check.py` take the mode. The other 125 `make_solver` calls, in 85 scripts of hp/, build the
   flag-off problem whatever the state, so they read a flag-on state in the wrong discrete problem. At D 241.96,
   `reseed.py truthq` on the flag-on base re-seeds m_h to 8.3×10⁻⁶⁰ of the state's own in the state's mode, and to
   4.7×10⁻³⁶ in mode 0 (its limit is 10⁻³⁵; Mac release-dev build).
9. hp/modecount_cx's pins (multidouble `=4.9.0` against hp's `=4.9.1`) did not resolve before #1050; main pins
   `=5.1.0` in both since #1056, and `cargo check --locked` in hp/modecount_cx passes with its `Ctx` literal's two new
   fields.

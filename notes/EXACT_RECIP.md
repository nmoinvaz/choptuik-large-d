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
| the same at D 241.96 | +3.290002×10⁻¹³; 0.87% of the quoted error; the first-order prediction misses it by a factor 11.5 | numerical evidence, dh610's release build (3.3) |
| the same at D 251.88 | no value: the flag-on polish stopped at its start, whose projection failed | dh610's release build (3.3) |

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

The four took 15 min together on dh370 (#935). They do not run the octo-double path, which the chain uses. `bits`
does, against a build of main whose `hp/src` is main's at this branch's base (6f401e0b: multidouble 5.0.0, multishoot
4.0.0, multisolver 0.2.1), run in that build's venv first. About 5 min each at 10 threads (one `newton_o` and three
orders of `series_o` at D 242.36):

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
4.4×10⁻³⁰ (D 73) the linear response is exact to 10⁻⁹; at a miss of 7×10⁻⁷ through pivots of 4.6×10⁻²⁵ it is not. The
likely cause is the second-order term along the weak directions, which a step of that size reaches (not tested). The
D 251.88 prediction was therefore not run to its end.

Expected times, from the chain's own steps at these states (an estimate):

| run | per octo-double Newton step | re-seed | the two polishes |
|---|---|---|---|
| D 241.96, ks 130 | 16 s on win2 at 18 threads (`crawlJ.log`, D 242.36), about 30–40 s on dh370 | 24 s | 5–20 min |
| D 251.88, ks 293 | 350–460 s on dh370 at 10 threads; the strong chord adds up to 1400 s (leg 6's log) | 236 s | 1–3 h |
| D 73, ks 130 | as D 242.36 | none | 5–15 min |

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
   an attempt whose first projection fails above `start_F` (10⁻⁶), and the flag-on start misses by 2.0×10⁻⁵. A
   flag-on value there needs either a projection that succeeds from that start or a polish that may start unprojected
   (`ElimO`'s `start_F=None`, which `anm_run.py` does not expose); neither is in this PR. The first-order prediction
   is no substitute (it misses the D 241.96 change by a factor 11).
7. **A base polish stopped at its start says nothing:** with no `od_F` and no `z8`, `anm_run.py`'s adoption branch
   saves and prints nothing (dh610's report); the run then reads as if the stored state were polished.
8. hp/modecount_cx's pins (multidouble `=4.9.0` against hp's `=4.9.1`) did not resolve before #1050; main now pins
   `=5.0.0` in both, and `cargo check --locked` in hp/modecount_cx passes with its `Ctx` literal's two new fields.

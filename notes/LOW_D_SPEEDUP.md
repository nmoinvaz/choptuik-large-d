# Faster low-D points: two analyses and their synthesis (2026-11-04)

File and line references here are to hp at 9e63140, before #523 moved `qsolve.rs`, `osolve.rs`, `xsolve.rs` and
`block.rs` into multishoot's engine.

The user asked whether the points below D = 3.1 can be computed faster, by side-steps or by parallelization. Two
subagents answered independently: one read the code, the logs and the timing data (appendix A), one surveyed the
literature without reading the code (appendix B). Both reports are read-only analyses; nothing was built or run beyond
a seconds-long probe of a stored state. The coordinator checked the stale-checkpoint finding against the log and the
code ([LOW_D.md](LOW_D.md) section 12) and the survey's identity |C(1, iy)|² = tanh(πy)/(πy) numerically.

## Where the two agree

| Option | Code analysis (A) | Survey (B) | Gain | Effort | What it must pass |
|---|---|---|---|---|---|
| Restart 3.04375 from its 6.6×10⁻⁴ predictor, stale checkpoint set aside | first item: unblocks the chain | not seen (B quotes the stall as real) | unblocks the leg | 10 lines in `solve_point`, one rerun | M 320 converges with full steps |
| More threads per call (4 now, 16–20 after T99) | 2–3× per call; 1.75× measured from 4 to 8 | not covered | measured 1.87× (saturates at 12 threads, [LOW_D.md](LOW_D.md) 13) | none | bit identity across thread counts: passes at 4 and 16 |
| Reuse the factored Jacobian: chord steps within a point (A), a stale LU from the previous D as GMRES preconditioner (B) | 1.5–1.7× (chord after the second call) | minutes instead of 43 min per step if GMRES needs under ~50 products | 1.5× to 10×+ | Python only | Δ equal to the LU path's to the double-double floor; product count logged; mutated preconditioner as control |
| Exact Taylor predictor in D (the ANM of the large-D work), and predicting in s = √(D − 3) | 5–20× on the chain (steps 0.01–0.02 instead of 0.0006–0.0025); needs jet lanes for M 320/384 and a rebuild | first idea; the branch point at D = 3 degrades every predictor in D | 5–20× fewer points | rebuild (~1 h) + driver work | jet control `--mutate 21`; the 3.044375 state reproduced to the floor |
| Mapped τ grid concentrating modes on the pulse | probe on the stored D 3.0444 state: 1.8× fewer modes, ~6× per call | the only idea that attacks the growth of M; ~4× fewer modes estimated | 6× (measured probe) to 30× (estimate) | new discretization, validation at D 3.05–3.15 | identity map bit-identical; converged values at D 3.05 and 3.15 from fewer modes under the two-M rule; planted pulse |
| FFT for the τ derivative | 1.4–1.8× per call | 1.7–2× | 1.5–2× | kernel + a double-double FFT in multidouble | Δ to the floor (bits change); its own references |
| Ruled out | Newton–Krylov with a fresh LU (the LU is 0.3% of a call), Richardson in M, arc length (no fold), higher kt | parallel-in-time, multiple shooting in τ, f32 preconditioners | | | |

Where the time goes (A, measured): the Jacobian build is 99.5% of a Newton call, the LU 0.3% (block f64 storage, not
a dense double-double LU above M = 128), the residual the rest; the per-call cost grows like M³ times the number of
Taylor plan steps, and the plan grows toward D = 3 (1619 steps at 3.0475, 2955 at 3.044375). A point at 3.044375 cost
7.8 h of solves on win2 at 4 threads. The literature (B) solves the same problem the same way (uniform Fourier in τ,
shooting in x, dense Jacobian), and the one group that went toward D = 3 stopped at 3.05 as "too costly", with a
method slower per digit than this one.

Parallelization (A): the column groups of each node block are independent and already run on the thread pool; the
lever is the thread count. Running the two M levels of a point concurrently pays only if a call stops scaling near
8–10 threads; a pipeline over D needs the last converged state, so speculative starts cost what they save.

## Recommended order

| Step | What | Cost | Pass criterion |
|---|---|---|---|
| 1 | Move the stale checkpoint aside; make `solve_point` ignore a checkpoint with another plan; rerun 3.04375 from the predictor | one point, 3–8 h on win2 | M 320 converges with λ = 1 steps |
| 2 | Thread scaling on win2 (`highm_check.py profile` at 4–20 threads) and bit identity at 4 and 20 | ~40 min | Jacobian at 16–20 threads ≤ 0.5× the 4-thread time; hashes equal |
| 3 | Chord steps and a stale-LU GMRES within the leg | Python, a pilot at M 80–128 on the Mac | Δ to the floor; product count per step |
| 4 | The exact Taylor predictor at M 320 (jet lanes for B 40/48, rebuild, the planted pilot of appendix A, section 4 B) | half a day on win2 | prediction at −0.000625 within 1e-8 on its own plan; control fires |
| 5 | The mapped τ grid | a study first (Mac, minutes), then a new path | the converged values at D 3.05 and 3.15 from ≤ half the modes |


## Step 3: chord steps in the point solver (2026-11-04)

`hp/lowD_dd.py` `chord_phase`, used by `solve_point` when `--chord_from` (of `hp/lowD_highM.py`) is set, off by
default: after a full Newton step whose max |F| is below the threshold, chord steps with that step's LU (one residual
and one block solve each, `Solver.chord`), each taken only if it at least halves max |F|; the phase ends converged, at a
rejected trial, or after 20 steps, and the LU is refreshed once at the end for the sensitivities. `hp/test_chord.py`
on the D 4, M 48 state perturbed as `highm_check.py converge` does (Δ + 10⁻⁶, every unknown + 10⁻⁸ N(0, 1)), Jacobian
builds counted on the solver:

| case | converged | Δ | Jacobian builds |
|---|---|---|---|
| Newton only (the former rule) | yes | 3.44545279616531097 | 6 |
| chord from 10⁻⁸ | yes | the same, every digit | 3 |
| sign-flipped chord (the guard's control) | yes, 4 trials rejected | the same | 6 |
| guard removed (`--mutate`) | no: diverges to a non-finite pivot | | |

At M 320 a Jacobian build is about 2000 s and a chord step about 10 s, so the gain per point is the builds saved:
here half, at the leg's points about a quarter to a half (pass 1 from a predictor at 10⁻³ needs two full steps before
the threshold; pass 2 starts below it). Not yet run in the leg; T105 runs the former rule.

## Step 5, the study: a shared map saves about 3× in modes (2026-11-04)

`hp/tau_map_study.py` on the converged D 3.044375, M 384 state (`T100-win2/states`), output
[`hp/data/tau_map_study.json`](../hp/data/tau_map_study.json): the modes each of the 88 field blocks needs (the highest
harmonic above a level eps of its largest), on the uniform grid and after one periodic map τ = g(s) shared by all
blocks, the mapped fields evaluated on a grid eight times finer, the map's centre scanned over the half period.

| grid | modes needed by the slowest block, eps 10⁻⁴ | eps 10⁻³ | slowest block |
|---|---|---|---|
| uniform | 251 | 181 | centre p |
| sine, α 0.7 | 115 | 85 | centre p |
| arctan, L 0.3 | 91 | 67 | centre p |
| arctan, L 0.2 | 85 | 57 | node 7 p |
| arctan, L 0.1 | 145 | 97 | node 7 m |

Three readings. (i) About 3× fewer modes for the best shared map (arctan, L 0.2, centred on the centre field's pulse),
the same at both levels; by the M³ law of a Newton call that is about 25× per call, an extrapolation until a mapped
solve is run. (ii) The limit is the pulse's motion across the nodes: its position in τ sweeps the whole half period
from the centre (1.97) through node 8 (0.56) and node 12 (3.06) to the horizon side, so a sharper map moves the
slowest block to node 7, whose pulse lies off the map's centre. A map that follows the pulse in x as well is a change of
coordinates in (τ, x), not of the grid. (iii) The levels are where the state is resolved: its centre fields' top
harmonics sit at 3.5×10⁻⁶ of their peak, so the deeper levels (10⁻¹⁰, 10⁻¹²) measure the M 384 truncation, not the
pulse, and gave mapped counts equal to 384 times the map's largest stretch; those runs are not used. The analysis's
1.8× (the centre field alone) and the survey's 30–60× (an estimate) bracket the measured 3×.

## Step 5, the implementation so far (2026-11-04)

`hp/tau_map.py` builds the solver on the mapped grid from the two inputs that carry τ (the odd-sector d/dθ, rows
scaled by 1/g′, and the phase row, sin g · g′), with the map written as the identity plus terms in L − 1 so that L = 1
is exact; no kernel change. `hp/tau_map_check.py`: the identity map gives the uniform inputs and the residual bit for
bit (0 differing words of 4704 inputs; 0 of the residual). On the resolved D 8, M 64 state (its top harmonic 2×10⁻¹¹ of
its largest) mapped with L 0.9 the start's residual is 1.1×10⁻¹², Newton converges in 4 calls to |F| 5.5×10⁻³¹, and Δ
equals the uniform grid's to every digit: the mapped grid solves the same problem. Not yet measured: L ≤ 0.7 with each
state's own Taylor plan (an L 0.7 solve on the uniform plan overflowed, the plan's steps being too long for the
mapped samples), the plan's length on mapped grids (it multiplies the cost of a call), and the gain at low D (the
uniform M 48 state at D 4 is too under-resolved, 4×10⁻⁷ in Δ, to be mapped onto the same M). These runs need more than
the Mac's two threads until T103 ends.

## Step 5, the measurement (#1033, 2026-11-19)

`hp/tau_map_gain.py` measures, for each arctan map L, the modes M′ the mapped grid needs for Δ within 10⁻¹² of the
uniform grid's, the steps of its Taylor plan, and the cost of a Newton call against the uniform grid's. A call is 99.5%
Jacobian (appendix A, 1.2), and a Jacobian costs the plan's steps times c(M), the seconds per step, which grows about
like M³. The solves need hp's built module, so they run on a Linux runner (the commands below). What needs no kernel
has been run. The results section is filled from the runner's raw folder `I1033-<machine>/`.

### What the stored states say

`cd hp && python3 tau_map_gain.py stored <states>` on the uniform states (raw `T82-win/states`, `T100-win2/states`,
the stored states at D 3.0475 and 3.044375; each state's stored plan):

| D | M | Δ − Δ(previous M) | plan steps | of them in 0.01 → 0.02 (the valley) | from x 0.6 to the horizon |
|---|---|---|---|---|---|
| 3.0475 | 256 | | 1557 | 850 | 273 |
| 3.0475 | 320 | −1.95×10⁻⁹ | 1619 | 992 | 231 |
| 3.0475 | 384 | −3.77×10⁻¹¹ | 1623 | 942 | 277 |
| 3.044375 | 320 | | 2832 | 2150 | 232 |
| 3.044375 | 384 | −7.67×10⁻¹¹ | 2796 | 2070 | 278 |

- At M 384 the valley segment holds 58% (3.0475) and 74% (3.044375) of the plan. Its steps hardly change with M
  (850, 992 and 942 at M 256, 320 and 384), so they follow the solution's regularity near x = c (LOW_D.md 17), not the
  top harmonic's stiffness. A mapped grid that resolves the same solution should then keep the plan's length. This
  is the hypothesis the runs test.
- At 3.0475 the three levels converge geometrically, with a ratio of differences of 0.019, which puts Δ(384) about
  7.4×10⁻¹³ from its limit. That figure is an estimate that assumes geometric convergence, not a bound. So the uniform
  grid needs about M 384 for Δ to 10⁻¹², and Δ(384) can serve as the reference at that level. At 3.044375 only two
  levels exist, and the 10⁻¹² criterion is judged against Δ(384), whose own error is not estimated.
- `stored` on the D 3.7 ladder (hp/data/lowD_states, highm15) gives the first check's bound: |Δ(128) − Δ(96)| =
  3.36×10⁻¹¹, while Δ(128) lies 3.2×10⁻¹⁵ from Δ(256).

### Mode counts for every L

The study's scan, extended to L 0.9 to 0.2 (`cd hp && python3 tau_map_study.py RAW --state <state> --eps
1e-4,1e-3,1e-5 --sine "" --arctan 0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2 --out data/tau_map_study_L_D3.0444.json`, and
`_D3.0475.json` for the other state, the raw clone as RAW), gives these factors of fewer modes for the slowest block
at eps 10⁻⁴ and D 3.044375:

| L | 0.9 | 0.8 | 0.7 | 0.6 | 0.5 | 0.4 | 0.3 | 0.2 |
|---|---|---|---|---|---|---|---|---|
| fewer modes | 1.11 | 1.22 | 1.39 | 1.60 | 1.89 | 2.35 | 2.76 | 2.95 |

At the other two levels and at D 3.0475 the factors agree to within 0.08 for L ≥ 0.3. The gain is about 1/L down to
L 0.4 and saturates below 0.3, where the pulse's motion across the nodes sets the limit (reading (ii) above). The
kernel builds M = 128, 160, 192, 256, 320 and 384 in this range (`hp/src/python.rs`, `make_ctx`). Dividing 384 by the
factor and rounding up to one of these sizes predicts M′ = 384 at L 0.9, so no gain is available there, 320 at L 0.8
and 0.7, 256 at 0.6 and 0.5, 192 at 0.4 and 160 at 0.3. Those are the study's proxy. The runs measure M′ by Δ itself.

Two risks follow from the map's shape. At the same M the mapped grid's top harmonics are stiffer by up to 1/L, which
is why the earlier L 0.7 solve overflowed on the uniform plan. At M′ ≈ L·M the stiffness is the uniform grid's
again. Also, the map is centred on the centre field's pulse (s0 1.83–1.96), so the valley's τ point (grid point 27 of
384, θ ≈ 0.22) lies near s0 − π/2, where the grid is sparsest (spacing 1/L times the uniform one). Whether the valley
needs τ resolution there is part of what the runs show.

### Design

- **Starts.** The uniform M 384 state is interpolated onto the mapped grid in double-double (`tau_map.interp_dd`, the
  folded periodic sinc kernel built in mpmath). `map_state` keeps only f64, and a converged state's plan needs its lo
  parts: rounded to f64 it doubles the steps (`lowD_highM.py`, `plan_of`).
- **Solve.** These are the legs' two passes, on the mapped grid. Pass 1 runs on the start's own plan to |F| < 10⁻¹⁴.
  Pass 2 runs on the plan of the pass-1 state to 10⁻²⁸. Both use the f64 Jacobian in block storage. Then the converged
  state's own plan is made, the lean plan a leg reuses. A start is planned only after its residual is finite on a gate
  plan: the stored plan, then that plan split 2, 4 and 8 times, then the valley split further (`gate_plan`). Planning
  from a state that is nan does not return.
- **Cost.** The projected cost of a call is the own plan's steps times c(M′), against the uniform grid's own plan at
  M 384. It is given three ways. The first is the M³ law. The second is c(M) from `cost`: one f64 Newton call per
  repetition on a converged state at each M, on that state's own plan, as wall seconds per plan step on the same
  machine. The third is measured: the wall seconds of the pass-2 calls against those of the L 1 solve at M 384, also
  on the same machine. A plan made at another M is refused (`require_plan_at`), because its steps are too long for the
  tangents of a larger M. On win2 the D 3.7 state resampled to M 320 on the M 256 plan gave a nan pivot; the control
  `cost --mutate plan_m` exits 1. `Solver.step_timing()` is not used. In block storage its second entry is only the
  staircase LU's fill, a copy, because multishoot computes every block's columns before the factorization (multishoot
  3.3.0, `jacobian_staircase`). So since #523 its four entries no longer add up to the call (HIGH_M.md §4, #1040).
- **Modes needed.** For each L this is the smallest M′ whose Δ, and the Δ of every larger M′ of that L, lies within
  10⁻¹² of the uniform Δ(384).

### Checks, fixed before the runs

1. `tau_map_gain.py selftest` (no kernel, run on the Mac). `interp_dd`'s identity map returns a state bit for bit, and
   planted band-limited fields pass through five grid pairs within 4.6×10⁻³² (bound 10⁻²⁸). One pair has a target
   point 10⁻¹⁷ from a source point, because a map's centre is a float. The mapped d/dθ of a planted field is exact to
   2.9×10⁻³¹ (bound 10⁻²⁶), and the mapped phase row reproduces the uniform sum to 8.7×10⁻³³. The exact values come
   from the map's closed form and a Newton inverse, not from `tau_map`'s own map. Part 3 runs `report` on planted solve
   records. A failed identity solve must make it exit 1, and a failed larger M′ must leave no smaller M′ certified. Part
   4 tests the resume rules: a checkpoint of another configuration is refused, a converged result not yet published is
   recovered, and only a converged, saved result is reused. Each of the eight controls makes it fail:
   - `lo`, the lo parts dropped (up to 8.5×10⁻¹⁷);
   - `parity`, the odd kernel for even fields (0.9–1.4);
   - `dps40`, the interpolation matrix at 40 digits (2.1×10⁻²⁵ at the near point);
   - `inverse`, the inverse map skipped (1.3–1.5);
   - `gp`, the derivative rows without 1/g′ (0.31);
   - `phase_gp`, the phase row without g′ (0.19);
   - `lenient`, `report` dropping failed records, as it did at first (it exits 0 and certifies M′ 80 past a failed
     M′ 96);
   - `unbound`, any checkpoint restored and any record with Δ reused, as at first.
2. `check` at D 3.7. The mapped Δ at M 128 must equal the uniform Δ(128) within the uniform grid's own resolution error
   there, 3.36×10⁻¹¹, at L 0.9, 0.7 and 0.5. The control `--mutate gp` must exit 1. `--mutate phase_gp` is no control,
   because the phase row fixes only the τ origin. Dropping its g′ moves Δ by no more than the resolution error, and
   only the derivative rows' g′ can be tested through Δ.
3. `report` at low D. For each L at its largest M′, |Δ − Δ(384)| must lie within the uniform grid's own resolution
   error |Δ(384) − Δ(320)|, which is 3.77×10⁻¹¹ at 3.0475 and 7.67×10⁻¹¹ at 3.044375.
4. The identities. An L 1 solve from the stored M 384 state must converge to its stored Δ within 10⁻²⁰, with its own
   plan within 10% of the stored plan's steps. An L 1 solve at M 320 from the M 384 state must reach the stored
   Δ(320) within 10⁻²⁰. The plan bound was 1% at first and is 10%: the plan's tolerance (10⁻³⁰) sits near
   double-double's resolution, so copies of one converged state that differ below 10⁻²⁵ get plans a few percent
   apart (the Mac's D 3.7 pilot). Plan differences of that size between grids are noise. The control
   `report --mutate delta` moves every stored Δ by 10⁻¹⁰ and must exit 1.

### Runs for a Linux runner, cheapest decisive first

From `hp/` of a checkout of this branch, with hp built from main (any build since #523), the raw-data clone beside
the checkout, and 8 threads per job. The durations are estimates. They assume 0.35 s per plan step for an M 384
Jacobian at 8 threads (0.61 s at 4 threads on win2, section 1.1, with T98's 1.75× from 4 to 8 threads), c(M)
between the M³ and the M² law below that, one pass-1 call on a plan about 2.7 times the lean one, and three pass-2
calls.

```bash
RAW=$(cd ../../choptuik-large-d-raw && pwd); MACH=<machine>; OUT=$RAW/I1033-$MACH; mkdir -p $OUT/states
printf '# I1033-%s: the tau-mapped grid at low D, measured (hp/tau_map_gain.py)\n\n%s, https://github.com/nmoinvaz/choptuik-large-d/issues/1033\n' $MACH $MACH > $OUT/README.md
A=$RAW/T82-win/states/lowDhm_x0.000157-0.975_D3.0475; B=$RAW/T100-win2/states/lowDhm_x0.000157-0.975_D3.0444
G="nice -n 10 python3 tau_map_gain.py"; S37=data/lowD_states/highm15_D3.7000

# run 0, seconds: exit 0, then exit 1 for each control
$G selftest; for m in lo parity dps40 inverse gp phase_gp lenient unbound; do $G selftest --mutate $m >/dev/null; echo "control $m: exit $? (1)"; done
# run 1, D 3.7, about 15 min: the check (exit 0), its control (exit 1), the phase row (information), the gain at D 3.7
$G check --threads 8 --out $OUT/check37.jsonl
$G check --threads 8 --Ls 0.7 --mutate gp --out $OUT/check37.jsonl; echo "control gp: exit $? (1)"
$G check --threads 8 --Ls 0.7 --mutate phase_gp --out $OUT/check37.jsonl
$G solve ${S37}_M128.npz --pairs 1:128,0.9:96,0.7:80,0.7:96,0.5:64,0.5:80,0.3:64,0.3:80 --threads 8 --states $OUT/states --out $OUT/solve37.jsonl
$G report $OUT/solve37.jsonl --ref ${S37}_M256.npz --uniform ${S37}_M128.npz --json $OUT/report37.json
# run 2, about 45 min: c(M) on this machine, each converged state at its own M on its own plan
$G cost ${S37}_M128.npz ${S37}_M160.npz ${S37}_M192.npz ${S37}_M256.npz ${A}_M256.npz ${A}_M320.npz ${A}_M384.npz --threads 8 --reps 2 --out $OUT/cost2.jsonl
$G cost ${A}_M320.npz ${A}_M384.npz --mutate plan_m --out $OUT/cost2_control.jsonl; echo "control plan_m: exit $? (1)"
# run 3, D 3.0475, about 45 min, the decisive pilot: the uniform call, then L 0.4 and 0.5 at M' 192
$G solve ${A}_M384.npz --pairs 1:384,0.4:192,0.5:192 --threads 8 --states $OUT/states --out $OUT/solve_D3.0475.jsonl
$G report $OUT/solve_D3.0475.jsonl --ref ${A}_M384.npz --ref2 ${A}_M320.npz --cost $OUT/cost2.jsonl --json $OUT/report_D3.0475.json
# run 4, D 3.0475, the ladder: two jobs side by side, about 3.2 h and 2.7 h
$G solve ${A}_M384.npz --pairs 0.9:320,0.8:320,0.7:320,0.8:256,0.7:256 --threads 8 --states $OUT/states --out $OUT/solve_D3.0475.jsonl
$G solve ${A}_M384.npz --pairs 1:320,0.6:256,0.5:256,0.6:192,0.5:160,0.4:160,0.3:160,0.3:128,0.2:160,0.2:128 --threads 8 --states $OUT/states --out $OUT/solve_D3.0475b.jsonl
$G report $OUT/solve_D3.0475*.jsonl --ref ${A}_M384.npz --ref2 ${A}_M320.npz --cost $OUT/cost2.jsonl --json $OUT/report_D3.0475.json
# run 5, D 3.044375, about 5 h: the uniform call and, for each L, the M' that passed at 3.0475 and the one below
$G solve ${B}_M384.npz --pairs 1:384,1:320,<L:M' pairs from run 4> --threads 8 --states $OUT/states --out $OUT/solve_D3.0444.jsonl
$G report $OUT/solve_D3.0444.jsonl --ref ${B}_M384.npz --ref2 ${B}_M320.npz --cost $OUT/cost2.jsonl --json $OUT/report_D3.0444.json
```

Each solve checkpoints every Newton call in `$OUT/states` and resumes where it stopped. A configuration is the
source state, L, M′ and the map's centre s0, all in the file names, and a checkpoint restores only into its own
configuration. A rerun skips a configuration only when it converged and its state was saved, so a failed or
unconverged one continues from its checkpoint. Calls are capped (4 in pass 1, 5 in pass 2), so every job ends. Run 3
decides whether to go on. It goes on if both maps converge and L 0.4 at M′ 192 gives Δ within 10⁻¹² of Δ(384). Its
own plan's steps against the L 1 solve's then show whether the plan holds, and the wall seconds of its pass-2 calls
against the L 1 call's give the measured gain per call.

L 0.9 is tried only at M′ 320. At M′ 384 the mapped grid cannot cost less than the uniform one, so if 320 misses
10⁻¹² the answer for L 0.9 is that it gains nothing, and run 5 leaves it out. Run 5 repeats both identities
(`1:384` and `1:320`) at D 3.044375.

### Results

Pending the runs above.

## Appendix A: code analysis


Read-only analysis of main at 449db6e and the raw folders T82-win, T98-win2, T100-win2 (2026-11-04). Every number
below is from a file named next to it. One numpy probe of a stored state was run (seconds, `nice -n 10`, one thread;
script `pulse_probe.py` beside this file).

### 0. Corrections to the premise

| premise | what the files say |
|---|---|
| "dense double-double LU, L² double-doubles" | Above M = 128 the Newton step is `Solver.newton_mixed`: f64 Jacobian built with `Fun<4>` lanes, factored in f64 in block storage (a staircase LU over the node blocks, `hp/src/block.rs` 1–30, multidouble `linalg.rs:1778 StaircaseLu`), solves in double-double with the f64 factors. The dense paths refuse above `DENSE_M_MAX = 128` (`hp/src/python.rs:329, 370–378`; `driver.py:33`). The chain runs it: `lowD_highM.py:209–212` (`jac="f64"`). The LU is 418–691 MB and 0.2–0.3 % of a step (section 1). No L² anything exists at M = 320. |
| "one Newton step about 2600 s on win2, 32–56 % cpu, so the LU does not scale" | 2035–2088 s per call at M 320 on a 5491-step plan, 4026–4150 s at M 384 on 6698 steps (`T100-win2/down25e_points.jsonl`, `t_its`), at `--threads 4` by instruction (comms 2026-11-03 02:15 UTC, line 9140: affinity to 12 cores shared with T99's 10 busy processes). The 45–60 % machine cpu in the win2 posts is T99 + T100 together. It measures nothing about the kernel's scaling. The only scaling measurement is T98's: 4 → 8 threads gives 1.75× on the Jacobian (`T98-win2/hp_build/profile_M320.jsonl`). |
| "T100 at D 3.04375 did not converge" | The down25e attempt never ran from its best start. Its M 320 predictor had |F| 6.60e-4 on its own 6623-step plan (`down25e.log:63–68`), but `lowD_dd.solve_point` resumed the checkpoint left by the killed down25d attempt (`down25e.log:69–70`: "resuming from checkpoint …_p1_D3.0438_M320_ckpt.npz (iteration 4)", then "steps 4992, start |F| 1.0e-02"), the state that had already wandered (|dz| 20–40 at |F| 1e-2, `down25d.log:29–33`). down25d was stopped by the lam rule, so its checkpoint was never renamed `_failed` (`lowD_dd.py:181–188` resumes any checkpoint of the tag; the rename happens only on a returned failure, `lowD_highM.py:376–378`). The "stalls again with the same pivot drop" reading (comms 23:59 UTC, line 9684) is therefore not established: the half step from 3.044375 is untested. |
| "T100 ran with 6 threads" | The README of T100-win2 says 6; the down25e records say 4 (`threads: 4`), the 02:15 command says `--threads 4`. |

### 1. Where the time goes at M = 320–384

### 1.1 The point D = 3.044375 (down25e, win2, 4 threads beside T99)

| solve | start (|F|) | plan steps | Newton calls | s per call | seconds |
|---|---|---|---|---|---|
| M 256 helper | predictor 5.3e-2 | 3512 | 1 (`--helper_it_max 1`, fails) | 737 | 760 |
| M 320 pass 1 | predictor 3.2e-3 | 5491 | 4, all λ = 1 | 2035–2088 | 8297 |
| M 320 pass 2 | pass-1 state 8.8e-28 | 2832 | 2 | 1034–1091 | 2166 |
| M 384 pass 1 | lift of M 320, 1.1e-5 | 6698 | 3, all λ = 1 | 4026–4150 | 12385 |
| M 384 pass 2 | 6.2e-28 | 2796 | 3 | 1670–1740 | 5170 |
| all | | | 13 | | 28077 s = 7.8 h (wall 491 min) |

Source `T100-win2/down25e.log`, `down25e_points.jsonl`, `down25e.jsonl`. The cost per plan step is 0.37 s (M 320) and
0.61 s (M 384) at 4 threads, ratio 1.64 for (384/320)³ = 1.73. The two earlier points on win (laptop, 4 threads):
3.0475 took 81096 s = 22.5 h and 3.045 139284 s = 38.7 h of solves (`T82-win/down25_points.jsonl`; the 47797 s first
M 384 call there was under load). SOLVER_HP's "about 86 h for everything below D 4" is the CSV's sum of 31 `lowD_all`
rows that stop at D 3.075 (`hp/data/solve_times.csv`, 85.8 h); the three points below it add 69 h.

### 1.2 One Newton call, by part

| M | nodes | plan steps | threads | Jacobian | of it τ-derivative | LU factor | residual | LU MB | peak RSS | source |
|---|---|---|---|---|---|---|---|---|---|---|
| 320 | 13 | 825 | 4 | 382 s | 43 % | 0.68 s | 1.3 s | 418 | | T98-win2 `profile_M320.jsonl` |
| 320 | 13 | 825 | 8 | 218 s | 44 % | 0.48 s | 0.9 s | 418 | | same |
| 384 | 15 | 450 | 2 (load 5, cloud) | 4494 s | | 9.9 s | 14 s | 691 | 1.4 GB | `hp/data/lowD_highM/profile_big.jsonl` |
| 320 / 384 | 21 | 2800–6700 | 4 | 99.5 % of the call | | | | | 2.0 / 2.8 GB | T82-win records `peak_rss_mb` |

The Jacobian is 99.5 % of a call; the LU and the residual together under 0.5 % (HIGH_M.md §4, lines 312–352, found the
same at M 160–256). There is nothing to gain in the linear algebra: a Newton–Krylov or a better LU attacks 0.3 %.
These shares were measured on builds before #523. Since then the Jacobian is read from the call's wall clock, because
step_timing's own Jacobian entry holds only the copy into the blocks (HIGH_M.md §4, #1040). In #945's M 384 records,
made since #523, the residual alone took 92–115 s of Newton iterations of 7137–9737 s (raw
`I945-win2/leg/leg_points.jsonl`, the last record), over the 0.5% above. Whether the build or that run's plan and
series orders cause the difference is not known.

### 1.3 The Jacobian's structure and cost

- Unknowns: `L = 4M(K+1) + 1` (`model.rs:115–117`): centre data (p1, c0), K nodes × 4 fields, horizon data
  (m_h, a_h), Δ. At M 320, K 21: 28161.
- Residual block b (4M rows) depends on two neighbouring unknown groups and on Δ (`model.rs:128–140`, `block.rs:5–12`):
  block 0 on (p1, c0) and node 0 (the centre series, kc = 100 orders), blocks 1…K−1 on nodes b−1, b (a Taylor path over
  the segment, `block_state` 702–716), block K on node K−1 and (m_h, a_h) (the horizon series, ks = 250 orders with
  M × M double-double solves). Within a block every entry is nonzero: a unit tangent on one τ point of one field at
  the input node gives a dense 4M response, because the products and the τ-derivative couple all τ points. So J is
  block-bidiagonal in the nodes and dense in τ: 22 blocks of 1280 × 2561 ≈ 7.2×10⁷ f64 = 577 MB at M 320 (a dense
  matrix would be 6.3 GB f64). The staircase LU keeps exactly these blocks plus Δ copies (block.rs 8–16).
- Cost: every column task (`jac_tasks` 922–934: (4M + 1) per block) recomputes the block with f64 tangent lanes, four
  tasks per evaluation (`jac_columns_n` 854–912, `Fun<4>`). A Taylor segment costs steps × kt orders × [two dense
  M × M τ-derivatives (`taylor_coeffs` 350–351, `DMat::apply`) + about ten Cauchy sums of length k]. Per Jacobian the
  total is ∝ (plan steps) × M × kt × (M² + kt M): the M³ × steps law of HIGH_M §9.3, with the τ-derivative 43 % at
  M 320 (above) and 57 % at 256 (§4), the rest memory-bound Cauchy sums (MIXED_PRECISION §5, lines 293–352: the
  coefficient arrays, 12 × 38 × M × 5 lanes, leave L2 above M ≈ 128).
- The plan steps are the other factor, and they grow fast toward D 3: own plans of converged M 320 states 1619 (3.0475),
  2348 (3.045), 2955 (3.044375) (`steps`, `steps_own_w` in the records); a predictor's or lifted start's own plan is
  1.9–2.3× that (5491, 6623, 6698), and the reused lean plan of the last state no longer fits the predictors (|F| nan,
  `down25e.log:54–57`, `T82-win/down25c.log`). HIGH_M §1.5 explains why: a state that is not a solution at the D of
  the equations is not regular at the sonic line x = c, and its Taylor coefficients there are large. The two-pass
  rule (`lowD_highM.py:321–345`) therefore runs pass 1 on the fat plan and pass 2 on the lean one, which costs about
  1.3–1.5× the lean-plan solve (section 1.1: 8297 + 2166 against about 4 × 1050).
- Newton calls per point: 4 + 2 (M_w) and 3 + 3 (M_n) at 3.044375; 8 at the M 256 helper of 3.0475. Pass 2's calls
  move |F| from 1e-27 to 1e-29 at |dz| 1e-15: they are polish, 25 % of the point.
- Helper: 760–3700 s per point and it has failed at every point since 3.0475 (`down25_points`, `down25c_points`,
  `down25e_points`: 1–3 calls, |F| 3e-4 to 0.5); with `--helper_fallback` the point goes on without it. Cost 3 %, value
  nil at these D.
- Plans and start evaluations: about 20 min per point (the timestamps between solves in `down25e.log`).

### 1.4 What the kernel already parallelizes

| work | parallel over | source | scaling evidence |
|---|---|---|---|
| Jacobian build | the (4M+1)/4 ≈ 321 column groups of one block (`groups_b.par_iter`), blocks one after another, each factored before the next is built | `model.rs:1203, 1208`; HIGH_M §1.2 | 4 → 8 threads 1.75× (T98) |
| staircase LU factor | trailing-column updates in chunks (`par_chunks_mut`) | multidouble `linalg.rs:1468–1469` | 0.68 → 0.48 s; irrelevant |
| residual | the K+1 blocks (`into_par_iter`) and the ks+1 horizon LUs | `model.rs:743, 481` | 1.3 → 0.9 s |
| plan | the segments | `python.rs:140–150` | |
| `lu_solve_many`, `jvp`, `residual_jet` | columns / blocks | `python.rs:1099`, `model.rs:1127, 802` | |

The pool is one rayon pool per `Solver` with `threads` workers and 256 MB stacks (`python.rs:482–485`). Every task is
independent and the collects keep order, so the result should not depend on the thread count (expected, not verified:
no hash comparison across thread counts exists; the `scaled` check gives the hashes to do it). The 1.75× for 2× threads
at M 320 is consistent with the memory-bound Cauchy sums; win2's 12 E-cores beyond the 8 P-cores are untested
(unverified).

### 2. Side-steps, ranked

Gain is per point of the chain unless stated; the convergence rule is the pair rule "Δ(M_n) − Δ(M_w) below 1e-8"
(`--diff_max`, `lowD_highM.py:121`), which at 3.044375 is met with a 130× margin (7.7e-11).

| # | idea | expected gain | effort | risk | status in the notes |
|---|---|---|---|---|---|
| 1 | Restart 3.04375 from the 6.6e-4 predictor, stale checkpoint set aside; make `solve_point` ignore a checkpoint whose plan differs from the start's (or rename on kill) | unblocks the chain (no speedup) | 10 lines | none | the bug is new (section 0) |
| 2 | More threads on win2 now that T99 ends: 16–20 | 2–3× per call (1.75× measured for 4 → 8; the rest unverified) | none | bitwise identity across thread counts to be checked once | HIGH_M §9.3 "parallelizes nearly linearly" is an expectation, §9.7 the only measurement |
| 3 | Exact Taylor predictor in D (the ANM of QUAD_DOUBLE 13.15–13.17) at M 320/384 | fewer points (steps 0.01–0.02 instead of 0.000625–0.0025: 4–30× fewer), 2 Newton calls instead of 4, lean plans from the start (2×); 5–20× on the chain | kernel: jet lanes instantiated for B 40, 48 (now M 32/48/64 only, `python.rs:271–297`), a rebuild (52 min on win2, T98 `build.log:65`); driver: anm_step without the pinning | R_D at D 3.044 unknown (the sqrt branch at D 3 bounds it by 0.044); a_max then shrinks ∝ D − 3, still 5 steps to 3.005 at ε 1e-10 | built and chained at D 73–82 (13.16–13.17: predictions 1e-16 to 1e-9 at h ≤ 2, the LU and 24 right-hand sides 21 s at M 32); not tried at low D; `anm_step.py` uses `lu_solve` and `jvp`, both available in block storage |
| 4 | Chord steps and the lean plan inside a solve: after the second Newton call re-plan from the iterate, finish with `Solver.chord` (one residual + one block solve, about 10 s at M 320) instead of 2000–4000 s calls; pass 2 entirely by chord steps with pass 1's LU (the two plans differ at the 1e-30 truncation level, so that LU is an excellent preconditioner) | 1.5–1.7× per point (removes 25 % polish + halves the last fat-plan calls) | Python only (`newton_solve` has `chord=` already, `driver.py:334–335`; `lowD_dd.ns()` does not pass it) | the quadratic contraction constant C ≈ 140 at 3.044375 (1.4e-6 / (3.2e-3)²) says chord steps contract by C·|F| of the LU's point: 2e-4 per step from the |F| 1e-11 iterate, 0.4 from the first; so chord only after the second call | not tried below D 4; chord is covered by the bit-identity refs at M ≤ 128 (HIGH_M §2.1) |
| 5 | Solve M_n only every second or third point, with the pair difference projected (the driver already fits a log-linear trend in D, `projected_diff`) | 2.5× (M 384 is 63 % of a point) | small | changes what each quoted point carries: a measured pair becomes a projected one at skipped points; coordinator's policy | the rule is the README's "two largest M agree" per point |
| 6 | Drop the helper at these D (it fails every time) | 3 % | one flag | none | records above |
| 7 | FFT or DCT τ-derivative | 1.4–1.8× per call (43–57 % of the Jacobian, HIGH_M §4, §9.7) | kernel + multidouble (a double-double FFT over the lanes; "no Rust crate found", SOLVER_SURVEY §4.3) | changes bits: a new path with its own references and a control, the uniform `DMat` kept for the bit-identity refs | listed as the first optimization in HIGH_M §4 and SOLVER_SURVEY §4.3; not built |
| 8 | A τ mapping concentrating points on the pulse | the probe (below) says 1.8× fewer modes for the same tail in p1 → about 6× per call by M³, more if the plan steps follow M | DMat is an input (`dmat_odd`, `driver.py:37`), so a mapped derivative diag(1/g′)·D and mapped `phase_weights`, `resample_z`, tails and harmonics are Python; the kernel is untouched if the map keeps g(s + π) = g(s) + π | a new discretization: conditioning, the horizon series' periodic solves, the plan steps (the local frequency is k/g′, so the steps may not fall); the pulse moves with D | not in the notes |
| 9 | Deflated Newton with the junk directions frozen (`deflated_newton(junk_only=True, jac="f64")`, block storage) | for stalls only | exists (`driver.py:547`; `lowD_dd.py --solver dn`) | | HIGH_M §1.4: started at D 3.7, M 192, stopped once the node set was found to be the cause; at 3.044 the tails are fine (`tails_ok` true, 2e-59 / 7e-67), so the stall there is a start problem (section 0) |
| 10 | Newton–Krylov / sparse LU / f64 LU as preconditioner | none | | | the LU is 0.3 % of a call; the f64 LU already is the step's solver and GMRES with exact `jvp` products exists (`f64g`, `driver.py:168–186`); it corrects the step's direction, not its cost |
| 11 | Higher Taylor order kt or a looser plan | about none | | | h ∝ tol^{1/kt}: kt 36 → 60 gives 2.15× fewer steps for 1.7× (τ-derivative) to 2.8× (Cauchy) more per step |
| 12 | Richardson in M | not applicable | | | the M differences are e^{−2βM}, not a power law (LOW_D §2), the pair rule is already 130× inside its bound; an extrapolation would need a planted-truth test and would still be an estimator |
| 13 | Continuation in another parameter, arc-length | no fold near 3.044 (dΔ/dD ≈ 8 from the records; a_max from #3 would show one) | | | LARGE_D_LITERATURE 93: "walls are not folds"; SOLVER_SURVEY §4.4 "no folds". Predicting in s = √(D − 3) instead of D costs nothing and removes the D = 3 branch point from the predictor's series (the README's fit is a polynomial in s); untested |
| 14 | An asymptotic pulse form near D 3 as start or ansatz | unknown | analysis first | | LOW_D §6: the family degenerates as D → 3⁺, |F₁| ∝ (D − 3)^{−1.09}, spectrum flattens; no asymptotic form is derived anywhere in the notes (unverified that one exists) |

**The pulse probe** (`pulse_probe.py` on `T100-win2/states/lowDhm_x0.000157-0.975_D3.0444_M384.npz`): |p1| exceeds half
its maximum (674) on 7.3 % of the period; the odd harmonics fall by 0.0335 per mode (69 modes per decade; |F₁| = 82,
|F₃₀₁| = 1.6e-3). Composed with the map θ = s − α sin(2(s − θ_pulse))/2 (g′ = 1 − α at the pulses, half-period
symmetric), the decay is 0.052 (α 0.3), 0.060 (α 0.5), 0.055 (0.6), 0.046 (0.7): at best 1.8× fewer modes. One field,
one map family; the other fields and the horizon block were not probed.

Combining #2, #4 and #6 (no kernel change): a point of 7.8 h at 4 threads becomes roughly 7.8 / (2.5 × 1.6 × 1.03) ≈
1.9 h on 16–20 threads (the 2.5 is the unverified part). #3 then cuts the number of points.

### 3. Parallelization on win2 (20 cores, 95 GB) and the Mac

| what | concurrency | expected | memory |
|---|---|---|---|
| one call at more threads | the column groups (321 per block) | 1.75× at 8 (measured); 2.5–3× at 16–20 (unverified: E-cores and bandwidth) | 2.0 GB (M 320), 2.8 GB (M 384) per process |
| M_w and M_n of one point at once | M_n from its own predictor (|F| 4e-3 at 3.044375 against 1.1e-5 for the lift: 1–2 extra calls) at 10 threads each | wins only if the per-call scaling saturates near 8–10 threads; at 20-thread scaling the serial order is faster (4.2 h against 2.6 h by the 1.1 figures); the pilot's scaling curve decides | 5 GB |
| the helper beside M_w | its own predictor | 3 % saved | 1.3 GB |
| several D points in a pipeline | the predictor needs the last converged state (cubic through 4, `predict`, `driver.py:420`) | a speculative start from the current iterate costs the extra Newton calls it saves; no gain over threads unless scaling saturates | 2–3 GB each |
| plans | the segments already in parallel | nothing left | |
| the jets of #3 | `residual_jet` is parallel over blocks (`model.rs:802`) | one jet set ≈ 200 residual-equivalents at M 32 (13.16: 21 s with a 0.1 s residual); at M 320 a residual on a 3000-step plan ≈ 5 s at 4 threads → about 1000 s at 4 threads, a few minutes at 16 (unverified) | the `Jet<32>` work arrays about 85 MB per thread at B 48 |

The Mac (8 threads, 32 GB, full with T103 until about 05:00 UTC): not the chain (a Newton call at M 320 is hours
there). It can take the seconds-to-minutes work: the stale checkpoint's diagnosis (`lowD_dd.py --start_residual_only`
on `…_p1_D3.0438_M320_ckpt.npz`, one residual), the predictor-in-s against predictor-in-D comparison at 3.044375
(residual evaluations), the τ-mapping study of #8 on stored states (numpy), the planted-truth tests of #4 at M 80–128
on the D 3.7 states (minutes), and the ANM pilot's controls at M 32–64 where the jets already exist.

### 4. The first pilot

Two parts. Part A has no new code and is needed anyway; part B tests the top idea (#3).

**A. Threads and the untested start (win2, after T99's stage 2 ends).**
1. Scaling: `cd hp && python3 highm_check.py profile --M=320 --threads=N --reps=1` for N = 4, 8, 12, 16, 20 (T98 ran
   4 and 8 on the same default state, 825-step plan: 5–7 min each at N ≥ 8). Record `jac`, `residual`, `factor`.
   Pass: the Jacobian at 16 or 20 threads at most 0.5× its 4-thread time. Bit identity: `highm_check.py scaled
   --compare=T81/scaled_main.json --Ms=192,256` at 4 and at 20 threads, 22 hashes equal, the control (mutate 6) firing.
2. The point: move `T100-win2/states/lowDhm_x0.000157-0.975_p1_D3.0438_M320_ckpt.npz` aside, rerun the 02:15
   command with `--threads` at the best N of step 1. Stop: the lam rule as written (two M 320 calls with λ ≤ 0.5 at
   |F| > 1e-4). Pass: M 320 converges from the 6.6e-4 start with λ = 1 calls, as 3.044375 did from 3.2e-3. Cost: 3–4 h
   at 16 threads if scaling holds, 8 h otherwise.

**B. The exact Taylor predictor at M 320, planted truth before real data.**
- Kernel: add `at!(40); at!(48);` to `residual_jet_sized` (`python.rs:286–288`), rebuild on win2 (about 1 h), rerun
  `highm_check.py ref --compare … --Ms=48,64,80` and `scaled --compare` (no existing lane changes, so bit identity is
  expected, as the anm branch found, QUAD_DOUBLE 13.16).
- Base: the converged M 320 state at D 3.045 (`T82-win/states/lowDhm_x0.000157-0.975_D3.0450_M320.npz`) on its own
  plan, no pinning (kfreeze none below D 35); the LU from one `newton_mixed`, the jets to N = 16 with one `jvp`
  refinement per order as `anm_step.py:103–111` does.
- Truth: the converged states at 3.044375 (T100-win2) and 3.0475 (T82-win). Pass criteria, written first:
  (i) z₁[Δ] agrees with the slope through Δ(3.045) and Δ(3.044375) to 1e-3 relative;
  (ii) the prediction at h = −0.000625 has |F| ≤ 1e-8 on its own plan (the cubic predictor had 6.6e-4 to 3.2e-3) and
  that plan has at most 1.3 × 2955 steps; Newton from it reaches the stored Δ(320) = 2.4414880960708945 to 1e-20 in at
  most 2 calls;
  (iii) the prediction at h = +0.0025 (to 3.0475) has |F| ≤ 1e-6 on its own plan;
  (iv) the control `mutate=21` (JET_MUT_FREEZE_N, `model.rs:1082`) predicts z₀ and fails (ii) and (iii).
  Record the coefficient ratios ‖z_k‖/‖z_{k+1}‖: they are R_D, the distance to the nearest singularity in complex D,
  which answers the "branch or Jacobian near 3.044" question independently of any Newton run.
- Stop: the ratios fall below 1/0.0025 (a singularity nearer than the present step) or (ii) fails: then the ANM is
  not the predictor here and the ratios are the diagnosis.
- Cost: 1 h build, 1 h checks, one jet set (minutes to 20 min at 16 threads, unverified), two Newton calls at M 320
  (about 1 h at 16 threads): half a day on win2, under 4 GB. If it passes, the chain's step becomes a_max with
  ε = 1e-10 (about 0.4 (D − 3) per step), and the M 384 confirmation runs as now.

### 5. Discrepancies to report (not fixed here)

1. The down25e stall at 3.04375 resumed a stale checkpoint of the killed down25d run (section 0); the predictor start
   with |F| 6.6e-4 was never solved. `solve_point` resumes any `<tag>_D_M_ckpt.npz`, and a kill leaves it unrenamed.
2. T100-win2/README.md says `--threads 6`; down25e ran at 4 (its records and the 02:15 command).
3. HIGH_M §9.3's "parallelizes nearly linearly" is unmeasured beyond 8 threads; T98 measured 1.75× for 4 → 8.
4. The jet lanes exist only for M 32, 48, 64 (`python.rs:289`), so the ANM cannot run at M 320 without a rebuild;
   the notes do not say so for low D.
5. `hp/data/solve_times.csv` stops at D 3.075; the points 3.0475, 3.045 and 3.044375 (69 h) are not exported.

## Appendix B: methods survey


Second of two analyses. This one looks only at how the field and neighbouring fields solve this kind of problem; the
code itself is the first analyst's. Nothing was edited, built or run. Every statement about a paper comes from a page
opened in this session (WebSearch/WebFetch) or from a note in this repository, cited inline; "verified" below means the
bibliographic record was seen on a publisher, arXiv or indexing page, "partly verified" that some field (pages, venue)
rests on a secondary listing, "unverified" that no record was reached. Inferences of mine are marked *inferred*.

### 0. The problem, as the notes state it

| Fact | Value | Source |
|---|---|---|
| Formulation | periodic BVP in τ (period Δ unknown), x ∈ [0, 1], regular centre, regular singular point at x = 1; Fourier collocation on the half period (p, m odd, a, c even harmonics), products pointwise, no spectral truncation; Taylor stepping in x (order 36, planned steps), multiple shooting over K nodes; centre series kc = 100, horizon series ks = 250 as ks + 1 double-double M×M LUs | notes/SOLVER_HP.md; notes/HIGH_M.md 1.1–1.2 |
| Newton | exact double-double Jacobian by duals at M ≤ 128; above, f64 Jacobian (4 tangent columns per pass) in block storage with a staircase LU, double-double residual; GMRES correction with exact double-double J·v (`driver.gmres_refine`, 3–6 products at large D) | notes/HIGH_M.md 1.2; notes/LARGE_D_MIXED.md §3 |
| Size and cost | L = 4M(K+1)+1: 33 793 at M = 384 on 21 nodes; one M = 320 step 2618 s on win2 (4992-step plan), 15 044 s on win (7640 steps); the Jacobian is 98–99 % of a step, the τ-derivative matrix 45–57 % of the Jacobian at M ≥ 256 | comms 2026-11-03 23:59; notes/HIGH_M.md 9.3, 9.7 |
| Predictor | cubic through the last four points in D; below D ≈ 3.25 the predictor start has \|F\| 0.15–0.7, so each resolution starts from the rung below, resampled | notes/HIGH_M.md 5.1 |
| The echo at D = 3.044375 | 37.06 % of the odd field's power in the first harmonic, centre amplitude 674.3 (0.32 at D = 73) | hp/data/echo_shapes.csv; README "What repeats with that period" |
| Harmonic decay | \|F_k\| ~ e^(−βk): β = 0.157 (D = 4), 0.134 (3.7), ≈ 0.03 (3.046, Ecker et al.'s data); amplitude ~ (D − 3)^(−1.09) | notes/LOW_D.md §2 |
| Modes needed | M for 10⁻⁸: 80 at 3.7, 160 at 3.15, 320/384 at 3.05; projected 290–640 at 3.02, divergent at D = 3 in the amplitude-growth variants | notes/HIGH_M.md 5.3, 9.3 |
| Series radii | horizon series radius in u: 0.29 (3.7) → 0.155 (3.2); first node 0.000157 and last 0.975 at D = 3.05 | notes/HIGH_M.md 5.4, 9.6 |
| Where it stopped | D = 3.044375 converged (M 320/384, Δ = 2.441488095994242); at 3.04375 the M = 320 start from the reused plan has \|F\| 10⁻², two steps λ ≤ 0.25, smallest pivot 1.3×10⁻⁶; the predictor on its own plan had \|F\| 6.6×10⁻⁴ (not tried) | comms 2026-11-03 23:59 |
| Convergence rule | the two largest M agree to 10⁻⁸; the ± quoted is the resolution step, an estimator | notes/HIGH_M.md 5.1 |
| D → 3 | at n = 1 the scalar decouples and the free wave has no regular periodic solution (proved, Prop. 2 at n = 1); Δ fits a polynomial in s = √(D − 3) with a finite intercept | notes/LOW_D.md §6; README "Below D = 4" |

### 1. How the field computes DSS solutions, and what it cost near D = 3

| Work | Reference | Method | Resolution and precision | Cost / limit |
|---|---|---|---|---|
| Gundlach 1995 | PRL 75, 3214, gr-qc/9507054 (verified) | DSS + analyticity + reflection symmetry as a BVP; relaxation | Δ = 3.4439 ± 0.0004 | — |
| Gundlach 1997 | PRD 55, 695, gr-qc/9604019 (verified; ar5iv text read) | pseudo-Fourier in τ ("Typically I use 2N = 128", dealiased by doubling), implicit midpoint in ζ (51–1601 steps), relaxation (Numerical Recipes), Newton to machine precision; analyticity at the SSH by one condition (Eq. 25) | Δ = 3.4453 ± 0.0005; error = N vs 2N difference ×4; convergence "quadratic on the average" | not stated |
| Martín-García & Gundlach 2003 | PRD 68, 024011, gr-qc/0304070 (verified; ar5iv Sec. IV read) | uniform Fourier grid, N = 256 (tested 32–512), dealiased; shooting in x with IRK2 (implicit, 2-stage, order 4) on a fixed logarithmic grid (h_max 7×10⁻⁴), from x_left = 0.001 and x_right = 0.9999 to x_mid = 0.01; second-order series at both singular points; Newton on the free boundary data, mismatch 10⁻¹³ | Δ = 3.445452402(3); the series cutoffs set a floor "about 10⁻⁸ to 10⁻⁹ relative" | not stated; no mapped or adaptive τ grid |
| Ecker, Ecker, Grumiller & Jechtl 2026 | SciPost Phys. 20, 176, arXiv:2602.10185 (verified; HTML v1 Sec. 4.1, 5, 6.2, 8, App. A–B read) | same architecture: Fourier in τ (FFTW, zero-padded products), shooting from both cutoffs to x_match = 0.1 with a 2-stage implicit RK of order 4 on a logistic x-grid; Taylor series O(x_L⁵) at the centre and O((1−x_R)^2–3) at the SSH; Jacobian by forward differences (ε = 10⁻¹⁰), LAPACK, damping η = 0.1; continuation δD ≈ 0.01 with quadratic extrapolation | N_τ = 512 for D ≳ 3.2, "at least N_τ = 1024" below; N_x = 8000 (D ≥ 4) → 1.55×10⁵ (D = 3.05); x_L = 5×10⁻⁵ for 3.05 ≤ D < 3.1; "accurate to approximately one part in a million" (their estimate); the repo's comparison: +2.8×10⁻⁷ at D = 4, their 256→512 change 10⁻³ at D = 3.15 (notes/LOW_D.md §5) | "numerically too costly to go below D = 3.05"; Vienna Scientific Cluster. *Inferred:* one Jacobian ≈ (free data ≈ 3N_τ/2 columns) × (1.55×10⁵ RK steps × 1024-mode FFTs), of order 10¹² flops per Newton step, with η = 0.1 damping |
| Ecker, Ecker & Grumiller 2026 | PRL 136, 191401, arXiv:2601.14358 (verified) | analytic large-D family; no numerics near D = 3 | — | — |
| Bland 2007 thesis | U. Manitoba, handle 1993/20437 (record verified; text not read: the PDF is a fax-encoded scan) | evolution code (per notes/LOW_D.md §9) | D = 3.02: Δ = 0.1379 ± 0.0042 as quoted by Ecker et al.; not verified in the thesis | — |
| Reiterer & Trubowitz 2019 | CMP 368, 143, arXiv:1203.3766 (verified in notes/LIT_REVIEW.md) | computer-assisted existence proof at D = 4, Δ to 80 digits | — | not a continuation tool |
| 2+1 with Λ < 0 | Pretorius & Choptuik, PRD 62, 124012; Garfinkle, PRD 63, 044007; Jałmużna, Gundlach & Chmaj, PRD 92, 124044 (all verified) | evolutions; Garfinkle's exact CSS solution | CSS with c ≠ 0, not DSS | no seed for the D → 3⁺ DSS family |

What this says:

- Every DSS solver in print is uniform Fourier in τ plus shooting in x with a dense Jacobian and a dense LU; none uses
  a mapped or adaptive τ grid, extended precision, exact derivatives or planned Taylor steps. On precision and on the
  singular points the repository's solver is already past the field (the 10⁻⁸ floor of Martín-García & Gundlach came
  from second-order series at the cutoffs, which the double-double series to 10⁻³³ remove).
- The one group that went toward D = 3 (Ecker et al.) hit the same wall, from the same cause, at the same place
  (N_τ ≥ 1024 and N_x = 1.55×10⁵ at 3.05, then "too costly"), with a method that is slower per digit than ours.
- The literature has nothing on *representing* the pulse; that is the gap Section 2 addresses.

### 2. Methods that could replace or accelerate ours

Columns: what it exploits; fit (good / partial / poor, with the reason); expected gain; how it would be validated here.

| # | Method | Reference | Exploits | Fit | Expected gain | Validation here |
|---|---|---|---|---|---|---|
| 1 | **Mapped Fourier collocation** in τ: collocate uniformly in θ with τ = g(θ) periodic, g′ small at the pulse (Boyd's arctan/tan map), D_τ = diag(1/g′(θ)) D_θ | Boyd, J. Comput. Phys. 98 (1992), DOI 10.1016/0021-9991(92)90136-M (DOI and title verified; pages 181–193 unverified); Bayliss & Turkel, J. Comput. Phys. 101, 349–359 (1992) (verified; Chebyshev analogue); Tee & Trefethen, SIAM J. Sci. Comput. 28, 1798–1811 (2006), DOI 10.1137/050641296 (verified; adaptive map from Chebyshev–Padé poles) | β is the distance of the nearest complex-τ singularity from the real axis; a map that stretches the pulse moves that singularity away in the θ-plane, so the modes needed grow with log(1/width) instead of 1/width (Boyd's and Tee–Trefethen's claim, qualitative in what I read) | **good**: products stay pointwise, only D_τ and the phase weights change, the horizon recursion keeps its M×M LU per order; the half-period parity needs a Δ/2-periodic map centred on the pulse (the phase condition fixes it); at the continuum level the map is a periodic reparametrisation of τ, i.e. a lapse, so the solution is unchanged | at D ≈ 3.044 the pulse holds 63 % of the power outside the first harmonic and β ≈ 0.03; if a map brings β to the D = 3.7 value (0.13), M falls about 4× (384 → ~100) and a step (∝ M³) 30–60× (an estimate, to be measured); it is the only item that attacks the divergence of M at D → 3 | identity map must be bit-identical to main (control); at D = 3.05 and 3.15, where M = 320/384 and 192/256 are converged, the mapped grid at M′ = 96–160 must reproduce Δ to the two-M′ rule; planted truth: an antiperiodic pulse with known Fourier coefficients (e.g. a periodised sech) interpolated and differentiated on mapped vs uniform grids, error vs M |
| 2 | **Equidistribution (moving mesh) as the map's rule**: choose g from a monitor (arc length of p₁(τ)) | Huang & Russell, *Adaptive Moving Mesh Methods*, Springer AMS 174 (2011) (verified); Russell & Christiansen, SIAM J. Numer. Anal. 15, 59–80 (1978) (verified); AUTO's adaptive collocation, Doedel, Keller & Kernévez, IJBC 1, 745–772 (1991) (verified) | the mesh follows the solution as D falls | **partial**: as a rule for method 1's one or two map parameters, good; as a basis change (piecewise polynomial collocation in τ) poor, since it loses spectral accuracy per point and the horizon recursion's structure | sets w(D) automatically; no gain by itself | the map parameter's value against the pulse width measured on the converged states |
| 3 | **Jacobian-free Newton–Krylov** with a preconditioner: (a) a stale LU from the previous D (chord / Shamanskii); (b) a coarse-M LU prolonged by Fourier interpolation (two-level); (c) the τ-averaged, harmonic-diagonal Jacobian of harmonic balance | Knoll & Keyes, J. Comput. Phys. 193, 357–397 (2004) (verified); Sánchez, Net, García-Archilla & Simó, J. Comput. Phys. 201, 13–33 (2004) (verified); Viswanath, J. Fluid Mech. 580, 339 (2007), physics/0604062 (verified; hookstep); Kelley, *Iterative Methods for Linear and Nonlinear Equations*, SIAM 1995 (verified); Zang, Wong & Hussaini, J. Comput. Phys. 48, 485–501 (1982) (verified; spectral multigrid); Hall, Thomas & Clark, AIAA J. 40, 879–886 (2002) (verified); Feldmann, Melville & Long, CICC 1996 (partly verified: venue from a secondary source) | the Jacobian build (L/4 ≈ 7000 tangent passes) is 98 % of a step while one J·v is one dual pass (`Solver.jvp` exists); GMRES-IR theory (row 4) says the preconditioner only sets the iteration count, not the accuracy | **good for (a) and (b)**, conditional on the count of products; **poor for (c)**: the pulse is far from its τ-average (37 % in the first harmonic), so a harmonic-diagonal preconditioner models little of the coupling | if a stale or coarse LU keeps GMRES under ~50 products, a step at M = 320 is ~50 dual passes instead of ~7000: minutes instead of 43 min, with one Jacobian build per several D points; the repo's loop needs 3–6 products against a fresh f64 LU at large D (LARGE_D_MIXED §3), the open question is the count against a stale LU across a 0.00125 step near D = 3.044 | the fixed point is the same, so Δ must agree with the LU path to the double-double floor (as MIXED_PRECISION §4.4 did, 5×10⁻³¹); log the products per step; control: a scaled block in the preconditioner must change the product count but not the converged Δ; stop and report if the weak directions (HIGH_M 1.4) make GMRES stall |
| 4 | **Mixed-precision iterative refinement** (low-precision LU, high-precision residual, GMRES on the LU-preconditioned system) | Carson & Higham, SIAM J. Sci. Comput. 40, A817–A847 (2018), DOI 10.1137/17M1140819 (verified); Amestoy, Buttari, Higham, L'Excellent, Mary & Vieublé, SIAM J. Matrix Anal. Appl. 45, 529–552 (2024) (verified) | convergence to working precision needs only κ(A) ≲ 1/u_f for the factorisation precision u_f | **already in place** (f64 LU + double-double GMRES is GMRES-IR); as a new step **poor**: the LU is 0.3 % of a step, and an f32 Jacobian would fail, since the f64 one already agrees with the exact one only to 10⁻⁷ through the Taylor plans (MIXED_PRECISION §3) | none beyond what exists; the theory justifies row 3's stale LU | — |
| 5 | **Asymptotic numerical method** (Taylor series of the branch in the parameter from one LU and N jet right-hand sides; step from the series' radius) | Cochelin, Comput. Struct. 53, 1181–1192 (1994) (partly verified: secondary listings); Guillot, Cochelin & Vergez, IJNME 119, 261–280 (2019), DOI 10.1002/nme.6049 (verified via listings) | the predictor error is (h/R)^(N+1) instead of the cubic's h⁴ × fourth derivative; the branch's radius about D₀ is at least D₀ − 3 (the branch point at D = 3), so with h = 0.00125 at D₀ = 3.044, h/R ≤ 0.03 and N = 20 gives 10⁻³² | **good**, and the code exists for the pinned large-D route (`hp/anm_step.py`, `Solver.residual_jet`; notes/QUAD_DOUBLE.md 13.15); what is missing is its use in the low-D leg at M = 320 with the block f64 LU + `gmres_refine` as the solver of each order | lands the start inside Newton's basin where the cubic gave \|F\| 10⁻² (λ ≤ 0.25 and the stall); saves the failed steps and the halvings, and the series' radius is a step controller | ANM's control `--mutate 21` (frozen dimension, every z_k = 0); the predicted state's \|F\| on its own plan at D = 3.04375 against the cubic's 6.6×10⁻⁴ and the reused-plan 10⁻²; at D = 3.044375 the Newton from the ANM start must reproduce Δ = 2.441488095994242 to the double-double floor |
| 6 | **Parameterise the branch by s = √(D − 3)** (or arclength) instead of D | Keller, in *Applications of Bifurcation Theory*, Academic Press 1977, 359–384 (verified; origin contested, see Riks/Wempner); Allgower & Georg, *Introduction to Numerical Continuation Methods*, SIAM Classics 45 (2003) (verified) | the data say Δ (and plausibly the state) is a series in s with a finite limit: dΔ/dD = (1/2s) dΔ/ds, so in D the k-th derivative grows like s^(1−2k) and every polynomial predictor in D degrades toward D = 3; in s the branch is smooth and a step of 0.00125 in D is 0.003 in s | **good**, nearly free: a change of the predictor's abscissa (the cubic through four stored points, or row 5's series variable); notes/LARGE_D_LITERATURE.md (d) ruled arclength out for large D (no folds), the low-D end differs by the branch point at D = 3 | the 4-point cubic in s at spacing 0.003 should predict to ~10⁻¹⁰ relative where the cubic in D gave 10⁻²; also a diagnostic: the pivot drop to 1.3×10⁻⁶ at \|F\| 10⁻² is not a fold unless σ_min also drops along the *converged* branch (`hp/lowD_weak.py` can tell) | predict the converged D = 3.044375 state from the T82/T100 states above it in s and in D, compare \|F\| of the two starts; control: the s-predictor applied to a planted branch z(D) = polynomial in D must be no better than the D-predictor there |
| 7 | **Newton–Picard / deflation of weak directions** (Newton on the few strongly coupled directions, Picard on the rest) | Lust, Roose, Spence & Champneys, SIAM J. Sci. Comput. 19, 1188–1209 (1998) (verified) | the top centre and horizon harmonics couple to the matching point only exponentially weakly (HIGH_M 1.4) | **partial**: the repo's `deflated_newton` is this idea; near D = 3 the "weak" set is small since the pulse fills the spectrum | little new | — |
| 8 | **Multiple shooting / Chebyshev in τ** | Guckenheimer & Meloon, SIAM J. Sci. Comput. 22, 951–985 (2000) (verified; Taylor + AD multiple shooting for periodic orbits) | — | **poor**: the problem's evolution direction is x (Gundlach: ζ is time, τ is the periodic "space"); shooting in x with Taylor and AD is what `hp/` already does; Chebyshev is for non-periodic directions | none | — |
| 9 | **Parallel-in-time, periodic parareal, all-at-once circulant preconditioners** | Gander, in *Multiple Shooting and Time Domain Decomposition Methods*, Springer 2015, 69–113 (verified); Gander, Jiang, Song & Zhang, SIAM J. Sci. Comput. 35 (2013) (verified); Kulchytska-Ruchka & Schöps, arXiv:1908.05245 (verified; journal unverified); McDonald, Pestana & Wathen, SIAM J. Sci. Comput. 40, A1012–A1033 (2018) (verified) | parallelism over the "time" direction, circulant structure of time-periodic linear operators | **poor**: our x-direction is already decomposed (multiple shooting, block Jacobian, rayon over columns), and the circulant structure in τ is what the Fourier basis already diagonalises; the dense coupling comes from the x-propagation of the nonlinearity, which no basis removes | none | — |
| 10 | **FFT for D_τ** (in-repo item, SOLVER_SURVEY §4.3) | — | 45–57 % of the Jacobian at M ≥ 256 is the dense M×M derivative | **good**, orthogonal to the rest; with a mapped grid it is an FFT in θ times a diagonal | 1.7–2× per step | bit-level agreement not expected (different rounding); Δ to the floor |
| 11 | **Amplitude-and-width rescaled unknowns** (p, m divided by A(D), τ stretched by the pulse width) | Ecker et al. 2026 §6.2.2 scaling ansatz (verified) | if the rescaled pulse converges as D → 3 the unknowns stay O(1) | **partial**: it is method 1 plus a normalisation; Ecker's fitted exponents (δ ≈ 0.6, β ≈ 0.3) are from ten points at 3.05–3.056 and "apparently not small enough" | a predictor, not a solver | the rescaled states at 3.05–3.044 must collapse onto one curve before the ansatz is trusted |

Not in the table, for the record: f32 lanes for the preconditioner's Jacobian (2× on the build) are ruled out by the
10⁻⁷ accuracy of the f64 Jacobian already (MIXED_PRECISION §3); Boyd's "imbricate series" (a periodised pulse as a
basis function plus a Fourier remainder, same 1992 paper) needs the pulse shape and is a Section 3 question.

### 3. The structure near D = 3: is there a reduced problem?

**What is in print.**

- Ecker et al. 2026, §6.2.1 (Fuchsian system, D = 3 + ε): at leading order "the matter equations decouple from the
  geometric equations and can be solved independently", mode by mode; each mode equation is Fuchsian with regular
  singular points x = 0, ±1; regularity at the centre forces a (1 − x)^(−1/2) divergence at the SSH; "there is no
  quantization condition on the free parameter Δ" that removes it; "Δ → 0 damps the singular term via the cosh
  factor". §6.2.2 (scaling regime): x ~ ε^δ, e^ω = a/ε, f ~ ε^β, Δ ~ ε^(δ−β); fitted δ ≈ 0.6, β ≈ 0.3, so α ≈ 0.3,
  against α ≳ 0.15 from their numerics; "D = 3.05 is apparently not small enough". §5.1: narrow, tall peaks of Ψ and a
  sharp peak of the ℓ² norm of ω "at a decreasing value of x", so the pulse narrows in x as well as in τ; "such a limit
  may produce a δ-like distribution" (a conjecture). This is the only reduced-problem attempt found, and it is
  inconclusive.
- This repository (notes/LOW_D.md §6, proved): at n = 1 the scalar obeys the free 2+1 wave equation in self-similar
  form and no time-periodic solution is regular at both the centre and the light cone (C(1, s) ≠ 0). Hence the family
  degenerates as D → 3⁺; the data show how (amplitude (D − 3)^(−1.09), β → 0.03, radii shrinking), and the fit in
  s = √(D − 3) keeps Δ(3) finite where Ecker et al. read a power law.
- Clark & Pimentel 2025 (JHEP 11 (2025) 017; verified in notes/LIT_REVIEW.md) do matched asymptotics at the *other*
  end (large D, CSS); nothing analogous exists for D → 3.
- The 2+1 critical solution with Λ < 0 (Pretorius–Choptuik; Garfinkle; Jałmużna–Gundlach–Chmaj) is continuously
  self-similar with c ≠ 0 and needs Λ < 0; nothing ties it to the D → 3⁺ DSS family (notes/LOW_D.md §6).

**One observation of mine (hand derivation, not checked by code; a sympy check in the style of
`solver/lowD_n1_check.py` would settle it).** With the repository's C(1, s) = Γ(s + ½) / (Γ((1+s)/2) Γ(1 + s/2)) and
s = iy, the reflection and duplication identities give |C(1, iy)|² = tanh(πy)/(πy). So the free floor's k-th harmonic
(y ∝ kω = 2πk/Δ) carries a light-cone singular coefficient falling like (kω)^(−1/2): never zero (as proved), but
weaker for high harmonics and for small Δ. That is Ecker et al.'s "cosh factor" made explicit, and it says what a
reduced problem would have to do: the floor is a free wave whose every harmonic is mildly singular at the SSH, and the
pulse, where (D − 3)a² = O(1), must cancel all of them at once. By analogy with the large-D side (Props. 3–5: selection
beyond all orders), a leading-order matched problem may again leave Δ free, with the selection exponentially small in
the pulse width. I would not build a solver on such an ansatz before that is understood.

**Could an asymptotic ansatz enter the discretisation?** Two safe ways and one unsafe:

1. The mapped grid of method 1, with the map's width parameter w(D) following the pulse, is the discrete form of the
   stretched inner variable θ = (τ − τ₀)/w. It assumes nothing about the pulse's shape, only its location and width.
2. A predictor that rescales the last converged state by A(D) ∝ (D − 3)^(−1.09) in amplitude and by w(D) in width
   (method 11) can be tested now on the stored T82/T100 states (3.05 → 3.044375) against the cubic.
3. A split p = A(D)[P(θ) + floor] with P fixed by an inner equation is a research problem (the matching above), not a
   discretisation change.

### 4. Verification ledger

| Reference | Status |
|---|---|
| Gundlach 1995 PRL 75, 3214; Gundlach 1997 PRD 55, 695; Martín-García & Gundlach 2003 PRD 68, 024011 | verified (arXiv/ar5iv; publisher refs in notes/REF_CHECK.md) |
| Ecker, Ecker, Grumiller & Jechtl 2026, SciPost Phys. 20, 176 | verified; HTML v1 read through §8.3 and App. A–B; no App. C/D/F exists in that version |
| Boyd 1992, J. Comput. Phys., DOI 10.1016/0021-9991(92)90136-M | DOI, title, journal, date verified (MaRDI/zbMATH); volume 98, pages 181–193 unverified (ScienceDirect refused) |
| Bayliss & Turkel 1992, J. Comput. Phys. 101, 349–359 | verified |
| Tee & Trefethen 2006, SIAM J. Sci. Comput. 28, 1798–1811 | verified (abstract); the quantitative N comparisons were not read (PDF not renderable here) |
| Huang & Russell 2011; Russell & Christiansen 1978; Doedel, Keller & Kernévez 1991 | verified |
| Knoll & Keyes 2004; Sánchez et al. 2004; Viswanath 2007; Lust et al. 1998; Guckenheimer & Meloon 2000; Kelley 1995; Zang, Wong & Hussaini 1982 | verified |
| Hall, Thomas & Clark 2002 | verified |
| Feldmann, Melville & Long 1996 | partly verified (title and venue from Wikipedia and a Lucent patent family) |
| Carson & Higham 2018; Amestoy et al. 2024 | verified (secondary reference lists and MaRDI; publisher pages not opened) |
| Cochelin 1994; Guillot, Cochelin & Vergez 2019 | partly verified (secondary listings; notes/LARGE_D_LITERATURE.md (d) has the same status) |
| Keller 1977; Allgower & Georg 2003 | verified (library records) |
| Gander 2015; Gander, Jiang, Song & Zhang 2013; McDonald, Pestana & Wathen 2018 | verified |
| Kulchytska-Ruchka & Schöps | arXiv:1908.05245 verified; journal (DOI 10.1137/20M1314756 from a listing) unverified |
| Pretorius & Choptuik 2000; Garfinkle 2001; Jałmużna, Gundlach & Chmaj 2015; Clark & Pimentel 2025; Reiterer & Trubowitz 2019 | verified (this session or notes/LIT_REVIEW.md) |
| Bland 2007 thesis, the D = 3.02 point | record verified; content unverified (scan not readable here) |

### 5. The three ideas to test first

1. **The predictor: s = √(D − 3) and the existing ANM (rows 5 and 6).** It is the cheapest test and it addresses the
   observed failure (a start at |F| 10⁻² that Newton damps to λ ≤ 0.25 and then stalls). Everything needed is stored:
   predict the converged D = 3.044375 state from the points above it, in D and in s, and measure |F| of each start on
   its own plan; then run `anm_step.py`'s machinery from the 3.044375 state toward 3.04375 with its `--mutate 21`
   control. Pass: the start's |F| well below 6.6×10⁻⁴, full Newton steps, Δ at 3.044375 reproduced to the floor. A
   literature-standard remedy for a branch point in the parameter, and no change to the discretisation.
2. **A mapped Fourier grid in τ (row 1, with row 2's rule).** It is the only item aimed at the quantity that diverges,
   the number of modes, and it fits the kernel's structure (pointwise products, M×M horizon LUs, phase condition).
   Pass criteria written first: identity map bit-identical to main; the planted pulse's error vs M on mapped and
   uniform grids; at D = 3.05 and 3.15 Δ from M′ ≪ M agreeing with the converged values under the two-M′ rule. If the
   mode count does not fall by at least 2× at D = 3.05, stop.
3. **Matrix-free Newton–Krylov with a stale or coarse LU (row 3).** It attacks the 98 % of a step that is the Jacobian
   build, and the repository already has the pieces (`Solver.jvp`, `gmres_refine`, stored LUs). The measurement is one
   number: GMRES products per Newton step at M = 320 near D = 3.044 against the previous point's LU (and against the
   M/2 LU prolonged). Below ~50, a step costs minutes; above ~500 the idea is dead there. Pass: Δ equal to the LU path's
   to the double-double floor, with the product count logged and a mutated preconditioner as the control.

None of these bypasses the checks: each is scored against a converged pair that already exists (D = 3.05, 3.15,
3.044375), with a control that must fire and the two-M agreement rule unchanged.

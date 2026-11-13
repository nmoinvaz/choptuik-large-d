# The crawl at the chain's front: where the residual jumps along the Newton step (#700)

**Result.** The jump is ElimO's projection switch (`hp/octo_corrector.py`, `ElimO.project`): a trial whose pinned
quad-double max |F| is at most 1e-15 is not projected. At chain_ns's crawl iterate the corrector's residual along its
step is affine to 1.6e-54 lam² up to lam* = 1 - 7.77e-15. There it jumps by 5e38, to 8.9e-16, in the horizon
block's c, a and m rows, where the switch stops projecting. The iterate itself is never projected and carries a
residual of 0.129 in its horizon data, which only the full step removes. With every point projected the same
iterate converges in two full Newton steps, to 9.8e-80 (section 8).

## 1. The question

Past D 251.9 the octo-double corrector of #603 crawls. At each Newton step the natural test refuses lam = 1, at a
simplified-correction ratio of 34 to 9.4e16, and accepts lam = 1/2 at a ratio of exactly 0.5000. So max |F| and |dz|
halve per step until the step cap (#674 at D 251.886, a = 0.7276; #603's chain_ns at D 253.88, a = 0.0543). A smooth
residual along z + lam dz cannot be affine at lam = 1/2 to four digits and 1e22 to 1e36 times larger at lam = 1.
This note locates the point lam* where it stops being affine, names the rows that jump there, and names the branch in
the code that makes them jump.

## 2. Method

`hp/crawl_probe.py`. Its output is in folder `I700-mac/` of the raw-data repository, at commit f2c81ea (clone
github.com/nmoinvaz/choptuik-large-d-raw first; the commands below take its root as RAW). The Mac ran it on 2
threads with main 3c35f0c's kernel build.

- `locate_jump` evaluates F(lam) = F(z + lam dz) on a ladder of lam and measures its deviation from the affine model,
  e(lam) = max |F(lam) - (1 - lam) F(0)|. Along an exact Newton step a smooth map deviates by its quadratic
  remainder, which grows like lam². A rung whose e exceeds K = 1e3 times the previous rung's e carried by
  (lam_b / lam_a)² is a jump. Its bracket is bisected, with the threshold set at the geometric mean of the two sides,
  until it is narrower than 1e-6. The rows that carry the jump are the fewest whose squared entries hold 99% of
  |F(lam_b) - F(lam_a) + (lam_b - lam_a) F(0)|².
- `iterate` rebuilds one attempt of anm_run.py with #603's flags and runs `newton_od` to its first crawl iterate. It
  saves that iterate z and its Newton step dz exactly as `ElimO.newton_o` formed them inside `newton_od`.
- `probe` runs `locate_jump` at that iterate on three residuals. Arm E is the corrector's own (ElimO: the trial
  projected when its pinned quad-double max |F| exceeds the switch, 1e-15, as the natural test evaluates it). Arm U is
  never projected (PinnedO). Arm P is always projected. It also forms the natural test's ratio at lam = 1 and 1/2,
  with the trial as the corrector projects it and with the trial projected.
- `fixrun` continues the corrector from the saved iterate with ElimO's switch set as asked (section 8).

The runs, from `hp/` with `OPENBLAS_NUM_THREADS=1` and the kernel on `PYTHONPATH`, RAW the raw-data clone's root:

```
python3 crawl_probe.py selftest        # PASS; --mutate shift, comp, smooth, control, coarse and hint_trust FAIL
python3 crawl_probe.py iterate RAW I603-win2/chain_ns/anm_pin3_D253.8825.npz --a 0.05429687500004832 \
    --jet I603-win2/chain_ns/jet_pin3_D253.8825.npz --check_pred I603-win2/chain_ns/pred_pin3_D253.9368.npz \
    --out RAW/I700-mac --threads 2 --strong 4 --steps 2
python3 crawl_probe.py probe RAW RAW/I700-mac/crawl_iter_D253.9368.npz --out RAW/I700-mac --threads 2 --rungs 6 --arms U,E
python3 crawl_probe.py fixrun RAW RAW/I700-mac/crawl_iter_D253.9368.npz --out RAW/I700-mac --threads 2 --switch 0 --steps 4
```

## 3. The planted check (fixed before the first real run)

A smooth map on 161 unknowns, F(x) = A x + b + eps x∘x with eps = 1e-6, at a random start with its exact Newton step,
so that F(lam) = (1 - lam) F(0) + eps lam² dz∘dz. A jump of amplitudes (0.7, 0.7, 0.1) on rows 140, 141 and 150 is
switched on where a random linear form passes its value at lam_true = 0.7316807. Pass: lam* within 1e-6 of lam_true,
the 99% rows equal to the planted rows, and the same map without the jump reported affine to lam = 1. The five
mutations must each make it fail.

| run | lam* | rows carrying 99% | control | verdict |
|---|---|---|---|---|
| as planted | 0.731680727 (2.7e-8 off) | 140, 141, 150 | affine to 1 (growth 1.00) | PASS |
| `--mutate shift` (planted 5e-6 later) | 0.731685686 | 140, 141, 150 | affine | FAIL, as it must |
| `--mutate comp` (planted one row over) | 0.731680727 | 141, 142, 151 | affine | FAIL, as it must |
| `--mutate smooth` (no jump planted) | none | none | affine | FAIL, as it must |
| `--mutate control` (a jump in the control) | 0.731680727 | 140, 141, 150 | jumped | FAIL, as it must |
| `--mutate coarse` (bisection stopped at 1e-2) | 0.734375000 | 140, 141, 150 | affine | FAIL, as it must |

The first plant had eps = 1e-3, so its quadratic remainder at lam = 1 was 9e-3, and the jump stood only 135 times
above it, under K. The detector reported it affine, as specified, and that plant was resized to eps = 1e-6 before
any real run. So `locate_jump` resolves a jump only when it is more than K = 1e3 times the remainder at its rung.

The detector also takes a bracket hint, a narrower bracket proposed from a cheaper quantity, and keeps it only when F
reads affine at its lower end and jumped at its upper end. Otherwise those two evaluations narrow the bracket as
bisection steps would. The planted check runs three times: without a hint, with a hint around lam_true, and with a
hint 0.01 away, which is refused. A sixth mutation, `hint_trust`, takes the hint without evaluating its ends and must
fail (the wrong hint then gives lam* 0.7417). Each of the six mutations fails, and the unmutated check passes all three
runs (`selftest.log`).

## 4. The crawling attempt, rebuilt

This is #603's chain_ns attempt from D 253.88245811752 at a = 0.05429687500004832 (refused on win2 after 20 steps, at
4.3e-59). It is rebuilt from `I603-win2/chain_ns/anm_pin3_D253.8825.npz` and its stored octo-double series
`jet_pin3_D253.8825.npz` (`iterate_ns_D253.9368.log`, 2 threads, 3433 s).

| | win2 (`chain_ns.log`) | Mac rebuild |
|---|---|---|
| D + a | 253.93675499252282794053481307372521769138984382152 | the same string |
| prediction | `pred_pin3_D253.9368.npz` | equal bit for bit |
| re-seed | failed, prediction kept | failed, prediction kept |
| start max\|F\| | 4.01e-20 | 4.01e-20 |
| chord max\|F\| | 1.42e-17, 2.01e-37, 1.85e-54 | 1.42e-17, 3.55e-37, 4.25e-53 |
| chord \|dw\| | 2.081e-37, 2.076e-37, 2.446e-53 | 3.687e-37, 3.659e-37, 1.659e-53 |
| theta at the chord's point | 7.993e-12, accepted | 7.993e-12, accepted |
| step 0 | 4.01e-20 → 2.26e-53, \|dz\| 3.46e-26 → 3.46e-26 | 4.01e-20 → 2.58e-53, \|dz\| 3.46e-26 → 3.46e-26 |
| step 1, lam 1 | max\|F\| 1.42e-17, ratio 9.448e+16, refused | max\|F\| 1.42e-17, ratio 9.448e+16, refused |
| step 1, lam 1/2 | max\|F\| 1.13e-53, ratio 5.000e-01, accepted | max\|F\| 1.29e-53, ratio 5.000e-01, accepted |

From the first chord step on, the two runs differ in the chord's |dw|, by up to a factor 1.8. The chord's weak
subspace comes from six subspace iterations through numpy's QR and eigh, which round differently under the Mac's
LAPACK. The crawl itself is reproduced: the same refusal at lam = 1, with the same ratio to four digits, and the
acceptance at lam = 1/2 at exactly 0.5000. The iterate after step 0 (the chord's point) and its Newton step are saved
as `crawl_iter_D253.9368.npz`.

**The saved iterate's own pinned quad-double max |F| is 0.129.** That is the residual of the point without its
projection, which is what ElimO's switch tests. The residual of its projection, the one the corrector reports, is
2.58e-53.

## 5. Along the step

`probe_ns_D253.9368.log`, `crawl_probe_D253.9368.jsonl`.

**Arm U, never projected: smooth, no jump.** The kernel's residual along z + lam dz, with F(0) = 0.129:

| lam | max\|F\| | e = max\|F - (1 - lam) F(0)\| |
|---|---|---|
| 0.5 | 6.442e-02 | 1.002e-10 |
| 0.6 | 5.154e-02 | 9.623e-11 |
| 0.7 | 3.865e-02 | 8.420e-11 |
| 0.8 | 2.577e-02 | 6.415e-11 |
| 0.9 | 1.288e-02 | 3.609e-11 |
| 1.0 | 1.421e-17 | 1.421e-17 |

The largest rung-to-rung growth is 0.67, against K = 1e3. So the deviation is 4.0e-10 lam (1 - lam) plus the
1.42e-17 left at lam = 1. That is smooth to the ladder's resolution, and no row of the kernel's own residual jumps on
this step. The value ElimO's switch tests is this residual, falling as (1 - lam) 0.129. It crosses 1e-15 at
1 - lam = 7.8e-15, about 35 ulps of 1.

**Arm E, the corrector's own residual: affine, then a jump of 5e38 at lam* = 1 - 7.77e-15.** Here F(0) is the
residual of z's projection, 2.578e-53. That is the corrector's value: a point is projected when the value its switch
tests, in the last column, exceeds 1e-15.

| 1 - lam | max\|F\| | e | switch's test value | projected | s |
|---|---|---|---|---|---|
| 0.5 | 1.291e-53 | 4.03e-55 | 6.44e-02 | yes (u0q) | 290 |
| 0.4 | 1.035e-53 | 5.80e-55 | 5.15e-02 | yes | 312 |
| 0.3 | 7.783e-54 | 7.89e-55 | 3.87e-02 | yes | 570 |
| 0.2 | 5.220e-54 | 1.03e-54 | 2.58e-02 | yes | 281 |
| 0.1 | 2.659e-54 | 1.31e-54 | 1.29e-02 | yes | 350 |
| 0 (lam = 1) | 1.421e-17 | 1.42e-17 | 1.42e-17 | no (skipped) | 2 |
| 8.5487e-15 | 1.611e-54 | 1.61e-54 | 1.087e-15 | yes | 303 |
| 6.9944e-15 | 8.871e-16 | 8.87e-16 | 8.87e-16 | no | 2 |

Up to 1 - lam = 0.1 the deviation is the eliminated system's quadratic remainder, e = 1.61e-54 lam². Each rung
grows by (lam_b / lam_a)² to within 1%. Between 0.9 and 1 the growth is 8.8e36. The bracket comes from the hint: the
switch's test value is bisected in log(1 - lam), 14 evaluations without a projection, and crosses 1e-15 at
1 - lam = 7.7755e-15. F was evaluated at both ends, and the hint was taken. **lam* = 1 - 7.77e-15, in a bracket of
width 1.6e-15 (7 ulps of 1).** It lies within the bracket of the switch's crossing.

The deviation per block of rows on each side of lam* (block 5 is the horizon block, its fields p, m, a, c; blocks 0-4
are the nodes' matching rows):

| rows | below lam* (1 - lam 8.55e-15, projected) | above lam* (1 - lam 6.99e-15, not projected) |
|---|---|---|
| horizon c (5c) | 9.97e-55 | 8.87e-16 |
| horizon a (5a) | 6.4e-83 | 4.49e-16 |
| horizon m (5m) | 2.7e-80 | 1.73e-16 |
| horizon p (5p) | 1.50e-54 | 1.69e-17 |
| every other block (largest: 4p) | 1.61e-54 | 1.61e-54, unchanged |

**The rows carrying 99% of the jump are 44 rows of the horizon block: 19 c rows, 17 a rows and 4 m rows.** The largest
are c at grid points 30, 29, 28, 31 and 4, and a at 30 and 29. By harmonic, the jump in the c rows has norm 4.9e-16,
spread over the odd harmonics 5 to 29 (largest k 7, 19, 29 and 27, at 1.5e-16 to 1.9e-16 each). In the a rows it has
the same pattern at half the size (2.5e-16), and in the m rows 7e-17, flat across the harmonics. No row outside the
horizon block moves.

**The same point projected is back on the affine model.** At 1 - lam = 6.99e-15 with the projection forced, e is
1.611e-54, with the same per-block values as below lam*. The natural test at lam = 1 tells the same story:

| trial at lam = 1 | max\|F\| | simplified correction / \|dz\| (bound 0.75) |
|---|---|---|
| as the corrector evaluates it (not projected) | 1.421e-17 | 9.448e+16, refused |
| projected | 1.611e-54 | 2.384e-24, accepted |

The step formed again at the iterate equals the stored one bit for bit.

## 6. The branch

**The jump is ElimO's projection switch** (`hp/octo_corrector.py`, `ElimO.project`, lines 498-501):

```python
nf = qmax(self.sq.residual_q(*z4))
if nf <= self.switch:            # 1e-15
    self.counts["skipped"] += 1
    return z
```

The corrector's residual is F(P(x)) where the pinned quad-double max |F| of x exceeds 1e-15, and F(x) below that. The
crawl iterate z is never projected itself. Its own residual is 0.129, carried by its horizon data. Its step dz from
`ElimO.newton_o` is the projection's displacement P(z) - z plus the Newton step from P(z). Along z + lam dz the
unprojected residual is (1 - lam) 0.129 + 4e-10 lam (1 - lam) + lam² 1.42e-17 (arm U). So every trial with
1 - lam > 7.8e-15 is projected and lies on the eliminated system's affine model. The trials closer to 1, the full
step among them, are not projected and keep the horizon rows' second-order term, 1.42e-17 at lam = 1. J^-1 turns that
term into a correction 9.4e16 times |dz|.

This accounts for each feature of the crawl as #674 and #603 logged it, as an inference from the measurement above:

- lam = 1 is refused, and its max |F| falls by 4 per step (chain_ns: 1.42e-17, 3.55e-18, 8.88e-19, ...; #674 at
  251.886: 4.41e-23, 1.10e-23, 2.76e-24, ...). The unprojected term at lam = 1 is quadratic in the step, and the step
  halves. The ratio falls by 4 to 8 per step (#674: 5.8e7, 1.2e7, 2.6e6; chain_ns: 9.4e16, 1.2e16, 1.5e15).
- lam = 1/2 is accepted at a ratio of exactly 0.5000. The trial is projected, the eliminated system is affine along
  the step to 1.6e-54 lam², and the displacement half of the step is halved with it.
- max |F| and |dz| halve per step, and the iterate's own residual halves with them. The crawl therefore lasts at
  least until that residual falls under 1e-15, log2(0.129 / 1e-15) = 47 steps at chain_ns's iterate, past any step
  cap. It ended at the cap after 20 steps on win2.
- The lam = 1 trial is fast (about 0.4 s on win2) and the lam = 1/2 trial slow (35-48 s): only the second one is
  projected (ElimO's counts in the logs: 21 skipped, 23 u0q on chain_ns's attempt).

**Where the iterate's offset comes from (measured in part).** `chord_strong` steps zc by its corrections and evaluates
F(P(zc)), but never puts P(zc)'s horizon data into zc. Its first point z0 + dz0 had a switch's test value of 1.42e-17
and was not projected. From the second point on, the points were projected, so their own residual was above 1e-15
(ElimO's counts in the rebuild: 3 skipped, the start, the chord's first point and the lam = 1 trial; 5 u0q, the
chord's second and third points, its last point twice, and the lam = 1/2 trial).
The chord's last point, the crawl iterate, carries 0.129. How a chord step of |dw| 3.7e-37 raises the unprojected
residual from 1.4e-17 past 1e-15 is not measured here. The mass mode's gain, about 1e39 on a_h at this D, makes a
change of 1e-40 in a_h worth 0.1 in the horizon rows. That is a candidate, not a finding.

## 7. Candidates ruled out

- **The kernel's residual** (`hp/src/model.rs`). Arm U, the kernel's own residual along the same step, is smooth to the
  ladder's resolution (growth at most 0.67 against K = 1e3), and the jumping rows are smooth there too. On reading:
  the centre and horizon series end early (`TangentTail`) only in tangent lanes, never in the value lanes the residual
  uses. The horizon solves (`psolve`) take a fixed number of refinement steps (`OD_REFINE` = 4) through LUs factored at
  the double-double rounding of a_h and Delta. That rounding is piecewise constant in z, but the refined solve
  converges to the octo-double solution past the rounding's reach. The frozen p rows use a fixed projector (kfreeze
  3). Taylor orders and the plan are fixed during an attempt.
- **The plan's segment choice.** A prediction is planned again only when its double-double |F| is at most 1e-6. Here it
  is 8.7e7, so the plan is the base's stored one, fixed through the attempt.
- **The held offset (PinnedO, c8)** is a constant vector. It cannot depend on lam.
- **The projection's own branches** (`Reseeder.project`: u0, u0q, ladder, failed; `refine_o`'s refactoring rule). Every
  projected evaluation on the step was u0q, and the projected side is affine to 1.6e-54 lam² through lam*.
- **The first step and the chord** are not on this line. The probe's step is the corrector's step at the chord's point.

## 8. The fix, and its A/B on the crawl iterate

The switch dates from the double-double projection, which held every point at the double-double floor
(notes/RESEED.md 3.3, amendment 4). Under #603's flags the projection is the quad-double polish followed by three
octo-double corrections (`--reseed_q 1 --reseed_o 3`). Its rung residual reaches about 1e-80, below the corrector's
1e-60. So nothing is lost by projecting every point. The proposed change, as a diff (not applied; `hp/anm_run.py`,
`correct`):

```diff
             if elim.get("rs") is not None:
-                so = ElimO(so, elim["rs"])
+                # with --reseed_o the projection is octo-double, so every point is projected: below the switch a
+                # projected iterate's full step went unprojected and was refused for its horizon rows (#700)
+                so = ElimO(so, elim["rs"], switch=0.0 if opt["reseed_o"] else 1e-15)
```

**A/B on the crawl iterate** (`crawl_probe.py fixrun`, `fixrun_ns_D253.9368_switch0.log`). The corrector continues
from the saved iterate with its own code (`newton_od`, the natural test, ElimO around the attempt's PinnedO and
re-seeder). Only the switch is set to 0.

| | switch 1e-15 (as it runs) | switch 0 |
|---|---|---|
| step 1 | lam 1 refused (ratio 9.4e16), lam 1/2 accepted, 2.58e-53 → 1.29e-53 | lam 1 accepted (ratio 2.4e-24), 2.58e-53 → 1.61e-54 |
| step 2 | the same crawl (win2: 18 more steps, ending at 4.3e-59 at the cap) | lam 1 accepted (ratio 3.7e-24), 1.61e-54 → 9.82e-80 |
| result | refused | converged; projection max\|F\| 9.82e-80, the point itself 2.03e-65 |
| Delta | 1.684328705808 (win2's refused iterate, 12 digits) | 1.684328705807959325195082725310726557634 |

The switch-1e-15 column is the rebuild's step 1 and win2's log. The switch-0 run used 4 projections (u0q) and no
skips, in 2995 s on the Mac. No converged state of this D exists to compare Delta with to 40 digits, so this A/B
shows convergence, not the state's agreement with another route.

The cost is one projection, 34 s on win2, at each point the switch now skips. chain_ns's accepted attempts skipped 7
to 11 evaluations, so they would take about 4 to 6 min more each. Its crawling refusal cost 31 min. A cheaper rule,
not tested here, would project the trials of an iterate that was itself projected and keep the switch otherwise.

## 9. What is measured and what is inferred

Measured at chain_ns's attempt (D 253.8825 + 0.0543):

- the corrector's residual along its step at the crawl iterate, lam* = 1 - 7.77e-15, and the jump per block;
- that the kernel's own residual is smooth along the step;
- that the point above lam* is affine when projected;
- the natural test at lam = 1 both ways;
- the A/B.

Inferred, not measured:

- That #674's 251.886 crawl has the same cause. Its log (`I674-win2/251.886_armA.log`) shows the same signature as
  chain_ns's: each lam = 1 trial takes 0.35 to 0.43 s with max |F| 4.4e-23 to 4.2e-29, under 1e-15 and so not
  projected, and each lam = 1/2 trial takes 39 to 49 s, projected, at a ratio of 0.5000. It was not probed, and
  #603's older crawls (chain at a = 0.111, chain_nat at a = 0.0556) were not read for this signature.
- How the chord gives its point its horizon offset (section 6).
- That the default test (`accept="own"`) and the quad-double corrector (`ElimQ`, the same switch) can crawl the same
  way. Both evaluate trials through the same switch.
- The difference between the rebuild's chord |dw| and win2's (section 4).

## 10. Recommended tasks

1. **The switch-0 A/B on win2, at #603's crawling attempts.**
   - What: anm_run.py with the diff of section 8 (or an `--elim_switch` option), at chain_ns's a = 0.0543 from D
     253.8825, at #674's 251.886 arm A (a = 0.7275957614183426), and at the old chain's a = 0.111 from D 253.6411.
     Each also runs with switch 1e-15 as the control.
   - Why: one attempt converged here, on the Mac, from the crawl iterate; the chain needs the whole attempt on its
     own machine.
   - Acceptance (can fail): each switch-0 attempt converges to at most 1e-60 within 8 Newton steps, and each control
     crawls (lam 1/2 at ratio 0.5000). At 251.886, the state's Delta agrees with #603's state at D 252.6135 to 1e-40.
   - Depends on #700.
2. **The octo-double corrector's planted checks with switch 0.**
   - What: `octo_corrector.py planted` (D 200 and a state past D 250) and `wall` with ElimO's switch 0, and their
     controls `--mutate_offset qd` and `--mutate 10`.
   - Why: the switch was part of the corrector those checks passed.
   - Acceptance (can fail): planted converges below 1e-40 with Delta within 1e-30 of the quad-double corrector's, and
     both controls fail.
   - Depends on #700.
3. **How the strong-row chord moves its point's horizon data.**
   - What: at chain_ns's attempt, the switch's test value and P(zc) - zc (in m_h and a_h, by harmonic;
     `crawl_probe.py offset` is written for this) at every chord point.
   - Why: the chord took the iterate's own residual from 1.4e-17 to 0.129. With switch 0 that no longer stops the
     corrector, but a chord that drifts in a_h at the 1e-40 level is worth knowing about before it is trusted further.
   - Acceptance (can fail): the trace names the chord step and the unknowns that carry the offset. Otherwise it shows
     every chord point below 1e-15, which refutes section 6's account.
   - Depends on #700.

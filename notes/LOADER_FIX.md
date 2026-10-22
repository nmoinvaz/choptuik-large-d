# Loader fix: `dd_fits_mp.load_points` reads the doubles, not their decimal strings

**Summary.** The fix leaves Δ∞ and its bar as quoted: the 48-fit range is still 1.643893744–1.643893763, and the
proposed headline 1.643893746 ± 1.5×10⁻⁸ is unchanged at 10⁻⁹ (the centre moves by 3×10⁻¹¹). The cleaning keeps
D = 10.75 on the D ≤ 60.1 snapshot (25 dropped instead of 26), which changes the second digit of most held-out
statistics on that snapshot, one end of the D ≤ 60.1 limit range (…752 → …753), the `--mixed` sentence, and a few
cells of the candidate table. "823 cleaned points" and "27 dropped of 850" do not change.

## The fix and its check

NUMERICS_REVIEW F7: for `hp/data/delta_dd.csv`, `load_points()` did `mp.mpf(c[1]) + mp.mpf(c[2])`, which reads the
shortest-repr strings of hi and lo as decimals at 60 digits. The fix reads each as the double it denotes,
`mp.mpf(float(c[1])) + mp.mpf(float(c[2]))` (`hp/dd_fits_mp.py`, `load_points`); 60 digits hold hi + lo exactly. The
`--scan=` branch already read JSON floats and is unchanged.

- **Regression check:** `check_numbers.py`, check `loader_exact` (default mode, under 1 s). Every one of the 857 rows
  must load as exactly hi + lo, compared as fractions. As a built-in control, the old decimal-string parse must
  differ on at least one row.
- **Mutation control, done once:** with the old line restored in `load_points`, `python3 check_numbers.py --selftest`
  exits 1 with `FAIL CSV rows whose loaded value differs from the exact hi + lo: computed 857, quoted 0`. With the fix,
  41 of 41 checks pass and all 41 one-unit shifts are caught; `--full` passes 44 of 44.
- **Size of the parse error:** every row moves; max 2.12×10⁻¹⁶, rms 4.99×10⁻¹⁷; above D = 45, max 1.07×10⁻¹⁶
  (`hp/data/loader_fix/cleaning_compare.log`), as the review found.
- **Independent agreement:** with the fix, `dinf_budget.py` agrees with the review's separate exact-value code
  (`hp/review/rv2_fits_budget.log`: python-flint, monomial basis, own cleaning) in everything compared: every group's
  point count, and the range of each data snapshot and of the 48 fits to 11 digits. The two share only the CSV and
  the cleaning rule's definition.
- **No `check_numbers.py` constant is made wrong by the fix** (RANGE = 1.643893744, 1.643893763 still holds within
  0.44×10⁻⁹ and 0.13×10⁻⁹), so none was changed.

## What the fix changes in the cleaning (`dd_fits_mp.clean`)

| fit range | points | dropped, old parse | dropped, fixed | kept, old → fixed | difference |
|---|---|---|---|---|---|
| 10 ≤ D ≤ 60.1 (`--dmax=60.1`) | 756 | 26 | 25 | 730 → 731 | D = 10.75 now kept |
| 10 ≤ D ≤ 61.03 (all) | 850 | 27 | 27 | 823 → 823 | none |
| 10 ≤ D ≤ 59.8 (`dd_candidates.py --dmax=59.8`) | 729 | 26 | 26 | 703 → 703 | none |

So "823 cleaned points" and "27 dropped of 850" do **not** change. The expectation that they would become 26 and 824
is wrong. The review's own exact-value cleaning (`hp/review/rv2_misc.log`, item 2) also drops 27 on 10–61.03. The
D = 10.75 effect is on the D ≤ 60.1 snapshot only, where the README quotes "26 on today's file with `--dmax=60.1`".
The 731 in `dinf_budget.py`'s output ("D 10..60.1 clean … N 731") is the fixed count; the repository's code gave 730
before the fix.

## Commands rerun

Unless a directory is given, each was run from the repository root, at `nice -n 10`, one at a time. The 2-core
machine was shared with other agents (load average 7–24), so most times are 3–10 times the idle ones. Two container
restarts interrupted R8 and R11b, which were rerun. Logs are in `hp/data/loader_fix/`; `queue.log` has the start and end
times. The "old parse" runs (O…) use `hp/data/loader_fix/oldrun.py`, which runs the same script with only
`load_points` swapped back to the decimal-string parse. `attrib.py` (A…) separates the two effects of the fix: the
values, and the cleaning set.

| run | command (as documented) | quoted in | time |
|---|---|---|---|
| R1 / O1 | `python3 hp/dd_fits_mp.py 10 45 --clean --dmax=60.1` | README:610 | 267 / 830 s |
| R2 / O2 | `python3 hp/dd_fits_mp.py 10 37.5 --clean --dmax=60.1` | README:610 | 169 / 345 s |
| R3 / O3 | `python3 hp/dd_fits_mp.py 10 45 --clean --mixed --dmax=60.1` | README:631 | 166 / 468 s |
| R4 | `cd hp && python3 plot_heldout.py --dmax=60.1` (writes `hp/heldout.svg`) | README:1890 | 141 s |
| R5 | `python3 hp/dd_fits_mp.py 10 1e9 --clean --dmax=60.1` | README:636 | 1994 s |
| R6 | `python3 hp/dd_fits_mp.py 10 1e9 --clean` (the "all 823 cleaned points" numbers; the README gives no command) | README:633–637 | 4422 s |
| R7 | `./reproduce.sh` | README:664–668 | 1764 s |
| R8 | `python3 hp/dinf_budget.py`, through `dinf_budget_wrap.py` (same output, plus the grid rows at 20 digits in `R8_dinf_budget_rows.json`) | README:639 | 1803 s |
| R9 | `python3 check_numbers.py --full` | README:1934 | 696 s |
| R10 / O10 | `python3 hp/dd_candidates.py 45 --dmax=59.8` | README:365 | 1894 s / about 10 min |
| R11, R11b | `cd solver/eeg20 && python3 zero_fits.py`, `… zero_fits2.py` ("Scripts: solver/eeg20/") | README:492–498 | 142 s, under 10 min |
| X2 | `python3 hp/review/rv2_fits.py budget --decimal` (the review's code, emulating the old parse) | — | 49 s |
| A1, A2 | `python3 hp/data/loader_fix/attrib.py 45`, `… 37.5` | — | 642, 319 s |

**Not run:** `hp/planted_dinf.py`, which takes about 50 minutes on an idle core and several hours on the shared machine;
the coordinator asked to skip it. Its log and JSON are unchanged (see "Planted truth" below). Also not run:
`dd_fits_mp.py 10 60.1 --clean`, the likely source of README:543/611's "2.6×10⁻¹⁵", which quotes no command.

## Old → new, quoted numbers

"Old" is the quoted value. For every changed number it was also confirmed as the old code's output on today's file
(O-runs, X2, or notes/REPRODUCTION.md); where it was not, the note says so. "New" is the rerun with the fix. Line
numbers refer to the files at commit 7028a4b; `main.tex` is the combined draft `paper/main.tex`; `physics` is
`paper/physics/main.tex`. `paper/math/main.tex` and GAMMA.md quote no number that depends on the loader (the math paper
quotes only Δ∞ ± 1.5×10⁻⁸ and ω∞; see the last section).

### The held-out table (R1, R2)

README:618–622, physics:317–321, main.tex:307–311. Every old cell is reproduced exactly by O1/O2 (the old code on
today's file), so every change below is the fix.

| model | max residual, 10 ≤ D ≤ 45 | worst error, 45 < D ≤ 60.1 | fitted on D ≤ 37.5: worst error | limit |
|---|---|---|---|---|
| 1/D, degree 16 | 4.1×10⁻¹⁵ (=) | **5.2×10⁻¹³ → 5.6×10⁻¹³** | 4.2×10⁻¹¹ (=) | 1.64389375 (=) |
| 1/D, degree 18 | **4.0 → 4.1×10⁻¹⁵** | **1.8×10⁻¹² → 1.6×10⁻¹²** | **1.5×10⁻¹⁰ → 1.4×10⁻¹⁰** | 1.64389377 (=) |
| 1/(D−2), degree 18 | 4.0×10⁻¹⁵ (=) | **2.8×10⁻¹² → 3.1×10⁻¹²** | **7.9×10⁻¹¹ → 7.7×10⁻¹¹** | 1.64389377 (=) |
| (D−2)^−1/2, degree 20 | **3.9 → 4.0×10⁻¹⁵** | **2.0×10⁻¹¹ → 1.3×10⁻¹¹** | **3.9×10⁻¹⁰ → 6.6×10⁻¹⁰** | **1.636 → 1.638** |
| D^−1/2, degree 20 | **3.8 → 3.9×10⁻¹⁵** | **1.6×10⁻¹⁰ → 1.4×10⁻¹⁰** | **6.2×10⁻⁹ → 5.3×10⁻⁹** | **1.547 → 1.556** |

The fitted sets change too: 351 → 352 points on 10 ≤ D ≤ 45 and 278 → 279 on 10 ≤ D ≤ 37.5 (D = 10.75 kept); the held-out
counts (379, 452) are unchanged.

**Cause: D = 10.75, not the parse.** A1/A2 refit with each combination. For 1/D degree 16, held out to 60.1:

| | old point set (730) | fixed point set (731) |
|---|---|---|
| old values | 5.24×10⁻¹³ | 5.69×10⁻¹³ |
| fixed values | 5.22×10⁻¹³ | 5.64×10⁻¹³ |

The other rows behave the same way, for example 2.01 vs 1.97×10⁻¹¹ (values) against 2.0 vs 1.3×10⁻¹¹ (point set) for
(D−2)^−1/2. A single point at the low end of the fit range, 1×10⁻¹⁵ off its neighbours and on the cleaning threshold,
moves these statistics by up to 70% (3.9 → 6.6×10⁻¹⁰). So the second digit of every held-out error and every half-integer limit depends
on a threshold decision, not on the data's precision. The 1/D degree-16 cell moves by 8%.

### Other numbers

| quantity | quoted at | old | new | changed? |
|---|---|---|---|---|
| points `--clean` drops with `--dmax=60.1` | README:596 | 26 | **25** | yes (D = 10.75 kept) |
| "about 25 of them sit visibly off their neighbours" | README:595 | about 25 | 25 | no |
| 1/D series on D ≤ 45 predicts 15 units ahead to | README:625; physics:304; main.tex:31, 295 | 5×10⁻¹³ (5.2) | **5.6×10⁻¹³** | yes (5 → 6 at one figure) |
| half-integer series miss by … times more | README:626; physics:305; main.tex:31, 295–296 | 40 to 300 (ratios 38, 310) | **20 to 250** (23, 250) | yes |
| held-out to D = 61.03, 1/D degree 16 (`reproduce.sh`) | main.tex:31 ("to D = 61: 7×10⁻¹³") | 7.4×10⁻¹³ | 7.4×10⁻¹³ | no |
| half-integer limits, degree 20–24, "on all the points to D = 60" | README:627 | 1.636 to 1.694; 1.54 to 2.75 | to 60.1 (R5): 1.644 to 1.677; 1.642 to 2.63. To 61.03 (R6): 1.644 to 1.695; 1.642 to 2.41 | see note 1 |
| `--mixed`: prediction at D = 60 with one added term, degree 16 | README:629 | 1.6–5×10⁻¹³ (O3: 1.7–4.8) | **1.2–3.2×10⁻¹³** | yes |
| `--mixed`: limit with one added term, degree 16 | README:630 | 1.64389375 (O3: all five) | **1.643893736–1.643893754** | yes (note 2) |
| `--mixed`: D^−1/2 coefficient, degree 14 | README:630 | 3×10⁻⁵ | 3.2×10⁻⁵ | no |
| `--mixed`: D^−1/2 coefficient, degree 16 | README:631 | 1×10⁻⁷ (O3: 1.23×10⁻⁷) | **6.0×10⁻⁷** | yes |
| points fit smooth curves to about (degree 18–20, to 60.1) | README:60, 593; main.tex:28 | 4×10⁻¹⁵ | 4.3×10⁻¹⁵ | no |
| limits, 1/D 16–20 and 1/(D−2) 18–20, 10 ≤ D ≤ 60.1 | README:635–636; physics:337; main.tex:325 | 1.643893744 to 1.643893752 | **1.643893744 to 1.643893753** (1.64389374444 to 1.64389375273) | yes, last digit (note 3) |
| the same on all cleaned points to 61.03 | README:636–637; physics:337–338; main.tex:325–326 | 1.643893745 to 1.643893752 | 1.643893745 to 1.643893752 (1.64389374468 to 1.64389375212) | no |
| cleaned points to 61.03 | README:636; physics:338; main.tex:326 | 823 | 823 | no |
| residual of those fits | README:634 | 4.2×10⁻¹⁵ | 4.3×10⁻¹⁵ | not by the fix (note 4) |
| … 1/D degree 16; 1/(D−2) degree 18 | README:635 | 8.6×10⁻¹⁵; 1.5×10⁻¹⁴ | 8.6×10⁻¹⁵; 1.5×10⁻¹⁴ | no |
| 1/(D−2) degree 16 "leaves" | README:634 | 3×10⁻¹³ | 3.6×10⁻¹³ | not by the fix (note 4) |
| **48-fit range** | README:639; physics:339; main.tex:327; check_numbers.py RANGE | 1.643893744–1.643893763 | 1.643893744–1.643893763 | no |
| 48-fit range, 11 digits | (NUMERICS_REVIEW 4a, exact values) | 1.64389374439–1.64389376313 (X2, old parse) | 1.64389374444–1.64389376313 | +5×10⁻¹¹ at the low end |
| its centre | (coordinator's 1.6438937535) | 1.64389375376 (X2) | 1.64389375379 | +3×10⁻¹¹ |
| fitting on D ≤ 45 alone pins it to | README:640; physics:340; main.tex:327–328 | 2.5×10⁻⁷ | 2.5×10⁻⁷ (24 fits, 1.64389375338–1.64389399905) | no |
| quoted Δ∞ = 1.64389375 ± 1.5×10⁻⁸ | README:62, 136, 208, 299, 374, 639; physics:34, 84, 248, 342, 385; math:231 | | centre 1.643893754 | no |
| `reproduce.sh`: checks; Δ∞ from 5 fits | README:668; reproduce.py | 14; 1.643893745–1.643893752 | 14 passed, 0 failed; 1.643893745–1.643893752 | no |
| power-law approach to 0 misses by; analytic | README:492–493; physics:357–358 | 6×10⁻⁸; 7×10⁻¹³ | 6.3×10⁻⁸; 7.4×10⁻¹³ | no |
| zero-limit 1/ln D series: misses by; Δ at 10³⁰ | README:494; physics:359–360 | 1.3×10⁻¹¹; 0.88 | 1.3×10⁻¹¹; 0.884 | no |
| c/ln D term: c; limit | README:495; physics:360–361 | 4×10⁻⁷; 1.6438937 | 4.01×10⁻⁷; 1.64389369 | no |

Notes:

1. These quotes do not come from either run as now documented. With the fixed code, "to D = 60" gives the R5 values
   and "all the points" gives the R6 values. The old parse was not rerun for the half-integer models on all points (O5
   was dropped for CPU), so whether the old code reproduced them is not known. The sentence's point stands either way
   ("their limits also jump"); quote the run it names.
2. With a term added, the limits are 1.643893736 (D^−1/2), 1.643893743 (ln D / D), 1.643893746 (D^−3/2),
   1.643893747 (ln D / D²) and 1.643893754 (e^−D/4); with no term, 1.643893754. All lie inside the bar, but they no
   longer all round to 1.64389375 (the old code gave 1.64389375 for every term).
3. The upper end is 1/(D−2) degree 18 on the 731 points: 1.64389375273 with the fix, 1.643893752 before (X2, 730
   points). It changes because D = 10.75 is now in the fit.
4. Old and new agree here. The old parse gives 4.3×10⁻¹⁵ for 1/D degree 18 on the 823 points (4.305×10⁻¹⁵,
   `hp/review/rv2_misc.log`) and 3.6×10⁻¹³ for 1/(D−2) degree 16 (X2). The quotes were already off in the last
   digit; the 3×10⁻¹³ matches the D ≤ 60.1 snapshot (3.0–3.3×10⁻¹³).

### The candidate table (R10, O10)

README:365–382 (table 372–382), physics:369–395, main.tex:353–380. The README says the table was made on an earlier
snapshot, so O10 (old parse, today's file) separates data drift from the fix. The cleaning set is the same under both
parses, so R10 − O10 is the parse alone.

| quantity | quoted | old code, today's file (O10) | fixed (R10) | changed by the fix at the quoted precision? |
|---|---|---|---|---|
| held-out points 45–59.8 | 351 | 352 | 352 | no (drift: 351 → 352) |
| free fit, best held-out (degree 16) | 4.7×10⁻¹³ | 4.8×10⁻¹³ | 4.8×10⁻¹³ | no (drift) |
| 5/7 √G j₀,₁ | 6.2×10⁻¹¹ (130×) | 6.2×10⁻¹¹ (130×) | **6.3×10⁻¹¹** (132×) | yes, error 6.2 → 6.3 |
| 7/5 (ln 2)⁻² π^−1/2 | 1.1×10⁻¹⁰ (220×) | 1.1×10⁻¹⁰ (225×) | 1.1×10⁻¹⁰ (**227× → 230**) | ratio, at the rounding edge |
| 7/2 √(ln 2/π) | 1.3×10⁻¹⁰ (270×) | 1.3×10⁻¹⁰ (274×) | 1.3×10⁻¹⁰ (**277× → 280**) | yes, ratio |
| 9/8 √(j₂,₁/j₀,₁) | 1.4×10⁻¹⁰ (290×) | 1.4×10⁻¹⁰ (293×) | 1.4×10⁻¹⁰ (**296× → 300**) | yes, ratio |
| 10/9 √(j₀,₁/ln 3) | 5.4×10⁻¹¹ (110×) | 5.4×10⁻¹¹ (113×) | **5.3×10⁻¹¹** (111×) | yes, error 5.4 → 5.3 |
| π²/6 | 1.7×10⁻⁹ (3500×) | 1.7×10⁻⁹ (3540×) | 1.7×10⁻⁹ (3550×) | no |
| 2π/j₁,₁ | 7.0×10⁻⁹ (15000×) | 7.0×10⁻⁹ (14700×) | 7.0×10⁻⁹ (14800×) | no |
| √e | 8.1×10⁻⁹ (17000×) | 8.1×10⁻⁹ (17000×) | 8.1×10⁻⁹ (17000×) | no |
| φ | 4.4×10⁻⁸ (92000×) | 4.4×10⁻⁸ (91900×) | 4.4×10⁻⁸ (92200×) | no |
| "at least 100 times worse"; closest "1.3×10⁻⁵ … about 840 times its uncertainty" | README:351–352, 389; physics:371–372; main.tex:355 | | 111× minimum; 1.25×10⁻⁵ = 835 × 1.5×10⁻⁸ | no |
| free fits overfit at "about 5×10⁻¹¹ at degree 24–26" | README:388 | | free 7.5×10⁻¹¹ (24), 2.4×10⁻¹⁰ (26); forced best 5.3–6.3×10⁻¹¹ at 26 | no |

The "minus Δ∞" column is computed from the quoted Δ∞ and is unchanged (also with 1.643893746: the closest is
+1.25×10⁻⁵, about 835 bars).

### Planted truth (not rerun)

README:651 ("0.79–0.88×10⁻⁸"), README:652–654 (the missed junk point), physics:348–350 ("0.8–0.9×10⁻⁸"):
`hp/data/planted_dinf.log` and `.json` are the old loader's run and were left unchanged. What the fix touches there is the
base polynomial P, the degree-18 1/D fit to the real 823 cleaned points (the same set under both parses). Its limit moves
by 6×10⁻¹² (1.64389374468460 → 1.64389374467885, NUMERICS_REVIEW 4a); its residual moves from 4.305 to 4.338×10⁻¹⁵.
The planted data are P plus seeded noise, so the centre offsets should move by about 10⁻¹¹. Two things are untested:
those offsets, and whether the one junk point planted at the 10⁻¹⁵ cleaning threshold, missed in all six plants, is
still missed. That criterion sits on a cliff, as D = 10.75 shows for the real data.

## The headline Δ∞ = 1.643893746 ± 1.5×10⁻⁸: unchanged at 10⁻⁹

- 48-fit range, fixed: 1.64389374444–1.64389376313, centre **1.64389375379**, half-width 9.34×10⁻⁹
  (`R8_dinf_budget_fullprec.txt`). Old parse (X2): 1.64389374439–1.64389376313, centre 1.64389375376. The fix moves
  the centre by +3×10⁻¹¹.
- The planted offsets (old loader) are 0.785–0.877×10⁻⁸, the smallest in T1 and T4. Centre minus 0.785×10⁻⁸ is
  1.64389374594; minus 0.78×10⁻⁸ it is 1.64389374599. Both give **1.643893746**, about 4.5×10⁻¹⁰ from a rounding edge,
  15 times the fix's shift.
- The coordinator's centre 1.6438937535 is the midpoint of the 9-digit ends. The 11-digit centre is 1.64389375379,
  2.9×10⁻¹⁰ higher, which does not change the rounded value.
- Not caused by the fix, but follows from the new headline: ω∞ = 2π/1.643893746 = 3.82213590 (not 3.82213589). The bar
  is unchanged (3.5×10⁻⁸, quoted 4×10⁻⁸). `check_numbers.py`'s `dinf_quotes` check (the range centre rounds to DINF at
  10⁻⁸) and `omega_quotes` will need the new definition (centre minus bias) when the headline changes.

## Files written

- `hp/dd_fits_mp.py`: the fix. `check_numbers.py`: check `loader_exact`.
- `hp/heldout.svg`: regenerated by its documented command (R4). The committed file was stale before the fix: it was
  made on an earlier snapshot to D = 61.0 (279/517 and 352/444 points), matching neither the documented command nor
  today's file (notes/REPRODUCTION.md, fix 6). It now shows the D ≤ 60.1 snapshot (279/452 and 352/379 points), as
  README:609 says. `paper/figures/heldout.pdf` was **not** regenerated. physics:302 and main.tex:293 ("Fig. … uses all
  points to D = 61.0") describe the old PDF; if the PDF is rebuilt from the new SVG, that clause has to change.
- `hp/data/loader_fix/`: the logs above and the helper scripts (`oldrun.py`, `attrib.py`, `cleaning_compare.py`,
  `dinf_budget_wrap.py`, `budget_rows.py`).
- Not edited: `reproduce.py`, whose expected-value strings quote the README ("README: 5.2e-13 on the D <= 60.1
  snapshot"; that would become 5.6e-13); its pass criteria are unaffected. `hp/planted_dinf.py`'s docstring says "like
  the 27 points the real cleaning drops", which is still right.

## Proposed wording (for the coordinator)

README:
- 596: "… `hp/dd_fits_mp.py --clean` drops them (25 on today's file with `--dmax=60.1`)."
- 618–622: the new table above.
- 625–627: "Fitted on D ≤ 45, the 1/D series predicts Δ fifteen units of D ahead to 6×10⁻¹³, about 100 times the noise
  of the points, while the series in half-integer powers miss by 20 to 250 times more. Their limits also jump with the
  degree (1.644 to 1.677 for (D-2)^-1/2 and 1.64 to 2.63 for D^-1/2 at degree 20–24 on the points to D = 60.1), so
  they give no usable Δ∞."
- 628–631: "Adding one non-analytic term (…) to the degree-16 series does not degrade the prediction (1.2–3.2×10⁻¹³ at
  D = 60) and keeps the limit within the bar (1.643893736 to 1.643893754), and the term's coefficient is not
  determined: D^-1/2 gets 3×10⁻⁵ at degree 14 and 6×10⁻⁷ at degree 16."
- 633–636: "… (1/(D-2) at degree 16 leaves 4×10⁻¹³) fit the points from D = 10 to 61.03 to 4.3×10⁻¹⁵ (…) and give
  limits from 1.643893744 to 1.643893753 on 10 ≤ D ≤ 60.1 (…) and from 1.643893745 to 1.643893752 on all 823 cleaned
  points to D = 61.03." (The 4.3 and 4 corrections are not caused by the fix and are optional.)
- 368–370 and the candidate table: "compared with the 352 cleaned points from 45 to 59.8 … (4.8×10⁻¹³ at degree 16 on
  this range; held out to 60.1 the same fit gives 5.6×10⁻¹³ …)". In the table, take the R10 column (6.3×10⁻¹¹ (130×),
  1.1×10⁻¹⁰ (230×), 1.3×10⁻¹⁰ (280×), 1.4×10⁻¹⁰ (300×), 5.3×10⁻¹¹ (110×), 1.7×10⁻⁹ (3500×), 7.0×10⁻⁹ (15000×),
  8.1×10⁻⁹ (17000×), 4.4×10⁻⁸ (92000×), free 4.8×10⁻¹³). The alternative is to keep the table as the earlier snapshot
  it says it is.

Papers: the same changes at physics:304–305 ("to 6×10⁻¹³ … 20 to 250 times"), 317–321 (table), 337 ("…752" →
"…753"), 369 (352) and 385–395, and in main.tex:31, 295–296, 307–311, 325, 353 and its table.

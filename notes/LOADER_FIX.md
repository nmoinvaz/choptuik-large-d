# Loader fix: `dd_fits_mp.load_points` reads the doubles, not their decimal strings

DRAFT, being filled in as the reruns finish.

## The fix and its check

NUMERICS_REVIEW F7: for `hp/data/delta_dd.csv`, `load_points()` did `mp.mpf(c[1]) + mp.mpf(c[2])`, which reads the
shortest-repr strings of hi and lo as decimals at 60 digits. The fix reads each as the double it denotes,
`mp.mpf(float(c[1])) + mp.mpf(float(c[2]))` (`hp/dd_fits_mp.py`, `load_points`); 60 digits hold hi + lo exactly. The
`--scan=` branch already read JSON floats and is unchanged.

- **Regression check:** `check_numbers.py`, check `loader_exact` (runs in the default mode, under 1 s): every one of the
  857 rows must load as exactly hi + lo (compared as fractions), and, as a built-in control, the old decimal-string
  parse must differ on at least one row.
- **Mutation control, done once:** with the old line restored in `load_points`, `python3 check_numbers.py --selftest`
  exits 1 with `FAIL CSV rows whose loaded value differs from the exact hi + lo: computed 857, quoted 0`; with the fix,
  41 of 41 checks pass and all 41 one-unit shifts are caught.
- **Size of the parse error:** max 2.12×10⁻¹⁶, rms 4.99×10⁻¹⁷ over all 857 rows (every row moves); above D = 45 max
  1.07×10⁻¹⁶ (`hp/data/loader_fix/cleaning_compare.log`), as the review found.

## What the fix changes in the cleaning (`dd_fits_mp.clean`)

| fit range | points | dropped, old parse | dropped, fixed | kept, old → fixed | difference |
|---|---|---|---|---|---|
| 10 ≤ D ≤ 60.1 (`--dmax=60.1`) | 756 | 26 | 25 | 730 → 731 | D = 10.75 now kept |
| 10 ≤ D ≤ 61.03 (all) | 850 | 27 | 27 | 823 → 823 | none |
| 10 ≤ D ≤ 59.8 (`dd_candidates.py --dmax=59.8`) | 729 | 26 | 26 | 703 → 703 | none |

So "823 cleaned points" and "27 dropped of 850" do **not** change. The expectation that they become 26 and 824 is not
borne out: the review's own exact-value cleaning (`hp/review/rv2_misc.log`, item 2) also drops 27 on 10–61.03; the
D = 10.75 effect is on the D ≤ 60.1 snapshot only (26 → 25 dropped, 730 → 731 kept), where the README quotes "26 on
today's file with `--dmax=60.1`". The 731 in `dinf_budget.py`'s output ("D 10..60.1 clean ... N 731") is the fixed
count; the repository's code printed 730 before the fix (the review's exact-value budget already showed 731).

## Commands rerun

All from the repository root unless a directory is given, at `nice -n 10`, one at a time, on a 2-core machine shared
with other agents (load average 7–22 during the runs, so the times are 3–10 times the idle ones). Logs are in
`hp/data/loader_fix/` (`queue.log` has start and end times). "Old parse" runs use `hp/data/loader_fix/oldrun.py`, which
runs the same script with only `load_points` swapped back to the decimal-string parse.

| run | command (as documented) | quoted in | log | time |
|---|---|---|---|---|
| R1 | `python3 hp/dd_fits_mp.py 10 45 --clean --dmax=60.1` | README:610 | `R1_fits_10_45.log` | 267 s |
| R2 | `python3 hp/dd_fits_mp.py 10 37.5 --clean --dmax=60.1` | README:610 | `R2_fits_10_37.5.log` | 169 s |
| R3 | `python3 hp/dd_fits_mp.py 10 45 --clean --mixed --dmax=60.1` | README:631 | `R3_mixed.log` | 166 s |
| R4 | `cd hp && python3 plot_heldout.py --dmax=60.1` (writes `hp/heldout.svg`) | README:1890 | `R4_plot_heldout.log` | 141 s |
| R5 | `python3 hp/dd_fits_mp.py 10 1e9 --clean --dmax=60.1` | README:636 | `R5_fits_all_601.log` | 1994 s |
| R6 | `python3 hp/dd_fits_mp.py 10 1e9 --clean` (the "all 823 cleaned points" numbers; not spelled out in the README) | README:633–637 | `R6_fits_all_6103.log` | 4422 s |
| X2 | `python3 hp/review/rv2_fits.py budget --decimal` (the review's independent code, emulating the old parse) | — | `X2_rv2_budget_decimal.log` | 49 s |
| R7 | `./reproduce.sh` | README:664–668 | `R7_reproduce.log` | 1764 s |

## Old → new, quoted numbers

"Old" is the quoted value (and, where noted, the old code's output on today's file: the clean-room run in
notes/REPRODUCTION.md, the review's `--decimal` emulation X2, or the old-parse reruns). "New" is the rerun with the fix.
Line numbers are those of the files at commit 7028a4b. `paper/main.tex` (the combined draft) is listed where it repeats
a number.

### The held-out table (R1, R2): changes in the second digit

`hp/dd_fits_mp.py 10 45 --clean --dmax=60.1` and `10 37.5 …`: README:618–622, paper/physics/main.tex:317–321,
paper/main.tex:307–311. Old values of the first three columns are also the old code's output on today's file
(notes/REPRODUCTION.md, "matches every quoted cell").

| model | max residual, 10 ≤ D ≤ 45 | worst error, 45 < D ≤ 60.1 | fitted on D ≤ 37.5: worst error | limit |
|---|---|---|---|---|
| 1/D, degree 16 | 4.1×10⁻¹⁵ → 4.1×10⁻¹⁵ | **5.2×10⁻¹³ → 5.6×10⁻¹³** | 4.2×10⁻¹¹ → 4.2×10⁻¹¹ | 1.64389375 → 1.64389375 (1.643893754) |
| 1/D, degree 18 | **4.0 → 4.1×10⁻¹⁵** | **1.8×10⁻¹² → 1.6×10⁻¹²** | **1.5×10⁻¹⁰ → 1.4×10⁻¹⁰** | 1.64389377 → 1.64389377 (1.64389376509) |
| 1/(D−2), degree 18 | 4.0×10⁻¹⁵ → 4.0×10⁻¹⁵ | **2.8×10⁻¹² → 3.1×10⁻¹²** | **7.9×10⁻¹¹ → 7.7×10⁻¹¹** | 1.64389377 → 1.64389377 (1.64389377166) |
| (D−2)^−1/2, degree 20 | **3.9 → 4.0×10⁻¹⁵** | **2.0×10⁻¹¹ → 1.3×10⁻¹¹** | **3.9×10⁻¹⁰ → 6.6×10⁻¹⁰** | **1.636 → 1.638** |
| D^−1/2, degree 20 | **3.8 → 3.9×10⁻¹⁵** | **1.6×10⁻¹⁰ → 1.4×10⁻¹⁰** | **6.2×10⁻⁹ → 5.3×10⁻⁹** | **1.547 → 1.556** |

The fitted set changes too: 351 → 352 points on 10 ≤ D ≤ 45 (D = 10.75 kept), 379 held out (unchanged).

### Other numbers from R1–R7

| quantity | quoted at | old | new | changed? |
|---|---|---|---|---|
| points `--clean` drops with `--dmax=60.1` | README:596 | 26 | **25** | yes (D = 10.75 kept) |
| "about 25 of them sit visibly off their neighbours" | README:595 | about 25 | 25 | no |
| 1/D series on D ≤ 45 predicts 15 units ahead to | README:625, physics:304, main.tex:31, 295 | 5×10⁻¹³ (5.2) | **5.6×10⁻¹³** | yes, at one figure (5 → 6) |
| half-integer series miss by … times more | README:626, physics:305, main.tex:31, 295–296 | 40 to 300 (38, 310) | **20 to 250** (23, 250) | yes |
| held-out to D = 61.03, 1/D degree 16 (`reproduce.sh`) | main.tex:31 ("to D = 61: 7×10⁻¹³") | 7.4×10⁻¹³ (REPRODUCTION) | 7.4×10⁻¹³ | no |
| half-integer limits at degree 20–24 "on all the points to D = 60" | README:627 | 1.636 to 1.694; 1.54 to 2.75 | to 60.1: 1.644 to 1.677; 1.642 to 2.63. To 61.03: 1.644 to 1.695; 1.642 to 2.41 | see note 1 |
| `--mixed`: prediction at D = 60 with one added term, degree 16 | README:629 | 1.6–5×10⁻¹³ | **1.2–3.2×10⁻¹³** (5.6×10⁻¹³ with no term) | yes |
| `--mixed`: limit with one added term, degree 16 | README:630 | 1.64389375 | 1.643893736–1.643893754 | see note 2 |
| `--mixed`: D^−1/2 coefficient at degree 14 | README:630 | 3×10⁻⁵ | 3.2×10⁻⁵ | no |
| `--mixed`: D^−1/2 coefficient at degree 16 | README:631 | 1×10⁻⁷ | **6.0×10⁻⁷** | yes |
| smooth curves to about … (degree 18–20, to D = 60.1) | README:60, 593; main.tex:28 | 4×10⁻¹⁵ | 4.3×10⁻¹⁵ | no |
| limits, 1/D degree 16–20 and 1/(D−2) degree 18–20, 10 ≤ D ≤ 60.1 | README:635–636, physics:337, main.tex:325 | 1.643893744 to 1.643893752 | **1.643893744 to 1.643893753** (1.64389374444 to 1.64389375273) | yes, last digit |
| the same on all cleaned points to 61.03 | README:636–637, physics:337–338, main.tex:325–326 | 1.643893745 to 1.643893752 | 1.643893745 to 1.643893752 (1.64389374468 to 1.64389375212) | no |
| cleaned points to 61.03 | README:636, physics:338, main.tex:326 | 823 | 823 | no |
| residual of those fits | README:634 | 4.2×10⁻¹⁵ | 4.3×10⁻¹⁵ (1/D 18 and 1/(D−2) 20) | not by the fix (note 3) |
| … 1/D degree 16; 1/(D−2) degree 18 | README:635 | 8.6×10⁻¹⁵; 1.5×10⁻¹⁴ | 8.6×10⁻¹⁵; 1.5×10⁻¹⁴ | no |
| 1/(D−2) degree 16 "leaves" | README:634 | 3×10⁻¹³ | 3.6×10⁻¹³ (all points); 3.3×10⁻¹³ (to 60.1) | not by the fix (note 3) |
| `reproduce.sh` checks | README:668 | 14 | 14 passed, 0 failed | no |
| `reproduce.sh` Δ∞ from 5 fits | reproduce.py | 1.643893745–1.643893752 | 1.643893745 .. 1.643893752 | no |

Notes to the table:

1. Neither old quote reproduces from today's file with either loader as far as checked: the fixed code gives the
   values shown, and with the old parse the review code does not fit half-integer models. These limits jump with the
   degree by design of the argument ("their limits also jump"), so the sentence's point stands; the numbers should be
   replaced by the ones of the run the sentence names.
2. The limits with a term added are 1.643893736 (D^−1/2), 1.643893743 (ln D / D), 1.643893746 (D^−3/2),
   1.643893747 (ln D / D²), 1.643893754 (e^−D/4); without a term 1.643893754.
3. Old and new agree here: the old parse gives 4.3×10⁻¹⁵ for 1/D degree 18 on the 823 points (4.305×10⁻¹⁵,
   `hp/review/rv2_misc.log`) and 3.6×10⁻¹³ for 1/(D−2) degree 16 (X2). The quotes were already off in the last digit;
   the 3×10⁻¹³ matches the D ≤ 60.1 snapshot (3.0–3.3×10⁻¹³).

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

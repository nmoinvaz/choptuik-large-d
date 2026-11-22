# Reproduction test

This is a pass over README.md as an outside reader would make it, with a fresh clone, a venv built from SETUP.md and
every command the README quotes. For each command it asks three things:

- does the command run as written, from the stated directory?
- do the files it names exist?
- does it print the number the text quotes, to the digits quoted?

Scope is README.md only, not the papers or the other notes. Nothing in the repository was changed except this file and
the new `reproduce_readme.py` at the root (#349). Every fix below is a proposal. Discrepancies are reported and not
fixed, and no threshold was loosened to make a check pass.

The test ran on 2026-11-22 (03:33 to 09:10 UTC) from a fresh clone of https://github.com/nmoinvaz/choptuik-large-d at
**d7d54ccf** ("LARGE_D_WALL: the order-293 system's real singularity near D 265.078 lies on the chain's path (#966)",
2026-11-21). It replaces the record of 2026-10-21, which is in the git history.

## Environment

| | |
|---|---|
| Machine | dh170: Linux 7.2.11 x86_64 (Fedora 45), Intel Core i7-7700 @ 3.6 GHz (4 cores, 8 threads, AVX2 and FMA, no AVX-512), 31 GB |
| Load | 4.3 to 9.5 during every run, from other jobs on the machine (#1025's scans). Wall times are 1.5 to 3 times those of an idle machine, so only `reproduce.sh`, the one command with a quoted time, is compared with its time, and not as a failure. |
| Threads | 2 at `nice -n 19` for 107 of the 115 runs; 4 at nice 10 for 8 that started with the load under 4. One command at a time. |
| Python | 3.14.10 in a fresh venv built as SETUP.md says (`uv venv --seed --python 3.14`, `uv pip install -r requirements.txt`): numpy 2.5.4, scipy 1.18.1, mpmath 1.3.0, sympy 1.14.0, python-flint 0.9.0, maturin 1.16.0 |
| Builds | none. The runs forbade `hp/build.sh`, `pip install .` and cargo. The rows that need hpkernel or rustkernel ran in a second pass with an existing build's interpreter (`~/Source/choptuik-large-d-1058b/.venv`: Python 3.14.10, hpkernel from main 8179068d), installing nothing. Those rows are marked "existing build". |
| Raw output | `I349-dh170/` of choptuik-large-d-raw (commit 98d9044a): every command's log, the records, the tables, the control and the diffs of regenerated files |
| Raw clone | `../choptuik-large-d-raw`, cloned by HTTPS (4 min 41 s, 36879 files), read by the commands that take it as an argument |
| Stop rules | A command not finished in 30 min is stopped by its pid and listed "not finished in 30 min". A command the README gives as hours long is not started. The pass was capped at 10 h and used about 5 h 40 min. |
| Runs | `reproduce_readme.py`, 111 rows (115 runs counting adapted reruns) in a throwaway git worktree that was restored after each row, so no run saw the output of another |

## Summary

- **111 rows** from README.md: the file-name check and 110 command rows: 21 quote an invocation with arguments or a
  directory, 89 name a script as the source of a number or figure. Of the 111 (115 runs, counting adapted reruns; cpu
  9090 s, wall 12977 s in all):

  | Verdict | Rows | What it covers |
  |---|---|---|
  | OK | 68 | exit 0 (or the exit the README gives), files exist, every quoted number found. 9 of them regenerated a figure or table that differs from the committed one (fix 6, fix 11). |
  | FAIL | 32 | see below |
  | NOT RUN | 11 | 4 builds, 5 names with no invocation, 1 hours long, 1 not finished in 30 min (Commands skipped) |

  FAIL, by cause:

  | Cause | Rows |
  |---|---|
  | The script's own check or its exit status fails | 9: `reproduce.sh` (4 runs, fix 1), `formula_cut.py` (5), `singularity_type.py planted` (10), `check_numbers.py` and `--selftest` (13), `export_delta_all.py --verify` (6) |
  | A printed number disagrees with the README | 3: `planted_dinf.py` (2), `pert_check3.py 45 11` (7), `gamma_compare.py` (8) |
  | The command fails as quoted: a missing argument, directory, data or a NameError | 19 (fix 3, 4, 9); 14 of them run and match after an adapted rerun, 5 were not rerun |
  | File names in code spans | 1: 28 of 119 are abbreviated and 1 is a fragment (fix 15) |

- **The headline numbers mostly reproduce.**
  - 41 rows quote numbers: 203 numbers in all. 170 were found as quoted, 30 more after an adapted rerun with the
    argument or data the README lacks, and 3 were not found: the 0.79 and 0.88 of `planted_dinf.py` and the 2.2 of
    `gamma_compare.py`.
  - `./reproduce.sh` with hpkernel: 13 of 14 checks pass; `--full`: 14 of 15. The one failure is the held-out bar
    (fix 1). The README's check counts (14, 15) are right with hpkernel; without it they are 13 and 14.
  - `check_numbers.py`: 2260 of 2262 checks pass; `--selftest` finds that all 2262 reject a shifted value. The 2
    failures are the formula-refit items of SETUP.md's known issue #183.
  - The tables that `largeD_F`, `largeD_exponent`, `largeD_fast_transport`, `largeD_dingle_K`,
    `largeD_linear_late`, `largeD_Fratios`, `eeg20`, `tilt`, `maxc` and `dd_D5` print (3.22170194461806 against 3.22170)
    match to the digits quoted.
- **What disagrees with the text** (fixes 1, 2, 7, 8): the held-out bar of `reproduce.sh`; the centre offsets of
  `planted_dinf.py` (0.84–0.94×10⁻⁸ against 0.79–0.88×10⁻⁸); `pert_check3.py` (10⁻¹⁷⁸ at j = 11, against 10⁻¹⁸⁰); the
  standard error of `gamma_compare.py` (2.4×10⁻⁶ against 2.2×10⁻⁶).
- **What does not run as written** (fixes 3, 4, 9): 19 commands, including a NameError in `formula_all.py` that a
  one-line patch removes.
- **Links:** every relative link and anchor resolves; 7 footnotes are defined and never referenced (fix 12).
- **The controls pass** (How to rerun this, step 6).

## Prioritised fixes

Each fix is a proposal. "Acceptance" is a check that can fail.

1. **`./reproduce.sh` exits 1: its held-out check fails on data it was not written for** (README 2170, 2179).
   - What happens: from the clean venv, 13 checks run: 11 pass, 1 fails, 1 skips (the checks on saved solutions need
     hpkernel). The failure: `held-out error of the 1/D series (degree 16): got 3.2e-12 (expected <= 3e-12; README:
     5.6e-13 on the D <= 60.1 snapshot)`.
   - Cause: `reproduce.py` section 3 is titled "fit on D <= 45, predict 45 < D <= 61.03", but its code is
     `hold = [d for d in Dc if d > 45]`, with no upper bound, and the data now run to D = 67.77 (1199 points; the
     README's 857 points end at 61.03). The "points and range" check passes with "got 1199 points, D = 8.25 .. 67.7671
     (expected 857 points, D = 8.25 .. 61.0264)" because it tests `>=`.
   - Count and time: the README says "Its 14 checks" and "`--full` adds the Proposition 4 check". Without hpkernel the
     runs print 13 and 14 checks. With the existing hpkernel build (`reproduce_sh_hp`, `reproduce_sh_full_hp`) they
     print 14 and 15 checks, as the README says: 13 passed and 1 failed, then 14 passed and 1 failed (506 s and 474 s,
     cpu 447 s and 443 s). Without hpkernel the two checks on saved solutions are one SKIP. The README's "about 6
     minutes on a busy 2-core machine" against 1037 s (cpu 444 s) at load 8.7; not counted as a failure.
   - Proposed fix: `hold = [d for d in Dc if 45 < d <= 61.03]` in `reproduce.py`, as its section title says. Tested
     read-only on the committed data (`evidence/heldout_cap.log`): fit on D ≤ 45 and predict the cleaned points above it
     gives 3.2e-12 on 813 points (what `reproduce.py` does), 7.4e-13 on 471 points with the cap at 61.03, and 5.6e-13 on
     379 points with the cap at 60.1, which is the README's number.
   - Acceptance: `./reproduce.sh` exits 0 on the committed data, and a copy whose held-out points carry a planted
     1×10⁻¹¹ offset exits 1.
2. **`hp/planted_dinf.py`: the range's centre sits 0.84–0.94×10⁻⁸ above the planted limit, not 0.79–0.88×10⁻⁸**
   (README 872–880).
   - What happens: six plants of 48 fits each. Centre offsets 8.40, 8.41, 9.36, 8.40, 8.55 and 8.44 ×10⁻⁹. The rest
     of the paragraph reproduces: the planted limit is inside the range in all six, the offset is inside the 1.5×10⁻⁸
     bar, the cleaning misses one of the 20 junk points in all six (`junk caught False (20)`), and the script exits 1
     ("0 of 5 graded plants passed").
   - Evidence: two complete runs (06:00 UTC, load 5.2, 1190 s each) printed the same numbers. A rerun at load 8.9 was
     stopped at 30 min after T5 (1614 s); its T1 to T5 equal the complete runs'. The committed log
     `hp/data/planted_dinf.log` was last changed on 2026-10-21; `hp/data/delta_dd.csv` changed on 2026-11-13, -14, -16
     and -19 (the likely cause: the plants' base fit is "a degree-18 1/D fit to the real cleaned points"). The committed
     log reads 7.85–8.77×10⁻⁹, which is the README's 0.79–0.88×10⁻⁸ rounded. The table's row for this command is
     therefore the first complete run's, with its log spliced from the rerun's T1–T5 and the first run's T6 and last
     line (`planted_dinf.log` says so on its last line; the rerun's own log is `planted_dinf.redo.log`).
   - Proposed fix: rerun on an idle machine (about 20 min), commit the log, and change the README's "0.79–0.88×10⁻⁸" to
     the log's range. The README calls that offset "the correction applied above": the owner should check whether the
     quoted Δ∞ and its range move with it.
   - Acceptance: the README's range equals the log's range at the quoted digits, and
     `reproduce_readme.py --only planted_dinf` says OK; the same run on a README copy with "0.79" changed to "0.80"
     says FAIL.
3. **Commands quoted without the arguments or directory they need** (14 rows run after an adapted rerun, 5 not rerun).

   | Command (README line) | What it lacks | Adapted rerun |
   |---|---|---|
   | `hp/plot_solver_limits.py` (757), `hp/plot_echo.py` (251), `hp/plot_wave_family.py` (978) | the raw clone: `../../choptuik-large-d-raw` | exit 0 (plot_wave_family: existing build) |
   | `hp/plot_dropoffs.py` (2038) | `../../choptuik-large-d-raw/T76` | exit 0 |
   | `hp/track_c.py` (567), `hp/track_c_fold.py` (2014) | a subcommand: `real`, `run` | exit 0 |
   | `arg_crossing.py` (1587) | a scan folder: `scan13b` | exit 0 |
   | `solver/data_all.py` (657), `solver/rational_fit.py` (299), `solver/plot_extrapolation.py` (499) | to run from `solver/data`, as `python3 ../x.py` | exit 0; `data_all` prints the quoted D = 5 value |
   | `hp/compare/cmp_ecker.py` (660), `hp/lowD_formula.py` (390) | the Ecker data clone as the argument (fix 9) | exit 0; the numbers match |
   | `hier.py --schur` (1436), `amp_flint_par.py --fit phys` (1586) | M, A and a frequency; a frequency, K and J (hours) | not rerun |
   | `hp/gamma_dd.py` (457), `hp/modecount_test.py` (602), `hp/gamma_m48_gap.py` (461) | an output directory; a state path; a subcommand and states | not rerun |

   - Proposed fix: quote each full command line, as its script's docstring has it.
   - Acceptance: each quoted command exits 0 from the stated directory, and its row in this table has no "Adapted"
     clause.
4. **`solver/formula_all.py` and `formula_all.py --refit` raise NameError** (README 299, 312).
   - What happens: `NameError: name 'pd' is not defined` at line 100, in `main()`, where the error table uses `pd, qd,
     pg, qg`. Since 90eeef5c (2026-11-03) they are no longer set in `main()`. The README gives the path from the root
     (`solver/formula_all.py`), but the script reads `data/` and runs from `solver/data`.
   - Evidence: both exit 1. With this line added after `c, rec = coefficients()` in a scratch copy, both exit 0 and the
     quoted numbers print and agree (D = 4: Δ = 3.4454524, γ = 0.3739610):
     `pd, qd, pg, qg = c["a"], c["b"], c["e"], c["f"]`. `check_numbers.py` uses `refit_distance()`, not `main()`, so it
     does not hit this.
   - Proposed fix: that line, and "`cd solver/data && python3 ../formula_all.py`" in the README.
   - Acceptance: both commands exit 0 from `solver/data` on an unpatched clone and print the D = 4 numbers.
5. **`hp/formula_cut.py` exits 1: its own check (1) fails** (README 335).
   - What happens: `(1) 1235 points: largest |model - point| / sigma 521.30 at D 6.25, against 3.5: FAIL`. Checks (2)
     (series = integral to 10⁻⁵⁹ at 8 values of D) and (3) (Δ∞ = 1.64389374520412) pass. The README says the script
     "evaluates both forms and checks them".
   - The script's own breakdown puts the largest model-point differences at 5×10⁻¹⁰ (D 3–3.1), 1.2×10⁻¹⁰ (D 4.01–8.3)
     and below 6×10⁻¹⁵ above D 8.3, and lists "the unresolved M 48 rows 4.25–8.2 (not fitted, accuracy unmeasured)".
   - Proposed fix: the owner decides whether the σ column of those rows or the model is stale. Not tested.
   - Acceptance: `python3 hp/formula_cut.py` exits 0, or the README says it exits 1 and why; a planted 10σ offset at one
     resolved point flips check (1) to FAIL.
6. **`export_delta_all.py` rebuilds a different `delta_all.csv`, and `--verify` fails** (README 548).
   - What happens: `cd hp && python3 export_delta_all.py ../../choptuik-large-d-raw` exits 0 and writes a file of 2431
     rows against the committed 2512 (185 lines added, 266 removed). `--verify` on the rebuilt file exits 1 with 20
     `DIFFERS` lines: `chain_fit` rows from D 258.685 "no series value to check its err against", and `delta_series.csv`
     "fitted on other rows" (input 165feeccec218c45 against the table's f8d4438d73e6c796: "rerun track_c.py csv").
     On the committed `delta_all.csv`, `--verify` also exits 1: 2289 of 2512 rows equal their sources, 207 rows of
     `delta_fits.csv` are rebuilt, and 223 lines say DIFFERS (89 of them `delta_fits.csv`: "no chain_fit row has it")
     (`evidence/export_verify_committed.log`).
   - Proposed fix: rerun `track_c.py csv`, then the exporter, and commit the regenerated tables; or state in the README
     that the exporter's output depends on the raw clone's state. Not tested.
   - Acceptance: `export_delta_all.py ... --verify` exits 0 on a clean clone with the committed tables, and the
     rebuilt `delta_all.csv` has no diff against the committed one.
7. **`pert_check3.py 45 11` agrees to 10⁻¹⁷⁸, not 10⁻¹⁸⁰, at j = 11** (README 1836).
   - What happens: it prints the worst relative deviation per harmonic: 1.3×10⁻¹⁸⁰ (j = 1), 1.2, 2.3, 3.7, 7.1×10⁻¹⁸⁰,
     1.0×10⁻¹⁷⁹ (j = 6), 1.5, 2.1×10⁻¹⁷⁹, 9.9×10⁻¹⁷⁹ (j = 9), 2.8×10⁻¹⁷⁹, 2.1×10⁻¹⁷⁸ (j = 11). The README says "to
     10⁻¹⁸⁰ relative in every harmonic ... through j = 11". 2 of the 11 harmonics round to 10⁻¹⁸⁰.
     The 2026-10-21 record scored the same output "ok".
   - Proposed fix: "to 10⁻¹⁷⁸ relative in every harmonic through j = 11 (10⁻¹⁸⁰ at j = 1)".
   - Acceptance: the quoted exponent equals the exponent of the printed worst deviation, and the row is OK.
   - Also in this family: `pert_check2.py` prints 1.5×10⁻¹⁷⁷ against "10⁻¹⁷⁷" (agrees by truncation only), and
     `pert_check.py` prints 6.45×10⁻³⁰ against "6×10⁻³⁰".
8. **`gamma_compare.py`: the standard error of the last λ∞ − 2 is 2.4×10⁻⁶, not 2.2×10⁻⁶** (README 461–466).
   - What happens: with the arguments the README gives, D ≥ 30 prints `lam_inf - 2 = +1.57e-06 +- 2.4e-06` for eight
     powers; the README quotes (1.6 ± 2.2)×10⁻⁶. The other four values (2.4×10⁻⁴, 6.7×10⁻⁵, 1.9×10⁻⁵, 5.5×10⁻⁶)
     and a₁ = 3.0324 agree. The row needs hpkernel (existing build).
   - Proposed fix: "(1.6 ± 2.4)×10⁻⁶". The data have grown since the sentence was written (the file now has points to
     D 67.77), so the other quoted digits are worth a recheck.
   - Acceptance: the quoted ± equals the printed one at 2 digits.
9. **The Ecker data clone is not documented** (README 660, 390).
   - What happens: `hp/compare/cmp_ecker.py` as quoted raises `FileNotFoundError: ... 'ecc/complete_data/FortranData/'`.
     `hp/lowD_formula.py` needs the same. With
     `git clone https://github.com/EckerChristian/CriticalSpacetimeCrystal ../ecc` (f002d0a, 2026-04-23; its content was
     read as data and nothing in it was run) both run with it as the argument and agree with the quoted numbers
     (cmp_ecker: −1.66×10⁻⁴ and 2.7×10⁻⁷; lowD_formula: the polynomial coefficients).
   - Proposed fix: put the clone step next to each of the two commands, as SETUP.md does for the raw clone.
   - Acceptance: following the README text alone, both commands exit 0 from a clean clone.
10. **`hp/singularity_type.py planted` exits 1** (README 2006).
    - What happens: the clean plants pass and the contaminated S and P plants fail the 10% bar (`S contaminated mu +0.5:
      ... off (unclassified); ... FAIL`; `P contaminated mu -1.0: ... FAIL`), so `planted: FAIL`, exit 1. The script's
      docstring has said so since 2026-11-14. The README says only that an isolated planted square root is returned
      (it is: sqrt exponent 0.5).
    - Proposed fix: one README sentence saying that the contaminated plants fail the bar by design and the script
      exits 1.
    - Acceptance: the README states the exit status, and `reproduce_readme.py` runs the row with `expect_rc=1` and OK.
11. **Figures that do not show the committed data** (README 246, 412, 580 and others).
    - What happens: of the 17 SVGs that commands regenerate, 8 differ from the committed file.
      - `solver/delta.svg`, `hp/lowD_limits.svg` and `hp/lowD_logslope.svg`: the regenerated figures have 3 or 4 more
        points at D 3.0431–3.0438. The committed delta.svg's tooltip reads "D = 3.045 to 8.25 ... the largest error is
        4.1e-10, at D = 3.05625"; the regenerated one reads "D = 3.04315 to 8.25 ... 5.2e-09, at D = 4.25". The tables
        changed on 2026-11-16 and 2026-11-19 (#945).
      - `hp/lowD_limits.svg` also stamps the machine (committed: "Apple M5, 32 GB, macOS 27.0.1"; regenerated: "Intel(R)
        Core(TM) i7-7700 CPU @ 3.60GHz") and drops a source line, `T100-win2/down25e.jsonl`. It cannot regenerate
        byte-identically on another machine.
      - `solver/physics.svg`, `spectrum.svg`, `first_order.svg`, `gamma.svg` and `hp/modecount.svg` differ in one rule
        of the dark-mode `<style>`: the regenerated files have
        `[fill="#f5f4f0"]{fill:#232221}[stroke="#f5f4f0"]{stroke:#232221}` and the committed ones do not.
    - The other 9 SVGs regenerate byte-identically. `hp/data/delta_all.csv` is in fix 6.
    - Proposed fix: regenerate and commit the 8 figures, and leave the machine stamp out of `lowD_limits.svg` or say
      that it is expected to differ.
    - Acceptance: `reproduce_readme.py` prints "identical" for the 17 SVGs on a clean clone (the machine stamp
      excepted); the diffs are in `svgdiff/diffs/` of the raw folder.
12. **Footnotes** (README, end of file).
    - What happens: 7 footnotes are defined and never referenced ([^15], [^16], [^18] to [^22]); [^17] is neither
      defined nor referenced. 16 of 23 definitions are referenced.
    - Proposed fix: reference them where they were meant to be cited, or remove them, and renumber.
    - Acceptance: `python3 reproduce_readme.py --links` prints 0 broken.
13. **`check_numbers.py` exits 1 (2 of 2262 checks fail), as SETUP.md's known issue #183 says** (README 2160).
    - What happens: the two failing checks are `refit within 1e-9 of the stored Delta on 4-30.2` and `refit - stored
      Delta(80) between 0.5e-7 and 2e-7 in size`, both "computed 0.0, quoted 1" (existing build; 1 of the 3 refit items
      passes). `check_numbers.py --selftest` prints "2262 of 2262 reject a shifted value" and exits 1 for the same
      reason. The README says the file "ties more than 800 quoted numbers to their source files".
    - Proposed fix: none beyond #183; the README sentence could name the known failure.
    - Acceptance: both commands exit 0 on a second machine, or the README states the exit status.
14. **Build commands were not run** (README 2197–2200). `pip install .`, `cargo test --release`,
    `python tests/test_bindings.py` and `./build.sh main` need builds that this pass could not make, and no existing
    venv on dh170 has rustkernel. A clean-room pass that may build should run them and `hp/build.sh`.
15. **Abbreviated file names** (28 of 119). The README names `amp_flint.py`, `scan15phys/` and others by their
    basename where the sentence's directory gives the rest. `_J1000_phys.json` (line 1595) is a fragment.
    Low priority; the cost is a reader who searches for the name.


## Commands checked

One row per command or per script the README names as the source of a number or figure. "From" is the directory the
command ran in, relative to the repository root. "Number printed vs quoted" counts the quoted numbers that a printed
number agrees with, to the digits quoted (rounded or truncated: 3.46677179 and 3.46677169 do not agree at 6 digits).
`n/m` is m quoted numbers, n of them found. "Regenerated" rows rewrote a committed figure or table and the new file
differs from the committed one; the verdict is still the verdict of the three checks. A row with "Adapted" ran the
command a second time with the argument, directory or data it lacked; the first run's verdict stands.

### Whole README

| id | command | from | exit | files | number printed vs quoted | wall | verdict |
|---|---|---|---|---|---|---|---|
| files_named | `every file name in a code span` | . | 0 | none named | none quoted | 0 s | FAIL: 119 file names in code spans: 89 exist as written, 28 found elsewhere in the repository or the raw clone, 1 patterns, 1 not found / I200-mac... (fix 15) |

### Main results

| id | command | from | exit | files | number printed vs quoted | wall | verdict |
|---|---|---|---|---|---|---|---|
| largeD_formal_check | `solver/largeD_formal_check.py` | . | 0 | none named | none quoted | 39 s | OK |
| largeD_transport_sympy | `solver/largeD_transport_sympy.py` | . | 0 | none named | none quoted | 18 s | OK |
| heat_form | `solver/inner_pert/heat_form.py` | . | 0 | none named | none quoted | 2 s | OK |
| fixed_point | `fixed_point.py` | . |  | none named | none quoted |  | NOT RUN: named as the code of certified lemmas; the README quotes no invocation and no number (PROOF.md and proof/A_certify.md hold the commands: out... |
| fixed_point_M | `fixed_point_M.py` | . |  | none named | none quoted |  | NOT RUN: named as the code of certified lemmas; the README quotes no invocation and no number (PROOF.md and proof/A_certify.md hold the commands: out... |
| validated_int | `validated_int.py` | . |  | none named | none quoted |  | NOT RUN: named as the code of certified lemmas; the README quotes no invocation and no number (PROOF.md and proof/A_certify.md hold the commands: out... |
| krawczyk | `krawczyk.py` | . |  | none named | none quoted |  | NOT RUN: named as the code of certified lemmas; the README quotes no invocation and no number (PROOF.md and proof/A_certify.md hold the commands: out... |
| fast_eigenvalues | `solver/proof/fast_eigenvalues.py` | . | 0 | none named | none quoted | 3 s | OK |

### Results

| id | command | from | exit | files | number printed vs quoted | wall | verdict |
|---|---|---|---|---|---|---|---|
| plot_delta | `cd solver/data && python3 ../plot_delta.py --raw ../../../choptuik-large-d-raw` | solver/data | 0 | all exist | none quoted | 4 s | OK; the regenerated file differs from the committed one: solver/delta.svg (fix 11) |
| plot_echo | `hp/plot_echo.py` | hp | 2 | all exist | none quoted | 4 s | FAIL: exit 2. Adapted `python3 plot_echo.py ../../choptuik-large-d-raw` from hp/: exit 0, OK (fix 3) |
| formula_all | `solver/formula_all.py` | solver/data | 1 | all exist | D = 4: 0/2 MISMATCH | 2 s | FAIL: exit 1; D = 4: not printed: 3.4454524 (nearest printed 2, 0 digits agree); 0.3739610 (nearest printed 0.500000, 0 digits agree). Adapted `python3 ../formula_all.py` from solver/data/: exit 0, OK (one-line patch in a scratch copy) (fix 4) |
| rational_fit | `solver/rational_fit.py` | solver | 1 | none named | none quoted | 5 s | FAIL: exit 1. Adapted `python3 ../rational_fit.py` from solver/data/: exit 0, OK (fix 3) |
| formula_refit | `python3 ../formula_all.py --refit` | solver/data | 1 | none named | none quoted | 2 s | FAIL: exit 1. Adapted `python3 ../formula_all.py --refit` from solver/data/: exit 0, OK (one-line patch in a scratch copy) (fix 4) |
| formula_cut_fit | `hp/formula_cut_fit.py` | . | 0 | all exist | limit: 1/1 | 62 s | OK |
| formula_cut | `hp/formula_cut.py` | hp | 1 | none named | none quoted | 95 s | FAIL: exit 1 (fix 5) |
| lowD_max | `cd hp && python3 lowD_max.py` | hp | 0 | all exist | maximum: 2/2 | 1 s | OK |
| export_lowD | `hp/export_lowD.py` | . | 0 | none named | none quoted | 1 s | OK |
| export_dd | `hp/export_dd.py` | . | 0 | none named | none quoted | 1 s | OK |
| plot_lowD_values | `hp/plot_lowD_values.py` | . | 0 | all exist | none quoted | 5 s | OK |
| lowD_formula | `hp/lowD_formula.py` | hp | 1 | none named | delta poly: 0/9 MISMATCH; gamma poly: 0/7 MISMATCH | 4 s | FAIL: exit 1; delta poly: not printed: 1.752403 (nearest printed 2, 0 digits agree); 2.486429 (nearest printed 2, 1 digits agree); 8.257623 (neare.... Adapted `python3 lowD_formula.py ../ecc` from hp/: exit 0, OK (fix 9) |
| plot_lowD_limits | `hp/plot_lowD_limits.py` | . | 0 | all exist | none quoted | 5 s | OK; the regenerated file differs from the committed one: hp/lowD_limits.svg (fix 11) |
| plot_lowD_logslope | `hp/plot_lowD_logslope.py` | . | 0 | all exist | none quoted | 4 s | OK; the regenerated file differs from the committed one: hp/lowD_logslope.svg (fix 11) |
| lowD_logslope | `hp/lowD_logslope.py` | . | 0 | none named | prediction: 2/2 | 3 s | OK |
| gamma_dd | `hp/gamma_dd.py` | . | 1 | all exist | none quoted | 1 s | FAIL: exit 1 (existing build) (fix 3) |
| gamma_m48_gap | `hp/gamma_m48_gap.py` | . | 1 | none named | none quoted | 1 s | FAIL: exit 1 (existing build) (fix 3) |
| gamma_compare | `cd hp && python3 gamma_compare.py data/gamma_dd.jsonl data/local/gamma_dd.jsonl` | hp | 0 | all exist | lambda-2 by powers: 5/6 MISMATCH; lambda a1: 1/1; gamma a1 = lambda a1 / 4 (derived): 1/1; lambda a2 (not printed by this script): not checked | 1 s | FAIL: lambda-2 by powers: not printed: 2.2×10⁻⁶ (nearest printed 0.0000021, 1 digits agree) (existing build) (fix 8) |
| plot_extrapolation | `solver/plot_extrapolation.py` | solver | 1 | all exist | none quoted | 21 s | FAIL: exit 1. Adapted `python3 ../plot_extrapolation.py` from solver/data/: exit 0, OK (fix 3) |
| export_delta_all | `cd hp && python3 export_delta_all.py ../../choptuik-large-d-raw` | hp | 0 | all exist | none quoted | 4 s | OK; the regenerated file differs from the committed one: hp/data/delta_all.csv (fix 6) |
| export_delta_all_verify | `--verify` | hp | 1 | none named | none quoted | 48 s | FAIL: exit 1 (fix 6) |
| values_table | `cd hp && python3 values_table.py` | hp | 0 | none named | D = 4: 2/2; gamma(70): 1/1; gamma(100): 1/1; gamma(1000): 1/1 | 3 s | OK |
| track_c | `hp/track_c.py` | hp | 1 | none named | series: 0/6 MISMATCH; limit: 0/2 MISMATCH | 3 s | FAIL: exit 1; series: not printed: 1.684955205950 (nearest printed 2, 1 digits agree); 8.1×10⁻¹¹ (nearest printed -, 0 digits agree); 1.6781710154.... Adapted `python3 track_c.py real` from hp/: exit 0, OK (fix 3) |
| plot_physics | `solver/plot_physics.py` | . | 0 | all exist | none quoted | 2 s | OK; the regenerated file differs from the committed one: solver/physics.svg (fix 11) |
| modecount_test | `hp/modecount_test.py` | . | 1 | none named | none quoted | 1 s | FAIL: exit 1 (existing build) (fix 3) |
| dd_D5 | `hp/compare/dd_D5.py` | . | 0 | all exist | D = 5: 2/2 | 246 s | OK (existing build) |
| data_all | `solver/data_all.py` | solver | 1 | none named | D = 5 f64: 0/1 MISMATCH | 19 s | FAIL: exit 1; D = 5 f64: not printed: 3.221701945 (nearest printed 2, 0 digits agree). Adapted `python3 ../data_all.py` from solver/data/: exit 0, OK (fix 3) |
| cmp_ecker | `hp/compare/cmp_ecker.py` | hp/compare | 1 | none named | moves: 0/1 MISMATCH; offset: 0/2 MISMATCH | 1 s | FAIL: exit 1; moves: not printed: -1.66×10⁻⁴ (nearest printed -, 0 digits agree); offset: not printed: 2.7×10⁻⁷ (nearest printed -, 0 digits agree.... Adapted `python3 cmp_ecker.py ../ecc` from hp/compare/: exit 0, OK (fix 9) |
| identity_check | `I200-mac/identity_check.py` | . | 0 | all exist | none quoted | 7 s | OK |
| eeg20 | `solver/eeg20/` | . | 0 | none named | ratio: 2/2 | 3 s | OK (existing build) |
| tilt | `solver/eeg20/` | . | 0 | none named | measured/predicted at x = 1 (derived): 4/4 | 2 s | OK (existing build) |
| maxc | `solver/eeg20/` | . | 0 | none named | ratio: 2/2 | 3 s | OK (existing build) |

### How the points are computed

| id | command | from | exit | files | number printed vs quoted | wall | verdict |
|---|---|---|---|---|---|---|---|
| plot_solver_limits | `hp/plot_solver_limits.py` | hp | 2 | all exist | none quoted | 3 s | FAIL: exit 2. Adapted `python3 plot_solver_limits.py ../../choptuik-large-d-raw` from hp/: exit 0, OK (fix 3) |

### Validation and controls

| id | command | from | exit | files | number printed vs quoted | wall | verdict |
|---|---|---|---|---|---|---|---|
| rv2_noise | `hp/review/rv2_noise.py` | . | 0 | none named | rms: 1/1 | 4 s | OK |
| dd_fits_clean | `hp/dd_fits_mp.py --clean` | . | 0 | none named | dropped: 1/1 | 335 s | OK |
| dd_fits_10_45 | `hp/dd_fits_mp.py 10 45 --clean --dmax=60.1` | . | 0 | none named | 1/D, degree 16: 3/3; 1/D, degree 18: 3/3; 1/(D-2), degree 18: 3/3; (D-2)^{-1/2}, degree 20: 3/3; D^{-1/2}, degree 20: 3/3 | 106 s | OK |
| dd_fits_10_375 | `10 37.5 --clean --dmax=60.1` | . | 0 | none named | 1/D, degree 16: 1/1; 1/D, degree 18: 1/1; 1/(D-2), degree 18: 1/1; (D-2)^{-1/2}, degree 20: 1/1; D^{-1/2}, degree 20: 1/1 | 119 s | OK |
| dd_fits_mixed | `hp/dd_fits_mp.py 10 45 --clean --mixed --dmax=60.1` | . | 0 | none named | coefficient: 2/2; limits: 2/2 | 58 s | OK |
| dd_fits_1e9 | `hp/dd_fits_mp.py 10 1e9 --clean --dmax=60.1` | . | 0 | none named | limits: 2/2 | 344 s | OK |
| dinf_budget | `hp/dinf_budget.py` | . | 0 | none named | range: 3/3 | 214 s | OK |
| planted_dinf | `hp/planted_dinf.py` | hp | 1 | all exist | centre: 0/2 MISMATCH | 2382 s | FAIL: centre: not printed: 0.79×10⁻⁸ (nearest printed 8.4e-9, 0 digits agree); 0.88×10⁻⁸ (nearest printed 8.55e-9, 1 digits agree) [inside range True in 6 of 6, junk caught False in 6 of 6 (README: all six; every case)] [composite log, see Prioritised fixes] (fix 2) |
| rv2_planted | `hp/review/rv2_planted.py` | . | 0 | none named | offset: 1/1 | 13 s | OK |

### Large D: what the 1/D expansion can and cannot see

| id | command | from | exit | files | number printed vs quoted | wall | verdict |
|---|---|---|---|---|---|---|---|
| largeD_scaling | `hp/largeD_scaling.py` | . | 0 | none named | none quoted | 2 s | OK (existing build) |
| largeD_F | `hp/largeD_F.py` | . | 0 | none named | D = 20: 5/5; D = 30: 5/5; D = 40: 5/5; D = 50: 5/5; D = 60: 5/5 | 3 s | OK (existing build) |
| plot_wave_family | `hp/plot_wave_family.py` | . | 2 | all exist | none quoted | 1 s | FAIL: exit 2. Adapted `python3 plot_wave_family.py ../../choptuik-large-d-raw` from hp/: exit 0, OK (existing build) (fix 3) |
| largeD_expand | `largeD_expand.py 4` | . | 0 | none named | none quoted | 584 s | OK |
| largeD_fast_transport | `solver/largeD_fast_transport.py` | . | 0 | none named | no gravity: 5/5; with gravity: 5/5 | 36 s | OK |
| largeD_fast_transport_full | `solver/largeD_fast_transport_full.py` | . | 0 | none named | full background: 5/5 | 24 s | OK |
| largeD_exponent | `hp/largeD_exponent.py` | . | 0 | none named | n<G(1)>/2: 5/5; <F^2>: 5/5; shift: 5/5 | 2 s | OK (existing build) |
| plot_light_cone | `hp/plot_light_cone.py` | . | 0 | all exist | none quoted | 1 s | OK |
| largeD_linear_late | `solver/largeD_linear_late.py` | . | 0 | none named | 1/\\sqrt2: 4/4; 0\.8: 4/4; 0\.9: 4/4 | 2 s | OK |
| largeD_dingle_K | `solver/largeD_dingle_K.py` | . | 0 | none named | K ratio: 10/10 | 2 s | OK |
| largeD_linear_poles | `solver/largeD_linear_poles.py` | . | 0 | none named | none quoted | 59 s | OK |
| largeD_linear_poisson | `solver/largeD_linear_poisson.py` | . | -15 | none named | none quoted | 1810 s | NOT RUN: not finished in 30 min (stopped by pid 1352830) |
| euler_late | `solver/largeD_prop4/` | . | 0 | none named | none quoted | 85 s | OK |
| kcheck | `kcheck.py` | solver/largeD_prop4 | 0 | none named | deviation c0: 2/2 | 17 s | OK [1 + c/k: c = 3.652, 3.651, 3.651 (README 3.65)] |
| switch2 | `switch2.py` | . | 0 | none named | singulant: 1/1 | 1 s | OK |
| largeD_linear_truncation | `solver/largeD_linear_truncation.py` | . | 0 | none named | x = 0.9: 4/4 | 1 s | OK |
| hier_schur | `hier.py --schur` | solver/inner_stokes | 1 | none named | none quoted | 3 s | FAIL: exit 1 (fix 3) |
| largeD_stokes_C | `solver/largeD_stokes_C.py` | . | 0 | none named | ratios: 2/2; n(ratio-1): 2/2; |s^2| (derived, not printed): not checked | 1 s | OK |
| largeD_inner | `solver/largeD_inner.py` | . | 0 | none named | none quoted | 1 s | OK |
| amp_series | `amp_series.py` | . | 0 | none named | agreement: 2/2 | 5 s | OK |
| largeD_Fratios | `hp/largeD_Fratios.py` | . | 0 | none named | ratios (|F5/F1| derived as |F3/F1||F5/F3|): 6/6 | 4 s | OK (existing build) |
| amp_flint_par | `solver/inner_amp/amp_flint_par.py --fit phys` | solver/inner_amp | 1 | all exist | none quoted | 1 s | FAIL: exit 1 (fix 3) |
| zero_selection | `zero_selection.py` | . | 0 | none named | none quoted | 1 s | OK |
| arg_crossing | `arg_crossing.py` | solver/inner_amp | 1 | none named | none quoted | 3 s | FAIL: exit 1. Adapted `python3 arg_crossing.py scan13b` from solver/inner_amp/: exit 0, OK (fix 3) |
| amp_flint | `amp_flint.py` | . |  | none named | none quoted |  | NOT RUN: named as the legacy basis; no invocation or number quoted, and it is the serial reference (AGENTS.md: hours) |
| harm_pred | `harm_pred.py` | . | 0 | none named | tracks: 10/10 | 643 s | OK |
| plot_selection | `solver/plot_selection.py` | . | 0 | all exist | none quoted | 11 s | OK |
| plot_spectrum | `solver/plot_spectrum.py` | . | 0 | all exist | none quoted | 15 s | OK; the regenerated file differs from the committed one: solver/spectrum.svg (fix 11) |
| expand_pert | `solver/inner_pert/expand_pert.py` | . | 0 | none named | none quoted | 16 s | OK |
| pert_check | `pert_check.py` | . | 0 | none named | residual: 1/1 | 1 s | OK |
| pert_check2 | `pert_check2.py` | . | 0 | none named | residual: 1/1 | 1 s | OK (fix 7) |
| rw_map | `solver/inner_pert/rw_map.py` | . | 0 | none named | none quoted | 2 s | OK |
| pert_check3 | `solver/inner_pert/pert_check3.py 45 11` | . | 0 | none named | misses: 1/1 | 3 s | FAIL: worst deviation 2.1e-178 against 10^-180 (j = 1: 1.3e-180; 2 of 11 j round to 10^-180) (fix 7) |
| plot_first_order | `solver/plot_first_order.py` | . | 0 | all exist | none quoted | 3 s | OK; the regenerated file differs from the committed one: solver/first_order.svg (fix 11) |
| struct_css | `solver/matter/struct_*.py` | . | 0 | none named | none quoted | 4 s | OK |
| struct_inner | `solver/matter/struct_*.py` | . | 0 | none named | none quoted | 10 s | OK |
| struct_outer | `solver/matter/struct_*.py` | . | 0 | none named | none quoted | 45 s | OK |
| struct_symmetry | `solver/matter/struct_*.py` | . | 0 | none named | none quoted | 227 s | OK |
| struct_eqs | `solver/matter/struct_*.py` | . | 0 | none named | none quoted | 376 s | OK |
| plot_matter | `solver/plot_matter.py` | . | 0 | all exist | none quoted | 3 s | OK |

### Limits and open problems

| id | command | from | exit | files | number printed vs quoted | wall | verdict |
|---|---|---|---|---|---|---|---|
| singularity_type | `hp/singularity_type.py planted` | hp | 1 | none named | sqrt exponent: 1/1 | 49 s | FAIL: exit 1 (fix 10) |
| track_c_fold | `hp/track_c_fold.py` | hp | 1 | all exist | none quoted | 1 s | FAIL: exit 1. Adapted `python3 track_c_fold.py run` from hp/: exit 0, OK (fix 3) |
| horizon_toy | `hp/horizon_toy.py` | . | 0 | none named | none quoted | 1 s | OK |
| plot_dropoffs | `hp/plot_dropoffs.py` | hp | 1 | all exist | none quoted | 4 s | FAIL: exit 1. Adapted `python3 plot_dropoffs.py ../../choptuik-large-d-raw/T76` from hp/: exit 0, OK (fix 3) |
| plot_gamma | `solver/plot_gamma.py` | . | 0 | all exist | none quoted | 3 s | OK; the regenerated file differs from the committed one: solver/gamma.svg (fix 11) |
| plot_modecount | `hp/plot_modecount.py` | . | 0 | all exist | none quoted | 3 s | OK; the regenerated file differs from the committed one: hp/modecount.svg (fix 11) |

### How this work was done

| id | command | from | exit | files | number printed vs quoted | wall | verdict |
|---|---|---|---|---|---|---|---|
| mutation_controls | `solver/proof/mutation_controls.py` | . | 0 | none named | eight versions: 1/1 | 1 s | OK |
| check_numbers_selftest | `check_numbers.py` | . | 1 | none named | none quoted | 23 s | FAIL: exit 1 (existing build) (fix 13) |
| check_numbers | `check_numbers.py` | . | 1 | none named | none quoted | 29 s | FAIL: exit 1 (existing build) (fix 13) |

### Reproduce

| id | command | from | exit | files | number printed vs quoted | wall | verdict |
|---|---|---|---|---|---|---|---|
| reproduce_sh | `./reproduce.sh` | . | 1 | none named | none quoted | 1037 s | FAIL: exit 1; 13 checks (11 passed, 1 failed, 1 skipped), README says 14 (fix 1) |
| reproduce_sh_hp | `./reproduce.sh` | . | 1 | none named | none quoted | 506 s | FAIL: exit 1 [14 checks (13 passed, 1 failed, 0 skipped), README says 14] (existing build) (fix 1) |
| reproduce_sh_full | `--full` | . | 1 | none named | none quoted | 510 s | FAIL: exit 1; 14 checks (12 passed, 1 failed, 1 skipped), README says 15 (fix 1) |
| reproduce_sh_full_hp | `--full` | . | 1 | none named | none quoted | 474 s | FAIL: exit 1 [15 checks (14 passed, 1 failed, 0 skipped), README says 15] (existing build) (fix 1) |
| hp_build_sh | `hp/build.sh` | . | 0 | none named | none quoted | 1 s | OK (existing build) |
| states_module | `hp/states.py` | . |  | none named | none quoted |  | NOT RUN: a module the profile scripts import (where the states are read from), not a command; the README quotes no invocation |
| pip_install | `pip install .` | . | 1 | none named | none quoted | 1 s | NOT RUN: no existing build on this machine has rustkernel (the 5 venvs under ~/Source tried) (existing build) (fix 14) |
| cargo_test | `cargo test --release` | . |  | none named | none quoted |  | NOT RUN: cargo build: no builds on this pass (coordinator) (fix 14) |
| test_bindings | `python tests/test_bindings.py` | . | 1 | all exist | none quoted | 1 s | NOT RUN: no existing build on this machine has rustkernel (the 5 venvs under ~/Source tried) (existing build) (fix 14) |
| build_sh_main | `./build.sh main` | . |  | none named | none quoted |  | NOT RUN: build: no builds on this pass (coordinator) (fix 14) |
| git_clone_raw | `git clone https://github.com/nmoinvaz/choptuik-large-d-raw ../choptuik-large-d-raw` | . | 0 | all exist | none quoted | 281 s | OK |


## Commands skipped

11 rows were not run. Two of the 11 came out as "not finished", the rest were not started.

| Command | README | Why it did not run | README's quoted time |
|---|---|---|---|
| `pip install .` | 2197 | No builds in this pass. The second pass looked for rustkernel in the 5 existing venvs under `~/Source` (1058b, 1058, 935, 958, main) and found it in none. | none |
| `python tests/test_bindings.py` | 2199 | Needs rustkernel: same. | none |
| `cargo test --release` | 2198 | No builds. | none |
| `./build.sh main` | 2200 | No builds. | none |
| `hp/states.py` | 2186 | A module the profile scripts import, not a command; the README quotes no invocation and no number. | none |
| `fixed_point.py`, `fixed_point_M.py`, `validated_int.py`, `krawczyk.py` | 206 | Named as the code of the certified lemmas; no invocation, no number. PROOF.md and proof/A_certify.md hold their commands, out of scope (README only). | none |
| `amp_flint.py` | 1590 | Named as the legacy basis; no invocation and no number. It is the serial reference, hours long (AGENTS.md). | none |
| `solver/largeD_linear_poisson.py` | 1334 | Not finished in 30 min, twice: 1810 s with 690 s of cpu, then 1810 s with 960 s of cpu, at load 8.7 to 9. Stopped by its pid. | none |

Five commands failed as written for lack of arguments and were not given an adapted rerun, because the arguments are
saved states or production solves that are not in the repository: `hier.py --schur` (needs M, A and a frequency;
hours at M ≥ 7), `amp_flint_par.py --fit phys` (needs ω, K, J; hours), `hp/gamma_dd.py` (needs an output directory; a
double-double solve per point), `hp/gamma_m48_gap.py` (subcommands that read saved states), `hp/modecount_test.py` (a
state path). They are FAIL rows in the table above and fix 3 below.

`planted_dinf.py` took 1190 s a run at load 5.2 and was not finished in 30 min in its rerun at load 8.9. Its first
complete run is the row's record; see fix 2.

## Links, figures, Layout

- **Links.** `reproduce_readme.py --links` checks every link of README.md: 27 external (not fetched), 159 relative, of
  which 18 are images, and 99 anchors, each resolved with GitHub's slug rule. All relative links and anchors resolve.
  **7 footnotes are defined and never referenced**: [^15], [^16] and [^18] to [^22]. [^17] is neither defined nor
  referenced (fix 12). 16 of the 23 defined footnotes are referenced. The check can fail: a README copy with a planted
  broken anchor, a missing file and an undefined footnote gives 3 more BROKEN lines
  (`out/control/links_planted.log` in the raw folder).
- **File names.** 139 mentions of 119 distinct file names in code spans: 89 exist as written from the repository root,
  28 are abbreviated paths that exist elsewhere in the repository (`amp_flint.py` for `solver/inner_amp/amp_flint.py`,
  `scan15phys/` for `solver/inner_amp/scan15phys/`, `T76/` for the raw clone's folder), 1 is a pattern and
  **1 is not a file**: `_J1000_phys.json`, a fragment at line 1595. The README names no file that does not exist.
- **Figures.** All 18 images the README embeds exist. 19 figure or table files were regenerated by their commands:
  - 10 are identical: `hp/echo_shapes.svg` and `hp/data/echo_shapes.csv`, `solver/extrapolation.svg`,
    `hp/light_cone.svg`, `hp/lowD_values.svg`, `solver/matter.svg`, `solver/selection.svg`, `hp/solver_limits.svg`,
    `hp/large_d_dropoffs.svg`, `hp/wave_family.svg`.
  - 9 differ: 5 SVGs by one dark-mode style rule (`physics`, `spectrum`, `first_order`, `gamma`, `modecount`); 3 SVGs
    by data points the committed figures lack (`solver/delta.svg`, `hp/lowD_limits.svg`, `hp/lowD_logslope.svg`; the
    first two also by the tooltip and a machine stamp); and `hp/data/delta_all.csv` (2431 rows against the committed
    2512). Fixes 6 and 11; the diffs are in `svgdiff/diffs/` of the raw folder.
- **Layout.** The README has no layout block (the old record's Layout section no longer applies); SOFTWARE.md and
  SETUP.md describe the files. Nothing to check.

## Dependencies

- **SETUP.md's list is enough for the clean venv.** Python 3.14, numpy, scipy, mpmath, sympy and python-flint ran
  everything in pass 1 except the rows below. No undeclared module was found: the only missing modules were `hpkernel`
  (15 rows, deferred to pass 2) and `rustkernel` (`test_bindings.py`, and `pip install .` which builds it).
- **hpkernel** (`hp/build.sh`, Rust, maturin, and multidouble's Python package from GitHub) is needed by every `hp/`
  script that solves or reads saved states: `gamma_compare.py`, `largeD_scaling.py`, `largeD_F.py`,
  `largeD_exponent.py`, `largeD_Fratios.py`, `dd_D5.py`, `eeg20.py` and others. The README says so only for
  `reproduce.sh`'s checks on saved solutions. A reader who follows the README commands without building it gets a
  traceback from each of those rows.
- **The Ecker data** (`cmp_ecker.py`, `lowD_formula.py`) is a clone of
  https://github.com/EckerChristian/CriticalSpacetimeCrystal passed as the argument. The README gives no clone step
  (fix 9). The clone is data: it was read, never run.
- **The raw clone** (`../choptuik-large-d-raw`) is needed by the 8 rows that pass its path or run a script in it
  (`plot_delta`, `export_delta_all` and its `--verify`, `plot_solver_limits`, `plot_echo`, `plot_wave_family`,
  `plot_dropoffs`, `identity_check`); SETUP.md has the clone step.
- **Not needed:** matplotlib, numba, cairosvg, pdflatex, MPFR (SETUP.md lists them as optional); no README command used
  them.

## How to rerun this

`reproduce_readme.py` is at the repository root and reads the README's quoted numbers at run time: a number changed
in a copy of the README (`--readme COPY`) changes what the row expects.

1. Clone the repository, make a venv as SETUP.md says, and clone the raw-data repository beside it.
2. Make a throwaway worktree for the runs: `git worktree add ../run HEAD && touch ../run/.i349_run_tree`. The
   runner restores a tree that has the marker after every row (`git checkout`, `git clean`) and leaves any other tree as
   it finds it.
3. Pass 1, in the clean venv, one command at a time:
   `python3 reproduce_readme.py --tree ../run --out out/pass1 --python .venv/bin/python`. It picks 4 threads at nice 10
   when the load is under 4 and 2 at nice 19 otherwise, stops a command at `--limit` seconds (1800) by its pid, and
   stops the pass at `--cap` hours (10). It prints a row per command and writes `results.jsonl`, `table.md` and
   `logs/<id>.log`.
4. Pass 2, with an existing build (never builds): `--pass2 out/pass1/results.jsonl --python <interpreter with hpkernel>
   --out out/pass2`. `--only ID,ID` narrows either pass; `--redo ID,ID` runs rows again and keeps the old results in
   `results.first.jsonl`; `--rescore DIR` scores saved logs again after a change to a row's locators, with no rerun.
5. `--merge out/pass1 out/pass2` prints the merged table and counts, with cpu and memory; `--merge ... --compact` prints
   the tables of "Commands checked" above; `--links` checks links, anchors and footnotes; `--coverage` lists code
   spans that look like commands and are no row; `--list` shows each row, its quote and its command.
6. **The control** (about 30 s): `python3 reproduce_readme.py --tree ../run --out out/control --python .venv/bin/python
   --control`. It runs five fast rows on the real README and on three copies with a planted number, a changed command
   and a planted number in a second row, plus a command that exits 1 and one that names a missing file. It passes when
   exactly the planted row flips and the two synthetic failures flag. It passed on 2026-11-22 08:42 UTC with the final
   checker (`out/control/control.md`).

The machine's load roughly doubles to triples the wall times; only `reproduce.sh`'s quoted time was compared.

The logs and records of the run in this file are in `I349-dh170/` of choptuik-large-d-raw (commit 98d9044a).

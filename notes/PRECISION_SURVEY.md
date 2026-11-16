# Decimals read before the precision is set, in solver/proof (#879)

python-flint starts at 53 bits. A decimal string turned into a ball before the working precision is set is still an
enclosure of the decimal, but its radius is about 10⁻¹⁶ relative, so a "point" becomes a small box and a box's centre
moves by 10⁻¹⁷. This was found twice by chance: in `fixed_point_M.data_table` (#484) and in `fixed_point_M_b.main`
(#867, F2). This note surveys every command line of `solver/proof/` for it.

No verdict changes. Every flagged command that a stored run used was run again with the precision set first, and
each gives the same exit status. The Theorem A chain at M = 3 and M = 5 has no early read on any traced path.

## Contents

- [How the survey was made](#how-the-survey-was-made)
- [1. Early reads that are followed by a higher precision](#1-early-reads-that-are-followed-by-a-higher-precision)
- [2. The reruns with the precision set first](#2-the-reruns-with-the-precision-set-first)
- [3. Scripts that work at 53 bits throughout](#3-scripts-that-work-at-53-bits-throughout)
- [4. Reads at 53 bits that are intended](#4-reads-at-53-bits-that-are-intended)
- [5. What the survey did not reach](#5-what-the-survey-did-not-reach)
- [6. Reproduce](#6-reproduce)

## How the survey was made

`solver/proof/prec_trace.py` runs a script under a stand-in for the flint module whose `arb` and `acb` record every
construction from a string while the precision is below 100 bits: the file, line and function of the caller, the
string and the precision. After the run it lists the module-level balls whose relative radius is above 2⁻¹⁰⁰. Its
selftest must report a script that reads its argument first and must not report the same script with the precision
set first, and it does both.

`solver/proof/prec_survey.py` collected every `python3 <script>.py ...` line of the notes, the README, PROOF.md, the
shell scripts under `solver/proof/` and the raw-data repository's run records, for scripts present in `solver/proof/`.
It traced each for 5 seconds, with every output option pointed into a scratch folder, and traced every script that
imports flint with no argument as well.

| | count |
|---|---|
| command lines collected | 330, of 66 scripts |
| traced runs, with the 68 runs without an argument | 347, of 81 scripts |
| skipped for a placeholder or an undefined shell variable | 51 |
| traced by hand afterwards, with the variables filled in | 21 |
| runs that showed an early read | 52 |
| places in the code that read a decimal early | 28 |
| scripts with no early read on any traced path | 63 |
| reviewers' scripts in `review*/` traced with no argument | 18 |

## 1. Early reads that are followed by a higher precision

These are the defect: a number read at 53 bits and then used at 192 to 512 bits.

| script and line | what is read early | stored runs on that path | proposed fix |
|---|---|---|---|
| `fixed_point_M.py:705`, `data_table` | the real box x̂ ± h of the first row of an invocation | the first rows of `lemmas_M_runs/fixed_point_M_data_M3.json` and `_M5.json` (3π/8, h = 10⁻⁹) | set the precision at the top of `data_table`. Held: stored runs record this file's SHA-256 |
| `fixed_point_M.py:81`, `W0` | 1.64389375, from `r0_inner_np` in `table` and `data_table` | every `--table` and `--data-table` run | none needed: the value becomes a float printed to four decimals |
| `fixed_point.py:185` | `A_FAST` = √2/4, the kernel rate, made at import | every M = 1 run of `fixed_point.py` and of `lemma6_sector.py`. `fixed_point_M.py` replaces it with its own rate | make it inside `run`, after the precision is set |
| `fixed_point.py:193`, `set_sector` | the sector's half-width 0.2 | `fixed_point.py --sector=0.2` (A_certify.md) | set the precision at the top of `main` |
| `b8_formal.py:32–33` | W₀, A\*, ω\*/ω∞ at import. `params` then takes A\*'s midpoint as the box centre | every run of `b8m_atoms.py`, `b8m_scan.py`, `b8m_h8.py`, `b8m_lc.py`, which raise the precision to 256 bits later | done in #885: the precision is set in `b8_formal.py` before these three lines |
| `b8m_scan.py:26–27` | the tuning constants `slacks` and `eps` | the three `b8m_scan.py` runs of B8_constants.md and B_chain.md | done in #887: the script states its precision first |
| `b8_bounds.py:21`, `b8m_bounds.py:31`, `b8m_lc.py:54` | π at import | the runs of `b8m_lc.py`, `b8m_h8.py`, `b8m_scan.py` | done in #885 and #887: the precision is set before these lines |
| `b5_bounds.py:447` | the default argument `rhoW = arb("1e12")`, at import | every run of `b5_bounds.py`, `b5_label.py`, `b7_ift.py` | none needed: 10¹² is exact at 53 bits |
| `fixed_point_M_b.py:912–917`, `main` | `--x`, `--box`, `--cbox` on the plain path | none | #878 |
| `review_867/rv_system.py:119` | the mutation's factor 1 + 10⁻⁹ | the two mutation runs of #867 | fixed on this branch: the reviewer's own script |

In `fixed_point_M.py` the `--box` and `--cbox` values of `main` are read after the precision is set, and so is
`--x`. `validated_int.py` sets its precision before its module-level W₀.

## 2. The reruns with the precision set first

`prec_trace.py --prec=320` sets the precision before the script and its modules load. `prec_survey.py rerun` ran each
of the 31 documented commands with an early read twice, as written and with `--prec=320`, and compared the output.

| commands | exit status | lines of output that differ |
|---|---|---|
| `b5_bounds.py`, 6 commands | equal | 0 |
| `b5_label.py`, 3 | equal | 0 |
| `b5_jump.py`, 3 | equal | 0 |
| `b5_threshold.py`, 1 | equal | 0 |
| `b7_ift.py`, 2 | equal | 0 |
| `b8m_atoms.py`, 1 | equal | 0 |
| `b8m_lcj.py`, 2 | equal | 0 |
| `b8m_h8.py`, 2 | equal | 1 of 12 each, in the last digits of printed floats |
| `b8m_scan.py`, 3 | equal | every record line, in the last digits of printed floats |
| `b8m_lc.py`, 3 | equal | 1 line each for two of them, in the last digits of printed floats |
| `fixed_point.py --sector=0.2` | equal | 0 of 32 |
| `fixed_point_M.py --table`, 3, and `--data-table --Ms=3` | equal | 0 |

So 0 of 31 commands change their exit status, and 7 print other last digits. Two results were also compared as files.

**The lemma's start-data rows.** Only the first row of each invocation moves, as #484 said:

| stored file | row | as written, against stored | precision set first, against stored |
|---|---|---|---|
| `fixed_point_M_data_M3.json` | 3π/8, r₀ = 0.085, h = 10⁻⁹ | at most 1.3×10⁻¹⁴ | value radius 2.0×10⁻⁸, sup\|η\| 3.7×10⁻⁸, Cauchy part 2.3×10⁻¹¹ |
| | the other three rows | at most 1.2×10⁻¹⁴ | the same |
| `fixed_point_M_data_M5.json` | 3π/8, r₀ = 0.07, h = 10⁻⁹ | at most 6.0×10⁻¹³ | value radius 1.2×10⁻⁹, derivative radius 2.6×10⁻⁹, sup\|η\| 1.6×10⁻⁹, Cauchy part 1.2×10⁻⁸ |
| | the other three rows | at most 4.0×10⁻¹³ | the same |

The numbers are relative differences. The rows on the ray 5π/16, which contain the point certificates' start sets
(A_lemmas_M.md §2.4), do not move.

**Lemma LC's Lipschitz matrix** (`b8m_lc.py 4 73.89 --rA=2e-6 --rW=2e-6 --lip`, which writes `b8m_lc_lip.json`). Run as
written, it reproduces the stored file in all 1600 entries. With the precision set first, 552 entries move by more
than 10⁻⁹, by factors between 0.733 and 1.014, and the certified contraction constant is θ = 0.7355 from either matrix
(`--lipcheck`). This is the only place where the early constants show beyond the last digits. The stored matrix is a
valid bound either way, since a 53-bit ball contains the decimal.

The cause is rounding noise (#885, raw `I885-mac/`). An entry of the matrix is a secant, the difference of two
256-bit evaluations divided by η = 10⁻⁶. Where the bound does not depend on the perturbed input, that difference is
the rounding of one number against itself. 1108 of the 1600 entries are of this kind, all below 2.3×10⁻⁶⁹. With
η = 10⁻³ they shrink by 10³, and at 320 bits by 2⁶⁴, while every entry of 10⁻⁶⁰ or more moves by at most 9×10⁻⁵ and
2×10⁻⁹. Every entry that moved is below 6.1×10⁻⁶³, and the entries below 10⁻⁶⁰ add at most 2.3×10⁻⁶⁰ to a weighted
row sum of θ.

## 3. Scripts that work at 53 bits throughout

These never set the precision, so their balls are 53-bit balls from the first line to the last. That is not the
defect above, since nothing is mixed, and their bounds are rigorous at that precision. It is a choice the code does
not state.

| script | what it reads | stored runs | with the precision at 320 bits |
|---|---|---|---|
| `b5_jump.py` | the inflation 10⁻¹², u\*, and the values of `b5_label.json` and the supersolution files (lines 35, 54, 79–90) | 3 commands of B_chain.md | same exit status, 0 of 16 lines differ |
| `b5_threshold.py` | A\* and the supersolution files (lines 44, 96); its second part is labelled an estimate | 1 command of B5_inner.md | same exit status, 0 of 29 lines differ |
| `b8m_lcj.py` | the record of `b8m_lc_K4.log` | 2 commands of B_chain.md | same exit status, 0 of 13 lines differ |
| `b8_atoms.py`, `b8_bounds.py`, `b8_formal.py`, `b8m_bounds.py` run as scripts | the zero and the domain constants | none documented | not rerun |

Done in #887: each of these scripts, and `b8m_bounds.py`, `b8m_lc.py` and `b8m_scan.py`, states its working precision
in one line after its imports, 256 bits in the B8 scripts and 192 bits in `b5_jump.py` and `b5_threshold.py`. The
tracer then reports no early read for them, and their printed output is unchanged (raw `I887-mac/`).

## 4. Reads at 53 bits that are intended

| script and line | why |
|---|---|
| `start_cover_M2.py:195` | it rebuilds `data_table`'s box both ways, at 53 and at 320 bits, and tests containment in both (#484) |
| `review/rv_krawczyk_recheck.py:49` | a round trip of stored balls, parsed at 320 and at 53 bits on purpose |
| `review_867/rv_cli_precision.py:38` | it shows #867's F2 |

## 5. What the survey did not reach

- A read is seen only if the run reaches it within the time limit, 5 seconds, or 8 to 15 for the 21 commands traced
  by hand. A script that reads a decimal later, at a precision that it lowered again, would be missed. None lowers it.
- 51 documented commands have a placeholder or an undefined shell variable. The 21 traced by hand cover the
  covering's `check`, `c1check`, `ray` and `row` with the certificates' centres, lemma 6a at M = 3, lemma S, both
  rechecks, both Krawczyk tests and both certificate box runs. All show no early read.
- `theorem_A_sets_M.py` and `start_cover_lemmas_M.py` at M = 5 need raw files that the local raw clone does not hold.
  Both set the precision at the top of `check` and of `load`, by reading.
- Balls made from floats are not reads of a decimal and are not recorded. A script that starts other scripts as
  processes, such as `mutations_M.py`, is traced without them.
- The list of module-level balls with a wide radius also names results that are wide by design, such as the start
  values that `fixed_point.py` leaves at module level. Those were not examined one by one. The raw table has them.
- The reviewers' scripts were traced with no argument only. Two of them import `b8_formal.py` and inherit its
  constants.

## 6. Reproduce

From `solver/proof/`, one process. The traces write into `prec_survey_tmp/`, and some scripts also write files of
fixed name beside themselves, so run the survey in a worktree and restore it afterwards.

```
python3 prec_trace.py --selftest                           # exit 0: an early read is reported, a late one is not
python3 prec_trace.py -- fixed_point_M.py --data-table --Ms=5   # exit 1: the reads of data_table, in seconds
python3 prec_survey.py collect --raw=<raw clone>           # 330 command lines of 66 scripts
python3 prec_survey.py run --limit=5 --raw=<raw clone>     # 11 min -> prec_survey_runs.json
python3 prec_survey.py table                               # sections 1, 3 and 4
python3 prec_survey.py rerun --limit=240                   # 14 min -> prec_survey_rerun.json, section 2
```

The collected commands, the traces and the reruns are in raw `I879-mac/`.

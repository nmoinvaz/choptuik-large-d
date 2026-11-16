# Independent review of the floor of lemma LC's Lipschitz matrix

Task: an independent review of the floor that `b8m_lc.lipschitz` puts under the entries of lemma LC's Lipschitz
matrix (`--lipfloor`, 10⁻⁵⁰; #891), of its check `b8m_lc_liprep.py`, and of what [B_chain.md](B_chain.md) §11 and
§13.4 say about it, at main dbaf7af3 (#894). Reviewer: an agent that did not write the floor or any of
`b8m_lc.py`. Check script: [`solver/proof/review_894/rv_floor.py`](../solver/proof/review_894/rv_floor.py). Runs and
logs: raw `I894-mac/`. [A_M_review.md](A_M_review.md) was read as the template.

## Contents

- [Verdict](#verdict)
- [Summary of the checks](#summary-of-the-checks)
- [1. What the floor does](#1-what-the-floor-does)
- [2. The floored matrix against the unfloored one](#2-the-floored-matrix-against-the-unfloored-one)
- [3. The row sums and θ](#3-the-row-sums-and-θ)
- [4. Two centres](#4-two-centres)
- [5. The saved file and the stored matrix](#5-the-saved-file-and-the-stored-matrix)
- [6. The author's check, its control and the reviewer's mutation](#6-the-authors-check-its-control-and-the-reviewers-mutation)
- [7. Findings](#7-findings)
- [8. What I did not check](#8-what-i-did-not-check)
- [9. Reproduce](#9-reproduce)

## Verdict

**The floor is sound, and θ = 0.7355 stays certified.** Writing an entry at or below 10⁻⁵⁰ as 10⁻⁵⁰ can only raise
it, so every entry of the saved matrix is still an upper bound of the entry that `lipschitz` computed, whatever made
that entry small. The floored matrix is the one θ is computed from in the run, and the one saved. I found no error in
the code, in the stored matrix or in the numbers of B_chain.md §13.4, and I reproduced each of them:

- the stored `b8m_lc_lip.json` equals my rerun in all 1600 entries, in every digit;
- the 396 entries above the floor equal the unfloored ones in every digit, and no entry is lowered;
- 1204 entries are at the floor, 1184 of them below 10⁻⁶⁰ without it and 20 between 10⁻⁶⁰ and 10⁻⁵⁰;
- they add at most 2.34×10⁻⁴⁸ to a weighted row sum, and θ is 0.735467756408 with and without the floor.

**One finding is about check coverage (F1): none of the author's checks can tell a raised entry from a lowered one.**
With the reviewer's mutation, which writes every entry as the floor, the certificate's command still ends with
"certified theta = 0.0000 (< 1: True)" and exit status 0, and `b8m_lc_liprep.py` and `--lipcheck` pass on the result.
The reviewer's comparison with the unfloored matrix catches it. This does not touch the stored matrix, which that
comparison passes; it says that the present checks would not notice such an error in a later change.

No finding changes a number or a verdict. F2 and F4 ask for a few words in a note and a sentence, and F3 and F5 are
observations.

## Summary of the checks

| check | how | verdict |
|---|---|---|
| R1: the flooring line can only raise an entry, its comparison is decided, and the floored matrix is the one certified and saved | reading `lipschitz`, `lip_floor`, `contraction_check`, `lipcheck_file` and `main` (§1) | OK |
| R2: floored against unfloored at the same centre | K1: two runs, entry by entry (§2) | OK: 0 lowered, 396 above the floor equal in every digit, 1204 written as the floor |
| R3: the floored entries' share of a weighted row sum, and θ | K2: exact rational row sums (§3) | OK: 2.34×10⁻⁴⁸ against the stated 2.4×10⁻⁴⁸; θ rises by 5.5×10⁻⁵⁰; F4 |
| R4: the entries above the floor between two centres 10⁻¹⁷ apart | K3 (§4) | OK: within 5.29×10⁻¹¹, the same 1204 at the floor |
| R5: the saved fields, and the stored matrix against the rerun | K4, K5 (§5) | OK; F2 on the note's wording |
| R6: the author's check and its control | rerun (§6) | OK: `b8m_lc_liprep.py` exits 0; `--mutate=nofloor` exits 1 with 636 entries differing |
| R7: the reviewer's mutation, every entry written as the floor | K1 on the mutated matrix (§6) | fires: K1 fails, 396 entries lowered; the author's checks do not notice (F1) |

## 1. What the floor does

`lipschitz` builds the matrix M column by column. An entry is `arb(x.upper())` for an enclosure x: an exact number,
the upper end of a ball that contains the secant slope of a majorant. The secant is taken between two evaluations in
ball mode, so the ball contains the true difference, and its upper end bounds the true secant from above. The
induction of B_chain.md §13.4 then makes the entry an upper bound of the Lipschitz constant it stands for. That part
is not new and not reviewed here (§8).

The new lines are

```
fl = lip_floor(floor)                     # arb(arb(floor).upper()) if float(floor) > 0 else None
if fl is not None:
    M = {c: {r: (v if v > fl else fl) for r, v in M[c].items()} for c in cols}
```

- `fl` is exact, the upper end of the ball of the decimal 10⁻⁵⁰. Both sides of `v > fl` are exact, so the comparison
  is decided and never falls back on arb's "cannot tell".
- The new entry is max(v, fl) ≥ v. It is an upper bound whenever v was. The reason v is small plays no part: noise
  from a bound that does not depend on the perturbed input, a real slope of 10⁻⁵³, or zero.
- The floor is applied inside `lipschitz`, before it returns. `main` computes θ from the returned matrix
  (`contraction_check`), counts the entries at the floor from it, and saves it. So the certified matrix, the counted
  one and the saved one are the same object.
- `--lipfloor=0` gives `None` and no floor, which is the path of the author's control.
- The saved entries are doubles, `float(M.upper())`, rounded to nearest. `lipcheck_file` multiplies each by
  1 + 10⁻¹² on loading, which covers the rounding. The double 1e-50 itself lies above the decimal 10⁻⁵⁰ by a relative
  7.6×10⁻¹⁸ (F5).

## 2. The floored matrix against the unfloored one

K1 compares the saved matrix of a run with the floor (`lip.json`) and of a run without it (`lip_nofloor.json`), both
at the certificate's centre and from the same code, so that the same arithmetic produced both. These are the first
runs of `b8m_lc_liprep.py` and of its `--mutate=nofloor`.

| | count |
|---|---|
| entries | 1600 |
| entries lowered by the floor | 0 |
| unfloored entries above the floor | 396, each equal to the floored one in every digit |
| unfloored entries at or below the floor | 1204, each written as 10⁻⁵⁰ |
| of those, below 10⁻⁶⁰ | 1184 |
| of those, between 10⁻⁶⁰ and 10⁻⁵⁰ | 20 |
| smallest entry above the floor | 4.912×10⁻⁵⁰ |
| largest entry at or below it | 4.852×10⁻⁵³ |

These are the numbers of B_chain.md §13.4 and of `I891-mac/README.md`.

## 3. The row sums and θ

K2 takes the saved doubles as exact rationals and forms the weighted row sums max_r Σ_c M_rc v_c / v_r without any
rounding.

| quantity | value |
|---|---|
| θ from `lipcheck_file`, floored and unfloored | 0.735467756408 and 0.735467756408 |
| the reviewer's exact row sum with the certificate's weights | 0.735467756407 (without `lipcheck_file`'s inflation of 10⁻¹²) |
| what the 1204 entries at the floor add to a weighted row sum, certificate's weights | at most 2.339×10⁻⁴⁸ (stated: 2.4×10⁻⁴⁸) |
| rise of θ from the unfloored to the floored matrix, same weights | 5.5×10⁻⁵⁰ |
| the reviewer's own weights (power iteration, an approximate Perron vector): θ floored and unfloored | 0.705132216712 both, equal to 9×10⁻⁵¹ |

The last row is the spectral radius 0.705 of §13.4, so θ < 1 holds with weights that share nothing with
`contraction_check`. The stated 2.4×10⁻⁴⁸ belongs to the certificate's weights (F4).

## 4. Two centres

K3 compares the floored matrices of two runs whose centres A\* differ by 10⁻¹⁷ (`lip.json`, `lip_dA.json`).

- The same 1204 entries are at the floor in both.
- The entries above the floor differ by at most 5.29×10⁻¹¹, at `der/S/Nb<-val/b/N0`, and 37 differ by more than
  10⁻¹². B_chain.md states 5.3×10⁻¹¹.
- The cause is as stated. Among the numbers of the two run records the largest change is that of h_β in the band N0,
  21.476554750635 against 21.476554749499, with the hypothesis and P(h) built on it: the same 5.29×10⁻¹¹. The next
  largest change in the record is 2.6×10⁻¹³.

The floor leaves these entries alone (§2), so each is the bound that `lipschitz` computed for its own run's h, as
before the floor. The closure's h is any vector that passes the invariance test, and the matrix is computed for that
h, so a change of h in the tenth digit changes the bound and not its validity.

## 5. The saved file and the stored matrix

K4: the saved file has `floor` 10⁻⁵⁰, `floored` 1204 (recounted from the entries: 1204), `eta` 10⁻⁶ and the note
"M: upper bounds of the secant slopes. An entry below floor is written as floor: the secant of a bound that does not
depend on the perturbed input is rounding noise over eta". The fields say what the matrix is. The note's reason
covers 1184 of the 1204 entries (F2).

K5: the stored `solver/proof/b8m_lc_lip.json` equals the rerun in all 1600 entries, with the same `floor` and
`floored`. The stored record `b8m_lc_K4.log` equals the rerun's in all 255 numbers apart from the two run times,
including `lip.theta` = 0.7354677564068197, `lip.floor` and `lip.floored`.

## 6. The author's check, its control and the reviewer's mutation

**Rerun.** `b8m_lc_liprep.py` exits 0: 0 of 1600 entries differ by more than 10⁻⁹ between the two centres, the
largest difference is 5.3×10⁻¹¹, θ = 0.7355 and 0.7355. Its control `--mutate=nofloor` exits 1: 636 entries differ,
by up to 0.27. Both as B_chain.md §11 and §13.4 say.

**The reviewer's mutation, `cap`.** Written down before the run: the flooring expression `(v if v > fl else fl)` is
replaced by `(fl)` in a copy of `b8m_lc.py` held in memory, so that every entry, including the 396 true bounds above
10⁻⁵⁰, is written as the floor. K1 must fail on the result.

| what ran on the mutated code | result |
|---|---|
| the certificate's command, `b8m_lc.py 4 73.89 --rA=2e-6 --rW=2e-6 --lip` | exit 0; "1600 of 1600 entries are at the floor"; "certified theta = 0.0000 (< 1: True)" |
| K1, the reviewer's comparison with the unfloored matrix | **fails**, exit 3: 396 entries lowered, the largest from 1.0 to 10⁻⁵⁰ at `der/V/N0<-val/S/N0` |
| the author's `b8m_lc_liprep.py --compare` on the two mutated runs (centres 10⁻¹⁷ apart) | PASS, exit 0: 0 entries differ, θ = 0.0000 and 0.0000 |
| the author's `b8m_lc.py --lipcheck` on the mutated matrix | exit 0: certified θ = 0.0000 |

So the mutation fires in the reviewer's check and in none of the author's (F1).

## 7. Findings

**F1. No author's check can tell a raised entry from a lowered one.** Check coverage. `b8m_lc_liprep.py` passes when
two runs agree and θ < 1. A matrix whose entries were all written down to the floor agrees with itself and has
θ ≈ 0, so it passes, and so do `--lipcheck` and the certificate's own exit status (§6). Nothing ties the quoted
θ = 0.7355 to the stored record either: `check_numbers.py` has no check on it. The stored matrix is not affected: it
passes K1. *Proposed:* (a) let `b8m_lc_liprep.py` compare its floored run with its unfloored one entry by entry, as
K1 does, since it already makes both; (b) add θ = 0.7355 to `check_numbers.py`, from `b8m_lc_K4.log`'s `lip.theta`.

**F2. The saved note names one cause, and the floor covers two.** Text. The note in the saved file and the docstring
of `lipschitz` give rounding noise as the reason. 20 of the 1204 floored entries are real bounds between 10⁻⁶⁰ and
10⁻⁵⁰ (the largest 4.9×10⁻⁵³), as B_chain.md §13.4 says. The floor is valid for them for the same reason, but a
reader of the file alone would take all 1204 for noise. *Proposed:* "An entry at or below floor is written as floor,
whatever its cause; most are secants of bounds that do not depend on the perturbed input, which are rounding noise
over eta."

**F3. The floor lies a factor 5 below a real entry.** Observation. The smallest entry above the floor is 4.9×10⁻⁵⁰
and the largest below it 4.9×10⁻⁵³. At this box the two populations are three decades apart and the same entries are
floored at both centres. At another box, or with another floor, an entry could sit at the floor's level and be
floored in one run and not in the next. The bound would stay valid; only the run-to-run comparison would flag it.
No change is proposed.

**F4. "At most 2.4×10⁻⁴⁸ to a weighted row sum" holds for the certificate's weights.** Statement precision. With
v = (I − M/λ)⁻¹1, whose components are all at least 1, the floored entries add at most 2.34×10⁻⁴⁸, and θ itself rises
by 5.5×10⁻⁵⁰. With an approximate Perron vector the rows that hold only floored entries get weights near the floor,
and the floored share of their row sums is not small, while θ is unchanged to 10⁻⁵⁰. *Proposed:* add "with the
weights of the certificate" to the sentence of §13.4.

**F5. The saved floor is a double.** Observation, no action. The double 1e-50 exceeds the decimal 10⁻⁵⁰ by a relative
7.6×10⁻¹⁸, so the saved floor entries are not below the floor that the run used by more than rounding, and
`lipcheck_file`'s inflation covers every saved entry.

## 8. What I did not check

- That an unfloored entry bounds the true Lipschitz constant: the induction over the bound algebra, the convexity of
  the candidates and the ball-mode secant of B_chain.md §13.4. The floor inherits that claim and adds nothing to it.
  `b8m_lc_numcheck.py` tests the invariance bounds numerically and does not test the matrix.
- The closure, the atoms and the rest of lemma LC, and lemma J and Theorem B(1), which use θ only through θ < 1.
- Any box other than the certificate's (K = 4, L₀ = 73.89, r_A = r_W = 2×10⁻⁶), and any floor other than 10⁻⁵⁰.
- Windows and aarch64. The runs here are on the Mac, with python-flint 0.9.0 and FLINT 3.6.0.
- Why the closure's h_β moves in the tenth digit with a 10⁻¹⁷ move of the centre. I checked only that it is the
  largest change in the record, 200 times the next.

## 9. Reproduce

From `solver/proof/`, one process, each run under a time limit. D is a directory outside `solver/proof`.

```
python3 b8m_lc_liprep.py --out=D                      # exit 0, 4 min: lip.json, lip_dA.json
python3 b8m_lc_liprep.py --out=D --mutate=nofloor     # exit 1, 4 min: lip_nofloor.json, lip_nofloor_dA.json
python3 review_894/rv_floor.py --dir=D                # K1 to K5: RV_FLOOR: PASS, exit 0, seconds
python3 review_894/rv_floor.py cap --out=D/cap --dA=1e-17     # the mutated runs, 4 min
python3 review_894/rv_floor.py --dir=D --floored=D/cap/lip_cap.json    # K1 fails: exit 3
python3 b8m_lc_liprep.py --compare=D/cap/lip_cap.json,D/cap/lip_cap_dA.json   # PASS, exit 0: not noticed
python3 b8m_lc.py --lipcheck=D/cap/lip_cap.json       # exit 0, theta = 0.0000: not noticed
```

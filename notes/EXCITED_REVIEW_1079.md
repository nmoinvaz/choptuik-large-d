# EXCITED_REVIEW_1079: independent review of the two excited M = 3 zeros' certificates (#1079)

A review of the claim of issue #1079: two zeros of the three-harmonic inner selection problem, besides the fundamental
one, meet every hypothesis (H0)–(H3) of Theorem A(M) ([proof/A_lemmas_M.md](../proof/A_lemmas_M.md) §6), so each is a
certified, simple zero of the M = 3 truncation, assuming Braaksma's theorem. Written 2026-11-23 by Claude (Anthropic)
under Nathan Moinvaziri's direction, as a separate agent. Status words as in AGENTS.md.

What was reviewed:

- branch `feat/certify-excited` (1addcada, 89553c6b, c40fbc77): the options of `certify_M.py`,
  `lemma6_sector_M.py --x`, `theorem_A_sets_M.py --cert --ray`;
- branch `feat/lean-excited` (18550de1): the Lean instances `M3exc1`, `M3exc2` and their controls;
- the runs in the raw-data repository, `I1079-macm3/` (macOS) and `I1079-dh610/` (Linux), at raw 143d4d5b and
  bab3a952;
- the result comments of #1079, which I read before starting. The checks below share no script with the runners'
  chains, except where a line says that a stored command was rerun.

The review's scripts are in `solver/proof/review_1079/`. Its own runs are raw output, in `I1079-mac-review/` of the
raw-data repository. Machine: Apple arm64, macOS 27.0.1, Python 3.13.15, python-flint 0.9.0, FLINT 3.6.0, numpy 2.5.4
on Accelerate.

Theorem A(M) is §6 of A_lemmas_M.md. The issue's comments and `certify_M.py` line 90 cite it as §8, which is the
mutation table (finding 21).

## Verdict

**Confirmed, with three corrections to the record and eight gaps.** No error was found in the claim itself. At both
zeros the Krawczyk step passes and (H0)–(H3) hold, each tied to the same centre and the same box by a check that can
fail. Two corrections concern the margin gap between machines (findings 16 and 17), and one a section number (21).
The gaps are in the options, in what the files record, in two controls and in the theorem's wording.

| statement of #1079 | after the review | findings |
|---|---|---|
| the options leave main's stored results unchanged when unset | confirmed. One printed line of `theorem_A_sets_M.py` changes | 1, 5 |
| the options cannot give a wrong result | not so: `CERTIFY_M_NJAC` above `CERTIFY_M_NTAY` + 1 is wrong without an error. The runs at 20/12 and 28/16 are sound | 2 |
| the Krawczyk step passes at both zeros | confirmed: rerun here, cross-loaded both ways, decided by Lean | 6, 13, 14, 15 |
| (H0), with r₀ = 0.0741 at every centre | valid as run, and proved a second time by a separate call | 7 |
| (H1), lemma S and lemma 6b₀ over p̂ ± 10⁻⁶ | confirmed, rerun here | 8 |
| (H2), 183 discs over p̂ ± 10⁻¹¹ and the inclusion | confirmed. One cited control, `c1check`, does not pass its own mutation at this box | 9, 10 |
| (H3), ρ₆ₐ = 0.01365 and 0.01259 | confirmed from the stored enclosure. The `coords` mutation is not caught there | 11 |
| the quoted numbers | 89 of 89 entries agree with the raw files | 12 |
| the Linux margins are smaller because the box run's columns widen from last-digit differences in the start data | not supported. The frames of the Lohner sets carry it, two Macs differ as much, and the start data play no part | 16, 17 |

## Findings

Each is marked **error** (a statement that is wrong), **gap** (something missing that the claim does not rest on, or
rests on less than stated) or **confirmed**.

| # | mark | finding | where |
|---|---|---|---|
| 1 | confirmed | With the options unset, `path_for`, `orders`, R0, ARG and the precisions equal main's for M = 1, 3, 5, and the only code that differs from main is the option lines | §1.1 |
| 2 | gap | `CERTIFY_M_NJAC` ≥ `CERTIFY_M_NTAY` + 2 gives a wrong remainder of the step Jacobian, with no error. `CERTIFY_M_NTAY=10` alone does it, since NJAC stays 12 | §1.2 |
| 3 | gap | The box and point files do not record the Taylor orders. `krawczyk --point` compares p and M of the two files and not their paths, so a point run at ξ₁ = 0.2 would be accepted with a box run at 0.3. `CERTIFY_M_XI1` below 0.2 is not refused | §1.3 |
| 4 | confirmed | The options also reach the M = 5 certificate (`certify_M2.py` takes `CM.orders` and `CM.path_for`). Unset, nothing changes there | §1.1 |
| 5 | gap | Unset, `theorem_A_sets_M.py --M=3` now prints the certificate's absolute path where main prints `data/A_M_box_M3_1e-10.json`. The verdict and the numbers are the same | §1.1 |
| 6 | confirmed | The cited box and point files of each zero have the same centre, the path of Theorem A(M) with ξ₁ = 0.3, and rigorous start records | §2.1 |
| 7 | confirmed | (H0) is valid as run. The lemma holds for any r₀ at which its computation succeeds, and the path starts at that r₀. A separate call proves step (i) over p̂ ± 10⁻⁹ and its Cauchy box with the certificate's N | §2.2 |
| 8 | confirmed | (H1): lemma S(3), the old sector lemma and the flat estimate hold over p̂ ± 10⁻⁶ at both centres | §2.3 |
| 9 | confirmed | (H2): each covering has 183 certified discs at its own centre, its check passes and fails with a tile dropped, X_c's hull lies in X_b and X_b in X₆ | §2.4 |
| 10 | gap | `c1check`, cited as a control of (H2), does not pass its own mutation at h = 10⁻¹¹: with the derivative columns dropped, the points moved in ω/ω∞ stay inside at rows 1 and 6 | §2.4 |
| 11 | confirmed | (H3): ρ₆ₐ follows from the stored backward enclosure, and the lemma 6a input is the export of the cited box. The `coords` mutation gives ρ₆ₐ = 0.28 and 0.17, below 1 | §2.5 |
| 12 | confirmed | Every number quoted in the result comments that I entered (89) is in the raw files | §3 |
| 13 | confirmed | The four Lean files regenerate byte for byte from the raw box and point files | §4 |
| 14 | confirmed | The Lean project builds. The widened controls are decided false. The ten new theorems use propext, Classical.choice and Quot.sound only | §4 |
| 15 | gap | The #1077 review harness (`regen_check.sh`, `reproduce.py`, `Axioms.lean`, `Mutations.lean`) was not extended to the new instances. At the second zero a transposed YJ still passes `check` | §4 |
| 16 | error | The stated cause of the margin gap is not supported. Last-digit changes of the start data leave the margins unchanged. A 1-ulp change of the frames Q moves them by as much as the machines differ, and this Mac differs from macm3 as much as dh610 does | §5 |
| 17 | error | "Their radius is 1.8× (first zero) and 8.5× (second) wider on Linux" names the variational columns. Those ratios are of the largest radius of Y·Dg̃(X). The columns' radii are 0.98× and 1.66× | §5 |
| 18 | gap | The margins are not a property of the zero and the box. A rerun elsewhere can land anywhere in the range seen, and its `controls` can report C6 as unexpected | §5 |
| 19 | gap | Theorem A(M)'s hypotheses are written with the fundamental zero's sets (x̂ ± 10⁻⁶, p̂ ± 1.1×10⁻⁹). At the excited zeros they hold with each zero's own centre and covering. The text needs that reading stated | §2.6 |
| 20 | gap | `lemma6_sector_M.py --json` and `lemma6_jump_M.py --json` do not record the centre given by `--x`. The default-order point files that comment 7104591124 cites are not in the raw repository | §3 |
| 21 | error | Theorem A(M) is §6 of A_lemmas_M.md, not §8 (`certify_M.py` line 90 and the issue's comments) | above |

## 1. The options (check 1)

### 1.1 Unset, the code runs what main runs

`solver/proof/review_1079/defaults_check.py`, from `solver/proof`:

```
python3 review_1079/defaults_check.py              # DEFAULTS CHECK: PASS, exit 0
python3 review_1079/defaults_check.py --mutate=env # CERTIFY_M_NTAY=28 in the "unset" run: FAIL, exit 1
```

- Static part. Of the 273 entries of `origin/main:solver/proof`, three Python files differ in the branch. Compared
  unit by unit (ast, comments and docstrings ignored), the code that differs is `(NTAY, NJAC)` and `path_for` in
  `certify_M.py`, `main` in `lemma6_sector_M.py`, and `check`, `verdict`, `selftest`, `main` in `theorem_A_sets_M.py`.
- Dynamic part. Both versions of `certify_M` are imported in subprocesses with the `CERTIFY_M_*` variables removed.
  `path_for(M)`, `orders(M)`, R0, ARG, PREC and KPREC are equal for M = 1, 3, 5. With the options set to 28, 16
  and 0.2, M = 1 is unchanged and M = 3 and 5 change.

So the certificates of record are untouched (finding 1). Four remarks:

- `NTAY` and `NJAC` are read once, at import. `CERTIFY_M_XI1` is read at every call of `path_for`. Within one process
  both are fixed unless the process changes its own environment, which no script does.
- The options reach M = 5 as well (finding 4): `certify_M2.py` lines 150, 240 and 243 call `CM.path_for` and
  `CM.orders`. The comment in `certify_M.py` says "M >= 3", which covers it.
- A checkpoint cannot be resumed under other settings: the key of `box_run` holds the path and the orders, and
  `certify_M_lohner.integrate` refuses another key.
- `theorem_A_sets_M.py --M=3` with no option prints one line differently (finding 5):
  `c against the certificate's p^ [<absolute path>/data/A_M_box_M3_1e-10.json]`, where main prints
  `[data/A_M_box_M3_1e-10.json]` (raw `I1079-mac-review/logs/theorem_A_sets_M3_default_{main,branch}.log`). The path
  depends on the machine, so two stored logs no longer compare equal. Printing `CERT[M]` when `--cert` is not given
  restores main's line.

### 1.2 An order of the step Jacobian above the trajectory's is wrong without an error (finding 2)

`LohnerM._step` (`certify_M_lohner.py` lines 144–155) forms the remainder of the step Jacobian from the context of
`taylor_Y(.., N + 1)`, whose series have N + 2 terms, truncated only when NJ < N. Its docstring states NJ ≤ N. Nothing
checks it. For NJ ≥ N + 2 the coefficients A_k with k ≥ N + 2 are missing and are read as zero.

```
python3 review_1079/nj_guard_test.py     # NJ GUARD TEST: AS PREDICTED, exit 0
```

At one point of the start ray, the coefficient V_(NJ+1) from that line against the one from a context of the right
length:

| N, NJ | agree | largest certain difference |
|---|---|---|
| 8, 6 | yes | 0 |
| 8, 8 | yes | 0 |
| 8, 9 | yes | 0 |
| 8, 10 | **no** | 3.7×10⁵ |
| 8, 12 | **no** | 1.8×10¹⁰ |
| 10, 12 | **no** | 1.3×10⁶ |
| 20, 12 (default) | yes | 0 |
| 28, 16 (the excited runs) | yes | 0 |

The two environment variables are read independently, so `CERTIFY_M_NTAY=10` alone runs N = 10 with NJ = 12. The
result would be called rigorous and would not be an enclosure. Neither setting in use is affected. Proposed:
`assert NJ <= N` in `orders()` or in `start_set`.

### 1.3 What the files record, and what `krawczyk` compares (finding 3)

- A box or point file has the keys C, M, h, kind, p, path, prec, sec, sets, start, steps, w (and g, Dg for a point
  run). The orders are only in the checkpoint key. The step counts do not tell them apart either: the default runs and
  the 28/16 runs both took [726, 397, 57] and [718, 394, 64] steps. dh610 had to read the orders from the environment
  of sibling processes. What does identify them after the fact is the radius of g(p̂): 5.3×10⁻¹⁴ at 20/12 against
  1.571×10⁻²¹ at 28/16 for the first zero, and dh610's point run has 1.571×10⁻²¹ too.
- The path is recorded, ξ₁ included. `krawczyk_test` asserts `pt["p"] == bx["p"]` and `pt["M"] == bx["M"]`
  (`certify_M.py` line 355) and nothing about the path. Before the option there was one path per M. Now a point run at
  ξ₁ = 0.2 and a box run at 0.3 are two different functions g, and the test would combine them:

  ```
  python3 review_1079/defaults_check.py pathmix BOX.json POINT.json     # PATHMIX: NOT REFUSED
  ```

  In the stored runs this did not happen. The cited files all record ξ₁ = 0.3 (§2.1), and the ξ₁ = 0.2 runs (E1, E3)
  have another centre in the last digits, which the assertion on p would have caught.
- `CERTIFY_M_XI1` below 0.2 gives a real leg that runs inward. Theorem A(M) asks for ξ₁ ≥ ρ_arc. It is not refused.
- An order of the Taylor expansion does not affect validity (given §1.2), so the missing record is a matter of
  reproduction, not of proof. Proposed: write `"N": [NT, NJ]` into the result, and assert equal paths in
  `krawczyk_test`.

## 2. The hypotheses (check 2)

`review_1079/sets_check.py` reads the raw files of one zero and tests that the pieces belong together. Nothing is
integrated. From `solver/proof`, with RAW the raw-data clone:

```
python3 review_1079/sets_check.py RAW 1              # SETS CHECK (zero 1, mutation None): HOLD
python3 review_1079/sets_check.py RAW 2              # HOLD
python3 review_1079/sets_check.py RAW 1 --selftest   # the clean check and five mutations: PASS (and for 2)
```

Its mutations each fail in the test they aim at: the other zero's covering (centre, hull), the other zero's lemma 6a
input (centre, sixa), R₊ = 0.0099 (g0), the point file's ξ₁ relabelled 0.2 (path), X_b of half-width 10⁻¹² (hull).

### 2.1 One centre, one path (finding 6)

- The box file's p, the point file's p, the lemma 6a input's x and p, the covering's `ray.json` x and the centre
  printed by the H1 run are the same four decimal strings, for each zero.
- Box and point files have the path (0.0741, 5π/16, 0.2, 0.3), start records marked rigorous, and the same
  `fixed_point_M.py` hash 589e5a810bc50289.
- The path is of the kind Theorem A(M) allows: the ray's angle 5π/16 exceeds 3π/8 − 0.2 by 0.0037, r₀ = 0.0741 ≤ 0.1,
  ρ_arc = 0.2, and the start point lies in the covering's annulus (1/r₀² = 182.1 ≤ 260).

### 2.2 (H0): the start-value lemma at r₀ = 0.0741 (finding 7)

The question put to the review: the certificates use r₀ = R0[3] = 0.0741 at every centre, while `fixed_point_M.py`'s
table takes r₀ = `r0_inner_np(M, ω/ω∞)`, which is 0.0982 and 0.1068 at the excited centres. Is (H0) valid as run?

It is.

- Lemma FP(M) (A_lemmas_M.md §2.1) is stated for any r₀ > 0, any ray θ ∈ (π/4, 3π/4) and any parameter set for which
  `run_M` succeeds. `r0_inner_np` is the double-precision solver's start radius. The table uses it to test that ray. It
  is not a hypothesis of the lemma.
- `certify_M.start_data` passes `path[0]` = R0[M] to `start_data_M`, and `legs_for(path)` starts the ray at the same
  `path[0]`. So the start values are those of the point where the integration starts.
- `start_data_M` runs the fixed point on its set and on that set's Cauchy box and raises unless both hold
  (`fixed_point_M.py` line 462). The certificate calls it at p̂ and over the hull of the box, so step (i) is proved on
  the four start sets, as §6 says of every certificate that ran. `start_cover_M2.py`, the second look of §7, compares
  with stored rows around the fundamental zero and does not apply here.

A second look by a separate call, `review_1079/h0_recheck.py BOX.json`, on X = p̂ ± 10⁻⁹ and on its Cauchy box (real
parts ± (10⁻⁹ + 10⁻³), imaginary parts ± 10⁻³), which contain the four sets:

| | first zero | second zero |
|---|---|---|
| N (optimal at p̂), k | 95, 40 | 94, 40 |
| contraction at p̂, over X, over the Cauchy box | 0.3805, 0.3805, 0.3814 | 0.3892, 0.3892, 0.3901 |
| sup\|η\| at p̂, here and in the box file | 1.508017×10⁻⁷², the same to 5×10⁻¹⁶ | 6.336883×10⁻⁷², the same to 1.4×10⁻¹⁵ |
| sup\|η\| over X, over the Cauchy box | 5.9×10⁻²⁷, 1.4×10⁻²⁰ | 9.2×10⁻²⁸, 2.3×10⁻²¹ |
| the hull inside X; diag side condition 0.01147 ≤ 3/2 | yes; yes | yes; yes |
| verdict | PASS | PASS |

The equal η at p̂ shows that the certificate's N is this N. The three mutations fail as they must, at both zeros: one
series coefficient off by 10⁻⁶ (the structural orders fail), r₀ = 0.3 (contraction 21 and 20), and the hull against a
box of half-width 10⁻¹³.

### 2.3 (H1): lemma S, lemma 6b₀ and X₆ (finding 8)

- The stored run is `lemma6_sector_M.py --M=3 --rp=0.1 --k=20 --box=1e-6 --Z=0.004489 --x=p̂`. `--x` replaces
  `FPM.ZEROS[M]`, which is the only centre `xs_of` reads, so lemma S, the old sector lemma and the flat estimate are
  all at p̂.
- Rerun here, it passes at both centres with the stored contractions (lemma S 0.6961 and 0.6782, old sector lemma
  0.6839 and 0.7008), and `--mutate=series` gives "lemma S holds: False".
- Lemma 6b₀(3) needs lemma S at R₊ and the old sector lemma on 3π/8 ± 0.2 up to 0.1, both over a box that contains
  X_c. Both are in this run, over X₆ = p̂ ± 10⁻⁶ in all four parameters.
- N is 62 at both centres (53 at the fundamental zero): the optimal order follows the centre.

### 2.4 (H2): the coverings, X_b and the inclusions (findings 9, 10)

- `lemma6b1_cover_M.py --x` sets `FPM.ZEROS[M]` before anything reads it: the start data (`start_data_u`), the ω and A
  balls, the C¹ columns and `ray.json`'s x. Each row asserts that `ray.json`'s x is its own centre.
- The covering's own start set is proved in its run by the old sector lemma at p̂ and on the complex box
  p̂ ± (10⁻¹¹ + 10⁻³). The run raises otherwise.
- G₀'s radius 257 is a constant of the covering. It needs lemma S at R₊ = 0.01 over a box that contains X_b, which is
  (H1)'s run: 257 ≥ 1/(R₊ sin 0.4) = 256.79, the first row 262 > 257, and X_b ⊂ X₆ with slack 9.9999×10⁻⁷.
  `start_cover_lemmas_M.py`, which checks these uses for the fundamental zero, has a fixed table of runs and was not
  run here. `sets_check.py` tests the same three facts.
- Each stored covering has the design's 12 rows, 183 discs, none uncertified, all at h = 10⁻¹¹.
- X_c's hull has half-widths (1.000, 1.000, 1.641, 0.163)×10⁻¹² and (1.000, 1.000, 1.722, 1.365)×10⁻¹². It lies in
  X_b = c ± (10⁻¹¹ + 1.4×10⁻²⁰) with slack 8.359×10⁻¹² and 8.278×10⁻¹² at B₃, formed here without
  `theorem_A_sets_M.py`.

Reruns of stored commands on this Mac (raw `I1079-mac-review/light_steps.txt`):

| command, on macm3's stored covering | first zero | second zero |
|---|---|---|
| `check` | PASS, 183 tiles | PASS, 183 tiles |
| `check --drop=5,6` (not run before) | FAIL, 3 failures | FAIL, 3 failures |
| `row --j=11` from the stored ray state | 12 discs certified, Λ within 6×10⁻¹⁵ of macm3's, box radii equal | 12 discs certified |
| `theorem_A_sets_M.py --cert --ray`, and `--selftest` | HOLD, PASS | HOLD, PASS |
| the same against the other zero's covering | FAIL | FAIL |
| `c1check` | True | True |
| `c1check --mutate=nov` (not run before) | **exit 3: must be True, is False** | **exit 3** |

`c1check` is a planted truth in double precision: y⁺ at the centre and at p̂ ± h eᵢ must lie in the saved C¹ sets,
inflated by a fixed 10⁻¹¹ for the double-precision error. Its mutation drops the derivative columns, and then every
moved point must fall outside. At the fundamental zero h is 1.1×10⁻⁹. Here h = 10⁻¹¹ equals the inflation, and the
points moved in ω/ω∞ stay inside at rows 1 and 6 (they fall outside at rows 9 and 11, and in the other three
parameters at every row). So at this box the control cannot see a missing ω/ω∞ column at the outer rows
(finding 10). It is a control, not a step of the proof: the discs are certified on the rigorous sets. A tolerance
proportional to h, or the check at the 1.1×10⁻⁹ box, would restore it.

### 2.5 (H3): lemma 6a (finding 11)

- The lemma 6a input is the export of the cited box: `certify_M.export_6a` run here on the box file gives the stored
  state box string for string (`sets_check.py`, test sixa). Its ξ₁ is 0.3, the box file's.
- Its parameter box is x ± h with x = p̂ and h = 10⁻¹², in ω/ω∞ and A. That is right: the tilt T is the identity in the
  two outer coordinates, so they range over exactly ± h, and the flow does not depend on B₃ and C₃, which enter the
  start values only.
- Lemma S and the flat estimate are proved in the run over p̂ ± 10⁻⁶ (`--x`), and ξ_a = 0.067 lies in lemma S's real
  reach.
- From macm3's stored backward enclosure B (`--load-B`), rerun here: ρ₆ₐ = 1.365×10⁻² and 1.259×10⁻², and `partial`
  is caught ("G_slow is 9 x 11").
- `--mutate=coords` (not run before; the wrong mode normalization at ξ_a) gives ρ₆ₐ = 0.2824 and 0.1727, both below
  1. So this mutation is not caught at the excited zeros. At the fundamental zero's certificate it is caught (1.45,
  A_M_review.md F8), and at M = 1 it is not (0.12). A_lemmas_M.md §8 lists it as report-only and names what guards the
  normalization: `lemma6_usys_M.py`'s test (c), which does not depend on the centre.

### 2.6 The theorem's text (finding 19)

§6 states the hypotheses with the fundamental zero's sets: X₆ = x̂ ± 10⁻⁶ with x̂ the double-precision zero of §2.4,
X_b = p̂ ± 1.1×10⁻⁹, and "x̂ ± 10⁻⁶ at M = 3" in (H3). The proof uses only that lemma S and lemma 6b₀ hold on a box
that contains X_c, that a complete covering holds on a box that contains X_c, and that lemma 6a holds with its own
lemma S. All three are met at each excited zero with its own centre. So the claim is right, and the text of §6 must say
"a centre x̂" before the excited zeros are cited under it. Proposed wording is in §7 below.

The caveat of A_lemmas_M.md's opening carries over unchanged: the majorant code of lemma S and lemma 6b₀ and the
M = 3 disc certificates have not had an independent review.

## 3. The quoted numbers (check 3, findings 12, 20)

```
python3 review_1079/check_quoted.py RAW              # CHECK QUOTED: 89 of 89 entries agree: PASS
python3 review_1079/check_quoted.py RAW --selftest   # every entry's mutation fires: PASS
```

The entries are the margins of the four passing boxes, the controls' verdicts and the values quoted for C5 to C8, the
recheck and its mutation, the zero enclosures, the radii of g(p̂) at the default orders and at E1, E2, E3, η, the
contractions and Λ of (H1) with its mutation, ρ₆ₐ with its two mutations, the disc counts, the gluing counts, the
inclusion's slack and selftest, and dh610's margins, controls, cross-load, ρ₆ₐ, check and slack. Each mutation changes
the last digit of the expected value, and each is then reported as not found.

Two things are missing from the record (finding 20):

- The JSON written by `lemma6_sector_M.py --json` and by `lemma6_jump_M.py --json` has no centre. With `--x` the
  first prints it in its log, and the second prints nothing. The contractions identify the centre after the fact.
- Comment 7104591124 cites `data/A_M_point_M3x{1,2}.json`, the default-order point runs. They were never committed to
  the raw repository. Their radii, 5.337×10⁻¹⁴ and 1.188×10⁻¹⁴, are in `I1079-macm3/steps.txt` and the logs.

## 4. Lean (check 4, findings 13–15)

From a checkout of `feat/lean-excited` (18550de1), with the exporter of that branch and the raw files at 143d4d5b
(the same blobs at raw HEAD):

- `export_krawczyk.py certM BOX M3excN OUT --point=POINT --proof=solver/proof --label=...`, and the same with
  `--widen=1000000` for the controls: `M3exc1.lean`, `M3exc1Wide.lean`, `M3exc2.lean` and `M3exc2Wide.lean` are
  **identical, byte for byte**, to the branch's files. With `--label` no path enters the file. The sha256 in the
  headers are those of the raw files.
- `lake build`, one module at a time and then the whole target (`review_1079/lean_build.sh`): every module builds,
  with `warningAsError`. Lean 4.34.1, Mathlib d13f23b7.
- `M3exc1Wide_check_fails` and `M3exc2Wide_check_fails` are theorems of the build, so the widened controls are
  decided false.
- `#print axioms` for `M3excN_check`, `M3excN_unique_simple_zero`, `M3excNWide_check_fails`, `M3excN_detY` and
  `M3excN_nonvacuous`, N = 1, 2: each depends on **propext, Classical.choice, Quot.sound** and nothing else
  (`review_1079/Exc1079.lean`, raw `I1079-mac-review/lean/lean_build.log`).
- `Exc1079.lean` also decides: no interval of either instance is empty; the half-widths are (10⁻¹², 10⁻¹², 10⁻¹³,
  10⁻¹³) and (10⁻¹², 10⁻¹², 10⁻¹², 10⁻¹²); a zero planted outside in the first or third unknown fails at both; a zero
  planted inside passes; a transposed YJ fails at the first zero. Each verdict was predicted first
  (`review_1079/lean_predict.py`). `Exc1079MustFail.lean` holds three false statements and is rejected, exit 1 with
  three errors.

What the Lean theorems say is what #1077 found for `M3cert`: for every C¹ map whose Y g(0) and Y Dg over the box lie
in the file's intervals, there is exactly one zero in the box and the Jacobian is invertible there. That the map of
#1079 has those enclosures is `certify_M.py`'s work and stays a hypothesis (`EnclOK`).

Finding 15:

- `lean/review/regen_check.sh` and `reproduce.py` list eight instance files. `Axioms.lean` and `Mutations.lean` name
  no `M3exc` instance. The branch adds the instances and the non-vacuity theorems only.
- At the second zero all four half-widths are equal, and a transposed YJ still passes `check` (largest |K_i|/w_i
  0.073; decided by Lean in `Exc1079.lean`). As #1077 found for Y, only the comparison with the source catches such a
  transcription error there, which is the byte comparison above.

## 5. The margin gap between machines (check 5, findings 16–18)

The record says: dh610's margins in A and ω/ω∞ are 18 to 44% below macm3's because the box run's variational columns
widen on the real leg from last-digit float64 differences in the start data between macOS and Linux.

The test (`review_1079/perturb_box.py`, written into `I1079-mac-review/README.md` before the runs): the box run of each
zero on this Mac at 28/16, from macm3's refine file, with one last-bit perturbation planted, then
`krawczyk --box=<this run> --point=<macm3's point file>`. Each run is as rigorous as an unperturbed one: Q is any
invertible float matrix used exactly, and the Perron weights are any positive numbers.

<!-- perturb table -->
First zero, box (h, w) = (1e-12, 1e-13):

| box run | passes | margins (A, ω/ω∞, B₃, C₃) |
|---|---|---|
| macm3, stored | yes | +0.702, +0.758, +0.997, +0.998 |
| dh610, stored | yes | +0.468, +0.619, +0.997, +0.998 |
| this Mac, unchanged code | yes | +0.610, +0.721, +0.998, +0.998 |
| this Mac, frames Q moved by 1 ulp, seed 1 | yes | +0.624, +0.731, +0.998, +0.998 |
| this Mac, frames Q moved by 1 ulp, seed 1079 | yes | +0.453, +0.608, +0.997, +0.998 |

5 box files, 5 pass; A margin from +0.453 to +0.702, ω/ω∞ margin from +0.608 to +0.758; smallest margin of any coordinate +0.453.

Second zero, box (h, w) = (1e-12, 1e-12):

| box run | passes | margins (A, ω/ω∞, B₃, C₃) |
|---|---|---|
| macm3, stored | yes | +0.943, +0.945, +0.988, +0.995 |
| dh610, stored | yes | +0.711, +0.528, +0.981, +0.992 |
| this Mac, unchanged code | yes | +0.632, +0.398, +0.983, +0.993 |
| this Mac, unchanged code, repeated | yes | +0.632, +0.398, +0.983, +0.993 |
| this Mac, start lemma's weights moved by 1 ulp | yes | +0.632, +0.398, +0.983, +0.993 |
| this Mac, start lemma's weights moved by 4096 ulp | yes | +0.632, +0.398, +0.983, +0.993 |
| this Mac, frames Q moved by 1 ulp, seed 1079 | yes | +0.942, +0.943, +0.987, +0.995 |

7 box files, 7 pass; A margin from +0.632 to +0.943, ω/ω∞ margin from +0.398 to +0.945; smallest margin of any coordinate +0.398.

<!-- end perturb table -->

The table is written by `review_1079/fill_note.py` from the logs (raw `I1079-mac-review/logs/`).

What this shows:

- **Not macOS against Linux.** This Mac and macm3 are both Apple arm64 machines with numpy 2.5.4 (on Accelerate here;
  macm3's LAPACK is not recorded), and their margins differ by as much as dh610's do. My prediction, written before
  the run, was that this Mac would reproduce macm3's file. It did not.
- **Not the start data.** Moving the start lemma's Perron weights by 1 ulp, or by 4096 ulp, changes the start records
  in their last digits, as between the machines, and leaves all four margins unchanged to the three digits printed.
  The start radii are 10⁻⁷² in the values and 10⁻¹⁷ in the columns, against a state radius of 10⁻¹³.
- **The frames.** `validated_int.qr_pivot` takes Q from numpy's float QR of the propagated frame, with the columns
  sorted by norm. Moving every entry of every Q by 1 ulp moves the weakest margin of the second zero from 0.398 to
  0.943. The runs print the same radii, to the digits printed, along the ray and most of the arc. They part near the
  end of the arc and on the real leg, where the derivative radius grows a thousandfold.
- **Deterministic on one machine.** The unperturbed run repeated gives the same margins.
- **The columns are not what widens** (finding 17). In dh610's `compare_json.txt` the largest radius of the box
  Jacobian's columns is 9.26×10⁻⁵ against macm3's 9.46×10⁻⁵ at the first zero (0.98×) and 1.05×10⁻⁴ against 6.33×10⁻⁵
  at the second (1.66×). The ratios 1.8 and 8.5 that the closing comment gives for the columns are those of the
  largest radius of Y·Dg̃(X): 0.407 against 0.228, and 0.271 against 0.032. dh610's own report has this right. What
  differs is the shape of the Lohner sets, the frame QV and the box qV, to which the product (Yℓ)QV is sensitive
  because Yℓ cancels the leakage to five or six digits.

Does it matter for the claim? No, in one respect, and yes in another.

- The stored files are proofs wherever they are loaded. macm3's files give macm3's margins on dh610 and on this Mac,
  and dh610's files give dh610's margins on this Mac (+0.468, +0.619, +0.997, +0.998 and +0.711, +0.528, +0.981,
  +0.992; `recheck` exit 0 and its mutation exit 1). So the cross-load passes three ways, and the rule of #111 is met.
- But the margin is not a number that belongs to the zero and the box (finding 18). It is one draw from a range that
  a last bit selects. Every draw seen passes. A rerun on another machine "passes on its own bound" only if its draw
  does, and its `controls` exit 1 whenever the A margin falls below 0.5 (C6, as with dh610's file of the first zero).
  The text should quote the margins as "of the stored file". The 1% bar of the Linux task was not one a second machine
  could be expected to meet.

The box file alone is a certificate as well: `krawczyk --box` without `--point` gives the same margins to three
digits at both zeros, and that is the test `certify_M_recheck.py` mirrors.

## 6. What was not checked

- What the review's checks share with the work: the proof code itself (`fixed_point_M.run_M`,
  `certify_M.krawczyk_test` and `export_6a`, the covering's design and check) and python-flint. What they do not share:
  the runners' chains, `start_cover_M2.py`, `theorem_A_sets_M.py` (the hull and the inclusions are formed again in
  `sets_check.py` and `h0_recheck.py`), and the Lean kernel's arithmetic.
- The validated integrator, `fixed_point_M.py`'s majorants, the disc lemma and the gluing check were not re-derived.
  They are the fundamental zero's code, unchanged, and their reviews are A_M_review.md and #39.
- No second implementation of the Krawczyk step at the excited zeros (CAPD or other). The Lean instances decide the
  last arithmetic step only.
- The 183 discs were not rerun, except row 11 of each covering.
- Whether these zeros persist as M grows is outside the claim.

## 7. Proposed changes

Code (for the branch's author; the review changed no file under review):

1. `certify_M.py`: refuse NJ > N; record `"N": [NT, NJ]` in every result; refuse `CERTIFY_M_XI1` below ρ_arc.
2. `krawczyk_test`: assert `pt["path"] == bx["path"]`.
3. `theorem_A_sets_M.py`: print `CERT[M]` when `--cert` is not given.
4. `lemma6_sector_M.py`, `lemma6_jump_M.py`: write the centre into the JSON, and print it in the second.
5. `c1check`: a tolerance that scales with h.
6. `certify_M.py` line 90: "A_lemmas_M §6".
7. `lean/review/`: the two instances in `regen_check.sh`, `reproduce.py`, `Axioms.lean`, `Mutations.lean`.

Text, for A_lemmas_M.md §6–§7 (proposed, not applied: agents do not edit `proof/`):

> Let x̂ be a centre: the double-precision zero of §2.4, or the centre p̂ of another certificate. […] (H1) X_c ⊂ X₆ =
> x̂ ± 10⁻⁶, the box over which lemma S(M) and the old sector lemma are proved (`lemma6_sector_M.py --box=1e-6`, with
> `--x` for another centre); (H2) X_c ⊂ X_b, the box of a complete 6b₁ covering run at its own centre.

> **The two excited zeros at M = 3** (#1079). Centres p̂ = (3.446218246366727…, 0.347606965217407…) and
> (3.526473056695715…, 0.211497981391215…). With `CERTIFY_M_NTAY=28 CERTIFY_M_NJAC=16` the Krawczyk test passes on
> the boxes (h, w) = (10⁻¹², 10⁻¹³) and (10⁻¹², 10⁻¹²). The margins of the stored files are +0.702, +0.758, +0.997,
> +0.998 and +0.943, +0.945, +0.988, +0.995. They are properties of those files: reruns on two other machines and with
> the Lohner frames moved by one unit in the last place gave A margins between RANGE_A1 and RANGE_A2, every one
> passing. (H0) by the certificates' own start-data calls and by a second call over p̂ ± 10⁻⁹ (N = 95 and 94,
> contraction 0.38 and 0.39); (H1) lemma S 0.6961 and 0.6782, old sector lemma 0.6839 and 0.7008 over p̂ ± 10⁻⁶;
> (H2) 183 discs over p̂ ± 10⁻¹¹, the hull of X_c inside with slack 8.36×10⁻¹² and 8.28×10⁻¹²; (H3) ρ₆ₐ = 0.01365 and
> 0.01259. Status: *certified*, assuming Braaksma's theorem, with the trusted code as at the fundamental zero.

## 8. Reproduce

From `solver/proof` of branch `review/excited-1079`, with the raw-data repository cloned beside the checkout
(`git clone https://github.com/nmoinvaz/choptuik-large-d-raw ../choptuik-large-d-raw`) and
`RAW=../../../choptuik-large-d-raw`:

```
python3 review_1079/defaults_check.py                       # PASS
python3 review_1079/nj_guard_test.py                        # AS PREDICTED
python3 review_1079/check_quoted.py $RAW --selftest         # PASS
python3 review_1079/sets_check.py $RAW 1 --selftest         # PASS (and 2)
D=$RAW/I1079-macm3/levers/data
python3 review_1079/h0_recheck.py $D/A_M_box_M3x1E2_h1e-12_w1e-13.json     # H0 SECOND LOOK: PASS
python3 review_1079/h0_recheck.py $D/A_M_box_M3x2E2_h1e-12_w1e-12.json     # PASS
bash review_1079/light_checks.sh $RAW                       # writes $RAW/I1079-mac-review/light_steps.txt
bash review_1079/run_perturb.sh $RAW x2 qr 1079             # one box run, about 4 minutes
```

The scripts that write (`light_checks.sh`, `run_perturb.sh`) write into `$RAW/I1079-mac-review/`. Lean: copy
`review_1079/Exc1079.lean` and `Exc1079MustFail.lean` to `lean/review/` of a checkout of `feat/lean-excited`, then
`bash review_1079/lean_build.sh <that lean/> <log>` and `lake env lean review/Exc1079MustFail.lean` (must fail).

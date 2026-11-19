# QUADRATIC_FORMULA: a formula for Δ(D) from a quadratic equation

Written 2026-11-19 by a subagent of the coordinator on the Mac, branch `wip/quadratic-formula`. Scripts and their
outputs are in [`solver/quadform/`](../solver/quadform/). The question was Nathan's: look at new ways to find a formula
for Δ(D) "using quadratic base equations", and say whether that gets anything.

## Answer

**It gets a better formula, and no new structure.** With Δ the root of a quadratic equation whose three coefficients
are polynomials in ε = 1/(D − 2):

| What | Result | The number's kind |
|---|---|---|
| 11 constants, D 4 to 245.64 | every point within 1.8×10⁻⁹, against 5.1×10⁻⁸ for the best rational function with 11 constants on the same points | largest miss over the computed points, an interpolation error |
| 24 constants, D 3.04 to 67.77 | every one of 1235 points within 3.04σ (χ²/dof 0.18), where the cut-plane series needs 29 | misfit in units of each point's err |
| the same 24 constants, on points they never saw | the 46 quad-double values at D 67.8–73 within 1.1×10⁻¹⁵, the chain to D 245.64 within 2.4×10⁻¹¹ (0.6 of its err) | largest miss on held-out points |
| Δ∞ | 1.64389374513, inside the README's and track C's bars | post hoc, largest planted miss 7×10⁻¹¹, not a calibrated bar |
| a branch point at D = 3 | found from the points alone at D = 2.990 to 3.003 by all 18 forms | spread over forms |
| the negative D axis, cut or poles | not decided, the planted test failed | — |
| Δ at D = 3 | not determined, 1.40 to 2.03 | spread over forms |

Of the three frozen experiments, A (equal constants, held out) and C (a shorter all-D formula) pass on the real
points, and B (where the branch points are) is not decided. Three planted expectations failed, and section 3 says what
that removes. One discrepancy in `hp/data/delta_all.csv` turned up on the way (section 3).

## 1. Which reading

Three readings were considered. The first two already exist in the repository, so the third was pursued.

1. **Quadratization, then Taylor recursions.** Done. The field equations are exactly quadratic in
   (b, c, G, s, p, m, ω) with two algebraic unknowns ([RT_FORMULATION.md](RT_FORMULATION.md) 5(1)), and Reiterer and
   Trubowitz's form is quadratic from the start. The Taylor recursion in D through the residual, the "quadratic
   recast" of the asymptotic numerical method, is `hp/anm_run.py` ([LARGE_D_LITERATURE.md](LARGE_D_LITERATURE.md)
   2(d)). It gives the chain's series in D at a base, not a formula.
2. **An expansion about the D = ∞ base problem.** Done to second order. [FIRST_ORDER.md](FIRST_ORDER.md) and
   [SECOND_ORDER.md](SECOND_ORDER.md) derive Δ₁ and Δ₂ from the inner problem at 1/n and 1/n², in the truncation to
   M harmonics (M = 5: Δ₁ = 10.402 against the fitted 10.3753, Δ₂ = −50.06 against −49.1685). SECOND_ORDER.md 5
   already tried Δ∞ with Δ₁ and Δ₂ held fixed, and the derived values are too far off to help. The next step there is
   an untruncated first order, which is not a small experiment.
3. **What a quadratic system implies for the form of Δ(D): pursued.** A quadratic system, discretized by a Galerkin
   truncation (RT's Fourier × Chebyshev coefficients), is a set of polynomial equations in the unknowns and in n. By
   elimination, the Δ of such a truncation is an algebraic function of D. Its singularities are poles and algebraic
   branch points, and the generic one on a solution branch is a fold, a square root. The chain's own fold near D 259.9
   is one, with exponent 1/2 to 8.9×10⁻¹⁵ ([LARGE_D_WALL.md](LARGE_D_WALL.md), `hp/fold_mirror.py`). The continuum
   Δ(D) need not be algebraic. The matched approximant class is still the algebraic one, whose first member past the
   rational functions is the quadratic approximant[^shafer] below.

```math
P(\varepsilon)\,\Delta^2 + Q(\varepsilon)\,\Delta + R(\varepsilon) = 0, \qquad \varepsilon = \frac{1}{D-2},
\qquad \Delta = \frac{2R}{-Q + \sqrt{Q^2 - 4PR}} .
```

P, Q and R are polynomials of degrees (p, q, r), with p + q + r + 2 free constants. The repository's formulas are a
rational function of ε (the case P = 0), power series, and a series in a variable with a fixed square-root cut
([FORMULA_FORMS.md](FORMULA_FORMS.md)). Quadratic approximants were used here for the fold (`hp/fold_mirror.py`, M2)
and for the amplitude series ([PADE_FAR_END.md](PADE_FAR_END.md)), not for Δ(D) over all D. A quadratic approximant
finds its own branch points, the zeros of Q² − 4PR, which bears on FORMULA_FORMS.md's open question, a cut or poles on
the negative D axis. It also joins the README's two forms in one: analytic in 1/D at D = ∞, with a square root at a
finite D, as the README's √(D − 3) formula has below D = 4.

## 2. Aim and pass criteria (frozen before any fit of the real points)

Committed before `algfit.py` read a Δ value of `hp/data/delta_all.csv`. The code was developed on the plants only.

**Points.** `hp/data/delta_all.csv`, D ≤ 245.64 (the comparison frontier): kind dd with err_kind check (less the
points `hp/dd_fits_mp.clean` drops on D ≥ 10), dM and dM neighbours, kind qd, kind pinned, and kind chain_fit with
err_kind `fits d10`, `fits d6` and their `interp` rows. σ is `err` floored at 10⁻¹⁵. A D listed twice keeps the row
with the smaller err.

**One code path** for every class: an algebraic approximant Σₖ Aₖ(ε) Δᵏ = 0 of degree d in Δ, fitted by weighted least
squares in Δ (linearized iteration, then Gauss–Newton), in ball arithmetic at 1024 bits. d = 1 with A₁ = 1 is a
polynomial, d = 1 a rational function [L/M], d = 2 the quadratic approximant (p, q, r), the degrees of P, Q, R. N
counts the free constants: L + M + 1, and p + q + r + 2.

**Plants**, on the real D grid with noise N(0, σ), through the same path:

| plant | what it is | singularities |
|---|---|---|
| T_cut | the series of `hp/data/formula_cut.json` | square-root branch points at D = −7 and 3 |
| T_pow | Δ∞ + c (1 − ((D − 3)/(D + 7))^0.3), c set by the slope 10.375 | branch points at −7 and 3, exponent 0.3 |
| T_pole | the README's rational formula | poles at D = −3.2 and −11.0, no branch point |
| T_pole7 | Δ∞ + Σⱼ rⱼ ε/(1 − ε/εⱼ), poles at D = −2, −4, −6, −8, −10, −13, −17, residues 1, 1.5, 2, 2, 1.875, 1.25, 0.75 | seven poles, no branch point |

**A. Equal constants on the README formula's window.** Fit on 4 ≤ D ≤ 30.2, N = 9, 11, 13 (the README formula has
11), under two weightings, 1/σ and uniform. Forms: rational [L/M] with 1 ≤ M ≤ 4 and L ≥ M, quadratic (p, q, r) with
degrees within 3 of each other. Each class enters the form with the smallest in-sample sum of squares, so the
held-out points select nothing. The score is the largest |miss| on the held-out points 30.2 < D ≤ 245.64.

- *Pass A:* the quadratic's score is at most a tenth of the rational's at two or more of the three N, under both
  weightings.
- *Planted expectation:* T_cut passes A, T_pole and T_pole7 do not. T_pow is reported. If a pole plant passes, a pass
  on the real points means a better formula at equal constants and says nothing on cut against poles.

**B. Where the branch points are.** Fit all points 3.04 ≤ D ≤ 245.64 with quadratic forms whose degrees differ by at
most 1, N = 11 to 26. Accept a fit with χ²/dof ≤ 4 and no defect (Q² − 4PR > 0 at every D ≥ 3.04). If none is
accepted, B is not decided. The branch points are the real zeros of Q² − 4PR, mapped to D.

- *Pass B (planted):* on T_cut at least 80% of the accepted fits have a branch point within 0.05 of D = 3 and one
  within 1 of D = −7. On T_pole7, the control, fewer than half do.
- *Real points:* reported as located if the accepted fits' branch points have an interquartile range below 0.05 near
  D = 3 and below 1 on the negative axis.

**C. One short formula for every D.** On 3.04 ≤ D ≤ 67.77, the range of the cut-plane series (29 constants, every
point within 3.47σ), the smallest N at which a form puts every point within 3.47σ with no defect.

- *Pass C:* a quadratic form does it with N ≤ 24. The rational class goes through the same search for comparison.

**Controls that must fail.** `--mutate=root` evaluates the other root of the quadratic, and `--mutate=shift` moves the
held-out D by 0.01. Either must break A's scores (above 10⁻³ and above 10⁻⁵). `selftest` recovers an in-class plant
without noise to 10⁻²⁰, and fails with one planted coefficient scaled by 1 + 10⁻⁶.

**What counts as getting something.** A pass of A or C on the real points is a better formula. A located branch point
in B, with B's plants behaving, is new structure. Anything else is "nothing", reported with the numbers.

## 3. What did not go as frozen

Reported here before the results, because it limits what they mean.

**Amendment 1, the points (made after A's first run on the real points).** At N = 13 every form, rational or quadratic,
missed one held-out row by the same 8.4×10⁻⁸, the quad-double row at D 72.321. Its Δ equals the pinned row's at D
72.320955 to 2×10⁻¹⁶. Seven quad-double rows of `hp/data/delta_all.csv` carry a D rounded to four decimals, the D in
their state file's name, with the Δ of the unrounded D: 67.7422, 71.0622, 71.4622, 71.9482, 72.0619, 72.321 and 72.4303.
Six equal the pinned row beside them in Δ, at a D up to 4.5×10⁻⁵ away, which is up to 8.4×10⁻⁸ in Δ against a quoted err
of 2.4×10⁻²³. The seventh, 67.7422, sits 3×10⁻⁵ from a double-double row, and with it the best scores below D 67.77 stop
at 3.3×10⁻⁹ where they reach 6×10⁻¹¹ without it. The rule added to the loader drops a quad-double row that has another
row within 10⁻⁴ of its D (those seven, 2314 points left). Every run below, plants included, uses the amended points. The
first run is kept (`data/A_real_before_amendment.json`), and `algfit.py A --data=real --no_amend` repeats it. The table
is not changed here: the rows need their D from the states, which is the coordinator's to decide.

**Three planted expectations failed.**

- *A does not tell a cut from poles.* T_cut was expected to pass and does not (ratios 7.6 and 8.5 under uniform
  weights, against the 10 required). T_pole7, seven poles and no branch point, was expected to fail and passes. So A's
  pass on the real points means a better formula at equal constants, as section 2 said it would in that case.
- *B's planted pass failed.* On T_cut one form of 18 is accepted, and its branch points are at D = 3.0008 and −2.28,
  not −7. So nothing is read from the real fits about the negative axis.
- *No plant passes C.* Their noise is the full quoted σ, so 3.47σ asks for a fit at the noise floor at every one of
  1235 points. The real points scatter by about a third of their σ (χ²/dof 0.12 to 0.18 for every form that fits).
  C's pass on the real points therefore compares two formulas on the same points and the same bar, and no more.

**Other discrepancies** (reported, not fixed).

- Eleven pinned rows at D 73.07 to 73.26 lie 5 to 48 of their quoted err from the 24-constant formula, up to 5.3×10⁻¹³
  (`data/cut_compare.json`). The quad-double rows up to D 73.0 agree with it to 1.1×10⁻¹⁵, and the chain rows at
  D 75 to 79 within their err.
- The stored `hp/data/formula_cut.json` has χ²/dof 543 on today's table. That is known: it was fitted to the tables
  of commit 437b82a5 (FORMULA_FORMS.md, #765). Refitted on today's points with 29 constants it has χ²/dof 0.18.

## 4. Results of the frozen experiments

Every number is printed by `python3 summarize.py` from the files in `solver/quadform/data/`.

**A. Equal constants, fitted on 4 ≤ D ≤ 30.2 (210 points), scored on 30.2 < D ≤ 245.64 (2064 points).** The score is
the largest miss on points the fit never saw.

| points | weights | N = 9: rational, quadratic, ratio | N = 11 | N = 13 | pass A |
|---|---|---|---|---|---|
| real | 1/σ | 8.7×10⁻⁶, 4.3×10⁻⁷, 20 | 1.1×10⁻⁶, 6.2×10⁻⁸, 17 | 6.5×10⁻⁹, 1.1×10⁻⁹, 6.0 | |
| real | uniform | 3.2×10⁻⁶, 2.8×10⁻⁶, 1.1 | 3.1×10⁻⁶, 2.9×10⁻⁷, 11 | 7.6×10⁻⁷, 4.0×10⁻⁸, 19 | **yes** |
| T_cut | 1/σ | ratio 20 | 17 | 5.6 | |
| T_cut | uniform | 1.2 | 7.6 | 8.5 | no (expected yes) |
| T_pow | both | 60, 32 | 17, 5.8 | 19, 20 | yes |
| T_pole | both | 1.8, 1.1 | 5×10⁻⁵, 0.03 | 2×10⁻⁴, 1.5×10⁻⁴ | no (expected no) |
| T_pole7 | both | 6.7, 0.12 | 18, 11 | 80, 23 | yes (expected no) |

The controls fire: with the other root the quadratic scores are 2.0 to 2.8, and with the held-out D moved by 0.01
every score is 9.9×10⁻⁵. The selftest recovers its plant to 10⁻³⁰⁷ and fails at 1.8×10⁻⁷ with one coefficient scaled.

**B. Branch points, all 2314 points 3.04 ≤ D ≤ 245.64, 18 quadratic forms with N = 11 to 25.**

| points | accepted (χ²/dof ≤ 4, no defect) | frozen rule: branch points at 3 and −7 | post hoc: forms with a real branch point in 2.9 < D < 3.04 | where |
|---|---|---|---|---|
| real | 1 of 18, (7,7,7) | 0 | 18 of 18 | 2.9896 to 3.0032 |
| T_cut (truth 3) | 1 of 18 | 0 | 18 of 18 | 2.9892 to 3.0026 |
| T_pow (truth 3, exponent 0.3) | 0 of 18 | 0 | 18 of 18 | 3.0003 to 3.0054 |
| T_pole (no singularity at 3) | 2 of 18 | 0 | 0 of 18 | |
| T_pole7 (no singularity at 3) | 4 of 18 | 0 | 0 of 18 | |

By the frozen rule B is not decided: the planted pass failed, and one accepted form gives no interquartile range.
Read post hoc, the forms locate a singularity at D = 3 to about 0.01 whenever the truth has one and report none
otherwise, and the real points have one at 2.990 to 3.003. That agrees with what the equations say (at D = 3 the
scalar decouples from gravity, README "Relation to published work") and with the README's √(D − 3) formula. It does
not give the exponent: T_pow's 0.3 is located the same way. Δ at that branch point is 1.40 to 2.03 over the 18 real
forms, so Δ(3) stays undetermined. The real branch points on the negative axis move with the form (−2.05 for the
accepted one, −16.3 and pairs near −8.2 in others).

**C. One formula for 3.04 ≤ D ≤ 67.77 (1235 points), every point within 3.47σ.**

| points | class | first N within 3.47σ | post hoc: first N with χ²/dof ≤ 4, no defect |
|---|---|---|---|
| real | quadratic | **24**, (7,7,8): 2.76σ, χ²/dof 0.116 | 24 |
| real | rational | none to 28 | 28, [21/6]: 10.5σ |
| T_cut | quadratic | none to 28 | 24, (7,7,8): 6.5σ, χ²/dof 1.35 |
| T_cut | rational | none to 28 | 28, [21/6]: 8.6σ |
| T_pow | quadratic | none to 28 | none to 28 |
| T_pow | rational | 26, [19/6] | 26 |
| T_pole7 | quadratic, rational | none to 28 | 14, (4,4,4) and [7/6] |

C passes on the real points: 24 constants where the cut-plane series has 29. All three forms at N = 24 do it (2.76σ
to 2.80σ). On T_cut, the real points' twin with full-σ noise, the same form reaches χ²/dof 1.35 at N = 24 and the
rational class needs 28.

## 5. The formulas (post hoc)

Not frozen experiments. Each was fitted once, by the same code, and its coefficients are in a file that
`formula.py check` evaluates in float64 (with a mutation that must fail).

**Eleven constants, 4 ≤ D ≤ 245.64** (`data/formula_q11_uniform.json`; uniform weights, 2274 points, form (2, 2, 5)):

```math
P = -0.643873998937 - 2.93272993527\,\varepsilon - 4.86170957080\,\varepsilon^2, \qquad
Q = 1 + 8.87963443127\,\varepsilon + 17.7111053002\,\varepsilon^2,
```

```math
R = 0.0961027478146 + 4.91656912743\,\varepsilon + 6.32783375532\,\varepsilon^2 + 49.2309999662\,\varepsilon^3
- 70.2779720648\,\varepsilon^4 - 13.3348132752\,\varepsilon^5,
\qquad \Delta = \frac{2R}{-Q + \sqrt{Q^2 - 4PR}} .
```

The largest miss by range, each an interpolation error over the computed points. The points' own err is up to
5×10⁻⁹ below D 8.25, at most 2×10⁻¹⁵ on 8.25–67.77 and up to 4×10⁻¹¹ above:

| D | README formula (11 constants, fitted on 4–30.2 on 2026-10-19) | best rational with 11 constants, [6/4] | quadratic (2, 2, 5) |
|---|---|---|---|
| 4–8.25 | 6.5×10⁻⁹ | 5.1×10⁻⁸ | 1.8×10⁻⁹ |
| 8.25–30.2 | 8.1×10⁻⁹ | 1.8×10⁻⁸ | 6.3×10⁻¹⁰ |
| 30.2–67.77 | 5.8×10⁻⁸ | 4.4×10⁻⁹ | 9.5×10⁻¹¹ |
| 67.77–130 | 6.2×10⁻⁷ | 7.8×10⁻⁹ | 1.6×10⁻¹⁰ |
| 130–245.64 | 1.6×10⁻⁶ | 9.0×10⁻⁹ | 1.8×10⁻¹⁰ |

The two refits saw every point and the README formula saw none above 30.2, so only the two middle columns compare
like with like: 28 times tighter at the worst range. The rational forms tried are [9/1] to [5/5]. With the printed
12 digits the misses are the same to 1% (`formula.py check --file=data/formula_q11_uniform.json --digits=12`). Its
branch point is at D = 3.003, and it is not meant for D < 4 (2.3×10⁻² there). Its Δ∞, 1.6438937424, is 3×10⁻⁹ below
the 24-constant value: at 11 constants the formula is tighter than its own limit. With 1/σ weights
(`data/formula_q11.json`) the same form is at 3.5×10⁻¹¹ to 1.8×10⁻⁹ above D 8.25 and 7.5×10⁻⁷ below it, and the best
rational, [5/5], is comparable above D 8.25 and at 9.2×10⁻⁶ below it.

**Twenty-four constants, fitted on 3.04 ≤ D ≤ 67.77** (`data/formula_q24.json`, form (7, 7, 8), the coefficients
there), beside the cut-plane series refitted on the same 1235 points (`cut_compare.py`):

| formula | constants | χ²/dof | largest miss/σ | unseen, D 67.77–130 | unseen, D 130–245.64 |
|---|---|---|---|---|---|
| cut-plane series, K = 23 | 24 | 101 | 117 | 4.8×10⁻¹² | 6.4×10⁻¹¹ |
| cut-plane series, K = 25 | 26 | 7.35 | 40.3 | 5.8×10⁻¹³ | 1.8×10⁻¹¹ |
| cut-plane series, K = 28 | 29 | 0.18 | 3.78 | 2.6×10⁻¹² | 4.0×10⁻¹¹ |
| quadratic (7, 7, 8), in float64 | 24 | 0.181 | 3.04 | 6.8×10⁻¹³ | 2.4×10⁻¹¹ |

In ball arithmetic the quadratic has χ²/dof 0.116 and 2.76σ, and float64's rounding adds the rest. On the points above
D 67.77, which it never saw, it is within 1.1×10⁻¹⁵ of all 46 quad-double rows (D 67.8 to 73), and within 0.6 of the
err of all 190 chain rows up to D 245.64 (largest miss 2.4×10⁻¹¹, at D 245.64). Its misses inside the fit are
6.9×10⁻¹⁰ below D 4, 7.8×10⁻¹¹ on 4–8.25, 5.4×10⁻¹⁶ on 8.25–30.2 and 5.5×10⁻¹⁵ on 30.2–67.77.

**Δ∞ from the quadratic forms.**

| source | Δ∞ | the bar's kind |
|---|---|---|
| README (48 fits to D 61.03) | 1.643893746 ± 1.5×10⁻⁸ | fit spread, corrected by a planted test |
| track C (1/n series to D 245.64, `hp/data/delta_series.csv`) | 1.6438937453 ± 1.9×10⁻⁹ | series fits |
| quadratic (7,7,7), all points to D 245.64 | 1.643893745128 | none calibrated |
| quadratic (7,7,8), points to D 67.77 only | 1.643893745129 | none calibrated |
| the 7 quadratic forms with N ≥ 21, all points | 1.643893745012 to 1.643893745128 | spread over forms |

On the plants, the 24 forms with χ²/dof ≤ 4 return the planted limit within 7.0×10⁻¹¹ (T_cut 7.0×10⁻¹¹ from one form,
T_pole7 4.6×10⁻¹¹ from 13, T_pole 2.8×10⁻¹² from 3, T_pow 5×10⁻¹³ from 7). That is a largest planted miss with one
noise draw per plant, read after the runs, so it is not an error bar. The value assumes Δ analytic in 1/D, as the
README's does. It is *numerical evidence*.

## 6. Does it get anything

- **A better formula: yes.** At 11 constants it is 28 times tighter than a rational function on the same points, and
  at 24 constants it does what the cut-plane series does with 29 and predicts the chain to its err. It evaluates in
  float64 with one square root.
- **A new exact relation: no.** The form is a fit. That a finite Galerkin truncation of a quadratic system has an
  algebraic Δ(D) motivates the class and proves nothing about the continuum Δ(D).
- **New structure: no.** The branch point at D = 3 was known from the equations and from the √(D − 3) formula. The
  negative axis is not decided, and Δ(3) is not determined.
- **A tighter Δ∞: possibly.** The forms agree to 1.2×10⁻¹⁰ and the plants came back within 7×10⁻¹¹, against a quoted
  1.5×10⁻⁸. It needs the planted protocol of `hp/planted_dinf.py` before it is quoted (section 9).

## 7. Prior work (proposed entries for notes/LIT_REVIEW.md)

No novelty is claimed. Read for this note: abstracts and bibliographic records only, not the papers.

| Work | What it has | What it does not |
|---|---|---|
| R. E. Shafer, *On quadratic approximation*, SIAM J. Numer. Anal. 11(2), 447–460 (1974), doi:10.1137/0711037 | the quadratic approximant from a power series, "an attractive alternative to Padé approximation if convergence of the Padé approximation is not rapid" (abstract) | fits to points, this problem |
| P. G. Drazin, Y. Tourigny, *Numerical study of bifurcations by analytic continuation of a function defined by a power series*, SIAM J. Appl. Math. 56(1), 1–18 (1996), doi:10.1137/S0036139994272436 | algebraic (Hermite–Padé) approximants built from the series of one solution branch of a nonlinear problem in its parameter, which "reveal solution branches of the underlying problem in addition to the one represented by the original series" (abstract). The closest prior work to section 1's reading 3 | critical collapse, fits to points |
| C. K. Sanathanan, J. Koerner, *Transfer function synthesis as a ratio of two complex polynomials*, IEEE Trans. Autom. Control 8(1), 56–58 (1963) | the reweighted linear least squares for a rational fit, used in `algfit.py` step 1 | — |

In the repository, quadratic approximants already serve the fold (`hp/fold_mirror.py`, M2) and the amplitude series
(`solver/inner_amp/pade_chord_arc.py`).

## 8. Proposed text for the README (not applied)

For "The formula", after the cut-plane series:

> **A shorter form from a quadratic equation.** With Δ the root of P(ε)Δ² + Q(ε)Δ + R(ε) = 0, ε = 1/(D − 2), eleven
> constants reproduce every computed point from D = 4 to 245.64 within 1.8×10⁻⁹, against 5.1×10⁻⁸ for a rational
> function with eleven constants fitted to the same points, and 24 constants reproduce the points from D = 3.04 to
> 67.77 within 3.04 of their errors, where the cut-plane series needs 29. Fitted below D = 67.77 only, the 24-constant
> form gives the quad-double values at D 67.8–73 to 1.1×10⁻¹⁵ and the chain's values to D 245.64 within their errors.
> It is a fit, not a law ([notes/QUADRATIC_FORMULA.md](notes/QUADRATIC_FORMULA.md); quadratic approximants[^n]).

with the footnote Shafer 1974. The numbers would need checks in `check_numbers.py` against
`solver/quadform/data/formula_q11_uniform.json` and `cut_compare.json`, which this branch does not add.

## 9. Next steps

1. **Calibrate Δ∞.** Run `hp/planted_dinf.py`'s plants T1 to T5, with their junk points, through `algfit.py` with the
   criteria frozen first. If the 10⁻¹⁰ level holds, the README's bar on Δ∞ falls a hundredfold. Decision it informs:
   whether to quote Δ∞ to ten digits.
2. **Fix the seven rows.** `hp/export_delta_all.py` should take a quad-double row's D from the state, not from the
   file name. Until then a weighted fit that includes those rows is off by up to 8.4×10⁻⁸ at 10⁻²³.
3. **γ(D).** The same code fits λ = 1/γ. The README's γ formula is rational with 9 constants.
4. **The chain's reference.** `hp/anm_run.py`'s "formula diff" uses the README formula, 1.6×10⁻⁶ off above D 130. The
   24-constant form is within the chain's err there.

## 10. Reproduction

From `solver/quadform`, with the repository's venv. Each command takes 1 to 6 minutes on one core.

```
python3 algfit.py selftest                      # and --mutate=coef, which must exit 1
python3 algfit.py A --data=real --out=data/A_real.json        # also T_cut, T_pow, T_pole, T_pole7
python3 algfit.py A --data=real --mutate=root   # and --mutate=shift: the controls
python3 algfit.py B --data=real --out=data/B_real.json
python3 algfit.py C --data=real --out=data/C_real.json
python3 formula.py fit --N=11 --dmin=4 --weights=uniform --Mmax=5 --out=data/formula_q11_uniform.json
python3 formula.py fit --N=24 --dmin=3.04 --dmax=67.77 --spread=1 --Mmax=0 --out=data/formula_q24.json
python3 formula.py check --file=data/formula_q24.json         # and --mutate=coef, which must exit 1
python3 cut_compare.py
python3 summarize.py
```

`algfit.py` reads `hp/data/delta_all.csv`, `hp/data/formula_cut.json` and `solver/data/formula_coefficients.json`, and
on its first run calls `hp/dd_fits_mp.clean` (cached in `data/cleaned_drop.json`). The plants' noise uses the fixed
seed 20261119. The fits and the checks share the loader of the points and nothing else: `formula.py check` and
`cut_compare.py` evaluate the written coefficients in float64 by their own few lines.

[^shafer]: R. E. Shafer, *On quadratic approximation*, SIAM J. Numer. Anal. 11(2), 447–460 (1974),
    doi:10.1137/0711037 (the record and abstract were checked, the paper was not read).

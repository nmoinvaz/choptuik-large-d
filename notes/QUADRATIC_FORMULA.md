# QUADRATIC_FORMULA: a formula for Δ(D) from a quadratic equation

Written 2026-11-19 by a subagent of the coordinator on the Mac, branch `wip/quadratic-formula`. Scripts in
[`solver/quadform/`](../solver/quadform/). The question was Nathan's: look at new ways to find a formula for Δ(D)
"using quadratic base equations", and say whether that gets anything.

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
   rational functions is the quadratic approximant[^shafer]

```math
P(\varepsilon)\,\Delta^2 + Q(\varepsilon)\,\Delta + R(\varepsilon) = 0, \qquad \varepsilon = \frac{1}{D-2},
\qquad \Delta = \frac{-Q \pm \sqrt{Q^2 - 4PR}}{2P},
```

   with P, Q, R polynomials. The repository's formulas are a rational function of ε (the case P = 0), power series,
   and a series in a variable with a fixed square-root cut ([FORMULA_FORMS.md](FORMULA_FORMS.md)). Quadratic
   approximants were used here for the fold (`hp/fold_mirror.py`, M2) and for the amplitude series
   ([PADE_FAR_END.md](PADE_FAR_END.md)), not for Δ(D) over all D. A quadratic approximant finds its own branch points,
   the zeros of Q² − 4PR, which bears on FORMULA_FORMS.md's open question, a cut or poles on the negative D axis.

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

[^shafer]: R. E. Shafer, *On quadratic approximation*, SIAM J. Numer. Anal. 11, 447 (1974), as cited in
    `solver/inner_amp/pade_chord_arc.py` *(unverified)*.

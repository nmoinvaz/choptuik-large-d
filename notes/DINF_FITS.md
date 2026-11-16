# The fits for Delta at infinite D: the transseries test and the data range

Moved verbatim from README.md (commit ea42b3e, lines 1037–1051, 1088–1092) on 2026-11-04, in the reorganization of
[README_PLAN.md](README_PLAN.md); the README keeps a pointer where each block stood. Links were made relative to this
folder.

An exponentially small term, c εᵖ e^(−a/ε) with ε = 1/(D − 1), is the first non-perturbative piece if Δ is a
transseries in ε rather than a series (a = 1 for a one-instanton sector). It was added to the series at a = 0.5, 1, 1.5,
2 and p = 0, 1, with the same fit range and held-out points (`cd hp && python3 transseries_fit.py --planted`; logs in
[`hp/data/transseries/`](../hp/data/transseries/)):

- **Planted truth first.** On synthetic data at the same D (the degree-18 series fitted to every point, plus a planted
  term and noise), a planted c = 1 at a = 1, p = 0 makes the series alone miss the held-out points by about 10⁻¹⁰,
  against 5×10⁻¹² with the term. The fit recovers c to 4% at degree 14 and 11% at degree 18 at the points' own noise
  (3×10⁻¹⁵). The 10% criterion written before the runs holds at noise 10⁻¹⁷ and is missed at degree 18 at 3×10⁻¹⁵
  (`--noise=3e-15 --degs=16,18 --as=1 --ps=0`, which exits 1).
- **The real points.** The term changes nothing. At degree 16 the held-out error is 2.0×10⁻¹² without it and
  2.2×10⁻¹² with it (a = 1, p = 0). Its coefficient is −7×10⁻⁴ at degree 16 and 0.10 at degree 18, the scatter a
  planted c = 0 also gives (0.065 at degree 18), and Δ∞ moves by at most 1.3×10⁻⁷ over every (a, p) at degrees 14–18.
- **What it excludes.** A term that moved Δ∞ by 10⁻³ would need c of order 10³ at a = 1; a planted c = 100 already
  leaves the series alone missing the held-out points by 9×10⁻⁹. The test does not cover a < 0.5 or factors of ln ε.

Data range: the cloud runs reach D = 60.06 (and a few points at 60.96–60.97); the climb above that was done mostly on a
second machine (runs in [`hp/data/local/`](../hp/data/local/), coordinated through `comms.txt`), to D =
61.03. It was stopped there because the steps in D had shrunk to 0.0015–0.005 and further points add little to Δ∞; why
the steps shrink, and the solver options used (`hp/scan_hp.py --junk_hold`, `--junk_trend`, `--corrector dnp`), are in
[`solver/notes.md`](../solver/notes.md).

## Two checks without the planted offset (#906)

The quoted Δ∞ is the centre of 48 fits lowered by 0.8×10⁻⁸, the offset a planted-truth test found
([NUMERICS_REVIEW.md](NUMERICS_REVIEW.md) F3). `hp/dinf_check.py` tries two estimators that need no offset, each on
planted_dinf.py's plants T1–T5 before the real points, with the criteria frozen on #906 (rpi5, raw `I906-rpi5/`
75ef066a; the Mac reproduced the real lines).

**Bulirsch–Stoer extrapolation agrees with the quoted value.** It shares no code with the least-squares fits
[^bst]. It is the median of 54 tables, in h = 1/D and 1/(D − 2), Dmin 10, 15 and 20, with 8 to 16 points evenly
spaced in 1/D up to 61.03.
- On the plants it lands within 4.8×10⁻⁹ of the truth, against a bar of 1.5×10⁻⁸. With the wrong exponent
  (`--mutate omega2`) it misses all five by 0.032.
- On the real points it gives 1.64389374643, 4.3×10⁻¹⁰ from 1.643893746. The middle half of the tables spans ±1.2×10⁻⁸,
  an estimator spread.

**Akaike-weighted model averaging does not work on the real points.** It averages the 120 weighted fits of
dinf_budget.py's grid [^aic].
- On the plants it passes, within 2.3×10⁻¹⁰ with no offset, with the weight spread over many fits (largest 0.41 to
  0.55).
- On the real points one fit takes all the weight: degree 22 in 1/D, D 15–61.03, cleaned, limit 1.643896209. That is
  2.5×10⁻⁶ from the quoted value. With the noise estimated from the points (5.3×10⁻¹⁶ and 2.2×10⁻¹⁶ instead of the
  plants' 10⁻¹⁶ and 7×10⁻¹⁶, `--noise est`) the same fit wins, at 1.643896299.
- The criterion scores the fit inside the data, and on the real points the degree-22 1/D fit has the lowest χ² by 41
  against the next. Its limit lies at 1/D = 0, where the highest degrees are least constrained. dinf_budget.py's
  selection uses the two lowest degrees at the noise floor for that reason. So the average is not used, and the
  offset stays.

[^bst]: R. Bulirsch, J. Stoer, Numer. Math. 6, 413 (1964) *(unverified)*; the form of K. Krebs, M. Pfannmüller,
    H. Simon, B. Wehefritz, cond-mat/9402019, section 3.
[^aic]: W. I. Jay, E. T. Neil, Phys. Rev. D 103, 114502 (2021), arXiv:2008.01069.

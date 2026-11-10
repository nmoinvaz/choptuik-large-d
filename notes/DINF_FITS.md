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

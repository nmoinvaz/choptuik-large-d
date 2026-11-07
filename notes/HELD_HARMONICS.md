# Held harmonics: what holding them costs Δ at large D, and a better way to hold them (2026-11-06)

## Recommendation

Hold the harmonics on a fit of the full-system states in 1/n (`hp/held_fit.py`, `hp/data/held_fit.json`) instead of
the power-law trend, and restart the chain from the full-system quad-double state at 73.0 (T81) with
`anm_run.py --held_fit ../hp/data/held_fit.json`. The defaults are unchanged. The evidence is numerical:

- **The cause.** The chain's gap to the series is its held harmonic 5. The pinned system's first-order response,
  s · (h_chain − h_fit), gives −5.003e-8 at 79.77 against the measured −5.002e-8, and 90% of it is harmonic 5's cosine
  (section 3).
- **Harmonics 11 and 13.** They are held constant from 73.27 on, at values T75 had already polluted: harmonic 13 is
  13–50 times its full-system size. Δ hardly feels them, though, and they cost about 3e-12 at 80 (section 3).
- **Planted truth.** Fitted on the double-double states alone, the fit predicts the full system's Δ 7.5–12 units ahead
  to 3e-13, where the trend's rule in the same geometry is off by 1–2e-7 (4.1, 4.2). Its control fires.
- **In the chain.** PLACEHOLDER-CHAIN
- **Holding less.** The Schur complement on the held rows (4.3) shows the full equations determine harmonics 5, 7 and
  9 at 80, and harmonics 11 and 13 are the near-null directions. Solving 5–9 and holding only 11 and 13 gives, to
  first order, the same Δ as the fit (PLACEHOLDER-SCHUR-SHORT). That is the extrapolation-free route past D 100, where
  the fit extrapolates 42% beyond its range. Its chain form is `--kfreeze 9` or `7`, which is untested here past 80.

## 1. The question

Past D ≈ 73 the full system's Newton fails along near-null directions (QUAD_DOUBLE.md 12), so the chain
(`hp/anm_run.py`) solves a pinned system with `--kfreeze 3`. Its rows for the matching of p's harmonics 5–13 at the
horizon block are replaced by "p1's coefficient = held value", and p1 is the centre data, p ≈ x p1(τ) near x = 0. The
held values follow a power law in n = D − 2 per coefficient, fitted on T75's pinned states at D 70.7–73.3 and taken
relative to the chain's first base, T75's state at 73.27 (`--trend T75`, QUAD_DOUBLE.md 13.18). Against the 1/D and
1/n series fitted on the full-system double-double points (D ≤ 67.77), the chain lies below by 5.0e-8 at 79.77 and
1.44e-7 at 85.75 (13.19), and by 3.2e-6 at 200 (13.29). Releasing the held harmonics does not converge (13.19,
`T136-mac/`). The question is which holding removes most of that bias.

## 2. Criteria and controls, fixed before the runs

The bias of a held value h at a given D is Δ_pinned(h) − Δ_full. The pinned system's solution with h = h_full is the
full solution, so to first order the bias is s · (h − h_full), where s = dΔ/dh is the pinned system's response
(one quad-double LU per state, `held_fit.py sens`).

- **Planted truth** (`held_fit.py planted`): the full-system quad-double states at 68.5–73.0 (T84-mac, T81) are the
  truth.
  - P1: a fit on every full state with D ≤ 71.0 must predict 71.5, 72.8 and 73.0 with |bias| ≤ 1e-12.
  - P2: a fit on the double-double states alone (30 ≤ D ≤ 61.03) must predict 68.5–73.0, which is 7.5–12 units ahead,
    with |bias| ≤ 1e-10. That is the chain's own distance, 6.5 units at 79.77 and 12 at 85.
  - B1, the baseline: the chain's trend at the same D, reported.
  - B2: the trend's rule (a power law per coefficient, fitted the way `anm_run.py` fits it) on the double-double states
    58–61.03, relative to the state at 61.03, at 68.5–73.0, reported.
- **Control:** `--mutate=scale5` multiplies harmonic 5's fitted values by 1 + 1e-6, and P1 must then fail.
- **Series check, 79–86:** the candidate's Δ, Δ_chain + s · (h_candidate − h_chain) at a chain state, against
  `chain_vs_series.py`'s reference (the 1/D series of degree 20 on the double-double points, D ≤ 67.77). The criterion
  is that the gap shrinks by at least 100 times. The control is `--plant`, which adds a known offset to the candidate's
  harmonic-5 coefficient and must move the column by s₅ₐ times that offset.
- **In the chain:** the candidate runs as an option of `anm_run.py` (`--held_fit`, defaults unchanged), in
  quad-double, from a stored state. Its Δ must lie on the series at 75–80 within 1e-10.

## 3. What the chain holds, and where its bias comes from

**Harmonics 11 and 13 are held constant, at polluted values.** The trend's rule gives a coefficient an exponent only
when its sign is constant over the fitted states and the fit's residual in ln|v| is below 0.05. On T75's states
harmonics 11 and 13 fail that rule, so their exponents are 0 and the chain holds them at its first base's values from
73.27 on. Those values are not the full system's. T75's held 11 and 13 drift in its last steps (72.89–73.27).
In the table, √n times p1's coefficients are from `held_fit.py`'s `coeffs_of`, on the states named:

| state | 11 cos | 11 sin | 13 cos | 13 sin |
|---|---|---|---|---|
| full system, quad-double, 73.0 (T81) | −3.217e-5 | +4.460e-6 | +3.969e-6 | −6.926e-7 |
| T75 pin 7, 72.89 | −3.230e-5 | +4.487e-6 | +3.769e-6 | −5.654e-7 |
| T75 pin 7, 73.18 | −3.145e-5 | +4.130e-6 | +8.647e-6 | −2.759e-6 |
| T75 pin 7, 73.27 (the chain's first base) | −2.596e-5 | −1.830e-7 | +5.325e-5 | −3.376e-5 |

So at the chain's base harmonic 13 is 13 to 50 times its full-system size, and harmonic 11's sine has the wrong
sign. Harmonics 5–9 of that state lie on the full-system curve.

**What that costs is small.** The response s = dΔ/dh of the pinned system, at the full state at 73.0 and at the
chain's state at 79.77 (`held_fit.py sens`), per unit of the coefficient:

| D | 5 cos | 5 sin | 7 cos | 7 sin | 9 cos | 9 sin | 11 cos | 11 sin | 13 cos | 13 sin |
|---|---|---|---|---|---|---|---|---|---|---|
| 73.0 | 8.49e-3 | 7.15e-3 | 1.24e-4 | −8.58e-4 | 1.35e-5 | 3.78e-5 | −6.27e-6 | −6.68e-6 | 7.56e-7 | 1.11e-6 |
| 79.77 | 9.20e-3 | 7.25e-3 | 1.59e-4 | −8.79e-4 | 9.02e-6 | 2.97e-5 | −6.11e-6 | −5.52e-6 | 7.04e-7 | 9.66e-7 |

At 79.77 the chain's 11 and 13 differ from the full-system fit of section 4 by up to 6e-6, and together they move Δ by
about +3e-12 (the sum of their four terms, 2.6e-12 + 4.2e-12 − 3.8e-12 − 0.06e-12). Holding them constant costs about
3e-12 at 80, four orders below the bias.

**The bias is harmonic 5's trend.** The same products, s times the chain's held value minus the fit's, give the
bias term by term. At 79.77 harmonic 5's cosine gives −4.5e-8 and its sine −4.9e-9, and harmonic 7 gives +1.1e-10.
Their sum, −5.003e-8, is the chain's measured gap to the series, −5.002e-8. At 82.72 the sum is −8.91e-8, against a
measured −9.17e-8, and at 84.47 it is −1.154e-7 against −1.21e-7. Those two use s at 79.77, so they are estimates. The
exponent −1.21 of harmonic 5's cosine was fitted over 2.6 units of D and is used over 6–140 units. The full-system
curve of √n h₅ bends away from that power law, and harmonic 5 is the held harmonic that Δ is most sensitive to.

## 4. Candidates

### 4.1 The baseline: a power law per coefficient (the chain's `--trend T75`)

Planted truth (`held_fit.py planted`), with s from the nearest of the full states at 68.5, 71.0 and 73.0:

| method | 68.5 | 70.5 | 71.5 | 73.0 |
|---|---|---|---|---|
| B1, the chain's trend from T75's 73.27 state | −1.21e-8 | −1.00e-9 | +1.16e-9 | +5.61e-10 |
| B2, the trend's rule on dd 58.01–61.03, from 61.03 | −9.68e-8 | −1.44e-7 | −1.67e-7 | −2.09e-7 |

B1 at 73.0, a quarter unit from its base, is +5.6e-10. Its base already carries a held error, and B1 moves away
from the full Δ as D moves away from 73.27 in either direction. B2 is the same rule in the chain's geometry, a fit
over 3 units extrapolated 7.5–12 units ahead, and it is off by 1–2e-7, the size of the chain's bias at 80–86.

### 4.2 A fit of the full-system states in 1/n (`held_fit.py`, `anm_run.py --held_fit`)

**The method.** For each held coefficient v, √n v is a Chebyshev series in x = 1/n, fitted by least squares on the
full-system states. These are the double-double states of `hp/data/states` and of T12-win and T13-win, and the
quad-double states of T84-mac and T81. The form is the centre scaling p ≈ x F(θ)/√n, with F converging as D grows
(README, "How the solution scales with D"). `pinning.HeldFit` evaluates the fit and its Taylor coefficients in D by
power-series arithmetic in mpmath, and `anm_run.py --held_fit FILE` holds the harmonics on it. As with the trend, the
held values are taken relative to the base, v(D) = v(D₀) + fit(D) − fit(D₀). `held_fit.py check` compares the Taylor
coefficients with the values (2.2e-16 relative, pass below 1e-14), and its control `--mutate=jet` fails at 1.4e-10.

There are two versions.

- **The run's fit** (`hp/data/held_fit_deg8.json`) has degree 8 for every coefficient, on 146 states with
  30.07 ≤ D ≤ 73.0. The chain run below used it.
- **The recommended fit** (`hp/data/held_fit.json`, `fit --deg=8 --deg_hi=4 --Dmin=38`) has degree 8 for harmonics
  5–9 and 4 for 11 and 13, on 138 states with 38.02 ≤ D ≤ 73.0. Two things in the run's fit prompted the change,
  both seen after the run. The double-double states below D 36.5 step in harmonics 11 and 13, with residuals up to
  9e-7 in √n v at 35–37. Also, the degree-8 and degree-10 fits of harmonic 13 extrapolate to −5e-5 and +5e-4 at
  D 200, against a full-system size of 5e-7 at 73. With the change, harmonic 13 at 200 is −2e-8 to +5e-8 at
  degrees 6, 8 and 10 (`held_fit_d6.json`, `held_fit_d10.json`, degree 3 and 5 for 11 and 13). The largest
  residuals are 8e-14 (harmonic 5), 1.4e-11 (7), 5e-10 (9), 4e-9 (11) and 2e-8 (13), the last being the T12 and T13
  states near 60.95. Times s, each is at most 3e-14 in Δ. The two versions agree in harmonic 5 to 1e-6 relative
  at 79.8–200 (the 7 digits printed), and in harmonic 7 to 3e-4 relative at 200.

**Planted truth**, with s from the nearest of the full states at 68.5, 71.0 and 73.0. The recommended fit gave
(`planted --deg=8 --deg_hi=4 --Dmin=38`):

| fit | predicted | bias in Δ | criterion |
|---|---|---|---|
| P1, every full state with D ≤ 71.0 | 71.5, 72.8, 73.0 | +5.6e-16, +8.1e-16, +8.4e-16 | 1e-12, pass |
| P2, the double-double states alone (38 ≤ D ≤ 61.03) | 68.5–73.0 | −9.8e-14 (68.5) to −3.2e-13 (73.0) | 1e-10, pass |

The run's fit and its degree-6 and degree-10 variants (D ≥ 30, with s from 73.0 only) pass as well: P1 within 3e-14
and P2 within 7e-13.

**Control:** with `--mutate=scale5` (harmonic 5 × (1 + 1e-6)), P1 fails at 71.5–73.0 with +2.08e-11, and the run
exits 1. Over the distance at which B2 is off by 1–2e-7, P2 is off by at most 3e-13.

**In the chain** (the nonlinear test). `anm_run.py --held_fit data/held_fit_deg8.json` ran from T81's full-system
quad-double state at 73.0. The settings were `--kfreeze 3 --corrector qd --h_max 2 --fix1 1 --F_pred_max 1.5e-14
--threads 2`, everything else default, and the run used the first fit (degree 8, D ≥ 30, every harmonic at degree 8).
The log, `anm_run.jsonl` and the states are in the scratch folder `held_harmonics/chain_fit73/`. The run took 8
steps to 79.83, with the Mac's load average at 53–74 when it was read. Every point converged in quad-double (no
stall accepted), and the run was stopped there. `chain_vs_series.py --folders=<that folder>,anm-mac/run_trend_pin3,anm-mac/run_trend_pin3_qd3`
gives Δ minus the series. The spread is over the six series (degrees 16–20 in 1/D and 1/n), a spread over fit choices
and not a bound:

| D | run | Δ | 1/D 20 | 1/n 20 | spread of the six | \|mean\| / spread |
|---|---|---|---|---|---|---|
| 73.0 | full system (delta_qd.csv) | 1.781239681930926 | −7.0e-15 | −3.9e-15 | 1.6e-12 | 0.2 |
| 75.27 | old chain (trend) | 1.77723094864874 | −8.31e-9 | −8.31e-9 | 3.0e-12 | 2749 |
| 75.43 | new chain (fit) | 1.776944961053467 | −1.9e-14 | −1.2e-14 | 3.2e-12 | 0.2 |
| 77.27 | old chain | 1.773882303309153 | −2.32e-8 | −2.32e-8 | 4.9e-12 | 4787 |
| 77.43 | new chain | 1.773610529686257 | −3.6e-14 | −2.2e-14 | 5.0e-12 | 0.2 |
| 78.90 | new chain | 1.771269501318305 | −5.3e-14 | −3.3e-14 | 6.8e-12 | 0.2 |
| 79.77 | old chain | 1.769926258937742 | −5.00e-8 | −5.00e-8 | 8.0e-12 | 6225 |
| 79.83 | new chain | 1.769822950693969 | −6.7e-14 | −4.1e-14 | 8.1e-12 | 0.2 |

The new chain lies on the series inside the series' own spread at every point, with the same residual pattern as the
full-system points at 68.5–73 (|mean| / spread 0.2). The old chain lies 2700–6200 spreads below. At 79.8 the gap
shrinks from 5.0e-8 to under 1e-13 against the degree-20 series, a factor of more than 10⁵. The criterion was 100
and agreement within 1e-10, so the test passes. The old chain's points at 75.27 and 77.27 are its double-double run
(`run_trend_pin3`).

PLACEHOLDER-SERIES

### 4.3 Holding less: solve the well-determined held directions (`held_schur.py`)

**The method.** At a pinned state, block elimination of the full system's Newton step on the ten held rows gives the
Schur complement S = dg/dh. Here g is the full system's residual in the dropped matching rows, harmonics 5–13 of the
horizon block's p rows, and it is the only part of the full residual that the pinned solution leaves nonzero.
Truncating S's SVD solves the directions the full equations determine and holds the others. That is a linear estimate
of "hold only the near-null combination" and of the self-consistent update of the held values, which are the same
computation once iterated.

**The spectrum at the chain's state at 79.77** (`held_schur.py RAW anm-mac/run_trend_pin3_qd3/anm_pin3_D79.7657.npz`).
Each singular pair of S is one harmonic, its cosine and sine:

| harmonic | σ (from S at eps) | Δ per unit along v | full Newton step along v | its Δ change |
|---|---|---|---|---|
| 5 | 7.42e-13, 7.42e-13 | −6.0e-3, +1.0e-2 | −4.4e-6, +2.4e-6 | +2.61e-8, +2.42e-8 |
| 7 | 2.46e-16, 2.46e-16 | −5.1e-4, −6.4e-4 | 3.1e-8, 5.0e-7 | −1.6e-11, −3.2e-10 |
| 9 | 1.75e-19, 1.69e-19 | 5.6e-5, −1.4e-5 | −6.9e-6, −1.7e-5 | −3.9e-10, +2.4e-10 |
| 11 | 3.37e-22, 2.52e-22 | −1.3e-6, −4.1e-6 | 1.1e-3, 2.3e-3 | −1.4e-9, −9.3e-9 |
| 13 | 2.24e-23, 1.45e-23 | 3.0e-7, 2.8e-7 | −0.060, −0.077 | −1.8e-8, −2.2e-8 |

The full equations' grip falls by about 10³ per harmonic. Harmonics 11 and 13 are the near-null directions: the full
Newton step wants to move harmonic 13's coefficients by 0.06–0.08, against a full-system size of 4e-7. Their pairs
carry the 0.51 step that stopped the release of 13.19 and T136. Truncated steps (S from eps, and from eps/2 as a check
of the finite differences):

| solved | held | Δ change of the truncated step | from S at eps/2 | chain − series + change |
|---|---|---|---|---|
| 5 | 7–13 | +5.031e-8 | PLACEHOLDER | +2.9e-10 |
| 5, 7 | 9–13 | +4.997e-8 | PLACEHOLDER | −4.7e-11 |
| 5, 7, 9 | 11, 13 | +4.983e-8 | PLACEHOLDER | −1.9e-10 |
| 5–11 | 13 | +3.91e-8 | PLACEHOLDER | −1.1e-8 |
| all | none | −4.0e-10 | PLACEHOLDER | −5.0e-8 |

The chain's gap to the series is −5.002e-8. Solving 5 and 7, or 5 to 9, and holding the rest removes it to 5e-11 to
2e-10, to first order. Solving everything (the full Newton step) does not, because the near-null steps of 11 and 13 are
far outside the linear range. Their Δ terms, −1.1e-8 and −4.0e-8, cancel the gain from harmonic 5. So "hold only the
near-null combination" works here when the near-null combination is harmonics 11 and 13 (or 9–13). The first-order
estimate leaves 5e-11 to 2e-10, which is larger than the fit's +1e-11 at the same state. The fit's estimate is first
order too, but in a much smaller held error.

**Planted truth** (`--plant=5a:1e-8,13a:1e-6` at the full state at 73.0, where g = 0). A planted held error of 1e-8 in
harmonic 5's cosine and 1e-6 in 13's, in the linear model, has an exact Δ correction of −8.565e-11. Of that, −8.49e-11
is the 5a part and −7.6e-13 the 13a part. Solving 5–9 and holding 11 and 13 recovers −8.520e-11 (−8.514e-11 from S at
eps/2), which is the 5a part plus half the 13a part. Solving 5 alone recovers −8.596e-11. The method therefore
recovers what lies in the directions it solves.

**Its check fails, and its control fires.** The check fixed beforehand was that S from eps and eps/2 agree to 1e-6 in
every singular value above 1e-12 σ₁, and it fails. At 73.0 the relative differences are 1.5e-6 for harmonic 5, 5e-5
for 7, 2e-3 for 9, 1e-2 for 11 and 0.1 for 13, and at 79.77 the worst is 0.76. Central differences at a relative step
of 2⁻⁴⁰ do not resolve the near-null singular values. So those values are bounds of order 1e-22 to 1e-23, not
measurements. The truncated steps that solve 5–9 change by less than 1e-3 relative between eps and eps/2. The control
`--mutate=pinned_rows` reads g and S from the pinned kernel's own rows, which do not contain the dropped matching.
There σ₁ is 1.3e-33 and the check fails, exit 1.

### 4.4 Not run

- **A junk-only deflated Newton on the full system** (`newton_qd`'s `junk` mode, from a chain state). Section 4.3
  shows the junk at 79.77 is harmonics 11 and 13. Freezing them would solve 5–9, as 4.3's truncated step does. Any
  stored chain state is off by 4e-6 in harmonic 5, though, and the quad-double Newton's basin is about 2e-9 in its
  first step (QUAD_DOUBLE.md 13.19). The junk mode also keeps the rows of the frozen harmonics, which threw the step in
  13.19 (a).
- **Deflation** (Farrell, Birkisson and Funke) removes known solutions to find others. The problem here is not
  finding a second solution but choosing held values, so it does not apply.
- **A Tikhonov penalty** toward the smoothest of the nearly equivalent solutions. Section 4.3 shows the near-equivalence
  is in harmonics 11 and 13 alone, whose effect on Δ is about 1e-12 at 80. A penalty would choose values Δ does not
  feel, so the fit of 4.2, which is smooth in D by construction, does the same job at no cost.
- **The D = ∞ anchor.** The inner problem's tuned profile would pin the fit at x = 0, making D = 200 (x = 0.0051) an
  interpolation rather than an extrapolation 42% of the fitted range beyond its end (x 0.0141–0.0357). Its harmonics in
  the solver's phase convention (the phase row Σ wⱼ p1ⱼ = 0) were not computed here. Section 4.2's spread over degree
  at 100–200 measures what the anchor would buy.

## 5. For the coordinator

- **Raw output.** The chain run of 4.2 wrote to the scratch folder named in its log line (`--allow_unraw`, as the
  task brief asked; the raw-data repository was read-only here). Its states and `anm_run.jsonl` belong in a raw-data
  folder, `held-harmonics-mac/`, before anything in main cites them.
- **Proposed text for QUAD_DOUBLE.md 13.19 and 13.29** (the held harmonics' bias): "The bias is harmonic 5's held
  trend. Its power law, fitted over 2.6 units of D, is used over 6–140, and the pinned system's response to the held
  values reproduces the chain's gap to the series at 79.77 to 0.03% (notes/HELD_HARMONICS.md). Harmonics 11 and 13,
  held constant at polluted values, cost about 3e-12. Holding the harmonics on a fit of the full-system states in 1/n
  removes the bias (`anm_run.py --held_fit`)."
- **The existing chain to 214.** It can be corrected to first order at each stored state by s · (h_fit − h_chain),
  with one quad-double LU per state (`held_fit.py sens`, then `series`). Past D 100 the held harmonic 5 is off by 2–18%,
  so the first-order correction is an estimate there, and a restarted chain is the test.

## 6. References

Verified against the publisher's page, arXiv or a library record on 2026-11-06 (web search; the full texts were not
read, so nothing here rests on their content beyond the method each names):

- H. B. Keller, *The bordering algorithm and path following near singular points of higher nullity*, SIAM J. Sci.
  Stat. Comput. 4 (1983) 573–582, doi:10.1137/0904039. The block elimination of a bordered Newton system, here the
  Schur complement on the held rows.
- H. B. Keller, *Numerical solution of bifurcation and nonlinear eigenvalue problems*, in *Applications of Bifurcation
  Theory*, ed. P. H. Rabinowitz, Academic Press, New York (1977) 359–384. Pseudo-arclength continuation (Riks and
  Wempner had related methods earlier, per secondary sources; not checked).
- P. C. Hansen, *The truncated SVD as a method for regularization*, BIT 27 (1987) 534–553, doi:10.1007/BF01937276.
  The truncated step of 4.3.
- P. E. Farrell, Á. Birkisson, S. W. Funke, *Deflation techniques for finding distinct solutions of nonlinear partial
  differential equations*, SIAM J. Sci. Comput. 37 (2015) A2026–A2045, doi:10.1137/140984798, arXiv:1410.5620.
  Deflation of known solutions, which 4.4 sets aside.
- A. N. Tikhonov, *Solution of incorrectly formulated problems and the regularization method*, Soviet Math. Dokl. 4
  (1963) 1035–1038, the translation of Dokl. Akad. Nauk SSSR 151 (1963) 501–504. One library record gives the
  translation's volume as 5 *(volume unverified)*. The smoothest-solution penalty of 4.4.

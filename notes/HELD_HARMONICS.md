# Held harmonics: what holding them costs Δ at large D, and a better way to hold them (2026-11-06)

## Recommendation

Hold the harmonics on a fit of the full-system states in 1/n (`hp/held_fit.py`, `hp/data/held_fit.json`) instead of
the power-law trend, and restart the chain from the full-system quad-double state at 73.0 (T81) with
`anm_run.py --held_fit data/held_fit.json` (run from `hp/`). The defaults are unchanged. The evidence is numerical:

- **The cause.** The chain's gap to the series is its held harmonic 5. The pinned system's first-order response,
  s · (h_chain − h_fit), gives −5.003e-8 at 79.77 against the measured −5.002e-8, and 90% of it is harmonic 5's cosine
  (section 3).
- **Harmonics 11 and 13.** They are held constant from 73.27 on, at values T75 had already polluted: harmonic 13 is
  13–50 times its full-system size. Δ hardly feels them, though, and they cost about 3e-12 at 80 (section 3).
- **Planted truth.** Fitted on the double-double states alone, the fit predicts the full system's Δ 7.5–12 units ahead
  to 3e-13, where the trend's rule in the same geometry is off by 1–2e-7 (4.1, 4.2). Its control fires.
- **In the chain.** From T81's full state at 73.0, the chain with `--held_fit` reached 79.83 in 8 quad-double steps.
  Every point lies on the series within the series' spread: −6.7e-14 against the degree-20 series at 79.83, where the
  old chain is at −5.0e-8 at 79.77 (4.2). The control, a fit with harmonic 5's increments 1% wrong, shows −2.99e-9
  after one step against a predicted −2.96e-9.
- **The old chain, corrected to first order.** s · (h_fit − h_chain) at its stored states brings it onto the series
  to 6e-14 at 79.77, 1e-12 at 100, 2.5e-11 at 150 and 9e-11 at 200 (4.2, with the octo-double LU at 200). The whole
  gap of 13.29, −3.2e-6 at 200, is the held harmonics' first-order error.
- **Holding less.** The Schur complement on the held rows (4.3) shows the full equations determine harmonics 5, 7 and
  9 at 80, and harmonics 11 and 13 are the near-null directions. Solving 5 and 7 and holding the rest gives, to first
  order, the same Δ as the fit, +5.00e-8 against the +5.002e-8 needed. Solving 5–9 gives the same to 2e-10. Its
  finite-difference check fails in the near-null singular values (4.3). This is the route that needs no extrapolation
  past D 100, where the fit extrapolates 42% beyond its range. Its chain form is `--kfreeze 7` or `9`, untested here.

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

**The bias is harmonic 5's trend.** The same products, s times the chain's held value minus the fit's, give the bias
term by term. At 79.77 harmonic 5's cosine gives −4.5e-8 and its sine −4.9e-9, and harmonic 7 gives +1.1e-10. Their sum,
−5.003e-8 (this first estimate used the run's fit of 4.2), is the chain's measured gap to the series, −5.002e-8. With s
at each state and the recommended fit, the sum equals the gap to 1e-13 at 79.77–85.75, 2.5e-11 at 150 and 9e-11 at 200
(4.2). The exponent −1.21 of harmonic 5's cosine was fitted over 2.6 units of D and is used over 6–140 units. The
full-system curve of √n h₅ bends away from that power law, and harmonic 5 is the held harmonic that Δ is most sensitive
to.

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
quad-double state at 73.0. The settings were
`--kfreeze 3 --corrector qd --h_max 2 --fix1 1 --F_pred_max 1.5e-14 --threads 2`, everything else default, and the run
used the first fit (degree 8, D ≥ 30, every harmonic at degree 8). The log, `anm_run.jsonl` and the states are in the
raw-data repository's `held-harmonics-mac/chain_fit73/` (acaccfa). The run took 8 steps to 79.83, with the Mac's load
average at 53–74 when it was read. Every point converged in quad-double (no stall accepted), and the run was stopped
there. `chain_vs_series.py RAW --folders=<that folder>,anm-mac/run_trend_pin3,anm-mac/run_trend_pin3_qd3` gives Δ minus
the series (that command, with the folder path written out). The spread is over the six series (degrees 16–20 in 1/D and
1/n), a spread over fit choices and not a bound:

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

**The chain's control** (fixed before its run): one step from 73.0 to 73.81 on a fit whose harmonic 5 is 1% wrong
(`fit --deg=8 --mutate=scale5:0.01`, the run's fit otherwise). In relative mode that puts 1% of harmonic 5's increment
into the held values, −3.48e-7 in its cosine. The predicted bias, s(73) times that error, is −2.96e-9. The step
converged (quad-double, 3 steps), with Δ 1.779777558776957 and Δ − series −2.99e-9, 1470 spreads off the series and
1% from the prediction. The unmutated step to the same D gave −1.0e-14 against the degree-20 series.

**The series check at 79–200, to first order** (`held_fit.py series`, with s from `sens` at each chain state, by the
octo-double LU at 200). Each row is a stored state of the old chain, corrected by s · (h_fit − h_chain). The last
column is the same with the degree-6 and degree-10 variants and with the run's fit:

| D | chain − series | s · (h_fit − h_chain) | corrected − series | other fits − series |
|---|---|---|---|---|
| 79.77 | −5.002e-8 | +5.002e-8 | −6.3e-14 | −4.1e-14, −1.0e-13, −5.8e-14 |
| 82.72 | −9.167e-8 | +9.167e-8 | −1.2e-13 | −5.5e-14, −2.1e-13, −9.8e-14 |
| 85.75 | −1.438e-7 | +1.438e-7 | −1.9e-13 | −6.7e-14, −3.7e-13, −1.5e-13 |
| 100.12 | −4.775e-7 | +4.775e-7 | −1.2e-12 | −2.7e-13, −2.4e-12, −8.7e-13 |
| 149.56 | −1.895e-6 | +1.895e-6 | −2.5e-11 | −1.4e-11, −3.2e-11, −2.1e-11 |
| 200.27 | −3.158e-6 | +3.158e-6 | −8.8e-11 | −6.0e-11, −8.8e-11, −7.9e-11 |

To first order, the held values' error accounts for the whole of the old chain's gap to the series from 80 to 200.
What remains is 1e-13 at 80–86, 1e-12 at 100, 2.5e-11 at 150 and 9e-11 at 200. That lies inside the series' own
spread over its six fits, 8.4e-12 at 80, 9.1e-11 at 100, 9.0e-10 at 150 and 2.3e-9 at 200
(`hp/data/published_vs_series.txt`), and it is of the size of the held fits' spread over degree. At 150 and 200 the
chain's held harmonic 5 is 10% and 18% off the fit, and the first-order term is still all of the gap. The
second-order term is therefore below 1e-10 there.

The sensitivities at 200 need the octo-double LU. The quad-double LU's smallest pivot there is 5.1e-15, against
1.3e-12 for the octo-double one and 6e-8 to 4e-7 below 150, and its s comes out of order 10 with random signs. That is
the quad-double LU failure of QUAD_DOUBLE.md 13.28 and OCTO_DOUBLE.md 5. The octo-double LU at 79.77 reproduces the
quad-double s there to all five printed digits (`sens --octo=1`, 79.77 and 200.27). At 200 its s continues the trend
from 100 and 150 (harmonic 5's cosine 1.11e-2, 1.47e-2, 1.77e-2).

**Control:** `series --plant=1e-6` adds 1e-6 to the fit's harmonic-5 cosine. It moves the corrected column by
+9.20e-9 at 79.77 and +1.76e-8 at 200, which is s₅ₐ times 1e-6 at each state (9.197e-3 and 1.767e-2).

Two references, two extrapolations. The series extrapolates Δ itself in 1/D from D ≤ 67.77. The fit extrapolates
p1's harmonics in 1/n from D ≤ 73 and passes them through the chain's pinned system. They share the full-system
double-double states (the fit's 38–61 and the series' 10–67.77 overlap) and the assumption of analyticity in 1/n.
They do not share the chain, the held-value treatment or the Δ data above 61. Their agreement to 1e-13–9e-11 at 80–200
is numerical evidence for both. It is not a bound. For 13.29 it means our side of the gap to arXiv:2611.07007 is the
held harmonic 5: with it corrected, our values lie on the series, and the published table lies above both by the
published-minus-series column there (+8.6e-8 at 80 to +4.3e-6 at 200).

**The rerun on the fit** (T138, win2, `T138-win2/chain` at raw 247b1ec; QUAD_DOUBLE.md 13.25). The chain was rerun
from T81's full-system state at 73.0 with `--held_fit data/held_fit.json`, in steps of 1.0 to 173.5. It is the
nonlinear chain, not the first-order correction. `chain_vs_series.py RAW --folders=T138-win2/chain` compares each of
its 101 states with the six series fits (`hp/data/chain_fit_vs_series.txt`):

| D | rerun − 1/D degree 18 | rerun − 1/D degree 20 | spread of the six fits | \|mean residual\| / spread |
|---|---|---|---|---|
| 80 | −3.0e-15 | −7.6e-14 | 8.4e-12 | 0.2 |
| 100 | −2.2e-13 | −1.6e-12 | 9.1e-11 | 0.2 |
| 150 | −2.2e-12 | −2.7e-11 | 9.0e-10 | 0.2 |
| 173.5 | −3.7e-12 | −5.1e-11 | 1.5e-9 | 0.2 |

Every state lies inside the series' spread, with the same ratio 0.2 throughout. The control, `--plant=1e-9`
(`hp/data/chain_fit_vs_series_plant.txt`), shows as a residual above the spread in 72 of the 101 states, all those below
about 145, where the spread is under 1e-9. Above that the spread is the check's resolution. What the two share is as
above, the full-system states at D ≤ 73. The junk-frozen full-system step, which uses neither a fit nor the series,
resolves chain_fit's error above 145 too: from T138's states it is +1.2e-14 at 80 to −3.1e-12 at 215, against +5.0e-8 to
+3.4e-6 from the trend-held chain's (T141, QUAD_DOUBLE.md 13.29).

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

| solved | held | Δ change, S at eps | S at eps/2 | chain − series + change (eps; eps/2) |
|---|---|---|---|---|
| 5 | 7–13 | +5.031e-8 | +5.031e-8 | +2.9e-10; +2.9e-10 |
| 5, 7 | 9–13 | +4.997e-8 | +5.001e-8 | −4.7e-11; −1.3e-11 |
| 5, 7, 9 | 11, 13 | +4.983e-8 | +5.020e-8 | −1.9e-10; +1.8e-10 |
| 5–11 | 13 | +3.91e-8 | +3.08e-8 | −1.1e-8; −1.9e-8 |
| all | none | −4.0e-10 | −3.58e-8 | −5.0e-8; −8.6e-8 |

The chain's gap to the series is −5.002e-8. To first order, solving 5 and 7 and holding the rest removes it to
1–5e-11, and solving 5 to 9 removes it to 2e-10. Harmonic 9's singular values are uncertain by 3% in the finite
differences, which moves the second result by 4e-10. Solving everything (the full Newton step) does not remove it,
because the near-null steps of 11 and 13 are far outside the linear range. Their Δ terms, −1.1e-8 and −4.0e-8 from S
at eps, cancel the gain from harmonic 5, and they change with eps, since their singular values are not resolved. So
holding only the near-null combination works here when that combination is harmonics 11 and 13 (or 9–13). The
first-order estimate leaves 1e-11 to 2e-10, against the fit's 6e-14 at the same state (4.2).

**Planted truth** (`--plant=5a:1e-8,13a:1e-6` at the full state at 73.0, where g = 0). A planted held error of 1e-8 in
harmonic 5's cosine and 1e-6 in 13's, in the linear model, has an exact Δ correction of −8.565e-11. Of that, −8.49e-11
is the 5a part and −7.6e-13 the 13a part. Solving 5–9 and holding 11 and 13 recovers −8.520e-11 (−8.514e-11 from S at
eps/2), the 5a part and 3e-13 more. Solving 5 alone recovers −8.596e-11. Within 1e-12, the method therefore
recovers the error in the directions it solves and leaves the error in the ones it holds.

**Its check fails, and its control fires.** The check fixed beforehand was that S from eps and eps/2 agree to 1e-6 in
every singular value above 1e-12 σ₁, and it fails. At 73.0 the relative differences are 1.5e-6 for harmonic 5, 5e-5
for 7, 2e-3 for 9, 1e-2 for 11 and 0.1 for 13. At 79.77 they are 4e-5 for 5, 3e-5 to 2e-4 for 7, 0.5–3% for 9,
3–19% for 11 and 66–76% for 13. Central differences at a relative step of 2⁻⁴⁰ do not resolve the near-null singular
values, so those values are estimates of order 1e-22 to 1e-23, not measurements. Between eps and eps/2, the
truncated steps that solve 5 or 5–7 change by less than 1e-3 relative, and the one that solves 5–9 by 7e-3. The
control `--mutate=pinned_rows` reads g and S from the pinned kernel's own rows, which do not contain the dropped
matching. There σ₁ is 1.3e-33 and the check fails, exit 1.

**The exact Jacobian (#479).** `held_schur.py --jac=exact` forms S without differences, as the horizon p rows of the
full system's exact quad-double Jacobian (`Solver.jacobian_q`) applied to each dz_j. `--eps_pow=70` takes the
differences with a step 2³⁰ times smaller than above. At 79.77 the two agree to 8e-10 in all ten singular values, and
S at eps and eps/2 agree to 6e-9, so the check fixed beforehand now passes (raw `I479-mac/`,
`schur_D79.7657_exact_e70.log`). The controls fire: `--mutate=jac_scale` (the exact rows times 1.01) and
`--mutate=pinned_rows` both exit 1. The values of harmonics 5 and 7 above stand. Those of 9, 11 and 13 were off,
13's by a factor of 11 to 16:

| harmonic | σ, exact (its cosine-sine pair) | σ from the differences at 2⁻⁴⁰, above |
|---|---|---|
| 5 | 7.422e-13, 7.420e-13 | 7.42e-13, 7.42e-13 |
| 7 | 2.461e-16, 2.461e-16 | 2.46e-16, 2.46e-16 |
| 9 | 1.682e-19, 1.682e-19 | 1.75e-19, 1.69e-19 |
| 11 | 2.877e-22, 2.876e-22 | 3.37e-22, 2.52e-22 |
| 13 | 1.364e-24, 1.362e-24 | 2.24e-23, 1.45e-23 |

The full Newton step along harmonic 13's pair is 0.44 and 0.10, not 0.06–0.08. The truncated steps from the exact S
move Δ by +5.031e-8 (solving 5), +5.002e-8 (5 and 7), +5.028e-8 (5–9), +2.68e-8 (5–11) and +1.15e-7 (all). The
conclusions above stand: the grip falls by 200 to 3000 per harmonic, 11 and 13 are the near-null directions, and
solving 5 and 7 removes the chain's gap of −5.002e-8.

Past 80 the step 2⁻⁴⁰ is worse. At 150 the residual's cubic term along dz_j outweighs S, and S at eps/2 is a quarter
of S at eps in every value. With `--eps_pow=70` the differences resolve harmonics 5 and 7 at 100 and 150, and agree
with the exact S there to 3e-4 and 4e-6. Past D 150 quad-double is not enough: at 200 and 232.6 (pivots up to 10²⁸ and
10³³) its S breaks the cosine-sine pairs, which `held_grip.py` refuses. `--od` does the same in octo-double. At 150 it
gives the quad-double exact values in every printed digit, with the differences at 2⁻¹⁰⁰ agreeing to 7e-22. The grip
on harmonic 5 falls to 3.0e-17 at 200 and 1.1e-24 at 232.6, below the check's floor of 1e-14, which was set from the
states near 80. `--sigma_min=1e-25` lowers it for those two. The control `--mutate=pinned_rows` at 200 still fails
(σ₁ 1.3e-33). At 232.6 the differences need the step 2⁻¹¹⁰: at 2⁻¹⁰⁰ they agree with the exact S only to 1.1e-3.

| D | harmonic 5 | 7 | 9 | 11 | 13 | arithmetic, difference step | exact against differences |
|---|---|---|---|---|---|---|---|
| 79.77 | 7.42e-13 | 2.46e-16 | 1.68e-19 | 2.88e-22 | 1.36e-24 | quad, 2⁻⁷⁰ | 7.5e-10 |
| 100 | 3.14e-13 | 5.93e-17 | 1.95e-20 | 1.17e-23 | 1.25e-26 | quad, 2⁻⁷⁰ | 2.8e-4 |
| 150 | 1.73e-13 | 2.79e-17 | 7.44e-21 | 5.22e-24 | 6.42e-27 | quad, 2⁻⁷⁰ (octo, 2⁻¹⁰⁰: the same values) | 4.4e-6 (6.7e-22) |
| 175 | 2.63e-14 | 1.29e-17 | 1.18e-20 | 1.93e-23 | 5.07e-26 | octo, 2⁻¹⁰⁰ | 1.6e-18 |
| 200 | 2.98e-17 | 5.01e-20 | 1.68e-22 | 9.66e-25 | 8.30e-27 | octo, 2⁻¹⁰⁰ | 6.1e-13 |
| 232.6 | 1.06e-24 | 3.45e-26 | 1.70e-27 | 1.05e-28 | 7.07e-30 | octo, 2⁻¹¹⁰ | 1.5e-9 |

Each value is one of its cosine-sine pair, which agree to 1e-3 or better (raw `I479-mac/` e3b67c22,
`hp/data/held_grip.csv`). Past 175 the grip on every held harmonic falls fast with D: harmonic 5's by 900 times from 175
to 200 and by 3×10⁷ from 200 to 232.6. The fall from one harmonic to the next flattens there, to 15–30 times at
232.6.

### 4.4 Releasing the held harmonics: the fit's own error at D 74–77 (#640, #642)

`hp/release_held.py` re-converges a chain state at its own D with harmonics 5–13 free (`--kfreeze 13`, the full
system), in quad-double, so Δ_released − Δ_held is the held fit's error at that D. Its planted control moves T81's
full-system state at 73.0 by 1e-10 in the held coefficients and must return to the unperturbed release (Δ within
1e-25, coefficients within 1e-20), while the run with no Newton step must fail. It passed on win and macm3. The
unperturbed release moves Δ by −8.4e-28 on kernels from before #523 (win and macm3 agree to every digit) and by
−2.76e-23 on kernels from #523 on (the exact freeze projector, two builds on the Mac agree), so the floor is the
projector change, not the platform.

| D | released | Δ_released − Δ_held | largest move of a held coefficient | `fits` spread in delta_all.csv |
|---|---|---|---|---|
| 74 | yes, 6 steps | +4.62e-16 | 1.8e-10 (harmonic 13) | 1.4e-15 |
| 75 | yes, 7 steps | +1.25e-15 | 3.8e-10 | 3.9e-15 |
| 76 | yes, 7 steps | +2.43e-15 | 5.9e-10 | 7.6e-15 |
| 77 | yes, 8 steps | +4.06e-15 | 8.2e-10 | 1.3e-14 |
| 78, 79 | stalled at \|F\| 1e-20 and 2e-19 | not a value | | 2.0e-14, 3.0e-14 |
| 80, 100, 150, 200 | stalled or every trial refused | not a value | | 4.2e-14 to 2.8e-11 |

Where it converges, the measured error is about a third of the `fits` spread quoted for the same D, so the spread
is a conservative bound there.

**A partial release reaches D 200** (#641, macm3; raw `I641-macm3/` 5c3443e3). With `anm_run.py --kfreeze 7
--held_fit`, harmonics 5 and 7 are solved and 9–13 stay held on the fit. Its base polish converges at every D tried:

| D | Δ_kf7 − Δ_held | `fits` spread | ratio |
|---|---|---|---|
| 80 | +3.67e-14 | 4.2e-14 | 0.87 |
| 100 | +9.69e-13 | 1.2e-12 | 0.81 |
| 150 | +9.52e-12 | 1.1e-11 | 0.87 |
| 200 | +2.00e-11 | 2.8e-11 | 0.71 |

This is the part of the held fit's error that comes from harmonics 5 and 7. It is not a bound on the whole error,
since 9–13 stay held and their part may have either sign. It is 0.71–0.87 of the quoted spread at every D, so the
spread is the right size from D 80 to 200. Releasing harmonic 9 too (`--kfreeze 9`) converges at D 80 only
(+1.21e-14), and its polish stalls at D 100, 150 and 200. The control, the unreleased system, returns the stored Δ to
−7.07e-29. Above 77 no release converges from the chain's states, although they satisfy the full
system to 5e-26 to 4e-23 at 80–200. The first Newton step raises |F| from about 1e-24 to 1e-14, along the near-null
directions of section 4.3, so the stop is the Newton step's and not the held values'. Raw `I642-win/` and
`I640-macm3/` (725ecb92).

**Continued in D, the full system reaches D 100** (#682, win; raw `I682-win/` 0e9206ac, `cont_table_682.txt`).
Instead of releasing each chain state at its own D, `anm_run.py --kfreeze 13` without `--held_fit` continues the D 77
release in D, with the quad-double jet predictor, re-seed and corrector (`--tol_qd 1e-44`). It converges at 33 points
from D 78 to 100.19, each to a quad-double max|F| of at most 4.4e-45. The prediction's |F| is flat in the step a
(slope −0.11, 5.8e-36 to 5.2e-35 over 44 attempts), so the predictor's error is not first order there.

| D | Δ_full − Δ_held | `fits` spread | ratio | Δ_kf7 − Δ_held (#641) |
|---|---|---|---|---|
| 78 | +6.20e-15 | 2.0e-14 | 0.31 | |
| 80 | +1.22e-14 | 4.2e-14 | 0.29 | +3.67e-14 |
| 90 | +8.41e-14 | 3.8e-13 | 0.22 | |
| 99 | +2.06e-13 | 1.1e-12 | 0.19 | |
| 100 | +2.22e-13 | 1.2e-12 | 0.19 | +9.69e-13 |

**The held fit's whole error is 0.19–0.31 of the quoted spread from D 78 to 100, and smaller than the part from
harmonics 5 and 7 alone.** So harmonics 9–13 contribute with the opposite sign, −2.4e-14 at D 80 and −7.5e-13 at
D 100. The comparison shares its D 77 origin and its kernel with the held chain, so it is not an independent check
of that chain. Past D 99 the corrector stalled at a ≥ 0.1, where five attempts ended at |F| 5e-25 to 6e-24, and the
step fell to about 0.05. The cause is the quad-double corrector's projection switch, as in the octo-double crawl of
notes/CRAWL.md. Every point lay below it, so the horizon rows' second-order term was never eliminated. With
`--elim_switch 0` both refused attempts converge in 3 steps to 6e-49, and the continuation's steps grow back to a = 1
up to D 104.5 (#710, raw `I710-win/` 978179f6). Their Δ differs from the default switch's states at the same D by
3e-26 and 1e-25. The double-double |F| of the converged states rises from 2.9e-20 at D 78 to 9.0e-17 at D 100. That
has not been investigated.

How far two converged roots agree depends on D (#725, raw `I725-win/` 9672e70d). Two a = 1/16 steps from cont3's
D 99.75 state, each converged to a quad-double max|F| of 2e-49 to 4e-49, reproduce cont3's Δ to 5.7e-34 at D 99.8125
and only to 1.7e-25 at D 99.875. The three converged states at D 99.875 (cont3's, #710's A1 and #725's X2) span
2.0e-25 in Δ. A weak direction of fixed gain near 5e-31 would allow a larger difference at D 99.8125 than the one
seen, so it does not explain both D by itself.

**The spread is the segment plan, not three roots** (#727, raw `I727-win/` 34651ada). The three D 99.875 states were
converged on different plans, with interior nodes up to 9.5e-4 apart, while the two D 99.8125 states shared one plan.
Copied onto cont3's plan and polished in octo-double to 2.4e-62 or below, all three give Δ =
1.745150028164499252944352893640343550633, the same to 1.1e-46. The first Newton step on the common plan moves A1 and
X2 by exactly their offsets on their own plans. So at D 99.875 the plan sets Δ to about 25 digits.

One node does not carry that dependence (#741, raw `I741-win/` 86eeacdc). Shifting the plan's node `segs[3][1]` by
1e-4 to 1e-2 moves Δ by −1.0e-27, +7.9e-28, +2.6e-27, +8.5e-28 and +8.2e-29. The effect is not a power law, and it
does not grow with the shift. Each value is reproduced from two other starts to 1.9e-54 or better, so it belongs to
the plan. It is at most a few 1e-27, against #727's offsets of 3.0e-26 and 1.7e-25 between plans that moved several
nodes by 5e-4 to 9.5e-4. #751 shifts each of the other six nodes.

With `--elim_switch 0` the continuation reaches D 112.75 (#726, raw `I726-win/` 647cd9f5). Every accepted step
converges to at most 7.0e-45. The difference to the held chain keeps growing, by a factor of 1.055 per unit of D
falling to 1.039, while the quoted spread grows faster:

| D | Δ_full − Δ_held | `fits` spread | ratio |
|---|---|---|---|
| 105 | +3.05e-13 | 1.8e-12 | 0.170 |
| 108 | +3.56e-13 | 2.3e-12 | 0.155 |
| 112 | +4.21e-13 | 2.9e-12 | 0.145 |

**So the held fit's error stays at 0.15–0.17 of the quoted spread up to D 112, below the 0.19 of D 99–100.** The
expectation written before the run, that the ratio stays near 0.19, failed in that direction. From D 112.75 the four
attempts, at a = 1/4 down to 1/32, all stall between 1.1e-44 and 7.8e-44 against the bar of 1e-44, and the run stops
there. The stall does not fall with a, while the first full step's |F| falls from 1.05e-21 to 1.7e-23, so it is a floor
of the corrector, not the step's quadratic error. The 17 Newton steps taken at the floor move Δ by at most 4.4e-45, and
the elimination's pivots span 8e-20 to 9e14.

**The floor is the quad-double precision** (#739, raw `I739-win/` 51a8e41f). From the D 112.75 state, the octo-double
corrector with `--reseed_o 3` converges the same two attempts, to D 112.875 and D 113, to 7.8e-89 and 7.3e-85 in five
steps each. An octo-double attempt takes 1.2 to 1.4 times as long as a stalled quad-double one. Its control, the
D 112.75 state polished alone in octo-double, keeps the quad-double Δ to 7.0e-46. At D 113, Δ_full − Δ_held is
+4.37e-13, 0.146 of the spread. Octo-double changes the corrector and the projection together, so which of the two
sets the quad-double floor is not separated.

The octo-double corrector carries the full system on to D 120 (#742, raw `I742-win/` c5ec16fb). Every accepted step
converges to at most 1e-60. The steps are at a = 1 to D 117, and at a = 1/2 from there, since the a = 1 step from D 117
crawled under both trial tests (#752 asks why).

| D | Δ_full − Δ_held | `fits` spread | ratio |
|---|---|---|---|
| 113 | +4.37e-13 | 3.0e-12 | 0.146 |
| 116 | +4.81e-13 | 3.5e-12 | 0.137 |
| 120 | +5.32e-13 | 4.2e-12 | 0.127 |

The shift grows smoothly from D 105 to D 120. Its ratio per unit of D falls by about 0.002 a step, from 1.055 to 1.023,
and is continuous across the change of step size. **So the held fit's error falls to 0.127 of the quoted spread at
D 120**, since the spread grows by a factor 1.4 from D 113 and the shift by 1.22. The expectation written before the
run, a ratio near 0.146, failed in that direction.

### 4.5 Not run

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

- **Proposed text for QUAD_DOUBLE.md 13.19 and 13.29** (the held harmonics' bias): "The bias is harmonic 5's held
  trend, a power law fitted over 2.6 units of D and used over 6–140. The pinned system's first-order response to the
  held values, s · (h_fit − h_chain), with h_fit a fit of the full-system states in 1/n, accounts for the chain's
  whole gap to the series at 79.77–200.27, leaving 6e-14 to 9e-11 (notes/HELD_HARMONICS.md 4.2). Harmonics 11 and 13,
  held constant at polluted values, cost about 3e-12. A chain on the fit (`anm_run.py --held_fit`) from the
  full-system state at 73.0 lies on the series to 1e-13 up to 79.83. So the gap to arXiv:2611.07007 above D 80 is the
  published values' minus the series, +4.3e-6 at 200, with our side of it removed; this is numerical evidence that
  rests on the two extrapolations agreeing."
- **The existing chain to 214.** Its stored states can be corrected to first order by s · (h_fit − h_chain), with one
  LU per state (`held_fit.py sens`, then `series`), quad-double below about 185 and octo-double above. The correction
  agrees with the series to 9e-11 at 200, where harmonic 5 is held 18% off. That agreement is with an extrapolation,
  though. A restarted chain on the fit, or a `--kfreeze 7` chain that solves harmonics 5 and 7, is the independent
  test of the correction at 100–200.
- **Raw output** of this task: `chain_fit73/` (8 states and `anm_run.jsonl`, the logs beside it), `chain_wrong5/` (the
  control step) and the sensitivity JSONs, all in the raw-data repository's `held-harmonics-mac/` (acaccfa).

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
  Deflation of known solutions, which 4.5 sets aside.
- A. N. Tikhonov, *Solution of incorrectly formulated problems and the regularization method*, Soviet Math. Dokl. 4
  (1963) 1035–1038, the translation of Dokl. Akad. Nauk SSSR 151 (1963) 501–504. One library record gives the
  translation's volume as 5 *(volume unverified)*. The smoothest-solution penalty of 4.5.

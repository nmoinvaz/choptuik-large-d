# Held harmonics: what holding them costs Δ at large D, and a better way to hold them (2026-11-06)

## Recommendation

PLACEHOLDER

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

Planted truth with s from the nearest of 73.0 and 79.77 (`held_fit.py planted`, degree 8):

| method | 68.5 | 70.5 | 71.5 | 73.0 |
|---|---|---|---|---|
| B1, the chain's trend from T75's 73.27 state | −1.28e-8 | −1.03e-9 | +1.19e-9 | +5.61e-10 |
| B2, the trend's rule on dd 58.01–61.03, from 61.03 | −1.02e-7 | −1.47e-7 | −1.71e-7 | −2.09e-7 |

B1 at 73.0, a quarter unit from its base, is +5.6e-10. Its base already carries a held error, and B1 moves away
from the full Δ as D moves away from 73.27 in either direction. B2 is the same rule in the chain's geometry, a fit
over 3 units extrapolated 7.5–12 units ahead, and it is off by 1–2e-7, the size of the chain's bias at 80–86.

### 4.2 A fit of the full-system states in 1/n (`held_fit.py`, `anm_run.py --held_fit`)

**The method.** For each held coefficient v, √n v is a Chebyshev series of degree 8 in x = 1/n, least squares on 146
full-system states, 30.07 ≤ D ≤ 73.0. These are the double-double states of `hp/data/states` and of T12-win and T13-win,
and the quad-double states of T84-mac and T81. The form is the centre scaling p ≈ x F(θ)/√n, with F converging as D
grows (README, "How the solution scales with D"). `pinning.HeldFit` evaluates the fit and its Taylor coefficients in D
by power-series arithmetic in mpmath, and `anm_run.py --held_fit FILE` holds the harmonics on it. As with the trend,
the held values are taken relative to the base, v(D) = v(D₀) + fit(D) − fit(D₀). The fit's largest residuals are
5.8e-13 for harmonic 5, 1.4e-11 for 7 and 5e-10 for 9. For 11 and 13 they are 1e-8 and 1e-7, the noise of the
double-double states in those harmonics (the jump study of QUAD_DOUBLE.md 13.19 found the same), which costs Δ at most
1e-13. `held_fit.py check` compares the Taylor coefficients with the values (2.2e-16 relative, pass below 1e-14), and
its control `--mutate=jet` fails at 1.4e-10.

**Planted truth** (s from the nearest of 73.0 and 79.77; degree 8, then 6 and 10):

| fit | predicted | degree 8 | degree 6 | degree 10 | criterion |
|---|---|---|---|---|---|
| P1, every full state with D ≤ 71.0 | 71.5, 72.8, 73.0 | −0.8e-15 to −2.4e-15 | +1.2e-14 to +3.1e-14 | +2.1e-15 to +7.9e-15 | 1e-12, pass |
| P2, the double-double states alone (D ≤ 61.03) | 68.5–73.0 | −9.6e-15 to −3.4e-14 | +6.4e-14 to +1.9e-13 | +1.5e-13 to +6.8e-13 | 1e-10, pass |

**Control:** with `--mutate=scale5`, P1 fails at 71.5–73.0 with biases of +2.08e-11 to +2.14e-11, and the run exits 1.

Over the distance at which B2 is off by 1–2e-7, P2 is off by at most 7e-13 at any of the three degrees.

PLACEHOLDER-SERIES

### 4.3 Holding less: solve the well-determined held directions (`held_schur.py`)

PLACEHOLDER-SCHUR

### 4.4 Not run

PLACEHOLDER-NOTRUN

## 5. References

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

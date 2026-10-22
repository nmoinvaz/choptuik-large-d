# Low D: how close to D = 3 the solvers get

Status of 2026-10-22 (cloud, 2 shared cores; load 10-25 throughout). This note answers the question of Ecker et al. how
close to D = 3 our solvers get, with the convergence evidence for every value. Status words as in AGENTS.md: every
number here is *numerical evidence* unless marked otherwise.

## Summary

- **Lowest D with a demonstrated-converged Delta:** *pending (D = 3.7, double-double, M = 48 / 64)*.
- **The obstruction is not precision but representation.** Below D = 4 the echo's harmonic content grows (the centre
  slope's harmonics decay like e^(-beta k) with beta = 0.157 at D = 4, 0.134 at 3.7, about 0.03-0.05 at 3.05 in Ecker
  et al.'s data) and its amplitude grows like (D - 3)^(-1.07). Three things follow, each measured here: more tau modes
  (about 75 at D = 4 and 85-90 at D = 3.7 for 1e-10 in Delta), centre and horizon series that converge ever more
  slowly at the matching points used at large D, and Newton paths from coarser solutions that need many damped steps.
- **A converged residual is not a converged answer.** The double-double solver reached |F| = 3e-30 at D = 4 with the
  large-D node set while its centre series still had terms of 7e-6 at the first node; Delta moved by 1.35e-7 when the
  nodes were fixed (Sec. 3). This is the authors' warning in concrete form.
- **D -> 3 itself:** at n = D - 2 = 1 the field equations decouple (c = 1, the scalar obeys the free 2+1 wave
  equation) and the free wave has no time-periodic self-similar solution regular at the centre and analytic at the
  light cone (*proved*, the n = 1 case of Proposition 2, checked at 40 digits). So the D -> 3 limit of the family cannot
  be a regular solution; the measured amplitude growth (D - 3)^(-1.07) and spectral flattening say how it degenerates
  (*conjecture* for the limit itself).

## 1. What failed before (T15, comms.txt 2026-10-20/21)

- kc 47 (scan_lowD3_kc47.jsonl): the first step 4.0 -> 3.95 failed (|F| 1.3e-4). Re-seeded at 3.7938 and 3.7, it reached
  3.675 (|F| 1.8e-10) and 3.6425 (2.3e-9, accepted only by the relaxed NE_STRONG_TOL 2e-8), then failed at 3.6214 and
  3.6002 (|F| 1.5e-6, 3.4e-6).
- kc 55: D = 4.0 and 3.7938 converged; at 3.7 the regridded kc 47 state landed on a false minimum (|F| 7.8e-5, the same
  point to all digits after a long LM). Continuation from 3.7938 needed steps <= 0.0125 at 1000-4800 s each, reached
  3.7689, then failed at 3.7564-3.7626 with weak residuals 8e-11 to 1.5e-10 above the 5e-11 cut.
- kc 63: only D = 4.0 was accepted (weak residual 1.6e-11), 1.0e-6 above kc 55 and above Martin-Garcia & Gundlach;
  3.7938 and 3.7 stalled at strong residuals 8e-7 and 1e-6.

Why, from the measurements below: a seed without harmonics 49-63 is far from the kc 55/63 solution in the near-centre
blocks (the double-double solver shows the same at M = 64: |F| 4e-2 in the first block), the horizon series at
u0 = 0.4 converges ever more slowly as D falls, and at kc >= 55 the top centre and horizon harmonics fall into
ne_solve's weak band, where the f64 Schur-complement step stalls at 1e-12 to 1e-10. With |dDelta/dF|_1 of 2e3 to 8e4 at
kc 47 and larger at higher kc, a weak residual of 1.6e-11 can be worth 1e-7 to 1e-6 in Delta: the order of the kc 63
error at D = 4.

## 2. The obstruction, measured

### Harmonic content

`cd solver/data && python3 ../lowD_spectrum.py neB_kc47_rk2_D4.0000.npy ... --fit=21:35` (records in
solver/data/lowD_spectrum.jsonl; the fit window and the planted-truth selftest, `--selftest`, are in the script):

| D | state | \|F_1\| | F_3/F_1 | beta (k = 21-35) | F_31 | F_47 |
|---|---|---|---|---|---|---|
| 4.0 | f64 kc 47 | 16.23 | 0.438 | 0.157 | 5.5e-2 | 2.9e-3 |
| 3.875 | f64 kc 39 | 18.37 | 0.451 | 0.158 | 8.0e-2 | |
| 3.794 | f64 kc 47 | 20.12 | 0.459 | 0.141 | 1.2e-1 | 7.9e-3 |
| 3.7 | f64 kc 47 | 22.65 | 0.469 | 0.134 | 1.8e-1 | 1.3e-2 |
| 3.046 | Ecker et al.'s exported field | about 435 | 0.556 | about 0.03-0.05 | | |

F = sqrt(D - 2) dp/dx at x = 0 (README normalisation). The kc 39 states give beta 0.01-0.012 higher than kc 47 at the
same D (the truncation depresses the top harmonics). The D = 3.046 row is from Ecker et al.'s published field
(plots/fields_3.046000_0.json in their repository: U/x at x = 5e-5 times sqrt(n); their D = 4 field gives
|F_1| = 16.21 and beta = 0.153, against our 16.23 and 0.157, so the normalisations agree). Its tau samples are not exactly
one uniform period (4% antiperiodicity error at the best period), so its beta is indicative only.

- **Amplitude:** |F_1| grows like (D - 3)^(-q) with q = 0.93 between D = 4 and 3.7 and 1.09 between 3.7 and 3.046.
- **Delta's error in the mode count** falls like e^(-2 beta kc): at D = 4 the steps kc 23 -> 31 -> 39 -> 47 (README) and
  47 -> 55 (T15) shrink by 12.6, 12.1, 11.2 per 8 modes, 0.30-0.32 per mode, twice beta. hp/lowD_cost.py turns this into
  the modes needed (its inputs and a planted check are in the script): *pending numbers*.

### Conditioning (f64)

`cd solver/data && python3 ../lowD_svd.py <state> --json=lowD_svd.jsonl` (exact Rust Jacobian, full SVD):

| D | kc | s_min/s_max | weakest direction | \|dDelta/dF\|_1 | weak / junk band |
|---|---|---|---|---|---|
| 4.0 | 47 | 1.9e-7 | centre p, k = 45 | 2.0e3 | 0 / 0 |
| 3.875 | 39 | 2.7e-7 | centre p, k = 37 | 5.5e3 | 0 / 0 |
| 3.794 | 47 | 4.5e-8 | centre p, k = 45 | 2.3e4 | 0 / 0 |
| 3.7 | 47 | 1.7e-8 | centre p, k = 45 | 7.8e4 | 0 / 0 |

At kc 47 nothing is near-null down to D = 3.7: the ladder of centre-data harmonics is 1e-5 to 1e-8, not the 1e-10 to
1e-24 "junk" of large D. It steepens as D falls (0.15 per harmonic at D = 4, 0.29 at 3.7), so the harmonics that 1e-10
needs (k about 85-95 at 3.7) sit near 1e-14 s_max: at f64 noise. The kc 55 state in the repository
(neB_kc55_rk2_D3.7689.npy) is not a converged state in this build (|F| 2.5e-8 at its D = 3.7688625, all of it in the
matching block; its file name rounds D, and at 3.7689 |F| = 1.5e-5), so its row (five weak directions, |dDelta/dF|_1
5e5) is not used.

### Series at the matching points

f64, horizon series at u0 = 1 - x_K = 0.4 (`python3 ../lowD_horizon.py`, solver/data/lowD_horizon.log):

| D | change of the sum, Ks 60 -> 90 | 90 -> 120 | radius in u (root test) |
|---|---|---|---|
| 4.0 | 8.6e-11 | 1.5e-14 | 0.57-0.62 |
| 3.794 | 7.8e-9 | 6.3e-12 | 0.52-0.56 |
| 3.7 | 7.6e-8 | 1.4e-10 | 0.48-0.53 |

The truncation at the production Ks = 60 grows tenfold per 0.1 in D. It moves Delta(3.7, kc 47) by +3.7e-8 (Ks 90);
RK steps x4 move it by +8.8e-9 (solver/data/lowD_f64.jsonl). Both are far below the kc error there (about 3e-6).

Double-double (`python3 hp/lowD_series.py <state> --kc=200 --ks=300`, hp/data/lowD_logs/series.log), D = 4, M = 48:
the centre series at the large-D first node x = 0.02 has last terms 6.7e-6 with 60 terms (1e-9 with 100, 2e-13 with
150; the radius is about 0.025), while at x = 0.01 60 terms leave 2.3e-23. The horizon series has radius about 0.455
in u; at u = 0.4 a hundred terms leave 5.7e-7, at u = 0.3 two hundred leave 2.8e-37. The solver had converged to
|F| = 3.3e-30 with 60 / 100 terms at 0.02 / 0.4. Moving to nodes 0.01, 0.02, ..., 0.6, 0.7 with 80 centre and 200
horizon terms changed Delta(4, M = 48) from 3.445452796165 to 3.445452661587 (-1.35e-7). All double-double values
below use the new node set, whose series tails (recorded per point as tail_c, tail_h) are below 1e-48.

### Newton

From an f64 kc 47 seed the double-double Newton goes straight to M = 64 only by failing (line search at |F| 4.3e-2,
all of it in the first block, centre series to first node: the seed lacks harmonics 49-63; filling them by each field's
geometric trend, `--fill_tail`, only brings the start to 1.4e-2). Through
M = 48 it converges in 10-11 Jacobians, the first five damped or linear (|dz| halving per step), then quadratic. From a
converged M = 48 state, M = 64 converges quadratically in 4-5 (|F| 2e-4, 2e-8, ...). The weakest directions of the
double-double Jacobian at D = 4, M = 48 are the centre p harmonics k = 31-47 at 6e-8 to 1e-6 s_max
(hp/lowD_weak.py, hp/data/lowD_weak.jsonl): well conditioned at double-double precision.

### Cost toward D = 3

*pending: the per-iteration times at M = 48 and 64 and the extrapolation.*

## 3. Converged points

*pending*

## 4. gamma

*pending*

## 5. Comparison with Ecker et al. and Bland et al.

*pending*

## 6. What should happen as D -> 3

- **At n = 1 the equations decouple** (`python3 solver/lowD_n1_check.py`, log solver/data/lowD_n1_check.log): in
  solver/dss.py's equations c_x = c (n - 1)(a^2 - 1)/x vanishes, so c = c(tau, 1) = 1 by the gauge, and the p, m
  equations no longer contain a. The scalar obeys the free 2+1 wave equation in self-similar form (checked on the code's
  right-hand side; the control n = 1.01 shows the coupling).
- **The free wave cannot echo at n = 1.** Proposition 2's argument goes through at kappa = (n + 1)/2 = 1: the solution
  regular at the centre is p - m = x 2F1((1+s)/2, (2+s)/2; 1; x^2) (the other has a logarithm), and its coefficient of
  the non-analytic light-cone branch (1 - x^2)^(-1/2 - s) is C(1, s) = Gamma(s + 1/2)/(Gamma((1+s)/2) Gamma(1 + s/2)),
  which has no zero for imaginary s != 0 (*proved*; the solution, the first-order equations and Gauss's connection
  formula checked at 40 digits for mu = 1, 3, 5 and omega = 1-3.82, |C| >= 0.13 there).
- **Consequence.** A finite-amplitude regular limit of the DSS family at D = 3 would be a nonzero periodic regular
  solution of the free equation, which does not exist. So as D -> 3+ the family must degenerate: amplitude to infinity,
  loss of regularity, or both. The data show the amplitude growing like (D - 3)^(-1.07) and the spectrum flattening
  (beta from 0.157 to about 0.04), consistent with Ecker et al.'s narrow, large-amplitude peaks (their Psi reaches about
  3e5 at D = 3.05). Gravity's only coupling is the factor n - 1 = D - 3, so the field must grow until (D - 3) a^2 is of
  order one; a heuristic, not derived here. That Delta -> 0 (Ecker et al.'s conjecture, their fit 3.878 (D - 3)^0.15 near
  3.01-3.05 in their plotting notebook) is not tested by anything here.
- **In 2+1 itself** (Lambda < 0 needed for black holes) the critical solution is continuously self-similar with a scalar
  field linear in tau (c != 0), modelled by Garfinkle's n = 4 solution, with gamma = 8/7 predicted and 1.23(4) measured
  (Jalmuzna, Gundlach & Chmaj, PRD 92, 124044 (2015), arXiv:1510.02592; the review of Gundlach, Hilditch & Martin-Garcia,
  arXiv:2507.07636, Sec. IV.A.1, quotes gamma = 1.20 +- 0.05 from Pretorius & Choptuik). Nothing connects that solution
  to the D -> 3+ limit of the asymptotically flat DSS family; Delta -> 0 would be suggestive of a CSS limit, gamma -> 0
  (Ecker et al.) would not match 8/7.

## 7. Literature read

- Ecker, Ecker, Grumiller & Jechtl, SciPost Phys. 20, 176 (2026), arXiv:2602.10185: Secs. 1-5 read in the arXiv PDF
  text; Sec. 6.2 (the small (D - 3) expansion) and Appendices C, D (convergence, tables) could not be read, the fetched
  text ending in Sec. 6; their data repository was used instead (hp/lowD_ecker.py). Read: range 3.05 <= D <= 5.5;
  "numerically too costly to go below D = 3.05"; obstacles: narrow, large-amplitude peaks of Psi near the centre (more
  Fourier modes) and many x points (N_x = 1.55e5 at D = 3.05, 8000 at D >= 4); N_tau = 512 enough for D >~ 3.2, at least
  1024 below; x_L = 5e-5 for D < 3.1; results "converge to machine precision for any number of Fourier modes" but differ
  from the continuum limit; about 1e-6 claimed at D = 4 and at the maximum. Conjecture: Delta -> 0 and gamma -> 0 as
  D -> 3+, gamma faster.
- Jalmuzna, Gundlach & Chmaj (2015), as in Sec. 6 (abstract and text read on arXiv; DOI 10.1103/PhysRevD.92.124044).
- Gundlach, Hilditch & Martin-Garcia (2025), arXiv:2507.07636, Sec. IV.A.1-2 (read).
- Bland, Preston, Becker, Kunstatter & Husain, gr-qc/0507088, Table I (read): gamma(3.5) = 0.349 +- 0.003, 4: 0.374 +- 0.002,
  5: 0.412 +- 0.004, from fits of the apparent-horizon radius in evolutions; the error is the range of a* over which the fit
  stays good, "about 1%". Bland's 2007 thesis values (gamma(3.5) = 0.3235 +- 0.0018, Delta(3.5) = 3.411 +- 0.017) are taken
  from Ecker et al.'s data file plots/gBlandThesis.dat; the thesis itself was not read *(unverified)*.

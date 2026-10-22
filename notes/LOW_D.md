# Low D: how close to D = 3 the solvers get (draft, in progress)

Draft written while the runs are going; sections marked *pending* are being computed. The final version replaces it.

## 1. What failed before (T15, comms.txt 2026-10-20/21)

- kc 47 (scan_lowD3_kc47.jsonl): the first step 4.0 -> 3.95 failed (|F| 1.3e-4). Re-seeded at 3.7938 and 3.7, it reached
  3.675 (|F| 1.8e-10) and 3.6425 (2.3e-9, accepted only by the relaxed NE_STRONG_TOL 2e-8), then failed at 3.6214 and
  3.6002 (|F| 1.5e-6, 3.4e-6).
- kc 55: D = 4.0 and 3.7938 converged; at 3.7 the regridded kc 47 state lands on a false minimum (|F| 7.8e-5, the same
  point to all digits after a long LM). Continuation from 3.7938 needed steps <= 0.0125 at 1000-4800 s each, reached
  3.7689, then failed at 3.7564-3.7626 with weak residuals 8e-11 to 1.5e-10 above the 5e-11 cut.
- kc 63: only D = 4.0 was accepted (weak residual 1.6e-11), at Delta 1.0e-6 above kc 55 and above Martin-Garcia &
  Gundlach, the wrong way; 3.7938 and 3.7 stalled at strong residuals 8e-7 and 1e-6.

## 2. Diagnosis (measured here)

### f64

- Harmonic content (solver/lowD_spectrum.py): the centre slope F = sqrt(n) dp/dx|0 decays like e^(-beta k) with
  beta = 0.157 (D = 4), 0.142 (3.79), 0.135 (3.70) over k = 21..39; |F_1| = 16.2, 20.1, 22.7. Delta's error in the mode
  count falls like e^(-2 beta kc) (D = 4: 0.31 per mode from the kc 23..55 sequence).
- Conditioning (solver/lowD_svd.py, solver/data/lowD_svd.jsonl): at kc 47 every singular value is >= 1.7e-8 s_max at
  D >= 3.7 (the weakest are the centre-data harmonics k = 45, 39, 35, ...); no direction in ne_solve's weak (< 1e-10) or
  junk (< 3e-14) band. The ladder steepens as D falls (s_min/s_max 1.9e-7, 4.5e-8, 1.7e-8 at D = 4, 3.79, 3.7) and
  |dDelta/dF|_1 grows (2.0e3, 2.3e4, 7.8e4).
- Horizon series at u0 = 1 - x_K = 0.4 (solver/lowD_horizon.py, solver/data/lowD_horizon.log): its truncation at Ks = 60
  grows tenfold per 0.1 in D: the series sum changes by 8.6e-11 (D = 4), 7.8e-9 (3.79), 7.6e-8 (3.70) from Ks 60 to 90.
  Delta(3.7, kc 47) moves by +3.7e-8 from Ks 60 to 90 (solver/data/lowD_f64.jsonl).
- The kc 55 state in the repository (neB_kc55_rk2_D3.7689.npy, the D = 3.7688625 point) has a horizon series that
  converges badly at u0 = 0.4 (sum changes 3.7e-5 from Ks 60 to 90, 4.1e-6 from 90 to 120), unlike the kc 47 states, and a
  residual of 2.5e-8 in the matching block in this build (1.5e-16 elsewhere).

### Double-double

- Newton from the f64 kc 47 seed: straight to M = 64 the line search fails at once (|F| 4.3e-2, in the centre block: the
  seed lacks harmonics 49-63, which the near-centre dynamics amplify). Through M = 48 it converges, linearly at first
  (|dz| halving per step for five steps), then quadratically: 11 Jacobians at D = 4.
- Series truncation (hp/lowD_series.py, hp/data/lowD_logs/series.log): with the nodes used at large D (first 0.02, last
  0.6) and 60 centre / 100 horizon terms, the D = 4, M = 48 solution reached |F| = 3e-30 while the centre series at
  x = 0.02 still had terms of 7e-6 and the horizon series at u = 0.4 terms of 6e-7 (radius about 0.455). A converged
  residual does not measure this. At x = 0.01 sixty centre terms leave 2e-23; at u = 0.3 two hundred horizon terms leave
  3e-37. The low-D runs therefore use nodes 0.01, 0.02, ..., 0.6, 0.7 with 80 centre and 200 horizon terms.

## 3. Literature read so far

- Ecker, Ecker, Grumiller & Jechtl, SciPost Phys. 20, 176 (2026), arXiv:2602.10185 (Secs. 1-5 read in the arXiv PDF
  text; Sec. 6.2 and Appendices C, D could not be read: the fetched text ends at Sec. 6). Range 3.05 <= D <= 5.5;
  "numerically too costly to go below D = 3.05"; obstacles: narrow, large-amplitude peaks of Psi near the centre (more
  Fourier modes) and many x points (N_x = 1.55e5 at D = 3.05, 8000 at D >= 4); N_tau = 512 enough for D >~ 3.2, at
  least 1024 below. Claimed accuracy about 1e-6 at D = 4 and at the maximum. Conjecture: Delta -> 0 and gamma -> 0 as
  D -> 3+, gamma faster. Their data repository (github.com/EckerChristian/CriticalSpacetimeCrystal) has Delta and gamma
  per resolution; hp/lowD_ecker.py tabulates it.
- Jalmuzna, Gundlach & Chmaj, PRD 92, 124044 (2015), arXiv:1510.02592: in 2+1 (Lambda < 0 needed) the critical
  solution is CSS with c != 0, modelled by Garfinkle's n = 4 solution; gamma = 8/7 predicted, 1.23(4) measured.
- Bland et al. 2005 (gr-qc/0507088), Table I: gamma(3.5) = 0.349 +- 0.003 (about 1%, from the a* fit range).

*pending*: the double-double values at D = 4 and 3.7 with their convergence, gamma, the f64 checks at 3.7, the
comparison table, the cost extrapolation and the local task.

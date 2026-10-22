# Low D: how close to D = 3 the solvers get

Status of 2026-10-22 (cloud, 2 shared cores, load 10-25 for most of the day). Ecker et al. asked how close to D = 3
our solvers get. This note gives the values reached, the convergence evidence for each, why it gets hard, and a task
for the local machine to go further. Status words as in AGENTS.md; every number is *numerical evidence* unless marked.

## Summary

- **Lowest D with a demonstrated-converged Delta: D = 3.7** (double-double solver): Delta(3.7) = __D37__, converged in
  the tau resolution M = 48 / 64 / 80 and unchanged by the node layout, series lengths and Taylor-step tolerance to
  __D37VAR__ (Sec. 3). gamma(3.7) = __G37__. The f64 solver does not get below D = 4 with a demonstrated 1e-8; its
  kc 47 values at 3.64-3.7 are 3.6-5.2e-6 high (Sec. 3).
- **Validation at D = 4:** Delta = 3.4454524022410 (M = 80; M = 48 / 64 / 80 differences -2.6e-7, -1.8e-9), against
  Martin-Garcia & Gundlach's 3.445452402(3): 2.4e-10. gamma(4) = 0.3739607120 (M = 64).
- **The obstruction is representation, not precision.** Below D = 4 the echo's harmonics decay more slowly
  (|F_k| ~ e^(-beta k): beta = 0.157 at D = 4, 0.134 at 3.7, about 0.03 at 3.05 in Ecker et al.'s data) and its
  amplitude grows like (D - 3)^(-1.09). This costs modes (about 73 at D = 4, 85-88 at 3.7, 104-118 at 3.5, 135-177 at
  3.3, several hundred near 3.05 for 1e-10 in Delta), and with the modes the centre and horizon series converge ever
  more slowly at the matching points, so the nodes must move toward the centre and the horizon as M grows.
- **A converged residual is not a converged answer,** twice over here: the double-double solver reached |F| = 3e-30 at
  D = 4 (M = 48) and at D = 3.7 (M = 64) on node sets whose series were still truncated at 1e-6, and Delta moved by
  1.35e-7 and 4.1e-8 when the nodes were fixed. Each value below records its series tails.
- **D -> 3:** at n = D - 2 = 1 the scalar decouples from gravity in these equations, and the free wave has no
  time-periodic self-similar solution regular at the centre and analytic at the light cone (*proved*, the n = 1 case of
  Proposition 2, checked at 40 digits). The family therefore cannot tend to a regular solution as D -> 3 (Sec. 6).

## 1. What failed before (T15, comms.txt 2026-10-20/21)

- kc 47 (solver/data/scan_lowD3_kc47.jsonl): the first step 4.0 -> 3.95 failed (|F| 1.3e-4). Re-seeded at 3.7938 and 3.7,
  it reached 3.675 (|F| 1.8e-10) and 3.6425 (2.3e-9, accepted only by the relaxed NE_STRONG_TOL 2e-8), then failed at
  3.6214 and 3.6002 (|F| 1.5e-6, 3.4e-6).
- kc 55: D = 4.0 and 3.7938 converged; at 3.7 the regridded kc 47 state landed on a false minimum (|F| 7.8e-5, the same
  point to all digits after a long LM). Continuation from 3.7938 needed steps <= 0.0125 at 1000-4800 s each, reached
  3.7689, then failed at 3.7564-3.7626 with weak residuals 8e-11 to 1.5e-10 above the 5e-11 cut.
- kc 63: only D = 4.0 was accepted (weak residual 1.6e-11), 1.0e-6 above kc 55 and above Martin-Garcia & Gundlach;
  3.7938 and 3.7 stalled at strong residuals 8e-7 and 1e-6.

Why, from Sec. 2: a seed without harmonics 49-63 is far from the kc 55/63 solution in the near-centre blocks (the
double-double solver shows it at M = 64: |F| 4e-2, all in the first block); the horizon series at u0 = 0.4 converges
ever more slowly as D falls and as the top harmonics fill in; and at kc >= 55 the top centre and horizon harmonics fall
into ne_solve's weak band, where the f64 Schur-complement step stalls at 1e-12 to 1e-10. With |dDelta/dF|_1 of 2e3 to
8e4 at kc 47 and more at higher kc, a weak residual of 1.6e-11 can be worth 1e-7 to 1e-6 in Delta: the order of the
kc 63 error at D = 4.

## 2. The obstruction, measured

### Harmonic content

`cd solver/data && python3 ../lowD_spectrum.py <state> --fit=21:35 --json=lowD_spectrum.jsonl` (planted-truth selftest:
`--selftest`), and `cd hp && python3 lowD_ecker.py <ecc> --fields` for Ecker et al.'s exported fields
(hp/data/lowD_logs/ecker_fields.log; ecc = a clone of github.com/EckerChristian/CriticalSpacetimeCrystal):

| D | source | \|F_1\| | F_3/F_1 | beta |
|---|---|---|---|---|
| 5.5 | Ecker et al., field | 3.30 | 0.333 | 0.277 (k = 9-31) |
| 4.0 | f64 kc 47 | 16.23 | 0.438 | 0.157 (k = 21-35) |
| 4.0 | Ecker et al., field | 16.21 | 0.438 | 0.161 (k = 9-31) |
| 3.875 | f64 kc 39 | 18.37 | 0.451 | 0.158 |
| 3.794 | f64 kc 47 | 20.12 | 0.459 | 0.141 |
| 3.7 | f64 kc 47 | 22.65 | 0.469 | 0.134 |
| 3.046 | Ecker et al., field | 437 | 0.558 | 0.032 (k = 9-31) |

F = sqrt(D - 2) dp/dx at x = 0, the README's normalisation (their U/x at their smallest x times sqrt(n); the D = 4
rows agree). Their D = 3.046 samples are not one exact uniform period (4% antiperiodicity error at the best period), so
that row is indicative. The kc 39 states give beta 0.01 higher than kc 47 at the same D (truncation depresses the top).

- **Amplitude:** |F_1| ~ (D - 3)^(-q), q = 0.93 between D = 4 and 3.7 and 1.09 between 3.7 and 3.046.
- **Modes:** Delta's error in the mode count falls like e^(-2 beta K): at D = 4 the steps kc 23 -> 31 -> 39 -> 47
  (README) and 47 -> 55 (T15) shrink by 12.6, 12.1, 11.2 per 8 modes, and the double-double M = 48 -> 64 -> 80 steps by
  140 per 16 (2 beta = 0.31). `cd hp && python3 lowD_cost.py` (hp/data/lowD_logs/cost.log; planted check `--selftest`)
  calibrates A e^(-2 beta K) at D = 4 and continues beta and |F_1| to 3.046 two ways: for 1e-10, K = 73 (D = 4), 81-82
  (3.8), 85-88 (3.7), 95-101 (3.6), 104-118 (3.5), 117-142 (3.4), 135-177 (3.3), 167-234 (3.2), 239-343 (3.1),
  340-450 (3.05). The double-double kernel stops at M = 128 (K = 127): 1e-10 to about D = 3.4, 1e-8 to about 3.3.

### Conditioning (f64)

`cd solver/data && python3 ../lowD_svd.py <state> --json=lowD_svd.jsonl` (exact Rust Jacobian, full SVD):

| D | kc | s_min/s_max | weakest direction | \|dDelta/dF\|_1 | weak / junk band |
|---|---|---|---|---|---|
| 4.0 | 47 | 1.9e-7 | centre p, k = 45 | 2.0e3 | 0 / 0 |
| 3.875 | 39 | 2.7e-7 | centre p, k = 37 | 5.5e3 | 0 / 0 |
| 3.794 | 47 | 4.5e-8 | centre p, k = 45 | 2.3e4 | 0 / 0 |
| 3.7 | 47 | 1.7e-8 | centre p, k = 45 | 7.8e4 | 0 / 0 |

At kc 47 nothing is near-null down to D = 3.7: the centre-data harmonics sit at 1e-5 to 1e-8, not the 1e-10 to 1e-24 of
large D. The ladder steepens as D falls (about 0.15 per harmonic at D = 4, 0.29 at 3.7), so the harmonics that 1e-10
needs at 3.7 (k about 85-90) extrapolate to about 1e-14: f64 noise. In double-double the weakest directions at D = 4,
M = 48 are the same centre harmonics at 6e-8 to 1e-6 (hp/lowD_weak.py, hp/data/lowD_weak.jsonl): no obstacle at 1e-32.
The kc 55 state in the repository (neB_kc55_rk2_D3.7689.npy) is not converged in this build (|F| 2.5e-8 at its
D = 3.7688625, all in the matching block; its file name rounds D, and at 3.7689 |F| = 1.5e-5), so its SVD row in
lowD_svd.jsonl is not used.

### Series at the matching points

f64, horizon series at u0 = 1 - x_K = 0.4 (`cd solver/data && python3 ../lowD_horizon.py <state>`, lowD_horizon.log):
the sum changes by 8.6e-11 (D = 4), 7.8e-9 (3.79), 7.6e-8 (3.7) from Ks = 60 (production) to 90; the radius in u falls
from 0.57-0.62 to 0.48-0.53. At kc 47, D = 3.7 this moves Delta by +3.7e-8 (Ks 90) and RK steps x4 by +8.8e-9
(solver/data/lowD_f64.jsonl), both far below the kc error there.

Double-double (`python3 hp/lowD_series.py <state>`, hp/data/lowD_logs/series.log):

| state | centre series, last terms | horizon series radius in u | horizon, last terms |
|---|---|---|---|
| D = 4, M = 48 | 6.7e-6 at x = 0.02 with 60 terms; 2.3e-23 at 0.01 | 0.455 | 5.7e-7 at u = 0.4 with 100; 2.8e-37 at 0.3 with 200 |
| D = 3.7, M = 64 | 4.2e-5 at x = 0.01 with 60, 2e-8 with 100; 6e-38 at 0.005 with 100 | 0.316 | 3.9e-6 at u = 0.3 with 200; 7e-41 at 0.2 with 200 |

The radius of the horizon series shrinks as D falls and as M grows (at D = 3.7, 0.47 at M = 48 from its tail, 0.32 at
M = 64), and the centre series' radius is about 0.025 at D = 4 and about 0.012 at 3.7. So the node set has to follow:
first node 0.02 -> 0.01 -> 0.005, last node 0.6 -> 0.7 -> 0.8, with 80-100 centre and 200-250 horizon terms. The
solver's tails are recorded with every point (tail_c, tail_h in hp/data/lowD_dd.jsonl); hp/lowD_dd.py --auto_nodes
extends the nodes during a continuation when they exceed 1e-30.

### Newton and cost

From an f64 kc 47 seed the double-double Newton goes straight to M = 64 only by failing (line search at |F| 4.3e-2, all in
the first block; filling the missing harmonics by each field's geometric trend, `--fill_tail`, only brings the start to
1.4e-2). Through M = 48 it converges in 5-11 Jacobians (linear at first, then quadratic); each higher M from the lower M's
solution in 4-6. Time per Newton iteration here (2 threads, shared): about 260-340 s at M = 48, 600-700 s at M = 64,
860 s at M = 80 (hp/data/lowD_dd.jsonl, seconds / its): about M^3, so about 2100 s at M = 96 and 5000 s at M = 128.

## 3. Converged points

Double-double, `python3 hp/lowD_conv.py` (hp/data/lowD_logs/conv.txt). Error = the last M difference when the
contraction between successive differences is below 0.5 (an estimator, tested on planted sequences: `--selftest`;
`--mutate` must refuse an unconverged one).

| D | Delta | M = 48 / 64 / 80 differences | other discretisation | tails (centre, horizon) | gamma |
|---|---|---|---|---|---|
| 4.0 | 3.4454524022410 | -2.6e-7, -1.8e-9 (ratio 0.007) | nodes 0.005..0.8, 100/250 terms, step tol 1e-30 at M = 64: < 1e-16 | 5e-57, 9e-31 | 0.3739607120 (M = 64) __G4_80__ |
| 3.7 | __D37__ | __D37DIFFS__ | __D37VAR__ | __D37TAILS__ | __G37__ |

- D = 4 against others: Martin-Garcia & Gundlach 3.445452402(3) (2.4e-10); f64 kc 55 3.4454524280 (+2.6e-8);
  f64 kc 47 3.4454526772 (+2.75e-7); Ecker et al. 3.445452682 (+2.8e-7, their 512 modes).
- The f64 solver at D = 3.7 (solver/data/lowD_f64.jsonl): kc 39 3.4652276243, kc 47 3.4652040679 (+3.6e-6 from the
  double-double value), Ks 90 +3.7e-8, RK x4 +8.8e-9. Its kc 47 points at 3.675 and 3.6425 are 4.7e-6 and 5.2e-6 above
  Ecker et al.'s interpolated values. Pushing it lower needs kc of 95-120 (Sec. 2), a kernel size above Ctx<96>, and
  directions near f64 noise: it is stopped at D = 3.7, without a demonstrated 1e-8 even there (kc 55 / 63 did not
  converge at 3.7 in T15).
- What the double-double and f64 values share: the field equations and gauge, the multiple-shooting idea with a
  centre and a horizon series; not the code, the tau discretisation (collocation vs dealiased Galerkin), the
  x-integration (Taylor vs RK4) or the precision.

## 4. gamma

- Double-double, `python3 hp/lowD_gamma.py <out> <state> --lam0=<guess>` (root of det M(lambda), the hp kernel's
  growth-mode matrix; hp/data/lowD_gamma.jsonl): gamma(4) = 0.3739607120 at M = 64 __G4_80__; gamma(3.7) = __G37__.
- f64 (Rust mode matrix): gamma(3.7) = 0.3486470 (kc 39), 0.3486496 (kc 47).
- **gamma(3.5) by extrapolation** (`python3 hp/lowD_gamma_extrap.py <ecc>`, hp/data/lowD_logs/gamma_extrap.log): from our
  points at 3.6425-4.0, degree-2 and -3 fits, gamma(3.5) = 0.3235 +- 0.0030 (twice the spread of the fits; an estimator
  range). The rule was first applied to Ecker et al.'s gamma curve at the same D and contained their gamma(3.5) =
  0.32239 (error 1.1e-3); `--mutate` (a curve turning by 0.01) fails it. Our points sit within 8e-7 of their curve.
- Against Bland et al. 2005 (Table I): 0.349 +- 0.003 at 3.5 is 0.026 above, about 8 combined ranges; Ecker et al.'s
  direct 0.32239 and Bland's 2007 thesis value 0.3235 +- 0.0018 *(unverified, from Ecker et al.'s data file)* agree with
  ours. So the 2005 value at 3.5 looks wrong; the README's "gamma(3.5) <~ 0.33, an open check" can be replaced.

## 5. Comparison with Ecker et al.

`python3 hp/lowD_ecker.py <ecc>` tabulates their Fortran data per Fourier resolution (128 / 256 / 512 / 1024) for D <= 4
and our values at their D. Their own resolution changes 256 -> 512 in Delta: 8.9e-10 (D = 4), 3.0e-8 (3.7), 4.9e-7 (3.5),
1.8e-5 (3.3), 2.2e-4 (3.2), 1.0e-3 (3.15), and alternate in sign (128 -> 256 is -8.5e-8 at 3.7): their Fourier error is
not the limit at D >~ 3.6, while at D = 4 their 512-mode value is 2.8e-7 above Martin-Garcia & Gundlach and ours, an
error their Fourier test cannot see. At common D:

| D | ours | Ecker et al. (512) | Ecker - ours |
|---|---|---|---|
| 4.0 | 3.4454524022 (dd) | 3.4454526819 | +2.8e-7 |
| 3.7 | __D37SHORT__ (dd) | 3.4652005045 | __D37ECK__ |
| 3.7 | gamma __G37SHORT__ (dd) | 0.3486503 | __G37ECK__ |

Their maximum (Eq. 70, D = 3.755726, Delta = 3.466772) agrees with the README's kc 47 estimate to 2e-6. Below 3.7 they
are, as far as we can check, the only values; with their 256 -> 512 changes they are good to about 5e-7 at 3.5, 2e-5 at
3.3 and 1e-3 at 3.15, and they need 1024 modes below 3.2. Sec. 6.2 and Appendices C, D of their paper could not be read
(the fetched text ends in Sec. 6).

## 6. What should happen as D -> 3

- **At n = 1 the equations decouple** (`python3 solver/lowD_n1_check.py`, solver/data/lowD_n1_check.log): in
  solver/dss.py's equations c_x = c (n - 1)(a^2 - 1)/x vanishes, so c = c(tau, 1) = 1 by the gauge, and the p, m equations
  no longer contain a: the scalar obeys the free 2+1 wave equation in self-similar form (checked on the code's
  right-hand side; the control n = 1.01 shows the coupling).
- **The free wave cannot echo at n = 1.** Proposition 2's argument goes through at kappa = (n + 1)/2 = 1: the solution
  regular at the centre is p - m = x 2F1((1+s)/2, (2+s)/2; 1; x^2) (the other one has a logarithm), and its coefficient of
  the non-analytic light-cone branch (1 - x^2)^(-1/2 - s) is C(1, s) = Gamma(s + 1/2)/(Gamma((1+s)/2) Gamma(1 + s/2)), with
  no zero for imaginary s != 0 (*proved*; the solution, both first-order equations and Gauss's connection formula
  checked at 40 digits for mu = 1, 3, 5 and omega = 1-3.82, |C| >= 0.13 there; a wrong kappa makes the check fail).
- **Consequence.** A regular finite-amplitude limit of the family at D = 3 would be a nonzero periodic regular solution
  of the free equation, which does not exist. So as D -> 3+ the family degenerates. The data show how: |F_1| grows like
  (D - 3)^(-1.09), the spectrum flattens (beta from 0.157 at D = 4 to about 0.03 at 3.046) and the series radii shrink,
  consistent with Ecker et al.'s narrow, large-amplitude peaks (their Psi reaches about 3e5 at 3.05). That the field has
  to grow until (D - 3) a^2 is of order one, gravity's only coupling being the factor n - 1, is a heuristic (*conjecture*).
  Their conjecture Delta -> 0 (fit 3.878 (D - 3)^0.15 on 3.01-3.05 in their plotting notebook) is not tested here.
- **2+1 itself** (Lambda < 0 is needed for black holes): the critical solution is continuously self-similar with a scalar
  linear in tau (c != 0), modelled by Garfinkle's n = 4 solution; gamma = 8/7 predicted, 1.23(4) measured (Jalmuzna,
  Gundlach & Chmaj, PRD 92, 124044 (2015), arXiv:1510.02592); the review of Gundlach, Hilditch & Martin-Garcia
  (arXiv:2507.07636, Sec. IV.A.1) quotes gamma = 1.20 +- 0.05 (Pretorius & Choptuik). Nothing ties that solution to the
  D -> 3+ limit of the asymptotically flat family: Delta -> 0 would suggest a CSS limit, gamma -> 0 would not match.

## 7. Proposed task for the local machine

Double-double below D = 3.7: Delta and gamma with M = 96 / 128 and a node-set variant, down to D = 3.3. Needs this
branch (hp/lowD_dd.py with --Dlist / --auto_nodes, the states hp/data/lowD_states/lowDv_D3.7000_M{64,80}.npz); hpkernel as
built for T17 (hp/src unchanged since e0c77e5). From the repo root, every process `--threads 1` at below-normal
priority, at most 6 at a time, output to the logs named.

- **A (pilot and chain, 1 process).** M = 96 at 3.7, then continuation in steps of 0.05:

      python -u hp/lowD_dd.py hp/data/lowD_dd_local.jsonl --start hp/data/lowD_states/lowDv_D3.7000_M80.npz --nodes 0.005,0.01,0.02,0.05,0.1,0.16,0.23,0.3,0.4,0.5,0.6,0.7,0.8 --Ms 96 --kcs 100 --ks 250 --tol_step 1e-30 --it_max 30 --threads 1 --tag lowDv --resume --auto_nodes --Dlist 3.65,3.6,3.55,3.5,3.45,3.4,3.35,3.3 > hp/data/lowD_logs/local_A.log 2>&1

  Pilot rule: after 3.65, report the time per Newton iteration (expected about 15 min, a point in 4-6) and stop if the
  projected total exceeds 24 h. It stops by itself when a step below 0.005 fails; it restarts with the same command.
- **B (as A's points appear, up to 4 processes).** For each new state S = hp/data/lowD_states/<tag>_D<D>_M96.npz at
  D = 3.6, 3.5, 3.4, 3.3 (use the nodes and tag that S was solved with; auto_nodes changes them, the tag says how):

      python -u hp/lowD_dd.py hp/data/lowD_dd_local.jsonl --start S --nodes <S's nodes> --Ms 128 --kcs 100 --ks 250 --tol_step 1e-30 --it_max 30 --threads 1 --tag <S's tag> --resume > hp/data/lowD_logs/local_B_<D>.log 2>&1
      python -u hp/lowD_dd.py hp/data/lowD_dd_local.jsonl --start S --nodes <S's nodes> --Ms 80 --kcs 100 --ks 250 --tol_step 1e-30 --it_max 30 --threads 1 --tag <S's tag> --resume >> hp/data/lowD_logs/local_B_<D>.log 2>&1
      python -u hp/lowD_gamma.py hp/data/lowD_gamma_local.jsonl S --threads=1 > hp/data/lowD_logs/local_G_<D>.log 2>&1

  M = 128 is about 35 min per iteration locally (3-4 h per D); gamma about 8 determinants at M = 96 (2 h).
- **C (optional, 1 process):** f64 kc 55 / 63 at D = 3.7 from the double-double state, to see whether the f64 solver
  converges in kc there: `python -u solver/lowD_f64.py solver/data/lowD_f64_local.jsonl --dd hp/data/lowD_states/lowDv_D3.7000_M80.npz --kcs 55,63 --rks 2 --Ks 90 --threads 1 --gamma > solver/data/lowD_f64_local.log 2>&1`
  (needs rustkernel at cf79fba or later).
- **Stop conditions:** a record with "converged": false that a rerun does not fix; M = 128 differing from M = 96 by more
  than 1e-8 at some D (that D is then not resolved: finish it, then stop the chain); --auto_nodes asking for a first node
  below 0.0025; 24 h in all.
- **Push:** hp/data/lowD_dd_local.jsonl, hp/data/lowD_gamma_local.jsonl, the *_M96 and *_M128 states (not *_ckpt),
  hp/data/lowD_logs/local_*.log, solver/data/lowD_f64_local.*, and in the comms note the output of
  `python hp/lowD_conv.py hp/data/lowD_dd.jsonl hp/data/lowD_dd_local.jsonl`.

## 8. Proposed README text

For "Relation to published work", replacing the "Below D = 4" bullet:

> - **Below D = 4.** The double-double solver, with its first node moved to x = 0.005 and its last to 0.8 (the series
>   there converge ever more slowly as D falls), gives Δ(3.7) = __D37SHORT__ and γ(3.7) = __G37SHORT__, converged in the
>   τ resolution (48, 64 and 80 points) and in the node layout (notes/LOW_D.md); Ecker et al. 2026 give 3.4652005
>   and 0.3486503 there. The f64 points below D = 4 (47 Fourier modes) are 4-5×10⁻⁶ high in Δ. Extrapolated from
>   3.64 ≤ D ≤ 4, γ(3.5) = 0.3235 ± 0.0030, in agreement with Ecker et al.'s 0.32239 and not with Bland et al.'s
>   0.349 ± 0.003. The modes needed grow from about 73 at D = 4 to about 110 at 3.5 and several hundred near 3.05, and
>   at D = 3 the scalar decouples from gravity in these equations, so the family cannot reach a regular D = 3 limit.

and in "The formula", the sentence "Below D = 4 the formula is not fitted; Δ(D) turns over there." can point to
notes/LOW_D.md for the converged values below 4.

## 9. Literature read for this note

- Ecker, Ecker, Grumiller & Jechtl, SciPost Phys. 20, 176 (2026), arXiv:2602.10185: Secs. 1-5 read (arXiv PDF text);
  Sec. 6.2 and Appendices C, D not read (text truncated); their data repository used instead. Range 3.05 <= D <= 5.5;
  "numerically too costly to go below D = 3.05"; narrow, large-amplitude peaks of Psi near the centre; N_x = 1.55e5 at
  D = 3.05, 8000 at D >= 4; N_tau = 512 enough for D >~ 3.2, at least 1024 below; x_L = 5e-5 for D < 3.1; results
  "converge to machine precision for any number of Fourier modes" but differ from the continuum limit; about 1e-6
  claimed at D = 4 and at the maximum; conjecture Delta -> 0 and gamma -> 0 as D -> 3+.
- Jalmuzna, Gundlach & Chmaj, PRD 92, 124044 (2015), arXiv:1510.02592 (abstract and text read; DOI from the arXiv page).
- Gundlach, Hilditch & Martin-Garcia (2025), arXiv:2507.07636, Sec. IV.A.1-2 (read).
- Bland, Preston, Becker, Kunstatter & Husain, gr-qc/0507088, Table I (read): gamma(3.5) = 0.349 +- 0.003 from fits of
  the apparent-horizon radius in evolutions, the error the a* range over which the fit stays good, "about 1%". Bland's
  2007 thesis values are from Ecker et al.'s data file plots/gBlandThesis.dat *(unverified)*.

## Files

Scripts: hp/lowD_dd.py, hp/lowD_conv.py, hp/lowD_gamma.py, hp/lowD_gamma_extrap.py, hp/lowD_cost.py, hp/lowD_ecker.py,
hp/lowD_series.py, hp/lowD_weak.py, solver/lowD_spectrum.py, solver/lowD_svd.py, solver/lowD_horizon.py, solver/lowD_f64.py,
solver/lowD_n1_check.py. Data: hp/data/lowD_dd.jsonl, hp/data/lowD_gamma.jsonl, hp/data/lowD_weak.jsonl,
hp/data/lowD_states/, hp/data/lowD_logs/, solver/data/lowD_*.jsonl, solver/data/lowD_*.log, solver/data/lowD_f64_*.npy.
The f64 kc 55 / 63 runs used a rustkernel built from this tree (cf79fba) into a private directory, bitwise equal to the
installed one at kc 47.

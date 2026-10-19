# Working context (handoff)

State of the project as of 2026-10-18, 23:40 PT, written so that the work can be picked up from the repo alone
(the cloud workspace it ran in is ephemeral). `notes.md` is the full lab notebook in chronological order, with
every correction; this file is the summary.

## Goal

A formula for Choptuik's echo period Δ(D) and critical exponent γ(D) of massless scalar critical collapse in
D spacetime dimensions (n = D − 2), in particular the D → ∞ behaviour.

## Method

- Discretely self-similar (DSS) critical solution: fields p, m, a, c on a Fourier grid in τ (kc = 23, N = 72,
  folded by half-period parity), multiple shooting in x = r/(−t) with nodes
  [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6] (+0.7 for x_K = 0.7). Centre Taylor series (Kc = 4, x0 = 1e-4);
  regular horizon (SSH) Taylor series with Ks = 60 terms evaluated at u = 1 − x_K; RK4 with frozen step counts.
  941 unknowns (1035 at x_K = 0.7), Δ last.
- Rust kernel (repo root): residual (SIMD, threads), double-double residual, exact Jacobian and growth-mode matrix
  M(λ) by dual numbers, `match_node` option. γ = 1/λ where det M(λ) = 0.
- Solvers (solver/): LM (`fast.lm_svd3`, `dd_solver.dd_solve`), nonlinear elimination (`ne_solver.ne_solve`),
  double-double NE (`ne_dd.ne_dd`), smoothness fix of the junk directions (`smooth_junk.py`).

## What is established

- D = 4: Δ = 3.4454524, γ = 0.3739609; D = 5: Δ = 3.2217019, γ = 0.4131269. Ecker, Ecker, Grumiller, Jechtl
  (arXiv 2602.10185, Feb 2026) give 3.445453 / 0.373961 and 3.22176 / 0.41322, over 3.05 < D < 5.5 only.
- Δ(D) for D ≤ 17.5 is determined to ~1e-12: every solver and every starting point agree to 1e-13.
- Δ decreases monotonically and levels off. The local exponent −d ln Δ / d ln n peaks at 0.26 near D ≈ 7 and falls
  (0.145 at D ≈ 31). The data favour a finite limit Δ∞ ≈ 1.635–1.653 (fit families on D ≥ 8–16,
  `fit_ne.py`), against the expectation Δ → 0 stated in Ecker, Ecker, Grumiller (arXiv 2601.14358, Jan 2026).
  Candidates inside the window: 2π/j₁,₁ = 1.63979, π²/6 = 1.64493, √e = 1.64872. Not discriminated yet.
- γ(D) rises towards 1/2 (0.4852 at D = 39.3 on the x_K = 0.7 scan). 1/2 − γ falls like n^(−p), p ≈ 0.6–0.7.
- Large-D structure (notes.md, "Zero-crossing condition", "Attempt to derive the first gamma coefficient"): the
  regular 1/D expansion leaves Δ, the profile B(τ) and the growth rate free at every order. They are selected only
  by the exponentially small singular horizon branch u^σ, Re σ = s0 = n/(2λ_g) − 1 ≈ 0.53 n. So Δ∞ is a
  beyond-all-orders selection problem, like Saffman–Taylor finger selection; an analytic Δ∞ would come from
  exponential asymptotics of the leading-order family, not from the 1/D series.

## Current best formula (2026-10-19, 04:30)

Step-converged data D = 8–30.3 (x_K 0.6 RK x2 + Richardson, junk-solved above 17.2; x_K 0.7 RK x4 above 30.2),
`final_fit.py`, weighted fits in x/max(x) over x = 1/D, 1/(D−1), 1/(D−2), degrees 8–11:

  **Δ(D) = Δ∞ + c₁/D + c₂/D² + …, Δ∞ = 1.64394 ± 0.00005**, c₁ ≈ 10.3
  **γ(D) = 1/2 − 0.7577/D + …** (free fit: γ∞ = 0.49995–0.50016)

π²/6 = 1.64493 is excluded (forcing it: χ²/dof 140–1300 against ~4); 2π/j₁,₁ and √e are far outside.
ω∞ = 2π/Δ∞ = 3.82205.

Hard-won lessons for these fits:
- fit in x/max(x): an unscaled degree-10 Vandermonde loses ~1e-9 and made the limit wander (`series_factory.py`,
  `refit_rk.py`, `constr_test.py` are fixed);
- RK x1 data carry a ±2e-8 sawtooth in D (the ceil() in freeze_steps), RK x2 is ~16x smaller: use NE_RK = 2 (and
  Richardson RK2 + (RK2 − RK1)/15);
- above D ≈ 17.5 the junk directions must be solved (`junk_solve.py`, NE_JUNK) or fixed by smoothness (NE_SMOOTH).

High D: x_K 0.6 is limited by the horizon-series radius (≈ 0.47 at D = 30, 0.43 at 36); x_K 0.7 converges there but
needs NE_RK ≥ 3–4 (D = 29.3: RK x1 −7.9e-6, x2 −3.7e-7, x3 ~−1e-7 from the limit). The local runner's T2 is the
x_K 0.7, NE_RK = 4, junk-solved scan to D = 45.

Coordination: `comms.txt` at the repo root is the message log between the cloud session and the local runner
(append only, pull before writing, commit only that file with the message).

**Shape of the solution at large D** (a lead for the exact value): the centre field's first harmonic falls like
~n^(−1/2) (h1·√n = 2.64 at D = 7.75, 1.59 at 32.5, still drifting), the profile becomes sinusoidal (h3/h1 = 0.25 →
0.10) and the metric approaches flat (1 − min a ≈ 0.58 → 0.18). So the D → ∞ critical solution looks weak-amplitude
and single-harmonic, and Δ∞ may be the eigenvalue of a weakly nonlinear problem about flat space.

## The numerical difficulty above D ≈ 17.5

All of it is one mechanism: horizon regularity of each τ-harmonic is felt in the residual only through
u_K^{s0} times a coupling that shrinks with the harmonic. The Jacobian therefore has a ladder of near-null
directions (SVD at D = 28, x_K = 0.6, s/s_max):

| directions | s/s_max | what they are |
|---|---|---|
| Δ | ~1e-8 | the period |
| "shape" pair k = 3 | ~4e-11 | centre-data harmonic k = 3 |
| pair k = 5 | ~2e-13 | centre-data harmonic k = 5 |
| 16 "junk" | < 3e-14 | centre-data harmonics k ≈ 7–23 |

1. **LM floor (fixed).** LM stalls with the residual in the k = 3 shape pair (Gauss-Newton step only ~4e-4 but far
   outside the linear range of the stiff part). The old scans above D ≈ 17.5 were biased by 1e-6 (D = 18) to 1.2e-5
   (D = 23) and drifted further above. `ne_solve` (nonlinear elimination: Newton on the strong equations for each
   value of the weak coordinates, Schur complement for the weak ones) removes it: weak residual 1.5e-8 → 2e-14.
2. **Junk ambiguity (open).** `ne_solve` freezes the junk directions. Their coordinates are not fixed by F to
   better than ~1e-12, they sit at garbage values ~1e-2 (the physical centre spectrum decays: 0.4, 0.04, 8e-3,
   1e-3, ...), and they move Δ: two converged solutions at D = 23 differ by 0.1 in junk coordinates and by 7.5e-6
   in Δ (`junk_delta.py`). Consequences, measured:
   - x_K = 0.7 minus x_K = 0.6 (both NE continuation, `cmp_setups.py`): +8e-6 (D = 23), +1.7e-5 (25), −7e-5 (28),
     −1.8e-4 (29) and growing. At D = 23: kc = 31 gives 2.0546897, x_K = 0.6 2.0546914, x_K = 0.7 2.0546921 or
     2.0546996 depending on the start.
   - `smooth_junk.py` (junk coordinates chosen to minimise centre harmonics k ≥ 9) makes two x_K = 0.7 starts agree
     to 6e-8 (2.0546955), but x_K = 0.6 then gives 2.0546914 (4e-6 apart).
   - `ne_dd.py` (double-double unknowns, DD differences for the 22 near-null directions) drives the strong residual
     to 1e-28 but its outer Newton step does not reduce the junk residual.
   - Matching closer to the centre (x_K = 0.4, Ks = 150; `scratch/xk04_dbg.py`) does NOT lift the junk band (only
     the weak band, slightly), and the horizon series needs Ks ≈ 260 at u = 0.6. Not the fix.
   - What the junk is (`scratch/junk_profile.py`): evanescent high-k perturbations. Their harmonic-k content peaks
     near x ≈ 0.05–0.1 and decays outward, faster for higher k (k = 23 falls 1e6 between x = 0.1 and 0.6). They carry a
     small Δ component: the tail at the horizon can stand in for a change of Δ. In the continuum too, the high-k
     centre content is fixed only by exponentially weak horizon conditions; since the true solution is analytic in τ
     that content is small, which justifies the smoothness prior (`NE_SMOOTH=1`). With it, different starts converge
     to the same Δ (2.0546955863 at D = 23.04, x_K = 0.7); x_K = 0.6 gives 2.0546914107 (4.2e-6 lower), which is the
     current error bar there.
   - The near-null sector is the same for every growth rate λ in [−3, 0.3] (`scratch/mode_spectrum.py`): it is not a
     set of weakly damped modes, it is the high-k sector itself. Its origin is open; the junk equations' residual
     (~1e-13) is at the level of the discretisation errors, so their "solution" is noise that moves Δ.

So: Δ is good to ~1e-12 up to D ≈ 17.5, ~1e-5 up to D ≈ 25, and only ~1e-3 above D ≈ 29 until the junk
directions are resolved. The continuation scans also run off: x_K = 0.7 steepens from slope −0.0117 (D = 37) to
−0.033 (D = 40), and x_K = 0.6 blew up at D = 35.15.

## Data (solver/data)

| file | contents |
|---|---|
| `scan_ne6lo.jsonl` | NE re-solves D = 7.75–20.4 (x_K = 0.6, kc = 23) |
| `scan_ne6.jsonl` | NE re-solves D = 19.2–23.7 (some later seeds failed: ok = false) |
| `scan_ne6b.jsonl` | NE continuation x_K = 0.6, kc = 23, D = 22.4 → ~33 (cloud, running) |
| `scan_ne7.jsonl` | NE continuation x_K = 0.7, kc = 23, D = 23 → ~39 (cloud, running) |
| `scan_ne7_local.jsonl`, `scan_ne6_kc31.jsonl` | first points of the local runs (x_K = 0.7 agrees with the cloud to 3e-8) |
| `gamma_rust.jsonl` | γ from the Rust mode matrix on the old solutions, D ≤ 24.9 (old Δ bias applies above 17.5) |
| older `scan_*.jsonl` | LM-era scans (biased above D ≈ 17.5) |

Every record has D, Δ, γ, the residual split (strong / weak / junk) and `ok`. `data_ne.load()` merges the old data
below D = 17.2 with the NE records. Solution vectors: `seeds/` (long_D, longP_D, longB_D, long7_D = old;
neB_D, ne7_D = NE) and `data/ne*_D*.npy`.

## How to run

```
pip install .                                    # repo root; rebuild after every pull that touches src/ or python/
cd solver/data
python ../scan_ne.py out.jsonl D_from D_end h hmax [xK] [seed prefixes] [kc]   # NE_THREADS, NE_SEED_MAX env
python ../fit_ne.py                              # Delta_inf fits on data_ne.load()
python ../cmp_setups.py [kc]                     # setup-vs-setup differences (the error bar)
python ../plot_delta.py -o ../delta.svg          # the graph
python ../run_local.py                           # parallel scans on a local machine
```

Set `OPENBLAS_NUM_THREADS=1` (a contended multithreaded 941² SVD takes 3–7 s instead of 0.4 s). `scratch/` holds
every other script from the cloud workspace (one-off diagnostics, older variants). They expect to run from the
directory that holds the seeds.

## Next steps

1. Resolve the junk directions. Find out why the high-k sector is near-null (compare the centre-regular and
   horizon-regular high-k solutions at the match; vary kc and the dealiasing). Meanwhile the smoothness prior
   (`smooth_junk.py`, `NE_SMOOTH=1`) is the regularisation in use, and the local job matrix measures its effect.
2. Error bars from at least three discretisations per D (x_K 0.6 / 0.7 / 0.4, kc 23 / 31). Refit Δ∞ with them.
   Discriminate 2π/j₁,₁, π²/6, √e.
3. γ: redo with the final solutions. Fit 1/2 − γ.
4. Theory: exponential asymptotics of the leading-order large-D family (Π = β/√(1+β²x²)) for the selection of Δ∞.

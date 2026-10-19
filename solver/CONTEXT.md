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
   - Being tested: matching closer to the centre (x_K = 0.4, Ks = 150; `xk04_test.py`), which strengthens every
     horizon-regularity condition by (u_K/0.4)^{s0} ≈ 150× at D = 23. The horizon series radius is ≈ 0.74 in u
     (D ≈ 19), so u = 0.6 needs Ks ≈ 150–220.

So: Δ is good to ~1e-12 up to D ≈ 17.5, ~1e-5 up to D ≈ 25, and only ~1e-4 around D ≈ 29–40 until the junk
directions are resolved.

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

1. Resolve the junk directions: the x_K = 0.4 / Ks = 150 test; if it lifts the junk band, rescan D = 17–40 with it.
   Otherwise, a smoothness criterion justified by the continuum problem, or a formulation where centre data for
   high harmonics are not free unknowns.
2. Error bars from at least three discretisations per D (x_K 0.6 / 0.7 / 0.4, kc 23 / 31). Refit Δ∞ with them.
   Discriminate 2π/j₁,₁, π²/6, √e.
3. γ: redo with the final solutions. Fit 1/2 − γ.
4. Theory: exponential asymptotics of the leading-order large-D family (Π = β/√(1+β²x²)) for the selection of Δ∞.

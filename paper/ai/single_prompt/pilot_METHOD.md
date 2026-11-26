# Echoing period Δ of the Choptuik critical solution in D = 4, 6, 7, 8

| D | Δ (six decimals) | best estimate (uncertainty in last digit) |
|---|---|---|
| 4 | 3.445452 | 3.4454524022(2) |
| 6 | 3.009483 | 3.009483220(1) |
| 7 | 2.842753 | 2.842753206(2) |
| 8 | 2.711790 | 2.711790054(5) |

Nothing was looked up or quoted; every number comes from the code in `src/`. The uncertainties
are my reading of the scatter in the convergence tables of section 4, not rigorous bounds.
The distance of each value from the nearest rounding boundary of the sixth decimal is 1e-7 (D = 4),
2.8e-7 (D = 6), 2.9e-7 (D = 7) and 4.5e-7 (D = 8), while the scatter among the well-resolved rows of the
tables is ≤ 1e-10 (D = 4, 6), ≤ 2e-9 (D = 7) and ≤ 6e-9 (D = 8).

## 1. Equations

D = n + 2 dimensions, spherical symmetry, outgoing null (Bondi) coordinates

    ds² = −g ḡ du² − 2 g du dr + r² dΩ_n ,     R_ab = (4/n) ∂_aφ ∂_bφ

(the massless scalar is rescaled by a constant so that the coupling is 4/n; this does not
affect Δ). `src/derive_bondi.py` (sympy) computes the Ricci tensor explicitly for D = 4, 5, 6,
checks it against the warped-product formulae used for general n, and verifies everything below
(all printed differences are exactly 0):

    rr:        (ln g)_r = (4/n²) r φ_r²
    angular:   (r^{n−1} ḡ)_r = (n−1) r^{n−2} g
    □φ = 0
    uu (redundant):  n ḡ_u/(2r) − n ḡ g_u/(2 r g) = (4/n) φ_u (φ_u − ḡ φ_r)

A discretely self-similar (DSS) solution is periodic in τ at fixed x in the coordinates

    τ = −ln(−U),   x = r/(−U),   g = 2G(τ,x),  ḡ = 2Ḡ(τ,x),  φ = φ(τ,x),

where U < 0 is a null coordinate that vanishes at the accumulation point. The remaining freedom
U → U'(U) is fixed by Ḡ(τ, 1) = 1, which makes x = 1 an ingoing null cone: the past light cone
of the accumulation point (self-similarity horizon, SSH). The equations become

    (W)  x (x−Ḡ) φ_xx + x φ_xτ + (n/2) φ_τ + [(n/2+1) x − Ḡ − (n−1) G] φ_x = 0
    (G)  G_x = (4/n²) x φ_x² G
    (B)  x Ḡ_x = (n−1)(G − Ḡ),        Ḡ(τ,1) = 1
    (C)  G Ḡ˙ − Ḡ G˙ = (4/n²) x G φ˙ (φ˙ − 2 Ḡ φ_x),    f˙ ≡ f_τ + x f_x    (redundant)

(W) is singular at the centre x = 0 and at the SSH x = 1. Demanding a solution that is
periodic in τ with period Δ and analytic at both singular points is a nonlinear eigenvalue
problem for Δ. The critical solution has the extra symmetry φ(τ+Δ/2, x) = −φ(τ, x), with G, Ḡ
periodic with period Δ/2, which is built into the discretisation. Δ is the period in τ; it is
the same number in terms of the logarithm of central proper time, because the two time
coordinates differ by a periodic function.

n enters only as a parameter, so non-integer n can be used to continue between dimensions.

## 2. Solver (`src/bondi.py`, `src/bondi_ld.py`)

* Unknowns: φ, G, Ḡ on a tensor grid and ω = 2π/Δ. θ = ωτ: M equispaced Fourier collocation
  points on the half period θ ∈ [0, π) (antiperiodic differentiation matrix for φ, periodic for
  G, Ḡ). x: Nx+1 Chebyshev–Gauss–Lobatto points on [0, 1], optionally through the analytic map
  x = x0 + ε tan(a0 + (a1−a0) s), s ∈ [0,1], which concentrates points around x0.
* (W), (G), (B) are collocated at all points including x = 0 and x = 1; there the degenerate
  equations themselves are the regularity conditions. The (G) row at x = 1 is replaced by the
  gauge condition Ḡ = 1, and the τ-translation freedom is removed by Σ_j φ(θ_j, 0) sin θ_j = 0,
  which is the equation paired with the unknown ω.
* Newton iteration with the dense analytic Jacobian (checked against finite differences) and LU.
* **Extended precision.** The Jacobian has a few very small singular values (σ_min ≈ 1e-6 for
  D = 6 and 1e-8…1e-9 for D = 8, decreasing with Nx; `src/svdprobe.py`). The corresponding
  singular vectors are smooth perturbations of φ whose residual lives in the last few points
  next to the SSH. My interpretation: Δ is fixed by analyticity at the SSH, and the non-analytic
  solutions of the linearised wave equation behave like (1−x)^s there, with s ≈ 1.7, 3.1, 3.9,
  4.6 for D = 4, 6, 7, 8 (indicial exponent with τ-averaged coefficients, a rough estimate), so in
  higher D they are nearly polynomial and only weakly excluded. In double precision the roundoff
  of the residual (~1e-11) was amplified into a run-to-run scatter of Δ of about 2e-8 (D = 6),
  4e-7 (D = 7) and 2e-6 (D = 8) – not good enough for six decimals in D = 8. All final numbers
  therefore come from `bondi_ld.py`: the residual is evaluated in 80-bit long double (ε = 1.1e-19,
  differentiation matrices built in long double as well) and the Newton corrections use the
  double precision Jacobian (iterative refinement) until the corrections stagnate. The scatter of
  Δ drops to ≲ 5e-9 in D = 8 and ≲ 1e-10 in D = 4.

## 3. How the solution was found

1. **Time evolution, D = 4** (`src/nullevo.py`, `src/crit3.py`). A characteristic code (grid points
   ride ingoing null rays, variables h = φ + (2/n) r φ_r, g, ḡ, second-order Runge–Kutta, a grid
   that shrinks self-similarly towards the estimated accumulation time) evolves the family
   h(0, r) = A r² exp(−((r−0.3)/0.1)²). A was bisected between dispersal and collapse to 1 part
   in 1e15.
2. **Initial guess** (`src/extract_bondi.py`). In the near-critical evolution the SSH was traced
   backwards, the fields were mapped to (τ, x) and one period was sampled on the collocation grid.
   Newton converged from this guess (M = 32, Nx = 40, 4 iterations, quadratic).
3. **D = 4 refinement** (`src/refineB.py`): without a map the Chebyshev coefficients decay only
   like 0.85^k in D = 4 (steep transition at x ≈ 0.2); with the map (x0, ε) = (0.2, 0.15) the
   decay is ≈ 0.72^k and Nx ≈ 56 suffices (`src/probe_map.py` was used to choose the map).
4. **Continuation in n** (`src/cont.py`): n = 2 → 6 in steps of 0.25 at M = 32, Nx = 48 with secant
   predictor; Newton converged at every step. Δ decreases monotonically (by-product, not
   convergence-tested: Δ ≈ 3.2217 at D = 5). For D ≥ 6 the solution is smoother in x and no map
   is needed.
5. **Polishing and convergence tables** (`src/tableL.py`), section 4.

## 4. Convergence tests

For each D the long-double solver was run for a range of M (Fourier) and Nx (Chebyshev) and
for two different x-maps. "Tail" is the largest magnitude among the last 3–4 retained Fourier
(resp. Chebyshev) coefficients, maximised over the other direction. The last column is the
maximum over the grid of the residual of equation (C), which is not used in the solve.
Raw log: `src/tablesL.log`.

Reading the tables:

* Δ converges geometrically in both M and Nx. At fixed Nx the values for M = 32, 48, 64 agree to
  ≤ 1e-10 in D = 6, 7 and to ≤ 4e-9 in D = 8 (roundoff scatter); in D = 4, M = 48, 64, 80 agree to
  1e-11 and M = 32 is 3–6e-9 away. At fixed M the changes Nx = 56 → 64 → 72 are ≤ 5e-9 in D = 6, 7, 8
  and 6e-11 (56 → 64) in D = 4.
* Two different maps (i.e. two different sets of collocation points) agree to 1e-10 (D = 4, 6),
  1e-9 (D = 7) and 6e-9 (D = 8). The unmapped D = 4 grid is still converging at Nx = 96 and is
  2e-8 away there.
* The residual of the redundant field equation decreases together with the truncation error, to
  1e-7 (D = 4, M = Nx = 64), 2e-11 (D = 6), 1e-12 (D = 7) and 3e-14 (D = 8) at the best resolutions.
  In D = 4 it is the least converged diagnostic (it is limited by M: 4e-4, 5e-6, 1e-7 for
  M = 32, 48, 64).
* Under-resolved rows are kept on purpose (e.g. Nx = 40 or M = 24) to show the rate.

### D = 4  (n = 2):  Delta = 3.4454524022(2)  ->  **3.445452**

| x-map (x0, eps) | M | Nx | Delta | Fourier tail phi / G | Chebyshev tail phi / G | max residual of redundant eq. (C) |
|---|---|---|---|---|---|---|
| (0.2, 0.15) | 24 | 48 | 3.44545307705 | 1.9e-05 / 6.4e-05 | 1.7e-09 / 3.1e-08 | 3.8e-03 |
| (0.2, 0.15) | 32 | 32 | 3.44545350458 | 1.2e-06 / 6.0e-06 | 5.8e-07 / 6.7e-06 | 1.2e-03 |
| (0.2, 0.15) | 32 | 40 | 3.44545248130 | 1.1e-06 / 5.3e-06 | 2.7e-08 / 4.7e-07 | 4.0e-04 |
| (0.2, 0.15) | 32 | 48 | 3.44545241855 | 1.1e-06 / 5.7e-06 | 1.7e-09 / 2.8e-08 | 4.5e-04 |
| (0.2, 0.15) | 32 | 56 | 3.44545240531 | 1.1e-06 / 5.3e-06 | 1.1e-10 / 2.2e-09 | 4.2e-04 |
| (0.2, 0.15) | 32 | 64 | 3.44545240793 | 1.1e-06 / 5.7e-06 | 5.1e-12 / 1.2e-10 | 4.3e-04 |
| (0.2, 0.15) | 48 | 48 | 3.44545240787 | 5.6e-09 / 3.9e-08 | 1.7e-09 / 3.2e-08 | 1.3e-05 |
| (0.2, 0.15) | 48 | 56 | 3.44545240218 | 5.5e-09 / 4.2e-08 | 9.7e-11 / 2.2e-09 | 5.0e-06 |
| (0.2, 0.15) | 48 | 64 | 3.44545240223 | 5.5e-09 / 4.3e-08 | 6.7e-12 / 1.4e-10 | 5.4e-06 |
| (0.2, 0.15) | 64 | 48 | 3.44545240786 | 2.0e-11 / 2.0e-10 | 1.7e-09 / 3.0e-08 | 1.3e-05 |
| (0.2, 0.15) | 64 | 56 | 3.44545240217 | 3.5e-11 / 3.2e-10 | 9.6e-11 / 2.2e-09 | 1.2e-06 |
| (0.2, 0.15) | 64 | 64 | 3.44545240223 | 3.4e-11 / 3.4e-10 | 6.0e-12 / 1.6e-10 | 1.1e-07 |
| (0.2, 0.15) | 80 | 56 | 3.44545240217 | 8.7e-14 / 2.0e-12 | 9.5e-11 / 2.2e-09 | 1.2e-06 |
| (0.25, 0.2) | 48 | 56 | 3.44545240156 | 5.5e-09 / 4.3e-08 | 1.2e-09 / 1.4e-08 | 1.4e-05 |
| (0.25, 0.2) | 48 | 64 | 3.44545240226 | 5.5e-09 / 4.2e-08 | 8.6e-11 / 9.4e-10 | 5.2e-06 |
| (0.25, 0.2) | 64 | 64 | 3.44545240224 | 2.8e-11 / 3.1e-10 | 1.7e-10 / 2.0e-09 | 2.8e-06 |
| none | 48 | 80 | 3.44545360212 | 3.0e-09 / 2.4e-08 | 5.6e-09 / 2.1e-07 | 3.3e-03 |
| none | 48 | 96 | 3.44545238235 | 5.0e-09 / 3.5e-08 | 2.7e-10 / 1.2e-08 | 2.6e-04 |

### D = 6  (n = 4):  Delta = 3.009483220(1)  ->  **3.009483**

| x-map (x0, eps) | M | Nx | Delta | Fourier tail phi / G | Chebyshev tail phi / G | max residual of redundant eq. (C) |
|---|---|---|---|---|---|---|
| none | 24 | 48 | 3.00948344902 | 1.4e-07 / 2.3e-07 | 5.9e-10 / 5.2e-09 | 1.9e-05 |
| none | 32 | 40 | 3.00948847267 | 2.1e-09 / 1.9e-09 | 1.8e-08 / 1.1e-07 | 3.0e-04 |
| none | 32 | 48 | 3.00948332770 | 2.1e-09 / 1.9e-09 | 8.7e-10 / 6.7e-09 | 2.8e-05 |
| none | 32 | 56 | 3.00948321559 | 1.0e-09 / 2.2e-09 | 4.0e-11 / 3.0e-10 | 1.6e-06 |
| none | 32 | 64 | 3.00948322057 | 1.0e-09 / 2.2e-09 | 1.1e-12 / 1.9e-11 | 1.4e-07 |
| none | 32 | 72 | 3.00948322050 | 9.9e-10 / 2.2e-09 | 5.5e-14 / 1.2e-12 | 1.2e-07 |
| none | 48 | 48 | 3.00948332771 | 3.5e-14 / 2.4e-13 | 9.2e-10 / 5.6e-09 | 2.1e-05 |
| none | 48 | 56 | 3.00948321560 | 1.1e-13 / 2.0e-13 | 4.8e-11 / 4.0e-10 | 2.2e-06 |
| none | 48 | 64 | 3.00948322052 | 1.3e-13 / 1.6e-13 | 2.4e-12 / 2.5e-11 | 1.8e-07 |
| none | 64 | 56 | 3.00948321559 | 1.2e-16 / 1.1e-16 | 4.6e-11 / 3.9e-10 | 2.2e-06 |
| (0.4, 0.4) | 32 | 56 | 3.00948322049 | 1.0e-09 / 2.3e-09 | 3.6e-15 / 5.4e-14 | 1.3e-07 |
| (0.4, 0.4) | 32 | 64 | 3.00948322049 | 1.0e-09 / 2.2e-09 | 8.5e-16 / 9.0e-16 | 1.1e-07 |
| (0.4, 0.4) | 48 | 64 | 3.00948322050 | 6.9e-14 / 2.4e-13 | 7.3e-16 / 1.6e-15 | 2.2e-11 |

### D = 7  (n = 5):  Delta = 2.842753206(2)  ->  **2.842753**

| x-map (x0, eps) | M | Nx | Delta | Fourier tail phi / G | Chebyshev tail phi / G | max residual of redundant eq. (C) |
|---|---|---|---|---|---|---|
| none | 24 | 48 | 2.84275324135 | 3.2e-08 / 4.0e-08 | 4.4e-11 / 3.8e-10 | 1.5e-06 |
| none | 32 | 40 | 2.84274893458 | 6.2e-10 / 1.7e-10 | 2.5e-09 / 1.0e-08 | 2.7e-05 |
| none | 32 | 48 | 2.84275324152 | 5.7e-10 / 2.0e-10 | 8.3e-11 / 4.0e-10 | 1.5e-06 |
| none | 32 | 56 | 2.84275320598 | 1.2e-10 / 2.1e-10 | 2.0e-12 / 1.5e-11 | 8.0e-08 |
| none | 32 | 64 | 2.84275320689 | 1.2e-10 / 2.2e-10 | 5.1e-14 / 5.7e-13 | 1.1e-08 |
| none | 32 | 72 | 2.84275320828 | 1.2e-10 / 2.3e-10 | 2.0e-15 / 1.8e-14 | 1.1e-08 |
| none | 48 | 48 | 2.84275324155 | 5.4e-15 / 5.7e-15 | 6.9e-11 / 4.0e-10 | 1.5e-06 |
| none | 48 | 56 | 2.84275320597 | 7.4e-15 / 5.0e-15 | 2.7e-12 / 1.5e-11 | 8.0e-08 |
| none | 48 | 64 | 2.84275320688 | 1.9e-14 / 5.1e-15 | 8.8e-14 / 5.7e-13 | 3.9e-09 |
| none | 64 | 56 | 2.84275320598 | 1.2e-16 / 1.1e-16 | 2.4e-12 / 1.5e-11 | 8.0e-08 |
| (0.5, 0.5) | 32 | 56 | 2.84275320644 | 1.2e-10 / 2.3e-10 | 1.7e-15 / 1.4e-14 | 1.1e-08 |
| (0.5, 0.5) | 32 | 64 | 2.84275320645 | 1.2e-10 / 2.2e-10 | 8.5e-16 / 4.3e-16 | 9.3e-09 |
| (0.5, 0.5) | 48 | 64 | 2.84275320633 | 2.3e-15 / 6.5e-15 | 8.5e-16 / 5.0e-16 | 9.8e-13 |

### D = 8  (n = 6):  Delta = 2.711790054(5)  ->  **2.711790**

| x-map (x0, eps) | M | Nx | Delta | Fourier tail phi / G | Chebyshev tail phi / G | max residual of redundant eq. (C) |
|---|---|---|---|---|---|---|
| none | 24 | 48 | 2.71179002540 | 1.0e-08 / 1.1e-08 | 8.1e-12 / 4.4e-11 | 3.5e-07 |
| none | 32 | 40 | 2.71179056656 | 3.0e-10 / 2.6e-11 | 5.6e-10 / 1.7e-09 | 4.4e-06 |
| none | 32 | 48 | 2.71179003327 | 1.4e-10 / 3.3e-11 | 1.1e-11 / 4.7e-11 | 1.7e-07 |
| none | 32 | 56 | 2.71179005559 | 2.3e-11 / 3.4e-11 | 2.0e-13 / 1.2e-12 | 5.8e-09 |
| none | 32 | 64 | 2.71179005505 | 2.3e-11 / 3.4e-11 | 3.4e-15 / 2.6e-14 | 1.6e-09 |
| none | 32 | 72 | 2.71179004964 | 2.3e-11 / 3.5e-11 | 1.1e-15 / 8.7e-16 | 1.6e-09 |
| none | 48 | 48 | 2.71179003296 | 8.4e-16 / 2.9e-16 | 1.3e-11 / 4.7e-11 | 1.7e-07 |
| none | 48 | 56 | 2.71179005549 | 1.3e-15 / 3.2e-16 | 2.8e-13 / 1.2e-12 | 5.9e-09 |
| none | 48 | 64 | 2.71179005888 | 2.0e-15 / 3.1e-16 | 6.7e-15 / 3.0e-14 | 1.7e-10 |
| none | 64 | 56 | 2.71179005540 | 2.2e-16 / 1.1e-16 | 2.6e-13 / 1.2e-12 | 5.9e-09 |
| (0.5, 0.5) | 32 | 56 | 2.71179005435 | 2.3e-11 / 3.5e-11 | 1.2e-15 / 4.5e-16 | 1.6e-09 |
| (0.5, 0.5) | 32 | 64 | 2.71179005426 | 2.3e-11 / 3.5e-11 | 8.5e-16 / 3.7e-16 | 1.6e-09 |
| (0.5, 0.5) | 48 | 64 | 2.71179005320 | 1.7e-16 / 4.0e-16 | 8.9e-16 / 5.1e-16 | 2.5e-14 |

## 5. Independent checks

* **Symbolic**: `derive_bondi.py` (equations of sections 1; exact zeros). An earlier, independent
  derivation in polar–areal coordinates is in `derive.py`.
* **Redundant equation (C)**: see tables.
* **Time evolution in every dimension** (`src/evcheck.py`, logs `src/evcheck_n*.log`; for D = 4
  `crit3.py` + `ustar.py`). The family of section 3 was tuned to the threshold to 1e-15 in D = 6, 7, 8
  too, completely independently of the DSS solver. The accumulation time u* and Δ were read off
  successive extrema u_k of the central scalar field, u* − u_k ∝ e^{−kΔ/2}. Results from the
  triples of extrema that lie in the self-similar regime:

  | D | Δ from evolution (successive triples) | Δ from DSS solver |
  |---|---|---|
  | 4 | 3.4455, 3.4455, 3.4452 | 3.445452 |
  | 6 | 3.0076, 3.0092, 3.0097, 3.0101 | 3.009483 |
  | 7 | 2.8444, 2.8421, 2.8428, 2.8439 | 2.842753 |
  | 8 | 2.7120, 2.7110, 2.7137, 2.7115 | 2.711790 |

  The evolution is only good to about ±0.002, but it confirms that the solution branch reached by
  continuation in n is the black-hole-threshold solution in each dimension, and it tests the
  general-n equations in a second code. It does **not** test the 4th–6th decimals; those rest on
  the convergence tables and on the residual of the redundant equation.

## 6. Things that did not work (kept in `src/` for the record)

* A first DSS solver in polar–areal coordinates (`dss.py`, `refine.py`, `extract.py`). In that
  gauge the solution is almost a square wave in τ, the Fourier coefficients decay only like
  0.67^k, and Δ had not converged (3.21 at M = 16, 3.35 at M = 32). Abandoned.
* A multi-start least-squares search without an evolution-based guess (`search.py`) only found
  spurious low-resolution solutions.
* Bondi-gauge solver at too low resolution (M = 16, Nx = 24) converged to a spurious Δ ≈ 4.12;
  unmapped Chebyshev grids in D = 4 gave Δ = 3.4520 at Nx = 40 and still moved in the 5th decimal
  at Nx = 80.
* Double precision alone (see section 2).

## 7. Reproducing

    cd src
    python3 derive_bondi.py                                   # symbolic checks
    python3 crit3.py 2 14.03 14.08 0.8355 3e-4 1e-12 0.3 0.003 n2_s1
    python3 crit3.py 2 14.0504 14.05045 0.835641 1e-5 1e-15 0.82 0.003 n2_s2
    python3 extract_bondi.py crit_n2_s2.pkl 32 40 10.5 solB_n2_a.npz
    python3 refineB.py solB_n2_a.npz 32 80 solB_n2_M32_N80.npz
    python3 refineB.py solB_n2_M32_N80.npz 32 48 solB_n2_M32_N48t.npz x0=0.2 eps=0.15
    python3 cont.py solB_n2_M32_N48t.npz 6 0.25 32 48 c1
    for n in 4 5 6; do python3 table.py solB_c1_n$n.npz 0 0 32,48; done
    python3 tableL.py solB_n2_M32_N48t.npz 0.2 0.15 24,48 32,32 ... 80,56   # etc., see tablesL.log
    python3 tableL.py solT_n6_M32_N48_0_0.npz 0 0 24,48 32,40 ... 64,56
    python3 evcheck.py 4; python3 evcheck.py 5; python3 evcheck.py 6
    python3 make_report.py                                    # results.csv and the tables above

(The accumulation-time estimates passed to `crit3.py` came from earlier, coarser bisections.)

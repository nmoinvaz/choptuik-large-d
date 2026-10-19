# Choptuik's critical solution as a periodic boundary-value problem

## Equations

Spherically symmetric massless scalar field in D = n + 2 dimensions, polar-areal gauge,
ds² = −α² dt² + a² dr² + r² dΩₙ². With Φ = ∂ᵣφ, Π = (a/α)∂ₜφ and the normalisation
G_ab = 2n (∂_aφ ∂_bφ − ½ g_ab (∂φ)²):

- Hamiltonian constraint: ∂ᵣa/a = r(Φ² + Π²) + (n−1)(1 − a²)/(2r)
- Slicing condition:      ∂ᵣα/α = r(Φ² + Π²) + (n−1)(a² − 1)/(2r)
- Wave equation:          ∂ₜΦ = ∂ᵣ(cΠ),  ∂ₜΠ = r⁻ⁿ ∂ᵣ(rⁿ cΦ),  c = α/a

Self-similar coordinates τ = −ln(−t), x = r/(−t); fields p = r(Φ + Π), m = r(Φ − Π):

- a_x = a[(p² + m²) + (n−1)(1 − a²)]/(2x)
- c_x = c(n−1)(a² − 1)/x
- p_τ + (x − c)p_x =  c_x p − cp/x + nc(p + m)/(2x)
- m_τ + (x + c)m_x = −c_x m + cm/x − nc(p + m)/(2x)

Gauge: the time coordinate is chosen so the past light cone of the singularity (the self-similarity
horizon, SSH) is x = 1, i.e. c(τ, 1) = 1. Equivalently τ = −ln r_SSH(t) and x = r / r_SSH(t).

## Boundary-value problem

The critical solution is periodic in τ with period Δ, with p, m odd and a, c even under τ → τ + Δ/2.
Regularity at x = 0: m₁ = −p₁ (free function p₁(τ), plus the free central lapse c(τ,0)).
Regularity at x = 1: the p-equation degenerates; p(τ,1) solves ∂_τp + (1 + c₁ − n/2)p = (n/2)m,
c₁ = −(n−1)(a(τ,1)² − 1). Free data there: m(τ,1), a(τ,1). Both endpoints use Taylor series of order 4
(centre, started at x = 10⁻⁴) and 8 (SSH, started at 1 − x = 0.03), generated order by order.

Discretisation: Fourier in τ up to harmonic k_c, dealiased (products on N ≥ 3k_c points, right-hand
sides truncated to k ≤ k_c). Multiple shooting in x with 12 interior nodes, RK4, steps in ln x near the
centre and ln(1 − x) near the SSH. Newton/Levenberg–Marquardt with a block-structured finite-difference
Jacobian. Unknowns: Fourier coefficients of the boundary data and node states, and Δ; one phase condition.

## Growing modes

Perturbations δZ = e^{λτ} × (periodic): the same machinery with every τ-derivative of a perturbation
replaced by ∂_τ + λ. λ is a root of det M(λ) = 0, M the linearised mismatch.
γ = 1/λ₀ for the single physical growing mode.

A second root in (0, 4) is a gauge mode: shifting which ingoing light ray is used as x = 1 (equivalently a
time translation of the singularity). Light rays near the SSH separate at the rate 1 + c₁(τ), so this mode
has λ_gauge = mean over τ of 1 − (n−1)(a(τ,1)² − 1); the computed root matches that to 10⁻⁵ and converges
to it with resolution.

## Initial guess

From our own near-critical collapse simulation (dx = 0.0004 run 1 part in 10¹⁴ below threshold):
slices re-expressed in self-similar variables over one echo, with the SSH found as the periodic ingoing
light ray. Note the factor √(2π) between the simulation's field (G = 1) and the solver's normalisation.

## Large-D structure (session of 2026-10-18)
- Delta-selection strength s_Delta (Schur complement of dF/dDelta, resolution- and regularisation-independent):
  1.5e-3 (D=4.1), 2e-4 (5), 4.2e-5 (6), 6.8e-6 (7), 2.1e-6 (8), 1.1e-6 (8.5), 4e-7 (9.2), 1.7e-7 (9.85): ~e^{-1.4 D}.
- Mechanism: horizon-singular piece ~ (1-x)^{s0}, s0 = n/(2 lambda_g) - 1. Moving the last node x_K: s_Delta ~ (1-x_K)^{2.5} at D=6
  (s0=2.2) and (1-x_K)^{4.1} at D=8 (s0=3.6); the |F| floor scales the same way, so Delta precision ~|F|/s_Delta is intrinsic:
  ~5e-8 (D=6), 3e-5 (D=8), 1.3e-4 (D=9.85), 8e-4 (D=10.1), >1e-2 beyond D=10.4 in double precision.
- Solver fixes: RK4 step counts depended on Delta via ceil() (1e-9 jumps) -> freeze_steps; central-difference Jacobian with
  eps=3e-7 (horizon segment strongly nonlinear in a_SSH); Levenberg-Marquardt (spec_c.lm_solve). Near-null Jacobian directions
  are top harmonics of the centre data (numerical), not physical.
- gamma sensitivity along the weak Delta direction: dgamma/dDelta ~ 0.16.
- Out-of-sample fits (train D<=8, predict to 10.1):
  gamma = 1/2 - n^-1/2 (0.0677 + 0.307 n^-1/2 - 0.597/n + 0.543 n^-3/2): prediction error 1.9e-5 (1/n series: 1e-3).
  Delta = n^-1/4 (4.76 - 4.04 n^-1/2 + 10.1/n - 8.1 n^-3/2): prediction error ~5e-4; exponent 1/4 not pinned by free fits.
  Adding e^{-kappa n} terms did not improve predictions (kappa unstable).
- EEG NLO condition beta''+3beta'=0 at beta=0 (B''/B' = -3): measured +1.8 (D=5) -> ~+0.8..1.4 (D=8..10), noisy; inconclusive.

## Attempt to derive the first gamma coefficient (2026-10-18)
- Growth mode at D=6-8 is not an amplitude/phase change of the background (corr with B, B' small) and varies strongly with x.
- In the reformulated equations the SSH constraint is identically implied at x=1; with r = s/n slaved, the 1/n expansion never
  sees the characteristic singularity at the SSH. The singular branch ~ (1-x)^{n/(2 lambda_g)} is exponentially small in n.
  => every order of the regular 1/n expansion leaves a periodic function B(tau) (and any growth rate) free, apart from local
  conditions at zero crossings of B (EEG's NLO condition is the first). Delta, the profile and lambda0 are fixed beyond all orders.
- Natural expansion parameter mu = (2 pi/Delta)^2 / n: 1.66 (D=4), 1.27 (5), 1.09 (6), 0.98 (7), 0.89 (8). Not small in the data
  range; mu ~ 1.74/sqrt(n) -> mu = 0.1 needs n ~ 300. The fitted c0 = 0.0677 is a pre-asymptotic interpolation coefficient.

## Zero-crossing condition at next order (2026-10-18, /home/claude/largeD)
- verify_eqs.py: E1-E4 follow exactly from the DSS equations (sympy residuals 0).
- nlo.py: with B := Q(tau, 0), Q1(tau,1) = B''/2 + 3B'/2 + B + 2BB'^2 + O(B^2 ...) (small-B), and the NLO term of
  (1-eps)A(1) is 2B(B''+3B'+4BB'^2): reproduces EEG's eps*beta(beta''+3beta') near zeros (beta = sqrt2 B).
- Exact identity at the horizon edge: (1-eps)A(1) = 2Q(1)^2 + 2eps^2 s(1)^2 - eps A_x(1)/(1+eps A(1)).
- nnlo_zero.py: near a zero, through eps^2: 2B'^2 d^2 + 2eps B'(B''+3B') d + 2eps^2[((B''+3B')/2)^2 + B'^2];
  minimum = +2 eps^2 B'^2 > 0 for ANY profile. At NLO-only truncation minimum = -(eps^2/2)(B''+3B')^2 -> EEG's
  condition B''+3B'=0 is a truncation artifact; positivity imposes no condition at any order.
- zero_shift.py (exact solutions, B_c = centre value): B_c''/B_c' at zeros = -1.49 (D=4), -1.86 (5), -1.74 (6), -1.66 (7),
  -1.46 (8), -1.10 (9.2) [10.1 noisy]: not -3 and drifting away from it. Zero shift horizon-centre: +0.24, +0.03, -0.044,
  -0.072, -0.083, -0.087 vs first-order prediction -(eps/2)(B''/B'+3): -0.38, -0.19, -0.157, -0.134, -0.129, -0.132.
  Same sign from D>=6, ratio 0.28 -> 0.66 rising with D (pre-asymptotic: mu ~ 1).
- Conclusion: the regular 1/D expansion has no constraint at any order (centre regularity, SSH regularity, positivity all
  automatic). Profile, Delta and lambda0 are selected only by the exponentially small horizon branch.

## Long-series scan to D=18.2 and the Delta(D->inf) limit (2026-10-18 evening)
- Long horizon series (spec_long.py, x_K=0.6, Ks=60) validated: D=4 agrees with standard kc23 to 3e-9; kc31 vs kc23 at D=8: 5e-9,
  at D=16.3: 1.1e-8; x_K=0.5/Ks=90 vs x_K=0.6 Delta agree to 1.2e-8 (D=14) and 1e-10 (D=17.3). Scan stopped at D=18.2 (residual
  floor ~1e-9 in junk directions; Delta unaffected).
- Delta levels off: Delta = L + a/n + b/n^1.5 + c/n^2 (no n^-1/2 term; free quartic gives its coefficient -0.14) fits D>=8 to ~1e-6;
  L = 1.6346 (D>=8), 1.6343 (D>=10), 1.6348 (D>=12), 1.63-1.64 with one more term; a ~ 12.2. Local exponent of Delta peaked 0.259 at
  D~7 and falls to 0.195 at D=18. Contradicts the expectation Delta -> 0. Plot: delta_graph.png.
- gamma above D~14 is NOT reliable yet: determinant roots and regularised sigma_min dips differ between x_K=0.6 and 0.5 by
  delta-lambda ~ 0.004 at D=14 (gamma ~1e-3); gamma non-monotonic beyond D~17. Junk near-null directions contaminate det M(lambda).
- formula_zoo.py: no standard family extrapolates gamma (D<=11 -> 11..17.4) better than ~3e-4, Delta worse when D<4 included.

## Jacobian accuracy fix (2026-10-18 night)
- The residual floor above D~17 was a solver artefact: the central-difference Jacobian (eps=3e-7) of the horizon-series
  block has error ~ eps^2 * F''' ~ 6 (|J|~1.6e4) at D=19.8 (interior blocks fine at 1e-8). Function noise is only ~2e-12.
- Fix: complex-step derivative of the SSH-series block (spec_c.horizon_block_cs / jacobian_cs; _D_unit, trunc, phys, spec
  wrapped to act on Re/Im separately) + chord-corrected LM (spec_c.lm_svd2: one SVD per Jacobian, damping grid, up to 3 chord
  corrections per trial).  D=19.846: |F| 6e-9 -> 6e-13 in 3 Jacobians.
- Consequence: old scan Delta values at D>~18 were off by up to ~1e-5 (D=19.846: old 2.115439137, converged 2.115425952;
  x_K=0.5/Ks=90 converged 2.115425924, setups agree to 3e-8).
- Near-null structure at D~20: profile harmonics k=3,5,7,... are weakly determined (sigma/smax 1e-9, 3e-11, 1e-12, ...),
  i.e. the shape is fixed only by the exponentially weak horizon condition; Delta direction sigma ~2e-7 relative.
- Same horizon-block FD error affects the mode matrix (eps=1e-6, error ~70!) -> explains unreliable gamma above D~14.
  spec_c.mode_matrix_cs replaces the horizon block by complex step. D=8 check: sign change and sigma_min dip at 2.2385-2.2390.

## Speed (2026-10-18 evening)
- Profile: one residual = ~160k numpy calls on 72-point arrays; Python dispatch dominates (batch of 100 costs 9x less per
  point than single calls). numpy/OpenBLAS already SIMD (Haswell kernels) -- SIMD is not the bottleneck.
- fast.py (fast.install()): SSH series O(K^3)->O(K^2) (cached A^2, P^2+M^2, tau-derivatives; einsum Cauchy products),
  batched periodic solves, tau-derivative and truncation as dense 72x72 GEMMs. Verified: F at solution identical to 7e-17,
  J to 2e-14 relative. jacobian_cs 9.1s -> 4.4s, batched F 2.1x.
- fast.lm_svd3: all damping trials and chord corrections batched; stops on stagnation. Scan step D=19.85->20.15:
  converged to 1.5e-11 in 2 Jacobians / 48 s total (old solver: >1 h without finishing).
- Porting pitfall (my bug in fast.py, fixed before use): in the m-step the P+M sum must include the freshly computed P[K].
- Seed 19.489 re-solved: Delta differs by 5e-9 between two converged runs at |F|~1e-11 -> Delta noise ~5e-9 there.

## D>26: the asymptotic fit breaks (2026-10-18 late)
- Rust kernel (/home/claude/rustkernel, RustResidual) now drives the scan: rust_solver.jacobian_rust (dedup FD + complex
  step for horizon & Delta) 0.48 s vs 5.1 s python; agrees to 3e-9 on shooting blocks. Cubic predictor needed (residual
  extremely sensitive to the horizon data: quadratic predictor |F|~0.2, cubic ~1e-2, constant 1e12).
- Fits Delta_inf + sum c_k n^-k/2 made on D<=24-27 miss D>27 by up to 4-7e-4 (smooth, growing). Local exponent
  q=-dlnDelta/dln n: slope dq/dln n steady -0.093 for D=17-25, then decays to ~-0.03 at D=30-31 (q flattening ~0.145).
  g=n^2|dDelta/dn| accelerates again after D~27 (7.59 at 26 -> 8.20 at 30.85). c0_min (centre lapse ratio) peaks at D~26.4.
- Numerical checks at D=30.05 (Delta changes): Ks 60->90->120: <1e-10; RK4 steps x2, x4: 2e-10; kc 23->31: 9.5e-8;
  stripping centre high-harmonic junk: ~5e-6; matching x_K 0.6->0.7: -1.2e-5 (x_K=0.7 converges to |F|=2e-10).
  All << 4e-4: the bend is not a resolution artefact of these kinds.
- Running: full x_K=0.7 scan from D=23 (scan_long5.py -> scan_long7.jsonl, long7_D*.npy) to confirm and extend to 40.
- CORRECTION (same night): the D>26 bend IS a precision artefact. Pinning test (start from the converged solution with
  Delta displaced, re-solve): D=24.5 (x_K=0.7): +-1e-4 displacement returns to within 1.6e-6 -> Delta pinned to ~1e-6.
  D=30.05 (x_K=0.7): displacements +-1e-4 and +-1e-3 are kept (90-99%) with |F| <= 6e-9 -> Delta NOT determined to 1e-3.
  x_K 0.6 vs 0.7 continuation differs by -1e-6 (25.3), -3e-6 (25.7), -6e-6 (26.1), growing ~x2 per 0.4 in D.
  The selection (beyond all orders) falls below double-precision noise around D~26. Reliable data: D<=~25.
  Fix would need extended precision (double-double) in the Rust kernel (generic Sc trait) and in z.
- Reliable-range fits (8<=D<=25): Delta_inf 1.635-1.652 depending on family; best 4-param fits on D>=12: 1.641-1.649.
  omega_inf = 2pi/Delta_inf 3.79-3.86. Candidates inside the window (cannot discriminate): 2pi/j_{1,1}=1.63979,
  pi^2/6=1.64493, sqrt(e)=1.64872.

## Rust gamma, double-double, and the profile method (2026-10-18, ~22:00)
- Rust kernel (repo nmoinvaz/dss-residual-kernel, main): exact Jacobian and growth-mode matrix M(lam) by forward-mode
  Dual numbers; agree with python jacobian_cs / mode_matrix_cs to ~1e-10 relative; growth rate to 5e-10.
  gamma_rust_sweep.py: gamma for all 56 reliable points D=7.75-24.9 in ~90 s (gamma_rust.jsonl).
- SVD: the 3-7 s per 941x941 SVD was OpenBLAS oversubscription; OPENBLAS_NUM_THREADS=1 -> 0.40 s (faer in Rust 0.43 s).
- residual_dd: double-double residual; D=30 rounding noise 1.3e-11 (f64) -> ~1e-27. dd_solver.dd_solve: LM with DD z and
  residual, f64 Jacobian. At D=30 it does NOT lower |F| below 1.8e-10: that floor is the junk (near-null) directions,
  not rounding.
- CORRECTION OF THE CORRECTION: the displacement ("pinning") test measured the solver's ability to walk the weak valley,
  not whether the equations pin Delta. Profile method (dd_solver.profile: fix Delta, minimise |F| over the rest):
  phi(Delta) is V-shaped with a clear minimum. D=28.0: Delta* 1.9868625 (scan 1.9868673); D=30.05: 1.9642327 (x_K=0.6),
  1.9642238 (x_K=0.7), scan 1.9642342; D=31.3: 1.9517443 (scan 1.9517412). x_K=0.6 Delta-signal ~1e-2 per unit Delta
  (floor 1.8e-7), x_K=0.7 ~5e-6 per unit Delta (floor 1.8e-10).
  => scan values to D=31.3 are good to ~1e-5, and the bend (up to 7e-4 below the D<26 asymptotic fit at D=31) is REAL.
- gamma law (consistent Rust data D<=24.9): limit 0.500-0.505 depending on family; 1/2-gamma local exponent 0.63->0.71.
- CORRECTION #3 (22:40): the V-shaped profiles above were ALSO a solver artefact. The fixed-Delta inner solves were
  warm-started from the neighbouring Delta and did not converge for displaced Delta, so phi rose artificially.
  Starting each inner solve on the valley tangent (dd_solver.valley_tangent: J_rest t = -J_Delta) they converge:
  D=24.5: phi 3.3e-10 at the centre, 4e-9 at +-2e-4 -> V-shaped, Delta pinned (~1e-5).
  D=28.0: phi = 1.49e-8 +-1% over +-2e-4 -> flat, Delta NOT pinned. D=30 (x_K=0.7): flat/monotonic, not pinned.
  => Reliable Delta only up to D ~ 25-26. The "bend" above 26 is NOT established.
- Where the floor sits (floor_anatomy.py): almost entirely in the last two blocks (0.5->0.6 segment and horizon
  series->0.6), equal size, low odd harmonics (k=5,7 at D=24.5: 2.4e-10; k=3 at D=31.3: 3.2e-7). It is the mismatch
  between the interior solution and the regular horizon series at x_K: the singular horizon modes u^{sigma_k}
  (Re sigma_k = s0) that the regular series cannot represent, suppressed by u_K^{s0}. The parameters that remove them
  (Delta and the shape modes) are the near-null directions, so LM cannot remove them and the selection drowns in the
  floor around D ~ 26. Fix needs the singular-mode amplitudes as explicit, O(1)-scaled conditions (precondition by
  u_K^{-sigma_k}), or a matching point further from the horizon (needs many more series terms).

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

## Attempts to pin Delta above D ~ 26 (2026-10-18, night)
- regular_newton.py: Newton with the matching rows split into the part the horizon data absorb (Q_reg^T d) and the
  singular-mode content (W Q_perp^T d, W = Sigma^-1 U^T). Did not converge: steps that include the near-null
  directions are far outside the linear range; with a 1e-9 cut the decrease is small.
- Kernel option match_node (inward segments beyond the match node). Reproduces Delta at D=20.446 to 2.7e-8
  (match at x = 0.5/0.4/0.3). At D=27.96 the profile is still flat (phi 1.94-2.03e-8 over +-1e-4, match at 0.4 and 0.3).
- floor_anat2.py (match at 0.3, D=27.96): the whole floor (2.0e-8) moves to the series->0.6 block (k=3 harmonic
  1e-8); all shooting blocks <= 7e-11. In the SVD of J the floor lies in 4 directions with sigma/sigma_max in
  [1e-13, 1e-10) (Delta + shape modes), 6e-10 in the 18 junk directions, 1e-9 in the well-conditioned rest.
- Reading of the floor size: if it were singular-mode content at a fixed amplitude it would scale like u_K^{s0}
  ~ e^{-0.55 D}; instead it GROWS ~e^{+D} (2.4e-10, 1.5e-8, 3.2e-7 at D = 24.5, 28, 31.3). The old Ks/RK4/kc checks
  compared free-solve Delta only, which is not informative here (Delta is not pinned). floor_test.py re-does them
  comparing the floor itself.

## The floor is a solver failure: nonlinear elimination (2026-10-18, ~23:00)
- floor_test.py (D=27.96): converged |F| unchanged by Ks 60/90/120, RK4 steps x2, log_steps x2 (1.485-1.489e-8).
  Not a discretisation floor.
- weak_dirs.py: the 1.5e-8 lies in ONE pair of directions, s/smax = 4.1e-11 / 3.9e-11, mostly centre-data p, k = 3
  (cos/sin), with Gauss-Newton step only 3.8e-4 / 4.2e-5. The Delta direction (s/smax 8.5e-9) carries ~0. Next pair
  (k = 5, 1.8e-13) carries 6.5e-10; below 7e-14 junk (high centre harmonics). LM could not take the k = 3 step: it
  is outside the linear range of the stiff part.
- ne_solver.py: nonlinear elimination. Strong equations (s/smax >= 1e-10) solved exactly by Newton for each value of
  the weak coordinates (1e-10 > s/smax >= 3e-14), weak equations by Newton on the Schur complement; junk frozen.
  D=27.96: weak residual 1.48e-8 -> 3.4e-11 -> 1.8e-14 in 2 outer steps; |F| = 3.8e-10, all of it junk.
  Delta 1.9868678268 -> 1.9868672129 (-6.1e-7).
- Fixed-Delta profile with NE (ne_prof.py): inner solves converge only within ~1e-5 of Delta*, selection residual
  4e-13 (0), 4.5e-11 (+1e-5), 3e-12 (-1e-5); further out the weak equations stall at ~1e-9. Not a clean V; the
  selection signal (~1e-11 per 1e-5) is comparable to the junk residual (1e-10). Robustness of Delta is being
  tested instead by changing the discretisation (ne_disc.py).

## NE scans and the junk ambiguity (2026-10-18, ~23:20)
- scan_ne.py runs: re-solve of all stored seeds (D = 7.75-23.3) and continuation with NE (x_K = 0.6 from 23.3, x_K = 0.7
  from 24.0), ~15 s per point, gamma per point. LM pre-solve needed for continuation predictors (|F| ~ 1e-2).
- Below D ~ 17.5 NE agrees with the old data to 1e-13. Above, the old values were higher by 1e-6 (D = 18) to 1.2e-5 (D = 23).
- BUT the NE answer is not unique either: at D = 23.038 (x_K = 0.7) NE from the seed gives Delta 2.0546921187 (|F| 9e-14,
  junk 1e-14), LM then NE gives 2.0546996221 (|F| 1.2e-12, junk 1.1e-12). junk_delta.py: the two differ by 0.097 in the 16
  junk directions (centre p harmonics k ~ 7-19, changes 1e-2..5e-2), 5e-4 in the weak ones. Junk coordinates move F by
  only ~1e-12 but move Delta by 7.5e-6: Delta resolution needs junk residual << 1e-14.
- ne_dd.py (double-double z and residual, DD central differences for J V_o in the 22 weak+junk directions, Schur
  complement): strong residual -> 1e-25..1e-28, but the outer Newton step does not reduce the junk residual (1.0e-14 and
  1.1e-12 stay); Delta of the second moves only to 2.0546984. Open.
- x_K = 0.7 minus x_K = 0.6 (both NE continuation, cmp_setups.py): +8e-6 (D = 23) .. +1.6e-5 (24.5); gamma -1e-6 .. -2e-6.
  This spread is the current error bar on Delta above D ~ 18.

## Junk sector: what it is not (2026-10-19, ~00:10)
- Matching at x_K = 0.4 with Ks = 150 (xk04_dbg.py, D = 23.04): the x_K = 0.6 NE solution leaves 2.2e-6 in the series
  block (singular content seen ~150x more strongly at u = 0.6), but the junk band of J does NOT move (smallest s/smax
  still 2e-14 .. 2e-18; weak band slightly lifted to 1e-10 .. 2e-11). Also the horizon series is not converged at u = 0.6
  with 150 terms: |c_k| u^k = 5e-11 at k = 150 (ratio ~0.886 per term, radius ~0.68), would need Ks ~ 260.
  So the junk is not controlled by the strength of horizon regularity; x_K = 0.4 is not the fix.
- mode_spectrum.py (D = 23.04): the smallest singular values of M(lambda) are 1e-17..1e-18 relative for EVERY real
  lambda in [-3, 0.3]: no isolated weakly damped modes; the near-null sector is lambda-independent (high-k sector,
  where d/dtau ~ i k omega >> lambda). Origin still open (candidates: centre-regular and horizon-regular high-k
  solutions nearly parallel at the match; truncation/dealiasing of the top harmonics).
- Consequence: the junk equations' residual (~1e-13) is at the level of discretisation errors, so their 'solution'
  is noise, and that noise moves Delta. smooth_junk (prior: small high centre harmonics) is the regularisation in
  use; the local matrix (kc 31, smooth vs frozen, x_K 0.6/0.7) measures how much the answer depends on it.
- Continuation drift: the x_K = 0.7 NE scan reached D = 40, but its slope steepens from -0.0117 (D = 37) to -0.033
  (D = 40) with junk residual 1-3e-9: the branch runs off above D ~ 38. x_K = 0.6 blew up at D = 35.15 (|F| 1.5e4) and
  is retrying. Values above D ~ 26 are not trustworthy beyond ~1e-3.
- sj_test.py with 8 rounds (D = 23.04): smoothing + NE alternation converges in ~4 rounds. x_K = 0.7 -> 2.0546955863
  (all centre harmonics k >= 9 -> 0); x_K = 0.6 -> 2.0546914107 (7.3e-6 of k >= 9 content stays: it lies in weak, not
  junk, directions there). 4.2e-6 apart: the prior is not the same constraint in both setups. scan_ne NE_SMOOTH now
  iterates to |dDelta| < 1e-10.
- x_K = 0.6 continuation stopped at D = 34.94 (steps shrinking, |F| 4.7e-8).
- junk_profile.py (D = 23.04, x_K = 0.6): the junk directions are EVANESCENT high-k perturbations: their harmonic-k
  content peaks at x ~ 0.05-0.1 and decays outward, faster for higher k (k = 23: 3e-2 at x = 0.1, 2e-5 at 0.3,
  5e-8 at 0.6; k = 11: 1e-1 -> 2e-4). They carry a small Delta component (5.8e-5 aggregated) and horizon data (4e-4):
  a 0.1 move along them changes Delta by ~6e-6 while F changes by ~1e-12, i.e. the evanescent tail at the horizon
  can stand in for a Delta change. The weak (shape) directions are the same thing for k = 3-7, less evanescent.
  So the high-k centre content is fixed only by exponentially weak horizon conditions, in the continuum too; with
  a smooth (analytic in tau) solution its true size is small, so the smoothness prior should cost < ~1e-8 in Delta
  from that source. The 4.2e-6 between x_K 0.6 / 0.7 (smoothed) is something else (discretisation of the weak part).
- Running in the cloud: kc = 31 with smoothing, x_K 0.6 and 0.7 (scan_ne6_kc31_sj, scan_ne7_kc31_sj).
- D = 23.038 with smoothing: x_K 0.6 / kc 23: 2.0546914107; 0.6 / 31: 2.0546914340; 0.7 / 31: 2.0546917021;
  0.7 / 23: 2.0546955863 (kmin 9) or 2.0546959388 (kmin 13). Three setups agree within 3e-7; x_K = 0.7 at kc = 23
  is an outlier (+4e-6), whatever the smoothing threshold. Likely a constrained direction falls below cut_junk there.
- consensus.py: x_K = 0.6 plain continuation (kc 23) agrees with kc 31 + smoothing to ~1e-8 up to D = 25.7. The outlier
  is x_K = 0.7 at kc = 23 (+1e-5), plain or smoothed. Explanation: at x_K = 0.7 the selection strength s_Delta is ~700x
  weaker (2e-8 vs 1.6e-5 at D = 28), so the same discretisation error (kc truncation) moves Delta 700x more; kc 31
  shrinks that error and 0.7/31 agrees again (3e-7). => smaller x_K is better, provided the horizon series converges
  there: x_K = 0.5 needs Ks ~ 120, x_K = 0.4 needs Ks ~ 260 (radius ~0.68-0.74 in u).

## A series that predicts unseen points (2026-10-19, ~00:40)
- Agreement of setups: x_K = 0.5 (Ks 120) vs 0.6 (kc 23 plain / kc 31 smoothed) within 1e-8..1e-7 for D <= 25.5;
  x_K = 0.4 (Ks 260) = 2.0546914639 at D = 23.04 (x_K 0.5: 2.0546914635). Above 26, x_K = 0.6 kc 23 plain drifts
  (+3e-5 at D = 30 vs kc 31 smoothed).
- series_factory.py / coeff_stab.py: weighted polynomial fits in x = 1/n (and 1/(D-1)) on D = 8..25.5 (errors 1e-8 below
  17.2, 3e-8 above). Converge (chi2/dof 0.6-0.9) at degree 9-10: c0 = 1.64407-1.64409, c1 = 10.35, c2 ~ -47 (1/n);
  c0 stable for Dmax 25.5 / 29 / 32 and lower cuts 8-12 (1.6432-1.6445 across all variants). 1/sqrt(n) series unstable
  (half-integer powers not supported by the data).
- Hold-out: the D <= 25.5 series predicts the kc 31 smoothed points at D = 28, 30, 32 to 1e-7..2e-7 (1/(D-1) series: 2e-8
  at 28 and 30). Forcing Delta_inf = pi^2/6 (chi2/dof 1.0 on the fit range) misses by +6.6e-7, 1.7e-6, 3.6e-6; the
  half-integer family (L = 1.6418) by -1.1e-6, -2.7e-6, -4.9e-6. Forcing 2pi/j11 or sqrt(e): chi2/dof 25-70.
  => Delta_inf ~ 1.6441 (pending the x_K = 0.5 / 0.4 values above D = 26). c1/c0 = 6.295 (2 pi = 6.283).
- gamma_series.py: gamma = g0 + c1/n + ... on D = 8..25.5 (Rust gamma below 17.2, NE above), hold-out kc 31 smoothed
  D = 25.5..33: free g0 -> 0.4997..0.4998 (1/n, 1/(D-1)), with g0 = 1/2 fixed: c1 = -0.755 (-0.7550 1/n, -0.7552 1/(D-1)),
  c2 ~ +9, c3 ~ -120; hold-out max 1.1e-7..1.5e-7. Forcing c1 = -3/4 makes the hold-out worse (1-2e-6).
- pade_fit.py (rational in x = 1/n, fit D = 8..25.5): Delta [4/3]: Delta_inf = 1.643883, fit rms 2e-8, hold-out max
  2.1e-7; [3/3]: 1.644131 (hold-out 2.6e-6). gamma [3/3]: gamma_inf = 0.499995, fit rms 1e-9, hold-out 1.6e-7.
  => gamma_inf = 1/2 to ~1e-5; Delta_inf = 1.6439..1.6442.
- Shape of the critical solution with D (centre data, kc 23 below 25.5, kc 31 smoothed above): first harmonic of the
  centre field h1 = 1.10 (D = 7.75), 0.49 (16), 0.36 (23.7), 0.29 (32.5): h1 sqrt(n) = 2.64 -> 1.59, still falling
  slowly; h3/h1 = 0.25 -> 0.10 (profile becoming sinusoidal); 1 - min(a0) = 0.58 -> 0.18 (~ n^-0.43..-0.58);
  mean(1 - a0) * n ~ 3.0-3.2 (metric deviation ~ 1/n on average). => the D -> infinity critical solution looks
  weak-amplitude and single-harmonic: Delta_inf may be an eigenvalue of a (weakly nonlinear) problem about flat
  space, which would be the analytic route to the exact value.
- The x_K = 0.6 kc 23 PLAIN continuation above D ~ 26 drifted into a distorted solution (h3/h1 0.29, h5/h3 0.64 at
  D = 29.3): its Delta values there are wrong (up to 3e-5). Use only the smoothed / x_K <= 0.5 scans above 25.5.
- Smoothing blow-up at x_K = 0.5 (Ks 120): the junk move (|a| ~ 0.2) is only linearly null; its second-order residual is
  ~0.12 and NE's undamped inner Newton cannot recover. Fix (scan_ne): after smooth_junk, if |F| > 1e-8 run LM with the
  junk directions excluded (dd_solve cut = 3e-14), then NE. D = 24.596: 0.12 -> 4e-5 -> 4e-10, converged in 3 rounds,
  Delta = 2.0304571464.

## The junk coordinates set Delta; solving them (2026-10-19, ~00:00-01:30, local i7-1185G7)
- run_local scans: x_K 0.7 reached D = 40. cmp_setups x_K 0.7 - 0.6: +8e-6 (23), +1.8e-5 (25.3), 0 (26.9), -1.8e-4 (28.9).
  kc 31 - 23: -1.7e-6 .. -3.9e-6 (23-25).
- ne_disc.py: the stored seeds have |F| 1e-4..7e-3 on the current kernel, so NE straight from them does not converge; far
  starts now get the damped LM pre-solve, variants start from the converged base, and j<scale> moves the start along the
  junk directions. Change of Delta from the x_K = 0.6 base:
    D        xK07     kc31     Ks90     RK4 x2   d+1e-4   d-1e-4   j+0.05   j-0.05
    24.535   +2.5e-6  -1.1e-8  +1.6e-9  -2.1e-8  +6.8e-9  -3.8e-9  -3.6e-7  +3.6e-7
    26.892   -1.8e-6  +2.5e-8  +1.1e-8  +3.2e-8  +5.7e-7  -5.5e-7  -6.9e-6  +6.9e-6
    27.962   +5.2e-6  +2.9e-7  +1.7e-7  +3.4e-7  +7.3e-7  -3.8e-7  -6.8e-6  +6.6e-6
    30.050   -9.6e-6  (base 1.9644572, weak residual stalled at 1e-13)
  Delta is pinned once the junk coordinates are fixed, and the resolution is converged to < 4e-7 up to D = 28. The junk
  coordinates move Delta linearly, ~20x more at D = 27 than at 24.5. The scan-to-scan spread is mostly this.
- The junk directions are the centre-p harmonics k >= 7 (D = 28: 16 directions, 97% c.p, s/smax 6e-18..3e-15). The seeds
  carry a bump at k = 7..9 growing from 2e-3 (D = 19) to 0.2 (D = 30): undetermined coordinates drifting through the cubic
  predictor.
- Smooth gauge (those harmonics zeroed, or minimised along the junk directions) rejected: it leaves the junk residual at
  2e-11, against 1e-13..1e-14 for the best NE solutions. The equations do fix the junk coordinates, at finite amplitude,
  where the f64 Jacobian cannot see them.
- junk_solve.py: Gauss-Newton on the junk coordinates with a forward-difference Jacobian (h = 1e-2, columns on a process
  pool) and the strong and weak equations re-solved by ne_solve at every evaluation. Converges quadratically. D = 23.038:
    setup          Delta (NE)     junk res   Delta (junk-solved)   junk res
    x_K 0.6 kc 23  2.0546896840   2.0e-11    2.0546914289          7.6e-17
    x_K 0.6 kc 31  2.0546897062   2.0e-11    2.0546914511          8.0e-16
    x_K 0.7 kc 23  2.0546996631   1.1e-12    2.0546921687          2.1e-16
  The setup spread drops from 1.0e-5 to 7.4e-7 (x_K) and 2.2e-8 (kc). The cloud x_K = 0.6 scan (junk 1e-13) was already
  right at D = 23; the local LM-then-NE path left 1e-12 and was 7.5e-6 high.
- Against the smoothness prior (smooth_junk, previous section): x_K 0.6 junk-solved 2.0546914289 vs smoothed
  2.0546914107 (1.8e-8), and both match the x_K 0.4 / 0.5 / 0.6 consensus 2.05469146 to 3e-8. x_K 0.7 junk-solved is
  7.4e-7 off, smoothed 4.2e-6. The junk residual converges quadratically to 1e-16 with a 16x16 Jacobian of rank 16,
  so it is a deterministic equation rather than noise, and solving it is the better-defined fix for the weak setups.
- scan_ne NE_JUNK=<workers> junk-solves every point (records js, Delta_ne, junk_ne). Running x_K 0.6 and 0.7, D = 23-40.

## The horizon series' radius shrinks with D (2026-10-19, ~01:30)
- Radius of convergence (in u = 1 - x) of the regular horizon series, from coefficient growth (k = 60..160), the same for
  every tau-harmonic (so a property of the solution): 0.70 (D = 23), 0.60 (25.7), 0.47 (30.1), 0.49 (33.1), 0.43 (36.0);
  roughly r ~ 2.5 / sqrt(n) above D ~ 30 (a near-horizon layer of width ~ n^-1/2).
- Consequences: x_K = 0.4 (u = 0.6) is outside the radius above D ~ 25.5, x_K = 0.5 above ~29 (at D = 34 the x_K = 0.5
  residual from the x_K = 0.6 solution is 6e3), and x_K = 0.6 (u = 0.4) approaches it by D ~ 36 (u/r = 0.92): this is
  why the kc 31 smoothed x_K = 0.6 scan departs from the D <= 25.5 series above D ~ 32 (-1e-6 at 34, -4e-6 at 36) and
  why the plain x_K = 0.5 scan scattered by 3e-7 near D ~ 27. High D needs x_K -> 1 as ~1 - 1.6 / sqrt(n).
- The cloud x_K = 0.5 smoothed scan is replaced by x_K = 0.7, kc 31, smoothed (to D = 45). Local jobs x4_23 / x5_31 are
  pointless above D ~ 25 / 29.

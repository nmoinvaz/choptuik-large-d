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

## RK step error (2026-10-19, ~01:30-02:00, cloud + local)
- x_K 0.7 junk-solved at D = 29.326 vs J1 (x_K 0.6): RK x1 -7.9e-6, x2 -3.7e-7 (consistent with h^4), x4 -1.0e-5 but with
  junk 2e-13 (unconverged, spoiled). Local runner T1 is settling it (RK 1, 2, 3, 4, 6, junk <= 1e-15). Option NE_RK.
- RK x2 re-solve of the fit set D = 7.75-17.3 (x_K 0.6, kc 23): RK2 - RK1 oscillates +-2e-8 with D (period ~4 in D: the
  ceil() in freeze_steps). RK2 data are smooth: series fits D = 8-17.3 to rms 1e-9 (1/n) and 2e-10 (1/(D-1)), 3e-11 with
  D >= 10 and Richardson (RK2 + (RK2 - RK1)/15).
- Limit from the step-converged D <= 17.3 data alone, across variables 1/D, 1/(D-1), 1/(D-3/2), 1/n and cuts D >= 8-11:
  1.6441-1.6450 (1/D lowest, 1/n highest): Delta_inf = 1.6446 +- 0.0004; pi^2/6 inside again. The earlier 1.6441 used RK x1
  data to 25.5. Step-converged D = 17-30.5 data (cloud: 17-25.5 RK2 junk-solved; local T4: 25.5-30.5) will narrow it.
- CORRECTION (02:20): the "1.6446 +- 0.0004 / pi^2/6 inside" numbers above came from unscaled Vandermonde fits of degree
  7-10 (x ~ 0.05-0.17, so x^10 ~ 1e-13), which lose ~1e-9 and made the limit wander. refit_rk.py, series_factory.py and
  constr_test.py now fit in x / max(x). On the step-converged set D = 8-22.4 (RK x2 + Richardson; RK x2 junk-solved above
  17.2): fit rms 2e-11 (D >= 10), limit 1.64389-1.64407 for x = 1/D, 1/(D-1), 1/n. Forcing pi^2/6 raises the rms 3-13x
  (6.9e-11 .. 2.6e-10); forcing 1.6440 does not (2.1-2.2e-11). D <= 17.3 alone does not constrain the limit at degree 10.
  => Delta_inf = 1.6440 +- 0.0002; pi^2/6 disfavored. Step-converged D = 22.4-30.5 still to come.

## Step-converged fit to D = 30 (2026-10-19, ~04:30)
- Data: x_K 0.6 RK x2 (Richardson with RK x1 where available; junk-solved above 17.2): cloud D = 8-24.5, local T4
  D = 25.7-30.1; x_K 0.7 RK x4 junk-solved above 30.2 (cloud). At D = 30.124 x_K 0.6 RK x2 and x_K 0.7 RK x4 agree to
  4.1e-8; at 29.326 the two x_K 0.7 RK x4 lineages differ by 1.5e-7 (f64 floor; DD test queued locally).
- final_fit.py (weights 3e-11 / 5e-10 / 1e-9 / 3e-8 by range, x / max(x), degrees 8-11, cuts D >= 10, 12; variables 1/D,
  1/(D-1), 1/(D-2)): Delta_inf = 1.64390-1.64397, chi2/dof ~4 (errors ~2x optimistic). Forced pi^2/6: chi2/dof 140-1300
  (D >= 10); forced 1.6438 / 1.6442: 6-34 / 13-90; forced 1.6440: 4.4-7.4.
  => Delta_inf = 1.64394 +- 0.00005 (omega_inf = 2 pi / Delta_inf = 3.82205). pi^2/6 excluded.
- gamma on the same points (D >= 10): free gamma_inf = 0.49995-0.50016; with gamma_inf = 1/2 the 1/D coefficient is
  -0.7577 +- 0.0005.
- Precision floor at D ~ 29 (local T1 + DD test): two x_K 0.7 RK x4 junk-solved fixed points 1.5e-7 apart in Delta
  (1.9718774140 J1 lineage, 1.9718773020 long7 lineage). Smaller junk-Jacobian steps (h = 1e-3) walk one onto the
  other (RK x6 h = 1e-3: 1.9718774115). With double-double z and residual (ne_dd) the strong residual drops to 1e-25..1e-28
  but the weak+junk residual stays at ~1.5e-15 and neither point moves: a nonlinear valley flat to ~1.5e-15 along which
  Delta varies by >= 1.1e-7. ~1e-7 is the intrinsic precision of this discretisation at D ~ 29. Runs from here use
  NE_RK = 3 and NE_JUNK_H = 1e-3 (cloud D 30-45, local T2 D 23-31).

## The junk stall above D ~ 27-31 and its fix (2026-10-19, ~06:00-08:05, cloud + local)
- Symptom: the junk solve (finite-difference Gauss-Newton, steps 1e-2, 3e-3, 1e-3) stalls at junk 1.7-4e-14 and takes
  hours: local T2 (x_K 0.7 RK x3) from D = 27.7, J1 (x_K 0.6) at 29.8-30.1, local T5 (kc 31) at 29.3, the cloud x_K 0.7
  RK x3 chain twice at 30.984 (2.5 h, Delta 1.7e-4 off the series at its NE start).
- Three causes, each fixed by an opt-in option of scan_ne / junk_solve:
  1. Misfiled junk directions. The junk set was s/s_max < 3e-14 at the start point. Converged points have a clean
     spectrum (30.584: 5.0e-10, 8.6e-11, 4.6/4.4/2.1/2.1e-13, 2.0e-15 x2, then 16 below 7e-17: 18 junk), but at a
     predictor the junk singular values smear upward (30.984: 6 of the 18 above 3e-14), land in the weak set, where the
     f64 Schur complement cannot resolve them, and the junk solve no longer moves them. NE_NJUNK=18: the junk solve
     takes the 18 smallest directions (the first NE solve keeps the cut, which pulls the smeared ones down).
  2. Predictor. The junk coordinates are not smooth in D (each point stops at a different place in the flat valley:
     30.324 and 30.584 are 0.1 apart in them), so the cubic predictor is 0.36 off there (linear: 0.16), and freezing all
     18 from that start leaves |F| at 0.04. NE_PRED_JUNK0=1: extrapolate the non-junk coordinates only, keep the junk
     coordinates of the last solution.
  3. Step control. Near the valley bottom most of the 18 x 18 difference Jacobian is noise: the Gauss-Newton steps
     were 0.2-0.6 long and the best of (1, 1/2, 1/4, 1/10) of them still raised the junk residual from 5.5e-15 to
     2-3e-14 (h 1e-2, 3e-3) or barely lowered it (h 1e-3: 5.36e-15). NE_JUNK_DAMP=1: Levenberg-damped steps,
     mu / sigma_max in (0, 1e-3, 1e-2, 0.1, 0.3, 1), evaluated in parallel. A Broyden update (local test at 30.584) did
     not help: the steps are short and an updated Jacobian does not predict the next one.
- With all three at D = 30.984: junk 6.7e-15 -> 5.0e-16 in one iteration (mu = 1e-3, |da| = 3.5e-3), 263 s for the
  whole point; Delta = 1.9553690245, +2e-8 from the D <= 30.2 series; gamma = 0.48128.
- Valley sensitivity: the local re-solve of 30.584 from junk 2.2e-15 to 8.9e-16 moved Delta by -1.75e-7 (to
  1.9591962887, -4.7e-8 from the series). So points must be driven to <= 1e-15, and older points left at ~2e-15 can be
  ~2e-7 up the valley.
- kc 31 at D = 29.326 (local T5, x_K 0.7 RK x3, junk stalled at 1.7e-14 under the old settings): -2.1e-8 from kc 23
  RK x4. The tau resolution is fine there.

## Forcing candidate constants as Delta_inf (2026-10-19, ~11:10)
- Search: r b1^p1 b2^p2 (r = i/j, i, j <= 12; p in +-1/2, +-1, +-2; pi, e, sqrt2, sqrt3, ln2, ln3, Euler gamma, zeta(3),
  golden ratio, Catalan, first Bessel zeros j01, j11, j21, j02, j12, y01, j_{3/2,1}, j_{5/2,1}), plus 2 pi / (r b) and
  r ln b: only 5 expressions land in 1.64388-1.64403, all contrived (7/5 ln2^-2 pi^-1/2 = 1.6440014,
  7/2 sqrt(ln2/pi) = 1.6440152, 9/8 sqrt(j21/j01) = 1.6440207, 5/7 sqrt(Catalan) j01 = 1.6439747,
  10/9 sqrt(j01/ln3) = 1.6439063).
- Forced as c0 in 12 fits (1/D, 1/(D-1), 1/(D-2); degree 9, 10; D >= 10, 12), chi2/dof min .. max and the rms offset of
  the held-out D = 34-39.5 points: free fit 0.62 .. 3.4, 1.4e-6; the five above 0.62 .. 9-118, 1.2-1.5e-6 (as good as
  free); pi^2/6 3.6 .. 13000, 1.9e-6; sqrt(e) 82 .. 3e5, 1.3e-5; 2 pi/j11 67 .. 3e5, 1.3e-5; golden ratio 2500 .. 1e7.
- So forcing a constant can exclude one (pi^2/6, sqrt e, 2 pi/j11, phi) but not select one: everything inside the window
  fits equally well, and the candidates differ at D <= 40 by less than the held-out points' own error (~2e-6). Telling
  them apart needs Delta_inf to ~1e-8, i.e. the D = infinity problem itself.

## One formula for all D >= 4 (2026-10-19, ~12:00-13:00)
- Old D < 8 values (data_best) were mixed: D = 4.1-4.75 at kc 31 (under-resolved by up to 8.4e-6 at 4.25), D = 6.75-7.5
  off by up to 3.8e-6, D = 4.0 at kc 39. Re-solved 8.71 -> 3.7 downward (scan_ne D_end < D_from), x_K 0.6, at kc 23, 31,
  39 (RK x1 and x2, Richardson). kc 23 -> 31 -> 39 at D = 4: -4.3e-4, -3.4e-5 (ratio ~13); above D = 5.5 the kc 31 -> 39
  step is <= 1e-8, above 6.5 <= 1e-10. kc 39 + Richardson joins the D >= 8 data (kc 23 + Richardson) to <= 4e-10 at
  D = 8-8.7. data_all.load_all: sigma at low D = the kc 31 -> 39 step / 13 (D = 4: 2.6e-6, 4.25: 6.5e-7, 5: 1e-8).
  kc 47 at D = 8.71 did not converge (strong floor 6.2e-9, Delta +8.4e-7): not resolution. kc 47 re-solves at 3.7-4.25
  (local T10) are pending.
- Delta(D) has its maximum ~3.4661 near D = 3.8 (kc 39 RK x1: 3.4661347 at 3.7938).
- rational_fit.compact on the clean data: Delta = P7(x)/Q2(x), x = 1/(D-2), 10 numbers, fitted on 4 <= D <= 30.2 with
  sigma floored at 1e-7: within the point precision everywhere (D = 4: 1.6e-6 vs 2.6e-6; 4.3-5: 4e-8; 5-30: <= 5e-8;
  held out 34-40: <= 1.3e-6), limit 1.643952, poles at D = -1.65, -9.37 (the 1/D series' pole near -9).
  gamma = P7/Q2 with P(0) = 1/2, 9 numbers: D = 4: 3.9e-7 (sigma 5e-7), 4.3-5: 4e-8, 5-30: <= 4.5e-9, held out to 35.8:
  <= 5e-7; above D ~ 36 the computed gamma scatter by up to 6e-5 around both this and the 1/D series (unreliable growth-
  rate solves there). formula_all.py; README "The formula"; delta.svg draws it.
- Covering D >= 3.7 (below the maximum) costs more (P3/Q6 at 1e-6, P8/Q2 at 1e-7 with 1.1e-6 at 3.7).

## The limit is conditional on analyticity in 1/D (2026-10-19, ~13:00)
- Forcing a candidate as the limit of the all-D rational formula with the denominator free: P8/Q2 cannot reach pi^2/6 or
  sqrt(e) without a pole at a physical D (sqrt e: +298), but P8/Q3 fits D = 4-30.2 AND the held-out D = 34-40 for pi^2/6
  (pole at D ~ -436; held-out chi2 0.3) and sqrt(e) (pole ~ -5854; 0.4); 2 pi/j11 (-588; 5.7) and phi (-5002; 3.9) are
  only mildly disfavored. A distant pole is a slow bend that starts at D in the hundreds.
- Non-analytic forms fit equally: series in n^(-1/2) of degree 14 / 16 / 18 on 10 <= D <= 30.2 have chi2/dof 0.66-0.71
  (1/D series 0.63), held-out chi2 1.0-1.4, limits 1.6908 / 1.6659 / 1.6075. Adding one n^(-1/2), log(n)/n or n^(-3/2)
  term to the 1/D series: chi2 unchanged, limit 1.6437-1.6448.
- All of these agree with the analytic formulas to 1e-5 up to D ~ 60 (2e-4 at D = 100, 0.02 at D = 1000).
- So Delta_inf = 1.64395 +- 0.00008 holds under analyticity in 1/D at D = infinity (natural: no other scale in the
  large-D expansion, only the D ~ -9 pole in the data; not proven: the period is selected beyond all orders, and EEG
  argue Delta -> 0). The candidate exclusions are conditional on it. plot_extrapolation.py / extrapolation.svg.
- Influence function of the forced-limit fits (power series): forced - free = (candidate - free limit) h(D), h ~ 1e-7 for
  D <= 22, ~1e-5 at 24-28 (error bars step up at 22 and 29.5), h(30) 7.5e-5, h(40) 3.8e-3, h(100) 0.17, h(1000) 0.85.

## Towards the D -> infinity problem (2026-10-19, ~13:00-14:30)
- Amplitude scaling (scratch/largeD_amp.py, centre value B = c q / x at x = 0.02 of the converged solutions): the first
  harmonic B_1 sqrt(n) = 2.688 (n = 10), 2.707 (20), 2.715 (28), 2.717 (38); B_3/B_1 ~ 0.8/sqrt(n), B_5/B_1 smaller
  still. So the critical solution tends to flat space as a weak, nearly sinusoidal oscillation with amplitude
  ~ 2.72/sqrt(n). Fits of B_1 sqrt(n) in 1/n or 1/sqrt(n) are equally good (limit 2.704-2.724); no decision on
  analyticity from this.
- Complex-loop (monodromy) formulation of horizon regularity (monodromy.py): continue the centre-regular solution
  around x = 1 on semicircles and compare. Fails: on a complex path the tau-harmonic k grows like exp(k omega arg),
  harmonic 23 overflows within a few degrees of arc. Not usable with Fourier in tau.
- Horizon series radius (mp_series.py, scratch/mp_test.py, 40 digits): at D = 10 the mp and f64 coefficients agree
  to 4 digits up to order 120, radius ~0.8: a real singularity of the horizon-regular solution, not rounding. At
  D = 30.1 f64 is contaminated above order ~20, but the mp coefficients also grow: radius ~0.58 (f64 suggested 0.47).
  So the radius shrinks with n (0.8 at n = 8, 0.58 at n = 28); its location is complex (the real continuation is
  smooth). Matching farther from the horizon therefore gains only modestly (u_K 0.5 vs 0.3: x1300 at n = 28).
- The real-axis "boundary layer in sqrt(n)(1 - x)" seen in layer_test.py is the outer O(1) variation, not a layer:
  in the inner scaling the p, m equations reduce to m = const, p = -m across u ~ 1/sqrt(n).
- Status: the period is selected by the horizon-singular branch u^(n/2), whose visibility at any usable matching
  point is exponentially small in n; direct numerics need exponentially small discretization errors (high-precision
  series and high-order integration in arbitrary precision). An analytic D = infinity theory would go through the
  complex singularity of the horizon-regular solution (radius ~0.6 at n = 28) and the beyond-all-orders selection.

## Double-double solver hp/ (2026-10-19, afternoon)
- New crate hp/ (hpkernel): DD arithmetic on 8-lane AVX-512 vectors, Taylor stepping in x (order 36, adaptive frozen
  steps, truncation 1e-32), centre / horizon series to 1e-33, pure collocation in tau on the half grid (no spectral
  truncation), exact Jacobian by forward-mode duals, DD LU (+ transpose solve: y = J^-T e_Delta = dDelta/dF).
- Checks: DD ops 1.3e-32; horizon series vs 40-digit mpmath 2e-32 (an apparent 1e-16 disagreement was mpmath's omega
  computed at 15 digits); Jacobian vs DD differences; D = 8 quadratic to |F| 5e-31. Delta(7) +3.1e-12 and Delta(7.75)
  -2.8e-12 vs the kc 39 references (+-3e-11). Delta(8) = 2.711790054444401: the kc 23 points from D = 8 up are
  2-4e-10 high (their 3e-11 error bars counted RK steps only). The formula P8/Q2 agrees with the DD values to
  <= 2.7e-9 on 8 <= D <= 21 (its stated tolerance 4e-9).
- tau resolution: Delta(M = 32) = Delta(48) = Delta(64) to 1e-16 at D = 7-17, although at M = 32 the top harmonic is
  still 1e-10 near the centre: errors there barely reach Delta (dDelta/dF ~ 1e-11 in the centre block, up to 6e4 at
  the matching blocks for k <= 15, 280 for k >= 24). M = 80 does not converge even at D = 8.
- The near-null ("junk") directions, now resolved: a ladder of p1 harmonics, one cos/sin pair per odd k, singular
  values ~ e^(-0.8 k) (D = 21.35: 1.3e-10 at k = 7 ... 6.5e-18 at k = 31; sigma_max ~ 2e3), independent of where the
  first node is (x0 = 0.1 or 0.3). The true p1 harmonics fall at a similar rate (2e-6 at k = 17, 7e-11 at k = 31), so
  p1_k reaches the matching point as ~e^(-1.45 k). Their Delta components are <= 1.5e-12 for k >= 17 but up to 0.04-0.7
  for k <= 9, where they mix with the physical selection direction. The centre-series map p1 -> state(x0) itself is
  well conditioned (40-digit SVD: 0.14 .. 4e-3).
- Why Newton's basin is tiny: second-order errors of the well-determined directions leak into the weak left singular
  directions and are amplified by 1/sigma ~ 1e17. A J-based tangent dz/dD is O(1) wrong in the junk directions
  (residual noise ~1e-28 x 1/sigma); differences of converged neighbours are smooth (dp1_31/dD ~ 3e-11).
- Fixes: (1) first node at x = 0.3 (the centre series' radius is ~0.47 at D = 21, 0.31 at D = 12, the same for every
  harmonic: a real singularity; ~200 terms); (2) kfreeze: p1 harmonics above 19 pinned to 0 and the horizon block's p
  matching above 19 dropped (Delta changes by 7e-18; above 15: 4.5e-15; above 11: 8.6e-12); (3) Lagrange predictor from
  converged neighbours; (4) deflated Newton: Newton steps with the 24 weakest right singular directions projected out
  (inverse subspace iteration with the stored LU) until |F| stalls, then plain Newton. Steps of 0.1 / 0.2 / 0.4 in D
  converge (6 / 8 / 18 iterations, 6-16 s at M = 32, L = 769) where plain Newton could not do 0.05.
- Selection sensitivity max |dDelta/dF|: 1.2e2 (D = 8.25), 2.2e2 (10), 5.9e2 (12), 2.1e3 (14), 6.8e3 (16), 2e5 (21):
  ~e^(0.57 D). With |F| ~ 1e-30 that leaves Delta at ~1e-6 at D = 100 if the rate holds.
- (evening) Freezing p1 above k = 19 (kfreeze 19) contaminated k = 15-19: pinning ~1e-7 harmonics at zero is a
  forcing that the next weak directions absorb with 1/sigma. Deflated Newton with the junk left free also drifts
  (junk fixed only to tol / sigma: sigma ~1e-20 at D = 24, 1e-24 at D = 35); junk held on the predictor's trend
  (junk_only + junk_hold, acceptance by the linearised Delta correction -y.F < 1e-14) reached D = 35.2, then the
  frozen-junk floor reached Delta. Moving the matching node inward (x = 0.5, 0.4, with inward segments from the
  horizon side) changes neither the weak spectrum nor dDelta/dF (both belong to the boundary-value problem, not to
  the shooting layout); Delta agrees to 1e-15.
- What works: kfreeze 13 at D = 35.2: Delta - unfrozen = -5.5e-15 (kfreeze 15: +1.1e-15, 11: +6.3e-13, 9: -1.2e-11),
  weakest sigma 1e-17 instead of 1e-24, every remaining weak direction has Delta content (>= 1e-8), plain Newton
  converges in 7 iterations to |F| 3e-27. Lower kfreeze as D grows to keep the weakest sigma >= ~1e-18.
- DD vs f64 data: D = 15.6-21: within 5.5e-11 (f64 sigma 3e-11); 22-29.3: <= 3.6e-9 (sigma 1.5-2e-9); 29.8: -1.1e-8,
  30.124: -1.8e-8 (sigma 3e-8); 34.18: -7.1e-8 (sigma 5.6e-7). All models fitted to the f64 data agree with the DD
  values to <= 9e-8 up to D = 36; they separate at the 1e-8 level (1/D deg 11 and n^-1/2 deg 18: 3-4e-9; n^-1/2 deg
  14/16: 8e-8; rational P8/Q2: 2.5e-8).
- dDelta/dF: 2e5 (D = 21), 6.2e8 (31.6), 2e10 (34.7), 3e10 (35.2): ~x3.5 per unit D above 30.

## Double-double climb to D = 45 and the large-D test (2026-10-19, ~16:00-18:45)

- Resolution at D = 40.02 (kfreeze 13, M = 32): M = 48 moves Delta by 9e-22; kfreeze 15 by 3e-17; kfreeze 11 does not
  converge (|F| 2e-9 after 20 iterations). At D = 45.33 the p1 harmonics are 9: 9e-5, 11: 1.4e-5, 13: 2.1e-6, so 13 is
  physical: kfreeze 11 stalls at |F| 2e-14. kfreeze 13 stays.
- Climb (up7, tol 1e-25): D = 35.35 -> 45.33, 30-60 s per point on two cores, |F| 2e-26 -> 1e-25, max dDelta/dF
  7e8 -> 1.5e11 (x1.7-1.8 per unit D), linearised Delta correction -y.F ~ 1e-21 .. 1e-20. At 45.3 |F| floored at
  1.2e-25 and the step halved; weakest sigma there is the harmonic-13 pair at 4.4e-20 (44.96: 5.2e-20), then 11:
  1.7e-18, 9: 1.1e-16, 7: 1.5e-14, with Delta components 3e-8, 5e-7, 9e-6, 1e-4. The LU's smallest pivot dropped from
  7e-5 to 1e-7 there without any change in the spectrum (pivoting artefact). Continued as up8 with tol 1e-24.
- Data quality: the DD set fits smooth curves to 9e-15, not 1e-20. 25 points between 27 and 35.4 (the unfrozen up6 run,
  accepted by the Delta-correction rule) sit 1-9e-15 off their neighbours (dd_fits_mp.py --clean drops them); the
  kfreeze-13 points 35.35-37 carry the cut's truncation, 4.6e-15 at 35.35 decaying to ~0 by 37. After cleaning, fits
  reach 3.5-4e-15.
- Fits (dd_fits_mp.py, Chebyshev basis at 60 digits), fitted on 10-37.5 (cleaned), error at the held-out D = 45.1:
  1/D deg 16 +1.0e-12, deg 18 +2.5e-12, 1/n deg 18 +1.6e-12, n^-1/2 deg 20 +6.7e-12, D^-1/2 deg 20 -4.9e-12. Half-integer
  series limits are unstable (deg 20: 1.636 / 1.215; deg 24+: -100 and worse). Predictions D = 50: all within 1.3e-10
  (1.84165950301); D = 60: 1/D 1.80988946002-1.80988946011, n^-1/2 +5e-10, D^-1/2 -5e-9.
- All cleaned points 10-45.3: 1/D and 1/n deg 16-20 limits 1.64389375-1.64389393 -> Delta_inf = 1.6438938 +- 2e-7
  (omega_inf 3.8221358 +- 5e-7) if analytic. The f64 estimate was 1.64395 +- 8e-5; the rational formula's a0 = 1.643898.
  Forced limits (1/D series fitted on 10-37.5, degree up to 24), held-out max to 45.3: free 1.0e-12; 10/9 sqrt(j01/ln3)
  8.4e-12; 5/7 sqrt(G) j01 1.8e-11; pi^2/6 1.5e-10. With all data, forced fits reach the residual floor only at degree
  22-24, so the exclusion rests on prediction, not on fit quality.
- Formula P8/Q2 vs DD: 2.7e-9 (8-21), 8.1e-9 (21.5-30.2), 2.4e-8 (30.2-35), 4.2e-8 (35-40), 5.0e-8 (40-45).

## Double-double climb to D = 60 (2026-10-19, ~18:45-23:05)

- Runs up8-up13 (cloud): 45.33 -> 60.06, M = 32, kfreeze 13, tol 1e-24 -> 1e-23 (51.26) -> 1e-22 (55.66), raised each
  time the |F| floor (x2 per unit D: 1e-25 at 45, 1e-24 at 51, 1e-23 at 57, 2e-23 at 60) reached the tolerance.
  Steps 0.015-0.05, 30-50 s per point on two cores, 17 basin failures (step too long, |F| 1e-10..1e-16, halved) in
  51-60. Two container restarts (53.74, 56.30) and the 3 h timeouts cost only the point in flight.
- max dDelta/dF 1.5e12 (50), 1.2e13 (55), 8.9e13 (60); dDelta_est (-y.F) 1e-20 .. 2e-18. All 750 points in
  hp/data/delta_dd.csv; fits stay at 4e-15 to D = 60.
- Held-out (fit on cleaned 10-45, worst on 45-60.0): 1/D16 5.1e-13, 1/D18 1.8e-12, 1/n18 2.7e-12, n^-1/2 20 1.9e-11,
  D^-1/2 20 1.5e-10. Fit 10-37.5, worst to 60: 4.1e-11, 1.5e-10, 7.8e-11, 3.9e-10, 6.1e-9. Mixed (1/D16 + one term,
  fit 10-45): held-out 1.6-5e-13, limit 1.64389375 for every term, D^-1/2 coefficient 3e-5 (deg 14) / 1e-7 (deg 16).
- All points 10-60: 1/D and 1/n degree 16-20 limits 1.643893744-1.643893752 -> Delta_inf = 1.64389375 +- 1e-8,
  omega_inf = 3.82213589 +- 2e-8. Delta(60) = 1.809889459975, predicted from data <= 56.3 as 1.8098894599746.
  Extrapolations: Delta(70) = 1.78693476494 (all models within 1e-11), Delta(100) = 1.7450264844 (2e-8 incl.
  half-integer series), Delta(1000) = 1.65424086 (analytic; half-integer 1.6542 .. 1.671).
- Candidates (hp/dd_candidates.py, forced 1/D series on 10-45, degree 14-26, held out to 59.8): best 5.4e-11
  (10/9 sqrt(j01/ln3)) .. 4.4e-8 (golden ratio) vs free 4.7e-13: all ruled out, 110x-92000x. The closed-form search
  (solver/candidates.py) finds nothing in 1.6438936-1.6438939 nor in 1.643890-1.643898.
- Formula P8/Q2 vs DD: <= 5.8e-8 on 30.2-60 (never fitted there).
- Handed the climb past 60 to the local machine (comms T12, seeds in hp/data/seeds/), cloud run stopped at 60.06.

## Towards D = infinity: the 1/n expansion is regular, the period is selected beyond all orders (2026-10-20 morning)

- Scaling from the DD solutions D = 12-60 (hp/largeD_scaling.py, hp/largeD_F.py), n = D - 2: p ~ x F(theta)/sqrt(n)
  (sqrt(n) dp/dx at x = 0: |F_1| 3.48 (D=20) .. 2.96 (D=60), ~2.73 extrapolated; F3/F1 0.130 .. 0.080, F5/F1 0.024 ..
  0.0085), p + m = O(n^-3/2) (n(p+m) slowly falling), c = 1 - O(1/n) with n(1 - <c(0)>) -> <F^2> (3.42 vs 4.42 at
  D = 60, both extrapolating to ~3.65), a^2 - 1 = (p^2 + m^2)/(n-1) to 3%, x - c stays at x - 1 to 1e-3. Exact
  symmetry of the centre series: m(x) = p(-x).
- Formal expansion (solver/largeD_expand.py, sympy, eps = 1/n, p = eps^1/2 P, m = eps^1/2 M, c = 1 - sum eps^k G_k,
  a^2 = 1 + sum eps^k B_k): P0 = x F(theta), M0 = -P0; B2 = 2 x^2 F^2; G1 = F^2 (1 - x^2); S1 = P1 + M1 =
  2 x^2 (F + w0 F'); D1 = P1 - M1 = x^3 (w0^2 F'' + 3 w0 F' + 2F - 2F^3) + x K1(theta); and so on. Through order 4
  every field is a polynomial in x, regular at the centre (no x log x) and at the light cone, for ANY periodic F
  and any w: no solvability condition appears. The period (and F) are selected by terms exponentially small in n,
  e.g. the WKB modes ((1-x)/x)^(n/2) of the p equation (the same thinning seen in the weak directions).
- Numerical check (solver/largeD_formal_check.py): the order-2 formal solution (plus S_3, G_3, B_4) for the D = 60
  F(theta), K_k = 0, plugged into the exact equations, leaves relative residuals falling exactly as n^-3 in all four
  equations, for omega = omega_inf and for a wrong omega (Delta = 1.4) alike: the family is free in (F, omega).
- Late orders (solver/largeD_late_orders.py: the same recursion numerically, x-polynomials with theta-grid
  coefficients, roundoff harmonics projected out; matches sympy at orders 1-2 to 1e-11): the series diverges
  factorially, |D_k| ratio / k ~ 7-8.6 at orders 10-30 for F = 2.72 sin(theta), omega = omega_inf; the rate depends on
  the amplitude A and omega (A = 1.36: 5.3; A = 5.44: 10.6; omega/2: 2.7; 2 omega: 20.8; omega = 0: 0.94; A^2/(rate)
  = 1.005 at the physical point, possibly a coincidence). The late terms are dominated by ever-higher harmonics
  (h3/h1 ~ 20-70); the harmonic-1 ratio/k ~ -3 + 1.5i (a complex pair of singulants with Re chi < 0).
- Weak singular values of the DD Jacobian by p1 harmonic (D = 20..60): k = 7, 9, 11, 13 fall like n^-10.5, n^-14,
  n^-17.5, n^-21 (~ n^-1.6k); k = 1, 3 mix with other directions. The decay rate of F's harmonics grows like ~0.25 ln n
  (0.58 at D = 12 .. 1.02 at D = 60, local rate at k ~ 9-11), with no collapse in k/n.

## The fast mode with gravity and the Stokes picture (2026-10-20, 08:10-09:30)

- Weakly nonlinear test (scratch cubic.py: C_lin + |A|^2 K = 0 by monodromy) abandoned: mpmath odefun far too slow,
  the monodromy basis needs six branch factors plus a log at lambda+ (non-integer n), and the amplitude is not small
  (centre slope ~0.64 at D = 20, <F^2> ~ 3.7 at D = infinity). Proposition 3 explains why it cannot work.
- Proposition 3 (README): fast perturbations e^{-n Phi} A, Phi = ln x - ln(1-x^2)/2, delta m/delta p = (1-x)/(1+x);
  transport x(1-x^2) A_x - 2 w x^2 A_th = (1 + x - 2 x^2 F^2) A from the order-1 solvability (left null vector
  (1+x, -(1-x))). b, c perturbations and G dp are O(1/n). largeD_fast_transport.py (x 0.9 -> 0.72, 64 theta points):
  error n x 6.0 (no gravity), n x 49-57 (with gravity), i.e. 1/n; gravity factor ~40 over the interval. Light-cone
  exponent shifts by <F^2>, consistent with the exact local exponent (n/2 - 1 - s + <G(1)>)/(1 - <G(1)>).
- Linear late orders (largeD_linear_late.py, 90 digits, order 122): consecutive ratios give chi = Phi(x) - i pi/2 to
  3e-3..7e-3 at x = 0.71-0.9 (orders 100-120), g ~ -0.5 - 4.3i (predicted -(s + 1/2) = -0.5 - 3.82i, Im drifting);
  x < 0.7: interference with Phi + i pi/2. Singulant born at x = infinity; Stokes line = real axis x > 1 from above
  (chi = ln(x/sqrt(x^2-1)) real), ends at the light cone: C(n, s) is the Stokes multiplier. On the Stokes line the
  characteristics sit at Im theta = -pi w ~ -12. Inner region x ~ sqrt(n): gravity O(1), fast/slow not separated.
- New local points (T13, up22-23) to D = 61.006: 23 points > 60.95 agree with the 1/D deg-18 fit on <= 60.1 to
  2.6e-15 (deg 16: 7e-15, 1/n 18: 1e-14).
- Two forks launched on (A) the inner problem at x ~ sqrt(n) and (B) independent checks of Proposition 3 / the Stokes
  claims and the nonlinear late-order singulant.
- Fork A (inner problem): xi = x/sqrt(n) gives I1-I3 (README), n-free; the inner series reproduces the outer top-degree
  coefficients to 2e-13 (orders 1-10). Linear: 2F0(a, b; ; 2 xi^2); C(n,s) = -2 pi i 2^(-s-1/2)/(Gamma(a)Gamma(b))
  (n+1)^(s+1/2) e^(-i pi lambda)(1 + O(s^2/n)) (ratio 1.00014 at n = 1e5); Dingle constant of the linear late orders
  K = e^(-i pi(1+s)) 2^(-s-1/2)/(Gamma(a)Gamma(b)) verified to 6e-4 (x = 0.9), 1e-3 (0.95) after Richardson; g = -(s+1/2)
  exact. Im theta = -pi omega remark withdrawn (branch bookkeeping). With gravity: inner series growth dominated by high
  harmonics (ratio/j 6.8 -> 4.2), harmonic-1 fast constant below double-precision floor; one-harmonic Galerkin hints at
  +-omega mixing. Fork A continues: direct multiplier with gravity (lateral solutions / Borel-Pade).
- Fork B (checks): Prop 3 transport re-derived in sympy with db, dc (enter at eps^3/2) - exact; full-background check
  n*err 62 -> 51.7. Local exponent formula verified (c_x(1) = G(1) to 1.7e-14 on DD states); n<G(1)>/2 = 2.04..2.96 vs
  <F^2> 6.16..4.42 at D = 20..60, both extrapolate to ~3.70. Proposition 4 (README): exact pole sum of the linear late
  orders (1e-49..1e-91), Poisson summation gives the singulant tower Phi + i pi (l + 1/2) and g = -(s+1/2); weights
  1 : 1.6e-4 : 1e-14 at x = 0.5. Truncated series vs exact inside x < 1: no fast mode (4e-40 vs 7e-21 at x = 0.9, n = 60).
  Nonlinear late orders: rate x-independent (7-8.5 at x = 0.25..1), dominant harmonic ~0.55 k: free-data divergence, not
  a heat-type (x-dependent) singulant - that hypothesis refuted.

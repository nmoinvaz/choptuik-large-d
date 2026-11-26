# Echoing period Δ of the Choptuik critical solution, D = 6 … 200

STATUS (14:40, hour 1.7 of 12): PRELIMINARY.  Only D = 6…11 have values, from a
low-resolution run (N = 32 τ-points per half period); their uncertainty (5e-5) is an
estimate from the N=16 → N=32 change and from the D=4 validation, not yet a full
convergence test.  This file is rewritten as work proceeds.

## Equations (derived here)

Metric ds² = −α²dt² + a²dr² + r²dΩ_n², n = D−2, massless scalar ψ (rescaled so that
κ/n is absorbed).  With U = r(ψ_r + aψ_t/α), V = r(ψ_r − aψ_t/α), c = α/a, f = a²:

    r f_r = −(n−1) f (f−1) + f (U²+V²)/2
    r c_r = (n−1)(f−1) c
    U_t = c[U_r + ((n−1)(f−1) + n/2 − 1) U/r + n V/(2r)]
    V_t = −c[V_r + ((n−1)(f−1) + n/2 − 1) V/r + n U/(2r)]

Self-similar coordinates τ = −ln(−t), x = −r/t, time labelling fixed so that the
self-similarity horizon (past light cone of the accumulation point) is x = 1, i.e.
c(τ,1) = 1.  The critical solution is a solution periodic in τ with period Δ
(f, c have period Δ/2; u = U/x, v = V/x are antiperiodic over Δ/2), regular at the
centre x = 0 (f = 1, u + v = 0) and analytic at x = 1.  This is a nonlinear
eigenvalue problem for Δ.

## Solver

* τ: Fourier pseudo-spectral, N points on a half period (symmetry built in).
* x ∈ [0, x_m]: piecewise-polynomial collocation (degree p, Gauss points).
* x ∈ [x_m, 1]: power series in (1−x) to order K about the horizon, generated
  recursively from f(τ,1), v(τ,1); this imposes analyticity at the horizon exactly.
* Newton iteration on all unknowns + Δ, sparse direct solves.
* Initial guess at D = 4 from my own near-critical time evolution (amplitude bisection
  to 1e-13), then continuation in the dimension parameter n.

## Validation at D = 4 (not requested; pipeline check only)

Δ(N=32) = 3.445494, Δ(N=48) = 3.4454526, Δ(N=64) = 3.445452405, Δ(N=96) = 3.445452402;
independent of x-resolution, matching point and series order to 1e-11.

## Values so far

D = 6…11: N = 32, 10–12 elements of degree 8.  Uncertainty 5e-5 (provisional).
D ≥ 12: not yet computed (continuation is being made robust).

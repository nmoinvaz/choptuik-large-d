# The σ-model's echoing (DSS) critical solution in D dimensions

DRAFT — results pending.

## 2. Equations

The model, the coupling η = 4πGf² and the self-similar equations are those of [SIGMA.md §2](SIGMA.md#2-equations)
(checked there by 48 sympy checks in [`sigma_eqs.py`](../solver/matter/sigma_eqs.py)). In the first-order form of
[`solver/dss.py`](../solver/dss.py), with the field normalised as the main solver's scalar,

```math
\psi = \sqrt{\eta/n}\;u, \qquad p, m = \sqrt{\eta/n}\;r(\Phi\pm\Pi), \qquad \kappa^2 = \frac{n}{\eta},
\qquad \sigma(\psi) = \frac{\sin\kappa\psi}{\kappa}, \qquad g(\psi) = \cos\kappa\psi ,
```

the equations are (η sin²u = nσ², and √(ηn) sin 2u/2 = nσg)

```math
x a_x = \tfrac a2\big[p^2+m^2 + 2na^2\sigma^2 + (n-1)(1-a^2)\big], \qquad
x c_x = c\big[(n-1)(a^2-1) - 2na^2\sigma^2\big], \qquad x\psi_x = \tfrac12(p+m),
```
```math
p_\tau + (x-c)\,p_x = c_xp - \frac{cp}{x} + \frac{nc\,(p+m)}{2x} - \frac{nca^2\sigma g}{x}, \qquad
m_\tau + (x+c)\,m_x = -c_xm + \frac{cm}{x} - \frac{nc\,(p+m)}{2x} + \frac{nca^2\sigma g}{x} .
```

At κ = 0 (η = ∞ at fixed ψ) σ = ψ and g = 1, and these are exactly the ℓ = 1 multiplet's equations with
L = ℓ(ℓ + n − 1) = n: the hedgehog becomes a multiplet of D − 1 free scalars. So 1/η = κ²/n is a homotopy parameter
from the ℓ = 1 critical solution to the σ-model's, at every D.

**Boundary conditions.** As for ℓ = 1: at the centre ψ ≈ q(τ)x and p ≈ m ≈ q(τ)x (u ≈ κq(τ)x, the hedgehog's
regularity condition u = O(x)), with free data q(τ) and c₀(τ) = c(τ, 0); at the self-similarity horizon x = 1 (c = 1)
free data m, a, ψ and p from regularity. The equations are invariant under (p, m, ψ) → −(p, m, ψ), i.e. u → −u, and the
solution is sought with p, m, ψ odd and a, c even under τ → τ + Δ/2 (checked in §4).

**Series.** The Taylor series at both ends are generated order by order as in the ℓ-multiplet solver, with the series of
σ(ψ) and g(ψ) built from dσ = g dψ, dg = −κ²σ dψ. At the centre σ₀ = 0, so the order-k coefficient of σg enters the
order-k system for (P_k, M_k) with the same coefficient as ψ_k does for ℓ = 1; the matrix of that system is unchanged.

## 3. Method

[`sigdss_solver.py`](../solver/matter/sigdss_solver.py), class `SigDSS`, is a copy of the ℓ-multiplet agent's
`DSSEll` (worktree `agent-a5bec5ffdf3db4330`, `solver/matter/dss_ell.py`, itself built on this repository's
[`solver/spec.py`](../solver/spec.py)) with the σ-model terms of §2. Nothing is imported from that worktree. As there:
dealiased Fourier series in τ up to harmonic k_c (N ≥ 3k_c points), parities imposed, multiple shooting in x from the
centre series (order 9) and the horizon series (order 16, evaluated at u = 1 − x = 0.01) through 12 nodes
0.02 … 0.9 with RK4, step counts frozen from the stiffness k_cω/|x ∓ c| of the solution (target κ_RK = 0.5 at D = 4 and
0.7 above), phase fixed by Im q̂₁ = 0, every derivative (Jacobian, growth-mode matrix, ∂F/∂n, ∂F/∂η⁻¹) by complex step,
Levenberg–Marquardt with one SVD per Jacobian. Changes:

- **The σ-model terms**, with σ(ψ) and g(ψ) evaluated as entire functions of κ²ψ² (Taylor series for |κ²ψ²| below 0.5)
  so that complex steps in η⁻¹ and n stay exact at η⁻¹ = 0. Checks: at η = ∞ the residual of the ℓ-agent's D = 4,
  k_c = 23 solution is 1.57×10⁻¹⁵, as in their code; ∂F/∂η⁻¹ and ∂F/∂n against central differences agree to
  2–9×10⁻⁹ (the difference error); Jacobian columns against central differences to 2–5×10⁻¹⁰.
- **Speed.** NumPy has no BLAS path for a complex array times a real matrix (it is about 20 times slower). Splitting the
  complex-step arrays into real and imaginary parts before each spectral matrix product makes the complex-step residual
  11 times faster; the Jacobian and the mode matrix gain about as much. (The ℓ-multiplet code has the same bottleneck.)
- **Continuation** in η⁻¹ at fixed D and in D at fixed η (tangent predictor from the last Newton Jacobian, quadratic
  correction from the previous point, adaptive steps landing on target values; [`sigdss_run.py`](../solver/matter/sigdss_run.py)).
- **Growth rate** λ of the unstable mode: real root of det M(λ), with M the linearised matching map for perturbations
  e^{λτ} × (periodic, same parity). |det M| varies exponentially in λ, so a secant on det crawls; the root is taken
  from the model log|det M| = log|λ − λ₀| + a + bλ through the last three evaluations (5–7 evaluations from a guess
  within a few per cent; a sign-change bracket with Brent's method as fallback). The guess is extrapolated in γ = 1/λ.
- **Diagnostics** per point: |F|, the ψ-constraint K = ψ_τ + (p + m)/2 − c(p − m)/(2x) relative to max|p| (zero for an
  exact solution, not imposed: it measures τ-truncation and integration error), the gauge rate
  λ_g = mean(1 − c_x(τ, 1)), and max|u| over the nodes.

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

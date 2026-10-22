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
L = ℓ(ℓ + n − 1) = n ([ELL.md §1] of the ℓ-multiplet agent; STRUCTURE.md §2): the hedgehog becomes a multiplet of D − 1
free scalars. So η⁻¹ = κ²/n is a homotopy parameter from the ℓ = 1 critical solution to the σ-model's, at every D.

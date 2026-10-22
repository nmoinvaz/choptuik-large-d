# The 1/n coefficients of ω and γ from the inner problem

This note derives the first-order large-D coefficients of the echo frequency and of the growth rate from the O(1/n)
corrections to the D = ∞ inner problem, and compares them with the coefficients fitted to the double-double solutions.
It was written by Claude (Anthropic) under Nathan Moinvaziri's direction and has not been checked by a mathematician.
Notation as in the [README](../README.md#large-d-what-the-1d-expansion-can-and-cannot-see): n = D − 2, ξ = x/√n,
y = p − m, σ = √n (p + m), b = 1 + η/n, ω = 2π/Δ, λ = 1/γ, F(θ) the profile and M the number of harmonics kept.
Scripts are in [`solver/inner_pert/`](../solver/inner_pert/), all named `first_order_*`.

Contents

- [Summary](#summary)
- [1. The inner equations to O(1/n)](#1-the-inner-equations-to-o1n)
- [2. The O(1/n) inner solution and its matching data](#2-the-o1n-inner-solution-and-its-matching-data)
- [3. The selection at O(1/n), and what the outer region contributes](#3-the-selection-at-o1n-and-what-the-outer-region-contributes)
- [4. The growth rate at O(1/n)](#4-the-growth-rate-at-o1n)
- [5. Numerical method and checks](#5-numerical-method-and-checks)
- [6. Results against the fits](#6-results-against-the-fits)
- [7. What is rigorous, what is numerical, what is truncated](#7-what-is-rigorous-what-is-numerical-what-is-truncated)
- [8. Scripts](#8-scripts)

## Summary

RESULTS_TABLE_PLACEHOLDER

## 1. The inner equations to O(1/n)

Write the exact self-similar equations (README, "Large D") in the inner variables and put ν = n^{−1/2}. Lemma B5.0 of
[proof/C_selection.md](../proof/C_selection.md#9-new-second-pass-lemma-b50-the-inner-limit-is-reached-in-powers-of-1n-proved)
(script `solver/proof/inner_corrections.py`) shows that only even powers of ν occur. The b equation is singularly
perturbed (its O(1/n) part contains $`\xi\eta_\xi`$, the b fast mode $`\eta \sim x^{-n}`$, which is exponentially small for
|x| > 1), so η is slaved:

```math
\eta = \tfrac12 y^2 + \frac1n\Big(\tfrac12 y^2 + \tfrac12\sigma^2 - \xi y y_\xi\Big) + O(n^{-2}) .
```

Substituting it into the other three equations and keeping O(1/n) gives, with $`L = \xi^2\partial_\xi + \xi\,\partial_\tau`$
($`\partial_\tau = \omega\partial_\theta`$ on a discretely self-similar solution, $`\lambda + \omega\partial_\theta`$ on a perturbation
$`e^{\lambda\tau}`$), $`E = \tfrac12 y^2`$ and $`\Psi = \tfrac12\sigma^2 - \xi y y_\xi`$:

```math
L y = c\,\sigma - \frac1n\,c\,\big(\sigma - \xi\sigma_\xi - E\sigma\big), \qquad
L\sigma = c\,\xi y_\xi + \Big(cE + \frac1n\,c\Psi\Big)\,y - c\,y, \qquad
\xi c_\xi = cE + \frac1n\,c\Psi .
```

At 1/n = 0 this is the inner problem of the README. Two facts simplify it:

- Given the c equation, the σ equation is $`L\sigma = (\xi\partial_\xi - 1)(c\,y)`$ exactly, including the O(1/n) terms: the
  correction to the σ equation is entirely the correction of $`\xi c_\xi`$.
- In the physical inner variables of Proposition 5 (ρ = r/√n, t; s = ρ²/2, $`y = \sqrt s\,Y`$, $`\sigma = \sqrt2\,s\,S`$; $`L = \rho\,\partial_t`$)
  the system becomes the heat system of Proposition 5 plus explicit 1/n terms:

```math
Y_t = cS + \frac1n\,c\,\Big(S + 2sS_s + \tfrac12 sY^2 S\Big), \qquad
S_t = (cY)_s, \qquad
c_s = c\,\Big(\tfrac14 Y^2 + \frac1n\big(\tfrac12 sS^2 - \tfrac12 Y^2 - sYY_s\big)\Big).
```

The O(1) part contains no explicit s; the O(1/n) part does (the flat-space part is the radial Laplacian of
$`\mathbb R^{n+1}`$ in s = r²/(2n), $`\partial_s + \tfrac1n(\partial_s + 2s\partial_s^2)`$). So the 1/n terms break the
s-translation that gives the λ = 2 mode, and nothing else: the time reparametrization t → T(t), c → T′c remains an
exact symmetry of the corrected equations (it is one of the exact equations).

*Checks* (`first_order_equations.py`, sympy): the three equations above are the exact equations to O(1/n) with no odd
powers of ν; the heat form; the identity $`L\sigma = (\xi\partial_\xi - 1)(cy)`$; the s-translation breaking term of
§4; the exactness of the time reparametrization. All hold identically.

## 2. The O(1/n) inner solution and its matching data

Write $`Z = (y, \sigma, c) = Z_0 + Z_1/n + \dots`$ and $`\omega = \omega_* + \omega_1/n + \dots`$, with $`Z_0`$ the D = ∞ inner
solution (a zero of the multipliers). $`Z_1`$ solves the inner problem linearized about $`Z_0`$, forced by the O(1/n) terms
and by $`\omega_1\xi\partial_\theta Z_0`$. Its formal power series ($`y_1 = \sum_j a_j\xi^{2j+1}`$, $`\sigma_1 = \sum_j b_j\xi^{2j+2}`$,
$`c_1 = \sum_j h_j\xi^{2j}`$, the same structure as $`Z_0`$) is fixed by three pieces of data:

- **$`K_1`$**, the 1/n correction of the profile, $`F_n = F + K_1/n + \dots`$: $`a_0 = K_1`$ (Proposition 1's free functions $`K_k`$; only
  $`K_1`$ reaches this order of the inner problem);
- **$`\omega_1`$**;
- **$`h_0 = c_1(0) = -F^2`$** (on the model's harmonics: $`-g_1`$, the ξ² coefficient of $`c_0`$). This is the light-cone gauge: the outer
  normalization c(x = 1) = 1 is $`c(\xi = n^{-1/2}) = 1`$ in inner variables, i.e. $`1 + g_1/n + c_1(0)/n + O(n^{-2}) = 1`$. It is
  the only datum that comes from outside the inner region at this order.

**Check** (`first_order_checks.py outer`). The outer 1/n expansion of Proposition 1, re-expanded at $`x = \sqrt n\,\xi`$, has
its top-degree coefficients in $`Z_0`$ (README) and its next-to-top coefficients ($`D_k`$ at $`x^{2k-1}`$, $`S_k`$ at
$`x^{2k-2}`$, $`G_k`$ at $`x^{2k-2}`$) in $`Z_1`$. With F = 2.72 sin θ, ω = ω∞ and $`K_k = 0`$, $`\omega_k = 0`$
(`solver/largeD_late_orders.py`, orders 1–9), the recursion for $`Z_1`$ with $`K_1 = 0`$, $`\omega_1 = 0`$, $`h_0 = -F^2`$ reproduces
them to 1×10⁻¹³ relative at every order (y, σ and c). This checks the O(1/n) equations, the slaving of η and the gauge
datum at once.

## 3. The selection at O(1/n), and what the outer region contributes

**The condition.** Conjecture 1 (README) selects $`(F, \omega)`$ at n = ∞ by the vanishing of the fast-mode Stokes
multipliers S of the inner solution. At finite n the same chain of lemmas (C_selection.md §6, §10) relates the
light-cone coefficient to the multiplier of the exact inner solution, whose expansion is

```math
S(Z_n) = S(Z_0) + \frac1n\,S^{(1)} + O(n^{-2}), \qquad S^{(1)} = \text{multiplier of } Z_1 \quad (\text{when } S(Z_0) = 0) .
```

$`Z_1`$ is linear in its data, so with $`x = (A, \omega/\omega_\infty, B_3, C_3, \dots)`$ the parameters of $`F = A\sin\theta + \sum_\nu (B_\nu\sin\nu\theta + C_\nu\cos\nu\theta)`$
(phase fixed by $`F_1`$ pure sine) and $`\delta x`$ their 1/n corrections ($`K_1`$ and $`\omega_1`$),

```math
S^{(1)} = S_{1/n} + \frac{\partial S}{\partial x}\,\delta x = 0
\quad\Longrightarrow\quad
\delta x = -\Big(\frac{\partial S}{\partial x}\Big)^{-1} S_{1/n}, \qquad \omega_1 = \omega_\infty\,\delta x_2 ,
```

where $`S_{1/n}`$ is the multiplier induced by the O(1/n) terms of §1 with $`K_1 = 0`$, $`\omega_1 = 0`$ and the gauge datum
$`h_0 = -F^2`$. This is "$`\omega_1 = -S_{1/n}/\partial_\omega S`$ with $`A_1`$ eliminated": by Cramer's rule, $`\omega_1`$ is
$`-S_{1/n}`$ projected on the left null direction of the other columns of $`\partial S/\partial x`$, divided by
$`\partial_\omega S`$ projected the same way.

**The outer region contributes nothing else at O(1/n).** The chain from the inner multiplier to the light-cone coefficient
$`C_{n,\mu}`$ is a product of linear maps on the label space (vectors over harmonics):

- the matching factor $`\kappa_{n,\mu}`$ of Lemma B5.1 (powers of n, the half-monodromy phases $`e^{\mp in\pi/2}`$,
  $`e^{\mp i\pi\langle F^2\rangle}`$, $`e^{\mp\pi\mu\omega}`$, and corrections in $`1/(n u)`$ and $`1/(n u^2)`$ that describe the shape of the fast
  mode, not its label);
- the transport along the characteristics on 0 \< x \< 1 (Proposition 3; never zero);
- the light-cone matching constant (Lemma B6) and the factor $`e^{-i\pi\Lambda_\mu} - e^{i\pi\Lambda_\mu}`$, with the light-cone
  exponent $`\Lambda_\mu = n/2 - 1 - i\mu\omega + \langle F^2\rangle + O(1/n)`$. Its gravitational shift $`\langle F^2\rangle`$ and its O(1/n)
  correction enter only here, and $`\sin\pi\Lambda_\mu \ne 0`$ because $`\mathrm{Im}\,\Lambda_\mu \ne 0`$.

Each map is invertible (non-resonance, B6 step 3), and each has a 1/n expansion. At a zero of $`S(Z_0)`$ their 1/n
corrections multiply $`S(Z_0) = 0`$, so $`C_n = 0`$ to O(1/n) exactly when $`S^{(1)} = 0`$: the shift of the light-cone
exponent changes the size of the light-cone coefficient, not where it vanishes. The outer region's only input at
this order is the matching data of §2: $`K_1`$ and $`\omega_1`$ (the unknowns) and the gauge datum $`h_0`$. In the full problem
even $`h_0`$ drops out: a θ-dependent change of c at ξ = 0 is a time reparametrization (§1), which maps solutions with
zero jump to solutions with zero jump, so the selected $`\omega_1`$ does not depend on it. In the truncated models it does
slightly, and that dependence is a measure of the truncation (§6).

*Status of this argument.* It uses the lemma chain of C_selection.md at O(1/n): that the outer solution equals the inner
lateral sum including $`Z_1`$ up to $`O(n^{-2+\epsilon})`$ ("the 1/n refinement" of Lemma B5(2)), which is written there at
appendix level, and Conjecture 1 itself.

## 4. The growth rate at O(1/n)

A perturbation $`e^{\lambda\tau}\delta Z`$ obeys the equations of §1 linearized about $`Z_n`$, with $`\partial_\tau \to \lambda + \omega\partial_\theta`$
(the O(1/n) terms contain no τ-derivative). A growth rate is a λ for which a real profile $`f`$ (δy = 2fξ + …) gives a
perturbation with zero multiplier, with the gauge δc(x = 1) = 0, i.e. $`\delta c_0(0) = 0`$ and $`\delta c_1(0) = -h_1`$ ($`h_1`$ the ξ²
coefficient of $`\delta c_0`$). Let $`\mathcal S(\lambda)`$ be the real 2N × 2N map from the profile to the multiplier at n = ∞, $`f_*`$ its
null vector at $`\lambda_*`$ and $`\ell`$ the left null vector. Then, at first order in perturbation theory in 1/n,

```math
\lambda = \lambda_* + \frac{\lambda_1}{n} + \dots, \qquad
\lambda_1 = -\,\frac{\ell\cdot S_{1/n}}{\ell\cdot\partial_\lambda\mathcal S(\lambda_*)\,f_*} ,
```

where $`S_{1/n}`$ is the multiplier of the O(1/n) part of the perturbation. Two ways to get it:

**Direct route.** The O(1/n) part of the perturbation solves the perturbation equations at $`\lambda_*`$ forced by the O(1/n)
terms linearized about $`Z_0`$ and by the second variation of the inner equations in the directions $`Z_1`$ and $`\delta Z_0`$
(including $`\omega_1\xi\partial_\theta\delta Z_0`$). It needs $`Z_1`$, i.e. the selected $`K_1`$ and $`\omega_1`$ of §3. In the jets of §5 this
is the mixed component $`\epsilon_e\epsilon_n`$ of a hyper-dual number; nothing has to be derived by hand. In a truncated
model it gives the model's exact first-order coefficient, about the model's own $`\lambda_*`$.

**Symmetry route (the formulation of GAMMA.md §6).** At n = ∞, $`\lambda_* = 2`$ and $`\delta Z_0 = \mathcal D Z_0`$ (Proposition 5:
s-translation plus a time reparametrization, $`P + \omega P' = -2F^2`$). Write the finite-n mode as $`\mathcal D Z_n + W/n`$. Because the
O(1) equations are invariant under s-translation and the reparametrization is exact, $`\mathcal D Z_n`$ fails to solve the
finite-n perturbation equations at λ = 2 only through the explicit s of the O(1/n) terms
(`first_order_equations.py`, section 3): W solves the n = ∞ perturbation equations at λ = 2 with the forcing

```math
L_2 W_y = \dots - \frac{2}{\xi^2}\,c\,\big(\xi\sigma_\xi - 2\sigma + E\sigma\big), \qquad
\xi\,\partial_\xi W_c = \dots - \frac{2}{\xi^2}\,c\,\big(\Psi + 2E\big)
```

(the dots are the linearized inner equations; the σ equation, written as $`(\xi\partial_\xi - 1)(c y)`$, is not forced), and

```math
\lambda_1 = -\,\frac{\ell\cdot S(W)}{\ell\cdot\partial_\lambda\mathcal S(2)\,f_2} ,
```

since $`\mathcal D Z_n`$ has zero multiplier when $`Z_n`$ has. This route needs only n = ∞ quantities, not $`Z_1`$, $`K_1`$ or $`\omega_1`$.
*Check* (`first_order_checks.py symmetry`): with 25 harmonics, $`\mathcal D Z_1 + W`$ equals the direct route's mixed component
order by order to 3×10⁻¹⁵ (orders 1–8, harmonics not reached by the truncation). So the two routes are the same
in the full problem. In a truncated model they are not: there $`\mathcal S(2)`$ is not singular (the gauge step multiplies by
P(θ) and does not commute with the truncation), and $`\mathcal D Z_1`$ has a jump, so the symmetry-route number depends on the
matching radius (0.029 at ξ₁ = 0.2, 0.053 at 0.175 for M = 3) and is not a model quantity. The truncated results below
use the direct route.

## 5. Numerical method and checks

**Jets of the lateral integration.** The multipliers are computed as in `solver/inner_stokes/inner_np.py`: the formal series
is summed (optimally truncated) on the ray $`\arg\xi^2 = \pi/2 + \pi/8`$, where every fast mode is exponentially large, the
equations are integrated out to $`\xi_1`$ and along the arc to the real axis, and the jump is $`y^+ - \overline{y^+}`$ (real data).
The model is inner_np's M-harmonic truncation (y, σ on odd |μ| ≤ M, c and E on even |μ| ≤ M + 1, same product order), so
its zeros are those of the five-harmonic selection of the README. `first_order_jet.py` carries every field as a
truncated multivariate dual number: component 0 is the solution and each other component a first-order derivative
along a "direction" (a parameter, ω, the O(1/n) terms, a perturbation with its own λ, d/dλ), with mixed components
for second order. The series recursion and the right-hand side are written once in jet arithmetic, so every Jacobian
is exact (the derivative of the numerical scheme), and a linear response costs one more component of the same
integration. The jumps are de-mixed and normalized as in inner_np (`residual_demixed`) with the map fixed at the
zero.

**Deflation for M = 5** (`first_order_hier.py`). The de-mixed residual is block triangular with huge couplings
(harmonic ν leaks into the harmonic-1 row with weight ~$`e^{\pi(\nu-1)\omega/2}`$ times the finite-ξ de-mixing error;
at M = 5 the one-shot Jacobian has singular values from 6×10¹⁵ to 1.4×10⁻⁴ and Newton diverges from the
nine-digit zero of hier.py). Every harmonic-1 direction is therefore deflated: a first integration gives the
coefficients of the higher-harmonic directions that cancel its own higher-harmonic jump (block back-substitution from
the top harmonic), a second integration of the deflated direction gives its harmonic-1 jump without the large
leakage, and the remaining Schur correction acts on residuals of ~10⁻⁵ instead of ~10². This is hier.py's nested solve
done with exact derivatives.

**Checks** (`first_order_checks.py`):

| check | result |
|---|---|
| background series = `inner_series.py` (flint), M = 3, orders ≤ 30 | 1.6×10⁻¹³ relative |
| perturbation series = `pert.py` (flint), M = 5, λ = 1.7, orders ≤ 20 | 4.6×10⁻¹⁵ |
| O(1/n) series = next-to-top outer coefficients (§2) | 1×10⁻¹³ |
| value jump = inner_np's, M = 3 | 8×10⁻¹³ relative |
| ∂/∂A, ∂/∂ω jets = centred differences of inner_np's jumps | 2×10⁻⁹, 1.6×10⁻⁸ (difference error) |
| right-hand side = derivative of the summed series, every jet feature, ξ = 0.04 e^{0.3i} | 10⁻¹⁰–10⁻¹⁵ |
| exact modes λ = 0 (f = F′) and λ = 1 (f = F + ωF′) at the zeros: smallest singular value | CHECKS_PLACEHOLDER |
| symmetry route = direct route, untruncated series (§4) | 3×10⁻¹⁵ |
| matching radius ξ₁ = 0.2 vs 0.175, M = 3 | ω₁ to 3×10⁻⁹, λ* to 3×10⁻¹⁰, λ₁ to 10⁻⁷ |

## 6. Results against the fits

FITS_AND_RESULTS_PLACEHOLDER

## 7. What is rigorous, what is numerical, what is truncated

STATUS_PLACEHOLDER

## 8. Scripts

| script | what it does | time (2 shared cores) |
|---|---|---|
| `first_order_equations.py` | sympy: the O(1/n) inner equations from the exact ones, heat form, s-translation breaking, exact reparametrization | 1 min |
| `first_order_jet.py` | module: jets of the series and of the lateral integration (M-harmonic), the breaking forcing | |
| `first_order_run.py M` | one-shot jets: zero, ω₁, checks at λ = 0, 1, λ*, λ₁ (direct), symmetry-route number; M = 1, 3 | 3–8 min |
| `first_order_hier.py M` | the same with deflated directions (needed for M = 5), plus the gauge diagnostic | M = 5: ~1 h |
| `first_order_checks.py` | the checks of §5 | 2 min |
| `first_order_fits.py` | Δ₁, ω₁ and λ₁ fitted to the double-double points | 15 min |

Results are in `solver/inner_pert/data/first_order_*.json`, logs in `solver/inner_pert/logs/`.

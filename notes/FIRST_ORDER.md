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

The O(1/n) inner problem gives both coefficients, in the M-harmonic truncation of the selection used in the README
(M = 1, 3, 5), against the double-double fits:

| | M = 1 | M = 3 | M = 5 | fit (double-double) |
|---|---|---|---|---|
| ω*/ω∞ (n = ∞, for reference) | 0.776994 | 1.015603 | 0.999228 | 1 |
| ω₁ | −21.052 | −23.709 | −24.149 | −24.1231 |
| ω₁/ω* | −7.0887 | −6.1078 | −6.3230 | −6.3114 |
| Δ₁ = −2π ω₁/ω*² | 15.00 | 9.886 | 10.402 | 10.3753 |
| λ* (n = ∞) | 1.76912 | 2.00680 | 1.99967 | 2 |
| λ₁ | 7.757 | 3.0797 | 3.0223 | 3.0324 |

with ω = ω* + ω₁/n + …, Δ = Δ* + Δ₁/n + …, λ = 1/γ = λ* + λ₁/n + …. At five harmonics the derived values are within
0.1% (ω₁), 0.2% (ω₁/ω*), 0.3% (Δ₁) and 0.3% (λ₁) of the fitted ones. Every sequence approaches the fit as harmonics are
added, the relative ones alternating like the truncation sequence of ω* itself, and for every quantity the fitted
value lies between the M = 3 and M = 5 values, at 5% (ω), 6% (ω₁) and 18% (λ₁) of the way from M = 5 to M = 3. The
truncated model's own growth rate tends to the exact 2 of Proposition 5 (−0.23, +0.0068, −0.00033).

**Method.** The inner equations are expanded to O(1/n) from the exact equations (§1); the O(1/n) inner solution is
fixed by the profile correction $`K_1`$, by $`\omega_1`$ and by the light-cone gauge (§2); the multiplier of the corrected
inner solution must vanish at O(1/n), which is a linear system for $`(K_1, \omega_1)`$ whose ω component is
ω₁ = −S₁/ₙ/∂_ωS with the profile correction eliminated (§3). The outer region adds nothing at this order: the
light-cone exponent's gravitational shift and every matching factor multiply a multiplier that already vanishes (§3).
λ₁ is first-order perturbation theory of the growth-rate condition, −ℓ·S₁/ₙ / ℓ·∂_λS (§4). All derivatives are exact
jets of the lateral Stokes-jump integration (§5).

**Status.** The O(1/n) inner equations, the matching data and the symmetry-breaking term are derived and checked
symbolically and against the outer 1/n expansion. The selection at O(1/n) is as rigorous as the selection itself
(Conjecture 1 and the lemma chain of C_selection.md, applied one order further). The numbers are those of truncated
models; the truncation error at M = 5 is estimated from the sequences at ~0.05–0.3% and is the dominant
uncertainty (the numerical error of each M = 5 value is far smaller, §6). An untruncated computation was not done.

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

**The fitted coefficients** (`first_order_fits.py`). Δ: the 819 cleaned double-double points (as `hp/dd_fits_mp.py
--clean`), least squares at 60 digits with polynomials of degree 12–20 in 1/n or 1/D on D ≥ 10, 12 or 16. The first
coefficient is the same in both variables (1/D = 1/n − 2/n² + …), and the fits that reach the points' precision
(max |residual| ≤ 10⁻¹⁴) give

```math
\Delta_1 = 10.37528 \pm 0.00001, \qquad \omega_1 = -\frac{\omega_\infty\Delta_1}{\Delta_\infty} = -24.12306 \pm 0.00003, \qquad \frac{\omega_1}{\omega_\infty} = -6.311408 \pm 0.000005
```

(one under-constrained fit, D ≥ 16 at degree 20, gives −24.1233). λ: the 52 double-double growth rates (one per 0.5 in
D, as `hp/gamma_compare.py`), polynomials in 1/n with λ∞ = 2 fixed. The 1/n coefficient rises with the degree and
settles: on D ≥ 30, 3.03236, 3.03243, 3.03244 at degrees 8, 9, 10 (D ≥ 24: 3.03225, 3.03239, 3.03243); free fits
put λ∞ − 2 at 5×10⁻⁸ and λ₁ = 3.03242. So

```math
\lambda_1 = 3.0324 \pm 0.0001 .
```

**The derived coefficients.** For each M: the model's zero (n = ∞), then the O(1/n) selection (§3), the model's growth
rate near 2 and its first-order shift (§4, direct route). Full output in `data/first_order_M{1,3}_xi0.2.json` (one-shot
jets) and `data/first_order_hier_M{1,3,5}_xi0.2.json` (deflated; the same numbers for M = 1, 3).

| M | A | ω*/ω∞ | ω₁ | ω₁/ω* | Δ₁ | λ* | λ₁ |
|---|---|---|---|---|---|---|---|
| 1 | 2.5421480 | 0.776994014 | −21.0519 | −7.08873 | 14.9977 | 1.769122008 | 7.7565 |
| 3 | 2.7264759 | 1.015603095 | −23.7089 | −6.10776 | 9.88625 | 2.006804459 | 3.07967 |
| 5 | 2.7185792 | 0.999228403 | −24.1488 | −6.32302 | 10.4024 | 1.99966851 | 3.0223 |
| fit | | 1 | −24.1231 | −6.31141 | 10.3753 | 2 | 3.0324 |

The 1/n corrections of the profile come out large (for M = 5, $`A_1 = 14.45`$ against A = 2.72, and $`K_1`$'s harmonic 3
relatively six times larger than F's), as the O(1/n) corrections of the solution are at finite D (README, "How the
solution scales with D").

**Convergence in M.** Differences from the fits:

| | M = 1 | M = 3 | M = 5 | ratio of successive errors |
|---|---|---|---|---|
| ω*/ω∞ − 1 (README's sequence) | −0.223 | +0.0156 | −0.00077 | −0.07, −0.05 |
| ω₁/ω* | +12.3% | −3.2% | +0.18% | −0.26, −0.06 |
| Δ₁ | +44.6% | −4.7% | +0.26% | −0.11, −0.06 |
| ω₁ | −12.7% | −1.7% | +0.11% | 0.13, −0.06 |
| λ* − 2 | −0.231 | +0.0068 | −0.00033 | −0.03, −0.05 |
| λ₁ | +156% | +1.56% | −0.33% | 0.01, −0.21 |

From M = 3 to M = 5 every error changes sign and shrinks, by a factor of about 5 (λ₁) to 20 (the ω quantities), as
the error of ω* does (20). With alternating signs the M → ∞ limit lies between the M = 3 and M = 5 values, and the
fitted value does, for every quantity. If the next error ratio r is like ω*'s (r ≈ −0.05), the limit is
$`x_5 + \tfrac{-r}{1-r}(x_3 - x_5) \approx x_5 + 0.048\,(x_3 - x_5)`$:

| | M = 5 | limit with r = −0.05 | limit with r = −0.2 | fit |
|---|---|---|---|---|
| ω₁ | −24.1488 | −24.128 | −24.075 | −24.1231 |
| ω₁/ω* | −6.3230 | −6.3128 | −6.287 | −6.3114 |
| Δ₁ | 10.4024 | 10.378 | 10.316 | 10.3753 |
| λ₁ | 3.0223 | 3.0250 | 3.0319 | 3.0324 |

The ω coefficients match the r = −0.05 reading (to 0.02–0.03%), λ₁ the r = −0.2 one, which is also the ratio its own
last step showed. This is a reading of three-term sequences, not an error bound: the truncation error of the M = 5
values is ~0.1–0.3%, and a sharper statement needs M = 7.

**A second measure of the truncation error: the gauge datum.** In the full problem ω₁ does not depend on the gauge datum
$`c_1(0)`$ (§3). In the models it does: with $`c_1(0) = 0`$ instead of $`-g_1`$, ω₁ is −28.61, −23.81, −24.20 for M = 1, 3, 5,
i.e. it moves by 7.6, 0.10 and 0.052. At M = 5 that is twice the remaining difference from the fit (0.026), consistent
with the estimate above.

ROBUSTNESS_PLACEHOLDER

## 7. What is rigorous, what is numerical, what is truncated

**Derived and checked (exact, symbolic or to roundoff):**

- the O(1/n) inner equations, from the exact equations, with η slaved and no half-integer powers (§1; sympy);
- the matching data of the O(1/n) inner solution: $`K_1`$, $`\omega_1`$ and the gauge datum $`c_1(0) = -F^2`$ (§2; the O(1/n) inner
  series equals the outer expansion's next-to-top coefficients to 10⁻¹³);
- the symmetry-breaking term of the λ = 2 mode (§4; sympy for arbitrary fields, and the series identity
  $`\mathcal D Z_1 + W`$ = direct route to 3×10⁻¹⁵);
- the exactness of the time reparametrization at O(1/n), hence the gauge independence of ω₁ and λ₁ in the full problem.

**Argued, at the level of rigor of the selection rule itself:**

- that the selection at O(1/n) is the vanishing of the multiplier of $`Z_0 + Z_1/n`$, and that the outer region (matching
  factors, transport, the light-cone exponent and its gravitational shift, the half-monodromy) contributes only
  invertible factors at this order (§3). This is Conjecture 1 plus the lemma chain of C_selection.md carried one order
  further (its "1/n refinement" of Lemma B5(2), written at appendix level there);
- first-order perturbation theory for λ: λ* is a simple root and ℓ·∂_λS·f* ≠ 0 (seen numerically in every model:
  ℓ·∂_λS·f* = −4.4×10³, −6.5×10⁴, −5.9×10⁴ for M = 1, 3, 5).

**Numerical, for each truncated model:** the zeros, ω₁, λ* and λ₁ of the M-harmonic models, with exact jets and the
numerical error of §6 (far below the truncation error).

**Truncated:** the numbers are those of the M = 1, 3, 5 models. Their sequences converge toward the fits and bracket
them, but the M → ∞ limit is not computed. In particular:

- harmonic truncation is not a controlled approximation in general (README: the multipliers involve F continued to
  Im θ ≈ πω/2). Here the sequences behave regularly from M = 3 on, as ω*'s does, but three terms are a thin basis;
- in a truncated model λ = 2 is not exact and the symmetry route does not apply (§4), so λ₁ is the model's own
  first-order coefficient about its own λ*.

**Open:**

- M = 7, or an untruncated computation. The amplitude expansion of `solver/inner_amp/` (which gave ω∞ to 10⁻⁴ without
  truncation) would need its linear-response version: the O(1/n) forcing and the perturbation at λ = 2 expanded in the
  amplitude, with their late-order fits. With the symmetry route (§4) the λ₁ computation then needs only n = ∞
  objects: the λ = 2 mode of Proposition 5, the forced response W, and ∂_λS at λ = 2;
- a proof of the O(1/n) step of the matching lemmas.

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

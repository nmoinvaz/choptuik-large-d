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
| ω₁ | −21.052 | −23.709 | −24.149 | −24.123 |
| ω₁/ω* | −7.0887 | −6.1078 | −6.3229 | −6.3114 |
| Δ₁ = −2π ω₁/ω*² | 15.00 | 9.886 | 10.402 | 10.3753 |
| λ* (n = ∞) | 1.76912 | 2.00680 | 1.99967 | 2 |
| λ₁ | 7.757 | 3.0797 | 3.023 ± 0.001 | 3.0324 |

with ω = ω* + ω₁/n + …, Δ = Δ* + Δ₁/n + …, λ = 1/γ = λ* + λ₁/n + …. At five harmonics the derived values are within
0.1% (ω₁), 0.2% (ω₁/ω*), 0.3% (Δ₁) and 0.3% (λ₁) of the fitted ones. Every sequence approaches the fit as harmonics are
added, alternating from M = 3 on like the truncation sequence of ω* itself, and for every quantity the fitted value lies
between the M = 3 and M = 5 values: 5% of the way from M = 5 to M = 3 for ω*, ω₁/ω* and Δ₁, 6% for ω₁ and 17% for λ₁.
The truncated model's own growth rate tends to the exact 2 of Proposition 5 (λ* − 2 = −0.23, +0.0068, −0.00033).

**Method.** The inner equations are expanded to O(1/n) from the exact equations (§1); the O(1/n) inner solution is
fixed by the profile correction $`K_1`$, by $`\omega_1`$ and by the light-cone gauge (§2); the multiplier of the corrected
inner solution must vanish at O(1/n), which is a linear system for $`(K_1, \omega_1)`$ whose ω component is
ω₁ = −S₁/ₙ/∂_ωS with the profile correction eliminated (§3). The outer region adds nothing at this order: the
light-cone exponent's gravitational shift and every matching factor multiply a multiplier that already vanishes (§3).
λ₁ is first-order perturbation theory of the growth-rate condition, −ℓ·S₁/ₙ / ℓ·∂_λS (§4). All derivatives are exact
jets of the lateral Stokes-jump integration (§5).

**Status.** The O(1/n) inner equations, the matching data and the symmetry-breaking term are derived and checked
symbolically and against the outer 1/n expansion. The selection at O(1/n) is conditional on the same things as the
selection itself (Conjecture 1 and the lemma chain of C_selection.md, applied one order further). The numbers are
numerical evidence from truncated models: the truncation error at M = 5, estimated from the sequences and from the
gauge dependence of the models (which vanishes in the full problem), is ~0.2% for ω₁ and ~0.3–0.8% for λ₁, and is the
dominant uncertainty (the numerical error of each M = 5 value is ~10⁻⁵ for ω₁ and 3×10⁻⁴ for λ₁, §6). M = 7 and an
untruncated computation were not done.

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
exact symmetry of the corrected equations (it is a symmetry of the exact equations).

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
in the full problem. In a truncated model they are not. The gauge step multiplies by P(θ) and does not commute with the
truncation, so there $`\mathcal D Z_0`$ and $`\mathcal D Z_1`$ do not solve the model's equations, $`\mathcal S(2)`$ is not singular, and the
"jump" of $`\mathcal D Z_1`$ at ξ₁ is not a fast-type solution of the model. The symmetry-route number then depends on the
matching radius (0.029 at ξ₁ = 0.2, 0.053 at 0.175 for M = 3, against 3.080 by the direct route at both) and is not a
model quantity. The truncated results below use the direct route; the symmetry route is the one to use in an
untruncated computation, where it needs only n = ∞ objects.

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
nine-digit zero of hier.py; `logs/first_order_M5_oneshot_diverged.log`). Every harmonic-1 direction is therefore
deflated: a first integration gives the coefficients of the higher-harmonic directions that cancel its own
higher-harmonic jump (block back-substitution from the top harmonic), further integrations of the deflated direction
refine them until that jump stops falling, and the harmonic-1 jump of the deflated direction then carries no large
leakage. The remaining Schur correction acts on small residuals: for the zero ~10⁻⁵ instead of ~10²; for the 1/n
direction 2×10⁻³ instead of 2.6×10¹¹; for the mixed component of λ₁ 10⁻² instead of 2.6×10¹². This is hier.py's nested
solve done with exact derivatives. For M = 1, 3 the deflated and the one-shot drivers agree (§6).

**Checks** (`first_order_checks.py`, which exits non-zero if any fails; each has a mutation that makes it fail,
`--mutate=NAME`, all run and seen to fail):

| check | independent of the jets in | result (threshold) | mutation → result |
|---|---|---|---|
| background series = `solver/inner_borel/inner_series.py`, M = 3, orders ≤ 30 | flint, separate recursion | 2.7×10⁻¹³ (10⁻¹¹) | |
| perturbation series = `pert.py`, M = 5, λ = 1.7, orders ≤ 20 | flint, separate recursion | 3.8×10⁻¹⁵ (10⁻¹¹) | λ + 10⁻³ → 2×10⁻³ |
| O(1/n) series = next-to-top outer coefficients (§2), orders ≤ 8 | outer recursion (`largeD_late_orders.py`) | 1.7×10⁻¹³ (10⁻¹⁰) | sign of the y term → 0.64; c term dropped → 0.17; c₁(0) = 0 → 3.2 |
| right-hand side = derivative of the summed series, every jet feature | series recursion vs ODE right-hand side | 1.0×10⁻⁹ (10⁻⁸) | sign of the y term in the rhs only → 460 |
| value jump = `inner_np.py`'s, M = 3 | the lateral method is shared | 1.9×10⁻¹¹ (10⁻¹⁰) | |
| ∂/∂A, ∂/∂ω jets = centred differences of inner_np's jumps | | 2×10⁻⁹, 1.6×10⁻⁸ (10⁻⁶) | ω direction scaled by 2 → 1.0 |
| symmetry route = direct route, untruncated series (§4) | hand-written 𝒟Z₁ | 3.2×10⁻¹⁵ (10⁻¹²) | sign of the forcing → 3.8; sign of the y term → 0.48 |

In `first_order_hier.py --checks` (the exact modes of every model): at the model's zero the 2 × 2 Schur complement of
the perturbation map is singular at λ = 0 (null vector the harmonic-1 part of F′) and at λ = 1 (of F + ωF′), and not
at λ = 0.5 (control):

| M | λ = 0: σ_min/σ_max, angle to F′ | λ = 1: σ_min/σ_max, angle to F + ωF′ | λ = 0.5: σ_min/σ_max |
|---|---|---|---|
| 1 | 2.6×10⁻¹¹, 9.6×10⁻¹² | 1.0×10⁻¹⁰, 1.2×10⁻¹¹ | 3.4×10⁻² |
| 3 | 5.4×10⁻¹⁰, 6.8×10⁻¹¹ | 8.6×10⁻¹¹, 5.5×10⁻¹¹ | 3.9×10⁻² |
| 5 | 2.0×10⁻⁶, 2.9×10⁻⁶ | 1.1×10⁻⁶, 2.1×10⁻⁶ | 3.9×10⁻² |

(At M = 5 the floor is the noise of the deflated Schur complement, ~10⁻² against entries of 10⁴.)

**Robustness:** the matching radius and the integration tolerance (§6).

## 6. Results against the fits

**The fitted coefficients** (`first_order_fits.py`). Δ: the 819 cleaned double-double points (as `hp/dd_fits_mp.py
--clean`), least squares at 60 digits with polynomials of degree 12–18 in 1/n or 1/D on D ≥ 10, 12 or 16. The first
coefficient is the same in both variables (1/D = 1/n − 2/n² + …). The 17 fits that reach the points' precision
(max |residual| ≤ 10⁻¹³) give Δ₁ from 10.375279 to 10.375321 (16 of them within 10.375279–10.375286), so

```math
\Delta_1 = 10.37528 \;(+0.00004), \qquad \omega_1 = -\frac{\omega_\infty\Delta_1}{\Delta_\infty} = -24.1231 \pm 0.0001, \qquad \frac{\omega_1}{\omega_\infty} = -6.31141 \pm 0.00002 ,
```

the ranges being the spread over fit choices. λ: the 52 double-double growth rates (one per 0.5 in D, as
`hp/gamma_compare.py`), polynomials in 1/n with λ∞ = 2 fixed. The 1/n coefficient rises with the degree and settles:
on D ≥ 30, 3.03236, 3.03243, 3.03244 at degrees 8, 9, 10 (D ≥ 24: 3.03225, 3.03239, 3.03243); free fits put λ∞ − 2 at
5×10⁻⁸ and λ₁ = 3.03242. So, as a spread over fit choices,

```math
\lambda_1 = 3.0324 \pm 0.0001 .
```

*Planted truth* (`first_order_fits.py --planted`, pass criteria fixed before the first run): the same fits on
synthetic data at the same D, with known first coefficients (the README's rational formulas for Δ and γ, noise 10⁻¹⁴
on Δ and 10⁻¹² on λ). Every Δ fit of degree 12–18 that meets the residual criterion recovers Δ₁ within 1.5×10⁻⁵;
the λ fits of degree 9–10 recover λ₁ within 8×10⁻⁵ (all a little low, by 3–8×10⁻⁵). The first run also had degree-20
Δ fits, which missed by 4×10⁻⁵ to 5×10⁻⁴ (overfitting); degree 20 was therefore removed from the estimator
(`logs/first_order_fits_planted_first.log`), and the test passes as rerun.

**The derived coefficients.** For each M: the model's zero (n = ∞), then the O(1/n) selection (§3), the model's growth
rate near 2 and its first-order shift (§4, direct route). Full output in `data/first_order_M{1,3}_xi0.2.json` (one-shot
jets) and `data/first_order_hier_M{1,3,5}_xi0.2.json` (deflated; the same numbers for M = 1, 3).

| M | A | ω*/ω∞ | ω₁ | ω₁/ω* | Δ₁ | λ* | λ₁ |
|---|---|---|---|---|---|---|---|
| 1 | 2.5421480 | 0.776994014 | −21.0519 | −7.08873 | 14.9977 | 1.769122008 | 7.7565 |
| 3 | 2.7264759 | 1.015603095 | −23.7089 | −6.10776 | 9.88625 | 2.006804459 | 3.07967 |
| 5 | 2.7185792 | 0.999228403 | −24.1485 | −6.3229 | 10.4023 | 1.9996689 | 3.023 |
| fit | | 1 | −24.1231 | −6.31141 | 10.3753 | 2 | 3.0324 |

(M = 5: means of the three runs of the robustness table below; their spread is ±0.0003 in ω₁ and ±0.001 in λ₁.)

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
| λ₁ | +156% | +1.56% | −0.31% | 0.01, −0.20 |

From M = 3 to M = 5 every error changes sign and shrinks, by a factor of about 5 (λ₁) to 20 (the ω quantities), as
the error of ω* does (20). With alternating signs the M → ∞ limit lies between the M = 3 and M = 5 values, and the
fitted value does, for every quantity. If the next error ratio r is like ω*'s (r ≈ −0.05), the limit is
$`x_5 + \tfrac{-r}{1-r}(x_3 - x_5) \approx x_5 + 0.048\,(x_3 - x_5)`$:

| | M = 5 | limit with r = −0.05 | limit with r = −0.2 | fit |
|---|---|---|---|---|
| ω₁ | −24.1485 | −24.128 | −24.075 | −24.1231 |
| ω₁/ω* | −6.3229 | −6.3127 | −6.287 | −6.3114 |
| Δ₁ | 10.4023 | 10.378 | 10.316 | 10.3753 |
| λ₁ | 3.023 | 3.026 | 3.032 | 3.0324 |

The ω coefficients match the r = −0.05 reading (to 0.02–0.03%), λ₁ the r = −0.2 one, which is also the ratio its own
last step showed. This is a reading of three-term sequences, not an error bound; a sharper statement needs M = 7.

**A second measure of the truncation error: the gauge data.** In the full problem ω₁ and λ₁ do not depend on the gauge
data $`c_1(0)`$ and $`\delta c_1(0)`$ (§3, §4). In the models they do. With both set to 0 instead of the light-cone values
(`--lam-gauge`; `omega1_gauge0`, `lam1_gauge0` in the JSON files):

| M | ω₁, light-cone gauge | ω₁, gauge 0 | λ₁, light-cone gauge | λ₁, gauge 0 |
|---|---|---|---|---|
| 1 | −21.05 | −28.61 | 7.757 | 0.740 |
| 3 | −23.709 | −23.811 | 3.0796 | 2.8268 |
| 5 | −24.149 | −24.201 | 3.022 | 3.047 |
| fit | −24.123 | | 3.0324 | |

The gauge dependence falls with M (ω₁: 7.6, 0.10, 0.052; λ₁: 7.0, 0.25, 0.025), and at M = 5 it is about twice the
remaining difference from the fit, consistent with the estimates above. The gauge-0 values of λ₁ also alternate, and
at M = 5 the two gauges bracket the fitted value (3.022 \< 3.0324 \< 3.047). So the truncation error of the M = 5
values is ~0.2% for ω₁ and ~0.3–0.8% for λ₁.

**Numerical error of the model values.** At M = 3, the one-shot and the deflated drivers and two matching radii
(ξ₁ = 0.2, 0.175) agree to 10⁻⁸ in ω₁ and λ*, and to 10⁻⁵ in λ₁ (the deflated driver takes ∂_λ by centred differences).
At M = 5, three runs that differ in the matching radius (0.2, 0.19) and the integration tolerance (10⁻¹², 10⁻¹³) give

| run | ω₁ | ω₁/ω* | λ* | λ₁ |
|---|---|---|---|---|
| ξ₁ = 0.2 | −24.1488 | −6.32302 | 1.9996685 | 3.0223 |
| ξ₁ = 0.19 | −24.1485 | −6.32295 | 1.9996694 | 3.0220 |
| ξ₁ = 0.2, rtol 10⁻¹³ | −24.1482 | −6.32286 | 1.9996688 | 3.0242 |

so ω₁ = −24.1485 ± 0.0003 and λ₁ = 3.023 ± 0.001 (half the spread of the three runs), well below the truncation
error. What limits λ₁ is the mixed component's higher-harmonic jump, which the iterated deflation brings from 2.6×10¹²
down to only ~10⁻², so that its Schur correction still carries ~10⁻³ of the result.

## 7. What is rigorous, what is numerical, what is truncated

**Checked symbolically** (sympy, `first_order_equations.py`), and numerically against independent code:

- the O(1/n) inner equations, from the exact equations, with η slaved and no half-integer powers (§1);
- the symmetry-breaking term of the λ = 2 mode (§4), for arbitrary fields; and the series identity
  $`\mathcal D Z_1 + W`$ = direct route to 3×10⁻¹⁵;
- the exactness of the time reparametrization at O(1/n), hence the gauge independence of ω₁ and λ₁ in the full problem.

**Derived, and checked numerically to roundoff:** the matching data of the O(1/n) inner solution, $`K_1`$, $`\omega_1`$ and the gauge
datum $`c_1(0) = -F^2`$ (§2; the O(1/n) inner series equals the outer expansion's next-to-top coefficients to 10⁻¹³).

**Conditional**, on Conjecture 1 and on the lemma chain of C_selection.md carried one order further (its "1/n
refinement" of Lemma B5(2), written at appendix level there):

- that the selection at O(1/n) is the vanishing of the multiplier of $`Z_0 + Z_1/n`$, and that the outer region (matching
  factors, transport, the light-cone exponent and its gravitational shift, the half-monodromy) contributes only
  invertible factors at this order (§3);
- first-order perturbation theory for λ, which also needs λ* to be a simple root with ℓ·∂_λS·f* ≠ 0 (seen in every
  model: ℓ·∂_λS·f* = −4.4×10³, −6.5×10⁴, −5.9×10⁴ for M = 1, 3, 5, in the units of the de-mixed residual).

**Numerical evidence**, for each truncated model: the zeros, ω₁, λ* and λ₁ of the M-harmonic models, with exact jets and
the numerical error of §6 (far below the truncation error).

**Truncated:** the numbers are those of the M = 1, 3, 5 models. Their sequences converge toward the fits and bracket
them, but the M → ∞ limit is not computed. In particular:

- harmonic truncation is not a controlled approximation in general (README: the multipliers involve F continued to
  Im θ ≈ πω/2). Here the sequences behave regularly from M = 3 on, as ω*'s does, but three terms are a thin basis;
- in a truncated model λ = 2 is not exact and the symmetry route does not apply (§4), so λ₁ is the model's own
  first-order coefficient about its own λ*.

**Open:**

- M = 7. Its zero is not known (hier.py's M = 7 runs only solved the higher levels at fixed (A, ω)). A pilot of the
  deflated Newton from the M = 5 zero with hier.py's M = 7 harmonics (`logs/first_order_hier_M7_approach.log`, with
  hier.py's approach radii 0.119 and 0.103 for harmonics 5 and 7) starts in the nonlinear regime (|r_H| ~ 4×10¹⁰) and
  its linear inner step makes it worse (2×10¹⁵); at ~9 min per iteration it was stopped. The M = 7 zero needs hier.py's
  level-by-level damped approach first; the O(1/n) steps would then follow as at M = 5;
- an untruncated computation. The amplitude expansion of `solver/inner_amp/` (which gave ω∞ to 10⁻⁴ without
  truncation) would need its linear-response version: the O(1/n) forcing and the perturbation at λ = 2 expanded in the
  amplitude, with their late-order fits. With the symmetry route (§4) the λ₁ computation then needs only n = ∞
  objects: the λ = 2 mode of Proposition 5, the forced response W, and ∂_λS at λ = 2;
- the λ₁ precision at M = 5 (±0.001) is set by the deflation of the mixed component; higher working precision (or a
  better de-mixing of the jump at finite ξ₁) would improve it;
- a proof of the O(1/n) step of the matching lemmas.

## Proposed text for the README and GAMMA.md

For the coordinator (agents do not edit the README). In the README after Proposition 5, "Still open: … the coefficient
3.0324, which needs the O(1/n) inner equations …", and in GAMMA.md §6, "The 1/n coefficient", something like:

> **The 1/n coefficients from the inner problem** ([notes/FIRST_ORDER.md](notes/FIRST_ORDER.md)). The inner equations
> carry O(1/n) corrections (integer powers only, Lemma B5.0); with η slaved they are three explicit terms, and in the
> heat form they are the only place where s = r²/(2n) appears explicitly. The O(1/n) inner solution is fixed by the
> profile's 1/n correction, by ω₁ and by the light-cone gauge, and the vanishing of its multiplier gives ω₁; the
> outer region contributes only invertible factors at this order (the gravitational shift of the light-cone exponent
> included). First-order perturbation theory gives λ₁. In the M-harmonic truncation, with exact derivatives:
> ω₁ = −21.05, −23.71, −24.149 and λ₁ = 7.76, 3.080, 3.023 for M = 1, 3, 5, against the fitted −24.123
> (Δ₁ = 10.3753) and 3.0324. The sequences converge like the truncation sequence of ω* itself, and the fitted values
> lie between the M = 3 and M = 5 values; at M = 5 the differences are 0.1% (ω₁) and 0.3% (λ₁).

## 8. Scripts

| script | what it does | time (2 shared cores) |
|---|---|---|
| `first_order_equations.py` | sympy: the O(1/n) inner equations from the exact ones, heat form, s-translation breaking, exact reparametrization | 1 min |
| `first_order_jet.py` | module: jets of the series and of the lateral integration (M-harmonic), the breaking forcing | |
| `first_order_run.py M` | one-shot jets: zero, ω₁, checks at λ = 0, 1, λ*, λ₁ (direct), symmetry-route number; M = 1, 3 (diverges at M = 5) | 3–8 min |
| `first_order_hier.py M [--checks] [--lam-gauge]` | the same with deflated directions (needed for M = 5), the gauge diagnostic of ω₁ (and of λ₁ with `--lam-gauge`), and with `--checks` the exact modes λ = 0, 1 | M = 5: 30–40 min |
| `first_order_checks.py [--mutate=NAME]` | the checks of §5; exit status 1 on failure | 1 min |
| `first_order_fits.py [--planted]` | Δ₁, ω₁ and λ₁ fitted to the double-double points; `--planted` the planted-truth test | 10–15 min each |

The numbers of §6 are reproduced, from `solver/inner_pert/`, by `python3 first_order_hier.py 5` (and `1`, `3`; add
`--xi1=0.19` or `--rtol=1e-13` for the robustness runs; each resumes from its JSON, `--redo=omega1,lam,lam1` recomputes),
`python3 first_order_run.py 3` and `python3 first_order_fits.py`.

Results are in `solver/inner_pert/data/first_order_*.json`, logs in `solver/inner_pert/logs/`.

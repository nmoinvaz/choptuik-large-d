# Other matter models at large D: structure

For the massless real scalar, the large-D analysis of this repository rests on three facts
([README, "Large D"](../README.md#large-d-what-the-1d-expansion-can-and-cannot-see), [GAMMA.md](../GAMMA.md)):

- the outer 1/n expansion (x = O(1)) can be carried to all orders for any period and any profile (Proposition 1), so
  the period is selected beyond all orders;
- at $`x \sim \sqrt n`$ there is an n-free inner problem, and Δ∞ is (conjecturally) selected by the vanishing of its
  fast-mode Stokes multipliers;
- in $`s = r^2/(2n)`$ and the time t the inner problem is autonomous, a nonlinear heat equation; its s-translation is an
  exact growth rate λ = 2, so γ∞ = 1/2.

This page asks which of these survive for other matter models, and derives as much as can be derived. The models are
the scalar ℓ-multiplet (ℓ fixed and ℓ ∝ n), Yang–Mills with the SO(D − 1) magnetic ansatz, self-gravitating wave maps
(hedgehog σ-models) into $`S^{D-1}`$, and, as a control, scalars without a potential (complex scalar, Hirschmann–Eardley
family, axion–dilaton), for which the phase-rotating self-similar solution closes on one harmonic and its D = ∞ selection
was computed here. The literature on these models is in [LITERATURE.md](LITERATURE.md).

Notation as in the README: n = D − 2, ε = 1/n, x = r/(−t), τ = −ln(−t), θ = ωτ, $`s = r^2/(2n)`$, b = a², c = α/a, and
γ is the radius exponent, γ = 1/λ. Every statement is marked:
- **derived**: checked by a sympy script in [`solver/matter/`](../solver/matter/) (named in brackets), with all residuals 0;
- **computed**: a numerical result, with its checks;
- **argued**: a derivation that is not complete, or an argument from scalings;
- **conjecture**: rests on the README's conjecture that D = ∞ selection is the vanishing of the inner Stokes multipliers, or
  on something not shown.

Contents

- [1. Summary](#1-summary)
- [2. The common framework](#2-the-common-framework)
- [3. The scalar ℓ-multiplet, ℓ fixed](#3-the-scalar-ℓ-multiplet-ℓ-fixed)
- [4. The ℓ-multiplet with ℓ ∝ n](#4-the-ℓ-multiplet-with-ℓ--n)
- [5. Yang–Mills](#5-yangmills)
- [6. Wave maps](#6-wave-maps)
- [7. Scalars without a potential, and the phase-rotating CSS solution at D = ∞](#7-scalars-without-a-potential-and-the-phase-rotating-css-solution-at-d--)
- [8. Other models](#8-other-models)
- [9. Prediction table](#9-prediction-table)
- [10. Open issues](#10-open-issues)
- [11. Scripts](#11-scripts)

## 1. Summary

- **The mechanism that gives γ∞ = 1/2 is specific to matter without a potential on the sphere.** In s the radial
  operator is $`\partial_s - V/(2ns)`$ ([LITERATURE.md §1](LITERATURE.md#1-the-structural-question-which-models-keep-the-s-translation)). Every model whose potential is O(n) (centrifugal barrier,
  σ-model, Yang–Mills) has an O(1) term ∝ 1/s in its D = ∞ inner problem. The s-translation is then broken, and a search
  for Lie point symmetries finds no replacement: the only exact positive growth rate left is λ = 1 (the time
  translation). So for the ℓ ≥ 1 multiplet, the Yang–Mills type II solution and wave maps, **γ∞ is not fixed by a
  symmetry**; it is a root of a transcendental condition, and γ∞ ≠ 1/2 is the generic expectation.
- **The rest of the structure survives.** For ℓ fixed and for wave maps at fixed coupling the inner problem is n-free,
  the outer expansion is formally free (checked to third and fourth order), and the light-cone exponent is unchanged. So
  Δ∞ is finite and selected beyond all orders, as for the scalar (for wave maps, on the DSS branch at large enough
  coupling; conjecture at the same level as the scalar's).
- **Gravity is weaker at x = O(1)**: $`c - 1 = O(n^{-\ell-1})`$ instead of O(1/n). The gravitational shift of the light-cone
  exponent (about 3.7 for the scalar) vanishes at D = ∞ for ℓ ≥ 1.
- **Three models share one D = ∞ problem.** The type II Yang–Mills threshold (which lives in the small-scale system
  linearized about the vacuum) and wave maps at infinite coupling have exactly the inner problem of the ℓ = 1
  multiplet. Prediction: their Δ∞ and γ∞ coincide, although the solutions differ at D = 4 (γ = 0.196 for Yang–Mills,
  0.119 for ℓ = 1).
- **ℓ ∝ n is different**: the problem is WKB at x = O(1), the "inner region" becomes a turning point at
  $`x_t = \sqrt{1 + 1/(4\lambda(1+\lambda))} > 1`$ (λ = ℓ/n), and gravity is exponentially localized there. Inconclusive.
- **Control case.** For the free complex scalar the inner problem keeps the s-translation (λ = 2 exact, γ∞ = 1/2), and
  the Hirschmann–Eardley phase-rotating CSS solution closes exactly on one harmonic. Its D = ∞ selection condition has a
  clean zero at ω∞ = 3.9789695 (Δ = 2π/ω∞ = 1.5790986), amplitude A = 1.9970774, a number that can be checked against
  finite-D CSS solutions, which are ODE problems (ω = 1.9154446 at D = 4).
- **Flat-space blowup** cannot compete in Yang–Mills (the theory has a length scale; a shrinking O(1) field becomes
  strongly gravitating), and does compete in wave maps at small coupling, where it is an exact solution of the D = ∞
  inner problem.

## 2. The common framework

Metric $`ds^2 = -\alpha^2dt^2 + a^2dr^2 + r^2d\Omega_n^2`$. Each model is one field ψ(t, r) whose SO(n + 1)-symmetric stress
tensor has, on the (t, r) block, the form

```math
T_{AB} = K\,\partial_A\psi\,\partial_B\psi + g_{AB}\,\mathcal L, \qquad \mathcal L = -\tfrac12 K\,g^{CD}\partial_C\psi\,\partial_D\psi - V,
\qquad \rho = \frac{K(\Phi^2+\Pi^2)}{2b} + V, \qquad P_r = \frac{K(\Phi^2+\Pi^2)}{2b} - V,
```

with $`\Phi = \psi_r`$, $`\Pi = a\psi_t/\alpha`$, and K(r, ψ), V(r, ψ) given in the table. Einstein's equations $`G_{ab} = \kappa T_{ab}`$ in polar-areal
gauge and the field equation (from the reduced action, by symmetric criticality) are

```math
\frac{a_r}{a} = \frac{\kappa}{n}\,r b\,\rho + \frac{(n-1)(1-b)}{2r}, \qquad
\frac{\alpha_r}{\alpha} = \frac{\kappa}{n}\,r b\,P_r + \frac{(n-1)(b-1)}{2r}, \qquad
\frac{a_t}{a} = \frac{\kappa}{n}\,r\,K\psi_t\psi_r,
```
```math
\partial_t\big(r^nK\Pi\big) = \partial_r\big(r^n cK\Phi\big) + r^n c\,\Big[\tfrac12 K_\psi(\Pi^2-\Phi^2) - b\,V_\psi\Big], \qquad \psi_t = c\,\Pi .
```

**Derived** [`struct_eqs.py`]: the Einstein tensor against brute-force curvature for n = 2, 3, 4; for every model, the
stress tensor of explicit non-symmetric fields (a tight frame of harmonics for the multiplet, the map into $`S^{n+1}`$,
the SO(n + 1) gauge field) is SO(n + 1)-symmetric and has this form, and the covariant field equation ($`\Box\phi_I = 0`$, the
harmonic-map equation, $`D_\mu F^{\mu\nu} = 0`$) is equivalent to the reduced one, at n = 2 and 3 (for Yang–Mills at
n = 3, the t-component and one angular component of $`D_\mu F^{\mu\nu}`$, for cost).

| Model | ψ | K | V | coupling |
|---|---|---|---|---|
| scalar ℓ-multiplet | φ | 1 | $`L\phi^2/(2r^2)`$, L = ℓ(ℓ + n − 1) | absorbed in φ (κ = 2n, as in the solver) |
| wave map into $`S^{n+1}`$ | u | f² | $`n f^2\sin^2u/(2r^2)`$ | η = κf²/2, dimensionless in every D |
| SO(n + 1) Yang–Mills | W | $`n/(e^2r^2)`$ | $`n(n-1)(1-W^2)^2/(4e^2r^4)`$ | $`\ell_{\rm YM}^2 = \kappa/e^2`$, a length² in every D |

The flat Yang–Mills equation that follows is $`W_{tt} = W_{rr} + (n-2)W_r/r + (n-1)W(1-W^2)/r^2`$, i.e. (d − 3) and (d − 2)
with d = n + 1 space dimensions (Bizoń–Tabor); a coefficient (d − 4) would give $`-W_r/r`$ in D = 4, against Gundlach's
equations. η is the D = 4 coupling 4πGf² of Husa et al. and Bizoń–Wasserman.

**D = 4 checks** (derived, [`struct_eqs.py`]): the multiplet equations are those of Olabarrieta, Ventrella, Choptuik and
Unruh (PRD 76, 124014, eqs. for $`a_r`$, $`\alpha_r`$, $`\Pi_t`$, with κ = 8π and $`\sum_m Y_{\ell m}^2 = (2\ell+1)/4\pi`$); the Yang–Mills
equations are Gundlach's (PRD 55, 6002, eqs. 10–12, in units κ/e² = 2); the wave-map potential is the standard hedgehog
energy sin²u/r².

**One identity drives everything below.** Adding the two constraints,

```math
\frac{r\,(\alpha a)_r}{\alpha a} = \frac{\kappa}{n}\,r^2 b\,(\rho + P_r) = \frac{\kappa}{n}\,r^2K\,(\Phi^2+\Pi^2) ,
```

so the potential never enters the equation for c = α/a beyond the term $`-r\,b_r/b`$: in the inner region c is driven by
the kinetic energy alone, in every model.

**Self-similar first-order form** (derived, [`struct_eqs.py`], against the covariant equations; κ = 2n). For the multiplet,
with p, m = r(Φ ± Π) as in the solver and the field φ itself,

```math
x\,a_x = \tfrac{a}{2}\big[p^2+m^2 + 2La^2\phi^2 + (n-1)(1-a^2)\big], \qquad
x\,c_x = c\,\big[(n-1)(a^2-1) - 2La^2\phi^2\big], \qquad x\,\phi_x = \tfrac12(p+m),
```
```math
p_\tau + (x-c)\,p_x = c_x p - \frac{cp}{x} + \frac{nc\,(p+m)}{2x} - \frac{c\,a^2L\phi}{x}, \qquad
m_\tau + (x+c)\,m_x = -c_x m + \frac{cm}{x} - \frac{nc\,(p+m)}{2x} + \frac{c\,a^2L\phi}{x} .
```

At L = 0 these are exactly the equations of [`solver/dss.py`](../solver/dss.py), with φ decoupled (derived). For the
wave map, replace $`p^2+m^2 \to (\eta/n)(p^2+m^2)`$, $`2La^2\phi^2 \to 2\eta a^2\sin^2u`$ and $`L\phi \to n\sin(2u)/2`$ (derived).

## 3. The scalar ℓ-multiplet, ℓ fixed

### Scaling (b)

Inner region, $`r = \sqrt{2ns}`$: $`\phi = n^{-1/2}\varphi(s,t)`$, b = 1 + β/n, c = O(1). This is the distinguished limit (argued): with
$`\phi = n^{-\sigma}\varphi`$ the c-equation reads $`c_s = 2n^{1-2\sigma}\varphi_t^2/c`$; σ > 1/2 leaves the flat (linear) inner
problem, whose multipliers do not vanish (Proposition 2 below), and σ < 1/2 makes c vary on $`s \ll 1`$, i.e. moves the inner
region. So, as for the scalar, the amplitude is set by requiring gravity to be O(1) in the inner region.

Outer region, x = O(1): matching to the inner solution, which behaves like $`\varphi \propto \xi^\ell`$ at small ξ = x/√n, gives

```math
\phi = \varepsilon^{(\ell+1)/2}\,x^\ell\,\Big[G(\theta) + \sum_{k\ge1}\varepsilon^k\Psi_k(\theta,x)\Big], \qquad
b - 1 = 2\ell\,\varepsilon^{\ell+1}x^{2\ell}G^2 + \dots, \qquad c - 1 = O(\varepsilon^{\ell+1}) ,
```

derived [`struct_outer.py`] (ℓ = 1, 2). **Gravity at x = O(1) is weaker than for the scalar**: $`O(n^{-\ell-1})`$ instead of O(1/n).
For ℓ = 0 this is the README's scaling ($`p \propto \varepsilon^{1/2}`$).

### The outer problem (c)

At leading order the centrifugal term, which is O(n) at x = O(1), balances the O(n) radial derivative:
$`n\,c\,(x\phi_x - \ell b\,\phi) = O(1)`$, so $`\phi = x^\ell G(\tau)`$ with G free. Flat space makes this exact: with $`\phi = r^\ell\psi`$,

```math
\phi_{rr} + \frac{n}{r}\phi_r - \frac{\ell(\ell+n-1)}{r^2}\phi = r^\ell\Big(\psi_{rr} + \frac{n+2\ell}{r}\psi_r\Big) ,
```

so the flat ℓ-field in D dimensions is the radial field in D + 2ℓ dimensions (derived [`struct_outer.py`]; also in
LITERATURE.md).

- **ω stays free** (derived to relative order ε³ for ℓ = 1 and ε⁴ for ℓ = 2 [`struct_outer.py`]; the general step argued).
  Order by order $`x^{\ell+1}\partial_x\Psi_k = R_k`$, and $`R_k/x^\ell`$ is an even polynomial starting at x², so $`\Psi_k`$ is an
  even polynomial plus a free $`K_k(\theta)`$, for any G and any ω₀, ω₁, …; the only possible obstruction, an x⁰ term
  in $`R_k/x^\ell`$ (it would give log x), never occurs. The reason is the same as in Proposition 1: every flat term carries
  x² (the x⁰ parts cancel exactly by the D + 2ℓ identity), and gravity enters through b − 1 and c − c(0), which are
  $`O(x^{2\ell})`$. The first term is $`\Psi_1 = K_1 + \tfrac12 x^2(\omega\partial_\theta + \ell)(\omega\partial_\theta + \ell + 1)G`$.
  So, as for the scalar, Δ is not fixed at any finite order of the 1/n expansion.
- **The characteristic structure and the light-cone exponent are unchanged** (derived [`struct_outer.py`]). The
  potential has no derivative, so it cannot enter the indicial equation at the characteristic x = c, however large it is:
  for φ ∝ $`e^{\sigma\tau}`$ the flat exponents at x = 1 are 0 and n/2 − σ for every L (p − m then carries Λ = n/2 − 1 − σ,
  the README's value). The characteristic speeds x = ± c are those of the scalar.

**Proposition 2 carries over** (derived from Proposition 2 and the D + 2ℓ map): the regular flat solution has the
light-cone coefficient C(n + 2ℓ, σ + ℓ), up to an analytic non-vanishing factor,

```math
C_\ell(n,\sigma) = \frac{\Gamma\!\left(\frac{n+1}{2}+\ell\right)\,\Gamma\!\left(\sigma + 1 - \frac n2\right)}{\Gamma\!\left(\frac{1+\sigma+\ell}{2}\right)\Gamma\!\left(\frac{2+\sigma+\ell}{2}\right)} \neq 0 ,
```

finite and non-zero for imaginary σ ≠ 0: no flat self-similar ℓ-wave is analytic at the light cone. In the inner
(Weber) language the Stokes constant is $`\propto 1/\Gamma(1+\sigma+\ell)`$, the scalar's with the index shifted by ℓ.

**No gravitational dressing of the light-cone exponent at D = ∞** (derived from the order of c − 1): for the scalar,
gravity shifts the exponent by n⟨G(1)⟩/2 → ⟨F²⟩ ≈ 3.7 (Proposition 3). Here $`n\,c_x(1) = O(\varepsilon^\ell)`$, so the shift
vanishes at D = ∞ for ℓ ≥ 1, and the fast mode crosses $`0 \lt x \lt 1`$ as in flat space.

### The inner problem (d)

Derived [`struct_inner.py`] from the exact covariant equations at $`r = \sqrt{2ns}`$, leading order in 1/n:

```math
\partial_t\Big(\frac{\varphi_t}{c}\Big) = c\Big(\varphi_s - \frac{\ell\,\varphi}{2s}\Big), \qquad c_s = \frac{2\varphi_t^2}{c}, \qquad
\beta = \frac{4s\,\varphi_t^2}{c^2} + 2\ell\varphi^2 .
```

It is n-free; β = n(b − 1) does not enter. For ℓ = 0 it is the potential form of GAMMA.md's heat system:
$`Y = 2\sqrt2\,\varphi_t/c`$, $`S = 2\sqrt2\,\varphi_s`$ satisfy $`Y_t = cS`$, $`S_t = c(Y_s + Y^3/4)`$, $`c_s = cY^2/4`$ (derived). The
substitution $`\varphi = s^{\ell/2}\psi`$, $`Q = \psi_t/c`$ removes the 1/s term from the field equation and moves it into gravity:

```math
\psi_t = c\,Q, \qquad Q_t = c\,\psi_s, \qquad c_s = 2\,s^\ell\,c\,Q^2 .
```

The flat part is the heat equation for every ℓ; gravity weights it by $`s^\ell`$, the $`r^{2\ell}`$ of the centrifugal energy.

**Symmetries** (derived [`struct_symmetry.py`]: Lie point symmetries with polynomial infinitesimals of total degree ≤ 2 and
≤ 3 in (s, t, ψ, Q, c), solved exactly; ℓ = 0, 1, 2, 3). $`Z = 2s\partial_s + t\partial_t - \ell\psi\partial_\psi - (\ell+1)Q\partial_Q`$ is the self-similar
scaling; a generator X with [Z, X] = −λX gives a mode $`\propto e^{\lambda\tau}`$ of the self-similar solution.

| generator | ℓ = 0 | ℓ ≥ 1 | λ |
|---|---|---|---|
| $`\partial_s`$ (shift of r²) | yes | **no** | 2 |
| $`\partial_t`$ (time translation) | yes | yes | 1 |
| Z (self-similarity) | yes | yes | 0 |
| $`\partial_\psi`$ (ψ → ψ + const, i.e. $`\phi \to \phi + k\,r^\ell`$) | yes | yes | −ℓ |
| t → T(t), c → T′c (gauge; T = 1, t, t², t³ in the ansatz) | yes | yes | 1, 0, −1, −2 |

So **the s-translation, and with it the exact λ = 2 mode, is broken for every ℓ ≥ 1, and nothing replaces it**: the only
positive λ generated by a point symmetry is the time translation, λ = 1, a known mode of every self-similar solution.
The ψ-shift gives a decaying mode (λ = −ℓ; it is the static harmonic $`r^\ell`$). Within the ansatz this is complete; a
non-polynomial or a generalized (non-point) symmetry would not be seen.

**Consequence** (argued; the value is a conjecture): λ∞(ℓ), the growth rate of the D = ∞ critical solution, is the root
of a transcendental condition, the vanishing of the perturbation's Stokes multipliers (GAMMA.md §5), exactly as ω∞ is.
There is no reason for it to equal 2, and **γ∞ ≠ 1/2 is predicted for ℓ ≥ 1** (as a smooth function of a continuous ℓ,
λ∞(ℓ) = 2 + O(ℓ) near ℓ = 0, with a slope that would need the perturbation multipliers). This is directly testable by
computing γ(D) for the ℓ = 1 multiplet at large D: an extrapolation to λ∞ = 2 would contradict it. For orientation only,
in D = 4, γ₁ = 0.119 against γ₀ = 0.376 (OVCU07).

### Flat-space blowup (e)

None: the multiplet is linear in flat space.

## 4. The ℓ-multiplet with ℓ ∝ n

Put ℓ = λn with λ fixed (here λ is not the growth rate). Now the centrifugal term is O(n²) and the field varies on the
scale 1/n at x = O(1): the problem is WKB everywhere. With $`\phi = A\,e^{nS(x)}`$, P = xS′, the O(n²) part of the flat
self-similar equation is (derived [`struct_outer.py`])

```math
(1 - x^2)\,P^2 + P - \lambda(1+\lambda) = 0 .
```

- The eikonal does not contain ω: the time dependence enters first in the O(n) transport equation (derived).
- The branch regular at the centre, P(0) = λ, is analytic through the light cone, P(1) = λ(1 + λ); the other branch carries
  the light-cone singularity (derived).

The two branches meet at a turning point **outside the light cone** (derived):

```math
x_t^2 = 1 + \frac{1}{4\lambda(1+\lambda)} , \qquad x_t \simeq \sqrt{\frac{n}{4\ell}} \ \ (\lambda = \ell/n \to 0) .
```

For ℓ fixed this is the inner region x ∼ √n, where the large parameter is lost; for ℓ ∝ n the coalescence of the fast and
slow branches, which for the scalar happens at x = ∞, happens at the finite point $`x_t`$, and its local problem is Airy's.

- Gravity is ∝ φ² ∝ $`e^{2nS}`$, and S increases along the regular branch, so it is exponentially localized near $`x_t`$:
  relative to $`x_t`$ it is suppressed at the light cone by $`e^{-2n[S(x_t)-S(1)]}`$, with $`S(x_t) - S(1)`$ = 0.110, 0.144, 0.152
  for λ = 1/4, 1, 4 (computed [`struct_outer.py`]). There is no n-free heat-type inner region.

What this means for Δ and γ is **inconclusive**. An argument from orders (argued, not checked): in the turning-point layer
(width $`n^{-2/3}`$) the Airy terms are $`O(n^{4/3})`$ and the time derivatives O(nω); with ω = O(1) the layer is quasi-static, so a
leading-order selection condition would act at each θ separately on the local amplitude, which an oscillating profile
cannot satisfy. That suggests that the time dependence must enter the layer, i.e. ω grows with n ($`\omega \sim n^{1/3}`$ if the
layer balance is kept) and Δ → 0, consistent in spirit with the rapid fall of $`\Delta_\ell`$ with ℓ in D = 4 (OVCU07), but this is
not a derivation. No symmetry fixes γ.

## 5. Yang–Mills

### The scale (b)

κ/e² is a length² in every D, so Einstein–Yang–Mills has the length $`\ell_{\rm YM} = \sqrt{\kappa}/e`$ and no exactly
self-similar solutions. In self-similar variables the Yang–Mills source in the Hamiltonian constraint is
$`\ell_{\rm YM}^2 e^{2\tau}`$ times a τ-free expression (derived [`struct_eqs.py`]). As a configuration shrinks,
gravity becomes strong relative to the field: an O(1) field at radius r has $`(n-1)(b-1) \sim n\,\ell_{\rm YM}^2/r^2`$.

Gundlach's resolution in D = 4 (W = 1 − rS, PRD 55, 6002) works in every D: on scales $`r \ll \ell_{\rm YM}`$ the type II
threshold lives near the vacuum, $`W = 1 - (r/\ell_{\rm YM})\,U`$, and at leading order in $`r/\ell_{\rm YM}`$ (derived [`struct_eqs.py`])

- the field equation is that of the ℓ = 1 multiplet (L = n) plus the metric terms $`-(n-2)(b-1)cU/r^2 + c_rU/r`$;
- the stress tensor has $`\kappa(\rho + P_r) = n\,[U_t^2/\alpha^2 + (U + rU_r)^2/(r^2a^2)]`$ and $`\kappa(\rho - P_r) = 2n(n-1)U^2/r^2`$;
- the Yang–Mills nonlinearity W(1 − W²) has dropped out; only the coupling to gravity is nonlinear.

This "U-system" is exactly scale invariant; its corrections carry positive powers of $`r/\ell_{\rm YM}`$, i.e. decay like
$`e^{-\tau}`$. It differs from the ℓ = 1 multiplet at finite n by the improvement-like term in the stress tensor; in D = 4 the two critical solutions are different
(γ = 0.196 for Yang–Mills, 0.119 for ℓ = 1).

### Outer and inner problems (c, d)

With $`U = n^{-1/2}V`$ the inner limit of the U-system is (derived [`struct_inner.py`])

```math
\partial_t\Big(\frac{V_t}{c}\Big) = c\Big(V_s - \frac{V}{2s}\Big), \qquad c_s = \frac{V_t^2}{c} ,
```

**the ℓ = 1 multiplet's inner problem** (V = √2 φ). The Yang–Mills-specific terms are O(1/n) there; β differs (2V² against
V²) but does not enter at leading order. The outer problem is that of ℓ = 1 (the extra metric terms are O(ε²) at
x = O(1)). So everything in §3 holds with ℓ = 1, and:

- **Prediction** (conjecture, given the selection rule): Δ∞ and γ∞ of the Einstein–Yang–Mills type II solution equal those
  of the ℓ = 1 multiplet; their difference at finite D is O(1/n). γ∞ is not fixed by a symmetry.

The flat Yang–Mills equation itself has the n-free inner limit $`W_{tt} = W_s + W(1-W^2)/(2s)`$ (derived [`struct_inner.py`]), a
nonlinear heat equation with a 1/s potential, solved exactly by the D = ∞ limit of the flat ground-state blowup
(LITERATURE.md §4.2). At x = O(1) the O(n) terms balance alone, $`xW_x + W(1-W^2) = 0`$, so $`W = K/\sqrt{x^2+K^2}`$ with K(τ)
free (elementary): the flat outer problem is nonlinear at leading order, unlike the U-system's.

### Flat-space blowup (e)

Cannot compete (argued). Flat Yang–Mills blows up self-similarly for d ≥ 5 (D ≥ 6) with O(1) amplitude, but with gravity
an O(1) field at radius r has gravitational strength ∼ $`n\,\ell_{\rm YM}^2/r^2`$; a blowup that shrinks to
$`r \lesssim \sqrt n\,\ell_{\rm YM}`$ is inside its own strong-gravity region and forms a black hole of that size. So the
flat blowup ends in black holes with a mass gap; it is not a threshold phenomenon. The type II threshold is the U-system's
critical solution, as in D = 4.

## 6. Wave maps

### Scaling (b)

The field u is an angle, so its amplitude is not free; the dimensionless coupling η = κf²/2 takes the role of the
amplitude. With η fixed as n → ∞, u = O(1) in the inner region and the inner problem is (derived [`struct_inner.py`])

```math
\partial_t\Big(\frac{u_t}{c}\Big) = c\Big(u_s - \frac{\sin 2u}{4s}\Big), \qquad c_s = \frac{2\eta\,u_t^2}{c} ,
```

n-free, with one parameter η. As η → ∞ with u = φ/√η it becomes the ℓ = 1 multiplet's inner problem (derived), the D = ∞
version of the D = 4 statement that the infinite-coupling σ-model is the ℓ = 1 triplet (γ → 0.1185 against γ₁ = 0.119).

### Outer problem (c)

At x = O(1), $`u = \varepsilon^{1/2}x\,[G(\theta) + O(\varepsilon)]`$: the outer problem is the ℓ = 1 multiplet's, with the
target nonlinearity entering at relative order ε (the −(2/3)u³ term, in $`\Psi_1 = K_1 + \tfrac16x^2(6G + 9\omega G' + 3\omega^2G'' - 2G^3)`$).
Formally free through ε³ (derived [`struct_outer.py`]); b − 1, c − 1 = O(ε²); no light-cone shift at D = ∞.

### Inner problem and symmetries (d)

The 1/s term is nonlinear and cannot be removed. The point symmetries (derived [`struct_symmetry.py`]; infinitesimals polynomial of degree ≤ 3, plus
cos 2u and sin 2u times polynomials of degree ≤ 1) are $`\partial_t`$ (λ = 1), the scaling $`Z = 2s\partial_s + t\partial_t - P\partial_P`$ (λ = 0) and the gauge t → T(t);
no $`\partial_s`$ and no ψ-shift. So γ∞ is not fixed by a symmetry (argued), and depends on η (conjecture).

### Flat-space blowup (e)

It competes. The flat ground-state blowup $`u_0 = 2\arctan(x/\sqrt{n-1})`$ (exact for every n, derived) becomes
$`u = 2\arctan(\sqrt{2s}/(-t))`$ at D = ∞, an exact solution of the η = 0 inner problem (derived), and the continuously
self-similar ansatz u = U(ξ), c = C(ξ) is consistent with the inner problem for every η (derived). So the D = ∞ inner
problem contains both kinds of critical behaviour, and which one controls the threshold is decided by η alone
(argued). In D = 4 the threshold is CSS for small η and DSS for η above $`\eta_c \approx 0.17`$ (LITERATURE.md §5.1).
**Prediction** (conjecture): the corresponding thresholds $`\eta_0(D)`$, $`\eta_c(D)`$ tend to finite limits as D → ∞, and the DSS
branch at large η tends to the ℓ = 1 multiplet's Δ∞. One difference from D = 4 is expected: for d ≥ 7 the flat excited
self-similar solutions that serve as CSS critical solutions in D = 4 appear not to exist (BBM17 conjecture that only u₀
exists), so the small-η threshold at large D need not be CSS; this is open.

## 7. Scalars without a potential, and the phase-rotating CSS solution at D = ∞

For the complex scalar, the Hirschmann–Eardley κ-family and the axion–dilaton system the fields are spherically
symmetric scalars with a target metric and no potential, so their inner problem has no explicit s, and the s-translation
survives. For the free complex scalar (two components, each normalized like the solver's scalar) the inner problem is
$`\partial_t(\phi_t/c) = c\,\phi_s`$, $`c_s = 2\lvert\phi_t\rvert^2/c`$, whose point symmetries include $`\partial_s`$ (λ = 2), $`\partial_t`$ (λ = 1),
Z, the shifts and the phase rotation (derived [`struct_symmetry.py`]). So λ = 2 is an exact growth rate of any D = ∞
critical solution of these models, and **γ∞ = 1/2 is predicted** wherever that mode is the relevant one (conjecture; in D = 4
the free complex scalar's CSS solution has more than one unstable mode, so it is not critical for κ = 0, while it is for
κ ≳ 0.075).

**The phase-rotating CSS solution closes on one harmonic.** With y = p − m complex, the README's derivation of the inner
problem applied to each component (η = n(b − 1) is then the sum of the components' p² + m²) gives (argued; it agrees
with the potential form above)

```math
L\,y = c\,\sigma, \qquad L\,\sigma = c\,\xi\,y_\xi + c\,(\eta - 1)\,y, \qquad \xi\,c_\xi = c\,\eta, \qquad \eta = \tfrac12\lvert y\rvert^2, \qquad L = \xi^2\partial_\xi + \omega\xi\,\partial_\theta ,
```

and $`y = e^{i\theta}Y(\xi)`$, $`\sigma = e^{i\theta}S(\xi)`$, c = c(ξ) is exact: a system of ODEs. It is the README's one-harmonic
model with F³ replaced by |F|²F, without truncation (the series coefficients agree: $`g_1 = A^2`$, $`s_0 = 2(1+i\omega)A`$,
$`d_1 = A(2 + 3i\omega - \omega^2) - 2A^3`$). With regular data Y = 2Aξ + …, c(0) = 1, the selection rule of the README
(fast-mode Stokes multiplier = 0) is one complex condition on (A, ω).

**Computed** [`struct_css.py`], by lateral solutions as in `solver/inner_stokes/lateral_np.py` (optimally truncated series
at $`\xi_0 = 0.1\,e^{3i\pi/8}`$, integrated along the ray and the arc; the lower lateral sum by reflection):

- without gravity the normalized jump is 1.000000000002 − 7×10⁻¹²i at r₁ = 0.3 and 0.25, the repository's linear
  check;

With gravity the jump has a zero at

```math
A = 1.9970774009, \qquad \omega_\infty^{\rm CSS} = 3.9789695074, \qquad \Delta = 2\pi/\omega = 1.5790986323 ,
```

the same to all ten digits at every setting tried: start radius r₀ = 0.09–0.11, matching radius r₁ = 0.20–0.33,
integrator tolerance 10⁻¹² and 10⁻¹³. At the zero the normalized jump is about 5×10⁻⁸ for every r₁ in 0.20–0.33,
against 3–8 when A is moved by 0.1%. Beyond |ξ| ≈ 0.335 the arc passes a movable singularity of the upper lateral
solution (the jump at the same point becomes about 10⁵ at r₁ = 0.34), so r₁ is kept below it, as the README keeps
ξ₁ ≤ 0.3. Further:

- scans over 0.5 ≤ A ≤ 2.6, 0.02 ≤ ω ≤ 6 find no other zero with ω > 0 below A ≈ 1.7; above that the jump winds
  rapidly in ω, the grids flag several cells, and the root finder started in five of them went either to the zero above
  or to zeros at ω = 0. A census of zeros at large A was not made;
- the ω = 0 zeros (A = 0.70710678, 1.14729637, 1.48880136, 1.77558388) are static, i.e. non-rotating continuously
  self-similar real-scalar inner solutions. The first is in closed form, $`c = (1-\xi^2)^{-1/2}`$, $`y = \sqrt2\,\xi c`$,
  $`\sigma = \sqrt2\,\xi^2c^2`$ (derived [`struct_inner.py`]; A = 1/√2), with η = ξ²/(1 − ξ²) diverging at ξ = 1, where the
  inner expansion breaks down. Whether these are related to the large-D CSS solutions of Clark and Pimentel
  (arXiv:2504.10590) was not checked.

So, under the selection rule, the D-dimensional Hirschmann–Eardley solution $`\phi = e^{i\omega\tau}f(x)`$ should have
ω(D) → 3.9789695 and Δ = 2π/ω → 1.5790986 as D → ∞, against ω = 1.9154446 at D = 4 (HE95a; Δ = 3.2803). The growth from
D = 4 to D = ∞ parallels the real scalar's (ω = 1.8236 → 3.8221): the ratio of the CSS to the DSS frequency is 1.050 at
D = 4 and 1.041 at D = ∞, and the gravitational strength at the zero, |F|² = A² = 3.988, is close to the real scalar's
⟨F²⟩ ≈ 3.7. Because the CSS ansatz reduces every finite D to an ODE boundary-value problem in x (with two light-cone
conditions and no harmonics), this is the cheapest available test of the D = ∞ selection rule without harmonic
truncation; its first 1/n correction would need the O(1/n) inner equations.

## 8. Other models

- **Scalar with a power-law self-interaction** $`V(\phi) = g\phi^p`$ (argued). With φ dimensionless (geometric units) g
  has dimension length⁻² for every p and D, so the potential is irrelevant on small scales, like a mass; the type II
  critical solution is the massless one in every D. In the inner variables the term is $`g\,t^2\,n^{1-p/2}`$ relative to the others: for p ≥ 2 it
  vanishes as t → 0, and for p > 2 also as n → ∞. Nothing new at large D.
- **Vacuum Bianchi IX in odd D** (not analysed here). LITERATURE.md §6.1 notes that the linearized squashing mode has
  the potential 2(D − 1)B/r², an ℓ = 2 tensor mode, which is O(n) and gives a 1/s term at D = ∞: it belongs to the
  class of §3, so no exact λ = 2 is expected. Only D = 5 and 9 are known.

## 9. Prediction table

| Model | gravity at x = O(1) | D = ∞ inner problem | s-translation | Δ∞ | γ∞ |
|---|---|---|---|---|---|
| real scalar (reference) | O(1/n) | autonomous heat system | yes | 1.6439, selected beyond all orders (README; conjecture) | 1/2 (GAMMA.md) |
| ℓ-multiplet, ℓ ≥ 1 fixed | $`O(n^{-\ell-1})`$ (derived) | heat + ℓφ/(2s), n-free (derived) | **broken** (derived) | finite, ℓ-dependent, selected beyond all orders (outer freedom derived to ε³–ε⁴; finiteness argued; selection conjecture) | **not fixed by a symmetry; generically ≠ 1/2** (symmetry search derived; value conjecture) |
| ℓ = λn | exponentially localized at $`x_t > 1`$ (derived) | none of heat type; Airy turning point at $`x_t`$ (derived) | — | inconclusive; Δ → 0 argued | inconclusive |
| Einstein–Yang–Mills, type II | $`O(n^{-2})`$ (U-system) | = ℓ = 1 multiplet's (derived) | broken | = Δ∞ of ℓ = 1 (conjecture) | = γ∞ of ℓ = 1, ≠ 1/2 generically (conjecture) |
| wave map, η fixed | $`O(n^{-2})`$ (derived) | η-family with sin(2u)/(4s), n-free (derived) | broken (derived) | DSS branch: finite for η above a finite $`\eta_c(\infty)`$, → ℓ = 1 value as η → ∞ (conjecture) | η-dependent, not fixed by a symmetry (argued) |
| complex scalar, κ-family, axion–dilaton | O(1/n) | autonomous heat system (derived) | yes (derived) | phase-rotating CSS: ω∞ = 3.9789695, Δ = 2π/ω∞ = 1.5790986 (computed; selection conjecture) | 1/2 where the λ = 2 mode is the relevant one (conjecture) |
| power-law self-interaction | — | coupling irrelevant on small scales | yes | as the real scalar (argued) | 1/2 (argued) |

## 10. Open issues

- **The value of γ∞ for ℓ ≥ 1** needs the perturbation multipliers of the ℓ-inner problem, i.e. first the ℓ = 1 selection
  itself (the analogue of the README's amplitude expansion with the 1/s term). Nothing here computes it. A cheap first
  step: continuation in a continuous ℓ from ℓ = 0, where λ = 2 and its multiplier derivative are available.
- **The symmetry search is complete only within polynomial infinitesimals of bounded degree** (≤ 3 for the multiplet
  and wave maps, the latter with cos 2u, sin 2u factors on degree ≤ 1; ≤ 2 for the complex scalar). A generalized or non-local symmetry of weight 2 for ℓ ≥ 1 is not
  excluded, though nothing suggests one.
- **ℓ ∝ n** needs the turning-point layer with gravity written out, including how the time dependence enters.
- **Yang–Mills:** that the type II threshold in D ≥ 6 is the U-system's critical solution is argued, not shown; no
  Einstein–Yang–Mills collapse in D ≥ 5 has been computed.
- **Wave maps:** the D = ∞ inner problem as a function of η (its CSS solutions and the DSS branch) was not computed.
- **The CSS check:** the finite-D Hirschmann–Eardley solutions were not computed here; that, and the next order in 1/n
  of the CSS inner problem, would test the D = ∞ selection rule cleanly.

## 11. Scripts

All in [`solver/matter/`](../solver/matter/); run with `nice -n 10 python3 <script>`. Each symbolic script prints
`[ok]`/`[FAIL]` per check and `all ok` at the end; `struct_css.py` prints the numbers. The outputs of the final runs are
saved in [`solver/matter/struct_data/`](../solver/matter/struct_data/) (`*.log`, with `timings.txt`).

| Script | What it checks | Time (2 shared cores) |
|---|---|---|
| `struct_common.py` | helpers: brute-force Einstein tensor, the (K, V) framework, reduced Euler–Lagrange | — |
| `struct_eqs.py` | (a): Einstein tensor n = 2, 3, 4; stress tensors and field equations by brute force (n = 2, 3); D = 4 forms (OVCU, Gundlach); self-similar systems, ℓ = 0 = solver; Yang–Mills scale and U-system | 8–9 min |
| `struct_inner.py` | (b, d): the inner problems of all models from the exact equations; ℓ = 0 = GAMMA.md's heat system; η → ∞ = ℓ = 1; YM-U = ℓ = 1; flat blowup inner limits | seconds |
| `struct_outer.py [N]` | (c): self-similar field equation; flat ℓ ↔ D + 2ℓ; light-cone exponent independent of L; formal outer expansion (ℓ = 1 to ε³, ℓ = 2 to ε⁴, wave map to ε³); ℓ = λn eikonal and turning point | 40 s |
| `struct_symmetry.py [deg]` | (d): Lie point symmetries of the inner systems (multiplet ℓ = 0–3, wave map, complex scalar) and their growth rates | 70 s (degree 3) |
| `struct_css.py check / scan / solve / verify` | §7: the phase-rotating complex scalar at D = ∞: linear check, scans, the zero of the Stokes jump at five settings | about 1 s per evaluation; `verify` 1 min |

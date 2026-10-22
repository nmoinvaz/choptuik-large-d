# Layer A prototype: certifying the inner zero, one harmonic first

This is a prototype of layer A of [PROOF.md](../PROOF.md): a computer-assisted proof that a truncated inner Stokes
multiplier has a simple zero. It starts with the smallest real instance, the one-harmonic model (M = 1, F = A sin θ),
whose multiplier vanishes at A = 2.542148, ω = 2.969777 (0.776994 ω∞). That is not the physical point; it is the test
case. The code is in [`solver/proof/`](../solver/proof/): `certify_M1.py` (driver), `lin_check.py`, `series_arb.py`,
`residual_poly.py`, `fixed_point.py` (rigorous, python-flint ball arithmetic), `krawczyk.py` (the interval-Newton test,
rigorous once it is fed rigorous enclosures; its accuracy budget is a dry run) and `pm_dryrun.py` (double precision, not
rigorous).

**Status in one paragraph.** All four steps run, and the certificate passes (§9). An independent review
([A_review.md](A_review.md), with test scripts in [`solver/proof/review/`](../solver/proof/review/)) reproduced every
input of the final test by separate methods and found no error in any enclosure, but it narrows what is proved. Proved:
with y⁺ continued from the start ray arg ξ = 3π/8 along that ray to radius 0.3 and then along the arc of radius 0.3,
the D-component of the jump at ξ₁ = 0.3, $`J_D = D^+_{+1}(0.3) - \overline{D^+_{-1}(0.3)}`$, has exactly one zero in the box
(A, ω/ω∞) = (2.5421480475, 0.7769940136) ± 10⁻⁹, and it is simple. Two lemmas are needed to turn that into the
statement about the selection condition (review §6a and §6b, both judged fixable):
- **6a: one component at one point is the whole jump.** The template's step 4 (a contraction for the difference
  y⁺ − y⁻, which is one complex Stokes constant times the fast modes) would show J_D(ξ₁) = 0 ⇔ y⁺ ≡ y⁻. In double
  precision every component of the jump vanishes at the certified zero, at ξ₁ = 0.25 and 0.3.
- **6b: the path gives the Borel sum.** y⁺ is defined as a germ at ξ = 0, and the computation continues it along one
  path. The two agree if y⁺ is analytic in the region the path sweeps. That holds numerically below radius 0.3, but y⁺
  has a singularity between radii 0.30 and 0.35, so ξ₁ = 0.3 is near the edge; a validated sweep over arc radii, or the
  lower path through 0.2, would settle it.

With 6b the certificate gives at most one zero of the one-harmonic selection condition in the box; with 6a as well,
exactly one, at A ∈ [2.54214804760473, 2.54214804762839], ω/ω∞ ∈ [0.776994013601194, 0.776994013618254]
(ω ≈ 2.969776709). An independent re-implementation in C++ with CAPD is in progress.

Contents

- [1. The template](#1-the-template)
- [2. The M = 1 inner problem](#2-the-m--1-inner-problem)
- [3. The four steps](#3-the-four-steps)
- [4. What is certified](#4-what-is-certified)
  - [4.1 Linear normalization](#41-linear-normalization)
  - [4.2 Formal series in ball arithmetic](#42-formal-series-in-ball-arithmetic)
  - [4.3 The forcing term](#43-the-forcing-term)
  - [4.4 Step (i): the start-value fixed point](#44-step-i-the-start-value-fixed-point)
  - [4.5 Start data with parameter derivatives](#45-start-data-with-parameter-derivatives)
- [5. What the certificate must reach (dry run)](#5-what-the-certificate-must-reach-dry-run)
  - [5.1 The Krawczyk test and its accuracy budget](#51-the-krawczyk-test-and-its-accuracy-budget)
  - [5.2 The interface the validated integration must provide](#52-the-interface-the-validated-integration-must-provide)
- [6. What is missing, and how to build it](#6-what-is-missing-and-how-to-build-it)
- [7. From M = 1 to M = 5](#7-from-m--1-to-m--5)
- [8. Reproduce](#8-reproduce)

## 1. The template

Baldomá, Capiński, Guardia and Seara, "Breakdown of heteroclinic connections in the analytic Hopf-zero singularity:
rigorous computation of the Stokes constant", J. Nonlinear Sci. (2023), doi:10.1007/s00332-022-09882-x. (also arXiv:2206.13840,
confirmed by a search that returns that arXiv PDF under this title.) Their pipeline, for an inner equation $`\Psi' = A(s)\Psi + S(\Psi, s)`$ with
A(s) = diag(−iα + 1/s, iα + 1/s):

1. **Two solutions on two domains.** ψ± are defined on $`D^\pm_\rho`$ (sectors at s → ±∞ minus a neighbourhood of the
   origin), each characterized by ψ± → 0 at its own infinity.
2. **Fixed point in a weighted space.** ψ± = B±(S(ψ±)), with B± an integral operator built from the fundamental matrix
   of the linear part, in the norm sup |s^ν h(s)|. Explicit bounds give the size M₀ of the forcing and a Lipschitz
   constant L(ρ); choosing ρ with L(ρ) ≤ 1/2 gives a contraction on a ball of radius about 2M₀.
3. **Validated values far out.** Rigorous computation (their Appendix B) gives ψ±(−iρ*) in small intervals at
   ρ* ≈ 16. The truncated asymptotic expansion is used with a remainder bound B/|s|⁶.
4. **The Stokes constant from the difference.** ψ = ψ+ − ψ− solves a linear equation, written as ψ = (Id − G)⁻¹ψ₀
   with operator bounds, so that Σ = κ₀(ρ)(1 + g(ρ)), |g| ≤ M(ρ). Final enclosures reach relative accuracy 10⁻²–10⁻³.

Our problem has the same skeleton. The difference is that we need the multiplier's *zero* in two parameters, not its
non-vanishing, so an interval Newton or Poincaré–Miranda step is added.

## 2. The M = 1 inner problem

Inner equations (README, "The inner problem at large x"), with y = Dd, $`L = \xi^2\partial_\xi + \omega\xi\partial_\theta`$:

```math
L\,y = c\,\sigma, \qquad L\,\sigma = c\,\xi\,\partial_\xi y + c\,\big(\tfrac12 y^2 - 1\big)\,y, \qquad \xi\,\partial_\xi c = \tfrac12\,c\,y^2 .
```

The M = 1 truncation is exactly what `solver/inner_stokes/lateral_np.py` integrates: y and σ on the harmonics μ = ±1,
c on ν = 0, ±2, and every product projected onto its result's space. That is 7 complex unknowns, a 14-dimensional real
system, polynomial in the unknowns and rational in ξ. The irregular singular point is ξ = 0, with u = 1/(2ξ²) and
fast modes $`e^{-u}\xi^{-2s}`$. The formal solution regular at ξ = 0 is $`y = \sum_j d_j\xi^{2j+1}`$,
$`\sigma = \sum_j s_j\xi^{2j+2}`$, $`c = \sum_j g_j\xi^{2j}`$ with d₀ = 2F, g₀ = 1. It is Gevrey-1 in ξ².

**The lateral solution y⁺** is the unique solution asymptotic to the formal series in a sector of the ξ-plane that
contains the ray arg ξ² = 3π/4, where every fast mode is exponentially *large* (Re u < 0), and the real axis. For real
F the solution from below is $`y^-_\mu = \overline{y^+_{-\mu}}`$ on the real axis, so only y⁺ is needed. The jump
$`J_\mu = y^+_\mu - \overline{y^+_{-\mu}}`$ is the Stokes jump. The selection condition is J₁ = 0, in the code
$`G(A, \omega) = R_1\,\xi_1^{-A^2} = 0`$ with R₁ = J₁ normalized by the linear reference.

## 3. The four steps

Mapped onto the template:

- **(i) Enclose y⁺ at the start point ξ₀ = r₀e^{3iπ/8}.** Write y⁺ = y_N + η, with y_N the formal series truncated at
  order N. Then η satisfies L_lin η = −R_N − 𝒢(η). Here:
  - L_lin is the linear, no-gravity operator, whose fundamental solutions are explicit: Kummer U(α, ½, x) and its
    recessive partner per harmonic, with x = −1/(2ξ²), and constants for c;
  - R_N is the residual of y_N;
  - 𝒢 holds the gravity and nonlinear terms, which are O(ξ²)η + O(η²) because c − 1 = O(ξ²) and y² = O(ξ²) near ξ = 0.

  As in the template, invert L_lin by integrating from ξ = 0 (u = ∞) along rays in the sector, in the weighted norm
  $`\lVert\eta\rVert = \sup\lvert\xi\rvert^{-k}\lvert\eta(\xi)\rvert`$. The integration direction is chosen for each
  mode so the kernel is bounded: the fast modes decay toward ξ = 0 on this sector, and the slow ones are power laws. A
  contraction on |ξ| ≤ r₀ needs two things:
  - M₀ = ‖R_N‖, certified here;
  - a Lipschitz constant of 𝒢 composed with the inverse of L_lin below 1/2; it is ≤ 0.292, so this step is now done
    (§4.4, with a simpler splitting of L_lin).
- **(ii) Carry the enclosure to the real axis.** Validated integration of the 14-dimensional system from ξ₀ along the
  ray to |ξ| = r₁ and along the arc |ξ| = r₁ to the real axis (r₀ = 0.1, r₁ = 0.2 in the double-precision code).
- **(iii) The jump.** $`J_1 = y^+_1 - \overline{y^+_{-1}}`$ comes from the same enclosure; there is no second
  integration. Its normalization is certified in the linear case (§4).
- **(iv) The zero.** For existence, apply Poincaré–Miranda to the preconditioned map H = J₀⁻¹G on a box: H₁ must have
  opposite signs on the two A-faces and H₂ on the two ω-faces. Only enclosures of G on the faces are needed, not
  derivatives. For uniqueness and simplicity, apply Krawczyk with an interval Jacobian over the box, obtained by
  carrying (A, ω) as extra state variables through steps (i)–(ii) or by integrating the variational equations. The
  Krawczyk test is implemented and specified in §5.1–5.2.

## 4. What is certified

All items below are rigorous ball-arithmetic computations (python-flint 0.9, arb/acb), at 256–320 bits.

### 4.1 Linear normalization

The check is in `lin_check.py`. In the linear problem (c = 1), harmonic μ has the lateral solutions
$`y^\pm = 2F_\mu\,\xi\,x^{\alpha}U(\alpha, \tfrac12, x)`$, x = −1/(2ξ²), α = (1+s)/2, s = iμω. On the real ξ axis,
y⁺ uses the upper side of U's cut (arg x = π⁻, $`x^\alpha = X^\alpha e^{i\pi\alpha}`$) and y⁻ the lower side
(arg x = −π⁺, $`x^\alpha = X^\alpha e^{-i\pi\alpha}`$, computed as the conjugate of U with conjugated parameters). arb
evaluates U rigorously on both sides. The result: the jump divided by the normalization that `lateral_np.py` uses,
$`2\pi i K'(s) f(\xi_1)\,2F_1`$ with $`K' = 2^{-s-1/2}/(\Gamma(\alpha)\Gamma(\beta))`$ and f the recessive solution,

| ξ₁ | ω/ω∞ | J / (2πi K′ f 2F₁) |
|---|---|---|
| 0.2 | 1 | 1 ± 3×10⁻³⁰ (imaginary part 0 ± 5×10⁻⁶³) |
| 0.2 | 0.777 | 1 ± 3×10⁻³⁰ |
| 0.3 | 1 | 1 ± 10⁻²⁹ |
| 0.3 | 0.777 | 1 ± 3×10⁻³⁰ |

So R₁ = 1 holds exactly in the linear problem, with the orientation (+1) the codes assume. The double-precision check in
`lateral_np.py check` gave 1.000000000002 − 7×10⁻¹²i.

### 4.2 Formal series in ball arithmetic

`series_arb.py` is a port of `lateral.py`'s recursion to acb, for the same
truncated system. A and ω may be balls, so the coefficients enclose the whole parameter box. At the M = 1 zero, at
r₀ = 0.1, 0.08 and 0.06 on the start ray, optimal truncation is at N = 53, 80 and 144 terms, with smallest terms
9×10⁻³⁵, 2×10⁻⁴⁷ and 10⁻⁷⁴.

### 4.3 The forcing term

`residual_poly.py` computes the residual of the truncated series in the three ODEs as an exact ball polynomial in
ξ. At N = 53 its coefficients are zero up to ξ¹⁰⁶ (balls containing 0 that come from rounding) and
nonzero from there on, and for all |ξ| ≤ 0.1 in every direction:

```math
\lvert R_N(\xi)\rvert \le \sum_k \lvert\rho_k\rvert\,\lvert\xi\rvert^k \le 1.0\times10^{-33} .
```

The largest component is the σ equation (I2); the y and c equations are at 4×10⁻³⁶. This is the M₀ of step (i). It is
so small that even a crude contraction constant yields an enclosure of y⁺(ξ₀) far tighter than step (iv) needs.

### 4.4 Step (i): the start-value fixed point

`fixed_point.py` proves the start-value lemma in ball arithmetic at 320 bits. It follows step 2 of the template: an
integral equation built from the linear fundamental solutions, a weighted sup norm, explicit bounds on the inverse
operator and on the Lipschitz constant, and a contraction checked in arb.

**Lemma (start values, M = 1).** Let (A, ω/ω∞) be the dry-run zero x̂ of §5, or any point of the complex box around it of
half-width 3×10⁻² in each parameter (real and imaginary parts). Let Y_N be the formal series truncated at N = 53 terms.
On each ray arg ξ = ±3π/8 with 0 < |ξ| ≤ 0.1, the truncated system has a solution Y with Y − Y_N in an explicit
ball of the weighted space below, and it is the only solution in that ball. Every solution with
$`Y - Y_N = o(\lvert\xi\rvert^{82})`$ along the ray is this one. The lateral solution y⁺ is such a solution on the upper
ray and y⁻ on the lower ray, so Y = y⁺ and Y = y⁻ there. At ξ₀ = 0.1e^{3iπ/8} and x̂,
|D − D_N| ≤ 2.4×10⁻³⁴, |S − S_N| ≤ 6.0×10⁻³⁴ and |C − C_N| ≤ 5.9×10⁻³⁶ on every harmonic. The same holds on every ray with |arg ξ ∓ 3π/8| ≤ 0.2.

The proof follows.

**Variables.** Let z = ξ². On each harmonic set D = ξP, S = zQ and V = Q − (1+s)P, with s = iμω. Let
W = V + [(C−1)Q] and Ê = [P²/2], where [·] projects a product onto its result's harmonics, as in `lateral_np.py`.
For z ≠ 0 the truncated system of §2 is equivalent to

```math
P' = \frac{V + [(C-1)Q]}{2z}, \qquad
V' = \frac{V}{2z^2} + \frac{[(C-1)Q] + [(C-1)W]}{2z^2} + \frac{-(2+s)\,Q - (1+s)\,W + [C\hat E P]}{2z}, \qquad
C' = \tfrac12\,[C\hat E] ,
```

with ′ = d/dz. The only rank-one singular term is V/(2z²). Its solution $`e^{-1/(2z)}`$ is the fast mode, and the slow
fundamental solutions are constants. Every other term is a perturbation: C − 1 = O(z), so it is at most O(1/z) times
the unknowns. This splitting is simpler than the Kummer fundamental matrix of §3, and it makes the inverse operator and
its bounds elementary.

**Integral equation.** Write Y = Y_N + η with η = (p, v, c), and let Φ be the right-hand side minus the principal term
V/(2z²), evaluated at Y_N + η, minus Y_N′. Then Φ = ρ_N + (the change of the right-hand side from Y_N to Y_N + η),
where ρ_N is the residual of the truncated series. Integrating from z = 0 along the ray,

```math
p(z) = \int_0^z \Phi_p(t)\,dt, \qquad
v(z) = \int_0^z e^{\frac{1}{2t} - \frac{1}{2z}}\,\Phi_v(t)\,dt, \qquad
c(z) = \int_0^z \Phi_c(t)\,dt .
```

This is the template's ψ = B(S(ψ)), with B built from the fundamental solutions 1 and $`e^{-1/(2z)}`$.

**The inverse operator.** On the ray arg z = θ with θ = ±3π/4, the fast kernel is bounded:

```math
\Big\lvert e^{\frac{1}{2t} - \frac{1}{2z}} \Big\rvert = e^{-a\left(\frac{1}{\lvert t\rvert} - \frac{1}{\lvert z\rvert}\right)} \le 1 \quad (\lvert t\rvert \le \lvert z\rvert), \qquad a = -\tfrac12\cos\theta = \tfrac{\sqrt2}{4} .
```

With r = |z|, the two integral operators are bounded by

```math
\int_0^r \varrho^m\,d\varrho = \frac{r^{m+1}}{m+1} \quad (m \gt -1), \qquad
\int_0^r e^{-a\left(\frac1\varrho - \frac1r\right)}\varrho^m\,d\varrho \le \frac{r^{m+2}}{a} \quad (m \ge -2) .
```

The second follows from the substitution u = 1/ϱ and $`\varrho^m \le r^{m+2}\varrho^{-2}`$. The fast operator gains two
powers of r at the cost of the factor 1/a. On a ray at angle θ the rate is a = −cos θ/2. It stays positive throughout
π/2 < θ < 3π/2, which is how the sector version holds: on |arg ξ ∓ 3π/8| ≤ 0.2 the smallest rate is a = 0.188.

**Norm.** Let R = r₀² = 0.01 and k = 40. The component norms are

```math
\lVert p\rVert = \sup_{0 \lt r \le R} r^{-k}\lvert p\rvert, \qquad \lVert v\rVert = \sup_{0 \lt r \le R} r^{-k}\lvert v\rvert, \qquad \lVert c\rVert = \sup_{0 \lt r \le R} r^{-k-1}\lvert c\rvert ,
```

for each of the 7 components. The space norm is the weighted maximum $`\lVert\eta\rVert = \max_j w_j\lVert\eta_j\rVert`$,
with the weights taken from the Perron vector of the Lipschitz matrix. Any positive weights are valid; these make the
weighted row sums close to the spectral radius (Collatz–Wielandt).

**Bounds.** Every function on 0 < |t| ≤ R is bounded by a majorant K|t|^q, with K a sum of |coefficient| × R^power.
Sums, products and the harmonic convolutions are bounded by the triangle inequality. The Lipschitz matrix L_ij bounds
the weighted norm of ∂(Tη)_i/∂η_j uniformly over the ball. It uses majorants of the derivatives (the product rule
applied to majorants) and the mean-value inequality for holomorphic maps on the convex ball. Every bound is an arb upper
bound, and every bound is non-decreasing in R.

**The forcing.** The numerators of the residual ρ_N are polynomials in z. Their coefficients below order 53 (p and v)
and 52 (c) are exactly zero: they are combinations of the order-by-order equations that the recursion solves, and they
involve only the coefficients that were kept. The computation checks that each of these balls contains 0. It bounds the
rest and pushes them through the inverse operator, which gives F = ‖T(0)‖.

**Contraction.** Let Λ be the Lipschitz constant of T on the ball of radius ϱ. The check is Λ < 1 and F + Λϱ < ϱ. Then
T maps the ball into itself and contracts, so it has exactly one fixed point in the ball.

| case | N | contraction Λ ≤ | forcing F | ball ϱ | F + Λϱ |
|---|---|---|---|---|---|
| the zero, rays arg ξ = ±3π/8 | 53 | 0.2918 | 1.32×10⁴⁸ | 7.43×10⁴⁸ | 3.48×10⁴⁸ |
| complex box ±10⁻⁵(1+i) | 53 | 0.2918 | 8.13×10⁶⁷ | 4.59×10⁶⁸ | 2.15×10⁶⁸ |
| complex box ±10⁻³(1+i) | 53 | 0.2923 | 8.57×10⁶⁹ | 4.84×10⁷⁰ | 2.27×10⁷⁰ |
| complex box ±10⁻²(1+i) | 53 | 0.2960 | 1.38×10⁷¹ | 7.83×10⁷¹ | 3.69×10⁷¹ |
| complex box ±3×10⁻²(1+i) | 53 | 0.3048 | 1.17×10⁷² | 6.73×10⁷² | 3.22×10⁷² |
| the zero, every ray \|arg ξ ∓ 3π/8\| ≤ 0.2 | 53 | 0.5107 | 1.32×10⁴⁸ | 1.08×10⁴⁹ | 6.80×10⁴⁸ |
| the zero, r₀ = 0.08 | 80 | 0.1951 | 2.93×10⁴³ | 1.46×10⁴⁴ | 5.78×10⁴³ |

F and ϱ are in the weighted norm, so they carry the factor $`R^{-k} = 10^{80}`$. In absolute terms the ball at the zero is
|η| ≤ 5.3×10⁻³² in the (p, v, c) variables. The estimate "≈ 0.3" of the earlier draft is now the bound 0.292. Its
largest entries are the couplings of v to p (0.67 and 0.26 before weighting), of v to itself (0.20) and of v to c
(0.15). Unweighted, the v rows sum to 1.47; the weights bring the largest weighted row sum down to 0.292. On boxes, N and the vanishing orders come from the box centre. That is valid because the vanishing
is algebraic and holds for every parameter value at fixed N.

**Uniqueness and identification.** Every bound is non-decreasing in R, so for each R′ ≤ R the contraction also holds on
(0, R′]. The operator is of Volterra type, so the fixed point on (0, R] restricted to (0, R′] is the fixed point there.
Any solution with p, v = o(rᵏ) and c = o(rᵏ⁺¹) satisfies the integral equation. The boundary terms vanish because
$`e^{1/(2\epsilon)} \to 0`$ as ε → 0 along the ray, where the fast mode is exponentially large. Its weighted norm on (0, R′] tends
to 0, so for small R′ it lies in the ball and equals the fixed point. By uniqueness for the regular ODE it then equals
the fixed point on all of (0, R]. The lateral solution y⁺ is asymptotic to the formal series on a sector containing the
upper ray (§2; Braaksma's 1-summability theorem, as in PA3), so $`y^+ - Y_N = O(z^{53})`$, and y⁺ is this solution. The lower
ray gives y⁻ in the same way. The two enclosures obey $`D_{-1}(\bar\xi_0) = \overline{D_{+1}(\xi_0)}`$ and the analogous
identities for S and C, as the real-F symmetry requires.

**Enclosures at the start point** (the zero, ξ₀ = 0.1e^{3iπ/8}; each value ± the stated radius in its real and its
imaginary part):

| | value | radius |
|---|---|---|
| D₋₁ | −0.25456359271845 + 0.11388094341079i | 2.4×10⁻³⁴ |
| D₊₁ | 0.23063658495062 − 0.12113752692877i | 2.4×10⁻³⁴ |
| S₋₁ | −0.083670851983953 + 0.040323898358046i | 6.0×10⁻³⁴ |
| S₊₁ | −0.023851921967651 + 0.074016029211036i | 6.0×10⁻³⁴ |
| C₋₂ | 0.012219201192828 − 0.012291277836186i | 3.0×10⁻³⁶ |
| C₀ | 0.97718752662789 + 0.024774249371824i | 5.9×10⁻³⁶ |
| C₊₂ | 0.010661568640759 − 0.012165794858459i | 3.0×10⁻³⁶ |

The digits shown are rounded for display. The code keeps the 320-bit midpoints.

### 4.5 Start data with parameter derivatives

The Krawczyk test (§5.1) needs ∂Y/∂(A, ω/ω∞) at ξ₀ over a parameter box X, because the variational equations start
there. `fixed_point.start_data` returns them in `lateral_np.py`'s state order [D₋₁, D₊₁, S₋₁, S₊₁, C₋₂, C₀, C₊₂],
built from two parts:

- **Y_N and its derivatives.** These are exact, from first-order jets: forward-mode dual numbers in (A, ω/ω∞) carried
  through the same recursion, over X. They agree with finite differences of the double-precision solver, for example
  ∂D₊₁/∂A = 0.0926793 − 0.0578402i at the zero.
- **η.** Its values are bounded by the fixed point over X. Its derivatives are bounded by Cauchy's estimate (below).

The contraction holds uniformly on the complex box X + c(1+i) with c = 10⁻³, so η is holomorphic in the parameters
there, as a uniform limit of holomorphic Picard iterates. That box contains the disc of radius c around each point of
X, so

```math
\lvert \partial_{x_i}\eta \rvert \le \frac{1}{c}\,\sup_{X + c(1+i)} \lvert\eta\rvert, \qquad x = (A,\ \omega/\omega_\infty) .
```

| box X (real half-width) | sup \|η\| over X, (p, v, c) variables | Cauchy part of ∂D/∂x, ∂S/∂x, ∂C/∂x |
|---|---|---|
| the zero (a point) | 5.3×10⁻³² | 1.6×10⁻⁹, 4.0×10⁻⁹, 3.8×10⁻¹¹ |
| 10⁻⁵ | 1.6×10⁻¹² | 1.6×10⁻⁹, 4.1×10⁻⁹, 3.9×10⁻¹¹ |
| 10⁻³ | 1.7×10⁻¹⁰ | 2.4×10⁻⁹, 6.2×10⁻⁹, 5.9×10⁻¹¹ |

Over a box the value bound on η is inflated by ball arithmetic in the residual: the forcing over a box is 10²⁰–10²²
times the forcing at the point. That does no harm. The test needs values only at the box centre, where they are known to
6×10⁻³⁴, and derivatives over the box, whose added uncertainty at ξ₀ is below 10⁻⁸. That is far inside the budget at
ξ₁ = 0.3 (10⁻² in ∂J_D, §5.1). At ξ₁ = 0.2 (8×10⁻⁷) it fits only if the flow from ξ₀ amplifies it by less than about 100.

## 5. What the certificate must reach (dry run)

`pm_dryrun.py`, in double precision with the lateral solver, so **not rigorous**:

- **The zero:** A = 2.5421480475, ω/ω∞ = 0.7769940136 (ω = 2.96977671). Newton stalls at |G| ≈ 2×10⁻⁸, the
  double-precision noise floor.
- **Jacobian:** J₀ = ∂G/∂(A, ω/ω∞) has singular values 303 and 87, so it is well conditioned.
- **Poincaré–Miranda on H = J₀⁻¹G**, sampled at 9 points per face:

| box half-width h | min face margin | G must be enclosed to |
|---|---|---|
| 10⁻³ | 9.7×10⁻⁴ (all four faces > 0) | 0.076 |
| 10⁻⁵ | 1.0×10⁻⁵ (all four faces > 0) | 7.8×10⁻⁴ |

On this scale R₁ = 1 without gravity, so a 10⁻³ box needs the multiplier to roughly 10⁻² relative and a 10⁻⁵ box
to 10⁻³. That matches what the template achieved for its Stokes constant. These targets are relative to the jump, which
is exponentially smaller than the lateral solutions; §5.1 converts them into absolute targets for the integration.

### 5.1 The Krawczyk test and its accuracy budget

`krawczyk.py` implements step (iv) as an interval-Newton test. One test gives existence, uniqueness and simplicity of
the zero.

**The map.** Let x = (A, ω/ω∞) ∈ ℝ² with box centre x̂, and define

```math
g(x) = \kappa\,J_D(x), \qquad J_D(x) = D^+_{+1}(\xi_1; x) - \overline{D^+_{-1}(\xi_1; x)}, \qquad
\kappa = \frac{\xi_1^{-\hat A^2}}{2\pi i\,K'(s)\,f(\xi_1)\,2F_1}\bigg|_{x = \hat x} ,
```

read as a map ℝ² → ℝ² through its real and imaginary parts. κ is a non-zero constant, computed rigorously in arb as in
`lin_check.py`. So g vanishes exactly where J₁(ξ₁) = 0, which is the selection condition of §2. The dry-run map of §5 is
$`G = R_1\,\xi_1^{-A^2} = \varphi(x)\,g(x)`$, with φ smooth, non-vanishing and φ(x̂) = 1. So at a zero at x̂ the Jacobians of g and G
coincide, and the dry-run Jacobian is the right planning proxy. The test needs no derivative of the normalization or
of $`\xi_1^{-A^2}`$. For real x, the derivative of the conjugate is the conjugate of the derivative.

**The Jacobian over the box.** Carry the parameters through the validated integration as constant state variables
(x′ = 0), together with the 2 × 14 real partials ∂Y/∂x, which obey the variational equations

```math
\frac{d}{d\xi}\,\partial_x Y = D_Y\mathcal F(\xi, Y; x)\,\partial_x Y + \partial_x\mathcal F(\xi, Y; x) .
```

Start them from the data of §4.5 and integrate with x ranging over the box X. The result encloses
$`\partial_x D^+_{\pm1}(\xi_1)`$ for every x ∈ X, and κ times its jump combination is J(X) ⊇ {Dg(x) : x ∈ X}.

**The operator, the box and the conditions.** Let X = x̂ + [−h, h]² and let Y be any real 2 × 2 matrix; the code takes
the inverse of the midpoint of J(X), rounded to floats. Define

```math
K(X) = \hat x - Y\,g(\hat x) + \big(I - Y\,J(X)\big)(X - \hat x) .
```

If K(X) ⊂ int X, then g has exactly one zero in X and Dg is non-singular there (Krawczyk 1969; Moore 1977; Neumaier,
*Interval Methods for Systems of Equations*, 1990). The inputs are g at the single point x̂ and J over the whole box.
Write g(x̂) ∈ g₀ ± δ_g and J(X) ⊆ J₀ ± (V_h + δ_J), where V_h is the true variation of the Jacobian over X (about the
Hessian times h) and δ_J is the excess of the enclosure. Take Y = J₀⁻¹. Then, componentwise,

```math
\lvert K(X) - \hat x\rvert \le \lvert Y\rvert\,\big(\lvert g_0\rvert + \delta_g\big) + \lvert Y\rvert\,\big(V_h + \delta_J\big)\,h\,\mathbf 1 ,
```

and the test passes when the right-hand side is less than h in both components.

**Budget.** These numbers come from the double-precision dry run at the 10-digit zero (`pm_dryrun.py`), so they are
**not rigorous**; they say what each stage must reach. Each tolerance assumes the other enclosure is exact; in practice
split it, for example half each.

| | ξ₁ = 0.2 | ξ₁ = 0.3 |
|---|---|---|
| singular values of J₀ | 302.7, 87.2 | 117.1, 85.8 |
| normalization N(x̂) = 2πi K′ f 2F₁ (rigorous) | −4.699×10⁻⁴ + 4.970×10⁻⁴i | 0.8281 + 0.0212i |
| \|κ\| (rigorous) | 4.81×10⁷ | 2.89×10³ |
| variation V_h of J over X, h = 10⁻³ | ≤ 3.15 | ≤ 1.59 |
| tolerated radius of g(x̂), h = 10⁻³ | 7.4×10⁻² | 6.6×10⁻² |
| tolerated radius of g(x̂), h = 10⁻⁵ | 7.8×10⁻⁴ | 6.8×10⁻⁴ |
| tolerated excess δ_J per entry of J(X) | 37–39 | 33–34 |
| **what the integration must reach:** | | |
| radius of J_D(x̂), absolute, h = 10⁻³ | 1.5×10⁻⁹ | 2.3×10⁻⁵ |
| radius of J_D(x̂), absolute, h = 10⁻⁵ | 1.6×10⁻¹¹ | 2.4×10⁻⁷ |
| excess of ∂J_D/∂x over X, absolute | 8×10⁻⁷ | 1.2×10⁻² |

**Match at ξ₁ = 0.3.** At ξ₁ = 0.3 the absolute targets are about 15 000 times weaker. The jump is exponentially small
next to the lateral solutions: |N(x̂)| is 7×10⁻⁴ at ξ₁ = 0.2 against 0.83 at 0.3 (the factor $`e^{-1/(2\xi_1^2)}`$ is
$`e^{-12.5}`$ against $`e^{-5.6}`$), while D⁺ itself is of order one (|D₊₁| ≈ 0.26 already at |ξ| = 0.1). §5's
"10⁻² relative" targets are relative to the jump. At ξ₁ = 0.2 they mean 10⁻⁹ absolute in each D⁺, and 10⁻¹¹ for a
10⁻⁵ box, which is demanding for a double-precision integrator on a stiff path. At ξ₁ = 0.3 they mean 2×10⁻⁵
(2×10⁻⁷), which is comfortable. The stiffness is also milder at 0.3: on the arc the fast modes go from $`e^{3.9}`$ to
$`e^{-5.6}`$, instead of from $`e^{8.8}`$ to $`e^{-12.5}`$. The zero does not move with ξ₁:
|g(x̂)| is at the double-precision floor at both radii (4×10⁻⁸ and 10⁻⁸).

**Mock validation.** There are no rigorous integration enclosures yet, so the test runs on mocks: the dry-run g₀ and J₀,
with radii set to fractions of the budget.

| h | g radius | J radius | centre | K(X) ⊂ int X | margins (ξ₁ = 0.3) |
|---|---|---|---|---|---|
| 10⁻³ | 0.25 × budget | 0.25 × budget | at the zero | yes | 4.9×10⁻⁴, 6.2×10⁻⁴ |
| 10⁻³ | 2.5 × budget | 0.25 × budget | at the zero | no | −1.7×10⁻³, −9.6×10⁻⁴ |
| 10⁻⁵ | 0.25 × budget | 0.25 × budget | at the zero | yes | 5.0×10⁻⁶, 6.4×10⁻⁶ |
| 10⁻⁵ | 0.25 × budget | 0.25 × budget | moved 3h along A | no | −2.5×10⁻⁵, 6.4×10⁻⁶ |

The results are the same at ξ₁ = 0.2. This validates the code path of the test; it does not certify the zero.

### 5.2 The interface the validated integration must provide

The validated integration (step ii) is being built separately. This is what the Krawczyk step needs from it.

**Input (from `fixed_point.start_data`).**

- At the centre x̂, as point balls: the start state Y(ξ₀) at ξ₀ = 0.1e^{3iπ/8}, in `lateral_np.py`'s order
  [D₋₁, D₊₁, S₋₁, S₊₁, C₋₂, C₀, C₊₂], with radius 6×10⁻³⁴.
- Over the real box X: Y(ξ₀) and the partials ∂Y/∂A and ∂Y/∂(ω/ω∞) at ξ₀, valid for every x ∈ X. Their uncertainty is
  below 10⁻⁸ on top of the true variation over X.

Call it as `start_data(A, wf, r0="0.1", theta_sign=+1, cauchy=arb("1e-3"))`, where A and wf are real balls created at
320 bits or more (decimal strings are safest for the centre; a ball made at 53 bits is a box of width 10⁻¹⁶). The parameter is ω/ω∞, with ω∞ = 2π/1.64389375 as in the solver. The lemma also holds on every
ray with |arg ξ − 3π/8| ≤ 0.2, so the start angle can move if the integration prefers another one.

**Output (two runs of the validated integration).** Both use the path from ξ₀: the ray arg ξ = 3π/8 out to |ξ| = ξ₁,
then the arc |ξ| = ξ₁ down to the real axis, with ξ₁ = 0.3.

1. A point run at x̂, returning enclosures of D⁺₊₁(ξ₁) and D⁺₋₁(ξ₁). The radius of J_D must be ≤ 2×10⁻⁵ absolute for
   a box of half-width 10⁻³, or ≤ 2×10⁻⁷ for 10⁻⁵, so each value needs half of that.
2. A box run over X with the variational equations, returning enclosures of ∂D⁺_{±1}(ξ₁)/∂A and ∂D⁺_{±1}(ξ₁)/∂(ω/ω∞)
   valid for all x ∈ X. The excess must be ≤ 10⁻² beyond the true variation (about 5×10⁻⁴ for h = 10⁻³).

If the integration stays at ξ₁ = 0.2, the targets become 1.5×10⁻⁹ (1.6×10⁻¹¹) and 8×10⁻⁷.

**Assembly.** With x̂ and h as arb values:

```
N, xs = krawczyk.normalization(xhat, "0.3")
g_hat, JX = krawczyk.g_from_states(Dm1, Dp1, [dDm1_dA, dDm1_dwf], [dDp1_dA, dDp1_dwf], N, xs)
ok, K, margins = krawczyk.krawczyk(xhat, g_hat, JX, [h, h])
```

All values are python-flint acb balls. If `ok`, the M = 1 multiplier has exactly one zero in X, and it is simple.

## 6. What is missing, and how to build it

1. **The contraction constant (step i): done** (§4.4). Splitting off only the rank-one principal part makes the
   inverse operator elementary, so the Kummer fundamental matrix is not needed. The contraction constant is ≤ 0.292 at
   r₀ = 0.1 (0.195 at r₀ = 0.08), against the earlier estimate of 0.3. r₀ = 0.1 stays the start radius. The start data
   with parameter derivatives are in §4.5.
2. **Validated integration (step ii).** This is the binding difficulty, because the system is stiff.
   - **Along the ray** arg ξ² = 3π/4 from 0.1 to 0.2, the fast modes shrink by about e⁻³¹. Along the arc they go from
     e⁺⁹ to e⁻¹² relative to the slow ones. At the recommended ξ₁ = 0.3 (§5.1) the ray is longer but the arc is
     milder, e⁺⁴ to e⁻⁶.
   - **Naive intervals fail.** A naive interval Taylor method overestimates by exp(∫|λ|) instead of exp(∫Re λ), about
     e⁵⁰ on the ray.
   - **What works:** a Lohner-type method that propagates the set as c + Φr, with Φ the enclosed flow Jacobian (QR
     re-orthogonalization), captures the contraction.
   - **Options:** CAPD's C¹ Lohner integrator on the 14-dimensional real system in double or long-double precision
     (enough at ξ₁ = 0.3, where §5.1 needs 2×10⁻⁵ absolute in the jump for a 10⁻³ box; tight at ξ₁ = 0.2, where it
     needs 10⁻⁹), or a small arb Taylor–Lohner integrator (python-flint acb_poly
     arithmetic; polynomial right-hand side, ξ(t) linear on the ray and r₁e^{it} on the arc).
   - **Step-size guide:** the double-precision DOP853 path needs about 10³ steps at tolerance 10⁻¹²; a Taylor method of
     order 20–30 needs far fewer.
3. **Parameter boxes.** For Poincaré–Miranda, run (i)–(ii) on about 4 × 10 sub-boxes of the faces, with (A, ω) as balls
   (`series_arb.py` already accepts balls). For Krawczyk, add (A, ω) as constant state variables so the Lohner
   Jacobian contains ∂/∂(A, ω); §5.2 specifies the input, the output and the radii.

Rough effort: (1) is done; (2) is the main work, about 1–2 weeks with CAPD or about 2–3 weeks for a custom arb Lohner. The run itself is small: each G evaluation is about 1.5 s in double precision.

## 7. From M = 1 to M = 5

- **Size:** 19 complex unknowns (y, σ on |μ| ≤ 5, c on |ν| ≤ 6), a 38-dimensional real system. Lohner integration in
  that dimension is routine.
- **Conditioning is the new issue.** The higher harmonics leak into the lower conditions with weights
  $`e^{\pi(\mu-\nu)\omega/2}`$ (∂r₁/∂B₅ ≈ 10¹⁷ in `hier.py`). A Krawczyk test on the raw 10-dimensional system
  (A, ω, F₃, F₅) would be hopeless. It should be posed on the de-mixed, Schur-corrected conditions, the same reduction
  `hier.py --schur` uses: enclose the higher-harmonic unknowns at their zeros by an inner interval Newton, then certify
  the outer 2 × 2.
- **Accuracy needed:** enclosures of the higher-harmonic jumps to about one part in 10¹⁷ relative, so multiprecision
  (arb) rather than double. The outer 2 × 2 is well conditioned (singular values about 6.8×10³ and 3.9×10³), and a
  10⁻³ box needs r₁ to about 4 (`solver/proof/feasibility.py`).
- **Beyond M = 5:** certifying the untruncated zero needs a rigorous bound on the harmonic tail (gap (b) of PROOF.md);
  that is not a computer-assisted step.

## 8. Reproduce

```
cd solver/proof
python3 certify_M1.py          # sections 4.1-4.4 (rigorous), about 1 min
python3 certify_M1.py --pm     # plus the Poincare-Miranda dry run of section 5 (double precision), about 2 min
python3 residual_poly.py 0.08  # the forcing bound at another start radius
python3 fixed_point.py                  # step (i): the start-value contraction at the zero, both rays (rigorous), seconds
python3 fixed_point.py --box=1e-3,1e-3  # the same, uniformly over a complex parameter box of half-width 1e-3 (1+i)
python3 fixed_point.py --sector=0.2     # the same on every ray |arg xi -+ 3pi/8| <= 0.2
python3 fixed_point.py 0.08             # at start radius 0.08 (N = 80)
python3 krawczyk.py --xi1=0.3           # Krawczyk budget (dry run) and the mock validation of the test, about 15 s
```

## 9. Step (ii): the validated integration (agent VI)

`validated_int.py` integrates the truncated M = 1 system of §2 in ball arithmetic (python-flint arb at 256 bits). It
starts from the step-(i) data of §4.5 and produces the two runs §5.2 asks for. Its `krawczyk` mode then calls the
assembly of §5.2 unchanged: `krawczyk.normalization` → `g_from_states` → `krawczyk`.

**Result.** At ξ₁ = 0.3 the Krawczyk test of §5.1 passes on

```math
X = \hat x + [-10^{-9}, 10^{-9}]^2, \qquad \hat x = (2.5421480475,\ 0.7769940136) .
```

K(X) lies in the interior of X, with margins 7.9×10⁻¹⁰ and 9.2×10⁻¹⁰ (out of 10⁻⁹). So $`J_D(0.3)`$, the D-component of
the jump at ξ₁ = 0.3 with y⁺ continued along the ray-then-arc path, has exactly one zero in X, and it is simple. That it
is the zero of the whole M = 1 selection condition J₁ ≡ 0 needs two lemmas, 6a and 6b of the review
([A_review.md](A_review.md); see the status paragraph at the top). The same runs pass the test on the smaller box
x̂ ± 1.3×10⁻¹⁰ (review §4), which encloses the zero, with endpoints rounded outward, in

```math
A \in [2.54214804760473,\ 2.54214804762839], \qquad \omega/\omega_\infty \in [0.776994013601194,\ 0.776994013618254] .
```

(An earlier version quoted A = 2.542148047617 ± 9.1×10⁻¹¹, which misses the lower end of K(X) by 4×10⁻¹³; review §4.)

Every input is rigorous:

- step (i), the start data of §4.4–4.5;
- this integration;
- κ, from `krawczyk.normalization`.

The certificate rests on the correctness of the code. The review ([A_review.md](A_review.md)) re-derived every input
independently (start values to 5×10⁻³⁵, the point run to 10⁻³¹, one Lohner step against a 50-digit solution, K(X) with
separate code) and found no error; a second implementation, in CAPD, is in progress.

The box is much smaller than the 10⁻³ and 10⁻⁵ boxes §5.1 budgets for. The point values are far inside every budget.
The obstacle is the derivative enclosures over a box: their excess grows like 10⁶ h (§9.3–9.4).

### 9.1 Method

**Path and state.** The path is the one of §5.2, parametrized by t ∈ [0, 1] on each leg:

```math
\xi(t) = \big(r_0 + (\xi_1 - r_0)\,t\big)\,e^{3i\pi/8} \ \text{(ray)}, \qquad
\xi(t) = \xi_1\,e^{3i\pi(1-t)/8} \ \text{(arc)}, \qquad r_0 = 0.1,\ \xi_1 = 0.3 .
```

Every step is a multiple of 2⁻³⁰, so every node t is exact in binary; r₀, ξ₁ and π are arb balls. The state is the 7
complex unknowns in `lateral_np.py`'s order [D₋₁, D₊₁, S₋₁, S₊₁, C₋₂, C₀, C₊₂], plus ω and A as constant states. ω enters
the right-hand side; A enters only through the start data. So the step Jacobian contains the ω-derivatives.

**The set.** A QR-Lohner set (Lohner 1987; Nedialkov and Jackson) in real coordinates: the state is
c + (Qq)₁…ₙ + i(Qq)ₙ₊₁…₂ₙ, with c a complex point, Q a real 18 × 18 matrix and q a real box. One step of size h:

1. **A-priori enclosure.** An enclosure Ỹ of every trajectory from the set over [t, t + h], by interval Picard
   iteration with t as a ball. The step is halved when it fails, and grows by 5/4 after each success.
2. **Taylor parts, order 24.** The Taylor polynomial of the trajectory from c, and that of the variational matrix along
   the trajectories from the whole set. Both come from Picard iteration on power series (acb_series); the variational
   coefficients use the exact recursion $`V_{k+1} = \frac{1}{k+1}\sum_{j \le k} A_{k-j} V_j`$.
3. **Remainders.** The order-25 coefficient of the trajectory, evaluated on Ỹ. For the Jacobian, the order-25
   coefficient of the fundamental matrix along Ỹ, times an a-priori enclosure of the variational matrix.
4. **Update (mean value form).** Let φ be the step map and J the enclosure of its Jacobian over the whole set. Then
   φ(c + Qq) ⊂ φ(c) + JQq. Set c′ = mid φ(c). Q′ is the Q factor of a QR factorization of mid(JQ), computed in floating
   point and then used as an exact matrix. Its inverse is enclosed rigorously. Then q′ = (Q′⁻¹JQ)q + Q′⁻¹(φ(c) − c′).

The interval Taylor arithmetic only ever acts on the centre trajectory and on J. The set itself moves linearly. So the
stiff fast modes, which decay along both legs, only shrink q, and nothing wraps by $`\exp\int\lvert\lambda\rvert`$.

Two details decide whether the method works on this problem:

- **Real coordinates.** A complex ball is a rectangle, so a complex Q and q inflate under every phase rotation. Real
  coordinates remove that source of wrapping. Here it made no measurable difference.
- **Pivoting.** The columns of mid(JQ) are pivoted by their norm times the size of the matching q component (Lohner's
  rule), not by norm alone. With norms alone, the start of the arc at ξ₁ = 0.3 grew the enclosures by about 15% per
  step, and by e¹⁸ over the arc. The flow there is strongly non-normal: the real parts of the eigenvalues move from
  +10 to −14 per unit t within Δθ = 0.1. With Lohner's rule the growth is about 2–3% per step, close to the true
  expansion rate.

**Point run.** The set starts as the step-(i) enclosure of y⁺(ξ₀; x̂), radius about 5×10⁻³², from
`fixed_point.start_data` (§4.5).

**Box run (C¹).** Each parameter derivative v = ∂(state)/∂x_k, with x = (A, ω/ω∞), is carried as a Lohner set of its
own. Every trajectory from the box has its step Jacobian inside J, so v′ ∈ J c_V + (JQ_V)q_V, with the same QR update
and no remainder. The two columns start from `start_data` over X, as (∂y/∂A, 0, 1) and (∂y/∂(ω/ω∞), ω∞, 0) in the rows
(y, ω, A).

The state set over the box is never propagated by itself. At every step it is rebuilt from the mean value theorem as

```math
y(t; x) \in y(t; \hat x) + V(t; X)\,(x - \hat x), \qquad x \in X ,
```

with y(t; x̂) from the centre trajectory's Lohner set and V(t; X) the current derivative enclosure. J is then computed
over that set. Propagating a C⁰ Lohner set over the box instead gave almost the same widths here. Both diverge for
boxes of 10⁻⁴ and wider (§9.3).

### 9.2 Checks

- **Linear problem.** Without gravity the jump is known in closed form (Kummer U, `lin_check.py`). The integrator gives
  R₁ = 1 ± 5×10⁻¹⁷ at ξ₁ = 0.2, from a start ball of 10⁻³⁰ (377 steps, 24 s). Its D₁(ξ₁) overlaps the exact value.
- **Double precision.** The validated D⁺ agree with `lateral_np.py` to its accuracy (§9.3).
- **Gravity point at ξ₁ = 0.2.** An earlier run started from the truncated series plus an assumed ball of 10⁻³⁰. It
  enclosed J_D to 3×10⁻¹⁹ and G to 1.2×10⁻¹¹. The double-precision G there, about 4×10⁻⁸, is only that solver's noise
  floor.

### 9.3 Results at ξ₁ = 0.3

All runs use the path of §5.2 (ray to 0.3, then the arc). Each run takes 743 steps and about 4 minutes on one core.
Radii are the larger of the real and the imaginary radius.

**Point run at x̂** (`kpoint_job`):

| quantity | enclosure | radius | budget (§5.1, h = 10⁻⁵) |
|---|---|---|---|
| D⁺₋₁(ξ₁) | 0.094871263087140 + 0.403634942280012i | 3×10⁻¹⁹ | 1.2×10⁻⁷ |
| D⁺₊₁(ξ₁) | 0.094871263088194 − 0.403634942283502i | 3×10⁻¹⁹ | 1.2×10⁻⁷ |
| J_D(x̂) | 1.054740×10⁻¹² − 3.490429×10⁻¹²i | 4.1×10⁻¹⁹ | 2.4×10⁻⁷ |
| g(x̂) = κ J_D, \|κ\| = 2889.9 | 2.78871×10⁻⁹ − 1.01616×10⁻⁸i | 1.2×10⁻¹⁵ | 6.8×10⁻⁴ |

- **Double precision.** `lateral_np.py` at rtol 10⁻¹³ agrees to 9×10⁻¹⁶, and its J_D = 1.0535×10⁻¹² − 3.4908×10⁻¹²i.
- **The dry-run centre.** g(x̂) ≠ 0 is now proved. The dry-run zero is accurate to about 10⁻¹⁰, which matches
  |J₀⁻¹ g(x̂)|.
- **Growth along the path.** The state radius grows from 5×10⁻³² at ξ₀ to 1×10⁻¹⁹ at the end of the ray, and ends at
  3×10⁻¹⁹.

**Box runs** (`kbox_job`): the derivative enclosures over X = x̂ + [−h, h]², and the Krawczyk test.

| h | radius of ∂D⁺₋₁/∂(A, ω/ω∞) | radius of ∂D⁺₊₁/∂(A, ω/ω∞) | radius of ∂J_D (budget 1.2×10⁻²) | K(X) ⊂ int X |
|---|---|---|---|---|
| 10⁻³ | diverges on the ray (t ≈ 0.35) | | | — |
| 10⁻⁴ | diverges at the start of the arc | | | — |
| 10⁻⁵ | 60, 79 | 38, 49 | ≈ 100 | no |
| 10⁻⁶ | 0.67, 0.86 | 0.31, 0.39 | ≈ 1 | no |
| 10⁻⁷ | 0.064, 0.081 | 0.029, 0.037 | ≈ 0.1 | no |
| 10⁻⁸ | 6.4×10⁻³, 8.1×10⁻³ | 2.9×10⁻³, 3.7×10⁻³ | ≈ 10⁻² | no, narrowly (see below) |
| 10⁻⁹ | 6.4×10⁻⁴, 8.1×10⁻⁴ | 2.9×10⁻⁴, 3.7×10⁻⁴ | 0.9×10⁻³, 1.2×10⁻³ | **yes**: margins 7.9×10⁻¹⁰, 9.2×10⁻¹⁰ |

From 10⁻⁶ down the excess is linear, about 7×10⁵ h in ∂D⁺₋₁ and 3.5×10⁵ h in ∂D⁺₊₁. At 10⁻⁵ it is already
superlinear, because the wide derivative enclosures widen the state set and so J.

The 10⁻⁸ test failed with margins −7×10⁻⁹ and −5×10⁻¹⁰. It read its enclosures back from a lossy printout, which
tripled the ∂J_D radii; that is now fixed. Without the loss the ∂J_D radius is about 10⁻², right at the budget. It was
not rerun.

At h = 10⁻⁹:

```math
J(X) = \begin{pmatrix} -15.32 \pm 2.74 & -103.12 \pm 3.47 \\ 90.86 \pm 2.74 & -44.09 \pm 3.47 \end{pmatrix} .
```

The midpoints match the dry-run J₀ (singular values 117 and 86). The radii are about a tenth of the tolerated 33.

**A lower path.** Take the ray to 0.2, the arc at 0.2, then the real axis from 0.2 to 0.3 (`--rho=0.2`). It cuts the
excess 3–4 times: at h = 10⁻⁷ the radii are 0.015 and 0.030 for ∂D⁺₋₁, and 0.005 and 0.010 for ∂D⁺₊₁. So h = 10⁻⁸
should pass on that path. It has not been run.

### 9.4 What remains

1. **Larger boxes.** The obstacle is the excess of the derivative enclosures, not the point values.
   - **Where it comes from.** At every step the derivative columns are multiplied by J, which is enclosed over the
     whole state set. Its width is the genuine variation of the Jacobian across the box. That variation is large
     because the solution itself is large near ξ = 0.3e^{3iπ/8}: in double precision |S₋₁| ≈ 6.8 and
     |∂S₋₁/∂(ω/ω∞)| ≈ 35 there, against about 0.4 on the real axis.
   - **Why it does not cancel.** The per-step widths add in absolute value. The true ∂J_D, a difference of conjugate
     partners, varies little: V_h ≈ 5×10⁻⁴ for h = 10⁻³ (§5.1).
   - **Remedies**, in order of promise:
     - **A C² Lohner step.** Enclose the second variational equations, so J over the set comes in mean value form
       around the centre trajectory. The excess then becomes second order in h, and the midpoints keep the
       conjugate-pair cancellation.
     - **A path through smaller |y|.** The lower path gains 3–4 times; an arc nearer 0.15 should gain more.
     - **Sub-box covering.** `--sub=k` runs k² boxes of half-width h/k and takes the union. That suits factors of a
       few, not 10⁶.
0. **The two lemmas of the review** (6a: J_D(ξ₁) = 0 implies the whole jump vanishes, by the template's step 4;
   6b: the ray-then-arc continuation is the Borel sum, by a validated sweep over arc radii and a start domain reaching
   the real axis). These are what stands between the certificate and the statement about the selection condition.
2. **An independent check of the code** (done by review, [A_review.md](A_review.md); CAPD re-implementation in
   progress). The certificate now depends on `validated_int.py` together with
   `fixed_point.py` and `krawczyk.py`. A second implementation or a careful review should come next. CAPD's C¹ Lohner
   on the 18-dimensional real system is the natural one. Points to audit:
   - the remainder terms of steps 3–4 of §9.1;
   - the exact dyadic time grid;
   - the mean value start set.
3. **The C⁰ route.** Poincaré–Miranda with a C⁰ mean value enclosure of G over the faces, as in §5, is not workable.
   At ξ₁ = 0.2 a face of half-width 3×10⁻⁶ in A gave H radii of 2.9 against the required 1 (`boxtest`): a C⁰ set over
   the box loses the cancellation in the jump. The C¹ Krawczyk route replaces it.
4. **Matching at 0.2.** Not needed. For reference, the point run there encloses J_D to 3×10⁻¹⁹ (§9.2), but the
   derivative budget at 0.2 is 8×10⁻⁷, 15 000 times tighter.

**Reproduce.**

```
cd solver/proof
python3 validated_int.py krawczyk 1e-9                 # the certificate: point run + box run (2 processes) + Krawczyk, ~5 min
python3 validated_int.py krawczyk 1e-9 --only=point --out=p.json      # the runs separately, saved as JSON
python3 validated_int.py krawczyk 1e-9 --only=box --out=b.json
python3 validated_int.py krawczyk 1e-9 --point=p.json --box=b.json    # the test from saved runs, seconds
python3 validated_int.py krawczyk 1e-9 --point=validated_int_krawczyk_1e-9.json --box=validated_int_krawczyk_1e-9.json
                                                       # re-check the saved certificate runs (JSON next to the script), seconds
python3 validated_int.py krawczyk 1e-8 --rho=0.2       # next: the lower path, expected to pass at 1e-8
python3 validated_int.py linear                        # the linear check of 9.2 (xi1 = 0.2)
```

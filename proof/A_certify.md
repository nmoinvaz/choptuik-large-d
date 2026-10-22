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
(ω ≈ 2.969776709). An independent re-implementation in C++ with the CAPD library (§10), which shares no code with
§9 beyond the step-(i) start data, confirms the certificate. It also certifies the larger box ± 10⁻⁸, puts the zero
in A ∈ [2.54214804761512, 2.54214804761800], ω/ω∞ ∈ [0.77699401360886, 0.77699401361058], and passes along the
lower path through radius 0.2, away from the singularity.

**Since then (§11):** lemma 6a is proved, and so is the part of 6b near ξ = 0 (y⁺ analytic for |ξ| ≤ 0.0624 in the
sector, by a start-value lemma on a domain that reaches the real axis). The certificate is redone on the lower path,
where 6b only needs |ξ| ≤ 0.2. What remains is one lemma, 6b₁: y⁺ is analytic on the annular sector
0.0624 ≤ |ξ| ≤ 0.2, 0 ≤ arg ξ ≤ 3π/8 for every x in the box. Given 6b₁, the theorem holds: **the M = 1 selection
condition has exactly one zero in the box, and it is simple**, at A ∈ [2.54214804761338, 2.54214804761975],
ω/ω∞ ∈ [0.776994013607438, 0.776994013612010] (§11.7). 6b₁ holds numerically (two paths agree to 2×10⁻¹²); a
validated sweep of the annulus is the remaining computation (§11.4 lists three routes).

**Since then (§12):** lemma 6b₁ is proved. The annulus is covered by 178 validated Taylor discs (radius 10–24 in
u = 1/ξ², orders 40–70). Their centre values come from single-trajectory Lohner runs along the start ray and 12 arcs,
and the disc functions glue to one analytic continuation of y⁺. With §11 this gives **Theorem A(M = 1)** without
conditions, given Braaksma's theorem for the existence of y⁺ (§12.6): the M = 1 selection condition has exactly one
zero in the box, it is simple, at A ∈ [2.54214804761338, 2.54214804761975], ω/ω∞ ∈ [0.776994013607438, 0.776994013612010].

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
- [9. Step (ii): the validated integration (agent VI)](#9-step-ii-the-validated-integration-agent-vi)
  - [9.5 Mutation controls: the test can fail](#95-mutation-controls-the-test-can-fail)
- [10. Independent check with CAPD (agent IC)](#10-independent-check-with-capd-agent-ic)
  - [10.1 What was built](#101-what-was-built)
  - [10.2 How independent it is](#102-how-independent-it-is)
  - [10.3 Results at ξ₁ = 0.3](#103-results-at-ξ₁--03)
  - [10.4 Evidence on lemmas 6a and 6b](#104-evidence-on-lemmas-6a-and-6b)
  - [10.5 Reproduce](#105-reproduce)
- [11. Lemmas 6a and 6b](#11-lemmas-6a-and-6b)
  - [11.2 The certificate on the lower path](#112-the-certificate-on-the-lower-path)
  - [11.5 Lemma 6a](#115-lemma-6a)
  - [11.7 The resulting theorem, and what remains](#117-the-resulting-theorem-and-what-remains)
- [12. Lemma 6b₁](#12-lemma-6b₁)
  - [12.2 The disc lemma](#122-the-disc-lemma)
  - [12.6 Theorem A(M = 1)](#126-theorem-am--1)

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
- **The point certificate at M = 3** ([A_M.md](A_M.md)). The same validated integration and Krawczyk test, on the
  M + 1 = 4 real D-jump conditions at ξ₁ = 0.3 (path: ray 5π/16 from 0.0741, arc at 0.2, real axis), show that the D-jump
  function has exactly one zero in a parallelepiped of outer half-width 10⁻¹⁰ about (A, ω/ω∞, B₃, C₃) =
  (2.7264758856, 1.0156030949, 0.1444194785, 0.0110874259), with a non-singular Jacobian there. A separate assembly in
  mpmath intervals gives the same verdict; at M = 1 the same code reproduces Theorem A(1)'s box. The M = 5 box runs are a
  local-machine task (A_M.md §7). The code has not been reviewed independently.
- **The lemmas around the point certificate** ([A_lemmas_M.md](A_lemmas_M.md)). Step (i), lemma S, lemma 6b₀, the
  flat estimate of 6a and the disc lemma of 6b₁ carry over to M harmonics with the same proofs. The fast eigenvalue
  repeats across harmonics, so the inverse operator, its domains and the kernel bounds are those of M = 1. The labels
  enter through s_μ = iμω in the O(1/z) terms. On rays their weight cancels exactly, since |(t/z)^{s_μ}| = 1, and a
  second splitting of the fast row ("diag") removes them from the contraction. On lemma S's paths they remain
  perturbations of size |s_μ|R₊. Step (i) is certified at M = 3 and 5 at `hier.py`'s zeros, at points and over boxes,
  with start data in all M + 1 parameters (`fixed_point_M.py`); M = 1 reproduces `fixed_point.py` bit for bit.
  At M = 3, lemma S, lemma 6b₀ and the flat estimate of 6a are certified over x̂ ± 10⁻⁶, and lemma 6b₁ is certified
  on the point certificate's box p̂ ± 1.1×10⁻⁹ by a covering with 183 discs in a C¹ mean-value form. The M = 1
  covering method fails on the innermost row at M = 3. The backward step of 6a with the certificate's state box at
  ξ₁ = 0.3 passes (ρ₆ₐ = 0.041), so **Theorem A(3) is certified**, assuming Braaksma's theorem (A_lemmas_M.md §7). At M = 5, lemma S, 6b₀ and the flat estimate are certified, and
  6b₁ is a local-machine task.

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
separate code) and found no error. A second implementation, in C++ with CAPD (§10), confirms the certificate and
extends it to the box ± 10⁻⁸.

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
2. **An independent check of the code.** Done twice:
   - the review ([A_review.md](A_review.md)) re-derived every input;
   - the CAPD re-implementation of §10 shares no code with `validated_int.py` or `krawczyk.py`. Its J_D(x̂)
     agrees with §9's to 10⁻²⁸, it passes the same test with slightly larger margins, and passes at h = 10⁻⁸ as well.
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

### 9.5 Mutation controls: the test can fail

Until now the Krawczyk test had only been seen to pass. A check that has never been seen to fail is untested, so
`mutation_controls.py` mutates the *function*, reruns the full assembly from the saved runs, and compares each verdict
with one written down before the controls were first run. (A mutation of J(X) alone is not a meaningful control: the
test's soundness assumes J(X) encloses the true derivative, so a wrong J(X) can make it pass or fail either way.) The
design follows the planted-truth discipline that the BootLoops review recommended ([notes/BOOTLOOPS.md](../notes/BOOTLOOPS.md)).

| control | mutation | expected | result |
|---|---|---|---|
| C0 | none (clean twin) | pass | pass, margins 7.925×10⁻¹⁰, 9.247×10⁻¹⁰ |
| C1 | jump sign flipped, D⁺₊₁ + conj(D⁺₋₁) | fail | fail, margins −59, −22 |
| C2 | conjugation dropped, D⁺₊₁ − D⁺₋₁ | fail | fail, margins −2.0, −0.35 |
| C3 | equivalent form −conj(J_D) (null) | pass, same zero | pass; K overlaps C0's |
| C4 | normalization N → 2N (null) | pass, same K | pass; K identical to 30 digits |
| C5 | zero planted outside the box, moved by (2h, 0) | fail | fail, margin −1.2×10⁻⁹ in A |
| C6 | zero planted inside, moved by (h/2, −3h/10) | pass, zero recovered | pass; K contains C0's zero + d and excludes C0's zero + d + (h/4, 0) |
| C7 | point enclosure of D⁺₊₁(ξ₁) widened by 10⁻⁹ | fail | fail, margins −4×10⁻⁸ |

A planted zero is made by subtracting L d from g, with L the exact binary midpoint of J(X): that moves the zero by d
up to the second-order remainder, and leaves J(X) unchanged. All 8 controls behave as expected (`python3
mutation_controls.py`, under a second, exit status 0 only if every verdict matches). The controls were designed after
the certificate had passed, which weakens them somewhat; but their expected verdicts follow from the construction, not
from the real run's output.

## 10. Independent check with CAPD (agent IC)

`solver/proof/capd_check/` redoes steps (ii)–(iv) in C++ with CAPD (Computer Assisted Proofs in Dynamics), a published
library for validated integration of ODEs. It shares no code with `validated_int.py` or `krawczyk.py`. The only input it
takes from the Python proof is the step-(i) start data of §4.5, and those are checked against an independent evaluation
of the formal series.

**Result.** The CAPD check confirms the certificate of §9, in the narrowed form of the review
([A_review.md](A_review.md)):

> For x = (A, ω/ω∞) in X = (2.5421480475, 0.7769940136) ± h, the function
> $`J_D(x) = D^+_{+1}(0.3) - \overline{D^+_{-1}(0.3)}`$, with y⁺ continued from the ray arg ξ = 3π/8 along that ray
> to radius 0.3 and then along the arc of radius 0.3, has exactly one zero in X, and it is simple.

It proves this for h = 10⁻⁹, the box of §9, with margins 8.7×10⁻¹⁰ and 9.8×10⁻¹⁰, and for h = 10⁻⁸, which §9 could
not reach, with margins 8.7×10⁻⁹ and 9.3×10⁻⁹. Its enclosures agree with §9's: the two enclosures of J_D(x̂)
intersect, and their midpoints differ by less than 10⁻²⁸. The zero lies in

```math
A \in [2.54214804761512,\ 2.54214804761800], \qquad \omega/\omega_\infty \in [0.77699401360886,\ 0.77699401361058]
```

(endpoints rounded outward), inside the enclosure of §9 and eight to ten times narrower.

Lemmas 6a and 6b are still needed to turn this into a statement about the selection condition. §10.4 gives evidence on
both: rigorous bounds on the other jump components at the zero, and the same certificate along the lower path.

### 10.1 What was built

| file | content |
|---|---|
| `capd_check.cpp` | the M = 1 vector field, the path, the two validated integrations, the Krawczyk test, the mutation controls |
| `export_start.py` | the one call into the Python proof: `fixed_point.start_data` (and κ from `krawczyk.normalization`), written to `data/` as outward-rounded double and double-double intervals in exact hex |
| `series_check.py` | an independent check of the exported start values against the formal series (mpmath, 60 digits) |
| `compare.py` | compares saved point runs with each other and with §9's saved run |
| `CMakeLists.txt`, `build.sh` | `build.sh` clones CAPD at the pinned commit, builds the library, then builds `capd_check` |
| `results/` | the logs of the runs below, the saved point runs, and the Krawczyk inputs as exact hex bounds (`capd_enclosures_*.json`) |

**CAPD** is free software under the GNU General Public License v3. It is not vendored: `build.sh` clones
<https://github.com/CAPDGroup/CAPD> at commit `37057e5899bccaa6cd5701057b7e6ea5e232ef8b` (version 6.1.0) and builds
only the library and filib, with MPFR support (about 5 minutes on 2 cores). `capd_check` links against it.

**The vector field.** It is written in `capd_check.cpp` from the equations of §2, not from `validated_int.py`, and
cross-checked against `lateral_np.py`'s `rhs`. With D_μ, S_μ (μ = ±1) and C_ν (ν = 0, ±2), and every product projected
onto its result's harmonics,

```math
D_\mu' = \frac{[CS]_\mu - i\mu\omega\xi D_\mu}{\xi^2}, \qquad
S_\mu' = \frac{[C\,\xi D']_\mu + [(CE)\,D]_\mu - [CD]_\mu - i\mu\omega\xi S_\mu}{\xi^2}, \qquad
C_\nu' = \frac{[CE]_\nu}{\xi}, \qquad E = \tfrac12 [DD] .
```

The state is the 14 real parts and imaginary parts plus ω/ω∞, which is constant. A enters only through the start data.
The complex arithmetic is spelled out on CAPD's automatic-differentiation nodes, so CAPD computes the Taylor
coefficients and the Jacobians itself. The parameters are intervals: ω∞ = 2π · 160000/263023 (since
1.64389375 = 263023/160000 exactly), 0.1, 0.3, 3π/8 and its cosine and sine.

**The path.** One real time s ∈ [0, 2] covers the path of §5.2:

```math
\xi(s) = (0.1 + 0.2\,s)\,e^{3i\pi/8} \ \ (0 \le s \le 1), \qquad
\xi(s) = 0.3\,e^{3i\pi(2-s)/8} \ \ (1 \le s \le 2),
```

with dY/ds = F(ξ(s), Y) ξ′(s). One set object is carried through both pieces. CAPD chooses the steps.

**The point run, at x̂.** It uses multiprecision intervals (MPFR, 128 bits), a C⁰ doubleton Lohner set (`MpC0Rect2Set`),
Taylor order 40 and step tolerance 10⁻²⁸. It starts from the step-(i) enclosure of y⁺(ξ₀; x̂), exported as
double-double bounds, about 10⁻³³ wide. In double precision the same run ends with radius 1.2×10⁻¹⁰ (§10.3), too wide
for g(x̂) at h = 10⁻⁹. The per-step rounding is about 10⁻¹⁵, and the flow amplifies it by up to 10³ along the path.

**The box run, over X = x̂ + [−h, h]².** It uses double-precision intervals and CAPD's C¹ Lohner set (`C1Rect2Set`,
Zgliczyński's C¹ algorithm). That set carries the state and the monodromy matrix Dφ, each in doubleton form, at Taylor
order 20. The start set is the mean value form of the step-(i) data over X:

```math
z_0(x) \in z_0(\hat x) + \partial_x z_0(X)\,(x - \hat x), \qquad z_0 = (y^+(\xi_0),\ \omega/\omega_\infty),
```

written as x + C r₀, where C holds the midpoints of the two derivative columns, r₀ = x − x̂, and the rest goes into
the interval vector x. Then ∂z(ξ₁)/∂x ∈ Dφ(start set) · ∂z₀/∂x(X) by the chain rule. Dφ is enclosed over the whole
start set.

**Correlations.** J_D is linear in the state: $`\mathrm{Re}\,J_D = \ell_0\cdot y`$ and
$`\mathrm{Im}\,J_D = \ell_1\cdot y`$. It is evaluated twice. The first evaluation uses the interval hull. The second
uses CAPD's doubleton representations and applies ℓ before the interval products: x + C r₀ + B r for the state, and
D + C R₀ + B R for Dφ. Both are enclosures, and the code takes their intersection. The second keeps part of the
correlation between D⁺₊₁ and D⁺₋₁.

**The Krawczyk test.** It is written anew in `capd_check.cpp`. The test runs twice, on two maps with the same zeros:

- F = (Re J_D, Im J_D), which needs no normalization at all;
- g = κ J_D, with κ from `krawczyk.normalization`, to compare with §9.

Y is the double inverse of mid J(X), used as an exact matrix. The test needs K(X) ⊂ int X. x̂ is the exact decimal,
enclosed in a double interval of width at most 4.4×10⁻¹⁶. The start data cover x̂ ± h(1 + 2⁻¹⁶), so they contain X
with room to spare.

After a pass, the code shrinks the box to x̂ ± h′ with h′ = 0.95ᵏh while the test still passes. J(X) encloses the
derivative over every sub-box, so the last K(X′) is a sharper enclosure of the same zero.

### 10.2 How independent it is

| | §9 (`validated_int.py`, `krawczyk.py`) | §10 (`capd_check`) |
|---|---|---|
| step-(i) start data | `fixed_point.start_data` | the same, exported; checked against an independent series (below) |
| vector field | `validated_int.Sys`, Python | `capd_check.cpp`, from §2; Taylor coefficients by CAPD's automatic differentiation |
| arithmetic | arb balls, 256 bits | MPFR intervals, 128 bits (point); filib double intervals (box) |
| integrator | own QR-Lohner, own a-priori enclosure and remainders, order 24, dyadic steps | CAPD's C⁰ and C¹ doubleton Lohner algorithms, its own enclosure and step control |
| derivatives over X | variational columns as Lohner sets, state set rebuilt by the mean value theorem | CAPD's C¹ set from a mean value start set; chain rule at the end |
| path parametrization | t ∈ [0, 1] per leg | one time s ∈ [0, 2] |
| Krawczyk test | `krawczyk.krawczyk`, arb | new code, CAPD intervals; also on F, without κ |

What is shared:

- the M = 1 equations of §2;
- the step-(i) lemma and its code (`fixed_point.py`), which the review checked (R1, R2, R7);
- κ, which only the second, κ-scaled test uses.

**The start data.** `series_check.py` re-derives the formal series from §2 alone, with its own coefficient recursion
(mpmath, 60 digits, N = 53 terms). At ξ₀ = 0.1e^{3iπ/8} it lies inside every exported interval:

- the double-double bounds of the values, which are step (i)'s balls rounded outward (half-widths 5×10⁻³⁵ to
  3×10⁻³³, η included);
- the derivative bounds (Cauchy radii 1.6×10⁻⁹ to 4×10⁻⁹), for the series' own derivatives by central differences.

### 10.3 Results at ξ₁ = 0.3

All runs start from the same exported step-(i) data and use the path of §5.2. Radii are the larger of the real and the
imaginary radius. The logs are in `results/`.

**Point run at x̂** (`MpC0Rect2Set`, 128 bits): 205 steps, 112 on the ray and 93 on the arc. The state radius grows
from 10⁻³³ to 8×10⁻²⁵ at the end of the ray and to 9×10⁻²⁴ at ξ₁.

| quantity | CAPD (§10) | radius | §9 | radius |
|---|---|---|---|---|
| D⁺₋₁(ξ₁) | 0.094871263087139723 + 0.40363494228001201i | 6×10⁻²⁴ | 0.094871263087140 + 0.403634942280012i | 3×10⁻¹⁹ |
| D⁺₊₁(ξ₁) | 0.094871263088194463 − 0.40363494228350244i | 4×10⁻²⁴ | 0.094871263088194 − 0.403634942283502i | 3×10⁻¹⁹ |
| J_D(x̂) | 1.054739676292166×10⁻¹² − 3.490429414033870×10⁻¹²i | 6.3×10⁻²⁴ | the same digits | 4.1×10⁻¹⁹ |
| g(x̂) = κ J_D | 2.78871×10⁻⁹ − 1.01616×10⁻⁸i | 1.9×10⁻²⁰ | 2.78871×10⁻⁹ − 1.01616×10⁻⁸i | 1.2×10⁻¹⁵ |

The two enclosures of J_D(x̂) intersect, and their midpoints differ by 4.5×10⁻²⁹ and 6.6×10⁻²⁹ (`compare.py`).

The same run in double precision (`C1Rect2Set`, order 20, 311 steps) agrees with these to about 10⁻¹⁶, but its radii
are 5–8×10⁻¹¹, and 1.2×10⁻¹⁰ in J_D. That is too wide for h = 10⁻⁹: g = κJ_D with |κ| = 2890 then has radius
3.5×10⁻⁷, against the 7×10⁻⁸ that §5.1's budget allows at that h. It is enough for h = 10⁻⁸ (below).

**Box runs** (`C1Rect2Set`, double, order 20, 311 steps, about a minute each). They give the derivative enclosures over
X, and then the test.

| h | radius of ∂D⁺₋₁/∂(A, ω/ω∞) | radius of ∂D⁺₊₁/∂(A, ω/ω∞) | radii of J(X) for g = κJ_D | K(X) ⊂ int X | margins (of h) |
|---|---|---|---|---|---|
| 10⁻⁹ | 1.0×10⁻⁴, 1.2×10⁻⁴ | 6.4×10⁻⁵, 7.4×10⁻⁵ | 0.29–0.48 | **yes** | 8.72×10⁻¹⁰, 9.83×10⁻¹⁰ |
| 10⁻⁸ | 9.9×10⁻⁴, 1.2×10⁻³ | 6.3×10⁻⁴, 7.2×10⁻⁴ | 2.8–4.7 | **yes** | 8.73×10⁻⁹, 9.30×10⁻⁹ |
| 10⁻⁷ | 1.0×10⁻², 1.2×10⁻² | 6.3×10⁻³, 7.3×10⁻³ | 29–48 | no | −1.7×10⁻⁸, 3.0×10⁻⁸ |
| §9, 10⁻⁹ | 6.4×10⁻⁴, 8.1×10⁻⁴ | 2.9×10⁻⁴, 3.7×10⁻⁴ | 2.7–3.5 | yes | 7.9×10⁻¹⁰, 9.2×10⁻¹⁰ |

- **Agreement.** The midpoints of J(X) are those of §9: −15.32, −103.12, 90.86, −44.09.
- **Where it stops.** At 10⁻⁷ the radii of J(X) (29–48) reach the tolerance of about 33 that §5.1 estimates, and the
  test fails.
- **Tighter enclosures.** The derivative enclosures are 5–7 times tighter than §9's at the same h, and J(X) 7–10
  times. The excess is again linear in h. That is why h = 10⁻⁸ passes here, where §9's run failed narrowly.
- **With or without κ.** The test on F = (Re J_D, Im J_D) and the test on g = κJ_D give the same margins to three
  digits (8.7173 and 8.7153×10⁻¹⁰ in A).
- **All in double.** With the double-precision point value instead (`--prec=0`), h = 10⁻⁸ still passes, with
  margins 4.2×10⁻⁹ and 6.7×10⁻⁹. It fails at 10⁻⁹, as it must with a g(x̂) radius of 3.5×10⁻⁷. So the 10⁻⁸
  certificate does not even need the multiprecision run.
- **The zero.** The derivative enclosure over X also holds on every sub-box x̂ ± h′. The test passes down to
  h′ = 1.22×10⁻¹⁰, and that K(X′) is the enclosure quoted in the Result, with radii 1.4×10⁻¹² in A and 8.5×10⁻¹³
  in ω/ω∞.

**Mutation controls** (as in §9.5; each changes the function, never an enclosure, and the expected verdict follows from
the construction). All five give the expected verdict at both h:

| control | expected | h = 10⁻⁹: margins | h = 10⁻⁸: margins |
|---|---|---|---|
| C1 jump sign flipped, D⁺₊₁ + conj(D⁺₋₁) | fail | −59, −22 | −59, −22 |
| C2 conjugation dropped, D⁺₊₁ − D⁺₋₁ | fail | −2.0, −0.35 | −2.0, −0.35 |
| C5 zero planted outside, moved by (2h, 0) | fail | −1.1×10⁻⁹, 9.8×10⁻¹⁰ | −1.1×10⁻⁸, 9.3×10⁻⁹ |
| C6 zero planted inside, moved by half the clean margins | pass | 4.4×10⁻¹⁰, 5.1×10⁻¹⁰ | 4.4×10⁻⁹, 4.7×10⁻⁹ |
| C7 J_D(x̂) widened by 10³h | fail | −4.3×10⁻⁵, −3.1×10⁻⁵ | −4.3×10⁻⁴, −3.1×10⁻⁴ |

A planted zero is F − L d with L the exact midpoint of J(X). It shifts K(X) by Y L d, which is d up to rounding, so a
shift by half the clean margins must still pass.

### 10.4 Evidence on lemmas 6a and 6b

**The other jump components at the zero (6a).** The point run also encloses every jump component at x̂, with radii of
about 10⁻²³:

```math
J_S = S^+_{+1} - \overline{S^+_{-1}}, \qquad J_{C_2} = C^+_{+2} - \overline{C^+_{-2}}, \qquad
J_{C_0} = C^+_0 - \overline{C^+_0} = 2i\,\mathrm{Im}\,C^+_0 .
```

The box run encloses their derivatives over X. The zero z lies in K(X′) ⊂ X, so the mean value theorem gives
$`J_k(z) \in J_k(\hat x) + \partial_x J_k(X)\,(K(X') - \hat x)`$. From the 10⁻⁹ run:

| component | at x̂ | at the certified zero |
|---|---|---|
| J_D | 1.05×10⁻¹² − 3.49×10⁻¹²i | \|Re\| ≤ 5.1×10⁻¹⁴, \|Im\| ≤ 7.7×10⁻¹⁴ (contains 0, as it must) |
| J_S | 1.93×10⁻¹² − 1.38×10⁻¹¹i | \|Re\| ≤ 1.7×10⁻¹³, \|Im\| ≤ 2.2×10⁻¹³ |
| J_C₂ | −8.0×10⁻¹⁴ − 8.0×10⁻¹⁴i | \|Re\| ≤ 2.6×10⁻¹⁴, \|Im\| ≤ 2.6×10⁻¹⁴ |
| J_C₀ | 7.4×10⁻¹⁴ i | \|Im\| ≤ 3.2×10⁻¹⁴ |

Every component is consistent with vanishing at the zero of J_D. J_S, the largest, is at most 1/50 of its value at x̂,
a point only 1.2×10⁻¹⁰ away. These are rigorous bounds, not a proof of 6a. The C components are too small at x̂
(about 10⁻¹³) for the bounds to say much. This agrees with the double-precision finding of the review (R6), now with
rigorous bounds at ξ₁ = 0.3.

**The lower path (6b).** `--rho=0.2` runs the path of the review's fix 3: the ray to 0.2, the arc at 0.2, then the
real axis from 0.2 to 0.3 (s ∈ [2, 3]). That path stays away from the singularity of y⁺ between radii 0.30 and 0.35.

- **The point run** (176 steps) encloses all eight real jump functionals at ξ₁ = 0.3, with radii up to 6×10⁻²². Each
  enclosure intersects the standard path's, and their midpoints agree to 6×10⁻³⁰ (`compare.py`). J_D(x̂) agrees with
  §9's to 7×10⁻²⁹.
  - Overlapping enclosures do not prove that the two continuations are equal.
  - But if a singularity lay between the paths, going around it would have to change all eight functionals by less
    than 10⁻²². The review found that continuations on either side of the singularity beyond 0.3 differ by 82.
- **The certificate on the lower path** passes at h = 10⁻⁹ (margins 8.32×10⁻¹⁰, 9.51×10⁻¹⁰) and at h = 10⁻⁸ (margins
  4.77×10⁻⁹, 6.12×10⁻⁹). Its zero lies in A ∈ [2.54214804760992, 2.54214804762320],
  ω/ω∞ ∈ [0.77699401360470, 0.77699401361475], which contains the standard path's.
- **What this changes for 6b.** On the lower path, 6b only needs y⁺ analytic on the sector 0 < |ξ| ≤ 0.2,
  0 ≤ arg ξ ≤ 3π/8. The last piece, along the real axis from 0.2 to 0.3, is part of the path itself. That sector lies
  well inside the radius of the singularity.
- **Derivative enclosures on the lower path.** In CAPD they are 3–5 times wider than on the standard path (4.8×10⁻⁴
  against 1.0×10⁻⁴ for ∂D⁺₋₁ at 10⁻⁹). That is the opposite of what §9.3 found for `validated_int.py`.

**Not done: a sweep over arc radii.** The review's fix for 6b sweeps the arc radius r over [r_min, 0.3]. Two quick
attempts in CAPD failed, and the code was not kept.

- **r as an interval parameter**, on the piece [0.20, 0.21]: the enclosure blew up on the ray, near |ξ| = 0.12.
- **r as a constant state variable in the Lohner set**, with a common ray to r_min, on the pieces [0.29, 0.30] and
  [0.29, 0.291]: the short ray segment from 0.29 to r alone inflated the set to radius 20 and 0.09, and the arc then
  blew up. Near 0.3e^{3iπ/8}, y⁺ varies at a rate of order 10² per unit ξ, because the singularity is close.

A sweep would need much narrower pieces near r = 0.3, or a family of paths that avoids that corner.

### 10.5 Reproduce

```
cd solver/proof/capd_check
./build.sh                    # clones CAPD 37057e5 into ~/.cache/capd_check/CAPD, builds it (MPFR on) and capd_check
python3 series_check.py       # the start data against the independent series, seconds
build/capd_check data/start_point.txt data/start_box_1e-9.txt data/start_box_1e-8.txt \
    --save-point=results/point_standard.txt --enclosures=results       # the certificate: one point run, two box runs
                                                                       # (about 10 CPU minutes)
build/capd_check data/start_point.txt data/start_box_1e-9.txt data/start_box_1e-8.txt data/start_box_1e-7.txt \
    --load-point=results/point_standard.txt                            # the box runs from the saved point run, minutes
build/capd_check data/start_point.txt data/start_box_1e-8.txt --prec=0 # all in double precision, about a minute
build/capd_check data/start_point.txt data/start_box_1e-9.txt data/start_box_1e-8.txt --rho=0.2 \
    --save-point=results/point_rho0.2.txt                              # the lower path
python3 compare.py results/point_standard.txt results/point_rho0.2.txt # the two paths and section 9's run
python3 export_start.py       # regenerate data/ from fixed_point.start_data, seconds
```

The exit status is 0 only if every Krawczyk test passes and every control gives its expected verdict.

## 11. Lemmas 6a and 6b

The review ([A_review.md](A_review.md) §6a, §6b) listed two steps between the certificate of §9 and a statement about
the selection condition. This section proves 6a and the part of 6b near ξ = 0, moves the certificate to the lower path,
and states exactly what is still open. All computations are in ball arithmetic (python-flint 0.9, 256–320 bits), on
the parameter box

```math
X = \hat x + [-10^{-9}, 10^{-9}]^2, \qquad \hat x = (2.5421480475,\ 0.7769940136), \qquad x = (A,\ \omega/\omega_\infty) .
```

Times are wall-clock on the shared 2-core machine (load 8–10), single process under `nice`.

**Summary.**

| item | status | key numbers |
|---|---|---|
| certificate on the lower path (11.2) | **proved** | K(X) ⊂ int X, margins 8.57×10⁻¹⁰, 9.71×10⁻¹⁰ |
| lemma S, start values on a domain reaching the real axis (11.3) | **proved** | contraction ≤ 0.4903 |
| lemma 6b near 0: y⁺ analytic for 0 < \|ξ\| ≤ 0.0624, 0 ≤ arg ξ ≤ 3π/8 (11.4) | **proved** | |
| lemma 6b on the annulus 0.0624 ≤ \|ξ\| ≤ 0.2 (11.4) | **open** here (numerical evidence only); proved in §12 | validated sweeps attempted, see 11.4 |
| lemma 6a, J_D(0.3) = 0 ⇔ y⁺ ≡ y⁻ (11.5) | **proved, given 6b** | ρ₆ₐ = 4.2×10⁻³ < 1 |
| mutation tests (11.6) | two caught, one not caught (explained) | |

So the theorem about the selection condition (11.7) is proved **conditionally on the annulus part of lemma 6b**.

### 11.1 Definitions

The system is the truncated M = 1 system of §2 with real parameters x ∈ X. With z = ξ², the lateral solution y⁺ is
the solution asymptotic to the formal series on a sector of small radius containing 0 ≤ arg z ≤ 3π/4 (Braaksma's
1-summability, as in §4.4 and review 2f; this is the one fact assumed, not proved, here). The reflection
σ(y)_μ = conj(y₋μ) (and the same for the S and C components) maps solutions to solutions under ξ ↦ ξ̄ and fixes the formal series, so
y⁻(ξ) = σ(y⁺(ξ̄)) is the lateral solution from below.

The **selection condition** is y⁺ ≡ y⁻. Both continue along the real axis from near 0; write y⁺_ℝ for the continuation
of the germ y⁺ along the real axis, and

```math
\psi = y^+_{\mathbb R} - y^-_{\mathbb R} = y^+_{\mathbb R} - \sigma(y^+_{\mathbb R}), \qquad J_D(\xi_1) = \psi_{D,+1}(\xi_1), \qquad \psi_{D,-1}(\xi_1) = -\overline{J_D(\xi_1)} .
```

ψ is anti-symmetric, σψ = −ψ. In the variables of §4.4 (p, v, c), β = v_ψ(Z) ∈ ℂ² satisfies β₋₁ = −conj(β₊₁), and
Σ := β₊₁ is the single complex Stokes constant: 11.5 shows ψ ≡ 0 ⇔ Σ = 0.

The certificate computes the continuation along a path. On the **lower path** (ray arg ξ = 3π/8 from 0.1 to 0.2, arc
|ξ| = 0.2 down to the real axis, real axis from 0.2 to 0.3) it equals y⁺_ℝ(0.3) if y⁺ is analytic on

```math
\Omega = \{\, 0 \lt |\xi| \le 0.2,\ 0 \le \arg\xi \le 3\pi/8 \,\}
```

(the region between the path and the real axis; the last leg is the real axis itself). That is lemma 6b for this
path.

**Why the lower path.** On the path of §9 (arc at 0.3) lemma 6b would need analyticity up to |ξ| = 0.3, next to the
singularity of y⁺ between radii 0.30 and 0.35 (review R6), where |y⁺| reaches 8.4 on the arc. The lower path needs
only |ξ| ≤ 0.2, a third less radius and far from that singularity, and its certificate costs one more run of the
existing integrator. The price is the real leg 0.2 → 0.3, along which the fast modes grow by about e⁷; it is
harmless at this box size (11.2).

### 11.2 The certificate on the lower path

`lemma6_path.py` runs `validated_int.py`'s point and box runs (its integrator, start data and Krawczyk assembly,
unchanged) on the lower path, and saves every state component, which 11.5 needs. One heuristic differs: the a-priori
enclosure (`lemma6_common.py`). validated_int inflates each real and imaginary radius by 1.5 per Picard iteration; on
the real axis the imaginary radius of the nearly real components starts at about 10⁻¹⁵ and never catches up with the
wrapping from the other components, so the step size collapses to about 10⁻⁶ (the real leg would take 10⁶ steps). The
replacement adds a common floor, 10⁻² times the largest radius, and refuses steps whose a-priori box grows by more
than 5% of the state's size. Containment is still verified with `contains`, so every enclosure stays rigorous; only
the step sizes change.

| run | steps (ray, arc, real) | time | final radius |
|---|---|---|---|
| point at x̂ | 303, 322, 47 | 285 s | state 1.8×10⁻¹⁶ |
| box over X, variational columns | 303, 322, 47 | 333 s | state set 5.2×10⁻¹⁰ |

J_D(x̂) = 1.0547397×10⁻¹² − 3.4904294×10⁻¹²i (radius 6×10⁻¹⁷), the same as on the upper path (§9.3) to the digits shown there.
The Krawczyk test of §5.1 passes: K(X) ⊂ int X with margins 8.570×10⁻¹⁰ and 9.712×10⁻¹⁰. J(X) also encloses the
Jacobian over every sub-box centred at x̂, and the test passes on x̂ ± 1.2×10⁻¹⁰ as well. Hence, with endpoints rounded
outward,

```math
J_D^{\rm path}(0.3;\,\cdot\,) \text{ has exactly one zero in } X, \text{ it is simple, and it lies in } A \in [2.54214804761338,\ 2.54214804761975], \quad \omega/\omega_\infty \in [0.776994013607438,\ 0.776994013612010]
```

(ω ≈ 2.96977670855). This box lies inside the enclosure from the upper path (§9), which is consistent with the two
paths giving the same function, but does not prove it.

### 11.3 Lemma S: start values on a domain that reaches the real axis

`fixed_point.py` integrates the fast component v from z = 0 along the ray to z, which needs Re(1/z) < 0 on the ray, so
its domains stay at |arg z − 3π/4| ≤ 0.4. To reach the real axis, change the integration path of v and nothing else.

**Domain.** With u = 1/z, R₊ > 0, U₀ = √2/R₊ and the cone K = {w : −3π/4 ≤ arg w ≤ 0}, let

```math
D = \{\, z : 1/z \in U_0 + K \,\} .
```

Three properties follow from elementary geometry of the translated cone:

- **Radius and coverage.** |z| ≤ R₊ on D. In the direction arg z = θ ∈ [0, 3π/4), D contains every z with
  |z| ≤ R₊ sin(3π/4 − θ), which is at least R₊/√2 for θ ≤ π/2 and includes the real segment 0 < z ≤ R₊/√2.
- **Star shape.** D is star-shaped about 0, so the slow components p and c are still integrated along the segment
  [0, z], with the bound of §4.4.
- **Path.** The path $`u + s\,e^{-3i\pi/4}`$, s ≥ 0, stays in D for every u ∈ D, and ends at z → 0 along arg z → 3π/4.

**Fast operator.** For z ∈ D, integrate the v-equation along that path:

```math
v(z) = \int_{\gamma_z} e^{\frac{1}{2t}-\frac{1}{2z}}\,\Phi_v(t)\,dt, \qquad \frac1t = \frac1z + s\,e^{-3i\pi/4},\quad s \text{ from } \infty \text{ to } 0 .
```

Along it $`|e^{1/(2t)-1/(2z)}| = e^{-as}`$ with a = √2/4, and |dt| = |t|² ds. Let c = −cos φ, φ the angle between u and
$`e^{-3i\pi/4}`$, and x = s/|u|. Then

```math
\frac{|u|^2}{|u + s e^{-3i\pi/4}|^2} = \frac{1}{1 - 2cx + x^2} \le e^{2cx} \qquad (0 \le c \le 1/\sqrt2),
```

because f(x) = ln(1 − 2cx + x²) + 2cx has f(0) = 0 and f′(x) = 2x(1 − 2c² + cx)/(1 − 2cx + x²) ≥ 0. On D one has
c ≤ 1/√2 always, and c/|u| ≤ 1/(√2 U₀) = R₊/2 (the worst case is the real end of D, u = U₀). For c ≤ 0 the ratio is
at most 1. Hence

```math
\int_{\gamma_z} \Big|e^{\frac{1}{2t}-\frac{1}{2z}}\Big|\,|t|^m\,|dt| \le |z|^{m+2} \int_0^\infty e^{-as + (m+2)\frac{c}{|u|}s}\,ds \le \frac{|z|^{m+2}}{a - (m+2)R_+/2} .
```

For an integrand $`K|t|^q`$ with q ≥ k − 2, write $`|t|^q \le R_+^{q-k+2}|t|^{k-2}`$. In the weighted norm of §4.4 the fast bound
becomes $`K R_+^{q+2-k}/a_{\rm eff}`$ with

```math
a_{\rm eff} = \frac{\sqrt2}{4} - \frac{k R_+}{2} ,
```

the same form as fixed_point's $`K R^{q+2-k}/a`$. The slow-type alternative that `fixed_point.intbound` also allows for
fast rows (integration from 0 with kernel ≤ 1) does not hold on these paths, so `lemma6_sector.py` removes it. Every
other bound of §4.4 is a sum of |coefficients| times powers of the radius and holds on D with R = R₊.

**Weights.** The Perron weights are made symmetric under μ ↦ −μ (w_j := min(w_j, w_σ(j)); any positive weights are
valid), so that σ maps the ball to itself (needed in 11.5).

**Result** (`lemma6_sector.py`, over X, R₊ = 0.1², weight exponent k = 30, N = 53):

| quantity | value |
|---|---|
| a_eff | 0.20355 |
| contraction constant | ≤ 0.4903 |
| forcing F (weighted) | 1.04×10⁴⁴ |
| ball ρ (weighted) | 8.14×10⁴⁴, with F + Λρ = 5.03×10⁴⁴ < ρ |

**Lemma S.** For every x ∈ X the truncated system has a solution Y = Y_N + η on D, with η analytic in the interior of D
and in the weighted ball above; it is unique in that ball, and it is y⁺.

*Proof of the identification.* The domains D(U₀′) = {1/z ∈ U₀′ + K} decrease as U₀′ grows, the operator on D(U₀′)
only uses values on D(U₀′) (paths and segments stay inside), and every bound is non-decreasing in R₊. So the
contraction holds on each D(U₀′), U₀′ ≥ U₀. For large U₀′, D(U₀′) lies in the sector of asymptoticity of y⁺, where
$`y^+ - Y_N = O(|z|^{53})`$. So y⁺ − Y_N satisfies the integral equation on D(U₀′): the boundary terms of p and c vanish at 0,
and that of v is $`e^{-as}`$ times a bounded function. Its weighted norm on D(U₀′) tends to 0, so it lies in the ball and
equals the fixed point there. Both are solutions of the ODE on the connected set D, so they agree on D. ∎

The same run re-proves the sector lemma of §4.4 (`--sector=0.2`) over X (contraction 0.5107). Its fixed points on the
rays |arg ξ − 3π/8| ≤ 0.2 form one analytic function on that sector: the ray integrals from 0 are path-independent
there, because Re(1/t) → −∞ uniformly as t → 0 in that sector. It is y⁺ on the rays with arg z ≤ 3π/4, hence the
continuation of y⁺ on the whole sector.

### 11.4 Lemma 6b

**What is proved.** D contains the directions 0 ≤ arg z ≤ 3π/4 − 0.4 up to a radius of at least R₊ sin 0.4 = 0.003894, and the sector
lemma covers 3π/4 − 0.4 ≤ arg z ≤ 3π/4 up to |z| = 0.01. So:

**Lemma 6b₀.** For every x ∈ X, y⁺ extends analytically to the interior of Ω₀ = {0 < |ξ| ≤ 0.0624, 0 ≤ arg ξ ≤ 3π/8},
continuously up to its boundary, and to the real segment 0 < ξ ≤ 0.0841, where it is enclosed by lemma S.

**What remains (lemma 6b₁).** Show that y⁺ extends analytically to the annular sector
A₁ = {0.0624 ≤ |ξ| ≤ 0.2, 0 ≤ arg ξ ≤ 3π/8}, for every x ∈ X. Together with 6b₀ this gives analyticity on Ω, hence
J_D^path(0.3) = J_D(0.3).

The tool for 6b₁ is a validated sweep. A family of paths (t, d) ↦ u_c(t) + d, with the same velocity du/dt for every
member, is integrated as one Lohner set with d as a constant state. If the run finishes, y⁺ continues analytically to
the tube {u_c(t) + d}. The argument: let Ψ(t, d) be the solution along u_c + d started from y⁺(u_c(0) + d), and
W = ∂_dΨ − F(u_c + d, Ψ). Then

```math
\partial_t W = F_Y(u_c + d, \Psi)\,u_c'(t)\,W, \qquad W(0, d) = 0 ,
```

so W ≡ 0. Hence Ψ(t, u − u_c(t)) does not depend on t, and it defines an analytic solution on the tube. For arcs with
varying radius the same computation, with ∂_θ in place of ∂_t, gives the Cauchy–Riemann equations. Consecutive tubes
overlap in connected sets that reach the start ray, where both functions are y⁺, so the tube functions glue to one
analytic continuation of y⁺.

The sweep was attempted in two sets of variables and with three kinds of family. None finishes at a practical cost:

1. **Arcs in ξ with the radius as a ball** (validated_int's system). These blow up at once near r = 0.05. The fast rate
   φ₀/r² ≈ 490 per unit t, evaluated on a radius interval, has an absolute width larger than the contraction it should
   resolve.
2. **The u = 1/ξ² variables** (`lemma6_usys.py`). These are the variables (P, V, C) of §4.4 with u as the independent
   variable. There the fast mode is $`e^{-u/2}`$, with a constant rate, and the C components are scaled by κ = 100. The
   system agrees with validated_int's at random points to 2×10⁻⁵⁸ (`python3 lemma6_usys.py`). A single arc is easy: at
   |ξ| = 0.062 (|u| = 260) it takes 750 steps and 110 s, and the enclosure stays at 10⁻⁸. Families are not:
   - with the radius as a ball, or as a state, they blow up for a relative radius interval of 1% (at 0.1% the
     enclosure still grows, at a rate of about 4 per unit of the arc parameter);
   - translated tubes (`lemma6_sweep.py`, `SysUD`, offset d as a state, its column in mean value form) blow up at
     half-width 1.0 at |u| = 260 (t = 0.55 of the arc) and at 0.5 at |u| = 110 (t = 0.16); half-width 0.26 at
     |u| = 260 finishes.
3. **Diagnosis.** The parameter range widens the Jacobian of every step (first order in the width). That excess
   multiplies the state set, whose width is also of first order, and the product feeds back quadratically. So a
   first-order (C¹) family is limited to widths of about 0.1–0.2% of the radius.

**Cost and what would finish 6b₁.**

- **Brute force.** Covering A₁ with tubes of half-width 0.1% of the radius takes about 1200 tubes and 4×10⁵ steps, about
  30 CPU hours on this machine. One such tube was run: at |u| = 260 with half-width 0.26 (`--rhos=260 --dfrac=0.001`) it
  finishes in 787 steps and 209 s, with the enclosure growing from 2×10⁻³ to 1.8×10⁻². The nodes and the overlap
  rule are in `lemma6_sweep.py`; the narrow width at smaller |u| has not been tested.
- **A second-order treatment of the family parameter** (a C² Lohner step for d alone, or Taylor models in (t, d)). It
  makes the excess quadratic in the width, so widths of about 5% become possible: about 30 tubes and 10⁴ steps.
- **A start-value lemma reaching |ξ| = 0.2 near the real axis.** This needs the contraction to be built on the exact
  linear (Kummer) fundamental matrix rather than on the rank-one principal part, as in the template; the majorant
  bound of lemma S grows like R₊ and exceeds 1 well before R₊ = 0.04.

**Numerical evidence for 6b₁** (not a proof):

- In double precision the arcs at 0.2 and 0.3 give the same y⁺(0.3) to 2×10⁻¹² (review R6).
- The certified zero on the lower path (11.2) lies inside the certified zero on the upper path (§9).
- The backward real-axis run of 11.5 reproduces y⁺ along 0.067 ≤ ξ ≤ 0.3 with radius ≤ 2×10⁻⁹.

### 11.5 Lemma 6a

**Lemma 6a.** For every x ∈ X: if y⁺_ℝ(0.3; x) lies in the state-set box B₁ of the lower-path box run (11.2), then

```math
J_D(0.3;\,x) = 0 \iff \psi \equiv 0 \iff y^+ \equiv y^- \iff \Sigma(x) = 0 .
```

By 6b the hypothesis holds, because y⁺_ℝ(0.3) is then the path value enclosed in B₁. The implications from right to
left are trivial.

The proof has three steps. Set ξ_a = 0.067 and Z = ξ_a² = 0.004489 ≤ R₊/√2.

**Step 1: flat estimate on the real segment 0 < z ≤ Z.** By lemma S, y⁺ = Y_N + η⁺ there with η⁺ in the ball. By the
reflection and the symmetric weights, y⁻ = Y_N + σ(η⁺) is in the same ball, and so is the segment between them. So
ψ = η⁺ − η⁻ satisfies, exactly,

```math
\psi = b + T_0\psi, \qquad b = \big(0,\ e^{\frac{1}{2Z}-\frac{1}{2z}}\,\beta,\ 0\big), \qquad \beta = v_\psi(Z) .
```

Here T₀ integrates the difference of the perturbation terms (linear in ψ by the mean value theorem on the ball):

- p and c are integrated from 0, which is valid because ψ = O(z^k);
- v is integrated backward from Z, by variation of constants: $`v(z) = e^{1/(2Z)-1/(2z)}v(Z) + \int_Z^z e^{1/(2t)-1/(2z)}\Phi_v\,dt`$.

The backward kernel obeys, for n + 2 ≥ 0 and (n + 2)Z < 1/2,

```math
\int_z^Z e^{-\frac12\left(\frac1z-\frac1t\right)}\,t^n\,dt \le \frac{z^{n+2}}{\tfrac12 - (n+2)Z} .
```

To see this, substitute u = 1/t and use $`(U/u)^{n+2} \le e^{(n+2)Z(U-u)}`$. The weighted fast bound is therefore $`K Z^{q+2-k}/a_{\rm eff}`$
with a_eff = 1/2 − kZ = 0.3653. With the derivative majorants of `fixed_point.lipschitz_and_forcing` over the ball of
lemma S, the operator norm is

```math
\lVert T_0\rVert \le \Lambda = 0.1513 ,
```

with its own Perron weights w (`lemma6_sector.py`). Since $`z^{-k}e^{-1/(2z)}`$ increases on (0, Z] (kZ < 1/2),
$`\lVert b\rVert = Z^{-k}\max_\mu w_{v,\mu}|\beta_\mu|`$. Hence ε := ‖ψ − b‖ ≤ Λ/(1 − Λ)‖b‖, and

```math
|p_\mu(Z)| \le Z^{k}\,\varepsilon/w_{p,\mu}, \qquad |c_\nu(Z)| \le Z^{k+1}\,\varepsilon/w_{c,\nu}, \qquad v_\psi(Z) = \beta .
```

**Step 2: the backward flow.** If J_D(0.3) = 0, then ψ(0.3) = (0, 0, u) with u = (ψ_S, ψ_C) ∈ ℂ⁵. The C¹ Lohner
integrator runs backward along the real axis from 0.3 to ξ_a, where the fast modes decay by about $`e^{-106}`$. It starts
from the hull of B₁ and σ(B₁), with the 7 identity columns as variational columns. The resulting enclosure B ∋ DΦ(y)
holds for every y in that hull (291 steps, 130 s, max radius of B 1.8×10⁻⁶). By the mean value theorem
ψ(ξ_a) ∈ B ψ(0.3). In the (p, v, c) variables at ξ_a, with T_a the exact change of variables
(p = D/ξ, v = S/ξ² − (1 + iμω)D/ξ, c = C) and G = T_a B restricted to the columns of u:

```math
(p, c)(Z) = G_{\rm slow}\,u, \qquad \beta = G_{\rm fast}\,u .
```

**Step 3: the contraction in β.** G_slow (5×5) is invertible: arb inverts the whole ball, with entries of G_slow⁻¹ at
most 1.26. So β = H (p, c)(Z) with H = G_fast G_slow⁻¹, whose entries are at most 7.7. With step 1,

```math
\max_\mu w_{v,\mu}|\beta_\mu| \le \rho_{6a}\,\max_\mu w_{v,\mu}|\beta_\mu|, \qquad
\rho_{6a} = \frac{\Lambda}{1-\Lambda}\,\max_\mu w_{v,\mu}\sum_j |H_{\mu j}|\,\frac{Z^{k_j-k}}{w_j} \le 4.22\times10^{-3} .
```

So β = 0, hence b = 0 and ψ = T₀ψ with ‖T₀‖ < 1, so ψ = 0 on (0, Z]. By ODE uniqueness y⁺_ℝ ≡ y⁻_ℝ, and the two germs
coincide. ∎

The margin is large (ρ₆ₐ = 0.004 against 1). The large entries of H sit in the c-columns, which carry the extra factor
Z/w_c.

### 11.6 Mutation tests

Each test breaks one input and checks that the corresponding check then fails.

| mutation | where | result |
|---|---|---|
| one coefficient of the formal series (D₊₁, order z⁵) changed by a relative 10⁻⁶ | lemma S, `lemma6_sector.py --mutate=series` | **caught**: the residual no longer vanishes below order N, the lemma reports False |
| y⁻ taken as conj(y⁺) without the reflection μ ↦ −μ (wrong mode labelling) | 6a, `lemma6_jump.py --mutate=swap` | **caught**: the start hull has radius 0.40 and the backward integration fails (step-size underflow); 6a is not established |
| (1 − iμω) instead of (1 + iμω) in the change of variables at ξ_a (wrong mode normalization) | 6a, `lemma6_jump.py --mutate=coords` | **not caught**: ρ₆ₐ = 0.12 < 1 |

The third mutation is not caught, and that is a real limit of the harness. The wrong v differs from the right one by
2iμω p, and p(Z) is itself bounded by the flat estimate. So the error changes ρ₆ₐ by a second-order amount (0.004 to
0.12) and does not flip the outcome. Read the other way, the 6a check is not sensitive to the normalization of v at
the level of a sign in s. The change of variables used there is that of fixed_point.py (V = Q − (1 + s)P, review 2a),
and the same formula in `lemma6_usys.py` is checked against validated_int's system by its self-test; that, not the 6a
check, is what guards it.

The sweep of 11.4 has no mutation test, because it does not pass even unmutated. `--mutate=sign` (the fast term with
the wrong sign) is in the script for when it does.

### 11.7 The resulting theorem, and what remains

**Proved** (rigorous, given Braaksma's theorem for the existence of y⁺):

1. J_D^path(0.3), the D-component of the jump along the lower path, has exactly one zero in X, and it is simple. It lies
   in A ∈ [2.54214804761338, 2.54214804761975], ω/ω∞ ∈ [0.776994013607438, 0.776994013612010].
2. Lemma S and lemma 6b₀: y⁺ is analytic on {0 < |ξ| ≤ 0.0624, 0 ≤ arg ξ ≤ 3π/8} and enclosed there and on the real
   segment up to 0.0841.
3. Lemma 6a in the form of 11.5.

**Theorem (conditional on lemma 6b₁).** Assume that, for every x ∈ X, y⁺ is analytic on the annular sector
0.0624 ≤ |ξ| ≤ 0.2, 0 ≤ arg ξ ≤ 3π/8. Then the M = 1 selection condition y⁺ ≡ y⁻ holds at exactly one point x* of X,

```math
x^* \in [2.54214804761338,\ 2.54214804761975] \times [0.776994013607438,\ 0.776994013612010],
```

and the zero is simple: x ↦ J_D(0.3; x) has a non-singular Jacobian at x*. Moreover Σ has a simple zero at x*. The
reason: at x*, ψ ≡ 0, so φ = ∂ψ/∂x_j is a solution of the variational equation along y⁺ = y⁻ with finite weighted norm
(each ψ(·; x) obeys the flat estimate). The argument of 11.5 applies to φ verbatim, because its coefficients lie in the
same hulls. So the real-linear map 𝓛 : ∂Σ ↦ ∂J_D(0.3) is injective, DJ_D(x*) = 𝓛 ∘ DΣ(x*), and DΣ(x*) is non-singular.

*Proof.* By 6b₀ and 6b₁, y⁺ is analytic on Ω, so J_D^path(0.3; x) = J_D(0.3; x) for x ∈ X. By 11.2 this function
has exactly one simple zero x* in X. By 11.5 (whose hypothesis now holds), J_D(0.3; x) = 0 ⇔ y⁺ ≡ y⁻. ∎

**What remains:** lemma 6b₁ only (11.4 lists three routes and their cost). Without it, the honest statement is item 1
above together with 6a as stated. (Lemma 6b₁ is proved in §12, and the theorem is restated there without the
condition, §12.6.)

**Independence.**

- The 6a backward run and the lower-path runs use validated_int's integrator, which the review audited; the
  independent CAPD check is the subject of §10. Nothing here depends on it.
- Lemma S reuses fixed_point.py's majorant code. Only the two kernel bounds above are new, and they are proved in the
  text.
- The u-system is checked against validated_int's system, not against independent code.
- No reference value was used to tune anything: the box, ξ_a, k and R₊ were chosen for margin before the final runs.

**Reproduce** (in `solver/proof/`):

```
python3 lemma6_sector.py --json=lemma6_sector_1e-9.json      # lemma S over X, the old sector lemma, the flat estimate; ~5 s
python3 lemma6_path.py point --out=lemma6_path_point.json      # lower path, point run, ~5 min
python3 lemma6_path.py box --h=1e-9 --out=lemma6_path_box_1e-9.json   # lower path, box run, ~6 min
python3 lemma6_path.py krawczyk --point=lemma6_path_point.json --box=lemma6_path_box_1e-9.json   # K(X), seconds
python3 lemma6_jump.py --box=lemma6_path_box_1e-9.json --json=lemma6_jump_1e-9.json   # lemma 6a, ~2-3 min
python3 lemma6_sector.py --mutate=series                      # mutation: must report False
python3 lemma6_jump.py --box=lemma6_path_box_1e-9.json --mutate=swap     # mutation: must fail
python3 lemma6_jump.py --box=lemma6_path_box_1e-9.json --mutate=coords   # mutation: not caught (see 11.6)
python3 lemma6_usys.py                                         # u-system vs validated_int's system
python3 lemma6_sweep.py --rhos=260 --dfrac=0.001               # one narrow tube of the (unfinished) 6b1 sweep
```

The saved runs are `lemma6_path_point.json`, `lemma6_path_box_1e-9.json`, `lemma6_sector_1e-9.json` and
`lemma6_jump_1e-9.json`.

## 12. Lemma 6b₁

This section proves lemma 6b₁ (11.4), the last open step of §11.7, and states Theorem A(M = 1) without conditions.
Everything that enters the proof is ball arithmetic (python-flint 0.9). The parameter box is the X of §11. Times are
wall-clock on the shared 2-core machine during the runs (load 11–14, so each process got about a fifth of a core),
single process under `nice`.

**Summary.**

| item | status | key numbers |
|---|---|---|
| route assessment (12.1) | double precision, not a proof | route-1 loop gain 3.1 at \|ξ\| = 0.2 |
| disc lemma (12.2) | proved | |
| lemma 6b₁ (12.3, 12.4) | **proved** | 178 discs in 12 rows; contraction ≤ 0.439 |
| mutation tests and a negative control (12.5) | all caught | |
| Theorem A(M = 1) (12.6) | **proved**, given Braaksma's theorem | |

### 12.1 Which route, and why

`lemma6b1_estimate.py` (double precision, not rigorous) integrates y⁺ over the annulus from the formal series at
|u| = 260 on the start ray. In the variables of `lemma6_usys.py`, with Q = V + (1 + s)P and W = V + [(C − 1)Q], the
maxima over each arc are:

| \|u\| | \|ξ\| | max \|P\| | max \|V\| | max \|Q\| | max \|C₀ − 1\| | route-1 loop gain g |
|---|---|---|---|---|---|---|
| 260 | 0.0620 | 2.63 | 0.23 | 8.44 | 0.013 | 0.09 |
| 150 | 0.0816 | 2.70 | 0.41 | 8.81 | 0.022 | 0.17 |
| 100 | 0.1000 | 2.79 | 0.65 | 9.29 | 0.034 | 0.27 |
| 60 | 0.1291 | 2.98 | 1.22 | 10.38 | 0.058 | 0.57 |
| 40 | 0.1581 | 3.24 | 2.12 | 12.05 | 0.090 | 1.12 |
| 25 | 0.2000 | 3.85 | 4.53 | 16.25 | 0.154 | 3.11 |

**Route 1** is a start-value contraction built on the exact linear (Kummer) flow, with the nonlinear terms as the
perturbation. At |ξ| = 0.2 those terms are not a perturbation: C₀ − 1 reaches 0.15 and V reaches 4.5. The linearized
nonlinear part contains the loop P → C → V → P:

- C is driven by $`C_u = -z^2[CE]/2`$, whose P-derivative is about $`z^2|P|`$. Integrated from u = ∞ that gives
  $`|\delta C| \lesssim |P|\,\lVert\delta P\rVert/|u|`$.
- The fast equation contains −([(C − 1)Q] + [(C − 1)W])/2, with C-derivative about (Q + W)/2. Through the fast
  kernel (rate a = √2/4 along the path of 11.3, on which |t| ≥ |u|/√2) that gives
  $`|\delta V| \lesssim \tfrac{|Q| + |W|}{2}\,\tfrac{\sqrt2}{a}\,|\delta C|`$.
- P is driven by $`P_u = -zV/2`$, so $`|\delta P(u)| \lesssim \int_{|u|}^\infty |\delta V|/(2t)\,dt`$.

The gain of the loop in these natural norms on |u| ≥ ρ is

```math
g(\rho) = \max\frac{|Q| + |W|}{2}\cdot\frac{\sqrt2}{a}\cdot\frac{\max|P|}{2\rho} .
```

A contraction needs g < 1. Here g = 3.1 at |ξ| = 0.2, and it crosses 1 near |ξ| = 0.155. Power weights $`|u|^{-k}`$
divide the two slow integrals by (k + 1)², but they lower the fast rate (the a_eff of 11.3), and the operator would
need new kernel bounds on a non-convex domain. At best route 1 would reach about |ξ| = 0.13–0.15. That leaves the
inner band, roughly 0.14 ≤ |ξ| ≤ 0.2, which contains the place where the pilot tube failed (|u| ≈ 49,
|ξ| ≈ 0.143), so a hybrid would still need the hard part of the sweep. Route 1 was not pursued.

**Route 2, pushed to its limit.** The tube family is a translation, Ψ(t, d) = y(u_c(t) + d). A Taylor model of Ψ in d
at fixed t is therefore the Taylor expansion of y around u_c(t). Its natural radius is the distance from u_c(t) to
the nearest singularity of y⁺, not 0.1% of |u|. The nearest singularity is the branch point of review R6. The same
script locates it, by bisection on which side arcs and inward rays pass it:

```math
u^* = 9.6746\,e^{-2.1198\,i}, \qquad \xi^* = 0.3215\,e^{1.0599\,i} .
```

From the annulus it is at distance 15 or more. So the second-order treatment of d can be replaced by Taylor expansions
of order 40–70 at isolated points, each validated on a disc (12.2). No family is integrated: one single-trajectory
Lohner run per arc carries the values from disc to disc. The first-order excess that stopped the tubes (11.4 item 3)
never arises. The cost is 178 discs and 3591 units of arc length in u, against about 1300 tubes for the brute-force
sweep.

### 12.2 The disc lemma

The code is `lemma6b1_disc.py`. The variables are those of `lemma6_usys.SysU`,
Y = (P₋₁, P₊₁, V₋₁, V₊₁, κC₋₂, κ(C₀ − 1), κC₊₂) with κ = 100. The system is y′ = F(u, y; w), polynomial in y and in
z = 1/u (11.4).

**Lemma D.** Let c and w be complex balls, Y₀ a box in ℂ⁷, n ≥ 1 and 0 < r < |c|. Suppose the computation
below succeeds. Then for every exact c′ ∈ c, y₀ ∈ Y₀ and w′ ∈ w, the solution with y(c′) = y₀ extends analytically to
the open disc |u − c′| < r, continuously to its closure, and |y_j(u) − y₀,j| ≤ M_j there.

The computation has four parts.

1. **Taylor coefficients.** The coefficients p_k (k ≤ n) of the solution in τ = u − c come from Picard iteration on
   power series, in ball arithmetic, started from the balls. After n iterations the coefficients 0, …, n are final.
   They enclose the Taylor coefficients of every solution in the statement.
2. **The defect.** d = F(c + τ, p(τ)) − p′(τ) is computed as a polynomial with exact products. z enters through the
   expansion below, with J = n + 8 and ζ replaced by a ball of the stated radius. At each fixed τ, ζ(τ) is one number
   in that ball, so the coefficient balls enclose a polynomial whose value at τ is d(τ).
3. **Vanishing orders.** For each exact initial value, p is the exact Taylor polynomial, so the coefficients of d below
   order n vanish identically. The code checks that their balls contain 0 and drops them. Then
   |d_i(τ)| ≤ Δ_i|τ|ⁿ with $`\Delta_i = \sum_{k \ge n} |d_{i,k}|\,r^{k-n}`$.
4. **Bounds and the check.** It takes the sup bounds $`M^0_j = \sum_{1 \le k \le n} |p_{j,k}|\,r^k`$ of
   |p_j(τ) − y₀,j|, the box B_j = Y₀,j + (disc of radius M⁰_j + b_j), and the ball Z = 1/c + disc(r/(|c|(|c| − r))),
   which contains z. L_ij ≥ |∂F_i/∂y_j| over B × Z × w comes from `SysU.lin` in ball arithmetic. With positive
   weights ω (any; the code uses the Perron vector of L) it forms Λ, F₀ and ρ (below). The check is Λ < 1,
   F₀ + Λρ ≤ ρ and ρ r^{n+1}/ω_j ≤ b_j; the b_j are raised until the last holds. Then M_j = M⁰_j + b_j.

The expansion of z on |τ| ≤ r:

```math
z = \frac{1}{c + \tau} = \sum_{j \le J} \frac{(-\tau)^j}{c^{j+1}} + \tau^{J+1}\zeta(\tau), \qquad |\zeta(\tau)| \le \frac{1}{|c|^{J+2}\,(1 - r/|c|)} .
```

The constants of part 4:

```math
\Lambda = \frac{r}{n+2}\,\max_i \sum_j \frac{\omega_i L_{ij}}{\omega_j}, \qquad F_0 = \max_i \frac{\omega_i \Delta_i}{n+1}, \qquad \rho = \frac{2F_0}{1 - \Lambda} .
```

*Proof.* The remainder R = y − p solves R = T(R), R(0) = 0, with the integral along the segment [0, τ]:

```math
T(R)(\tau) = \int_0^\tau \Big[ d(s) + F\big(c' + s,\, p(s) + R(s)\big) - F\big(c' + s,\, p(s)\big) \Big]\,ds .
```

Take the space of R analytic in the open disc and continuous on its closure, with finite norm
$`\lVert R\rVert = \max_i \omega_i \sup_{|\tau| \le r} |R_i(\tau)|/|\tau|^{n+1}`$; it is a Banach space. Every R with
‖R‖ ≤ ρ has |R_j| ≤ ρr^{n+1}/ω_j ≤ b_j, so p + R ∈ B. The mean value inequality on the convex box B gives
|F_i(p + R) − F_i(p + R′)| ≤ Σ_j L_ij |R_j − R′_j|. Integrating $`|s|^n`$ and $`|s|^{n+1}`$ from 0 to |τ| gives

```math
\lVert T(0)\rVert \le F_0, \qquad \lVert T(R) - T(R')\rVert \le \Lambda\,\lVert R - R'\rVert .
```

So T maps the ball ‖R‖ ≤ ρ into itself and contracts. Its fixed point gives y = p + R, analytic in the disc, with
y(c′) = y₀ and y′ = F. ∎

The radius is limited by three things:

- the contraction Λ, which grows like r/(n + 2);
- the decay of the Taylor coefficients, set by the distance to the nearest singularity;
- the wrapping of the start box through the ball Taylor recursion, which multiplies its radius by roughly
  $`e^{r\lVert |F_y| \rVert}`$.

At the Lohner box radii met here (up to 1.8×10⁻⁵), radius 24 is safe for |u| ≥ 100 and radius 10 at |u| = 25.
The disc runs at 128 bits, the Lohner runs at 256.

### 12.3 The covering and the gluing

**Rows and tiles** (`lemma6b1_cover.py design`). Row j has a centre radius ρ_j, a disc radius r_j, an order n_j and a
band a_j < |u| < b_j, with a_j = ρ_j − 0.6 r_j and b_j = ρ_j + 0.6 r_j. Its K_j + 1 centres lie on the arc
|u| = ρ_j:

```math
c_{jk} = \rho_j\,e^{i\varphi_{jk}}, \qquad \varphi_{jk} = -\tfrac{3\pi}{4}\,(1 - t_{jk}), \qquad t_{jk} \approx k/K_j \text{ on the grid } 2^{-30}\mathbb Z, \qquad t_{j0} = 0, \quad t_{jK_j} = 1 .
```

So the first centre is on the start ray and the last on the real axis. Tile (j, k) is the open polar rectangle
a_j < |u| < b_j, |arg u − φ_jk| < α_j, with α_j = 0.6 · (3π/4)/K_j.

| row | ρ_j (in u) | \|ξ\| | band a_j – b_j | r_j | n_j | discs K_j + 1 |
|---|---|---|---|---|---|---|
| 0 | 262 | 0.0618 | 247.60 – 276.40 | 24 | 70 | 25 |
| 1 | 235.36 | 0.0652 | 220.96 – 249.76 | 24 | 70 | 23 |
| 2 | 208.72 | 0.0692 | 194.32 – 223.12 | 24 | 70 | 20 |
| 3 | 182.08 | 0.0741 | 167.68 – 196.48 | 24 | 70 | 18 |
| 4 | 155.44 | 0.0802 | 141.04 – 169.84 | 24 | 70 | 16 |
| 5 | 128.8 | 0.0881 | 114.40 – 143.20 | 24 | 70 | 13 |
| 6 | 102.16 | 0.0989 | 87.76 – 116.56 | 24 | 70 | 11 |
| 7 | 79.6 | 0.1121 | 70.00 – 89.20 | 16 | 60 | 13 |
| 8 | 61.84 | 0.1272 | 52.24 – 71.44 | 16 | 60 | 10 |
| 9 | 47.14 | 0.1456 | 41.14 – 53.14 | 10 | 40 | 12 |
| 10 | 36.04 | 0.1666 | 30.04 – 42.04 | 10 | 40 | 10 |
| 11 | 24.94 | 0.2002 | 18.94 – 30.94 | 10 | 40 | 7 |

The disc radii follow 12.2: 24 for ρ_j ≥ 100, 16 for 50 ≤ ρ_j < 100, 10 below. Each next band overlaps the previous one
by 15% of the smaller half-height. The numbers are chosen in floating point, and every condition is re-checked in arb.

**Values.** They come from three sources:

- **The start box** at s₀ = 262e^{−3iπ/4} comes from the sector start-value lemma over X (11.3, contraction 0.5107,
  via `lemma6_sweep.start_box`). It contains y⁺(s₀; x) for every x ∈ X; its radius is 2.9×10⁻⁸.
- **The ray run.** One QR-Lohner run (`validated_int.Lohner`, order 12, on `lemma6_usys.SysU`, with w and A as constant
  states, so it encloses the trajectory of every x ∈ X) goes inward along the start ray and stops at each
  s_j = ρ_j e^{−3iπ/4}.
- **The arc runs.** From the box at s_j, one Lohner run goes along the arc |u| = ρ_j and stops at each t_jk. Its box
  at c_jk contains y_arc(c_jk; x), the continuation of y⁺ along the ray and then the arc, for every x.

Lemma D is applied to each such box, with c_jk as a ball and w from the box.

**Lemma G (gluing).** Let y⁺ be analytic on an open set G₀. Let T₁, …, T_N be open sets, and let each f_m solve the
system and be analytic on an open disc D_m ⊃ T_m. Suppose that for each m the set
M_m = T_m ∩ (G₀ ∪ T₁ ∪ … ∪ T_{m−1}) is connected and contains a point q_m with f_m(q_m) = f(q_m). Here f is y⁺ (if
q_m ∈ G₀) or the f_l of an earlier tile T_l ∋ q_m. Then y⁺ extends analytically to G₀ ∪ T₁ ∪ … ∪ T_N and equals f_m
on T_m.

*Proof.* By induction on m. f_m and the function built so far both solve the system on the connected open set M_m and
agree at q_m. So they agree near q_m (uniqueness for the system, which is regular for u ≠ 0), hence on M_m (identity
theorem). ∎

Here G₀ = {|u| > 257, −3π/4 − 0.4 < arg u < 0}, and y⁺ is analytic on it:

- |u| > 257 is |ξ| < 0.06238, so G₀ is the union of the interior of Ω₀ (lemma 6b₀) and the part |u| > 257 of the open
  sector of the sector lemma (11.3);
- the two pieces overlap in a connected set where both are y⁺.

The tiles are taken row by row from the outside, each row from the ray to the real axis. f_jk is the solution through
(c_jk, y_arc(c_jk)), which by lemma D is analytic on the disc |u − c_jk| < r_j. `lemma6b1_cover.py check` verifies the
following in arb, for the exact numbers of the construction. The ρ_j are the decimals the runs used; a_j, b_j, α_j and
t_jk are binary floats.

- **C1.** All 178 discs are certified, and every run is present and was made with this design and this X.
- **C2.** Every closed tile lies in its open disc. For |θ − φ| ≤ α,
  $`|\rho e^{i\theta} - \rho_j e^{i\varphi}|^2 \le \rho^2 + \rho_j^2 - 2\rho\rho_j\cos\alpha`$, which is convex in ρ,
  so the check is at ρ = a_j and ρ = b_j.
- **C3.** On each arc, the half-arcs from c_{j,k−1} and from c_jk to the midpoint q (at angle
  (φ_{j,k−1} + φ_jk)/2) lie in their own discs: 2ρ_j sin((φ_jk − φ_{j,k−1})/4) < r_j.
- **C4.** Consecutive tiles of a row overlap and non-consecutive ones do not: Δφ/2 < α_j < Δφ. The first tile reaches
  below arg u = −3π/4 and the last above 0.
- **C5.** Between rows j − 1 and j, the ray point q at radius (a_{j−1} + b_j)/2 lies in both bands and between ρ_j
  and ρ_{j−1}, with ρ_{j−1} − |q| < r_{j−1} and |q| − ρ_j < r_j.
- **C6.** s₀ ∈ G₀ (ρ₀ = 262 > 257).
- **C7.** For each new tile, the meeting set M is a union of polar rectangles: its nonempty intersections with G₀ and
  the earlier tiles. Every one of them is linked to the one containing q by proven-nonempty overlaps, so M is
  connected, and q lies in the intersection with the right earlier piece.
- **C8.** b₀ > 260, a₁₁ < 25 and consecutive bands overlap, so the tiles cover A₁ = {25 ≤ |u| ≤ 260, −3π/4 ≤ arg u ≤ 0}.

The agreement points are where the functions provably meet:

- for k ≥ 1, at q on the arc: $`f_{jk}(q) = y_{\rm arc}(q) = f_{j,k-1}(q)`$, because each half-arc lies in its disc
  (C3) and solutions along a curve are unique;
- for k = 0 and j ≥ 1, at q on the ray: $`f_{j0}(q) = y_{\rm ray}(q) = f_{j-1,0}(q)`$ by C5, since c_j0 = s_j and
  f_j0 is the solution through (s_j, y_ray(s_j));
- for (0, 0), at q = s₀ ∈ G₀: $`f_{00}(s_0) = y^+(s_0)`$.

Every bound in the check is rigorous for those exact numbers: an intersection of two polar rectangles is kept unless
it is proven empty, and an overlap counts as a link only if it is proven nonempty.

### 12.4 Results

**The ray run.** The start box at s₀ has radius 2.9×10⁻⁸. The run takes 452 steps in 11 legs and 142 s, and the
state radius at s₁₁ (|u| = 24.94) is 7.9×10⁻⁸.

**The rows.** Every one of the 178 certificates passes. "Box radius" is the radius of the Lohner box at the disc
centres, the input of lemma D; it contains the spread over X. "sup" is the bound M on |y − y(c)| over the disc.

| row | ρ_j | discs | Lohner steps | max Λ | max box radius | max sup \|y − y(c)\| | time |
|---|---|---|---|---|---|---|---|
| 0 | 262 | 25 | 789 | 0.173 | 2.9×10⁻⁸ | 0.13 | 439 s |
| 1 | 235.36 | 23 | 717 | 0.174 | 1.2×10⁻⁸ | 0.16 | 383 s |
| 2 | 208.72 | 20 | 657 | 0.175 | 1.6×10⁻⁸ | 0.21 | 351 s |
| 3 | 182.08 | 18 | 579 | 0.177 | 2.3×10⁻⁸ | 0.28 | 232 s |
| 4 | 155.44 | 16 | 558 | 0.179 | 3.5×10⁻⁸ | 0.40 | 220 s |
| 5 | 128.8 | 13 | 461 | 0.183 | 5.7×10⁻⁸ | 0.62 | 166 s |
| 6 | 102.16 | 11 | 402 | 0.191 | 1.0×10⁻⁷ | 1.07 | 149 s |
| 7 | 79.6 | 13 | 350 | 0.152 | 2.1×10⁻⁷ | 1.15 | 124 s |
| 8 | 61.84 | 10 | 316 | 0.164 | 4.3×10⁻⁷ | 2.14 | 127 s |
| 9 | 47.14 | 12 | 275 | 0.158 | 1.2×10⁻⁶ | 2.27 | 106 s |
| 10 | 36.04 | 10 | 268 | 0.189 | 3.6×10⁻⁶ | 4.63 | 100 s |
| 11 | 24.94 | 7 | 249 | 0.439 | 1.8×10⁻⁵ | 15.5 | 95 s |
| all | | 178 | 5621 | 0.439 | 1.8×10⁻⁵ | 15.5 | 2491 s |

The contraction stays below 0.2 except on the innermost row, where the disc of radius 10 around the start-ray centre
comes within 5.7 of u*. The total is 6073 Lohner steps, about 44 minutes of wall-clock time.

**The check** (`lemma6b1_cover.py check`, 1 s) passes all of C1–C8. The meeting sets of the 178 tiles consist of 566
polar rectangles joined by 450 proven links. The result is saved in `lemma6b1_check_1e-9.json`.

**Lemma 6b₁ (proved).** For every x ∈ X, y⁺ extends analytically to the open set G = G₀ ∪ ⋃ T_jk. G contains the
annular sector A₁ = {25 ≤ |u| ≤ 260, −3π/4 ≤ arg u ≤ 0}, that is 0.0620 ≤ |ξ| ≤ 0.2, 0 ≤ arg ξ ≤ 3π/8. It also
contains a neighbourhood of A₁: the bands reach 0.0602 < |ξ| < 0.2298, and the tiles reach past both edge rays. On
each tile, y⁺ is enclosed by the Taylor model of its disc.

*Proof.* Lemma D for each disc (C1), then lemma G with C2–C7, then C8. ∎

### 12.5 Mutation tests and a negative control

Each test breaks one input and checks that the corresponding check then fails.

| test | what is broken | must | result |
|---|---|---|---|
| `--mutate=sign` | lemma D's defect computed with the fast term of the wrong sign (V_u = +V/2 + …), on the 7 discs of row 11 | certify no disc | **caught**: 0 of 7 certified. On every disc the defect coefficients below order n no longer contain 0 (`lemma6b1_runs/mutate_sign.log`, 80 s) |
| `check --drop=5,6` | tile (5, 6) left out of the covering | print FAIL | **caught**: FAIL, three failures. Tile (5, 7) has no agreement point, the meeting set of tile (6, 5) is disconnected, and the centre direction of (5, 6) at \|u\| = 128.8 is not covered |
| `control` | discs at 14e^{−3iπ/4} of radius 2 to 8; the branch point u* (12.1) is at distance 5.12 | certify no radius above 5.12 | **consistent**: radii 2 and 3 certified (Λ = 0.09, 0.26); radii 4, 4.5, 5.5, 6 and 8 not (Λ = 1.11, 467, then far above 1). The ray run continued to \|u\| = 14 in 82 steps (`lemma6b1_runs/control.log`, 32 s) |

The control is the meaningful one. Lemma D cannot pass on a disc that contains a singularity of the solution, and it
does not. It does pass on the smaller discs at the same centre, so the failure comes from the singularity, not from
the place. It stops before the singularity (radius 4 against distance 5.12) because y⁺ grows towards u*: on the disc
of radius 3 the bound is already sup |y − y(c)| ≤ 40.

What is not tested: a system that is wrong in the same way in the Taylor recursion and in the defect. Lemma D would
then prove analyticity for the wrong system, since its conclusion is about whatever system it is given. The identity
of the system is guarded instead by the self-tests: the generic right-hand side against `SysU.rhs`, `SysU` against
`validated_int.Sys` (`lemma6_usys.py`), and `validated_int.Sys` against `lateral.py` (review R3, R8). The start box and
the Lohner integrator have their own mutation tests (11.6, 9.5).

### 12.6 Theorem A(M = 1)

**Theorem A(M = 1).** Assume Braaksma's 1-summability theorem for the existence of the lateral solutions y± and
their asymptotics (11.1; the one fact taken from the literature). Let x = (A, ω/ω∞) and
X = (2.5421480475, 0.7769940136) + [−10⁻⁹, 10⁻⁹]². Then the M = 1 selection condition has exactly one zero in X, it
is simple, and it lies at

```math
A \in [2.54214804761338,\ 2.54214804761975], \qquad \omega/\omega_\infty \in [0.776994013607438,\ 0.776994013612010]
```

(ω ≈ 2.96977670855). Precisely:

- there is exactly one x* ∈ X with y⁺ ≡ y⁻, equivalently with Σ(x*) = 0, equivalently with J_D(0.3; x*) = 0;
- x* lies in the box above;
- the real 2 × 2 Jacobians of Σ and of x ↦ J_D(0.3; x) are non-singular at x*.

*Proof.* This is the conditional theorem of 11.7, whose hypothesis is now lemma 6b₁ (12.4). In order:

1. **J^path = J.** Lemma 6b₀ (11.4) gives the analyticity of y⁺ on the part |ξ| ≤ 0.0624 of Ω, lemma 6b₁ on a
   neighbourhood of the annular sector, and the two overlap in G₀, where both are y⁺. So y⁺ is analytic on Ω, the
   continuation along the lower path equals the continuation along the real axis, and
   J_D^path(0.3; x) = J_D(0.3; x) for every x ∈ X.
2. **One simple zero.** By 11.2, J_D^path(0.3; ·) has exactly one zero in X; it is simple and lies in the box above.
3. **It is the selection condition.** By step 1, y⁺_ℝ(0.3; x) is the path value, so it lies in the state set B₁ of
   the box run. That is the hypothesis of lemma 6a (11.5), which then gives J_D(0.3; x) = 0 ⇔ y⁺ ≡ y⁻ ⇔ Σ(x) = 0.
4. **Simplicity.** As in 11.7, DJ_D(0.3; x*) = 𝓛 ∘ DΣ(x*) with 𝓛 injective, so DΣ(x*) is non-singular. ∎

What is not claimed:

- this is the one-harmonic truncation (M = 1), not the physical problem (§7; review 6c);
- nothing is said about parameters outside X.

### 12.7 What is assumed, what is new, and how to reproduce

**Assumed.** Braaksma's theorem, as in §11 (the existence of y⁺ and its asymptotics on its sector). Nothing else.

**Reused, and how each piece is checked.**

- The start box is the sector start-value lemma of 11.3 (`lemma6_sector.sector_old` over X, evaluated by
  `lemma6_sweep.start_box`), which has its own mutation test (11.6).
- The integrator is `validated_int.Lohner` (reviewed, [A_review.md](A_review.md)), with the a-priori floor of
  `lemma6_common.py`; containment is still verified there.
- The system is `lemma6_usys.SysU`, checked against `validated_int.Sys` by `python3 lemma6_usys.py`.

**New.**

- `lemma6b1_disc.py`: lemma D. Its right-hand side is checked against `SysU.rhs` by `python3 lemma6b1_disc.py`
  (mismatch 2.4×10⁻⁵⁹). Its Jacobian is `SysU.lin`, which the self-test of `lemma6_usys.py` checks against series
  derivatives.
- `lemma6b1_cover.py`: the design, the runs, the check of 12.3, the mutation test and the control.
- `lemma6b1_estimate.py`: the double-precision assessment of 12.1. It is not part of the proof.
- The saved runs are `lemma6b1_runs/ray.json`, `lemma6b1_runs/row_00.json` … `row_11.json` and the check
  `lemma6b1_check_1e-9.json`.

No existing script was changed. A few further points:

- **Review.** The new files have not been reviewed independently. A second implementation of lemma D (for example the
  discs re-run in CAPD, §10) would be the natural check.
- **Arc starts.** Each arc run starts from the box of the ray run at s_j, not from its Lohner set. That drops the
  correlation with the parameters, which is rigorous; it only makes the boxes wider.
- **Precision.** Lemma D runs at 128 bits of relative precision. Its input boxes have radii of 10⁻⁹ and more, so
  rounding is not what limits it.
- **Powers.** No power of a ball that may contain 0 is taken (the python-flint pitfall in AGENTS.md); every power is
  of an exact positive number.

**Reproduce** (in `solver/proof/`; times as measured on the loaded machine):

```
python3 lemma6b1_disc.py                        # self-test of the generic right-hand side, seconds
python3 lemma6b1_cover.py design                # the 12 rows and 178 tiles, no computation
python3 lemma6b1_cover.py ray                   # start box and the ray run -> lemma6b1_runs/ray.json, ~2.5 min
python3 lemma6b1_cover.py rows                  # 12 arc runs and 178 discs -> lemma6b1_runs/row_JJ.json, ~42 min
python3 lemma6b1_cover.py check --json=lemma6b1_check_1e-9.json   # the checks of 12.3: must print PASS, 1 s
python3 lemma6b1_cover.py --mutate=sign         # mutation: no disc of row 11 may be certified, ~1.5 min
python3 lemma6b1_cover.py check --drop=5,6      # mutation: must print FAIL, 1 s
python3 lemma6b1_cover.py control               # negative control at the branch point, ~0.5 min
python3 lemma6b1_estimate.py                    # route assessment and the branch point (double precision), ~3 min
```

`rows` skips the rows already saved, so it can be restarted after an interruption; `row --j=J` runs one row.
`python3 check_numbers.py` (repository root) ties the quoted 178 discs, 12 rows, contraction ≤ 0.439 and 6073 Lohner
steps to `lemma6b1_check_1e-9.json`.

# Layer B for the M-harmonic model: the selection theorem

This note works out layer B of [PROOF.md](../PROOF.md) (the analytic selection theorem) as far as it can be taken now,
for the **M-harmonic model**: the large-D self-similar equations with the θ-dependence truncated to the odd harmonics
|μ| ≤ M of p and m and the even harmonics |μ| ≤ M − 1 of b and c. Gap (b) of the README (uniformity in M) and layer A
(the computer-assisted zero) are treated elsewhere; here M is fixed and every constant may depend on it.

It was written by Claude (Anthropic) under Nathan Moinvaziri's direction and has not been checked by a mathematician.
Statements marked **proved** are proved here or in the README; **cited** means reduced to a published theorem whose
hypotheses are checked below; **open** means new work is needed. Citations marked *(unverified)* were not checked against
the published record.

Contents

- [Summary](#summary)
- [1. The M-harmonic model](#1-the-m-harmonic-model)
- [2. What the rigorous precedents do](#2-what-the-rigorous-precedents-do)
- [3. The formal solution and the slow–fast structure](#3-the-formal-solution-and-the-slowfast-structure)
- [4. Gap (a): Borel summability](#4-gap-a-borel-summability)
- [5. Gap (c): the light cone](#5-gap-c-the-light-cone)
- [6. The selection theorem](#6-the-selection-theorem)
- [7. What remains](#7-what-remains)
- [References](#references)

## Summary

- **The theorem** (Section 6, Theorem B(M)): for n ≥ n₀(M), a discretely self-similar solution of the M-harmonic model
  that is regular at the centre, analytic at the light cone and asymptotic to the formal 1/n solution exists exactly when
  $`C_{n,\mu}(F_n, \omega_n) = 0`$ for every odd μ ≤ M. Here $`C_{n,\mu} = \kappa_{n,\mu}\,\big[S_\mu(F, \omega) + O(1/n)\big]`$, with
  S the inner Stokes multipliers and $`\kappa_{n,\mu}`$ an explicit non-vanishing factor. A zero of S that is simple modulo the
  phase continues to a unique branch with $`F_n - F_* = O(1/n)`$ and $`\omega_n - \omega_* = O(1/n)`$. The counting is exact: N = (M+1)/2
  complex conditions against 2N real unknowns (the harmonics of F and ω, less the phase).
- **The geometry is that of a turning point at infinity** (Lemma B2, proved). The outer system is slow–fast. Its fast
  eigenvalues are $`-n\Phi'(x) = n c^2/(x(x^2-c^2))`$ for p + m and −n/x for b − 1, and the slow ones (p − m, c) are zero. On the real
  axis they vanish nowhere and are finite except at the centre x = 0 and the light cone x = c(1) = 1, both
  regular-singular. As x → ∞ the fast eigenvalue n/x³ falls to the size of the slow terms at x ~ √n, so fast and slow
  modes merge there. That is exactly where the inner problem lives, and it is the analogue of the turning point in
  Xie–Tanveer's finger and Kruskal–Segur's dendrite problems.
- **Gap (a) splits in two.**
  - *Inner problem: reduced to a citation.* For fixed M the inner problem is a finite meromorphic ODE system in
    u = ξ², and its formal power-series solution is 1-summable by Braaksma's theorem (Écalle). The multipliers $`S_\mu`$
    are then well defined, as the coefficients of the first exponential in the difference of the two lateral Borel sums.
    Costin's sharper rank-one theorems do not apply as stated: they assume an invertible linear part with eigenvalues of
    distinct arguments, while here the fast eigenvalue is repeated across harmonics and the slow directions have
    eigenvalue 0.
  - *Outer problem: reduced to standard theory plus two checkable hypotheses.* Gevrey-1 asymptotics in ε = 1/n hold on
    compact x-sets without turning points (Sibuya; Canalis-Durand–Ramis–Schäfke–Sibuya), and the centre and the light
    cone are "doubly singular" points of the type Canalis-Durand–Mozo-Fernández–Schäfke treat. The checkable
    hypotheses are where the Borel singularities lie: those of the b-mode, and the x-independent ones of modulus about
    0.13 in the README.
- **Gap (c): reduced to standard asymptotic integration** (Lemma B6; the structural facts are proved, the error
  estimates are routine but not written).
  - Because x − c has a simple zero at the light cone, the WKB phase and the Frobenius exponent are the same object. The
    WKB fast mode, written with the resummed exponent Λ_μ, is an approximation with uniformly bounded relative error up
    to and including x = 1, in the sense of Olver's error control.
  - The Frobenius singular branch of exponent Λ_μ is unique, because Λ_μ is not an integer (Im Λ_μ ≠ 0) and the Λ_μ
    differ for different μ.
  - So the matching constant is well defined, equals 1 + O(1/n), and equals exactly 1 in the linear case (Proposition 3).
- **What is genuinely open** (Section 7): the global step that continues the centre-regular solution through an
  n-dependent complex domain from (0, 1) around x = 1 to the Stokes line x > 1 and out to |x| ~ √n, with uniform bounds;
  and the rigorous inner–outer matching (lemmas B4 and B5). Both follow known templates (Xie–Tanveer;
  Baldomá–Fontich–Guardia–Seara), but neither has been written for a system of this kind. With them done, Theorem B(M)
  is a theorem for each fixed M.

| Step | Content | Status |
|---|---|---|
| B1(M) | formal 1/n solution for any (F, ω), Galerkin-truncated | **proved** (Proposition 1, Section 3.1) |
| B2(M) | slow–fast structure; fast eigenvalues; no turning points on the real axis; turning point at ∞ | **proved** (Section 3.2) |
| B3(M) | centre-regular actual solution with Gevrey-1 asymptotics on compacts of (0, 1) | **cited**, with hypotheses checked except H-a1, H-a2 (Section 4.2) |
| B4(M) | inner Stokes multipliers well defined | **cited** (Braaksma), Section 4.1 |
| B5(M) | outer Stokes jump on x > 1 equals the inner multiplier times an explicit factor | **open** in general; **proved** in the linear case (Proposition 4 with the Weber constant) |
| B6(M) | light-cone matching: WKB fast mode ↔ Frobenius branch, half-monodromy | **reduced** to standard asymptotic integration; structural facts proved (Section 5) |
| B7(M) | implicit function theorem | **proved** given B3–B6 in C¹ form (Section 6) |
| global | continuation through the complex domain with uniform bounds | **open** (Section 7) |

## 1. The M-harmonic model

With n = D − 2, ε = 1/n, $`G = (n-1)\,c\,(b-1)`$, the self-similar equations (README, "Large D") are

```math
x c_x = (n-1)\,c\,(b-1), \qquad x b_x = b\,(p^2+m^2) - (n-1)\,b\,(b-1),
```
```math
x(x-c)\,p_x = -\omega x\,p_\theta + G p - c p + \tfrac{n}{2}\,c\,(p+m), \qquad
x(x+c)\,m_x = -\omega x\,m_\theta - G m + c m - \tfrac{n}{2}\,c\,(p+m),
```

on 0 ≤ x ≤ 1, regular at x = 0, analytic at x = 1, with c(1) = 1. Let $`P_M`$ be the projection that keeps harmonics
$`e^{i\mu\theta}`$ with |μ| ≤ M (odd μ for p and m, even μ, |μ| ≤ M − 1, for b and c). The **M-harmonic model** applies $`P_M`$
to every right-hand side. For real fields it is a finite analytic ODE system in x, with
4N + 2(N − 1) + 2 real unknowns per x. Here N = (M + 1)/2 is the number of positive odd harmonics.

Two remarks:
- The solvers in this repository (f64 and double-double) compute exactly such Galerkin or collocation truncations in θ,
  with kc Fourier modes. So Theorem B(M) is a statement about the systems actually solved, up to the convergence in M
  that gap (b) addresses.
- $`P_M`$ commutes with $`\partial_\theta`$ and preserves the parity classes (odd harmonics for p and m, even for b and
  c). That is all the induction in Proposition 1 uses.

## 2. What the rigorous precedents do

**Xie & Tanveer** (Saffman–Taylor fingers, ARMA 166 (2003) *(unverified)*); **Xie** (kinetic undercooling,
arXiv:1610.00076). The architecture is:

1. Reformulate as an integro-differential problem on a complex domain. Drop the symmetry condition to get a
   **half problem**, which has a solution for every value of the selection parameter.
2. Near the turning point (the complex point where the small-parameter expansion fails), rescale to an n-free
   **inner equation**. Prove that the half-problem solution matches an inner solution, and that the quantity measuring
   the symmetry condition is asymptotically $`\tilde S(\alpha)\,e^{-\eta}`$, with $`\tilde S`$ the Stokes constant of the
   inner equation (Theorem 1.11 in arXiv:1610.00076).
3. At a zero β₀ of $`\tilde S`$ with $`\tilde S'(\beta_0) \ne 0`$, the implicit function theorem gives an analytic branch β(ε^{2/3}) → β₀
   on which the symmetry condition holds (Theorem 1.12). The full "exists iff S = 0" statement is proved at the level
   of the inner equation. For the full problem, existence near each simple zero is what is proved.

The map onto our problem:

| Xie–Tanveer | here |
|---|---|
| finger shape, half strip | (p, m, b, c) on 0 ≤ x ≤ 1, continued into complex x |
| symmetry condition on the imaginary axis | analyticity at the light cone x = 1 (C_n = 0) |
| half problem (no symmetry) | the centre-regular solution for given (F_n, ω), which exists without the light-cone condition (B3) |
| turning point (complex) | x = ∞, where the fast eigenvalue n/x³ meets the slow scale (B2) |
| inner equation | the n-free inner problem at x ~ √n |
| Stokes constant $`\tilde S(\alpha)`$ | the multipliers $`S_\mu(F, \omega)`$ |
| width parameter λ (via α, β) | (F, ω): N complex conditions, 2N real unknowns |

The main difference: our selection parameter is a vector (F's harmonics and ω), and the inner multiplier is a vector
indexed by harmonics with a matrix structure (the harmonics couple through gravity). For fixed M this changes
nothing in the logic: the implicit function theorem is used in finite dimension 2N.

**Costin** (Duke 93 (1998), arXiv:math/0608408; IMRN 1995, arXiv:math/0608414 *(journal details unverified)*). The
system is put in the normal form

```math
y' = f_0(x) - \hat\Lambda\,y - \tfrac1x\,\hat B\,y + g(x, y), \qquad x \to \infty,
```

with $`\hat\Lambda`$ invertible, its eigenvalues of pairwise distinct arguments (the nonresonance condition), and $`\hat B`$
diagonalizable with $`\hat\Lambda`$. Under these hypotheses, transseries solutions are Borel summable in the generalized sense,
every solution decaying in a sector is the sum of a unique transseries, and across a Stokes line the transseries
parameter jumps by $`S_\beta`$, a fixed multiple of the Stokes constant. **These hypotheses fail for our inner problem**, for
two reasons:

- the fast eigenvalue is the same (½ in the variable 1/u, u = ξ²) for every harmonic;
- the slow directions, which carry the free data F, have eigenvalue 0.

So Costin's theorems cannot simply be quoted. What survives without nonresonance is the multisummability of *formal
power-series* solutions (Section 4.1), and that is enough to define the multipliers.

**Braaksma** (Ann. Inst. Fourier 42 (1992) 517–540): formal power-series solutions of nonlinear meromorphic
differential equations are multisummable. This is Écalle's theorem, proved without nonresonance assumptions; the
precise hypotheses should be checked against the paper.

**Singular perturbation in a parameter.** These are the results for $`\varepsilon y' = f(x, y, \varepsilon)`$ and their
doubly singular versions:

- Sibuya, *Gevrey property of formal solutions in a parameter* (1990) *(unverified)*;
- Canalis-Durand, Ramis, Schäfke & Sibuya, *Gevrey solutions of singularly perturbed differential equations*,
  J. reine angew. Math. 518 (2000) 95–129 *(page numbers unverified)*: Gevrey-1 asymptotics of the formal solution in ε
  on domains without turning points ($`\partial_y f`$ invertible);
- Canalis-Durand, Mozo-Fernández & Schäfke, *Monomial summability and doubly singular differential equations*,
  J. Differential Equations 233 (2007) 485–511: for $`\varepsilon^\sigma x^{r+1} y' = f(x, \varepsilon, y)`$ with f(0,0,0) = 0 and
  $`\partial_y f(0,0,0)`$ invertible, the unique formal solution is summable in the monomial $`\varepsilon^\sigma x^r`$.

**Exact WKB.** Nikolaev, CMP (2023), arXiv:2106.10248: existence and uniqueness of exact WKB solutions, and Borel
summability of the formal WKB solutions, for second-order linear singularly perturbed ODEs. This is the linear model of
B6. For the error control of Liouville–Green (WKB) approximations up to a regular singular point, Olver's
*Asymptotics and Special Functions* (1974, Chapters 10–12) is the standard reference.

**Inner–outer matching.** Baldomá, Fontich, Guardia & Seara, arXiv:1201.5152 *(journal unverified)*: rigorous matching
of inner-equation solutions to outer solutions in exponentially small splitting problems, in overlapping complex domains
near a singularity. This is the template for B5.

## 3. The formal solution and the slow–fast structure

### 3.1 Lemma B1(M): the formal solution (proved)

**Lemma B1(M).** For every smooth F with odd harmonics |μ| ≤ M, every ω₀ and every choice of the free data
(the ω_k and K_k, also truncated), the M-harmonic model has exactly one formal solution with the structure of
Proposition 1: integer powers of ε, $`p, m \propto \varepsilon^{1/2}`$, every order polynomial in x, regular at the centre,
analytic at the light cone, with G_k(1) = 0.

*Proof.* Repeat the induction of Proposition 1 with $`P_M`$ applied to each right-hand side. Steps 1–5 (B, G, S, D and the
symmetry m(x) = p(−x)) use only:
- algebraic operations and $`\partial_\theta`$;
- integration in x;
- the parity of x-polynomials.

$`P_M`$ acts on the θ-dependence only, commutes with all of these, and maps even (odd) x-polynomials with periodic
coefficients to the same class. Step 6 (harmonic parity) holds because $`P_M`$ preserves the odd and even harmonic classes.
$`\square`$

The free data F_n(θ) = F + εK₁ + ε²K₂ + … is the coefficient of x in p − m at the centre, i.e. the profile to all
orders; the K_k are its 1/n corrections, not extra unknowns.

### 3.2 Lemma B2(M): slow–fast structure (proved)

Write the system as $`Y_x = n\,\mathcal F_1(x, Y) + \mathcal F_0(x, Y, \partial_\theta Y)`$, with Y = (p, m, b, c) and
$`\mathcal F_1`$ collecting the terms of order n. The slow manifold of the formal solution is b = 1 + O(ε²), p + m = O(ε^{3/2}).

**Lemma B2(M).** On the slow manifold, the Jacobian of $`\mathcal F_1`$ with respect to (p, m, b, c) has eigenvalues

```math
\lambda_u = \frac{c^2}{x\,(x^2-c^2)}, \qquad \lambda_b = -\frac1x, \qquad 0, \qquad 0,
```

for each harmonic (the n-terms contain no θ-derivative, so at leading order they act diagonally on harmonics, with
c = 1 + O(ε)). The λ_u eigenvector is the combination u = p + m (with a b-component), the λ_b eigenvector is b − 1, and
v = p − m and c are slow. At c = 1, $`\lambda_u = -\Phi'(x)`$ with Φ = ln x − ½ ln(1 − x²), the singulant of Proposition 3.

*Proof.* Direct computation. Read the four right-hand sides off the equations as functions of (p, m, b, c), take the
Jacobian, divide by n, take n → ∞, and set b = 1, m = −p. A sympy script that does this and prints the eigenvalues
(c²/(x(x² − c²)), −1/x, 0, 0) is `solver/proof/fast_eigenvalues.py`. $`\square`$

**Consequences.**

1. **No turning points on the real axis** except at ∞. On 0 < x < 1 and 1 < x < ∞ (with c ≈ 1), λ_u and λ_b are finite and
   non-zero.
2. **x = 0 and x = c(1) = 1 are doubly singular points** of type $`\varepsilon\,(x - x_0)\,y' = f`$. Near x = 0,
   $`\varepsilon x\,u' \approx -u`$ and $`\varepsilon x\,(b-1)' \approx -(b-1)`$. Near x = 1, $`\varepsilon\,(1 - x)\,u' \approx \tfrac12 u`$. In
   Canalis-Durand–Mozo-Fernández–Schäfke form these have σ = 1, r = 0 and an invertible linear part in the fast
   components; the slow components (v, c) are regular there. This is a mixed system, which their theorem does not cover
   verbatim (Section 4.2).
3. **x = ∞ is a turning point.** As x → ∞, $`\lambda_u \sim n/x^3`$ and $`\lambda_b = -n/x`$, while the slow terms
   are of size ~1/x (the $`\omega x\,\partial_\theta`$ and $`x^2\partial_x`$ structure) and εx²F² grows. Fast and slow
   become comparable at x ~ √n, where n/x³ ~ 1/x · n/x². That is the inner scaling ξ = x/√n. So the inner problem is
   the turning-point (inner) equation of this singular perturbation problem, as in Xie–Tanveer.
4. **The fast eigenvalue is repeated across harmonics.** At leading order the fast block is λ_u times the identity on the
   N odd harmonics. The degeneracy is lifted at order 1 by the θ-transport ($`\omega x\,\partial_\theta`$ and the gravity
   coupling −2x²F²). This produces the characteristics θ = θ₀ + ω ln(1 − x²) of Proposition 3, and the matrix structure
   of the multipliers.

## 4. Gap (a): Borel summability

Two different summability statements are needed:
- one for the inner problem in its own variable, which is what defines the multipliers;
- one for the outer problem in the large parameter n, which is what connects them to finite n.

### 4.1 The inner problem: the multipliers are well defined (cited)

With $`\xi = x/\sqrt n`$, u = ξ², and the M-harmonic truncation, the inner problem

```math
L y = c\,\sigma,\qquad L\sigma = c\,\xi y_\xi + c\,\big(\tfrac12 y^2 - 1\big)\,y,\qquad \xi c_\xi = \tfrac12 c\,y^2,\qquad L = \xi^2\partial_\xi + \omega\,\xi\,\partial_\theta
```

becomes, for the harmonic components of $`y = \xi\,Y(u)`$, $`\sigma = \xi^2\Sigma(u)`$, c = C(u), a finite system

```math
u^2\,\frac{dZ}{du} = H(u, Z), \qquad Z = (Y_\mu, \Sigma_\mu, C_\mu)_{\lvert\mu\rvert \le M},
```

with H polynomial in (u, Z); products of fields are convolutions over harmonics, truncated by $`P_M`$. Explicitly, with
$`\xi\partial_\xi = 2u\,\partial_u`$ and $`\partial_\theta \to i\mu`$, the first equation is $`2uY_u + (1 + i\mu\omega)Y = (C\Sigma)_\mu`$, which is
regular-singular. The c equation is $`C_u = \tfrac14 (CY^2)`$, which is regular. Eliminating $`Y_u`$ from the second equation
leaves $`2u^2\Sigma_u = (C^2\Sigma)_\mu + O(u)`$. So u = 0 is an irregular singular point of Poincaré rank 1, the fast
exponential is $`e^{-1/(2u)}`$ (rate ½ in 1/u for every harmonic at leading order, C = 1 + O(u)), and Y, C are slow. Its formal power-series solution with Y(0) = 2F, C(0) = 1 is the
inner series of the README, $`y = \sum_j d_j\xi^{2j+1}`$, and so on. It is unique once F is given (proved by the recursion,
`solver/largeD_inner.py`).

**Lemma B4(M) (cited).**
- **Summability.** For fixed M, F and ω, the formal solution is multisummable (Braaksma 1992). Its only exponential is
  $`e^{-1/(2u)}`$ (the fast eigenvalue is −½ in the variable 1/u for every harmonic; the slow directions carry the
  free data), so the only level is 1, and the series is 1-summable in every direction except arg u = 0.
- **The multipliers.** Along arg u = 0 (real ξ, i.e. the Stokes line real x > 1) the two lateral sums differ by a
  transseries in $`e^{-1/(2u)}`$. The vector of coefficients of its first term, expressed in a basis of the first-order
  fast solutions $`e^{-1/(2u)}u^{\langle F^2\rangle}e^{\Pi(\theta)}e^{i\nu\theta_0}`$, is the multiplier vector $`S_\nu(F, \omega)`$.
- **Smoothness.** S depends analytically on (F, ω), since the sums depend analytically on parameters entering H
  analytically (standard for Borel–Laplace sums with uniform bounds).

*Hypothesis check.* Braaksma's theorem needs a meromorphic system at the singular point and a formal power-series
solution. Both hold (H rational in u, polynomial in Z; the recursion gives the series). The repeated fast eigenvalue
and the zero eigenvalues that block Costin's sharper theorems do not matter here, because only the formal power series,
not a general transseries, is summed. *To check in the paper:* whether Braaksma's statement covers parameter-analytic
dependence of the sums, which is needed for B7. If not, Balser's book (Formal power series and linear systems of
meromorphic ODEs, Springer 2000) and its nonlinear extensions give it *(unverified)*.

*Remark.* The repeated fast eigenvalue makes the first exponential term a vector over harmonics. Its N complex
coefficients are the $`S_\nu`$, which is the count the selection needs. This is the rigorous version of the README's "the
multiplier is a matrix over harmonics".

### 4.2 The outer problem: Gevrey asymptotics in ε (cited, with two hypotheses to verify)

**Lemma B3(M).** Fix 0 < δ < ½ and (F_n, ω) in a compact set. For n ≥ n₀ the M-harmonic model has a unique solution on
0 ≤ x ≤ 1 − δ that is regular at the centre, has the prescribed profile F_n (coefficient of x in p − m at x = 0) and
satisfies c(1 − δ) = c_formal(1 − δ). It is asymptotic to the formal solution of B1(M), in the Gevrey-1 sense, uniformly
on [δ, 1 − δ]: the remainder after K terms is bounded by $`C\,A^K K!\,\varepsilon^{K}`$.

*Argument.*
- **Away from the singular points.** On [δ, 1 − δ] the fast eigenvalues are bounded away from 0 and ∞ (B2), so the
  standard theory of singularly perturbed analytic systems without turning points gives Gevrey-1 asymptotics of the slow
  formal solution and an actual solution exponentially close to its optimal truncation, by the Ramis–Sibuya theorem:
  Sibuya 1990 and Canalis-Durand–Ramis–Schäfke–Sibuya 2000. Those results are stated for $`\varepsilon y' = f`$ with
  $`\partial_y f`$ invertible. Our system mixes fast components (u, b − 1) with slow ones (v, c); the standard reduction is
  to the slow manifold, whose Gevrey-1 character is the same statement, and then a regular-perturbation problem for
  the slow components. *A precise citation for the mixed case should be pinned down.*
- **At x = 0.** The doubly singular structure ($`\varepsilon x\,u' \approx -u`$, invertible) is the setting of
  Canalis-Durand–Mozo-Fernández–Schäfke 2007, with σ = 1, r = 0, so monomial summability reduces to summability in ε.
  It gives the unique solution regular at x = 0 for the fast components.
  - The slow components are regular-singular in x, with exponent 1 for v (the free profile F_n) and 0 for c (fixed by
    the gauge).
  - Fast-mode regularity excludes $`x^{2-n}`$ and $`x^{-(n-1)}`$, the singular branches of u and b − 1.
- **What is needed beyond the citations.**
  - A written proof that the mixed slow–fast system satisfies the conclusions. This is routine: the fast–slow splitting
    is block-triangular at leading order, B2.
  - The uniqueness statement, by a contraction in a weighted space on [0, 1 − δ]: the singular branches $`x^{2-n}`$ and
    $`x^{-(n-1)}`$ are excluded by the weight $`x^{-1}`$ at the centre.

**Borel singularities: two hypotheses to verify.** For the stronger statement used in Section 6, that the
centre-regular solution is the 1-sum in ε along arg ε = 0 on (0, 1) (the "half problem" solution), the Borel transform
in ε must have no singularity on the positive real axis for x ∈ (0, 1). Known singularities:

- **u-mode:** $`\chi_\ell(x) = \Phi(x) - i\pi(\ell + \tfrac12)`$. Its imaginary part is −π(ℓ + ½) ≠ 0 for real x ∈ (0, 1),
  so it is never on the positive real axis. **Checked** (Proposition 4, linear case; the nonlinear late orders confirm the
  singulant numerically, README).
- **(H-a1) b-mode.** λ_b = −n/x gives a singulant of the form ln(x/x_b), with x_b where the mode is generated. Since b − 1
  is slaved to p² + m² (Riccati), x_b is a singularity of the forcing, i.e. of the u-mode Borel plane. Hypothesis: these
  singularities are of the form $`\chi_\ell(x_*) + \ln(x/x_*)`$ and stay off the positive real axis for x ∈ (0, 1).
  *Test:* the late orders of B_k at x = 0.5, 0.8, 0.95 from the sympy expansion. Note that ln x → 0 at x = 1: a b-mode
  singularity approaching the origin near the light cone would make the series non-uniform there, which Proposition 1's
  polynomial structure makes unlikely but does not exclude.
- **(H-a2) x-independent singularities.** The README reports singularities of modulus about 0.13 in the nonlinear outer
  late orders, driven by high harmonics. Hypothesis: their arguments are not 0. In the linear case the analogous ±iπ
  are on the imaginary axis. *Test:* read the argument of the dominant singularity of the outer late orders at fixed
  harmonic content, with the existing late-order scripts.

If (H-a2) failed (a singularity on the positive real axis at modulus a ≈ 0.13), the 1-sum along arg ε = 0 would not
exist. The centre-regular solution would still exist (Lemma B3 needs only Gevrey asymptotics), but its exponentially
small content would include a term ~$`e^{-an}`$. The README's argument that such terms only shift the free data
(F's harmonics, K_k) would then have to be proved. They are x-independent, so they act like a change of the constant
data.

## 5. Gap (c): the light cone

### 5.1 The two local descriptions

Linearize the M-harmonic model around the centre-regular solution near x = 1. Since c(1) = 1 and x − c has a simple zero
there, x = 1 is a **regular singular point** of the linearized system.

**(i) Frobenius.** For each odd μ, |μ| ≤ M, there is a solution

```math
\phi_\mu = (1-x)^{\Lambda_\mu}\big(e_\mu + O(1-x)\big), \qquad
\Lambda_\mu = \frac{\tfrac n2 - 1 - i\mu\omega + \langle G(1)\rangle}{1 - \langle G(1)\rangle} + O(\text{coupling}),
```

This is the README's exact local analysis (Proposition 3, Consequence 1). The coupling term is O(1/n): the
harmonics of G(1) mix the μ-components, but its diagonal part ⟨G(1)⟩ dominates. All other fundamental solutions are
analytic at x = 1, with integer exponents. The coefficient of $`\phi_\mu`$ in the actual solution is $`C_{n,\mu}`$, and
analyticity at the light cone is $`C_{n,\mu} = 0`$ for all μ.

**(ii) WKB.** On 0 < x < 1 the fast mode is $`e^{-n\Phi}\big(A + O(1/n)\big)`$, with A transported along characteristics
(Proposition 3). Near x = 1, $`e^{-n\Phi} = x^{-n}(1-x^2)^{n/2}`$ and A ∝ (1 − x)⁻¹ (1 − x²)^{⟨F²⟩} e^{Π}. With
⟨G(1)⟩ = 2⟨F²⟩/n + O(n⁻²) (README, Proposition 3, both → 3.70), the exponents agree:
Λ_μ = n/2 − 1 − iμω + ⟨F²⟩ + O(1/n).

### 5.2 Lemma B6(M): matching at the light cone (reduced to standard asymptotic integration)

**Lemma B6(M).** Let $`W_\mu`$ be the WKB fast solution with label harmonic μ, normalized at x = ½. Let $`\tilde W_\mu`$ be the
same solution written with the resummed exponent, i.e. with phase $`n\tilde\Phi`$, where $`\tilde\Phi'`$ is the exact fast
eigenvalue $`-c^2/(x(x^2-c^2))`$ of the actual background, and with the order-1 transport including ⟨G(1)⟩. Then:

1. $`\tilde W_\mu`$ approximates an exact solution of the linearized system with relative error O(1/n), uniformly on
   [½, 1), including the limit x → 1.
2. That exact solution equals $`k_{n,\mu}\,\phi_\mu`$ + (solution analytic at x = 1), with $`k_{n,\mu} = k_\mu^{(0)}\,(1 + O(1/n))`$ and
   $`k_\mu^{(0)} \ne 0`$ explicit from the transport.
3. In the linear case (c = b = 1, one harmonic), $`W_\mu`$ is an exact solution, $`\delta p - \delta m = 2x^{2-n}(1-x^2)^{n/2-1-s}`$,
   and the identification is exact (Propositions 2–3).

*Proof sketch.*

- **Step 1: why the plain WKB series is not uniform, and the fix.** The plain WKB series in 1/n is not uniform at x = 1.
  The transport operator is Euler-type there ($`(1-x)\partial_x`$), so the O(1/n) corrections to the exponent appear as
  terms $`n^{-k}\ln^k(1-x)`$, and the series holds only while |ln(1 − x)| ≪ n. Writing the exponent exactly removes this.
  Use the phase $`n\tilde\Phi`$ with the true fast eigenvalue of the background, and put ⟨G(1)⟩ into the transport
  exponent, so that $`\tilde W_\mu \sim (1-x)^{\Lambda_\mu}`$ exactly.
- **Step 2: the error equation.** The error of the Liouville–Green approximant satisfies a Volterra integral equation
  whose kernel is controlled by the error-control function (Olver, Ch. 10 §§2–4, written for second-order scalar
  equations; for first-order systems the same Volterra argument works after diagonalizing the fast block, Levinson /
  Wasow). Near x = 1 its integrand is O(1/n) times (1 − x)^{-1+γ} for some γ > 0, once the logarithmic term is absorbed
  in Λ_μ. The integral converges at x = 1, so the relative error is O(1/n) uniformly up to the endpoint. This is the
  "LG approximation at a regular singularity" case in Olver's book.
- **Step 3: uniqueness.** Λ_μ is not an integer, since Im Λ_μ = −μω/(1 − ⟨G(1)⟩) ≠ 0. And Λ_μ − Λ_ν ∉ ℤ for μ ≠ ν
  (including ν = −μ), since their imaginary parts differ by about (μ − ν)ω, at least 2ω in modulus. The O(1/n) coupling
  through the harmonics of G(1) moves the exponents (eigenvalues of the indicial matrix) by O(1/n), which cannot close
  a gap of 2ω. So the solutions of the linearized system behaving like $`(1-x)^{\Lambda_\mu}`$ at x = 1 form a
  one-dimensional space modulo solutions analytic at x = 1. The exact solution of step 2 has exactly this behaviour,
  which gives 2.
- **Step 4: the constant.** $`k_{n,\mu}`$ is read off by comparing the leading terms at x → 1. Its non-vanishing follows
  from Proposition 3, Consequence 2: the transport factor never vanishes. $`\square`$

The non-resonance in step 3 is the structural fact that makes gap (c) a standard matching. What is not written is the
explicit Volterra bound for the M-harmonic system. It is routine, but it needs the coupling matrix of G(1) harmonics.

### 5.3 The half-monodromy

Continue $`\tilde W_\mu`$ from x < 1 to x > 1 through the upper or lower half plane. The factor $`(1-x)^{\Lambda_\mu}`$ picks up
$`e^{\mp i\pi\Lambda_\mu}`$, and the label θ₀ − ω′ln(1 − x) shifts by ∓iπω′, which multiplies harmonic μ by $`e^{\pm\pi\mu\omega'}`$. This is the
README's map (3). It is an exact identity for the Frobenius branch, so once B6 holds it needs no separate proof.

## 6. The selection theorem

**Theorem B(M).** Fix odd M and let (F*, ω*) be a zero of the inner multipliers, S_μ(F*, ω*) = 0 for all odd μ ≤ M,
at which the Jacobian of (Re S_μ, Im S_μ) with respect to (F's harmonics, ω), with the phase of F₁ fixed, is invertible.
Assume B3–B6 (Sections 4–5) and the global continuation of Section 7. Then the following hold.

*(1) Characterization.* For n ≥ n₀, the centre-regular solution with data (F_n, ω) near (F*, ω*) is analytic at the light
cone exactly when, for all odd μ ≤ M,

```math
C_{n,\mu}(F_n, \omega) = \kappa_{n,\mu}\,\big[S_\mu(F_n^{(0)}, \omega) + R_{n,\mu}(F_n, \omega)\big] = 0 .
```

Here $`\kappa_{n,\mu} \ne 0`$ is explicit, and $`R_{n,\mu} = O(1/n)`$ in C¹, uniformly near (F*, ω*). $`F_n^{(0)}`$ is the leading part of
the profile; the 1/n corrections $`K_k`$ enter only through R. In the linear case the factor is the one in the large-n form
of C(n, s) (README),

```math
\kappa_{n,\mu} \propto -2\pi i\,(n+1)^{s+1/2}\,e^{-i\pi\Lambda_\mu}, \qquad s = i\mu\omega ,
```

times the transport and matching factors of B5–B6.

*(2) Existence and uniqueness of the branch.* There is a unique solution $`(F_n, \omega_n)`$ of these equations near
(F*, ω*), with $`F_n - F_* = O(1/n)`$ and $`\omega_n - \omega_* = O(1/n)`$. It gives a discretely self-similar solution of the
M-harmonic model, regular at the centre and analytic at the light cone.

*(3) Converse.* If a sequence of such solutions has $`(F_n, \omega_n) \to (F_\infty, \omega_\infty)`$, then $`S(F_\infty, \omega_\infty) = 0`$.

*Proof given the lemmas.* Divide by $`\kappa_{n,\mu}`$, which is allowed because it is non-zero. The map
$`Z \mapsto S(Z) + R_n(Z)`$ on a neighbourhood of Z* = (F*, ω*) in ℝ^{2N} is C¹-close to S, which has an invertible
derivative at Z*. The finite-dimensional implicit function theorem, or a quantitative inverse-function theorem since
‖R_n‖_{C¹} = O(1/n), gives a unique zero Z_n with |Z_n − Z*| = O(1/n). The converse follows by continuity:
$`S(Z_n) = -R_n(Z_n) = O(1/n) \to 0`$. $`\square`$

*Where the lemmas enter.* The first equality in 1 is the chain:
- (light-cone coefficient) = $`k_{n,\mu}`$ × (coefficient of the fast mode just inside x = 1), by B6;
- = $`k_{n,\mu}`$ × transport factor × (fast-mode content generated on x > 1), by Proposition 3 and §5.3;
- = $`k_{n,\mu}`$ × transport × (outer Stokes jump on the line x > 1), by B3 and the global continuation;
- = $`k_{n,\mu}`$ × transport × matching × $`n^{(\dots)}`$ × [S_μ + O(1/n)], by B5 with B4.

Every factor is non-zero and explicit at leading order, and the O(1/n) is uniform in C¹ near Z*.

*The non-degeneracy hypothesis is what layer A certifies.* Numerically, at M = 5 the Schur-complemented 2×2 outer
Jacobian has singular values about 6.8×10³ and 3.9×10³, and the inner levels are invertible (hier.py).

*Why O(1/n), not O(n^{−1/2}).* The outer equations depend on n only through n and x², and the inner variable is
ξ² = x²/n. The inner expansion therefore proceeds in powers of 1/n. In the linear case the correction is $`O(s^2/n)`$, from
the large-n form of C(n, s). This matches the double-double finding that Δ(D) is analytic in 1/D.

## 7. What remains

**Open, and the main work: the global continuation (Xie–Tanveer's step 2).**
- Prove that the centre-regular solution of B3, continued through a complex domain $`\mathcal D_n`$, exists with uniform
  bounds on all of $`\mathcal D_n`$. The domain must:
  - contain the real segment [δ, 1 − δ];
  - wrap around x = 1 through the upper half plane;
  - reach the Stokes line x > 1;
  - extend to |x| ~ √n.
- The method is a contraction in weighted spaces of functions analytic on $`\mathcal D_n`$. The weights must track the
  exponentially large fast mode $`e^{-n\Phi}`$ where Re Φ < 0, i.e. x < 1/√2 on the real axis. In Xie–Tanveer this is the
  longest part of the proof.
- The M-harmonic system adds the N-fold fast block, which by B2 is λ_u times the identity at leading order. So the
  contraction can be set up harmonic by harmonic with order-1 coupling. That is harmless for fixed M; uniformity in M is
  gap (b).

**Open: B5, inner–outer matching for the nonlinear system.**
- Show that on the overlap $`1 \ll x \ll \sqrt n`$ (in the sector around the Stokes line) the outer solution equals the
  rescaled inner lateral sum up to O(1/n), so that the outer Stokes jump equals the inner one times the explicit factor
  $`n^{s+1/2}`$ (the power of the matching).
- In the linear case this is proved (Proposition 4 with the Weber constant).
- For the nonlinear case the template is Baldomá–Fontich–Guardia–Seara: solve the inner equation in a complex domain
  near ξ = 0, solve the outer equation in a domain avoiding x = ∞, and bound the difference in the overlap with explicit
  constants.

**To verify (cheap, numerical):**
- (H-a1): the b-mode Borel singularities stay off ℝ⁺ on (0, 1);
- (H-a2): the argument of the x-independent outer singularities (modulus about 0.13);
- none for the non-resonance at x = 1: Λ_μ − Λ_ν ∉ ℤ is automatic (B6, step 3).

**To write (routine):**
- the mixed slow–fast version of B3 (or a precise citation);
- the Volterra error bound of B6 for the M-harmonic coupling;
- parameter-analyticity of the inner lateral sums for B4 (a citation check in Braaksma or Balser).

**Not addressed here: uniformity in M (gap (b)).** Every constant above (n₀, the O(1/n), the contraction radii)
depends on M. The selection theorem for the full problem needs them uniform in M, or a strip-analytic formulation in θ.

**Net assessment.**
- Gap (c) is reduced to standard asymptotic integration, with the structural facts proved (exponent agreement,
  non-integrality, uniqueness, exact linear case).
- Gap (a) is reduced, for the inner problem, to Braaksma's theorem. That makes the multipliers rigorously defined
  objects for every M, which layer A also needs.
- For the outer problem, gap (a) is reduced to standard Gevrey theory plus two checkable hypotheses (H-a1, H-a2).
- What stands between this note and a theorem for fixed M is the global continuation and the nonlinear inner–outer
  matching. Both are research-level but follow published templates. With them, Theorem B(M) holds for every fixed M,
  and with layer A at M = 5 it would give a rigorous five-harmonic selection ω_n → ω*(M = 5).

## References

- O. Costin, *On Borel summation and Stokes phenomena of nonlinear differential systems*, Duke Math. J. 93 (1998)
  289–344 *(journal details unverified)*, arXiv:math/0608408.
- O. Costin, *Exponential asymptotics, transseries, and generalized Borel summation for analytic, nonlinear, rank-one
  systems of ODEs*, arXiv:math/0608414 (Int. Math. Res. Notices 1995 *(unverified)*). Normal form
  $`y' = f_0 - \hat\Lambda y - x^{-1}\hat B y + g`$, $`\hat\Lambda`$ invertible, eigenvalues of distinct arguments.
- O. Costin and R. D. Costin, *On the formation of singularities of solutions of nonlinear differential systems in
  antistokes directions*, Invent. Math. 145 (2001) *(volume unverified)*, arXiv:math/0202234.
- B. L. J. Braaksma, *Multisummability of formal power series solutions of nonlinear meromorphic differential
  equations*, Ann. Inst. Fourier 42 (1992) 517–540.
- M. Canalis-Durand, J.-P. Ramis, R. Schäfke and Y. Sibuya, *Gevrey solutions of singularly perturbed differential
  equations*, J. reine angew. Math. 518 (2000) 95–129 *(pages unverified)*.
- M. Canalis-Durand, J. Mozo-Fernández and R. Schäfke, *Monomial summability and doubly singular differential
  equations*, J. Differential Equations 233 (2007) 485–511, doi:10.1016/j.jde.2006.11.005.
- Y. Sibuya, *Gevrey property of formal solutions in a parameter*, in Asymptotic and Computational Analysis (1990)
  *(unverified)*.
- N. Nikolaev, *Existence and uniqueness of exact WKB solutions for second-order singularly perturbed linear ODEs*,
  Commun. Math. Phys. (2023), doi:10.1007/s00220-022-04603-7, arXiv:2106.10248.
- F. W. J. Olver, *Asymptotics and Special Functions*, Academic Press 1974, Chapters 10–12 (Liouville–Green
  approximation with error bounds, including at singularities).
- W. Balser, *Formal Power Series and Linear Systems of Meromorphic Ordinary Differential Equations*, Springer 2000.
- X. Xie and S. Tanveer, *Rigorous results in steady finger selection in viscous fingering*, Arch. Ration. Mech. Anal.
  166 (2003) 219–286 *(unverified)*; X. Xie, *Rigorous results in existence and selection of Saffman–Taylor fingers by
  kinetic undercooling*, arXiv:1610.00076.
- I. Baldomá, E. Fontich, M. Guardia and T. M. Seara, *Exponentially small splitting of separatrices beyond Melnikov
  analysis: rigorous results*, arXiv:1201.5152.

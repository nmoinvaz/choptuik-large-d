# Layer B for the M-harmonic model: the selection theorem

This note works out layer B of [PROOF.md](../PROOF.md) (the analytic selection theorem) as far as it can be taken now,
for the **M-harmonic model**: the large-D self-similar equations with the θ-dependence truncated to the odd harmonics
|μ| ≤ M of p and m and the even harmonics |ν| ≤ M + 1 of b and c (Section 1). Gap (b) of the README (uniformity in M) and layer A
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
- [8. New (second pass): the outer Borel plane (superseded by B3_borel.md)](#8-new-second-pass-the-outer-borel-plane-superseded-by-b3_borelmd)
- [9. New (second pass): Lemma B5.0, the inner limit is reached in powers of 1/n (proved)](#9-new-second-pass-lemma-b50-the-inner-limit-is-reached-in-powers-of-1n-proved)
- [10. New (second pass): Lemma B5, the inner–outer matching](#10-new-second-pass-lemma-b5-the-innerouter-matching)
  - [10.1 Lemma B5.1: the fast-mode variable and the matching factor (proved)](#101-lemma-b51-the-fast-mode-variable-and-the-matching-factor-proved)
  - [10.2 Domains and norms](#102-domains-and-norms)
  - [10.3 Statement and proof](#103-statement-and-proof)
  - [10.4 Status of B5](#104-status-of-b5)
- [11. New (second pass): Lemma B8, the global continuation](#11-new-second-pass-lemma-b8-the-global-continuation)
- [12. New (second pass): Lemma B6 completed, with the light-cone exponents and the Volterra bound](#12-new-second-pass-lemma-b6-completed-with-the-light-cone-exponents-and-the-volterra-bound)
- [13. Status](#13-status)
  - [13.1 Theorem B(1): the status of the chain](#131-theorem-b1-the-status-of-the-chain)
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
  - *Outer problem: no Borel sum needed.* The outer formal series is Gevrey-1 on every disc in x, for every M
    ([B3_borel.md](B3_borel.md), Lemma B3.7). The actual centre-regular solution with fixed-order asymptotics is Lemma B8
    restricted to the real segment. That it is also the Borel sum of the outer series is hypothesis (H-σ) (Section 4.2),
    which Theorem B(M) does not use.
- **Gap (c): reduced to standard asymptotic integration** (Lemma B6; the structural facts are proved, the error
  estimates are routine but not written).
  - Because x − c has a simple zero at the light cone, the WKB phase and the Frobenius exponent are the same object. The
    WKB fast mode, written with the resummed exponent Λ_μ, is an approximation with uniformly bounded relative error up
    to and including x = 1, in the sense of Olver's error control.
  - The Frobenius singular branch of exponent Λ_μ is unique, because Λ_μ is not an integer (Im Λ_μ ≠ 0) and the Λ_μ
    differ for different μ.
  - So the matching constant is well defined, equals 1 + O(1/n), and equals exactly 1 in the linear case (Proposition 3).
- **The two contractions are written** (Sections 10–11): the global continuation B8 and the inner–outer matching B5(1) are
  proved for each fixed M and certified with explicit constants for M = 1 ([B8_constants.md](B8_constants.md),
  [B5_inner.md](B5_inner.md)). **Theorem B(1) is not yet proved**: Section 13.1 lists every link of the chain with its
  status and its threshold in n, and what is still missing.

| Step | Content | Status (details in Section 13) |
|---|---|---|
| B1(M) | formal 1/n solution for any (F, ω), Galerkin-truncated | **proved** (Proposition 1, Section 3.1) |
| B2(M) | slow–fast structure; fast eigenvalues; no turning points on the real axis; turning point at ∞ | **proved** (Section 3.2) |
| B3(M) | centre-regular actual solution, asymptotic to the formal series | formal half (Gevrey-1) **proved**; existence with fixed-order asymptotics = B8 on the real segment; the Borel-sum form is (H-σ), not needed (Section 4.2) |
| B4(M) | inner Stokes multipliers well defined, analytic in (F, ω) | **cited** (Braaksma); parameter-analyticity **proved** given Braaksma (Section 4.1) |
| B5(M) | outer Stokes jump on x > 1 equals the inner multiplier times an explicit factor | B5.0, B5.1 **proved**; B5(1) **proved given (H8)**, **certified** for M = 1; B5(2) **proved given B5(1)** (Sections 9–10) |
| B6(M) | light-cone matching: WKB fast mode ↔ Frobenius branch, half-monodromy | **proved** for fixed M up to a routine block-diagonalization (Sections 5, 12) |
| B7(M) | implicit function theorem | **proved** given B3–B6 in C¹ form (Section 6) |
| B8(M) | continuation through the complex domain with uniform bounds | **proved** for each fixed M; **certified** for M = 1 (Section 11) |

## 1. The M-harmonic model

With n = D − 2, ε = 1/n, $`G = (n-1)\,c\,(b-1)`$, the self-similar equations (README, "Large D") are

```math
x c_x = (n-1)\,c\,(b-1), \qquad x b_x = b\,(p^2+m^2) - (n-1)\,b\,(b-1),
```
```math
x(x-c)\,p_x = -\omega x\,p_\theta + G p - c p + \tfrac{n}{2}\,c\,(p+m), \qquad
x(x+c)\,m_x = -\omega x\,m_\theta - G m + c m - \tfrac{n}{2}\,c\,(p+m),
```

on 0 ≤ x ≤ 1, regular at x = 0, analytic at x = 1, with c(1) = 1. Let $`P_M`$ be the projection that keeps p and m on
the odd harmonics $`e^{i\mu\theta}`$, |μ| ≤ M, and b and c on the even harmonics |ν| ≤ M + 1 (the truncation of the inner
solvers and of Theorem A(M = 1); with |ν| ≤ M − 1 the inner limit would be a different problem, with c on {0} instead of
{0, ±2} at M = 1). The **M-harmonic model** applies $`P_M`$ to every right-hand side. For real fields it is a finite
analytic ODE system in x, with 4N + 2(2N + 1) = 8N + 2 real unknowns per x. Here N = (M + 1)/2 is the number of
positive odd harmonics.

Three remarks:
- Lemmas B5(1) and B8 ([B5_inner.md](B5_inner.md) §1.1, [B8_constants.md](B8_constants.md) §1.1) project every binary
  product onto its result's harmonics, e.g. Gp as [[c(b − 1)]p]. For M = 1 this is the same as projecting each
  right-hand side once; for M ≥ 3 it is not (B8_constants.md §11). Which of the two the general-M statements should use
  has not been decided.
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
differential equations are multisummable. This is Écalle's theorem, proved without nonresonance assumptions; its
hypotheses were checked against the paper ([B3_borel.md](B3_borel.md) §7.2).

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
u^2\,\frac{dZ}{du} = H(u, Z), \qquad Z = \big((Y_\mu, \Sigma_\mu)_{\lvert\mu\rvert \le M},\ (C_\nu)_{\lvert\nu\rvert \le M+1}\big),
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
not a general transseries, is summed. Braaksma's theorem (checked against the paper) is stated for
$`x^{\nu+1}y' = F(x, y)`$ without parameters, and neither it nor the abstract of Ramis–Sibuya (1994) addresses parameter
dependence. Analytic dependence on (F, ω), which B7 needs, follows from uniqueness on a ray where $`e^{-1/(2u)}`$ is large
and a contraction uniform in the parameters ([B3_borel.md](B3_borel.md) §7.2, Lemma B4.P, **proved** given Braaksma's
theorem; carried out for M = 1 on a box in A_certify.md §4.5).

*Remark.* The repeated fast eigenvalue makes the first exponential term a vector over harmonics. Its N complex
coefficients are the $`S_\nu`$, which is the count the selection needs. This is the rigorous version of the README's "the
multiplier is a matrix over harmonics".

### 4.2 The outer problem: Gevrey asymptotics in ε

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
  the slow components. No theorem read covers the mixed case: Canalis-Durand (1993) and
  Canalis-Durand–Mozo-Fernández–Schäfke (2007) assume every component fast ([B3_borel.md](B3_borel.md) §7.1). The
  formal half is proved directly (B3_borel.md, Lemma B3.7). The existence half with fixed-order asymptotics is B8
  restricted to the real segment.
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

**Borel singularities: hypothesis (H-σ).** For the stronger statement that the centre-regular solution is the 1-sum
in ε along arg ε = 0, the Borel transform in ε must continue, with exponential bounds, to a neighbourhood
$`S_\sigma = \{\zeta : \mathrm{dist}(\zeta, [0, \infty)) \lt \sigma\}`$ of the positive real axis, uniformly on compacts of
(0, 1] (Nevanlinna–Sokal). This is not used by B5–B8, which work with actual solutions and fixed-order truncations.
The earlier formulation, hypotheses H-a1 and H-a2 ("every singularity at |arg| ≥ 0.48π"), is false for x > 1/√2,
because the u-mode singulant Φ(x) − iπ/2 itself has argument down to −0.30π at x = 0.95; it is harmless, since that
singulant lies on Im ζ = −π/2. (H-σ) is proved in the linear case with σ = π/2, and near ζ = 0 for every M, where the
outer series is Gevrey-1 on every disc in x. That excludes the would-be 2Φ(x) singularity near x = 1/√2. Elsewhere it
is supported numerically. See [B3_borel.md](B3_borel.md) §§1–5.

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
harmonics of G(1) mix the μ-components, but its diagonal part ⟨G(1)⟩ dominates. *(Corrected in Section 12: the
coupling is O(1), since n·G(1) = 2F² + O(1/n), but the gauge e^{Π} removes it exactly, so the exponents and the
non-resonance below are unchanged.)* All other fundamental solutions are
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
  a gap of 2ω. *(Second pass: the coupling is O(1) but gauged away exactly; in the truncations the differences have
  |Im| ≥ 6.89 and |Re| < 0.02 (from `lightcone_exponents.py`: 6.894 and 0.0194). See Section 12.)* So the solutions of the linearized system behaving like $`(1-x)^{\Lambda_\mu}`$ at x = 1 form a
  one-dimensional space modulo solutions analytic at x = 1. The exact solution of step 2 has exactly this behaviour,
  which gives 2.
- **Step 4: the constant.** $`k_{n,\mu}`$ is read off by comparing the leading terms at x → 1. Its non-vanishing follows
  from Proposition 3, Consequence 2: the transport factor never vanishes. $`\square`$

The non-resonance in step 3 is the structural fact that makes gap (c) a standard matching. What is not written is the
explicit Volterra bound for the M-harmonic system. It is routine, but it needs the coupling matrix of G(1) harmonics.
*(Written in the second pass: Lemma B6′, Section 12.)*

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

*Note (P1, for M = 1).* The chain above (light-cone coefficient, transport along x > 1, Stokes jump) is not the one
that was closed. [B_chain.md](B_chain.md) §8.4 states Theorem B(1) with the selection function "label of the outer jump
at a fixed point u* of the Stokes line" in place of C_{n,μ}, which needs neither κ_{n,μ}, nor the transport, nor B6's
coefficient; instead it needs the three hypotheses at the light cone of §13.1.

*The non-degeneracy hypothesis is what layer A certifies.* Numerically, at M = 5 the Schur-complemented 2×2 outer
Jacobian has singular values about 6.8×10³ and 3.9×10³, and the inner levels are invertible (hier.py).

*Why O(1/n), not O(n^{−1/2}).* The outer equations depend on n only through n and x², and the inner variable is
ξ² = x²/n. The inner expansion therefore proceeds in powers of 1/n. In the linear case the correction is $`O(s^2/n)`$, from
the large-n form of C(n, s). This matches the double-double finding that Δ(D) is analytic in 1/D.

## 7. What remains

*Superseded in part by the second pass: see Section 13 for the current status.*

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

**To verify:**
- (H-σ), which replaces H-a1 and H-a2 (Section 4.2) and is not needed for Theorem B(M);
- none for the non-resonance at x = 1: Λ_μ − Λ_ν ∉ ℤ is automatic (B6, step 3).

**To write (routine):**
- the mixed slow–fast version of B3 (or a precise citation) *(since done: Section 4.2)*;
- the Volterra error bound of B6 for the M-harmonic coupling *(since done: Section 12)*;
- parameter-analyticity of the inner lateral sums for B4 (a citation check in Braaksma or Balser) *(since done:
  Section 4.1)*.

**Not addressed here: uniformity in M (gap (b)).** Every constant above (n₀, the O(1/n), the contraction radii)
depends on M. The selection theorem for the full problem needs them uniform in M, or a strip-analytic formulation in θ.

**Net assessment.**
- Gap (c) is reduced to standard asymptotic integration, with the structural facts proved (exponent agreement,
  non-integrality, uniqueness, exact linear case).
- Gap (a) is reduced, for the inner problem, to Braaksma's theorem. That makes the multipliers rigorously defined
  objects for every M, which layer A also needs.
- For the outer problem, gap (a) is reduced to standard Gevrey theory plus hypothesis (H-σ) (first stated as H-a1,
  H-a2), which Theorem B(M) turned out not to need.
- What stands between this note and a theorem for fixed M is the global continuation and the nonlinear inner–outer
  matching. Both are research-level but follow published templates. With them, Theorem B(M) holds for every fixed M,
  and with layer A at M = 5 it would give a rigorous five-harmonic selection ω_n → ω*(M = 5).

## 8. New (second pass): the outer Borel plane (superseded by B3_borel.md)

*Superseded by [B3_borel.md](B3_borel.md) §5: the double-precision Padé readings of this section's first version were
unreliable, and its table and its conclusions about H-a1 and H-a2 have been removed. The M = 1 "cluster" is the u-mode
singulant itself, which at x = 0.8 and 0.95 lies at |arg| 0.44π and 0.30π on Im ζ = −π/2. The x-independent
singularity is real for M ≥ 3 but lies near the negative real axis (ζ ≈ −1.2 at M = 3).*

What the first version did, and why it failed (B3_borel.md §5.1):
- It ran the outer 1/n recursion of `solver/largeD_late_orders.py` to order 36 in double precision (F = 2.72 sin θ,
  ω = ω∞, M = 1, 3, 5, 7 and untruncated) and read the nearest Borel singularities in ε off diagonal Padé approximants
  (`solver/proof/outer_borel.py`).
- *Precision.* At x = 1 the double-precision evaluation of D_k cancels coefficients of about 10⁴⁶ down to about 10³⁰.
- *The doublet rule.* `outer_borel.py` drops every pole with a Padé zero within relative distance 10⁻⁴ as a Froissart
  doublet. Genuine singularities of these series show up exactly as such close pole–zero pairs, so the rule removed
  the singularities it was looking for.
- *Too few terms.* Fed exact coefficients rounded to double and truncated to 36 terms, the procedure returns a ring of
  poles at the edge of the resolved disc, not the nearest singularity.

The recomputation in ball arithmetic (M = 1 to order 120, M = 3 to 100, M = 5 to 70; B3_borel.md §5.3) finds:
- M = 1: the only singularities near ℝ⁺ are the u-mode ones on Im ζ = ∓π/2; the x-independent ones lie near ±3.1i;
- M = 3: in addition, an x-independent singularity near ζ ≈ −1.22 ± 0.15i, opposite to ℝ⁺;
- M = 5: not resolved by 70 orders;
- nothing stable on or near ℝ⁺ for M = 1 and 3.

That recomputation uses the truncation with b and c on |ν| ≤ M − 1, each right-hand side projected once
(`b3_outer_m1.py`, `b3_outer_M.py`), not the model of Section 1. For M = 1 that means b and c θ-independent.

**The 2Φ(x) candidate.** The sum of the two conjugate u-mode singulants, 2Φ(x), is real and positive for x > 1/√2 and
reaches 0 at x = 1/√2. It is excluded near x = 1/√2 for every M by Lemma B3.7 (the outer series is Gevrey-1 on every
disc in x), and certified absent for x ∈ [0.7044, 0.7098] at M = 1 (B3_borel.md §6, for the truncation just named).
B3.5 explains why: sums of singularities from opposite sides of ℝ⁺ do not reach the principal sheet of a convolution
fixed point.

The b-mode remark of the first version stands as an analytic observation: the resonance between the free b-mode and the
forced response to the u-mode, x Φ′(x) = 1, happens only at x = 0, so the b-mode singulants are
χ_b(x) = ln x − iπ(ℓ + ½), whose imaginary part is never 0.

## 9. New (second pass): Lemma B5.0, the inner limit is reached in powers of 1/n (proved)

*Script: `solver/proof/inner_corrections.py` (sympy).*

**Lemma B5.0.** Write the exact self-similar equations (Section 1) in the inner variables x = √n ξ, y = p − m,
σ = √n (p + m), b = 1 + η/n, put ν = n^{−1/2}, and multiply the p − m equation by ν. They become

```math
\mathcal I(Z) + \nu^2\,\mathcal Q_1(Z) + \nu^4\,\mathcal Q_2(Z) = 0, \qquad Z = (y, \sigma, \eta, c).
```

𝓘 is the inner problem, and 𝓠₁, 𝓠₂ are polynomials in Z and its first derivatives. Only even powers of ν occur, so the
corrections are integer powers of 1/n:
- c equation: $`\xi c_\xi - c\eta`$, with correction $`+\nu^2 c\eta`$ (from n − 1);
- b equation: $`\eta - \tfrac12 y^2`$, with correction $`\nu^2(\xi\eta_\xi + \eta^2 - \tfrac12\eta y^2 - \eta - \tfrac12\sigma^2)`$ and a ν⁴ term;
- p − m equation: $`\xi^2 y_\xi + \omega\xi y_\theta - c\sigma`$, with correction $`\nu^2 c\,(\sigma - \xi\sigma_\xi - \eta\sigma)`$ and a ν⁴ term;
- p + m equation: $`\xi^2\sigma_\xi + \omega\xi\sigma_\theta - c\,\xi y_\xi + c\,(1 - \eta)\,y`$, with correction $`\nu^2 c\eta y`$.

With η = ½y², the leading equations are exactly the README's inner problem. *Proof:* the computation, done symbolically
by the script. Projection onto |μ| ≤ M commutes with every step, so the lemma also holds for the M-harmonic model
(checked for the model of Section 1 by `b5_system.py`, [B5_inner.md](B5_inner.md) §1.2). $`\square`$

*Remarks.*
- The η equation is singularly perturbed: its ν² term contains $`\xi\eta_\xi`$. This is the b fast mode of B2,
  $`\eta \sim x^{-n}`$. For |x| > 1 that mode is exponentially small in every direction, so η is slaved to y,
  $`\eta = \tfrac12 y^2 + n^{-1}\eta_1 + O(n^{-2})`$, by the usual slow-manifold reduction. It is not involved in any
  Stokes phenomenon on the domains below.
- **Consequence.** A solution of the inner problem solves the exact equations up to an explicit residual
  $`n^{-1}\mathcal Q_1 + O(n^{-2})`$. The inner multipliers therefore receive corrections in powers of 1/n, the first
  one, $`S^{(1)}`$, coming from the linearized inner problem forced by 𝓠₁. This answers Section 6's question
  "O(1/n) or O(n^{−1/2})" at the level of the equations: no half-integer power occurs.

## 10. New (second pass): Lemma B5, the inner–outer matching

*Template: Baldomá, Fontich, Guardia and Seara (arXiv:1201.5152).* Their method has three parts:
- the outer parametrizations are controlled in complex "boomerang" domains that reach close to the singularities, beyond
  the Stokes direction;
- an inner equation is solved near the singularity;
- the two are matched in an intermediate region, and the difference of two inner solutions replaces the Melnikov
  function.

Here the singularity is at x = ∞, the inner equation is the inner problem of Section 4.1, and the role of
"distance κε from the singularity" is played by |u| = κ₁/ln n, with u = x²/n.

### 10.1 Lemma B5.1: the fast-mode variable and the matching factor (proved)

For |x| > 1 put

```math
w(x) = -\tfrac12\ln\big(1 - x^{-2}\big), \qquad u = \xi^2 = x^2/n .
```

**(1) Half-monodromy of the phase.** Continuing Φ = ln x − ½ ln(1 − x²) from 0 < x < 1 to x > 1 through the upper or
lower half plane gives Φ = w ± iπ/2. So $`e^{-n\Phi} = e^{\mp i n\pi/2}\,e^{-nw}`$.

**(2) The phase in inner variables.** With x² = nu,

```math
n w = \frac1{2u} + \frac1{4nu^2} + \frac1{6n^2u^3} + \dots, \qquad e^{-nw} = e^{-1/(2u)}\,\Big(1 + O\big(\tfrac1{n\lvert u\rvert^2}\big)\Big).
```

The relative correction is $`O(1/(n\lvert u\rvert^2))`$, not $`O(1/(n\lvert u\rvert))`$. It is small only for $`\lvert u\rvert \gg n^{-1/2}`$, i.e.
$`\lvert x\rvert \gg n^{1/4}`$, and this sets the outer edge of the matching domain (below).

**(3) The amplitudes.** The fast mode of Proposition 3 has $`\delta p = e^{-n\Phi}A`$ and $`\delta m = e^{-n\Phi}\tfrac{1-x}{1+x}A`$ with
$`A = A_0(\theta_0)\tfrac{x}{1-x}(\dots)`$. So it contributes to y and σ the factors $`-2x^2/(x^2-1) = -2\,(1 + \tfrac1{nu} + \dots)`$ and
$`-2\sqrt n\,x/(x^2-1) = -2u^{-1/2}(1 + \tfrac1{nu} + \dots)`$. These are again series in 1/n.

**(4) The label and gravity factors.** Along the characteristic, the label continues as

```math
\theta_0 = \theta - \omega\ln(1 - x^2) = \theta - 2\omega\ln\xi - \omega\ln n \pm i\pi\omega + O\big(\tfrac1{nu}\big),
```

and the gravity factor contains

```math
(1 - x^2)^{\langle F^2\rangle} = e^{\mp i\pi\langle F^2\rangle}\,n^{\langle F^2\rangle}u^{\langle F^2\rangle}\Big(1 - \frac{\langle F^2\rangle}{nu} + \dots\Big).
```

Items 2–4 are printed by `inner_corrections.py`.

**Conclusion.** In inner variables, the outer fast mode with label harmonic μ, in the gauge $`\tilde A_0 = A_0e^{-\Pi}`$, is the inner fast
mode $`e^{-1/(2u)}u^{\langle F^2\rangle}e^{\Pi(\theta)}e^{i\mu\theta_0^{\rm in}}`$ times

```math
\kappa^{\pm}_{n,\mu} = -2\,e^{\mp i n\pi/2}\,e^{\mp i\pi\langle F^2\rangle}\,n^{\langle F^2\rangle - i\mu\omega}\,e^{\mp\pi\mu\omega}\,\Big(1 + O\big(\tfrac1{n\lvert u\rvert^2}\big)\Big),
```

which is explicit and non-zero.
- **The M-harmonic model.** There the transport is the Galerkin-truncated system, and the scalar factor
  $`n^{\langle F^2\rangle - i\mu\omega}`$ becomes the matrix $`n^{\mathcal G_M}`$, with $`\mathcal G_M`$ the truncated generator. This matrix is invertible, and it
  is diagonal in the gauge up to the truncation error of Section 12 (exponentially small in M at the central
  harmonics). Invertibility is all Theorem B(M) needs.
- **Linear check.** At F = 0 the factor is the power $`n^{-s}`$ times the σ-normalization $`n^{1/2}`$, with the half-monodromy
  phase. This is the large-n form $`(n+1)^{s+1/2}`$ of the linear C(n, s).

$`\square`$

### 10.2 Domains and norms

Fix ¼ < γ < ½, a small angle κ ∈ (0, π/4) and κ₁ > ½.

- **Inner lateral sums.**
  - In the Borel plane conjugate to 1/u, the inner formal solution has only one singular direction, ℝ⁺. Its non-zero
    linear eigenvalue is ½ for every harmonic (B2 in inner variables), and nonnegative combinations of ½ stay on ℝ⁺.
  - Braaksma's theorem (B4) then gives two lateral sums. $`Z^+_{\rm in}`$ is analytic and asymptotic to the formal series
    on $`-\pi/2 \lt \arg u \lt 5\pi/2`$ (on the Riemann surface of log u), and $`Z^-_{\rm in}`$ on $`-5\pi/2 \lt \arg u \lt \pi/2`$.
  - In particular $`Z^+_{\rm in}`$ is asymptotic to the formal series in the directions $`\pi/2 \lt \arg u \le \pi/2 + \kappa`$, where
    $`e^{-1/(2u)}`$ is exponentially large. That property characterizes it.
**Matching domains.** The domains are

```math
\mathcal M^+ = \big\{u : n^{-1+2\gamma} \le \lvert u\rvert \le \kappa_1/\ln n,\ 0 \le \arg u \le \pi/2 + \kappa\big\}, \qquad \mathcal M^- = \overline{\mathcal M^+}.
```

In the outer variable this is $`n^{\gamma} \le \lvert x\rvert \le (\kappa_1 n/\ln n)^{1/2}`$, $`0 \le \arg x \le \pi/4 + \kappa/2`$, a subset of the
outer domain $`\Sigma^+_n`$ of Section 11. The small-|u| edge lies towards the outer region and the large-|u| edge towards
the inner one. On $`\mathcal M^\pm`$, the corrections of B5.0 and B5.1 have relative size at most $`n^{1-4\gamma} \to 0`$.
**Norm.** Let m(u) be the size of $`Z_{\rm in}`$ (polynomial in u near 0). Differences are measured by
$`\lVert W\rVert = \sup_{\mathcal M^\pm} n\lvert u\rvert^2\,\lvert W(u)\rvert/m(u)`$.

### 10.3 Statement and proof

**Lemma B5(M).** Let $`Y^\pm_{\rm out}`$ be the continuation of the centre-regular solution through the upper (lower) half plane,
as constructed in Section 11, written in inner variables on $`\mathcal M^\pm`$.

*(1) Sectorial identification* ([B5_inner.md](B5_inner.md) §4). Assume (H8) (B5_inner.md §7: $`Y^+_{\rm out}`$ analytic
on the working domain $`\mathcal W^\pm_n \supset \mathcal M^\pm`$, and within O(1/n) of $`Z^+_{\rm in}`$ on the inner arc |x| = n^γ).
In the variables (P, Φ = 2uP_u, C, ϑ = Θ − [PP]/2) of B5_inner.md §1, $`W = Y^\pm_{\rm out} - Z^\pm_{\rm in}`$ satisfies

```math
\lvert W_j(u)\rvert \le \frac{x_j\,m_j(u)}{n}, \qquad m_\Phi = \lvert u\rvert,\quad m_j = 1 \text{ otherwise},
```

where x is any positive super-solution of the explicit linear inequality b + Lx < x of B5_inner.md §4. A super-solution
exists for n ≥ n₀(M). For M = 1 it is certified for n ≥ max(e^{73.89}, n₈) (κ = 0.3, κ₁ = 0.6, γ = 0.3), with
x = (392, 9.2×10⁴, 21, 30); n₈ is B8's threshold, and B8 delivers (H8) for n ≥ e^{73.89} (Section 11). Since |u| ≤ 1 on
$`\mathcal M^\pm`$, this implies the weaker bound $`\lvert W^\pm(u)\rvert \le C\,m(u)/(n\lvert u\rvert^2)`$ of the first version.

*(2) The jump (corrected, see the note below).* On the Stokes line arg u = 0 let
$`\Psi_{\rm out} = Y^+_{\rm out} - Y^-_{\rm out}`$ and $`\psi_{\rm in} = Z^+_{\rm in} - Z^-_{\rm in}`$, and let
$`L_u(Y) := e^{1/(2u)}u^{-\mathcal G}Y_\Phi`$ be the label map, with $`\mathcal G = [F^2\,\cdot\,] - \tfrac32 - s`$ the generator of
the fast modes $`e^{-1/(2u)}u^{\mathcal G}`$ of the Φ-component ([B_chain.md](B_chain.md) §2.1). For every u* in the
matching domain,

```math
L_{u_*}(\Psi_{\rm out}) = \big(I + R(u_*)\big)\,S(F, \omega) + E_n,\qquad \lVert R(u_*)\rVert = O(u_*),\qquad \lvert E_n\rvert = O\Big(\frac{e^{1/(2u_*)}u_*^{1-\mathrm{Re}\,\mathcal G}}{n}\Big),
```

uniformly in C¹ near (F*, ω*), where S is the inner multiplier vector, $`S = \lim_{u\to0}L_u(\psi_{\rm in})`$. For M = 1 this is
**certified** at the fixed point u* = 0.0081 for every n ≥ e^{73.89}, with ‖R‖ ≤ 0.353 (eigen-coordinates) and
|E_n| ≤ e^{76.94}/n for every normalization |c(0) − c^K(0)|₁ ≤ 10⁷n⁻⁵ (B_chain.md §§2–4, `b5_label.py`,
`b5_jump.py`). The evaluation point need not move with n: the
certificates of (1) and of B8 hold on the fixed matching radius |u| ≤ 0.00812 for every n ≥ e^{73.89} (B_chain.md §1),
so the error is O(1/n) without the refinement by $`S^{(1)}`$, which would only identify the coefficient of 1/n.

*Note (correction).* The first version stated (2) as "$`Y^+_{\rm out} - Y^-_{\rm out}`$ is a fast-type solution of the
variational equation with label function $`J_{n,\mu} = \kappa_{n,\mu}[S_\mu + \dots]`$", and its proof asserted that
"both jumps are fast-type" because each is the difference of two solutions asymptotic to the same series. For the inner
jump this is true and is now proved (lemma L of B_chain.md §2: the solutions of the inner secant equation that are flat
at u = 0 form a two-dimensional space of fast-type solutions, by lemma 6a's flat estimate). For the outer jump it is
**not proved**: B5(1) bounds $`Y^\pm_{\rm out} - Z^\pm_{\rm in}`$ separately by O(1/n), and B8 bounds the remainders
polynomially, so neither sees that $`\Psi_{\rm out}`$ is exponentially small or that it is determined by its label.
What (2) proves is a statement about one value, $`L_{u_*}(\Psi_{\rm out})`$. That $`\Psi_{\rm out}`$ vanishes identically
exactly when that value vanishes is hypothesis (H-J) of Section 13.1, open. The identification of
$`L_{u_*}(\Psi_{\rm out})`$ with the light-cone coefficient through the factor κ_{n,μ} (the transport along x > 1) is not
needed by Theorem B(1) once the selection function is taken to be $`L_{u_*}(\Psi_{\rm out})`$ (Section 13.1).
The first version's error model, $`x_\Phi n^{-1+1/(2\kappa_1)}(\ln n/\kappa_1)^{\mathrm{Re}\langle F^2\rangle}`$ (B5_inner.md §7,
`b5_threshold.py`), used the exponent ⟨F²⟩ = 3.23 instead of Re 𝒢 = ⟨F²⟩ − 3/2 = 1.73 and omitted the weight
m_Φ = |u| of the Φ-bound; its estimates ln n ≳ 285 (κ₁ = 0.6) and ≳ 104.7 (κ₁ = 0.85) are superseded by the
certificate above.

*Proof of (1).* [B5_inner.md](B5_inner.md) §§2–4: a bootstrap (not a contraction mapping, because the b-mode needs
radial paths and the fast mode progressive ones) on the working domain, with data on the whole inner arc |x| = n^γ from
(H8), fast components integrated along progressive paths with kernels of modulus at most 1, and constants defined as
suprema of explicit expressions in the inner solution. No exponentially small information is used: actual functions
are compared with polynomial accuracy, as in BFGS's matching.

*Proof of (2).* B_chain.md §3. In short: $`L_{u_*}(\Psi_{\rm out}) = L_{u_*}(\psi_{\rm in}) + L_{u_*}(W^+ - W^-)`$. The
first term is $`(I + R(u_*))S`$ by lemma L (the inner jump is a fast-type solution of the inner secant equation, and
lemma L's Volterra and Grönwall bounds give R). The second is bounded by (1) at u*, |W^±_Φ(u*)| ≤ x_Φ u*/n, times
$`\lVert e^{1/(2u_*)}u_*^{-\mathcal G}\rVert`$. Every term is holomorphic in the parameters on a complex box, so Cauchy's
estimate gives the C¹ statement. $`\square`$

### 10.4 Status of B5

- **Proved.**
  - B5.0, the exact form of the corrections.
  - B5.1, the fast variable, the label continuation and the explicit matching factor κ.
  - Part (2) given part (1), in the corrected form above (a statement about the label of the outer jump at one point);
    **certified** for M = 1 at u* = 0.0081 for every n ≥ e^{73.89} ([B_chain.md](B_chain.md) §§2–3; trusted code
    `b5_label.py`, `b5_jump.py`, not yet reviewed independently). The earlier estimate of its error (`b5_threshold.py`)
    used a wrong exponent and is superseded.
- **Part (1).** **Proved conditional on (H8)** for fixed M; **certified** for M = 1 with n₀(1) = max(e^{73.89}, n₈)
  ([B5_inner.md](B5_inner.md); trusted code `b5_bounds.py`, `b5_formulas.py`, not yet independently reviewed). The
  input from B8 is (H8) of B5_inner.md §7: analyticity on the working domain and the O(1/n) bound on the inner arc,
  whose upper end is the quantitative "no exponentially large fast content". B8 delivers it for n ≥ e^{73.89}
  (Section 11). The weak point is as predicted: the outer edge κ₁/ln n must lie within |u| ≤ 0.009, where the
  perturbative fixed point closes, so n₀ = exp(κ₁/(0.0085 cos κ)). A practical n₀ needs the exact inner variational flow
  on a fixed sector (B5_inner.md §5.4).

## 11. New (second pass): Lemma B8, the global continuation

This is the step Xie and Tanveer call the main estimate.

**Domain.** Fix δ ∈ (0, ¼), κ ∈ (0, π/4) and κ₁ > ½. Put α = π/4 + κ/2 and $`R_n = (\kappa_1 n/\ln n)^{1/2}`$. Define

```math
\Sigma^+_n = \big\{x : 0 \lt \lvert x\rvert \le R_n,\ 0 \le \arg x \le \alpha\big\} \setminus \big\{\lvert x - 1\rvert \lt \delta/2\big\}, \qquad \Sigma^-_n = \overline{\Sigma^+_n}.
```

$`\Sigma^+_n`$ contains:
- the segment (0, 1 − δ/2];
- an arc around the light cone through the upper half plane;
- the Stokes line [1 + δ/2, R_n];
- in u = x²/n, the large-exponential directions arg u ∈ (π/2, π/2 + κ] near |u| = κ₁/ln n, which is what B5 needs.

It avoids the singular points x = 0 (an endpoint), ±1, and x = c(x) ≈ n/F² beyond R_n.

**Lemma B8.1 (progressive paths; proved).** For x ∈ $`\Sigma^+_n`$ let P_x be the path from 0 along the ray arg = α to radius
|x|, followed by the arc of radius |x| clockwise to arg x. Then P_x ⊂ $`\Sigma^+_n`$, and Re Φ is non-decreasing along P_x.

*Proof.* Write r = |x| and $`Q = \lvert 1 - x^2\rvert^2 = 1 - 2r^2\cos 2\varphi + r^4`$, so that $`\mathrm{Re}\,\Phi = \ln r - \tfrac14\ln Q`$.
- On the ray φ = α, $`\partial_r\mathrm{Re}\,\Phi = (1 - r^2\cos 2\alpha)/(rQ) \gt 0`$, since cos 2α \< 0. This is where α > π/4 is used, the
  boomerang condition.
- On the arc, $`\partial_\varphi\mathrm{Re}\,\Phi = -r^2\sin 2\varphi/Q \lt 0`$ for 0 \< φ \< π/2. So Re Φ increases as φ decreases from α.
- For r near 1 the arc stops at arg x before reaching the disc |x − 1| \< δ/2, because that disc lies at smaller
  arguments.
- The b-mode phase n ln x has non-decreasing real part along P_x as well.

$`\square`$

So every fast Volterra kernel $`e^{-n(\Phi(x) - \Phi(s))}`$, with s before x on P_x, has modulus ≤ 1. The transport factor
contributes at most $`\lvert x/s\rvert^{2\mathrm{Re}\langle F^2\rangle}`$.

**Lemma B8.2 (Stokes geometry; proved).** On $`\Sigma^+_n`$ the anti-Stokes curve Re Φ = 0 runs from x = 1/√2 to ∞ e^{iπ/4}, and
the only Stokes line is x > 1, where χ = Φ ∓ iπ/2 = w is real and positive. So the continuation of the centre-regular
solution through the upper half plane has no free fast mode on all of $`\Sigma^+_n`$, in the fixed-order sense of Lemma B8
(its identification with the upper lateral Borel sum is the complex-x form of hypothesis (H-σ),
[B3_borel.md](B3_borel.md), and is not needed). The reasons:
- it has none near the centre, where $`e^{-n\Phi}`$ is exponentially large;
- no Stokes line is crossed before x > 1 is reached from above.

Its light-cone singularity is the monodromy of that sum around x = 1. The difference $`Y^+ - Y^-`$ on x > 1 is the Stokes
jump, and it equals $`C_n(e^{-i\pi\Lambda} - e^{i\pi\Lambda})\phi`$ there (§5.3 with B6). This is the chain used in Section 6, now with the
domain made explicit.

**Lemma B8 (statement; proof plan below).** For each K₀ ≥ K* and n ≥ n₀ the centre-regular solution continues
analytically to $`\Sigma^\pm_n`$, and there

```math
Y^\pm(x) = \sum_{k \le K_0}\varepsilon^k\,Y_k(x) + R^\pm(x), \qquad \lvert R^\pm(x)\rvert \le C_{K_0}\,(1 + \lvert x\rvert)\Big(\frac{(1+\lvert x\rvert)^2}{n}\Big)^{K_0+1} .
```

The Y_k are the polynomial coefficients of Proposition 1, of degree ≤ 2k + 1. The bound is that of the first omitted term.
On $`\Sigma^\pm_n`$ the ratio $`(1+\lvert x\rvert)^2/n \le 2\kappa_1/\ln n`$ is small. In particular $`Y^\pm`$ has no exponentially large fast content
anywhere on $`\Sigma^\pm_n`$, and at the small-|u| edge of $`\mathcal M^\pm`$ its error is polynomially small. Both are what B5(1)
uses.

*Proof plan.*
- **The contraction.** Use the norm $`\sup_{\Sigma^\pm_n}\lvert R\rvert / [(1+\lvert x\rvert)((1+\lvert x\rvert)^2/n)^{K_0+1}]`$. The residual of the
  truncated series is of the next order by construction.
  - The fast components (p + m per harmonic, and b − 1) are integrated along P_x, with kernels bounded by B8.1. The gain
    is $`1/(n\lvert\Phi'(x)\rvert\,\lvert x\rvert) \sim \lvert x\rvert^2/n \le \kappa_1/\ln n`$ at large |x|, and O(1/n) on bounded sets.
  - The slow components are integrated from x = 0, where regularity fixes them (B3, centre regular singular point).
- **The transport loss.** The factor $`\lvert x/s\rvert^{2\mathrm{Re}\langle F^2\rangle}`$ is dominated by the growth
  $`\lvert s\rvert^{2K_0+3}`$ of the forcing once $`2K_0 + 3 \gt 2\,\mathrm{Re}\langle F^2\rangle + 1`$. This fixes K* (K* = 3 for ⟨F²⟩ ≈ 3.7).
- **Near the excluded disc.** The structure at x = 1 is B6. On the arc |x − 1| = δ/2 the light-cone term is of size
  $`(\delta/2)^{n/2}`$, far below the bound.
- **Fixed M.** Every step is finite-dimensional, so the constants depend on M.

**Lemma B8 with constants** ([B8_constants.md](B8_constants.md)). For the M-model with b, c on |ν| ≤ M + 1, κ = 0.3,
κ₁ = 0.6, δ = 0.05 and K = 4, the centre-regular solution (gauge c(0) = c^K(0)) continues analytically to
𝒟_n = {|x| ≤ (κ₁n/(cos κ ln n))^{1/2}, 0 ≤ arg x ≤ π/4 + κ/2} ∖ {|x − 1| < δ/2} for every n ≥ e^{62}, with
|R_V| ≤ h_Vρ^5, |R_S| ≤ h_Sθρ^5, |R_β − qR_S| ≤ h_βθρ^5, |R_c| ≤ h_cρ^5 (ρ = (1 + |x|)²/n, θ = |x|/(1 + |x|), in the
variables of B8_constants.md §1.2; **certified** for M = 1 over B5(1)'s parameter box; h = (2.4×10⁶, 6.3×10⁶, 6.2×10⁵,
1.7×10⁵) at n₀ = e^{73.89}). It gives B5(1)'s hypothesis (H8) for n ≥ e^{73.89} with δ = (8.3×10⁻⁵, 79.04, 3.232, 6.464)
≤ δ_ref. For every fixed M the same holds for n ≥ n₀(M) (**proved**, no numerical value). Trusted code: `b8_alg.py`,
`b8m_formal.py`, `b8m_atoms.py`, `b8m_bounds.py`, `b8m_h8.py`; not yet reviewed independently.

Where B8_constants.md departs from the plan above (its §11):
- the domain carries the factor 1/cos κ in R′_n = (κ₁n/(cos κ ln n))^{1/2}, which B5(1) needs; R_n above lacks it;
- the transport loss sits inside the fast-kernel bound and imposes no condition on K; K = 4 is what (H8)'s Φ-bound needs
  at γ = 0.3, and only K = 4 is certified (the K* = 3 above is superseded);
- B8 never enters the light-cone disc, and B5(1) does not need it; B6 matters only for the Stokes jump on x > 1;
- the gauge is c(0) = c^K(0), the value at 0 of the series normalized order by order by c_k(1) = 0, so formally the
  light-cone gauge up to O(n^{−5}). The extension to c(0) = c^K(0) + d with |d|₁ ≤ d₀n^{−K−1} (B8_constants.md §8.4) is
  **certified** for d₀ ≤ 10⁷ at n ≥ e^{73.89}, with (H8) (B_chain.md §4, `b8m_bounds.py --d0`). That the exact
  light-cone gauge c(1) = 1 lies in this range is hypothesis (H-G) of §13.1, open;
- the outer radius may be held fixed: B8 and (H8) hold for every n ≥ n₀ on 𝒟′_n = {|x|² ≤ Z_B n, …} with
  Z_B = 0.0085 and the constants certified at n₀ (Lemma D′, B_chain.md §1, proved).

## 12. New (second pass): Lemma B6 completed, with the light-cone exponents and the Volterra bound

*Script: `solver/proof/lightcone_exponents.py`.*

**Correction to Section 5.1.** The coupling between harmonics in the light-cone exponents is O(1), not O(1/n).

Near x = 1, with t = 1 − x, the transport of Proposition 3 becomes $`t A_t = \mathcal T A`$ with

```math
\mathcal T = -\omega\,\partial_\theta - 1 + F(\theta)^2 .
```

Here F² couples the harmonics at order 1. It is the order-1, θ-dependent part of n/(2(1 − G(1)(θ))), since
G(1) = 2F²/n + O(n⁻²). The coupling is nevertheless removed exactly by the gauge e^{Π}: the equation
$`(-\omega\partial_\theta + F^2)\psi = \lambda\psi`$ has the periodic solutions $`\psi = e^{\Pi(\theta)}e^{i\mu\theta}`$ with λ = ⟨F²⟩ − iμω. So, untruncated,

```math
\Lambda_\mu = \tfrac n2 - 1 - i\mu\omega + \langle F^2\rangle + O(1/n)
```

exactly. The exponents differ pairwise by the non-zero imaginary amounts i(ν − μ)ω, so no two differ by an integer for
any n. This is the non-resonance B6 needs.

*In the M-harmonic model* 𝓣 is replaced by its Galerkin truncation $`P_M\mathcal T P_M`$, which breaks the gauge step. The script
uses the M = 5 zero (A = 2.7186, F₃/F₁ = 0.0549 + 0.0054i, F₅/F₁ = 0.0031 + 0.0005i) and tests M = 1, 3, 5, 7, 9, 15, 25.
- **Central harmonics.** The eigenvalues converge to ⟨F²⟩ − 1 − iμω exponentially in M: errors 1×10⁻², 2×10⁻⁴, 3×10⁻⁶,
  4×10⁻⁸ at M = 3, 5, 7, 9.
- **Outermost harmonics.** The two eigenvalues for |μ| = M are shifted by about 0.37, almost entirely in the imaginary
  part.
- **Non-resonance.** For every M tested, all pairwise differences have |Im| ≥ 6.89 and |Re| < 0.02. So the truncated
  models are non-resonant with a wide margin too.

**Lemma B6′ (the Volterra bound).** Work on t ∈ (0, t₀], t₀ = ½. Split the linearized M-harmonic system into fast
components a (the p + m direction of B2, one per odd harmonic) and slow components v (p − m, b − 1 and c, after
eliminating b − 1 along its own fast direction). The system reads

```math
t\,a' = \big(n\,\ell(t)\,I + \mathcal T_0 + t\,\mathcal T_1(t)\big)\,a + t\,Q(t)\,v, \qquad v' = R(t)\,a + S(t)\,v .
```

- $`\ell(t) = t\,\Phi'(1-t) = 1/((1-t)(2-t))`$, so ℓ(0) = ½ and ℓ is analytic and positive on [0, t₀].
- $`\mathcal T_0 + t\mathcal T_1`$ is the order-1 transport, with $`\mathcal T_0 = P_M\mathcal T P_M`$.
- Q, R and S are analytic and bounded uniformly in n.

Let U(t) solve the transport $`t\,U' = (\mathcal T_0 + t\mathcal T_1)\,U`$ exactly (Proposition 3 along characteristics), and put $`\psi(t) = \int^t \ell(s)\,ds/s`$.
Then, for every c, there is a solution

```math
a = e^{n\psi(t)}\,U(t)\,\big(c + r(t)\big), \qquad v = e^{n\psi(t)}\,O(t/n), \qquad \sup_{0\lt t\le t_0}\lvert r(t)\rvert \le C\lvert c\rvert/n ,
```

with C depending only on M, F, ω and t₀.

*Proof.*
1. **The phase is monotone.** ψ′ = ℓ/t > 0 and $`e^{n\psi} \sim t^{n/2} \to 0`$ as t → 0.
2. **Slow components.** Take $`v = \int_0^t(\dots)`$, the particular solution that is fast-type. Then
   $`v' = Sv + R\,e^{n\psi}U(c+r)`$. One integration by parts in the factor $`e^{n\psi}`$ gives
   $`v = e^{n\psi}\big[t\,R\,U(c+r)/(n\ell) + O(t/n^2)\big]`$, using ℓ ≠ 0. The boundary term at t = 0 vanishes.
3. **The remainder equation.** Substituting the ansatz for a, the transport cancels exactly, leaving
   $`r' = e^{-n\psi}U^{-1}Qv = t\,U^{-1}QRU\,(c+r)/(n\ell) + O(t/n^2)`$.
   - Write U(t) = t^{𝓣₀}(I + O(t)).
   - The entries of $`U(t)^{-1}(QR)\,U(t)`$ are $`O(t^{-\rho})`$, with ρ the largest difference of the real parts of the eigenvalues of
     𝓣₀.
   - ρ = 0 untruncated and ρ < 0.02 in the truncations.
   - So $`\lvert r'\rvert \le C_2\,t^{1-\rho}(\lvert c\rvert + \lvert r\rvert)/n`$, which is integrable at 0.
4. **Gronwall.** With r(t₀) = 0, $`\lvert r(t)\rvert \le \lvert c\rvert\,\big(e^{C_2 t_0^{2-\rho}/n} - 1\big) \le C\lvert c\rvert/n`$, uniformly up to t = 0. $`\square`$

**Consequence: B6 holds for each fixed M.**
- As t → 0, $`a \sim t^{n\ell(0)}\,t^{\mathcal T_0}(c + r(0))`$. This is the Frobenius branch with Λ_μ = n/2 + (eigenvalues of 𝓣₀).
- So the exact solution that agrees with the WKB mode at t₀ equals $`k_{n,\mu}\phi_\mu`$ plus a solution analytic at x = 1, with
  $`k_{n,\mu} = k^{(0)}_\mu(1 + O(1/n))`$.
- Non-resonance makes the singular branch unique.
- What remains is the routine block-diagonalization of the order-n part on [t₀, 1 − δ] by an analytic transformation
  bounded uniformly in n. This holds because the fast eigenvalue there is simple and non-vanishing (B2).

*Note (P1).* B6 and B6′ concern the *linearized* system along a background assumed analytic up to x = 1, and they give
the light-cone coefficient. Theorem B(1) in the form of B_chain.md §8.4 does not use that coefficient. What it needs at
the light cone is control of the nonlinear solution itself, (H-LC) of §13.1, which B6′ does not give.

## 13. Status

This section is the canonical statement of layer B's status; PROOF.md and the README point here.

| Step | Status | Details |
|---|---|---|
| H-a1, H-a2 (for B3) | **false as stated, harmlessly; replaced by (H-σ)**: proved in the linear case and near ζ = 0 for every M (Gevrey-1 on every disc; 2Φ excluded near 1/√2, certified for M = 1 in the truncation with b, c θ-independent); conjecture beyond, with ball-arithmetic numerical evidence (M = 1, 3); not used by Theorem B(M). The first version's numerical check is withdrawn | Sections 4.2, 8; [B3_borel.md](B3_borel.md) |
| B3: formal series Gevrey-1; actual solution | formal half **proved** for every M (B3.7); existence with fixed-order asymptotics = B8 on the real segment; the 1-sum ⇔ (H-σ), open and not needed | Section 4.2; B3_borel.md §7.1 |
| B4: inner multipliers | **cited** (Braaksma 1992, hypotheses checked against the paper); parameter-analyticity **proved** given Braaksma (B4.P) | Section 4.1; B3_borel.md §7.2 |
| B5.0: exact equations = inner problem + n⁻¹𝓠₁ + n⁻²𝓠₂ | **proved**; **checked symbolically** | Section 9 |
| B5.1: fast variable, label continuation, matching factor κ | **proved** | Section 10.1 |
| B5(1): sectorial identification on $`\mathcal M^\pm`$ | **proved conditional on (H8)** for fixed M; **certified** for M = 1, n ≥ max(e^{73.89}, n₈) | Section 10.4; [B5_inner.md](B5_inner.md) |
| B5(2): the label of the outer jump at a fixed u* is (I + R)S + E_n, E_n = O(1/n) | **proved given B5(1)** (corrected form, Section 10.3); **certified** for M = 1 at u* = 0.0081: ‖R‖ ≤ 0.353, \|E_n\| ≤ e^{76.94}/n for n ≥ e^{73.89}, with lemma L (labels of the inner fast-type solutions) | Section 10.3; [B_chain.md](B_chain.md) §§2–3 |
| B6: light-cone matching | **proved for fixed M** up to a routine block-diagonalization on [t₀, 1 − δ]: exact non-resonance (gauge), checked for the truncations; Volterra bound B6′; Section 5.1's "O(1/n) coupling" corrected to O(1). Not used by Theorem B(1) in the form of B_chain.md §8.4 | Section 12 |
| B8: global continuation | B8.1, B8.2 **proved**; B8 **certified** for M = 1 (n ≥ e^{62}), with (H8) of B5(1) **certified** for n ≥ e^{73.89}, also for normalizations \|c(0) − c^K(0)\|₁ ≤ 10⁷n⁻⁵ and on a fixed outer radius (Lemma D′); **proved** for every fixed M without a numerical n₀(M) | Section 11; [B8_constants.md](B8_constants.md); B_chain.md §§1, 4 |
| Lemma P and B7 (M = 1) | **certified** given the links above: for n ≥ e^{89.75}, exactly one parameter in Theorem A's box makes the label of the outer jump vanish, within e^{67.84}/n of Theorem A's zero | B_chain.md §8 |
| Light cone: (H-LC), (H-G), (H-J) | **open** | Section 13.1; B_chain.md §§6–7 |

### 13.1 Theorem B(1): the status of the chain

**Theorem B(1), the selection theorem for the one-harmonic model, is conditional** on three open hypotheses at the
light cone, (H-LC), (H-G) and (H-J) below, and on Braaksma's theorem. Given them it holds with an explicit threshold
([B_chain.md](B_chain.md) §8.4): for every n ≥ n₀ = e^{89.75} (D = n + 2) exactly one parameter p_n = (A_n, ω_n/ω∞) in
Theorem A's box X = x̂ ± 10⁻⁹ gives a centre-regular solution analytic at the light cone, and
|p_n − p*|_∞ ≤ e^{67.84}/n. Every other link is proved or certified. None of the certificate code of B5, B8, lemma L,
lemma P or B7 has been reviewed independently.

The model throughout is that of Section 1 with M = 1: p, m on the harmonics ±1, b and c on {0, ±2}. Theorem A(M = 1),
B5(1) and B8 all use it. The parameters are (A, ω/ω∞) in the complex box 𝒦 of B5_inner.md §5.1, which contains a
polydisc of radius 10⁻⁶ around every point of Theorem A's box.

| Link | Status | Threshold in n |
|---|---|---|
| Theorem A(M = 1): the zero p* = (A*, ω*) of S₁, simple | **certified** (A_certify.md §12.6), assuming Braaksma's theorem; the Krawczyk certificate was reviewed independently (A_review.md) and re-implemented with CAPD; the disc covering of lemma 6b₁ has not been re-implemented independently | none (n-free) |
| B1(1): formal 1/n series | **proved** (Section 3.1; for this model Lemma F of B8_constants.md, checked) | none |
| B2: slow–fast structure | **proved**, checked symbolically | none |
| B3, formal half: the outer series is Gevrey-1 | **proved** (B3.7); not used by the links below, which need only fixed order | none |
| B4, B4.P: multipliers defined, analytic in (A, ω) | **cited** (Braaksma) and **proved** given Braaksma; for M = 1 the contraction of B4.P is certified on a box (A_certify.md §4.5) | none |
| B5.0, B5.1 | **proved** (B5.0 checked symbolically for this model) | none |
| B8(1): continuation to 𝒟′_n with remainder bounds | **certified** (κ₁ = 0.6, K = 4) for every normalization \|c(0) − c^K(0)\|₁ ≤ 10⁷n⁻⁵; the fixed outer radius \|x\|² ≤ 0.0085 n is Lemma D′ (**proved**, B_chain.md §1) | n ≥ e^{62}; with the normalization offset, n ≥ e^{73.89} |
| (H8): B8's output in B5(1)'s form | **certified** (κ₁ = 0.6, γ = 0.3), same normalizations | n ≥ e^{73.89} |
| B5(1): W = O(1/n) on the matching domain | **certified** given (H8) (κ = 0.3, Z = 0.0085) | n ≥ e^{73.89} |
| Lemma L: the inner jump is a fast-type solution; labels of fast-type solutions on the Stokes line | **certified** (ε_L ≤ 0.353 at u* = 0.0081; B_chain.md §2) | none |
| B5(2): $`V^{-1}L_{u_*}(\Psi_{\rm out}) = (I + \tilde R)V^{-1}S + \tilde E`$ | **certified**: ‖R̃‖ ≤ 0.353, \|Ẽ\| ≤ e^{76.94}/n, holomorphic in p on 𝒦 (B_chain.md §3) | n ≥ e^{73.89} |
| Lemma P: Theorem A's J_D equals ℓ̃·(multiplier), ℓ̃ holomorphic and bounded | **certified** (backward adjoint of the validated secant flow, B_chain.md §8.2) | none |
| B7: exactly one zero of the outer selection condition in X | **certified** given the links above (Krawczyk test for uniqueness and location, a degree argument for existence; B_chain.md §8.3) | n ≥ e^{89.75} |
| (E): analytic at x = 1 ⇔ Ψ_out ≡ 0 on the Stokes segment | **proved** given (H-LC) (B_chain.md §6) | — |
| (H-LC), (H-G), (H-J) | **open** (below) | — |
| B6, B6′: light-cone coefficient | B6′ **proved**; B6 **proved** up to the block-diagonalization on [t₀, 1 − δ]; not needed by Theorem B(1) in the form above | — |

**The open hypotheses.** All three concern the light cone x = 1, which every certified link avoids: B8's domain
excludes the disc |x − 1| < δ/2 (δ = 0.05), and the outer jump Ψ_out = Y⁺ − Y⁻ is the difference of the
continuations of the centre-regular solution around that disc through the upper and lower half planes, on the Stokes
segment 1 + δ/2 ≤ x ≤ (0.0081 n)^{1/2}. (H-G) and (H-J) rest on (H-LC). They are open problems, not missing write-ups:
no note contains an argument for them, and the existing lemmas do not give them.

- **(H-LC), the solution near the light cone.** *Statement.* For n ≥ n₀ and real p ∈ X, the centre-regular solution in
  the gauge of (H-G):
  1. continues analytically from (0, 1 − δ/2] to (0, 1) and to the open half-discs
     $`H^\pm = \{0 \lt \lvert x - 1\rvert \lt \delta/2,\ \pm\mathrm{Im}\,x \gt 0\}`$, and is bounded there;
  2. has continuations Y^± that are C¹ up to x = 1 from within $`\overline{H^\pm}`$, with equal values and derivatives
     at x = 1 (so that |Ψ_out(x)| ≤ C(x − 1) on (1, 1 + δ/2)).

  Part 1 gives (E). Part 2 is what (H-J)'s route uses. *Why B8 and B6′ do not give it.* B8's remainder system contains
  D⁻¹, which is O(1/(x − 1)) at the light cone, and its bound algebra estimates D⁻¹AS and D⁻¹x²BV separately, although
  their combination xS′ is bounded for a regular solution; its lower bound on |D| vanishes at x = 1. B6′ is a Volterra
  bound for the linearized system along a background assumed analytic up to x = 1: it does not control the nonlinear
  solution, which is the point at issue. *Closest tools.* For the linear part, Olver-type error bounds for
  Liouville–Green approximations up to a regular singular point (Olver, *Asymptotics and Special Functions*, Ch. 10–12;
  B6′ is of this type). For the nonlinear statement, the treatment near the finger tip in Xie and Tanveer's selection
  proofs (C_selection.md §2), whose role is the same; and a bootstrap of B8's kind on H^± in which the fast kernel near
  x = 1 is |(x − 1)/(s − 1)|^{(n−1)/2} along B8.1's progressive paths, the remainder system is rewritten so that
  D⁻¹(AS − x²BV) enters as xS′, and the gauge is imposed by integrating R_c from x = 1. Reiterer and Trubowitz's D = 4
  existence proof (Commun. Math. Phys. 368 (2019) 143, arXiv:1203.3766) treats the light cone as an interior line of the domain of analyticity; their method is the model to
  study first.
- **(H-G), the exact light-cone gauge.** *Statement.* For n ≥ n₀ and every p ∈ 𝒦 there is d(p) ∈ ℂ³ (harmonics 0, ±2),
  holomorphic on 𝒦 with |d(p)|₁ ≤ 10⁷n⁻⁵, such that the centre-regular solution with c(0) = c^K(0) + d(p) has
  c(x) → 1 as x → 1⁻ (c₀ → 1, c_{±2} → 0). *Why it is needed.* Theorem B(1) is about the light-cone gauge c(1) = 1,
  while B8, (H8), B5(1) and B5(2) are certified for normalizations c(0) = c^K(0) + d with |d|₁ ≤ 10⁷n⁻⁵ (holomorphy in
  p feeds B5(2)'s Cauchy estimates). *Why B8 does not give it.* c(1) is a value inside the excluded disc. Formally
  (numerical evidence, `b8m_formal.py`) the offset is $`\sum_{k\ge5}\varepsilon^kc_k(0)`$ with |c₅(0)|₁ = 1.23×10⁵, about
  a factor 80 inside the certified range. *Closest tools.* An implicit-function argument for d ↦ c(1⁻; d) − 1 once
  (H-LC) gives c(1⁻) with bounds and its dependence on d.
- **(H-J), the outer jump is determined by its label.** *Statement.* For n ≥ n₀ and real p ∈ X: if
  $`L_{u_*}(\Psi_{\rm out}) = 0`$ then Ψ_out ≡ 0 on the Stokes segment. *Why it is needed.* B7 locates the zeros of the
  label; (E) is about Ψ_out itself. Section 10.3's earlier "both jumps are fast-type" asserted it without proof (now
  corrected there); for the inner jump the corresponding statement is lemma L(2). *Why the certified links do not give
  it.* B5(1) bounds W^± separately by O(1/n) and B8's remainder bounds are polynomial, so nothing certified sees the
  exponential smallness of Ψ_out on the Stokes segment. *Closest tools.* An outer analogue of lemma 6a's flat estimate
  on the Stokes segment, with the fast components integrated backward from x* = (n u*)^{1/2} (kernel ≤ 1) and the slow
  ones forward from x = 1⁺ with zero data, which needs (H-LC)(2) at x = 1 (B_chain.md §7).

**Also not done.**
1. *Independent review* of the certificate code: `b5_bounds.py`, `b5_formulas.py` (B5(1)); `b8_alg.py`,
   `b8m_formal.py`, `b8m_atoms.py`, `b8m_bounds.py` (including `--d0`), `b8m_h8.py` (B8 and (H8)); `b5_label.py`,
   `b5_jump.py`, `b7_adjoint.py`, `b7_ift.py` (lemma L, B5(2), lemma P, B7; list and hand proofs to check in
   B_chain.md §10); `b3_gevrey_certificate.py` (B3.7, not on the critical path). Theorem A's disc covering (lemma 6b₁)
   has not been re-implemented independently either.
2. *The coefficient of 1/n* (S^{(1)}) is not computed; it is not needed for the rate (B_chain.md §5), but it would
   lower n₀.
3. *Assumed:* Braaksma's summability theorem, as in Theorem A(M = 1).

Superseded by B_chain.md: the earlier items "B5(2)'s error constant is only estimated" (`b5_threshold.py`'s exponent
was wrong; B5(2) is now certified), "B8's normalization" (run), "the 1/n rate" (follows at a fixed u*), "B6's
block-diagonalization" and "the transport of the jump along the Stokes line" (not needed for Theorem B(1); replaced by
(H-LC) and (H-J)).

## References

- O. Costin, *On Borel summation and Stokes phenomena for rank-1 nonlinear systems of ordinary differential equations*,
  Duke Math. J. 93 (1998) 289–344, arXiv:math/0608408 (notes/REF_CHECK.md §3).
- O. Costin, *Exponential asymptotics, transseries, and generalized Borel summation for analytic, nonlinear, rank-one
  systems of ODEs*, arXiv:math/0608414 (Int. Math. Res. Notices 1995 *(unverified)*). Normal form
  $`y' = f_0 - \hat\Lambda y - x^{-1}\hat B y + g`$, $`\hat\Lambda`$ invertible, eigenvalues of distinct arguments.
- O. Costin and R. D. Costin, *On the formation of singularities of solutions of nonlinear differential systems in
  antistokes directions*, Invent. Math. 145 (2001) *(volume unverified)*, arXiv:math/0202234.
- B. L. J. Braaksma, *Multisummability of formal power series solutions of nonlinear meromorphic differential
  equations*, Ann. Inst. Fourier 42 (3) (1992) 517–540, doi:10.5802/aif.1301.
- J.-P. Ramis, Y. Sibuya, *A new proof of multisummability of formal solutions of non linear meromorphic differential
  equations*, Ann. Inst. Fourier 44 (3) (1994) 811–848.
- M. Canalis-Durand, *Solution formelle Gevrey d'une équation singulièrement perturbée : le cas multidimensionnel*,
  Ann. Inst. Fourier 43 (2) (1993) 469–483, doi:10.5802/aif.1341.
- A. D. Sokal, *An improvement of Watson's theorem on Borel summability*, J. Math. Phys. 21 (2) (1980) 261–263
  *(volume, issue and pages from a citing record)*.
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

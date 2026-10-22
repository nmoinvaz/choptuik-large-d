# B3 and B4: the outer Borel plane, and the two citation checks

This note does two things for layer B of [PROOF.md](../PROOF.md) and its working note [C_selection.md](C_selection.md):

1. It replaces hypotheses H-a1 and H-a2 of C_selection.md §4.2 and §8 by a correctly stated hypothesis, proves what can
   be proved about the outer Borel plane, and runs a computer-assisted pilot (sections 1–6).
2. It records the literature behind the two citation checks of C_selection.md §7: the mixed slow–fast version of B3,
   and the parameter-analyticity of the inner lateral sums (Braaksma 1992 and successors) (section 7).

It was written by Claude (Anthropic) under Nathan Moinvaziri's direction and has not been checked by a mathematician.
Status words are those of [AGENTS.md](../AGENTS.md): *proved* (by hand), *certified* (computer-assisted, trusted code
named), *conditional*, *conjecture*, *numerical evidence*. Every number quoted comes from a log in `solver/proof/`
written by the `b3_*.py` script named next to it. All runs and their mutation controls are repeated by
`python3 b3_checks_all.py` (from `solver/proof/`; summary in `b3_checks_all.log`).

Contents

- [Summary](#summary)
- [1. Where H-a1 and H-a2 enter, and what replaces them](#1-where-h-a1-and-h-a2-enter-and-what-replaces-them)
- [2. The outer Borel plane: structure](#2-the-outer-borel-plane-structure)
- [3. The hypothesis (H-σ) and what it gives](#3-the-hypothesis-h-σ-and-what-it-gives)
- [4. Toward a proof of (H-σ) for M = 1](#4-toward-a-proof-of-h-σ-for-m--1)
- [5. Numerical evidence, recomputed in ball arithmetic](#5-numerical-evidence-recomputed-in-ball-arithmetic)
- [6. The computer-assisted step: design and pilot](#6-the-computer-assisted-step-design-and-pilot)
- [7. The citation checks of C_selection.md §7](#7-the-citation-checks-of-c_selectionmd-7)
- [8. Status](#8-status)
- [9. Proposed text for the other files](#9-proposed-text-for-the-other-files)
- [References](#references)

## Summary

- **H-a1/H-a2 are not used by the current proof of Theorem B(M).** Lemmas B5 and B8 (C_selection.md §10–11) work with
  actual solutions and fixed-order truncations, never with an outer Borel sum. H-a1/H-a2 matter for a stronger
  statement: that the half-problem solution *is* the 1-sum of the outer series in 1/n. That statement also underlies the
  nonlinear analogue of Proposition 4 (outer late orders carry the multiplier) (§1).
- **As stated, H-a1/H-a2 are false, harmlessly.** "Every Borel singularity at |arg| ≥ 0.48π" fails for x > 1/√2: the
  u-mode singulant Φ(x) − iπ/2 itself has argument −0.44π at x = 0.8 and −0.30π at x = 0.95. It lies on the line
  Im ζ = −π/2, so it never obstructs the sum along ℝ⁺. The correct hypothesis is **(H-σ)**: the Borel transform is
  analytic, with exponential bounds, in a neighbourhood $`S_\sigma`$ of [0, ∞) of some fixed width σ > 0. By the
  Nevanlinna–Sokal theorem that is exactly the condition for the half-problem solution to be the Borel sum (§3).
- **Proved** (§2):
  - B3.1: Φ maps the right half-plane slit along [1, ∞) conformally onto the strip |Im w| < π/2; so does ln x on the
    right half-plane.
  - B3.2: an exact scaling symmetry (x, c) → (λx, λc). It reduces the light-cone gauge c(1) = 1 to the gauge c(0) = 1
    plus one scalar series c(0)(1/n) at M = 1.
  - B3.3: in the gauge c(0) = 1 every Taylor coefficient at x = 0 is a rational function of n, so its Borel transform is
    entire.
  - B3.4: in the linear case the Borel transform is analytic in the open strip |Im ζ| < π/2, uniformly for 0 < x ≤ 1,
    with all singularities on the lines Im ζ = ±π/2, ±3π/2, … and Im ζ = ±π, ±2π, ….
  - B3.5: convolution keeps analyticity in star-shaped domains. So the sum 2Φ(x) of two u-mode singulants from opposite
    sides of ℝ⁺ cannot appear on the principal sheet of a convolution fixed point.
  - B3.6: the linear fast transport in the Borel plane is singular exactly at the u-mode singulants Φ(x) − iπ(ℓ + ½),
    the branch points of Φ⁻¹.
  - **B3.7: the outer formal series of the M-harmonic model is Gevrey-1 uniformly on every disc |x| ≤ ρ, for every M,
    F and ω, in both gauges.** The proof is a majorant argument closed by the implicit function theorem for a positive
    polynomial system. It covers the formal half of B3, the case that no theorem read in §7 covers (slow and fast
    components, with a regular singular point at x = 0).
- **Certified** (§6, the pilot): for M = 1, F = 2.72 sin θ, ω = ω∞, the Borel transform is analytic on |ζ| < 0.0154
  uniformly for |x| ≤ 0.75 (light-cone gauge: |ζ| < 0.0061 for |x| ≤ 1.05). In particular the would-be 2Φ(x) singularity
  is absent for x ∈ [0.7044, 0.7098], the one place where it would approach the origin. Trusted code:
  `b3_gevrey_certificate.py` and python-flint.
- **Numerical evidence** (§5): the high orders were recomputed in ball arithmetic (M = 1 to order 120, M = 3 to 100,
  M = 5 to 70). This replaces C_selection.md §8, whose double-precision Padé readings are wrong in places.
  - At M = 1 the only singularities near ℝ⁺ are the u-mode ones on Im ζ = ∓π/2. The x-independent ones (c(0), the
    light-cone position) lie near ±3.1i, plausibly ±iπ as in the linear case.
  - At M = 3 there is in addition an x-independent singularity near ζ ≈ −1.2, close to the negative real axis. So the
    old "x-independent cluster" is real for M ≥ 3, but it lies opposite to ℝ⁺. It limits σ without obstructing the
    1-sum.
  - Strip checks with planted ℝ⁺ singularities of weight 10⁻³ fire, so the checks can fail.
- **Literature** (§7):
  - Braaksma 1992 (read) covers B4's summability; its hypotheses hold. It says nothing about parameters, nor does the
    abstract of Ramis–Sibuya 1994. Parameter-analyticity needs a short extra argument, given here as Lemma B4.P: on a
    ray where e^{−1/(2u)} is large, the lateral sum is the fixed point of a contraction that is uniform in (F, ω). It is
    the argument A_certify.md §4.5 carries out on a box for M = 1.
  - For the mixed slow–fast B3, every theorem read (Canalis-Durand 1993; Canalis-Durand–Mozo-Fernández–Schäfke 2007, by
    its abstract) assumes an invertible fast linear part on all components. Ramis–Schäfke 1996 is about averaging, not
    slow manifolds. None covers our case. The formal half is now Lemma B3.7. The fixed-order existence half is
    contained in B8, and the Borel-sum half is (H-σ).

## 1. Where H-a1 and H-a2 enter, and what replaces them

**The hypotheses as stated** (C_selection.md §4.2, §8, §13):
- H-a1: the outer Borel singularities lie at |arg ζ| ≥ 0.48π, with ζ the Borel variable dual to ε = 1/n.
- H-a2: the would-be singularity at 2Φ(x), the sum of the two conjugate u-mode singulants, is absent.

They were introduced for "the stronger statement used in Section 6, that the centre-regular solution is the 1-sum in ε
along arg ε = 0 on (0, 1)".

**What the current proof uses.** The chain of Theorem B(M) (C_selection.md §6) now runs through:
- B8: the continuation Y± of the centre-regular solution to $`\Sigma^\pm_n`$, with a fixed-order remainder bound and "no
  exponentially large fast content";
- B5(1): matching to the inner lateral sums on $`\mathcal M^\pm`$, with data from fixed-order truncations;
- B5(2): the jump Y⁺ − Y⁻ on the Stokes line, read at one evaluation point u*;
- B6: the light-cone lemma.

None of these takes a Borel sum of the outer series. Y⁺ and Y⁻ are continuations of one actual function, and their
difference on x > 1 is the monodromy of its light-cone branch (B6), not a difference of lateral Borel sums. So H-a1/H-a2
are **not on the critical path of Theorem B(M)**.

One sentence of C_selection.md does use the 1-sum: B8.2, "the continuation … is the upper lateral sum of the slow
series on all of $`\Sigma^+_n`$". As used by B5 it means only the fixed-order statement of B8. As written it is the
1-summability of the outer series for complex x, which is (H-σ) below extended to complex x. A rewording is proposed in
§9.

**Where the stronger statement matters.**
1. The interpretation of the half-problem solution as the Borel sum of the outer series. This gives uniqueness (Watson /
   Nevanlinna–Sokal) and a constructive definition.
2. The nonlinear analogue of Proposition 4, that the outer late orders carry the inner multiplier. This is what lets the
   numerics read S from the outer series. It needs the singularity structure of the outer Borel plane near the Stokes
   line.
3. The numerical use of outer late orders in general (§5 shows how misleading they can be without it).

**The correct hypothesis.** The statement "all singularities at |arg| ≥ 0.48π" is false for every x ∈ (1/√2, 1), in
every model including the linear one. The u-mode singulant $`\chi_0(x) = \Phi(x) - i\pi/2`$ has Re Φ(x) > 0 there:
- arg χ₀(0.8) = −0.44π;
- arg χ₀(0.95) = −0.30π;
- arg χ₀(x) → 0 as x → 1.

The high-precision data confirm a singularity exactly there (§5.1). It is harmless, because Im χ₀ = −π/2 never vanishes.
The property that matters for summation along ℝ⁺ is distance from the half-line [0, ∞), not argument:

**(H-σ).** For x in a compact set K ⊂ (0, 1] there are σ > 0, C and ν such that the Borel transform Ŷ(ζ, x) of the
outer series continues analytically to

```math
S_\sigma = \{\zeta : \mathrm{dist}(\zeta, [0,\infty)) \lt \sigma\}, \qquad \lvert \hat Y(\zeta, x)\rvert \le C\,e^{\nu\lvert\zeta\rvert}\quad (\zeta \in S_\sigma,\ x \in K).
```

H-a2 (2Φ absent) is the special case of (H-σ) on the half-line, for x > 1/√2, where 2Φ(x) > 0. At x = 1/√2 the would-be
singularity reaches the origin (Φ(1/√2) = 0).

## 2. The outer Borel plane: structure

Notation as in C_selection.md §1: n = D − 2, ε = 1/n, the M-harmonic model, and Proposition 1's formal solution
$`Y(x;\varepsilon) = \sum_k \varepsilon^k Y_k(x)`$ with polynomial $`Y_k`$ of degree ≤ 2k + 1. Its Borel transform at fixed x is
$`\hat Y(\zeta, x) = \sum_{k\ge1} Y_k(x)\,\zeta^{k-1}/(k-1)!`$.

### 2.1 Lemma B3.1 (the characteristic variables; proved)

Let Ω = {Re x > 0} \ [1, ∞) and Σ = {w : |Im w| < π/2}.

1. $`\Phi(x) = \ln x - \tfrac12\ln(1 - x^2) = \tfrac12\,\mathrm{Log}\big(x^2/(1-x^2)\big)`$ maps Ω conformally onto Σ, and (0, 1) onto ℝ. The
   boundary correspondence is:
   - the imaginary half-axes ↔ the boundary lines Im w = ±π/2 with Re w < 0;
   - the two sides of the slit x > 1 ↔ Im w = ±π/2 with Re w > 0, where Φ = w(x) ± iπ/2 as in B5.1(1);
   - x = ∞ ↔ the boundary points ±iπ/2.
2. The inverse is $`\Phi^{-1}(w) = e^{w}(1 + e^{2w})^{-1/2}`$. It continues analytically to every w except the branch points
   w ∈ iπ(ℤ + ½), the images of x = ∞.
3. ln x maps {Re x > 0} conformally onto Σ, and its inverse $`e^w`$ is entire.

*Proof.*
- z = x² maps Ω biholomorphically onto ℂ \ ((−∞, 0] ∪ [1, ∞)).
- The Möbius map T(z) = z/(1 − z) has real coefficients and T′ > 0 on ℝ. It sends (−∞, 0] onto (−1, 0] and [1, ∞]
  onto [−∞, −1], so it maps ℂ \ ((−∞, 0] ∪ [1, ∞)) onto ℂ \ (−∞, 0].
- The principal logarithm maps ℂ \ (−∞, 0] onto |Im| < π, and Φ = ½ Log T(x²) on (0, 1), hence on the simply connected
  Ω.
- The boundary values follow from T preserving the upper half-plane.
- For the inverse: $`x^2/(1-x^2) = e^{2w}`$ gives $`x^2 = e^{2w}/(1+e^{2w})`$, singular only where $`e^{2w} = -1`$.

Part 3 is the standard property of the logarithm. $`\square`$

### 2.2 Lemma B3.2 (scaling symmetry and the two gauges; proved)

If (p, m, b, c)(x) solves the self-similar equations (C_selection.md §1; any n, ω, M), then so does

```math
(p_\lambda, m_\lambda, b_\lambda, c_\lambda)(x) = \big(p(x/\lambda),\ m(x/\lambda),\ b(x/\lambda),\ \lambda\,c(x/\lambda)\big), \qquad \lambda \ne 0 .
```

*Proof.* With X = x/λ:
- $`x\,\partial_x c_\lambda = \lambda X c'(X) = \lambda (n-1) c (b-1) = (n-1)\,c_\lambda (b_\lambda - 1)`$;
- the b equation is invariant;
- $`x(x - c_\lambda)\,\partial_x p_\lambda = \lambda\,X(X - c)\,p'(X)`$, and every term on the right of the p equation
  ($`-\omega x p_\theta`$, $`Gp`$ with $`G_\lambda = \lambda G`$, $`-c p`$, $`\tfrac n2 c(p+m)`$) is multiplied by λ;
- the same holds for m.

$`P_M`$ acts on θ only, so the truncated models inherit the symmetry. $`\square`$

**Consequences.**
- *Centre gauge.* Call Y_ce the formal solution of Proposition 1 with the gauge G_k(0) = 0 instead of G_k(1) = 0, i.e.
  c(0) = 1, with profile F̃ and coefficient of x in p − m equal to 2F̃/√n.
- *Light-cone gauge at M = 1.* b and c have only the harmonic 0 when M = 1. The light-cone-gauge series with profile F is
  then Y_ce composed with x ↦ x/c₀(ε) and F̃ = c₀(ε)F (c multiplied by c₀). Here $`c_0(\varepsilon) = 1 - \sum_k G_k(0)\varepsilon^k`$ is
  the light-cone-gauge series of c(0), fixed by c₀ C(1/c₀) = 1, i.e. by the light-cone position X* = 1/c₀ in the centre
  gauge, C(X*) = X*.
  - So at M = 1, everything in the light-cone-gauge Borel plane that is not in the centre gauge comes from one scalar
    series c₀(ε).
- *M ≥ 3.* The light-cone gauge c(1, θ) = 1 for all θ is θ-dependent, and no scaling achieves it. The even harmonics of
  c(0, θ) are then global data fixed at the light cone. They carry the x-independent singularities seen at M = 3 (§5.2).

### 2.3 Lemma B3.3 (centre gauge: rational Taylor coefficients; proved)

In the centre gauge, write p = P/√n, m = M/√n. Every Taylor coefficient at x = 0 of the centre-regular solution of the
M-harmonic model is a polynomial in (F̃'s harmonics, their conjugates, ω), with coefficients rational in n. The only
possible poles are at n = 1 − j for 2 ≤ j ≤ (the order of the coefficient). Each Taylor coefficient of the formal series
Y_ce is the expansion at n = ∞ of that rational function, so it converges, and its Borel transform is entire.

*Proof.* At x = 0 (C_selection.md §3.2) the coefficient of $`x^j`$ in each equation is determined by lower coefficients
except through the indicial terms:
- c: $`j\,c_j = (n-1)[c(b-1)]_j`$;
- b: $`(j + n - 1)\,b_j = \dots`$;
- (p, m): a 2 × 2 system with determinant −(j − 1)(j − 1 + n), from $`-x p_x - p + \tfrac n2 (p+m)`$ and
  $`x m_x + m - \tfrac n2 (p+m)`$.

The terms $`\omega x\partial_\theta`$, $`x^2\partial_x`$ and G·p raise the power of x, so they enter the right-hand sides only. j = 1
is the free datum F̃; c₀ = 1 and b₀ = 1. By induction each coefficient is rational in n with the stated denominators.
The formal series solves the same recursion order by order in ε. The recursion has a unique solution in formal Laurent
series in ε, because the determinants are invertible there. So the two agree, and the rational functions are bounded at
n = ∞, since Proposition 1 has no negative powers of ε. $`\square`$

*Meaning.* In the centre gauge an x-independent Borel singularity has zero strength at every fixed order in x. It can
come only from the summation over the Taylor index. In the linear case the poles at n = −2, −4, … (spacing 2, from the
parity of j) are what produce the Γ-function factors and the singularities at iπm (Lemma B3.4).

### 2.4 Lemma B3.4 (the linear case; proved)

Let F = 0 (no gravity), take one harmonic, s = iμω with ω ≠ 0 real, and let y = x ₂F₁(α, β; κ; x²) with
α = (1+s)/2, β = (2+s)/2, κ = (n+1)/2 (Proposition 2, normalized so that the coefficient of x is 1).

For every x ∈ (0, 1], the Borel transform of the series of y/x in ε is analytic in the open strip |Im ζ| < π/2. It has
exponential bounds on every closed sub-strip |Im ζ| ≤ π/2 − δ, uniformly for x ∈ [x₀, 1]. Its Laplace transform along
ℝ⁺ is y/x for Re n large.

Its singularities lie in the set

```math
\{\Phi(x) - i\pi(\ell + \tfrac12) : \ell \in \mathbb Z\} \;\cup\; \{i\pi m : m \ne 0\},
```

and in sums of points of this set. All of them have |Im| ≥ π/2. The two nearest to ℝ⁺,
$`\tau_0, \tau_{-1} = \Phi \mp i\pi/2`$, are attained: they govern the late orders (Proposition 4).

*Proof.* Start from Proposition 4's Euler integral:

```math
y/x = P(\kappa)\cdot 2\!\int_0^\infty e^{-n\tau}\,\tau^{\beta-1}H(\tau)\,d\tau, \qquad P = \frac{\Gamma(\kappa)}{\Gamma(\beta)\Gamma(\kappa-\beta)},
```

with $`\tau^{\beta-1}H(\tau) = e^{(2\beta-1)\tau}(1-e^{-2\tau})^{\beta-1}(1 - z + ze^{-2\tau})^{-\alpha}`$ and z = x².

1. *H.* H is analytic except at the zeros τ_ℓ of $`1 - z + ze^{-2\tau}`$ and at iπm, m ≠ 0.
   - On a closed sub-strip |Im τ| ≤ π/2 − δ, $`ze^{-2\tau}`$ lies in the sector |arg| ≤ π − 2δ. Its sum with 1 − z ≥ 0 is bounded
     away from 0, relative to |ze^{−2τ}|, so |H| ≤ C e^{C|τ|} there, uniformly for z ∈ [x₀², 1].
2. *The factor $`n^\beta`$ times the integral.* Its Borel transform is $`\Psi(\zeta) = 2\sum_{m\ge1} H_m\frac{\Gamma(m+\beta)}{\Gamma(m)}\zeta^{m-1}`$. From
   $`\frac{\Gamma(m+\beta)}{\Gamma(m+2)}\zeta^{m+1} = \frac1{\Gamma(2-\beta)}\int_0^\zeta(\zeta-s)^{1-\beta}s^{m+\beta-1}ds`$ (valid since Re(2 − β) = 1 > 0) one gets the
   formula (B3.4a) below. The weights in it have modulus 1, since Re β = 1. So Ψ is analytic wherever H is analytic on
   the segment [0, ζ], in particular on the strip, with exponential bounds.
3. *The factor $`P\,n^{-\beta}`$.* By Binet's formula
   $`\ln\Gamma(z) = (z-\tfrac12)\ln z - z + \tfrac12\ln 2\pi + \int_0^\infty \varphi(t)e^{-zt}dt/t`$, with
   $`\varphi(t) = 1/(e^t-1) - 1/t + \tfrac12`$ (Re z > 0):
   - $`\ln\frac{\Gamma(\kappa)}{\Gamma(\kappa-\beta)} - \beta\ln\kappa`$ is a series in 1/κ convergent for |κ| > |β|, plus the Laplace transform of
     $`\varphi(t)(1 - e^{\beta t})/t`$;
   - that function is analytic except at t = 2πim, m ≠ 0, with exponential bounds;
   - in n (κ = (n+1)/2) the singular points become τ = t/2 = iπm, and $`(\kappa/n)^\beta = 2^{-\beta}(1+\varepsilon)^\beta`$ converges.
4. *Exponential and products.* The exponential of a series whose Borel transform f is analytic in a star-shaped domain,
   with |f| ≤ ‖f‖e^{ν|ζ|}, has Borel transform Σ_k f^{*k}/k!. Since $`\lvert f^{*k}(\zeta)\rvert \le \lVert f\rVert^k e^{\nu\lvert\zeta\rvert}\lvert\zeta\rvert^{k-1}/(k-1)!`$ (convolution along
   the segment, by induction), it is analytic there with growth $`e^{(\nu+\lVert f\rVert)\lvert\zeta\rvert}`$. Products are convolutions
   (Lemma B3.5).
5. *Conclusion.* All factors are analytic on the open strip, which is star-shaped from 0. Each factor is an actual
   Laplace transform for Re n large, so the product is the Laplace transform of the convolution. $`\square`$

The formula of step 2:

```math
\Psi(\zeta) = \frac{2}{\Gamma(2-\beta)}\,\frac{d^2}{d\zeta^2}\Big[\zeta\int_0^1 (1-t)^{1-\beta}\,t^{\beta-1}\,\big(H(\zeta t) - H_0\big)\,dt\Big]. \qquad \text{(B3.4a)}
```

*Remark.* The linear case thus satisfies (H-σ) with σ = π/2 exactly, and uniformly up to the light cone. Its
x-independent singularities iπm come from the Γ-function factors, i.e. from the pole lattice of Lemma B3.3.

### 2.5 Lemma B3.5 (convolution, star-shaped domains, and the 2Φ singularity; proved)

Let U ∋ 0 be star-shaped with respect to 0, and f, g analytic on U with |f|, |g| ≤ C e^{ν|ζ|}. Then

```math
(f * g)(\zeta) = \int_0^\zeta f(s)\,g(\zeta - s)\,ds
```

(straight segment) is analytic on U with |f * g| ≤ C²|ζ| e^{ν|ζ|}.
- Both s and ζ − s run over [0, ζ] ⊂ U.

*Consequence for 2Φ.* Suppose f is singular at χ₀ = Φ − iπ/2 and g at χ₋₁ = Φ + iπ/2. For real ζ, the singular points of
the integrand in s are s = χ₀ and s = ζ − χ₋₁ = (ζ − Φ) − iπ/2. Both lie on the line Im s = −π/2, on the same side of
the contour [0, ζ]. As ζ → 2Φ they meet without pinching the contour. So the principal branch of f * g is analytic at
ζ = 2Φ(x), whatever the sign of Φ, and the sum χ₀ + χ₋₁ lives only on sheets reached by going around χ₀ or χ₋₁.
- Hence: if the outer Borel transform is the fixed point of a convolution equation posed on a star-shaped neighbourhood
  of [0, ∞) in which the linear operators are regular (the plan of §4), then 2Φ(x) cannot be a singularity on ℝ⁺.
  $`\square`$

### 2.6 Lemma B3.6 (the fast transport in the Borel plane; proved for the scalar model)

Consider the scalar fast equation $`\varepsilon\,a(x)\,u' = -u + \varepsilon h(x,\varepsilon)`$, with $`h = \sum_{k\ge0}h_k\varepsilon^k`$ whose coefficients are entire in x
(polynomials, as in Proposition 1). Its formal solution $`u = \sum_{k\ge1}u_k\varepsilon^k`$ ($`u_1 = h_0`$, $`u_{k+1} = h_k - a u_k'`$) has a Borel
transform û that solves a transport equation and is given by characteristics:

```math
\partial_\zeta \hat u + a(x)\,\partial_x \hat u = \hat h, \qquad \hat u(0, x) = h_0(x), \qquad
\hat u(\zeta, x) = h_0\big(X(0)\big) + \int_0^\zeta \hat h\big(\zeta', X(\zeta')\big)\,d\zeta' ,
```

with $`A(X(\zeta')) = A(x) - \zeta + \zeta'`$ and A′ = 1/a. (Check: $`\partial_\zeta\hat u = \sum_k u_{k+1}\zeta^{k-1}/(k-1)! = \hat h - a\,\partial_x\hat u`$.)

- For the u-mode, a = x(1 − x²) and A = Φ. By Lemma B3.1, X(ζ′) = Φ⁻¹(Φ(x) − ζ + ζ′) is analytic in ζ except where
  Φ(x) − ζ + ζ′ ∈ iπ(ℤ + ½).
  - The integrand's singular point in ζ′ hits the endpoint ζ′ = 0 exactly when ζ = Φ(x) − iπ(ℓ + ½) = χ_ℓ(x).
  - It hits the other endpoint only at x = ∞. A single moving point cannot pinch.
  - So û(·, x) is analytic on the universal cover of ℂ \ {χ_ℓ(x)}. For real x ∈ (0, 1) these points lie on
    Im ζ = −π(ℓ + ½), and the open strip |Im ζ| < π/2 contains none.
- For the b-mode, a = x, A = ln x and X(ζ′) = x e^{ζ′−ζ} is entire. The b transport creates no singularity of its own.
  - Its singularities come from the forcing (the u-mode content of p² + m²) transported along ζ − ln x = const.
  - The tangency condition XΦ′(X) = 1, where a b-characteristic touches the u-mode locus, holds only at X = 0. The
    resulting b-mode loci ζ = ln x − iπ(ℓ + ½) (C_selection.md §8) also lie on the lines Im ζ = −π(ℓ + ½).

$`\square`$ This is the geometric core of (H-σ): **every characteristic singularity of the fast transports lies on
the lines Im ζ ∈ π(ℤ + ½).** The x-independent ones in the linear case (iπm) lie on Im ζ ∈ πℤ \ {0}.

### 2.7 Lemma B3.7 (the outer formal series is Gevrey-1; proved for every M, certified constants for M = 1)

**Statement.** For every odd M, every profile F (finitely many odd harmonics), every ω and every ρ > 0, the formal
solution of Proposition 1 for the M-harmonic model satisfies

```math
\sup_{\lvert x\rvert\le\rho}\ \lvert Y_k(x)\rvert \le C\,k!\,t_1^{-k} \qquad (k \ge 0)
```

for some C and t₁ > 0 depending on (M, F, ω, ρ). This holds in the centre gauge for every ρ, and in the light-cone
gauge for every ρ ≥ 1, hence on every disc. So the Borel transform Ŷ(ζ, x) is analytic on the bidisc |ζ| < t₁,
|x| ≤ ρ.

**Proof.**

*1. Norm.* For a trigonometric polynomial in θ with polynomial coefficients in x, $`f = \sum f_{\mu j}e^{i\mu\theta}x^j`$, put
$`\lVert f\rVert = \sum_{\mu,j}\lvert f_{\mu j}\rvert\rho^j`$. Then:
- $`\lvert f(x,\theta)\rvert \le \lVert f\rVert`$ for |x| ≤ ρ and real θ;
- $`\lVert fg\rVert \le \lVert f\rVert\lVert g\rVert`$ and $`\lVert P_M f\rVert \le \lVert f\rVert`$;
- $`\lVert xf\rVert = \rho\lVert f\rVert`$, $`\lVert x\partial_x f\rVert \le \deg(f)\lVert f\rVert`$, and $`\lVert\partial_\theta f\rVert \le M\lVert f\rVert`$.

*2. Majorants.* Take the recursion of Proposition 1, as implemented in `b3_outer_m1.py` / `b3_outer_M.py`:
- B_{k+1} from [bQ]_{k−1}, x B_k′ and products of B's;
- G_k = −∫(rhs)/x, plus the gauge constant;
- S_k from the projected equation $`P_M[(1-\varepsilon)cbS]_k = P_M[x^2D_{k-1}' - x(cS')_{k-1} + \omega x\partial_\theta D_{k-1}]`$;
- D_k from $`(x\partial_x - 1)D_k = R_k`$.

Bound every term by the rules above, using:
- deg D_k = 2k + 1, deg S_k = 2k, deg B_k = 2k − 2;
- the lowest powers x³ for D_k (k ≥ 1) and R_k, and x² for B_k, S_k and [c(b − 1)];
- (x∂ − 1)⁻¹ ≤ ½ on O(x³), with the exact identity $`(x\partial_x - 1)^{-1}x^2\partial_x S = xS`$;
- −∫ rhs/x ≤ ½ on O(x²).

In the light-cone gauge with ρ ≥ 1 the constant −g(1) of G_k obeys |g(1)| ≤ ‖g‖, so ‖G_k‖ ≤ 2‖g‖.

The result is a recursion of inequalities for majorants $`m_k \ge \lVert D_k\rVert`$, $`s_k \ge \lVert S_k\rVert`$, $`g_k \ge \lVert G_k\rVert`$,
$`\beta_k \ge \lVert B_k\rVert`$, $`\mu_k \ge \lVert[c(b-1)]_k\rVert`$, $`\kappa_k \ge \lVert[cb]_k\rVert`$, $`q_k \ge \lVert Q_k\rVert`$. Every right-hand side is a nonnegative
polynomial in lower-order majorants, with factors at most linear in k from the degree bounds. The terms are listed in
the docstring of `b3_gevrey_certificate.py` and in step 3.

*3. Normalization turns the recursion into an algebraic system.* Divide by k!, using $`\tilde\beta_k = \beta_{k+1}/k!`$ and
$`\tilde\mu_k = \mu_{k+1}/k!`$. Then:
- a derivative loss, (ak + b)·x_{k−1}/k! ≤ (a + b)·x̂_{k−1}, becomes multiplication by t;
- an ordinary convolution obeys $`\sum_i x_i y_{k-i}/k! \le \sum_i \hat x_i\hat y_{k-i}`$, because i!(k − i)! ≤ k!, and becomes a product.

So the generating functions $`B_x(t) = \sum_k \hat x_k t^k`$ are bounded coefficientwise by the minimal solution of a positive
polynomial system V = Φ(t, V). For M = 1, in the centre gauge (ω_M = ω):

```math
\begin{aligned}
m &= m_0 + \rho(1+\tfrac{\omega_M}{2})\,s + \tfrac12(1+t)\,\mu m + g\,m, &
s &= (2\rho + \omega_M\rho)\,t\,m + 2t(1+g)\,s + (g + t\beta + tg\beta)\,s + t(1+g)(1+t\beta)\,s,\\
\beta &= t(1+t\beta)\,q + 3t\beta + t\beta^2 + t^2\beta^2, &
\mu &= (1+g)\,\beta, \qquad g = \gamma(1+t)\,\mu, \qquad \kappa = (1+g)(1+t\beta), \qquad q = s^2 + m^2 .
\end{aligned}
```

Here m₀ = ‖D₀‖ and γ = ½ (centre gauge) or 1 (light-cone gauge).
- In this M = 1 instance the norm is taken on the harmonic +1 coefficient, so m₀ = Aρ and q = s² + m².
- For general M, use the full norm of step 1: m₀ = 2ρ‖F‖₁, q = (s² + m²)/2, ω_M = Mω; nothing else changes.
- The coefficient is well-founded: the order-k coefficient of each right-hand side involves only lower orders, or the
  same order in the acyclic sequence b̃ → μ̃ → g → κ → s → m → q.
- Every right-hand side has nonnegative coefficients, so Φ is monotone in V and t, and the minimal solution is the limit
  of the iteration from V = 0.

*4. Radius.* At t = 0 the system has the fixed point V₀ = (m₀, 0, 0, 0, 0, 1, m₀²), in the order
(m, s, β, μ, g, κ, q).
- Its Jacobian there is nilpotent: the dependency graph m ← s, μ, g; μ ← β; g ← μ; κ ← g; q ← m has no cycle, since the
  s and β rows vanish at t = 0.
- By the implicit function theorem the fixed point continues analytically to |t| < t₀ for some t₀ > 0, and for small
  t > 0 it is the minimal solution.
- So for 0 < t₁ < t₀ the majorants satisfy $`\hat x_k \le V_x(t_1)\,t_1^{-k}`$, i.e. $`\lVert Y_k\rVert \le V(t_1)\,k!\,t_1^{-k}`$. $`\square`$

**Certified constants** (`b3_gevrey_certificate.py`; logs `b3_gevrey_certificate_*.log`). The certificate is a vector V
of exact binary numbers with Φ(t₁, V) ≤ V componentwise, checked in arb. By monotonicity the minimal solution at t₁ is
then ≤ V. For M = 1, F = 2.72 sin θ, ω = 2π/1.64389375:

| gauge | ρ | certified t₁ | V_m (bound on Σ m̂_k t₁^k) | 2Φ(x) excluded on | actual ‖D_k‖, ‖S_k‖ / bound, k ≤ 40 |
|---|---|---|---|---|---|
| centre | 0.75 | 0.015363 | 3.196 | [0.70439, 0.70982] | ≤ 0.368 |
| centre | 1.05 | 0.008119 | 4.496 | [0.70568, 0.70854] | ≤ 0.377 |
| light cone | 1.05 | 0.006066 | 4.269 | [0.70604, 0.70818] | ≤ 0.384 |

The last column is an independent check that can fail. With the mutation `--mutate=m0` (a wrong majorant built with
‖D₀‖/10) it does fail: ratio 1.53 at k = 1 (`b3_gevrey_certificate_mutate_m0.log`).

*What this settles.*
- The Gevrey-1 property of the outer formal series, uniformly on discs that contain the centre and reach past the
  light cone. This is the formal half of B3, which C_selection.md §4.2 cited to Sibuya and to
  Canalis-Durand–Ramis–Schäfke–Sibuya, whose theorems do not cover the mixed slow–fast system verbatim (§7.1).
- H-a2 near x = 1/√2. The would-be singularity 2Φ(x) → 0 as x → 1/√2, and it would have made the Gevrey constant
  blow up there. Lemma B3.7 excludes this for every M; the certificate gives the interval for M = 1.

*What it does not settle.*
- The radius: t₁ ≈ 0.015 against the true distance ≈ 1.6 to the nearest singularity. The majorant loses the
  cancellations between x²D′ and xcS′, and the generating function also loses the 1/k smallness of interior
  convolutions. The plain majorant recursion keeps that smallness and gives ≈ 0.146 at ρ = 0.75.
- Anything beyond the disc, i.e. (H-σ) proper.

## 3. The hypothesis (H-σ) and what it gives

**Proposition B3.8 (conditional on (H-σ)).** Assume (H-σ) on a compact K ⊂ (0, 1). Then:
1. the outer series is 1-summable along ℝ⁺, uniformly for x ∈ K, and its Borel sum Y_B(x; n) is analytic for Re n > ν;
2. Y_B satisfies uniform Gevrey-1 remainder bounds in that half-plane;
3. Y_B solves the M-harmonic equations, because Borel summation in a fixed direction is a homomorphism of differential
   algebras;
4. Y_B is the centre-regular (half-problem) solution of B3/B8. Both have the same fixed-order asymptotics on K, and a
   solution with prescribed profile and fixed-order asymptotics is unique (B3's contraction, C_selection.md §4.2).

*Proof.* Items 1 and 2 are Nevanlinna's theorem in Sokal's form (Sokal 1980; statement as given by Gillam–Gurarii, see
References). For a function P of the large variable, these are equivalent:
- P is analytic in a half-plane and obeys uniform remainder bounds $`\lvert R_N\rvert \le A\sigma^{-N}N!\lvert n\rvert^{-N}`$;
- its Borel transform continues to the σ-neighbourhood of [0, ∞) with exponential bounds, and P is its Laplace
  transform.

(H-σ) is the second condition with x as a parameter. Items 3 and 4 are as stated. $`\square`$

**Status of (H-σ).**
- **Proved** in the linear case, with σ = π/2, uniformly up to x = 1 (Lemma B3.4).
- **Proved** near ζ = 0 for every M: analyticity on |ζ| < t₁ (Lemma B3.7).
- **Conjecture** in general, supported by numerical evidence (§5):
  - M = 1: σ ≈ π/2 at x = 0.5, 0.8, 0.95, set by the u-mode on Im ζ = ∓π/2. At x = 1 it is about 3.09.
  - M = 3: σ is smaller (see §5.2 for the number), set by an x-independent singularity near ζ ≈ −1.2.
- The structural reason it should hold is Lemmas B3.1, B3.5 and B3.6. The characteristic singularities lie on
  Im ζ ∈ π(ℤ + ½). Sums of singularities from opposite sides of ℝ⁺ do not reach the principal sheet. The x-independent
  singularities come from global data: Γ-type at iπm in the linear case, and the even harmonics of c(0, θ) for M ≥ 3.
- **For the untruncated model, (H-σ) uniform in M is doubtful.** If the x-independent singularity of the M-harmonic
  model approaches the origin as M grows (§5.2 compares M = 3 and 5), then σ(M) → 0, and the untruncated outer series
  would not be Borel summable in Sokal's sense. That is a gap-(b) question; the selection theorem B(M) for fixed M is
  unaffected.

## 4. Toward a proof of (H-σ) for M = 1

This is the plan for the Borel-plane fixed point. It is not written with constants.

**Setting.** M = 1, centre gauge. By Lemma B3.2 the light-cone gauge differs by the scalar c₀(ε), treated at the end. The
unknowns are the Borel transforms Ŝ, Δ̂ (harmonic 1), Ĝ and η̂ = B̂.

**1. The equations in the Borel plane.** Substitute Δ̂ = xŜ + (x∂_x − 1)⁻¹(iωxŜ + …) from the D equation into the S
equation. The terms with x³Ŝ_x combine into the principal part, and the S equation becomes a transport along the
characteristics ζ − Φ(x) = const:

```math
\partial_\zeta\hat S + x(1-x^2)\,\partial_x\hat S = \mathcal A[\hat S, \hat\Delta, \hat G, \hat\eta] + \mathcal N .
```

- In it:
  - 𝒜 is linear and of order zero in x: (x∂_x − 1)⁻¹ and ∫₀ˣ dx′/x′ are bounded on functions vanishing at x = 0 to the
    right order, and ∂_x(x∂_x − 1)⁻¹ is too.
  - 𝒩 collects convolutions with at least one factor ε, i.e. one integration in ζ.
- The b equation becomes the transport $`\partial_\zeta\hat\eta + x\partial_x\hat\eta = \dots`$ along ζ − ln x = const.
- G is slow: x∂_xĜ = … at fixed ζ, integrated from x = 0.

**2. Domain.** Take $`D_{\sigma} = \{(\zeta, x) : \zeta \in S_\sigma,\ \Phi(x) \in W\}`$ with W = {w : |Im w| < σ₁, Re w < Φ(ρ)}, σ + σ₁ < π/2. By
Lemma B3.1, Φ⁻¹(W) is a horn domain in Ω that touches x = 0 in a sector and stays away from x = 1.
- The characteristic from (ζ, x) back to ζ′ ∈ [0, ζ] lands at Φ(X) = Φ(x) − (ζ − ζ′), with |Im| < σ₁ + σ and
  Re ≤ Re Φ(x) + σ. Nested domains, or a weight in Re Φ, absorb the drift.
- The b-characteristic X = x e^{ζ′−ζ} likewise stays in a sector at 0.

**3. Norm.** Use $`\lVert f\rVert_\nu = \sup_{D_\sigma} e^{-\nu\lvert\zeta\rvert}\lvert x\rvert^{-p}\lvert f\rvert`$, with vanishing orders p from Proposition 1.
- Integration along a characteristic gains a factor 1/ν, since |X(ζ′)| ≤ C|x| as X moves toward 0.
- Convolutions are bounded, $`\lvert f * g\rvert \le \lvert\zeta\rvert e^{\nu\lvert\zeta\rvert}\lVert f\rVert\lVert g\rVert`$. Each appears with an extra integration, so they gain 1/ν too
  (or use Costin's L¹-type norm along segments, which is submultiplicative).

**4. Contraction.** For ν large the map is a contraction on a ball, and the fixed point is the Borel transform. It agrees
with Ŷ near ζ = 0 by Lemma B3.7.

**5. The open technical points.**
- (a) *Derivative loss from the ε-dependence of the characteristic speed.* The true fast eigenvalue is c²/(x(x² − c²))
  with c = 1 − εG. Its ε-corrections produce terms Ĝ * ∂_xŜ, an x-derivative inside a convolution.
  - Remedies: Nagumo-type norms on a family of shrinking horn domains (the convolution's integration supplies the time
    the derivative needs), or straightening the characteristics with the exact phase computed from the formal c.
  - This is the one step without a template in the repository.
- (b) *The light-cone gauge.* c₀(ε) = 1/X*(ε) is fixed at X* ≈ 1 + 3.7/n, inside the boundary layer of the light cone
  where Φ → ∞. Its Borel transform needs the behaviour of Ŷ uniformly up to x = 1.
  - The linear case has it (Lemma B3.4 holds on (0, 1]).
  - For M = 1 the evidence (§5.1) puts the singularities of c₀ and X* near ±3.1i.
- (c) *M ≥ 3.* There is no scaling reduction. The even harmonics of c(0, θ) are global data with their own singularities
  (§5.2), so the domain must avoid the singularity near ζ ≈ −1.2, i.e. σ < 1.2 at M = 3.

Lemma B3.7 already provides step 4's start. Lemmas B3.1 and B3.6 provide the geometry of steps 1–2. What remains is (a)
with explicit constants, then (b).

## 5. Numerical evidence, recomputed in ball arithmetic

**Codes.**
- `b3_outer_m1.py` (M = 1, either gauge) and `b3_outer_M.py` (any M, the projected ODE) compute Proposition 1's
  recursion in python-flint balls, at 640 bits (M = 1, K = 120) and 512 / 400 bits (M = 3, K = 100; M = 5, K = 70).
  The relative ball radii at the last order are below 10⁻⁷⁰.
- Cross-checks:
  - against `solver/largeD_late_orders.py` (double, θ grid; shares only the equations): 3.1 × 10⁻¹⁴ at K = 12, M = 1
    (`b3_crosscheck_m1.log`);
  - the two ball codes against each other: 5.6 × 10⁻⁵⁷ at M = 1;
  - against the grid code at M = 3, orders 1–3: 9.3 × 10⁻¹⁶ of the order's size. From order 4 the truncations differ,
    because the grid code divides by cb on the grid and then projects (`b3_crosscheck_M.log`).
  - Each cross-check has mutation controls (`omega_S`, `drop_bQ`, `scale_G`), which fail as they must.
- Analysis: diagonal Padé approximants at 120 digits of the Borel transform (`b3_borel_scan.py`, `b3_strip_check.py`,
  logs `b3_scan_all.log`, `b3_strip_check_*.log`), and a two-term-recurrence fit for conjugate pairs (`b3_pair_fit.py`).
  Both were first run on planted series with known singularities:
  - planted pair at 1.5866e^{±0.545πi}: nearest Padé pole at 1.5820, inside by 0.3 %;
  - a weight-10⁻³ singularity on ℝ⁺ at 2.5, behind the pair: found at 2.5019;
  - pair fit at ±iπ: error 8 × 10⁻⁴.

**A methodological finding.** Genuine singularities of these series show up as Padé poles with a zero very close by:
pole–zero distances 10⁻¹⁴ to 10⁻⁷, because of their large complex exponents (Im g ≈ μω). outer_borel.py drops every pole
with a zero within 10⁻⁴ as a "Froissart doublet", so it removes exactly the singularities it is looking for. The checks
here drop only exact doublets (zero distance below 10⁻³⁰). A candidate near ℝ⁺ counts only if the approximant from K − 6
terms has a pole within 5 % of it. That rule was added after one unstable pole at M = 3 (§5.2), and both planted controls
were rerun under it.

### 5.1 M = 1

| x | probe | nearest Borel singularity (Padé) | prediction | min \|Im\| of poles with Re > −0.5 in the disc 1.6\|a\| |
|---|---|---|---|---|
| 0.5 | D | −0.571 + 1.563i (and conjugate) | Φ ∓ iπ/2 = −0.549 ∓ 1.571i | 2.43 |
| 0.8 | D, B | 0.242 ∓ 1.59i | 0.288 ∓ 1.571i | 1.57 |
| 0.95 | D, B, G | 1.067 ∓ 1.609i | 1.113 ∓ 1.571i | 1.61 |
| 1 | D, B | −0.024 − 3.085i | u-mode at ∞; x-independent | 3.05 |

(`b3_strip_check_M1.log`; the nearest Padé pole sits 1–3 % inside the true branch point and along its cut, as in the
planted test.) The strip check passes: no pole with |Im ζ| < 1 near ℝ⁺. With a planted ℝ⁺ singularity of weight 10⁻³ at
1.3 or at 2.2 it fails at both x, finding the planted point at 1.3005 and at 2.2021 / 2.2041
(`b3_strip_check_planted_*.log`).

**x-independent singularities at M = 1** (`b3_scan_all.log`):
- c(0) (light-cone gauge): the nearest poles start at 0.008 ± 3.095i, with a cut running into the left half-plane. The
  pair fit gives 0.11 ∓ 3.18i.
- The light-cone position X* (centre gauge): 3.128i; pair fit 0.10 ∓ 3.26i.
- So both lie at distance ≈ 3.1 from ℝ⁺, near ±iπ, the x-independent singularities of the linear case. The two
  estimators do not pin the point to better than ≈ 0.1. There is no x-independent singularity near |ζ| ≈ 1.6.

**What C_selection.md §8 reported for M = 1, and why.** §8's table gives "1.44–1.86, smallest |arg| 0.51π" and calls the
cluster x-independent. In fact:
- at x = 0.5 the nearest singularity is the u-mode χ₀(0.5) (|χ₀| = 1.664);
- at x = 0.8 and 0.95 it is the u-mode at arg −0.44π and −0.30π, which the 0.48π criterion would have flagged;
- at x = 1 it is at ≈ ±3.09i.

The old procedure misses these for two reasons (`b3_old_check_controls.log`, `b3_double_noise.log`):
- *Precision.* Evaluating D_k at x = 1 in double precision cancels coefficients up to ≈ 10⁴⁶ down to ≈ 10³⁰. The result
  differs from the ball value by 1.8 × 10⁻⁶ at k = 29 and 2.3 × 10⁻⁴ at k = 33.
- *Too few terms.* Fed the exact (ball) coefficients of B at x = 0.95, rounded to double and truncated to 36 terms, the
  old procedure returns a ring of poles at |a| ≈ 1.84–1.86, arg 0.71–0.97π. The true nearest singularity is at 0.31π.

The old procedure is not blind: it fires on a planted ℝ⁺ singularity inside the dominant pair's modulus, and on a
planted u-mode pair at 0.30π. But it does not see a weight-0.05 ℝ⁺ singularity just beyond the pair (control P2).

### 5.2 M = 3 (and M = 5)

*The x-independent singularity.* The two series of c(0, θ) (harmonics 0 and 2, light-cone gauge) have their nearest Padé
poles near ζ ≈ −1.22 ± 0.15i, with a cut through the region Re ζ ≈ −1.2. Their root-test radius is 0.95 at k = 100 and
still rising (`b3_scan_all.log`). The same singularity appears in D1, D3, B0, B2 at x = 0.5 (nearest poles −1.03 to −1.26)
and at x = 0.8 (−1.15 to −1.23). It is x-independent and sits near the negative real axis, |arg| ≈ 0.9–1.0π.

So the "x-independent cluster" of C_selection.md §8 is real for M ≥ 3. It is not at 0.66–0.88 and 0.60π as tabulated
there, and it lies opposite to ℝ⁺. It limits σ to about its distance from the origin, and does not obstruct the 1-sum.
Its origin is not identified. By Lemma B3.2 it must come from the θ-dependent light-cone normalization c(1, θ) = 1, i.e.
from the even harmonics of c(0, θ) that no scaling removes. An x-independent real number near −1.2 is suggestive of
global data at the light cone, not of the x-dependent singulants. (−π²/8 = −1.2337 is close, but nothing here supports a
closed form.)

*Near ℝ⁺.* In the first run of the strip check, B0 at x = 0.8 had one pole at 1.653 − 0.696i. It appears only with
K = 100 terms and is absent with K = 94, 88, 80 (`b3_pole_stability_M3_B0.log`), so it is drawn by the approximant
outside the disc of convergence. This is what prompted the stability rule above. The full M = 3 result under that rule
is in `b3_strip_check_M3.log`: [M3_RESULT]

*M = 5.* [M5_RESULT]

### 5.3 Summary of the numerical picture

| model | singularities found near the origin (\|ζ\| ≲ 2) | σ estimate | anything on or near ℝ⁺ |
|---|---|---|---|
| linear (proved) | Φ ∓ iπ/2; iπm | π/2 | no |
| M = 1 | Φ ∓ iπ/2 (x-dependent); near ±3.1i (x-independent) | ≈ 1.6 (x < 1), ≈ 3.1 (x = 1) | no |
| M = 3 | Φ ∓ iπ/2; near −1.2 (x-independent) | [M3_SIGMA] | [M3_NEAR] |

## 6. The computer-assisted step: design and pilot

The natural certificate for "no singularity in a neighbourhood of ℝ⁺" has two stages.

**Stage 1, the disc: done for M = 1 (Lemma B3.7, certified constants above).**
- It is the positive polynomial system of Lemma B3.7 with one inequality check, Φ(t₁, V) ≤ V, in ball arithmetic. It is
  closed for all k by monotonicity, with no induction by hand.
- Trusted code: `b3_gevrey_certificate.py` (the system Φ and the check) and python-flint's arb.
- Run time: seconds.
- *Improvement path, which costs only work.* The loss factor ≈ 100 against the true radius comes from the generating
  function treating interior convolutions as O(1). An induction on the plain majorant recursion keeps their O(1/k) size
  and gives ≈ 0.146 at ρ = 0.75 (float prototype; not certified). It needs a by-hand tail estimate with the factorial
  convolution bound $`\sum_{i=a}^{k-a} i!(k-i)! \le (k-2a+1)\,a!(k-a)!`$ and a certified base range k ≤ K₀.

**Stage 2, the neighbourhood $`S_\sigma`$ of [0, ∞): designed, not run.**
- The Borel-plane system of §4, integrated along ℝ⁺ in ζ by validated Taylor models in (ζ, x). The x-dependence is
  carried on a few horn-domain Taylor discs (the technique of `lemma6b1_disc.py`). The ζ-steps are certified Taylor discs
  whose radius is limited by the distance to the nearest singularity, ≥ σ ≈ 1.5 at M = 1.
- Each ζ-step solves the transport equations by characteristics. The characteristic map Φ⁻¹ is explicit (Lemma B3.1), so
  its enclosures are cheap.
- The start disc is stage 1.
- The end condition is an exponential bound $`\lvert\hat Y\rvert \le Ce^{\nu\lvert\zeta\rvert}`$ beyond some ζ₁. It is proved by the contraction of §4 in
  the weighted norm, where the gain 1/ν is uniform.
- The blocking issue is §4(a), the derivative loss. Until it is resolved by hand, a certified integration covers only a
  bounded piece of $`S_\sigma`$, which is not a proof of (H-σ).
- Cost estimate: for M = 1, a 6-dimensional system with the x-dependence on ~10 discs and ~50 ζ-steps to |ζ| ≈ 6, about an
  hour of CPU at 200 bits. Not attempted.

**Mutation controls kept as cross-checks.**
- The old check `outer_borel.py` is kept. `b3_old_check_controls.py` gives it four planted inputs (P0 must pass; P1 and
  P3 must fire; P2 documents its blindness behind the nearest pair) and runs it on the ball data. All behave as
  documented.
- The new check `b3_strip_check.py` passes on M = 1 and fails under `--plant=1.3,1e-3` and `--plant=2.2,1e-3`.
- The certificate fails under `--mutate=m0`.
- `b3_checks_all.py` runs all of these and checks their exit codes.

## 7. The citation checks of C_selection.md §7

The channels are those of notes/REF_CHECK.md:
- WebSearch and WebFetch of publisher, numdam/centre-mersenne, eudml, arXiv and catalogue pages;
- `curl` is refused by the session proxy;
- Project Euclid, HAL and ScienceDirect refused the fetch (bot protection or robots);
- arXiv returned "rate limited" twice for Costin's paper, so it was not reread.

"Read" below means the statement was read on the page named. Nothing is quoted from memory.

### 7.1 The mixed slow–fast version of B3

**What is needed.** The outer system has fast components (u = p + m per harmonic, b − 1) and slow ones (p − m, c). Near
x = 0 it is doubly singular (ε x y′ = …) in the fast components, while the slow components are regular-singular in x
without ε. B3 as stated needs:
1. the formal series is Gevrey-1;
2. an actual centre-regular solution exists, asymptotic to it;
3. (strong form) it is the 1-sum.

**Theorems read.**

| work | read on | equation and hypotheses | covers our case? |
|---|---|---|---|
| M. Canalis-Durand, *Solution formelle Gevrey d'une équation singulièrement perturbée : le cas multidimensionnel*, Ann. Inst. Fourier 43 (2) (1993) 469–483 | numdam landing page and full text (OCR) | ε dz/dx = G(z, x, a, ε), G analytic, Gevrey-1 in ε, a control parameter; a slow curve φ(x) with $`D_zG`$ having 0 as a simple eigenvalue at x = 0 (turning point, canard type). Every component carries ε. Conclusion: the formal solutions and the control parameter are Gevrey-1 (Theorem 3) | no: no slow components, and the setting is a turning point |
| M. Canalis-Durand, J. Mozo-Fernández, R. Schäfke, *Monomial summability and doubly singular differential equations*, J. Differential Equations 233 (2007) 485–511 | abstract only (R. Schäfke's talk page at RIMS Kyoto, kurims.kyoto-u.ac.jp/~kenkyubu/kyodo/takei/schaefke.pdf); HAL record blocked; ScienceDirect blocked | $`\varepsilon^\sigma x^{r+1}y' = f(x, \varepsilon, y)`$, f analytic near 0, f(0,0,0) = 0, ∂_y f(0,0,0) invertible ⇒ the unique formal solution is monomially summable | no: invertibility of ∂_y f on all components excludes slow components. It could apply to the fast subsystem at x = 0 only with the slow components treated as given (not checked) |
| J.-P. Ramis, R. Schäfke, *Gevrey separation of fast and slow variables*, Nonlinearity 9 (2) (1996) 353–384, doi:10.1088/0951-7715/9/2/004 | abstract (neicon archive record) | perturbations of one-phase Hamiltonian systems in action–angle variables; iterated averaging removes the fast phase with a Gevrey-1 transformation, giving Neishtadt's exponential estimates | no: "fast" here is a rotating phase, not a normally hyperbolic fast direction; not a slow-manifold theorem |
| M. Canalis-Durand, J.-P. Ramis, R. Schäfke, Y. Sibuya, *Gevrey solutions of singularly perturbed differential equations*, J. reine angew. Math. 518 (2000) 95–129 | not read: the publisher page was not reached; academia.edu shows only the journal name and year and a garbled summary | (statement as recalled in C_selection.md: ε y′ = F(x, y, ε) with ∂_yF invertible on a domain without turning points) *(unverified)* | by that statement, no (all components fast) |
| Y. Sibuya, *Gevrey property of formal solutions in a parameter* (1990) | not found | — | *(unverified)* |

**Conclusion and the extra argument.**
- None of the theorems read covers a system with slow components and a regular singular point. The closest (CMS 2007)
  needs every component fast at the singular point.
- The three parts of B3 are now handled as follows:
  1. *Gevrey-1 of the formal series:* **proved** here (Lemma B3.7), for every M, in both gauges, on discs containing the
     centre and the light cone. No citation is needed.
  2. *An actual solution with fixed-order asymptotics:* this is the restriction of B8 to the real segment (C_selection.md
     §11). B8's contraction, with the centre as a regular-singular point and the slow components integrated from
     x = 0, is the argument C_selection.md §4.2 calls "routine". Writing B8 with constants proves B3 in the form Theorem
     B(M) uses. The alternative, citing CMS 2007 for the fast subsystem and closing over the slow components by a
     contraction, needs the same work.
  3. *The 1-sum:* this is (H-σ) (§3–4), **open**. For the selection theorem it is not needed (§1).

### 7.2 Parameter-analyticity of the inner lateral sums (B4)

**What is needed.** B7 (the implicit function theorem) needs the multipliers S_ν(F, ω) to be C¹, best analytic, in
(F, ω). They are read off the lateral sums Z±(u; F, ω) of the inner formal series.

**Theorems read.**

| work | read on | statement | parameters? |
|---|---|---|---|
| B. L. J. Braaksma, *Multisummability of formal power series solutions of nonlinear meromorphic differential equations*, Ann. Inst. Fourier 42 (3) (1992) 517–540, doi:10.5802/aif.1301 | numdam landing page; full text on aif.centre-mersenne.org (OCR) | $`x^{\nu+1}y' = F(x, y)`$, x ∈ ℂ, y ∈ ℂⁿ, ν ≥ 1, F analytic in a neighbourhood of (0, a), with a formal power-series solution y = Σ c_m x^m, c₀ = a. Then y is multisummable (a proof of Écalle's theorem); the levels and singular directions come from the linear part's normal form (eigenvalues of the leading matrices); no nonresonance assumption | **no**: the words "parameter", "parameters" and "depending analytically" do not occur in the paper |
| J.-P. Ramis, Y. Sibuya, *A new proof of multisummability of formal solutions of non linear meromorphic differential equations*, Ann. Inst. Fourier 44 (3) (1994) 811–848 | eudml record (abstract) | the same theorem, via the Malgrange–Ramis definition and the cohomological (Ramis–Sibuya) method | not in the abstract; the full text was not searched |
| W. Balser, *Formal Power Series and Linear Systems of Meromorphic Ordinary Differential Equations*, Springer (2000) | not reread (bibliographic data verified in notes/REF_CHECK.md) | linear systems | — |
| O. Costin, Duke Math. J. 93 (1998) 289–344 | arXiv page rate-limited; not reread | hypotheses (invertible Λ, nonresonance) fail here (C_selection.md §2) | — |

**Hypothesis check for B4.** The inner problem is u²Z′ = H(u, Z) with H polynomial (C_selection.md §4.1). That is
Braaksma's form with ν = 1, F = H analytic everywhere, and a formal power-series solution (the inner series, unique given
F). So **summability is covered by Braaksma 1992 as read**. Its only level is 1, and its singular direction is arg u = 0,
since the only nonzero eigenvalue of the linear part is ½ in 1/u. Parameter dependence is not covered.

**Lemma B4.P (analytic dependence of the lateral sums on (F, ω); proved, given Braaksma's theorem).** Let λ = (F's
harmonics, ω) range over a complex neighbourhood U of a compact set of real data on which the inner recursion's divisors
2j + 1 + iμω do not vanish. Then Z±(u; λ) and the multipliers S_ν(λ) are holomorphic in λ ∈ U.

*Proof.*
1. *The coefficients.* Z_j(λ) of the formal series are rational in λ with poles off U, by the recursion. So
   Ẑ_N(u; λ) = Σ_{j<N} Z_j u^j is holomorphic in λ.
2. *A ray in the large-exponential sector.* Fix a ray r_θ = {u = re^{iθ} : 0 < r ≤ r₀} with π/2 < θ < π. There
   Re(1/u) < 0, so the fast homogeneous solutions e^{−1/(2u)}u^g grow as u → 0. The slow homogeneous solutions are
   powers u^{−(1+iμω)/2} and u⁰.
3. *Uniqueness.* For N ≥ N₀ a solution with Z − Ẑ_N = O(u^N) on r_θ is unique: the difference of two such solutions
   solves a linear system all of whose nonzero solutions are either exponentially large or of bounded power order.
4. *Existence, uniformly in λ.* Write Z = Ẑ_N + η. Then η solves a Volterra equation along r_θ:
   - the fast components are integrated from u = 0, with kernels $`e^{-1/(2u)+1/(2s)}`$ of modulus ≤ 1 for |s| < |u|;
   - the slow components are also integrated from 0.

   For r₀ small this is a contraction on {|η(u)| ≤ C|u|^N}, with constants continuous in λ, hence uniform on compact
   subsets of U.
5. *Holomorphy.* The Picard iterates are holomorphic in λ and converge uniformly, so η is holomorphic in λ (Weierstrass).
6. *Identification.* Braaksma's lateral sum Z⁺ is asymptotic to Ẑ on a sector containing r_θ, since the 1-sum in any
   direction of (0, π) has Gevrey asymptotics there. So Z⁺ − Ẑ_N = O(u^N), and by step 3, Z⁺ = Ẑ_N + η on r_θ.
7. *Propagation.* Away from u = 0 the equation is regular, so Z⁺ is holomorphic in λ on its whole sector (analytic
   dependence on initial data and parameters). The same holds for Z⁻ (ray with −π < θ < −π/2).
8. *The multipliers.* S_ν is read off Z⁺ − Z⁻ on arg u = 0 against a basis of fast solutions that is itself holomorphic
   in λ, so S_ν is holomorphic. $`\square`$

For M = 1, steps 2–5 are exactly what A_certify.md §4.5 certifies on a complex box (θ = 3π/4, N = 53, contraction
constant 0.29). Lemma B4.P is the general-M, all-parameters version. Braaksma's theorem enters only in step 6 (existence
of Z⁺ and its asymptotics on the sector), as in A_certify.md.

### 7.3 The summability criterion used in §3

| work | read on | statement used |
|---|---|---|
| A. D. Sokal, *An improvement of Watson's theorem on Borel summability*, J. Math. Phys. 21 (2) (1980) 261–263 | title, journal and year on sugaku.net; volume, issue and pages from the bibliography of Gillam–Gurarii (arXiv:math/0508425, a citing record, not the publisher) | the statement read is Nevanlinna's theorem as given by Gillam–Gurarii. P analytic in a half-plane of the large variable with uniform Gevrey-1 remainder bounds ⟺ the Borel transform continues from the disc to a half-strip (a neighbourhood of [0, ∞)) with exponential bounds, and P is its Laplace transform. Sokal's own paper was not opened |

## 8. Status

| item | content | status |
|---|---|---|
| H-a1/H-a2 as stated | all singularities at \|arg\| ≥ 0.48π; 2Φ absent | **false as stated** (the u-mode is at −0.30π at x = 0.95), harmlessly; replaced by (H-σ) |
| role in Theorem B(M) | — | **not used** by B5–B8 as written; B8.2's "lateral sum" wording to be changed (§9) |
| B3.1 Φ, ln x conformal onto the strip | | **proved** |
| B3.2 scaling symmetry, gauges | | **proved** |
| B3.3 centre-gauge Taylor coefficients rational in n | | **proved** |
| B3.4 linear case: strip of width π/2 | | **proved** |
| B3.5 convolution in star-shaped domains; 2Φ not on the principal sheet of a convolution fixed point | | **proved** |
| B3.6 fast transport in the Borel plane; singular loci on Im ζ ∈ π(ℤ + ½) | scalar model | **proved** |
| B3.7 outer formal series Gevrey-1 on every disc, every M, both gauges | | **proved**; constants **certified** for M = 1 (`b3_gevrey_certificate.py`) |
| 2Φ absent near x = 1/√2 | | **proved** (every M, from B3.7); interval **certified** for M = 1 |
| (H-σ), M = 1 | Borel transform analytic near [0, ∞) | **conjecture**; **numerical evidence** σ ≈ π/2; proof plan §4 with one open technical point (derivative loss) |
| (H-σ), M ≥ 3 | | **conjecture**; numerical evidence that σ is set by an x-independent singularity near −1.2 (M = 3) |
| B3 formal half | | **proved** (B3.7) |
| B3 actual solution, fixed order | | = B8 on the real segment (to be written with constants) |
| B3 strong form (1-sum) | | ⇔ (H-σ) (Nevanlinna–Sokal), **open** |
| B4 summability | | **cited**, Braaksma 1992, hypotheses checked against the paper |
| B4 parameter-analyticity | | **proved** given Braaksma (Lemma B4.P); not in Braaksma 1992 or the Ramis–Sibuya 1994 abstract |

## 9. Proposed text for the other files

These are proposals for the coordinator; this note does not edit those files.

**C_selection.md §4.2, replacing the paragraph "Borel singularities: two hypotheses to verify" and the H-a1/H-a2 bullets:**

> **Borel singularities: hypothesis (H-σ).** For the stronger statement that the centre-regular solution is the 1-sum
> in ε along arg ε = 0, the Borel transform in ε must continue, with exponential bounds, to a neighbourhood
> S_σ = {dist(ζ, [0, ∞)) < σ} of the positive real axis, uniformly on compacts of (0, 1] (Nevanlinna–Sokal). This is
> not used by B5–B8, which work with actual solutions and fixed-order truncations. The earlier formulation ("every
> singularity at |arg| ≥ 0.48π") is false for x > 1/√2, because the u-mode singulant Φ(x) − iπ/2 itself has argument
> down to −0.30π at x = 0.95, but it lies on Im ζ = −π/2. (H-σ) is proved in the linear case with σ = π/2, and near ζ = 0
> for every M, where the outer series is Gevrey-1 on every disc in x. That excludes the would-be 2Φ(x) singularity near
> x = 1/√2. Elsewhere it is supported numerically. See [B3_borel.md](B3_borel.md).

**C_selection.md §4.2, the sentence "A precise citation for the mixed case should be pinned down":**

> No theorem read covers the mixed case: Canalis-Durand (1993) and Canalis-Durand–Mozo-Fernández–Schäfke (2007) assume
> every component fast (B3_borel.md §7.1). The formal half is proved directly (B3_borel.md, Lemma B3.7). The existence
> half with fixed-order asymptotics is B8 restricted to the real segment.

**C_selection.md §4.1, the paragraph "To check in the paper: whether Braaksma's statement covers parameter-analytic
dependence":**

> Braaksma's theorem (checked against the paper) is stated for x^{ν+1}y′ = F(x, y) without parameters, and neither it
> nor the abstract of Ramis–Sibuya (1994) addresses parameter dependence. Analytic dependence on (F, ω) follows from
> uniqueness on a ray where e^{−1/(2u)} is large and a contraction uniform in the parameters (B3_borel.md, Lemma B4.P;
> carried out for M = 1 on a box in A_certify.md §4.5).

**C_selection.md §8:** add at the top: "*Superseded by B3_borel.md §5: the double-precision Padé readings below are
unreliable. The M = 1 'cluster' is the u-mode singulant itself, which at x = 0.8 and 0.95 lies at |arg| 0.44π and 0.30π
on Im ζ = −π/2. The x-independent singularity is real for M ≥ 3 but lies near the negative real axis (ζ ≈ −1.2 at
M = 3).*"

**C_selection.md §11, B8.2:** replace "is the upper lateral sum of the slow series on all of $`\Sigma^+_n`$, with no free
fast mode" by "has no free fast mode on all of $`\Sigma^+_n`$, in the fixed-order sense of Lemma B8 (its identification
with the upper lateral Borel sum is the complex-x form of hypothesis (H-σ), B3_borel.md, and is not needed)".

**C_selection.md §13, row H-a1/H-a2:** "replaced by (H-σ) (B3_borel.md): proved in the linear case and near ζ = 0 for
every M (Gevrey-1 on every disc; 2Φ excluded near 1/√2, certified for M = 1); conjecture beyond, with ball-arithmetic
numerical evidence (M = 1, 3); not used by Theorem B(M)". Row B3: "formal half proved (B3_borel.md, B3.7); existence =
B8 on the real segment; 1-sum = (H-σ)". Row B4: "cited (Braaksma 1992, checked); parameter-analyticity proved given
Braaksma (B4.P)".

**C_selection.md References:**
- "Braaksma … 517–540" add doi:10.5802/aif.1301.
- Add Ramis–Sibuya 1994, Canalis-Durand 1993 and Sokal 1980 as in the references below.
- Keep CDRSS 2000 and Sibuya 1990 marked *(unverified)*.

**PROOF.md §5, the sentence "The outer half rests on Gevrey-1 theory without turning points plus two cheap numerical
hypotheses about where Borel singularities lie, now checked numerically …":**

> The outer formal series is Gevrey-1 on every disc in x, for every M (proved by majorants, constants certified for
> M = 1; proof/B3_borel.md). Whether it is Borel summable along ℝ⁺ (hypothesis (H-σ), proved in the linear case) is not
> needed for Theorem B(M), whose outer steps use fixed-order truncations only.

**README** ("Where rigour fails" (a)): "the outer nonlinear late orders also carry x-independent singularities of
modulus about 0.13" is not supported at fixed M by the ball-arithmetic recomputation. At M = 1 there is none near the
origin; at M = 3 there is one near −1.2. Suggested: "for M ≥ 3 the outer late orders carry an x-independent singularity
near the negative real axis (ζ ≈ −1.2 at M = 3), which does not obstruct summation along ℝ⁺ but may approach the origin
as M grows (proof/B3_borel.md §5)".

**notes/REF_CHECK.md, proposed lines (addendum, B3_borel.md):**

| Entry | Checked against | Verdict |
|---|---|---|
| Braaksma, Ann. Inst. Fourier 42 (3) (1992) 517–540, doi:10.5802/aif.1301 | numdam item AIF_1992__42_3_517_0 (title, author, volume, issue, pages, DOI); full text on aif.centre-mersenne.org | OK; claim check: no parameter dependence in the paper |
| Ramis, Sibuya, Ann. Inst. Fourier 44 (3) (1994) 811–848 | eudml record 75082 | OK (bibliographic); abstract read |
| Canalis-Durand, Ann. Inst. Fourier 43 (2) (1993) 469–483, doi:10.5802/aif.1341 | numdam item AIF_1993__43_2_469_0 and full text | OK |
| Ramis, Schäfke, Nonlinearity 9 (2) (1996) 353–384, doi:10.1088/0951-7715/9/2/004 | neicon archive record (arch.neicon.ru) | OK (bibliographic); not applicable to B3 |
| Canalis-Durand, Mozo-Fernández, Schäfke, J. Differential Equations 233 (2007) 485–511 | earlier REF_CHECK addendum; statement from the abstract on Schäfke's RIMS talk page; HAL and ScienceDirect blocked | bibliographic as before; claim: ∂_y f(0,0,0) invertible |
| Canalis-Durand, Ramis, Schäfke, Sibuya, J. reine angew. Math. 518 (2000) 95–129 | academia.edu listing (journal, year only) | PARTIAL: keep *(unverified)* on volume/pages and on the statement |
| Sokal, J. Math. Phys. 21 (2) (1980) 261–263 | sugaku.net (title, journal, year); bibliography of Gillam–Gurarii, arXiv:math/0508425 (volume, issue, pages) | PARTIAL: citing record for volume/pages |
| Sibuya, *Gevrey property of formal solutions in a parameter* (1990) | not found | UNVERIFIABLE |

## References

- B. L. J. Braaksma, *Multisummability of formal power series solutions of nonlinear meromorphic differential
  equations*, Ann. Inst. Fourier 42 (3) (1992) 517–540, doi:10.5802/aif.1301.
- J.-P. Ramis, Y. Sibuya, *A new proof of multisummability of formal solutions of non linear meromorphic differential
  equations*, Ann. Inst. Fourier 44 (3) (1994) 811–848.
- M. Canalis-Durand, *Solution formelle Gevrey d'une équation singulièrement perturbée : le cas multidimensionnel*,
  Ann. Inst. Fourier 43 (2) (1993) 469–483, doi:10.5802/aif.1341.
- M. Canalis-Durand, J. Mozo-Fernández, R. Schäfke, *Monomial summability and doubly singular differential equations*,
  J. Differential Equations 233 (2007) 485–511, doi:10.1016/j.jde.2006.11.005.
- M. Canalis-Durand, J.-P. Ramis, R. Schäfke, Y. Sibuya, *Gevrey solutions of singularly perturbed differential
  equations*, J. reine angew. Math. 518 (2000) 95–129 *(volume and pages unverified)*.
- J.-P. Ramis, R. Schäfke, *Gevrey separation of fast and slow variables*, Nonlinearity 9 (2) (1996) 353–384,
  doi:10.1088/0951-7715/9/2/004.
- A. D. Sokal, *An improvement of Watson's theorem on Borel summability*, J. Math. Phys. 21 (2) (1980) 261–263
  *(volume, issue and pages from a citing record)*.
- D. W. H. Gillam, V. Gurarii, *On a Watson-like uniqueness theorem and Gevrey expansions*, arXiv:math/0508425 (for the
  statement of Nevanlinna's theorem used in §3).
- Y. Sibuya, *Gevrey property of formal solutions in a parameter* (1990) *(unverified)*.
- W. Balser, *Formal Power Series and Linear Systems of Meromorphic Ordinary Differential Equations*, Springer (2000).
- O. Costin, *On Borel summation and Stokes phenomena for rank-1 nonlinear systems of ordinary differential equations*,
  Duke Math. J. 93 (1998) 289–344, arXiv:math/0608408.

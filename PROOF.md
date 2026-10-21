# Toward a proof of the selection rule

This page is a roadmap for turning Conjecture 1 of the [README](README.md#large-d-what-the-1d-expansion-can-and-cannot-see)
into a theorem: at D = ∞ the echo frequency ω∞ = 2π/Δ∞ is selected by the vanishing of the Stokes multipliers of the
n-free inner problem. It says what is already proved here, which existing theorems and techniques supply the remaining
steps, what is genuinely new, and in what order to attack it. It is written by Claude (Anthropic) under Nathan
Moinvaziri's direction and has not been reviewed by a mathematician; the citations marked *(from memory)* should be
checked against the published records.

Contents

- [Summary](#summary)
- [1. The statement, in three layers](#1-the-statement-in-three-layers)
- [2. What is already proved here](#2-what-is-already-proved-here)
- [3. Precedents](#3-precedents)
- [4. Layer A: the inner zero, computer-assisted](#4-layer-a-the-inner-zero-computer-assisted)
- [5. Layer B: the analytic selection theorem](#5-layer-b-the-analytic-selection-theorem)
- [6. Layer C: identifying the zero with the solutions' limit](#6-layer-c-identifying-the-zero-with-the-solutions-limit)
- [7. Staged plan](#7-staged-plan)
- [8. Who works on this](#8-who-works-on-this)
- [References](#references)

## Summary

- **The structure of the problem is a known one.** "A smooth solution exists only if an exponentially small Stokes
  multiplier of an inner equation vanishes, and the selected parameter is a zero of that multiplier" is exactly the
  structure of Saffman–Taylor finger selection and dendrite (needle crystal) selection. For those problems the selection
  was derived formally in 1986–91 (Combescot et al., Hong–Langer, Tanveer, Kruskal–Segur) and *proved* about 15 years
  later by Xie and Tanveer, by analytic continuation into the complex plane, contraction mappings in weighted spaces of
  analytic functions, and the implicit function theorem at a simple zero of the multiplier. That is the template for
  layer B.
- **Rigorous Stokes constants of inner equations are now computer-provable.** Baldomá, Capiński, Guardia and Seara
  (arXiv:2206.13840) give a computer-assisted proof of the value (and non-vanishing) of the Stokes constant of an inner
  equation in a splitting-of-separatrices problem: a fixed-point argument in weighted spaces at large |u| for the
  asymptotic solutions, validated integration in between, and rigorous bounds on the difference. That is the template
  for layer A.
- **What is genuinely new here** is that the unknown is a function of two variables (x, θ), periodic in θ, so the inner
  problem is an infinite system of ODEs coupled through harmonics, and its Stokes multipliers carry weights
  $`e^{\pi\lvert\nu\rvert\omega/2}`$ that grow faster than the profile's harmonics decay (gap (b)). Neither template has
  that. Costin–Tanveer's Borel summability for PDEs with data analytic in a strip is the closest tool.
- **Realistic targets, in order:** (A) a computer-assisted proof that the truncated (M = 5) inner multiplier has a simple
  zero in an explicit small box around (A, ω/ω∞) = (2.7186, 0.9992), then a proof with a rigorous harmonic-tail bound
  for the untruncated multiplier; (B) the selection theorem for large n, first for a model with finitely many harmonics,
  then in a weighted space of functions analytic in a θ-strip; (C) then follows as a corollary of (B), with no further
  numerics. (A) is a substantial but standard computer-assisted proof; (B) is a research-level paper in exponential
  asymptotics; together they would be the first rigorous derivation of a critical-collapse echo period.
- **A new structural fact that helps both:** the amplitude-series coefficients satisfy
  |σ_k/σ_{k+1}| ≈ 0.60 k + 2.5 with remarkable regularity (k = 3–11, `solver/proof/feasibility.py`), so S₁(·, ω) is,
  numerically, an *entire* function of x = A² of order 1 and exponential type (about 1.67 in x;
  [proof/D_entire.md](proof/D_entire.md)). Its logarithm's singularities are then exactly its zeros, which makes
  argument-principle and Miranda-type root certificates natural. Factoring out the exponential,
  G = e^{−κ̄x}S₁, removes the cancellation (the terms at x₀ are O(1) and fall to 4×10⁻⁴ by k = 12 instead of peaking near
  240), puts the real zero at ω/ω∞ = 0.99988 at twelve orders (A₀ = 2.7194), and shows the zero x₀ ≈ 7.40 is isolated
  (the next zeros lie at |x| > 10).

## 1. The statement, in three layers

Notation as in the README: n = D − 2, s = iω, F(θ) the leading-order echo profile (F = A sin θ + higher odd harmonics),
ξ = x/√n the inner variable, θ₀ = θ − 2ω ln ξ the characteristic label, and $`S_\nu(F, \omega)`$ the fast-type Stokes
multipliers of the inner problem

```math
L y = c\,\sigma,\qquad L\sigma = c\,\xi y_\xi + c\,\big(\tfrac12 y^2 - 1\big)\,y,\qquad \xi c_\xi = \tfrac12 c\,y^2,\qquad L = \xi^2\partial_\xi + \omega\,\xi\,\partial_\theta .
```

**Theorem A (inner zero; computer-assisted).** There is an explicit box
$`B = [A_-, A_+]\times[\omega_-, \omega_+]`$ and an explicit profile family F(A, ω) with $`S_\nu(F, \omega) = 0`$ for all
odd |ν| ≥ 3, such that $`S_1(F(A,\omega), \omega)`$ has exactly one zero (A*, ω*) in B, and it is simple (the real 2×2
Jacobian is invertible).

**Theorem B (selection for large n).** Let (A*, ω*) be a simple zero as in Theorem A. Then for all sufficiently large n
there is a discretely self-similar solution, regular at the centre and analytic across the light cone x = 1, whose
profile and frequency satisfy F_n → F(A*, ω*) and ω_n = ω* + O(1/n). Conversely, any family of such solutions with
F_n → F_∞, ω_n → ω_∞ in the sense of hypothesis (H1) has S(F_∞, ω_∞) ≡ 0.

**Corollary C (identification).** If the critical solutions computed for D = 8–61 form the branch of Theorem B, then
ω∞ = ω*, and Δ∞ = 2π/ω* is determined by the inner problem alone.

The numerics so far are consistent with all three: the twelve-order amplitude expansion brackets ω*/ω∞ in
0.99915–1.00011 with Padé estimates 0.9999, and the five-harmonic truncation gives 0.999228 (README, "Where that leaves
it"). They are evidence, not part of the proof.

## 2. What is already proved here

Proved in the README and the paper, by the methods named; not yet refereed.

| Result | Content | Method | Role in the proof |
|---|---|---|---|
| Proposition 1 | the 1/n expansion of the self-similar equations can be carried to all orders for any profile F and any ω | parity induction on the formal series | B.1: the formal outer solution exists; the selection must be beyond all orders |
| Proposition 2 | the leading-order (no gravity) problem is solved by a hypergeometric function; its light-cone coefficient C(n, s) is explicit and non-zero | exact solution | the linear case of Theorem B, exactly |
| Proposition 3 | with gravity, the fast WKB mode is transported inside the light cone by an invertible map | characteristics of a first-order transport equation | B.4 |
| Proposition 4 | the late orders of the leading-order series carry the light-cone coefficient as a Stokes multiplier, with the ±iπ structure near the light cone, for 0.066 < x < 0.998 | Euler integral, Watson's lemma, Darboux | the linear case of B.3 (outer late orders ↔ inner Stokes constant) |
| Inner linear constant | the inner Weber (parabolic-cylinder) Stokes constant equals the outer one: $`2^{-s-1/2}/(\Gamma(\alpha)\Gamma(\beta)) = 1/(\sqrt{2\pi}\,\Gamma(1+s))`$ | duplication formula; checked to 6×10⁻²⁷ by Borel–Darboux | the linear case of B.3 and A.1 |
| Exact inner modes | the inner growth rates λ = 0 (θ shift) and λ = 1 (time shift) are exact | symmetries | the kernel to quotient out in B.7 |

So the linear (no-gravity) version of the whole chain, outer late orders → inner Stokes constant → light-cone
coefficient, is proved. Everything nonlinear is open.

## 3. Precedents

| Work | What it proves | How it maps onto this problem |
|---|---|---|
| Kruskal & Segur, *Asymptotics beyond all orders in a model of crystal growth*, Stud. Appl. Math. 85 (1991) 129–181 *(from memory)* | the formal selection mechanism: a solution exists only if an exponentially small term, the Stokes multiplier of an inner equation near a complex singularity, vanishes | the same mechanism; our inner problem plays the role of their inner equation, and our light cone that of their symmetry condition |
| Combescot, Dombre, Hakim, Pomeau & Pumir, PRL 56 (1986) 2036; Hong & Langer, PRL 56 (1986) 2032; Tanveer, Phys. Fluids 30 (1987) 1589 *(from memory)* | finger-width selection in Hele-Shaw flow by the zeros of an inner Stokes multiplier (formal) | a selected continuous parameter (finger width λ ↔ our ω) from the discrete zeros of S |
| Xie & Tanveer, *Rigorous results in steady finger selection in viscous fingering*, Arch. Ration. Mech. Anal. 166 (2003) 219–286; Tanveer & Xie, *Analyticity and nonexistence of classical steady Hele-Shaw fingers*, Comm. Pure Appl. Math. 56 (2003) 353–402 *(from memory)* | the rigorous version: a steady finger exists for small surface tension iff the inner Stokes multiplier vanishes; at a simple zero the implicit function theorem gives the selected width | the template for Theorem B: weighted spaces of functions analytic in a complex domain, contraction for a "half problem" without the symmetry condition, then the symmetry condition ⇔ S = 0, then the IFT |
| Xie, *Rigorous results in existence and selection of Saffman–Taylor fingers by kinetic undercooling*, arXiv:1610.00076 | the same for a variant where the coefficients are non-analytic at the tip; Theorem 1.11 there gives Im ψ₀ ~ S̃(α)e^(−η), and S̃(β₀) = 0, S̃′(β₀) ≠ 0 gives the selected branch by the IFT | the closest modern statement of "existence ⇔ S = 0 at a simple zero"; also how to handle Stokes lines meeting a non-analytic point (our light cone is a regular singular point, gap (c)) |
| Xie, *Rigorous results in selection of steady needle crystals*, J. Differential Equations 197 (2004) 349–426 *(from memory)* | dendrite selection with anisotropic surface tension | a second instance of the same proof architecture |
| Costin, *On Borel summation and Stokes phenomena for rank-1 nonlinear systems of ordinary differential equations*, Duke Math. J. 93 (1998) 289–344, arXiv:math/0608408; Costin, *Asymptotics and Borel summability*, CRC 2008 *(from memory)* | formal transseries solutions of analytic rank-one nonlinear ODE systems are Borel summable; resummed transseries are exactly the solutions decaying in sectors; the Stokes phenomenon as a change of transseries constant across a Stokes line | gap (a) for each fixed harmonic truncation: the inner problem at ξ → ∞ is (after t = 1/ξ) a rank-one irregular singular point; its fast mode $`e^{-1/(2\xi^2)}`$ is the exponential, and S is the Stokes constant in Costin's sense |
| Costin & Tanveer, *Nonlinear evolution PDEs in R⁺ × C^d: existence and uniqueness of solutions, asymptotic and Borel summability properties*, Ann. IHP (C) 24 (2007) 795–823 *(journal details from memory)*; *Borel summability of Navier–Stokes equation in R³ and small time existence*, arXiv:math/0612063 | Borel summability for PDEs with data analytic in a strip, with weighted-Fourier-space norms | the tool for gap (b): treat the θ-dependence in a space of functions analytic on a strip $`\lvert\mathrm{Im}\,\theta\rvert\lt\sigma`$ instead of harmonic by harmonic |
| Chapman, King & Adams, *Exponential asymptotics and Stokes lines in nonlinear ordinary differential equations*, Proc. R. Soc. A 454 (1998) 2733–2755 *(details from memory)* | the late-order (Dingle) method for nonlinear ODEs: singulant, prefactor, Stokes switching | the formal backbone of Proposition 4 and the outer–inner matching; already used here |
| Baldomá, Fontich, Guardia & Seara, *Exponentially small splitting of separatrices beyond Melnikov analysis: rigorous results*, arXiv:1201.5152 | rigorous matching of inner-equation solutions to outer approximations; the difference of two inner solutions replaces the Melnikov integral | rigorous inner–outer matching (lemma B.3) |
| Baldomá, Capiński, Guardia & Seara, *Breakdown of heteroclinic connections in the analytic Hopf-zero singularity: rigorous computation of the Stokes constant*, arXiv:2206.13840 | a computer-assisted proof of the value and non-vanishing of the Stokes constant of an inner equation: existence domains for the inner solutions, explicit operator bounds, fixed points, rigorous bounds | the template for Theorem A (with the difference that we need a *zero* of a multiplier, certified by an interval Newton/Krawczyk test, not a non-zero value) |
| Lombardi, *Oscillatory integrals and phenomena beyond all algebraic orders*, Lecture Notes in Math. 1741, Springer 2000 *(from memory)* | rigorous beyond-all-orders analysis for reversible systems: existence of homoclinic or generalized solitary waves iff an exponentially small coefficient vanishes | another rigorous instance of "existence ⇔ exponentially small coefficient = 0", for a problem with an oscillatory (θ-like) component |
| Reiterer & Trubowitz, *Choptuik's critical spacetime exists*, arXiv:1203.3766 | computer-assisted existence of a real-analytic discretely self-similar solution of the 4D Einstein–scalar equations near the past light cone, from a Fourier–Chebyshev approximate solution by a contraction argument with exact rational arithmetic; K (Δ = 2K) enclosed to 80 digits | proves that finite-D critical solutions can be certified; layer C can use the same method at a few finite D to identify the computed branch, and its high/low-frequency split is a model for the harmonic-tail bound in A.3 |
| Gundlach, *The Choptuik spacetime as an eigenvalue problem*, gr-qc/9507054 | the formulation of the DSS solution as a nonlinear eigenvalue problem for Δ, with regularity at the centre and the light cone | the formulation used by both solvers here; the "eigenvalue" is selected by analyticity at the light cone, exactly the condition C = 0 |
| Chatzikaleas, Donninger & Glogić, *On blowup of co-rotational wave maps in odd space dimensions*, arXiv:1701.05082 (and Donninger–Schörkhuber's method it builds on) | nonlinear asymptotic stability of self-similar blowup solutions in similarity coordinates | the functional-analytic setting (similarity coordinates, light cone as a boundary) for perturbations; relevant to γ (λ∞ = 2), not to the selection itself |

## 4. Layer A: the inner zero, computer-assisted

Two routes. The second is the one to build.

**Route A1: sum the amplitude series rigorously.** S₁/S₁⁽¹⁾ = Σ σ_k x^k, x = A². The σ_k are computed in ball arithmetic,
but each comes from a *fit* of late orders (j ≤ J) to an asymptotic form, which is not a rigorous enclosure; the ball radii
(10⁻²⁰) measure rounding, not the fit's truncation. A rigorous σ_k needs the late-order remainder bounded, i.e. a
rigorous Darboux/Borel statement for each order, which is as hard as the direct route. Worse, at the zero the series is
far from summed at K = 25: the terms |σ_k| x₀^k grow to about 240 at k = 9 and are still 160 at k = 12, and with the
observed growth law (|σ_k/σ_{k+1}| ≈ 0.60k + 2.5) they fall below 10⁻⁶ only near k ≈ 37 (K ≈ 75)
(`solver/proof/feasibility.py`). Route A1 is useful as a cross-check and for the structure (S₁ entire in x), not for the
certificate.

**Route A2: validated integration of the inner problem, then a Krawczyk test.** For given (A, ω) and a profile with M
harmonics, the multiplier is the jump across the Stokes line of solutions started from the formal series at large |ξ|
(`solver/inner_stokes/lateral_np.py`, `hier.py`). The rigorous version needs four lemmas:

- **A.1 Asymptotic start.** For |ξ| ≥ R on a ray, the formal inner series (Gevrey-1 in 1/ξ², with the fast modes
  $`e^{-1/(2\xi^2)}\xi^{2\langle F^2\rangle}e^{\Pi(\theta)}`$) is the asymptotic expansion of true solutions with explicit
  remainder bounds. Source: Costin (1998) for existence and summability in sectors; for explicit constants, the fixed-point
  construction in weighted spaces of Baldomá–Capiński–Guardia–Seara. New: the system is (2M+1)-dimensional with
  θ-coupling, but for fixed M it is a finite ODE system with a rank-one irregular singular point, inside Costin's theory.
- **A.2 Validated integration** from |ξ| = R to the matching radius ξ₁ ≈ 0.2 along rays on both sides of the Stokes
  line, in interval or ball arithmetic (Taylor models; the double-double kernel's Taylor stepping is already a natural
  base). Standard.
- **A.3 Harmonic tail.** A bound on how much the omitted harmonics |ν| > M change r₁. This is gap (b) in finite form,
  and the only genuinely hard lemma of layer A. The measured truncation sequence (ω/ω∞ − 1 = −22%, +1.56%, −0.077% for
  M = 1, 3, 5; about twentyfold per pair) says the effect is small and decays geometrically. A proof needs the
  higher-harmonic multipliers (tuned to zero) and their coupling into r₁ bounded in a weighted norm; the leakage
  coefficients measured in `hier.py` (∂r₃/∂B₅ ~ 10¹³ and ∂r₁/∂B₅ ~ 10¹⁷ at M = 5; at M = 7 one ulp of B₇ moves r₁ by
  about 200) show that this must be done for the *tuned* family (the Schur complement, in which the higher harmonics sit
  on their own zeros), not harmonic by harmonic.
  *Status* ([proof/B_tail.md](proof/B_tail.md)): the weights cancel in the tuned problem (they are the connection
  coefficients of label harmonics tuned to zero; the tuning ratio is ν-independent to six digits, and neighbour-family
  amplitudes fall by 10⁴–10⁸ per pair), so the condition needs F only on the real line. Naive Galerkin truncation brings
  the weights back (σ_{M+1} wrong by 10⁴–10⁶), so the lemma must be stated for the jump-conditioned truncation. Proposed
  framework: a strip-analytic weighted ℓ¹ space X_σ (σ ≈ 1), in which the leading tuning map has norm 0.0633·e^{2σ};
  the target is a Lipschitz bound with per-pair constant about 0.05 (0.25 would suffice). Estimated tail in r₁: 4–6 above
  M = 5, 0.2–0.35 above M = 7, so certifying a 10⁻³ box needs M = 7 explicit with a margin of about 13. Prediction for
  M = 7: ω/ω∞ = 0.99997–1.00017. No evidence of secondary Stokes lines from F's complex singularities.
- **A.4 Krawczyk / interval Newton** on the 2×2 real system r₁(A, ω) = 0 (real and imaginary parts), with the tuned
  higher harmonics eliminated. The outer Jacobian at M = 5 is well conditioned: singular values about 6.8×10³ and
  3.9×10³ (condition 1.7), with second derivatives about 1.3×10⁵. A box of radius ρ in (A, ω/ω∞) is certified if the
  enclosure error of r₁ is below about 3.9×10³ ρ: 3.9 for ρ = 10⁻³, 0.39 for ρ = 10⁻⁴, 4×10⁻³ for ρ = 10⁻⁶
  (`feasibility.py`). Double precision already resolves r₁ to about 8×10⁻⁷ with the Schur correction, so precision is
  not the obstacle; the obstacle is A.3 (the M = 3 → 5 change of the zero corresponds to about 60–110 in r₁, so the
  M > 5 tail must be bounded below about 4 to certify a 10⁻³ box).

**Prototype status at M = 1** ([proof/A_certify.md](proof/A_certify.md)). The one-harmonic model's zero
(A = 2.5421480475, ω/ω∞ = 0.7769940136; Jacobian singular values 303 and 87) is the test case. Rigorous already, in ball
arithmetic: the linear normalization (the lateral jump over the closed-form Stokes constant is 1 ± 3×10⁻³⁰ at two radii
and two ω, from arb's Kummer U on either side of its cut; `lin_check.py`), the formal series of the truncated system with
(A, ω) boxes (`series_arb.py`), and the forcing bound of the start-value fixed point (the optimally truncated series has
residual 1.0×10⁻³³ on |ξ| ≤ 0.1; `residual_poly.py`). In double precision, Poincaré–Miranda holds on all four faces of the
box (`pm_dryrun.py`): half-width 10⁻³ needs the multiplier enclosed to 0.076, 10⁻⁵ to 7.8×10⁻⁴, i.e. 10⁻² to 10⁻³
relative, the accuracy the Baldomá et al. template achieved. Since proved: the start-value fixed point (`fixed_point.py`: on the rays arg ξ = ±3π/8,
|ξ| ≤ 0.1, exactly one solution in an explicit weighted ball around the N = 53 series, contraction constant ≤ 0.2918, start
values enclosed to 6×10⁻³⁴, with ∂/∂(A, ω) over a parameter box), and the Krawczyk test is built and validated on mock
enclosures (`krawczyk.py`; matching at ξ₁ = 0.3 rather than 0.2 makes the absolute targets about 15,000× easier: the
state at the centre to about 2×10⁻⁵ for a 10⁻³ box, the derivatives over the box to 10⁻²). Missing: validated integration
along the ray and arc, which is stiff
(fast modes shrink by e⁻³¹, so naive interval Taylor overestimates by about e⁵⁰; it needs a Lohner/QR integrator, e.g.
CAPD, an estimated 1–3 weeks), and Krawczyk with (A, ω) as state variables for uniqueness. At M = 5 the system is
38-dimensional, and the certificate must use the Schur-corrected conditions (leakage up to 10¹⁷), i.e. multiprecision
Lohner.

Staged targets: **A(M=5)**, a certified simple zero of the five-harmonic multiplier (lemmas A.1, A.2, A.4 only; a
standard computer-assisted proof); then **A(∞)**, the untruncated zero, adding A.3, or by the entire-function route below.

**Route A3: the entire-function certificate** ([proof/D_entire.md](proof/D_entire.md), `solver/proof/entire_cert.py`).
The coefficients follow σ_k/σ_{k+1} ≈ (k + β)/κ⋆ with κ⋆ ≈ 1.28 + 1.08i and β ≈ 4.2 − 1.6i, the same at ω/ω∞ = 0.99,
1.00, 1.01 to about 1%: S₁ is entire of order 1 and exponential type, a Laplace integral over an arc of Borel
singularities (from about 0.75 + 0.16i to κ⋆) rather than a sum of isolated exponents. With G = e^{−κ̄x}S₁ (κ̄ ≈ 0.89 + 0.55i,
the arc's centre; same zeros) the truncated series at K = 9–12 puts the real zero at ω/ω∞ = 0.99767, 1.00035, 0.99987,
0.99988, with A₀(ω∞) = 2.7194 − 0.0002i, matching the central Padé approximants (0.99987–0.99991); the plain Taylor
truncation is useless at these orders. Certificate margins at ω∞: Rouché on |x − x₀| = 0.01 needs total error below
5.7×10⁻⁴; Krawczyk/Miranda in (A, ω/ω∞) (Jacobian singular values 0.53, 0.26) needs below about 2.6×10⁻³ for a box of
radius 0.01 and 2.6×10⁻⁴ for 10⁻³. A type bound calibrated at k = 12 gives a tail of about 2×10⁻⁴, so K = 12 would certify
a 0.01 box and K ≈ 14–15 a 10⁻³ box, if the coefficients and the tail were rigorous. What that needs: rigorous σ_k for
k ≤ 14 (standard but laborious: each order is a linear inhomogeneous Stokes problem, enclosed by validated lateral
integration or an exact adjoint pairing instead of the late-order fit), and the tail bound, which is the new, hard part.
A majorant on the recursion can only give geometric bounds, so it needs either a bound on G at complex amplitude on one
circle |x| ≈ 15–20 (|A| ≈ 4–4.5) via Cauchy estimates, i.e. control of the tuned inner solution at complex amplitude, or a
Laplace/Paley–Wiener representation proving the exponential type. This replaces the harmonic tail (gap (b)) by an
amplitude tail that is much better behaved; a conditional theorem (zero certified assuming an explicit type bound) is
within reach now.

*Progress on route A3* ([proof/D_entire.md](proof/D_entire.md) §6–7, `exact_sigma.py`). The late-order fit is replaced by
an exact formula for σ₁: the order-3 recursion telescopes (base_j·T_j = 1/(Γ(a)Γ(b)) exactly), the increment is O(j⁻²)
after two exact cancellations of its O(1/j) parts, its leading part is summed in closed form (digamma / Hurwitz ζ, and two
summations by parts), and an explicit O(j⁻³) remainder bound closes it. In ball arithmetic σ₁ is now **rigorously
enclosed** to ±3.5×10⁻⁵ at ω/ω∞ = 0.99, 1.00, 1.01 (it contains the fitted values; a 10⁻³ box needs about 2×10⁻³). σ₂ to σ₆
telescope the same way (increments O(logᵏ j/j²)); they need the lower orders' large-j forms by induction, which is
mechanical but long. Enclosures uniform over an ω box need log T_j carried through lgamma. The exponential type is
characterised but not derived: |σ_k| Γ(k + 4.16)/1.675ᵏ stays within 24.9–27.3 for k = 4–12, the Borel transform has a cut
from about 0.75 + 0.16i to κ⋆ ≈ 1.28 + 1.08i with density vanishing at κ⋆, and the mechanism is not a single exponential
factor. A proof of the type, and so of the tail bound, still needs one a-priori bound at complex amplitude; writing order
2k + 1 as k-fold Borel convolutions over a simplex is the proposed route.

A cheaper intermediate certificate uses the entire-function structure: if S₁(·, ω) is entire of order 1 in x (lemma to
prove: a bound $`\lvert\sigma_k\rvert \le C\,a^k/\Gamma(k+b)`$ with explicit C, a, b), then rigorous enclosures of σ₁…σ_K
plus that tail bound give S₁ on a contour, and the argument principle counts its zeros near x₀ for each ω; Miranda's
theorem in (A, ω) then gives the zero. This replaces A.2 by a proof about the amplitude recursion, which is exact (a
convolution recursion with rational coefficients, `solver/inner_amp/amp_flint.py`); the late-order step would need a
rigorous Darboux bound instead of a fit.

A plausible structural reason for the entire-function behaviour, worth checking: the Dingle exponents depend on the
amplitude through ⟨F²⟩ (g_ν = −(½ + iνω) + ⟨F²⟩), so the normalised multiplier contains Γ-function factors of
amplitude-dependent arguments, and 1/Γ is entire of order 1. Factoring those out explicitly should leave a function with
a finite radius of convergence and faster, cleaner convergence of all the estimators.

## 5. Layer B: the analytic selection theorem

**Status for the M-harmonic model** ([proof/C_selection.md](proof/C_selection.md)). Theorem B(M): for n ≥ n₀(M), the
centre-regular solution with profile F_n and frequency ω is analytic at the light cone exactly when
C_{n,μ} = κ_{n,μ}[S_μ + O(1/n)] = 0 for every odd μ ≤ M, with κ explicit and non-zero; a zero of S that is simple once
the phase is fixed gives a unique branch with ω_n − ω* = O(1/n), and any convergent sequence of solutions converges to a
zero of S. Given the lemmas it is the finite-dimensional implicit function theorem, as in Xie–Tanveer. Lemma B2 is proved
(`solver/proof/fast_eigenvalues.py`, sympy): the outer system's fast eigenvalues are n·c²/(x(x² − c²)) for p + m (equal to
−nΦ′ at c = 1) and −n/x for b − 1, with no turning points on the real axis, regular-singular points at the centre and
the light cone, and the turning point at infinity (x ~ √n) being exactly the inner problem. Gap (c) reduces to Olver's
error control (the WKB phase and the Frobenius exponent are the same object, since x − c has a simple zero at the light
cone; the exponents of different harmonics differ by at least 2ω in imaginary part, so no resonance), with only a
routine Volterra bound left to write. The inner half of gap (a) reduces to a citation: at fixed M the inner problem is a
finite system with a rank-1 irregular point and is 1-summable by Braaksma's theorem (1992), so the multipliers are
rigorously defined. Costin's rank-one theorems do not apply as stated: the fast eigenvalue repeats across harmonics, and
the slow directions have eigenvalue 0. The outer half rests on Gevrey-1 theory without turning points plus two cheap
numerical hypotheses about where Borel singularities lie, now checked numerically for M = 1, 3, 5, 7 and untruncated
(every genuine Borel singularity at |arg| ≥ 0.48π; `outer_borel.py`). Further progress
([proof/C_selection.md](proof/C_selection.md) §8–13): in inner variables the exact equations are the inner problem plus
corrections in integer powers of 1/n only (B5.0, `inner_corrections.py`), so the matching error is O(1/n), not
O(n^{−1/2}); the change to the fast-mode variable and the non-zero matching factor κ are explicit (B5.1, reducing to the
linear C(n, s) at F = 0); the outer jump equals κ[S + O(1/n)] given the inner fixed point (B5 part 2); the continuation
domain is explicit, with progressive paths along which Re Φ never decreases; and the Volterra bound of the light-cone
lemma is written out (B6′). A correction: the inter-harmonic coupling in the light-cone exponents is O(1), not O(1/n), but
the gauge e^{Π} removes it exactly, and the exponent differences stay at least 7.27 apart in imaginary part
(`lightcone_exponents.py`), so non-resonance holds. Still to write with explicit constants: the inner fixed point on the
matching domain (B5 part 1) and the global continuation contraction (B8), both needing only fixed-order truncations.
Uniformity in M (gap (b)) is a separate question.

**Architecture (after Xie–Tanveer).** Work with the self-similar unknowns (p, m, b, c) as functions of (x, θ) analytic
in θ on a strip and in x on a complex domain that reaches the inner region x ~ √n and the light cone x = 1.

- **B.1 Formal outer solution.** Proposition 1 (proved).
- **B.2 Actual solutions near the formal one.** Borel summability of the 1/n expansion in sectors of the complex x-plane,
  for each fixed (F, ω, K_k), with the summed solution regular at the centre. This is gap (a). For finite harmonic
  truncation it is a parameter-dependent version of Costin (1998); for the full θ-dependence it needs Costin–Tanveer's
  strip-analytic framework. A specific obstruction to check: the outer nonlinear late orders also carry x-independent
  singularities of modulus about 0.13 driven by high harmonics; one must show these produce only ambiguities in the free
  data (F's harmonics, K_k), not new light-cone branches. In the linear case the analogous ±iπ terms are pure phases
  (Proposition 4), which is the model case.
- **B.3 Inner–outer matching.** The outer fast mode e^{−nΦ}A matches the inner fast mode on 1 ≪ x ≪ √n, and the outer
  late-order constant equals the inner Stokes multiplier. Proved in the linear case (Proposition 4 with the Weber
  constant); the nonlinear matching follows Baldomá–Fontich–Guardia–Seara: solve the inner problem in a domain near the
  complex singularity (here ξ = ∞, i.e. x ~ √n), the outer problem in a domain avoiding it, and bound the difference in
  the overlap.
- **B.4 Outer transport.** Proposition 3: transport of the fast mode inside the light cone is invertible (proved, as an
  ODE statement along characteristics; needs the error terms of B.2).
- **B.5 The light cone x = 1 (gap (c)).** A regular singular point; the singular Floquet branch
  $`(1-x)^\Lambda\varphi(\theta - \omega'\ln(1-x))`$ must be matched to the outer WKB amplitude (singular there, A ∝ (1 − x)⁻¹)
  on 1/n ≪ |1 − x| ≪ 1. The exponents agree on both sides (Λ = n/2 − 1 − s + ⟨F²⟩ + O(1/n)). This is a classical
  turning-point / regular-singular-point matching; Xie (arXiv:1610.00076) handles the analogous difficulty of Stokes
  lines meeting a non-analytic point at the finger tip. Moderate difficulty.
- **B.6 Non-degeneracy.** The zero of S is simple modulo the symmetries (θ shift, the λ = 0 mode; the scale/time shift,
  the λ = 1 mode). This is the output of Theorem A, plus the statement that the tuned higher harmonics depend smoothly on
  (A, ω) (the tuning conditions S_ν = 0, |ν| ≥ 3, have an invertible Jacobian; measured: condition about 2 for the
  outer 2×2 after elimination).
- **B.7 Implicit function theorem.** As in Xie–Tanveer and Xie (2016, Theorem 1.11 and its corollary): the light-cone
  obstruction is $`C_n(A, \omega) = \kappa_n\,[S(A,\omega) + O(n^{-1})]`$ with an explicit non-vanishing factor κ_n
  (the half-monodromy and transport maps of B.4–B.5), so a simple zero of S continues to a zero of C_n for large n, with
  ω_n − ω* = O(1/n).

**Gap (b), harmonic uniformity: why it is not obviously fatal, and how to formulate it.**

The worry, stated in the README: the linear multipliers grow like $`S_\nu \propto F_\nu e^{\pi\lvert\nu\rvert\omega/2}`$
(e^{6.0|ν|}) while F's harmonics decay only like e^{−1.4|ν|}, so S(θ₀) "involves F continued to Im θ ≈ ∓πω/2 ≈ ∓6",
far outside the strip |Im θ| < 1.4 where F is analytic. Three observations:

1. **The condition is diagonal.** S ≡ 0 means S_ν = 0 for every ν, and the map S ↦ C to the light cone is diagonal in
   ν with non-zero entries (B.4–B.5). Nothing requires summing $`\sum_\nu S_\nu e^{i\nu\theta_0}`$ at complex θ₀; each S_ν is
   a finite number, defined by its own late orders. So there is no divergent sum in the statement, only in naive
   estimates.
2. **The higher harmonics are tuned, not free.** The S_ν with |ν| ≥ 3 vanish by construction, so the large weights
   multiply zero; what enters r₁ is the *sensitivity* of S₁ to errors in the tuned harmonics. That sensitivity is enormous
   harmonic by harmonic (∂r₁/∂B₅ ~ 10¹⁷), but the tuning also absorbs it: in `hier.py`'s Schur-complement form the
   effective coupling is what the truncation sequence measures, and it decays about twentyfold per pair of harmonics. A
   proof should therefore estimate the *Schur complement* (r₁ with the higher levels eliminated), in a norm with weights
   $`e^{\pi\lvert\nu\rvert\omega/2}`$ on the multipliers and $`e^{-\sigma\lvert\nu\rvert}`$ on the profile, dual to each other.
3. **Each order of the amplitude expansion is a trigonometric polynomial,** so in the amplitude expansion every
   object is entire in θ and the strip problem does not arise order by order; it is moved into the convergence of the
   series in A², which numerically is that of an entire function of x. A proof route: establish S₁ as an analytic
   function of (A², ω) via the amplitude recursion (where all θ-functions are polynomials), prove the growth bound on
   σ_k, and only then continue to A² ≈ 7.4.

The formulation that avoids the issue is Tanveer's: continue the characteristics into complex θ and track where the fast
mode's label θ₀ meets the complex singularities of F; the Stokes switching then happens on characteristics that stay in
the analyticity strip, and the weights appear only in the final diagonal map to the light cone. Whether secondary Stokes
lines generated by F's own singularities (at |Im θ| ≈ 1.4) contribute at the same exponential order is the one place
where the formal argument could fail; the numerical agreement of two independent routes (truncated lateral solutions and
the untruncated amplitude expansion, which share no code) to 10⁻³ in ω suggests that they do not.

## 6. Layer C: identifying the zero with the solutions' limit

Given Theorem B, C is a corollary: the branch of critical solutions continued from finite D converges to a zero of S,
and if that zero is the unique simple zero in a box containing the numerical limit, ω∞ = ω*. Two points need care:

- **Uniqueness in the box.** Theorem A must certify that there is exactly one zero of S₁ in a box that also contains
  the numerically extrapolated ω∞ = 3.82213589 (±4×10⁻⁸) and A∞ ≈ 2.70–2.72. A box of radius 10⁻³ suffices for the
  identification; it does not need the numerics' digits.
- **That the computed branch is the branch of Theorem B.** The double-double solutions at D = 8–61 are numerical. A
  Reiterer–Trubowitz-style computer-assisted existence proof at one large D (say D = 40–60), with the certified solution
  inside the basin of Theorem B's construction, would close this; alternatively, Theorem B's construction at n ≥ n₀ with
  explicit n₀ and a certified solution at some D ≥ n₀ + 2.

The numerical agreement (ω*/ω∞ = 0.9999–1.0001) then becomes a consistency check of the theorem rather than its basis.

## 7. Staged plan

| Stage | Goal | Ingredients | Difficulty | Who |
|---|---|---|---|---|
| 1 | Write the inner problem as a first-order system at the irregular singular point ξ = ∞ (t = 1/ξ²), list the transseries, and identify S_ν with Costin's Stokes constants | Costin (1998); the existing formal series (`solver/inner_borel/`, `inner_np.py`) | low–moderate, weeks | here |
| 2 | Certified M = 5 zero (Theorem A(M=5)) | A.1 by fixed point with explicit constants; A.2 validated Taylor integration; A.4 Krawczyk; tolerances from `feasibility.py` | moderate, a computer-assisted-proof project of a few months | computer-assisted-proof community |
| 3 | Growth bound for σ_k and the entire-function structure; factor out the Γ dependence | the exact amplitude recursion; rigorous Darboux bounds for its late orders | moderate–hard | exponential asymptotics |
| 4 | Harmonic tail bound A.3, then Theorem A(∞) | weighted dual norms on the tuned family; Schur complement estimates | hard, genuinely new | both communities |
| 5 | Theorem B for a finite-harmonic model (θ-dependence truncated in the full equations, not only the inner problem) | Xie–Tanveer architecture; Costin; B.3–B.7 | hard, a full paper | exponential asymptotics |
| 6 | Theorem B in the strip-analytic setting | Costin–Tanveer PDE Borel summability; gap (b) | research-level | exponential asymptotics |
| 7 | Corollary C, with a certified large-D solution | Reiterer–Trubowitz method at one D | moderate given 2–6 | GR / computer-assisted proofs |

Stages 1 and 2 are achievable with current tools and would already be a publishable result: a computer-assisted
certificate that the five-harmonic inner problem selects ω/ω∞ = 0.9992 ± 10⁻⁶, together with the measured convergence
in M and the untruncated bracket. Stage 4 is where a new idea is needed.

## 8. Who works on this

- **Exponential asymptotics and selection problems:** S. Tanveer and O. Costin (Ohio State; Borel summability, rigorous
  selection, PDE Borel summability), X. Xie (Morgan State; rigorous finger and needle-crystal selection), S. J. Chapman
  (Oxford; exponential asymptotics for nonlinear and PDE problems), J. R. King (Nottingham), A. B. Olde Daalhuis
  (Edinburgh; resurgence, hyperasymptotics), C. J. Lustri (Sydney/Macquarie; exponential asymptotics in PDEs).
- **Computer-assisted proofs of Stokes constants and invariant objects:** I. Baldomá, M. Guardia, T. M. Seara (UPC
  Barcelona), M. J. Capiński (AGH Kraków); J. Gómez-Serrano (Brown; computer-assisted proofs in PDE); J.-P. Lessard,
  J. B. van den Berg, J. D. Mireles James (validated numerics for ODE/PDE).
- **Critical collapse and self-similar blowup:** C. Gundlach (Southampton), R. Donninger and I. Glogić (Vienna),
  B. Schörkhuber (Innsbruck), M. Reiterer and E. Trubowitz (computer-assisted existence of Choptuik's solution),
  and the large-D critical-collapse authors already contacted (Ecker, Grumiller and coauthors).

A natural first contact for layer A is the Barcelona–Kraków group (their 2022 paper is the closest template); for layer
B, Tanveer or Costin.

## References

Citations confirmed against arXiv abstracts during this write-up carry their arXiv number; those marked *(from memory)*
should be checked.

- O. Costin, *On Borel summation and Stokes phenomena for rank-1 nonlinear systems of ordinary differential equations*,
  Duke Math. J. 93 (1998) 289–344 *(journal details from memory)*, [arXiv:math/0608408](https://arxiv.org/abs/math/0608408);
  see also [arXiv:math/0608414](https://arxiv.org/abs/math/0608414).
- O. Costin, *Asymptotics and Borel Summability*, Chapman & Hall/CRC (2008) *(from memory)*.
- O. Costin, S. Tanveer, *Nonlinear evolution PDEs in R⁺ × C^d: existence and uniqueness of solutions, asymptotic and
  Borel summability properties*, Ann. Inst. H. Poincaré Anal. Non Linéaire 24 (2007) 795–823 *(volume and pages from
  memory)*.
- O. Costin, S. Tanveer, *Borel summability of Navier–Stokes equation in R³ and small time existence*,
  [arXiv:math/0612063](https://arxiv.org/abs/math/0612063).
- M. D. Kruskal, H. Segur, *Asymptotics beyond all orders in a model of crystal growth*, Stud. Appl. Math. 85 (1991)
  129–181 *(details from memory)*.
- R. Combescot, T. Dombre, V. Hakim, Y. Pomeau, A. Pumir, PRL 56 (1986) 2036; D. C. Hong, J. S. Langer, PRL 56 (1986)
  2032; S. Tanveer, Phys. Fluids 30 (1987) 1589 *(from memory)*.
- X. Xie, S. Tanveer, *Rigorous results in steady finger selection in viscous fingering*, Arch. Ration. Mech. Anal. 166
  (2003) 219–286 *(from memory)*.
- S. Tanveer, X. Xie, *Analyticity and nonexistence of classical steady Hele-Shaw fingers*, Comm. Pure Appl. Math. 56
  (2003) 353–402 *(from memory)*.
- X. Xie, *Rigorous results in selection of steady needle crystals*, J. Differential Equations 197 (2004) 349–426
  *(from memory)*.
- X. Xie, *Rigorous results in existence and selection of Saffman–Taylor fingers by kinetic undercooling*,
  [arXiv:1610.00076](https://arxiv.org/abs/1610.00076).
- S. J. Chapman, J. R. King, K. L. Adams, *Exponential asymptotics and Stokes lines in nonlinear ordinary differential
  equations*, Proc. R. Soc. Lond. A 454 (1998) 2733–2755 *(details from memory)*.
- I. Baldomá, E. Fontich, M. Guardia, T. M. Seara, *Exponentially small splitting of separatrices beyond Melnikov
  analysis: rigorous results*, [arXiv:1201.5152](https://arxiv.org/abs/1201.5152).
- I. Baldomá, M. J. Capiński, M. Guardia, T. M. Seara, *Breakdown of heteroclinic connections in the analytic Hopf-zero
  singularity: rigorous computation of the Stokes constant*, [arXiv:2206.13840](https://arxiv.org/abs/2206.13840);
  J. Nonlinear Sci. (2023), doi:10.1007/s00332-022-09882-x.
- E. Lombardi, *Oscillatory Integrals and Phenomena Beyond all Algebraic Orders*, Lecture Notes in Math. 1741, Springer
  (2000) *(from memory)*.
- M. Reiterer, E. Trubowitz, *Choptuik's critical spacetime exists*, [arXiv:1203.3766](https://arxiv.org/abs/1203.3766).
- C. Gundlach, *The Choptuik spacetime as an eigenvalue problem*, PRL 75 (1995) 3214,
  [gr-qc/9507054](https://arxiv.org/abs/gr-qc/9507054) *(journal details from memory)*.
- A. Chatzikaleas, R. Donninger, I. Glogić, *On blowup of co-rotational wave maps in odd space dimensions*,
  [arXiv:1701.05082](https://arxiv.org/abs/1701.05082).

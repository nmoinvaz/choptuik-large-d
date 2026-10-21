# Layer A prototype: certifying the inner zero, one harmonic first

This is a prototype of layer A of [PROOF.md](../PROOF.md): a computer-assisted proof that a truncated inner Stokes
multiplier has a simple zero. It starts with the smallest real instance, the one-harmonic model (M = 1, F = A sin θ),
whose multiplier vanishes at A = 2.542148, ω = 2.969777 (0.776994 ω∞). That is not the physical point; it is the test
case. The code is in [`solver/proof/`](../solver/proof/): `certify_M1.py` (driver), `lin_check.py`, `series_arb.py`,
`residual_poly.py` (rigorous, python-flint ball arithmetic) and `pm_dryrun.py` (double precision, not rigorous).

**Status in one paragraph.** Two of the four ingredients are certified: the normalization of the multiplier (exactly, in
the linear case) and the forcing term of the fixed-point lemma that encloses the lateral solution at its start point
(a rigorous bound of 10⁻³³). Two are missing: the contraction constant of that lemma, and validated integration of the
lateral solution from the start point to the real axis. A non-rigorous dry run shows how much accuracy the certificate
needs: to certify the zero in a box of half-width 10⁻⁵ in (A, ω/ω∞), the multiplier must be enclosed to 8×10⁻⁴; for
10⁻³, to 0.08. These are modest targets. The binding difficulty is the stiffness (wrapping effect) of the integration,
not precision.

Contents

- [1. The template](#1-the-template)
- [2. The M = 1 inner problem](#2-the-m--1-inner-problem)
- [3. The four steps](#3-the-four-steps)
- [4. What is certified](#4-what-is-certified)
- [5. What the certificate must reach (dry run)](#5-what-the-certificate-must-reach-dry-run)
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
  - a Lipschitz constant of 𝒢 composed with the inverse of L_lin below 1/2, which is not yet done.
- **(ii) Carry the enclosure to the real axis.** Validated integration of the 14-dimensional system from ξ₀ along the
  ray to |ξ| = r₁ and along the arc |ξ| = r₁ to the real axis (r₀ = 0.1, r₁ = 0.2 in the double-precision code).
- **(iii) The jump.** $`J_1 = y^+_1 - \overline{y^+_{-1}}`$ comes from the same enclosure; there is no second
  integration. Its normalization is certified in the linear case (§4).
- **(iv) The zero.** For existence, apply Poincaré–Miranda to the preconditioned map H = J₀⁻¹G on a box: H₁ must have
  opposite signs on the two A-faces and H₂ on the two ω-faces. Only enclosures of G on the faces are needed, not
  derivatives. For uniqueness and simplicity, apply Krawczyk with an interval Jacobian over the box, obtained by
  carrying (A, ω) as extra state variables through steps (i)–(ii) or by integrating the variational equations.

## 4. What is certified

All three items below are rigorous ball-arithmetic computations (python-flint 0.9, arb/acb), at 256–320 bits.

**Linear normalization (`lin_check.py`).** In the linear problem (c = 1), harmonic μ has the lateral solutions
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

**Formal series in ball arithmetic (`series_arb.py`).** This is a port of `lateral.py`'s recursion to acb, for the same
truncated system. A and ω may be balls, so the coefficients enclose the whole parameter box. At the M = 1 zero, at
r₀ = 0.1, 0.08 and 0.06 on the start ray, optimal truncation is at N = 53, 80 and 144 terms, with smallest terms
9×10⁻³⁵, 2×10⁻⁴⁷ and 10⁻⁷⁴.

**The forcing M₀ (`residual_poly.py`).** The residual of the truncated series in the three ODEs, computed as an exact
ball polynomial in ξ. At N = 53 its coefficients are zero up to ξ¹⁰⁶ (balls containing 0 that come from rounding) and
nonzero from there on, and for all |ξ| ≤ 0.1 in every direction:

```math
\lvert R_N(\xi)\rvert \le \sum_k \lvert\rho_k\rvert\,\lvert\xi\rvert^k \le 1.0\times10^{-33} .
```

The largest component is the σ equation (I2); the y and c equations are at 4×10⁻³⁶. This is the M₀ of step (i). It is
so small that even a crude contraction constant yields an enclosure of y⁺(ξ₀) far tighter than step (iv) needs.

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
to 10⁻³. That matches what the template achieved for its Stokes constant.

## 6. What is missing, and how to build it

1. **The contraction constant (step i).** It needs explicit bounds on the integral operator inverting L_lin on the
   sector |ξ| ≤ r₀, arg ξ ∈ [−δ, 3π/8 + δ], in the weighted norm. Per harmonic the fundamental matrix is explicit
   (Kummer U and M). The bounds can come from the large-|x| asymptotics of U with rigorous remainder (DLMF 13.7.10–14
   gives explicit error bounds for the U expansion) or from arb evaluations on a covering of the domain. 𝒢 is
   polynomial with coefficients O(ξ²) near 0, so its Lipschitz constant on a ball of radius ~M₀ is about
   sup|c − 1| + sup|y|² ≈ F²r₀² + 4A²r₀² ≈ 0.3 at r₀ = 0.1 (an estimate, not a bound yet). Shrinking r₀ makes the
   contraction easy but raises N (144 terms at r₀ = 0.06) and lengthens step (ii). r₀ ≈ 0.08–0.1 looks right.
2. **Validated integration (step ii).** This is the binding difficulty, because the system is stiff.
   - **Along the ray** arg ξ² = 3π/4 from 0.1 to 0.2, the fast modes shrink by about e⁻³¹. Along the arc they go from
     e⁺⁹ to e⁻¹² relative to the slow ones.
   - **Naive intervals fail.** A naive interval Taylor method overestimates by exp(∫|λ|) instead of exp(∫Re λ), about
     e⁵⁰ on the ray.
   - **What works:** a Lohner-type method that propagates the set as c + Φr, with Φ the enclosed flow Jacobian (QR
     re-orthogonalization), captures the contraction.
   - **Options:** CAPD's C¹ Lohner integrator on the 14-dimensional real system in double or long-double precision
     (enough: §5 needs only 10⁻³ relative), or a small arb Taylor–Lohner integrator (python-flint acb_poly
     arithmetic; polynomial right-hand side, ξ(t) linear on the ray and r₁e^{it} on the arc).
   - **Step-size guide:** the double-precision DOP853 path needs about 10³ steps at tolerance 10⁻¹²; a Taylor method of
     order 20–30 needs far fewer.
3. **Parameter boxes.** For Poincaré–Miranda, run (i)–(ii) on about 4 × 10 sub-boxes of the faces, with (A, ω) as balls
   (`series_arb.py` already accepts balls). For Krawczyk, add (A, ω) as constant state variables so the Lohner
   Jacobian contains ∂/∂(A, ω).

Rough effort: (1) is a few pages of estimates plus code; (2) is the main work, about 1–2 weeks with CAPD or about 2–3
weeks for a custom arb Lohner. The run itself is small: each G evaluation is about 1.5 s in double precision.

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
python3 certify_M1.py          # items 1-2 of section 4 (rigorous), about 1 min
python3 certify_M1.py --pm     # plus the Poincare-Miranda dry run of section 5 (double precision), about 2 min
python3 residual_poly.py 0.08  # the forcing bound at another start radius
```

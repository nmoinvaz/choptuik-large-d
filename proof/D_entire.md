# D. The untruncated route to layer A: the multiplier as an entire function of the amplitude

Part of the proof roadmap ([PROOF.md](../PROOF.md), layer A). Question: can the root of the untruncated harmonic-1
multiplier, S₁(A, ω) = 0 near A ≈ 2.7196, ω/ω∞ ≈ 0.9999, be certified without summing the amplitude series to order
≈ 75? Numbers below come from `solver/proof/entire_cert.py` (runs in seconds) on `solver/inner_amp/scan25phys/`
(σ₀–σ₁₂ at ω/ω∞ = 0.99, 1.00, 1.01; fit errors ~10⁻¹⁹). Nothing here is rigorous yet; section 4 says what would make
it so.

## Summary

- **S₁ is (numerically) an entire function of exponential type in x = A².** The ratios obey
  σ_k/σ_{k+1} ≈ (k + β)/κ⋆ with constant increments 1/κ⋆ ≈ 0.462 − 0.384i for k ≥ 8, so
  κ⋆ ≈ 1.28 + 1.08i: order 1, type |κ⋆| ≈ 1.67, the same at all three ω to 1%.
- **Its Borel transform has a continuous singular support (an arc), not isolated exponents.** Prony fits of
  k!σ_k as sums of exponentials do not settle: the exponents spread along an arc from about 0.75 + 0.16i to
  κ⋆ ≈ 1.28 + 1.08i. So S₁(x) behaves like a Laplace integral of a density over that arc.
- **Factoring out the arc's centre removes the cancellation.** With G = e^{−κ̄x}S₁ (same zeros), κ̄ ≈ 0.89 + 0.55i,
  the Taylor terms of G at x₀ are O(1) and fall to 3.6×10⁻⁴ by k = 12, instead of terms ≈ 240 that cancel. The
  truncated G_K locates the zero well: at K = 11 and 12 the zero is real at ω/ω∞ = 0.99987 and 0.99988, with
  A₀(ω∞) = 2.7194 − 0.0002i. That is a third, independent resummation, and it agrees with the central Padé
  approximants (0.99987–0.99991).
- **The "second zero at |x| ≈ 8–9" in PROOF.md is not confirmed.** The other zeros of G_K lie at |x| > 10 and move
  outward with K (10.1, 10.9, 11.4 for K = 10, 11, 12), so x₀ ≈ 7.40 is well isolated. The slow convergence of the
  log-coefficient estimates comes from the arc-shaped Borel singularity, not from a nearby zero.
- **Certificate margins.** In G form the outer 2×2 Jacobian in (A, ω/ω∞) has singular values 0.53 and 0.26. A box of
  radius 0.01 needs the total error of G below ≈ 2.6×10⁻³; radius 10⁻³ needs ≈ 2.6×10⁻⁴. A type bound calibrated at
  k = 12 puts the K = 12 tail at ≈ 2×10⁻⁴. So **K = 12 would certify a box of radius 0.01, and K ≈ 14–15 a box of
  10⁻³, if the coefficients and the tail were rigorous.**
- **Verdict.** The approach is sound and much cheaper than direct summation (K ≈ 14 instead of 75). The obstacle is
  the tail: a recursion majorant cannot see the 1/k! decay, so the a priori bound has to come from a representation of
  S₁ at complex amplitude (section 4). That is new mathematics, but well defined.

## 1. Why entire, and with what type

The ratio law, per ω (complex linear fit for k = 6–11; increments shown for k = 7–11):

| ω/ω∞ | ratio increments (→ 1/κ⋆) | κ⋆ | type ∣κ⋆∣ | β |
|---|---|---|---|---|
| 0.99 | 0.420−0.394i … 0.459−0.384i | 1.271 + 1.081i | 1.668 | 4.10 − 1.56i |
| 1.00 | 0.414−0.389i … 0.461−0.384i | 1.279 + 1.081i | 1.675 | 4.16 − 1.60i |
| 1.01 | 0.406−0.384i … 0.462−0.384i | 1.289 + 1.080i | 1.682 | 4.23 − 1.64i |

A constant increment means k!σ_k grows like κ⋆^k times a power, so |σ_k| ≈ C|κ⋆|^k/Γ(k + β): S₁ is entire of order
1 and type ≈ 1.67 in x (≈ 6.7 in h = ⟨F²⟩/2 ≈ x/4). By the Pólya representation, an entire function of exponential type is a
Laplace transform of its Borel transform B(t) = Σ k!σ_k t^{−k−1} around the conjugate indicator diagram, whose
singular support here is the arc found by Prony (m = 3, 4, 5 terms: residuals 8×10⁻⁴, 1×10⁻⁵, 3×10⁻⁸; the exponents
shift and multiply as m grows, which is the signature of a continuous density, not of a finite sum).

Where the type comes from is not settled. In the recursion, the amplitude enters products polynomially, which by
itself would give a finite radius, not 1/k!. The 1/k! must come from the amplitude entering exponents: the fast
exponents g_ν = −(½ + iνω) + ⟨F²⟩, the normalisation ξ^{2⟨F²⟩}, and the de-mixing factor e^{Π}, with Π ∝ x. A factor
e^{x φ} integrated over a bounded complex path gives exactly an exponential type with a continuous Borel density along
the image of the path. Identifying the path is the first analytic task (section 4).

## 2. The resummation

G = e^{−κ̄x}S₁ shifts the Borel transform by κ̄, so G's type is the distance from κ̄ to the farthest point of the arc
(≈ 0.66 for κ̄ = 0.893 + 0.550i, against 1.67 for S₁). κ̄ is chosen per K to minimise the last two terms of G_K at x₀;
the zeros do not depend on κ̄ in the limit, only the rate.

| K | κ̄ | last terms at x₀ | A₀(ω∞) | real zero at ω/ω∞ |
|---|---|---|---|---|
| 9 | 0.944 + 0.503i | 8.1×10⁻³ | 2.71494 − 0.00284i | 0.99767 |
| 10 | 1.017 + 0.464i | 2.9×10⁻³ | 2.72235 + 0.00054i | 1.00035 |
| 11 | 0.951 + 0.507i | 1.0×10⁻³ | 2.71986 − 0.00019i | 0.99987 |
| 12 | 0.893 + 0.550i | 3.6×10⁻⁴ | 2.71943 − 0.00017i | 0.99988 |

The plain Taylor truncation (κ̄ = 0) is useless at these K: its zero drifts from A = 1.99 (K = 6) to 2.53 (K = 12).
Terms of G₁₂ at x₀: 1.0, 1.1, 0.83, 0.76, 0.40, 0.23, 0.12, 0.038, 0.024, 3.1×10⁻³, 3.4×10⁻³, (≈0), 3.6×10⁻⁴.

## 3. The certificate, assuming rigorous coefficients and a tail bound

Two equivalent formulations.

- **Rouché in x at fixed ω.** On the circle |x − x₀| = ρ, the minimum of |G₁₂| is 5.7×10⁻⁵ (ρ = 0.001), 5.7×10⁻⁴
  (ρ = 0.01) and 5.6×10⁻³ (ρ = 0.1). If the coefficient errors plus the tail are below that on the circle, G has exactly one
  zero inside it.
- **Krawczyk or Miranda in (A, ω/ω∞)** for (Re G, Im G) = 0. The Jacobian at the zero, from the three ω scans with a
  common κ̄, is

```math
J = \begin{pmatrix} -0.290 & 0.373 \\ -0.112 & -0.331 \end{pmatrix},\qquad s_{\max} = 0.527,\quad s_{\min} = 0.262 ,
```

  so a box of radius r needs a total enclosure error of G below about 0.26 r: 2.6×10⁻³ for r = 0.01 and 2.6×10⁻⁴ for
  r = 10⁻³. The curvature term is negligible at these radii. The ω-direction needs σ_k at ω values bracketing the box
  (here 0.99 and 1.01), with the ω-dependence of each σ_k bounded between them (e.g. by an enclosure of ∂σ_k/∂ω).

The tail. With a type bound |g_k| ≤ C r^k/k! (r the distance from κ̄ to the arc's far end, ≈ 0.66) and C fixed by
the last computed coefficient, the K = 12 tail at x₀ is ≈ 2×10⁻⁴ (1.4–2.8×10⁻⁴ for r = 0.5–0.8). That fits
r = 0.01 with a factor 13 to spare, and r = 10⁻³ only marginally; K = 14–15 would fit r = 10⁻³ comfortably. A certified
box of radius 10⁻³ in ω/ω∞ around 0.9999 contains ω∞, but containing it is not proving equality: layer C still needs
the analytic selection theorem.

## 4. What would make it rigorous

1. **Rigorous σ_k, k ≤ K (standard, laborious).** At each order in ε the inner equations are linear in the new
   unknowns, with forcing from lower orders, and the tuning of f_ν is linear too. So each σ_k is a Stokes constant of a
   linear inhomogeneous ODE system with an irregular singular point. It can be enclosed either by validated lateral
   integration, as `solver/inner_stokes/lateral_np.py` does in floating point (two solutions on either side of the
   Stokes line, differenced, divided by the fast mode), or by the exact pairing formula S = ∫_γ W f with W the recessive
   adjoint solution. Either replaces the late-order fit, which is accurate (J = 750 and 1000 agree to 10⁻¹⁹) but not an
   enclosure. This is the Baldomá–Capiński–Guardia–Seara pattern applied order by order; K ≈ 14 orders and |ν| ≤ K
   harmonics.
2. **The tail bound (the hard part).** Two ways:
   - **Cauchy estimates on one circle.** If |G(x)| ≤ M for |x| = R, then |g_k| ≤ M/R^k for every k and the tail at x₀
     is at most M (x₀/R)^{K+1}/(1 − x₀/R). With the observed type, R ≈ 15–20 and K = 12 give ≈ 2×10⁻³; K = 14 gives
     ≈ 2×10⁻⁴. This needs a rigorous bound on the multiplier at complex amplitude |A| ≈ 4–4.5. For that, the tuned family
     and the inner solution must exist there. Whether they do is a nonlinear existence question; the observed type says
     yes for the series, but that is evidence, not proof.
   - **A representation proof of exponential type.** Find the path behind the arc (section 1) and write S₁(x) as a
     Laplace-type integral of e^{xφ} times a density analytic in x with at most polynomial growth. Paley–Wiener–Pólya
     then gives |g_k| ≤ C r^k/k! with explicit C, r. A Grönwall estimate along a complex path of the inner ODE at
     complex amplitude, in which x enters the coefficients linearly over a path of bounded log-length, would give
     |S₁(x)| ≤ C e^{τ|x|}. Again it needs existence of the solution along the path at complex amplitude.

   A majorant argument on the amplitude recursion alone cannot give 1/k!. Majorants give geometric bounds C ρ^{−k},
   which do not even converge at x₀ for the ρ the convolutions allow. Also, σ_k is a j → ∞ limit (a Stokes constant),
   which finite-j majorants do not control.
3. **Uniform-in-ω control** between the scanned ω values: enclosures of ∂σ_k/∂ω, or σ_k on a finer validated grid.

## 5. Verdict

- **Feasible now, modulo engineering:** rigorous σ₀–σ₁₄ by order-by-order validated lateral integration (linear
  problems), plus the Rouché and Krawczyk tests in G form. This would certify the zero *given* a stated tail
  hypothesis (|g_k| ≤ C r^k/k! with explicit constants, checked against the computed orders). That is a conditional
  theorem, but a precise one.
- **Genuinely new and needed for an unconditional result:** the exponential-type bound, i.e. control of the inner
  problem at complex amplitude. It plays the role that the harmonic tail plays in the truncated route (PROOF.md, gap
  (b)). The entire-function structure turns that tail, in harmonics, into a tail in amplitude that is far better
  behaved: geometric decay ≈ 0.3 per order in G against a non-uniform ν-tail. It is the more promising of the two.
- **Immediate cheap gains:** use G = e^{−κ̄x}S₁ as the standard estimator of the zero (it agrees with the central Padé
  approximants and needs no ω-interpolation of arguments), and identify the path behind the Borel arc, which is
  analysis on known formulas and the natural first step toward the representation proof.

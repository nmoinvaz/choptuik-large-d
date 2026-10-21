# D. The untruncated route to layer A: the multiplier as an entire function of the amplitude

Part of the proof roadmap ([PROOF.md](../PROOF.md), layer A). Question: can the root of the untruncated harmonic-1
multiplier, S₁(A, ω) = 0 near A ≈ 2.7196, ω/ω∞ ≈ 0.9999, be certified without summing the amplitude series to order
≈ 75? Numbers below come from `solver/proof/entire_cert.py` (runs in seconds) on `solver/inner_amp/scan25phys/`
(σ₀–σ₁₂ at ω/ω∞ = 0.99, 1.00, 1.01; fit errors ~10⁻¹⁹). Sections 1–5 are not rigorous; section 4 says what would make
them so. Section 6 replaces the late-order fit by an exact telescoping formula and encloses σ₁ rigorously. Section 7
studies the Borel arc behind the exponential type.

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

## 6. Exact σ_k without late-order fits: Darboux telescoping, and a rigorous σ₁

Code: `solver/proof/exact_sigma.py` (orders 1–3, all in arb balls; J = 1600 runs in 13 s). This section replaces the
late-order fit for σ₁ by an enclosure, and sets out how the same scheme reaches σ₂–σ₆. It covers item 1 of section 4
for σ₁ only.

### 6.1 The exact representation

Use the amplitude recursion of `amp_flint.py` in the scaled variables $`\tilde d_j = d_j/2^j`$, for harmonic +1. Write
$`s = i\omega`$, $`a = (1+s)/2`$, $`b = (2+s)/2`$, and

```math
a_j=\frac{(j+a)(j+b)}{j+1},\qquad T_j=\frac{(a)_j(b)_j}{j!}=\prod_{m\lt j}a_m,\qquad c_j=\frac{2j+2+s}{4(j+1)},\qquad K=\frac{1}{\Gamma(a)\Gamma(b)} .
```

Order 1 is $`\tilde d^{(1)}_j=d_0T_j`$ with $`d_0=-i`$. The −1 harmonic is the complex conjugate: $`T^-_j=\overline{T_j}`$
for real ω. At order 3 the two s-steps of the recursion combine into one first-order recursion,

```math
\tilde d^{(3)}_{j+1}=a_j\,\tilde d^{(3)}_j-b_j,\qquad b_j=c_j\,\mathcal S^s_j+\mathcal S^d_{j+1},\qquad
\mathcal S^s_j=\sum_{(\nu,\mu)}\sum_{i=1}^{j}g^{[\nu]}_i\,\tilde s^{(1)}_{j-i}[\mu],\quad
\mathcal S^d_{j+1}=\sum_{(\nu,\mu)}\sum_{i=1}^{j+1}g^{[\nu]}_i\,\tilde d^{(1)}_{j+1-i}[\mu].
```

Here the channels are $`(\nu,\mu)\in\{(0,+1),(2,-1)\}`$, $`\tilde s^{(1)}_l=(2l+1+s_\mu)\tilde d^{(1)}_l`$, and
$`g^{[\nu]}_m`$ is the order-2 lapse coefficient, $`\tfrac{1}{8m}`$ times a convolution of two order-1 sequences. The
recursion solves exactly in Darboux form, as the brief suggested: $`\tilde d^{(3)}_j=-T_j\sum_{i\lt j}b_i/T_{i+1}`$
(here $`d_0=0`$ at order 3).

The multiplier sequence of `AmpPar._ys` is $`R_j=\mathrm{base}_j\,(\tilde d^{(3)}_j+E_2\tilde d^{(1)}_j[-1])+h_2\,(\psi(a+j)+\psi(b+j))\,d_0K`$,
with $`E_2=1/(8i\omega)`$ and $`h_2=1/4`$. Since $`\mathrm{base}_j\,T_j=K`$ exactly, it becomes

```math
\frac{R_j}{K}=-\sum_{i\lt j}\frac{b_i}{T_{i+1}}+E_2\,d_0^-\,\rho_j+h_2\,d_0\,\bigl(\psi(a+j)+\psi(b+j)\bigr),\qquad \rho_j=\frac{T^-_j}{T_j},\quad |\rho_j|=1 .
```

So $`S^{(3)}=R_J+\sum_{j\ge J}\Delta R_j`$ exactly, where $`\Delta R_j=R_{j+1}-R_j`$ is an explicit finite expression.
At order 1, $`R^{(1)}_j=d_0K`$ for every j, so $`S^{(1)}=d_0K`$ and $`\sigma_1=S^{(3)}/(d_0K)`$.

The script's recursion is an implementation independent of `amp_flint`. It reproduces AmpPar's $`R_j`$ to $`10^{-39}`$
for j ≤ 400.

### 6.2 The cancellation lemma: the increment is O(j⁻²)

Split each convolution into near-end terms (i = 1, 2), far-end terms (i near j), and the interior. Write every ratio
$`T^{\pm}_{j-k}/T_{j+1}`$ as a rational function of j, times $`\rho_j`$ for the −1 family. The increment then
decomposes exactly as

```math
\frac{\Delta R_j}{K}=F(j)+\rho_j\,G(j)+\frac{r_j}{K},
```

where F and G are explicit rational functions, of degree 3 over degree 5 (built symbolically in `_sym_tail`):

```math
j^2F(j)\to\frac{i\,(7-\omega^2-2i\omega)}{8},\qquad j^2G(j)\to\frac{i\,(7-\omega^2+5i\omega)}{16} .
```

They collect five kinds of term:
- the i = 1 and i = 2 terms of both convolutions;
- the far-end term i = j + 1 of $`\mathcal S^d`$, at l = 0 and l = j of the inner convolution of $`g_{j+1}`$;
- the de-mixing term $`E_2d_0^-(\rho_{j+1}-\rho_j)`$;
- the ψ-normalisation term $`h_2d_0\,(1/(a+j)+1/(b+j))`$.

The O(1/j) parts cancel exactly in two pairs:
- In the +1 family, the i = 1 terms of both convolutions, −1/(4j) each, cancel the ψ normalisation, +1/(2j).
- In the −1 family, the i = 1 terms, +1/(8j) each, cancel the de-mixing, −1/(4j), through
  $`a^-_j-a_j=-i\omega(2j+\tfrac32)/(j+1)`$.

Hence $`|\Delta R_j|=O(j^{-2})`$. This is why $`R_J`$ converges at all, at rate 1/J: an O(1/j) increment would make
it drift like log J. Because
$`\rho_j\sim j^{-2i\omega}`$ rotates, $`j^2|\Delta R_j/K|`$ oscillates. The measured values are 1.92, 2.48, 2.56,
2.05 and 1.04 at j = 50, 100, 200, 400 and 800. They lie inside the predicted range $`|F_\infty|\pm|G_\infty|`$ = 0.07 to 2.63.

### 6.3 The remainder bound (log-convexity)

Let $`t_l=|T_l|=\prod_{m\lt l}r_m`$ with $`r_m=|a_m|`$. Two facts drive the bound.

**$`r_m`$ is increasing for m ≥ 1.** It is not increasing from m = 0: at ω∞, $`r_0=4.26 > r_1=3.36`$. For
$`m\le\lfloor\omega^2/2\rfloor+2`$ this is checked in arb. Beyond that range,
$`r_{m+1}\ge m+\tfrac32 > m+1+\omega^2/(4(m+1))\ge r_m`$. Hence $`t_l`$ is log-convex on l ≥ 1, and for every n
the product $`t_lt_{n-l}`$ decreases from both ends toward the middle on $`1\le l\le n-1`$.

**Two consequences.**
- For all i, $`|g^{[\nu]}_i|\le n_\nu(2+t_1)\,t_{i-1}/(8i)`$, where $`n_0=2`$ and $`n_2=1`$.
- Every term left out of F and G is O(j⁻³). This covers the i ≥ 3 near-end terms, the interior (bounded by its count
  times its largest term), the far-end terms with i ≤ j, and the non-leading part of $`g_{j+1}`$.

The bound uses $`t_{j-k}/t_{j+1}\le\prod_{q=0}^{k}(j-q+\tfrac12)^{-1}`$, which follows from $`r_m\ge m+\tfrac12`$.
That makes $`j^3B(j)`$ a rational function of x = 1/j, and its supremum over j ≥ J comes from one interval evaluation
on x ∈ [0, 1/J]:

```math
\sup_{j\ge J}j^3\,\frac{|r_j|}{|K|}\le C_3=110.3\ (\omega_\infty,\ J=800),\qquad \sum_{j\ge J}\frac{|r_j|}{|K|}\le\frac{C_3}{2(J-1)^2} .
```

$`C_3`$ is conservative by a factor of about 20: the measured $`j^3|r_j/K|`$ lies between 0.7 and 6 at the sampled
j ≤ 800.

### 6.4 The tail sums in closed form

- **The F sum.** Take partial fractions in j. The simple-pole residues sum to zero, which is checked symbolically and
  is the cancellation of 6.2. So $`\sum_{j\ge J}F(j)=-\sum_pc_p\,(\psi(J-p)-\psi(J))+\sum_{p,k\ge2}c_{p,k}\,\zeta(k,J-p)`$,
  which is exact in arb.
- **The ρG sum.** Here $`\rho_{j+1}-\rho_j=\varphi(j)\rho_j`$, with $`\varphi(j)=-i\omega(2j+\tfrac32)/((j+a)(j+b))`$.
  Summation by parts, $`\sum_{j\ge J}\rho_jG_j=-\rho_JQ_J+\sum_{j\ge J+1}\rho_j(Q_{j-1}-Q_j)`$ with $`Q=G/\varphi`$,
  gives a new summand of the same form, smaller by about $`1/(2\omega)`$. After two passes,
  $`\sum_{j\ge J+2}|G^{(2)}_j|\le C_2/(J+1)`$ with $`C_2=0.025`$, again from an interval evaluation in x. A third pass
  brings the constant down to 0.0038, but costs a minute of symbolic algebra.

### 6.5 Result: σ₁ is now rigorous

Each enclosure below holds componentwise, mid ± half-width. It contains every rounding error, the j⁻³ remainder and
the SBP remainder:

| ω/ω∞ | J | Re σ₁ | Im σ₁ | half-width | fitted σ₁ (scan25phys, J = 750) | fitted − mid |
|---|---|---|---|---|---|---|
| 1.00 | 200 | 0.743564001 | 0.524462563 | 1.55×10⁻³ | 0.743584580 + 0.524474853i | 2.4×10⁻⁵ |
| 1.00 | 800 | 0.743582678 | 0.524478602 | 1.15×10⁻⁴ | (same) | 4.2×10⁻⁶ |
| 1.00 | 1600 | 0.743582522 | 0.524475851 | 3.54×10⁻⁵ | (same) | 2.3×10⁻⁶ |
| 0.99 | 1600 | 0.736457587 | 0.528917680 | 3.46×10⁻⁵ | 0.736459965 + 0.528917547i | 2.4×10⁻⁶ |
| 1.01 | 1600 | 0.750661133 | 0.520061240 | 3.62×10⁻⁵ | 0.750662501 + 0.520059649i | 2.1×10⁻⁶ |

The fitted values lie inside every enclosure. That is the validation the brief asked for: the fit (error ~10⁻¹⁹) and the
exact formula agree to the width of the enclosure.

The certificate of section 3 at radius 10⁻³ allows roughly $`|\delta\sigma_k|\le 0.016/x_0^{k}`$ per order, about 2×10⁻³
for σ₁. The J = 1600 enclosure is 60 times inside that.

`python3 solver/proof/exact_sigma.py J wf` runs it, and `--check` prints the decomposition test. Cost is O(J²):
J = 4000 takes about a minute and gives about ±1×10⁻⁵. The two-pass summation-by-parts remainder then dominates, and a
third pass removes it.

### 6.6 What is and is not rigorous; σ₂–σ₆

**Rigorous now.** σ₁ at each of ω/ω∞ = 0.99, 1.00 and 1.01, as point values of ω. The whole chain is ball arithmetic
plus three ingredients:
- the exact telescoping identity and the closed-form F sum;
- the summation-by-parts identity, with its remainder bound;
- the log-convexity remainder bound.

The only inputs are the definitions of $`R_j`$, meaning the de-mixing constant $`E_2`$, the normalisation $`h_2`$ and the
recursion, which are taken from `amp_flint` as the definition of the multiplier.

**Not yet.** Uniformity in ω over a box is not done. With ω given as a ball, the enclosure is already useless at radius
10⁻⁷ in ω/ω∞ ($`R_J`$ half-width 2×10³), and it is NaN at 10⁻⁵. The rectangular complex balls wrap through J
multiplications by $`a_j`$. Two fixes would work:
- carry $`\log T_j`$ (via lgamma), so radii add instead of compounding;
- use a mean-value form with an ω-derivative computed by the same recursion.

**Orders 5 to 9.** These are σ₂–σ₄, from `AmpPar._ys` at J = 800. The exact $`R_j`$ also telescope, but the increment
carries powers of log j:

| order | σ | j²∣ΔR_j/S⁽¹⁾∣ at j = 100, 200, 400, 800 | R₈₀₀ vs fitted, relative |
|---|---|---|---|
| 3 | σ₁ | 2.48, 2.56, 2.05, 1.04 | 2×10⁻³ |
| 5 | σ₂ | 1.42, 2.12, 2.38, 2.09 (≈ 0.3–0.4 · log j) | 4×10⁻³ |
| 7 | σ₃ | 0.04, 0.43, 0.84, 1.07 (≈ 0.02 · log² j) | 5×10⁻³ |
| 9 | σ₄ | 0.22, 0.11, 0.04, 0.20 (oscillating) | 6×10⁻³ |

The scheme of 6.2–6.4 extends by induction on the order. The near-end terms at order 2k+1 contain the normalised
lower-order sequences $`\tilde d^{(q)}_j/T_j`$, and each of those has a closed large-j form from its own telescoping:
its limit $`S^{(q)}/K`$, plus explicit ψ and ρ terms, plus a remainder already bounded at order q. So each order needs
three things:
1. The endpoint terms in closed form. These are now rational functions times $`\rho_j^{\,m}`$ and powers of
   ψ(a+j) + ψ(b+j). Summing them needs polygamma sums and summation by parts with log weights.
2. The log-convexity bound with polylog factors.
3. Bounds on $`g^{(q)}`$ inherited from order q.

Each step is mechanical. The bookkeeping grows with the order, and it is not done here.

The required accuracy tightens slowly. At radius 10⁻³ the relative budget per order is 2.4×10⁻³ (σ₁), 7×10⁻⁴,
3×10⁻⁴, 1.7×10⁻⁴, 1.1×10⁻⁴ and 8.7×10⁻⁵ (σ₆). The 1/J remainders without the closed-form sums, at J = 800, are
2–6×10⁻³. So the closed-form endpoint sums are needed from σ₂ on. A crude C·logᵏJ/J bound would need J ≈ 10⁵, which is
too slow at O(J²).

## 7. The exponential type: what the Borel plane shows, and what is still missing

Reproduce with `python3 solver/proof/exact_sigma.py --borel`, which reads the σ₀–σ₁₂ of `scan25phys`. Floating point;
nothing in this section is rigorous.

### 7.1 The singular support

These are the Padé poles of $`B(t)=\sum_k k!\,\sigma_k\,t^{-k-1}`$ (as a series in 1/t) with $`0.3\lt|t|\lt 3`$. The
[5/7] and [6/6] approximants agree to all printed digits, and [7/5] agrees to about 0.03.

| ω/ω∞ | poles t_p ([6/6]) | far end κ⋆ (ratio law) |
|---|---|---|
| 0.99 | 0.739+0.160i, 0.781+0.370i, 0.853+0.603i, 0.984+0.805i, 1.186+0.997i | 1.271+1.081i |
| 1.00 | 0.749+0.155i, 0.793+0.360i, 0.863+0.588i, 0.986+0.786i, 1.178+0.975i | 1.279+1.081i |
| 1.01 | 0.760+0.151i, 0.806+0.350i, 0.876+0.573i, 0.992+0.766i, 1.179+0.947i | 1.289+1.080i |

The poles are interpolation nodes of a cut, not real poles; the Dlog residues sum to about −1. The curve runs from
about 0.75 + 0.16i to κ⋆. Its near end moves with ω: Re t₁ goes as roughly ω^1.4. The far end hardly moves: |κ⋆| =
1.668, 1.675, 1.682.

The density vanishes at the far end. The ratio law $`\sigma_k/\sigma_{k+1}\approx(k+\beta)/\kappa_\star`$ is the
coefficient asymptotics of a Laplace integral whose density behaves like $`(\kappa_\star-\zeta)^{\beta-1}`$ at the
endpoint. That makes the endpoint exponent about 3.2 − 1.6i, from β ≈ 4.16 − 1.60i at ω∞.

The ratio law holds well. For k = 4–12 at ω∞:

```math
|\sigma_k|\,\Gamma(k+4.16)/1.675^k = 27.3,\ 26.8,\ 26.3,\ 25.9,\ 25.6,\ 25.5,\ 25.3,\ 25.1,\ 24.9 .
```

So the conjectured tail bound, in the form the brief asks for, is $`|\sigma_k|\le C\,a^k/\Gamma(k+b)`$ with
a = |κ⋆| = 1.675, b = 4.16 and C = 27.3. It is calibrated on k ≤ 12, not proved.

### 7.2 What a Laplace representation would give, rigorously

Pólya's theorem makes "S₁ is a Laplace integral over the arc" the same statement as "S₁ is entire of exponential type
and its conjugate indicator diagram is the convex hull of the arc". In that case, for any contour Γ around the hull,

```math
S_1(x)=\frac{1}{2\pi i}\oint_\Gamma e^{x t}B(t)\,dt,\qquad |\sigma_k|\le\frac{|\Gamma|}{2\pi}\,\max_\Gamma|B|\;\frac{\max_\Gamma|t|^k}{k!} .
```

The type is the largest |t| over the hull, which is |κ⋆| if κ⋆ is the farthest point. If moreover the density φ on
the arc is integrable and vanishes at κ⋆ like $`|\kappa_\star-\zeta|^{\mathrm{Re}\,\beta-1}`$, the bound sharpens to
$`C\,|\kappa_\star|^k/\Gamma(k+\mathrm{Re}\,\beta)`$. That is the bound of 7.1.

Every version needs one a priori input: a bound on B on a contour, or on S₁ on a circle |x| = R (section 4). Without
it, the computed σ_k cannot certify the tail. **No rigorous type bound was obtained in this pass.**

### 7.3 Where the arc comes from: tested and open

**Not the transport-exponent map.** $`u(\theta)=(\theta-\tfrac12\sin2\theta)/(2\omega)`$ is the lapse exponent
$`\int y_1^2\,d\xi/(2\xi)`$ along the characteristics θ = ω log ξ + const of $`L=\xi^2\partial_\xi+\omega\xi\partial_\theta`$.
Its range does not match the arc, in position or in ω-dependence. The far end would scale like 1/ω; it does not.

**Not a single exponential factor.** The de-mixing factor $`e^{\Pi}`$ (Π ∝ x) and the finite part of the lapse exponent
each give one point in the Borel plane, not a cut. By the Leibniz rule for alien derivatives, a factor
$`e^{x\Omega}`$ enters the Stokes constant through its germ at the singular point. A continuous arc therefore needs a
one-parameter family: an angle average, or an integral along a path.

**Observation: the exponentials of the arc points stay bounded near Re ≈ 2.**

| t | 0.749+0.155i | 0.793+0.360i | 0.863+0.588i | 0.986+0.786i | 1.178+0.975i | κ⋆ |
|---|---|---|---|---|---|---|
| $`e^{t}`$ | 2.09+0.33i | 2.07+0.78i | 1.97+1.31i | 1.89+1.90i | 1.82+2.69i | 1.70+3.18i |

So $`e^{xt}=\xi^{x}`$ with ξ on a bounded curve running from about 2.1 + 0.3i to 1.7 + 3.2i. That suggests
$`S_1(x)\approx\int_\gamma\xi^{x}\psi(\xi)\,d\xi`$. To leading order ⟨F²⟩ = x/2, so the late-order normalisation
power $`\xi^{2\langle F^2\rangle}`$ is exactly $`\xi^{x}`$. That power is also the lapse's large-ξ growth.

**Candidate mechanism, untested.** The x-dependence of the multiplier is the normalisation and lapse power ξ^x,
integrated against an x-independent kernel along a bounded complex path in the inner variable. The pairing integral
S = ∫ W f of section 4 is the natural place to look for γ. Its endpoints would set κ⋆ = log ξ_B, independent of ω, and
the near end, which depends on ω.

Two tests:
1. Compute the order-3 pairing integrand in the Borel variable of z = 2ξ², where the order-1 Borel transform is
   ₂F₁(a, b; 1; τ). Find where it localises.
2. Check that the order-(2k+1) multiplier, written as k-fold Borel convolutions over a simplex, produces the 1/k! as a
   simplex volume, with the exponent's values on the simplex filling the arc.

Test 2 is the most promising route to a proof of the type. A k-fold convolution of a bounded kernel over the unit
simplex is bounded by M^k/(k−1)!, which is exactly a type bound, with constants that can be made explicit from the
order-1 and order-2 Borel transforms.

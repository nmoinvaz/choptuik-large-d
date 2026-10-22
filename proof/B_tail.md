# Gap (b): the harmonic tail

This note analyses gap (b) of Conjecture 1 (README, "What is known about the selection"; [PROOF.md](../PROOF.md) §4 A.3
and §5): uniformity in the harmonic index. It states the problem without a harmonic expansion, tests the two worries
against the amplitude expansion (where every harmonic is exact) and the hierarchical solver, proposes the function space
in which a tail lemma should be proved, estimates its constants from the data, and gives a verdict. Written by Claude
(Anthropic) under Nathan Moinvaziri's direction; not reviewed by a mathematician.

Scripts: `solver/proof/tail_harmonics.py` (tuned harmonics and their decay), `solver/proof/reconcile.py` (truncated
against untruncated selection), `solver/proof/amp_trunc.py` (the amplitude expansion projected onto |μ| ≤ M, order by
order), `solver/proof/family_amps.py` (neighbour-family amplitudes in the late orders).

Contents

- [Verdict](#verdict)
- [1. The two worries](#1-the-two-worries)
- [2. The condition without harmonics, and why the weights cancel](#2-the-condition-without-harmonics-and-why-the-weights-cancel)
- [3. F's complex singularities in the inner problem](#3-fs-complex-singularities-in-the-inner-problem)
- [4. The tail, order by order](#4-the-tail-order-by-order)
- [5. Reconciling the truncated and untruncated selections](#5-reconciling-the-truncated-and-untruncated-selections)
- [6. A framework for the tail lemma, with constants](#6-a-framework-for-the-tail-lemma-with-constants)
- [7. What remains open](#7-what-remains-open)

## Verdict

*Note added: the untruncated "bracket" 0.99915–1.00011 used below is not a coverage guarantee, and twelve coefficients
fix the zero only if S₁ has no other zeros within |x| ≈ 9 ([notes/PLANTED_SELECTION.md](../notes/PLANTED_SELECTION.md)).*

Gap (b) is real but not fatal, and it is narrower than the README's statement of it suggests.

- **Worry 1 (weights against decay) does not arise in the tuned problem.** The condition S ≡ 0 is stated on the real
  label circle and involves F only on the real line (§2). The weights e^{π|ν|ω/2} are the linear connection coefficients
  of each label harmonic. They multiply zero in S ≡ 0, and they cancel in each tuning equation, because the free and
  forced parts of S_ν are jumps of the same mode. The data confirm it:
  - the tuning ratio q_ν = f_{ν+2}/f_ν (leading order) is O(1) and independent of ν to six digits, q∞ = 0.008524 + 0.000808i;
  - the tuned profile decays by 0.0574 per pair of harmonics, so it is analytic in |Im θ| < 1.428;
  - the neighbour-family amplitudes in the late orders fall by 10⁴–10⁸ per pair instead of growing by about 25.
- **But naive truncation brings the weights back.** Projecting the amplitude recursion onto |μ| ≤ M changes σ_M by about
  5×10⁻⁴, as order counting predicts. It then gives σ_{M+1} too large by a factor 10⁴–10⁶, worse at M = 5 than at M = 3 (§4).
  A tail lemma must therefore be proved for the *jump-conditioned* truncation (S₃ = … = S_M = 0 imposed, the higher labels
  eliminated: `hier.py --schur`), whose sequence converges, not for a Galerkin projection of the equations.
- **Worry 2 (tail size): the needed bound looks provable.** The jump-conditioned sequence alternates and shrinks by
  r ≈ −0.045 to −0.057 per pair, the profile's own decay times a coupling of 0.7–1.0 (§5). That puts the M > 5 tail at
  about 4–6 in r₁, the M > 7 tail at 0.2–0.35 and the M > 9 tail at about 0.015. With M = 7 carried explicitly, a 10⁻³ box
  needs a tail bound below about 4 and has a margin of about 13. Even a proven ratio of 0.25 per pair would suffice.
  A 10⁻⁴ box needs M = 9 explicit.
- **The two routes are consistent.** The truncated sequence and the alternation sign put the limit at 0.99923–1.00011. The
  untruncated bracket at order 12 is 0.99915–1.00011, with the central Padé value 0.99989. They overlap, and both
  contain 1. M = 7 should land at 0.99997–1.00017, a sharp test when it runs.
- **Secondary Stokes lines from F's complex singularities: no sign of them.** They are absent at every finite order in
  A, where each order is a trigonometric polynomial and the late orders contain only fast-mode families. S₁ is numerically
  entire in A², and the two routes agree. What is not excluded is a non-uniformity in the order of limits (sum in A,
  then late orders in j). The strings of singular points that F's singularities create in complex ξ, at arg ξ ≈ ∓21°
  for large |ξ|, have not been located at |ξ| = O(1). A proof should define S₁ order by order through lateral
  Borel–Laplace sums and sum in A² last (§3).

**Bottom line:** the hard lemma is a Lipschitz bound for the jump-conditioned (Schur-complement) residual in a strip
space X_σ, σ ≈ 1, with a per-pair constant of about 0.05 (needed: ≤ 0.25 with M = 7 explicit). It is a new lemma, but a
finite-dimensional, explicit-constant one, of the kind computer-assisted proofs handle; nothing found here suggests it
is false for the jump-conditioned truncation.

## 1. The two worries

Notation as in the README: F(θ) = A sin θ + Σ_{ν≥3} f_ν(A) e^{iνθ} + c.c. is the D = ∞ echo profile, F₁ = A/(2i), ξ the
inner variable, s = iνω, and S_ν the fast-type Stokes multiplier of label harmonic ν. The linear multiplier of harmonic
ν is $`S^{lin}_\nu = 2F_\nu/(\Gamma(\alpha_\nu)\Gamma(\beta_\nu))`$ with α_ν = (1 + iνω)/2, β_ν = (2 + iνω)/2, and by Stirling

```math
\frac{1}{\lvert\Gamma(\alpha_\nu)\Gamma(\beta_\nu)\rvert} \simeq \frac{(\nu\omega/2)^{-1/2}}{2\pi}\,e^{\pi\lvert\nu\rvert\omega/2}, \qquad \frac{\pi\omega_\infty}{2} = 6.004 .
```

**Worry 1 (weights against decay).** The tuned harmonics decay like $`\lvert F_{\nu+2}/F_\nu\rvert \approx 0.0574`$, i.e.
e^{−1.428|ν|} (§2), while the weights grow like e^{6.004|ν|}. Summed over ν, the linear multipliers
$`\sum_\nu S^{lin}_\nu e^{i\nu\theta_0}`$ are the profile's positive- and negative-frequency parts continued to
Im θ₀ = ∓πω/2 ≈ ∓6.0, far outside the strip |Im θ| < 1.428 in which F's Fourier series converges.

**Worry 2 (tail size).** Truncating at M = 5 harmonics moves the selected (A, ω) by an amount that corresponds to a change
of about 60–110 in the harmonic-1 residual r₁ between M = 3 and M = 5, and a certified box of radius 10⁻³ needs the
contribution of everything above the explicit harmonics enclosed to about 4 in r₁ ([PROOF.md](../PROOF.md) A.4).

## 2. The condition without harmonics, and why the weights cancel

**The label function.** With δy = e^{−1/(2ξ²)} a, the fast mode obeys ξ∂_ξ a + 2ω∂_θ a = 2F² a (README, "The fast mode
with gravity"), so

```math
a(\xi,\theta) = \xi^{2\langle F^2\rangle}\,e^{\Pi(\theta)}\,g(\theta_0), \qquad \theta_0 = \theta - 2\omega\ln\xi, \qquad \Pi(\theta) = \frac{1}{\omega}\int^\theta \big(F^2 - \langle F^2\rangle\big),
```

and the jump of the inner solution across the Stokes line (real ξ), i.e. the difference of the lateral Borel–Laplace sums
along arg ξ² = 0±, is

```math
[y](\xi,\theta) \sim S(\theta_0)\,e^{-1/(2\xi^2)}\,\xi^{2\langle F^2\rangle}\,e^{\Pi(\theta)}\,\big(1 + O(\xi^2)\big), \qquad \xi\to0^+ ,
```

with S a 2π-periodic function of the label. The selection condition is S ≡ 0 on the label circle. Stated this way it
involves only real θ and real labels: Π is evaluated on the real circle, where F is analytic, and no harmonic expansion
and no continuation of F is needed.

**Where the weights enter.** Expanding S in label harmonics, S_ν is the jump carried by label harmonic ν. In the linear
problem it is the connection coefficient of Weber's equation with parameter iνω, $`S^{lin}_\nu = 2F_\nu/(\Gamma(\alpha_\nu)\Gamma(\beta_\nu))`$:
a unit amplitude in harmonic ν produces a jump e^{π|ν|ω/2} times larger than in harmonic 1. Summed over ν, this linear map
is formally F⁺ and F⁻ continued to Im θ₀ = ∓πω/2 (worry 1). But it is only ever applied harmonic by harmonic.

**The tuning equations are ratios.** For ν ≥ 3 the condition S_ν = 0 fixes f_ν. Its two parts, the linear jump of the
free harmonic f_ν e^{iνθ} and the jump generated in the same label harmonic by the nonlinear forcing from lower harmonics,
are jumps of the same label-ν mode, so both carry the same connection factor $`1/(\Gamma(\alpha_\nu)\Gamma(\beta_\nu))`$, and
it cancels: f_ν = −(forced jump)/(2 × linear coefficient) is an O(1) functional of the lower harmonics. In S ≡ 0 the
weights multiply zero, and in the tuning they cancel; the weight e^{π|ν|ω/2} never meets the decay e^{−1.428|ν|}.

**Test** (`tail_harmonics.py`, amplitude expansion at ω∞, K = 25, phys fits, A = 2.7196). The leading-order ratio
$`q_\nu = f^{(\nu+2)}_{\nu+2}/f^{(\nu)}_\nu`$ is O(1) and independent of ν to six digits, with no trace of e^{2·6.004} = 1.6×10⁵:

| ν | q_ν | \|q_ν\| A² | \|F_ν/F_{ν−2}\| (summed in A) |
|---|---|---|---|
| 3 | 0.008512 + 0.000849i | 0.06327 | 0.0551 |
| 5 | 0.008521 + 0.000820i | 0.06332 | 0.0572 |
| 7 | 0.008523 + 0.000813i | 0.06333 | 0.0574 |
| 9 | 0.008524 + 0.000811i | 0.06333 | 0.0574 |
| 11 | 0.008524 + 0.000809i | 0.06333 | 0.0574 |
| 13–23 | 0.008524 + 0.000808i | 0.06333 | 0.0575 (ν ≤ 15; fewer A-terms beyond) |

The A-series of each tuned harmonic converges at the selected amplitude: the ratio of successive terms
$`\lvert f^{(k+2)}_\nu A^2/f^{(k)}_\nu\rvert`$ settles at about 0.25 for ν = 3–11 (radius of convergence in A about twice A*).
So the tuned profile is analytic in the strip |Im θ| < σ_F with

```math
\sigma_F = \tfrac12\ln\frac{1}{0.0574} = 1.428, \qquad \lvert F_\nu\rvert \simeq \lvert F_1\rvert\,e^{-\sigma_F(\lvert\nu\rvert-1)},
```

and the profile is, to leading order, $`F \approx 2\,\mathrm{Re}\big[F_1 e^{i\theta}/(1 - q_\infty A^2 e^{2i\theta})\big]`$ with
q∞ = 0.008524 + 0.000808i, whose positive-frequency part has simple poles at Im θ = −1.38 (leading order) to −1.43 (summed).

**Consequence for worry 1.** The formal sum $`\sum_\nu S^{lin}_\nu e^{i\nu\theta_0}`$ diverges, but it never has to be formed:
the condition is diagonal (S_ν = 0 for each ν), and the tuned F lives in a space of strip-analytic functions in which the
tuning map is O(1). Read as a function, the "continuation to Im θ = −6" is the continuation of F⁺, which exists (in the
leading-order model F⁺ = F₁z/(1 − q A² z²) with z = e^{iθ}, it is meromorphic in z, and at |z| = e^{6} ≈ 400 it lies far
beyond its pole at |z| ≈ 4); nothing in the selection uses it.

## 3. F's complex singularities in the inner problem

**Where they sit.** At large |ξ| the slow inner solution tends to the outer profile transported along φ = θ − ω ln ξ
(README: L = −∂_t at fixed φ, t = 1/ξ). So at real θ a singularity θ_s of F, Im θ_s = ±1.428, reaches complex ξ with

```math
\arg\xi \to \mp\frac{1.428}{\omega_\infty} = \mp 0.374\ (21.4^\circ), \qquad \lvert\xi\rvert = \exp\Big(\frac{\theta - \mathrm{Re}\,\theta_s + 2\pi m}{\omega}\Big),\quad m\in\mathbb{Z},
```

a string of singular points along two directions, spaced by the factor e^{2π/ω} = e^{Δ∞} = 5.18 (the echo). At finite |ξ|
the nonlinear and ∂_φ terms move them; where they sit at |ξ| ~ 0.2–1 has not been computed. For the fast mode's label
θ₀ = θ − 2ω ln ξ the corresponding directions are arg ξ = ∓0.187 (10.7°). Both lie inside the sector between the Stokes
line (real ξ) and the anti-Stokes lines (arg ξ = ±π/4). So the worry is real in principle: continuing the inner solution in
complex ξ, one can meet singularities that are not those of the fast mode, and with them secondary exponentially small
terms.

**At each order in A they are absent.** In the amplitude expansion every order is a trigonometric polynomial in θ, so
there is no complex θ-singularity at any finite order, and the late orders in j of each order are fitted by the main
family plus the fast-mode families ν′ (|ν′| ≤ k) alone, with relative residuals of about 10⁻²⁵ and agreement between
J = 750 and 1000 to about 10⁻¹⁹ for σ₁–σ₁₂ (README). A secondary Borel singularity of comparable strength at any other
location would spoil that convergence in J. So, order by order in A², there are no secondary exponentially small terms
at the resolved accuracy, and S₁ is a well-defined power series in A².

**After summing in A.** The singularities of F appear only in the A-summed profile (q A² e^{2iθ} = 1). Three facts bound
their effect on the selection:

- each tuned f_ν(A) is analytic in A out to about 2A* (§2), and S₁(A², ω) is numerically entire in A² (its coefficients
  satisfy |σ_k/σ_{k+1}| ≈ 0.60k + 2.5, PROOF.md), so summing in A creates no singularity of the multiplier near the
  selected amplitude;
- the two routes, which handle the θ-dependence in opposite ways (truncation in harmonics with direct integration in
  ξ, and exact harmonics with the series in A), agree on the selected ω to about 10⁻³ (§5), which they would not if a
  secondary Stokes term of the size of the primary one were present in one and not the other;
- the neighbour families, which is where uniformity in ν would fail, decay instead of growing
  (`family_amps.py`; K = 11, J = 500, phys fits at ω∞). Refitting the late orders of harmonic 1 and reading off each
  family's coefficient at relative order 1/j (its amplitude relative to S₁⁽ᵏ⁾):

| order k | ν′ = −1 | ν′ = ±3 | ν′ = ±5 | ν′ = ±7 | ν′ = ±9 | ν′ = ±11 |
|---|---|---|---|---|---|---|
| 5 | 0.37 | 2×10⁻¹⁷–1×10⁻¹⁶ | 3×10⁻²³–1×10⁻²² | | | |
| 7 | 0.55 | 1×10⁻¹⁶–8×10⁻¹⁶ | 6×10⁻²¹–2×10⁻²⁰ | 1×10⁻²⁷–3×10⁻²⁷ | | |
| 9 | 0.73 | 1×10⁻¹³–1×10⁻¹² | 5×10⁻¹⁷–4×10⁻¹⁶ | 6×10⁻²²–3×10⁻²¹ | 4×10⁻²⁹–2×10⁻²⁸ | |
| 11 | 0.89 | 2×10⁻⁹–2×10⁻⁸ | 2×10⁻¹²–2×10⁻¹¹ | 2×10⁻¹⁶–1×10⁻¹⁵ | 7×10⁻²²–4×10⁻²¹ | 1×10⁻²⁹–6×10⁻²⁹ |

  Apart from the conjugate label ν′ = −1, which is O(1) because F is real, every neighbour family is tiny and falls by
  four to eight orders of magnitude per pair of harmonics. Had the weights e^{π|ν′|ω/2} survived in the tuned problem,
  these amplitudes would grow by about e^{πω}|q|A² ≈ 25 per pair. They do the opposite: the tuning removes the label-ν′
  jumps (S_{ν′} = 0) and with them the weighted contributions. (The ν′ = ±3 entries grow with k, from 10⁻¹⁶ to 10⁻⁸;
  whether that is a genuine subleading term or fit contamination at the higher orders is not resolved here.)

**What is not settled.** The order of limits. S₁ is defined at each order in A by the j → ∞ behaviour of that order, and
the onset of the asymptotic regime in j moves out with the order (σ₁₃ does not converge at J = 750). The family test
shows no growth in |ν′| at the orders probed (k ≤ 11), but it does not prove uniformity in k. A proof should therefore
(i) define S₁ order by order in A² through lateral Borel–Laplace sums (where the families are just further terms of the
trans-series), (ii) prove the growth bound on σ_k, and (iii) only then sum in A². In that order of operations F's complex
singularities never enter. Where the strings of singular points of §3 sit at |ξ| = O(1), and whether the lateral sums can
be taken between them, is the one place where they could; it has not been computed.

## 4. The tail, order by order

**Where the omitted harmonics first enter.** Counting orders in the amplitude expansion (each step down by two
harmonics costs A²), the even harmonic M + 1 of c, e and Π reaches harmonic 1 of y at relative order A^{2M}, i.e. at σ_M,
and the odd profile harmonic M + 2 at σ_{M+1}. So a truncation at M can change S₁ only from σ_M on.

**A Galerkin projection of the recursion** (`amp_trunc.py`): the amplitude recursion of `amp_flint.py` with every θ-
function projected onto |μ| ≤ M, harmonics 3…M tuned, and late-order fits with every family present (shifts up to 2M).
At ω∞, J = 500, against the converged full σ_k:

| k | \|Δσ_k\|, M = 3 | \|Δσ_k\|, M = 5 | full σ_k |
|---|---|---|---|
| 1 | 3×10⁻¹⁰ (fit bias) | 2×10⁻¹⁰ | 0.744 + 0.524i |
| 2 | 3×10⁻¹⁰ | 3×10⁻¹⁰ | 0.119 + 0.398i |
| 3 | **5.3×10⁻⁴** | 2×10⁻¹⁰ | −0.0540 + 0.1166i |
| 4 | 4.0×10² | 3×10⁻¹¹ | −0.0289 + 0.0108i |
| 5 | 3.4×10² | **4.4×10⁻⁴** | −0.00555 − 0.00281i |
| 6 | 1.4×10² | 1.4×10³ | −0.00028 − 0.00108i |

The first change appears exactly at σ_M, as counted, and is modest (5×10⁻⁴). One order later the projected system gives
σ_{M+1} that are 10⁴–10⁶ times the true values (−399 at M = 3, −1371 at M = 5), and the late-order fit residual rises at
that order (from about 10⁻³² to 10⁻²⁵ at M = 3). Either the projected system's multiplier really is that large, or its
late orders no longer have the full system's family structure. In both cases the naive Galerkin truncation is **not** a
controlled approximation, and it gets worse, not better, with M (2×10⁸ against 1×10⁶ at A* in |Δσ_{M+1}| A*^{2(M+1)}).

This is worry 1 in finite form. In the full problem, the weighted jump of every label ν ≥ 3 is cancelled between its free
and forced parts (§2). The projection cuts the harmonics through which that cancellation propagates (Π, and the fast
modes of the projected system are no longer those of e^{Π}). The weights then reappear, at the first order where the
missing harmonic M + 2 would have been tuned. The hierarchical solver does not do this: it imposes S₃ = … = S_M = 0 on
the truncated system itself, de-mixes with the truncated Π, and removes the leakage by the Schur complement. Its
sequence converges (−22%, +1.56%, −0.077%, §5).

**Consequence.** The tail lemma cannot be a statement about Galerkin projections P_M of the equations. It has to be a
statement about a truncation that keeps the jump conditions: the M-harmonic problem with S₃ = … = S_M = 0 imposed and
the higher-label jumps eliminated (`hier.py --schur`), or, equivalently, a bound on the tuned coefficients f_ν^{(k)} of
the untruncated amplitude recursion with ν > M.

## 5. Reconciling the truncated and untruncated selections

The truncated zeros (`hier.py`, README) are c₁ = 0.776994 (M = 1), c₃ = 1.0156 (M = 3), c₅ = 0.999228 (M = 5); the
untruncated selection at order 12 lies in 0.99915–1.00011, with the central Padé value 0.99989. If c_M − L alternates with
a fixed ratio r per pair of harmonics, then L = c₅ + (c₅ − c₃) r/(1 − r) (`reconcile.py`):

| limit L | d₅/d₃ = r | M > 5 tail in ω/ω∞ | in r₁ (×6.7×10³) |
|---|---|---|---|
| 0.99915 (bracket, low end) | +0.005 | −0.00008 | 0.5 |
| 0.99989 (central Padé) | −0.042 | +0.00066 | 4.4 |
| 1.00000 | −0.0495 | +0.00077 | 5.2 |
| 1.00011 (bracket, high end) | −0.057 | +0.00088 | 5.9 |

The alternation (−22%, +1.56%, −0.077%) fixes the sign, r < 0, so L lies in the upper part of the untruncated bracket,
0.99923–1.00011. The required ratio, −0.042 to −0.057, is the measured harmonic decay per pair (|F_{ν+2}/F_ν| = 0.0574)
times a coupling factor 0.73–1.0: the M > 5 tail is just the first omitted harmonic pair, attenuated like the profile.
With r between −0.045 and −0.057, the prediction for M = 7 is c₇ = 0.99997–1.00017, and the M > 7 tail is 3–5×10⁻⁵ in
ω/ω∞, i.e. 0.2–0.35 in r₁. The two routes are therefore consistent, and the M = 7 run, when it completes, is a sharp test.

## 6. A framework for the tail lemma, with constants

**Space.** For 0 < σ < σ_F = 1.428, let X_σ be the odd trigonometric series with

```math
\lVert F\rVert_\sigma = \sum_{\nu\ \mathrm{odd}} \lvert F_\nu\rvert\,e^{\sigma\lvert\nu\rvert} \lt \infty ,
```

an algebra under multiplication (‖FG‖_σ ≤ ‖F‖_σ‖G‖_σ) of functions bounded and analytic on |Im θ| < σ. The multipliers S_ν
are measured with the dual weights $`e^{-\pi\lvert\nu\rvert\omega/2}`$, i.e. one works with the normalised multipliers
$`\hat S_\nu = \Gamma(\alpha_\nu)\Gamma(\beta_\nu)\,S_\nu`$, in which form the tuning equations are O(1) (§2).

**The tuned family as a fixed point.** Write the tuning as f_ν = T_ν(F) for ν ≥ 3. At leading order in A² the map is
the weighted shift f_{ν+2} = q∞A² f_ν, of norm

```math
\lVert T_0\rVert_{X_\sigma} = \lvert q_\infty\rvert A^2 e^{2\sigma} = 0.0633\,e^{2\sigma}, \qquad \lt 1 \iff \sigma \lt 1.380 ,
```

e.g. 0.47 at σ = 1. The corrections are higher orders in A², whose series converge with term ratio about 0.25 at A*
(§2). So F = F₁ e^{iθ} + c.c. + T(F) is a contraction in X_σ (σ ≈ 1) with explicit constants, which is the analogue of a
"high-frequency" bound in a computer-assisted proof (Reiterer–Trubowitz split the Fourier modes the same way).

**The tail of r₁.** Let P_M project onto |ν| ≤ M and let r₁^{(M)} be the harmonic-1 residual of the M-harmonic model
with the harmonics 3…M tuned (the Schur complement of `hier.py`, in which the higher harmonics sit on their own zeros).
The hierarchical sequence (§5) gives

```math
\big\lvert r_1^{(\infty)} - r_1^{(M)}\big\rvert \simeq C_T\,\rho^{M-1}, \qquad \rho^2 \approx 0.045\text{–}0.057, \qquad C_T \approx 2.4\times10^{3},
```

calibrated on the M = 3 → 5 change (about 110 in r₁): about 5 at M = 5, 0.25–0.35 at M = 7, 0.012–0.02 at M = 9.

**What the proof needs.** A certified box of radius 10⁻³ needs the remainder enclosed to about 4. Carrying M = 7
explicitly (with `--schur`, which removes the double-precision barrier of the harmonic-7 level) leaves a tail of about
0.3, a margin of 13. Even a crude proven ratio ρ² ≤ 0.25 per pair would suffice with M = 7 explicit (bound
5.9 × 0.25/0.75 ≈ 2 < 4). A box of radius 10⁻⁴ (0.39) needs M = 9 explicit. The lemma to prove is therefore:

> **Tail lemma (target).** In X_σ with σ ≈ 1, the tuned family satisfies ‖(I − P_M)F‖_σ ≤ C₁ (|q∞|A²e^{2σ})^{(M+1)/2}, and
> the residual of the jump-conditioned M-harmonic problem (S₃ = … = S_M = 0 imposed, higher labels eliminated) differs from
> the untruncated one by |r₁^{(∞)} − r₁^{(M)}| ≤ L ‖(I − P_M)F‖_σ, with explicit C₁, L, such that the bound at M = 7 is
> below about 2. It is false for a plain Galerkin projection of the equations (§4).

The first half is a contraction estimate of the kind above. The second half is the genuinely new part: L must be
estimated for the *tuned* system, since the raw sensitivity of r₁ to one harmonic coefficient is about 10¹⁷ (PROOF.md
A.3). In the amplitude-expansion form this becomes a bound on how the tuned coefficients f_ν^{(k)} feed σ_k, which is a
finite algebraic recursion order by order (§4).

## 7. What remains open

1. **The jump-conditioned tail bound itself.** The per-pair ratio −0.045 to −0.057 is measured on three points (M = 1, 3,
   5), with M = 1 pre-asymptotic. M = 7 (`hier.py 7 … --predict --floor --schur`) is the next data point. The prediction is
   c₇ = 0.99997–1.00017.
2. **Why the Galerkin projection explodes at σ_{M+1}.** Either its multiplier is genuinely large or its late orders have
   a family structure outside the fit basis (the fit residual rises). Deciding this is a few hours' work: compute the
   projected system's fast-mode exponents, refit, and compare with a lateral (`inner_np.py`) evaluation of the same
   projected system. That would also show exactly which harmonics carry the cancellation.
3. **The location of the secondary singular strings at |ξ| = O(1)** (§3), and whether the lateral Borel–Laplace paths
   used in the Stokes jump pass between them uniformly in the amplitude order.
4. **Uniformity in k of the late-order onset.** σ₁₃ needs J > 750. That growth should be quantified, with J needed
   against k, to check that it is a resolution matter (more neighbour families, more rows) and not a breakdown of the
   order-by-order definition.
5. **The constants in X_σ.** These are estimates from data: |q∞|A²e^{2σ} for the profile map, about 0.25 per A² order
   for the corrections, and the per-pair coupling 0.7–1.0 for r₁. Each needs a rigorous bound; the first is the easiest,
   since q_ν has a closed leading-order form to be derived (1/q ≈ 8ω² − 4iω − 1.3, README).

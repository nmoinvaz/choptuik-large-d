# Independent review of the M = 1 certificate

Task: an independent soundness review, by reading the code and testing it, of the one-harmonic (M = 1) computer-assisted
proof at commit 779ae9c (`proof/A_certify.md` §4.4–4.5 and §9; `solver/proof/fixed_point.py`, `series_arb.py`,
`residual_poly.py`, `validated_int.py`, `krawczyk.py`, `lin_check.py`). Reviewer: agent RV. Test scripts:
`solver/proof/review/`.

## Contents

- [Verdict](#verdict)
- [Summary of the checks](#summary-of-the-checks)
- [1. Rigor of every enclosure](#1-rigor-of-every-enclosure)
- [2. The fixed-point lemma (step i)](#2-the-fixed-point-lemma-step-i)
- [3. The Lohner step (step ii)](#3-the-lohner-step-step-ii)
- [4. The Krawczyk test (step iv)](#4-the-krawczyk-test-step-iv)
- [5. The linear check and the conventions of lateral_np.py](#5-the-linear-check-and-the-conventions-of-lateral_nppy)
- [6. Hidden assumptions](#6-hidden-assumptions)
- [7. Experiments](#7-experiments)
- [8. Reproduce](#8-reproduce)

## Verdict

**The computation is sound.** I found no error in any enclosure that enters the certificate, and every input of the
Krawczyk test was reproduced by an independent method: the start values to 5×10⁻³⁵, the point run to 10⁻³¹, one
Lohner step against a 50-digit solution, and K(X) itself with separate code. What the code proves is:

> For x = (A, ω/ω∞) in X = (2.5421480475, 0.7769940136) ± 10⁻⁹, the function
> $`J_D(x) = D^+_{+1}(0.3) - \overline{D^+_{-1}(0.3)}`$, with y⁺ continued from the ray arg ξ = 3π/8 along that ray
> to radius 0.3 and then along the arc of radius 0.3, has exactly one zero in X, and it is simple.

**Two steps between that statement and the stated theorem are not proved** (§6a, §6b; both *fixable*):

- **6a.** Only one complex component of the jump, at one point ξ₁, is shown to vanish. The selection condition is
  that the whole Stokes jump vanishes. `A_certify.md` identifies the two (§5.1: "g vanishes exactly where J₁(ξ₁) = 0,
  which is the selection condition of §2") without proof.
- **6b.** y⁺ on the real axis is defined in §2 as the Borel sum, but computed by continuation along one path. The two
  agree only if y⁺ is analytic between that path and the real axis. A movable singularity of y⁺ lies just outside the
  path, between |ξ| = 0.30 and 0.35: continuations along arcs on either side of it differ by 82.

Numerically both steps hold at the certified zero (§7, R6): every jump component vanishes at the same point, inside
K(X), at ξ₁ = 0.25 and 0.3. The arcs at 0.2 and 0.3 give the same y⁺(0.3) to the double-precision solver's accuracy.
The missing argument is the template's step 4 (§1 of `A_certify.md`, "the Stokes constant from the difference"), which
§3 replaced by "(iii) the jump comes from the same enclosure".

Until it is supplied, the precise claim is the quoted statement above. Combined with 6b, it gives **at most one**
zero of the M = 1 selection condition in X; **exactly one** needs 6a as well.

One quoted number is wrong as an enclosure (§4, F3, *cosmetic*): A = 2.542148047617 ± 9.1×10⁻¹¹ misses the lower
end of K(X) by 4×10⁻¹³. The same runs give a correctly rounded enclosure about 8 times tighter (§4).

## Summary of the checks

| check | verdict | severity |
|---|---|---|
| 1. no float/numpy value in a bound; JSON save/load | OK | — |
| 1′. `max` over overlapping balls in `fixed_point.run` | issue (no effect here) | cosmetic |
| 2a. integral equation = the M = 1 system | OK (exact identities, R1) | — |
| 2b. kernel bound, sign of Re, sector rate | OK (R2) | — |
| 2c. inverse-operator bounds | OK (R2) | — |
| 2d. Lipschitz majorants, weighted norm, monotonicity in R | OK (R7) | — |
| 2e. residual vanishes below order 53 (also over boxes) | OK (exact, R1); code only checks "ball ∋ 0" | cosmetic (cite R1) |
| 2f. y⁺ identification (Braaksma, Volterra uniqueness) | OK (R5) | — |
| 2g. Cauchy estimate for the parameter derivatives | OK | — |
| 3a. a-priori enclosure, Taylor and variational remainders | OK (R3) | — |
| 3b. step control, dyadic grid, legs | OK | — |
| 3c. QR-Lohner update, mean value form | OK | — |
| 3d. variational columns are true derivatives | OK (R3, R8) | — |
| 3e. ray-then-arc path and chain-rule factor | OK (R3, with a negative control) | — |
| 3f. tightness of the remainder | radius ~10¹² × the true error | not a soundness issue |
| 4a. K(X) formula, J over the whole box, fixed real Y, strict interior | OK (R4) | — |
| 4b. g = κ J_D, κ constant, no derivative of the normalization | OK | — |
| 4c. quoted zero enclosure A = 2.542148047617 ± 9.1×10⁻¹¹ | not outward-rounded | cosmetic (F3) |
| 5. linear check, normalization, orientation vs `lateral_np.py` | OK (R4, R8) | — |
| 6a. J_D(ξ₁) = 0 ⇒ whole Stokes jump = 0 | not proved | **fixable** (missing lemma) |
| 6b. path continuation = Borel sum on the real axis | not proved; singularity near \|ξ\| ≈ 0.33 | **fixable** (missing lemma) |
| 6c. the M = 1 system is the one the claims are about | OK | — |
| 6d. ω∞ used consistently | OK; it is a convention | cosmetic |
| 6e. "± 10⁻⁹" refers to (A, ω/ω∞), max norm | OK | — |

## 1. Rigor of every enclosure

I read every `float(`, `np.` and `str(` in the certificate path (`fixed_point.start_data` → `validated_int`'s
`krawczyk` mode → `krawczyk.normalization`, `g_from_states`, `krawczyk`). Each one is one of:

- **a choice, not a bound**, where any value would keep the proof valid:
  - the series length J and the truncation N in `setup`;
  - the Perron weights (`np.linalg.eig`, converted exactly by `arb(float)` and then used consistently);
  - the step size, a multiple of 2⁻³⁰;
  - the pivot order and the Q factor in `qr_pivot` (exact once converted; its inverse is enclosed by `arb_mat.inv`);
  - the preconditioner Y in `krawczyk` (exact binary floats);
  - the inflation factors in `apriori` (containment is then verified with `contains`);
- **a report**: printed radii, margins, `eta0`, `rad_of`;
- **outside the certificate path**: `start_lohner` with its assumed float `eps0`, `kappa`, `H_enclosure`, `precond`,
  the `point`, `boxtest` and `miranda` modes.

The arithmetic itself is ball arithmetic throughout.

Every input enters as a decimal string at 320 bits: r₀ = 0.1, ξ₁ = 0.3, 1.64389375, x̂ and h. So every ball contains
the exact decimal. The time grid is exact: every node is a multiple of 2⁻³⁰, and the last step lands on t = 1, so the
arc ends exactly at ξ = ξ₁.

**JSON save/load (the old precision bug).** `kpoint_job` and `kbox_job` write `x.str(600, radius=True, more=True)`.
`parse_acb` reads it back with `arb(str)` at 320 bits.

- arb guarantees containment for this format. `arb_get_str` adds the decimal conversion error to the printed radius,
  also with `ARB_STR_MORE`; "more" only means that some printed digits may not be correct.
- R4 checks it on 20 000 random balls of the sizes validated_int produces. No parsed ball failed to contain its
  original, whether parsed at 320 or at 53 bits, and the radius grew by at most a factor 1.0099.
- The old 12-digit printout was also an enclosure, but up to 10⁷⁰ times wider. That loss is what tripled the
  ∂J_D radii in §9.3.
- **Verdict: OK.**

**1′. `max` over balls (cosmetic).** `fixed_point.run` takes `F = max(...)`, `lip0 = max(...)` and `lip = max(...)`
over arb balls with Python's `max`. arb's `>` is True only when certain, so for overlapping balls `max` keeps the first
one, which need not have the largest upper bound. Overlaps do occur: the μ = ±1 rows are mathematically equal by
conjugation symmetry. The error is at most one ball radius (10⁻⁹⁰ relative), far inside every margin.

R7 reruns the lemma with the exact maximum of the upper bounds: the same F, Lipschitz constant and conclusion. No pair
of entries overlaps without having equal upper bounds.

**Fix:** use `arb(max(x.upper() for x in ...))`. The same applies to `residual_poly`'s `worst`, which is only printed.

## 2. The fixed-point lemma (step i)

**2a. Integral equation.** By hand: with z = ξ², D = ξP, S = zQ, V = Q − (1+s)P and W = V + [(C−1)Q], equations
(I1)–(I3) become exactly the (P, V, C) system in the docstring of `fixed_point.py`. Only V/(2z²) is singular at rank
one. The integral equation solves η′ = ρ + F(Y_N + η) − F(Y_N), with the principal part inverted by the kernel
$`e^{1/(2t)-1/(2z)}`$.

As polynomial identities, for any P, Q, C, the residual numerators relate to the ξ-form residuals by

```math
r_1 = -z\,N_p, \qquad r_2 = -\xi\,\big(N_v + (1+s)\,z\,N_p - [C\,N_p]\big), \qquad r_3 = -2z\,N_c .
```

R1 verifies these identities exactly in GF(p). The [C N_p] term vanishes for the series below order 53 anyway.
**Verdict: OK.**

**2b. Kernel bound.** On the ray arg t = arg z = θ with |t| ≤ |z|:

```math
\mathrm{Re}\Big(\frac{1}{2t}-\frac{1}{2z}\Big) = \frac{\cos\theta}{2}\Big(\frac{1}{\lvert t\rvert}-\frac{1}{\lvert z\rvert}\Big) \le 0 \quad (\cos\theta \lt 0),
```

so the kernel's modulus is $`e^{-a(1/|t|-1/|z|)}`$ with a = −cos θ/2 = √2/4 at θ = ±3π/4.
Over the sector where |θ ∓ 3π/4| ≤ 2d, the smallest a is $`\tfrac{\sqrt2}{4}(\cos 2d-\sin 2d)`$, at θ = ±(3π/4 − 2d). That is
`set_sector`'s value: 0.18796 for d = 0.2. R2 checks the identity at 2000 random points to 10⁻²⁶. **Verdict: OK.**

**2c. Inverse-operator bounds.** For m ≥ −2, $`\rho^m \le r^{m+2}\rho^{-2}`$ and
$`\int_0^r e^{-a(1/\rho-1/r)}\rho^{-2}\,d\rho = 1/a`$, so the fast bound is r^{m+2}/a. It is an equality at m = −2. The
slow bound is exact. In `intbound` the weighted sup is attained at r = R, because the exponents q + 1 − k and
q + 2 − k are ≥ 0. Quadrature (R2) confirms all of this, for m from −2 to 60, a = √2/4 and 0.188, and five spot
checks of `intbound` against a brute-force sup. **Verdict: OK.**

**2d. Lipschitz majorants and the weighted norm.**

- `MD` carries majorants of $`|\partial F_i/\partial\eta_j|\,|t|^{k_j}`$, built over the whole ball by the product
  rule and the triangle inequality. Each L_ij is that majorant pushed through the inverse operator.
- The norm is the weighted max norm $`\max_j w_j\lVert\eta_j\rVert`$. Its Lipschitz constant is
  $`\max_i\sum_j w_iL_{ij}/w_j`$, and the ball is ‖η_j‖ ≤ ρ/w_j. These are consistent.
- The c components carry weight r^{k+1} everywhere: `kw`, the ball, the unit derivative, `intbound` and the start
  values.
- The uniqueness argument restricts to (0, R′], so every bound must be non-decreasing in R. They are, by
  construction, and R7 shows the Lipschitz constant rising monotonically from 0.121 at r₀ = 0.06 to 0.292 at 0.1, with
  N and the vanishing orders fixed.

**Verdict: OK.**

**2e. Vanishing below order 53.** `tail_maj` drops every residual coefficient below kmin. kmin is read off at the box
centre as the first coefficient ball that does not contain 0. On a parameter box, the code only checks that the
dropped balls contain 0, which is nearly vacuous for wide balls. So soundness rests on those coefficients being
identically zero as polynomials in (A, ω).

R1 proves it. It redoes the series and the residuals exactly in GF(p), with p ≈ 2⁶¹ and p ≡ 1 (mod 4) so that i
exists, at three random (A, ω):

- the (P, V, C) numerators start exactly at orders [53, 53, 53, 53, 52, 52, 52], which are `fixed_point`'s kmins;
- the coefficient at kmin is nonzero;
- the ξ-form residuals start at ξ¹⁰⁸, ξ¹⁰⁷ and ξ¹⁰⁶, so the recursion solves (I1)–(I3) exactly through order N − 1;
- every coefficient is a polynomial in (A, ω) of degree below 300, so by Schwartz–Zippel the chance of a false pass
  is below 10⁻⁵⁰.

**Verdict: OK.** **Fix (cosmetic):** state the algebraic reason in §4.4, that the recursion solves the order-by-order
equations exactly so every coefficient below N vanishes identically, and keep R1 as a regression test.

**2f. Identification of y⁺.**

- y⁺ is the 1-sum (Braaksma). On the ray, y⁺ − Y_N = O(|z|⁵³), and 53 > k + 1 = 41. So on (0, R′] with R′ small,
  y⁺ − Y_N lies in the ball, and the contraction (monotone in R) makes it the fixed point there.
- ODE uniqueness then extends the equality to (0, R]. y⁻ follows by the conjugation symmetry, which I checked on the
  equations: with real A and ω, (y_μ, c_ν) ↦ (conj y₋μ, conj c₋ν), ξ ↦ ξ̄ maps solutions to solutions and fixes the
  formal series.
- Independent check (R5): start from `lateral.py`'s series at |ξ| = 0.06 and 0.07 (truncation errors 10⁻⁷⁴ and
  5×10⁻⁵⁸) and integrate outward at 70 digits. Both land inside the step-(i) enclosure, within 5.3×10⁻³⁵ of its
  midpoint (radius 6×10⁻³⁴), and agree with each other to 5×10⁻⁵⁹.

**Verdict: OK.**

**2g. Derivatives at ξ₀ (§4.5).**

- Y_N and its parameter derivatives are exact jets over the box.
- η is holomorphic in (A, ω), as a uniform limit of holomorphic iterates. For each real x in X, the disk of radius c
  around it lies in the complex box X + c(1+i), so $`|\partial\eta| \le \sup|\eta|/c`$ holds. The S component
  includes |1+s| over the complex box.
- With c = 10⁻³, this leaves a floor of about 4×10⁻⁹ on the derivative radius (R7), which is negligible next to the
  final 6×10⁻⁴.

**Verdict: OK.**

## 3. The Lohner step (step ii)

**3a. Remainders.**

- **A-priori enclosure.** Picard: if X + [0, h]·F(Ỹ, ξ(T))ξ′ ⊆ Ỹ with T = t + [0, h] as a ball, then every
  trajectory from X stays in Ỹ. This is Schauder's theorem on a convex box; uniqueness follows from the Lipschitz
  right-hand side.
- **Taylor remainder.** In integral form the remainder is $`h^{N+1}\int_0^1 (N+1)(1-s)^N c_{N+1}(t+sh)\,ds`$. Its
  weight is non-negative and integrates to 1, so it lies in h^{N+1} times the convex hull of c_{N+1} over the path,
  and so in h^{N+1} c_{N+1}(Ỹ, T).
- `taylor_Y(…, T, Ỹ, N+1)` makes N + 2 Picard passes on series of length N + 2, so coefficient N + 1 is exact.
- **Variational remainder.** V(τ + σ) = Φ(τ + σ, τ)V(τ), so the (N+1)-st coefficient is Φ_{N+1}(Ỹ, T)·V(τ), with
  V(τ) ∈ Ṽ by `apriori_V`'s inclusion. The code computes `VR[N+1] * Vt`.

**Verdict: OK.**

**3b. Steps.** Every t is a multiple of 2⁻³⁰, so `t + hs` is exact, and the last step is `1 - t`. The ray ends and the
arc starts at the same exact point r₁e^{3iπ/8}, computed as two balls that both contain it. Step control (halve on
failure, grow by 5/4) has no bearing on rigor. **Verdict: OK.**

**3c. Update.**

- **Mean value form.** φ(c + Qq) ⊆ φ(c) + J·Qq, with J enclosing Dφ over the box X = `box()`. The segment from c to x
  stays in X because q ∋ 0, an invariant: the start residuals contain 0, e = φ(c) − mid φ(c) ∋ 0, and products and sums
  preserve it.
- **The new set.** Q′ is any exact invertible matrix, `Qn.inv()` encloses its inverse, and
  q′ = (Q′⁻¹JQ)q + Q′⁻¹e gives c′ + Q′q′ ⊇ φ(c) + JQq.
- **Real coordinates.** These are the R-linear realification of the complex (holomorphic) Jacobian. The imaginary
  parts of ω and A stay at 0.

**Verdict: OK.**

**3d. Variational columns.**

- **True derivatives.** ∂ₓy(t + h; x) = Dφ(y(t; x))·∂ₓy(t; x). y(t; x) lies in the state set, J ⊇ Dφ there, so the
  update v′ ∈ J c_V + (JQ_V)q_V contains it. Subdistributivity covers J(a + b).
- **Start.** The columns start as (∂y/∂A, 0, 1) and (∂y/∂(ω/ω∞), ω∞, 0) in the rows (y, ω, A). J contains ∂/∂ω through
  `Sys.lin`'s dω terms, which I checked term by term against the derivative of `Sys.rhs`.
- **State set over the box.** It is rebuilt as y(t; x̂) + V(t; X)(x − x̂). This is valid because V encloses ∂ₓy on
  the convex box X.

**Verdict: OK.**

**3e. Path and chain rule.** On the ray, ξ = (r₀ + (ξ₁ − r₀)t)e^{iφ₀} and ξ′ = (ξ₁ − r₀)e^{iφ₀}. On the arc,
ξ = ξ₁e^{iφ₀(1−t)} and ξ′ = −iφ₀ξ. The series form is `x0 * exp(-i phi0 tau)`, and the a-priori form is `dxi(xi)`
with ξ a ball over T.

R3 tests one step on each leg at Taylor orders 6 and 24. A 50-digit `odefun` solution, with an independently written
right-hand side (`lateral.py`) and path, lies in every Lohner box. J contains the central-difference Jacobian columns,
including the ω column.

Both negative controls fail, as they must:

- with the remainder removed, the order-6 box misses by 4×10⁻²² against a radius of 2×10⁻⁷⁷;
- with the sign of the arc factor flipped, the reference falls outside the box.

**Verdict: OK.**

**3f. Tightness (not a soundness issue).** The remainder bound is very pessimistic. In R3 the order-6 radius is
1.4×10⁻⁷ against a true truncation error of 4×10⁻²², and the order-24 radius is 3×10⁻²⁸. The
(N+1)-st coefficient is evaluated by interval Picard over the whole a-priori box, and wraps there. (At order 24
the reference lies 4×10⁻⁵² from the box midpoint.)

The point run's final radius of 3×10⁻¹⁹ is therefore about 10¹² times its true error: R8 puts the 30-digit
reference 10⁻³¹ from the midpoints. This does not matter for this certificate, whose bottleneck is J(X). If point
accuracy ever binds, use a narrower a-priori enclosure (smaller inflation, or a first-order enclosure around the Taylor
polynomial) or a smaller h with a higher order.

## 4. The Krawczyk test (step iv)

**4a. The test.**

- `krawczyk.krawczyk` computes K(X) = x̂ − Y g(x̂) + (I − Y J(X))(X − x̂), with X − x̂ ⊆ [−h, h]² taken from
  `h.upper()`.
- J(X) comes from the box run over x̂ ± h, which contains X, for every x in it.
- Y is a fixed exact real matrix.
- Interior containment uses arb's certain `<` against the decimal centre and h = 10⁻⁹. x̂ = arb("2.5421480475") is a
  ball, so the comparison is with the exact decimal.
- K(X) ⊂ int X then gives a unique zero, and it is simple (Krawczyk 1969; Neumaier 1990).

R4 recomputes K(X) from the saved runs with its own code and its own evaluation of N(x̂). It uses two different
preconditioners, the float inverse of mid J(X) and a 4-digit rational inverse. The margins are 7.9247×10⁻¹⁰ and
9.2466×10⁻¹⁰ with the first, and 7.9238×10⁻¹⁰ and 9.2441×10⁻¹⁰ with the second.

The test still passes with the J(X) radii multiplied by 8, and with the g(x̂) radius multiplied by 3×10⁷. With the
centre moved by 3h along A, it fails, as it must. **Verdict: OK.**

**4b. g = κ J_D.**

- κ = ξ₁^{−Â²}/N(x̂) is one complex constant. Its ball encloses the exact κ, and the same ball multiplies both g(x̂)
  and J(X), so both enclose the values of one fixed function.
- κ ≠ 0, so g has the zeros of J_D(ξ₁). The Jacobian needs no derivative of N or of ξ₁^{−A²}.
- For real x, ∂ conj(D) = conj(∂D).
- The columns are ordered (A, ω/ω∞) consistently: `kbox_job` stores column j of V, which `g_from_states` reads as
  ∂/∂x_j.

**Verdict: OK.** Whether J_D(ξ₁) = 0 is *the selection condition* is §6a.

**4c. The quoted zero (F3, cosmetic).**

- §9 of `A_certify.md` gives A = 2.542148047617 ± 9.1×10⁻¹¹. The commit message gives ± 9×10⁻¹¹.
- K(X)_A is [2.5421480475256, 2.5421480477075], whose lower end lies 4×10⁻¹³ below 2.542148047617 − 9.1×10⁻¹¹.
  The midpoint was rounded up, but the radius was not enlarged to match.
- The ω/ω∞ enclosure 0.7769940136097 ± 6.6×10⁻¹¹ does contain K(X).

**Fix.** Quote the interval endpoints, or better the tighter box the same runs give. J(X) over the 10⁻⁹ box also
encloses Dg over any sub-box, and K(X′) ⊂ int X′ already holds for X′ = x̂ ± 1.3×10⁻¹⁰ (R4). So:

```math
A \in [2.54214804760473,\ 2.54214804762839], \qquad \omega/\omega_\infty \in [0.776994013601194,\ 0.776994013618254],
```

that is, A = 2.5421480476166 ± 1.2×10⁻¹¹ and ω/ω∞ = 0.7769940136097 ± 8.6×10⁻¹² (endpoints rounded outward).

## 5. The linear check and the conventions of lateral_np.py

- **The linear identity.** `lin_check.py` gives J/(2πi K′ f 2F₁) = 1 ± 3×10⁻³⁰ at ξ₁ = 0.2 and 0.3 (rerun). It
  evaluates both lateral solutions from arb's Kummer U, so it checks the connection formula behind the normalization.
- **Orientation.** For arg ξ slightly positive, x = −1/(2ξ²) has arg x → π⁻, the upper side of U's cut, which is
  where arb's principal value is continuous. So y⁺ ↔ `U_upper`, and y⁻ is the Schwarz reflection
  conj U(ā, b̄, ·). Two checks confirm it:
  - The validated linear run (`validated_int.py linear`, rerun: R₁ = 1 ± 4×10⁻¹⁷) comes down from arg ξ = 3π/8 and
    lands on `lin_check`'s D⁺, overlapping it.
  - With gravity, at x̂: the point run, `lateral_np.py` (§9.3: 9×10⁻¹⁶) and the mpmath reference of R8 (10⁻³¹) all
    give J_D = 1.0547×10⁻¹² − 3.4904×10⁻¹²i.
- **Normalization.** R4 evaluates N(x̂) at ξ₁ = 0.3 independently: powers by exp/log, 1/Γ by `rgamma`, and
  U(1−α, 3/2, X) written out. It overlaps `krawczyk.normalization`. `lateral_np`'s double-precision N, with f obtained
  by ODE integration, agrees to 8.7×10⁻¹².
- **Signs.** J_μ = y⁺_μ − conj(y⁺_{−μ}) everywhere, F₁ = −iA/2 everywhere (`Fharm`, `series_arb`, `fixed_point`,
  `validated_int`, `krawczyk`), and K′ = 2^{−s−1/2}/(Γ(α)Γ(β)) everywhere.

**Verdict: OK.**

## 6. Hidden assumptions

### 6a. One component at one point is not the whole jump (fixable; a missing lemma)

`A_certify.md` §2 defines the Stokes jump $`J_\mu = y^+_\mu - \overline{y^+_{-\mu}}`$ (y = Dd) and the selection
condition "J₁ = 0", that is, y⁺ ≡ y⁻. The certificate proves J₁(ξ₁) = 0 at the single point ξ₁ = 0.3. §5.1 and §9
then treat the two as the same, which is not shown.

Why the step is plausible:

- On the real axis, ψ = y⁺ − y⁻ is exponentially small and anti-real ($`\psi_\mu = -\overline{\psi_{-\mu}}`$).
- So it lies in a real-two-dimensional family, one complex Stokes constant Σ, spanned by the μ = ±1 fast modes.
  Their D components at ξ₁ are about f(ξ₁) ≠ 0 for μ = +1, and small for μ = −1 (by nonlinear coupling only).
- J₁(ξ₁) is two real numbers, and the map Σ ↦ J₁(ξ₁) is close to Σ ↦ f(ξ₁)Σ, which is invertible.

Numerically (R6, double precision) every component of the jump vanishes at the certified zero. At ξ₁ = 0.25 and 0.3,
each component predicts its own zero:

- D₊₁ and S₊₁ give x̂ + (1.16–1.17×10⁻¹⁰, 0.93–0.97×10⁻¹¹);
- C₊₂ gives x̂ + (1.2–1.4×10⁻¹⁰, 0.3–1.4×10⁻¹¹), less accurately, because its jump is 30 times smaller.

All of these lie inside K(X). At ξ₁ = 0.3 the single real C₀ equation is 5×10⁻¹⁵ at the certified zero, against
1.7×10⁻¹³ for a 10⁻¹⁰ move.

**Fix.** This is the template's step 4, which the M = 1 pipeline skipped ("(iii) … there is no second integration"):

- In a domain reaching the real axis, write ψ = y⁺ − y⁻ through a contraction for the difference equation,
  $`\psi = \Sigma\,\phi_{+1} + \overline{\Sigma}\,\phi_{-1} + O(\delta)\,\psi`$.
- Bound the D₊₁ components at ξ₁ so that |(φ₊₁)_{D,+1}(ξ₁)| > |(φ₋₁)_{D,+1}(ξ₁)| plus the correction.
- Then J₁(ξ₁) = 0 ⇔ Σ = 0 ⇔ y⁺ ≡ y⁻.

Until then, state the result as a zero of J₁(0.3), as in the Verdict.

### 6b. y⁺ on the real axis depends on the path beyond |ξ| ≈ 0.3 (fixable; a missing lemma)

§2 defines y⁺ as the solution asymptotic to the series in a sector containing the ray *and the real axis*, a germ at
ξ → 0. The code continues it from ξ₀ along the ray to 0.3 and the arc at 0.3. The two agree on the real axis only if
y⁺ is analytic in the region the path sweeps past, 0 < |ξ| ≤ 0.3 with 0 ≤ arg ξ ≤ 3π/8.

R6 shows the question is not academic:

- **At ξ = 0.35.** y⁺ continued along the arc at 0.3 and then along the real axis, and y⁺ continued along the arc
  at 0.35, differ by 82, with |y⁺| = 1.2 against 82.
- **Size of y⁺ on the arcs.** On the arc at 0.30, |y⁺| reaches 8.4; at 0.32, 27; at 0.34, 81.
- **So there is a singularity.** A movable branch point of y⁺ lies between radii 0.30 and 0.35 in the upper sector,
  and on the arc at 0.35 the "jump" at x̂ is O(10²).
- **Below 0.3 the paths agree.** The arc at 0.2 followed by the real axis gives the same y⁺(0.3) as the arc at 0.3,
  to 2.2×10⁻¹², the solver's accuracy.
- **What it explains.** It also accounts for the large solution near 0.3e^{3iπ/8} (|S₋₁| ≈ 6.8, §9.4) that drives the
  derivative excess.

So ξ₁ = 0.3 is numerically legitimate, but it sits near the edge.

**Fix.**

1. State the theorem for the path, as in the Verdict.
2. To reach the germ statement, prove that y⁺ is analytic on the swept region:
   - **Away from 0.** Run the C⁰ validated integration with the arc radius as an interval parameter,
     r ∈ [r_min, 0.3]. The paths then sweep the annular sector, and finite enclosures prove analyticity there.
   - **Near 0, for |ξ| ≤ r_min.** Use a step-(i) domain that reaches the real axis (the template's D^± domains, with
     the fast mode integrated from the ray rather than from 0).
3. Once the derivative budget allows, prefer ξ₁ ≤ 0.25 or the lower path (`--rho=0.2`), away from the singularity.

### 6c. The system is the one the claims are about: OK

`series_arb`/`fixed_point.series_jet`, `fixed_point`'s (P, V, C) system (R1 identities), `validated_int.Sys` (`rhs`
and `lin`) and `lateral.py`/`lateral_np.py` are the same M = 1 projected system. Several checks confirm it:

- R1: the recursion solves the ξ-form exactly;
- R3: one step agrees with `lateral.py`'s right-hand side to 50 digits;
- R8: the whole point run agrees with `lateral.py`'s mpmath integration to 10⁻³¹.

README (§ inner selection) and the paper (§ computations with gravity) call it "the one-harmonic selection condition".
They say "a zero of the multiplier", not a zero of the physical problem, and they note that it is not the physical
solution. That is accurate, except for 6a and 6b: "has exactly one zero" should carry the qualification of the
Verdict.

### 6d. ω∞: OK (cosmetic note)

Every file uses ω∞ = 2π/1.64389375, built from the decimal string. `fixed_point` works at 320 bits, `validated_int`
at 256, `krawczyk` at 320, and `lateral_np` in double precision. This is a convention: 1.64389375 is a numerical
estimate of Δ∞. The convention-free result is ω itself, ω ∈ 2.969776709 ± 7.1×10⁻¹⁰ from K(X), or ± 2.6×10⁻¹¹ from
the tighter box of §4. Quote it next to ω/ω∞.

### 6e. "± 10⁻⁹": OK

It is the half-width in A and in ω/ω∞ (not ω), in the max norm. The box run covers Â ± 10⁻⁹ and ω̂/ω∞ ± 10⁻⁹, plus
the radius of the decimal ball. Krawczyk's interior test is against the same h.

### Cosmetic items

- `certify_M1.py`'s docstring still says "Missing for a certificate: validated integration (step ii)".
- `Sys.lin`'s docstring says "V[-1] = dw". The code reads dω from index nd = 7; V[−1] is dA, which the right-hand side
  does not depend on.
- §5.2 says the start radius is 6×10⁻³⁴, and §9.1 says "about 5×10⁻³²". The first is the value radius; the second is
  the η bound (`eta0`, 5.2×10⁻³²). Say which is which.

## 7. Experiments

All scripts are in `solver/proof/review/`. Every one is light: seconds, except R5 (2 min) and R8 (about 4 min), on one
core.

| script | what it checks | result |
|---|---|---|
| R1 `rv_residual_modp.py` | exact GF(p) series and residuals: vanishing orders, the (P,V,C) ⇔ ξ-form identities | kmins [53,53,53,53,52,52,52] exact; identities hold |
| R2 `rv_kernel_bounds.py` | kernel modulus, sector rate, fast/slow operator bounds, `intbound` vs brute force | all hold (m = −2 is an equality) |
| R3 `rv_taylor_step.py` | one Lohner step (ray, arc; order 6 and 24) vs 50-digit odefun; J vs central differences; 2 negative controls | contained; controls fail as required |
| R4 `rv_krawczyk_recheck.py` | JSON round trip (20 000 balls); K(X) recomputed independently; tighter box; robustness | 0 failures; margins 7.92×10⁻¹⁰, 9.25×10⁻¹⁰; X′ = x̂ ± 1.3×10⁻¹⁰ passes; F3 found |
| R5 `rv_start_values.py` | step-(i) enclosure vs 70-digit integration from \|ξ\| = 0.06, 0.07 | inside; 5×10⁻³⁵ from the midpoint |
| R6 `rv_jump_xi1.py` | (double precision) every jump component's zero at ξ₁ = 0.2–0.35; path dependence | all components vanish at the certified zero (0.25, 0.3); singularity between 0.30 and 0.35 |
| R7 `rv_fixed_point_audit.py` | rigorous `max`; monotonicity in R; growth of the enclosures with the box | unchanged; monotone; grows linearly |
| R8 `rv_point_run_mp.py` | the whole point run vs 30-digit mpmath from \|ξ\| = 0.07; derivative quotients vs the box run | inside, 10⁻³¹ from the midpoints; all four difference quotients inside the box-run enclosures |

## 8. Reproduce

```
cd solver/proof/review
python3 rv_residual_modp.py        # R1, < 1 s
python3 rv_kernel_bounds.py        # R2, ~10 s
python3 rv_taylor_step.py          # R3, ~30 s
python3 rv_krawczyk_recheck.py     # R4, ~10 s (reads ../validated_int_krawczyk_1e-9.json)
python3 rv_start_values.py         # R5, ~2 min
python3 rv_jump_xi1.py             # R6, ~1 min (double precision, not rigorous)
python3 rv_fixed_point_audit.py    # R7, ~10 s
python3 rv_point_run_mp.py         # R8, ~4 min
```

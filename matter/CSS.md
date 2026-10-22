# Phase-rotating CSS critical collapse at large D: the axion–dilaton test

This page tests the large-D mechanism of [GAMMA.md](../GAMMA.md) (Proposition 5: an exact λ = 2 mode from the
s-translation of the D = ∞ inner heat system, hence γ∞ = 1/2) on a second family of critical solutions: the
continuously self-similar (CSS), phase-rotating critical solutions of the Hirschmann–Eardley κ-model, whose κ = 1 member
is the axion–dilaton system. These are spherically symmetric scalars with no centrifugal 1/r² term, the class in which
[LITERATURE.md](LITERATURE.md) §1 and [STRUCTURE.md](STRUCTURE.md) §7 predict that the mechanism survives. Compiled in
October 2026.

Conventions as in the README and LITERATURE.md: D = n + 2, x = r/(−t), τ = −ln(−t), perturbations ∝ $`e^{\lambda\tau}`$,
**γ = 1/λ is the radius exponent** ($`r_{\rm BH}\propto(p-p_*)^\gamma`$), the mass exponent is (D − 3)γ, and the two
coincide in D = 4. The phase-rotation frequency ω is the one of $`F\propto(-t)^{i\omega}`$; it is invariant under
t → kt, and it is the same ω = 2π/Δ convention as the real scalar's (the field is one harmonic of θ = ωτ).

**Result in brief.**

- **Validation at D = 4.** The κ-model equations in D dimensions are checked in sympy against the Einstein tensor
  computed from the metric and against the published D = 4 forms. At D = 4 the axion–dilaton CSS solution reproduces
  Hamadé, Horne and Stewart to their quoted precision (ω = 1.1769527242, z₊ = 2.6090934730). Its unique growing mode
  gives γ = 0.2641066000 (HHS: 0.2641066). The free complex scalar reproduces Hirschmann–Eardley (ω = 1.9154446899,
  γ = 0.3871055, and their second, oscillatory unstable mode).
- **Continuation in D.** The solution was followed to D = 80 by two independent methods: collocation, reliable to
  D ≈ 8, and complex-path shooting for the whole range. The latter imposes analyticity at the self-similarity
  horizon, which is what selects ω; the first-order regularity condition is automatic there. γ(D) rises monotonically,
  0.2641, 0.3394, 0.3758, 0.4120, 0.4304, 0.4636, 0.4804 at D = 4, 5, 6, 8, 10, 20, 40, and 0.4896 at D = 80.
  The published γ(d) for d = 5–10 [AGH12] is not reproduced: those solutions are not analytic at the horizon (§3).
- **Uniqueness.** At every D checked the only unstable mode besides the exact symmetry modes (λ = 1 time translation,
  λ = 0 phase, λ = ±iω target isometries) is the relevant one. This rests on the full collocation spectra at D = 4–8
  and on winding numbers of the growth-rate determinant at D = 4, 10, 20, 40, in boxes up to Re λ = 4.5–6 and
  |Im λ| = 1.2–2.9.
- **Extrapolation.** λ = 1/γ in powers of 1/n: λ∞ = 2 to about 10⁻⁵, with λ = 2 + 3.60/n − 27/n² + …. **So γ∞ = 1/2**,
  as for the real scalar (γ = 1/2 − 0.90/n here, against 1/2 − 0.758/n there).
- **Mechanism.** The solution has the real scalar's large-D structure. Gravity is weak inside the light cone
  ($`x_h \approx 1 + 2.4/n`$), the field is O(1) and varies on x ~ √n beyond it, and there the equations become an autonomous
  heat-type system in s = r²/(2n) and t, with the target curvature as an O(1) term. So the r²-shift symmetry of
  Proposition 5 holds for every κ, which explains γ∞ = 1/2.
- **The D = ∞ selection rule, tested without truncation.** The Stokes zero of the D = ∞ inner problem gives ω∞; it is
  compared with the extrapolated finite-D values. For the axion–dilaton, ω∞ = 3.1998939 (inner) against 3.1994
  (extrapolated). For the free complex scalar, 3.9789695 (STRUCTURE.md, reproduced here with different code) against
  3.9784. They agree to 1–2×10⁻⁴ in ω and 3–5×10⁻⁵ in |f(0)|∞ (relative; §7).
- **Clark–Pimentel.** Their γ_r → 0 concerns a different solution: a real scalar with no phase rotation, no regular
  centre and no analytic horizon (§9). It limits the scope of the rule, not its content.

Contents

- [1. Model and equations in D dimensions](#1-model-and-equations-in-d-dimensions)
- [2. Methods](#2-methods)
- [3. Regularity at the horizon is analyticity, and why the published D ≥ 5 values disagree](#3-regularity-at-the-horizon-is-analyticity-and-why-the-published-d--5-values-disagree)
- [4. Validation at D = 4](#4-validation-at-d--4)
- [5. Results: ω(D), λ(D), γ(D)](#5-results-ωd-λd-γd)
- [6. Uniqueness of the unstable mode](#6-uniqueness-of-the-unstable-mode)
- [7. Extrapolation to D = ∞](#7-extrapolation-to-d--)
- [8. How the solution scales with D](#8-how-the-solution-scales-with-d)
- [9. Clark–Pimentel and the scope of the rule](#9-clarkpimentel-and-the-scope-of-the-rule)
- [10. Verdict](#10-verdict)
- [11. Open issues](#11-open-issues)
- [12. Scripts and data](#12-scripts-and-data)
- [13. References](#13-references)

## 1. Model and equations in D dimensions

**Model** (HE97 eq. (4), written in D dimensions; G = 1 in their units):

```math
S = \int d^Dx\,\sqrt{-g}\,\Big[R - \frac{2\,\lvert\nabla F\rvert^2}{(1-\kappa\lvert F\rvert^2)^2}\Big] ,
\qquad
R_{ab} = h\,(F_a\bar F_b + \bar F_a F_b), \qquad
\Box F = -\frac{2\kappa\bar F\,(\nabla F\cdot\nabla F)}{1-\kappa\lvert F\rvert^2}, \qquad h = (1-\kappa\lvert F\rvert^2)^{-2} .
```

F is complex and the target is a disc of constant negative curvature (−2κ for the target metric
$`2\lvert dF\rvert^2/(1-\kappa\lvert F\rvert^2)^2`$ read off from S). κ = 1 is the axion–dilaton system: with
$`\tau = i(1-F)/(1+F)`$, Hamadé, Horne and Stewart's Lagrangian $`R - \tfrac12\lvert\partial\tau\rvert^2/({\rm Im}\,\tau)^2`$ [HHS96] is
exactly this one (checked in sympy). κ = 0 is the free complex scalar, with $`F = \sqrt{4\pi}\,\phi`$ for the field φ of
HE95a (their $`G_{ab} = 8\pi T_{ab}`$). The trace-reversed Einstein equations $`R_{ab} = h(F_a\bar F_b + {\rm c.c.})`$ have no
explicit D, so the coupling is the same in every D. There is no potential and no 1/r² term: the angular derivatives of
a spherically symmetric scalar vanish.

**Spherical symmetry.** With $`ds^2 = -\alpha^2dt^2 + a^2dr^2 + r^2d\Omega_n^2`$, $`\Phi = F_r`$, $`\Pi = (a/\alpha)F_t`$ and
$`\rho = 1-\kappa\lvert F\rvert^2`$:

```math
\partial_t(r^n\Pi) = \partial_r\Big(r^n\frac{\alpha}{a}\Phi\Big) + \frac{\alpha}{a}\,r^n\,\frac{2\kappa\bar F}{\rho}\,(\Phi^2-\Pi^2),
```
```math
\frac{a_r}{a} = \frac{r}{n}\,h\,(\lvert\Pi\rvert^2+\lvert\Phi\rvert^2) - \frac{(n-1)(a^2-1)}{2r}, \qquad
\frac{\alpha_r}{\alpha} - \frac{a_r}{a} = \frac{(n-1)(a^2-1)}{r}, \qquad
\frac{n\,a_t}{r\,a} = h\,(F_t\bar F_r + \bar F_t F_r) .
```

**CSS ansatz** [HE95a, EHH95]: $`F = (-t)^{i\omega}f(x)`$, α = α(x), a = a(x). This is the "elliptic" class: the field
rotates about a fixed point of the target (for κ > 0 any rotation is conjugate to this one under SL(2,R)). With
$`b = \alpha/a`$, $`u = a^2-1`$, $`q = f'`$, $`p = (i\omega f - xq)/b`$ (HE95a's p) and $`\Delta = b^2-x^2`$:

```math
\frac{x\,b'}{b} = (n-1)\,u, \qquad
u = \frac{2xh}{n(n-1)}\Big[x\,(\lvert p\rvert^2+\lvert q\rvert^2) + 2b\,{\rm Re}(p\bar q)\Big],
```
```math
\Delta\,f'' = q\Big[2x(1-i\omega) - (n-1)ux - \frac{b^2}{x}\big(n+(n-1)u\big) - \frac{4i\kappa\omega x\lvert f\rvert^2}{\rho}\Big]
- f\big[\omega^2 + i\omega\big(1-(n-1)u\big)\big] - \frac{2\kappa\bar f}{\rho}\big(\Delta q^2 + \omega^2f^2\big) .
```

The algebraic u is the combination of the Hamiltonian and momentum constraints. Boundary conditions: f′(0) = 0 and
b(0) = 1 (α = 1 on the axis, so t is central proper time), Im f(0) = 0 (the U(1) phase), and at the self-similarity
horizon (the past light cone of the singularity) $`b(x_h) = x_h`$, with f analytic there. The unknowns are ω, |f(0)| and
$`x_h`$; the solution is isolated.

**Checks** (`css_equations.py`, sympy, all residuals 0):

- the Einstein tensor of the warped metric is computed from explicit sphere coordinates in D = 4, 5, 6 and agrees
  with the general-n formulas used above, and so does □F;
- the matter equation is the Euler–Lagrange equation of S (F and F̄ independent), and HE97's (8a)–(8b) hold;
- the three constraints are the tt, rr + tt and tr Einstein equations;
- the first-order system used by the numerics (`css_model.pde_terms`, in τ and $`w = (x/x_h)^2`$) is a nonzero multiple of
  the covariant equations, term by term;
- at n = 2 the CSS ODEs are HE95a's eqs. (17a)–(17d) (κ = 0) and HE97's eqs. (22a)–(22c) (all κ) as printed;
- for all n and κ = 1, $`b' = (n-1)bu/x`$ with the algebraic u is the first equation (12) of Álvarez-Gaumé and Hatefi
  [AGH12] (d = n + 2). Their eq. (11), $`u = -zb'/((d-3)b)`$, has the opposite sign to ours and to HE95a's D = 4 form
  $`b' = bu/z`$; as the same group's later paper [AH20a] writes $`u = zb'/((q-1)b)`$, we take it for a sign slip in print.

**Perturbations.** $`\psi = f + \epsilon e^{\lambda\tau}A(x)`$, $`\bar\psi = \bar f + \epsilon e^{\lambda\tau}B(x)`$ (A and B
independent: the real system, complexified), $`b \to b+\epsilon e^{\lambda\tau}\delta b`$, $`u\to u+\epsilon e^{\lambda\tau}\delta u`$.
The linearized equations are the wave equations, the Hamiltonian constraint (an ODE for δu, with δu = O(x²)) and the
slicing condition with the gauge δb(0) = 0 (α = 1 on the axis). The momentum constraint is not imposed. On the
collocation eigenvectors (`css_momcheck.py`, D = 4–6) its relative residual is 10⁻¹¹–3×10⁻⁹ for the relevant mode,
λ = 1 and the decaying modes, against 10⁻⁶–10⁻³ for spurious high-frequency eigenvectors. For the ±iω modes the metric
perturbation vanishes identically, so the check is empty there. Four growth rates are exact and known in advance:

| λ | Mode |
|---|---|
| 0 | U(1) phase rotation = the homothety t → kt (on a CSS solution these are the same) |
| 1 | time translation t → t + ε (the shift of the accumulation time; "gauge" for γ) |
| ±iω | target isometries: $`\delta F = c - \kappa\bar cF^2`$ (SL(2,R) boosts for κ > 0, shifts F → F + c for κ = 0) |

HE97 list the same four ("σ = −i, ±ω", with λ = iσ in their convention). The relevant mode is the one other real λ > 0.

## 2. Methods

Three independent pieces of code:

1. **Collocation** (`css_model.py`, `css_bg.py`, `css_scan.py`). Chebyshev–Gauss–Lobatto collocation in $`w = (x/x_h)^2`$
   mapped by $`w = (e^{A\zeta}-1)/(e^A-1)`$ (A = 2–5 clusters nodes at the centre, where the solution has a complex
   singularity close to the real axis: the point where κ f f̄ = 1 in the complexified x-plane, at x ≈ 0.25i in D = 4).
   Every equation is collocated at every node, so regularity at both regular-singular ends comes from the polynomial
   basis. The background is found by Newton (Jacobian by complex step); the perturbation operator gives a generalized
   eigenproblem $`J\,v = \lambda B\,v`$ whose full spectrum is computed at once.
2. **Complex-path shooting** (`css_lateral.py` for the background, `css_pert.py` for perturbations). Shoot from the
   centre with (ω, |f(0)|); treat f and f̄ as independent analytic functions; integrate along the real axis to
   $`x_h - R`$ and then along the upper and the lower half-circles $`\lvert x-x_h\rvert = R`$ to $`x_h + R`$. An analytic solution
   arrives at the same values; a solution with an irregular part $`c\,(x_h-x)^\mu`$ arrives with the jump
   $`c\,(e^{i\pi\mu}-e^{-i\pi\mu})R^\mu+\dots`$. The background is the zero of the complex jump J(ω, |f(0)|) (R = 0.85 x_h;
   the root does not move between R = 0.3 x_h and 0.95 x_h, to 10⁻¹¹). For perturbations the two jumps of (A, B) are linear
   in the centre data (A(0), B(0)), $`J = M(\lambda)\,(A(0),B(0))^T`$, and the growth rates are the zeros of the entire
   function det M(λ). Horizon values ($`x_h`$, $`f(x_h)`$, $`u_h`$) come from Cauchy integrals around the circle. This is the
   method of `solver/inner_stokes/lateral_np.py` and `struct_css.py`, applied at finite D.
3. **The D = ∞ inner problem** (`css_inner.py`), §7, with its Stokes jump computed as in `struct_css.py`.

Collocation and complex-path shooting share only the ODE (checked in sympy); the inner problem shares nothing with
either but the derivation.

## 3. Regularity at the horizon is analyticity, and why the published D ≥ 5 values disagree

Near $`x_h`$ the general solution is (analytic part) + $`c\,(x_h-x)^\mu(1+\dots)`$. From the f-equation, with
$`\kappa_s = 1-(n-1)u_h`$ (HE95a's "surface gravity" 1 − u(z₂) at n = 2),

```math
\mu = 1 + \frac{n-2+2(n-1)u_h + 2i\omega\,\big(1 + 2\kappa\lvert f_h\rvert^2/\rho_h\big)}{2\kappa_s}, \qquad
\mu_{\rm pert}(\lambda) = \mu - \frac{\lambda}{\kappa_s}\ \ \text{for a perturbation } e^{\lambda\tau} .
```

At n = 2 this is Re μ = 1/(1 − u_h). Checked numerically (`css_mucheck.py`): off the solution the jump scales as
$`\lvert J\rvert \propto R^{{\rm Re}\,\mu}`$, and the local slope d ln|J|/d ln R extrapolated to R → 0 is 1.326, 2.60, 4.99 against
the predicted 1.323, 2.607, 4.993 at D = 4, 6, 10.

Two consequences:

- **Re μ > 1 always**, and on the κ = 1 branch Re μ = 1.32, 1.98, 2.61, 3.82, 4.99, 10.5, 26.0 at D = 4, 5, 6, 8, 10, 20, 50
  (about n/2). So every solution shot from the centre reaches the horizon with f′ finite, and the first-order
  condition at $`x_h`$ (the numerator of f″ vanishing there) is satisfied automatically. What selects ω is
  analyticity: the coefficient c must vanish. For D ≥ 6 even f″ is finite on the irregular branch. A horizon start
  from a Taylor expansion of order k cannot exclude the irregular branch once Re μ ≳ k.
- **The selection weakens with D.** Collocation in a polynomial basis sees $`(x_h-x)^\mu`$ only through an error of order
  $`N^{-2\,{\rm Re}\,\mu}`$, so the Newton system loses the selection. In practice ω from collocation agrees with the complex-path
  value to 5×10⁻¹⁰ at D = 4, 10⁻⁸ at D = 5–6, 10⁻⁵ at D = 7 and 10⁻⁴ at D = 10, and Newton fails at D = 16. The complex-path jump measures c directly. Its
  sensitivity |∂J/∂ω| (at R = 0.85 x_h) decays only like $`e^{-0.2n}`$: 3.6 at D = 4, 1.6 at D = 10, 0.09 at D = 24,
  3×10⁻⁴ at D = 50. This is the CSS counterpart of the README's statement that the period is selected beyond all orders
  in 1/n.

**The published D ≥ 5 values.** Our branch is the D = 4 solution validated in §4, continued in D through non-integer
dimensions (steps of 0.1–0.25), and at D = 5 the two independent methods agree to 1.3×10⁻⁸. It does **not** agree with
the published axion–dilaton values for D ≥ 5:

| D | ω here | ω [AGH12] | $`x_h`$ here | z₊ [AGH12] | \|f(0)\| here | \|f(0)\| [AGH12] | γ here | γ [AGH12] |
|---|---|---|---|---|---|---|---|---|
| 4 | 1.1769527 | 1.176 | 2.6090935 | 2.609 | 0.8925554 | 0.892 | 0.2641066 | 0.261 |
| 5 | 1.3563472 | 1.297 | 1.8812478 | 2.674 | 0.8116648 | 0.903 | 0.3394388 | 0.278 |
| 6 | 1.5125487 | 1.469 | 1.6097130 | 2.771 | 0.7537778 | 0.910 | 0.3758114 | 0.287 |
| 7 | 1.6429318 | 1.610 | 1.4681499 | 2.822 | 0.7106773 | 0.914 | 0.3974655 | 0.311 |
| 8 | 1.7530672 | 1.721 | 1.3812037 | 2.871 | 0.6773497 | 0.918 | 0.4119644 | 0.352 |
| 9 | 1.8474971 | 1.791 | 1.3222907 | 2.929 | 0.6507805 | 0.923 | 0.4224313 | 0.373 |
| 10 | 1.9295309 | 1.852 | 1.2796674 | 2.998 | 0.6290768 | 0.928 | 0.4303925 | 0.347 |

[AGH12] warn that their third digit of γ is unreliable. Antonelli and Hatefi [AH20a] report three different elliptic
solutions at D = 5, (ω, |f(0)|, z₊) = (0.999, 0.673, 1.246), (1.680, 0.644, 1.397), (2.304, 0.700, 1.694), none of them ours
or AGH12's. Tests (`css_lateral.py`, `css_bg.py`):

- Shooting from the centre with any of these published (ω, |f(0)|) reaches b = x at the published z₊ (to the digits
  given). So the equations agree: the D-dimensional equations here and theirs are the same up to the horizon.
- None of them is analytic there. Their jumps are |J| = 0.06 (AGH12) and 0.5, 4.0, 2.0 (AH20a) at R = 0.3, against
  3×10⁻⁴ for our solution with ω moved by 10⁻³.
- Newton on the jump, and collocation Newton at N = 40–80, started from each of them, do not converge to a nearby
  solution. They run to the degenerate limits ω → 0 or |f(0)| → 1/√κ.

Our reading: those solutions satisfy a horizon condition that is weaker than analyticity, which the first point above
makes easy to do once Re μ > 1. The extraction of [AH20a] we read gives their horizon condition as "G = 0 at z₊"; we
could not confirm its order from the text available, so this is a likely explanation, not a verified one. Within the
equations checked in §1 the solution continued from the validated D = 4 solution is the analytic one, and it is the
one used below.

## 4. Validation at D = 4

| Quantity | Collocation | Complex-path shooting | Published |
|---|---|---|---|
| κ = 1: ω | 1.1769527247 ± 3×10⁻¹⁰ | 1.176952724177 | 1.17695272200 ± 2.7×10⁻⁹ [HHS96] |
| κ = 1: $`x_h`$ (z₊) | 2.6090934733 | 2.6090934730 | 2.60909347510 ± 2.2×10⁻⁹ [HHS96] |
| κ = 1: \|f(0)\| | 0.8925554115 | 0.892555411574 | 0.892555411872 [HHS96] |
| κ = 1: \|f(x_h)\| | 0.3642108743 | 0.3642108745 | 0.364210875022 [HHS96] |
| κ = 1: λ | 3.7863499062–3.7863499081 (N = 30–56) | 3.786349906110 | |
| κ = 1: γ = 1/λ | 0.26410660 | 0.264106599970 | 0.2641066 [HHS96]; 0.2641 [HE97]; 0.264 numerical [HHS96]; 0.261 [AGH12] |
| κ = 0: ω | 1.9154446900 | 1.915444689893 | 1.9154446 ± 1×10⁻⁷ [HE95a] |
| κ = 0: $`x_h`$ (z₂) | 5.0035382599 | 5.0035382522 | 5.0035380 ± 2×10⁻⁷ [HE95a] |
| κ = 0: p(0)/√(4π) | 0.6721726675 | 0.672172667 | 0.67217263 ± 4×10⁻⁷ [HE95a] |
| κ = 0: u(x_h) | 0.3970720442 | | 0.39707205 ± 3×10⁻⁸ [HE95a] |
| κ = 0: λ (real) | 2.5832752 | | γ = 0.387106 [HE95b] (λ = 2.58327) |
| κ = 0: oscillatory mode | 0.2586698 ± 3.5022488i | | "unstable via a growing oscillatory mode" [HE95b] |

The symmetry modes come out exact. At D = 4, κ = 1 the collocation gives λ = 0.9999999999 (time translation),
3×10⁻⁸ (phase) and 2×10⁻⁹ ± 1.1769527294i (target isometries), and the complex-path method gives the time-translation
root at 0.999999998755.

## 5. Results: ω(D), λ(D), γ(D)

κ = 1 (axion–dilaton), complex-path shooting for both the background and λ. All points are in
`solver/matter/data/css_bg_k1.jsonl` and `css_gamma_k1.jsonl`; this table is printed by `css_table.py`. Δ = 2π/ω is
the period of the phase rotation in τ, and (D − 3)γ is the mass exponent.

| D | ω | Δ = 2π/ω | \|f(0)\| | $`x_h`$ | \|f(x_h)\| | Re μ | \|∂J/∂ω\| | λ | γ = 1/λ | (D − 3)γ | n(λ − 2) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 1.1769527242 | 5.33851970 | 0.8925554116 | 2.609093473 | 0.36421087 | 1.32 | 3.6e+00 | 3.7863499061 | 0.2641066000 | 0.264107 | 3.5727 |
| 5 | 1.3563472388 | 4.63243123 | 0.8116647742 | 1.881247798 | 0.37936193 | 1.98 | 3.3e+00 | 2.9460390622 | 0.3394388122 | 0.678878 | 2.8381 |
| 6 | 1.5125487346 | 4.15403826 | 0.7537778469 | 1.609712973 | 0.38448366 | 2.61 | 3.0e+00 | 2.6609091321 | 0.3758114052 | 1.127434 | 2.6436 |
| 7 | 1.6429317975 | 3.82437379 | 0.7106773145 | 1.468149921 | 0.38721848 | 3.22 | 2.6e+00 | 2.5159417625 | 0.3974654799 | 1.589862 | 2.5797 |
| 8 | 1.7530672061 | 3.58410977 | 0.6773497030 | 1.381203746 | 0.38903437 | 3.82 | 2.3e+00 | 2.4273942087 | 0.4119644005 | 2.059822 | 2.5644 |
| 10 | 1.9295309155 | 3.25632788 | 0.6290768423 | 1.279667442 | 0.39146806 | 4.99 | 1.6e+00 | 2.3234604011 | 0.4303925298 | 3.012748 | 2.5877 |
| 12 | 2.0654787606 | 3.04199948 | 0.5956846887 | 1.221970646 | 0.39312988 | 6.13 | 1.1e+00 | 2.2634569774 | 0.4418020797 | 3.976219 | 2.6346 |
| 14 | 2.1739640617 | 2.89019741 | 0.5711368341 | 1.184591873 | 0.39438213 | 7.24 | 7.7e-01 | 2.2238340856 | 0.4496738343 | 4.946412 | 2.6860 |
| 16 | 2.2628229844 | 2.77670209 | 0.5522917691 | 1.158313976 | 0.39537531 | 8.34 | 5.1e-01 | 2.1954371040 | 0.4554901610 | 5.921372 | 2.7361 |
| 20 | 2.4001765791 | 2.61780127 | 0.5251959744 | 1.123658982 | 0.39686915 | 10.49 | 2.2e-01 | 2.1570093551 | 0.4636048507 | 7.881282 | 2.8262 |
| 24 | 2.5017512811 | 2.51151478 | 0.5066072699 | 1.101719668 | 0.39794853 | 12.60 | 9.1e-02 | 2.1318984091 | 0.4690655032 | 9.850376 | 2.9018 |
| 30 | 2.6129541759 | 2.40462897 | 0.4875378031 | 1.080539233 | 0.39910921 | 15.73 | 2.5e-02 | 2.1068819362 | 0.4746350438 | 12.815146 | 2.9927 |
| 40 | 2.7356002136 | 2.29682147 | 0.4679224270 | 1.059962435 | 0.40036531 | 20.87 | 2.8e-03 | 2.0816418623 | 0.4803900316 | 17.774431 | 3.1024 |
| 50 | 2.8156301030 | 2.23153791 | 0.4558583544 | 1.047836145 | 0.40117133 | 25.96 | 3.3e-04 | 2.0662314281 | 0.4839728921 | 22.746726 | 3.1791 |
| 60 | 2.8720402109 | 2.18770799 | 0.4476803331 | 1.039818855 | 0.40173280 | 31.03 | 4.0e-05 | 2.0557856880 | 0.4864320273 | 27.726626 | 3.2356 |
| 70 | 2.9139684143 | 2.15622972 | 0.4417681227 | 1.034116468 | 0.40214644 | 36.07 | 5.1e-06 | 2.0482176464 | 0.4882293646 | 32.711367 | 3.2788 |
| 80 | 2.9463680052 | 2.13251885 | 0.4372933729 | 1.029849437 | 0.40246388 | 41.11 | 6.6e-07 | 2.0424735722 | 0.4896024182 | 37.699386 | 3.3129 |

**Accuracy.**

- *Background.* |J| ≤ 9×10⁻¹³ at every point, typically 10⁻¹⁵–10⁻¹⁴. Divided by |∂J/∂ω|, that bounds the error of ω
  (and similarly of |f(0)|) by about 10⁻¹¹ for D ≤ 60, 10⁻¹⁰ at D = 70 and 10⁻⁸ at D = 80. The jump also vanishes, to the
  same level, on a second circle of radius 0.6 x_h.
- *Growth rate.* λ found with the circle radius 0.85 x_h and again with 0.6 x_h agrees to 6×10⁻¹¹ at D = 4–5, to
  7×10⁻¹³ at D = 6–40, and to 1.3×10⁻¹¹ at D = 45–80.
- *Cross-checks.* Against the collocation, where that converges: D = 4–6 to 5×10⁻⁸ (λ) and 3×10⁻⁸ (ω). At D = 7–10
  the differences, 5×10⁻⁶ to 10⁻⁴, are the collocation's error.

## 6. Uniqueness of the unstable mode

γ = 1/λ needs the relevant mode to be the *only* growing mode other than the exact symmetry modes (§1). Checked in
two independent ways:

1. **Collocation spectra, D = 4–8** (`css_scan.py`, N = 40 and 56, `data/css_scan_k1.jsonl`). The full spectrum of the
   discretized operator is computed, and an eigenvalue counts as converged when it agrees between the two resolutions
   (to 10⁻⁴). The converged eigenvalues with Re λ > −0.6 are, at every D = 4, 5, 6, 7, 8: λ_phys, λ = 1, λ = 0 and
   ±iω. At D ≥ 6 the last pair is displaced by collocation error (Re λ = 1×10⁻⁵ at D = 6, 2×10⁻³ at D = 8, with
   Im λ = ±ω), and it is identified by its imaginary part. At D = 4 the next mode is λ = −0.954 ± 1.156i. The same
   code reproduces the κ = 0 spectrum of HE95b, *including* its second (oscillatory) unstable mode (§4). So it does
   find a second unstable mode when one exists.
2. **Winding numbers of det M(λ)** (`css_pert.py count`, `css_count.py`, `css_locate.py`, `data/css_count_k1.jsonl`,
   `data/css_locate_k1.jsonl`). det M is an entire function of λ. Its zeros are the growth rates, which are zeros for
   every circle radius R, plus artefacts of the jump method. The artefacts move or disappear when R is changed, lie
   near the line Im λ = κ_s Im μ on which $`\mu_{\rm pert}(\lambda)`$ is real, and are absent from the collocation
   spectrum. So **the number of growth rates in a box is at most the smallest count over the radii tried**. When one R
   gives exactly 2 (λ = 1 and λ_phys, both known growth rates), the box holds no other growth rate.
   - D = 4, R = 0.85 x_h: 1 zero in 0.2 < Re λ < 2.5, |Im λ| < 1.5 (λ = 1), and 1 in 2.5 < Re λ < 5, |Im λ| < 0.5
     (λ = 3.786).
   - D = 4, R = 0.85 x_h: 1 in each of 2.5 < Re λ < 5, 0.5 < ±Im λ < 1.5. Located at 2.532 ± 1.486i, it moves to 2.521 ± 1.510i
     for R = 0.6 x_h, out of that box (next to κ_s(μ + 2) = 2.511 + 1.537i): an artefact.
   - D = 4, R = 0.85 x_h: 3 in 0.2 < Re λ < 5, 1.5 < Im λ < 2.5, on the resonance line Im λ = 1.537. The collocation finds no
     eigenvalue there.
   - D = 10, R = 0.85 x_h: 4 zeros in 0.2 < Re λ < 4.5, |Im λ| < 3. They are λ = 1, λ_phys and a pair at 3.3226 ± 2.5045i,
     next to κ_s(μ − 1) = 3.20 ± 2.63i. For R = 0.6 x_h the secant iteration started there runs to λ = 1: the pair is an
     artefact.
   - D = 20, R = 0.85 x_h: 4 zeros in 0.2 < Re λ < 6, |Im λ| < 3. The upper half-box (Im λ > 0.05) holds exactly one, at
     4.8288 + 2.9969i, at its edge and next to the resonance line Im λ = 3.30. For R = 0.6 x_h it is gone (the secant
     iteration runs to λ = 1): an artefact.
   - D = 40, R = 0.85 x_h: exactly 2 zeros in 0.2 < Re λ < 5, |Im λ| < 2.5 (the resonance line is at Im λ = 3.78).
   - D = 10, R = 0.85 x_h, direct check: exactly 2 zeros in 0.2 < Re λ < 4.5, |Im λ| < 2.
   - D = 10, R = 0.6 x_h: again 4 zeros in the D = 10 box above. The artefact pair has moved, not vanished, so this count
     alone does not settle the question; the location test does.

   Putting the located zeros together, the boxes that hold **exactly two zeros, λ = 1 and λ_phys**, are:
   0.2 < Re λ < 5, |Im λ| < 1.2 at D = 4; 0.2 < Re λ < 4.5, |Im λ| < 2.4 at D = 10; 0.2 < Re λ < 6, |Im λ| < 2.9 at D = 20;
   0.2 < Re λ < 5, |Im λ| < 2.5 at D = 40. In each larger box the extra zeros form one conjugate pair (D = 10, 20) or a
   few pairs (D = 4) on the resonance line, and each one located moves with R.

So in every box examined, the only zeros of det M with Re λ > 0.2 that do not move with R are λ = 1 and λ_phys; the
collocation agrees at D ≤ 8. **Within these boxes the relevant mode is unique at every D checked**, and λ_phys(D) is
continuous along the whole branch. Not covered: growth rates above the boxes (|Im λ| beyond 1.2–2.9, or Re λ above
4.5–6), and dimensions other than D = 4–8, 10, 20, 40. The collocation finds none at D ≤ 8. The D = ∞ question
(is λ = 2 the only unstable mode of the inner problem?) is open here, as it is for the real scalar (GAMMA.md §6).

## 7. Extrapolation to D = ∞

**γ.** Fits of λ in powers of 1/n (`css_fit.py`, points D = 4–80). Free fits of λ∞, D ≥ Dmin, K powers of 1/n:

| D ≥ | K = 3 | K = 4 | K = 5 | K = 6 | K = 7 |
|---|---|---|---|---|---|
| 12 | 2.72×10⁻³ | 1.05×10⁻³ | 3.98×10⁻⁴ | 1.51×10⁻⁴ | 5.78×10⁻⁵ |
| 16 | 1.62×10⁻³ | 5.50×10⁻⁴ | 1.86×10⁻⁴ | 6.26×10⁻⁵ | 2.16×10⁻⁵ |
| 20 | 1.08×10⁻³ | 3.27×10⁻⁴ | 9.93×10⁻⁵ | 3.02×10⁻⁵ | 8.64×10⁻⁶ |
| 24 | 7.76×10⁻⁴ | 2.17×10⁻⁴ | 6.21×10⁻⁵ | 1.68×10⁻⁵ | (1.70×10⁻⁵) |
| 30 | 5.08×10⁻⁴ | 1.32×10⁻⁴ | 3.33×10⁻⁵ | 8.18×10⁻⁶ | (8.27×10⁻⁶) |

(The entries in parentheses use 8 parameters on 11–13 points.)

Each added power divides λ∞ − 2 by about 3, with no sign of levelling off until the number of points runs out
(K = 7 with D ≥ 24 or 30). At every order, the fit with λ∞ = 2 fixed has an rms residual about half that of the free
fit with the same number of parameters. This is the same pattern as the real scalar's (GAMMA.md §1). With λ∞ = 2 fixed,

```math
\lambda = 2 + \frac{3.601}{n} - \frac{27.0}{n^2} + \frac{4\times10^2}{n^3} + \dots, \qquad
\gamma = \frac12 - \frac{0.900}{n} + \frac{8.4}{n^2} + \dots
```

The 1/n coefficient is 3.598–3.601 over the fits with 5–6 powers and D ≥ 20–24, rising slowly with the order, so
c₁ = 3.60 ± 0.01. For the real scalar's DSS solution GAMMA.md has λ = 2 + 3.0324/n − 34.3/n², a similar series.
**Conclusion: λ∞ = 2 to about 10⁻⁵, γ∞ = 1/2 to about 3×10⁻⁶** (an extrapolation from D ≤ 80; at D = 80,
γ = 0.4896024).

**ω.** The phase-rotation frequency extrapolates to

| | κ = 1 (axion–dilaton) | κ = 0 (free complex scalar) |
|---|---|---|
| ω∞ extrapolated from finite D (D ≤ 80) | 3.1994 (fits: 3.1985–3.1999) | 3.9784 (3.9776–3.9787) |
| ω∞ from the D = ∞ inner Stokes zero (§8) | **3.1998939020** | **3.9789695074** (STRUCTURE.md: 3.9789695074) |
| \|f(0)\|∞ extrapolated | 0.404889 (0.404878–0.404914) | 0.501934 (0.501924–0.501972) |
| \|f(0)\|∞ inner | 0.4048768438 | 0.5019081944 |
| A∞² = ω²\|f(0)\|²h₀, extrapolated | 2.40123 (2.40120–2.40131) | 3.98842 (3.98838–3.98856) |
| A∞², inner | 2.40119167 | 3.98831815 |
| Δ∞ = 2π/ω∞ | 1.9635605 | 1.5790986 |
| $`x_h`$ extrapolated | 1.00002 | 1.00002 |

The extrapolations and the D = ∞ selection rule agree to 1.6×10⁻⁴ (κ = 1) and 1.3×10⁻⁴ (κ = 0) in ω, and to
2–5×10⁻⁵ in |f(0)| and A². The inner values lie just outside the narrow fit ranges, so those ranges understate the
extrapolation error by a factor of about 2–3, as expected from 20–35 points. This is the test proposed in
LITERATURE.md §10 (1a) and STRUCTURE.md §7: the selection of the frequency by the vanishing of a Stokes jump of the
D = ∞ inner problem, with no harmonic truncation. It passes for both κ.

## 8. How the solution scales with D

The structure is the real scalar's ([README, "How the solution scales with D"](../README.md#how-the-solution-scales-with-d)):

- **Weak gravity inside the light cone.** The horizon tends to x = 1 like $`x_h = 1 + A^2/n + \dots`$: n(x_h − 1) = 2.24,
  2.23, 2.28, 2.33 at D = 10, 20, 40, 80, against A∞² = ω∞²|f(0)|∞²h₀ = 2.401 (§7). Inside the light cone b − 1 = O(1/n),
  and the field is f = f(0) + O(x²/n). The central value |f(0)| tends to a finite limit 0.4049, so F = O(1) in the HE
  normalization, and κ|F|² stays O(1) (κ|f(0)|² → 0.164).
- **An inner region at x ~ √n, beyond the horizon.** Continued analytically past $`x_h`$, the finite-D solutions at fixed
  ξ = x/√n converge to the D = ∞ inner solution, with O(1/n) corrections (`css_scaling.py`, `data/css_scaling_k*.json`;
  table below). There gravity is O(1): b grows from 1 to about 1.2 (κ = 1) or 1.3 (κ = 0) by ξ ≈ 0.8.
- **The inner problem.** With $`s = r^2/(2n)`$ and t, at leading order in 1/n (derived from the equations of §1;
  `css_inner_pde.py` checks in sympy that its CSS reduction is exactly the inner ODE used below, and `css_inner.py check`
  that this ODE is the n → ∞ limit of the finite-n ODE, with residual ∝ 1/n):

```math
F_s = \frac{1}{c}\,\partial_t\Big(\frac{F_t}{c}\Big) + \frac{2\kappa\bar F}{1-\kappa\lvert F\rvert^2}\,\frac{F_t^2}{c^2},
\qquad
c_s = \frac{2\,\lvert F_t\rvert^2}{(1-\kappa\lvert F\rvert^2)^2\,c} .
```

This is Rozali and Way's heat-type system (GAMMA.md §3) for a complex field, with the target curvature as one extra
term. It contains no explicit s or t, for every κ: s-translation (λ = 2), t-translation (λ = 1), the scaling
(t, s) → (at, a²s), time reparametrizations (t → T(t), c → T′c), the U(1) phase and, for κ > 0, the SL(2,R) target
isometries are all symmetries. The CSS ansatz $`F = (-t)^{i\omega}f(\xi)`$, c = b(ξ), ξ = √(2s)/(−t) reduces it to one
ODE in ξ, exactly with one harmonic (STRUCTURE.md §7).

- **Selection beyond all orders.** The horizon exponent grows like Re μ ≈ n/2 (Re μ/n = 0.62, 0.58, 0.55, 0.53 at
  D = 10, 20, 40, 80), and the sensitivity of the analyticity condition to ω decays like $`e^{-0.2n}`$ (§3). This is the
  CSS form of the README's Proposition 1: the 1/n expansion inside the light cone cannot fix ω. ω is fixed at x ~ √n,
  by the Stokes jump of the inner problem.

**The finite-D solutions against the inner solution** (κ = 1; b and |f| at x = √n ξ):

| ξ | b: D = 20 | 30 | 40 | ∞ | \|f\|: D = 20 | 30 | 40 | ∞ |
|---|---|---|---|---|---|---|---|---|
| 0.4 | 1.19377 | 1.17540 | 1.16591 | 1.13531 | 0.31195 | 0.27280 | 0.25253 | 0.18862 |
| 0.5 | 1.23064 | 1.20566 | 1.19286 | 1.15245 | 0.26382 | 0.22679 | 0.20785 | 0.14943 |
| 0.6 | 1.25765 | 1.22700 | 1.21145 | 1.16333 | 0.22725 | 0.19263 | 0.17514 | 0.12217 |
| 0.8 | 1.29300 | 1.25378 | 1.23423 | 1.17552 | 0.17822 | 0.14807 | 0.13310 | 0.08904 |

At ξ = 0.8, n(|f_D| − |f_∞|) = 1.61, 1.65, 1.67 and n(b_D − b_∞) = 2.11, 2.19, 2.23 at D = 20, 30, 40, i.e. O(1/n)
convergence. For κ = 0 the same quantities are 1.98, 1.99, 1.99 and 3.8, 4.0, 4.1 at D = 16, 24, 40. At D = 80 the
continuation to x ≈ 5 is spoiled by the irregular mode, of order $`(x-x_h)^{41}`$, seeded by the 10⁻⁸ error of the
background: the same beyond-all-orders sensitivity, seen from the other side.

**The D = ∞ selection, without truncation.** The inner ODE has an irregular singular point at ξ = 0, with fast mode
$`e^{-b^2/(2\xi^2)}`$ (the heat kernel). The selection rule of the README (Conjecture 1) asks that the Stokes jump of the
regular formal solution vanish. That is one complex condition on (|f(0)|, ω), computed by lateral solutions as in
`struct_css.py` (`css_inner.py`). For κ = 0 this reproduces STRUCTURE.md's zero to all ten digits, in different
variables and code: |f(0)| = 0.5019081944, ω = 3.9789695074, A = ω|f(0)| = 1.9970774009. Continued in κ (at two matching
radii, which agree to ≤ 7×10⁻¹²):

| κ | \|f(0)\|∞ | ω∞ | Δ∞ = 2π/ω∞ | A∞² | κ\|f(0)\|∞² |
|---|---|---|---|---|---|
| 0 | 0.5019081944 | 3.9789695074 | 1.5790986323 | 3.98831815 | 0 |
| 0.1 | 0.4880267141 | 3.8739534108 | 1.6219052324 | 3.75088288 | 0.0238 |
| 0.2 | 0.4756065499 | 3.7755638626 | 1.6641713757 | 3.53729396 | 0.0452 |
| 0.4 | 0.4539973262 | 3.5985546092 | 1.7460302787 | 3.17028991 | 0.0824 |
| 0.6 | 0.4355202632 | 3.4457917351 | 1.8234373375 | 2.86772760 | 0.1138 |
| 0.8 | 0.4193198399 | 3.3139830103 | 1.8959618343 | 2.61495766 | 0.1407 |
| 1 | 0.4048768438 | 3.1998939020 | 1.9635605116 | 2.40119167 | 0.1639 |

(`data/css_inner_kpath.jsonl`; at each κ the zero is found with matching radii r₁ = 0.3 and 0.25, which agree to
≤ 7×10⁻¹² in ω.) The target curvature lowers ω∞ by 20% between κ = 0 and κ = 1. Its effect is O(1), as the scaling
F = O(1) implies.

## 9. Clark–Pimentel and the scope of the rule

Clark and Pimentel [CP25] (read via the arXiv abstract page and the HTML full text, October 2026) study "spherical
gravitational collapse of a massless, scale-invariant scalar field with continuous self-similarity". What the text
establishes:

- The field is a single real scalar (their action (2.2) and constraint (2.24b) are written in $`(\phi')^2`$).
- The general CSS form is $`\phi = \bar\phi(x) - \bar k\log(-\bar u)`$ (2.17), and they impose the scale-invariant profile
  $`\bar k = 0`$ (2.19). There is no phase rotation.
- Their D = 4 critical solution is the Roberts/BONT solution. The critical member of the family ends in a null
  singularity at u = 0 (2.54–2.56), and the continuation beyond the self-similarity horizons has a timelike
  singularity at the origin, which is why they restrict the domain.
- In their self-similar time x = log(−v̄/ū) the matched regions are A (x = O(1), weak gravity), B (x = O(√ε), ε = 1/(D − 2),
  a saddle with growth $`e^{\pm x/\sqrt{2\varepsilon}}`$) and C (strong gravity).
- Their exponents are mass exponents: (D − 3)/√(2(D − 2)) for CSS-preserving perturbations and 2(D − 3)/(D − 2) for
  CSS-breaking ones (their (1.4)–(1.5)). In the radius convention these are γ = 1/√(2n) and 2/n, both → 0, i.e. λ = √(2n)
  and n/2, both → ∞.

So their data point concerns a different solution: a real scalar with no phase rotation, no regular centre and no
analytic past light cone, whose growth rates come from a saddle at the scale x ~ √ε of their logarithmic time. It is
not the continuation of a regular-centre, analytic-horizon critical solution whose selection happens at x ~ √n in the
autonomous inner heat system. Proposition 5 needs exactly that structure, so CP25 does not contradict it. It does show
that the rule "no 1/r² term ⇒ γ∞ = 1/2" needs its hypotheses stated: (i) the critical solution is regular at the centre
and analytic at the self-similarity horizon; (ii) at large D it has the inner scaling (fields O(1) at x ~ √n, weak
gravity inside the light cone), so its D = ∞ limit is an inner solution selected by the vanishing Stokes jump; (iii) the
λ = 2 mode is its only physical unstable mode there; (iv) λ(D) is continuous at 1/n = 0. §§6–8 check (i), (ii) and (iii)
numerically (to D = 40–80) for the axion–dilaton solution. Whether CP25's solutions are related to the static (ω = 0)
zeros of the inner jump found in STRUCTURE.md §7 was not checked.

## 10. Verdict

**γ → 1/2 for the axion–dilaton CSS critical solution, and the r²-shift mechanism applies.**

- *Numbers.* γ(D) rises monotonically from 0.2641066 at D = 4 to 0.4896024 at D = 80. The free fits give
  λ∞ − 2 = 8.6×10⁻⁶ at the highest order, still falling about 3× per added power. With λ∞ = 2 fixed,
  λ = 2 + 3.60/n − 27/n² + … and γ = 1/2 − 0.90/n + …. So γ∞ = 1/2, to about 3×10⁻⁶ (an extrapolation: the computed
  points end at D = 80, where γ = 0.4896024). The mass exponent (D − 3)γ grows like (D − 3)/2.
- *Why, in terms of the scaling.* The CSS solution has the real scalar's large-D structure. Gravity is weak inside the
  light cone ($`x_h \to 1`$, b − 1 = O(1/n)). The field is O(1) and varies on x ~ √n, beyond the light cone, where
  gravity is O(1) and the equations become the autonomous heat-type system of §8. Its ω is selected there, beyond all
  orders in 1/n (horizon exponent Re μ ≈ n/2). The D = ∞ ω from the inner Stokes condition agrees with the
  extrapolated finite-D ω: 3.1998939 against 3.1994 for κ = 1, and 3.9789695 against 3.9784 for κ = 0 (§7). Because
  the inner system has no explicit s for any κ, the s-translation, i.e. the shift of r², is an exact symmetry, and it
  gives the λ = 2 mode as in Proposition 5. The target curvature
  does not spoil this: it enters the inner system at O(1) (κ|F|² → 0.16) but without explicit s. The finite-D relevant
  mode is unique (§6) and continuous in D, and its λ(D) tends to 2 at the rate of an ordinary 1/n series. That is
  what the corollary to Proposition 5 predicts.
- *The rule's scope.* Clark–Pimentel's γ_r → 0 (§9) is for a real-scalar CSS solution with no regular centre and no
  analytic horizon, whose growth rates come from a saddle at a different scale. It does not satisfy the hypotheses of
  the mechanism, so it is not a counterexample. The rule should be stated with them: no O(n)/r² term, *and* a critical
  solution with a regular centre and an analytic self-similarity horizon whose large-D limit is the inner solution at
  x ~ √n, *and* uniqueness of the relevant mode.
- *The published γ(d).* Álvarez-Gaumé and Hatefi's γ(d) (0.278–0.373 for d = 5–10, non-monotonic) is not reproduced.
  Their solutions for d ≥ 5 are not analytic at the horizon in the equations checked here (§3).

## 11. Open issues

- **Uniqueness at D = ∞ is not shown**, here or for the real scalar (GAMMA.md §6). At finite D it is checked by the
  full collocation spectrum for D ≤ 8, and by winding numbers in the boxes of §6 (Re λ ≤ 4.5–6, |Im λ| ≤ 1.2–2.9 at
  D = 4, 10, 20, 40). The jump method has artefacts near the resonance line Im λ = κ_s Im μ; they were identified by
  their R-dependence one by one, not removed systematically.
- **The λ = 2 mode of the CSS inner solution was not constructed explicitly.** The analogue of Proposition 5(ii) and of
  `pert_check3.py` would be the s-translate of the inner solution plus the gauge term; for CSS the gauge function P is
  a constant. That λ = 2 is a growth rate at D = ∞ follows from the autonomy of the inner system (§8, checked) and the
  argument of Proposition 5(iii). It inherits the status of the selection rule (Conjecture 1), here tested by the
  agreement of ω∞ in §7.
- **The coefficient c₁ = 3.60** of λ = 2 + c₁/n is not derived. As for the real scalar's 3.0324, it would need the
  O(1/n) inner equations.
- **The published D ≥ 5 solutions** [AGH12, AH20a]: the explanation offered in §3 (a horizon condition weaker than
  analyticity) is a likely reading, not verified against their codes.
- **κ = 0 (free complex scalar)**: ω(D) and |f(0)|(D) were computed to D = 80 and extrapolated (§7). λ(D) and its
  second, oscillatory unstable mode (λ = 0.2587 ± 3.5022i at D = 4) were not followed in D. So whether the κ = 0
  solution becomes critical (one unstable mode) at large D is open.
- **Other κ.** The inner zero was followed for κ ∈ [0, 1]; finite-D runs were made only for κ = 1 and κ = 0. In
  solver units (G = 2nT) the HE field is F = √n φ, so a fixed HE κ means a curvature κn growing with D. At D = ∞
  the HE normalization is the natural one: F = O(1) there (§8).
- **Clark–Pimentel and the static (ω = 0) inner zeros** of STRUCTURE.md §7: not compared.
- **Precision.** Double precision throughout. At D = 80, |∂J/∂ω| = 7×10⁻⁷ limits ω and |f(0)| to about 10⁻⁸; λ
  agrees between two circle radii to ≤ 6×10⁻¹¹ at all D. |∂J/∂ω| falls like $`e^{-0.2n}`$, so going much beyond D ≈ 100
  would need extended precision.

## 12. Scripts and data

All in `solver/matter/`; run single-process under `nice -n 10`. Times are for one loaded cloud core.

| Script | What it does | Time |
|---|---|---|
| `css_equations.py` | sympy: the D-dimensional equations from the metric (D = 4, 5, 6 explicit), the model's PDE terms, and the D = 4 published forms (§1) | 30 s |
| `css_model.py` | the equations in (τ, w) as plain arithmetic, Chebyshev collocation, background Newton, perturbation matrices | library |
| `css_bg.py [kappa] [N]` | centre shooting and the collocation solve; D = 4 validation against HHS96 / HE95a | seconds |
| `css_scan.py KAPPA DLIST --N 40,56` | collocation continuation in D with the full spectrum at two resolutions (§6) → `data/css_scan_k1.jsonl` | seconds per D |
| `css_lateral.py cont KAPPA D0 OMEGA F0 DLIST [RFRAC] [OUT] [STEP]` | the background by complex-path shooting, continued in D; horizon data from Cauchy integrals → `data/css_bg_k*.jsonl` | 5–20 s per D |
| `css_pert.py D KAPPA OMEGA F0 LAM` / `count …` | the growth rate as a zero of det M(λ); winding number of det M in a box | 10 s per root; minutes per box |
| `css_gamma.py BG OUT` | λ(D) and γ(D) for every background point → `data/css_gamma_k1.jsonl` | 1 min per D |
| `css_count.py BG OUT D:RE0:RE1:IM[:RFRAC] …` | winding number of det M in a box, for a given circle radius (§6) → `data/css_count_k1.jsonl` | 20–60 min per box |
| `css_locate.py BG OUT D RE0 RE1 IM0 IM1` | the zeros of det M in a box (winding number and their sum), refined, and recomputed with a second radius: growth rate or artefact (§6) → `data/css_locate_k1.jsonl` | 20–40 min |
| `css_inner_pde.py` | sympy: the autonomous D = ∞ inner PDE reduces under the CSS ansatz to the inner ODE (§8) | 1 min |
| `css_momcheck.py [KAPPA] [DLIST]` | the linearized momentum constraint on the collocation eigenvectors (§1) | 1 min |
| `css_mucheck.py BG DLIST` | the horizon exponent μ against the scaling of the jump with R (§3) | 1 min |
| `css_fit.py [BG] [GAMMA]` | fits in powers of 1/n; λ∞, γ∞, ω∞ (§7) → `data/css_fit_k1.json` | seconds |
| `css_inner.py check` / `solve` / `kpath` | the D = ∞ inner problem for any κ: n → ∞ limit check, κ = 0 against STRUCTURE.md, the Stokes zero continued in κ (§7) → `data/css_inner_kpath.jsonl` | minutes per zero |
| `css_scaling.py BG KAPPA F0 OMEGA [DLIST]` | finite-D solutions continued past the horizon to x = √n ξ, against the inner solution (§8) → `data/css_scaling_k*.json` | 1–5 min |
| `css_table.py` | the table of §5 | seconds |

## 13. References

Journal references and arXiv numbers as in [LITERATURE.md](LITERATURE.md) §11, where they were checked; the content
used here was read from the arXiv pages in October 2026 (extractions, not byte-compared with the PDFs).

- [HE95a] E. W. Hirschmann, D. M. Eardley, PRD 51, 4198 (1995), gr-qc/9412066: the complex-scalar CSS solution, eqs.
  (13)–(17), (24), ω = 1.9154446, z₂ = 5.0035380, P = 0.67217263, u(z₂) = 0.39707205.
- [HE95b] E. W. Hirschmann, D. M. Eardley, PRD 52, 5850 (1995), gr-qc/9506078: γ ≈ 0.387106 and a growing
  oscillatory mode (abstract).
- [HE97] E. W. Hirschmann, D. M. Eardley, PRD 56, 4696 (1997), gr-qc/9511052: the κ-model, eqs. (4), (8), (21)–(25);
  γ(κ) Table 1 (0.2641 at κ = 1); the gauge modes; the stability change at κ ≈ 0.0754.
- [EHH95] D. M. Eardley, E. W. Hirschmann, J. H. Horne, PRD 52, R5397 (1995), gr-qc/9505041.
- [HHS96] R. S. Hamadé, J. H. Horne, J. M. Stewart, CQG 13, 2241 (1996), gr-qc/9511024: eqs. (2.3), (4.3), (4.5), (4.6a–d),
  (6.3): ω = 1.17695272200(270), z₊ = 2.60909347510(216), |f(0)| = 0.892555411872, |f(z₊)| = 0.364210875022,
  γ = 0.2641066.
- [AGH12] L. Álvarez-Gaumé, E. Hatefi, CQG 29, 025006 (2012), arXiv:1108.0078: d = D, γ = 1/Re κ (radius convention);
  Tables 1–2.
- [AH20a] R. Antonelli, E. Hatefi, JHEP 03 (2020) 074, arXiv:1912.00078: D = 5 elliptic solutions.
- [CP25] C. R. Clark, G. L. Pimentel, JHEP 11 (2025) 017, arXiv:2504.10590.
- STRUCTURE.md §7 and `struct_css.py` (on main): the κ = 0 inner problem and its Stokes zero.

# The self-gravitating σ-model (hedgehog wave map) in D dimensions

For the massless scalar the radius exponent tends to γ∞ = 1/2, because the D = ∞ inner equations in s = r²/(2n) are
invariant under s-translation, which gives an exact λ = 2 mode ([GAMMA.md](../GAMMA.md), README Proposition 5).
[LITERATURE.md §1](LITERATURE.md) predicts that a centrifugal or potential term of order n/r², which is ∝ 1/s in the
inner variables, breaks this. This page tests the prediction on the self-gravitating hedgehog σ-model (wave map into
$`S^{D-1}`$, coupling η = 4πGf²). It covers: the D-dimensional equations (checked against Bizoń–Wasserman in D = 4); the
continuously self-similar (CSS) critical solution and its growing mode, validated against every published D = 4 and
flat-space number found; its continuation in D; and the D = ∞ inner problem. The DSS branch was not computed (§7).

Notation as in the README: D = n + 2, d = n + 1 space dimensions, x = r/(−t), τ = −ln(−t), s = r²/(2n), and γ = 1/λ is
the radius exponent ($`r_{\rm BH} \propto (p-p_*)^\gamma`$; the mass exponent is (D − 3)γ). Statements are marked, as in
[STRUCTURE.md](STRUCTURE.md): **derived** (sympy script, residuals 0), **computed** (numerical, with its checks),
**argued** (a derivation that is not complete, or an argument from scalings), **conjecture**.

Contents

- [1. Summary](#1-summary)
- [2. Equations](#2-equations)
- [3. Method](#3-method)
- [4. Validation in D = 4 and in flat space](#4-validation-in-d--4-and-in-flat-space)
- [5. The CSS branch in D: γ(D)](#5-the-css-branch-in-d-γd)
- [6. Large D](#6-large-d)
- [7. The DSS branch](#7-the-dss-branch)
- [8. Verdict](#8-verdict)
- [9. Open issues](#9-open-issues)
- [10. Scripts and data](#10-scripts-and-data)
- [11. References](#11-references)

## 1. Summary

- **Equations (derived).** Hedgehog target $`S^{D-1}`$; η = 4πGf² is dimensionless in every D. The polar-areal and
  self-similar equations in dss.py's conventions (§2) pass 48 sympy checks, including Bizoń & Wasserman's D = 4
  equations (8)–(17) term by term. In D = 4 the past light cone forces U(1) = π/2; in D > 4 it does not.
- **Validation in D = 4 (computed).** Flat space: f₁ and λ₁ = 6.3336253 against Bizoń–Chmaj–Tabor's 6.333625, and
  D = 5, 6, 7 against Biernat–Bizoń–Maliborski to all their digits. With gravity: **λ = 5.1428 at η = 0.17** against
  Lechner et al.'s "λ ≈ 5.14" at η_c ≈ 0.17; **γ_CSS = 0.179–0.188 for η = 0.10–0.14** against their ≈ 0.18; one unstable
  mode; **α₀ = 0.0686–0.0687 and α₁ = 0.1518–0.1519** against Bizoń–Wasserman's 0.0688 and 0.1518; their apparent
  horizon ρ_B ≈ 3.3683 for f₀ at η = 0.1, reproduced as 3.36861.
- **γ(D) on the CSS branch (computed).** At fixed η = 0.14, γ = 1/λ rises monotonically: 0.188, 0.261, 0.333, 0.401,
  0.477 at D = 4, 5, 6, 7, 8.125. There is exactly one growing mode all along, and f₀ keeps its black-hole-forming
  horizon (checked to D = 7.5). The branch **ends at a fold at D_f = 8.5444**, where γ = 0.5135. In flat space it ends instead by collapsing onto the
  singular equator cone as D → 8 (fit: 7.998), with γ → 1/2 (fit: λ* = 2.001). That value is forced by the cone, whose
  exponents at n = 6 are −2 and −3. **γ does not tend to 1/2 as D → ∞; there is no D → ∞.** The γ ≈ 1/2 at the ends has a different
  origin (§5, §8).
- **Large D (derived, argued, computed).** At fixed η the D = ∞ inner problem is n-free:
  $`\partial_t(u_t/c) = c(u_s - \sin 2u/(4s))`$, $`c_s = 2\eta u_t^2/c`$ (as in STRUCTURE.md). It has no s-translation, so
  λ∞ = 2 is not protected for any critical solution that survives to D = ∞. Its CSS solutions are zeros of a Stokes
  multiplier. The ground state's zero A₀ = 1.9225686 (η = 0.14) agrees with the finite-D ground state extrapolated from
  D ≤ 7.5 (1.9226 ± 0.0003), a check of the D = ∞ selection rule. The excited family needs the singular equator cone to
  be a spiral point, $`n^2 - (6+2\eta)n + 1 + 2\eta \lt 0`$, which fails at large n. f₁ outlives that edge only by a little and
  then ends (a fold at η = 0.14; a collapse onto the cone at D = 8 in flat space).
- **Verdict.** The CSS branch cannot test γ∞, because it has no large-D limit: computed at η = 0 and 0.14, and argued
  for every fixed η and for fixed η̂ < 1/4 (the β₊ = −2 dimension of the cone is finite there; §8). Fixed η̂ ≥ 1/4 is
  not covered. The structural part of the prediction (the 1/s term kills the symmetry behind γ∞ = 1/2) is confirmed. A finite γ∞ ≠ 1/2 would have to come
  from the DSS branch, which was not computed here (§7).

## 2. Equations

**Model.** In D = n + 2 dimensions,

```math
S = \int d^Dx\,\sqrt{-g}\,\Big[\frac{R}{16\pi G} - \frac{f^2}{2}\,g^{ab}\,G_{AB}(X)\,\partial_aX^A\partial_bX^B\Big],
```

with target the round unit sphere $`S^{n+1} = S^{D-1}`$ and the hedgehog (corotational) ansatz
$`X = (\cos u,\ \sin u\,\hat n)`$, where $`\hat n`$ is the unit normal of the spatial $`S^n`$. $`S^{D-1}`$ is the target the
ansatz requires: the identity map on the orbit spheres plus one polar angle u(t, r). The pull-back of the target metric
is $`h_{ab} = u_au_b + \sin^2u\,\gamma_{ab}`$ (γ the unit $`S^n`$ metric), so the matter Lagrangian is
$`-\tfrac{f^2}{2}\big[(\nabla u)^2 + n\sin^2u/r^2\big]`$ and $`T_{ab} = f^2\big(h_{ab} - \tfrac12 g_{ab}\,g^{cd}h_{cd}\big)`$.

**The coupling.** G has dimension length^(D−2) and f² that of 1/G (both multiply two derivatives in the action), so

```math
\eta = 4\pi G f^2
```

is dimensionless in every D, and the model is scale invariant in every D. In D = 4 it is the α of Bizoń–Wasserman
[BW00] and the η of Husa et al. [HLPTA00]. Einstein's equations are $`G_{ab} = 2\eta\,(h_{ab} - \tfrac12g_{ab}h)`$.

**Polar-areal gauge.** With $`ds^2 = -\alpha^2dt^2 + a^2dr^2 + r^2d\Omega_n^2`$, c = α/a, $`\Phi = u_r`$, $`\Pi = u_t/c`$:

```math
\frac{a_r}{a} = \frac{\eta r}{n}(\Phi^2+\Pi^2) + \frac{\eta a^2\sin^2u}{r} + \frac{(n-1)(1-a^2)}{2r}, \qquad
\frac{c_r}{c} = \frac{(n-1)(a^2-1)}{r} - \frac{2\eta a^2\sin^2u}{r}, \qquad
\frac{a_t}{a} = \frac{2\eta r}{n}\,u_tu_r,
```
```math
\Phi_t = (c\Pi)_r, \qquad \Pi_t = r^{-n}\big(r^nc\,\Phi\big)_r - \frac{n\,c\,a^2\sin 2u}{2r^2} .
```

The flat limit (η = 0, a = c = 1) is $`u_{tt} = u_{rr} + (d-1)u_r/r - (d-1)\sin(2u)/(2r^2)`$ with d = n + 1, the equation
of the task. The coefficients agree with the coordinator's [`struct_eqs.py`](../solver/matter/struct_eqs.py) (wave map:
K = f², V = n f² sin²u/(2r²)), which checked them against the covariant harmonic-map equation at n = 2, 3.

**Self-similar variables**, as in [`solver/dss.py`](../solver/dss.py): τ = −ln(−t), x = r/(−t), the past light cone
of the singularity (SSH) at x = 1, i.e. c(τ, 1) = 1. With $`Q = u_\tau + xu_x`$ (so $`r\Phi = xu_x`$, $`r\Pi = xQ/c`$):

```math
\frac{Q}{c} + (\partial_\tau + x\partial_x)\frac{Q}{c} = x^{-n}\partial_x\big(x^nc\,u_x\big) - \frac{n\,c\,a^2\sin 2u}{2x^2},
```
```math
\frac{x a_x}{a} = \frac{\eta}{n}\Big(x^2u_x^2 + \frac{x^2Q^2}{c^2}\Big) + \eta a^2\sin^2u + \frac{(n-1)(1-a^2)}{2}, \qquad
\frac{xc_x}{c} = (n-1)(a^2-1) - 2\eta a^2\sin^2u, \qquad
(\partial_\tau + x\partial_x)\ln a = \frac{2\eta}{n}\,x u_x Q .
```

In dss.py's first-order form, with the field normalised like dss.py's scalar ($`\psi = \sqrt{\eta/n}\,u`$, so that the
kinetic part of $`G_{ab}`$ is $`2n(\partial\psi\partial\psi - \dots)`$), $`p, m = \sqrt{\eta/n}\;r(\Phi\pm\Pi)`$:

```math
x a_x = \tfrac a2\big[p^2+m^2 + (n-1)(1-a^2) + 2\eta a^2\sin^2u\big], \qquad x c_x = c\big[(n-1)(a^2-1) - 2\eta a^2\sin^2u\big],
```
```math
p_\tau + (x-c)p_x = c_xp - \frac{cp}{x} + \frac{nc(p+m)}{2x} - \frac{\sqrt{\eta n}\,c\,a^2\sin 2u}{2x}, \qquad
m_\tau + (x+c)m_x = -c_xm + \frac{cm}{x} - \frac{nc(p+m)}{2x} + \frac{\sqrt{\eta n}\,c\,a^2\sin 2u}{2x},
```

with $`xu_x = \sqrt{n/\eta}\,(p+m)/2`$. At η → 0 with ψ fixed these are dss.py's equations plus the potential terms.

**CSS reduction.** For u = U(x), a(x), c(x), the Hamiltonian and momentum constraints together are algebraic in a:

```math
a^2 = \frac{(n-1)/2 - (\eta/n)\,x^2U'^2\,(1 - x^2/c^2)}{(n-1)/2 - \eta\sin^2U},
```
```math
(c^2-x^2)\,U'' + U'\Big[cc' + \frac{nc^2}{x} - 2x + \frac{x^2c'}{c}\Big] = \frac{n\,c^2a^2\sin 2U}{2x^2}, \qquad
\frac{xc'}{c} = (n-1)(a^2-1) - 2\eta a^2\sin^2U .
```

At the SSH (x = c = 1): $`a^2 = (n-1)/(n-1-2\eta\sin^2U)`$, c′(1) = 0 for every n and η, and analyticity requires

```math
(n-2)\,U'(1) = \tfrac n2\,a(1)^2\sin 2U(1) .
```

So in D = 4 the past light cone forces U(1) = π/2 with U′(1) free (Bizoń–Wasserman's expansion), while for D > 4
U(1) is free and U′(1) follows. The non-analytic solution at the SSH behaves as $`(1-x)^{n/2}`$ (independent of η);
for even n ≥ 4 it resonates with the Taylor series (D = 6: the two branches U(1) = π/2 and U(1) = π/3 of Biernat,
Bizoń and Maliborski [BBM17] at η = 0). The regularity requirement $`a^2 > 0`$ at the light cone needs
$`2\eta\sin^2U(1) \lt n - 1`$: the hedgehog's solid-angle deficit, $`2\eta/(n-1)`$ for U = π/2, must stay below one.
For D = 4 this is Bizoń–Wasserman's bound η < 1/2; in D dimensions it is η < (D − 3)/2.

**Checks** ([`sigma_eqs.py`](../solver/matter/sigma_eqs.py), 48 sympy checks, all pass, 30 s):
- the Einstein tensor for n = 2, 3, 4 in explicit coordinates on $`S^n`$ against the general-n formulas, and the three
  equations above from $`G_{ab} = 2\eta(h_{ab} - \tfrac12 g_{ab}h)`$; the wave equation against □u from the action;
- the self-similar forms (wave, Hamiltonian, momentum), the p, m form, the CSS reduction and the SSH conditions;
- **D = 4: Bizoń & Wasserman, PRD 62, 084031, Eqs. (8)–(17)**, term by term under $`A = a^{-2}`$, $`e^{-\delta} = ca^2`$,
  z = x/c: the wave equation (8)–(9), the Einstein equations (10)–(12), and the CSS system (13), (15), (17);
- the flat ground state $`U_0 = 2\arctan\big(x/\sqrt{n-1}\big)`$ for n = 2, 3, 7/2, 10;
- the large-n inner limit (§6).

## 3. Method

**Background** ([`sigma_css.py`](../solver/matter/sigma_css.py)). Shooting with three unknowns: the centre data
q = U′(0) and c₀ = c(0), and the SSH datum U(1) (U′(1) in D = 4). At both singular points the regular solution is
generated as a Taylor series to order 24 (centre) and 30 (SSH) by truncated-power-series arithmetic, order by order
(each new coefficient enters its order linearly). The series are evaluated at 0.3 of their estimated radius, and
DOP853 (rtol 10⁻¹²) integrates to the matching point x = 0.5. Newton with a central-difference Jacobian converges to
$`|F| \sim 10^{-14}`$. For even n ≥ 4 the SSH series is resonant, so those D are computed at D ± 10⁻³.

**Modes.** $`u = U + e^{\lambda\tau}v(x)`$, $`\ln a \to \ln a + e^{\lambda\tau}\alpha(x)`$, $`\ln c \to \ln c + e^{\lambda\tau}\gamma(x)`$, from the
linearised wave, Hamiltonian and slicing equations, with the gauge γ(1) = 0 (the SSH stays at x = 1). Regular series
at both ends (two free data at each end: v′(0) and γ(0); v(1) and α(1)), integration to x = 0.5, and λ a root of the
4 × 4 matching determinant. The momentum constraint is **not** imposed; on every eigenfunction it is checked and holds to
10⁻¹¹–10⁻¹⁶ relative (field `RM_rel` in the data). The gauge mode (a shift of the light ray called x = 1) has
λ = 1 + c₁ with c₁ = −c′(1); since c′(1) = 0 for every CSS solution here it is exactly λ = 1, and the solver returns
1 to 10⁻¹² (10⁻⁷ at η ≥ 0.3 in D = 4, where U′ is steep at the SSH). Resonances of the SSH series at
λ = n/2 − 1 − k (k = 0, 1, …) are poles of the determinant and are excluded from root brackets. A root lying within a grid
step of such a pole can therefore be missed; this happened once, for λ₋₁ on the flat branch near D = 7.86.

**Counting growing modes.** The number of zeros of the determinant in the box 0.05 ≤ Re λ ≤ 20, |Im λ| ≤ 12
(argument principle on the boundary, adaptive, poles added back) is reported as `modes_in_box`; it includes the
gauge mode. One physical unstable mode means a count of 2.

**Past the light cone.** The background is continued to x > 1 (still t < 0) in ln x up to x = 10⁸. Where it meets
x = c again, $`a^2 \to \infty`$ ($`A = 1/a^2 \to 0`$): an apparent horizon outside the past light cone, Bizoń–Wasserman's
type B. Its position is x_B; "none" means no horizon up to x = 10⁸, i.e. essentially up to t = 0.

**Numerical error.** At D = 4, η = 0.14, moving the matching point (x = 0.4, 0.5, 0.6), lowering the series orders
(16/20 instead of 24/30) and loosening the tolerance change λ by at most 3×10⁻¹³ (2×10⁻⁹ for the low orders) and q by
2×10⁻⁹ ([`sigma_conv.py`](../solver/matter/sigma_conv.py), `data/sigma_conv.jsonl`). The digits quoted below are far
inside this.

## 4. Validation in D = 4 and in flat space

**Flat space (η = 0): the excited solution f₁ and its growing mode** against Bizoń, Chmaj & Tabor [BCT00] (d = 3) and
Biernat, Bizoń & Maliborski [BBM17, Tables 1–2] (d = 3–6, i.e. D = 4–7). [BCT00] write $`f \sim \pi/2 + b(\rho-1)`$, so
their b is U′(1).

| D | quantity | here | published |
|---|---|---|---|
| 4 | U′(0), U′(1), λ₁ | 21.757414, −0.305664, 6.3336253 | 21.757413, −0.305664, 6.333625 [BCT00]; 21.75741, −0.30566, 6.33363 [BBM17] |
| 5 | U′(0), U(1), U′(1), λ₁ | 10.995299, 1.606338, −0.106535, 3.9988317 | 10.9953, 1.60634, −0.10654, 3.99883 |
| 6 | U′(0), λ₁ (mean of D = 6 ± 10⁻³) | 7.821194, 3.0000002 | 7.82119, 3 |
| 7 | U′(0), U(1), U′(1), λ₁ | 6.715075, 1.535335, 0.059052, 2.4262393 | 6.71508, 1.53534, 0.059052, 2.42624 |

Where the growing modes were counted (D = 4, and D = 7.125 and 7.875 on the continuation below) the only other root
in Re λ > 0 is the gauge mode λ = 1, as in [BBM17]. The D = 7 point is in `data/sigma_checks.jsonl`.

**With gravity, D = 4** (`data/sigma_d4.jsonl`). f₁ continued from flat space in η:

| η | 0 | 0.04 | 0.08 | 0.10 | 0.12 | 0.14 | 0.16 | 0.17 | 0.20 | 0.30 | 0.40 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| U′(0) | 21.7574 | 18.6076 | 15.8899 | 14.6742 | 13.5449 | 12.4960 | 11.5217 | 11.0609 | 9.7765 | 6.3968 | 4.0780 |
| λ | 6.33363 | 6.01602 | 5.72272 | 5.58463 | 5.45197 | 5.32454 | 5.20217 | 5.14282 | 4.97187 | 4.47308 | 4.07052 |
| γ = 1/λ | 0.1579 | 0.1662 | 0.1747 | 0.1791 | 0.1834 | 0.1878 | 0.1922 | 0.1944 | 0.2011 | 0.2236 | 0.2457 |
| x_B | none | none | none | none | none | none | 82.5 | 14.6 | 3.73 | 1.38 | 1.06 |

Against the literature:
- **λ = 5.1428 at η = 0.17**, where Lechner et al. [LTHA02] find the DSS–CSS transition at η_c ≈ 0.17 and quote
  "λ ≈ 5.14", slowly varying near η_c, for the coefficient of $`\Delta = -(2/\lambda)\ln(\eta-\eta_c)`$.
- **γ = 0.179–0.188 for η = 0.10–0.14**, against "γ_CSS ≈ 0.18 for 0.1 ≤ η ≲ 0.14" from their evolutions.
- **One unstable mode**: the box count is 2 (gauge + one) at η = 0, 0.10, 0.14, 0.17, 0.30, as [LTHA02] state (one
  unstable mode up to η = 0.5); for the ground state f₀ it is 1 (gauge only) at η = 0, 0.1, 0.3.
- **Apparent horizons outside the past light cone** (Bizoń & Wasserman [BW00], who define α_n as the coupling above
  which the n-th solution has one): for f₀ at η = 0.1 the horizon is at x_B = 3.36861 (their Fig. 2: ρ_B ≈ 3.3683).
  The coupling where x_B first stays below 10⁸ is 0.06923 for f₀ and 0.15273 for f₁; x_B diverges as η decreases to
  the threshold, and $`\ln x_B = K/(\eta-\alpha) + c`$ fits the last 4–5 bisection points better than a power law
  (`data/sigma_d4_horizon.jsonl`, points in `data/sigma_d4_horizon_points.jsonl`), giving **α₀ = 0.0686–0.0687 and α₁ = 0.1518–0.1519**, against
  [BW00]'s α₀ ≈ 0.0688 and α₁ ≈ 0.1518. ([BW00] print α₁ ≈ 0.1588 once, in their §4, and 0.1518 three times; the
  computation supports 0.1518.)

So the D = 4 CSS critical solution is f₁ for α₀ < η < α₁ (black holes form generically, f₁ has one unstable mode and
no horizon outside the light cone), in line with [BW00] and [LTHA02], and the equations and the solver reproduce every
published D = 4 number tested.

## 5. The CSS branch in D: γ(D)

**Fixed η = 0.14** (inside the D = 4 CSS window, below α₁). f₁ is continued from (D, η) = (4, 0) to (4, 0.14) and then
in D in steps of 1/8 (`data/sigma_path_eta0.14.jsonl`), and through its end by pseudo-arclength continuation in
(ln q, c₀, U(1), D) ([`sigma_fold.py`](../solver/matter/sigma_fold.py), `data/sigma_fold_eta0.14.jsonl`). **Computed:**

| D | U′(0) | U(1) | λ | γ = 1/λ | modes in box | f₀: x_B | f₁: x_B |
|---|---|---|---|---|---|---|---|
| 4 | 12.4960 | π/2 | 5.32454 | 0.18781 | 2 | 1.80 | none |
| 4.5 | 9.8013 | 1.61709 | 4.46752 | 0.22384 | | 2.20 | none |
| 5 | 8.0230 | 1.61483 | 3.82990 | 0.26110 | | 2.58 | none |
| 5.5 | 6.8281 | 1.59561 | 3.35828 | 0.29777 | | 2.94 | none |
| 5.625 | 6.5923 | 1.58968 | 3.25974 | 0.30677 | 2 | | none |
| 6.125 | 5.8306 | 1.56433 | 2.92346 | 0.34206 | | 3.28 (D = 6) | none |
| 6.5 | 5.4072 | 1.54514 | 2.71874 | 0.36782 | | 3.60 | none |
| 6.625 | 5.2887 | 1.53891 | 2.65754 | 0.37629 | 2 | | none |
| 7 | 4.9918 | 1.52112 | 2.49108 | 0.40143 | | 3.91 | none |
| 7.5 | 4.7259 | 1.50058 | 2.30057 | 0.43467 | | 4.20 | none |
| 7.625 | 4.6844 | 1.49626 | 2.25722 | 0.44302 | 2 | | none |
| 8.125 | 4.6760 | 1.48478 | 2.09474 | 0.47739 | | | none |
| 8.25 | 4.7497 | 1.48452 | 2.05531 | 0.48654 | | | none |
| 8.375 | 4.9079 | 1.48674 | 2.01503 | 0.49627 | | | none |
| 8.412 | 4.9867 | 1.48824 | 2.00275 | 0.49931 | | | |
| 8.5 | 5.3487 | 1.49572 | 1.97080 | 0.50741 | | | none |
| **8.5444 (fold)** | 6.2607 | 1.51227 | **1.94724** | **0.51355** | | | |
| 8.523 (past the fold) | 7.6311 | 1.52912 | 1.94570 | 0.51395 | λ₋₁ = +0.019 | | |

The momentum constraint holds on every eigenfunction to ≤ 10⁻¹¹ relative, the gauge mode is 1 to 10⁻¹² (10⁻⁴ at the
point nearest the n = 6 resonance, D = 8.0003, not tabulated), and the box count is 2 (one physical growing mode) at
every point where it was taken. f₀ keeps its apparent horizon outside the light cone at least to D = 7.5 and f₁ never
has one, so f₁ stays the black-hole threshold of the D = 4 picture along the whole branch where this was checked.

What the numbers show:
- **γ rises monotonically**, from 0.188 at D = 4 through 1/2 at D ≈ 8.42 (λ = 2.0028 at D = 8.412), to γ = 0.5135 (λ = 1.9472) at the fold, where
  the branch ends.
- **The branch ends at a fold** at D_f = 8.5444 (η = 0.14). Approaching it, the least-stable decaying eigenvalue of f₁
  rises to 0 (λ₋₁ = −0.556, −0.236, −0.102, −0.056 at D = 4, 6.75, 8.125, 8.412; `data/sigma_lowmodes.jsonl`), U′(0)
  turns sharply, and dD/ds → 0 along the arclength. Across the fold λ₋₁ changes sign (−0.0066 at D = 8.5424 before,
  +0.0095 at D = 8.5399 after it; `data/sigma_fold_eta0.14.jsonl`), so the branch that continues back to lower D has two
  growing modes besides the gauge mode, as an f₂ would. λ is smooth through the fold, with a shallow minimum of 1.9446
  just past it. The fold is in D at fixed η, so beyond D_f there is no CSS solution of this family at η = 0.14.
- **No extrapolation to D = ∞ is possible.** Fits of λ in powers of 1/n on D ≤ 7, 8 or 8.5 give λ∞ anywhere between
  −1.4 and 0.38 depending on degree and range; they only reflect that λ(D) heads into a fold.

**Flat space (η = 0)**, the gravity-free limit and the setting of Biernat–Bizoń–Maliborski's conjecture that only f₀
exists for d ≥ 7. f₁ is continued from their d = 6 solution (D = 7, reproduced in §4) upward (`data/sigma_from_flat7.jsonl`)
and through its end by arclength (`data/sigma_fold_flat.jsonl`). **Computed:**

| D | 7 | 7.125 | 7.25 | 7.5 | 7.75 | 7.875 | 7.903 | 7.965 | 7.979 | → 8 |
|---|---|---|---|---|---|---|---|---|---|---|
| U′(0) | 6.7151 | 6.6900 | 6.7008 | 6.8846 | 7.5709 | 8.6192 | 9.0765 | 11.432 | 12.879 | → ∞ |
| λ | 2.42624 | 2.36942 | 2.31471 | 2.21026 | 2.10932 | 2.05792 | 2.04619 | 2.01810 | 2.01142 | → 2.001 |
| γ | 0.4122 | 0.4220 | 0.4320 | 0.4524 | 0.4741 | 0.4859 | 0.4887 | 0.4955 | 0.4972 | → 0.4998 |

One growing mode besides the gauge mode (box count 2 at D = 7.125 and 7.875), and λ₋₁ = −0.074 at D = 7.782. Beyond
that, λ₋₁ sits next to a pole of the light-cone series (§3) and was not resolved. **There is no fold.** The arclength
continuation, run to D = 7.9789 and U′(0) = 12.88, shows dD/ds, the D increments and the λ decrements all shrinking by a
factor of about 0.84 per step, while ln U′(0) grows linearly. So U′(0) → ∞ at finite D: the core of f₁ shrinks to a
point and f₁ collapses onto the singular equator cone. Fits $`D = D_* - K\,U'(0)^{-p}`$ and $`\lambda = \lambda_* + L\,U'(0)^{-p'}`$
to the last 5–7 steps give **D* = 7.9978–7.9984** (p ≈ 4.5; the estimate rises with fewer, later points) and
**λ* = 2.0007–2.0010** (p ≈ 4.1).

*Argued:* both numbers are what the cone predicts. As the core size ε → 0, f₁ outside the core is the cone plus a
time-independent (λ = 0) perturbation, which must be analytic at the light cone. Near the centre it must match the
core's tail, $`c_+(x/\varepsilon)^{\beta_+} + c_-(x/\varepsilon)^{\beta_-}`$, whose $`\beta_-`$ part dies as ε → 0. So the cone needs a
λ = 0 mode $`\propto x^{\beta_+}`$ analytic at x = 1, and its modes are λ = −β − 2k (§6.3), so β₊ = −2. At η = 0 that is
exactly n = 6 (β = −2, −3), i.e. **D = 8, d = 7**. The growing mode of f₁ tends to the cone's λ = −β₊ = 2, and its
decaying λ₋₁ to the cone's λ = 0. So flat f₁ disappears exactly at d = 7, as [BBM17] conjectured (only f₀ for d ≥ 7),
and **its terminal γ = 1/2 comes from the cone at n = 6, not from s-translation**.

With gravity the same condition, $`4 - 2(n-1) + n\,a_e^2 = 0`$, is met at D = 8.351 for η = 0.14. f₁ passes that
dimension with U′(0) = 4.9 and folds at 8.544, and the partner branch beyond the fold runs back toward lower D with U′(0)
growing (6.3 → 8.3 between D = 8.544 and 8.509). That fits a collapse of the partner onto the cone at D = 8.351, with
λ → 2 there; it was not followed (conjecture).

**The value λ = 3 at D = 6.** In flat space [BBM17] find U(1) = π/2, U′(1) = 0 and λ = 3 exactly at d = 5 (D = 6). With
gravity the mean of D = 6 ∓ 10⁻³ gives λ = 3.0000001 at η = 0.14 and 3.0000001 at η = 0.3, with U(1) = π/2 and U′(1) = 0
to about 10⁻⁷ (`data/sigma_checks.jsonl`). So the three statements hold for every η tested, to the precision of the
averaging. D = 6 is where the light-cone series is resonant at first order. No explanation was found; it hints at an
exact structure (perhaps a closed-form f₁) at n = 4.

## 6. Large D

### 6.1 Which scaling of η

Two scalings are natural, for different parts of the problem.

*Fixed η* makes the inner region (x ∼ √n) n-free (**derived**, [`sigma_eqs.py`](../solver/matter/sigma_eqs.py) §6).
In s = r²/(2n) and the physical time t, the combination (S) + 2(H) of the slicing and Hamiltonian equations is exactly
free of n-sized terms. With a² − 1 = O(1/n) from (H) it gives

```math
\partial_t\Big(\frac{u_t}{c}\Big) = c\Big(u_s - \frac{\sin 2u}{4s}\Big), \qquad c_s = \frac{2\eta\,u_t^2}{c} .
```

This is the system the coordinator derived independently ([STRUCTURE.md §6](STRUCTURE.md)). The potential's own gravity,
the conical deficit 2η/(n − 1), drops out at D = ∞, and gravity acts through the lapse alone, driven by the kinetic
energy. This is the region where the scalar's period and its λ = 2 are selected. The 1/s term breaks the s-translation,
and the coordinator's search for point symmetries ([`struct_symmetry.py`](../solver/matter/struct_symmetry.py)) finds only
the time translation (λ = 1), the scaling and the gauge. **Nothing in this model fixes λ∞ = 2** (derived as far as point
symmetries go).

*Fixed η̂ = η/(D − 3)* keeps the conical deficit 2η̂ and the bound η̂ < 1/2 (Bizoń–Wasserman's η < 1/2 in D = 4). It
controls everything that needs $`a^2 - 1 = O(1)`$: apparent horizons outside the light cone, which decide whether f₀
forms black holes and so whether f₁ is a black-hole threshold, and the equator structure of §6.3. In the inner region it
makes gravity O(n); with u = φ/√η the inner problem tends to the ℓ = 1 multiplet's (STRUCTURE.md).

**Fixed η is the continuation used in §5.** It is natural for whatever is selected in the inner region. Along it gravity at
x = O(1) weakens (deficit 2η/(n − 1)), yet f₀ keeps its horizon to D = 7.5 at η = 0.14, so in the range where f₁ exists
the black-hole role of the D = 4 picture is not yet lost. A fixed-η̂ continuation was not run (§9).

### 6.2 The D = ∞ CSS problem

**Derived** ([`sigma_inner.py`](../solver/matter/sigma_inner.py) `check`). With u = W(ξ), c = C(ξ), ξ = r/(√n (−t)) = x/√n:

```math
\xi^4W'' + 2\xi^3W' - \xi^4W'\frac{C'}{C} - C^2\xi W' + \frac{C^2\sin 2W}{2} = 0, \qquad CC' = 2\eta\,\xi^3W'^2, \qquad C(0) = 1 .
```

ξ = 0, where the whole interior of the past light cone sits at D = ∞, is an irregular singular point. Its slow (formal)
solutions $`W = A\xi + O(\xi^3)`$ form a one-parameter family, $`A = \sqrt n\,U'(0)`$ of the finite-n solution, and its fast
solution carries $`e^{-1/(2\xi^2)} = e^{-n/(2x^2)}`$. At finite n the solution must be analytic at the light cone. As for the
scalar (README, Conjecture 1), the D = ∞ form of that condition is **taken** to be the vanishing of the Stokes multiplier
of the formal series. It is evaluated by the lateral method: the solution asymptotic to the series in the sector
π/4 < arg ξ < 3π/4 (where it is unique) is carried to the real axis, and the jump is 2i Im W⁺. A zero is accepted only if it
is the same at two radii (0.4 and 0.28); zeros of the fast solution at one radius are rejected that way. The linearised
problem (u → W + e^{λτ}v, c → C(1 + e^{λτ}g), g(0) = 0) is treated the same way.

**Computed:**
- *η = 0.* The only regular zero for 0.25 ≤ A ≤ 4.5 is A = 2, the exact solution W = 2 arctan ξ (the D = ∞ limit of
  the flat ground state $`2\arctan(x/\sqrt{n-1})`$; jump 10⁻¹²). Its D = ∞ growth rates in −1.5 ≤ λ ≤ 8 are λ = 1 (the time
  translation, exact) and λ = −0.5936 only: the ground state is stable at D = ∞, as at every finite d. The finite-D flat
  ground state's least-stable decaying eigenvalue is −0.58124 at D = 8.5 and −0.58831 at D = 16.5
  (`data/sigma_lowmodes.jsonl`); λ∞ + a/n through the two gives −0.5941, against −0.5936 from the inner problem. This
  checks the D = ∞ perturbation machinery against finite D.
- *η = 0.14.* The ground state moves to A₀ = 1.9225686. **The finite-D ground state f₀ converges to it**: its amplitude
  $`\sqrt n\,U'(0)`$ at D = 4.5, 5, …, 7.5 (7 points) fits $`A_\infty + \sum_{k=1}^{m} a_k n^{-k}`$ with
  A∞ = 1.92257, 1.92282, 1.92247 for m = 3, 4, 5 (largest residual 8×10⁻⁷ at m = 3), i.e. **A∞ = 1.9226 ± 0.0003
  against the D = ∞ zero 1.9225686**, with a₁ ≈ 0.93 (`data/sigma_f0scan_eta0.1400.jsonl`). The D = ∞ number uses no
  finite-D input, so this is an independent check of the selection rule on the CSS ground state.
- *Excited solutions.* No other zero appears at η = 0 and 0.1 for 0.25 ≤ A ≤ 4.5, nor at η = 0.14 for 1 ≤ A ≤ 3.
  (The η = 0.1 scan and the wide η = 0 scan used the first version of the lateral path, without the two-radius test.) At
  η = 0.5 and 1 the jump
  changes sign near A ≈ 3.3–3.8, but there the real-axis solution blows up (W′ → ∞) before ξ = 0.4: these are not
  regular solutions. For A ≳ 5 the jump falls below double precision ($`\propto e^{-cA^2}`$), so excited D = ∞ CSS solutions
  at large A are not excluded by this search.

**Why f₁ is not described by the weak-field inner problem (argued).** Along the η = 0.14 branch, f₁ never enters the
regime U = O(n^{-1/2}) at x = O(1) that the slow family assumes: U′(0) stays at 4.6–5.4 for D = 7–8.5, so
A = √n U′(0) ≈ 11–14 and grows like √n. U(1) stays near 1.49, close to the equator. For such a solution the inner region
sees $`u = \pi/2 - w`$ with $`w \approx (-t)/(\kappa r)`$ small (κ = U′(0)), and the inner equation linearised about the
equator is $`w_{tt} = w_s + w/(2s)`$, with c = 1 + O(w²). With $`w = s^{-1/2}g`$ this is the free heat equation $`g_{tt} = g_s`$ (the ℓ = −1 case of the
multiplet substitution of STRUCTURE.md §3), and $`w = K/\xi`$ is an exact CSS solution with no Stokes jump. So at leading
order the D = ∞ problem neither selects a near-equator CSS solution nor fixes its growth rate. Whatever selects f₁ and its λ
at large D would sit at relative order n^{−1/2}. This is moot at η = 0.14, where the branch folds at D = 8.5444.

### 6.3 The equator cone, and what it does not decide

U ≡ π/2, c = 1, $`a^2 = a_e^2 = (n-1)/(n-1-2\eta)`$ is an exact, singular CSS solution: the self-similar "global monopole"
with deficit 2η/(n − 1). Its linear perturbations decouple from the metric, and $`v = x^\beta`$ with $`\lambda = -\beta`$ is an exact
mode, analytic at the light cone, for either root of $`\beta^2 + (n-1)\beta + n a_e^2 = 0`$ (**derived**,
[`sigma_equator.py`](../solver/matter/sigma_equator.py)). The roots are complex, so that solutions spiral around the cone near
the centre and an infinite sequence f₁, f₂, … of excited CSS solutions accumulates at it, iff

```math
n^2 - (6+2\eta)\,n + 1 + 2\eta \lt 0 .
```

At η = 0 this is n < 3 + 2√2, i.e. d ≤ 6, the dimensions in which [BBM17] find an infinite family. With gravity the edge
is $`n_c(\eta) \approx 2\eta + 6`$: D_c = 7.828, 8.000, 8.069, 8.121, 8.702, 9.606 for η = 0, 0.1, 0.14, 0.17, 0.5, 1. At fixed
η̂ the condition is $`(1-2\hat\eta)n^2 - (6-4\hat\eta)n + 1 - 2\hat\eta \lt 0`$, with D_c = 8.85, 9.42, 10.55, 13.9 and 24.0 for
η̂ = 0.1, 0.14, 0.2, 0.3, 0.4. D_c → ∞ only if η̂ → 1/2 with n(1 − 2η̂) below 4, a deficit within O(1/n) of total.

Above D_c the infinite family cannot exist, but its low members can: **f₁ continues smoothly through D_c** (η = 0.14:
D_c = 8.07, the fold is at 8.5444; flat: D_c = 7.83, the collapse at 8). The first draft of this page argued that f₁ ends at
D_c by merging into the cone, and the continuation refuted that. What does end the flat branch is the second condition
of §5: the cone's exponent β₊ reaching −2, where a time-independent perturbation of the cone becomes analytic at the
light cone (D = 8 at η = 0, D = 8.351 at η = 0.14).

## 7. The DSS branch

Not computed: the time went to the CSS branch, and the ℓ = 1 multiplet (the η → ∞ limit) is another agent's task. What is known about it here:
- *Argued.* At fixed η its D = ∞ inner problem is the system of §6.1, n-free and without s-translation, so its λ∞ is not
  protected at 2. It depends on η and tends, as η → ∞, to the ℓ = 1 multiplet's value (STRUCTURE.md). In D = 4 the
  DSS exponent is about 0.11 for η ≳ 0.2 and tends to 0.1185 ± 0.0005 as η → ∞ [HLPTA00]. [LTHA02] give γ_DSS ≈ 0.11.
- *Not known.* Whether the DSS branch exists at all D for fixed η, and where η_c(D) (the DSS–CSS switch) goes. A DSS
  cross-check at one large-D point (the task's item 3) would need the periodic solver of `solver/dss.py` extended by the
  field u and the potential terms of §2; the equations for that extension are in §2 and checked in `sigma_eqs.py`.

## 8. Verdict

**Does γ∞ differ from 1/2 for the σ-model?** For the CSS critical solution the question has no answer, because the
branch does not reach large D. At fixed η = 0.14 it ends at a fold at D_f = 8.5444; in flat space it collapses onto the
singular equator cone as D → 8 (computed). Along the way γ rises monotonically from 0.188 (D = 4, η = 0.14) or 0.158
(D = 4, flat). At the end it is 0.5135 (η = 0.14 fold) and → 1/2 (flat collapse). That 1/2 is not a D → ∞ limit, and
nothing in the mechanism of Proposition 5 is at work: f₁ never enters the weak-field inner regime where the s-translation
acts (U′(0) stays of order 5–13, not of order n^{−1/2}). It comes from the equator cone instead (§5, argued). A collapse
onto the cone needs its exponent β₊ = −2, and the collapsing solution inherits the cone's growth rate −β₊ = 2. So
"γ = 1/2" appears at the end of the flat CSS branch for a reason unrelated to the scalar's, and at η = 0.14 the fold comes
first, at γ = 0.5135.

**Why the prediction of LITERATURE.md still stands, structurally.** At fixed η the D = ∞ inner problem is n-free, keeps O(1)
gravity, and contains the term sin(2u)/(4s), which breaks the s-translation (derived here and in STRUCTURE.md). Only
λ = 1 (time translation) is left as an exact growth rate. So for any critical solution that does survive to D = ∞, which
here can only be the DSS branch (not computed), γ∞ = 1/2 is not protected. Its value would be a transcendental function of
η, tending to the ℓ = 1 multiplet's as η → ∞.

**Fixed η or another scaling?** Fixed η is the natural continuation for the inner region, where the scalar's γ∞ is
decided: it gives an n-free D = ∞ problem with one parameter, and the D = ∞ selection rule checks out on its CSS ground
state (A∞ = 1.9226 ± 0.0003 from D ≤ 7.5 against 1.9225686 from the inner problem alone). Fixed η̂ = η/(D − 3) is natural
for horizons and for the conical deficit, but sends η → ∞ in the inner region. The infinite family of excited CSS
solutions needs $`n^2 - (6+2\eta)n + 1 + 2\eta \lt 0`$, which fails at large n for any fixed η or η̂ < 1/2. f₁ itself ends
a little above that edge, at a fold or at the dimension where the cone's β₊ = −2. At fixed η that dimension solves
$`n^2 - (7+4\eta)n + 6 + 12\eta = 0`$ and is finite for every η (n ≈ 4η + 4 at large η), so the CSS branch has no D = ∞ limit
at any fixed η (argued; computed at η = 0, 0.14). At fixed η̂ the condition $`n(1-4\hat\eta) = 6(1-2\hat\eta)`$ has a solution
only for η̂ < 1/4. For 1/4 ≤ η̂ < 1/2 neither the end of f₁ nor its large-D fate is covered here.

**Merging with the flat-space blowup profile.** Not seen. f₁ stays far from f₀ (U′(0) about 5–8 against 1 or less for f₀
at D ≈ 7–8.5). f₀ keeps a horizon outside the light cone at η = 0.14 to D = 7.5, and in D = 4 it has no growing mode
besides the gauge mode. f₁ ends by merging with
another excited solution at a fold (η = 0.14) or by collapsing onto the singular equator cone (flat space), not with
the blowup profile.

## 9. Open issues

- **The fold partner.** The branch that continues past the fold has λ₋₁ > 0 (two growing modes besides the gauge mode),
  as f₂ would. That it is f₂ is not checked (f₂ was not computed).
- **The collapse onto the cone** (flat, D → 8) rests on arclength data to U′(0) = 12.9 and a power-law fit. The
  argument for β₊ = −2 and λ → 2 is a matching argument, not a proof. The prediction that the η = 0.14 partner branch
  collapses at D = 8.351 is untested, as is the general rule "a fold first if the branch reaches the β₊ = −2 dimension
  with finite U′(0)".
- **λ = 3 at D = 6** (with U(1) = π/2, U′(1) = 0) for η = 0, 0.14, 0.3, to 10⁻⁷: unexplained.
- **Black-hole role along the branch.** α₀(D) and α₁(D) were computed only at D = 4. Along η = 0.14, f₀ keeps a horizon
  outside the light cone to D = 7.5 and f₁ has none to D = 8.5, so the D = 4 roles hold there. Beyond D = 7.5 for f₀, and
  for other η, this is not checked.
- **Fixed η̂.** Not run. The cone criterion puts the end of the infinite family at D_c = 9.42 for η̂ = 0.14, and f₁ would
  presumably fold somewhat above it, as at fixed η. Not tested.
- **D = ∞ excited CSS solutions at large A** need extended precision (the jump is $`\propto e^{-cA^2}`$); the double-precision
  lateral search covers A ≲ 4.5 only.
- **The D = ∞ selection rule** is the README's Conjecture 1 transplanted. The f₀ check (§6.2) supports it for the CSS
  ground state at η = 0.14; it is not a proof.

## 10. Scripts and data

All in [`solver/matter/`](../solver/matter/), run with `nice -n 10 python3 <script>` from that directory, single process.
Data in `solver/matter/data/`.

| Script | What it does | Time (2 shared cores) |
|---|---|---|
| `sigma_eqs.py` | 48 sympy checks: Einstein tensor n = 2–4, the D-dimensional equations, self-similar and p, m forms, CSS reduction, SSH conditions, Bizoń–Wasserman (8)–(17), flat ground state, inner limit | 30 s |
| `sigma_css.py` | the CSS solver (series at both ends, shooting, modes, determinant, mode count, continuation past the SSH); `python3 sigma_css.py flat` reruns the flat checks | library |
| `sigma_runs.py d4`, `d4f0`, `alpha` | D = 4: f₁ and f₀ against η, horizons, α₀, α₁ (`sigma_d4.jsonl`, `sigma_d4_horizon.jsonl`) | 30 min |
| `sigma_runs.py path NAME h D η …` | f₁ along a path in (D, η) with f₀ alongside (`sigma_path_*.jsonl`) | 1–2 min per point |
| `sigma_runs.py from NAME D0 η q c0 U1 λ0 D1 h` | f₁ continued in D from a given solution (`sigma_from_*.jsonl`) | 1–2 min per point |
| `sigma_runs.py f0scan η Dmax h` | f₀ in D (`sigma_f0scan_*.jsonl`) | 10–30 s per point |
| `sigma_fold.py NAME η D q c0 U1 λ0 h steps` | pseudo-arclength continuation through the fold (`sigma_fold_*.jsonl`) | 1–3 min per step |
| `sigma_lowmode.py D η q c0 U1` | real eigenvalues in (−1.2, 0.95), e.g. λ₋₁ (`sigma_lowmodes.jsonl`) | 1–3 min |
| `sigma_conv.py D η q c0 U1 λ0` | resolution check (`sigma_conv.jsonl`) | 5 min |
| `sigma_checks.py` | flat f₁ at D = 7; f₁ at D = 6 ∓ 10⁻³ for η = 0.14, 0.3 (`sigma_checks.jsonl`) | 5–10 min |
| `sigma_equator.py` | the equator cone: exact modes (sympy), D_c(η), D_c(η̂) | seconds |
| `sigma_inner.py check / scanA / zeros / modes` | the D = ∞ CSS problem: sympy checks of the reduction and linearisation, Stokes jump, zeros in A, growth rates | 1–10 min |

The point at D = 8.625 in `sigma_path_eta0.14.jsonl` is a Newton failure next to the fold, not a result.
`sigma_path_flat.jsonl` (η = 0 from D = 4, stopped at D = 5.25 for CPU) overlaps the validation points.

## 11. References

- [BW00] P. Bizoń, A. Wasserman, PRD 62, 084031 (2000), [gr-qc/0006034](https://arxiv.org/abs/gr-qc/0006034).
- [HLPTA00] S. Husa, C. Lechner, M. Pürrer, J. Thornburg, P. C. Aichelburg, PRD 62, 104007 (2000), [gr-qc/0002067](https://arxiv.org/abs/gr-qc/0002067).
- [LTHA02] C. Lechner, J. Thornburg, S. Husa, P. C. Aichelburg, PRD 65, 081501 (2002), [gr-qc/0112008](https://arxiv.org/abs/gr-qc/0112008).
- [BCT00] P. Bizoń, T. Chmaj, Z. Tabor, Nonlinearity 13, 1411 (2000), [math-ph/9912009](https://arxiv.org/abs/math-ph/9912009).
- [BBM17] P. Biernat, P. Bizoń, M. Maliborski, Nonlinearity 30 (2017), [arXiv:1608.07707](https://arxiv.org/abs/1608.07707).
- [BB15] P. Biernat, P. Bizoń, Comm. Math. Phys. 338, 1443 (2015), [arXiv:1409.8214](https://arxiv.org/abs/1409.8214).
- [OVCU07] I. Olabarrieta, J. F. Ventrella, M. W. Choptuik, W. G. Unruh, PRD 76, 124014 (2007), [arXiv:0708.0513](https://arxiv.org/abs/0708.0513).

The numbers quoted from [BW00], [HLPTA00], [LTHA02], [BCT00] and [BBM17] were read from the arXiv versions in this session
(Oct 2026). [BW00] print α₁ ≈ 0.1588 once and 0.1518 three times; [HLPTA00] contains no CSS data (the CSS window and
λ ≈ 5.14 are from [LTHA02]).


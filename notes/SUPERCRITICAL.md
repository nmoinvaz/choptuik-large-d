# The supercritical branch of Ecker, Ecker & Grumiller at large D

Ecker, Ecker and Grumiller (EEG, PRL 136, 191401 (2026), [arXiv:2601.14358](https://arxiv.org/html/2601.14358v1))
describe, in their supplemental material ("More on CSS solutions") and in "Summary and Outlook", a second branch of
large-D self-similar solutions, obtained from their regular one by β → iβ, f₀ → if₀: $`f_{LO} = f_0\sqrt{\beta^2x^2-1}`$,
$`\Pi_{LO} = \beta/\sqrt{\beta^2x^2-1}`$ on β²x² > 1, with a second free function, β without zeros, and the singular point
β²x² = 1 "sometimes used as a proxy for the apparent horizon, see, e.g., [14]" (Rozali and Way). This note analyses that
branch. Scripts and logs: [`solver/supercrit/`](../solver/supercrit/).

**Results** (status words as in AGENTS.md):

1. In this repository's variables the leading order is y = p − m = √2 βx/w, c = f w, (n−1)(b−1) = β²x²/w²,
   w = √(β²x² − 1), and it solves the exact equations to O(1/n) for arbitrary β(τ), f(τ) (*checked symbolically*).
   Both branches are one family, y = √2 x/√(x² + C(τ)), with C = +1/β² (regular) or −1/β² (supercritical).
2. **Nothing is forced at any order.** The expansion in 1/n exists to all orders, with two free functions of τ per order,
   which are shifts of β and f (*proved* by hand; *checked symbolically* through NNLO for CSS and through NLO for
   general β(τ), f(τ), [NNLO general: see §2.4]). Every order solves the exact equations to the next power of 1/n
   (*numerical evidence*). There is no analogue of EEG's Eq. (20), and f(τ) is pure gauge (a time reparametrization)
   unless the solution is restricted to CSS, where κ_f = fβ is physical.
3. **β²x² = 1 is not an apparent horizon.** The expansion of outgoing null rays, θ₊ = n/(r√b), is positive at every
   order. The series is one in 1/(n z²), z = β²x² − 1, and breaks down at z ~ n^{−1/2}. There the compactness
   2m/r^{n−1} = 1 − 1/b is only of order n^{−1/2}. Inside that layer the solution passes smoothly through β²x² = 1 into a
   region where the polar lapse collapses exponentially and 1 − 1/b → 1.627 κ̃/√n: a **lapse-collapse front, spacelike**,
   not a marginally trapped surface (*numerical evidence*: exact CSS equations at n = 100–6400; for general β(τ) a
   formal layer analysis, matched symbolically to the outer series at NLO). Only for a front at rest (β + β′ = 0) does the compactness behind it become large; Rozali and Way's
   exact divergence is of that static kind.
4. **Known solutions.** The branch is not a large-D limit of the Roberts family or of its D-dimensional versions
   (Frolov; Kim, Moon and Yee; Clark and Pimentel), whose scalar field is invariant under the homothety, while here
   ψ ≈ √n fβ τ (*proved*). Its large-x limit is an exact stationary solution, present at every D, which at D = 4 is
   Brady's stationary solution of the κ ≠ 0 self-similar family (PRD 51, 4168 (1995)). The regular branch and the
   supercritical branch are exchanged by the r²-shift that generates the λ = 2 mode of Proposition 5 (*checked
   symbolically*): a regular solution pushed along that mode develops the supercritical front, which decelerates and
   stops at r² = 2n|σ₀|.
5. **Finite D.** Only the CSS members are cheap at finite D (ordinary differential equations); §5 lists what they would
   show. The physically interesting case, the evolution along λ = 2 into a horizon, needs an evolution code in a
   horizon-penetrating gauge, not the project's self-similar solvers.

## 1. The branch in this repository's variables

**The map** (`eqmap.py`, *checked symbolically*). EEG's Eqs. (3a)–(3d), with their ε_E = 1/(D−1) = 1/(n+1), are the
repository's E1–E4 (README "Large D"; `hp/src/model.rs`) under

```math
c = f,\qquad (n-1)(b-1) = x^2\Omega,\qquad y \equiv p - m = \sqrt2\,x\,\Pi,\qquad \sigma \equiv p + m = \sqrt2\,x^2\Psi ,
```

with (3a) = E2/x², (3b) = E1/(x²c), (3c) = (A)/(√2x²), (3d) = (B)/(√2x³), where (A), (B) = E3 ∓ E4. Their (3b) is
printed with 1/r; 1/x is the only reading consistent with their Eq. (11).

**Leading order** (`lo_check.py`, *checked symbolically*). With ε = 1/n, w = √(β²x² − 1), β(τ) > 0, f(τ) > 0:

```math
y = \frac{\sqrt2\,\beta x}{w},\qquad c = f\,w,\qquad (n-1)(b-1) = \frac{\beta^2x^2}{w^2},\qquad
p + m = \varepsilon\,\frac{x\,(x\partial_x + \partial_\tau)\,y}{c} = -\varepsilon\,\frac{\sqrt2\,x^2(\beta+\beta')}{f\,w^4} .
```

The O(1) part of every residual of E1–E4 vanishes identically and the O(ε) part does not; for β = 1.7 + 0.4 sin τ,
f = 0.8 + 0.3 cos 2τ the residuals times n are constant (1.62, 0.137, 4.08 for E2–E4) from n = 10² to 10⁴.

*A sign in EEG's supplement.* The supplement prints only f_LO and says the other fields follow by "analogous changes of
1 + β²x² → β²x² − 1 in all other fields in (S1)". Applied literally to (S1)'s Ψ_LO = β/(f₀(1+β²x²)²) this gives
+β/(f₀(β²x²−1)²), which with f_LO = +f₀√(β²x²−1), f₀ > 0, has the wrong sign: EEG's own Eq. (8) gives
Ψ_LO = D₁Π_LO/f_LO = −β/(f₀(β²x²−1)²). The continuation itself is consistent (it also gives f_LO = −f₀√(β²x²−1) on the
branch √(1−β²x²) = i√(β²x²−1)); only the rule as worded is off by a sign. `lo_check.py --mutate=psisign` shows the
wrong sign failing at O(1).

**One family, and what is gauge** (`shift_check.py`, *checked symbolically*). The leading-order equations,
x y_x = (1 − y²/2) y, x c_x = c y²/2, (n−1)(b−1) = y²/2, are solved by

```math
y = \frac{\sqrt2\,x}{\sqrt{x^2 + C(\tau)}},\qquad c = k(\tau)\sqrt{x^2 + C(\tau)}
```

for any real C(τ): C = 1/β² is EEG's regular branch (k = f₀β), C = −1/β² the supercritical one (k = fβ). In
r = x e^{−τ}, t = −e^{−τ}, y and b depend only on r and ρ(t) = C t², and c = K(t)√(r² + ρ) with K = k/(−t). A time
reparametrization t → T(t) is an exact symmetry of the polar-areal equations (α → α/T′), and it changes K only. So in
the general τ-dependent class **f is gauge**; the physical datum is ρ(t), i.e. the singular sphere r_s(t) = √(−ρ) =
(−t)/β. In the CSS class only t → λt survives, and κ_f = fβ (= c/x at x → ∞, and minus the homothetic derivative of
ψ/√n) is physical. EEG's "second free function" is therefore a gauge function except for its CSS restriction.

**The stationary solution** (`y1_modes.py`, *checked symbolically*). At C = 0, and as x → ∞ for every member,
y → √2. For every n and κ > 0,

```math
p = -m = 1/\sqrt2,\qquad b = \frac{n}{n-1},\qquad c = \kappa\,x,\qquad \psi = -\sqrt n\,\kappa\,\ln(-t)
```

solves E1–E4 exactly: a scalar field linear in ln(−t) in a metric with constant g_rr, (∇r)² = (n−1)/n. Both branches
approach it at large x. Its perturbations ∝ x^λ satisfy (exact, finite n)

```math
\lambda\,(\lambda + n)\,\big[(1-\kappa^2)\lambda^2 - \kappa^2 n\,\lambda - 2\kappa^2 n\big] = 0 ,
```

with the slow decaying root λ₋ = −2 + 4(1 − κ²)/(κ²n) − 16(1 − κ²)²/(κ⁴n²) + O(n⁻³).

## 2. The 1/n expansion: what is forced and what is free

### 2.1 Scheme (`expand.py`)

In z = β²x² − 1 ∈ (0, ∞) and x² = (z+1)/β², write

```math
y = \sqrt2\,\frac{x}{w}\,\hat Y,\qquad c = w\,\hat C,\qquad (n-1)(b-1) = \hat U,\qquad p + m = \sqrt2\,x^2\,\hat S,
\qquad \hat Y = \sum_{k\ge0}\varepsilon^k Y_k,\ \ldots,\ \hat S = \sum_{k\ge1}\varepsilon^k S_k ,
```

with Y₀ = β, C₀ = f, U₀ = (z+1)/z, S₁ = −(β+β′)/(f z²). The equations become rational in z (x∂_x = 2(z+1)∂_z; ∂_τ at
fixed x is D + 2(β′/β)(z+1)∂_z, with D = ∂_τ at fixed z), and at order k ≥ 1:

- E2 gives U_k algebraically;
- (B) reduces to −2f((z+1)/z)(zY_k)′ = (known): the C_k terms cancel identically, because (B) is homogeneous of degree
  one in c at leading order, so Y_k = z⁻¹[∫(…)dz + A_k(τ)];
- E1 reduces to 2(z+1)C_k′ = (known), so C_k = ∫(…)dz + B_k(τ);
- (A) at the next order gives S_{k+1} algebraically.

**Proposition S** (*proved* by hand). For any smooth β(τ) > 0, f(τ) > 0 and any functions A_k(τ), B_k(τ), there is
exactly one formal solution of this form, every order defined on all of 0 < z < ∞. No step has a solvability
condition: the ODEs for zY_k and C_k have no singular point on z > 0 (in particular none at the self-similarity
horizon x = c, which in this scaling enters only through the O(ε) term x²∂_x(p+m)), and U_k, S_{k+1} need only division
by z f. The homogeneous parts are first-order changes of the leading order: A_k/z in Y_k is β → β − ε^k A_k, and B_k
is f → f + ε^k B_k (`expand.py 1`, the free run, checks this at k = 1 symbolically: `[True, True, True]`). The
"zero" normalization A_k = B_k = 0 (zY_k and C_k have no z⁰ term in their expansion at z = 0 apart from logarithms)
defines β and f beyond leading order.

The integrals are elementary: every order is a polynomial in ln z with coefficients that are Laurent polynomials in z
(no ln(1+z) and no dilogarithm occurred through NNLO; a dilogarithm would need a nonzero coefficient of ln^a z/(z+1),
a ≥ 1, which `expand.py` would report). `expand.py` integrates these forms term by term in closed form, falls back to
sympy otherwise, and checks every antiderivative by differentiation. The term-by-term integrator and sympy's
`integrate` give identical series for the general-τ NLO and the CSS NNLO (`cmp_series.py`).

### 2.2 Next-to-leading order

General β(τ), f(τ) (zero normalization; factored by `nlo_factor.py`, log `logs/nlo_factor.log`):

```math
Y_1 = -\Lambda\,\frac{\ln z}{z} - \frac{2(\beta+\beta')^2}{\beta^3f^2\,z^2},\qquad
\Lambda = \frac{2\beta^4f^3 - 2\beta^2 f - \beta^2 f' - 5\beta\beta' f - \beta\beta' f' + \beta\beta'' f - 4\beta'^2 f}{2\beta^3f^3},
```

and C₁, U₁, S₂ of the same type (`logs/expand_K1_free.log`). For CSS, with κ = κ_f = fβ:

```math
Y_1 = \frac{1}{\beta f^2}\Big[(1-\kappa^2)\frac{\ln z}{z} - \frac{2}{z^2}\Big],\qquad
C_1 = \frac{1}{\beta^2 f}\Big[(\kappa^2-1)\frac{\ln z}{z} - \frac1z + \frac{1}{z^2}\Big],\qquad
U_1 = \frac{2(z+1)}{z^2}\Big[\Big(\frac{1}{\kappa^2}-1\Big)\ln z + 1 - \frac{2}{\kappa^2 z}\Big],
```
```math
S_2 = \frac{1}{\beta f^3}\Big[\frac{2(\kappa^2-1)(z+2)\ln z}{z^3} + \frac{2-4\kappa^2}{z^2} + \frac{9-5\kappa^2}{z^3} + \frac{11}{z^4}\Big].
```

**Forced: nothing. Free: A₁, B₁.**

### 2.3 Next-to-next-to-leading order (CSS)

`expand.py 2 --norm=zero --css` (log `logs/expand_K2_zero_css.log`): Y₂, C₂, U₂, S₃ contain ln²z, ln z and rational
terms. **Forced: nothing. Free: A₂, B₂.** The series computed with sympy's integrator and with the exact integrator
are identical (`cmp_series.py logs/expand_K2_zero_css_sympyint.pkl logs/expand_K2_zero_css.pkl`).

### 2.4 Next-to-next-to-leading order, general β(τ), f(τ)

`expand.py 2 --norm=zero` (log `logs/expand_K2_zero.log`). STATUS: see the end of this file (§9).

### 2.5 The two ends of the interval

**z → 0 (β²x² → 1).** Each order is more singular by z⁻²: Y_k, C_k ~ z^{−2k}, U_k ~ z^{−2k−1}, S_k ~ z^{−2k}
(through NNLO). The leading terms give Y = β[1 − 2ε/(κ̃z)² + …] with

```math
\tilde\kappa = \frac{f\,\beta^2}{|\beta+\beta'|}\quad(\text{CSS: } \tilde\kappa = \kappa_f = f\beta) ,
```

so near the singular point the series is one in **ε/(κ̃z)²** and breaks down at z ~ n^{−1/2}/κ̃. No boundary condition
can be imposed at z = 0 within the series. The layer of §3.3 is matched to it instead, and it fixes nothing physical:
the layer is unique up to its two symmetries (a shift in θ and a scaling), which are the outer β and f, and a change of
the normalization A_k (β → β − ε^k A_k) moves it by O(n^{1/2−k}) in its own variable θ.

**z → ∞ (spatial infinity on a slice).** The terms ln z/z, ln²z/z, … are the expansion of x^{λ₋} for the stationary
solution's slow mode: from the CSS series, λ₁ = −1.0779457959, the ln² coefficient = λ₁²/8 = 0.1452458924 and
λ₂ = −1.1619671389, equal to the exact root to 10 digits at κ = 1.17 (`y1_modes.py`). So the logarithms are not an
obstruction and impose no condition at x → ∞. (For general β(τ) the logarithm's coefficient Λ above plays the same
role; not compared with an exact exponent.)

**The self-similarity horizon.** For κ_f > 1 the supercritical CSS solution has c = x at x_lc = f/√(f²β² − 1) > 1/β
(for κ_f < 1 there is none). As in Proposition 1 it is not a singular point of any order, so analyticity there is
automatic order by order; any condition from it is beyond all orders.

### 2.6 Check against the exact equations (`formal_check.py`)

The truncated series, rebuilt as p, m, b, c in (x, τ) and differentiated numerically at 30 digits (no use of the
operator algebra of `expand.py`), leaves residuals of E1–E4 falling with fitted exponents (n = 200–1600, eight points
with z = 0.5–10):

| series | profile | exponents of E1/c, E2, E3, E4 | expected |
|---|---|---|---|
| NLO (K = 1) | β = 1.7 + 0.4 sin τ, f = 0.8 + 0.3 cos 2τ | −1.997, −2.011, −1.995, −2.002 | −2 |
| NNLO (K = 2), CSS | β = 1.3, f = 0.9 | −2.969, −3.023, −2.988, −3.001 | −3 |
| NNLO, CSS, Y₂ and C₂ dropped (mutation) | same | −2.000, −1.961, −2.075, −1.915 | fails |

### 2.7 Answers for task 2

| order | forced | free |
|---|---|---|
| LO | the branch choice C < 0 (β²x² > 1); β has no zeros | β(τ), f(τ) (f gauge outside CSS) |
| NLO | nothing | A₁ (β), B₁ (f) |
| NNLO | nothing | A₂, B₂ |
| all orders | nothing (Proposition S) | A_k, B_k |

- **Does a condition constrain β(τ), f(τ) or the period, like EEG's Eq. (20)?** No. Their condition comes from
  f ≤ 1 near the self-similarity horizon x = 1 at the zeros of β, with f(1) = 1 imposed. In this branch β has no zeros
  (the domain β²x² > 1 would be empty there), no horizon condition is imposed, and the inequality x ≤ f of their
  Eq. (6) fails near β²x² = 1 by construction. For members with β periodic and positive (discretely self-similar with a
  homothetic slope; the half-period sign flip of the critical solution is excluded because β has no zeros) the period
  is free at every order.
- **Or does everything stay free, as in Proposition 1?** Yes, as in Proposition 1, and for the same structural reason
  (no singular point of the order-by-order equations on the interval). Any selection would have to come from terms
  beyond all orders or from the layer and the interior, which the series does not see.

## 3. Is β²x² = 1 an apparent horizon?

### 3.1 The expansion of outgoing null rays

In the polar-areal metric −α²dt² + a²dr² + r²dΩ²_{n}, with the null vectors k_± = α⁻¹∂_t ± a⁻¹∂_r,

```math
\theta_\pm = \pm\frac{n}{a\,r} = \pm\frac{n}{r\sqrt b},\qquad \theta_+\theta_- \propto (\nabla r)^2 = \frac1b = 1 - \frac{2m}{r^{n-1}} .
```

A marginally trapped sphere (θ₊ = 0, θ₋ < 0) needs 1/b = 0. Order by order, with (n−1)(b−1) = Û and
n − 1 = (1−ε)/ε,

```math
\frac{r\,\theta_+}{n} = b^{-1/2} = 1 - \frac{\varepsilon}{2}\,U_0 + \varepsilon^2\Big(\frac38 U_0^2 - \frac12 U_0 - \frac12 U_1\Big) + O(\varepsilon^3),
\qquad U_0 = \frac{z+1}{z},
```

and for CSS, with U₁ from §2.2, the ε² bracket is (3/8)U₀² − U₀/2 − ((z+1)/z²)[(1/κ² − 1) ln z + 1 − 2/(κ²z)]. Each
coefficient is finite for z > 0, so **θ₊ > 0 at every point of the branch at every finite order**, and near z → 0
the k-th term grows like ε^k/(κ̃^{2k−2} z^{2k−1}) (U_k ~ z^{−2k−1}). The truncated
θ₊ ∝ 1 − εU₀/2 vanishes at z ≈ ε/2, inside the region z ≲ n^{−1/2} where the series has already broken down: that zero
is an artefact. At leading order the formal zero sits at z = 0 (U₀ → ∞), which is what makes β²x² = 1 look like a
horizon.

### 3.2 Where the expansion breaks down

At z ~ n^{−1/2}/κ̃ (§2.5): (n−1)(b−1) ~ n^{1/2}, so 1 − 1/b ~ n^{−1/2}; c ~ n^{−1/4}; y ~ n^{1/4}; p + m ~ 1. The time
derivatives along the front enter at the same order as the radial ones (through β + β′), and the O(ε) term
x²∂_x(p+m) of (B) is no longer small. Gravity is still weak there: the region where the series fails is far from trapped.

### 3.3 The layer (`layer.py`)

With y = n^{1/4}Y, c = n^{−1/4}C, p + m = S, ln x − ln(1/β) = n^{−1/2}η and x̃₀ = (β+β′)/β² (for CSS 1/β; in general
x̃₀ = −dr_s/dt, the velocity of the singular sphere), the leading order of (A), (B), E1, E2 in the layer is
(*derived* by hand as a formal leading-order reduction; its scalings are confirmed at finite n in §3.4)

```math
\tilde x_0\,Y_\eta = C\,S,\qquad \tilde x_0\,S_\eta - C\,Y_\eta = \tfrac12 C\,Y^3,\qquad C_\eta = \tfrac12 C\,Y^2,\qquad (n-1)(b-1) = \tfrac12 y^2 ,
```

quasi-static in τ (τ-derivatives at fixed η are n^{−1/2} smaller). These are Rozali and Way's D = ∞ equations
(README, Proposition 5(i): Y_t = cS, S_t = c(Y_s + Y³/4), c_s = cY²/4) for a front moving in s = r²/(2n), without the
linear term (y ≫ 1). Since C_η Y = CY³/2, the middle equation is (x̃₀S − CY)_η = 0: **x̃₀ S − C Y = −√2 f**
exactly (the constant from the outer region). Scaling η = (|x̃₀|/f)θ, Y = (f/|x̃₀|)^{1/2}Y₁, C = √(f|x̃₀|)C₁ removes x̃₀
and f; the universal problem is

```math
Y_1' = C_1\,(C_1Y_1 - \sqrt2),\qquad C_1' = \tfrac12 C_1Y_1^2,\qquad
Y_1 = \theta^{-1/2}\big(1 - \tfrac{1}{2\theta^2}\big),\ C_1 = \sqrt{2\theta}\big(1 + \tfrac{1}{4\theta^2}\big)\ (\theta\to+\infty),
```

where the mode the outer region forbids grows outward (like e^{θ²/2}), so inward integration is stable. Result
(`logs/layer.log`; the full third-order system conserves S₁ − C₁Y₁ = −√2 to 4×10⁻¹¹; starts at θ₀ = 40 and 80
agree to 10⁻¹⁰):

- **no blow-up**: Y₁ rises monotonically from 0 to **Y₁(−∞) = 1.803807327** and C₁ → 0 like e^{1.626860 θ}
  (= e^{Y₁(−∞)²θ/2}); the would-be singularity at θ = 0 (β²x² = 1) is crossed smoothly (Y₁ = 1.047, C₁ = 1.070 there);
- the front, defined by S₁ = −√2/2, is at θ_f = −1.002877, i.e. **z_f = −2.0058/(κ̃√n)**: just inside β²x² = 1;
- behind it, (n−1)(b−1) → ½y² = **1.62686 κ̃ √n**, so 1 − 1/b → 1.627 κ̃/√n, and the lapse
  c ∝ exp(0.81343 κ̃√n z) (z < 0): it collapses exponentially in √n. With c ≈ 0 every τ-derivative at fixed r drops out
  of (A), (B): p, m freeze as functions of r. This is the behaviour of polar slicing near a forming black hole, here
  produced at small compactness because u = (n−1)(b−1) ~ √n is large although b − 1 ~ n^{−1/2} is small.

`layer.py --mutate=sign` (the cubic's sign flipped) breaks the first integral (0.88) and FAILS.

*Matching* (`match_check.py`, *checked symbolically*). The layer's own large-θ expansion,
Y₁ = θ^{−1/2}(1 − 1/(2θ²) + 13/(12θ⁴) + …), C₁ = √(2θ)(1 + 1/(4θ²) − 13/(48θ⁴) + …), derived from the layer ODE
alone, reproduces exactly the most singular terms of the outer CSS series through NNLO: [z⁻²]Y₁/β = −2/κ²,
[z⁻⁴]Y₂/β = 52/(3κ⁴), [z⁻²]C₁/f = 1/κ², [z⁻⁴]C₂/f = −13/(3κ⁴), with no ln z at those powers. The two sides share no
code. For general β(τ), f(τ) the same identity holds at NLO with κ → κ̃ = fβ²/(β+β′): [z⁻²]Y₁/β = −2(β+β′)²/(f²β⁴),
[z⁻²]C₁/f = (β+β′)²/(f²β⁴) (`match_check.py logs/expand_K1_free.pkl`), a symbolic check of the quasi-static layer
reduction, since τ-derivatives in the layer are n^{−1/2} smaller and reach the outer series only at z^{−(2k−1)}.
`--mutate=layer` (C₁′ = C₁Y₁²/4) FAILS.

### 3.4 Finite-n check: the exact CSS equations across the front (`css_bvp.py`)

Integrating the exact equations inward from the outer region fails at once (`css_inward.py`, kept as a record): the b
equation has a mode growing like x^{−(n−1)} inward, and p + m one growing outward. So `css_bvp.py` solves a
boundary-value problem on [θ = −7, z = 0.6]: y, c, p + m from the NLO series at the outer end, 1/b slaved at the
inner end; no large-n approximation. Against the layer (β = 1, f = 0.8, κ_f = 0.8; `logs/css_bvp_b1_f0.8.log`):

| n | max\|y/y_layer − 1\| for \|θ\| < 5 | y/y_layer at θ = −6 | (n−1)(b−1)/(κ√n) at θ = −6 | 1 − 1/b at θ = −6 | min 1/b on the interval | c at θ = −6 |
|---|---|---|---|---|---|---|
| 100 | 0.121 | 1.0054 | 1.7169 | 0.1218 | 0.8782 | 6.5×10⁻⁵ |
| 400 | 0.128 | 1.0128 | 1.7082 | 0.0641 | 0.9359 | 3.3×10⁻⁵ |
| 1600 | 0.082 | 1.0071 | 1.6695 | 0.0323 | 0.9677 | 2.9×10⁻⁵ |
| 6400 | 0.040 | 1.0036 | 1.6481 | 0.0162 | 0.9838 | 2.3×10⁻⁵ |

The deviation from the layer falls (fitted n^{−0.27}; the last step halves it), (n−1)(b−1)/(κ√n) approaches the
layer's 1.627, and 1 − 1/b falls like n^{−1/2}. With κ_f = 1.5 (β = 1.5, f = 1; the self-similarity horizon
x_lc = 0.894 lies outside the interval; `logs/css_bvp_b1.5_f1.0.log`) the deviation is 0.135, 0.092, 0.047, 0.024
(fitted n^{−0.42}), (n−1)(b−1)/(κ√n) = 1.819, 1.723, 1.675, 1.651 and 1 − 1/b = 0.216, 0.115, 0.059, 0.030 at
n = 100–6400; min 1/b ≥ 0.78. The lapse ratio c/c_layer agrees less well (16% at n = 6400, in the tail where c is
already 10⁻⁵ of its outer value). `--mutate=cubic` (a wrong layer) FAILS: at n = 6400 the solve converges to the same
solution and misses the wrong layer by a factor 8.6. This is *numerical evidence*
for the CSS case; the general-τ statements rest on the quasi-static layer reduction, which is matched symbolically to
the outer series (§3.3) but not checked at finite n.

### 3.5 Verdict

- **β²x² = 1 is not an apparent horizon.** On every polar slice, at every order and through the layer, θ₊ = n/(r√b) > 0
  with 1 − 1/b ≤ 1.63 κ̃/√n + o(n^{−1/2}). The leading order's singular point is a **lapse-collapse front**: the polar
  time stops advancing behind it, which is why the leading-order lapse c = f√(β²x²−1) vanishes there.
- **How it moves at higher orders.** It is a layer of width z ~ n^{−1/2}/κ̃ centred at z_f = −2.006/(κ̃√n), not a
  sharp surface. The corrections from the outer series are shifts of β at O(1/n), smaller than the width.
- **Causal character.** The front x = X(τ), X ≈ 1/β, has normal of norm −(e^{2τ}/α²)[(X − X′)² − c²] with
  X − X′ = x̃₀ and c = O(n^{−1/4}) at the front: it is **spacelike** whenever β + β′ ≠ 0 (always for CSS). At leading
  order the surface β²x² = 1 itself, where c = 0, is spacelike by the same formula, null only if β + β′ = 0.
- **When could a horizon form?** The compactness behind the front, 1.63 κ̃/√n, grows as the front slows
  (κ̃ = r_s/|dr_s/dT| → ∞ with dT = K dt) and the layer analysis fails when κ̃ ~ √n. So a horizon near β²x² = 1 needs a
  front nearly at rest in r (*conjecture*; the next layer, u ~ n, is not analysed). Rozali and Way's exact oscillaton is
  exactly that case: c² = 1 − δ₀ = (1 − ε²e^{−ω²r²})/(1 − ε²e^{−ω²r²}sin²(ωτ−τ₀)) vanishes at r_div for every τ while
  ∂_τφ stays finite (`rw_oscillaton.py`, their Eq. (10) checked to 10⁻¹⁹⁰), so their divergence is a static front
  where Y = 2φ_τ/c diverges.
- **Which side is frozen, and what the polar chart cannot see.** The frozen region is always on the side r < r_s(t).
  For CSS in the past patch the front r_s = (−t)/β shrinks, so at fixed r the frozen region lies to the *past* of the
  front's passage: the CSS member is the time reverse of an expanding front. The equations are time-symmetric (in
  polar slicing θ₊ = −θ₋, so trapped and anti-trapped cannot be told apart). The time-reversed member (t > 0, x = r/t)
  and the shifted solution of §4.3 have an expanding front with the frozen region to its future, the configuration of
  black-hole formation. There the polar slices cover only a finite proper time behind the front, so any horizon that
  forms later lies beyond the chart. In the CSS ODE the interior also carries a free mode
  δ(1/b) ∝ x^{−(n−1+P)} (a self-similar central mass; the inner boundary condition of `css_bvp.py` sets it to zero).
  A positive central mass would give 1/b = 0 inside the frozen region, at a radius set by that mass and not by
  β²x² = 1.

## 4. Relation to known solutions

### 4.1 Literature record

Read for this note (October 2026). "Read" means the text named was fetched and read, not recalled. Bibliographic data
were checked against arXiv abstract pages unless marked.

| Source | Read | Contains | Does not contain | Reference check |
|---|---|---|---|---|
| C. Ecker, F. Ecker, D. Grumiller, PRL 136, 191401 (2026), [arXiv:2601.14358](https://arxiv.org/abs/2601.14358) | HTML v1: Eqs. (1)–(22), Summary and Outlook, supplement (S1)–(S29) | the supercritical branch by β → iβ, f₀ → if₀ (supplement); "excludes β having zeros", "contain a second arbitrary function", "resemble the supercritical solutions above the threshold of black hole formation", "A detailed analysis of this branch would be interesting" (Summary); "The singular point β²x² = 1 is sometimes used as a proxy for the apparent horizon, see, e.g., [14]" | any NLO of the branch; any computation of θ±; explicit Ψ_LO, Ω_LO of the branch (only the substitution rule, see §1) | in README (verified there) |
| M. Rozali, B. Way, JHEP 11 (2018) 106, [arXiv:1807.10283](https://arxiv.org/abs/1807.10283) | ar5iv, Secs. 1–2.3 (again; see LIT_REVIEW.md) | "We conjecture that the divergence in those solutions is a signal of horizon formation at large but finite D"; "strictly speaking, horizon formation does not occur at infinite D"; the oscillaton Eq. (10); r_div and γ = 1/2 from it; A₀ "diverges" | the words "apparent horizon"; a boundary layer; time-independent or quasi-static solutions | verified (REF_CHECK.md) |
| P. R. Brady, PRD 51, 4168 (1995), [gr-qc/9409035](https://arxiv.org/abs/gr-qc/9409035) | arXiv PDF | homothetic field ψ = h̄(r/\|u\|) − κ ln\|u\| (Eq. 2.8, "the most general ... consistent with the homothetic symmetry"); classes split at 4πκ² = 1; exact stationary solution, Eqs. (2.24)/(3.1), y = ḡ/g = 1/2, γ± = ±1/√(4π); class I (4πκ² > 1) approaches it; class II contains black holes ("signalled by y → 0") and naked singularities; κ = 0 is Roberts | higher D; a mass exponent; the causal character of the horizons | DOI 10.1103/PhysRevD.51.4168 (arXiv abstract page) |
| P. R. Brady, CQG 11, 1255 (1994), [gr-qc/9402023](https://arxiv.org/abs/gr-qc/9402023) (arXiv title "Does scalar field collapse produce 'zero mass' black holes?") | arXiv PDF | Roberts-type family (parameter α), apparent horizon at u = 2αv, "spacelike everywhere"; M ≃ \|α − α_crit\|^{1/2} | κ ≠ 0; higher D | journal and published title (*Analytic example of critical behaviour in scalar field collapse*) from Frolov 1999's reference list and an INSPIRE search title; the arXiv page shows no journal-ref: *(partly unverified)* |
| Y. Oshiro, K. Nakamura, A. Tomimatsu, Prog. Theor. Phys. 91, 1265 (1994), [gr-qc/9402017](https://arxiv.org/abs/gr-qc/9402017) | ar5iv | Roberts family (parameter p), "apparent horizon exists only in the strong-field implosion of p > 1", at u = (1−p²)v; M_h ≃ ((p−1)/8)^{1/2}v | perturbations; higher D | journal-ref and DOI 10.1143/PTP.91.1265 (arXiv abstract page) |
| A. V. Frolov, PRD 56, 6433 (1997), [gr-qc/9704040](https://arxiv.org/abs/gr-qc/9704040) | arXiv PDF (results) | perturbations of the critical Roberts solution, solved exactly; spectrum continuous, 1/2 < Re k < 1; mass exponent 1/κ = 1, "different from the exponent β=1/2 found in [10–12]" | the horizon's causal character; higher D | DOI 10.1103/PhysRevD.56.6433 (arXiv abstract page) |
| A. V. Frolov, CQG 16, 407 (1999), [gr-qc/9806112](https://arxiv.org/abs/gr-qc/9806112) | ar5iv and arXiv PDF | D-dimensional Roberts family, φ = φ(z), z = −v/u; ẏ² = y² − 2 + c₁y^{−2(n−3)} (n = D); apparent horizon at (∇r)² = 0, y_AH² = (c₁/2)^{1/(n−3)}; critical c₁* | a mass exponent; the large-n limit | journal from the author's CV page (sfu.ca/~frolov); the arXiv page shows none *(partly unverified)* |
| H.-C. Kim, S.-H. Moon, J. H. Yee, JHEP 02 (2002) 046, [gr-qc/0108071](https://arxiv.org/abs/gr-qc/0108071) | ar5iv | D-dimensional Roberts family; "Higher dimensions seem to favor formation of black hole"; cut-and-paste γ = (n−3)/√(2(n−2)), attributed to their Ref. [21] | κ ≠ 0; large-n analysis | DOI 10.1088/1126-6708/2002/02/046 (arXiv abstract page) |
| C. R. Clark, G. L. Pimentel, JHEP 11 (2025) 017, [arXiv:2504.10590](https://arxiv.org/abs/2504.10590) | arXiv PDF (to the region equations) | CSS with k̄ = 0: "we make the further assumption that, within CSS solutions, the scalar field profile is scale-invariant"; the large-D Roberts family, (y′)² = y² − 2y + y²(2q/y)^{1/ε}; supercritical: apparent horizon in their region C; γ = (D−3)/√(2(D−2)) (CSS-preserving), 2(D−3)/(D−2) (CSS-breaking) | κ ≠ 0 | verified (LIT_REVIEW.md) |
| P. R. Brady, M. W. Choptuik, C. Gundlach, D. W. Neilsen, CQG 19 (2002), DOI 10.1088/0264-9381/19/24/306, [gr-qc/0207096](https://arxiv.org/abs/gr-qc/0207096) | arXiv PDF | stiff fluid ↔ scalar field with timelike gradient; CSS ψ = h̄₀(x) + κτ; the 4πκ² ≈ 0.577 solution with an apparent horizon at x₂ ≈ 1.2 ("not actually on the black hole threshold" for the scalar field); Christodoulou's 4πκ² = 1/3 FRW solution | higher D | DOI from the arXiv page; page number not verified *(partly unverified)* |
| Koushiki, R. Goswami, P. S. Joshi, [arXiv:2512.20998](https://arxiv.org/abs/2512.20998) | arXiv PDF (summary) | 4D, ansatz-free 1+1+2 analysis of apparent-horizon slopes | self-similar families; higher D | not needed |

Not read (cited by the sources above): M. D. Roberts, Gen. Rel. Grav. 21, 907 (1989) (as listed in Frolov 1999's
references); D. Christodoulou, "Examples of naked singularity formation in the gravitational collapse of a scalar
field", Annals of Mathematics (1994) (cited by Brady 1995 and BCGN; volume and pages *(unverified)*); Soda and Hirata
(the CSS-preserving exponent quoted by Clark and Pimentel);
A. V. Frolov, PRD 59, 104011 (1999), gr-qc/9811001 (non-spherical perturbations of Roberts; from the author's CV).
No D-dimensional version of Brady's κ ≠ 0 family was found in one targeted search.

### 4.2 Is the branch a large-D limit of a known family?

- **Roberts and its D-dimensional versions: no** (*proved* by hand). For the homothety ξ = t∂_t + r∂_r, every member
  of the Roberts family (Brady 1994; Oshiro et al.; Frolov 1999; Kim et al.; Clark–Pimentel) has ξ(φ) = 0. On the
  branch ψ = √n fβ τ + O(n^{−1/2}) (from (∂_τ + x∂_x)ψ = (−t)α Π̂ and §1), so ξ(ψ)/√n = −κ_f ≠ 0 in the CSS case and
  −fβ(τ) pointwise in general. A limit of κ = 0 solutions has ξ(ψ)/√n → 0. This is also why EEG find their CSS
  solutions different from Clark and Pimentel's.
- **Brady's κ ≠ 0 family: the same structure at D = 4.** The stationary solution of §1 at n = 2 has (∇r)² = 1/2 (b = 2)
  and r∂_rψ along outgoing null rays equal to ±1/√(4π) in Brady's normalization (ψ_here = √(8π)ψ_Brady at D = 4),
  independent of κ: Brady's y = 1/2 and γ± = ±1/√(4π) (*checked* by hand; the full coordinate map to his Bondi
  variables was not done). The homothetic slopes match as √n κ_c = √(8π)κ_Brady, so his class boundary 4πκ² = 1 is
  κ_c = 1. In the large-D family the same value separates the cases with and without a self-similarity horizon:
  regular members have one iff κ_f < 1 (EEG's normalization f(1) = 1 gives κ_f = β/√(1+β²) < 1), supercritical
  members iff κ_f > 1. Regular members with κ_f > 1 approach the stationary solution with no horizon, like Brady's
  class I. Whether D-dimensional κ ≠ 0 families exist and tend to the large-D one was not checked (none was found).
- **Stiff fluid.** At leading order ∇ψ is timelike (|Π̂| ≫ |Φ̂|), so both branches are, at leading order, stiff-fluid
  configurations in the sense of Brady, Choptuik, Gundlach and Neilsen. Their D = 4 CSS solution at 4πκ² ≈ 0.577,
  with an apparent horizon at x₂ ≈ 1.2, belongs to Brady's class II; its large-D counterpart was not identified.

### 4.3 The late stage of supercritical collapse and the λ = 2 mode

(`shift_check.py`, *checked symbolically* at leading order.) In Proposition 5's heat variables the family of §1 is
Y = √2/√(s + s_c(t)), s_c = ρ/(2n), and solves Y_s = −Y³/4, c_s = cY²/4, the inner equations without t-derivatives.
The s-translation s → s + σ₀, an exact symmetry of the D = ∞ inner equations, maps s_c → s_c + σ₀, i.e.

```math
C(\tau) \;\to\; C(\tau) + 2n\,\sigma_0\,e^{2\tau},
```

and its generator equals Proposition 5's δy = e^{2τ}(xy_x − y)/x² (in x = √n ξ). So at leading order **the
supercritical branch is the regular branch moved a finite distance along the λ = 2 direction**. For a regular CSS
member (C₀ > 0) and σ₀ < 0:

- C(τ) changes sign at e^{2τ*} = C₀/(2n|σ₀|). At that instant y ≡ √2: the stationary solution.
- After it, the front r_s(t)² = 2n|σ₀| − C₀t² is born at r = 0 and expands. It decelerates and stops at
  **r_BH = √(2n|σ₀|)** as t → 0⁻.
- κ̃ = k₀(2n|σ₀|e^{2τ}/C₀ − 1) grows like e^{2τ}.

By §3.5 the compactness behind the front, 1.63 κ̃/√n, therefore grows as the front stops, and trapping becomes possible
near r_BH once κ̃ ~ √n (*conjecture*). This is the departure from a regular solution along the unstable mode, followed
nonlinearly, in EEG's leading-order regime. For the DSS critical solution the shift is still exact (it is a symmetry of
the inner equations), but whether the shifted solution develops this quasi-static front is not shown here
(*conjecture*).

**Does anything bear on γ?** Only qualitatively. The front stops at r_BH² = 2n|σ₀| ∝ (p − p*), which is the mechanism of
Proposition 5 (γ∞ = 1/2) made explicit in a nonlinear family; it adds no new number. The O(1/n) part of the branch's
NLO breaks the s-shift and could in principle feed λ₁ (= 3.03 from the data), but extracting it needs the matching to
the DSS inner solution, which is not done. (For comparison, the Roberts family gives 1/2 by cut-and-paste in D = 4,
Frolov's perturbations 1 instead, and Kim et al. quote (n−3)/√(2(n−2)), which grows with n: none of them bears on the
λ = 2 mode of the DSS solution.)

## 5. Would a finite-D computation be worth doing?

- **CSS members: cheap and worth doing as a check.** The CSS equations are ODEs in x. `css_bvp.py` already solves them
  across the front (κ_f = 0.8, and 1.5 inside the self-similarity horizon). Extending it is a modest change:
  - from the frozen interior, with the central-mass mode as a parameter;
  - through the front and through the self-similarity horizon for κ_f > 1, with analyticity there as the extra
    condition (one real condition, as for the regular branch's light cone);
  - out to the stationary asymptotics.

  It would show:
  1. whether analyticity at x_lc restricts κ_f at finite n (expected: a condition invisible to the 1/n series, as
     for the regular branch's light cone; the local exponents at x_lc were not computed);
  2. how the front and the frozen interior look at D = 5–10;
  3. whether the family continues to D = 4 into Brady's class I/II solutions.

  The project's DSS solvers are not needed for this.
- **DSS-type members (β periodic, positive) with the repository's solvers: not worth it now.** `hp/` and
  `solver/dss.py` assume odd harmonics in p, m and a regular centre plus an analytic light cone. The supercritical
  members have ψ with a linear part (even and odd harmonics), no regular centre and a front with exponential lapse
  collapse, so the boundary blocks and the harmonic symmetry would have to be replaced. The period is free at every
  order (§2.7), so such a computation would only probe beyond-all-orders effects, as for the regular branch.
- **The physically interesting case, the evolution along λ = 2 into a horizon, needs a different tool.** It is not
  self-similar: §4.3 is a time-dependent solution. Polar slicing freezes behind the front (§3.3), so it needs an
  evolution in a gauge that penetrates horizons, e.g. double-null or Bondi coordinates as in Brady's and Frolov's
  papers. At moderate D (10–20) that would test whether the horizon forms where §3.5 and §4.3 predict, at
  r ≈ √(2n|σ₀|) as the front stops, and would measure the compactness behind a moving front against 1.63 κ̃/√n.

## 6. Status summary

| statement | status |
|---|---|
| EEG's Eqs. (3a)–(3d) = E1–E4 under the map of §1 | checked symbolically (`eqmap.py`) |
| the leading order solves the exact equations to O(1/n), any β(τ), f(τ) | checked symbolically (`lo_check.py`) |
| EEG's rule gives the wrong sign of Ψ_LO for f₀ > 0 | checked symbolically (`lo_check.py --mutate=psisign`) |
| f is gauge outside CSS; κ_f physical in CSS | proved; checked symbolically (`shift_check.py`, item 2) |
| stationary solution exact; its exponents | checked symbolically (`y1_modes.py`) |
| Proposition S: all orders exist, nothing forced, A_k, B_k = shifts of β, f | proved; checked symbolically through NNLO (CSS) and NLO (general) |
| large-x logarithms = expansion of the stationary exponent | checked symbolically through NNLO (CSS) |
| series residuals fall like n^{−(K+1)} | numerical evidence (`formal_check.py`) |
| breakdown at z ~ n^{−1/2}/κ̃ | checked symbolically (orders' singular powers, `expand.py`) |
| layer equations and first integral | derived (formal leading order); first integral proved |
| layer matches the outer series' most singular terms through NNLO (CSS) and at NLO for general β(τ), f(τ) | checked symbolically (`match_check.py`) |
| layer: no blow-up, Y₁(−∞) = 1.803807327, front at θ_f = −1.003 | numerical evidence (`layer.py`) |
| no apparent horizon across the front, CSS, finite n | numerical evidence (`css_bvp.py`, n ≤ 6400) |
| no apparent horizon for general β(τ) with β + β′ ≠ 0 | conjecture (quasi-static layer: matched symbolically to the outer series, not integrated at finite n) |
| front spacelike for β + β′ ≠ 0 | proved from the layer scalings (formal) |
| trapping needs a nearly static front; it forms near r_BH as the shifted front stops | conjecture |
| Rozali–Way's divergence is a static lapse-zero front | checked symbolically (`rw_oscillaton.py`) |
| not a limit of the Roberts family (any D) | proved |
| the stationary solution at D = 4 is Brady's | checked by hand (two invariants); coordinate map not done |
| the LO family is closed under the s-shift, generator = Prop. 5's λ = 2 mode | checked symbolically (`shift_check.py`) |

## 7. Reproduce

From `solver/supercrit/` (2 cloud cores shared with other jobs, `nice -n 10`; times under that load):

```
python3 eqmap.py                                   # PASS                                   ~1 min
python3 lo_check.py                                # PASS; residual x n constant             ~2 min
python3 expand.py 1                                # NLO, A_1/B_1 free; writes logs/expand_K1_free.pkl   ~2 min
python3 expand.py 2 --norm=zero --css              # NNLO, CSS; writes logs/expand_K2_zero_css.pkl       ~5 min
python3 expand.py 2 --norm=zero                    # NNLO, general beta, f (see section 9)
python3 nlo_factor.py                              # the NLO coefficients of section 2.2, factored          ~1 min
python3 cmp_series.py logs/expand_K2_zero_css_sympyint.pkl logs/expand_K2_zero_css.pkl   # identical series
python3 cmp_series.py logs/expand_K1_free_sympyint.pkl logs/expand_K1_free.pkl           # identical series
python3 css_bvp.py 1.5 1.0                         # kappa_f = 1.5, section 3.4              ~10 min
python3 formal_check.py logs/expand_K1_free.pkl tau        # exponents about -2                 ~10 min
python3 formal_check.py logs/expand_K2_zero_css.pkl css    # exponents about -3                 ~5 min
python3 y1_modes.py                                # lambda_1 = -1.0779457959, lambda_2 = -1.1619671389   ~2 min
python3 layer.py                                   # Y1(-inf) = 1.803807327, theta_f = -1.002877           <1 min
python3 match_check.py                             # layer vs outer singular terms through NNLO: PASS  ~1 min
python3 match_check.py logs/expand_K1_free.pkl     # the same for general beta(tau), f(tau) at NLO: PASS
python3 css_bvp.py 1 0.8                           # table of section 3.4                    ~10 min
python3 shift_check.py                             # PASS                                    ~1 min
python3 rw_oscillaton.py                           # PASS                                    ~1 min
```

`css_inward.py` (the failed inward integration) is kept to document why the boundary-value formulation is needed.
Every validation script exits non-zero when its check fails.

## 8. Mutation table

Each check was broken once on purpose and seen to fail (logs `logs/*_mut_*.log`):

| check | mutation | result |
|---|---|---|
| `eqmap.py` | `--mutate=eps` (ε_E = 1/n instead of 1/(n+1)) | FAIL (exit 1) |
| `eqmap.py` | `--mutate=psi` (p + m = √2 xΨ) | FAIL |
| `lo_check.py` | `--mutate=regular` (lapse of the regular branch) | FAIL: E1 at O(1) |
| `lo_check.py` | `--mutate=psisign` (p + m with the other sign, = EEG's rule) | FAIL: E3, E4 at O(1) |
| `formal_check.py` | `--mutate=drop` (Y₂, C₂ dropped) | FAIL: exponents −2.0 instead of −3 |
| `y1_modes.py` | `--mutate=y1` (Y₁ × 1.01) | FAIL: λ₁ −1.08873 vs −1.07795 |
| `layer.py` | `--mutate=sign` (cubic's sign) | FAIL: first integral broken (0.88) |
| `css_bvp.py` | `--mutate=cubic` (wrong layer: u = y²/4) | FAIL |
| `match_check.py` | `--mutate=layer` (C₁′ = C₁Y₁²/4) | FAIL: [z⁻²]C₁ 2/5 vs 1 |
| `shift_check.py` | `--mutate=rate` (shift ∝ e^{τ} instead of e^{2τ}) | FAIL: generator differs |
| `rw_oscillaton.py` | `--mutate=sign` (sign in δ₀'s denominator) | FAIL: residual 10¹ |

`expand.py` asserts at every order that (B) and E1 hold exactly and that every antiderivative differentiates back to
its integrand; its two integrators give identical series (CSS, NNLO).

## 9. Notes for the coordinator

- **Discrepancy with EEG, minor.** Their substitution rule, applied to Ψ_LO, gives the wrong sign for f₀ > 0 (§1).
- **Discrepancy with EEG's and Rozali–Way's reading of β²x² = 1, substantive.** EEG call the singular point a proxy
  for the apparent horizon, citing Rozali and Way's conjecture. In this branch it is a lapse-collapse front with
  compactness O(n^{−1/2}), not a horizon (§3), except for a front at rest. Rozali and Way's own exact divergence is
  static, so the two findings are compatible; their conjecture is not tested by this note.
- **General-τ NNLO.** The run `expand.py 2 --norm=zero` (exact integrator) was still in progress when this note was
  first written; its outcome is recorded at the end of this section.
- **Not done.**
  - the next layer (u ~ n) for nearly static fronts;
  - a numerical check of the layer for non-CSS β(τ);
  - the matching of the s-shifted DSS solution;
  - the full coordinate map to Brady's stationary solution;
  - D-dimensional κ ≠ 0 families in the literature;
  - LIT_REVIEW.md entries for the sources of §4.1 (needed before any novelty sentence in the README or papers).

## 10. Proposed README text

For the end of "Relation to Ecker, Ecker & Grumiller" (after the paragraph on their conjecture Δ → 0):

> **Their supercritical branch.** EEG also note a second branch, $`f_{LO} = f_0\sqrt{\beta^2x^2-1}`$ on β²x² > 1, whose
> singular point β²x² = 1 they treat as a proxy for the apparent horizon, citing Rozali and Way.[^6] In our variables the
> two branches are one leading-order family, y = √2 x/√(x² + C(τ)), with C > 0 regular and C < 0 supercritical, and
> the r²-shift of Proposition 5 moves C by 2nσ₀e^{2τ}: the supercritical branch is the regular one pushed along the
> λ = 2 mode. Its 1/n expansion exists to all orders with nothing forced (as in Proposition 1). Near β²x² = 1 it is a
> series in 1/(n(β²x²−1)²) and breaks down at β²x² − 1 ~ n^{−1/2}, where 2m/r^{n−1} is only of order n^{−1/2}. There
> the solution crosses β²x² = 1 smoothly into a region where the polar lapse collapses exponentially. Behind this
> spacelike front, 2m/r^{n−1} stays near 1.63 κ̃ n^{−1/2}, with κ̃ = fβ²/|β + β′| the inverse speed of the front.
> So, for a moving front, β²x² = 1 is not an apparent horizon. Trapping needs the front to come to rest, as it does
> when the shifted solution's front stops at r² = 2n|σ₀|. The exact CSS equations at n = 100–6400 bear this out.
> Details, checks and the comparison with the Roberts and Brady self-similar families:
> [notes/SUPERCRITICAL.md](notes/SUPERCRITICAL.md).

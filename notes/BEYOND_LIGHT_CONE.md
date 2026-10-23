# A solver beyond the light cone: feasibility study

Branch `wip/beyond-lightcone`. Written by Claude (Anthropic) under Nathan Moinvaziri's direction; not reviewed. Scripts,
logs and data are in [`solver/beyond_lc/`](../solver/beyond_lc/). Status words follow AGENTS.md.

## The answer

**No, not as proposed, because its premise does not hold.** The proposal assumes that inside the past light cone the
selecting effect has the size of the inner fast mode $`e^{−n/(2x²)}`$, about $`e^{−n/2}`$ near x = 1. Then a solver on
0 ≤ x ≤ 1 would have to resolve an exponentially small effect, and a solver whose conditions sit at x ~ √n would not.

The exact fast mode inside the light cone is $`e^{-n\Phi(x)} = x^{-n}(1 - x^2)^{n/2}\cdots`$, with Φ = ln x − ½ ln(1 − x²). Its
modulus is exactly 1 on the curve Re x² = ½ (§2.6). That curve crosses the real axis at x = 1/√2 = 0.7071 and runs out
to the inner region along arg x → π/4. The inner form $`e^{−n/(2x²)}`$ holds only for |x| ≫ $`n^{1/4}`$; it is not the size of
the effect inside the light cone.

The `hp/` solver matches its centre-side integration to the horizon series at its last node, x_m = 0.7, on that
curve: $`e^{−nΦ(0.7)}`$ = $`e^{+0.020n}`$. So the effect the solver must resolve is already of order one where it imposes its
condition. There is no $`e^{n/2}`$ to remove. Three independent checks support this:

1. **The data** (`sens_fit.py`). From D = 35 on the scan froze the harmonics above 13. In that regime, D = 40–61,
   max|∂Δ/∂F| is a power law $`n^{21.7}`$: the residual in ln is 0.023 against 0.167 for $`e^{n/2} n^p`$, and a free fit
   a·n + p·ln n gives a = 0.004. The local growth falls from 1.74 to 1.40 per unit of D, which is $`n^p`$ with p = 19.6–22.1
   in every 3-unit bin.
2. **The full equations** (`hp_node_test.py`). The converged double-double solutions at D = 40, 50 and 60 were
   re-matched with the last node at x_m = 0.6, 0.65, 0.75 and 0.8. max|∂Δ/∂F| changes by the fast-mode factor
   $`e^{n[Φ(x_m) − Φ(0.7)]}`$, over 13 orders of magnitude: at D = 60 the measured ln ratios are −14.4, −7.5, +8.2 and +17.2,
   against −15.5, −7.9, +8.5 and +17.9 predicted (table in §5). Inside 0.7 the residual floor rises as the sensitivity
   falls. So x_m = 0.7 already is "where the selecting effect is O(1)".
3. **The toy** (`toy.py`; criteria registered before running, §0). The toy is a linear model with an exactly known
   selected value.
   - The standard layout matched at x_m = 0.7 has a condition number flat in n: 259, 384, 293 and 80 at n = 20, 60,
     100 and 200.
   - The complex-path candidate is flat too, at 0.28.
   - Extending the domain along the real axis beyond the light cone costs $`e^{0.55n}`$.
   - Matching nearer the light cone costs $`e^{0.73n}`$ at x_m = 0.9.

   The registered criterion is **not met**. Its second half fails: the standard layout does not grow like $`e^{n/2}`$, so
   the candidate has nothing exponential to remove.

**What does limit `hp/`.** The sensitivity grows as a power of n, from the high τ-harmonics.
- In the linear free-wave problem, the light-cone obstruction of harmonic k as seen at x_m = 0.7 weakens like $`n^{−11.3}`$,
  $`n^{−16.9}`$, $`n^{−21.7}`$ and $`n^{−25.6}`$ for k = 7, 9, 11 and 13, between n = 39 and 59 (`lin_ladder.py`).
- That is close to the measured weak directions of `hp/` ($`n^{−10.5}`$, $`n^{−14}`$, $`n^{−17.5}`$, $`n^{−21}`$) and to the
  sensitivity's $`n^{21.7}`$.
- It is the regime (kω)² ≳ n, before each harmonic reaches its large-n limit. For k = 13 it lasts until
  n ~ (13ω)² ≈ 2500.

Separately, the residual floor at x_m = 0.7 rises 1.46-fold per unit of D. The node test shows that this does not come
from the fast mode: at x_m ≥ 0.7 the floor hardly depends on x_m. Its source is not identified here.

**Why the complex path does not pay for the full problem** (§7). Moving the condition to the Stokes line beyond the
light cone makes each harmonic's selection signal independent of n. But it replaces the n-dependent ladder with the
half-monodromy weights, about $`e^{πkω}`$ per harmonic.
- In the free-wave problem the jump across the Stokes line is 1.4×10⁴ times the solution for k = 1 and 10⁶⁵ times for
  k = 13, at every n.
- A nonlinear solver on such a path amplifies rounding in harmonic k by that factor. It needs about πω/ln 10 ≈ 5.2 more
  digits per harmonic kept: some 70–85 digits for harmonics up to 13, at every D.
- Its Newton iterates would have to satisfy the harmonic-k conditions to about $`e^{−πkω}`$ before the continuation is even
  close to linear.

The other two candidates fare no better:
- **Formulation (a) on the real axis** has no exact condition at large x: x is a time coordinate there (§2.2). It also
  amplifies the fast direction outward from the light cone by at least $`2^{n/2}`$.
- **Formulation (b)** is a change of variable. It inherits the conditioning of whatever path and condition it is used
  with.

**What it would take instead.** Not a new formulation. The existing layout's obstacles are a power law in n (the
sensitivity) and a floor that grows 1.46-fold per unit of D. The extrapolations below are estimates, not measurements
(§6):
- The sensitivity would reach about 10¹⁹ at D = 100 and 10²³ at D = 200. This uses the linear ladder, with 13 as the
  top kept harmonic.
- The floor would reach about 10¹⁶ and 10³² times the arithmetic's ε.
- In quad-double (`hp/src/qd.rs`, about 7 times the double-double cost per residual), the worst-case bound
  floor × sensitivity would be about 10⁻²⁹ at D = 100 and 10⁻⁹ at D = 200. The linearised estimate is usually far
  smaller: at D = 60 it is 3×10⁻¹⁸ against a worst case of 1.6×10⁻⁹.
- Newton's reach in the continuation, the steps shrinking by 12% per unit of D, is a separate limit not studied here.

No effort estimate is given for a beyond-light-cone solver: the toy did not pass, and the work stopped there as the task
required. Two cheap measurements on the full equations were added to say why: the node test of §5 and the series radii of
§2.3.

## 0. Pre-registered criteria for the toy test (committed in 327f939 before any run; content unchanged since)

**Toy model.** The linear, gravity-free, one-harmonic problem of Proposition 2 in z = x², y = x w(z), is Gauss's
equation z(1−z)w″ + [κ − (α+β+1)z]w′ − αβw = 0 with κ = (n+1)/2. In Proposition 2, α = (1+s)/2, β = (2+s)/2 and the
light-cone coefficient C(n,s) ∝ 1/(Γ(α)Γ(β)) never vanishes for imaginary s. The toy shifts α by a constant so that it
does vanish at a chosen s*:

- α = (s − s*)/2 − 1, β = 1 + s/2, κ = (n+1)/2, unknown complex s (Re s plays the growth rate, Im s the frequency).
- Exact answer: s = s* for every n (α = −1, C(n, s*) = 0 by Gauss's connection formula; the centre-regular solution is
  the polynomial 1 − βz/κ).
- Same structure as the full problem at leading order: the 1/n expansion exists for every s and is analytic at the light
  cone order by order; the selection is the vanishing of an n-independent Stokes constant 1/(Γ(α)Γ(β)); the obstruction
  is the fast mode $`x^{2-n}(1-x^2)^{\kappa-\alpha-\beta}`$.
- Plants: s* = iω* with ω* = 2π/1.6438937, and s* = 0.3 + 2.5i.

**Methods compared**, each a residual R(u) solved by (Gauss–)Newton in mpmath:

- **S(x_m)**, the standard layout of `hp/`: centre series to x = 0.3, Taylor steps along the real axis to x_m, horizon
  series at the light cone evaluated at x_m, match the state (w, (1−z)zw′/κ); unknowns s and the horizon amplitude. x_m =
  0.7 is the `hp/` solver's last node; x_m = 0.5, 0.8, 0.9 are variations.
- **C(ξ₀)**, candidate (c): the centre-regular solution continued from z = ½ along two paths in Re z ≥ ½ (where the fast
  mode is never larger than the slow one), one above and one below the light cone, to the real point z₀ = ξ₀²n (x₀ =
  ξ₀√n); residual Y⁺(z₀) − Y⁻(z₀); unknown s. ξ₀ = 1.
- **A**, candidate (a) on the real axis: the light-cone-analytic solution continued outward along real z from z = 1.9 (the
  horizon series converges for z < 2) to z₀ = n and matched there against Y⁺(z₀) of C.

**Measures**, at n = 20, 60, 100, 200: the condition number κ_s of the selected s (absolute error in s per unit
relative perturbation of the computed states entering R, from the exact Jacobian), and the working precision P_min (decimal
digits) at which the full pipeline returns s* to 10⁻⁸.

**Pass criterion (from the task, made operational).** Fit ln κ_s = a n + p ln n + c over the four n. "Polynomial" means
a < 0.05; "like $`e^{n/2}`$" means a > 0.3. The candidate passes if a_C < 0.05 and a_S(0.7) > 0.3, and its control fails.

**Controls that must fail.** (i) C with both paths through the upper half plane: the residual vanishes identically, so no
s is selected (Jacobian zero). (ii) C started from the centre-singular solution (the $`x^{2−n}`$ branch) instead of the regular
one: it must not return s*. (iii) `sens_fit.py --mutate=exp`: a planted $`e^{n/2}`$ law must be classified as exponential.

**Prediction, written before running** (from §2 below): the fast mode is exactly $`x^{2−n}`$$`(1−x²)^{n/2}`$·(…), whose modulus is
1 on the curve Re x² = ½, which passes through x = 1/√2 ≈ 0.707. So S(x_m) should have κ_s ∝ $`e^{nΦ(x_m)}`$ × poly(n), with
Φ(x) = ln x − ½ln(1−x²): a ≈ −0.02 at x_m = 0.7 (polynomial), a ≈ +0.72 at x_m = 0.9; C polynomial; A exponential
(a ≈ 0.37). If this holds, the second half of the pass criterion fails: the standard layout at the `hp/` matching point is
not $`e^{n/2}`$-ill-conditioned in the toy, and the candidate has nothing exponential to remove.

*Two departures from this plan, both made before the results they affect were seen:*
- A's horizon series is evaluated at z = 1.5, not 1.9. At 1.9 the alternating series cancels catastrophically at large n
  (terms up to about $`e^{1.1n}`$). The prediction for A becomes $`e^{(n/2)ln(z_s/(z_s−1))}`$ = $`e^{0.55n}`$.
- The empirical P_min is measured with injected noise rather than by lowering the working precision. At s* the toy's
  solution is a polynomial and its series terminate, so plain low-precision arithmetic is almost exact and would
  understate the error (first pilot: κ_emp 10³ times below κ_J). The noise model adds a random error of relative size
  10⁻ᴾ to the state after every series evaluation and Taylor step.

## 1. The system the double-double solver solves

From [`hp/src/model.rs`](../hp/src/model.rs), with n = D − 2, ω = 2π/Δ, θ = ωτ, b = a² and G = (n−1)c(b−1):

```math
x c_x = (n-1)\,c\,(b-1), \qquad x b_x = b\,(p^2+m^2) - (n-1)\,b\,(b-1),
```
```math
x(x-c)\,p_x = -\omega x\,p_\theta + G p - c p + \tfrac{n}{2}\,c\,(p+m), \qquad
x(x+c)\,m_x = -\omega x\,m_\theta - G m + c m - \tfrac{n}{2}\,c\,(p+m).
```

(the code integrates a with 2x a_x = a(p² + m²) + (n − 1)(a − a³), the same equation). The residual is multiple
shooting on 0 ≤ x ≤ 1. The unknowns are:
- the centre data p₁(θ) and c₀(θ);
- the states (p, m, a, c) at the nodes 0.3, 0.4, …, 0.7;
- the horizon data m_h(θ) and a_h(θ), with c(1) = 1 as the gauge;
- Δ.

The residual blocks are:
- the centre series at the first node, minus the node state;
- the Taylor integration from each node to the next, minus the next state;
- the horizon series at u₀ = 1 − 0.7, minus the last node's state;
- a phase row.

The horizon series imposes analyticity at x = 1: at each order kk it solves (ωD + κ_kk)p_kk = …, with
κ_kk = (c₁ + 1)kk + c₁ + 1 − n/2. The branch it excludes is $`(1 - x)^{\Lambda_\mu}`$, with
Λ_μ = (n/2 − 1 − iμω + ⟨G(1)⟩)/(1 − ⟨G(1)⟩) (proof/C_selection.md §5.1).

## 2. The equations past the light cone

### 2.1 The same system, for all x > 0

Nothing in §1 is specific to x < 1. The equations hold for every x > 0 and are singular only where x(x − c) = 0 or
x(x + c) = 0. For x > 1 the coefficients x(x ∓ c) are positive while c < x. In the inner region c − 1 = O(1) but
c ≪ x (ξ c_ξ = ½cy² at leading order: c grows like a power of ξ, x linearly).

### 2.2 Characteristics: past the light cone, x is a time coordinate

With ω∂_θ = ∂_τ, the principal parts are x[(x − c)∂_x + ∂_τ] for p and x[(x + c)∂_x + ∂_τ] for m. So p is carried along
dx/dτ = x − c and m along dx/dτ = x + c. With r = x $`e^{−τ}`$, these are the ingoing and outgoing radial null rays
(d ln r/dτ = ∓c/x).

- **Inside the light cone (x < c),** the two families move in opposite directions in x. The problem is a boundary-value
  problem between the centre and the light cone.
- **On x = c (= 1),** the p-family is stationary: the past light cone is a characteristic.
- **Outside (x > c),** both families have dx/dτ > 0. The surfaces x = const are spacelike, and x is a time coordinate.

So past the light cone the self-similar problem is a Cauchy problem in x with τ-periodic data. Its solution on x > 1 is
fixed by its data at x = 1⁺. No boundary condition can be imposed at large real x: in particular, nothing at
x ~ ξ_max√n on the real axis can select Δ. (This answers part of question 2(a), §3.)

### 2.3 Singular points and how far the solution continues

- **x = 0** (regular singular): centre regularity removes the $`x^{−n}`$ branches of both fast modes, as the centre series
  does.
- **x = 1** (regular singular for p): exponents 0 and Λ_μ, with Re Λ_μ ≈ n/2. The non-analytic branch
  $`(1 - x)^{\Lambda_\mu}\varphi(\theta - \omega'\ln(1 - x))`$ is $`C^{\lfloor \mathrm{Re}\,\Lambda \rfloor}`$. At large n the selection therefore asks for more than any fixed order
  of smoothness at the light cone: it asks for analyticity.
- **x = −1** (the m-equation's x + c = 0): the mirror of x = 1 under the centre series' symmetry m(x) = p(−x).
- **x = ∞** is the t = 0 hypersurface. For c = b = 1, in w = 1/x, (1 − w)p_w = (ω/w)p_θ + …: a regular singular point
  with exponents iμω, so p → $`e^{iμ(θ − ω ln x)}`$ = $`e^{−iμω ln r}`$, a function of r alone at t = 0. The inner problem is
  regular there in t = 1/ξ (README, the Weber form).
- **The real continuation.** The solution analytic at x = 1 continues along the real axis through t = 0 to the future
  light cone. For D = 4, Martín-García and Gundlach continued the critical solution regularly up to the future light
  cone, a Cauchy horizon where the curvature is continuous but not differentiable; its continuation beyond it is not
  unique[^mg].
- **Complex singularities** (`series_radius.py`, numerical evidence). Root tests on the double-double series coefficients
  of the saved scan states give finite radii for both series:

  | D | 17.02 | 40.02 | 45.05 | 50.01 | 55.04 | 60.00 |
  |---|---|---|---|---|---|---|
  | radius of the centre series (in x) | 0.474 | 0.560 | 0.593 | 0.622 | 0.649 | 0.674 |
  | radius of the horizon series (in 1 − x) | 0.921 | 0.832 | 0.812 | 0.793 | 0.774 | 0.746 |

  Both are consistent with one complex-conjugate pair of singularities: near x ≈ $`0.47e^{±1.16i}`$ at D = 17 and
  $`0.67e^{±0.84i}`$ at D = 60, that is x² ≈ $`0.45e^{±1.68i}`$ at D = 60. The pair is off the real axis, so it does not limit the
  real continuation, but every complex path (formulation (c)) must avoid it. The horizon series converges at u₀ = 0.3 with
  a margin that shrinks with D (radius 0.75 at D = 60). |x|² grows by about 0.0053 per unit of n over D = 17–60; if that
  continued, the pair would sit at a fixed ξ² of about 0.005 in the inner variable. That is a guess from two radii per D,
  not checked.

### 2.4 What happens to the τ-periodicity

It survives. Past x = 1 the Fourier harmonics obey an ODE system in x, now an evolution. What changes is that the
p-characteristics leave the light cone with τ ≈ ln(x − 1)/(1 − ⟨G(1)⟩) → −∞. A point just outside the light cone
therefore depends on the light-cone data at all phases, and the non-analytic branch winds infinitely often:
$`(x - 1)^{\Lambda}\varphi(\theta - \omega' \ln(x - 1))`$. The analytic solution does not wind. Going around x = 1 in the complex plane shifts the
label by ∓iπω′, which multiplies label harmonic μ by $`e^{±πμω′}`$ (C_selection.md §5.3). §7 is about that factor.

### 2.5 The finite-n equations in the inner variable, exactly

Put x = ξ√n, y = p − m, σ = √n(p + m), η = n(b − 1), and L = ξ²∂_ξ + ωξ∂_θ. Taking the difference and sum of the p and m
equations, with no expansion in 1/n (checked by hand against §1):

```math
L y = c\,\sigma + \frac{c\,\xi\,\sigma_\xi + (G - c)\,\sigma}{n}, \qquad
L \sigma = c\,\xi\,y_\xi + (G - c)\,y, \qquad G = \Big(1 - \frac1n\Big) c\,\eta,
```
```math
\xi c_\xi = \Big(1 - \frac1n\Big) c\,\eta, \qquad
\frac{\xi\,\eta_\xi}{n} = \Big(1 + \frac{\eta}{n}\Big)\Big[\tfrac12\Big(y^2 + \frac{\sigma^2}{n}\Big) - \Big(1 - \frac1n\Big)\eta\Big].
```

At n = ∞ this is the README's inner system (Rozali and Way's leading-order equations): L y = cσ, Lσ = cξy_ξ + c(½y² − 1)y,
ξc_ξ = ½cy², η = ½y². At finite n it keeps three features:
- the light cones, where the principal matrix [[ξ², −cξ/n], [−cξ, ξ²]] is singular: ξ² = c²/n, i.e. x = ±c;
- the stiff b-mode η ∝ $`ξ^{−(n−1)}`$;
- the order-1/n couplings.

The whole region 0 ≤ x ≤ 1 is squeezed into 0 ≤ ξ ≤ $`n^{−1/2}`$.

### 2.6 The fast mode, and where it is of order one

Without gravity (c = b = 1) the fast solution is exactly $`x^{2-n}(1 - x^2)^{n/2-1-s}\,{}_2F_1(\dots)`$ (Propositions 2–3). Its
exponential factor is $`e^{−nΦ}`$, and with z = x²:

```math
\lvert e^{-n\Phi}\rvert = \Big\lvert \frac{1 - z}{z} \Big\rvert^{n/2}, \qquad
\lvert e^{-n\Phi}\rvert = 1 \iff \lvert 1 - z\rvert = \lvert z\rvert \iff \mathrm{Re}\,z = \tfrac12 .
```

The anti-Stokes curve of the fast mode relative to the slow one is therefore the hyperbola Re x² = ½ (derived).
- On the real axis it passes through x = 1/√2.
- At large |x| it is asymptotic to arg x = ±π/4, i.e. the inner anti-Stokes rays arg ξ² = ±π/2, where $`e^{−1/(2ξ²)}`$ is a
  pure phase.
- For Re x² < ½ (which contains 0 < x < 1/√2) the fast mode is exponentially large.
- For Re x² > ½ (which contains 1/√2 < x < 1 and the Stokes line x > 1) it is exponentially small. It vanishes like
  $`\lvert 1 - x\rvert^{n/2}`$ at the light cone and reaches $`e^{−1/(2ξ²)}`$ at x = ξ√n.

So the selecting effect is O(1) along the whole curve from x = 1/√2 to the inner region, and the inner estimate
$`e^{−n/(2x²)}`$ applies only for |x| ≫ $`n^{1/4}`$ (C_selection.md Lemma B5.1(2)). Gravity changes nΦ by terms of order one
(Proposition 3's transport factor and the $`n^{⟨F²⟩}`$ of Lemma B5.1), not by terms of order n, so the curve is the same to
leading order.

### 2.7 Exact conditions equivalent to analyticity at x = 1, evaluated where the effect is O(1)

Let Y = (p, m, b, c) be the centre-regular solution with data (p₁, c₀, Δ).

- **(E1) Matching on the anti-Stokes curve, inside the light cone.** Y(x_m) must lie on the family of solutions
  analytic at x = 1 (the horizon series) at x_m = 1/√2, or at the `hp/` node 0.7. This is exactly what `hp/` does. Away
  from the curve the same condition is exponentially better or worse conditioned, by $`e^{±nΦ(x_m)}`$ (§5).
- **(E2) Reality on the Stokes line, beyond the light cone.** Let Y⁺ be Y continued from the real segment through
  Im x > 0 around x = 1 to a real point x₀ > 1. Then Y is analytic at x = 1 if and only if Y⁺(x₀, θ) is real for all
  real θ. With x₀ = ξ₀√n, Im Y⁺(x₀) is of order $`e^{−1/(2ξ₀²)}`$·S(θ₀) times powers of n, with S the inner multiplier.

  *Argument* (proof sketch; a standard ODE argument for each fixed number of harmonics). For real data the continuation
  through Im x < 0 is Y⁻ = conj Y⁺ on the real axis. If Y⁺(x₀) is real, Y⁺ and Y⁻ agree at x₀. Both solve the same
  equations on (1, x₀], where x is timelike and the Cauchy problem in x is unique, so they agree on all of (1, x₀]. Then
  Y is single-valued and bounded on a punctured neighbourhood of x = 1, so the singularity is removable, and conversely.
  Unlike a real-axis condition at large x (§2.2), (E2) needs the complex continuation around x = 1.
- **(E3) Anywhere on Re x² = ½.** Both (E1) and the lateral comparisons of (E2) can be made at any point of the curve;
  the size of the effect is the same order there. The inner solvers of `solver/inner_stokes/` start their lateral
  solutions on the inner end of it, arg ξ² = π/2 + π/8.

## 3. The three formulations and how their conditioning scales with n

- **(a) Extend the domain along the real axis to x ~ ξ_max√n.** Past the light cone x is timelike (§2.2), so the
  extension is an evolution and gives no new condition. The only exact selection is still analyticity at x = 1 (or
  (E2), which needs complex x). Carrying the light-cone-analytic solution outward costs more:
  - its fast part grows from $`\lvert 1 - x\rvert^{n/2}`$ at the light cone to $`e^{−1/(2ξ²)}`$, so rounding in the fast direction made at
    the starting point x_s is amplified by $`(1 - 1/x_s^2)^{-n/2}`$;
  - the horizon series converges only for z = x² < 2, so the amplification is at least $`2^{n/2}`$ = $`e^{0.35n}`$;
  - in the toy, with z_s = 1.5: κ_emp ∝ $`e^{0.546n}`$ (5.6×10⁵³ at n = 200), against $`e^{0.549n}`$ predicted.

  **Exponential.**
- **(b) The finite-n equations in ξ (§2.5).** A change of variable does not change what is computed. Its conditioning is
  that of the path and condition it is used with: (E1) in ξ is the `hp/` solver, (E2) in ξ is (c). Its use would be
  resolution: uniform steps in ln ξ through the inner region. The stiffness of the b-mode, $`ξ^{−(n−1)}`$, and of the light
  cone at ξ = $`n^{−1/2}`$ stays. **As good as the formulation it is combined with.**
- **(c) A complex path.** Integrate the centre-regular solution along the real axis to near 1/√2, then through
  Re x² ≥ ½ around x = 1 above, to x₀ = ξ₀√n, and impose (E2).
  - Along such a path |$`e^{−nΦ}`$| ≤ 1, so no direction is amplified exponentially in n.
  - In the toy: κ_J = 0.272, 0.277, 0.278, 0.279 and κ_emp = 6.4×10², 3.9×10³, 8.7×10³, 8.2×10³ at n = 20, 60, 100, 200.
    **Polynomial (flat).**
  - Its cost is a per-harmonic weight independent of n (§7), and it must avoid the complex singularities of §2.3.

The standard layout, for comparison, is (E1) at x_m = 0.7. It is **already polynomial** in the toy (κ_J = 259, 384,
293, 80) and on the full equations its sensitivity changes with x_m exactly as $`e^{nΦ(x_m)}`$ (§5).

## 4. The toy test: results

`python3 toy.py` (sweep, ≈ 20 min on one core), `python3 toy.py --redo --n 100,200 --methods S:0.9,S:0.8` (a five-point
stencil, after the first run's central difference masked κ > 10⁴⁰ at n = 200; values below that are unchanged),
`python3 toy.py --plant` (plants), `python3 toy.py --controls`, `python3 toy_table.py` (tables and the criterion; `--selftest`
checks the classifier). Data: `toy_results.jsonl`; logs: `toy_run.log`, `toy_redo.log`, `toy_plant.log`,
`toy_table.log`.

κ_J is the error in s per unit relative perturbation of the matched states. κ_emp is the same with relative noise
injected after every series evaluation and Taylor step (the largest of 3 draws), and P_min = ⌈log₁₀ κ_emp⌉ + 8 digits.

| method | n = 20 | 60 | 100 | 200 | fitted a in ln κ = a n + p ln n + c | prediction $`e^{a n}`$ |
|---|---|---|---|---|---|---|
| S(0.5), standard, x_m = 0.5 | 7.9×10⁻³ | 7.3×10⁻¹² | 3.5×10⁻²¹ | 9.9×10⁻⁴⁵ | −0.550 | −0.549 |
| S(0.6) | 1.3 | 4.3×10⁻⁵ | 7.3×10⁻¹⁰ | 4.7×10⁻²² | −0.288 | −0.288 |
| **S(0.7), the `hp/` layout** | **2.6×10²** | **3.8×10²** | **2.9×10²** | **8.0×10¹** | **−0.021** | −0.020 |
| S(1/√2) | 3.9×10² | 1.3×10³ | 2.2×10³ | 4.4×10³ | −0.001 | 0 |
| S(0.8) | 1.2×10⁵ | 4.1×10¹⁰ | 7.0×10¹⁵ | 4.4×10²⁸ | +0.287 | +0.288 |
| S(0.9) | 9.3×10⁸ | 1.2×10²² | 8.3×10³⁴ | 5.2×10⁶⁶ | +0.724 | +0.725 |
| **C(1), candidate (c)** | **0.27** | **0.28** | **0.28** | **0.28** | **0.000** | polynomial |
| A(1.5), candidate (a) on the real axis (κ_emp) | 7.2×10⁹ | 9.6×10¹⁹ | 4.2×10²⁹ | 5.6×10⁵³ | +0.546 | +0.549 |

(κ_J except for A, whose amplification happens along the path and shows only in κ_emp; A's κ_J is flat at 5–6.5×10⁵.)
Precision needed for s to 10⁻⁸: 11–12 digits for S(0.7) and C(1) at every n; 18, 28, 38, 62 digits for A; 18, 31, 43,
75 for S(0.9).

**Criterion** (`toy_table.log`): a_C = 0.000 < 0.05 holds; a_S(0.7) = −0.021 > 0.3 fails. **NOT MET**, as predicted in §0.

**Plants and controls** (`toy_plant.log`, `toy_controls.log`).
- Newton from s* + (0.05 − 0.04i) converges quadratically to both plants, s* = iω* and 0.3 + 2.5i, at n = 60, for S(0.7)
  and C(1): final |s − s*| = 4×10⁻⁶²–9×10⁻⁶¹ at 60 digits.
- **Control Cuu** (both paths above the light cone): the residual's s-derivative is 3.7×10⁻⁹¹ and 1.3×10⁻⁹⁰ of the
  state at n = 20 and 60, the rounding level at 110 digits, against 3.6–3.7 for C(1). Nothing is selected: the condition
  needs the monodromy around x = 1.
- **Control Csing** (started from the centre-singular branch): |R(s*)| = 2.3|Y| at n = 20 and 60, so s* is not a root.
  The condition needs the centre-regular solution.
- **Control on the fit** (`sens_fit.py --mutate=exp`): a planted $`e^{n/2}`$n² is classified exponential (a = 0.500).

All controls fire (`python3 toy.py --controls` exits 0; `toy_table.py --selftest` and `series_radius.py --mutate` check
the classifier and the radius estimator the same way).

What the toy cannot show: it is linear, so it has no Newton-basin effects and no nonlinear mixing of harmonics. Its
selected solution is a polynomial, so it has no growing intermediate terms in its series either. The full-equation test
of §5 covers what it misses on the first point.

## 5. The full equations: moving the matching point

`python3 hp_node_test.py` (installed hpkernel, 15 runs, ≈ 10 min). Each run takes a saved scan state (M = 32, harmonics
above 13 frozen, nodes 0.3–0.7), re-nodes it with `driver.renode` so the last node is x_m, runs double-double Newton to
its floor, and records max|∂Δ/∂F| per block (as `scan_hp.py`). Prediction: ln[sens(x_m)/sens(0.7)] = n[Φ(x_m) − Φ(0.7)].

| D | x_m | max\|∂Δ/∂F\| | ln ratio measured | predicted | \|F\| reached | linearised ΔΔ |
|---|---|---|---|---|---|---|
| 40.02 | 0.60 | 1.40×10⁶ | −8.83 | −10.18 | 5.3×10⁻²² | +4.5×10⁻¹⁵ |
| 40.02 | 0.65 | 8.84×10⁷ | −4.69 | −5.18 | 3.8×10⁻²⁴ | +1.6×10⁻¹⁵ |
| 40.02 | 0.70 | 9.59×10⁹ | 0 | 0 | 1.2×10⁻²⁶ | −1.6×10⁻²² |
| 40.02 | 0.75 | 2.38×10¹² | +5.51 | +5.54 | 3.4×10⁻²⁷ | +1.9×10⁻¹⁵ |
| 40.02 | 0.80 | 1.30×10¹⁵ | +11.82 | +11.70 | 1.9×10⁻²⁷ | +5.4×10⁻¹⁵ |
| 50.01 | 0.60 | 1.30×10⁷ | −11.64 | −12.85 | 8.6×10⁻²² | +8.8×10⁻¹⁶ |
| 50.01 | 0.65 | 2.91×10⁹ | −6.23 | −6.54 | 2.1×10⁻²³ | +3.9×10⁻¹⁶ |
| 50.01 | 0.70 | 1.49×10¹² | 0 | 0 | 5.5×10⁻²⁵ | +4.3×10⁻²⁰ |
| 50.01 | 0.75 | 1.30×10¹⁵ | +6.77 | +6.99 | 3.8×10⁻²⁶ | +4.1×10⁻¹⁶ |
| 50.01 | 0.80 | 2.76×10¹⁸ | +14.44 | +14.77 | 9.5×10⁻²⁷ | −2.7×10⁻¹⁴ |
| 60.00 | 0.60 | 4.89×10⁷ | −14.41 | −15.53 | 1.5×10⁻¹⁹ | +2.2×10⁻¹⁶ |
| 60.00 | 0.65 | 5.04×10¹⁰ | −7.47 | −7.90 | 1.6×10⁻²¹ | +9.6×10⁻¹⁷ |
| 60.00 | 0.70 | 8.87×10¹³ | 0 | 0 | 1.8×10⁻²³ | −3.1×10⁻¹⁸ |
| 60.00 | 0.75 | 3.07×10¹⁷ | +8.15 | +8.45 | 1.4×10⁻²³ | −5.3×10⁻¹⁵ |
| 60.00 | 0.80 | 2.63×10²¹ | +17.20 | +17.85 | 1.4×10⁻²³ | −4.1×10⁻¹¹ |

What the table shows (numerical evidence):

- **The sensitivity follows the fast mode.** All 12 off-0.7 ratios follow $`e^{n[Φ(x_m) − Φ(0.7)]}`$, a prediction with no
  fitted parameter, over 13 orders of magnitude. They agree to within 1.4 in the logarithm, slightly smaller in size
  inside 0.7. So the double-double solver's selection signal at its matching point is the fast mode $`e^{−nΦ(x_m)}`$, which
  is of order one at 0.7.
- **The floor moves the other way inside 0.7.** For x_m < 0.7 the |F| at which Newton stops (here mostly the re-noded
  residual, since the line search found no decrease) rises roughly as the sensitivity falls. Their product stays within
  a factor of about 6 at D = 40 (0.6: 7×10⁻¹⁶, 0.7: 1.2×10⁻¹⁶). For x_m > 0.7 the floor no longer falls, about 10⁻²³ at D = 60,
  while the sensitivity keeps growing. The `hp/` choice of 0.7 is at, or near, the optimum: at D = 50–60 the product is
  70–220 times smaller at 0.6, a hint that the optimum drifts inward with D, worth a separate check with converged
  Newton runs.
- **Δ is the same in every layout** (hi part 1.8884516644906137, 1.8416311332022643/6, 1.8098842580017149), and the
  linearised corrections are within the expected errors.
- **The floor at 0.7 is not fast-mode amplification:** at D = 60 it is the same at 0.75 and 0.8. Its 1.46-fold growth per
  unit of D has another source, not identified here.

### 5.1 Follow-up: Newton does not converge at 0.6, but the 0.7 solution can be certified there (2026-10-23)

The converged runs asked for above: the local machine's step-size pilot (T43, `hp/step_test.py`, comms 2026-10-23
12:43 local, `hp/data/step_test/pilot/` on wip/step-test) started a continuation at D = 60 with the last node at 0.6.
It converged at the start (floor 1.1×10⁻¹⁹, max|∂Δ/∂F| 4.9×10⁷, product 5.3×10⁻¹²; at 0.7: 1.6×10⁻²³, 8.9×10¹³,
1.4×10⁻⁹), but no continuation step converged, down to steps of 3.4×10⁻⁴. The diagnosis there: the predictor is good
(4.6×10⁻⁹ from the 0.7 solution cut to 0.3–0.6), the double-double Jacobian is right, but the residual at 0.6 is
strongly nonlinear in the horizon data (a 3.3×10⁻¹⁴ move changes Jacobian entries by up to 35%, against 3×10⁻⁴ at
0.7), and the linearisation error, divided by the near-null singular values, throws the Newton step out.

The 0.7 solution, cut to 0.3–0.6, already sits at the 0.6 floor. So `hp/scan_hp.py --certify_nodes 0.3,0.4,0.5,0.6`
steps at 0.7 and checks each converged point in the 0.6 layout without a Newton step: the residual there, max|∂Δ/∂F|
from that layout's double-double LU, the linearised correction −y·F, and the first-order bound Σ|yᵢFᵢ|
(`hp/data/step_test/cert_test/`, two points at D = 60.02 and 60.05, one thread on the cloud machine, about 15 s per
check):

| layout of the check | \|F\| | max\|∂Δ/∂F\| | −y·F | Σ\|yᵢFᵢ\| |
|---|---|---|---|---|
| 0.3–0.7 (the run's own: control) | 2.8×10⁻²³, 1.5×10⁻²³ | 8.9×10¹³, 9.0×10¹³ | +1.4×10⁻¹⁷, +3.6×10⁻¹⁸ | 7.2×10⁻⁹, 4.9×10⁻⁹ |
| 0.3–0.6 | 1.8×10⁻¹⁹, 1.0×10⁻¹⁹ | 4.9×10⁷, 5.0×10⁷ | +2.3×10⁻¹⁶, +2.3×10⁻¹⁶ | 2.7×10⁻¹¹, 2.1×10⁻¹¹ |

The control reproduces the run's own |F| and max|∂Δ/∂F|. At 0.6 the first-order bound is 230–270 times smaller than
at 0.7, the gain section 5 predicted from the products. Both bounds are first-order (exact y and F assumed, second-order
terms dropped): worst cases over the sign of F, not enclosures. The linearised corrections themselves are 10⁻¹⁶ or
less in both layouts.

## 6. What limits the double-double solver, and how that extrapolates

- **Sensitivity: a power law from the high harmonics** (`sens_fit.py`, `lin_ladder.py`). The free-wave obstruction of
  harmonic k at x_m = 0.7 (`lin_ladder.log`) is:

  | k | n = 19 | 39 | 59 | 99 | 199 | local power, n = 39–59 |
  |---|---|---|---|---|---|---|
  | 1 | 3.2×10⁻² | 4.8×10⁻² | 8.2×10⁻² | 0.18 | 0.80 | $`n^{+1.3}`$ |
  | 7 | 1.4×10⁻⁶ | 6.9×10⁻¹¹ | 6.5×10⁻¹³ | 1.9×10⁻¹⁴ | 5.6×10⁻¹⁵ | $`n^{−11.3}`$ |
  | 9 | 8.6×10⁻⁷ | 5.0×10⁻¹² | 4.6×10⁻¹⁵ | 9.5×10⁻¹⁸ | 3.1×10⁻¹⁹ | $`n^{−16.9}`$ |
  | 11 | 6.8×10⁻⁷ | 9.7×10⁻¹³ | 1.2×10⁻¹⁶ | 1.4×10⁻²⁰ | 3.0×10⁻²³ | $`n^{−21.7}`$ |
  | 13 | 5.8×10⁻⁷ | 3.3×10⁻¹³ | 8.3×10⁻¹⁸ | 5.4×10⁻²³ | 5.2×10⁻²⁷ | $`n^{−25.6}`$ |

  - Harmonic 1 is not weakened at all. The high harmonics weaken with n while (kω)² ≳ n: their slow solution is damped
    like $`e^{−(kω)²x²/(2n)}`$ before the matching point, and the damping lifts as n grows. They level off only at their
    large-n limit, of order $`e^{−πkω/2}`$√n.
  - This is close to the measured weak directions of `hp/` (README: $`n^{−10.5}`$, $`n^{−14}`$, $`n^{−17.5}`$, $`n^{−21}`$ for
    k = 7–13 over D = 20–60; the free-wave exponents are 8–20% steeper, with gravity left out), and the power $`n^{21.7}`$
    of max|∂Δ/∂F| is that of the top kept harmonics. This identification is numerical evidence (similar exponents), not
    a proof.
  - Below D = 35 all 16 odd harmonics were active and the growth looked exponential (a = 0.42–0.61 on D = 15–26.5); a
    ladder with ever higher k does that.
- **Extrapolation** (estimates from the linear ladder, with 13 as the top harmonic and gravity's effect assumed to stay
  as at D ≤ 60). The k = 13 obstruction falls by 1.5×10⁵ from n = 59 to 99 and by 1.6×10⁹ to 199, so max|∂Δ/∂F| would
  be about 10¹⁹ at D = 100 and 10²³ at D = 200.
- **Floor.** 1.46-fold per unit of D (D = 36–60, `sens_fit.log`). If it continues, it reaches 10¹⁶ ε at D = 100 and
  10³² ε at D = 200 (ε the arithmetic's unit). That is fatal in double-double at D = 200, about 10⁻³² in quad-double.
- **Worst-case product, in quad-double:** about 10⁻²⁹ at D = 100 and 10⁻⁹ at D = 200. Linearised errors have run far
  below this worst case (D = 60: 3×10⁻¹⁸ against 1.6×10⁻⁹, a factor 5×10⁸).

## 7. Why the complex path does not pay for the full problem

The second half of `lin_ladder.log`: the jump |w(z₀ + i0) − w(z₀ − i0)|/|w(z₀ + i0)| of harmonic k of the free-wave
solution across the Stokes line at z₀ = n is 1.4×10⁴, 6.6×10¹³, 7.0×10²³, 1.0×10³⁴, 1.9×10⁴⁴, 3.8×10⁵⁴ and 8.2×10⁶⁴ for
k = 1, 3, …, 13 at n = 59. It hardly depends on n (powers $`n^{−0.1}`$ to $`n^{−2.4}`$).

- **So (c) removes the n-dependent ladder,** and the n-dependence of the selection along with it.
- **It replaces it with weights $`e^{πkω}`$ ≈ $`10^{5.2k}`$.** These are the half-monodromy factors of §2.4: one lateral
  continuation of harmonic k is that much larger than the other.
- **For a nonlinear solver this is a precision requirement.** Rounding in harmonic k, made where that continuation is
  large, enters the jump at that size. The toy already shows it for k = 1, with half the exponent: κ_emp/κ_J ≈ 2×10³–3×10⁴.
  With harmonics up to 13 the jump spans 61 decades, and products of harmonics mix the scales. A complex-path solver for
  the full equations would need roughly 70–85 digits at every D.
- **The basin is tiny.** A Newton iterate whose harmonic-k conditions are off by δ carries $`e^{πkω}`$δ of fast content
  after the half-turn. The continuation stays close to linear only for δ ≲ $`e^{−πkω}`$.
- **The profile's analyticity strip is too narrow.** The label shift of the half-turn, πω′ ≈ 12, exceeds the width of
  the profile's analyticity strip (|Im θ| ≈ 1.4, README "Leading structure and a prediction"). Only the exactly selected
  solution, whose fast part vanishes, continues around x = 1 with a convergent harmonic series. The same mechanism is
  why harmonic truncation of the inner problem "is not a controlled approximation" (README).

The inner solvers of `solver/inner_stokes/` work around these weights at n = ∞ for M ≤ 7, with scaled starts, de-mixing
and hierarchical Schur solves (`hier.py --schur`). Doing the same at finite n with the 13–31 harmonics `hp/` keeps would
be a research project in itself, with no gain in n to pay for it.

## 8. Discrepancies, for the coordinator

- **The README calls the effect exponentially weakening.** The double-double section says the period "is selected by an
  effect that weakens exponentially with D". "What is known…" says data at n = 18–58 "cannot tell such power laws from
  the exponential weakening seen in max|∂Δ/∂F|".
  - In the frozen regime they can: $`n^{21.7}`$ fits D = 40–61 with a residual of 0.023 in ln, against 0.167 for
    $`e^{n/2} n^p`$, and with a constant implied power in every bin.
  - The task premise ("its size inside the light cone is about $`e^{−n/(2x²)}`$, about $`e^{−n/2}`$ at x = 1") uses the inner
    form where it does not apply. Inside the light cone the size is $`e^{−nΦ(x)}`$, which is 1 at x = 1/√2.
- **Proposed README text** (for "How fast the selection weakens"):

  > The growth of max|∂Δ/∂F| is a power of n, not an exponential, once the harmonic set is fixed: on D = 40–61 (harmonics
  > above 13 frozen) it is $`n^{21.7}`$ to 2% in ln, while $`e^{n/2} n^p`$ misses by 17% (`solver/beyond_lc/sens_fit.py`). The
  > fast mode that carries the selection, $`e^{−nΦ}`$, Φ = ln x − ½ln(1 − x²), has modulus 1 on Re x² = ½, which crosses the
  > axis at x = 1/√2. The solver matches at 0.7, where it is of order one, and moving the last node changes
  > max|∂Δ/∂F| by $`e^{n[Φ(x_m) − Φ(0.7)]}`$ (`solver/beyond_lc/hp_node_test.py`). What grows is the high-harmonic ladder:
  > in the free-wave problem the obstruction of harmonic k at x = 0.7 weakens like $`n^{−11}`$…$`n^{−26}`$ for k = 7–13 at
  > n = 39–59, before levelling off at n ≫ (kω)² (notes/BEYOND_LIGHT_CONE.md).

## 9. Files, commands, and what shares code

All in `solver/beyond_lc/`, run from there:

| file | what it does | output |
|---|---|---|
| `sens_fit.py` | fits of max\|∂Δ/∂F\| (hp/data/delta_dd.csv); `--mutate=exp` plants $`e^{n/2}`$n² (must be classified exponential) | `sens_fit.log` |
| `lin_angle.py` | free-wave matching angle per harmonic at x_m = 0.5, 1/√2, 0.8, 0.9 (exact ₂F₁) | `lin_angle.log` |
| `lin_ladder.py` | the ladder at x_m = 0.7 and the Stokes-line jump per harmonic | `lin_ladder.log` |
| `toy.py` | the pre-registered toy test; `--plant` plants; `--controls` controls (exit 1 if one does not fire); `--redo` | `toy_results.jsonl`, `toy_*.log` |
| `toy_table.py` | tables, fits, the criterion; `--selftest` | `toy_table.log` |
| `hp_node_test.py` | the full-equation node test (installed hpkernel) | `hp_node_test.jsonl`, `.log` |
| `series_radius.py` | radii of the centre and horizon series; `--mutate` plants R = 0.5 | `series_radius.log`, `.json` |

Independence:
- `toy.py` (own series and Taylor recurrences in mpmath) and `lin_angle.py`/`lin_ladder.py` (mpmath's `hyp2f1`) share no
  code. They agree on the $`e^{nΦ(x_m)}`$ law: toy slopes −0.549, −0.288, −0.020, +0.288, +0.725; `lin_angle.log`'s harmonic-1
  rows.
- `hp_node_test.py` and `series_radius.py` use the installed `hpkernel` (the `hp/` solver itself) and share nothing with
  the toy.

[^mg]: J. M. Martín-García and C. Gundlach, *Global structure of Choptuik's critical solution in scalar field collapse*,
    PRD 68, 024011 (2003), [gr-qc/0304070](https://arxiv.org/abs/gr-qc/0304070) (already in the README's references).

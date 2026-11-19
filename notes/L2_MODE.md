# L2_MODE: is the ℓ = 2 mode exactly marginal at D = ∞?

Discovery candidate 1 of [DIRECTION_REVIEW.md](DIRECTION_REVIEW.md) §2.1, issue #993. The review argues that deforming s
= r²/(2n) by a harmonic polynomial h of degree ℓ gives an exact mode of the D = ∞ inner problem with λ = 2 − ℓ, so that
polar ℓ = 2 would be exactly marginal, and it names the gravitational step as the one that may fail. This note is
experiment (a) of §2.1, a sympy check at leading order in 1/n. The script is `solver/inner_pert/l2_mode.py`. Written
2026-11-19 by Claude (Anthropic) under Nathan Moinvaziri's direction; not reviewed. Status words as in AGENTS.md.

## Result

- **The ℓ = 2 deformation solves the D = ∞ inner equations, gravity included, at leading order in 1/n** (*checked
  symbolically*, within the ansatz of §1). The scalar and the lapse follow S = s + e h/(2n). The O(1/n) part of the
  spatial metric along the normal of the level sets of S adjusts at O(e) by A = μ δS/s. Every Einstein component and the
  scalar equation then vanish at their leading orders. This holds for the two representatives tried, p = q = (n + 1)/2
  and p = 1 (§4).
- **The gravitational step the review flagged holds.** The O(1) lapse equation E^t_t and the scalar equation are
  satisfied with no correction at all. The variation of the level sets' curvature does not reach the lapse at this
  order. Only the momentum constraint and the O(1/n) spatial equations need the metric to adjust.
- **The non-harmonic control fails, and no correction rescues it** (§5). With h = |y|² and p = q, the scalar and lapse
  equations fail at O(1). Every O(1/n) metric correction can be left free, and the lapse equation then still leaves −κ/2
  on admissible data.
- **A correction to the review's control.** h = |y|² is non-harmonic at leading order only when p/n is of order one,
  since Δ(|y|²/(2n)) = p/n. For p fixed it is harmonic at leading order, and the check rightly passes it (§5).
- **What it means.** At D = ∞ the inner problem has a λ = 0 deformation for ℓ = 2: a marginal, non-spherical one, so
  λ₂(D) = c₂/n + O(1/n²) if it continues. A reading, not proved: it is a volume-preserving linear stretch of space
  applied to the lump, which the leading-order inner equations do not see (§6). What this does not cover is listed in
  §7. The next step is (b), c₂ (§8).

## 1. The ansatz

Spatial ℝ^{n+1} = ℝ^p × ℝ^q, with p + q = n + 1, y ∈ ℝ^p and z ∈ ℝ^q. In inner variables r₁ = |y| = √(2ns) cos χ and
r₂ = |z| = √(2ns) sin χ, so that s = r²/(2n), the metric is O(p) × O(q) invariant with zero shift:

```math
g = -c^2\,dt^2 + n\Big(\frac{ds^2}{2s} + 2s\,d\chi^2\Big) + \frac{\mu}{n}\,\frac{dS^2}{|\nabla S|^2} + r_1^2\,d\Omega_{p-1}^2 + r_2^2\,d\Omega_{q-1}^2 + (\text{corrections}) .
```

It is flat space, plus the lapse c, plus an O(1/n) spatial part along the normal of the level sets of S (|∇S| in the
flat metric). The scalar is φ = Φ.

- **The deformation.** S = s + e δS. The fields c, μ and Φ are functions of (S, t), linearised in e:
  F(S, t) → F(s, t) + e δS F_s.
- **The corrections.** Unknown functions of (s, t, χ) at O(e):
  - g_ss += eA/(2s), g_sχ += eB and g_χχ += 2s eC, each of relative order 1/n;
  - each fibre warp U_k += eV_k/n;
  - the lapse c += eL, of order one.

  The claim being tested is that c and Φ follow S, so L = 0 throughout.
- **The curvature.** The multiply-warped-product formulas. The base is (t, s, χ) and the fibres are unit spheres of
  dimension m₁ = p − 1 and m₂ = q − 1 with warps e^{U₁} = r₁ and e^{U₂} = r₂:

```math
R_{ab} = \tilde R_{ab} - \sum_k m_k\big(\tilde\nabla_a\tilde\nabla_b U_k + \partial_aU_k\,\partial_bU_k\big),\qquad
R^{(k)} = (m_k - 1)e^{-2U_k} - \tilde\Box U_k - \nabla U_k\cdot\sum_l m_l\nabla U_l ,
```

  and □φ = □̃φ + Σ_k m_k ∇U_k·∇φ. The equations are R_μν = κ ∂_μφ ∂_νφ and □φ = 0.
- **The harmonic.** h = q|y|² − p|z|². For p = q it is ∝ s cos 2χ. For p = 1 it is ∝ s (cos²χ − sin²χ/n).
- **Planted check.** Flat space (c = 1, μ = Φ = 0) gives zero residuals for every δS and both representatives
  (`--flat`).

## 2. The background

At e = 0 the residuals at their leading orders are as follows. E^t_t, E^t_s and W are of order one. E^s_s, E^χ_χ and the
fibre equations are of order 1/n. They give

```math
c_s = \frac{\kappa\,\Phi_t^2}{c},\qquad \Phi_s = \frac1c\Big(\frac{\Phi_t}{c}\Big)_t,\qquad \mu = \frac{2s\,c_s}{c},
```

with μ_t = 4κ s Φ_s Φ_t (E^t_s) following from them. At κ = 1 this is Proposition 5's system in the form of
`rw_map.py` (README, Proposition 5, "the same system as Rozali and Way's"). Reduced on shell, every background residual
vanishes at its natural order (`--check`, first line). The script eliminates μ and every s-derivative of c and Φ, using
s as the evolution variable.

## 3. Calibration: the ℓ = 0 shift

δS = 1 is the s-translation, Proposition 5's exact λ = 2 mode. It does **not** pass with the metric's μ following S.
Since μ = 2s c_s/c carries an explicit s, the exact shift changes it by e(μ_s − μ/s), not eμ_s. With that correction, A
= −μ/s, every residual vanishes. Without it, E^t_s, E^s_s, E^χ_χ and the fibre equations fail. So the test reproduces a
known exact mode, and it fails when the metric is wrong.

## 4. ℓ = 2

`python3 l2_mode.py --check` (from `solver/inner_pert/`, about 45 s; sympy 1.14.0):

| case | correction | result |
|---|---|---|
| background, p = q | — | all zero on shell |
| ℓ = 0, δS = 1 | A = −μ/s | all zero |
| ℓ = 0, δS = 1 | none | nonzero in E^t_s, E^s_s, E^χ_χ, F₀, F₁ |
| **ℓ = 2, p = q, δS = s cos 2χ** | **A = μ cos 2χ** | **all zero** |
| ℓ = 2, p = q | none | nonzero in E^t_s, E^s_s, E^χ_χ, E^s_χ, E^χ_s, F₀, F₁. E^t_t, E^t_χ and W vanish |
| **ℓ = 2, p = 1, δS = s (cos²χ − sin²χ/n)** | **A = μ δS/s** | **all zero** |
| control, p = q, δS = s cos²χ (h = \|y\|²) | none | nonzero in every component, E^t_t and W included |
| control, p = q, every correction free | A, B, C, V₁, V₂ free, L = 0 | W fixes (V₁ + V₂)_t, then E^t_t leaves −κ/2 at c = 1, Φ = t |

"All zero" means that every component vanishes at every power of n from its natural order up, after the background
equations are used:
- the order-one components E^t_t, E^t_s, E^t_χ and W;
- the order-1/n components E^s_s, E^χ_χ, E^s_χ, E^χ_s and the fibre equations.

The only correction ℓ = 2 needs is in the O(1/n) spatial metric along the normal. To first order in e the normal part is
μ(S) S/s for ℓ = 2, against μ(S) s/S for the ℓ = 0 shift. So the review's sentence "if the gravitational equations also
depend on x only through S" holds for the lapse and the scalar, and not for μ.

Control of the script: `--check --mutate=curv` drops the fibres' own curvature (m − 1)e^{−2U}. The background check then
fails in the fibre equations and the script exits 1. That term carries no δS, so the e-linear cases do not see it. Their
power to fail is shown by the control rows instead.

## 5. The control, and when it is one

h = |y|² is the review's control. Its Laplacian is Δ|y|² = 2p, so in inner variables Δ(|y|²/(2n)) = p/n.

- **p of order n (p = q here).** The deformation changes the scalar equation's effective Laplacian at O(1), like a
  rescaling of s without t, and W and E^t_t fail. The O(1/n) corrections A, B and C do not enter those two equations.
  The warp corrections V₁ and V₂ enter them through (V₁ + V₂)_t. W fixes that combination. E^t_t is then a condition on
  the background alone, ∂_t(V_t/c) = κΦ_t²/c with V_t fixed by W, and at the admissible local data c = 1, Φ = t (c_s =
  κ, Φ_s = 0 there) it leaves −κ/2. So no choice of corrections makes the control a solution.
- **p fixed (p = 1).** Δ(|y|²/(2n)) = 1/n, and |y|²/(2n) differs from h/(2n²) by an s-rescaling of relative order 1/n.
  At leading order the two are the same deformation, so the control passes, as it should. The review's wording "h = |y|²
  alone, which is not harmonic" needs p/n of order one to be a control at this order.

## 6. A reading of the result (not proved)

For a quadratic h = ½ x·Hx with H traceless, the linear map x → (1 + αH)x applied to the spherical solution is pure
gauge. With α = ε/4, so that |(1 + αH)x|²/(2n) = s + εh/(2n), it gives the same S as the review's deformation S = s +
εh/(2n). It differs from the configuration checked here by the constant metric perturbation 2αH_ij dx^i dx^j, since the
checked configuration keeps the flat metric exact, and by terms in the O(1/n) part of the metric. A constant metric
perturbation, with the matter held fixed, enters the scalar equation at leading order only through its trace, tr H
Φ_s/n, as the s-part of the flat Laplacian does. For harmonic h the trace vanishes. For the control it is 2p. On this
reading:
- The ℓ = 2 mode is a volume-preserving stretch of the lump in a flat inner metric: an ellipsoidal critical solution
  that costs nothing at D = ∞. It is physical, not gauge, since the matter's shape changes in proper distance.
- It is specific to quadratic h. The review's λ = −1 for ℓ = 3 (cubic h) does not follow from it and was not checked.

## 7. What this does not cover

1. **The ansatz.** Zero shift, a flat spatial metric plus O(1/n) corrections in the components listed in §1, and O(p) ×
   O(q) symmetry. The check finds one solution in this ansatz, which shows existence at leading order. Other gauges were
   not tried.
2. **Linear order in e** only.
3. **Selection.** Proposition 5 (iii) shows that the λ = 2 mode keeps the background's vanishing fast-type Stokes
   multipliers. The analogous statement for this deformation is not shown. Along each ray it is a rescaling of s by
   1 + e cos 2χ, which is not a symmetry of the spherical system by itself.
4. **The outer region and the light cone.** Only the inner region's leading order is checked.
5. **The sign of c₂.** The check says λ₂ = 0 at D = ∞. Whether the mode is unstable at large finite D is (b).

## 8. Step (b), first step: the O(1/n) problem for c₂ (#1001)

If the mode continues to finite n, λ₂(D) = c₂/n + O(1/n²), and the sign of c₂ decides whether the critical solution
gains a second, non-spherical unstable mode at large D. This section reduces the ℓ = 2 residuals one power of n below
their natural orders with the O(1/n) background, writes the O(1/n) equations for the deformation's correction, and asks
whether c₂ follows from a solvability condition on the inner problem alone, as λ₁ does in
[FIRST_ORDER.md](FIRST_ORDER.md) §4. Written 2026-11-19 for issue #1001.

**Result.**

- **c₂ follows from a solvability condition on the inner problem alone.** No O(1/n) outer data are needed. The finite-n
  deformation fails the finite-n equations at O(1/n) only by a forcing F built from the n = ∞ background and the inner
  equations' O(1/n) terms. c₂ multiplies a second forcing F_λ, from the ln(−t) of e^{c₂τ/n}. c₂ is then fixed by
  requiring the correction to have zero multiplier, as FIRST_ORDER.md §4's symmetry route fixes λ₁ (§8.4).
- **The O(1/n) equations for the correction are written down** (§8.3). They are 𝓛₀[W₂] = −F + c₂F_λ, with F and F_λ
  explicit. F is nonzero in every component, ∝ cos 2χ in the scalar-type rows and ∝ sin χ cos χ in the χ rows. The
  p = 1 representative gives the same structure, with its own F.
- **On the lapse and the scalar, the ℓ = 2 operator is the ℓ = 0 operator conjugated by sY** (*checked symbolically*,
  with V₁ + V₂ = 0, §8.3). So c₂ can reuse FIRST_ORDER.md's λ = 2 multiplier map, null vector and profile, with a new
  forcing.
- **Both planted checks pass, and both controls fire** (*checked symbolically*, §8.1–8.2):
  - this background equals FIRST_ORDER.md §1's heat system and η to O(1/n) through C = α/a, Y = 2Φ_t/C, S = 2Φ_s;
  - the ℓ = 0 shift reproduces §4's λ₁ forcing exactly;
  - dropping the O(1/n) background changes the background, the ℓ = 0 forcing and the ℓ = 2 forcing.
- **Assumed, not shown** (§8.5): a one-dimensional kernel at λ = 0, V₁ + V₂ = 0 for general ℓ = 2 perturbations, and
  the multiplier of the angular problem taken along rays.

### 8.1 The background to O(1/n)

E^t_t, W and E^χ_χ one order below their natural orders give the background's O(1/n) rules (`bg_rules1`):

```math
c_s = \frac{\kappa\Phi_t^2}{c} + \frac{c}{n}\,E^t_{t,1},\qquad \Phi_s = \frac1c\Big(\frac{\Phi_t}{c}\Big)_t - \frac1n W_1,\qquad
\mu = \frac{2s\,c_s}{c} - \frac{2s}{n}\,E^\chi_{\chi,2},
```

Here E^t_{t,1}, W₁ and E^χ_{χ,2} are those residuals' coefficients one power of n below natural, each reduced by the
leading rules. With the rules every component vanishes on shell to O(1/n), including E^t_s, E^s_s and the fibre
equations, which were not used. In the p = 1 frame the same rules make every component vanish too.

**Planted check 1, against FIRST_ORDER.md §1.** Through

```math
C = \frac{\alpha}{a} = \frac{c}{\sqrt{1+\mu/n}},\qquad Y = \frac{2\Phi_t}{C},\qquad S = 2\Phi_s\qquad(\kappa = 1),
```

this background is FIRST_ORDER.md §1's heat system to O(1/n), its Y-equation and C-equation (the S-equation holds
identically), and μ is its slaved η = ½y² + (1/n)(½y² + ½σ² − ξyy_ξ). All three residuals vanish (`--first-order`). The
two derivations are independent: this one from Einstein's equations in the O(p) × O(q) frame, FIRST_ORDER's from the
solver's equations. The map is exact: the wave equation in polar-areal form gives Y_t = CS + (1/n)C(S + 2sS_s + ½sY²S)
by itself. **Control:** with the leading rules only, all three leave O(1/n) residuals.

### 8.2 Planted check 2: the ℓ = 0 shift gives FIRST_ORDER.md's λ₁ forcing

FIRST_ORDER.md §4's symmetry route forces the correction W of the λ = 2 mode by the s-translation breaking terms of the
O(1/n) equations, C(2S_s + ½Y²S) in the Y-equation and C(½S² − YY_s) in the C-equation, the S-equation unforced. Here
the ℓ = 0 shift is built to translate (C, Y, S) exactly. Φ follows S. The normal metric part takes
A = −∂_s^{expl}η, so that μ = η holds to O(1/n). The lapse takes L = cA/(2n), so that C translates. Since
W = −FO_Y/(2c) and E^t_t = −FO_C/c at leading order, their O(1/n) parts must be −½(2S_s + ½Y²S) and −(½S² − YY_s).
They are, with zero difference and zero O(1) parts (`--first-order`). **Control:** with the leading background only,
both differ from the forcing.

### 8.3 The O(1/n) equations for the ℓ = 2 correction

Write the finite-n mode as e^{λτ}(D₂Z_n + W₂/n), with D₂ the deformation of §4, λ = c₂/n and
e^{λτ} = (−t)^{−λ} = 1 − (c₂/n) ln(−t) + …. At O(1/n) the correction obeys

```math
\mathcal L_0[W_2] = -F + c_2\,F_\lambda ,\qquad F_\lambda = \mathcal L_0\big[\ln(-t)\,D_2Z_0\big],
```

where 𝓛₀ is the leading-order linearised operator on the O(e) fields (lapse L, scalar P, metric A, B, C, warps V₁, V₂),
F is the deformation's own O(1/n) residual on shell (the reduced residuals one order below natural, with W₂ = 0), and
F_λ comes from t-derivatives acting on ln(−t), since 𝓛₀[D₂Z₀] = 0.

**The operator.** Its order-one rows are

```math
W:\ P_s - \frac{P_{tt}}{c^2} + \frac{c_tP_t}{c^3} + \frac{\Phi_tL_t}{c^3} + \frac{2\Phi_{tt}L}{c^3} - \frac{3\Phi_tc_tL}{c^4} + \frac{(\cot\chi-\tan\chi)P_\chi}{4s} - \frac{\Phi_t(V_1+V_2)_t}{2c^2},
```
```math
E^t_t:\ -\frac{L_s}{c} + \frac{2\kappa\Phi_tP_t}{c^2} - \frac{\kappa\Phi_t^2L}{c^3} + \frac{(\tan\chi-\cot\chi)L_\chi}{4sc} + \frac{(V_1+V_2)_{tt}}{2c^2} - \frac{c_t(V_1+V_2)_t}{2c^3},
```

and the momentum constraints E^t_s, E^t_χ and the O(1/n) spatial rows bring in A, B, C and the warps (all printed by
`--first-order --print`). With V₁ + V₂ = 0 and L = sY γ, P = sY ψ, Y = cos 2χ, the angular terms cancel the extra γ and
ψ that ∂_s(s·) produces, and

```math
W[sY\gamma, sY\psi] = sY\,W^{(0)}[\gamma,\psi],\qquad E^t_t[sY\gamma, sY\psi] = sY\,E^{t(0)}_t[\gamma,\psi],
```

the ℓ = 0 operator (`--first-order`, item 6, exact in sympy). Control: with cos²χ in place of cos 2χ, which is not an
ℓ = 2 harmonic, the differences are −γ/(2c) and ψ/2. So on the lapse and the scalar the ℓ = 2 problem at n = ∞ is the
ℓ = 0 problem conjugated by sY, which is the review's λ = 2 − ℓ in operator form. Whether the constraint rows allow
V₁ + V₂ = 0 for every ℓ = 2 perturbation, and not only for the deformation, is not shown here (§8.5).

**The forcing** (`--first-order --print` prints every component). Per unit e, on shell, with κ = 1 and Y = cos 2χ:

```math
F[E^t_t] = \frac{2sY}{c^6}\big(2c^2\Phi_t^4 - c^2\Phi_t\Phi_{ttt} + c^2\Phi_{tt}^2 + c\,\Phi_t^2c_{tt} + c\,\Phi_t\Phi_{tt}c_t - 2\Phi_t^2c_t^2\big),\qquad
F[E^t_\chi] = -\frac{2s\sin 2\chi}{c^5}\,\Phi_t\big(c\,\Phi_{tt} - \Phi_tc_t\big),
```
```math
F_\lambda[E^t_t] = \frac{2sY}{t\,c^5}\,\Phi_t\big(c\,\Phi_{tt} - \Phi_tc_t\big),\qquad
F_\lambda[E^t_\chi] = \frac{s\sin 2\chi}{t\,c^4}\,\Phi_t^2 .
```

F[W] has nine terms of the same kind times −2sY/c⁷, and the other rows are of the same form. In F_λ the ln(−t) and iπ
parts cancel exactly, because 𝓛₀[D₂Z₀] = 0. What is left are 1/t terms, zero in E^χ_χ and the fibre rows, which carry
no t-derivative. The deformation written as explicit corrections is L = sY c_s, P = sY Φ_s, A = (sμ_s + μ)Y,
B = −μ sin 2χ, with C = V₁ = V₂ = 0. 𝓛₀ annihilates it, a second check of §4's result.

| `--first-order` (about 12 min, one process at nice 10) | result | required |
|---|---|---|
| background, every component on shell to O(1/n) | all zero | zero |
| planted 1: FIRST_ORDER.md §1's Y- and C-equations and η | all zero | zero |
| planted 1 control, leading rules only | all nonzero | nonzero |
| planted 2: the ℓ = 0 shift's W and E^t_t against §4's forcing | equal; O(1) parts zero | equal |
| planted 2 control, leading background only | both differ | differ |
| ℓ = 2: O(1) parts on the O(1/n) shell | all zero | zero |
| ℓ = 2: the O(1/n) forcing F | nonzero in all ten components | nonzero |
| ℓ = 2 control, F with the leading background only | differs in all ten | differs |
| p = 1: background on shell to O(1/n) with the same rules | all zero | zero |
| p = 1, ℓ = 2, δS = s(cos²χ − sin²χ/n): O(1) parts | all zero | zero |
| p = 1, ℓ = 2: F, and its control | nonzero in all nine; control differs in all nine | nonzero; differs |
| 𝓛₀ on D₂Z₀ written as corrections | all zero | zero |
| F_λ | nonzero in the seven rows with t-derivatives | nonzero |
| conjugation of W and E^t_t by sY (V₁ + V₂ = 0) | zero difference | zero |

The zero tests evaluate each on-shell expression exactly, on explicit test functions Φ(t) and c(t) at two points, to 50
digits, against 10⁻³⁰. After the reduction only t-derivatives at fixed s remain, so this decides identities that
`simplify` takes too long to settle. Planted 2 was also settled by `simplify`, with zero difference
(`l0_first_order_check()`, about 12 min).

### 8.4 Inner solvability or outer data

c₂ follows from a solvability condition on the inner problem alone, by the analogue of FIRST_ORDER.md §4's symmetry
route:

1. **The forcing is inner.** F and F_λ are built from the n = ∞ background Z₀ and the inner equations' O(1/n) terms. The
   reduction used only the inner rules of §8.1 and the deformation, with no Z₁, no K₁ or ω₁ (FIRST_ORDER.md §3) and no
   outer matching. That is the property that makes the symmetry route need only n = ∞ objects for λ₁.
2. **The finite-n deformation has zero multiplier.** Along each ray D₂ acts as sY ∂_s at fixed t, which is (Y/2) ξ∂_ξ at
   fixed τ. That commutes with taking the jump across the Stokes line in ξ, so the jump of D₂Z_n is (Y/2) ξ∂_ξ of the
   jump of Z_n, and it vanishes when Z_n is selected.
3. **Solvability.** The multiplier of W₂ must vanish, and it is linear in the forcing, so
   c₂ = −ℓ·S(W^F)/(ℓ·S(W^λ)), with W^F and W^λ the responses to −F and F_λ and ℓ the left null vector of the n = ∞ ℓ = 2
   multiplier map at λ = 0.
4. **It survives truncation.** ℓ = 0's symmetry route fails in a truncated model because its gauge step multiplies by
   P(θ), which mixes θ-harmonics (FIRST_ORDER.md §4). δS = s cos 2χ vanishes at the centre, so ℓ = 2 needs no gauge
   step. D₂ acts harmonic by harmonic in θ, and the route should hold in the M-harmonic models too (not checked).

So no O(1/n) outer data are needed. This is a structural argument (items 1–4), with item 1 checked by the reduction.
What it assumes is listed in §8.5.

### 8.5 What is assumed, and the next task

- The kernel of the n = ∞ ℓ = 2 multiplier map at λ = 0 is one-dimensional, spanned by D₂Z₀. With the conjugation of
  §8.3 it is the image of ℓ = 0's kernel at λ = 2, the s-translation, whose simplicity Proposition 5 and FIRST_ORDER.md
  use. That holds if the constraint rows allow V₁ + V₂ = 0 for a general ℓ = 2 perturbation, which is open.
- Item 2 of §8.4 takes the Stokes jump of the ℓ = 2 perturbation along rays at fixed χ. The multiplier map of the
  angular problem is not written down.
- Only linear order in e, and the O(p) × O(q) ansatz with zero shift.

**Recommended next task: c₂ in the M-harmonic truncation.**
- Map F and F_λ to FIRST_ORDER.md's inner variables (y, σ, c) with the map of §8.1.
- Solve 𝓛₀W = −F and 𝓛₀W = F_λ on the hierarchical solver's states at M = 1, 3, 5, through the n = ∞ ℓ = 0 machinery at
  λ = 2, using the conjugation.
- Take c₂ from the ratio of the two multipliers' projections on ℓ.
- Planted check: the same code with FIRST_ORDER's ℓ = 0 forcing must return λ₁ = 3.0324 at M = 5.
- Control: F with its O(1/n) background terms dropped must change c₂.
- First, settle V₁ + V₂ = 0 by solving the ℓ = 2 constraint rows symbolically for A, B, C at V = 0.

Cost: DIRECTION_REVIEW.md's 2–4 agent-days for (b), and CPU of the order of FIRST_ORDER.md's λ₁ runs, since the
machinery is the same.

## 9. Reproduce

From `solver/inner_pert/` with venv-314 (sympy 1.14.0):

```
python3 l2_mode.py --flat                    # planted: flat space, zero residuals; PASS
python3 l2_mode.py --check                   # sections 2-5 (~45 s); PASS, exit 0
python3 l2_mode.py --check --mutate=curv     # the fibre curvature dropped: FAIL, exit 1
python3 l2_mode.py --first-order             # section 8 (~12 min): the O(1/n) background, planted checks 1 and 2
                                             # with their controls, F with its control, L0 on D2 Z0, F_lambda; PASS
python3 l2_mode.py --first-order --print     # the same, and prints the operator L0, F and F_lambda
```

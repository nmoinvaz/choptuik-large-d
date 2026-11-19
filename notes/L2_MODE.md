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

## 8. Scoping (b): c₂

The review proposes (b) by [FIRST_ORDER.md](FIRST_ORDER.md)'s method, which gave λ₁ = 3.0324 for ℓ = 0 to 0.3% at
M = 5. That needs three things:
- the O(1/n) inner equations of FIRST_ORDER.md §1 extended to the O(p) × O(q) ansatz, or the linearised ℓ = 2 equations
  to O(1/n), which this script's machinery gives directly (the residuals at the next power of n);
- the O(1/n) correction to the deformed solution;
- a solvability condition for c₂ in the M-harmonic truncation, as FIRST_ORDER.md §4 does for λ₁.

The first step is the residuals of the ℓ = 2 case one power of n below their natural orders. The script computes them
already, but does not reduce them. That step decides whether c₂ follows from a solvability condition or needs the
O(1/n) outer data too. Cost (DIRECTION_REVIEW.md): 2–4 agent-days.

## 9. Reproduce

From `solver/inner_pert/` with venv-314 (sympy 1.14.0):

```
python3 l2_mode.py --flat                    # planted: flat space, zero residuals; PASS
python3 l2_mode.py --check                   # sections 2-5 (~45 s); PASS, exit 0
python3 l2_mode.py --check --mutate=curv     # the fibre curvature dropped: FAIL, exit 1
```

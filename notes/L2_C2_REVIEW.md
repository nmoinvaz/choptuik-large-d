# L2_C2_REVIEW: independent review of c₂ = −λ₁ (#1010)

A review of [L2_MODE.md](L2_MODE.md) §9, which claims that the ℓ = 2 mode's growth rate at large D is
λ₂ = c₂/n + O(1/n²) with c₂ = −λ₁, λ₁ the ℓ = 0 first-order coefficient of [FIRST_ORDER.md](FIRST_ORDER.md). Written
2026-11-20 by Claude (Anthropic) under Nathan Moinvaziri's direction, as a separate agent that had not seen the work.
Status words as in AGENTS.md.

The order of work is part of the record. §1 was derived, checked with `solver/inner_pert/l2_c2_review_sym.py` and
committed before L2_MODE.md §9.1–9.6 and the `l2_c2*.py` files were opened. Only L2_MODE.md §§1–8 and FIRST_ORDER.md
were read for it. §2 onward was written after reading them.

## 1. The derivation, made before reading §9

### 1.1 The frame: a linear map of space, minus a constant metric

Take the exact spherical solution at finite n in polar-areal form, with zero shift,

```math
g = -c^2dt^2 + \delta_{ij}dx^idx^j + \frac{\mu}{n}\,dr^2,\qquad \phi = \Phi,\qquad c,\ \mu,\ \Phi\ \text{functions of}\ (s,t),\quad s = \frac{r^2}{2n}.
```

Let H be a constant, symmetric, traceless matrix, h = ½ x·Hx and ψ the linear map x → (1 + (ε/4)H)x.

- |ψ(x)|²/(2n) = s + εh/(2n) + O(ε²). So the pulled-back fields follow S = s + εh/(2n), which is L2_MODE.md §1's
  deformation. Write εh/(2n) = e s Y₂, with Y₂ = x·Hx/(2r²)·(ε/e) the ℓ = 2 harmonic (cos 2χ for p = q).
- The pulled-back flat metric is δ + k, with k = (ε/2)H constant. Its radial component is k_NN = k(N, N) = e Y₂, where
  N = x/r.
- ψ is a diffeomorphism that commutes with the scaling (t, x) → (at, ax). So ψ*(g, φ) solves the finite-n equations
  exactly, and its linearisation, the Lie derivative along ξ = (ε/4)Hx·∂ₓ, is an exact λ = 0 solution of the finite-n
  linearised equations 𝓛ₙ. It is pure gauge.

So the deformation of L2_MODE.md §1, which follows S and keeps the flat metric, is

```math
D_2Z_n = \mathcal L_\xi(g,\phi) - (k, 0) + (\text{metric terms of relative order } 1/n),
\qquad n\,\mathcal L_n[D_2Z_n] = -\,n\,\mathcal L_n[(k,0)] + \mathcal L_0[\text{those terms}].
```

The ℓ = 2 forcing is minus the linearised Einstein–scalar equations applied to a constant traceless spatial metric
perturbation with no scalar and no lapse perturbation. On the lapse and the scalar the two frames have the same fields,
since ξ(s) = e s Y₂ gives 𝓛_ξ(c, Φ) = e s Y₂ ∂ₛ(c, Φ) exactly.

This frame was chosen because it makes the forcing a two-line covariant computation, and because it differs from the
frame of `l2_mode.py`, which deforms the fields and solves for the metric corrections.

### 1.2 The spherical equations to O(1/n), by hand

Two rows, κ = 1: W = □φ and E = c (R^t_t + Φ_t²/c²). With zero shift, R^t_t = −D²c/c − ∂ₜK/c + K_ijK^ij, with
K = −∂ₜ ln√γ/c the trace of the extrinsic curvature and γ = δ + (μ/n)NN the spatial metric, det γ = a² = 1 + μ/n.
For a function f of (s, t), ∂ᵢf = fₛxᵢ/n and

```math
D^2f = \frac1a\,\partial_i\big(a\,\gamma^{ij}\partial_jf\big) = \Big(1+\frac1n\Big)\frac{f_s}{a^2} + \frac{2s}{n\,a}\,\partial_s\Big(\frac{f_s}{a}\Big).
```

To O(1/n), with μ still free:

```math
W = \Phi_s - \frac1c\Big(\frac{\Phi_t}{c}\Big)_t + \frac1n\Big[\Phi_s + 2s\Phi_{ss} + \frac{2s\,\Phi_sc_s}{c} - \mu\Phi_s - \frac{\mu_t\Phi_t}{2c^2}\Big],
```
```math
E = -c_s + \frac{\Phi_t^2}{c} + \frac1n\Big[-c_s - 2s\,c_{ss} + \mu\,c_s + \frac12\Big(\frac{\mu_t}{c}\Big)_t\Big].
```

The other rows give μ = 2s cₛ/c (χχ row) and μₜ = 4sΦₛΦₜ (ts row) at leading order. With them the two μ terms of W
cancel the lapse term, and

```math
W = W_0 + \frac1n\big(\Phi_s + 2s\,Q_W\big),\qquad E = E_0 + \frac1n\big(-c_s + 2s\,Q_E\big),
```
```math
Q_W = \Phi_{ss} - \frac{\Phi_s\Phi_t^2}{c^2},\qquad Q_E = -\,c_{ss} + \frac{c_s^2}{c} + \Big(\frac{\Phi_s\Phi_t}{c}\Big)_t .
```

Each O(1/n) part is the leading row's own s-derivative term, with no explicit s, plus 2s times a part Q that comes from
the radial second derivative (2s/n)∂ₛ² of the flat Laplacian and from ∂ₜμ.

**Against FIRST_ORDER.md §1.** With C = c/a, Y = 2Φₜ/C and S = 2Φₛ these are its Y-equation and C-equation to O(1/n):
FO_Y = −2c W and FO_C = −E on the leading shell (check C2). This is a third derivation of that system, from the
covariant equations in Cartesian-type coordinates. FIRST_ORDER.md's is from the solver's equations and L2_MODE.md §8.1's
is from the O(p) × O(q) frame.

### 1.3 The two forcings

Convention: for a leading-order mode D, the forcing is F = n 𝓛ₙ[D Zₙ], and the correction W obeys 𝓛₀[W] = −F + (the
term in the eigenvalue's shift). Rows as in §1.2.

**ℓ = 0.** The mode is ∂ₛ(c, Φ), the s-translation of the lapse and the scalar. Differentiating the equations in s,
𝓛ₙ[∂ₛZₙ] = −(1/n) ∂ₛ^expl(O(1/n) rows), so

```math
F_0 = -2\,(Q_W,\ Q_E).
```

**ℓ = 2.** F₂ = −n δ_k(W, E), the variation of the rows under γ → γ + k at fixed c and Φ. With g^{ij} = δ − βNN,
β = μ/(n + μ):
- δ ln√γ = ½ tr_γ k = −½ β k_NN = −(μ/2n) e Y₂, since k is traceless in δ. It depends on t through μ.
- δ(D²f) = −∇ᵢ(k^{ij}∂ⱼf) + γ^{ij}∂ᵢ(δ ln√γ)∂ⱼf. With ∂ᵢ(Hx)ᵢ = tr H = 0 and N·∂Y₂ = 0, the first term is
  −(e Y₂/n)[2s(fₛₛ + fₛ(ln w)ₛ) − μfₛ] for a divergence with weight w (w = c a for □, a for D²), and the second is
  O(1/n²).
- δ(∂ₜ part of □φ) = −(1/c²) ∂ₜ(δ ln√γ) Φₜ = (e Y₂/2nc²) μₜΦₜ, and δK = (e Y₂/2nc) μₜ.

So

```math
\delta_kW = -\frac{eY_2}{n}\Big[2s\Phi_{ss} + \frac{2s\Phi_sc_s}{c} - \mu\Phi_s - \frac{\mu_t\Phi_t}{2c^2}\Big],\qquad
\delta_kE = \frac{eY_2}{n}\Big[2s\,c_{ss} - \mu\,c_s - \frac12\Big(\frac{\mu_t}{c}\Big)_t\Big],
```

and with the rules for μ, per unit e,

```math
F_2 = 2\,s\,Y_2\,(Q_W,\ Q_E).
```

Its O(1) part vanishes because k is traceless. This holds for every traceless H, so for both representatives of
L2_MODE.md §4 at once. Reduced with s as the evolution variable, F₂[E]/c is the F[E^t_t] printed in L2_MODE.md §8.3,
term by term (check F).

### 1.4 The identity

```math
\frac{F_2}{s\,Y_2} + F_0 = 0\quad\text{in both rows},
```

for the ℓ = 0 mode ∂ₛ(c, Φ). **The reason.** A traceless k leaves the trace part of the Laplacian alone and multiplies
its radial part g^{rr}∂ᵣ² by 1 − k_NN. The mode is minus k, so its residual is +e Y₂ times the radial terms,
(2s/n)Q. The s-translation's residual is minus the explicit s-derivative of the same terms, −(2/n)Q. Dividing the first
by s Y₂ gives the opposite of the second.

FIRST_ORDER.md §4's ℓ = 0 mode is not ∂ₛ(c, Φ). It translates (C, Y, S), with C = c/a. Since a = (1 + μ/n)^½ and
μ = η carries an explicit s (∂ₛ^expl η = μ/s at leading order), its lapse part is cₛ − cμ/(2sn) = cₛ − cₛ/n, and its
scalar part is Φₛ. A change δ/n of the mode changes the forcing by 𝓛₀[δ]. So

```math
\frac{F_2}{s\,Y_2} + F_0^{\rm FO} = \mathcal L_0[(-c_s,\ 0)] = \mathcal L_0[(0,\ \Phi_s)],
```

the second equality because 𝓛₀ annihilates the translation mode (cₛ, Φₛ). In these rows F₀^FO is FIRST_ORDER.md §4's
forcing, −½(2Sₛ + ½Y²S) in W and −c(½S² − YYₛ) in E (check E2), which are the values L2_MODE.md §8.2 states.

So, before reading §9: **the identity holds, with the field (0, Φₛ)** (lapse part 0, scalar part Φₛ; equivalently
(−cₛ, 0)), for FIRST_ORDER.md's ℓ = 0 forcing, and with no extra field at all for the ℓ = 0 mode ∂ₛ(c, Φ). *Proved* by
hand above and *checked symbolically* (§1.6).

### 1.5 What the identity needs before it gives c₂ = −λ₁

Written before reading §9, as the points to compare.

1. **The same operator and the same λ.** On the lapse and the scalar the ℓ = 2 rows at O(1) are s Y₂ times the ℓ = 0
   rows (check G, with its control). Since s = ξ²e^{−2τ}/2, a perturbation e^{λτ} s Y₂ (γ, ψ) is the ℓ = 0 perturbation
   (γ, ψ) at λ + 2, for every λ. So the derivative in λ is the same in the two problems, and a forcing of opposite sign
   moves the two eigenvalues in opposite directions: c₂ = −λ₁ if nothing else differs.
2. **The local field drops out** if it has zero multiplier and the solvability functional annihilates the multiplier
   of every homogeneous solution. (0, Φₛ) is a derivative of the selected background, so its lateral sums agree if the
   background's do.
3. **A residual gauge family.** The slices t → t + e s Y₂ ε(t), with the compensating spatial diffeomorphism that keeps
   the shift zero, give L = s Y₂ (εc)ₜ, P = s Y₂ εΦₜ and V₁ + V₂ = 0 at leading order, for every ε(t). They solve the two
   rows (check H). Under the conjugation they are the time reparametrisations of the ℓ = 0 problem. So the ℓ = 2 problem
   in the gauge V₁ + V₂ = 0 has the same pure-gauge family as ℓ = 0 has, and the conjugated mode ∂ₛZ₀ has a lapse part
   cₛ that does not vanish at the centre, as in ℓ = 0. L2_MODE.md §8.4 item 4 says ℓ = 2 "needs no gauge step". My
   expectation is that this is not so: either a centre condition on the lapse part is imposed and the mode needs the
   same reparametrisation step as ℓ = 0, or none is imposed and the map from the data to the multiplier has the gauge
   family in its kernel. In the full problem the family has zero multiplier and the answer does not depend on the
   choice. In an M-harmonic model the reparametrisation is not a symmetry, so c₂(M) can depend on it.
4. **The truncated models are not singular at λ = 2.** FIRST_ORDER.md §4 says that in a truncated model 𝓢(2) is not
   singular and its symmetry-route number "is not a model quantity". The conjugated ℓ = 2 problem at λ = 0 is the
   ℓ = 0 problem at λ = 2. So I expect c₂(M) to need a convention, and the comparison "c₂(M) against λ₁(M)" to be a test
   of the identity only if both come from the same convention.
5. **The other rows.** The identity is on two rows. c₂ needs the remaining rows of the O(1/n) problem (momentum
   constraints, spatial rows, fibres) to be solvable for the metric part of the correction with no condition of their
   own.
6. **Selection.** That an ℓ = 2 perturbation is selected by a vanishing multiplier along rays, and that the outer
   region adds nothing at O(1/n), is the ℓ = 2 analogue of Conjecture 1 and of FIRST_ORDER.md §3. Neither is shown.

### 1.6 The symbolic checks

`python3 l2_c2_review_sym.py` from `solver/inner_pert/` (about 25 s, one process; sympy 1.14.0). It computes the
warped-product curvature over the base (t, s, u = tan(χ/2)) in first-order perturbation theory, exactly in n, expands
in 1/n, and reduces on the leading shell with s as the evolution variable. All pass.

| check | result | mutation, and the checks that then fail |
|---|---|---|
| A: Schwarzschild–Tangherlini is Ricci flat in the p = q frame, n = 3, 4, 7 | zero | `fibre` (one fibre dimension + 1): A, C, C2, D, D1, E, E3, F |
| B: a diffeomorphism's first-order rows are the Lie derivative of the zeroth-order rows, off shell, exact in n | zero difference | `lin` (the inverse metric's variation dropped from the connection): B, D, E, E3, F, G |
| B2: that diffeomorphism is the linear map: 𝓛_ξ(flat) = k, warps shift by ±e/2, Φ by s Y₂ Φₛ | equal | `k`: B2 |
| C0: μ = 2s cₛ/c and μₜ = 4sΦₛΦₜ from the χχ and ts rows | equal | |
| C: the spherical rows to O(1/n) are §1.2's hand-derived ones, μ free, off shell | zero difference | `bg` (Φₛ added to the hand W₁): C |
| C2: FIRST_ORDER.md §1's Y- and C-equations to O(1/n) | zero difference | `fibre` |
| D0: the ℓ = 2 rows on k vanish at O(1), off shell | zero | `k` (a trace part added to k): D0, D, E, E3, F |
| D, D1: F₂ = 2 s Y₂ Q, and the spherical O(1/n) rows are (Φₛ, −cₛ) + 2sQ | zero difference | `k`, `fibre`, `lin` |
| E0: F₀ = −2Q for the mode ∂ₛ(c, Φ), as a Gateaux derivative | zero difference | |
| E: F₂/(s Y₂) + F₀ = 0 | zero | `sign` (F₂/(s Y₂) − F₀): E, E3 |
| E1: 𝓛₀[(−cₛ, 0)] = 𝓛₀[(0, Φₛ)] | zero difference | |
| E2: F₀^FO is FIRST_ORDER.md §4's forcing in these rows | zero difference | |
| E3: F₂/(s Y₂) + F₀^FO = 𝓛₀[(0, Φₛ)] | zero difference | `sign` |
| F: F₂[E]/c is L2_MODE.md §8.3's printed F[E^t_t] | zero difference | `k`, `fibre`, `lin` |
| G, G1: conjugation by s Y₂ at O(1); control cos²χ does not conjugate | zero; nonzero | `lin` |
| H: ((εc)ₜ, εΦₜ) is in the kernel of the ℓ = 0 rows for every ε(t) | zero | |
| I: among metric perturbations of relative order 1/n only V₁ + V₂ reaches the two rows at O(1) | as stated | |

E0, E1, E2 and H have no mutation of their own. E0 and E2 are two routes to the same F₀ (a Gateaux derivative, and
FIRST_ORDER.md's printed forcing), so each tests the other.

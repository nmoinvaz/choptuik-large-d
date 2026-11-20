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

**Corrections to §1, made after reading §9.** They are kept apart so that §1 stays as it was committed (5112139e).

- §1.1's display is loose. L2_MODE.md §8.3's deformation, with its metric corrections A = (sμₛ + μ)Y₂ and
  B = −μ sin 2χ, **is** 𝓛_ξ(g, φ) − (k, 0), with no further terms (check B3). The corrections are the Lie derivative of
  the radial part (μ/n)dr². So the two frames hold the same mode, and their forcings must agree in every row.
- A metric perturbation of relative order 1/n does reach the two rows, at O(1/n). Check I says only that it does not
  reach them at O(1). So the forcing depends on the mode's metric part, and B3 is what makes §1.3's F₂ the forcing of
  L2_MODE.md's mode.
- §1.1's Y₂ is the harmonic x·Hx/r² normalised so that εh/(2n) = e s Y₂.

## 2. Comparison with L2_MODE.md §9 and the code

Findings are numbered through the whole note. Each is marked agreement, correction, gap or code.

### 2.1 The forcing and the identity

1. **Agreement: the identity.** §9.2's F̃⁽²⁾ + F̃⁽⁰⁾ = 𝓛₀[(0, Φₛ)], with the rows −4cₛΦₛ/c (W) and 2ΦₜΦₜₛ/c² (E^t_t), is
   §1.4's result. It is *proved* by hand in §1.3–1.4 and *checked symbolically* by two codes that share no source
   (`l2_c2_forcing.py`; `l2_c2_review_sym.py`, checks E–E3). The forcings stored in `data/l2_c2_forcing.json` (l2, l0
   and G, both rows) equal this note's, as rational functions of the t-derivatives (check K, added after reading).
2. **Agreement, and a shorter form.** §9.2 prints F̃⁽²⁾[W] as nine terms with Φ to fourth order in t. With
   s-derivatives it is 2Q. In FIRST_ORDER.md's heat variables (Y = 2Φₜ/C, S = 2Φₛ):

   | row | F̃⁽²⁾ | F̃⁽⁰⁾ (FIRST_ORDER.md §4) | sum = 𝓛₀[(0, Φₛ)] |
   |---|---|---|---|
   | W | Sₛ − ¼Y²S | −Sₛ − ¼Y²S | −½Y²S |
   | E^t_t | ½S² − ½YYₛ + ⅛Y⁴ | −½S² + YYₛ | ½YYₛ + ⅛Y⁴ |

   §9 gives no reason for the identity. §1.4 does: the mode is minus a constant traceless metric, which rescales
   the radial second derivatives by 1 + e Y₂, and the s-translation's residual is minus the explicit s-derivative of
   the same terms.
3. **Agreement: the extra field is a matter of the ℓ = 0 representative.** With the ℓ = 0 mode ∂ₛ(c, Φ) of the lapse,
   F̃⁽²⁾ + F̃⁽⁰⁾ = 0 and there is no G. G = 𝓛₀[(0, Φₛ)] = 𝓛₀[(−cₛ, 0)] appears because FIRST_ORDER.md translates
   C = α/a (§1.4). §9.2's control "(cₛ, 0) leaves twice the right-hand side" is the same statement with the sign.
4. **Agreement: the deformation is exactly a diffeomorphism minus a constant metric** (check B3). L2_MODE.md §6 gives
   this as "a reading of the result (not proved)", up to "terms in the O(1/n) part of the metric". At linear order in
   e it is exact at every n, including those terms. So §6's reading can be stated as *checked symbolically*.
5. **Agreement: the mode solves every row at leading order** (L2_MODE.md §4). In this note's frame that is the
   statement that every other row (the two momentum constraints, the three spatial rows of the base, the two fibre
   rows) vanishes on k at its natural order. It does (check J, p = q only).
6. **Agreement: every representative** (§9.6). §1.3 uses only that H is traceless and that k_NN = e Y₂. So F̃⁽²⁾ is
   the same for every ℓ = 2 harmonic, p = 1 included, by the hand derivation. §9.6 reaches it for p = 1 with a second
   symbolic run. The symbolic script of this review covers p = q only.
7. **Agreement: the conjugation and its control** (§8.3; checks G, G1), and the map to the solver's variables (§9.1):
   Fᵧ = √2ρ²c F̃[W] and F_c = ρ²c F̃[E^t_t], entering as L₂W_y = … + Fᵧ. The y-equation's residual is −√2ρ²c times the
   W row, and the correction obeys 𝓛₀W = −F, which gives that sign.

### 2.2 The gauge, and the truncated models

8. **Agreement: the residual gauge of ℓ = 2** (§9.1, by hand there). §1.5 item 3 derived the same family before
   reading: slices t → t + e s Y₂ ε(t) with the compensating displacement, acting on (L, P) as s Y₂((εc)ₜ, εΦₜ), with
   V₁ + V₂ = 0. §9.1's formula for ∂ₜδ(V₁ + V₂) is the one found here. Both derivations are by hand. Neither is
   checked symbolically beyond check H (the family is in the kernel of the two rows).
9. **Correction: L2_MODE.md §8.4 item 4 contradicts §9.1.** Item 4 says that ℓ = 2 "needs no gauge step" and that the
   route "should hold in the M-harmonic models too". §9.1 shows the opposite: the conjugated mode ∂ₛZ₀ has
   c(0) = 2g₁ ≠ 0, the reparametrisation that would remove it is not exact in a truncated model, and a slice has to be
   chosen. c₂(M) then depends on the slice (−9.09 to −5.95 at M = 1). §8.4 item 4 should point to §9.1.
10. **Correction: what the M table tests.** By linearity c₂(M) + λ₁(M) = G(M) exactly, for any forcing and slice
    (the note says so in §9.4). So the pairs "−7.96 against 7.81" are one number and G, not two measurements. The
    numerical content of §9.5 is:
    - G(M) → 0: −0.150, 0.022, −0.0016. This tests §9.4 item 4, that a local field drops out. It is a property of
      the ℓ = 0 machinery on the slice. Nothing in it is specific to ℓ = 2.
    - λ₁(M) on the slice → the fit: 7.81, 2.50, 3.044 against 3.0324.

    The ℓ = 2 content of the claim is the symbolic identity, the conjugation and the hypotheses of §3. The issue's
    summary and the Result bullet ("the truncated models agree") should say this.
11. **Correction: c₂ at M = 1 depends on how the forcing is written.** The compact form of finding 2 is the same
    function of the fields as §9.2's form in the full problem. In a truncated model the products associate
    differently. The second implementation (§4) gives at M = 1:

    | forcing | c₂ on φ = e₁, e₂, f₂ | G on e₁, e₂, f₂ |
    |---|---|---|
    | §9.2's form (t-derivatives) | −8.322, −5.948, −9.086 | −0.189, 0.068, −0.272 |
    | compact form (s-derivatives) | −2.860, −1.281, −3.368 | 0.075, 0.302, 0.002 |

    This is the same effect as §9.5's λ₁ = 7.81 against 2.73 for the two associations of the ℓ = 0 forcing. L2_MODE.md
    says that M = 1 is far from converged. The value −7.96 should not be quoted without that.
12. **Correction: the primary value depends on the matching radius.** §9.5's primary c₂ uses the left singular vector
    of Sch(2). When Sch is not singular that vector depends on the normalisation of the jump, and so on ξ₁. At M = 1
    the second implementation gives −7.9591 at ξ₁ = 0.2 and −7.9436 at ξ₁ = 0.25. The slice values c₂(φ) are the same
    at both radii to 10⁻¹⁰. FIRST_ORDER.md §4 makes the same remark about its symmetry route. At M = 5, where Sch is
    nearly singular, the effect is within the slice spread.

## 3. The solvability argument (§9.4), by hand

**The setting.** At λ = 2 a formal solution of the forced ℓ = 0 rows, 𝓛₀W = R, is a particular solution plus the
homogeneous solution with data (f, k): a profile f and a lapse datum k = δc(0). Write S[W] for the multiplier, the
difference of the two lateral sums, and S[hom(f, k)] = Af + Bk.

**The claim.** If V is a local field with S[V] = 0 and 𝓛₀V = G, the response W^G to G with zero data has
ℓ·S[W^G] = 0, for every ℓ with ℓA = 0 and ℓB = 0.

**The proof** is one line: W^G = V − hom(f_V, k_V), so S[W^G] = −A f_V − B k_V, which ℓ annihilates. So the claim is
right. It needs exactly:

- (a) **S[V] = 0.** V = (0, Φₛ) is a differential polynomial in the background. Its two lateral sums are the same
  polynomial in the background's lateral sums. So S[V] = 0 if the background's jump vanishes in every harmonic, which
  is the selection (Conjecture 1), and if the lateral sum of a forced linear problem is unique, which holds on a ray
  where every fast mode is exponentially large.
- (b) **ℓA = 0.** A(2) is singular: the gauge-stepped s-translation is a mode at λ = 2 with zero multiplier
  (Proposition 5 (iii)).
- (c) **ℓB = 0.** Every lapse datum k is the datum of a time reparametrisation ((εc)ₜ, εΦₜ), which is local, so
  A f_g + B k = 0 and ℓB = 0. For the ℓ = 2 problem the same family is the residual gauge of finding 8.
- (d) **ℓ is unique**, so one condition fixes c₂: the kernel of the reduced problem is one-dimensional once the gauge
  family is removed. And ℓ·S[W^λ] ≠ 0, so the eigenvalue is simple in λ.

I checked §9.4's four items against this.

- Items 1, 2 and 4 are (b), (c) and the proof above.
- Item 3 (the denominators are equal) holds. It needs the reparametrisation family for λ near 2, not only at 2:
  differentiating A(λ)f_g(λ) + B(λ)k_g(λ) = 0 in λ gives ℓ·(∂_λA f_g + ∂_λB k_g) = −ℓ·(A f_g′ + B k_g′) = 0. The
  family exists for every λ, with ε(t) of the matching weight, so this is fine. §9.4 does not say it.
- The sign: both problems are 𝓛₀W = −F + δλ F_λ with the same operator, the same mode ∂ₛZ₀ and the same F_λ, since
  e^{λτ} s Y₂ w is the ℓ = 0 perturbation w at λ + 2 (§1.5 item 1). So F̃⁽²⁾ = −F̃⁽⁰⁾ + G gives c₂ = −λ₁.

13. **Agreement: §9.4 is correct** under (a)–(d) and the hypotheses below.

**Are §9.7's hypotheses the complete set? No.** §9.7 lists the symmetry route with ℓ·B = 0, "§8.5's hypotheses: the
one-dimensional kernel and multipliers along rays", "Conjecture 1 and the lemma chain at O(1/n), as for λ₁ itself", and
the ansatz. Against (a)–(d) and the steps that lead to the reduced problem:

14. **Gap: the selection of an ℓ = 2 perturbation is its own hypothesis.** "As for λ₁ itself" is not right. Conjecture
    1 and the lemma chain of C_selection.md concern the spherical solution, and FIRST_ORDER.md §3 extends them to
    spherical perturbations. Nothing in the repository relates the regularity of a non-spherical perturbation at the
    light cone to its inner multipliers. That the condition is "the multiplier of w = W₂/(s Y₂) vanishes" is the
    ℓ = 2 analogue of Conjecture 1. It should be listed as such.
15. **Gap: the outer region at O(1/n) for ℓ = 2.** FIRST_ORDER.md §3 shows that the outer maps (matching factor,
    transport, light-cone constant) multiply a multiplier that already vanishes. For ℓ = 2 those maps are different
    ones. The conjugation by s Y₂ is a statement about the inner rows at leading order. In the outer region the
    centrifugal term ℓ(ℓ + n − 1)/r² is of the same order as the radial terms, and the ℓ = 2 problem there is not the
    ℓ = 0 problem. The argument needs the ℓ = 2 maps to be invertible with 1/n expansions. Not shown, not listed.
16. **Gap, small: the kernel hypothesis holds only modulo gauge, and only for the lapse–scalar block.** §8.5 assumes
    that the kernel of the n = ∞ ℓ = 2 problem at λ = 0 is spanned by D₂Z₀. In the ansatz of L2_MODE.md §1 it is
    not. K = k/n is in it: L = P = 0, A = Y₂, B = −sin 2χ, C = −Y₂, V₁ = −V₂ = ½, a constant traceless metric of
    relative order 1/n, on which every row vanishes at its natural order (checks D0 and J). K is the diffeomorphism
    ξ/n at leading order, and that diffeomorphism moves the lapse and the scalar by D₂Z₀/n at the next order. The
    ansatz leaves every time-independent spatial diffeomorphism of relative order 1/n free in the same way. None of
    this changes c₂: the two rows vanish on K through O(1/n), and they do not see the metric functions at O(1)
    (check I). The hypothesis should be stated for the reduced lapse–scalar problem, modulo the gauge family of
    finding 8.
17. **Gap: the metric part of the correction.** The spatial rows are evolution equations for the metric functions
    (check J2: the χχ row holds ∂ₜ²C at its natural order). So the correction W₂ has a gravitational part with its
    own data and its own regularity, forced by F's other rows and by W₂'s lapse and scalar. c₂ = −λ₁ is a necessary
    condition from the lapse–scalar block alone, which is triangular at leading order. That the metric part exists
    with the right regularity is part of "the mode continues to finite n", and is assumed. §9.7 should say so.
18. **Gap, small: the ansatz for the correction.** Check I and the gauge V₁ + V₂ = 0 cover corrections of the form of
    L2_MODE.md §1 (zero shift, metric of relative order 1/n). §9.7 lists the ansatz for the mode. It holds for W₂ too.

None of findings 14–18 changes the algebra. They change what "conditional" stands on. In the note's own words the
status is: the identity *checked symbolically* (and *proved* by hand here), c₂ = −λ₁ *conditional* on (i) Conjecture 1
and FIRST_ORDER.md's route for λ₁, (ii) the ℓ = 2 selection hypothesis of finding 14, (iii) the ℓ = 2 outer maps of
finding 15, (iv) the existence of the continuation, with its metric part (finding 17), (v) a simple eigenvalue, and
(vi) the ansatz at linear order in e. The value −3.0324 is *numerical evidence* through λ₁'s fit.

## 4. The numerical check: a second implementation of c₂

`solver/inner_pert/l2_c2_review_num.py`. It imports nothing from `first_order_*.py`, `l2_c2*.py` or `inner_np.py`.
§6 lists what it shares with them. In short: the model and the conventions are shared, because a truncated number is
defined only with them, and no code is.

**Its own planted checks** (`l2_c2_review_num.py checks`, 10 s; exit 0 iff all pass):

| check | result | bar |
|---|---|---|
| 1. the M = 1 zero by its own Newton on the raw jump, from five digits, against the stored zero | difference 2.8×10⁻¹¹ | 10⁻⁸ |
| 2. the linearised code against the nonlinear one: profile sin θ at λ = 0 is d(jump)/dA, five-point differences | 1.3×10⁻⁷ | 10⁻⁶ |
| 3. exact modes: A(0)F′ = 0 and A(1)(F + ωF′) = 0; control: A(0.5) is not singular | 6.4×10⁻¹¹, 1.2×10⁻¹⁰; control σ_min/σ_max 2.0×10⁻² | 10⁻⁶; > 10⁻³ |
| 4. λ* of the M = 1 model, secant on det A(λ) | 1.7691220085, against 1.7691220084 | 10⁻⁷ |
| 5. d/dλ of the mode by the variational equation against centred differences in λ | 4.6×10⁻⁸ | 10⁻⁶ |
| 6. each forcing two ways at \|ξ\| = 0.12: summed series, and Taylor-jet value at the point | 1.1×10⁻¹³ | 10⁻⁸ |

Checks 1 and 4 are also results: the stored zero and FIRST_ORDER.md's λ* = 1.769122008 are reproduced by code that
shares only the model with the code that made them.

**c₂ at M = 1** (`l2_c2_review_num.py c2`, 2 min), at the stored zero, on the ray arg ξ² = π/2 + π/5, ξ₁ = 0.2,
against `data/l2_c2_M1_xi0.2_l2-l0-G-sym.json`:

| forcing | φ = e₁ | e₂ | f₂ | left singular vector | max relative difference from l2_c2.py |
|---|---|---|---|---|---|
| l2 (c₂) | −8.32247565 | −5.94806674 | −9.08611454 | −7.95908853 | 2.2×10⁻¹¹ |
| l0 (λ₁) | 8.13341780 | 6.01651566 | 8.81423936 | 7.80944033 | 2.2×10⁻¹¹ |
| G | −0.18905786 | 0.06844892 | −0.27187518 | −0.14964820 | 1.3×10⁻¹⁰ |
| sym (λ₁, SymForce's terms) | 2.93575322 | 1.58371484 | 3.37058527 | 2.72883295 | 4.8×10⁻¹¹ |
| l2c (c₂, compact form) | −2.86036018 | −1.28133939 | −3.36819254 | −2.61870180 | not in l2_c2.py |
| Gc (G, compact form) | 0.07539304 | 0.30237545 | 0.00239273 | 0.11013115 | not in l2_c2.py |

Also equal: the mode's profile f₂ = (−41.979377, 22.648836) and 2g₁, Sch(2) after de-mixing (3.3×10⁻¹⁰), and the
mode's jump over |A||f₂|, 5.9×10⁻¹¹ (§9.1 gives 5×10⁻¹¹).

Variations, each a separate run:

| run | result | verdict |
|---|---|---|
| `--mutate=fc` (the sign of F_c flipped in the converted forcings) | c₂ = −3.87, −0.71, −4.89; relative differences 0.46 to 1.2; exit 1 | fails, as required |
| ray π/2 + π/8 and ξ₁ = 0.25 | slice values within 2.6×10⁻¹⁰ of ξ₁ = 0.2's; the left-singular-vector values move: c₂ −7.94356, λ₁ 7.79560 | slices do not depend on the path or ξ₁ (finding 12) |
| own zero (check 1's) and ray π/2 + 0.9 | within 4.4×10⁻⁹ | OK (bar 10⁻⁶) |

**c₂ at M = 3** (`--M=3`, 7 min), at the stored zero. The raw 4 × 4 jump matrix is solved in one shot, with rows and
columns scaled (condition number 1.5×10⁹), where l2_c2.py deflates and takes a Schur complement:

| forcing | φ = e₁ | e₂ | f₂ | max relative difference from l2_c2.py |
|---|---|---|---|---|
| l2 (c₂) | −2.47017214 | −2.52151634 | −2.45047292 | 2.1×10⁻⁹ |
| l0 (λ₁) | 2.49365023 | 2.53725690 | 2.47691966 | 2.1×10⁻⁹ |
| G | 0.02347808 | 0.01574056 | 0.02644674 | 4.7×10⁻⁸ |
| sym | 2.87058941 | 2.91747121 | 2.85260228 | 1.6×10⁻⁹ |
| l2c (c₂, compact form) | −2.88001265 | −2.93466923 | −2.85904257 | not in l2_c2.py |
| Gc (G, compact form) | −0.00942324 | −0.01719802 | −0.00644029 | not in l2_c2.py |

19. **Agreement: l2_c2.py's numbers at M = 1 and M = 3 are reproduced**, to 1.3×10⁻¹⁰ and 4.7×10⁻⁸ (bar 10⁻⁶). That
    covers the expression graph, the conversion to the solver's variables, the forced series, the d/dλ component, the
    slice formula and, at M = 3, the deflation. Status: *numerical evidence* that the code does what §9.1–9.3 say.
    It is not evidence for c₂ = −λ₁ beyond finding 10.
20. **The compact form converges too, from the other side.** With the forcing in s-derivatives, c₂ is −2.86, −1.28,
    −3.37 at M = 1 and −2.880, −2.935, −2.859 at M = 3, and its G is 0.075, 0.302, 0.002 and then −0.009, −0.017,
    −0.006. §9.2's form gives G = −0.19, 0.07, −0.27 and then 0.023, 0.016, 0.026. The two forms of G have opposite
    signs at M = 3 and both fall. That is what a truncation residue of an exact zero looks like. Status: *numerical
    evidence*, two points per form.

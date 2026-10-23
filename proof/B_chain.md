# Theorem B(1): closing the chain (P1)

This note is the work of item P1 of [notes/PROOF_QUEUE.md](../notes/PROOF_QUEUE.md): the links of
[C_selection.md](C_selection.md) §13.1 that were missing for Theorem B(1), the selection theorem for the one-harmonic
model. It was written by Claude (Anthropic) under Nathan Moinvaziri's direction and has not been checked by a
mathematician. Status words as in [AGENTS.md](../AGENTS.md): *proved* (by hand), *checked symbolically*, *certified*
(computer-assisted, trusted code named), *conditional* (hypothesis named), *numerical evidence*. Scripts are in
`solver/proof/`.

Contents

- [0. Summary](#0-summary)
- [1. The matching radius can be held fixed](#1-the-matching-radius-can-be-held-fixed)
- [2. Lemma L: labels of fast-type solutions on the Stokes line](#2-lemma-l-labels-of-fast-type-solutions-on-the-stokes-line)
- [3. Lemma B5(2) with constants](#3-lemma-b52-with-constants)

## 0. Summary

*(Written as the work proceeds; see the end of the note for the final status.)*

## 1. The matching radius can be held fixed

C_selection.md §10.2 and B5_inner.md put the outer edge of the matching domain at $`\lvert u\rvert = \kappa_1/\ln n`$ with
κ₁ fixed, and B5(2) evaluates the jump at $`u_* = \kappa_1/\ln n`$. That choice makes the evaluation point move toward
the inner arc as n grows, and it is the only reason for the factor $`n^{1/(2\kappa_1)}(\ln n)^{\dots}`$ in B5(2)'s error
and for the rate $`O(n^{-1+1/(2\kappa_1)})`$ of §13.1 item 5. The certificates allow a fixed point instead.

**Lemma D′ (proved).** Let Z_B := R′_{n₀}²/n₀ for the certified run of B8 (κ = 0.3, κ₁ = 0.6, ln n₀ = 73.89, so
Z_B = κ₁/(cos κ ln n₀) = 0.0085). Then for every n ≥ n₀ the conclusions of Lemma B8(1) (B8_constants.md §6) hold, with
the same constants h, on the domain

```math
\mathcal D'_n = \{x : \lvert x\rvert^2 \le Z_B\,n,\ 0 \le \arg x \le \pi/4 + \kappa/2\} \setminus \{\lvert x - 1\rvert \lt \delta/2\},
```

and (H8) of B5(1) holds for every n ≥ n₀ with the same δ.

*Proof.* `b8m_bounds.py` uses the outer radius only through ρ_max = (1 + R′)²/n, the upper bound of ρ = (1 + |x|)²/n
on the far region F; every other quantity is a sum of terms c X^a θ^b n^e with e ≤ 0 on region boxes that do not depend
on n, evaluated at n = n₀ (B8_constants.md Lemma D). On 𝒟′_n, ρ ≤ (1 + (Z_B n)^{1/2})²/n, which decreases in n, so
ρ ≤ ρ_max(n₀) on F for every n ≥ n₀. The kernel constants of Lemma K use the domain only through the same bound:
N_F = (1 − 1/n₀)/(ρ_max q_F) is a lower bound of (n − 1)r²/Q_r for r ≥ x_F and any r with (1 + r)²/n ≤ ρ_max, and
|D_r| ≤ r² q_F^{1/2} ≤ n ρ_max q_F^{1/2}; on the region N the cell bounds use n₀ − 1 ≤ n − 1 in the direction that
makes them valid for n ≥ n₀. The (H8) bounds use B8's h and the inner arc |x| = n^γ only. ∎

**Consequence.** B5(1) (certified for the working domain |z| ≤ Z = 0.0085, B5_inner.md §5), B8 and (H8) together give,
for every n ≥ n₀ = e^{73.89}, the bound $`\lvert W_j(z)\rvert \le x_j m_j(z)/n`$ on the matching domain with the *fixed*
outer edge

```math
\mathcal M^+_n = \{z : r_n \le \lvert z\rvert \le Z\cos\kappa,\ 0 \le \arg z \le \pi/2 + \kappa\},\qquad Z\cos\kappa = 0.008120 .
```

The working domain 𝒲⁺_n of B5_inner.md §3 built from this 𝓜⁺_n lies in |z| ≤ Z (B5_inner.md Lemma 4.1), the ε-parts of
B5(1) are monotone in n, and B8 delivers analyticity on 𝒟′_n ⊃ 𝒲⁺_n by Lemma D′. The jump is then evaluated at the
fixed point u* = 0.0081 of the Stokes line, for every n ≥ n₀.

*Confirmation runs* (not needed for the proof): `b8m_scan.py` at κ₁ = 0.85, ln n₀ = 104.68 (the run the queue asked
for; ρ_max is the same 0.0085) and at the fixed-radius parametrization κ₁ = Z cos κ ln n₀ with ln n₀ = 90
(`b8m_scan_K4_k0.85.log`, `b8m_scan_K4_fixedZ.log`); see §3.3.

## 2. Lemma L: labels of fast-type solutions on the Stokes line

*Script: `solver/proof/b5_label.py` (certificate, `b5_label.json`, `b5_label.log`).*

### 2.1 Setting

On the Stokes line z = ξ² > 0 the inner system at ε = 0, in the state (P, Φ, C) of B5_inner.md §1, reads

```math
P' = \frac{\Phi}{2z},\qquad \Phi' = \frac{\Phi}{2z^2} + G^0(z, P, \Phi, C),\qquad C' = \tfrac12[C\hat E_0],\qquad \hat E_0 = \tfrac12[PP],
```

with z G⁰ analytic at z = 0 (`b5_formulas.zG0`). For p in the complex parameter box 𝒦 of B5(1) (§5.1 there), let
Y⁺ be the inner lateral sum (lemma S over 𝒦, as rerun by `b5_bounds.py`) and Y⁻(z; p) = σ(Y⁺(z; p̄)) the lower one
(σ: μ ↦ −μ and complex conjugation). The inner jump ψ = Y⁺ − Y⁻ solves exactly the **secant equation**

```math
\psi_P' = \frac{\psi_\Phi}{2z},\qquad \psi_\Phi' = \frac{\psi_\Phi}{2z^2} + \frac{J_P\psi_P + J_F\psi_\Phi + J_C\psi_C}{z},\qquad \psi_C' = K_P\psi_P + K_C\psi_C,
```

where $`J = \int_0^1 D(zG^0)(Y^- + s\psi)\,ds`$ and K the same for the C-row. These are fixed functions of z. Lemma S's
enclosure of the inner solution contains Y⁺, Y⁻ and the segment between them on the real axis, because its η-ball and
the formal series are σ-symmetric (the weights are symmetrized, lemma6_sector.py). So the Jacobian blocks of
`b5_formulas.jac_blocks`, evaluated on that enclosure, enclose J and K.

**The transport generator.** At z = 0, $`J_F(0) = \mathcal G := [F^2\,\cdot\,] - \tfrac32 - s`$ (s = iμω), a 2 × 2 matrix with
eigenvalues ⟨F²⟩ − 3/2 ± i(ω² − A⁴/16)^{1/2} = 1.7313 ± 2.4918i at the zero of Theorem A(M = 1). Write
$`\mathcal G = V\Lambda V^{-1}`$ (columns of V of unit length; V is analytic in p on 𝒦). The fast modes of the jump are
$`e^{-1/(2z)}z^{\mathcal G}`$ in the Φ-component: with $`\mathcal G`$'s real part ⟨F²⟩ − 3/2, not ⟨F²⟩ as
B5_inner.md §7 and `b5_threshold.py` assumed (see §3.2).

**The label map.** For a state Y at z on the Stokes line, $`L_z(Y) := e^{1/(2z)}z^{-\mathcal G}Y_\Phi \in \mathbb C^2`$.

### 2.2 Statement

**Lemma L (M = 1; certified).** Let Z_L = 0.0081. For every p ∈ 𝒦:

1. *(Fast-type solutions.)* For every k ∈ ℂ² the secant equation has a solution Ψ(z)k on (0, Z_L] with
   $`L_z(\Psi k) = (I + R(z))k`$, $`R(z) = V\tilde R(z)V^{-1}`$ and

```math
\lVert\tilde R(z)\rVert_\infty \le \varepsilon_L(z) := \exp\Big(\int_0^z g\Big) - 1,\qquad \varepsilon_L(Z_L) \le 0.3527 .
```

   Ψ(z)k is O(e^{−1/(2z)}) in every component as z → 0, and k ↦ Ψk is injective.
2. *(Identification.)* The inner jump is one of them: ψ = Ψ k(p) for a unique k(p) ∈ ℂ², and
   $`k(p) = \lim_{z\to0}L_z(\psi)`$. This k(p) is the Stokes multiplier vector; it is holomorphic in p on 𝒦. For real p,
   σψ = −ψ gives $`k_{-1} = -\overline{k_{+1}}`$, and the one-harmonic selection condition y⁺ ≡ y⁻ of Theorem A(M = 1) is
   k₊₁(p) = 0.

### 2.3 Proof

*Part 1.* Put $`\psi_\Phi = e^{-1/(2z)}Vz^\Lambda\tilde a(z)`$ with ã continuous on (0, Z_L], ã(0⁺) = V⁻¹k, and define
the slow components by integration from 0:

```math
\psi_P(z) = V\tilde p(z),\quad \tilde p(z) = \int_0^z e^{-1/(2t)}t^{\Lambda}\tilde a(t)\,\frac{dt}{2t},\qquad
\psi_C(z) = \int_0^z U_C(z, t)K_P(t)\psi_P(t)\,dt ,
```

with U_C the propagator of $`\psi_C' = K_C\psi_C`$. Then ψ solves the secant equation if and only if

```math
\tilde a' = z^{-\Lambda}\,V^{-1}\frac{J_F - \mathcal G}{z}V\,z^{\Lambda}\,\tilde a + z^{-\Lambda}V^{-1}\,e^{1/(2z)}\,\frac{J_P\psi_P + J_C\psi_C}{z} .
```

Bounds, with |·| the ∞-norm and A(z) = sup over (0, z] of |ã|:
- $`(J_F - \mathcal G)/z`$ is analytic (`b5_label.jf_div`, the identity below), and the entries of
  $`z^{-\Lambda}Mz^{\Lambda}`$ are $`M_{ij}z^{\lambda_j - \lambda_i}`$, of modulus ≤ $`\lvert M_{ij}\rvert z^{-\delta}`$, δ := max |Re(λ₁ − λ₂)|
  over 𝒦 (δ ≤ 4.1×10⁻⁵). The first term is at most $`\gamma_F z^{-\delta}A`$.
- $`\int_0^z e^{-1/(2t)}t^q\,dt \le 2z^{q+2}e^{-1/(2z)}`$ for q + 2 ≥ 0 (substitute u = 1/t). So
  $`\lvert\tilde p_j(z)\rvert \le A\,e^{-1/(2z)}z^{\mathrm{Re}\lambda_j+1}`$, and the P-term is at most $`\gamma_P z^{-\delta}A`$.
- $`\lvert\psi_C(z)\rvert \le 2e^{\kappa_C z}\kappa_P A\,z^{\mu_+ + 3}e^{-1/(2z)}`$, and the C-term is at most
  $`2\gamma_C\kappa_P e^{\kappa_C z}z^{2}A`$.

Here γ_F, γ_P, γ_C are suprema of $`\lVert V^{-1}\tfrac{J_F - \mathcal G}{z}V\rVert`$, $`\lVert V^{-1}J_PV\rVert`$,
$`\lVert V^{-1}J_C\rVert`$, and κ_P, κ_C of $`\lVert K_PV\rVert`$, $`\lVert K_C\rVert`$ (∞-norms), over 60 cells of
(0, Z_L] and over 𝒦. So $`\lvert\tilde a'(t)\rvert \le g(t)A(t)`$ with
$`g(t) = t^{-\delta}(\gamma_F + \gamma_P) + 2\gamma_C\kappa_P e^{\kappa_C t}t^2`$. The map ã ↦ V⁻¹k + ∫₀^z (right-hand side) is a
Volterra operator with an integrable kernel bound, so it has a unique fixed point in the bounded continuous functions,
and Grönwall's inequality gives $`A \le \lvert\tilde a(0)\rvert e^{\int g}`$ and $`\lvert\tilde a(z) - \tilde a(0)\rvert \le \lvert\tilde a(0)\rvert(e^{\int_0^z g} - 1)`$.
Finally $`L_z(\Psi k) = V\tilde a(z)`$. Each component of Ψk carries the factor e^{−1/(2z)}, and ã(0⁺) = V⁻¹k gives
injectivity.

The identity used for $`(J_F - \mathcal G)/z`$: with Ct = (C − 1)/z, Ct₂ = (Ct − Ct(0))/z and dQ = 𝒞⁻¹dΦ,

```math
\frac{(J_F - \mathcal G)\,d\Phi}{z} = \tfrac12[Ct[Ct\,d\Phi]] + [Ct_2\,d\Phi] - \tfrac12\big([Ct\,D_2\,dQ] - D_2[Ct\,dQ]\big) + \tfrac12[C\hat E_0\,dQ] ,
```

which follows from jac_blocks' Φ-block, $`\mathcal C - I = z\,[Ct\,\cdot]`$ and $`\mathcal G = [Ct(0)\,\cdot] - (D_2 + D)/2`$.

*Part 2.* Lemma 6a's flat estimate (A_certify.md §11.5, step 1; `lemma6_sector.flat_estimate`, unchanged), run on
lemma S over 𝒦 at Z_f = 0.004489, gives ‖T₀‖ ≤ 0.1794 < 1 in the weighted norm sup z^{−k_j}|·| (k = 12). Every
solution χ of the secant equation on (0, Z_f] with finite weighted norm satisfies χ = b(β) + T₀χ, where
β = v_χ(Z_f) ∈ ℂ², so χ = (I − T₀)⁻¹b(β): this space has dimension at most 2. It contains every Ψk (exponentially
small), and k ↦ Ψk is injective, so it is exactly {Ψk}. The jump ψ = η⁺ − σ(η⁺(p̄)) lies in it (lemma S's ball),
so ψ = Ψk on (0, Z_f], and on (0, Z_L] by uniqueness for the linear equation. The limit formula follows from part 1;
holomorphy in p from that of ψ and of 𝒢. For real p, σψ = −ψ and σ commutes with $`z^{\mathcal G}`$ (𝒢 is
σ-symmetric for real p), so $`k_{-1} = -\overline{k_{+1}}`$. y⁺ ≡ y⁻ ⇔ ψ ≡ 0 ⇔ k = 0 ⇔ k₊₁ = 0. ∎

### 2.4 Numbers and checks

`b5_label.py` (python-flint, 192 bits; lemma S at 320; 1 s): lemma S contraction 0.9519; flat estimate 0.1794;
‖V‖ ≤ 1.243, ‖V⁻¹‖ ≤ 1.481; sups γ_F ≤ 21.33, γ_P ≤ 16.21, γ_C ≤ 42.56, κ_P ≤ 2.544, κ_C ≤ 6.463;
$`\int_0^{Z_L}g \le 0.3021`$, ε_L(Z_L) ≤ 0.3527.

| check | result | mutation → result |
|---|---|---|
| identity for (J_F − 𝒢)/z vs jac_blocks at random complex states (mpmath, 50 digits) | 9.9×10⁻⁴⁹ | [Ct₂ ·] term dropped → 0.72, caught |
| certified ε_L(z) vs the label variation of a double-precision integration of the linearized inner system (from z₀ = 10⁻⁴, not rigorous) | 0.054 ≤ 0.353 at z = 0.0081 (every z checked) | every coupling enclosure × 0.01 → bound 0.0035 < 0.054, caught |

The enclosures of the inner solution and of the Jacobian blocks on cells are those of `b5_bounds.py`, whose point
containment checks (`b5_bounds.py check`) cover them.

## 3. Lemma B5(2) with constants

*Script: `solver/proof/b5_jump.py` (`b5_jump.json`, `b5_jump.log`).*

### 3.1 Statement

Let Ψ_out = Y⁺_out − Y⁻_out be the outer jump on the Stokes line, in the inner variables (P, Φ, C, ϑ) of B5_inner.md
§1, Y⁻_out(z; p) = σ(Y⁺_out(z; p̄)), and u* = 0.0081.

**Lemma B5(2) (M = 1; certified, conditional only on what B5(1), B8 and (H8) assume).** For every n ≥ n₀ = e^{73.89}
and every p ∈ 𝒦,

```math
V^{-1}L_{u_*}(\Psi_{\rm out}) = (I + \tilde R)\,V^{-1}k(p) + \tilde E,\qquad \lVert\tilde R\rVert_\infty \le 0.3527,\qquad \lvert\tilde E\rvert_\infty \le \frac{C_E}{n},\quad C_E = e^{76.914} .
```

R̃ and Ẽ are holomorphic in p on 𝒦. For real p with |p − x̂| ≤ 10⁻⁶ (componentwise), Cauchy's estimate on the polydisc
of radius 10⁻⁶ gives $`\lvert\partial_j\tilde E\rvert \le 10^6C_E/n`$ and $`\lVert\partial_j\tilde R\rVert \le 10^6\cdot0.3527`$.

In the form of C_selection.md §10.3: the label of the outer jump at u* is $`S + R_n`$ with S = k(p) the inner multiplier,
$`R_n = R(u_*)S + E`$; the first part is multiplicative (it vanishes with S and changes the Jacobian of S by at most a
factor 1 ± 0.353 in the eigen-coordinates), the second additive and O(1/n).

### 3.2 Proof

$`L_{u_*}(\Psi_{\rm out}) = L_{u_*}(\psi_{\rm in}) + L_{u_*}(W^+ - W^-)`$, with W^± = Y^±_out − Z^±_in, and
$`L_{u_*}(\psi_{\rm in}) = (I + R(u_*))k`$ by lemma L. B5(1) (with §1) bounds $`\lvert W^+_{\Phi,\mu}(u_*)\rvert \le x_{\Phi,\mu}u_*/n`$
on the Stokes segment of 𝒲⁺_n, and by reflection $`\lvert W^-_{\Phi,\mu}\rvert \le x_{\Phi,-\mu}u_*/n`$. In eigen-coordinates
$`V^{-1}L_{u_*}(Y) = e^{1/(2u_*)}u_*^{-\Lambda}V^{-1}Y_\Phi`$ and $`\lvert u_*^{-\lambda_i}\rvert \le u_*^{-\mu_-}`$, μ₋ = min Re λ. So

```math
\lvert\tilde E\rvert_\infty \le \frac{e^{1/(2u_*)}\,u_*^{1-\mu_-}\,\lVert V^{-1}\rVert_\infty\,(x_{\Phi,-1} + x_{\Phi,+1})}{n} .
```

x_Φ is the B5(1) super-solution for the (H8) data certified by B8, δ = (8.2×10⁻⁵, 79.03, 3.231, 6.463), by
superposition of the certified super-solutions (B5_inner.md §5.2): x_Φ = (39784, 38776) for μ = (−1, +1). With
e^{1/(2u*)} = e^{61.728}, $`u_*^{1-\mu_-}`$ = 33.84 and ‖V⁻¹‖ ≤ 1.481, C_E = 2.53×10³³ = e^{76.914}. ∎

**Correction to B5_inner.md §7 and `b5_threshold.py`.** Their model of the relative error,
$`x_\Phi n^{-1+1/(2\kappa_1)}(\ln n/\kappa_1)^{\langle F^2\rangle}`$, has two inaccuracies:
- the fast mode of the Φ-component is $`e^{-1/(2z)}z^{\mathcal G}`$ with Re 𝒢 = ⟨F²⟩ − 3/2 = 1.73, not ⟨F²⟩ = 3.23;
- the bound on W_Φ carries the weight m_Φ = |z|, a further factor u*.

Together these make the true error smaller by about $`u_*^{5/2}`$, i.e. 10⁻⁵·⁰ at u* = 0.008. The estimate's margin 10⁻⁷
was also not derived from the implicit function theorem; §8 replaces it.

### 3.3 Numbers

| ln n | 73.89 | 80 | 85 | 90 | 95 | 100 | 105 |
|---|---|---|---|---|---|---|---|
| bound on \|Ẽ\| | 21 | 4.6×10⁻² | 3.1×10⁻⁴ | 2.1×10⁻⁶ | 1.4×10⁻⁸ | 9.4×10⁻¹¹ | 6.3×10⁻¹³ |

Check: the superposition of the certified super-solutions reproduces the super-solution certified directly for the
reference data to 3.1×10⁻³ (dropping the forcing term gives 0.30, `--mutate=sum`, caught).

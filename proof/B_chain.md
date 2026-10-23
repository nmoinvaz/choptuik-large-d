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
- [4. B8's normalization](#4-b8s-normalization)
- [5. The 1/n rate](#5-the-1n-rate)
- [6. The light cone: what Theorem B(1) needs there](#6-the-light-cone-what-theorem-b1-needs-there)
- [7. The outer jump is determined by its label](#7-the-outer-jump-is-determined-by-its-label)
- [8. Lemma P, B7 and Theorem B(1)](#8-lemma-p-b7-and-theorem-b1)
- [9. Discrepancies with the existing notes](#9-discrepancies-with-the-existing-notes)
- [10. What the independent review (P2) should check](#10-what-the-independent-review-p2-should-check)
- [11. Reproduce](#11-reproduce)
- [12. P1L: plan for the light-cone hypotheses](#12-p1l-plan-for-the-light-cone-hypotheses)
- [13. Lemma LC: the solution up to the light cone in the light-cone gauge (certified)](#13-lemma-lc-the-solution-up-to-the-light-cone-in-the-light-cone-gauge-certified)
- [14. Lemma J: the outer jump is determined by its label (certified)](#14-lemma-j-the-outer-jump-is-determined-by-its-label-certified)
- [15. Theorem B(1) (certified, assuming Braaksma's theorem)](#15-theorem-b1-certified-assuming-braaksmas-theorem)

## 0. Summary

**Theorem B(1) is certified for M = 1, assuming Braaksma's theorem (§15).** For every n ≥ n₀ = e^{89.75}, exactly
one parameter in Theorem A's box gives a light-cone-gauge solution that is analytic at the light cone, within
e^{67.84}/n of Theorem A's zero. None of the certificate code has had an independent review yet (§10).

The certified links (python-flint; trusted code named in each section):
- *B5(2)* with constants (§§2–3). The label of the outer jump at the fixed point u* = 0.0081 of the Stokes line is
  $`(I + R)S + E`$, S the inner multiplier, with ‖R‖ ≤ 0.353 and |E| ≤ e^{76.94}/n, for every n ≥ e^{73.89} and every
  normalization |c(0) − c^K(0)|₁ ≤ 10⁷n⁻⁵. The new ingredient is lemma L, on fast-type solutions of the inner secant
  equation and their labels.
- *B8's normalization* (§4). B8 and (H8) hold for |c(0) − c^K(0)|₁ ≤ 10⁷n⁻⁵.
- *The 1/n rate* (§5) follows from the fixed evaluation point; S^{(1)} is not needed.
- *Lemma P and B7* (§8). For every n ≥ e^{89.75}, exactly one parameter in Theorem A's box makes the label of the outer
  jump vanish, within e^{67.84}/n of Theorem A's zero.
- *Lemma LC* (§13; P1L (a) and (b)). B8 is continued into the light-cone disc in the exact gauge c(1) = 1, as a
  contraction (θ = 0.74) with constants per band. It gives the light-cone-gauge solution for n ≥ e^{73.89}: unique
  near the formal series, holomorphic in the parameters, C¹ up to x = 1, with gauge offset
  |c(0) − c^K(0)|₁ ≤ 3.59×10⁶n⁻⁵. So (H-LC) and (H-G) hold.
- *Lemma J* (§14; P1L (c)). The jump vanishes if its label does: a contraction (θ = 0.18) in a norm weighted by half
  the fast exponent. So (H-J) holds.

The plan of P1L is §12. Discrepancies with the earlier notes are listed in §9, and the review list for P2 in §10.

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

This is for B8's gauge c(0) = c^K(0). For every normalization c(0) = c^K(0) + d with $`\lvert d\rvert_1 \le 10^7n^{-5}`$
(§4) the same holds with C_E = e^{76.940} (`b5_jump_d0_1e7.json`); for |d|₁ ≤ 10⁶n⁻⁵, C_E = e^{76.916}.

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

## 4. B8's normalization

*Scripts: `b8m_bounds.py --d0=...` (new option), `b8m_scan.py`, `b8m_h8.py`; logs `b8m_scan_K4_d0.log`,
`b8m_h8_K4_L73.89_d0_1e6.log`, `b8m_h8_K4_L73.89_d0_1e7.log`.*

B8 was certified in the gauge c(0) = c^K(0). B8_constants.md §8.4 proves by hand that the bootstrap is unchanged for
c(0) = c^K(0) + d with $`\lvert d\rvert_1 \le d_0n^{-K-1}`$, except that R_V(0) = [d·2F] and R_c(0) = d enter the slow
integrals: P_V and P_c increase by $`\lvert 2F\rvert_1d_0 \le 4\max_\mu\lvert F_\mu\rvert\,d_0`$ and d₀, since
ρ^{K+1} ≥ n^{−K−1} everywhere. `b8m_bounds.py --d0` adds exactly these two terms.

**Result (certified; K = 4, κ = 0.3, κ₁ = 0.6, ln n₀ = 73.89, B5(1)'s box):**

| d₀ | closes | h = (h_V, h_S, h_β, h_c) | P(h)/h | (H8): δ = (δ_P, δ_Φ, δ_C, δ_ϑ) |
|---|---|---|---|---|
| 0 (certified before) | yes | (2.41×10⁶, 6.27×10⁶, 6.17×10⁵, 1.67×10⁵) | ≤ 0.990 | (8.2×10⁻⁵, 79.03, 3.231, 6.463) |
| 10⁵ | yes | (3.00×10⁶, 6.58×10⁶, 6.72×10⁵, 2.80×10⁵) | ≤ 0.983 | not run |
| 10⁶ | yes | (1.07×10⁷, 1.45×10⁷, 1.45×10⁶, 1.37×10⁶) | ≤ 0.988 | (8.2×10⁻⁵, 82.18, 3.231, 6.463) |
| 10⁷ | yes | (9.71×10⁷, 1.20×10⁸, 1.13×10⁷, 1.28×10⁷) | ≤ 0.989 | (8.2×10⁻⁵, 122.31, 3.231, 6.463) |
| 10⁸ | no (c lower bound lost) | | | |

So B8(1) and (H8) hold for every normalization $`\lvert c(0) - c^K(0)\rvert_1 \le 10^7n^{-5}`$, n ≥ e^{73.89}, with δ ≤ δ_ref.

**What this does and does not settle.** The run is what §13.1 item 4 asked for. It does not show that the exact
light-cone gauge c(1) = 1 lies in that range, and nothing in B8_constants.md or C_selection.md shows it either: c(1) is
the value at the light cone, inside the disc |x − 1| < δ/2 that B8 excludes, so it needs control of the solution there
(§6). As a formal estimate (numerical evidence, `b8m_formal.formal`): the gauge that makes c(1) = 1 order by order
differs from c^K(0) by $`\sum_{k\ge5}\varepsilon^kc_k(0)`$ with $`\lvert c_5(0)\rvert_1 = 1.227\times10^5`$, and the
next coefficients grow by about 12–13 per order ($`\lvert c_6(0)\rvert_1 = 1.52\times10^6`$, $`\lvert c_7(0)\rvert_1 = 1.95\times10^7`$),
so for n ≥ e^{73.89} the expected offset is about 1.23×10⁵ n⁻⁵, a factor 8 inside d₀ = 10⁶.

## 5. The 1/n rate

§13.1 item 5 said that without the refinement by S^{(1)} the branch converges like $`O(n^{-1+1/(2\kappa_1)}(\ln n)^{3.23})`$.
That rate comes only from evaluating the jump at $`u_* = \kappa_1/\ln n`$. With the fixed u* = 0.0081 of §1, B5(2)
(§3) gives an additive error $`\lvert\tilde E\rvert \le C_E/n`$ and a C¹ bound $`10^6C_E/n`$, uniformly on the box, for every
n ≥ e^{73.89}. The implicit function theorem (§8) then gives $`\lvert(A_n, \omega_n) - (A_*, \omega_*)\rvert \le C_7/n`$ with
an explicit C₇. So the O(1/n) rate of Theorem B needs no S^{(1)}.

What S^{(1)} would add: the coefficient of 1/n. Replacing Z_in by Z_in + n⁻¹Z^{(1)} in B5(1) makes the forcing of the
W-system O(n⁻²), and the jump becomes $`S + n^{-1}S^{(1)} + O(e^{1/(2u_*)}n^{-2})`$, so
$`(A_n, \omega_n) = (A_*, \omega_*) - n^{-1}DS^{-1}S^{(1)} + O(n^{-2})`$. It would also lower n₀, since C_E = e^{76.9} is
dominated by e^{1/(2u*)} times the O(1/n) data of (H8), which Z^{(1)} absorbs. That needs Z^{(1)} as a certified lateral
solution of the linearized inner problem with forcing (a lemma-S-type fixed point for an inhomogeneous linear
problem, then its own validated integration for S^{(1)}). It was not attempted: it is not needed for Theorem B(1) as
stated.

## 6. The light cone: what Theorem B(1) needs there

Items 4 and 5 of the queue asked for B6's block-diagonalization with constants and for the transport of the jump along
x > 1. Working them through shows that the chain needs something different, and harder, than either. This section
states it.

**What the chain uses.** Theorem B(1) is about solutions *analytic at the light cone*. The certified links (B8, B5(1),
B5(2), lemma L, lemma P, B7) all concern the centre-regular solution on the sector 𝒟′_n, which excludes the disc
|x − 1| < δ/2, and its outer jump $`\Psi_{\rm out} = Y^+ - Y^-`$ on the Stokes segment, Y^± the continuations around that
disc through the upper and lower half planes. To connect them one needs:

**(E)** *For n ≥ n₀ and real p in the box, the solution in the light-cone gauge is analytic at x = 1 if and only if
Ψ_out ≡ 0 on the Stokes segment.*

*(H-LC), (H-G) and (H-J) were open when this section was written; §§13–14 prove them for M = 1.*

**Hypothesis (H-LC).** For n ≥ n₀ and real p in Theorem A's box, the centre-regular solution in the light-cone
gauge (H-G below):
1. continues analytically from (0, 1 − δ/2] to the real segment (0, 1) and to the open half-discs
   $`H^\pm = \{0 \lt \lvert x - 1\rvert \lt \delta/2,\ \pm\mathrm{Im}\,x \gt 0\}`$, and is bounded there;
2. has continuations Y^± that are C¹ up to x = 1 from within $`\overline{H^\pm}`$, with the same value and derivative at
   x = 1, so that $`\lvert\Psi_{\rm out}(x)\rvert \le C(x - 1)`$ on (1, 1 + δ/2).

**Hypothesis (H-G).** There is a normalization c(0) = c^K(0) + d with $`\lvert d\rvert_1 \le 10^7n^{-5}`$ (the range
in which B8 and (H8) are certified, §4) for which c(x) → 1 as x → 1⁻ on every harmonic.

*Proof of (E) given (H-LC).* (⇐) If Ψ_out ≡ 0, put Y = Y⁺ on H⁺ and Y = Y⁻ on H⁻. Both continue the real solution
across (1 − δ/2, 1) and agree across (1, 1 + δ/2), so Y is holomorphic on the punctured disc; it is bounded by (H-LC)(1),
so x = 1 is a removable singularity. (⇒) If Y is analytic on |x − 1| < r, then on H⁺ ∩ {|x − 1| < r} it coincides
with the continuation into H⁺ (both continue the real solution from (1 − r, 1)). The region 𝒟′_n ∪ H⁺ is simply
connected and Y is analytic on it (B8 and (H-LC)(1)), so the continuation around the arc and the one through H⁺ agree on
(1, 1 + r): Y⁺ = Y there, and likewise Y⁻ = Y. So Ψ_out = 0 on (1, 1 + r), and on the whole Stokes segment by
uniqueness for the ODE, which is regular there (B8's lower bounds on D). ∎

**Why B6′ and B8 do not give (H-LC).**
- *B8* excludes the disc |x − 1| < δ/2. Its remainder system (B8_constants.md §4) contains $`D^{-1}`$, which is
  O(1/(x − 1)) at the light cone. In the exact equation $`xDS' = AS - x^2BV`$ the combination
  $`D^{-1}(AS - x^2BV) = xS'`$ is bounded for a solution regular at x = 1, but B8's bound algebra estimates the two
  terms separately (`G1` and `W2` in `b8m_bounds.improve` use `iD`), and its lower bound on D vanishes at x = 1. In the
  B8 gauge the singular points x_± (where x is an eigenvalue of C(x)) also move off x = 1, by
  $`O(\lvert R_c(1)\rvert)`$.
- *B6′* (C_selection.md §12) is a Volterra bound for the *linearized* system along a background that is assumed
  analytic up to x = 1. It does not control the nonlinear solution itself, and the block-diagonalization it relies on
  is not written. With (H-LC) in hand, B6′ would only be needed for the light-cone *coefficient* (the factor
  κ_{n,μ}), which Theorem B(1) does not need (§8.4).

**Closest tools.** For the linear part, Olver's error-controlled Liouville–Green approximation up to a regular singular
point (*Asymptotics and Special Functions*, Ch. 10–12); C_selection.md §12's Volterra bound is of this type. For the
nonlinear statement, a bootstrap of B8's kind on H⁺ with three changes: the fast kernel near x = 1 is
$`\lvert(x-1)/(s-1)\rvert^{(n-1)/2}`$ along progressive paths (B8.1's paths reach H⁺ and keep Re Φ monotone); the
remainder system is rewritten so that $`D^{-1}(AS - x^2BV)`$ enters as $`xS'`$; and the gauge is imposed by integrating
R_c from x = 1 (a contraction mapping rather than a continuation in |x|). The analogous step in the rigorous finger
selection proofs is Xie and Tanveer's treatment near the finger tip (Xie, arXiv:1610.00076, as summarized in
C_selection.md §2); those papers were not read for this note.

**Done in §13** (Lemma LC), by the bootstrap route described above with the gauge imposed at x = 1 and a
contraction in place of B8's continuation argument.

## 7. The outer jump is determined by its label

**Hypothesis (H-J)** (proved for M = 1 in §14). For n ≥ n₀ and real p in the box: if $`L_{u_*}(\Psi_{\rm out}) = 0`$ then Ψ_out ≡ 0 on the
Stokes segment.

This is what C_selection.md §10.3's "both jumps are fast-type" asserted for the outer jump (now corrected there). For
the inner jump the corresponding statement is proved (lemma L, §2.2): the solutions of the inner secant equation that
are flat at z = 0 form a two-dimensional space on which $`L_z`$ is injective. For the outer jump nothing certified so
far sees its exponential smallness: B5(1) bounds W^± separately by O(1/n), and B8's remainder bounds are polynomial.

*Route.* An outer analogue of lemma 6a's flat estimate on the Stokes segment (1, √(n u*)]: for a solution v of the
outer secant equation that tends to 0 as x → 1⁺, write v = b + T₀v, with the fast components integrated backward from
x* = √(n u*) (the kernel $`e^{-n(w(s) - w(x))}`$, w = −½ ln(1 − x⁻²), is at most 1 for s ≤ x) and the slow ones
forward from x = 1⁺ with zero boundary values; then ‖T₀‖ < 1 in a norm weighted by the fast mode gives v = 0 when its
fast part at x* vanishes. Its inputs are B8's bounds on [1 + δ/2, √(nZ)], lemma L's inner bounds near x*, and (H-LC)
near x = 1 (for v = Ψ_out to tend to 0 there). So (H-J) rests on (H-LC).

**Where the transport of the jump (queue item 5) stands.** Conclusions (2) and (3) of Theorem B(1) do not need the
transport of the jump along x > 1 with error control, nor the factor κ_{n,μ}: once the selection function is
$`L_{u_*}(\Psi_{\rm out})`$, the chain is (E) + (H-J) + B5(2) + B7. Conclusion (1) in the form
"$`C_{n,\mu} = \kappa_{n,\mu}[S_\mu + R]`$" (the Frobenius coefficient at the light cone) would need it, together with
B6 for the nonlinear solution; Theorem B(1) is restated in §8.4 without it.

## 8. Lemma P, B7 and Theorem B(1)

*Scripts: `solver/proof/b7_adjoint.py` (Lemma P), `solver/proof/b7_ift.py` (B7). Outputs `b7_box.json`,
`b7_adjoint.json`, `b7_ift.json`, logs `b7_box.log`, `b7_adjoint.log`, `b7_ift.log`, `b7_checks.log`.*

### 8.1 Why a further link is needed

B7 needs the Jacobian of the multiplier S = k₊₁ (the quantity B5(2) delivers), but Theorem A(M = 1) certifies the
Jacobian of a different function: the D-component of the jump at ξ = 0.3, $`J_D(p) = \psi_{D,+1}(0.3; p)`$. Lemma 6a gives
J_D = 0 ⇔ k = 0, a qualitative statement. The quantitative link is a forward propagation of fast-type solutions from
small z up to ξ = 0.3, over which the fast mode grows by about e^{56}. The notes did not list it.

### 8.2 Lemma P (certified)

Let $`X_{\mathbb C} = \hat x + [-1.1\times10^{-8}, 1.1\times10^{-8}]^2 + i[-10^{-8}, 10^{-8}]^2`$ in (A, ω/ω∞), a complex box
containing a polydisc of radius 10⁻⁸ around every point of Theorem A's box X = x̂ ± 10⁻⁹, and contained in B5(1)'s
box 𝒦.

**Lemma P.** For every p ∈ X_ℂ, $`J_D(p) = \chi(p)^T\psi(0.09; p)`$ in the variables (D, S, C) of `validated_int.py`, where
χ(p) is holomorphic and lies in the enclosure of `b7_adjoint.json` (entries about 10²⁵, relative radius 2.3×10⁻⁴).

*Proof.* The backward run starts from the hull of {y⁺(0.3; p), y⁻(0.3; p) : p ∈ X_ℂ} (the lower-path run over X_ℂ,
`b7_adjoint.py box`, state radius 8.3×10⁻⁹; y⁻ = σ(y⁺(p̄)) and X_ℂ is conjugation-invariant) and integrates along the
real axis from 0.3 to 0.09 with validated_int's QR-Lohner integrator (196 steps). At step k the Lohner set X_k is a
parallelepiped containing y^±(ξ_k; p) and the segment between them, and the integrator encloses the step Jacobian over
X_k in an interval matrix J_k. By the mean value theorem in integral form, $`\psi(\xi_{k+1}) = \bar B_k\psi(\xi_k)`$ with
$`\bar B_k = \int_0^1 D\Phi_k(y^- + s\psi)\,ds \in J_k`$ (dynamic 7 × 7 block; y⁺ and y⁻ carry the same parameters). So
$`J_D = e^T\bar B_0^{-1}\cdots\bar B_{K-1}^{-1}\psi(0.09)`$ and $`\chi_{k+1} = \bar B_k^{-T}\chi_k \in J_k^{-T}\chi_k`$, χ₀ = e_{D,+1},
carried as a QR-Lohner set. The backward steps shrink the fast modes of ψ, so χ grows in the dual fast direction, where
the QR-Lohner representation keeps relative accuracy. χ is holomorphic in p because every B̄_k is. ∎

**With lemma L.** At ξ = 0.09, z = u* = 0.0081, lemma L gives ψ in terms of S̃ = V⁻¹k:
$`\psi_\Phi = e^{-1/(2u_*)}Vu_*^{\Lambda}(I + \tilde R)\tilde S`$, and the slow components ψ_P, ψ_C bounded by lemma L's
estimates. With D = ξP, S = zQ and $`\psi_S = u_*\mathcal C^{-1}(\psi_\Phi + D\psi_P - [\psi_C Q^-])`$ this gives

```math
J_D(p) = \tilde\ell(p)\,\tilde S(p),\qquad \tilde\ell = \tilde\ell_0(I + \tilde R) + \tilde\ell_{\rm rest},\qquad \tilde\ell_0 = u_*e^{-1/(2u_*)}\chi_S^T\mathcal C^{-1}Vu_*^{\Lambda},
```

with $`\sup_{X_{\mathbb C}}\lVert\tilde\ell\rVert_1 \le 1.626\times10^{-6}`$ (ℓ̃₀ part 1.139×10⁻⁶, remainder 8.4×10⁻⁸), and by Cauchy
$`\lvert\partial_j\tilde\ell\rvert \le 162.6`$ on X (`b7_ift.ell_bound`).

**Checks.** χ at the box centre agrees with an independent double-precision forward integration of the variational
equation along y⁺ (`b7_adjoint.py check`): every entry lies in its ball (relative agreement about 10⁻⁸). Mutation:
$`J_k^{-1}`$ in place of $`J_k^{-T}`$ (`--mutate=transpose`) gives a relative excess of 5.6, caught.

### 8.3 B7 (certified, given the lemmas)

With F(p) := J_D(p) + Q(p), $`Q := \tilde\ell(I + \tilde R)^{-1}\tilde E`$, and real p ∈ X:

1. *Zeros of the selection condition are zeros of F.* If L_{u*}(Ψ_out) = 0, B5(2) gives $`\tilde S = -(I + \tilde R)^{-1}\tilde E`$, so
   $`J_D = \tilde\ell\tilde S = -Q`$.
2. *F has exactly one zero p_n in X* by Krawczyk's test with Theorem A's data: J_D(x̂) (point run) and the interval
   Jacobian J(X) over X (box run, `lemma6_path_box_1e-9.json`), widened by bounds on DQ from the product rule
   ($`\lvert D\tilde\ell\rvert`$ by Lemma P, $`\lVert D\tilde R\rVert \le 0.353\times10^6`$, $`\lvert D\tilde E\rvert \le 10^6C_E/n`$, B5(2)).
   Its zero satisfies $`\lvert p_n - p_*\rvert_\infty \le \lVert J(X)^{-1}\rVert\,\sup\lvert Q\rvert`$, with ‖J(X)⁻¹‖ ≤ 44.4.
3. *Existence.* The homotopy $`g_t = [V((I + t\tilde R)\tilde S + t\tilde E)]_{+1}`$ runs from g₀ = k₊₁ = S to g₁ =
   [L_{u*}(Ψ_out)]₊₁. S has exactly one zero in X, at p*, and it is simple: S = 0 ⇔ J_D = 0 (lemma L and lemma 6a),
   and DJ_D(p*) = ℓ(p*)DS(p*) with DJ_D(p*) invertible, so DS(p*) is injective. So deg(g₀, X, 0) = ±1. On ∂X,
   $`\lvert g_t\rvert \ge (1 - \lVert R\rVert)\lvert k_{+1}\rvert - \lVert V\rVert\lvert\tilde E\rvert`$ with ‖R‖ ≤ ‖V‖‖V⁻¹‖ε_L = 0.649
   (harmonic coordinates), and $`\lvert k_{+1}\rvert = \lvert k\rvert_\infty \ge \lvert J_D\rvert/(\lVert\tilde\ell\rVert\,\lVert V^{-1}\rVert)`$ with
   min_{∂X}|J_D| ≥ 2.63×10⁻¹¹ (from J_D(x̂) + J(X)(p − x̂) on 160 boundary segments). When this is positive,
   g₁ has a zero in X; for real p, L_{u*}(Ψ_out) is antisymmetric, so it vanishes there.

**Result** (`b7_ift.log`, with B5(2) for the (H8) data of d₀ = 10⁷, so for every normalization |d|₁ ≤ 10⁷n⁻⁵):

| ln n | 86 | 88 | 89 | 89.75 | 90 | 92 | 95 |
|---|---|---|---|---|---|---|---|
| Krawczyk | fails | fails | passes | passes | passes | passes | passes |
| degree condition | fails | fails | fails | passes | passes | passes | passes |
| bound on \|p_n − p*\|_∞ | | | 6.4×10⁻¹⁰ | 3.0×10⁻¹⁰ | 2.4×10⁻¹⁰ | 3.2×10⁻¹¹ | 1.6×10⁻¹² |

The scan runs over ln n ∈ [74, 110] in steps of 0.25. n enters only through |Ẽ| ≤ C_E/n: the Krawczyk test is
inclusion-monotone in the widths |Q|, |DQ| (both increasing in C_E/n), and the degree condition is monotone in C_E/n,
so passing at n₀ gives every n ≥ n₀ (the script also checks that every later row of the scan passes).
So for every n ≥ n₀ = e^{89.75} exactly one p_n ∈ X satisfies L_{u*}(Ψ_out) = 0, and
$`\lvert p_n - p_*\rvert_\infty \le C_7/n`$ with C₇ = 2.90×10²⁹ = e^{67.84}. With B5(2) for the data of d₀ = 0 and 10⁶ the
threshold is the same and C₇ = e^{67.81}, e^{67.82} (`b7_ift_d0_0.log`, `b7_ift_d0_1e6.log`). Mutation: J_D(x̂) shifted by 5×10⁻¹¹ (a wrong input that moves
the zero of J_D out of X) fails at every n of the scan (`--mutate=JD`), caught.

### 8.4 Theorem B(1)

*Superseded by §15: (H-LC), (H-G) and (H-J) are now proved for M = 1.*

**Theorem B(1) (M = 1; conditional on (H-LC), (H-G), (H-J) and on Braaksma's theorem).** For every n ≥ n₀ = e^{89.75}
(D = n + 2) there is exactly one p_n = (A_n, ω_n/ω∞) in Theorem A's box X = x̂ ± 10⁻⁹ for which the centre-regular
solution of the one-harmonic model with profile A_n sin θ, frequency ω_n and the light-cone gauge is analytic at the
light cone, and $`\lvert p_n - p_*\rvert_\infty \le e^{67.84}/n`$. In particular, every such sequence converges to the zero p* of
Theorem A(M = 1) at the rate 1/n.

*Proof.* (E) of §6 (from (H-LC), (H-G)) and (H-J) make "analytic at the light cone" equivalent to L_{u*}(Ψ_out) = 0;
§8.3 does the rest. (H-G) must supply the gauge as c(0) = c^K(0) + d(p) with d holomorphic on B5(1)'s complex box and
$`\lvert d\rvert_1 \le 10^7n^{-5}`$, the range in which B8, (H8), B5(1) and B5(2) are certified (§4). ∎

What is certified without the hypotheses: for every n ≥ e^{89.75} and every normalization c(0) = c^K(0) + d(p) as above,
exactly one p ∈ X makes the label of the outer jump at u* = 0.0081 vanish, within e^{67.84}/n of p*.

Relative to Theorem B(M) of C_selection.md §6: conclusion (2) is the statement above; conclusion (3) (a limit of
selected parameters is a zero of S) holds for limits in X; conclusion (1) in the form "the light-cone coefficient is
κ_{n,μ}[S_μ + R]" is replaced by "the label of the outer jump at u* is (I + R)S + E", which is what (2) and (3) use.

## 9. Discrepancies with the existing notes

Reported, and corrected only in C_selection.md §10.3, §10.4, §11, §13 (the canonical status, as asked) and in
annotations to B5_inner.md and `b5_threshold.py`:

1. **The outer jump was asserted to be fast-type** (C_selection.md §10.3, "Both jumps are fast-type"): unproved; now
   hypothesis (H-J), resting on the light cone (H-LC).
2. **`b5_threshold.py`'s error model** used the exponent ⟨F²⟩ = 3.23 for the fast mode, where the Φ-component's is
   Re 𝒢 = ⟨F²⟩ − 3/2 = 1.73, and omitted the weight m_Φ = |z|: its estimates (ln n ≳ 285, ≳ 104.7) are about 10⁵ too
   pessimistic in the error, and its margin 10⁻⁷ was not derived from the implicit function theorem.
3. **The moving evaluation point** κ₁/ln n is unnecessary (Lemma D′): the certificates already cover a fixed u*, which
   gives the O(1/n) rate without S^{(1)} (§13.1 item 5 is resolved this way, not by S^{(1)}).
4. **B8's normalization** (§13.1 item 4): the run of §8.4's extension does not settle the exact light-cone gauge, which
   needs the solution at x = 1; that is (H-G).
5. **B6** was described as "proved up to a routine block-diagonalization". The chain needs control of the nonlinear
   solution near x = 1, (H-LC), which B6′ (linear) does not give; and B6's light-cone coefficient is not needed.
6. **B7** needed a link the notes did not list, from Theorem A's Jacobian of J_D(0.3) to the Jacobian of the multiplier:
   Lemma P (§8.2).
7. **notes/PROOF_QUEUE.md, P7 row** quotes "certified threshold n ≥ e^{62}–e^{74}, estimated ln n ≳ 105–285" for
   Theorem B(1). The chain's threshold is now n₀ = e^{89.75} (§15). That row is left to the coordinator.
8. **B8's gauge statement.** C_selection.md §11 and B8_constants.md call c(0) = c^K(0) "the light-cone gauge up to
   O(n⁻⁵)". This is true (§13: |d| ≤ 3.59×10⁶n⁻⁵), but the notes gave no proof, and the formal argument alone does not
   give it.
9. **The S-mode at the light cone is attracting toward x = 1** (|E(x, s)| ≤ (|x − 1|/|s − 1|)^{N_L}, N_L ≈ n/2). In the
   exact gauge the light cone helps the estimates rather than obstructing them. The notes treated it through the
   linear B6/B6′ (C_selection.md §12), which are not needed.

## 10. What the independent review (P2) should check

New certificate code and the claims it carries (none reviewed yet):

| file | claim | shares machinery with |
|---|---|---|
| `b5_label.py` | lemma L (L1): the Volterra/Grönwall bound ε_L(0.0081) ≤ 0.3527; the identity for (J_F − 𝒢)/z (`jf_div`); the 2 × 2 eigen-decomposition over 𝒦 (`eigen`); (L2) via `lemma6_sector.flat_estimate` on lemma S over 𝒦 | `b5_bounds.py` (inner enclosures on cells), `b5_formulas.py` (Jacobian blocks), `lemma6_sector.py`, `fixed_point.py` |
| `b5_jump.py` | B5(2): C_E = e^{76.940} (d₀ = 10⁷ data), the superposition of B5(1)'s super-solutions | the B5(1) certificate JSON, the (H8) logs |
| `b8m_bounds.py --d0` | the two added terms of B8_constants.md §8.4 | B8 (unchanged otherwise) |
| `b8m_lc.py` | Lemma LC: the closure with constants per band and the region L (`build_L`: the bounded forms of §13.3; `kernel_lc`: N_L; the S-, c- and d-formulas on L in `evaluate`); the Lipschitz matrix (`lipschitz`: ball-mode secants per candidate, the sum hypotheses on the ray parts, the perturbation of the derivative bounds through `CONTAINS`) and the Perron check (`contraction_check`) | `b8m_bounds.py` (`known`, the atoms), `b8_alg.py` (the up/lo functions are swapped for ball-mode versions; `b8_alg` itself is unchanged) |
| `b8m_lcj.py` | Lemma J: the forward differentiation `DQ` (the secant coefficients, inverses with frozen lower bounds, the v_c′ pseudo-component on L), the gain matrix Γ and its derivation (§14), the endpoint at x* | `b8m_lc.py`, `b8m_bounds.py` |
| `b7_adjoint.py` | Lemma P: `start_data_c` (fixed_point.start_data for complex boxes), `AdjLohner.step` (validated_int's step plus the adjoint block), the secant argument B̄_k ∈ J_k | `validated_int.py`, `fixed_point.py`, `lemma6_common.py` |
| `b7_ift.py` | B7: `ell_bound` (the pairing and its remainder terms), the Krawczyk assembly, `min_bd_JD`, the degree condition | Theorem A's saved runs |

Hand proofs to check:
- Lemma D′ (§1), including that `b8m_bounds.py` uses the outer radius only through ρ_max;
- lemma L's proof (§2.3);
- B5(2) (§3.2);
- (E) given (H-LC) (§6);
- Lemma P (§8.2);
- the B7 logic and the degree argument (§8.3);
- Lemma LC (§13): the fixed-point formulation, the bounded forms near x = 1, the L kernel, the majorant-derivative
  argument for the Lipschitz matrix, and the double-sector symmetry;
- Lemma J (§14): the secant system, the label endpoint, the gain matrix, and the a priori argument.
A second implementation of lemma L's bound (for example a validated integration of the label equation) and of the
adjoint (for example a forward Lohner run of the variational equation in the growing direction, or CAPD) would be the
natural independent checks.

## 11. Reproduce

In `solver/proof/` (times on the shared 2-core machine under load):

```
python3 b5_label.py --json=b5_label.json            # lemma L, ~2 s (b5_label.log)
python3 b5_label.py check                           # formula + numerical label variation (b5_label_check.log)
python3 b5_label.py check --mutate=Ct2              # must be caught (also --mutate=scale; b5_label_mutations.log)
python3 b5_jump.py --json=b5_jump.json              # B5(2), d0 = 0 data (b5_jump.log)
python3 b5_jump.py --mutate=sum                     # must be caught
python3 b8m_scan.py 4 73.89 --d0=1e7 --rA=2e-6 --rW=2e-6 --log=b8m_scan_K4_d0_1e7.log   # B8 with the offset, ~1 min
python3 b8m_h8.py 4 73.89 --log=b8m_scan_K4_d0_1e7.log --d0=1e7 > b8m_h8_K4_L73.89_d0_1e7.log   # (H8), a few min
python3 b5_jump.py --h8log=b8m_h8_K4_L73.89_d0_1e7.log --json=b5_jump_d0_1e7.json   # (same for 1e6)
python3 b8m_scan.py 4 104.68 --kappa1=0.85 --rA=2e-6 --rW=2e-6 --log=b8m_scan_K4_k0.85.log   # confirmation only
python3 b7_adjoint.py box --hr=1.1e-8 --hi=1e-8 --out=b7_box.json             # ~6 min
python3 b7_adjoint.py adjoint --box=b7_box.json --out=b7_adjoint.json         # ~1-2 min
python3 b7_adjoint.py point --out=b7_box_point.json                           # ~6 min
python3 b7_adjoint.py adjoint --box=b7_box_point.json --out=b7_adjoint_point.json
python3 b7_adjoint.py check --adj=b7_adjoint_point.json --start=b7_box_point.json
python3 b7_adjoint.py adjoint --box=b7_box_point.json --out=b7_adjoint_point_mutT.json --mutate=transpose
python3 b7_adjoint.py check --adj=b7_adjoint_point_mutT.json --start=b7_box_point.json --mutate=transpose   # caught
python3 b7_ift.py --adj=b7_adjoint.json --jump=b5_jump_d0_1e7.json --json=b7_ift.json    # n0 = e^89.75 (b7_ift.log)
python3 b7_ift.py --adj=b7_adjoint.json --jump=b5_jump_d0_1e7.json --mutate=JD         # caught (b7_ift_mutJD.log)
python3 b8m_lc.py 4 73.89 --rA=2e-6 --rW=2e-6 --lip --log=b8m_lc_K4.log   # Lemma LC, ~4 + 8 min (b8m_lc_K4.out)
python3 b8m_lc.py --lipcheck=b8m_lc_lip.json        # contraction constant from the saved matrix: theta = 0.7355
python3 b8m_lc.py 4 73.89 --rA=2e-6 --rW=2e-6 --mutate=NL   # the closure must fail (b8m_lc_mutNL.log)
python3 b8m_lc_numcheck.py 1000 --log=b8m_lc_K4.log # numerical check at n = 1e3 (b8m_lc_numcheck_n1e3.log)
python3 b8m_lc_numcheck.py 1000 --log=b8m_lc_K4.log --mutate=d0      # caught (also --mutate=weight)
python3 b8m_lcj.py --log=b8m_lc_K4.log --json=b8m_lcj.json   # Lemma J, ~1 min (b8m_lcj.log)
python3 b8m_lcj.py --log=b8m_lc_K4.log --mutate=kernel     # caught (also --mutate=Om; b8m_lcj_mutations.log)
```

## 12. P1L: plan for the light-cone hypotheses

*Plan only (queue item P1L); nothing in this section is proved yet.*

### 12.1 What the precedents give

- **Reiterer–Trubowitz** (D = 4; read through a summary of arXiv:1203.3766, §§1–3; their §4 estimates not seen). The
  light cone is the coordinate line ξ = 1, interior to the domain of analyticity (a Fourier–Chebyshev space on
  |ξ| < 41/40), and the operator they invert, J = (1 + ξ)∂_ξ + 1, is the transport along the null direction transverse
  to it, so nothing is divided by the coefficient that vanishes there; no boundary condition is imposed. Two things
  carry over: fix the light cone at a coordinate value for all harmonics (here the exact gauge c(1) = 1), and write the
  singular equation so that only the S-block sees the vanishing coefficient, in Fuchsian form. One thing does not:
  their space contains only functions analytic across the light cone, i.e. only the selected solution, whereas
  Theorem B(1)'s uniqueness needs control of the centre-regular solution for every p in the box, which for p ≠ p_n has
  a $`(x-1)^{\Lambda}`$ branch (Re Λ ≈ n/2). A Chebyshev-type space is also out of reach at n ≈ e^{90}.
- **Briot–Bouquet.** In the exact gauge, x = 1 is a regular singular point for S only: with t = x − 1,
  $`tS' = \hat\Lambda(t)\,(S - S_*(t, V, \beta, c))`$, $`\hat\Lambda(0) = \tfrac{n-2}2 + \mathfrak B - \Omega + O(1)`$, and V, β, c are regular. The
  exponents have imaginary parts ∓ω + O(1/n) ≠ 0, so no resonance, and the analytic solutions form a family
  parametrized by (V(1), β(1)). The plan uses this structure through the inward Fuchsian kernel, rather than through
  an explicit majorant series and a matching: a matching at a point x₁ = 1 − δ/2 would have to resolve the gauge
  offset d = c(0) − c^K(0) (about 1.2×10⁵ n⁻⁵) from B8's enclosure, whose radius there is about 10¹⁰ n⁻⁵, so it would
  need a variational bootstrap in d anyway.
- **Olver / B6′.** Volterra bounds at a regular singular point of a linear system: the near-x = 1 piece of (c).

### 12.2 Why (a) and (b) are one certificate

In B8's gauge (c(0) = c^K(0) + d with d not the exact offset) the singular points are where x is an eigenvalue of
C(x): x_± = c₀ ± (c₂c₋₂)^{1/2}, split by O(|R_c(1)|), real for real p and complex for complex p. B8's bound algebra
divides by D = x² − C², whose lower bound vanishes near both, so B8 cannot be continued into the light-cone disc
before the gauge is fixed. In the exact gauge, R_c(1) = 0, so δD := D − D^K = −(C^KR_C + R_CC^K + R_C²) is O(|x − 1|),
D = (x − 1)D̃ and D^K = (x − 1)D̃^K with D̃, D̃^K invertible near x = 1, and every D⁻¹ in Lemma R can be rewritten in
bounded form:
- δD D⁻¹ = (δD/(x − 1))D̃⁻¹, with |δD/(x − 1)| bounded by sup|R_c′| on the segment [1, x];
- $`xW_2 = \Lambda G_2`$, $`G_2 = (A^K)^{-1}D^KD^{-1}(CM_hS + x^2M_hC^{-1}Y)`$, with D^KD⁻¹ = D̃^KD̃⁻¹;
- q = ℓ((n − 1)I + Λ)⁻¹ and xq′ are already bounded.

So the S-equation becomes $`xR_S' = \Lambda(R_S + G_1 + G_2)`$ with G₁, G₂ bounded up to x = 1. But the gauge condition
is imposed at x = 1 while centre-regularity is imposed at x = 0, so the problem becomes a two-point problem, and B8's
continuation argument (Lemma B8.b, an initial-value argument) is replaced by a fixed-point argument.

### 12.3 Lemma LC (target; gives (H-LC) and (H-G))

Setting: M = 1, K = 4, κ = 0.3, κ₁ = 0.6, δ = 0.05, the complex parameter box 𝒦, and the domain
$`\hat{\mathcal D}_n = \mathcal D'_n \cup \overline{H^+}`$ (B8's fixed-radius domain together with the closed upper half-disc at x = 1,
including the real segment [1 − δ/2, 1 + δ/2]).

**Statement.** There are n_LC and constants ĥ such that for every n ≥ n_LC and p ∈ 𝒦:
1. exactly one centre-regular solution Z of the reduced system with Z − Z^K in the class (B_ĥ) on $`\hat{\mathcal D}_n`$
   satisfies c(x) → 1 as x → 1 in $`\overline{H^+}`$;
2. Z is holomorphic in p, and d(p) := c(0) − c^K(0) satisfies |d(p)|₁ ≤ d̂ n⁻⁵;
3. Z and xZ′ are bounded on $`\overline{H^+}\setminus\{1\}`$ and extend continuously to x = 1.

From these, Y⁺ = Z on H⁺ and Y⁻ = σZ(p̄) on H⁻ give (H-LC)(1)–(2), and part 2 gives (H-G). If d̂ ≤ 10⁷, then by
uniqueness for the initial-value problem Z is B8's solution with normalization d(p), so (H8), B5(1), B5(2) and B7 apply
as certified. Otherwise (H8) is rerun from ĥ, followed by `b5_jump.py` and `b7_ift.py` (minutes).

**Route.** Reuse `b8m_bounds.improve`, with these changes:
1. *c from x = 1.* $`R_c(x) = R_c(0) + \int_{Q_x}`$ with $`R_c(0) = -\int_0^1(\text{(I4) forcing})\,ds/s`$, and R_V(0) = [R_c(0)·2F], as in
   the `--d0` extension, but with d self-consistent rather than given. The explicit part ρ₄ is integrated exactly.
2. *New region L* = $`\overline{H^+}`$, using the bounded forms of §12.2, and a lower bound on D̃.
3. *Paths.*
   - S: P_{x′} to the radial projection x′ of x on the arc |x − 1| = δ/2, then radially toward 1. Along this path
     |1 − s⁻²| is non-increasing (d/dr ln|1 − s⁻²| ≥ 1/r − 1/(2 − r) − 2/(1 − r) > 0 for r ≤ δ/2), so the kernel is
     $`\lvert E(x,s)\rvert \le \lvert (x-1)/(s-1)\rvert^{(n-1)/2 - \lVert\eta\rVert/2 - \dots}`$, with gain ≈ |x − 1|/(n/2) after
     integration by parts.
   - β: radial from 0 at angle arg x (|s| increasing; the path stays in $`\hat{\mathcal D}_n`$).
4. *Fixed point.* Closure P̂(ĥ) < ĥ shows that the integral operator T maps the class into itself. In addition, a
   Lipschitz bound in the ĥ-weighted norm, with value and x-derivative components (the derivative is needed because
   of the integration by parts), shows T is a contraction. This gives existence, uniqueness and holomorphy in p.
   Planned certificate: P̂ is built from operations that are majorants with non-negative coefficients (sums, products,
   1/(1 − ‖E‖), suprema), so t ↦ P̂(tĥ) is convex and $`DP̂(\hat h)\hat h \le (P̂((1+\eta)\hat h) - P̂(\hat h))/\eta`$. Each
   operation will be checked for this property before relying on it; otherwise explicit difference bounds will be used.

*Checks.*
- the rewritten identities, in mpmath at random complex points near x = 1, with a mutation (as `b8m_identities.py`);
- a high-precision numerical solution at moderate n (shooting for c(1) = 1, integrating into H⁺) compared with the
  bound, and its d compared with the formal 1.2×10⁵ n⁻⁵ (numerical evidence);
- closure and contraction mutations (drop the L-region kernel exponent, or the R_c(0) term): must fail.

### 12.4 Lemma J (target; gives (H-J))

**Statement.** For n ≥ n_J and real p ∈ X, let v solve the outer secant equation on the Stokes segment (1, x*],
x* = (n u*)^{1/2}, with coefficients $`\int_0^1Df(Y^- + s(Y^+ - Y^-))\,ds`$. If v(x) → 0 as x → 1⁺ and L_{u*}(v) = 0,
then v ≡ 0. Applied to v = Ψ_out, this gives (H-J); Lemma LC supplies v → 0 at 1⁺.

**Route.** A flat estimate (the outer analogue of lemma 6a): v = T₀v, where:
- the growing fast components (S near x = 1, Φ in inner variables) are integrated backward from x*, where the label
  condition gives zero data; the kernel is $`\lvert e^{(n-1)(\Phi(x)-\Phi(y))}\rvert \le 1`$ for x < y;
- the slow components (V, c) and the decaying b-mode are integrated forward from 1⁺ with zero data.

Then ‖T₀‖ < 1 in a norm weighted by $`m(x) = \lvert e^{(n-1)\Phi(x)}\rvert\,x^{-a}(x-1)^{-b}`$. A rough count: the slow→fast
gain is about sup‖B‖/(a − 1), the fast→slow gain about |ω|x²/n ≤ |ω|u*, so the loop gain is small. It runs on cells in
three pieces:
- (1, 1 + δ/2]: Fuchsian, with coefficients from Lemma LC;
- [1 + δ/2, n^γ]: B8;
- [n^γ, x*]: inner variables, with B5(1)'s enclosures and lemma L's eigen-coordinates.

*Checks.* A numerical 2 × 2 matrix from light-cone amplitudes to labels at moderate n (numerical evidence), and a
mutation of the kernel weight.

### 12.5 Order, cost, and when to stop

**Order:**
1. LC identities and region-L kernel lemma;
2. LC closure;
3. LC contraction;
4. consequences, with (H8), B5(2) and B7 rerun if needed;
5. Lemma J;
6. Theorem B(1) without (H-LC), (H-G) and (H-J).

**Cost.** Each LC or J run is expected to take minutes on the shared 2-core machine, as B8's scans did; nothing
long-running is foreseen.

**When to stop and report:**
- (R1) LC does not close because the self-consistent d is too large. Global constants give |d| ≲ h_c(4/n)⁵ ≈ 10⁸ n⁻⁵,
  about 10³ times the formal value, and B8 itself does not close at d₀ = 10⁸. The remedy is constants per radial
  band on |x| ≤ 1.5 together with the exact integral of ρ₄. If that fails too, report.
- (R2) The contraction bound fails, or needs more than the majorant argument.
- (R3) Lemma J's weighted norm does not contract across the transition between the outer and inner pieces.

Braaksma's theorem remains assumed throughout.

## 13. Lemma LC: the solution up to the light cone in the light-cone gauge (certified)

*Script: `solver/proof/b8m_lc.py` (log `b8m_lc_K4.log`, output `b8m_lc_K4.out`, Lipschitz matrix `b8m_lc_lip.json`);
numerical check `solver/proof/b8m_lc_numcheck.py`. This gives (H-LC) and (H-G) of §6 for M = 1.*

### 13.1 Statement

Let Σ_n = {0 ≤ arg x ≤ α, |x| ≤ R′_n} be B8's closed sector (α = π/4 + κ/2, κ = 0.3; R′_n for κ₁ = 0.6, fixed outer
radius as in Lemma D′), *with* the light-cone disc: Σ_n = 𝒟′_n ∪ L, L = {|x − 1| ≤ δ/2, Im x ≥ 0}, δ = 0.05. The bands
are:
- N0: |x| ≤ 0.05;
- Na: 0.05 ≤ |x| ≤ 1.5, outside the disc;
- L;
- Nb: 1.5 ≤ |x| ≤ 20;
- F: |x| ≥ 20.

The paths use the ray parts Ra and Rb and the arc Ac = {|x − 1| = δ/2} ∩ Σ_n. Let w₁ = ρ^{K+1} and w = θρ^{K+1} (K = 4).

For constants h (per band, for R_V, R_S, R̂_β, R_c) and h_Sp, let 𝒦_h be the set of U = (R_V, R_S, R̂_β, R_c) that are
analytic in the interior of Σ_n, continuous with x-derivative on Σ_n ∖ {0}, and satisfy:
- on each band r, |R_V| ≤ h_{V,r}w₁, |R_S| ≤ h_{S,r}w, |R̂_β| ≤ h_{β,r}w and |R_c| ≤ h_{c,r}w₁;
- the x-derivative bounds D₀(h) that the equations give (pass 1 of B8_constants.md Lemma B8.a; on L,
  |xR_S′| ≤ h_Sp n w).

**Lemma LC (M = 1; certified).** Take the constants of `b8m_lc_K4.log`, any n ≥ n₀ = e^{73.89} and any p in B8's
complex parameter box (A, ω/ω∞ within 2×10⁻⁶(1 + i) of B5's centre, which contains 𝒦).
1. Exactly one U ∈ 𝒦_h makes Z = Z^K + U a centre-regular solution of the reduced system with c(x) → 1 as x → 1 (the
   light-cone gauge, on every harmonic).
2. U is holomorphic in p. The gauge offset d(p) := c(0) − c^K(0) satisfies |d(p)|₁ ≤ d₀n⁻⁵ with d₀ = 3.59×10⁶.
3. Z and xZ′ are bounded on L and continuous up to x = 1.

The same holds on the double sector Σ_n ∪ Σ̄_n, slit along the Stokes segment (1, R′_n] (the mirror estimates are
those for p̄, which lies in the same box). For real p the solution there is σ-symmetric.

**Consequences.**
- *(H-G)* holds with d(p) holomorphic and |d|₁ ≤ 3.59×10⁶ n⁻⁵ ≤ 10⁷ n⁻⁵. The centre-regular solution with a given
  c(0) is unique, so on 𝒟′_n, Z coincides with B8's solution for the normalization d(p). Hence B8, (H8) (the data of
  d₀ = 10⁷, §4), B5(1), B5(2) (C_E = e^{76.94}) and B7 (n ≥ e^{89.75}) apply to the light-cone-gauge solution.
- *(H-LC)* (1): Z is analytic and bounded on H⁺ (and its mirror on H⁻). (2): Y⁺ := Z|_{Σ_n} and Y⁻ := Z|_{Σ̄_n} are
  C¹ up to x = 1, and both continue the same solution on [0, 1). So they have the same value and derivative at x = 1,
  and |Ψ_out(x)| ≤ C(x − 1) on (1, 1 + δ/2).

### 13.2 The fixed-point formulation

For U ∈ 𝒦_h put F_c(U) = [R_cβ] + [c^KR_β] − ρ₄ (the (I4) forcing, R_β = R̂_β + qR_S) and

```math
d(U) = -\int_0^1 F_c(U)\,\frac{ds}{s},\qquad T_c(U)(x) = d(U) + \int_{Q_x}F_c(U)\,\frac{ds}{s},\qquad
T_V(U)(x) = [d(U)\,2F] + \int_{Q_x}\Omega R_S\,\frac{ds}{s},
```
```math
T_\beta(U)(x) = \int_{Q_x}(s/x)^{n-1}\,\hat g(U)\,\frac{ds}{s},\qquad
T_S(U)(x) = \int_{\hat P_x}E(x,s)\Big[\frac{\Lambda(s)}{s}G(U)(s) + W(U)(s)\Big]ds .
```

Here G = G₁ and W = W₂ off L, and G = G₁ + G₂ with W = 0 on L (§13.3). The S-path P̂_x is B8's P_x off L. For x ∈ L it
is P_{x′} to the radial projection x′ of x on the arc Ac, then the segment toward 1. The integrands are analytic and the
sector is simply connected, so the integrals do not depend on the path, and T(U) is analytic.

A fixed point is a solution: by Lemma R it satisfies (I2)–(I4) with zero defects. It is centre-regular, with S(0) = 0,
β(0) = 0, c(0) = c^K(0) + d and V(0) = [c(0)2F]. Its c(1) is c^K(1) + d + ∫₀¹F_c ds/s = 1.

### 13.3 The light-cone region

Since R_c(1) = 0, δD := D − D^K = −(C^KR_C + R_CC^K + R_C²) vanishes at x = 1, so D = (x − 1)D̃ and
D^K = (x − 1)D̃^K with D̃^K = (x + 1)(I − Γ). Every D⁻¹ of Lemma R then takes a bounded form:
- δD D⁻¹ = (δD/(x − 1))D̃⁻¹, with |δD/(x − 1)| ≤ (2|c^K| + |R_c|) sup_{[1,x]}|R_c′| (the segment [1, x] lies in L);
- xW₂ = ΛG₂ with G₂ = (A^K)⁻¹D̃^KD̃⁻¹(CM_hS + x²M_hC⁻¹Y);
- q·xgS = ℓx⁻²(I + e)⁻¹ε₁A^K(G₁ + G₂).

The lower bound of D̃ is (2 − δ/2)(1 − ‖Γ‖) − sup|δD/(x − 1)| = 1.975. On L the S-equation is xR_S′ = Λ(R_S + G).

*Kernel.* Along a segment toward 1, d/dr ln|1 − s⁻²| ≥ 1/r − c₁ with c₁ = 1/(2 − δ/2) + 2/(1 − δ/2), and
|λ|‖H′‖ ≤ η_L/r. Therefore |E(x, s)| ≤ (|x − 1|/|s − 1|)^{N_L}, where

```math
N_L = \frac{(n_0-1)(1 - c_1\delta/2)}{2} - \eta_L \approx 5.8\times10^{31},\qquad \eta_L = 6.17 .
```

*S on L.* With u = R_S + G, u′ = (Λ/x)u − G′ on the segment, so

```math
R_S(x) = -G(x) + E(x,x')\,u(x') + \int_{x'}^xE(x,s)\,G'(s)\,ds ,
```

with |∫| ≤ sup|G′|·|x − 1|/(N_L − 1). The value u(x′) = G₂(x′) + ∫_{P_{x′}}E(G₁′ + W₂) comes from B8's integration by
parts at x′. Hence |xR_S′| = ‖Λ‖|u| ≤ ‖(D̃^K)⁻¹A^K‖·(|u(x′)|/(δ/2) + sup|G′|/(N_L − 1)): bounded up to x = 1, which is
part 3.

*c on L.* |R_c(x)| ≤ |x − 1| sup_L|R_c′|.

### 13.4 Invariance and contraction (certified, `b8m_lc_K4.log`)

**Closure.** P̂(h) < h holds for every component and band (ratios ≤ 0.990), with

| band | h_V | h_S | h_β | h_c |
|---|---|---|---|---|
| N0 | 1.84×10⁷ | 3.84×10³ | 21.7 | 3.62×10⁶ |
| Na | 1.84×10⁷ | 5.15×10⁴ | 3.74×10⁴ | 3.64×10⁶ |
| L | 1.84×10⁷ | 2.06×10³ | 1.75×10³ | 63.7 |
| Nb | 1.85×10⁷ | 1.70×10⁵ | 1.97×10⁵ | 3.67×10⁶ |
| F | 3.41×10⁷ | 4.12×10⁷ | 4.12×10⁶ | 4.63×10⁶ |

and h_Sp = 6.0×10⁻⁷. The gauge offset is d₀ = 3.59×10⁶, of which the explicit residual ρ₄ contributes 1.90×10⁵
(computed exactly; the formal value |c₅(0)|₁ is 1.23×10⁵). Constants per band are what make d small: with B8's
single constant per component, |d| would be bounded only by about 10⁸ n⁻⁵.

So T maps 𝒦_h into itself:
- the value bounds follow from the closure;
- the derivative bounds follow from pass 1 with own values ≤ h.

**Contraction.** Use the norm N(U) = max over components and bands of sup|U|/(hw) and sup|xU′|/(h′m), where
m = nw (nw/X² on F) for S and β, m = w for V and c, and h′ = sup D₀/m.

*Why the matrix below bounds the Lipschitz constant.* By induction over the bound algebra (sums, products, inverses
with ‖g⁻¹ − g̃⁻¹‖ ≤ ‖g⁻¹‖‖g̃⁻¹‖‖g − g̃‖, and the integral operators, which are linear), the difference of any node at
U, Ũ ∈ 𝒦_h is bounded by the derivative of its majorant along the perturbed input. Each output is a maximum of
candidates. Each candidate is a sum of term-wise suprema of power series in the perturbation t with non-negative
coefficients, hence convex in t. So its slope at t = 0 is at most the secant (M(η) − M(0))/η.

*Computation.* The secant is computed exactly, in ball mode with no outward rounding between the two runs (η = 10⁻⁶).
This gives a 40 × 40 matrix (values and derivatives × 4 components × 5 bands), with spectral radius 0.705. The weights
v = (I − M/λ)⁻¹1 at λ = 0.755 give Mv ≤ θv with θ = 0.7355 (recomputed in arb from the saved upper bounds,
`b8m_lc.py --lipcheck=b8m_lc_lip.json`).

So T is a contraction on the complete set 𝒦_h. This gives existence and uniqueness. Holomorphy in p follows because
the iterates T_p^k(0) are holomorphic in p and converge uniformly. This proves parts 1–2.

*Double sector.* On Σ_n ∪ Σ̄_n slit along (1, R′_n], define T on the upper part as above and on the lower part as its
mirror; the c-offset is the shared integral over [0, 1]. Off the slit, the upper and lower paths to a point of (0, 1)
are homotopic, so T(U) is analytic across (0, 1). The estimates are those for p and p̄, so T is a contraction there
too. Its unique fixed point restricts to the solution above on each half. For real p, its mirror image is also a fixed
point, so it is σ-symmetric.

### 13.5 Checks

The checks are run by `b8m_lc_numcheck.py` at n = 10³, at the box centre. They compare against the light-cone-gauge
solution integrated with mpmath residuals in double precision (not rigorous): the remainder equation is integrated
along the real axis to 1 − 0.02, then around the circle |x − 1| = 0.02 through both half planes.

*(see the log, filled in below)*

The mutation NL (N_L = 1 + 10⁻⁹, no decay toward the light cone) makes the closure fail (`b8m_lc_mutNL.log`).

## 14. Lemma J: the outer jump is determined by its label (certified)

*Script: `solver/proof/b8m_lcj.py` (`b8m_lcj.log`, `b8m_lcj.json`, mutations `b8m_lcj_mutations.log`). This gives
(H-J) of §7.*

**Lemma J (M = 1; certified).** For every n ≥ e^{73.89} and real p in the box, let Y^± be the light-cone-gauge
solution of Lemma LC and its mirror, and Ψ_out = Y⁺ − Y⁻ on the Stokes segment (1, x*], x* = (nu*)^{1/2}. If
$`L_{u_*}(\Psi_{\rm out}) = 0`$, then Ψ_out ≡ 0.

*Proof.* v = Ψ_out = R⁺ − R⁻ solves the differences of the remainder equations (in the bounded form of §13.3):

```math
xv_V' = \Omega v_S,\qquad xv_S' = \Lambda\,(v_S + \delta G),\qquad xv_\beta' + (n-1)v_\beta = \delta\hat g,\qquad xv_c' = \delta F_c .
```

*Coefficients.* Both R⁺ and R⁻, and every point of the segment between them, lie in 𝒦_h on the Stokes segment. So the
secant coefficients are bounded locally, |δG(x)| ≤ Σ_j ℓ_{G,j}(x)|v_j(x)|, and likewise for δĝ and δF_c. The ℓ are
computed by forward differentiation of the bound algebra (`DQ`), with the inverses' lower bounds held fixed. On L,
the term δD/(x − 1) brings in the pseudo-component sup_{[1,x]}|v_c′|, which is bounded through δF_c.

*Label condition.* L_{u*}(v) = 0 means Φ⁺ = Φ⁻ at u* in B5's variables, with Φ = C⁻¹(xY′ − [xc′P]). Since
δ(xY′) = Ωv_S + Λ(v_S + δG), this gives

```math
\Lambda(v_S + \delta G)(x_*) = -\Omega v_S + \delta C\,\Phi^- + \delta[xc'P] .
```

*The map T₀.* It takes the fast component backward from x*, with that endpoint value, and the slow components and the
b-mode forward from 1 with zero data. Use the norm N(v) = max_j sup|v_j|/(s_j m), with m = e^{(n−1)Φ(x)/2} (half the
fast exponent) and the nondecreasing shapes s_S = 1, s_V = s_c = ρ, s_β = ρ/n + ρ².

Backward, ‖E(x, y)‖ ≤ e^{−(n−1)(1−h_H)(Φ(y)−Φ(x))}, with h_H = sup‖H′‖ = 0.108 on the segment. Forward,
∫₁^x m dy/y ≤ m(x)·2(x² − 1)/(n − 1). Hence N(T₀v)_i ≤ Σ_j Γ_ij N_j(v), with

```math
\Gamma_{S,j} = \frac{1 + h_H}{1/2 - h_H}\sup(\ell_{G,j}s_j) + e_j s_j(x_*),\quad \Gamma_{V,S} = \frac{2\lvert\omega\rvert n}{n-1},\quad
\Gamma_{c,j} = \frac{2n}{n-1}\sup(\ell_{F_c,j}s_j),\quad \Gamma_{\beta,j} = \frac{n}{n-1}\sup\frac{\ell_{\hat g,j}s_j}{n s_\beta} ,
```

where e_j are the endpoint coefficients (‖Λ⁻¹(x*)‖ ≤ 9.5×10⁻³, |Φ⁻| ≤ 2.0). The certified matrix
(`b8m_lcj.log`) has a Perron weight with θ = 0.176 < 1. The largest loops are S → V → S (5.9 × 1.4×10⁻³) and the
self-coupling of S (0.046). So T₀ is a contraction.

*Ψ_out lies in the space.* On (1, 1 + δ/2], with m = 1, the affine map with the actual endpoint value v_S(1 + δ/2) is
a contraction in both norms (θ₀ = 5.8×10⁻⁵). Ψ_out is bounded and tends to 0 at x = 1 (Lemma LC), so it is the sum of
the Neumann series there. That series converges in N as well, so N(Ψ_out) < ∞.

Ψ_out = T₀Ψ_out with N(Ψ_out) < ∞, and T₀ is a contraction, so Ψ_out = 0. ∎

*Mutations* (`b8m_lcj_mutations.log`), both caught:
- the backward kernel without decay (Γ_S multiplied by n₀): θ = 10³⁰;
- |ω| multiplied by 10⁴: θ = 283.

## 15. Theorem B(1) (certified, assuming Braaksma's theorem)

**Theorem B(1) (M = 1).** For every n ≥ n₀ = e^{89.75} (D = n + 2) there is exactly one p_n = (A_n, ω_n/ω∞) in
Theorem A's box X = x̂ ± 10⁻⁹ for which the centre-regular solution of the one-harmonic model in the light-cone gauge
(c(1) = 1, the solution of Lemma LC) is analytic at the light cone x = 1. Moreover
$`\lvert p_n - p_*\rvert_\infty \le e^{67.84}/n`$, where p* is Theorem A(M = 1)'s zero of the inner multiplier.

*Proof.*
- (H-LC) and (H-G) hold by Lemma LC (§13), and (H-J) by Lemma J (§14).
- By (E) (§6), the solution is analytic at x = 1 if and only if Ψ_out ≡ 0.
- By (H-J) and (E), that holds if and only if L_{u*}(Ψ_out) = 0.
- By B5(2) (§3, with the data of d₀ = 10⁷) and B7 (§8.3, with d(p) holomorphic by Lemma LC), exactly one p ∈ X
  satisfies L_{u*}(Ψ_out) = 0, within e^{67.84}/n of p*. ∎

**What it rests on.**
- Braaksma's theorem, through Theorem A(M = 1) and lemma S.
- Certificate code not yet reviewed independently (§10).

Uniqueness holds among light-cone-gauge solutions in the class 𝒦_h, a neighbourhood of the formal series. That is the
sense of "the" centre-regular solution in the light-cone gauge.

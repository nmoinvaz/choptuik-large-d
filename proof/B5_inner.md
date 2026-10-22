# Lemma B5(1): the inner fixed point on the matching domains, with explicit constants

This note proves part (1) of Lemma B5(M) of [C_selection.md](C_selection.md#103-statement-and-proof), the sectorial
identification of the continued outer solution with the inner lateral sum on the matching domains
$`\mathcal M^\pm`$, for every fixed M, and evaluates every constant for M = 1 in ball arithmetic. It was written by Claude
(Anthropic) under Nathan Moinvaziri's direction and has not been checked by a mathematician. Status words as in
[AGENTS.md](../AGENTS.md): *proved* (by hand), *checked symbolically*, *certified* (computer-assisted, trusted code named),
*conditional* (hypothesis named), *numerical evidence*. The scripts are `solver/proof/b5_*.py`.

Contents

- [Summary](#summary)
- [1. The model and the variables](#1-the-model-and-the-variables)
- [2. The equation for the difference W](#2-the-equation-for-the-difference-w)
- [3. Domains, paths and kernels](#3-domains-paths-and-kernels)
- [4. Lemma B5(1): statement and proof](#4-lemma-b51-statement-and-proof)
- [5. M = 1: the certified constants and n₀(1)](#5-m--1-the-certified-constants-and-n1)
- [6. Checks against numerics](#6-checks-against-numerics)
- [7. What B5(1) takes from B8, and what it gives B5(2)](#7-what-b51-takes-from-b8-and-what-it-gives-b52)
- [8. Status](#8-status)
- [9. Proposed text for the other files](#9-proposed-text-for-the-other-files)
- [10. Reproduce](#10-reproduce)

## Summary

- **The lemma, sharper than stated in C_selection.md.** With W = Y⁺_out − Z⁺_in in suitable variables
  (P, Φ, C, ϑ) (section 1), the bound is $`\lvert W_j(z)\rvert \le x_j\,m_j(z)/n`$ on the matching domain, with
  $`m_\Phi = \lvert z\rvert`$ and $`m_j = 1`$ otherwise. That is relative size O(1/n), without the factor
  $`\lvert u\rvert^{-2}`$ of C_selection.md §10.3, which it implies.
- **What is proved and what is assumed.**
  - Proved for every fixed M (section 4): conditional on one hypothesis, (H8), and with constants defined as suprema of
    explicit expressions in the inner solution over a fixed sector.
  - (H8) is the input from Lemma B8, stated exactly in section 7: Y⁺_out is analytic on the working domain, and on the
    inner arc $`\lvert x\rvert = n^\gamma`$ it differs from Z⁺_in by O(1/n) in these variables. At the upper end of the arc
    this is the quantitative form of "no exponentially large fast content along arg u = π/2 + κ".
  - Certified for M = 1 (section 5): every constant is evaluated in arb/acb ball arithmetic over a complex parameter box
    around the zero of Theorem A(M = 1). The inner solution comes from lemma S of [A_certify.md](A_certify.md#113-lemma-s-start-values-on-a-domain-that-reaches-the-real-axis),
    rerun over that box.
- **n₀(1).** With (κ, κ₁, γ) = (0.3, 0.6, 0.3), the certified fixed point holds for every
  $`n \ge n_0(1) = \max(e^{73.89}, n_8) = \max(1.23\times10^{32}, n_8)`$ (D = n + 2), where n₈ is Lemma B8's threshold.
  - The constant that sets it is the largest admissible radius of the working domain, Z = 0.0085 in $`z = x^2/n`$, since
    $`n_0 = \exp(\kappa_1/(Z\cos\kappa))`$. The scan also certifies Z = 0.009, i.e. $`n_0(1) = \max(e^{69.78}, n_8)`$, with
    constants ten times larger (section 5.3).
  - Two things limit Z, and at κ = 0.3 they nearly coincide. First, the radius on which lemma S encloses the inner
    solution, $`R_+\sin(\pi/4-\kappa) = 0.0091`$. Second, the contraction itself, whose constant is dominated by the loops
    through the fast row (Φ ← P and Φ ← C, closed by P ← Φ and C ← P). At Z = 0.0085 it is 0.966 (certified), and the
    same norm sampled in double precision is 0.937. So the second limit is intrinsic to a perturbative fixed point on a
    disc around z = 0, not an artifact of the enclosures.
  - A practical n₀ needs the exact inner variational flow on a fixed sector up to |z| ≈ 0.04 (section 5.4). The
    matching domain's outer edge κ₁/ln n cannot be made n-independent without it.
- **B5(2) needs more.** The jump estimate of B5(2) divides by $`\lvert e^{-1/(2u_*)}u_*^{\langle F^2\rangle}\rvert`$. With these
  constants its error is small enough for the implicit function theorem on the certified box only for ln n ≳ 105
  (with the best κ₁ ≈ 0.85; `b5_threshold.py`). That is an estimate, not part of this lemma (section 7).
- **A discrepancy, to resolve before B(1) and A(1) can be combined.**
  - C_selection.md §1 and the paper truncate b and c to even |ν| ≤ M − 1.
  - The certified inner problem (and every inner solver, `lateral_np.py`, `hier.py`, `first_order_jet.py`) keeps c on
    |ν| ≤ M + 1. For M = 1 that is c on {0} against {0, ±2}.
  - The M-model whose inner limit is the certified problem keeps b and c on |ν| ≤ M + 1; this note uses that one, and
    Lemma B5.0 holds for it as stated (checked, `b5_system.py`). Lemma B8 must use the same model.
- **Checks.** Every formula is checked against an independent evaluation, every enclosure contains the independent value
  at sample points, an end-to-end integration of the exact model stays inside the certified bound, and every check has a
  mutation that makes it fail (section 6).

## 1. The model and the variables

### 1.1 The M-harmonic model

For odd M let 𝒪 = {odd μ : |μ| ≤ M} and ℰ = {even ν : |ν| ≤ M + 1}. The **M-model** is the outer system of
C_selection.md §1 with p, m on 𝒪, b, c on ℰ, every binary product projected onto its result's set, Gp computed as
$`[[c(b-1)]\,p]`$ and $`b(p^2+m^2)`$ as $`[b\,[p^2+m^2]]`$:

```math
x c_x = (n-1)[c(b-1)],\qquad x b_x = [b\,[p^2+m^2]] - (n-1)[b(b-1)],
```
```math
x^2p_x - x[c\,p_x] = -\omega x\,p_\theta + (n-1)[[c(b-1)]\,p] - [cp] + \tfrac n2[c(p+m)],\qquad
x^2m_x + x[c\,m_x] = -\omega x\,m_\theta - (n-1)[[c(b-1)]\,m] + [cm] - \tfrac n2[c(p+m)] .
```

The even set ℰ is |ν| ≤ M + 1, not M − 1 as in C_selection.md §1. With M − 1 the inner limit would be a different
problem from the one certified in Theorem A(M = 1) (c on {0} instead of {0, ±2}). The coordinator should change §1 and
the paper's Definition of the M-model, and B8 should use this one. For M = 1 the projections are harmless: y² has
harmonics {0, ±2} exactly.

### 1.2 Inner variables

With ε = 1/n, x = √n ξ, y = p − m, σ = √n (p + m), b = 1 + εη (Lemma B5.0), put

```math
z = \xi^2 = x^2/n,\qquad y = \xi P,\qquad \sigma = zQ,\qquad \eta = z\Theta,\qquad
\Phi := 2zP_z,\qquad \vartheta := \Theta - [PP]/2 .
```

Here z is C_selection.md's u and the paper's z. On each harmonic s = iμω, D = 1 + s, D₂ = 2 + s (diagonal). The
operators on odd vectors are 𝒞X = [CX], ℬX = [[CΘ]X], and $`\hat E_0 = [PP]/2`$.

**Lemma 1 (the model in inner variables; checked symbolically and numerically).** For z ≠ 0 and ε ≠ 0 the M-model is
equivalent to

```math
2zP_z = \Phi,\qquad \Big(I - \tfrac{\varepsilon}{z}\mathcal C^2\Big)\Phi = \mathcal E Q - DP + \varepsilon(1-\varepsilon)\,\mathcal C\mathcal B P,\qquad
2z^2Q_z = \mathcal C\Phi + A_2,
```
```math
C_z = (1-\varepsilon)[C\Theta]/2,\qquad 2\varepsilon z\,\Theta_z = h,
```

with $`\mathcal E = \mathcal C(I - \varepsilon D) + \varepsilon(1-\varepsilon)z\,\mathcal B`$, $`A_2 = -zD_2Q + (1-\varepsilon)z\,\mathcal BP`$,
$`h = \hat E_\varepsilon - (1+\varepsilon)\Theta + \varepsilon z([\Theta\hat E_\varepsilon] - (1-\varepsilon)[\Theta\Theta])`$ and
$`\hat E_\varepsilon = ([PP] + \varepsilon z[QQ])/2`$. At ε = 0 (Θ = Ê₀) it is the inner system of `fixed_point.py`, with
Φ = W := V + [(C − 1)Q] there.

*Check.* `b5_system.py` derives the outer equations in inner variables per harmonic (sympy) and evaluates them at random
complex states with these rates: residual 6×10⁻⁶³ at 60 digits. The ε = 0 limit agrees with `fixed_point.py`'s system to
2×10⁻³⁹. Mutations: dropping the ε(1−ε)𝒞ℬP term, or (1+ε) → (1−ε) in h, give 9×10⁻⁴ and 4×10⁻⁴ (caught).

### 1.3 The state (P, Φ, C, ϑ)

The state used throughout is Y = (P, Φ, C, ϑ). Q is the derived quantity
$`Q = \mathcal E^{-1}\big((I - \tfrac{\varepsilon}{z}\mathcal C^2)\Phi + DP - \varepsilon(1-\varepsilon)\mathcal C\mathcal BP\big)`$.
Differentiating the Φ relation gives, with ε' := ε(1 − ε), $`\mathcal M := I - \tfrac\varepsilon z\mathcal C^2`$ and Θ' = h/(2εz),

```math
\Phi_z = \mathcal M^{-1}\Big[\frac{\mathcal E(\mathcal C\Phi + A_2)}{2z^2} - \mathcal M_z\Phi + \mathcal E_zQ - \frac{(D - \varepsilon'\mathcal C\mathcal B)\Phi}{2z} + \varepsilon'(\mathcal C_z\mathcal B + \mathcal C\mathcal B_z)P\Big],
```
```math
\vartheta_z = -\frac{(1+\varepsilon)\vartheta}{2\varepsilon z} + \frac{-\hat E_0 + z[QQ]/2 - [P\Phi]}{2z} - \frac{[\hat E_0\vartheta] + [\vartheta\vartheta]}{2} + \frac{\varepsilon}{2}\Big([\Theta\Theta] + \frac z2[\Theta[QQ]]\Big).
```

**Lemma 2 (the inner limit does not see ϑ; proved, checked numerically).** As ε → 0 with the state held fixed, P_z and
Φ_z tend to the inner rates and do not depend on ϑ:

```math
\Phi_z \to \frac{\Phi}{2z^2} + G^0(P, \Phi, C),\qquad
G^0 = \frac{(\mathcal C^2 - I)\Phi}{2z^2} + \frac{\mathcal C(\mathcal B_0P - D_2Q_0)}{2z} + [C_0'\,Q_0] - \frac{D\Phi}{2z},
```

with $`Q_0 = \mathcal C^{-1}(\Phi + DP)`$, $`\mathcal B_0 = [[C\hat E_0]\,\cdot]`$, $`C_0' = [C\hat E_0]/2`$. C_z tends to $`[C(\hat E_0 + \vartheta)]/2`$,
which does depend on ϑ.

*Proof.* In the limit, Θ enters Φ_z in three places:
- $`\mathcal C\mathcal BP/(2z)`$ (from A₂), which contributes $`+\mathcal C[[C\vartheta]P]/(2z)`$;
- the term $`\varepsilon'\mathcal C\mathcal B_zP`$, through $`\varepsilon\Theta_z = h/(2z) = -(1+\varepsilon)\vartheta/(2z) + O(\varepsilon)`$, which contributes
  $`-\mathcal C[[C\vartheta]P]/(2z)`$;
- in $`\mathcal E_zQ`$, $`[([C\vartheta]/2)Q]`$ from $`\mathcal C_z`$ and $`-[[C\vartheta]Q]/2`$ from $`\varepsilon'z\mathcal B_z`$.

The pairs cancel. $`\square`$

*Check* (`b5_system.py`): Φ_z above equals the total derivative of the Φ relation along the flow to 10⁻⁵⁰. At
ε = 10⁻³⁰ with ϑ arbitrary, $`\lvert f_\varepsilon - f_0\rvert\,\lvert z\rvert^2/\varepsilon \le 15`$, where an O(1) ϑ-dependence would
give 10³⁰. Mutation: dropping the ε'(𝒞_zℬ + 𝒞ℬ_z)P term gives 6×10²⁸ (caught).

Why these variables:
- In A_certify's variables (P, V, C) the C-row couples into the fast row with an O(1/z²) coefficient Q/(2z²). Since
  the C-component of W is O(1/n) (the light-cone gauge shift, section 7), that coupling destroys the gain of the fast
  kernel. With Φ = 2zP_z it is O(1/z), because Φ = O(z).
- With ϑ instead of Θ, the b-mode decouples from Φ at leading order (Lemma 2). Its slaving is then exact and costs
  nothing in the contraction.

### 1.4 The inner solution

Z⁺_in is the inner lateral sum of [A_certify.md](A_certify.md#111-definitions) (Braaksma's theorem, as there), with P(0) = 2F,
C(0) = 1, the same ω, ϑ ≡ 0 and Φ = 𝒞Q − DP. For M = 1, lemma S (A_certify.md §11.3) gives Z⁺_in = Y_N + η on

```math
D = \{z : 1/z \in U_0 + K\},\qquad K = \{w : -3\pi/4 \le \arg w \le 0\},\qquad U_0 = \sqrt2/R_+ .
```

Here Y_N is the formal series truncated at N terms and $`\lvert\eta_j(z)\rvert \le (\rho/w_j)\lvert z\rvert^{k_j}`$ (lemma S's
weighted ball). D contains $`\{\lvert z\rvert \le R_+\sin(3\pi/4 - \theta)\}`$ in each direction θ ∈ [0, 3π/4).

## 2. The equation for the difference W

Let Y⁺_out be the continued centre-regular solution of the M-model with profile F_n and frequency ω (hypothesis (H8)). Let
Z⁺_in be built with the same profile, P(0) = 2F_n, and the same ω; the 1/n corrections of Theorem B(M) then enter only
through B5(2)'s remainder. Put W = Y⁺_out − Z⁺_in in the state (P, Φ, C, ϑ). Both satisfy P_z = Φ/(2z) exactly, and ϑ_in ≡ 0. So, with
$`\Delta_\varepsilon := f^\varepsilon_\Phi - f^0_\Phi`$ (the exact Φ-rate minus its ε → 0 limit, as functions of the state):

```math
W_{P,z} = \frac{W_\Phi}{2z},\qquad
W_{\Phi,z} = \frac{W_\Phi}{2z^2} + \big[G^0(Y_{\rm out}) - G^0(Y_{\rm in})\big] + \big[\Delta_\varepsilon(Y_{\rm out}) - \Delta_\varepsilon(Y_{\rm in})\big] + r_\Phi,
```
```math
W_{C,z} = \tfrac12\big([C(\hat E_0 + \vartheta)]_{\rm out} - [C\hat E_0]_{\rm in}\big) - \tfrac\varepsilon2[C(\hat E_0 + \vartheta)]_{\rm out},\qquad
W_{\vartheta,z} = -\frac{(1+\varepsilon)W_\vartheta}{2\varepsilon z} + \Xi(Y_{\rm out}),
```

with $`r_\Phi := \Delta_\varepsilon(Y_{\rm in})`$ and Ξ the ϑ-rate of section 1.3 minus its principal term.

**Lemma 3 (the residual is O(ε/z), not O(ε/z²); proved, checked numerically).** On the inner solution,

```math
\frac{z\,r_\Phi}{\varepsilon} = \mathcal M^{-1}\Big\{\mathcal C^2\Phi_{\rm in}' + z(-\mathcal CD + (1-\varepsilon)z\mathcal B_0)Q_0' - \tfrac12\mathcal E(D_2\widehat{\delta Q} + \mathcal B_0P) - \mathcal C^2\tilde\Phi + (1-\varepsilon)z(\mathcal C_0'\mathcal C + \mathcal C\mathcal C_0')\tilde\Phi
```
```math
\qquad + z\mathcal A + \tfrac{1-\varepsilon}2 z\,\mathcal C\mathcal B_0\tilde\Phi + z(1-\varepsilon)^2\big(\mathcal C_0'\mathcal B_0P + \mathcal C[[C_0'\hat E_0]P]\big) + \tfrac{1-\varepsilon}2\,\mathcal C[[CH]P]\Big\},
```

where:
- $`\tilde\Phi = \Phi/z`$;
- $`Q_0' = S/(2z^2)`$ with $`S = \mathcal C\Phi + A_2^0`$, so that S = O(z²) on the solution;
- $`\Phi'_{\rm in} = \mathcal CQ_0' + [C_0'Q_0] - D\tilde\Phi/2`$;
- $`Q = Q_0 + \varepsilon\widehat{\delta Q}`$ with $`\widehat{\delta Q} = \mathcal E^{-1}[-\mathcal C^2\tilde\Phi - (1-\varepsilon)\mathcal C\mathcal B_0P + \mathcal CDQ_0 - (1-\varepsilon)z\mathcal B_0Q_0]`$;
- $`H = -\hat E_0 + z[QQ]/2 + \varepsilon z[\hat E_0\hat E_0] + \tfrac{\varepsilon z^2}2[\hat E_0[QQ]]`$;
- $`\mathcal A = \mathcal C_0'\widehat{\delta Q} - \mathcal C_0'Q - (1-\varepsilon)\mathcal C_0'DQ + (1-\varepsilon)\mathcal B_0Q + (1-\varepsilon)^2z[[C_0'\hat E_0]Q] + \tfrac{1-\varepsilon}2[[CH]Q]`$.

Every term is bounded as z → 0, so $`\lvert r_\Phi\rvert \le R_\Phi\,\varepsilon/\lvert z\rvert`$. Also $`r_C = -\varepsilon[C\hat E_0]/2`$, and the ϑ
forcing satisfies $`\lvert z\,\Xi(Y_{\rm in})\rvert \le T_F`$.

*Proof.* Expand $`f^\varepsilon_\Phi = \mathcal M^{-1}\mathcal K`$ and $`f^0_\Phi = \mathcal K^0`$ (the brackets of section 1.3 and their limits). Write
$`r_\Phi = \mathcal M^{-1}(\mathcal K - \mathcal K^0 + \tfrac\varepsilon z\mathcal C^2\mathcal K^0)`$, and use two facts:
- 𝒞Φ + A₂ = S + (A₂ − A₂⁰) with $`A_2 - A_2^0 = -\varepsilon z(D_2\widehat{\delta Q} + \mathcal B_0P)`$;
- S/(2z²) = Q₀' on the inner solution, the only place where the solution property enters.

The O(ε/z²) pieces then appear only as $`\tfrac\varepsilon z\times S/z^2`$ and cancel into $`\tfrac\varepsilon z\,\mathcal C^2\Phi'_{\rm in}`$. $`\square`$

*Checks.*
- `b5_formulas.py`: the formula equals $`(f^\varepsilon_\Phi - f^0_\Phi)z/\varepsilon`$ of `b5_system.Model` at random states to 9×10⁻⁴⁸.
  Mutation: dropping the [[CH]P] term gives 8×10⁻³ (caught).
- `b5_numeric.py residual`: on rays z → 0 at ε = 10⁻¹⁰, $`\lvert r_\Phi\rvert\lvert z\rvert/\varepsilon`$ stays at 32–40 (spread 1.04 over
  |z| ≤ 10⁻³). Mutation: evaluated off the solution (Φ + 10⁻²z) it grows like 1/|z| (caught).
- Leading value: $`R(0) = (D_2DP_0 - 2[\hat E_0P_0])/2`$, |R(0)| = 35.18 for M = 1 (`b5_data.py`). It is cross-checked against
  the independent O(1/n) recursion of [notes/FIRST_ORDER.md](../notes/FIRST_ORDER.md), section 6 below.

Without Lemma 3 the fast row's forcing would be O(ε/z²), so W_Φ = O(ε). Then the slow row W_P = ∫W_Φ/(2t) would pick up
ε ln n, and with that weight the loop P → Φ → P has gain of order κ₁, not small. Lemma 3 lets the weights be
m_Φ = |z| and m_P = 1, so that every loop carries a factor |z|.

## 3. Domains, paths and kernels

Fix κ ∈ (0, π/4], γ ∈ (1/4, 1/2), κ₁ > 1/2, and write α_z = π/2 + κ, d = e^{−iα_z}, U = 1/z, r_n = n^{−1+2γ},
Z_n = κ₁/ln n.

**The matching domain** (C_selection.md §10.2) is $`\mathcal M^+_n = \{r_n \le \lvert z\rvert \le Z_n,\ 0 \le \arg z \le \alpha_z\}`$. In U it is
$`\mathcal N = \{1/Z_n \le \lvert U\rvert \le 1/r_n,\ -\alpha_z \le \arg U \le 0\}`$.

**The working domain** is the set of points reachable from 𝒩 by the backward moves used below:

```math
\mathcal W^+_n = \{\lambda(U + s\,d) : U \in \mathcal N,\ s \ge 0,\ \lambda \ge 1,\ \lvert\lambda(U + s\,d)\rvert \le 1/r_n\}\quad(\text{in } U) .
```

**Lemma 4 (geometry; proved).**
1. $`\mathcal W^+_n`$ is contained in the sector −α_z ≤ arg U ≤ 0 and in $`\lvert U\rvert \ge \cos\kappa/Z_n`$. In z this is
   $`r_n \le \lvert z\rvert \le Z_n/\cos\kappa`$, 0 ≤ arg z ≤ α_z, and it contains 𝓜⁺_n.
2. It is closed under the backward moves U ↦ U + s d (s ≥ 0) and U ↦ λU (λ ≥ 1), as long as |U| ≤ 1/r_n.
3. Along U + s d the fast kernel satisfies $`\lvert e^{(U_s - U)/2}\rvert = e^{-as}`$ with a = sin κ/2. With
   $`c := -\mathrm{Re}(\bar U d)/\lvert U\rvert \le \sin\kappa \le 1/\sqrt2`$ and c⁺ = max(c, 0), one also has
   $`\lvert U\rvert^2/\lvert U + sd\rvert^2 \le e^{2c^+s/\lvert U\rvert}`$.
4. The functional ℓ(U) = Re(e^{iβ}U), β = 5π/4 + κ/2, strictly decreases along both backward moves.

*Proof.*
1. The sector is a convex cone containing d. The minimum of |U + sd| over s ≥ 0 is $`\lvert U\rvert\sqrt{1-c^2}`$ when c > 0,
   and c = −cos(α_z + arg U) ≤ −cos α_z = sin κ.
2. A backward move of λ(U + sd) is λ(U + (s + s'/λ)d) or λλ'(U + sd).
3. Re d = cos α_z = −sin κ. For 0 ≤ c ≤ 1/√2 the bound is A_certify's: $`\ln(1 - 2cx + x^2) + 2cx`$ is non-decreasing.
   For c < 0, |U + sd| ≥ |U|, so the ratio is at most 1.
4. d/ds ℓ = cos(β − α_z) = cos(3π/4 − κ/2) < 0, and $`d\ell(\lambda U)/d\lambda = \lvert U\rvert\cos(\beta + \arg U) < 0`$, since
   β + arg U ∈ [3π/4 − κ/2, 5π/4 + κ/2]. $`\square`$

**Paths.** From a target U ∈ 𝒲⁺_n, each component is integrated backward to the inner arc A_n = {|U| = 1/r_n}:
- **Φ along the line** $`\gamma_\Phi(U) = \{U + sd : 0 \le s \le s_e\}`$, parallel to the upper boundary ray. Its exit point is e_Φ.
- **P, C, ϑ radially**, along $`\gamma_r(U) = \{\lambda U : 1 \le \lambda \le 1/(r_n\lvert U\rvert)\}`$, i.e. in z from $`r_ne^{i\arg z}`$ out to z.

**Lemma 5 (kernel bounds; proved).** Let |g(t)| ≤ G|t|^q on the paths, write |z| ≤ Z and assume
a − (q+2) sin κ Z > 0.
- **Fast:** $`\int_{\gamma_\Phi}\lvert e^{1/(2t) - 1/(2z)}\rvert\,\lvert g(t)\rvert\,\lvert dt\rvert \le G\,\lvert z\rvert^{q+2}/(a - (q+2)\sin\kappa\,Z)`$ for q ≥ −2.
- **Slow (radial):** $`\int_{\gamma_r}\lvert g\rvert\,\lvert dt\rvert \le G\lvert z\rvert^{q+1}/(q+1)`$ for q > −1.
- **b-mode (radial):** $`\int_{\gamma_r}\lvert(t/z)^{(1+\varepsilon)/(2\varepsilon)}\rvert\,\lvert g(t)\rvert\,\lvert dt\rvert \le \tfrac{2}{n+1}\sup_{\gamma_r}\lvert t\,g(t)\rvert`$.

*Proof.*
- Fast: |dt| = ds/|U_s|². By Lemma 4.3, $`\lvert t_s\rvert^{q+2} \le \lvert z\rvert^{q+2}e^{(q+2)c^+\lvert z\rvert s}`$ with c⁺ ≤ sin κ;
  integrate $`e^{-as + (q+2)\sin\kappa\,Z s}`$ over s ≥ 0.
- Slow: direct integration.
- b-mode: $`\lvert(t/z)^{(1+\varepsilon)/(2\varepsilon)}\rvert = (\lvert t\rvert/\lvert z\rvert)^{(n+1)/2}`$ on a radial path, and
  $`\int_0^{\lvert z\rvert}(\rho/\lvert z\rvert)^{(n+1)/2}d\rho/\rho = 2/(n+1)`$. $`\square`$

**The integral identities.** Every term below is an exact consequence of section 2 (variation of constants along each
path):

```math
W_P(U) = W_P(e_r) + \int_{\gamma_r}\frac{W_\Phi}{2t}\,dt,\qquad
W_\Phi(U) = e^{\frac1{2e_\Phi} - \frac1{2z}}W_\Phi(e_\Phi) + \int_{\gamma_\Phi}e^{\frac1{2t} - \frac1{2z}}\big[\Delta G^0 + \Delta\Delta_\varepsilon + r_\Phi\big]dt,
```
```math
W_C(U) = W_C(e_r) + \int_{\gamma_r}(\text{C-rate difference})\,dt,\qquad
W_\vartheta(U) = \Big(\frac{e_r}{z}\Big)^{\frac{1+\varepsilon}{2\varepsilon}}W_\vartheta(e_r) + \int_{\gamma_r}\Big(\frac tz\Big)^{\frac{1+\varepsilon}{2\varepsilon}}\Xi\,dt .
```

The fast path is progressive: Re U grows along it, which is B8.1's monotonicity in the variable z. The b-mode needs
paths along which |z| does not decrease, because its kernel has modulus 1 on arcs and grows otherwise. That is why the
two kinds of path differ, why the data are needed on the whole inner arc and not at one point, and why the proof uses a
bootstrap rather than a contraction mapping.

## 4. Lemma B5(1): statement and proof

**Constants.** All suprema are over the closed sector $`\Sigma_Z = \{0 < \lvert t\rvert \le Z,\ 0 \le \arg t \le \alpha_z\}`$, over
the parameter set 𝒦 and over states within the bootstrap ball of Y_in (whose radius is O(1/n)). The weights are
m_P = m_C = m_ϑ = 1 and m_Φ(t) = |t|.
- **Main Jacobian blocks**: $`\gamma_{\Phi_\mu,j} := \sup\lvert\partial(tG^0_\mu)/\partial Y_j\rvert`$ for j ∈ {P, Φ, C}, and
  $`\gamma_{C_\nu,j} := \sup\lvert\partial(\tfrac12[C(\hat E_0 + \vartheta)]_\nu)/\partial Y_j\rvert`$ for j ∈ {P, C, ϑ}.
  The blocks are written out in `b5_formulas.jac_blocks`; they contain no negative power of t.
- **ε-part**: $`\ell_j \ge \sup\lvert\partial_{Y_j}(g(t,\varepsilon/t,\cdot) - g(t,0,\cdot))\rvert`$, where g = t·(exact Φ-rate) is written
  without division by t (`b5_formulas.g_scaled`) and is analytic in τ := ε/t. By Schwarz's lemma in τ and Cauchy's
  estimate in Y, $`\ell_j \le 4\,G\,\tau_0/(\rho_j\tau_1)`$. Here G = sup|g| over |τ| ≤ τ₁ and a state polydisc of radii
  ρ_j, and τ₀ = sup ε/|t| = n^{−2γ}.
- **Residuals** (Lemma 3): $`R_\Phi = \sup\lvert t\,r_\Phi/\varepsilon\rvert`$, $`R_C = \sup\lvert[C\hat E_0]\rvert/2`$, $`T_F = \sup\lvert t\,\Xi(Y_{\rm in})\rvert`$.
- **ϑ-row Lipschitz constant** Λ_ϑ of tΞ (Cauchy's estimate).

With f₁ = 1/(a − sin κ Z) and f₂ = 1/(a − 2 sin κ Z), the matrix L (unweighted entries, rows and columns over the
components P_μ, Φ_μ, C_ν, ϑ_ν) is:

| row ← column | entry | from |
|---|---|---|
| P_μ ← Φ_μ | Z/2 | slow kernel, integrand W_Φ/(2t) with W_Φ ~ \|t\| |
| Φ_μ ← P_ν, C_ν | γ f₁ + ℓ f₁ | fast kernel, q = −1 |
| Φ_μ ← Φ_ν | γ Z f₂ + ℓ_Φ f₁ | fast kernel, q = 0 (main), q = −1 (ε-part) |
| Φ_μ ← ϑ_ν | ℓ_ϑ f₁ | ε-part only (Lemma 2) |
| C_ν ← P, C, ϑ | γ Z + ε₀(γ + 1)Z | slow kernel, q = 0 |
| ϑ_ν ← all | 2ε₀ Λ_ϑ | b-mode kernel |

The forcing vector is F_Φ = R_Φ f₁, F_C = R_C Z, F_ϑ = 2T_F, F_P = 0.

**Hypothesis (H8)** (the input from Lemma B8; section 7). For the given n:
- (i) Y⁺_out is analytic on a neighbourhood of $`\mathcal W^+_n`$ and analytic in the parameters on 𝒦;
- (ii) on the inner arc A_n, for every harmonic, $`\lvert W_P\rvert \le \delta_P\varepsilon`$, $`\lvert W_\Phi\rvert \le \delta_\Phi\varepsilon r_n`$,
  $`\lvert W_C\rvert \le \delta_C\varepsilon`$ and $`\lvert W_\vartheta\rvert \le \delta_\vartheta\varepsilon`$.

**Lemma B5(1).** Assume (H8), and suppose the constants above (with ε₀ ≥ 1/n and τ₀ ≥ n^{−2γ}) admit a vector x > 0
with

```math
b + Lx \lt x\quad\text{componentwise},\qquad b = F + \delta .
```

Then on $`\mathcal W^+_n \supset \mathcal M^+_n`$

```math
\lvert W_j(z)\rvert \le \frac{x_j\,m_j(z)}{n}\qquad\text{for every component } j .
```

The same holds on 𝓜⁻_n by the reflection symmetry (real data, and 𝒦 closed under conjugation). Since |u| ≤ 1 there,
this implies the bound $`\lvert W\rvert \le C\,m(u)/(n\lvert u\rvert^2)`$ of C_selection.md §10.3, with m the size of the
corresponding component of Z_in and C proportional to max_j x_j.

*Proof.* Let B be the set of points of $`\mathcal W^+_n`$ where some $`\lvert W_j\rvert > x_j\varepsilon m_j`$. Suppose B ≠ ∅.
1. B is relatively open, since W is continuous by (H8)(i). Let τ* = inf_B ℓ, and take U* in the closure of B with
   ℓ(U*) = τ*, by compactness.
2. By continuity some component has $`\lvert W_j(U^*)\rvert \ge x_j\varepsilon m_j`$.
3. Every other point t of the backward paths from U* has ℓ(t) < τ* (Lemma 4.4), so the bound holds there. The data
   satisfy it on A_n, since x_j > b_j ≥ δ_j.
4. In the identities of section 3, estimate the differences ΔG⁰, ΔΔ_ε and the C- and ϑ-rate differences by the mean
   value theorem. This is legitimate because the segment from Y_in(t) to Y_out(t) lies in the bootstrap ball, over which
   the Jacobian blocks are taken. Then estimate the integrals by Lemma 5.
5. This gives $`\lvert W_j(U^*)\rvert \le \varepsilon m_j(U^*)(b_j + (Lx)_j) \lt \varepsilon m_j(U^*)x_j`$, a contradiction. $`\square`$

**Existence of x for every fixed M.**
- As Z → 0 the γ's tend to their values at t = 0, the ε-parts tend to 0, and every loop of L passes through an entry
  proportional to Z. Through P ← Φ: the gain of P → Φ → P is γ_{ΦP}f₁ Z/2. Through C ← ·: the loops P → C → Φ → P
  are O(Z²).
- So the spectral radius of L is $`O(\sqrt Z)`$, and a positive super-solution exists for Z ≤ Z_Λ(M, κ).
- For general M the γ's are suprema of explicit expressions in the inner solution, bounded by Braaksma's Gevrey
  asymptotics on a small sector. This gives some n₀(M) under (H8), without a numerical value.
- For M = 1 every constant is certified (section 5).

## 5. M = 1: the certified constants and n₀(1)

`b5_bounds.py` evaluates every constant in python-flint ball arithmetic (192 bits; lemma S at 320).

### 5.1 Inputs

- **Parameters.** 𝒦 is the complex box (A, ω/ω∞) ∈ x̂ + [−2×10⁻⁶, 2×10⁻⁶]² + i[−10⁻⁶, 10⁻⁶]², with x̂ the centre of the box
  of Theorem A(M = 1). It contains a polydisc of radius 10⁻⁶ around every point of that theorem's real box, so the C¹
  bounds of Theorem B(1) follow by Cauchy's estimate, with a factor 10⁶.
- **Inner solution.** Lemma S is rerun over 𝒦 with R₊ = 0.14² and weight exponent k = 12 (N = 31 terms). Its
  contraction is 0.9519 < 1.
  - A strict run at the box centre checks the structural vanishing of the residual below order N to rounding level.
    Over a box the coefficient balls are too wide to test it.
  - At |z| = 0.0085 the η-bound is 2.7×10⁻⁸ for p, 5.5×10⁻⁷ for v and 6.5×10⁻¹¹ for c (lemma S's components), and
    3.8×10⁻⁴, 7.7×10⁻³ and 9×10⁻⁷ after division by |z|². These are included in every enclosure.
- **Domain.** κ = 0.3, so α_z = 1.871 < 3π/4 − 0.4, and lemma S's D alone covers the working domain:
  $`R_+\sin(\pi/4 - \kappa) = 0.00914 \ge Z = 0.0085`$.
- **Cells.** A disc cell |z| ≤ 0.12 Z and 144 polar cells (6 radial × 24 angular). On each cell the inner quantities P,
  Φ/z, C, (C − 1)/z, Q and Q₀' = S/(2z²) are enclosed from the polynomials. Exact divisions by z and z² are taken after
  checking that the dropped coefficients contain 0. η enters through its weighted bound, with explicit Lipschitz factors
  for the derived quantities.
- **ε₀, τ₀.** The run assumes exactly the n₀ it concludes: ln n ≥ κ₁/(Z cos κ) = 73.89 with κ₁ = 0.6 and γ = 0.3, so
  ε₀ = e^{−73.89} and τ₀ = e^{−44.33}.

### 5.2 Results (κ = 0.3, Z = 0.0085; `b5_bounds_k0.3_Z0.0085.log`)

| quantity | certified bound | double-precision value (`b5_numeric.py`, not rigorous) |
|---|---|---|
| largest entries of L | Φ₋₁ ← C₀ 278; Φ₊₁ ← C₀ 235; Φ₊₁ ← C₂ 173; Φ ← P 101 | 237; 196; 137; 97 |
| contraction (Perron-weighted norm of L) | 0.9663 | 0.937 |
| R_Φ | 91.5 | 32–40 on the domain; 35.18 at z → 0 |
| R_C | 3.43 | 3.23 at z → 0 |
| T_F | 4.15 | |
| ε-part ℓ_j | ≤ 7.6×10⁻¹⁴ (G ≤ 34) | |

The certified componentwise super-solutions (b + Lx < x checked in arb; maxima over harmonics):

| right-hand side b | x_P | x_Φ | x_C | x_ϑ |
|---|---|---|---|---|
| forcing only | 65.8 | 1.53×10⁴ | 2.27 | 8.94 |
| unit data in P | 10.05 | 2130 | 0.235 | 0.001 |
| unit data in Φ | 0.101 | 23.6 | 0.0034 | 0.001 |
| unit data in C | 29.4 | 6911 | 1.74 | 0.001 |
| unit data in ϑ | 0.170 | 39.8 | 0.0098 | 1.002 |

Super-solutions add, so the bound for data δ is $`x = x^{\rm forcing} + \sum_B\delta_B\,x^{(B)}`$. Two cases:
- With the leading data of section 7, δ = (0, 76.56, 3.23, 6.46): x = (169.6, 3.97×10⁴, 8.23, 15.5) (`b5_threshold.py`).
- With the reference data δ_ref = (1, 200, 10, 20), certified directly: x = (392, 9.18×10⁴, 20.8, 29.2). The
  superposition reproduces this to 3×10⁻³ (the check in `b5_threshold.py`; dropping the forcing term gives 0.3).

So, for M = 1, every n ≥ n₀(1) and (H8) with δ ≤ δ_ref:

```math
\lvert W_P\rvert \le \frac{392}{n},\qquad \lvert W_\Phi\rvert \le \frac{9.2\times10^4\,\lvert z\rvert}{n},\qquad \lvert W_C\rvert \le \frac{21}{n},\qquad \lvert W_\vartheta\rvert \le \frac{30}{n}\qquad\text{on }\mathcal W^+_n \supset \mathcal M^+_n .
```

### 5.3 n₀(1)

The conditions on n, all certified or elementary at ln n ≥ 73.89:
1. $`Z_n/\cos\kappa \le Z`$, i.e. ln n ≥ κ₁/(Z cos κ) = 73.89. This one decides.
2. ε ≤ ε₀ and ε/|z| ≤ τ₀ on the working domain, monotone in n.
3. r_n ≤ Z_n: e^{−29.6} ≤ 0.0081.
4. max x ≤ 10¹², the a priori radius used for the state enclosures.
5. n ≥ n₈, the threshold of Lemma B8, which is not known here.

So, with the constants of section 5.2, $`n_0(1) = \max(e^{73.89}, n_8) = \max(1.23\times10^{32}, n_8)`$, and D = n + 2.

Scan over κ, Z (`b5_bounds_scan.log`; κ₁ = 0.6; "certified" means the super-solution for δ_ref is certified and Z is
inside lemma S's coverage; Λ is the Perron-weighted norm of L, for information):

| κ | lemma S coverage R₊ sin(π/4 − κ) | Z | Λ | certified | ln n₀ = κ₁/(Z cos κ) |
|---|---|---|---|---|---|
| 0.25 | 0.0100 | 0.0075 | 0.985 | yes | 82.57 |
| 0.25 | 0.0100 | 0.0082 | 1.045 | no (no super-solution) | — |
| 0.25 | 0.0100 | 0.0097 | 1.173 | no (no super-solution) | — |
| 0.30 | 0.00914 | 0.0085 | 0.966 | yes (reference) | 73.89 |
| 0.30 | 0.00914 | 0.0090 | 1.004 | yes | 69.78 |
| 0.30 | 0.00914 | 0.0091 | 1.012 | no (no super-solution) | — |
| 0.35 | 0.00827 | 0.0075 | 0.818 | yes | 85.16 |
| 0.35 | 0.00827 | 0.0082 | 0.867 | yes | 77.89 |
| 0.35 | 0.00827 | 0.0097 | 0.972 | no (Z outside the coverage) | — |

- Smaller κ widens lemma S's domain but lowers the fast rate a = sin κ/2. At κ = 0.25 the contraction is the limit, at
  κ = 0.35 the coverage is. The best is near κ = 0.3, Z ≈ 0.009.
- The smallest certified value is at Z = 0.009: $`n_0(1) = \max(e^{69.78}, n_8) = \max(2.0\times10^{30}, n_8)`$. There the
  super-solution is about ten times larger (x_Φ = 9.7×10⁵ for δ_ref, against 9.2×10⁴), since Λ is close to 1.
- The reference choice (0.3, 0.0085) keeps a margin and is the one quoted elsewhere in these notes.

### 5.4 What prevents a smaller n₀

- **The constant is Z_max ≈ 0.009**, the largest radius in z = x²/n of a disc around z = 0 on which the fixed point
  closes. n₀ = exp(κ₁/(Z_max cos κ)) because the matching domain must reach |u| = κ₁/ln n with κ₁ > 1/2: the jump
  e^{−1/(2u*)} must dominate W = O(1/n) there.
- **Two limits on Z_max, nearly equal here:**
  - (a) the radius on which lemma S encloses Z⁺_in, R₊ sin(π/4 − κ) ≈ 0.0091. Its own contraction is 0.952 at
    R₊ = 0.14², but 1.055 at R₊ = 0.145² and 1.171 at R₊ = 0.15² (`b5_bounds_rp.log`), so with k = 12 this radius
    cannot grow by more than about 7 %.
  - (b) the contraction of the perturbative splitting, i.e. the loops of the inner variational operator through the
    fast row. The coefficients $`\lvert t\,\partial G^0/\partial P\rvert \approx 14`$ (|3Ê₀ − D₂D|/2 at t = 0) and
    $`\lvert t\,\partial G^0/\partial C\rvert \approx 34`$ enter with the fast-kernel factor 1/(a − sin κ Z) and come back
    through P' = Φ/(2t) and C' = [CÊ₀]/2, each of which gains a factor Z. The norm of the map reaches 1 near
    Z ≈ 0.009 for κ = 0.3. The sampled double-precision value (0.937 at Z = 0.0085) shows the limit is intrinsic.
- **Improving n₀ needs an n-independent outer edge.** Prove B5(1) on a fixed sector up to |z| = Z_b (say 0.03–0.04)
  using the exact inner variational flow there instead of its principal part.
  - The tools exist: A_certify.md §12 covers |z| ≤ 0.04 with 178 validated Taylor discs. Lemma D with fat initial boxes
    would give the propagator by Cauchy estimates.
  - The ε-perturbation on that fixed region is O(1/(nZ_crit)), so the threshold would become polynomial in the
    constants. The jump could then be evaluated at a fixed u* = Z_b, so κ₁ would disappear and B5(2)'s error would be
    O(e^{1/(2Z_b)}/n) directly.
  - B8 would then only need the inner arc |x| = n^γ, not |x| ~ (n/ln n)^{1/2}.
  - This is the natural next step. It was not attempted here.

## 6. Checks against numerics

Each check names what it compares, shares as little as possible with the code it tests, and has a mutation that makes it
fail. All were run and the mutations seen to fail.

| check | script | result | mutation → result |
|---|---|---|---|
| the M-model in inner variables (Lemma 1) vs the outer equations (sympy, per harmonic) | `b5_system.py` | 6×10⁻⁶³ | ε(1−ε)𝒞ℬP dropped → 9×10⁻⁴; (1+ε) → (1−ε) in h → 4×10⁻⁴ |
| ε = 0 limit vs `fixed_point.py`'s inner system | `b5_system.py` | 2×10⁻³⁹ | (shares the formula for W; guarded by the next two) |
| Φ_z formula vs total derivative of the Φ relation | `b5_system.py` | 1×10⁻⁵⁰ | t5 term dropped → 0.70 |
| ε → 0 limit independent of ϑ (Lemma 2) | `b5_system.py` | \|Δ\|\|z\|²/ε ≤ 15 | t5 dropped → 6×10²⁸ |
| scaled field g, G⁰, Jacobian blocks, residual (Lemma 3), ϑ forcing vs `b5_system.Model` | `b5_formulas.py` | ≤ 2×10⁻⁴⁶ | [[CH]P] dropped → 8×10⁻³ |
| residual is O(ε/z) on the solution | `b5_numeric.py residual` | spread 1.04 | off the solution → grows ×3.7 |
| enclosures contain independent values at points (inner solution, residual, Jacobian vs central differences, ϑ-integrand) | `b5_bounds.py check` | excess 0 | residual term dropped → 0.36; P block ×2 → 0.89 |
| the contraction must fail for a wrong Jacobian | `b5_bounds.py --mutate=jac` | — | Φ←P block ×2 → no super-solution (Perron norm 1.27) |
| lemma S must fail for a wrong series coefficient (strict centre run) | `b5_bounds.py --mutate=series` | — | caught |
| leading data φ₁ vs the O(1/n) recursion of notes/FIRST_ORDER.md (`first_order_jet.py`, shares only the equations) | `b5_data.py` | 8×10⁻⁸ | gauge c₀ = 0 on the jet side → 1.8 |
| sampled contraction vs certified | `b5_numeric.py lip` | 0.937 vs 0.966 | (the same Jacobian mutation applies) |
| end-to-end: the exact model integrated from u₀ = r_n e^{iα_z} (ray, then the arc to the Stokes line), n = 10⁸, compared with the certified bound | `b5_path.py` | W/bound: P 2.2×10⁻³, Φ 3.2×10⁻³, C 0.41 | start on the Stokes side → fast mode blows up at \|z\| = 6.6×10⁻⁴; W_Φ/bound = 3.8×10⁹ |
| x for the leading data by superposition; the same superposition for δ_ref vs the directly certified x | `b5_threshold.py` | 3×10⁻³ | forcing term dropped → 0.3 |

On the end-to-end check:
- **What it uses.** The b-mode deviation ϑ is slaved to first order rather than integrated, since its rate n/(2|z|)
  is not integrable numerically along the arc. The data are the leading O(1/n) data of section 7. n = 10⁸ is far below
  n₀(1), so this is numerical evidence that the bound's ingredients behave as computed, not a test of the lemma at its
  n₀.
- **What it shows.** max|W_Φ|/(ε|z|) = 126 is close to |φ₁| = 77 near z = 0, as W ≈ εZ^{(1)} predicts, and
  max|W_C|/ε = 3.40 against |[F²]| = 3.23.
- **The mutation** starts the same integration on the Stokes line with a relative 10⁻¹² free fast mode. That mode grows
  like $`e^{(1/r_n - 1/z)/2}`$, and the bound must be violated. It is: the integration stops at |z| = 6.6×10⁻⁴, just past
  r_n = 6.3×10⁻⁴, with max|W_Φ|/(ε|z|) = 1.5×10¹⁴ (`b5_path.py --mutate=stokes`, in `b5_mutations.log`).

What is not tested:
- a system that is wrong in the same way in `b5_system.py` and in `b5_formulas.py`. The guard is the first check (the
  outer equations, derived independently in sympy) and the FIRST_ORDER cross-check, which shares only the equations.
- The new scripts have not been reviewed independently. A second implementation of `b5_bounds.py` (for example the cell
  bounds by Taylor models) would be the natural check.

## 7. What B5(1) takes from B8, and what it gives B5(2)

**The input from B8, exactly** (hypothesis (H8) of section 4):
- (i) The centre-regular solution of the M-model of section 1.1, with profile F_n = F and frequency ω, continued through
  the upper half x-plane, is analytic on a neighbourhood of $`\mathcal W^+_n`$, analytically in (F, ω) ∈ 𝒦.
  - In the outer variable this is $`n^\gamma \le \lvert x\rvert \le (\kappa_1 n/(\cos\kappa\,\ln n))^{1/2}`$,
    0 ≤ arg x ≤ π/4 + κ/2.
  - That is B8's domain $`\Sigma^+_n`$ with κ₁ replaced by κ₁/cos κ.
- (ii) On the inner arc |x| = n^γ, 0 ≤ arg x ≤ π/4 + κ/2, every harmonic satisfies
  $`\lvert W_P\rvert \le \delta_P/n`$, $`\lvert W_\Phi\rvert \le \delta_\Phi\,n^{-2+2\gamma}`$, $`\lvert W_C\rvert \le \delta_C/n`$,
  $`\lvert W_\vartheta\rvert \le \delta_\vartheta/n`$.
  - For M = 1 the certified bound is stated for δ ≤ δ_ref = (1, 200, 10, 20), and is affine in δ for other data
    (section 5.2).

**Why (ii) is the "no exponentially large fast content" of C_selection.md §10.4.** At the upper end of the arc,
$`z = r_ne^{i\alpha_z}`$, the inner fast mode has $`\lvert e^{-1/(2z)}\rvert = e^{\sin\kappa/(2r_n)}`$. A fast component $`k\,e^{-1/(2z)}`$
of Y⁺_out relative to Z⁺_in is therefore allowed only with $`\lvert k\rvert \le \delta_\Phi\varepsilon r_n e^{-\sin\kappa/(2r_n)}`$, which is
exponentially small. (ii) on the rest of the arc is needed because P, C and ϑ are integrated radially from it (the
b-mode forces radial paths, section 3).

**How (ii) follows from B8** (outline; the constants belong to B8). On the arc, W = (Y_out − T_out) + (T_out − T_in) + (T_in − Z_in):
- T_out is the outer series truncated at ε^{K₀}; T_in is the inner series truncated at the same order.
- **Y_out − T_out** is B8's remainder. In inner variables it must be o(εr_n) in Φ and o(ε) in the others, which
  needs $`(1-2\gamma)(K_0+1)`$ to exceed 2 − 2γ plus the exponents of the change of variables. For γ = 0.3 that is K₀ ≥ 4
  or so, to be fixed by B8's normalization.
- **T_out − T_in** is the sum of the outer coefficients below top degree, $`\sum_{k \le K_0}\sum_{j \lt k}a_{kj}\varepsilon^{k-j}z^j`$. Its
  leading part is εZ^{(1)}(r_n), the O(1/n) inner correction of notes/FIRST_ORDER.md §2 with K₁ = 0 (Z_in carries the
  actual profile) and the light-cone gauge C^{(1)}(0) = −[F²].
- **T_in − Z_in** is lemma S's η, below 10⁻⁴⁰ at |z| = r_n.

The leading data (`b5_data.py`, numerical; cross-checked against `first_order_jet.py` to 8×10⁻⁸):

```math
\delta_P \to 0\ (\lvert W_P\rvert \approx \varepsilon\lvert\phi_1\rvert r_n/2),\qquad \delta_\Phi \to \max\lvert\phi_1\rvert = 76.56,\qquad
\delta_C \to \max\lvert[F^2]\rvert = 3.23,\qquad \delta_\vartheta \to 2\max\lvert[F^2]\rvert = 6.46,
```

where φ₁ = −2(J_ΦC(0)c₀ + R(0)) is the z-coefficient of Φ^{(1)} (φ₁ = 74.93 ± 15.71i for μ = ±1). These are inside δ_ref
with a factor of at least 2.6. (H8)(ii) therefore holds for n ≥ n₀ once B8's remainder and the next-order coefficients
are below the margins.

**What it gives B5(2).** At the evaluation point u* = κ₁/ln n on the Stokes line, $`\lvert W^\pm(u_*)\rvert \le x\,m(u_*)/n`$, against a
jump of size $`\lvert S\,e^{-1/(2u_*)}u_*^{\langle F^2\rangle}\rvert = \lvert S\rvert\,n^{-1/(2\kappa_1)}(\kappa_1/\ln n)^{\mathrm{Re}\langle F^2\rangle}`$. So the
relative error of the jump is of order $`x\,n^{-1+1/(2\kappa_1)}(\ln n/\kappa_1)^{\mathrm{Re}\langle F^2\rangle}`$.
- This is C_selection.md's $`n^{-1+1/(2\kappa_1)}(\ln n)^2`$ with the power of ln n set by ⟨F²⟩ = A²/2 = 3.23 (M = 1) instead of
  2. The normalization $`u_*^{\langle F^2\rangle}`$ of the fast mode should be carried explicitly in B5(2).
- The error enters Theorem B(1) additively, S + R_n, at a zero of S. The implicit function theorem on the certified
  box needs |R_n| and, by Cauchy's estimate with the factor 10⁶ of section 5.1, its derivative small against
  σ_min(DS). That asks for an error of about 10⁻⁷ or less.
- An estimate, not certified (`b5_threshold.py`): with x_Φ = 9.18×10⁴ that needs $`(1 - 1/(2\kappa_1))\ln n \gtrsim 43`$.
  Together with B5(1)'s ln n ≥ κ₁/(Z cos κ), the best choice on the grid is κ₁ = 0.85, giving ln n ≥ 104.7; κ₁ = 1
  gives 123.2. With the leading-data x_Φ = 3.97×10⁴ the optimum is the same, since B5(1)'s condition binds there.
- So the threshold of the full chain B(1) is set by B5(2) together with B5(1)'s Z_max, not by B5(1) alone.

## 8. Status

| item | status |
|---|---|
| M-model with b, c on even \|ν\| ≤ M + 1, in inner variables (Lemma 1) | **checked symbolically** (sympy-derived outer equations at random points, 60 digits) |
| Lemma 2 (inner limit independent of ϑ), Lemma 3 (residual O(ε/z)) | **proved** (by hand, any M); checked numerically for M = 1 |
| Lemma 4 (geometry), Lemma 5 (kernels), the bootstrap | **proved** |
| Lemma B5(1) for fixed M: \|W_j\| ≤ x_j m_j/n given a super-solution | **proved, conditional on (H8)** |
| existence of a super-solution, hence some n₀(M), for each fixed M | **proved, conditional on (H8)**, with Braaksma's Gevrey bounds; no numerical n₀(M) |
| M = 1: all constants, the super-solution, n₀(1) = max(e^{73.89}, n₈) (max(e^{69.78}, n₈) at Z = 0.009) | **certified** (`b5_bounds.py`, python-flint; with lemma S of A_certify.md rerun over the complex box); **conditional on (H8)** and on Braaksma's theorem as in Theorem A(M = 1) |
| leading data δ⁰ of (H8) | **numerical evidence** (`b5_data.py`; cross-checked) |
| (H8) itself | **open**: B8's remainder in these variables, plus the series comparison of section 7 |
| B5(2) error at n₀(1) | not part of this lemma; estimated in section 7 (ln n ≳ 105, `b5_threshold.py`) |

The trusted code: `b5_bounds.py` (new, not reviewed independently), `b5_formulas.py` (new), `fixed_point.py` and
`lemma6_sector.py` (A_certify.md; reviewed in A_review.md, rerun over a different box), python-flint 0.9.

## 9. Proposed text for the other files

**C_selection.md §1** (replace the harmonic sets): "with p and m on the odd harmonics |μ| ≤ M and b and c on the even
harmonics |ν| ≤ M + 1 (the truncation of the inner solvers and of Theorem A(M = 1); with |ν| ≤ M − 1 the inner limit
would be a different problem)".

**C_selection.md §10.3, Lemma B5(M)(1)** (replace the statement): "Assume (H8) (proof/B5_inner.md §7: Y⁺_out analytic on
the working domain $`\mathcal W^\pm_n \supset \mathcal M^\pm`$, and within O(1/n) of Z⁺_in on the inner arc |x| = n^γ). In the
variables (P, Φ = 2zP_z, C, ϑ = Θ − [PP]/2), $`\lvert W_j\rvert \le x_j m_j(u)/n`$ with m_Φ = |u|, m_j = 1 otherwise, where x
is any positive super-solution of the explicit linear inequality b + Lx < x of B5_inner.md §4. A super-solution exists
for n ≥ n₀(M). For M = 1 it is certified for n ≥ max(e^{73.89}, n₈) (κ = 0.3, κ₁ = 0.6, γ = 0.3), with
x = (392, 9.2×10⁴, 21, 30)." Replace the proof sketch by a pointer to B5_inner.md §§2–4.

**C_selection.md §10.4 and §13** (B5(1) row): "**proved conditional on (H8)** for fixed M; **certified** for M = 1 with
n₀(1) = max(e^{73.89}, n₈). The input from B8 is (H8) of B5_inner.md §7: analyticity on the working domain and the
O(1/n) bound on the inner arc, whose upper end is the quantitative 'no exponentially large fast content'. The weak point
is as predicted: the outer edge κ₁/ln n must lie within |u| ≤ 0.009, where the perturbative fixed point closes, so
n₀ = exp(κ₁/0.0085 cos κ). A practical n₀ needs the exact inner variational flow on a fixed sector (B5_inner.md §5.4)."

**PROOF.md §5** (status paragraph, after "B5 part 2"): "B5 part 1, the inner fixed point on the matching domain, is
proved for each fixed M given B8's bound on the inner arc |x| = n^γ ([proof/B5_inner.md](proof/B5_inner.md)), in the
sharper form W = O(1/n) relative. For M = 1 every constant is certified in ball arithmetic and the fixed point holds for
D ≳ 10³²; the threshold is set by the radius |x²/n| ≈ 0.009 on which a perturbative fixed point around the inner series
closes."

**Paper, Table "The lemmas of Theorem B(M)"**, row B5(M)(1): "sectorial identification on $`\mathcal M^\pm`$ | proved given
B8's bound on the inner arc; constants certified for M = 1, n ≥ e^{73.9}".

## 10. Reproduce

In `solver/proof/` (times on the shared 2-core machine under load):

```
python3 b5_system.py                                    # Lemma 1, Lemma 2 checks, ~1 min; exit 0
python3 b5_system.py --mutate=epsCB                     # must be caught (also --mutate=theta, --mutate=t5)
python3 b5_formulas.py                                  # formulas of Lemma 3 etc. vs the model, ~1 min
python3 b5_formulas.py --mutate=res                     # must be caught
python3 b5_numeric.py residual                          # O(eps/z) structure; --mutate=offsol must be caught
python3 b5_numeric.py lip --Z=0.0085 --kappa=0.3        # sampled contraction 0.937 (not rigorous)
python3 b5_bounds.py --kappa=0.3 --Z=0.0085 --json=b5_bounds_k0.3_Z0.0085.json   # the certificate, ~10 s
python3 b5_bounds.py --kappa=0.3 --Z=0.0085,0.009,0.0091     # scan of section 5.3 (also --kappa=0.25, 0.35); exit 1 if any Z fails
python3 b5_bounds.py --kappa=0.3 --Z=0.0085 --rp=0.15         # lemma S at a larger radius (section 5.4); fails, exit 1
python3 b5_bounds.py check                              # point containment; --mutate=res / --mutate=jacP caught
python3 b5_bounds.py --kappa=0.3 --Z=0.0085 --mutate=jac      # must lose the super-solution
python3 b5_bounds.py --kappa=0.3 --Z=0.0085 --mutate=series   # lemma S must fail
python3 b5_data.py                                      # leading data and the FIRST_ORDER cross-check; --mutate=gauge caught
python3 b5_threshold.py                                 # x for the leading data, n0 values, B5(2) estimate; --mutate=sum caught
python3 b5_path.py --n=1e8                              # end-to-end, ~1 min; --mutate=stokes must be caught
```

The saved outputs, all in `solver/proof/`: `b5_bounds_k0.3_Z0.0085.{log,json}`, `b5_bounds_scan.log`, `b5_bounds_rp.log`,
`b5_bounds_check.log`, `b5_system.log`, `b5_formulas.log`, `b5_numeric_residual.log`, `b5_numeric_lip.log`,
`b5_data.log`, `b5_threshold.log`, `b5_path_n1e8.log`, and `b5_mutations.log` (every mutation above, with its verdict).

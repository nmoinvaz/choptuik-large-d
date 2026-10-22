# Lemma B8 with constants: the global continuation of the outer solution

*Notes and proof. Scripts: `solver/proof/b8m_*.py` (the M + 1 harmonic model) and `solver/proof/b8_alg.py`; the
`b8_*.py` scripts without the m are the earlier scalar model, kept for the record and not used below.*

## Contents

- [0. Summary](#0-summary)
- [1. The model, the domain and the paths](#1-the-model-the-domain-and-the-paths)
- [2. The truncated series (Lemma F)](#2-the-truncated-series-lemma-f)
- [3. Weighted bounds and the bound algebra (Lemma W)](#3-weighted-bounds-and-the-bound-algebra-lemma-w)
- [4. The remainder equations (Lemma R)](#4-the-remainder-equations-lemma-r)
- [5. The fast kernel (Lemma K)](#5-the-fast-kernel-lemma-k)
- [6. The bootstrap (Lemma B8)](#6-the-bootstrap-lemma-b8)
- [7. M = 1: the constants and n₀(1)](#7-m--1-the-constants-and-n1)
- [8. What B8 gives B5(1): hypothesis (H8)](#8-what-b8-gives-b51-hypothesis-h8)
- [9. Checks](#9-checks)
- [10. Every fixed M](#10-every-fixed-m)
- [11. Discrepancies and caveats](#11-discrepancies-and-caveats)
- [12. Proposed text for C_selection.md, PROOF.md and the paper](#12-proposed-text-for-c_selectionmd-proofmd-and-the-paper)
- [13. Reproducing](#13-reproducing)

## 0. Summary

Status words as in AGENTS.md. "Certified" below means: every constant is a python-flint ball, the trusted code is
`b8_alg.py`, `b8m_formal.py`, `b8m_atoms.py`, `b8m_bounds.py` (and `b8m_h8.py` for section 8), and the analytic
lemmas it evaluates are proved here by hand.

| statement | status |
|---|---|
| reduced form of the M-model (section 1.2) | proved; checked numerically for M = 1, 3, 5 to 10⁻⁴⁸ (`b8m_model.py`) |
| formal 1/n series, its structure and its matching with the inner series (Lemma F) | proved; checked (`b8m_formal.py`) |
| remainder equations (Lemma R, identities I1 to I4) | proved; checked at random complex points to 3×10⁻⁷⁷ (`b8m_identities.py`) |
| fast-kernel bounds (Lemma K) | proved |
| **Lemma B8(1)**: continuation of the centre-regular solution to the domain 𝒟_n with explicit remainder bounds, for every n ≥ e^{62} | **certified** (K = 4, over B5's parameter box; section 7) |
| **(H8) of B5(1)**, both parts, for every n ≥ e^{73.89} = n₀(1) | **certified** (`b8m_h8.py`; section 8), using lemma S's η bound as certified by `b5_bounds.py` |
| Lemma B8(M) and (H8) for every fixed odd M, for n ≥ some n₀(M) | proved, without a numerical value of n₀(M) (section 10) |

**What is proved for M = 1.** Take the M-model with b and c on |ν| ≤ M + 1 (section 1.1, the model of Theorem A and of
B5(1)), the parameters (A, ω/ω∞) in the complex box 𝒦 of B5(1) section 5.1, κ = 0.3, κ₁ = 0.6, δ = 0.05 and K = 4.
For every n ≥ e^{62}, the centre-regular solution normalized by c(0) = c^K(0) continues analytically to

```math
\mathcal D_n = \Big\{x : \lvert x\rvert \le R'_n,\ 0 \le \arg x \le \alpha\Big\} \setminus \big\{\lvert x - 1\rvert \lt \delta/2\big\},\qquad R'_n = \Big(\frac{\kappa_1 n}{\cos\kappa\,\ln n}\Big)^{1/2},\quad \alpha = \frac\pi4 + \frac\kappa2,
```

analytically in the parameters, and there, with ρ = (1 + |x|)²/n and θ = |x|/(1 + |x|),

```math
\lvert R_V\rvert \le h_V\rho^{K+1},\qquad \lvert R_S\rvert \le h_S\,\theta\rho^{K+1},\qquad \lvert R_\beta - qR_S\rvert \le h_\beta\,\theta\rho^{K+1},\qquad \lvert R_c\rvert \le h_c\rho^{K+1}
```

for the remainder R = Z − Z^K in the variables (V, S, β, c) of section 1.2 (l¹ norms over harmonics). At n₀ = e^{73.89}:
h = (2.415×10⁶, 6.270×10⁶, 6.166×10⁵, 1.666×10⁵). The constants for other n₀ are in section 7.

**The statement delivered to B5(1)** (its hypothesis (H8), in exactly the form of B5_inner.md section 7). For every
n ≥ n₀(1) = e^{73.89}, every (A, ω/ω∞) ∈ 𝒦, κ = 0.3, κ₁ = 0.6, γ = 0.3, with W = Y⁺_out − Z⁺_in in the variables
(P, Φ, C, ϑ) of B5 section 1:
- (i) Y⁺_out is analytic, and analytic in the parameters, on a neighbourhood of
  $`\{n^\gamma \le \lvert x\rvert \le (\kappa_1 n/(\cos\kappa\,\ln n))^{1/2},\ 0 \le \arg x \le \pi/4 + \kappa/2\}`$;
- (ii) on the inner arc |x| = n^γ, 0 ≤ arg x ≤ π/4 + κ/2, every harmonic satisfies

```math
\lvert W_P\rvert \le \frac{8.3\times10^{-5}}{n},\qquad \lvert W_\Phi\rvert \le 79.04\,n^{-2+2\gamma},\qquad \lvert W_C\rvert \le \frac{3.232}{n},\qquad \lvert W_\vartheta\rvert \le \frac{6.464}{n} .
```

So δ = (8.3×10⁻⁵, 79.04, 3.232, 6.464) ≤ δ_ref = (1, 200, 10, 20). Where this differs from B5's wording, or makes it
precise, is listed in section 8.4: the normalization of Y⁺_out (c(0) = c^K(0), the light-cone gauge up to O(n^{−5})),
the phase convention of the profile (immaterial for |W|), and a parameter box that contains 𝒦.

**n₀(1).** B8(1) closes for every n ≥ e^{62} (section 7; it fails at e^{58}), and (H8)(ii) is certified at
ln n₀ = 73.89, which is B5(1)'s own threshold, ln n ≥ κ₁/(Z cos κ) = 73.89 (B5 section 5.1). So B8 does not raise
n₀(1): n₀(1) = e^{73.89} as far as B5(1) and B8(1) are concerned. The constant that blocks smaller n₀ in B8 is the
fast-kernel gain γ_F on the far region (section 7.3).

**Harmonic truncation.** As B5(1), B8 uses b, c on |ν| ≤ M + 1. C_selection.md §1 and the paper's Definition of the
M-model say |ν| ≤ M − 1 (section 11).

## 1. The model, the domain and the paths

### 1.1 The M-model

For odd M let 𝒪 = {odd μ : |μ| ≤ M} and ℰ = {even ν : |ν| ≤ M + 1}. The model is that of B5_inner.md section 1.1 and of
`b8m_model.py`: p, m on 𝒪, b, c on ℰ, every binary product projected onto its result's set,

```math
x c_x = (n-1)[c(b-1)],\qquad x b_x = [b\,[p^2+m^2]] - (n-1)[b(b-1)],
```
```math
x^2p_x - x[c\,p_x] = -\omega x\,p_\theta + (n-1)[[c(b-1)]\,p] - [cp] + \tfrac n2[c(p+m)],\qquad
x^2m_x + x[c\,m_x] = -\omega x\,m_\theta - (n-1)[[c(b-1)]\,m] + [cm] - \tfrac n2[c(p+m)] .
```

The profile enters through the centre: p − m = ε^{1/2}(2F x + O(x³)) at x = 0, ε = 1/n. For M = 1 the profile is
normalized to F₊₁ = F₋₁ = A/2 (`b8m_formal.params`). The model is invariant under θ ↦ θ + θ₀ (harmonic μ multiplied
by e^{iμθ₀}), so this differs from the convention F_{±1} = ∓iA/2 of `fixed_point.py` by θ₀ = π/2. Every bound below is on
moduli per harmonic and is the same in both conventions.

### 1.2 Reduced variables

With p = ε^{1/2}P̃, m = ε^{1/2}M̃, put S = P̃ + M̃, Δ = P̃ − M̃, β = (n − 1)(b − 1), and V = ([cΔ] − xS)/x. On odd vectors
let C f = [cf], 𝔅 f = [[cβ]f] and Ω = diag(iμω). Then, with D = x² − C², Y = V + S and H = ([SS] + [ΔΔ])/2,

```math
xV' = \Omega S,\qquad x\,D\,S' = A\,S - x^2B\,V,\qquad x\beta' + (n-1)\beta + [\beta\beta] = (1-\varepsilon)H + \varepsilon[\beta H],\qquad xc' = [c\beta],
```
```math
A = (n-1)C^2 + C\mathfrak B - x^2\Omega - x^2B,\qquad B = I - \mathfrak B C^{-1} + C\Omega C^{-1},\qquad \Delta = x\,C^{-1}Y .
```

*Proof.* The sum of the p- and m-equations gives x²S' − x[cΔ'] = −xΩS + 𝔅Δ − [cΔ]; with xV = [cΔ] − xS and xC' = 𝔅
this is xV' = ΩS. The difference gives x²Δ' − x[cS'] = −xΩΔ + 𝔅S + (n − 1)[cS]. Substituting Δ = xC⁻¹Y and
eliminating Δ' with the first identity gives the S-equation (this is identity I1 of section 4 with zero residuals).
The b- and c-equations are rewritten with b − 1 = β/(n − 1). ∎ Checked numerically at random complex points for M = 1, 3,
5, against the original equations, to 10⁻⁴⁸ (`b8m_model.py`; `--mutate=sign` fails).

S and β are the fast components (rates of order n), V and c the slow ones.

**Symmetries and normalization.** The reduced system is invariant under (x, c) ↦ (λx, λc). The centre-regular solution
is analytic at x = 0 with S(0) = 0, β(0) = 0, V(0) = [c(0)2F]; it is unique once c(0) is fixed. We fix c(0) = c^K(0),
the value at 0 of the truncated series of section 2 (which is normalized by c^K(1) = 1, the light cone at x = 1 to
all orders). So the remainder vanishes at the centre: R(0) = 0.

### 1.3 Domain, regions and paths

Fix κ = 0.3, κ₁ = 0.6, δ = 0.05, α = π/4 + κ/2 = 0.9354, φ_δ = arcsin(δ/2). The domain 𝒟_n is the closed sector
0 ≤ arg x ≤ α, |x| ≤ R'_n, minus the light-cone disc |x − 1| < δ/2 (section 0). It is simply connected. R'_n carries the
factor 1/cos κ that B5(1) needs (B5_inner.md section 7); C_selection.md §11 has R_n without it.

Weights: X = 1 + |x|, θ = |x|/X, ρ = X²/n. Regions (all n ≥ n₀ at once; `b8m_bounds.Geo`):

| region | where | box used by the algebra |
|---|---|---|
| N0 | \|x\| ≤ r₁ = 0.05 | X ∈ [1, 1.05], θ ≤ 0.0477 |
| N1 | r₁ ≤ \|x\| ≤ x_F = 20 | X ∈ [1.05, 21], θ ∈ [0.0477, 0.953] |
| R | the ray arg x = α, \|x\| ≤ x_F | X ∈ [1, 21] |
| F | x_F ≤ \|x\| ≤ R'_n | X ≥ 21, θ ≥ 0.952, ρ ≤ ρ_max(n₀) |
| A | the inner arc \|x\| = n^γ (section 8 only) | X = 1 + n^γ ≥ 1 + n₀^γ |

**Lemma D (monotonicity in n).** ρ_max(n) = (1 + R'_n)²/n is decreasing for n ≥ e. Every bound below is a sum of terms
c X^a θ^b n^e with e ≤ 0 on a region whose box does not grow with n. So a bound proved at n₀ holds for every n ≥ n₀. ∎

**Paths** (Lemma B8.1 of C_selection.md §11, with the low path added).
- P_x (fast path): from 0 along the ray arg = α to radius |x|, then along the arc of radius |x| to arg x.
- Q_x (low path): from 0 radially at angle max(arg x, φ_δ) to radius |x|, then along the arc of radius |x| (angle at
  most φ_δ) to x.

Both lie in 𝒟_n: the light-cone disc lies in 0 ≤ arg < φ_δ, and at a radius r with |r − 1| < δ/2 the points of 𝒟_n
with arg < φ_δ lie above the disc, so the arc from a larger argument down to them stays outside it. Every point of P_x
and Q_x has modulus ≤ |x|, and the points of modulus |x| are on the final arc.

## 2. The truncated series (Lemma F)

`b8m_formal.py` solves the model order by order in ε = 1/n: Z = Σ_k ε^k Z_k with Z₀ = (V₀, S₀, β₀, c₀) = (2F, 0, 0, 1),
Δ₀ = 2Fx, and for k ≥ 1 the recursion of its docstring (S_k from the Δ-equation, β_k from the β-equation, c_k from
xc_k' = [cβ]_k with c_k(1) = 0, V_k = Ω∫₀^x S_k ds/s + [c_k(0)2F], Δ_k from [cΔ] = x(V + S)). Z^K = Σ_{k≤K} ε^k Z_k.

**Lemma F.** For every odd M and every K:
1. Z_k (k ≥ 1) is a polynomial in x of degree ≤ 2k, even (Δ_k odd, of degree ≤ 2k + 1); S_k, β_k = O(x²).
2. The residuals of Z^K, defined in section 4, are polynomials in (ε, x) with τ_D, τ_b = O(ε^K) and τ_v, ρ₄ = O(ε^{K+1}).
3. Writing ε^k x^{2j} = ε^{k−j} z^j with z = x²/n, the coefficients with j = k (top degree) are the inner formal series
   of A_certify.md (Theorem A's series), with P(0) = 2F, C(0) = 1.

*Proof.* 1 and 2 by induction: each order solves the equations of that order exactly, and the residual of the truncated
sum consists of the cross terms of total order > K. The c-gauge fixes the constant of integration; V_k(0) = [c_k(0)2F]
keeps Δ'(0) = 2F at every order. For 3, substitute x² = nz: the ε⁰ part of the double series solves the ε = 0 limit of
the system in z, which is the inner system (B5_inner.md Lemma 1 at ε = 0), with the same data, and its formal solution
is unique. ∎

*Checks* (`b8m_formal.py --check`): the residual of the original projected equations at the truncated series decays
like n^{−(K+1)} (sum and c-equations) and n^{−K} (difference and b-equations), fitted exponents within 0.15; the top-degree
coefficients agree with `series_arb.py` to 3.7×10⁻⁷⁵ (relative). `--mutate=coef` fails both.

In the bounds, the structural zeros of Lemma F (parity, vanishing order at x = 0, ε-order of the residuals) are imposed
exactly. At the box centre the removed entries are rounding noise (≤ 1.8×10⁻⁶⁶; `b8m_atoms_K4.log`); over the box they
are balls containing 0 whose radius reflects the box width, and Lemma F (an identity in the parameters) removes them.

## 3. Weighted bounds and the bound algebra (Lemma W)

**Lemma W.** Let p = Σ_{j,m} p_{jm} ε^j x^m (vector or operator coefficients) with m ≥ m₀ and m − 2j ≤ w for every
nonzero term. Then for every x and n,

```math
\lvert p(x)\rvert \le \theta^{m_0}X^w\sum_j\Big(\sum_m\lvert p_{jm}\rvert\Big)\rho^j .
```

*Proof.* $`\lvert\varepsilon^jx^m\rvert = \theta^mX^m n^{-j} = \theta^m X^{m-2j}\rho^j \le \theta^{m_0}X^{w}\rho^j`$, since θ ≤ 1 ≤ X. ∎

Norms: l¹ over harmonics for vectors; for operators on odd vectors the induced l¹ norm (largest column sum). A
multiplication operator f ↦ [gf] has norm ≤ |g|₁, and ‖[fg]‖₁ ≤ |f|₁|g|₁.

`b8m_atoms.py` gives these descriptors for the known quantities (and their x d/dx and (x d/dx)²): V, S, S/x², β, β/x²,
c − 1, [cβ], [cβ]/x², Y, Δ^K, the operators C, Γ = (C² − I)/(x² − 1), det C, the transport η̃ (section 5) and the residuals
τ_D, τ_v, τ_b, ρ₄. Leading entries at K = 4 (sums over m of the coefficient of ε^j; `b8m_atoms_K4.log`):

| quantity | j = 0 | j = 1 | j = 2 | j = 3 | j = 4 | j = 5 |
|---|---|---|---|---|---|---|
| V | 5.084 | 48.3 | 526 | 6.0×10³ | | |
| S | | 15.9 | 254 | 4.0×10³ | 5.8×10⁴ | |
| β | | 12.9 | 254 | 3.3×10³ | 4.9×10⁴ | |
| c − 1 | | 12.9 | 177 | 2.1×10³ | 2.5×10⁴ | |
| Γ | | 9.69 | 191 | 2.8×10³ | 3.8×10⁴ | |
| det C · η̃ | 11.3 | 283 | 6.1×10³ | 1.1×10⁵ | | |
| τ_D, τ_b | | | | | 9.3×10⁵, 7.9×10⁵ | |
| τ_v, ρ₄ | | | | | | 1.3×10⁶, 9.3×10⁵ |

**The bound algebra** (`b8_alg.py`). An upper bound is kept per region as a sum of monomials c X^a θ^b n^e. Terms with
the same (b, e) are merged into the larger X-power, the other coefficient multiplied by X_lo^{a'−a} ≤ 1. On the far
regions θ is replaced by its bound at once. The supremum over a region for all n ≥ n₀ uses X^a ≤ X_hi^a (a > 0) or
X_lo^a, θ^b ≤ θ_hi^b or θ_lo^b, n^e ≤ n₀^e (e ≤ 0), and on F also X^{a+2e}ρ^{−e} with ρ ≤ ρ_max; anything else stops the
computation. Inverses of operators I + E use ‖(I + E)⁻¹‖ ≤ 1/(1 − sup‖E‖), with the lower bound checked positive.
Products of bounds with derivative bounds follow the product rule. Every constant is a ball; only its upper (or lower)
end is used.

## 4. The remainder equations (Lemma R)

Superscript K marks quantities built from Z^K (with Δ^K the series Δ, not xC^{K,−1}Y^K). The residuals of Z^K are

```math
\tau_D = xC S' - x^2\Delta' - x\Omega\Delta + \mathfrak BS + (n-1)CS,\qquad \tau_v = C\Delta - xY,
```
```math
\tau_b = x\beta' + (n-1)\beta + [\beta\beta] - (1-\varepsilon)H - \varepsilon[\beta H],\qquad \rho_4 = xc' - [c\beta],
```

all evaluated at Z^K. Let Z be any function and R = Z − Z^K, and define the defects of Z: defect_S = xDS' − (AS − x²BV),
defect_β = xβ' + (n − 1)β + [ββ] − (1 − ε)H − ε[βH] (with Δ = xC⁻¹Y), defect_c = xc' − [cβ]. For the solution all
three vanish.

**Lemma R.** With Λ = (D^K)⁻¹A^K and δD = D − D^K:

- (I1) $`xD^KS^{K\prime} - A^KS^K + x^2B^KV^K = \rho_2`$, with
  $`\rho_2 = x^2M_{\rho_4}C^{-1}Y - C\tau_D - x^2\tau_v' + x(xc')C^{-1}\tau_v - xC\Omega C^{-1}\tau_v`$ (all at Z^K).
- (I2) $`xR_S' = \Lambda(R_S + G_1) + xW_2 + D^{-1}\,\mathrm{defect}_S`$, where, with A₀, B₀ the operators A, B evaluated
  at (β^K + qR_S, c),

```math
G_1 = (A^K)^{-1}\big[(A_0 - A^K)S - x^2(B_0V - B^KV^K) - \delta D\,D^{-1}(A_0S - x^2B_0V) - \rho_2\big],\qquad
W_2 = (xD)^{-1}\big(C M_hS + x^2M_hC^{-1}Y\big),
```

  M_h f = [[c R̂_β] f], R̂_β = R_β − qR_S, and the decoupling operator (odd → even vectors)

```math
q = \ell\,\big((n-1)I + \Lambda\big)^{-1} = \ell\,(I + e)^{-1}\frac{D^K}{(n-1)x^2},\qquad \ell f = (1-\varepsilon)\big[b^K([S^Kf] + [\tilde\Delta\,xC^{K,-1}f])\big],
```

  with $`\tilde\Delta = xC^{K,-1}Y^K`$, b^K = 1 + β^K/(n − 1) and $`e = (C^K\mathfrak B^K/x^2 - \Omega - B^K)/(n-1)`$.
- (I3) $`xR̂_\beta' + (n-1)R̂_\beta = \hat g + \mathrm{defect}_\beta - qD^{-1}\mathrm{defect}_S`$, where

```math
\hat g = -[R_\beta(2\beta^K + R_\beta)] + \varepsilon[R_\beta H] + (1-\varepsilon)[b^Kg_3] - xq'R_S - q(\Lambda G_1 + xW_2) - \rho_3,
```
```math
g_3 = \tfrac12[R_SR_S] + [\tilde\Delta\,(xC^{-1}R_V - C^{-1}R_C(xC^{K,-1}R_S + \tilde\Delta))] + \tfrac12[\delta\Delta\,\delta\Delta],\qquad \delta\Delta = xC^{-1}(R_V + R_S) - C^{-1}R_C\tilde\Delta,
```
```math
\rho_3 = \tau_b + \tfrac{1-\varepsilon}2\big[b^K[(C^{K,-1}\tau_v)(\tilde\Delta + \Delta^K)]\big],
```

  R_C f = [R_c f].
- (I4) $`xR_c' = [R_c\beta] + [c^KR_\beta] - \rho_4 + \mathrm{defect}_c`$, and $`xR_V' = \Omega R_S`$ for the solution.

*Proof.*
- (I1) Repeat the derivation of section 1.2 with the residuals kept: from τ_v, Δ = C⁻¹(xY + τ_v); differentiate, use
  xV' = ΩS (exact for every truncation) and multiply τ_D by C. The operator xC' appears; for the truncated series
  xC^{K}' = 𝔅^K + M_{ρ₄}, which gives the first term of ρ₂.
- (I2) x D^K S' = (I − δD D⁻¹)·xDS' for any Z; subtract (I1) and split A − A^K, B − B^K into the part with β replaced by
  β^K + qR_S and the R̂_β-part, which is C M_h + x²M_hC⁻¹ for A and −M_hC⁻¹ for B. The R̂_β-part enters multiplied by
  I − δD D⁻¹ = D^K D⁻¹; dividing by D^K gives xW₂. The two forms of q agree because
  (n − 1)I + Λ = (n − 1)(D^K)⁻¹x²(I + e).
- (I3) Subtract the truncated β-equation, expand H − H^K with δΔ = Δ − Δ̃ and Δ^K = Δ̃ + C^{K,−1}τ_v, and collect the
  part linear in R_S: it is ℓR_S. Then substitute R_β = R̂_β + qR_S and xR_S' from (I2): q((n − 1)I + Λ)R_S = ℓR_S
  cancels it.
- (I4) Direct. ∎

*Checks* (`b8m_identities.py`, mpmath at 80 digits, random complex x and n, random R and R', for M = 1 with K = 3, 4 and
M = 3 with K = 2): every identity holds to a relative residual ≤ 3×10⁻⁷⁷ (`b8m_identities.log`). Mutations
(`b8m_identities_mut.log`): ρ₂ without its first term fails I1 by 0.07 to 0.36; q without (I + e)⁻¹ fails I2 and I3.

**Correction made in this pass.** The first term of ρ₂ was missing from `b8m_bounds.py` until commit 610cc68; it was
found by I1. Every closure run made before it was discarded and redone.

## 5. The fast kernel (Lemma K)

Write Γ = (C^{K2} − I)/(x² − 1) (an exact polynomial division), so D^K = (x² − 1)(I − Γ), and
A^K = (n − 1)(I − Γ) + η with η = x²η̃. Then

```math
\frac{\Lambda}{x} = \lambda(x)\,(I + H'),\qquad \lambda(x) = \frac{n-1}{x(x^2-1)} = (n-1)\,\Phi'(x),\quad \Phi(x) = \tfrac12\ln(1 - x^{-2}),\qquad (n-1)H' = (I-\Gamma)^{-1}\eta .
```

Let E(x, s) be the propagator of R' = (Λ/x)R from s to x. On every path, Grönwall's inequality with the log-norm gives
ln‖E(x, s)‖ ≤ ∫_s^x (Re(λ dt) + |λ dt|‖H'‖), and ‖H'‖ ≤ ‖η‖/((n − 1)(1 − ‖Γ‖)).

**Lemma K.**
1. *(Ray.)* On t ↦ te^{iα}, $`\mathrm{Re}(\lambda\,dx) = -(n-1)\frac{1 + t^2\sin\kappa}{t\lvert D_t\rvert^2}dt`$ with
   $`\lvert D_t\rvert = (1 + 2t^2\sin\kappa + t^4)^{1/2}`$, and (1 + t² sin κ)/|D_t| ≥ sin κ. So for s, x on the ray,
   $`\lVert E(x,s)\rVert \le \exp\big(-(n-1)s_*\int_{\lvert s\rvert}^{\lvert x\rvert}\frac{dt}{t\lvert D_t\rvert}\big)`$ with the decay
   margin $`s_* = \sin\kappa - \sup_{\rm ray}\lVert H'\rVert \gt 0`$.
2. *(Arc.)* On |s| = r with arguments φ_s ≥ φ_x, u = φ_s − φ_x ≤ α:
   $`(n-1)\,\mathrm{Re}(\Phi(x) - \Phi(s)) \le -N_r s_\alpha^2u^2`$, with $`N_r = (n-1)r^2/Q_r`$, $`Q_r = 1 + 2r^2\sin\kappa + r^4`$
   and s_α = sin α/α; and $`\int\lvert\lambda\,dt\rvert\lVert H'\rVert \le H_ru`$ with
   $`H_r = \sup\lVert\eta\rVert/((1 - \lVert\Gamma\rVert)\lvert s^2 - 1\rvert)`$. So $`\ln\lVert E\rVert \le H_ru - N_rs_\alpha^2u^2`$.
3. *(Gains.)* For a weight w(s) = θρ^{K+1} (increasing in |s|) and g with |s g(s)| ≤ I w(s) on P_x,
   $`\int_{P_x}\lVert E(x,s)\rVert\,\lvert g(s)\rvert\,\lvert ds\rvert \le \gamma\,I\,w(x)`$, where on F

```math
\gamma_F = K_F\,\frac{\sqrt{q_F}\,\rho_{\max}}{s_*}\frac{n_0}{n_0-1} + K_F\sqrt{\frac{\pi}{N_Fs_\alpha^2}},\qquad K_F = \exp\Big(\frac{H_F^2}{4N_Fs_\alpha^2}\Big),
```

   $`q_F = 1 + 2\sin\kappa/x_F^2 + x_F^{-4}`$, $`N_F = (1 - 1/n_0)/(\rho_{\max}q_F)`$, $`H_F = \sup_F\lVert\eta\rVert\ell_F/(X^2(1 - \lVert\Gamma\rVert))`$,
   ℓ_F = X_F/(x_F − 1) ≥ X²/|x² − 1|. On N0 and N1 the same two terms are evaluated on 300 radial cells with the exact
   r-dependence, and on cells where N_r is too small the arc term α e^{H_rα} is used instead.

*Proof.*
1. With e^{2iα} = −sin κ + i cos κ, t²e^{2iα} − 1 = −(1 + t² sin κ) + it² cos κ, which gives Re(λ dx). The inequality
   (1 + t² sin κ)² ≥ sin²κ (1 + 2t² sin κ + t⁴) reduces to cos²κ (1 + 2t² sin κ) ≥ 0. And |λ dx| = (n − 1)dt/(t|D_t|).
2. $`4\,\mathrm{Re}\,\Phi(re^{i\varphi}) = \ln\big((1 - r^{-2})^2 + 4\sin^2\varphi/r^2\big) = \ln(\lvert s^2-1\rvert^2/r^4)`$. With ln(1 − y) ≤ −y,
   (n − 1)Re(Φ(x) − Φ(s)) ≤ −(n − 1)r²(sin²φ_s − sin²φ_x)/|s² − 1|², and
   sin²φ_s − sin²φ_x = sin u sin(φ_s + φ_x) ≥ sin²u ≥ s_α²u² (u ≤ φ_s + φ_x ≤ π − u, concavity of sin on [0, α]).
   |s² − 1|² ≤ Q_r because cos 2φ ≥ cos 2α = −sin κ. On the arc |λ dt| = (n − 1)du/|s² − 1|.
3. Split P_x into the ray (to re^{iα}) and the arc; ‖E(x, s)‖ ≤ ‖E(x, re^{iα})‖‖E(re^{iα}, s)‖. On the arc,
   max_u (H u − N s_α²u²) = H²/(4N s_α²) = ln K, and ∫ e^{Hu − Ns_α²u²}du ≤ K√(π/(Ns_α²)), with w(s) = w(x) and
   |ds|/|s| = du. On the ray, with Ψ' = 1/(t|D_t|), $`\int_0^re^{-(n-1)s_*(\Psi(r)-\Psi(t))}\frac{dt}{t} \le \frac{\lvert D_r\rvert}{(n-1)s_*}`$,
   w(t) ≤ w(r), and |D_r| ≤ r²√q_F ≤ nρ_max√q_F for r ≥ x_F. ∎

At n₀ = e^{73.89} (K = 4, the box): s_* = 0.1875, H_F = 14.04, N_F = 117.5, K_F = 1.763, γ_F = 0.4150,
γ_N0 = 0.9354, γ_N1 = 3.7×10⁻¹⁵ (`b8m_scan_K4_box.log`). The kernel constants absorb the transport factor
|x/s|^{2Re⟨F²⟩} of C_selection.md §11: it is part of η (through (n − 1)Γ), so no separate condition on K is needed.

## 6. The bootstrap (Lemma B8)

**Hypothesis (B_h)** on a set U ⊂ 𝒟_n: Z is analytic on U and, with w₁ = ρ^{K+1}, w = θρ^{K+1},

```math
\lvert R_V\rvert \le h_Vw_1,\qquad \lvert R_S\rvert \le h_Sw,\qquad \lvert R̂_\beta\rvert \le h_\beta w,\qquad \lvert R_c\rvert \le h_cw_1 .
```

**Lemma B8.a (improvement; proved).** Suppose (B_h) holds on P_x ∪ Q_x. Then at x the four bounds hold with h replaced
by P(h) = (P_V, P_S, P_β, P_c), where:

- Derivative bounds first (pass 1, from the equations of Lemma R with the defects zero): |xR_V'| ≤ |ω|h_S w,
  |xR_S'| ≤ ‖Λ‖(|R_S| + |G₁|) + |x||W₂|, |xR̂_β'| ≤ |ĝ| + (n − 1)|R̂_β|, |xR_c'| ≤ |R_c||β| + |c^K||R_β| + |ρ₄|, with
  R_β = R̂_β + qR_S, and every operator bounded from the descriptors of section 3, the hypothesis and the lower bounds
  of C and D (checked positive).
- Then (pass 2), with these derivative bounds in the product rule:

```math
P_V = \lvert\omega\rvert\Big(\frac1{2K+2} + \varphi_\delta\Big)h_S,\qquad
P_c = \Big(\frac1{2K+2} + \varphi_\delta\Big)\sup\frac{\lvert R_c\rvert\lvert\beta\rvert + \lvert c^K\rvert\lvert R_\beta\rvert + \lvert\rho_4\rvert}{w},
```
```math
P_S = \max_{r}\Big[\sup_r\frac{\lvert G_1\rvert}{w} + \gamma_r\sup_{P(r)}\frac{\lvert xG_1'\rvert + \lvert x\rvert\lvert W_2\rvert}{w}\Big],\qquad
P_\beta = \max_r\Big[\sup_r\frac{\lvert\hat g\rvert}{(n-1)w} + \Big(\frac{1}{n_0-1} + \varphi_\delta\Big)\sup_{Q(r)}\frac{\lvert x\hat g'\rvert}{(n-1)w}\Big],
```

  r over N0, N1, F; P(r) and Q(r) the regions the paths P_x, Q_x of a target in r meet (N0: N0; N1: R, N1 for P and
  N0, N1 for Q; F: R, F for P and N0, N1, F for Q).

*Proof.*
- **V and c (slow).** R_V(0) = R_c(0) = 0 (section 1.2). Integrate xR_V' = ΩR_S and (I4) along Q_x. On the radial part
  $`\int_0^r\theta\rho^{K+1}dt/t = n^{-K-1}\int_0^r(1+t)^{2K+1}dt \le \rho(r)^{K+1}/(2K+2)`$; on the arc the integrand is at
  most w(x) ≤ ρ^{K+1} over an angle ≤ φ_δ.
- **S (fast).** By (I2) and variation of constants from x = 0 along P_x, where R_S(0) = 0 and E(x, s) → 0 as s → 0:
  $`R_S(x) = \int_{P_x}E(x,s)\big[\tfrac{\Lambda(s)}{s}G_1(s) + W_2(s)\big]ds`$. Since ∂_s E(x, s) = −E(x, s)Λ(s)/s,
  integration by parts gives $`R_S(x) = -G_1(x) + \int_{P_x}E(x,s)\big[G_1'(s) + W_2(s)\big]ds`$, and Lemma K.3 bounds
  the integral.
- **R̂_β (b-mode).** By (I3), $`R̂_\beta(x) = \int_{Q_x}(s/x)^{n-1}\hat g(s)\,ds/s`$, and $`\lvert(s/x)^{n-1}\rvert \le 1`$ on Q_x.
  Integration by parts with d(s/x)^{n−1} = (n − 1)(s/x)^{n−1}ds/s gives
  $`R̂_\beta(x) = \frac{\hat g(x)}{n-1} - \frac1{n-1}\int_{Q_x}(s/x)^{n-1}\hat g'(s)\,ds`$. On the radial part
  $`\int_0^r(t/r)^{n-1}dt/t = 1/(n-1) \le 1/(n_0-1)`$; on the arc the factor is the angle, ≤ φ_δ. ∎

**Lemma B8.b (continuation; proved).** If P(h) < h componentwise at n₀ (with the bounds valid for all n ≥ n₀ by Lemma D),
then for every n ≥ n₀ the centre-regular solution continues analytically along the paths to all of 𝒟_n, analytically
in the parameters, and (B_h) holds on 𝒟_n. It extends analytically to a neighbourhood of 𝒟_n.

*Proof.* Let r* be the supremum of the r ≤ R'_n such that the solution continues to 𝒟_n ∩ {|x| ≤ r} and (B_h) holds
there.
- Near the centre the solution is analytic (x = 0 is a regular singular point; the centre-regular solution is a
  convergent series in x²), and R = O(x²) while w ≥ c|x| and w₁ ≥ n^{−K−1}, so (B_h) holds strictly on a small disc:
  r* > 0.
- On 𝒟_n ∩ {|x| ≤ r*} the bounds hold by continuity, and the operators C and D stay invertible there (their lower
  bounds are part of P). The right-hand side is analytic near every such point (x ≠ 0, D invertible), so the solution
  continues to a neighbourhood, single-valued because 𝒟_n is simply connected.
- For a point x with |x| ≤ r*, P_x and Q_x lie in {|s| ≤ |x|}, so Lemma B8.a applies and gives the strict bound P(h) < h at
  x. By continuity (B_h) holds on 𝒟_n ∩ {|x| ≤ r* + η} for some η > 0. So r* = R'_n.
- The constants are uniform in the parameters, and the solution of an analytic ODE with analytic data is analytic in
  the parameters. ∎

**Lemma B8(M).** If P(h) < h, then for every n ≥ n₀ the conclusions of Lemma B8.b hold. In the variables of
C_selection.md §11 (p, m, b, c): p ± m = ε^{1/2}(S ± Δ)/2 with Δ − Δ^K = xC⁻¹R_Y − C⁻¹R_CΔ̃ − C^{K,−1}τ_v, so
|R_{p,m}| ≤ C ε^{1/2}(1 + |x|)ρ^{K+1}, which is the form stated there (with the factor ε^{1/2} of the scaling).

## 7. M = 1: the constants and n₀(1)

### 7.1 The certificate

`b8m_scan.py 4 L0,... --rA=2e-6 --rW=2e-6` (atoms over a parameter box that contains 𝒦: A ∈ A* ± 2×10⁻⁶(1 + i),
ω/ω∞ ∈ w* ± 2×10⁻⁶(1 + i), with B5's centre and ω∞ = 2π/1.64389375). For each L0 = ln n₀ it iterates h ↦ P(h) from 0,
then tests closure by descending from 1.1 times the iterate: h_{k+1} = 1.01·P(h_k) until 1.01·P(h_k) ≤ h_k, which
gives P(h_k) < h_k strictly. (The plain test P(sh*) < sh* cannot work: P_V is linear and homogeneous in h_S.)

| ln n₀ | ρ_max | s_* | K_F | γ_F | h_V | h_S | h_β | h_c | P(h)/h (V, S, β, c) |
|---|---|---|---|---|---|---|---|---|---|
| 73.89 | 0.00850 | 0.1875 | 1.763 | 0.415 | 2.415×10⁶ | 6.270×10⁶ | 6.166×10⁵ | 1.666×10⁵ | 0.964, 0.943, 0.990, 0.937 |
| 70 | 0.00897 | 0.1807 | 1.835 | 0.449 | 2.804×10⁶ | 7.383×10⁶ | 6.166×10⁵ | 1.786×10⁵ | 0.977, 0.958, 0.990, 0.943 |
| 66 | 0.00952 | 0.1728 | 1.924 | 0.493 | 3.581×10⁶ | 9.436×10⁶ | 6.529×10⁵ | 1.985×10⁵ | 0.978, 0.984, 0.984, 0.986 |
| 62 | 0.01013 | 0.1636 | 2.033 | 0.548 | 7.159×10⁶ | 1.908×10⁷ | 1.088×10⁶ | 3.570×10⁵ | 0.989, 0.989, 0.989, 0.989 |

(`b8m_scan_K4_box.log`.) Each line is a certificate: Lemma B8(1) holds for every n ≥ n₀ with those h. At ln n₀ = 58
the iteration does not converge: h grows until the lower bound of C is lost ("c lower bound" in the log). So B8(1)'s
own threshold, with these settings (K = 4, κ = 0.3, κ₁ = 0.6, δ = 0.05, x_F = 20, r₁ = 0.05), lies between 58 and 62.

### 7.2 Where the bound comes from (ln n₀ = 73.89)

- P_β = 6.10×10⁵ is almost all forcing: on N1 the local term |ĝ|/((n − 1)w) is 5.1×10⁵, and P_β is the same at
  ln n₀ = 73.89 and 70 although h_S changes by 18%.
- P_S = 5.91×10⁶ comes from F: local 5.2×10⁵ plus γ_F = 0.415 times 1.30×10⁷ for the integrand (|xG₁'| + |x||W₂|)/w.
  On N1 the integrand is larger (2.4×10⁸) but γ_N1 = 3.7×10⁻¹⁵.
- P_V = |ω|(1/10 + φ_δ)h_S and P_c follow.

At n = 10³ the actual remainder on the sample paths reaches 20% of this bound (section 9, check (B)).

### 7.3 What blocks a smaller n₀ in B8

The loop S → β̂ → S closes through W₂ (the R̂_β-terms of the S-equation) on F, with gain γ_F, and γ_F grows as ρ_max
grows: γ_F ≈ K_F√(π/(N_Fs_α²)) with N_F ∝ 1/ρ_max ∝ ln n₀, and K_F = exp(H_F²/(4N_Fs_α²)). The forcing of S grows the
same way. So the binding constant is the far-region fast-kernel gain γ_F (with K_F); between ln n₀ = 73.89 and 62 it
grows from 0.415 to 0.548 and h_S triples. The first things to sharpen would be its arc part (the bound
sin²φ_s − sin²φ_x ≥ s_α²u², and N_F taken at ρ_max on all of F) and the size of the S-forcing on F. None of this
matters for n₀(1), which B5(1) fixes at e^{73.89}.

## 8. What B8 gives B5(1): hypothesis (H8)

### 8.1 Part (i)

B5's working domain in x is n^γ ≤ |x| ≤ R'_n, 0 ≤ arg x ≤ α. For n ≥ n₀ it lies in 𝒟_n (n^γ ≥ e^{22} > 1 + δ/2), so
(i) is Lemma B8.b. "Analytic in the parameters on 𝒦" holds because the certificate is computed over a box containing 𝒦.

### 8.2 Part (ii): the three pieces

B5's variables in terms of section 1.2 (ξ = x/√n, z = x²/n): y = p − m = ε^{1/2}Δ, so

```math
P = \frac{\Delta}{x} = C^{-1}Y,\qquad \Phi = 2zP_z = xP' = C^{-1}\big(xY' - [xc'\,P]\big),\qquad C = c,\qquad
\Theta = \frac{\beta}{(1-\varepsilon)z},\qquad \vartheta = \Theta - \tfrac12[PP],
```

with xY' = ΩS + xS' and xc' = [cβ]. On the arc |x| = n^γ write W = (Y_out − T) + (T − Z_in), T the series Z^K
converted by these formulas.

**(R) The remainder Y_out − T** (`b8m_h8.py`, part R). With R_C f = [R_c f] and P^K = C^{K,−1}Y^K:

```math
W^R_P = C^{-1}\big(R_V + R_S - [R_cP^K]\big),\qquad W^R_\Phi = x\,(W^R_P)',\qquad W^R_C = R_c,\qquad
W^R_\vartheta = \frac{R_\beta}{(1-\varepsilon)z} - \tfrac12\big[W^R_P(2P^K + W^R_P)\big],
```

bounded in the algebra on the region A (X = 1 + n^γ) from the certified h, with the derivative bounds of pass 1 for
xR_S' and xR_c'. A term cX^an^e of a bound gives $`n^s cX^an^e \le c(1 + n_0^{-\gamma})^{\max(a,0)}n_0^{\gamma a + e + s}`$ for
all n ≥ n₀ when γa + e + s ≤ 0 (s = 1 for P, C, ϑ; s = 2 − 2γ for Φ); the script stops otherwise. The Φ-part needs
K ≥ 4: xR_S' ~ (n/x²)R_S ~ h_Sρ^K, and ρ^K = n^{−(1−2γ)K} must beat n^{−2+2γ}, i.e. 0.4K > 1.4.

**(T) The series part T − Z_in** (part T). On the arc, ε^k x^{2j} = ε^{k−j}z^j has size n^{−wt} with
wt = (k − j) + (1 − 2γ)j. T is computed as a Taylor model in (ε, z): the monomials of weight ≤ 2.6 exactly (balls), the
rest as a remainder that scales at least like n^{−2.8}; C^{−1} as a Neumann series with its tail, 1/(1 − ε) as a
geometric series with its tail.
- By Lemma F.3 the ε⁰ part of T equals the inner series up to z^K for P, Φ, C and up to z^{K−1} for ϑ (Θ = β/z needs
  one more order of β). These terms are removed. At the box centre they agree with `series_arb.py` to 10⁻⁶⁹
  (`b8m_h8.py --centre`).
- Z_in = Y_N + η with N = 31 (lemma S as rerun by `b5_bounds.py` over 𝒦). The inner coefficients K < j ≤ 30 are
  subtracted with their balls.
- What remains has weight ≥ 1 (≥ 1.4 for Φ). Its leading terms are the O(1/n) inner correction:

| component | leading monomial | coefficient (per harmonic) | B5's value (`b5_data.py`) |
|---|---|---|---|
| Φ_{±1} | ε z | −15.714 ± 74.930 i, modulus 76.560 | φ₁ = 74.93 ± 15.71 i, 76.56 |
| C₀, C_{±2} | ε | −3.2313, −1.6156 | −[F²], max 3.23 |
| ϑ₀, ϑ_{±2} | ε | −6.4625, −3.2313 | −2[F²], max 6.46 |
| P | ε z | φ₁/2 (Φ = 2zP_z), so δ_P ∝ n₀^{−0.4} | δ_P → 0 |

  The factor i between the Φ columns is the θ-rotation of section 1.1. The independent computation in B5 agrees to all
  digits given.

**(η) Lemma S's term.** B5's certified bound (`b5_bounds_k0.3_Z0.0085.log`): |η| ≤ 5.5×10⁻⁷ at |z| = 0.0085, scaling as
|z|^{12} (p, v) and |z|^{13} (c), the weights of `fixed_point.py`'s norm with k = 12. On |z| = n^{2γ−1} ≤ e^{−29.6} this is
below 10⁻¹⁰⁰/n; for Φ = v + [(C − 1)Q] (B5 Lemma 1 at ε = 0) a factor 100 covers |1 + s| ≤ 5 and |Q| ≤ 14. ϑ_in ≡ 0.

### 8.3 Result (`b8m_h8_K4_L73.89.log`)

K = 4, ln n₀ = 73.89, γ = 0.3, the box; the bootstrap re-verified there (P(h) < h).

| | remainder (R) | series (T) | η | δ (sum, rounded up) | δ_ref |
|---|---|---|---|---|---|
| P | 7.8×10⁻²⁶ | 8.23×10⁻⁵ | 4×10⁻¹⁰⁴ | 8.3×10⁻⁵ | 1 |
| Φ | 2.46 | 76.568 | 3×10⁻⁸⁹ | 79.04 | 200 |
| C | 1.4×10⁻²⁷ | 3.2313 | 4×10⁻¹⁰⁴ | 3.232 | 10 |
| ϑ | 3.4×10⁻¹⁴ | 6.4632 | 0 | 6.464 | 20 |

(The P series entry is the width of a ball over the box: at the centre the ε-coefficient of P is exactly 0, since
P(0) = 2F at every order.) **So (H8)(ii) holds for every n ≥ e^{73.89}, every parameter in 𝒦, with δ ≤ δ_ref.**

### 8.4 Where this differs from B5's wording

1. **The normalization of Y⁺_out.** B5 speaks of "the" centre-regular solution with profile F. For given F and ω the
   centre-regular solutions form a family parametrized by the vector c(0) (on ℰ), which contains the scaling
   (x, c) ↦ (λx, λc). B8 fixes c(0) = c^K(0), the value at 0 of the series normalized by c_k(1) = 0 on every harmonic,
   i.e. the light-cone gauge to all orders in the series. B5's leading data assume this gauge (C^{(1)}(0) = −[F²]); the
   δ_C, δ_ϑ above are for it. If another normalization c(0) = c^K(0) + d with |d| ≤ d₀n^{−K−1} is wanted (for instance
   an exact condition at the light cone), the proof is unchanged except that R_V(0) = [d2F] and R_c(0) = d enter the
   slow integrals: P_V and P_c increase by 2|F|₁d₀ and d₀ (since ρ^{K+1} ≥ n^{−K−1}). This extension is proved by hand
   but was not run.
2. **Profile phase.** B8 uses F_{±1} = A/2, B5 (`fixed_point.py`) F_{±1} = ∓iA/2: a rotation θ₀ = π/2, which multiplies
   harmonic μ by e^{iμθ₀} in both Y_out and Z_in. The moduli |W_μ| are unchanged.
3. **Parameter box.** Certified over A* ± 2×10⁻⁶(1 + i) and w* ± 2×10⁻⁶(1 + i), which contains 𝒦 (imaginary parts
   ±10⁻⁶ there).
4. **The domain of (i)** is a subset of 𝒟_n, which also contains the segment (0, n^γ) and the arc around the light cone.
   B8 makes no statement inside |x − 1| < δ/2; B5(1) does not need one.
5. **Range of n.** (ii) is certified for n ≥ e^{73.89}, B5(1)'s own threshold. B8(1)'s bootstrap closes down to
   ln n₀ = 62; (ii) was not evaluated below 73.89 because B5(1) does not need it.

### 8.5 No exponentially large fast content

At the upper end of the arc a fast component k e^{−1/(2z)} of Y⁺_out relative to Z⁺_in has modulus |k|e^{sin κ/(2r_n)}.
(ii) bounds W_Φ by 79.04 εr_n, so |k| ≤ 79.04 εr_n e^{−sin κ/(2r_n)}: exponentially small, as C_selection.md §10.4 requires.

## 9. Checks

| check | script | result | mutation (must fail) |
|---|---|---|---|
| reduced system = original model, M = 1, 3, 5 | `b8m_model.py` | 10⁻⁴⁸ | Ω sign flipped: fails |
| truncated series: residual decay n^{−(K+1)}, n^{−K}; top degree = inner series | `b8m_formal.py --check` | exponents within 0.15; 3.7×10⁻⁷⁵ | Ω Δ-term × 1.001: both fail |
| structural zeros of the atoms at the centre | `b8m_atoms.py 4 0 0` | removed entries ≤ 1.8×10⁻⁶⁶ | |
| identities I1 to I4 (M = 1, K = 3, 4; M = 3, K = 2) | `b8m_identities.py` | ≤ 3×10⁻⁷⁷ | old ρ₂: I1 fails (0.07 to 0.36); q without (I+e)⁻¹: I2, I3 fail |
| series part of (H8) vs B5's leading data; ε⁰ part vs the inner series | `b8m_h8.py --centre` | 76.5602, 3.23126, 6.46252; 10⁻⁶⁹ (`b8m_h8_centre.log`) | Φ without [xc′P]: the ε⁰ part no longer matches, fails |
| (A1) actual remainder R at n = 10³ on the ray and 4 arcs (79 points) vs the next formal orders ε^{5..8}Z_k (in the solution's gauge), where ρ ≤ 0.03 | `b8m_numcheck.py 1000 4` | 2.0×10⁻³ (limit 0.1; the largest value is at \|x\| = 0.02, set by the start at x₀ = 10⁻³) | τ_S × 1.01 moves it to 6.0×10⁻³, below this check's resolution; caught by (A2) |
| (A2) R vs Z − Z^K from an independent integration of the original (p, m, b, c) model, where resolvable | same | 5.0×10⁻⁴ (limit 2×10⁻³, four times the spline error; up to \|R\|/\|Z\| = 0.35) | τ_S × 1.01: 6.2×10⁻³ at every point, fails |
| (B) R vs the certified bounds (h at ln n₀ = 73.89) | same | ratios V 0.013, S 0.024, β 0.20, c 0.080 | weight ρ⁶ instead of ρ⁵: β 1.8, c 1.9, fails |
| (C) Lemma K: fast propagator on the arc \|x\| = 5 vs the log-norm bound | same | ln‖E‖ = −21.9 ≤ −15.4 | ‖H'‖ dropped: bound −26.6, fails |

What the numerical checks share with the certificate: the formal series (`b8m_formal.py`) and the definition of the
model. (A2) shares nothing else: it integrates the original equations in (p, m, b, c). (B) uses only the certified
constants h, not the bound code.

What is not tested: an error made identically in `b8m_formal.py` and in the identity script (they share the formal
coefficients but not the equations; the formal series is checked against the original model and the inner series
independently); the scripts have not been reviewed by a second person.

## 10. Every fixed M

**Claim (proved, without a numerical n₀(M)).** For every odd M, κ ∈ (0, π/4), κ₁ > 0, δ ∈ (0, ¼), γ ∈ (0, ½) and
K ≥ K_γ := ⌊(2 − 2γ)/(1 − 2γ)⌋ + 1, the least K with (1 − 2γ)K > 2 − 2γ (K_γ = 4 for γ = 0.3), there is n₀(M) such that Lemma B8(M) and (H8) hold for n ≥ n₀(M),
with δ = (δ_P, δ_Φ, δ_C, δ_ϑ) any numbers above the leading data (the ε-coefficients of section 8.2 for that M).

*Proof.* Every step above is written for operators on the (M + 1)-dimensional odd space; only the numbers depend on M
(‖Ω‖ = M|ω|, the atoms of the formal series, which exists for every M by Lemma F, and the identities, checked for M = 3).
As n₀ → ∞ with the atoms fixed:
- the kernel constants: ‖H'‖ ≤ ρ‖η̃‖/(1 − ‖Γ‖) → 0 on the ray, so s_* → sin κ; N_F ∝ 1/ρ_max → ∞, so K_F → 1 and
  γ_F = O(ρ_max^{1/2}) → 0; γ_N1 → 0; γ_N0 → α but multiplies only terms carrying a factor x² ≤ r₁²;
- every coupling of P(h) between different components either tends to 0 (S ← V, S ← c, S ← β̂ on F and N1, β̂ ← S, β̂ ← c,
  all through factors ρ, 1/n or γ_F) or is one-directional (V ← S and c ← β̂, S are O(1), but the reverse couplings
  vanish);
- the quadratic terms carry an extra factor ρ^{K+1}.

So the spectral radius of the linear part of P tends to 0 and the forcing stays bounded: P(h) < h has a solution for
large n₀. For (H8)(ii) the remainder part tends to 0 like n₀^{−(1−2γ)K + 2 − 2γ} and the series part to its leading
coefficients. ∎ The scripts are written for M = 1; running them for M = 3, 5 needs the M-dependent atoms
(`b8m_atoms.py` hard-codes the 2×2 determinant) and the kernel cells.

## 11. Discrepancies and caveats

1. **Harmonic truncation.** C_selection.md §1 and the paper's Definition of the M-model keep b, c on |ν| ≤ M − 1. Theorem
   A(M = 1), B5(1) and this lemma use |ν| ≤ M + 1. `b3_outer_M.py` (commit bcc3eaa, not on main) also uses M − 1, and
   it projects each equation once after forming full products, where B5(1) and B8 project every binary product; the
   two projections agree for M = 1 and differ for M ≥ 3 (`b8m_model.py --mutate=proj`). The coordinator should change
   §1 and the paper.
2. **R_n.** C_selection.md §11 defines R_n = (κ₁n/ln n)^{1/2}; B5(1) needs (κ₁n/(cos κ ln n))^{1/2}. B8 is proved on the
   larger domain.
3. **K\*.** C_selection.md §11 fixes K* = 3 from the transport loss. Here the transport is inside Lemma K and imposes no
   condition on K of its own. K = 4 is what (H8)(ii)'s Φ-bound needs at γ = 0.3 (section 8.2), and only K = 4 is
   certified. (K = 3 closed in runs made before the ρ₂ correction; those runs are withdrawn and were not repeated.)
4. **Light-cone disc.** C_selection.md §11 handles the excluded disc through B6. B8 never enters it, and B5(1) does not
   need it. B6 matters only for the Stokes jump on x > 1.
5. **The ρ₂ correction** (section 4): found and fixed in this pass; all certificates postdate it.
6. **How sharp the bounds are.** At n = 10³ (far below n₀) the actual remainder on the sample paths uses at most 20%
   of the bound with the n₀-certificate's h (β-component; 1 to 8% for V, S, c), and one order more of ρ in the weight
   is violated by a factor 1.9 (section 9). So the shape of the bound is right and its constants are not wasteful at
   moderate n; at n ≥ n₀ the remainder is much smaller relative to the bound. The large h (10⁵ to 10⁶) reflects the
   growth of the formal coefficients (about ×15 per order), not slack.

## 12. Proposed text for C_selection.md, PROOF.md and the paper

**C_selection.md §11**, replace "What remains in B8" by:

> **Lemma B8 with constants** (proof/B8_constants.md). For the M-model with b, c on |ν| ≤ M + 1, κ = 0.3, κ₁ = 0.6,
> δ = 0.05 and K = 4, the centre-regular solution (gauge c(0) = c^K(0)) continues analytically to
> 𝒟_n = {|x| ≤ (κ₁n/(cos κ ln n))^{1/2}, 0 ≤ arg x ≤ π/4 + κ/2} ∖ {|x − 1| < δ/2} for every n ≥ e^{62}, with
> |R_V| ≤ h_Vρ^5, |R_S| ≤ h_Sθρ^5, |R_β − qR_S| ≤ h_βθρ^5, |R_c| ≤ h_cρ^5 (certified for M = 1 over B5(1)'s parameter box;
> h = (2.4×10⁶, 6.3×10⁶, 6.2×10⁵, 1.7×10⁵) at n₀ = e^{73.89}). It gives B5(1)'s hypothesis (H8) for n ≥ e^{73.89} with
> δ = (8.3×10⁻⁵, 79.04, 3.232, 6.464) ≤ δ_ref. For every fixed M the same holds for n ≥ n₀(M) (proved, no numerical
> value).

**PROOF.md**, the B8 row of the status table: "global continuation (B8) | **certified for M = 1** (n ≥ e^{62}; (H8) of
B5(1) for n ≥ e^{73.89}); proved for every fixed M without a numerical n₀(M) | proof/B8_constants.md".

**Paper, Layer B**: after the statement of the continuation lemma, "For M = 1 the lemma holds with explicit constants
for n ≥ e^{62}, and it supplies the matching hypothesis of the inner lemma for n ≥ e^{73.89} (computer-assisted, in ball
arithmetic)."

## 13. Reproducing

```
cd solver/proof
python3 b8m_model.py                       # section 1.2
python3 b8m_formal.py 1 4 --check          # Lemma F
python3 b8m_atoms.py 4 0 0                 # atoms and structural zeros at the centre (b8m_atoms_K4.log)
python3 b8m_identities.py                  # Lemma R (b8m_identities.log); --mutate=rho2old, --mutate=q
nice -n 10 python3 b8m_scan.py 4 73.89,70,66,62 --rA=2e-6 --rW=2e-6 --log=b8m_scan_K4_box.log      # ~1-5 min each
python3 b8m_h8.py 4 73.89 --log=b8m_scan_K4_box.log --rA=2e-6 --rW=2e-6      # (H8)(ii); --centre for the series check
python3 b8m_numcheck.py 1000 4 --log=b8m_scan_K4_box.log      # section 9 (b8m_numcheck_n1e3.log; saves the .npz)
python3 b8m_numcheck.py 1000 4 --log=b8m_scan_K4_box.log --load=b8m_numcheck_n1000.npz --mutate=weight   # must fail
python3 b8m_numcheck.py 1000 4 --log=b8m_scan_K4_box.log --load=b8m_numcheck_n1000.npz --mutate=kernel   # must fail
python3 b8m_numcheck.py 1000 4 --mutate=tau                                                             # must fail
```

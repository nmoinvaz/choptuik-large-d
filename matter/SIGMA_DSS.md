# The σ-model's echoing (DSS) critical solution in D dimensions

[SIGMA.md](SIGMA.md) built the D-dimensional self-gravitating hedgehog σ-model (wave map into $`S^{D-1}`$, coupling
η = 4πGf²) and followed its continuously self-similar (CSS) critical solution, which ends at a fold at D = 8.544
(η = 0.14) and so cannot test the large-D exponent. This page computes the other critical solution, the discretely
self-similar (DSS, echoing) one, as a periodic boundary-value problem: validated at D = 4 against Husa et al. and Lechner
et al., then continued in D at η = 1 and η = 0.3. The question is the one of [STRUCTURE.md](STRUCTURE.md): at fixed η the
D = ∞ inner problem has the term sin 2u/(4s), which breaks the s-translation behind γ∞ = 1/2 for the scalar, so γ∞ need
not be 1/2 and should depend on η, tending to the ℓ = 1 multiplet's value as η → ∞.

Notation as in the README and SIGMA.md: D = n + 2, x = r/(−t), τ = −ln(−t), s = r²/(2n), γ = 1/λ the radius exponent
(the mass exponent is (D − 3)γ). Statements are marked **computed** (numerical, with its checks), **argued**, or
**not established**.

Contents

- [1. Summary](#1-summary)
- [2. Equations](#2-equations)
- [3. Method](#3-method)
- [4. Validation in D = 4](#4-validation-in-d--4)
- [5. Continuation in D](#5-continuation-in-d)
- [6. Checks along the branch](#6-checks-along-the-branch)
- [7. Large D: what the data allow](#7-large-d-what-the-data-allow)
- [8. Verdict](#8-verdict)
- [9. Open issues and a proposed local-machine task](#9-open-issues-and-a-proposed-local-machine-task)
- [10. Scripts and data](#10-scripts-and-data)
- [11. References](#11-references)

## 1. Summary

- **Solver (computed).** The ℓ-multiplet agent's periodic DSS solver, copied and extended by the σ-model terms. At
  η = ∞ (in the field normalisation ψ = √(η/n) u) the σ-model is exactly the ℓ = 1 multiplet, so the DSS solution is
  reached by continuation in η⁻¹ from the ℓ = 1 solution at any D.
- **D = 4 validation (computed).** Δ(η) for 0.18 ≤ η ≤ 10 agrees with Husa et al.'s Table I to 0.6% or better for
  η ≥ 0.225 and 1–2% at η = 0.2; γ = 0.11563 at η = 1 and 0.11551 at η = 0.3 against their 0.1163/0.1155 and
  0.1152/0.1146; at η = ∞, γ = 0.11878 against the large-η limit 0.1185 ± 0.0005. The DSS–CSS transition fits
  η_c = 0.1702 with a slope 2/a = 5.6 (LTHA02: η_c ≈ 0.17, slope ≈ λ_CSS = 5.14). The half-period symmetry
  u(τ + Δ/2) = −u(τ) is the published one and is stable: without it imposed, Newton returns to it (D = 6).
- **Δ(D), γ(D) (computed; D ≥ 10 preliminary, see next item).** One branch, continued without folds from D = 4 to D = 10
  at η = 1 and η = 0.3, with k_c = 23 checks at D = 6, 9, 10 (λ agrees to 10⁻⁴ or better); growth rates in the table
  below, Δ in §5. One real unstable mode besides the gauge mode (the real roots of det M in 0.2 ≤ λ ≤ 7.4 at D = 6 are
  the gauge rate and λ, nothing else). Δ(D) rises from 0.46 and levels off near 1.03 (η = 1) and 1.15–1.17 (η = 0.3)
  for D ≈ 8–10.
- **Above D ≈ 10 (preliminary).** The Jacobian has a near-null band in the high harmonics of the centre datum (the main
  solver's "junk" directions: s/s_max = 1.5×10⁻¹⁰ at D = 9, 1.1×10⁻¹¹ at D = 10.25, no Δ component). A continuation
  that ignores it stalls at D ≈ 10.5; with a smoothness prior in both the Newton solve and the tangent it passes
  D = 10.5. At η = 0.3 the branch stays smooth to D = 11 and λ(D) turns: 3.0814, 3.0685, 3.1127 at D = 10, 10.5, 11
  (γ = 0.3245, 0.3259, 0.3213), unchanged to 10⁻⁴ under k_c = 23, a ten times weaker prior and a longer horizon series.
  At η = 1 the solver found two different solutions at D = 10.475 (Δ = 0.9935 and 1.0340), so η = 1 above D ≈ 10.25 is
  not reliable. No fold direction was seen. The gauge mode's root drifts away from its exact value above D ≈ 8 (by 0.25
  at D = 11), unexplained; so the D ≥ 10 growth rates are preliminary. Going further needs more care than short runs on
  the shared machine allow (§9).
- **γ∞ (not established).** Free fits of λ = λ∞ + Σ c_k n⁻ᵏ to D ≤ 9 give λ∞ = 1.76, 2.74, 2.14, 2.31 (η = 1) and 1.87,
  2.66, 2.15, 2.34 (η = 0.3) for 1–4 powers (D ≥ 5); imposing λ∞ = 2 fits about as well (residuals within a factor 2,
  except a fragile factor 7 at η = 1 with four powers). A planted control shows the method would separate λ∞ = 2.3 from
  2 on these D if three powers described the data, but on the real data the three-power residuals alternate in sign at
  5×10⁻⁴, so the truncation is not adequate. Adding D = 10 makes the fits worse (λ(D) changes curvature near
  D ≈ 9.5), and the preliminary points beyond turn over (η = 0.3: γ = 0.3259 at D = 10.5, 0.3213 at D = 11). The D ≤ 9
  extrapolations allow γ∞ anywhere in about 0.36–0.57; the turn at D ≈ 10.5, if it is the branch and not the
  discretization, would put γ∞ below 1/2 at η = 0.3. Neither is established, so γ∞ ≠ 1/2 is not tested.
- **What the data do show (computed).** At every D from 4 to 9, γ of the σ-model lies below the ℓ = 1 multiplet's by an
  amount that depends on η and is nearly D-independent: −0.0028 to −0.0033 at η = 1, and −0.0146 to −0.0165 at η = 0.3
  for D = 6–9. This is the η-dependence the prediction expects, in the direction η → ∞ → ℓ = 1, but whether the offset
  survives to D = ∞ is an extrapolation of five points (fits: −0.003 to −0.005 at η = 1, −0.008 to −0.022 at η = 0.3).

| D | 4 | 5 | 5.5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|
| γ, η = 1 | 0.11563 | 0.21265 | 0.24130 | 0.26273 | 0.29274 | 0.31307 | 0.32798 | 0.3425 |
| γ, η = 0.3 | 0.11551 | 0.20568 | 0.23136 | 0.25097 | 0.27949 | 0.29966 | 0.31493 | 0.3245 |
| γ, ℓ = 1 (η = ∞) | 0.11890 | 0.21551 | | 0.26557 | 0.29573 | 0.31620 | 0.33129 | |

(k_c = 15 for the σ-model; the ℓ = 1 row is the ℓ-multiplet agent's k_c = 23 values, copied with attribution to
`data/sigdss_l1_reference.jsonl`. k_c = 23 checks at D = 6, 9, 10 agree to 10⁻⁴ in λ or better, §6. Preliminary,
with the stronger prior: γ = 0.3444 (η = 1, D = 10.5), 0.3259 and 0.3213 (η = 0.3, D = 10.5 and 11).)

## 2. Equations

The model, the coupling η = 4πGf² and the self-similar equations are those of [SIGMA.md §2](SIGMA.md#2-equations)
(checked there by 48 sympy checks in [`sigma_eqs.py`](../solver/matter/sigma_eqs.py)). In the first-order form of
[`solver/dss.py`](../solver/dss.py), with the field normalised as the main solver's scalar,

```math
\psi = \sqrt{\eta/n}\;u, \qquad p, m = \sqrt{\eta/n}\;r(\Phi\pm\Pi), \qquad \kappa^2 = \frac{n}{\eta},
\qquad \sigma(\psi) = \frac{\sin\kappa\psi}{\kappa}, \qquad g(\psi) = \cos\kappa\psi ,
```

the equations are (η sin²u = nσ², and √(ηn) sin 2u/2 = nσg)

```math
x a_x = \tfrac a2\big[p^2+m^2 + 2na^2\sigma^2 + (n-1)(1-a^2)\big], \qquad
x c_x = c\big[(n-1)(a^2-1) - 2na^2\sigma^2\big], \qquad x\psi_x = \tfrac12(p+m),
```
```math
p_\tau + (x-c)\,p_x = c_xp - \frac{cp}{x} + \frac{nc\,(p+m)}{2x} - \frac{nca^2\sigma g}{x}, \qquad
m_\tau + (x+c)\,m_x = -c_xm + \frac{cm}{x} - \frac{nc\,(p+m)}{2x} + \frac{nca^2\sigma g}{x} .
```

At κ = 0 (η = ∞ at fixed ψ) σ = ψ and g = 1, and these are exactly the ℓ = 1 multiplet's equations with
L = ℓ(ℓ + n − 1) = n: the hedgehog becomes a multiplet of D − 1 free scalars. So 1/η = κ²/n is a homotopy parameter
from the ℓ = 1 critical solution to the σ-model's, at every D.

**Boundary conditions.** As for ℓ = 1: at the centre ψ ≈ q(τ)x and p ≈ m ≈ q(τ)x (u ≈ κq(τ)x, the hedgehog's
regularity condition u = O(x)), with free data q(τ) and c₀(τ) = c(τ, 0); at the self-similarity horizon x = 1 (c = 1)
free data m, a, ψ and p from regularity. The equations are invariant under (p, m, ψ) → −(p, m, ψ), i.e. u → −u, and the
solution is sought with p, m, ψ odd and a, c even under τ → τ + Δ/2 (checked in §4).

**Series.** The Taylor series at both ends are generated order by order as in the ℓ-multiplet solver, with the series of
σ(ψ) and g(ψ) built from dσ = g dψ, dg = −κ²σ dψ. At the centre σ₀ = 0, so the order-k coefficient of σg enters the
order-k system for (P_k, M_k) with the same coefficient as ψ_k does for ℓ = 1; the matrix of that system is unchanged.

## 3. Method

[`sigdss_solver.py`](../solver/matter/sigdss_solver.py), class `SigDSS`, is a copy of the ℓ-multiplet agent's
`DSSEll` (worktree `agent-a5bec5ffdf3db4330`, `solver/matter/dss_ell.py`, itself built on this repository's
[`solver/spec.py`](../solver/spec.py)) with the σ-model terms of §2. Nothing is imported from that worktree. As there:
dealiased Fourier series in τ up to harmonic k_c (N ≥ 3k_c points), parities imposed, multiple shooting in x from the
centre series (order 9) and the horizon series (order 16, evaluated at u = 1 − x = 0.01) through 12 nodes
0.02 … 0.9 with RK4, step counts frozen from the stiffness k_cω/|x ∓ c| of the solution (target κ_RK = 0.5 at D = 4 and
0.7 above), phase fixed by Im q̂₁ = 0, every derivative (Jacobian, growth-mode matrix, ∂F/∂n, ∂F/∂η⁻¹) by complex step,
Levenberg–Marquardt with one SVD per Jacobian. Changes:

- **The σ-model terms**, with σ(ψ) and g(ψ) evaluated as entire functions of κ²ψ² (Taylor series for |κ²ψ²| below 0.5)
  so that complex steps in η⁻¹ and n stay exact at η⁻¹ = 0. Checks: at η = ∞ the residual of the ℓ-agent's D = 4,
  k_c = 23 solution is 1.57×10⁻¹⁵, as in their code; ∂F/∂η⁻¹ and ∂F/∂n against central differences agree to
  2–9×10⁻⁹ (the difference error); Jacobian columns against central differences to 2–5×10⁻¹⁰.
- **Speed.** NumPy has no BLAS path for a complex array times a real matrix (it is about 20 times slower). Splitting the
  complex-step arrays into real and imaginary parts before each spectral matrix product makes the complex-step residual
  11 times faster; the Jacobian and the mode matrix gain about as much. (The ℓ-multiplet code has the same bottleneck.)
- **Continuation** in η⁻¹ at fixed D and in D at fixed η (tangent predictor from the last Newton Jacobian, quadratic
  correction from the previous point, adaptive steps landing on target values; [`sigdss_run.py`](../solver/matter/sigdss_run.py)).
- **Growth rate** λ of the unstable mode: real root of det M(λ), with M the linearised matching map for perturbations
  e^{λτ} × (periodic, same parity). |det M| varies exponentially in λ, so a secant on det crawls; the root is taken
  from the model log|det M| = log|λ − λ₀| + a + bλ through the last three evaluations (5–7 evaluations from a guess
  within a few per cent; a sign-change bracket with Brent's method as fallback). The guess is extrapolated in γ = 1/λ.
- **Smoothness prior** (from D = 9 at η = 1, D = 8 at η = 0.3): Levenberg–Marquardt minimises |F|² + μ²|Wz|², with
  W = (k/k_c)⁴ on each spectral unknown of harmonic k and μ = 10⁻⁸ or 10⁻⁷ times the largest singular value of the
  Jacobian (the ℓ-agent's `_lm_reg`, after the main solver's `smooth_junk`); the continuation tangent solves the same
  augmented least-squares system. It fixes the near-null high centre harmonics (§6).
- **Diagnostics** per point: |F|, the ψ-constraint K = ψ_τ + (p + m)/2 − c(p − m)/(2x) relative to max|p| (zero for an
  exact solution, not imposed: it measures τ-truncation and integration error), the gauge rate
  λ_g = mean(1 − c_x(τ, 1)), and max|u| over the nodes.

## 4. Validation in D = 4

**Published values.** Husa, Lechner, Pürrer, Thornburg and Aichelburg [HLPTA00, Table I] list, for two families of
initial data, the echoing period of 2m/r, which they call Δ/2, and the mass exponent γ (in D = 4 the mass and radius
exponents coincide). Their Table I was read twice from the arXiv HTML version in this session (identical both times):
for η = 0.18, 0.2, 0.225, 0.25, 0.3, 0.4, 0.5, 1, 2, 5, 10, 100, Δ/2 = 0.5522/0.5478, 0.4367/0.4327, 0.3464/0.3472,
0.3043/0.3046, 0.2668/0.2675, 0.2452/0.2445, 0.2386/0.2386, 0.2314/0.2313, 0.2295/0.2305, 0.2304/0.2308, 0.2293/0.2312,
0.2302/0.2311 and γ = 0.1063/0.1028, 0.1091/0.1150, 0.1207/0.1169, 0.1173/0.1173, 0.1152/0.1146, 0.1132/0.1139,
0.1152/0.1130, 0.1163/0.1155, 0.1179/0.1167, 0.1183/0.1178, 0.1186/0.1182, 0.1187/0.1182 (family 3.1/family 3.2). At large
η, "Δ/2 asymptotes to 0.2300 ± 0.0003" and γ to 0.1185 ± 0.0005. They note that at η = 0.18 the evolution is only
approximately DSS. Lechner, Thornburg, Husa and Aichelburg [LTHA02] place the DSS–CSS transition at η_c ≈ 0.17 with
Δ ≈ −a ln(η − η_c) + b, a ≈ 2/λ_CSS (λ_CSS ≈ 5.14), fitted on 0.1726 ≤ η ≤ 0.195. The η → ∞ limit is the ℓ = 1 multiplet:
Δ₁ = 0.460 ± 0.002, γ₁ = 0.119 ± 0.001 [OVCU07, via LITERATURE.md], and the ℓ-agent's periodic solution at k_c = 39 has
Δ = 0.4598611, λ = 8.41042 at k_c = 23 (γ = 0.11890).

**The half-period symmetry.** HLPTA00 measure Δ/2 from the maxima of 2m/r, which is even in u, and give the full period
of the field as Δ = 0.46 at η = 1 (their Fig. 2); LTHA02 note that the near-CSS episodes alternate between +u_CSS and
−u_CSS. Both say that u(τ + Δ/2) = −u(τ), which is the ansatz used here. The computed full periods agree with twice
their Δ/2 at every η (below), and at D = 6 the symmetric solution is an isolated solution of the problem without the
symmetry (§6).

**Computed** (k_c = 15, continuation in η⁻¹ from the ℓ = 1 solution at η = ∞; `data/sigdss_d4eta15.jsonl`,
`data/sigdss_lam_d4b.jsonl`):

| η | Δ here | 2 × (Δ/2) [HLPTA00] | λ here | γ = 1/λ here | γ [HLPTA00] | max abs u | constraint |
|---|---|---|---|---|---|---|---|
| ∞ | 0.460021 | 0.4600 ± 0.0006 (limit) | 8.41903 | 0.11878 | 0.1185 ± 0.0005 (limit) | — | 1.2e-1 |
| 10 | 0.459963 | 0.4586 / 0.4624 | | | 0.1186 / 0.1182 | 0.19 | 1.1e-1 |
| 5 | 0.459956 | 0.4608 / 0.4616 | | | 0.1183 / 0.1178 | 0.27 | 9.6e-2 |
| 2 | 0.460277 | 0.4590 / 0.4610 | | | 0.1179 / 0.1167 | 0.44 | 7.4e-2 |
| 1 | 0.462226 | 0.4628 / 0.4626 | 8.64838 | 0.11563 | 0.1163 / 0.1155 | 0.64 | 4.8e-2 |
| 0.5 | 0.474696 | 0.4772 / 0.4772 | | | 0.1152 / 0.1130 | 0.97 | 1.3e-2 |
| 0.4 | 0.488419 | 0.4904 / 0.4890 | | | 0.1132 / 0.1139 | 1.12 | 6.0e-3 |
| 0.3 | 0.533562 | 0.5336 / 0.5350 | 8.65735 | 0.11551 | 0.1152 / 0.1146 | 1.33 | 4.4e-3 |
| 0.25 | 0.608074 | 0.6086 / 0.6092 | | | 0.1173 / 0.1173 | 1.50 | 4.8e-3 |
| 0.225 | 0.694164 | 0.6928 / 0.6944 | | | 0.1207 / 0.1169 | 1.59 | 6.2e-3 |
| 0.2 | 0.880336 | 0.8734 / 0.8654 | | | 0.1091 / 0.1150 | 1.73 | 1.0e-2 |
| 0.18 | 1.276195 | 1.1044 / 1.0956 (approximately DSS) | | | 0.1063 / 0.1028 | 1.84 | 3.0e-2 |

- **Δ agrees with HLPTA00** for 0.225 ≤ η ≤ 10 to 0.6% or better (differences ≤ 0.0025), mostly inside the spread of
  their two families (at η = 0.5 both of theirs are 0.5% above ours), and to 1–2% at η = 0.2. At η = 0.18 it does not
  (1.276 against 1.10): there Δ varies as dΔ/dη ≈ −a/(η − η_c) ≈ −37, the published value lies on our curve at
  η ≈ 0.186, and HLPTA00 say their evolution is only approximately DSS there.
- **γ agrees** at the two couplings computed: 0.11563 at η = 1 (theirs 0.1163/0.1155) and 0.11551 at η = 0.3
  (0.1152/0.1146). The η = ∞ value 0.11878 is within the large-η limit 0.1185 ± 0.0005 and OVCU07's 0.119 ± 0.001.
- **Resolution.** At η = ∞ the k_c = 15 values differ from the ℓ-agent's high-resolution ones by +1.6×10⁻⁴ in Δ
  (k_c = 39) and +0.0086 in λ (k_c = 23), i.e. −1.2×10⁻⁴ in γ. The constraint diagnostic is 2.5 to 27 times smaller at
  η ≤ 1 than at η = ∞, so these are upper estimates of the k_c = 15 error at D = 4. Both are far below the scatter of the
  evolution data.
- **The DSS–CSS transition.** A fit of Δ(η) = −a ln(η − η_c) + b to the 7 points with η ≤ 0.195
  ([`sigdss_etac.py`](../solver/matter/sigdss_etac.py), `data/sigdss_etac_d4.json`) gives η_c = 0.1702 and a = 0.359
  (largest relative residual 1.8×10⁻⁴); with η ≤ 0.21 and 0.23, a = 0.334 and 0.305. LTHA02 find η_c ≈ 0.17 and a within
  7% of 2/λ_CSS = 0.389 (λ_CSS = 5.1428 at η = 0.17, SIGMA.md §4). Here a rises toward 0.389 as the fit window narrows
  (8% short at η ≤ 0.195), consistent with the heteroclinic picture: near η_c the DSS orbit spends a time
  (1/λ_CSS) ln 1/(η − η_c) near each of ±u_CSS per period.

## 5. Continuation in D

**Path.** The D = 4 continuation in η⁻¹ is §4. For D ≥ 5 the ℓ = 1 agent's k_c = 15 solution at D = 5 was continued
in η⁻¹ to η = 1 and η = 0.3 (`data/sigdss_d5eta15*.jsonl`; at D = 5, Δ = 0.78335 → 0.80584 → 0.96431 for
η = ∞, 1, 0.3), and from there in D at fixed η with k_c = 15 (`data/sigdss_m1up*.jsonl`, `data/sigdss_m03up*.jsonl`;
the marches were restarted several times, after root-finder fixes and container restarts, always from saved
solutions). From D = 9 on (η = 1) and D = 8 on (η = 0.3) the Newton solves use the main solver's smoothness prior on
the high harmonics (`--reg 1e-8`, as in the ℓ-agent's code; §6) and accept |F| below 10⁻⁶. **Computed:**

| D | η = 1: Δ | λ | γ | constraint | max abs u | λ_g | η = 0.3: Δ | λ | γ | constraint | max abs u |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.462226 | 8.64838 | 0.115629 | 4.8e-2 | 0.64 | 0.8981 | 0.533562 | 8.65735 | 0.115509 | 4.4e-3 | 1.33 |
| 5 | 0.805845 | 4.70247 | 0.212654 | 1.9e-3 | 0.67 | 0.8684 | 0.964306 | 4.86186 | 0.205683 | 9.3e-4 | 1.31 |
| 5.5 | 0.890368 | 4.14422 | 0.241300 | 5.2e-4 | 0.63 | 0.8574 | 1.059871 | 4.32227 | 0.231360 | 3.5e-4 | 1.21 |
| 6 | 0.944491 | 3.80622 | 0.262728 | 2.1e-4 | 0.63 | 0.8485 | 1.115282 | 3.98458 | 0.250968 | 2.8e-4 | 1.19 |
| 7 | 1.001249 | 3.41596 | 0.292744 | 2.6e-4 | 0.58 | 0.8359 | 1.163078 | 3.57795 | 0.279490 | 1.7e-4 | 1.09 |
| 8 | 1.022509 | 3.19414 | 0.313073 | 1.8e-4 | 0.54 | 0.8283 | 1.171733 | 3.33713 | 0.299658 | 3.1e-5 | 1.01 |
| 9 | 1.026904 | 3.04893 | 0.327984 | 6.0e-5 | 0.51 | 0.8243 | 1.164945 | 3.17532 | 0.314930 | 1.6e-5 | 0.95 |
| 10 | 1.026142 | 2.91980 | 0.342490 | 1.6e-3 | 0.48 | 0.8222 | 1.154559 | 3.08142 | 0.324525 | 1.6e-3 | 0.89 |
| 10.5 (prior 10⁻⁷; η = 1 uncertain, see text) | 0.994292 | 2.90366 | 0.344392 | 1.6e-3 | | | 1.139115 | 3.06853 | 0.325888 | 1.7e-3 | |
| 11 (prior 10⁻⁷) | | | | | | | 1.117461 | 3.11265 | 0.321271 | 1.8e-3 | |

(λ_g = mean over τ of 1 − c_x(τ, 1), the growth rate of the gauge mode that relabels the light cone; max abs u is over
the nodes 0.02 ≤ x ≤ 0.9 and the light cone.)

- **γ rises monotonically** at both couplings, like the ℓ = 1 multiplet's and the scalar's, and stays below the ℓ = 1
  values at every D (§7).
- **Δ is nearly flat above D ≈ 8.** At η = 1, Δ rises from 0.462 (D = 4) to 1.0269 at D = 9, dips to 1.0244 at
  D = 9.71 and rises again (1.0261 at D = 10, confirmed at k_c = 23 to 10⁻⁶; 1.0270 at 10.1). At η = 0.3 it peaks at
  1.1717 near D = 8 and decreases to 1.1545 at D = 10 and 1.1517 at 10.1. The ℓ = 1 multiplet's Δ(D) levels off in the
  same way (0.9838, 0.9882 at D = 8, 9, k_c = 23; their k_c = 15 march turns over near D = 9).
- **λ(D) changes its curvature near D ≈ 9.5 at η = 1.** Its second differences at unit spacing are 0.168, 0.077, 0.016
  (D = 6–8, 7–9, 8–10), against 0.166, 0.079, 0.068 at η = 0.3; the D = 10 values are confirmed at k_c = 23 (§6), so this
  is not a resolution effect.
- **The amplitude** max|u| on 0 ≤ x ≤ 1 decreases with D (0.67 → 0.48 at η = 1, 1.31 → 0.89 at η = 0.3), and grows as η
  decreases, as expected from u = κψ with ψ ∼ n^(−1/2) at fixed x. The O(1) nonlinearity that distinguishes the σ-model
  from the multiplet at D = ∞ sits in the inner region x ∼ √n, outside the light cone (SIGMA.md §6.1), which the
  periodic problem on 0 ≤ x ≤ 1 sees only through the analyticity condition at x = 1.
- **No fold up to D = 10.** The secant tangent |dz/dD| falls smoothly from 0.91 (D = 6.1) to 0.23 (D = 9.7), with a
  small Δ component. The first marches stopped at D = 10.475 (η = 1) and 10.597 (η = 0.3) because of the near-null
  band (§6), not because the branch ends: between D = 10 and the stop the tangent grows from 0.3 to 6 (η = 1) and to
  200–300 (η = 0.3), almost entirely in the centre datum q(τ) at harmonics 3–15, whose amplitudes grow from a few 10⁻²
  to the size of the fundamental; Δ then jumps between neighbouring points (η = 0.3: 1.108–1.125 at D = 10.592–10.598),
  which a smooth branch does not do. The tangent was computed without the smoothness prior, so it fed the near-null
  directions. With the prior also in the tangent (`sigdss_run.py` now does this) and a stronger prior (10⁻⁷), the
  centre harmonics stay small and the marches pass D = 10.5 (`data/sigdss_m1hi.jsonl`, `data/sigdss_m03hi.jsonl`;
  stopped there to keep the runs short). At η = 0.3 the continuation is smooth (Δ = 1.1517, 1.1486, 1.1400, 1.1391 at
  D = 10.1, 10.25, 10.475, 10.5). At η = 1 it is not: Δ = 1.0270, 1.0295 at D = 10.1, 10.25, then 0.9935 at 10.475,
  while the first march had found 1.0340 at the same D. Two different solutions at D = 10.475 mean that at η = 1 above
  D ≈ 10.25 the solver does not follow a single smooth branch, and the η = 1 values there are not reliable (at D = 10.25
  the weakest Jacobian directions are still the junk band, s/s_max = 1.1×10⁻¹¹ at harmonics 9–11 with Δ components
  ≤ 4×10⁻⁵, `data/sigdss_svd.jsonl`, so no isolated fold direction was seen).
- **λ(D) turns at D ≈ 10.5.** λ = 2.9198 → 2.9037 (η = 1) and 3.0814 → 3.0685 (η = 0.3) between D = 10 and 10.5,
  slopes −0.03 per unit D against −0.13 and −0.09 between D = 9 and 10; at η = 0.3, D = 11 gives λ = 3.1127: λ(D) has a
  minimum near D = 10.5, i.e. γ(D) a maximum of about 0.326, and γ(11) = 0.3213. At η = 0.3 the D = 10.5 and 11 values
  survive k_c = 23, a ten times weaker prior and a longer horizon series to 10⁻⁴ (§6); at η = 1 the D = 10.5 value
  rests on the second of the two D = 10.475 solutions. Δ keeps decreasing smoothly at η = 0.3 (1.1391, 1.1300, 1.1175
  at D = 10.5, 10.75, 11).

## 6. Checks along the branch

- **Resolution, k_c = 23 against 15** ([`sigdss_run.py refine`](../solver/matter/sigdss_run.py),
  `data/sigdss_refine_kc23.jsonl`). D = 6, η = 1: Δ = 0.94449116 against 0.94449059, λ = 3.8062216 against 3.8062242
  (7×10⁻⁷ relative); constraint 2×10⁻⁶ against 2×10⁻⁴. D = 9, η = 1 (with the prior): λ = 3.048509 against 3.048927
  (1.4×10⁻⁴ relative); Δ = 1.02703 against 1.02690, but at k_c = 23 Δ was still drifting by 10⁻⁵ per Newton step at
  |F| = 2×10⁻¹⁰, along the near-null directions below, so it is determined only to about 10⁻⁴ there. D = 10 (with the
  prior): η = 1, Δ = 1.0261425 against 1.0261416 and λ = 2.919835 against 2.919800 (1.2×10⁻⁵ relative); η = 0.3,
  Δ = 1.15452 against 1.15456 and λ = 3.081100 against 3.081424 (1.0×10⁻⁴). The constraint at k_c = 23 is 8×10⁻⁶ and
  8×10⁻⁵, against 1.6×10⁻³ at k_c = 15: at D = 10 the k_c = 15 constraint overstates the error of Δ and λ, which sits
  in the high centre harmonics that Δ and λ hardly feel. D = 10.5 (prior 10⁻⁷): η = 0.3, λ = 3.068523 at k_c = 23
  against 3.068528 (2×10⁻⁶ relative), Δ = 1.1391179 against 1.1391146; η = 1, λ = 2.903687 against 2.903657 (10⁻⁵),
  Δ = 0.994302 against 0.994292. **Prior strength** at D = 10.5, η = 0.3 (k_c = 15): with 10⁻⁸ instead of 10⁻⁷,
  λ = 3.068601 (2×10⁻⁵ relative) and Δ = 1.138994 (10⁻⁴). D = 11, η = 0.3: λ = 3.112655 (k_c = 15), 3.112436
  (k_c = 23), 3.112305 (k_c = 15 with horizon series of order 30 evaluated at u = 0.04 instead of order 16 at 0.01); at
  D = 10 the longer horizon series gives 3.081369 against 3.081424. So the D = 10.5 and 11 values do not depend on
  resolution, prior or horizon treatment beyond about 10⁻⁴; what remains open at η = 1 is which of two solutions is the
  continuation of the branch (§5).
- **The gauge root drifts above D ≈ 8** (`data/sigdss_roots.jsonl`). The horizon-relabelling mode has the exact rate
  λ_g = mean(1 − c_x(τ, 1)), and det M vanishes there to 5×10⁻⁷ at D = 6. At η = 0.3 the real root near λ_g lies at
  0.80907 (λ_g = 0.80916, D = 8), 0.80872 (0.80787, D = 9), 0.87131 (0.80683, D = 10) and 1.06271 (0.81164, D = 11), and
  no other root lies in 0.05 ≤ λ ≤ 1.6 at D = 10 and 11. The departure, 9×10⁻⁵, 9×10⁻⁴, 0.064, 0.25, is the same with the
  longer horizon series (1.06371 at D = 11), so it is not a horizon-series artifact. Either the mode matrix becomes
  inaccurate in this direction above D ≈ 9, growing roughly tenfold per unit D like the near-null band, or a real mode
  crosses near λ_g and the two mix; this was not resolved. The physical root passes the k_c, prior and horizon-series
  checks above, but this drift is the main reason the D ≥ 10 growth rates are marked preliminary.
- **The near-null band** ([`sigdss_svd.py`](../solver/matter/sigdss_svd.py), `data/sigdss_svd.jsonl`). At D = 9, η = 1
  the smallest singular values of the Jacobian are 1.5×10⁻¹⁰, 6.6×10⁻¹⁰, 1.8×10⁻⁹, 5.3×10⁻⁹ relative (in Re/Im pairs),
  with singular vectors almost entirely in the centre datum q(τ), harmonics k = 7–11, and Δ components below 10⁻⁶. This
  is the "junk" band of the main solver (solver/notes.md: evanescent high-harmonic perturbations near the centre), here
  already at D ≈ 9 for the hedgehog. It is why Newton stalls at |F| ≈ 10⁻⁸ above D ≈ 9 without the smoothness prior,
  why the k_c = 15 constraint grows above D ≈ 9.7, and why the first marches stopped near D = 10.5 (§5). The prior
  itself moves the solution little: re-converging the D = 9 (η = 1) and D = 8 (η = 0.3) solutions with it changes Δ by
  1.4×10⁻⁸ and 1×10⁻⁹. The ℓ = 1 agent's k_c = 23 march with the prior stalls at D ≈ 9.2–9.8 in the same way (their
  `logs/march_l1_kc23m.log`).
- **Half-period symmetry** ([`sigdss_parity.py`](../solver/matter/sigdss_parity.py), `data/sigdss_parity.jsonl`). At
  D = 6, η = 1 the symmetric solution is moved to a layout in which every field carries all harmonics 0 … 15 (2016
  unknowns instead of 1015). Its residual is the same (1.7×10⁻¹⁰). The harmonics forbidden by u(τ + Δ/2) = −u(τ) were
  then perturbed by 2×10⁻² (relative 10⁻³ per field) and Newton run without the symmetry: it returns to |F| = 7×10⁻¹⁵
  with the forbidden harmonics at 5×10⁻¹⁰ and Δ unchanged to 3×10⁻¹⁰. So the symmetric solution is an isolated solution
  of the unrestricted problem, as the evolutions of HLPTA00 and LTHA02 indicate.
- **Unstable modes** (`data/sigdss_roots.jsonl`). At D = 6, η = 1 the sign changes of det M(λ) on 0.2 ≤ λ ≤ 7.4 (10
  grid points, refined) are λ = 0.848495, equal to the gauge rate λ_g = 0.8484955, and λ = 3.806224: one physical real
  unstable mode. Not checked: complex growth rates, and real ones of the opposite parity sector (perturbations that
  break the half-period symmetry); an argument-principle count over a strip of height 2ω, prepared in
  [`sigdss_count.py`](../solver/matter/sigdss_count.py), was too expensive for the shared machine (§9). At D = 4,
  Lechner's thesis reports one unstable mode of the DSS solution, and the γ found here matches the evolutions, which
  select the solution with one unstable mode.
- **Growth-rate root finding.** The first version of the root finder (a secant on det M) stopped at its iteration cap
  at two points, 3% and 8% from the root (D = 4 and 5.5 at η = 1); those values are discarded (records with
  `last_step` above 10⁻³ in the jsonl files) and were recomputed.

## 7. Large D: what the data allow

**Fits of λ(D)** ([`sigdss_fit.py`](../solver/matter/sigdss_fit.py); `data/sigdss_fit_lam_eta1.json`,
`data/sigdss_fit_lam_eta0.3.json`, `data/sigdss_fit_lam_l1ref.json`), D ≤ 9, λ = λ∞ + Σ_{k ≤ m} c_k n⁻ᵏ free, against
λ = 2 + Σ_{k ≤ m+1} c_k n⁻ᵏ (same number of parameters), rms residuals:

| coupling | D range | m | λ∞ (free) | rms free | rms with λ∞ = 2 |
|---|---|---|---|---|---|
| η = 1 | 5–9 (6 points) | 1 | 1.758 | 6.8e-2 | 5.0e-2 |
| | | 2 | 2.736 | 8.9e-3 | 2.1e-3 |
| | | 3 | 2.141 | 4.7e-4 | 8.4e-4 |
| | | 4 (1 dof) | 2.311 | 1.8e-5 | 1.3e-4 |
| | 6–9 (4 points) | 1 | 2.019 | 1.6e-2 | 1.6e-2 |
| η = 0.3 | 5–9 (6 points) | 1 | 1.870 | 5.5e-2 | 4.5e-2 |
| | | 2 | 2.660 | 7.7e-3 | 2.2e-3 |
| | | 3 | 2.147 | 5.3e-4 | 9.1e-4 |
| | | 4 (1 dof) | 2.337 | 6.5e-5 | 9.8e-5 |
| | 6–9 (4 points) | 1 | 2.081 | 1.2e-2 | 1.5e-2 |
| ℓ = 1 (reference) | 5–9 (5 points) | 1 | 1.731 | 6.6e-2 | 4.6e-2 |
| | | 2 | 2.658 | 6.6e-3 | 1.8e-3 |

- **The free λ∞ moves by ±0.5 with the number of terms**, and imposing λ∞ = 2 fits as well or better at m ≤ 2 and is
  within a factor 1.8 at m = 3. With four powers (one degree of freedom left) the free fits give 2.31 (η = 1) and 2.34
  (η = 0.3) and the λ∞ = 2 fits are 7 and 1.5 times worse; but at η = 1 the factor drops to 2 (λ∞ = 2.29) when the D = 9
  value is replaced by its k_c = 23 value, a change of 4×10⁻⁴, so this is not robust. The spread over fit choices,
  λ∞ ≈ 1.7–2.7, i.e. γ∞ ≈ 0.36–0.57, is the honest range; it is not a statistical error bar.
- **Adding D = 10 makes it worse** (`data/sigdss_fit_lam_eta1_D10.json`, `..._eta0.3_D10.json`): with D = 5–10 the free
  λ∞ is 1.82, 2.59, 1.91, 1.34 (η = 1) and 1.95, 2.65, 2.41, 3.19 (η = 0.3) for m = 1–4, with residuals of 3–5×10⁻³ at
  m = 3–4, ten times those on D ≤ 9. The D = 10 values are accurate (k_c = 23), so this is the curvature change of λ(D)
  near D ≈ 9.5 (§5): low-order expansions in 1/n do not describe λ(D) across D = 5–10.
- **Planted control** (same code path, same D grid, Gaussian noise 10⁻⁴; `--plant`): data made from
  λ = 2.3 + 6.5/n − 3/n² + 12/n³ give free λ∞ = 2.154, 2.444, 2.297 for m = 1, 2, 3, and the λ∞ = 2 fit is rejected at
  m = 3 (rms 7.6×10⁻⁴ against 5.8×10⁻⁵); data made with λ∞ = 2 (2 + 8.5/n − 8/n² + 15/n³) give 2.037, 2.180, 1.997
  and the λ∞ = 2 fit is as good (6.3×10⁻⁵ against 5.8×10⁻⁵). So the method separates λ∞ = 2.3 from 2 on D ≤ 9 when
  three powers of 1/n describe the data. On the real data they do not: the three-power free fit leaves residuals of
  5×10⁻⁴ that alternate in sign from point to point (truncation, not noise; the resolution error of λ is 3×10⁻⁶ at
  D = 6 and 4×10⁻⁴ at D = 9, from the k_c = 23 checks), so higher powers matter and the extrapolation is not
  controlled. The scalar needed D ≥ 24 and seven powers for 10⁻⁵ (GAMMA.md); D ≤ 9 is far from
  that.
- **The difference from the ℓ = 1 multiplet** ([`sigdss_diff.py`](../solver/matter/sigdss_diff.py),
  `data/sigdss_diff.json`) is smoother than either curve:

| D | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|
| γ(η = 1) − γ(ℓ = 1) | −0.00327 | −0.00286 | −0.00284 | −0.00299 | −0.00313 | −0.00330 |
| γ(η = 0.3) − γ(ℓ = 1) | −0.00339 | −0.00983 | −0.01460 | −0.01624 | −0.01654 | −0.01636 |

At η = 1 the offset is −0.003 at every D from 4 to 9; at η = 0.3 it grows to −0.0165 by D = 8 and is flat or slightly
decreasing in magnitude at D = 9. Fits in 1/n over D ≥ 5, 6, 7 with 0–2 powers give d∞ between −0.0030 and −0.0048 at
η = 1 and between −0.008 and −0.022 at η = 0.3 (spread over fit choices; the two-power fits on 4–5 points are the least
reliable). The ℓ = 1 values are another agent's work in progress at k_c = 23; the σ-model's k_c = 15 values differ from
k_c = 23 by at most 4×10⁻⁴ in λ (5×10⁻⁵ in γ) at D ≤ 9, well below the offsets.

## 8. Verdict

**Does γ∞ differ from 1/2 for the σ-model?** Not settled by these data.

- The DSS branch exists and was continued, without a fold, from D = 4 to D = 10 at η = 1 and η = 0.3, with one real
  unstable mode where this was checked (D = 6) and k_c = 23 checks at D = 6, 9, 10 (computed); at η = 0.3 it continues
  smoothly to D = 10.5 (preliminary). Unlike the CSS branch (fold at D = 8.544 for η = 0.14, SIGMA.md) it shows no sign
  of ending, so it is the candidate for a D = ∞ limit, as STRUCTURE.md expected. Its survival to large D is not
  established: above D ≈ 10 the near-null band of the centre data makes the solver unreliable (two solutions at
  D = 10.475 for η = 1).
- γ(D) rises monotonically, 0.116 → 0.342 (η = 1) and 0.116 → 0.325 (η = 0.3) between D = 4 and 10, with a change of
  curvature near D ≈ 9.5; at η = 0.3 it then has a maximum of about 0.326 near D = 10.5 and falls to 0.321 at D = 11
  (preliminary: checked against k_c, prior and horizon series, but the gauge root drifts there). Extrapolations from
  D ≤ 9 give γ∞ anywhere in about 0.36–0.57, fits with λ∞ = 2 imposed are as good as free ones, and adding D = 10 makes
  all fits worse. So γ∞ = 1/2 is neither confirmed nor excluded (not established). If the maximum at D ≈ 10.5 is real,
  γ(D) at η = 0.3 does not approach 1/2 from below the way the scalar's does, which would favour γ∞ ≠ 1/2.
- What the data do support is the structural half of the prediction at finite D: the σ-model's γ(D) depends on η and
  lies below the ℓ = 1 multiplet's by an amount that is nearly independent of D for 6 ≤ D ≤ 9 (−0.003 at η = 1, −0.016 at
  η = 0.3) and grows as η decreases, i.e. it vanishes in the direction η → ∞ where the σ-model becomes the multiplet.
  If these offsets persisted to D = ∞, the σ-model and the multiplet could not both have γ∞ = 1/2, which is the
  prediction's content (γ∞ is not protected and depends on η). The persistence is an extrapolation from D ≤ 9 (argued,
  not established).

## 9. Open issues and a proposed local-machine task

- **Larger D.** The decisive data are γ(D) for D ≈ 10–30 at two or more couplings. With k_c = 15 the solver becomes
  unreliable above D ≈ 10 (§5–6), and the near-null centre-harmonic band appears already at D ≈ 9. Proposed for the
  local machine (several hours, 2–4 threads, one process each): continue `sigdss_run.py cont ... --reg 1e-7` (prior in
  the tangent too) at k_c = 23 and 31 from the saved solutions (`data/sigdss_sol_m1up_D10.0000_eta1.0000.npy`,
  `data/sigdss_sol_m03hi_D11.0000_eta0.3000.npy`), with the long horizon series of `solver/spec_long.py` (matching node
  0.6, Ks ≥ 60) that carried the scalar to D ≈ 30; compare two prior strengths (10⁻⁸, 10⁻⁷) at each target, since the
  prior acts in the same harmonics as the near-null band; and run the ℓ = 1 multiplet (η = ∞) on the same grid with the
  same code so that the offsets of §7 can be extrapolated. Pilot: η = 1 from D = 10 to 10.5 at k_c = 23, which must
  reproduce a single smooth branch (the k_c = 15 runs found two solutions at D = 10.475); stop condition: two solutions at
  the same D, constraint above 10⁻³, or three failed steps in a row.
- **The turn of λ(D) at D ≈ 10.5** (§5) passed the resolution, prior and horizon-series checks at η = 0.3, but it rests
  on two points (D = 10.5, 11) in the range where the gauge root drifts (§6). Whether λ(D) keeps rising (γ falling
  well below 1/2), levels off, or turns back toward 2 is exactly what the larger-D runs above must decide; they should
  also explain the gauge-root drift (track the root near λ_g against D and resolution, and look for a second real
  mode near it).
- **Mode count.** Complex growth rates and the opposite parity sector were not counted (§6). `sigdss_count.py`
  (argument principle over Re λ ∈ [0.05, 12], −ω/2 ≤ Im λ ≤ 3ω/2, light-cone resonance poles removed) is ready but not
  run or tested; at k_c = 15 it needs a few hundred complex mode matrices.
- **D = ∞ inner selection** (task item 4) was not attempted.
- **η → ∞ check at large D.** The offsets of §7 use the ℓ = 1 agent's values; a run of this solver at η = ∞ (it
  reproduces their D = 4 solution to 10⁻¹⁵) would make the comparison self-contained.
- **η_c(D).** The DSS–CSS transition was located only at D = 4 (η_c = 0.170). Where it goes with D, and whether the DSS
  branch at fixed small η meets the CSS branch, was not followed.

## 10. Scripts and data

All in [`solver/matter/`](../solver/matter/), run from that directory with `nice -n 10 python3 <script>`, single process.
Data in `solver/matter/data/` (prefix `sigdss_`).

| Script | What it does |
|---|---|
| `sigdss_solver.py` | the σ-model DSS solver (class `SigDSS`), root finders for λ (`lam_model`, `lam_root`), `make` |
| `sigdss_run.py cont TAG SEED --par eps or D ...` | continuation in η⁻¹ or D, with λ at target points (`sigdss_{TAG}.jsonl`, `sigdss_sol_{TAG}_*.npy`, `sigdss_{TAG}.npz`) |
| `sigdss_run.py lam`, `roots`, `refine` | growth rate near a guess; all real roots on a grid; regrid to another k_c, re-solve, λ |
| `sigdss_parity.py SOL D eps kc` | the half-period symmetry check (§6) |
| `sigdss_svd.py SOL D eps kc` | weakest Jacobian directions (§6) |
| `sigdss_count.py SOL D eps kc` | argument-principle mode count (prepared, not run) |
| `sigdss_etac.py` | D = 4 fit of Δ(η) near η_c |
| `sigdss_fit.py FILES --eps E [--plant ...]` | fits of λ(D) with λ∞ free and fixed, planted controls |
| `sigdss_diff.py` | γ(σ) − γ(ℓ = 1) and its fits |
| `sigdss_tables.py` | tables from all jsonl files; `sigdss_tables.py tangent`: secant tangents and centre harmonics along the marches (§5) |

Data: `sigdss_d4eta15.jsonl` (D = 4 against η), `sigdss_lam_d4b.jsonl`, `sigdss_lam_pts.jsonl` (growth rates at D = 4,
5, 5.5), `sigdss_m1up*.jsonl` and `sigdss_m03up*.jsonl` (the D marches at η = 1 and 0.3), `sigdss_m1hi.jsonl` and
`sigdss_m03hi.jsonl` (D = 10–11 with the stronger prior), `sigdss_refine_kc23.jsonl`, `sigdss_refine_reg8.jsonl`,
`sigdss_refine_ks30.jsonl`, `sigdss_lam_kc23.jsonl` (the checks of §6),
`sigdss_parity.jsonl`, `sigdss_roots.jsonl`, `sigdss_svd.jsonl`, `sigdss_fit_*.json`, `sigdss_diff.json`,
`sigdss_etac_d4.json`, `sigdss_l1_reference.jsonl` (the ℓ = 1 agent's values, copied), `sigdss_seed_*.npy` (their
solutions used as seeds, copied), solutions `sigdss_sol_*.npy`, logs `sigdss_log_*.txt`. The record files from before
the root-finder fix (`sigdss_lam_d4.jsonl`, `sigdss_m1up_part1.jsonl`) contain two unconverged growth rates, marked by
`last_step` above 10⁻³.

## 11. References

- [HLPTA00] S. Husa, C. Lechner, M. Pürrer, J. Thornburg, P. C. Aichelburg, PRD 62, 104007 (2000),
  [gr-qc/0002067](https://arxiv.org/abs/gr-qc/0002067). Table I and the quoted sentences were read from the arXiv HTML
  version in this session.
- [LTHA02] C. Lechner, J. Thornburg, S. Husa, P. C. Aichelburg, PRD 65, 081501 (2002),
  [gr-qc/0112008](https://arxiv.org/abs/gr-qc/0112008). Read from the arXiv HTML version: η_c, the fit form and range,
  γ_DSS ≈ 0.11, and the direct construction of the DSS solution "following Gundlach"; the values of Δ in their Fig. 1
  are not given in the text.
- [L01] C. Lechner, PhD thesis, University of Vienna (2001), [gr-qc/0507009](https://arxiv.org/abs/gr-qc/0507009). Only
  the abstract and chapters 1–3 could be read here ("For large couplings we construct a DSS solution. Both solutions
  have one unstable mode."); chapter 4, with the DSS construction and its numbers, could not be retrieved.
- [OVCU07] I. Olabarrieta, J. F. Ventrella, M. W. Choptuik, W. G. Unruh, PRD 76, 124014 (2007),
  [arXiv:0708.0513](https://arxiv.org/abs/0708.0513), as summarised in [LITERATURE.md](LITERATURE.md).
- The ℓ-multiplet agent's solver and results (worktree `agent-a5bec5ffdf3db4330`, work in progress), as described in §3
  and §7.

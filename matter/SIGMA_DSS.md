# The σ-model's echoing (DSS) critical solution in D dimensions

DRAFT — results pending.

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

- **Δ agrees with HLPTA00 to within the spread of their two families** for 0.225 ≤ η ≤ 10 (differences ≤ 0.003, i.e.
  ≤ 0.6%), and within 1–2% at η = 0.2. At η = 0.18 it does not (1.276 against 1.10): there Δ varies as
  dΔ/dη ≈ −a/(η − η_c) ≈ −35, so the published value corresponds to η ≈ 0.185 on our curve, and HLPTA00 say their
  evolution is only approximately DSS there.
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

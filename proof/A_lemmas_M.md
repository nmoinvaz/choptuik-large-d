# Theorem A with M harmonics: the start-value lemma and lemmas S, 6a, 6b₀, 6b₁

This note generalizes the parts of Theorem A(M = 1) ([A_certify.md](A_certify.md) §4.4–4.5, §11, §12) that turn a point
certificate into the theorem, from one harmonic to M harmonics (M = 3 and 5). The point certificate itself (validated
integration and a Krawczyk test at one radius, `solver/proof/certify_M*.py`, notes in `proof/A_M.md`) is built
separately and is not duplicated here. Everything is written so that it can be applied to that certificate's box and
path once they exist; §7 lists exactly what waits for them.

Status words as in AGENTS.md: *proved* (by hand), *certified* (computer-assisted, the trusted code named),
*conditional* (with the hypothesis named), *numerical evidence*. The one fact assumed throughout, as at M = 1, is
Braaksma's summability theorem for the existence of the lateral solutions y± and their asymptotics
([C_selection.md](C_selection.md) Lemma B4(M)). Nothing here has been reviewed independently yet.

Contents

- [Summary](#summary)
- [1. The M-harmonic setting](#1-the-m-harmonic-setting)
- [2. The start-value fixed point (step (i))](#2-the-start-value-fixed-point-step-i)
- [3. Lemma S and lemma 6b₀](#3-lemma-s-and-lemma-6b)
- [4. Lemma 6a](#4-lemma-6a)
- [5. Lemma 6b₁](#5-lemma-6b)
- [6. Theorem A(M), conditional on the point certificate](#6-theorem-am-conditional-on-the-point-certificate)
- [7. What waits for the point certificate](#7-what-waits-for-the-point-certificate)
- [8. Mutation table](#8-mutation-table)
- [9. Cost, and a proposed task for the local machine](#9-cost-and-a-proposed-task-for-the-local-machine)
- [10. Reproduce](#10-reproduce)
- [11. Proposed text for A_certify.md and PROOF.md](#11-proposed-text-for-a_certifymd-and-proofmd)

## Summary

| item | M = 3 | M = 5 | where |
|---|---|---|---|
| step (i): the start-value fixed point with derivatives in all M + 1 parameters | **certified** at the `hier.py` zero, at points and over real (±10⁻⁶) and complex (±10⁻³(1 + i)) boxes, on the rays 3π/8 and 5π/16 | **certified**, the same | §2 |
| M = 1 reproduces `fixed_point.py` | start values and start data bit for bit; constants to 10⁻¹⁶ | | §2.4 |
| lemma S, lemma 6b₀ | **certified** over x̂ ± 10⁻⁶, R₊ = 0.01 (the M = 1 domain) | **certified** over x̂ ± 10⁻⁶, R₊ = 0.0064 | §3 |
| lemma 6a | **certified** with the certificate's state box at ξ₁ = 0.3: ρ₆ₐ = 0.041 (`lemmas_M_runs/lemma6a_M3_cert.*`); the flat estimate Λ ≤ 0.29 | the flat estimate **certified** (Λ ≤ 0.42) | §4 |
| lemma 6b₁ | **certified** on the point certificate's box p̂ ± 1.1×10⁻⁹ (C¹ mode, 183 discs, check C1–C8 passes) | design only (366 discs): a local-machine task | §5 |
| Theorem A(M) | **certified** (H1–H3 of §6 hold, §7) | **conditional** on the point certificate (H1–H3 of §6) | §6–7 |

- **What carries over verbatim:** the inverse operators, domains and kernel bounds (the fast eigenvalue repeats across
  harmonics), the identification with y± (the same Stokes directions), lemma D, lemma G, the geometric gluing check (its
  code is reused unchanged) and the 6a argument.
- **What changes:** the labels s_μ = iμω.
  - On rays their weight cancels exactly, |(t/z)^{s_μ}| = 1, and the new "diag" splitting removes them from the
    contraction.
  - Elsewhere they are perturbations of size |s_μ|R₊. That forces R₊ ∝ 1/(Mω) in lemma S at M = 5, smaller start radii
    at M = 5, and more D-conditions in 6a: all (M + 1)/2 complex J_{D,μ}.
  - They make y⁺ about 30 times more sensitive to the parameters near |ξ| = 0.19. Together with the restart of each arc
    from a full box, that is why the M = 1 covering method fails on the innermost row at M = 3. The C¹ mean-value
    mode (§5.3) replaces it.
- **Found along the way:**
  - `fixed_point.py`'s weight exponent adapts to a wrong series, so a mutated coefficient yields a weaker but true
    lemma; `fixed_point_M.py` now checks the structural vanishing orders.
  - `hier.py` prints (B_ν, C_ν) only at intermediate iterates; `refine_zero_M.py` gives them at the final (A, ω).
  - The coords mutation of 6a, not caught at M = 1 (ρ₆ₐ = 0.12), is still not caught at M = 3, but ρ₆ₐ = 0.89 there.

## 1. The M-harmonic setting

The truncated inner system is the one `solver/inner_stokes/lateral_np.py`, `inner_np.py` and `hier.py` integrate:
D, S on the odd harmonics |μ| ≤ M, C on the even harmonics |ν| ≤ M + 1, every product projected onto its result's
harmonics. That is 2(M + 1) + (M + 2) complex unknowns: 13 at M = 3, 19 at M = 5 (7 at M = 1). The profile and the
real parameters are those of `hier.py`:

```math
F = A\sin\theta + \sum_{\nu = 3, 5, \ldots, M}\big(B_\nu\sin\nu\theta + C_\nu\cos\nu\theta\big), \qquad x = (A,\ \omega/\omega_\infty,\ B_3, C_3, \ldots, B_M, C_M) \in \mathbb R^{M+1},
```

with $`F_{\pm1} = \mp iA/2`$ and $`F_{\pm\nu} = \mp iB_\nu/2 + C_\nu/2`$, holomorphic in x, so complex parameter boxes
make sense. In the variables of `fixed_point.py` (z = ξ², D = ξP, S = zQ, V = Q − (1 + s_μ)P, s_μ = iμω,
W = V + [(C − 1)Q], Ê = [P²/2]):

```math
P' = \frac{V + [(C-1)Q]}{2z}, \qquad
V' = \frac{V}{2z^2} + \frac{[(C-1)Q] + [(C-1)W]}{2z^2} + \frac{-(2+s_\mu)\,Q - (1+s_\mu)\,W + [C\hat E P]}{2z}, \qquad
C' = \tfrac12\,[C\hat E] .
```

Two structural facts organize everything below.

- **The fast eigenvalue repeats across harmonics.** The only rank-one singular term is V/(2z²), the same scalar on every
  V_μ. So the inverse operator of step (i), its domains, its kernel bounds and the rate a are exactly those of M = 1, and
  so are the Stokes directions: every label has the same exponential $`e^{-1/(2z)}`$, the Borel singularities sit at the
  same places, and y± are defined on the same sectors as at M = 1 ([C_selection.md](C_selection.md) Lemma B4(M)).
  It is also why Costin's rank-one theorems do not apply as stated and Braaksma's theorem is the one cited.
- **The labels enter only through s_μ = iμω in the O(1/z) terms.** The fast mode of label μ is
  $`e^{-1/(2z)}z^{-(3+2s_\mu)/2}`$ to leading order (times $`z^{\langle F^2\rangle}`$ with gravity), and
  $`\lvert z^{-s_\mu}\rvert = e^{\mu\omega\arg z}`$. On a ray from 0 this factor is the same at both ends of every
  integral, so it cancels; on any other path it does not, and its size, up to $`e^{3\pi\lvert\mu\rvert\omega/4}`$ over the
  sector, is the label weight (the $`e^{\pi\mu\omega/2}`$ of PROOF.md). Where the weights cannot be cancelled they are
  carried as perturbations of size |s_μ| |z|, which grow like Mω.

The selection condition is y⁺ ≡ y⁻, equivalently that the (M + 1)/2 complex Stokes multipliers S_1, S_3, …, S_M all
vanish: M + 1 real conditions on the M + 1 real parameters.

The system code is checked once, here, and then shared by every lemma (`lemma6_usys_M.py`, exit status 0 only if all
pass, run at M = 3 and 5):

| test | what is compared | M = 3 | M = 5 |
|---|---|---|---|
| (a) | M = 1: `SysXiM` = `validated_int.Sys`, `SysUM` = `lemma6_usys.SysU` (rhs and lin) | 1.4×10⁻⁵⁵ | 1.4×10⁻⁵⁵ |
| (b) | `SysXiM` against `inner_np.Inner.rhs` (numpy, the code `hier.py` integrates), relative | 3.5×10⁻¹⁶ | 1.4×10⁻¹⁵ |
| (c) | `SysUM` against `SysXiM` through the exact change of variables and the chain rule | 4.0×10⁻⁵⁸ | 6.4×10⁻⁵⁸ |
| (d) | `fixed_point.residual_numerators` (the system step (i) encodes) against `SysUM` | 9.2×10⁻⁵⁹ | 8.0×10⁻⁵⁹ |
| (e) | lin against the series derivative of rhs, both classes, ω direction included | 5.3×10⁻⁵⁶ | 5.7×10⁻⁵⁵ |
| (f) | `rhs_gen_M` (the disc lemma's right-hand side) against `SysUM.rhs` | 1.6×10⁻⁵⁸ | 2.5×10⁻⁵⁷ |

So the chain from the system `hier.py` solves, through the ξ-form of the integrators, to the (P, V, C) form of the
start-value lemma and the disc lemma, is closed. (b) is the only link to independent code, and it is in double
precision.

## 2. The start-value fixed point (step (i))

Code: `solver/proof/fixed_point_M.py`. It imports the majorant machinery of `fixed_point.py` unchanged (`Maj`, `MD`,
`conv`, `poly_maj`, `tail_maj`, `intbound`, `residual_numerators`) and adds the harmonic sets of M, the profile
parameters, a second splitting of the fast row, jets in all M + 1 parameters, and the start ray as a parameter.

### 2.1 Statement

**Lemma FP(M).** Let M be odd, θ ∈ (π/4, 3π/4) the argument of the start ray (so arg z = 2θ and the kernel rate
a = −cos(2θ)/2 is positive; θ = 3π/8 is the ray of `fixed_point.py`, θ = 5π/16 that of `inner_np.py`), r₀ > 0, and X a
parameter set (a point, a real box or a complex box in the M + 1 parameters). Let Y_N be the formal series truncated
at N terms, N optimal at the centre of X. Suppose the computation of `fixed_point_M.run_M` succeeds:

- every coefficient of the residual numerators below its structural first nonzero order is a ball containing 0;
- the integral operator T has weighted Lipschitz constant Λ < 1 on the ball of radius ρ, and ‖T(0)‖ + Λρ < ρ.

Then for every x ∈ X, on the ray arg ξ = θ, 0 < |ξ| ≤ r₀, the truncated M-harmonic system has a solution
Y = Y_N + η with ‖η‖ ≤ ρ, and it is the only one in that ball. Every solution with η = o(|z|^{k+1}) along the ray is this
one, and the same holds on the conjugate ray. So y⁺ is the solution on arg ξ = θ and y⁻ the one on arg ξ = −θ. With
`--sector=d` the same holds on every ray |arg ξ − θ| ≤ d, and the fixed points there form one analytic function.

### 2.2 Proof

**What carries over verbatim from §4.4.** None of the following uses M; the harmonic sets enter only as the index sets
of the convolutions.

- the integral equation for the slow components p and c, integrated from 0;
- the norms: r^{−k} on p and v, r^{−k−1} on c, and the weighted maximum over components with Perron weights;
- the majorant calculus: every function is bounded by K|t|^q, products and harmonic convolutions by the triangle
  inequality, derivative majorants by the product rule, and differences by the mean value inequality on the convex ball;
- the forcing: the residual numerators vanish identically below their first order, which is checked by containment of 0;
- the contraction check;
- the identification of the fixed point with y⁺: the operator is of Volterra type, every bound is non-decreasing in R,
  the regular ODE has unique solutions, and y⁺ − Y_N = O(z^N) on its sector by Braaksma's theorem. That sector does not
  depend on M (§1).

**What is new: the fast row.** Two splittings are proved.

*Plain* (the splitting of §4.4). The principal part is V/(2z²), the kernel is $`e^{1/(2t)-1/(2z)}`$, and on the ray
$`\lvert e^{1/(2t)-1/(2z)}\rvert = e^{-a(1/\lvert t\rvert - 1/\lvert z\rvert)}`$. The label terms −(3 + 2s_μ)V/(2z) stay in
Φ_v. Their Lipschitz entries are |3 + 2s_μ|/2 times the weighted bound of $`t^{k-1}`$, min(1/k, R/a). At M = 5,
μ = ±5, |3 + 2s_μ|/2 = 19.1 and the entry is 0.48 at k = 40. That is why the plain contraction degrades with M (§2.4).

*Diag* (rays only). The principal part is V/(2z²) − (3 + 2s_μ)V/(2z). With

```math
(2+s)\,Q + (1+s)\,W = (3+2s)\,V + (2+s)(1+s)\,P + (1+s)\,[(C-1)Q],
```

the remainder is $`\Phi_v = \frac{[(C-1)Q] + [(C-1)W]}{2z^2} + \frac{-(2+s)(1+s)P - (1+s)[(C-1)Q] + [C\hat EP]}{2z} + \rho_v`$.
The homogeneous solution is $`\varphi_\mu(z) = e^{-1/(2z)}z^{-(3+2s_\mu)/2}`$ (principal branch on the ray), and

```math
v(z) = \int_0^z K_\mu(t, z)\,\Phi_v(t)\,dt, \qquad K_\mu(t, z) = e^{\frac{1}{2t}-\frac{1}{2z}}\Big(\frac tz\Big)^{(3+2s_\mu)/2} .
```

On a ray t/z = ϱ/r ∈ (0, 1] is real and positive, so $`\lvert(t/z)^{s_\mu}\rvert = 1`$ for every label and

```math
\lvert K_\mu(t, z)\rvert = e^{-a\left(\frac1\varrho - \frac1r\right)}\Big(\frac\varrho r\Big)^{3/2} \le e^{-a\left(\frac1\varrho - \frac1r\right)} .
```

So both kernel bounds of §4.4 hold unchanged (∫ ≤ r^{m+2}/a for m ≥ −2, and ∫ ≤ r^{m+1}/(m+1) from |K| ≤ 1), and the
label-dependent diagonal entries leave the Lipschitz matrix.

- *Uniqueness.* A solution with η = o(r^k) satisfies the integral equation: $`\lvert\varphi_\mu(z)\rvert = e^{a/\lvert z\rvert}\lvert z\rvert^{-3/2}`$
  grows faster than every power as z → 0 along the ray, so the homogeneous component must vanish. This is the
  boundary-term argument of §4.4, with φ_μ in place of $`e^{-1/(2z)}`$.
- *Sector.* The ray integrals from 0 are path-independent in the sector: $`K_\mu(t, z)\Phi_v(t)`$ is analytic in t there
  (one continuous branch of the power), and it decays as t → 0 because Re(1/t) → −∞ uniformly in the sector. This is
  the argument of §11.3, last paragraph. ∎

The diag splitting is not valid on other paths. There $`\lvert(t/z)^{s_\mu}\rvert = e^{-\mu\omega(\arg t - \arg z)}`$, the label
weight, which reaches $`e^{3\pi\lvert\mu\rvert\omega/4}`$ on the paths of lemma S. The code refuses to combine the diag
splitting with lemma S's kernel bound (mutation `diagpath`, §8).

### 2.3 Start data with parameter derivatives

`start_data_M` is the interface `fixed_point.start_data` gives at M = 1 (§4.5), for M + 1 parameters:

- Y_N and its derivatives in all M + 1 parameters, exactly, by first-order jets carried through the same recursion over
  the real box X;
- η over X, from the fixed point over X;
- the derivatives of η by Cauchy's estimate, |∂η/∂x_i| ≤ sup|η|/c over the complex box X + c(1 + i). The polydisc of
  radius c around each point of X lies in that box, and η is holomorphic there (a uniform limit of holomorphic Picard
  iterates).

```
import fixed_point_M as FPM
xi0, vals, ders, (eta_bound, deta_bound) = FPM.start_data_M(x, M, r0, arg, theta_sign=+1, cauchy="1e-3", diag=True)
```

x is a list of M + 1 real arb balls (the box X, made at 320 bits or more from decimal strings), in the order
(A, ω/ω∞, B₃, C₃, …). `vals` is the state at ξ₀ = r₀e^{iθ} in `lateral_np.py`'s order
[D_μ (odd μ ascending), S_μ, C_ν (even ν ascending)], valid for every x ∈ X, and `ders[i]` is the list of ∂/∂x_i of
those components, valid over X. ω∞ = 2π/1.64389375, as in the solvers.

### 2.4 Results

**M = 1 reproduces fixed_point.py** (`python3 fixed_point_M.py --check-m1`, exit status 0, `lemmas_M_runs/check_m1.log`):

| | fixed_point.py | fixed_point_M.py at M = 1 |
|---|---|---|
| at x̂, ray 3π/8, r₀ = 0.1: N, k, contraction, forcing, ρ | 53, 40, 0.291842, 1.315553×10⁴⁸, 7.430851×10⁴⁸ | the same; upper bounds equal to 8×10⁻¹⁷ relative |
| complex box ±10⁻³(1 + i): contraction, forcing, ρ | 0.292255, 8.569284×10⁶⁹, 4.843148×10⁷⁰ | the same, to 10⁻²¹ relative |
| start values on both rays (7 balls each) | e.g. D₊₁ = 0.23063658495062 − 0.12113752692877i, radius 2.4×10⁻³⁴ | bit for bit, midpoints and radii |
| `start_data` over x̂ ± 10⁻⁹, Cauchy radius 10⁻³ (7 values, 14 derivatives) | | bit for bit |

The constants differ at 10⁻¹⁶ only through how a ball's radius is counted. `fixed_point.py` makes its rate √2/4 at
import time, at 53 bits, and takes Python's `max` of balls, where an undecided comparison keeps the first ball; this
module takes the largest upper bound. That can understate `fixed_point.py`'s bound by one ball radius, about 10⁻¹⁶
relative, which is immaterial next to the margin (0.29 against 1).

**M = 3 at the hier.py zero** x̂ = (2.726475886, 1.015603095, 0.144419478488, 0.011087425887). (A, ω/ω∞) are as
`hier.py --schur` printed them; (B₃, C₃) are re-solved at that (A, ω) by `refine_zero_M.py`, which leaves
|r₁| = 3×10⁻⁶ and |r₃| = 1.4×10⁻¹⁰ (`lemmas_M_runs/zero_M3.json`). Double-precision numerics, not rigorous; x̂ is within
about 5×10⁻¹⁰ of the zero in (A, ω/ω∞). Rows of kind "lemma" are the recommended configurations, which must hold;
"limit" rows are measured and reported; "control" must fail. `python3 fixed_point_M.py --table --Ms=3`, exit status 0
(`lemmas_M_runs/fixed_point_M_table_M3.json`):

| M | kind | configuration | start ray, r₀ | splitting | parameters | N | contraction | sup\|η\| at \|z\| = R | holds |
|---|---|---|---|---|---|---|---|---|---|
| 3 | lemma | point | 3pi/8, 0.085 | diag | point | 72 | 0.2564 | 2.9×10⁻³⁶ | yes |
| 3 | lemma | point | 3pi/8, 0.085 | plain | point | 72 | 0.4160 | 3.7×10⁻³⁶ | yes |
| 3 | lemma | real box | 3pi/8, 0.085 | diag | real ±1e-6 | 72 | 0.2564 | 1.1×10⁻¹² | yes |
| 3 | lemma | complex (Cauchy) box | 3pi/8, 0.085 | diag | complex ±1e-3(1+i) | 72 | 0.2571 | 2.4×10⁻⁹ | yes |
| 3 | lemma | point | 5pi/16, 0.0741 | diag | point | 94 | 0.3159 | 5.5×10⁻⁴⁶ | yes |
| 3 | lemma | point | 5pi/16, 0.0741 | plain | point | 94 | 0.5501 | 8.4×10⁻⁴⁶ | yes |
| 3 | lemma | real box | 5pi/16, 0.0741 | diag | real ±1e-6 | 94 | 0.3159 | 2.6×10⁻²⁰ | yes |
| 3 | lemma | complex (Cauchy) box | 5pi/16, 0.0741 | diag | complex ±1e-3(1+i) | 94 | 0.3168 | 5.9×10⁻¹⁷ | yes |
| 3 | lemma | sector (old lemma) | 3pi/8 ± 0.2, 0.1 | diag | point | 53 | 0.5408 | 2.1×10⁻²⁷ | yes |
| 3 | limit | point | 3pi/8, 0.1 | diag | point | 53 | 0.3359 | 1.4×10⁻²⁷ | yes |
| 3 | limit | point | 3pi/8, 0.1 | plain | point | 53 | 0.5613 | 2.1×10⁻²⁷ | yes |
| 3 | limit | real box | 3pi/8, 0.1 | diag | real ±1e-6 | 53 | 0.3359 | 2.2×10⁻⁶ | yes |
| 3 | limit | complex (Cauchy) box | 3pi/8, 0.1 | diag | complex ±1e-3(1+i) | 53 | 0.3374 | 5.0×10⁻³ | yes |
| 3 | limit | sector (old lemma) | 3pi/8 ± 0.2, 0.1 | plain | point | 53 | 0.9888 | 8.2×10⁻²⁶ | yes |
| 3 | control | r0 too large | 3pi/8, 0.3 | diag | point | 3 | 16.3920 | — | **no** |

**M = 5 at the hier.py zero** (`--table --Ms=5`, `lemmas_M_runs/fixed_point_M_table_M5.json`):

| M | kind | configuration | start ray, r₀ | splitting | parameters | N | contraction | sup\|η\| at \|z\| = R | holds |
|---|---|---|---|---|---|---|---|---|---|
| 5 | lemma | point | 3pi/8, 0.07 | diag | point | 105 | 0.2500 | 3.2×10⁻⁴⁵ | yes |
| 5 | lemma | point | 3pi/8, 0.07 | plain | point | 105 | 0.4299 | 4.2×10⁻⁴⁵ | yes |
| 5 | lemma | real box | 3pi/8, 0.07 | diag | real ±1e-6 | 105 | 0.2500 | 5.1×10⁻¹⁷ | yes |
| 5 | lemma | complex (Cauchy) box | 3pi/8, 0.07 | diag | complex ±1e-3(1+i) | 105 | 0.2506 | 1.2×10⁻¹³ | yes |
| 5 | lemma | point | 5pi/16, 0.0622 | diag | point | 132 | 0.3095 | 3.4×10⁻⁵⁷ | yes |
| 5 | lemma | point | 5pi/16, 0.0622 | plain | point | 132 | 0.5838 | 5.6×10⁻⁵⁷ | yes |
| 5 | lemma | real box | 5pi/16, 0.0622 | diag | real ±1e-6 | 132 | 0.3095 | 1.6×10⁻²⁶ | yes |
| 5 | lemma | complex (Cauchy) box | 5pi/16, 0.0622 | diag | complex ±1e-3(1+i) | 132 | 0.3102 | 3.6×10⁻²³ | yes |
| 5 | lemma | sector (old lemma) | 3pi/8 ± 0.2, 0.1 | diag | point | 52 | 0.6384 | 6.8×10⁻²² | yes |
| 5 | limit | real box | 3pi/8, 0.085 | diag | real ±1e-6 | 72 | 0.3250 | 2.3×10⁻⁶ | yes |
| 5 | limit | complex (Cauchy) box | 3pi/8, 0.085 | diag | complex ±1e-3(1+i) | 72 | 0.3263 | 5.3×10⁻³ | yes |
| 5 | limit | point | 3pi/8, 0.1 | diag | point | 52 | 0.4142 | 4.2×10⁻²² | yes |
| 5 | limit | point | 3pi/8, 0.1 | plain | point | 52 | 0.8046 | 1.3×10⁻²¹ | yes |
| 5 | limit | real box | 3pi/8, 0.1 | diag | real ±1e-6 | 52 | 0.4222 | 5.2×10⁻¹ | yes |
| 5 | limit | complex (Cauchy) box | 3pi/8, 0.1 | diag | complex ±1e-3(1+i) | 52 | 44.8089 | — | **no** |
| 5 | limit | sector (old lemma) | 3pi/8 ± 0.2, 0.1 | plain | point | 52 | 1.4162 | — | **no** |
| 5 | control | r0 too large | 3pi/8, 0.3 | diag | point | 3 | 18.6888 | — | **no** |

**Start data with derivatives** (`--data-table`), real boxes around x̂, Cauchy radius 10⁻³, diag splitting. The value
and derivative radii are those of the balls handed to the integrator: the true variation over the box plus the
enclosure. The Cauchy part is the bound on |∂η/∂x| in the (p, v, c) variables:

| M | start ray, r₀ | real box | value radius | derivative radius | Cauchy part (p, v, c) | sup\|η\| over X |
|---|---|---|---|---|---|---|
| 3 | 3pi/8, 0.085 | ±1e-9 | 5.2×10⁻¹⁰ | 2.9×10⁻⁸ | 2.4×10⁻⁶ | 1.1×10⁻¹⁵ |
| 3 | 3pi/8, 0.085 | ±1e-6 | 5.2×10⁻⁷ | 1.3×10⁻⁶ | 2.4×10⁻⁶ | 1.1×10⁻¹² |
| 3 | 5pi/16, 0.0741 | ±1e-9 | 2.6×10⁻¹⁰ | 5.6×10⁻¹⁰ | 5.9×10⁻¹⁴ | 2.6×10⁻²³ |
| 3 | 5pi/16, 0.0741 | ±1e-6 | 2.6×10⁻⁷ | 5.6×10⁻⁷ | 5.9×10⁻¹⁴ | 2.6×10⁻²⁰ |
| 5 | 3pi/8, 0.07 | ±1e-9 | 7.4×10⁻¹⁰ | 2.7×10⁻⁹ | 1.2×10⁻¹⁰ | 5.1×10⁻²⁰ |
| 5 | 3pi/8, 0.07 | ±1e-6 | 7.4×10⁻⁷ | 2.7×10⁻⁶ | 1.2×10⁻¹⁰ | 5.1×10⁻¹⁷ |
| 5 | 5pi/16, 0.0622 | ±1e-9 | 3.9×10⁻¹⁰ | 1.2×10⁻⁹ | 3.6×10⁻²⁰ | 1.6×10⁻²⁹ |
| 5 | 5pi/16, 0.0622 | ±1e-6 | 3.9×10⁻⁷ | 1.2×10⁻⁶ | 3.6×10⁻²⁰ | 1.6×10⁻²⁶ |

**Where the label weights show: the forcing over parameter boxes.** At a point, the forcing is the optimal truncation
error of the series, and it is far smaller near the zero than away from it. At the M = 3 zero, sup|η| at r₀ = 0.1 is
1.4×10⁻²⁷; at the cruder point (B₃, C₃ off by 6×10⁻¹⁰) it was 2.3×10⁻²⁴. That is expected: at a zero of the
multipliers the late orders lose their leading growth. Over a box they regain it at first order, because the zero is
simple. In ball arithmetic the residual's higher coefficients then lose all cancellation, and the forcing over a box of
half-width h grows like a constant times h:

- M = 1, r₀ = 0.1: about 10⁻⁷ h (§4.5);
- M = 3, r₀ = 0.1: about 2h;
- M = 5, r₀ = 0.1: about 5×10⁵ h. Over the complex box ±10⁻³ the contraction itself fails (Λ = 45), because the ball
  of η is no longer small.

A cruder estimate of the true variation (the jump of harmonic ν at the start point, ∝ h|∂S_ν|·|K′_ν|·e^{−1/(2R)},
with $`\lvert K'_\nu\rvert \sim e^{\pi\nu\omega/2}`$) is many orders smaller, so most of this is dependency loss in the
ball recursion. It is still a real cost. The label weights make the late orders of the higher harmonics large, and the
cure is a smaller start radius (a factor $`e^{-1/(2R)}`$): at r₀ = 0.0741 (M = 3) and 0.0622 (M = 5) on `inner_np.py`'s
ray, the complex box gives sup|η| = 5.9×10⁻¹⁷ and 3.6×10⁻²³. A mean-value form of the forcing (the residual at the
centre plus the box times its derivative from jets) would remove most of the loss; it is not implemented.

**Recommendation for the point certificate:**

- M = 3: the ray θ = 5π/16 at r₀ = 0.0741 (the start point of `inner_np.py`), diag splitting. Cauchy part of the
  derivatives 5.9×10⁻¹⁴, sup|η| over a 10⁻⁹ box 2.6×10⁻²³.
- M = 5: the same ray at r₀ = 0.0622, or θ = 3π/8 at r₀ ≤ 0.07.
- At M = 5, r₀ = 0.1 is too large for any box, and 0.085 gives Cauchy bounds too weak to use.

## 3. Lemma S and lemma 6b₀

Code: `solver/proof/lemma6_sector_M.py`.

### 3.1 Statement

**Lemma S(M).** Let R₊ > 0, U₀ = √2/R₊ and D = {z : 1/z ∈ U₀ + K}, K = {−3π/4 ≤ arg w ≤ 0}, as in §11.3. Suppose
the computation of `lemma_S_M` succeeds for X: plain splitting, the strict kernel bound with
a_eff = √2/4 − kR₊/2 (no slow-type alternative on fast rows), weights symmetric under μ ↦ −μ, Λ < 1 and
F + Λρ < ρ. Then for every x ∈ X the truncated M-harmonic system has a solution Y = Y_N + η on D, with η analytic in
the interior of D and in the weighted ball. It is unique in that ball, and it is y⁺.

**Lemma 6b₀(M).** If lemma S(M) holds with R₊ and the old sector lemma (Lemma FP(M) with `--sector=0.2` at θ = 3π/8 and
radius 0.1) holds for X, then for every x ∈ X, y⁺ extends analytically to the interior of
Ω₀ = {0 < |ξ| ≤ (R₊ sin 0.4)^{1/2}, 0 ≤ arg ξ ≤ 3π/8}, continuously to its boundary, and to the real segment
0 < ξ ≤ (R₊/√2)^{1/2}.

### 3.2 Proof

**Verbatim from §11.3–11.4.** These parts do not use M:

- the domain D, its three geometric properties (radius and coverage, star shape, the path), and the fast operator
  along u + s e^{−3iπ/4};
- the kernel inequality, which gives the bound $`\lvert z\rvert^{m+2}/(a - (m+2)R_+/2)`$ and hence a_eff;
- the symmetric weights;
- the identification proof. It uses only that y⁺ − Y_N = O(|z|^N) on a sector containing D(U₀′) for large U₀′, and
  that sector is the same for every M (§1).

Lemma 6b₀(M) follows from lemma S(M) and the old sector lemma exactly as in §11.4: D contains the directions
0 ≤ arg z ≤ 3π/4 − 0.4 up to radius R₊ sin 0.4; the sector lemma covers 3π/4 − 0.4 ≤ arg z ≤ 3π/4 up to |z| = 0.01; and
the two overlap in a connected set where both are y⁺.

**What changes.** The path of lemma S is not a ray, so the splitting must be plain, and the label terms are
perturbations with Lipschitz entries |3 + 2s_μ|R₊/(2a_eff). These grow like |μ|ω, so lemma S needs R₊ roughly
proportional to 1/(Mω). The old sector lemma is on rays, so it can use the diag splitting, and the label weight drops
out there. ∎

### 3.3 Results

**Where lemma S closes** (`--scan`, at the point x̂; `lemmas_M_runs/lemmaS_scan_M3.json`, `lemmaS_scan_M5.log`). The
contraction constant, by r₊ (R₊ = r₊²) and weight exponent k:

| r₊ | M = 1 (§11.3) | M = 3, k = 10 / 20 / 30 | M = 5, k = 10 / 20 / 30 |
|---|---|---|---|
| 0.1 | 0.4903 (k = 30) | 0.829 / 0.825 / 0.945 | **1.186 / 1.183 / 1.354 (fails)** |
| 0.09 | | 0.682 / 0.644 / 0.690 | 0.980 / 0.927 / 0.993 |
| 0.08 | | 0.557 / 0.502 / 0.510 | 0.803 / 0.725 / 0.737 |
| 0.07 | | 0.450 / 0.388 / 0.378 | 0.650 / 0.563 / 0.549 |
| 0.06 | | 0.358 / 0.297 / 0.279 | 0.517 / 0.431 / 0.406 |
| 0.05 | | 0.278 / 0.223 / 0.203 | 0.402 (k = 10) |

The contraction at fixed R₊ grows with M through the label terms, as §3.2 predicts. At M = 3 the M = 1 domain
(R₊ = 0.01) still closes, so G₀ and the 6b₁ annulus are those of M = 1. At M = 5 it does not, and R₊ = 0.0064
(contraction 0.72 at k = 20) gives g₀ = 1/(R₊ sin 0.4) = 401.2.

**Over the box X₆ = x̂ ± 10⁻⁶** (`lemma6_sector_M.py --box=1e-6`, exit status 0):

| M | R₊ (r₊²) | k | parameters | lemma S contraction | old sector lemma (diag) | Z | flat estimate Λ plain / diag | all |
|---|---|---|---|---|---|---|---|---|
| 3 | 0.1² | 20 | x̂ ± 1e-6 | 0.8252 | 0.5408 | 0.004489 | 0.2923 / 0.2199 | yes |
| 5 | 0.08² | 20 | x̂ ± 1e-6 | 0.7247 | 0.6493 | 0.004489 | 0.4233 / 0.2979 | yes |

**Lemma S(3), lemma 6b₀(3) and the flat estimate at M = 3: certified** over X₆ = x̂ ± 10⁻⁶ (trusted code:
`lemma6_sector_M.py`, `fixed_point_M.py` and `fixed_point.py`'s majorant helpers, python-flint).

- Lemma S: R₊ = 0.01, k = 20, a_eff = 0.2536, contraction ≤ 0.8252.
- Old sector lemma: diag splitting, contraction ≤ 0.5408.
- Lemma 6b₀(3): y⁺ is analytic for 0 < |ξ| ≤ 0.0624, 0 ≤ arg ξ ≤ 3π/8, and on the real segment up to 0.0841, for every
  x ∈ X₆. These radii are exactly those of M = 1.
- Flat estimate at Z = 0.004489 (ξ_a = 0.067): Λ ≤ 0.2923 (plain) or 0.2199 (diag).

**Lemma S(5), lemma 6b₀(5) and the flat estimate at M = 5: certified** over X₆ = x̂ ± 10⁻⁶ at the M = 5 zero
x̂ = (2.718579189, 0.999228404, 0.149203574672, 0.014719052460, 0.008405346879, 0.001470345176). (A, ω/ω∞) are from
`hier.py --schur`; the B, C are re-solved at that (A, ω), with Schur-corrected |r₁| = 6.3×10⁻⁶ and r₃, r₅ at about
3×10⁻¹⁰ (`lemmas_M_runs/zero_M5.json`).

- Lemma S: R₊ = 0.0064 (r₊ = 0.08), k = 20, a_eff = 0.2896, contraction ≤ 0.7247.
- Old sector lemma: diag splitting, contraction ≤ 0.6493.
- Lemma 6b₀(5): y⁺ is analytic for 0 < |ξ| ≤ 0.0499 (|u| ≥ 401.2), 0 ≤ arg ξ ≤ 3π/8, and on the real segment up to
  0.0673, for every x ∈ X₆.
- Flat estimate at the same Z = 0.004489, which still lies in D's real reach R₊/√2 = 0.004525: Λ ≤ 0.4233 (plain) or
  0.2979 (diag).

So at M = 5, G₀ shrinks to |u| > 402, and the 6b₁ annulus grows to 25 ≤ |u| ≤ 402 (§9).

## 4. Lemma 6a

Code: `solver/proof/lemma6_jump_M.py` (the backward flow and ρ₆ₐ) and `lemma6_sector_M.py` (the flat estimate).

### 4.1 Statement

Let ψ = y⁺_ℝ − y⁻_ℝ = y⁺_ℝ − σ(y⁺_ℝ) on the real axis, where σ maps the label μ to −μ and conjugates. Its D-components
at ξ₁ are $`\psi_{D,\mu}(\xi_1) = J_{D,\mu}(\xi_1) = D^+_\mu(\xi_1) - \overline{D^+_{-\mu}(\xi_1)}`$, with
$`J_{D,-\mu} = -\overline{J_{D,\mu}}`$.

**Lemma 6a(M).** Let B₁ be a box that contains y⁺_ℝ(ξ₁; x) for every x ∈ X. Suppose:

- lemma S(M) holds for X with R₊ ≥ √2 Z, Z = ξ_a²;
- the flat estimate holds on (0, Z] with ‖T₀‖ ≤ Λ < 1;
- the backward computation of `lemma6_jump_M.py` from the hull of B₁ and σ(B₁) succeeds with ρ₆ₐ < 1.

Then for every x ∈ X:

```math
J_{D,\mu}(\xi_1; x) = 0 \ \text{ for } \mu = 1, 3, \ldots, M \iff \psi \equiv 0 \iff y^+ \equiv y^- \iff S_1(x) = S_3(x) = \cdots = S_M(x) = 0 .
```

### 4.2 Proof

The three steps of §11.5 carry over, with these dimensions.

- **Hypothesis.** If all J_{D,μ}(ξ₁) with μ > 0 vanish, then ψ_{D,μ}(ξ₁) = 0 for every μ of either sign, so
  ψ(ξ₁) = (0, u) with u = (ψ_S, ψ_C) ∈ ℂ^{2M+3}.
- **Step 1, the flat estimate.** On 0 < z ≤ Z, both y⁺ and y⁻ = σ(y⁺(z̄)) lie in the ball of lemma S, since σ maps the
  ball to itself under symmetric weights. So ψ = b + T₀ψ exactly, with b = (0, $`e^{1/(2Z)-1/(2z)}`$β, 0) and β = v_ψ(Z)
  ∈ ℂ^{M+1}, β_{−μ} = −conj(β_μ). The slow components are integrated from 0 and the fast ones backward from Z, with the
  kernel bound $`z^{n+2}/(\tfrac12 - (n+2)Z)`$ and a_r = ½ − kZ.
  - The real axis is a ray, so the diag splitting is also valid here. With the label terms in the principal part the
    kernel gains (t/z)^{(3+2s)/2}, whose modulus is (t/z)^{3/2} ≤ (Z/z)^{3/2} for z ≤ t ≤ Z; that shifts n by 3/2 and
    gives a_r = ½ − (k + 3/2)Z.
  - In either splitting, b's weighted norm is attained at z = Z, because $`z^{-k}e^{-1/(2z)}`$ (times (Z/z)^{3/2} in the
    diag case) increases on (0, Z] when (k + 3/2)Z < ½.
  - Hence ε := ‖ψ − b‖ ≤ Λ/(1 − Λ)‖b‖, |p_μ(Z)| ≤ Z^k ε/w_{p,μ} and |c_ν(Z)| ≤ Z^{k+1} ε/w_{c,ν}.
- **Step 2, the backward flow.** The C¹ Lohner integrator of `validated_int.py`, unchanged, on the ξ-form
  `lemma6_usys_M.SysXiM`, runs from the hull of B₁ and σ(B₁) along the real axis to ξ_a, with the identity as
  variational columns. The hull is convex and contains y⁺(ξ₁) and y⁻(ξ₁), hence the segment between them. So
  ψ(ξ_a) ∈ B ψ(ξ₁), where B encloses DΦ over the hull. With T_a the exact change of variables
  (p = D/ξ, v = S/ξ² − (1 + iμω)D/ξ, c = C) and G = T_a B restricted to the u-columns,
  (p, c)(Z) = G_slow u and β = G_fast u. G_slow is (2M + 3) × (2M + 3): M + 1 rows of p and M + 2 of c, against
  M + 1 S-columns and M + 2 C-columns.
- **Step 3, the contraction in β.** If G_slow is invertible over the whole ball, then β = H(p, c)(Z) with
  H = G_fast G_slow⁻¹, and

```math
\max_\mu w_{v,\mu}\lvert\beta_\mu\rvert \le \rho_{6a}\max_\mu w_{v,\mu}\lvert\beta_\mu\rvert, \qquad
\rho_{6a} = \frac{\Lambda}{1-\Lambda}\,\max_\mu w_{v,\mu}\sum_j \lvert H_{\mu j}\rvert\,\frac{Z^{k_j-k}}{w_j} .
```

ρ₆ₐ < 1 gives β = 0, then b = 0 and ψ = T₀ψ with ‖T₀‖ < 1, so ψ = 0 on (0, Z], and y⁺ ≡ y⁻ by uniqueness for the
ODE. The multipliers are the coefficients of the first exponential term of y⁺ − y⁻ (Lemma B4(M)), so they all vanish.
The converse implications are trivial. ∎

**What changes with M, and what does not.**

- *Unchanged: the fast decay along the real axis.* It is label-independent, since $`\lvert\xi^{-2s_\mu}\rvert = 1`$ for
  real ξ. The weights $`e^{\pi\mu\omega/2}`$ only relate β_μ to S_μ by non-zero factors, which does not matter for
  β = 0 ⇔ S = 0.
- *Changed: the number of conditions.* One complex D-component is the whole jump at M = 1. At M harmonics, all
  (M + 1)/2 complex components J_{D,μ}, μ > 0, are needed, because u must have as many entries as there are slow rows.
  If J_{D,3} is dropped, ψ_{D,±3}(ξ₁) joins u and G_slow becomes 9 × 11, which no argument of this kind can invert
  (mutation `partial`, §8). The de-mixed residuals r_ν of `hier.py` are an invertible linear image of the J_{D,μ}
  ([PROOF.md](../PROOF.md) §4; `inner_np.residual_demixed`), so a certificate of their simultaneous zero is a
  certificate of the hypothesis.

**Simplicity.** The argument of §11.7 carries over verbatim. At a zero x*, φ = ∂ψ/∂x_j solves the variational
equation along y⁺ = y⁻ and obeys the same flat estimate and backward inclusion, since its coefficients lie in the same
hulls. So the real-linear map 𝓛 : ∂Σ ↦ ∂J_D(ξ₁), now from ℝ^{M+1} to ℝ^{M+1}, is injective. Since
DJ_D(x*) = 𝓛 ∘ DΣ(x*), the Jacobian DΣ(x*) is non-singular whenever DJ_D(x*) is.

### 4.3 Results

**Flat estimate: certified** at M = 3 and 5 over X₆ = x̂ ± 10⁻⁶, at Z = 0.004489 (ξ_a = 0.067, the M = 1 value), §3.3:

| M | Λ plain | Λ diag | largest entries |
|---|---|---|---|
| 1 (§11.5) | 0.1513 | | |
| 3 | 0.2923 | 0.2199 | v±3 ← p±3: 1.16 |
| 5 | 0.4233 | 0.2979 | v±5 ← p±5: 2.6 |

The largest entries come from the (1 + s)(2 + s)P/(2z) coupling, which grows like (μω)².

**Backward flow and ρ₆ₐ: a pilot, not a proof** (`lemma6_jump_M.py --pilot`, `lemmas_M_runs/lemma6a_pilot_M3_xi0.2.*`).
B₁ is the double-precision y⁺(0.2) at x̂ (from `lateral_np.py`'s path, the ray 3π/8 from 0.085 to 0.2 and the arc at
0.2), inflated by 10⁻¹⁰. That is the role the point certificate's box plays; here it is not rigorous. Lemma S and the
flat estimate are over X₆, and ω is a ball over X₆.

| | M = 1 (§11.5, rigorous) | M = 3 (pilot) |
|---|---|---|
| ξ₁ → ξ_a | 0.3 → 0.067 | 0.2 → 0.067 |
| start hull radius | (box run, h = 10⁻⁹) | 3.9×10⁻¹⁰ |
| Lohner steps, time | 291, 130 s | 389, 3014 s |
| max radius of B | 1.8×10⁻⁶ | 1.1×10⁻³ |
| entries of G_slow⁻¹, of H = G_fast G_slow⁻¹ | ≤ 1.26, ≤ 7.7 | ≤ 1.13, ≤ 10.7 |
| ρ₆ₐ | 4.22×10⁻³ | **3.04×10⁻²** |

From the certificate's matching radius ξ₁ = 0.3, at its centre p̂, the same pilot gives **ρ₆ₐ = 0.052** (§7). From x̂
it fails, because x̂ is too far from the zero for a hull at 0.3 (§7).

So the 6a computation works at M = 3 with a margin of about 20–30, as numerical evidence. B is wider than at M = 1. One
plausible contributor, not separated out here: ω ranges over X₆ (10⁻⁶) in this pilot, against a 10⁻⁹ box at M = 1.
The run itself is the cost to plan for: about 50 minutes at load 15–19.

## 5. Lemma 6b₁

Code: `solver/proof/lemma6b1_cover_M.py` (design, runs, check), `lemma6b1_disc_M.py` (lemma D) and
`lemma6b1_estimate_M.py` (planning, double precision).

### 5.1 Statement

**Lemma 6b₁(M).** Suppose every run of the covering design of `lemma6b1_cover_M.py` for M and X succeeds (all discs
certified) and the check passes (C1–C8 of §12.3). Then for every x ∈ X, y⁺ extends analytically to an open set
G = G₀ ∪ ⋃ T_jk that contains the annular sector A₁ = {25 ≤ |u| ≤ 260, −3π/4 ≤ arg u ≤ 0}, i.e.
0.062 ≤ |ξ| ≤ 0.2, 0 ≤ arg ξ ≤ 3π/8. Here G₀ = {|u| > g₀, −3π/4 − 0.4 < arg u < 0}, where y⁺ is analytic by lemma 6b₀(M).

### 5.2 Proof

**Verbatim from §12.** None of the following depends on M:

- **Lemma D.** Its proof uses only that the system is polynomial in y and in z = 1/u. `lemma6b1_disc_M.certify` is the
  same computation with nd components; at M = 1 it reproduces `lemma6b1_disc.certify` exactly (§5.3).
- **Lemma G, the gluing.** It needs only the identity theorem and uniqueness for a regular ODE.
- **The geometric conditions C2–C8 and their check.** `lemma6b1_cover_M.py check` calls `lemma6b1_cover.check`
  unchanged, with its design, run directory and G₀ pointed at the M versions.
- **The agreement points** on the arcs, on the ray, and at s₀ ∈ G₀.

**What changes.**

- The system: `SysUM` with nd = 2(M + 1) + (M + 2) components.
- The start box at s₀ = ρ₀e^{−3iπ/4}: the old sector lemma for M over X, with the diag splitting.
- The radius g₀ of G₀, which comes from lemma S(M): g₀ = 1/(R₊ sin 0.4). This gives 257 at M = 1 and 3 (R₊ = 0.01)
  and 402 at M = 5 (R₊ = 0.0064).
- The disc radii and orders at the inner rows. At M = 3, y⁺ is larger near |ξ| = 0.2 (max |V| 8.6 against 4.5 at
  M = 1 on |u| = 25), and its nearest singularity is closer: the root-test estimate at 25e^{−3iπ/4} is 12.7, against
  15.0 at M = 1, where the branch point located in §12.1 is 15.8 away. So the rows with ρ < 32 use r = 6 and n = 50.
- How the values reach the discs. The M = 1 method (one Lohner set over the box, arcs restarted from the ray's hull)
  fails on the innermost row at M = 3. The C¹ mean-value mode of §5.3 replaces it, and needs no change to lemma D,
  lemma G or the check.
- The cost per disc and per Lohner step, from the dimension.

With these, the proof is that of §12.4: lemma D for each disc (C1), then lemma G with C2–C7, then C8. ∎

### 5.3 Results

**Planning (double precision, not rigorous; `lemma6b1_estimate_M.py`).** On the arcs |u| = ρ, from the start ray to
the real axis:

| \|u\| | \|ξ\| | M = 1: max \|P\|, \|V\|, \|C₀ − 1\| | M = 3: max \|P\|, \|V\|, \|C₀ − 1\| | root-test singularity distance at ρe^{−3iπ/4}, M = 1 / M = 3 |
|---|---|---|---|---|
| 262 | 0.0618 | 2.63, 0.23, 0.013 | 2.86, 0.33, 0.015 | |
| 100 | 0.1000 | 2.79, 0.65, 0.034 | 3.11, 1.01, 0.040 | 63 / 52 |
| 60 | 0.1291 | 2.98, 1.22, 0.058 | 3.41, 1.97, 0.069 | 45 / 36 |
| 40 | 0.1581 | 3.24, 2.12, 0.090 | 3.87, 3.61, 0.110 | 30 / 24 |
| 25 | 0.2000 | 3.85, 4.53, 0.154 | 5.01, 8.62, 0.198 | 15.0 / 12.7 |

The M = 1 column reproduces §12.1's table. At M = 1 the root test at 25e^{−3iπ/4} (15.0) is close to the distance to
the branch point located there (15.8).

**Lemma D at M = 1 reproduces lemma6b1_disc** (`python3 lemma6b1_disc_M.py`, exit status 0). On the M = 1 data, at
|u| = 102.16 (r = 24, n = 70) and 24.94 (r = 10, n = 40), Λ is 0.184226 and 0.438972 in both codes, and Λ and F₀
agree exactly.

**Disc radii (probe, not rigorous).** Discs certified around double-precision values of y⁺ with inflated boxes
(`probe_M3*.log`):

- The M = 1 radii work at M = 3 down to row 10.
- The innermost row's start-ray disc needs r = 6 (r = 8 gives Λ = 1.34 with 10⁻⁷ boxes).
- How much box radius a disc tolerates depends on where it sits. With r = 6, the start-ray disc of row 11 passes with
  2×10⁻⁵ boxes (Λ = 0.28) and fails with 10⁻⁴; the real-axis disc passes with 10⁻³.

The M = 3 design (`design --M=3`) therefore has the 12 rows of M = 1, with r = 6 below |u| = 32: 183 discs.

**The ray run over x̂ ± 10⁻⁸** (rigorous; `lemma6b1_runs_M3/ray_h1e-8.json`). The start box from the old sector lemma
has radius 5.8×10⁻⁷. Across the 11 legs, 903 steps:

| \|u\| | 235 | 209 | 182 | 155 | 129 | 102 | 80 | 62 | 47 | 36 | 27 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| radius (×10⁻⁷) | 1.06 | 1.16 | 1.27 | 1.40 | 1.93 | 3.09 | 5.51 | 10.7 | 25.2 | 69.8 | 254 |

At M = 1 the same run grew from 2.9×10⁻⁸ to 7.9×10⁻⁸.

**Row 11 over x̂ ± 10⁻⁸ fails** (`cover_M3_row11_h1e-8_hullrestart.log`). Discs 0–3 are certified (Λ ≤ 0.34). Then
the Lohner box grows from 2.5×10⁻⁵ at the arc's start to 2.7×10⁻³ (t = 0.18) and 5.6×10⁻² (t = 0.36), and disc 4 fails
(Λ = 2.2).

The cause, from `lemma6b1_estimate_M.py --arc=26.98` (double precision; `arc_diagnostic_M3_row11.log`):

- **The box at the arc start is real parameter spread.** y⁺ near 27e^{−3iπ/4} moves by about 630·h over the box
  (6.3×10⁻⁶ at h = 10⁻⁸). The same diagnostic at M = 1, on row 11 at 24.94 (`arc_diagnostic_M1_row11.log`), gives
  22h. The higher harmonics make y⁺ about 30 times more sensitive to the parameters near |ξ| = 0.19.
- **The linear flow would grow a restarted box only moderately.** Its largest absolute row sum stays at or below 18
  up to t = 0.73 and reaches 691 only at the real axis (at M = 1 it stays at or below 18 to the axis). The Lohner run
  exceeds that hull by 10–170× from t = 0.18 on. That excess is the quadratic feedback diagnosed in §11.4: a wide set
  widens every step's Jacobian, which widens the set. At M = 1 the same excess was about 13×: row 11's boxes reached
  1.8×10⁻⁵ against a hull of 1.4×10⁻⁶.

**A smaller box does not cure it** (h = 10⁻⁹; `cover_M3_ray.log`, `cover_M3_row11_h1e-9_hullrestart.log`). The ray
run scales exactly: 5.8×10⁻⁸ at the start, 2.5×10⁻⁶ at |u| = 27. On the arc the boxes are 10× smaller than at 10⁻⁸ up
to disc 4 (3.8×10⁻³). They still jump by 10–15× at the same discs as at h = 10⁻⁸, about 10⁴ times the true spread, and
disc 6 fails (box 1.4×10⁻², Λ = 2.5). So the excess is mostly the Lohner set's own growth on this arc after a restart
from a full box; it is not quadratic feedback alone.

**The C¹ (mean-value) mode** (`--c1`). The set over X is y(t; x̂) + V(t; X)(x − x̂):

- the single trajectory from the box centre is a Lohner set;
- the M + 1 parameter-derivative columns are Lohner sets of their own (`validated_int.Lohner`'s `var` and `dx`, the
  machinery of the box runs of §9);
- the step Jacobian is enclosed over that set, which by the mean value theorem contains every trajectory from X;
- each arc continues from the ray's exact Lohner state (c, Q, q and the columns, saved at every row radius in
  `ray.json`), not from its hull.

The start data at s₀ = 262e^{−3iπ/4} come from the old sector lemma evaluated at z₀ = 1/s₀:

- the value at x̂, with |η_j(z₀)| ≤ (ρ/w_j)|z₀|^{k_j};
- the derivatives in all M + 1 parameters over X, by jets plus Cauchy's estimate over X + 10⁻³(1 + i).

At M = 3, with h = 10⁻⁹:

- the centre is enclosed to 4×10⁻⁴⁴;
- the derivative columns are enclosed to 8×10⁻⁸ (|∂y/∂x| ≤ 1.3);
- the set over X has radius 1.6×10⁻⁹, against 5.8×10⁻⁸ for the box start.

A save and restore of the Lohner state contains the original set.

**The C¹ runs on the point certificate's box** (`lemma6b1_runs_M3_cert/`). The centre is the certificate's p̂ (`p_out`
of its `data/A_M_refine_M3_a.json`); the box is p̂ ± 1.1×10⁻⁹ in all four parameters. It contains the axis-aligned hull
of the certificate's tilted box with h = 10⁻⁹, and the hull of the h = 10⁻¹⁰ box it was running at the time of writing
around its later centre (8×10⁻¹⁸ away).

The ray run (resumed once after a container restart, from the saved Lohner state at row 6) has these set radii over X:

| \|u\| | 262 | 235 | 209 | 182 | 155 | 129 | 102 | 80 | 62 | 47 | 36 | 27 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C¹ set radius (×10⁻⁹) | 1.75 | 1.89 | 2.24 | 2.75 | 3.52 | 4.83 | 7.42 | 12.7 | 23.9 | 54.1 | 144 | 503 |
| hull run at h = 10⁻⁹ (×10⁻⁹) | 58 | 10.6 | 11.6 | 12.7 | 14.0 | 19.3 | 30.9 | 55.1 | 107 | 252 | 698 | 2530 |

At |u| = 27 the C¹ set (5.0×10⁻⁷) is about the true spread over this box (about 7×10⁻⁷ by the double-precision
diagnostic).

**Row 11, the hardest row: certified** (`row_11.json`, `cover_M3_cert_row11.log`). All 12 discs pass with
r = 6, n = 50: Λ ≤ 0.179, box radius ≤ 3.4×10⁻⁵ (at the real axis; at most 1.1×10⁻⁶ before the last two discs),
sup|y − y(c)| ≤ 7.5, 834 Lohner steps, 1566 s. The hull runs failed at disc 4 (h = 10⁻⁸) and disc 6 (h = 10⁻⁹) of
this row.

**All rows, and the check: lemma 6b₁(3) certified on the certificate's box** (`lemma6b1_runs_M3_cert/row_JJ.json`;
`python3 lemma6b1_cover_M.py check ...`, exit status 0, `lemmas_M_runs/lemma6b1_check_M3_cert.json`). Trusted code:
`lemma6b1_cover_M.py`, `lemma6b1_disc_M.py`, `lemma6_usys_M.py`, `fixed_point_M.py`, and from M = 1, unchanged,
`validated_int.Lohner`, `lemma6_common.py` and `lemma6b1_cover.check`.

| row | ρ_j | r, n | discs | Lohner steps | max Λ | max box radius | max sup\|y − y(c)\| | time |
|---|---|---|---|---|---|---|---|---|
| 0 | 262 | 24, 70 | 25 | 1299 | 0.185 | 1.8×10⁻⁹ | 0.151 | 2144 s |
| 1 | 235.36 | 24, 70 | 23 | 1198 | 0.188 | 1.9×10⁻⁹ | 0.191 | 1919 s |
| 2 | 208.72 | 24, 70 | 20 | 1138 | 0.191 | 2.3×10⁻⁹ | 0.248 | 1486 s |
| 3 | 182.08 | 24, 70 | 18 | 1088 | 0.196 | 2.8×10⁻⁹ | 0.336 | 1422 s |
| 4 | 155.44 | 24, 70 | 16 | 1034 | 0.202 | 3.6×10⁻⁹ | 0.48 | 1055 s |
| 5 | 128.80 | 24, 70 | 13 | 946 | 0.212 | 4.9×10⁻⁹ | 0.743 | 975 s |
| 6 | 102.16 | 24, 70 | 11 | 891 | 0.231 | 7.5×10⁻⁹ | 6.16 | 491 s |
| 7 | 79.60 | 16, 60 | 13 | 835 | 0.191 | 1.3×10⁻⁸ | 1.42 | 495 s |
| 8 | 61.84 | 16, 60 | 10 | 803 | 0.221 | 2.4×10⁻⁸ | 2.72 | 380 s |
| 9 | 47.14 | 10, 50 | 12 | 811 | 0.182 | 5.5×10⁻⁸ | 2.97 | 424 s |
| 10 | 36.04 | 10, 50 | 10 | 799 | 0.245 | 1.5×10⁻⁷ | 6.41 | 208 s |
| 11 | 26.98 | 6, 50 | 12 | 834 | 0.179 | 3.4×10⁻⁵ | 7.54 | 1566 s |
| all | | | 183 | 11 676 (+ 903 on the ray) | 0.245 | 3.4×10⁻⁵ | 7.54 | 12 565 s of process time |

The check passes C1–C8: 183 tiles glued, with meeting sets of 579 polar rectangles joined by 461 proven links. The
row times are wall-clock at loads from 2 to 22, and the rows ran two at a time.

**Lemma 6b₁(3) (certified).** For every x in the box p̂ ± 1.1×10⁻⁹ (p̂ the certificate's centre above), y⁺ extends
analytically to an open set containing the annular sector 25 ≤ |u| ≤ 260, −3π/4 ≤ arg u ≤ 0, i.e. 0.062 ≤ |ξ| ≤ 0.2,
0 ≤ arg ξ ≤ 3π/8. Together with lemma 6b₀(3), which holds over x̂ ± 10⁻⁶ ⊃ p̂ ± 1.1×10⁻⁹, y⁺ is analytic on
Ω = {0 < |ξ| ≤ 0.2, 0 ≤ arg ξ ≤ 3π/8} for every such x.

**Controls for the C¹ mode** (`c1check`, double precision, not rigorous). At rows 1, 6, 9 and 11, y⁺ at p̂ and at the
eight points p̂ ± h·e_i, computed independently of the Lohner code (series, then DOP853 along the ray), lies in the
saved sets inflated by 10⁻¹¹. With the derivative columns dropped (`--mutate=nov`), all eight off-centre points fall
outside while the centre stays inside, so the control discriminates. The centre trajectory alone is enclosed to about
4×10⁻¹² at |u| = 27 (`c1check_M3_cert*.log`).

## 6. Theorem A(M), conditional on the point certificate

Let x̂ be the double-precision zero (§2.4), X_c the box of the point certificate, and ξ₁ and ρ_arc its matching radius
and arc radius. Assume the certificate passes along a path of the kind used in §11.2: the start ray at an angle
θ ∈ [3π/8 − 0.2, 3π/8] from a radius r₀ ≤ 0.1, that ray out to ρ_arc ≤ 0.2, the arc |ξ| = ρ_arc to the real axis, and
then (if ξ₁ > ρ_arc) the real axis to ξ₁.

**Theorem A(M) (conditional).** Assume Braaksma's theorem (Lemma B4(M)). Suppose the point certificate proves that
x ↦ (J^path_{D,μ}(ξ₁; x))_{μ = 1, 3, …, M} has exactly one zero in X_c and that it is simple. Suppose further:

- (H1) X_c ⊂ X₆ = x̂ ± 10⁻⁶, the box over which lemma S(M), the old sector lemma and the flat estimate are proved here
  (§3.3);
- (H2) X_c ⊂ X_b, the box of a complete 6b₁ covering: at M = 3, X_b = p̂ ± 1.1×10⁻⁹ (§5.3), certified;
- (H3) lemma 6a(M) holds with B₁ the certificate's state box at ξ₁ over X_c.

Then the M-harmonic selection condition y⁺ ≡ y⁻ (all of S_1, …, S_M vanish) holds at exactly one point x* of X_c, and
the real (M + 1) × (M + 1) Jacobian of (Re S_μ, Im S_μ)_μ is non-singular at x*.

*Proof.* The steps are those of §12.6 in order.

1. Lemma 6b₀(M) (from H1) and lemma 6b₁(M) (from H2) make y⁺ analytic on Ω = {0 < |ξ| ≤ 0.2, 0 ≤ arg ξ ≤ 3π/8}. The
   certificate's path lies in Ω up to the real leg, and its start point lies in the domain of the old sector lemma, where
   the start value is y⁺. So the path continuation is the continuation along the real axis, and J^path = J.
2. The certificate gives one simple zero of J in X_c.
3. Lemma 6a(M) (H3, with lemma S and the flat estimate from H1) gives J(ξ₁; x) = 0 ⇔ y⁺ ≡ y⁻.
4. Simplicity transfers to S by §4.2. ∎

At M = 1 the same assembly is Theorem A(M = 1) of §12.6.

## 7. What waits for the point certificate

**Result at M = 3 (added after the certificate).** The certificate's M = 3 run at h = 10⁻¹⁰ ([A_M.md](A_M.md) §6.2)
passes its Krawczyk test. Its state box at ξ₁ = 0.3 (`solver/proof/data/A_M_6a_M3_1e-10.json`, with x set to its centre
p in `A_M_6a_M3_1e-10_in.json`) gives lemma 6a: `lemma6_jump_M.py --M=3 --box=data/A_M_6a_M3_1e-10_in.json` runs the
backward flow 0.3 → 0.067 in 449 steps (B's largest radius 2.0×10⁻³, |G_slow⁻¹| ≤ 1.62, |H| ≤ 11.6) and proves
**ρ₆ₐ = 0.041 < 1** (`lemmas_M_runs/lemma6a_M3_cert.log`, `.json`). Its parallelepiped lies in p̂ ± 1.1×10⁻⁹: the
centre is within 2.7×10⁻¹³ of p̂ and the hull's half-widths are 10⁻¹⁰, 10⁻¹⁰, 3.9×10⁻¹¹ and 2.5×10⁻¹² in
(A, ω/ω∞, B₃, C₃). So (H1)–(H3) hold, and **Theorem A(3) is certified**, assuming Braaksma's theorem: the three-harmonic
selection condition (S₁ = S₃ = 0, for the truncation of `inner_np.py`: b, c on |ν| ≤ 4, each binary product projected)
has exactly one zero in the certificate's box, and it is simple. The code (this note's and A_M.md's) has not been
reviewed independently.

The rest of this section was written before the certificate finished.

The point certificate is built on the branch `wip/layer-a-cert-m` (`solver/proof/certify_M.py`, notes
`proof/A_M.md`). At the time of writing, its notes and data, read only, fix the following.

- *Path:* the start ray 5π/16 from r₀ = 0.0741, out to 0.2, the arc |ξ| = 0.2 to the real axis, then the real axis to
  ξ₁ = 0.3. This is the lower path of §11.2, with `inner_np.py`'s start ray.
- *Start data:* from `fixed_point_M.start_data_M`, with the configuration §2.4 recommends.
- *Box:* a tilted box p̂ + TX around a refined centre p̂, with h = 10⁻⁹ (a later run uses 10⁻¹⁰) in (A, ω/ω∞) and
  inner widths of order 10⁻¹⁶ in (B₃, C₃).
- *Status:* its M = 3 box runs were still in progress.

Against that, this note's status for M = 3:

| hypothesis of §6 | status here | what it needs from the certificate |
|---|---|---|
| (H1) X_c ⊂ x̂ ± 10⁻⁶ (lemma S, 6b₀, flat estimate) | **certified** over x̂ ± 10⁻⁶; p̂ is within 4×10⁻¹⁰ of x̂ | only the inclusion, which holds for any h ≤ 10⁻⁷ around p̂ |
| (H2) X_c ⊂ p̂ ± 1.1×10⁻⁹ (lemma 6b₁) | **certified** on that box (§5.3) | the inclusion: its final centre within about 10⁻¹⁰ of p̂ and h ≤ 10⁻⁹ (true of both of its runs) |
| the path lies in Ω | yes: arc 0.2, start ray 5π/16 within 0.2 of 3π/8, start point at 0.0741 < 0.1 | none |
| (H3) lemma 6a from its B₁ at ξ₁ = 0.3 | **certified**: ρ₆ₐ = 0.041 (below) | its hull of y⁺(0.3; X_c), as JSON (keys M, h, x, xi1, yX_xi1); then `lemma6_jump_M.py --box=` (about 1 h) |
| the zero and its simplicity | — | its Krawczyk test on the (M + 1) real D-jump conditions at ξ₁ |

**The 6a pilots from ξ₁ = 0.3** (not rigorous). From x̂, the backward flow ran (450 steps), but G_slow was not
proved invertible: the start hull was 8.0×10⁻⁶ and B's entries reached radius 0.49. The hull is mostly the pilot's
own ψ(0.3) at x̂, which is about 4×10⁻¹⁰ from the zero in A, magnified by ∂J/∂x at 0.3. From the certificate's p̂,
which is closer to the zero, the start hull is 3.9×10⁻⁹ and the pilot passes: 449 steps, 912 s at low load, max
radius of B 5.1×10⁻³, |G_slow⁻¹| ≤ 1.65, |H| ≤ 13.1, and **ρ₆ₐ = 0.052** (`lemma6a_pilot_M3_xi0.3_phat.*`). The
certificate's B₁ is a box over X_c around the zero, so its hull is its own width, about |∂y/∂x|·h. That is comparable
to this pilot's hull, so the run with the real B₁ is expected to pass with a similar margin. It remains to be run.

If the certificate's final box or path differs from the above, these must be rerun:

- a larger or shifted box: `lemma6_sector_M.py --box=` for (H1), and the covering with `--x=` and `--h=` for (H2), about
  3.5 hours of process time at M = 3;
- an arc above 0.2: a covering with u_in = 1/ρ_arc², with the inner rows probed first, since they are where the cost and
  the singularity distance bite;
- a certificate of S₁ = 0 alone, at fixed higher harmonics: that would not satisfy 6a, which needs all (M + 1)/2
  complex J_{D,μ}(ξ₁), or an invertible linear image of them such as `hier.py`'s de-mixed residuals.

At M = 5 everything waits for its certificate except lemma S, 6b₀, the flat estimate and step (i), which are certified
here. 6b₁(5) is a local-machine task (§9).

## 8. Mutation table

Every check added here has a runnable mutation or control. Each verdict was written down before the run (the expected
column), and `mutations_M.py` compares it with the outcome; exit status 0 only if all match. Results are in
`solver/proof/lemmas_M_runs/mutations_M.json`.

| name | what is broken | expected | exit | as expected |
|---|---|---|---|---|
| `fp_series_M3` | step (i): one coefficient of the harmonic-3 D series off by a relative 1e-6 | caught | 0 | yes |
| `fp_series_M5` | step (i), M = 5 on inner_np's ray: one coefficient of the harmonic-5 D series off by 1e-6 | caught | 0 | yes |
| `fp_cdrop_M3` | step (i): the series built without the c harmonics +-4 (an inconsistent harmonic set) | caught | 0 | yes |
| `fp_diagpath_M3` | guard, not a mutation: the diag splitting with a non-ray kernel bound (invalid there) must be refused | caught | 0 | yes |
| `fp_control_r0` | control: start radius 0.3, the contraction must fail | fails | 2 | yes |
| `fp_m1` | control: M = 1 reproduces fixed_point.py (constants, start values, start data) | passes | 0 | yes |
| `usys_label` | system: s_mu = i abs(mu) w in the u-form (the label sign lost); the comparisons with the xi-form and with fixed_point's encoding must fire | caught | 0 | yes |
| `usys_clean` | control: the system self-tests at M = 3, 5 | passes | 0 | yes |
| `S_series_M3` | lemma S: one coefficient of the harmonic-3 series off by 1e-6 | caught | 0 | yes |
| `S_control_rp` | control: lemma S at R+ = 0.13^2, beyond the contraction (0.83 at 0.1, label terms grow like R+) | fails | 2 | yes |
| `disc_m1` | control: lemma D at M = 1 reproduces lemma6b1_disc.certify | passes | 0 | yes |
| `jump_partial_M3` | 6a: only J_D,1 = 0 assumed (J_D,3 dropped): G_slow is not square, 6a must not be established | caught | 0 | yes |
| `jump_swap_M3` | 6a: y- = conj(y+) without the reflection mu -> -mu: must fail | caught | 0 | yes |
| `jump_coords_M3` | 6a: (1 - i mu w) in the change of variables at xi_a; at M = 1 not caught (second order); reported only | report | 0 | yes |
| `cover_check_M3` | control: the 6b1 gluing and covering check C1-C8 on the M = 3 runs (C^1 mode, certificate's box): PASS | passes | 0 | yes |
| `cover_drop_M3` | 6b1 gluing check with tile (5, 6) left out: must print FAIL | fails | 3 | yes |
| `cover_sign_M3` | 6b1: every disc of the innermost row with the fast term of the wrong sign: none may be certified | caught | 0 | yes |
| `c1_truth_M3` | control (double precision): y+ at x^ and x^ +- h e_i lies in the saved C^1 sets at rows 1, 6, 9, 11 | passes | 0 | yes |
| `c1_nov_M3` | C^1 sets with the derivative columns dropped: the off-centre points must fall outside | caught | 0 | yes |

Every verdict matches (`python3 mutations_M.py --all`; the entries ran in four batches, recorded in
`lemmas_M_runs/mutations_cheap.log`, `mutations_jump.log` and `mutations_cover.log`). Notes:

- `fp_series_M3` was first seen *not* to fire. Without the structural vanishing check, the harness lowered the weight
  exponent to fit the mutated polynomial and proved a weaker, true lemma about it. The check was added and the entry
  rerun (§2.2).
- `jump_coords_M3` is reported, not caught: ρ₆ₐ = 0.89 < 1 with the wrong mode normalization (0.12 at M = 1). As at
  M = 1 (§11.6), the error enters at second order through the flat estimate. What guards the normalization is
  `lemma6_usys_M.py`'s test (c).
- The 6a mutations ran on the pilot's B₁, so they test the harness, not a proof.

## 9. Cost, and a proposed task for the local machine

**Measured on the cloud machine** (2 cores at load 15–19; each process got about a fifth of a core; wall-clock):

| piece | M = 3 | M = 5 |
|---|---|---|
| step (i), one run at a point / over a box (table rows) | 4–8 s / 6–17 s | 7–32 s / 15–56 s |
| start data with derivatives over a box | 32–59 s | 188–285 s |
| lemma S + old sector lemma + flat estimate over a box | minutes (not timed separately) | minutes |
| `hier.py` zero, then `refine_zero_M.py` | 9 min, then 76 s | (8-thread run, README), then 35 min |
| 6a backward flow (pilot), ξ₁ = 0.2 → 0.067 | 389 steps, 50 min | not run |
| 6b₁ ray run (11 legs), C¹ mode | 903 steps, about 30 min (resumed once) | not run |
| 6b₁ rows 0–11, C¹ mode, 183 discs | 11 676 Lohner steps, 12 565 s of process time (two processes, about 2 h wall) | not run |

Per Lohner step, M = 3 costs about 6 times M = 1 (2 s against 0.3 s at comparable load). That is consistent with the
dimension (15 against 9 states, cubic in the matrix work) and the number of harmonic products. The C¹ columns add
almost nothing per step: 100 s for the first ray leg, against 102 s without them.

**Estimate for M = 5 (not measured):** 18 rows and 366 discs, 2.2 times the arc length of M = 3, at about 3 times the
cost per step (19 against 13 states). That is roughly 6 times M = 3, about 20 hours of process time: 4–5 hours on
5 local threads with the rows in parallel. A pilot of its innermost row and its disc radii comes first.

**Proposed task for the local 8-thread machine** (for the coordinator to queue in comms.txt; nothing has been written
there). Below-normal priority, one process per thread, from `solver/proof/` on a checkout containing this branch.

*Part A, lemma 6a(3) with the certificate's state box: 1 thread, about 1 hour.* When the certificate's M = 3 box run is
final, write its hull of y⁺(0.3; X_c) as JSON with keys M, h, x (its p̂), xi1 = "0.3" and yX_xi1 (13 [re, im] arb
strings in `lateral_np.py`'s order). Then:

```
python3 lemma6_jump_M.py --M=3 --box=<that JSON> --json=lemmas_M_runs/lemma6a_M3_cert.json
```

- Pass: ρ₆ₐ < 1.
- Stop if the backward integration fails (step-size underflow) or G_slow is not proved invertible. B₁'s width is then
  the limit, as in the x̂ pilot of §7.
- Also check (H2): its final centre within about 10⁻¹⁰ of the p̂ used in §5.3, and h ≤ 10⁻⁹. Otherwise rerun the
  covering with `--x=` and `--h=` (Part B's commands with --M=3 and rows 0–11, about 3.5 hours of process time).
- Push the JSON and the log.

*Part B, lemma 6b₁(5), after the M = 5 certificate: 5 threads, about 5 hours.* With P the certificate's M = 5 centre
and H ≥ its h:

```
python3 lemma6b1_cover_M.py probe --M=5                                           # NOT rigorous: disc radii, ~15 min
python3 lemma6b1_cover_M.py ray --c1 --M=5 --h=$H --tag=_cert --x=$P              # ~1-2 h; --resume after a restart
python3 lemma6b1_cover_M.py row  --M=5 --h=$H --tag=_cert --x=$P --j=17          # the innermost row first (pilot)
python3 lemma6b1_cover_M.py rows --M=5 --h=$H --tag=_cert --x=$P --only=...      # the other 17 rows over 5 threads
python3 lemma6b1_cover_M.py check --M=5 --h=$H --tag=_cert --x=$P --json=lemmas_M_runs/lemma6b1_check_M5_cert.json
python3 lemma6b1_cover_M.py c1check --M=5 --h=$H --tag=_cert --x=$P                # control
python3 lemma6b1_cover_M.py c1check --mutate=nov --M=5 --h=$H --tag=_cert --x=$P   # mutation
```

- Stop if the probe shows the M = 3 radii failing at the inner rows. In that case set `disc_params` for M = 5, which
  changes the design, so the ray run must come after it.
- Stop if row 17 has a disc that is not certified.
- Stop if the ray run's start fails. The C¹ start data at M = 5 use the old sector lemma at r₀ = 0.07
  (`START_R0`), whose complex Cauchy box has not been run at M = 5 with `--sector`.
- Push `lemma6b1_runs_M5_cert/` and the logs.

## 10. Reproduce

From `solver/proof/`. Times are wall-clock on the shared 2-core machine at load 15–19, single process under
`nice -n 10`.

```
python3 lemma6_usys_M.py                                   # the system self-tests (a)-(f), M = 3 and 5, ~15 s
python3 fixed_point_M.py --check-m1                        # M = 1 reproduces fixed_point.py, ~20 s
python3 fixed_point_M.py --M=3                             # step (i) at the M = 3 zero, ray 3pi/8, r0 = 0.1, plain, seconds
python3 fixed_point_M.py --M=3 --diag --r0=0.0741 --arg=5pi/16 --cbox=1e-3   # over the complex box, seconds
python3 fixed_point_M.py --table --Ms=3 --json=lemmas_M_runs/fixed_point_M_table_M3.json        # section 2.4, ~3 min
python3 fixed_point_M.py --table --Ms=5 --json=lemmas_M_runs/fixed_point_M_table_M5.json        # ~6 min
python3 fixed_point_M.py --data-table --Ms=3 --json=lemmas_M_runs/fixed_point_M_data_M3.json    # start data, ~3 min
python3 fixed_point_M.py --M=3 --data --box=1e-9 --r0=0.0741 --arg=5pi/16 --diag    # start data, printed per component
python3 lemma6_sector_M.py --M=3 --scan --json=lemmas_M_runs/lemmaS_scan_M3.json      # lemma S over (r+, k), ~10 min
python3 lemma6_sector_M.py --M=3 --rp=0.1 --k=20 --box=1e-6 --Z=0.004489 --json=lemmas_M_runs/lemma6_sector_M3_rp0.1_1e-6.json
                                                           # lemma S, old sector lemma, flat estimate over x^ +- 1e-6, ~1 min
python3 lemma6_sector_M.py --M=5 --rp=0.08 --k=20 --box=1e-6 --Z=0.004489 --json=lemmas_M_runs/lemma6_sector_M5_rp0.08_1e-6.json
python3 lemma6_jump_M.py --M=3 --pilot=1e-10 --xi1=0.2 --rho=0.2 --save-B=lemmas_M_runs/lemma6a_pilot_M3_B.json \
    --json=lemmas_M_runs/lemma6a_pilot_M3_xi0.2.json       # 6a pilot (NOT rigorous: double-precision B1), ~1 h
python3 lemma6_jump_M.py --M=3 --box=<certificate box JSON>   # 6a proper, once the point certificate exists
python3 lemma6b1_estimate_M.py --M=3                       # planning: sizes and singularity distances, ~2 min
python3 lemma6b1_disc_M.py                                 # lemma D at M = 1 reproduces lemma6b1_disc, ~15 s
python3 lemma6b1_cover_M.py design --M=3                   # the rows and tiles
python3 lemma6b1_cover_M.py probe --M=3                    # NOT rigorous: disc radii at double-precision centres, ~5 min
python3 lemma6b1_estimate_M.py --M=3 --arc=26.98 --h=1e-8 --box0=2.5e-5   # why the hull restart fails (row 11)
python3 lemma6b1_cover_M.py ray --M=3 --h=1e-8             # the hull method (fails on row 11, kept for the record)
P=2.7264758856311463335934955261561100705557331671411,1.0156030948577872100059533565790022881883333496973,0.14441947847010155933389143095194965903793037966287,0.011087425888116783540971263261694760868044985018780
python3 lemma6b1_cover_M.py ray --c1 --M=3 --h=1.1e-9 --tag=_cert --x=$P   # C^1 ray run on the certificate's box,
                                                           # ~30 min -> lemma6b1_runs_M3_cert/ray.json (--resume after a restart)
python3 lemma6b1_cover_M.py row  --M=3 --h=1.1e-9 --tag=_cert --x=$P --j=11            # the hardest row, ~26 min
python3 lemma6b1_cover_M.py rows --M=3 --h=1.1e-9 --tag=_cert --x=$P [--only=J1,J2,..] # the other rows, ~3 h
python3 lemma6b1_cover_M.py check --M=3 --h=1.1e-9 --tag=_cert --x=$P --json=lemmas_M_runs/lemma6b1_check_M3_cert.json
                                                           # C1-C8: must print PASS, 1 s
python3 lemma6b1_cover_M.py c1check --M=3 --h=1.1e-9 --tag=_cert --x=$P    # planted truth for the C^1 sets, ~5 min
python3 mutations_M.py [--all]                             # section 8; exit 0 only if every verdict matches
python3 refine_zero_M.py --M=3 --A=2.726475886 --wf=1.015603095 --x=0.144419478,0.011087426   # the zero (double precision)
python3 lemmas_M_report.py all                             # the tables of this note, from the saved runs
```

## 11. Proposed text for A_certify.md and PROOF.md

The coordinator edits those files; this is proposed text only.

**For A_certify.md §7 ("From M = 1 to M = 5"), a new last bullet:**

> - **The lemmas around the point certificate** ([A_lemmas_M.md](A_lemmas_M.md)). Step (i), lemma S, lemma 6b₀, the
>   flat estimate of 6a and the disc lemma of 6b₁ carry over to M harmonics with the same proofs. The fast eigenvalue
>   repeats across harmonics, so the inverse operator, its domains and the kernel bounds are those of M = 1. The labels
>   enter through s_μ = iμω in the O(1/z) terms. On rays their weight cancels exactly, since |(t/z)^{s_μ}| = 1, and a
>   second splitting of the fast row ("diag") removes them from the contraction. On lemma S's paths they remain
>   perturbations of size |s_μ|R₊. Step (i) is certified at M = 3 and 5 at `hier.py`'s zeros, at points and over boxes,
>   with start data in all M + 1 parameters (`fixed_point_M.py`); M = 1 reproduces `fixed_point.py` bit for bit.
>   At M = 3, lemma S, lemma 6b₀ and the flat estimate of 6a are certified over x̂ ± 10⁻⁶, and lemma 6b₁ is certified
>   on the point certificate's box p̂ ± 1.1×10⁻⁹ by a covering with 183 discs in a C¹ mean-value form. The M = 1
>   covering method fails on the innermost row at M = 3. The backward step of 6a waits for the certificate's state box
>   at ξ₁ = 0.3. At M = 5, lemma S, 6b₀ and the flat estimate are certified, and 6b₁ is a local-machine task.

**For PROOF.md §4, after "Prototype status at M = 1" (one paragraph):**

> *Toward M = 3 and 5* ([proof/A_lemmas_M.md](proof/A_lemmas_M.md)). The parts of Theorem A that turn a point
> certificate into the theorem have been generalized to M harmonics: the start-value fixed point with parameter
> derivatives, lemma S, and lemmas 6a, 6b₀ and 6b₁. At M = 3 all of them are certified on boxes that contain the
> point certificate's box, except the backward step of 6a, which needs that certificate's state box at ξ₁ = 0.3.
> At M = 5, 6b₁ remains to be run. Theorem A(3) then follows from a point
> certificate (validated integration and a Krawczyk test on the M + 1 real D-jump conditions at one radius) whose box
> lies in the boxes used there.

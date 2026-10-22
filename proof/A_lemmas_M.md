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

RESULTS_SUMMARY

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

FP_TABLE_M3

**M = 5 at the hier.py zero** (`--table --Ms=5`, `lemmas_M_runs/fixed_point_M_table_M5.json`):

FP_TABLE_M5

**Start data with derivatives** (`--data-table`), real boxes around x̂, Cauchy radius 10⁻³, diag splitting. The value
and derivative radii are those of the balls handed to the integrator: the true variation over the box plus the
enclosure. The Cauchy part is the bound on |∂η/∂x| in the (p, v, c) variables:

DATA_TABLE

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

SECTOR_TABLE

**Lemma S(3), lemma 6b₀(3) and the flat estimate at M = 3: certified** over X₆ = x̂ ± 10⁻⁶ (trusted code:
`lemma6_sector_M.py`, `fixed_point_M.py` and `fixed_point.py`'s majorant helpers, python-flint).

- Lemma S: R₊ = 0.01, k = 20, a_eff = 0.2536, contraction ≤ 0.8252.
- Old sector lemma: diag splitting, contraction ≤ 0.5408.
- Lemma 6b₀(3): y⁺ is analytic for 0 < |ξ| ≤ 0.0624, 0 ≤ arg ξ ≤ 3π/8, and on the real segment up to 0.0841, for every
  x ∈ X₆. These radii are exactly those of M = 1.
- Flat estimate at Z = 0.004489 (ξ_a = 0.067): Λ ≤ 0.2923 (plain) or 0.2199 (diag).

M5_SECTOR

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

SECTION_4_RESULTS

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
  15.0 at M = 1, where the branch point located in §12.1 is 15.8 away. So the rows with ρ < 32 use r = 8 and n = 50.
- The cost per disc and per Lohner step, from the dimension.

With these, the proof is that of §12.4: lemma D for each disc (C1), then lemma G with C2–C7, then C8. ∎

SECTION_5_RESULTS

## 6. Theorem A(M), conditional on the point certificate

Let x̂ be the double-precision zero (§2.4), X_c the box of the point certificate, and ξ₁ and ρ_arc its matching radius
and arc radius. Assume the certificate passes along a path of the kind used in §11.2: the start ray at an angle
θ ∈ [3π/8 − 0.2, 3π/8] from a radius r₀ ≤ 0.1, that ray out to ρ_arc ≤ 0.2, the arc |ξ| = ρ_arc to the real axis, and
then (if ξ₁ > ρ_arc) the real axis to ξ₁.

**Theorem A(M) (conditional).** Assume Braaksma's theorem (Lemma B4(M)). Suppose the point certificate proves that
x ↦ (J^path_{D,μ}(ξ₁; x))_{μ = 1, 3, …, M} has exactly one zero in X_c and that it is simple. Suppose further:

- (H1) X_c ⊂ X₆ = x̂ ± 10⁻⁶, the box over which lemma S(M), the old sector lemma and the flat estimate are proved here
  (§3.3);
- (H2) X_c ⊂ X_b, the box of the 6b₁ covering (x̂ ± 10⁻⁸ at M = 3, §5.3), and the covering is complete;
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

What is proved or certified here holds for the parameter boxes named. What needs the point certificate's output:

1. **Its box X_c and the zero.** (H1) and (H2) are inclusions to check against x̂ ± 10⁻⁶ and x̂ ± 10⁻⁸. If X_c is
   larger, or centred more than about 10⁻⁸ from x̂, `lemma6_sector_M.py --box=` and the covering must be rerun on a box
   containing it. The covering's cost then grows with the box: the disc lemma tolerates Lohner boxes up to about 10⁻⁵.
2. **Its state box B₁ at ξ₁ over X_c** (H3). `lemma6_jump_M.py --box=f.json` takes it in the format of
   `lemma6_path_box_1e-9.json` (keys M, h, xi1, yX_xi1). Only the pilot of §4.3 has run, on a non-rigorous B₁.
3. **Its path.** The covering here is for the region of §11.1 (|ξ| ≤ 0.2, 0 ≤ arg ξ ≤ 3π/8). A path with an arc above
   0.2, or a start ray outside 3π/8 ± 0.2, needs a different covering: the arc radius sets u_in = 1/ρ_arc², and the
   inner rows are where the cost and the singularity distance bite.
4. **Its conditions.** 6a needs all (M + 1)/2 complex D-jumps J_{D,μ}(ξ₁) at one radius, or an invertible linear image
   of them, such as `hier.py`'s de-mixed residuals. A certificate of S₁ = 0 alone at fixed higher harmonics would not
   suffice.
5. **Start data.** The certificate is expected to start from `fixed_point_M.start_data_M` (§2.3), with the recommended
   configuration of §2.4 for its M.

## 8. Mutation table

Every check added here has a runnable mutation or control. Each verdict was written down before the run (the expected
column), and `mutations_M.py` compares it with the outcome; exit status 0 only if all match. Results are in
`solver/proof/lemmas_M_runs/mutations_M.json`.

MUTATION_TABLE

## 9. Cost, and a proposed task for the local machine

COST_SECTION

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
python3 lemma6b1_cover_M.py ray --M=3 --h=1e-8             # start box and ray run -> lemma6b1_runs_M3/ray.json
python3 lemma6b1_cover_M.py row --M=3 --j=11 --h=1e-8      # one row (the hardest, innermost)
python3 lemma6b1_cover_M.py rows --M=3 --h=1e-8 [--only=J1,J2,..]   # the remaining rows
python3 lemma6b1_cover_M.py check --M=3 --h=1e-8 --json=lemmas_M_runs/lemma6b1_check_M3.json   # C1-C8: PASS / FAIL
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
>   STATUS_SENTENCE

**For PROOF.md §4, after "Prototype status at M = 1" (one paragraph):**

> *Toward M = 3 and 5* ([proof/A_lemmas_M.md](proof/A_lemmas_M.md)). The parts of Theorem A that turn a point
> certificate into the theorem have been generalized to M harmonics: the start-value fixed point with parameter
> derivatives, lemma S, and lemmas 6a, 6b₀ and 6b₁. PROOF_STATUS_SENTENCE Theorem A(3) then follows from a point
> certificate (validated integration and a Krawczyk test on the M + 1 real D-jump conditions at one radius) whose box
> lies in the boxes used there.

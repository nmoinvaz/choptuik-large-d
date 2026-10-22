# The scalar ℓ-multiplet in D dimensions: critical solutions and γ

This page computes the discretely self-similar (DSS) critical solution of a massless scalar ℓ-multiplet in D = n + 2
dimensions, with the periodic boundary-value method of the main solver, and follows Δ and γ = 1/λ in D. The question
is the one raised in [GAMMA.md](../GAMMA.md) and [STRUCTURE.md](STRUCTURE.md): for the ℓ = 0 scalar, γ → 1/2 because
the D = ∞ inner problem has an exact s-translation (λ = 2); a centrifugal term ∝ n/r², which is ∝ 1/s in inner
variables, breaks it. Does γ(D) of the ℓ = 1 multiplet still tend to 1/2?

Contents

- [1. Model and equations](#1-model-and-equations)
- [2. Boundary conditions and symmetry](#2-boundary-conditions-and-symmetry)
- [3. Numerical method](#3-numerical-method)
- [4. Validation](#4-validation)
- [5. D = 4: comparison with Olabarrieta et al.](#5-d--4-comparison-with-olabarrieta-et-al)
- [6. ℓ = 1 across D](#6-ℓ--1-across-d)
- [7. Extrapolation to D = ∞](#7-extrapolation-to-d--)
- [8. ℓ = 2](#8-ℓ--2)
- [9. Open issues](#9-open-issues)
- [10. Scripts and data](#10-scripts-and-data)

## 1. Model and equations

Take real massless scalars $`\phi_m = \psi(t,r)\,Y_m`$, where the $`Y_m`$ are real degree-ℓ harmonics on $`S^n`$ normalised so
that $`\sum_m Y_m^2 = 1`$; then $`\sum_m \nabla Y_m\cdot\nabla Y_m`$ is proportional to the sphere metric and the summed
stress tensor is spherically symmetric. With the main solver's normalisation, $`G_{ab} = 2n\sum_m\big(\partial_a\phi_m\partial_b\phi_m - \tfrac12 g_{ab}(\partial\phi_m)^2\big)`$,
ℓ = 0 is the single field Y = 1 and reproduces [`solver/dss.py`](../solver/dss.py) exactly. In polar-areal gauge,
$`ds^2 = -\alpha^2dt^2 + a^2dr^2 + r^2d\Omega_n^2`$, with $`\Phi = \psi_r`$, $`\Pi = (a/\alpha)\psi_t`$, c = α/a and

```math
L = \ell(\ell + n - 1),
```

the field equations are

```math
\frac{a_r}{a} = r(\Phi^2+\Pi^2) + \frac{L a^2\psi^2}{r} + \frac{(n-1)(1-a^2)}{2r}, \qquad
\frac{\alpha_r}{\alpha} = r(\Phi^2+\Pi^2) - \frac{L a^2\psi^2}{r} + \frac{(n-1)(a^2-1)}{2r},
```
```math
\partial_t\Phi = \partial_r(c\,\Pi), \qquad \partial_t\Pi = r^{-n}\partial_r\big(r^n c\,\Phi\big) - \frac{L\,\alpha a\,\psi}{r^2}, \qquad \psi_t = c\,\Pi .
```

In the self-similar variables of the solver, τ = −ln(−t), x = r/(−t), p = r(Φ + Π), m = r(Φ − Π), and with ψ itself
as a fifth field:

```math
x a_x = \tfrac{a}{2}\big[p^2+m^2 + 2La^2\psi^2 + (n-1)(1-a^2)\big], \qquad
x c_x = c\big[(n-1)(a^2-1) - 2La^2\psi^2\big], \qquad x\psi_x = \tfrac12(p+m),
```
```math
p_\tau + (x-c)\,p_x = c_x p - \frac{cp}{x} + \frac{nc\,(p+m)}{2x} - \frac{L c a^2\psi}{x}, \qquad
m_\tau + (x+c)\,m_x = -c_x m + \frac{cm}{x} - \frac{nc\,(p+m)}{2x} + \frac{L c a^2\psi}{x} .
```

**Checks** ([`solver/matter/ell_equations.py`](../solver/matter/ell_equations.py), sympy, all residuals 0):

- the Einstein tensor of the metric and the stress tensor of explicit harmonic multiplets, for (n, ℓ) = (2, 1),
  (2, 2), (3, 1), (3, 2), (4, 1): tt, rr and tr components give the constraints above and $`a_t = 2r\alpha\Phi\Pi`$, the
  mixed and off-diagonal angular components of the summed stress tensor vanish, and $`\Box\phi_m = 0`$ is the Π equation
  with L = ℓ(ℓ + n − 1);
- the self-similar equations follow from the (t, r) ones, and at L = 0 they are the solver's;
- at n = 2 they are Eqs. (15)–(18) of Olabarrieta, Ventrella, Choptuik and Unruh, PRD 76, 124014 (2007)
  ([arXiv:0708.0513](https://arxiv.org/abs/0708.0513)), with $`\psi = \sqrt{(2\ell+1)/2}\;\psi_{\rm OVCU}`$;
- with ψ = r^ℓ χ, in flat space the ℓ-mode operator in D dimensions is the radial operator in D + 2ℓ dimensions, and
  the χ-form equations used by the evolution code ([`ell_collapse.py`](../solver/matter/ell_collapse.py)) follow from the
  equations above.

The same equations were derived independently in [STRUCTURE.md](STRUCTURE.md) (§2), which agrees term by term.

## 2. Boundary conditions and symmetry

- **Centre.** The regular solution has ψ ≈ q(τ) x^ℓ and p ≈ m ≈ ℓ q x^ℓ. In the centre series the order-k system for
  $`(P_k, M_k)`$, after eliminating $`\Psi_k = (P_k+M_k)/(2k)`$, has determinant $`-c_0^2(k-1)(k-\ell)(k+\ell+n-1)/k`$ (sympy).
  At k = ℓ the null vector is $`P_\ell = M_\ell`$: the free data are q(τ) and c₀(τ) = c(τ, 0), as for ℓ = 0 (p₁, c₀).
  Orders below ℓ vanish; the root k = 1 for ℓ ≥ 2 would carry Π(τ, 0) ≠ 0, which the constraint below forbids.
- **The ψ constraint.** $`K = \psi_\tau + \tfrac12(p+m) - c\,(p-m)/(2x)`$ is ψ_t = cΠ in self-similar form. It is
  x-independent by the equations (sympy) and vanishes in the centre series. At the horizon it reads K = ψ_τ + m, so
  ψ(τ, 1) is a third free horizon datum (with m and a), and the matching of ψ at the last node enforces K = 0
  everywhere: unknowns and conditions balance. For ℓ = 0, ψ(τ, 0) is fixed by K = 0 at the centre and ψ decouples.
- **Horizon** (x = 1, c = 1): the p equation degenerates and regularity gives
  $`p_\tau + (1 - c_x(1) - n/2)\,p = \tfrac n2 m - La^2\psi`$, with $`c_x(1) = (n-1)(a^2-1) - 2La^2\psi^2`$ (sympy).
- **Symmetry.** The equations are invariant under (p, m, ψ) → −(p, m, ψ), so the ansatz p, m, ψ odd and a, c even
  under τ → τ + Δ/2 is consistent. The near-critical evolutions show it (the centre value of ψ changes sign every
  half period, with equal spacing in ln(T* − T)), and every solution below was found with it.

## 3. Numerical method

[`solver/matter/dss_ell.py`](../solver/matter/dss_ell.py), class `DSSEll(DSSSpec)`: the main solver's dealiased Fourier
discretisation in τ (harmonics up to k_c, N ≥ 3k_c points), multiple shooting in x with 12 interior nodes
(0.02 … 0.9), RK4, centre and horizon Taylor series generated order by order (the horizon series in the fast O(K²)
form of `solver/fast.py`), now with five fields and parities (odd, odd, even, even, odd). Changes relative to the ℓ = 0
solver:

- **Every derivative by complex step**: the Jacobian, ∂F/∂n (the continuation tangent) and the growth-mode matrix
  M(λ) are exact to rounding.
- **RK step counts from the stiffness of the actual solution.** The transport terms have rates $`k_c\omega/|x \mp c|`$;
  for ℓ ≥ 1 the centre lapse c(τ, 0) dips far below the ℓ = 0 values (to 0.068 at D = 4), and the ℓ = 0 rule (step ∝ 1/ω
  only) under-resolves the centre. Each segment's count is fixed from max $`k_c\omega/|x-c|`$ along it (times x in
  ln x, times 1 − x in ln(1 − x)) with target hλ ≤ κ, frozen while Newton iterates.
- **The ψ constraint as a resolution diagnostic.** K is not imposed; its size on the converged solution, relative to
  max |p|, measures τ-truncation and integration error. It exposed the first problem below.
- **Initial guess from evolution** ([`ell_collapse.py`](../solver/matter/ell_collapse.py),
  [`ell_bisect.py`](../solver/matter/ell_bisect.py), [`ell_guess.py`](../solver/matter/ell_guess.py)): the project's
  ℓ = 0 collapse code adapted to χ = ψ/r^ℓ, evolved in central proper time (α(t, 0) = 1) because the lapse collapses
  strongly; bisection on the amplitude of a Gaussian shell; the self-similarity horizon is the periodic orbit of
  dx_c/dτ_c = x_c − c_c in central-time similarity variables, found as the fixed point of the one-period map; one
  period of slices is Fourier-fitted onto the BVP unknowns.
- **Continuation in D** ([`ell_march.py`](../solver/matter/ell_march.py)): tangent predictor dz/dD = −J⁻¹∂F/∂n with a
  quadratic term from the previous point, adaptive steps landing on target D; Levenberg–Marquardt with one SVD per
  Jacobian. Above D = 8 the LM steps use weighted damping: each step minimises the linearised residual plus
  μ² times the (k/k_c)⁴-weighted step, with μ = 10⁻⁷ σ_max, so the near-null high harmonics of the centre datum (the
  "junk" directions of [solver/notes.md](../solver/notes.md)) are not moved, while the iteration still converges to an
  exact zero of F (no bias). The tangent is solved with the same weighting. RK step counts are re-frozen during the
  iteration only if one changes by more than 15% (each change moves the discrete residual by ~10⁻⁸).
- **Growth rate**: λ is the real root of det M(λ), bracketed around the value extrapolated from neighbouring D and
  refined by brentq; γ = 1/λ is the exponent of the black-hole radius (in D = 4 also the mass exponent; in D dimensions
  the mass exponent is (D − 3)γ).

## 4. Validation

**ℓ = 0 reproduces the main solver** ([`ell_validate.py`](../solver/matter/ell_validate.py),
`data/ell0_validation.jsonl`). The stored 4-field solutions are converted (ψ = 0, which decouples at L = 0) and
re-solved by `DSSEll` on the same layout (x_K = 0.6 long horizon series, frozen RK steps):

| D | k_c | Δ, DSSEll | Δ, stored solution | λ, DSSEll | λ, stored |
|---|---|---|---|---|---|
| 4 | 39 | 3.4454554891937152 | 3.4454554891937237 | 2.6740837626423 | 2.6740837626424 |
| 8 | 23 | 2.7117900432302107 | 2.7117900432302644 | 2.2385567932663 | 2.2385–2.2390 (notes.md) |

The agreement is to rounding (≤ 9×10⁻¹⁴). The k_c = 39 value at D = 4 lies 3×10⁻⁶ above the accepted
3.445452402(3) (Martín-García & Gundlach 2003); that is the stored solution's own τ truncation (k_c = 47 gives
3.44545268), not a difference between the codes. γ = 1/λ = 0.373960 at D = 4.

**The ψ constraint as a check.** On every ℓ ≥ 1 solution, K is x-independent to rounding from the first node out, as
the analysis requires; its size is a resolution measure (table in §5).

## 5. D = 4: comparison with Olabarrieta et al.

**Reference values.** Olabarrieta, Ventrella, Choptuik and Unruh, PRD 76, 124014 (2007), Table III (checked against
[arXiv:0708.0513](https://arxiv.org/abs/0708.0513)): Δ₁ = 0.460 ± 0.002, γ₁ = 0.119 ± 0.001, Δ₂ = 0.119 ± 0.003,
γ₂ = 0.0453 ± 0.0002, with Δ measured in central proper time (the same invariant period as ours) and error bars from
the spread over six initial-data families. Their γ is the mass exponent, which in D = 4 equals our radius exponent.

**Evolution** (ℓ = 1, Gaussian shell in χ, central slicing, dx = 6×10⁻⁴; `data/bisect_n2_l1_dx0.0006.jsonl`). Bisection
to a relative bracket of 8×10⁻¹³ gives 8 consecutive zero-crossing triples of ψ's centre value with
Δ = 0.4594 and T* = 7.6493 (central proper time), and subcritical runs whose peak 2m/r converges to 0.688. One caveat
found on the way: the ℓ = 0 code's black-hole flag (max 2m/r ≥ 0.95) and its time normalised at r_max are both wrong
here. The flag trips before a horizon forms, so a bisection on it tunes to a different threshold, and the lapse
collapses so far that central proper time barely advances in outer time.

**BVP** (`data/l1_D4_kc*.jsonl`), from that guess, with the ψ constraint as the resolution measure:

| k_c | Δ | ψ-constraint |
|---|---|---|
| 11 | 0.46077 | 0.26 |
| 23 | 0.4598654552 | 1.5×10⁻² |
| 31 | 0.4598612002 | 1.7×10⁻³ |
| 39 | 0.4598611019 | 2.0×10⁻⁴ |

The k_c = 11 solution satisfies its discrete equations to 10⁻¹⁵ and is still wrong by 10⁻³ in Δ: the centre lapse
c(τ, 0) dips to 0.068 once per half period, the centre data carry (q_τ + q)/c₀, and their τ spectrum falls only like
$`e^{-0.3k}`$. The constraint is what showed it. The differences shrink by a factor of ~40 from k_c = 31 to 39, so

```math
\Delta_{\ell=1}(D=4) = 0.4598611 \pm 3\times10^{-7}
```

(the error bar is dominated by the RK step size, estimated at D = 8 from κ = 0.5 against κ = 1).
The growth rate at k_c = 23 is λ = 8.410420, i.e.

```math
\gamma_{\ell=1}(D=4) = 0.118900 ,
```

found by bracketing det M(λ) around the published value (a full scan over [0.3, 20] was started and lost to a time
limit); the gauge rate is 0.895. At k_c = 23 and D = 4 the solution is the least resolved of all points (constraint
1.5%), so this γ is good to about 10⁻⁴ at best (not checked at higher k_c). Both numbers agree with Olabarrieta et al.
within their error bars.

## 6. ℓ = 1 across D

**Continuation.** From D = 4 the solution changes fast: ‖dz/dD‖ = 18 against ‖z‖ ≈ 10, and dΔ/dD = +0.50. A first
pass at k_c = 15 (steps 0.02 → 0.3; `data/march_l1_kc15.jsonl`) reached D = 9.3, but its ψ-constraint grew from
8×10⁻⁴ (D = 6) to 5×10⁻² (D = 9.3) as the centre data's high harmonics drifted (|q₁₁| = 0.15 > |q₃| at D = 9): the
"junk" of [solver/notes.md](../solver/notes.md), here a k_c = 15 truncation artefact. All values below are k_c = 23,
re-solved from those points (D ≤ 8) or continued at k_c = 23 (D ≥ 8; `data/march_l1_kc23m.jsonl`), with κ = 0.7,
horizon series from u = 0.04 (30 terms), every point converged to the residual shown.

| D | n | Δ | λ = 1/γ | γ | (D − 3)γ | abs(F) | ψ-constraint |
|---|---|---|---|---|---|---|---|
| 4 | 2 | 0.4598611 | 8.410420 | 0.118900 | 0.118900 | 5×10⁻¹⁵ | 2×10⁻⁴ (k_c 39) |
| 5 | 3 | 0.7833439 | 4.640174 | 0.215509 | 0.431018 | 2×10⁻¹⁵ | 3×10⁻⁴ |
| 6 | 4 | 0.9115281 | 3.765555 | 0.265565 | 0.796695 | 7×10⁻¹³ | 2×10⁻⁵ |
| 7 | 5 | 0.9639878 | 3.381429 | 0.295733 | 1.182932 | 3×10⁻¹³ | 2×10⁻⁵ |
| 8 | 6 | 0.9837905 | 3.162529 | 0.316203 | 1.581013 | 1×10⁻¹² | 2×10⁻⁶ |
| 9 | 7 | 0.988205 | 3.018537 | 0.331286 | 1.987718 | 1×10⁻¹¹ | 2×10⁻⁶ |

Additional Δ points from the k_c = 23 continuation: Δ(8.15) = 0.9851763, Δ(8.375) = 0.9867067, Δ(8.7125) = 0.9879556,
Δ(9.05) = 0.9881824. (D − 3)γ is the mass exponent. Data: `data/l1_kc23r.jsonl`, `data/l1_D4_kc*.jsonl`,
`data/l1_kc23_gamma.jsonl`; table from [`ell_table.py`](../solver/matter/ell_table.py).

**Accuracy.** τ truncation: k_c 23 → 31 at D = 6 changes Δ by 7×10⁻⁸ and λ by 2.6×10⁻⁷. RK steps: at D = 8, κ = 1 → 0.5
moves Δ by 3.4×10⁻⁶ (∝ κ⁴), so κ = 0.7 carries ~1×10⁻⁶; horizon start u = 0.01 → 0.04 moves it by 4×10⁻⁷
(`data/ell_tests.jsonl`). Selection: see below; at D = 9 it adds ~4×10⁻⁶ to Δ. So Δ is good to ~1×10⁻⁶ for
D = 5–8 and ~5×10⁻⁶ at D = 9; λ to ~10⁻⁶ relative (the RK error of λ was not measured separately).

**How the solution changes with D.** Δ rises steeply from D = 4, has a maximum Δ ≈ 0.98821 near D ≈ 9.0, and then
decreases (dΔ/dD ≈ −4×10⁻⁴ at D = 9.05). γ rises monotonically, 0.119 → 0.331. The centre lapse dip relaxes,
min c(τ, 0) = 0.068, 0.25, 0.40, 0.46 at D = 4, 6, 8, 9, and with it the τ spectrum narrows: the ψ-constraint at fixed
k_c falls by two orders of magnitude from D = 4 to 8. The gauge rate (horizon relabelling) falls slowly, 0.895 → 0.826,
far from λ. At D = 5 a scan of det M(λ) over [1, 12] (k_c = 15) finds exactly one real root, λ = 4.6403, the one
followed here; at the other D the root was followed by bracketing around its extrapolation, so further real roots
were not searched for there, and complex roots were never searched for.

**The selection weakens fast, and stops the double-precision continuation at D ≈ 9.** The sensitivity of Δ to the
residual, abs(dΔ/dF) = ‖J⁻ᵀe_Δ‖, is 1.0×10⁵ (D = 8.15), 1.5×10⁵ (8.375), 2.5×10⁵ (8.71), 3.8×10⁵ (9.0), i.e. ×~5
per unit D. For the ℓ = 0 scalar the same quantity is ~10² at D = 8 and 2×10⁵ only at D = 21
([solver/notes.md](../solver/notes.md)). Re-solving the D = 9 point from abs(F) = 5.5×10⁻¹¹ to 9.7×10⁻¹² moved Δ by
1.8×10⁻⁵, as that sensitivity predicts. Past D = 9 the Newton basin becomes tiny along the weak directions
(σ_min/σ_max ≈ 2×10⁻¹² of the Jacobian): a tangent step of 0.2 stalls at abs(F) = 1.6×10⁻⁹ off the branch, a step of
0.05 still converges to 4×10⁻¹⁵ (`data/ell_tests.jsonl`). Continuing would cost ~2 h per unit of D at this rate and
then hit the same wall as the ℓ = 0 scalar at D ≈ 26 (rounding against selection), much earlier. Both facts fit
[STRUCTURE.md](STRUCTURE.md): for ℓ ≥ 1, gravity at x = O(1) is O(n^(−ℓ−1)) instead of O(1/n), so the beyond-all-orders
selection is weaker at a given D.

## 7. Extrapolation to D = ∞

**Direct fits of λ(D)** ([`ell_fit.py`](../solver/matter/ell_fit.py), `data/ell_fit.json`), D = 5–9, with λ∞ free and with
λ∞ = 2 imposed (same number of fitted coefficients):

| variable | points | free coefficients | λ∞ free | rms, λ∞ free | rms, λ∞ = 2 |
|---|---|---|---|---|---|
| 1/n | D = 5–9 | 3 | 2.658 | 6.6×10⁻³ | 1.8×10⁻³ |
| 1/n | D = 5–9 | 4 | 2.146 | 1.1×10⁻⁴ | 3.4×10⁻⁴ |
| 1/n | D = 6–9 | 3 | 2.458 | 1.0×10⁻³ | 6.5×10⁻⁴ |
| 1/D | D = 5–9 | 3 | 3.953 | 1.6×10⁻² | 7.0×10⁻³ |
| 1/D | D = 5–9 | 4 | 0.555 | 1.3×10⁻³ | 6.5×10⁻⁴ |
| 1/D | D = 6–9 | 3 | 3.039 | 2.3×10⁻³ | 7.5×10⁻⁴ |

With n = 3–7 the free limit ranges from 0.55 to 3.95, and imposing λ∞ = 2 fits as well or better in five of six cases.
**These fits do not determine λ∞**; the points are far from the asymptotic regime (for ℓ = 0 the fits needed
D ≳ 10–20).

**Against the ℓ = 0 scalar.** A sharper test subtracts the known ℓ = 0 growth rate at the same D, λ₀(D) → 2 with
λ₀ − 2 ≈ 3.03/n ([GAMMA.md](../GAMMA.md); values from `solver/data/scan_ne6_lowD_kc39_rk2.jsonl` and `gamma_rust.jsonl`):

| D | n | λ₀ | δ = λ₁ − λ₀ | n δ |
|---|---|---|---|---|
| 5 | 3 | 2.420564 | 2.219610 | 6.659 |
| 6 | 4 | 2.325152 | 1.440403 | 5.762 |
| 7 | 5 | 2.272773 | 1.108657 | 5.543 |
| 8 | 6 | 2.238557 | 0.923972 | 5.544 |
| 9 | 7 | 2.213933 | 0.804605 | 5.632 |

δ falls like ~5.6/n over n = 4–7. Its limit c = λ₁∞ − λ₀∞ from consecutive windows:

| window | c from c + a/n | c from c + a/n + b/n² |
|---|---|---|
| n = 3–4 / 3–5 | −0.897 | +0.800 |
| n = 4–5 / 4–6 | −0.218 | +0.438 |
| n = 5–6 / 5–7 | +0.001 | +0.308 |
| n = 6–7 | +0.088 | – |

The two-term estimates rise and the three-term estimates fall; if both converge monotonically, they bracket

```math
\lambda_{\ell=1,\infty} - 2 \in (0.09,\ 0.31), \qquad \gamma_{\ell=1,\infty} \in (0.43,\ 0.48) ,
```

and geometric extrapolation of the two sequences gives ~0.15 and ~0.23. **This bracket is a heuristic, not an error
bar**: it rests on monotone convergence over four points at n ≤ 7. What the data do show is that λ₁ − λ₀ shrinks like
1/n down to D = 9, so any offset λ₁∞ − 2 is small (≲ 0.3), and λ₁∞ = 2 is not excluded.

**Δ.** Δ₁ has a maximum 0.98821 near D ≈ 9 and then decreases, while the ℓ = 0 period falls towards Δ₀∞ = 1.6439 from
above (Δ₀(9) = 2.60). Unless Δ₁ turns up again, Δ₁∞ < 0.99 ≠ Δ₀∞. Δ₁ shows no sign of going to 0 or ∞; free
polynomial fits give 0.64–0.93 depending on variable and degree, i.e. Δ₁∞ is not determined by these data.

**Verdict.**
- γ∞ for ℓ = 1: not determined at the level needed. The direct fits are compatible with γ∞ = 1/2; the comparison with
  ℓ = 0 suggests λ₁∞ slightly above 2 (γ₁∞ ≈ 0.43–0.48, heuristic) and bounds the departure from λ₀ to a few tenths.
  This is consistent with the structural prediction (no s-translation, so no reason for λ∞ = 2) but does not confirm
  it.
- Δ∞ for ℓ = 1: finite on present evidence and, given the maximum at D ≈ 9, different from the ℓ = 0 limit 1.6439, in
  line with STRUCTURE.md's "finite and ℓ-dependent".

## 8. ℓ = 2

Only D = 4 was reached, and only partly.

- **Evolution** (`data/bisect_n2_l2_dx0.0006_r150.jsonl`, `pstar_n2_l2_dx0.0006_r150.json`). With γ₂ ≈ 0.045 the
  unstable mode grows like $`e^{22\tau}`$, so even tuning to 10⁻¹² gains only ~1.3 e-folds of scale: the self-similarity
  horizon stays at r ≈ 40–60 during the echoing, and the grid has to reach r_max = 150. (A first bisection with
  r_max = 30 and t_end = 15 was limited by t_end: black-hole times reached 14.97, so its last verdicts were
  undecided.) The r_max = 150 bisection to a relative bracket of 7×10⁻¹³ gives 12 consecutive consistent crossing
  triples with Δ = 0.1193 and T* = 23.70, against Olabarrieta et al.'s Δ₂ = 0.119 ± 0.003.
- **BVP.** From one fitted period (fit residuals 1–6%), k_c = 15, the residual fell from 2.2×10⁻² to 5.3×10⁻⁵ in two
  Jacobians (Δ = 0.11961 at that iterate) before the run was stopped: at ω ≈ 53 each Jacobian took ~6 min, and the
  machine was then shared by other jobs. The ℓ = 1 experience at D = 4 says a converged value needs k_c ≳ 31 with ~4×
  the ℓ = 1 step counts. **No converged ℓ = 2 solution and no γ₂ were obtained**, so the question whether the
  departure from λ = 2 grows with ℓ is not addressed here.

## 9. Open issues

- **Larger D.** Settling λ∞ needs ℓ = 1 points to D ≈ 15–20. In double precision the continuation stalls at D ≈ 9
  (§6). What would help, in order of effort: (i) continuing in steps of ≤ 0.05 (converges, ~2 h per unit D here);
  (ii) the project's nonlinear elimination of the weak directions (`solver/ne_solver.py`) or frozen top harmonics of
  the centre data (`hp/`'s kfreeze); (iii) a double-double port (the `hp/` kernel with the ψ field and the L terms).
- **Growth-rate checks.** λ(D = 4) at k_c = 31 was started twice and lost (container restart, then stopped for load);
  k_c = 23 is the least resolved point (constraint 1.5%), so γ(4) = 0.118900 should be taken as ±10⁻⁴. The RK error
  of λ was not measured on its own. Complex growth rates were never searched for, and real ones only near
  the followed root for D ≥ 6.
- **ℓ = 2 across D** was not attempted beyond D = 4.
- The heuristic bracket of §7 should be replaced by fits on larger D before anything is quoted from it.

## 10. Scripts and data

All in [`solver/matter/`](../solver/matter/); data in `solver/matter/data/`.

| Script | What it does |
|---|---|
| `ell_equations.py` | sympy: Einstein tensor and multiplet stress tensor for explicit harmonics, self-similar form, ℓ = 0 limit, OVCU07 equations, centre determinant, horizon condition, ψ constraint, χ form (`--chi`: that part only) |
| `dss_ell.py` | `DSSEll`: the 5-field periodic BVP (series, shooting, complex-step Jacobian, ∂F/∂n and M(λ), stiffness-frozen RK steps), LM solvers (plain and weighted-damping), growth-rate root finding |
| `ell_validate.py` | ℓ = 0 against stored 4-field solutions (`ell0_validation.jsonl`) |
| `ell_collapse.py`, `ell_bisect.py` | evolution of the multiplet (χ = ψ/r^ℓ, central slicing) and amplitude bisection (`bisect_*.jsonl`, `pstar_*.json`) |
| `ell_guess.py`, `ell_make_guess.py` | near-critical slices → self-similar data → Fourier-fitted BVP guess |
| `ell_scan.py`, `ell_march.py` | D-continuation (fixed list / adaptive with tangent predictor) |
| `ell_refine.py`, `ell_gamma.py`, `ell_gamma_series.py` | re-solve at higher k_c, growth rates |
| `ell_regtest.py`, `ell_spectrum.py` | sensitivity to prior, κ and horizon start; harmonic content and constraint |
| `ell_fit.py`, `ell_table.py` | large-D fits (λ∞ free / = 2, λ₁ − λ₀) and the results table |

Main data: `l1_D4_kc{11,23,31,39}.jsonl` (D = 4 k_c convergence), `l1_kc23r.jsonl` (Δ, λ at D = 5–9),
`l1_kc23_gamma.jsonl` (λ at D = 4), `march_l1_kc15.jsonl` / `march_l1_kc23m.jsonl` (continuations, with dz/dD, dΔ/dD,
abs(dΔ/dF)), `ell_tests.jsonl` (accuracy and convergence tests), `ell_fit.json`, `ell_gamma.jsonl`; solutions
`sol_*.npy` (DSSEll layout, `NODES_STD`).

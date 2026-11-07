# Layer A beyond one harmonic, second order: the C² point certificate

This note is the second-order version of [A_M.md](A_M.md): the same function g (the D-jumps at ξ₁ = 0.3 on the M + 1
real conditions, A_M.md §1), the same path and start-value lemma, and the same Krawczyk theorem, with a second-order
chart and second-order (C²) enclosures of the flow, so that the Jacobian over the box is enclosed in mean value form.
It is the design change A_M.md §2 and §7 name after the first-order M = 5 runs failed. Written by Claude (Anthropic)
under Nathan Moinvaziri's direction; not reviewed.

**Status.** M = 5: **certified**, given the M-harmonic start-value lemma `fixed_point_M.start_data_M` and Braaksma's
theorem, on a curved parallelepiped of outer half-width 10⁻¹⁰ and inner widths 10⁻¹⁸ (harmonic 3) and 10⁻¹⁹
(harmonic 5), with margins of at least 0.889 (§6.4); the new code is not reviewed. The same code path reproduces
Theorem A(M = 1) on a box 2×10⁴ times larger than the first-order one and the M = 3 certificate on boxes 10² and 10⁴
times larger (§6.1–6.2). What limited the first-order runs was the excess of the C¹ enclosure: their derivative
radius at ξ₁ was 10² to 10⁵ times the true variation over the box (§1).

Code: [`solver/proof/certify_M2_lohner.py`](../solver/proof/certify_M2_lohner.py) (the second derivative of the
right-hand side, the second variational series and its remainder, the C² Lohner step),
[`certify_M2.py`](../solver/proof/certify_M2.py) (the chart, the start data with second derivatives, the run, the
test, the controls), [`certify_M2_fd.py`](../solver/proof/certify_M2_fd.py) (finite differences of the C¹ numerics:
independent of the C² code, for the chart, the scale estimates and the control D2). Data and logs:
`solver/proof/data/A_M2_*`, `solver/proof/logs/A_M2_*`.

Contents

- [1. What limited the first-order test at M = 5](#1-what-limited-the-first-order-test-at-m--5)
- [2. Design](#2-design)
- [3. Scale estimates](#3-scale-estimates)
- [4. What the code proves, and what it assumes](#4-what-the-code-proves-and-what-it-assumes)
- [5. Controls, written down before the runs](#5-controls-written-down-before-the-runs)
- [6. Results](#6-results)
- [7. Reproduce](#7-reproduce)

## 1. What limited the first-order test at M = 5

The one first-order run that reached ξ₁ (h = 10⁻¹³, w = (10⁻¹⁹, 10⁻²³), `data/A_M_box_M5_1e-13.json`) splits as
follows (`python3 certify_M2.py diag1 --box=data/A_M_box_M5_1e-13.json --fd=data/A_M2_fd_M5.json`,
`logs/A_M2_diag1_M5_1e-13.log`). The radius of Y g̃(0) contributes at most 10⁻⁵ of a width, and the point part of
Y Dg̃ is the identity to 10⁻²¹². The failure is the radius of Y Dg̃(X), and most of it is in the columns of the
harmonic-3 coordinates:

| row block | outer columns (rad·h/wᵢ) | harmonic-3 columns (rad·w₃/wᵢ) | harmonic-5 columns (rad·w₅/wᵢ) |
|---|---|---|---|
| outer | 0.01–0.02 | 6.4–7.3 | 0.008–0.009 |
| harmonic 3 | 0.007–0.013 | 0.32–0.34 | 0.0004 |
| harmonic 5 | 6.2–11.3 | 247–263 | 0.31–0.33 |

That radius is mostly wrapping, not curvature. The true variation of Y Dg̃ over the box, Σ_k |Y D²g̃(0)|_{i,jk} w_k
from the finite differences of §3 (the first-order version of |Dg̃(p̂ ± w_k e_k) − Dg̃(p̂)|), is smaller than the C¹
radius by these factors (same log; the control is the certified M = 3 first-order box, h = 10⁻¹⁰,
`logs/A_M2_diag1_M3_1e-10.log`):

| ratio C¹ radius / true variation | outer columns | inner columns |
|---|---|---|
| M = 5, T33's box (h = 10⁻¹³) | 4.3×10⁴ – 4.5×10⁵ | 290 – 4150 |
| M = 3, the certified box (h = 10⁻¹⁰) | 7.2×10³ – 5.3×10⁵ | 99 – 1690 |

So the excess dominates at both M, by two to five orders of magnitude; at M = 3 the margins happened to absorb it,
at M = 5 they did not. The C¹ excess is linear in the box (the derivative radii in the four T33 logs are proportional
to h at equal steps), as at M = 1 (A_certify.md §9.3–9.4). The intrinsic curvature alone would allow far larger boxes
(§3), so a second-order chart without second-order enclosures, or a smaller first-order box, is not what is missing.

## 2. Design

**Chart (second order).** Measure the inner unknowns from the quadratic approximation of the zero manifold of g_H:

```math
p(x) = \hat p + T x + \tfrac12 \begin{pmatrix} 0 \\ \Gamma[x_o, x_o] \end{pmatrix}, \qquad
T = \begin{pmatrix} I_2 & 0 \\ C & I_{M-1} \end{pmatrix}, \qquad
\Gamma_{i,jk} = -\big[(\partial_H \tilde g_H)^{-1} \partial^2_{jk} \tilde g_H\big]_i ,
```

with C the tilt of A_M.md §2 and the second derivatives taken in the first-order chart at p̂ (j, k outer). Then
D²_oo g̃_H(0) = 0 up to the accuracy of Γ, so g_H changes only at third order along the outer directions. Γ comes from
central differences of the C¹ numerics (`certify_M2_fd.py`; non-rigorous, and it need not be: any chart is allowed).
p(·) is injective with a unit lower-triangular derivative, so zeros of g̃ = g∘p in X and zeros of g in p(X)
correspond one to one, with the same simplicity.

**What is enclosed.** Over the box X = [−h, h]² × Π[−w_ν, w_ν]² (centred at 0), at every step of the path:

- (a) the centre trajectory y(t; 0), as a Lohner set c + Qq;
- (b) the first derivative V₀ = ∂y/∂x (t; 0) along the centre trajectory, m = M + 1 columns, as a Lohner set;
- (c) the second derivative W = ∂²y/∂x∂x (t; x) for every x ∈ X, one column per pair j ≤ k, as a Lohner set.

From these and Taylor's theorem with integral remainder, the state and the first derivative over the box lie in

```math
Y(X) = c + Qq + V_0\,x + \tfrac12\,W(X)[x, x], \qquad V(X) = V_0 + W(X)[\,\cdot\,, x] ,
```

which replace the C¹ state set c + Qq + V(X)·x (A_M.md §3). One step with flow map φ: the centre set and V₀ are
propagated with the step Jacobian J₀ enclosed over the centre set alone (the C¹ point run), and

```math
W_{jk} \;\mapsto\; D\varphi(y)\,W_{jk} + D^2\varphi(y)[V_j, V_k] \;\in\; J_X\,W(X)_{jk} + H_{jk},
```

with J_X the step Jacobian enclosed over Y(X) and H an enclosure of D²φ(y)[a, b] over y ∈ Y(X), a ∈ V(X)_j, b ∈ V(X)_k.
H is the second variational equation Ψ′ = A Ψ + B[Φa, Φb], Ψ(0) = 0 (B the second derivative of the right-hand side,
`secondvar`), integrated by its Taylor coefficients (the recursion Ψ_{l+1} = (Σ A_{l−i}Ψ_i + β_l)/(l + 1) on the
trajectory series over Y(X)) plus the Lagrange-type remainder h^{NJ+1}Ψ^{(NJ+1)}(s)/(NJ+1)!, enclosed by the same
recursion at the interval time t₀ + [0, h] from a-priori enclosures of the trajectory, of Φ and of Ψ (`apriori_Psi`:
[0, h](F_y Ψ̃ + B) ⊂ Ψ̃, the Picard argument of validated_int.apriori_V). The integral form of the remainder (the mean
of a function with values in a box lies in the box) makes this valid for complex values, as for J in A_M.md §3.

**The Jacobian over the box in mean value form.** At ξ₁, with ℓ the real jump functionals,

```math
J(X)_{ij} = (\ell V_0)_{ij} + \sum_k (\ell W(X))_{i,jk}\, X_k \;\ni\; D\tilde g(x)_{ij} \quad (x \in X),
```

because Dg̃(x) = Dg̃(0) + ∫₀¹ D²g̃(sx)[·, x] ds. The Krawczyk operator with Y = (mid ℓV₀)⁻¹ (exact) is

```math
K(X) = -Y\tilde g(0) + \big(I - Y J(X)\big) X, \qquad
\big(I - YJ(X)\big)_{ij} = \delta_{ij} - (Y\ell V_0)_{ij} - \sum_k (Y\ell W(X))_{i,jk}\,X_k ,
```

evaluated with Yℓ applied to the Lohner sets before any interval product (A_M.md §2). If K(X) ⊂ int X, g̃ has
exactly one zero in X and every matrix in J(X) is non-singular (Krawczyk 1969; Moore 1977; Neumaier 1990): the
interval matrix J(X) contains Dg̃ on the convex box X, so it contains every mean-value matrix ∫₀¹ Dg̃(y + s(x − y)) ds,
which is what the theorem needs.

**Why the inner widths become third order.** The radius of I − Y J(X) is now |YℓW(X)|·|X|: the size of the second
derivative times the box, not the C¹ excess. Its contribution to K is second order, Σ_{jk}|YℓW|_{i,jk} w_j w_k. In the
inner rows the outer-outer part (YℓW)_{H,oo} has midpoint (Y D²g̃)_{H,oo}(0), which the chart makes 0 up to the
accuracy of Γ, and radius of first order in |X| (third derivatives and the C² excess), so its contribution to K_H is
O(h³) instead of the first-order chart's Γh². What remains in K_H at second order are the mixed and inner-inner terms
2|YℓW|_{H,oH} h w and |YℓW|_{H,HH} w², which are below w once h·|Y D²g̃|_{H,oH} and w·|Y D²g̃|_{H,HH} are small.

**An alternative weighed: a Poincaré–Miranda test.** Existence alone would follow if each component of Y g̃ had opposite
constant signs on the two opposite faces of X (Poincaré–Miranda), with no uniqueness. It does not avoid the derivative
growth. On the face x_i = ±w_i the sign of (Y g̃)_i needs ±w_i to dominate everything else in it, Σ_{j≠i} |(Y Dg̃ −
I)_{ij}| w_j plus the second-order terms, which is the Krawczyk condition for row i with the diagonal entering only
through (1 ± its radius) w_i. To evaluate g̃ on a face one needs either a mean value form on the face (the same C¹ or C²
columns over an 11-dimensional face, so the same excess per unit width) or a C⁰ enclosure of the state over the face,
which loses the cancellation in the jumps (A_certify.md §9.4 item 3: at M = 1 a face of half-width 3×10⁻⁶ already
failed). On T33's box it would not have passed either: every failing contribution in §1 is off-diagonal (j ≠ i; the
outer rows fail through the harmonic-3 columns, the harmonic-5 rows through the harmonic-3 and outer columns), so the
face test fails by the same margins. The C² route attacks the excess itself, and keeps uniqueness and non-singularity.

**Start data.** The start-value lemma gives y⁺(ξ₀; p) = Y_N(p) + η(p) with Y_N the series truncated at the lemma's N
and |η_k| ≤ e_k on the complex box p̂ + c(1 + i), c = 10⁻³ (`fixed_point_M.start_data_M`, A_lemmas_M.md §2). The
second derivatives along the chart's columns d_k(x) = ∂p/∂x_k over X are

- D²Y_N[d_j, d_k]: exact jets (acb_series of length 3) of the series recursion over the hull of p(X), with the
  directions as balls; the mixed pairs by polarization, D²[a, b] = c₂(a + b) − c₂(a) − c₂(b);
- D²η[d_j, d_k]: by Cauchy's estimate on the bidisc |z₁|, |z₂| < (c − hw)/(2ν) (hw the half-width of the hull of p(X)
  about p̂, ν the largest max-norm of the directions), which lies in the complex box: |D²η_k| ≤ e_k(2ν/(c − hw))²;
  e_k is read off the lemma's derivative balls at p̂ (start_data_M adds a ball of radius e_k/c to each derivative);
- the chart term Dy⁺(ξ₀; p(x))·∂²p/∂x_j∂x_k = Dy⁺·(0, Γ_jk) for j, k outer, with the lemma's derivatives over the hull.

As a consistency check the run refuses to start unless this file's Y_N(p̂) and DY_N(p̂)T₀ lie in the lemma's balls at
p̂ (they differ by η only if the truncation is the same). M = 1 uses `fixed_point_M.start_data_M` with M = 1 and the
plain splitting, which is fixed_point.py exactly (`fixed_point_M.py --check-m1`); M ≥ 3 the diag splitting, as A_M.md.

## 3. Scale estimates

Numerics, not rigorous: central differences of the C¹ numerics (`certify_M2_fd.py`, 256 bits, steps ±δ and ±2δ along
the columns of T, Richardson; δ = 10⁻⁸ at M = 1; 10⁻⁹ outer and 10⁻¹⁵ inner at M = 3; 10⁻¹⁰ outer, 10⁻¹⁹ harmonic 3 and
10⁻²² harmonic 5 at M = 5), differenced in 768-bit arithmetic (in double precision the rounding of Dg̃, whose entries
reach 10¹⁴, would swamp the inner differences). `python3 certify_M2_fd.py structure M ...`
(`logs/A_M2_structure_M{M}.log`) gives the largest |Y D²g̃|_{i,jk} at 0 per block (rows: outer, harmonic 3,
harmonic 5; pairs of coordinates), Y = Dg̃(0)⁻¹. If the enclosures were exact, an outer half-width h needs
|Y D²g̃|_{o,oo} h < 1 and 2|Y D²g̃|_{z,oz} h < 1, and the inner widths need 2|Y D²g̃|_{o,oz} w < 1 and
|Y D²g̃|_{z,zz} w < 1 and the cross conditions between w₃ and w₅.

The tables and the limits they imply are in §6 with the runs. What these estimates cannot give is the excess of the C²
enclosure (the radius of W(X) beyond the true third-order variation); it is measured by the runs.

## 4. What the code proves, and what it assumes

**The claim** (kept separate from the theorem it serves). Given enclosures of y⁺(ξ₀; p) and of its first and second
parameter derivatives (from the start-value lemma, as in §2), the code proves:

> g has exactly one zero in the curved parallelepiped p(X) = {p̂ + Tx + ½(0, Γ[x_o, x_o]) : x ∈ X}, and Dg is
> non-singular there,

for an explicit p̂, T, Γ and X. Nothing else: not lemma 6a or 6b, not the untruncated problem.

**Assumed facts.**

- Braaksma's 1-summability theorem (existence of y⁺ and its asymptotics on its sector), as in A_M.md §1.
- The M-harmonic start-value lemma `fixed_point_M.start_data_M` (A_lemmas_M.md §2; SHA-256 prefix recorded in every
  run), including its bound on the complex Cauchy box, from which the second-derivative bound of §2 is derived.
- The correctness of the code: python-flint 0.9 (arb/acb); validated_int.py's Lohner step and certify_M_lohner.py
  (A_M.md §3, reviewed with the M = 3 certificate, A_M_review.md); and the new files certify_M2_lohner.py and
  certify_M2.py (not reviewed).

## 5. Controls, written down before the runs

Every new check has a runnable mutation and the verdict it must give.

| check | control | how to run | expected |
|---|---|---|---|
| C² self-test (secondvar vs central differences of lin at M = 1, 3, 5; one step's H series vs central differences of the J series; one C² step's W vs central differences of two C¹ steps) | none | `python3 certify_M2_lohner.py selftest` | PASS |
| | one sign in secondvar flipped | `--mutate=lin2` | FAIL |
| D2: the rigorous Y ℓ W(X) at ξ₁ contains Y D²g̃(0) from the independent finite differences (tolerance: radius + 3× the Richardson error + 10⁻¹⁰ relative) | none | `python3 certify_M2.py d2check --box=... --fd=...` | pass |
| | a run with secondvar's sign flipped | `box ... --mutate=lin2`, then d2check | fail |
| | a run with the start values' second derivatives dropped | `box ... --mutate=w0`, then d2check | fail |
| | a run with the chart term of the start data dropped (M ≥ 3) | `box ... --mutate=chart`, then d2check | fail |
| Krawczyk test | C0–C8 and C6b of A_M.md §5, on the C² test | `python3 certify_M2.py controls --box=...` | as A_M.md §5 |
| independent assembly (mpmath) | none | `python3 certify_M_recheck.py data/A_M2_box_....json` | the same verdict, enclosure overlapping |
| | jump sign flipped | `--mutate` | fail |
| planted truth M = 1 | Theorem A(M = 1)'s zero; box at least A_M.md §6.1's (h = 10⁻⁹) | `krawczyk --box=data/A_M2_box_M1_*.json` | pass; enclosure overlaps Theorem A(M = 1)'s |
| planted truth M = 3 | the certified M = 3 zero; box at least A_M.md §6.2's (h = 10⁻¹⁰, w = 4×10⁻¹⁶) | `krawczyk --box=data/A_M2_box_M3_*.json --hier` | pass; enclosure overlaps A_M.md §6.2's |

A control that fires on the clean run stops the work until its cause is known; no threshold is loosened.

## 6. Results

All runs on the Mac (Apple silicon, 10 cores, at most 4 runs at once under `nice -n 10`), orders N = NJ = 24 at
M = 1 and N = 20, NJ = 12 at M ≥ 3 (A_M.md §3), 256-bit integration, 768-bit assembly. Margins are relative,
(w_i − |K_i|)/w_i. The contributions are the terms of |K_i|/w_i (`krawczyk` prints them): |Y g̃(0)|, the first-order
part |I − YℓV₀|·X, and the second-order part Σ|YℓW(X)|_{i,jk} w_j w_k.

### 6.1 M = 1: Theorem A(M = 1) reproduced (planted truth)

| run | box | steps (ray, arc, real) | time | the test on the run's box | margins |
|---|---|---|---|---|---|
| A_M.md §6.1 (C¹, for comparison) | h = 10⁻⁹ | 303, 320, 46 | | passes | 0.863, 0.976 |
| `data/A_M2_box_M1_1e-6.json` | h = 10⁻⁶ | 303, 320, 46 | 104 s | passes | 1.000, 1.000 |
| `data/A_M2_box_M1_1e-4.json` | h = 10⁻⁴ | 303, 321, 46 | 98 s | fails (−3.69, −2.17) | |
| the same, sub-box (`--best`) | h′ = 2.06×10⁻⁵ | | | passes | 0.034, 0.348 |

(`logs/A_M2_krawczyk_M1_1e-6.log`, `logs/A_M2_krawczyk_M1_1e-4_best.log`.) At h = 10⁻⁶ the contributions are at most
1.2×10⁻⁴ (|Y g̃(0)|: the centre is Theorem A's 10-digit point), 1.5×10⁻⁶ (first order) and 3.8×10⁻⁵ (second order).
At h = 10⁻⁴ the second-order part reaches 2.1: the radius of YℓW at ξ₁ grows faster than the box (about 19 at
h = 10⁻⁶ and 1.4×10⁴ at h = 10⁻⁴, against true second derivatives of at most 18.5), which is the C² excess. The largest
passing box, 2.06×10⁻⁵, is 2×10⁴ times the C¹ one.

The zero, from the smallest passing box of the h = 10⁻⁶ run (θ = 1.29×10⁻⁴, endpoints rounded outward):
A ∈ [2.54214804761655, 2.54214804761657], ω/ω∞ ∈ [0.776994013609722, 0.776994013609726]. It lies inside Theorem
A(M = 1)'s enclosure in both coordinates (`--reference=thmA1`: REPRODUCED); with the reference shifted by ten of its
widths in A (`--mutate=ref`, `logs/A_M2_krawczyk_M1_1e-6_mutate_ref.log`) the check reports NOT REPRODUCED and exits 1.

### 6.2 M = 3: the certified zero reproduced (planted truth)

| run | box (h; w) | steps | time | the test | margins |
|---|---|---|---|---|---|
| A_M.md §6.2 (C¹, for comparison) | 10⁻¹⁰; 4×10⁻¹⁶ | 955, 552, 72 | | passes | 0.994, 0.997, 0.778, 0.778 |
| `data/A_M2_box_M3_1e-8.json` | 10⁻⁸; 10⁻¹³ | 955, 552, 72 | 414 s | passes | 1.000 (all four) |
| `data/A_M2_box_M3_1e-6.json` | 10⁻⁶; 10⁻¹¹ | 955, 551, 72 | 421 s | passes | 0.997, 0.998, 0.995, 0.995 |

(`logs/A_M2_krawczyk_M3_1e-8.log`, `logs/A_M2_krawczyk_M3_1e-6.log`.) The chart's Γ is in `data/A_M2_chart_M3.json`.
At h = 10⁻⁸ every contribution is at most 2.8×10⁻⁶ of its width, at h = 10⁻⁶ at most 2.3×10⁻³ (the largest in the
inner rows from the (outer, outer) pairs, whose midpoint the chart has removed: what is left is the radius, the third
order term of §2): the boxes are 10² and 10⁴ times the C¹ one in the outer
directions and 2.5×10² and 2.5×10⁴ times in the inner ones. The zero, from the smallest passing box of the h = 10⁻⁸ run
(θ = 1.05×10⁻⁴):
A ∈ [2.72647588563088, 2.72647588563089], ω/ω∞ ∈ [1.01560309485773, 1.01560309485774],
B₃ ∈ [0.144419478470076, 0.144419478470077], C₃ ∈ [0.0110874258881167, 0.0110874258881168], inside the certified
enclosure of A_M.md §6.2 in every coordinate (`--reference=AM3`: REPRODUCED).

The D2 check (`logs/A_M2_d2check_M3_1e-8.log`) passes: the largest entries of Y D²g̃(0), up to 3.9×10¹¹ (the outer rows
in the harmonic-3 pairs), agree with the finite differences to the enclosure's radius 4.7×10⁷; the worst deviation
is 3.5×10⁻¹⁰ of its tolerance (6.3×10⁻¹¹ on the h = 10⁻⁶ run).

### 6.3 Controls at M = 1 and M = 3

| check | run | result | expected |
|---|---|---|---|
| C² self-test | | PASS | PASS |
| | `--mutate=lin2` | FAIL (secondvar vs differences 0.25–3.3 relative; one step's W not contained) | FAIL |
| D2 | M = 1, h = 10⁻⁹, clean (`logs/A_M2_d2check_M1_1e-9_none.log`) | pass (worst ratio 4.8×10⁻¹²) | pass |
| | M = 1, h = 10⁻⁹, secondvar's sign flipped (`_lin2.log`) | fail (worst ratio 17, exit 1) | fail |
| | M = 1, h = 10⁻⁹, start second derivatives dropped (`_w0.log`) | fail (worst ratio 4.5, exit 1) | fail |
| | M = 3, h = 10⁻⁸, clean (`logs/A_M2_d2check_M3_1e-8.log`) | pass (worst ratio 3.5×10⁻¹⁰) | pass |
| | M = 3, h = 10⁻⁸, secondvar's sign flipped (`logs/A_M2_d2check_M3_1e-8_mut_lin2.log`) | fail (worst ratio 2.7×10⁴, exit 1) | fail |
| | M = 3, h = 10⁻⁸, the chart term of the start data dropped (`_mut_chart.log`) | fail (worst ratio 1.4×10³, exit 1) | fail |
| Krawczyk C0–C8, C6b | M = 1, h = 10⁻⁶ (`logs/A_M2_controls_M1_1e-6.log`) | all as expected (no C8 at M = 1) | as A_M.md §5 |
| | M = 3, h = 10⁻⁸ (`logs/A_M2_controls_M3_1e-8.log`) | all as expected | as A_M.md §5 |
| recheck (mpmath) | M = 1, h = 10⁻⁶ (`logs/A_M2_recheck_M1_1e-6.log`) | pass, margins 1.000, 1.000; A ∈ [2.5421480475422, 2.5421480476910] | the same verdict |
| | the same, jump sign flipped (`_mutate.log`) | fail (exit 1) | fail |
| | M = 3, h = 10⁻⁸ (`logs/A_M2_recheck_M3_1e-8.log`) | pass, margins 1.000 (all four); A ∈ [2.7264758856308, 2.7264758856310] | the same verdict |
| | the same, jump sign flipped (`_mutate.log`) | fail (exit 1) | fail |

**One finding about the D2 control.** Run on the h = 10⁻⁶ box at M = 1, the two mutated runs did *not* fail the D2
check (`logs/A_M2_d2check_M1_1e-6_mut_lin2.log`, `_w0.log`): there the radius of YℓW(X) (about 20) is as large as the
second derivatives themselves (at most 18.5), so a containment test cannot see an error of order one. D2 is a test
only where the radius is small against the entries; at M = 1 that needs a small box (h = 10⁻⁹: radius 0.5, where both
mutations fire), at M = 3 the h = 10⁻⁸ box already has a relative radius of 10⁻⁴ in the largest entries. The table
reports the controls on those boxes; the h = 10⁻⁶ results are kept in the logs.

### 6.4 M = 5

**The chart** (`data/A_M2_chart_M5.json`): p̂ and C from `data/A_M_refine_M5_d.json` (the centre of A_M.md §6.3), Γ
from the finite differences. In the second-order chart the largest |Y D²g̃|_{H,oo} drops from 0.66 (harmonic-3 rows)
and 0.13 (harmonic-5 rows) to 3.6×10⁻¹⁰ and 3.1×10⁻¹¹ (`logs/A_M2_structure_M5.log`).

**What the structure allows** (§3, numerics; `logs/A_M2_structure_M5.log`). The blocks of |Y D²g̃| reach 1.9×10²²,
2.3×10²³ and 3.0×10²⁴ in the outer rows for the (z3, z3), (z3, z5) and (z5, z5) pairs, 6.1×10¹⁵, 8.2×10¹⁶ and
1.1×10¹⁸ in the harmonic-3 rows, 4.5×10¹⁴, 6.1×10¹⁵ and 8.2×10¹⁶ in the harmonic-5 rows, and 6.0×10⁵ in the inner
rows for the (outer, same harmonic) pairs. With exact enclosures this caps the inner widths near w₃ ≈ 10⁻¹⁷ and
w₅ ≈ 10⁻¹⁸ (from the inner-inner pairs), requires w₅ ≳ 4.5×10¹⁴ w₃² and w₅ ≳ 9×10⁴ h w₃ (the harmonic-5 rows), and
caps the outer half-width at a few 10⁻⁷ (the inner rows' (outer, inner) pairs: 2·6.0×10⁵ h < 1, and
2·8.2×10⁶ h w₅/w₃ < 1). The pilot
(`logs/A_M2_pilot_M5_1e-9.log`, 150 steps at h = 10⁻⁹) took 0.8 s per step; three full runs followed.

| run | box (h; w₃, w₅) | steps | time | the test | margins |
|---|---|---|---|---|---|
| T33 (C¹, A_M.md §6.3, for comparison) | 10⁻¹³; 10⁻¹⁹, 10⁻²³ | 2118, 863, 101 | | fails | −13.7, −11.8, +0.29, +0.33, −542, −511 |
| `data/A_M2_box_M5_1e-11.json` | 10⁻¹¹; 10⁻¹⁹, 10⁻²⁰ | 2118, 863, 101 | 38 min | passes | 1.000, 1.000, 0.989, 0.989, 0.991, 0.991 |
| `data/A_M2_box_M5_1e-10.json` | 10⁻¹⁰; 10⁻¹⁸, 10⁻¹⁹ | 2118, 863, 101 | 38 min | passes | 0.996, 0.998, 0.889, 0.889, 0.912, 0.911 |

(`logs/A_M2_krawczyk_M5_1e-11.log`, `logs/A_M2_krawczyk_M5_1e-10.log`; the h = 10⁻⁹ run is in §6.5.) On the
h = 10⁻¹⁰ box the contributions are at most 6.8×10⁻¹⁰ (|Y g̃(0)|), 2.6×10⁻⁷ (first order) and 2.2×10⁻² per pair
(second order); the largest are the inner-inner pairs of the inner rows, the intrinsic terms that set the inner
widths, while the (outer, outer) pairs that the chart removes contribute at most 2.9×10⁻⁶ there. In the outer rows the
largest pair term is 6.1×10⁻⁴.

> **Certified** (given the start-value lemma `fixed_point_M.start_data_M`, SHA-256 prefix 589e5a810bc50289, and
> Braaksma's theorem; trusted code certify_M2_lohner.py, certify_M2.py, certify_M_lohner.py, certify_M_sys.py,
> validated_int.py and python-flint 0.9; certify_M2*.py not reviewed): the D-jump function g of A_M.md §1 at M = 5 has
> exactly one zero in the curved parallelepiped p(X), X = [−10⁻¹⁰, 10⁻¹⁰]² × [−10⁻¹⁸, 10⁻¹⁸]² × [−10⁻¹⁹, 10⁻¹⁹]²,
> and its Jacobian is non-singular there. The zero lies in (endpoints rounded outward; the smallest passing box,
> θ = 1.05×10⁻⁴, intersected with the whole box's enclosure):
>
> A ∈ [2.71857918921174, 2.71857918921175], ω/ω∞ ∈ [0.999228403206567, 0.999228403206568],
> B₃ ∈ [0.149203574911447, 0.149203574911448], C₃ ∈ [0.0147190524902513, 0.0147190524902514],
> B₅ ∈ [0.00840534690576166, 0.00840534690576167], C₅ ∈ [0.00147034518159157, 0.00147034518159158].

The axis-aligned hull of p(X) has half-widths 1.0×10⁻¹⁰, 1.0×10⁻¹⁰, 4.1×10⁻¹¹, 4.6×10⁻¹², 4.4×10⁻¹², 8.2×10⁻¹³ (the
krawczyk log). The refined numerical centre lies inside the enclosure.

**Controls on the h = 10⁻¹⁰ run.** C0–C8 and C6b of A_M.md §5 all give their expected verdicts
(`logs/A_M2_controls_M5_1e-10.log`, exit 0): C1, C2 fail by relative margins of order −10⁹ in every row, C5 by −1.00
in A, C7 by −10³, C8 by −1.11 in B₃; the null mutations C3, C4 pass with the clean margins and overlapping
enclosures; the planted interior zeros C6 and C6b pass (0.50–0.70 and 0.44–0.50). The independent assembly
(`logs/A_M2_recheck_M5_1e-10.log`, mpmath) gives the same verdict and margins and the zero in
A ∈ [2.7185791892113759, 2.7185791892121089]; with the jump sign flipped (`_mutate.log`) it fails (exit 1).
**hier.py's zeros** (same krawczyk log): both solves stored in certify_M.py (`HIER[5]`) lie in the hull of p(X), and
agree with the certified zero within 3× their spread in every coordinate (|difference| ≤ 1.81×10⁻¹¹ in A against a
spread of 1.7×10⁻¹¹); with them moved by 100× their spread (`--mutate=hier`, `logs/A_M2_hier_M5_1e-10_mutate.log`)
the check fails (exit 1). On the h = 10⁻¹¹ box statement (a) fails for one solve, as A_M.md §6.3 predicted for boxes
below 2×10⁻¹¹; (b) holds. **D2** passes on both runs (`logs/A_M2_d2check_M5_1e-10.log`, `_1e-11.log`; worst ratio
2.6×10⁻⁷ and 1.0×10⁻⁶); there the largest entries, about 3×10²⁴, carry a relative radius of 5×10⁻³, so the check is
sharp in them.

**For lemma 6a(M).** The state box at ξ₁ over p(X) is in `data/A_M2_6a_M5_1e-10.json`
(`python3 certify_M2.py export6a --box=data/A_M2_box_M5_1e-10.json --out=...`; the format of certify_M.py's export,
with V the x-derivatives in the second-order chart), max radius 4.3×10⁻⁴; a sub-box (`--sub=`) gives a smaller one.

### 6.5 M = 5 on a ten times larger outer box

The run at h = 10⁻⁹, w = (10⁻¹⁷, 10⁻¹⁸) (`data/A_M2_box_M5_1e-9.json`, 2118, 863, 101 steps, 37 min) fails on its own
box in the harmonic-3 rows (margins 0.961, 0.981, −0.119, −0.121, 0.109, 0.107; `logs/A_M2_krawczyk_M5_1e-9.log`).
The failing terms are the inner-inner pairs, ten times those of the h = 10⁻¹⁰ run because the inner widths are ten
times larger: the intrinsic limit of §6.4, not the enclosure. On its sub-box h = 10⁻⁹, w = (10⁻¹⁷, 5×10⁻¹⁹) (found by
`krawczyk --search`, a grid of sub-box factors; `logs/A_M2_krawczyk_M5_1e-9_search.log`) the test passes, with
margins 0.980, 0.990, 0.414, 0.413, 0.068, 0.067, hier.py's two zeros inside the hull and within 3× their spread,
and the independent assembly agrees (`logs/A_M2_recheck_M5_1e-9_sub.log`, same margins). Of the controls
(`logs/A_M2_controls_M5_1e-9_sub.log`) all give their expected verdict except C6: the zero planted at a quarter of the
inner widths falls outside the operator's image (margins −0.18 in the harmonic-5 rows) because the clean margins there
are only 0.067. This is C6's known limitation (A_M.md §6.2 records the same at M = 3); C6b, planted at half the clean
margins, passes. It is recorded, not changed; the h = 10⁻¹⁰ certificate of §6.4, where every control behaves, is the
one this note states. D2 passes on this run as well (`logs/A_M2_d2check_M5_1e-9.log`, worst ratio 9.9×10⁻⁹).

**Why D2 is needed.** The Krawczyk test alone does not see a wrong second derivative when the second-order terms are
small: on the run with secondvar's sign flipped, the M = 3 test still passes with margins 1.000
(`logs/A_M2_krawczyk_M3_1e-8_mut_lin2.log`), although the enclosed Y D²g̃ is off by a factor of about two. A wrong
enclosure can make the test pass; D2 is what rules that out on the runs where it is sharp.

## 7. Reproduce

All commands from `solver/proof/`, with python-flint 0.9.0, mpmath and numpy; `fixed_point_M.py` sits next to these
files. Times are on the Mac with four runs at once.

```
python3 certify_M2_lohner.py selftest                  # PASS (secondvar, H series, one C2 step); seconds
python3 certify_M2_lohner.py selftest --mutate=lin2    # must FAIL
# finite differences of the C1 numerics (independent of the C2 code; 2-3 min per point at M = 5), then combine:
python3 certify_M2_fd.py run 5 --centre=data/A_M_refine_M5_d.json --use_out --k=K --s=S --delta=D --out=PTS/M5_kK_sS.json
#   for K = 0..5, S = +1, -1, +2, -2, D = 1e-10 (K = 0, 1), 1e-19 (K = 2, 3), 1e-22 (K = 4, 5); M = 3: centre
#   data/A_M_refine_M3_c.json, D = 1e-9, 1e-15; M = 1: centre data/A_M_point_M1.json (no --use_out), D = 1e-8
python3 certify_M2_fd.py combine 5 --centre=data/A_M_refine_M5_d.json --use_out --glob=PTS/M5_ --out=data/A_M2_fd_M5.json
python3 certify_M2_fd.py structure 5 --centre=data/A_M_refine_M5_d.json --fd=data/A_M2_fd_M5.json [--chart=data/A_M2_chart_M5.json]
python3 certify_M2.py chart 5 --centre=data/A_M_refine_M5_d.json --use_out --fd=data/A_M2_fd_M5.json --out=data/A_M2_chart_M5.json
python3 certify_M2.py diag1 --box=data/A_M_box_M5_1e-13.json --fd=data/A_M2_fd_M5.json     # section 1
# M = 1 (about 100 s per run)
python3 certify_M2.py box 1 --chart=data/A_M2_chart_M1.json --h=1e-6 --w=0 --out=data/A_M2_box_M1_1e-6.json --log=logs/A_M2_box_M1_1e-6.log
python3 certify_M2.py krawczyk --box=data/A_M2_box_M1_1e-6.json --shrink --reference=thmA1                # REPRODUCED
python3 certify_M2.py krawczyk --box=data/A_M2_box_M1_1e-6.json --shrink --reference=thmA1 --mutate=ref   # must not
python3 certify_M2.py controls --box=data/A_M2_box_M1_1e-6.json                                           # exit 0
python3 certify_M_recheck.py data/A_M2_box_M1_1e-6.json [--mutate]                                        # pass [fail]
python3 certify_M2.py box 1 --chart=data/A_M2_chart_M1.json --h=1e-4 --w=0 --out=data/A_M2_box_M1_1e-4.json --log=logs/A_M2_box_M1_1e-4.log
python3 certify_M2.py krawczyk --box=data/A_M2_box_M1_1e-4.json --best                                    # 2.06e-5
python3 certify_M2.py box 1 --chart=data/A_M2_chart_M1.json --h=1e-9 --w=0 --mutate=none|lin2|w0 --out=data/A_M2_box_M1_1e-9_d2_T.json --log=logs/A_M2_box_M1_1e-9_d2_T.log
python3 certify_M2.py d2check --box=data/A_M2_box_M1_1e-9_d2_T.json --fd=data/A_M2_fd_M1.json            # none: pass; lin2, w0: fail
# M = 3 (about 7 min per run)
python3 certify_M2.py box 3 --chart=data/A_M2_chart_M3.json --h=1e-8 --w=1e-13 --out=data/A_M2_box_M3_1e-8.json --log=logs/A_M2_box_M3_1e-8.log
python3 certify_M2.py krawczyk --box=data/A_M2_box_M3_1e-8.json --shrink --reference=AM3                  # REPRODUCED
python3 certify_M2.py controls --box=data/A_M2_box_M3_1e-8.json                                           # exit 0
python3 certify_M2.py d2check --box=data/A_M2_box_M3_1e-8.json --fd=data/A_M2_fd_M3.json                  # pass
python3 certify_M_recheck.py data/A_M2_box_M3_1e-8.json [--mutate]                                        # pass [fail]
python3 certify_M2.py box 3 --chart=data/A_M2_chart_M3.json --h=1e-8 --w=1e-13 --mutate=lin2|chart --out=data/A_M2_box_M3_1e-8_mut_T.json --log=logs/A_M2_box_M3_1e-8_mut_T.log
python3 certify_M2.py d2check --box=data/A_M2_box_M3_1e-8_mut_T.json --fd=data/A_M2_fd_M3.json            # must fail
# M = 5 (about 40 min per run, 78 s of it the start data); a pilot first:
python3 certify_M2.py box 5 --chart=data/A_M2_chart_M5.json --h=1e-9 --w=1e-17,1e-18 --out=PILOT.json --log=logs/A_M2_pilot_M5_1e-9.log --pilot=150
python3 certify_M2.py box 5 --chart=data/A_M2_chart_M5.json --h=1e-10 --w=1e-18,1e-19 --out=data/A_M2_box_M5_1e-10.json --log=logs/A_M2_box_M5_1e-10.log
python3 certify_M2.py krawczyk --box=data/A_M2_box_M5_1e-10.json --shrink --hier                         # passes; hier (a), (b)
python3 certify_M2.py krawczyk --box=data/A_M2_box_M5_1e-10.json --hier --mutate=hier                    # must fail
python3 certify_M2.py controls --box=data/A_M2_box_M5_1e-10.json                                         # exit 0
python3 certify_M2.py d2check --box=data/A_M2_box_M5_1e-10.json --fd=data/A_M2_fd_M5.json                 # pass
python3 certify_M_recheck.py data/A_M2_box_M5_1e-10.json [--mutate]                                       # pass [fail]
python3 certify_M2.py export6a --box=data/A_M2_box_M5_1e-10.json --out=data/A_M2_6a_M5_1e-10.json        # for lemma 6a(M)
```

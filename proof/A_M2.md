# Layer A beyond one harmonic, second order: the C² point certificate

This note is the second-order version of [A_M.md](A_M.md): the same function g (the D-jumps at ξ₁ = 0.3 on the M + 1
real conditions, A_M.md §1), the same path and start-value lemma, and the same Krawczyk theorem, with a second-order
chart and second-order (C²) enclosures of the flow, so that the Jacobian over the box is enclosed in mean value form.
It is the design change A_M.md §2 and §7 name after the first-order M = 5 runs failed. Written by Claude (Anthropic)
under Nathan Moinvaziri's direction; not reviewed.

**Status.** See §6 (results), filled in as the runs finish.

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

That radius is the C¹ enclosure's, and it is mostly excess: the true variation of Y Dg̃ over the box (from the
second derivatives of §3) is smaller by the factors in the same log. The C¹ excess is linear in the box (the
derivative radii in the four T33 logs are proportional to h at equal steps), as at M = 1 (A_certify.md §9.3–9.4).

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

with J_X the step Jacobian enclosed over Y(X) and H an enclosure of D²φ(y)[a, b] over y ∈ Y(X), a ∈ V(X)_j,
b ∈ V(X)_k. H is the second variational equation Ψ′ = A Ψ + B[Φa, Φb], Ψ(0) = 0 (B the second derivative of the
right-hand side, `secondvar`), integrated by its Taylor coefficients (the recursion Ψ_{l+1} = (Σ A_{l−i}Ψ_i + β_l)/(l + 1)
on the trajectory series over Y(X)) plus the Lagrange-type remainder h^{NJ+1}Ψ^{(NJ+1)}(s)/(NJ+1)!, enclosed by the
same recursion at the interval time t₀ + [0, h] from a-priori enclosures of the trajectory, of Φ and of Ψ
(`apriori_Psi`: [0, h](F_y Ψ̃ + B) ⊂ Ψ̃, the Picard argument of validated_int.apriori_V). The integral form of the
remainder (the mean of a function with values in a box lies in the box) makes this valid for complex values, as for J
in A_M.md §3.

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

**An alternative weighed: a Poincaré–Miranda test.** Existence alone would follow if each component of Y g̃ had
opposite constant signs on the two opposite faces of X (Poincaré–Miranda), with no uniqueness. It does not avoid the
derivative growth. On the face x_i = ±w_i the sign of (Y g̃)_i needs ±w_i to dominate everything else in it,
Σ_{j≠i} |(Y Dg̃ − I)_{ij}| w_j plus the second-order terms, which is the Krawczyk condition for row i with the
diagonal entering only through (1 ± its radius) w_i. To evaluate g̃ on a face one needs either a mean value form on the face (the same C¹ or C² columns over
an 11-dimensional face, so the same excess per unit width) or a C⁰ enclosure of the state over the face, which loses
the cancellation in the jumps (A_certify.md §9.4 item 3: at M = 1 a face of half-width 3×10⁻⁶ already failed). On
T33's box it would not have passed either: every failing contribution in §1 is off-diagonal (j ≠ i; the outer rows
fail through the harmonic-3 columns, the harmonic-5 rows through the harmonic-3 and outer columns), so the face test
fails by the same margins. The C² route attacks the excess itself, and keeps uniqueness and non-singularity.

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

(`logs/A_M2_krawczyk_M3_1e-8.log`.) The chart's Γ is in `data/A_M2_chart_M3.json`. Every contribution is at most
2.8×10⁻⁶ of its width: the box could be much larger (§6.4). The zero, from the smallest passing box (θ = 1.05×10⁻⁴):
A ∈ [2.72647588563088, 2.72647588563089], ω/ω∞ ∈ [1.01560309485773, 1.01560309485774],
B₃ ∈ [0.144419478470076, 0.144419478470077], C₃ ∈ [0.0110874258881167, 0.0110874258881168], inside the certified
enclosure of A_M.md §6.2 in every coordinate (`--reference=AM3`: REPRODUCED).

The D2 check (`logs/A_M2_d2check_M3_1e-8.log`) passes: the largest entries of Y D²g̃(0), up to 3.9×10¹¹ (the outer rows
in the harmonic-3 pairs), agree with the finite differences to the enclosure's radius 4.7×10⁷; the worst deviation
is 3.5×10⁻¹⁰ of its tolerance.

## 7. Reproduce

(To be filled in by the runs.)

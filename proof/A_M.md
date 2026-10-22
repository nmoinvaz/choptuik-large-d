# Layer A beyond one harmonic: the point certificate for the M = 3 and M = 5 inner zeros

This note is the analogue of [A_certify.md](A_certify.md) §9–10 for the truncations with more than one harmonic:
validated integration plus one Krawczyk test, giving "exactly one simple zero of the jump function in an explicit
box". The analytic lemmas that turn the point certificate into the statement about the selection condition (the
analogues of lemmas S, 6a and 6b of A_certify.md §11–12) are not part of this note; the M-harmonic start-value fixed
point is being written separately (`solver/proof/fixed_point_M.py`, notes in `proof/A_lemmas_M.md`). Written by Claude
(Anthropic) under Nathan Moinvaziri's direction; not reviewed.

Code: [`solver/proof/certify_M.py`](../solver/proof/certify_M.py) (driver, the Krawczyk test, refinement, controls),
[`certify_M_lohner.py`](../solver/proof/certify_M_lohner.py) (the runs), [`certify_M_sys.py`](../solver/proof/certify_M_sys.py)
(the M-harmonic system, path legs, formal series with parameter jets, placeholder start values, self-test). Data and
logs: `solver/proof/data/A_M_*`, `solver/proof/logs/A_M_*`.

Contents

- [1. The function and what is certified](#1-the-function-and-what-is-certified)
- [2. Design: the Schur-corrected Krawczyk test in a tilted box](#2-design-the-schur-corrected-krawczyk-test-in-a-tilted-box)
- [3. The runs](#3-the-runs)
- [4. The M = 1 case as a special case](#4-the-m--1-case-as-a-special-case)

## 1. The function and what is certified

**System.** The truncated inner system of A_certify.md §2 with M harmonics, exactly as
`solver/inner_stokes/inner_np.py` and `lateral_np.py` integrate it: D, S on the odd harmonics |μ| ≤ M, C on the even
harmonics |ν| ≤ M + 1, every product projected onto its result's harmonics; n = 3M + 4 complex unknowns (7, 13, 19 for
M = 1, 3, 5). The profile F = A sin θ + Σ_{ν ≥ 3}(B_ν sin νθ + C_ν cos νθ) enters only the start values; ω = wf·ω∞ enters
the right-hand side, with ω∞ := 2π/1.64389375 exactly (the codes' convention).

**Parameters and conditions.** p = (A, wf, B₃, C₃, …, B_M, C_M) ∈ ℝ^{M+1}, and

```math
g(p) = (\mathrm{Re}\,J_1, \mathrm{Im}\,J_1, \mathrm{Re}\,J_3, \mathrm{Im}\,J_3, \dots, \mathrm{Re}\,J_M, \mathrm{Im}\,J_M), \qquad
J_\nu = D^+_\nu(\xi_1) - \overline{D^+_{-\nu}(\xi_1)} ,
```

with y⁺ continued from the start ray along the path below. At M = 1 this is the function J_D of A_certify.md §11.2.

**Path.** The lower path of Theorem A(M = 1): the ray arg ξ = φ₀ = 3π/8 from r₀ to ρ = 0.2, the arc |ξ| = 0.2 down to the
real axis, the real axis from 0.2 to ξ₁ = 0.3. r₀ = 0.1 at M = 1; at M ≥ 3, r₀ = 0.08 until the M-harmonic start-value
lemma fixes it. (hier.py evaluates at ξ₁ = 0.2 on the upper path from a ray at arg ξ = 5π/16. Same zero set if y⁺ is
analytic between the paths, which is lemma 6b's business; ξ₁ = 0.3 is used because the outer signal there is about
2×10⁴ times larger in absolute terms, as at M = 1, A_certify.md §5.1.)

**What the code proves (the claim, kept separate from the theorem it serves).** Given enclosures of y⁺ and of its
parameter derivatives at the start point (the start-value lemma), the code proves:

> g has exactly one zero in the parallelepiped P = p̂ + T·X, and Dg is non-singular there,

for an explicit centre p̂, an explicit matrix T (§2) and an explicit box X. That is all. It does not prove that the
path continuation is the Borel sum (lemma 6b), that J_ν(ξ₁) = 0 for all ν forces the whole jump to vanish (lemma 6a),
or anything about M > 5 or the untruncated problem.

**Assumed facts.**

- Braaksma's 1-summability theorem, for the existence of y⁺ and its asymptotics on its sector (as in Theorem A(M = 1)).
- The start-value lemma for the M-harmonic system: at M = 1 this is `fixed_point.start_data` (proved, A_certify.md §4.4–4.5).
  **At M ≥ 3 it does not exist yet; the runs use a non-rigorous placeholder** (the series truncated at its smallest
  term plus an assumed error ball, `certify_M_sys.placeholder_start`). Until `fixed_point_M.py` replaces it, every
  M ≥ 3 result here is numerical evidence produced by the certificate's code path, not a certificate.
- The correctness of the code: python-flint 0.9 (arb/acb), validated_int.py's Lohner step (reviewed,
  [A_review.md](A_review.md)), and the new files above (not reviewed).

## 2. Design: the Schur-corrected Krawczyk test in a tilted box

**The difficulty** (A_certify.md §7, PROOF.md A.3–A.4). In the raw jumps the higher harmonics dominate everything.
Double-precision Jacobians of hier.py's de-mixed residual (`jac_explore`, scratch, not kept) at the zeros:

| | M = 3 | M = 5 |
|---|---|---|
| ∂r_M/∂(B_M, C_M) | 5×10⁵ | 4×10⁷ |
| ∂r₁/∂(B_M, C_M) | 6×10⁹ | 6×10¹⁵ |
| ∂r₁/∂(A, wf), raw | 10⁸–10⁹ | 10¹³ |
| outer Schur complement, singular values | 8.8×10³, 4.5×10³ | not resolved in double precision |
| dx_H/dx_o (tangent of the higher harmonics' zero manifold) | 0.004–0.25 | 0.003–0.21 |

So r₁ ≈ L·r_H + (intrinsic part), with a leakage L ≈ 10⁴ (M = 3) and 1.5×10⁸ (M = 5, harmonic 5 into 1), and the
outer derivative that matters is the Schur complement, 10⁴–10¹² times smaller than the raw entries. In the raw jumps
J_ν at ξ₁ = 0.3 (M = 5) the state's sensitivity to B₅ is of order 10¹³–10¹⁴ against an outer signal of order 1.

Two things go wrong in a naive Krawczyk test on (A, wf, B₃, …, C_M) with an axis-aligned box:

1. **Interval hulls lose the cancellation.** The outer rows of the preconditioned Jacobian are S⁻¹(Dg₁ − L Dg_H).
   If Dg is first enclosed entrywise over the box, the radius of Dg₁ (the true variation of a 10¹⁵-sized entry) is not
   cancelled by L Dg_H, and the test needs boxes of about 10⁻¹⁴ at M = 5.
2. **The higher-harmonic box is first order.** An axis-aligned box must contain x_H*(x_o) over the outer box, which
   moves by |dx_H/dx_o|·h ≈ 0.2h. Through the 10¹³-sized sensitivity that makes the state set over the box, and with it
   the enclosed step Jacobians, enormously wider than the outer directions alone would.

**The design.** One Krawczyk test on the full (M + 1)-dimensional system, posed so that both are avoided:

- **Chart (tilt).** Measure the higher-harmonic unknowns from the linearized zero manifold of g_H:

```math
p = \hat p + T x, \qquad T = \begin{pmatrix} I_2 & 0 \\ C & I_{M-1} \end{pmatrix}, \qquad C = -\big(\partial_{p_H} g_H\big)^{-1}\partial_{p_o} g_H\big|_{\hat p},
```

  and test g̃(x) = g(p̂ + Tx) on the box X = [−h, h]² × Π[−w_ν, w_ν]². This is hier.py's `--predict` linear predictor
  made into coordinates. Along the outer directions g_H now changes only at second order, so the inner widths w_ν need
  only contain the curvature of the manifold, about |d²x_H/dx_o²|·h², plus the enclosure's own width.
- **Preconditioner.** Y = (mid Dg̃(0))⁻¹, computed at 768 bits and then used as an exact matrix. Since the lower-left
  block of Dg̃(0) vanishes,

```math
Y = \begin{pmatrix} S^{-1} & -S^{-1}L \\ 0 & (\partial_{p_H} g_H)^{-1} \end{pmatrix}, \qquad
S = \partial_o g_o - L\,\partial_o g_H, \quad L = \partial_H g_o\,(\partial_H g_H)^{-1} ,
```

  so the outer rows of Y g are S⁻¹(g_o − L g_H), the Schur-corrected outer condition (hier.py's correction
  r_l − K_lH K_HH⁻¹ r_H), and the inner rows are the interval Newton step for the higher-harmonic unknowns. The de-mixing
  of hier.py (multiplication by e^{−Π} and the K′ normalizations) is a fixed invertible linear map of the conditions;
  the Krawczyk test is invariant under such maps (they are absorbed into Y), so posing the test on the raw J_ν with
  this Y is posing it on the de-mixed, Schur-corrected conditions.
- **No hull before the cancellation.** The validated integration delivers the final state and the derivative columns
  ∂y/∂x_k as Lohner sets c + Qq (Q an exact float matrix, q a ball vector). Y g(p̂) and Y Dg̃(X) are evaluated as
  (Yℓ)c + ((Yℓ)Q)q, where ℓ is the real matrix of the jump functionals: the leakage cancels in the point matrix Yℓ
  before any interval product. (CAPD's check of §10 did the same for one functional, "apply ℓ before the interval
  products".)

**The test.**

```math
K(X) = -\,Y\,\tilde g(0) + \big(I - Y\,D\tilde g(X)\big)\,X \subset \mathrm{int}\,X
\;\Rightarrow\; \tilde g \text{ has exactly one zero in } X, \text{ and } D\tilde g \text{ is non-singular there}
```

(Krawczyk 1969; Neumaier, *Interval Methods for Systems of Equations*, 1990, Thm 5.1.8), hence g has exactly one zero in
P = p̂ + T X, and it is simple (T is invertible). The zero is reported in p-coordinates as p̂ + T·K(X).

**Why this is "inner interval Newton, then the outer 2×2".** The inner rows of K are the interval Newton operator for
the higher-harmonic unknowns over the whole box (they enclose x_H at the zeros of g_H for every outer point), the outer
rows are the 2×2 Krawczyk test of the Schur-corrected outer condition, and the theorem is applied once to the whole
block-triangular system, so no separate nested-existence lemma is needed. A literally nested version (first an inner
test parametrized by x_o ∈ X_o, then an outer test on G(x_o) = g_o(x_o, x_H*(x_o)) with DG enclosed as a Schur
complement) needs the same inputs and gives the same conclusion; it was not built.

**Scale estimates (double precision, not rigorous; to be replaced by the runs).** With the tilt, the outer rows lose
nothing to the leakage at first order. What limits the box:

- the inner widths must contain the manifold's curvature, w ≳ 0.1 h²;
- the state's sensitivity to the inner coordinates times w must stay below its sensitivity to the outer ones times h,
  about 10¹³·w ≲ h at M = 5 (10⁶·w ≲ h at M = 3), so h ≲ 10⁻¹¹–10⁻¹² at M = 5 and h ≲ 10⁻⁵ at M = 3 on this count;
- the excess of the derivative enclosures grows linearly in h, as at M = 1 (A_certify.md §9.3: about 7×10⁵ h), which
  limited M = 1 to 10⁻⁹ (10⁻⁸ in CAPD).

A larger box at M = 5 would need a second-order chart and second-order (C²) enclosures, so that the inner widths are
third order in h and the Jacobian is enclosed in mean value form; that is not attempted here.

**The centre.** The box must be centred within a fraction of its widths of the zero, in every coordinate:
|x_o − x_o*| ≪ h and the inner coordinates to ≪ w. hier.py's zeros are good to about 2×10⁻¹¹ (the spread between two
solves). The centre is refined by Newton steps computed from C¹ point runs at 256 bits (non-rigorous numerics; any
centre is allowed, the test is what is rigorous): first a Newton step in the higher harmonics alone at fixed (A, wf),
then full Newton steps. (A full step from an untuned centre would leak the inner quadratic error into the outer
coordinates through L.)

## 3. The runs

All runs use `validated_int.Lohner` (A_certify.md §9.1) through `certify_M_lohner.LohnerM`, which changes three things
that do not touch the mathematics of the step:

- the a-priori enclosures are lemma6_common's (a common inflation floor; containment still verified);
- the Taylor coefficients come from Picard iteration with growing series length (iteration k needs length k + 2; the
  coefficients 0..N are the same numbers);
- the system is `certify_M_sys.SysM` and the legs `certify_M_sys.LegM` (the start-ray angle is a parameter).

**Point run** (C¹ along one trajectory): the set of y⁺(·; p̂) and the M + 1 columns ∂y⁺/∂p_k at p̂, each a Lohner set,
started from the start values and their derivatives at p̂. Gives g(p̂) and Dg(p̂) with small radii. Used for g̃(0) in
the test, for C and Y, and (its midpoints) for the Newton refinement of the centre.

**Box run** (C¹ over P): the centre trajectory's set and the M + 1 tilted columns ∂y⁺/∂x_k = (∂y⁺/∂p)T_k, started
from their enclosures over the hull of P (exact parameter jets of the series along the columns of T, so the tilted
combination is formed before any interval operation). At every step the state set over P is rebuilt as
y(t; p̂) + V(t; X)·x (mean value theorem, validated_int's centre_set mode). Gives Dg̃(X) as Lohner sets.

**Self-test** (`python3 certify_M_sys.py selftest`): SysM's right-hand side and linearization against
validated_int.Sys at M = 1 (to 10⁻⁷²) and against inner_np.Inner.rhs at M = 3, 5 (double precision, 5×10⁻¹⁶), the
linearization against a central difference, the series recursion against series_arb.SeriesArb, and the parameter jets
against central differences. Mutation: `--mutate=coupling` (one coupling sign flipped) must and does fail.

## 4. The M = 1 case as a special case

At M = 1 there is no inner block: T = I, Y = (mid Dg(p̂))⁻¹ and g = (Re J_D, Im J_D). That is krawczyk.py's test on
J_D without the constant factor κ, to which the test is invariant (A_certify.md §10.3 checked both forms in CAPD and got
the same margins to three digits). The start data are fixed_point.start_data's, the path is the lower path of §11.2,
and the step is validated_int's. `python3 certify_M.py m1` runs it and checks that the zero enclosure overlaps the one
of Theorem A(M = 1) (A_certify.md §12.6); `--mutate=m1ref` shifts the reference by 10⁻¹¹ and must fail.

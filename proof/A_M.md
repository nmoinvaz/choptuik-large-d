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

**Path.** A start ray arg ξ = φ₀ from r₀ to ρ = 0.2, the arc |ξ| = 0.2 down to the real axis, the real axis from 0.2 to
ξ₁ = 0.3 (the lower path of Theorem A(M = 1)).

| M | φ₀ | r₀ | why |
|---|---|---|---|
| 1 | 3π/8 | 0.1 | Theorem A(M = 1) exactly (fixed_point.py's ray) |
| 3 | 5π/16 | 0.0741 | hier.py's / inner_np.py's start ray and radius (r0_for); the M-harmonic start-value lemma holds there with derivative bounds 6×10⁻¹⁴ |
| 5 | 5π/16 | 0.0622 | the same; derivative bounds 4×10⁻²⁰, while on 3π/8 they are 5.3 (useless) |

The derivative bounds are those of `fixed_point_M.py`'s table (the lemma agent's worktree, `lemmas_M_runs/fixed_point_M_table.log`,
Cauchy radius 10⁻³). Up to ρ = 0.2 the path is hier.py's; hier.py stops at ξ₁ = 0.2 on the real axis, this certificate
continues along the real axis to 0.3, where the outer signal is about 2×10⁴ times larger in absolute terms while the
state stays of order one (as at M = 1, A_certify.md §5.1). The zero set is the same as at 0.2 if lemma 6a holds at both
radii; the real leg needs no analyticity beyond the real axis.

**What the code proves (the claim, kept separate from the theorem it serves).** Given enclosures of y⁺ and of its
parameter derivatives at the start point (the start-value lemma), the code proves:

> g has exactly one zero in the parallelepiped P = p̂ + T·X, and Dg is non-singular there,

for an explicit centre p̂, an explicit matrix T (§2) and an explicit box X. That is all. It does not prove that the
path continuation is the Borel sum (lemma 6b), that J_ν(ξ₁) = 0 for all ν forces the whole jump to vanish (lemma 6a),
or anything about M > 5 or the untruncated problem.

**Assumed facts.**

- Braaksma's 1-summability theorem, for the existence of y⁺ and its asymptotics on its sector (as in Theorem A(M = 1)).
- The start-value lemma for the M-harmonic system: at M = 1 this is `fixed_point.start_data` (proved, A_certify.md §4.4–4.5).
  At M ≥ 3 it is `fixed_point_M.start_data_M` (another agent's file, notes in `proof/A_lemmas_M.md`; not yet merged or
  reviewed when these runs were made). certify_M.py imports it when it is importable (next to it after the merge, or
  from the directory in the environment variable `CERTIFY_M_FPM_DIR`) and records the file's path and SHA-256 in every
  run; otherwise it falls back to a **non-rigorous placeholder** (the series truncated at its smallest term plus an
  assumed error ball, `certify_M_sys.placeholder_start`), and a result obtained that way is numerical evidence produced
  by the certificate's code path, not a certificate. `python3 certify_M.py startcheck M` checks that the lemma's
  enclosure contains this file's own series (independent code; at M = 3 they agree to 4×10⁻⁷⁶).
- The correctness of the code: python-flint 0.9 (arb/acb), validated_int.py's Lohner step (reviewed,
  [A_review.md](A_review.md)), and the new files above (not reviewed).

## 2. Design: the Schur-corrected Krawczyk test in a tilted box

**The difficulty** (A_certify.md §7, PROOF.md A.3–A.4). In the jumps the higher harmonics dominate everything. In
hier.py's de-mixed residuals at M = 5 (its docstring): ∂r₅/∂B₅ ≈ 4×10⁸, ∂r₃/∂B₅ ≈ 10¹³, ∂r₁/∂B₅ ≈ 10¹⁷, against
∂r₃/∂B₃ ≈ 5×10⁵ and ∂r₁/∂A ≈ 10⁴. So r₁ ≈ L·r_H + (intrinsic part), with a leakage L of the higher conditions into the
lower ones, and the outer derivative that matters is the Schur complement S, many orders of magnitude smaller than
the raw entries. The same in the raw jumps J_ν at ξ₁ = 0.3, from the Newton runs of the centre refinement
(`python3 certify_M.py plan --centre=data/A_M_refine_M{M}_....json`, numerics; logs `logs/A_M_plan_M3.log`, `logs/A_M_plan_M5_it0.log`):

| raw jump units at ξ₁ = 0.3 | M = 3 | M = 5 |
|---|---|---|
| outer Schur complement S, singular values | 0.443, 0.238 | 5.45, 0.248 |
| inner block ∂g_H/∂p_H, singular values | 1.05×10⁷, 1.04×10⁷ | 3.24×10¹⁴ (twice), 1.07×10⁷ (twice) |
| raw outer block ∂g_o/∂p_o, largest entry | 3.7×10⁵ | 6.9×10¹⁰ |
| ∂g_o/∂p_H, largest entry | 1.8×10⁶ | 6.5×10¹² |
| leakage L = ∂_H g_o (∂_H g_H)⁻¹, largest entry | 0.17 | 0.18 |
| condition of Dg | 4.7×10⁷ | 1.6×10¹⁵ |

Two things go wrong in a naive Krawczyk test on (A, wf, B₃, …, C_M) with an axis-aligned box:

1. **Interval hulls lose the cancellation.** The outer rows of the preconditioned Jacobian are S⁻¹(Dg₁ − L Dg_H).
   If Dg is first enclosed entrywise over the box, the radius of Dg_o (the true variation of the leakage-dominated
   entries, about 10⁶ times larger than S at M = 3) is not cancelled by L Dg_H.
2. **The higher-harmonic box is first order.** An axis-aligned box must contain x_H*(x_o) over the outer box, which
   moves by |C|·h ≈ 0.2h. Through the inner sensitivity (the norm of Dg_HH) that makes the state set over the box, and
   with it the enclosed step Jacobians, far wider than the outer directions alone would.

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

(Krawczyk 1969; Moore 1977; Neumaier, *Interval Methods for Systems of Equations*, 1990; as in A_certify.md §5.1), hence g has exactly one zero in
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
solves), far from the 10⁻¹⁶–10⁻²⁰ the inner coordinates need. The centre is refined by Newton steps from
`certify_M_lohner.numerics_run`, a plain Taylor integrator at 256 bits (order 30, steps from the Taylor coefficients and
‖A₀‖h ≤ 2, and from the Taylor coefficients of the derivative columns; non-rigorous, and it need not be: any centre is allowed, the test is what is rigorous). (A first version took its steps from the trajectory alone, which is a slow solution: 98 steps for the whole path at M = 3. Its Dg was then accurate to only about 10⁻⁹ relative, because the derivative columns carry the large Stokes-switched fast components, and with the 10⁶-fold cancellation in the Schur complement the outer Newton steps converged only linearly, by a factor of about 80 per step; `logs/A_M_refine_M3_a.log`. A second version still accumulated t in double precision, so every step restarted the series about 10⁻¹⁶ off the true time; the Stokes switching amplifies such slow-direction errors by about 10⁷ at M = 3, which left a noise floor of 1.4×10⁻¹⁰ in J₃ (10⁻¹⁷ in B₃) and stalled the Newton iteration at 3–4×10⁻¹³ in the outer coordinates; `logs/A_M_refine_M3_b.log`. The steps now lie on the exact grid 2⁻⁴⁰ℤ, as the rigorous integrator's lie on 2⁻³⁰ℤ.) The steps alternate:
a Newton step in the higher harmonics alone at fixed (A, wf), then a full Newton step. (A full step from an untuned
centre would leak the inner quadratic error into the outer coordinates through L.) The last refinement also gives the
tilt C (from its Dg).

At M = 5 the higher-harmonic Newton step at ξ₁ = 0.3 converges only linearly at first: the harmonic-5 state at ξ₁
changes by about 3×10¹⁴ per unit of B₅ (the singular values of Dg_HH), so a step of 4.6×10⁻¹⁶ in B₅ is 15% of its
natural scale and J₅ is not linear on it (`logs/A_M_refine_M5_a.log`, `_b.log`: H steps 4.6×10⁻¹⁶, then 1.1×10⁻¹⁶).
The refinement therefore also has a *mixed* mode (`refine ... --mixed`): the outer conditions at ξ₁ = 0.3, the inner
ones at the end of the arc, ξ = ρ = 0.2, where the higher-harmonic jumps are smaller by the Stokes factor
e^{−1/(2ρ²)+1/(2ξ₁²)} and linear much farther out (hier.py's polish radius). Both systems vanish at the same point if
the vanishing D-jumps at one radius force the whole jump to vanish (lemma 6a(M)); the refinement is numerics either
way, and its record keeps g, Dg and the full Newton step of the ξ₁ system at every iterate, and the tilt C is taken
from the ξ₁ system.

**One rigorous run.** The certificate needs one box run only: its centre-trajectory set (the set of y⁺(·; p̂), started
from the point enclosure at p̂ and propagated with the step Jacobians enclosed over the box) gives g̃(0) = g(p̂), its
columns give Dg̃(X), and Y is the inverse of the midpoint of the columns (any Y is allowed). A separate C¹ point run at
p̂ can be supplied (`--point=`) but is not needed.

## 3. The runs

All runs use `validated_int.Lohner` (A_certify.md §9.1) through `certify_M_lohner.LohnerM`, which changes four things:

- the a-priori enclosures are lemma6_common's (a common inflation floor; containment still verified), as on the lower
  path of Theorem A(M = 1);
- the Taylor coefficients come from Picard iteration with growing series length (iteration k needs length k + 2; the
  coefficients 0..N are the same numbers);
- the columns of the linearization are computed by `SysM.lin_unit` (sparse: products with zero series skipped;
  self-tested against `lin` to 10⁻⁷¹);
- the step Jacobian may use a lower Taylor order NJ than the trajectory (order N). The enclosure
  J ∈ Σ_{k≤NJ} V_k(X)h^k + h^{NJ+1}Φ_{NJ+1}(Ỹ)Ṽ holds for any order (integral form of the remainder:
  V^{(NJ+1)}(s) = (NJ+1)! Φ_{NJ+1}(Y(s)) V(s) with Y(s) ∈ Ỹ, V(s) ∈ Ṽ), so a lower NJ only widens J, by about
  (λh)^{NJ+1}/(NJ+1)!, far below the width the parameter box gives it. M = 1 uses N = NJ = 24 (validated_int's
  choice); M ≥ 3 uses N = 20, NJ = 12.

**Checkpoints.** A box run saves its exact Lohner sets, the leg, t and the step size about every ten minutes and at the end of each leg (`data/...json.ckpt`, written atomically) and resumes from them when restarted with the same parameters (a key of M, p̂, C, h, w, the path, the orders and the start-data source must match). A resumed run continues from the saved balls, so its result is an enclosure; on a test path it agreed with an uninterrupted run in every midpoint and radius. Background commands on the cloud machine are stopped after two hours, so the long runs are relaunched until they finish.

At M = 1 one step of LohnerM and one step of validated_int.Lohner from the same set give identical midpoints and radii
(`certify_M_sys.py selftest`). One step costs about 1.5 s at M = 5 on an idle core (python-flint; 4–13 s on the shared
machine at load 13–17). The steps are limited by the first-order a-priori enclosure, which fails beyond λh ≈ 0.2–0.3 for
the local fast rate λ (`|ξ′|/|ξ|³` on the ray), whatever MAXREL is; the trajectory and its parameter derivatives are
slow solutions and would allow λh of order one (the numerics integrator takes such steps). A higher-order a-priori
enclosure would cut the step count several-fold; it was not built.

**Point run** (C¹ along one trajectory): the set of y⁺(·; p̂) and the M + 1 columns ∂y⁺/∂p_k at p̂, each a Lohner set,
started from the start values and their derivatives at p̂. Gives g(p̂) and Dg(p̂) with small radii. Used for g̃(0) in
the test, for C and Y, and (its midpoints) for the Newton refinement of the centre.

**Box run** (C¹ over P): the centre trajectory's set and the M + 1 tilted columns ∂y⁺/∂x_k = (∂y⁺/∂p)T_k, started
from their enclosures over the hull of P (exact parameter jets of the series along the columns of T, so the tilted
combination is formed before any interval operation). At every step the state set over P is rebuilt as
y(t; p̂) + V(t; X)·x (mean value theorem, validated_int's centre_set mode). Gives Dg̃(X) as Lohner sets.

**Cross-check against independent code** (`python3 certify_M.py crosscheck M`). The jumps J_ν(0.2) at a point near
the zero (where they have their natural size; at the zero they vanish), on hier.py's own path (ray 5π/16 from inner_np's
r₀, arc at 0.2), from this code's Taylor integrator and from `inner_np.py` (scipy DOP853 at rtol 10⁻¹², its own series
start, shares no integration code): at M = 3 they agree to 8.5×10⁻¹⁴ of the state (J₁, J₃ at
p = (2.7, 1.0, 0.14, 0.01)); with inner_np at wf + 10⁻⁶ (`--mutate`) they must disagree. This checks the system, the
start values and the path against the double-precision solver the zeros came from; it is not a check of the enclosure
machinery (that is the Lohner step check above and A_review.md).

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

## 5. Controls, written down before the runs

Every check below has a runnable mutation, with the verdict it must give. The Krawczyk controls follow A_certify.md §9.5:
they mutate the function or plant a zero, never an enclosure alone (a wrong J(X) can make the test pass or fail either
way), except C7, which widens g(p̂) and must make it fail.

| check | control | how to run | expected |
|---|---|---|---|
| system self-test | none | `python3 certify_M_sys.py selftest` | PASS |
| system self-test | one coupling sign flipped (`[(C E) D]` with the wrong sign) | `--mutate=coupling` | FAIL |
| Krawczyk test | C0 clean | `python3 certify_M.py controls --box=...` | pass |
| | C1 jump sign flipped, D_ν + conj D₋ν | (same command, all rows) | fail |
| | C2 conjugation dropped, D_ν − D₋ν | | fail |
| | C3 −conj J (null: same zeros) | | pass, enclosure overlapping C0's |
| | C4 J_ν → (ν + 1)J_ν (null: same zeros) | | pass, enclosure overlapping C0's |
| | C5 zero planted outside, moved by 2h in A | | fail |
| | C6 zero planted inside, moved by (h/2, −3h/10, w/4, …) | | pass |
| | C7 g(p̂) widened by 10³ h·σ_min(S) in the outer components | | fail |
| | C8 zero planted outside in B₃ (moved by 2w) — the inner interval Newton must reject it | (M ≥ 3) | fail |
| M = 1 reproduction | none | `python3 certify_M.py m1` | REPRODUCED |
| M = 1 reproduction | Theorem A(M = 1)'s A-interval shifted by 10⁻¹¹ | `--mutate=m1ref` | NOT REPRODUCED |
| hier.py agreement | none | `krawczyk ... --hier` | both statements hold |
| hier.py agreement | hier's zeros moved by 100× their spread (≥ 10⁻⁹) in A | `--hier --mutate=hier` | fails |
| independent assembly (mpmath intervals) | none | `python3 certify_M_recheck.py data/A_M_box_....json` | the same verdict, enclosure overlapping |
| independent assembly | jump sign flipped | `--mutate` | fail |
| start data | none | `python3 certify_M.py startcheck M` | contains |
| start data | this file's series shifted by 10⁻⁴⁰ | `--mutate` | does not contain |

A planted zero is g − L d with L the exact midpoint of Dg̃(X): it moves the zero by d up to second order and leaves
Dg̃(X) unchanged.

## 6. Results

### 6.1 M = 1: Theorem A(M = 1) reproduced through this code path

`python3 certify_M.py m1` (lower path, fixed_point.start_data, h = 10⁻⁹; data `data/A_M_point_M1.json`,
`data/A_M_box_M1_1e-9.json`, log `logs/A_M_m1.log`):

| | this code | Theorem A(M = 1), A_certify.md §11.2 |
|---|---|---|
| steps (ray, arc, real), point run | 303, 320, 46 | 303, 322, 47 |
| J_D(x̂) | 1.054739676292166×10⁻¹² − 3.490429414033870×10⁻¹²i, radius 7×10⁻¹⁷ | the same digits (§10.3), radius 6×10⁻¹⁷ |
| K(X) ⊂ int X at h = 10⁻⁹ | yes, relative margins 0.863, 0.976 | yes, 0.857, 0.971 |
| smallest passing sub-box | x̂ ± 1.22×10⁻¹⁰ | x̂ ± 1.2×10⁻¹⁰ |
| the zero, A | [2.54214804761406, 2.54214804761906] | [2.54214804761338, 2.54214804761975] |
| the zero, ω/ω∞ | [0.776994013607969, 0.776994013611479] | [0.776994013607438, 0.776994013612010] |

The enclosure lies inside Theorem A(M = 1)'s (endpoints rounded outward). The test here is posed on g = J_D without κ
and with Y from the box run's own column midpoints, so the agreement checks the generic assembly, the tilted-box code
with an empty inner block, and LohnerM against validated_int. The independent assembly `certify_M_recheck.py` (mpmath interval arithmetic, its own jump functionals and its own preconditioner, sharing only the saved Lohner sets) gives the same verdict and margins (0.863, 0.976) and the enclosure A ∈ [2.5421480475960, 2.5421480476371]. With Theorem A's A-interval shifted by 10⁻¹¹
(`--mutate=m1ref`) the check reports NOT REPRODUCED. All M = 1 controls give their expected verdict
(`logs/A_M_controls_M1.log`); C1 and C2 have absolute margins −59, −22 and −2.0, −0.35, the same as A_certify.md §9.5.

### 6.2 M = 3: certified (given the M-harmonic start-value lemma)

**The centre** (`data/A_M_refine_M3_c.json`, numerics): p̂ = (A, wf, B₃, C₃) =
(2.72647588563088004, 1.01560309485773330, 0.144419478470076894, 0.0110874258881167086) (the last Newton steps are
1.5×10⁻²⁵ outer and 4×10⁻²⁸ inner). The box run below was centred at the previous refinement's centre
(`data/A_M_refine_M3_a.json`, `p_out`), which is 2.7×10⁻¹³ (outer) and 8×10⁻¹⁸ (inner, B₃) from it.

**Structure in raw jump units at ξ₁ = 0.3** (`logs/A_M_plan_M3.log`): outer Schur complement S with singular values
0.443 and 0.238; inner block Dg_HH with singular values 1.05×10⁷ (twice); raw outer block up to 3.7×10⁵ and Dg_oH up to
1.8×10⁶ (the leakage L = Dg_oH Dg_HH⁻¹ is at most 0.17); tilt C = [[0.142, −0.245], [0.004, −0.021]]; condition of Dg
4.7×10⁷.

**The run** (`data/A_M_box_M3_1e-9.json`, log `logs/A_M_box_M3_1e-9.log`): h = 10⁻⁹, w = 10⁻¹⁶, start data from
`fixed_point_M.start_data_M` (SHA-256 prefix 589e5a810bc50289, the lemma agent's working copy at the time), 955 + 552 +
72 steps, 72 minutes at load 17–22.

**The test** (`logs/A_M_krawczyk_M3_1e-9.log`). On the run's own box the outer rows pass (relative margins 0.944,
0.970) but the inner rows fail: |K_z| ≈ 89 w. The inner rows of Y Dg̃(X) have radii 3–6×10⁻⁶ in the outer columns,
because the enclosure excess of the outer columns in the J₃ components reaches 30–60 (absolute) at ξ₁: interval excess
in slow harmonic-3 directions is amplified by the Stokes switching, and the largest derivative radius grows from
2×10⁻² at the end of the arc to 2–7×10² (per column) at ξ₁, on the real leg where the fast modes grow. Dg̃(X) encloses the Jacobian on every sub-box of X, so the
test was repeated on sub-boxes with the inner widths kept: it passes for outer half-widths up to 9.7×10⁻¹².

> **Certified** (given the start-value lemma `fixed_point_M.start_data_M` and Braaksma's theorem; code certify_M*.py on
> python-flint 0.9, not reviewed): the D-jump function g of §1 at M = 3 has exactly one zero in the parallelepiped
> P = p̂ + T·([−9.70×10⁻¹², 9.70×10⁻¹²]² × [−10⁻¹⁶, 10⁻¹⁶]²), and its Jacobian is non-singular there. Relative margins
> 0.916, 0.964, 0.100, 0.057. The zero lies in (endpoints rounded outward; the smallest passing sub-box, θ = 0.656):
>
> A ∈ [2.72647588563052, 2.72647588563124], ω/ω∞ ∈ [1.01560309485754, 1.01560309485793],
> B₃ ∈ [0.144419478469979, 0.144419478470175], C₃ ∈ [0.0110874258881111, 0.0110874258881224].

The refined numerical zero lies inside this enclosure in every coordinate. The independent assembly
(`certify_M_recheck.py --sub=...`, `logs/A_M_recheck_M3_1e-9.log`) gives the same verdict and margins.

**Controls** (`logs/A_M_controls_M3_1e-9.log`, on the certified sub-box): C0–C5, C7, C8 as expected. **C6 does not
give its expected verdict**: the zero planted inside at (h/2, −3h/10, w/4, w/4) is not certified (inner margins −0.15,
−0.19). Its planted inner shift, a quarter of w, is larger than the clean inner margins (0.100, 0.057), so the test
cannot certify it; the control's expected verdict assumed margins of order one, as at M = 1. It is reported, not
changed. C6b, added afterwards with CAPD's rule (A_certify.md §10.3: shift by half the clean margins), passes. At
M = 1 every control, C6 and C6b included, gives its expected verdict.

**hier.py's zeros** (`logs/A_M_hier_M3_1e-9.log`): hier.py's zero lies inside the axis-aligned hull of P; nested.py's
lies 1.7×10⁻¹² outside it in A (the two solvers differ by 1.7×10⁻¹¹ in A and ω/ω∞, more than the certified outer
half-width). Both agree with the certified zero to within their own spread in every coordinate (|difference| ≤ 1.14×10⁻¹¹
in A, 1.09×10⁻¹¹ in ω/ω∞, 1.2×10⁻¹² in B₃, 1.8×10⁻¹³ in C₃). With hier's zeros moved by 100 times their spread the check
fails, as it must.

**For lemma 6a(M)**: the state box at ξ₁ over the run's whole parallelepiped (which contains P) is in
`data/A_M_6a_M3_1e-9.json` (keys M, h, xi1, yX_xi1, and V), max radius 5.2×10⁻⁷. The certified box lies within
x̂ ± 10⁻⁸ of `fixed_point_M.py`'s x̂ for M = 3 (hypotheses H1, H2 of A_lemmas_M.md §6), and the path (ray 5π/16 from
0.0741, arc at 0.2, real axis to 0.3) is of the kind its §6 assumes.

## 7. Cost, and a proposed task for the local machine

Wall-clock times on the shared 2-core cloud machine at load 17–25 (each process got a fifth to an eighth of a core),
one process each, under `nice -n 10`:

| run | n (complex) | steps (ray, arc, real) | time | per step |
|---|---|---|---|---|
| M = 1 point run (C¹, 2 columns), N = NJ = 24 | 7 + ω | 303, 320, 46 | 18 min | 1.6 s |
| M = 1 box run | 7 + ω | 303, 320, 46 | 21 min | 1.8 s |
| M = 3 box run, N = 20, NJ = 12 | 13 + ω | 955, 552, 72 | 72 min | 2.7 s |
| M = 3 refinement, one Newton iteration (numerics, order 30) | 13 + ω | 193 in all | 11–13 min | 3.6 s |
| M = 5 refinement, one Newton iteration | 19 + ω | 307 in all | 41 min | 8 s |
| M = 5 box run | 19 + ω | M5_STEPS | M5_TIME | M5_PER |

The step counts are set by the first-order a-priori enclosure (§3), not by the accuracy needed.

**Proposed task for the local 8-thread machine** (for the coordinator to queue; COST_LOCAL).

- *What:* LOCAL_WHAT
- *Prerequisites:* python-flint 0.9.0 and numpy; this branch merged with the lemma agent's (so that
  `solver/proof/fixed_point_M.py` sits next to `certify_M.py`), or `CERTIFY_M_FPM_DIR` pointing at it. Check first:
  `python3 certify_M_sys.py selftest` (PASS) and `python3 certify_M.py startcheck 5` (contains).
- *Threads:* LOCAL_THREADS, one process each, at below-normal priority.
- *Commands* (from `solver/proof/`):

```
LOCAL_COMMANDS
```

- *Stop conditions:* a box run whose log shows `step size underflow`, or a derivative radius above 10³ in its progress
  lines, is stopped (it will not pass); the refinement is stopped if its full Newton step does not shrink by at least
  10³ per full iteration (the numerics are then not resolving Dg). A run interrupted by a restart is simply relaunched
  with the same command: it resumes from its `.ckpt` file.
- *Push:* `solver/proof/data/A_M_*M5*.json` (not the `.ckpt` files) and `solver/proof/logs/A_M_*M5*.log`.

## 8. Reproduce

All commands from `solver/proof/`. At M ≥ 3 the rigorous start data come from `fixed_point_M.py`: after the merge it
sits next to these files; before it, point `CERTIFY_M_FPM_DIR` at the directory that holds it. Times are on the shared
2-core cloud machine at load 17–20 (one process gets a fifth of a core).

```
python3 certify_M_sys.py selftest                       # system, series, jets, one Lohner step vs validated_int; seconds
python3 certify_M_sys.py selftest --mutate=coupling     # must FAIL
python3 certify_M.py m1                                 # M = 1 through this code path (point + box run, ~40 min)
python3 certify_M.py m1 --mutate=m1ref                  # must print NOT REPRODUCED (reuses the saved runs)
python3 certify_M.py startcheck 3                       # rigorous start data vs this file's series (~2 min)
python3 certify_M.py startcheck 3 --mutate              # must fail
python3 certify_M.py crosscheck 3                       # J_nu(0.2) vs inner_np.py (independent integrator)
python3 certify_M.py crosscheck 3 --mutate              # must DISAGREE
python3 certify_M.py refine 3 --iters=3 --tag=a         # centre (numerics) -> data/A_M_refine_M3_a.json
python3 certify_M.py plan --centre=data/A_M_refine_M3_a.json
python3 certify_M.py box 3 --centre=data/A_M_refine_M3_a.json --use_out --h=H --w=W \
    --out=data/A_M_box_M3_H.json --log=logs/A_M_box_M3_H.log          # the rigorous run
python3 certify_M.py krawczyk --box=data/A_M_box_M3_H.json --hier       # the test (seconds), then hier.py's zeros
python3 certify_M.py controls --box=data/A_M_box_M3_H.json              # C0-C8, exit 0 iff all as expected
python3 certify_M.py krawczyk --box=data/A_M_box_M3_H.json --hier --mutate=hier   # must fail
python3 certify_M.py export6a --box=data/A_M_box_M3_H.json --out=data/A_M_6a_M3_H.json   # state box for lemma 6a(M)
python3 certify_M_recheck.py data/A_M_box_M3_H.json [--mutate]          # independent assembly (mpmath)
```

## 9. Proposed text for README.md, PROOF.md and A_certify.md

(For the coordinator; numbers to be checked against §6 before use.)

**PROOF.md §4, after the paragraph on Theorem A(M = 1):** "Beyond one harmonic ([proof/A_M.md](proof/A_M.md)). The point
certificate is posed on the full (M + 1)-dimensional system of D-jumps at ξ₁ = 0.3, in coordinates that follow the
higher harmonics' zero manifold to first order (hier.py's linear predictor), with a preconditioner whose outer rows are
the Schur-corrected outer condition and whose inner rows are the interval Newton step for the higher-harmonic
unknowns; the preconditioner is applied to the Lohner sets before any interval hull, so the 10¹⁷-sized leakage cancels
exactly. RESULT_SENTENCE"

**A_certify.md §7 ("From M = 1 to M = 5"), a closing line:** "Done in [A_M.md](A_M.md): the de-mixing and the Schur
correction are a choice of preconditioner, the inner unknowns are measured from the linearized zero manifold, and one
Krawczyk test on the whole system does what the nested test would. Multiprecision is needed for the preconditioner
(condition about 10⁷ at M = 3 in raw jump units) but not for the integration, which runs at 256 bits as at M = 1."

**README, where the M = 3 and M = 5 zeros are quoted (the selection-rule section):** "RESULT_README"

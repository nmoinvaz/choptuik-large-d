# The strong-row term S of the reduced cubic at the D 242.5 starts (#528)

#499 found that the first Newton step of the refused corrector attempts at the D 242.5 wall lies on the four weak
directions, and that on the affine subspace z0 + Vc the weak equations are cubic, with T = UᵀF‴ ≈ 1.6×10³³. With the
weak part of the first step put on that cubic's root, newton_od was still refused (#499's F4). #527 pointed out that the
reduced equation, with the strong rows solved, has the cubic coefficient T − 3S (Kuznetsov 1999, §5, for one null
vector), where S is the strong rows' curvature fed back into the weak rows. This note measures S at #499's two
D 242.5146 starts, and the reduced map itself on the strong-solved manifold.

Results are in section 4 onward, after the protocol (section 3), which was committed before the first real run.
Section 9 (#677) follows the reduced map along rays out to 10⁵|a|, at those two starts, at D 232.13 and at a class A
start of #603. Section 10 (#694) finds where the weak rows live at the same four starts. Section 11 (#698) follows the
map out to |c| = 10⁻², with the strong rows solved by Newton, at REF and the class A start.

## Verdicts

| question | verdict | numbers (REF; ACC) |
|---|---|---|
| Plant: tensors, root, controls recovered; mutations fail | PASS, and all four mutations FAIL | worst tensor error 3.8e-11 (bound 1e-6); root 2.6e-5 (bound 1e-3) |
| S resolved, h against h/10 | PASS | 1.7e-17; 1.7e-17 |
| S control, w → 2w against −½S(c\*, c\*, c\*) | FAIL | 2.72e-2; 3.55e-2 (bound 1e-2), the control's own truncation (5.4) |
| **3S against T** | **3S = T** | 3max\|S\|/max\|T\| 1.000, max\|T − 3S\|/max\|T\| 1.6e-19; 1.000, 1.6e-19 |
| **The reduced map on the strong-solved manifold** | **affine; its root is Newton's first move a** | \|g(a)\|/\|Ba\| 2.7e-25; 4.5e-25 |
| The issue's verdict | the coupling is this term; the cubic with T − 3S has the root c\*\* = a | \|g(c\*\*)\|/\|Ba\| 7.9e-10; 1.2e-9 |
| #499's root c\* on that manifold | not a root | \|g(c\*)\|/\|Ba\| 0.176; 0.208 |
| The refused start, strong rows solved at c = a | converged, below the chain's 1e-40 | max\|F\| 2.0e-51; 1.1e-50 |
| The projector (exact here, rounded in #499) | no effect at #499's printed digits | section 5.1 |
| **Where g stops being affine** (#677: rays to 10⁵\|a\|, four starts) | **affine with slope B + L, plus a quadratic; no cubic or quartic resolved** | the quadratic reaches 10⁻²\|Ba\| near 10¹³\|a\| (extrapolated); section 9 |
| **Does the map curve on \|c\| of order 0.02 to 0.1?** (#698: \|c\| = 10⁻⁶ to 10⁻², REF and class A) | **not decided**: Newton with a refactored LU stops at 10⁻⁶ and the chord near 10⁻¹⁸; the flat-chart control fires | to 3×10⁻¹⁸ at REF the map is quadratic, with a resolved cubic of ordinary size (19.2\|Be\| per t², 0.95 of κ²); the linear pair test puts a fold of the strong block near 5×10⁻¹⁷ (REF) and 10⁻¹⁷ (class A) (*inferred*); section 11 |
| **Do the weak rows live on rows affine in z?** (#694, four starts) | **no: U lives on the last matching (node 4's Taylor arrival against the horizon series), on no affine row** | share of \|U\| on rows affine to 10⁻¹⁰: 1.9×10⁻¹⁸ to 4.2×10⁻¹⁸; Q and T − 3S are cancellations between rows that are not affine; section 10 |

- **At these starts T = 3S.** With the strong rows solved, the weak equations are affine to 2.7×10⁻²⁵, so #499's cubic
  is the curvature of the affine subspace z0 + Vc, and its root c* is not a root of the system.
- **Newton's first move is right in the weak directions.** What the refused attempt lacks is the strong rows' own
  correction at c = a, which three chord steps with the start's LU find. After it the residual is 2.0×10⁻⁵¹ at REF and
  1.1×10⁻⁵⁰ at ACC. Numerical evidence, at two starts.

## 1. Notation and method (`hp/reduced_cubic.py`)

As in #499 and #527: z0 is the corrector's start, J = F′(z0) the pinned system's Jacobian, V the four weakest right
singular vectors (subspace iteration in f64, then three steps of block inverse iteration at 140 digits, as
`hp/dz0_weak.py`), U = orth(J⁻ᵀV) with J⁻ᵀV = UR, B = UᵀJV = R⁻ᵀ, and a = Vᵀdz0 the weak part of Newton's first move.
With U built this way, UᵀJw = 0 for every w orthogonal to V, and J⁻¹ maps the complement of U onto that of V. So
Kuznetsov's bordered solve A^INV P_X is J⁻¹(I − UUᵀ) here, one solve with z0's octo-double LU.

- **The strong solve.** At fixed weak coordinates c, w(c) ⊥ V solves the strong rows, (I − UUᵀ)F(z0 + Vc + w) = 0, by
  the chord iteration w ← w − J⁻¹(I − UUᵀ)F with z0's LU. Each step stays orthogonal to V. Its convergence is judged
  by the size of the correction |dw| (2-norm), converged below 10⁻⁴⁰|w0|. The max norm of the strong residual is not
  used, since the strong rows differ in scale by 10⁵³ and that norm can grow while w converges (seen on the plant).
  One step whose |dw| does not fall is allowed, since in the softest strong directions the second correction can undo
  the first. A solve that stalls for two steps counts only if its last correction moves g by less than 10⁻⁶|Ba|.
- **The reduced map** is g(c) = UᵀF(z0 + Vc + w(c)) (Keller 1987, Method III; Decker, Keller and Kelley 1983,
  Thm 2.6). w0 = w(0) is the start's own strong correction: z0 does not solve its strong rows exactly.
- **The tensors at z0**, each by differences of F at two steps h and h/10:
  - Q_mjl = u_m·F″(v_j, v_l), central differences at h = 10⁻³², 10⁻³³;
  - T_mijl = u_m·F‴(v_i, v_j, v_l), antisymmetric four-point differences and polarization (as `dz0_weak.cubic`), at
    h = 10⁻²³, 10⁻²⁴;
  - w_jl = J⁻¹(I − UUᵀ)F″(v_j, v_l), central differences at h = 10⁻³², 10⁻³³ and 10⁻³⁴, taken from 10⁻³⁴ and judged
    on the last two. The steps are small because J⁻¹(I − UUᵀ) multiplies the difference's h² truncation along the
    weakest strong directions (σ₅ ≈ 10⁻¹⁹) by up to 10¹⁹. At REF the truncation was all of w_jl at h = 10⁻²⁴, and
    1.4×10⁻² of it at 10⁻³² (section 3);
  - S_m,i,jl = u_m·F″(v_i, w_jl), mixed central differences with the step h along v_i and along w_jl/|w_jl|, at
    h = 10⁻³², 10⁻³³, symmetrized as S_mijl = (S_m,i,jl + S_m,j,il + S_m,l,ij)/3;
  - L_mj = u_m·F″(v_j, w0) and q00 = UᵀF″(w0, w0), the coupling through w0, at the same steps as S.
- **The expansion.** With w(c) = w0 − ½Σ w_jl c_j c_l + …, the reduced map is

```math
g(c) = g(0) + (B + L)\,c + \tfrac12 Q(c, c) + \tfrac16 (T - 3S)(c, c, c) + \dots
```

  This is Kuznetsov's coefficient c = (1/6)⟨p, C(q,q,q) − 3B(q, A^INV B(q,q))⟩, here with four null vectors and a
  start that is not a root (*inferred*, the same Taylor expansion). The map's own linear and quadratic coefficients,
  B_LS and Q_LS, are also formed directly, by central differences of g at h = 10⁻³³, each point a strong solve.
- **The models.** Each is g(0) + B′c + ½Q′(c, c) + (1/6)C′(c, c, c):

| model | g(0) | B′ | Q′ | C′ |
|---|---|---|---|---|
| T (#499's) | −Ba | B | 0 | T |
| T,Q | −Ba | B | Q | T |
| T-3S (the issue's cubic, with Q) | g(0) | B | Q | T − 3S |
| T-3S,L | g(0) | B + L | Q | T − 3S |
| T-3S,LS | g(0) | B_LS | Q_LS | T − 3S |

  Their real roots are found by Newton at 140 digits from c = a, from 0 and from 24 random starts. g is evaluated at
  the six roots nearest a, and the root with the smallest |g| is the model's c**. c* is the root of model T whose
  affine weak residual UᵀF(z0 + Vc) is smallest, #499's root, recomputed here.
- **The controls at c\*.** The S control replaces dw = w(c*) − w(0) by 2dw. That changes g by UᵀF″(Vc*, dw) + …, which
  is −½S(c*, c*, c*) to leading order. The w control replaces w(c*) by 2w(c*), which changes g by
  Lc* − ½S(c*, c*, c*) + (3/2)q00 to leading order. The issue's "w" is read as dw, the part of w that depends on c:
  at these starts w0 is not zero, and its coupling L enters the w control at first order in c.

## 2. The plant (`reduced_cubic.py planted`)

A synthetic system with a known reduced map, through the same functions as the real one; only F and the solves with
J differ. It has canonical unknowns (y ∈ R⁴, x ∈ R⁸), rotated on both sides by random orthogonal matrices at 140
digits, and z0 = 0:

```math
G_w = \beta + \Sigma y + \tfrac12 Q(y,y) + \tfrac16 T(y,y,y) + \tfrac1{24} P_4(y,y,y,y) + K(y, x), \qquad G_s = \gamma + (\Lambda + M(y))\,x + \tfrac12 C(y,y) + \tfrac1{24} c_4 (b_4 \cdot y)^4 e_0.
```

The strong rows are linear in x at fixed y, so the reduced map is G_w(y, x(y)) with x(y) = −(Λ + M(y))⁻¹(γ + ½C(y, y)
+ …), and its truth is B = Σ, w0 = −Λ⁻¹γ, L = K(·, w0), S_m,i,jl = K_m(e_i, Λ⁻¹C(e_j, e_l)) and q00 = 0. The scales are
the D 242.5 starts': σ = 2.1×10⁻²⁷ … 7.1×10⁻²⁶, strong singular values Λ = 10⁻¹⁹ … 3×10³⁴, max|T| 6.3×10³³,
3max|S|/max|T| 0.53, |L| comparable with |B|, |Q| ≈ 2×10³, |w0| 5.9×10⁻³³, |w_jl| 6×10² … 6×10³.
- The quartic P₄ and the strong-row coupling M make the cubic model inexact, by about 10⁻⁴|Ba| at the root.
- The quartic c₄ = 10⁴³ in the weakest strong row (λ = 10⁻¹⁹) reproduces the w_jl truncation found at REF: from
  h = 10⁻²⁴ it gives |w_jl| up to 10¹², all truncation.
- β is set so that g(y*) = 0 at a planted root y* of size 4.5×10⁻³⁰. Of 20 random directions for y*, the one whose
  root is best conditioned is taken, with κ = σ_min(g′(y*))|y*|/|Ba| largest (0.53 at seed 5). A first plant at an
  arbitrary direction put y* in a cluster of three roots within 10⁻³⁰ of each other, which tests the cluster's
  conditioning rather than the recovery.

Pass criteria, fixed before the plant's first run except where noted:
- **(P1)** B, L, Q, T and S, rotated into the plant's basis, equal the planted ones to 10⁻⁶ of their largest entry.
- **(P2)** S at h against h/10 to 10⁻² of max|S|, and w_jl at h/10 against h/100 to 10⁻² (2-norm, the worst pair).
- **(P3)** Model T-3S,L has a root within 10⁻³ relative of the planted root, with |g| < 10⁻²|Ba| there. The plant must
  show: |g(c*)| ≥ 0.1|Ba| at model T's root, so that #499's root is not a root of the reduced map.
- **(P4)** The S control to 10⁻².
- **Mutations.** `--mutate=w0` skips the strong solve (w(c) = 0, so g is the affine map on z0 + Vc): (P3) and (P4)
  must fail. `--mutate=Ssign` flips S's sign in the models and in the control's prediction: (P3) and (P4) must fail.
  `--mutate=Sh` steps 10h along w_jl/|w_jl| at h/10 but divides by h², so S(h/10) is ten times too large: (P2) must
  fail. `--mutate=hF` forms w_jl at h = 10⁻²⁴: its h check in (P2) must fail. A failed check exits 1.

Added after the first plant runs: the conditioning rule for y* and the strong solve's |dw| test. Added after the
stopped REF runs (section 3): the w_jl check at h = 10⁻³², 10⁻³³ and 10⁻³⁴, the plant's quartic c₄ and `--mutate=hF`.
No criterion was loosened.

## 3. Protocol for the real starts (fixed before the first real run)

- **Inputs.** Base `I461-win2/probeG03125a/anm_pin3_D242.5146.npz`; the accepted attempt
  `I461-win2/probeG0078/dz0_pin3_D242.5225_a0.007812.npz` (a = 0.0078125, "ACC") and the refused one
  `I461-win2/probeG03125b/dz0_pin3_D242.5459_a0.031250.npz` (a = 0.03125, "REF"), raw-data repository at 5e8dbd6.
- **Kernel and projector.** main's 3c35f0c build (multishoot 1.2.2, multidouble 4.2.1), the exact freeze projector,
  the Mac's scratchpad build `i599/site2`. #499's runs used venv-314's build of 2026-11-08, before #523, with the
  double-double (rounded) projector, which is also what win2 made these starts with. The projector differs only in the
  frozen rows, which are the horizon block's p rows, and the recovered offset c = F_bare(z0) + J dz0 is taken on
  exactly those rows. So the recovered system has the attempt's start and first step whichever projector is used, and
  its Jacobian differs in the frozen rows by the projector's rounding. The comparison of σ, |Ba|, T, c* and the affine
  residual at c* with #499's values measures what that changes. It is reported, not judged.
- **(R0)** The held offset off its rows below 10⁻⁶ of its largest entry (exit 1 otherwise); the recovered start's
  max|F| against #499's 9.041×10⁻³⁶ (REF).
- **(R1)** S at h = 10⁻³² against 10⁻³³ to 10⁻² of max|S|. If they differ by the factor of an h² truncation
  (max|S(h)|/max|S(h/10)| between 50 and 200), S is reported as bounded by max|S(h/10)|, an estimate, and (R1) passes.
  w_jl at h = 10⁻³³ against 10⁻³⁴ to 10⁻² (added after the stopped runs below).
- **(R2)** The S control to 10⁻² of the larger of the two changes, judged when that is above 10⁻⁸⁵. Below that it is
  reported as not resolved.
- **(R3)** The issue's verdict. If 3max|S| < 10⁻²max|T| and |g(c*)| < 10⁻²|Ba|, the coupling is not this term.
  Otherwise model T-3S's c** is reported with |g(c**)|/|Ba| against 10⁻², or that it has no root. Models T-3S,L and
  T-3S,LS are reported beside it.
- **Diagnostics, not judged.** B_LS against B + L, Q_LS against Q, dw against its quadratic model −½Σw_jl c_j c_l, the w
  control, max|F| at each root, and the chord's steps.
- **Stop conditions.** A strong solve that does not converge at c = 0 or at c*, or a failed (R2), stops the work for
  that start, which is then reported. REF runs first, as the pilot.
- **The two stopped runs.** The first REF run formed w_jl at h = 10⁻²⁴ and 10⁻²⁵. It found |w_jl| = 4.5×10¹¹ to
  1.2×10¹³, with h against h/10 at 0.99, the factor 100 of an h² truncation (raw `I528-mac/ref_hF1e-24_stopped.log`).
  That w_jl would have made S wrong in a way S's own h check cannot see, since both of S's steps use the same w_jl. It
  was stopped there by hand, before any S was formed. The second run, with the new w_jl check at h = 10⁻³² against
  10⁻³³, found |w_jl| = 0.058 to 0.144 and the check at 1.41×10⁻², which fails the bound 10⁻²
  (`ref_hF1e-32_stopped.log`). It was stopped too. The cause is the same truncation: 1.4×10⁻² of |w_jl| ≈ 0.1 at
  10⁻³², scaled by (10⁻²⁴/10⁻³²)² = 10¹⁶, predicts about 10¹³ at 10⁻²⁴, which the first run measured. So w_jl is now
  taken one decade lower, and the bound is unchanged. The setup, basis and Q lines of both runs are results
  (section 5).
- **Runs 3 and 4.** REF's third run (d0225a8b) completed, and its S control failed (2.72×10⁻²), so the work stopped
  until the cause was known. Run 4 (30a7bd35) added the control's scan along c* (section 5.4) and a stall rule for the
  chord. That rule stopped the chord at c* between two corrections that cancel: |dw| went 5.44×10⁻⁵⁶, 5.44×10⁻⁵⁶,
  then 5×10⁻⁸² in run 3, which kept going. The stall left g unchanged to 7 digits but dw 4×10⁴ times too large. The
  final rule allows one non-contracting step. The results are those of 2946e078 (section 5); runs 3 and 4 agree with
  them in every number they share, except run 4's dw and its |g(a)|, 2.75×10⁻²⁵ against 2.72×10⁻²⁵.

## 4. The plant's results

`python3 reduced_cubic.py planted` from `hp/` (seed 5; raw `I528-mac/planted*.log`):

| check | result | control | control's result |
|---|---|---|---|
| (P1) recovery, relative to the largest entry | B 6e-107, L 2e-57, Q 3.8e-11, T 2e-118, S 2.9e-33: PASS | `--mutate=Sh` | S 9.0: FAIL |
| (P2) S at h against h/10 | 2.6e-98: PASS | `--mutate=Sh` | 0.9: FAIL |
| (P2) w_jl at 10⁻³³ against 10⁻³⁴ (10⁻³² against 10⁻³³) | 7.9e-8 (7.9e-6): PASS | `--mutate=hF` | 99 (99): FAIL |
| (P3) model T-3S,L's root against the planted one | 2.6e-5, \|g\|/\|Ba\| 2.2e-5: PASS | `--mutate=w0`, `Ssign` | \|g\|/\|Ba\| 1.24, 1.03: FAIL |
| (P3) the plant shows | \|g(c\*)\|/\|Ba\| 9.94 | | |
| (P4) S control | 1.4e-4: PASS | `--mutate=w0`, `Ssign` | 1.0, 2.0: FAIL |

Every run's exit is as required: 0 for the plant, 1 for each of the four mutations (`exits.txt`). The errors of (P1)
are those of the steps: Q's 3.8×10⁻¹¹ is P₄'s h² truncation, and S's 2.9×10⁻³³ is c₄'s truncation in w_jl.

On the plant, model T-3S (B, Q and T − 3S, without L) leaves |g(c**)|/|Ba| = 3.18, since the plant's L is as large as B
by design. The map's own coefficients agree with the tensors: B_LS against B + L to 2.7×10⁻⁸, and Q_LS against Q to
1.1×10⁻⁴, which is the plant's M(y)w0 term. Under `--mutate=hF` the truncated w_jl (up to 1.5×10¹⁰) leaves S
unchanged to 3×10⁻¹⁷, because the plant's K is small in the weakest strong row. So S's correctness cannot be inferred
from a downstream check, and the w_jl check is what guards it.

## 5. Results at the D 242.5146 starts

### 5.1 The rebuilt system, and the projector

The recovered systems on main's exact-projector kernel reproduce #499's numbers on the rounded kernel at every digit
#499 printed. So at these starts the projector's rounding is absorbed by the recovered offset, as section 3 expected.

| quantity | REF here | REF, #499 | ACC here | ACC, #499 |
|---|---|---|---|---|
| held offset off its rows (bound 10⁻⁶ of 8.16×10⁻⁴) | 1.62e-99: pass | 1.62e-99 | 2.1e-99: pass | 2.03e-99 (dz0_weak's check) |
| recovered start max\|F\| | 9.0408e-36 | 9.041e-36 (logged 9.04e-36) | 1.4652e-35 | logged 1.47e-35 |
| σ₁…σ₄, subspace iteration | 2.1390e-27, 2.1387e-27, 3.1943e-26, 7.0800e-26 | the same | 2.1882e-27, 2.1857e-27, 3.3268e-26, 7.1880e-26 | the same |
| a = Vᵀdz0 | −5.149e-30, −5.814e-30, −2.989e-30, 2.297e-31 | the same | −8.165e-30, −8.547e-30, −4.486e-30, 2.894e-31 | the same |
| \|Ba\| | 9.828e-56 | 9.83e-56 | 1.529e-55 | 1.53e-55 |
| max\|T\|, h against h/10 | 1.631e33, 3.11e-18 | 1.631e33, 3.1e-18 | 1.618e33, 3.09e-18 | 1.618e33, 3.1e-18 |
| c\*, the root of model T | 7.24e-31, −3.05e-30, −3.16e-30, 1.07e-31 | the same | 1.0e-30, −2.97e-30, −4.8e-30, 2.91e-32 | the same |
| affine weak residual at c\*, of \|Ba\| | 0.00444 | 0.00444 | 0.00682 | 0.0068 |

### 5.2 The tensors

| quantity | REF | ACC |
|---|---|---|
| max\|Q\| at k = 4, h 10⁻³³ (Q(10⁻³²)/Q(10⁻³³)) | ≲ 6.4e-6 (99: pure h² truncation, constant 6.4e60) | ≲ 6.3e-6 (99) |
| \|w_jl\| | 0.058–0.144 | 0.058–0.144 |
| w_jl, 10⁻³² against 10⁻³³; 10⁻³³ against 10⁻³⁴ (bound 10⁻²) | 1.41e-2; 1.41e-4: pass | 1.4e-2; 1.4e-4: pass |
| max\|S\|, h against h/10 (bound 10⁻²) | 5.438e32, 1.7e-17: pass | 5.395e32, 1.7e-17: pass |
| S's unsymmetric part, of max\|S\| | 0.053 | 0.054 |
| **3max\|S\|/max\|T\|** | **1.000** | **1.000** |
| max\|T − 3S\|/max\|T\| | 1.6e-19 (the two step pairs differ by 1.6e-17) | 1.6e-19 (1.6e-17) |
| max\|L\|, L = UᵀF″(v_j, w0) | ≲ 4.3e-38 (h against h/10: 99, truncation) | ≲ 6.8e-38 (99) |
| \|q00\| = \|UᵀF″(w0, w0)\| | 1.9e-68 | 4.8e-68 |
| \|w0\| | 7.03e-33 | 1.13e-32 |

Every ≲ is an estimate: the difference at the smaller step, when the two steps differ by the factor 100 of an h²
truncation. The other ± here are the differences between two steps, not bounds.

### 5.3 The reduced map on the strong-solved manifold

| quantity | REF | ACC |
|---|---|---|
| \|g(0) + Ba\|/\|Ba\| | 2.7e-25 | 4.4e-25 |
| B_LS against B + L, of max\|B + L\| | 6.0e-13 | 9.5e-13 |
| max\|Q_LS\| | 2.6e-24 | 2.6e-24 |
| **\|g(a)\|/\|Ba\|** | **2.7e-25** | **4.5e-25** |
| max\|F\| at z0 + Va + w(a) | 2.0e-51 | 1.1e-50 |
| \|g(c\*)\|/\|Ba\| (affine) | 0.176 (0.0044) | 0.208 (0.0068) |
| model T: real roots; \|g\|/\|Ba\| at the nearest | 4; 0.176 | 4; 0.208 |
| model T-3S: real roots; c**; \|g(c**)\|/\|Ba\| | 1; c** = a to 3 digits; 7.9e-10 | 1; c** = a to 3 digits; 1.2e-9 |
| model T-3S,LS: \|g(c**)\|/\|Ba\| | 1.6e-19 | 3.6e-19 |

With the strong rows solved, the weak equations are affine to the measured precision: g(c) = g(0) + Bc, with g(0) =
−Ba, so the root is c = a, Newton's own first move. The cubic that #499 found on z0 + Vc is cancelled by the term 3S,
which comes from a strong correction of only |dw| = 1.3×10⁻⁶⁰ at c* through the mixed curvature UᵀF″(v, w): S/|w_jl|
is about 5×10³³ per unit of w, against UᵀF″(v, w0)/|w0| ≲ 6×10⁻⁶ along w0. (Numerical evidence.)

### 5.4 The S control, and what it measures

At REF's c*, replacing dw = w(c*) − w(0) by 2dw changes g by 1.684×10⁻⁵⁶, against −½S(c*, c*, c*) = 1.729×10⁻⁵⁶: a
difference of 2.72×10⁻² of it, so **(R2) fails** against 10⁻². At ACC the two are 3.076×10⁻⁵⁶ and 3.188×10⁻⁵⁶, a
difference of 3.55×10⁻². The w control (w → 2w) gives 2.66×10⁻² and 3.47×10⁻², since Lc* and (3/2)q00 (10⁻⁶⁷ and below)
are negligible. The work on REF was stopped there until the cause was known. The
diagnostic added for it repeats the control at f c*, f = 1, ½, ¼, and forms UᵀF″(V f c*, dw) by a mixed difference
with the actual dw (relative differences):

| start, f | 2dw change against −½S | 2dw change against UᵀF″(Vfc*, dw) | UᵀF″(Vfc*, dw) against −½S | UᵀF″(Vfc*, dw_quad) against −½S | dw against dw_quad |
|---|---|---|---|---|---|
| REF, 1 | 0.0272 | 0.0259 | 0.00176 | 1.9e-19 | 0.0104 |
| REF, ½ | 0.0169 | 0.0140 | 0.00353 | 1.9e-19 | 0.0209 |
| REF, ¼ | 0.0158 | 0.0092 | 0.00705 | 1.9e-19 | 0.0417 |
| ACC, 1 | 0.0355 | 0.0337 | 0.00210 | 2.0e-19 | 0.0140 |
| ACC, ½ | 0.0222 | 0.0183 | 0.00421 | 2.0e-19 | 0.0279 |
| ACC, ¼ | 0.0203 | 0.0122 | 0.00841 | 2.0e-19 | 0.0559 |

- **S's assembly.** UᵀF″(Vc, dw_quad), with dw_quad = −½Σw_jl c_j c_l, equals −½S(c, c, c) to 1.9×10⁻¹⁹. This shares
  the mixed-difference code with S, so it checks the tensor's symmetrization and contraction, not its differences.
- **dw's linear part.** dw against dw_quad doubles as f halves, so dw has a part linear in c, 1.0% (REF) and 1.4%
  (ACC) of |dw| at c*. Its weak coupling (column 4) doubles too. It is a quadratic term, 0.18% and 0.21% of −½S at c*,
  which the control counts and S, the quadratic part of w, does not contain. In g itself, Q_LS ≲ 2.6×10⁻²⁴, so that
  term is cancelled by others there (*inferred*).
- **The replacement's own orders.** Column 3 fits αf + β/f with α = 0.025, β = 0.0007 (REF) and α = 0.033,
  β = 0.0010 (ACC), to 2% (two parameters, three points). The next orders of the 2dw replacement,
  ½UᵀF‴(Vc, Vc, dw) and (3/2)UᵀF″(dw, dw), grow as f, and the coupling through w(0) falls as 1/f.
- **Together.** The control's error (column 2) fits αf + β/f with α = 0.025, β = 0.0022 (REF, to 5%) and α = 0.033,
  β = 0.0030 (ACC, to 2%). Its smallest value along c* is about 1.5% (REF) and 2.0% (ACC), near f = 0.3. So with the
  bound 10⁻² the control fails at every point of this ray at both starts. The cause is the control's own truncation
  at |c| of 10⁻³⁰ to 10⁻²⁹, not S.
- **The test of S that does not share the control's truncation** is the reduced map itself. If S were off by 1%, model
  T-3S's cubic would leave about 10⁻² of T(a, a, a)/6 ≈ |Ba| at a, and its root would not be a root of g. g at that root
  is 7.9×10⁻¹⁰|Ba| with the model's Q (its truncation), and 1.6×10⁻¹⁹ with Q_LS, and g(a) itself is 2.7×10⁻²⁵|Ba|. That
  test shares with S only the residual F and the basis, and uses the strong solve where S uses w_jl and mixed
  differences.

### 5.5 Verdicts against the protocol

| criterion | REF | ACC |
|---|---|---|
| (R0) offset on its rows; start as #499 | pass; 9.0408e-36 against 9.041e-36 | pass; 1.4652e-35 against the logged 1.47e-35 |
| (R1) S at h against h/10 (bound 10⁻²) | 1.7e-17: pass | 1.7e-17: pass |
| (R1) w_jl at 10⁻³³ against 10⁻³⁴ (bound 10⁻²) | 1.4e-4: pass | 1.4e-4: pass |
| (R2) S control at c\* (bound 10⁻²) | 2.72e-2: FAIL (section 5.4) | 3.55e-2: FAIL |
| (R3) 3max\|S\| < 10⁻²max\|T\| and \|g(c\*)\| < 10⁻²\|Ba\| | no: 3max\|S\| = max\|T\|, and \|g(c\*)\| = 0.176\|Ba\| | no: 3max\|S\| = max\|T\|, and \|g(c\*)\| = 0.208\|Ba\| |
| (R3) model T-3S's root c\*\* with \|g(c\*\*)\| < 10⁻²\|Ba\| | yes: c\*\* = a, 7.9e-10 | yes: c\*\* = a, 1.2e-9 |
| strong solves | 49, all converged | 49, all converged |

So by the issue's rule the coupling is this term, and the cubic with T − 3S has a root, c** = a.

## 6. Reading (inferred, not shown)

- **What the refused start needs.** Newton's first move already has the reduced root's weak coordinates, a. What it
  lacks is the strong rows' own correction at c = a (at c*, |dw| = 1.3×10⁻⁶⁰), which the chord iteration with z0's LU
  finds in three steps. After it, max|F| is 2.0×10⁻⁵¹ at REF and 1.1×10⁻⁵⁰ at ACC, below the chain's tolerance of
  10⁻⁴⁰. So at these starts the Lyapunov–Schmidt corrector of #530 needs one reduced step: the weak equation is affine
  on the strong-solved manifold, and its root is the linear one.
- **Why newton_od walks away.** From z0 + dz0 the remaining residual, 7.3×10⁻²⁶, is in the strong rows. A Newton step
  there treats the weak and strong rows together with J(z0 + dz0), whose weak block is about T(a, a, ·)/2 ≈ 10⁻²⁵,
  fifty times B. The cancellation T = 3S makes the step's weak part vanish only to the order the linearization
  keeps. #499's step from its weak-first start was 3.7×10⁻³⁰ on span(V) and nothing off it. That is the size that the
  next order, about 10% of the cubic at |a| (#499's u·F″″ ≈ 7.5×10⁶¹), would give after division by B. So each step
  moves the weak coordinates by a fraction of |a| that the strong-solved map does not ask for.
- **F4.** #499's weak-first step put the weak part at c* ≈ 0.53a, the root of the affine map. On the strong-solved
  map that point leaves 0.18|Ba|, so the step moved the weak part away from the root. The refusal that followed is
  what the reduced map predicts.
- **Why T = 3S.** No source read explains it. The reduced map being affine to 2.7×10⁻²⁵ means that, on the manifold
  where the strong rows hold, the weak rows' residual is linear in the weak coordinates to every order measured. A
  near-symmetry of the equations, broken only at the linear order B, would do that. This is a conjecture.

## 7. Discrepancies (reported, not fixed)

- **Q at k = 4 against #529's Q at k = 5.** #529's largest entries, 2.75×10⁴ (REF) and 5.4×10⁴ (ACC), were formed with
  the fifth direction, whose σ₅ does not converge. At k = 4, Q is below its h² truncation even at h = 10⁻³³ (≲ 6.4×10⁻⁶
  at REF). #529's |Ba| at k = 5, 1.36×10⁻³⁶ (REF) and 7.95×10⁻³⁷ (ACC), is 10¹⁹ times the k = 4 value, so its reduced
  system at k = 5 is not that of a converged four-dimensional subspace either (*inferred*). #529's raw JSON keeps only
  max|Q|, so which entries are large cannot be checked. notes/LARGE_D_WALL.md's "So Q is not zero" and the remark on
  #499 that "the reduced model for #528 and #530 must carry ½Q(c, c)" rest on the k = 5 tensor.
- **#499's root c\*.** LARGE_D_WALL.md's "At D 242.55 the root nearest the move is real, at about half of |a|" holds
  for the affine map on z0 + Vc. On the strong-solved manifold that point leaves 0.18|Ba| at REF, and the only real
  root is a.
- **The coupling through the first move's rest.** LARGE_D_WALL.md reads #499's split as the mixed curvature UᵀF″(Vc, x)
  raising the weak residual from 0.004 to 0.52 of |Ba|. Along w0, the strong part of the first move, that curvature
  is L ≲ 4.3×10⁻³⁸ (an estimate), so it gives at most 2×10⁻⁶⁷ at c*, not 5×10⁻⁵⁶. Repeated here in z0's frame (U and
  B from z0's LU), adding the first move's rest changes the weak residual at c* from 0.00444 to 0.00409 of |Ba| at REF,
  and from 0.00682 to 0.00620 at ACC; w0 and w(0) give the same. Only w(c*) moves it, to 0.176 and 0.208, by the S
  term. `hp/weak_corrector.py` forms the U and B of its split after newton_od has run (lines 128–129), from the LU of
  newton_od's last trial point, not z0's. That is the likely source of #499's 0.517 and 0.384 (*inferred* from the
  code, not rerun).
- **The S control fails** at REF (2.72×10⁻² against 10⁻²) and ACC (3.55×10⁻²). Section 5.4 gives its diagnosis.
- **This task's own protocol** changed four times after real data, each recorded with its cause (section 3): the
  step of w_jl twice (h² truncation amplified by J⁻¹), and the chord's stall rule twice (a stall test that cut a
  converging iteration, then one that stopped between two corrections that cancel). No bound was loosened.

## 8. Commands and raw data

From `hp/`, with a clone of the raw-data repository (`git clone https://github.com/nmoinvaz/choptuik-large-d-raw RAW`)
and main's hpkernel build on the path:

```
python3 reduced_cubic.py planted                     # check: pass, exit 0
python3 reduced_cubic.py planted --mutate=w0         # check: FAIL, exit 1 (also --mutate=Ssign, Sh, hF)
python3 reduced_cubic.py run RAW I461-win2/probeG03125b/dz0_pin3_D242.5459_a0.031250.npz \
    I461-win2/probeG03125a/anm_pin3_D242.5146.npz --threads=2 --json=ref.json
```

The real runs exit 1, because the S control fails (R2). `I528-mac/run_i528.sh` runs all of them. Raw: `I528-mac/` of
the raw-data repository at 625997f9, with every run's log and JSON, the stopped runs included (its README lists
them).

## 9. Where the reduced map stops being affine: rays in |c| (#677)

The strong-row chord of #674 keeps Newton's first move a in the weak coordinates and solves only the strong rows. It
relies on g being affine from 0 to a. This section measures how far that holds, along rays out to 10³|a| and beyond,
at four starts. Sections 9.1 to 9.3 are the protocol, committed before the ray plant's first run and before any real
ray run.

### 9.1 Method (`reduced_cubic.py ray`)

- **The basis.** V, U, B = UᵀJV, a and w0 as in section 1, from the same code.
- **The rays.** g on the strong-solved manifold at c = ±t e, with e = a/|a| (judged) and two random unit vectors r1, r2
  (seed 7, diagnostic). t runs over 10^(n/4) from 10⁻³⁰ to 10⁻²⁴, 25 points. The issue's range, 10⁻²⁹ to 10⁻²⁶, is
  judged and the rest is diagnostic.
- **The strong solve.** Section 1's chord through z0's LU, now run to |dw| < 10⁻⁶⁰|w0|, or until the smallest |dw| so
  far has not fallen by 0.3 in three steps (the chord's floor). The iterate with the smallest |dw| is kept, and it
  counts as converged when that |dw| is below section 1's 10⁻⁴⁰|w0|. A step may grow |dw|. At |c| ≫ |a| the chord's
  error map J⁻¹(I − UUᵀ)F″(Vc, ·) can have a norm above 1 and a square far below it, as at #528's c* (|dw|
  5.44×10⁻⁵⁶, 5.44×10⁻⁵⁶, then 5×10⁻⁸²). A solve fails on a non-finite |dw|, on |dw| above 10⁻³ max(|c|, |w0|), or
  when no iterate in 40 steps gets below 10⁻⁴⁰|w0|. Each t starts from w0 + (t/t′)²(w(t′) − w0), with t′ the ray's
  previous point. A ray stops at its first failed solve, and that |c| is reported. (This rule and the second start of
  floor0 below replaced the first ones after the REF pilot, section 9.4.)
- **The departure.** dep(c) = g(c) − g(0) − Bc, with the basis's B. The departure from B(c − a), the issue's affine
  part, is dep(c) + g(0) + Ba, and the constant ε0 = |g(0) + Ba|/|Ba| is reported (2.7×10⁻²⁵ at REF).
- **Its parts.** The even part E(t) = (dep(te) + dep(−te))/2 holds the orders 2, 4, …, and the odd part
  O(t) = (dep(te) − dep(−te))/2 the orders 1, 3, …. An error in B shows as order 1 in the odd part.
- **The floor of each t.** The largest of three things. Each sign's strong-solve error estimate,
  |Uᵀ[F(w − dw) − F(w)]|, is what the last correction, not applied, would change in g. floor0 is the difference of
  g(0) between solves from w0 and from w0(1 + 10⁻⁵), together with both their estimates. The floor is
  an estimate, not a bound, since the next correction can be larger by the error map's norm. A part counts as
  resolved at t where it exceeds 100 floors.
- **The order.** The leading order p and constant K of each part, |part| ≈ K|Ba|(t/|a|)^p, come from a least-squares
  line through ln|part| against ln t at its first four resolved points (the smallest t). The local order between every
  two neighbouring resolved points is printed too, so a change of regime shows. The fit's rms in ln is printed. It is a
  measure of the fit, not an error bar on p.
- **The crossing.** The first t at which |dep(±te)| reaches 10⁻²|Ba|, by log-log interpolation between neighbouring
  points. If it is not reached, a power-law extrapolation from the last two points is given, labelled as such. The same
  is printed for |g(c) − B(c − a)|.
- **c = a.** |g(a)|/|Ba|, |dep(a)|/|Ba| and max|F| at z0 + Va + w(a), the point #674's chord aims for.
- **With `--tensors=1`.** The whole of `run` runs first (section 1's tensors, models and controls). The tensors'
  cubic along e, |(T − 3S)(e, e, e)|/6 · |a|³/|Ba|, at both step pairs (their difference shows its resolution), is the
  constant that the odd part's K would have if its order were 3.

### 9.2 The plant (`reduced_cubic.py ray planted`)

RayPlanted is section 2's plant with Q = 0, L = 0 (K zero on x's rows 0–3, where w0 lives), M = 0 and T = 3S, and with
a planted departure. Its reduced map is then exactly

```math
g(y) = \beta + \Sigma y + \tfrac1{24} P_4(y,y,y,y) + \tfrac1{120} P_5(y,y,y,y,y), \qquad P_{4,m} = p_m (a_m \cdot y)^4, \quad P_{5,m} = q_m (b_m \cdot y)^5.
```

- β = −Σy_a with |y_a| = 8.3×10⁻³⁰, REF's |a|. P₄ is about 10⁻²|Ba| near |c| = 20|a|, and P₅ equals P₄ near
  |c| = 10⁻²⁶. So the even part has order 4 and the odd part order 5, and the crossing lies inside the judged range.
- Section 2's quartic c₄ stays in strong row 0, where K is now zero, so it reaches w_jl but not g.
- **(PY0)** The plant checks its own construction: G_w(y, x(y)) − β − Σy against P₄/24 + P₅/120 at |y| = 10⁻²⁹ and
  10⁻²⁷, to 10⁻²⁰. This is where T = 3S is tested.

Pass criteria, fixed before the first run:
- **(PY1)** At every point and in both parts, |measured − planted| < 10⁻³ max(|planted|, 100 floors).
- **(PY2)** Along every ray, the leading orders are 4 (even) and 5 (odd), to 0.05.
- **(PY3)** Along every ray and sign, the crossing of 10⁻²|Ba| is within 10⁻² (relative) of the planted one, found
  by bisection on the planted departure.
- **Controls.** `--mutate=w0` drops the strong solve, so g is the map on z0 + Vc, whose odd part carries T(c, c, c)/6
  (order 3): (PY1), (PY2) and (PY3) must fail. `--mutate=B` scales B by 1 + 10⁻¹² in dep, an order-1 odd term: (PY1)
  and (PY2) must fail. A failed check exits 1.

### 9.3 The real starts

Every start is analysed on main's 3c35f0c kernel with the exact projector (`i599/site2`). The recovered offset
absorbs the projector's difference in the frozen rows (section 3).

| start | DZ0 (raw-data repository) | BASE | made on |
|---|---|---|---|
| REF, D 242.5459 (refused) | `I461-win2/probeG03125b/dz0_pin3_D242.5459_a0.031250.npz` | `I461-win2/probeG03125a/anm_pin3_D242.5146.npz` | win2, before #523: rounded projector |
| ACC, D 242.5225 (accepted) | `I461-win2/probeG0078/dz0_pin3_D242.5225_a0.007812.npz` | the same | win2, rounded projector |
| D 232.13, win_d10 (refused) | `I513-win/run_d10/dz0_pin3_D232.1312_a1.000000.npz` | `I462-win/chain_d10/anm_pin3_D231.1312.npz` | win, venv-t151 (#462's build, before #523): rounded projector |
| class A, D 249.39 (refused) | `I677-mac/classA/dz0_pin3_D249.3927_a0.762939.npz` | `I603-win2/chain/anm_pin3_D248.6298.npz` | made again here on `i599/site2` (exact projector); #603 itself ran on venv-i600, also exact |

- **The class A start.** #644's `I644-win2/own/` holds that attempt's prediction before its re-seed, not the
  corrector's start and first step, which `run` and `ray` need. So the start is made again on the Mac with
  `anm_run.py --steps 1 --save_dz0 1`, #603's flags and 2 threads (`I677-mac/run_classA.sh`), and stopped after its
  first Newton step.
- **(Y0)** As (R0), the held offset is on its rows. For REF and ACC, the basis line (σ, a, |Ba|) must equal #528's to
  its printed digits (the same code and inputs). For the class A start, anm_run's series, rung and first Newton lines
  must equal #603's (`I644-win2/own.log` lines 3 to 10, the base polish to the first Newton step, which #644 found
  identical to #603's chain.log), or the difference is reported.
- **(Y1)** At D 232.13 and the class A start, `--tensors=1`, with #528's (R1) h checks judged: w_jl at 10⁻³³ against
  10⁻³⁴, and S at h against h/10, both to 10⁻². For REF and ACC, #528's (R1) passed with the same code and inputs
  (`I528-mac/`) and is not rerun. #528's S control (R2) is reported, not judged, since its failure at REF and ACC is the
  control's own truncation (section 5.4). If an h check fails with the factor 100 of an h² truncation, the tensors are
  rerun once with their steps one decade lower. Any other failure is reported, and the ray result, which does not use
  the tensors, stands.
- **(Y2)** Along e = a/|a|, every strong solve on the judged range converges, at both signs. Otherwise the |c| where the
  ray stops is reported, and the order is fitted below it.
- **(Y3)** Along e, at least one part has a fitted leading order (three or more resolved points). Otherwise it is
  reported as not resolved, with the largest |dep| against its floor.
- **The verdict per start.** The leading order and K of each part, the local orders, the |c|/|a| at which |dep| reaches
  10⁻²|Ba| (measured, extrapolated, or not reached), and |dep(a)|/|Ba|.
- **Order and stops.** The plant and its controls first, then REF as the pilot, ACC, D 232.13 and the class A start. A
  failed plant or control stops the real runs. A control that fires on real data stops the work for that start until
  its cause is known.
- **Diagnostics, not judged.** The rays r1 and r2, max|F| and the chord's steps at each point, ε0, and the tensors'
  cubic against the odd part's K.

### 9.4 Changes after real data

- **The REF pilot, first run** (code 99ca8e3b, raw `I677-mac/ray_ref_pilot1_99ca8e3b.log`). The basis line equals
  #528's to every printed digit, so (Y0) passes. Then a strong solve at c = 0 failed, and the scan stopped before any
  ray. The first rule had two parts that could fail there. The second start of floor0 was w0 + ξ, with ξ random and
  orthogonal to V, |ξ| = 10⁻⁴|w0|. Its square reaches the strong rows through F″, and J⁻¹ multiplies that along the
  softest strong direction (σ₅ ≈ 10⁻¹⁹) by 10¹⁹: about 10³³ × (7×10⁻³⁷)² / 10⁻¹⁹ ≈ 5×10⁻²¹, far above the failure
  bound of 7×10⁻³⁶ (*inferred*: the run did not log the steps). And the stall test needed two consecutive steps
  without a cut, which a fluctuating floor need not give. The stop rule now keeps the best iterate (9.1), and failed
  solves at c = 0 print their steps. The second start was first w0(1 + 10⁻³), and the plant's rerun (c4496cb1) failed
  on it: its first correction, 10⁻³|w0|, sat on the failure bound 10⁻³|w0|. It is now w0(1 + 10⁻⁵). The plant and its
  controls were rerun with the final rule. No pass criterion changed.
- **`raysum`, added after the runs.** It reads the runs' JSON files and prints the orders, constants and crossings with
  estimates of the next orders, K3 for a cubic in the odd part and K4 for a quartic in the even part. These come from
  the relative change ρ of |O|/t and |E|/t² across the resolved range: ρ for a next order along the leading one, and
  √(2ρ) for one orthogonal to it, whichever is larger. ρ is floored at the 16 stored digits, 2.2×10⁻¹⁶. Its first
  version took E and O from the stored dep vectors, which are floats. Where one part exceeds the other by more than 16
  digits, as at D 232.13 and the class A start, those half sums are rounding. The estimates are diagnostics and
  judge nothing.

### 9.5 The plant's results

`python3 reduced_cubic.py ray planted` from `hp/` (code 321954dc; raw `I677-mac/ray_planted*.log`):

| check | plant | `--mutate=w0` | `--mutate=B` |
|---|---|---|---|
| (PY0) its own construction against P₄/24 + P₅/120 (bound 10⁻²⁰) | 5.3×10⁻¹³⁶: pass | | |
| (PY1) worst point, both parts (bound 10⁻³) | 3.6×10⁻⁵¹: pass | 2.5×10¹¹: FAIL | 24: FAIL |
| (PY2) leading orders, even and odd, along a | 4.000, 5.000: pass | 4.000, 3.000: FAIL | 4.000, 3.17: FAIL |
| (PY3) crossing of 10⁻²\|Ba\| along a, measured against planted, in \|a\| | 9.7138 against 9.7138: pass | 0.241: FAIL | 9.7138: pass (not required to fail) |
| exit | 0 | 1 | 1 |

All three rays pass (PY1) and (PY2). The worst crossing error over the rays and signs is 0.87%, on r2 (17.44 against
17.29), within the bound 10⁻². Under `--mutate=B` the odd part's local orders run 1.16 → 3.35 → 4.93 → 5.00 along a,
so the fit sees the planted order-1 term and then P₅.

### 9.6 Results at the four starts

Raw `I677-mac/ray_{ref,acc,d232,classA}.log` and `.json`, and `raysum.txt` (`reduced_cubic.py raysum`, 5f755c56). REF
ran with code 321954dc and the others with 11c31fbd, whose ray mode is the same. dep(c) = g(c) − g(0) − Bc. Along
e = a/|a|, in units of |Ba| with K as in |part| ≈ K(|c|/|a|)^p:

| start | weak σ | \|a\|, \|Ba\| | even part: order, K | odd part: order, K | K3, K4 (estimates) | 10⁻²\|Ba\| at \|c\|/\|a\|: dep; even part alone |
|---|---|---|---|---|---|---|
| REF, D 242.5459 | 2.1×10⁻²⁷ … 7.1×10⁻²⁶ | 8.32×10⁻³⁰, 9.83×10⁻⁵⁶ | 2.000, 3.750×10⁻²⁹ | 1.000, 5.303×10⁻³⁰ | ≲ 7.7×10⁻⁴⁸, ≲ 7.7×10⁻⁴⁷ | 1.6×10¹³; 1.6×10¹³ (both extrapolated) |
| ACC, D 242.5225 | 2.2×10⁻²⁷ … 7.2×10⁻²⁶ | 1.26×10⁻²⁹, 1.53×10⁻⁵⁵ | 2.000, 5.584×10⁻²⁹ | 1.000, 8.001×10⁻³⁰ | ≲ 2.7×10⁻⁴⁷, ≲ 1.9×10⁻⁴⁶ | 1.3×10¹³; 1.3×10¹³ (extrapolated) |
| D 232.13, win_d10 | 2.1×10⁻²³ … 5.0×10⁻²¹ | 1.43×10⁻²⁹, 1.79×10⁻⁵⁰ | 2.000, 1.758×10⁻²⁸ | 1.000, 7.420×10⁻¹⁴ | ≲ 3.2×10⁻³¹, ≲ 7.6×10⁻⁴⁶ | 1.35×10¹¹ (extrapolated, order 1); 7.5×10¹² |
| class A, D 249.3927 | 2.4×10⁻³⁰ … 8.2×10⁻³⁰ | 8.02×10⁻³⁰, 5.70×10⁻⁵⁹ | 2.000, 1.892×10⁻²⁸ | 1.000, 7.882×10⁻⁷ | ≲ 1.1×10⁻²⁴, ≲ 4.4×10⁻⁴⁴ | **1.27×10⁴ (measured)**; 7.3×10¹² |

| start | ε0 = \|g(0) + Ba\| | \|g(a)\| | \|dep(a)\| | max\|F\| at z0 + Va + w(a) | floor0 | chord steps | judged points converged |
|---|---|---|---|---|---|---|---|
| REF | 2.72×10⁻²⁵ | 2.72×10⁻²⁵ | 4.27×10⁻²⁹ | 7.3×10⁻⁷⁷ | 3.0×10⁻⁴⁶ | 5–8 | all, and every point to 10⁻²⁴ |
| ACC | 4.39×10⁻²⁵ | 4.39×10⁻²⁵ | 6.38×10⁻²⁹ | 9.0×10⁻⁷⁶ | 2.1×10⁻⁴⁵ | 5–8 | all |
| D 232.13 | 8.49×10⁻¹⁶ | 7.42×10⁻¹⁴ | 7.42×10⁻¹⁴ | 1.7×10⁻⁶⁴ | 6.7×10⁻⁵¹ | 5–7 | all |
| class A | 1.42×10⁻¹⁰ | 7.88×10⁻⁷ | 7.88×10⁻⁷ | 5.4×10⁻⁶⁶ | 5.3×10⁻³⁹ | 5–9 | all |

- **The orders.** At every start and on every ray, the even part has local order 2.00 and the odd part 1.00 between
  every two neighbouring points, from 10⁻³⁰ to 10⁻²⁴ (six decades, up to 1.2×10⁵|a| at REF). No cubic or quartic is
  resolved. The random rays r1 and r2 give the same orders, with even K 1.3 to 29 times that along a.
- **The odd order-1 term** is the difference between the map's slope and B. At D 232.13 it is L c: L = UᵀF″(v, w0)
  is resolved there (h against h/10 0.25%), max|L||a|/|Ba| is 8.7×10⁻¹⁴ against the measured 7.4×10⁻¹⁴, and #528's
  model with L leaves 4.3×10⁻¹⁷ at its root against 7.4×10⁻¹⁴ without it. At the class A start the map's own slope
  B_LS accounts for it: model T-3S,LS leaves 1.2×10⁻¹⁴. L is not resolved there (h against h/10 3.3), but its size,
  max|L||a|/|Ba| = 9.6×10⁻⁷, is that of the measured 7.9×10⁻⁷ (*inferred*). At REF and ACC it is B's numerical error
  or L, whose bound there (L ≲ 4.3×10⁻³⁸, section 5.2) is far above it, so the two cannot be separated.
- **So, with the strong rows solved, the weak equations are affine with slope B + L** to the order-2 term. That term
  reaches 10⁻²|Ba| only near 10¹³|a| (an extrapolation of a pure t² law measured to 10⁵|a|).
- **The issue's affine part B(c − a)** differs from that by ε0 and L. At the class A start the departure from B(c − a)
  reaches 10⁻²|Ba| at |c| = 1.27×10⁴|a|, measured, and it is linear in |c|. At c = a it is 7.9×10⁻⁷|Ba|.
- **The cubic.** K3 converts to |(T − 3S)(e, e, e)|/max|T| ≲ 2.6×10⁻²⁵ at D 232.13 and ≲ 3.6×10⁻²⁹ at the class A
  start (raysum, with each run's max|T|). The tensors' own cubic along e, |(T − 3S)(e, e, e)|/6 · |a|³/|Ba|, is
  1.05×10⁻²⁶ (D 232.13) and 1.2×10⁻¹⁴ (class A) at the smaller steps and 100 times that at the larger ones, so it is the
  differences' h² truncation, not a value. Against K3 ≲ 3.2×10⁻³¹ and ≲ 1.1×10⁻²⁴, the rays bound the reduced cubic
  4 to 10 orders below what the tensors resolve. At REF, K3 ≲ 7.7×10⁻⁴⁸ against max|T − 3S| = 1.6×10⁻¹⁹max|T|
  (section 5.2) says the same far more strongly.

The tensors at the two new starts (`--tensors=1`, section 1's measurement):

| quantity | D 232.13 | class A |
|---|---|---|
| (R1) w_jl at 10⁻³² against 10⁻³³; 10⁻³³ against 10⁻³⁴ (bound 10⁻²) | 1.0×10⁻⁴; 1.0×10⁻⁶: pass | 0.334; 3.3×10⁻³: pass |
| (R1) S at h against h/10 (bound 10⁻²) | 6.1×10⁻¹⁸: pass | 5.0×10⁻¹⁷: pass |
| max\|T\|, h against h/10 | 4.457×10³¹, 1.2×10⁻¹⁸ (#499's value) | 1.994×10³⁴, 6.1×10⁻¹⁸ |
| **3max\|S\|/max\|T\|** | **1.000** | **1.000** |
| max\|T − 3S\|/max\|T\| (the two step pairs differ by) | 5.7×10⁻²⁰ (5.7×10⁻¹⁸) | 4.7×10⁻¹⁹ (4.7×10⁻¹⁷) |
| max\|Q\| at h 10⁻³³ (Q(10⁻³²)/Q(10⁻³³)) | ≲ 6.6×10⁻⁸ (99) | ≲ 1.4×10⁻⁴ (99) |
| max\|L\| (h against h/10) | 1.10×10⁻³⁴ (0.0025) | 6.8×10⁻³⁶ (3.3, not resolved) |
| \|B_LS − (B + L)\|/max\|B + L\| | 5.6×10⁻¹⁹ | 2.8×10⁻⁸ |
| (R2) S control at c\*, reported (section 5.4) | 0.81% | 1.31% |
| \|g(c\*)\|/\|Ba\|, #499's root (affine) | 4.2×10⁻⁷ (4.3×10⁻⁹) | 0.939 (0.0048) |
| model T-3S, T-3S,L, T-3S,LS: \|g(c\*\*)\|/\|Ba\| | 7.4×10⁻¹⁴, 4.3×10⁻¹⁷, 1.1×10⁻²⁶ | 4.1×10⁻⁵, 4.2×10⁻⁵, 1.2×10⁻¹⁴ |

At the class A start the models with Q leave 4×10⁻⁵, against 7.9×10⁻⁷ at c = a itself. Their Q is the differences'
truncation (the h² ratio 99), and ½max|Q||a|² is of that size (*inferred*). Its S control scan along c* repeats REF's
pattern: dw departs from its quadratic model by 0.43%, 0.86% and 1.7% at f = 1, ½ and ¼.

### 9.7 Verdicts against the protocol

| criterion | REF | ACC | D 232.13 | class A |
|---|---|---|---|---|
| (Y0) offset on its rows; the start as recorded | pass; basis = #528's | pass; basis = #528's | pass; basis = #499's (σ, a, \|Ba\| 1.79×10⁻⁵⁰) | pass; anm_run's lines = #603's |
| (Y1) #528's h checks | #528 (not rerun) | #528 (not rerun) | pass | pass |
| (Y2) every judged solve along e converged | yes (to 10⁻²⁴) | yes | yes | yes |
| (Y3) a fitted leading order along e | even 2, odd 1 | even 2, odd 1 | even 2, odd 1 | even 2, odd 1 |
| exit | 0 | 0 | 0 | 0 |

### 9.8 Reading (inferred, not shown)

- **The premise of #674 holds at all four starts.** At c = a, Newton's first move, the strong-solved weak residual is
  2.7×10⁻²⁵, 4.4×10⁻²⁵, 7.4×10⁻¹⁴ and 7.9×10⁻⁷ of |Ba|. With the chord run to its floor, max|F| there is 7×10⁻⁷⁷,
  9×10⁻⁷⁶, 2×10⁻⁶⁴ and 5×10⁻⁶⁶, below the chain's 10⁻⁶⁰ at every start. At the class A start, where the full step of
  newton_od lands far outside its basin (#644, #655), the reduced map is affine to 7.9×10⁻⁷ at c = a. The weak residual
  left after the chord is that fraction of |Ba|, and a Newton step after it moves the weak coordinates by a fraction
  of |a| of that order (*inferred* from the slope).
- **What separates the starts** is the linear term L, the coupling through the start's own strong correction w0, not
  a nonlinearity. At c = a it is 5×10⁻³⁰ (REF, or B's error), 7×10⁻¹⁴ (D 232.13) and 8×10⁻⁷ (class A) of |Ba|. The
  class A start's weak σ are about 10³ times smaller than at D 242.5, with |w0| of the same size.
- **T = 3S to all measured orders.** At four starts the map has no cubic and no quartic above the estimates of 9.6,
  far below the tensors' resolution. So the cancellation that #528 found at two starts is exact to the ray's precision,
  along a and along random directions. A near-symmetry that leaves the weak rows affine on the strong-solved manifold,
  broken at the linear order B and the quadratic order Q, would do this. This is a conjecture.

### 9.9 Discrepancies (reported, not fixed)

- **#528's S control fails again at the class A start** (1.31% against 10⁻²) and passes at D 232.13 (0.81%). As at REF
  and ACC, the scan along c* shows the control's own truncation (dw's linear part doubles as f halves). Here it is
  reported, not judged, by 9.3's protocol.
- **The class A start's tensors on #528's steps.** Its w_jl at 10⁻³² against 10⁻³³ differ by 0.334, against 0.014 at
  REF. The judged pair (10⁻³³ against 10⁻³⁴, 3.3×10⁻³) passes with the h² ratio 100. So the steps chosen at D 242.5
  are near their limit at the class A start's weaker σ.
- **#499's root c\* at the class A start** leaves 0.94|Ba| on the strong-solved manifold, against 0.0048 on the affine
  subspace: the same misreading as at REF (section 7), larger here.

### 9.10 Commands

From `hp/`, with a clone of the raw-data repository at `RAW` and main's hpkernel build on the path:

```
python3 reduced_cubic.py ray planted                     # check: pass, exit 0
python3 reduced_cubic.py ray planted --mutate=w0         # check: FAIL, exit 1 (also --mutate=B)
python3 reduced_cubic.py ray RAW I461-win2/probeG03125b/dz0_pin3_D242.5459_a0.031250.npz \
    I461-win2/probeG03125a/anm_pin3_D242.5146.npz --threads=2 --json=ray_ref.json
python3 reduced_cubic.py raysum ray_ref.json             # the orders, constants, crossings and estimates
```

`I677-mac/run_i677.sh` runs all of them (D 232.13 and the class A start with `--tensors=1`), and `run_classA.sh` makes
the class A start. REF takes about 16 minutes at 2 threads, and a start with the tensors 25 to 47 minutes. Raw:
`I677-mac/` of the raw-data repository at a8de27c, with every run's log and JSON (its README lists them).

## 10. Where the weak rows live (#694)

With the strong rows solved, the weak equations are affine at four starts (section 9). #694 tests one explanation:
that the weak left vectors U live on rows of F that are affine in z, so that U^T F has no curvature to begin with.
Sections 10.1 to 10.3 are the protocol, committed before the plant's first run and before any real run.

### 10.1 Method (`reduced_cubic.py rows`)

- **The rows.** The residual has K + 1 blocks of 4M rows (`hp/src/model.rs`): block 0 is the centre series at node 0
  minus node 0 ("centre"), block b the Taylor step from node b − 1 to node b minus node b ("node b"), block K the
  horizon series minus node K − 1 ("horizon"). Each block holds the fields p, m (odd) and a, c (even) on the M points
  of the half grid, and the last row is the phase row. Each field's M rows are expanded on its harmonics, an
  orthonormal change of rows (cos and sin of k = 1, 3, …, M − 1 for an odd field; cos of k = 0 and M, cos and sin of
  k = 2, …, M − 2 for an even one). A row is then labelled block.field.harmonic, such as horizon.p.k5c.
- **The rows linear by construction.** The horizon block's p rows above kfreeze are multishoot's projected rows,
  Π′p1 minus the held offset, and the phase row is Σ w_j p1_j. With kfreeze = 3 these are horizon.p.k5 to k31 (cos
  and sin) and phase, 29 rows.
- **U's share.** With U from section 1's basis, a row's share is Σ_m U_mr²/k, which does not depend on the basis of
  span(U), and the shares sum to 1. The share of a set of rows is the sum of their shares. The smallest eigenvalue of
  U_Aᵀ U_A, the share of the worst unit vector of span(U), is printed beside it.
- **Each row's curvature along V.** q_r = max over j ≤ l of |F″_r(v_j, v_l)|, from central second differences at
  h = 10⁻³² and 10⁻³³ (#528's Q steps), judged at 10⁻³³. Their ratio is printed for each row: about 1 where F″ is
  resolved, 100 for an h² truncation and 0.01 for rounding.
- **Affine to 10⁻¹⁰.** A row is affine when q_r is at most 10⁻¹⁰ of the largest q_r over all rows and at most 10⁻¹⁰ of
  its own |J_r| (the 2-norm of its row of J, in f64). The first is the issue's reading. The second stops a row from
  counting as affine only because its scale is small, since the rows' scales differ by 10⁵³. The shares under each
  alone are printed too.
- **The rows carrying the rest.** The rows not affine, by share, and their shares summed by block and field.
- **Where V lives** (diagnostic). The same split of the weak right vectors over the unknowns (p1, c0 | node i: p, m, a,
  c | m_h, a_h | Δ), by field and harmonic.
- **The relation to 3S = T, along e = a/|a|.** Three vectors per row: F″(e, e) (central differences at 10⁻³³, the step
  of Q), T(e) = F‴(e, e, e) (four-point differences at 10⁻²³ and 10⁻²⁴, T's steps) and S(e) = F″(e, w_ee), with
  w_ee = J⁻¹(I − UUᵀ)F″(e, e) from the differences at 10⁻³³ and 10⁻³⁴ (w_jl's steps) and mixed differences at 10⁻³² and
  10⁻³³ (S's steps). Then Uᵀx = Σ_r U_r x_r = (Q(e, e), T(e, e, e) and 3S(e, e, e) of section 1), its part on the
  affine rows and on the rest, and the size it would have without cancellation between rows, √(Σ_m (Σ_r |U_mr x_r|)²).
  If the conjecture holds, the affine rows contribute nothing to any of them, and T and 3S come only from the rest.
  If it fails, the ratio of |Uᵀx| to its no-cancellation size shows whether Q = 0 and T = 3S come from cancellation
  between rows that are not affine.

### 10.2 The plant (`reduced_cubic.py rows planted`)

RowPlanted is a system on the residual's row layout with M = 4 and K = 1: the centre and horizon blocks of four fields
each and the phase row, n = 33. Its rows are given in its own harmonic basis, written independently of the code's, and
its unknowns are rotated by a random orthogonal matrix. In the harmonic rows,

```math
G(\zeta) = \beta + D\zeta + \sum_{r \notin A} e_r \left[\tfrac12 q_r (b_r\cdot\zeta)^2 + \tfrac16 t_r (b'_r\cdot\zeta)^3\right], \qquad D = U_c \Sigma E_w^{T} + U_c' \Lambda E_s^{T}.
```

- The planted affine rows A are centre.m.k1c, horizon.p.k3c, horizon.p.k3s and phase, which carry U, and horizon.a.k0c,
  which does not. They have no nonlinear term, so they are affine in z exactly.
- The planted weak left vectors are u_m = √(1 − ε_m²) a_m + ε_m n_m, with a_m an orthonormal basis of A's four rows
  that carry U (randomly rotated), n_m orthonormal on the nonlinear rows, and ε = 0.1, 0.15, 0.2, 0.25. Their share on
  A is 1 − mean(ε²) = 0.96625.
- Σ is the D 242.5 starts' weak σ, Λ 29 strong singular values log-spaced from 10⁻¹⁹ to 3×10³⁴, and |q_r| (10²⁸ to
  10³³) and |t_r| (10³⁰ to 10³⁴) log-uniform.

Pass criteria, fixed before the plant's first run:
- **(PR1)** The rows found affine are exactly A, by label.
- **(PR2)** The share on them equals 0.96625 to 10⁻⁶ (relative), and every row's share equals the planted one to
  10⁻⁶ of the largest.
- **(PR3)** q_r on the nonlinear rows equals the planted one, along the code's V, to 10⁻⁶ (relative).
- **(PR4)** F″(e, e), T(e) and S(e) per row equal the planted ones to 10⁻⁶ of each one's largest entry, and the affine
  rows' part of Uᵀx is below 10⁻¹⁰ of its no-cancellation size for all three.
- **The harmonic basis** is orthonormal to 10⁻¹²⁰.
- **Controls.** `--mutate=quadA` adds ½q_A(b·ζ)², q_A = 10⁻⁶ max|q_r|, to A's four rows that carry U: (PR1) and (PR2)
  must fail. `--mutate=parity` expands every field on the harmonics of the other parity, which keeps the change of rows
  orthogonal but labels each row wrongly: (PR1) and (PR2) must fail. A failed check exits 1.

### 10.3 The real starts (fixed before the first real run)

The four starts of section 9.3, on main's 3c35f0c kernel (exact projector, `i599/site2`), each with section 1's
basis.
- **(W0)** As (Y0): the held offset is on its rows, and the basis line (σ, a, |Ba|) equals #677's to its printed
  digits.
- **(W1)** The 29 rows linear by construction are all classified affine. This is a control on real data: if it fails,
  the classification or the labels are wrong, and the start's result is not used until the cause is known.
- **(W2)** The change of rows is orthonormal to 10⁻¹²⁰, and Uᵀ T(e) summed over the harmonic rows equals the sum over
  the grid rows to 10⁻¹⁰⁰ of its no-cancellation size.
- **(W3)** w_ee at 10⁻³³ against 10⁻³⁴, Uᵀ T(e) at 10⁻²³ against 10⁻²⁴ and Uᵀ S(e) at 10⁻³² against 10⁻³³ agree to
  10⁻², and |Uᵀ(T − 3S)(e)| < 10⁻¹² |Uᵀ T(e)|, the consistency with #528 and #677. A failure is reported as a
  discrepancy, and the share (which does not use these) stands.
- **(W4) The verdict per start.** The share of |U| on the affine rows at h = 10⁻³³: above 0.99 supports the conjecture
  of affine weak rows, below 0.5 refutes it, and between is reported as it is. The same at 10⁻³²; a different verdict
  there is reported as step-dependent.
- **Reported, not judged.** The rows carrying the rest by block, field and harmonic; V by unknowns; the parts of
  Q(e, e), T(e, e, e) and 3S(e, e, e) on the affine rows and on the rest, and their no-cancellation sizes.
- **Order and stops.** The plant and its two controls first, then REF as the pilot, ACC, D 232.13 and the class A
  start. A failed plant or control stops the real runs, and so does (W1) failing at REF.

### 10.4 Changes after the first runs

- **The planted share.** 10.2 and the docstring gave 1 − mean(ε²) as 0.964375. The plant's first run (b6abb010)
  computed 0.96625, which is 1 − 0.135/4, and the text now says so. No criterion changed.
- **Diagnostics after the REF pilot** (8cbd3f31, raw `rows_ref_pilot_8cbd3f31.log`). The pilot found U on the last two
  blocks. Code 848f7c06 adds four diagnostics, none judged: each block's part of Uᵀx against its no-cancellation size;
  the pairing |u(node K − 1 rows) + u(horizon rows)|/|u|; w_ee by unknowns; and |P_V e_Δ|², how much of the Δ axis
  lies in span(V). The plant, its controls and all four starts were rerun on 848f7c06. REF's judged lines equal the
  pilot's. `rowsum` (9a7455f4) prints the summary below from the runs' JSON files.

### 10.5 The plant's results

`python3 reduced_cubic.py rows planted` from `hp/` (code 848f7c06; raw `I694-mac/rows_planted*.log`):

| check | plant | `--mutate=quadA` | `--mutate=parity` |
|---|---|---|---|
| (PR1) the affine rows found against the planted five, by label | equal: pass | only horizon.a.k0c: FAIL | only phase, labels wrong: FAIL |
| (PR2) share 0.96625, every row's share (bound 10⁻⁶) | 1.4×10⁻⁶³, 6.2×10⁻⁶³: pass | share 2.4×10⁻¹²⁴: FAIL | share 0.240, rows 0.88: FAIL |
| (PR3) q_r against the planted (bound 10⁻⁶) | 1.8×10⁻¹⁰⁴: pass | pass (not required to fail) | 1.9×10⁴: FAIL |
| (PR4) F″(e, e), T(e), S(e) per row; the affine rows' part | 6×10⁻¹⁰⁶, 4×10⁻¹⁰³, 5×10⁻⁵⁵; 7×10⁻¹⁰³: pass | pass | 1.5, 0.71, 1.7: FAIL |
| the harmonic basis orthonormal (bound 10⁻¹²⁰) | 3.9×10⁻¹⁴¹ | | |
| exit | 0 | 1 | 1 |

The first runs (b6abb010) gave the same lines.

### 10.6 Results at the four starts

Raw `I694-mac/rows_{ref,acc,d232,classA}.log` and `.json` (code 848f7c06) and `rowsum.txt` (`reduced_cubic.py rowsum`,
9a7455f4). Every start has K = 5 nodes and M = 32, so L = 769. Block 4 ("node4") is the Taylor step from node 3 to node
4 minus node 4, and block 5 ("horizon") is the horizon series at node 4 minus node 4.

| start | share of \|U\| on affine rows, h 10⁻³³; 10⁻³² | verdict | against the largest q alone | the 29 rows linear by construction: their share of \|U\|; all affine | U by block | \|u(node4) + u(horizon)\|/\|u\| |
|---|---|---|---|---|---|---|
| REF, D 242.5459 | 2.07×10⁻¹⁸; 2.07×10⁻¹⁸ | **refutes** | 0.5 | 6.4×10⁻⁵⁶; yes | node4 0.5, horizon 0.5, node3 3.3×10⁻¹¹ | 2.7×10⁻²⁶ |
| ACC, D 242.5225 | 2.07×10⁻¹⁸; 2.07×10⁻¹⁸ | **refutes** | 0.5 | 6.8×10⁻⁵⁶; yes | node4 0.5, horizon 0.5, node3 3.3×10⁻¹¹ | 2.7×10⁻²⁶ |
| D 232.13 | 4.24×10⁻¹⁸; 4.24×10⁻¹⁸ | **refutes** | 0.5 | 3.6×10⁻⁴⁶; yes | node4 0.5, horizon 0.5, node3 1.1×10⁻¹⁷ | 2.2×10⁻²¹ |
| class A, D 249.3927 | 1.88×10⁻¹⁸; 1.88×10⁻¹⁸ | **refutes** | 0.5 | 1.1×10⁻⁶¹; yes | node4 0.5, horizon 0.5, node3 2.8×10⁻⁴³ | 3.6×10⁻³⁰ |

- **Where U lives.** On the matching at the last node, half on each of its two blocks, row for row: each (field,
  harmonic) row of node4 has the share of the same row of horizon. Both blocks hold minus node 4's state, and U pairs
  them so that it cancels (to 2.7×10⁻²⁶ at REF). So the weak equations are the mismatch between the Taylor arrival at
  node 4 and the horizon series there, with node 4's own unknowns eliminated.
- **Its harmonics.** The same at all four starts to two digits: p and m at k = 1 and 3 (0.22 each of m.k1, m.k3, p.k1,
  p.k3), and a at k = 0, 2, 4 (0.026, 0.051, 0.027). Every other harmonic is below 10⁻⁴. These are the low harmonics,
  the ones the freeze keeps matched (k ≤ kfreeze = 3 for p).
- **How far from affine those rows are.** The 30 rows with a share above 10⁻⁶ have q_r/|J_r| from 7×10⁻⁵ to 1.2×10⁻²,
  six to eight orders above the cut, so the verdict does not depend on the threshold. At REF the node4 rows have q_r =
  0.03 to 0.1 against |J_r| ≈ 22, and the horizon rows q_r = 3×10³² to 6×10³² against |J_r| ≈ 3×10³⁶.
- **The rows linear by construction** (the horizon p rows above kfreeze, and the phase row) are all classified
  affine, with q_r ≤ 1.5×10⁻⁶⁵, and U has no weight on them (10⁻⁴⁶ to 10⁻⁶¹).
- **Where V lives** (diagnostic). The Δ axis lies in span(V): |P_V e_Δ|² = 0.99949 (REF and ACC), 0.99942 (D 232.13)
  and 0.99953 (class A). The rest is p1, m_h and node 4's p and m at k = 1 (cos) and k = 3: |P_V e|² = 0.26 for p1.k3c
  and p1.k3s, 0.22 for p1.k1c, 0.18 for m_h.k1c and 0.14 for m_h.k3c and m_h.k3s.

Along e = a/|a|, the parts of Uᵀx (each against the size it would have without cancellation between rows):

| start | \|Uᵀ F″(e, e)\|; no-cancellation size | its horizon part; its node4 part | \|Uᵀ T(e)\| (of its size) | \|Uᵀ 3S(e)\| (of its size) | \|Uᵀ(T − 3S)(e)\| of its size | (W3) \|Uᵀ(T − 3S)\|/\|Uᵀ T\| |
|---|---|---|---|---|---|---|
| REF | 2.1×10⁻⁸; 1.4×10³³ | 3.21×10⁻³ (of 1.4×10³³); 3.21×10⁻³ (of 0.092) | 1.51×10³³ (0.77) | 1.51×10³³ (0.55) | 3.1×10⁻¹⁹ of 9.4×10³² | 1.9×10⁻¹⁹ |
| ACC | 2.1×10⁻⁸; 1.4×10³³ | 3.14×10⁻³; 3.14×10⁻³ (of 0.089) | 1.51×10³³ (0.76) | 1.51×10³³ (0.54) | 3.1×10⁻¹⁹ of 9.3×10³² | 1.9×10⁻¹⁹ |
| D 232.13 | 3.6×10⁻¹¹; 2.0×10³¹ | 6.56×10⁻³; 6.56×10⁻³ (of 0.021) | 1.53×10³¹ (0.63) | 1.53×10³¹ (0.34) | 1.6×10⁻²⁰ of 2.5×10³¹ | 2.6×10⁻²⁰ |
| class A | 4.9×10⁻⁷; 1.9×10³⁴ | 2.60×10⁻³; 2.60×10⁻³ (of 0.0074) | 1.35×10³⁴ (0.85) | 1.35×10³⁴ (0.63) | 9.0×10⁻¹⁹ of 6.4×10³³ | 4.3×10⁻¹⁹ |

- **Q along e** is a cancellation between rows that are not affine. Within the horizon block, U's part of F″(e, e) is
  10⁻³⁴ to 10⁻³⁷ of its rows' size, and the 3×10⁻³ left over equals the node4 block's part, which cancels it in turn
  to the differences' truncation (#528's Q ≲ 6×10⁻⁶ per entry at REF).
- **T and 3S** each come from the horizon block, at 0.63 to 0.85 (T) and 0.34 to 0.63 (3S) of their rows' sizes, so each
  is a sum of rows that do not cancel much. Summed without cancellation, T − 3S is half of T's size (9.4×10³² against
  1.95×10³³ at REF), so T = 3S does not hold row by row. It holds in U's sum over the horizon rows, to 3×10⁻¹⁹ of that
  size, the differences' resolution (#528's T − 3S between step pairs, 1.6×10⁻¹⁷).
- The rows classified affine contribute at most 4×10⁻²¹ of each no-cancellation size (4×10⁻²⁴ at the class A start).
- (W3)'s consistency with #528 and #677 holds: |Uᵀ(T − 3S)(e)| = 1.9×10⁻¹⁹ |UᵀT(e)| at REF, against #528's max|T − 3S| =
  1.6×10⁻¹⁹ max|T|.

The REF pilot (8cbd3f31) printed the same judged lines to every digit.

### 10.7 Verdicts against the protocol

| criterion | REF | ACC | D 232.13 | class A |
|---|---|---|---|---|
| (W0) offset on its rows; the basis line equals #677's | pass; equal | pass; equal | pass; equal | pass; equal |
| (W1) the 29 rows linear by construction classified affine | pass | pass | pass | pass |
| (W2) the change of rows (orthonormal to 2.6×10⁻¹⁴⁰; rows against grid ≤ 2×10⁻¹⁴¹) | pass | pass | pass | pass |
| (W3) w_ee, T(e), S(e) at two steps; 3S = T along e | 4.4×10⁻⁵, 1.2×10⁻¹⁸, 2.0×10⁻¹⁷; pass | 4.4×10⁻⁵, 1.3×10⁻¹⁸, 2.0×10⁻¹⁷; pass | 9.7×10⁻⁸, 2.2×10⁻¹⁹, 2.8×10⁻¹⁸; pass | 6.0×10⁻⁴, 2.1×10⁻¹⁸, 4.4×10⁻¹⁷; pass |
| (W4) the share on affine rows; the verdict, at both steps | 2.1×10⁻¹⁸: refutes | 2.1×10⁻¹⁸: refutes | 4.2×10⁻¹⁸: refutes | 1.9×10⁻¹⁸: refutes |
| exit | 0 | 0 | 0 | 0 |

### 10.8 What 3S = T expresses there (reading; inferred, not shown)

- **The conjecture of affine weak rows is refuted at all four starts.** The weak left vectors do not avoid curvature:
  they sit on the last matching, whose horizon block holds the largest curvature along V in the system (up to 3.6×10³⁴
  at the class A start). That U lives on the last matching fits #543's and #544's finding that only the last node's
  position moves the weak σ.
- **What the cancellations are.** By section 1's expansion, the reduced map's own coefficients along e are Q(e, e) and
  (T − 3S)(e, e, e), and both are cancellations between rows that are not affine, to 10⁻⁴¹ and 3×10⁻¹⁹ of their rows'
  sizes. So 3S = T says that
  the reduced map has no cubic at the scale of its rows' cubic. T is the horizon series' third derivative along V, seen
  through U. 3S is the same rows' mixed curvature between V and w_ee, the strong correction's shift of the horizon data,
  p1 and the centre and nodes' c (w_ee by unknowns, `rowsum.txt`). On z0 + Vc the horizon series bends at 10³³; along
  the strong-solved manifold the strong rows remove that bend in every direction U sees.
- **The reduced map's own scale.** #677's even parts give the reduced map's curvature against its slope along e,
  |g″(e, e)|/|Be| = 2K/|a|: 9.0 (REF), 8.8 (ACC), 24.6 (D 232.13) and 47.2 (class A) per unit of |c|. The weak equation
  therefore curves on |c| of order 0.02 to 0.1 in the unknowns, an ordinary scale for a function of the period Δ, whose
  axis is in span(V). At |c| from 10⁻³⁰ to 10⁻²⁴, its quadratic term is 5×10⁻³⁰ to 2×10⁻²³ of its linear one. That is
  all #677 found, with no special structure needed. In this reading the weak equations look affine because the weak
  coordinates are 27 to 28 orders below their own curvature scale. The 10³³ cubic on z0 + Vc is the curvature of that
  flat chart, carried by the horizon series, which the strong correction removes. This is a conjecture, and its test is
  in the closing summary's recommended tasks: at REF it predicts a departure of 4.5|c| of |Bc| along e, 0.45% at |c| =
  10⁻³ and 4.5% at 10⁻².

### 10.9 Discrepancies (reported, not fixed)

- **The issue's literal reading.** With "affine to 10⁻¹⁰ of the largest q" alone, the share is 0.5 at all four
  starts, "between". The node4 rows' q (0.03 to 0.1) is 10⁻³⁴ of the horizon rows', while their own q/|J_r| is about
  10⁻³. The protocol's verdict, which also asks q_r ≤ 10⁻¹⁰ |J_r|, refutes. Both are reported.
- **Section 9.8's wording.** "A near-symmetry that leaves the weak rows affine on the strong-solved manifold" read as
  rows affine in z is refuted here. Read as the reduced map being affine on that manifold, it is #677's measurement, and
  10.8 offers a reading that needs no symmetry, itself a conjecture until its test runs.

### 10.10 Commands

From `hp/`, with a clone of the raw-data repository at `RAW` and main's hpkernel build on the path:

```
python3 reduced_cubic.py rows planted                     # check: pass, exit 0
python3 reduced_cubic.py rows planted --mutate=quadA      # check: FAIL, exit 1 (also --mutate=parity)
python3 reduced_cubic.py rows RAW I461-win2/probeG03125b/dz0_pin3_D242.5459_a0.031250.npz \
    I461-win2/probeG03125a/anm_pin3_D242.5146.npz --threads=2 --json=rows_ref.json
python3 reduced_cubic.py rowsum rows_ref.json             # the shares, rows, pairing and cancellations
```

`I694-mac/run_i694.sh` runs all of them. A start takes 6 to 13 minutes at 2 threads, most of it the octo-double LU.
Raw: `I694-mac/` of the raw-data repository at af311e7, with every run's log and JSON (its README lists them).

## 11. The reduced map's own scale: |c| = 10⁻⁶ to 10⁻² (#698)

Section 10.8 read #677's even part as the reduced map's ordinary curvature: |g″(e, e)|/|Be| = 2K/|a| is 9.0 per unit
of |c| at REF and 47.2 at the class A start, so the weak equation would curve on |c| of order 0.02 to 0.1 and look
affine only because the chain's |c| is 27 orders smaller. That reading predicts the departure from affine along e at
|c| far beyond #677's range: 4.5|c| of |Bc| at REF and 23.6|c| at the class A start. This section measures it at
|c| = 10⁻⁶ to 10⁻². Sections 11.1 to 11.3 are the protocol, committed before any real run of this mode.

### 11.1 Method (`reduced_cubic.py curve`)

- **The basis.** V, U, B = UᵀJV, a and w0 at z0 as in section 1, from the same code. The held offset's J dz0 is formed
  by the octo-double jet with the dimension frozen (`Solver.residual_jet_o`, its first coefficient J dz0 by forward
  mode) instead of the explicit Jacobian, which saves about 200 s per start. (Z0) checks that the basis is unchanged.
- **The points.** c = ±t e, e = a/|a|, at t = 10⁻⁶, 10⁻⁵, 10⁻⁴, 10⁻³, 10^−2.5 and 10⁻², and on the + ray of a real start
  first at t = 10⁻²⁴, #677's largest |c| (the lead-in). t = 10⁻³ and 10⁻² are judged.
- **The strong solve by Newton** (`newton_strong`). At |c| above about 10⁻¹⁷ the chord through z0's LU no longer
  contracts: at REF its corrections fall in pairs by about 5×10³² t² (`I677-mac/ray_ref.json`, *inferred* from the
  steps). Here the octo-double Jacobian and its LU are refactored at the iterate z = z0 + Vc + w (`Solver.newton_o`,
  about 220 s at 2 threads). Each correction is the bordered solve with the current LU, J dw = −r + Uλ with Vᵀdw = 0,
  r = (I − UUᵀ)F the strong residual: dw = −J⁻¹r + J⁻¹Uλ, with λ from the 4 × 4 system Vᵀdw = 0. At z0 this is section
  1's chord, since there VᵀJ⁻¹U = Rᵀ and VᵀJ⁻¹r = 0. The first correction after a refactorization is a Newton step.
  Further corrections reuse that LU while each cuts |dw| by 10⁻³ or more; otherwise the LU is refactored at the new
  iterate, and a correction that grows |dw| is not applied. The fixed point solves the strong rows whatever LU is used.
  The iteration stops when |dw| < 10⁻⁶⁰ max(|w|, |w0|), when a refactorization does not cut the smallest |dw| so far
  (the floor), or after six LUs, and keeps the iterate with the smallest |dw|.
- **Convergence.** The kept iterate's error estimate is err = |Uᵀ[F(w + dw) − F(w)]|, the change of g by the correction
  not applied, as in section 9. A solve converged when err < 10⁻²⁰|Bc|. It failed on a non-finite |dw| or on one above
  max(|c|, |w0|), a strong correction larger than the weak move. A ray stops at its first failed solve.
- **The start of each solve.** w at the new t from the converged points: the two with the largest |t| at most the new
  one, fitted by w0 + αt² + βt³ (signed t, the same sign first); one point, its quadratic scaling; none, w0.
- **The departure.** dep(c) = g(c) − g(0) − Bc, with g(0) and its floor floor0 from section 9's chord at c = 0 through
  z0's LU, from w0 and from w0(1 + 10⁻⁵). At these |c| the natural unit is |Bc| = t|Be|, not |Ba|, and every part is
  given in it. E(t) and O(t) are the even and odd parts, as in section 9. The floor of each t is the larger of the two
  signs' error estimates and floor0, and a part is resolved where it exceeds 100 floors.
- **Orders.** The local order of each part between neighbouring points, of the departure itself (E|Bc| and O|Bc| in
  t). The plant's leading orders come from a least-squares line at its first four resolved points, as in section 9.
- **The flat chart** (w = 0, z0 + Vc, residuals only) at the same t: its departure, its parts, O/t³, and the change of
  the strong residual, max over rows of |[(I − UUᵀ)(F(z0 + Vc) − F(z0))]_r|, over t².
- **The prediction.** κ = K/|a| from #677's even part along e (`I677-mac/raysum.txt`): REF K = 3.750×10⁻²⁹,
  |a| = 8.3246×10⁻³⁰, κ = 4.505; the class A start K = 1.892×10⁻²⁸, |a| = 8.0246×10⁻³⁰, κ = 23.58. The reading predicts
  E(t) = κt|Bc|, local order 2.

### 11.2 The plant (`reduced_cubic.py curve planted`)

CurvePlanted is section 9.2's plant (Q = L = 0, T = 3S) carried to this range. Its reduced map is still exactly
β + Σy + P₄/24 + P₅/120, now with P₄/24 = 10⁻²|Bc| at |c| = 10⁻³ along e and P₅/120 equal to it at |c| = 3×10⁻³, so the
even part has order 4 and the odd part order 5 across the range. Changed from 9.2:
- no quartic c₄ in the strong rows, and Λ⁻¹C ≈ 0.1 in every strong row, so that |w(c)| ≈ 0.05|c|² as at REF
  (|w_jl| 0.06 to 0.14). T = 3S is formed again with that C;
- x's rows 0 to 3, which K does not reach, get M(y)x = Λ m (b·y) R x with R skew-symmetric (so Λ + M(y) is never
  singular) and ½Λ n (d·x)², with m = 10⁵ and n = 10⁷. They leave g unchanged, but the chord through z0's LU cannot
  follow them once m|b·y| > 1 (|c| above about 10⁻⁵), and Newton can;
- J at an iterate is formed by central differences at 400 digits (h = 10⁻¹⁶⁰) and solved by mp.lu_solve.

Pass criteria, fixed before the plant's first run:
- **(PC0)** As (PY0): the plant's reduced map minus β + Σy against P₄/24 + P₅/120 at |y| = 10⁻⁵ and 10⁻³, to 10⁻²⁰.
- **(PC1)** At every t and in both parts, |measured − planted| < 10⁻³ max(|planted|, 100 floors).
- **(PC2)** The leading orders are 4 (even) and 5 (odd), to 0.05.
- **(PC3)** Every strong solve converged, at all six t on both rays.
- **(PC4)** The verdict's self-test: `curve_verdict` on planted rows with E = κt and O = E/10 must give *supports* with
  κ, *refutes* with 20κ, *between* with 4κ, *refutes* with O = 2E, and *not decided* with the 10⁻² row dropped.
- **Controls.** `--mutate=w0` drops the strong solve (g on the flat chart, whose odd part carries T(c, c, c)/6, of
  order 3): (PC1) and (PC2) must fail. `--mutate=chord` solves the strong rows by section 9's chord through z0's LU:
  (PC3) must fail. A failed check exits 1.

The plant's first runs, during the code's development and before this protocol, found one bug that (PC4) caught: the
verdict took the order of E in units of |Bc| (order 1) instead of the departure's (order 2).

### 11.3 The real starts (fixed before the first real run)

- **Inputs.** REF and the class A start, as in section 9.3, with #677's ray runs (`I677-mac/ray_ref.json`,
  `ray_classA.json`) for (Z1) and #694's rows runs (`I694-mac/rows_ref.json`, `rows_classA.json`) for (Z2).
- **(Z0)** As (Y0): the held offset is on its rows, and the basis line (σ, a, |Ba|) equals #677's to its printed digits.
- **(Z1)** The lead-in at t = 10⁻²⁴ on the + ray: |dep|/|Ba| from `newton_strong` equals #677's chord value to 10⁻⁶
  (relative). The two share F, the basis and g(0), not the strong solve.
- **(Z2) The flat chart, a control on real data.** (a) At t = 10⁻⁶, the strong residual's change over t² equals
  ½ max_r |F″_r(e, e)| of #694 (2.34×10³³ at REF, 2.70×10³⁴ at the class A start) within a factor 2, at both signs:
  the flat chart's 10³³|c|² bend. (b) Its departure exceeds κt|Bc| by more than 10 at every t from 10⁻⁶ to 10⁻², so
  that without the strong solve the test refutes. Its odd part O/t³ is reported against |UᵀT(e)|/6 (#694: 1.51×10³³/6
  at REF, 1.35×10³⁴/6 at the class A start).
- **(Z3)** Every strong solve converged. Otherwise the |c| where the ray stops is reported, with the solve's steps.
- **The verdict per start** (`curve_verdict`). At t = 10⁻³ and 10⁻² the ratios E/(κt), |dep(+te)|/(κt|Bc|) and
  |dep(−te)|/(κt|Bc|), and the least-squares order of the departure's even part over 10⁻³, 10^−2.5 and 10⁻².
  - *Supports*: all six ratios within a factor 2 (0.5 to 2), the order within 2 ± 0.25, and no t below 10⁻² where the
    odd part, resolved, is at least the even part.
  - *Refutes*: a ratio off by more than 10 (above 10 or below 0.1), or an odd part at least as large as the even part
    at a resolved t below 10⁻² (a cubic as large as the quadratic there, so the map bends on a scale below 10⁻²).
  - *Between* otherwise, reported as it is; *not decided* when a judged t is missing (the ray stopped) and nothing
    refutes.
- **The reading's verdict.** Supported if both starts support, refuted if either refutes, and otherwise reported as it
  is.
- **Reported, not judged.** Every point's departure and parts, their local orders, O/E per t, each solve's LUs, steps,
  error estimate and max|F|, the error of each solve's start, and the flat chart's parts at every t.
- **Order and stops.** The plant and its two controls first, then REF as the pilot, then the class A start. A failed
  plant or control stops the real runs. (Z1) or (Z2) failing at a start stops that start until the cause is known. The
  solver's rules in 11.1 may change after the pilot, with the change and its cause recorded in 11.4; no criterion of
  11.2 or 11.3 changes.

### 11.4 Changes after real data

- **The REF pilot** (code 085fbdff, raw `I698-mac/curve_ref.log`). (Z0) and (Z1) pass. The flat-chart control (Z2a)
  fires: at |c| = 10⁻⁶ the flat chart's strong residual is 1.2×10²⁰ times ½F″(e, e)t². Newton stops at 10⁻⁶ on both
  rays (11.5). By 11.3 that stops REF's judged measurement until the cause is known, and `curve_verdict` gives *not
  decided* (both judged t missing).
- **A defect of (Z2a) as written.** #694 stored F″(e, e) in its harmonic rows, and (Z2a) compares its max norm with one
  over grid rows. A max norm changes under that orthogonal change of rows, which is why the ratio is 0.45 even at
  10⁻²⁴, where the flat chart is exactly quadratic. A 2-norm does not change. The failure at 10⁻⁶ (10²⁰ times) does
  not depend on it. The criterion is left as frozen, and the bridge below prints the 2-norm.
- **A diagnostic added for the cause, the chord bridge** (`--bridge=1`, code 46f250b3; nothing in 11.2 or 11.3
  changed). Through z0's LU, before any refactorization, section 9's chord follows both rays from 10⁻²⁴ upward, four
  per decade to 10⁻¹⁴, until it fails. Per t it prints the departure's parts (of |Bc|), |w − w0|/t² against ½|w_ee|
  (#694), and the flat chart's strong residual change in 2-norm over t² against ½|F″(e, e)|₂. It also prints the
  chord's first three corrections from w0, |dw₁|/|dw₀| and the signed pair ratio ρ = ⟨dw₂, dw₀⟩/|dw₀|². If the chord's
  error map N acts on two directions, as #677's paired steps suggest, then N² = ρ there. The strong block's
  determinant is then z0's times 1 − ρ, so ρ → +1 marks a fold of the strong-solved manifold, and ρ < 0 an
  oscillation without one (*inferred* from that model). Its plant check (PB): the curve plant's skew coupling must
  give ρ < 0 at every t. Its control `--mutate=fold` (a symmetric coupling) must give ρ > 0, failing (PB).
- **The class A start** runs with `--bridge=1`: the bridge first, then the protocol's Newton points.
- **The pair test in the linear regime** (203e56a0). REF's bridge showed the test from w0 leaving its t² law near
  10⁻²¹, where its start error 0.072t² reaches about 10⁻⁴³. So each converged point also gets the test from the
  converged w moved by 10⁻⁶⁰ along w − w0, far inside the linear regime. The class A run has both.
- **REF's bridge again** (`--bridge_to=1e-21`, code 2169dc00, `curve_ref_bridge2`), since at the class A start the two
  tests disagree in sign, so REF's sign from w0 is not evidence. This run also stores the parts as vectors at 30
  digits (4bb29b7f), for a cubic that an order-1 term hides in |O|.

### 11.5 The plant's results

`python3 reduced_cubic.py curve planted` from `hp/` (code 085fbdff, the bridge from 46f250b3; raw
`I698-mac/curve_planted*.log`):

| check | plant | `--mutate=w0` | `--mutate=chord` | `--mutate=fold` (with `--bridge=1`) |
|---|---|---|---|---|
| (PC0) the reduced map against P₄/24 + P₅/120 (bound 10⁻²⁰) | 1.1×10⁻⁹⁰: pass | | | |
| (PC1) worst point, both parts (bound 10⁻³) | 1.3×10⁻⁴⁸: pass | 7.3×10⁵⁶: FAIL | pass (one t) | pass |
| (PC2) leading orders 4 and 5 (to 0.05) | 4.000, 5.000: pass | 4.000, 3.000: FAIL | none: FAIL | pass |
| (PC3) every strong solve converged | pass (23 LUs) | pass | stops at 10⁻⁵: FAIL | pass |
| (PC4) the verdict's self-test | pass | pass | pass | pass |
| (PB) ρ < 0 at every t of the bridge (`--bridge=1`) | pass (82 of 82, both tests) | | | ρ > 0 at 82 of 82: FAIL |
| exit | 0 | 1 | 1 | 1 |

The plant's skew coupling gives ρ = −1.9×10⁻⁴⁰ (t/10⁻²⁴)², which is −(m (b·e) t)² with (b·e)² = 0.019, as planted.

### 11.6 Results at REF (the protocol)

Raw `I698-mac/curve_ref.log` and `.json` (code 085fbdff, exit 1), summarized by `reduced_cubic.py curvesum`.

| criterion | result |
|---|---|
| (Z0) offset on its rows; basis line against #677's | pass; σ, a and \|Ba\| equal to every printed digit. The jet's J dz0 leaves 6.9×10⁻⁶⁵ off the held rows (1.6×10⁻⁹⁹ by columns), and \|Vᵀw0\| is 3.5×10⁻¹²⁸ against 4.3×10⁻¹²⁸ |
| (Z1) the lead-in at 10⁻²⁴ against #677's chord | 5.4114114×10⁻¹⁹ \|Ba\| both: 2.3×10⁻¹⁷ (bound 10⁻⁶): pass. Newton took 2 LUs: on the first, the corrections came in a pair, 3.2×10⁻³⁶ twice, against \|w − w0\| = 7.2×10⁻⁵⁰ |
| (Z2a) the flat chart's strong residual at 10⁻⁶ against ½max\|F″(e, e)\| | 1.2×10²⁰ times (within a factor 2): FAIL |
| (Z2b) the flat chart's departure against κt | 1.0×10⁷⁹ times or more (above 10): pass |
| (Z3) every strong solve converged | no: both rays stop at 10⁻⁶ |
| the verdict (`curve_verdict`) | **not decided**: both judged t missing |

- **Newton at 10⁻⁶.** From w0 + 10³⁶(w(10⁻²⁴) − w0), the quadratic prediction (\|w − w0\| = 7.2×10⁻¹⁴), max\|F\| is
  2.7×10²² (+) and 3.1×10²² (−), and Newton's first step is 5.9×10²⁰ (+) and 3.5×10¹⁹ (−), far above \|c\|, so the
  solve fails on its first LU.
- **The flat chart** (strong residual change over t², max over grid rows; odd part over t³):

| \|c\| | strong residual / t² | even part / \|Bc\| | odd part / t³ |
|---|---|---|---|
| 10⁻²⁴ | 1.049×10³³ | 8.8×10¹³ | 2.516×10³² = \|UᵀT(e)\|/6 (#694: 1.5095×10³³/6) |
| 10⁻⁶ | 2.9×10⁵³ | 4.6×10⁷³ | 2.2×10⁵³ |
| 10⁻⁵ | 1.9×10⁵⁹ | 2.9×10⁸⁰ | 1.1×10⁵⁹ |
| 10⁻⁴ | 2.4×10⁶⁶ | 3.6×10⁸⁸ | 9.7×10⁶⁵ |
| 10⁻³ to 10⁻² | NaN (overflow) | NaN | NaN |

  At 10⁻²⁴ the flat chart is the 10³³|c|² bend that #694 measured: in 2-norm the strong residual over t² is
  3.3310×10³³ against ½|F″(e, e)|₂ = 3.331×10³³ (the bridge, 11.7). The max norm gives 0.45 of #694's, which is the
  defect of 11.4. Its quadratic holds only at small |c|: in 2-norm it has changed by 5.2×10⁻¹⁰ at 3.16×10⁻¹⁸, as
  (t/1.4×10⁻¹³)², and at 10⁻⁶ it is 10²⁰ times the quadratic.

### 11.7 The chord bridge from 10⁻²⁴ (diagnostic, 11.4)

Raw `I698-mac/curve_ref_bridge.log` (code 46f250b3), `curve_ref_bridge2.log` (REF again, with the linear-regime pair
test and the parts as vectors, to 10⁻²¹; code 2169dc00) and the bridge in `curve_classA.log` (code 203e56a0), each
summarized by `curvesum`. All judge nothing.

| quantity | REF | class A |
|---|---|---|
| the chord converged, both rays | 10⁻²⁴ to 3.16×10⁻¹⁸ (failed at 5.62×10⁻¹⁸, 40 steps) | 10⁻²⁴ to 10⁻¹⁸ (failed at 1.78×10⁻¹⁸, 40 steps) |
| E/t against its value at 10⁻²⁴ | constant to 2.2×10⁻¹⁵ | constant to 2.2×10⁻¹⁶ to 5.6×10⁻¹⁹; 1.7×10⁻¹¹ at 10⁻¹⁸, at its floor |
| E/(κt), κ from #677's 4-digit K | 1.00001 | 1.00013 |
| \|w − w0\|/t² against ½\|w_ee\| (#694) | 0.0718098 against 0.07181, constant to 10⁻¹¹ | 0.0539856 against 0.05399, constant to 10⁻¹¹ from 10⁻²² (a term linear in t shows as ∓5.6×10⁻⁹ at 10⁻²⁴) |
| the odd part | its order-1 term (5.3×10⁻³⁰ \|Bc\|) plus a cubic: O − O(10⁻²⁴) of local order 3.000 at every pair of neighbours from 5.6×10⁻²³ to 3.16×10⁻¹⁸, (O − O₀)/t² = −7.8157 \|Be\| per unit t² (the cubic's component along the order-1 term). From the vectors (`curve_ref_bridge2`), the whole cubic: \|O(t) − O(t₀)\|/(t² − t₀²) = 19.2053 \|Be\| per unit t², the same to 6 digits from 3.2×10⁻²⁴ to 10⁻²¹, local order 3.000 | its order-1 term Lc (7.9×10⁻⁷ \|Bc\|) hides any cubic below 10⁻²⁹ of it in \|O\|, which is constant to 13 digits |
| the flat chart's strong residual change, 2-norm/t², against ½\|F″(e, e)\|₂ | 3.3310×10³³ against 3.331×10³³; changed by 5.2×10⁻¹⁰ at 3.16×10⁻¹⁸, as (t/1.4×10⁻¹³)² | 3.5401×10³⁴ against 3.54×10³⁴; changed by 2.6×10⁻¹¹ at 10⁻¹⁸, as (t/2.0×10⁻¹³)² |
| ρ from w0 | −1.631×10³² t² to 5.6×10⁻²², then off its law | −9.94×10³³ t² to 1.8×10⁻²², then off its law |
| ρ in the linear regime | **+4.6054×10³² t²**, both rays, at every t to 10⁻²¹ (`curve_ref_bridge2`) | **+8.6447×10³³ t²**, both rays, at every t to 10⁻¹⁸ (8.665×10⁻³ there; its excess over the t² law, 2.3×10⁻³ at 10⁻¹⁸, grows as (t/2.1×10⁻¹⁷)²) |

- **The two pair tests disagree at the class A start.** From w0 the start error is 0.054t², and from the converged point
  it is 10⁻⁶⁰. Both ratios follow t², but with opposite signs. The w0 test leaves its law first, near 3×10⁻²², while
  the linear one keeps it to the last digit. So the w0 test is not in the chord's linear regime (*inferred*), and its
  sign at REF is not evidence. That is why REF's bridge was run again with the linear test.
- **What the linear ratio says** (*inferred*, in the two-direction model of 11.4). At the class A start ρ = 1 at
  t = 1.0755×10⁻¹⁷ on both rays, where the strong block (I − UUᵀ)J restricted to V's complement would be singular, a
  fold of the strong-solved manifold. At REF the linear ratio is +4.6054×10³² t², so ρ = 1 at t = 4.66×10⁻¹⁷; the w0
  test's −1.63×10³² t² there was not in the linear regime either. The chord's own step limit (40) stops the bridge
  long before: at the class A start ρ is 8.7×10⁻³ at its last converged point, so the bridge cannot see the fold.
- **The map within the bridge.** At REF the even part is exactly quadratic over 6.5 more decades than #677's, and the
  odd part has a resolved cubic of local order 3.00. Its coefficient is 19.2|Be| per unit t² (7.8 of it along the
  order-1 term), 0.95 of κ² = 20.3, the size the reading assumes. The quadratic and the cubic would be equal near
  |c| = 4.5/19.2 = 0.23. The strong correction w is exactly quadratic too. No quartic is resolved.

### 11.8 Results at the class A start (the protocol)

Raw `I698-mac/curve_classA.log` and `.json` (code 203e56a0, exit 1), with the bridge of 11.7 run first.

| criterion | result |
|---|---|
| (Z0) offset on its rows; basis line against #677's | pass; σ, a and \|Ba\| equal to every printed digit (\|Vᵀw0\| 2.6×10⁻¹²⁸ against 8.4×10⁻¹²⁹); the jet's J dz0 leaves 9.1×10⁻⁶⁶ off the held rows |
| (Z1) the lead-in at 10⁻²⁴ against #677's chord | 0.098216381 \|Ba\| both: 4.6×10⁻¹⁷ (bound 10⁻⁶): pass. Newton took 3 LUs: the corrections came in a pair on the first, 3.3×10⁻³⁵ twice, against \|w − w0\| = 5.4×10⁻⁵⁰, and grew once on the second |
| (Z2a) the flat chart's strong residual at 10⁻⁶ against ½max\|F″(e, e)\| (2.70×10³⁴) | 1.7×10²⁰ times (within a factor 2): FAIL |
| (Z2b) the flat chart's departure against κt | 5.4×10⁸² times or more (above 10): pass |
| (Z3) every strong solve converged | no: both rays stop at 10⁻⁶. From the quadratic prediction max\|F\| is 1.4×10²³, and Newton's first step 2.6×10¹⁹ (+) and 2.2×10¹⁹ (−) |
| the verdict (`curve_verdict`) | **not decided**: both judged t missing |

The flat chart at the class A start, as at REF but without overflow: its strong residual over t² is 7.6×10³³ at
10⁻²⁴ and 4.5×10⁵⁴, 2.5×10⁶⁰, 2.6×10⁶⁷, 3.2×10⁷⁶ and 7.0×10⁸⁸ at 10⁻⁶, 10⁻⁵, 10⁻⁴, 10⁻³ and 10⁻², and its odd part over
t³ is 2.255×10³³ at 10⁻²⁴, which is \|UᵀT(e)\|/6 = 1.3531×10³⁴/6 (#694) to four digits.

### 11.9 Verdicts against the protocol

| criterion | REF | class A |
|---|---|---|
| (PC0)–(PC4) and (PB), the plant; its controls w0, chord and fold fail | pass; all three fail | (the same plant) |
| (Z0) | pass | pass |
| (Z1) the lead-in against #677 | pass (2.3×10⁻¹⁷) | pass (4.6×10⁻¹⁷) |
| (Z2) the flat chart, a control | **fires**: (a) 1.2×10²⁰ at 10⁻⁶ | **fires**: (a) 1.7×10²⁰ at 10⁻⁶ |
| (Z3) every strong solve converged | no: both rays stop at 10⁻⁶ | no: both rays stop at 10⁻⁶ |
| departure at 10⁻³ and 10⁻² against κt (factor 2 supports, 10 refutes) | not measured | not measured |
| the verdict | **not decided** | **not decided** |
| exit | 1 | 1 |

**The reading's verdict, by 11.3: not decided at either start.** The frozen test needs g at |c| = 10⁻³ and 10⁻², and
neither solve reaches them: Newton with the refactored LU fails at 10⁻⁶ on its first step at both starts, and the chord
through z0's LU at 5.6×10⁻¹⁸ (REF) and 1.8×10⁻¹⁸ (class A). The flat-chart control fires at both starts, since the flat
chart is quadratic only below about 10⁻¹³. That control's premise, "the flat chart's 10³³|c|²", holds at 10⁻²⁴ and not
in the issue's range.

### 11.10 Reading (inferred, not shown)

- **Within its reach the map is ordinary.** At REF the even part is exactly quadratic from 10⁻³⁰ (#677) to 3×10⁻¹⁸, a
  constant coefficient over 12.5 decades. The odd part carries a resolved cubic of local order 3.00 whose coefficient,
  19.2|Be| per unit t², is 0.95 of κ² = 20.3, the size the reading assumes. The quadratic and the cubic would be equal
  near |c| = 0.23. So the map's own Taylor coefficients are those of a function that curves on |c| of order 0.1, which
  is the reading's claim about the coefficients. It is numerical evidence at 3×10⁻¹⁸ and below, not at 10⁻³.
- **The strong-solved chart may not reach 10⁻³.** The chord's error map has a pair of directions with N² = ρ, which
  grows as t². In the linear regime ρ is positive at the class A start, +8.6×10³³ t², so in the two-direction model the
  strong block (with U and V fixed at z0) turns singular at |c| ≈ 10⁻¹⁷ on both rays: a fold, beyond which w(c), and so
  g(c), would not continue as one smooth branch. At REF the linear ratio is positive too, +4.6×10³² t², which puts
  the fold at |c| ≈ 4.7×10⁻¹⁷. If the fold is real, g's Taylor series at 0 has a radius
  of about 10⁻¹⁷, and the reading's prediction at 10⁻³ has no meaning on this chart, whatever the coefficients. Its
  test is a recommended task (the strong block's smallest singular value along the ray, from the refactored LU).
- **Why Newton fails at 10⁻⁶.** The strong rows are extremely nonlinear in their softest direction. The chord's pair
  test from w0 leaves its t² law once its start error passes about 10⁻⁴³ (REF, 0.072t² at t ≈ 10⁻²¹), and at the class
  A start its sign already disagrees with the linear test at 10⁻²⁴, where that error is 5×10⁻⁵⁰. Newton's step from
  the quadratic prediction at 10⁻⁶, which leaves a residual of 10²² there, is outside that basin, and so would be any
  prediction made by extrapolation from |c| ≤ 3×10⁻¹⁸.
- **What 3S = T then is.** Section 10.8 read the reduced map's cubic as the ordinary one of a function of Δ. The bridge
  adds that the cubic is resolved and ordinary in size at REF, 19.2|Be| per unit t², so 3S = T leaves a cubic of that
  size rather than none. The map's cubic along e is about 10⁻⁵⁷ of the flat chart's, far below the tensors' resolution:
  |UᵀT(e)|/6 = 2.5×10³² per t³ against 19.2|Be| = 2.3×10⁻²⁵ per t³.

### 11.11 Discrepancies (reported, not fixed)

- **(Z2a) as written compares max norms in two row bases** (#694's harmonic rows against grid rows), which gives 0.45 at
  10⁻²⁴, where the flat chart is exactly quadratic. In 2-norm the agreement there is 3.3310×10³³ against 3.331×10³³
  (REF) and 3.5401×10³⁴ against 3.54×10³⁴ (class A). Its failure at 10⁻⁶ (10²⁰ times) does not depend on the norm.
- **The issue's control premise.** "The w = 0 control shows the flat chart's 10³³|c|²" holds at 10⁻²⁴ and not at
  10⁻⁶ to 10⁻²: the flat chart is 10²⁰ to 10³³ times its quadratic there and overflows at REF from 10⁻³.
- **"A resolved cubic below 10⁻² refutes."** Read literally, the bridge resolves a cubic at REF below 10⁻² (from
  5.6×10⁻²³ to 3.2×10⁻¹⁸). The protocol (11.3) froze the criterion as an odd part at least as large as the even part
  below 10⁻², a cubic that bends the map on a scale below 10⁻². By that it does not refute: the cubic is 19.2/4.5 t =
  4.3t of the quadratic. Both readings are stated; the bridge is a diagnostic and judges nothing.
- **The task's premise on the chord.** The chord through z0's LU fails already at 5.6×10⁻¹⁸ (REF) and 1.8×10⁻¹⁸ (class
  A), and Newton with the LU refactored at every iterate fails at 10⁻⁶ as well.
- **The two pair tests disagree in sign** at the class A start (11.7); the one from w0 is not in the chord's linear
  regime. REF's rerun with the linear test gives +4.6054×10³² t², positive like the class A start's. My progress post
  of 17:48 on #698 had read REF's negative ratio from w0 as "no fold"; the post of 19:22 corrected it.
- **Conduct.** Twice, a plant run of a few seconds on one thread ran while a real run was computing (the bridge's
  plant checks), against "one compute process at a time".

### 11.12 Commands

From `hp/`, with a clone of the raw-data repository at `RAW` and main's hpkernel build on the path:

```
python3 reduced_cubic.py curve planted                      # check: pass, exit 0
python3 reduced_cubic.py curve planted --mutate=w0          # check: FAIL, exit 1 (also --mutate=chord)
python3 reduced_cubic.py curve planted --bridge=1 --mutate=fold   # (PB) FAIL, exit 1; without the mutation, pass
python3 reduced_cubic.py curve RAW I461-win2/probeG03125b/dz0_pin3_D242.5459_a0.031250.npz \
    I461-win2/probeG03125a/anm_pin3_D242.5146.npz --threads=2 --ref677=RAW/I677-mac/ray_ref.json \
    --ref694=RAW/I694-mac/rows_ref.json --json=curve_ref.json       # exit 1: (Z2) fires, Newton stops at 1e-6
python3 reduced_cubic.py curvesum curve_ref.json             # the parts, the checks, the bridge's coefficients
```

`I698-mac/run_i698.sh` runs all of them (`--bridge=1` adds the bridge, `--newton=0` skips the Newton points,
`--bridge_to` caps the bridge). A real start takes 6 to 8 minutes to set up at 2 threads, each octo-double LU 3.5 to
9 minutes under the Mac's load, and the bridge 6 to 45 minutes. Raw: `I698-mac/` of the raw-data repository at
af339c17, with every run's log and JSON and `curvesum.txt` (its README lists them).

# The strong-row term S of the reduced cubic at the D 242.5 starts (#528)

#499 found that the first Newton step of the refused corrector attempts at the D 242.5 wall lies on the four weak
directions, and that on the affine subspace z0 + Vc the weak equations are cubic, with T = UᵀF‴ ≈ 1.6×10³³. With the
weak part of the first step put on that cubic's root, newton_od was still refused (#499's F4). #527 pointed out that the
reduced equation, with the strong rows solved, has the cubic coefficient T − 3S (Kuznetsov 1999, §5, for one null
vector), where S is the strong rows' curvature fed back into the weak rows. This note measures S at #499's two
D 242.5146 starts, and the reduced map itself on the strong-solved manifold.

Results are in section 4 onward, after the protocol (section 3), which was committed before the first real run.

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
- **The strong solve.** Section 1's chord through z0's LU, now run to |dw| < 10⁻⁶⁰|w0|, or until |dw| stops falling
  (two steps in a row that cut it by less than 0.3). One step may grow |dw|. At |c| ≫ |a| the chord's error map
  J⁻¹(I − UUᵀ)F″(Vc, ·) can have a norm above 1 and a square far below it, as at #528's c* (|dw| 5.44×10⁻⁵⁶,
  5.44×10⁻⁵⁶, then 5×10⁻⁸²). A solve fails on a non-finite |dw|, on |dw| above 10⁻³ max(|c|, |w0|), or after 40
  steps. Each t starts from w0 + (t/t′)²(w(t′) − w0), with t′ the ray's previous point. A ray stops at its first failed
  solve, and that |c| is reported.
- **The departure.** dep(c) = g(c) − g(0) − Bc, with the basis's B. The departure from B(c − a), the issue's affine
  part, is dep(c) + g(0) + Ba, and the constant ε0 = |g(0) + Ba|/|Ba| is reported (2.7×10⁻²⁵ at REF).
- **Its parts.** The even part E(t) = (dep(te) + dep(−te))/2 holds the orders 2, 4, …, and the odd part
  O(t) = (dep(te) − dep(−te))/2 the orders 1, 3, …. An error in B shows as order 1 in the odd part.
- **The floor of each t.** The largest of three things. Each sign's strong-solve error estimate,
  |Uᵀ[F(w − dw) − F(w)]|, is what the last correction, not applied, would change in g. floor0 is the difference of
  g(0) between solves from w0 and from w0 + ξ (ξ ⊥ V, |ξ| = 10⁻⁴|w0|), together with both their estimates. The floor is
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

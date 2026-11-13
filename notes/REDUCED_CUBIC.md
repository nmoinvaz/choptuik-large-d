# The strong-row term S of the reduced cubic at the D 242.5 starts (#528)

#499 found that the first Newton step of the refused corrector attempts at the D 242.5 wall lies on the four weak
directions, and that on the affine subspace z0 + Vc the weak equations are cubic, with T = UᵀF‴ ≈ 1.6×10³³. With the
weak part of the first step put on that cubic's root, newton_od was still refused (#499's F4). #527 pointed out that the
reduced equation, with the strong rows solved, has the cubic coefficient T − 3S (Kuznetsov 1999, §5, for one null
vector), where S is the strong rows' curvature fed back into the weak rows. This note measures S at #499's two
D 242.5146 starts, and the reduced map itself on the strong-solved manifold.

Results are in section 4 onward, after the protocol (section 3), which was committed before the first real run.

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
- **The controls at c\*.** The S control replaces dw = w(c*) − w0 by 2dw. That changes g by UᵀF″(Vc*, dw) + …, which
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
stopped REF run (section 3): the w_jl check, h_F = 10⁻³², the plant's quartic c₄ and `--mutate=hF`. No criterion was
loosened.

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

Pending.

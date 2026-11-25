# LARGE_D_FACTORED: a series formulation of the large-D problem, form S, and what still grows with D (#1088)

Track B of #1088. Written by a subagent on the Mac, 2026-11-25, under Nathan Moinvaziri's direction; not reviewed.
Code: [`solver/factored/`](../solver/factored/); `bash solver/factored/run_track_b.sh OUT STATES` reruns every check
and solve (the clone step for STATES is in its header). Raw logs: folder `I1088-mac/` of the raw-data repository.
Status words as in AGENTS.md. Every number below is from a log in `I1088-mac/` named beside it.

## 0. Verdicts

| question | answer | status |
|---|---|---|
| Where do the fast modes enter the unknowns? | nowhere in the solution; in the Jacobian through the horizon block: the mass mode's datum and the scalar mode's shadow at horizon orders about n/2 to n (section 1) | checked symbolically (derive_factored.py), numerical evidence |
| A layout without the layout's exponential factors | centre series in z = x² summed to x* = 1/√2, horizon series in s = (1 − x²)/(1 − x*²), no Taylor steps, matched at x* where \|F\| = 2^(−κ/2) for every n (section 2) | checked symbolically (checks 1–5, 7); reproduces hp's nodes at D 100 (check_state, check_s) |
| Is factoring the mass mode's datum enough (form A)? | no: its horizon coefficients peak at order k* ≈ σ_R, where the scalar's horizon map amplifies rough perturbations by about 2^(n/2); the a-datum's Jacobian column carries rounding × that gain, slope 0.135 per unit n against log10(2)/2 = 0.1505 (section 3) | numerical evidence (anchor_noise.log) |
| Form S (a slaved by the momentum constraint, #1089 slice 3) | removes the mass mode from both series: no a datum, no x-recurrence for a (section 2.3) | checked numerically on hp's states (check_s); Newton converges at D 100 |
| Form S at a solution | D 100, M 8, double-double: max\|F\| 1.9e-22; scaled singular values 1.9 down to the three selection pairs of harmonics 3, 5, 7: 5.2e-10, 1.4e-13, 2.6e-17/2.1e-17; rounding of every series coefficient moves the residual by 0.77 per unit (section 5.1) | numerical evidence |
| Is a fixed precision enough for every D? | **no, not in this formulation.** Off the solution the scalar mode's shadow carries the same 2^(n/2) from horizon order σ_R to x*: at a jump's seed the float64 and double-double horizon sums differ by 2e-14 at D 100, 1e-10 at 175, 4e-6 at 200 and 2.5 at 245, and the double-double sum itself reaches 2e16 times U(1) at D 400 (section 5.2) | numerical evidence (seed_precision.log) |
| Jumps in D (Track A's question, in this layout) | none converged: 100 → 150 and 100 → 105 at M 8, and D 200, 245, 263 from hp's states at M 6. The weak directions' basin is about σ_min²/curvature, and a seed's shadow grows like 2^(n/2) (section 5.3) | numerical evidence |
| The conditioning of form S at fixed M against D | **not measured**: no second converged D at M 8 (section 5.3) | |
| What still grows with D | the horizon series' length ks ∝ n (the selection sits at orders about n/2 to n); with the banded recurrence the cost is O(ks·kb·M), 3.0 s for one double-double residual at D 1000 (section 6) | proved by hand (the count of orders), numerical evidence (cost.log) |

## 1. Where the fast components sit

The equations are hp's (`hp/src/model.rs` header; LARGE_D_BASICS §1). The solution itself has no fast part (LARGE_D_BASICS
§3.5). The fast modes enter through the unknowns and rows of a layout:

- **Centre.** The scalar mode F (x^(1−n), p = m) and the mass mode A (a − 1 ∝ x^(1−n)) are singular at x = 0. A centre
  series excludes both exactly, so the centre data p₁(θ), c₀(θ) carry no fast part.
- **Matching region.** F's modulus is e^(−nΦ), Φ = ln x − ((1 + K)/2) ln(1 − x²). With K = κ/n it is 2^(−κ/2) at
  x* = 1/√2 for every n, against e^(0.020n) at hp's last node 0.7: ln\|F(0.7)\| = −0.59, 17.41 and 197.43 at n = 100,
  1000 and 10000 (check 7, `derive.log`).
- **Horizon.** F behaves as u^σ, σ_R ≈ n/2 + 2.6. The horizon-analytic solution carries F's shadow in its Taylor
  coefficients at orders σ to about 2σ (in s, the shadow of the free field peaks near order n at x*). The selection is
  the vanishing of that shadow, so any layout that imposes analyticity at the SSH carries Taylor orders up to about
  n(1 + O(n^(−1/2))) (proved by hand for the free field's 2F1; LIT_GAPS_EEG's slice 1 finds the same, a fixed degree N
  seeing the condition at n^−(N+1)).
- **Mass mode (form A only).** On the horizon side A is analytic, with gain G = x_K^−(n−1) inward: 2^((n−1)/2) at x*.
  Its s-coefficients rise to k* = ⌊(n − 1 − 2/λ)/(2/λ − 2)⌋ ≈ (n/2)λ/(1 − λ), λ = 1 − x_K², and fall after: at x* (λ = 1/2)
  k* ≈ n/2 ≈ σ_R, the orders where the scalar's horizon map has its largest gain (section 3). Both fast modes come from
  the same x^−n, which is why the two orders meet at x*.

## 2. The formulation

### 2.1 Variables

U = p + m and V = (p − m)/x are even in x, as are a and c, so every field is a series in z = x² with polynomial
equations (check 1):

```
A = 2z U_z − 2cz V_z + ω U_θ − G V = 0
B = 2z² V_z − 2cz U_z + ωz V_θ + zV − (G + (n−1)c) U = 0
2z c_z = G = (n−1) c (a² − 1)
```

The singular combinations S1 = cA + B and S2 = zA + cB carry the factor 2z(z − c²) on V_z and U_z alone (check 2).

### 2.2 The two series

- **Centre**, in z, data V(0) = 2p₁, c(0) = c₀: order k gives U_k, a_k (or X_k in form S), c_k, V_k explicitly, with no
  M × M solve in form A and one (for a) in form S (check 5 for form A). Summed at z_K = 1/2 with kc = 160; its last term
  is below 1e-87 at D 100 (`S64_D100_M8.log`).
- **Horizon**, in s = w/w_K, w = 1 − z, with c(1) = 1 (the gauge) and Q₀ = U − V = 2m_h at the SSH: order 0 gives
  (ωD − (n/2 − 1 + g)) P₀ = (n/2) Q₀, P = U + V; at order k ≥ 1, S1 − S2 gives Q_k pointwise and S2 gives
  (ωD + k(1 − g) − (n/2 − 1 + g)) P_k = (lower orders), one M × M solve (check 4). The variable s keeps the shadow's
  coefficients of order one in floating point (in w they reach 2^n). Summed at s = 1 with ks = n + 14√n + 80.
- Neither sum cancels: the free field's horizon-regular solution summed at x = 0.7 loses at most 0.10 digit as a series in
  w and as a series in u, n = 40 to 320 (check 6). My first guess, that u cancels, was wrong; w is chosen for the parity.

### 2.3 The a field: forms A and S

- **Form A** (hp's): a by the Hamiltonian equation, a recurrence in x whose homogeneous solution is the mass mode. On the
  horizon side β = (n−1)(1 − 1/a²) is swept from the anchor k* in both directions, each its stable one, so a_h is an
  output (`fseries.py`, `horizon_anchor`): at D 100 the horizon side's a_h matches hp's to 3.65e-13 relative to a_h − 1
  (`check_state_D100.log`).
- **Form S** (Gundlach's, gr-qc/9604019 Eq. 19; proposed by #1089 slice 3): with X = 1/a² − 1,
  (ω∂_θ − (n−1) − q_S) X = q_S, q_S = ((x − c)p² + (x + c)m²)/x = (U² + zV²)/2 − cUV, one periodic M × M solve at each
  order of each series and no x-recurrence. At the horizon q_S(0) = Q₀²/2 = 2m_h², so a_h follows from m_h. On hp's
  states the slaving holds at every node to 2.5e-12 (D 100) and 1.5e-11 (D 245) relative (`check_s_D100.log`,
  `check_s_D245.log`).

### 2.4 Rows, unknowns and Δ

Unknowns: p₁, c₀, Q₀ on M points of the half τ grid θ_j = πj/M, and Δ (3M + 1). Rows: U, V, c matched at z_K = 1/2
(3M) and the phase row Σ_j sin(πj/M) p₁_j (hp's). In form S the a field matches by itself, since X is a function of
(U, V, c) at each z. Δ enters as ω = 2π/Δ in both series' θ-derivatives.

## 3. Form A: the mass mode at the order of the scalar's largest gain

`anchor_noise.py` (log `anchor_noise.log`, M = 4, float64, the D 100 state rescaled to each D):

| n | k* | gain of one s-coefficient of β at k* on U(x*) | a-datum column, complex step against differences |
|---|---|---|---|
| 20 | 7 | 1.06e-01 | 1.9e-06 |
| 40 | 17 | 3.51e+04 | 1.8e-06 |
| 60 | 27 | 2.26e+08 | 2.2e-06 |
| 70 | 32 | 2.06e+10 | 2.8e-04 |
| 80 | 37 | 1.78e+12 | 7.3e-02 |
| 90 | 42 | 1.14e+14 | 7.8e-01 |
| 98 | 46 | 2.70e+15 | 1.69 |

The discrepancy is the differences' own error (about 2e-6) up to n = 60; above it grows with slope 0.135 per unit n,
against log10(2)/2 = 0.1505 for ε·2^(n/2) (PASS within 30%; numerical evidence). So form A with the datum factored (FACTORED_HORIZON §3, or the anchor here) still needs about
0.15n digits for the a direction. This is the same rate as hp's 0.155n (LARGE_D_BASICS §4, there attributed to G at
0.7, (n − 1) log10(1/0.7)); at x* the two are one number, 2^(n/2) = (1/x*)^n. A layout that sweeps β forward from a_h
(hp's horizon block, `A_h` in `cond_table.py`) has σ_max between 1.8e74 and 2.4e115 at D 60–100 in float64, at form
S's states (`cond_M6.log`).

## 4. Checks against hp's states

- Form A's series reproduce hp's first and last nodes at D 100: V's harmonics 1–7 at x = 0.7 to 4.7e-11–8.2e-10
  (`check_state_D100.log`); the control (the SSH exponent without g) moves them by 1.7e-4–2.4e-3 and FAILs.
- Form S's series reproduce them too: harmonics 1–7 to 1.5e-10–3.4e-9 (`check_s_D100.log`). At D 245 the horizon
  series built from hp's m_h on hp's grid (32 points, harmonics 15 and up pinned, 5–13 held) diverges, 1e14 at
  x = 0.7 (`check_s_D245.log`). Reported, not explained: a reading (inferred) is that hp's pinned state is not
  horizon-analytic in its high harmonics, so the series carries a large shadow, and the shadow's square a second one at
  twice the order, beyond ks.
- The Hamiltonian equation, which form S does not impose: see check 4 of `check_s.py` (log).

## 5. Newton, the conditioning, and Δ

### 5.1 The solution at D 100

Form S, M = 8, from hp's D-100 state cut to 8 points: fsolve.py's float64 Newton (complex-step Jacobian, backtracking)
reaches max|F| = 5.5e-10 in 10 iterations (`S64_D100_M8.log`), then solve_dd.py in double-double (`Sdd_D100_M8.log`):

- The Jacobian is exact: forward-mode dual numbers in double-double (`ddnp.Dual`). It agrees with the float64 complex
  step to 6e-10 column by column. Central differences are useless here: along some directions the residual's third
  derivative is large enough to spoil any step size.
- Plain Newton's step overshoots along the weak directions, so the weak directions are split off (`lsnewton.py`, a
  Lyapunov–Schmidt reduction). The strong rows are solved by a damped chord iteration. The reduced residual
  g(b) = U_wᵀ F, with r = 6 weak coordinates, is solved by Newton with a difference Jacobian.
- Result: max|F| = 1.9e-22 and Δ_M8(100) = 1.7449155179348. The table's 1.7450264844326 is 1.1e-4 higher, the τ
  truncation at harmonic 7. The scaled singular values at the solution run from 1.9 down to
  5.17e-10, 5.15e-10 (harmonic 3), 1.37e-13, 1.36e-13 (harmonic 5) and 2.58e-17, 2.14e-17 (harmonic 7). That is the
  ladder of the free field's grips (LARGE_D_BASICS §3.4: at n ≈ 98 about 2e-8, 2.5e-12 and 1e-15 for j = 3, 5, 7),
  each about 20–40 times smaller. The grip on Δ is |e_Δᵀ J⁻¹| = 5.3e10 with unit rows.
- Rounding: every series coefficient U_k, V_k multiplied by (1 + ηr) moves the residual by 0.77η (horizon) and 0.04η
  (centre) (`s_noise_D100.log`). The control, absolute noise of U(1)'s size at every horizon order, moves it by 1.0e18η
  (`s_noise_D100_mutate_rough.log`): the scalar's horizon map has the 2^(n/2)-type gain, but at a solution the
  coefficients near order σ_R are tiny and nothing is amplified.
- M = 6 in float64 at D 100: max|F| 8.7e-14 and Δ_M6 = 1.7487024, 3.7e-3 above the table. The scaled σ_min is 1.6e-13,
  the harmonic-5 pair (`cond_M6.log`). So Δ_M − Δ is +3.7e-3 at M 6 and −1.1e-4 at M 8 (numerical evidence of τ
  convergence; M 10 in `Sdd_D100_M10.log` if it ran).

### 5.2 Off the solution: the shadow's gain

`seed_precision.py` (log `seed_precision.log`) takes the D-100 solution rescaled to D (p₁, m_h by √(n₀/n), c₀ − 1 by
n₀/n, Δ from the 1/D fit), the seed of a jump. It sums its horizon series at x* in float64 and in double-double
(banded at kb = 150, which equals the full recurrence at a solution):

| D | float64 against dd, relative | dd sum / \|U(1)\| |
|---|---|---|
| 100 (the solution) | 2.0e-14 | 0.54 |
| 125 | 8.1e-13 | 0.53 |
| 150 | 5.8e-12 | 0.52 |
| 175 | 1.0e-10 | 0.52 |
| 200 | 3.8e-6 | 0.52 |
| 245 | 2.5 | 0.51 |
| 300 | 2.1e7 | 27 |
| 400 | 1.8e7 | 2.2e16 |

The horizon coefficients of such a seed have a trough near order σ_R and then the shadow's rise, by about 2^(n/2), to
its peak near order n. At D 400 the coefficients are 1e-40 at order 199 and 2e19 at order 398 (`cost.py`'s seeds;
`cost.log` shows the sums). Rounding at the trough is amplified the same way. So evaluating the residual at a seed
needs digits growing with D: float64 fails past D ≈ 200, double-double past about 250–300. At D 400 the double-double
sum is 2e16, either the seed's true shadow or still rounding; a quad-double sum would tell (not done). This is the
remaining exponential of form S. It is the rough-perturbation gain of section 3, the same 2^(n/2) = (1/x*)^n, now
acting through the scalar field itself rather than through the mass mode. At a solution it is dormant (5.1).

### 5.3 Jumps and continuation (none converged)

- D 100 → 150 at M 8 (`Sdd_D150_M8.log`): float64 Newton takes max|F| from 1.6e-2 to 5.8e-8. The weak residual of
  1.5e-12 then needs weak moves of about |g|/σ_w, far outside the curvature's range. The split Newton accepts only
  steps of 1/512 and the strong residual grows to 4.8e-9.
- D 100 → 105 at M 8 (`cont_M8.log`, a continuation step with the linear predictor): |g| = 6e-13 and no weak step
  reduces it. At that point the weak singular values are 4.6e-10, 4.5e-10, 1.8e-13, 1.4e-13, 2.9e-14, 7.6e-17. They are
  close to the D-100 solution's, but the reduced problem is too curved.
- From hp's states at M 6 (`S64_D200_M6.log`, `S64_D245_M6.log`, `S64_D263_M6.log`): at D 200 Newton stalls at
  max|F| = 2.6e-7; at D 245 and 263 it diverges, Δ to 1.29 and 1.23. hp's states are pinned (harmonics 15 and up zero,
  5–13 held), so their 6-point cuts are seeds, not solutions, and 5.2 applies.

Reading (inferred): a jump needs its seed's weak coordinates (the high harmonics' data) right to about
σ_min²/curvature, and its shadow below 2^(−n/2) of the trough. Neither a rescaled solution nor a pinned state is that
close. The project's chain manages with small continuation steps in octo-double and the high harmonics held, which
is consistent with this.

## 6. What blocks the next step

1. **The shadow's 2^(n/2) off the solution** (5.2). A formulation in which a fixed precision serves every D has to
   carry the horizon coefficients from order σ_R on in factored form. To leading order they are those of the shadow,
   C(k, σ)λ^k times an amplitude smooth in k/n (FACTORED_HORIZON §1). Factored that way, the injection at order σ_R would
   become an explicit unknown, the shadow's amplitude, and the selection its vanishing. That is the Stokes constant of
   the slow manifold, the object of the inner problem in ξ = x/√n (#1089 slice 2's Rozali–Way region, E2), whose cost is
   flat in n but about 5.2 digits per harmonic kept (BEYOND_LIGHT_CONE §7). Not attempted.
2. **The weak directions' basin** (5.3). With every harmonic free, the smallest singular values are the selection
   grips, 2.6e-17 at harmonic 7 at D 100, and Newton's basin there is about their square. Jumps need either a
   predictor of that accuracy or the high harmonics held, as hp holds them.
3. **The τ truncation** (5.1). Δ_M8 is 1.1e-4 below the table at D 100. A Δ to 1e-10 needs M ≈ 16, whose weakest
   grips, about e^(−πjω/2) at j = 15, ask for about 45 digits: quad-double, not available in multidouble's Python
   package.
4. **Cost.** ks = n + 14√n + 80. The banded recurrence (`fseries_sdd.py`, `jset`; kb = 150) keeps only the products
   with one factor in the slow part. At the D-100 solution it equals the full recurrence exactly, residual and Jacobian
   column alike (`cost.log`, and the comparison in `fseries_sdd.py`'s test). One double-double residual then takes
   0.55 s (D 100), 1.00 (245), 1.13 (400) and 2.98 s (D 1000); an exact Jacobian is 3M + 1 = 25 of them, times about
   2.5 (`cost.log`). At the rescaled seeds the banded and full sums differ (1.2e-7 at D 245), as they should: the
   dropped products are the seed's shadow squared. cost.py's PASS criterion, written for solutions, FAILs there.

**What would test the formulation at large D, as stated:** a converged solution at a second D, at M 8 and in
double-double. It could come from a continuation from D 100 in steps small enough for the weak basin, or from a
better-held seed. With it, the weak singular values against D would decide algebraic against exponential decay (#941's
e^(−0.41) per unit D, or the free field's power laws). This prototype did not reach one.

## 7. Commands

```
bash solver/factored/run_track_b.sh OUT STATES            # every check and solve, logs into OUT
python3 solver/factored/derive_factored.py [--mutate=1..7]
python3 solver/factored/check_state.py STATES/anm_pin3_D100.0000.npz [--mutate=sigma]
python3 solver/factored/check_s.py STATES/anm_pin3_D100.0000.npz [--mutate=noslave|evenD|sigma]
python3 solver/factored/anchor_noise.py STATES/anm_pin3_D100.0000.npz
python3 solver/factored/fsolve.py --form=S --D=100 --M=8 --start=STATES/anm_pin3_D100.0000.npz --out=S64.json
python3 solver/factored/solve_dd.py --D=100 --M=8 --start=S64.json --out=Sdd.json
```

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
| The conditioning of form S at fixed M against D | (in progress; section 5) | |
| Is a fixed precision enough for every D? | (in progress; section 5) | |
| What still grows with D | the horizon series' length ks ∝ n (the selection sits at orders about n/2 to n), cost O(ks² M) per evaluation (section 6) | proved by hand (the count of orders), numerical evidence (the profile) |

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

## 5. Newton, the conditioning against D, and Δ

(in progress)

## 6. What blocks the next step

(in progress)

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

# LARGE_D_BASICS: the large-D DSS problem from basics, set against our solver and EEG26b

Issue #1086. Independent review by a subagent on the Mac, 2026-11-24. Sections 1–4 were derived before reading the
project's large-D notes; sections 5–6 compare; section 7 was written after reading notes/LARGE_D_WALL.md,
notes/RT_FORMULATION.md, notes/QUAD_DOUBLE.md §12–13 and issues #91, #731, #903, #966, #1008 and #1035.

Scripts are in [`solver/basics/`](../solver/basics/); `bash solver/basics/run_all.sh OUT STATES` reruns every check with
its control (the clone step for STATES is in the script's header). Raw logs: folder `I1086-mac/` of the raw-data
repository. Status words as in AGENTS.md. Every number below comes from a log in `I1086-mac/` or from a file named
beside it.

## 0. Verdicts

| question | answer | status |
|---|---|---|
| Which condition fixes Δ? | analyticity of p at the SSH, i.e. the absence of the fast scalar mode F; no single row | proved (count), §1 |
| SSH Frobenius exponents | σ_j = (n/2 − 1 + ḡ − ijω)/(1 − ḡ), j odd, ḡ = (n−1)⟨a_h² − 1⟩; ≈ n/2 + 2.6 on the chain | checked symbolically, §2 |
| Amplitude at large D | n⟨p₁²⟩ = 3.86–3.89 from D 226 to 263.6: weak field, p₁ ∝ n^−1/2 | numerical evidence, §3.1 |
| Leading order | a continuum: any profile p₁(τ), any Δ | checked symbolically, §3.2 |
| Selection | the coefficient of F, a mode singular at both ends; O(1) only near x* = 1/√2 | proved (free field), §3.4 |
| How the grip weakens | free field: algebraically in n, as e^−(πω/2)j in the harmonic j | numerical evidence (mpmath), §3.4 |
| ks for the SSH exponent | ks ≥ σ_R + 7 keeps Δ to ≤ 3×10⁻¹¹; the fold at 259.9 is σ_R crossing ks = 130 | numerical evidence, §5 |
| ks for the mass mode | order 293 leaves a relative tail of 5×10⁻³⁰ at D 262 (1.7×10⁻³⁰ measured, #1035) | numerical evidence, §4 |
| The chain past D 254 | Δ swings around the D ≤ 61 1/D series, period ≈ 4.3 in D, amplitude ×e^0.5 per unit D, 1.07×10⁻⁶ at 262.57 | numerical evidence, §6.3 |
| EEG26b's Table I | 0.5–4.8×10⁻⁶ above both our chain and the D ≤ 61 1/D series at D 100–200 | numerical evidence, §6.2 |
| Δ analytic in 1/D? | supported up to terms the D ≤ 61 data bound below 10⁻¹⁴ there; not proved | conjecture, §8 |

**The decision #1086 informs.** The derivation finds nothing in the continuum problem near D 265. In the free field the
selection weakens algebraically, the scales grow linearly in n, and no layer appears. The chain's Δ, though, leaves the
smooth curve with a growing oscillation from D ≈ 254. Both point to the discrete system, not the solution. The route
past 265 is our formulation with every requirement of §4 tied to D, not a new formulation: ks from the mass-mode tail,
the Taylor plan and series tails at the arithmetic's precision, and the held harmonics released (§5, tests T1–T4).

## 1. The problem

**Equations** (`hp/src/model.rs` header; n = D − 2, θ = ωτ, ω = 2π/Δ, x = r/(−t) normalized so the SSH is at x = 1):

    x c_x = (n−1) c (a² − 1)
    2x a_x = a (p² + m²) + (n−1) a (1 − a²)
    x(x − c) p_x = −ωx p_θ + G p − c p + (n/2) c (p + m)
    x(x + c) m_x = −ωx m_θ − G m + c m − (n/2) c (p + m),      G = (n−1) c (a² − 1)

Here c = α/a, and ∂_τ + (x ∓ c)∂_x are the null directions. p + m ∝ r∂_rφ and p − m ∝ rΠ, the scalar's spatial and time
derivatives (inferred from their centre behaviour, p + m = O(x²), p − m ≈ 2p₁x). I took these as given; the project
checks them against the covariant equations elsewhere.

**Gauge.** The slicing freedom that keeps the coordinates adapted to the DSS is τ → τ + h(τ) with h periodic, which maps
x → x e^h. It is fixed by placing the SSH at x = 1 for every τ, i.e. c(τ, 1) = 1, one function. A constant shift of τ
remains, which the phase row fixes (proved by hand).

**Counting.** Per τ-harmonic, the four first-order equations in x have, at the centre, indicial exponents 1, 1 − n (p,
m) and 1 − n (a − 1) (§2). Regularity there leaves p₁ and c₀, two functions. At the SSH only p's equation is singular.
Its regular solutions take m_h and a_h freely, with c_h = 1 (the gauge) and p_h from the periodic solve of p's equation
at x = 1, two functions. The matching gives 4 functions of conditions for 4 functions of data plus Δ. The phase row
takes the extra unknown. Without the SSH condition the local solutions there add the singular coefficient C(τ) of u^σ
q(θ), u = 1 − x, one function more. So **Δ is fixed by the SSH analyticity condition jointly with the profile**, not by
any single row (proved by hand). Section 3.4 says what that condition becomes at large D.

## 2. Local exponents

`derive.py` checks 1–2 (sympy; each check has a `--mutate` control that fails; log `derive.txt`).

- **Centre.** (p, m) ∝ x^s with s = 1 (p = −m, the regular branch, p ≈ p₁x) or s = 1 − n (p = m, the singular fast
  scalar mode). a − 1 ∝ x^(1−n), the singular mass mode. Checked symbolically.
- **SSH.** With c = 1 − g u + O(u²), g = c_x(1) = (n−1)(a_h² − 1), p = u^σ q(θ) gives
  ω q_θ = [(g − 1 + n/2) − σ(1 − g)] q. Periodicity in the odd sector forces
  **σ_j = (n/2 − 1 + ḡ − ijω)/(1 − ḡ), j odd**, with ḡ the θ-average when g depends on θ. The θ-independent case is
  checked symbolically; the averaging is proved by hand. This is #731's Re s = n/(2(1 + c₁)) − 1 with c₁ = −ḡ,
  derived independently.
- **On the chain** (`sigma_vs_ks.py`, logs `sigma_vs_ks_ref_*.txt`): ḡ = 0.0274 at D 258, n ḡ ≈ 7.0, and
  σ_R = Re σ = 120.98 at D 238.76, 130.0 at D 256.76 and 133.41 at D 263.60. So σ_R ≈ n/2 + 2.6.

## 3. The large-D structure

### 3.1 Amplitude

The stored states have n⟨p₁²⟩ = 3.888 (D 226.13), 3.879 (238.76) and 3.863 (263.60) (`slow_check.py`, log
`slow_check.txt`). So p₁ ∝ n^−1/2 and K ≡ 2p₁² = O(1/n): the critical solution is a weak field at large D. The metric
deviation (n−1)(a² − 1) is 0.003–0.015 on the nodes at D 263.6, and c varies by O(1/n) (numerical evidence). EEG26b's
rescaling ψ = √ε ψ̃ says the same.

### 3.2 The slow manifold: a continuum at leading order

Write u = p + m and v = p − m. The (n/2)c(p + m) terms force u = O(v/n):
u ≈ (x² v_x + ωx v_θ)/(nc). The a-equation relaxes on the scale x/n to (n−1)(a² − 1) ≈ p² + m². At leading order the
τ-derivatives drop out, and for every θ separately

    v = 2p₁ x / √(1 + Kx²),   c = c₀ √(1 + Kx²),   c₀ = 1/√(1 + K),   (n−1)(a² − 1) = Kx²/(1 + Kx²),

for **any** K(θ) = 2p₁(θ)² and **any** ω. Checked symbolically (`derive.py` check 3: the four residuals fall like 1/n
for a θ-dependent K). EEG's 2601.14358 states the same freedom ("any periodic function β(τ) with any echoing period Δ",
as quoted in notes/LIT_REVIEW.md). The stored states follow it (`slow_check.txt`):

| D | c₀²(1 + 2p₁²) | max\|p − m\| at x 0.7, state / leading order | c / c_pred at x 0.7 |
|---|---|---|---|
| 226.13 | 0.99985–1.00245 | 0.2401 / 0.2417 | 0.99966–1.00002 |
| 238.76 | 0.99986–1.00220 | 0.2336 / 0.2351 | 0.99969–1.00002 |
| 263.60 | 0.99989–1.00181 | 0.2224 / 0.2237 | 0.99975–1.00001 |

Control `--mutate=K2` (K = 4p₁²) moves c₀²(1 + K) to 1.056 (`slow_check_mutate_K2.txt`).

**To all orders in 1/n** the slow problem stays underdetermined. At each order the correction solves an ODE in x whose
leading operator has no τ-derivative and is regular at x = c. The centre picks the regular homogeneous solution, and the
SSH imposes nothing on slow fields. So the slow expansion has no solvability condition, and the selection is not in it
(conjecture: argued, not proved). An NLO "selection" obtained from the slow expansion alone cannot be right. That
agrees with the project's refutation of EEG's NLO condition (physics paper, Sec. eeg).

### 3.3 The fast modes

With (u, v) ∝ e^{nS}, the eikonal is (x² S_x + ωx S_θ)² = x c S_x (x c S_x + c) (checked symbolically, `derive.py`
check 4). Its nonzero root at fixed θ is S_x = λ = c²/(x(x² − c²)). On the slow solution,
∫λ dx = −ln x + ((1 + K)/2) ln(1 − x²), so

- **F, the fast scalar mode**: F = x^−n (1 − x²)^{n(1+K)/2}. It is singular at the centre (x^−n) and at the SSH (u^σ,
  with σ = n(1 + K)/2 at leading order, matching §2), and it decays monotonically outward.
- **A, the mass mode**: a − 1 ∝ x^−(n−1). It is singular at the centre and analytic at the SSH, and it decays outward.

Both are singular at the centre, so the centre series removes them. Outward shooting is stable for both. F's modulus
crosses 1 at x* where −ln x + ½ ln(1 − x²) = 0, **x* = 1/√2** in the flat limit. With K = κ/n the K term adds (κ/2) ln(1
− x²) = O(1) to the exponent and moves x* by O(1/n).

### 3.4 The selection: where the information lives

In the flat limit the regular-at-centre self-similar wave with e^{λτ}, λ = ijω, is
f = ₂F₁(λ/2, (λ+1)/2; (n+1)/2; x²) (`derive.py` check 5). At x = 1 it splits into A·(regular) + B·(1 − x²)^{n/2 − λ}(…).
The second part is F, and B = Γ(c)Γ(a + b − c)/(Γ(a)Γ(b)) vanishes only for λ ∈ {0, −1, −2, …}. **No harmonic λ = ijω
is regular at both ends: a free wave cannot echo** (proved by hand). The project's physics paper has the same statement
as a section title.

`free_field.py` gives the weight of F relative to the regular part at x* (mpmath at 120 + n digits; the connection
identity closes to 10⁻¹¹⁰ or better; log `free_field.txt`; control `--mutate=A` flips the j-slope's sign). ω = 3.73:

| j | n 60 | n 120 | n 240 | n 480 | n 960 |
|---|---|---|---|---|---|
| 1 | 6.37e-04 | 4.01e-04 | 2.67e-04 | 1.83e-04 | 1.28e-04 |
| 3 | 5.74e-08 | 1.43e-08 | 5.99e-09 | 3.26e-09 | 2.02e-09 |
| 5 | 2.31e-11 | 9.56e-13 | 1.59e-13 | 5.45e-14 | 2.68e-14 |
| 7 | 4.47e-14 | 1.45e-16 | 6.16e-18 | 1.05e-18 | 3.66e-19 |
| 9 | 3.56e-16 | 5.10e-20 | 3.61e-22 | 2.46e-23 | 5.37e-24 |
| 11 | 9.17e-18 | 4.09e-23 | 3.25e-26 | 7.05e-28 | 8.64e-29 |
| 13 | 5.89e-19 | 7.26e-26 | 4.52e-30 | 2.50e-32 | 1.53e-33 |

- **In j**: d ln R/dj tends to −πω/2 = −5.86 (−2.86 at n 60, −5.56 at n 960). The grip on harmonic j falls
  exponentially in j. This is the ladder of weak directions.
- **In n**: algebraically, n^−0.58 for j = 1 and n^−11.8 for j = 13 over n 60–960. There is **no e^−cn**.
- **In x**: F is O(B) only near x*. It is exponentially small for x > x* (beyond all orders in 1/n at fixed x) and
  exponentially large for x < x*. A match at x_K costs about n|ln(√(1 − x_K²)/x_K)|/ln 10 digits: 0.0087n at
  x_K = 0.7 and 0.125n at 0.8. **The project's last node at 0.7 is near the optimum.**

So the selection is "beyond all orders" only when read at a fixed x > x*. Read at x* it is algebraically weak in the
free field. How the nonlinear O(κ) terms, κ = nK ≈ 7.7, turn the free obstruction into a solvable condition that fixes
the profile and Δ was not derived here (open). The measured weakening is faster than the free field's: σ_Δ falls e^−0.41
per unit D at order 240 (#941), and max|∂Δ/∂F| ∝ n^21.7 at D 40–61 (LARGE_D_WALL). Section 7 lists this as open.

### 3.5 No layers

At the critical amplitude, no field varies on a scale shorter than O(1) in x: the slow fields are smooth on [0, 1], and
the centre and SSH are regular singular points with exponents ∝ n. The "layers" are the fast modes' scales x/n, present
in every perturbation but not in the solution. Harmonics with j ≳ √(2n)/ω feel the heat-equation damping
e^−j²ω²x²/(2nc²) of the slow manifold's τ-structure, so the high harmonics become *less* damped as n grows (a
leading-order estimate from the flat heat kernel; conjecture).

## 4. What a discretization must resolve as D grows

`order_laws.py` (log `order_laws.txt`, tails in `order_laws_tails*.txt`; control `--mutate=binom` makes the
orders flat in n):

| n | σ_R | ks, mass-mode tail 2⁻²¹² | ks, 2⁻⁴²⁴ | mass-mode gain at 0.7, digits | F's decay 0.3 → 0.7, digits |
|---|---|---|---|---|---|
| 120 | 62.7 | 287 | 447 | 18.4 | 59.2 |
| 240 | 122.6 | 398 | 580 | 37.0 | 118.5 |
| 263 | 134.1 | 417 | 603 | 40.6 | 129.8 |
| 300 | 152.6 | 448 | 640 | 46.3 | 148.1 |
| 400 | 202.6 | 526 | 732 | 61.8 | 197.5 |
| 1000 | 502.5 | 936 | 1206 | 154.7 | 493.7 |

- **Horizon order, SSH exponent:** ks ≥ σ_R + about 7 (§5, D3). It grows like n/2.
- **Horizon order, mass mode:** every a_h column of the horizon block carries (1 − u)^−(n−1), whose series in u has
  terms C(n−2+k, k)u^k peaking at k ≈ (n−2)u/(1−u) = 0.43n at u = 0.3. Its relative tail beyond ks:

| D | ks 130 | ks 293 | ks 315 | ks 330 | ks 360 | ks 420 |
|---|---|---|---|---|---|---|
| 150 | 1.13e-9 | 6.26e-57 | 1.04e-64 | 3.93e-70 | 3.04e-81 | 2.34e-104 |
| 200 | 6.8e-5 | 6.97e-43 | 1.16e-49 | 1.98e-54 | 2.77e-64 | 4.38e-85 |
| 232.63 | 0.00519 | 1.34e-35 | 8.86e-42 | 3.75e-46 | 3.01e-55 | 1.21e-74 |
| 262.01 | 0.0646 | 5.05e-30 | 1.08e-35 | 9.92e-40 | 3.52e-48 | 2.26e-66 |
| 265.08 | 0.079 | 1.74e-29 | 4.18e-35 | 4.15e-39 | 1.71e-47 | 1.46e-65 |
| 300 | 0.397 | 6.13e-24 | 5.46e-29 | 1.29e-32 | 2.82e-40 | 5.43e-57 |

  #1035 measured the order-293 state at D 262.0099 at 1.476×10⁻⁴¹ on its own system and 1.671×10⁻³⁰ at order 315. The
  table's tail difference there is 5.05×10⁻³⁰, the same order (numerical evidence, one point). **Order 293 is a
  10⁻³⁰-accurate horizon block at the front**, inside an octo-double chain whose corrector stops at 10⁻⁴⁰.
- **Digits:** the mass-mode gain 0.155(n − 1) digits at x_K = 0.7, plus the match's 0.0087n (§3.4), plus the weak
  directions' gain. All are linear in n or algebraic; none is exponential in n² or worse.
- **Taylor steps:** the fast rates are n c²/(x(c² − x²)) and (n−1)/x. A step of order 36 resolves F to 2⁻⁴²⁴ only for
  h ≲ 5×10⁻⁶ at x = 0.3, n = 263. The chain's plans resolve the solution, not F. `plan_stability.py` on the stored plans
  (log `plan_stability.txt`; control `--mutate=h2` raises the error by about 2³⁷):

| D | steps | max \|nλ_F h\| | F's transfer error per step | max \|T₃₆\| | max \|nλ_A h\| |
|---|---|---|---|---|---|
| 259.90 (order 130) | 61 | 11.90 | 5.1×10¹ | 0.179 | 10.22 |
| 256.53 (order 293) | 23 | 14.02 | 1.75×10⁵ | 0.143 | 12.33 |
| 263.60 (order 293) | 23 | 14.16 | 2.89×10⁵ | 0.205 | 12.38 |

  The discrete step still damps both modes (|T| < 1), but F's damping per step is wrong by up to 3×10⁵.
- **τ-harmonics:** p₁'s solved and held harmonics fall about ×15 per odd step, and c₀'s harmonic 30 is 3.1×10⁻¹⁷ at
  D 238.76 (`slow_check.txt`). The grip on harmonic j falls like e^−(πω/2)j but only algebraically in n (§3.4). So
  the number of harmonics that can be *solved* at a fixed precision is roughly independent of n. Nothing in the
  derivation makes M grow with D.

## 5. Departures of our solver, ranked by how likely each caps the computation

Each test can fail. "Pass" is what the derivation predicts.

**D1. Holding p₁'s harmonics 5–13 on a fit, and pinning 15 and up to zero** (ALGORITHM.md §3). The pinned system drops
the horizon block's p-matching rows of harmonics 5 and up (`kfreeze` 3) and fixes those unknowns instead, so it is
a different problem. Its singular points need not be the continuum's. The
held values come from a fit on D 38–73 extrapolated in 1/n to 1/261. The derivation predicts no feature of the continuum
near D 265 (§3), while the chain's Δ oscillates with a growing amplitude from D 254 (§6.3), the signature of a discrete
system near its own singular point. *Most likely cap.*
- **T1.** Solve the full system (every harmonic free) in octo-double at one point of the oscillation, D 262.57, from the
  chain's state. Pass: Δ moves toward the D ≤ 61 1/D series by about 1.07×10⁻⁶ (§6.3), and the Δ-mode's σ is not
  smaller than the pinned one. Fail: Δ stays.
- **T1b**, cheaper: free harmonics 5 and 7 only, at D 258.69 and 262.57 (opposite extrema). Pass: the two departures
  shrink by the same factor.

**D2. Fixed truncation levels below the arithmetic's.** The plan is made at 10⁻³² (`--plan_tol`; kept fixed across legs
by `--keep_plan 1`). The series tails are cut at 10⁻³³, and ks = 293 leaves a mass-mode tail of 5×10⁻³⁰ at 262 that
grows e^0.4 per unit D (§4). All of these sit far above od's 10⁻¹²⁷ and the corrector's 10⁻⁴⁰. Each changes the
discrete system by its own amount, and the weak directions' gain (about 3.4×10²¹ at 259.6, #1008) carries it into Δ.
#461 already saw a replan move Δ by 10⁻¹⁵ past D 230.
- **T2.** At D 262.57, evaluate (no solve) the stored state's residual at ks 293, 315, 330, 360 and 420. Pass: the
  change from 293 is about 10⁻²⁹ and flat from 315 on (the tails beyond 315 are below 4×10⁻³⁵). Fail: it keeps
  changing.
- **T2b.** Raise the order by a homotopy in the truncation (weight s ∈ [0, 1] on the terms 294…420), continued in s
  by the ANM. A jump changes the system by about 10⁻³⁰ against a weak-direction basin of about 10⁻¹⁶ (#944), which is
  why every raise at the front failed (#1035, #1064). Then read the singularity at ks 420. Pass for "the order's":
  265.08 moves.
- **T2c.** Replan at 10⁻⁴⁸ with series tails at 10⁻⁵⁰, same order. Pass: Δ at 262.57 moves by less than 10⁻¹².

**D3. Fixed horizon order against σ_R** (resolved by #731's rule; listed for the record). `sigma_vs_ks.py` (log
`sigma_vs_ks_ref_chain_k135.txt`), order-130 chains against the order-135 chain's cubic spline (leave-one-out error
1.6×10⁻¹¹ median, 4.2×10⁻¹¹ max):

| D | ks − σ_R | Δ(130) − Δ(135) |
|---|---|---|
| 236.13 | 10.34 | −2.9×10⁻¹² |
| 242.71 | 7.05 | −4.0×10⁻¹² |
| 246.26 | 5.27 | −8.5×10⁻¹¹ |
| 249.01 | 3.89 | +3.3×10⁻⁹ |
| 252.61 | 2.09 | −3.0×10⁻⁷ |
| 254.06 | 1.37 | +4.7×10⁻⁶ |
| 255.77 | 0.51 | +4.3×10⁻⁵ |
| 256.95 | −0.09 | −2.7×10⁻⁴ |
| 258.07 | −0.65 | −3.5×10⁻³ |

The order-130 fold at D 259.90 lies at σ_R − ks = +1.55. Between rows the departure falls by e^−2.5 to e^−3.8 per unit
of margin, and every row with a margin of 6.6 or more is within 3×10⁻¹¹ (numerical evidence). The mechanism: an order
below σ_R does not reach the order where the analytic solution and u^σ differ, so the truncated series imposes no
analyticity at all (EEG26b's SM states the same for Gundlach's Taylor expansion). #731's ks ≥ ⌈1.95 Re s⌉ + 10 is safe
by this measure. Control `--mutate=g` (σ with ḡ = 0) moves σ_R by 4.4 (`sigma_vs_ks_ref_chain_k135_mutate_g.txt`).

**D4. Taylor steps that misrepresent the fast modes** (§4: up to 2.9×10⁵ per step in F's transfer at D 263.6). F carries
the selection, so a wrong transfer of F between nodes changes the discrete selection. *Moderate likelihood.*
- **T4.** At D 262.57, a plan with every |nλ_F h| ≤ 4 (about 3.5 times more steps). Pass: Δ moves by less than 10⁻¹².

**D5. M = 32 at every D.** τ-truncation near 10⁻¹⁷ in c₀. The octo-double kernel is M = 32 only, so M was never
varied above D 73. *Low likelihood for the cap; unknown for Δ at 10⁻¹².*
- **T5.** M 48 in quad-double at D 200, against the chain. Pass: Δ within 10⁻¹².

**D6. The last node at 0.7 and the first at 0.3.** These are near-optimal for F's crossover (§3.4); a node at 0.8 would
cost 0.125n digits. *Not a cap.* No test needed beyond the node test of LARGE_D_WALL
(`solver/beyond_lc/hp_node_test.py`), which already confirms e^{n[Φ(x_m) − Φ(0.7)]}.

## 6. The published setups

### 6.1 What they do

- **EEG26b (arXiv:2611.07007)**, read from the PDF's text layer (`solver/basics` has no copy; quotes are data). It uses
  RT's gauge and variables on the parity-extended domain, Fourier × Chebyshev with N_Fourier = 40 and
  N_Chebyshev = 80 at every D, and Newton–Raphson with GMP dyadic arithmetic. It rounds "only ... once the Jacobian is
  inverted at each Newton step, which we need to do with increasingly higher precision as D grows". It steps D by 1/2
  with linear extrapolation and reaches D = 200. Its SM says that in GG gauge "the singular branch becomes increasingly
  difficult to distinguish numerically from the regular one ... the exponent grows approximately as D/2, distinguishing
  the two branches by a Taylor expansion requires a number of terms that grows with D/2", which was EEGJ's limit at
  D 5.5. It fits Δ ≈ 1.6439 + 10.37/D, and its D = ∞ profile model is A Re(e^{iθ}/(1 + q e^{i(2θ+δ)})) with q ≈ 0.0551,
  i.e. odd harmonics falling ×0.055 per step. Our p₁ falls ×0.06–0.15 per step on the solved and held harmonics (§4).
- **EEG 2601.14358**: the large-D expansion with ψ = ε^−1/2 Σψ_n ε^n, the leading-order freedom of §3.2, and an NLO
  condition (from notes/LIT_REVIEW.md's reading; not re-read here).
- **EEGJ 2602.10185**: GG gauge (as ours), D 3.05–5.5; not re-read beyond the search snippet and EEG26b's account.

**Departures of ours from theirs.** EEG26b's Chebyshev degree 80 is fixed while σ_R ≈ n/2 passes 80 near D 157. By
D1–D3's logic, a fixed spectral degree imposes SSH analyticity ever more weakly. That is the RT note's measured
conditioning (singular value falling as a power of the resolution, exponent +1.4 per unit D). They solve every
harmonic; we hold 5–13.

### 6.2 EEG26b's values against ours

`eeg_gap.py` fits degree-16 and degree-18 polynomials in 1/D to the 853 double-double points 10 ≤ D ≤ 61.1 of
`hp/data/delta_all.csv` (mpmath QR, 60 digits). It evaluates them at the D where delta_all carries EEG's value (log
`eeg_gap.txt`; control `--mutate=shift` moves chain − fit by exactly −10⁻⁶):

| D | chain − fit (d16, d18) | EEG − fit |
|---|---|---|
| 70 | −1.4×10⁻¹³, +3.1×10⁻¹⁴ | +2.4×10⁻⁷ |
| 100 | −9.5×10⁻¹², +2.0×10⁻¹² | +5.2×10⁻⁷ |
| 150 | −8.8×10⁻¹¹, +1.9×10⁻¹¹ | +3.0×10⁻⁶ |
| 175 | −1.5×10⁻¹⁰, +3.4×10⁻¹¹ | +3.8×10⁻⁶ |
| 200 | −2.3×10⁻¹⁰, +5.1×10⁻¹¹ | +4.3×10⁻⁶ |

Our chain (held harmonics) stays within the two fits' spread to D 200, while EEG's six-digit values sit above by up to
4.8×10⁻⁶ (4.8 units of their last digit, at D 198). The fit shares our solver but not the held fit, so this is numerical
evidence that the gap is EEG's, and that the held fit costs Δ less than about 2×10⁻¹⁰ up to D 200. EEG claim all
displayed digits for D > 28. **Discrepancy reported, not explained.** Their fixed Chebyshev degree (§6.1) is the
candidate. The test is theirs to run.

### 6.3 Our chain past D 245 against the D ≤ 61 series

`eeg_gap.py --at` on every stored chain state (log `chain_vs_dd_fit.txt`; the d16 and d18 columns differ by at most
5×10⁻¹⁰ there):

- Orders 135 and 142, D 226–251.8: −1.02×10⁻⁸ to −7.9×10⁻⁹, smooth and slowly shrinking. That is within the D ≤ 61
  extrapolation's own uncertainty, of the size of Δ∞'s ±1.5×10⁻⁸.
- Order 293, D 238.8–253.8: scatter of ±(0.5–8)×10⁻⁹ about zero. It differs from orders 135/142 by about 1.5×10⁻⁸ at
  the same D. **Discrepancy, not explained**: the chains' other flags differ, and this review did not separate them.
- **Order 293 from D 254: a growing oscillation.** Extrema +4.0×10⁻⁸ (256.85), −9.4×10⁻⁸ (258.69), +2.7×10⁻⁷ (260.93)
  and −1.07×10⁻⁶ (262.57). `oscillation_fit.py` (log `oscillation_fit.txt`) fits A e^{r(D−260)} cos(2π(D − D₀)/P) + b
  with **P = 4.35, r = 0.51 per unit D**, leaving an rms of 8.4×10⁻⁸ of 5.0×10⁻⁷. The same fit on permuted data
  leaves 4.0×10⁻⁷ (`oscillation_fit_mutate_shuffle.txt`), and from D 256 on it gives P 4.33, r 0.50.

  r is close to the Δ-mode's fall, e^0.41 per unit D at order 240 (#941). So the oscillation reads as a fixed-size
  perturbation of the discrete system divided by a weakening grip, the same form as #805's 2.2×10⁻³⁷/p_min on the
  order-130 chain. It is consistent with xfchecks' complex pair P1 at 262.09 ± 0.97i (#966). It precedes the real
  singularity at 265.08.
- **A period near 4 in D has a candidate source in the continuum problem** (conjecture). The free-field connection
  coefficient carries 1/sin(π(ijω − n/2)), whose phase turns by π/2 per unit n, the same for every harmonic. So it is
  not a τ-translation. An error in the discrete transfer of F (D4) or in the held profile (D1) would interfere with the
  true F-component with that phase. The D ≤ 61 points fit a 1/D polynomial to 4×10⁻¹⁵ (paper, Sec. limit, held-out
  table), so the continuum's Δ carries no such term there above 10⁻¹⁴.

## 7. The project's conclusions against the derivation

| conclusion (where) | verdict |
|---|---|
| Δ is an eigenvalue of a periodic BVP fixed by SSH regularity (paper §2) | supports (§1) |
| Re s = n/(2(1 + c₁)) − 1 and ks ≥ ⌈1.95 Re s⌉ + 10 (#731, FACTORED_HORIZON) | supports; the margin needed for 3×10⁻¹¹ is about 7 (§5 D3) |
| The fold at 259.90 is the order-130 truncation's (#1008, LARGE_D_WALL "Order 142 follows track C") | supports, with its mechanism: σ_R crosses 130 at D 256.8 and the fold is at σ_R − ks = +1.55 |
| The fast mode e^−nΦ, Φ = ln x − ½ln(1 − x²), matched at 0.7 near 1/√2 (LARGE_D_WALL, BEYOND_LIGHT_CONE) | supports (§3.3–3.4), derived independently |
| "A free wave cannot echo" (paper, Sec. picture) | supports (§3.4, proved by hand for the free field) |
| The weak directions are a ladder in the harmonic k (LARGE_D_WALL) | supports: free-field grip ∝ e^−(πω/2)k asymptotically |
| The period is selected beyond all orders (paper, Sec. picture) | **partly contradicts**: beyond all orders at fixed x > 1/√2 only; at x* the free-field grip is algebraic in n (§3.4) |
| σ_min ∝ n^−20.8, max\|∂Δ/∂F\| ∝ n^21.7 at D 40–61 (LARGE_D_WALL) | consistent in kind (algebraic) with §3.4; the exponent is not derived |
| σ_Δ falls e^0.41 per unit D at order 240, "the problem's, not the cut's" (#941, #948) | **open**: the free field predicts algebraic decay; this exponential rate is the *pinned* system's. Test T1 separates the two |
| H1: the mass mode makes Newton's basin shrink (QUAD_DOUBLE 13.1) | supports: same mode (A, §3.3); its series tail sets ks's growth (§4) |
| "No method raises the horizon order at the front" (#1035) | explained, not contradicted: a raise changes the system by about the tail (10⁻³⁰), and the gain of 10²¹ puts that outside the basin. T2b is a route |
| The singularity at 265.08 is "real" on the order-293 path (#966) | supports *as a singularity of the order-293 pinned system*; the derivation gives the continuum no reason for one there |
| ks_chain3b's Δ "stays on track C's series to 1e-7" to 261.27 (#1008 closing) | **discrepancy**: against the D ≤ 61 series it is +2.7×10⁻⁷ at 260.93 and −1.07×10⁻⁶ at 262.57 (§6.3) |
| EEG's NLO condition is not a selection (DINF_ANALYTIC) | supports: the slow expansion has no solvability condition at any order (§3.2, conjecture) |
| RT formulation not adopted for the chain (#91) | supports: a fixed Chebyshev degree meets the same σ ∝ n/2 wall (§6.1) |

## 8. Is Δ analytic in 1/D?

What the derivation gives:
- The equations' coefficients are polynomials in n, and the slow expansion is a regular series in 1/n (§3.2).
- The amplitude enters as K = κ/n, an integer power (§3.1).
- The selection enters through connection coefficients that are ratios of Γ-functions of n/2 (§3.4, free field). Their
  Stirling series run in powers of 1/n, with three further pieces:
  - a common power n^−1/2, which drops out of a homogeneous condition;
  - phases n^{ijω} = e^{ijω ln n}, the same for every harmonic up to j, i.e. a τ-shift by ln n, which leaves Δ alone
    (proved by hand);
  - the factor 1/|sin(π(ijω − n/2))|, whose modulus has a relative period-2 modulation 2e^−2πjω sin²(πn/2), 1.3×10⁻¹⁰
    for j = 1 (`free_field.txt`), and whose phase turns by π/2 per unit n.
- After the common n^−1/2 and the τ-shift are removed, the free field's coefficients have Stirling series in integer
  powers of 1/n, whose coefficients grow like (jω)^2k. So no ln D and no half-integer power appears, and the j-dependent
  slopes of §3.4 are pre-asymptotic for large jω (proved by hand for the free field).
- The modulation and the phase are not analytic in 1/n and do not decay with n. If the nonlinear selection inherited
  them with an O(1) coupling, Δ(D) would carry a period-2 or period-4 term of 10⁻¹⁰. The D ≤ 61 fits' residual of
  4×10⁻¹⁵ rules that out at D ≤ 61 (numerical evidence). The coupling at D → ∞ is not known.

So the derivation **supports** the assumption behind Δ∞ = 1.643893746 ± 1.5×10⁻⁸: Δ analytic in 1/D up to terms that are
exponentially small in ω, not in D. That is a conjecture with numerical support, not proved. It predicts no half-integer
powers and no logs. Since Δ∞ is fitted on D ≤ 61.03 only, none of the large-D chain's issues (§5, §6.3) touch it.

## 9. What I could not check

- The nonlinear D = ∞ selection rule: its form, and why the measured grip falls faster than the free field's.
- Any solve of our system: T1–T5 need hp and hours of octo-double. The tests are written to be run as stated.
- EEG26b's equations beyond the text layer (Greek letters were lost in extraction); its Table I was read numerically
  and matched delta_all's `published_Delta` at D 60, 100, 150, 175, 200.
- EEGJ 2602.10185 and EEG 2601.14358 were not re-read; their content here is from LIT_REVIEW.md and search snippets.
- Whether the order-293 chain's ~1.5×10⁻⁸ offset from orders 135/142 at D 240–252 comes from its plan, its acceptance
  flags or its held values.

# RT_FORMULATION: the Reiterer–Trubowitz formulation of arXiv:2611.07007, evaluated for the large-D chain

Issue #91. Subagent of the coordinator on the Mac, 2026-11-08. Code in [`solver/rt/`](../solver/rt/), raw output in
folder `I91-mac/` of the raw-data repository.

**Verdict: adopt in part.** Adopt the map between the gauges with its identity Δ = 4πμ/(1 − (n − 1)⟨a_h² − 1⟩), the
RT prototype as an independent check of Δ(D) at moderate D, and RT's quadratic framework for a certified solution at
one D. Do not adopt the global Fourier × Chebyshev Newton for the chain. It does not avoid our horizon wall. The
horizon's singular branch is nearly a polynomial, and the Chebyshev system's smallest singular value falls like a power
of the resolution whose exponent grows by about 1.4 per unit of D. A coarse grid keeps κ modest but does not hold the
solution (Newton from our solution runs away at D = 20–40), and a grid that does needs κ beyond f64 already at D = 10,
so the resolution and the arithmetic must both grow with D, on 14 to 24 times our unknowns in a dense LU. The
quadratic form of the equations is not what sets the conditioning.

What was run: the D-dimensional RT equations checked in sympy against the covariant ones (section 2), the map checked
against RT's 80-digit μ (7.7×10⁻²⁸), a collocation prototype that reproduces our Δ at D = 4 and 10 to 1.5×10⁻¹³ and
1.1×10⁻¹² (section 3), and the condition numbers of both formulations against D (section 4). EEG's Eq. (5) and their
Eq. (1) use different normalizations of the scalar (section 2), which does not affect Δ.

## 1. The two formulations

### Ours (`hp/src/model.rs`)

The unknowns are metric functions and the scalar's null derivatives in the self-similar coordinates (τ, x) of our
gauge, ds² = e^{−2τ}[a²((x² − c²)dτ² − 2x dτ dx + dx²) + x² dΩₙ²], with the horizon at x = 1 where c = 1:

    x c_x = (n−1)(c a² − c);  2x a_x = a(p² + m²) + (n−1)(a − a³);
    x(x − c) p_x = −ωx p_θ + G p − c p + (n/2) c (p + m);  x(x + c) m_x = −ωx m_θ − G m + c m − (n/2) c (p + m),

with G = x c_x, n = D − 2 and ω = 2π/Δ. The residual is cubic in (a, c, p, m) and quartic through G p, and it is
discretized by multiple shooting: a series at the centre, Taylor steps between nodes, a series at the horizon, and a
Newton solve for the node values, the centre and horizon data, and Δ (L = 4M(K + 1) + 1, 769 unknowns in the chain).

### Reiterer and Trubowitz (RT), as generalized by Ecker, Ecker and Grumiller (EEG)

RT's gauge (arXiv:1203.3766 Sec. 1.1, 1.4, 2.1; EEG Eq. (2) in D dimensions):

    ds² = e^{−2ζ}( −(1 − ξ²) dt² − (2/μ) ξ dt dξ + dξ²/μ² + ξ²/(μ + Q ξ²)² dΩₙ² ),

with the null coordinates u^± = −(1 ± ξ) e^{−μt}. The self-similarity map Θ is t → t + 2π, and it rescales u^± by
e^{−2πμ}. The centre is ξ = 0 and the self-similarity horizon is ξ = 1. RT show (Sec. 1.4, from three facts about germs
of diffeomorphisms) that an ingoing null coordinate with du ≠ 0 across the critical light cone and a linear action of Θ
is unique up to a constant factor, so μ is a geometric invariant and the gauge has no freedom left but a shift of t.
The unknowns are μ and seven frame quantities, six of them null derivatives, which RT and EEG define in Eq. (8) and
Eq. (4):

    ω₁ = 2ξQ + ω₂⁺ + ω₂⁻,  ω₂^± = D^±(ζ + ln(μ + Qξ²)) ∓ μ,  ω₃^± = D^± ζ ∓ μ,  ω₄^± = D^± φ,
    D^± = ±∂_t + μ(1 ± ξ)∂_ξ.

The field equations Ω_i^± = 0 (RT Eq. (11), EEG Eq. (5)) are first order and at most quadratic in (μ, ω). The parity
ω_i^+(t, ξ) = −ω_i^−(t, −ξ) (i = 2, 3, 4), with ω₁ odd in ξ, puts the unknowns on [−1, 1], where four fields ω₁, ω₂⁻,
ω₃⁻, ω₄⁻ and the equations (1 + P)Ω₁⁺, Ω₂⁺, Ω₃⁺, Ω₄⁺ (each divided by ξ) form a square system with μ and the phase
condition Re(ω₄)₁,₀ = 0. The constraints Ω₅ (RT's Ω₂♯) then hold (RT Sec. 2.3, the ♯ system). No boundary condition is
imposed anywhere: in RT's words, frequency space "disposes of the geometric boundaries (central geodesic and especially
the critical light cone) in a seamless way, and one does not specify initial data anywhere". Δ = 2K = 4πμ −
2∫₀^{2π} ω₃(t, 0) dt (RT Eq. (15b)).

RT discretize in coefficient space, Fourier in t (period 4π) times Chebyshev in ξ, with weighted ℓ¹ norms (κ₁ = 65/64,
κ₂ = 5/4) and a reference solution with m < 250 Fourier and n < 750 Chebyshev indices; their D = 4 proof encloses μ and
K to 10⁻⁸⁰ in integer (GMP) arithmetic. EEG use the same infrastructure in D dimensions with 40 Fourier and 80
Chebyshev modes (11,200 unknowns), Newton in GMP dyadic arithmetic with "increasingly higher precision" as D grows,
and steps of 1/2 in D with a linear predictor from 4 to 200.

## 2. The equations and the map, checked

`solver/rt/rt_eqs_check.py` (sympy, about 6 s, exit 0): each of the ten Ω_i^± of EEG's Eq. (5), written in ζ, Q and
φ, is a combination of the covariant components E₊₊, E₋₋, E₊₋ (E_ab = R_ab − κ ∂_aφ ∂_bφ along D^±), the sphere
component and □φ, with coefficients fixed by the second derivatives (a rank-5 system) and an identically vanishing
remainder, for general D. The curvature formulas it uses are checked against brute-force Ricci tensors at D = 4 and 5,
and at D = 4 the equations are RT's Eq. (11) term by term. The controls `--mutate=coef` (one coefficient of Ω₃ scaled
by 1.001) and `--mutate=warp` (the warped-product formula broken) exit 1 (logs `I91-mac/rt_eqs_check*.log`).

**A discrepancy in EEG's normalization (reported, not fixed).** EEG's Eq. (5) holds as printed for
Ric = (D − 2) dφ dφ. Their Eq. (1) reads R_μν = 2 ∂_μφ ∂_νφ (4πG = 1), and with that coupling the ω₄ terms would be
−(2/(D − 2)) ξ (ω₄^±)² in Ω₁ and Ω₅ (printed −ξ (ω₄^±)²) and −ξ ω₄⁺ω₄⁻ in Ω₃ (printed −((D − 2)/2) ξ ω₄⁺ω₄⁻). The
script prints these remainders and checks that they carry the factor D − 4. Their supplement's φ_num with normalization
𝒞 = 1 may be the scalar of Eq. (5). The scalar's normalization does not change Δ, so this is notation, not a result.
Our scalar has Ric = 2n dφ dφ, so EEG's ω₄ is √2 D^±φ in our normalization.

**The map between the gauges** (`solver/rt/gg_to_rt.py`; EEG's supplement VII.2 does the same by integrating
characteristics). Write RT's null coordinates in our coordinates as u^± = −e^{−λτ} w^±(τ, x) with w^± periodic and
λ = 4πμ/Δ. Then u⁻ constant along the ingoing rays and u⁺ along the outgoing ones are the linear transport equations

    w⁻_τ + (x − c) w⁻_x = λ w⁻ (regular at x = 1),   w⁺_τ + (x + c) w⁺_x = λ w⁺,  w⁺(τ, 0) = w⁻(τ, 0).

At the horizon w⁻ = (1 − x) h with h_τ = (λ − 1 + c_x(τ, 1)) h, and a periodic h ≠ 0 exists only if
λ = 1 − ⟨c_x(1)⟩, with c_x(1) = (n − 1)(a_h² − 1) from the c equation. So

    Δ = 4πμ / (1 − (n − 1)⟨a_h² − 1⟩),

our form of EEG's Eq. (3) (⟨ψ₋₁²⟩ there). Numerical check (`solver/rt/mu_check.py`, log `I91-mac/mu_check.log`;
its control `--mutate=n`, n in place of n − 1, exits 1): from our D = 4 double-double state at M = 192 (folder
`T45-win/hp/data/lowD_states/`), μ comes out 7.7×10⁻²⁸ from RT's enclosure 0.168307078963449969510134979042857…
(radius 10⁻⁸⁰), and Δ is 3.1×10⁻²⁶ from their 2K. The M = 160 state gives 8.4×10⁻²⁵ and 1.1×10⁻²² (numerical evidence;
the differences are our truncation in M, the enclosure's radius being negligible). The rest of the map follows from
the metric: ξ = (w⁺ − w⁻)/(w⁺ + w⁻), t = (4π/Δ)τ − ln((w⁺ + w⁻)/2)/μ, e^{−ζ} = μ e^{−τ} a ((w⁺ + w⁻)/2) / √(−w⁺_x w⁻_x),
and D^± = A^±(∂_τ + (x ± c)∂_x) with A^± = −μ(w⁺ + w⁻)/(2c w^∓_x), which gives the ω's from (a, c, p, m):
ω₄⁺ = √2 A⁺ c p / x, ω₄⁻ = −√2 A⁻ c m / x, ω₂^± = μ/ξ ∓ A^± c / x, ω₃^± = D^± ζ ∓ μ. The transport equations are
solved by Fourier × Chebyshev collocation in our (τ, x), with the rows w⁻_x(τ, 1) = −h(τ, 1) fixing w⁻'s scale (they
also exclude a spurious regular solution carried by the Nyquist mode of the even τ grid). The map is exercised as the
first guess of every RT solve below: at D = 4 Newton converges from it, and at D = 20–100 the mapped fields satisfy the
RT equations on a 24 × 21 grid to 10⁻⁷ (section 4), an independent check of the map's formulas (they and the RT
equations share no code).

## 3. The prototype and its checks

`solver/rt/rt_model.py` discretizes the square system of section 1 by collocation rather than RT's coefficient space:
N_t points in t on [0, 2π) with exact Fourier derivatives (integer wavenumbers for ω₁, ω₂, ω₃, half-integer for ω₄)
and the Chebyshev–Gauss–Lobatto points ξ_j = cos(πj/N) with N odd, so that ξ = 0 is not a point and ξ = ±1 are.
L = N_t(N + 1)/2 + 3N_t(N + 1) + 1 unknowns. No row imposes a boundary condition: at ξ = −1 the collocated equation,
whose D⁺ has no ξ component there, is the only regularity condition. The residual is exactly quadratic in (μ, ω), so
each Jacobian column is an exact central difference with unit step. The first guess comes from our D = 4 double-double
state at M = 48 (folder `T106-mac/m48_states/`, whose own Δ is 3.9×10⁻⁷ off at that M), mapped by `gg_to_rt.py`;
Newton converges from it in three steps although the mapped state's residual is 1.2, largest in the Ω₃ rows at the
horizon (the map's ω₃⁺ there needs second x-derivatives of w⁻ at its singular point). The continuation in
D follows EEG's protocol (steps of 1/2, a linear predictor). `rt_mp.py` evaluates the same residual in arb (160 bits)
and iterates with the f64 Jacobian; that converges while κ(J)·2⁻⁵³ < 1 and removes f64's rounding along the weak
direction (section 4).

**Planted truth first** (`rt_planted.py`, exit 0; logs `I91-mac/planted*.log`). Smooth random fields with the
unknowns' symmetries and μ* = 0.17, the forcing f* = F(z*), Newton from z* + 10⁻³: the residual's third difference is
1.2×10⁻¹⁴ of max|F| (it is exactly quadratic), and z* comes back to 2×10⁻¹⁴, 6×10⁻¹¹ and 8×10⁻¹² at D = 4, 10 and 20,
μ* to 10⁻¹³ or better. The criteria were revised once after the first run, with the reason in the script's docstring:
the start 10⁻² lay outside the basin of the random D = 10 state (its Jacobian has σ_min = 7×10⁻⁶), and a test of the
convergence order from the residual history was dropped after it failed on quadratic histories. Both earlier logs are
in the folder. The controls `--mutate=coef` (the solve's coefficient scaled by 1 + 10⁻³) and `--mutate=cubic` (a cubic
term added) exit 1.

**Δ against ours** (`rt_run.py`, records `I91-mac/accept/accept.jsonl`, `D4/res4.jsonl`, `cont/cont10.jsonl`):

| D | grid N_t × N | L | residual in | Δ_RT − Δ_hp |
|---|---|---|---|---|
| 4 | 32 × 41 | 4,705 | f64 | −6.4×10⁻⁷ |
| 4 | 32 × 61 | 6,945 | f64 | +1.2×10⁻⁹ |
| 4 | 48 × 81 | 13,777 | arb, 160 bits | −1.5×10⁻¹³ |
| 10 | 32 × 61 | 6,945 | f64 | −1.2×10⁻⁶ |
| 10 | 32 × 61 | 6,945 | arb, 160 bits | +1.7×10⁻⁷ |
| 10 | 48 × 81 | 13,777 | f64 | −3.3×10⁻⁴ |
| 10 | 48 × 81 | 13,777 | arb, 160 bits | −1.1×10⁻¹² |
| 4, 10 | 48 × 81, `--mutate=coef` | 13,777 | arb, 160 bits | +1.07×10⁻³, −6.5×10⁻⁶ (exit 1) |

Δ_hp is `hp/data/delta_all.csv` (3.4454524022278213 and 2.5210789195619445, errors 10⁻²² and 10⁻¹⁶). The acceptance
holds: Δ at D = 4 and 10 to 1.5×10⁻¹³ and 1.1×10⁻¹² (tolerance 10⁻⁸), and the mutated coefficient moves Δ by 10⁻³ and
6.5×10⁻⁶ and the run exits 1. At D = 4, μ is 3.8×10⁻¹⁵ from RT's enclosure (the f64 rounding of the stored μ). The
error is set by the ξ resolution (40 × 41 gives the same Δ as 32 × 41 to 13 digits). Each error is the difference from
our value, whose own error is far smaller, so it is the prototype's discretization error at that grid (a measured
difference, not a bound).

## 4. The Jacobian's condition number against D

κ = σ_max/σ_min of the matrix each Newton factors, in the 2-norm, at a solved or mapped state (a measured value;
no error bar beyond the stated resolution checks). Scripts: `solver/rt/rt_cond.py` (f64; σ_max by power iteration,
σ_min by inverse iteration with the LU, planted selftest at κ = 10¹⁰, control `--mutate=transpose`), and
`solver/rt/rt_cond_mp.py` (the Jacobian in arb from exact central differences of the arb residual, σ_min from an arb
inverse and block power iteration at two precisions that must agree, planted selftest at κ = 10⁴⁰, control
`--mutate=lowprec`); ours from `hp/stall_margin.py` (T76's script: the double-double Jacobian, σ_max from its SVD, σ_min
by Rayleigh–Ritz on the weak subspace). Ruiz equilibration (rows and columns scaled to unit max norm) changes RT's κ by
at most a factor 7 (κ_eq in the records), so the growth is not a scaling artifact of the collocation.

**What sets RT's κ.** Three measurements in f64 at solved states (`I91-mac/rt_kappa*.jsonl`).

It does not depend on the t resolution. At D = 4, κ = 9.28×10⁶ with N = 81 for N_t = 16, 32 and 48, and at D = 10 it is
3.2×10¹⁴ with N = 61 for N_t = 32 and 64. The weakest direction is a low harmonic (ω₃ or ω₂ at harmonic 0 of Δ, ω₄ at
harmonic 1), with about 1% of its norm in μ, spread over ξ rather than at one end. Our κ is the opposite, set by the
highest τ harmonic kept (the weak directions are p₁'s top harmonics, LARGE_D_WALL.md).

It grows with the ξ resolution N. σ_max grows like N² (the Chebyshev derivative, fitted slope 2.00), and σ_min falls
like a power of N whose exponent grows with D. The slopes are least-squares fits of ln σ_min against ln N over
N = 41…81 at N_t = 16 (`I91-mac/rt_kappa_N.jsonl`, printed by `rt_summary.py`):

| D | σ_min at N = 41 | σ_min at N = 81 | slope |
|---|---|---|---|
| 4 | 4.4×10⁻⁴ | 1.3×10⁻⁴ | −1.8 |
| 6 | 8.3×10⁻⁷ | 3.0×10⁻⁸ | −4.9 |
| 8 | 5.6×10⁻⁹ | 2.9×10⁻¹¹ | −7.7 |
| 10 | 9.7×10⁻¹¹ | 3.4×10⁻¹³ (f64's floor) | −10.2 over N = 41…61, above the floor |

Below σ_min/σ_max ≈ 2×10⁻¹⁵ the f64 rounding of J dominates (at D = 20 on 24 × 21 the f64 and arb values agree to three
digits at 3.4×10⁻¹⁴). The slope steepens by about 1.4 per unit of D. That is what a near-null vector made of the
singular branch gives. The part of (1 + ξ)^s that degree-N polynomials miss falls like N^(−2 Re s − 1), so
σ_min ~ N^(−2 Re s − 1) with Re s growing linearly in D, here about 0.7 per unit of D (numerical evidence from four D).

It grows fast with D at fixed N. σ_min at N = 61 is 2.1×10⁻⁴, 1.2×10⁻⁷, 2.6×10⁻¹⁰ and 1.7×10⁻¹² at D = 4, 6, 8 and 10,
about 1.3 digits per unit of D.

At D = 10 this is already f64's limit: on the 48 × 81 grid (κ = 1.7×10¹⁵), Newton in f64 stalls at a residual of 10⁻¹⁰
with Δ 3.3×10⁻⁴ off, and the same start polished with the arb residual lands 1.1×10⁻¹² from ours (section 3). The f64
error is rounding along the weak direction, which carries Δ.

**At D = 20 to 100, in arb.** No RT solution was continued past D = 10 here (that needs a multiprecision LU on grids
that resolve Δ, question 3). Instead our states at D = 10 (M = 48), 20.04, 30.07, 40.02, 60.00 (double-double, M = 32)
and 100 (the chain's pinned quad-double state) were mapped onto a coarse 24 × 21 grid, where the mapped fields leave an
RT residual of only 10⁻⁷ to 4×10⁻⁷, and the Jacobian there was measured in arb at two precisions
(`I91-mac/kappa_mp.jsonl`):

| D | state | mapped residual | σ_max | σ_min | κ | weakest |
|---|---|---|---|---|---|---|
| 10 | up_D10.0000 (dd, M = 48) | 2.4×10⁻⁶ | 78.7 | 8.2×10⁻⁸ | 9.6×10⁸ | ω₃, harmonic 0 (below it the Nyquist artifact, 1.3×10⁻⁸) |
| 20.04 | up2_D20.0437 (dd, M = 32) | 3.8×10⁻⁷ | 382 | 1.3×10⁻¹¹ | 3.0×10¹³ | ω₃, harmonic 0 |
| 30.07 | up6_D30.0688 (dd, M = 32) | 3.5×10⁻⁷ | 932 | 1.3×10⁻¹¹ | 6.9×10¹³ | ω₃, harmonic 0 |
| 40.02 | up7_D40.0201 (dd, M = 32) | 1.2×10⁻⁷ | 1,722 | 1.2×10⁻¹² | 1.4×10¹⁵ | ω₃, harmonic 0 |
| 60.00 | up13_D60.0020 (dd, M = 32) | 1.0×10⁻⁷ | 4,048 | 2.3×10⁻¹³ | 1.8×10¹⁶ | ω₃, harmonic 0 |
| 100 | T139 anm_pin3_D100.0000 (qd, pinned) | 4.7×10⁻⁷ | 11,670 | 8.2×10⁻¹⁴ | 1.4×10¹⁷ | ω₃, harmonic 0 |

The weakest direction carries 0.07% to 1.4% of its norm in μ. The others found are ω₃ at harmonic 2 and the
collocation's Nyquist mode (harmonic 24, a periodic field at the top t frequency, whose discrete derivative is zero), an
artifact of even N_t that RT's coefficient space does not have. σ_max grows like D², from the coefficients
(D − 3)(D − 2)μ/2 and the like, and σ_min falls by 2.2 orders from D = 20 to 100 at this N (κ by 3.7, about 0.05 digit
per unit of D), far slower than the 1.3 digits per unit of D at N = 61 for D = 4–10. At D = 100 f64 is at its end on
this grid (it says 1.36×10¹⁷).

**A coarse grid does not hold the solution.** The coarse grid's κ is modest, but its discrete solution is not near the
true one. Newton started from our solution sampled on the grid (`solver/rt/rt_solve_mapped.py`, records
`I91-mac/solve_mapped*.jsonl`) runs away at every D tried: on 24 × 21 and 24 × 31 at D = 20 and 30, and on 24 × 21
at D = 40, f64 Newton starts at a residual of 1–4×10⁻⁷, ends after 15 steps at 2×10⁻⁷ to 5×10⁻⁵ without
converging, and leaves Δ off by 0.3 to 2.4. The first full Newton step from our solution tells why
(`solve_mapped2.jsonl`, printed by `rt_summary.py`):

| D | grid | mapped residual | first step max\|dz\| | its change of Δ | residual after it | κ after it |
|---|---|---|---|---|---|---|
| 10 | 24 × 21 | 2.4×10⁻⁶ | 1.5×10⁻² | +8.9×10⁻² | 2.6×10⁻³ | |
| 10 | 32 × 61 | 1.1×10⁻⁷ | 4.2×10⁻⁷ | +2.0×10⁻⁶ | 1.1×10⁻¹² | 3.2×10¹⁴ at the solution |
| 20.04 | 24 × 21 | 3.8×10⁻⁷ | 4.5×10⁻² | +0.93 | 8.9×10⁻³ | |
| 20.04 | 24 × 31 | 3.8×10⁻⁷ | 8.8×10⁻³ | +0.16 | 4.9×10⁻⁴ | |
| 20.04 | 32 × 41 | 1.6×10⁻⁷ | 7.5×10⁻⁶ | +1.0×10⁻⁴ | 5.0×10⁻¹⁰ | 6.4×10¹⁵ |
| 40.02 | 24 × 21 | 1.2×10⁻⁷ | 5.7×10⁻² | −0.62 | 7.8×10⁻⁴ | |
| 40.02 | 32 × 41 | 8.4×10⁻⁸ | 9.4×10⁻⁵ | +5.5×10⁻⁴ | 9.6×10⁻⁸ | |
| 40.02 | 32 × 61 | 8.4×10⁻⁸ | 5.2×10⁻⁵ | +3.8×10⁻⁴ | 5.1×10⁻⁸ | |
| 60.00 | 32 × 41 | 1.1×10⁻⁷ | 7.4×10⁻⁵ | +4.3×10⁻⁴ | 3.9×10⁻⁸ | 7.4×10¹⁶ |
| 60.00 | 32 × 61 | 1.1×10⁻⁷ | 5.4×10⁻⁵ | +5.0×10⁻⁵ | 4.0×10⁻⁸ | 3.7×10¹⁶ |
| 100 | 32 × 41 | 1.3×10⁻⁷ | 2.9×10⁻⁵ | +2.8×10⁻⁴ | 2.1×10⁻⁸ | 1.9×10¹⁷ |
| 100 | 32 × 61 | 2.3×10⁻⁷ | 4.7×10⁻⁵ | −7.5×10⁻⁵ | 1.1×10⁻⁷ | 1.2×10¹⁷ |

The control is D = 10 on 32 × 61, where the continuation of section 3 found a discrete solution: there the first step
is 4×10⁻⁷, about the map's own error, the residual falls from 1.1×10⁻⁷ to 1.1×10⁻¹² (quadratically), and the solve
lands on the continuation's Δ to all digits (+1.7×10⁻⁷ from ours). On a grid too coarse for its D the first step is
10⁻² and moves Δ by 0.1 to 1, the residual after it is larger than before, and Newton never recovers: the discrete
solution is not near ours. Refining the grid brings it back, at a price. At D = 20 the 32 × 41 grid holds it (the step
is 7.5×10⁻⁶, the residual falls to 5×10⁻¹⁰, and Δ moves by 10⁻⁴, about that grid's error), but its κ there,
6.4×10¹⁵, leaves f64 Newton unable to finish. At D = 40, 60 and 100 the 32 × 41 and 32 × 61 grids give first steps of
3×10⁻⁵ to 10⁻⁴ that move Δ by 5×10⁻⁵ to 6×10⁻⁴, and the residual after them changes only from 1–2×10⁻⁷ to 2–11×10⁻⁸.
These grids sit at the edge, with κ past f64's limit (4×10¹⁶ to 2×10¹⁷ after the step), so f64 cannot tell whether
they hold the solution, and Δ on them would be off by about 10⁻⁴ if they do. So the accuracy demands N, and N brings κ.

**At matched resolution.** Matched in τ harmonics (our M = 32 against RT's N_t = 64) the comparison is RT's κ at any
N_t, since RT's does not depend on it. Matched in accuracy of Δ the two meet at D = 10: RT's 48 × 81 grid gives Δ to
1.1×10⁻¹² with κ = 1.7×10¹⁵, and our M = 48 state (Δ to about 10⁻¹⁶) has κ = 1.2×10¹³ (`hp/stall_margin.py`,
`I91-mac/ours_kappa.jsonl`). Ours then grows to 4.6×10¹⁹ at D = 20.04 (M = 32, every harmonic free) and 1.7×10³¹ at
D = 60 (M = 32, harmonics above 13 frozen, folder `T76/kappa_states.jsonl`), and the pinned system the chain solves
has LU pivots spanning 6.9×10¹⁹ at D = 100 and 5×10³⁹ at D = 200 (folders `T139-macm3/chain/` and
`T138-win2/chain_o/`, `anm_run.jsonl`: the ratio of the largest to the smallest LU pivot, an estimate of κ, not an
SVD).

So the two cannot be compared beyond D = 10 at matched accuracy with these data. At D = 10 RT needs κ = 1.7×10¹⁵ on
13,777 unknowns for Δ to 10⁻¹², ours κ = 1.2×10¹³ on 2,113 unknowns for Δ to 10⁻¹⁶. At D = 20–60 the coarse RT grid
has κ = 3×10¹³ to 2×10¹⁶ against ours 4.6×10¹⁹ to 1.7×10³¹, but that grid does not hold the solution, and the grid that
would has a κ larger by the factor N^p of the table above, with p growing with D. Ours is dominated by p₁'s top τ
harmonics, a weakness of the shooting parametrization (the centre data reach the matching point only through
exponentially small couplings) that RT's global unknowns do not show at the resolutions tried; RT's is dominated by
the horizon branch, which our series removes exactly.

## 5. The three questions

### (1) Can our equations be made quadratic with auxiliary unknowns, and what does that do to the conditioning?

Yes. With b = a² in place of a and two algebraic unknowns, G = (n − 1)c(b − 1) and s = p² + m², the system is

    x c_x = G,   x b_x = b s + (n−1)(b − b²),   G = (n−1)(c b − c),   s = p² + m²,
    x² p_x − x c p_x + ωx p_θ − G p + c p − (n/2) c (p + m) = 0,   and the m equation likewise,

every term at most quadratic in (b, c, G, s, p, m, ω), and equivalent to ours wherever a > 0 (proved, by
substitution: x b_x = 2a · x a_x). It changes nothing that sets the conditioning, for two reasons.

- In our discretization the residual is not a polynomial in the unknowns at all: the unknowns are node values and
  centre and horizon data, and the residual is a flow map (series and Taylor steps), whatever the degree of the ODE.
  RT's property, a Jacobian linear in the background, needs a global discretization whose unknowns are the fields at
  every point, as theirs is.
- The condition number belongs to the linearized problem, not to the degree. Eliminating the algebraic auxiliaries
  from the quadratic system's Jacobian (a Schur complement with a well-conditioned block, ∂(G, s)/∂(G, s) = I) gives
  back the original system's Jacobian, so its near-null directions are the same ones (proved for the elimination;
  the measurement of section 4 is the RT system's own).

What the quadratic form does buy: an exact Jacobian from one central difference (we already have exact Jacobians
from forward-mode duals), and the Newton–Kantorovich constants of a computer-assisted proof, where the remainder
B(x_corr, x_corr) has a fixed bound (RT Sec. 1.2). That is RT's use of it, and it is the relevant one for PROOF.md's
item 7 (a certified solution at one large D).

### (2) Does a global Chebyshev discretization through the horizon avoid our horizon-series wall?

No. It moves the wall from the series' radius into the conditioning of the polynomial system.

RT's Chebyshev domain ends at the horizon: in the parity-extended form the horizon is the endpoint ξ = −1 of the +
equations (and ξ = 1 of the − equations); the discretization is not "through" it, although RT prove analyticity
beyond it (ξ < 41/40). Regularity there is not imposed by any row. It is left to the basis: a polynomial cannot
contain the singular branch exactly. The singular branch at the horizon has an exponent whose real part grows about
linearly in D, 21.1 at D = 40 in our variables (notes/QUAD_DOUBLE.md 13.7; EEG: it "becomes increasingly difficult to
distinguish numerically from the regular one"; EEGJ's "less and less singular"; the toy of
notes/LARGE_D_LITERATURE.md 2(e), whose integral form needs (D − 4)/2 subtracted Taylor terms). A function (1 + ξ)^s
with Re s ≫ 1 is C^⌊Re s⌋ and its Chebyshev coefficients fall like n^(−2 Re s − 1)
(Boyd, Math. Comp. 37 (1981) 155, cited in LARGE_D_LITERATURE.md 2(b)), so the degree-N polynomials contain it to that
accuracy, and the discrete operator has a near-null vector made of it. This is the mechanism that failed our own
collocation of the horizon block (notes/QUAD_DOUBLE.md 13.3, 13.7: 14 near-null directions at D = 40 with 16 to 24
points, Δ undetermined at 10⁻⁹ to 10⁻⁶), now in RT's global discretization, and section 4 measures it.

The measurements of section 4 bear this out. RT's σ_min falls like N^(−p) with p growing by about 1.4 per unit of D,
the weakest direction carries μ and so Δ, and at D = 10 its κ already exceeds what f64 can resolve on a grid that
gives Δ to 10⁻¹². Where our formulation pays digits for the horizon data's accuracy (the series' radius law,
LARGE_D_WALL.md), RT's pays them for the smallest singular value of the polynomial system, and in both the digits grow
about linearly with D. EEG's own words fit this: their earlier shooting code stopped at D = 5.5 because the branches
became hard to tell apart, and their spectral code needs "increasingly higher precision" as D grows.

At D = 20–60 the same mechanism shows from the other side. A coarse grid (24 × 21) has a modest κ there, but it does
not hold the solution: Newton from our solution sampled on it runs away (section 4), because the grid's truncation
residual is far above its σ_min. A grid that holds the solution needs its truncation residual below σ_min, and σ_min
falls with N faster the larger D is. EEG's fixed 40 × 80 grid fits this picture: their six decimals agree with ours
to within their rounding up to D ≈ 40 and depart by 5×10⁻⁷ at D = 60–100 and by 3×10⁻⁶ to 4×10⁻⁶ at D = 150–200
(`hp/data/delta_published_compare.csv`), against their statement that for D > 28 "we trust all displayed digits".
That is consistent with a truncation error growing with D at fixed N; it is not proved by it, since their code was
not run here.

### (3) What would a Fourier × Chebyshev Newton cost at the digits our chain carries?

The unknowns are L = 3.5 N_t N (EEG: 11,200 at N_t = 40, N = 80). To carry our M = 32 odd harmonics of Δ, ω₄ needs
N_t = 64, and with N = 81 that is L = 18,369, against our L = 769 in the chain. In coefficient space (RT, EEG) the
products of fields are convolutions and the Jacobian is dense; in our collocation the products are diagonal and the
derivatives Kronecker products, but an LU of such a two-dimensional spectral matrix fills in, and the prototype factors
it densely. RT's proof inverts a 654,375-dimensional matrix approximately, by Gaussian elimination on a dense block
and a Neumann series that uses its near-banded form (Sec. 1.3), in "a few thousand CPU hours" at D = 4. A dense LU
costs about L³/3 multiply-adds per Newton step.

On this machine a dense arb solve (`solver/rt/lu_cost.py`, python-flint's approximate LU, one thread, L = 300, log
`I91-mac/lu_cost.log`; control `--mutate=perturb` exits 1) costs per L³:

| bits | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|
| seconds / L³ | 3.8×10⁻⁹ | 6.4×10⁻⁹ | 1.1×10⁻⁸ | 3.0×10⁻⁸ | 8.2×10⁻⁸ |

So one Newton step's LU at EEG's L = 11,200 takes about 4.2 hours at 512 bits and 11.5 hours at 1024 bits on one core,
and at L = 18,369 (our harmonic content, N = 81) 19 and 51 hours (`rt_summary.py`, estimates from the L³ law; flint's
blocking may do somewhat better at large L). EEG's 392 steps of 1/2 from 4 to 200, with "typically" four Newton
iterations each, need that LU about 1,600 times. Our chain's step at D = 200 took 460 s on 18 threads, mostly the 24
Taylor coefficients in octo-double (T138-win2 `chain_o/anm_run.jsonl`: t_coef 415 s, the corrector 29 s, the re-seed 14
s). The precision the RT system needs is κ at a grid that holds the solution (section 4). That was measured only to D =
10, where Δ to 10⁻¹² takes the 48 × 81 grid with κ = 1.7×10¹⁵, already past f64, on 6.5 times our unknowns at that D
(13,777 against 2,113). Beyond D = 10 the grid must grow and σ_min falls with it at a rate that grows with D, so both L
and the precision grow. A number for D = 200 would need these laws carried far past the measured range, and it is not
given here. What the data do say is that RT's form buys no digits: at the one D where both were run to the same
accuracy, it needs more precision on more unknowns than ours.

## 5b. What to adopt, and what not

Adopt:

- **The map and its Δ–μ identity** (`gg_to_rt.py`, `mu_check.py`). They tie our horizon data to RT's geometric
  invariant. At D = 4 RT's enclosure of 2K (radius 2×10⁻⁸⁰) is a reference for our Δ in place of MGG2003's ±3×10⁻⁹:
  our M = 192 point agrees with it to 3.1×10⁻²⁶ (`mu_check.py`), against 2.3×10⁻¹⁰ with MGG2003. Proposed: add it to
  `hp/data/delta_published_compare.csv` as the D = 4 reference.
- **The RT prototype as an independent check of Δ(D) at moderate D.** It shares nothing with `hp/` but the first guess
  at D = 4 (another gauge, other unknowns, a global discretization, no series), and it reproduces Δ(10) to 1.1×10⁻¹² of
  ours. With a double-double or quad-double LU, continued in D on grids fine enough to hold the solution (a coarse
  grid does not, section 4), it can check our values at D = 12–30, where the published ones (EEG, six decimals) are
  far less precise than ours.
- **RT's framework for a certified solution at one D** (PROOF.md item 7). The quadratic structure is what makes the
  Newton–Kantorovich bounds of a computer-assisted proof simple, and RT's code did it at D = 4. Section 4 says what it
  would cost at larger D: the near-null direction makes the inverse's norm grow with D, so the proof's constants (and
  the precision of its arithmetic) grow as κ does.

Do not adopt: the global Fourier × Chebyshev Newton as the chain's solver. The horizon wall comes back as the
conditioning of the polynomial system (question 2, section 4). A grid that holds the solution needs its truncation
residual below σ_min, and σ_min falls with N faster the larger D is, so both the resolution and the arithmetic must
grow with D, on 14 to 24 times our unknowns in a dense LU (question 3). At D = 10, the one D where both were run to
the same accuracy, RT needs more precision (κ = 1.7×10¹⁵ against 1.2×10¹³) on 6.5 times the unknowns. Its error at a
fixed resolution grows with D (the prototype's on 32 × 61 from 1.2×10⁻⁹ at D = 4 to 1.7×10⁻⁷ at D = 10, section 3),
and EEG's six decimals leave ours by 3×10⁻⁶ to 4×10⁻⁶ at D = 150–200, beyond their last digit, which is consistent with
that and does not prove it.

## 6. Proposed entries for notes/LIT_REVIEW.md

To be added under the 2026-10-21 addendum's EEG entry and as a new entry (the coordinator edits LIT_REVIEW.md):

- **Reiterer & Trubowitz**, "Choptuik's critical spacetime exists", Commun. Math. Phys. 368, 143–186 (2019),
  doi:10.1007/s00220-019-03413-8, arXiv:1203.3766v1 (16 Mar 2012). Verified: the journal reference against the Springer
  page (REF_CHECK.md, the PROOF.md list); the arXiv v1 PDF read for this entry on 2026-11-08 (Secs. 1–3 and 5, text
  extracted from the PDF). What it contains: a computer-assisted proof, at D = 4, of a real-analytic discretely
  self-similar solution on a neighbourhood of the past light cone, with μ and K enclosed to 10⁻⁸⁰ (Eqs. (2), (3));
  the gauge u^σ = −(1 + σξ)e^{−μτ} and the argument that it is unique up to a constant (Sec. 1.4); the frame variables
  ω₁…ω₄ of Eq. (8), "first order and quadratically nonlinear" (Sec. 1.1), the ten equations Ω (Eq. (11)), the periodic
  problem on the cylinder with parity (Sec. 2.2) and the constraint (♯) system (Sec. 2.3); the Fourier–Chebyshev
  operator calculus (Sec. 3); a weighted ℓ¹ framework with κ₁ = 65/64, κ₂ = 5/4, a reference with m < 250, n < 750
  modes in 650-bit dyadic numbers, and an approximate inverse of a 654,375-dimensional matrix by Gaussian elimination
  on a dense block plus a Neumann series (Secs. 1.3, 5). What it does not contain: other dimensions, perturbations
  (no γ), or any statement about how the method's cost grows with D.
- **Ecker, Ecker & Grumiller**, arXiv:2611.07007v1 (4 Nov 2026), the existing entry, extended: Sec. II uses RT's gauge
  and variables in D dimensions (Eqs. (2)–(5)); "the equations are at most quadratic in the unknowns", "so the
  linearized operator depends linearly on the background solution", which "adds to the numerical stability of their
  scheme" (supplement VII.1); Δ = 4πμ/(1 − ⟨ψ₋₁²⟩) (Eq. (3)); Sec. III: Fourier × Chebyshev on [0, 4π] × [−1, 1] with 40
  and 80 modes, bilinear convolutions in spectral space, 11,200 unknowns, Newton in GMP dyadic arithmetic with rounding
  only at the Jacobian's inversion and "increasingly higher precision" with D, residuals near 10⁻²⁵ after about four
  steps, D from 4 to 200 in steps of 1/2 with a linear predictor; that "the singular branch becomes increasingly
  difficult to distinguish numerically from the regular one", which stopped their earlier shooting code at D = 5.5; "the
  results for D>28 have sufficient accuracy so that we trust all displayed digits", from varying the mode numbers;
  supplement VII.2 maps RT gauge to the gauge of their earlier work by integrating characteristics. Not stated: the
  precision used at large D, mode numbers other than 40 and 80, or an error bar on Δ. Their Eq. (5) holds for Ric = (D −
  2) dφ dφ, their Eq. (1) has Ric = 2 dφ dφ (section 2 above).

## 7. Proposed text for the README ("The large-D wall")

A sentence for the bullet "The large-D wall" of "Limits and open problems", after its first sentence:

> Ecker, Ecker and Grumiller reach D = 200 with Reiterer and Trubowitz's formulation, quadratic first-order equations
> discretized globally by Fourier and Chebyshev series with no horizon series. It meets the same wall in another form.
> The horizon's singular branch, whose exponent grows linearly in D, is nearly a polynomial, so the Chebyshev system's
> smallest singular value falls like a power of the resolution whose exponent grows by about 1.4 per unit of D
> (1.3 digits per unit of D at 61 Chebyshev points for D = 4–10), and the arithmetic must lengthen with D as ours does
> ([notes/RT_FORMULATION.md](notes/RT_FORMULATION.md)).

## 8. Reproduction

From the repository root, with the raw-data repository cloned beside it
(`git clone https://github.com/nmoinvaz/choptuik-large-d-raw ../choptuik-large-d-raw`), python-flint, mpmath, sympy,
numpy and scipy; `gg_to_rt.py profile` and `hp/stall_margin.py` also need hpkernel. RAW stands for
`../choptuik-large-d-raw`. Each script exits non-zero when its check fails.

```
python3 solver/rt/rt_eqs_check.py                    # sympy; --mutate=coef and --mutate=warp must exit 1
python3 solver/rt/mu_check.py RAW/T45-win/hp/data/lowD_states/d4hm15_D4.0000_M192.npz     # --mutate=n must exit 1
python3 solver/rt/gg_to_rt.py profile RAW/T106-mac/m48_states/D4.0000.npz PROF4.npz 48
python3 solver/rt/rt_planted.py                      # --mutate=coef and --mutate=cubic must exit 1
python3 solver/rt/rt_run.py --profile PROF4.npz --grids 32x61,48x81 --cont_grid 32x61 --D_to 10 --mp 160 \
    --out OUTDIR --tag accept                        # --mutate=coef must exit 1
python3 solver/rt/rt_cond.py --selftest              # --mutate=transpose must exit 1
python3 solver/rt/rt_cond_mp.py --selftest           # --mutate=lowprec must exit 1
python3 solver/rt/rt_cond.py OUT.jsonl OUTDIR/accept_D10.0000_32x61.npz [--grid=64x61]
python3 solver/rt/rt_cond.py OUT.jsonl OUTDIR/accept_D10.0000_48x81.npz --grid=16x41       # and 16x51 ... 16x81
python3 solver/rt/rt_cond_mp.py OUT.jsonl --profile PROF.npz --grid 24x21 --prec 512
python3 solver/rt/rt_solve_mapped.py OUT.jsonl --profile PROF.npz --grids 24x21,32x41 --mp 192
python3 solver/rt/lu_cost.py OUT.jsonl                # --mutate=perturb must exit 1
python3 solver/rt/rt_summary.py RAW/I91-mac           # every table of this note from the records
cd hp && python3 stall_margin.py OUT.jsonl STATE.npz --k 2 --threads 2      # OUT inside a raw-data folder
```

PROF.npz files are `gg_to_rt.py profile` dumps of hp states (`I91-mac/profiles/`: up_D10.0000, up2_D20.0437,
up6_D30.0688, up7_D40.0201 and up13_D60.0020 from `hp/data/states` of the main checkout, T139-macm3's
anm_pin3_D100.0000).

## 9. Past D = 10: the multiprecision Jacobian solve

Issue #105. Subagent of the coordinator on the Mac, 2026-11-08. Code in `solver/rt/`: `rt_mpsolve.py` and
`rt_mp_cost.py` (new), `rt_run.py --solver mp`, and `rt_grids.py` (new). Raw output is in folder `I105-mac/` of the
raw-data repository.

**Result.** On 32 × 101 the prototype reproduces our Δ at D = 15, 20.043653 and 29.984746 to 1.1×10⁻¹⁶, 1.7×10⁻¹⁶
and 3.2×10⁻¹⁶. On 32 × 121 it does so to 2.3×10⁻²², 1.7×10⁻²⁰ and 5.3×10⁻¹⁸. On 32 × 81 it misses the acceptance at
D = 20.04 and 29.98, at 1.4×10⁻¹⁰ and 1.2×10⁻⁹ against the threshold of 10⁻¹⁰. So do the two grids against each
other. The cause is the truncation of 81 Chebyshev points, which grows with D, not the solve. From D = 15 to 30 a
Newton step costs 13 to 35 s at L = 9,185 and 11,425, the arb parts on one core. A dense LU in extended precision at
those sizes would take an estimated 4 to 7 min in double-double, or 50 to 98 min in arb, per step.

### 9.1 The method

Section 4 left the prototype at D = 10. There κ(J) on a grid that resolves Δ passes 2⁵³, and refinement with the f64
LU alone (`rt_mp.newton_mixed`) stops converging. `rt_mpsolve.newton_mp` replaces the linear solve and keeps the
residual.

- Newton's residual is the arb residual of `rt_mp.py`, at 192 bits.
- Each Newton step is solved in arb by flexible GMRES (Saad 1993), without restart, to a relative residual of 10⁻³⁰.
  The products J(z)u are exact central differences of the arb residual, which is quadratic.
- The right preconditioner is the f64 LU of the f64 Jacobian (`rt_model.RT.jacobian`). Its triangular solves run in
  double-double, each row a `dd_dot` of multidouble's Python package. This is GMRES-based iterative refinement (Carson
  and Higham 2017) with the factors applied in higher precision.
- The f64 Jacobian and its LU are rebuilt after a Newton step larger than 10⁻³. The two continuations of 9.4 also
  rebuilt after an inner solve of more than 25 products (commit a9cb86b). Later runs rebuild only when the count
  exceeds 1.5 times that of the first solve with the current factors (1090efb).
- Newton stops at max|F| < 10⁻⁴⁰.

Why it works. The exact inverse of the f64 factors is (J + E)⁻¹ with |E| near 2⁻⁵³|J|. So J(LU)⁻¹ is the identity, plus
a term of low rank from the weak directions, plus a term small in norm, and GMRES spends about one product on each weak
direction. The preconditioner must be one fixed operator applied in extended precision. |(LU)⁻¹| is near 1/(2⁻⁵³|J|),
so f64 triangular solves put errors the size of the solution into the well-conditioned directions.

The inner solve runs to 10⁻³⁰, not to the loose tolerance of an inexact Newton method. An inner residual r leaves an
error of about |r|/σ_min along the weak directions, and the quadratic term of that error would exceed the next residual.

### 9.2 Planted truth first

`rt_mpsolve.py --selftest` (logs `I105-mac/planted*.log`, records `planted.jsonl`) solves A x = b with A = U diag(s) Vᵀ
in arb. U and V are products of eight Householder reflections, so they are exactly orthogonal. L = 1500, and s is
log-uniform in [10⁻⁴, 1] except for twelve values log-spaced from 10⁻⁹ to 10⁻²⁰, so κ(A) = 10²⁰ exactly. The test runs
`newton_mp` with the same central-difference products as for the RT residual. The criteria were written before the
first run.

- The error is below 10⁻³⁰. The expected error is about κ·2⁻¹⁹², or 2×10⁻³⁸.
- It takes at most 8 Newton steps.
- The f64 LU alone leaves an error above 10⁻⁶, so the system is beyond f64. It leaves 66.

| inner solver | preconditioner | error | products per Newton step | verdict |
|---|---|---|---|---|
| flexible GMRES | f64 LU, double-double solves (the design) | 9.3×10⁻⁴⁰ | 11 | pass |
| GCR | f64 LU, double-double solves | 6.1×10⁻⁴¹ | 11 | pass |
| flexible GMRES | f64 LU, f64 solves | 1.4×10⁻³⁹ | 45–59 | pass |
| flexible GMRES or GCR | explicit f64 inverse, applied in double-double | 9.3×10⁻³⁷ | 80 (the cap), 8 steps | pass |
| GCR | f64 LU, f64 solves (the first design, `planted_first_run.log`) | 47 | 80 | fail |
| the design, `--mutate=f64apply` (products with A rounded to f64) | | 64 | 23–38 | fail, exit 1 |
| the design, `--mutate=lowprec` (64 bits) | | 5×10¹⁵ | 80 | fail, exit 1 |

The explicit inverse does worse than f64 solves, because LAPACK builds each of its columns by such a solve. The
factorization of Jᵀ in place, which the controls of 9.4 use (commit 1d577e7), passes the same test with 6.9×10⁻⁴⁰, and
its two mutations exit 1 (`controls/planted_inplace*.log`).

### 9.3 The pilot

The pilot ran on 32 × 101 (L = 11,425), in folder `I105-mac/pilot/`.

- At D = 10, the 48 × 81 state of section 3 was resampled and re-solved in 3 Newton steps. It lands 3.0×10⁻¹⁷ from
  ours.
- From that solution, one Newton solve at D = 12 started from a zeroth-order guess with residual 6.2×10⁻². It converged
  in 8 steps to max|F| = 1.6×10⁻⁴², with Δ 5.8×10⁻¹⁷ from ours. It took 150 s: the Jacobian 9 s, its LU 7 s, the
  Newton steps 49 s, and six rebuilds of the preconditioner 86 s. The pilot rebuilt after every step larger than 10⁻⁶
  (commit 54220e6).
- Without the rebuilds (`pilot12_32x101_norefresh.log`), a preconditioner from a point 8×10⁻² away needed 62 to 76
  products per step instead of 10, and the run was stopped.
- The guess lies outside Newton's quadratic region, so the residual falls only by a factor of about 3 per step down to
  10⁻⁵.
- The continuation in steps of 1/2 (`cont12_32x101.log`) lands 3.6×10⁻¹⁷ to 5.8×10⁻¹⁷ from ours at D = 10.5, 11, 11.5
  and 12. At D = 12 it gives the same Δ as the direct solve to 20 digits.

### 9.4 The continuation, D = 10 to 30

`rt_run.py --solver mp` ran one process per grid (`I105-mac/cont/`), starting from the 48 × 81 state at D = 10 of
section 3 (`I91-mac/accept/`). Each grid is re-solved at D = 10 and continued in steps of 1/2 to D = 30 with EEG's
linear predictor. Two extra steps land on the hp rows nearest 20 and 30, D = 20.043653 and 29.984746, from runs up2 and
up6. hp's continuation rounds D to six decimals, so these are the D it solved. D = 15 is an hp row too. Δ_hp is the
row's Delta_q0 + Delta_q1 (`hp/data/delta_all.csv`, double-double, err 10⁻¹⁶). Every difference is taken in arb from
Δ_RT's 30-digit string. Every step converged, to a final max|F| of at most 3.5×10⁻⁴⁷ on 32 × 81 and 3.5×10⁻⁴² on
32 × 101.

The table gives Δ_RT − Δ_hp at the check D, from `rt_grids.py` and the logs named. The 48 × 101 and 32 × 121 entries
re-solve the 32 × 101 state on those grids (`controls/res_*`). They are controls of the t and ξ resolution, outside the
acceptance.

| D | 32 × 81 | 32 × 101 | 32 × 81 − 32 × 101 | 48 × 101 | 32 × 121 |
|---|---|---|---|---|---|
| 15 | +1.13×10⁻¹¹ | −1.13×10⁻¹⁶ | +1.13×10⁻¹¹ | −1.13×10⁻¹⁶ | −2.3×10⁻²² |
| 20.043653 | +1.40×10⁻¹⁰ (fail) | −1.68×10⁻¹⁶ | +1.40×10⁻¹⁰ (fail) | −1.68×10⁻¹⁶ | +1.7×10⁻²⁰ |
| 29.984746 | +1.24×10⁻⁹ (fail) | +3.20×10⁻¹⁶ | +1.24×10⁻⁹ (fail) | +3.21×10⁻¹⁶ | +5.3×10⁻¹⁸ |

The verdicts on the acceptance of #105:

- |Δ_RT − Δ_hp| < 10⁻¹⁰ on both grids. It is met on 32 × 101 at all three D. On 32 × 81 it is met at D = 15 only.
  The 32 × 81 run exits 1 and the 32 × 101 run exits 0.
- The two grids agree to 10⁻¹⁰ at D = 15 only, and `rt_grids.py` exits 1 (`accept_grids.log`). Its own control runs on
  the 32 × 121 and 32 × 101 pair, which passes. That pair exits 0, and with `--mutate=shift` (2×10⁻¹⁰ added to one Δ)
  it exits 1 (`controls/grids_121_101*.log`).
- `--mutate=coef` from the unmutated 32 × 81 states moves Δ by 3.75×10⁻⁴ at D = 15 and 3.06×10⁻³ at D = 29.98, and
  exits 1 both times (`controls/coef_*`).

What the columns say. 48 × 101 differs from 32 × 101 by 1.3×10⁻²², 2.0×10⁻²² and 1.0×10⁻¹⁹, so N_t = 32 is not what
limits Δ. 32 × 121 moves Δ by 1.1 to 3.2×10⁻¹⁶, so the 10⁻¹⁶ offset of 32 × 101 is its own ξ truncation, not hp's
error. Our double-double values agree with the 32 × 121 RT values to 2×10⁻²², 2×10⁻²⁰ and 5×10⁻¹⁸, well inside the
10⁻¹⁶ their rows state. The truncation of 32 × 81 along the path is its difference from 32 × 101, whose own offset
stays below 4×10⁻¹⁶.

| D | 10 | 12 | 14 | 16 | 18 | 20 | 22 | 24 | 26 | 28 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 × 81 − 32 × 101 | −1.1×10⁻¹² | −7.7×10⁻¹³ | +4.5×10⁻¹² | +2.2×10⁻¹¹ | +6.3×10⁻¹¹ | +1.4×10⁻¹⁰ | +2.6×10⁻¹⁰ | +4.3×10⁻¹⁰ | +6.5×10⁻¹⁰ | +9.3×10⁻¹⁰ | +1.24×10⁻⁹ |
| products per Newton step, 32 × 101 | 7–8 | 10–15 | 15–25 | 20–29 | 25–36 | 30–31 | 35 | 39–40 | 43–44 | 47 | 47–51 |

N = 81's error changes sign between D = 12.5 and 13 and then grows about ninefold from D = 20 to 30. The number of
products per Newton step counts roughly the directions the f64 LU gets wrong. It grows about linearly in D, by about 2
per unit of D on 32 × 101.

Independence. The RT code shares three things with hp. The first is the D = 10 start state, which section 3 continued
from our D = 4 double-double state at M = 48 mapped by `gg_to_rt.py`. That was only a first guess, and Newton
converged far from it at D = 4. The second is multidouble's `dd_dot`, here used only inside the preconditioner. It sets
how fast GMRES converges, not the solution, which the arb residual fixes. The third is Python and numpy. The equations
(EEG's Eq. (5), checked in sympy in section 2), the gauge, the unknowns, the discretization and the residual's
arithmetic are separate. The agreement is numerical evidence: a measured difference between two discretizations, not a
bound.

### 9.5 Costs

`rt_mp_cost.py` ran at D = 15 (`I105-mac/controls/mp_cost3_*.log`, `mp_cost3.jsonl`). The arb parts use one thread and
BLAS two, on the Mac shared with one other job. The dense rows are the alternative, a full LU in extended precision at
every Newton step, estimated from the L³ law of the largest size timed.

| one evaluation | 32 × 81 (L = 9,185) | 32 × 101 (L = 11,425) |
|---|---|---|
| arb residual, 192 bits | 0.15 s | 0.22 s |
| one product J u (two arb residuals) | 0.35 s | 0.48 s |
| double-double triangular solves (one preconditioner application) | 0.073 s | 0.103 s |
| f64 Jacobian | 6.8 s | 12.2 s |
| f64 LU | 2.5 s | 4.3 s |
| dense double-double LU (multidouble `dd_solve`, about 2.8×10⁻¹⁰ s/L³) | 223 s (estimate) | 420 s (estimate) |
| dense arb LU at 192 bits (`arb_mat.solve`, about 3.9×10⁻⁹ s/L³) | 50 min (estimate) | 98 min (estimate) |

The products dominate a Newton step. From D = 15 to 30, a step took 13 to 18 s on 32 × 81 and 19 to 35 s on 32 × 101,
and a step of 1/2 in D took 1.4 to 2.3 min and 2.3 to 4.3 min (`rt_grids.py --table`, `accept_grids.log`). The whole
continuation took 70 min on 32 × 81 and 117 min on 32 × 101, of which preconditioner rebuilds were 14 and 30 min.

The control is the backward error of the double-double solves against the factors they apply,
|UᵀLᵀPᵀx − b|/(|Uᵀ||Lᵀ||x| + |b|). It is 3.5×10⁻³⁶ and 2.9×10⁻³⁷, against a criterion of 10⁻²⁶. `--mutate=perturb`
changes one pivot by 10⁻¹⁰ relative after the solve, and one entry of the dd_solve matrix by 10⁻¹⁰. It gives 1.2×10⁻¹⁷
and 6.2×10⁻¹² and exits 1 (`mp_cost2_mutate.log`, `mp_cost3_mutate.log`). The first version of the check compared with
the f64 Jacobian itself and failed with 2.5×10⁻¹⁴ (`controls/mp_cost_32x81.log`). That residual is the factorization's
E x, not the solve's error, so the check was wrong and the solve was not.

### 9.6 Discrepancies with the text above (reported, not fixed)

- **The cost argument of 5(3) and 5b.** Both price the RT Newton by a dense LU in extended precision, hours per step
  at EEG's L. The verdict rests partly on that, "on 14 to 24 times our unknowns in a dense LU". The solve here factors
  only in f64 and needs 13 to 35 s per Newton step at L near 10⁴ up to D = 30. What still grows with D is the number
  of GMRES products, which tracks the weak directions, and the Chebyshev resolution that holds Δ: N = 81 stops holding
  10⁻¹⁰ near D = 20. The coordinator decides whether the verdict changes.
- **The LU pivot ratio is not κ.** The ratio was 5.5×10¹² to 1.4×10¹⁷ along the runs. The f64 LU cannot resolve
  pivots below its rounding, so the ratio saturates near 10¹⁶ to 10¹⁷. κ on the resolving grids past D = 10 was not
  measured.
- **hp's stated error is conservative here.** The rows give 10⁻¹⁶ (`check`), and the 32 × 121 values agree with them
  to 2×10⁻²², 2×10⁻²⁰ and 5×10⁻¹⁸ at the three D. At D = 29.98 the 5.3×10⁻¹⁸ is the size of hp's own estimate of its
  Δ error from the Newton residual there (dDelta_est 5.4×10⁻¹⁸ in `hp/data/delta_dd.csv`).

### 9.7 What this means for #106

#106 asks for Δ at EEG's resolution (40 Fourier and 79 Chebyshev points, L = 11,201) and at N = 121 (L = 17,081 at
N_t = 40), at D = 20, 40 and 60.

- The solver carries over unchanged, for example `rt_run.py --solver mp --grids 40x79`. The in-place LU holds one copy
  of the f64 Jacobian, 1.0 GB at L = 11,201 and 2.3 GB at L = 17,081, and both fit this Mac.
- At about 2 more products per unit of D, the 32 × 101 count of 50 at D = 30 reaches the cap of 80 near D = 45. #106
  should raise `--max_products`, since FGMRES without restart keeps every vector.
- A step of 1/2 in D took 2 to 4 min at L = 11,425 from D = 20 to 30. From D = 30 to 60 that is 60 steps per grid,
  and each costs more as the products grow. A double-double product in place of the two arb residuals would cut the
  dominant cost (Recommended tasks of #105).
- The truncation of about 80 Chebyshev points is measured here in collocation: 1.1×10⁻¹² at D = 10, 1.4×10⁻¹⁰ at 20,
  1.2×10⁻⁹ at 30. EEG's 40 × 80 truncates in coefficient space, so their error need not be the same, but the scale is
  set. Growing ninefold per 10 units of D, which is an extrapolation and not a measurement, it would reach EEG's
  departure from ours (5×10⁻⁷ at D = 60–100, `hp/data/delta_published_compare.csv`) between D = 50 and 60. #106
  measures this at their grid.

### 9.8 Proposed text for the README

For the coordinator, after the RT sentence of "The large-D wall" (section 7):

> With a multiprecision Jacobian solve (f64 LU, flexible GMRES in arb), the RT prototype reproduces our Δ from D = 10
> to 30 on a 32 × 101 grid to 3×10⁻¹⁶, and on 32 × 121 to 2×10⁻²², 2×10⁻²⁰ and 5×10⁻¹⁸ at D = 15, 20 and 30. It is an
> independent check by another gauge and discretization ([notes/RT_FORMULATION.md](notes/RT_FORMULATION.md) section 9).

### 9.9 Reproduction

From the repository root, with the raw-data repository cloned beside it (RAW = `../choptuik-large-d-raw`). The
scripts need python-flint and multidouble's Python package with `dd_dot`, which `hp/build.sh` installs. On the Mac use
`/Users/nathan/venv-qd/bin/python` with `MULTIDOUBLE_SITE=/Users/nathan/Source/venv-anm/lib/python3.13/site-packages`.
Each command exits non-zero when its check fails.

```
python3 solver/rt/rt_mpsolve.py --selftest                   # --mutate=f64apply and --mutate=lowprec must exit 1
python3 solver/rt/rt_mpsolve.py --selftest --inner gcr --precond lu_f64        # a variant of 9.2 (exits 1)
python3 solver/rt/rt_run.py --start RAW/I91-mac/accept/accept_D10.0000_48x81.npz --grids 32x101 --solver mp \
    --tol 1e-10 --out OUT --tag d10                           # 9.3: D = 10 on 32x101
python3 solver/rt/rt_run.py --start OUT/d10_D10.0000_32x101.npz --D_start 12 --grids 32x101 --solver mp \
    --tol 1e-10 --out OUT --tag pilot12                       # 9.3: the pilot
python3 solver/rt/rt_run.py --start RAW/I91-mac/accept/accept_D10.0000_48x81.npz --grids 32x101 --solver mp \
    --D_to 30 --D_extra 20.043653,29.984746 --check_D 15,20.043653,29.984746 --tol 1e-10 --out OUT \
    --tag cont_32x101                                         # 9.4; with --grids 32x81 it exits 1
python3 solver/rt/rt_run.py --start RAW/I105-mac/cont/cont_32x81_D15.0000_32x81.npz --grids 32x81 --solver mp \
    --tol 1e-10 --mutate=coef --out OUT --tag coef            # must exit 1, Delta moved by 3.75e-4
python3 solver/rt/rt_run.py --start RAW/I105-mac/cont/cont_32x101_D15.0000_32x101.npz --grids 32x121 --solver mp \
    --tol 1e-10 --out OUT --tag res                           # 9.4 controls, also --grids 48x101
python3 solver/rt/rt_grids.py RAW/I105-mac/cont/cont_32x81.jsonl RAW/I105-mac/cont/cont_32x101.jsonl --table \
    --extra RAW/I105-mac/controls/res_32x121_D15.0000.jsonl  # the tables of 9.4 (exits 1)
python3 solver/rt/rt_grids.py RAW/I105-mac/controls/res_32x121_all.jsonl RAW/I105-mac/cont/cont_32x101.jsonl
                                                              # exits 0; with --mutate=shift it must exit 1
python3 solver/rt/rt_mp_cost.py OUT.jsonl --state RAW/I105-mac/cont/cont_32x81_D15.0000_32x81.npz --grid 32x81
                                                              # 9.5; --mutate=perturb must exit 1
```

## 10. Past D = 30: the Chebyshev N against D, and the cost

Issue #145. Subagent of the coordinator on the Mac, 2026-11-08. Code in `solver/rt/`: new options of `rt_run.py`
(`--start_prev`, `--pred_order`, `--extras_off_path`, `--patient`, `--it_max`, `--full_below`, `--newton_tol`) and of
`rt_mpsolve.newton_mp` (`stall_any`, `full_below`), and the new `rt_cost_fit.py`, `rt_pred_check.py` and
`rt_roots.py`. Raw output is in folder `I145-mac/` of the raw-data repository.

**Result.** The acceptance is met at the rows nearest 40 and 50 and missed at the one nearest 60. On 32 × 101
the prototype reproduces our Δ to 3.5×10⁻¹⁵ at D = 40.02, 3.3×10⁻¹⁵ at 50.01 and 2.1×10⁻¹⁴ at 55.00, and misses by
1.27×10⁻¹² at 60.00. No grid on the ladder does better there. From D ≈ 40 refining the Chebyshev grid makes Δ worse:
32 × 141 is 3.8×10⁻¹² off at D = 45 and 3.0×10⁻⁷ at 55, and 32 × 121 is 2.9×10⁻¹² off at 50 and 3.5×10⁻¹¹ at 60. The
finer grids' roots are not unique along the weak direction of section 4. Meanwhile 32 × 81 was already 1.2×10⁻⁹ off at
D = 30 (#105). So the criterion for N written before the runs fails from D = 50. With this collocation on N_t = 32,
no grid tried holds Δ to 10⁻¹² at D = 60. A grid between 81 and 101 points was not tried.

The cost stays modest. A unit of D took 3 to 11 min on one core at L = 11,425. The products of a Newton step with
fresh factors grew about 0.4 per unit of D from 30 to 60, after 2 per unit from 10 to 30. The Newton steps per D step
grew about 0.12 per unit of D. Staying on 32 × 101 would cost about 13 to 16 hours from D = 30 to 100 and 110 to 170
hours to 215 (an extrapolation, cost only). #91's verdict on the chain should stand on its conditioning ground, not
its cost ground (10.7).

### 10.1 The protocol

The solver is #105's (9.1), at 192 bits with N_t = 32. Five things changed, each after a run showed the need. The
criteria for N were written down before the main runs (`I145-mac/README.md`).

- **Steps of 1 with a cubic predictor.** `--dD 1 --pred_order 3` extrapolates in D through the last four solutions
  by Lagrange's formula, and `--start_prev` seeds them, so a run restarts from its saved states. On #105's states at
  D = 30 on 32 × 101 (`rt_pred_check.py`), the linear predictor at steps of 1/2 starts at max|F| = 1.8×10⁻⁴, the
  quadratic at steps of 1 at 1.8×10⁻⁴ and the cubic at 4.9×10⁻⁵. The main run used the quadratic one to D = 33 and
  the cubic one from there.
- **One rebuild per D step.** In the pilot (D 30 → 30.5), a Newton step took 45 products with factors fresh at the
  predictor and 64 after the first step of 4×10⁻⁵. #105's relative rule (1.5 times the first count) does not fire
  there, so the runs rebuild after any step larger than 10⁻⁵ (`--refresh_dz 1e-5`).
- **Patient Newton.** #105's rule stops Newton when max|F| stays above half its previous value after step 3, meant for
  the precision's floor. Near this nearly singular root it fires far above that floor while max|dz| still falls. It
  ended a 32 × 121 solve at D = 32 at max|F| = 1.5×10⁻²⁶ and the 32 × 101 step to D = 34 at 1.2×10⁻²⁶.
  `--patient --it_max 20` applies the rule only within 10⁶ of the target.
- **Small steps taken in full.** Halving a step whenever max|F| grew stalled Newton too. At D = 35, a full step of
  3.4×10⁻⁹ was halved to 1/128 for 8 steps while max|F| crept from 4.4×10⁻¹⁹ to 2.2×10⁻¹⁸. For the quadratic residual
  a full step dz leaves F = B(dz, dz) exactly, so `--full_below 1e-4` takes every step below 10⁻⁴ in full. Steps then
  raise max|F| once or twice, by up to 10⁹, before the quadratic phase.
- **Newton's target.** Up to D = 53 it is #105's 10⁻⁴⁰, with stalls accepted below 10⁻³⁴. From D = 54 it is 10⁻³⁸
  and from D = 57 10⁻³⁶ (`--newton_tol`), with stalls accepted below 10⁻³² and 10⁻³⁰. The D = 54 step had bounced for
  15 Newton steps between max|F| = 6×10⁻³⁵ and 1.2×10⁻³³, at max|dz| = 1–3×10⁻¹⁸. The D = 57 step bounced between
  1.2×10⁻³² and 1.0×10⁻³¹, at max|dz| up to 1.2×10⁻¹⁶. That is a floor set by the weak direction, not by the precision.
  It rises with D, and a step of 10⁻¹⁶ moves Δ by about 10⁻¹⁵. Both steps were stopped by hand and redone from the
  last saved state.

The stalled D = 34 solve shows why Δ is fixed only in Newton's quadratic phase. At max|F| = 1.2×10⁻²⁶ its Δ was
1.9292728794391906, 2.0×10⁻¹² from the converged 1.9292728794411864.

**The grid above the continuation's is a follower.** A continuation on 32 × 121 diverged at its first step, D 30 →
31 (`cont/cont_32x121.*`). It started at max|F| = 1.5×10⁻⁴, the same as on 32 × 101, but the first Newton step was
6.6×10⁻³, and after 4 steps and 3 rebuilds max|F| was 3.6×10⁻⁵. So the 32 × 121 and 32 × 141 values are re-solves:
the coarser grid's state at the same D, resampled and solved by Newton (`rt_run.py --start STATE --grids 32xN`,
queued by `run_worker*.sh`). They start at max|F| = 1–3×10⁻¹³.

### 10.2 Δ on three grids

At each check D, Δ − Δ_hp on 32 × 101 (the continuation), 32 × 121 and 32 × 141. hp is the double-double row of
`hp/data/delta_all.csv` (runs up6, up7, up8, up10, up13), with err 1×10⁻¹⁶ at 34.99 and 2×10⁻¹⁵ from 40 on. Two
entries in a cell are two roots of that grid from different starts. Every value is converged, to max|F| below 10⁻³².

| D | 32 × 101 | 32 × 121 | 32 × 141 |
|---|---|---|---|
| 34.993266 | +9.9×10⁻¹⁶ | −3.9×10⁻¹⁸ | +3.1×10⁻¹⁸ |
| 40.020065 | +3.5×10⁻¹⁵ (twice) | +1.5×10⁻¹⁵ (twice) | +1.0×10⁻¹⁴, +1.7×10⁻¹⁴ |
| 44.963411 | +2.9×10⁻¹⁵ (twice) | −8.5×10⁻¹⁵, +3.2×10⁻¹⁵ | +3.8×10⁻¹² |
| 50.007491 | +3.3×10⁻¹⁵, +7.5×10⁻¹⁵ | −2.9×10⁻¹² (twice), +4.5×10⁻¹² | −4.1×10⁻¹¹ |
| 54.996085 | −2.1×10⁻¹⁴, +1.5×10⁻¹⁴ | +8.5×10⁻¹², +9.7×10⁻¹³ | −3.0×10⁻⁷ |
| 60.001952 | **−1.27×10⁻¹², −2.33×10⁻¹²** | **+3.5×10⁻¹¹** | not run |

From D = 40 the finer grids are no closer to hp. The best grid falls with D: at least 141 at D = 35, 121 at 40, 101
from 45. A grid past its best has an error growing about 0.5 decade per unit of D: 32 × 141 from 1.0×10⁻¹⁴ at 40.02 to
3.8×10⁻¹² at 44.96, and 32 × 121 from 8.5×10⁻¹⁵ at 44.96 to 2.9×10⁻¹² at 50.01. On 48 × 141 at 40.02, Δ moves by
1.3×10⁻²⁰ from 32 × 141, so N_t = 32 is not the cause there.

**The criterion for N fails from D = 50, and it is reported, not changed.** It defined N(D) as the smallest N whose Δ
agrees with N + 20 to 10⁻¹², and moved the continuation up a grid at 10⁻¹³. `rt_grids.py --tol 1e-12` on the 101 and
121 records exits 0 at D 34.99 and 40.02 (exit 1 with `--mutate=shift`). It exits 1 at 50.01, where |Δ_101 − Δ_121| =
2.9×10⁻¹². Read literally, 101 no longer resolves Δ and the run should move to 121. But 121 is the grid that misses
hp. The criterion's premise, that the finer grid is the better reference, fails from D ≈ 45. The continuation stayed
on 32 × 101, and its verdicts are taken against hp.

### 10.3 Why: the roots along the weak direction

Each grid was re-solved at the check D from a second start, another grid's root resampled (`run_roots.sh`,
`run_worker4.sh`), and `rt_roots.py` compares the two roots (`controls/roots_*.log`). The spread is the difference of
the two converged Δ. It is a lower bound on how far the grid leaves Δ undetermined.

| D | 32 × 101 | 32 × 121 | 32 × 141 |
|---|---|---|---|
| 40.020065 | 5.5×10⁻²³ | 4.6×10⁻²⁰ | 6.4×10⁻¹⁵ |
| 44.963411 | 1.9×10⁻²⁰ | 1.2×10⁻¹⁴ | |
| 50.007491 | 4.2×10⁻¹⁵ | 5.4×10⁻¹⁵ (starts 4.2×10⁻¹⁵ apart), 7.4×10⁻¹² (the 141 start) | |
| 54.996085 | 3.6×10⁻¹⁴ | 7.6×10⁻¹² (starts 3.6×10⁻¹⁴ apart) | |
| 60.001952 | 1.1×10⁻¹² | | |

The second 101 start is the 121 root resampled. The 121 starts are the 101 roots and, where marked, the 141 root.

The two 32 × 141 roots at 40.02 reach max|F| = 5.4×10⁻⁵² and 2.5×10⁻⁴⁷ in the runs, and their states differ by max|z₂
− z₁| = 4.3×10⁻¹⁶. The difference is mostly μ and harmonic 0 of every field, spread evenly over ξ: section 4's weakest
direction. For the quadratic residual F(z₂) − F(z₁) = J(z_m)(z₂ − z₁) exactly, with z_m the midpoint. So the runs'
residuals bound |J(z_m)e|/|e| below 6×10⁻³² in the max norm along that difference. (The script evaluates the residuals
of the states stored in double-double, about 10⁻³⁰, and so prints the weaker bound 5.7×10⁻¹⁵.) At 50.01 the two
32 × 121 roots from the two 32 × 101 roots lie 2.9×10⁻¹² below hp, and the one from 32 × 141's root lies 4.5×10⁻¹² above
it. Their states differ by 6.5×10⁻¹³. Which root Newton finds depends on where it starts.

This is section 4's mechanism, now seen as error on solved grids. σ_min(J) falls like a power of N whose exponent
grows with D, because the horizon's singular branch is nearly a polynomial. The part of a grid's truncation residual
along the weak direction is divided by that σ_min. Past some N it falls more slowly than σ_min does, so refining N
makes Δ worse, and nearly equal roots appear along that direction. The 101 grid reaches the same regime: its root
spread grows from 5.5×10⁻²³ at D = 40 to 4.2×10⁻¹⁵ at 50, 3.6×10⁻¹⁴ at 55 and 1.1×10⁻¹² at 60, about 0.5 decade per
unit of D (`fit/cost_fit.log`; a fit through five points). Below 101 the truncation is too large, since #105's 32 × 81
missed by 1.2×10⁻⁹ at D = 30.

### 10.4 Cost against D

All times are on the shared Mac (Apple M5 with 4 performance and 6 efficiency cores, at nice 15, with up to three
other agents' jobs). They are single measurements, and the same step varied by up to 1.5 times with the load. The arb
parts use one thread and BLAS two. From `rt_cost_fit.py --table` on `cont/cont_32x101.jsonl` (32 × 101, L = 11,425):

| D | Newton steps | products per Newton step | s for the D step |
|---|---|---|---|
| 31 | 5 | 46–53 | 248 |
| 35 | 6 | 45–75 | 193 |
| 40 | 7 | 49–85 | 337 |
| 45 | 8 | 51–91 | 435 |
| 50 | 7 | 53–87 | 385 |
| 51 | 12 | 56–104 | 660 |
| 53 | 10 | 53–114 | 604 |
| 55 | 11 | 54–105 | 621 |
| 57 | 9 | 55–108 | 518 |
| 60 | 8 | 57–104 | 403 |

- **Products per Newton step.** The first Newton step of each D step uses factors fresh at the predictor. Its count
  grew about 2 per unit of D from D = 10 to 30 (#105) and about 0.4 per unit from 30 to 60: 46 at D = 31, 57 at 60.
  The fit over all 77 D from 10.5 to 60 gives 0.89 per unit (windows: 1.8 for the first half, 0.36 for the second, a
  spread over fit windows). The mean over a D step's Newton steps, which includes the steps on stale factors, grows
  0.92 per unit of D from 30 to 60 (0.45 to 1.5 over the windows).
- **Newton steps per D step.** These grew from 5 at D 31–36 to 7–12 at D 48–60. The fit over D 30–60 gives 0.12 per
  unit of D (0.11 for the first half; the second half is biased by the change of target, 10.5). The extra steps are
  Newton's jumps along the weak direction. A step from max|F| below 10⁻¹⁷ moves by 10⁻⁷ to 2×10⁻⁶ there and raises
  max|F| to 10⁻¹¹ before the quadratic phase resumes.
- **Seconds.** A product takes 0.48 s on 32 × 101, 0.65 s on 32 × 121 and 0.83 s on 32 × 141 (medians, L = 11,425,
  13,665, 15,905), about (N + 1)^1.7. A preconditioner application takes a quarter of that. The f64 Jacobian and its
  LU take 15, 24 and 33 s. The peak resident memory is 1.5, 1.9 and 2.5 GB, and 5.1 GB for 48 × 141 (L = 23,857). A D
  step took 190–350 s at D 31–41 and 380–680 s at D 45–60. By #105's linear predictor at steps of 1/2, one unit of D
  near D = 30 took about 7 min (2 × 256 s with its frequent rebuilds). Here it took 4 min, from the steps of 1 with
  the cubic predictor and one rebuild per D step.

### 10.5 Projection to D = 100 and 215 (an extrapolation)

`rt_cost_fit.py --project 60,100,215 --fixed_N 101` continues the two linear fits, products per Newton step and
Newton steps per D step, at 32 × 101's measured seconds per product. It gives the cost of continuing on 32 × 101
alone. It is an extrapolation over a factor of 2 to 4 in D beyond the data, and it is the cost only.

| D | window | products per Newton step | Newton steps per D step | s per D step | hours from D = 30 |
|---|---|---|---|---|---|
| 60 | all | 80 | 9.3 | 554 | 3.3 |
| 60 | first half | 92 | 8.8 | 625 | 3.4 |
| 100 | all | 116 | 13.9 | 1,269 | 13.2 |
| 100 | first half | 152 | 13.1 | 1,665 | 15.7 |
| 215 | all | 222 | 27.3 | 5,625 | 114 |
| 215 | first half | 323 | 25.4 | 8,918 | 166 |

The windows are those of the two fits: all D from 30 to 60, and the first half. The second half (D 45–60) gives
falling Newton steps (−0.09 per unit of D). That window is biased by the change of Newton's target for the steps
from D = 54 and 57, which dropped the last one or two Newton steps, so it is left out. Its rows are in
`fit/cost_fit.log`. The 30 measured steps from D = 31 to 60 took 3.35 hours in all, against 3.3 from the fits, with
the check points and the restarts not counted.

N(D) is not projected. The method planned, the grid differences fitted in D and continued in N at the measured rate
per 20 points, needs Δ_N − Δ_{N+20} to measure truncation. From D ≈ 40 it measures the finer grid's error along the
weak direction instead. With N held at 101, the error reaches 1.3×10⁻¹² at D = 60, and with N raised it is larger.
So the projection above prices a grid that no longer holds Δ to 10⁻¹² past D ≈ 60. A grid that does, if one exists,
needs a different discretization (10.7). For
scale, our chain's step at D = 200 took 460 s on 18 threads (section 5(3)).

### 10.6 Controls

| control | expected | result | log |
|---|---|---|---|
| `rt_run.py --mutate=coef` at D = 60.001952 from the unmutated 32 × 101 state | Δ moves by > 10⁻⁶, exit 1 | Δ moved by 5.3×10⁻⁴, exit 1 | `controls/coef_D60*` |
| the same at D = 54.996085 | Δ moves by > 10⁻⁶, exit 1 | Δ moved by 3.8×10⁻², exit 1 | `controls/coef_D54*` |
| `rt_grids.py --tol 1e-12`, 32 × 101 and 32 × 121 at D 34.99, 40.02 | exit 0 | exit 0 | `fit/grids_101_121_D35_40.log` |
| the same with `--mutate=shift` (2×10⁻¹² added to one Δ) | exit 1 | exit 1 | `fit/grids_101_121_D35_40_mutate_shift.log` |
| `rt_grids.py --tol 1e-12` at D 40.02, 44.96, 50.01 | (the acceptance on N and N + 20) | exit 1, at 50.01 | `fit/grids_101_121_D40_50.log` |
| `rt_grids.py --tol 1e-12`, the two 32 × 101 roots at D 40.02–60.00 | (the acceptance on 101) | exit 1, at 60.00 (−1.27×10⁻¹² and −2.33×10⁻¹²) | `fit/grids_101_roots.log` |
| `rt_mpsolve.py --selftest` at this code | error < 10⁻³⁰, exit 0 | 6.9×10⁻⁴⁰, exit 0 | `controls/planted.log` |
| `rt_cost_fit.py --planted` (slopes, N(D), the root-spread fit and the fixed-N cost recovered) | exit 0 | exit 0 | `fit/planted.log` |
| its `--mutate=rate` and `--mutate=slope` | exit 1 | exit 1, exit 1 | `fit/planted_mutate_*.log` |
| `rt_pred_check.py` (a planted cubic path predicted exactly at order 3) and its `--mutate=order` | exit 0, exit 1 | exit 0, exit 1 | `fit/pred_check*.log` |
| `rt_roots.py` (a planted 10⁻²⁰ difference along μ recovered) and its `--mutate=same` | exit 0, exit 1 | exit 0, exit 1 | `controls/roots_*.log` |

Neither mutated solve converged in its 20 Newton steps: max|F| fell from 4.1×10⁻⁴ to 6.0×10⁻⁸ at D = 60 and to
1.5×10⁻⁹ at 54.996, with Newton jumping along the weak direction. In #105 the same mutation converged at D = 15 and
29.98 and moved Δ by 3.75×10⁻⁴ and 3.06×10⁻³. The control still fires, through the check against hp, and its Δ
moved far more than the 10⁻⁶ required. But at D ≥ 55 a solve from 10⁻³ away no longer reaches the mutated root in 20
steps.

### 10.7 What this says about #91's verdict

#91 advised against adopting the global Fourier × Chebyshev Newton for the chain, for two reasons. The first was cost,
a dense extended-precision LU at hours per step. #105 refuted it, and this run confirms that: a unit of D costs 3 to
11 min on one core at L = 11,425 up to D = 60. Even continued at its fitted growth, staying on 32 × 101 would cost
13 to 16 hours from D = 30 to 100 (an extrapolation). The second reason was conditioning. A grid that holds the
solution needs its truncation below σ_min, and σ_min falls with N faster the larger D is. That reason stands, and it
is now measured as error in Δ on solved grids rather than inferred from κ:

- From D ≈ 40 refinement makes Δ worse. 32 × 141 is 1.0×10⁻¹⁴ off hp at D = 40.02, 3.8×10⁻¹² at 44.96 and 4.1×10⁻¹¹ at
  50.01. 32 × 121 is 2.9×10⁻¹² and 4.5×10⁻¹² off at 50.01, from different starts, and 3.5×10⁻¹¹ at 60.00. Meanwhile
  32 × 101 stays within 7.5×10⁻¹⁵ to D = 50 and 2.1×10⁻¹⁴ at 55, and misses by 1.27×10⁻¹² at 60.
- The finer grids' roots come in close sets along the weak direction, and the start decides which one Newton finds.
- Below the optimum the truncation is too large: 32 × 81 missed hp by 1.2×10⁻⁹ at D = 30 (#105).
- The optimal N falls with D: at least 141 at D = 35, 121 at 40, 101 at 45 to 55. The window of N that holds Δ to
  10⁻¹² therefore narrows with D, and at D = 60 it holds no N of the ladder (101 misses by 1.27×10⁻¹², 121 by
  3.5×10⁻¹¹). An N between 81 and 101 was not tried.

So the verdict on the chain should stay "do not adopt", with its reason restated. The obstacle is not the linear
algebra, which this solver makes cheap, but the discretization. A polynomial basis that does not exclude the
horizon's singular branch cannot fix Δ once the branch is smooth enough to pass for a polynomial at the resolution
needed for the rest of the solution. This is consistent with section 5(2): the same wall as our horizon series'
radius, met from the other side.

One thing could change the reading. A discretization that imposes the regular horizon behaviour explicitly, with
rows or a basis that remove the singular branch, might reopen the window. Whether it can do so without inheriting our
series' wall is open (Recommended tasks). What the prototype already gives is an independent check of Δ to about
10⁻¹⁴ up to D ≈ 55 on 32 × 101, by another gauge, other unknowns and another discretization. At D = 40 and 50 it lies
within 3.5×10⁻¹⁵ of our double-double rows.

### 10.8 Discrepancies (reported, not fixed)

- **The criterion for N, written before the runs, fails from D = 50.** |Δ_101 − Δ_121| = 2.9×10⁻¹² there, and its rule
  (move up a grid once the difference passes 10⁻¹³) would have moved the run to the grid that misses hp. The criterion
  assumed that refinement converges. From D ≈ 45 it does not (10.2, 10.3).
- **hp's row at D = 40.020065 (up7, err 2×10⁻¹⁵) sits 1.5–3.5×10⁻¹⁵ below the 32 × 101 and 32 × 121 values,** which
  agree with each other to 2.0×10⁻¹⁵ and whose roots are unique to 5×10⁻²³ and 5×10⁻²⁰. That is inside hp's stated
  error, so it is no disagreement, but the RT values do not test hp below 10⁻¹⁵ there.
- **The acceptance at the row nearest 60 is missed:** Δ_RT − Δ_hp = −1.27×10⁻¹² on 32 × 101 at D = 60.001952 (second
  root −2.33×10⁻¹²), against 10⁻¹². hp's row there (up13, err 2×10⁻¹⁵) is not in question. Every RT grid tried is
  further off, so the miss is the RT discretization's.
- **Section 9.7's estimate for #106** had the 32 × 101 count of products reaching 80 near D = 45, at about 2 more per
  unit of D. The first-step count grew only about 0.4 per unit of D from 30 to 60: 46 at D = 31, 57 at 60. The cost
  grew through extra Newton steps instead (10.4).
- **Section 5(2)'s mechanism, σ_min falling with N faster at larger D, now shows as error in Δ** on grids that hold the
  solution, not only as κ. The sentence of 5b that a grid fine enough to hold the solution "can check our values at D
  = 12–30" holds and extends to D ≈ 55 at 10⁻¹⁴. The README sentence proposed in section 7 stays correct.
- **Timings vary by up to 1.5 times** with the load on the shared Mac, for the same step: the pilot's step took 25 s
  and then 46 s per Newton step. The fits of seconds are fits of these.

### 10.9 Proposed text for the README

For the coordinator, after the RT sentence of "The large-D wall" (sections 7 and 9.8):

> Continued past D = 30 on 32 × 101, the RT prototype reproduces our Δ to 3×10⁻¹⁵ at D = 40 and 50 and to 2×10⁻¹⁴ at
> 55. It misses by 1.3×10⁻¹² at 60. There it also shows the wall in its own form. From D ≈ 40, refining the Chebyshev
> grid makes Δ worse (32 × 141 is 4×10⁻¹² off at D = 45, 32 × 121 3×10⁻¹² at D = 50), and the finer grids' roots are
> not unique along the horizon's near-polynomial singular branch. The usable resolution therefore narrows with D, and
> on these grids it closes for 10⁻¹² near D = 60 ([notes/RT_FORMULATION.md](notes/RT_FORMULATION.md) section 10).

### 10.10 Reproduction

From the repository root, with the raw-data repository cloned beside it (RAW = `../choptuik-large-d-raw`). On the
Mac, use `/Users/nathan/venv-qd/bin/python` with the MULTIDOUBLE_SITE of 9.9. The drivers in `RAW/I145-mac/` hold the
exact commands of the runs: `run_cont.sh` (the continuation; its five restarts are listed in the folder's README),
`run_worker*.sh` (the re-solves), `run_roots.sh`, `run_roots_logs.sh`, `run_coef.sh` and `run_final60.sh`.

```
python3 solver/rt/rt_pred_check.py RAW/I105-mac/cont cont_32x101 --grid 32x101   # 10.1; --mutate=order must exit 1
RAW/I145-mac/run_cont.sh 101 RAW/I105-mac/cont/cont_32x101_D30.0000_32x101.npz \
    RAW/I105-mac/cont/cont_32x101_D28.0000_32x101.npz,RAW/I105-mac/cont/cont_32x101_D29.0000_32x101.npz cont_32x101
                                                     # the continuation's first leg (quadratic predictor)
python3 solver/rt/rt_run.py --start RAW/I145-mac/cont/cont_32x101_D40.0201_32x101.npz --grids 32x141 --solver mp \
    --tol 1e-12 --max_products 250 --refresh_dz 1e-5 --patient --it_max 20 --full_below 1e-4 --out OUT --tag t
                                                     # one re-solve (10.2): Delta 1.65e-14 from hp, exit 0
python3 solver/rt/rt_roots.py RAW/I145-mac/fol/fol_32x141_D40.0201_32x141.npz \
    RAW/I145-mac/controls/from101_32x141_D40.0201_32x141.npz   # 10.3: Delta_2 - Delta_1 = +6.4e-15; --mutate=same exit 1
python3 solver/rt/rt_grids.py RAW/I145-mac/cont/cont_32x101.jsonl RAW/I145-mac/fol/fol_32x121.jsonl \
    --D 34.993266,40.020065 --tol 1e-12              # exit 0; with --mutate=shift exit 1
python3 solver/rt/rt_grids.py RAW/I145-mac/cont/cont_32x101.jsonl RAW/I145-mac/fol/fol_32x121.jsonl \
    --D 40.020065,44.963411,50.007491 --tol 1e-12    # exit 1, at 50.007491
python3 solver/rt/rt_grids.py RAW/I145-mac/cont/cont_32x101.jsonl RAW/I145-mac/fit/roots101_second.jsonl \
    --D 40.020065,44.963411,50.007491,54.996085,60.001952 --tol 1e-12   # exit 1, at 60.001952
python3 solver/rt/rt_cost_fit.py --planted           # exit 0; --mutate=rate and --mutate=slope exit 1
python3 solver/rt/rt_cost_fit.py RAW/I145-mac/cont/cont_32x101.jsonl RAW/I145-mac/fol/fol_32x121.jsonl \
    RAW/I145-mac/fol/fol_32x141.jsonl --old RAW/I105-mac/cont/cont_32x101.jsonl --table --project 60,100,215 \
    --fixed_N 101 --alt RAW/I145-mac/fit/roots101_second.jsonl,RAW/I145-mac/controls/from141_32x121.jsonl,\
RAW/I145-mac/fol/from141_32x121.jsonl,RAW/I145-mac/fol/from101b_32x121.jsonl,RAW/I145-mac/controls/from101_32x141.jsonl
                                                     # 10.2-10.5 (fit/cost_fit.log)
RAW/I145-mac/run_coef.sh D60.0020                    # 10.6: must exit 1
```

## 11. Between 81 and 101 Chebyshev points at D = 55 and 60

Issue #164. Subagent of the coordinator on the Mac, 2026-11-08. The solver is #145's (`rt_run.py --solver mp`,
`rt_grids.py`), unchanged. `rt_roots.py` has a new pair test (11.4). Raw output is in folder `I164-mac/` of the
raw-data repository.

**Result.** Yes. On 32 × 95 the prototype holds Δ to 10⁻¹² at D = 60.001952. It lies −6.4×10⁻¹³ and −5.9×10⁻¹³ from
hp from the two starts of the criteria, the two roots agree to 5.3×10⁻¹⁴, and `rt_grids.py --tol 1e-12` exits 0. Of
the seven grids run at both D from both starts (85 to 97), it is the only one. Two further starts land within
1.2×10⁻¹⁵ of the first root. Outside the criteria, 32 × 99 holds too, at −3.5×10⁻¹³ and −2.7×10⁻¹³. At D = 54.996085,
32 × 95 lies −5.5×10⁻¹³ from hp and 32 × 97 +1.25×10⁻¹³.

The odd N fall into two families by N mod 4. The family sets the sign of the truncation and, at D = 60, how far apart
the grid's roots lie.

- **Truncation.** From 87 to 97, Δ_N − Δ_hp has the sign of (−1)^((N−1)/2), and within a family it falls about a
  decade per step of 4 in N. From D = 55 to 60 it changes by less than a factor 1.7, for example 2.9×10⁻¹¹ and
  2.5×10⁻¹¹ at N = 89. The sign rule was written down before five of its results, and all five followed it.
- **Pairs of roots.** At D = 54.996085 the two starts give the same root to 2×10⁻¹⁴ on every grid from 85 to 97. At
  D = 60.001952 the two starts find two distinct roots on 89, 93, 95, 97 and 99: the residual between them rises to
  10⁻²⁸ to 10⁻²⁴, against 2×10⁻³¹ at the roots. The roots of the N ≡ 1 mod 4 grids (89, 93, 97, and #145's 101) differ
  by 4.6×10⁻¹³ to 1.1×10⁻¹² in Δ. Those of 95 and 99 differ by only 5.3×10⁻¹⁴ and 7.5×10⁻¹⁴.

So 32 × 97 misses at D = 60 by its pair (+8×10⁻¹⁴ and +1.08×10⁻¹² from hp), not by its truncation. #145's ladder,
N = 81, 101, 121 and 141, lies entirely in the family N ≡ 1 mod 4. Its reading that refinement makes Δ worse from
D ≈ 40 rests on that family alone (11.6).

### 11.1 The protocol

The criteria were written into `I164-mac/README.md` before the main runs (raw commit f1447e6), and not changed.

- Options as #145 at D = 55–60: `--solver mp --tol 1e-12 --max_products 250 --refresh_dz 1e-5 --patient --it_max 20
  --full_below 1e-4`, with `--newton_tol 1e-38` at D = 54.996085 and 1e-36 at 60.001952. A root counts when its record
  has ok true.
- Two starts per grid and D, each resampled to 32 × N and solved there at the same D. Start a is #145's 32 × 101
  continuation state (−1.27×10⁻¹² from hp at D = 60.001952). Start b is #145's 32 × 121 root solved from it
  (+3.5×10⁻¹¹). Their Δ differ by 3.6×10⁻¹¹ at D = 60 and 8.6×10⁻¹² at 55. Resampled to 85–99, both start at max|F| =
  4×10⁻¹⁴ to 1.2×10⁻¹³.
- Pass for grid N at D = 60.001952: `rt_grids.py` on its two roots with `--tol 1e-12` exits 0, that is both converged,
  both within 10⁻¹² of hp, and within 10⁻¹² of each other. The verdict of #164 is yes when some grid passes.
- hp is the double-double row of `hp/data/delta_all.csv` (up10 at 54.996085, up13 at 60.001952, err 2×10⁻¹⁵).
- Grids. The main runs were N = 85, 91 and 95. The rule for more, written beforehand, fired on their D = 60 results:
  the smallest error, 6.4×10⁻¹³ at 95, was below 10⁻¹¹, and 95 beat 101. So 87, 89, 93 and 97 were run too. After the
  D = 60 results of 93 and 97, two additions outside the verdict were run. One is 32 × 99 from both starts, the one odd
  N between 97 and 101 not yet run. The other is 32 × 95 from two more starts on the other side of hp, the start-b
  roots of 93 and 97 resampled.

Every 32 × N solve converged in 4 to 7 Newton steps. The first step takes max|F| to 2×10⁻²⁹–5×10⁻²³. A later step
then moves the state by 10⁻¹⁴ to 2.4×10⁻¹⁰, by more on the coarser grids (3×10⁻¹¹ on 85, 10⁻¹⁴ to 3×10⁻¹² on 97),
while max|F| stays put or rises. Newton then converges quadratically. That step is the move along the weak direction to
the grid's own root.

### 11.2 Δ against N

Δ_RT − Δ_hp for each start, and the spread Δ_b − Δ_a of the two roots. The spread is a lower bound on how far the grid
leaves Δ undetermined, not an error bar. The 101 and 121 rows are #145's (10.2, 10.3), whose second roots came from
other starts. From `fit/table_32xN.log` and `fit/verdict_32xN.log` (`rt_grids.py`).

| N | N mod 4 | D = 54.996085, a | b | spread | D = 60.001952, a | b | spread | at D = 60 |
|---|---|---|---|---|---|---|---|---|
| 85 | 1 | +1.89×10⁻¹⁰ | +1.89×10⁻¹⁰ | −2×10⁻²⁰ | −3.42×10⁻¹¹ | −3.42×10⁻¹¹ | −1.0×10⁻¹⁵ | fail |
| 87 | 3 | −8.65×10⁻¹¹ | −8.65×10⁻¹¹ | +2.0×10⁻¹⁴ | −5.24×10⁻¹¹ | −5.24×10⁻¹¹ | +2.2×10⁻¹⁵ | fail |
| 89 | 1 | +2.88×10⁻¹¹ | +2.88×10⁻¹¹ | +8×10⁻¹⁹ | +2.58×10⁻¹¹ | +2.51×10⁻¹¹ | −7.1×10⁻¹³ | fail |
| 91 | 3 | −8.32×10⁻¹² | −8.32×10⁻¹² | +1.2×10⁻¹⁶ | −8.37×10⁻¹² | −8.37×10⁻¹² | +2.3×10⁻¹⁵ | fail |
| 93 | 1 | +2.20×10⁻¹² | +2.20×10⁻¹² | +9.4×10⁻¹⁶ | +2.35×10⁻¹² | +2.81×10⁻¹² | +4.6×10⁻¹³ | fail |
| 95 | 3 | −5.46×10⁻¹³ | −5.46×10⁻¹³ | −1.1×10⁻¹⁶ | **−6.38×10⁻¹³** | **−5.85×10⁻¹³** | +5.3×10⁻¹⁴ | **pass** |
| 97 | 1 | +1.25×10⁻¹³ | +1.25×10⁻¹³ | −5×10⁻¹⁷ | +8.4×10⁻¹⁴ | +1.08×10⁻¹² | +1.0×10⁻¹² | fail |
| 99 | 3 | not run | | | −3.45×10⁻¹³ | −2.70×10⁻¹³ | +7.5×10⁻¹⁴ | pass (outside the criteria) |
| 101 (#145) | 1 | −2.1×10⁻¹⁴ | +1.5×10⁻¹⁴ | +3.6×10⁻¹⁴ | −1.27×10⁻¹² | −2.33×10⁻¹² | −1.1×10⁻¹² | fail |
| 121 (#145) | 1 | +8.5×10⁻¹² | +9.7×10⁻¹³ | −7.6×10⁻¹² | +3.5×10⁻¹¹ | | | fail |

The 28 solves of the criteria, 85–97, converged to a final max|F| of at most 7.8×10⁻³³, in 104 to 316 s on one core.
Each needed at most one rebuild of the preconditioner and peaked at 1.0 to 1.3 GB of resident memory. The four solves
added afterwards took 3 to 4 min each. The further starts of 32 × 95 at D = 60.001952 give −6.37×10⁻¹³ (from 97's
root) and −6.39×10⁻¹³ (from 93's root), within 1.2×10⁻¹⁵ of start a's root (`fit/verdict_32x95_a_*.log`, both exit 0).
Three of the four starts find one root, and start b finds the other root of 95's pair, 5.3×10⁻¹⁴ away.

At D = 54.996085 every grid from 85 to 97 has its two roots within 2×10⁻¹⁴, and the error is the truncation alone.
There 97 holds 1.25×10⁻¹³ and #145's 101 holds 2×10⁻¹⁴. At D = 60.001952 the error of 95 and 99 is still of the size
their truncation trend gives, and their pairs are 10 to 20 times closer than those of 93, 97 and 101.

### 11.3 Truncation: two families by N mod 4

The magnitudes, at D = 54.996085 and then 60.001952 (both roots where they differ):

| N | 85 | 87 | 89 | 91 | 93 | 95 | 97 | 99 |
|---|---|---|---|---|---|---|---|---|
| (−1)^((N−1)/2) | + | − | + | − | + | − | + | − |
| D = 54.996085 | +1.89×10⁻¹⁰ | −8.65×10⁻¹¹ | +2.88×10⁻¹¹ | −8.32×10⁻¹² | +2.20×10⁻¹² | −5.46×10⁻¹³ | +1.25×10⁻¹³ | |
| D = 60.001952 | −3.42×10⁻¹¹ | −5.24×10⁻¹¹ | +2.5–2.6×10⁻¹¹ | −8.37×10⁻¹² | +2.4–2.8×10⁻¹² | −5.9–6.4×10⁻¹³ | +0.08–1.08×10⁻¹² | −2.7–3.5×10⁻¹³ |

Within each family the error falls by 1.0 to 1.25 decades per step of 4 at D = 55 (87 → 91 → 95 and 89 → 93 → 97),
and by 0.8 to 1.1 at D = 60 up to 95. Across the families the magnitudes interleave, so |Δ_N − Δ_hp| falls about 0.27
decade per point of N, while its sign alternates with each step of 2. From 95 to 99 at D = 60 it falls only by a factor
1.7 to 2.2.

The sign rule was recorded in `I164-mac/README.md` at 14:02 UTC (raw commit 014f3f4), from the D = 60 results of 89 to
97 and the D = 55 results of 85, 91 and 95. It named the signs of 87 at D = 60 and of 87, 89, 93 and 97 at D = 55
before they ran, and all five came out as named. 85 at D = 60 breaks it. So does 101 at both D, where the spread of
#145's roots is as large as the error. A second prediction, recorded at 14:06 UTC (raw commit 3b92a77), gave 32 × 99
at D = 60 a negative error of about 5×10⁻¹⁴ to 10⁻¹³ with its roots within 10⁻¹³. The sign and the spread came out as
predicted, the size did not (−2.7×10⁻¹³ and −3.5×10⁻¹³).

A reading, not tested here: the equations are divided by ξ (RT's S), and ξ = 0 lies between two grid points for odd N.
The leading neglected Chebyshev term there carries T_{N+1}(0) = (−1)^((N+1)/2), whose sign flips with each step of 2
in N. That is the observed pattern up to an overall sign. The error is not the interpolation of ω₃ to ξ = 0, since Δ
evaluated at ξ = 1 equals Δ at ξ = 0 to f64 rounding on every grid (the records' Delta_xi1 and Delta_xi0). It is in
the discrete solution itself.

### 11.4 Pairs of roots at D = 60

`rt_roots.py` now tests whether two roots are distinct. For the quadratic residual, F at the midpoint zm of z1 and z2
is (F(z1) + F(z2))/2 − B(e, e)/4, with e = z2 − z1. So along the line through them F is the parabola
−s(1 − s) B(e, e), up to the roots' residuals. When max|F(zm)| is far above both roots' residuals, they are two
distinct roots of the discrete system. Then J at either root acts along e with max|B(e, e)|/max|e|, an upper bound on
how strongly it acts in its weakest direction (max norms). The states are stored in double-double, so their residuals
are about 2×10⁻³¹. The test's planted pair, z and z(1 + 10⁻¹²), recovers B(e, e) = 10⁻²⁴ B(z, z) to 10⁻³⁰, where
B(z, z) comes from F at 0, z and 2z (`fit/roots_*.log`).

| N, D = 60.001952 | max\|z₂ − z₁\| | Δ₂ − Δ₁ | max\|F(zm)\| | max\|B(e, e)\|/max\|e\| | verdict |
|---|---|---|---|---|---|
| 85 | 1.0×10⁻¹⁶ | −1.0×10⁻¹⁵ | 5.8×10⁻³¹ | | hidden by the floor |
| 87 | 1.9×10⁻¹⁶ | +2.2×10⁻¹⁵ | 5.9×10⁻³¹ | | hidden by the floor |
| 89 | 1.07×10⁻¹³ | −7.1×10⁻¹³ | 8.6×10⁻²⁵ | 3.2×10⁻¹¹ | two roots |
| 91 | 1.5×10⁻¹⁶ | +2.3×10⁻¹⁵ | 2.4×10⁻³¹ | | hidden by the floor |
| 93 | 4.7×10⁻¹⁴ | +4.6×10⁻¹³ | 1.1×10⁻²⁵ | 9.7×10⁻¹² | two roots |
| 95 | 7.7×10⁻¹⁵ | +5.3×10⁻¹⁴ | 4.4×10⁻²⁷ | 2.3×10⁻¹² | two roots |
| 97 | 7.0×10⁻¹⁴ | +1.0×10⁻¹² | 3.5×10⁻²⁷ | 2.0×10⁻¹³ | two roots |
| 99 | 4.9×10⁻¹⁵ | +7.5×10⁻¹⁴ | 1.8×10⁻²⁸ | 1.5×10⁻¹³ | two roots |

At D = 54.996085 every pair of 85–97 is hidden by the floor, with max|z₂ − z₁| at most 1.5×10⁻¹⁵ (87) and below
2×10⁻¹⁶ on the others. The further starts of 95 at D = 60 lie 3×10⁻¹⁷ and 10⁻¹⁶ from start a's root, also hidden.

What the pairs are made of. The difference of each pair is mostly μ and harmonic 0 of ω₂, ω₃ and ω₄, spread over ξ
with no peak at the ends ξ = ±1, where the horizon lies in the parity-extended form. μ is the largest component on
93, 97 and 99, and 13% of the largest on 89. That is section 4's weakest direction. The strength of J along e at the
roots falls from 3.2×10⁻¹¹ at 89 to 1.5×10⁻¹³ at 99, about two decades over 10 points, as section 4's σ_min falls
with N.

So past D = 55 the discrete systems from N = 89 up have two nearby roots along their weakest direction, a nearly
degenerate pair such as a quadratic system has close to a fold, and the start decides which one Newton finds. The pair
is 10 to 20 times tighter for N ≡ 3 mod 4 (95, 99) than for its neighbours N ≡ 1 mod 4 (93, 97). Whether the two
families differ because the truncation, whose sign depends on the family (11.3), pushes the system toward the fold in
one and away from it in the other, is a hypothesis not tested here. For #163 (horizon regularity), the measurable
targets are the pair's max|z₂ − z₁| and max|B(e, e)|/max|e| on the same grids at D = 60, with and without rows that
impose the regular horizon behaviour.

### 11.5 Controls

| control | expected | result | log |
|---|---|---|---|
| `rt_run.py --mutate=coef` at D = 60.001952 on 32 × 91 from its unmutated start-a root | Δ moves by > 10⁻⁶, exit 1 | Δ moved by 1.31×10⁻², exit 1 | `controls/coef_32x91_mutcoef*` |
| `rt_grids.py --mutate=shift` on 32 × 95's two roots at D = 60.001952 | exit 1 | exit 1 | `fit/verdict_32x95_mutate_shift.log` |
| `rt_roots.py --mutate=same` on 32 × 95's roots at D = 60.001952 | exit 1 | exit 1 | `fit/roots_32x95_D60.0020_mutate_same.log` |
| `rt_roots.py --mutate=mid` on 32 × 97's roots at D = 60.001952 (the pair test's planted check) | exit 1 | exit 1, B(e, e) off by 2×10¹¹ relative | `fit/roots_32x97_D60.0020_mutate_mid.log` |
| 48 × 95 against 32 × 95, both from start a at D = 60.001952 (`nt_check.py`) | \|difference\| < 10⁻¹³ | +1.5×10⁻¹⁶, exit 0 | `fit/nt_48x95.log` |
| the same check with a tolerance of 10⁻¹⁷ | exit 1 | exit 1 | `fit/nt_48x95_tol1e-17.log` |

The mutated solve did not converge in its 20 Newton steps (max|F| from 4.1×10⁻⁴ to 1.1×10⁻⁸, 14 rebuilds), as in
#145 at D = 55 and 60 (10.6). It fires through the check against hp. The 48 × 95 solve started at max|F| = 7.6×10⁻¹⁰,
since resampling from 32 to 48 points in t leaves a larger residual than resampling in ξ (4×10⁻¹⁴). Newton then moved
along the weak direction by up to 8.8×10⁻⁷ for six steps, with max|F| between 10⁻¹³ and 7×10⁻¹¹, before it converged
in 12 steps (26 min, 2.5 GB). Its Δ lies 1.5×10⁻¹⁶ from 32 × 95's, so N_t = 32 is not what sets 32 × 95's error.

### 11.6 Discrepancies (reported, not fixed)

- **Section 10's reading of refinement rests on one family.** 10.2, 10.3 and 10.7 conclude that from D ≈ 40 refining
  the Chebyshev grid makes Δ worse and that the finer grids' roots are not unique along the weak direction. Every grid
  of that ladder (81, 101, 121, 141) has N ≡ 1 mod 4. At D = 60 the grids N ≡ 3 mod 4 run here (87, 91, 95, 99) have
  roots unique to the floor or paired within 7.5×10⁻¹⁴, and 95 and 99 hold 10⁻¹², while 93, 97 and 101 do not.
  Whether the family N ≡ 3 mod 4 keeps converging past 99, and how it behaves from D = 40 to 55, was not measured. Its
  gain from 95 to 99 at D = 60 is only a factor 2, so it may stall near 3×10⁻¹³ there (an extrapolation). The
  coordinator decides whether 10.7's reading of the discretization changes (Recommended tasks).
- **10.9's proposed README sentence** says that on these grids the usable resolution "closes for 10⁻¹² near D = 60".
  32 × 95 and 32 × 99 hold 10⁻¹² at D = 60.001952, so the sentence needs the qualification of 11.7.
- **The issue's premise** (from #145) is that the finer grids fail because the horizon's singular branch makes the
  roots non-unique. That does not explain a dependence on N mod 4, since the part of the branch that degree-N
  polynomials miss changes smoothly with N. The sign of the truncation does depend on N mod 4 (11.3), so the
  separation of the pairs may come from how the truncation meets the weak direction (the hypothesis of 11.4).
- **The size of the prediction for 32 × 99 failed** (11.3). Its error is −2.7×10⁻¹³ to −3.5×10⁻¹³, not 5×10⁻¹⁴ to
  10⁻¹³. Its sign and the closeness of its roots came out as predicted.
- **32 × 85 at D = 60** breaks the sign rule (−3.42×10⁻¹¹ where 87–97 predict +). Its error changes by a factor 5.5
  with a change of sign from D = 55, where every other grid changes by less than 1.7. Its two roots agree to
  10⁻¹⁵. No cause was found.
- **Two of the checks ran wrong first, and were redone.** The first `nt_check.py` compared the 48 × 95 record at
  D = 60 with 32 × 95's last record, at D = 55, and failed (`checks.stdout`). It now takes the record at the same D.
  `run_checks.sh` reached the third start of 95 before its solve had finished, and `run_checks_c97.sh` redid that part.
- **hp's rows** are not in question. The closest RT values lie on both sides of them: 97 and 101 at D = 55
  (+1.25×10⁻¹³, −2.1×10⁻¹⁴ and +1.5×10⁻¹⁴), and 99 and 97 at D = 60 (−2.7×10⁻¹³, +8×10⁻¹⁴).

### 11.7 Proposed text for the README

For the coordinator, in place of the last sentence of 10.9's text:

> Between 85 and 99 Chebyshev points, Δ's truncation error alternates in sign with N mod 4 and falls about a decade per
> 4 points, nearly the same at D = 55 and 60. At D = 60 the 32 × 95 and 32 × 99 grids hold Δ to 6×10⁻¹³ and 3.5×10⁻¹³.
> At D = 60 most grids of 89 points and more have two roots close together along the weak direction, and the start
> decides which one Newton finds. These pairs are 4.6×10⁻¹³ to 1.1×10⁻¹² apart in Δ for N ≡ 1 mod 4 (93, 97, 101) and
> 10 to 20 times closer for N ≡ 3 mod 4 (95, 99). Every grid #145 refined (81, 101, 121, 141) has N ≡ 1 mod 4
> ([notes/RT_FORMULATION.md](notes/RT_FORMULATION.md) section 11).

### 11.8 Reproduction

From the repository root, with the raw-data repository cloned beside it (RAW = `../choptuik-large-d-raw`). On the
Mac, use `/Users/nathan/venv-qd/bin/python` with the MULTIDOUBLE_SITE of 9.9. The drivers in `RAW/I164-mac/` hold the
exact commands: `run_main.sh` (one start, a list of grids, both D), `run_queue.sh` (the added grids, 48 × 95 and the
coefficient mutation), `run_extra.sh` and `run_extra_a.sh` (32 × 99 and the further starts of 32 × 95), and
`run_checks.sh` with `run_checks_c97.sh` (every `rt_grids.py`, `rt_roots.py` and `nt_check.py` log of `fit/`). Their
CODE line names the checkout.

```
python3 solver/rt/rt_run.py --start RAW/I145-mac/cont/cont_32x101_D60.0020_32x101.npz --grids 32x95 --solver mp \
    --tol 1e-12 --max_products 250 --refresh_dz 1e-5 --patient --it_max 20 --full_below 1e-4 --newton_tol 1e-36 \
    --out OUT --tag a101_32x95                   # 11.2: Delta -6.38e-13 from hp, exit 0 (about 5 min)
python3 solver/rt/rt_grids.py RAW/I164-mac/roots/a101_32x95.jsonl RAW/I164-mac/roots/b121_32x95.jsonl \
    --D 60.001952 --tol 1e-12                    # the verdict: exit 0; with --mutate=shift exit 1
python3 solver/rt/rt_grids.py RAW/I164-mac/roots/a101_32x97.jsonl RAW/I164-mac/roots/b121_32x97.jsonl \
    --D 60.001952 --tol 1e-12                    # exit 1 (start b +1.08e-12)
python3 solver/rt/rt_roots.py RAW/I164-mac/roots/a101_32x97_D60.0020_32x97.npz \
    RAW/I164-mac/roots/b121_32x97_D60.0020_32x97.npz   # 11.4: two roots; --mutate=mid and --mutate=same exit 1
python3 RAW/I164-mac/nt_check.py RAW/I164-mac/roots/a101_48x95.jsonl RAW/I164-mac/roots/a101_32x95.jsonl
                                                 # 11.5: +1.5e-16, exit 0; with --tol 1e-17 exit 1
```

### 11.9 The N ≡ 3 mod 4 ladder past 99 (#404)

Issue #404, run on win (Windows, Python 3.10), 2026-11-12, with #164's solver and options. The criteria were written
into `I404-win/README.md` before the first solve (raw d9bef65e). The question was whether the family that held Δ at
D = 60 on 95 and 99 keeps converging as N grows. It does at D = 40, not at D ≥ 55. E(N) is the larger of the two
roots' |Δ_RT − Δ_hp|, from both starts (raw `I404-win/` at 373ee026):

| N | D = 40.020065 | D = 50.007491 | D = 54.996085 | D = 60.001952 |
|---|---|---|---|---|
| 99 | 8.2×10⁻¹⁵ | 2.3×10⁻¹⁴ | 3.8×10⁻¹⁴ | 1.8×10⁻¹² |
| 103 | 1.2×10⁻¹⁵ | 3.0×10⁻¹⁵ | 7.5×10⁻¹⁴ | 4.9×10⁻¹¹ |
| 107 | 1.5×10⁻¹⁵ | 7.5×10⁻¹⁵ | 5.4×10⁻¹² | 1.3×10⁻¹⁰ |
| 111 | 1.6×10⁻¹⁵ | 6.6×10⁻¹⁴ | 6.9×10⁻¹² | 5.7×10⁻¹⁰ |
| 115 | 1.6×10⁻¹⁵ | 3.1×10⁻¹³ | 8.5×10⁻¹¹ | 2.1×10⁻¹¹ |
| 119 | 1.6×10⁻¹⁵ | 2.6×10⁻¹² | 2.3×10⁻¹⁰ | 1.0×10⁻¹¹ |
| 123 | 1.5×10⁻¹⁵ | 3.3×10⁻¹¹ | 1.2×10⁻⁹ | 1.6×10⁻⁷ |
| the two roots distinct from N | none to 123 | 111 | 107 | 99 |

- At D = 40 the ladder falls below 10⁻¹⁴ at 99 and stays, on a plateau of +1.55×10⁻¹⁵, inside hp's own error
  (2×10⁻¹⁵).
- At D = 50 it reaches 3.0×10⁻¹⁵ at 103, and 103 and 107 pass.
- At D = 55 and 60 it stops falling at 99, and no grid passes 10⁻¹⁴.
- The rise is a family of nearly degenerate roots: residuals below 10⁻³⁰, spread over Δ by 10⁻¹² to 10⁻⁷. Its onset
  falls with D, from above 123 at D = 40 to 111, 107 and 99 at D = 50, 55 and 60.

So the N ≡ 3 mod 4 family has a usable resolution up to about D = 50 and not past it. Section 11's reading, that
refining makes Δ worse only in the other family, holds below that onset and fails above it. `rt_grids.py
--mutate=shift`, `rt_roots.py --mutate=mid` and `--mutate=same` exit 1 at D = 60, as they must.

**Discrepancies.** At D = 60, 32 × 99's root b is −1.80×10⁻¹² on win against −2.70×10⁻¹³ on the Mac (#164), while root a
agrees to 10⁻¹⁵. That is the size of the root family's own spread. The RT code needed three stand-ins on win's
Python 3.10 (`multidouble._dd_dot`, `math.fma`, the Unix `resource` module), with no repository file changed, recorded
in the raw README. #405 (past D = 60 on 95 and 99) is closed, since the family degenerates there already.

## 12. EEG's resolution: the truncation of their 40 × 80 grid against D

Issue #106. Subagent of the coordinator on the Mac, 2026-11-08. Code in `solver/rt/`: the new `rt_trunc_fit.py`;
the runs use #145's `rt_run.py` unchanged. Raw output is in folder `I106-mac/` of the raw-data repository, whose README
holds the criteria, written before the main runs.

**Result.** EEG's 40 × 80 grid is `--grids 40x79` in the prototype. It has the same 11,200 unknowns plus μ, the same
Chebyshev degree and the same Fourier frequencies, but it collocates where they truncate in coefficient space.

- **At the three D of the criteria.** The prototype's Δ departs from ours by δ = −5.6×10⁻¹⁰ at D = 20.04, −7.1×10⁻⁹ at
  40.02 and +5.8×10⁻⁸ at 60.00. Two starts agree to 10⁻²⁰ or better at each D. The controls lie 1.7×10⁻²⁰, 1.5×10⁻¹⁵
  and 2.3×10⁻¹² from hp, all within 10⁻⁹.
- **The fit written before the runs cannot decide.** δ crosses zero at D = 46.5, and the log fits of the tail after it
  spread over more than 20 decades at D = 100–190.
- **δ itself is smooth.** From D = 47 on it grows like a cubic in D. Continued past the criteria to D = 84, it follows
  the prediction registered before those runs, and at D = 70 and 80 it lies inside the predicted ranges themselves.
  Carried on to D = 100, 150 and 190, that prediction gives 6.2–9.5×10⁻⁷, 2.1–4.9×10⁻⁶ and 4.0–11.5×10⁻⁶ (an
  extrapolation, a spread over fit choices).
- **Against EEG.** From D = 66 to 84, δ is 0.64 to 1.35 times EEG's published departure from our values, 0.8 on
  average. Their departure is averaged over D ± 5, and it is +1.7×10⁻⁷ at 66 and +4.4×10⁻⁷ at 80. At 100, 150 and 190
  it is 9.4×10⁻⁷, 2.8×10⁻⁶ and 4.1×10⁻⁶, inside the extrapolated ranges.

So at their resolution, collocation's truncation has the size and the growth of their departure from D ≈ 60. That is
evidence that their 40 × 80 truncation can account for it. It is not a measurement of their truncation. The sign of the
collocation's error alternates from one grid to the next (40 × 77 is −3.0×10⁻⁷ and 40 × 81 is −9.9×10⁻⁹ at D = 60),
and each grid's error crosses zero at its own D. A Galerkin projection of the same degree has its own zeros, so the
collocation fixes the scale of their truncation, not its sign or its value.

### 12.1 Their grid and ours

EEG expand each field as Σ g_nm exp(i n t/2 + i m θ), θ = arccos ξ, on (t, ξ) ∈ [0, 4π] × [−1, 1], with
N_Fourier = 40 and N_Chebyshev = 80. They count 40 · 80 · 2 · (1/4 + 1/2 + 1/2 + 1/2) = 11,200 real unknowns
(arXiv:2611.07007v1, Sec. III). Their infrastructure is RT's (their words: "directly adopted from Sec. 3 in [31]"). RT
count non-negative indices, "0 ≤ m < 250" Fourier and "0 ≤ n < 750" Chebyshev, with the negative ones following from
reality and evenness in θ (arXiv:1203.3766v1, Sec. 1.3 and 3). So 40 × 80 means the following.

- In t, each 2π-periodic field keeps the even n, which are the frequencies 0 to 19 of e^{ikt}. The antiperiodic ω₄ keeps
  the odd n, the frequencies 1/2 to 39/2.
- In ξ, the Chebyshev degrees are 0 to 79, in ξ itself with no map.

`rt_run.py --grids NtxN` is N_t collocation points in t on [0, 2π) and the N + 1 Chebyshev–Gauss–Lobatto points
ξ_j = cos(πj/N). N must be odd (`rt_model.RT` asserts it), so that ξ = 0, where the equations are divided by ξ, is not a
point. The unknowns number L = 3.5 N_t (N + 1) + 1. EEG's 40 × 80 is therefore `--grids 40x79`. It has L = 11,201,
which is EEG's 11,200 plus μ. It shares their polynomial degree in ξ, their frequencies in t and the halving of the odd
ω₁. It also shares their independent equations ((1 + P)Ω₁, Ω₂⁺, Ω₃⁺, Ω₄⁺, "similarly to RT"), their phase condition
Re(ω₄⁻)₁,₀ = 0, the absence of boundary rows, and Δ from their Eq. (3). Their 40 × 80 written as N = 80 (`40x80`) is not
a valid grid here.

What no NtxN can match:

- **The projection.** EEG truncate in coefficient space. Products are exact convolutions, and the residual's retained
  Fourier–Chebyshev coefficients are set to zero, which is a Galerkin projection. The prototype collocates: the residual
  vanishes at the 40 × 80 points, so its truncation error carries aliasing that theirs does not.
- **The division by ξ.** The prototype divides the collocated equations by ξ at the points. RT's coefficient-space S is
  the regularized left inverse (f − f(0))/ξ (their Eq. (18e)). EEG do not state theirs.
- **The Nyquist mode.** On an even N_t each periodic field also carries a cosine at frequency 20, with its t-derivative
  set to zero. EEG's index set has no such mode.
- **The path.** The prototype starts at D = 20 and steps by about 1 with a cubic predictor. EEG run from D = 4 in steps
  of 1/2 with a linear predictor. Roots are not unique along the weak direction (section 10.3), so the path can decide
  which root is found.

So the neighbouring grids on both sides in N, 40 × 77 and 40 × 81, were run as well. They show how much Δ at this
resolution depends on the details of the discretization.

### 12.2 The protocol

The solver is #105's (9.1) with #145's options (10.1): `rt_run.py --solver mp` at 192 bits, `--max_products 250
--refresh_dz 1e-5 --patient --it_max 20 --full_below 1e-4`, Newton's target 10⁻⁴⁰ below D = 54 and 10⁻³⁶ from
D = 54.996. The criteria were written into the raw folder's README before the main runs and were not changed. They are
the control, the two starts at each D, the fit models and windows, EEG's window means and the verdict rule.

- **Start A**, the continuation on 40 × 79, starts from #105's 32 × 81 state at D = 20.043653, resampled. It lands on
  the hp row nearest each integer D from 21 to 60 (41 points), with a cubic predictor through the last four solutions.
- **Start B** at D = 20.043653, 40.020065 and 60.001952 is the 32 × 101 state of #105 or #145 at that D, resampled to
  40 × 79 and solved by Newton.
- **The control** is 32 × 121 at D = 20.04 and 40.02, and 32 × 101 at 60.00 from #145's 32 × 121 root. Each must lie
  within 10⁻⁹ of hp.
- **δ(D) = Δ(40 × 79) − Δ_hp**, where Δ_hp is the double-double row of `hp/data/delta_all.csv` (err 10⁻¹⁶ to
  2 × 10⁻¹⁵). Each difference is taken in arb from Δ_RT's 30-digit string. It is a measured difference between two
  discretizations, not a bound.
- **The fit** (`rt_trunc_fit.py`) fits ln|δ| by three models: E, a + bD (exponential); P, a + b ln D (power); and Q,
  a + bD + cD². It uses four windows, [20, 60], [30, 60], [40, 60] and [50, 60], each on the points at and above the
  last sign change of δ. The spread of the 12 extrapolations is an estimator range, not an error bar. E and P on
  [30, 60] and [40, 60] are the primary set.
- **EEG's departure** d_EEG = Δ_EEG − Δ_ours is taken from `hp/data/delta_published_compare.csv` and averaged over the
  windows 55–65, 90–110, 140–160 and 180–200. The ± of each mean is its standard error from the scatter of the
  six-decimal rounding, not a bound.
- **The verdict at each D.** δ "reproduces" d_EEG when it is positive and overlaps [d_EEG/3, 3 d_EEG]. It is "far
  below" when its upper end lies under d_EEG/10, which includes any δ of the opposite sign, and "far above" when its
  lower end exceeds 10 d_EEG.

The continuation's step from 21.91 to 23.04 did not converge. After the first Newton step the steps were halved to
1/128, and max|F| crept from 5 × 10⁻⁸ to 10⁻⁶ over 20 steps. The predictor ran through the resampled 32 × 81 state at
D = 19, which is not a root of 40 × 79. Solved directly from #105's 32 × 101 state at D = 23 (`diag/`), the 40 × 79 root
at 23.043278 converged in 9 steps and lies 1.29 × 10⁻⁹ below hp. So the root exists, and the failure was the start. The
continuation then restarted from that root with only 40 × 79 roots in its history (`run_cont2.sh`). Every later step
converged in 4 to 8 Newton steps.

### 12.3 Δ at their grid, D = 20, 40 and 60

δ = Δ(40 × 79) − Δ_hp. Every root converged to max|F| below 10⁻⁴⁰. The two starts are A, the continuation, and B, the
32 × 101 state of #105 or #145 at that D, resampled.

| D | δ (A) | δ (B) | A − B | control (grid) | control − Δ_hp |
|---|---|---|---|---|---|
| 20.043653 | −5.56×10⁻¹⁰ | −5.56×10⁻¹⁰ | 0 in 30 figures | 32 × 121 | +1.7×10⁻²⁰ |
| 40.020065 | −7.13×10⁻⁹ | −7.13×10⁻⁹ | 4.9×10⁻²⁷ | 32 × 121 | +1.5×10⁻¹⁵ |
| 60.001952 | +5.80×10⁻⁸ | +5.80×10⁻⁸ | 9.9×10⁻²¹ | 32 × 101 | −2.3×10⁻¹² |

A third start at D = 60, #91's map of hp's own up13 state solved on 40 × 79 (`res/M_40x79_D60.0020`), lands 1.6×10⁻²⁰
from A. At D = 40 the same kind of start (`M_40x79_D40.0201`) did not converge in 20 Newton steps (Δ 8.4×10⁻⁴ off). So
the 40 × 79 roots are unique among the starts that converged, to 2×10⁻²⁰ or better. That is far below δ, unlike #145's
finer grids. The controls hold: all three lie within 10⁻⁹ of hp, and they repeat #105's and #145's values. The
coefficient mutation on 40 × 79 at D = 20.04 moves Δ by −1.14×10⁻⁴ and exits 1. On 32 × 79, Δ is the same as on 40 × 79
to all 20 printed digits at D = 40.02 and to 5×10⁻¹⁸ at 60.00, so the t resolution plays no part at this N.

### 12.4 Along the path, and on the neighbouring grids

The continuation gives δ on 41 hp rows from 20.04 to 60.00 (`fit/trunc_fit.log`). The neighbouring grids are re-solves
from the 40 × 79 roots at the same D (`res/`).

| D | 40 × 75 | 40 × 77 | 40 × 79 | 40 × 81 | 40 × 83 |
|---|---|---|---|---|---|
| 20.04 | −7.5×10⁻⁹ | +2.1×10⁻⁹ | −5.6×10⁻¹⁰ | +1.4×10⁻¹⁰ | −3.3×10⁻¹¹ |
| 25.08 | | +7.1×10⁻⁹ | −2.0×10⁻⁹ | +5.4×10⁻¹⁰ | |
| 29.98 | | +1.4×10⁻⁸ | −4.3×10⁻⁹ | +1.2×10⁻⁹ | |
| 34.99 | | +1.9×10⁻⁸ | −6.7×10⁻⁹ | +2.1×10⁻⁹ | |
| 40.02 | +1.9×10⁻⁹ | +1.4×10⁻⁸ | −7.1×10⁻⁹ | +2.6×10⁻⁹ | −8.3×10⁻¹⁰ |
| 44.96 | | −1.1×10⁻⁸ | −3.3×10⁻⁹ | +2.3×10⁻⁹ | |
| 50.01 | | −6.7×10⁻⁸ | +7.5×10⁻⁹ | +4.9×10⁻¹⁰ | |
| 55.00 | | −1.6×10⁻⁷ | +2.7×10⁻⁸ | −3.4×10⁻⁹ | |
| 60.00 | +1.4×10⁻⁶ | −3.0×10⁻⁷ | +5.8×10⁻⁸ | −9.9×10⁻⁹ | +1.3×10⁻⁹ |

Each column is smooth in D and crosses zero once in this range. Its zero moves up with N: between 20 and 40 for 75, near
42 for 77, at 46.5 for 79 (−1.75×10⁻⁹ at 46.00, +7.4×10⁻¹¹ at 47.01) and between 50 and 55 for 81. Away from these
zeros the sign alternates from one grid to the next, and the size falls by a factor of 3 to 8 per step of 2 in N. So
the sign of the error at N = 79 and a given D is set by where that D falls relative to the zero of the column. The size
is the robust part. From 40 × 77 to 40 × 81 it runs from 2×10⁻⁹ to 10⁻¹⁰ at D = 20, from 2×10⁻⁸ to 10⁻⁹ at 30–45, and
from 3×10⁻⁷ to 10⁻⁸ at 60. A Galerkin projection at the same degree, like EEG's, has its own zeros, so the collocation
measures the scale of their truncation, not its sign.

### 12.5 The growth in D: the criteria's fit, and δ itself

**The fit as written** (criteria 3 and 6, `fit/trunc_fit.log`). Only the points at and above the last sign change enter,
so all four windows reduce to the 14 points from D = 47.0 to 60.0, or 11 for [50, 60]. The fits of ln|δ| there measure
the climb out of the zero at 46.5, not the growth: E and P on the 14 points leave an rms of 0.9 in ln|δ|. Their
extrapolations to D = 100 run from 6×10⁻⁴⁵ to 0.16, and to D = 190 from below 10⁻⁸⁰ to 10¹³.

| D | primary set (E, P on [30, 60], [40, 60]) | all 12 fits | d_EEG | verdict (primary) |
|---|---|---|---|---|
| 60 | +5.80×10⁻⁸ (measured) | | +2.1×10⁻⁷ ± 0.8×10⁻⁷ | between |
| 100 | +2.0×10⁻³ to +1.6×10⁻¹ | +5.8×10⁻⁴⁵ to +1.6×10⁻¹ | +9.4×10⁻⁷ ± 0.8×10⁻⁷ | far above |
| 150 | +4.6 to +7.6×10⁶ | | +2.8×10⁻⁶ ± 0.8×10⁻⁷ | far above |
| 190 | +4.2×10² to +1.0×10¹³ | | +4.1×10⁻⁶ ± 0.9×10⁻⁷ | far above |

By the rule as written the verdict is "not decided". The ranges are a spread over fit choices, not error bars. The ± of
d_EEG is the standard error of a window mean from EEG's six-decimal rounding.

**δ itself, outside the criteria.** δ is smooth in D through its zero. Polynomials in D of degree 2 and 3 on [40, 60],
[47, 60] and [50, 60] fit it to an rms of 5×10⁻¹⁰ down to 2×10⁻¹² (`fit/signed_prediction.log`). Degree 1 fits it only
to 2×10⁻⁹ to 6×10⁻⁹. Their extrapolations, written into the raw README before the extension past D = 60 had produced any
data (raw 686cece), are these:

| D | 70 | 80 | 90 | 100 | 150 | 190 |
|---|---|---|---|---|---|---|
| δ, degree 2–3 | 1.4–1.6×10⁻⁷ | 2.7–3.3×10⁻⁷ | 4.2–5.9×10⁻⁷ | 6.2–9.5×10⁻⁷ | 2.1–4.9×10⁻⁶ | 4.0–11.5×10⁻⁶ |
| d_EEG | | | | 9.4×10⁻⁷ | 2.8×10⁻⁶ | 4.1×10⁻⁶ |

The test written with them, which can fail: the measured δ at 70, 80, 90 and 100 must lie within [½ × the lower end, 2 ×
the upper end].

### 12.6 Past D = 60 (outside the criteria)

`run_ext.sh` continued 40 × 79 from D = 60 in steps of 2 on hp's rows. Those rows are dd to 66, pinned at 68–72 and
chain_fit from 74, with stated errors of 1.2×10⁻¹³ or less to D = 84. Newton's target was 10⁻³⁶, with stalls accepted
below 10⁻³⁰. The extension was stopped at D = 84 (12.8). The grids 40 × 77 and 40 × 81 were re-solved from its roots at
D = 70 and 80. At D = 80 the 40 × 81 root, resampled back to 40 × 79, is a second start (`res/R_40x79_from81_D80`). It
lands 7.9×10⁻¹¹ from the continuation's root, which is 0.03% of δ. The last column is EEG's departure averaged over
their rows within D ± 5, with the standard error of that mean.

| D | δ (40 × 79) | predicted, degree 2–3 (12.5) | criteria's log fits (primary) | 40 × 77 | 40 × 81 | d_EEG (D ± 5) | δ / d_EEG |
|---|---|---|---|---|---|---|---|
| 62.01 | +7.38×10⁻⁸ | +5.2 to +7.4×10⁻⁸ | +2.2 to +2.5×10⁻⁷ | | | +1.7×10⁻⁷ ± 0.8×10⁻⁷ | 0.44 |
| 64.01 | +9.15×10⁻⁸ | +5.8 to +9.2×10⁻⁸ | +4.0 to +5.0×10⁻⁷ | | | +1.9×10⁻⁷ ± 0.8×10⁻⁷ | 0.49 |
| 66.00 | +1.11×10⁻⁷ | +6.5 to +11.3×10⁻⁸ | +0.7 to +1.0×10⁻⁶ | | | +1.7×10⁻⁷ ± 0.7×10⁻⁷ | 0.64 |
| 68 | +1.33×10⁻⁷ | +7.1 to +13.6×10⁻⁸ | +1.3 to +2.0×10⁻⁶ | | | +1.7×10⁻⁷ ± 0.7×10⁻⁷ | 0.76 |
| 70 | +1.57×10⁻⁷ | +1.43 to +1.61×10⁻⁷ (registered) | +2.2 to +4.1×10⁻⁶ | −7.20×10⁻⁷ | −3.19×10⁻⁸ | +1.2×10⁻⁷ ± 0.6×10⁻⁷ | 1.35 |
| 72 | +1.83×10⁻⁷ | +0.84 to +1.90×10⁻⁷ | +3.7 to +8.3×10⁻⁶ | | | +2.2×10⁻⁷ ± 0.6×10⁻⁷ | 0.84 |
| 74 | +2.11×10⁻⁷ | +0.90 to +2.21×10⁻⁷ | +0.6 to +1.7×10⁻⁵ | | | +3.1×10⁻⁷ ± 0.8×10⁻⁷ | 0.69 |
| 76 | +2.41×10⁻⁷ | +0.96 to +2.55×10⁻⁷ | +1.1 to +3.4×10⁻⁵ | | | +2.9×10⁻⁷ ± 0.8×10⁻⁷ | 0.82 |
| 78 | +2.74×10⁻⁷ | +1.03 to +2.93×10⁻⁷ | +1.7 to +6.9×10⁻⁵ | | | +3.1×10⁻⁷ ± 0.8×10⁻⁷ | 0.89 |
| 80 | +3.08×10⁻⁷ | +2.65 to +3.33×10⁻⁷ (registered) | +2.8 to +14×10⁻⁵ | −1.34×10⁻⁶ | −6.70×10⁻⁸ | +4.4×10⁻⁷ ± 1.0×10⁻⁷ | 0.70 |
| 82 | +3.44×10⁻⁷ | +1.15 to +3.77×10⁻⁷ | +0.4 to +2.9×10⁻⁴ | | | +4.7×10⁻⁷ ± 0.9×10⁻⁷ | 0.73 |
| 84 | +3.82×10⁻⁷ | +1.22 to +4.25×10⁻⁷ | +0.7 to +5.8×10⁻⁴ | | | +5.3×10⁻⁷ ± 0.9×10⁻⁷ | 0.72 |

The registered ranges are those of the raw README at D = 70 and 80, from degree 2 and 3 only. The other rows of that
column come from `rt_trunc_fit.py --beyond`, whose range also includes degree 1 (`fit/beyond.log`). All ranges are a
spread over fit choices, not error bars.

**The registered test passes at every D it reached.** δ at D = 70 and 80 lies inside the predicted ranges themselves,
not only inside the factor-2 band. The measured δ follows the top of the polynomial range, the cubic fits, to within
8%. The criteria's log fits (primary set) overshoot by a factor of 3 at D = 62 and 90 to 450 at 80.

**Against EEG's departure.** From D = 66 to 84, δ is 0.64 to 1.35 times their departure from our values, 0.8 on average.
Over the same range their departure grows from +1.7×10⁻⁷ to +5.3×10⁻⁷. By criterion 6's rule, applied here outside its
pre-registered use, δ(80) = +3.1×10⁻⁷ against +4.4×10⁻⁷ ± 1.0×10⁻⁷ "reproduces" it. At D = 60 the measured +5.8×10⁻⁸
against +2.1×10⁻⁷ ± 0.8×10⁻⁷ is "between", and EEG's departure there is barely resolved above their rounding. The
registered polynomial range carried on to D = 100, 150 and 190 is an extrapolation, a spread over fit choices. By the
same rule it "reproduces" EEG's departure there: 6.2–9.5×10⁻⁷ against 9.4×10⁻⁷, 2.1–4.9×10⁻⁶ against 2.8×10⁻⁶, and
4.0–11.5×10⁻⁶ against 4.1×10⁻⁶.

What this does and does not show:

- **It shows** that at EEG's degree and frequencies, a spectral discretization of these equations misses our Δ by the
  size of their departure, with the same growth from D ≈ 60. Truncation at their resolution is therefore large enough
  to account for the departure. It also grows like it.
- **It does not show** that their departure is that truncation. Their projection differs. The collocation's error
  alternates in sign between neighbouring grids, and at D = 80 the factor runs from 4 above (40 × 77, −1.3×10⁻⁶) to 5
  below (40 × 81, −6.7×10⁻⁸). The agreement in sign at N = 79 is therefore not evidence about the sign of their error.
- **The opposite case is excluded.** Truncation is not far below their departure at their resolution. That was the
  finding the issue asked to report if it held, and it does not hold.

### 12.7 Controls

| control | expected | result | log |
|---|---|---|---|
| 32 × 121 at D = 20.043653, from #105's 32 × 101 state | within 10⁻⁹ of hp, exit 0 | +1.68×10⁻²⁰, exit 0 | `controls/C_32x121_D20.0437*` |
| 32 × 121 at D = 40.020065, from #145's 32 × 101 state | within 10⁻⁹ of hp, exit 0 | +1.54×10⁻¹⁵, exit 0 | `controls/C_32x121_D40.0201*` |
| 32 × 101 at D = 60.001952, from #145's 32 × 121 root | within 10⁻⁹ of hp, exit 0 | −2.33×10⁻¹², exit 0 | `controls/C_32x101_D60.0020*` |
| `rt_run.py --mutate=coef` on 40 × 79 at D = 20.04, from root A | Δ moves by > 10⁻⁶, exit 1 | −1.14×10⁻⁴, exit 1 | `controls/coef_40x79_D20.0437*` |
| `rt_trunc_fit.py --planted` (rates, extrapolation, window means, verdict classes, signed fit) | exit 0 | exit 0 | `fit/planted.log` |
| the same with `--mutate=rate` | exit 1 | exit 1 | `fit/planted_mutate_rate.log` |
| 32 × 79 against 40 × 79 at D = 40.02 and 60.00 | (t resolution) | equal to 20 digits, 5×10⁻¹⁸ | `res/Nt_32x79_*` |

The three controls repeat #105's and #145's values (1.7×10⁻²⁰, 1.5×10⁻¹⁵, and #145's second root at 60, −2.33×10⁻¹²).
The planted test's first run exited 1 because its own expectation was wrong (`fit/planted_first_run.log`). It had a
negative δ planted as "between", while the rule as written makes it "far below". The planted case was corrected and
the rule left as it was, before any real data were fitted.

### 12.8 Discrepancies (reported, not fixed)

- **The criteria's fit cannot answer the question.** It fits ln|δ| on the points past the last sign change, and δ on
  40 × 79 crosses zero at D = 46.5. So it fits the climb out of a zero, and its extrapolations at D = 190 run from below
  10⁻⁸⁰ to 10¹³. The verdict as written is "not decided". The analysis of δ itself (12.5, 12.6) was added after seeing
  the data. Its prediction for D > 60 was written down before the extension ran.
- **The continuation's first leg** failed at D = 23.04. Its cubic predictor ran through a 32 × 81 state, resampled,
  that is not a 40 × 79 root. The root exists (12.2). A predictor's history should hold roots of the grid it predicts
  on. `run_cont2.sh` did that from then on.
- **Section 9.7's expectation** had N ≈ 80's error growing about ninefold per 10 units of D past 30, which would reach
  EEG's departure between D = 50 and 60. At N = 79 it grows from 4.3×10⁻⁹ at D = 30 to 7.1×10⁻⁹ at 40, crosses zero at
  46.5, and reaches +5.8×10⁻⁸ at 60. At N = 77 it is 3.0×10⁻⁷ at 60. 32 × 81's ninefold growth from 20 to 30 was one
  N's curve on its way to a zero, not a law in D.
- **The issue's control** names N = 121 at N_t = 40. The coordinator's brief replaced it with 32 × 121 at D 20 and 40
  and 32 × 101 at 60, and that is what ran. N_t does not matter at this precision (12.3; #105, #145).
- **EEG's bump at D 21–27.** Their table lies 1.4×10⁻⁵ to 2×10⁻³ above ours at D = 21–22.5 and 23.5–27, and back within
  10⁻⁶ at 23.0 and from 27.5 (`hp/data/delta_published_compare.csv`). At their resolution in collocation, the root near
  ours is smooth through D = 21–27. δ goes from −7.5×10⁻¹⁰ to −3.3×10⁻⁹ there, and every step converged once the
  predictor held only 40 × 79 roots. So the bump is not a property of the resolution as collocation sees it. A second
  root of their Galerkin system there was not looked for.
- **D = 100 straight from #91's map of the chain state** did not converge in 20 Newton steps (`res/X_40x79_D100`).
  After one step to max|F| = 8×10⁻¹⁴, Newton jumped 6.7×10⁻⁵ along the weak direction and ended 3×10⁻² off. The
  continuation reached D = 84 instead (12.6).
- **The extension stopped at D = 84, not 100.** The shared Mac's load (other agents' jobs, macOS media analysis and
  indexing) slowed a Newton step from about 1 min to 5–10 min. The step from 84 to 86 ran for about 40 min without
  finishing, against 11–15 min for the steps from 78 to 84. It was
  stopped at 19:40 UTC with its states saved (`cont/cont_40x79_D84.0000_40x79.npz`), so `run_ext.sh` resumes from there.
  The second start at D = 80 from the 40 × 77 root was stopped with it, unconverged.

### 12.9 Proposed text for the README

For the coordinator, after the RT sentences of "The large-D wall" (sections 7, 9.8 and 10.9), and next to the
comparison with Ecker, Ecker and Grumiller's table:

> At the resolution of Ecker, Ecker and Grumiller (40 Fourier and 80 Chebyshev modes), the RT prototype's Δ departs
> from ours by −6×10⁻¹⁰ at D = 20, −7×10⁻⁹ at 40 and +6×10⁻⁸ at 60. From D = 66 to 84 it is 0.6 to 1.4 times
> their published departure from our values (+4×10⁻⁷ near D = 80), and it grows the same way. So truncation at
> their resolution has the size of their departure. The prototype collocates where they truncate coefficients, and the
> sign of its error alternates between neighbouring grids. This sets the scale of their truncation, not its value
> ([notes/RT_FORMULATION.md](notes/RT_FORMULATION.md) section 12).

### 12.10 Reproduction

From the repository root, with the raw-data repository cloned beside it (RAW = `../choptuik-large-d-raw`). On the Mac,
use `/Users/nathan/venv-qd/bin/python` with the MULTIDOUBLE_SITE of 9.9. The drivers in `RAW/I106-mac/` hold the exact
commands of the runs. `run_cont2.sh` is the continuation with its restarts, and `run_ext.sh` takes it past D = 60.
`run_worker*.sh` hold the re-solves, `run_diag.sh` the D = 23.04 diagnosis, and the folder's README their order.

```
python3 solver/rt/rt_trunc_fit.py --planted                 # exit 0; --mutate=rate exits 1
python3 solver/rt/rt_run.py --start RAW/I105-mac/cont/cont_32x101_D20.0437_32x101.npz --grids 40x79 --solver mp \
    --max_products 250 --refresh_dz 1e-5 --patient --it_max 20 --full_below 1e-4 \
    --newton_tol 1e-40 --tol 1e-4 --out OUT --tag b20      # 12.3: start B at D 20.04, delta -5.56e-10, exit 0
python3 solver/rt/rt_run.py --start RAW/I106-mac/cont/cont_40x79_D20.0437_40x79.npz --grids 40x79 --solver mp \
    --max_products 250 --refresh_dz 1e-5 --patient --it_max 20 --full_below 1e-4 \
    --newton_tol 1e-40 --tol 1e-4 --mutate=coef --out OUT --tag coef   # 12.7: Delta moves by -1.14e-4, exit 1
python3 solver/rt/rt_run.py --start RAW/I145-mac/fol/fol_32x121_D60.0020_32x121.npz --grids 32x101 --solver mp \
    --max_products 250 --refresh_dz 1e-5 --patient --it_max 20 --full_below 1e-4 \
    --newton_tol 1e-36 --tol 1e-9 --out OUT --tag c60      # 12.3: the control at D 60, -2.33e-12, exit 0
python3 solver/rt/rt_trunc_fit.py RAW/I106-mac/cont/cont_40x79.jsonl --second \
    RAW/I106-mac/res/B_40x79_D20.0437.jsonl,RAW/I106-mac/res/B_40x79_D40.0201.jsonl,\
RAW/I106-mac/res/B_40x79_D60.0020.jsonl,RAW/I106-mac/res/M_40x79_D60.0020.jsonl
                                                            # 12.3 and 12.5 (fit/trunc_fit.log), exit 0
python3 solver/rt/rt_trunc_fit.py RAW/I106-mac/cont/cont_40x79.jsonl --at 70,80,90,100,150,190 \
    --beyond RAW/I106-mac/cont/ext_40x79.jsonl              # 12.5 and 12.6: delta past D = 60 against the fits
```

## 13. Horizon rows: Taylor conditions at ξ = −1 in place of collocation

Issue #163. Subagent of the coordinator on the Mac, 2026-11-08. Code in `solver/rt/`:

- the option `horizon=K` of `rt_model.py` and `rt_mp.py`, used through `rt_run.py --horizon K`, `--mutate=norows` and
  `--resample_mp`;
- the new `rt_horizon.py`, `rt_resample_res.py`, `rt_pair_fine.py`, `rt_sigma.py`, `rt_branch_cheb.py` and
  `rt_constraint.py`;
- `rt_grids.py --tol_ab`, and `rt_roots.py`, which now evaluates the residual that a state was solved with.

Raw output is in folder `I163-mac/` of the raw-data repository. Its README holds the criteria, written before the main
runs (raw a38ea6c). python-flint 0.9.0, sympy 1.14.0, mpmath 1.3.0.

**Result. The acceptance fails.** The new rows replace the collocation at the K points next to ξ = −1 by the Taylor
coefficients of orders 1 to K of the equations there. With K = 7, at D = 50.007491:

- 32 × 121 lands −1.86×10⁻¹² and −1.97×10⁻¹² from hp from the two starts, and 32 × 123 lands −4.29×10⁻¹² and
  −4.14×10⁻¹². The acceptance asks for 10⁻¹⁴ from hp and 10⁻¹⁸ between the starts.
- The control without the rows misses by −2.89×10⁻¹² and −2.90×10⁻¹² (32 × 121) and −5.74×10⁻¹² and −5.73×10⁻¹²
  (32 × 123), so it fires.
- The rows remove a quarter to a third of the miss. They widen the spread of the two starts from 9×10⁻¹⁵ to
  1.1–1.4×10⁻¹³.

At D = 60.001952 the rows remove the pair of 32 × 93 and leave two roots on 95, 97 and 99. There they are 7 to 12
times farther apart on 95 and 99 than without the rows, and about as far apart on 97.

**What the rows exclude, and what they do not.**

- The sympy derivation (13.1) gives two bounded horizon branches: ω₂'s, with exponent λ₂ = 5.3 to 5.9 at D = 40–60,
  and the scalar's, with λ₄ = 21 to 31.
- K = 7 excludes λ₂'s branch for every harmonic. The misses survive that, so they are not λ₂'s.
- λ₄'s branch would need rows of order 27 to 32. The f64-LU preconditioner fails already at order 30.

**What the weak directions are.** Two kinds were measured (13.7).

- Newton's jumps between nearby roots go along a smooth direction: μ plus harmonic 0 of ω₂, ω₃ and ω₄, the same as
  the pair differences of #145 and #164. Along it the Jacobian's gain is at most 10⁻²⁴ to 10⁻²¹ at D = 50.
- Rough right-hand sides find gains of at most 9×10⁻³⁵ to 2×10⁻³³ on 32 × 101 to 32 × 141. Those directions are
  dominated by ω₄ near ξ = 0, at the frequencies 3/2 and 5/2. On 32 × 101 the bound is the same at D = 40 and D = 50
  (1.7×10⁻³³ and 1.5×10⁻³³). A polynomial copy of λ₄'s branch would change by about six decades between those D.

Neither kind is shown to be the horizon branch that the issue blamed. The issue's premise is not confirmed, and the
origin of both directions is open (Recommended tasks of #163).

### 13.1 The horizon's branches, symbolically

`rt_horizon.py symbolic` (sympy, about 1 min, exit 0) starts from EEG's Eq. (5) as `rt_eqs_check.py` transcribes and
checks it. In the parity-extended form the collocated equations are Ω₂,₃,₄⁺/ξ, in which D⁺ acts on the unknowns
ω_i⁻. At ξ = −1, D⁺ = ∂_t + μ s ∂_s with s = 1 + ξ, so each equation reads D⁺ω_i⁻ + G_i(ω⁻, ω⁺, ω₁). The script
derives A = ∂G/∂ω⁻ at s = 0. It finds A lower triangular in the order (ω₂, ω₄, ω₃), with the diagonal

    a₂ = ((D − 3)(ω₂⁻ − ω₁) − (D − 1) ω₂⁺)/2,   a₄ = −((D − 4) μ + (D − 2) ω₂⁺)/2,   a₃ = μ,

all at ξ = −1. A Frobenius solution of the linearization, ω_j⁻ ~ s^λ v(t), therefore has

    λ_j = −(⟨a_j⟩ + i k)/μ   for harmonic k,   v = e^{ikt} exp(−∫ (a_j − ⟨a_j⟩) dt),

with ⟨·⟩ the mean over t. For ω₃ this gives λ₃ = −1 − ik/μ, which is unbounded and so excluded by any polynomial.
At order n of the Taylor series in s, the operator on the n-th coefficient is ∂_t + μn + A (checked on a series to
order 3). So the analytic solution is fixed order by order, except at a resonance n = λ_j, which needs k = 0. ω₁'s own
equation, (1 + P)Ω₁⁺/ξ = 2(∂_t + μξ∂_ξ + μ)ω₁ + …, is characteristic at the centre, not at the horizon.

The check compares A with rt_model's Jacobian entries at ξ = −1. Those entries are exactly A, since (1 + ξ) and the
diagonals of the Fourier matrices vanish there. On the D = 50.007491 root resampled to 16 × 41 they agree to
3.9×10⁻¹⁵ at all t. `--mutate=a4` (a₄ scaled by 1 + 10⁻³) gives 7.7×10⁻⁴ and exits 1 (`checks/symbolic*.log`).

The real parts at k = 0, from the solved states (`rt_horizon.py exponents`, `checks/exponents.log`, a measured value
at each state):

| D | 4 | 6 | 8 | 10 | 15 | 20 | 30 | 40 | 45 | 50.01 | 55.00 | 60.00 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| λ₂ | −0.41 | 0.25 | 1.08 | 1.78 | 3.02 | 3.82 | 4.77 | 5.32 | 5.52 | 5.68 | 5.82 | 5.94 |
| λ₄ | 0.63 | 2.19 | 3.59 | 4.88 | 7.85 | 10.63 | 15.94 | 21.12 | 23.68 | 26.23 | 28.76 | 31.30 |

- λ₄ is the scalar's branch. At D = 40 it is 21.1, the exponent notes/QUAD_DOUBLE.md 13.7 finds in our variables,
  and it grows by about 0.51 per unit of D.
- Section 4's slopes of σ_min against N at D = 4 to 10 (2 Re s + 1 = 1.8, 4.9, 7.7, 10.2) give Re s = 0.4, 2.0, 3.4
  and 4.6. These are close to λ₄ (0.63, 2.19, 3.59, 4.88) and not to λ₂.
- λ₂ is ω₂'s branch. It grows slowly and crosses the integers, where harmonic 0 is resonant: 4 at D = 21.5 (4.001),
  5 between D = 33 and 34 (4.965, 5.024), and 6 near D = 62 by extrapolation (`diag/exponents_crossings.log`).

### 13.2 The horizon rows

`--horizon K` replaces the collocated rows of Ω₂,₃,₄⁺/ξ at the K points next to ξ = −1 (j = N − 1, …, N − K) by the
Taylor coefficients of orders 1 to K of the same residuals at ξ = −1. The row at ξ = −1 itself is order 0.

- The coefficients come from those of the fields there by truncated series products. The fields' coefficients are the
  rows of Dⁿ/n! at ξ = −1, computed at 200 bits. So the residual stays exactly quadratic (third difference 7×10⁻⁵¹ in
  arb).
- Row n is scaled so that its largest entry is N², the size of a collocated derivative near the ends.
- An analytic solution satisfies every row. A branch s^λ with Re λ < K violates the row of order ⌈Re λ⌉, where its
  derivative is singular.
- A row of order n amplifies the rounding and the truncation of the regular part by about N²ⁿ. So K must stay small.
  K = 7 excludes λ₂ at every D from 40 to 62, for every harmonic (they share its real part), but not λ₄.
- `--horizon 0`, the default, is the collocation of sections 3 to 12. Its residual is bit-identical, checked in f64
  and arb.

The rows also amplify the f64 rounding of `rt_model.resample_to` (about 10⁻¹⁷) to a residual of about 10⁻¹¹. So every
run here resamples its start exactly in arb (`--resample_mp`, barycentric interpolation of the double-double state).

### 13.3 Planted checks

`rt_horizon.py planted` (exit 0, `checks/planted.log`), with its criteria in its docstring before the first run:

- (a) The manufactured state is analytic: Chebyshev degree ≤ 9 in ξ, harmonics ≤ 2 and 5/2 in t, μ = 0.13, D =
  50.007491, on 16 × 41. Its rows of orders 0 to 7 equal sympy's Taylor coefficients at ξ = −1 of Ω₂,₃,₄⁺/ξ of the
  closed forms, to 1.9×10⁻⁴⁷ in arb (criterion 10⁻³⁰). In f64, scaled as in the residual, they agree to 3.3×10⁻¹⁴ of
  N² (criterion 10⁻¹²).
- (b) newton_mp with the LUdd preconditioner solves the forced system at the D = 50 root, resampled to 16 × 41, with
  K = 7. It returns the planted root to 3.5×10⁻⁴⁰ in 6 Newton steps (criterion 10⁻³⁰ in 10).
- (c) The order-0 row equals the collocated row at ξ = −1 to 4.1×10⁻⁵⁶.
- `--mutate=taylor` scales one series coefficient by 1 + 10⁻³. Then (a) gives 7.8×10⁻⁴ and (c) 3.2×10⁻³, and the
  script exits 1.

The f64 criterion of (a) was revised once, after the first run. That run's criterion was a relative error of 10⁻⁹ on
the unscaled rows, and it gave 4.6×10⁻⁵. Those rows have entries up to 10¹³ at order 7, and they were built by
repeated f64 products, which lose 10⁻² there. The rows are now rounded from 200 bits, and the criterion is on the rows
as the residual scales them. The f64 rows only build the preconditioner.

### 13.4 The protocol

The criteria are in `I163-mac/README.md`, written at 21:45 UTC before the main runs and not changed since. The options
are #145's (`--solver mp --tol 1e-12 --max_products 250 --refresh_dz 1e-5 --patient --it_max 20 --full_below 1e-4`,
Newton's target 10⁻⁴⁰ at D = 50.007491 and 10⁻³⁶ at 60.001952), with K = 7 and `--resample_mp`.

- At D = 50.007491, start a is #145's 32 × 101 continuation state (+3.3×10⁻¹⁵ from hp). Start b is #145's 32 × 141
  root (−4.1×10⁻¹¹).
- At D = 60.001952, the starts are #164's: the 32 × 101 continuation state (−1.27×10⁻¹²) and #145's 32 × 121 root
  (+3.5×10⁻¹¹).
- hp is the double-double row of `hp/data/delta_all.csv` (up8 and up13, err 2×10⁻¹⁵).

Two pilots at D = 54.996085 on 32 × 121 came first (`pilot/`):

- K = 30, which would also pass λ₄ there (28.8): GMRES reached only 6.1×10⁻³ in 250 products at the first Newton step,
  and the run was stopped.
- K = 7 from #145's 32 × 101 state, resampled in f64: the start's residual is 1.9×10⁻¹¹, and Newton then moved by
  10⁻⁷ to 6.5×10⁻⁵ per step for six steps. It was stopped, and `--resample_mp` was added.
- K = 7 from #145's 32 × 121 root: one step of 1.8×10⁻²⁰, landing on that root (+8.54×10⁻¹²).
- K = 7 from the 32 × 101 state, resampled exactly: one jump of 3.4×10⁻¹¹, landing −4.38×10⁻¹⁰ from hp.

The README's predictions (P1, P2) were written after these pilots.

All 16 main solves converged, to a final max|F| of at most 6.4×10⁻⁴³ at D = 50 and 7.7×10⁻⁴⁰ at D = 60. Each took
1.5 to 7 min on one core and peaked at 2.0 GB or less.

### 13.5 The acceptance at D = 50.007491

Δ − Δ_hp from `rt_grids.py` (`fit/acc_*.log`). The difference a − b is taken from the 30-digit Δ strings.

| grid | N mod 4 | rows | start a | start b | a − b | verdict |
|---|---|---|---|---|---|---|
| 32 × 121 | 1 | K = 7 | −1.86×10⁻¹² | −1.97×10⁻¹² | +1.12×10⁻¹³ | criterion 1 fails |
| 32 × 121 | 1 | none (`--mutate=norows`) | −2.89×10⁻¹² | −2.90×10⁻¹² | +8.8×10⁻¹⁵ | control fires (> 10⁻¹²) |
| 32 × 123 | 3 | K = 7 | −4.29×10⁻¹² | −4.14×10⁻¹² | −1.43×10⁻¹³ | criterion 3 fails |
| 32 × 123 | 3 | none | −5.74×10⁻¹² | −5.73×10⁻¹² | −8.7×10⁻¹⁵ | control fires |

- Both families fail, with or without rows. The rows move each root by a quarter to a third of its miss toward hp,
  and they separate the two starts' roots by 1.1–1.4×10⁻¹³ in Δ instead of 9×10⁻¹⁵.
- The pair test (`fit/roots_acc_*.log`) finds both pairs at D = 50 hidden by the residual floor. The roots lie 7×10⁻¹⁵
  apart (rows) and 7×10⁻¹⁶ apart (none) in max|z₂ − z₁|. With the rows the stored states' floor is 5×10⁻²⁹ rather
  than 7×10⁻³¹, since the rows amplify the double-double rounding.
- Neither start found the root nearest hp. #145's 32 × 101 root, +3.3×10⁻¹⁵ from hp, is a root of the 32 × 121 system
  to the floor of its double-double storage (13.7). But from it, Newton on 32 × 121 jumps 2.3×10⁻¹³ (no rows) or
  1.5×10⁻¹³ (rows) and lands near −2.9×10⁻¹² or −1.9×10⁻¹².
- Prediction P1 failed. It had start a staying within 10⁻¹³ of +3.3×10⁻¹⁵.

### 13.6 The pairs at D = 60.001952

`rt_roots.py` on each pair (`fit/pair_*.log`). The values without rows are #164's roots, re-evaluated by this code,
and they repeat 11.4. max|B(e, e)|/max|e| is the strength of J along e at either root (max norms).

| N | N mod 4 | max\|z₂ − z₁\|, none | with K = 7 | max\|B(e,e)\|/max\|e\|, none | with K = 7 | Δ₂ − Δ₁, none | with K = 7 | verdict, none → K = 7 |
|---|---|---|---|---|---|---|---|---|
| 93 | 1 | 4.65×10⁻¹⁴ | 2.2×10⁻¹⁶ | 9.7×10⁻¹² | 1.3×10⁻¹⁴ | +4.62×10⁻¹³ | −3.6×10⁻¹⁵ | two roots → hidden by the floor |
| 95 | 3 | 7.7×10⁻¹⁵ | 5.2×10⁻¹⁴ | 2.3×10⁻¹² | 7.8×10⁻¹² | +5.3×10⁻¹⁴ | +8.32×10⁻¹³ | two roots → two roots |
| 97 | 1 | 7.0×10⁻¹⁴ | 8.0×10⁻¹⁴ | 2.0×10⁻¹³ | 7.5×10⁻¹² | +9.96×10⁻¹³ | +1.33×10⁻¹² | two roots → two roots |
| 99 | 3 | 4.9×10⁻¹⁵ | 5.7×10⁻¹⁴ | 1.5×10⁻¹³ | 7.5×10⁻¹² | +7.5×10⁻¹⁴ | +9.69×10⁻¹³ | two roots → two roots |

Δ − Δ_hp with the rows (`fit/pair_grids_*.log`):

- 93: +1.65×10⁻¹² from both starts.
- 95, 97, 99: start a gives −1.27×10⁻¹² on all three, which is the start's own Δ. Newton took one step of 4×10⁻²¹
  there, so the 32 × 101 root is a root of these three K = 7 systems.
- 95, 97, 99: start b gives −4.35×10⁻¹³, +5.9×10⁻¹⁴ and −2.99×10⁻¹³.

What the rows change at D = 60:

- They remove the pair on 93 only.
- The pairs of 95 and 99 (N ≡ 3 mod 4) grow 6.7 and 11.7 times apart. With the rows, all three remaining pairs are
  5–8×10⁻¹⁴ apart, so the family difference of 11.4 disappears.
- Prediction P2 had every factor below 10. It fails on 93 (a factor of 210, the other way) and on 99 (11.7).

### 13.7 Where the weak directions live

These diagnostics are outside the criteria (`diag/`). Every bound below is an upper bound on the Jacobian's smallest
gain in the max norm, not an SVD.

**Newton's jumps.** Each main solve starts at a root of another grid, which the first step brings to max|F| = 10⁻³⁵
to 10⁻²⁷. The next step is then 1.5×10⁻¹³ to 3.2×10⁻¹² long, and the step after that converges. A Newton step solves
J dz = −F to GMRES's 10⁻³⁰, so the gain of J along that step is at most |F|/|dz|:

- 6×10⁻²⁴ to 8×10⁻²² at D = 50, with or without the rows;
- 6×10⁻²¹ to 10⁻¹⁵ at D = 60 with the rows.

The direction of each jump is the one #145 and #164 found between pairs of roots: mostly μ, with harmonic 0 of ω₂, ω₃
and ω₄.

**The pair differences are the same on finer grids.** `rt_pair_fine.py` resamples both roots of a pair exactly and
evaluates J(z₁)e on finer grids.

- For #145's 32 × 121 pair at D = 50 it gives 4.60×10⁻¹¹ on 121, 141 and 181 points.
- For #164's 32 × 97 pair at D = 60 it gives 2.00×10⁻¹³ on 97, 141 and 181 points.
- Both roots keep their stored residual (8×10⁻³¹ and 2×10⁻³¹) on every finer grid.
- Each difference has Chebyshev coefficients that fall geometrically to the floor by m = 30, and no structure at
  ξ = ±1.
- The control `--mutate=f64` resamples in f64. It raises the residual on 141 to 2.8×10⁻¹³ and exits 1.

`rt_resample_res.py` gives the same for single roots. #145's 32 × 101 root at D = 50 and 60, and its 32 × 121 root at
D = 50, keep 2–8×10⁻³¹ on every finer grid. Its control `--mutate=perturb` exits 1.

**Smoothness does not exclude λ₄.** `rt_branch_cheb.py` (`diag/branch_cheb*.log`) prints the Chebyshev coefficients
of s^λ cos(ω ln s), relative to the maximum:

- s^5.68 (λ₂ at D = 50) falls only algebraically: 10⁻¹⁴ at m = 30, 2×10⁻¹⁸ at m = 60, 4×10⁻²² at m = 120.
- s^26.23 (λ₄ at D = 50, k = 0) falls through 9×10⁻²³ at m = 30, 2×10⁻⁴² at m = 60 and 4×10⁻⁵⁹ at m = 120. With
  k = 1/2 (ω = 3.87): 2×10⁻¹⁹, 5×10⁻³⁸ and 3×10⁻⁵⁵.
- Degree-N interpolants hold s^26.23 to 7×10⁻⁴³ at N = 61 and 5×10⁻⁵⁹ at N = 121. At D = 40 (λ₄ = 21.12), N = 101
  holds the branch to 1.0×10⁻⁴⁸ (k = 0) and
  4.6×10⁻⁴⁴ (k = 1/2), against 8.0×10⁻⁵⁵ and 6.5×10⁻⁵⁰ at D = 50, six decades apart
  (`diag/branch_cheb_D40.log`).

Up to m ≈ 60 the scalar's branch falls faster than the background itself, which reaches 4×10⁻³⁸ at m = 60. So a copy
of it in a difference vector cannot be seen in the coefficients, nor in residuals at the double-double floor. The
planted check (`--planted`, integer exponents) gives coefficients beyond the degree below 10⁻¹²⁰ and exits 0.

**The smallest gain, from J x = v.** `rt_sigma.py` solves J x = v in arb and bounds the smallest gain by 1/max|x|, for
smooth right-hand sides (low-degree fields) and rough ones (independent normal entries). It is not an inverse
iteration, since J is not normal. Its planted check, a diagonal operator with one entry 10⁻²², gives a bound of
1.4×10⁻²¹ and exits 0. Its control, that entry raised to 10⁻²⁰, gives 1.4×10⁻¹⁹ and exits 1
(`diag/sigma2_planted*.log`). Two starts of the rough bound agree within a factor 1 to 2.2, except on 32 × 121 with
the rows (7.1), where one start reached the cap of 250 products. The solve residual max|J x − v| is at most 4×10⁻¹⁴
with the rows and 5×10⁻¹⁸ without them.

| state | rows | bound, smooth v | bound, rough v (best of 2 starts) |
|---|---|---|---|
| #145's 32 × 101 root, D = 40.020065 | none | 5.9×10⁻¹⁸ (1 start) | 1.7×10⁻³³ |
| #145's 32 × 101 root, D = 50.007491 | none | 4.5×10⁻¹⁸ (1 start) | 1.5×10⁻³³ |
| this task's 32 × 121 root (start a), D = 50.007491 | none | 2.9×10⁻¹⁹ (best of 3) | 1.2×10⁻³⁴ |
| #145's 32 × 141 root, D = 50.007491 | none | 8.7×10⁻²⁰ (best of 3) | 9.2×10⁻³⁵ |
| this task's 32 × 121 root (start a), D = 50.007491 | K = 7 | 2.7×10⁻³³ (best of 3) | 4.9×10⁻³⁴ |

What x is in each case:

- With rough v, x is dominated by ω₄ near ξ = 0, at the frequencies 3/2 and 5/2. μ is 1% to 34% of its maximum, and
  ω₄ at ξ = ±1 is 10% to 40% of it.
- With smooth v and the rows, x is μ plus harmonic 0 of every field, the direction of the pairs and the jumps.

Reading these numbers:

- The weakest directions that rough v finds have bounds of 9×10⁻³⁵ to 2×10⁻³³. Two starts agree within a factor 2.2,
  so these bounds are probably near the smallest singular value.
- On 32 × 101 the bound does not change from D = 40 to 50. From N = 101 to 141 it falls by a factor 16. So these
  directions do not follow either horizon branch: a polynomial copy of λ₄'s would weaken by about six decades from
  D = 40 to 50 at N = 101, and the rows exclude λ₂'s. With the rows the rough bound on 32 × 121 is 4.9×10⁻³⁴, about
  what it is without them.
- Smooth v reach them only weakly (9×10⁻²⁰ to 6×10⁻¹⁸ without the rows), so their left vectors are rough on the
  grid.
- With the rows, smooth v do reach a direction of that strength, at 2.7×10⁻³³ (three starts within a factor 2.1),
  and it is the smooth direction of the pairs.

What these directions are is not identified here.

**The constraint.** Ω₅⁺, which the square system does not impose, differs between #145's two 32 × 121 roots by
2.8×10⁻²¹ (`rt_constraint.py`, `diag/constraint_D50.log`). So it does not tell them apart.

**Each grid's own root is stable.** Newton from a stored root on its own grid takes one step of 1–3×10⁻²⁴ and
stops. This holds for #145's 32 × 101 root at D = 50 (+3.4×10⁻¹⁵ from hp) and for this task's 32 × 121 root without
rows (`diag/again_*.stdout`). So the jumps belong to the points where one grid's root sits in another grid's system,
not to the roots themselves.

### 13.8 What this says about sections 10 and 11

- **Section 10's mechanism.** Section 10 reads the finer grids' error as truncation along a weak direction divided
  by σ_min, and that reading stands. The measured gains, 10⁻³⁴ to 10⁻²¹, lie far below the double-double floor of
  stored states. So a residual difference of 10⁻³⁴ between two grids, invisible at that floor, can move a root by
  10⁻¹³.
- **Section 10's cause.** Section 10 attributes the weak direction to the horizon's singular branch. That is not
  confirmed. The rows exclude λ₂'s branch without changing the outcome, λ₄'s branch cannot be told apart from smooth
  at these N, and the weakest directions measured (13.7) do not follow either exponent in D.
- **Section 11's N mod 4 families.** They depend on the collocation rows near ξ = −1. With K = 7 Taylor rows there,
  the difference between the families' pairs disappears at D = 60 (13.6).
- **Starts.** A root of one grid is a root of the next to about 10⁻³⁰. From it, Newton on the other grid jumps to a
  root 10⁻¹³ to 10⁻¹¹ away, so the re-solves of #145 and #164 measured those roots. #145's continuation on 32 × 101
  stays within 2×10⁻¹⁴ of hp to D = 55. The continuation and the hp chain agree there; the re-solves do not show
  whether this is a property of the 101 grid or of continuing along the path.

### 13.9 Discrepancies (reported, not fixed)

- **The acceptance of #163 fails.** At D = 50.007491, 32 × 121 with K = 7 lands 1.9×10⁻¹² from hp, against 10⁻¹⁴,
  and its two starts agree to 1.1×10⁻¹³, against 10⁻¹⁸. 32 × 123 does the same. The control fires.
- **Both predictions written before the main runs failed** (13.5, 13.6). P1 had start a staying at the 32 × 101 root.
  P2 had the rows changing every pair's separation by less than a factor 10.
- **The issue's premise** (#145, section 10.7: the horizon's singular branch makes the roots non-unique) is not
  confirmed (13.8).
- **My progress posts on #163 were wrong in two places.** The 21:52 UTC post read "exact in ξ" as "the weak
  direction is not the horizon's". That test only reaches the double-double floor, and λ₄'s branch is invisible there
  (13.7). The 21:22 and 21:52 posts also quoted J along #164's 32 × 97 pair as 1.7×10⁻¹² on finer grids. That value
  came from an f64 resampling of the root; resampled exactly, it is 2.0×10⁻¹³ on every grid.
- **The f64 resampling of `rt_model.resample_to`** leaves a residual of about 10⁻¹³ in the plain collocation and
  10⁻¹¹ with the rows. #145's and #164's re-solves started from it. With exact resampling, the landing roots repeat
  to 10⁻¹⁴ (32 × 121 at D = 50: −2.89×10⁻¹² here, −2.9×10⁻¹² in #145). The exception is #145's start from the
  32 × 141 root, which reached +4.5×10⁻¹² there and reaches −2.90×10⁻¹² here.
- **Two planted criteria were revised once each, after their first runs**: the f64 tolerance of `rt_horizon.py
  planted` (13.3), and `rt_sigma.py`'s planted window (a factor 3 at first, then 100, its docstring). The
  inverse-iteration version of `rt_sigma.py` ran first and gave meaningless second iterates, since J is not normal.
  Its logs (`diag/sigma_*.log`, code 2862224) are kept, and only their first solve, which is the smooth-v bound of the
  table, is used.
- **rt_grids' `--mutate=shift`** on the 32 × 121 acceptance exits 1, but so does the unmutated check, so here it shows
  nothing. That control was demonstrated in #145 and #164.

### 13.10 Proposed text for the README

For the coordinator, after section 11.7's text:

> Taylor conditions at the horizon in place of the nearest collocation rows (#163) exclude the branch of ω₂, whose
> exponent is 5.3 to 5.9 at D = 40–60. They do not restore uniqueness: at D = 50 on 32 × 121 the miss falls only from
> 2.9×10⁻¹² to 1.9×10⁻¹². The Jacobian there has directions with gains below 10⁻³³, which do not follow the horizon
> branches' exponents in D, so the cause of the non-uniqueness is open
> ([notes/RT_FORMULATION.md](notes/RT_FORMULATION.md) section 13).

### 13.11 Reproduction

From the repository root, with the raw-data repository cloned beside it (RAW = `../choptuik-large-d-raw`). On the
Mac, use `/Users/nathan/venv-qd/bin/python` with the MULTIDOUBLE_SITE of 9.9. The drivers in `RAW/I163-mac/` hold the
exact commands: `run_one.sh` (one solve), `queue_A.txt` to `queue_D.txt` (the main runs), `run_checks.sh`,
`run_verdicts.sh`, `chain_diag.sh` and `chain_sigma*.sh`.

```
python3 solver/rt/rt_horizon.py symbolic RAW/I145-mac/cont/cont_32x101_D50.0075_32x101.npz   # exit 0; --mutate=a4 exit 1
python3 solver/rt/rt_horizon.py planted --raw RAW            # exit 0; --mutate=taylor exits 1
python3 solver/rt/rt_run.py --start RAW/I145-mac/cont/cont_32x101_D50.0075_32x101.npz --grids 32x121 --solver mp \
    --tol 1e-12 --max_products 250 --refresh_dz 1e-5 --patient --it_max 20 --full_below 1e-4 --newton_tol 1e-40 \
    --horizon 7 --resample_mp --out OUT --tag a_32x121        # 13.5: -1.86e-12 from hp, exit 1 (about 7 min)
python3 solver/rt/rt_grids.py RAW/I163-mac/acc/a_32x121.jsonl RAW/I163-mac/acc/b_32x121.jsonl --D 50.007491 \
    --tol 1e-14 --tol_ab 1e-18                               # criterion 1: exit 1
python3 solver/rt/rt_roots.py RAW/I163-mac/pair/a_32x93_D60.0020_32x93.npz RAW/I163-mac/pair/b_32x93_D60.0020_32x93.npz
                                                             # 13.6: hidden by the floor
python3 solver/rt/rt_pair_fine.py RAW/I145-mac/fol/fol_32x121_D50.0075_32x121.npz \
    RAW/I145-mac/fol/from141_32x121_D50.0075_32x121.npz 121,141,181   # 13.7: 4.60e-11 on each; --mutate=f64 exit 1
python3 solver/rt/rt_branch_cheb.py --lam 5.68,26.23,26.23 --om 0,0,3.87   # 13.7; --planted exits 0
python3 solver/rt/rt_sigma.py RAW/I145-mac/cont/cont_32x101_D50.0075_32x101.npz --rough --starts 2
                                                             # 13.7: 1.5e-33 (about 4 min); --planted exits 0
```

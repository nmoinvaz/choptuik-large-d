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

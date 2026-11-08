# RT_FORMULATION: the Reiterer–Trubowitz formulation of arXiv:2611.07007, evaluated for the large-D chain

Issue #91. Subagent of the coordinator on the Mac, 2026-11-08. Code in [`solver/rt/`](../solver/rt/), raw output in
folder `I91-mac/` of the raw-data repository.

**Verdict: adopt in part.** Adopt the map between the gauges with its identity Δ = 4πμ/(1 − (n − 1)⟨a_h² − 1⟩), the
RT prototype as an independent check of Δ(D) at moderate D, and RT's quadratic framework for a certified solution at
one D. Do not adopt the global Fourier × Chebyshev Newton for the chain. It does not avoid our horizon wall. The
horizon's singular branch is nearly a polynomial, and the Chebyshev system's smallest singular value falls like a power
of the resolution whose exponent grows by about 1.4 per unit of D, so its arithmetic must lengthen with D as ours does,
on 14 to 24 times our unknowns in a dense LU. The quadratic form of the equations is not what sets the conditioning.

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
ω₄⁺ = √2 A⁺ c p / x, ω₄⁻ = −√2 A⁻ c m / x, ω₂^± = μ/ξ ∓ A^± c / x, ω₃^± = D^± ζ ∓ μ.

MAP_PLACEHOLDER

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

COND_ARB_PLACEHOLDER

**At matched resolution.** Matched in τ harmonics (our M = 32 against RT's N_t = 64) the comparison is RT's κ at any
N_t, since RT's does not depend on it. Matched in accuracy of Δ the two meet at D = 10: RT's 48 × 81 grid gives Δ to
1.1×10⁻¹² with κ = 1.7×10¹⁵, and our M = 48 state (Δ to about 10⁻¹⁶) has κ = 1.2×10¹³ (`hp/stall_margin.py`,
`I91-mac/ours_kappa.jsonl`). Ours then grows to 4.6×10¹⁹ at D = 20.04 (M = 32, every harmonic free) and 1.7×10³¹ at
D = 60 (M = 32, harmonics above 13 frozen, folder `T76/kappa_states.jsonl`), and the pinned system the chain solves
has LU pivots spanning 6.9×10¹⁹ at D = 100 and 5×10³⁹ at D = 200 (folders `T139-macm3/chain/` and
`T138-win2/chain_o/`, `anm_run.jsonl`: the ratio of the largest to the smallest LU pivot, an estimate of κ, not an
SVD).

COND_COMPARE_PLACEHOLDER

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

Q2_DATA_PLACEHOLDER

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

So one Newton step's LU at EEG's L = 11,200 takes about 4.2 hours at 512 bits and 11.5 hours at 1024 bits on one
core, and at L = 18,369 (our harmonic content, N = 81) 19 and 51 hours (`rt_summary.py`, estimates from the L³ law;
flint's blocking may do somewhat better at large L). EEG's 392 steps of 1/2 from 4 to 200, with "typically" four Newton iterations each, need that
LU about 1,600 times. Our chain's step at D = 200 took 460 s on 18 threads, mostly the 24 Taylor coefficients in
octo-double (T138-win2 `chain_o/anm_run.jsonl`: t_coef 415 s, the corrector 29 s, the re-seed 14 s). The precision
the RT system needs is set by section 4: Q3_PREC_PLACEHOLDER

## 5b. What to adopt, and what not

Adopt:

- **The map and its Δ–μ identity** (`gg_to_rt.py`, `mu_check.py`). They tie our horizon data to RT's geometric
  invariant. At D = 4 RT's enclosure of 2K (radius 2×10⁻⁸⁰) is a reference for our Δ in place of MGG2003's ±3×10⁻⁹:
  our M = 192 point agrees with it to 3.1×10⁻²⁶ (`mu_check.py`), against 2.3×10⁻¹⁰ with MGG2003. Proposed: add it to
  `hp/data/delta_published_compare.csv` as the D = 4 reference.
- **The RT prototype as an independent check of Δ(D) at moderate D.** It shares nothing with `hp/` but the first guess
  at D = 4 (another gauge, other unknowns, a global discretization, no series), and it reproduces Δ(10) to 1.1×10⁻¹² of
  ours. With a double-double or quad-double LU it can check our values at D = 12–30, where the published ones (EEG,
  six decimals) are far less precise than ours.
- **RT's framework for a certified solution at one D** (PROOF.md item 7). The quadratic structure is what makes the
  Newton–Kantorovich bounds of a computer-assisted proof simple, and RT's code did it at D = 4. Section 4 says what it
  would cost at larger D: the near-null direction makes the inverse's norm grow with D, so the proof's constants (and
  the precision of its arithmetic) grow as κ does.

Do not adopt: the global Fourier × Chebyshev Newton as the chain's solver. The horizon wall comes back as the
conditioning of the polynomial system (question 2, section 4), so its arithmetic must lengthen with D as ours does,
with 14 to 24 times our unknowns in a dense LU (question 3), and its truncation error at fixed resolution grows with D
(the prototype's on a fixed grid, section 3; EEG's six decimals leave ours by 3×10⁻⁶ to 4×10⁻⁶ at D = 150–200,
`hp/data/delta_published_compare.csv`, beyond their last digit, which is consistent with that and does not prove it).

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
  and variables in D dimensions (Eqs. (2)–(5)); "the equations are at most quadratic in the unknowns", "so the linearized
  operator depends linearly on the background solution", which "adds to the numerical stability of their scheme"
  (supplement VII.1); Δ = 4πμ/(1 − ⟨ψ₋₁²⟩) (Eq. (3)); Sec. III: Fourier × Chebyshev on [0, 4π] × [−1, 1] with 40 and 80
  modes, bilinear convolutions in spectral space, 11,200 unknowns, Newton in GMP dyadic arithmetic with rounding only
  at the Jacobian's inversion and "increasingly higher precision" with D, residuals near 10⁻²⁵ after about four steps,
  D from 4 to 200 in steps of 1/2 with a linear predictor; that "the singular branch becomes increasingly difficult to
  distinguish numerically from the regular one", which stopped their earlier shooting code at D = 5.5; "the results for
  D>28 have sufficient accuracy so that we trust all displayed digits", from varying the mode numbers; supplement
  VII.2 maps RT gauge to the gauge of their earlier work by integrating characteristics. Not stated: the precision used
  at large D, mode numbers other than 40 and 80, or an error bar on Δ. Their Eq. (5) holds for Ric = (D − 2) dφ dφ,
  their Eq. (1) has Ric = 2 dφ dφ (section 2 above).

## 7. Proposed text for the README ("The large-D wall")

A sentence for the bullet "The large-D wall" of "Limits and open problems", after its first sentence:

> Ecker, Ecker and Grumiller reach D = 200 with Reiterer and Trubowitz's formulation, quadratic first-order equations
> discretized globally by Fourier and Chebyshev series with no horizon series. It meets the same wall in another form:
> the horizon's singular branch, of exponent near (D − 4)/2, is nearly a polynomial, so the Chebyshev system's
> condition number grows README_RATE_PLACEHOLDER, and the arithmetic must lengthen with D as ours does
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
python3 solver/rt/rt_cond_mp.py OUT.jsonl --profile PROF.npz --grid 16x21 --prec 512
cd hp && python3 stall_margin.py OUT.jsonl STATE.npz --k 2 --threads 2      # OUT inside a raw-data folder
```

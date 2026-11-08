# RT_FORMULATION: the Reiterer–Trubowitz formulation of arXiv:2611.07007, evaluated for the large-D chain

Issue #91. Subagent of the coordinator on the Mac, 2026-11-08. Code in [`solver/rt/`](../solver/rt/), raw output in
folder `I91-mac/` of the raw-data repository.

**Verdict: adopt in part.** VERDICT_PLACEHOLDER

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
The unknowns are μ and seven null derivatives, which RT and EEG define in Eq. (8) and Eq. (4):

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

our form of EEG's Eq. (3) (⟨ψ₋₁²⟩ there). Numerical check: from our D = 4 double-double state at M = 192 (folder
`T45-win/hp/data/lowD_states/`), μ comes out 7.7×10⁻²⁸ from RT's enclosure 0.168307078963449969510134979042857…
(radius 10⁻⁸⁰), and Δ is 3.1×10⁻²⁶ from their 2K. The M = 160 state gives 8.4×10⁻²⁵ and 1.1×10⁻²² (numerical evidence;
the differences are our truncation in M, the enclosure's radius being negligible). The rest of the map follows from
the metric: ξ = (w⁺ − w⁻)/(w⁺ + w⁻), t = (4π/Δ)τ − ln((w⁺ + w⁻)/2)/μ, e^{−ζ} = μ e^{−τ} a ((w⁺ + w⁻)/2) / √(−w⁺_x w⁻_x),
and D^± = A^±(∂_τ + (x ± c)∂_x) with A^± = −μ(w⁺ + w⁻)/(2c w^∓_x), which gives the ω's from (a, c, p, m):
ω₄⁺ = √2 A⁺ c p / x, ω₄⁻ = −√2 A⁻ c m / x, ω₂^± = μ/ξ ∓ A^± c / x, ω₃^± = D^± ζ ∓ μ.

MAP_PLACEHOLDER

## 3. The prototype and its checks

PROTO_PLACEHOLDER

## 4. The Jacobian's condition number against D

COND_PLACEHOLDER

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

QUESTIONS_REST_PLACEHOLDER

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

README_PLACEHOLDER

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

# LIT_GAPS_DSS: how DSS critical solutions are constructed and proved, read against the large-D problem

Issue #1089, slice 3, for the large-D quest #1088. A subagent on the Mac wrote it on 2026-11-25 after reading
notes/LIT_REVIEW.md, notes/LARGE_D_WALL.md, notes/LARGE_D_BASICS.md, notes/LARGE_D_LITERATURE.md,
notes/RT_FORMULATION.md, notes/FACTORED_HORIZON.md and notes/QUAD_DOUBLE.md §12–13.

**How the papers were read.** Every paper in §1 was read from its arXiv e-print, the TeX source fetched with curl on
2026-11-25. The quotes are verbatim from the TeX, with the math rewritten in Unicode. The equation numbers were counted
in the TeX, so the published PDF could differ by an offset. Bibliographic data were checked against the arXiv API on
2026-11-25 (§6). No text in the sources is addressed to an AI agent.

**Scripts and raw data.** The checks are in [`solver/lit_gaps_dss/`](../solver/lit_gaps_dss/). Their logs are in the
raw-data repository, folder `I1089-dss-mac/` (commits c56eb797d and cb95b34d7). The status words are those of AGENTS.md;
*inferred* marks my own reasoning from a source.

## 0. Verdicts

| question | answer | status |
|---|---|---|
| Does a published construction avoid our 0.155n digits? | **Yes, in part.** Gundlach (1995, 1997, and EYM 1997) never integrates a in x; he rebuilds a at each x from the momentum constraint. In that form the mass mode x^−(n−1) does not exist. | checked symbolically, §2 |
| Is our mass mode physical? | No. It is the violation of the momentum constraint, which form A (ours, MGG 2003's) leaves free: x E_x = −(n − 1) a² E exactly | checked symbolically, §2 |
| Does the slaved form hold on our states? | yes, to ≤ 2.6×10⁻¹⁴ at D 68–263.6; its operator has (n − 1)‖L⁻¹‖ = 0.9987–1.0000 | numerical evidence, §2 |
| Is H1's horizon pole the momentum-constraint violation? | **yes**: the horizon series' radius follows the data's momentum residual alone. m_h moved by up to 10⁻² with a_h re-slaved leaves it at 0.40–0.48 (D 68), while a_h moved by 10⁻⁸ takes it from 0.585 to 0.214 | numerical evidence, §2.1 |
| Is there a published SSH condition that does not degrade as σ ≈ n/2 + 2.6 grows? | No. Every DSS construction read imposes an order-0 to order-3 condition (Gundlach, MGG, Hirschmann–Eardley), or none at all (Reiterer–Trubowitz's function space). The ODE works with large exponents (Bizoń–Glogić–Wasserman, Buckmaster–Cao-Labora–Gómez-Serrano) run the light-cone series to order ≈ exponent, as our horizon series does. | read, §1 |
| What does σ itself cost us? | about σ_R + 7 to 1.9σ_R series orders and 0.0087n digits at x_K = 0.7, none at F's crossover | numerical evidence (LARGE_D_BASICS §4–5, FACTORED_HORIZON M3) |
| Published D-trends our data contradict | Sorkin–Oren's minimum of Δ near D 12 and maximum of γ near 11; Bland et al.'s Δ → "just below 2.4" and γ → 0.466 | discrepancy, §3 |

**The decision this informs (#1088).** Track B's mass-mode factor can be had exactly, with no eikonal, by changing
which Einstein equation sets a: lead 1 of §5. What remains of order n is the scalar's fast mode F and the SSH exponent
σ. The series already handles σ at a cost polynomial in n.

## 1. What each paper does and does not contain

### 1.1 The DSS constructions

**Gundlach, PRL 75, 3214 (1995), gr-qc/9507054.** Choptuik's polar-areal metric with α(r = 0, t) = 1 (Eq. (2)), fields
X± = X ± Y, coordinates φ = 2πτ/Δ and ζ = ξ − ξ₀(τ) with a free periodic gauge function ξ₀.
- **Equations** (6)–(9): a_ζ, g_ζ = g(1 − a²), X±,ζ = B±/(1 ± D), and the constraint (12),
  z(2π/Δ) a_φ/a = ½[(1 − a²) + a²(X₊² + X₋²) + a²D⁻¹(X₊² − X₋²)].
- **Regularity.** The sonic line D = 1 is placed at ζ = 0 by the choice of ξ₀. "We can enforce analyticity simply by
  expanding in powers of ζ." At ζ = 0 the code imposes D = 1 and B₋ = 0, both order-0 conditions.
- **Method.** Pseudo-spectral Fourier in φ with 2N collocation points ("dealiasing turns out to be essential for
  stability"); one-sided shooting to tune ξ₀, two-sided shooting to ζ = −1, then box-scheme relaxation.
- **Where a comes from.** "a is not considered as independent, but reconstructed at each step from the other fields by
  solving equation (constraint)." Result: Δ = 3.4439 ± 0.0004.
- **Does not contain:** any D ≠ 4, any condition of order above 0 at the sonic line, any conditioning study.

**Gundlach, PRD 55, 695 (1997), gr-qc/9604019.** The full version.
- **Field equations** (5)–(8): r g_r = (1 − a²) g, r a_r = C₁ (Hamiltonian), g r a_t = C₂ = ½a³(X₊² − X₋²)
  (momentum), and r(X±,r ∓ g X±,t) = C±.
- **Eigenvalue form** (16)–(19), with the constraint (19), a_τ = e^{−(ζ+ξ₀)} g⁻¹ C₂ + (1 + ξ₀′) C₁. That is the momentum
  constraint with a_r replaced by the Hamiltonian right-hand side.
- **Regularity** (Sec. II C). The coordinate condition (24) and the numerator condition (25). The approximate equation
  (26) near ζ = 0 has the general solution X₋^inhom + ζ^{Â₀/B̂₀}(…)F, "analytic if and only if F vanishes identically".
  Then: "It is easy to see that both Â₀ and B̂₀ are negative in our case, so that the solution either blows up or is
  analytic. So with the one condition (25) we automatically enforce analyticity."
- **Where a comes from.** "The field a is not evolved in ζ, but reconstructed at each step from the constraint (19)."
  App. C: "it can be reduced to the linear form (C7) by the substitution f = a⁻²", solved in closed form by
  f = −e^{−I₀g} I_{ĝ₀}(e^{I₀g} h).
- **Perturbations** (Sec. IV, App. B). δa is rebuilt the same way, and "δa will match automatically by virtue of the
  linearized constraint". At ζ = 0 the coefficient of δa₀ in Eq. (B4) averages to λ + 1, so "for λ = −1, there are no
  perturbations that are regular on ζ = 0. In the λ plane this gives rise to a pole at λ = −1." A second pole lies at
  λ ≃ −0.385. App. E counts zeros minus poles with ∮A′/A.
- **Result.** Δ = 3.4453 ± 0.0005, with "a change in numerical details" explaining the move from 1995.
- **Does not contain:** D ≠ 4, a statement of what the condition (25) imposes when the exponent exceeds 1, any
  advantage claimed for rebuilding a (it is presented as the method, not compared).

**Gundlach, PRD 55, 6002 (1997), gr-qc/9610069 (Einstein–Yang–Mills).** "We choose to evolve only Π₊, Π₋ and g in ζ …
and to determine a and S at each new value of ζ from the constraints, eqns. (3) and (4)" (Eqs. (23)–(24) as counted).
- **A failure of the slaved form, in the perturbations.** Rebuilding δa from Eq. (54) "needs to divide the average of h
  by the average of g", and "As ζ → −∞, the average of g … is λ + 1 + O(e^{2ζ}) … this goes through zero at some small
  value of ζ". The result is "essentially random behavior for λ ≲ −1". The paper adds: "We have not found a simple way
  of fixing this problem, as our algorithm relies in an essential way on reconstructing S and a and δS and δa from
  the constraints at each ζ."
- **D-dependence** (*inferred*, from §2's operator): the same average is λ − (n − 1)⟨…⟩ in our sign. The trouble
  moves to Re λ ≈ n − 1, far from the physical mode near λ ≈ 2 once D ≫ 4.

**Martín-García and Gundlach, PRD 68, 024011 (2003), gr-qc/0304070.** Past patch in exactly our gauge: x = r/(−t),
τ = −ln(−t), f = α/a (our c), and the light cone at x = x_p = 1. "we do not follow the convention of [Choptuik] that
α = 1 at the center, which in [Gundlach 1997] forced us to introduce an additional free function of time".
- **Equations** (15)–(19):
  - f_x = (a² − 1)f/x;
  - (a⁻²)_x = [1 − (1 + U² + V²)a⁻²]/x (Hamiltonian);
  - (a⁻²)_τ = [((f + x)U² − (f − x)V²)/x + 1]a⁻² − 1 (momentum, with a_x eliminated);
  - the matter equations U_x and V_x.
- **Regularity** at the light cone, Eq. (20): V_τ − (1 − a²)V − U = 0.
- **Method** (Sec. IV A). Shoot from x_left = 0.001 and x_right = 0.9999 to x_mid = 0.01. Use a second-order power
  expansion at the light cone, "errors of order (x_p − x_right)³". IRK2 (order 4), N up to 512 Fourier points, Newton.
- **What a does.** The free data are f_c, Ψ_c and U_p only, so a on the light cone comes from Eq. (17) (*inferred* from
  that count), and a is then integrated inward with the Hamiltonian Eq. (16). That is form A of §2.
- **The error floor.** "errors from the expansions at the singular points cannot be brought down with our code below
  a limit which we estimate … between 10⁻⁸ and 10⁻⁹". Of the modes of U_p above k = 15: "they propagate inwards and
  get amplified".
- **Result.** Δ = 3.445452402(3) (Eq. (58)). The patches beyond the light cone are continued from the past patch with
  their own free data at the Cauchy horizon; continuation adds no condition on the past patch.
- **Does not contain:** D ≠ 4, the mass mode, any remark on why the Hamiltonian rather than Eq. (17) is integrated.

**Hirschmann and Eardley, PRD 51, 4198 (1995), gr-qc/9412066 (complex scalar, CSS).** The regular and irregular
solutions at the similarity horizon behave as f ~ (z − z₂)^{(iω+1)/κ} with κ = 1 − db/dz|_{z₂}, the "surface gravity"
(Eqs. (irreg a)–(irreg b) in the plain-TeX labels).
- **The difficulty.** "To enforce numerically the boundary condition on the horizon is difficult, because the
  irregular part of φ, which we want to annul, vanishes anyway if κ < 1, and the smaller κ, the faster it vanishes."
- **The fix and its limit.** "We handled this by computing analytically the second derivative of the regular solution
  at z = z₂, and using a second-order Taylor expansion … Even so we encountered numerical problems for κ ≲ 1/3", that
  is for an exponent near 3.
- **What this is** (*inferred*). The CSS precedent of our wall: a large exponent at the horizon. It reaches the
  exponent through 1/κ, where ours reaches it through the numerator n/2.

**Reiterer and Trubowitz, CMP 368, 143 (2019), arXiv:1203.3766.**
- **Method.** Fourier × Chebyshev in frequency space with weighted ℓ¹ norms. Coefficients decaying as
  O(κ₁^−|m| κ₂^−|n|) give functions analytic on a strip times "the open ellipse with radii ½(κ₂ ± 1/κ₂)" (Sec. 1). The
  ellipse contains ξ = 1, the critical light cone.
- **Regularity.** In frequency space "one does not specify initial data anywhere", and it "disposes of the geometric
  boundaries (central geodesic and especially the critical light cone) in a seamless way".
- **Result.** A computer-assisted proof at D = 4, with μ and K enclosed to 80 digits.
- **Does not contain:** D ≠ 4. In a truncation, analyticity at ξ = 1 rests on how fast the coefficients decay, and
  notes/RT_FORMULATION.md measured that this degrades with D: σ_min falls as a power of the resolution, with an
  exponent that grows by about 1.4 per unit of D.

### 1.2 Evolutions in D dimensions (no construction)

**Sorkin and Oren, PRD 71, 124005 (2005), hep-th/0502034.** Double-null evolutions for 4 ≤ D ≤ 11.
- **Δ.** "This extrapolation indicates that Δ might have a minimum … at about D ≃ 12" (Sec. V), from a second-order
  polynomial fit.
- **γ.** A fourth-order fit on 4 ≤ D ≤ 10: "γ seems to reach a maximum at about D = 11 and then decreases".

**Bland, Preston, Becker, Kunstatter and Husain, CQG 22, 5355 (2005), gr-qc/0507088.** Evolutions for 3.5 ≤ D ≤ 14.
- **γ.** "γ = 0.466(1 − e^{−0.408D})" (Introduction), and "strong evidence that this exponent increases monotonically
  to an asymptote around 0.466" (Sec. IV).
- **Δ.** "Δ decreases monotonically to an asymptotic value of just below 2.4" (Sec. IV), with Δ to ±0.1 from the
  oscillation period of the ln–ln fit.
- **On Sorkin–Oren.** "we do not observe the maximum for the critical exponent near D = 11", and in their own earlier
  data "the maximum was determined to be an artefact of the inaccuracy of the calculation".

**Birukou, Husain, Kunstatter, Vaz and Olivier, PRD 65, 104036 (2002), gr-qc/0201026.** Evolutions in d = 4, 5 and 6.
Their γ values are already compared in `hp/data/gamma_published_compare.csv`.

### 1.3 Self-similar solutions with large light-cone exponents (ODEs)

**Bizoń, Glogić and Wasserman, Phys. Rev. D 114, 064021 (2026), arXiv:2602.00345.** SO(d) Yang–Mills, CSS, odd d ≥ 11.
- **The equations.** The profile equation (Sec. 1) is y²(1 − y²)u″ + [(d − 3)y − 2y³]u′ + (d − 2)u(1 − u²) = 0. At y = 1
  it gives "an infinite algebraic system in the triangular form", whose coefficient of c_n begins (d − 5), (d − 7), (d −
  9), ….
- **The resonance.** "For d = 2m + 5 … the factor multiplying c_{m+1} in the triangular system vanishes". Smoothness
  is then "a codimension-one condition", c² a root of a polynomial P_m of degree m = (d − 5)/2 (Sec. 2). A
  generic solution has the term b(y − 1)^{m+1} log(1 − y) and "is only C^m at y = 1".
- **Even d.** "If d is even, then the above triangular system can be solved recursively for any given c, so all
  values of u(1) are a priori admissible."
- **Result.** N = 3 smooth solutions for the odd d computed (m = 3 to 15).
- **What this is** (*inferred*). The CSS, integer-exponent version of our situation: the light-cone information sits at
  order ≈ (d − 3)/2 of the light-cone series. Our σ ≈ n/2 + 2.6 is complex and never an integer. So, as in their
  even-d case, the analytic branch exists for any data, and the SSH condition removes the free coefficient of u^σ.
- **Does not contain:** gravity, DSS, conditioning.

**Buckmaster, Cao-Labora and Gómez-Serrano, Forum Math. Pi 13, e6 (2025), arXiv:2208.09445.** Smooth imploding Euler
profiles.
- **The sonic point.** "There do however exist non-smooth solutions of the form y = Cx^k which are C^k regular and
  whose Taylor series agrees with the solution y = 0 up to order ⌊k⌋".
- **The proof.** For r near the limit r*, the exponent k grows. "we employ a computer-assisted proof to compute the
  first 10000 coefficient pairs (W_j, Z_j) at r = r* with rigorous error bounds … about 14 hours on a single CPU",
  with an inductive tail bound for higher orders.
- **What this is** (*inferred*). The one published rigorous treatment of a regular singular point whose exponent grows
  with a parameter: high-order Taylor at the singular point with a certified tail.

**Biernat and Bizoń, CMP 338, 1443 (2015), arXiv:1409.8214; Biernat, Bizoń and Maliborski, Nonlinearity 30, 1513
(2017), arXiv:1608.07707.** Wave maps and Yang–Mills in d dimensions, CSS only. For d ≥ 7 the second conjectures
"that f₀ is the only self-similar solution". No DSS solution appears in either.

### 1.4 The 2025 review

Gundlach, Hilditch and Martín-García, arXiv:2507.07636 (read in full in LIT_REVIEW's addendum of #380). Sec. III C:
the critical solution "was determined accurately in [MGG 2003] by assuming analyticity at the centre of spherical
symmetry and at the past light cone". It lists Reiterer–Trubowitz as the only DSS existence proof. It names
computer-assisted CSS proofs that "control convergence of the Taylor series … about the sonic point" (Guo, Hadžić,
Jang and Schrecker). It contains no large-D construction.

### 1.5 How each formulation scales with D

| formulation | SSH condition | metric sector | what grows with n | published max D |
|---|---|---|---|---|
| Gundlach 1995/1997 (S) | order 0, Eq. (25) | a rebuilt from the momentum constraint | nothing in a (§2); the condition is blind once σ > 1 | 4 |
| MGG 2003 (A); EEGJ follow its method | Taylor O(u²) (MGG), O(u²)–O(u³) (EEGJ) | Hamiltonian integrated inward, in MGG to x_mid = 0.01 | mass mode 0.01^−(n−1) = 10^{2(n−1)} on the inward shot (*inferred*); blind once σ > 2–3 | 4; 5.5 (EEGJ) |
| Hirschmann–Eardley 1995 (CSS) | Taylor O((z − z₂)²) | ODE | problems at exponent ≈ 3 (κ ≲ 1/3) | 4 |
| Reiterer–Trubowitz (frequency space) | none: function space | quadratic frame variables | truncated σ_min falls as a power of the resolution, with an exponent that grows with D (RT_FORMULATION) | 4 (EEG26b: 200, slice 1) |
| ours (A, multiple shooting) | exact series to ks | Hamiltonian, a_h a free datum | G = x_K^−(n−1), 0.155n digits; ks from the mass-mode tail | 263.6 |

## 2. The mass mode is a constraint violation (lead 1, checked)

In model.rs's variables (n = D − 2, B = 1/a²) the two constraints are

```
Hamiltonian (G^t_t):   x a_x/a = ½[p² + m² + (n − 1)(1 − a²)]  =: H_rhs
momentum (G_tr):       (a_τ + x a_x)/a = c (p² − m²)/(2x)      =: flux
```

The second is `solver/matter/struct_common.py`'s a_t/a = (κ/n) r T_tr, in self-similar variables (τ = −ln(−t), so
(−t)∂_t = ∂_τ + x∂_x).

**Form A** (ours, and MGG 2003 inside the light cone) integrates the Hamiltonian constraint in x and never imposes the
momentum one. **Form S** (Gundlach 1995, 1997) imposes the momentum constraint at every x with a_x replaced by H_rhs,
and integrates only c, p and m in x. In terms of B, with q := ((x − c)p² + (x + c)m²)/x, form S reads

```
ω ∂_θ B = [(n − 1) + q] B − (n − 1),   equivalently   (ω∂_θ − (n − 1) − q)(B − 1) = q          (S)
```

At the SSH (x = c = 1) q = 2m_h². At n = 2 this is MGG 2003's Eq. (17) term by term (U ↔ m, V ↔ p, f ↔ c).

`mc_check.py` (sympy, log `mc_check.txt`) proves two identities along each form's x-equations:

| form | residual | identity | consequence |
|---|---|---|---|
| A | E = (a_τ + x a_x)/a − flux | x E_x = −(n − 1) a² E | a momentum violation set at the SSH reaches x as about x^−(n−1): **this is the mass mode** of LARGE_D_BASICS §3.3, the a_h column gain G of FACTORED_HORIZON §1 |
| S | H = x a_x/a − H_rhs | H_τ = (n − 1) a² H | at each x, H grows in τ at the rate (n − 1)a², so the only periodic H is zero: **periodicity enforces the Hamiltonian constraint**, with no x-mode left |

The control `--mutate=flux` flips the flux term of S, and the identity fails (`mc_check_mutate_flux.txt`).

`mc_states.py` (log `mc_states.txt`) solves (S) for B from each stored node's p, m and c, and from m_h at the SSH:

| D | worst residual of (S) / max\|(n−1)(B−1)\| | worst \|B_slaved − B\| / max\|B − 1\| | (n − 1)‖L⁻¹‖₂ |
|---|---|---|---|
| 68.08 | 2.6×10⁻¹⁴ (x 0.3) | 2.3×10⁻¹⁴ | 0.9987–0.9998 |
| 100.00 | 2.5×10⁻¹⁵ (x 0.4) | 2.5×10⁻¹⁵ | 0.9994–0.9999 |
| 226.13 | 7.3×10⁻¹⁶ | 9.8×10⁻¹⁶ | 0.9999–1.0000 |
| 263.60 | 6.6×10⁻¹⁶ | 1.1×10⁻¹⁵ | 0.9999–1.0000 |

The controls fire. Swapping (x − c) and (x + c) in q gives residuals of 0.037–1.0 (`mc_states_mutate_flux.txt`).
Using 1 for n − 1, the D = 4 coefficient, gives 0.985–0.996 (`mc_states_mutate_n1.txt`). The largest residual, at
D 68 and x 0.3, is above the double-precision floor of the others; the M = 32 τ-truncation is the likely cause, and it
was not tested.

**What follows** (*inferred*, untested in a solve):
- The slaving is a periodic linear solve per x whose inverse has norm 1/(n − 1). So ∂B/∂(p, m, c) = O(q/n): the
  metric sector is better conditioned as D grows, not worse.
- Without the a-equation, nothing of order n is left in x except the scalar's fast mode F (the (n/2)c(p + m) terms) and
  the SSH exponent σ. x c_x = −(n − 1)c(B − 1)/B is O(q), so even c's equation loses its factor n − 1.
- In the horizon series, a_h is no longer an unknown. a's coefficients come order by order from
  (ω∂_θ − (n − 1) − q₀)B_k = Σ_{j<k} q_{k−j}B_j, with one M/2 × M/2 factorization of L₀ per series. The mass-mode
  tail C(n−2+k, k) that sets ks at the front (LARGE_D_BASICS §4: 417 orders at n 263, 526 at n 400 for 2⁻²¹²) leaves.
  ks is then set by σ_R (≥ σ_R + 7 for Δ to 3×10⁻¹¹, LARGE_D_BASICS D3) and F's shadow (≈ 1.9σ_R for all of it,
  FACTORED_HORIZON M3).
- H1's pole (QUAD_DOUBLE 13.1, 13.6; LARGE_D_LITERATURE 2(f)) is the apparent horizon B → 0 of the mass mode that a
  data error excites. In S an error δ in m_h moves B by O(δ/n), so B cannot reach zero, and Newton's basin should stop
  shrinking with D. §2.1 tests the first half of this on stored states.
- The re-seed (RESEED.md) exists only to remove the a_h error that G amplifies. It becomes unnecessary.
- Unknowns: 4M(K + 1) + 1 = 769 becomes 3M(K + 1) + 1 = 577 at M 32, K 5.

**Caveat: Gundlach's EYM failure** (§1.1). The slaving operator of the *perturbation* problem, ω∂_θ + λ − (n − 1) − q,
is singular where λ − (n − 1)⟨1 + …⟩ meets ikω. At D = 4 that is λ ≈ 1 (his λ = −1), between 0 and the physical λ ≈
2.67, so mode counts there get a pole. At large D the line Re λ ≈ n − 1 is far from λ ≈ 2. The background problem (λ =
0) has a margin of n − 1: the measured (n − 1)‖L⁻¹‖ = 1.0000.

### 2.1 H1's pole follows the momentum residual of the horizon data

`pole_mc.py` (log `pole_mc.txt`) recomputes the horizon series with the kernel's form-A recurrence, as
`hp/pole_test.py` does. It moves the horizon data three ways by a relative δ:
- **A** scales a_h, pole_test's move;
- **S** scales m_h and re-slaves a_h from it by (S) at x = 1, in mpmath at 60 digits, split into double-double;
- **S0** scales m_h and holds a_h.

For each it prints pole_test's ratio-test radius R, beside E, the data's momentum residual at the SSH (the residual
of (S), relative to max|(n − 1)(B − 1)|). At D 68.08:

| δ | R_A | R_S | R_S0 | E_A | E_S | E_S0 |
|---|---|---|---|---|---|---|
| 0 | 0.585 | 0.467 | 0.585 | 5.9×10⁻¹⁷ | 5.9×10⁻³⁰ | 5.9×10⁻¹⁷ |
| 10⁻⁸ | 0.214 | 0.482 | 0.294 | 7.9×10⁻⁶ | 7.3×10⁻³⁰ | 2.0×10⁻⁸ |
| 10⁻⁶ | 0.152 | 0.482 | 0.235 | 7.9×10⁻⁴ | 6.7×10⁻³⁰ | 2.0×10⁻⁶ |
| 10⁻⁴ | 0.080 | 0.448 | 0.178 | 7.3×10⁻² | 7.2×10⁻³⁰ | 2.0×10⁻⁴ |
| 10⁻² | 0.027 | 0.404 | 0.117 | 8.9×10⁻¹ | 6.6×10⁻³⁰ | 2.0×10⁻² |

- **R is a function of E.** S0's points fall on A's curve read at the same E (linear in log E, printed by the
  script). At D 68.08, A's curve gives 0.300, 0.234, 0.171 and 0.101 at S0's E of 2×10⁻⁸, 2×10⁻⁶, 2×10⁻⁴ and 2×10⁻²,
  against the measured 0.294, 0.235, 0.178 and 0.117. The largest gap over all four states is 0.016 (D 68.08,
  E 2×10⁻²).
- **The size of the move barely matters.** m_h moved by 1% with a_h re-slaved keeps R at 0.404, above u₀ = 0.3, so
  the series still converges at the last node. a_h moved by 10⁻⁸ takes R to 0.214, below u₀.
- **S's baseline.** R_S(0) = 0.467 is below the stored state's 0.585. The slaved a_h differs from the stored one by
  5.9×10⁻¹⁷ of max|a_h − 1|, the M = 32 system's own departure from the continuum constraint (`mc_states` measures
  the same). The form-A series reads that difference as a momentum violation of that size. A form-S kernel slaves a
  in its own discretization and should keep the on-solution radius.
- **D 100, 226.13 and 263.60.** The same split: R_A falls to 0.039, 0.018 and 0.015, and R_S0 to 0.081, 0.031 and
  0.035, while R_S stays at 0.369–0.415, 0.176 and 0.151.
  - At the two large D the on-solution ratio-test radius (0.176, 0.151) is itself below u₀. With ks = 135 and 293 the
    last 40 orders there are not asymptotic (LARGE_D_BASICS §4's mass-mode tail), so R is a reading of those orders'
    growth, not a radius.
  - Their growth is still unchanged under S's moves and steepened under A's and S0's.

Verdict: the collapse of the horizon series off the solution, which sets Newton's basin at large D (13.6), is driven
by the horizon data's momentum residual. It is not driven by the size of the data error (numerical evidence, four
states, the venv-314 kernel). A test with a form-S kernel, and of the basin itself, is still to be done (§5, lead 1,
tests 3 and 4).

## 3. Gaps: assumptions that fail at large D, and claims our data contradict

**G1. "With the one condition (25) we automatically enforce analyticity"** (Gundlach 1997, Sec. II C).
- **When it holds.** The argument rests on the sign of Â₀/B̂₀. It is valid while the exponent of the singular branch is
  below 1: at D = 4, σ = ḡ/(1 − ḡ) with ḡ ≈ 0.386, so σ ≈ 0.63. Here 1 − ḡ ≈ 0.614 is identified with the surface
  gravity ν_SH ≃ 0.6138 that the 2025 review quotes, and ḡ with LIT_REVIEW #619's 1 − 0.6139 (*inferred*).
- **When it fails.** In D dimensions σ_j = (n/2 − 1 + ḡ − ijω)/(1 − ḡ) (LARGE_D_BASICS §2). Once σ > 1 a branch with
  F ≠ 0 is C^⌊σ⌋ and satisfies (25), so the condition stops selecting. The same holds for MGG's and EEGJ's order-2–3
  Taylor once σ > 2–3 (EEGJ Sec. 8.3 says so), and for Hirschmann–Eardley's order 2 at κ ≲ 1/3. *Inferred*: σ reaches
  1 at n = 4 − 4ḡ, near D 5 given D = 4's ḡ.
- **Our solver.** It does not share the gap. Its series imposes the analytic branch at every order up to ks, and D3
  measures what an order below σ_R costs.

**G2. Form A's free momentum constraint.** No paper read discusses it. MGG 2003 shoot the Hamiltonian equation inward
from x = 0.9999 to x_mid = 0.01, which amplifies a momentum violation by about 0.01^−(n−1) (*inferred*): 10² at
D = 4, 10¹⁴ at D = 10. EEGJ follow MGG's construction (LIT_REVIEW, #802 addendum). Whether their inward shot has this
factor is slice 1's to read. Our all-outward layout meets the same mode only through the horizon block's a_h columns
(G).

**G3. The slaved form's own pole** (Gundlach's EYM paper, §1.1) is a real limitation at D ≈ 4 for perturbations with
Re λ ≳ n − 1. *Inferred*: it is harmless for the background at every D and for γ once n − 1 ≫ 2.

**G4. Published D-trends that our data contradict.** All are extrapolations from evolutions, which the authors flag as
such.

| claim | source | ours | verdict |
|---|---|---|---|
| Δ has a minimum near D ≃ 12 | Sorkin–Oren, Sec. V | Δ falls monotonically: 2.4496 (D 11), 2.3892 (12), 2.3374 (13), 2.2924 (14), 1.6829 (263.60); Δ∞ = 1.643893746 ± 1.5×10⁻⁸ | contradicted (`hp/data/delta_all.csv`, README) |
| γ has a maximum near D ≈ 11 | Sorkin–Oren, Sec. V | γ rises: 0.45559 (10), 0.45876 (11), 0.46141 (12), 0.46564 (14), 0.48935 (61.03) | contradicted (`hp/data/gamma_dd.jsonl`); Bland et al. also see no maximum |
| Δ → "just below 2.4" | Bland et al., Sec. IV | 2.2924 at D 14; Δ∞ = 1.6439 | contradicted |
| γ → 0.466, γ = 0.466(1 − e^{−0.408D}) | Bland et al., Introduction and Sec. IV | 0.46737 at D 15, 0.48935 at D 61.03 | contradicted from D 15 on |
| "one condition … automatically enforce[s] analyticity" | Gundlach 1997, Sec. II C | true at D = 4 only (G1) | limited to σ < 1 |

The values at D 10–15 come from delta_all's dd rows and gamma_dd.jsonl; Δ at D 263.60 is the stored state's
(`mc_states.txt`). Bland's thesis "period of 5/2" at large dimension (LIT_REVIEW §1, not re-read here) is
contradicted by Δ∞ in the same way.

**G5. "Seamless" boundaries in frequency space** (Reiterer–Trubowitz, Sec. 1). True of the infinite-dimensional problem.
In a truncation it fails at large D, as measured in RT_FORMULATION; this is listed here only for completeness.

**G6. What none of the papers has.** No construction read imposes regularity at a regular singular point of a PDE in
(τ, x) whose exponent grows with a parameter. The two ODE works that meet a growing exponent (Bizoń–Glogić–Wasserman,
Buckmaster–Cao-Labora–Gómez-Serrano) do what our horizon series does: an exact or certified Taylor recurrence at the
singular point to order ≈ exponent. No integral condition, contour or monodromy method appears in any of them
(LARGE_D_LITERATURE (b) and (e) already rule these out for our case). Continuation past the SSH (MGG 2003,
Hirschmann–Eardley) is done after the solve and adds no condition on the past patch.

## 4. Connections to our formulation and to D = ∞

- **Our past patch is MGG 2003's.** Their Eqs. (15)–(19) at n = 2 are model.rs's system, with the same gauge and the
  same choice of the Hamiltonian equation for a. Their Eq. (17) is the n = 2 case of (S). The project's equations
  generalize theirs. The choice of a's equation is the one thing that matters at large D and that neither they nor we
  revisited.
- **Gundlach 1997's pole at λ = −1 is (S)'s slaving operator** (*inferred*, from App. B's Eq. (B4) against §2).
  LIT_REVIEW's #619 addendum noted that his λ = −1 "would sit at λ = 1 here, where the D = 4 real scan finds no sign
  change of det M". That fits: form A, which MODE_COUNT.md uses, has no slaving operator and so no such pole. His other
  pole, λ ≃ −0.385, comes from the light-cone condition, which both forms share. It is MODE_COUNT's kk = 0 horizon
  resonance (LIT_REVIEW #619).
- **The metric sector at D = ∞.** From (S), B − 1 = L⁻¹q with L = ω∂_θ − (n − 1) − q. Expanding the inverse in
  (ω∂_θ − q)/(n − 1) gives (n − 1)(B − 1) = −q − (ω∂_θ q − q²)/(n − 1) − …, convergent for every harmonic
  j < (n − 1 − ‖q‖)/ω. So the metric is slaved to the scalar to all orders in 1/n, with no layer (*inferred*;
  convergence checked numerically only through (n − 1)‖L⁻¹‖ = 1.0000 on the states). LARGE_D_BASICS §3.2's
  (n − 1)(a² − 1) = Kx²/(1 + Kx²) is its first term on the slow manifold. The selection that the inner problem must
  carry then lives in the scalar sector alone, through F. That agrees with LARGE_D_BASICS §3.4 and gives a smaller
  inner system: two fields, p and m, and c, which is slaved too at leading order.
- **Large-d rigidity** (Bizoń–Glogić–Wasserman). Their N = 3 for every odd d up to 35 is an ODE example of a
  selection that does not wash out as the light-cone exponent grows. It is an analogy only; Yang–Mills in flat space
  shares no equation with ours.

## 5. Ranked leads

**1. Slave a to the momentum constraint (form S).** This is the top lead.
- **Formula.** (S) of §2 at every node, Taylor order and series order; drop the a rows and a_h.
- **Precedent.** Gundlach 1995 and 1997, and the EYM paper, at D = 4.
- **Already shown** (§2). Checked symbolically, and it holds on our states with conditioning 1/(n − 1).
- **Predicted at n = 400** (*inferred*, from LARGE_D_BASICS §4's table). The mass-mode gain of 61.8 digits goes to zero.
  ks falls from 526 (the tail at 2⁻²¹²) to between σ_R + 7 ≈ 210 and 1.9σ_R ≈ 385. What is left is the F match
  (0.0087n, 3.5 digits) and the weak directions' gain. At n = 1000 the gain is 154.7 digits.
- **Tests, in order, each able to fail.**
  1. **f64 prototype on `solver/dss.py`.** Integrate p, m and c only, and take a from (S) with `DSS.solve_periodic`
     at every stage; the unknowns are p1, c0, m_h and Δ. Pass: Δ at D 10, 20 and 27 equals delta_all to the f64
     discretization's own spread (steps doubled). Fail: a difference beyond it. About 3 h of coding; seconds per solve.
  2. **Condition number of the shooting Jacobian against D**, forms A and S on hp's all-outward layout, D 10–27 in f64
     (the f64 solver's range, solver/CONTEXT.md). Pass: S has no factor 10^{0.155(n−1)}, and the ratio κ_A/κ_S grows as
     x_K^−(n−1). Fail: S's κ grows exponentially in n. Minutes once test 1 runs.
  3. **The horizon series' radius under perturbed data.** §2.1 did the half that the form-A kernel allows: the radius
     follows the momentum residual of the data alone, and m_h moves of up to 10⁻² with a_h re-slaved keep it above u₀.
     What remains is the same reading with a form-S series, whose a comes from the slaving at every order. Pass: R stays
     at the on-solution value for δ up to 10⁻², and that value at D 68 is no lower than form A's 0.585. Fail: R falls
     with ln(1/δ)/(n − 1). About 2 h, in mpmath or in the prototype of test 1.
  4. **hp port.** NF 3 and NR 1 in multishoot's layout; B's Taylor and series coefficients from one factorization of
     L₀ per base point. Pass: Δ at D 200 within the chain's 1.6950887524638418 ± 2.8×10⁻¹¹ (chain_fit,
     delta_all.csv), then a jump to D 300 and 400 within track C's 1.678171015 ± 1.3×10⁻¹⁰ and
     1.669658064 ± 2.3×10⁻¹⁰. Fail: a disagreement beyond the stated errors. About a day.

**2. Under S, put the last node at F's crossover.**
- **Formula.** x_K solves −ln x + ((1 + K)/2) ln(1 − x²) = 0, which is 1/√2 in the flat limit (LARGE_D_BASICS §3.3–3.4).
  The 0.0087n digits of the match at 0.7 then go to about zero.
- **Why it matters.** With G gone, this is the only term linear in n left in the digit budget. It is 17 digits at
  n = 2000 (*inferred*, 0.0087n).
- **Test.** `solver/beyond_lc/hp_node_test.py`'s e^{n[Φ(x_m) − Φ(0.7)]} law, in form S. Pass: the residual floor and
  Δ's sensitivity at D 263.6 fall by that factor when the node moves from 0.70 to the crossover. Minutes once lead 1
  runs.

**3. A jump seed with no constraint violation** (Track A of #1088, through S).
- **Idea.** Rescale a stored state's p (p·√n, #1088) and m to the target D. Take c and a from the slaving and the c
  equation instead of rescaling them. The seed then satisfies both constraints exactly, and its error lies only in the
  slow fields. In A, a rescaled a carries mass-mode content that only the re-seed removes.
- **Test.** Newton from the seed, 245 → 250, in S against A: count the iterations and halvings. Pass: S converges
  with no re-seed. An hour once lead 1's hp port exists.

**4. γ(D) past D 61 in form S.**
- **Idea.** The perturbation problem loses G the same way. Its slaving pole sits at Re λ ≈ n − 1, far from λ ≈ 2 (G3).
- **Test.** S's mode matrix at D 30–61 against `hp/data/gamma_dd.jsonl`, with a winding count that must find the
  physical zero and no pole below Re λ = n − 2. A few hours after lead 1.

**5. A certified solution at one large D.**
- **Route.** S's slaving gives a provable D-uniform bound, ‖L⁻¹‖ ≤ 1/(n − 1 − ‖q‖), in any weighted ℓ¹ Fourier norm:
  ω∂_θ − (n − 1) is diagonal there with |ikω − (n − 1)| ≥ n − 1, and q acts with at most its own norm. Add a
  Buckmaster–Cao-Labora–Gómez-Serrano-style rigorous Taylor recurrence at the SSH to order about 2σ with a tail bound,
  and a Reiterer–Trubowitz-style Newton–Kantorovich bound.
- **Test.** Certify at D = 10 first. Weeks of work; not for the 24 h quest.

**Ruled out by the reading:**
- order-0 to order-3 conditions at the SSH (G1);
- two-sided shooting to a point near the centre (G2);
- a global Chebyshev representation across the SSH (RT_FORMULATION);
- continuation past the SSH as a regularity condition (G6).

## 6. References checked (arXiv API, 2026-11-25)

| entry | journal reference and DOI (arXiv) | verdict |
|---|---|---|
| C. Gundlach, gr-qc/9507054 | PRL 75, 3214 (1995), 10.1103/PhysRevLett.75.3214 | OK |
| C. Gundlach, gr-qc/9604019 | PRD 55, 695 (1997), 10.1103/PhysRevD.55.695 | OK |
| C. Gundlach, gr-qc/9610069 | PRD 55, 6002 (1997), 10.1103/PhysRevD.55.6002 | OK |
| J. M. Martín-García, C. Gundlach, gr-qc/0304070 | PRD 68, 024011 (2003), 10.1103/PhysRevD.68.024011 | OK |
| E. W. Hirschmann, D. M. Eardley, gr-qc/9412066 | PRD 51, 4198 (1995), 10.1103/PhysRevD.51.4198 | OK |
| M. Reiterer, E. Trubowitz, arXiv:1203.3766 | CMP 368, 143 (2019), 10.1007/s00220-019-03413-8 | OK |
| E. Sorkin, Y. Oren, hep-th/0502034 | PRD 71, 124005 (2005), 10.1103/PhysRevD.71.124005 | OK |
| J. Bland, B. Preston, M. Becker, G. Kunstatter, V. Husain, gr-qc/0507088 | DOI 10.1088/0264-9381/22/24/009 (CQG 22, 5355 from REF_CHECK.md) | OK |
| M. Birukou, V. Husain, G. Kunstatter, E. Vaz, M. Olivier, gr-qc/0201026 | DOI 10.1103/PhysRevD.65.104036 | OK |
| P. Bizoń, I. Glogić, A. Wasserman, arXiv:2602.00345 (v3) | Phys. Rev. D 114, 064021 (2026); no DOI listed | OK as listed by arXiv |
| T. Buckmaster, G. Cao-Labora, J. Gómez-Serrano, arXiv:2208.09445 | Forum Math. Pi 13, e6 (2025), 10.1017/fmp.2024.12 | OK |
| P. Biernat, P. Bizoń, arXiv:1409.8214 | CMP 338, 1443 (2015), 10.1007/s00220-015-2404-y | OK |
| P. Biernat, P. Bizoń, M. Maliborski, arXiv:1608.07707 | DOI 10.1088/1361-6544/aa61ab (Nonlinearity 30, 1513 (2017), from citing papers) | volume and page *(unverified)* |

**Read ledger.** All of each TeX source above was searched by keyword, and the sections quoted were read in full. That
covers Gundlach 1995 (all), Gundlach 1997 (Secs. II and IV, Apps. A–D), the EYM paper (Secs. II–III and the
perturbation section), MGG 2003 (Secs. III and IV), Hirschmann–Eardley (Secs. III–IV), Bizoń–Glogić–Wasserman (Secs.
1–2) and Bland et al. and Sorkin–Oren (results and conclusions). Reiterer–Trubowitz (Sec. 1) and
Buckmaster–Cao-Labora–Gómez-Serrano (Secs. 1–2, the Taylor sections) were read in part. Read only by keyword:
arXiv:math/0702156, gr-qc/0411078 and the two wave-map papers. EEG, EEGJ and EEG26b belong to slice 1 and were not
re-read.

## 7. Proposed text (for the coordinator)

For ALGORITHM.md §1, if lead 1 is adopted:

> The metric function a is not integrated in x. At each x it is the periodic solution of the momentum constraint,
> ω∂_θB = [(n − 1) + q]B − (n − 1) with B = 1/a², as in Gundlach's construction of the D = 4 solution (PRD 55, 695
> (1997), Eq. (19)). The Hamiltonian constraint then holds by periodicity, and the mass mode x^−(n−1), which is the
> momentum constraint's violation, is absent.

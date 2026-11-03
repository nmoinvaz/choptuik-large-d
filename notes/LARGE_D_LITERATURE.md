# Literature search: ways to continue the branch to large D (2026-11-03)

Written by a research subagent of the coordinator session on 2026-11-03 at the author's request ("have a subagent scour
the internet and scientific papers for ways to continue large D"), after reading notes/QUAD_DOUBLE.md 12–13 and
notes/BEYOND_LIGHT_CONE.md. Sources marked *read* were read in full text; entries not checked at the publisher or arXiv
are marked *(unverified)*. The coordinator's reading is in notes/QUAD_DOUBLE.md 13.10.


Read first: notes/QUAD_DOUBLE.md 12.2–12.7, 13.1–13.9; notes/BEYOND_LIGHT_CONE.md 2.3, 2.7, 3, 7; README "Large D" and
"What the horizon block's wall is". Sources marked *read* were read in full text (arXiv/ar5iv); the rest come from
abstracts and bibliographic listings. Entries not checked at the publisher or arXiv are marked *(unverified)*.

## 1. Verdict

No published method gets past the wall as stated. Every numerical treatment of a regular singular point in the
literature imposes either a finite-order condition (singular-BVP collocation; Taylor conditions of order 2–3 in the
critical-collapse codes) or the convergent local series, which is what `hp/` already does; the first is what T3 ruled
out, the second has the pole. The wall is physical: the pole of 13.6 is the apparent horizon of the mass perturbation
that the data error carries (section 2f, a one-line derivation giving C = 1 against the measured 0.87–0.92), so any
formulation that evaluates the perturbed horizon family on [0, u₀] has the same basin. What the literature offers is on
the predictor and continuation side, not the condition: (i) high-order Taylor continuation in D (the Asymptotic
Numerical Method), which turns the predictor error from h⁴ into (h/R_D)^{N+1} and reaches 10⁻³¹ if the branch's radius
of convergence R_D in complex D is about ten times the step, a quantity nobody has measured; (ii) analytic continuation
of the horizon series past its pole (Fornberg–Weideman pole-field integration, or Padé), which widens the basin at a
cost of (u₀/u_pole)^{Re s} digits, about 130 digits in the horizon block for a 10⁻⁶ predictor at D = 200. Both are
*possible*, neither *promising* until measured. No other group has a discretely self-similar solution past D = 14, and
the large-D theory fixes Δ at no order, so it cannot supply a guess.

## 2. Candidate methods

### (a) Collocation for singular BVPs of the first kind. Ruled out.

de Hoog and Weiss, SIAM J. Numer. Anal. 13 (1976) (difference methods) and 15 (1978) 198–217 (collocation); the Vienna
line (Weinmüller, Koch, Auzinger; bvpsuite, Kitzhofer et al., JNAIAM 5 (2010) 113–134 *(unverified)*). The class is
z′ = M(t)z/t + f(t, z). The theory sorts the eigenvalues λ of M(0): components with Re λ < 0 are killed by the condition
at t = 0; components with Re λ > 0 behave as t^λ, are continuous, and are *admitted*, fixed by the condition at the
other end. Koch, Numer. Math. 101 (2005) 143–164, covers "the most general class of singular problems, where the
coefficient matrix may have eigenvalues with positive real parts"; the convergence orders hold "provided that their
solutions are appropriately smooth" (Burkotová, Rachůnková, Weinmüller, ASC report 30/2017).

So the framework treats the project's u^{s_j} branches as legitimate solutions selected by the far boundary, with no
analyticity condition anywhere, and its codes are polynomial collocation: the block of 13.2 with theory attached. It
confirms T3 (13.7), nothing more.

### (b) Spectral bases with the exponents built in; nonlocal analyticity. Ruled out.

Boyd, Math. Comp. 37 (1981) 155, gives the Chebyshev coefficients of an endpoint power u^α, decaying as n^−(2α+1); the
singular-basis-function strategy (Boyd, Appl. Math. Comput. 1989) *adds* such terms to represent a singular solution.
Here they must be excluded, and for α = 35–98 they lie inside the polynomial span to 10⁻⁷⁰ on [0, u₀] (13.3), so a basis
that carries them explicitly is numerically dependent and their amplitudes stay as undetermined as in T3. The only
nonlocal condition equivalent to analyticity is monodromy (the solution continued once around u = 0 returns to
itself); for harmonic j the singular mode's monodromy factor is e^{2πjω/(1+c₁)} ≈ 10^{10j}, the Stokes-line weights
BEYOND_LIGHT_CONE 7 costed at 70–85 digits, and on a small circle of radius ρ the same factor is multiplied by
ρ^{Re s} (10⁻⁵¹ at ρ = 0.3, D = 200). Rigorous analytic-continuation machinery (Mezzarobba, arXiv:1607.01967; Ann. H.
Lebesgue 2 (2019) 99–148) is for linear equations and does what the horizon block does: a generalized series at the
singular point, Taylor steps along a path.

### (c) Analytic continuation past the pole. Possible, costly.

Fornberg and Weideman, *A numerical methodology for the Painlevé equations*, J. Comput. Phys. 230 (2011) 5957–5973: a
Taylor/Padé initial-value integrator that steps through pole fields in the complex plane, "including right up to poles
at unknown locations", coupled to a boundary-value solver on pole-free regions (extended to multivalued transcendents
by Fasondini, Fornberg, Weideman, J. Comput. Phys. 2017). Applied here: for data δ off, the family member's singularity
sits at u_pole on the positive real axis (13.6); its continuation to u₀ along a complex detour, or by a Padé approximant
of the series, is finite and analytic in the data, so the residual it defines has an ordinary Newton basin, and the
series summed at u₀ is the special case u_pole > u₀.

The cost is conditioning: the function's content beyond the pole sits (u₀/u_pole)^k below the pole's geometric growth
at order k, and the singular modes regenerated by rounding on the detour grow by (u₀/r₁)^{Re s} from the summation
point r₁ ≈ 0.8 u_pole. With P digits one needs (u₀/r₁)^{Re s} ≲ 10^{P−35} (35 digits to land inside the plain basin).
At D = 200, Re s = 98, u₀ = 0.3: quad-double (P = 64) allows r₁ ≥ 0.15 and widens the basin from 10⁻³⁰ to about 10⁻¹⁹;
P = 130 (python-flint at 430 bits, horizon block and its tangents only) allows r₁ ≈ 0.034, so a 10⁻⁶ predictor
suffices. Prerequisite, unmeasured: whether the singularity is a simple pole in b = a² and c (single-valued
continuation; the a-equation's Riccati form suggests b ~ 1/(x_p − x)) or a branch point, where the two sides of the
detour define different residuals, both vanishing on the solution. 13.6 did not read the exponent.

### (d) Continuation methods. ANM possible; the rest ruled out.

High-order Taylor continuation (Asymptotic Numerical Method). Cochelin, Comput. Struct. 53 (1994) 1181–1192;
Cochelin, Damil, Potier-Ferry, Int. J. Numer. Methods Eng. 37 (1994) 1187–1213 *(unverified pages)*, with Padé
approximants; Guillot, Cochelin, Vergez, Nonlinear Dyn. (2019), DOI 10.1007/s11071-019-04989-5. The branch z(D₀ + h) =
Σ z_k h^k is computed to order N from one LU of the Jacobian and N right-hand sides formed by truncated-power-series
arithmetic through the residual (the "quadratic recast"); the step is a_max = (ε‖z₁‖/‖z_N‖)^{1/(N−1)}, N = 15–30 in
practice. The predictor error is (h/R_D)^{N+1}. From 13.6 the requirement at D = 200 is (N + 1) ln(R_D/h) ≥ 69 at
u₀ = 0.3 (δ_max 10⁻³⁰): N = 30 needs h ≤ R_D/10, N = 40 needs h ≤ R_D/5.5; at u₀ = 0.2 (5×10⁻²¹) R_D/5 and R_D/3.3.
Whether that is sixty steps or six thousand depends on R_D, measurable now at D = 60–78 from the stored states (a
Domb–Sykes plot of the z_k the kernel would produce), and checkable by stepping to a D where δ_max is known. What it
takes: TPS arithmetic in D through `model.rs` (the residual is polynomial in the unknowns and in n, the horizon
recurrence included) in quad-double; the flat direction of 12.3 makes each z_k solve ill-conditioned by 1/σ_min, which
the quad-double floor covers to about D 150 under 12.4's 0.73-per-unit law. This is the one literature method aimed
at the actual wall, the predictor, without touching the horizon condition.

Pseudo-arclength and deflation (Farrell, Birkisson, Funke, SIAM J. Sci. Comput. 37 (2015) A2026) do not change a
Newton basin; the walls are not folds or multiple roots. The solution's own 1/D series as predictor: its coefficients
can only come from fits (10⁻¹⁵, 12.5), since Proposition 1 shows the 1/n expansion carries any period; ruled out.
Transseries in 1/D (Emparan and Herzog, Rev. Mod. Phys. 92 (2020) 045005, Sec. III.F) concern the n = ∞ expansion, not
the Taylor series of the branch about a finite D.

### (e) Reformulations: coordinates, gauges, split unknowns. Ruled out as a cure.

Gundlach, PRD 55 (1997) 695, gr-qc/9604019 *(read)*: coordinates (τ, ζ), Fourier in τ, relaxation in ζ; the past light
cone is a "sonic line" where the equations are singular; the general solution there has a homogeneous part
ζ^{Â₀/B̂₀} F(τ), "the solution is analytic if and only if F vanishes identically", and analyticity is imposed by the
single condition that the numerator of the X₋,ζ equation vanish at ζ = 0. Martín-García and Gundlach, gr-qc/0304070
*(read)*: two-sided shooting to x_mid = 0.01 with a second-order Taylor series at the light cone; "they are regular
singular points of the equations". Ecker, Ecker, Grumiller, Jechtl, SciPost Phys. 20 (2026) 176, arXiv:2602.10185
*(read)*: the same construction, Taylor order 2 at the SSH for D < 4.8 and 3 for 4.8 ≤ D ≤ 5.5, and the explicit
statement that "the singular branch gets less and less singular the higher the dimension and therefore, is harder to
discriminate numerically", with "adding more Taylor expansion coefficients near the SSH" as their next step. All three
impose a finite-order condition, sufficient while the exponent is below the Taylor order; EEGJ meet the project's
mechanism in its infancy at D = 5.

Coordinates cannot remove it. In the reduced DSS system the SSH is the orbit of a null generator under the
self-similarity, hence a characteristic in any coordinates, and the exponents are eigenvalues of the residue matrix,
invariant under coordinate changes and shifted only by integers when unknowns are rescaled. Double-null or Bondi
coordinates move the conditioning factor e^{nΦ(x_m)} the project measured (BEYOND_LIGHT_CONE 5), not the exponents.
Writing p = analytic part + u^s part with an explicit amplitude reintroduces the unknown the collocation could not
determine (13.9).

### (f) The pole as an apparent horizon; Newton basins at large D. Nothing in print; one derivation.

No paper on either was found. The perfect-fluid literature (Ori and Piran 1990; Carr and Coley, CQG 16 (1999) R31
*(unverified)*) has the same structure, families analytic at the sonic point meeting singularities when shot toward the
centre, hence two-sided shooting, and nothing on basins in D. The elementary mechanism behind 13.6: the mass mode
solves x δb_x = −(n − 1) δb, so a relative mass perturbation δ at the horizon is δ x^{−(n−1)} inside, the Tangherlini
term 2M/r^{n−1} in self-similar units, and b = g_rr blows up where it reaches order one, x_AH = δ^{1/(n−1)}, i.e.
u_pole = ln(1/δ)/(n − 1): C = 1 and R₀ = 0 at leading order, against the measured 0.87–0.92 and −2.6/(n − 1) (13.6's
prefactor 14). The basin is "no trapped surface before the matching node", a property of the Einstein equations, and
no formulation that evaluates the perturbed family on [0, u₀] escapes it; (c) and (d) escape by not evaluating it
there, or by not being outside it.

## 3. What the other groups do at high D

| group | method | SSH treatment | max D |
|---|---|---|---|
| Gundlach 1997 | Fourier in τ, relaxation in ζ | one condition (numerator = 0) | 4 |
| Martín-García, Gundlach 2003 | two-sided shooting, IRK2, 256 Fourier points | Taylor order 2 | 4 |
| Ecker, Ecker, Grumiller, Jechtl 2026 | two-sided shooting, IRK4, 512 modes, Newton, ΔD ≈ 0.01 | Taylor order 2–3 | 5.5 (none for 4.81–5.38) |
| Garfinkle, Cutler, Duncan 1999 (gr-qc/9908044) | evolution, tuning | none | 6 |
| Sorkin, Oren 2005 (PRD 71, 124005) | evolution | none | 11 |
| Bland et al. 2005 (CQG 22, 5355); Taves, Kunstatter 2011 (arXiv:1105.0878) | evolution; Painlevé–Gullstrand | none | 14; 7 |
| Ecker, Ecker, Grumiller 2026 (PRL 136, 191401) *(read)* | analytic 1/(D − 1) expansion | f(τ, 1) = 1 as a boundary condition | ∞; NNLO example viable for D ≥ 52, "not convergent for D ≲ 100" |
| Clark, Pimentel, JHEP 11 (2025) 017 *(read in part)* | CSS only, matched asymptotics in 1/(D − 2) | fixed point in phase space | ∞; numerics shown at ε = 0.05 |
| Emparan, Herzog 2020, Sec. X.C *(read)* | review | — | γ ≈ ½(1 − 1/(D + 1)); Δ "hard to extract numerically" |
| this project | series block, double/quad-double | exact (the analytic family) | 78 |

Nobody else has a DSS solution past D = 14 or a horizon treatment beyond a low-order Taylor condition. EEG's LO
solution admits "any periodic function β(τ) with any echoing period Δ"; their NLO condition is the one the project
disputes (LIT_REVIEW C8). Kol, JHEP 0610 (2006) 017, estimates Δ(D) "with un-controlled accuracy". No external guess
exists at any accuracy useful to the series block.

## 4. Open questions

1. R_D, the branch's radius of convergence in complex D at D ≈ 60–78: one kernel measurement (D-Taylor coefficients
   from the stored states, Domb–Sykes), and it decides (d). R_D ≳ 20 would make a 10⁻³¹ predictor a matter of N ≈ 30
   and steps of 2.
2. The nature of the pole (simple pole in b, c, or branch point), from the late-order exponent of 13.6's Domb–Sykes
   fit with more orders; it decides whether (c) is single-valued.
3. Homotopy in the matching point at fixed D (a homotopy in the discretisation, not in the literature): solve first
   with u₀′ < u_pole(δ_pred), where the series converges, then re-layout outward (`hp/relayout.py`), each rung starting
   from data off by the previous rung's floor/σ_min(u₀′); it closes when R₀ + C ln(σ_min(u₀_k)/floor)/(n − 1) > u₀_{k+1}.
   With 12.7's five decades per 0.05 of u₀ and 12.4's 0.73 per unit of D, quad-double closes the ladder at D = 100 and
   150 and misses 200 by a few decades; under H2's pure u₀^{Re s} it fails everywhere. Needs σ_min(u₀, D) measured.
4. The premise "precision is not the obstacle" holds under 12.4's empirical σ_min law (10⁻³² at D 200, 0.7 layout) and
   fails under H2's (0.3⁹⁸ = 10⁻⁵¹, Δ at 10⁻⁹ in quad-double); T3 at D = 40 supports H2. Reconcile before planning
   D = 200 in quad-double.
5. From 2f, untested: the pole is driven by the mass-mode content of the data error (M numbers) and a_h(τ) has M
   harmonics, so maximizing the series' radius estimate over a_h at fixed other data would remove that content and
   might enlarge the basin with no new formulation. A cheap prototype test on the stored states.

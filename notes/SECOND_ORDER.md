# The 1/n² coefficients of Δ and γ from the inner problem, and EEG's NNLO

This note carries [FIRST_ORDER.md](FIRST_ORDER.md) one order further: the O(1/n²) inner equations, the selection at that
order, Δ₂ and λ₂ in the M-harmonic truncation, and their comparison with the double-double fits. It also checks the
large-D expansion of Ecker, Ecker and Grumiller (EEG, arXiv:2611.07007v1) to next-to-next-to-leading order against
Proposition 1. Issue #94. It was written by Claude (Anthropic), a subagent of the coordinator on the Mac, and has not
been checked by a mathematician. Notation as in FIRST_ORDER.md: n = D − 2, ξ = x/√n, y = p − m, σ = √n (p + m),
ω = 2π/Δ, λ = 1/γ, F the profile, M the number of harmonics kept. Scripts are in
[`solver/inner_pert/`](../solver/inner_pert/), named `second_order_*`. Raw output (logs, JSON) is in the folder
`I94-mac/` of the raw-data repository.

Contents

- [Summary](#summary)
- [1. EEG's expansion against Proposition 1](#1-eegs-expansion-against-proposition-1)
- [2. The inner equations to second order](#2-the-inner-equations-to-second-order)
- [3. Second-order jets and their checks](#3-second-order-jets-and-their-checks)
- [4. The selection at second order: Δ₂ and λ₂](#4-the-selection-at-second-order-δ₂-and-λ₂)
- [5. Against the fits, and Δ∞ with Δ₁ and Δ₂ held fixed](#5-against-the-fits-and-δ-with-δ₁-and-δ₂-held-fixed)
- [6. What EEG's e^(−1/ε) corresponds to here (a reading)](#6-what-eegs-e1ε-corresponds-to-here-a-reading)
- [7. What is rigorous, what is numerical, what is truncated](#7-what-is-rigorous-what-is-numerical-what-is-truncated)
- [8. Discrepancies](#8-discrepancies)
- [9. Proposed text](#9-proposed-text)
- [10. Scripts and reproduction](#10-scripts-and-reproduction)

## Summary

The O(1/n²) inner selection gives both second coefficients, in the M-harmonic truncation, against the double-double
fits:

| | M = 1 | M = 3 | M = 5 | fit (double-double) |
|---|---|---|---|---|
| Δ₂ | −45.86 | −44.42 | −50.06 ± 0.18 | −49.1685 [−49.16867, −49.16780] |
| λ₂ | −141.05 | −43.76 | −33.37 ± 0.38 | −34.38 (± about 0.1) |

with Δ = Δ* + Δ₁/n + Δ₂/n² + … and λ = 1/γ = λ* + λ₁/n + λ₂/n² + …, n = D − 2. At M = 5 the derived values are 1.8%
(Δ₂) and 2.9% (λ₂) from the fits. From M = 3 to M = 5 each error changes sign and shrinks, by factors of 5 and 9, and
each fitted value lies between the M = 3 and M = 5 values, as for every first-order coefficient. The ± of the M = 5
values is half the spread of three runs (numerical error); the bracket of the fitted Δ₂ is the spread over fit
choices; the ± of the fitted λ₂ is the largest miss of its planted test, which failed its 0.02 criterion (§5).

**EEG's expansion.** Under the map f = c, x²Ω = (n − 1)(b − 1), Π = y/(√2 x), Ψ = (p + m)/(√2 x²), ε = 1/(n + 1), with
β̃ the exact central profile, EEG's large-D expansion is Proposition 1's. Their NLO agrees term by term. Their printed
NNLO does not: Π₃ (S46) violates their own (S39) at ε^{5/2}, and Ω₃ and f₃ inherit the error. The corrected NNLO (§1)
agrees with Proposition 1 and satisfies their equations. *Checked symbolically*, with controls that fail. Against 63
solutions from D = 15 to 215 the corrected NNLO's largest error falls like ε⁴ (f, Ω) and ε^{7/2} (Π), and the printed
one's like ε³ and ε^{5/2}. In EEG's Figs. 4 and 5, which use the D-independent profile of their Eq. (16), the error is
small next to the profile's own (§1.1, *numerical evidence*).

**Method.** In inner variables the exact equations close with one slaved function, which gives the O(1/n²) inner
equations in a few terms (§2). Second-order jets by polarization (two nilpotent 1/n directions) carry the selection one
order further (§3): the O(1/n²) inner series equals the outer expansion's next-to-next-to-top coefficients to 6.5×10⁻¹³,
and the θ-shift mode keeps λ₁ = λ₂ = 0 through the whole growth-rate step. The multiplier's 1/n² part must vanish; that
fixes ω₂ (Δ₂), and second-order perturbation theory gives λ₂ (§4).

**Δ∞ with Δ₁ and Δ₂ held fixed.** On planted data, holding the true coefficients fixed shrinks the spread of Δ∞ from
6.8×10⁻⁹ to 2.5×10⁻¹¹. On the real points the derived values are too far off: Δ₁ must be known to about 3×10⁻⁶ for that
gain, and at M = 5 it is off by 0.027, so no fixed fit reaches the points' precision (§5).

**EEG's e^{−1/ε}** (§6, a reading): the inner problem's Stokes jump is $`e^{-n\Phi(x)}`$ in the outer variables, an
instanton-type term invisible to every order of their expansion. Its cancellation fixes Δ's power-series
coefficients order by order (Δ₁, Δ₂ here). In this reading it is not an additive term in Δ(D), for which the fits find
no evidence.

**Status.** EEG's check and the O(1/n²) equations are *checked symbolically*; the selection at O(1/n²) is
*conditional* on Conjecture 1 and the lemma chain one order further; the numbers are *numerical evidence* from truncated
models, whose M → ∞ limit is not computed (truncation error at M = 5 about 2% for Δ₂ and 3% for λ₂).

## 1. EEG's expansion against Proposition 1

**Source.** EEG's supplement, section "Simplified large-D perturbation theory", Eqs. (S35)–(S51), read from the arXiv
PDF of v1 (the HTML conversion garbles several equations). Their GG gauge is ours. With
$`\varepsilon = 1/(D-1) = 1/(n+1)`$ and $`\beta = \sqrt\varepsilon\,\tilde\beta`$ (their Eq. (10)) they expand

```math
f = 1 - \varepsilon\,\tfrac{1-x^2}{2}\,\tilde\beta^2 + \sum_{k\ge2} f_k\varepsilon^k, \quad
\Omega = \varepsilon\tilde\beta^2 + \sum_{k\ge2}\Omega_k\varepsilon^k, \quad
\Pi = \varepsilon^{-1/2}\Big(\varepsilon\tilde\beta + \sum_{k\ge2}\Pi_k\varepsilon^k\Big), \quad
\Psi = \sqrt\varepsilon\Big(\varepsilon(\tilde\beta' + \tilde\beta) + \sum_{k\ge2}\Psi_k\varepsilon^k\Big),
```

primes being derivatives in τ, and print $`f_2, \Omega_2, \Pi_2, \Psi_2`$ (NLO, (S40)–(S43)) and
$`f_3, \Omega_3, \Pi_3`$ (NNLO, (S44)–(S51); "Ψ₃ = not needed"). From (S43) on they write β for β̃; we read it as β̃.

**The map.** EEG's equations (S36)–(S39) are our E1–E4 under (solver/supercrit/eqmap.py, *checked symbolically*)

```math
f = c, \qquad x^2\Omega = (n-1)(b-1), \qquad y = \sqrt2\,x\,\Pi, \qquad p + m = \sqrt2\,x^2\Psi,
```

with the same τ, so θ = ωτ and $`d/d\tau = \omega\,d/d\theta`$. Their free function is the exact central value,
$`\tilde\beta = \Pi(\tau,0)/\sqrt\varepsilon`$ (S34). In Proposition 1, $`D_0 = 2xF`$ and $`D_k = xK_k + O(x^3)`$, so

```math
\tilde\beta = \sqrt{2(1 + 1/n)}\;F_n, \qquad F_n = F + \tfrac12\sum_{k\ge1} n^{-k}K_k = \sqrt n\,\partial_x p\,(x = 0) .
```

EEG's period is implicit in τ-derivatives, ours enters through $`\omega = \omega_0 + \omega_1/n + \dots`$. Both
expansions are therefore the same family, parametrized by the exact central profile, and must agree coefficient by
coefficient once ε is re-expanded in 1/n. Their μ is RT's echoing parameter, related to Δ by the identity of
[RT_FORMULATION.md](RT_FORMULATION.md), $`\Delta = 4\pi\mu/(1 - \langle\Omega(\tau,1)\rangle)`$ (their Eq. (3),
with $`\langle\psi_{-1}^2\rangle = \langle x^2\Omega\rangle`$ at x = 1).

**The check** (`second_order_eeg.py`, sympy, 20 s). Proposition 1's recursion (written again in the script, with F,
K₁, K₂ and ω₀, ω₁, ω₂ symbolic) is mapped to EEG's variables and compared with their series re-expanded in 1/n
(check A); their series is also substituted into their own (S36)–(S39) (check B).

- **NLO: agrees.** f₂, Ω₂, Π₂ and Ψ₂ as printed equal Proposition 1 at every order (f and Ω to ε², Π and Ψ to ε^{3/2}
  and ε^{5/2}).
- **NNLO: the printed Π₃ is wrong, and Ω₃ and f₃ inherit it.** As printed, f₃, Ω₃ and Π₃ differ from Proposition 1
  (`--printed`, exit 1). Their own (S39) fails too: with the printed series it leaves a residual at ε^{5/2} (S36, S37
  and S38 hold to the order the printed terms fix). Π₃ enters (S39) at that order only through $`x^{-1}\partial_x\Pi`$,
  so the residual fixes the correction exactly, and Ω₃ (= 2β̃Π₃ + …, their (S45)) and f₃ (from (S37) with f(1) = 1)
  follow. The corrected NNLO is

```math
\Pi_3 = \frac{x^2}{2}\,\tilde\beta\,\big(4\tilde\beta^2 + 4\tilde\beta\tilde\beta' + \tilde\beta\tilde\beta'' + \tilde\beta'^2\big)
+ \frac{x^4}{8}\big(3\tilde\beta^5 - 24\tilde\beta^3 - 40\tilde\beta^2\tilde\beta' - 8\tilde\beta^2\tilde\beta'' - 8\tilde\beta\tilde\beta'^2
+ 24\tilde\beta + 50\tilde\beta' + 35\tilde\beta'' + 10\tilde\beta''' + \tilde\beta''''\big),
```

  with Ω₃ = 2β̃Π₃ + (the rest of (S45) unchanged) and $`f_3 = c_0 + c_2x^2 + c_4x^4 + c_6x^6`$,

```math
\begin{aligned}
c_0 &= -\tfrac{5}{16}\tilde\beta^6 - \tfrac{13}{12}\tilde\beta^4 + \tfrac{31}{24}\tilde\beta^3\tilde\beta' + \tfrac{7}{24}\tilde\beta^3\tilde\beta''
+ \tfrac{1}{12}\tilde\beta^2\tilde\beta'^2 + \tfrac{7}{12}\tilde\beta^2 - \tfrac{1}{12}\tilde\beta\tilde\beta' - \tfrac58\tilde\beta\tilde\beta''
- \tfrac{5}{12}\tilde\beta\tilde\beta''' - \tfrac{1}{24}\tilde\beta\tilde\beta'''' - \tfrac58\tilde\beta'^2 - \tfrac14\tilde\beta'\tilde\beta'' - \tfrac{1}{24}\tilde\beta''^2,\\
c_2 &= \tfrac{3}{16}\tilde\beta^6 + \tfrac34\tilde\beta^4 - \tfrac38\tilde\beta^3\tilde\beta' - \tfrac18\tilde\beta^3\tilde\beta'' \quad\text{(as printed)},\\
c_4 &= \tfrac{1}{16}\tilde\beta^6 + \tfrac54\tilde\beta^4 + \tfrac58\tilde\beta^3\tilde\beta' + \tfrac18\tilde\beta^3\tilde\beta''
+ \tfrac14\tilde\beta^2\tilde\beta'^2 - \tfrac74\tilde\beta^2 - \tfrac52\tilde\beta\tilde\beta' - \tilde\beta\tilde\beta'' + \tfrac14\tilde\beta'^2,\\
c_6 &= \tfrac{1}{16}\tilde\beta^6 - \tfrac{11}{12}\tilde\beta^4 - \tfrac{37}{24}\tilde\beta^3\tilde\beta' - \tfrac{7}{24}\tilde\beta^3\tilde\beta''
- \tfrac13\tilde\beta^2\tilde\beta'^2 + \tfrac76\tilde\beta^2 + \tfrac{31}{12}\tilde\beta\tilde\beta' + \tfrac{13}{8}\tilde\beta\tilde\beta''
+ \tfrac{5}{12}\tilde\beta\tilde\beta''' + \tfrac{1}{24}\tilde\beta\tilde\beta'''' + \tfrac38\tilde\beta'^2 + \tfrac14\tilde\beta'\tilde\beta'' + \tfrac{1}{24}\tilde\beta''^2 .
\end{aligned}
```

  With these, f to ε³, Ω to ε³, Π to ε^{5/2} and Ψ to ε^{5/2} equal Proposition 1, and (S36)–(S39) hold to the
  order the terms fix (default run, exit 0). The corrected x⁴ part of Π₃ carries the operator
  $`(D+1)(D+2)(D+3)(D+4)/8`$, D = d/dτ, where the printed one has $`(D+1)(D+2)(D+3)/3`$; Ψ₂'s x² part has
  $`(D+1)(D+2)(D+3)/2`$. *Checked symbolically.* Two independent routes give the same correction: Proposition 1's
  recursion, and EEG's own (S39).
- **The controls.** The mutations `c0` (S48's 17/48 → 18/48), `Pi3` (S46's 24 → 23), `Psi2` (S43's 11 → 10),
  `Omega3` (S45's −7 → −6) and `map` (ε = 1/n instead of 1/(n + 1)) each make the default run exit 1 (logs
  `second_order_eeg_mut_*.log`, summary `second_order_eeg_mutations.txt`).
- **What it changes.** EEG compare their perturbative fields with the D = 200 solution (their Fig. 4; EEG_PROFILE.md §1
  reads Fig. 5 the same way for D = 25–200). The corrected Π₃ changes Π at ε^{5/2} and Ω, f at ε³, that is at relative
  order ε² against the leading terms. Whether their figures used the printed expressions is not known from the paper.

**μ.** Expanding the gauge identity in ε (check C):

```math
4\pi\mu_0 = \Delta_0, \qquad 4\pi\mu_1 = \Delta_1 - \Delta_0\,N_0, \qquad
4\pi\mu_2 = \Delta_2^{(\varepsilon)} - \Delta_1 N_0 - \Delta_0\big(\langle 2\tilde\beta_0\tilde\beta_1\rangle - \langle\tilde\beta_0'^2\rangle - \langle\tilde\beta_0^4\rangle\big),
```

with $`N_0 = \langle\tilde\beta_0^2\rangle`$ (their (S59), *checked symbolically*) and
$`\langle\Omega_2(\tau,1)\rangle = -\langle\tilde\beta'^2\rangle - \langle\tilde\beta^4\rangle`$ from (S41). With
N₀ = 2⟨F²⟩ = 7.416425 of the tuned D = ∞ profile (EEG_PROFILE.md §3.5), Δ∞ = 1.643893746 and Δ₁ = 10.37528:
μ₀ = 0.1308169 and μ₁ = −0.14456, against EEG's fits 0.130818 and −0.1449 (*numerical evidence*; the 0.2% in μ₁ is
within what their printed digits of Δ₁ allow).

### 1.1 EEG's NNLO against the finite-D solutions (issue #126)

**Verdict.** The comparison used 63 states from D = 15 to 215, with β̃ the exact central profile of each state, as
EEG's (S34) fixes it. The corrected NNLO's maximal residual on 0 ≤ x ≤ 1 falls like ε⁴ for f and Ω and like ε^{7/2}
for Π. The printed NNLO's residual falls like ε³ and ε^{5/2}, the orders of its first wrong terms. This confirms the
symbolic finding above numerically. The code shares only EEG's coefficients and the correction with
`second_order_eeg.py`, and none of Proposition 1's recursion. *Numerical evidence.*

In EEG's own figures the error is hard to see. Those figures put the D-independent profile of their Eq. (16) into the
NNLO. The rms deviations quoted in Fig. 5 move by about 3% or less, except f at D = 25 (25%) and 50 (9%). The
surfaces of Fig. 5 differ by up to 18% of max |Ω| at D = 25, 5% at D = 50 and 0.4–0.5% at D = 200. In the centre
profiles of Fig. 4 only f differs, by 0.18% of max |1 − f| at D = 200. With the exact profile the printed NNLO's rms
deviation is 2.2 times the corrected one's at D = 60, and 6–7 times for Ω and Π at D = 200.

**Fields from a state** (`nnlo_vs_solution.py extract`):

- **Representations.** Centre series in double-double for x ≤ 0.3, Taylor steps in double-double from the nearest node
  for 0.3 < x < 0.7, and the horizon series in quad-double for x ≥ 0.7 (hp/src/model.rs layout, as
  hp/largeD_profiles.py).
- **The horizon wall.** The double-double series (`Solver.series`) loses every digit near x = 1 at D = 200: it misses
  the node at 0.7 by 7×10⁻³ in p. So the series is evaluated as the horizon block of `Solver.residual_q` on a two-node
  solver whose node states are zero. There it reproduces the stored node at 0.7 to 5×10⁻²⁶.
- **The map.** f = c, Ω = (n − 1)(a² − 1)/x² (b = a² in model.rs), Π = (p − m)/(√2 x) and Ψ = (p + m)/(√2 x²), with
  ε = 1/(D − 1). Below x = 0.3 the divisions by x are done on the series coefficients, and a − 1 is formed part by part.
- **The profile.** β̃ = Π(τ, 0)/√ε. Its τ derivatives are spectral, with ω = 2π/Δ of the state.
- **The grid.** x = 0, 0.01, …, 1, and the state's 32 τ points per half period.

The states are the double-double states of hp/data/states (D 15–60, full system) and T81's quad-double state at 73.
They also include the held-on-fit chain T138-win2 (`chain/` 80–170, `chain_o/` 180–215), whose held harmonics are
pinned (HELD_HARMONICS.md). Each state's own error is at most 10⁻⁶ of the residual R it is compared with: continuity at
the nodes 5×10⁻¹², and the quad-double horizon block 8×10⁻¹⁹. Other chains at equal D (T140-win d6 and d10, I13-win)
give residual fields up to 6.8% of R apart at D = 180. Their held harmonics differ along near-null directions that the
states' residual (10⁻²²) does not fix, and the high τ derivatives of the NNLO amplify them. By the criterion written
first (C6, raw README), D = 150 and 180 leave the fit. With D ≤ 140 only, the exponents move by at most 0.04.

**Exponents** of R(D) = max |F_sol − F_model|. The fit is ln R = a + p ln ε + b ε over D ≥ 40. The expected value
is the first omitted or wrong order. Each verdict asks for agreement within 0.25 and for a fit farther than 0.5 from
one order lower (criteria C1–C3, written before the runs):

| model | f | Ω | Π | Ψ | expected (f, Ω, Π, Ψ) | verdict |
|---|---|---|---|---|---|---|
| LO | 1.986 | 1.982 | 1.493 | 2.490 | 2, 2, 1.5, 2.5 | pass |
| NLO | 2.957 | 2.975 | 2.484 | 3.473 | 3, 3, 2.5, 3.5 | pass |
| NNLO printed | 3.044 | 2.986 | 2.484 | 3.473 | 3, 3, 2.5, 3.5 | pass |
| NNLO corrected | 3.971 | 3.967 | 3.442 | 3.473 | 4, 4, 3.5, 3.5 | pass |

The local exponent between the two largest D is 3.998, 3.986, 3.469 and 3.453 for the corrected NNLO, and 3.171,
3.111, 2.493 and 3.453 for the printed one. The fits over D ≥ 73 agree with the main fit to 0.03. Over D ≥ 20 they
agree to 0.05 for both NNLO, and to 0.08 for NLO's f. Ψ₃ is "not needed" in EEG, so Ψ is the same in both versions.

**The size of the terms.** R/ε^p agrees between D = 60 and 200 to 8% or better. For Π it is 51, 825 and 2.8×10⁴ at
LO, NLO and NNLO. For Ω it is 320, 6.0×10³ and 1.6×10⁵, and for f 105, 1.2×10³ and 2.4×10⁴. So each order costs a
factor of 12–34, and the expansion's effective parameter is about 25ε. A reading of the cause: each order brings τ
derivatives of β̃, that is powers of νω on the harmonics ν = 3, 5. Below D ≈ 37 for Π, 31 for Ω and 22 for f, the
corrected NNLO's maximal residual is larger than NLO's. At D = 25 its Π residual is 0.46 against NLO's 0.29. At D =
200 the printed Π₃ removes only 5% of NLO's maximal Π residual (783 against 825 ε^{5/2}). The corrected Π₃ removes 83%.

**Controls.**

- **Planted truth (C5), run first.** Each state's fields were replaced by the corrected NNLO of its own β̃ plus
  (1 + 4ε + 20ε²)ε^p g. The same code returned p = 4.008, 4.006, 3.491 and 3.492 against the planted 4, 4, 3.5 and
  3.5. It returned 2.991 for a plant at ε³ in Π. The printed NNLO's fit on the planted data was 2.95, 2.92 and 2.52.
  The mutation broke the plant. All runs pass.
- **Π₃ × 1.01 (C4) did not fire on the solutions.** The Π exponent moved from 3.442 to 3.386 and Ω's from 3.967 to
  3.942, inside the tolerance (exit 1). The cause is the size of the true ε^{7/2} term, whose coefficient is about 30
  times Π₃'s. So 1% of Π₃ is only 2–7% of R over D = 40–215, and it would dominate only near D ≈ 3000. A planted run
  whose plant is as large as the real residual (`--plant_scale=1000`) stays just as silent, while the default plant is
  1000 times smaller and lets the control fire. The test resolves Π₃ × 1.05 (Π 3.22) and × 1.2 (Π 2.94, Ω 3.63). The
  printed Π₃'s error is of the order of Π₃ itself, far above that resolution. The tolerance was not changed.

**Figs. 4 and 5.** Relative rms deviation rms(F_sol − F)/rms(F_sol) over the (τ, x) grid, for f relative to
rms(1 − f), printed / corrected. The profile is either EEG's Eq. (16) model, A* = 3.8455, q = 0.0551, δ = 0.0985,
phase-aligned on the fundamental with θ = 2πτ/Δ(D), or the exact profile. The last column is
max |F_printed − F_corrected| over max |F_sol| with Eq. (16), and for f over max |1 − f|:

| D | Eq. (16): f | Ω | Π | exact: f | Ω | Π | max change, Eq. (16): f, Ω, Π |
|---|---|---|---|---|---|---|---|
| 25.08 | 0.148 / 0.186 | 0.308 / 0.318 | 0.185 / 0.188 | 0.589 / 0.400 | 0.266 / 0.178 | 0.105 / 0.088 | 0.115, 0.181, 0.132 |
| 50.01 | 0.145 / 0.158 | 0.180 / 0.185 | 0.105 / 0.105 | 0.065 / 0.034 | 0.038 / 0.019 | 0.021 / 0.011 | 0.028, 0.056, 0.051 |
| 60.00 | 0.132 / 0.141 | 0.155 / 0.158 | 0.089 / 0.090 | 0.040 / 0.018 | 0.024 / 0.011 | 0.014 / 0.0062 | 0.020, 0.041, 0.039 |
| 100.0 | 0.092 / 0.095 | 0.100 / 0.101 | 0.056 / 0.056 | 0.011 / 0.0035 | 0.0075 / 0.0023 | 0.0050 / 0.0014 | 0.0070, 0.016, 0.016 |
| 200.0 | 0.051 / 0.052 | 0.053 / 0.053 | 0.029 / 0.029 | 0.0023 / 0.00041 | 0.0017 / 0.00029 | 0.0012 / 0.00018 | 0.0018, 0.0044, 0.0046 |

All entries are measured on states, none extrapolated (Fig. 5's D are 25, 50, 100 and 200). With Eq. (16) the
deviations are set by the profile. The D-independent profile misses the exact one by 5–6ε in rms (3.0% at D = 200,
21% at D = 25). The printed and corrected curves then differ by little compared with the deviation. Our rms definition
and the phase alignment are assumptions, since EEG give no formula (Fig. 5 caption: "relative RMS deviations over the
(τ,x) domain"). With f normalized by rms(f) instead, its Eq. (16) entries at D = 25 are 0.0175 and 0.0220. Fig. 4
shows the profiles at x = 0, where Π₃ and Ω₃ vanish and Ψ₃ is not used. There only f differs, through c₀. At D = 200
the change is 1.8×10⁻³ of max |1 − f|, against a deviation of 4.4×10⁻² from the solution with Eq. (16). With the exact
profile the change is 2.0×10⁻³, and the printed and corrected deviations are 2.2×10⁻³ and 4.8×10⁻⁴.

**The D = ∞ profile in the corrected NNLO** (the issue's extended scope, a first step). The tuned D = ∞ profile
(EEG_PROFILE.md §3.3; #133's rational fits reproduce it to 10⁻⁶) goes into the corrected NNLO. Its relative rms
residual at D = 40, 60, 100 and 200 is 0.22, 0.16, 0.10 and 0.053 for Ω, and 0.127, 0.090, 0.056 and 0.029 for Π.
These are within 2% of Eq. (16)'s, because both are set by the profile's own 1/n dependence. A closed form accurate to
O(ε³) also needs the profile's 1/n terms, F_n = F + K₁/(2n) + K₂/(2n²) + … (§1), which are known here only in the
M-harmonic truncation.

**The issue's wording.** #126 asks the corrected NNLO's residual to fall like ε³ (Ω, f) and ε^{5/2} (Π). Those are
the printed one's orders. For the absolute residual the corrected one falls one order faster, as measured. Relative to
the leading terms, which are O(ε) for 1 − f and Ω and O(√ε) for Π, the corrected residual is O(ε³) for all three.

Raw output: folder `I126-mac/` of the raw-data repository (fields of every state, logs, JSON, the commands in
`*.cmd`, exits in `exits.txt`).

### 1.2 The closed-form large-D solution (agent-comms#42)

**Verdict.** The corrected NNLO with Proposition 1's profile to O(1/n²),

```math
\tilde\beta_D(\theta) = \sqrt{2(1 + 1/n)}\,\Big(F + \frac{K_1}{2n} + \frac{K_2}{2n^2}\Big), \qquad
\theta = \frac{2\pi\tau}{\Delta(n)}, \qquad \Delta(n) = \Delta_\infty + \frac{\Delta_1}{n} + \frac{\Delta_2}{n^2},
```

gives f, Ω, Π and Ψ on 0 ≤ x ≤ 1 from three D-independent functions and three numbers. Its relative rms residual for
Ω against the stored solutions is 0.039, 0.011, 0.0023 and 2.8×10⁻⁴ at D = 40, 60, 100 and 200. It falls like
ε^{3.02}, ε = 1/(D − 1). This is the corrected NNLO's own floor: with each state's exact profile the residual is 0.038,
0.011, 0.0023 and 2.9×10⁻⁴. With K₁ alone the exponent is 2.46, and with F alone 0.88. All criteria written before the
runs pass (the started comment of agent-comms#42). F is the tuned D = ∞ profile, so the closed form is *conditional* on
the selection rule (Conjecture 1). K₁ and K₂ are fitted from the states' central profiles, so they are *numerical
evidence*. With F fitted as well, the closed form uses no D = ∞ input, and its exponent is 3.05.

**The closed form's data** (`solver/inner_pert/data/closed_form_K.json`, every odd ν ≤ 31). The coefficients c_ν are
those of $`b = \sum_\nu 2\,\mathrm{Re}(c_\nu e^{i\nu\theta})`$, in the gauge where the fundamental of every profile is
a positive multiple of sin θ (F₁ = −iA/2). F is the J 1500 block of
`solver/eeg_profile/data/tuned_profile_A2.71918.txt`. Each ± is the largest deviation over nine fit choices (a spread
over fit choices, below).

| ν | F_ν | K₁,ν | K₂,ν |
|---|---|---|---|
| 1 | −1.35959i | −14.529i ± 0.007 | 32.84i ± 1.8 |
| 3 | 0.007383 − 0.074488i | −0.6247 − 4.9710i ± 0.0018 | 5.33 − 11.21i ± 0.36 |
| 5 | 0.000756 − 0.004212i | −0.0294 − 0.7766i ± 0.0004 | −0.249 − 11.07i ± 0.09 |
| 7 | 0.0000613 − 0.0002377i | 0.0051 − 0.0869i ± 0.0002 | −0.129 − 2.955i ± 0.07 |

So the fundamental's amplitude is A(n) = 2.71918 + 14.529/n − 32.84/n² + …. The period is Δ∞ = 1.643893746 (README),
Δ₁ = 10.3753 (FIRST_ORDER.md, the fit row) and Δ₂ = −49.1685 (Summary above), all from the double-double fits. The time
origin is aligned on each state's fundamental, one gauge number per state. K₂'s higher harmonics are large next to
F's: |K₂,₇|/(2n²) exceeds |F₇| for n below about 78, so the 1/n series of the high harmonics converges slowly. Their
absolute size stays small, 7×10⁻⁴ of |F₁| at D = 40.

**Method** (`solver/inner_pert/nnlo_closed_form.py`, which imports `nnlo_vs_solution.py`):

- **Profiles.** For each of the 63 states of §1.1, $`F_n = \tilde\beta/\sqrt{2(1+1/n)}`$ with β̃ = Π(τ, 0)/√ε. Its
  harmonics are taken by FFT and rotated in time so that the fundamental is −i|c₁|.
- **The fit.** Harmonic by harmonic, $`F_{n,\nu} - F_\nu = \sum_{k=1}^{6} a_k n^{-k}`$ by least squares over the
  states with D ≥ 20, leaving out the four test states, and K_k = 2a_k. K₃–K₆ are fitted and dropped.
- **The closed form.** β̃_D as above, rotated back to the state's time origin, its τ derivatives taken spectrally with
  ω = 2π/Δ(n), and put into `model_fields` (corrected, order 3).
- **The measure.** R = rms(F_sol − F_cf)/rms(F_sol) over the (τ, x) grid (f relative to rms(1 − f)), as in §1.1's
  Fig. 5 measure. Its exponent p is the least-squares slope of ln R against ln ε over the four test D. The profile's
  own error is rms(β̃_cf − β̃_sol)/rms(β̃_sol).

**Results.** R(Ω), its exponent, and the profile's own error at D = 200 with its exponent:

| β̃ in the corrected NNLO | D 40 | D 60 | D 100 | D 200 | p(Ω) | profile, D 200 | p(profile) |
|---|---|---|---|---|---|---|---|
| √2 F (agent-comms#5, §1.1) | 0.222 | 0.158 | 0.101 | 0.0530 | 0.88 | 0.030 | 0.94 |
| F | 0.203 | 0.145 | 0.0923 | 0.0484 | 0.88 | 0.028 | 0.94 |
| F + K₂/2n² (the control K₁ = 0) | 0.216 | 0.151 | 0.0946 | 0.0490 | 0.91 | 0.028 | 0.96 |
| F + K₁/2n | 0.0393 | 0.0125 | 0.00339 | 7.1×10⁻⁴ | 2.46 | 3.1×10⁻⁴ | 1.81 |
| the closed form | 0.0388 | 0.0108 | 0.00226 | 2.81×10⁻⁴ | 3.02 | 2.5×10⁻⁵ | 2.84 |
| the closed form, K₂ × (−1) | 0.0499 | 0.0177 | 0.00564 | 1.35×10⁻³ | 2.20 | 6.5×10⁻⁴ | 1.89 |
| the closed form, F fitted too | 0.0409 | 0.0110 | 0.00227 | 2.80×10⁻⁴ | 3.05 | 2.2×10⁻⁵ | 2.87 |
| each state's exact profile (§1.1) | 0.0383 | 0.0108 | 0.00229 | 2.85×10⁻⁴ | 3.01 | 0 | |

The other fields of the closed form, at D = 40, 60, 100 and 200:

| field | D 40 | D 60 | D 100 | D 200 | p |
|---|---|---|---|---|---|
| f (relative to rms(1 − f)) | 0.069 | 0.017 | 0.0033 | 3.8×10⁻⁴ | 3.20 |
| Π | 0.021 | 0.0062 | 0.0014 | 1.8×10⁻⁴ | 2.93 |
| Ψ | 0.087 | 0.039 | 0.014 | 0.0035 | 1.96 |

Ψ₃ is "not needed" in EEG, so Ψ is right to relative order ε² only, with any profile. Since Ω ∝ β̃², twice the
profile's own error estimates its share of R(Ω). With K₁ and K₂ that is 13–18% of the NNLO floor, so the residual is
the NNLO's. With K₁ alone the missing K₂ term matches the floor at D = 100 and dominates above it.

**Error bars.** The spread over fit choices (k_max = 5, 6, 7 and D_lo = 15, 20, 30) moves p(Ω) within 3.01–3.27 for the
closed form, 2.45–2.48 with K₁ and 0.91–0.92 with K₁ = 0. Every choice passes C1, C2 and C3. The high values of the
closed form come from a larger residual at D = 40 (up to 0.059 at k_max 7, D_lo 30), where the fitted K₂ of the
higher harmonics is least determined. At D = 200 the residual moves by 1.5%. The fundamental's 1/n coefficients range
over A₁ = 14.522–14.534 and A₂ = −34.63 to −31.39.

**Controls** (commands as run in `planted.cmd` and `compare.cmd` of raw `AC42-mac/`, exits in its `exits.txt`):

- **Planted truth, run first.** Each state's fields were replaced by the corrected NNLO of a planted profile, a
  rational function of 1/n whose series does not terminate, sized from the README's profile table, shifted in time by
  a known phase per state, plus §1.1's planted remainder. The same code recovered K₁ to 3×10⁻⁵ and K₂ to 1.1×10⁻³ for
  ν = 1, 3. Its exponents were 0.914 (K₁ = 0), 1.926 (K₁), 3.193 (K₁ and K₂) and 1.998 (K₂ × (−1)), each within 0.044
  of those with the planted K. The mutations `k1x0`, `k2x-1` and `k2x1.1` are detected. All exits 0. With the remainder
  at the real floor's size (`--plant_scale=740`, information) the K₁ exponent rises to 2.70 and the closed form's is
  3.22. The floor then hides the missing K₂ term at low D, as on the real data.
- **K₁ = 0 (`--mutate=k1x0`) and K₂ × (−1) (`--mutate=k2x-1`)** fire: 0.913 ≤ 1.2 and 2.202 < 2.6. F alone gives 0.882.
- **The pipeline.** √2 F reproduces agent-comms#5's numbers (I126-mac/figs.json) to 3×10⁻¹⁵.
- **What the test resolves** (scan, information). The closed form's exponent falls below 2.6 for K₁ × 0.99, 1.02 and
  1.05, and stays above it for K₁ × 1.01 (2.606). For K₂ it falls below only at × 2 (2.548) and × (−1). It stays above
  at × 0.5 (2.751), 1.1 (3.005) and 1.5 (2.784). So the test resolves K₁ to about 1% and K₂ only to its own size. The
  fitted K₂'s spread of 5% is far inside that.

**Cross-checks** (information):

- **The D = ∞ profile.** With the intercept fitted too, the states extrapolate to A = 2.719209 (spread over fit choices
  2.7192055–2.7192530), against the tuned profile's 2.71918. That value is EEG's A*/√2. The M = 7 zero is 2.71922
  (EEG_PROFILE.md 3.5). Harmonics 3 and 5 of the fitted F agree with the tuned ones to 1.3×10⁻⁵ and 8.7×10⁻⁵. So the
  finite-D states extrapolate to the tuned profile's shape and to an amplitude about 1×10⁻⁵ above the paper's.
- **K₁ from the inner problem.** The fitted A₁ = 14.529 (14.519 with F fitted) against the M = 5 inner run's 14.4532
  (FIRST_ORDER.md 6, which writes F_n = F + K₁/n). The M = 5 value is 0.5% low, half the 1% that the test resolves.
  The M = 5 run shares none of the finite-D machinery.
- **The period.** With each state's own Δ in place of Δ(n) the exponent is 3.019, against 3.023.
- **Chains at equal D.** At D = 100, the T140-win chains d6 and d10 change R(Ω) by 1.3×10⁻⁷ and 3.0×10⁻⁷ relative.

**What it shares and what it does not show.** The closed form shares with §1.1 the fields of the states and
`nnlo_vs_solution.py`'s NNLO coefficients. Its K₁ and K₂ come from the same family of states it is compared with. The
four test states are left out of the fit, but their neighbours are not. So this shows that three D-independent functions
describe the solutions to O(ε³) relative across D = 40–200. It is not a prediction from the field equations. A
first-principles K₁ and K₂ need the inner problem at O(1/n) and O(1/n²) with enough harmonics (FIRST_ORDER.md, §4).
The chain states above D = 73 hold p1's harmonics 5–13 at `hp/data/held_fit.json`, a fit over D 38–73. For those
harmonics above D = 73 both the fit and the comparison see that fit's extrapolation.

**Proposed README text** (for the coordinator, near Proposition 1 or the relation to EEG):

> *A closed-form large-D solution.* EEG's NNLO with the correction above, with β̃ = √(2(1 + 1/n)) (F + K₁/2n + K₂/2n²)
> from Proposition 1, gives f, Ω, Π and Ψ on the whole interior from three D-independent functions. F is the tuned
> D = ∞ profile, and K₁, K₂ are fitted from the solutions' central profiles. Its relative rms error in Ω is 3.9%,
> 1.1%, 0.23% and 0.028% at D = 40, 60, 100 and 200 and falls like ε³, as with each solution's exact profile
> (notes/SECOND_ORDER.md 1.2). It is *conditional* on the selection rule for F, and *numerical evidence* for K₁, K₂.

**Reproduction.** From `solver/inner_pert/`, with the raw-data repository cloned beside the checkout (`git clone
https://github.com/nmoinvaz/choptuik-large-d-raw ../choptuik-large-d-raw`), each command takes under a second:

```
python3 nnlo_closed_form.py planted ../../../choptuik-large-d-raw/I126-mac/fields/fields_D*.npz
python3 nnlo_closed_form.py compare ../../../choptuik-large-d-raw/I126-mac/fields/fields_D*.npz \
    --figs=../../../choptuik-large-d-raw/I126-mac/figs.json      # C1, C2, C5: exit 0
python3 nnlo_closed_form.py controls ../../../choptuik-large-d-raw/I126-mac/fields/fields_D*.npz \
    --figs=../../../choptuik-large-d-raw/I126-mac/figs.json      # C3, C4: exit 0
```

Raw output: folder `AC42-mac/` of the raw-data repository, at its commit 443c504f (logs, JSON, the commands in `*.cmd`,
exits in `exits.txt`).

## 2. The inner equations to second order

In the inner variables the exact equations close with one slaved function, u = (n − 1)(b − 1)
(`second_order_equations.py`, *checked symbolically*, exactly, with e = 1/n and
$`L = \xi^2\partial_\xi + \xi\partial_\tau`$):

```math
L y = c\sigma + e\,c\,(\xi\sigma_\xi - \sigma + u\sigma), \qquad
L\sigma = (\xi\partial_\xi - 1)(c\,y), \qquad
\xi c_\xi = c\,u, \qquad
e\,\xi u_\xi = (1 - e + e u)(P - u),\quad P = \tfrac12(y^2 + e\sigma^2) .
```

Only the u equation is singular in e (its fast mode is $`b \sim x^{-n}`$), so u is slaved,
$`u = E + e\Psi + e^2\Phi + O(e^3)`$ with

```math
E = \tfrac12 y^2, \qquad \Psi = \tfrac12\sigma^2 - \xi y y_\xi, \qquad \Phi = -\xi\Psi_\xi - (1 - E)\,\xi E_\xi ,
```

and the inner equations to O(1/n²) are

```math
L y = c\sigma - e\,c\,(\sigma - \xi\sigma_\xi - E\sigma) + e^2\,c\,\Psi\sigma, \qquad
L\sigma = (\xi\partial_\xi - 1)(c\,y), \qquad
\xi c_\xi = c\,(E + e\Psi + e^2\Phi) .
```

The O(e) part is FIRST_ORDER.md §1. The script checks that the four exact equations are the outer ones in inner
variables, that the slaved u solves its equation to O(e³), that the system above is the exact one expanded to
O(n^{−2}) with no odd powers of n^{−1/2}, and that the time reparametrization t → T(t), c → T′c is an exact symmetry of
the exact system (in ρ = r/√n, t). Its control `--mutate=phi` ((1 − E) → 1 in Φ) exits 1.

The data of the O(1/n²) inner solution $`Z_2`$ are, as at first order, the profile correction K₂
($`F_n = F + K_1/n + K_2/n^2`$), ω₂, and the light-cone gauge. The outer normalization c(x = 1) = 1 is
$`c(\xi = n^{-1/2}) = 1`$, which, with $`c_0 = 1 + g_1\xi^2 + g_2\xi^4 + \dots`$ and $`c_1 = h_0 + h_1\xi^2 + \dots`$,
gives

```math
c_1(0) = -g_1, \qquad c_2(0) = -g_2 - h_1 .
```

In the outer polynomials this is $`-G_{2,0} = G_{2,1} + G_{2,2}`$, i.e. $`G_2(1) = 0`$. For a perturbation the same
holds linearized: $`\delta c_2(0) = -\delta g_2 - \delta h_1`$. In the full problem these data drop out (the
reparametrization symmetry above, for growth rates λ ≠ 1); in a truncated model they do not, and their effect measures
the truncation.

## 3. Second-order jets and their checks

`second_order_jet.py` extends the jets of `first_order_jet.py` by polarization: two nilpotent directions e₁, e₂ with
e = e₁ + e₂, so e² = 2e₁e₂ and the mixed component e₁e₂ holds twice the 1/n² part of every field, forcing and jump.
The O(e) terms are shifted by each eᵢ, the O(e²) terms by the pair, times 2. Further features: ω-jet entries on
composite components (2ω₂ on e₁e₂), a λ-jet for the perturbation components (λ₁ on e₁ and e₂, 2λ₂ on e₁e₂), and a
fixed-point pass per order of e in the right-hand side (σ_ξ enters the O(e) term of the y equation and y_ξ that of
σ). Φ needs $`y_{\xi\xi}`$, taken as the ξ-derivative of the model's own y equation, so every product in the
right-hand side is the same truncated product as in the series recursion and the series is the model's formal
solution.

**Checks** (`second_order_checks.py`, exit 1 on failure; each control seen to fail, logs `second_order_checks*.log`):

| check | independent of the jets in | result (threshold) | mutation → result |
|---|---|---|---|
| O(1/n) inner series = next-to-top outer coefficients, F = 2.72 sin θ, orders ≤ 7 | outer recursion (`largeD_late_orders.py`) | 1.2×10⁻¹³ (10⁻¹⁰) | |
| O(1/n²) inner series (e₁e₂/2) = next-to-next-to-top outer coefficients, orders ≤ 7 | the same | 6.5×10⁻¹³ (10⁻¹⁰) | `eps2y` 1.1×10⁻²; `phi` 3.8×10⁻³; `pol` (e² = e₁e₂) 1.6×10⁻²; `gauge2` (c₂(0) = 0) 3.6 |
| right-hand side = derivative of the summed series, every feature at once, M = 3 | series recursion vs right-hand side | 4.2×10⁻⁹ (10⁻⁸) | `rhs2` 0.58 |
| the θ-shift mode (λ = 0, f = F_n′, exact at every order) keeps λ₁ = λ₂ = 0, M = 3 | the mode is exact by symmetry | λ₁ = 2.1×10⁻⁶ (10⁻⁴), λ₂ = −3.8×10⁻⁴ (4×10⁻²) | `pgauge2` (δc₂(0) dropped) λ₂ = −21.6 |

The outer comparison checks the O(1/n²) equations, the slaving, the polarization factor and the gauge formula
c₂(0) = −g₂ − h₁ at once. The θ-shift check runs the whole λ₂ step (f₁, the perturbation's gauge data, the
O(1/n²) perturbation jump) on a mode whose answer is known. The λ = 1 mode (time translation) is reported but not
checked: in the light-cone gauge it is not exact at finite n, since the mean of its δc(x = 1) is not a
reparametrization of the form $`e^{\tau}g(\theta)`$; the M = 3 model gives it λ₁ = −7.55, close to
−2⟨F²⟩ = −7.45 of that model, which is what a shift by $`-\langle c_x(1)\rangle`$ would give (not shown).

## 4. The selection at second order: Δ₂ and λ₂

**The condition.** As at first order (FIRST_ORDER.md §3), the multiplier of the corrected inner solution must vanish
order by order. With $`S(Z_0) = 0`$ and $`S^{(1)} = 0`$ (the first-order selection), the matching factors' 1/n and 1/n²
corrections multiply vanishing vectors, so the light-cone coefficient vanishes to O(1/n²) exactly when

```math
S^{(2)} = S_{\rm rest} + \frac{\partial S}{\partial x}\,\delta x_2 = 0, \qquad \delta x_2 = (A_2, \omega_2/\omega_\infty, B_{3,2}, \dots),
```

where $`S_{\rm rest}`$ is the 1/n² part of the multiplier with the selected first-order data (K₁, ω₁, c₁(0) = −g₁),
the gauge c₂(0) = −g₂ − h₁, and K₂ = ω₂ = 0. In the jets $`S_{\rm rest}`$ is half the jump of the e₁e₂ component. Then

```math
\omega = \omega_* + \frac{\omega_1}{n} + \frac{\omega_2}{n^2}, \qquad
\Delta_2 = 2\pi\Big(\frac{\omega_1^2}{\omega_*^3} - \frac{\omega_2}{\omega_*^2}\Big) .
```

The growth rate is second-order perturbation theory of the perturbation map $`\mathcal S(\lambda, 1/n)`$ (FIRST_ORDER.md
§4, direct route): $`f = f_0 + f_1/n + f_2/n^2`$ with $`f[0] = 1`$ at every order, and λ a jet. First f₁ solves
$`\mathcal S(\lambda_*)f_1 = -J_{p1}`$, $`J_{p1}`$ being the jump of the perturbation's 1/n component with f₁ = 0 (it
holds $`\mathcal S_1 f_0 + \lambda_1\partial_\lambda\mathcal S f_0`$, so $`\ell\cdot J_{p1} = 0`$ is a check of λ₁).
Then, with the perturbation's gauge data, the jump $`J_{p12}`$ of its 1/n² component (f₂ = 0, λ₂ = 0) gives

```math
\lambda_2 = -\,\frac{\ell\cdot J_{p12}/2}{\ell\cdot\partial_\lambda\mathcal S(\lambda_*)\,f_0} .
```

The jets contain every second-order term ($`\mathcal S_2 f_0`$, $`\mathcal S_1 f_1`$,
$`\lambda_1\partial_\lambda\mathcal S f_1`$, $`\lambda_1\partial_\lambda\mathcal S_1 f_0`$,
$`\tfrac12\lambda_1^2\partial_\lambda^2\mathcal S f_0`$) without hand derivation. M = 1, 3 use one-shot jets
(`second_order_run.py`); M = 5 the deflated directions of `first_order_hier.py` (`second_order_hier.py`): the e₁e₂
component, and the perturbation's 1/n and 1/n² components, get higher-harmonic profile data refined until their own
higher-harmonic jump stops falling (e₁e₂: from 1.9×10¹³ to 4 at M = 5), and the harmonic-1 condition is solved with the
deflated A and ω directions.

**Checks in the runs.** The first-order selected component has zero jump (M = 3: 9×10⁻⁴ against 3×10¹¹ for the
second-order component before selection); entering δx₂ as data brings the e₁e₂ jump from 3.5×10¹¹ to 0.39 (M = 3);
$`\ell\cdot J_{p1}`$ = −7×10⁻⁴ against $`\lambda_1\,\ell\cdot\partial_\lambda\mathcal S f_0`$ = 2×10⁵ (M = 3), so the
jets reproduce λ₁ of first_order_run.py; the gauge-0 runs reproduce FIRST_ORDER.md's gauge-0 λ₁ (0.740 at M = 1, 2.8268
at M = 3).

**Results** (`second_order_table.py`, from the JSON files in the raw folder):

| | M = 1 | M = 3 | M = 5 | fit (double-double) |
|---|---|---|---|---|
| Δ* (n = ∞) | 2.115710 | 1.618638 | 1.645163 | 1.643893746 |
| Δ₁ (FIRST_ORDER.md) | 14.998 | 9.886 | 10.4023 | 10.37528 |
| ω₂ | 213.60 | 251.34 | 268.89 ± 0.42 | 266.57 |
| **Δ₂** | **−45.859** | **−44.422** | **−50.06 ± 0.18** | **−49.1685** [−49.16867, −49.16780] |
| Δ₂, every gauge datum 0 | −14.70 | −36.10 | −49.82 | |
| λ* (n = ∞) | 1.769122 | 2.006804 | 1.999669 | 2 |
| λ₁ (FIRST_ORDER.md) | 7.7566 | 3.0797 | 3.022 | 3.0324 |
| **λ₂** | **−141.05** | **−43.76** | **−33.37 ± 0.38** | **−34.38** (± about 0.1, §5) |
| λ₂, every gauge datum 0 | −20.87 | −38.86 | −33.69 (ξ₁ = 0.2) | |

The M = 5 values are means of three runs (ξ₁ = 0.2, 0.19 and ξ₁ = 0.2 at rtol 10⁻¹³: Δ₂ = −50.134, −50.195, −49.838; ω₂
= 269.08, 269.22, 268.38), the ± half their spread, a numerical error of the model value. The M = 3 values are identical
at ξ₁ = 0.2 and 0.175 to the digits shown. The fit's bracket for Δ₂ is the spread over fit choices; its ω₂ is
$`2\pi(\Delta_1^2/\Delta_\infty^3 - \Delta_2/\Delta_\infty^2)`$. Δ₂ in other variables (M = 5 mean against the fit):
−39.65 against −38.79 in EEG's ε = 1/(D − 1), −29.25 against −28.42 in 1/D.

**Convergence in M.** Differences of Δ₂ from the fit: +3.31 (6.7%), +4.75 (9.7%), −0.89 (1.8%) for M = 1, 3, 5;
from M = 3 to M = 5 the error changes sign and shrinks by a factor of 5 (ratio −0.19), as Δ₁'s does (−0.06) and λ₁'s
(−0.20). The fitted value lies between the M = 3 and M = 5 values, 16% of the way from M = 5, as for every
first-order quantity (5–17%). The readings of FIRST_ORDER.md §6 give M → ∞ limits of −49.79 (next error ratio
r = −0.05) and −49.12 (r = −0.2), against −49.17; like λ₁ at first order, Δ₂ follows the r = −0.2 reading, which is
also its own last ratio. This is a reading of three-term sequences, not an error bar.

**The gauge data** (§2) change Δ₂ by 31, 8.3 and 0.32 at M = 1, 3, 5: the truncation's breaking of the
reparametrization symmetry falls fast with M, and at M = 5 it is a third of the remaining distance from the fit
(0.89). Unlike λ₁ at first order, the two gauges do not bracket the fitted Δ₂ at M = 5 (−50.13 and −49.82 against
−49.17). The truncation error of the M = 5 value is therefore about 2%, from the M = 3 to M = 5 step, not from the
gauge spread.

**λ₂.** Differences from the fitted −34.38: −106.7, −9.39, +1.01 for M = 1, 3, 5 (2.9% at M = 5). From M = 3 to M = 5
the error changes sign and shrinks by a factor of 9 (ratio −0.11), and the fit lies between the two values, 10% of the
way from M = 5. The readings give −33.87 (r = −0.05) and −35.10 (r = −0.2). The three M = 5 runs give −33.69, −33.48 and
−32.94 (ξ₁ = 0.2, 0.19, and ξ₁ = 0.2 at rtol 10⁻¹³), so the numerical error, ±0.38, is a third of the distance from the
fit. Its source is the deflation of the perturbation's 1/n² component, whose own higher-harmonic jump falls from
1.2×10¹⁴ only to 11–42, so that its Schur correction carries part of the result, and λ₁, which enters with each run's
stored first-order value (3.0223, 3.0220, 3.0242), whose error shows as $`\ell\cdot J_{p1}`$ = −83 against
$`\lambda_1\,\ell\cdot\partial_\lambda\mathcal S f_0`$ = 1.8×10⁵. The gauge data change λ₂ by 120 and 4.9 at M = 1 and
3, and by 0.003 at M = 5 (−33.688 with every gauge datum 0 and the stored gauge-0 λ₁ = 3.0471), far below the numerical
error; at M = 3 the two gauges (−43.76, −38.86) do not bracket the fit.

## 5. Against the fits, and Δ∞ with Δ₁ and Δ₂ held fixed

**The fitted coefficients** (`second_order_fits.py`). Δ: the 1159 cleaned double-double points of today's
hp/data/delta_dd.csv (D = 8.5–67.8), least squares at 60 digits, polynomials of degree 12–18 in 1/n or 1/D on
D ≥ 10, 12, 16 (degree ≤ 16 on D ≥ 16, below). The 1/D fits give the 1/n coefficient through
$`\Delta_2 = \Delta_2^{(D)} - 2\Delta_1`$. The 15 fits that reach the points' precision (max |residual| ≤ 10⁻¹³) give

```math
\Delta_2 = -49.1685\ \ [-49.16867,\ -49.16780], \qquad \Delta_1 \in [10.375281,\ 10.375285],
```

the bracket being the spread over fit choices (the 1/n fits alone: −49.16857 to −49.16780; the 1/D fits: −49.16867
to −49.16783). In EEG's ε = 1/(D − 1), $`\Delta_2^{(\varepsilon)} = \Delta_2 + \Delta_1 = -38.793`$.
λ = 1/γ: the double-double growth rates (one per 0.5 in D, as first_order_fits.py), λ∞ = 2 fixed, degrees 10–11 on
D ≥ 24, 30: $`\lambda_2 \in [-34.3788,\ -34.3738]`$, an estimator range whose planted test failed (below).

**Planted truth** (`second_order_fits.py --planted`, criteria in its docstring, fixed before each first run; logs
`second_order_fits_planted*.log`). Synthetic data: the README's rational formulas at the same D (their Taylor
coefficients in 1/n known: Δ₂ = −49.1016, λ₂ = −34.4434), noise 10⁻¹⁴ on Δ and 10⁻¹² on λ.

- P1 (Δ₂ within 2×10⁻³ by every fit meeting the residual criterion): the first run FAILED on the two degree-18 fits on
  D ≥ 16 (errors 4.7×10⁻³ and 1.35×10⁻², overfitting the shortest range as degree 20 did at first order). Degree 18 was
  dropped on D ≥ 16, not the threshold changed; rerun: PASS, every passing fit within 4.8×10⁻⁴.
- P2 (Δ∞ with the planted Δ₁, Δ₂ held fixed, within 1.5×10⁻⁸ and a spread no wider than the free fits'): PASS. Three
  fixed fits reach the residual criterion, all within 1.4×10⁻¹¹ of the planted Δ∞, against a spread of 6.8×10⁻⁹ for
  the free fits.
- P3 (control, must fire: Δ₁ held 0.027 too high, the M = 5 derivation's distance from the fit): fires. No fixed fit
  reaches the residual criterion (max |residual| 1.7×10⁻⁹ to 1.2×10⁻⁶) and Δ∞ moves by 1.0–1.8×10⁻⁴.
- P4 (λ₂ within 0.02): FAILED twice. With degrees 9–10 the degree-9 fit on D ≥ 24 missed by 0.027 (all errors of one
  sign, falling with the degree); with degrees 10–11 (the estimator now in the script) the degree-11 fit on D ≥ 30
  missed by 0.091. The threshold was not changed. The real growth rates fit to max |residual| 3×10⁻¹⁴ at these
  degrees, thirty times below the planted noise, so the planted test is harsher than the data; that is not shown to
  make the real estimate good to 0.02. The fitted λ₂ is quoted as −34.38 with an uncertainty of about 0.1, the
  largest planted miss.

**Δ∞ with Δ₁ and Δ₂ held fixed** (real points). Held at the fitted centre (Δ₁ = 10.375283, Δ₂ = −49.1685), one fixed
fit reaches the residual criterion, Δ∞ = 1.64389375259, inside the free fits' range 1.64389374563–1.64389375483
(circular, as the fixed values come from the same points). Held at the derived values, no fit reaches it:

| held fixed | Δ₁ | Δ₂ | smallest max \|residual\| of the 11 fixed fits | Δ∞ of the fits |
|---|---|---|---|---|
| M = 5 | 10.4023 | −50.056 | 1.5×10⁻⁹ | 1.643739–1.643792 |
| M → ∞ reading, r = −0.05 | 10.378 | −49.787 | 6.3×10⁻¹² | 1.6438883–1.6438911 |
| M → ∞ reading, r = −0.2 | 10.316 | −49.117 | 3.7×10⁻⁹ | 1.644134–1.644279 |

(log `second_order_fits.log`). P3 shows why: Δ∞ moves by about 4–7×10⁻³ times the error of Δ₁. To keep Δ∞ within the
README's 1.5×10⁻⁸, Δ₁ would have to be known to about 3×10⁻⁶ (3×10⁻⁷ relative) and Δ₂ correspondingly; the M = 5 values
are 0.26% and about 2% off. So holding the derived coefficients fixed does not sharpen Δ∞ at this truncation; with exact
Δ₁ and Δ₂ it would (P2: spread 2.5×10⁻¹¹ instead of 6.8×10⁻⁹).

## 6. What EEG's e^(−1/ε) corresponds to here (a reading)

EEG's outlook expects that deriving Δ "will likely require to add instanton corrections of the form e^{−1/ε} to our
power-law expansion in ε". This is a stated reading, not a claim.

- **The power-law expansion cannot fix Δ** (Proposition 1, and §1: EEG's NLO and corrected NNLO are Proposition 1
  re-parametrized by the exact central profile, with the period free).
- **The exponentially small sector is the inner problem's Stokes jump.** The inner fast modes are
  $`e^{-\phi(\xi)/\xi^2}`$-type; at $`\xi = x/\sqrt n`$ they are $`e^{-n\Phi(x)}`$ in the outer region, with the
  singulant Φ of Proposition 4 and Lemma B5.1 ($`\Phi = -\tfrac12\ln(1 - x^{-2}) \pm i\pi/2`$ for x > 1, without
  gravity). With $`n = 1/\varepsilon - 1`$ these are $`e^{-\Phi/\varepsilon}`$ times powers: instanton-type terms with
  an x-dependent action, invisible at every order of EEG's expansion, and tied to its factorial divergence (Proposition
  4). The multiplier S is their coefficient; Conjecture 1 sets it to zero.
- **Here that sector fixes the power-series coefficients of Δ, order by order.** Δ∞ from S(Z₀) = 0, Δ₁ from S⁽¹⁾ = 0,
  Δ₂ from S⁽²⁾ = 0. The values agree with the fits to 0.26% (Δ₁) and about 2% (Δ₂) at M = 5, with the fits between the
  M = 3 and M = 5 values. In this reading EEG's instanton corrections are the Stokes jump of the inner problem, whose
  cancellation selects the period; they are not an additive term $`c\,e^{-a/\varepsilon}`$ in Δ(D) itself.
- **What the data say about an additive term.** The transseries test (DINF_FITS.md) finds no such term in Δ(D) for
  a ≥ 0.5: its coefficient is the scatter a planted c = 0 also gives, and Δ∞ moves by at most 1.3×10⁻⁷. The test does
  not cover a < 0.5 or factors of ln ε, and the selection condition imposed as a formal series in 1/n does not exclude
  exponentially small corrections to Δ(D) of its own (the multiplier's own subleading exponentials). Nothing found
  here needs one: Δ₁ and Δ₂ come out of the power-series selection within the truncation error.

## 7. What is rigorous, what is numerical, what is truncated

**Checked symbolically** (sympy): EEG's NLO equals Proposition 1; their printed NNLO does not, and the corrected NNLO
of §1 does, and satisfies their (S36)–(S39) (`second_order_eeg.py`); the exact inner equations, the slaved u, the
O(1/n²) system with only integer powers of 1/n, and the exact reparametrization symmetry (`second_order_equations.py`).

**Derived, and checked numerically to roundoff:** the O(1/n²) inner series, including the gauge c₂(0) = −g₂ − h₁,
equals the outer expansion's next-to-next-to-top coefficients to 6.5×10⁻¹³ (§3).

**Conditional**, on Conjecture 1 and on the lemma chain of C_selection.md carried to O(1/n²) (the outer solution equals
the inner lateral sum including Z₂ up to O(n^{−3+ε}), not written there): that the selection at O(1/n²) is the vanishing
of the multiplier of Z₀ + Z₁/n + Z₂/n², with the outer region contributing only invertible factors; and second-order
perturbation theory for λ (λ* simple, ℓ·∂_λ𝒮·f₀ ≠ 0).

**Numerical evidence**, for each truncated model: Δ₂ and λ₂ of the M = 1, 3, 5 models (§4), with the numerical
errors stated there.

**Truncated:** the M → ∞ limit is not computed. The Δ₂ and λ₂ sequences alternate from M = 3 on and bracket the fits,
like the first-order ones; three terms are a thin basis, and the truncation error at M = 5, estimated from the distance
between M = 3 and M = 5 and from the gauge dependence, is about 2% for Δ₂ and 3% for λ₂.

## 8. Discrepancies

Reported, not fixed:

1. **EEG's printed NNLO** (arXiv:2611.07007v1, (S45), (S46), (S48)–(S51)) violates their own (S39) at ε^{5/2} and
   disagrees with Proposition 1; the correction is in §1. Their NLO is right.
2. **The stored first-order λ data of the one-shot runs** (`solver/inner_pert/data/first_order_M1_xi0.2.json`,
   `first_order_M3_xi0.2.json`, `first_order_M3_xi0.175.json`): the arrays `ell`, `Jen`, `Jel` do not belong
   together. $`-\ell\cdot J_{en}/\ell\cdot J_{el}`$ from them is −0.033, −0.0004, −0.0007 against the stored `lam1`
   7.757, 3.080, 3.080, and `ell @ Jel` is 172, 5.1×10⁸, 5.1×10⁸ against `ellJel` −65257 (M = 3). The `lam1` values
   themselves are right: the second-order jets reproduce them ($`\ell\cdot J_{p1}`$ ≈ 0, §4), with the denominator
   taken from the model's own d/dλ direction. A first run of `second_order_run.py` that took the denominator from
   `ell @ Jel` gave λ₂ = 0.6 and 0.006 (posted on the issue as preliminary); the gauge-0 run exposed it, and it was
   replaced (logs `second_order_run_M1.log`, `second_order_run_M3.log` keep it).
3. **The λ₂ fit's planted test failed** (P4, §5); the fitted λ₂ is an estimator range with an uncertainty of about 0.1
   read from the planted misses.
4. **Not a discrepancy, a data change:** the first-order Δ₁ fit used 819 points, today's file gives 1159 (D to 67.8).
   The fitted Δ₁ range is now 10.375281–10.375285 over 15 fits, inside FIRST_ORDER.md's 10.375279–10.375321.
5. **Issue #126's acceptance exponents** (ε³ for Ω and f, ε^{5/2} for Π, for the corrected NNLO) are the printed NNLO's
   orders. The corrected one's absolute residual falls one order faster, as measured (§1.1).
6. **#126's control Π₃ × 1.01 did not fire** on the solutions (§1.1). The exponent test resolves Π₃ errors of about 5%
   and more. The criterion was left as written, so the run exits 1.
7. **Chains at equal D disagree in the NNLO residual** by up to 6.8% of R at D = 180 (T138-win2 against I13-win
   chain_d10), although each state's residual is about 10⁻²². That is the held harmonics' freedom, seen through the
   NNLO's high τ derivatives (§1.1).

## 9. Proposed text

For the coordinator (agents do not edit the README or other notes).

**README, "Relation to Ecker, Ecker & Grumiller"**, after the paragraph on arXiv:2601.14358, something like:

> Their 2026 follow-up (arXiv:2611.07007) rescales the free function, β = √ε β̃ with ε = 1/(D − 1), and lists the
> expansion to next-to-next-to-leading order. Under the map f = c, x²Ω = (n − 1)(b − 1), Π = y/(√2 x),
> Ψ = (p + m)/(√2 x²), with β̃ the exact central profile, it is Proposition 1's expansion: the next-to-leading order
> agrees term by term. The printed next-to-next-to-leading Π₃ does not satisfy their own equation (S39) at ε^{5/2},
> and Ω₃ and f₃ inherit the error; the corrected terms agree with Proposition 1
> (notes/SECOND_ORDER.md §1).
> Against the solutions from D = 15 to 215, with β̃ each solution's central profile, the corrected expansion's
> largest error on 0 ≤ x ≤ 1 falls like ε⁴ for f and Ω and ε^{7/2} for Π, and the printed one's like ε³ and ε^{5/2}.
> At D = 200 the printed expansion's rms deviation is 6–7 times the corrected one's. In their own Figs. 4 and 5,
> which use the D-independent profile of Eq. (16), the error moves the quoted rms deviations by about 3% or less,
> except f at D = 25 and 50 (notes/SECOND_ORDER.md §1.1).

**README, Proposition 1, after "Consequence"**, something like:

> The coefficients are nevertheless fixed order by order by the D = ∞ inner selection: the vanishing of the inner
> multiplier at O(1/n) gives Δ₁ (notes/FIRST_ORDER.md) and at O(1/n²) gives Δ₂: with five
> harmonics −50.06 against the fitted −49.17, and λ₂ = −33.37 ± 0.38 against −34.38, the fits lying between the three-
> and five-harmonic values (notes/SECOND_ORDER.md). The exponentially small terms that do this are the inner problem's
> Stokes jump, e^{−nΦ(x)} in the outer variables; in this reading they are the "instanton corrections e^{−1/ε}" that
> Ecker, Ecker and Grumiller expect, and not a term in Δ(D) itself.

**notes/LIT_REVIEW.md**, proposed entry:

> **Ecker, Ecker, Grumiller, arXiv:2611.07007v1 (4 Nov 2026), "Critical collapse remains discretely self-similar at
> large D"** (read from the arXiv PDF for issue #94). Contains: the scaling β = √ε β̃, ε = 1/(D − 1) (Eq. (10)); the
> large-D expansion of f, Ω, Π, Ψ to NNLO in GG gauge (supplement, (S35)–(S51)); the relation N₀ = (Δ₁ − 4πμ₁)/(4πμ₀)
> = ⟨β̃₀²⟩ ((S59), Eq. (15)); the fits μ₀ ≈ 0.130818, μ₁ ≈ −0.1449, Δ₀ ≈ 1.6439, Δ₁ ≈ 10.37 (Eqs. (7), (9)); the
> large-D profile model Eq. (16); the expectation that deriving Δ needs "instanton corrections of the form e^{−1/ε}".
> Does not contain: a derivation of Δ or of any of its coefficients, the inner (x ∼ √n) region, or a selection
> condition. Their printed NNLO Π₃ (S46) violates their (S39) (notes/SECOND_ORDER.md §1).

## 10. Scripts and reproduction

| script | what it does | time (Apple M5, one process) |
|---|---|---|
| `second_order_eeg.py [--printed] [--mutate=NAME]` | §1: EEG's NLO and NNLO against Proposition 1, in their own equations, the μ map; exit 1 on failure | 20 s |
| `nnlo_vs_solution.py extract OUT STATES...` | §1.1: f, Ω, Π, Ψ of each state on 0 ≤ x ≤ 1 (needs hpkernel with `Solver.residual_q`) | 3 s per state |
| `nnlo_vs_solution.py compare FIELDS... [--spread=DIRS] [--mutate=pi3xS] [--dmin=] [--dmax=]` | §1.1: residuals, exponent fits, C1–C4, C6; exit 1 on failure | 1 min |
| `nnlo_vs_solution.py planted FIELDS... [--plant_pi=] [--plant_scale=] [--mutate=pi3xS]` | §1.1: planted truth (C5) | 1 min |
| `nnlo_vs_solution.py figs FIELDS...` | §1.1: Figs. 4–5 measures with the exact, Eq. (16) and D = ∞ profiles | 1 min |
| `second_order_equations.py [--mutate=phi]` | §2: the exact inner equations, the slaved u, the O(1/n²) system, the reparametrization symmetry | 1 s |
| `second_order_jet.py` | module: the second-order jets (polarization, ω and λ jets) | |
| `second_order_checks.py [outer] [rhs] [modes --raw=DIR] [--mutate=NAME]` | §3: the checks; exit 1 on failure | 1 s; modes 2.5 min |
| `second_order_run.py M [--xi1=] [--raw=DIR] [--redo=omega2,gauge,lam2]` | §4, M = 1, 3: ω₂, the gauge diagnostic, λ₂ (and its gauge-0 value) | 1–3 min |
| `second_order_hier.py 5 [--xi1=] [--rtol=] [--raw=DIR] [--omega-only] [--no-gauge] [--lam-gauge]` | §4, M = 5, deflated (`--lam-gauge`: λ₂ with every gauge datum 0) | ω₂ 2–5 min; λ₂ 6 min |
| `second_order_table.py DIR` | the tables of §4 from the JSON files | 1 s |
| `second_order_fits.py [--planted] [--fixed=D1,D2;...]` | §5: the fitted Δ₂, λ₂ and the fixed fits; `--planted` exits 1 (P4) | 5–10 min |

From `solver/inner_pert/`, with the raw-data repository cloned beside the checkout (RAW = its folder `I94-mac`):

```
python3 second_order_eeg.py                      # ALL PASS, exit 0
python3 second_order_eeg.py --printed            # exit 1: the printed NNLO fails A and B
python3 second_order_equations.py                # ALL PASS
python3 second_order_checks.py                   # outer and rhs, ALL PASS
python3 second_order_run.py 1 --raw=$RAW         # and 3, 3 --xi1=0.175
python3 second_order_hier.py 5 --raw=$RAW --lam-gauge   # and --xi1=0.19, --rtol=1e-13 with --no-gauge
python3 second_order_checks.py modes --raw=$RAW  # needs the M = 3 run
python3 second_order_table.py $RAW
python3 second_order_fits.py --planted           # exit 1 (P4, section 5)
python3 second_order_fits.py "--fixed=10.4023,-50.06"
```

The first-order inputs are the JSON files of `solver/inner_pert/data/` (FIRST_ORDER.md §8).

§1.1, from the repository's root, with RAW the raw-data repository's checkout. The extraction reads the chain states
there and writes `RAW/I126-mac/fields/`. The exact state lists and runs are `RAW/I126-mac/*.cmd`, and their exits
are in `exits.txt`.

```
zsh RAW/I126-mac/extract.cmd        # the paths W, R, PY at its top name this machine's checkouts
python3 solver/inner_pert/nnlo_vs_solution.py planted RAW/I126-mac/fields/fields_D*.npz            # ALL PASS, exit 0
python3 solver/inner_pert/nnlo_vs_solution.py compare RAW/I126-mac/fields/fields_D*.npz \
    --spread=RAW/I126-mac/fields_t140_d6,RAW/I126-mac/fields_t140_d10                             # ALL PASS, exit 0
python3 solver/inner_pert/nnlo_vs_solution.py compare RAW/I126-mac/fields/fields_D*.npz --mutate=pi3x1.01   # exit 1
python3 solver/inner_pert/nnlo_vs_solution.py compare RAW/I126-mac/fields/fields_D*.npz --mutate=pi3x1.2    # exit 0
python3 solver/inner_pert/nnlo_vs_solution.py figs RAW/I126-mac/fields/fields_D*.npz
```

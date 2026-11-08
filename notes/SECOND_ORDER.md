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
- [2. The inner equations to O(1/n²)](#2-the-inner-equations-to-o1n2)
- [3. Second-order jets and their checks](#3-second-order-jets-and-their-checks)
- [4. The selection at O(1/n²): Δ₂ and λ₂](#4-the-selection-at-o1n2-δ₂-and-λ₂)
- [5. Against the fits, and Δ∞ with Δ₁ and Δ₂ held fixed](#5-against-the-fits-and-δ-with-δ₁-and-δ₂-held-fixed)
- [6. What EEG's e^(−1/ε) corresponds to here (a reading)](#6-what-eegs-e1ε-corresponds-to-here-a-reading)
- [7. What is rigorous, what is numerical, what is truncated](#7-what-is-rigorous-what-is-numerical-what-is-truncated)
- [8. Discrepancies](#8-discrepancies)
- [9. Proposed text](#9-proposed-text)
- [10. Scripts and reproduction](#10-scripts-and-reproduction)

## Summary

SUMMARY_PLACEHOLDER

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
- **What it changes.** EEG's Fig. 4 and Fig. 5 compare the NNLO fields with their D ≤ 200 solutions; the corrected Π₃
  changes Π, Ω and f at ε^{5/2} and ε³. Whether their figures used the printed expressions is not known from the paper.

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

## 2. The inner equations to O(1/n²)

In the inner variables the exact equations close with one slaved function, u = (n − 1)(b − 1) (`second_order_equations.py`,
*checked symbolically*, exactly, with e = 1/n and $`L = \xi^2\partial_\xi + \xi\partial_\tau`$):

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

In the outer polynomials this is $`-G_{2,0} = G_{2,1} + G_{2,2}`$, i.e. $`G_2(1) = 0`$. For a perturbation the same holds
linearized: $`\delta c_2(0) = -\delta g_2 - \delta h_1`$. In the full problem these data drop out (the reparametrization
symmetry above, for growth rates λ ≠ 1); in a truncated model they do not, and their effect measures the truncation.

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

## 4. The selection at O(1/n²): Δ₂ and λ₂

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
$`\mathcal S(\lambda_*)f_1 = -J_{p1}`$, $`J_{p1}`$ being the jump of the perturbation's 1/n component with f₁ = 0 (it holds
$`\mathcal S_1 f_0 + \lambda_1\partial_\lambda\mathcal S f_0`$, so $`\ell\cdot J_{p1} = 0`$ is a check of λ₁). Then, with the
perturbation's gauge data, the jump $`J_{p12}`$ of its 1/n² component (f₂ = 0, λ₂ = 0) gives

```math
\lambda_2 = -\,\frac{\ell\cdot J_{p12}/2}{\ell\cdot\partial_\lambda\mathcal S(\lambda_*)\,f_0} .
```

The jets contain every second-order term ($`\mathcal S_2 f_0`$, $`\mathcal S_1 f_1`$, $`\lambda_1\partial_\lambda\mathcal S f_1`$,
$`\lambda_1\partial_\lambda\mathcal S_1 f_0`$, $`\tfrac12\lambda_1^2\partial_\lambda^2\mathcal S f_0`$) without hand derivation.
M = 1, 3 use one-shot jets (`second_order_run.py`); M = 5 the deflated directions of `first_order_hier.py`
(`second_order_hier.py`): the e₁e₂ component, and the perturbation's 1/n and 1/n² components, get higher-harmonic
profile data refined until their own higher-harmonic jump stops falling (e₁e₂: from 1.9×10¹³ to 4 at M = 5), and the
harmonic-1 condition is solved with the deflated A and ω directions.

**Checks in the runs.** The first-order selected component has zero jump (M = 3: 9×10⁻⁴ against 3×10¹¹ for the
second-order component before selection); entering δx₂ as data brings the e₁e₂ jump from 3.5×10¹¹ to 0.39 (M = 3);
$`\ell\cdot J_{p1}`$ = −7×10⁻⁴ against $`\lambda_1\,\ell\cdot\partial_\lambda\mathcal S f_0`$ = 2×10⁵ (M = 3), so the jets
reproduce λ₁ of first_order_run.py; the gauge-0 runs reproduce FIRST_ORDER.md's gauge-0 λ₁ (0.740 at M = 1, 2.8268 at
M = 3).

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
| **λ₂** | **−141.05** | **−43.76** | **L5_PLACEHOLDER** | **−34.38** (± about 0.1, §5) |
| λ₂, every gauge datum 0 | −20.87 | −38.86 | | |

The M = 5 values are means of three runs (ξ₁ = 0.2, 0.19 and ξ₁ = 0.2 at rtol 10⁻¹³: Δ₂ = −50.134, −50.195,
−49.838; ω₂ = 269.08, 269.22, 268.38), the ± half their spread, a numerical error of the model value. The M = 3 values
are identical at ξ₁ = 0.2 and 0.175 to the digits shown. The fit's bracket for Δ₂ is the spread over fit choices; its
ω₂ is $`2\pi(\Delta_1^2/\Delta_\infty^3 - \Delta_2/\Delta_\infty^2)`$. Δ₂ in other variables (M = 5 mean against the fit):
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

LAMBDA_TEXT_PLACEHOLDER

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
FIXED_PLACEHOLDER
P3 shows why: Δ∞ moves by about 4–7×10⁻³ times the error of Δ₁. To keep Δ∞ within the README's 1.5×10⁻⁸, Δ₁ would have to
be known to about 3×10⁻⁶ (3×10⁻⁷ relative) and Δ₂ correspondingly; the M = 5 values are 0.26% and about 2% off. So
holding the derived coefficients fixed does not sharpen Δ∞ at this truncation; with exact Δ₁ and Δ₂ it would
(P2: spread 2.5×10⁻¹¹ instead of 6.8×10⁻⁹).

## 6. What EEG's e^(−1/ε) corresponds to here (a reading)

EEG's outlook expects that deriving Δ "will likely require to add instanton corrections of the form e^{−1/ε} to our
power-law expansion in ε". This is a stated reading, not a claim.

- **The power-law expansion cannot fix Δ** (Proposition 1, and §1: EEG's NLO and corrected NNLO are Proposition 1
  re-parametrized by the exact central profile, with the period free).
- **The exponentially small sector is the inner problem's Stokes jump.** The inner fast modes are
  $`e^{-\phi(\xi)/\xi^2}`$-type; at $`\xi = x/\sqrt n`$ they are $`e^{-n\Phi(x)}`$ in the outer region, with the singulant
  Φ of Proposition 4 and Lemma B5.1 ($`\Phi = -\tfrac12\ln(1 - x^{-2}) \pm i\pi/2`$ without gravity). With
  $`n = 1/\varepsilon - 1`$ these are $`e^{-\Phi/\varepsilon}`$ times powers: instanton-type terms with an x-dependent
  action, invisible at every order of EEG's expansion, and tied to its factorial divergence (Proposition 4). The
  multiplier S is their coefficient; Conjecture 1 sets it to zero.
- **Here that sector fixes the power-series coefficients of Δ, order by order.** Δ∞ from S(Z₀) = 0, Δ₁ from S⁽¹⁾ = 0,
  Δ₂ from S⁽²⁾ = 0. The values agree with the fits to 0.26% (Δ₁) and about 2% (Δ₂) at M = 5, with the fits between the
  M = 3 and M = 5 values. In this reading EEG's instanton corrections are the Stokes jump of the inner problem, whose
  cancellation selects the period; they are not an additive term $`c\,e^{-a/\varepsilon}`$ in Δ(D) itself.
- **What the data say about an additive term.** The transseries test (DINF_FITS.md) finds no such term in Δ(D) for
  a ≥ 0.5: its coefficient is the scatter a planted c = 0 also gives, and Δ∞ moves by at most 1.3×10⁻⁷. The test does
  not cover a < 0.5 or factors of ln ε, and the selection condition imposed as a formal series in 1/n does not exclude
  exponentially small corrections to Δ(D) of its own (the multiplier's own subleading exponentials). Δ's coefficients,
  Δ₂/Δ₁ = −4.74, are what a convergent series with the singularity near D ≈ −9 to −11 gives, not a factorially
  divergent one; nothing in them asks for a Borel-type completion.

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

**Truncated:** the M → ∞ limit is not computed. Δ₂'s sequence alternates from M = 3 on and brackets the fit, like the
first-order ones; three terms are a thin basis, and the truncation error at M = 5, estimated from the distance between
M = 3 and M = 5 and from the gauge dependence, is a few percent for Δ₂ and larger for λ₂.

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
   `ell @ Jel` gave λ₂ = 0.6 and 0.006; it was found by the gauge-0 run and replaced (logs `second_order_run_M1.log`,
   `second_order_run_M3.log` keep it).
3. **λ₂** of the truncated models: LAMBDA_DISCREPANCY_PLACEHOLDER
4. **The λ₂ fit's planted test failed** (P4, §5); the fitted λ₂ is an estimator range with an uncertainty of about 0.1
   read from the planted misses.

## 9. Proposed text

For the coordinator (agents do not edit the README or other notes).

**README, "Relation to Ecker, Ecker & Grumiller"**, after the paragraph on arXiv:2601.14358, something like:

> Their 2026 follow-up (arXiv:2611.07007) rescales the free function, β = √ε β̃ with ε = 1/(D − 1), and lists the
> expansion to next-to-next-to-leading order. Under the map f = c, x²Ω = (n − 1)(b − 1), Π = y/(√2 x),
> Ψ = (p + m)/(√2 x²), with β̃ the exact central profile, it is Proposition 1's expansion: the next-to-leading order
> agrees term by term. The printed next-to-next-to-leading Π₃ does not satisfy their own equation (S39) at ε^{5/2},
> and Ω₃ and f₃ inherit the error; the corrected terms agree with Proposition 1
> ([notes/SECOND_ORDER.md](notes/SECOND_ORDER.md) §1).

**README, Proposition 1, after "Consequence"**, something like:

> The coefficients are nevertheless fixed order by order by the D = ∞ inner selection: the vanishing of the inner
> multiplier at O(1/n) gives Δ₁ ([notes/FIRST_ORDER.md](notes/FIRST_ORDER.md)) and at O(1/n²) gives Δ₂, M5_SENTENCE
> ([notes/SECOND_ORDER.md](notes/SECOND_ORDER.md)). The exponentially small terms that do this are the inner problem's
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
| `second_order_equations.py [--mutate=phi]` | §2: the exact inner equations, the slaved u, the O(1/n²) system, the reparametrization symmetry | 1 s |
| `second_order_jet.py` | module: the second-order jets (polarization, ω and λ jets) | |
| `second_order_checks.py [outer] [rhs] [modes --raw=DIR] [--mutate=NAME]` | §3: the checks; exit 1 on failure | 1 s; modes 2.5 min |
| `second_order_run.py M [--xi1=] [--raw=DIR] [--redo=omega2,gauge,lam2]` | §4, M = 1, 3: ω₂, the gauge diagnostic, λ₂ (and its gauge-0 value) | 1–3 min |
| `second_order_hier.py 5 [--xi1=] [--rtol=] [--raw=DIR] [--omega-only] [--no-gauge]` | §4, M = 5, deflated | ω₂ 2–5 min; λ₂ LAM5_TIME |
| `second_order_table.py DIR` | the tables of §4 from the JSON files | 1 s |
| `second_order_fits.py [--planted] [--fixed=D1,D2;...]` | §5: the fitted Δ₂, λ₂ and the fixed fits; `--planted` exits 1 (P4) | 5–10 min |

From `solver/inner_pert/`, with the raw-data repository cloned beside the checkout (RAW = its folder `I94-mac`):

```
python3 second_order_eeg.py                      # ALL PASS, exit 0
python3 second_order_eeg.py --printed            # exit 1: the printed NNLO fails A and B
python3 second_order_equations.py                # ALL PASS
python3 second_order_checks.py                   # outer and rhs, ALL PASS
python3 second_order_run.py 1 --raw=$RAW         # and 3, 3 --xi1=0.175
python3 second_order_hier.py 5 --raw=$RAW        # and --xi1=0.19, --rtol=1e-13 with --omega-only --no-gauge
python3 second_order_checks.py modes --raw=$RAW  # needs the M = 3 run
python3 second_order_table.py $RAW
python3 second_order_fits.py --planted           # exit 1 (P4, section 5)
python3 second_order_fits.py "--fixed=10.4023,-50.06"
```

The first-order inputs are the JSON files of `solver/inner_pert/data/` (FIRST_ORDER.md §8).

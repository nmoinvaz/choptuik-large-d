# DINF_ANALYTIC: can the infinite-dimension echo period be derived without numerics?

**Verdict.** No route found gives $`\Delta_\infty`$ (or $`\omega_\infty`$) without numerical evaluation, and none of the
routes tested here is close to one. The inner problem fails the Painlevé test in its gauge-fixed form and in both of its
Lie-point reductions, while integrable and non-integrable controls pass and fail as they should (checked symbolically;
non-integrability itself stays a conjecture, since the converse of the Ablowitz–Ramani–Segur conjecture is not a
theorem). So there is no Lax pair to look for and no isomonodromy or Riemann–Hilbert problem to solve for the Stokes
data. Rozali and Way's exact oscillaton is a single-mode solution that does not extend to superpositions, so it does not
hide a linearization. The first amplitude coefficient $`\sigma_1`$ is already an exact double hypergeometric sum
(solver/proof/exact_sigma.py), but an LLL search against a frozen basket of the constants that sum could produce found
no relation, with planted and random controls behaving as required (§2; the scope of the null is stated there). The
inner problem has no small parameter, and every approximation to the selected zero is itself a numerically computed
nonlinear Stokes constant (§3). The closest rigorous statement within reach is a *certified* enclosure of the selected
frequency of the inner problem, *conditional* on Braaksma's theorem and on a harmonic-tail lemma that is not yet proved
(§4). The truncations $`M = 1`$ and $`M = 3`$ are certified now. $`M = 5`$'s first-order certificate was run and
fails (T33), so it needs a second-order chart or C² enclosures first.
Matching eq. (9)'s $`\pm 0.00006`$ needs $`M = 9`$ explicit plus that lemma. Even then, the identification of the inner
zero with $`\lim_{D\to\infty}\Delta(D)`$ rests on Conjecture 1. A paper of 4 Nov 2026 (Ecker, Ecker, Grumiller,
arXiv:2611.07007) also has no analytic value. It extrapolates its $`D \le 200`$ solutions to 1.643911,
$`1.7\times10^{-5}`$ above the finite-$`D`$ limit of eq. (4), a discrepancy recorded in §5. (Equation numbers are
the physics paper's.)

Scripts: `solver/inner_analytic/`. Logs of the runs quoted here are in the raw-data repository's `dinf-analytic-mac/`
(f0a8a7a); every number below is printed by a command in §7.

## 0. The question and what "without numerics" can mean

Eq. (9) of the physics paper, $`\Delta_\infty = 1.64387 \pm 0.00006`$, is the real zero of the harmonic-1 Stokes
multiplier of the $`D = \infty`$ inner problem, located by calibrated estimators on 22 late-order-fitted amplitude
coefficients. The question has three readings.

1. **A closed form, or an exact equation in known functions** (Gamma, hypergeometric, Painlevé transcendents), whose
   root is $`\omega_\infty`$. Routes 1–3 address this.
2. **A rigorous enclosure**, computer-assisted but with no floating-point estimator: interval arithmetic, validated
   integration, Krawczyk tests. Route 4 addresses this. It is "without estimators", not "without computation".
3. **A value from the $`D = \infty`$ equations alone**, with no finite-$`D`$ input. Eq. (9) already is that (numerical
   evidence), so this reading needs nothing new.

## 1. Integrability of the inner problem

### 1.1 The forms tested

Eq. (8) (math paper Prop. 5(i)) is $`Y_t = cW`$, $`W_t = c(Y_s + Y^3/4)`$, $`c_s = cY^2/4`$. Two equivalent forms were
checked symbolically (`heat_forms.py`; its mutation, $`Y^3/2`$ in place of $`Y^3/4`$, makes the check fail).

- **(A) Gauge-fixed form.** With the proper length $`u = \int c\,dt`$ along $`s = \text{const}`$ and the drift
  $`V = u_s`$ at fixed $`t`$,

```math
Y_s + (VY)_u = Y_{uu}, \qquad V_u = \tfrac14 Y^2 .
```

  The cubic term and the lapse combine into a drift: the inner problem is a Fokker–Planck (drift-diffusion) equation
  whose drift the field generates. The time-reparametrization gauge drops out. What is left is
  $`u \to u + h(s)`$, $`V \to V + h'(s)`$. The residuals of (8) are $`-cA_1 + cYA_2`$ and $`cA_2`$, with $`A_1, A_2`$
  those of (A).
- **(L) List's flow on a line.** In Rozali and Way's potential form ($`\phi_s = c^{-1}\partial_t(c^{-1}\phi_t)`$,
  $`c_s = \phi_t^2/c`$, $`Y = 2\phi_t/c`$; the proof of Prop. 5(i)), with the metric $`g = c^2dt^2`$ on the $`t`$-line,
  $`\partial_s g = 2\,d\phi\otimes d\phi`$ and $`\partial_s\phi = \Delta_g\phi`$. That is the Ricci flow coupled to the
  harmonic map flow, $`\partial_s g = -2\,\mathrm{Rc} + 2\alpha\,d\phi\otimes d\phi`$ (Müller [M12]), with
  $`\alpha = 1`$ and target $`\mathbb R`$, on a one-dimensional domain, where $`\mathrm{Rc} = 0`$; with
  $`\phi = \sqrt2\,\psi`$ it is List's flow [L08] ($`\alpha = 2`$). This is a reformulation, recorded for orientation.
  The results on those flows are for closed manifolds, and none that was read bears on discretely self-similar
  solutions on a line.

### 1.2 Criterion and controls (fixed before the runs)

The Weiss–Tabor–Carnevale test with Kruskal's ansatz $`\psi = x - f(t)`$ [ARS80, WTC83, JKM82] (`painleve.py`). A system
passes when every dominant balance has integer exponents, the resonances are integers, each resonance $`r \ge 1`$ has as
many free coefficients as its multiplicity as a root of $`\det Q(r)`$ (otherwise $`\psi^r\log\psi`$ terms are needed),
and every compatibility condition holds identically in $`f(t)`$ and the earlier free coefficients.

| system | resonances | result | expected |
|---|---|---|---|
| Burgers $`u_t + uu_x = u_{xx}`$ | −1, 2 | passes | pass |
| KdV | −1, 4, 6 | passes | pass |
| cubic NLS as a system for $`(u, \bar u)`$ | −1, 0, 3, 4 | passes | pass |
| forced Burgers system $`Y_s + (VY)_u = Y_{uu}`$, $`V_u = Y`$ | −1, 2, 3 | passes; log coefficient forced to 0 | pass |
| Newell–Whitehead $`u_t = u_{xx} + u - u^3`$ | −1, 4 | fails at 4: $`f'(-2f'^3 + 9f' + 6f'') \ne 0`$ | fail |
| Fisher–KPP $`u_t = u_{xx} + u(1-u)`$ | −1, 6 | fails at 6 | fail |
| **(A) inner, gauge-fixed** | −1, 2, 2 | **fails**: double resonance at 2 with one free coefficient; the log coefficient is a free function | — |
| **(A0) $`s`$-independent reduction** $`Y' = VY + k_1`$, $`V' = Y^2/4`$ | −1, 2 | **fails** unless $`k_1 = 0`$ (condition $`-k_1/2 = 0`$) | — |
| **(A1) self-similar (CSS) reduction** $`g' = gh - zg/2 + k`$, $`h' = g^2/4`$ | −1, 2 | **fails** unless $`k = 1`$ (condition $`-(k-1)/2 = 0`$) | — |
| (B) eq. (8) in $`(s, t)`$ with the lapse | leading $`Y, c \sim \psi^{-1/2}`$ | movable square-root branch point | — |
| (C) the math paper's $`(\xi, \theta)`$ form | leading $`y \sim \psi^{-1/2}`$ | movable square-root branch point | — |

The forced Burgers system is the closest control: the same code path, with $`V_u = Y^2/4`$ replaced by $`V_u = Y`$. It
is Burgers' equation for $`V`$ with a free forcing $`g(s)`$, which is linearizable, and it passes. Two runnable
mutations swap the two source terms (`--mutate=real_to_forced`, `--mutate=forced_to_real`). Each flips its verdict and
makes the script exit non-zero (7 of 9 verdicts as expected).

For (A0) and (A1) the failed compatibility condition means that a solution with $`k_1 \ne 0`$ (or $`k \ne 1`$) needs a
$`\psi\log\psi`$ term in $`Y`$: a movable logarithmic branch point. In (A) the double resonance does the same with a
free function of $`s`$. `log_check` in `painleve.py` adds $`\psi\log\psi`$ terms at $`r = 2`$ and finds the log
coefficient free for (A) and forced to 0 for the forced Burgers control.

(B) and (C) are not decisive by themselves. The lapse gauge $`c \to T'(t)c`$ can move such branch points, and Rozali and
Way's exact oscillaton, for which $`c^2 = (1 - \epsilon^2e^{-2\omega^2s})/(1 - w^2)`$, $`w = \tanh\phi`$, has exactly
these square-root points where $`w = \pm1`$. (A) is the gauge-fixed form, and (A0), (A1) are reductions of it by its Lie
point symmetries.

### 1.3 Lie point symmetries and exact solutions

The Lie point symmetries of this system with polynomial infinitesimals of degree up to 3 were computed in
`solver/matter/struct_symmetry.py` (matter/STRUCTURE.md, the multiplet with $`\ell = 0`$, which is this system in
potential form): $`s`$- and $`t`$-translation, the scaling $`(t, s) \to (at, a^2s)`$, the shift $`\phi \to \phi + k`$
and the time reparametrizations. In (A) the reparametrizations are gone and the translations and the scaling remain,
with the Galilean shift $`u \to u + \kappa s`$, $`V \to V + \kappa`$. The reductions are therefore
- $`s`$-translation: (A0); travelling waves $`Y(u - \kappa s)`$ are (A0) after the Galilean shift;
- the scaling: (A1), the continuously self-similar solutions.

A discretely self-similar solution is invariant under no continuous subgroup, so no Lie reduction reaches it. Both
reductions fail the Painlevé test for general values of their integration constant, which by the ARS conjecture
indicates that (A) is not integrable.

Exact solutions found along the way:
- the pole family $`Y = 2/(u - f(s))`$, $`V = -1/(u - f(s)) + f'(s)`$ (the truncated Painlevé expansion; the drift
  carries the pole);
- Rozali and Way's oscillaton [RW18, Eq. (10)], $`\phi = \operatorname{artanh} w`$ with
  $`w = \epsilon e^{-\omega^2 s}\sin\omega t`$, a single heat mode. If $`\operatorname{artanh}`$ of every heat solution
  solved (8), the problem would be linearizable. It does not. `oscillaton_lin.py` reproduces the single mode to
  $`5\times10^{-32}`$ (planted truth; its mutation, $`c_s = 2\phi_t^2/c`$, misses by $`0.04`$–$`0.09`$) and gives
  residuals $`2\times10^{-3}`$ to $`3.6\times10^{-2}`$ for two modes ($`\omega = 1`$ and $`3`$), in Rozali and Way's
  gauge ($`c \to 1`$ as $`s \to \infty`$). Another gauge per solution is not excluded by this check. The Painlevé result
  in the gauge-fixed form argues against it.

### 1.4 What integrability would have given, and status

An integrable (A) would have a Lax pair whose spectral data are conserved under the flow in $`s`$. A discretely
self-similar solution would then correspond to spectral data invariant under the discrete scaling, a $`q`$-type
isomonodromy problem. The fast-mode Stokes multipliers would be functions of those data, as the Stokes constants of the
Painlevé I tritronquée solution follow from its isomonodromy problem (Kapaev; rederived by O. Costin, R. D. Costin and
Huang [CCH] from the Painlevé property itself). $`S = 0`$ would then be an equation in known functions. That route is
closed: (A) fails the test.

*Status:* the test results are **checked symbolically**; the non-integrability of (A) is a **conjecture**.

## 2. Exact structure in the amplitude coefficients

### 2.1 What is already exact

$`\sigma_1 = S_1^{(3)}/S_1^{(1)}`$ has an exact representation. The order-1 sequences are the hypergeometric terms
$`T^\pm_l = (a_\pm)_l(b_\pm)_l/l!`$, the order-2 lapse coefficients are their convolutions, and the order-3 multiplier
sequence $`R_j`$ telescopes, so $`\sigma_1`$ is the limit of an explicit double sum of hypergeometric terms
(solver/proof/exact_sigma.py; proof/D_entire.md §6). Each $`\sigma_k`$ is a nested sum of the same kind, $`2k`$ levels
deep. In that weak sense $`S_1(A, \omega) = 0`$ is already "an exact equation in known functions": a power series in
$`A^2`$ whose coefficients are nested hypergeometric sums. It is exact but not closed. Summing it in $`A^2`$ is the
problem itself, since the terms reach 240 at the zero.

### 2.2 Integer-relation test of $`\sigma_1`$

*Target.* $`\sigma_1`$ at $`\omega = 4`$ exactly (`sigma1_hp.py`, which drives the amplitude code with $`K = 3`$ and
overrides $`\omega`$; at $`\omega = 4`$ a rational function of $`\omega`$ in a closed form is a Gaussian rational). Two
fit windows, $`J = 2400`$ and $`3000`$ (1280 bits, families with 10–30 terms), agree to 44.8 digits:

```math
\sigma_1(\omega = 4) = 0.77614460080113095690617189690294953863517 + 0.50415694742534165372735881448622696512992\,i .
```

The same code at $`\omega_\infty`$, $`J = 1500`$, reproduces the stored 20-digit $`\sigma_1`$ in every digit, and its
fitted linear multiplier equals the closed form $`-i/(\Gamma(a)\Gamma(b))`$ to $`10^{-354}`$ at $`\omega = 4`$.

*Baskets, frozen in the script before the target was read:*
- A (polygamma): $`1, \ln2, \pi, \pi^2, \psi(1+z), \psi(a), \psi'(1+z), \pi\coth\pi\omega, \pi\coth(\pi\omega/2)`$, with
  $`z = i\omega`$, $`a = (1+z)/2`$. Through the reflection and duplication formulas its span contains $`\psi(1-z)`$,
  $`\psi(b)`$, $`\psi(\bar a)`$, $`\psi(\bar b)`$ and the reflection terms these produce.
- B (Gamma ratios): $`1, \ln2, \pi, \psi(1+z)`$, $`\rho_1 = \Gamma(1-z)/\Gamma(1+z)`$,
  $`\rho_2 = 2^z\Gamma(a)/\Gamma(b)`$, $`\rho_1\psi(1-z)`$, $`\rho_2\psi(a)`$, the kind of values Gauss- and Kummer-type
  sums of $`T^-/T^+`$ produce.

*Method and criterion, fixed before the run.* LLL on the lattice of Gaussian-integer relations at $`d = 42`$ digits.
A relation is accepted if it involves the target, has height (largest coefficient modulus) at most $`10^3`$, a residual
below $`10^{5-d}`$, and is returned unchanged at $`d - 6`$ digits.

| test | basket | height of the LLL vector | stable at d − 6 | verdict | expected |
|---|---|---|---|---|---|
| planted, random Gaussian combination of height 30 | A | 35.5 | yes | relation | relation |
| planted | B | 39.7 | yes | relation | relation |
| linear multiplier, $`\log S^{(1)}`$ from the fit, against $`1, i\pi, \ln2, \ln\pi, \ln\Gamma(1+z)`$ | | 8 | yes | relation $`2\log S^{(1)} = -i\pi + 8i\ln2 - \ln\pi - 2\ln\Gamma(1+4i)`$ | relation |
| random complex number | A | $`1.6\times10^4`$ | no | none | none |
| random | B | $`3.0\times10^4`$ | no | none | none |
| **$`\sigma_1`$** | **A** | $`1.4\times10^4`$ | no | **none** | |
| **$`\sigma_1`$** | **B** | $`3.3\times10^4`$ | no | **none** | |

A random target gives height about $`10^{d/(n+1)}`$, $`10^{4.2}`$ for A and $`10^{4.7}`$ for B, so each null excludes
relations of height up to $`10^3`$ within that basket and nothing else. $`\sigma_1`$ behaves as a random number does.
With `--mutate=perturb` ($`10^{-21}`$ added to the planted targets) both planted relations disappear and the script
exits 1.

*Status:* the null is **numerical evidence** of limited scope. Joining the two baskets (17 constants) would bring the
random height down to $`10^{2.3}`$, below the bound, so that test needs about 60 digits. A closed form for $`\sigma_1`$
would not give $`\Delta_\infty`$ anyway; it would only make one coefficient of route 4.3 exact.

### 2.3 What the ratio law implies

The coefficients follow $`\sigma_k \approx C\kappa_\star^k/\Gamma(k + \beta_\star)`$ with
$`\kappa_\star = 1.279 + 1.081i`$ and $`\beta_\star = 4.16 - 1.60i`$ at $`\omega_\infty`$ (math paper Sec. "Exponential
type"; D_entire.md §1). So $`S_1`$ is numerically an entire function of $`\mathcal A = A^2`$ of order 1 and type 1.675.
Three consequences:

- $`S_1(\mathcal A) = (2\pi i)^{-1}\oint B(\zeta)e^{\zeta\mathcal A}\,d\zeta`$ exactly, with
  $`B(\zeta) = \sum_k k!\,\sigma_k\zeta^{-k-1}`$ (Pólya), and
  $`S_1 = e^{c_0 + c_1\mathcal A}\prod_n(1 - \mathcal A/\mathcal A_n)e^{\mathcal A/\mathcal A_n}`$ (Hadamard). Neither
  representation says where the zeros are without $`B`$.
- $`S_1 = 0`$ would be an equation in special functions if $`B`$ had finitely many singular points with known exponents
  and connection data. Its jump across the cut would then be an explicit product of powers, and $`S_1`$ a finite sum of
  confluent hypergeometric (Kummer or Humbert) functions. The data show a density along an arc from about
  $`0.75 + 0.16i`$ to $`\kappa_\star`$, vanishing at $`\kappa_\star`$ like $`(\kappa_\star - \zeta)^{\beta_\star - 1}`$.
  That is consistent with a cut between two branch points, but the endpoints and $`\beta_\star`$ are known to about
  three digits, they move with $`\omega`$ ($`\kappa_\star = 1.271 + 1.081i`$ to $`1.289 + 1.080i`$ for
  $`\omega/\omega_\infty = 0.99`$–$`1.01`$), and none matches a combination of the linear exponents
  $`a, b = (1 + i\omega)/2, (2 + i\omega)/2`$. Determining the jump function would mean solving the inner problem to all
  orders in $`A`$.
- The proposed mechanism (the math paper: $`\langle F^2\rangle = \mathcal A/2`$ at leading order, so the late-order
  normalization $`\xi^{2\langle F^2\rangle} = \xi^{\mathcal A}`$ is a Laplace kernel in $`\zeta = \ln\xi`$) would make
  the arc the image of a bounded path in $`\xi`$. An exact form of that path's endpoints would give $`\kappa_\star`$ in
  closed form, and the type of $`S_1`$ with it, but not the zero.

*Status:* the order-1 behaviour is **numerical evidence**; no exact resummation of $`S_1`$ is in sight.

## 3. Analytic approximations

The inner problem has no parameter. Its symmetries remove every scale, and $`\omega`$ is the eigenvalue. A
small-parameter expansion needs a parameter of the solution, and the two natural ones do not give closed forms.

- **The harmonic ratio** $`|q|A^2 \approx 0.063`$. Expanding in it is the truncation sequence $`M = 1, 3, 5, \dots`$,
  which converges geometrically (ratio $`-0.045`$ to $`-0.057`$ per pair). Its leading term, the $`M = 1`$ zero, is
  itself a nonlinear Stokes constant of an ODE, computed (and certified) numerically.
- **The amplitude** $`A`$. The selected zero is at $`A_0 = 2.72`$, where the terms $`|\sigma_k|A_0^{2k}`$ reach 240.
  There is no small-amplitude regime, and the linear problem has no zero at all (Prop. 2, **proved**).

No WKB or matched limit of the inner problem itself was found. The weights $`e^{\pi\nu\omega/2}`$
($`\pi\omega_\infty/2 = 6.0`$) are large, but they cancel exactly in each tuning step (B_tail_M7.md, Lemma W), so they
supply no small parameter.

The approximations available, all numerical, against $`\Delta = 1.64387`$ (from the papers' numbers;
$`\Delta = 1.643893746/(\omega/\omega_\infty)`$):

| approximation | $`\omega/\omega_\infty`$ | $`\Delta`$ | minus 1.64387 | closed form? |
|---|---|---|---|---|
| free wave ($`C(n,s) = 0`$) | no zero | none | | the non-existence is proved |
| $`M = 1`$ truncation (Thm A, certified) | 0.776994 | 2.11571 | +0.472 | no |
| $`M = 3`$ truncation (certified) | 1.0156 | 1.6186 | −0.025 | no |
| $`M = 5`$ truncation | 0.999228 | 1.64516 | +0.0013 | no |
| $`M = 7`$ truncation | 1.000022 | 1.64386 | −0.00001 | no |
| amplitude series, 12 terms, factored $`G`$ | 0.99988 | 1.64409 | +0.0002 | no |
| Price–Pullin flat-space matching [PP96], $`D = 4`$ only | | 3.6 (vs 3.44) | | uncontrolled |
| the NLO condition of [EEG26], used by [LW26] | | not a selection | | refuted, physics paper Sec. eeg |

*Status:* no controlled closed-form approximation exists among these. The truncation and amplitude rows are
**numerical evidence**; the $`M = 1`$ and $`M = 3`$ zeros are **certified** for their truncated models.

## 4. Rigorous routes

### 4.1 What a proof-grade value is

A certified enclosure $`\omega_* \in [\omega_-, \omega_+]`$ of the selected frequency of the inner problem, assuming
Braaksma's theorem, as Thm A does for $`M = 1`$. This answers reading 2 of §0. It would make the inner value
**certified**, while the statement that it equals $`\lim\Delta(D)`$ stays **conditional** on Conjecture 1.

### 4.2 Route T: truncations plus a tail lemma

- *Now:* $`M = 1`$ (Thm A) and $`M = 3`$ (math paper Sec. A3) are certified. $`M = 5`$: the centre is refined to
  $`|g| \le 2.5\times10^{-29}`$, and the four box runs of proof/A_M.md §7 were made on 2026-10-22 (T33) and failed. Two
  grew a derivative radius past one on the arc, and the run that finished fails the Krawczyk test (proof/A_M.md, its
  opening summary; `solver/proof/logs/A_M_krawczyk_M5_1e-13.log`). A box large enough needs a second-order chart and
  C² enclosures (proof/A_M.md §2). $`M = 7`$: the zero is known to $`\pm2\times10^{-5}`$ only (proof/B_tail_M7.md),
  and its conditioning (couplings $`10^{14}`$–$`10^{17}`$ already at $`M = 5`$) needs the Schur-corrected conditions in
  multiprecision.
- *The tail lemma* ($`|\omega_* - \omega_M| \le`$ explicit bound) is the open mathematics. proof/B_tail_M7.md §5 states
  it and proves its algebraic parts (Lemmas R, O, L, C, W), and its §8.4 names the one inequality (★) that needs a new
  idea: a bound on the linearized coupling uniform in the harmonic.
- *Precision versus $`M`$* (from the measured tail sizes in $`r_1`$: 4–6 above $`M = 5`$, 0.2–0.35 above $`M = 7`$,
  0.015 above $`M = 9`$, and the requirement of about 4 for a box of radius $`10^{-3}`$, scaled linearly in the radius
  and continued at 0.05 per pair beyond $`M = 9`$; estimates, not measurements):

| target | box radius in $`\omega/\omega_\infty`$ | tail needed in $`r_1`$ | explicit $`M`$ |
|---|---|---|---|
| a box of radius $`10^{-3}`$ | $`10^{-3}`$ | ≲ 4 | 7 (margin about 13–16) |
| eq. (9), $`\pm0.00006`$ in $`\Delta`$ | $`3.6\times10^{-5}`$ | ≲ 0.15 | 9 |
| eq. (4), $`\pm1.5\times10^{-8}`$ | $`9\times10^{-9}`$ | ≲ $`4\times10^{-5}`$ | 13–15 |

- *Work:* $`M = 5`$ needs the second-order design of proof/A_M.md §2, new certificate code. $`M = 7`$ and $`M = 9`$ need
  the existing machinery extended to 25–31 complex unknowns with multiprecision Lohner steps (weeks of runs and
  engineering). The lemma needs (★): unknown effort, the hard part.

### 4.3 Route E: the amplitude series

The math paper's Sec. "What a certificate would need": rigorous $`\sigma_k`$ for $`k \le 14`$–15 (for a $`10^{-3}`$ box;
about 17–18 for eq. (9)'s precision, from the Jacobian singular values 0.53 and 0.26), a tail bound, and uniformity in
$`\omega`$. Only $`\sigma_1`$ is enclosed ($`3.5\times10^{-5}`$, Lemma $`\sigma_1`$). The tail bound needs control of
the inner problem at complex amplitude $`|A| \approx 4`$–4.5, as hard as the direct problem. On top of that, the
amplitude series defines $`S_1`$ order by order, and its equality with the multiplier of an actual solution of the
untruncated problem is gap (a)–(b). Route T is the nearer one.

*Status:* both routes are open; their estimates are **numerical evidence** about cost, not results.

## 5. Literature

Read for this note (the rest is in LIT_REVIEW.md, whose entries for Rozali–Way, EEG, EEGJ, Lai–Wen, Price–Pullin and
Costin–Tanveer are not repeated).

- **[RW18] Rozali, Way**, JHEP 11 (2018) 106, arXiv:1807.10283 (verified, JHEP DOI 10.1007/JHEP11(2018)106; arXiv title
  "Gravitating Scalars and Critical Collapse in the Large D Limit"). Read now beyond Sec. 2.3 (ar5iv, through a
  summarizing fetch). *Contains:* the leading-order equations (9a)–(9c); the exact oscillaton Eq. (10),
  $`\phi_0 = \tanh^{-1}(\epsilon e^{-\omega^2r^2/2}\sin(\omega\tau - \tau_0))`$; "One can use scale invariance to fix
  ω"; the oscillatons "do not exhibit self symmetry of any sort"; $`\gamma = 1/2`$ from the divergence. *Not there:*
  integrability, DSS solutions at large $`D`$, $`\Delta`$ at large $`D`$ (only mentioned in the introduction), Stokes
  phenomena.
- **[PP96] Price, Pullin**, gr-qc/9601009, PRD 54, 3792 (1996) (as in LIT_REVIEW.md). Read now for the number:
  "Δ∼2π/√3∼3.6" (their Eq. (50)) from ⟨m/r⟩ ≈ 1/6, with "Our argument can only establish that Δ is of order unity";
  $`D = 4`$ only.
- **[EEG26], [LW26]** as in LIT_REVIEW.md: no Stokes phenomena; the NLO condition, which the physics paper's Sec. eeg
  shows does not select $`\Delta`$.
- **[EEG26b] C. Ecker, F. Ecker, D. Grumiller**, *Critical collapse remains discretely self-similar at large D*,
  arXiv:2611.07007, submitted 4 Nov 2026 (verified, arXiv listing; recorded in LIT_REVIEW.md). Read through summarizing
  fetches of the abstract and the HTML. *Contains:* critical solutions from $`D = 4`$ to 200 in steps of 1/2 (spectral
  Newton, Reiterer–Trubowitz infrastructure, GMP); "we predict Delta remains finite at infinite dimension D, contrary to
  a recent conjecture" (their own $`\Delta \to 0`$ of EEGJ); the fit "Delta=1.6439+10.37/D-O(1/D^2)"; a Table 1 whose
  last row is $`D = \infty`$, 1.643911, with no fit range or method given; and "an explicit large-D model", the profile
  $`\tilde\beta = A_*\,\mathrm{Re}[e^{i\theta}/(1 + qe^{i(2\theta+\delta)})]`$ with $`q \approx 0.0551`$ and
  $`\delta \approx 0.0985`$ fitted to the numerical profile. That is the geometric form of the math paper's tuned
  profile ($`|q|A^2 \approx 0.063`$ at leading order; $`|F_3/F_1| \to 0.0549`$–$`0.0552`$). It cites this project: "A
  similar result was obtained independently by Nathan Moinvaziri, who used the AI Claude to determine the critical
  solution in dimensions up to D=61", and says $`\gamma \to 1/2`$ there "has not yet been verified by human analysis".
  *Not there:* any analytic value of $`\Delta_\infty`$ (it is an input of the model), Stokes phenomena, the heat
  equation. Rozali and Way appear only as a future matching target, and a first-principles derivation "will likely
  require to add instanton corrections" $`e^{-1/\varepsilon}`$. **A discrepancy:** their $`D = \infty`$ entry, 1.643911,
  is $`1.7\times10^{-5}`$ above the finite-$`D`$ limit of eq. (4), 1.643893746 ± 1.5×10⁻⁸, about 1100 times its bar,
  while their finite-$`D`$ values agree with ours in all six printed decimals at $`D = 20`$, 40 and 50 (README table:
  2.112080154580, 1.888568197910, 1.841659503007) and with the repository's formula
  (solver/data/formula_coefficients.json, fitted on $`D \le 30.2`$) to $`3\times10^{-6}`$ at $`D = 200`$ (1.695090
  against their 1.695093). Their $`4\pi\mu_0`$ with $`\mu_0 \approx 0.130818`$ is 1.643907. The inner-problem value of
  eq. (9), $`1.64387 \pm 0.00006`$, contains both. Reported, not resolved.
- **[KS91] Kruskal, Segur**, *Asymptotics beyond all orders in a model of crystal growth*, Stud. Appl. Math. 85 (1991)
  129–181 (verified, Colorado expert record and MaRDI). The prototype of a selection by a nonlinear inner Stokes
  constant: needle solutions exist for discrete values of the parameter. Its full text was not read here, so whether its
  constant is closed-form is *(unverified)*.
- **[CKA98] Chapman, King, Adams**, Proc. R. Soc. Lond. A 454 (1998) 2733 (verified, Crossref listing on the publisher
  page). A method for exponentially small terms in nonlinear ODEs. Full text not read.
- **[CCH] O. Costin, R. D. Costin, M. Huang**, *Tronquée solutions of the Painlevé equation PI*, arXiv:1310.5330
  (verified, ar5iv; journal *(unverified)*). Read through a summarizing fetch: "The Stokes constant μ was calculated in
  closed form, μ = i√(6/(5π))", "first by Kapaev using the method of isomonodromic deformations", and rederived there
  from the Painlevé property. That is the integrable case §1.4 has in mind. Kapaev's papers were not located
  *(unverified)*.
- **[MP16] Michalik, Podhajecka**, *The Stokes phenomenon for certain partial differential equations with meromorphic
  initial data*, Asymptotic Analysis 99 (2016) 163–182, arXiv:1603.04209 (verified, arXiv listing). For the linear
  complex heat equation with meromorphic initial data the jumps across Stokes lines are computed exactly through Borel
  summation. It is the linear analogue of $`C(n,s)`$ and $`K'(s)`$; nothing nonlinear.
- **[CT89] Cariello, Tabor**, *Painlevé expansions for nonintegrable evolution equations*, Physica D 39 (1989) 77–94,
  DOI 10.1016/0167-2789(89)90040-7 (verified, Arizona expert record). The WTC test on non-integrable evolution equations
  (the Newell–Whitehead and Fisher controls are of this kind). Full text not read; the failing conditions in §1.2 are
  this note's own computation.
- **[L08] List**, *Evolution of an extended Ricci flow system*, Comm. Anal. Geom. 16 (2008) 1007–1048, DOI
  10.4310/CAG.2008.v16.n5.a5 (verified, MaRDI/zbMATH record). **[M12] Müller**, *Ricci flow coupled with harmonic map
  flow*, Ann. Sci. ÉNS 45 (2012) 101–142, DOI 10.24033/asens.2161, arXiv:0912.2907 (verified, Numdam and arXiv). Its
  flow: "∂g/∂t = −2Rc + 2α∇ϕ⊗∇ϕ, ∂ϕ/∂t = τ_g ϕ", List's being "N ⊆ ℝ" and α ≡ 2. The part of [M12] read (its
  first 100 000 characters, through a summarizing fetch) does not treat a one-dimensional domain; [L08] was not read.
- **Methods:** [ARS80] Ablowitz, Ramani, Segur, J. Math. Phys. 21 (1980) 715; [WTC83] Weiss, Tabor, Carnevale, J. Math.
  Phys. 24 (1983) 522; [JKM82] Jimbo, Kruskal, Miwa, Phys. Lett. A 92 (1982) 59 (Kruskal's simplification); [LLL82]
  Lenstra, Lenstra, Lovász, Math. Ann. 261 (1982) 515. These four were not re-verified in this session *(unverified)*.

Searched without finding analytic results: discretely self-similar solutions of heat-type or large-$`D`$ collapse
equations in closed form; exact (closed-form) Stokes data for nonlinear parabolic problems. Michalik–Podhajecka is the
nearest, and it is linear.

## 5a. The same selection in other fields (2026-11-07)

Problems in gas dynamics, astrophysics, water waves and magnetism select a similarity exponent the way Δ is selected:
by regularity at a singular surface, or by a vanishing Stokes constant. What they found bears on routes 2–4. Sources
were read through their arXiv or journal pages, as marked. These are analogies, not prior work on this problem.

| field and problem | what selects the parameter | closed form? | what they do instead |
|---|---|---|---|
| Gas dynamics: Guderley's converging shock [RBB17] | numerators vanish where the denominator does (the sonic point) | no; "determined numerically" | numerics, approximations |
| Gas dynamics: smooth implosion of compressible Euler [BCG25, BCG23] | smoothness across the sonic point | no; r = 1.079404 at γ = 7/5, numerical | explicit brackets r_j and a computer-assisted proof |
| Astrophysics: Larson–Penston collapse of an isothermal gas [GHJ21] | regularity at a single sonic point y* ∈ (2, 3) | no | an existence proof (its method not read) |
| Self-gravitating Brownian particles in D dimensions [SC02] | an analyticity condition, then the largest admissible α | canonical: α = 2 and the profile, exactly; otherwise no | a 1/D expansion of α |
| Water waves and optics: embedded solitons [CMYK01, BK10] | the Stokes constant of the tail vanishes | asymptotically, in a large-parameter limit | Stokes-line asymptotics, numerics |
| Magnetism: harmonic map heat flow, the overdamped Landau–Lifshitz flow [BHK03, RS14] | quantized blow-up rates λ ∝ (T − t)^L/\|log(T − t)\|^{2L/(2L−1)} | yes, the rates | formal matched asymptotics, then proofs |

What carries over:

- **No closed form is the norm.** Wherever regularity at a sonic surface selects the parameter, it is computed, not
  derived. That includes Guderley's exponent, the implosion parameter and Larson–Penston's sonic point. §0's answer to
  reading 1 is what the neighbouring fields found too.
- **Closed-form brackets.** At the sonic point of the implosion problem, the ratio k(r) of the two local eigenvalues is
  explicit. The values r_j where it is an integer j are singular and explicit, for example r₃ = 6 − 2√6. A shooting
  function changes sign between consecutive r_j, which brackets the selected r between explicit numbers [BCG23]. The
  direct analogue fails here. With s = iμω, the zeros of C(n, s) of eq. (6) lie at s = −1 − 2m and −2 − 2m and its
  poles at s = n/2 − 1 − m, all at real s and so at imaginary ω, and C(n, s) ≠ 0 for real ω (Proposition 2). Nothing
  in it can bracket the real ω∞. A bracket would need conditions that depend on the amplitude, as in the large-index
  limit of the Hunter family of collapsing gas spheres, where the profiles are bracketed explicitly (Sandine,
  arXiv:2310.18870, verified, arXiv). (Found by the 2026-11-07 cross-field agents, below.)
- **Proof method.** The implosion proof combines a shooting argument with interval arithmetic in Arb on a few thousand
  Taylor coefficients at the sonic point, with a desingularization at a resonance and branch-and-bound barrier checks
  [BCG23]. That is the same family as route T (§4.2), at a larger scale than this project has run.
- **A large-D expansion can select an exponent.** The Smoluchowski–Poisson system is a drift-diffusion equation whose
  drift the density generates, the structure of form (A) in §1.1. In D dimensions its collapse exponent is
  α = 2 + 1/(2D) + 11/(16D²) + O(D⁻³), which gives 2.24 at D = 3 against 2.2097 computed [SC02]. The difference from
  this problem is the base. There the leading order is explicit (α = 2, profile 4/(D − 2 + x²)), and each order is a
  regularity condition on it. Here the D = ∞ problem has no explicit solution (§1, not integrable), and the 1/n
  expansion never fixes Δ (README, Proposition 1).
- **Vanishing Stokes constants.** An embedded soliton exists only where the Stokes constant of its radiating tail
  vanishes. That is codimension one, the structure of Conjecture 1 [CMYK01]. In a water-wave model, the leading-order
  Stokes constant is derived for a large parameter. There each Stokes line splits into a pair whose contributions cancel
  at a countable set of parameter values [BK10]. The analogue here is untested and asymptotic, not exact: whether S₁
  has a two-Stokes-line form in some large parameter (the harmonic index, or the amplitude), with zeros that continue to
  the physical one.
- **Magnetism: explicit only around an explicit solution.** In the harmonic map heat flow the singularity is an explicit
  static solution (the harmonic-map bubble) shrinking. Its quantized rates are explicit [BHK03, RS14]. In List's-flow
  form (§1.1, L) the inner problem is a harmonic map flow coupled to a metric flow. Its target is the line, though,
  which has no bubble, and the critical solution here is not explicit. The equivariant wave maps of Bizoń, Chmaj and
  Tabor, "a toy-model for some aspects of the critical behavior" in black-hole formation, have their threshold set by a
  self-similar solution with one instability [BCT00].
- **Discrete self-similarity in gas dynamics** arises as log-periodic perturbations of the continuously self-similar
  Sedov–Taylor–von Neumann explosion [YS13]. There the background is continuously self-similar. In Choptuik's problem
  the critical solution itself is discretely self-similar.

One test follows from this, open: the Stokes-line asymptotics of S₁ in a large parameter.

References:

- **[RBB17] Ramsey, Baty, Boyd**, *On the existence of self-similar converging shocks for arbitrary equation of state*,
  arXiv:1707.03792 (verified, arXiv; journal not checked).
- **[BCG25] Buckmaster, Cao-Labora, Gómez-Serrano**, *Smooth imploding solutions for 3D compressible fluids*, Forum
  Math. Pi 13 (2025) e6, doi:10.1017/fmp.2024.12, arXiv:2208.09445 (verified, arXiv and the journal record).
- **[BCG23] Buckmaster, Cao-Labora, Gómez-Serrano**, *Smooth self-similar imploding profiles to 3D compressible Euler*,
  arXiv:2301.10101 (verified, arXiv; read through ar5iv; journal not checked).
- **[GHJ21] Guo, Hadžić, Jang**, *Larson–Penston self-similar gravitational collapse*, Commun. Math. Phys. 386 (2021),
  doi:10.1007/s00220-021-04175-y, arXiv:2011.01013 (verified, arXiv and DOI; abstract only).
- **[SC02] Sire, Chavanis**, *Thermodynamics and collapse of self-gravitating Brownian particles in D dimensions*, Phys.
  Rev. E 66 (2002) 046133, doi:10.1103/PhysRevE.66.046133, arXiv:cond-mat/0204303 (verified, arXiv; read through
  ar5iv).
- **[CMYK01] Champneys, Malomed, Yang, Kaup**, *"Embedded solitons": solitary waves in resonance with the linear
  spectrum*, arXiv:nlin/0005056 (verified, arXiv listing; journal not checked).
- **[BK10] van den Berg, King**, *Vanishing beyond all orders: Stokes lines in a water-wave model equation*,
  Nonlinearity 23 (2010) 1–21, doi:10.1088/0951-7715/23/1/001 (verified, publisher record; abstract only).
- **[BHK03] van den Berg, King, Hulshof**, *Formal asymptotics of bubbling in the harmonic map heat flow*, SIAM J. Appl.
  Math. 63 (2003) 1682–1717, doi:10.1137/S0036139902408874 (verified, the journal's DOI and bibliographic records;
  content through later papers that cite it).
- **[RS14] Raphaël, Schweyer**, *Quantized slow blow up dynamics for the corotational energy critical harmonic heat
  flow*, Anal. PDE 7 (2014) 1713–1805, arXiv:1301.1859 (verified, arXiv).
- **[BCT00] Bizoń, Chmaj, Tabor**, *Dispersion and collapse of wave maps*, Nonlinearity 13 (2000) 1411–1423,
  arXiv:math-ph/9912009 (verified, arXiv).
- **[YS13] Yalinewich, Sari**, *Discrete self similarity in filled type I strong explosions*, Phys. Fluids 25 (2013)
  126101, arXiv:1407.6938 (verified, arXiv).

## 6. Proposed text for the papers

Eq. (9)'s standing does not change. Proposed additions, for the coordinator:

- *Physics paper, Sec. "The selection rule at D = ∞", after eq. (9):* "No closed form is expected. In the proper length
  $`u = \int c\,dt`$ the inner equations become $`Y_s + (VY)_u = Y_{uu}`$, $`V_u = Y^2/4`$ with $`V = u_s`$, and this
  system, like its stationary and continuously self-similar reductions, fails the Painlevé test, so its Stokes data
  cannot be obtained from an isomonodromy problem; the inner problem has no small parameter, and every approximation to
  the zero is itself a nonlinear Stokes constant."
- *Math paper, Sec. "The selection rule" or Prop. 5(i), a remark:* the gauge-fixed form (A), with the identity of §1.1
  and the Painlevé result as **checked symbolically**, and the non-integrability as a conjecture; and the List-flow
  reading (L) as a one-line remark crediting [L08, M12].
- *README, the "Candidate closed forms" paragraph:* "whether Δ∞ has one is open" can stay, with a pointer to this note
  for the integrability and integer-relation tests.

## 7. Files and commands

From the repository root, with python-flint 0.9.0, mpmath 1.3.0, sympy 1.14.0:

```
python3 solver/inner_analytic/painleve.py                       # 9/9 verdicts as expected, exit 0 (seconds)
python3 solver/inner_analytic/painleve.py --mutate=real_to_forced   # 7/9, exit 1
python3 solver/inner_analytic/heat_forms.py                     # identities hold, exit 0
python3 solver/inner_analytic/heat_forms.py --mutate=quarter    # an identity fails, exit 1
python3 solver/inner_analytic/oscillaton_lin.py                 # single mode 5e-32, two modes 3.6e-2, exit 0
python3 solver/inner_analytic/oscillaton_lin.py --mutate=coef   # planted single mode fails, exit 1
python3 solver/inner_analytic/sigma1_hp.py 4 2400 1280 --mf 10:30 > RUN_A.json   # 12 min (719 s) here
python3 solver/inner_analytic/sigma1_hp.py 4 3000 1280 --mf 10:30 > RUN_B.json   # 14 min (845 s) here
python3 solver/inner_analytic/sigma_lll.py RUN_A.json RUN_B.json                   # 44.77 digits, d = 42, exit 0
python3 solver/inner_analytic/sigma_lll.py RUN_A.json RUN_B.json --mutate=perturb  # planted lost, exit 1
python3 solver/inner_analytic/sigma1_hp.py 1.0 1500 1280 --wf --mf 8:20           # omega_inf: the stored 20 digits
```

The two $`\omega = 4`$ records (RUN_A, RUN_B) are in the raw-data repository's `dinf-analytic-mac/`,
`sigma1_w4_J2400.json` and `sigma1_w4_J3000.json`.

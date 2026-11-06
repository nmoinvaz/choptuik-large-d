# DINF_ANALYTIC: can the infinite-dimension echo period be derived without numerics?

**Verdict.** No route found gives $`\Delta_\infty`$ (or $`\omega_\infty`$) without numerical evaluation, and none of the
routes tested here is close to one. The inner problem fails the Painlevé test in its gauge-fixed form and in both of
its Lie-point reductions, while integrable and non-integrable controls pass and fail as they should (checked
symbolically; non-integrability itself stays a conjecture, since the converse of the Ablowitz–Ramani–Segur conjecture is
not a theorem). So there is no Lax pair to look for and no isomonodromy or Riemann–Hilbert problem to solve for the
Stokes data. Rozali and Way's exact oscillaton is a single-mode solution that does not extend to superpositions, so it
does not hide a linearization. The first amplitude coefficient $`\sigma_1`$ is already an exact double hypergeometric sum
(solver/proof/exact_sigma.py), but an LLL search against a frozen basket of the constants that sum could produce found no
relation, with planted and random controls behaving as required (§2; the scope of the null is stated there). The inner
problem has no small parameter, and every approximation to the selected zero is itself a numerically computed nonlinear
Stokes constant (§3). The closest rigorous statement within reach is a *certified* enclosure of the selected frequency of
the inner problem, *conditional* on Braaksma's theorem and on a harmonic-tail lemma that is not yet proved (§4). The
truncations $`M = 1`$ and $`M = 3`$ are certified now, and $`M = 5`$ is a few hours of local runs away. Matching eq. (9)'s
$`\pm 0.00006`$ needs $`M = 9`$ explicit plus that lemma. Even then, the identification of the inner zero with
$`\lim_{D\to\infty}\Delta(D)`$ rests on Conjecture 1.

Scripts: `solver/inner_analytic/`. Logs of the runs quoted here are in the session scratchpad
(`dinf_analytic/`); every number below is printed by a command in §7.

## 0. The question and what "without numerics" can mean

Eq. (9) of the physics paper, $`\Delta_\infty = 1.64387 \pm 0.00006`$, is the real zero of the harmonic-1 Stokes multiplier
of the $`D = \infty`$ inner problem, located by calibrated estimators on 22 late-order-fitted amplitude coefficients. The
question has three readings.

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
  harmonic map flow, $`\partial_s g = -2\,\mathrm{Rc} + 2\alpha\,d\phi\otimes d\phi`$ (Müller [M12]), with $`\alpha = 1`$ and
  target $`\mathbb R`$, on a one-dimensional domain, where $`\mathrm{Rc} = 0`$; with $`\phi = \sqrt2\,\psi`$ it is List's
  flow [L08] ($`\alpha = 2`$). This is a reformulation, recorded for orientation. The results on those flows are for
  closed manifolds of dimension at least 2, and none bears on discretely self-similar solutions on a line.

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

The forced Burgers system is the closest control: the same code path, with $`V_u = Y^2/4`$ replaced by $`V_u = Y`$. It is
Burgers' equation for $`V`$ with a free forcing $`g(s)`$, which is linearizable, and it passes. Two runnable mutations
swap the two source terms (`--mutate=real_to_forced`, `--mutate=forced_to_real`). Each flips its verdict and makes the
script exit non-zero (7 of 9 verdicts as expected).

For (A0) the log term is explicit. With $`Y = 2/\psi + y_2\psi + b\,\psi\log\psi + \dots`$, the order-$`\psi^0`$ equations
give $`b = k_1/3`$, so every solution with $`k_1 \ne 0`$ has a movable logarithmic branch point. In (A) the double
resonance does the same with a free function of $`s`$ (`log_check` in `painleve.py`).

(B) and (C) are not decisive by themselves. The lapse gauge $`c \to T'(t)c`$ can move such branch points, and Rozali and
Way's exact oscillaton, for which $`c^2 = (1 - \epsilon^2e^{-2\omega^2s})/(1 - w^2)`$, $`w = \tanh\phi`$, has exactly these
square-root points where $`w = \pm1`$. (A) is the gauge-fixed form, and (A0), (A1) are reductions of it by its Lie point
symmetries.

### 1.3 Lie point symmetries and exact solutions

The Lie point symmetries of this system with polynomial infinitesimals of degree up to 3 were computed in
`matter/struct_symmetry.py` (matter/STRUCTURE.md, the multiplet with $`\ell = 0`$, which is this system in potential
form): $`s`$- and $`t`$-translation, the scaling $`(t, s) \to (at, a^2s)`$, the shift $`\phi \to \phi + k`$ and the time
reparametrizations. In (A) the reparametrizations are gone and the translations and the scaling remain, with the
Galilean shift $`u \to u + \kappa s`$, $`V \to V + \kappa`$. The reductions are therefore
- $`s`$-translation: (A0); travelling waves $`Y(u - \kappa s)`$ are (A0) after the Galilean shift;
- the scaling: (A1), the continuously self-similar solutions.

A discretely self-similar solution is invariant under no continuous subgroup, so no Lie reduction reaches it. Both
reductions fail the Painlevé test for general values of their integration constant, which by the ARS conjecture
indicates that (A) is not integrable.

Exact solutions found along the way:
- the pole family $`Y = 2/(u - f(s))`$, $`V = -1/(u - f(s)) + f'(s)`$ (the truncated Painlevé expansion; the drift carries
  the pole);
- Rozali and Way's oscillaton [RW18, Eq. (10)], $`\phi = \operatorname{artanh} w`$ with $`w = \epsilon e^{-\omega^2 s}\sin\omega t`$,
  a single heat mode. If $`\operatorname{artanh}`$ of every heat solution solved (8), the problem would be linearizable.
  It does not. `oscillaton_lin.py` reproduces the single mode to $`5\times10^{-32}`$ (planted truth; its mutation,
  $`c_s = 2\phi_t^2/c`$, misses by $`0.04`$–$`0.09`$) and gives residuals $`2\times10^{-3}`$ to $`3.6\times10^{-2}`$ for
  two modes ($`\omega = 1`$ and $`3`$), in Rozali and Way's gauge ($`c \to 1`$ as $`s \to \infty`$). Another gauge per
  solution is not excluded by this check. The Painlevé result in the gauge-fixed form argues against it.

### 1.4 What integrability would have given, and status

An integrable (A) would have a Lax pair whose spectral data are conserved under the flow in $`s`$. A discretely
self-similar solution would then correspond to spectral data invariant under the discrete scaling, a $`q`$-type
isomonodromy problem. The fast-mode Stokes multipliers would be functions of those data, as the Stokes constants of
the Painlevé I tritronquée solution follow from its isomonodromy problem (Kapaev; rederived by O. Costin, R. D. Costin
and Huang [CCH] from the Painlevé property itself). $`S = 0`$ would then be an equation in known functions. That route is
closed: (A) fails the test.

*Status:* the test results are **checked symbolically**; the non-integrability of (A) is a **conjecture**.

## 2. Exact structure in the amplitude coefficients

PENDING

## 3. Analytic approximations

The inner problem has no parameter. Its symmetries remove every scale, and $`\omega`$ is the eigenvalue. A
small-parameter expansion needs a parameter of the solution, and the two natural ones do not give closed forms.

- **The harmonic ratio** $`|q|A^2 \approx 0.063`$. Expanding in it is the truncation sequence $`M = 1, 3, 5, \dots`$, which
  converges geometrically (ratio $`-0.045`$ to $`-0.057`$ per pair). Its leading term, the $`M = 1`$ zero, is itself a
  nonlinear Stokes constant of an ODE, computed (and certified) numerically.
- **The amplitude** $`A`$. The selected zero is at $`A_0 = 2.72`$, where the terms $`|\sigma_k|A_0^{2k}`$ reach 240. There is
  no small-amplitude regime, and the linear problem has no zero at all (Prop. 2, **proved**).

No WKB or matched limit of the inner problem itself was found. The weights $`e^{\pi\nu\omega/2}`$ ($`\pi\omega_\infty/2 = 6.0`$)
are large, but they cancel exactly in each tuning step (B_tail_M7.md, Lemma W), so they supply no small parameter.

The approximations available, all numerical, against $`\Delta = 1.64387`$ (from the papers' numbers; $`\Delta = 1.643893746/(\omega/\omega_\infty)`$):

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
  $`|g| \le 2.5\times10^{-29}`$ and four box runs of about 2000 Lohner steps each, 3–4 hours on 4 local threads, are
  specified (proof/A_M.md §6.3, §7). $`M = 7`$: the zero is known to $`\pm2\times10^{-5}`$ only (proof/B_tail_M7.md), and
  its conditioning (couplings $`10^{14}`$–$`10^{17}`$ already at $`M = 5`$) needs the Schur-corrected conditions in
  multiprecision.
- *The tail lemma* ($`|\omega_* - \omega_M| \le`$ explicit bound) is the open mathematics. proof/B_tail_M7.md §5 states it
  and proves its algebraic parts (Lemmas R, O, L, C, W), and its §8.4 names the one inequality (★) that needs a new idea:
  a bound on the linearized coupling uniform in the harmonic.
- *Precision versus $`M`$* (from the measured tail sizes in $`r_1`$: 4–6 above $`M = 5`$, 0.2–0.35 above $`M = 7`$, 0.015
  above $`M = 9`$, and the requirement of about 4 for a box of radius $`10^{-3}`$, scaled linearly in the radius and
  continued at 0.05 per pair beyond $`M = 9`$; estimates, not measurements):

| target | box radius in $`\omega/\omega_\infty`$ | tail needed in $`r_1`$ | explicit $`M`$ |
|---|---|---|---|
| a box of radius $`10^{-3}`$ | $`10^{-3}`$ | ≲ 4 | 7 (margin about 13–16) |
| eq. (9), $`\pm0.00006`$ in $`\Delta`$ | $`3.6\times10^{-5}`$ | ≲ 0.15 | 9 |
| eq. (7), $`\pm1.5\times10^{-8}`$ | $`9\times10^{-9}`$ | ≲ $`4\times10^{-5}`$ | 13–15 |

- *Work:* $`M = 5`$ is a queued run. $`M = 7`$ and $`M = 9`$ need the existing machinery extended to 25–31 complex
  unknowns with multiprecision Lohner steps (weeks of runs and engineering). The lemma needs (★): unknown effort, the
  hard part.

### 4.3 Route E: the amplitude series

The math paper's Sec. "What a certificate would need": rigorous $`\sigma_k`$ for $`k \le 14`$–15 (for a $`10^{-3}`$ box;
about 17–18 for eq. (9)'s precision, from the Jacobian singular values 0.53 and 0.26), a tail bound, and uniformity in
$`\omega`$. Only $`\sigma_1`$ is enclosed ($`3.5\times10^{-5}`$, Lemma $`\sigma_1`$). The tail bound needs control of the
inner problem at complex amplitude $`|A| \approx 4`$–4.5, as hard as the direct problem. On top of that, the amplitude
series defines $`S_1`$ order by order, and its equality with the multiplier of an actual solution of the untruncated
problem is gap (a)–(b). Route T is the nearer one.

*Status:* both routes are open; their estimates are **numerical evidence** about cost, not results.

## 5. Literature

Read for this note (the rest is in LIT_REVIEW.md, whose entries for Rozali–Way, EEG, EEGJ, Lai–Wen, Price–Pullin and
Costin–Tanveer are not repeated).

- **[RW18] Rozali, Way**, JHEP 11 (2018) 106, arXiv:1807.10283 (verified, JHEP DOI 10.1007/JHEP11(2018)106; arXiv title
  "Gravitating Scalars and Critical Collapse in the Large D Limit"). Read now beyond Sec. 2.3 (ar5iv, through a
  summarizing fetch). *Contains:* the leading-order equations (9a)–(9c); the exact oscillaton Eq. (10),
  $`\phi_0 = \tanh^{-1}(\epsilon e^{-\omega^2r^2/2}\sin(\omega\tau - \tau_0))$`; "One can use scale invariance to fix ω"; the
  oscillatons "do not exhibit self symmetry of any sort"; $`\gamma = 1/2`$ from the divergence. *Not there:* integrability,
  DSS solutions at large $`D`$, $`\Delta`$ at large $`D`$ (only mentioned in the introduction), Stokes phenomena.
- **[PP96] Price, Pullin**, gr-qc/9601009, PRD 54, 3792 (1996) (as in LIT_REVIEW.md). Read now for the number: "Δ∼2π/√3∼3.6"
  (their Eq. (50)) from ⟨m/r⟩ ≈ 1/6, with "Our argument can only establish that Δ is of order unity"; $`D = 4`$ only.
- **[EEG26], [LW26]** as in LIT_REVIEW.md: no Stokes phenomena; the NLO condition, which the physics paper's Sec. eeg
  shows does not select $`\Delta`$.
- **[KS91] Kruskal, Segur**, *Asymptotics beyond all orders in a model of crystal growth*, Stud. Appl. Math. 85 (1991)
  129–181 (verified, Colorado expert record and MaRDI). The prototype of a selection by a nonlinear inner Stokes constant:
  needle solutions exist for discrete values of the parameter. Its full text was not read here, so whether its constant is
  closed-form is *(unverified)*.
- **[CKA98] Chapman, King, Adams**, Proc. R. Soc. Lond. A 454 (1998) 2733 (verified, Crossref listing on the publisher
  page). A method for exponentially small terms in nonlinear ODEs. Full text not read.
- **[CCH] O. Costin, R. D. Costin, M. Huang**, *Tronquée solutions of the Painlevé equation PI*, arXiv:1310.5330
  (verified, ar5iv; journal *(unverified)*). Read through a summarizing fetch: "The Stokes constant μ was calculated in
  closed form, μ = i√(6/(5π))", "first by Kapaev using the method of isomonodromic deformations", and rederived there
  from the Painlevé property. That is the integrable case §1.4 has in mind. Kapaev's papers were not located
  *(unverified)*.
- **[MP16] Michalik, Podhajecka**, *The Stokes phenomenon for certain partial differential equations with meromorphic
  initial data*, Asymptotic Analysis 99 (2016) 163–182, arXiv:1603.04209 (verified, arXiv listing). For the linear complex
  heat equation with meromorphic initial data the jumps across Stokes lines are computed exactly through Borel summation.
  It is the linear analogue of $`C(n,s)`$ and $`K'(s)`$; nothing nonlinear.
- **[CT89] Cariello, Tabor**, *Painlevé expansions for nonintegrable evolution equations*, Physica D 39 (1989) 77–94,
  DOI 10.1016/0167-2789(89)90040-7 (verified, Arizona expert record). The WTC test on non-integrable evolution equations
  (the Newell–Whitehead and Fisher controls are of this kind). Full text not read; the failing conditions in §1.2 are
  this note's own computation.
- **[L08] List**, *Evolution of an extended Ricci flow system*, Comm. Anal. Geom. 16 (2008) 1007–1048,
  DOI 10.4310/CAG.2008.v16.n5.a5 (verified, MaRDI/zbMATH record). **[M12] Müller**, *Ricci flow coupled with harmonic map
  flow*, Ann. Sci. ÉNS 45 (2012) 101–142, DOI 10.24033/asens.2161, arXiv:0912.2907 (verified, Numdam and arXiv). Its
  flow: "∂g/∂t = −2Rc + 2α∇ϕ⊗∇ϕ, ∂ϕ/∂t = τ_g ϕ", List's being "N ⊆ ℝ" and α ≡ 2. Neither treats a one-dimensional domain.
- **Methods:** [ARS80] Ablowitz, Ramani, Segur, J. Math. Phys. 21 (1980) 715; [WTC83] Weiss, Tabor, Carnevale, J. Math.
  Phys. 24 (1983) 522; [JKM82] Jimbo, Kruskal, Miwa, Phys. Lett. A 92 (1982) 59 (Kruskal's simplification); [LLL82]
  Lenstra, Lenstra, Lovász, Math. Ann. 261 (1982) 515. These four were not re-verified in this session *(unverified)*.

Searched without finding analytic results: discretely self-similar solutions of heat-type or large-$`D`$ collapse
equations in closed form; exact (closed-form) Stokes data for nonlinear parabolic problems. Michalik–Podhajecka is the
nearest, and it is linear.

## 6. Proposed text for the papers

Eq. (9)'s standing does not change. Proposed additions, for the coordinator:

- *Physics paper, Sec. "The selection rule at D = ∞", after eq. (9):* "No closed form is expected. In the proper length
  $`u = \int c\,dt`$ the inner equations become $`Y_s + (VY)_u = Y_{uu}`$, $`V_u = Y^2/4`$ with $`V = u_s`$, and this system,
  like its stationary and continuously self-similar reductions, fails the Painlevé test, so its Stokes data cannot be
  obtained from an isomonodromy problem; the inner problem has no small parameter, and every approximation to the zero
  is itself a nonlinear Stokes constant."
- *Math paper, Sec. "The selection rule" or Prop. 5(i), a remark:* the gauge-fixed form (A), with the identity of §1.1 and
  the Painlevé result as **checked symbolically**, and the non-integrability as a conjecture; and the List-flow reading
  (L) as a one-line remark crediting [L08, M12].
- *README, the "Candidate closed forms" paragraph:* "whether Δ∞ has one is open" can stay, with a pointer to this note
  for the integrability and integer-relation tests.

## 7. Files and commands

From the repository root, with python-flint 0.9.0, mpmath 1.3.0, sympy 1.14.0:

```
python3 solver/inner_analytic/painleve.py                       # 9/9 verdicts as expected, exit 0 (seconds)
python3 solver/inner_analytic/painleve.py --mutate=real_to_forced   # 7/9, exit 1
python3 solver/inner_analytic/heat_forms.py                     # identities hold, exit 0; --mutate=quarter fails
python3 solver/inner_analytic/oscillaton_lin.py                 # single mode 5e-32, two modes 3.6e-2; --mutate=coef exit 1
```

PENDING (route 2 commands)

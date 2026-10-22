# Formal verification: parallels with anthropics/formal-math

This page compares this project with [anthropics/formal-math](https://github.com/anthropics/formal-math), Anthropic's
repository of machine-checked Lean 4 formalizations. It asks three questions:

- which of our results could be stated and proved in Lean 4 with Mathlib;
- how the computer-assisted certificate of layer A ([A_certify.md](A_certify.md)) could be connected to a proof assistant;
- what we can adopt from how that repository presents AI-written mathematics.

It was written by Claude (Anthropic) on 21 October 2026, from formal-math at commit
[`e1a4e65`](https://github.com/anthropics/formal-math/tree/e1a4e6508154ea59f030480661590a9fe3018011) (5 October 2026) and
from Mathlib at tag `v4.33.1`, the pin of formal-math's newer project. **Nothing on this page was compiled.** The Lean
blocks are sketches, and the effort estimates are judgements, not measurements.

Contents

- [1. What formal-math is](#1-what-formal-math-is)
- [2. Three parallels in brief](#2-three-parallels-in-brief)
- [3. Candidate formalizations](#3-candidate-formalizations)
- [4. The certificate and a proof assistant](#4-the-certificate-and-a-proof-assistant)
- [5. Recommended first target](#5-recommended-first-target)
- [6. Methodological lessons](#6-methodological-lessons)
- [7. Outside contributions](#7-outside-contributions)
- [8. Scope of this note](#8-scope-of-this-note)

## 1. What formal-math is

**Purpose.** formal-math collects self-contained Lean 4 / Mathlib projects, "machine-checked Lean 4 formalizations
published by Anthropic" ([root README](https://github.com/anthropics/formal-math/blob/e1a4e6508154ea59f030480661590a9fe3018011/README.md)),
each a companion to a recent research paper.

- For zeta23 the record is explicit: Claude wrote all the Lean, and the paper's authors directed the formalization,
  reviewed its outputs and wrote no Lean by hand
  ([zeta23/README.md](https://github.com/anthropics/formal-math/blob/e1a4e6508154ea59f030480661590a9fe3018011/zeta23/README.md),
  "Provenance and attribution";
  [zeta23/formalization.yaml](https://github.com/anthropics/formal-math/blob/e1a4e6508154ea59f030480661590a9fe3018011/zeta23/formalization.yaml),
  `automation`).
- 3sum-apsp carries Anthropic copyright headers but, in the files read here, does not say who wrote the Lean.

There are two projects at this commit:

| Project | Paper | What is proved | Toolchain |
|---|---|---|---|
| `zeta23/` | Alpöge–Furman, arXiv:2608.13637 (analytic number theory) | At least 2/3 of the zeros of ζ are simple and on the critical line, and at least 5/6 are distinct. The same holds for primitive Dirichlet L-functions, with Montgomery–Taylor-window constants. Every analytic input is proved over Mathlib: Weil's explicit formula, Riemann–von Mangoldt, Stirling for Γ′/Γ on vertical lines, Montgomery–Vaughan. The theorems have no hypotheses and use only the three standard axioms. | Lean `v4.33.0-rc2` |
| `3sum-apsp/` | Alman–Vassilevska Williams, arXiv:2610.06783 (algorithms) | Word-RAM programs with proved running times: 3SUM in $`O(n^{1.9992})`$, APSP and the (min,+)-product in $`O(n^{2.99942})`$, and further items of the paper one by one. The five headline claims sit in a 139-line file that imports nothing, not even Mathlib. | Lean `v4.33.1` |

**Conventions.**

- **Trusted statements, untrusted proofs.** Each project follows the layout of the [Palomar registry](https://palomar-registry.org/)
  template:
  - `Challenge.lean` holds the theorem statements, imports only Mathlib, and leaves every proof as `sorry`. This is
    what a reader audits.
  - `Solution.lean` restates them and proves them by delegating to the library.
  - `comparator.json` lists the theorem names and the permitted axioms.

  [leanprover/comparator](https://github.com/leanprover/comparator) builds both sides in a sandbox and checks three
  things: that each solution states exactly its challenge statement, that only `propext`, `Classical.choice` and
  `Quot.sound` are used, and that the proofs replay in Lean's kernel and in the independent `nanoda` kernel. CI runs
  this on every changed project and fails on any `sorry` outside the challenge files
  ([`.github/workflows/lean-projects.yml`](https://github.com/anthropics/formal-math/blob/e1a4e6508154ea59f030480661590a9fe3018011/.github/workflows/lean-projects.yml),
  [`.github/scripts/comparator-check.sh`](https://github.com/anthropics/formal-math/blob/e1a4e6508154ea59f030480661590a9fe3018011/.github/scripts/comparator-check.sh)).
- **Metadata.** `zeta23/formalization.yaml` (format v0.4) records:
  - sources and authors ("Claude");
  - automation ("autonomous"; humans chose the targets and reviewed);
  - `fidelity.divergences`: how each informal statement was rendered, for example "liminf ≥ c" as an ε-form;
  - review status ("author-verified", plus cold reads of the statements by Claude instances with no project context);
  - an alignment table from paper statement to Lean name to status.
- **Audit record.** [zeta23/AUDIT.md](https://github.com/anthropics/formal-math/blob/e1a4e6508154ea59f030480661590a9fe3018011/zeta23/AUDIT.md)
  records, revision by revision:
  - the exact commands;
  - verbatim `#print axioms` output;
  - Comparator runs, with their run-mode caveats ("executed without the landrun sandbox", "nanoda off");
  - what was *not* run ("No full build or Comparator run was performed for this revision").
- **Minimal trusted text and explicit gaps.** [3sum-apsp/README.md](https://github.com/anthropics/formal-math/blob/e1a4e6508154ea59f030480661590a9fe3018011/3sum-apsp/README.md)
  has three sections: "What a reader must trust", "What is not proved" and "Where the Lean text departs from the paper".
  Departures are also flagged in the statements by comments beginning `NOTE`.
- **Kernel-only computation.** Numerical facts are checked with `decide +kernel`, never with `native_decide` (which
  would add an axiom that Comparator rejects).

**The numerical-certificate pattern.** The closest thing in formal-math to our layer A is `zeta23/Zeta23/PairCeiling/`,
the "bandwidth-one ceiling" ([NumericCert.lean](https://github.com/anthropics/formal-math/blob/e1a4e6508154ea59f030480661590a9fe3018011/zeta23/Zeta23/PairCeiling/NumericCert.lean),
[LawN256.lean](https://github.com/anthropics/formal-math/blob/e1a4e6508154ea59f030480661590a9fe3018011/zeta23/Zeta23/PairCeiling/LawN256.lean),
[CeilingLaw256.lean](https://github.com/anthropics/formal-math/blob/e1a4e6508154ea59f030480661590a9fe3018011/zeta23/Zeta23/PairCeiling/CeilingLaw256.lean)).
It works as follows.

1. A form factor S(j), j = 1…256, is enclosed outside Lean by 70-digit interval arithmetic. The enclosures come from an
   exact-rational certificate file, identified by its sha256 and "available from the authors".
2. Lean has a checker `check : CertData → Bool` that uses only integer arithmetic, at a common scale $`K = 2^{140}`$.
3. A soundness theorem `cert_of_check` says: if `check d = true`, then for *every* real sequence inside the enclosures
   the bounds hold.
4. The instance `check LawN256 = true` is proved by `decide +kernel`.
5. The final theorems carry one displayed hypothesis, `EnclOK`: that the true S lies in the enclosures. Everything
   downstream of the enclosures is kernel-checked.

Because of that hypothesis, these theorems are audited with `#print axioms` but kept out of the Comparator configuration
submitted to Palomar. This is the template for §4–5 below.

**Reusable pieces.** zeta23 also proves things our programme needs and Mathlib lacks:

- a digamma series and a Weierstrass product for 1/Γ (`Zeta23/GammaFacts/Series.lean`);
- complex Stirling asymptotics for ψ on vertical lines (`Zeta23/Analytic/Stirling.lean`, `GammaFacts/StirlingVert.lean`);
- a weighted argument principle on rectangles (`Zeta23.Analytic.rectangleIntegral'_mul_logDeriv`, used in `Zeta23/RvM/`).

They are Apache-2.0 and could be ported with attribution, as zeta23 itself ported files from PrimeNumberTheoremAnd.

**Contributions.** Both project READMEs say: "Research artifact. Not maintained and not accepting contributions." See §7.

**Text addressed to AI agents.** None was found in the repository. The files mention Claude only as the author of the
Lean code.

## 2. Three parallels in brief

1. **Computer-assisted certificates with a displayed hypothesis.** PairCeiling and our M = 1 certificate have the same
   architecture: rigorous enclosures computed outside the prover, then a finite test. The final Krawczyk step of
   A_certify §9 can move into the Lean kernel exactly as PairCeiling's row checks did. The enclosure step becomes an
   `EnclOK` hypothesis, produced independently by `validated_int.py` (arb) and by the CAPD re-check (A_certify.md §10).
   This is the most valuable parallel (§4–5).
2. **An unconditional formal core.** Several of our results are algebraic or special-function identities that Mathlib can
   already express: Proposition 1's parity induction, Proposition 2's reduction to Gauss's equation, the duplication
   identity, and the series and symmetry identities behind λ∞ = 2. Mathlib `v4.33.1` now has ₂F₁ as a power series, the
   digamma function, Pochhammer symbols and Legendre's duplication formula. These could become Palomar-style
   Challenge/Solution pairs with no hypotheses (§3).
3. **How claims are stated and audited.** formal-math separates "what a reader must trust" from "why it holds", records
   per statement how faithfully it renders the source, and keeps an audit log of exactly what was run. PROOF.md's
   "proved / cited / reduced" labels and the README's "How this work was done" are the same instinct in prose. Several
   of their devices can be adopted without writing any Lean (§6).

## 3. Candidate formalizations

Effort: **S** = days of agent work, under about 300 lines of Lean. **M** = weeks, a few hundred to a few thousand lines.
**L** = months, including theory that belongs in Mathlib. **XL** = a formalization research project of its own.
Prerequisites were checked by name against Mathlib `v4.33.1` (file names, and theorem names in the files listed in §8);
"missing" means not found there.

| # | Result (where) | What the Lean statement would say | Present in Mathlib v4.33.1 | Missing | Effort | Value |
|---|---|---|---|---|---|---|
| 1 | Inner linear constant (PROOF.md §2; README, inner problem) | $`2^{-s-1/2}/(\Gamma(\alpha)\Gamma(\beta)) = 1/(\sqrt{2\pi}\,\Gamma(1+s))`$ for every $`s \in \mathbb C`$, with α = (1+s)/2, β = α + 1/2 | `Complex.Gamma_mul_Gamma_add_half`, `Complex.cpow_add` | nothing | S (hours) | low as mathematics; high as the first end-to-end run of a Palomar-layout project here (§5) |
| 2 | Proposition 2, "C(n,s) is finite and non-zero" | for n ≥ 2, real ω ≠ 0, odd μ, s = iμω: no Γ argument in C(n,s) is a pole, and C(n,s) ≠ 0 | `Complex.Gamma_ne_zero` | nothing | S | low: the substance of Proposition 2 is the connection formula (row 5) |
| 3 | Proposition 2, the reduction | the first-order system with c = b = 1 implies the second-order equation for y = p − m; y = x w(x²) turns it into Gauss's equation with α, β, κ = (n+1)/2; the formula for p + m | `HasDerivAt` calculus, `field_simp`, `ring` | nothing | S–M | medium: pins down the algebra behind C(n,s) |
| 4 | Proposition 2, the solution | $`y = x\,{}_2F_1(\alpha,\beta;\kappa;x^2)`$ solves that equation on 0 \< x \< 1 | `ordinaryHypergeometric` (series, radius 1), power-series calculus | the hypergeometric differential equation for `₂F₁` | M | medium; also a natural contribution to Mathlib |
| 5 | Proposition 2, the light-cone coefficient | the connection formula (A&S 15.3.6), hence y is not analytic at x = 1 and the coefficient of $`(1-x^2)^{n/2-1-s}`$ is C(n,s) | ₂F₁ series, Gamma, Beta integral | Gauss's sum ₂F₁(a,b;c;1), Euler's integral, continuation of ₂F₁ to z = 1, connection formulas | L | high (a headline result), but most of the work is Mathlib infrastructure |
| 6 | Proposition 1 (and Lemma B1(M), adding the projection $`P_M`$) | Take a commutative ℚ-algebra R with a derivation $`\partial_\theta`$ and an odd/even-harmonic grading; this abstracts smooth periodic functions, and the proof uses nothing more. Then the ε-expanded system has exactly one solution in $`R[x][[\varepsilon]]`$ with the stated parities, lowest degrees and $`G_k(1) = 0`$, for any F, $`\omega_k`$, $`K_k`$ | `Polynomial`, `PowerSeries` (with inverses of units), `Derivation` | a small even/odd-polynomial API, a polynomial antiderivative, and a faithful transcription of the expanded system | M–L | high: the structural claim "the 1/D expansion never selects Δ"; unconditional, and the ansatz becomes explicit |
| 7 | Proposition 3 | after factoring out $`e^{-n\Phi}`$, the O(n) and O(1) parts of the linearized equations vanish exactly when A solves the transport equation, and the solution along characteristics is the one displayed | real calculus, `Real.log`, `Real.exp` | a light framework for formal series in 1/n | M | medium–low |
| 8 | Proposition 4 | singulant, exponent and constant of the late terms of the 1/n series of $`x\,{}_2F_1`$ | `Asymptotics`, Stirling | Watson's lemma, Darboux's method, Euler's integral for ₂F₁ | L–XL | medium; it is "a derivation, not yet a full proof", so the proof would have to be completed first |
| 9 | Heat form (GAMMA.md §3) | under $`s = \rho^2/2`$, $`y = \sqrt s\,Y`$, $`\sigma = \sqrt2\,s\,S`$, $`\xi = \rho/(-t)`$, $`\theta = -\omega\ln(-t)`$, the inner system is equivalent to $`Y_t = cS`$, $`S_t = c\,(Y_s + Y^3/4)`$, $`c_s = cY^2/4`$ | multivariable chain rule (`HasFDerivAt.comp`) | nothing; the work is bookkeeping of partial derivatives in two coordinate systems | M | medium: this change of variables carries the whole γ∞ = 1/2 argument |
| 10 | The λ = 2 mode (GAMMA.md §4) | (a) $`a_k = 2(k+1)\,d_{k+1}`$, $`h_m = 2(m+1)\,g_{m+1}`$; (b) $`P + \omega P' = -2F^2`$ is solvable harmonic by harmonic, since $`1 + i\nu\omega \ne 0`$; (c) the ε-derivative of "s-shift composed with time reparametrization" solves the linearized inner equations | formal power series, Fourier coefficients, `HasFDerivAt`, symmetry of second derivatives | nothing beyond row 9 | S for (a)–(b); M for (c) | medium; that λ∞ = 2 *is the growth rate* stays conditional on Conjecture 1 |
| 11 | Lemma B2 (C_selection §3.2) | the Jacobian of $`\mathcal F_1`$ on the slow manifold has eigenvalues $`c^2/(x(x^2-c^2))`$, −1/x, 0, 0 (a factorization of its characteristic polynomial) | `Matrix.charpoly`, `ring` | nothing | S | low–medium (already checked in sympy) |
| 12 | Lemma B5.0 (C_selection §9) | in inner variables the exact equations read $`\mathcal I + \nu^2\mathcal Q_1 + \nu^4\mathcal Q_2 = 0`$: only even powers of ν occur | polynomial identities | a transcription of the exact equations | S–M | low–medium |
| 13 | Lemma B5.1 (C_selection §10.1) | (1) continuing Φ through the upper (lower) half plane gives Φ = w ± iπ/2; (2) $`nw = 1/(2u) + 1/(4nu^2) + \dots`$ | `Complex.log` limits at the slit, the log series | nothing | S–M | medium: branch-choice signs are a classic source of error |
| 14 | Lemmas B8.1–B8.2 (C_selection §11) | Re Φ is non-decreasing along the progressive paths; the anti-Stokes curve runs from $`1/\sqrt2`$ to $`\infty e^{i\pi/4}`$; the only Stokes line is x > 1 | real calculus, monotonicity from the sign of a derivative | nothing | S–M | medium: the domain geometry of layer B, proved by hand here |
| 15 | Lemma B6′ (C_selection §12) | the Volterra/Gronwall bound $`\sup\lvert r\rvert \le C\lvert c\rvert/n`$ | Gronwall (`norm_le_gronwallBound_of_norm_deriv_right_le`), matrix exponential | fundamental matrices $`t^{\mathcal T_0}(I + O(t))`$ at a regular singular point; explicit constants | M–L | medium |
| 16 | D_entire §6 | $`\mathrm{base}_j T_j = 1/(\Gamma(a)\Gamma(b))`$; the Darboux solution of $`\tilde d_{j+1} = a_j\tilde d_j - b_j`$; the exact cancellation of the O(1/j) parts; $`r_m`$ increasing (log-convexity) | `ascPochhammer`, `Complex.Gamma_add_one`, `Complex.digamma` with its functional equation, Hurwitz zeta | the explicit rational functions F, G (large but mechanical); verified evaluation of ψ and ζ(k, ·) | S for the identities, M for the bounds; the numerical σ₁ enclosure stays outside | medium: σ₁ feeds route A3 but is not yet a certificate |
| 17 | Krawczyk step of the M = 1 certificate (A_certify §5.1, §9) | `krawczyk_of_check` and the M = 1 instance under `EnclOK` (§5 below) | Banach's fixed point theorem on complete subsets (`ContractingWith.exists_fixedPoint'`), the mean value inequality (`Convex.norm_image_sub_le_of_norm_hasFDerivWithin_le`), the ℓ∞ operator norm (`Matrix.linfty_opNorm_mulVec`), `decide +kernel` | the Krawczyk theorem itself and an interval-matrix checker | M | **high**: the logic and arithmetic of the project's only certified result become kernel-checked, and the interface to the integrator is fixed |
| 18 | Start values and validated integration (A_certify §4.4, §9) | the lateral solution exists, lies in the computed balls, and so do its parameter derivatives | ODE existence and uniqueness, flows Lipschitz in the initial point, Gronwall | rounding-aware interval/ball arithmetic, Taylor models, differentiability of flows in parameters (the variational equation), Kummer U | XL | high in principle; in practice keep as `EnclOK`, produced by two independent implementations |

Not in Mathlib `v4.33.1`, by file name: a Brouwer or Poincaré–Miranda theorem, interval arithmetic, and a
Krawczyk/Newton–Kantorovich theorem. Miranda-type steps (`pm_dryrun.py`, route A3) should therefore be phrased in the
contraction form of §4 when they go to Lean.

## 4. The certificate and a proof assistant

**What moves into the kernel.** The M = 1 certificate has four stages. Only the last is a finite test on numbers, and it
is exactly where PairCeiling's pattern applies.

| Stage | Now | With the PairCeiling pattern |
|---|---|---|
| (i) start values, a contraction in a weighted space (`fixed_point.py`) | arb | stays outside; part of `EnclOK` |
| (ii) validated Lohner integration with variational columns (`validated_int.py`) | arb; CAPD re-check done (§10; exact hex bounds in `capd_check/results/capd_enclosures_*.json`) | stays outside; part of `EnclOK`, produced twice |
| normalization κ (`krawczyk.normalization`) | arb | stays outside; enters `EnclOK` through g = κ J_D |
| (iv) Krawczyk test (`krawczyk.krawczyk`) on the saved `validated_int_krawczyk_1e-9.json` | arb, about 20 lines of Python | a Lean theorem proved once, and a rational check of the saved data by `decide +kernel` |

The data are tiny: a centre, a half-width, a 2-vector of balls for $`g(\hat x)`$, a 2×2 interval matrix J(X) and the
float matrix Y. arb's midpoints are dyadic, so every lower and upper end is an exact rational. PairCeiling scales
everything to integers at a power of two for fast kernel evaluation, and the same works here.

**Krawczyk without Brouwer.** The test of A_certify §5.1 is

```math
K(X) = \hat x - Y\,g(\hat x) + \big(I - Y\,J(X)\big)(X - \hat x) \subset \mathrm{int}\,X, \qquad X = \hat x + [-h, h]^2 .
```

Let $`M_{ij}`$ be the largest modulus in the interval entry $`(I - Y J(X))_{ij}`$. Then the test is the same as

```math
\lvert Y g(\hat x)\rvert_i + h\sum_j M_{ij} \lt h, \qquad i = 1, 2 .
```

This is in exact arithmetic. arb's outward rounding only makes the computed test stricter, and `krawczyk.py` evaluates
exactly this combination. Put G(x) = x − Y g(x). Since DG(x) = I − Y Dg(x) lies entrywise in I − Y J(X), the mean value
inequality gives

```math
\lvert G(x) - \hat x\rvert_i \le \lvert Y g(\hat x)\rvert_i + h\sum_j M_{ij} \lt h, \qquad
\lVert G(x) - G(x')\rVert_\infty \le q\,\lVert x - x'\rVert_\infty, \qquad q = \max_i \sum_j M_{ij} \lt 1 .
```

So G maps the closed box into itself and is a contraction, and Banach's theorem gives exactly one fixed point in X.
For every A in J(X), $`\lVert I - YA\rVert_\infty \le q \lt 1`$, so YA is invertible. Hence Y is invertible, the fixed
points of G are exactly the zeros of g, and Dg is invertible at the zero. Every step is in Mathlib except the Krawczyk
theorem itself, which is this short argument. With equal half-widths, as in our box, the box is a closed ball for
Mathlib's sup norm on `Fin 2 → ℝ`. Unequal half-widths only need a diagonal rescaling.

**What the Lean theorem would and would not say.** It would say: for *any* C¹ map g on the box whose value at the centre
and whose Jacobian over the box lie in the recorded enclosures, g has exactly one zero in the box, and it is simple. It
would not say that the enclosures are right for the M = 1 selection map. That is `EnclOK`, and it rests on
`fixed_point.py`, `validated_int.py` and arb, as A_certify §9 already states.

Two things are gained:

- the logic and arithmetic of step (iv) no longer rest on Python;
- the interface becomes precise: a file of rational enclosures and the theorem that consumes it.

The CAPD re-check then has a sharp target: produce an independent enclosure file for the same box and pass the same
Lean check. Two independent producers of `EnclOK` and one kernel-checked consumer are the analogue of formal-math's two
kernels.

**Further out** (all from memory, not checked in this session):

- Lean 4 has at least one verified floating-point interval-arithmetic library (Geoffrey Irving's `interval`, used for his
  Mandelbrot-set formalization).
- Isabelle/HOL has Immler's verified ODE solver, which re-checked Tucker's computer-assisted proof of the Lorenz attractor.
- Coq has Melquiond's CoqInterval.

Only the Isabelle route integrates ODEs inside a prover, as far as is known here. Bringing step (ii) into Lean would be
a research project of its own (row 18 above). The realistic plan is the split above.

## 5. Recommended first target

**Step 0, a warm-up (hours).** Set up one Lake project in the Palomar layout: `Challenge.lean`, `Solution.lean`,
`comparator.json`, `lean-toolchain` pinned to formal-math's `v4.33.1`. Prove row 1 in it. This exercises the whole
pipeline (Mathlib cache, Comparator, `#print axioms`, CI) on a statement with no risk.

```lean
import Mathlib

open Complex

/-- PROOF.md §2, "Inner linear constant": with α = (1+s)/2 and β = (2+s)/2 = α + 1/2,
    2^(-s-1/2) / (Γ(α) Γ(β)) = 1 / (√(2π) Γ(1+s)).
    True for every s : ℂ under Mathlib's conventions (Γ = 0 at its poles, z / 0 = 0). -/
theorem inner_weber_constant (s : ℂ) :
    (2 : ℂ) ^ (-s - 1 / 2) / (Gamma ((1 + s) / 2) * Gamma ((2 + s) / 2))
      = 1 / ((Real.sqrt (2 * Real.pi) : ℂ) * Gamma (1 + s)) := by
  -- Legendre: Γ(α) Γ(α + 1/2) = Γ(2α) 2^(1-2α) √π, with 2α = 1 + s and 1 - 2α = -s.
  have h := Complex.Gamma_mul_Gamma_add_half ((1 + s) / 2)
  rw [show (1 + s) / 2 + 1 / 2 = (2 + s) / 2 by ring, show 2 * ((1 + s) / 2) = 1 + s by ring,
    show 1 - 2 * ((1 + s) / 2) = -s by ring] at h
  rw [h]
  -- remains: 2^(-s-1/2) = 2^(-s) 2^(-1/2) (Complex.cpow_add, 2 ≠ 0) and √(2π) = √2 √π
  sorry
```

**The target: a kernel-checked Krawczyk step for the M = 1 certificate (row 17).** It has the highest value for its
effort. It mirrors formal-math's own precedent closely, it complements the CAPD re-check, and it is reusable for A(M=5)
and for route A3's contour certificates. The statement:

```lean
import Mathlib

open Metric

namespace DSS.Krawczyk

/-- ℝ² with Mathlib's sup norm, so a box of half-width h is a closed ball. -/
abbrev V := Fin 2 → ℝ
abbrev M2 := Matrix (Fin 2) (Fin 2) ℝ

/-- One Krawczyk test. Every entry is an exact rational (arb's midpoints and radii are dyadic). -/
structure Data where
  xhat : Fin 2 → ℚ             -- box centre (A, ω/ω∞)
  h : ℚ                        -- half-width, the same in both coordinates
  glo : Fin 2 → ℚ              -- enclosure of g(x̂)
  ghi : Fin 2 → ℚ
  Jlo : Fin 2 → Fin 2 → ℚ      -- entrywise enclosure of Dg over the box
  Jhi : Fin 2 → Fin 2 → ℚ
  Y : Fin 2 → Fin 2 → ℚ        -- preconditioner: any matrix (krawczyk.py uses the rounded inverse of mid J)

def Data.box (d : Data) : Set V := closedBall (fun i => (d.xhat i : ℝ)) d.h

/-- The displayed hypothesis: what fixed_point.py + validated_int.py (and, independently,
    the CAPD re-check) certify outside Lean. -/
def EnclOK (d : Data) (g : V → V) (Dg : V → M2) : Prop :=
  (∀ i, (d.glo i : ℝ) ≤ g (fun k => d.xhat k) i ∧ g (fun k => d.xhat k) i ≤ d.ghi i) ∧
  ∀ x ∈ d.box, ∀ i j, (d.Jlo i j : ℝ) ≤ Dg x i j ∧ Dg x i j ≤ d.Jhi i j

/-- The test in exact rational interval arithmetic: for each i,
    max |Y g(x̂)|_i + h · Σ_j max |(I - Y J)_ij| < h.  (Body omitted; a few lines.) -/
def check (d : Data) : Bool := sorry

/-- Krawczyk's test in contraction form (§4): Banach's fixed point theorem, no Brouwer. -/
theorem krawczyk_of_check (d : Data) (hc : check d = true) (g : V → V) (Dg : V → M2)
    (hg : ∀ x ∈ d.box,
      HasFDerivWithinAt g (LinearMap.toContinuousLinearMap (Matrix.toLin' (Dg x))) d.box x)
    (hE : EnclOK d g Dg) :
    (∃! x, x ∈ d.box ∧ g x = 0) ∧ ∀ x ∈ d.box, g x = 0 → (Dg x).det ≠ 0 := by
  sorry

/-- The data of A_certify §9, generated from solver/proof/validated_int_krawczyk_1e-9.json:
    x̂ = (2.5421480475, 0.7769940136), h = 1e-9. -/
def M1 : Data := sorry  -- a generated file

theorem M1_check : check M1 = true := by decide +kernel

/-- The truncated one-harmonic selection condition has exactly one zero in the box, and it is
    simple, for every map that the validated integration encloses. -/
theorem M1_unique_simple_zero (g : V → V) (Dg : V → M2)
    (hg : ∀ x ∈ M1.box,
      HasFDerivWithinAt g (LinearMap.toContinuousLinearMap (Matrix.toLin' (Dg x))) M1.box x)
    (hE : EnclOK M1 g Dg) :
    (∃! x, x ∈ M1.box ∧ g x = 0) ∧ ∀ x ∈ M1.box, g x = 0 → (Dg x).det ≠ 0 :=
  krawczyk_of_check M1 M1_check g Dg hg hE

end DSS.Krawczyk
```

Not compiled; names follow Mathlib `v4.33.1` as read, but no elaboration was attempted.

**Plan.**

1. Write `check` over ℤ at a fixed power-of-two scale, as PairCeiling does.
2. Prove `krawczyk_of_check` by the argument of §4. This is the only real proof: a few hundred lines.
3. Write a small exporter from the saved JSON (arb's "[mid +/- rad]" strings) to `M1.lean`, recording the JSON's sha256
   in the file header as LawN256.lean does. Prove `M1_check` by `decide +kernel`.
4. Audit with `#print axioms`. Expect only the standard three; PairCeiling's checks need even fewer.
5. When CAPD produces its own enclosure file, add `M1_capd` and check it the same way.

Total: weeks, effort M. Like PairCeiling, the theorems carry `EnclOK` and do not belong in a Comparator configuration
meant as an unconditional claim.

**After that.** The best unconditional target is Proposition 1 (row 6). It is pure algebra, it is the structural result
the whole large-D section rests on, and stating it formally forces the ansatz to be written down completely. The
duplication identity, the reduction of Proposition 2 (row 3), the heat form (row 9) and the λ = 2 identities (row 10)
are smaller companions for the same project.

## 6. Methodological lessons

| | formal-math | This project | What to adopt |
|---|---|---|---|
| What a reader must trust | a short trusted file per topic (`Challenge.lean`; for 3sum-apsp a 139-line file that imports nothing), separate from proofs | statements and proofs are interleaved in README, PROOF.md and proof/*.md | a one-line-per-claim statement index: the statement, where it is proved, its status, its checker |
| Meaning of "proved" | kernel-checked, standard axioms only, two kernels | proved by hand by an agent, re-derived by a separate agent, checked in sympy or arb; no expert review | a fixed status vocabulary, used everywhere: *proved (by hand)*, *checked symbolically*, *certified (computer-assisted; trusted code named)*, *formalized (Lean)*, *conditional (hypothesis named)* |
| Conditional results | the hypothesis is displayed and named (`EnclOK`), and such theorems are kept out of the submitted configuration | A_certify §9 says the certificate "rests on the correctness of the code" | name the interface between numerics and logic, and say which programs produce it |
| Fidelity | `fidelity.divergences`, `NOTE` comments, "What is not proved", "Where the Lean text departs from the paper" | the Proposition 1 ansatz paragraph; "a derivation (not yet a full proof)" for Proposition 4 | a short "departures and gaps" list per document, e.g. that M = 1 is a truncated model, not the physical problem |
| Audit | AUDIT.md per revision: commands, toolchain, verbatim outputs, run-mode caveats, what was not run | "Reproduce" sections and saved JSON runs | for each certificate: tool versions (python-flint, arb, precision), sha256 of saved runs, verbatim pass/fail output, and an explicit list of what has not been re-run |
| Independent checks | Comparator with the `nanoda` second kernel; an independent re-run by a second party | two solvers; two Stokes-multiplier implementations sharing no code; the CAPD re-check | already the same principle; record the outcomes the AUDIT.md way |
| Review | zeta23: `review.status`, reviewers named; statements cold-read by fresh Claude instances | a referee pass by a fresh agent; no human expert | record per claim who reviewed it and how, and keep agent review and human review distinct |
| Disclosure | zeta23: structured (`authors: Claude`; humans directed, chose targets, reviewed, wrote no Lean) | "How this work was done", in prose | the prose is adequate; add the structured form if a Lean project is started |

The single most useful habit to copy is the formal-math split of every claim into three parts:

- the statement a reader must trust;
- the checker that establishes it;
- a record of the check actually having been run.

It costs nothing to adopt in Markdown, and it makes a later move to Lean mechanical.

## 7. Outside contributions

- Both projects say "Research artifact. Not maintained and not accepting contributions." There is no CONTRIBUTING file,
  no issue template and no list of open problems in the repository.
- The root README has a "To add a project" recipe (a lakefile, a Mathlib-only `Challenge`, a `Solution`, a Comparator
  configuration, a row in the table). Read with the per-project notices, this is the maintainers' own procedure, not an
  invitation.
- GitHub metadata, such as whether issues are enabled and who opens pull requests, could not be queried from this session.

The practical route for us is a separate Lean project of our own in the same Palomar layout. Once it has an
unconditional Challenge/Solution pair, it could be submitted to the [Palomar registry](https://palomar-registry.org/), the
venue formal-math itself uses. formal-math's README describes submission through <https://submit.palomar-registry.org/>
with a full commit SHA, a selected project directory and a Comparator configuration path. Palomar's own policy
([PalomarPolicy CONTRIBUTING](https://github.com/PalomarRegistry/PalomarPolicy/blob/main/CONTRIBUTING.md)) was not read in
this session. Nothing was submitted, opened or posted anywhere.

## 8. Scope of this note

**Read in formal-math** (commit `e1a4e65`):

- the root README, `.github/workflows/lean-projects.yml` and `.github/scripts/comparator-check.sh`;
- `zeta23/README.md`, `AUDIT.md`, `formalization.yaml`, `comparator*.json`, `lakefile.toml`, part of `Challenge.lean`;
- `Zeta23/PairCeiling/NumericCert.lean`, `LawN256.lean` (header), `CeilingLaw256.lean`;
- the headers of `Zeta23/Analytic/Stirling.lean` and `GammaFacts/StirlingVert.lean`;
- `3sum-apsp/README.md` and the head of `EndStatement.lean`.

The `docs/` files of 3sum-apsp were not read.

**Read in Mathlib** (`v4.33.1`, commit `0df444a`, by sparse checkout; file names over the whole tree):

- `Analysis/SpecialFunctions/Gamma/Beta.lean` (`Complex.Gamma_mul_Gamma_add_half`);
- `OrdinaryHypergeometric.lean`, `RegularizedHypergeometric.lean`, `Gamma/Digamma.lean`;
- `Analysis/ODE/{ExistUnique,PicardLindelof,Gronwall,Transform}.lean`;
- `Topology/MetricSpace/Contracting.lean`, `Analysis/Matrix/Normed.lean`.

**Read here:** README.md (overview, Large-D section, "How this work was done"), PROOF.md, A_certify.md (§1–5, §9),
C_selection.md (§3, §9–13), D_entire.md (§6), GAMMA.md, and `solver/proof/krawczyk.py` with the saved certificate JSON.

No Lean toolchain was installed and nothing was compiled.

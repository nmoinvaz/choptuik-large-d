# P5 scoping note: the type bound, and a conditional Theorem A(∞)

Queue item P5 of [notes/PROOF_QUEUE.md](../notes/PROOF_QUEUE.md): turn the amplitude series of the harmonic-1
multiplier into a certified zero, with the exponential type of S₁ as the named hypothesis. This note fixes the
statement, the tools, what is computable now and the cost, before any long computation. Written 2026-10-28 by Claude
(Anthropic) under Nathan Moinvaziri's direction; not reviewed. Everything numerical here is *numerical evidence*
from `solver/proof/entire_cert.py` on the seventeen-order scans (`solver/inner_amp/scan35J1000`, σ₀…σ₁₇ at
ω/ω∞ = 0.99, 1.00, 1.01); nothing in it is a bound. [D_entire.md](D_entire.md) is the background (written at twelve
orders; its sections 4, 5 and 7 are the starting point).

## 1. The statement needed

S₁/S₁⁽¹⁾ = Σ σ_k(ω) xᵏ, x = A², is the harmonic-1 multiplier of the tuned family (every S_ν, |ν| ≥ 3, tuned to zero
order by order in the amplitude). Theorem A(∞) is Theorem A of [PROOF.md](../PROOF.md) for that family, the harmonic
tail included, and in this route it is conditional on three named hypotheses:

- **(H_σ) The computed coefficients.** σ_k(ω) for k ≤ K at the scanned ω, each within a stated error. Rigorous today
  for σ₁ only (D_entire.md §6, `exact_sigma.py`); for σ₂…σ_K the fit errors (10⁻¹² relative at k = 17) are the
  hypothesis until the order-by-order enclosures exist.
- **(H_T) The type bound.** S₁(·, ω) is entire of exponential type, and for every k > K and every ω in the box the
  coefficients of G = e^{−κ̄x} S₁ obey |g_k(ω)| ≤ C_G rᵏ/Γ(k + b) with explicit C_G, r, b. By Pólya's theorem this
  is a statement about the Borel transform B(t) = Σ k! σ_k t^{−k−1}: its singularities lie in a stated convex set
  (the hull of the arc of D_entire.md §7.1), with a bound on B on a contour around it; r is the distance from κ̄ to the
  farthest point of that set and b the endpoint exponent of the density. The hypothesis is *explicit*, so that each new
  order tests it (section 2).
- **(H_ω) Uniformity in ω.** σ_k(ω) analytic in a stated strip around [ω₋, ω₊] with a stated bound, so that the
  scanned nodes control the whole box (interpolation with a remainder), or enclosures of ∂σ_k/∂ω.

**Conclusion.** In an explicit box B = [A₋, A₊] × [ω₋, ω₊], S₁(A, ω) has exactly one zero and it is simple. The proof is
a Rouché test in x at fixed ω (|G − G_K| < min |G_K| on a circle around x₀) or a Krawczyk test for (Re G, Im G) in
(A, ω/ω∞), in ball arithmetic, with the tail of (H_T) and the errors of (H_σ) and (H_ω) as the enclosure error. The
status word is *conditional*, with (H_σ), (H_T), (H_ω) named; dropping (H_σ) is the rigorous-coefficient project of
D_entire.md §4 item 1, dropping (H_T) is the new mathematics.

What it serves. Layer A with no truncation in harmonics: the zero whose existence Theorem B assumes, at the
untruncated profile. The truncated route (Theorem A(M), M = 1, 3 certified, M = 5 is P3) needs gap (b), uniformity in
M (P6); this route trades that for the type bound, in which the harmonic tail becomes a tail in the amplitude.

## 2. What the seventeen-order data say

`cd solver/proof && python3 entire_cert.py 17 --scan ../inner_amp/scan35J1000` (seconds; the default,
`entire_cert.py 12`, reproduces D_entire.md's twelve-order numbers unchanged).

| quantity (ω = ω∞) | K = 12 (D_entire.md) | K = 17 |
|---|---|---|
| ratio-law fit κ⋆ (far end of the arc), type ∣κ⋆∣, β | 1.279 + 1.081i, 1.675, 4.16 − 1.60i | 1.317 + 1.085i, 1.706, 4.38 − 1.79i |
| constant C = max_k ∣σ_k∣ Γ(k + Re β)/∣κ⋆∣ᵏ, k = 4…K | 27.3 (falling to 25.0 at k = 12) | 39.7 (falling to 35.6 at k = 17) |
| Rouché minimum of ∣G_K∣ on ∣x − x₀∣ = 10⁻³ | 5.7×10⁻⁵ | 3.7×10⁻⁵ |
| enclosure error of G allowed by a box of radius 10⁻⁴ (Krawczyk) | 2.6×10⁻⁵ | 1.7×10⁻⁵ |
| tail beyond K at x₀, S₁ form | 400 | 35 |
| tail beyond K at x₀, G form with rᵏ/k! | 3.5×10⁻³ | 6.1×10⁻⁵ |
| tail beyond K at x₀, G form with rᵏ/Γ(k + Re β) | 1.8×10⁻⁴ | 2.3×10⁻⁶ |

Three findings.

1. **The ratio law is not a constant-increment law.** The increments of σ_k/σ_{k+1} drift, 0.469 − 0.386i at k = 9
   to 0.437 − 0.337i at k = 16, so the fitted far end κ⋆ moves with the window (type 1.675 at twelve orders, 1.706 at
   seventeen) and the constant C with it. A per-coefficient bound fixed at one order therefore has a margin that
   shrinks: the bound calibrated on k ≤ 12 (C = 27.3, a = 1.6746, b = 4.16) holds at k = 13…17 with
   |σ_k|/bound = 0.912, 0.910, 0.911, 0.915, 0.923, and the one calibrated on k ≤ 14 at k = 15…17 with 0.904,
   0.907, 0.914. The hypothesis must carry an explicit margin (a factor in C, or r above the fitted far end), and it
   is falsifiable at every new order.
2. **Only the G form is usable, and only with the endpoint exponent.** The plain tail is 35 at seventeen orders (the
   terms at x₀ peak near 240 and cancel). With κ̄ chosen as in D_entire.md §2 and r = max |t − κ̄| over the arc,
   the rᵏ/k! tail is 6.1×10⁻⁵ (a box of radius 10⁻³, not 10⁻⁴); with rᵏ/Γ(k + Re β), which is what the density's
   vanishing at the far end gives, it is 2.3×10⁻⁶, sixteen times inside the Rouché margin at radius 10⁻³ in x and
   seven times inside the Krawczyk need at radius 10⁻⁴ in (A, ω/ω∞). The calibrated C_G′ = 11.4 is set by the last
   coefficient (κ̄ zeroes g_{K−1}, so g_K carries what is left); the earlier ones give 3–5.
3. **The σ₁₈, σ₁₉ and σ₂₀ tests passed; σ₂₁ is next.** The K = 17 calibration predicts |σ₁₈| ≤ 3.63×10⁻¹⁵ and |σ₁₉| ≤
   2.77×10⁻¹⁶. The committed σ₁₈ scans (`solver/inner_amp/scan37J1000`, T87, 2026-10-28) give |σ₁₈| = 3.28×10⁻¹⁵ at
   ω∞, 0.90 of the bound, the same margin as the orders before (3.35×10⁻¹⁵ and 3.20×10⁻¹⁵ at the nodes, 0.92 and 0.88
   of the ω∞ bound). Under the bound calibrated on k ≤ 12 the ratio |σ_k|/bound is 0.912, 0.910, 0.911, 0.915, 0.923,
   0.934 for k = 13…18 (`solver/proof/logs/entire_cert_K18.log`), still inside but rising by about 0.01 per order
   over the last three, which is the quantity to watch: a hypothesis with a fixed C must hold for every k, and the
   margin it needs is set by where that ratio levels off. Recalibrated at K = 18 (a = 1.7167, b = 4.46, C = 45.6)
   the bound predicts |σ₁₉| ≤ 2.79×10⁻¹⁶ (2.77×10⁻¹⁶ from K = 17). **σ₁₉ at ω∞ passed it** (T88's pilot (a),
   `solver/inner_amp/scan39J1000/sigF_w1.0000_K39_J1000_p960.json`, 2026-10-28 16:42 UTC): |σ₁₉| = 2.52×10⁻¹⁶,
   0.90 of the bound, the third order running at that margin, with |σ₁₈|/|σ₁₉| = 13.0 and the argument advancing
   by 0.76 (0.77 at the order before). Its fit error is 1.1×10⁻¹¹ of |σ₁₉| (2.0×10⁻¹² at σ₁₈), still three orders
   below what (H_σ) needs; the J 1250 pilot differs from it by 1.3×10⁻¹¹ of |σ₁₉|, 1.21 times that fit error, the
   first order where the cross-check exceeds the fit's own estimate (the J truncation and the precision each move
   σ₁₉ by two thirds of that fit error, PLANTED_SELECTION.md §11.0's amendment of 2026-10-29; the bound test is
   unaffected). With all three nodes in (`solver/proof/logs/entire_cert_K19.log`,
   2026-10-28 23:40 UTC; the per-node lines are new in `entire_cert.py`, and the K = 18 log gained the same four
   lines and nothing else): **σ₁₉ at wf 0.99 and 1.01 is 0.906 and 0.896 of the K = 18 bound calibrated on each
   node's own series** (0.901 at ω∞), the same 0.90 at all three. The k ≤ 12 calibration's ratio at k = 19 is
   0.965, 0.950, 0.935 at wf 0.99, 1.00, 1.01 (0.947, 0.934, 0.921 at k = 18), rising 0.014–0.018 per order, so a
   constant fixed at k ≤ 12 reaches 1 at the 0.99 node in about two orders: (H_T)'s constant has to be calibrated
   on the latest orders, where the margin holds at 0.90. Recalibrated at K = 19 (a = 1.7276, b = 4.54, C = 52.9)
   the bound predicts |σ₂₀| ≤ 2.06×10⁻¹⁷ (2.05×10⁻¹⁷ from K = 18). **σ₂₀ passed it** (T89, K 41, J 1250, 1280 bits,
   `solver/inner_amp/scan41J1250`; the ω∞ run landed 2026-10-30 03:03 UTC and the nodes 2026-10-31 03:58 UTC;
   `solver/proof/logs/entire_cert_K20.log`): |σ₂₀| = 1.86×10⁻¹⁷ at ω∞, 0.901 of the K = 19 bound, and at wf 0.99 and
   1.01 it is 0.905 and 0.896 of each node's own K = 19 bound, the fourth order running at 0.90. The k ≤ 12
   calibration's ratio at k = 20 is 0.988, 0.971, 0.953 at wf 0.99, 1.00, 1.01, up 0.018–0.023 from k = 19, so it
   reaches 1 at the 0.99 node at the next order, as predicted; the bound calibrated on the latest orders keeps its
   0.90. Recalibrated at K = 20 (a = 1.7386, b = 4.63, C = 62.0) the bound predicts |σ₂₁| ≤ 1.47×10⁻¹⁸. σ₂₀'s fit
   error at ω∞ is 7.2×10⁻¹³ of |σ₂₀| and the J 1500 pilot differs from it by 0.011 of that (PLANTED_SELECTION.md
   §12.0). σ₂₁ would be the next test; no run for it is queued.

So, numerically: at K = 17, a conditional certificate at radius about 10⁻⁴ in ω/ω∞ is within reach of (H_T) in its
sharpened form, and the coefficient errors of (H_σ) (fit errors ~10⁻¹² of each σ_k, 10⁻²⁵ absolute at k = 17) are
far below the budget. (H_ω) is the open part: three nodes 0.01 apart do not control a box to 10⁻⁵ in G by
interpolation alone (section 4, item 2).

**The certificate at fixed ω, done** (2026-10-28, `solver/proof/entire_cert_rig.py`; logs
`solver/proof/logs/entire_cert_rig_planted.log`, `entire_cert_rig_K17.log`, `entire_cert_rig_K18.log`). Ball
arithmetic throughout: the σ_k as
acb balls with their fit errors as radii, the g_k from them, Rouché on |x − x₀| = ρ against the exact midpoint
polynomial P (whose roots `acb_poly.roots()` isolates), with |G − P| ≤ δ + tail, δ the coefficient radii summed with
Rᵏ and the tail (H_T)'s sum bounded in arb. A planted function e^{κ₁x} + c e^{κ₂x} of known type with its zero placed
at x = 7.4 is certified, and the four mutations (disc moved to no zero, ρ = 7 with three zeros, the tail constant
×10⁵, the last coefficient unknown) each fail. On `scan35J1000` with K = 17, κ̄ from `entire_cert.best_kbar`,
C_G′ = 11.4 × 2 (margin), r = 0.749, b = 4.38:

```
cd solver/proof && python3 entire_cert_rig.py real ../inner_amp/scan35J1000 17 11.4 0.749 4.38 --rho=1e-3
```

certifies exactly one zero of S₁(·, ω) in |x − x₀(ω)| < 10⁻³ at each of ω/ω∞ = 0.99, 1.00, 1.01, with
(δ + tail)/min |P| = 0.125, 0.130, 0.137 (δ ≈ 3×10⁻⁸, tail ≈ 4–5×10⁻⁶). *Conditional* on (H_σ) and (H_T); it says
nothing across ω, so the real zero (A*, ω*) is not certified by it. That is the (H_ω) step.

Rerun at K = 18 on the σ₁₈ scans (2026-10-28, `logs/entire_cert_K18.log` for the constants, κ̄ = 1.033 + 0.372i,
r = 0.774, C_G′ = 12.0 × 2, b = 4.46; `logs/entire_cert_rig_K18.log`):

```
cd solver/proof && python3 entire_cert_rig.py real ../inner_amp/scan37J1000 18 12.0 0.774 4.46 --rho=1e-3
```

certifies one zero in the same discs at each node with (δ + tail)/min |P| = 0.105, 0.107, 0.112 (δ ≈ 7–9×10⁻⁸, tail
≈ 1.5–1.9×10⁻⁶ against a minimum of 1.5–1.7×10⁻⁵). One order more lowers the tail by 2.7 and the circle minimum by
1.8, so the margin improves slowly with K; the (H_ω) step, not K, is what the certificate now waits on.

## 3. The closest published tools

- **Entire functions of exponential type.** Pólya's theorem (Borel transform, conjugate indicator diagram) and the
  Paley–Wiener–Pólya coefficient bounds: Boas, *Entire Functions* (Academic Press, 1954) *(unverified)*; Levin,
  *Lectures on Entire Functions* (AMS, 1996) *(unverified)*. This is the standard form of (H_T).
- **The endpoint exponent.** Darboux's method and Watson's lemma for the late orders of a Laplace integral with an
  algebraic endpoint (Dingle 1973, verified in notes/REF_CHECK.md; already used for Proposition 4). Gives the
  Γ(k + β) in (H_T).
- **The certificate.** Rouché with interval evaluation on a circle; the Krawczyk operator (Krawczyk 1969 and
  Neumaier, checked for the mathematics paper, REF_CHECK.md's 2026-10-21 addendum; the code is
  `solver/proof/krawczyk.py`), in python-flint's acb balls.
- **Rigorous Stokes constants, order by order.** Baldomá, Capiński, Guardia and Seara (J. Nonlinear Sci. 33, 2023,
  verified) for the pattern: a validated lateral integration of a linear problem with an irregular singular point,
  which each σ_k is (D_entire.md §4 item 1); Costin (Duke 1998, verified) for the transseries setting.
- **The origin of the type.** No published result applies directly; D_entire.md §7.3 test 2 (the k-fold Borel
  convolution of the order-1 and order-2 transforms over a simplex, bounded kernel → Mᵏ/(k − 1)!) is the candidate
  mechanism. Nothing in notes/LIT_REVIEW.md covers an exponential-type bound for a tuned amplitude expansion of
  Stokes constants; a literature pass on "Borel transform of a Stokes multiplier as a function of a parameter" is part
  of item 4 below.

## 4. What is computable now, with the cost

1. **The conditional certificate at K = 17 (then 18).** A ball-arithmetic version of `entire_cert.py` sections 3–6:
   σ_k as acb balls with their fit errors, the G coefficients and the tail of (H_T) as balls, the Rouché test on a
   circle in x at each node, the Krawczyk test in (A, ω/ω∞) with (H_ω) as an explicit interpolation remainder, a
   mutation for each check (a scaled σ_k, a tail constant halved, a node dropped) and a planted run on a function of
   known type first (an explicit Laplace integral over an arc, whose coefficients the same code must certify). Output:
   the explicit box and the explicit constants (C_G, r, b, margin) of (H_T). Cost: 2–4 days of coordinator or cloud
   work; runs in seconds. This is D_entire.md §5's "feasible now, modulo engineering", with (H_σ) as a hypothesis in
   place of rigorous coefficients.
2. **(H_ω) by more nodes, measured first.** The Mac computes σ₀…σ₁₇ at ω/ω∞ = 0.995 and 1.005 (T86's recipe at two
   more nodes, about 4 h each with 2 processes; a task for after T88 if the author prefers it to σ₂₀). The planted
   check: predict σ_k(1.00) from the four other nodes and compare with the computed value; the miss, over k, is the
   interpolation error that (H_ω) must cover, and it says how many nodes a box of radius 10⁻⁴ needs. Alternative,
   cheaper in computer time and dearer in work: ∂σ_k/∂ω from the amplitude recursion with ω as a jet (the recursion
   is polynomial in ω), enclosed by the same late-order fit; not started.
   **Procedure for the measurement, fixed 2026-10-31 10:25 UTC, before T93's files were read.** T93 computed σ₀…σ₁₉ at
   ω/ω∞ = 0.995 and 1.005 (K 39, J 1250, 960 bits); with the J 1250 files at 0.99 and 1.01 (`scan41J1250`, 1280 bits) and
   1.00 (`scan39J1250p960`, 960 bits) that is five nodes 0.005 apart, all at J 1250 (the precision mix enters at the
   plateau-error level, 10⁻¹¹ of σ₁₉ at most, far below anything measured here). `solver/proof/homega_interp.py real`
   reports, for each k ≤ 19, in t = (ω/ω∞ − 1)/0.01: the relative miss of the cubic through the four outer nodes at the
   centre (m3, the four-node interpolation error), the misses of the quadratic through 0.99, 1.00, 1.01 at 0.995 and 1.005
   (m2∓, how wrong the three-node interpolation every estimator uses is halfway between its nodes), the radius of
   analyticity R = |a₃/a₄| from the degree-4 interpolant (exact for a simple pole: the aliasing cancels in that ratio on
   these symmetric nodes; √|a₂/a₄| beside it, against what a pure pole at R aliases to, as the consistency check of the
   single-pole picture at 30%), and the number of nodes a box of radius 10⁻⁴ in ω/ω∞ needs for a relative interpolation
   error below 10⁻⁶ under the single-pole picture (the single-pole function through the five node values, Chebyshev
   interpolation on n + 1 nodes, the smallest count whose error on the box is below 10⁻⁶; given only when the
   consistency check passes). No threshold is set on the real data: the deliverable is the table, read at k = 19 (the
   worst order) for the node count, and the m2 column as the measured (H_ω) error of the present three-node setup. The
   planted stage (`homega_interp.py planted`: a simple pole at t = 3 and at t = 1.5 on the real σ_k(1.00) scale) ran
   first and passed: R recovered exactly, the node count equal to the brute-force minimum at every k for both poles,
   and the control (σ_k(0.995) scaled by 1 + 10⁻⁶) moving m2− by 10⁻⁶; the real run repeats the control with
   `--mutate` and reports the shift. Two design changes were made on the planted data before any real file was read: a
   closed-form node-count formula (gain 1/R per node) overestimated the count by five nodes and was replaced by the
   brute-force count on the single-pole function; and √|a₂/a₄| as a second radius estimate aliases by a third for a pole
   at t = 1.5 and became a consistency check against its own aliasing instead. Output: `solver/proof/logs/homega_interp.json`.
3. **Rigorous σ₂…σ_K, (H_σ) dropped.** D_entire.md §6.6's scheme (Darboux telescoping, polygamma sums, log-convexity
   bounds) by induction on the order, or validated lateral integration order by order. Mechanical and heavy: weeks of
   agent work for σ₂…σ₆, a project (months) to k = 17. Not before item 4 says whether (H_T) is provable, because
   with (H_T) unproved the theorem stays conditional either way.
4. **The proof of (H_T): a scoping computation, not a proof attempt.** A cloud agent derives the order-3 and order-5
   Borel structure of the amplitude recursion (test 2 of D_entire.md §7.3): whether the order-(2k+1) multiplier is a
   k-fold convolution over a simplex of kernels bounded uniformly in k, which would give |σ_k| ≤ Mᵏ/(k − 1)! with
   explicit M, and whether the exponent's values on the simplex fill the arc. Report only, with the formulas and a
   numerical check of the predicted M against |κ⋆| ≈ 1.7. Cost: 1–2 weeks of an agent, open-ended; the first
   milestone (the order-3 structure, where σ₁ is already exact) is a few days and decides whether to continue.

## 5. Order of work and the decision points

| step | item | who | cost | decides |
|---|---|---|---|---|
| 1 | the conditional certificate (4.1) at K = 17, rerun at K = 18 | coordinator | 2–4 days; the fixed-ω part done at K = 17 and 18 on 2026-10-28 (section 2), the (A, ω) Krawczyk with (H_ω) remains | the box radius the route gives, with (H_T)'s constants fixed and falsifiable |
| 2 | the σ₁₈, σ₁₉ tests of (H_T) (section 2, item 3) | the Mac (T87, T88) | σ₁₈ and σ₁₉ passed at 0.90 of the bound at all three nodes (2026-10-28); the k ≤ 12 constant is at 0.965 at wf 0.99 and rising; σ₂₀ (T89, J raised) next | whether the calibrated bound keeps its margin |
| 3 | the (H_ω) measurement (4.2) | the Mac, after T88 | 8 h | the number of nodes a radius-10⁻⁴ box needs |
| 4 | the order-3 and order-5 Borel structure (4.4) | cloud agent | days, then 1–2 weeks | whether (H_T) is a theorem in reach or stays a hypothesis |
| 5 | rigorous σ₂…σ_K (4.3) | cloud agent | months | only if step 4 says (H_T) is provable |

Step 1 produces the deliverable of P5 as queued (the zero certified assuming an explicit type bound); steps 4 and 5
are what would make it unconditional, and P6's uniformity question becomes moot on this route if they succeed.

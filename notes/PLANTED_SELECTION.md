# Planted-truth test of the D = ∞ selection estimators

The README ("Where that leaves it") turns twelve amplitude coefficients σ₁…σ₁₂ of the harmonic-1 multiplier,
$`S_1/S_1^{(1)} = \sum_k \sigma_k x^k`$ with x = A², into the claim that the zero of S₁ is real at
ω*/ω∞ ∈ 0.99915–1.00011, with best estimates 0.9999–1.0001, i.e. Δ∞ = 1.6439 ± 0.0002. None of the estimators
behind that claim had been run on a function whose answer was known. This note runs them on such functions.

Sections 1–2 were written and committed **before any estimator was run by this test on any input** (real or planted).
The real-data outputs of the estimators were already published in the README, `crossing.txt`, `zero.txt` and
comms.txt, and were read before the freeze; that cannot be undone and is stated here. What the freeze guarantees is that
the plants, the corruptions, the detectors and the pass criteria below were fixed before this test produced any number.

## 1. The estimators, as the code implements them

All five read the σ_k at three frequencies, wf = ω/ω∞ ∈ {0.99, 1.00, 1.01} (`scan25phys/`, one JSON per wf, field
`sigma_str`), and look for the wf at which a zero x₀(wf) of S₁ near x ≈ 7.4 is real. With K coefficients:

- **τ_k.** $`\ln S_1 = \sum_k \tau_k x^k`$ by the recursion $`k\tau_k = k\sigma_k - \sum_{m<k} m\,\tau_m\sigma_{k-m}`$
  (`arg_crossing.taus`, identical to `zero_selection.log_series`). For an entire function of order 1,
  $`\tau_k = -\tfrac1k\sum_j x_j^{-k}`$ for k ≥ 2, the power sums of the reciprocal zeros; the zero nearest 0 dominates.
- **arg τ_k = π crossing, "from above"** (`plot_selection.crossings`, the README table). At each wf,
  f_k(wf) = (arg τ_k/π mod 2) − 1; a quadratic through the three points; the real root nearest 1 inside
  [0.96, 1.04]. `arg_crossing.py` does the same with linear interpolation between adjacent nodes (the
  `crossing.txt` lines, 4 decimals). Assumes x₀ is a simple zero and the nearest singularity of ln S₁.
- **Ratio estimate, "from below", exponent free** (`plot_selection.crossings`). g_k(wf) =
  $`\mathrm{Im}\sqrt{(k-1)\tau_{k-1}/(k\tau_k)}`$ (the square root of the x₀ that $`\tau_k = -c/(k x_0^k)`$ gives,
  independent of c); quadratic through the three nodes; root nearest 1. `arg_crossing.py`: the same with linear
  interpolation.
- **The bracket at order K** is [ratio_K, arg_K]: 0.99915–1.00011 at K = 12.
- **Padé of (ln S₁)′** (`zero_selection.zeros`). The n + 1 = K coefficients $`k\tau_k`$ of L′(x); every [m/q],
  m + q = n; the poles of each approximant with residue |res − 1| < 0.3 are zeros of S₁. The crossing in wf is not in a
  script: comms.txt (T24, T25) states "Im(eps) = 0 crossings, linear between the bracketing omegas, computed by
  script". This test reconstructs that rule: per approximant, at each node the accepted pole nearest x = 7.397, and the
  sign change of Im √x between adjacent nodes, linearly interpolated. "Central" approximants are m = 3 … n − 4
  ([3/8]–[7/4] at K = 12, the five the README quotes as 0.99987–0.99991).
- **Factored entire series** (`entire_cert.resummed`). $`G = e^{-\bar\kappa x}S_1`$ truncated at x^K; κ̄ chosen by
  Nelder–Mead at wf = 1.00 to minimise the last two terms at X0 = 7.397; the root of G_K from X0 at each node; a
  quadratic in wf through Im √x₀; root in (0.97, 1.03). README: 0.99987 (K = 11), 0.99988 (K = 12).
- **Aitken** of consecutive crossings: $`x_2 - (x_2-x_1)^2/((x_2-x_1)-(x_1-x_0))`$ on the arg crossings of orders
  k − 2, k − 1, k. `arg_crossing.py` applies it to its linear crossings (last three); the README quotes it on the
  quadratic ones ("0.99976 to 1.00017 for the last six").

The harness calls these functions themselves (imported from the repository, or `arg_crossing.py` run as a
subprocess), on plant JSONs written in the `scan25phys` format, so the plants go through the same reader and the same
code. The only reconstructed piece is the Padé crossing rule above.

## 2. Frozen controls

### 2.1 Common to all plants

- Nodes wf ∈ {0.99, 1.00, 1.01}; σ₀ = 1; coefficients computed at 50 digits and written with 30 significant digits
  into `sigma_str`; one directory per plant and per K, read by the production code exactly as `scan25phys/` is.
- Runs at K = 11, 12, 13 (σ₁…σ_K kept). K = 12 is the claim; 11 and 13 give the trend.
- **The planted zero.** $`x_0(\mathrm{wf}) = X_t + v\,(\mathrm{wf}-\omega_t) + w\,(\mathrm{wf}-\omega_t)^2`$ with X_t = 7.400 (real),
  v = 4.0 − 7.8i, w = −6.5 − 2.0i (the velocity and curvature of the zero across the three nodes, read off the real
  ratio estimates at k = 11 in `zero.txt`: an input feature, not the answer). Im x₀(ω_t) = 0 exactly; the other root of
  Im x₀ = 0 lies 3.9 away. **ω_t ∈ {0.9960, 0.9985, 0.9995, 1.0000, 1.00025, 1.0010, 1.0040}**: both sides of 1, on and off
  the middle node, near and far.

### 2.2 Plants

- **P-ARC (main family): a Laplace integral along the Borel arc, times the planted factor.**
  $`S(x,\mathrm{wf}) = (1 - x/x_0(\mathrm{wf}))\,L_{\mathrm{wf}}(x)/L_{\mathrm{wf}}(0)`$ with
  $`L(x) = \int_0^1\rho(s)\,e^{x\gamma(s)}\gamma'(s)\,ds`$, evaluated by 64-point Gauss–Legendre, so L is exactly a
  64-term exponential sum (closed form, zeros computable to any precision). γ: least-squares cubic in s through the
  [6/6] Borel–Padé poles of D_entire §7.1 and κ⋆ at that node (s by cumulative chord length). ρ(s) =
  $`(1-s)^{\beta-1}\sum_{m<M} c_m T_m(2s-1)`$ with β from D_entire §1 at that node (4.10 − 1.56i, 4.16 − 1.60i,
  4.23 − 1.64i), M = 6 (M = 8 if the weighted residual exceeds 10⁻²). c_m by complex linear least squares so that
  (1 − x/x_ref)·L matches the real σ₀…σ₁₂ of that node with weights 1/|σ_k|, x_ref = x₀(wf; ω_t = 1.0000). L is then
  held fixed and only x₀ moves with ω_t. Recorded: fit residual per k, terms |σ_k|x₀^k (must peak at 200–300), ratio
  law (κ⋆, β fitted on k = 6–11 as in `entire_cert.ratio_law`), the nearest other zeros of S (argument principle on
  circles plus Newton, |x| < 16). Required: no other zero within |x − x₀| < 1.
- **P-ARC-B (a different arc).** As P-ARC, with the arc replaced and the density refitted: (i) the straight chord from
  the first pole to κ⋆; (ii) the arc mirrored about that chord (bulging the other way); (iii) the original arc with the
  endpoint exponent β − 1 replaced by 1 (density vanishing linearly, β = 2). Same seven ω_t.
- **P-Z2 (a second zero nearby).** $`S = (1 - x/x_0)(1 - x/x_2)\,L_2`$, L₂ refitted with both factors, x₂ fixed in wf,
  x₂ ∈ {9.0 − 1.0i, 8.6 + 1.2i, 9.3}. (The real G_K's next zeros are at |x| ≈ 10–11.4.) Same seven ω_t.
- **P-NOISE.** P-ARC at ω_t = 1.0000 and 0.9985 plus independent complex Gaussian noise on every σ_k, k ≥ 1, at each
  node: (a) standard deviation sigma_err_k of the real JSON at that node (10 seeds); (b) relative noise ε|σ_k| with
  ε ∈ {10⁻¹², 10⁻⁹, 10⁻⁶, 10⁻⁴} (5 seeds each), to find where coverage breaks.
- **C (completions: the real coefficients with a stated tail; truth computed, not planted).** σ₀…σ₁₂ are the real
  `scan25phys` values at each node; for k ≥ 12, $`\sigma_{k+1} = \sigma_k\,\kappa/(k+\beta)`$ with (κ, β) from a complex
  linear fit of σ_k/σ_{k+1} = (k + β)/κ over the window C-W1: k = 6–11 (the `entire_cert` window), C-W2: 8–11,
  C-W3: 9–11; and C-EXP: the exponential sum of the [6/6] Borel–Padé approximant of the real a_k = k!σ_k, k ≤ 12
  (an exact completion; accepted only if every pole has |t| < 2). Truth: the zero x_R(wf) of the completed series
  (summed to k = 250, or the closed form, mp.findroot at 50 digits) at each node; ω_R = root of the quadratic through
  the three Im x_R(wf) (and through Im √x_R; the two must agree to 10⁻⁶). At K ≤ 12 the estimators see exactly the real
  data, so C asks: if the real S₁ had this tail, where is its zero, and do the real-data numbers cover it?

### 2.3 Nulls

- **N0, the harness is the production path.** On `scan25phys/` the harness must reproduce, to the printed digits:
  arg_12 = 1.00011, ratio_12 = 0.99915, arg_11 = 1.00020, ratio_11 = 0.99919 (README table); Padé central crossings
  0.99987–0.99991 (K = 12); G 0.99987 (K = 11), 0.99988 (K = 12); and `arg_crossing.py`'s stdout must equal
  `scan25phys/crossing.txt`.
- **N1, a lone simple zero.** $`S = e^{bx}(1 - x/x_0(\mathrm{wf}))`$, b = 0.89 + 0.55i: τ_k = −1/(k x₀^k) exactly. Every
  estimator must return ω_t to 10⁻⁶ for each of the seven ω_t.
- **N2, no zero.** S ≡ 1 and S = e^{bx}: no estimator may report a crossing (a printed number is a FAIL; no root or an
  exception is a pass).

### 2.4 Pass criteria per plant (K = 12 primary; 11 and 13 reported)

- **P1 coverage:** ratio_K ≤ ω_t ≤ arg_K.
- **P2 directions:** arg_k ≥ ω_t for every k = 8…K ("from above"); ratio_k ≤ ω_t for every k = 9…K ("from below").
- **P3 best estimates, at the claimed precision** (Δ∞ = 1.6439 ± 0.0002 is ω/ω∞ ∈ [0.99988, 1.00012], ±1.2×10⁻⁴):
  |mid of the Padé central crossings − ω_t| ≤ 1.2×10⁻⁴ and |G_K − ω_t| ≤ 1.2×10⁻⁴.
- **P4 Aitken:** |Aitken(arg_{K−2}, arg_{K−1}, arg_K) − ω_t| ≤ 2×10⁻⁴.

Every summary states its denominator (plants run / plants planned); a plant whose construction fails a requirement of
2.2 is reported as failed construction, not dropped.

### 2.5 Corruptions and detectors

Applied one at a time to P-ARC at ω_t = 1.0000 and to C-W1, each with its clean twin run alongside:

- **X1 sign flip:** σ_k → −σ_k at the wf = 1.00 node, k ∈ {12, 9, 5}.
- **X2 scale:** σ_k → (1 + 10⁻³)σ_k at the wf = 1.00 node, k ∈ {12, 9, 5, 2}; and σ₁₂ scaled at all three nodes.
- **X3 grid:** (a) all three wf labels shifted by +0.0005; (b) only the 1.01 node relabelled 1.0105; (c) only the 1.00
  node relabelled 1.0005 (data unchanged).

Detectors (thresholds fixed here; each must be silent on the clean twins):

- **D-arg:** the arg crossings exist for k = 8…K and decrease monotonically.
- **D-ratio:** the ratio estimates exist for k = 9…K and are non-decreasing.
- **D-bracket:** ratio_K ≤ arg_K, and the Padé central mid and G_K lie inside [ratio_K, arg_K].
- **D-padé:** every central approximant has an accepted pole near X0 at all three nodes, and the central crossings
  spread by ≤ 10⁻³.
- **D-G:** the K = 12 last-terms objective is ≤ the K = 11 one, and |G₁₂ − G₁₁| ≤ 5×10⁻⁴.
- **D-ratiolaw:** at each node, |second difference of σ_k/σ_{k+1}| ≤ 0.05 for k = 6…K−2.

An estimator **notices** a corruption if a detector fires; it **moves** if its output changes by more than 1.2×10⁻⁴.
"Moved and not noticed" is the dangerous case. X3(a) is expected to move everything by exactly +0.0005 and to be
invisible to every detector (no estimator has an independent ω scale); the test records whether that holds.

### 2.6 Controls that must be seen to fail

- The coverage check graded against ω_t + 0.002 on P-ARC (ω_t = 1.0000) must FAIL.
- N0 run on `scan23phys/` (eleven coefficients) against the twelve-coefficient expectations must FAIL.
- At least one detector must fire on X1 (σ₁₂ sign flip); if none fires on any corruption, the detectors are declared
  unable to fail.

### 2.7 Verdict rules

- **The bracket is honest for the class** if P1 holds at K = 12 for every plant of P-ARC, P-ARC-B, P-NOISE (a) and C.
  Failures in P-Z2 or at the larger P-NOISE (b) levels are reported as the conditions under which it is not.
- **Bias:** for each estimator, the signed errors (estimate − ω_t) over P-ARC, P-ARC-B and C. An estimator is called
  biased if at least 80% of its signed errors share a sign and the median exceeds 3×10⁻⁵ in size.
- **The best estimate (0.9999–1.0001) is supported** if P3 holds for at least 90% of those plants.

## 3. Results

Code: `solver/inner_amp/planted_estimators.py` (the harness), `planted_plants.py` (plants), `planted_run.py` (driver,
`nice -n 10 python3 planted_run.py STAGE`), `planted_post.py` (post-freeze additions), `planted_report.py` (prints every
table below from the records). Records: `planted_results.jsonl` (171 runs), `planted_checks.json` (construction checks,
completion truths, refit scans). All 148 planned runs exist (the report checks the denominator). 23 runs were added
after the freeze, and are marked *post-freeze* wherever they appear:

- REFIT, 13 runs: added after the P-ARC construction showed a 10⁻⁵ fit, to ask which ω_t the real coefficients prefer.
- C-ARC, 10 runs: added after the frozen completions put their zeros outside the bracket, to try smooth tails.

The order of events was: freeze (b30f613); nulls; P-ARC; C; corruptions; P-ARC-B; P-Z2; REFIT; noise; C-ARC; the zero
counts of section 3.4.

Deviations from section 2, all small: D-padé reads "near X0" as |pole − 7.397| ≤ 1. Noise (a) gives σ₁₃, which has no
sigma_err, the relative error of σ₁₂.

### 3.1 The harness is the production path; the nulls

| quantity | README / comms | harness |
|---|---|---|
| arg₁₂, ratio₁₂ | 1.00011, 0.99915 | 1.00011, 0.99915 |
| arg₁₁, ratio₁₁ | 1.00020, 0.99919 | 1.00020, 0.99919 |
| Padé central [3/8]–[7/4], K = 12 | 0.99987–0.99991 | 0.99987–0.99991 |
| G, K = 11 and 12 | 0.99987, 0.99988 | 0.99987, 0.99988 |
| `arg_crossing.py` stdout | `scan25phys/crossing.txt` | identical (modulo CRLF) |

- **N0 passes, 8/8.** Run on `scan23phys/` against the twelve-coefficient expectations, N0 fails, as required.
- **A discrepancy in the README's Aitken range.** The README quotes "Aitken's extrapolations of consecutive triples
  … 0.99976 to 1.00017 for the last six". This is not reproduced: consecutive triples of the quadratic crossings ending
  at k = 7…12 give 1.000662, 1.000168, 0.999762, 1.000044, 1.000135 and 0.999559.
- **N1 passes, 7/7.** A lone simple zero is recovered by arg, ratio, Padé and G at K = 11–13 to 3.5×10⁻⁷ at worst.
- **N2 passes, 2/2.** No estimator prints a number:
  - S ≡ 1 makes the ratio estimator raise ZeroDivisionError and G's findroot raise ValueError;
  - S = e^{bx} gives no crossing anywhere.

### 3.2 The plants

All the arc plants have a ratio law κ⋆ = 1.271–1.289 + 1.080i, type 1.668–1.682, β = 4.10–4.23 − (1.56–1.64)i, and a
peak term at x₀ of 236–253 at k = 9 (real: 244 at k = 9). In the table, "zeros" is the count by the argument principle
for |x| < 16, then the number located.

| family | max rel. fit residual, k ≤ 12 | zeros | nearest other zero |
|---|---|---|---|
| P-ARC | 1.6×10⁻⁵ | 3 / 3 | 12.1 + 0.9i (\|x − x₀\| = 4.8) |
| P-ARC-B chord | 3.9×10⁻⁵ | 4 / 3 | 12.4 + 0.8i (5.0); one zero not located, none within 1 of x₀ |
| P-ARC-B mirror | 8.5×10⁻⁵ | 4 / 4 | 13.0 + 0.15i (5.6) |
| P-ARC-B β = 2 | 4.6×10⁻⁵ | 3 / 3 | 11.9 + 0.9i (4.6) |
| P-Z2, x₂ = 9.0 − 1.0i / 8.6 + 1.2i / 9.3 | 7.6 / 5.4 / 4.1×10⁻³ | 7 / 5–7 | x₂ (1.7–1.9) |

Inside |x| < 11.8, every P-ARC and P-ARC-B plant has exactly one zero, x₀ (argument principle at wf = 1.00). So the
frozen requirement (no other zero within 1 of x₀) holds everywhere.

### 3.3 Coverage and errors in the frozen families

Passes / plants:

| family | P1, K = 11 / 12 / 13 | P3 Padé, K = 12 | P3 G, K = 12 / 13 | P4, K = 12 |
|---|---|---|---|---|
| P-ARC (7) | 7 / 7 / 7 | 0 | 0 / 7 | 0 |
| P-ARC-B chord (7) | 0 / 7 / 0 | 0 | 0 / 7 | 0 |
| P-ARC-B mirror (7) | 0 / 0 / 0 | 7 | 0 / 7 | 7 |
| P-ARC-B β = 2 (7) | 0 / 7 / 7 | 0 | 0 / 7 | 0 |
| C-W1–3 (3) | 0 / 0 / 0 | 0 | 0 / 0 | 0 |
| P-NOISE (a), sigma_err (20) | 20 / 20 / 20 | 0 | 0 / 20 | 0 |
| P-Z2 (21) | 0 / 0 / 0 | 0 | 0 / 0 | 0 |

P2 (all orders from the claimed side) holds in 0 of the 28 arc plants. Their crossing sequences oscillate between odd
and even orders. For P-ARC at ω_t = 1, arg_k for k = 6–12 is 1.00256, 1.00355, 1.00081, 1.00093, 1.00040, 1.00030,
1.00016, and ratio_k for k = 9–12 is 1.0017, 0.9953, 0.9993, 0.9986. The real sequences are monotone; section 3.6
explains the difference.

Signed errors at K = 12, (estimate − ω_t) × 10⁴, median [min, max]:

| family | arg₁₂ (top of bracket) | ratio₁₂ (bottom) | Padé central | G₁₂ | Aitken |
|---|---|---|---|---|---|
| P-ARC | +1.55 [+1.41, +1.69] | −13.9 [−15.0, −13.0] | −7.96 [−8.45, −7.27] | +2.97 [+2.74, +3.26] | +6.24 |
| chord | +0.71 [+0.62, +0.79] | −24.5 [−25.5, −23.6] | −4.30 [−4.73, −3.83] | +2.78 [+2.57, +3.01] | +2.19 |
| mirror | **−0.69** [−0.82, −0.56] | −49.6 [−53.6, −45.3] | +1.06 [+1.02, +1.12] | +3.48 [+3.16, +3.84] | +1.14 |
| β = 2 | +1.77 [+1.70, +1.85] | −23.2 [−26.4, −20.3] | −7.05 [−7.08, −6.94] | +6.15 [+6.02, +6.35] | +3.65 |
| C-W1 / W2 / W3 | +18.4 / +38.9 / +35.2 | +8.9 / +29.4 / +25.7 | +16.2 / +36.8 / +33.1 | +16.2 / +36.7 / +33.0 | +12.9 / +33.4 / +29.7 |

- **The errors hardly depend on ω_t.** Within a family the arg, Padé and G errors change by at most 1.2×10⁻⁴ over
  ω_t = 0.996–1.004 (the ratio errors by up to 8×10⁻⁴). They are properties of the function, not of where its zero
  sits.
- **The bracket is lopsided.** Its top edge (arg₁₂) errs by −0.8 to +1.9×10⁻⁴. Its bottom edge (ratio₁₂) is 1.3–5.4×10⁻³
  low. So the bracket covers mostly because of the ratio estimate's large negative bias. It fails whenever the arg
  crossing is not above the truth, which is the case for the mirrored arc, 7/7.
- **The best estimates miss the claimed ±1.2×10⁻⁴.** Padé passes only on the mirrored arc (7/28). G passes on 0/28 at
  K = 12 and on 28/28 at K = 13, where its errors are +0.31 to +0.75×10⁻⁴. On every arc family Padé and G disagree by
  2–13×10⁻⁴. On the real data they agree to 10⁻⁵ (0.99989, 0.99988).
- **The bias rule** (P-ARC, P-ARC-B and C-W pooled, n = 31):
  - G is biased high: 100% of errors positive, median +3.4×10⁻⁴.
  - Aitken is biased high: 100% positive, median +3.6×10⁻⁴.
  - The ratio estimate is biased low: 90% negative, median −2.4×10⁻³.
  - The arg crossing is not biased by the rule: 77% positive, median +1.5×10⁻⁴.
  - Padé is not biased by the rule: 32% positive, median −4.5×10⁻⁴.
- **A second zero near x₀ fails loudly, not silently** (P-Z2, x₂ at |x| ≈ 9). The ratio estimate and Padé find no
  crossing. arg₁₂ locks to 0.9872, 1.0368 or 0.9935 whatever ω_t is. G is off by 2–7×10⁻³. Every detector fires.

### 3.4 The completions: twelve coefficients do not fix the zero

Every completion has σ₀…σ₁₂ equal to the real ones, so every estimator returns the real-data numbers at K ≤ 12. Only
the tail differs.

| completion (tail from k = 13) | true ω_R | in 0.99915–1.00011? | zeros in \|x\| < 7.3 / \|x\| < 12 |
|---|---|---|---|
| C-W1: ratio law of D_entire (fit k = 6–11: κ⋆ = 1.2793 + 1.0805i, β = 4.160 − 1.599i) | 0.998266 | no | 6 / 13 |
| C-W2: ratio law, fit k = 8–11 | 0.996215 | no | |
| C-W3: ratio law, fit k = 9–11 | 0.996585 | no | |
| C-EXP: exponential sum of the [6/6] Borel–Padé approximant (*rejected* by the frozen rule: one pole at \|t\| = 2.0–2.26) | 1.000161 | no (above, by 5×10⁻⁵) | 0 / 1 |
| *post-freeze* C-ARC, arc tail fitted with the zero at ω_t = 1.0000 (arc, chord, mirror, β = 2) | 0.999972–1.000000 | yes | |
| *post-freeze* C-ARC, best-fit tail per shape (arc 0.99998, chord 1.00040, mirror 1.00101, β = 2 0.99926) | 0.999973, 1.000382, 1.000971, 0.999247 | 2 of 4 | arc: 2 / 10 |
| *post-freeze* C-ARC, arc tail fitted with the zero at 0.998 / 1.002 | 0.998145 / 1.001852 | no | |

**The mechanism.** For an entire function of order 1, $`\tau_k = -\tfrac1k\sum_j x_j^{-k}`$ for k ≥ 2. Joining the
real twelve-term polynomial to a tail it does not belong to creates a ring of about twelve zeros where the function is
exponentially small. C-W1 has 13 zeros in |x| < 12, on an arc from 11.8 + 1.8i through 2.5 + 8.2i and −6.0 + 0.4i to
−2.2 − 11.4i. Six of them lie closer to the origin than x₀ (|x| < 7.3). The best-fit arc completion has 10 zeros in
|x| < 12, two of them inside |x| < 7.3. Their power sums cancel for every k ≤ 12, which is exactly why these functions
share τ₁…τ₁₂ with the real S₁. The ring shows only at k = 13. On every completion except C-EXP, arg₁₃ leaves the trend of
arg₁₂ by 7×10⁻⁴ to 3×10⁻², in either direction, or has no crossing. (Zeros were counted only for C-W1 and the best-fit
arc completion.)

The functions without a ring behave differently:

- C-EXP continues the τ sequence smoothly (arg₁₃ = 1.000161, its own truth) and has x₀ as its only zero in |x| < 12.
- Every P-ARC and P-ARC-B plant has x₀ as its only zero in |x| < 11.8.

So the estimators carry a hidden premise, that S₁ has no ring of zeros inside |x| ≈ 9, and σ₀…σ₁₂ cannot test it. With
a ring, the same twelve coefficients are compatible with a real zero anywhere in 0.9962–1.0019.

### 3.5 Post-freeze: which planted zero do the real coefficients prefer? (REFIT)

The arc density is refitted to the real σ₀…σ₁₂ with the zero planted at ω_t. The maximum relative residual is V-shaped
around 1.0000:

| ω_t | 0.990 | 0.995 | 0.998 | 0.999 | 0.9995 | 1.0000 | 1.0005 | 1.001 | 1.002 | 1.005 | 1.010 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| residual × 10⁵ | 77 | 40 | 16 | 8.7 | 4.8 | 1.2 | 3.3 | 7.1 | 15 | 39 | 80 |

- The estimators follow the planted ω_t, with the errors of section 3.3. P1 holds for ω_t = 0.998–1.0005.
- On a 10⁻⁴ grid, the L2 residual is smallest at:

| arc shape | best ω_t | L2 residual floor |
|---|---|---|
| D_entire arc | 0.99998 | 5.1×10⁻⁵ |
| β = 2 | 0.99926 | 9.7×10⁻⁵ |
| chord | 1.00040 | 9.0×10⁻⁵ |
| mirror | 1.00101 | 2.5×10⁻⁴ |

  The D_entire arc fits best by a factor of 2, but six density parameters do not make that decisive. The spread of the
  four, 0.9993–1.0010, is a model spread, not a bound.

### 3.6 Noise: how accurate the σ_k must be

Shift of each estimate against the clean plant, at K = 12, × 10⁴ (so an entry of 2×10⁻⁴ is a shift of 2×10⁻⁸),
median / max (P-ARC, ω_t = 1.0000 and 0.9985):

| noise on every σ_k | P1 | arg₁₂ | ratio₁₂ | Padé | G₁₂ |
|---|---|---|---|---|---|
| sigma_err of the real JSON (≤ 10⁻¹² rel.) | 20/20 | 7.5×10⁻⁶ / 2.2×10⁻⁵ | 1.1×10⁻⁴ / 3.6×10⁻⁴ | 2×10⁻⁴ / 4.9×10⁻⁴ | 5.7×10⁻⁵ / 1.3×10⁻⁴ |
| 10⁻¹² relative | 10/10 | 4.5×10⁻⁵ / 5×10⁻⁴ | 0.002 / 0.009 | 0.004 / 0.021 | 0.0015 / 0.0025 |
| 10⁻⁹ relative | 10/10 | 0.044 / 0.5 | 1.8 / 8.8 | 2.9 / 9.2 (2 lost) | 1.6 / 5.2 |
| 10⁻⁶ relative | 0/10 | 11 / 93 | 27 / 110 | 17 / 34 (6 lost) | 18 / 65 |
| 10⁻⁴ relative | 0/10 | 80 / 120 | lost | lost | 38 / 82 |

- **The ln S estimators amplify coefficient errors.** A 10⁻⁹ change of σ₆ alone changes τ₁₂ by 3×10⁻⁴, relative. The
  log turns δS into δS/S, and S is exponentially small, of order |e^{κ̄x}|, on part of the circle |x| ≈ 7.4.
- **G amplifies them too, in two ways.**
  - The truncated $`G_K = \sum_{k\le K}g_kx^k`$ carries each error δσ_j x^j multiplied by a partial sum of
    $`e^{-\bar\kappa x}`$. That partial sum is of size about 500 at |κ̄x₀| ≈ 8.5, not e^{−6.6}.
  - Its κ̄ optimiser hops between two minima (1.013 + 0.486i and 0.889 + 0.554i) under 10⁻⁹ noise.
- **What this demands of the inputs.** The claimed 10⁻⁴ needs every σ_k to about 10⁻¹⁰ relative. The fits' own
  consistency (J = 750 against 1000, sigma_err ≤ 10⁻¹² relative) meets that. The only independent check, the rigorous
  enclosure of σ₁ (D_entire §6.5, ±3.5×10⁻⁵), is five orders short of it.
- **This also explains the oscillation of the arc plants.** Their 1–9×10⁻⁵ residuals are a smooth perturbation of the
  real coefficients, amplified the same way.

### 3.7 Corruptions

Shift against the clean twin at K = 12, × 10⁴. The twin of C-W1 is the real data, since K ≤ 12. The flip and scale
corruptions are at the wf = 1.00 node.

| corruption | P-ARC twin: largest shift; new detectors | real-data twin: largest shift; new detectors |
|---|---|---|
| σ₁₂ sign flip | 99; D-padé, D-G | 98; D-arg, D-bracket, D-padé, D-G |
| σ₉ / σ₅ sign flip | 113 / 124; D-padé, D-G (Padé lost for σ₉) | 108 / 119; same four |
| σ₁₂ × (1 + 10⁻³) | 57 (G), arg −6.2; D-padé, D-G | 64 (G), arg −6.0; D-bracket, D-padé, D-G |
| σ₉, σ₅, σ₂ × (1 + 10⁻³) | 113–114; D-padé, D-G (Padé lost for σ₉ and σ₅, G for σ₅) | 108; four detectors |
| σ₁₂ × (1 + 10⁻³) at all nodes | 58 (arg); G lost; D-padé, D-G | 60 (arg); G lost; D-bracket, D-padé, D-G |
| wf labels all + 0.0005 | **every estimator +5.00; no detector** | **every estimator +5.00; no detector** |
| 1.00 node relabelled 1.0005 | +4.6 to +5.0; no detector | +5.0; no detector |
| 1.01 node relabelled 1.0105 | ≤ 0.3; no detector | ≤ 0.2; no detector |

- **Every coefficient corruption is noticed.** Its largest shift is 5.7×10⁻³ to 1.2×10⁻², and D-padé and D-G fire; on
  the real-data twin D-arg and D-bracket fire as well. A 10⁻³ error in one coefficient is never silent.
- **A shifted ω grid is invisible to every estimator and every detector.** It moves the answer by exactly the shift.
- **Only `entire_cert.py` notices the grid shift**, and only by crashing (FileNotFoundError), because its WFS =
  (0.99, 1.0, 1.01) and its S[1.0] are hard-coded. The harness also ran it with the labels honoured, and then it too
  moves by +5×10⁻⁴.
- A relabelled far node is harmless, because the crossing sits at the middle node.

### 3.8 Controls that fired on real data, and controls seen to fail

Two frozen detectors fire on the real data (N0). Per the protocol, each is diagnosed, not overridden.

- **D-ratio is right.** ratio₁₂ = 0.999152 < ratio₁₁ = 0.999186: the "from below" sequence turned down at k = 12. The
  README's table shows the same two numbers (0.99919, 0.99915). Its text says the ratio estimates "come up from below
  (0.9978, 0.9992, 0.9992 at k = 10–12)", which rounds the turn away. On every ring-free plant the ratio estimate sits
  1.0–5.4×10⁻³ below the truth (C-EXP included), so its stall near 0.9992 is not a sign of convergence.
- **D-ratiolaw is a design error.** Its only violation is at k = 6 (|Δ²| = 0.072–0.087, before the ratio law is
  asymptotic); from k = 7 on, |Δ²| ≤ 0.048. It fires on every plant and on the real data, so as frozen it is noise. It
  was not retuned and is not used in the conclusions.

On the clean arc plants, D-arg, D-ratio and D-bracket also fire: the plants' sequences oscillate, and G lies above arg.
On the P-ARC corruption set, therefore, only D-padé and D-G can discriminate, and they do.

Controls seen to fail:

- the coverage check graded against ω_t + 0.002;
- N0 on `scan23phys/`;
- the detectors on the σ₁₂ flip.

## 4. Verdict

**Is 0.99915–1.00011 an honest bracket for this class? No. It is not a coverage guarantee, for two separate reasons.**

1. **Among functions that are what D_entire describes, it covers 21 of 28 and fails on a whole family.** These are
   Laplace integrals over a Borel arc with x₀ as the only zero in |x| < 11.8, with the real coefficients' ratio law,
   type and term sizes, matching σ₀…σ₁₂ to 1–9×10⁻⁵.
   - The bracket fails on the mirrored arc, 7/7, by 0.6–0.8×10⁻⁴: the arg crossing is not always "from above". The
     ring-free exact completion C-EXP (rejected by the frozen type rule, |t| = 2.2) is also missed, by 0.5×10⁻⁴.
   - The bracket's width comes entirely from the ratio estimate, which is 1.3–5.4×10⁻³ low on every such plant; the
     informative edge is arg₁₂.
   - Calibrated on these plants (arg₁₂ errors −0.8 to +1.9×10⁻⁴), the real arg₁₂ = 1.000109 would put the zero at
     about **0.99992–1.00019**. That is an estimator range from the four arc shapes and C-EXP, not a bound.
2. **The twelve coefficients do not determine the zero to 10⁻⁴ at all.** Functions with exactly the real σ₀…σ₁₂ have
   their real zero anywhere in 0.9962–1.0019, and the bracket covers only some of them (section 3.4):
   - with D_entire's own ratio-law tail, 0.99827;
   - with best-fit arc tails of four shapes, 0.99925–1.00097.

   The ones far away hide a ring of zeros inside |x| ≈ 9 that is invisible to τ₁…τ₁₂. Excluding that ring is an
   assumption about S₁, not something the estimators check. It needs the tail: a converged σ₁₃ good to about 10⁻¹⁰
   (section 3.6), or the representation of S₁ that D_entire §4 asks for.

**Is the best estimate biased, and in which direction?** The best estimate (0.9999–1.0001, Padé and G) fails the claimed
±1.2×10⁻⁴: Padé passes in 7/31 plants, G in 0/31, against the frozen 90%.

- **On the arc plants the two are biased in opposite directions**, and the sign depends on the arc:
  - G is high: +2.6 to +6.4×10⁻⁴ (biased high by the frozen rule).
  - Padé is low: −3.8 to −8.5×10⁻⁴, except +1.1×10⁻⁴ on the mirrored arc.
  - They never agree to better than 2×10⁻⁴.
- **Their agreement to 10⁻⁵ on the real data is therefore not evidence of accuracy.** It is reproduced only by plants
  built to copy the real coefficients exactly. On those, both sit 0.8–0.9×10⁻⁴ below the truth for the best-fit arc
  tail and 2.7–2.8×10⁻⁴ below it for C-EXP. For the other best-fit tails they err by up to 1.1×10⁻³, either way.
- **The arg crossing is the most reliable estimator.** Its error is −0.8 to +1.9×10⁻⁴ on every ring-free plant, and it
  moves only 5×10⁻⁵ under 10⁻⁹ noise. It errs high on three of the four arcs.
- **Aitken (high, +1 to +6×10⁻⁴) adds nothing.**
- **Converging σ₁₃ is the most useful next step.** At K = 13 the factored series G was within 0.75×10⁻⁴ on all 28 arc
  plants, provided σ₁₃ is converged to ~10⁻¹⁰.

**What would make the README claim honest** (proposed text, for the coordinator): "At twelve orders the arg crossing
gives 1.00011 and the Padé and factored estimates 0.9999. On planted functions of the same class
(notes/PLANTED_SELECTION.md) the arg crossing errs by −0.8 to +1.9×10⁻⁴ and the others by up to 8×10⁻⁴, so
ω*/ω∞ ≈ 0.9999–1.0002 (Δ∞ = 1.6438 ± 0.0002, an estimator range), provided S₁ has no zeros besides x₀ within
|x| ≈ 9. The twelve coefficients do not exclude such zeros; with them the zero can lie anywhere in 0.996–1.002."

The ± in the README's Δ∞ = 1.6439 ± 0.0002 is the spread of the best estimates, not a tested interval. Under the
ring-free assumption the calibrated range (0.99992–1.00019, i.e. Δ∞ = 1.64358–1.64402) gives a similar number, but for
a different reason, and with its centre 10⁻⁴ lower in Δ.

## 5. Follow-up: the converged σ₁₃ (post-freeze)

Everything in this section is post-freeze. Input: `solver/inner_amp/scan27J1000p960/`, σ₀…σ₁₃ at wf = 0.99, 1.00, 1.01,
computed at 960 bits. Its err13 is about 10⁻²² (1.5–1.9×10⁻¹³ relative), so it meets the 10⁻¹⁰ requirement of section
3.6. Its σ₀…σ₁₂ agree with `scan25phys` to 4×10⁻¹³. Script: `solver/inner_amp/planted_k13.py` (stages `real`, `calib`,
`predict`, `recomplete`); outputs in `planted_results.jsonl` ("N0 scan27J1000p960") and `planted_checks.json`
("K13 predict", "K13 recomplete").

### 5.1 The real estimates at K = 13

The harness gives:

| estimator | K = 13 | K = 12 |
|---|---|---|
| arg crossing | 1.000046 | 1.000109 |
| ratio | 0.999284 | 0.999152 |
| central Padé, [3/9]–[8/4] | 0.99980–0.99990 (mid 0.999847) | |
| G | 0.999995 | 0.999883 |
| Aitken, k = 11–13 | 0.999886 | |

The arg crossings keep falling monotonically: 1.000299, 1.000197, 1.000109, 1.000046 for k = 10–13. The frozen
detectors that fire are the same two as at K = 12, D-ratio and D-ratiolaw (section 3.8). D-arg, D-bracket, D-padé and
D-G stay silent.

### 5.2 Calibration at K = 13

Each ring-free plant's own error at K = 13 (estimate − ω_t), turned into the zero position the real K = 13 estimate
implies. Errors are × 10⁴.

| plants | arg error → zero | G error → zero |
|---|---|---|
| P-ARC (7) | +0.86 … +1.01 → 0.999945–0.999960 | +0.35 … +0.40 → 0.999955–0.999960 |
| chord (7) | +0.82 … +0.96 → 0.999950–0.999964 | +0.31 … +0.37 → 0.999959–0.999964 |
| mirror (7) | +0.53 … +0.69 → 0.999977–0.999992 | +0.36 … +0.46 → 0.999949–0.999959 |
| β = 2 (7) | +1.21 … +1.30 → 0.999916–0.999925 | +0.74 … +0.75 → 0.999920–0.999922 |
| **all 28 arc plants** | **+0.53 … +1.30 → 0.999916–0.999992** | **+0.31 … +0.75 → 0.999920–0.999964** |
| C-EXP (K = 12 completion; its σ₁₃ is its own) | −0.01 → 1.000047 | +1.64 → 0.999831 |
| REFIT plants with fit residual ≤ 10⁻⁴ (5) | +0.42 … +1.27 → 0.999919–1.000004 | +0.26 … +0.48 → 0.999947–0.999969 |

- **The coordinator's numbers are confirmed.** For arg and G over the 28 arc plants, mirror included: arg +5.3×10⁻⁵ to
  +1.30×10⁻⁴, so 0.999916–0.999992; G +3.1×10⁻⁵ to +7.5×10⁻⁵, so 0.999920–0.999964.
- **C-EXP widens the single-estimator ranges** to 0.999916–1.000047 for arg and 0.999831–0.999964 for G. Its two
  implied zeros disagree with each other by 2.2×10⁻⁴, so its error pattern does not fit the real data.
- **REFIT plants that fit the real coefficients poorly behave worse.** Those with residual 1.6–8×10⁻⁴ have arg errors
  down to −2.1×10⁻⁴ (ω_t = 1.005) and −1.2×10⁻³ (ω_t = 0.990). The calibration holds only for plants that match the
  real coefficients to about 10⁻⁴.
- **The other estimators do not calibrate.** Their errors depend on the family so strongly that the implied zeros do not
  overlap:
  - ratio: from 0.99744–0.99795 (mirror) to 0.99987–1.00000 (P-ARC);
  - Padé: from 0.99986 (mirror) to 1.00034–1.00068 (P-ARC);
  - Aitken: 0.99976–0.99991.

### 5.3 Does the real σ₁₃ rule out the ring?

Every completion of section 3.4 predicted σ₁₃ before it was known. The table compares those predictions with the
converged value, by relative error at the three nodes. "Rejected by" lists the frozen detectors that fire at K = 13 on
the completion but not on the real data.

| completion (K = 12) | σ₁₃ error | τ₁₃ error | its arg₁₃ | rejected by |
|---|---|---|---|---|
| ring-free simple-zero law from τ₁₁, τ₁₂ | — | 7–8×10⁻⁴ | — | — |
| C-W1–3, ratio-law tails (rings) | 1.1–2.1×10⁻³ | 1.9–3.2 (×2–3) | 1.0265 or none | D-arg, D-bracket, D-padé, D-G |
| C-ARC, arc tails fitted at ω_t = 1, four shapes (rings) | 0.7–5×10⁻⁴ | 0.10–0.79 | 1.0008–1.0312 | D-arg, D-bracket, D-padé, D-G |
| C-ARC, best-fit tails, four shapes (rings) | 1.5–5×10⁻⁴ | 0.25–0.80 | 0.9936–1.0295 | D-bracket, D-padé, D-G, and D-arg for three shapes |
| C-ARC, arc tails fitted at 0.998 / 1.002 (rings) | 0.9–1.2×10⁻³ | 1.4–2.0 | none | D-arg, D-bracket, D-padé, D-G |
| C-EXP, Borel–Padé [6/6] (ring-free) | 0.6–1.5×10⁻⁶ | 1.1–2.2×10⁻³ | 1.000161 | D-arg, D-bracket, D-padé |

**No section-3.4 completion survives.** Every ring that a K = 12 junction produces predicted a τ₁₃ off by 10% to 300%,
and the frozen detectors reject it. C-EXP, the ring-free one, came close: σ₁₃ to 10⁻⁶. But its arg₁₃ rises instead of
falling, so D-arg rejects it too. The best prediction of τ₁₃ came from the plain simple-zero law (7–8×10⁻⁴). That is
about what zeros at |x| ≈ 12 would cause, (7.4/12.1)¹³ ≈ 2×10⁻³, so the data look like the ring-free arc plants, whose
next zero is at 12.1 + 0.9i.

There are therefore no survivors to place the zero. The same recipes glued at σ₁₃ (tail from k = 14) give the
following, with zeros counted at wf = 1.00:

| re-glued completion (K = 13) | zero at | zeros in \|x\| < 7.3 / \|x\| < 12 |
|---|---|---|
| ratio-law tails, three windows | 0.997613–0.997934 | 7 / 13 |
| best-fit arc tails, four shapes (L2 floors 1.0–4.3×10⁻⁴) | 0.999754 (β = 2), 1.000180 (arc), 1.000380 (chord), 1.000533 (mirror) | 3–5 / 11–12 |
| Borel–Padé [6/7] from σ₀…σ₁₃ (all poles \|t\| ≤ 1.64, so it passes the frozen type rule; ring-free) | **0.999982** | 0 / 1 |

The re-glued ratio-law and arc completions again carry rings. These have 11–13 zeros, are invisible to τ₁…τ₁₃, and
will show at k = 14. So σ₁₃ has eliminated every ring that was on the table, but a ring of fourteen or more zeros cannot
be excluded by thirteen coefficients. Each new coefficient tests the previous generation of rings, as σ₁₃ just did.

The only ring-free exact completion is the [6/7] exponential sum. Against its zero:

| estimator, K = 13 | error |
|---|---|
| arg₁₃ | +6.4×10⁻⁵ |
| G₁₃ | +1.3×10⁻⁵ |
| Padé mid | −1.35×10⁻⁴ |

### 5.4 Verdict at K = 13

**The ring-free assumption is now supported by data, but not proved.** It made a falsifiable prediction, and the
converged σ₁₃ bore it out:

- the arg crossings kept falling, to 1.000046;
- τ₁₃ matched the simple-zero law to 8×10⁻⁴;
- no new frozen detector fired.

Every ring hypothesis on the table instead predicted a τ₁₃ off by 10%–300% and is rejected. A ring of fourteen or more
zeros remains logically possible; σ₁₄ is its test. Under the ring-free assumption, two numbers follow:

- **From the calibration.** arg₁₃ and G₁₃, calibrated on the 28 arc plants, put the zero at 0.999916–0.999992 and
  0.999920–0.999964. These overlap at **0.99992–0.99996**.
- **From the ring-free exact completion.** The Borel–Padé [6/7] function, the only one that passes the frozen type rule,
  has its zero at 0.999982.

Together these give **ω*/ω∞ = 0.99992–0.99998**, i.e. **Δ∞ = 1.64393–1.64403** (1.64398 ± 0.00005). That is an
*estimator range*: the spread of plant-calibrated estimates over five ring-free function shapes, plus one ring-free
completion. It is not a confidence interval and not a bound. Widened to the single-estimator envelope including C-EXP's
calibration, it is 0.99983–1.00005, i.e. Δ∞ = 1.6438–1.6442. The finite-D value Δ∞ = 1.64389375 lies 4×10⁻⁵ below the
narrow range and inside the wide one. Five plant shapes are too few to read that gap as evidence against Conjecture 1.

Proposed README text, for the coordinator:

> "With σ₁₃ converged (960 bits, `scan27J1000p960/`), the arg crossings keep falling (1.000109, 1.000046 at k = 12, 13)
> and the factored series gives 0.999995. Calibrated on planted functions of the same class (Laplace integrals over a
> Borel arc; notes/PLANTED_SELECTION.md §5), these put the real zero at ω*/ω∞ = 0.99992–0.99996. The one ring-free
> function that reproduces σ₀…σ₁₃ exactly, a [6/7] Borel–Padé sum, puts it at 0.99998. So the D = ∞ inner equations alone
> give Δ∞ = 1.6440 ± 0.0001 (estimator range, 1.64393–1.64403), against 1.64389375 from the finite-D solutions. This
> assumes S₁ has no other zeros within |x| ≈ 12. Twelve coefficients could not test that; the thirteenth rejected every
> alternative constructed so far, and σ₁₄ tests the next."

## 6. Follow-up: the converged σ₁₄ (post-freeze)

Everything in this section is post-freeze. Input: `solver/inner_amp/scan29J1000/`, σ₀…σ₁₄ at wf = 0.99, 1.00, 1.01
(K 29, J 1000, 960 bits; the `_p960` files), and a 1280-bit pilot at wf 1.00. Script:
`solver/inner_amp/planted_k14.py`; records in `planted_results_k14.jsonl`, checks in `planted_checks_k14.json`.

### 6.0 Procedure, fixed before any K = 14 estimate was computed

This subsection was written and committed before any estimator ran at K = 14, on any input. Already known at that
point: the σ₁₄ values and the README's order-14 projection. The procedure is §5's, one order up; the pass criteria are:

1. **σ₁₄ is converged** if, at every node: (a) err14 ≤ 10⁻¹⁰ |σ₁₄| (the requirement of §3.6); (b) σ₀…σ₁₃ are bitwise
   the `scan27J1000p960` values; (c) the 1280-bit pilot's σ₀…σ₁₄ equal the 960-bit ones in every printed digit (except
   σ₀'s imaginary round-off, below 10⁻²⁸⁰); (d) σ₁₄ continues the ratio law: the second difference of σ_k/σ_{k+1} at
   k = 12 is at most 0.05 (the D-ratiolaw threshold, which holds from k = 7 on). Each check gets a mutation that
   must make it fire.
2. **The pipeline on plants first.** N1 (seven lone zeros) must be recovered by arg, ratio, Padé and G to 10⁻⁶ at
   K = 14, and N2 must give no number. The 28 arc plants, C-EXP and the 13 REFIT plants are regenerated and run at
   K = 12, 13, 14; their K = 12 and 13 estimates must equal the stored records (`planted_results.jsonl`) to 10⁻⁹, or
   the plants are not the same functions and the work stops. Control that must fail: N1 graded against ω_t + 0.002.
3. **Real data at K = 14** (record "N0 scan29J1000"): its K = 12 and 13 estimates must equal those of
   "N0 scan27J1000p960" (the inputs are bitwise equal). Mutation control: σ₁₄ → −σ₁₄ at wf 1.00 must fire a detector.
4. **Ring test.** Each K = 13 re-glued completion of §5.3 (C13-W1–3; C13-ARC, four shapes at their best-fit ω_t;
   C13-EXP, the [6/7] sum) is regenerated, and its zero must reproduce the stored "K13 recomplete" truth to 10⁻⁹. Its
   σ₁₄ and τ₁₄ are compared with the measured ones, and it is run through the harness at K = 14. As in §5.3, it
   survives if no detector fires on it at K = 14 that does not fire on the real data. Benchmark: the simple-zero law
   from τ₁₂, τ₁₃.
5. **Calibration** as §5.2: the K = 14 error of each estimator on the ring-free plants (28 arc plants, C-EXP, REFIT
   with fit residual ≤ 10⁻⁴), and the zero each implies for the real K = 14 estimate.
6. **Next generation and ring-free completions.** The §5.3 recipes glued at σ₁₄ (tail from k = 15): ratio-law windows
   k = 6–13, 8–13, 9–13; best-fit arc tails, four shapes, ω_t grid 0.9990–1.0010; the Borel–Padé sum [6/8] (the
   recipe's [6/(K − 6)]). The other [m/(14 − m)], m = 3…7, are reported separately. The frozen type rule: every pole
   has |t| < 2 at all three nodes. Zeros are counted at wf = 1.00 in |x| < 7.3 and |x| < 12.
7. **Combination** as §5.4: the overlap of the calibrated arg and G ranges over the 28 arc plants, together with the
   zero of each ring-free completion that passes the type rule. This is an estimator range, not a bound.
8. **The order-14 projection.** The README's model, est_k = ω* + c r^k through k = 12 and 13, for ω* = 1 and the K = 13
   band, is compared with the real arg₁₄. The hypothesis whose prediction lies nearer is favoured. The comparison is
   called decisive only if the other prediction is farther from arg₁₄ than three times the model's own error, judged
   by the same model through k − 2, k − 1 predicting k on the real data (largest error over k = 11–13, for the
   hypothesis in question) and through k = 12, 13 predicting k = 14 on the 28 arc plants, given their true ω_t.

The stages are `converge`, `plants`, `real`, `ring`, `calib`, `recomplete`, `project` (section 6.0) and `posthoc`
(section 6.8, added after those runs), each run as `nice -n 10 python3 planted_k14.py STAGE` from `solver/inner_amp/`.
Every number below is printed by one of them and stored in `planted_checks_k14.json`.

### 6.1 σ₁₄ is converged

The local machine's values are the files' values, digit for digit:

| wf | σ₁₄ | err14 (relative) | σ₀…σ₁₃ against `scan27J1000p960` | ratio law: \|Δ²\| at k = 12 |
|---|---|---|---|---|
| 0.99 | +5.1349926985882716566×10⁻¹² − 6.1125424732211931245×10⁻¹¹i | 1.64×10⁻²³ (2.7×10⁻¹³) | bitwise equal, 14/14 | 0.0115 |
| 1.00 | +1.8162676402228000068×10⁻¹² − 6.0171529747632837488×10⁻¹¹i | 1.81×10⁻²³ (3.0×10⁻¹³) | bitwise equal, 14/14 | 0.0116 |
| 1.01 | −1.3887266863423045505×10⁻¹² − 5.9052712130189368330×10⁻¹¹i | 2.00×10⁻²³ (3.4×10⁻¹³) | bitwise equal, 14/14 | 0.0118 |

- **All four criteria of section 6.0 pass.** err14 is 3×10⁻¹³ relative, against the 10⁻¹⁰ that section 3.6 asks for.
  σ₀…σ₁₃ and their sigma_err are bitwise those of `scan27J1000p960`. The 1280-bit pilot equals the 960-bit run at wf
  1.00 in every printed digit, σ₁…σ₁₄. σ₁₄ continues the ratio law: the second differences of σ_k/σ_{k+1} for
  k = 7–12 are 0.002–0.048 and fall to 0.012 at k = 12. Each criterion has a mutation (`--mutate=err|low|pilot|ratiolaw`)
  that makes it fire.
- **What err14 is.** It is the late-order fit's plateau step, |S(Mf) − S(Mf − 1)| at fixed J = 1000: an estimate of the
  fit-truncation error, not a bound. No second J was run for σ₁₄. The pilot tests arithmetic precision only. The
  multiplier fits S₁^(k), k = 3–29, choose the same Mf with the same plateau step at 960 and 1280 bits. The pilot's own
  sigma_err is inf at every k, an artifact: its k = 1 fit has residual exactly 0 at every Mf, so `pick_plateau` finds no
  falling residual. "err14 under 2×10⁻²³" therefore refers to the 960-bit runs.
- **Spread in wf.** σ₁₄ changes by 5.7% of itself between the nodes, as σ₁₂ and σ₁₃ do (5.4%, 5.5%). Its second
  difference over its first is 0.059 (0.056, 0.057 for k = 12, 13). err14 is 6×10⁻¹² of that change.

### 6.2 The pipeline on plants first

These ran before the real data, as section 6.0 requires. 51 plants: N1 (7), N2 (2), the 28 arc plants, C-EXP and the 13
REFIT plants, at K = 12, 13, 14.

- **The plants are the same functions.** Their K = 12 and 13 estimates equal the stored records to 5.5×10⁻¹³ at worst.
- **N1 passes, 7/7.** A lone simple zero is recovered at K = 14 by arg, ratio, Padé and G to 3.5×10⁻⁷ at worst.
  Graded against ω_t + 0.002 it fails, as it must.
- **N2 passes, 2/2.** No estimator prints a number.

### 6.3 The real estimates at K = 14

The record "N0 scan29J1000" equals "N0 scan27J1000p960" at K = 12 and 13 to the last bit. At K = 14:

| estimator | K = 14 | K = 13 | K = 12 |
|---|---|---|---|
| arg crossing | 1.000026 | 1.000046 | 1.000109 |
| ratio | 0.999769 | 0.999284 | 0.999152 |
| central Padé | [3/10]–[9/4]: 0.99995–1.00013 (mid 1.000043) | [3/9]–[8/4]: 0.99980–0.99990 (mid 0.999847) | 0.99987–0.99991 |
| G (production κ̄) | 1.000027 | 0.999995 | 0.999883 |
| Aitken, k − 2 … k | 1.000017 | 0.999886 | 0.999559 |
| effective exponent c at ω∞ | 1.000 − 0.004i | 1.004 − 0.010i | 1.013 − 0.012i |

- **The arg crossings keep falling monotonically:** 1.000299, 1.000197, 1.000109, 1.000046, 1.000026 for k = 10–14.
  Their distance from 1 shrank by ×0.42, then ×0.57.
- **The simple-zero law predicts τ₁₄ from τ₁₂, τ₁₃ to 4.9–6.3×10⁻⁴.** It predicted τ₁₃ to 7.2–8.0×10⁻⁴.
- **The effective exponent reaches 1.** c = −kτ_k x₀^k with x₀ = (k − 1)τ_{k−1}/(kτ_k) reproduces the README's column
  exactly. Despite the README's "three consecutive orders", it uses two.
- **The numerical uncertainty is negligible.** Under noise of size sigma_err on every σ_k (20 seeds), arg₁₄ moves by
  at most 5×10⁻¹⁰.
- **Detectors.** D-ratio and D-ratiolaw fire as at K = 12 and 13 (section 3.8). D-arg, D-padé and D-G stay silent.
- **D-bracket fires on the real data for the first time.** Per the protocol it was diagnosed before going on:
  - The cause: G₁₄ lies 1.2×10⁻⁶ above arg₁₄, and the Padé mid 1.7×10⁻⁵ above it. At K = 12 they were 2.2×10⁻⁴ below.
  - It is not a sign of a ring. On the ring-free plants at K = 14 it fires 15 times out of 29 (every mirror and β = 2
    plant, and C-EXP), always for this reason.
  - The detector assumes the arg crossing is the top edge of the bracket. It stops discriminating once arg is as close
    to the zero as the other estimators' own errors.
  - It was not changed. Section 6.4 compares against the real data's detectors, so it no longer rejects a completion
    by itself.
- **The mutation controls fire.** σ₁₄ → −σ₁₄ at wf 1.00 fires D-arg, D-padé and D-G. σ₁₄ × (1 + 10⁻³) fires D-padé
  and D-G. Each moves the estimates by about 10⁻².

### 6.4 The ring test: what each K = 13 completion predicted for σ₁₄

Every completion of section 5.3 was regenerated, and its zero equals the stored one to 10⁻⁹. Its K = 13 estimates are
the real ones exactly. "Beyond the real data's" lists the detectors that fire on it at K = 14 but not on the real data
(the section 5.3 rule).

| completion (from σ₀…σ₁₃) | its zero | zeros in \|x\| < 7.3 / 12 | σ₁₄ rel. error | τ₁₄ rel. error | its arg₁₄ | beyond the real data's | survives |
|---|---|---|---|---|---|---|---|
| simple-zero law from τ₁₂, τ₁₃ | — | — | — | 4.9–6.3×10⁻⁴ | — | — | — |
| C13-W1–3, ratio-law tails | 0.99761–0.99793 | 7 / 13 | 1.7–3.2×10⁻³ | 2.3–3.9 | none | D-arg, D-padé, D-G | no |
| C13-ARC arc | 1.000180 | 3 / 11 | 2.7–3.5×10⁻⁴ | 0.35–0.40 | 1.0202 | D-arg, D-padé, D-G | no |
| C13-ARC chord | 1.000380 | 5 / 12 | 2.7–4.1×10⁻⁴ | 0.35–0.48 | 1.0048 | D-arg, D-padé, D-G | no |
| C13-ARC mirror | 1.000533 | 5 / 12 | 5.2–6.9×10⁻⁴ | 0.67–0.80 | 0.9857 | D-padé, D-G | no |
| C13-ARC β = 2 | 0.999754 | 4 / 12 | 2.8–4.2×10⁻⁴ | 0.36–0.49 | 1.0175 | D-arg, D-padé, D-G | no |
| C13-EXP, Borel–Padé [6/7] (ring-free) | 0.999982 | 0 / 1 | 1.3–1.4×10⁻⁷ | 1.6–1.7×10⁻⁴ | 1.000025 | none | **yes** |
| *post-hoc* Borel–Padé [5/8] (ring-free, section 6.6) | 0.999957 | — / 1 | 3.6–6.1×10⁻⁷ | 4.7–7.1×10⁻⁴ | 1.000034 | none | **yes** |

**No ring survives, and the ring-free sums do.**

- Every ring completion predicted a τ₁₄ off by 35% to 390%, and two or three frozen detectors reject it.
- The [6/7] Borel–Padé sum, the one ring-free completion of section 5.3, predicted σ₁₄ to 1.3×10⁻⁷. Its τ₁₄ is closer
  than the simple-zero law's, and its arg₁₄ differs from the real one by 1.1×10⁻⁶.
- The [5/8] sum, which section 5 did not examine (section 6.6), survives too.

So σ₁₄ has done to the second generation of rings what σ₁₃ did to the first.

### 6.5 Calibration at K = 14

As section 5.2, one order up. Each family's error range (estimate − ω_t, × 10⁴), and the zero that range implies for the
real K = 14 estimate:

| plants | arg: error → zero | G: error → zero | Padé mid: error → zero |
|---|---|---|---|
| P-ARC (7) | +0.52 … +0.64 → 0.999962–0.999974 | −0.27 … −0.19 → 1.000047–1.000054 | −2.19 … +0.05 → 1.000038–1.000262 |
| chord (7) | +0.26 … +0.33 → 0.999993–1.000001 | −0.19 … −0.14 → 1.000042–1.000046 | −0.89 … −0.69 → 1.000112–1.000132 |
| mirror (7) | −0.22 … −0.14 → 1.000040–1.000048 | −0.17 … −0.12 → 1.000039–1.000045 | +0.62 … +0.74 → 0.999969–0.999981 |
| β = 2 (7) | +0.70 … +0.74 → 0.999952–0.999956 | −0.44 … −0.35 → 1.000062–1.000071 | +1.25 … +2.54 → 0.999789–0.999918 |
| **all 28 arc plants** | **−0.22 … +0.74 → 0.999952–1.000048** | **−0.44 … −0.12 → 1.000039–1.000071** | −2.19 … +2.54 → 0.999789–1.000262 |
| C-EXP (the K = 12 completion) | −0.22 → 1.000048 | +0.78 → 0.999949 | +2.83 → 0.999760 |
| REFIT, residual ≤ 10⁻⁴ (5) | +0.52 … +0.68 → 0.999958–0.999975 | −0.28 … −0.17 → 1.000044–1.000056 | −0.06 … +5.16 → 0.999527–1.000049 |

- **The same calibration at K = 13 from the new records reproduces section 5.2.** It gives arg 0.999916–0.999992 and
  G 0.999920–0.999964.
- **The calibrated arg and G overlap at 1.000039–1.000048 at K = 14.** At K = 13 they overlapped at 0.99992–0.99996.
  The two ranges are disjoint. Moving one order up shifted the calibrated zero by 8×10⁻⁵, more than either range is
  wide.
- **G's error changed sign on every arc plant.** It was +0.31 … +0.75×10⁻⁴ at K = 13 and is −0.44 … −0.12×10⁻⁴ at
  K = 14. The arg errors shrank by about half, and the mirrored arc's turned negative again, as at K = 12.
- **The ratio and Aitken estimates still do not calibrate.** The ratio estimate is 4–12×10⁻⁴ low on every plant.
  Aitken errs on β = 2 by up to 4.8×10⁻³ either way.

### 6.6 The next generation of rings, and the ring-free completions at K = 14

The section 5.3 recipes glued at σ₁₄ (tail from k = 15), with zeros counted at wf = 1.00. Each function reproduces
σ₀…σ₁₄ exactly (the Borel–Padé sums to 10⁻⁴⁶), so the real estimates at K ≤ 14 are its estimates. The last column is the
real arg₁₄ minus the function's zero.

| completion (from σ₀…σ₁₄) | zero at | zeros in \|x\| < 7.3 / 12 / 16 | type rule (max \|t\|) | arg₁₄ error |
|---|---|---|---|---|
| ratio-law tails, windows k = 6–13, 8–13, 9–13 | 0.998876, 0.999218, 0.999343 | 7 / 14 / — | — | +6.8 … +11.5×10⁻⁴ |
| best-fit arc tails, four shapes (L2 floors 2.2–6.3×10⁻⁴) | 1.000097 (β = 2), 1.000223 (mirror), 1.000313 (arc), 1.000353 (chord) | 4–6 / 12–13 / — | — | −0.7 … −3.3×10⁻⁴ |
| Borel–Padé [3/11] | 1.000056 | 0 / 8 / 12 | passes (1.58) | −0.29×10⁻⁴ |
| Borel–Padé [4/10] | 0.999991 | 0 / 4 / 10 | passes (1.64) | +0.35×10⁻⁴ |
| **Borel–Padé [5/9]** (ring-free) | **0.999983** | 0 / 1 / 8 | passes (1.64) | +0.43×10⁻⁴ |
| **Borel–Padé [6/8]**, the recipe's [6/(K − 6)] (ring-free) | **0.999993** | 0 / 1 / 7 | passes (1.66) | +0.33×10⁻⁴ |
| Borel–Padé [7/7] | 0.999993 | 0 / 1 / 7 | passes (1.66) | +0.33×10⁻⁴ |

- **The ratio-law and arc completions again carry rings,** now of 12–14 zeros. They will show at k = 15.
- **[3/11] and [4/10] pass the type rule but carry rings** of 8 and 4 zeros inside |x| < 12.
- **[6/8] and [7/7] are the same function.** [6/8] is six exponentials on the arc, t = 0.74 + 0.14i … 1.30 + 1.03i,
  plus two spurious poles with residues 2.7×10⁻⁷ (t ≈ 0) and 7.8×10⁻⁹ (t = 0.44 − 0.96i). [7/7] carries the t ≈ 0
  term as its constant.
- **[5/9] has five poles along the same arc and one at 0.749 + 0.005i** (residue 3.5×10⁻³), plus a cluster of three
  poles at |t| < 0.01 (residues 3×10⁻³ each).
- **So the ring-free exact completions at K = 14 that pass the frozen type rule put the zero at 0.999983 and
  0.999993.** Against them the real K = 14 estimates err by: arg +0.33 to +0.43×10⁻⁴, G +0.34 to +0.44×10⁻⁴, Padé mid
  +0.50 to +0.60×10⁻⁴.
- **A discrepancy with section 5.3.** The same scan at K = 13 shows that [6/7] was not the only ring-free sum passing
  the type rule. [5/8] (max |t| 1.61, one zero in |x| < 12) puts the zero at 0.999957. [4/9] (1.000002) and [3/10]
  (1.000154) pass the type rule but have 3 and 7 zeros in |x| < 12. Section 5.3 examined only the recipe's
  [6/(K − 6)].

### 6.7 The README's order-14 projection

The model est_k = ω* + c r^k, through the arg crossings at k = 12 and 13, predicted these values of arg₁₄ − 1:

| zero at | predicted arg₁₄ − 1 | measured − predicted | same model's misses at k = 11, 12, 13 (real data) |
|---|---|---|---|
| 1 | +1.92×10⁻⁵ | +0.68×10⁻⁵ | +4.0, −2.1, −1.4×10⁻⁵ |
| 0.99998 | +1.36×10⁻⁵ | +1.25×10⁻⁵ | +4.4, −1.9, −1.1×10⁻⁵ |
| 0.99995 | +0.77×10⁻⁵ | +1.83×10⁻⁵ | +5.0, −1.6, −0.6×10⁻⁵ |
| 0.99992 | +0.38×10⁻⁵ | +2.23×10⁻⁵ | +5.6, −1.3, −0.3×10⁻⁵ |

**The measured arg₁₄ − 1 is +2.61×10⁻⁵.** It lies above every prediction, nearest the one for a zero at 1.

- **It favours a zero at 1 over the K = 13 band, but not decisively.** The disfavoured predictions lie 1.25×10⁻⁵ away or
  more. Section 6.0's criterion asks for three times the model's own error. That error is at least 1.4×10⁻⁵ (the
  model's k = 13 miss for a zero at 1) and at most 8.3×10⁻⁵ (its largest miss on the 28 arc plants, given their true
  ω_t). Either way the criterion fails.
- **The misses (measured − predicted) on the real data alternate in sign and shrink:** +4.0, −2.1, −1.4, +0.7×10⁻⁵ at
  k = 11–14 for a zero at 1. The 0.7×10⁻⁵ at k = 14 is within that pattern.
- **The long-baseline trend agrees.** The zoom figure's dashed line, a straight-line fit to log(arg_k − 1) over
  k = 4–13 (which assumes the zero at 1), extrapolates to 2.57×10⁻⁵ at k = 14, 0.4×10⁻⁶ from the measured value. That
  model too is uncalibrated.
- **The three-point geometric limit adds nothing.** Aitken on k = 12–14 gives 1.000017. Over k = 9–14 its values
  scatter over 0.99956–1.00014, and on the arc plants it errs by up to 4.8×10⁻³ either way.

### 6.8 Post-hoc: why the calibrated range moved

Added after the runs above (stage `posthoc`), to find out why the arc-plant calibration at K = 14 is disjoint from the one
at K = 13.

- **The arc plants do not follow the real coefficients from one order to the next. The Borel–Padé functions do.**
  - From K = 13 to 14 the real arg crossing moved −1.97×10⁻⁵. On the arc plants it moved −3.4 to −9.1×10⁻⁵, on C-EXP
    and C13-EXP −2.08×10⁻⁵.
  - The real G moved +3.2×10⁻⁵, on the arc plants −4.6 to −11.8×10⁻⁵.
  - The arc plants miss the measured σ₁₃ by at least 1.5×10⁻⁴ and σ₁₄ by at least 4.8×10⁻⁴ (relative, worst node). C-EXP
    misses them by 1.5×10⁻⁶ and 1.5×10⁻⁵.
  - A calibration transfers only between functions that behave alike at the order in question. At K = 14 the arc plants
    do not, for G at all.
- **G is fragile in a way the calibration hides.** Its objective, the last two terms of G_K at X0, has four to six local
  minima of similar depth. The production optimiser takes the one nearest its fixed start, 0.93 + 0.5i:
  - On the real data that is the global minimum at K = 12, the fourth-best at K = 13 and the third-best at K = 14.
  - Over the minima within a factor 5 of the best objective, the real G spans 0.99947–1.00020 at K = 12,
    0.99988–1.00029 at K = 13 and 0.99976–1.00003 at K = 14.
  - On the arc plants (ω_t = 1) the same spread at K = 14 is 0.99996–1.00016.
  - The calibration measures G with the optimiser's start held fixed, so it does not include this spread.
  - The global minimum separates the two families again. On the real data it gives 0.999883, 0.999878, 0.999886 at
    K = 12, 13, 14. On the four arc plants at K = 14 it is within +0.4 to +1.2×10⁻⁵ of the truth. On the ring-free
    completions, whose G at any κ̄ is the real one, it errs by −1.0 to −1.1×10⁻⁴.
- **G and Padé respond strongly to the last coefficient.** C13-EXP's σ₁₄ differs from the real one by 1.3×10⁻⁷, and
  σ₀…σ₁₃ are identical. Yet its K = 14 Padé mid differs from the real one by 3.1×10⁻⁴ and its G₁₄ by 3.2×10⁻⁵ (its
  optimiser also lands in another minimum). The arg crossing differs by 1.1×10⁻⁶. This is section 3.6's amplification,
  at the last order.

### 6.9 Verdict at K = 14

**The ring-free assumption passed its second test.**

- σ₁₄ is converged.
- Every ring on the table predicted a τ₁₄ off by 35%–390% and is rejected.
- The ring-free Borel–Padé sums of σ₀…σ₁₃ predicted σ₁₄ to 1.3×10⁻⁷ and 6×10⁻⁷.
- The arg crossings kept falling.
- The effective exponent reached 1.000.

A ring of fifteen or more zeros remains possible: the re-glued completions of section 6.6 carry rings of 12–14 zeros.
σ₁₅ is their test.

**The K = 13 estimator range did not survive.**

- By the rule of section 6.0 (the calibrated arg ∩ G overlap over the 28 arc plants, together with the ring-free
  completions that pass the type rule), K = 14 gives **ω*/ω∞ = 0.99998–1.00005**. Its two halves disagree:
  - the calibrated arg and G put the zero at 1.000039–1.000048;
  - the ring-free completions put it at 0.999983–0.999993.
- The K = 13 result, 0.99992–0.99998, lies entirely below.
- The cause is section 6.8:
  - the arc plants stopped resembling the real function at k = 13–14;
  - G depends on which minimum of its objective is taken.
- Without G, the calibrated arg crossing alone gives 0.99995–1.00005. That range contains every ring-free completion
  (0.99996–0.99999 over K = 13, 14) and contains 1.

**The honest statement is the union over the two orders.** It is ω*/ω∞ = 0.99992–1.00005, i.e. Δ∞ = 1.64381–1.64403
(1.6439 ± 0.0001).

- This is an *estimator range* over two orders, five plant shapes and four ring-free completions. It is not a
  confidence interval and not a bound.
- At K = 14 alone the range is 0.99998–1.00005 (Δ∞ = 1.64381–1.64393).
- The finite-D value, ω*/ω∞ = 1 (Δ∞ = 1.643893746), lies inside both. The 4×10⁻⁵ gap that section 5.4 reported is gone.
- Within the range:
  - the ring-free completions, the most stable element (0.999957 → 0.999983 and 0.999982 → 0.999993 from K = 13 to 14),
    sit 0.7–4×10⁻⁵ below 1;
  - the plant-calibrated K = 14 estimates sit 4–5×10⁻⁵ above;
  - the order-14 projection prefers 1.
- None of these is calibrated well enough to choose among them. Narrowing the range needs σ₁₅, or a calibration family
  that matches σ₀…σ₁₄ (the Borel–Padé functions do; the arc plants do not).

### 6.10 Proposed text, for the coordinator

**README, replacing the "*Thirteen orders.*" bullet:**

> - *Fourteen orders.* σ₁₃ and σ₁₄ converge at 960 bits with the late-order window J = 1000
>   (`solver/inner_amp/scan29J1000/`; σ₁₄ to 3×10⁻¹³ relative, and a 1280-bit run agrees in every printed digit; at
>   J = 750 σ₁₃ did not converge, `scan27phys/`). The arg crossings keep falling (1.000299, 1.000197, 1.000109,
>   1.000046, 1.000026 at k = 10–14), and the effective exponent reaches 1.000. At k = 13 and 14 the factored series
>   gives 0.999995 and 1.000027, and the central Padé approximants give 0.99980–0.99990 and 0.99995–1.00013. Every ring
>   of zeros constructed to fit the known coefficients mispredicted the next coefficient of ln S₁: by 10%–300% at
>   k = 13 and by 35%–390% at k = 14. The ring-free Borel–Padé sums of σ₀…σ₁₃ predicted σ₁₄ to 1–6×10⁻⁷. Their
>   zeros, and those of the ring-free sums of σ₀…σ₁₄, lie at 0.99996–0.99999. Calibrated on planted functions, arg
>   and G put the zero at 0.99992–0.99996 at thirteen orders but at 1.00004–1.00005 at fourteen. That shift is larger
>   than either range, because the planted functions do not follow the real coefficients from one order to the next
>   ([notes/PLANTED_SELECTION.md](notes/PLANTED_SELECTION.md) §6). Taken together, **ω*/ω∞ = 0.99992–1.00005,
>   Δ∞ = 1.6439 ± 0.0001 (1.64381–1.64403) from the D = ∞ inner equations alone**. This is an estimator range over two
>   orders, five plant shapes and four completions, not a bound, and it uses no input from the finite-D solutions.
>   Their value, 1.643893746, lies inside it. The range assumes no zeros besides x₀ within |x| ≈ 12. A ring of fifteen
>   or more zeros is still possible, and σ₁₅ is its test. The rule itself is still a conjecture (hypotheses H1–H2,
>   gaps (a)–(c)). Tightening the number needs more orders, approximants that model the competing singularity
>   explicitly (differential or algebraic approximants), or M = 7, which evaluates the multiplier at real amplitude
>   with no series to resum (done: 1.000022).

**README, the zoom figure's caption:**

> **The same estimates as distance from the finite-D answer** (`solver/plot_selection_zoom.py`, with σ₁₄). On a log
> scale steady convergence is a straight line: from above, the estimates fall by about ×0.53 per order from k = 4 to
> 14. The shaded band is where the K = 13 calibration put the zero, 2–8×10⁻⁵ below 1. The hollow marks at k = 14 are
> what a simple geometric model through k = 12 and 13 (not a calibrated estimator) predicted before σ₁₄ was known:
> 1.9×10⁻⁵ above 1 if the zero is exactly at 1, and 0.4–1.4×10⁻⁵ if it is in the band. The measured crossing is
> 2.6×10⁻⁵ above 1, nearer the first. The dashed trend of k = 4–13 extrapolated to 2.57×10⁻⁵. The same model had missed
> k = 13 by 1.4×10⁻⁵, so this favours a zero at 1 without excluding the band. At fourteen orders the calibrated range
> moved up and now contains 1 ([notes/PLANTED_SELECTION.md](notes/PLANTED_SELECTION.md) §6).

**Smaller README changes the new data imply:**

- *The table of crossings (after "Two families of estimates approach from opposite sides")*:
  - add the rows | 13 | 1.00005 | 0.99928 | 1.004 − 0.010i | and | 14 | 1.00003 | 0.99977 | 1.000 − 0.004i |;
  - name `scan29J1000/` for σ₁–σ₁₄.
- *The paragraph below the table*:
  - "from three consecutive orders" → "from two consecutive orders";
  - append 1.004, 1.000 to the exponent list;
  - the ratio estimates "turn down at the last order (0.9978, 0.99919, 0.99915 at k = 10–12)" now rise again (0.99928,
    0.99977 at k = 13, 14).
- *The main figure's caption*: "the Padé approximants (ranges) and the factored series land inside it" holds through
  k = 13. At k = 14 they reach just above its top edge (G by 1×10⁻⁶).
- *The other places that quote the K = 13 range need the same update*: the summary near the top of the README (line
  84), `PROOF.md` §1, `paper/physics/main.tex` (around "at thirteen orders") and `paper/math/main.tex` (the evidence
  table and the σ₁₃ paragraph). Suggested wording: "at fourteen orders, ω*/ω∞ = 0.99992–1.00005 (Δ∞ = 1.6439 ± 0.0001,
  an estimator range over two orders), which contains the finite-D value; σ₁₄ rejected every ring of zeros constructed
  from thirteen coefficients, and σ₁₅ tests the next."

## 7. Follow-up: σ₁₅ (post-freeze)

Input: `solver/inner_amp/scan31J1000/`, σ₀…σ₁₅ at wf = 0.99, 1.00, 1.01 (K 31, J 1000, 960 bits; T44 on the Mac), and
pilot (b) at wf 1.00 (`scan31J1250/`: K 31, J 1250, 1280 bits; its sigma_err is inf at every k, as for T38's pilot, so it
checks (a)'s value only). Script: `solver/inner_amp/planted_k15.py`; records in `planted_results_k15.jsonl`, checks in
`planted_checks_k15.json`.

### 7.0 Procedure, fixed before any K = 15 estimate was computed

This subsection was written and committed before any estimator ran at K = 15, on any input. Already known at that
point: the σ₁₅ values and errors at the three nodes and pilot (b)'s σ₁₅ at wf 1.00 (comms, 2026-10-25). The procedure is
section 6.0's one order up, with three changes that section 6's findings call for, stated here before the data: the
ring test also covers the Borel–Padé sums of σ₀…σ₁₄ that carry rings ([3/11], [4/10]); the calibration adds a family
that follows the real coefficients (section 6.8); and G enters the combination only through its spread over the local
minima of its objective (section 6.8), not as a calibrated estimator.

1. **σ₁₅ is converged** if, at every node: (a) err15 ≤ 10⁻¹⁰ |σ₁₅|; (b) σ₀…σ₁₄ are bitwise the `scan29J1000` 960-bit
   values; (c) pilot (b)'s σ₁₅ equals (a)'s within (a)'s err15, and its σ₁₂…σ₁₄ equal `scan29J1000`'s within their errors;
   (d) σ₁₅ continues the ratio law: the second difference of σ_k/σ_{k+1} at k = 13 is at most 0.05. Each check gets a
   mutation that must make it fire.
2. **The pipeline on plants first.** N1 must be recovered by arg, ratio, Padé and G to 10⁻⁶ at K = 15 and N2 must give no
   number. The 28 arc plants, C-EXP and the 13 REFIT plants are regenerated and run at K = 13, 14, 15; their K = 13 and 14
   estimates must equal the stored K = 14 records (`planted_results_k14.jsonl`) to 10⁻⁹, or the work stops. Control that
   must fail: N1 graded against ω_t + 0.002.
3. **Real data at K = 15** (record "N0 scan31J1000"): its K = 13 and 14 estimates must equal those of "N0 scan29J1000".
   Mutation controls: σ₁₅ → −σ₁₅ at wf 1.00 must fire a detector; σ₁₅ × (1 + 10⁻³) is reported.
4. **Ring test.** Every K = 14 completion of section 6.6 is regenerated and its zero must reproduce the stored
   "K14 recomplete" value to 10⁻⁹: C14-W1–3 (ratio-law tails), C14-ARC (four shapes at their best-fit ω_t) and the
   Borel–Padé sums C14-BP [m/(14 − m)], m = 3…7. For each: its σ₁₅ and τ₁₅ against the measured ones, and the harness at
   K = 15. It survives if no detector fires on it at K = 15 that does not fire on the real data (the section 5.3 rule).
   Benchmark: the simple-zero law from τ₁₃, τ₁₄. A completion that carries a ring (zeros in |x| < 12 besides x₀) and
   survives is a finding, reported as such.
5. **Calibration** at K = 15, as section 6.5: the error of arg, ratio, Padé mid and G on (i) the 28 arc plants, (ii)
   C-EXP, (iii) REFIT with fit residual ≤ 10⁻⁴, and (iv) new, the BP family: the ring-free Borel–Padé sums that pass the
   type rule, C13-BP [5/8], [6/7] and C14-BP [5/9], [6/8] (their σ₀…σ₁₃ or σ₀…σ₁₄ are the real ones), each with its own
   zero as truth. For each family, the zero each estimator's error range implies for the real K = 15 estimate.
6. **Next generation and ring-free completions,** glued at σ₁₅ (tail from k = 16): ratio-law windows k = 6–14, 8–14,
   9–14; best-fit arc tails, four shapes, ω_t grid 0.9990–1.0010; the Borel–Padé sums [m/(15 − m)], m = 3…7. Frozen type
   rule: every pole has |t| < 2 at all three nodes. Zeros counted at wf = 1.00 in |x| < 7.3, 12 and 16.
7. **Combination.** The K = 15 estimator range is the hull of: the calibrated arg range over the 28 arc plants; the
   calibrated arg range over the BP family; and the zeros of the ring-free completions of item 6 that pass the type
   rule (no zeros besides x₀ in |x| < 12). G is reported with its spread over the minima of its objective within a
   factor 5 of the best (section 6.8) and is not part of the hull. The headline is the union of the K = 14 range
   (0.99998–1.00005) and the K = 15 range if they overlap; if they are disjoint, both are reported and the reason is
   diagnosed before any headline is written. The range is an estimator range, not a bound.
8. **The order-15 projection.** The model est_k = ω* + c r^k through k = 13 and 14, for ω* = 1 and for 0.99998, 0.99995
   (the lower part of the K = 14 range), against the real arg₁₅; decisive only if the other prediction is farther from
   arg₁₅ than three times the model's own error (its largest miss over k = 12–14 on the real data, and over the 28 arc
   plants at k = 15 given their true ω_t), as in section 6.0.

The stages are `converge`, `plants`, `real`, `ring`, `calib`, `recomplete`, `project`, each run as
`nice -n 10 python3 planted_k15.py STAGE` from `solver/inner_amp/`.

### 7.1 σ₁₅ is converged

| wf | σ₁₅ | err15 (relative) | σ₀…σ₁₄ against `scan29J1000` | ratio law: \|Δ²\| at k = 13 |
|---|---|---|---|---|
| 0.99 | +4.3353007071037814558×10⁻¹² − 3.6503299238330505347×10⁻¹²i | 2.60×10⁻²⁴ (4.6×10⁻¹³) | bitwise equal, 15/15 | 0.0108 |
| 1.00 | +4.0489578842497621311×10⁻¹² − 3.8060086244191446298×10⁻¹²i | 2.86×10⁻²⁴ (5.1×10⁻¹³) | bitwise equal, 15/15 | 0.0109 |
| 1.01 | +3.7594108527414830562×10⁻¹² − 3.9427088921505715759×10⁻¹²i | 3.14×10⁻²⁴ (5.8×10⁻¹³) | bitwise equal, 15/15 | 0.0111 |

- **All four criteria of section 7.0 pass,** and each one's mutation (`--mutate=err|low|pilot|ratiolaw`) makes it fire.
  - err15 is 5×10⁻¹³ relative, against the 10⁻¹⁰ asked for.
  - Pilot (b) at wf 1.00 equals (a)'s σ₁₅ to 2.7×10⁻²⁶ (err15 2.9×10⁻²⁴). Its σ₁₂, σ₁₃, σ₁₄ equal `scan29J1000`'s to
    3.4×10⁻²⁴, 8.8×10⁻²⁵ and 1.8×10⁻²⁵, each 100–150 times below their errors.
  - The second differences of σ_k/σ_{k+1} stay at 0.011 for k = 11–13 (0.002–0.048 over k = 7–13).
- **What err15 is:** as err14 (section 6.1), the late-order fit's plateau step, an estimate of the fit-truncation
  error, not a bound. The pilot tests arithmetic precision only; its own sigma_err is again inf at every k.
- **Spread in wf.** σ₁₅ changes by 5.8% of itself between the nodes (5.5% and 5.7% for σ₁₃, σ₁₄). err15 is 10⁻¹¹ of
  that change.

### 7.2 The pipeline on plants first

51 plants at K = 13, 14, 15, before the real data. All four checks pass:
- the regenerated plants' K = 13 and 14 estimates equal the stored K = 14 records exactly (largest difference 0);
- N1, 7/7: a lone simple zero recovered at K = 15 by arg, ratio, Padé and G to 3.5×10⁻⁷ at worst; graded against
  ω_t + 0.002 it fails, as it must;
- N2, 2/2: no estimator prints a number.

### 7.3 The real estimates at K = 15

The record "N0 scan31J1000" equals "N0 scan29J1000" at K = 13 and 14 to the last bit.

| estimator | K = 15 | K = 14 | K = 13 |
|---|---|---|---|
| arg crossing | 1.000019 | 1.000026 | 1.000046 |
| ratio | 0.999920 | 0.999769 | 0.999284 |
| central Padé | [3/11]–[10/4]: 0.999984–1.000063 (mid 1.000024) | mid 1.000043 | mid 0.999847 |
| G (production κ̄) | 0.999999 | 1.000027 | 0.999995 |
| Aitken, k − 2 … k | 1.000015 | 1.000017 | 0.999886 |
| effective exponent c at ω∞ | 1.002 − 0.002i | 1.000 − 0.004i | 1.004 − 0.010i |

- **The arg crossings keep falling monotonically:** 1.000197, 1.000109, 1.000046, 1.000026, 1.000019 for k = 11–15.
  The distance from 1 shrank by ×0.57 from k = 13 to 14 and by ×0.73 from 14 to 15.
- **The simple-zero law predicts τ₁₅ from τ₁₃, τ₁₄ to 1.1–2.6×10⁻⁴,** better than it predicted τ₁₄ (4.9–6.3×10⁻⁴).
- **Detectors:** D-ratio, D-bracket and D-ratiolaw, as at K = 14 (section 6.3 explains D-bracket); D-arg, D-padé and
  D-G stay silent.
- **The mutation controls fire.** σ₁₅ → −σ₁₅ at wf 1.00 fires D-G, D-arg and D-padé; σ₁₅ × (1 + 10⁻³) fires D-G and
  D-padé. Each moves the estimates by about 10⁻².
- **Numerical noise:** under noise of size sigma_err on every σ_k (20 seeds), arg₁₅ moves by at most 1.1×10⁻⁹.

### 7.4 The ring test: what each K = 14 completion predicted for σ₁₅

Every completion was regenerated first and reproduces its stored zero to the printed digits. The benchmark, the
simple-zero law from τ₁₃, τ₁₄, misses τ₁₅ by 1.1–2.6×10⁻⁴.

| completion (K = 14) | its zero | zeros in \|x\| < 7.3 / 12 / 16 | σ₁₅ rel. error | τ₁₅ rel. error | its arg₁₅ | detectors beyond the real data's | survives |
|---|---|---|---|---|---|---|---|
| C14-W1, W2, W3 (ratio law) | 0.99888–0.99934 | 7 / 14 / – | 3.5–4.7×10⁻³ | 2.9–3.9 | none | D-G, D-arg, D-padé | no |
| C14-ARC, four shapes | 1.00010–1.00035 | 4–6 / 12–13 / – | 4.1–6.6×10⁻⁴ | 0.40–0.55 | 0.98–1.03 | D-G, D-padé (and D-arg for two) | no |
| C14-BP [3/11] | 1.000056 | 0 / 8 / 12 | 9×10⁻⁶ | 7.6–8.5×10⁻³ | 0.999981 | D-G, D-padé | no |
| **C14-BP [4/10]** | **0.999991** | **0 / 4 / 10** | 3.7–4.0×10⁻⁷ | 3.3–3.8×10⁻⁴ | 1.000002 | – | **yes** |
| C14-BP [5/9] | 0.999983 | 0 / 1 / 8 | 3.4–4.3×10⁻⁷ | 3.3–3.7×10⁻⁴ | 1.000011 | D-G | no |
| C14-BP [6/8], [7/7] | 0.999993 | 0 / 1 / 7 | 1.3–1.4×10⁻⁷ | 1.2×10⁻⁴ | 1.000021 | – | yes |
| C13-BP [5/8] | 0.999957 | – / 1 / – | 3.7–6.2×10⁻⁶ | 5.9–9.7×10⁻⁴ | 0.999988 | – | yes |
| C13-BP [6/7] | 0.999982 | – / 1 / – | 1.4–1.5×10⁻⁶ | 2.7–2.9×10⁻⁴ | 1.000007 | – | yes |

- **Every ratio-law and arc ring is rejected.** They miss τ₁₅ by 40–390%, and fire D-G and D-padé.
- **C14-BP [3/11] (8 zeros in |x| < 12) is rejected**, and so is the ring-free [5/9] (D-G).
- **Finding: C14-BP [4/10] carries a ring and survives.** Besides x₀ (at |x| ≈ 7.4) it has three zeros in |x| < 12,
  all between |x| = 7.3 and 12, none closer to the origin than x₀. It predicts σ₁₅ to 4×10⁻⁷ and τ₁₅ about as well as
  the benchmark, and fires no detector the real data does not. So σ₀…σ₁₅ do not exclude every ring: a ring just
  beyond x₀ is compatible with them. Here it does not move the estimate: its zero, 0.999991, is within 2×10⁻⁶ of the
  ring-free [6/8] and [7/7] (0.999993), inside both the K = 14 range and the K = 15 range below.
- The ring-free completions [6/8], [7/7] and the two K = 13 ones survive, as they did at K = 14.

### 7.5 Calibration at K = 15

The implied zero for the real K = 15 estimate is real − error over each family:

| family | n | arg | ratio | Padé mid | G |
|---|---|---|---|---|---|
| the 28 arc plants | 28 | **0.999975–1.000018** | 0.999534–1.000332 | 0.999950–1.000039 | 0.999999–1.000003 |
| C-EXP | 1 | 1.000023 | 0.999676 | 0.999810 | 0.999963 |
| REFIT (fit residual ≤ 10⁻⁴) | 5 | 0.999978–1.000001 | 1.000232–1.000385 | 1.000001–1.000063 | 1.000000–1.000001 |
| BP family (C13-BP [5/8], [6/7]; C14-BP [5/9], [6/8]) | 4 | **0.999989–0.999994** | 0.999967–1.000534 | 1.000078–1.000238 | 1.000010–1.000057 |

- The K = 14 calibration rebuilt from the new records reproduces section 6.5 (arg 0.999952–1.000048, G
  1.000039–1.000071).
- **The arc plants' arg error shrank:** +0.01…+0.44×10⁻⁴ at K = 15, against −0.22…+0.74×10⁻⁴ at K = 14, and it no
  longer changes sign across the family.
- **G on the arc plants is now within 3×10⁻⁶,** but section 6.8 showed that G's production value depends on which
  local minimum of its objective it picks; per section 7.0 it enters below only through that spread.
- The plants' σ₁₅ deviate from the measured one by 10⁻³–3×10⁻² (C-EXP 8×10⁻⁵): they are calibration shapes, not
  completions of the real data.

### 7.6 The next generation and the ring-free completions at K = 15

Glued at σ₁₅ (tails from k = 16); zeros counted at wf 1.00:

| completion (K = 15) | its zero | zeros in \|x\| < 7.3 / 12 / 16 | type rule |
|---|---|---|---|
| C15-W1, W2, W3 (ratio law, windows 6–14, 8–14, 9–14) | 1.00018–1.00043 | 7 / 14–15 / – | – |
| C15-ARC, four shapes (best ω_t 1.00035–1.00049, L2 floor 4–7×10⁻⁴) | 1.00004–1.00034 | 5 / 13 / – | – |
| C15-BP [3/12] | 1.000014 | 0 / 9 / 13 | passes |
| C15-BP [4/11] | 0.999992 | 0 / 4 / 11 | passes |
| C15-BP [5/10] | 0.999992 | 0 / 2 / 10 | passes |
| **C15-BP [6/9]** | **0.9999990** | **0 / 1 / 11** | passes |
| **C15-BP [7/8]** | **1.0000018** | **0 / 1 / 5** | passes |

The ring-free completions that pass the type rule are C15-BP [6/9] and [7/8], at 0.9999990 and 1.0000018. The
ratio-law and arc tails carry rings again, as at K = 14.

### 7.7 The order-15 projection

The model est_k = ω* + c r^k through k = 13 and 14 predicts arg₁₅ − 1 = +1.49×10⁻⁵ for ω* = 1, +1.23×10⁻⁵ for 0.99998
and +1.04×10⁻⁵ for 0.99995; the real value is +1.90×10⁻⁵. The zero at 1 is closer (4.1×10⁻⁶ against 6.7×10⁻⁶), but the
model's own error is 2.1×10⁻⁵ (its largest miss over k = 12–14 on the real data; 1.9×10⁻⁵ over the arc plants at
k = 15), so the projection is not decisive. The log-linear trend of arg_k − 1 over k = 4–15 falls ×0.535 per order.

### 7.8 Combination

- K = 15 estimator range (hull): the calibrated arg over the arc plants, 0.999975–1.000018; over the BP family,
  0.999989–0.999994; the ring-free completions, 0.9999990 and 1.0000018. **The hull is 0.999975–1.000018.**
- It overlaps the K = 14 range (0.99998–1.00005), so the headline is their union, **0.999975–1.000050**. As before it
  is an estimator range, not a bound.
- G, with its spread: production 0.999999 at K = 15 (1.000027 at K = 14); over the minima of its objective within a
  factor 5 of the best, 0.999971–1.000301 (0.999762–1.000027 at K = 14); its global minimum 1.000067 (0.999886).

### 7.9 Verdict at K = 15

- σ₁₅ is converged to 5×10⁻¹³ relative and continues every trend of σ₀…σ₁₄.
- The real estimates moved toward 1 again: arg 1.000019, G 0.999999, Padé mid 1.000024, Aitken 1.000015.
- The calibrated range shrank from 0.999952–1.000048 (arc plants, arg, K = 14) to 0.999975–1.000018. The headline
  union, 0.999975–1.000050, changes only at its lower end (0.99998 → 0.999975).
- ω∞ = 1 lies inside every range, as at K = 13 and 14. The data still do not decide between 1 and a value a few 10⁻⁵
  below it.
- One finding against the earlier picture: a ring whose extra zeros lie at 7.3 < |x| < 12 (C14-BP [4/10]) survives
  σ₁₅. The rings of the earlier generations (the ratio-law and arc tails, with zeros in |x| < 7.3) are all rejected.


## 8. Follow-up: σ₁₆ (post-freeze)

### 8.0 Procedure, fixed before any K = 16 estimate was computed

Written and committed before any estimator ran at K = 16, on any input. Already known at this point (comms, 2026-10-27
18:45 UTC, the Mac's T83 pilot): σ₁₆ at wf 1.00 from pilot (a), 4.87773992972141×10⁻¹³ + 1.258001748255×10⁻¹⁴ i, err16
4.1×10⁻²⁵ (8.4×10⁻¹³ relative), with σ₁₃…σ₁₅ equal to `scan31J1000`'s to the digit; pilot (b) at order 31 of 33. The
procedure is section 7.0's one order up, with the changes section 7's findings call for, stated here before the data:

1. **σ₁₆ is converged** if, at every node: (a) err16 ≤ 10⁻¹⁰ |σ₁₆|; (b) σ₀…σ₁₅ are bitwise the `scan31J1000` 960-bit
   values; (c) pilot (b)'s σ₁₆ equals (a)'s within (a)'s err16, and its σ₁₃…σ₁₅ equal `scan31J1000`'s within their errors;
   (d) the second difference of σ_k/σ_{k+1} at k = 14 is at most 0.05. Each check gets a mutation that must make it
   fire (`--mutate=err|low|pilot|ratiolaw`).
2. **The pipeline on plants first.** N1 recovered by arg, ratio, Padé and G to 10⁻⁶ at K = 16, N2 giving no number;
   the 28 arc plants, C-EXP and the 13 REFIT plants regenerated and run at K = 14, 15, 16, their K = 14 and 15 estimates
   equal to the stored K = 15 records (`planted_results_k15.jsonl`) to 10⁻⁹, or the work stops. Control: N1 graded
   against ω_t + 0.002 must fail.
3. **Real data at K = 16** (record "N0 scan33J1000"): its K = 14 and 15 estimates must equal those of "N0 scan31J1000".
   Mutation controls: σ₁₆ → −σ₁₆ at wf 1.00 must fire a detector; σ₁₆ × (1 + 10⁻³) is reported.
4. **Ring test.** Every K = 15 completion of section 7.6 is regenerated and its zero must reproduce the stored
   "K15 recomplete" value to 10⁻⁹: C15-W1–3, C15-ARC (four shapes at their stored best ω_t) and C15-BP [m/(15 − m)],
   m = 3…7. In addition, the one ring that survived σ₁₅ (section 7.9), C14-BP [4/10], is regenerated and tested
   again: its σ₁₆ and τ₁₆ against the measured ones and the harness at K = 16. For each completion: survives if no
   detector fires on it at K = 16 that does not fire on the real data (the section 5.3 rule). Benchmark: the
   simple-zero law from τ₁₄, τ₁₅. A ring that survives σ₁₆ as well is a finding, reported as such; C14-BP [4/10]
   rejected by σ₁₆ closes section 7.9's open item.
5. **Calibration** at K = 16, as section 7.5: the error of arg, ratio, Padé mid and G on (i) the 28 arc plants, (ii)
   C-EXP, (iii) REFIT with fit residual ≤ 10⁻⁴, and (iv) the BP family, now six sums (C13-BP [5/8], [6/7]; C14-BP
   [5/9], [6/8]; C15-BP [6/9], [7/8], the ring-free sums passing the type rule), each with its own zero as truth. The
   calibration at K = 15 from the new records must reproduce section 7.8's arc range 0.999975–1.000018 to 6×10⁻⁷.
6. **Next generation and ring-free completions,** glued at σ₁₆ (tail from k = 17): ratio-law windows k = 6–15, 8–15,
   9–15; best-fit arc tails, four shapes, ω_t grid 0.9990–1.0010; the Borel–Padé sums [m/(16 − m)], m = 3…8. Frozen
   type rule: every pole has |t| < 2 at all three nodes. Zeros counted at wf = 1.00 in |x| < 7.3, 12 and 16.
7. **Combination.** The K = 16 estimator range is the hull of: the calibrated arg range over the 28 arc plants; the
   calibrated arg range over the BP family; and the zeros of the ring-free completions of item 6 that pass the type
   rule (no zeros besides x₀ in |x| < 12). G is reported with its spread over the minima of its objective within a
   factor 5 of the best and is not part of the hull. The headline is the union of the K = 15 headline
   (0.999975–1.000050) and the K = 16 range if they overlap; if disjoint, both are reported and the reason diagnosed
   before any headline is written. The range is an estimator range, not a bound.
8. **The order-16 projection.** est_k = ω* + c r^k through k = 14 and 15, for ω* = 1, 0.99998 and 0.999975 (the lower
   end of the K = 15 range), against the real arg₁₆; decisive only if the other prediction is farther from arg₁₆ than
   three times the model's own error (its largest miss over k = 13–15 on the real data, and over the 28 arc plants
   at k = 16 given their true ω_t).

The stages are `converge`, `plants`, `real`, `ring`, `calib`, `recomplete`, `project`, `combine`, each run as
`nice -n 10 python3 planted_k16.py STAGE` from `solver/inner_amp/`, with the σ₁₆ JSONs copied from the raw-data
repository's T83-mac into `scan33J1000/` and `scan33J1250/` as the earlier orders' were.

The stages ran on 2026-10-27 22:19–22:25 UTC on the Mac (records `planted_results_k16.jsonl`, stored checks
`planted_checks_k16.json`; the stage logs are in the raw-data repository's `planted-k16-mac/`). One script defect
surfaced in the ring stage and was fixed before it completed: for the two completions glued at σ₁₃ (C13-BP [5/8], [6/7])
it asked for K = 13 estimates, which the K = 16 records do not hold; it now checks their K = 14 and 15 estimates against
the stored K = 15 records instead. Nothing in the procedure changed.

### 8.1 σ₁₆ is converged

| wf | σ₁₆ | err16 (relative) | σ₀…σ₁₅ against `scan31J1000` | ratio law: \|Δ²\| at k = 14 |
|---|---|---|---|---|
| 0.99 | +4.9643412758337798976×10⁻¹³ + 4.0468033239927522086×10⁻¹⁴i | 3.74×10⁻²⁵ (7.5×10⁻¹³) | bitwise equal, 16/16 | 0.0099 |
| 1.00 | +4.8777399297214145997×10⁻¹³ + 1.2580017482550000727×10⁻¹⁴i | 4.11×10⁻²⁵ (8.4×10⁻¹³) | bitwise equal, 16/16 | 0.0099 |
| 1.01 | +4.7770853894737897801×10⁻¹³ − 1.4262221779065917070×10⁻¹⁴i | 4.50×10⁻²⁵ (9.4×10⁻¹³) | bitwise equal, 16/16 | 0.0100 |

- **All four criteria of section 8.0 pass,** and each one's mutation (`--mutate=err|low|pilot|ratiolaw`) makes it fire.
  - err16 is 9×10⁻¹³ relative at worst, against the 10⁻¹⁰ asked for.
  - Pilot (b) at wf 1.00 (J 1250, 1280 bits) equals (a)'s σ₁₆ to 2.7×10⁻²⁶ (err16 4.1×10⁻²⁵). Its σ₁₃, σ₁₄, σ₁₅ equal
    `scan31J1000`'s to 8.8×10⁻²⁵, 1.8×10⁻²⁵ and 2.7×10⁻²⁶, each 100–120 times below their errors.
  - The second differences of σ_k/σ_{k+1} are 0.010 at k = 12–14 (0.002–0.048 over k = 7–14).
- **What err16 is:** as before, the late-order fit's plateau step, an estimate of the fit-truncation error, not a
  bound. The pilot tests arithmetic precision only; its own sigma_err is again inf at every k (section 7.1).
- **Spread in wf.** σ₁₆ changes by 5.9% of itself between the nodes (5.7%, 5.8% for σ₁₄, σ₁₅). err16 is 2×10⁻¹¹ of
  that change.

### 8.2 The pipeline on plants first

51 plants at K = 14, 15, 16, before the real data. All four checks pass:
- the regenerated plants' K = 14 and 15 estimates equal the stored K = 15 records to 9.1×10⁻¹³ (REFIT P-ARC
  ω_t = 0.99, ratio at K = 14; every other plant to 6×10⁻¹⁶). At K = 15 the match was exact; the records were made on
  another machine, and the difference is rounding, 10³ below the 10⁻⁹ asked for;
- N1, 7/7: a lone simple zero recovered at K = 16 by arg, ratio, Padé and G to 3.5×10⁻⁷ at worst; graded against
  ω_t + 0.002 it fails, as it must;
- N2, 2/2: no estimator prints a number.

### 8.3 The real estimates at K = 16

The record "N0 scan33J1000" equals "N0 scan31J1000" at K = 14 and 15 to 6×10⁻¹⁶ (rounding, as in 8.2).

| estimator | K = 16 | K = 15 | K = 14 |
|---|---|---|---|
| arg crossing | 1.000010 | 1.000019 | 1.000026 |
| ratio | 0.999882 | 0.999920 | 0.999769 |
| central Padé | [3/12]–[11/4]: 0.999992–1.000009 (mid 1.0000007) | mid 1.000024 | mid 1.000043 |
| G (production κ̄) | 1.000005 | 0.999999 | 1.000027 |
| Aitken, k − 2 … k | 1.000060 | 1.000015 | 1.000017 |
| effective exponent c at ω∞ | 1.002 − 0.002i | 1.002 − 0.002i | 1.000 − 0.004i |

- **The arg crossings keep falling monotonically:** 1.000109, 1.000046, 1.000026, 1.000019, 1.000010 for k = 12–16.
  The distance from 1 shrank by ×0.73 from k = 14 to 15 and by ×0.55 from 15 to 16 (×0.54 per order over k = 4–16).
- **Aitken moved away** (1.000060): the last two differences of arg_k grew (7×10⁻⁶ then 9×10⁻⁶), so the geometric
  extrapolation points the wrong way. It is context, not part of any range.
- **The simple-zero law predicts τ₁₆ from τ₁₄, τ₁₅ to 0.5–0.7×10⁻⁴,** better than it predicted τ₁₅ (1.1–2.6×10⁻⁴).
- **Detectors:** D-ratio and D-ratiolaw. **D-bracket, which fired on the real data at K = 14 and 15, is silent at
  K = 16:** its cause (section 6.3) was G and the Padé mid lying above the arg crossing, and at K = 16 both lie below it
  (1.000005 and 1.0000007 against 1.000010). D-arg, D-padé and D-G stay silent.
- **The mutation controls fire.** σ₁₆ → −σ₁₆ at wf 1.00 fires D-G, D-bracket and D-padé; σ₁₆ × (1 + 10⁻³) fires D-G,
  D-arg, D-bracket and D-padé. Each moves the estimates by about 10⁻².
- **Numerical noise:** under noise of size sigma_err on every σ_k (20 seeds), arg₁₆ moves by at most 2.8×10⁻⁹.

### 8.4 The ring test: what each K = 15 completion predicted for σ₁₆

Every completion was regenerated first and reproduces its stored zero to the printed digits. The benchmark, the
simple-zero law from τ₁₄, τ₁₅, misses τ₁₆ by 0.5–0.7×10⁻⁴.

| completion (K = 15) | its zero | zeros in \|x\| < 7.3 / 12 / 16 | σ₁₆ rel. error | τ₁₆ rel. error | its arg₁₆ | detectors beyond the real data's | survives |
|---|---|---|---|---|---|---|---|
| C15-W1, W2, W3 (ratio law) | 1.00018–1.00043 | 7 / 14–15 / – | 4.3–6.2×10⁻³ | 2.5–3.6 | none | D-G, D-arg, D-bracket, D-padé | no |
| C15-ARC, four shapes | 1.00004–1.00034 | 5 / 13 / – | 2.7–9.3×10⁻⁴ | 0.18–0.54 | 0.99–1.01 | D-G, D-bracket, D-padé (and D-arg for β = 2) | no |
| C15-BP [3/12] | 1.000014 | 0 / 9 / 13 | 7.2×10⁻⁶ | 4.2–4.8×10⁻³ | 0.999887 | D-G, D-bracket, D-padé | no |
| **C15-BP [4/11]** | **0.999992** | **0 / 4 / 11** | 4.8–6.1×10⁻⁷ | 2.8–4.1×10⁻⁴ | 0.999993 | – | **yes** |
| **C15-BP [5/10]** | **0.999992** | **0 / 2 / 10** | 4.3–5.0×10⁻⁷ | 2.8–2.9×10⁻⁴ | 0.999998 | – | **yes** |
| C15-BP [6/9] | 0.999999 | 0 / 1 / 11 | 0.8–1.0×10⁻⁷ | 5.3–5.7×10⁻⁵ | 1.000013 | – | yes |
| C15-BP [7/8] | 1.000002 | 0 / 1 / 5 | 1.9×10⁻⁸ | 1.1–1.3×10⁻⁵ | 1.000011 | D-bracket | no |
| C14-BP [4/10] (the ring of 7.9) | 0.999991 | 0 / 4 / 10 | 3.5–3.8×10⁻⁶ | 2.4–4.6×10⁻⁴ | 1.000024 | D-G, D-arg, D-bracket | no |
| C14-BP [5/9] | 0.999983 | 0 / 1 / 8 | 3.4–4.2×10⁻⁶ | 3.3–4.3×10⁻⁴ | 0.999988 | – | yes |
| C14-BP [6/8] | 0.999993 | 0 / 1 / 7 | 1.3–1.5×10⁻⁶ | 1.8×10⁻⁴ | 1.000003 | D-bracket | no |
| C13-BP [5/8] | 0.999957 | – / 1 / – | 2.0–3.3×10⁻⁵ | 0.5–1.0×10⁻³ | 0.999988 | D-arg, D-bracket | no |
| C13-BP [6/7] | 0.999982 | – / 1 / – | 7.7–8.5×10⁻⁶ | 2.7–3.2×10⁻⁴ | 0.999995 | D-bracket | no |

- **Every ratio-law and arc ring is rejected,** as at K = 14 and 15: they miss τ₁₆ by 18–360% and fire D-G and D-padé.
- **C14-BP [4/10], the ring that survived σ₁₅, is rejected by σ₁₆.** It mispredicts σ₁₆ by 3.5–3.8×10⁻⁶ (ten times
  worse than it predicted σ₁₅) and τ₁₆ five times worse than the benchmark, and fires D-G and D-arg. Section 7.9's open
  item is closed: that ring does not continue the series.
- **Finding: two new rings survive, C15-BP [4/11] and [5/10].** They are the Borel–Padé sums of σ₀…σ₁₅ with three and
  one extra zeros between |x| = 7.3 and 12 (none nearer the origin than x₀). They predict σ₁₆ to 4–6×10⁻⁷ and τ₁₆
  within four to six times the benchmark, and fire no detector the real data does not. The pattern repeats from K = 15:
  at each order the latest Borel–Padé sums with low numerator degree carry a ring just beyond x₀ and survive one more
  coefficient, and the next coefficient rejects them. As before they do not move the estimate: both have their zero at
  0.999992, within 7×10⁻⁶ of the ring-free [6/9] (0.999999) and inside the K = 15 and K = 16 ranges.
- **D-bracket now rejects on its own.** Because the real data stopped firing it (8.3), section 5.3's rule counts it
  as a detector "beyond the real data's", and it is the only ground on which C15-BP [7/8], C14-BP [6/8] and C13-BP
  [6/7] fail, though [7/8] is the best predictor of σ₁₆ in the table (1.9×10⁻⁸) and of τ₁₆ (1.1×10⁻⁵, four times better
  than the benchmark). Section 6.3 found D-bracket non-discriminating once arg is as close to the zero as the other
  estimators' errors, which is now the case for the real data too. The table applies the rule as section 8.0 fixed it;
  read without D-bracket, the survivors are the four Borel–Padé sums above plus [7/8], [6/8] and [6/7], and nothing
  else changes. The rule is left for the coordinator to revise before K = 17, not here.

### 8.5 Calibration at K = 16

The implied zero for the real K = 16 estimate is real − error over each family:

| family | n | arg | ratio | Padé mid | G |
|---|---|---|---|---|---|
| the 28 arc plants | 28 | **0.999982–1.000014** | 0.999922–1.000134 | 0.999998–1.000013 | 1.000004–1.000005 |
| C-EXP | 1 | 1.000019 | 0.999964 | 0.999864 | 0.999998 |
| REFIT (fit residual ≤ 10⁻⁴) | 5 | 0.999988–0.999994 | 0.999877–1.000144 | 0.999979–1.000005 | 1.000004 |
| BP family (six sums, section 8.0 item 5) | 6 | **0.999980–1.000005** | 0.999847–1.000222 | 0.999796–1.000224 | 0.999964–1.000012 |

- The K = 15 calibration rebuilt from the new records reproduces section 7.5 (arc 0.999975–1.000018, BP family of four
  0.999989–0.999994).
- **The arc plants' arg error shrank again:** −0.04…+0.28×10⁻⁴ at K = 16 against +0.01…+0.44×10⁻⁴ at K = 15; it
  changes sign across the family once more (the mirror shapes are now slightly low).
- G on the arc plants is within 5×10⁻⁶, as at K = 15; it enters below only through its spread over minima (8.8).
- The plants' σ₁₆ deviate from the measured one by 2×10⁻³–5×10⁻² (C-EXP 3×10⁻⁴): calibration shapes, not completions.

### 8.6 The next generation and the ring-free completions at K = 16

Glued at σ₁₆ (tails from k = 17); zeros counted at wf 1.00:

| completion (K = 16) | its zero | zeros in \|x\| < 7.3 / 12 / 16 | type rule |
|---|---|---|---|
| C16-W1, W2, W3 (ratio law, windows 6–15, 8–15, 9–15) | 1.00078–1.00092 | 8 / 15 / – | – |
| C16-ARC, four shapes (best ω_t 1.00035–1.00051, L2 floor 6–8×10⁻⁴) | 0.99997–1.00028 | 1–5 / 13 / – | – |
| C16-BP [3/13] | 1.0000005 | 0 / 8 / 14 | passes |
| C16-BP [4/12] | 0.9999957 | 0 / 5 / 11 | passes |
| C16-BP [5/11] | 0.9999960 | 0 / 3 / 14 | fails (a pole at \|t\| = 2.19 and 2.27 at wf 1.00, 1.01) |
| **C16-BP [6/10]** | **1.0000005** | **0 / 1 / 9** | passes |
| **C16-BP [7/9]** | **1.0000007** | **0 / 1 / 5** | passes |
| **C16-BP [8/8]** | **1.0000007** | **0 / 1 / 5** | passes |

The ring-free completions that pass the type rule are C16-BP [6/10], [7/9] and [8/8], at 1.0000005–1.0000007, the
tightest cluster so far (K = 15: 0.9999990 and 1.0000018). The ratio-law and arc tails carry rings again, and the
low-degree Borel–Padé sums [3/13] and [4/12] carry the rings beyond x₀ that 8.4 found at K = 15.

### 8.7 The order-16 projection

The model est_k = ω* + c r^k through k = 14 and 15 predicts arg₁₆ − 1 = +1.38×10⁻⁵ for ω* = 1, +1.29×10⁻⁵ for 0.99998
and +1.28×10⁻⁵ for 0.999975; the real value is +1.04×10⁻⁵. The lower zeros are closer (2.5×10⁻⁶ against 3.4×10⁻⁶), but
the model's own error is 1.5×10⁻⁵ (its largest miss over k = 13–15 on the real data, 1.4×10⁻⁵, and over the arc plants
at k = 16, −1.5…+0.5×10⁻⁵), so the projection is not decisive, as at K = 15. The log-linear trend of arg_k − 1 over
k = 4–16 falls ×0.540 per order (×0.535 over k = 4–15).

### 8.8 Combination

- K = 16 estimator range (hull): the calibrated arg over the arc plants, 0.999982–1.000014; over the BP family,
  0.999980–1.000005; the ring-free completions, 1.0000005–1.0000007. **The hull is 0.999980–1.000014.**
- It overlaps the K = 15 headline (0.999975–1.000050), so by section 8.0 item 7 the headline is their union,
  **0.999975–1.000050, unchanged.** The K = 16 range alone is narrower than the K = 15 one (0.999975–1.000018) at both
  ends. An estimator range, not a bound.
- G, with its spread: production 1.000005 at K = 16 (0.999999 at K = 15); over the minima of its objective within a
  factor 5 of the best, 0.999971–1.000005 (0.999971–1.000301 at K = 15); its global minimum 0.999971 (1.000067).

### 8.9 Verdict at K = 16

- σ₁₆ is converged to 9×10⁻¹³ relative and continues every trend of σ₀…σ₁₅.
- The real estimates moved toward 1 again: arg 1.000010, G 1.000005, Padé mid 1.0000007; the ring-free sums of
  σ₀…σ₁₆ sit at 1.0000005–1.0000007.
- The calibrated range shrank from 0.999975–1.000018 (arc plants, arg, K = 15) to 0.999982–1.000014, and the K = 16 hull
  is 0.999980–1.000014. The headline union stays 0.999975–1.000050 because the procedure keeps the earlier order's
  range; a headline from K = 15 and 16 alone would be 0.999975–1.000018.
- ω∞ = 1 lies inside every range, as at K = 13, 14 and 15. The data still do not decide between 1 and a value a few
  10⁻⁵ below it; the order-16 projection leans to the lower zeros by less than its own error.
- The ring that survived σ₁₅ (C14-BP [4/10]) is rejected by σ₁₆, closing section 7.9's finding; two rings of the same
  kind built from σ₀…σ₁₅ (C15-BP [4/11], [5/10]) survive σ₁₆ in turn. One coefficient rejects each generation's ring
  and admits the next one's, so σ₀…σ₁₆ exclude rings nearer than x₀ and rings built from fewer coefficients, not a
  ring just beyond x₀ fitted to all of them. Those rings put the zero at 0.999992, inside the range.
- A rule question for K = 17: D-bracket stopped firing on the real data, so section 5.3's comparison now rejects
  completions on it alone, including the best predictor of σ₁₆ (8.4). Whether to drop it from the comparison, as
  section 6.3's diagnosis suggests, is decided before σ₁₇, not after.

## 9. Follow-up: σ₁₇ (post-freeze)

### 9.0 Procedure, fixed before any K = 17 estimate was computed

Written and committed before any estimator ran at K = 17, on any input. Already known at this point (comms, 2026-10-27
23:26 UTC, the Mac's T86; its logs at 01:17 UTC on 2026-10-28): pilot (a) at wf 1.00 is at order 29 of 35 and pilot (b)
at order 25; no σ₁₇ exists yet. The procedure is section 8.0's one order up, with the changes section 8's findings call
for, stated here before the data:

0. **Dry run of the stages on a synthetic σ₁₇.** Before the real σ₁₇ exists, `planted_k17_dry.py` writes scan files
   whose σ₀…σ₁₆ are the real ones and whose σ₁₇ at each node is the C16-BP [7/9] sum's (ring-free, passing the type
   rule, zero at 1.0000007; section 8.6), with err17 = 10⁻¹² |σ₁₇| and the pilot's errors inf, and every stage runs on
   them (`planted_k17.py STAGE --data DIR`). Pass: every stage completes with its checks passing, and in the ring test
   C16-BP [7/9] has σ₁₇ and τ₁₇ errors below 10⁻¹⁸ and survives; whether its zero lies inside the dry run's K = 17
   range is reported. A stage that fails on the synthetic data is fixed before the real data, and the fix is recorded
   here. The dry run's records and checks are discarded (its logs go to the raw-data repository); it tests the code
   path, not the estimators (item 2 does that).
1. **σ₁₇ is converged** if, at every node: (a) err17 ≤ 10⁻¹⁰ |σ₁₇|; (b) σ₀…σ₁₆ are bitwise the `scan33J1000` 960-bit
   values; (c) pilot (b)'s σ₁₇ equals (a)'s within (a)'s err17, and its σ₁₄…σ₁₆ equal `scan33J1000`'s within their errors;
   (d) the second difference of σ_k/σ_{k+1} at k = 15 is at most 0.05. Each check gets a mutation that must make it
   fire (`--mutate=err|low|pilot|ratiolaw`).
2. **The pipeline on plants first.** N1 recovered by arg, ratio, Padé and G to 10⁻⁶ at K = 17, N2 giving no number;
   the 28 arc plants, C-EXP and the 13 REFIT plants regenerated and run at K = 15, 16, 17, their K = 15 and 16 estimates
   equal to the stored K = 16 records (`planted_results_k16.jsonl`) to 10⁻⁹, or the work stops. Control: N1 graded
   against ω_t + 0.002 must fail.
3. **Real data at K = 17** (record "N0 scan35J1000"): its K = 15 and 16 estimates must equal those of "N0 scan33J1000".
   Mutation controls: σ₁₇ → −σ₁₇ at wf 1.00 must fire a detector; σ₁₇ × (1 + 10⁻³) is reported.
4. **Ring test.** Every K = 16 completion of section 8.6 is regenerated and its zero must reproduce the stored
   "K16 recomplete" value to 10⁻⁹: C16-W1–3, C16-ARC (four shapes at their stored best ω_t) and C16-BP [m/(16 − m)],
   m = 3…8. In addition, the two rings that survived σ₁₆ (section 8.4), C15-BP [4/11] and [5/10], are regenerated and
   tested again: their σ₁₇ and τ₁₇ against the measured ones and the harness at K = 17. Benchmark: the simple-zero law
   from τ₁₅, τ₁₆. **The D-bracket question (8.9):** section 5.3's comparison stays as it is, a completion survives if no
   detector fires on it at K = 17 that does not fire on the real data; and because the real data stopped firing
   D-bracket at K = 16 (8.3), the table also gives a second reading with D-bracket left out of the comparison, fixed
   here so that neither reading is chosen after the data. A ring is a finding only if it survives under the first
   reading; a ring rejected under the first and surviving under the second is reported as that. This is the
   coordinator's provisional choice, made because the author's decision asked for in 8.9 had not come; if the author
   decides before any K = 17 estimate is computed, this item is amended with the date. A ring that survives σ₁₇ as
   well is a finding, reported as such; C15-BP [4/11] and [5/10] rejected by σ₁₇ closes section 8.4's open item.
5. **Calibration** at K = 17, as section 8.5: the error of arg, ratio, Padé mid and G on (i) the 28 arc plants, (ii)
   C-EXP, (iii) REFIT with fit residual ≤ 10⁻⁴, and (iv) the BP family, now nine sums (section 8.5's six and C16-BP
   [6/10], [7/9], [8/8], the ring-free sums passing the type rule at K = 16), each with its own zero as truth. The
   calibration at K = 16 from the new records must reproduce section 8.5's arc range 0.999982–1.000014 and BP-family
   range 0.999980–1.000005 to 6×10⁻⁷ (against the values stored in `planted_checks_k16.json`).
6. **Next generation and ring-free completions,** glued at σ₁₇ (tail from k = 18): ratio-law windows k = 6–16, 8–16,
   9–16; best-fit arc tails, four shapes, ω_t grid 0.9990–1.0010; the Borel–Padé sums [m/(17 − m)], m = 3…8. Frozen
   type rule: every pole has |t| < 2 at all three nodes. Zeros counted at wf = 1.00 in |x| < 7.3, 12 and 16.
7. **Combination.** The K = 17 estimator range is the hull of: the calibrated arg range over the 28 arc plants; the
   calibrated arg range over the BP family (nine); and the zeros of the ring-free completions of item 6 that pass the
   type rule (no zeros besides x₀ in |x| < 12). G is reported with its spread over the minima of its objective within a
   factor 5 of the best and is not part of the hull. The headline is the union of the K = 16 headline
   (0.999975–1.000050) and the K = 17 range if they overlap; if disjoint, both are reported and the reason diagnosed
   before any headline is written. The range is an estimator range, not a bound. Section 8.9 noted that the union never
   narrows; the rule is kept, and the range from the last two orders alone is reported beside it as context.
8. **The order-17 projection.** est_k = ω* + c r^k through k = 15 and 16, for ω* = 1, 0.99998 (the lower end of the
   K = 16 range, 0.999980) and 0.999975 (the lower end of the headline), against the real arg₁₇; decisive only if the
   other prediction is farther from arg₁₇ than three times the model's own error (its largest miss over k = 14–16 on
   the real data, and over the 28 arc plants at k = 17 given their true ω_t).

The stages are `converge`, `plants`, `real`, `ring`, `calib`, `recomplete`, `project`, `combine`, each run as
`nice -n 10 python3 planted_k17.py STAGE` from `solver/inner_amp/`, with the σ₁₇ JSONs copied from the raw-data
repository's T86-mac into `scan35J1000/` and `scan35J1250/` as the earlier orders' were.

The dry run (item 0) ran on 2026-10-28 01:29–01:39 UTC on the Mac, beside T86: all eight stages completed and their 62
checks passed (the logs are in the raw-data repository's `planted-k17-mac/dry/`). In the ring test the planted
C16-BP [7/9] reproduced its σ₁₇ to 1.2×10⁻²⁰ and survived under both readings, and its zero (1.0000007) lies inside
the dry run's K = 17 range, 0.999990–1.000009. One criterion of item 0 was missed: its τ₁₇ error was 3.3–4.7×10⁻¹⁸,
not below 10⁻¹⁸. The cause is the scans' 20-digit format, in which the synthetic σ₁₇ was written: τ₁₇ amplifies that
rounding about 400 times (τ₁₆ from the same file is exact to 10⁻⁴⁵, and the sum's exact σ₁₇ gives its exact τ₁₇), so
the criterion was too tight for 20-digit input; it is left as written. The two rings of 8.4 were rejected by the
synthetic σ₁₇ (D-G, D-arg), which says nothing about the real one. No stage code changed; the one defect was in the
coordinator's stage runner (a shell word split), outside the procedure.

The stages ran on 2026-10-28 07:21–07:30 UTC on the Mac, with the σ₁₇ scans of T86 (raw-data repository `T86-mac/` at
acc84cd) committed to `scan35J1000/` and `scan35J1250/` first. Records `planted_results_k17.jsonl`, stored checks
`planted_checks_k17.json`; the stage logs and the four convergence mutations are in the raw-data repository's
`planted-k17-mac/`. All 62 checks passed. No stage code changed between the dry run and this run, and the author's
decision on D-bracket did not come before the first estimate, so item 4 stands as written.

### 9.1 σ₁₇ is converged

| wf | σ₁₇ | err17 (relative) | σ₀…σ₁₆ against `scan33J1000` | ratio law: \|Δ²\| at k = 15 |
|---|---|---|---|---|
| 0.99 | +2.7351243334556598802×10⁻¹⁴ + 3.1563007458762078940×10⁻¹⁴i | 4.96×10⁻²⁶ (1.19×10⁻¹²) | bitwise equal, 17/17 | 0.0091 |
| 1.00 | +2.8468173979361812918×10⁻¹⁴ + 2.9335336047966674292×10⁻¹⁴i | 5.44×10⁻²⁶ (1.33×10⁻¹²) | bitwise equal, 17/17 | 0.0091 |
| 1.01 | +2.9434600330414006322×10⁻¹⁴ + 2.7089922226794287657×10⁻¹⁴i | 5.95×10⁻²⁶ (1.49×10⁻¹²) | bitwise equal, 17/17 | 0.0091 |

- **All four criteria of section 9.0 pass,** and each one's mutation (`--mutate=err|low|pilot|ratiolaw`) makes it fire
  (err17 × 10¹⁴: 133 relative; σ₁₆ at wf 0.99 scaled by 1 + 10⁻¹⁵: 16/17 bitwise; pilot (b) shifted: 2.9×10⁻²⁴ against
  5.4×10⁻²⁶; a scaled σ₁₇ at wf 1.01: |Δ²| = 1.09).
  - err17 is 1.5×10⁻¹² relative at worst, against the 10⁻¹⁰ asked for.
  - Pilot (b) at wf 1.00 (J 1250, 1280 bits) equals (a)'s σ₁₇ to 1.6×10⁻²⁷ (err17 5.4×10⁻²⁶). Its σ₁₄, σ₁₅, σ₁₆ equal
    `scan33J1000`'s to 1.8×10⁻²⁵, 2.7×10⁻²⁶ and 2.7×10⁻²⁶, 100, 110 and 15 times below their errors.
  - The second differences of σ_k/σ_{k+1} are 0.009–0.012 at k = 10–15 (0.002–0.048 over k = 7–15).
- **What err17 is:** as before, the late-order fit's plateau step, an estimate of the fit-truncation error, not a
  bound. The pilot tests arithmetic precision only; its own sigma_err is again inf at every k (section 7.1).
- **Spread in wf.** σ₁₇ changes by 6.0% of itself between the nodes (5.8%, 5.9% for σ₁₅, σ₁₆). err17 is 2.4×10⁻¹¹ of
  that change.
- **Cost.** On the Mac with two processes per run: pilot (a) 200 min, pilot (b) 394 min, the nodes 252 min each, the
  late-order fits taking 84% of the time (comms, 2026-10-28 02:47–06:59 UTC).

### 9.2 The pipeline on plants first

51 plants at K = 15, 16, 17, before the real data. All four checks pass:
- the regenerated plants' K = 15 and 16 estimates equal the stored K = 16 records exactly (both runs on the Mac);
- N1, 7/7: a lone simple zero recovered at K = 17 by arg, ratio, Padé and G to 3.5×10⁻⁷ at worst; graded against
  ω_t + 0.002 it fails, as it must;
- N2, 2/2: no estimator prints a number.

### 9.3 The real estimates at K = 17

The record "N0 scan35J1000" equals "N0 scan33J1000" at K = 15 and 16 exactly.

| estimator | K = 17 | K = 16 | K = 15 |
|---|---|---|---|
| arg crossing | 1.0000046 | 1.000010 | 1.000019 |
| ratio | 0.999912 | 0.999882 | 0.999920 |
| central Padé | [3/13]–[12/4]: 0.999978–1.000002 (mid 0.999990); [4/12]–[12/4]: 0.999995–1.000002 | 0.999992–1.000009 (mid 1.0000007) | mid 1.000024 |
| G (production κ̄) | 1.0000026 | 1.000005 | 0.999999 |
| Aitken, k − 2 … k | 0.9999925 | 1.000060 | 1.000015 |
| effective exponent c at ω∞ | 1.000 − 0.002i | 1.002 − 0.002i | 1.002 − 0.002i |

- **The arg crossings keep falling monotonically:** 1.000109, 1.000046, 1.000026, 1.000019, 1.000010, 1.0000046 for
  k = 12–17. The distance from 1 shrank by ×0.55 from k = 15 to 16 and by ×0.44 from 16 to 17 (×0.539 per order over
  k = 4–17).
- **Aitken came back, and overshot:** the last two differences of arg_k fall again (9.0×10⁻⁶, then 5.8×10⁻⁶), so the
  geometric extrapolation lands at 0.9999925, 7.5×10⁻⁶ below 1. Context, not part of any range.
- **The central Padé range widened downward,** to 0.999978, through the single approximant [3/13]; [4/12]–[12/4] lie at
  0.999995–1.000002, and the mid, 0.999990, is below 1 for the first time since K = 14.
- **The simple-zero law predicts τ₁₇ from τ₁₅, τ₁₆ to 0.8–0.9×10⁻⁴,** slightly worse than it predicted τ₁₆
  (0.5–0.7×10⁻⁴).
- **Detectors:** D-ratio and D-ratiolaw, as at K = 16. **D-bracket is silent again:** G (1.0000026) and the Padé mid
  (0.999990) both lie below the arg crossing (1.0000046), the configuration section 6.3 named. D-arg, D-padé and D-G
  stay silent.
- **The mutation controls fire.** σ₁₇ → −σ₁₇ at wf 1.00 fires D-G and D-bracket; σ₁₇ × (1 + 10⁻³) fires D-G, D-arg
  and D-bracket. Each moves the estimates by about 10⁻².
- **Numerical noise:** under noise of size sigma_err on every σ_k (20 seeds), arg₁₇ moves by at most 5.9×10⁻⁹.

### 9.4 The ring test: what each K = 16 completion predicted for σ₁₇

Every completion was regenerated first and reproduces its stored zero to the printed digits. The benchmark, the
simple-zero law from τ₁₅, τ₁₆, misses τ₁₇ by 0.8–0.9×10⁻⁴. The last two columns are the two readings of section 9.0
item 4: section 5.3's rule, and the same rule with D-bracket left out of the comparison.

| completion (K = 16) | its zero | zeros in \|x\| < 7.3 / 12 / 16 | σ₁₇ rel. error | τ₁₇ rel. error | its arg₁₇ | detectors beyond the real data's | survives | without D-bracket |
|---|---|---|---|---|---|---|---|---|
| C16-W1, W2, W3 (ratio law) | 1.00078–1.00092 | 8 / 15 / – | 5.1–7.5×10⁻³ | 2.0–2.9 | none | D-G, D-arg, D-bracket | no | no |
| C16-ARC, four shapes | 0.99997–1.00028 | 1–5 / 13 / – | 1.3×10⁻⁴–1.2×10⁻³ | 0.06–0.48 | 0.980–0.996 | D-G, D-bracket | no | no |
| C16-BP [3/13] | 1.0000005 | 0 / 8 / 14 | 5.2–5.6×10⁻⁶ | 2.0–2.5×10⁻³ | 0.999903 | D-G, D-bracket | no | no |
| C16-BP [4/12] | 0.9999957 | 0 / 5 / 11 | 7.8–8.4×10⁻⁷ | 3.2–3.4×10⁻⁴ | 0.999991 | D-G | no | no |
| C16-BP [5/11] | 0.9999960 | 0 / 3 / 14 | 0.6–1.0×10⁻⁶ | 2.7–3.9×10⁻⁴ | 0.999985 | D-G, D-bracket | no | no |
| C16-BP [6/10] | 1.0000005 | 0 / 1 / 9 | 5.2–5.5×10⁻⁸ | 2.1–2.3×10⁻⁵ | 1.000006 | – | yes | yes |
| C16-BP [7/9] | 1.0000007 | 0 / 1 / 5 | 2.0–2.4×10⁻⁸ | 9.0–9.2×10⁻⁶ | 1.000005 | D-bracket | no | yes |
| C16-BP [8/8] | 1.0000007 | 0 / 1 / 5 | 2.0–2.4×10⁻⁸ | 9.0–9.2×10⁻⁶ | 1.000005 | D-bracket | no | yes |
| **C15-BP [4/11]** (ring of 8.4) | 0.999992 | 0 / 4 / 11 | 4.7–5.6×10⁻⁶ | 2.7–4.7×10⁻⁴ | 1.000004 | D-G, D-arg, D-bracket | **no** | **no** |
| **C15-BP [5/10]** (ring of 8.4) | 0.999992 | 0 / 2 / 10 | 4.0–4.7×10⁻⁶ | 3.0–3.4×10⁻⁴ | 0.999998 | D-G, D-bracket | **no** | **no** |
| C15-BP [6/9] | 0.999999 | 0 / 1 / 11 | 0.9–1.0×10⁻⁶ | 7.1–8.0×10⁻⁵ | 1.000003 | D-bracket | no | yes |
| C15-BP [7/8] | 1.000002 | 0 / 1 / 5 | 2.3–2.4×10⁻⁷ | 2.6×10⁻⁵ | 1.000006 | – | yes | yes |
| C14-BP [5/9] | 0.999983 | 0 / 1 / 8 | 1.8–2.2×10⁻⁵ | 2.0–2.1×10⁻⁴ | 0.999997 | D-arg, D-bracket | no | no |
| C14-BP [6/8] | 0.999993 | 0 / 1 / 7 | 7.3–8.0×10⁻⁶ | 1.1–1.3×10⁻⁴ | 0.999999 | D-bracket | no | yes |
| C13-BP [5/8] | 0.999957 | – / 1 / – | 0.7–1.3×10⁻⁴ | 0.7–1.3×10⁻³ | 0.999968 | D-arg | no | no |
| C13-BP [6/7] | 0.999982 | – / 1 / – | 3.0–3.4×10⁻⁵ | 3.0–3.3×10⁻⁴ | 0.999989 | – | yes | yes |

- **Every ratio-law and arc ring is rejected,** as at K = 14, 15 and 16: they miss τ₁₇ by 6–290% and fire D-G and
  D-bracket (D-arg too for the ratio-law tails). D-padé, which fired on them at the earlier orders, does not at K = 17.
- **The two rings that survived σ₁₆, C15-BP [4/11] and [5/10], are rejected by σ₁₇,** under both readings. They
  mispredict σ₁₇ by 4–6×10⁻⁶ (ten times worse than they predicted σ₁₆) and τ₁₇ by three to five times the benchmark,
  and fire D-G ([4/11] also D-arg). Section 8.4's open item is closed: those rings do not continue the series.
- **No ring survives σ₁₇, and for the first time none of the latest generation does.** The ring-carrying sums of
  σ₀…σ₁₆, C16-BP [3/13], [4/12] and [5/11] (three to eight zeros in |x| < 12), fire D-G at once, where at K = 15 and
  16 the newest low-degree Borel–Padé sums survived one coefficient. The two readings agree on every ring: no ring
  survives with or without D-bracket, so the provisional rule of 9.0 item 4 changed no finding.
- **The ring-free sums of σ₀…σ₁₆ predicted σ₁₇ to 2–6×10⁻⁸** ([7/9] and [8/8] 2.0–2.4×10⁻⁸, [6/10] 5.2–5.5×10⁻⁸) and
  τ₁₇ to 0.9–2.3×10⁻⁵, four to ten times better than the benchmark. [6/10] survives under both readings; [7/9] and
  [8/8], the best predictors in the table, are rejected by D-bracket alone, as [7/8] was at K = 16, and so are C15-BP
  [6/9] and C14-BP [6/8]. The survivors under section 5.3's rule are C16-BP [6/10], C15-BP [7/8] and C13-BP [6/7];
  without D-bracket, those and C16-BP [7/9], [8/8], C15-BP [6/9], C14-BP [6/8]. All are ring-free, with zeros at
  0.999982–1.000002, and none moves the estimate.

### 9.5 Calibration at K = 17

The implied zero for the real K = 17 estimate is real − error over each family:

| family | n | arg | ratio | Padé mid | G |
|---|---|---|---|---|---|
| the 28 arc plants | 28 | **0.999990–1.000009** | 0.999907–1.000121 | 0.999928–0.999994 | 1.000002–1.000003 |
| C-EXP | 1 | 1.000008 | 0.999827 | 0.999834 | 1.000015 |
| REFIT (fit residual ≤ 10⁻⁴) | 5 | 0.999991–0.999997 | 1.000043–1.000057 | 0.999982–0.999988 | 1.000003 |
| BP family (nine sums, section 9.0 item 5) | 9 | **0.999990–1.000000** | 0.999750–1.000222 | 0.999926–1.000131 | 0.999972–1.000002 |

- The K = 16 calibration rebuilt from the new records reproduces section 8.5 exactly (arc 0.999982–1.000014, BP family
  of six 0.999980–1.000005).
- **The arc plants' arg error shrank again:** −0.04…+0.15×10⁻⁴ at K = 17 against −0.04…+0.28×10⁻⁴ at K = 16, still
  changing sign across the family (the mirror shapes low).
- G on the arc plants is within 3×10⁻⁶ of their zeros; it enters below only through its spread over minima (9.8).
- The Padé mid is now biased high on the arc plants (its error +0.5–0.6×10⁻⁴ on the mirror shapes), so its implied zero
  lies at 0.999928–0.999994; Aitken on the mirror plants runs away (−43×10⁻⁴). Neither is part of the hull.
- The plants' σ₁₇ deviate from the measured one by 4×10⁻³–8×10⁻² (C-EXP 10⁻³): calibration shapes, not completions.

### 9.6 The next generation and the ring-free completions at K = 17

Glued at σ₁₇ (tails from k = 18); zeros counted at wf 1.00:

| completion (K = 17) | its zero | zeros in \|x\| < 7.3 / 12 / 16 | type rule |
|---|---|---|---|
| C17-W1, W2, W3 (ratio law, windows 6–16, 8–16, 9–16) | 1.00068–1.00090 | 7 / 16 / – | – |
| C17-ARC, four shapes (best ω_t 1.00033–1.00043, L2 floor 0.8–1.2×10⁻³) | 0.99995–1.00019 | 2–5 / 14 / – | – |
| C17-BP [3/14] | 0.9999980 | 0 / 9 / 15 | passes |
| C17-BP [4/13] | 0.9999985 | 0 / 6 / 12 | passes |
| C17-BP [5/12] | 1.0000036 | 0 / 3 / 13 | fails (a pole at \|t\| = 2.38 at wf 0.99) |
| **C17-BP [6/11]** | **1.0000004** | **0 / 1 / 10** | passes |
| **C17-BP [7/10]** | **1.0000002** | **0 / 1 / 7** | passes |
| **C17-BP [8/9]** | **0.9999999** | **0 / 1 / 6** | passes |

The ring-free completions that pass the type rule are C17-BP [6/11], [7/10] and [8/9], at 0.9999999–1.0000004, the
tightest cluster so far (K = 16: 1.0000005–1.0000007) and the first to straddle 1. The ratio-law and arc tails carry
rings again, and the low-degree sums [3/14] and [4/13] carry rings beyond x₀, with their zero at 0.999998; σ₁₈ tests
them.

### 9.7 The order-17 projection

The model est_k = ω* + c r^k through k = 15 and 16 predicts arg₁₇ − 1 = +5.67×10⁻⁶ for ω* = 1, +3.67×10⁻⁶ for 0.99998
and +3.45×10⁻⁶ for 0.999975; the real value is +4.57×10⁻⁶, 1.1×10⁻⁶ from the first and 0.9×10⁻⁶ from the second. The
model's own error is 1.4×10⁻⁵ (its largest miss over k = 14–16 on the real data, 0.7–1.4×10⁻⁵ depending on ω*, and over
the arc plants at k = 17, −0.9…+0.5×10⁻⁵), so the projection is not decisive, as at K = 15 and 16. The log-linear trend
of arg_k − 1 over k = 4–17 falls ×0.539 per order (×0.540 over k = 4–16).

### 9.8 Combination

- K = 17 estimator range (hull): the calibrated arg over the arc plants, 0.999990–1.000009; over the BP family of nine,
  0.999990–1.000000; the ring-free completions, 0.9999999–1.0000004. **The hull is 0.999990–1.000009.**
- It overlaps the K = 16 headline (0.999975–1.000050), so by section 9.0 item 7 the headline is their union,
  **0.999975–1.000050, unchanged.** The K = 17 range alone is narrower than the K = 16 one (0.999980–1.000014) at both
  ends; the range from the last two orders alone, reported as context by item 7, is 0.999980–1.000014. An estimator
  range, not a bound.
- G, with its spread: production 1.0000026 at K = 17 (1.000005 at K = 16); over the minima of its objective within a
  factor 5 of the best, 0.999908–1.000003 (0.999971–1.000005 at K = 16), the low end from one minimum with a third of
  the best objective's weight; its global minimum 1.0000001 (0.999971).

### 9.9 Verdict at K = 17

- σ₁₇ is converged to 1.5×10⁻¹² relative and continues every trend of σ₀…σ₁₆.
- The real estimates moved toward 1 again: arg 1.0000046, G 1.0000026, central Padé [4/12]–[12/4] at 0.999995–1.000002;
  the ring-free sums of σ₀…σ₁₇ sit at 0.9999999–1.0000004, on both sides of 1.
- The calibrated range shrank from 0.999982–1.000014 (arc plants, arg, K = 16) to 0.999990–1.000009, and the K = 17 hull
  is 0.999990–1.000009. The headline union stays 0.999975–1.000050 because the procedure keeps the earlier orders'
  ranges; the last two orders alone give 0.999980–1.000014, and K = 15–17 would give 0.999975–1.000018.
- ω∞ = 1 lies inside every range, as at K = 13–16. The order-17 projection puts all three hypotheses within 1.1×10⁻⁶
  of arg₁₇, against a model error of 1.4×10⁻⁵: the data still do not decide between 1 and a value 10⁻⁵ below it.
- The two rings that survived σ₁₆ are rejected by σ₁₇, closing section 8.4's finding, and for the first time no ring
  of the latest generation survives either: the ring-carrying sums of σ₀…σ₁₆ fail at once. So σ₀…σ₁₇ exclude every
  ring built so far, nearer than x₀ or just beyond it, from any number of coefficients up to sixteen. The next
  generation (C17-BP [3/14], [4/13], zero at 0.999998) is tested by σ₁₈ (T87, running).
- D-bracket: the two readings of 9.0 item 4 agree on every ring. On its own it rejected only ring-free sums, among them
  the best predictors of σ₁₇ (2×10⁻⁸), as it did at K = 16. The rule stays as item 4 fixed it for K = 18 unless the
  author decides otherwise before σ₁₈; it changed no finding at K = 17.

## 10. Follow-up: σ₁₈ (post-freeze)

### 10.0 Procedure, fixed before any K = 18 estimate was computed

Written and committed before any estimator ran at K = 18, on any input. Already known at this point (the Mac's T87,
started 2026-10-28 06:58 UTC; its logs at 08:18 UTC): pilot (a) at wf 1.00 is at order 29 of 37 and pilot (b) at
order 25; no σ₁₈ exists yet. The procedure is section 9.0's one order up, with the changes section 9's findings call
for, stated here before the data:

0. **Dry run of the stages on a synthetic σ₁₈.** Before the real σ₁₈ exists, `planted_k18_dry.py` writes scan files
   whose σ₀…σ₁₇ are the real ones and whose σ₁₈ at each node is the C17-BP [7/10] sum's (ring-free, passing the type
   rule, zero at 1.0000002; section 9.6), with err18 = 10⁻¹² |σ₁₈| and the pilot's errors inf, and every stage runs on
   them (`planted_k18.py STAGE --data DIR`). Pass: every stage completes with its checks passing, and in the ring test
   C17-BP [7/10] has a σ₁₈ error below 10⁻¹⁸ and a τ₁₈ error below 10⁻¹⁷ (the scans' 20-digit format bounds τ at a
   few 10⁻¹⁸, as section 9.0's run note found; the σ criterion is unchanged) and survives; whether its zero lies inside
   the dry run's K = 18 range is reported. A stage that fails on the synthetic data is fixed before the real data, and
   the fix is recorded here. The dry run's records and checks are discarded (its logs go to the raw-data repository);
   it tests the code path, not the estimators (item 2 does that).
1. **σ₁₈ is converged** if, at every node: (a) err18 ≤ 10⁻¹⁰ |σ₁₈|; (b) σ₀…σ₁₇ are bitwise the `scan35J1000` 960-bit
   values; (c) pilot (b)'s σ₁₈ equals (a)'s within (a)'s err18, and its σ₁₅…σ₁₇ equal `scan35J1000`'s within their errors;
   (d) the second difference of σ_k/σ_{k+1} at k = 16 is at most 0.05. Each check gets a mutation that must make it
   fire (`--mutate=err|low|pilot|ratiolaw`).
2. **The pipeline on plants first.** N1 recovered by arg, ratio, Padé and G to 10⁻⁶ at K = 18, N2 giving no number;
   the 28 arc plants, C-EXP and the 13 REFIT plants regenerated and run at K = 16, 17, 18, their K = 16 and 17 estimates
   equal to the stored K = 17 records (`planted_results_k17.jsonl`) to 10⁻⁹, or the work stops. Control: N1 graded
   against ω_t + 0.002 must fail.
3. **Real data at K = 18** (record "N0 scan37J1000"): its K = 16 and 17 estimates must equal those of "N0 scan35J1000".
   Mutation controls: σ₁₈ → −σ₁₈ at wf 1.00 must fire a detector; σ₁₈ × (1 + 10⁻³) is reported.
4. **Ring test.** Every K = 17 completion of section 9.6 is regenerated and its zero must reproduce the stored
   "K17 recomplete" value to 10⁻⁹: C17-W1–3, C17-ARC (four shapes at their stored best ω_t) and C17-BP [m/(17 − m)],
   m = 3…8. No ring survived σ₁₇ (section 9.4), so nothing is carried over from an earlier order; the rings under test
   are this generation's, C17-BP [3/14] and [4/13] (and [5/12], which fails the type rule). Benchmark: the simple-zero
   law from τ₁₆, τ₁₇. **The D-bracket question** is handled as section 9.0 item 4 fixed it: section 5.3's comparison
   is the primary reading (a completion survives if no detector fires on it at K = 18 that does not fire on the real
   data), and the table gives a second reading with D-bracket left out, both fixed here. A ring is a finding only if it
   survives under the first reading; a ring rejected under the first and surviving under the second is reported as
   that. The author's decision asked for in 8.9 has still not come; if it comes before any K = 18 estimate is
   computed, this item is amended with the date. A ring that survives σ₁₈ is a finding, reported as such.
5. **Calibration** at K = 18, as section 9.5: the error of arg, ratio, Padé mid and G on (i) the 28 arc plants, (ii)
   C-EXP, (iii) REFIT with fit residual ≤ 10⁻⁴, and (iv) the BP family, now twelve sums (section 9.5's nine and C17-BP
   [6/11], [7/10], [8/9], the ring-free sums passing the type rule at K = 17), each with its own zero as truth. The
   calibration at K = 17 from the new records must reproduce section 9.5's arc range 0.999990–1.000009 and BP-family
   range 0.999990–1.000000 to 6×10⁻⁷ (against the values stored in `planted_checks_k17.json`).
6. **Next generation and ring-free completions,** glued at σ₁₈ (tail from k = 19): ratio-law windows k = 6–17, 8–17,
   9–17; best-fit arc tails, four shapes, ω_t grid 0.9990–1.0010; the Borel–Padé sums [m/(18 − m)], m = 3…9 (the
   diagonal [9/9] included, as [8/8] was at K = 16). Frozen type rule: every pole has |t| < 2 at all three nodes. Zeros
   counted at wf = 1.00 in |x| < 7.3, 12 and 16.
7. **Combination.** The K = 18 estimator range is the hull of: the calibrated arg range over the 28 arc plants; the
   calibrated arg range over the BP family (twelve); and the zeros of the ring-free completions of item 6 that pass the
   type rule (no zeros besides x₀ in |x| < 12). G is reported with its spread over the minima of its objective within a
   factor 5 of the best and is not part of the hull. The headline is the union of the K = 17 headline
   (0.999975–1.000050) and the K = 18 range if they overlap; if disjoint, both are reported and the reason diagnosed
   before any headline is written. The range is an estimator range, not a bound; the range from the last two orders
   alone is reported beside it as context, as at K = 17.
8. **The order-18 projection.** est_k = ω* + c r^k through k = 16 and 17, for ω* = 1, 0.99999 (the lower end of the
   K = 17 range, 0.999990) and 0.999975 (the lower end of the headline), against the real arg₁₈; decisive only if the
   other prediction is farther from arg₁₈ than three times the model's own error (its largest miss over k = 15–17 on
   the real data, and over the 28 arc plants at k = 18 given their true ω_t).

The stages are `converge`, `plants`, `real`, `ring`, `calib`, `recomplete`, `project`, `combine`, each run as
`nice -n 10 python3 planted_k18.py STAGE` from `solver/inner_amp/`, with the σ₁₈ JSONs copied from the raw-data
repository's T87-mac into `scan37J1000/` and `scan37J1250/` as the earlier orders' were.

The dry run (item 0) ran on 2026-10-28 08:23–08:34 UTC on the Mac, beside T87: all eight stages completed and their 64
checks passed (the logs are in the raw-data repository's `planted-k18-mac/dry/`). In the ring test the planted
C17-BP [7/10] reproduced its σ₁₈ to 1.4×10⁻²⁰ and its τ₁₈ to 3.7×10⁻¹⁸ at worst, within item 0's criteria, and
survived under both readings; its zero (1.0000002) lies inside the dry run's K = 18 range, 0.999992–1.000004. On the
synthetic data D-bracket fires on the "real" series itself (its Padé mid lies above its arg crossing), which says
nothing about the real σ₁₈. No stage code changed.

The real stages ran on 2026-10-28 13:16–13:26 UTC on the Mac, beside T88, from the four T87 scan files committed as
`solver/inner_amp/scan37J1000` and `scan37J1250` (raw-data repository T87-mac at 906916e); the logs are in the
raw-data repository's `planted-k18-mac/` root. The first real K = 18 estimate was computed at 13:22 UTC; no decision on
D-bracket had come, so item 4 stands as written.

### 10.1 σ₁₈ is converged

| wf | σ₁₈ | err18 (relative) | σ₀…σ₁₇ against `scan35J1000` | ratio law: \|Δ²\| at k = 16 |
|---|---|---|---|---|
| 0.99 | −1.8715942756321299754×10⁻¹⁶ + 3.3451545853624606799×10⁻¹⁵i | 6.11×10⁻²⁷ (1.82×10⁻¹²) | bitwise equal, 18/18 | 0.0084 |
| 1.00 | +3.9681809423786068411×10⁻¹⁸ + 3.2762821151945967956×10⁻¹⁵i | 6.69×10⁻²⁷ (2.04×10⁻¹²) | bitwise equal, 18/18 | 0.0084 |
| 1.01 | +1.8701563194321713779×10⁻¹⁶ + 3.1978690994501157227×10⁻¹⁵i | 7.31×10⁻²⁷ (2.28×10⁻¹²) | bitwise equal, 18/18 | 0.0083 |

- **All four criteria of section 10.0 item 1 pass,** and each one's mutation (`--mutate=err|low|pilot|ratiolaw`) makes
  it fire (err18 × 10¹⁴: 204 relative; σ₁₇ at wf 0.99 scaled by 1 + 10⁻¹⁵: 17/18 bitwise; pilot (b)'s σ₁₈ scaled by
  1 + 10⁻¹⁰: 3.3×10⁻²⁵ against 6.7×10⁻²⁷; σ₁₈ × 1.1 at wf 1.01: |Δ²| = 1.14).
  - err18 is 2.3×10⁻¹² relative at worst, against the 10⁻¹⁰ asked for.
  - Pilot (b) at wf 1.00 (J 1250, 1280 bits) equals (a)'s σ₁₈ to 4.5×10⁻²⁸ (err18 6.7×10⁻²⁷). Its σ₁₅, σ₁₆, σ₁₇ equal
    `scan35J1000`'s to 2.7×10⁻²⁶, 2.7×10⁻²⁶ and 1.6×10⁻²⁷, 110, 15 and 34 times below their errors. Its own sigma_err
    is inf at every k (family 1_1's order-1 fit), as at K = 16 and 17.
  - The second differences of σ_k/σ_{k+1} are 0.008–0.012 at k = 10–16 (0.002–0.048 over k = 7–16).
- **σ₁₈ is nearly imaginary at ω∞** (its real part 1.2×10⁻³ of its modulus), where the ratio law puts it: arg σ_k
  advances by 0.77 per order (σ₁₇/σ₁₈ = 8.96 − 8.68i); the nodes' real parts are ∓1.9×10⁻¹⁶.
- **Spread in wf.** σ₁₈ changes by 6.1% of itself between the nodes (5.9%, 6.0% for σ₁₆, σ₁₇). err18 is 3.6×10⁻¹¹ of
  that change.
- **Cost.** On the Mac with two processes per run: pilot (a) 180 min, pilot (b) 320 min, the nodes 195 min each, the
  late-order fits taking 84% of the time (comms, 2026-10-28 06:59–13:14 UTC).

### 10.2 The pipeline on plants first

51 plants at K = 16, 17, 18, before the real data. All four checks pass:
- the regenerated plants' K = 16 and 17 estimates equal the stored K = 17 records exactly (both runs on the Mac);
- N1, 7/7: a lone simple zero recovered at K = 18 by arg, ratio, Padé and G to 3.5×10⁻⁷ at worst; graded against
  ω_t + 0.002 it fails, as it must;
- N2, 2/2: no estimator prints a number.

### 10.3 The real estimates at K = 18

The record "N0 scan37J1000" equals "N0 scan35J1000" at K = 16 and 17 exactly.

| estimator | K = 18 | K = 17 | K = 16 |
|---|---|---|---|
| arg crossing | 1.0000024 | 1.0000046 | 1.000010 |
| ratio | 0.999966 | 0.999912 | 0.999882 |
| central Padé | [3/14]–[13/4]: 0.999997–1.000002 (mid 0.9999995) | [3/13]–[12/4]: 0.999978–1.000002 (mid 0.999990) | 0.999992–1.000009 (mid 1.0000007) |
| G (production κ̄) | 0.9999987 | 1.0000026 | 1.000005 |
| Aitken, k − 2 … k | 1.0000011 | 0.9999925 | 1.000060 |
| effective exponent c at ω∞ | 1.000 − 0.001i | 1.000 − 0.002i | 1.002 − 0.002i |

- **The arg crossings keep falling monotonically:** 1.000046, 1.000026, 1.000019, 1.000010, 1.0000046, 1.0000024 for
  k = 13–18. The distance from 1 shrank by ×0.44 from k = 16 to 17 and by ×0.52 from 17 to 18.
- **G crossed below 1 for the first time** (0.9999987, from 1.0000026), 1.3×10⁻⁶ under it, with the Padé mid
  (0.9999995) and every central approximant from [3/14] to [13/4] within 3×10⁻⁶ of 1: the central Padé range,
  0.999997–1.000002, is the narrowest at any order, and the [3/K−3] approximant that widened it downward at K = 17
  (0.999978) sits at 0.999999 now.
- **Aitken is back above 1** (1.0000011; the last two differences of arg_k are 5.8×10⁻⁶, then 2.2×10⁻⁶). Context, not
  part of any range.
- **Detectors:** D-ratio and D-ratiolaw, as at K = 16 and 17. **D-bracket is silent again:** G (0.9999987) and the
  Padé mid (0.9999995) both lie below the arg crossing (1.0000024).
- **The mutation controls fire.** σ₁₈ → −σ₁₈ at wf 1.00 fires D-G and D-bracket; σ₁₈ × (1 + 10⁻³) fires D-G, D-arg
  and D-bracket. Each moves the estimates by about 10⁻².

### 10.4 The ring test: what each K = 17 completion predicted for σ₁₈

Every completion was regenerated first and reproduces its stored zero to the printed digits, and its K = 17 (or
K = 16, 17) estimates equal the stored records exactly. The benchmark, the simple-zero law from τ₁₆, τ₁₇, misses τ₁₈
by 5.3–6.3×10⁻⁵. The last two columns are the two readings of section 10.0 item 4.

| completion (K = 17) | its zero | zeros in \|x\| < 7.3 / 12 / 16 | σ₁₈ rel. error | τ₁₈ rel. error | its arg₁₈ | detectors beyond the real data's | survives | without D-bracket |
|---|---|---|---|---|---|---|---|---|
| C17-W1, W2, W3 (ratio law) | 1.00068–1.00090 | 7 / 16 / – | 5.9–8.7×10⁻³ | 1.4–2.2 | none | D-G, D-arg, D-bracket | no | no |
| C17-ARC, four shapes | 0.99995–1.00019 | 2–5 / 14 / – | 3.5×10⁻⁴–1.6×10⁻³ | 0.10–0.39 | 0.979–0.9998 | D-G, D-bracket | no | no |
| **C17-BP [3/14]** (ring of 9.6) | 0.999998 | 0 / 9 / 15 | 3.9–4.3×10⁻⁶ | 0.9–1.2×10⁻³ | 0.999946 | D-G, D-bracket | **no** | **no** |
| **C17-BP [4/13]** (ring of 9.6) | 0.999998 | 0 / 6 / 12 | 6.8–8.6×10⁻⁷ | 1.9–2.1×10⁻⁴ | 0.999994 | D-G | **no** | **no** |
| C17-BP [5/12] (fails the type rule) | 1.000004 | 0 / 3 / 13 | 3.2–6.4×10⁻⁷ | 0.9–1.5×10⁻⁴ | 1.000005 | D-arg, D-bracket | no | no |
| C17-BP [6/11] | 1.0000004 | 0 / 1 / 10 | 5.5–5.6×10⁻⁸ | 1.3–1.5×10⁻⁵ | 1.000003 | D-bracket | no | yes |
| C17-BP [7/10] | 1.0000002 | 0 / 1 / 7 | 1.8–2.6×10⁻⁸ | 5.0–6.1×10⁻⁶ | 1.000003 | D-bracket | no | yes |
| C17-BP [8/9] | 0.9999999 | 0 / 1 / 6 | 4.3–7.1×10⁻⁹ | 1.2–1.7×10⁻⁶ | 1.000002 | D-bracket | no | yes |
| C16-BP [6/10] | 1.0000005 | 0 / 1 / 9 | 5.8–6.1×10⁻⁷ | 2.6–2.9×10⁻⁵ | 1.000003 | – | yes | yes |
| C16-BP [7/9] | 1.0000007 | 0 / 1 / 5 | 2.4–2.9×10⁻⁷ | 1.5–1.7×10⁻⁵ | 1.000003 | – | yes | yes |
| C16-BP [8/8] | 1.0000007 | 0 / 1 / 5 | 2.4–2.9×10⁻⁷ | 1.5–1.7×10⁻⁵ | 1.000003 | – | yes | yes |
| C15-BP [6/9] | 0.999999 | 0 / 1 / 11 | 4.9–5.7×10⁻⁶ | 3.9–4.4×10⁻⁵ | 1.000002 | D-bracket | no | yes |
| C15-BP [7/8] | 1.000002 | 0 / 1 / 5 | 1.5×10⁻⁶ | 3.2–3.5×10⁻⁵ | 1.000004 | – | yes | yes |
| C14-BP [5/9] | 0.999983 | 0 / 1 / 8 | 6.5–8.1×10⁻⁵ | 3.6–4.4×10⁻⁴ | 0.999981 | D-arg, D-bracket, D-padé | no | no |
| C14-BP [6/8] | 0.999993 | 0 / 1 / 7 | 2.8–3.1×10⁻⁵ | 1.3–1.4×10⁻⁴ | 0.999997 | – | yes | yes |
| C13-BP [5/8] | 0.999957 | – / 1 / – | 2.2–3.8×10⁻⁴ | 0.7–1.4×10⁻³ | 0.999967 | D-arg, D-bracket | no | no |
| C13-BP [6/7] | 0.999982 | – / 1 / – | 0.95–1.1×10⁻⁴ | 3.4–3.7×10⁻⁴ | 0.999986 | – | yes | yes |

- **Every ratio-law and arc ring is rejected,** as at K = 14–17: they miss τ₁₈ by 10–220% and fire D-G and D-bracket
  (D-arg too for the ratio-law tails).
- **The two rings of section 9.6, C17-BP [3/14] and [4/13], are rejected by σ₁₈,** under both readings. [3/14]
  mispredicts σ₁₈ by 4×10⁻⁶ and τ₁₈ by fifteen to twenty times the benchmark and fires D-G and D-bracket; [4/13]
  mispredicts σ₁₈ by 7–9×10⁻⁷ and τ₁₈ by three to four times the benchmark and fires D-G. [5/12], which already
  failed the type rule, fires D-arg and D-bracket. So for the second order running no ring of the latest generation
  survives its first test, and σ₀…σ₁₈ exclude every ring built so far.
- **The ring-free sums of σ₀…σ₁₇ predicted σ₁₈ to 4×10⁻⁹–6×10⁻⁸** ([8/9] 4.3–7.1×10⁻⁹, the best prediction at any
  order; [7/10] 1.8–2.6×10⁻⁸; [6/11] 5.5×10⁻⁸) and τ₁₈ to 1.2×10⁻⁶–1.5×10⁻⁵, four to fifty times better than the
  benchmark. **All three are rejected by D-bracket alone** under section 5.3's rule and survive without it; the
  detector again removes the best predictors in the table, as it did at K = 16 ([7/8]) and K = 17 ([7/9], [8/8]).
- **D-bracket's verdict on a ring-free sum is not stable from one order to the next:** C16-BP [7/9] and [8/8], which
  it rejected at K = 17, pass at K = 18 (their arg₁₈ 1.000003, no detector), while C15-BP [6/9] stays rejected by it
  at both orders. The survivors under section 5.3's rule are C16-BP [6/10], [7/9], [8/8], C15-BP [7/8], C14-BP [6/8]
  and C13-BP [6/7]; without D-bracket, those and C17-BP [6/11], [7/10], [8/9] and C15-BP [6/9]. All are ring-free,
  with zeros at 0.999982–1.000002, and none moves the estimate.
- C14-BP [5/9] fires D-padé as well now, the first time that detector fires at K = 18.

### 10.5 Calibration at K = 18

The implied zero for the real K = 18 estimate is real − error over each family:

| family | n | arg | ratio | Padé mid | G |
|---|---|---|---|---|---|
| the 28 arc plants | 28 | **0.999992–1.000002** | 0.999884–1.000045 | 0.999993–1.000000 | 0.999999 |
| C-EXP | 1 | 1.000005 | 0.999956 | 1.000012 | 0.999997 |
| REFIT (fit residual ≤ 10⁻⁴) | 5 | 0.999995–0.999997 | 0.999997–1.000060 | 0.999998–1.000000 | 0.999999 |
| BP family (twelve sums, section 10.0 item 5) | 12 | **0.999992–1.000004** | 0.999968–1.000244 | 0.999965–1.000754 | 0.999986–1.000002 |

- The K = 17 calibration rebuilt from the new records reproduces section 9.5 exactly (arc 0.999990–1.000009, BP family
  of nine 0.999990–1.000000).
- **The arc plants' arg error shrank again and stopped changing sign:** +0.00…+0.10×10⁻⁴ at K = 18 against
  −0.04…+0.15×10⁻⁴ at K = 17 (the mirror shapes, low before, are at +0.00…+0.01). The twelve-sum BP family's arg
  range equals the nine-sum's: the three sums added (C17-BP [6/11], [7/10], [8/9]) lie inside it.
- G on the arc plants is within 5×10⁻⁷ of their zeros (its implied zero 0.999999 on all 28).
- The Padé mid's high bias on the mirror shapes is gone (its error −0.01…+0.06×10⁻⁴ over the 28, +0.5–0.6×10⁻⁴ at
  K = 17), and Aitken on the mirror plants no longer runs away (−0.04…+0.17×10⁻⁴ over the 28). Neither is part of the
  hull.
- The plants' σ₁₈ deviate from the measured one by 5×10⁻³–1.2×10⁻¹ (C-EXP 2.5×10⁻³): calibration shapes, not
  completions.

### 10.6 The next generation and the ring-free completions at K = 18

Glued at σ₁₈ (tails from k = 19); zeros counted at wf 1.00:

| completion (K = 18) | its zero | zeros in \|x\| < 7.3 / 12 / 16 | type rule |
|---|---|---|---|
| C18-W1, W2, W3 (ratio law, windows 6–17, 8–17, 9–17) | 1.00037–1.00052 | 7–8 / 17 / – | – |
| C18-ARC, four shapes (best ω_t 1.00022–1.00031, L2 floor 0.9–1.7×10⁻³) | 0.99993–1.00009 | 3–5 / 15 / – | – |
| C18-BP [3/15] | 0.9999986 | 0 / 9 / 13 | passes (largest pole \|t\| = 1.67) |
| C18-BP [4/14] | 0.9999997 | 0 / 6 / 13 | passes (1.71) |
| C18-BP [5/13] | 1.0000004 | 0 / 2 / 11 | passes (1.70) |
| **C18-BP [6/12]** | **1.0000002** | **0 / 1 / 11** | passes (1.73) |
| **C18-BP [7/11]** | **1.0000000** | **0 / 1 / 7** | passes (1.83) |
| **C18-BP [8/10]** | **1.0000000** | **0 / 1 / 5** | passes (1.76) |
| **C18-BP [9/9]** | **1.0000000** | **0 / 1 / 5** | passes (1.76) |

The ring-free completions that pass the type rule are C18-BP [6/12], [7/11], [8/10] and [9/9], at
0.99999997–1.0000002 ([7/11] 0.99999997, [8/10] and [9/9] 0.99999998, [6/12] 1.0000002), the tightest cluster so far
(K = 17: 0.9999999–1.0000004), again straddling 1. Every Borel–Padé sum passes the type rule this time ([5/12] failed
it at K = 17). The ratio-law and arc tails carry rings again, and the sums [3/15], [4/14] and [5/13] carry one to eight
zeros beyond x₀ in |x| < 12 with their own zero at 0.9999986–1.0000004; σ₁₉ (T88) tests them.

### 10.7 The order-18 projection

The model est_k = ω* + c r^k through k = 16 and 17 predicts arg₁₈ − 1 = +2.01×10⁻⁶ for ω* = 1, +0.42×10⁻⁶ for
0.99999 and −0.28×10⁻⁶ for 0.999975; the real value is +2.41×10⁻⁶, 0.4×10⁻⁶ from the first, 2.0×10⁻⁶ from the second
and 2.7×10⁻⁶ from the third. The model's own error is 1.1×10⁻⁵ (its largest miss over k = 15–17 on the real data
0.1–0.7×10⁻⁵ depending on ω*, and over the arc plants at k = 18, −0.3…+1.1×10⁻⁵), so the projection favours ω* = 1 but
is not decisive, as at K = 15–17. The log-linear trend of arg_k − 1 over k = 4–18 falls ×0.539 per order (×0.539 over
k = 4–17) and puts arg₁₈ − 1 at 2.5×10⁻⁶, where it is. Under noise of size sigma_err on every σ_k (20 seeds), arg₁₈
moves by at most 5.1×10⁻⁹.

### 10.8 Combination

- K = 18 estimator range (hull): the calibrated arg over the arc plants, 0.999992–1.000002; over the BP family of
  twelve, 0.999992–1.000004; the ring-free completions, 0.99999997–1.0000002. **The hull is 0.999992–1.000004.**
- It overlaps the K = 17 headline (0.999975–1.000050), so by section 10.0 item 7 the headline is their union,
  **0.999975–1.000050, unchanged.** The K = 18 range lies inside the K = 17 one (0.999990–1.000009) at both ends;
  the range from the last two orders alone, reported as context by item 7, is 0.999990–1.000009, and K = 16–18
  would give 0.999980–1.000014. An estimator range, not a bound.
- G, with its spread: production 0.9999987 at K = 18 (1.0000026 at K = 17); over the minima of its objective within a
  factor 5 of the best, 0.999999–1.000010 (0.999908–1.000003 at K = 17); its global minimum 1.000010 (1.0000001).

### 10.9 Verdict at K = 18

- σ₁₈ is converged to 2.3×10⁻¹² relative and continues every trend of σ₀…σ₁₇.
- The real estimates moved toward 1 again, and for the first time two of them crossed below it: arg 1.0000024,
  G 0.9999987, central Padé [3/14]–[13/4] at 0.999997–1.000002; the ring-free sums of σ₀…σ₁₈ sit at
  0.99999997–1.0000002, on both sides of 1.
- The calibrated range shrank from 0.999990–1.000009 (arc plants, arg, K = 17) to 0.999992–1.000002, and the K = 18
  hull is 0.999992–1.000004, a width of 1.2×10⁻⁵ against 1.9×10⁻⁵ at K = 17 and 3.4×10⁻⁵ at K = 16. The headline
  union stays 0.999975–1.000050 because the procedure keeps the earlier orders' ranges; the last two orders alone give
  0.999990–1.000009, and K = 16–18 would give 0.999980–1.000014.
- ω∞ = 1 lies inside every range, as at K = 13–17. The order-18 projection puts ω* = 1 closest to arg₁₈ (0.4×10⁻⁶,
  against 2.0×10⁻⁶ for a zero 10⁻⁵ below 1), but the model's own error, 1.1×10⁻⁵, is larger than both: the data still
  do not decide between 1 and a value 10⁻⁵ below it.
- The two rings of the K = 17 generation, C17-BP [3/14] and [4/13], are rejected by σ₁₈ under both readings, as every
  ring of the K = 16 generation was by σ₁₇. So σ₀…σ₁₈ exclude every ring built so far, nearer than x₀ or just beyond
  it, from any number of coefficients up to seventeen. The next generation (C18-BP [3/15], [4/14], [5/13], zeros
  0.9999986–1.0000004) is tested by σ₁₉ (T88, running).
- D-bracket: the two readings of 10.0 item 4 agree on every ring, as at K = 17. On its own it rejected only ring-free
  sums, this time all three of the K = 17 generation, the best predictors of σ₁₈ (4×10⁻⁹–6×10⁻⁸), and its verdict on
  C16-BP [7/9] and [8/8] reversed from K = 17 to K = 18. The rule stays as item 4 fixed it unless the author decides
  otherwise before σ₁₉; it changed no finding at K = 18.

## 11. Follow-up: σ₁₉ (post-freeze)

### 11.0 Procedure, fixed before any K = 19 estimate was computed

Written and committed before any estimator ran at K = 19, on any input. Already known at this point (the Mac's T88,
started 2026-10-28 13:13 UTC; its logs at 16:18 UTC): pilot (a) at wf 1.00 is at order 37 of 39 and pilot (b) at
order 33; no σ₁₉ exists yet. The procedure is section 10.0's one order up, with the changes section 10's findings call
for, stated here before the data:

0. **Dry run of the stages on a synthetic σ₁₉.** Before the real σ₁₉ exists, `planted_k19_dry.py` writes scan files
   whose σ₀…σ₁₈ are the real ones and whose σ₁₉ at each node is the C18-BP [7/11] sum's (ring-free, passing the type
   rule, zero at 0.99999997; section 10.6), with err19 = 10⁻¹² |σ₁₉| and the pilot's errors inf, and every stage runs on
   them (`planted_k19.py STAGE --data DIR`). Pass: every stage completes with its checks passing, and in the ring test
   C18-BP [7/11] has a σ₁₉ error below 10⁻¹⁸ and a τ₁₉ error below 10⁻¹⁷ and survives; whether its zero lies inside
   the dry run's K = 19 range is reported. A stage that fails on the synthetic data is fixed before the real data, and
   the fix is recorded here. The dry run's records and checks are discarded (its logs go to the raw-data repository);
   it tests the code path, not the estimators (item 2 does that).
1. **σ₁₉ is converged** if, at every node: (a) err19 ≤ 10⁻¹⁰ |σ₁₉|; (b) σ₀…σ₁₈ are bitwise the `scan37J1000` 960-bit
   values; (c) pilot (b)'s σ₁₉ equals (a)'s within (a)'s err19, and its σ₁₆…σ₁₈ equal `scan37J1000`'s within their errors;
   (d) the second difference of σ_k/σ_{k+1} at k = 17 is at most 0.05. Each check gets a mutation that must make it
   fire (`--mutate=err|low|pilot|ratiolaw`).
2. **The pipeline on plants first.** N1 recovered by arg, ratio, Padé and G to 10⁻⁶ at K = 19, N2 giving no number;
   the 28 arc plants, C-EXP and the 13 REFIT plants regenerated and run at K = 17, 18, 19, their K = 17 and 18 estimates
   equal to the stored K = 18 records (`planted_results_k18.jsonl`) to 10⁻⁹, or the work stops. Control: N1 graded
   against ω_t + 0.002 must fail.
3. **Real data at K = 19** (record "N0 scan39J1000"): its K = 17 and 18 estimates must equal those of "N0 scan37J1000".
   Mutation controls: σ₁₉ → −σ₁₉ at wf 1.00 must fire a detector; σ₁₉ × (1 + 10⁻³) is reported.
4. **Ring test.** Every K = 18 completion of section 10.6 is regenerated and its zero must reproduce the stored
   "K18 recomplete" value to 10⁻⁹: C18-W1–3, C18-ARC (four shapes at their stored best ω_t) and C18-BP [m/(18 − m)],
   m = 3…9. No ring survived σ₁₈ (section 10.4), so nothing is carried over from an earlier order; the rings under test
   are this generation's, C18-BP [3/15], [4/14] and [5/13], all three passing the type rule this time. Benchmark: the
   simple-zero law from τ₁₇, τ₁₈. **The D-bracket question** is handled as section 9.0 item 4 fixed it: section 5.3's
   comparison is the primary reading (a completion survives if no detector fires on it at K = 19 that does not fire on
   the real data), and the table gives a second reading with D-bracket left out, both fixed here. A ring is a finding
   only if it survives under the first reading; a ring rejected under the first and surviving under the second is
   reported as that. The author's decision asked for in 8.9 has still not come; if it comes before any K = 19 estimate
   is computed, this item is amended with the date. A ring that survives σ₁₉ is a finding, reported as such.
5. **Calibration** at K = 19, as section 10.5: the error of arg, ratio, Padé mid and G on (i) the 28 arc plants, (ii)
   C-EXP, (iii) REFIT with fit residual ≤ 10⁻⁴, and (iv) the BP family, now sixteen sums (section 10.5's twelve and
   C18-BP [6/12], [7/11], [8/10], [9/9], the ring-free sums passing the type rule at K = 18), each with its own zero as
   truth. The calibration at K = 18 from the new records must reproduce section 10.5's arc range 0.999992–1.000002 and
   BP-family range 0.999992–1.000004 to 6×10⁻⁷ (against the values stored in `planted_checks_k18.json`).
6. **Next generation and ring-free completions,** glued at σ₁₉ (tail from k = 20): ratio-law windows k = 6–18, 8–18,
   9–18; best-fit arc tails, four shapes, ω_t grid 0.9990–1.0010; the Borel–Padé sums [m/(19 − m)], m = 3…9 (up to
   [9/10], as [8/9] was at K = 17). Frozen type rule: every pole has |t| < 2 at all three nodes. Zeros counted at
   wf = 1.00 in |x| < 7.3, 12 and 16.
7. **Combination.** The K = 19 estimator range is the hull of: the calibrated arg range over the 28 arc plants; the
   calibrated arg range over the BP family (sixteen); and the zeros of the ring-free completions of item 6 that pass the
   type rule (no zeros besides x₀ in |x| < 12). G is reported with its spread over the minima of its objective within a
   factor 5 of the best and is not part of the hull. The headline is the union of the K = 18 headline
   (0.999975–1.000050) and the K = 19 range if they overlap; if disjoint, both are reported and the reason diagnosed
   before any headline is written. The range is an estimator range, not a bound; the range from the last two orders
   alone is reported beside it as context, as at K = 17 and 18.
8. **The order-19 projection.** est_k = ω* + c r^k through k = 17 and 18, for ω* = 1, 0.999992 (the lower end of the
   K = 18 range) and 0.999975 (the lower end of the headline), against the real arg₁₉; decisive only if the other
   prediction is farther from arg₁₉ than three times the model's own error (its largest miss over k = 16–18 on the
   real data, and over the 28 arc plants at k = 19 given their true ω_t).

The stages are `converge`, `plants`, `real`, `ring`, `calib`, `recomplete`, `project`, `combine`, each run as
`nice -n 10 python3 planted_k19.py STAGE` from `solver/inner_amp/`, with the σ₁₉ JSONs copied from the raw-data
repository's T88-mac into `scan39J1000/` and `scan39J1250/` as the earlier orders' were. `planted_k19.py` is
`planted_k18.py` one order up (a mechanical renumbering) with the changes items 4–8 state: the ring test regenerates
the C18 completions, the calibration family has sixteen sums, the projection's second hypothesis is 0.999992.

The dry run (item 0) ran on 2026-10-28 16:26–16:41 UTC on the Mac, beside T88: all eight stages completed and their 72
checks passed (the logs are in the raw-data repository's `planted-k19-mac/dry/`). In the ring test the planted
C18-BP [7/11] reproduced its σ₁₉ to 2.4×10⁻²⁰ and its τ₁₉ to 3.5×10⁻¹⁸ at worst, within item 0's criteria, and
survived under both readings; its zero (0.99999997) lies inside the dry run's K = 19 range, 0.999996–1.000004. On the
synthetic data D-ratio and D-ratiolaw fire on the "real" series itself, which says nothing about the real σ₁₉;
C19-BP [9/10] fails the type rule there (largest pole |t| = 4.6) and [3/16], [4/15], [5/14] carry rings, so the
ring-free list was [6/13], [7/12], [8/11]. One stage was fixed: `combine` read the twelve-sum family's calibrated arg
range where item 7 asks for the sixteen-sum family's (the renumbering had not advanced the label); on the synthetic
data the two ranges coincide (0.999996–1.000000), so the dry run's hull did not change, and the stage was rerun after
the fix (16:42 UTC; `dry/combine_before_fix.log` is the first run). No other stage code changed.

**The pilot gate fired on the real σ₁₉ (2026-10-28 20:55 UTC).** Item 1(c) fails at wf 1.00: pilot (b) (J 1250, 1280
bits) differs from (a) (J 1000, 960 bits) by 3.39×10⁻²⁷ at σ₁₉ against (a)'s err19 of 2.79×10⁻²⁷, a ratio of 1.21
(`solver/inner_amp/pilot_agreement.py`, on the files in `scan39J1000/` and `scan39J1250/`); σ₁₆…σ₁₈ pass as at K = 18.
The difference is 1.3×10⁻¹¹ of |σ₁₉|, nothing to the estimators of items 2–8 (they need about 10⁻⁶), but the control
fired and the stages do not run until its cause is known. What the files say: (b)'s own plateau error at σ₁₉, read from
its `fit_info` with the order-1 term set to zero, is 1.2×10⁻²⁸ (its `sigma_err` is inf because family 1_1's residual
underflows to 0.0 at 1280 bits and `pick_plateau` reads a non-falling residual as no plateau; `scan29J1000`'s 1280-bit
file has it at J 1000, so it is the bits, not J 1250), so the two fits' own errors do not cover their difference, and
at least one plateau error omits something, most likely the J truncation: over k = 14…19 the (a)/(b) relative
difference went 3×10⁻¹⁵, 5×10⁻¹⁵, 5×10⁻¹⁴, 4×10⁻¹⁴, 1.4×10⁻¹³, 1.3×10⁻¹¹ while (a)'s relative plateau error went
3×10⁻¹³, 5×10⁻¹³, 8×10⁻¹³, 1.3×10⁻¹², 2.0×10⁻¹², 1.1×10⁻¹¹, and they met at k = 19; (a)'s fits all sit at Mf 7 from
k = 33 up (20 of 20 at k = 39), (b)'s at Mf 8 (14 of 20). The attribution runs as T91 on the Mac, a third pilot at
J 1250 and 960 bits ((a)'s bits with (b)'s J, so its differences to (a) and to (b) are the J effect and the bits
effect), started 2026-10-28 21:06 UTC. Item 1 is amended for the cause once T91 lands, the amendment dated and
committed before any K = 19 estimate; the `converge` stage runs as written on the three nodes first, and its failure
is kept in the log.

**Amendment of item 1 (2026-10-29 02:12 UTC), after T91 landed at 02:01 UTC.** Pilot (d), J 1250 at 960 bits
(`scan39J1250p960/`, from the raw-data repository's T91-mac at 4ce5097), puts σ₁₉ at 1.85×10⁻²⁷ from (a) and
1.94×10⁻²⁷ from (b), with its own plateau error 1.85×10⁻²⁷ (`pilot_agreement.py 39`): the J truncation and the
precision each move σ₁₉ by about two thirds of (a)'s plateau error, and the two add to the 3.39×10⁻²⁷ of the gate.
Neither is the S_1 family's plateau step, which is what `sigma_err` measures. The route is in the files: at 1280 bits
the fit residuals still fall at Mf 8, so at order 33 all 16 neighbour families pick Mf 8 where the 960-bit runs pick
Mf 7, and their tuning constants, added exactly into the later orders, differ by 1.9×10⁻⁵ of themselves at order 33
and 3.1×10⁻² at order 37. So (d) and (b), the same J, agree exactly at σ₁₂…σ₁₅ and to 2×10⁻³³ and 4×10⁻³³ at σ₁₇
and σ₁₈, differ at σ₁₆ by 2.94×10⁻²⁶, exactly (b)'s plateau step there (its S_1 family at Mf 8 against (d)'s at Mf 7),
and differ at σ₁₉ by 1.94×10⁻²⁷ with both S_1 fits at Mf 7. The J effect takes the same route through the J
truncation: the neighbour constants of (a) and (d) differ by 77% of themselves at order 33 and by more above. The
record above guessed the J truncation alone; it is half.

Item 1 is amended as follows, before any K = 19 estimate, and the sum below was chosen before the two differences it
needs at wf 0.99 and 1.01 were known:

- (a′) err19 at each node is the node's plateau error plus the node's pilot difference, |σ₁₉ at J 1000 and 960 bits
  − σ₁₉ at J 1250 and 1280 bits| at the same wf. At wf 1.00 the difference is pilot (b)'s, 3.39×10⁻²⁷, and
  err19 = 6.17×10⁻²⁷, 2.45×10⁻¹¹ of |σ₁₉|, passing 1(a) by 4.1. At wf 0.99 and 1.01 the difference comes from the σ₁₉
  of T89's node runs (K 41, J 1250, 1280 bits, started once T89's pilot gate passes; σ₁₉ does not depend on K), which
  land about 2026-10-30. Every later stage that reads σ₁₉'s error reads this one.
- (c) stays as written, recorded as failed in section 11.1, and is not evaluated again; (b) and (d) are unchanged.
- The K = 19 stages run when T89's nodes have landed. If (a′) fails at a node (the 0.99 node has a margin of 1.2 on
  its plateau error alone, and its pilot difference is not known), the work stops as the planted protocol says, and
  the fallback named now is a second dated amendment that takes T89's J 1250 / 1280-bit files as the K = 19 node data,
  with the error there the plateau error plus the pilot difference against J 1500 at wf 1.00.
- The `sigma_err` of a 1280-bit file is inf (the record above); `sigma_err_fit.py FILE --a1-from FILE2` recomputes it
  from `fit_info` with the order-1 term set to zero, the dropped term bounded at 10⁻²⁹⁰, and |S_1^(1)| from a 960-bit
  file at the same wf; its selftest reproduces every finite file's `sigma_err` to 10⁻¹² and its mutation fires. The
  `pick_plateau` change that would store a finite value is deferred past T89, so that every file of the K = 20
  campaign comes from the same code.

**Second amendment of item 1 (2026-10-31 04:32 UTC), after T89's nodes landed at 03:58 UTC.** The `converge` stage ran
under (a′) at 04:23 UTC (log `converge_amend.log` in the raw-data repository's `planted-k19-mac/real/`; the record
"K19 converge under (a')" in `planted_checks_k19.json`): (a′) fails at wf 0.99, where err19 = 2.07×10⁻²⁶ (the 960-bit
plateau error) + 2.02×10⁻²⁶ (the pilot difference) = 4.09×10⁻²⁶ is 1.59×10⁻¹⁰ of |σ₁₉|, a margin of 0.63; it passes at
wf 1.00 (2.45×10⁻¹¹) and 1.01 (5.75×10⁻¹²), and (b) and (d) pass at every node. The work stopped there, and the fallback
named above is taken as it was named:

- The K = 19 node data are T89's K 41, J 1250, 1280-bit files at wf 0.99, 1.00 and 1.01 (`scan41J1250/`, from the
  raw-data repository's T89-mac), σ₀…σ₁₉; their σ₂₀ belongs to section 12. The files' stored `sigma_err` is inf, so
  every error of the node data is the plateau error `sigma_err_fit.py` recomputes from `fit_info` (a1 from the 960-bit
  `scan39J1000` file at the same wf). The real record is "N0 scan41J1250".
- (a″) err19 at each node is that plateau error plus the pilot difference against J 1500 at wf 1.00,
  |σ₁₉(J 1250) − σ₁₉(J 1500)| from `scan41J1250/` and `scan41J1500/`, added at every node. It is the only J difference
  measured at 1280 bits; the fallback takes it for the two other nodes too, an assumption named with the fallback,
  before the data. The values are recorded in section 11.2 from the stage's log.
- (b″) at wf 1.00, σ₀…σ₁₉ are bitwise T88's pilot (b) (K 39, J 1250, 1280 bits, `scan39J1250/`), the one other file at
  this J and precision (T89's pilot gate found 20 of 20; the mutation `low` alters one of them and must fire). At wf
  0.99 and 1.01 there is no second file at J 1250 and 1280 bits, and the differences of σ₀…σ₁₈ from the 960-bit
  `scan39J1000` values are recorded with their ratio to that file's plateau error and the Mf picks behind them; they
  are not a check, because a threshold set now would be set after the data. What the files show, read before this was
  written: the ratio is 1.00, 0.99 and 1.17 at σ₁₁ (wf 0.99, 1.00, 1.01) and 0.01, 0.01 and 1.01 at σ₈, every other
  one at most 0.74; at order 23 (σ₁₁) seven or eight of the twelve neighbour families, and at order 17 (σ₈) all nine,
  pick Mf one higher at 1280 bits, the route the first amendment traced at σ₁₆ and σ₁₉. The same comparison is section
  12.0's item 1(b) at those nodes, which therefore fires; its amendment is at the end of section 12.0.
- (c) stays not evaluated; (d) is unchanged, read on the node data.
- Items 2–8 run on the node data; item 3's record is "N0 scan41J1250", and its K = 17, 18 estimates must equal
  "N0 scan37J1000"'s to 10⁻⁹ as written (the one comparison of the 1280-bit node data with a 960-bit computation that
  remains a check). The script is `planted_k19.py STAGE --nodes scan41J1250` (the `--amend` run stays as the record of
  the stop); mutations `err|low|pilot|ratiolaw`, `pilot` scaling the J 1500 file's σ₁₉ by 1 + 10⁻⁹.

**Third amendment, item 3 (2026-10-31 05:01 UTC), after the `real` stage fired under the second amendment.** The stage ran
at 04:51 UTC (`nodes/real.log`): the node data's K = 17 and 18 estimates differ from the 960-bit record
"N0 scan37J1000" by up to 9.3×10⁻⁹ (K = 17, the Padé crossing `pade10/6`) and 3.3×10⁻⁸ (K = 18, `pade11/6`) against
the 10⁻⁹ written for a same-data reproduction; the arg crossings move by at most 1.6×10⁻¹⁰, the ratio estimates by
5.0×10⁻⁹, G by 1.1×10⁻⁹. The cause, found before this was written (`noise_test.py` and its log in the raw-data
repository's `planted-k19-mac/nodes/`): the two data sets' σ₀…σ₁₈ differ by at most 1.7×10⁻¹³ of |σ_k| (section 11.2)
and the estimators amplify that. The 960-bit data with Gaussian noise of each σ_k's own plateau error (k = 1…18,
20 seeds, through the same harness) move those two Padé crossings by an rms of 1.7×10⁻⁷ and 9.3×10⁻⁷, ratio₁₇ and
ratio₁₈ by 5.4×10⁻⁸ and 8.3×10⁻⁸, arg₁₇ and arg₁₈ by 2.1×10⁻⁹ and 2.8×10⁻⁹; every observed difference is at most 0.61
of its rms (arg₁₁; the Padé crossings 0.01–0.07), and the unperturbed 960-bit data reproduce the stored record exactly.
Item 3 is amended as follows, before any K = 19 estimate is used:

- "N0 scan41J1250" is compared with "N0 scan37J1000" at K = 17 and 18 as before, and each estimate must agree within
  three times the rms of that noise reference (the same 20 seeds, which the stage writes to `planted_noise_k19.jsonl`,
  with the unperturbed 960-bit data as the control, which must reproduce the stored record to 10⁻⁹). The mutation
  `--mutate=real` scales σ₁₈ at wf 1.00 of the node data by 1 + 10⁻⁹ and must fire. The 10⁻⁹ reproduction reading
  stays for same-data comparisons (section 12.0's item 3 as amended). Item 3's σ₁₉ mutation controls are unchanged.

**Fourth amendment, item 4 (2026-10-31 05:09 UTC), after the `ring` stage fired under the second amendment.** The `real` stage
passed under the third amendment at 05:04 UTC (every K = 17, 18 estimate within 0.61 rms of the noise reference, the
control exact; `--mutate=real` fires). The `ring` stage then failed its reproduction check for the K = 17
and 18 generations' completions (C18-W1–3, C18-ARC, C18-BP, C17-BP), "its K = 18 estimates are the real ones", by the
same 3.3×10⁻⁸ (9.3×10⁻⁹ at K = 17): those completions carry the 960-bit σ₀…σ₁₈ of `scan37J1000` and were compared
with the 1280-bit real record. Their reference is the record of the data they carry, "N0 scan37J1000" at K = 17 and
18, to 10⁻⁹ as written; the K = 19 completions of section 11.6 carry the 1280-bit node data and are compared with
"N0 scan41J1250" as written. Nothing else in item 4 changes.

### 11.1 σ₁₉: items 1(a), (b), (d) pass, item 1(c) fails

The `converge` stage ran as written on 2026-10-28 23:16 UTC, after the three T88 nodes landed (logs in the raw-data
repository's `planted-k19-mac/real/`; the record in `planted_checks_k19.json`, "K19 converge").

| wf | σ₁₉ | err19 (relative) | σ₀…σ₁₈ against `scan37J1000` | ratio law: \|Δ²\| at k = 17 |
|---|---|---|---|---|
| 0.99 | −1.8854371907076884952×10⁻¹⁶ + 1.7574120025598862231×10⁻¹⁶i | 2.07×10⁻²⁶ (8.03×10⁻¹¹) | bitwise equal, 19/19 | 0.0079 |
| 1.00 | −1.7398705529980921796×10⁻¹⁶ + 1.8204762245817697829×10⁻¹⁶i | 2.79×10⁻²⁷ (1.11×10⁻¹¹) | bitwise equal, 19/19 | 0.0078 |
| 1.01 | −1.5938905230179470051×10⁻¹⁶ + 1.8736843172586269260×10⁻¹⁶i | 1.20×10⁻²⁷ (4.86×10⁻¹²) | bitwise equal, 19/19 | 0.0078 |

- **Items 1(a), (b) and (d) pass at every node; item 1(c) fails at σ₁₉** (pilot (b) 3.39×10⁻²⁷ from (a) against (a)'s
  err19 2.79×10⁻²⁷; its σ₁₆…σ₁₈ pass, 15, 34 and 15 times below their errors). Each mutation fires (err19 × 10¹⁴:
  1.1×10³ relative; σ₁₈ at wf 0.99 scaled by 1 + 10⁻¹⁵: 18/19 bitwise; pilot (b)'s σ₁₉ scaled by 1 + 10⁻¹⁰: 1.6×10⁻²⁶
  against 2.8×10⁻²⁷; σ₁₉ × 1.1 at wf 1.01: \|Δ²\| = 1.19). The stages past `converge` wait for T91 and the dated
  amendment of item 1 (the record at the end of section 11.0).
- **err19 at wf 0.99 is 8.0×10⁻¹¹ relative, a margin of 1.2 under the 10⁻¹⁰ asked for,** 7 times the ω∞ node's and
  17 times the 1.01 node's (at K = 18 the three were within 25% of each other, 1.8–2.3×10⁻¹²). The J 1000 fits are at
  their limit at this order; the follow-up order runs with J raised (T89).
- **σ₁₉ follows the ratio law's rotation:** σ₁₈/σ₁₉ = 9.39 − 9.00i at ω∞ (arg advances 0.76 per order, |σ₁₈|/|σ₁₉| =
  13.0); the second differences of σ_k/σ_{k+1} are 0.008–0.012 at k = 10–17.
- **Spread in wf.** σ₁₉ changes by 6.2% of itself between the nodes (6.0%, 6.1% for σ₁₇, σ₁₈). The largest err19 is
  1.3×10⁻⁹ of that change (3.6×10⁻¹¹ at σ₁₈), the wf 0.99 node's.
- **Cost.** On the Mac with two processes per run: pilot (a) 209 min, pilot (b) 461 min, the nodes 356 and 390 min,
  the late-order fits taking 86% of the time; T91 shared the cores during the nodes' last two hours (comms,
  2026-10-28 13:14–23:13 UTC).
- **T91, the attribution pilot (d), J 1250 at 960 bits, 293 min** (`scan39J1250p960/`; the amendment at the end of
  section 11.0): σ₁₉ = −1.7398705530035671689×10⁻¹⁶ + 1.8204762245640747355×10⁻¹⁶i, err19 1.85×10⁻²⁷ (7.35×10⁻¹²
  relative), all 20 fits at Mf 7. |a − d| at σ₁₆…σ₁₉: 5.4×10⁻²⁷, 1.6×10⁻²⁷, 4.5×10⁻²⁸, 1.85×10⁻²⁷ (the J effect);
  |d − b|: 2.94×10⁻²⁶, 2×10⁻³³, 4×10⁻³³, 1.94×10⁻²⁷ (the precision effect). The amended err19 at wf 1.00 is
  6.17×10⁻²⁷; the amended item 1(a) at the other two nodes waits for T89's node runs.

### 11.2 σ₁₉ under the second amendment: items 1(a″), (b″) and (d) pass

The `converge` stage ran under `--nodes scan41J1250` on 2026-10-31 04:38 UTC (logs in the raw-data repository's
`planted-k19-mac/nodes/`; the record "K19 converge" in `planted_checks_k19.json`; the (a′) run of 04:23 UTC that
stopped the work is "K19 converge under (a')" there and `real/converge_amend.log`).

| wf | σ₁₉ (K 41, J 1250, 1280 bits) | err19 = plateau + 1.57×10⁻³⁰ (relative, margin) | σ₀…σ₁₉ at wf 1.00 against `scan39J1250` | ratio law: \|Δ²\| at k = 17 |
|---|---|---|---|---|
| 0.99 | −1.8854371908881486933×10⁻¹⁶ + 1.7574120026504357237×10⁻¹⁶i | 1.16×10⁻²⁸ (4.49×10⁻¹³, 223) | – | 0.0079 |
| 1.00 | −1.7398705530218642941×10⁻¹⁶ + 1.8204762245576625758×10⁻¹⁶i | 1.26×10⁻²⁸ (5.01×10⁻¹³, 199) | bitwise equal, 20/20 | 0.0078 |
| 1.01 | −1.5938905230187631094×10⁻¹⁶ + 1.873684317260655231×10⁻¹⁶i | 1.38×10⁻²⁸ (5.60×10⁻¹³, 179) | – | 0.0078 |

- **The 1280-bit plateau errors are 1.14, 1.25 and 1.36×10⁻²⁸** at wf 0.99, 1.00, 1.01, where the 960-bit files had
  2.07×10⁻²⁶, 2.79×10⁻²⁷ and 1.20×10⁻²⁷ (181, 22 and 9 times larger); the J 1500 pilot difference added to each,
  1.57×10⁻³⁰, is about 1% of them. The three nodes are within 20% of each other again, as at K = 18.
- **σ₀…σ₁₈ against the 960-bit values, recorded (the second amendment):** the differences are at most 1.7×10⁻¹³ of
  |σ_k| at k ≤ 18, and over that file's plateau errors the ratios are 1.00, 0.99 and 1.17 at σ₁₁ (wf 0.99, 1.00, 1.01),
  0.01, 0.01 and 1.01 at σ₈, every other one at most 0.74 (0.74 at σ₁₀, wf 0.99); the S_1 family's Mf pick differs at σ₈
  (9 against 10 at wf 1.01) and agrees at σ₁₁ (8 and 8), where seven or eight of the twelve neighbour families pick one
  higher at 1280 bits.
- **Each mutation fires** (`converge_mutate_<err|low|pilot|ratiolaw>.log`): err19 × 10¹⁴ at wf 1.00; σ₁₈ at wf 1.00
  scaled by 1 + 10⁻¹⁵ (19/20 bitwise); the J 1500 file's σ₁₉ scaled by 1 + 10⁻⁹ (the pilot difference 1.8×10⁻²⁵ at every
  node, 7×10⁻¹⁰ relative); σ₁₉ × 1.1 at wf 1.01 (\|Δ²\| = 1.19).
- **Spread in wf.** σ₁₉ changes by 6.2% of itself between the nodes (6.0%, 6.1% for σ₁₇, σ₁₈); the largest err19 is
  8.78×10⁻¹² of that change (3.65×10⁻¹² at σ₁₇, 5.96×10⁻¹² at σ₁₈).
- **Cost.** The two node runs took 1460 min each of the script's clock (K 41, two processes each, started 2026-10-30
  03:38 UTC, JSON at 2026-10-31 03:58 UTC); that clock counts the Mac's deep-idle hours (about 04:30–13:00 UTC) and
  some four hours under a system indexer that took five of the cores (comms, 2026-10-30 07:16 and 13:02 UTC). The
  wf 1.00 runs' times are in section 12.0.

### 11.3 The pipeline on plants first

51 plants at K = 17, 18, 19 (11.8 min in all), before the real data. All four checks pass:
- the regenerated plants' K = 17 and 18 estimates equal the stored K = 18 records exactly (both runs on the Mac; the
  plants do not read the node data below σ₁₈, so the change of node data leaves them unchanged);
- N1, 7/7: a lone simple zero recovered at K = 19 by arg, ratio, Padé and G to 3.5×10⁻⁷ at worst; graded against
  ω_t + 0.002 it fails, as it must;
- N2, 2/2: no estimator prints a number.

### 11.4 The real estimates at K = 19

The record "N0 scan41J1250" (the 1280-bit node data, second amendment), run 2026-10-31 04:51 UTC and read under the
third amendment at 05:04 UTC (`nodes/real.log`, `nodes/real_mutate_real.log`; "K19 real" in `planted_checks_k19.json`).

| estimator | K = 19 | K = 18 | K = 17 |
|---|---|---|---|
| arg crossing | 1.0000015 | 1.0000024 | 1.0000046 |
| ratio | 0.999986 | 0.999966 | 0.999912 |
| central Padé | [3/15]–[14/4]: 0.9999996–1.0000061 (mid 1.0000029) | [3/14]–[13/4]: 0.999997–1.000002 (mid 0.9999995) | [3/13]–[12/4]: 0.999978–1.000002 (mid 0.999990) |
| G (production κ̄) | 0.9999995 | 0.9999987 | 1.0000026 |
| Aitken, k − 2 … k | 1.0000009 | 1.0000011 | 0.9999925 |
| effective exponent c at ω∞ | 1.000 − 0.000i | 1.000 − 0.001i | 1.000 − 0.002i |

- **Item 3 under the third amendment.** The 960-bit σ₀…σ₁₈ through the harness reproduce "N0 scan37J1000" exactly
  (the control, both orders). The node data's K = 17 and 18 estimates differ from it by at most 9.3×10⁻⁹ (K = 17,
  `pade10/6`) and 3.3×10⁻⁸ (K = 18, `pade11/6`), 0.01–0.07 of the noise rms for every Padé crossing, 0.58 at most for
  the ratio estimates (ratio₁₁), 0.61 at most over all (arg₁₁), 0.03 and 0.05 for G; the check passes at both orders,
  and `--mutate=real` (σ₁₈ at wf 1.00 scaled by 1 + 10⁻⁹) fires at K = 18.
- **The arg crossings keep falling, more slowly:** 1.0000104, 1.0000046, 1.0000024, 1.0000015 for k = 16–19. The
  distance from 1 shrank by ×0.44 from k = 16 to 17, ×0.53 from 17 to 18 and ×0.63 from 18 to 19.
- **G stays below 1** (0.9999995, from 0.9999987), 5×10⁻⁷ under it. **The Padé mid moved above 1** (1.0000029, from
  0.9999995): the [3/15] approximant sits at 1.000006 and widens the central range upward, as [3/13] widened it
  downward at K = 17; [4/14]–[14/4] all lie within 2×10⁻⁶ of 1.
- **Aitken** 1.0000009 (from 1.0000011; the last two differences of arg_k are 2.2×10⁻⁶, then 0.9×10⁻⁶). Context, not
  part of any range.
- **Detectors:** D-ratio and D-ratiolaw, as at K = 16–18. **D-bracket fires on the real data at K = 19:** G (0.9999995)
  lies below the arg crossing (1.0000015) and the Padé mid (1.0000029) above it. A detector that fires on the real data
  cannot reject a completion under item 4's primary reading, so at this order the primary and the second reading of the
  ring test coincide.
- **The mutation controls fire.** σ₁₉ → −σ₁₉ at wf 1.00 fires D-G; σ₁₉ × (1 + 10⁻³) fires D-G and D-arg. Each moves the
  estimates by about 10⁻².



## 12. Follow-up: σ₂₀ (post-freeze)

### 12.0 Procedure, fixed before any K = 20 estimate was computed

Written and committed before any estimator ran at K = 20, on any input. Already known at this point (2026-10-30,
16:20 UTC): T89's two wf 1.00 runs at K 41 landed, (a) J 1250 at 1280 bits (03:03 UTC, (a) 12.4 h on two processes) and
(b) J 1500 at 1280 bits (03:41 UTC, (b) 13.0 h), copied from the raw-data repository's T89-mac at 55ed38a into
`scan41J1250/` and `scan41J1500/`; the node runs at wf 0.99 and 1.01 (K 41, J 1250, 1280 bits) are at order 25 of 41
and land about 23:30 UTC; the K = 19 stages past `converge` (sections 11.2–11.9) have not run, they wait for the same
node files (section 11.0's amendment). What the two wf 1.00 files say was read before this section was written, by the
pilot gate the comms entry of 2026-10-29 02:16 UTC fixed (rulings 1 and 2) and by `sigma_err_fit.py`, and is stated
here so that nothing below can be read as chosen after it: err20(a) = 1.335×10⁻²⁹, 7.18×10⁻¹³ of |σ₂₀| (the 10⁻¹⁰ gate
passes with a margin of 139); (a)'s σ₀…σ₁₉ are bitwise T88's pilot (b)'s (K 39, J 1250, 1280 bits), 20 of 20 strings;
|σ₂₀(a) − σ₂₀(b)| = 1.524×10⁻³¹, 0.011 of (a)'s err20, so ruling 2's case (a difference beyond err20, posted and not a
stop) did not arise; (b)'s err20 = 6.431×10⁻³²; and at σ₁₆ the two differ by 2.98×10⁻²⁶, 1.01 times (a)'s plateau error
there, the Mf 7 against Mf 8 plateau step of the S_1 family at 1280 bits that section 11.0's amendment traced between
(d) and (b) at K = 19 (2.94×10⁻²⁶), the same route. No estimator has run on any K = 20 input; no σ₂₀ exists at 0.99 or
1.01. The procedure is section 11.0's one order up, with item 1 written in the amended form from the start, and the
changes section 11's findings call for, stated here before the data:

0. **Dry run of the stages on a synthetic σ₂₀.** Before the real nodes are used, `planted_k20_dry.py` writes scan files
   whose σ₀…σ₁₉ are the real ones (`scan41J1250`'s) and whose σ₂₀ at each node is a C19-BP sum's: the ring-free sum
   passing the type rule that section 11.6 lists first (on the synthetic K = 19 data of section 11.0's dry run that was
   [6/13]; the real one is named when 11.6 is written, before this dry run), with err20 = 10⁻¹² |σ₂₀|, and a pilot
   file whose σ₂₀ at wf 1.00 is the same sum's, so that item 1's pilot term is zero there and its mutation `amend`
   tests the term. Every stage runs on them (`planted_k20.py STAGE --data DIR`). Pass: every stage completes with its
   checks passing, and in the ring test the planted sum has a σ₂₀ error below 10⁻¹⁸ and a τ₂₀ error below 10⁻¹⁷ and
   survives; whether its zero lies inside the dry run's K = 20 range is reported. A stage that fails on the synthetic
   data is fixed before the real data, and the fix is recorded here. The dry run needs section 11.6's completions, so
   it runs after section 11's stages and before any real K = 20 stage; its records and checks are discarded (its logs
   go to the raw-data repository); it tests the code path, not the estimators (item 2 does that).
1. **σ₂₀ is converged** if, at every node: (a) err20 ≤ 10⁻¹⁰ |σ₂₀|, where err20 is the node's plateau error from
   `sigma_err_fit.py` (the files are 1280-bit, their stored `sigma_err` is inf; a1 from the 960-bit `scan39J1000` file
   at the same wf) plus the wf 1.00 pilot difference |σ₂₀(a) − σ₂₀(b)|, 1.524×10⁻³¹, added at every node: it is the only
   measured J difference at this order, and no second J runs at the nodes (13 h each) because no item below needs σ₂₀
   to better than 10⁻⁹ of itself; (b) σ₀…σ₁₉ at wf 1.00 are bitwise T88's J 1250 / 1280-bit values (known to hold), and
   at wf 0.99 and 1.01 σ₀…σ₁₈ equal `scan39J1000`'s within that file's plateau errors (the J 1000 against J 1250
   differences at σ₁₆…σ₁₈ were 5.4×10⁻²⁷, 1.6×10⁻²⁷, 4.5×10⁻²⁸ at wf 1.00 against plateau errors 2.9×10⁻²⁶, 9.0×10⁻²⁷,
   1.1×10⁻²⁷; section 11.1), while the σ₁₉ differences at those nodes are the measurement section 11.0's amendment takes
   and are not a check here; (c) pilot (b)'s σ₁₇…σ₁₉ equal (a)'s within (a)'s plateau errors at those orders, and its
   σ₂₀ difference is the term of (a), recorded and not evaluated as a gate (ruling 2); (d) the second difference of
   σ_k/σ_{k+1} at k = 18 is at most 0.05. Each check gets a mutation that must make it fire
   (`--mutate=err|low|pilot|ratiolaw|amend`, the last scaling the pilot's σ₂₀ by 1 + 10⁻⁹).
2. **The pipeline on plants first.** N1 recovered by arg, ratio, Padé and G to 10⁻⁶ at K = 20, N2 giving no number;
   the 28 arc plants, C-EXP and the 13 REFIT plants regenerated and run at K = 18, 19, 20, their K = 18 and 19 estimates
   equal to the stored K = 19 records (`planted_results_k19.jsonl`) to 10⁻⁹, or the work stops. Control: N1 graded
   against ω_t + 0.002 must fail.
3. **Real data at K = 20** (record "N0 scan41J1250"): its K = 18 and 19 estimates must equal those of "N0 scan39J1000"
   to 10⁻⁹ (the two files differ at σ₁₆…σ₁₉ by the J and precision effects, 10⁻¹¹ of σ₁₉ at most, far inside this).
   Mutation controls: σ₂₀ → −σ₂₀ at wf 1.00 must fire a detector; σ₂₀ × (1 + 10⁻³) is reported.
4. **Ring test.** Every K = 19 completion of section 11.6 is regenerated and its zero must reproduce the stored
   "K19 recomplete" value to 10⁻⁹: C19-W1–3, C19-ARC (four shapes at their stored best ω_t) and C19-BP [m/(19 − m)],
   m = 3…9. A ring that section 11.4 records as surviving σ₁₉ (the K = 18 generation's C18-BP [3/15], [4/14], [5/13]
   are under test there; none is known to survive at this writing) is carried over and tested again here; if none
   survives, nothing is carried over. Benchmark: the simple-zero law from τ₁₈, τ₁₉. **The D-bracket question** is
   handled as section 9.0 item 4 fixed it: section 5.3's comparison is the primary reading (a completion survives if no
   detector fires on it at K = 20 that does not fire on the real data), and the table gives a second reading with
   D-bracket left out, both fixed here. A ring is a finding only if it survives under the first reading; a ring
   rejected under the first and surviving under the second is reported as that. The author's decision asked for in
   8.9 has still not come; if it comes before any K = 20 estimate is computed, this item is amended with the date. A
   ring that survives σ₂₀ is a finding, reported as such.
5. **Calibration** at K = 20, as section 10.5: the error of arg, ratio, Padé mid and G on (i) the 28 arc plants, (ii)
   C-EXP, (iii) REFIT with fit residual ≤ 10⁻⁴, and (iv) the BP family: section 11.0's sixteen sums plus the C19-BP
   sums that section 11.6 records as ring-free and passing the type rule at K = 19 (named when 11.6 is written, by
   that rule and not by their values; if none, the family stays at sixteen), each with its own zero as truth. The
   calibration at K = 19 from the new records must reproduce section 11.5's arc and BP-family ranges to 6×10⁻⁷
   (against the values stored in `planted_checks_k19.json`).
6. **Next generation and ring-free completions,** glued at σ₂₀ (tail from k = 21): ratio-law windows k = 6–19, 8–19,
   9–19; best-fit arc tails, four shapes, ω_t grid 0.9990–1.0010; the Borel–Padé sums [m/(20 − m)], m = 3…10. Frozen
   type rule: every pole has |t| < 2 at all three nodes. Zeros counted at wf = 1.00 in |x| < 7.3, 12 and 16.
7. **Combination.** The K = 20 estimator range is the hull of: the calibrated arg range over the 28 arc plants; the
   calibrated arg range over the BP family of item 5; and the zeros of the ring-free completions of item 6 that pass
   the type rule (no zeros besides x₀ in |x| < 12). G is reported with its spread over the minima of its objective
   within a factor 5 of the best and is not part of the hull. The headline is the union of the K = 19 headline (section
   11.8) and the K = 20 range if they overlap; if disjoint, both are reported and the reason diagnosed before any
   headline is written. The range is an estimator range, not a bound; the range from the last two orders alone is
   reported beside it as context, as at K = 17, 18 and 19.
8. **The order-20 projection.** est_k = ω* + c r^k through k = 18 and 19, for ω* = 1, the lower end of the K = 19
   range (section 11.8) and the lower end of the headline, against the real arg₂₀; decisive only if the other
   prediction is farther from arg₂₀ than three times the model's own error (its largest miss over k = 17–19 on the
   real data, and over the 28 arc plants at k = 20 given their true ω_t).

The stages are `converge`, `plants`, `real`, `ring`, `calib`, `recomplete`, `project`, `combine`, each run as
`nice -n 10 python3 planted_k20.py STAGE` from `solver/inner_amp/`, on `scan41J1250/` (the three nodes) and
`scan41J1500/` (the pilot). `planted_k20.py` is `planted_k19.py` one order up (a mechanical renumbering) with the
changes items 1, 4, 5 and 8 state: item 1's error sum is the rule, not a flag (`--amend` goes); 1(b) and 1(c) read the
files this item names; the calibration family and the projection's second hypothesis come from sections 11.6 and 11.8.
Order of work: section 11's stages first (on T89's node files, through `--amend`), then item 0, then the K = 20 stages.
The comparison of σ₂₀ with P5's type bound (the K = 19 recalibration's prediction) belongs to P5 and is recorded in
`proof/P5_scope.md`, not here. The `pick_plateau` change deferred in section 11.0 stays deferred until the nodes have
landed, so that all four files of this order come from the same code.

**Amendment of items 1(b) and 3 (2026-10-31 04:32 UTC), before any K = 20 estimate.** Item 1(b)'s comparison at wf 0.99
and 1.01 fires on the real data: at wf 1.01 σ₁₁ differs from `scan39J1000`'s by 1.17 times that file's plateau error
and σ₈ by 1.01 times (section 11.0's second amendment has the ratios at every node and the Mf picks behind them); the
cause is the one section 11.0's amendments record: the plateau error is the S_1 family's step, and the neighbour
families pick Mf one higher at 1280 bits at orders 17 and 23. As there, the differences are recorded and are not a
check; 1(b)'s check is the bitwise equality at wf 1.00 of σ₀…σ₁₉ with T88's J 1250 / 1280-bit file, with its mutation
`low`. Item 3's comparison is with the K = 19 record "N0 scan41J1250" (section 11.0's second amendment replaced
"N0 scan39J1000"), its K = 18 and 19 estimates equal to 10⁻⁹: the same σ₀…σ₁₉ through the same code, a reproduction
check and no longer a cross-precision one; the cross-precision comparison that remains is section 11.0's item 3, the
node data's K = 17, 18 estimates against "N0 scan37J1000" (960 bits). Nothing else changes.

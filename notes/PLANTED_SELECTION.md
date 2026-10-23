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

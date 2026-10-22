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

(Filled in after the runs.)

## 4. Verdict

(Filled in after the runs.)

# The corrector's basin: Kantorovich estimates of the chain's attempts (#655)

The octo-double corrector of the large-D chain (`hp/octo_corrector.py` `newton_od`, run by `hp/anm_run.py`) refuses
some attempts, and each refusal costs 5–20 min at 9 threads. This note asks whether the first Newton step of an attempt
already tells that it will be refused, through the Kantorovich quantity h₀ of the affine covariant Newton method, and
whether a step-size rule for the chain follows. It is an analysis of logs only, with no solver runs.

Results are in section 4 onward, after the protocol (section 3), which was committed before the held-out log was read.

## 1. The estimators

Per Newton step k the log has the max norms |F_k| and |dz_k| (leading part) at the iterate, the damping λ_k, and
|dz_{k+1}|, the Newton correction at the accepted point with that point's own Jacobian. From these,

```math
\theta_k = \frac{|dz_{k+1}|}{|dz_k|}, \qquad [h_k] = 2\theta_k, \qquad [\omega]_k = \frac{[h_k]}{|dz_k|}.
```

They come from the affine covariant Newton–Mysovskikh theorem, |dz_{k+1}| ≤ (ω/2)|dz_k|², with ω the Lipschitz constant
of F′(z)⁻¹(F′(y) − F′(x)) and h_k = ω|dz_k| (Deuflhard 2004, §2.1.1, the local error oriented Newton method). They are
lower bounds of h_k and ω from one step each, not enclosures. When λ_k < 1 the logged values are used as they are, and
the CSV records λ_k beside each θ_k. Only the first step's quantities, θ₀, [h₀] and [ω]₀, are known before the attempt
has spent more than one Newton step (about 50 s at 9 threads).

`hp/kantorovich_log.py parse` reads anm_run.py's log line by line, with the formats of `newton_od` and anm_run.py, and
writes one row per attempt to `hp/data/corrector_attempts.csv`. With `--jsonl` it checks each parsed attempt's a,
acceptance and step count against anm_run.py's own records file, and exits 1 on a mismatch.

## 2. Planted check (`kantorovich_log.py selftest`)

Criteria, fixed before the first run. Newton histories of test functions with a known answer, run in mpmath (110
digits) with newton_od's iteration and its default trial test, are printed in newton_od's and anm_run.py's formats,
parsed by the same parser, and judged by the same estimator.

- **(P1) ω.** F(x) = A F₀(x − x*), with a weak row σy₀ + (c/2)y₀² and five strong rows S_i(y_i + (κ/2)y₀²),
  S_i = 1e30…1e34, and a fixed random A with row scales 1e-3…1e3. Newton corrections do not depend on A, while |F| does.
  In the max norm F′(z)⁻¹(F′(y) − F′(x)) = (y₀ − x₀) g(z) e₀ᵀ with g₀ = c/(σ + cz₀) and g_i = κσ/(σ + cz₀), so
  ω(z) = max(c, κσ)/|σ + cz₀|, which is ω = c/σ on z₀ ≥ 0, where the iterates stay (proved by hand; the derivation is
  this line). With σ = 1e-28 and c = 1, ω = 1e28. The starts lie on the strong rows' slow manifold,
  y_i = −(κ/2)y₀²(1 + 1e-3 noise), as the chain's starts (|F₀| ≈ 1e-34 with |dz₀| ≈ 1e-31). Pass: 0.5 ≤ [ω]₀/ω ≤ 2 for
  every start with h₀ = ω|dz₀| ≤ 0.5. Starts with h₀ up to 26 are reported, not judged.
- **(P2) the order.** θ₀ = K|dz₀|^β with β = 1 for the quadratic fold above and β = 2 for a cubic weak row σy₀ + (T/6)y₀³,
  where the first step contracts as T y₀²/(3σ). Pass: the least-squares β of five starts is within 0.2 of 1 and of 2.
- **(P3) the prediction's error against a.** Starts y₀ with |dz₀|² = (k|z_N|a^p)² + f², k = 0.05, p = 25,
  f = 1e-35 × U(0.7, 1.4), 14 attempts with a in 0.3…1 and |z_N| in 1e-29…1e-24. Pass: the fit recovers p within 1.5
  and f within a factor 2.
- **Controls.** `--mutate F` takes θ from |F| instead of |dz| (also in `analyse`): (P1) and (P2) must fail.
  `--mutate nofloor` fits |dz₀| without its floor: (P3) must fail. A failed check exits 1.

Results (log in the raw folder `I655-mac/selftest.log`):

| Check | Result | Control | Control's result |
|---|---|---|---|
| (P1) [ω]₀/ω at h₀ = 1.6e-6 … 0.36 | 0.94–1.02: PASS | `--mutate F` | 5.4e3–1.8e9: FAIL |
| (P1), not judged, h₀ = 0.69, 2.1, 7.4, 26 | 0.84, 0.44, 0.14, 0.04 | | |
| (P2) β, quadratic fold / cubic | 0.991 ± 0.002 / 1.984 ± 0.030: PASS | `--mutate F` | 0.41 / 0.72: FAIL |
| (P3) p, f | 25.00, 1.03e-35: PASS | `--mutate nofloor` | p = 16.0: FAIL |

The ± are standard errors of the regression, not bounds. Past h₀ ≈ 1 the estimate saturates: θ₀ → 1/2 on the quadratic
fold, so [h₀] → 1 however far out the start is, and [ω]₀ falls below ω (ratio 0.04 at h₀ = 26). On the cubic weak row
θ₀ tends to 2/3 instead, so [h₀] → 4/3. A first step with [h₀] near 1 therefore says that the start is at or beyond
the edge of the quadratic basin, but not how far beyond.

## 3. Protocol (committed before the held-out log was read)

- **Training sample.** `I603-win2/chain.log`, raw commit be58c93, lines 1–514 (steps 0–37, D 225.13 → 250.84):
  45 attempts, 7 refused. It was read in full, by eye, before this protocol was written, so no part of it is held out.
- **Held-out sample.** `I461-win2/crawlJ.log` (raw be58c93, the rounded freeze projector, the same flags but
  `--threads 18` and the venv-t126 build), not read before this commit. Its verdicts are printed first by the command in
  section 5. Attempts that #603 appends after line 514 by the time of the last run are a second held-out set.
- **The test.** An attempt is predicted refused when its first step's [h₀] = 2θ₀ ≥ h*. The threshold separates when
  every refused attempt is caught and no accepted attempt is flagged; otherwise the counts are reported.
- **The threshold**, from the training sample by a rule: the geometric mean of the largest accepted [h₀] and the
  smallest refused [h₀] above it. Training gives 0.175 (D 246.26 → 247.04, a 0.781) and 0.55 (D 247.04 → 248.02,
  a 0.977, whose first step has λ = 1/4), so **h\* = 0.31**. A second, a-priori threshold is **1.0**, the saturation
  value of [h₀] on a quadratic fold (θ₀ = 1/2).
- **The a-rule**, fitted on the training sample only (section 6) and applied unchanged to the held-out attempts with
  their own D and |z_N|: a* is the largest a ≤ 1 whose predicted [h₀] is below h* = 0.31. An attempt is predicted refused
  when its a exceeds a*.

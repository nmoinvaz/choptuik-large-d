# The corrector's basin: Kantorovich estimates of the chain's attempts (#655)

The octo-double corrector of the large-D chain (`hp/octo_corrector.py` `newton_od`, run by `hp/anm_run.py`) refuses
some attempts. In #603 a refused attempt spent 727–1246 s of corrector time at 9 threads, and the seven of them took
6513 of the chain's 17303 corrector seconds from D 225.13 to 250.84. This note asks whether the first Newton step of an
attempt already tells that it will be refused, through the Kantorovich quantity h₀ of the affine covariant Newton
method, and whether a step-size rule for the chain follows. It is an analysis of logs only, with no solver runs.

Results are in section 4 onward, after the protocol (section 3), which was committed before the held-out log was read.

## Verdicts

| Question | Verdict | Numbers |
|---|---|---|
| Planted check: [ω]₀ within a factor 2 of a planted ω | PASS, and its control fails | ratio 0.94–1.02 at h₀ ≤ 0.5; with θ from \|F\|, 5.4e3–1.8e9 |
| Does the first step's [h₀] separate refused from accepted attempts? | NO, on training and on the held-out half | h\* = 0.31: training 3/7 caught, 0/38 false alarms; held out 2/5 caught, 0/11 false alarms |
| The a-rule against every refused attempt | it refuses 4 of 12, all of class A, and no accepted attempt | training 3/7 and 0/38 (in sample); held out 1/5 and 0/11 |
| The issue's form of the rule, one [ω] per D | it refuses none | a\* = 1 at every base |
| #644's reading, a start outside the quadratic basin | right for D 248.63 → 249.39, wrong for D 249.87 → 250.47 | section 7 |

- **The refusals are of two kinds.** In class A (5 of 12) the first step has [h₀] = 0.55–2, and every full step
  (λ = 1) of these attempts contracts |dz| by only θ = 0.37–0.73: the start lies beyond the quadratic basin. In class B
  (7 of 12) the first step has [h₀] = 4e-13 to 0.05, so the start is deep inside the basin by Kantorovich's measure,
  and the attempt fails one or two steps later, after a damped step, when no trial passes newton_od's own-Jacobian
  test. No quantity of the first step separates class B from the accepted attempts.
- **Class A is not a nonlinearity of the weak equations.** At the class A start 248.63 → 249.39, with the strong rows
  solved, the weak residual at Newton's first move is 7.9×10⁻⁷ of |B a| and departs from affine only at second order
  (#677, [REDUCED_CUBIC.md](REDUCED_CUBIC.md) 9). The basin that refuses newton_od is that of the full system, and a
  corrector that keeps the first weak move and solves the strong rows (#674) is not bound by it.
- **The first step contracts at third order, at a rate set by the smallest pivot.** θ₀ = K|dz₀|^β with
  β = 2.38 ± 0.12, and K grows as e^{1.07 D}, while the base's smallest LU pivot falls as e^{−1.01 D}. With the pivot as
  the regressor in place of D, θ₀ ∝ |dz₀|^2.33 piv^−1.03±0.08. This is the scaling of a weak direction with a vanishing
  quadratic term, θ₀ = T|dz₀|²/(3σ) (β = 2, exponent −1 of σ), which #499 found (notes/LIT_REVIEW.md, F2). It is not the
  β = 1 of a Lipschitz Jacobian with one ω, so the Kantorovich ω of these steps is not a constant: [ω]₀ grows as
  |dz₀|^1.4/σ. (Numerical evidence; the ± are standard errors of the regression.)
- **The basin shrinks by a factor 0.64 per unit D.** The |dz₀| whose predicted [h₀] is 0.31 falls from 7.5e-27 at
  D 230 to 9.3e-31 at D 250. On the exact projector the start's floor, |dz₀| ≈ 8.3e-36, would reach it near D 276, an
  extrapolation 25 units of D past the data.

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
- **(P2) the order.** θ₀ = K|dz₀|^β with β = 1 for the quadratic fold above and β = 2 for a cubic weak row σy₀ +
  (T/6)y₀³, where the first step contracts as T y₀²/(3σ). Pass: the least-squares β of five starts is within 0.2 of 1
  and of 2.
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
  their own D and |z_N|: a* is the largest a ≤ 1 whose predicted [h₀] is below h* = 0.31. An attempt is predicted
  refused when its a exceeds a*.

## 4. The refused attempts and the separation test

Every refused attempt of both samples, from `hp/data/corrector_attempts.csv`. θ of the later steps is the logged ratio
at the logged λ. Class A has [h₀] ≥ h\* = 0.31, class B below it. crawlJ ran at 18 threads, #603 at 9.

| Sample | D, base to target | a | [h₀] (λ₀) | θ of the later steps (λ) | Stop | Corrector s | Class |
|---|---|---|---|---|---|---|---|
| I603 | 236.13 → 237.13 | 1.000 | 6.2e-07 (1) | 0.246 (1/2) | no trial at step 2 | 727 | B |
| I603 | 238.54 → 239.51 | 0.977 | 4.8e-08 (1) | 0.248 (1/2) | no trial at step 2 | 804 | B |
| I603 | 240.12 → 240.89 | 0.763 | 3.6e-13 (1) | 0.371 (1/4), 0.97 (1/128), 1 (1/256) | stalled at step 3 | 1246 | B |
| I603 | 244.64 → 245.64 | 1.000 | 1.3 (1) | 0.232 (1/2), 0.189 (1/4), 0.944 (1/16) | no trial at step 4 | 1180 | A |
| I603 | 247.04 → 248.02 | 0.977 | 0.55 (1/4) | 0.582 (1), 0.469 (1), 0.2 (1/2), 0.6 (1/4) | no trial at step 5 | 1021 | A |
| I603 | 248.63 → 249.39 | 0.763 | 0.77 (1) | 0.733 (1), 0.564 (1), 0.387 (1/2) | no trial at step 4 | 767 | A |
| I603 | 249.87 → 250.47 | 0.596 | 0.0026 (1) | 0.19 (1/2), 0.322 (1/2) | no trial at step 3 | 768 | B |
| crawlJ | 237.13 → 238.13 | 1.000 | 0.00071 (1) | 0.24 (1/2) | no trial at step 2 | 197 | B |
| crawlJ | 239.54 → 240.51 | 0.977 | 0.052 (1) | 0.52 (1/2), 1 (1/16) | no trial at step 3 | 283 | B |
| crawlJ | 241.89 → 242.84 | 0.954 | 2 (1/2) | 0.633 (1/4), 0.789 (1/8), then 0.93–1 at λ 1/64–1/8 | stalled at step 8 | 653 | A |
| crawlJ | 242.36 → 242.84 | 0.477 | 0.73 (1) | 0.864 (1/2) | no trial at step 2 | 208 | A |
| crawlJ | 242.36 → 242.60 | 0.238 | 0.045 (1) | 0.316 (1/2), 1 (1/2) | no trial at step 3 | 242 | B |

The test of section 3:

| Sample | Attempts (refused) | Threshold | Refused caught | False alarms | Separates |
|---|---|---|---|---|---|
| I603, training | 45 (7) | h\* = 0.31 | 3/7 | 0/38 | no |
| I603, training | | 1.0, a priori | 1/7 | 0/38 | no |
| crawlJ and #603's step 38, held out | 16 (5) | h\* = 0.31 | 2/5 | 0/11 | no |
| held out | | 1.0, a priori | 1/5 | 0/11 | no |

A threshold on [h₀] has no false alarms in either sample, since no accepted attempt has [h₀] above 0.175. It misses
every class B refusal, whose first step is as good as an accepted one's: class B's [h₀] lie between 4e-13 and 0.052,
inside the range of the accepted ones. The control, θ taken from |F| (`analyse --mutate F`), flags 7/7 and 5/5
refusals with 26/38 and 11/11 false alarms. |F| rose in the first step of 46 of the 61 attempts (by up to 7.9e10),
while |dz| fell in 60, because the residual left by a good step is the quadratic remainder in strongly scaled rows.

## 5. Commands

From `hp/`, with the committed CSV (the analysis needs no raw data):

```
python3 kantorovich_log.py selftest                    # selftest PASS, exit 0
python3 kantorovich_log.py selftest --mutate F         # (P1) and (P2) FAIL, exit 1
python3 kantorovich_log.py selftest --mutate nofloor   # (P3) FAIL, exit 1
python3 kantorovich_log.py analyse data/corrector_attempts.csv --train I603-win2 --held I461-win2-crawlJ,I603-win2-new --h_star 0.31
```

The CSV is rebuilt from the raw logs by `I655-mac/run_655.sh` of the raw-data repository, run from `hp/` as
`bash run_655.sh IN OUT`, where IN holds the six input files its header names, extracted with `git show` at raw
be58c93 and c5ed39b. It writes every log of the raw folder, and its CSV equals the committed one byte for byte.

## 6. The fits and the a-rule (training sample, I603)

**θ₀ against |dz₀|.** Over the 42 attempts whose first step has λ = 1 and θ₀ < 0.3, least squares give
θ₀ = K|dz₀|^β e^{γ(D − 239.85)} with β = 2.381 ± 0.122 and γ = 1.071 ± 0.087 per unit D, rms 1.52 dex. The bootstrap
over the 42 attempts (2000 resamples) puts β at 2.24–2.53 (16–84%) and 2.10–2.68 (2.5–97.5%), a spread, not a bound.
With the base's smallest LU pivot in place of D, β = 2.333 ± 0.118 and the pivot's exponent is −1.031 ± 0.083, while
the pivot itself falls as e^{−1.009 D} over the training bases. The two-point exponents of the bases with two attempts
(a and a/2 from the same base, so without a D term) are 1.70–3.16 in I603 and 0.92–2.47 in crawlJ. crawlJ alone,
fitted after its held-out use, gives β = 2.20 ± 0.25 and a pivot exponent of −1.45 ± 0.72. The planted check recovers
β = 0.99 and 1.98 for a quadratic and a cubic weak row (section 2), so the fit can tell the two apart.

**[ω]₀ against D**, the issue's form: log₁₀[ω]₀ = −36.67 + 0.2405 D, rms 3.04 dex, and the residual correlates with
log|dz₀| at +0.75. One ω per D does not describe these steps, because [ω]₀ also grows with |dz₀| (as |dz₀|^{β−1}).

**|dz₀| against a.** The fit |dz₀|² = (k|z_N|a^p)² + f², with |z_N| the series' last coefficient norm from the step
header, gives p = 24.30, k = 0.0911 and a floor f = 8.30e-36 over the 45 attempts, rms 0.43 dex. With p fixed at
N + 1 = 25 it gives k = 0.103 and f = 8.39e-36, rms 0.44 dex. So the start's error is the series' truncation error,
about 0.1|z_N|a^{N+1}, down to a floor near 1e-35. The two-point exponents per base, 9.2–24.0, are lower because the
a/2 attempt sits near the floor. On the rounded projector (crawlJ) the fit fails (p = 1.75, rms 1.8 dex). At D
241.89–242.36 its |dz₀| is 1.3e-30 to 1.2e-29 at every a from 0.12 to 0.95, the rounded projector's a-independent
first move (#499's F1 in notes/LIT_REVIEW.md, "win2's floor of 1.3e-30 to 2e-30 for the first move at any a").

**The a-rule.** a\* is the largest a ≤ 1 for which 2K|dz₀(a)|^β e^{γ(D − 239.85)} < 0.31, with |dz₀(a)| from the fit
above and the base's own |z_N| and D. This is the shape of Deuflhard's affine covariant step-size control for
continuation, where the predictor's error and the corrector's first contraction set the largest feasible step (Deuflhard
1979; Deuflhard 2004, §5.1.2–5.1.3; the formulas there were not read). Against every refused attempt:

| Sample | D, base to target | a taken | a\* | a\*, one ω per D | Rule | Class |
|---|---|---|---|---|---|---|
| I603 | 236.13 → 237.13 | 1.000 | 1.000 | 1.000 | allows | B |
| I603 | 238.54 → 239.51 | 0.977 | 1.000 | 1.000 | allows | B |
| I603 | 240.12 → 240.89 | 0.763 | 1.000 | 1.000 | allows | B |
| I603 | 244.64 → 245.64 | 1.000 | 0.980 | 1.000 | refuses | A |
| I603 | 247.04 → 248.02 | 0.977 | 0.803 | 1.000 | refuses | A |
| I603 | 248.63 → 249.39 | 0.763 | 0.723 | 1.000 | refuses | A |
| I603 | 249.87 → 250.47 | 0.596 | 0.679 | 1.000 | allows | B |
| crawlJ | 237.13 → 238.13 | 1.000 | 1.000 | 1.000 | allows | B |
| crawlJ | 239.54 → 240.51 | 0.977 | 1.000 | 1.000 | allows | B |
| crawlJ | 241.89 → 242.84 | 0.954 | 0.836 | 1.000 | refuses | A |
| crawlJ | 242.36 → 242.84 | 0.477 | 0.898 | 1.000 | allows | A |
| crawlJ | 242.36 → 242.60 | 0.238 | 0.898 | 1.000 | allows | B |

It refuses the three class A refusals of I603 (in sample, with margins of 2–18% in a) and one of crawlJ's two, and
none of the 49 accepted attempts. crawlJ's other class A refusal, at a = 0.477, starts with the rounded projector's
error of 1.2e-29, which the training fit, made on the exact projector, puts near its floor. The issue's form,
θ₀ = \[ω\](D)|dz₀|/2 with the fit of [ω]₀ above, gives a\* = 1 at every base and refuses nothing.

On #603's bases a\* is 1 up to D 243.64, then 0.98 (244.64), 0.86 (246.26), 0.80 (247.04), 0.75 (248.02), 0.72 (248.63),
0.66–0.71 (249.01–249.39), 0.68 (249.87) and 0.61–0.63 (250.17–250.84). Near D 250, ln a\* falls by about 0.07 per
unit D. Past D 244 the chain's first attempts used a = 0.30–1.00 against a\* = 0.61–0.98.

## 7. What the data say about the basin

- **#644's reading.** At D 248.63 → 249.39 the first step contracts |dz| by only 0.385 ([h₀] = 0.77) and the next two
  full steps by 0.733 and 0.564: a start outside the quadratic basin, as #644 read it. At D 249.87 → 250.47 the first
  step contracts |dz| by 1.3e-3 ([h₀] = 0.0026). |F| then halves twice because both steps were damped to λ = 1/2,
  which is the (1 − λ) of a damped step rather than a contraction rate, and |dz| falls by 0.19 and 0.32 at λ = 1/2. That
  attempt started inside the basin by this measure, and is of class B.
- **The rate outside the basin.** The seven full steps (λ = 1) of the class A attempts contract |dz| by 0.367–0.733,
  mean 0.537, and the four after the first step by 0.587 on average. A first-order singular root gives 1/2 and an
  order-2 one 2/3 (Decker, Keller and Kelley 1983, Thms 5.1 and 5.9, as read for notes/LIT_REVIEW.md). The log's
  2-digit |dz| carries about ±7% per ratio, and the first steps are in transition, so the data do not pick between
  the two. The planted quadratic and cubic weak rows saturate at [h₀] = 1 and 4/3, and class A's largest [h₀] at λ = 1
  is 1.32.
- **How the basin shrinks.** From the θ₀ fit, the |dz₀| whose predicted [h₀] is 0.31 is 7.5e-27 at D 230, 7.9e-28 at
  235, 8.3e-29 at 240, 8.8e-30 at 245 and 9.3e-31 at 250, a factor 0.638 per unit D (bootstrap 0.623–0.655, 16–84%).
  The prediction's error grows with |z_N|, whose logarithm rose by 0.45 per unit D from D 225 and by 1.12 per unit D
  from D 240. If the trends held, the exact projector's floor of 8.3e-36 would reach the basin's edge near D 276
  (bootstrap 273–279, 16–84%), where no a would start inside it. That is an extrapolation of exponential fits 25 units
  of D past the data, not a prediction.
- **Class B shows at the second step.** A damped second step (λ₁ < 1) flags 10 of the 12 refusals, every class B one
  among them, with 16 false alarms in 49 accepted attempts; with [h₀] ≥ 0.31 added, it flags all 12. This was noticed
  after both samples had been read, so it is not a held-out result.

## 8. Discrepancies

- #655 cites Deuflhard (2004) §3.2 for the natural monotonicity test. In the book's table of contents §3.2 is "Residual
  Based Descent" and §3.3 is "Error Oriented Descent", with 3.3.2 "Natural level function" and 3.3.3 "Adaptive trust
  region strategies". `octo_corrector.py` cites §3.3, which agrees with the table of contents. §2.1.1, the local
  ordinary Newton method in its error oriented form, is as #655 cites it.
- #655's list of refused D (236.13, 238.54, 240.12, 244.64, 247.04, 249.39, 249.87) names six bases and one target.
  The attempt refused at a = 0.763 started at D 248.63 and aimed at 249.39.
- #644 groups D 249.87 with 249.39 (section 7), but the two refusals are of different classes.
- #655 gives 5–13 min per refused attempt, while #603's log has 727–1246 s (12–21 min) of corrector time per refusal.

## References

- P. Deuflhard, *Newton Methods for Nonlinear Problems: Affine Invariance and Adaptive Algorithms*, Springer Series in
  Computational Mathematics 35, Springer (2004; 2011 printing doi:10.1007/978-3-642-23899-4). The section numbers are
  from the author's table of contents (zib.de/userpage/deuflhard/pub/contentsnewton.pdf, read 2026-11-13): §2.1 Error
  Oriented Algorithms, 2.1.1 Ordinary Newton method (p. 45); §3.2 Residual Based Descent (p. 125); §3.3 Error Oriented
  Descent (p. 134), 3.3.2 Natural level function (p. 138), 3.3.3 Adaptive trust region strategies (p. 145); §5.1.2
  Affine covariant feasible stepsizes (p. 242), 5.1.3 Adaptive pathfollowing algorithms (p. 245). The text was not read,
  as notes/LIT_REVIEW.md also records. The estimates of section 1 follow from the Newton–Mysovskikh bound stated there.
- P. Deuflhard, *A stepsize control for continuation methods and its special application to multiple shooting
  techniques*, Numer. Math. 33 (1979) 115–146 (EuDML 132632, whose keywords include the Newton–Kantorovich theorem,
  predictors, corrector and shooting). Not read, and cited for the idea of section 6's a-rule.
- D. W. Decker, H. B. Keller, C. T. Kelley, *Convergence rates for Newton's method at singular points*, SIAM J. Numer.
  Anal. 20 (1983) 296–314, read in full for notes/LIT_REVIEW.md (#499's review).

Raw: `I655-mac/` of the raw-data repository at c82eb8e, with the run script, the planted check and its controls, the
parse and analysis logs, and the CSV.

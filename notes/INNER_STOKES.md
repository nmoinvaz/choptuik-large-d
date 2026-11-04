# The inner problem: its properties and the M = 1, 3, 5, 7 computations

Moved verbatim from README.md (commit ea42b3e, lines 1637–1675, 1695–1770) on 2026-11-04, in the reorganization of
[README_PLAN.md](README_PLAN.md); the README keeps a pointer where each block stood. Links were made relative to this folder.

<details>
<summary>Its properties: it resums the outer expansion, Kummer's equation and its Stokes constant without gravity, the selection rule as an inner problem, the inner series with gravity</summary>

- **It resums the outer expansion.** Its formal solution regular at ξ = 0, $`y = \sum_j d_j\xi^{2j+1}`$,
  $`\sigma = \sum_j s_j\xi^{2j+2}`$, $`c = \sum_j g_j\xi^{2j}`$ with d₀ = 2F, g₀ = 1, has g₁ = F², s₀ = 2(F + ω F'),
  d₁ = ω²F'' + 3ω F' + 2F - 2F³: the top-degree coefficients of G₁, S₁, D₁ of Proposition 1. `solver/largeD_inner.py`
  runs the recursion and finds $`d_k`$, $`-g_k`$, $`s_{k-1}`$ equal to the top-degree coefficients of the outer $`D_k`$, $`G_k`$, $`S_k`$
  to 2×10⁻¹³ at orders 1–10 ($`F = 2.72\sin\theta`$, ω = ω∞).
- **Without gravity it is Kummer's equation, and C(n,s) is asymptotically its Stokes constant.** With c = 1 and one harmonic, the regular
  solution is $`y = 2\xi\,{}_2F_0\big(\alpha, \beta; ; 2\xi^2\big)`$, i.e. $`d_j = 2^j(\alpha)_j(\beta)_j/j!\;d_0`$, and the second solution
  carries $`e^{-1/(2\xi^2)}`$, the outer fast mode $`e^{-n/(2x^2)}`$. The Stokes line of $`{}_2F_0(\cdot;2\xi^2)`$ is real ξ, i.e. real
  x > 1, and its Stokes constant is proportional to 1/(Γ(α)Γ(β)) (DLMF §2.11 and §13.7 for the
  Stokes phenomenon of $`{}_2F_0`$, i.e. Kummer's U). Matching gives the constant K above and the large-n form of
  C(n,s) displayed after this list.
- **The selection rule as an inner problem (working hypothesis).** With gravity the rule becomes: the fast-type Stokes multipliers of the formal
  inner solution vanish, one complex condition per odd harmonic μ. The unknowns are F's odd harmonics (less one phase)
  and ω, so the count balances, and the problem contains F and ω only (the free functions $`K_k`$ of
  Proposition 1 multiply x, the lowest degree, so they never reach the top-degree terms the inner problem resums). This
  formulation is derived from the matching above but not yet proved.
- **What the inner series shows with gravity.** On the Stokes line the inner variable is real and the fast characteristic
  is $`\theta_0 + 2\omega\ln\xi`$, also real, so the multiplier needs F on the real θ axis only. (In the outer labels the same
  characteristic carries a shift -iπω; that is branch bookkeeping between 0 < x < 1 and x > 1, and it is the phase
  $`e^{-i\pi\Lambda}`$ in C below.) Without gravity, harmonic 1 of the inner series grows at ratio/j → 2 (the Borel singularity at
  $`\zeta = \tfrac12`$, dual to 1/ξ²). With gravity ($`F = 2.72\sin\theta`$, 60 orders) the growth is dominated by high harmonics,
  ratio/j 6.8, 4.9, 4.2 at j = 10, 30, 50. That growth is not a second singularity: it is the same fast singularity at
  $`\zeta = \tfrac12`$ (Borel variable u = ξ²) carried by the higher harmonics, whose exponents have large imaginary parts
  ($`\lvert\mathrm{Im}\,g\rvert \approx 8.45`$ for the ν = ±3 modes of the M = 3 truncation; untruncated it would be
  3ω∞ = 11.47) and whose constants are 10⁴ times larger; at M = 3 the four modes at
  $`\zeta = \tfrac12`$ alone fit the series to 2.5×10⁻⁸. Gravity couples the ±μ fast modes, so the multiplier is
  a matrix over harmonics. *Caveat:* those M = 3 constants come from fits that pin each mode's corrections with
  `solver/inner_borel/inner_fastmode.py`, whose correction terms were later found wrong from the third order on for M > 1
  (off by 3×10⁻³ at m = 3); the M = 1 values are unaffected (they agree with free fits to 8 digits). The exponents
  are reliable: the truncated ones converge to the untruncated $`g_\nu`$ as M grows (to 10⁻⁷ at M = 11,
  `solver/inner_borel/inner_Mconv2.py`). Whether the constants converge in M is undetermined: free fits stop pinning
  them from M = 7 on (modes with $`(\nu\omega)^2 \gtrsim j`$ are not yet asymptotic, and their constants are $`e^{\pi\lvert\nu\rvert\omega/2}`$
  larger), and a heuristic estimate of the gravity mixing at $`\mathrm{Im}\,\theta \approx \pi\omega/2`$ suggests that truncation would
  need M of order 10⁴, which favours the untruncated amplitude expansion below.

</details>

<details>
<summary>The computation with gravity in detail: checks without gravity, the fast mode, harmonic truncation, one to five harmonics</summary>

- *Both checked without gravity.* Lateral: the jump divided by $`2\pi i\,2^{-s-1/2}/(\Gamma(\alpha)\Gamma(\beta))`$ times the linear fast solution is
  1.000000000002 - 7×10⁻¹²i, the same at two matching radii (`solver/inner_stokes/lateral_np.py check`). Darboux:
  the constant d₀/(Γ(α)Γ(β)) to 6×10⁻²⁷ from 200 orders (`solver/inner_borel/inner_stokes.py linear`;
  despite its name this script belongs to the Borel–Darboux implementation).
- *The fast mode with gravity, in the inner region* (derived): with $`\delta y = e^{-1/(2\xi^2)}a`$, the order-ξ⁻² terms cancel and
  $`\xi\,\partial_\xi a + 2\omega\,\partial_\theta a = 2F^2a`$, the large-x limit of Proposition 3. So the multiplier is a function of the
  characteristic label, $`S(\theta_0) = \sum_\nu S_\nu e^{i\nu\theta_0}`$, multiplying $`e^{-1/(2\xi^2)}\xi^{2\langle F^2\rangle}e^{\Pi(\theta)}`$ with
  $`\Pi = \omega^{-1}\!\int(F^2 - \langle F^2\rangle)`$. The extra power $`\xi^{2\langle F^2\rangle}`$ is seen in the computed jumps. In Borel terms
  the singularity stays at $`\zeta = \tfrac12`$ and the Dingle exponents become $`g_\nu = -(\tfrac12 + i\nu\omega) + \langle F^2\rangle`$ (the convention
  of g in Proposition 4, where $`g = -(s + \tfrac12)`$ without gravity; the late orders grow like $`\Gamma(j - g_\nu)`$), each mode spread over
  harmonics ν + 2k by $`e^{\Pi}`$; Darboux fits with these exponents leave residuals of 5×10⁻⁸ (M = 1).
- *Harmonic truncation is not a controlled approximation* (derived and seen). The linear multipliers grow like
  $`S_\nu \propto F_\nu\,e^{\pi\lvert\nu\rvert\omega/2}`$ ($`1/\lvert\Gamma(\alpha)\Gamma(\beta)\rvert \approx 47`$ for ν = 1, 4.4×10⁶ for ν = 3), so
  S(θ₀) involves F continued to $`\mathrm{Im}\,\theta = \mp\pi\omega/2 \approx \mp6`$, and a harmonic generated only by the
  nonlinearity already carries a jump 10²–10⁴ times that of harmonic 1.
- *One harmonic* ($`F = A\sin\theta`$): the multiplier has a genuine, isolated zero at A = 2.542148, ω = 2.969777
  (Δ = 2.11571). Both methods find it: lateral (`solver/inner_stokes/zero_M1.py`, the same at three matching radii) and
  Darboux (`solver/inner_borel/inner_zero.py`, Newton converging quadratically, $`\lvert S\rvert = 3\times10^{-14}`$ at 220 orders; it is the only
  clean zero for 0.5 ≤ A ≤ 3, 0.5 ≤ ω/ω∞ ≤ 1.5). The amplitude is 6.5% below F₁ ≈ 2.72; ω is 22% below
  ω∞. Adding harmonic 3 destroys that zero (lateral residual 4.6×10⁶ at the one-harmonic zero with harmonic 3
  added); a first search without de-mixing (`solver/inner_stokes/zero_M3.py`) found only an unphysical zero
  (ω = 0.198 ω∞, F₃/F₁ = 0.65). (Δ = 2.1157 is close to Δ(20) = 2.1121, almost certainly a coincidence.) So one
  harmonic does not select ω∞.
- *More harmonics* (`solver/inner_stokes/inner_np.py`, `staged.py`, `continue_M.py`). Starting the lateral sum on the ray
  $`\arg\xi^2 = \pi/2 + \pi/8`$ from the series computed in scaled form removes the amplification along the arc: without gravity
  the jumps of harmonics 1–9 come out as 1.00000000 times their exact values, in double precision, about 1 s per
  evaluation (`inner_np.py checklin 9`). With gravity the residuals are de-mixed (multiplied by $`e^{-\Pi}`$ and taken on the
  label θ₀), which leaves the zeros unchanged but gives each condition its own weight. At the one-harmonic zero, solving
  only the harmonic-3 condition gives F₃/F₁ = 0.074; the harmonic-1 condition is then violated, so that zero does not
  survive harmonic 3.
- *Three harmonics: the full selection* (`solver/inner_stokes/nested.py`, `full_M.py`, `m3_diag.py`). Solving
  S₁ = S₃ = 0 jointly for (A, ω, F₃) is badly conditioned (Jacobian singular values 6×10⁹, 5×10⁹, 23, 0.4): the
  de-mixing leaves a leakage of the harmonic-3 condition into the harmonic-1 one about 10⁴ times r₃ itself (about
  $`e^{\pi\omega}`$ times the finite-ξ de-mixing error), so the joint solve stops on false near-zeros where a small S₃
  cancels S₁. Solved nested instead (S₃ = 0 for F₃ to 10⁻¹¹ at fixed A and ω, then Newton on (A, ω) for S₁ = 0,
  a 2×2 problem of condition about 2, converging quadratically), there is one zero. It is the same from two
  different starts and at three matching radii (ξ₁ = 0.175, 0.2, 0.225; it moves by less than 10⁻⁷):
  A = 2.7265, ω = 1.0156 ω∞ (Δ = 1.6186), F₃/F₁ = 0.0530 + 0.0041i, ⟨F²⟩ = 3.727. The shape is that of the
  D = ∞ solution (extrapolated from the data: A ≈ 2.70–2.72, |F₃/F₁| ≈ 0.055–0.058, ⟨F²⟩ ≈ 3.65–3.7), and ω is
  1.56% above ω∞, against 22% below with one harmonic.
- *Five harmonics* (`solver/inner_stokes/hier.py`). At M = 5 the couplings are larger still (∂r₅/∂B₅ ≈ 2×10⁸,
  ∂r₃/∂B₅ ≈ 10¹³, ∂r₁/∂B₅ ≈ 10¹⁷), and the nested solve's homotopy stalled. A hierarchical solve works: S_M = 0
  innermost, then S_{M−2}, …, S₃, and S₁ = 0 for (A, ω) outermost, each level a damped 2×2 Newton step judged by step
  size; each condition with ν ≥ 5 is first solved at a small radius where its jump is linear and then polished at
  ξ₁ = 0.2, where it is resolved about 10⁸ times better. At fixed (2.72, ω∞) it reproduces the amplitude expansion's
  harmonics, which share no code with it: F₃/F₁ = 0.05486 + 0.00541i and F₅/F₁ = 0.00309 + 0.00054i against
  0.05482 + 0.00543i and 0.00310 + 0.00056i (the differences are of the size of the omitted F₇). The harmonic-1
  residual there is 6.5 + 1.8i, against −59 − 99i at M = 3 (and 271 + 329i, 74 − 311i at ω = 1.05, 0.95 ω∞). The full
  selection, run to its noise floor on the 8-thread machine (four outer Newton steps, |r₁| 0.25 → 0.05; A and ω steady to
  4×10⁻⁵ and 8×10⁻⁶ over the last steps), gives **A = 2.71858, ω = 0.99923 ω∞ (Δ = 1.64516, 0.077% above Δ∞)**. So the
  offset in ω goes −22%, +1.56%, −0.077% for M = 1, 3, 5: it alternates in sign and shrinks fifteen- to twentyfold per
  added pair of harmonics. Aitken's extrapolation of the three gives ω/ω∞ = 1.0003, though three terms are a thin basis.
  At M = 7 double precision itself becomes the limit: near the zeros, one ulp of B₇ moves r₁ at ξ₁ = 0.2 by 200–250,
  which at ∂r₁/∂(ω/ω∞) ≈ 7×10³ is a floor of 0.3–3% on ω, far above the expected M = 7 correction. `hier.py --schur`
  removes it: after the first inner solve it takes the Jacobian K of all residuals with respect to the higher harmonics
  (finite steps of thousands of ulps, still linear) and uses $`r_l - K_{lH}K_{HH}^{-1}r_H`$, i.e. r_l as if the higher
  harmonics sat exactly on their zeros. At M = 5 it changes r₁ by 0.02 (the old floor) and leaves F₃/F₁, F₅/F₁ unchanged
  to six digits. Re-running the full M = 5 selection with it takes |r₁| from the old floor 0.05 to 8×10⁻⁷ and moves ω/ω∞ by
  only 2.6×10⁻⁶, to 0.999228 (A = 2.718579). At M = 7 each nested solve costs as much as a whole M = 5 level, so the
  full selection is slow; `--predict` (linear prediction of the higher levels' zeros from K) and `--floor` (stop each nested
  solve at 3× that level's measured noise floor) cut the nested solves after the first to a median of 1–3 residuals at
  levels 5 and 7 and 27 at level 3, and reproduce the M = 5 zero to 2×10⁻⁷ in about a third of the evaluations. M = 7 is
  now done ([proof/B_tail_M7.md](../proof/B_tail_M7.md)). Three changes made it affordable and resolvable: a compiled
  backend (`inner_stokes/inner_nb.py`, the same series and DOP853 integrator, 15–20× faster), a residual that skips a
  pathological arc while the top harmonic is untuned (`hier_nb.py`), and the matching radius ξ₁ = 0.15, at which the
  leakage of harmonics 5 and 7 into the low conditions is small enough for level 3 to converge (at 0.2 it stalls at
  |r₃| ≈ 10⁻²). The outer Newton runs over complete fixed-(A, ω) solves (`m7_outer.py`). The seven-harmonic zero is
  **A = 2.71922, ω/ω∞ = 1.000022** (±2×10⁻⁵, from the scatter of repeated solves; at the matching radii 0.14 and 0.16 the
  solves did not converge well enough to give a zero, so the independence of ξ₁ is untested), with F₇/F₅ = 0.0572. The offset in ω goes −22%, +1.56%, −0.077%, +0.0022% for M = 1, 3, 5, 7; the last
  ratio, −0.049 per pair, is the profile decay of the added pair (0.057) times a coupling of 0.85. The geometric limit
  is ω/ω∞ = 0.999985, and the part above M = 7 is about 4×10⁻⁵ in ω.

</details>

## What is known about the selection, in detail (from the README)

Moved verbatim from README.md (commit 96c7ab7, the four bullets of "Limits and open problems") on 2026-11-04 (README_PLAN.md,
stage 3); links made relative to this folder.

- **The expansion diverges.** `solver/largeD_late_orders.py` runs the recursion of Proposition 1 numerically to order 36,
  as polynomials in x with coefficients sampled in θ. It reproduces sympy at orders 1–2 to 10⁻¹¹, and roundoff
  harmonics are projected out. For $`F = 2.72\sin\theta`$ and ω = ω∞, the terms grow factorially (ratio of
  successive orders ≈ 7–$`8.6\,k`$), dominated by ever-higher harmonics. The rate depends on the amplitude and on
  ω. This is the setting of exponential asymptotics: the series is divergent and its Borel-type remainders carry the
  selection.
- **The weak directions look like the selection, measured.** The near-null directions of the Jacobian (the "junk"
  directions of the solver sections) are variations of F's high harmonics, which Proposition 1 leaves free, so their
  singular values plausibly measure how strongly the hidden condition pins each harmonic; that identification has not
  been checked directly. For harmonics k = 7, 9, 11, 13 they fall between D = 20 and 60 about like
  $`n^{-10.5}, n^{-14}, n^{-17.5}, n^{-21}`$, roughly $`n^{-1.6k}`$. $`\max\lvert\partial\Delta/\partial F\rvert`$ is itself a
  power law where the harmonic set is fixed, $`n^{21.7}`$ on D = 40–61, so these power laws, not an exponential, are what
  the solver measures: it matches at x = 0.7, where the selecting fast mode is of order one
  ([notes/BEYOND_LIGHT_CONE.md](BEYOND_LIGHT_CONE.md)).
- **F's singularities recede, to a finite distance.** The decay rate of F's harmonics grows from 0.58 at D = 12 to
  1.02 at D = 60, which first suggested $`\tfrac14\ln n`$ growth and an entire F∞. The harmonic ratios extrapolate instead to
  a finite rate, as the amplitude expansion of the inner problem predicts ([above](../README.md#the-inner-problem-at-large-x)), so
  F∞ is likely not entire.
- **The nonlinear late orders are dominated by something else.** In the nonlinear recursion the ratio of successive
  orders grows like 7k–8.6k, against $`\lvert\chi\rvert \ge \pi/2`$ for the fast mode, so the fast
  singulant is not directly visible. `solver/largeD_late_structure.py` characterizes the dominant growth: the rate is the
  same at x = 0.25, 0.5, 0.75, 1 (at x = 0.25 it is carried by the x³ coefficient; the top-degree share at x = 1
  falls to 0.008 by order 27), and the dominant harmonic grows like 0.55k, almost independently of the amplitude
  (13, 15, 17 at order 25 for A = 1.36, 2.72, 5.44) and of ω, while the rate does depend on them (about 6, 8.5,
  12 for those amplitudes, 25 at 2ω). An x-dependent singulant of either eikonal branch is ruled out: a heat-like
  slow-branch singulant would give a rate that depends on x, a fast one $`\lvert\chi\rvert \ge \pi/2`$. The divergence lives in the free data
  (θ-harmonics of F and the $`K_k`$), consistent with an x-independent Borel singularity per harmonic that moves
  towards 0 as the harmonic grows. Its ambiguities are exponentially small changes of F's high harmonics, like the
  solver's weak directions, not light-cone terms. (In the inner series the analogous fast high-harmonic growth turned out
  to be the fast singularity itself, carried by harmonics with large exponents; whether the same holds for this outer
  growth is not settled.) At a fixed number of harmonics there is no such singularity close to the origin: the
  ball-arithmetic recomputation finds none at one harmonic and one near ζ ≈ −1.2 at three
  ([proof/B3_borel.md](../proof/B3_borel.md) §5).

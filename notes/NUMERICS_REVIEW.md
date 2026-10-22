# Independent review of the numerics behind Δ∞

Task: an independent review, by reading the code and testing it, of the numerics behind the headline
Δ∞ = 1.64389375 ± 1.5×10⁻⁸ at commit 78de00b. In scope: the double-double solver (`hp/src`, `hp/python/hpkernel/driver.py`,
`hp/scan_hp.py`, `hp/pinning.py`), the per-point error estimate, the data pipeline (`hp/export_dd.py`,
`hp/data/delta_dd.csv`), the fits and the error bar (`hp/dd_fits_mp.py`, `hp/dinf_budget.py`, `hp/planted_dinf.py`), and
briefly `hp/gamma_dd.py`. Reviewer: agent RV2. Test scripts and their logs: `hp/review/rv2_*`.

## Contents

- [Verdict](#verdict)
- [Summary of the checks](#summary-of-the-checks)
- [Issues, ranked](#issues-ranked)
- [1. The solver](#1-the-solver)
- [2. The error per point](#2-the-error-per-point)
- [3. The data pipeline](#3-the-data-pipeline)
- [4. The fits and the error bar](#4-the-fits-and-the-error-bar)
- [5. γ](#5-γ)
- [6. What I could not check](#6-what-i-could-not-check)
- [7. Reproduce](#7-reproduce)

## Verdict

**The solver is right. The points are less noisy than the README says but carry a smooth bias of about 10⁻¹⁵ from
the kfreeze cut that its check missed; neither changes Δ∞ at the quoted precision. The error bar on Δ∞ covers every
truth I could plant, but it is a fragile statistic, and its centre is probably high by about 10⁻⁸.**

- **Equations and arithmetic.** I re-derived the centre, Taylor and horizon recursions from the field equations of
  `solver/dss.py` and found no error. An independent evaluation of the residual (R1: python-flint at 200 bits, my own
  τ-derivative from the DFT, my own Picard integrator, the kernel's centre and horizon series verified order by order)
  agrees with the kernel's residual to 2×10⁻³¹ in every block at D = 60.0 and at the second machine's last state
  (D = 61.026), and to 10⁻²⁵ relative away from the solution (|F| up to 8×10⁻²). The negative controls fire.
- **Discretisation at D = 60** (R2). Taylor order and step tolerance, series orders, the first node, and kicks of 10⁻⁸
  to the weak p1 harmonics 9, 11 and 13 move Δ by at most 6×10⁻¹⁸, and M = 48 τ-points by 2.5×10⁻¹⁸. The kfreeze-13
  cut is the one approximation with a visible effect: freeing harmonic 15 moves Δ by +1.5×10⁻¹⁵ at D = 40,
  −1.1×10⁻¹⁵ at 50 and −1.2×10⁻¹⁵ at 60, against the README's 3×10⁻¹⁷ at D = 40 and "falls off with D" (a check that
  passes when Newton takes no step). Through the fits it moves Δ∞ by at most 5.5×10⁻¹⁰.
- **Noise per point** (R3). Between D = 45 and 60.1 the points scatter about smooth local fits by 3–7×10⁻¹⁸ rms, and the
  second machine agrees with the cloud points to 7×10⁻¹⁸ rms (the README's "2–4×10⁻¹⁶" is the f64 resolution of the
  comparison, one ulp of Δ). As noise, "about 10⁻¹⁶" is conservative there; as error it is not, because of the cut's
  smooth 10⁻¹⁵ (above). Below 60.1 the junk-accepted run up6 (D = 26.9–35.2) carries errors up to 1.0×10⁻¹⁴, of which
  the cleaning removes 24 of 149 points and leaves errors up to 5×10⁻¹⁵, and the first kfreeze-13 points (35.35–37)
  carry up to 4.6×10⁻¹⁵ (as the README says). The linearised correction −∂Δ/∂F·F that every run recorded predicts the
  convergence errors in sign and size; it cannot see the cut.
- **Data pipeline.** The CSV is exactly the documented merge; no D appears twice; every hi + lo is normalised. One
  precision bug: `dd_fits_mp.load_points` parses the CSV's shortest-repr strings as decimals at 60 digits, which moves
  every point by up to 2.1×10⁻¹⁶. All fits, the budget and the planted test see the data rounded to about f64. The
  effect on the quoted range is 5×10⁻¹¹; on the noise diagnostics it is the dominant "noise" above D = 45.
- **The bar.** I reproduced the 48-fit range 1.64389374444–1.64389376313 with separate code. Its coverage holds: the
  repository's planted test and mine, whose truth (the README's rational formula) is not made by the fitting code and
  carries the real data's residual pattern, both put the truth inside the range. But:
  1. in all of those plants the truth sits at the bottom of the range and the centre is 0.8×10⁻⁸ high; removing the
     dirtiest data (27 ≤ D ≤ 37) moves the real centre down by 0.9×10⁻⁸, to 1.6438937451;
  2. the top of the quoted range (1.643893763) comes from a fit that the same rule rejects on cleaned data: it passes
     only because the floor of the uncleaned data is set by junk points;
  3. the rule "two lowest degrees at the floor" replaced "every degree at the floor" (spread about ±1.5×10⁻⁶) within one
     session, before the planted test existed; with "three lowest" the bar is ±6.5×10⁻⁸, and on cleaner data (the 168
     points the runs themselves flag removed) the same rule gives ±9.7×10⁻⁸.

  So Δ∞ = 1.64389375 ± 1.5×10⁻⁸ is consistent with everything I checked, but the bar is the spread of the
  lowest-degree fits, not a calibrated error, and a better central value is 1.643893746. I would quote
  Δ∞ = 1.643893746 ± 1.5×10⁻⁸ (or keep 1.64389375 with ± 2×10⁻⁸) and say what the bar is.

## Summary of the checks

| check | verdict | severity |
|---|---|---|
| 1a. centre, Taylor and horizon recursions = the equations of `solver/dss.py` | OK (by hand; R1) | — |
| 1b. boundary conditions: regular centre, c = 1 and analyticity at the light cone | OK (R1) | — |
| 1c. collocation, folded d/dθ, parity of products | OK (R1: DFT-defined derivative) | — |
| 1d. double-double kernels (two-sum, FMA two-product, Sum2 dot, divide) | OK | — |
| 1e. independent residual at saved states | OK, 2×10⁻³¹ (R1) | — |
| 1f. f64 drops in the driver (predictor, deflated step, junk hold, resampling) | OK: all self-corrected by Newton | — |
| 1g. Taylor order, step tolerance, series orders, node placement at D = 60 | OK, ≤ 6×10⁻¹⁸ (R2) | — |
| 1h. convergence in M at D = 60 | OK: M = 48 moves Δ by 2.5×10⁻¹⁸ (R2) | — |
| 1i. kfreeze-13 cut, D = 40, 50, 60 | +1.5, −1.1, −1.2×10⁻¹⁵, not the quoted 3×10⁻¹⁷ (R2) | cosmetic for Δ∞, ≤ 5.5×10⁻¹⁰ (F9) |
| 1j. junk hold / trend and the dnp corrector | OK: Δ returns to 6×10⁻¹⁸ after 10⁻⁸ kicks (R2) | — |
| 1k. junk_only acceptance (run up6) | accepts Δ errors up to 10⁻¹⁴ by design | affects error bar (small, F6) |
| 1l. series tails checked only after a point is accepted | issue (no effect found) | cosmetic (F8) |
| 2a. "about 10⁻¹⁶ to D = 60.1, a few 10⁻¹⁵ above" | noise: conservative at 45–60.1; error: 10⁻¹⁵ from 35.35 on (cut), up to 10⁻¹⁴ at 27–35 | cosmetic (F6, F9) |
| 2b. dDelta_est as the per-point error | OK: matches leave-one-out residuals (R3) | — |
| 2c. two machines "agree to 2–4×10⁻¹⁶" | agree to 7×10⁻¹⁸; the quoted check cannot resolve below one ulp | cosmetic (F5) |
| 3a. CSV = merge of the runs, local after cloud, last per D | OK (R4) | — |
| 3b. D values exact, no duplicates, hi + lo normalised | OK (R4) | — |
| 3c. `dd_fits_mp.load_points` parses hi as a decimal string | precision lost to ~2×10⁻¹⁶ per point | cosmetic for Δ∞ (F4) |
| 4a. 48-fit range, reproduced independently | OK (R5) | — |
| 4b. top of the range from an under-resolved, uncleaned fit | issue | affects error bar (F2) |
| 4c. selection rule chosen after the fact; fragile | issue | affects error bar (F1) |
| 4d. centre of the range vs planted truths | 0.8×10⁻⁸ high in every plant | affects value, within the bar (F3) |
| 4e. weight of the points above 60.1 | a coherent 3×10⁻¹⁵ there moves Δ∞ by up to 1.1×10⁻⁸ | affects error bar (F7) |
| 4f. cleaning threshold vs noise above 60.1 | drops 2 points above 60.1, both with recorded corrections; keeps one with a 1.7×10⁻¹⁵ error | cosmetic |
| 4g. planted test: truth made by the code under test | issue; an independent truth passes (R6) | cosmetic |
| 5. γ (det root) | read only; no issue found | — |

## Issues, ranked

**F1 (affects the error bar). The selection rule is fragile, and was chosen after seeing the spread.**
`dinf_budget.py`'s first version (commit 3f28532) used every degree that reaches the floor; ten minutes later (23946ff)
it used the two lowest, with the explanation that higher degrees "wander by 10⁻⁷ to 10⁻⁶". The README's bar was set
from the second version (32e4317). The explanation is reasonable (higher degrees add freedom only at the unconstrained
end), but the bar is then a property of the rule, not of the data, and it does not respond to the data monotonically
(R5, `rv2_fits.py rules`, `variants`):

| data, rule | fits | range of the limits | half-width |
|---|---|---|---|
| published data, two lowest (quoted) | 48 | 1.64389374444 – 1.64389376313 | 9.3×10⁻⁹ |
| published data, lowest only | 24 | 1.64389374524 – 1.64389376313 | 8.9×10⁻⁹ |
| published data, three lowest | 70 | 1.64389364807 – 1.64389377857 | 6.5×10⁻⁸ |
| published data, every degree at the floor | 94 | 1.64389318986 – 1.64389627808 | 1.5×10⁻⁶ |
| published data, two lowest, degrees 13–21 odd / 14–22 all / 12–22 all / 16–22 even | 48 | 1.6438937435 – 1.6438937699 | 0.9–1.3×10⁻⁸ |
| published data, two lowest, floor factor 1.5–10 instead of 2.5 | 48 | 1.64389374444 – 1.6438937616–63 | 0.9×10⁻⁸ |
| runs' own flags: 168 points with \|F\| > 10⁻²⁰ or \|dDelta_est\| > 10⁻¹⁶ removed, two lowest | 48 | 1.64389361982 – 1.64389381321 | 9.7×10⁻⁸ |
| 27 ≤ D ≤ 37 removed, two lowest | 48 | 1.64389374196 – 1.64389374830 | 3.2×10⁻⁹ |
| corrected by the recorded dDelta_est, two lowest | 48 | 1.64389374439 – 1.64389376156 | 8.6×10⁻⁹ |

The bar is robust to the degree set and to the floor factor; it is not robust to how many degrees count, nor to the
data's quality. *Failing scenario:* remove the points the runs themselves flag as least converged. The max-residual floor drops from
4×10⁻¹⁵ to 1–8×10⁻¹⁶, the floor test now admits degrees 20–22 on D ≥ 15, whose limits wander by 10⁻⁷, and the bar
grows tenfold although the data got better. What keeps the published bar at 10⁻⁸ is that the floor is set at 4×10⁻¹⁵
by a few dirty points (the kfreeze-cut points at 35.35–37 and the uncleaned up6 points), which stops the degree
escalation at 16–20.

**F2 (affects the error bar). The top of the quoted range is an under-resolved fit.** The floor test uses the largest
residual. On uncleaned data that is set by a junk point (9.5×10⁻¹⁵), so a fit with residual up to 2.4×10⁻¹⁴ passes.
The fit "D 15..60.1, uncleaned, 1/(D−2), degree 14" has residual 2.1×10⁻¹⁴ and passes; its cleaned twin has the same
residual and is rejected (floor 4.0×10⁻¹⁵). Its limit, 1.643893763, is the top of the quoted range. With factor 2
instead of 2.5 the range is 1.64389374444–1.64389376158. The bar is conservative because of it, so this does not make
the bar too small; but the centre (1.6438937538) is pulled up by it.

**F3 (affects the value, within the bar). The centre is about 0.8×10⁻⁸ high.**

- `planted_dinf.log`: the centre lands 0.79–0.88×10⁻⁸ above the planted limit in all six plants.
- R6 (`rv2_planted.py`), with a truth the fitting code did not make: +0.78×10⁻⁸ with the real residual pattern
  transplanted, +0.09×10⁻⁸ with iid noise, +1.1×10⁻⁸ with a coherent 3×10⁻¹⁵ added above D = 60.1.
- In every plant the truth sits within 3×10⁻⁹ of the *bottom* of the range.
- On the real data, removing 27 ≤ D ≤ 37 gives the centre 1.6438937451; the degree-18 1/D fit to all cleaned points
  gives 1.6438937447 (`planted_dinf.log`, "P(0)").

The real centre is therefore probably near 1.643893746, not 1.6438937538. That is inside the bar, so the quoted
statement is not wrong, but the last digit of 1.64389375 is the high side of the evidence.

**F4 (precision; cosmetic for Δ∞). The fits do not see the double-double values.** `dd_fits_mp.load_points` (the
default CSV path, line 57) does `mp.mpf(c[1]) + mp.mpf(c[2])`. `export_dd.py` writes `repr(hi)`, the shortest string
that round-trips to the double, but its decimal value differs from the double by up to half an ulp. Parsing it at 60
digits keeps that difference: every point moves by up to 2.1×10⁻¹⁶ (rms 5.0×10⁻¹⁷; above D = 45 rms 3.8×10⁻¹⁷,
ten times the real scatter). `--scan=` reads JSON floats and is exact. Fix: `mp.mpf(float(c[1])) + mp.mpf(float(c[2]))`.

- Effect on the quoted range: 5×10⁻¹¹ (R5 with `--decimal`, which emulates the loader).
- It is why the repository's cleaning drops D = 10.75 on the D ≤ 60.1 snapshot (26 points, README) where exact data drop
  25: that point sits 0.99×10⁻¹⁵ off its neighbours in exact arithmetic and 1.09×10⁻¹⁵ after the parse.
- It reaches `dinf_budget.py`, `planted_dinf.py` and `reproduce.py`, which all load through it.

**F5 (cosmetic). The cross-machine check is quoted at f64 resolution.** No two runs share a D (R4), and no script
records how "agrees to 2–4×10⁻¹⁶" was measured. 2–4×10⁻¹⁶ is one to two ulps of Δ ≈ 1.81, so the comparison was made in
f64 and could not have resolved anything smaller. In exact arithmetic the 13 local points at D = 59.84–60.09 lie
7.2×10⁻¹⁸ rms (max 1.9×10⁻¹⁷) from a degree-6 interpolant of the cloud points, the same as the cloud points'
own leave-one-out scatter (6.2×10⁻¹⁸ rms) (R3). The machines agree 30 times better than claimed.

**F6 (cosmetic, but it bounds what the fits can do). The error per point is not "about 10⁻¹⁶" everywhere below 60.1.**
Run up6 (D = 26.90–35.20, `--junk_only`) accepted 149 points at |F| up to 1.7×10⁻¹¹ whenever the linearised correction
was below 10⁻¹⁴. Their errors are real: the leave-one-out residuals match the recorded dDelta_est (−8.1×10⁻¹⁵ vs
−8.1×10⁻¹⁵ at D = 34.898, +1.0×10⁻¹⁴ vs +9.1×10⁻¹⁵ at 32.338, R3). The cleaning drops 24 of them; the rest carry up to
5×10⁻¹⁵ (leave-one-out rms 1.2×10⁻¹⁵ after cleaning). The README describes this ("a few 10⁻¹⁵ … 25 of them sit visibly
off"), so the issue is only that the headline sentence "about 10⁻¹⁶ to D = 60.1" omits it. The runs' own corrections
could replace the cleaning: adding dDelta_est moves the quoted range by at most 1.6×10⁻⁹.

**F7 (affects the error bar, small). The 94 points above 60.1 carry the most weight.** The limit is a linear functional
of the data, Δ∞ = Σ wᵢΔᵢ. For the used fits ‖w‖₁ = 5×10⁵–8×10⁷, and Σ|wᵢ| over D > 60.1 is 7×10⁴–6×10⁶ (R5,
`influence`). A coherent error of 3×10⁻¹⁵ on those points (the size of the largest recorded corrections there) moves
the limit by up to 1.1×10⁻⁸, comparable to the bar. Their actual scatter is 3.9×10⁻¹⁶ rms with corrections of mixed
sign (39 positive, 55 negative, mean 1.7×10⁻¹⁷), and adding the corrections moves the range by ≤ 1.6×10⁻⁹, so this is
not happening; but the fits are unweighted although these points are 100 times noisier than their neighbours below
60.1. The one coherent error I found there, the kfreeze cut (−1.2×10⁻¹⁵, F9), is smooth across 40–61 and moves Δ∞
by at most 5.5×10⁻¹⁰.

**F8 (cosmetic). Series tails are checked after a point is accepted.** `scan_hp.py` raises kc or ks for the *next*
point when the tail exceeds 10⁻³³, but records the current one. Two kept points have bad centre tails: D = 23.952249
(2×10⁻¹⁵) and 26.82972 (2×10⁻¹¹, ×4×10⁻⁴ centre sensitivity ≈ 10⁻¹⁴). Their leave-one-out residuals are 7×10⁻¹⁴
(spacing 0.13, interpolation-limited) and 1.4×10⁻¹⁶, so no harm was done, and removing 26.82972 moves the range by
10⁻¹¹; but the check cannot fail for the point it is about.

**F9 (cosmetic for Δ∞; a quoted check is wrong). The kfreeze-13 cut is a smooth bias of about 10⁻¹⁵ on every point
from D = 35.35 on, not 3×10⁻¹⁷.** The README: "harmonics frozen above 13 / above 15, D = 40: Δ agrees to 3×10⁻¹⁷",
and "the cut moves Δ by 5×10⁻¹⁵ at D = 35 and 3×10⁻¹⁷ at D = 40 … which falls off with D". Re-converging saved
kfreeze-13 states with harmonic 15 free (kfreeze 15; R2):

| D | Δ(kfreeze 15) − Δ(kfreeze 13) | \|F\| reached | remaining linearised correction |
|---|---|---|---|
| 34.6–35.2 | +5.6 to +7.5×10⁻¹⁵ (unfrozen − kfreeze 13, from the runs' restarts; README −5.5×10⁻¹⁵ for 13 − unfrozen) | | |
| 40.020065 | **+1.52×10⁻¹⁵** | 1.4×10⁻²⁶ | −7×10⁻²³ |
| 50.007491 | **−1.06×10⁻¹⁵** | 1.1×10⁻²⁴ | +1×10⁻¹⁹ |
| 60.001952 | **−1.20×10⁻¹⁵** | 6.6×10⁻²¹ | +1×10⁻¹⁷ |

Freeing harmonic 17 as well gives the same linearised change at D = 60 (−1.2×10⁻¹⁵), so harmonic 15 carries the cut.
The effect changes sign near D ≈ 45 and stays at about 10⁻¹⁵ to the top of the range: 50 times the quoted 3×10⁻¹⁷ at
D = 40, and not falling off.

*Failing scenario of the README's check:* at D = 60, plain Newton with line search from the kfreeze-13 state in the
kfreeze-15 layout finds no descent direction, takes no step, and returns Δ unchanged: "agrees to 0". The first kfreeze-15
run in R2 did exactly that. It needed full steps without line search to converge. I could not find a script for the
D = 40 check; a run that stopped without stepping would report agreement at the 10⁻¹⁷ noise level, as quoted.

*Effect on Δ∞:* correcting every kfreeze-13 point by the measured cut (linear in D between the table's points,
`rv2_fits.py variants --kfcut=…`) moves the 48-fit range to 1.64389374499–1.64389376335, at most 5.5×10⁻¹⁰. The cut is
smooth, so the polynomial fits absorb most of it. It matters for the per-point error statement (10⁻¹⁵, not 10⁻¹⁶,
from D = 35.35 on), not for Δ∞.

## 1. The solver

### 1a. The equations

`model.rs` integrates, with p, m odd and a, c even under τ → τ + Δ/2 and ω = 2π/Δ,

```math
x c_x = G,\quad 2x a_x = H,\quad x(x-c)\,p_x = -\omega x\,p_\theta + G p - c p + \tfrac n2 c (p+m),\quad
x(x+c)\,m_x = -\omega x\,m_\theta - G m + c m - \tfrac n2 c (p+m),
```

with G = (n−1)(ca² − c) and H = a(p² + m²) + (n−1)(a − a³). Multiplying the equations of `solver/dss.py`'s header by x
gives exactly these. I checked each recursion against them by hand:

- **Centre series.** At order k ≥ 2 the unknowns enter as c₀[(1−k−n/2) P_k − (n/2) M_k] and c₀[(n/2) P_k + (k−1+n/2) M_k];
  the determinant is −c₀²(k−1)(k−1+n) ≠ 0. A_k = (s/2 − (n−1)t₃/2)/(k+n−1) and k C_k = (n−1)Σ_{j<k} C_j (A²)_{k−j}
  follow from the a and c equations with A₀ = 1. The x(x∓c)p_x sums, the x c_x p sums and the excluded P_k, M_k
  terms are right.
- **Taylor step about x₀.** c_{k+1} = (G_k − k c_k)/(x₀(k+1)), a_{k+1} = (H_k/2 − k a_k)/(x₀(k+1)), and
  W₀(k+1)p_{k+1} = r_k − Σ_{j=1}^{k} W_j(k+1−j)p_{k+1−j} with W = x(x−c) = x₀² + 2x₀h + h² − (x₀ + h)c; the same for m with
  V = x(x+c).
- **Horizon series in u = 1 − x.** W = (1−u)(1−u−c) has W₀ = 0 (c₀ = 1), so the order-k p equation determines p_k through
  (ωD + κ_k)p_k = −r_k with κ_k = (c₁+1)k + c₁ + 1 − n/2 and c₁ = (n−1)(1 − a_h²), which is `horizon_lus`. The m equation
  has Z₀ = 2 and gives m_{k+1} explicitly. Tangents: f_d = A⁻¹(r_d − ω_d D f − κ_d f) is the derivative of A f = r.

R1 confirms all of it numerically (§1e).

### 1b. Boundary conditions

- Centre: P₀ = M₀ = 0, P₁ = −M₁ = p₁, A₀ = 1, C₀ = c₀ free. The regular series is unique given (p₁, c₀), since every
  order's 4×4 system is non-singular.
- Light cone: C₀ = 1 (the gauge c(τ, 1) = 1), m and a free (m_h, a_h), and p fixed by analyticity in u. For real
  antiperiodic fields ωD + κ is never singular (its kernel would need exp(−∫κ/ω) = −1 over half a period).
- Phase: Σⱼ sin(θⱼ) p₁(θⱼ) = 0.

R1 checks, on the kernel's own series at D = 60 and 61.03, that the leading data are exactly these and that the
equations hold order by order (§1e). **Verdict: OK.**

### 1c. Collocation and parity folding

`dmat_odd` is the even-N Fourier differentiation matrix ½(−1)^{i−j} cot((i−j)π/N), folded with f(θ + π) = −f(θ). For odd
fields only odd harmonics occur, and the Nyquist mode k = M is even (M is a multiple of 8), so the even-N ambiguity
does not arise. a and c are never τ-differentiated. Products are pointwise, so parity is kept (odd × odd = even,
even × odd = odd) and there is no dealiasing: the system is pure collocation, whose convergence in M is a separate
question (§1h). The freeze projector `plow_odd` is the interpolation projector onto odd k ≤ kf.

R1 builds d/dθ from the definition, −(4/N)Σ_{k odd < M} k sin(k(θᵢ − θⱼ)), not from the cot formula; its residuals agree
with the kernel's to 2×10⁻³¹, so the folded matrix and the parity handling are right. **Verdict: OK.**

### 1d. Double-double arithmetic

`dd.rs` and `dv.rs` are the QD library's algorithms: two-sum (6 flops), FMA two-product (`fms` = a·b − p with one
rounding, exact), the accurate ("IEEE") add, the multiply without the lo·lo term, and the three-quotient divide.
`DotAcc` is Ogita–Rump–Oishi Sum2 with exact products: the error terms accumulate in plain f64, which costs only a
relative 2⁻¹⁰⁴ × (sum of |terms|). All three ISA paths use fused operations (the portable one through `f64::mul_add`,
which is exact in software where the hardware lacks FMA). The DD constant 2π is correctly split (lo = 2.4492935982947064×10⁻¹⁶).
n = D − 2 is formed in mpmath from the float D, so it is exact. `Du` (dual numbers) has the right quotient rule.
**Verdict: OK.**

**Where precision drops to f64** (all in the Python driver, none in the residual):

- the Lagrange predictor (`predict`), the junk hold and trend, `freeze_p1` at a run's start, `resample_z`, the tangent;
- `deflated_newton` collapses each step to f64 (`d = dh + dl`) before projecting out the weak directions;
- the line search measures |F|₂ in f64.

All of these feed Newton's iterate, not its fixed point: a point accepted by plain Newton (or by the dnp polish) is a
zero of the double-double residual whatever the start. Only the junk_only acceptance (run up6) stops on a deflated
iterate (§1k). **Verdict: OK.** The precision drop that does matter is in the fits' loader (F4).

### 1e. Independent residual (R1)

`rv2_residual.py` evaluates F(z) at a saved state from scratch in python-flint at 200 bits:

- τ-derivative from the DFT definition (above);
- centre and horizon blocks: the kernel's series (`Solver.series`) are *checked*, not recomputed. Each τ-point's four
  series become `arb_poly`s, the leading data are compared with p₁, c₀, m_h, a_h, and the four equations (in x, or in
  u = 1 − x) are evaluated as polynomials and must vanish through order kc (ks − 1). Then the series are summed at the node;
- Taylor blocks: re-integrated by Picard iteration on power series (`arb_series`, order 40), with every kernel step split
  in two; midpoints only (balls overestimate by tens of orders through the τ mixing);
- the freeze projector and the phase row.

At D = 60.001952 (`up13_D60.0020`):

| block | max \|F_kernel\| | max \|F_indep\| | max \|diff\| | −y·F kernel | −y·F independent |
|---|---|---|---|---|---|
| 0 centre | 1.39×10⁻²³ | 1.39×10⁻²³ | 7×10⁻³³ | +1.5×10⁻³⁴ | +1.5×10⁻³⁴ |
| 1 (0.3→0.4) | 3.31×10⁻²⁵ | 3.31×10⁻²⁵ | 1.6×10⁻³¹ | −5.7×10⁻³⁴ | −5.7×10⁻³⁴ |
| 2 (0.4→0.5) | 2.43×10⁻²⁶ | 2.43×10⁻²⁶ | 1.9×10⁻³¹ | +9.4×10⁻³⁰ | +9.8×10⁻³⁰ |
| 3 (0.5→0.6) | 9.86×10⁻²⁸ | 9.86×10⁻²⁸ | 1.9×10⁻³¹ | −2.0×10⁻²⁴ | −1.4×10⁻²⁴ |
| 4 (0.6→0.7) | 3.47×10⁻²⁹ | 3.47×10⁻²⁹ | 1.4×10⁻³¹ | +4.9×10⁻¹⁸ | +6.1×10⁻¹⁸ |
| 5 horizon | 2.32×10⁻²³ | 2.32×10⁻²³ | 1.7×10⁻³² | −3.7×10⁻¹⁸ | −2.3×10⁻¹⁸ |

The series satisfy the equations to 7×10⁻³² (centre, coefficient k weighted by x₀^k, all 421 orders) and 1.4×10⁻³⁰
(horizon, weighted by u₀^k), with the leading data exact. y = J⁻ᵀe_Δ is the kernel's. Summed over the blocks the
linearised correction is +1.2×10⁻¹⁸ (kernel) and +3.8×10⁻¹⁸ (independent).

The second machine's last state, `seeds/local/up23_D61.0264` (D = 61.026449, its own Windows build), gives the same
picture: blocks agree to 2.3×10⁻³¹, block maxima 4.7×10⁻²⁴ (centre) and 3.8×10⁻²³ (horizon), series equations to
1.1×10⁻³¹ and 2.9×10⁻³⁰, linearised correction +1.0×10⁻¹⁹ (kernel) and +1.1×10⁻¹⁸ (independent). That is an
independent check of the other machine's point at the top of the range: it is a solution of the discretised
equations to the stated residual, and its Δ is within about 10⁻¹⁸ of the discrete solution.

So:

- the residual the kernel reports is the true residual of the saved z to 2×10⁻³¹; the floor |F| ≈ 2×10⁻²³ is real
  (Newton's linear solves stop reducing it), not evaluation noise;
- the kernel's own evaluation error, projected on y, moves Δ by about 10⁻¹⁸, and the linearised distance of the saved
  point from the exact discrete solution is 10⁻¹⁸ by both evaluations.

**Away from the solution** (`--perturb 1e-6`: every node field moved by 10⁻⁶ sin(3θ + q), Δ by 10⁻⁶), the blocks
checked (centre, first segment, horizon) have |F| = 1.0×10⁻⁶, 2.9×10⁻⁶ and 7.7×10⁻², and the two evaluations agree
to 6.8×10⁻³³, 1.4×10⁻³¹ and 1.3×10⁻³²: relative agreement 10⁻²⁵–10⁻³¹, so every term of the equations is the same
in both codes (a wrong coefficient would show at the 10⁻⁶ level).

**Negative controls.** They fire, and they show which part of R1 is independent:

- changing the n/2·c(p + m) coupling by a relative 10⁻¹⁵ in my code only: segment 1 now disagrees by 6.7×10⁻¹⁸
  (against 1.6×10⁻³¹);
- adding 10⁻²⁴ to Δ in my code only: the horizon *block* still agrees (1.7×10⁻³²), because that block sums the
  kernel's own series; what fires is the series check, whose worst equation residual rises from 1.4×10⁻³⁰ to
  7.3×10⁻²⁵. For the centre and horizon blocks the independent part of R1 is therefore the order-by-order check of
  the series, not the comparison of the block residuals; for the Taylor blocks both are independent.

### 1f. The f64 parts of the driver

See §1d. Newton (`newton_solve`) records |F| *before* each step and breaks without stepping once |F| < tol, so for a
converged point the recorded F is the residual of the stored z. With the dnp polish, the last record is the residual
before the last polish step (the stored z is at least as good, by the line search); the stored dDelta_est is
recomputed at the stored z. **Verdict: OK.**

### 1g–1j. Resolution and robustness at D = 60 (R2)

`rv2_resolve.py` re-converges `up13_D60.0020` (Δ = 1.809884258001715 − 1.091×10⁻¹⁶) with one thing changed, by plain
Newton to the floor:

| variant | \|F\| | Δ − saved Δ | linearised correction |
|---|---|---|---|
| same settings | 1.8×10⁻²³ | +3.2×10⁻¹⁸ | +2.4×10⁻¹⁸ |
| Taylor order 40, step tolerance 10⁻³⁴, h ≤ 0.025 (16 steps instead of 11) | 1.6×10⁻²³ | +2.1×10⁻¹⁸ | −1.9×10⁻¹⁸ |
| centre series 520 terms, horizon series 150 | 1.8×10⁻²³ | +1.2×10⁻¹⁸ | −3.1×10⁻¹⁸ |
| first node 0.25 instead of 0.3 | 2.8×10⁻²³ | +6.4×10⁻¹⁸ | −1.1×10⁻¹⁷ |
| p₁ harmonic 13 kicked by 10⁻⁸, re-converged | 1.9×10⁻²³ | +2.9×10⁻¹⁸ | −9.0×10⁻¹⁹ |
| p₁ harmonic 11 kicked by 10⁻⁸ | 2.1×10⁻²³ | −3.6×10⁻¹⁸ | +6.1×10⁻¹⁸ |
| p₁ harmonic 9 kicked by 10⁻⁸ | 2.4×10⁻²³ | −1.1×10⁻¹⁸ | +6.2×10⁻¹⁸ |
| last node 0.75 instead of 0.7 | 1.4×10⁻²³ (no step accepted) | — | −1.5×10⁻¹⁶ |
| kfreeze 15 instead of 13 | 2.3×10⁻²³ (no step accepted) | — | −1.2×10⁻¹⁵ |
| kfreeze 17 | 2.3×10⁻²³ (no step accepted) | — | −1.2×10⁻¹⁵ |
| M = 48 τ-points (resampled in double-double; deflated Newton, then Newton) | 1.7×10⁻²³ | +2.5×10⁻¹⁸ | +6.5×10⁻¹⁸ |
| kfreeze 15, full Newton steps without line search, then Newton | 6.6×10⁻²¹ | **−1.20×10⁻¹⁵** | +1.3×10⁻¹⁷ |
| M = 48 interpolated in f64 (`resample_z`), plain Newton | 6.0×10⁻⁸ (diverged) | −7.0×10⁻⁸ | |

- **Taylor stepping, series truncation, first node: OK.** Their effect is at the 10⁻¹⁸ floor.
- **Junk handling: OK.** Kicks of 10⁻⁸ along the three weakest directions come back to within 6×10⁻¹⁸, so the predictor
  options (`--junk_hold`, `--junk_trend`) and the corrector's starting point do not bias Δ once a point is iterated to
  the floor.
- **Last node 0.75:** the re-noded start is already at the floor and Newton finds no descent; the horizon block's
  sensitivity rises from 9×10¹³ to 3×10¹⁷ with u₀ = 0.25, and the linearised correction is −1.5×10⁻¹⁶. Inconclusive
  at the 10⁻¹⁶ level, consistent with "about 10⁻¹⁶".
- **kfreeze (F9).** With line search, Newton in the kfreeze-15 (and 17) layout takes no step from the kfreeze-13
  point, so Δ "agrees exactly"; the linearised correction says −1.2×10⁻¹⁵. Full steps without line search converge
  (|F| 6.6×10⁻²¹) and move Δ by −1.20×10⁻¹⁵, as linearised. The same at D = 50 and 40 is in F9.
- **M = 48.** Interpolated in f64 (`driver.resample_z`, which drops lo), the start carries 10⁻¹⁶ errors into the
  weak directions and plain Newton runs away (Δ off by 7×10⁻⁸ after 8 steps, |F| 6×10⁻⁸): that is the expected
  failure, and it shows the test can move Δ. Interpolated in double-double (`rv2_resolve.py`'s `resample_dd`, round
  trip 32 → 48 → 32 exact to 1.2×10⁻³²) and converged with the scans' corrector (deflated Newton, then Newton), M = 48
  reaches the floor and moves Δ by +2.5×10⁻¹⁸. PENDING_M48START

### 1k. junk_only acceptance and the up6 points

`deflated_newton(junk_only=True)` accepts a stalled iterate when |−∂Δ/∂F·F| < `--delta_tol` = 10⁻¹⁴. So by design the
up6 points can be wrong by about 10⁻¹⁴, which they are (F6). This is documented in the README's footnote; it is listed
here because the acceptance test is the only place where a non-converged iterate enters the data.

## 2. The error per point

### 2a. What the runs record

Each record carries |F|, the per-block max |∂Δ/∂F| and dDelta_est = −∂Δ/∂F·F, the linearised distance to the exact discrete
solution (R4, `rv2_data.py`):

| D | points | max \|F\| | max \|dDelta_est\| | worst case \|F\|·max\|∂Δ/∂F\| |
|---|---|---|---|---|
| 8.25–27 | 146 | 2.4×10⁻¹⁶ | 7.3×10⁻¹⁶ (15 recorded) | 1.5×10⁻⁹ |
| 27–35.25 (up6) | 146 | 1.7×10⁻¹¹ | 9.1×10⁻¹⁵ | 0.2 |
| 35.25–45.34 | 106 | 1.0×10⁻²⁵ | 2.3×10⁻²⁰ | 1.5×10⁻¹⁴ |
| 45.34–60.1 | 365 | 2.6×10⁻²² | 1.1×10⁻¹⁷ | 2.2×10⁻⁸ |
| 60.1–61.03 | 94 | 2.9×10⁻¹⁹ | 2.8×10⁻¹⁵ | 3.8×10⁻⁵ |

The worst-case product is meaningless (it ignores the direction of F); dDelta_est is the relevant number, and R1
confirms it at D = 60 and 61.03 with an independent residual.

### 2b. Is dDelta_est the error? (R3)

`rv2_noise.py` fits, for each point, a degree-6 polynomial in D to its 12 nearest neighbours (excluding it) in exact
arithmetic. For a smooth function plus independent errors the residual measures the point's own error. Where the
recorded correction is large, the two agree in sign and size:

| D | run | leave-one-out residual | dDelta_est |
|---|---|---|---|
| 29.348206 | up6 | +5.1×10⁻¹⁵ | +5.4×10⁻¹⁵ |
| 32.338234 | up6 | +1.0×10⁻¹⁴ | +9.1×10⁻¹⁵ |
| 33.576644 | up6 | +4.5×10⁻¹⁵ | +4.4×10⁻¹⁵ |
| 34.441550 | up6 | +9.2×10⁻¹⁵ | +8.5×10⁻¹⁵ |
| 34.898262 | up6 | −8.1×10⁻¹⁵ | −8.1×10⁻¹⁵ |
| 60.882916 | local-up15 | +9.4×10⁻¹⁶ | +8.3×10⁻¹⁶ |
| 60.925947 | local-up16 | +1.4×10⁻¹⁵ | +1.3×10⁻¹⁵ |
| 60.962646 | local-up17 | +3.5×10⁻¹⁵ | +2.8×10⁻¹⁵ |

(32 points with |dDelta_est| > 3×10⁻¹⁶ in the log; a few disagree in sign at the 10⁻¹⁵ level, e.g. 60.952991, +1.7×10⁻¹⁵
against −1.2×10⁻¹⁵.)

A second, direct confirmation comes from the runs themselves (`rv2_restarts.py`). Every run re-converges its start and
history states and prints Δ (hi, 16 decimals). For 65 of them the D is an earlier run's point:

- same layout, both converged to the floor: identical to the printed digit, or one ulp (2.2×10⁻¹⁶) apart;
- the cloud re-converging the second machine's 60.962646 (accepted at |F| 2.9×10⁻¹⁹): +3.3×10⁻¹⁵, against its recorded
  dDelta_est +2.8×10⁻¹⁵; and 60.959857: −8.9×10⁻¹⁶ against −1.0×10⁻¹⁵;
- layout changes: M 48 / 10 nodes → M 32 / 8 nodes at D = 16.75 and 8 → 5 nodes at 21.35, no change at the printed
  digit; the kfreeze-13 cut switched on at 34.6–35.2: −5.6 to −7.5×10⁻¹⁵ (the README's −5.5×10⁻¹⁵).

**Verdict: dDelta_est is a faithful per-point error estimate.** It is a better tool than the cleaning.

### 2c. The scatter by band (R3)

| D | leave-one-out rms (max), cleaned points | global 1/D degree-18 fit residual rms (max) |
|---|---|---|
| 27–35.25 | 1.2×10⁻¹⁵ (5.1×10⁻¹⁵) | 7.4×10⁻¹⁶ (1.7×10⁻¹⁵) |
| 35.25–37 | 1.6×10⁻¹⁵ (2.6×10⁻¹⁵) | 3.1×10⁻¹⁵ (4.3×10⁻¹⁵) |
| 37–45 | 8.6×10⁻¹⁶ (2.8×10⁻¹⁵) | 5.6×10⁻¹⁶ (1.7×10⁻¹⁵) |
| 45–55 | 7.1×10⁻¹⁸ (4.3×10⁻¹⁷) | 1.5×10⁻¹⁶ (2.8×10⁻¹⁶) |
| 55–60.1 | 3.0×10⁻¹⁸ (1.0×10⁻¹⁷) | 1.0×10⁻¹⁶ (1.5×10⁻¹⁶) |
| 60.1–61.03 | 3.9×10⁻¹⁶ (1.7×10⁻¹⁵) | 2.3×10⁻¹⁶ (1.2×10⁻¹⁵) |

Below D = 27 the spacing (0.25) makes the local interpolant itself the limit (10⁻¹⁰), so that band says nothing. At
37–45 (spacing 0.1) the leave-one-out figure is probably interpolation-limited too. The global fit's residuals at
45–60.1 (10⁻¹⁶) are its own truncation, not noise: the local scatter there is 30 times smaller. The 35.25–37 band is the
kfreeze-cut bias: a smooth 3×10⁻¹⁵ that the leave-one-out test hardly sees and the global fit does. This band, not
noise, sets the fits' 4×10⁻¹⁵ floor.

**Verdict on the README's sentence** ("about 10⁻¹⁶ to D = 60.1, a few 10⁻¹⁵ above"): as point-to-point noise it is
conservative for 45–60.1 (3–7×10⁻¹⁸; the Taylor, series, node and M tests at D = 60 agree) and right above 60.1. As
the error against the true solution it is low by ten from D = 35.35 on, where the kfreeze cut adds a smooth
10⁻¹⁵ that neither the noise tests nor dDelta_est can see (F9), and incomplete for 27–37 (up to 10⁻¹⁴ before cleaning,
5×10⁻¹⁵ after; documented further down the README). The two-machine agreement is 7×10⁻¹⁸ (F5).

## 3. The data pipeline

`rv2_data.py` rebuilds the merge from the runs' JSON files (cloud runs in `/home/claude/dss/hp_scan` in name order,
then `hp/data/local`), written from `export_dd.py`'s docstring, and compares (R4):

- **A.** 857 CSV rows = 857 distinct D in the runs, every (D, hi, lo, run) identical. **OK.**
- **B.** Every D equals round(D, 6) (so `dd_fits_mp`'s keys are the floats the solver ran at, and n = D − 2 is exact);
  no two D closer than 2.3×10⁻⁴. **OK.**
- **C.** |lo| ≤ ulp(hi)/2 and hi + lo = hi in f64 for every row. **OK.**
- **E.** No D appears in two runs, so "the local one is kept" never applied, and there is no pair to compare directly (F5).

`--break=merge`, `--break=round` and `--break=lo` make A, B and C fail. The loader issue is F4. Re-noded records
(`renoded: true`) would share a D with the point before them; none occur in the runs used.

## 4. The fits and the error bar

### 4a. Independent reproduction (R5)

`rv2_fits.py` refits in python-flint at 640 bits, with a monomial basis in the scaled variable instead of Chebyshev
polynomials, an `arb_mat` solve, the exact double values, and the cleaning re-implemented from its docstring. Its
self-test recovers a planted degree-12 polynomial's limit to 10⁻¹⁷³ at every degree 14–22, and fails as it must when the
degree is too low (7.9×10⁻¹⁰) or one point is moved by 10⁻¹² (1.6×10⁻¹⁰). The 48-fit selection gives

```math
1.64389374444 \le \Delta_\infty \le 1.64389376313 \quad (\text{centre } 1.64389375379,\ \text{half-width } 9.3\times10^{-9}),
```

the README's 1.643893744–1.643893763. One fit compared at full length: the degree-18 1/D fit to the 823 cleaned points
on 10–61.03 gives 1.64389374468460 with residual 4.305×10⁻¹⁵ when the repository's loader is emulated (`--decimal`),
and `planted_dinf.log` prints P(0) = 1.6438937446846, residual 4.31×10⁻¹⁵: every printed digit agrees. With the exact
values it is 1.64389374467885 (F4). **Verdict: OK.**

### 4b. How the limit depends on each point (R5, `influence`)

Δ∞ = Σ wᵢΔᵢ with w = X(XᵀX)⁻¹e₀. For the 48 used fits ‖w‖₁ = 5×10⁵–8×10⁷ and ‖w‖₂ = 2×10⁴–3×10⁶. Consequences:

- independent noise of 10⁻¹⁶ (D ≤ 60.1) and 3×10⁻¹⁵ (above) moves the limit by 2×10⁻¹² to 2×10⁻⁹;
- a coherent 10⁻¹⁵ on the kfreeze-cut points 35.35–37 moves it by up to 1.1×10⁻⁹;
- a coherent 3×10⁻¹⁵ above 60.1 by up to 1.1×10⁻⁸ (F7);
- 10⁻¹⁴ on the bad-tail point D = 26.83 by up to 1.3×10⁻⁹.

So the bar is not limited by noise; it is limited by model choice and by smooth errors at the 10⁻¹⁵ level, and only
coherent errors of a few 10⁻¹⁵ at the top of the range could reach it.

### 4c. The selection rule

See F1 and F2. In addition: the floor is a max-norm test, so it is driven by the single worst point of each group. The
groups without cleaning are therefore graded against junk, and the groups with cleaning against the kfreeze-cut band.
A test on the rms residual, or on a weighted residual using the recorded corrections as error estimates, would grade
every group against the data's actual precision.

### 4d. The cleaning

`clean()` (degree-18 global fit, residual minus the median of two neighbours on each side, threshold 10⁻¹⁵) drops 27 of 850
points on 10–61.03: 24 up6 points, D = 35.3532 (the first kfreeze-13 point) and two above 60.1 (60.9259, 60.9626, both
with recorded corrections of 1.3 and 2.8×10⁻¹⁵). It keeps 60.952991 (error 1.7×10⁻¹⁵) and up6 points with errors up to
5×10⁻¹⁵ that sit in clusters. With noise of 4×10⁻¹⁶ rms above 60.1 a 10⁻¹⁵ threshold is a 2.5σ cut there, so it drops
honest points and misses dishonest ones at about the same rate, which is what the planted test saw ("missed the junk
point at its threshold"). Replacing the cleaning with the recorded corrections (§2b) removes the threshold altogether;
it moves the quoted range by ≤ 1.6×10⁻⁹.

### 4e. The planted-truth tests (R6)

`planted_dinf.py` builds its truth from the degree-18 1/D fit to the real points: a fixture made by the code under test
and inside the fitted model class, with iid noise. It cannot detect a bias of the fitting procedure that comes from
the real function's shape. `rv2_planted.py` plants instead the README's rational formula P8/Q2 in 1/(D−2) (fitted to the
f64 points 4 ≤ D ≤ 30.2, limit a₀ = 1.6438975180 exactly, poles at D = −3.2 and −11.0, within 5.8×10⁻⁸ of the real points),
with three noise models, and runs the same 48-fit selection:

| plant | range about the truth | centre − truth | half-width | truth inside |
|---|---|---|---|---|
| N1: the real data's residual pattern (junk, kfreeze band, noisier top) | −1.8×10⁻¹⁰ … +1.58×10⁻⁸ | +7.8×10⁻⁹ | 8.0×10⁻⁹ | yes |
| N2: iid noise at the measured per-band levels | −1.7×10⁻⁹ … +3.6×10⁻⁹ | +0.9×10⁻⁹ | 2.6×10⁻⁹ | yes |
| N3: N1 plus a coherent +3×10⁻¹⁵ above 60.1 | −1.8×10⁻¹⁰ … +2.22×10⁻⁸ | +1.1×10⁻⁸ | 1.1×10⁻⁸ | yes |

**Verdict:** the coverage claim survives an independent truth; the centre bias (F3) is a property of the data's error
pattern, not of the truth, since N2 shows almost none.

### 4f. Failure classes

- *Checks that cannot fail:* the cross-machine comparison at f64 resolution (F5); the series-tail check after acceptance
  (F8); `dinf_budget.py`'s "consistency check vs dd_fits_mp.fit" prints a difference but asserts nothing.
- *Fixtures produced by the code under test:* `planted_dinf.py`'s truth (§4e); `reproduce.py`'s expected Δ∞ ± 10⁻⁸ is a
  regression value from the same code (fine as a regression test, not as a validation).
- *Silent passes:* the junk_only acceptance (§1k) by design; `export_dd.py` silently lets the last run win on a duplicate D
  (none occur); `gamma_dd.py` records skipped states as skipped (not silent).

## 5. γ

`gamma_dd.py` finds λ = 1/γ as the sign change of det M(λ), M the exact mode matrix in double-double (the Jacobian
with every τ-derivative of a perturbation shifted by λ, Δ fixed, no phase row), by `brentq` in f64 with xtol 10⁻¹⁴
inside a bracket around the rational formula's value, taking the sign change nearest to it and recording the number of
sign changes. I read it and found nothing wrong: the shifted horizon LUs (`lus_t`) and the λ term in `dtau` act on
tangents only, as they must; the kfreeze-13 cut is applied from D = 35, so γ is the eigenvalue of the same cut problem
as Δ. Limits: λ is an f64 root (10⁻¹⁴, enough for the quoted 10⁻¹²), and two roots inside one bracket would cancel in
the sign test (the bracket starts at ±3×10⁻⁴). I did not rerun it.

## 6. What I could not check

- **Analyticity in 1/D.** The bar assumes it (the README says so). Nothing here tests it beyond what the README's
  held-out tests do.
- **M convergence below D = 60** except PENDING_M_LIST; the README's own checks cover D = 7–17 and 40 (no script
  survives for the D = 40 ones), and the restart at D = 16.75 shows the M 48 → 32 change below the printed digit.
- **The kfreeze cut beyond harmonic 15–17**, and at D = 35.35–40 directly (I have the runs' restarts at 34.6–35.2 and
  my solves at 40, 50, 60). Taking kfreeze 15 as the truth assumes the harmonics above 15 matter less, which the
  kfreeze-17 linearisation supports at D = 60 only.
- **Points below D = 27 at the 10⁻¹⁶ level.** The spacing there (0.25) is too coarse for the leave-one-out test, and I
  ran no solves below D = 40; the layout-change restarts at 16.75 and 21.35 agree to the printed digit.
- **The runs' command lines.** kfreeze, tolerances and correctors are not stored in the JSON records; I took them from
  `comms.txt` and `solver/notes.md`. A record field would make the data self-describing.
- **The f64 solver**, the γ fits (`gamma_compare.py`) and the large-D theory are out of scope.
- **The second machine's build.** I evaluated its saved state with my own code (R1), not its binary.

## 7. Reproduce

From the repository root, with hpkernel built (`hp/build.sh`) and python-flint 0.9:

| | command | time on 2 busy cores |
|---|---|---|
| R1 | `python3 hp/review/rv2_residual.py hp/data/states/up13_D60.0020.npz` (`--perturb 1e-6`, `--break=delta`, `--break=eq`) | 7 min |
| R2 | `hp/review/rv2_batch.sh hp/data/states/up13_D60.0020.npz OUT.jsonl full`; `hp/review/rv2_batch2.sh` | 2–3 h |
| R3 | `python3 hp/review/rv2_noise.py` | 10 s |
| R4 | `python3 hp/review/rv2_data.py [/home/claude/dss/hp_scan] [--break=merge\|round\|lo]` | 2 s |
| R5 | `python3 hp/review/rv2_fits.py selftest\|budget\|influence\|rules\|variants\|clean [--decimal]` | 20 s – 2 min each |
| R6 | `python3 hp/review/rv2_planted.py` | 1 min |

Logs: `hp/review/rv2_*.log`, `rv2_resolve_D*.jsonl`, `rv2_residual.jsonl`. `rv2_runs.py` lists every run's settings.

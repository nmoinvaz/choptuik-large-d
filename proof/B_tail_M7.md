# Gap (b) continued: the M = 7 zero and the tail lemma

This note continues [B_tail.md](B_tail.md) (gap (b) of Conjecture 1, [PROOF.md](../PROOF.md) §4 A.3 and stage 4 of §7). It
computes the seven-harmonic zero, updates the truncation-sequence analysis with it, states the tail lemma for the
jump-conditioned truncation precisely, proves the parts of it that can be proved by hand, estimates every constant from
the data, and names the one inequality that needs a new idea. Written by Claude (Anthropic) under Nathan Moinvaziri's
direction; not reviewed by a mathematician. Status words as in AGENTS.md: *proved* (by hand), *numerical evidence*,
*conjecture*, *conditional* (hypothesis named).

Scripts (all new; nothing existing was changed):

- `solver/inner_stokes/inner_nb.py`: a compiled (numba) backend for `inner_np.Inner`: the same series, right-hand side
  and DOP853 integrator (scipy's algorithm transcribed), 15–20× faster at M = 7.
- `solver/inner_stokes/hier_nb.py`: `hier.py` unchanged, run on that backend, with a `--pretune` start and a lazy
  residual (only the radii whose entries `hier.py` reads are integrated while a level approaches).
- `solver/proof/trunc_seq_M7.py`: the truncation sequence, ratios, coupling and limit estimates.
- `solver/proof/amp_trunc_hiermodel.py`: the amplitude expansion of `hier.py`'s own M-harmonic model (§4).

## Summary

- **M = 7 is now affordable, but its zero was not obtained.** A compiled backend (equal to scipy's to scipy's own
  noise, 15–20× faster) and a lazy residual (no expensive arc at ξ = 0.2 while the top harmonic approaches; every entry
  `hier.py` reads is bit-identical) cut the first cold level-5 solve at M = 7 from more than 5 hours on the 8-thread
  machine to 6 minutes on a loaded cloud core (§1). The pilot, a fixed-(A, ω) solve at the M = 5 zero, tunes harmonics 5
  and 7 to their noise floors: F₅ moves onto the untruncated value (0.003104 + 0.000557i against 0.00310 + 0.00056i), and
  |F₇/F₅| = 0.0573 against the untruncated ρ_*² = 0.0574 (§2). Its level-3 Newton stalls at |r₃| ≈ 9, so r₁ is not
  resolved, c₇ is not known, and B_tail.md's prediction 0.99997–1.00017 is untested. PLACEHOLDER-DIAGSUM A local task
  (three jobs on 6 threads, about an hour) is proposed in §3.1.
- **A correction to B_tail.md §4.** `hier.py`'s model is a Galerkin truncation too; `amp_trunc.py`'s projection drops the
  even harmonic M + 1, which `hier.py` keeps. The amplitude expansion of `hier.py`'s own model (`amp_trunc_hiermodel.py`)
  is exact through σ_M, as order counting predicts, and changes σ₄ by 0.12% at M = 3 where `amp_trunc.py` gives −399. The
  lowest-order increments fall by 0.035 per pair (M = 1 → 3) (§4).
- **The tail lemma, stated** (§5): for consecutive jump-conditioned truncations, in X_σ and its dual, as a profile bound
  (T1), a coupling bound (T2) and consistency (T3), or in ratio form (R). A ratio κ ≤ ¼ per pair, enough with M = 7
  explicit, needs (T2) for some σ ≤ 0.736.
- **Proved by hand** (§6, §8.2): Lemma R (the Schur-corrected residual has exactly the zeros of the jump-conditioned
  multiplier, and the 10¹⁷ couplings of `hier.py` cancel exactly in its Schur complement; *conditional* on the jump
  determining the multipliers), Lemma O (order counting: the M-model is exact through σ_M, `amp_trunc.py`'s through
  σ_{M−1}), Lemma L (the lowest-order increment is linear in the tail), Lemma C (telescoping), identity H (the increment
  as an integral of a Schur-complement coupling), Lemma W (the weight e^{πνω/2} cancels exactly in every tuning step).
- **The inequality that needs a new idea** (§8.4): (★), a bound on the linearized coupling uniform in the harmonic, in
  X_σ* with σ ≤ 0.736; geometrically, a representation of the tuned linear response along paths within 5.5° of the Stokes
  line (the linear weights come from the anti-Stokes line at 45°). The data put the coupling at g₅ = 1.21 per pair
  (effective σ ≈ 0.1). The attempt through Lemma W stops at the conditional convergence of its Darboux sums.

## 1. Making M = 7 affordable

The README records that the first cold level-5 solve of M = 7 had not finished after 5 hours on the 8-thread machine.
Two causes, both in the cost of one residual, and two fixes that leave `hier.py`'s algorithm untouched.

- **Python overhead.** One M = 7 residual is one series (J ≈ 240 terms of a 25-component system) and four ray-plus-arc
  DOP853 integrations, ~10⁴ right-hand sides of ~20 µs of arithmetic each, wrapped in ~100 µs of `solve_ivp` and
  `np.convolve` overhead. `inner_nb.py` compiles the same series and the same right-hand side, and transcribes scipy's
  DOP853 (coefficients imported from scipy; initial step, error norm and step control as in `scipy/integrate/_ivp`). At
  M = 3 the two backends take exactly the same number of right-hand sides on every ray and arc (1706, 4322, 3290, 1442)
  and agree to 10⁻¹⁴. Through `hier.demixed_multi`, at the M = 1, 3, 5 points and the tuned M = 7 point of
  `logs/hier_M7_fixA_run2.log`, every residual entry that `hier.py` reads differs from scipy's by at most 3.6 times
  scipy's own noise (its change when rtol goes from 10⁻¹² to 0.8×10⁻¹²); a sign flipped in the compiled right-hand side
  makes the check fail by 10¹⁴ (`inner_nb.py check 1 3 5 7`, `--mutate=sign`; logs `inner_nb_check*.log`). The entries
  that `hier.py` does not read (r₁, r₃ at the approach radii, where e^{1/(2ξ²)} ≈ 10²⁰ amplifies rounding) differ by up
  to 100%, as they must. Speed under the cloud load of this session (load average 16–22 on 2 cores): 1.5 s against 25 s
  per M = 7 residual.
- **The arc at ξ = 0.2 while the top harmonic is untuned.** `hier.py` evaluates every residual at all radii from one
  integration. With F₇ off its tuned value by 1% (the amplitude-expansion seed), the harmonic-7 jump at ξ = 0.2 is large
  and nonlinear and the arc to the real axis there takes about 1.2×10⁶ right-hand sides (in either backend), against
  ~10⁴ near the tuned set. That is what made the first level-5 solve take hours. `hier_nb.py` does two things about it:
  `--pretune` solves S₇ = 0 alone first, integrating only to its approach and polish radii (15 s), and `LazyHier`, while
  a level ν ≥ 5 is in its approach mode, integrates only up to the largest radius whose entries `hier.py` can read (the
  approach radius of that level and the polish radii above it). The radii below that are `hier.py`'s own list, so the
  staged ray segments are the same and every entry read is bit-identical (`hier_nb.py selftest`, all four mode
  combinations; `--mutate=radius` fails by 290). The only visible difference is the "start residuals" line, which prints 0
  for the entries not integrated.

With both, the first cold level-5 solve at M = 7 took 362 s (229 residuals) under that load.

## 2. The M = 7 pilot: one fixed-(A, ω) solve at the M = 5 zero

Command (from `solver/inner_stokes/`; log `logs/hier_nb_M7_pilot_fixA_at_M5zero.log`):

```
python -u hier_nb.py 7 2.718579189229888 0.9992284031945021 --x=0.1492035749173264,0.01471905249087869,0.008405346906375517,0.0014703451817031647 --pretune --fixA --schur --predict --floor --procs 2 --verbose=2
```

It solves S₃ = S₅ = S₇ = 0 at the M = 5 zero p₅ = (2.718579189, 0.999228403), i.e. one level-3 solve with levels 5 and 7
nested. Timing under the cloud load of this session (2 cores, load average 16–22, ~12% of a core per process):

| stage | residuals | wall time |
|---|---|---|
| pretune of F₇ (approach radius 0.1029, then polish radius 0.1516) | ~25 | 15 s |
| first (cold) level-5 solve, levels 7 nested | 229 | 362 s |
| leakage Jacobian K (8 columns) | 8 | 13 s |
| level-3 Newton, 2 accepted steps | ~850 | 897 s |
| total | 1090 | 1358 s |

What it gives:

- **The profile at M = 7.** F₅/F₁ = 0.003104 + 0.000557i, against 0.003092 + 0.000541i at M = 5 and 0.00310 + 0.00056i
  from the untruncated amplitude expansion at (2.72, ω∞) (README): adding harmonic 7 moves F₅ onto the untruncated
  value. F₇/F₁ = 0.000175 + 0.000045i, so |F₇/F₅| = 0.0573 (the log rounds to six decimals), against the untruncated
  per-pair decay ρ_*² = 0.0574 (B_tail.md §2). F₃/F₁ = 0.054843 + 0.005438i (M = 5: 0.054883 + 0.005414i), not yet
  converged (below). The re-tuning that harmonic 7 causes is ΔF₅/F₁ = (1.2 + 1.6i)×10⁻⁵ and ΔF₃/F₁ = (−4.0 + 2.4i)×10⁻⁵,
  i.e. 0.11 and 0.26 of F₇/F₁: the lower harmonics move by fractions of the new one, not through a slow
  nearest-neighbour cascade.
- **No converged r₁.** The level-3 Newton went |r₃| 2×10⁷ → 34 → 16 → 9.1 and stopped there by `hier.py`'s own noise-floor
  rule (a fresh Jacobian no longer halves |r₃|). At |r₃| = 9 the harmonic-1 residual is dominated by leakage
  (Schur-corrected r₁ = −9.8×10⁴ − 6.3×10⁴i, raw −6.7×10⁴ − 3.2×10⁴i), so this run does **not** give r₁^{[7]}(p₅), and the
  M = 7 zero was not obtained.

PLACEHOLDER-DIAG

## 3. The truncation sequence, as far as it goes

`python3 trunc_seq_M7.py` (reads the refined M = 3 and M = 5 zeros of `solver/inner_pert/logs/first_order_hier_M{3,5}.log`
and, once it exists, the last `ZERO M 7` line of `logs/hier_nb_M7_full*.log`):

| M | c_M = ω/ω∞ | A | \|F_M/F_{M−2}\| at the zero | d_M = c_M − c_{M−2} | κ_M = d_M/d_{M−2} | coupling g_M = \|κ_M\|/\|F_M/F_{M−2}\| |
|---|---|---|---|---|---|---|
| 1 | 0.7769940136 | 2.542148048 | | | | |
| 3 | 1.0156030949 | 2.726475886 | 0.0531 | +0.23861 | | |
| 5 | 0.9992284032 | 2.718579189 | 0.0569 | −0.016375 | −0.0686 | 1.21 |
| 7 | not obtained | | 0.0573 (at p₅, §2) | | | |

What is new without c₇:

- **The profile ratio of the added pair converges to the untruncated one**: |F_M/F_{M−2}| = 0.0531, 0.0569, 0.0573 for
  M = 3, 5, 7 against 0.0574, and F₅ at M = 7 is the untruncated F₅ to the three digits available (§2). So the profile
  half of (T1) is seen at M = 7, with ρ_*² = 0.0574 and no sign of a change of regime.
- **The only measured coupling is g₅ = 1.21**, from a ratio that includes the pre-asymptotic M = 1. B_tail.md's prediction
  for M = 7 (0.99997–1.00017) assumes κ between −0.045 and −0.057 (coupling 0.78–1.0, fixed by requiring the limit to
  lie in the untruncated bracket); if κ stayed at −0.0686 the limit would be 1.00028 and c₇ = 1.00035, outside the
  untruncated bracket 0.99915–1.00011 (`reconcile.py`). So M = 7 decides between "the M = 1 → 3 step is pre-asymptotic"
  (B_tail.md's reading) and a tension between the two routes at the 2×10⁻⁴ level. It is still the sharp test it was.
- **The lowest-order increments decrease at the same kind of rate** (§4): x²Δσ₄/Δσ₂ has modulus 0.035 for the
  M = 1 → 3 step, an independent (order-by-order, no lateral integration) estimate of a per-pair ratio well below ¼.

### 3.1 Proposed task for the local 8-thread machine (for the coordinator to queue; not written to comms.txt)

**Goal.** The M = 7 zero of the jump-conditioned truncation (`hier.py --schur` on the compiled backend), and whether the
pilot's level-3 stall is a finite-difference-step effect.

**Why local.** In this session each process got about 12% of a cloud core (load average 16–22): the pilot's one level-3
solve took 22 min, and a full selection needs some ten of them. With 6 busy threads a local process gets roughly five
times that, and the compiled backend makes an M = 7 residual about 0.3 s of CPU.

**Prerequisites** (from `solver/inner_stokes/`, after pulling the branch): `pip install numba` if it is missing (tested
with numba 0.68.0), then

```
python inner_nb.py check 3 5 7        # must end with PASS (about 2 min); stop the task if it does not
python hier_nb.py selftest            # must end with PASS
```

**Threads.** 6 threads at below-normal priority, one process per thread: three jobs of `--procs 2` (each is a main
process, mostly idle, plus two workers), started together.

```
python -u hier_nb.py 7 2.718579189229888 0.9992284031945021 --x=0.1492035749173264,0.01471905249087869,0.008405346906375517,0.0014703451817031647 --pretune --schur --predict --floor --procs 2 > logs/hier_nb_M7_full_local_A.log
python -u hier_nb.py 7 2.718579189229888 0.9992284031945021 --x=0.1492035749173264,0.01471905249087869,0.008405346906375517,0.0014703451817031647 --pretune --schur --predict --floor --procs 2 --h3=1e-4 --h1=1e-4 > logs/hier_nb_M7_full_local_B.log
INNER_NB_RTOL=1e-13 python -u hier_nb.py 7 2.718579189229888 0.9992284031945021 --x=0.1492035749173264,0.01471905249087869,0.008405346906375517,0.0014703451817031647 --pretune --schur --predict --floor --procs 2 --h3=1e-4 --h1=1e-4 > logs/hier_nb_M7_full_local_C.log
```

(On Windows set the variable first: `set INNER_NB_RTOL=1e-13`, in the window of job C only.) A is the cloud pilot's
settings, B takes finite-difference steps 100× larger at level 3 (|Δr₃| ≈ 0.7 instead of 0.007 for the C₃ column) and
level 1, C also tightens the integrator.

**Stop conditions**, per job:

- the first `level 3 solved: |r_3| …` line shows |r₃| > 10⁻³ and the following `level 1 it` lines do not decrease |r₁|
  for two iterations: kill the job's process tree (`taskkill /T /F /PID <pid>`), keep the log;
- any `Traceback` (except the harmless one at interpreter exit after the result is printed);
- 4 hours of wall time;
- done when it prints `ZERO M 7: …` with the F_ν lines.

**Expected time.** First cold level-3 solve 5–10 min; a full selection 1000–3000 residuals, 30–90 min per job.

**Push.** `solver/inner_stokes/logs/hier_nb_M7_full_local_{A,B,C}.log` only. Then, on the cloud side,
`python3 solver/proof/trunc_seq_M7.py` reads the last `ZERO M 7` line (it globs `logs/hier_nb_M7_full*.log`; with several
logs, compare them first: the jobs differ only in numerical parameters, so their zeros must agree to their noise, and
that spread is the error bar to quote).

## 4. Which truncation: hier's model is a Galerkin truncation too

B_tail.md §4 concludes that a tail lemma "cannot be a statement about Galerkin projections" because `amp_trunc.py`, the
amplitude recursion with every θ-function projected onto |μ| ≤ M, gives σ_{M+1} too large by 10⁴–10⁶. That conclusion
needs one correction, reported here and not fixed in B_tail.md:

- `hier.py`'s model 𝔐_M is itself a Galerkin truncation of the inner equations in θ (`inner_np.py`: "Galerkin-truncated in
  theta"), with the jump conditions imposed on the truncated system. `amp_trunc.py` imposes the same jump conditions
  (it tunes f₃ … f_M order by order). The two differ in one harmonic: `inner_np.py` keeps the even harmonics of c and E up
  to |μ| = M + 1, `amp_trunc.py` stops them at M − 1 (amp_flint's truncation |μ| ≤ H with H = M).
- Lemma O (§6) shows that this one harmonic is exactly what moves `amp_trunc.py`'s first change to σ_M: with the even
  harmonic M + 1 dropped the threshold is 2M + 2 − |h| instead of 2M + 4 − |h|. The 5×10⁻⁴ changes of σ₃ (M = 3) and σ₅
  (M = 5) in B_tail.md's table are that harmonic.
- `amp_trunc_hiermodel.py` runs the amplitude expansion of 𝔐_M itself (amp_flint's truncation with H = M + 1, which
  keeps the odd harmonics ≤ M and the even ones ≤ M + 1; its de-mixing loop needed a parity fix, documented in the
  script). Results at ω∞, J = 500, legacy fit with every family present:

| Truncation | kept harmonics | σ₁ … σ_M | σ_{M+1} | source |
|---|---|---|---|---|
| hier's model, M = 1 | odd ≤ 1, even ≤ 2 | σ₁ unchanged (4.5×10⁻¹⁰, fit level) | σ₂ changed by 0.0590 (14% of \|σ₂\|) | `trunc_hier_M1_K5_J500_w1.0000.json` |
| hier's model, M = 3 | odd ≤ 3, even ≤ 4 | σ₁–σ₃ unchanged (≤ 2.7×10⁻¹⁰) | σ₄ changed by 3.74×10⁻⁵ (0.12% of \|σ₄\|) | `trunc_hier_M3_K9_J500_w1.0000.json` |
| hier's model, M = 5 | odd ≤ 5, even ≤ 6 | PLACEHOLDER-M5AMP | | `trunc_hier_M5_K13_J500_w1.0000.json` |
| `amp_trunc.py`, M = 3 | all ≤ 3 | σ₃ changed by 5.3×10⁻⁴ | σ₄ = −399 (true −0.029) | B_tail.md §4 |
| `amp_trunc.py`, M = 5 | all ≤ 5 | σ₅ changed by 4.4×10⁻⁴ | σ₆ = −1371 (true −0.0011) | B_tail.md §4 |

(`python3 trunc_seq_M7.py --amp`.) Lemma O's prediction holds exactly in both truncations: hier's model first changes at
σ_{M+1}, `amp_trunc.py`'s at σ_M. And the explosion is a property of dropping the even harmonic M + 1, not of Galerkin
truncation: with it kept, σ₄ changes by 0.12%, not by a factor 10⁴. The lowest-order increments themselves decrease
fast: x² Δσ₄/Δσ₂ = −0.025 + 0.024i, modulus 0.035, at A = 2.7196 (the M = 1 → 3 step of Lemma L, against −0.069 for the
same step of the zeros).

So the distinction that matters for the tail lemma is not "jump-conditioned versus Galerkin" but which harmonics the
truncation keeps, and whether the increment is measured order by order or at the actual amplitude. The statement of §5
is for 𝔐_M, the model whose sequence is measured.

## 5. The tail lemma, stated

### 5.1 Objects

For odd M, 𝔐_M is the model `hier.py` solves (`inner_np.Inner` with every homotopy weight 1): y and σ carry the odd
harmonics |μ| ≤ M, c (and E = y²/2) the even harmonics |μ| ≤ M + 1, and every product is projected onto these spaces.
Its data are u = (A, ω, f₃, …, f_M) with F = A sin θ + Σ (B_ν sin νθ + C_ν cos νθ) and f_ν = F_ν.

- **Multipliers.** At fixed M the formal solution regular at ξ = 0 is 1-summable (Braaksma;
  [C_selection.md](C_selection.md) B4), and the difference of its lateral sums defines the label multipliers
  S_ν^{[M]}(u), |ν| ≤ M. Normalised: Ŝ_ν = S_ν/(2κ_ν), with κ_ν the linear (Weber) connection coefficient of label ν, so
  that Ŝ_ν = F_ν + (nonlinear terms) and |κ_ν| ≍ e^{π|ν|ω/2}.
- **Jump-conditioned truncation.** Hypothesis (H_T^M): the Jacobian ∂(Ŝ₃, …, Ŝ_M)/∂(f₃, …, f_M) is invertible on the box.
  Then the tuned set {Ŝ₃^{[M]} = ⋯ = Ŝ_M^{[M]} = 0} is a graph f = f^{[M]}(A, ω), and
  s_M(A, ω) = Ŝ₁^{[M]}(A, ω, f^{[M]}(A, ω)). Its zero is p_M = (A_M, ω_M), and c_M = ω_M/ω∞.
- **Increment.** δ_M = s_{M+2} − s_M, a function on the box: two different models, each on its own tuned set.
- **Untruncated.** s_∞, under hypothesis (H_∞): the tuned untruncated profile exists in X_σ and its harmonic-1
  multiplier is defined (gaps (a) and (b) at the level of existence; the amplitude expansion defines it order by order).
- **Norms** (B_tail.md §6). Profile: ‖f‖_σ = Σ_{ν odd ≥ 3} |f_ν| e^{σν} (the space X_σ, σ < σ_F = 1.428). Multipliers, in
  the dual norm: |Ŝ|*_σ = sup_ν |Ŝ_ν| e^{−σν}, i.e. |S_ν| with the weight e^{−σν}/|κ_ν| ≍ e^{−(σ + πω/2)ν}.

### 5.2 The statement

> **Lemma T (tail lemma; target, not proved).** Let B be the box of radius ε about p₇ in (A, ω/ω∞). There are
> σ ∈ (0, σ_F) and explicit constants Λ, Q, C_f, ρ_* with ρ_*² e^{2σ} ≤ 1/4 such that for every odd M ≥ 5 and every
> (A, ω) ∈ B:
>
> - (T1, profile) |f^{[M+2]}_{M+2}(A, ω)| ≤ C_f ρ_*^{M+1};
> - (T2, coupling) |δ_M(A, ω)| ≤ Λ e^{σ(M+2)} |f^{[M+2]}_{M+2}| + Q (e^{σ(M+2)} |f^{[M+2]}_{M+2}|)²;
> - (T3, consistency) s_M → s_∞ uniformly on B.
>
> Consequently |s_∞ − s₇| ≤ Σ_{M ≥ 7} |δ_M| on B (Lemma C), and Theorem A(∞) follows from a certified Theorem A(M = 7)
> on B whenever that sum is below σ_min(J₇) ε, with J₇ the outer Jacobian of s₇.

In ratio form, which is what the data measure:

> **(R)** sup_B |δ_{M+2}| ≤ κ sup_B |δ_M| for odd M ≥ 5, with κ ≤ 1/4.

(T1) and (T2) give (R) with κ = ρ_*² e^{2σ} (1 + O(Q e^{σ(M+2)} |f_{M+2}| / Λ)).

Three remarks fix what the statement is and is not.

1. Both s_M and s_{M+2} are jump-conditioned, each in its own model: no Galerkin projection of one tuned solution onto
   fewer harmonics appears, only two tuned problems side by side. (T2) is B_tail.md's second half, "|r₁^{(∞)} − r₁^{(M)}|
   ≤ L ‖(I − P_M)F‖_σ", written for consecutive truncations, so that every quantity in it is one `hier.py` computes.
2. Λ is the norm of the linearized coupling in the dual space X_σ*: the derivative of the tuned harmonic-1 multiplier with
   respect to the newly added harmonic pair, the lower harmonics re-tuned (identity H in §6 makes this exact).
3. The weights e^{π|ν|ω/2} do not appear in (T2). In `hier.py`'s raw residuals they appear as couplings such as
   ∂r₁/∂B₅ ≈ 10¹⁷; Lemma R shows that these are properties of the residual, not of the tuned problem.

What κ ≤ 1/4 asks of σ: with the measured profile decay ρ_*² = 0.0574 per pair, ρ_*² e^{2σ} ≤ 1/4 holds for
σ ≤ ½ ln(1/(4 × 0.0574)) = 0.736. So the coupling bound is needed in X_σ* for some σ ≤ 0.736, i.e. the linearized
coupling of harmonic ν may grow at most like e^{0.736ν}, against e^{πων/2} = e^{6.0ν} for the raw couplings.

## 6. What is proved

All by hand, under the hypotheses named; none uses a numerical value.

**Lemma R (the Schur-corrected residual is representation-independent).** Fix M and let r(u) ∈ ℝ^{M+1} be `hier.py`'s
raw residual (the de-mixed normalised jump at the radii it uses). Hypothesis (H_6a^M): r(u) = Φ_u(Ŝ(u)) with Φ_u a
C¹ map, Φ_u(0) = 0, T(u) = DΦ_u(0) invertible (the jump at those radii is fixed by the multipliers, vanishes exactly when
they all vanish, and is linear in them to first order; for M = 1 this is lemma 6a of [A_certify.md](A_certify.md) §11).
Write 1 for the harmonic-1 rows and H for the rows 3, …, M. Then:

1. On {r_H = 0}, r₁ = τ(Ŝ₁) with τ(0) = 0 and τ′(0) = T̃ = T₁₁ − T₁H T_HH⁻¹ T_H1, the Schur complement of T, which is
   invertible. So the zero `hier.py` finds (r₁ = 0 on r_H = 0) is the zero of s_M (Ŝ₁ = 0 on Ŝ_H = 0), exactly.
2. At that zero the outer Jacobians are related by D_p(r₁|_{r_H=0}) = T̃ (I − G h′)⁻¹ J_s, where J_s = D_p s_M, G =
   ∂_fŜ₁ (∂_fŜ_H)⁻¹ and h′ = −T_HH⁻¹ T_H1, whenever I − G h′ is invertible. So the zero is simple for one exactly when
   it is simple for the other.
3. The same identity holds for the Schur complement of hier's leakage Jacobian K = D r = T DŜ: the entries of T, such as
   ∂r₁/∂B₅ ≈ 10¹⁷ (the label-5 jump leaking into the harmonic-1 row at finite ξ, with a weight of order
   e^{π(5−1)ω/2}), cancel exactly in it.

*Proof.* Assume also T_HH invertible (`hier.py` needs this to define its correction; its K_HH is invertible). Then
Φ_H(Ŝ₁, Ŝ_H) = 0 is solved by Ŝ_H = h(Ŝ₁) with h(0) = 0 and h′(0) = −T_HH⁻¹T_H1 (implicit function theorem), and r₁ =
Φ₁(Ŝ₁, h(Ŝ₁)) =: τ(Ŝ₁) with τ′(0) = T₁₁ + T₁H h′(0) = T̃, invertible since det T = det T_HH · det T̃. For 2,
differentiate Ŝ_H(p, f^r(p)) = h(Ŝ₁(p, f^r(p))) at the zero: with a = D_p(Ŝ₁ on {r_H = 0}), f^r′ = (∂_fŜ_H)⁻¹(h′a −
∂_pŜ_H), so a = J_s + G h′ a. For 3, K's Schur complement is D_p of r₁ along the linearised set {K_H du = 0}, which at
the zero is the tangent of {r_H = 0}; apply 2. ∎

h′ is the leakage of label 1 *up* into the rows of higher labels, which carries e^{−π(μ−1)ω/2}, so I − G h′ ≈ I. Lemma R
is the precise form of the sentence "the weights multiply zero and cancel in the tuning" (B_tail.md §2): the ten to
seventeen orders of magnitude in `hier.py`'s K are a property of measuring multipliers through a finite-ξ jump, and they
leave neither the zero nor its Jacobian.

**Lemma O (order counting in hier's model).** Expand 𝔐_M in the amplitude, F = A sin θ + Σ_{ν ≥ 3} f_ν(A) e^{iνθ} +
c.c., with every f_ν^{(k)}, 3 ≤ ν ≤ M, tuned order by order (Ŝ_ν^{(k)} = 0), and S₁/S₁^{(1)} = 1 + Σ σ_j^{[M]} x^j, x =
A². Compare with the untruncated expansion (σ_j, f_ν^{(k)}). Then every coefficient of 𝔐_M at order A^k and harmonic h
agrees with the untruncated one whenever k < 2M + 4 − |h|. In particular:

- σ_j^{[M]} = σ_j for j ≤ M, and f_ν^{(k),[M]} = f_ν^{(k)} for k ≤ 2M + 2 − ν;
- for the truncation of `amp_trunc.py` (every function at |μ| ≤ M, so the even harmonics stop at M − 1) the threshold is
  2M + 2 − |h| instead, and σ_M is the first coefficient that changes.

*Proof.* At order k only harmonics with |h| ≤ k, h ≡ k (mod 2), occur (true at k = 1; products add orders and harmonics;
a tuned term f_ν^{(k)} has |ν| ≤ k). Call (k, h) *different* if the order-k, harmonic-h coefficient (of y, σ, c, E, or
of the tuned F) differs between the two expansions. Claim: different ⇒ k ≥ τ(h) := 2M + 4 − |h|. A coefficient is
different for one of three reasons. (i) h lies outside 𝔐_M's space (odd |h| ≥ M + 2 or even |h| ≥ M + 3): then k ≥ |h| ≥
τ(h). (ii) It is a sum of products one of whose factors, (k₁, h₁), is different: by induction k₁ ≥ τ(h₁), and the other
factors have total order k − k₁ ≥ |h − h₁| ≥ |h₁| − |h|, so k ≥ τ(h₁) + |h₁| − |h| = τ(h). (iii) It is a tuned term
f_ν^{(k)} whose order-k multiplier Ŝ_ν^{(k)} differs: that multiplier is a functional of the de-mixed late orders Σ_μ
[e^{−Π}]^{(k″)}_{ν−μ} d^{(k′)}_μ with k′ + k″ = k, and a different d^{(k′)}_μ needs k′ ≥ τ(μ) while
[e^{−Π}]^{(k″)}_{ν−μ} ≠ 0 needs k″ ≥ |ν − μ|, so again k ≥ τ(ν); a different e^{−Π} coefficient (built from the tuned F)
needs k″ ≥ τ(ν − μ) and k′ ≥ |μ|, which gives the same bound, and so does the ⟨F²⟩ series of the normalisation (harmonic
0). For h = 1, k ≥ 2M + 3, i.e. relative order A^{2M+2}: σ_{M+1}. For `amp_trunc.py` the first dropped harmonic is the
even M + 1, at order M + 1, and the same induction with τ′(h) = 2M + 2 − |h| gives σ_M. ∎

(The multipliers here are the mathematical ones, defined order by order by lateral Borel sums; a late-order fit reads
them to its own accuracy, ~10⁻¹⁰.)

**Lemma L (the lowest-order increment is linear in the tail).** At the first order where δ_M can be non-zero,
δ_M = Δσ_{M+1} x^{M+1} + O(x^{M+2}) with Δσ_{M+1} = σ_{M+1} − σ_{M+1}^{[M]}, and Δσ_{M+1} is a real-linear functional of
the top-degree harmonic ±(M + 2) of the untruncated solution, y^{(M+2)}_{±(M+2)}(ξ) (= 2f^{(M+2)}_{M+2} times the regular
Kummer series plus the part forced by the lower top-degree harmonics).

*Proof.* In the induction of Lemma O, a coefficient at its threshold (k = τ(h), h ≠ 0) cannot contain two different
factors: different factors have k_i ≥ max(τ(h_i), |h_i|) ≥ M + 2, so two of them need k ≥ 2M + 4 > τ(h). So at threshold
the difference is linear in the threshold data, and descending from h = ±(M + 2), k = M + 2, every threshold
coefficient is a linear image of y^{(M+2)}_{±(M+2)}. (At h = 0 the threshold 2M + 4 allows the quadratic term
|f_{M+2}|² of ⟨F²⟩, which reaches harmonic 1 only at order 2M + 5.) ∎

**Lemma C (telescoping).** If (T3) holds and sup_B |δ_{M+2}| ≤ κ sup_B |δ_M| for odd M ≥ 5 with κ < 1, then
sup_B |s_∞ − s₇| ≤ κ/(1 − κ) · sup_B |δ₅|. *Proof:* s_∞ − s₇ = Σ_{j ≥ 1} δ_{5+2j}, and |δ_{5+2j}| ≤ κ^j sup|δ₅|. ∎

**Identity H (the increment as a coupling integral).** Call the harmonics of 𝔐_M old and the odd M + 2 and even
M + 3 new. Let 𝔐^λ_{M+2}, 0 ≤ λ ≤ 1, be 𝔐_{M+2} with two kinds of product terms multiplied by λ: in the equations of the
old harmonics, every term with a new factor (the back-reaction), and in the equations of the new harmonics, every term
with no new factor (the forcing by the old ones). Terms linear in the new harmonics with old coefficients (such as
c₀σ_{M+2}) and new–new terms keep weight 1. At λ = 0 the old harmonics obey 𝔐_M exactly and the new block is unforced,
so its tuning gives f_{M+2} = 0, y_new ≡ 0 and s^0 = s_M; at λ = 1, s^1 = s_{M+2}. If (H_T) holds along the path,

```math
\delta_M = \int_0^1 G(\lambda)\,d\lambda, \qquad G = \partial_\lambda \hat S_1 - \partial_f \hat S_1\,(\partial_f \hat S_H)^{-1}\,\partial_\lambda \hat S_H \quad (H = 3, \dots, M+2),
```

a Schur complement: the rate at which the tuned harmonic-1 multiplier changes as the coupling to the new pair is switched
on, every other label re-tuned. G(0) = 0: at λ = 0 the new harmonics vanish, so ∂_λ of the back-reaction terms and
∂Ŝ_old/∂f_{M+2} vanish there. So δ_M is first order in the forcing of the new pair and first order in its back-reaction.
This is the exact meaning of "the linearized coupling" below. *Proof:* chain rule and the implicit function theorem on
the tuned set. ∎ (`inner_np.Inner`'s homotopy weight is a different interpolation: it also multiplies the new harmonics'
linear terms, so at its λ = 0 the new block is degenerate rather than a decoupled tuned problem.)

## 7. Constants from the data

Every constant of Lemma T, with what it is (all *numerical evidence* unless marked).

| Constant | Role | Value | Source |
|---|---|---|---|
| σ_F | profile strip half-width | 1.428 | B_tail.md §2 (tuned harmonics, amplitude expansion) |
| ρ_*² = \|F_{ν+2}/F_ν\| | profile decay per pair, (T1) | 0.0574 untruncated (ν ≥ 5); at the truncated zeros \|F₃/F₁\| = 0.0531 (M = 3), \|F₅/F₃\| = 0.0569 (M = 5), \|F₇/F₅\| = 0.0573 (M = 7 at p₅) | B_tail.md §2; `trunc_seq_M7.py`; §2 |
| \|q∞\| A² | leading tuning map, (T1) contraction | 0.0633; × e^{2σ}: 0.27 at σ = 0.736, 0.47 at σ = 1 | B_tail.md §2, §6 |
| A-series term ratio of each f_ν | corrections to the leading tuning map | ≈ 0.25 per order at A* | B_tail.md §2 |
| σ needed for κ ≤ ¼ | (★) | ≤ 0.736 (*arithmetic*) | §5.2 |
| d_M = c_M − c_{M−2} | increments in ω/ω∞ | +0.23861 (M = 3), −0.016375 (M = 5); M = 7 not obtained | `trunc_seq_M7.py` |
| κ_M = d_M/d_{M−2} | ratio per pair, (R) | −0.0686 (M = 5; M = 1 is pre-asymptotic) | `trunc_seq_M7.py` |
| g_M = \|κ_M\|/ρ_M² | coupling per pair, Λe^{2σ} in (T2) | 1.21 (M = 5) | `trunc_seq_M7.py` |
| \|∂r₁/∂(ω/ω∞)\|, σ_min(J₅) | converts ω to r₁; certification margin | 6.7×10³; 3.9×10³ | `feasibility.py` |
| tail needed for a 10⁻³ box | | 3.9 in r₁ (*arithmetic*) | PROOF.md A.4 |
| raw leakage ∂r₁/∂B₅ | entries of T (Lemma R) | ~10¹⁷, against J ~ 7×10³ | README, `hier.py` |
| Δσ_{M+1} (lowest order) | Lemma L | Δσ₂ = 0.0053 + 0.0587i (M = 1), Δσ₄ = (−2.83 − 2.45i)×10⁻⁵ (M = 3); x²Δσ₄/Δσ₂ = 0.035 in modulus | `amp_trunc_hiermodel.py`, §4 |

## 8. The ratio bound: an attempt, and the inequality that needs a new idea

### 8.1 What the data say about the coupling

Two measurements bear on (★), both *numerical evidence*:

- **Size.** The only increment ratio available, κ₅ = −0.0686, is the profile decay of the added pair (0.0569) times a
  coupling g₅ = 1.21. In the language of (T2), Λ e^{2σ} ≈ 1.2 per pair, i.e. an effective σ of ½ ln 1.21 ≈ 0.1, far below
  the 0.736 that κ ≤ ¼ needs. If B_tail.md's reading is right and M = 1 is pre-asymptotic, g is 0.8–1.0 and σ_eff ≤ 0.
- **Structure.** Adding harmonic 7 at p₅ (§2) moves the re-tuned lower harmonics by |ΔF₅| ≈ 0.11 |F₇| and
  |ΔF₃| ≈ 0.26 |F₇| (with |r₃| = 9 left, F₃'s change is uncertain by ~15%). The lower harmonics move by O(1) fractions of
  the new one, and harmonic 3 moves more than harmonic 5: the effect is not a nearest-neighbour cascade down the tuning
  chain but a direct O(1) coupling of the new pair to the low harmonics in normalised units, as (T2) with a ν-independent
  Λ assumes.

So the data are consistent with the simplest form of (★): the linearized coupling Λ_ν is O(1) and roughly independent of
ν, as the tuning ratio q_ν is.

### 8.2 Lowest order in the amplitude: what can be proved

By Lemma L the lowest-order coefficient of the increment, Δσ_{M+1}, is linear in the top-degree harmonic M + 2, and
every coefficient of the amplitude expansion is built by one kind of step: at order k, harmonic ν obeys amp_flint's
scaled recursion (d̃_j = d_j/2^j; the operator is the c = 1 one, since order k depends on itself only through g₀ = 1)

```math
\tilde d_{j+1} = a_j\,\tilde d_j - b_j, \qquad a_j = \frac{(j+\alpha_\nu)(j+\beta_\nu)}{j+1}, \qquad \alpha_\nu = \tfrac{1+i\nu\omega}{2},\ \beta_\nu = \tfrac{2+i\nu\omega}{2},
```

with b_j the forcing from lower orders. This is the step for which the cancellation can be written exactly.

**Lemma W (the weight cancels in every tuning step; proved, as algebra).** With T_j = (α_ν)_j(β_ν)_j/j! = Π_{m<j} a_m,
the recursion solves in Darboux form, d̃_j = T_j (d̃₀ − Σ_{i<j} b_i/T_{i+1}). The late-order multiplier sequence of
amp_flint is R_j = base_j (d̃_j + de-mixing terms) + normalisation terms, with base_j T_j = K_ν = 1/(Γ(α_ν)Γ(β_ν))
exactly ([D_entire.md](D_entire.md) §6.1, where this telescoping is carried out and enclosed rigorously for σ₁). So the
normalised order-k multiplier is

```math
\hat S^{(k)}_\nu = \lim_{j\to\infty}\Big[\tilde d^{(k)}_0 - \sum_{i\lt j}\frac{b^{(k)}_i}{T_{i+1}} + C^{(k)}_j\Big], \qquad f^{(k)}_\nu = \tfrac12\lim_{j\to\infty}\Big[\sum_{i\lt j}\frac{b^{(k)}_i}{T_{i+1}} - C^{(k)}_j\Big],
```

where C_j collects the de-mixing and ⟨F²⟩-normalisation terms (explicit in D_entire.md §6.1 for k = 3). K_ν ≍ e^{πνω/2}
appears nowhere: B_tail.md §2's "the weights cancel in the tuning" is this identity. Moreover every denominator only grows
with ν: |T^{(ν)}_i| = Π_{m<i} |m + α_ν| |m + β_ν|/(m + 1) is increasing in ν for each i, so term by term
|b_i/T^{(ν)}_{i+1}| ≤ |b_i|/(½)_{i+1}. ∎

**Why this does not yet give (★).** The sum converges only through cancellations: D_entire.md §6.1 needs two exact
cancellations of the O(1/j) parts of the increments before they are O(j⁻²) and summable. The forcing b_i grows like the
late orders of the lower harmonics, |T^{(μ)}_i| ≍ (½)_i up to oscillating factors, so the term-by-term majorant
Σ|b_i|/(½)_{i+1} ~ Σ 1/i diverges (logarithmically). A ν-uniform bound needs the cancellations themselves to be
ν-uniform. That is a concrete, finite-dimensional question per order (the cancellations are identities between the
ψ-function terms of C_j and the first terms of the forcing), and it is where an attempt at (★) at lowest order should
continue; it was not completed here. Measured, the outcome is known: the leading-order tuning ratio q_ν is ν-independent
to six digits (B_tail.md §2), so the tuning steps are uniform in fact.

### 8.3 Why the lowest order does not settle the ratio at A*

Suppose (U) were proved, so that the lowest-order increments obey |Δσ_{M+3}| x² ≤ κ_lo |Δσ_{M+1}| with an explicit
κ_lo (measured: 0.035 for M = 1 → 3, §4). That is a statement about one coefficient of each increment's series in
x = A², at a value x* = 7.4 where the series of S₁ itself is far from its first terms (|σ_k| x*^k grows to about 240 at
k = 9 before decaying; PROOF.md route A1). The increments δ_M(x) = Σ_{k ≥ M+1} Δσ_k^{[M]} x^k are series of the same kind,
so their first terms neither bound them nor fix their ratios: the measured ratios differ by a factor two (0.035 at lowest
order, 0.069 for the zeros). Controlling δ_M at x* needs the whole series, i.e. the entire-function bounds of route A3
([D_entire.md](D_entire.md)) applied to the increments, or an argument at finite amplitude. The lowest order is therefore
a consistency check and a formal (order-by-order) version of (R), not a route to the certificate.

### 8.4 The inequality that needs a new idea

> **(★) Uniform coupling bound.** On B and for every odd ν ≥ 7, the increment caused by adding the pair ν,
> δ_{ν−2} = ∫₀¹ G dλ (identity H), satisfies |δ_{ν−2}| ≤ Λ e^{σν} |f^{[ν]}_ν| + (quadratic), with Λ independent of ν and
> σ ≤ 0.736.

This is the linear part of (T2), uniformly in ν.

Everything else in Lemma T is either proved above (R, O, L, C, H), a contraction estimate of a standard kind (T1, in
X_σ, B_tail.md §6), or a consistency statement (T3) that follows from (★) and (T1). The data say (★) holds with room to
spare: the coupling per pair is g₅ ≈ 1.2, i.e. the effective σ is ½ ln g ≈ 0.1, while 0.736 is
needed. The naive bound, through the linear connection coefficients, is e^{πων/2} = e^{6.0ν}.

**Where the weights come from, as an angle.** On a ray ξ = |ξ| e^{iβ} the label harmonic is
e^{iνθ₀} = e^{iν(θ − 2ω ln|ξ|)} e^{2νωβ}. So:

- the linear weight κ_ν ≍ e^{πνω/2} is the label continued from the Stokes line (β = 0) to the anti-Stokes line
  β = π/4 (arg ξ² = π/2), where the fast modes are O(1): the lateral sums of `inner_np.py` start there and the arc back
  to the real axis is where the amplification happens;
- F's singularities reach the label at Im θ₀ = ±σ_F, i.e. β = ±σ_F/(2ω) = ±0.187 (B_tail.md §3);
- (★) with σ ≤ 0.736 asks for β ≤ 0.736/(2ω∞) = 0.096 (5.5°), half-way to the first singularity.

So the new idea needed is a representation of the tuned linear response along paths that stay within about 5° of the
Stokes line, with bounds uniform in the label harmonic. Every construction of the multipliers used so far (lateral
Borel–Laplace sums, the jump at finite ξ, the late-order fits) goes through the anti-Stokes direction or through
individual labels, and so meets e^{πνω/2} before the cancellation. Two candidates:

1. **The reality form of the condition.** For real F the lower lateral sum is the conjugate of the upper one on the real
   ξ axis, so the jump is 2i Im y⁺ and S ≡ 0 says that y⁺ is real on the Stokes line. The linearized coupling is then
   Im δy⁺ on the Stokes line, and δy⁺ is a Laplace integral along arg ζ = 0⁺ of a Borel transform whose growth near the
   real ζ axis is what has to be bounded. In the label variable that is a Costin–Tanveer strip-analytic Borel estimate
   (PROOF.md §3) with a strip of half-width 2ωβ ≈ 0.74 inside the analyticity strip 1.43.
2. **The coupling as one operator.** If the map from the new pair's data to the re-tuned low harmonics is, for large ν,
   the same operator up to a shift in label space (as the tuning ratio q_ν is the same number for every ν to six digits),
   (★) reduces to a bound on one operator instead of on infinitely many Schur complements. The measured coupling (≈ 1,
   §8.1) and the ν-independence of q_ν both suggest that this limit exists.

## 9. Status

| Statement | Status |
|---|---|
| Compiled backend = scipy backend to scipy's own rtol noise, M = 1–7 (`inner_nb.py check`) | numerical check, with a mutation control |
| Lazy residual: every entry hier reads bit-identical (`hier_nb.py selftest`) | numerical check, with a mutation control |
| M = 7 at p₅: F₅/F₁ = 0.003104 + 0.000557i, \|F₇/F₅\| = 0.0573 | numerical evidence (levels 5, 7 at their floors) |
| M = 7 zero c₇ | not obtained (level-3 Newton stalls at \|r₃\| ≈ 9; local task §3.1) |
| B_tail.md's prediction c₇ ∈ [0.99997, 1.00017] | untested |
| hier's M-model expanded in A is exact through σ_M (M = 1, 3) and changes σ_{M+1} by 14% (M = 1), 0.12% (M = 3) | numerical evidence (fits); the exactness through σ_M is Lemma O, *proved* |
| B_tail.md §4's Galerkin explosion comes from dropping the even harmonic M + 1 | numerical evidence (M = 3) plus Lemma O; M = 5 running |
| Lemma T, ratio form (R) with κ ≤ ¼ | *conjecture*; numerical evidence κ₅ = −0.069, lowest-order 0.035 |
| Lemma R (Schur-corrected residual ⇔ jump-conditioned multiplier; weights cancel in the Schur complement) | *conditional* on (H_6a^M) and T_HH invertible; proved by hand |
| Lemma O (order counting), Lemma L (linearity at threshold), Lemma C (telescoping), identity H | *proved* (by hand) |
| Lemma W (the weight cancels exactly in every tuning step, Darboux form) | *proved* as algebra, given the late-order normalisation of D_entire.md §6.1 |
| (★) uniform coupling bound, σ ≤ 0.736 | *conjecture*; the one inequality that needs a new idea (§8.4) |
| (T1) profile bound in X_σ | not proved here (contraction framework of B_tail.md §6); numerical evidence ρ_*² = 0.0574 |

## 10. Reproduce

From the repository root unless stated; times are for this session's loaded cloud cores.

```
cd solver/inner_stokes
python3 inner_nb.py check 1 3 5 7                 # PASS; worst |numba - scipy| / scipy rtol noise 3.6 (~3 min)
python3 inner_nb.py check 3 5 --mutate=sign       # FAIL (exit 1), ratio ~1e14-1e15
python3 hier_nb.py selftest                       # PASS (~1 min)
python3 hier_nb.py selftest --mutate=radius       # FAIL (exit 1), 2.9e+02
python -u hier_nb.py 7 2.718579189229888 0.9992284031945021 --x=0.1492035749173264,0.01471905249087869,0.008405346906375517,0.0014703451817031647 --pretune --fixA --schur --predict --floor --procs 2 --verbose=2
                                                  # the pilot (§2), ~23 min here; FIXED-A line with |r_3| 9.1
cd ../proof
python3 trunc_seq_M7.py                           # the sequence table of §3
python3 trunc_seq_M7.py --planted                 # PASS (estimator on planted sequences)
python3 trunc_seq_M7.py --planted --mutate=nocorr # FAIL (exit 1)
python3 trunc_seq_M7.py --amp                     # Delta sigma_2, Delta sigma_4 and their ratio 0.0347 (§4)
python3 amp_trunc_hiermodel.py 3 9 500            # sigma_4^[3] = -0.0288364238 + 0.0108028097i (~35 min here)
```

## 11. Proposed text (for the coordinator; README.md, PROOF.md and B_tail.md were not edited)

**README.md**, "The inner problem at large x", *Five harmonics* bullet: replace the last sentence ("M = 7 is still out
of reach: … so the truncation sequence stops at M = 5.") by

> M = 7 is now affordable but not yet converged. On a compiled backend (`inner_stokes/inner_nb.py`: the same series and
> the same DOP853 integrator, 15–20× faster) with lazy radii (`hier_nb.py`), the first cold level-5 solve, which had not
> finished in 5 hours, takes 6 minutes. At the M = 5 zero, harmonic 7 moves F₅/F₁ to 0.003104 + 0.000557i, the
> untruncated value, with |F₇/F₅| = 0.0573; the level-3 solve there stops at |r₃| ≈ 9, so the M = 7 zero is still open
> ([proof/B_tail_M7.md](proof/B_tail_M7.md)).

**PROOF.md** §4, end of the A.3 *Status* paragraph, add:

> Since then ([proof/B_tail_M7.md](proof/B_tail_M7.md)): the tail lemma is stated for consecutive jump-conditioned
> truncations; proved are that hier's Schur-corrected residual has exactly the zeros of the jump-conditioned multiplier
> (the 10¹⁷ couplings cancel in its Schur complement), that the M-harmonic model is exact through σ_M in the amplitude
> expansion, and that the weight e^{πνω/2} cancels exactly in every tuning step. A ratio of ¼ per pair needs the
> linearized coupling bounded in X_σ* with σ ≤ 0.736, uniformly in the harmonic; that inequality is the open part.

**B_tail.md** §4, after "…not for a Galerkin projection of the equations.", add:

> (Correction: `hier.py`'s model is also a Galerkin truncation; it differs from `amp_trunc.py`'s in keeping the even
> harmonic M + 1 of c and E. With that harmonic kept, the amplitude expansion changes σ₄ by 0.12% at M = 3 instead of
> exploding; see [B_tail_M7.md](B_tail_M7.md) §4.)

**check_numbers.py**: two checks with machine-readable sources, if the coordinator quotes these numbers: Δσ₄ of hier's M
= 3 model (|Δσ₄| = 3.74×10⁻⁵, from `solver/proof/trunc_hier_M3_K9_J500_w1.0000.json` and the README's σ₄), and F₅/F₁ =
0.003104 + 0.000557i at M = 7 (from the `F_5/F_1` line of
`solver/inner_stokes/logs/hier_nb_M7_pilot_fixA_at_M5zero.log`).

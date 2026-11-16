# Independent review of the σ-model kernel's band solver

Task: an independent review of `smdriver.newton_band`, the deflated Newton that the σ-model kernel's runs from D 10
on rest on (#845, #913–#917), of its planted test `planted_band.py`, and of what its docstring and
[SIGMA_DSS.md](SIGMA_DSS.md) §11 say about it, at main 7f88c5bb (#922). The function's last change is 3ab276fb.
Reviewer: an agent that did not write `newton_band`, `planted_band.py` or any other σ-model code. Check scripts:
[`sigma_ms/review_922/`](sigma_ms/review_922/) (`rv_newton.py`, `rv_band_mut.py`, `run_i922.sh`). Runs and logs: raw
`I922-mac/`. [proof/A_M_b_review.md](../proof/A_M_b_review.md) and
[proof/B_LC_floor_review.md](../proof/B_LC_floor_review.md) were read as the templates.

## Contents

- [Verdict](#verdict)
- [Summary of the checks](#summary-of-the-checks)
- [1. What the function does](#1-what-the-function-does)
- [2. The claims against the code](#2-the-claims-against-the-code)
- [3. One Newton step from the residual alone](#3-one-newton-step-from-the-residual-alone)
- [4. How far Δ can move at a converged state](#4-how-far-δ-can-move-at-a-converged-state)
- [5. The planted test, its mutations and the reviewer's](#5-the-planted-test-its-mutations-and-the-reviewers)
- [6. Findings](#6-findings)
- [7. What I did not check](#7-what-i-did-not-check)
- [8. Reproduce](#8-reproduce)

VERDICT_AND_SUMMARY

## 1. What the function does

Each iteration calls `Solver.newton` at the state z. That call returns the residual F, Newton's step
d_N = −J⁻¹F with the exact double-double Jacobian, and keeps the LU factors. The iteration then takes one of three
paths.

- **Newton's own step**, if |d_N|₂ is at most the radius and the step lowers |F|₂. No SVD is made.
- **A resolved step.** An f64 SVD of the dense Jacobian gives the band, the k singular triplets with
  σ_i < `band_tol`·σ_max, with left vectors U_w and right vectors V_w. The residual splits into B = |U_wᵀF|₂ and
  S = |F − U_wU_wᵀF|₂, and the step into its band coordinates a = V_wᵀd_N and the rest d_s = d_N − V_w a. When S is
  above max(κB, 2E, 0.1 tol) the step is x·d_s, x halved from 1 until S falls.
- **A band step** otherwise: d_s + V_w y with y_i = w_i a_i, w_i = s_i²/(s_i² + μ²), followed by chord corrections.

**The linear model of a band step.** With R = U_wᵀJV_w, rotated so that R = diag(s_i), and L = JV_w − U_wR, the
image of the band is JV_w c = U_w(s∘c) + Lc for any coordinates c. Since J d_N = −F,

```math
F + J\,(d_s + V_w y) \;=\; -\,J V_w (a - y) \;=\; -\,U_w\,\bigl(s\circ(a-y)\bigr) \;-\; L\,(a-y) .
```

So the model's band residual after the step is |s∘(1 − w)∘a|, which the code writes as |r∘(1 − w)| with r = U_wᵀF
(`Bp`), and the model's resolved residual is |L(a − y)|, the floor E(a − y) (`Eh`). The two agree because r = −s∘a
when U_w and V_w are exact singular vectors. With the f64 vectors they differ by U_wᵀJ(d_N − V_w a), which the
stored records put at 10⁻¹³ of |r_i| in the best-resolved directions and at most 4×10⁻⁸ in any (start A's eleven band
steps, raw `I845-mac/`). For exact vectors L = 0, and d_s is the step of the system bordered with U_w and V_w, as the
docstring says: J d + U_wξ = −F with V_wᵀd = 0 gives d = d_N − V_w(V_wᵀd_N).

**The chord correction.** At the trial state t = z + d_s + V_w y the residual is the model's plus a second-order
term q. A chord step with the iteration's factors is c = −J⁻¹F(t) = V_w(a − y) − J⁻¹q. Its first part is the
held-back band step again, which the code subtracts as the f64 vector `held`. Its second part is the correction.
The code removes the band part of that as well, by V_wV_wᵀ, so a corrector moves the state in the resolved
directions only. The subtraction is exact to the rounding of `held`, about 10⁻¹⁶|a − y|, which lies below the floor
E(a − y) ≈ 1.5×10⁻¹⁴|a − y|.

**Convergence.** The function says converged in two places only: at the top of an iteration, when max|F| < tol for
the F that `Solver.newton` returned, and after the last iteration, on `Solver.residual`. Both are the whole residual
vector of the kernel's system. The damping, the floor and the split enter the step and the acceptance of a step, and
nothing else.

## 2. The claims against the code

Line numbers are `sigma_ms/smdriver.py`'s at 7f88c5bb. "Records" are the 80 run records and the logs of raw
`I845-mac/`, `I913-mac/`, `I915-mac/`, `I916-mac/` and `I917-mac/`.

**The docstring.**

| # | claim | lines | verdict |
|---|---|---|---|
| D1 | Newton's step has components of 10² to 10⁴ along the band | | holds in the records: 214 and 1.19×10⁴ at the seeds of starts B and A |
| D2 | each iteration takes the exact step (`Solver.newton`) and an f64 SVD of the dense Jacobian | 317, 346–347 | holds. `Solver.newton` is the residual, the exact Jacobian and multidouble's dense LU (`src/python.rs` 388–399), and its step equals `multidouble.dd_solve` of `Solver.jacobian` bit for bit (§3). An iteration that takes Newton's own step makes no SVD (D17) |
| D3 | the band is the singular triplets with σ_i < `band_tol` σ_max | 348–354 | holds |
| D4 | B = \|U_wᵀF\|₂, S = \|F − U_wU_wᵀF\|₂ | 374–380 | holds, computed in f64 from hi + lo of F |
| D5 | d_s = d_N − V_wV_wᵀd_N is the step of the system bordered with U_w and V_w | 363, 383–385 | holds for exact singular vectors, and up to the floor with the f64 ones (§1). a comes from the f64 value of d_N and d_s is formed in double-double, so d_s + V_w a is d_N to double-double rounding |
| D6 | JV_w is formed in double-double, and the SVD of U_wᵀJV_w rotates U_w and V_w within their spans | 355–361 | holds: `dd_matvec` with the double-double Jacobian, rounded to f64, then an f64 SVD of the k×k matrix, whose factors are orthogonal |
| D7 | the f64 SVD resolves nothing below 10⁻¹⁶ σ_max | | holds in run 3's log: its lowest singular values sit at 4.0×10⁻¹⁶ (σ_max 5.4), while the reduced ones go down to 6×10⁻¹⁸ |
| D8 | \|Ly\| is about 1.5×10⁻¹⁴\|y\| | 362, 386 | a measurement, and it holds: E/\|a\| is 0.8×10⁻¹⁴ to 1.9×10⁻¹⁴ over the 79 band steps of the records, median 1.55×10⁻¹⁴ |
| D9 | while S > max(κB, 2E(a)) the step is d_s alone, halved until S falls | 389, 408–411, 434–439 | holds. The max has a third term, 0.1 tol, that the docstring does not name. After 30 halvings without a fall the function returns with `failed` |
| D10 | otherwise the step is d_s + V_w y, y_i = a_i s_i²/(s_i² + μ²), of length at most the radius | 391–403, 412–418 | holds: μ by 200 bisections in log μ, and all of a when \|a\| is within the radius |
| D11 | up to `correctors` chord steps with the same factors remove the second-order residual | 421–430 | holds. `chord` solves with the factors of the last `newton` call (`src/python.rs` 421–459), and `jacobian` leaves them alone. A corrector is tried while S > max(κB, 1.5E(a − y), 0.1 tol) and kept only if it lowers S. The 1.5 is not in the docstring |
| D12 | the held-back step is taken off the chord step exactly, then the band part of what remains | 422–426 | holds (§1) |
| D13 | a trial is accepted when B has fallen and S is back below max(κB, 2E(a − y)), and the radius is halved otherwise | 431, 435–439 | **differs** (F3): S is also accepted anywhere up to its value before the step |
| D14 | up to `grow` doublings in the same iteration, each kept while it lowers B further | 441–446 | holds, after a first trial that held with μ > 0 |
| D15 | the radius is then the accepted step's length, doubled if the step gained at least half of the model's fall | 465–466 | **differs** in two details (F4) |
| D16 | nothing is added to the equations, near the root μ = 0 and the steps are Newton's own, and the result has max \|F\| < tol | 322, 394–395, 469–471 | holds for a state returned as converged (§1, §3). The function also returns states that did not converge, marked `failed` or `converged: False`, and `run_eta.py` exits 1 on them |
| D17 | a step no longer than the radius is tried whole first, with no SVD, and taken if it lowers \|F\|₂ | 329–341 | holds |
| D18 | `band=False` leaves Newton with backtracking on \|F\|₂ | 344, 374–376 | holds: k = 0 makes S = \|F\|₂ and B = 0 |
| D19 | the default radius is 1% of \|z\|₂ | 314–315 | holds |
| D20 | Keller's Method III and the bordering algorithm of his §4.11 | | not checked here. notes/LIT_REVIEW.md records the book as read in full |

**SIGMA_DSS §11, the band paragraphs** (from "A deflated Newton for starts on the band" to the table of run 4).

| # | claim | checked against | verdict |
|---|---|---|---|
| T1 | the singular vectors below 10⁻⁶ of the largest singular value span the band | the defaults of `newton_band` and `run_eta.py` | holds |
| T2 | while S exceeds a tenth of B, the step is Newton's without its band part, halved until S falls | D9 | holds, κ = 0.1 |
| T3 | up to ten chord steps, and the step is accepted when B has fallen, and shortened otherwise | D11, D13 | holds, with F3 |
| T4 | E ≈ 1.5×10⁻¹⁴ ℓ, and S is asked to come down to 2E, not below it | D8, D9 | holds. The correctors aim at 1.5E |
| T5 | reduced singular values from 3.6×10⁻¹⁰ down to 6×10⁻¹⁸ | records | holds: 3.56×10⁻¹⁰ and 6.40×10⁻¹⁸ at start A's first band step |
| T6 | at start A the four best-resolved directions hold 97% of B² and 0.2% of the squared length of Newton's band step | records | holds: 97.3% and 0.22% |
| T7 | the run that scaled the band step as a whole gained 3% of B per iteration and was stalling after 29 | `bandA3.txt` | **differs slightly** (F6): 2.4% per band iteration, mean and median of 25, and 1.1% over the last five |
| T8 | the damped step converged in 11 band iterations | records | holds |
| T9 | no term is added, and a converged state has max \|F\| below 10⁻²⁸ in the chain's system | D16 | holds |
| T10 | \|F\|² falls only while δ³ < 2Bσ/c² | algebra | holds: (B − σδ)² + c²δ⁴ < B² is c²δ³ + σ²δ < 2Bσ |
| T11 | a first version, a trust region on \|F\|₂, stalled on the toy | | not verifiable: that version is in no commit |
| T12 | the planted table, checks A to D | rerun (§5) | holds, every number |
| T13 | `noband` fails A, `easy` fails B and C, `root` fails A, `gamma` fails D with 6.2×10⁻¹² | rerun (§5) | holds (6.23×10⁻¹²) |
| T14 | the table of starts A and B: iterations, max \|F\|, Δ and its differences from the chain | records | holds |
| T15 | the band is 32 singular values below 10⁻¹⁰ of the largest, then a gap to 10⁻⁴, with 89% of it in q | records | holds: 32 below each of 10⁻⁵ to 10⁻¹⁰ and 33 below 10⁻⁴, and q's share is 88.8% to 89.0% |
| T16 | its largest Δ component is 0.059 | records | **one state's value** (F5) |
| T17 | at the converged states q agrees to 5×10⁻¹⁶, c₀ to 5×10⁻²⁰, the rest to 2×10⁻¹⁷ and Δ to 10⁻²³ | the three stored states | holds: 5.0×10⁻¹⁶, 4.7×10⁻²⁰, 1.8×10⁻¹⁷, 1.1×10⁻²³ |
| T18 | start B's band step was 2 to 9 long, start A's is 1.2×10⁴ at the seed and 73 when the band steps begin | records | holds |
| T19 | the table of run 4's band steps, and B = 5.8×10⁻²⁸ after the last | records | holds, every row |
| T20 | what runs 1, 2 and 3 of start A did | `bandA3.txt` | run 3 holds (29 iterations, B from 8.11×10⁻¹¹ to 4.42×10⁻¹¹, Newton's band step at 170). Runs 1 and 2 were not checked |

#913's table, which is the first value that rests on the function alone, also holds against its records: the
iteration counts, max |F| = 1.05×10⁻³¹, 1.17×10⁻³¹ and 1.31×10⁻³¹, the three Δ, their spread of 2.3×10⁻²³, and
8.8×10⁻²¹ before the extra step.

REST

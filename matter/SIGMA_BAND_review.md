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

## Verdict

**A state that `newton_band` returns as converged is a root of the kernel's unmodified system, to the tolerance it
was given. SIGMA_DSS.md can call the kernel's D ≥ 10 values checked by an independent review, with the two limits
below.**

- **By reading** (§1, §2): the function says converged only when max |F| of the kernel's whole residual is below tol.
  The split, its floor and the damping enter the choice and the acceptance of a step, and nothing is added to the
  equations.
- **By the key check** (§3): from the state that the band solver left at η = 0.3, D 10.5, M = 32, one Newton step
  whose Jacobian is built from `Solver.residual` alone, by finite differences, lands on max |F| = 1.17×10⁻³¹ and
  moves Δ by 9.6×10⁻²², against bounds of 10⁻³⁰ and 10⁻²⁰. The kernel's own Jacobian equals that finite-difference
  one to 1.2×10⁻²¹ in every entry. The three stored starts pass with the kernel's Jacobian. The nearest to the bound
  is the march start, whose Δ moves by 8.8×10⁻²¹.
- **By the stored states** (§3): all 75 states that the band solver's runs call converged, from D 9 to 11 at η = 1
  and 0.3, 55 of them at D ≥ 10, have max |F| below 10⁻²⁸ on their stored plan when the reviewer's script evaluates
  the residual. The largest is 9.6×10⁻²⁹, and each equals its record's value bit for bit.
- **The claims** (§2): of 40 claims of the docstring and of §11's band paragraphs, 33 hold, 4 differ and 3 were not
  checked in full. The four are two sentences of the docstring (F3, F4) and two numbers of §11 (F5, F6).

The two limits:

- **The planted test sees less of the function than its place in §11 suggests (F1).** Four of the reviewer's seven
  mutations of `newton_band` pass it unchanged: the damping inside the band, the floor of the split, the held-back
  step taken off a chord step, and the fall of B in the acceptance rule. No band trial was ever rejected, in the toy
  or in any stored run. This reaches no value, since a converged state is judged by its residual.
- **Δ's error bar is a bound of 8.7×10⁹ max |F| at D 10.5, and §11 quotes a spread (F2).** That is 8.7×10⁻¹⁹ at the
  runs' tol and 8.7×10⁻²² after the extra Newton step, where §11 says about 3×10⁻²³. The 16 digits of the chain
  tables are safe at D 10.5.

The review takes `Solver.residual` as the system. That the residual is the σ-model's equations, and that the
solution is the only one, are outside it (§7).

## Summary of the checks

Criteria were written on the issue before the runs they judge.

| check | criterion | result | verdict |
|---|---|---|---|
| R1, the claims against the code | each holds, differs or is unverified | 33 hold, 4 differ, 3 not checked in full (§2) | OK, with F3 to F6 |
| K1, a Newton step from the residual alone, D 10.5 | max \|F\| ≤ 10⁻³⁰, Δ moves ≤ 10⁻²⁰ | 1.30×10⁻³¹, 9.62×10⁻²² | OK |
| K1 with the Jacobian from two step sizes | the same | 1.17×10⁻³¹, 9.62×10⁻²² | OK |
| K2, the kernel's Jacobian, three starts | the same | ≤ 1.20×10⁻³¹, ≤ 8.77×10⁻²¹ | OK, one at 0.88 of the bound |
| K3, the two Jacobians entry by entry | a difference above 10⁻¹⁵ is a finding | 1.78×10⁻¹⁴ at one step size, which is the finite difference's truncation: 1.2×10⁻²¹ from two step sizes, over all entries | OK after K3b |
| K4, x = J⁻ᵀe_Δ | −x·F equals the step's ΔΔ to 10⁻³ | equal in the 4 digits printed, at three states; \|x\|₁ = 8.67×10⁹ | OK |
| R3, `planted_band.py` and its four mutations | exits 0, 1, 1, 1, 1 with the stored numbers | those exits, and all 32 entries of the record equal | OK |
| R4, the reviewer's `sonly`, named beforehand | the planted test fails | exit 1, by check A | OK |
| R4, six more mutations | reported either way | `loose` and `nochord` fail; `nodamp`, `nofloor`, `noheld`, `anyB` pass | F1 |
| R5, §11's numbers about the band | against the stored records | hold but two (F5, F6) | OK, with F5, F6 |
| R6, every stored state on its plan | max \|F\| < 10⁻²⁸ and equal to the record's to 10⁻³ | 75 states judged, the largest 9.65×10⁻²⁹, all equal bit for bit | OK |
| R6's control, 10⁻²⁰ added to one unknown | must fail | fails at all 75, after the control itself was corrected (§3) | OK |

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
(`Bp`), and the model's resolved residual is |L(a − y)|, the floor E(a − y) (`Eh`). The two ways of writing the band
residual agree because r = −s∘a when U_w and V_w are exact singular vectors. With the f64 vectors they differ by
U_wᵀJ(d_N − V_w a). In the records of start A's eleven band steps (raw `I845-mac/`) |r_i| and s_i|a_i| agree to
5×10⁻¹³ of |r_i| or better in the two best-resolved directions, and to 4×10⁻⁸ in every direction.

For exact vectors L = 0, and d_s is the step of the system bordered with U_w and V_w, as the docstring says:
J d + U_wξ = −F with V_wᵀd = 0 gives d = d_N − V_w(V_wᵀd_N).

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
| D7 | the f64 SVD resolves nothing below 10⁻¹⁶ σ_max | | holds in run 3's log, which prints the four lowest f64 singular values: three at 4.0×10⁻¹⁶ and one at 0.7×10⁻¹⁶ to 2.3×10⁻¹⁶, with σ_max = 5.4. The reduced ones go down to 6×10⁻¹⁸ |
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

## 3. One Newton step from the residual alone

The state is the one the band solver left for the m03hi start at η = 0.3, D 10.5, M = 32
(`I913-mac/out/sm_hi_D10.5_M32.npz`), loaded with its stored plan of Taylor steps. On that plan `Solver.residual`
gives max |F| = 2.494×10⁻³⁰, the value in the run's log. `rv_newton.py` calls `make_solver`, `set_segs` and
`Solver.residual`, and none of `newton_band`, `newton_solve` or `run_eta.py`. The criteria were written on the issue
before the runs: after the step max |F| ≤ 10⁻³⁰, and Δ moves by at most 10⁻²⁰.

| step | Jacobian | \|d\|₂ | Δ moves by | max \|F\| after | verdict |
|---|---|---|---|---|---|
| K1 | central differences of `Solver.residual`, h = 2⁻³³ | 1.565×10⁻¹⁵ | +9.618×10⁻²² | 1.30×10⁻³¹ | OK |
| K1, two step sizes | 4D(h) − D(2h), the h² term removed | 1.565×10⁻¹⁵ | +9.618×10⁻²² | 1.17×10⁻³¹ | OK |
| K2 | `Solver.jacobian` | 1.565×10⁻¹⁵ | +9.618×10⁻²² | 1.08×10⁻³¹ | OK |
| K2, the m03up start (max \|F\| 4.77×10⁻²⁹) | `Solver.jacobian` | 9.86×10⁻¹⁷ | +1.661×10⁻²³ | 1.20×10⁻³¹ | OK |
| K2, the march start (max \|F\| 7.02×10⁻³⁰) | `Solver.jacobian` | 1.371×10⁻¹⁵ | +8.769×10⁻²¹ | 1.11×10⁻³¹ | OK, at 0.88 of the bound |

Every step is solved by `multidouble.dd_solve`, a dense LU in double-double. Δ after the steps is
1.142980986472450205537241, …537250 and …537226, the values of §11's table for #913.

**What K1 shares with `newton_band`.** `Solver.residual`, the driver's construction of the Solver (`make_solver`, the
state file and its plan), and multidouble's double-double arithmetic and LU, which is library code outside the
σ-model kernel. It shares no Jacobian, no factorization and no line of `newton_band`.

**K2 is not independent of `Solver.newton`.** Its step equals `Solver.newton`'s bit for bit at the three states, so
`Solver.newton` is `Solver.jacobian` with multidouble's LU. K2's use is with K3. The F that `Solver.newton` returns,
which `newton_band` tests against tol, equals `Solver.residual`'s bit for bit at the three states.

**K3, the two Jacobians.** At h = 2⁻³³ the finite-difference Jacobian differs from `Solver.jacobian` by at most
1.78×10⁻¹⁴ in an entry (row 286, column 158; Frobenius norm of the difference 8.4×10⁻¹⁴ of 96.1). That is above the
mark of 10⁻¹⁵ that I had set, so I tested what it is. `rv_fdcol.py` takes that column and 40 random ones at h = 2⁻³³,
2⁻³² and 2⁻³¹. The difference falls by 4.00 per halving of h wherever it is above 10⁻²⁰, and the columns
extrapolated to h = 0 equal `Solver.jacobian`'s to 6.1×10⁻²² in every entry. So the difference is the finite
difference's truncation, and on these columns `Solver.jacobian` is the derivative of `Solver.residual` to the
rounding of the differences, about 10⁻³¹/2h = 4×10⁻²².

An error of 1.8×10⁻¹⁴ is more than the band's deepest reduced singular values at this state (2.4×10⁻¹⁰ down to
7.6×10⁻¹⁸), so the first K1 Jacobian is not J inside the band. Its step still lands on 1.30×10⁻³¹ and differs from
Newton's by 1.55×10⁻²² at most, and by 7.7×10⁻³¹ in Δ.

The second K1 run takes every column from two step sizes, 4n residuals. Its Jacobian differs from
`Solver.jacobian` by at most 1.20×10⁻²¹ in any of the 3521² entries (Frobenius norm 6.1×10⁻²⁰). That is below the
deepest singular value, so this step is Newton's in every direction. It lands on max |F| = 1.17×10⁻³¹ and equals
K2's step to 7.0×10⁻²⁷, and to 1.4×10⁻³⁴ in Δ. So at this state the whole of `Solver.jacobian` is the derivative of
`Solver.residual` to 1.2×10⁻²¹, and Newton's step from the residual alone is the kernel's.

**R6, every stored state.** `rv_states.py` loads each state file of the band solver's runs with its plan and
evaluates `Solver.residual` there, in the folders of #845, #913, #915, #916 and #917. Of 77 files, 75 belong to a
record that says converged, all from `--solver band` runs: 23 at η = 1 and 48 at η = 0.3 with M = 32, and one each at
M = 24 and 48 for both couplings. They run from D 9 to 11, 55 of them at D ≥ 10. Every one has max |F| < 10⁻²⁸: the
largest is 9.65×10⁻²⁹, the median 1.4×10⁻³¹, and 60 are below 10⁻³⁰. Every value equals its record's F bit for bit.
The other two files are a checkpoint without a record and the pilot of start B, whose record says not converged
(max |F| = 9.4×10⁻¹²).

The control adds 10⁻²⁰ to the first unknown of each state and fails at all 75, with max |F| = 1.0×10⁻²³. Its first
version added the 10⁻²⁰ to the high part of the double-double value, where it is rounded away, and exited 0. Both
logs are in the raw folder (`states/mut_state_noop.txt`, `states/mut_state.txt`).

## 4. How far Δ can move at a converged state

With x = J⁻ᵀe_Δ, Newton's step moves Δ by −x·F. `rv_newton.py` computes x from `Solver.jacobian` and checks that
identity against the step: +9.618×10⁻²², +1.661×10⁻²³ and +8.769×10⁻²¹ both ways at the three states. At D 10.5,
η = 0.3, M = 32:

```math
|x|_1 = 8.67\times10^{9}, \qquad |x|_2 = 6.36\times10^{8}, \qquad \max_i |x_i| = 1.01\times10^{8} .
```

So to first order a residual of size max |F| leaves Δ undetermined by up to |x|₁ max |F|:

| max \|F\| | \|x\|₁ max \|F\| | what the runs show |
|---|---|---|
| 10⁻²⁸, the runs' tol | 8.7×10⁻¹⁹ | Δ moved by 8.8×10⁻²¹ from the march start's converged state (max \|F\| 7.0×10⁻³⁰, bound 6.1×10⁻²⁰) |
| 10⁻³¹, after one more Newton step | 8.7×10⁻²² | the three states spread over 2.3×10⁻²³ |

10⁻³¹ is the floor that every run reaches (the median of R6's 75 states is 1.4×10⁻³¹), which I take to be the
rounding of the residual's own evaluation (*inferred*). If so, the second row is also how far the true root's Δ can
lie from a polished state's. §11 gives "about 3×10⁻²³" as the level to which the band lets a converged state fix Δ.
That number is a spread, over three starts, over M = 32 against 48, and over two plans. The first-order bound is 38
times larger. The spread is what a residual of random signs gives: |x|₂ times the root-mean-square entry of F is
1.0×10⁻²³ at each of the three polished states (F2).

## 5. The planted test, its mutations and the reviewer's

**Rerun.** `planted_band.py` exits 0. Its record equals `data/planted_band.json` in all 32 scalar entries and in the
10 steps of A's history. The four mutations exit 1, each failing the checks that §11 names: `noband` A, `easy` B and
C, `root` A (max |z − z\*| = 1.00×10⁻¹²), `gamma` D (6.23×10⁻¹²).

**What the four mutations change.** None changes a line of `newton_band`. `noband` switches the deflation off by the
function's own argument, `easy` changes the toy, `root` the reference and `gamma` the kernel. They show that the test
sees a missing deflation, a toy without a band, a wrong reference and a wrong residual.

**The reviewer's mutations.** `rv_band_mut.py` runs `planted_band.py` unchanged on a copy of `newton_band` with one
or two pieces of its text changed. `sonly` was named on the issue before any run as the one the test should catch.
`loose` was written into the script with the same expectation before its run, and it ran first.

| mutation | what changes | exit | which check notices |
|---|---|---|---|
| `none` | nothing (the harness) | 0 | |
| `sonly` | every iteration goes through the split, and a state is converged when S < tol, whatever B is | 1 | A: a state with max \|F\| = 7.6×10⁻²⁴ is returned, at max \|z − z\*\| = 5.5×10⁻¹⁴ |
| `noshort` | `sonly`'s first half alone, which leaves the function correct | 0 | none, as it should: A converges in 9 steps |
| `loose` | converged when max \|F\| < 10¹⁰ tol | 1 | A: a state with max \|F\| = 2.45×10⁻²³ is returned, at max \|z − z\*\| = 1.5×10⁻¹³ |
| `nochord` | no chord corrections | 1 | A: no convergence in 40 iterations, max \|F\| = 7.8×10⁻¹¹ |
| `nodamp` | the band step scaled as a whole, the version that §11 says stalled at start A | 0 | none. A converges in 9 steps, D in 7 |
| `nofloor` | E = 0 in the mode test, the correctors and the acceptance | 0 | none, and every number is unchanged |
| `noheld` | the held-back band step is left in the chord step | 0 | none. A's last max \|F\| is 3.81×10⁻⁴⁶ against 6.24×10⁻⁴⁶, so the changed line runs |
| `anyB` | a band trial is accepted without the fall of B | 0 | none, and every number is unchanged |

D notices none of the seven. Its Jacobian has no band (condition 8×10³, as the script's header says), and under
`sonly` and `loose` its last step carries it below either threshold.

**Why the toy cannot see the two refinements.** A's path is four resolved steps, two band steps and three of
Newton's own. The first band step is limited by the radius (μ = 7.4×10⁻⁹, three doublings, one corrector) and the
second takes Newton's whole band step. No trial is rejected. The toy's band rows are σ_j(b_j + b_j³), so once the
resolved steps have removed the strong offset, Newton's band step is −(b_j + b_j³)/(1 + 3b_j²) in each direction,
whatever σ_j is. At the first band step it is 0.39 long, and the best-resolved direction holds 99.6% of B² and 44%
of the squared length. At start A the four best-resolved directions hold 97% of B² and 0.2% of the squared length:
the toy does not have the difficulty that the damping answers. The floor is 2×10⁻¹⁵ there, against
S = 8×10⁻¹³ and κB = 3×10⁻¹⁰, so it never decides anything.

**The stored runs of the real system** (the 80 records of §2). With the current code there are 79 band steps, 14
limited by the radius and 65 whole, 2 with a doubling and none with a halving. There are 47 resolved steps, none
halved, and 282 of Newton's own. No record ends on a failed line search. So no band trial has been rejected in any
stored run or in the toy.

The band has 32 members in 140 of the 142 deflated iterations, with 32 singular values below each of 10⁻⁵ to 10⁻¹⁰
of the largest and 33 below 10⁻⁴. The other two are first iterations from predicted seeds of #916 (max |F| =
2.5×10⁻³), where the spectrum has no gap and the threshold of 10⁻⁶ takes 30. So the band's size is a property of the
state near the solution, and E/|a| stays between 0.8×10⁻¹⁴ and 1.9×10⁻¹⁴ at every D from 10.1 to 11.

| part of the function | the planted test | the real runs |
|---|---|---|
| the deflation | `noband`, and check C | #652's starts A and B fail without it and converge with it |
| the convergence test | `sonly`, `loose` | K1, R6 |
| the chord corrections | `nochord` | 2 to 10 corrections in every band step of run 4 |
| the damping inside the band | not seen | run 3 stalls without it, run 4 converges (one start) |
| the floor of the split | not seen | run 3's S follows E, 2.7×10⁻¹² at E = 2.7×10⁻¹² in its last step. Run 1, without the floor, is described in raw `I845-mac/README.md` and was not checked |
| taking the held-back step off a chord step | not seen | none |
| the fall of B, and the halving of the radius | not seen | never used: no trial was rejected |

## 6. Findings

By severity. None changes a value of Δ.

**F1. The planted test does not see four parts of the function.** Check coverage, no number changes. `nodamp`,
`nofloor`, `noheld` and `anyB` pass `planted_band.py` (§5), and no band trial has been rejected in any stored run. The
two refinements that §11 describes rest on one start, A's run 4 against its runs 1 and 3, and the rejection of a
band step has never run. This does not reach a value: a converged state is judged by its residual, whatever path
led to it (§1, §3). It does mean that a later change to these parts would pass every check there is. *Fix:* either a
second toy on which they matter, with these four mutations as its controls (recommended task 1), or a sentence in
§11 saying that the planted test covers the deflation, the chord corrections and the convergence test.

**F2. Δ's error bar at a converged state is |x|₁ max |F|, and §11 quotes a spread.** Statement precision, no number
changes. At D 10.5 the first-order bound is 8.7×10⁻¹⁹ at the runs' tol and 8.7×10⁻²² after the extra Newton step
(§4), against "about 3×10⁻²³" in §11, which is a spread. The last three digits of §11's 24-digit values are below
the bound. The chain tables' 16 digits are safe at D 10.5, by a factor 600 at tol. Whether |x|₁ grows along the chain
is not known.
*Fix:* name the 3×10⁻²³ as a spread and give the bound beside it, and compute |x|₁ at the chain's points
(recommended task 2).

**F3. The acceptance rule lets S stay as high as it was.** Text, the docstring. Line 431 accepts a band trial when
S ≤ max(κB, S before the step, 2E(a − y), 0.1 tol), and the docstring says "S is back below max(kappa B, 2 E(a −
y))". In the records 10 of 79 accepted band steps end with S above κB. Whether they were under 2E(a − y) or only under
the old S cannot be told, because E(a − y) is not recorded. A state left off the valley gets a resolved step in the
next iteration, so the result is not affected. *Fix:* the docstring's sentence, and E(a − y) in the record.

**F4. The radius rule differs from the docstring in two details.** Text, the docstring. Line 465 changes the radius
only after a step that the radius limited (μ > 0), and doubles it only if no halving was needed. The docstring says
the radius is then the accepted step's length, doubled on a gain of half the model's fall. *Fix:* the sentence.

**F5. "Its largest Δ component is 0.059" is one state's value.** Text, a number of §11. 0.059 is `band_Delta` at
start B's first band step. Over the D 10.475 runs it is 0.059 to 0.083, and 0.071 at the solution. At η = 0.3,
D 10.5 it is 0.090 to 0.119, and 0.105 at the solution. It is the largest entry of the Δ row of V_w in the band's
own basis. *Fix:* quote the value at the solution, 0.071.

**F6. "3% of B per iteration" is 2.4%.** Text, a number of §11. Run 3's 25 band iterations lower B by 2.4% each, in
the mean and in the median, between 0.0% and 5.0%, and by 1.1% over the last five (`I845-mac/out/bandA3.txt`).
*Fix:* 2.4%.

**F7. Three thresholds are not in the docstring.** Text, no action needed. The 0.1 tol in the three maxima (lines
389, 421, 431), the 1.5 E(a − y) at which the correctors stop (line 421), and the 10⁻⁴ of the model's fall that B
must gain (line 431).

## 7. What I did not check

- That the kernel's residual is the σ-model's equations: `src/model.rs`, its planted test `planted.py` and the
  comparison with sigdss are outside this task. Every check here takes `Solver.residual` as the system.
- Whether the solution is the only one at any D. §11 does not claim it.
- A Newton step at η = 1, at M = 24 or 48, or at another D. K1, K2 and K4 ran at η = 0.3, D 10.5, M = 32. R6
  covers the other stored states by their residual.
- `chain.py`, `renode.py` and `chain_table.py`, and `run_eta.py` beyond how it calls the solver and reads its
  result. The quad-double path `newton_qd`.
- Runs 1 and 2 of start A, and the first version of the function (T11).
- Keller's lectures (D20).
- The note of the private repository that the issue names. I did not read it.
- The kernel module is the build of #652, made on 2026-11-14 after the last change to `sigma_ms/src`
  (590f1413). I used it as it is and did not rebuild it.

## 8. Reproduce

`SITE` is the site directory of the kernel's build, and the raw-data repository is cloned beside the checkout
(`git clone https://github.com/nmoinvaz/choptuik-large-d-raw ../choptuik-large-d-raw`). From `matter/sigma_ms`, with
python 3.14:

```
export PYTHONPATH=$SITE
python3 planted_band.py                                  # PLANTED BAND: PASS, exit 0; --mutate NAME exits 1
python3 review_922/rv_band_mut.py --mutate sonly         # PLANTED BAND: FAIL, exit 1 (loose, nochord also)
python3 review_922/rv_band_mut.py --mutate nodamp        # PLANTED BAND: PASS, exit 0 (nofloor, noheld, anyB also)
python3 review_922/rv_newton.py --state ../../../choptuik-large-d-raw/I913-mac/out/sm_hi_D10.5_M32.npz --fd
python3 review_922/rv_fdcol.py --state ../../../choptuik-large-d-raw/I913-mac/out/sm_hi_D10.5_M32.npz --cols 158 --random 40
python3 review_922/rv_states.py ../../../choptuik-large-d-raw --folders I845-mac,I913-mac,I915-mac,I916-mac,I917-mac
```

The fourth prints `RV_NEWTON (sm_hi_D10.5_M32.npz): PASS` after about 10 minutes, with `--richardson` after about
18, and without `--fd` after 45 s. The fifth prints `K3b (41 columns): ... 6.08e-22: PASS`. The sixth prints
`R6: 77 state files, 75 judged, 0 fail; largest max |F| of the judged 9.646e-29; ... PASS`, and exits 1 with
`--mutate state`. `review_922/run_i922.sh` runs them all with their time caps, and raw `I922-mac/exits.txt` has each
run's exit status.

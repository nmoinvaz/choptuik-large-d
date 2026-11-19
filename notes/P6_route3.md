# P6 route 3: are Lemma W's O(1/j) cancellations identities in ν?

Stage 3 of [P6_scope.md](P6_scope.md) §4, issue #982. Lemma W ([proof/B_tail_M7.md](../proof/B_tail_M7.md) §8.2) says
the weight e^{πνω/2} cancels exactly in each tuning step. The Darboux sums that define the normalised multiplier
converge only after two exact cancellations of their O(1/j) parts, which [proof/D_entire.md](../proof/D_entire.md) §6.2
shows for harmonic 1 at order 3. Route 3 of P6_scope.md §3 needs them for every ν. This note tests that with sympy. The
script is `solver/proof/route3_cancel.py`. Written 2026-11-19 by Claude (Anthropic) under Nathan Moinvaziri's direction;
not reviewed. Status words as in AGENTS.md.

## Result

- **The O(1/j) cancellations hold identically in ν, in the one-step model of §1** (*checked symbolically*). Every
  family's O(1/j) coefficient vanishes for symbolic ν, ω and input amplitudes. Both mutations, one per cancellation,
  make it fail (§2).
- **The model reproduces the known case exactly.** At harmonic 1 it equals `exact_sigma._sym_tail` term for term, and
  its j⁻² limits are D_entire.md §6.2's i(7 − ω² − 2iω)/8 and i(7 − ω² + 5iω)/16.
- **The j⁻² remainder is not uniform in ν.** Its coefficients grow like ν in the families ν and ν ± 2 and like ν² in the
  families ±1 (§3). So the cancellations remove the O(1/j) obstruction for every ν, but a remainder bound C/j² with C
  independent of ν does not follow from them.
- **Numerical spot checks at ν = 3, 5, 7 agree** (§4). An exact recursion that does not use the sympy expressions has no
  O(1/j) part, and the sympy decomposition matches it to O(j⁻³). The controls fire as set in 47 of 48 runs. In one,
  `near2` at ν = 5 and small ω, interference between families hides a missing j⁻² term from the growth test, so
  `--matrix` exits 1. The cause is shown and the bar is unchanged.
- **Decision** (P6_scope.md §4, stage 3). The identities hold, so route 3 does not stop here. Its next gate is the
  remainder bound, which now has a concrete form to test: a j⁻² coefficient of order ν, and a range j ≲ ν² where the
  1/j expansion does not apply (§5, §6).

## 1. The model

The increment of the normalised multiplier sequence of harmonic ν at order 3 is (D_entire.md §6.1, in amp_flint's
scaled variables)

```math
\frac{\Delta R_j}{K_\nu} = -\frac{b_j}{T^\nu_{j+1}} + \sum_\lambda E[\nu-\lambda]\,d_0^\lambda\big(\rho^\lambda_{j+1}-\rho^\lambda_j\big) + h_2\,d_0^\nu\Big(\frac{1}{\alpha_\nu+j}+\frac{1}{\beta_\nu+j}\Big), \qquad \rho^\lambda_j = \frac{T^\lambda_j}{T^\nu_j} .
```

Here b_j = c_j S^s_j + S^d_{j+1} is the order-3 forcing, a sum of convolutions of the order-2 lapse g with the order-1
sequences. The de-mixing constant is E[h] = −F²_h/(ihω) for h ≠ 0, with F²_h summed over ordered label pairs λ₁ + λ₂ = h
and F_λ = d₀^λ/2. The normalisation constant is h₂ = F²₀/2. Every order-1 input of label λ is its exact Kummer sequence
d₀^λ T^λ_l with T^λ_l = (α_λ)_l (β_λ)_l / l!.

`tail()` splits ΔR_j/K_ν into families, Σ_λ C_λ(j) ρ^λ_j + O(j⁻³), with C_λ rational in j. It keeps the terms of
`_sym_tail`: the near end i = 1, 2 of both convolutions, the far end i = j + 1 at l = 0 and l = j, the de-mixing and the
ψ normalisation. Three cases are run.

| case | inputs (label, d₀) | target | what it is |
|---|---|---|---|
| h1 | (+1, −i), (−1, +i) | 1 | D_entire.md §6.1–6.2's case, to be reproduced |
| h3 | the same | 3 | harmonic 3 at order 3, its first tuned order, exactly |
| ν | (+1, b₊), (−1, b₋), and (ν − 2, p₋), (ν, p₀), (ν + 2, p₊), the terms linear in the p's | ν, symbolic | one A² step: a linear harmonic of label ν − 2, ν or ν + 2 acting on harmonic ν through the A sin θ background |

The ν case is not literally "the first tuned order of harmonic ν". For ν ≥ 5 that order is forced by the tuned
top-degree sequences of every lower harmonic, so it has no fixed symbolic form. The ν case keeps the Kummer part of each
input, which is the 2f T part of Lemma L's top-degree harmonic (B_tail_M7.md §6). §5 lists what that leaves out.

## 2. The cancellations

`python3 route3_cancel.py`, from `solver/proof/` (venv-314: python-flint 0.9.0, FLINT 3.6.0, sympy 1.14.0; about 3 s):

| case | family | pair that cancels | lim j C_λ(j) | under `--mutate=demix` (E × 3/2) | under `--mutate=psi` (h₂ × 3/2) |
|---|---|---|---|---|---|
| h1 | +1 | near-end terms of lapse harmonic 0, against ψ | 0 | 0 | nonzero |
| h1 | −1 | near-end terms of lapse harmonic 2, against de-mixing | 0 | nonzero | 0 |
| h3 | +1 | near-end terms of lapse harmonic 2, against de-mixing | 0 | nonzero | 0 |
| ν | ν | lapse harmonic 0 (background), against ψ | 0 | 0 | nonzero |
| ν | ν − 2, ν + 2 | lapse harmonic ±2 (background), against de-mixing | 0 | nonzero | 0 |
| ν | +1, −1 | lapse harmonic ν ∓ 1 (background × perturbation), against de-mixing | 0 | nonzero | 0 |

Each zero is an exact sympy result for symbolic ν, ω, b₊, b₋, p₋, p₀ and p₊. The script exits 0, and it exits 1 under
each mutation. Under `--mutate=demix` the coefficients become −i/8 (h1, family −1), i/8 (h3), b₊²p₋/8 (family ν − 2),
b₋²p₊/8 (family ν + 2), b₊(b₋p₀ + b₊p₋)/4 (family +1) and b₋(b₋p₊ + b₊p₀)/4 (family −1). Under `--mutate=psi` they
become −i/4 (h1, family +1) and b₊b₋p₀/4 (family ν). None of them depends on ν or ω. The mutation factor is 3/2 so that
the numeric test of §4 can resolve it. The symbolic test detects any factor other than 1. A series engine (truncated
Laurent series in 1/j, term by term) computes the ν case, and the script checks it against the exact rational functions
on h1 and h3.

The mechanism, as a reading that the symbolic result is consistent with: at m = 1 the order-2 lapse of harmonic h is
g₁[h] = F²_h/2 = (ihω/2) P_h, where P_h is the order-2 harmonic of Π. The near-end i = 1 terms give −2 g₁[h] d₀^λ ρ/j at
leading order, and the de-mixing gives E[h] d₀^λ(ρ_{j+1} − ρ_j) ≈ −P_h(−ihω/j) d₀^λ ρ, the same with the other sign. For
h = 0 the partner is ψ, with h₂ = F²₀/2 = g₁[0]. Neither pairing involves ν beyond h = ν − λ, which cancels. That is
also why the mutated coefficients above are free of ν and ω.

## 3. The j⁻² remainder

`python3 route3_cancel.py --c2` prints lim j² C_λ(j) for the ν case at the A sin θ background (b₊ = −i, b₋ = i), and its
leading behaviour as ν → ∞ per perturbation amplitude:

| family | amplitude | lim j² C_λ, as ν → ∞ |
|---|---|---|
| ν | p₀ | ν · iω/4 |
| ν − 2 | p₋ | ν · (−iω/8) |
| ν + 2 | p₊ | ν · (−iω/8) |
| +1 | p₋, p₀ | ν² · (∓ω²/16) |
| −1 | p₀, p₊ | ν² · (±ω²/16) |

So the coefficient of j⁻² grows like ν, or ν² in the families ±1. What that means for a bound depends on |ρ^λ_j|, which
multiplies each family.

- For |λ| < |ν|, |ρ^λ_j| is below 1 and decreasing in j, since each step ratio has
  |a^λ_m/a^ν_m|² = ((m + ½)² + λ²ω²/4)((m + 1)² + λ²ω²/4) / (((m + ½)² + ν²ω²/4)((m + 1)² + ν²ω²/4)) < 1
  (*proved*, elementary). This covers the families ν − 2 and ±1, so the ν² of the families ±1 comes with a small factor
  once j is past the first few steps.
- For the family ν, ρ = 1, and the coefficient iνω/4 · p₀ stands as it is.
- For |λ| > |ν| the ratio is above 1 and increasing, toward |K_{ν+2}/K_ν| ≍ e^{πω}. This is the untuned label ν + 2
  acting on ν. In the tuned problem the new harmonic's own family is tuned away (Ŝ_{ν+2} = 0), which this model does not
  include.

Against the cancelled O(1/j) pieces, whose size does not grow with ν, the j⁻² terms are larger by a factor of order ν/j
or ν²/j. So the large-j form needs j ≫ ν in the families ν and ν ± 2, and j ≫ ν² in the families ±1. That is a reading
of the coefficients above, not a bound.

## 4. Numerical spot checks

**Planted truth for the recursion.** `python3 route3_cancel.py --selfcheck` runs the numeric recursion `order3()` on the
background alone at harmonic 1. It matches `exact_sigma.sequences`, an earlier implementation of the same recursion, to
3.7×10⁻⁷⁰ relative for j ≤ 200 (bar 10⁻⁶⁰). Under `--mutate=selfcheck` (the sign of d₀ on label −1) the difference is
2.0 and the script exits 1.

**The ν case at ν = 3, 5, 7.** `python3 route3_cancel.py --matrix` (J = 800) computes ΔR_j/K_ν from the exact recursion,
without `tail()`, and divides it by Σ_λ |ρ^λ_j|. It then compares the maxima over j ∈ [J/8, J/4] and [J/2, J], four
times apart:

- the O(1/j) test: the ratio of the maxima of j|ΔR_j/K| must be below 0.5 (about 0.25 if no O(1/j) part is left, about
  1 if one is);
- the remainder test: the ratio of the maxima of j³|ΔR_j/K − Σ C_λ ρ^λ_j|, with `tail()`'s C_λ, must be below 2 (about 4
  if a j⁻² term is missing).

Each family is visible only where it is not dominated by the others, because the |ρ^λ| differ by up to e^{πω}. So the
matrix runs one perturbation at a time at the physical ω (wf = 1), and all three at wf = 0.05, where every |ρ| is of
order 1. The pass and fail pattern of each mutation was written in `matrix()` before the runs.

Ratios from `--matrix` (J = 800, 105 s on one process at nice 10). The O(1/j) test passes below 0.5 and the remainder
test below 2:

| ν | wf | perturbation | none: 1/j, remainder | demix: 1/j | psi: 1/j | near2: remainder | pattern as set |
|---|---|---|---|---|---|---|---|
| 3 | 1 | ν − 2 | 0.247, 0.966 | 0.980 | 0.247 | 3.988 | yes |
| 3 | 1 | ν | 0.264, 1.009 | 0.264 | 0.969 | 3.981 | yes |
| 3 | 1 | ν + 2 | 0.264, 1.063 | 0.932 | 0.264 | 4.006 | yes |
| 3 | 0.05 | all three | 0.267, 1.031 | 0.879 | 1.065 | 3.786 | yes |
| 5 | 1 | ν − 2 | 0.275, 0.931 | 0.984 | 0.275 | 3.993 | yes |
| 5 | 1 | ν | 0.267, 1.011 | 0.267 | 0.956 | 3.992 | yes |
| 5 | 1 | ν + 2 | 0.267, 1.099 | 0.908 | 0.267 | 4.023 | yes |
| 5 | 0.05 | all three | 0.121, 0.547 | 1.088 | 1.040 | 1.280 | **no**: near2 passes |
| 7 | 1 | ν − 2 | 0.273, 0.909 | 0.980 | 0.273 | 4.004 | yes |
| 7 | 1 | ν | 0.269, 1.015 | 0.269 | 0.938 | 4.008 | yes |
| 7 | 1 | ν + 2 | 0.270, 1.138 | 0.883 | 0.270 | 4.045 | yes |
| 7 | 0.05 | all three | 0.364, 1.576 | 1.235 | 1.030 | 8.465 | yes |

The unmutated model passes both tests in all twelve runs, so it has no O(1/j) part at ν = 3, 5 and 7, and `tail()`'s
C_λ match the exact recursion through O(j⁻²). The de-mixing and ψ mutations fire in every run where their family is
visible, and stay silent where it is not, as set.

**One control did not fire.** At ν = 5, wf = 0.05, with all three perturbations, `near2` passes the remainder test
(ratio 1.280 against the bar 2), so `--matrix` exits 1. The missing j⁻² term is there. The remainder's maxima rise from
0.65 and 0.36 to 26 and 34, as in the other runs. But the growth ratio stays low. With one perturbation at a time at the
same ν and wf, `near2` gives 2.52, 3.25 and 3.04 and fires (`--numeric 5 800 --wf=0.05 --pert=m|0|p --mutate=near2`).
The cause is interference. At small ω the five family terms rotate slowly, and their sum's maximum over a window does
not grow like j. The bar was not changed. A test that fits each family's j⁻² coefficient against the known ρ^λ_j would
not have this blind spot, but it was not written.

## 5. What this does not cover

1. **Tuned inputs.** For ν ≥ 5 the inputs at harmonic ν's first tuned order are tuned sequences. Their own family is
   removed (Ŝ = 0), so their late orders follow other families. My reading, not checked: for an input of θ-label λ whose
   late orders follow a family κ ≠ λ, the one-step pairing of §2 leaves a term proportional to κ − λ. In the full
   problem something else cancels it, since the increments at orders 5 to 9 fall like log^k j/j² (D_entire.md §6.6,
   numerical).
2. **Orders beyond one A² step.** Only order 3 is built.
3. **The early range of j.** The 1/j expansion says nothing for j ≲ ν, or j ≲ ν² in the families ±1 (§3). That range is
   where Lemma W's term-by-term majorant Σ |b_i|/(½)_{i+1} and the ν-monotonicity of |T^{(ν)}_i| would be used.
4. **Independence.** `tail()` and `order3()` share only the model's definition (the channels, E and h₂, taken from
   amp_flint's `pass_even`, `pass_odd`, `theta_series` and `S_at`). Both were written by the same author in one session.
   `order3()` is checked against `exact_sigma.sequences`, which a different session wrote. No other agent has reviewed
   either.

## 6. Next step for route 3

The remainder bound. The cheapest first step is to measure, with `--numeric`, how j³|ΔR_j/K − Σ C_λ ρ^λ_j| / Σ|ρ| grows
with ν at ν = 3 to 21. That takes minutes per ν at J = 800 on one process, and it gives the exponent a bound must carry.
After that comes the log-convexity bound of D_entire.md §6.3 (`remainder_sup`) written with ν symbolic. The decision it
informs is whether the formal (★) of route 3 can hold with a remainder C ν^p / j³ for j ≥ j₀(ν), and what j₀(ν) is. A
proof of (★) at A* still needs each increment's whole series (B_tail_M7.md §8.3), which this route does not supply.

## 7. Proposed text (for the coordinator)

**notes/P6_scope.md** §4, stage 3 row, decision cell, append: "Done (#982, [P6_route3.md](P6_route3.md)): the O(1/j)
cancellations are identities in ν in the one-A²-step model (*checked symbolically*), and the j⁻² remainder grows like ν
and ν². Next: the remainder bound."

**proof/B_tail_M7.md** §8.2, after "…it was not completed here.", add: "In a one-A²-step model, with Kummer inputs of
labels ν − 2, ν and ν + 2 on the A sin θ background, the O(1/j) cancellations hold identically in ν (*checked
symbolically*, [notes/P6_route3.md](../notes/P6_route3.md)). The j⁻² remainder grows like ν and ν², so the
ν-uniform bound is still open."

## 8. Reproduce

From `solver/proof/` with venv-314 (`/Users/nathan/venv-314/bin/python` on the Mac):

```
python3 route3_cancel.py                       # the cancellations, h1 against _sym_tail; PASS, exit 0 (~3 s)
python3 route3_cancel.py --mutate=demix        # FAIL, exit 1
python3 route3_cancel.py --mutate=psi          # FAIL, exit 1
python3 route3_cancel.py --c2                  # the j^-2 coefficients of the nu case and their growth in nu (~5 s)
python3 route3_cancel.py --selfcheck           # order3 against exact_sigma.sequences, 3.7e-70; PASS
python3 route3_cancel.py --selfcheck --mutate=selfcheck   # FAIL, exit 1
python3 route3_cancel.py --matrix              # nu = 3, 5, 7, every mode and mutation (~2 min); exits 1 today, on the
                                               # one near2 row of section 4
python3 route3_cancel.py --numeric 5 800 --wf=0.05 --pert=0 --mutate=near2   # that row with one perturbation: FAIL
```

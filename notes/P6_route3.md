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
- **The j⁻² remainder grows polynomially** (§9, #986). It grows exactly like ν and ν² (*checked symbolically*), and the
  fitted j⁻³ coefficient grows like about ν² over ν = 9 to 21 (numerical evidence). e^{−0.736ν} absorbs both with a
  constant of at most 2.55, and the families ±1 are suppressed by |ρ^{±1}| ≈ 10^{−2.5(ν−1)}.
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

*Done in §9 (#986): the growth is measured and polynomial. The text below is the plan as it stood.*

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

## 9. How the j⁻² remainder grows with ν (#986)

Script `solver/proof/route3_growth.py`, on `route3_cancel.py`'s `tail()` and its numeric increment. That increment now
sits in `dr_builder()` with a degree argument, and `route3_cancel.py`'s checks give the same numbers as before.

**Verdict.** The growth is polynomial, and the decay that (★) allows absorbs it.

- lim j² C_λ is a polynomial in ν, of degree 1 in the families ν and ν ± 2 and of degree 2 in the families ±1
  (*checked symbolically*, §9.1).
- The fitted j⁻² coefficient of the exact recursion equals it at ν = 3 to 21 in all 30 runs. The fitted j⁻³ coefficient
  grows like ν to a power between 1.6 and 2.4 over ν = 9 to 21 (numerical evidence, §9.2).
- Against (★): sup over odd ν ≥ 3 of |c₂(ν)| e^{−0.736ν} is at most 2.55 per unit amplitude in every family, reached at
  ν = 3 or 5. So the polynomial growth costs a constant, not a rate (§9.3).
- Weighted by |ρ^λ_j| at the physical ω, the family ν + 2 dominates by about 10⁵, then the family ν. The ±1 families'
  ν² comes with |ρ^{±1}_j| ≈ 10^{−2.5(ν−1)} (§9.4).
- Within the Darboux sums' range the remainder exceeds its family's cancelled 1/j piece only in the families ±1:
  - at j = 600, for ν ≥ 17 to 21;
  - at j = 1600, for ν ≥ 29 to 33.

  In the families ν and ν ± 2 that happens only from ν ≈ 311 at j = 600 and ν ≈ 835 at j = 1600 (§9.4).

### 9.1 The exact coefficients

`python3 route3_growth.py --sym` (from `solver/proof/`, about 3 s). The background is b₊ = −i, b₋ = i, and every value
is per unit perturbation amplitude:

| family | perturbation | lim j² C_λ, exact | degree in ν | cancelling piece, coefficient of 1/j |
|---|---|---|---|---|
| ν | p₀ | (2iνω + ω² − 7)/8 | 1 | 1/2 (ψ) |
| ν − 2 | p₋ | −(2iνω + ω² − 7iω − 7)/16 | 1 | −1/4 (de-mixing) |
| ν + 2 | p₊ | −(2iνω + ω² + 7iω − 7)/16 | 1 | −1/4 (de-mixing) |
| +1 | p₋ | −(ν²ω² − 4νω² − 3iνω + 5ω² + 7iω − 14)/16 | 2 | −1/2 (de-mixing) |
| +1 | p₀ | (ν²ω² − 3iνω + ω² + 7iω − 14)/16 | 2 | 1/2 (de-mixing) |
| −1 | p₀ | (ν²ω² − 3iνω + ω² − 7iω − 14)/16 | 2 | 1/2 (de-mixing) |
| −1 | p₊ | −(ν²ω² + 4νω² − 3iνω + 5ω² − 7iω − 14)/16 | 2 | −1/2 (de-mixing) |

The cancelling pieces do not depend on ν. Each is computed as twice the O(1/j) coefficient under the 3/2 mutation of its
term. At ν = 1 the families ν and ν − 2 give exactly D_entire.md §6.2's j²F and j²G, with p₀ = −i and p₋ = i (sympy).
The planted fit of §9.2 recovers those two values numerically.

### 9.2 The fit on the exact recursion

`python3 route3_growth.py --all` (about 5 min on one process at nice 10). The exact recursion runs at J = 1600 in
512-bit balls, one perturbation at a time at the physical ω. A least-squares fit of j² ΔR_j/K_ν against Σ_λ ρ^λ_j Σ_k
a_{λ,k} j^{−k} on 128 points of [J/4, J] follows, with the exact ρ^λ_j. The fitted families are those whose
|ρ| is within 10⁻¹² of the largest. a_{λ,0} is the measured c₂ and a_{λ,1} the measured c₃. The error estimate is the
spread over K = 6, 8, 10 terms and the ranges [J/4, J] and [J/2, J]. The pass rule, fixed before the runs, is
|a₀ − c₂| ≤ 10 × spread, with the spread at most 10⁻³ |c₂|.

- **Planted truth.** The same fit on the harmonic-1 background (ν = 1, no perturbation) recovers j²F and j²G to
  2.5×10⁻¹⁶ and 3.0×10⁻¹⁶, with spreads 1.4×10⁻¹² and 1.7×10⁻¹² (`--planted`). With h₂ × 3/2 in the recursion it fails
  by 2.3×10³ (`--planted --mutate=psi`).
- **ν = 3 to 21.** Every run passes:

| ν | max \|fit − symbolic\| | max spread | \|c₃\|, ν − 2 | \|c₃\|, ν | \|c₃\|, ν + 2 | pass |
|---|---|---|---|---|---|---|
| 3 | 3.5e-13 | 6.4e-10 | 18.6 | 48.47 | 63.93 | yes |
| 5 | 2.6e-12 | 9.1e-8 | 18.09 | 108.8 | 113.8 | yes |
| 7 | 3.1e-11 | 1.1e-7 | 41.25 | 197.2 | 178.0 | yes |
| 9 | 1.4e-10 | 6.9e-8 | 79.01 | 314.4 | 256.8 | yes |
| 11 | 4.9e-10 | 1.9e-7 | 131.6 | 460.6 | 350.1 | yes |
| 13 | 1.5e-9 | 4.4e-7 | 198.9 | 636.0 | 457.9 | yes |
| 15 | 3.9e-9 | 9.4e-7 | 280.9 | 840.6 | 580.4 | yes |
| 17 | 9.3e-9 | 1.9e-6 | 377.5 | 1074.0 | 717.4 | yes |
| 19 | 2.0e-8 | 3.5e-6 | 488.8 | 1337.0 | 869.0 | yes |
| 21 | 4.2e-8 | 6.1e-6 | 614.7 | 1630.0 | 1035.0 | yes |

  At ν = 3 the label ν − 2 is the background's +1, so that column is the merged family, and its symbolic value is the
  sum of both rows of §9.1.
- **The growth of c₃.** Least-squares slopes of log|c| against log ν over ν = 9 to 21:

  | family | slope of c₂ (symbolic) | slope of c₃ (fit) |
  |---|---|---|
  | ν − 2 | 1.34 | 2.42 |
  | ν | 0.99 | 1.94 |
  | ν + 2 | 0.79 | 1.65 |

  The c₂ slopes are not yet the asymptotic 1 because of the constant terms. The c₃ slopes are near 2. So, as a reading
  of numerical evidence, c₃/c₂ grows like ν, and the 1/j expansion in these families is in powers of about ν/j.
- **The families ±1.** These are invisible at the physical ω, so they were checked at wf = 0.05, where every |ρ| is of
  order 1 (`--fit NU --wf=0.05 --others` at ν = 3, 9, 15, 21). All 27 fits that resolve their family, 12 dominant and 15
  others, agree with §9.1 to at most 5.7×10⁻¹³, and the ±1 families are among them.
- **Controls.** `--fit 3 --mutate=demix` fails all three families. At ν = 21, `--pert=0 --mutate=psi` and
  `--pert=p --mutate=demix` fail by 1.2×10³ and 9.6×10².

### 9.3 Against the decay (★) allows

(★) lets the coupling of the pair ν grow like e^{σν} with σ ≤ 0.736 (B_tail_M7.md §8.4). A polynomial fits under any
σ > 0. The constant it costs at σ = 0.736 is sup over odd ν ≥ 3 of |c₂(ν)| e^{−0.736ν}, which `--sym` evaluates from the
exact polynomials at the physical ω:

| family, perturbation | ν, p₀ | ν − 2, p₋ | ν + 2, p₊ | +1, p₋ | +1, p₀ | −1, p₀ | −1, p₊ |
|---|---|---|---|---|---|---|---|
| sup \|c₂\| e^{−0.736ν} | 0.332 | 0.0585 | 0.345 | 0.214 | 0.909 | 1.00 | 2.55 |
| at ν | 3 | 3 | 3 | 5 | 3 | 3 | 3 |

So the j⁻² remainder adds at most a constant of order 1 to (★)'s Λ, not a rate. The fitted c₃ grows polynomially as
well, over the range measured. This verdict covers the coefficients, not the full remainder at j ≲ ν (§9.5).

### 9.4 Which families dominate, and where the remainder overtakes the leading term

**By weight.** Each family enters ΔR_j/K_ν multiplied by ρ^λ_j = T^λ_j/T^ν_j. Its exact log10 |ρ^λ_j| at the physical ω,
from log-gamma (`--sym`), is:

| family | j = 600, ν = 3 | ν = 11 | ν = 21 | j = 1600, ν = 3 | ν = 11 | ν = 21 |
|---|---|---|---|---|---|---|
| ν − 2 | −4.96 | −5.07 | −4.98 | −4.97 | −5.13 | −5.11 |
| ν + 2 | 5.06 | 5.05 | 4.96 | 5.09 | 5.13 | 5.11 |
| +1, −1 | −4.96 | −25.2 | −50.3 | −4.97 | −25.4 | −51.1 |

The weights go as follows:

- The family ν + 2 carries the raw weight of one pair, about e^{πω} ≈ 10⁵. It is the untuned label ν + 2 acting on ν, so
  in the tuned problem, where Ŝ_{ν+2} = 0 removes that label's own family, it would not appear in this form.
- The family ν carries |ρ| = 1, with a coefficient of order ν.
- The family ν − 2 carries about 10⁻⁵.
- The families ±1 carry about 10^{−2.5(ν−1)}, which outweighs their ν² by far.

**Where the remainder overtakes the cancelled term.** For j < j* = |c₂|/|piece| the j⁻² term is larger than either of
the two 1/j terms it is left over from, so the large-j form is not yet reliable there. Measured against the Darboux
sums' j, J = 600 in `exact_sigma.py` and J = 1600 in D_entire.md §6, `--sym` gives the first odd ν at which j* exceeds
J:

| family, perturbation | ν, p₀ | ν − 2, p₋ | ν + 2, p₊ | +1, p₋ | +1, p₀ | −1, p₀ | −1, p₊ |
|---|---|---|---|---|---|---|---|
| j* > 600 from ν = | 315 | 319 | 311 | 21 | 19 | 19 | 17 |
| j* > 1600 from ν = | 839 | 841 | 835 | 33 | 31 | 31 | 29 |

At ν = 21, j* is 33 to 47 in the families ν and ν ± 2 and 660 to 967 in the families ±1. So in the range route 3 cares
about (ν up to a few tens), only the families ±1 have a remainder larger than the leading term at the sums' j. Their
weight |ρ^{±1}| ≤ 10^{−20} from ν = 9 on makes that harmless at the physical ω.

### 9.5 What this does not settle

1. **The full remainder at small j.** c₂ is exact and c₃ is measured. The 1/j expansion itself needs j ≫ ν in the
   families ν and ν ± 2 and j ≫ ν² in the families ±1, so the range j ≲ j* is not covered by these coefficients. That
   is where Lemma W's term-by-term majorant would be needed (§5 item 3).
2. **Tuned inputs** (§5 item 1). The dominant family here, ν + 2, is an artefact of an untuned input label.
3. **The remainder bound itself.** These are coefficients and fits, numerical evidence for the growth, not a bound.

**Next step for route 3.** The remainder bound in the form these numbers suggest is C ν^p / j² for j ≥ j₀(ν) ∝ ν, with
p = 1 in the families ν and ν ± 2. The families ±1 can be bounded separately through their weight |ρ^{±1}|, which
decays like 10^{−2.5ν}. The log-convexity bound of D_entire.md §6.3 written with ν symbolic would be its first form. A
proof of (★) at A* still needs each increment's whole series (B_tail_M7.md §8.3).

### 9.6 Reproduce

From `solver/proof/` with venv-314:

```
python3 route3_growth.py --sym                              # 9.1, 9.3, 9.4 (~3 s)
python3 route3_growth.py --planted                          # PASS; --mutate=psi: FAIL, exit 1
python3 route3_growth.py --all                              # 9.2: planted, then nu = 3..21 (~5 min); PASS
python3 route3_growth.py --fit 3 --mutate=demix             # FAIL, exit 1
python3 route3_growth.py --fit 21 --pert=0 --mutate=psi     # FAIL, exit 1
python3 route3_growth.py --fit 21 --wf=0.05 --others        # the families +-1 at small w; PASS
```

## 10. A remainder bound C ν/j² for j ≥ 2ν (#990)

Script `solver/proof/route3_bound.py`, on `route3_cancel.py`'s `tail()` and `dr_builder()`. The object is §1's
one-step model with one perturbation of label μ ∈ {ν − 2, ν, ν + 2} and amplitude p on the A sin θ background. Its
increment splits exactly as

```math
\frac{\Delta R_j}{K_\nu} = \sum_\lambda C_\lambda(j)\,\rho^\lambda_j + r_j ,
```

with C_λ the exact near-end, far-end, de-mixing and ψ terms of `tail()`, where the cancellations live, and r_j the
interior convolution terms: Ss and Sd at 3 ≤ i ≤ j, and g_{j+1}'s inner terms 1 ≤ l ≤ j − 1.

**Result.** For odd ν ≥ 3 and j ≥ max(2ν, 10) (*certified*):

```math
\Big|\frac{\Delta R_j}{K_\nu}\Big| \le C_\mu\,|p|\,\frac{\nu}{j^2},\qquad C_{\nu-2} = 28.45,\quad C_\nu = 56.87,\quad C_{\nu+2} = 4.77\times10^6 .
```

- **How it is certified.** The hand lemmas B1–B4 are *proved* (§10.1). The sups are evaluated in ball arithmetic
  (`route3_bound.py`, python-flint 0.9.0, FLINT 3.6.0, the exact C_λ from sympy 1.14.0).
- **Consequence: the Darboux tail beyond j₀ is uniform in ν.** Σ_{j≥j₀} |ΔR_j/K_ν| ≤ C_μ|p| ν/(j₀ − 1) ≤ C_μ|p|/(2 −
  1/3), since j₀ ≥ 2ν and ν ≥ 3.
- **The families ±1 are bounded through their weight.** Their coefficients grow like ν², but sup ν|ρ^{±1}_{j₀}| ≤
  4.7×10⁻⁴.
- **Checked against the exact recursion** at ν = 3 to 21 and j < 1600, with no violation. The worst actual/bound ratios
  are 1.3×10⁻⁵, 1.8×10⁻² and 2.7×10⁻². A planted violation is caught in all 30 runs.
- **Tuned inputs keep the form,** *conditional* on two named hypotheses, an envelope (H1) and the cancellation (H2). On
  `amp_flint`'s real tuned sequences at the first tuned orders of harmonics 3, 5 and 7, the increments stay O(j⁻²)
  (numerical evidence, §10.4). The de-mixing control fails there.
- **The large-parameter literature was not needed.** The bound uses only B1's elementary inequalities and two Γ
  identities (DLMF §5.4), so Paris, Temme and Olde Daalhuis were not read for it.
- **Not covered:** the head j < 2ν of each sum, which (★)'s lowest-order form still needs (§10.5).

### 10.1 The hand lemmas

All for real ω and real labels. a^λ_m = (m + α_λ)(m + β_λ)/(m + 1) is the step T^λ_{m+1}/T^λ_m, α_λ = (1 + iλω)/2,
β_λ = (2 + iλω)/2, t_n = |T^1_n| and r_m = |a^1_m|.

- **B1 (steps).** |a^λ_m| ≥ m + ½ for every m ≥ 0, and |a^λ_m| ≤ |a^ν_m| whenever |λ| ≤ |ν|. *Proof.*
  |m + α_λ|² = (m + ½)² + λ²ω²/4 and |m + β_λ|² = (m + 1)² + λ²ω²/4 both increase with |λ|, and
  |a^λ_m| ≥ (m + ½)(m + 1)/(m + 1). ∎ (*proved*)
- **B2 (weights).** ρ^λ_l = T^λ_l/T^ν_l has ρ_0 = 1 and |ρ_{l+1}/ρ_l| = |a^λ_l|/|a^ν_l|.
  - (i) For |λ| < |ν|, |ρ^λ_l| ≤ 1 and it decreases in l, by B1.
  - (ii) For λ = ν + 2 with ν > 0 it increases, to P_{ν+2} = |Γ(α_ν)Γ(β_ν)/(Γ(α_{ν+2})Γ(β_{ν+2}))|. That limit follows
    from ∏_{m<L}(m + α′)/(m + α) = Γ(L + α′)Γ(α)/(Γ(L + α)Γ(α′)) and |Γ(L + α′)/Γ(L + α)| → |L^{iω}| = 1, since α′ − α =
    iω. With α_ν = ½ + iνω/2, β_ν = 1 + iνω/2, |Γ(½ + iy)|² = π/cosh πy and |Γ(1 + iy)|² = πy/sinh πy (DLMF §5.4), one
    has
    |Γ(α_ν)Γ(β_ν)|² = π²νω/sinh(πνω). So P_{ν+2}² = ν sinh(π(ν + 2)ω)/((ν + 2) sinh(πνω)) ≤ e^{2πω}/(1 − e^{−2πνω}), by
    sinh(A + B)/sinh A ≤ e^B/(1 − e^{−2A}). ∎ (*proved*; `--lemmas` checks both identities and the closed form against
    mpmath's Γ ratio.)
- **B3 (reduction).** In every interior term exactly one of the three factors carries the perturbation's label μ, and
  the other two carry the background's ±1. Each such term satisfies |T^p_a T^q_b T^r_c|/|T^ν_N| ≤ P_μ t_a t_b t_c/t_N
  with a + b + c < N, where P_μ = 1 for μ ≤ ν and P_μ = P_{ν+2} for μ = ν + 2. *Proof.* |T^{−1}_n| = |T^1_n| = t_n,
  since T^{−1} is the conjugate of T^1 for real ω. For the perturbation's factor at index k, |T^μ_k| = |ρ^μ_k||T^ν_k| ≤
  P_μ|T^ν_k| by B2, and |T^ν_k|/|T^ν_N| = ∏_{m=k}^{N−1} 1/|a^ν_m| ≤ ∏ 1/r_m = t_k/t_N by B1. ∎ (*proved*)
- **B4 (prefactors).** For j ≥ κν and x = 1/j, |c^ν_j| = |2j + 2 + iνω|/(4(j + 1)) ≤ (2 + 2x + ω/κ)/(4(1 + x)). For
  |λ| ≤ ν + 2, |2l + 1 + iλω| ≤ 2l + 1 + (ν + 2)ω. ∎ (*proved*)
- **B5 (the interior).** B3 turns every interior sum into the ν = 1 sums of D_entire.md §6.3, with B4's prefactors in
  place of ν = 1's. That section's log-convexity grouping applies unchanged: r_m increases for m ≥ 1, so Σ_l t_l t_{n−l}
  ≤ (2 + t₁) t_n and t_{j−k}/t_{j+1} ≤ ∏_{q≤k}(j − q + ½)^{−1}. The grouping covers i = 3, 4 ≤ i ≤ j − 3, j − 2, j − 1
  and j in Ss, the corresponding Sd terms, and g_{j+1}'s inner terms. `Cr_sup` assembles it in arb over x ∈ [0, 1/J₀]
  and checks r_m's monotonicity as D_entire.md does. Every interior term is O(j⁻³), except Ss at i = j, whose factor (1
  + (ν + 2)ω) makes it O(ν/j³). So the bound is written as j²|r_j|/(νP_μ|p|) ≤ Cr. (*certified*)
- **B6 (the families ±1).** By B2(i), |ρ^{±1}_j| ≤ |ρ^{±1}_{j₀}| for j ≥ j₀. W = sup over odd ν ≥ 3 of ν|ρ^{±1}_{j₀(ν)}|
  combines two pieces. For ν ≤ 99 it is an arb product. For ν ≥ 101 it is the hand bound ν q^M, with M = ⌊νω/4⌋ ≤ j₀ and
  q = ((νω/4 + 1)² + ω²/4)/(ν²ω²/4) < e^{−1}. That bound decreases in ν, and each step factor below M is at most q.
  (*certified* below ν = 101, *proved* above)

### 10.2 The certified constants

`python3 route3_bound.py --constants` (κ = 2, J₀ = 10, a 48 × 48 grid of (s, u) boxes, about 2.5 min) at
ω = 2π/1.64389375:

| perturbation μ | family λ | sup j²\|C_λ\|/ν^d, certified | lower bound (box centres) | d | weight |
|---|---|---|---|---|---|
| ν − 2 | ν − 2 | 0.5638 | 0.4719 | 1 | 1 |
| ν − 2 | +1 | 0.9229 | 0.9003 | 2 | W |
| ν | ν | 1.1028 | 0.9944 | 1 | 1 |
| ν | −1 | 1.0454 | 1.0071 | 2 | W |
| ν | +1 | 0.9530 | 0.9183 | 2 | W |
| ν + 2 | ν + 2 | 1.1775 | 1.0320 | 1 | P |
| ν + 2 | −1 | 2.6323 | 2.5528 | 2 | W |

The sups are over s = 1/ν ∈ [0, 1/3] and u = ν/j ∈ [0, 1/2], so they hold for every real ν ≥ 3 and j ≥ 2ν. The rational
function's denominator is evaluated as a product of its factors, each bounded away from zero, and its numerator in
Horner form. The box-centre values bound the sup from below, and they show the grid's overestimate is within 20%.

| constant | value |
|---|---|
| P = sup P_{ν+2} (B2) | ≤ 1.63995×10⁵ (e^{πω} = 1.63995×10⁵) |
| W = sup ν\|ρ^{±1}_{j₀}\| (B6) | ≤ 4.728×10⁻⁴ (hand tail at ν = 101: 1.2×10⁻⁵⁵) |
| Cr, interior (B5), μ = ν ∓ 2 (3 triples) | ≤ 27.885 |
| Cr, interior, μ = ν (6 triples) | ≤ 55.770 |
| **C_{ν−2}** | **28.449** |
| **C_ν** | **56.874** |
| **C_{ν+2}** | **4.7661×10⁶** |

C_μ = Σ_λ S_λ w_λ + Cr P_μ. The interior bound dominates. It comes from the crude grouping at J₀ = 10 and shrinks like
1/J₀, while the near-end and far-end part is of order one. With J₀ = 100 (`--constants --J0=100`) the same bound holds
for j ≥ max(2ν, 100) with C_{ν−2} = 1.599, C_ν = 3.173 and C_{ν+2} = 3.629×10⁵, the interior parts falling to 1.035 and
2.070 and W to 4.2×10⁻⁵. The check of §10.3 was run against the J₀ = 10 constants.

### 10.3 The check against the exact recursion

`python3 route3_bound.py --check` (512-bit balls, about 7.5 min) evaluates ΔR_j/K_ν by the exact recursion at ν = 3, 5,
…, 21, for every j from j₀ to 1599 and each perturbation, against the bound. There is no violation. The largest ratio
actual/bound is 1.3×10⁻⁵ for μ = ν − 2, 1.8×10⁻² for μ = ν and 2.7×10⁻² for μ = ν + 2. The largest relative ball radius
is 6×10⁻³², at ν = 21. A guard stops the run if any value's relative radius exceeds 10⁻⁶. An earlier version built ω
at 128 bits before raising the precision, and that propagated radius made the recursion NaN, which the guard now
catches.

**Planted violation.** `--check --mutate=plant` adds three times the bound at five j per ν and perturbation. It is
caught in all 30 runs, five violations each, and the script exits 1.

**ψ control.** `--check --mutate=psi` scales h₂ by 3/2 in the recursion, which leaves an O(1/j) term in the
same-label family. The bound then fails for the perturbation of label ν at ν = 3, 5 and 7, first at j = 680, 1135 and
1590, and the script exits 1. The crossing point grows like ν, because the O(1/j) term overtakes C_ν ν/j² at j ∝ ν. So
from ν = 9 on it lies beyond j = 1600. The perturbations ν ± 2 carry no ψ term and are unaffected.

### 10.4 What tuned inputs change

For ν ≥ 5 the inputs at harmonic ν's first tuned order are tuned sequences (§5 item 1). The bound uses two properties
of its inputs, and tuned inputs keep them only under hypotheses.

- **(H1) An envelope:** |X_l[μ]| ≤ X̄_μ |T^μ_l| for every l. A tuned input is X_l = −T^μ_l Σ_{i≥l} b_i/T^μ_{i+1} plus
  de-mixing terms, since its multiplier is tuned to zero. So X̄_μ is the supremum of that tail, finite when the Darboux
  sums converge. Given (H1), B3 holds with |p| replaced by X̄_μ, and the interior bound survives with that factor.
- **(H2) The cancellation:** the target's increment has no O(1/j) part. In the one-step model this is §2's pairing,
  which assumes each input's late orders follow its own label's family. A tuned input's own family is removed, so its
  late orders follow deeper families κ ≠ λ, and the one-step pairing alone would leave a term ∝ (κ − λ)/j (§5 item 1).
  (H2) says that the full de-mixing by e^{−Π}, at every order, supplies the partner, so that the full expansion's
  normalised multiplier sequence has no O(1/j) increments.

Under (H1) and (H2) the form C ν/j² survives, with C multiplied by X̄_μ and recomputed for the families the tuned
inputs carry (*conditional* on (H1) and (H2)).

**The test** (`python3 route3_tuned.py 7 800`, about 5 min). It takes `amp_flint.py`'s real expansion to order 7, with
its tuning, and the exact normalised multiplier sequence R_j of `S_at` for every j ≥ 2, the sequence that `S_at` fits.
It reports j²|ΔR_j/K_ν|/ν. Fixed before the run: the form holds if the window maxima over [J/8, J/4], [J/4, J/2] and
[J/2, J] grow by at most 1.25 per window. An O(1/j) part would give about 2.

| ν | order | first tuned order | M1 | M2 | M3 | M2/M1 | M3/M2 | form |
|---|---|---|---|---|---|---|---|---|
| 3 | 3 | yes | 2.543e-6 | 2.186e-6 | 2.025e-6 | 0.86 | 0.93 | holds |
| 5 | 5 | yes | 2.155e-12 | 1.438e-12 | 1.17e-12 | 0.67 | 0.81 | holds |
| 7 | 7 | yes | 7.381e-19 | 3.335e-19 | 2.258e-19 | 0.45 | 0.68 | holds |
| 1 | 3 | no | 2.556 | 2.607 | 2.622 | 1.02 | 1.01 | holds |
| 1 | 5 | no | 2.346 | 2.376 | 2.377 | 1.01 | 1.00 | holds |
| 3 | 5 | no | 2.026e-6 | 1.823e-6 | 1.801e-6 | 0.90 | 0.99 | holds |
| 5 | 7 | no | 1.864e-12 | 1.267e-12 | 1.051e-12 | 0.68 | 0.83 | holds |

So (H2) holds at the first tuned orders of harmonics 3, 5 and 7 and at the four later orders tried (numerical evidence).
The tuned inputs keep the increments O(j⁻²) times a bounded factor. The small M values for ν = 5 and 7 are the tuned
amplitudes. The windows do not resolve a log j factor: one of the size D_entire.md §6.6 reports for harmonic 1 at
order 5 would give about 1.1 per window, inside the bar. **Control:** `--mutate=demix` (E × 3/2 in every target's
de-mixing) fails every target, with ratios 1.29 to 2.01, and exits 1.

### 10.5 What is not covered

1. **The head j < j₀ = 2ν.** That is about 2ν terms of each Darboux sum, where the 1/j expansion does not apply. The
   bound says nothing there, and it is what (★)'s lowest-order form still needs. §11 certifies it.
2. **The untuned label ν + 2.** Its weight P ≈ e^{πω} ≈ 1.6×10⁵ is in C_{ν+2}. In the tuned problem that label's own
   family is removed, so the constant there is not this one (§10.4).
3. **Orders beyond one A² step**, and the whole series in the amplitude at A* (B_tail_M7.md §8.3).
4. **The physical ω only.** The constants are certified at ω = 2π/1.64389375 as a ball, not over an ω interval.

### 10.6 Reproduce

From `solver/proof/` with venv-314:

```
python3 route3_bound.py --lemmas                  # B2's Gamma identities and closed form, B2(i)'s monotonicity; PASS
python3 route3_bound.py --constants               # 10.2 (~2.5 min); --J0=100 for the tighter constants
python3 route3_bound.py --check                   # 10.3 (~7.5 min, 512 bits); PASS, exit 0
python3 route3_bound.py --check --mutate=plant    # planted violation: FAIL, exit 1
python3 route3_bound.py --check --mutate=psi      # h2 x 3/2: fails at nu = 3, 5, 7, exit 1
python3 route3_tuned.py 7 800                     # 10.4 (~5 min): the form holds on amp_flint's tuned sequences; PASS
python3 route3_tuned.py 7 800 --mutate=demix      # control: FAIL, exit 1
```

## 11. The head j < 2ν (#1015)

Script `solver/proof/route3_head.py`, on `route3_cancel.py`'s exact recursion (`dr_builder`) and §10's lemmas. The
object is §10's one-step model, with one perturbation of label μ ∈ {ν − 2, ν, ν + 2} and amplitude p. The normalised
multiplier is R₀ + Σ_{j≥0} ΔR_j/K_ν. §10 bounds the sum over j ≥ j₀ = max(2ν, 10), and this section bounds the head,
0 ≤ j < j₀. Written 2026-11-19 for issue #1015.

**Result.** Let H_μ(ν) = Σ_{j<j₀}|ΔR_j/K_ν|/|p|. For every odd ν ≥ 3 (*certified*)

```math
H_\mu(\nu) \le \Lambda^{\rm head}_\mu\,e^{0.736\nu},\qquad \Lambda^{\rm head}_{\nu-2} = 0.011012,\quad \Lambda^{\rm head}_\nu = 0.058818,\quad \Lambda^{\rm head}_{\nu+2} = 3.3073 .
```

- **So the head closes uniformly in ν.** The sup is reached at ν = 3. For ν ≤ 99 H comes from the exact recursion in
  ball arithmetic. Beyond, a closed form from the hand lemma B7 gives e^{−0.736ν}H ≤ 1.5×10⁻²⁶, 3.1×10⁻²⁶ and 2.5×10⁻²¹
  (§11.3).
- **The whole Darboux sum, with §10's tail** (*certified*): sup_ν e^{−0.736ν} Σ_{j≥0}|ΔR_j/K_ν|/|p| ≤ 1.0534, 2.1427 and
  1.7463×10⁵. The tail's C_μ ν/(j₀ − 1) dominates these, against the head's 0.011012, 0.058818 and 3.3073. The last
  constant carries the untuned weight P ≈ 1.6×10⁵ (§10.5 item 2).
- **The head does not grow with ν at all** (numerical evidence beyond the certificate, §11.1). H falls from 0.100 to
  0.034 for μ = ν − 2, from 0.535 to 0.406 for μ = ν, and levels off near 15 for μ = ν + 2 over ν = 3 to 99. The tail's
  form C_μ ν/j² also holds throughout the head, at most 8.1×10⁻³ of the bound.
- **Checked against the exact recursion** at ν = 3 to 21, at every head term, with no violation (§11.4). The worst
  ratios actual/bound are 0.024, 0.024 and 8.4×10⁻⁶. A planted violation is caught in all 30 runs.
- No cancellation is needed. The head is a finite sum, so the triangle inequality on every convolution term suffices.
  The cancellations of §2 matter only for the tail.

### 11.1 Scoping

`python3 route3_head.py --scope` (256 bits, about 3 min). It computes the exact recursion over the head for odd ν = 3 to
21, one perturbation at a time, per unit |p|. "Termwise" bounds every convolution term separately, using either exact
|T| or §10's ν-free reduction B3 (with B2's weights):

| ν | μ | j₀ | max_{1≤j<j₀} j²\|ΔR_j/K\|/ν (at j) | / C_μ | H | signed head | termwise, exact T | termwise, B3 |
|---|---|---|---|---|---|---|---|---|
| 3 | ν − 2 | 10 | 0.003511 (1) | 1.2×10⁻⁴ | 0.1002 | 0.09806 | 0.1576 | 6.189 |
| 3 | ν | 10 | 0.3161 (9) | 5.6×10⁻³ | 0.5351 | 0.4065 | 1.492 | 12.52 |
| 3 | ν + 2 | 10 | 257.2 (9) | 5.4×10⁻⁵ | 30.09 | 18.7 | 186.8 | 8.143×10⁵ |
| 9 | ν − 2 | 18 | 0.003075 (4) | 1.1×10⁻⁴ | 0.04914 | 0.04873 | 0.163 | 15.38 |
| 9 | ν | 18 | 0.4121 (17) | 7.3×10⁻³ | 0.4209 | 0.3687 | 0.7201 | 32.88 |
| 9 | ν + 2 | 18 | 61.15 (17) | 1.3×10⁻⁵ | 9.881 | 7.209 | 89.15 | 2.81×10⁶ |
| 15 | ν − 2 | 30 | 0.003531 (7) | 1.2×10⁻⁴ | 0.04205 | 0.04174 | 0.2545 | 33.66 |
| 15 | ν | 30 | 0.4466 (29) | 7.9×10⁻³ | 0.414 | 0.3661 | 0.6864 | 70.62 |
| 15 | ν + 2 | 30 | 79.69 (29) | 1.7×10⁻⁵ | 11.79 | 8.424 | 166.6 | 6.152×10⁶ |
| 21 | ν − 2 | 42 | 0.003746 (10) | 1.3×10⁻⁴ | 0.03922 | 0.03895 | 0.3522 | 59.94 |
| 21 | ν | 42 | 0.4616 (41) | 8.1×10⁻³ | 0.4112 | 0.3649 | 0.673 | 124.3 |
| 21 | ν + 2 | 42 | 89.5 (41) | 1.9×10⁻⁵ | 12.76 | 9.031 | 245.8 | 1.08×10⁷ |

Fitted over ν = 9 to 21, H grows like ν^−0.26, ν^−0.03 and ν^0.30. The exact termwise bound grows like ν^0.91, ν^−0.08
and ν^1.20, and the B3 termwise bound like ν^1.61, ν^1.57 and ν^1.59. Every finite-sum bound grows polynomially, so
e^{−0.736ν} absorbs it. That was the scoping's question, and it decided to certify. The largest term is at j = 0 for μ =
ν − 2 and ν, where the near-end, de-mixing and ψ terms all have their full size. For μ = ν + 2 it is at j = j₀ − 1,
where the weight ρ^{ν+2}_j has grown toward P.

### 11.2 The lemma B7, and its closed form

**B7 (the head, term by term).** Let t₁ = r₀, A₀ = (2 + t₁)²/8 and c_j = |2j + 2 + iνω|/(4(j + 1)). N_μ = 3, 6, 3 is the
number of (lapse pair, y label) triples with one perturbation factor, and n_μ = 1, 2, 1 of them have the perturbation
in the y slot. Then for every j ≥ 0 and odd ν ≥ 3

```math
\frac{|\Delta R_j/K_\nu|}{|p|} \le P_\mu A_0\Big[c_j\,\frac{n_\mu\Sigma_\mu(j) + (N_\mu - n_\mu)\Sigma_1(j)}{r_{j-1}r_j} + \frac{N_\mu H_{j+1}}{r_j}\Big] + D_j ,\qquad \Sigma_\lambda(j) = \sum_{i=1}^{j}\frac{|2(j-i)+1+i\lambda\omega|}{i} ,
```

with the first term absent for j = 0, H_n the harmonic numbers, and P_μ as in B3. D_j is the de-mixing and ψ terms,
Σ_r 2|E_r| w_r + h₂(1/|α_ν + j| + 1/|β_ν + j|), with |ρ^r| ≤ w_r from B2 (w = 1 for |λ_r| ≤ ν and P_{ν+2} for
λ_r = ν + 2).

*Proof.* In Darboux form (D_entire.md §6.1), ΔR_j/K_ν = −b_j/T^ν_{j+1} + Σ_r E_r d₀^r (ρ^r_{j+1} − ρ^r_j) plus the ψ
term for λ_r = ν. Here b_j = c_j Ss_j + Sd_{j+1}, which is `route3_cancel.order3`'s d̃_{j+1} = a^ν_j d̃_j − b_j. Ss_j is
the sum over triples and 1 ≤ i ≤ j of g_i s^r_{j−i}, Sd_{j+1} the same over 1 ≤ i ≤ j + 1 of g_i d^r_{j+1−i}, and g_i =
(8i)⁻¹ Σ_{l<i} d^p_l d^q_{i−1−l}.

1. Every input has |d₀| = 1 per unit amplitude. So by B3 each product of three inputs, divided by |T^ν_{j+1}|, is at
   most P_μ t_l t_{i−1−l} t_n/t_{j+1}.
2. Σ_l t_l t_{i−1−l} ≤ (2 + t₁)t_{i−1} (D_entire.md §6.3). Then t_{i−1}t_n ≤ (2 + t₁)t_{i−1+n}, which is one term of
   the same sum.
3. For Ss, i − 1 + n = j − 1, and t_{j−1}/t_{j+1} = 1/(r_{j−1}r_j). For Sd it is j, and t_j/t_{j+1} = 1/r_j.
4. |s^r_n| = |2n + 1 + iλ_rω||d^r_n|, and Σ_{i≤j+1} 1/(8i) = H_{j+1}/8.
5. The de-mixing and ψ terms are bounded by B2 and |α_ν + j| ≥ νω/2. ∎ (*proved*)

**The closed form.** The bound sums over the head to at most

```math
G_\mu(\nu) = P_\mu A_0 N_\mu\,\ell\,\Big[T_1 + (\nu+2)\omega + \frac{\nu\omega}{2}S_a + \frac{\nu(\nu+2)\omega^2}{4}S_b + \frac1{r_0} + \ln(2j_0-1)\Big] + j_0\bar D ,
```

with ℓ = 1 + ln j₀, T₁ = 2 + ln(2j₀ − 1), S_a = Σ_{j≥1} 1/((j + 1)(j − ½)) = (2/3)(1 + 2 ln 2) and
S_b = Σ_{j≥1} 1/((j + 1)(j² − ¼)) = (8/3)(1 − ln 2). D̄ is D_j with 1/|α_ν + j| ≤ 2/(νω), and P_μ ≤ P_sup for μ = ν + 2.
It follows from these inequalities:

- c_j ≤ ½ + νω/(4(j + 1)), and |2(j − i) + 1 + iλω| ≤ 2(j − i) + 1 + (ν + 2)ω;
- H_j ≤ 1 + ln j₀ on the head, r_{j−1}r_j ≥ j² − ¼ and r_j ≥ j + ½ (B1);
- Σ_{j=1}^{N} 1/(j − ½) ≤ 2 + ln(2N − 1), Σ_{j=1}^{N} 1/(j + ½) ≤ ln(2N + 1) (integral comparison) and
  Σ_{j≥1} 1/(j² − ¼) = 2 (telescoping);
- S_a and S_b by partial fractions, with ψ(2) − ψ(½) = 1 + 2 ln 2 and ψ(3/2) = 2 − γ − 2 ln 2 (DLMF §5.4).

Every piece of G is nonnegative and of the form ν^a L₁L₂, with a ≤ 2 and L₁, L₂ logarithms of ν, or a factor that falls
with ν. So d ln(e^{−0.736ν} G)/dν ≤ −0.736 + 4/ν < 0 for ν ≥ 7, and sup_{ν>ν₁} e^{−0.736ν} G = e^{−0.736(ν₁+2)} G(ν₁ +
2). (*proved*; `--lemmas` checks S_a, S_b, the telescoping sum and the two inequalities numerically, and samples the
decrease.)

### 11.3 The certified constants

`python3 route3_head.py --certify` (about 10 min, one process at nice 10). For odd ν = 3 to 99, H_μ(ν) is the sum of the
upper ends of the exact recursion's balls at 512 bits. The relative-radius guard of §10.3 never asked for more. Beyond
ν = 99, the closed form at ν = 101 is used:

| ν | H, μ = ν − 2 | H, μ = ν | H, μ = ν + 2 |
|---|---|---|---|
| 3 | 0.10019 | 0.53510 | 30.088 |
| 5 | 0.065829 | 0.43644 | 7.1812 |
| 11 | 0.045827 | 0.41776 | 10.690 |
| 21 | 0.039215 | 0.41116 | 12.761 |
| 41 | 0.035921 | 0.40770 | 14.103 |
| 99 | 0.033976 | 0.40557 | 15.025 |

| perturbation | sup e^{−0.736ν} H, ν ≤ 99 (at ν) | e^{−0.736ν} G at ν = 101 | Λ^head | sup e^{−0.736ν}(H + C_μ ν/(j₀ − 1)) (at ν) |
|---|---|---|---|---|
| ν − 2 | 0.011012 (3) | 1.53×10⁻²⁶ | **0.011012** | **1.0534** (3) |
| ν | 0.058818 (3) | 3.07×10⁻²⁶ | **0.058818** | **2.1427** (3) |
| ν + 2 | 3.3073 (3) | 2.52×10⁻²¹ | **3.3073** | **1.7463×10⁵** (3) |

The last column uses §10's Σ_{j≥j₀} |ΔR_j/K_ν| ≤ C_μ|p| ν/(j₀ − 1), with the J₀ = 10 constants. Beyond ν = 99 it uses
the same closed form, and it is at most 1.5×10⁻²⁶, 3.1×10⁻²⁶ and 2.5×10⁻²¹ there.

### 11.4 The checks

- **B7 against the exact recursion** (`--check`, 256 bits, about 2 min). At every head term of odd ν = 3 to 21 and each
  perturbation there is no violation. The worst ratios actual/bound are 0.02352 (μ = ν − 2), 0.02388 (ν) and 8.35×10⁻⁶
  (ν + 2).
- **The closed form above the per-term sum.** Σ_{j<j₀} B₇ ≤ G at every odd ν = 3 to 201. The worst ratios are 0.028,
  0.032 and 0.037, so G is loose. That is harmless where it is used, beyond ν = 99.
- **Planted violation.** `--check --mutate=plant` adds three times the bound at five head j per run. It is caught in all
  30 runs, five violations each, exit 1.
- **Control of the closed form.** `--check --mutate=gsmall` divides G by 100, and the second check then fails at 291
  points (ratios 2.79 to 3.70), exit 1.

### 11.5 What is not covered

The same limits as §10.5 items 2 to 4 apply:

- The untuned label ν + 2, whose P is in the third constant.
- Orders beyond one A² step.
- The physical ω only.

Tuned inputs (§10.4) enter the head as they enter the tail. Under (H1) the inputs keep an envelope, and B7 holds with
|p| replaced by X̄_μ. The head needs no cancellation, so (H2) is not used here.

### 11.6 Reproduce

From `solver/proof/` with venv-314 (python-flint 0.9.0, FLINT 3.6.0). The logs are in folder I1015-mac of the raw-data
repository.

```
python3 route3_head.py --scope                     # 11.1 (~3 min)
python3 route3_head.py --lemmas                    # 11.2's closed forms and inequalities; PASS
python3 route3_head.py --check                     # 11.4 (~2 min); PASS, exit 0
python3 route3_head.py --check --mutate=plant      # planted violation: FAIL, exit 1
python3 route3_head.py --check --mutate=gsmall     # G / 100: FAIL, exit 1
python3 route3_head.py --certify                   # 11.3 (~10 min): Lambda_head and the whole sum's constants
```

# P6 route 3: an independent review of the tail bound (#990) and the head (#1015)

Issue #1062. The object under review is [P6_route3.md](P6_route3.md) §10 and §11 and the scripts
`solver/proof/route3_bound.py`, `route3_head.py`, `route3_tuned.py`, `route3_cancel.py` and `route3_growth.py`, at main
5f169359. Written from 2026-11-21 02:08 UTC by a Claude (Anthropic) reviewer that did not write that work, under Nathan
Moinvaziri's direction. Status words as in AGENTS.md.

Order of work, as the issue sets it:

1. Re-derive each hand lemma (B1–B4) and the head's argument (B7 and its closed form) from its statement only (§3).
   This was written and committed before the written proofs were read.
2. Check the written proofs against those re-derivations (§4).
3. Re-implement the certified bounds from the statements, in other arithmetic (mpmath, with mpmath.iv intervals for
   every certified number), before reading the five scripts in detail (§5).
4. Compare the constants, name the shared machinery, and say what rests on (H1) and (H2) (§6–§8).

## Verdict

- **The tail bound of #990 holds.** An independent certificate (mpmath.iv, a direct termwise interior sum, an adaptive
  sup) gives |ΔR_j/K_ν| ≤ C′_μ |p| ν/j² for odd ν ≥ 3 and j ≥ max(2ν, 10), with C′_μ = 4.874, 9.800 and 8.928×10⁵.
  Those are 5.8, 5.8 and 5.3 times below the certificate's 28.449, 56.874 and 4.7661×10⁶, so the certificate's claim
  is implied. With exact prefactors the constants are 2.389, 5.396 and 5.782×10⁵ (§5.5).
- **The head of #1015 holds, with one quoted constant too small.** The review's enclosures of H_μ(ν) agree with §11.3
  at every quoted digit. But Λ^head_{ν−2} = 0.011012 lies below the sup it bounds, 0.01101243 at ν = 3, so it must
  read 0.011013. The cause is rounding to nearest where outward rounding is needed (finding 1). Λ^head_ν = 0.058818 and
  Λ^head_{ν+2} = 3.3073 are valid.
- **Every hand lemma holds as stated.** B1–B4, B6, B7, the closed form G and its monotonicity were re-derived from
  their statements before the proofs were read, and no step of the written proofs fails (§3, §4). The convolution
  bound that B5 and B7 cite is proved here in one line.
- **The model is a true sub-problem of amp_flint.** It equals amp_flint's own late-order sequence, linear in a Kummer
  input, to 2.5×10⁻³⁷ (§5.1).
- **The extension to tuned inputs is not proved.** It does not follow from (H1) and (H2) as stated (finding 2, §7).
  So what is certified is the one-step model with Kummer inputs, and not yet the first tuned order that route 3 is
  for.
- Ten findings: one wrong (finding 1, a rounding of 4×10⁻⁵ that changes no conclusion), one unproved, two unclear, six
  minor (§9). Every planted error makes the re-implementation fail, except M7, which did not fire for a reason that
  was found and recorded. M7′ replaces it (§6).

## 1. The claims, and the theorem they serve

**The theorem served.** (★) of [proof/B_tail_M7.md](../proof/B_tail_M7.md) §8.4 is a *conjecture*: the coupling of the
pair ν grows at most like Λ e^{σν}, with Λ independent of ν and σ ≤ 0.736. Route 3 works on its lowest-order, formal
form. That form is a statement about the normalised multiplier's Darboux sum, R₀ + Σ_{j≥0} ΔR_j/K_ν, at one order in
the amplitude. B_tail_M7.md §8.3 already says that the lowest order does not settle (★) at A*.

**The model.** P6_route3.md §1 and §10: amp_flint's order-3 recursion at target harmonic ν (odd, ≥ 3), with the
background inputs (+1, d₀ = −i) and (−1, d₀ = +i) and one perturbation of label μ ∈ {ν − 2, ν, ν + 2} and amplitude p,
kept to first order in p. Every input is its exact Kummer sequence d₀ T^λ_l.

**Claims under review.**

- **T (tail, #990).** For odd ν ≥ 3 and j ≥ j₀ = max(2ν, 10), |ΔR_j/K_ν| ≤ C_μ |p| ν/j², with C_{ν−2} = 28.45,
  C_ν = 56.87, C_{ν+2} = 4.77×10⁶ (*certified*), from hand lemmas B1–B4 (*proved*), B5 and B6.
- **H (head, #1015).** H_μ(ν) = Σ_{j<j₀} |ΔR_j/K_ν|/|p| ≤ Λ^head_μ e^{0.736ν} for every odd ν ≥ 3, with
  Λ^head = 0.011012, 0.058818, 3.3073 (*certified*): exact recursion in balls for ν ≤ 99, the hand lemma B7 and a closed
  form G above.
- **W (whole sum).** sup_ν e^{−0.736ν} Σ_{j≥0} |ΔR_j/K_ν|/|p| ≤ 1.0534, 2.1427, 1.7463×10⁵.
- **Conditional extension (§10.4).** Under (H1) and (H2) the form C ν/j² survives for tuned inputs.

## 2. Protocol and pass criteria

Fixed before the runs. Every certified number of the re-implementation is an mpmath.iv interval (mpmath 1.3.0, a
different arithmetic from python-flint's arb). Every script exits non-zero when its check fails.

**The re-implementation, `solver/proof/review_p6/`.**

- `model.py`: amp_flint's order-3 recursion written out from `amp_flint.py`'s docstring, `pass_even`, `pass_odd` and
  `S_at` (not from route3_cancel.py), for one perturbation, linear in p. It gives ΔR_j/K_ν exactly (mpmath at 80
  digits, or mpmath.iv), and the same increment split term by term into the explicit part and the interior.
- `explicit.py`: my own sympy derivation of the explicit coefficients C_λ(j) (near end i = 1, 2 of both convolutions,
  far end i = j + 1 at l = 0 and l = j, de-mixing, ψ), the symbolic O(1/j) check, and a certified sup of j²|C_λ|/ν^d
  over s = 1/ν ∈ [0, 1/3], u = ν/j ∈ [0, 1/2] by adaptive bisection in mpmath.iv. This is a different route from a
  fixed grid: boxes are split until the upper bound is within 0.5% of the best point value.
- `interior.py`: the interior bound by direct summation of the termwise majorant (B3 with B4's prefactors, or with the
  exact label prefactors) for 10 ≤ j ≤ J₁ = 2000, and my own log-convexity bound above J₁. This is a different route
  from the grouping of D_entire.md §6.3. It also certifies P = sup P_{ν+2} and W = sup ν|ρ^{±1}_{j₀}|.
- `head.py`: H_μ(ν) for odd ν = 3 to 99 in mpmath.iv, B7 term by term against the exact increments, and G at ν = 101.
- `ampcheck.py`: the model against `amp_flint.py`'s own `pass_even`, `pass_odd` and `S_at`, run as a black box with the
  perturbation injected through `add_linear` at order 1.

**Pass criteria.**

| id | check | pass |
|---|---|---|
| PC1 | model.py against amp_flint.py's own passes (linear part by ±p differences, p = 2⁻⁸⁰) at ν = 3, 5, each μ, j ≤ 40 | relative difference ≤ 10⁻³⁰ |
| PC2 | explicit part + interior = exact increment, at ν ∈ {3, 5, 9, 21}, j ∈ {10, 25, 60, 200}, each μ | relative difference ≤ 10⁻³⁰ |
| PC3 | the O(1/j) coefficient of every C_λ, symbolic in ν and ω | exactly 0 |
| PC4 | each constant S_λ, Cr, P, W, C_μ: my certified upper bound U and my lower bound L (point values inside the region) | L ≤ the certificate's value V; otherwise finding "wrong" |
| PC5 | my B4-mode interior majorant against the certificate's Cr | sup ≤ Cr (the grouping must lie above the termwise sum it bounds); otherwise a discrepancy to explain |
| PC6 | my H_μ(ν), odd ν = 3 to 99, against §11.3's table | the quoted digits lie in my enclosure, rounded to them |
| PC7 | B7 against the exact increments at every head term, odd ν = 3 to 21 | no violation |
| PC8 | my e^{−0.736·101} G(101) and Λ^head against §11.3 | both equal to the quoted digits |
| PC9 | my constants C'_μ against the exact increments, odd ν = 3 to 21, j₀ ≤ j < 1000 | no violation |

**Planted errors.** Each must make the re-implementation fail, with exit 1:

| id | mutation | must fail |
|---|---|---|
| M1 | de-mixing E × 3/2 in the explicit part (a scaled coefficient) | PC3, and the certified sup diverges |
| M2 | drop the far-end l = j term from C_λ (a dropped family term) | PC2 |
| M3 | use 3ω for (ν + 2)ω in B4's label prefactor (a wrong ν dependence) | the pointwise prefactor check in interior.py |
| M4 | g₂ × 1.01 in the explicit near end (a scaled coefficient) | PC2 |
| M5 | drop the Ss term at i = j from the interior majorant (a dropped term) | the ν = 1 equality test in interior.py |
| M6 | drop one ordered lapse pair from model.py (a dropped family) | PC1 |
| M7 | A₀ / 100 in B7 | PC7 |

The ν = 1 equality test: at ν = 1 with the background alone, every |T| equals t, so the termwise majorant with exact
prefactors equals the sum of the absolute values of the interior terms. Any dropped or double-counted term breaks it.

**Deviations from the protocol, as they happened.**

- PC1 was set as an injection at order 1 with p = 2⁻⁸⁰. That cannot work. `S_at` de-mixes the order-k′ sequence only
  over labels |μ| ≤ k′ (its `range(-min(H, kp), ...)`), so an order-1 input of label μ ≥ 3 is never seen there. The
  run of record injects at the input's natural order |μ| and reads at |μ| + 2, which is how harmonics enter amp_flint.
  It uses ε = 2⁻⁶⁰, because the difference y(+ε) − y(−ε) costs about 60 of mpmath's 266 bits. The bar did not change.
- The first ampcheck run failed for a reason in my own harness: `Amp(jlo=0)` falls back to jlo = (2J)//5. It was
  fixed by setting `A.jlo = 0` after construction.
- The sup tolerance of explicit.py was tightened from 2% to 0.5% before the run of record.
- M7 (A₀/100 in B7) did not fire, because B7 with A₀/100 still bounds every head term. M7′, all of B7 divided by
  100, replaces it (§6).
- interior.py crashed at --J1 below 197, because W needs r_m to m = 197. That was fixed before the controls of record.
- interior.py's first check of P compared the finite product at L = 3000 with the closed form to 1%. It failed, because
  the product converges like 1 − O(νω²/L). It now checks the closed form against mpmath's Γ ratio, which is the
  identity itself, and that the product lies below it. The error was in my check, not in the certificate.

## 3. Re-derivations from the statements

Written before reading the proofs in P6_route3.md §10.1 and §11.2 and the scripts. B1's proof is printed inline with
its statement and was seen; the others were masked when reading.

**Notation.** ω > 0 real. α_λ = (1 + iλω)/2, β_λ = 1 + iλω/2, a^λ_m = (m + α_λ)(m + β_λ)/(m + 1) = T^λ_{m+1}/T^λ_m,
T^λ_0 = 1. t_n = |T^1_n| = |T^{−1}_n| (T^{−1} is the conjugate of T^1), r_m = |a^1_m|.

**The model, from amp_flint.py.** With d̃^{(1)}_l[λ] = d₀^λ T^λ_l and s̃^{(1)}_l[λ] = (2l + 1 + iλω) d̃^{(1)}_l[λ],
`pass_even(2)` gives the order-2 lapse g_m = (1/(8m)) Σ_{(λ₁,λ₂)} d₀^{λ₁}d₀^{λ₂} Σ_{l=0}^{m−1} T^{λ₁}_l T^{λ₂}_{m−1−l},
summed over ordered input pairs, with g₀ = 0. `pass_odd(3)` at label ν gives d̃_{j+1} = a^ν_j d̃_j − b_j with d̃₀ = 0,
b_j = c_j S^s_j + S^d_{j+1}, c_j = (2j + 2 + iνω)/(4(j + 1)), S^s_j = Σ_{i=1}^{j} g_i s̃_{j−i}, S^d_{j+1} =
Σ_{i=1}^{j+1} g_i d̃_{j+1−i}, summed over (lapse pair, outer input) triples whose labels add to ν. `S_at` at order 3
gives R_j = base_j [d̃_j + Σ_λ E[ν − λ] d̃^{(1)}_j[λ] + h₂ (ψ(α_ν + j) + ψ(β_ν + j)) d̃^{(1)}_j[ν]], with E[h] =
−F²_h/(ihω) (E[0] = 0), h₂ = F²₀/2 = 1/4 and base_j T^ν_j = K_ν. Hence the increment of §1 of P6_route3.md, which I
take as the definition. A term linear in p has exactly one perturbation factor: for μ = ν − 2 the other two labels
are (+1, +1), for μ = ν they are (+1, −1), for μ = ν + 2 they are (−1, −1). That gives 3, 6 and 3 ordered triples.

**B1.** |m + α_λ|² = (m + ½)² + λ²ω²/4 and |m + β_λ|² = (m + 1)² + λ²ω²/4. Both increase with λ², so |a^λ_m| does.
Dropping the λ²ω²/4 terms gives |a^λ_m| ≥ (m + ½)(m + 1)/(m + 1) = m + ½. Holds as stated.

**B2.** |ρ^λ_{l+1}/ρ^λ_l| = |a^λ_l|/|a^ν_l| is ≤ 1 for |λ| ≤ |ν| and ≥ 1 for |λ| ≥ |ν| (B1); strict when the labels'
squares differ. (i) follows. (ii): |ρ^{ν+2}_L| = |(α′)_L(β′)_L/((α)_L(β)_L)| with α′ = α + iω, β′ = β + iω. By
(α′)_L/(α)_L = Γ(L + α′)Γ(α)/(Γ(L + α)Γ(α′)) and |Γ(L + α + iω)/Γ(L + α)| → 1 (the ratio is L^{iω}(1 + O(1/L))),
the limit is |Γ(α_ν)Γ(β_ν)/(Γ(α_{ν+2})Γ(β_{ν+2}))|, and the increasing sequence stays below it. With
|Γ(½ + iy)|² = π/cosh πy, |Γ(1 + iy)|² = πy/sinh πy and y = νω/2, |Γ(α_ν)Γ(β_ν)|² = π²(νω/2)/(cosh(πνω/2)sinh(πνω/2))
= π²νω/sinh πνω. So P²_{ν+2} = ν sinh(π(ν + 2)ω)/((ν + 2) sinh πνω). With sinh(A + B)/sinh A ≤ e^B/(1 − e^{−2A}),
P²_{ν+2} ≤ (ν/(ν + 2)) e^{2πω}/(1 − e^{−2πνω}), and this is below e^{2πω} whenever e^{−2πνω} < 2/(ν + 2), which
holds for every ν ≥ 1 at this ω (2πω ≈ 24). So P = sup_ν P_{ν+2} ≤ e^{πω}, approached as ν → ∞. Holds.

**B3.** Linearity in p puts the perturbation in exactly one slot of each triple (above). For the perturbation factor,
|T^μ_a| = |ρ^μ_a| |T^ν_a| ≤ P_μ |T^ν_a| by B2, and |T^ν_a|/|T^ν_N| = Π_{m=a}^{N−1} |a^ν_m|^{−1} ≤ Π_{m=a}^{N−1} r_m^{−1}
= t_a/t_N by B1 with |1| ≤ |ν|. The two background factors have |T^{±1}_b| = t_b. So |T^p_a T^q_b T^r_c|/|T^ν_N| ≤
P_μ t_a t_b t_c/t_N needs only a ≤ N. The index sums are a + b + c = j − 1 in S^s_j and j in S^d_{j+1}, with N = j + 1,
so a + b + c < N holds but is more than needed. The prefactors (c_j, the 2l + 1 + iλω of s̃, 1/(8m)) and the |d₀|
(1 for the background, |p| for the perturbation) are outside B3. Holds.

**B4.** |2j + 2 + iνω| ≤ 2j + 2 + νω ≤ 2j + 2 + jω/κ for ν ≤ j/κ; divided by 4(j + 1) = 4j(1 + x), this is
(2 + 2x + ω/κ)/(4(1 + x)). |2l + 1 + iλω| ≤ 2l + 1 + |λ|ω ≤ 2l + 1 + (ν + 2)ω. Both are the triangle inequality and
hold for ω > 0. The labels that occur in prefactors are outer labels λ₃ ∈ {μ, ±1}, so |λ₃| ≤ ν + 2. Holds.

**A fact B5 and B7 need, re-derived: the convolution bound U_n = Σ_{l=0}^{n} t_l t_{n−l}/t_n ≤ 2 + t₁.** r_m² =
((m + ½)² + w)(1 + w/(m + 1)²), w = ω²/4. r_m ≤ (m + 1)(1 + w/(m + 1)²) = m + 1 + w/(m + 1), and r_{m+1} ≥ m + 3/2, so
r_{m+1} > r_m once (m + 1)/2 > w, i.e. m > ω²/2 − 1 ≈ 6.3. Below that, monotonicity for m ≥ 1 is a finite check
(r₀ > r₁ fails it at m = 0). With r increasing on m ≥ 1, v_l = t_l t_{n−l} has v_{l+1}/v_l = r_l/r_{n−1−l} ≤ 1 for
1 ≤ l ≤ (n − 1)/2, so on 1 ≤ l ≤ n − 1 the largest v_l is at l = 1 or n − 1, equal to t₁t_{n−1}. Then for n ≥ 2,
U_n ≤ 2 + (n − 1) t₁ t_{n−1}/t_n = 2 + t₁(n − 1)/r_{n−1} ≤ 2 + t₁(n − 1)/(n − ½) < 2 + t₁; U₀ = 1, U₁ = 2. A corollary
used below: each single product t_a t_b ≤ U_{a+b} t_{a+b} ≤ (2 + t₁) t_{a+b}.

**B6.** For j ≥ j₀, |ρ^{±1}_j| ≤ |ρ^{±1}_{j₀}| by B2(i). For ν ≥ 101 and m < M = ⌊νω/4⌋: |a^1_m/a^ν_m|² =
A·B with A = ((m + ½)² + w)/((m + ½)² + ν²w), B = ((m + 1)² + w)/((m + 1)² + ν²w). Both have numerator ≤ (νω/4 + 1)²
+ w (as m + 1 ≤ M ≤ νω/4) and denominator ≥ ν²w, so each is ≤ q and the product ≤ q². The factors with m ≥ M are ≤ 1.
So ν|ρ_{j₀}| ≤ ν q^M, since M ≤ νω/4 < 2ν ≤ j₀. q = ¼ + (νω/2 + 1 + w)/(ν²w) decreases in ν. Going from ν to
ν + 2 multiplies ν q^M by at most ((ν + 2)/ν) q ≤ (103/101) q < 1 once q < 101/103, since M grows by at least
⌊2ω/4⌋ = 1. Holds as stated (the statement's "q < e^{−1}" is more than needed).

**B7 (head, term by term).** From the model: −b_j/T^ν_{j+1} has, per triple and per i, the S^s term
c_j g_i (2(j − i) + 1 + iλ₃ω) d₃ T^{λ₃}_{j−i}/T^ν_{j+1}, and |g_i T^{λ₃}_{j−i}|/|T^ν_{j+1}| ≤ (|p|P_μ/(8i))
Σ_l t_l t_{i−1−l} t_{j−i}/t_{j+1} (B3) ≤ (|p|P_μ/(8i)) (2 + t₁) t_{i−1}t_{j−i}/t_{j+1} (convolution bound) ≤
(|p|P_μ/(8i)) (2 + t₁)² t_{j−1}/t_{j+1} (corollary) = |p|P_μ A₀/(i r_{j−1} r_j). Summing i = 1 to j gives
c_j A₀ Σ_{λ₃}(j)/(r_{j−1}r_j), with Σ_λ(j) = Σ_{i=1}^{j} |2(j − i) + 1 + iλω|/i. The outer label is μ in n_μ triples
and ±1 (|λ| = 1, Σ₁) in the other N_μ − n_μ: n = 1, 2, 1 and N = 3, 6, 3. The S^d terms give, the same way,
(2 + t₁)² t_j/(8i t_{j+1}) = A₀/(i r_j), and summing i = 1 to j + 1 gives A₀ H_{j+1}/r_j per triple. The de-mixing
term is at most |E| |d₀| (|ρ_{j+1}| + |ρ_j|) ≤ 2|E| w |d₀| per family, and the ψ term is h₂ |p| (1/|α_ν + j| +
1/|β_ν + j|). This is B7 as stated. It needs the convolution bound, so it needs r increasing on m ≥ 1 at this ω.

**The closed form G (B7 summed over j < j₀).** c_j ≤ ½ + νω/(4(j + 1)); Σ_λ(j) ≤ (2j + 1 + (ν + 2)ω) H_j − 2j ≤
ℓ(2j + 1 + (ν + 2)ω) with ℓ = 1 + ln j₀ ≥ H_j; r_{j−1}r_j ≥ j² − ¼. Expanding the product, the four pieces are
(j + ½)/(j² − ¼) = 1/(j − ½), (ν + 2)ω/(2(j² − ¼)), (νω/2)/((j + 1)(j − ½)) and (ν(ν + 2)ω²/4)/((j + 1)(j² − ¼)). Their
sums over 1 ≤ j ≤ j₀ − 1 are at most 2 + ln(2j₀ − 3), (ν + 2)ω, (νω/2) S_a and (ν(ν + 2)ω²/4) S_b, with Σ 1/(j² − ¼) = 2
(telescoping), S_a = (2/3)(ψ(2) − ψ(½)) = (2/3)(1 + 2 ln 2) and S_b = −(4/3)ψ(2) − (2/3)ψ(½) + 2ψ(3/2) = (8/3)(1 − ln 2)
(partial fractions 4/3, 2/3, −2 at j + 1, j − ½, j + ½). The S^d part sums to at most ℓ(1/r₀ + Σ_{j=1}^{j₀−1} 1/(j + ½))
≤ ℓ(1/r₀ + ln(2j₀ − 1)). With N_μ triples each bounded by the |λ| ≤ ν + 2 sum, this is G as stated, with T₁ = 2 +
ln(2j₀ − 1) ≥ 2 + ln(2j₀ − 3). D_j ≤ D̄ uses |α_ν + j|, |β_ν + j| ≥ νω/2.

**The monotonicity of e^{−0.736ν}G.** For nonnegative pieces G_k, G′/G ≤ max_k G_k′/G_k. Each piece is a constant
times ν^a (a ≤ 2, counting ν + 2 as ν) times at most two of ℓ = 1 + ln 2ν, ln(4ν − 1), 2 + ln(4ν − 1), whose
logarithmic derivatives are each ≤ 1/ν for ν ≥ 7, or a factor that falls with ν (j₀D̄ = 2νD̄ is ν times a constant
plus falling terms). So d ln G/dν ≤ 4/ν and e^{−0.736ν}G decreases for ν > 4/0.736 = 5.4. Holds; it uses P_μ ≤ P
(a ν-free constant) for μ = ν + 2.

**The head for ν ≤ 99** is an exact finite sum per ν, so it needs no lemma: only that the recursion is the model and
that the enclosures are rigorous. That is what PC1 and PC6 test.

**The tail consequence.** Σ_{j≥j₀} ν/j² ≤ ν/(j₀ − 1) ≤ ν/(2ν − 1) ≤ 3/5 for ν ≥ 3. Holds.

## 4. The written proofs against the re-derivations

Read after §3 was committed (ddc1844a). P6_route3.md §10.1 and §11.2, and D_entire.md §6.3 for the convolution bound.

| lemma | written proof | against §3 |
|---|---|---|
| B1 | the two moduli and the drop of λ²ω²/4 | the same argument |
| B2 (i), (ii) | the step ratio, the Γ-ratio limit, DLMF §5.4, the sinh inequality | the same argument |
| B3 | conjugate symmetry of T^{±1}, then B2 and B1 on the perturbation factor | the same argument; "a + b + c < N" is not used, and "P_μ = 1 for μ ≤ ν" should read μ ∈ {ν − 2, ν} (finding 8) |
| B4 | stated without proof | the triangle inequality, as in §3 |
| B5 | D_entire.md §6.3's grouping; the convolution bound is cited, not shown | the convolution bound holds (§3); the grouping itself is checked in code (§5.3) |
| B6 | the hand bound ν q^M and "that bound decreases in ν" | holds; the decrease needs the one-line step argument of §3, which the code's docstring replaces by a weaker one (finding 7) |
| B7 | steps 1–5: B3, the convolution bound twice, the index sums, B2 | the same argument, step for step |
| G | the listed inequalities, partial fractions, ψ values | the same inequalities; T₁ = 2 + ln(2j₀ − 1) is slightly larger than needed (2 + ln(2j₀ − 3)) |
| monotonicity of e^{−0.736ν}G | "every piece ... of the form ν^a L₁L₂", d ln/dν ≤ 4/ν | holds; the step G′/G ≤ max_k G_k′/G_k and L ≥ 1 are implicit (finding 9) |

No step of a written proof fails. The scripts were then read: route3_bound.py's `Cr_sup`, `W_sup` and `sup_box`;
route3_head.py's `B_term`, `G_closed` and `certify`; route3_cancel.py's `tail`, `order3` and `dr_builder`; and
route3_tuned.py. `Cr_sup` was checked line by line against the grouping it states. Every term bounds its sum for
j ≥ 10: Ss at i = 3, 4 ≤ i ≤ j − 3, j − 2, j − 1 and j; Sd at i = 3, 4 ≤ i ≤ j − 2, j − 1 and j; and the inner far
end. The counts are j − 6 and j − 5, and the log-convexity maxima are t₃t_{j−4} and t₃t_{j−3}.

## 5. The re-implementation and its constants

Every number below is printed by a script of `solver/proof/review_p6/` (mpmath 1.3.0, sympy 1.14.0, the venv-314 of
the Mac) and is an mpmath.iv upper bound rounded up, or a lower bound rounded down, unless it is marked as a value.
The logs are in folder I1062-mac of the raw-data repository (commit 3042bbb3). ω = 2π·160000/263023 (= 2π/1.64389375)
is built as an interval.

### 5.1 The model against amp_flint (PC1)

`ampcheck.py` runs `amp_flint.Amp` untuned. It injects the Kummer input ε T^μ at its natural order |μ| through
`add_linear` and reads `S_at`'s own late-order sequence at order |μ| + 2, with mpmath.qr_solve replaced so as to capture
it. The part linear in ε is (y(+ε) − y(−ε))/(2ε) with ε = 2⁻⁶⁰. Against it, model.py's literal form (amp_flint's passes
over labels, with p as a dual number, and ψ, Γ) and its Darboux form over triples agree:

| ν | μ | injected at order | read at order | worst relative difference, j ≤ 40 |
|---|---|---|---|---|
| 3 | 1 | 1 | 3 | 2.5×10⁻³⁷ |
| 3 | 3 | 3 | 5 | 1.4×10⁻⁶⁵ |
| 3 | 5 | 5 | 7 | 1.0×10⁻⁶⁷ |
| 5 | 3 | 3 | 5 | 4.8×10⁻⁶⁶ |
| 5 | 5 | 5 | 7 | 3.9×10⁻⁶⁷ |
| 5 | 7 | 7 | 9 | 1.5×10⁻⁶⁹ |

At ν = 3, μ = 1 the perturbation shares the background's label, and the ε³ term limits the agreement to ε² ≈ 10⁻³⁶.
**So the one-step model is exactly the part of amp_flint's expansion that is linear in a Kummer input of label μ,
read two orders above it.** That makes the model a sub-problem of the real recursion, not a separate one. It does
not cover tuned inputs (§7).

### 5.2 The explicit part (PC2, PC3, PC4)

`explicit.py` builds the C_λ from model.py's definitions with sympy Polys over ℚ(i), on a common denominator of the
factors 2(j − q + α_λ), 2(j − q + β_λ), j + 1, ν ∓ 1 and ω. In s = 1/ν and u = ν/j, j²C_λ/ν^d is then a polynomial
Ñ(s, u) over a product of factors that stay away from zero on the box. That is an exact statement: every monomial
j^p ν^q of the numerator has p ≤ deg_j D − 2 and the right total degree. It holds for all seven families (PC3; deg_j
D = 7, deg_j N = 5). Σ_λ C_λ(j) ρ^λ_j equals model.split()'s explicit terms, and those plus model.split()'s
interior equal model.darboux()'s increment, both to 1.4×10⁻⁹³ at ν ∈ {3, 5, 9, 21} and j ∈ {10, 25, 60, 200}
(PC2). The sup of |Ñ|/|D̃| over s ∈ [0, 1/3], u ∈ [0, 1/2] comes from adaptive bisection in mpmath.iv (80 bits),
stopped when the largest box bound is within 0.5% of the best point value:

| μ | family | d | review U (certified) | review L (point value) | certificate | certificate's lower bound | boxes |
|---|---|---|---|---|---|---|---|
| ν − 2 | ν − 2 | 1 | 0.47868 | 0.47652 | 0.5638 | 0.4719 | 632 |
| ν − 2 | +1 | 2 | 0.91519 | 0.91066 | 0.9229 | 0.9003 | 140 |
| ν | ν | 1 | 1.0094 | 1.0044 | 1.1028 | 0.9944 | 368 |
| ν | −1 | 2 | 1.0152 | 1.0101 | 1.0454 | 1.0071 | 272 |
| ν | +1 | 2 | 0.9233 | 0.9187 | 0.9530 | 0.9183 | 3220 |
| ν + 2 | ν + 2 | 1 | 1.0493 | 1.0443 | 1.1775 | 1.0320 | 148 |
| ν + 2 | −1 | 2 | 2.5837 | 2.5714 | 2.6323 | 2.5528 | 164 |

PC4 holds for all seven: L ≤ the certificate's value, and here also U ≤ it. The certificate's 48 × 48 grid is 0.8% to
18% above my U.

### 5.3 The interior (PC5)

`interior.py` evaluates the termwise majorant of the interior, after B3, by direct summation for 10 ≤ j ≤ J₁ = 2000:

```math
c\sum_{i=3}^{j}\hat G_i X_{j-i}\frac{t_{j-i}}{t_{j+1}}+\sum_{i=3}^{j}\hat G_i\frac{t_{j+1-i}}{t_{j+1}}+\frac{1}{8(j+1)}\sum_{l=1}^{j-1}\frac{t_lt_{j-l}}{t_{j+1}},\qquad \hat G_i=\frac{1}{8i}\sum_{l<i}t_lt_{i-1-l},
```

per triple, with c and X_l the prefactor bounds. In B4 mode (c from B4 with κ = 2, X_l = 2l + 1 + (ν + 2)ω) it is
A_j + νB_j, and j²(A_j + νB_j)/ν ≤ j²(A_j/3 + B_j). In the tight mode, c = ½ + νω/(4(j + 1)), X_l = 2l + 1 + |μ|ω for
the perturbation outside and |2l + 1 ± iω| for a background label outside, with the sup over ν ∈ [3, j/2] on 24
ν-intervals. Above J₁ my own bound applies, with Ĝ_i ≤ (2 + t₁)t_{i−1}/(8i). The ends i = j, j − 1 are taken exactly.
The rest uses t_{i−1}t_{j−i} ≤ t₂t_{j−3} and t_{i−1}t_{j+1−i} ≤ t₂t_{j−2} (log-convexity), and t_{j−k}/t_{j+1} ≤
Π_{q≤k}(j − q + ½)⁻¹. All are evaluated with x = 1/j ∈ [0, 1/2001] and ln j/j ≤ ln 2001/2001.

| quantity | review | certificate |
|---|---|---|
| r_m increasing on 1 ≤ m ≤ 8 (checked), above ω²/2 − 1 ≤ 6.3044 (proved) | yes | yes (to m = 9) |
| max_{n≤2000} U_n/(2 + t₁) | 0.7773 | not printed |
| B4 mode, per triple, max over 10 ≤ j ≤ 2000 (at j = 10) | 1.4649 | 9.295 (27.885/3) |
| B4 mode, per triple, every j > 2000 | 0.0458 | |
| Cr, μ = ν ∓ 2 (3 triples), B4 mode | 4.3947 | 27.885 |
| Cr, μ = ν (6 triples), B4 mode | 8.7894 | 55.770 |
| Cr, tight mode, μ = ν − 2, ν, ν + 2 | 1.9094, 4.3856, 2.4762 | |

PC5 holds: the direct sum with B4's per-term bounds is 6.3 times below the certificate's grouping of the same terms.
The grouping is loose, not wrong. The ν = 1 equality test passes to 1.6×10⁻³⁶: at ν = 1 the majorant with exact
prefactors equals the sum of the moduli of every interior term computed by model.py from complex T's.

### 5.4 The weights

| quantity | review | certificate |
|---|---|---|
| P = sup P_{ν+2} ≤ e^{πω} | 163995.5 | 1.63995×10⁵ |
| P_{ν+2}'s closed form against mpmath's Γ ratio, ν = 3, 5, 9, 41 | 1.9×10⁻⁹⁵ relative | checked (`--lemmas`) |
| W, odd ν ≤ 99 (at ν = 3) | 4.7282×10⁻⁴ | 4.728×10⁻⁴ |
| hand bound ν q^M at ν = 101 (q ≤ 0.2554, M = 96) | 1.21×10⁻⁵⁵ | 1.2×10⁻⁵⁵ |

### 5.5 The tail constants

C′_μ = Σ_λ S_λ w_λ + Cr P_μ, the certificate's assembly with the review's parts (`constants.py`):

| μ | review, B4 mode | review, tight mode | certificate | implied |
|---|---|---|---|---|
| ν − 2 | 4.8738 | 2.3885 | 28.449 | yes |
| ν | 9.7996 | 5.3958 | 56.874 | yes |
| ν + 2 | 8.9279×10⁵ | 5.7816×10⁵ | 4.7661×10⁶ | yes |

**So claim T holds, by an independent certificate, with constants 5 to 12 times smaller.** The interior dominates
every C′. For μ = ν + 2 the weight P ≈ e^{πω} multiplies it and sets the size.

### 5.6 The head (PC6, PC7, PC8)

`head.py --certify` sums the upper ends of |ΔR_j/K_ν| over j < j₀ from model.darboux() in mpmath.iv at 200 bits. The
largest relative radius is 9×10⁻¹² (ν = 99, μ = ν − 2), from rectangular complex products, and harmless at the digits
quoted. Each cell gives my enclosure; the certificate's §11.3 value follows the slash:

| ν | H, μ = ν − 2 | H, μ = ν | H, μ = ν + 2 |
|---|---|---|---|
| 3 | [0.10018557, 0.10018558] / 0.10019 | [0.53509672, 0.53509673] / 0.53510 | [30.087974, 30.087975] / 30.088 |
| 5 | [0.065828953, 0.065828954] / 0.065829 | [0.43643742, 0.43643743] / 0.43644 | [7.1812233, 7.1812234] / 7.1812 |
| 11 | [0.045826518, 0.045826519] / 0.045827 | [0.417755, 0.41775501] / 0.41776 | [10.690416, 10.690417] / 10.690 |
| 21 | [0.039215188, 0.039215189] / 0.039215 | [0.41116362, 0.41116363] / 0.41116 | [12.761465, 12.761466] / 12.761 |
| 41 | [0.035920864, 0.035920865] / 0.035921 | [0.40769583, 0.40769584] / 0.40770 | [14.103416, 14.103417] / 14.103 |
| 99 | [0.033975676, 0.033975677] / 0.033976 | [0.40556992, 0.40556993] / 0.40557 | [15.024773, 15.024774] / 15.025 |

PC6 holds at all 18 entries. B7, coded from its statement, lies above every head term at odd ν = 3 to 21 (PC7). The
worst ratios are 0.023525 (μ = ν − 2), 0.023885 (ν) and 8×10⁻⁶ (ν + 2), each at ν = 3, as in §11.4. The sum of B7 over
the head is at most 0.0279, 0.0322 and 0.0287 of G at odd ν = 3 to 41.

| μ | sup_{ν≤99} e^{−0.736ν}H (at ν = 3) | e^{−0.736·101} G(101) | certificate Λ^head | certificate e^{−0.736·101} G(101) |
|---|---|---|---|---|
| ν − 2 | 0.011012426 | 1.535×10⁻²⁶ | **0.011012** | 1.53×10⁻²⁶ |
| ν | 0.058817977 | 3.069×10⁻²⁶ | 0.058818 | 3.07×10⁻²⁶ |
| ν + 2 | 3.3072783 | 2.517×10⁻²¹ | 3.3073 | 2.52×10⁻²¹ |

PC8 fails at one entry. The certificate's Λ^head_{ν−2} = 0.011012 is below the sup it is meant to bound
(finding 1). The other two Λ are valid upper bounds, at their last digit.

### 5.7 The whole sum

With the review's C′ and H, sup over odd ν of e^{−0.736ν}(H + C′ν/(j₀ − 1)) is at most 0.18959, 0.41788 and 32715 in
B4 mode, and 0.098526, 0.25652 and 21188 in the tight mode, each at ν = 3. Beyond ν = 99 it is at most 1.54×10⁻²⁶,
3.07×10⁻²⁶ and 2.52×10⁻²¹. The certificate's 1.0534, 2.1427 and 1.7463×10⁵ are true, by these smaller bounds, and 5
to 11 times larger, because its tail constants are.

### 5.8 The constants against the exact increments (PC9)

`tailcheck.py` computes |ΔR_j/K_ν| from the review's Darboux form in mpmath at 192 bits, for odd ν = 3 to 21, each μ and
j₀ ≤ j < 1000. It compares them with C ν/j² for three sets of constants. There is no violation:

| μ | worst ratio, review B4 | worst ratio, review tight | worst ratio, certificate | at (ν, j) |
|---|---|---|---|---|
| ν − 2 | 7.8×10⁻⁵ | 1.6×10⁻⁴ | 1.34×10⁻⁵ | (21, 42) |
| ν | 0.103 | 0.186 | 0.0176 | (3, 999) |
| ν + 2 | 0.140 | 0.216 | 0.0262 | (3, 999) |

The certificate's own §10.3 gives 1.3×10⁻⁵, 1.8×10⁻² and 2.7×10⁻² to j = 1600. For μ = ν and ν + 2 the ratio rises
toward |c₂|/(νC) at large j, which is why the worst is at the last j. That limit is at most S_λ/C, below 1. With
three times the bound planted at five or six j per run, all 90 runs fail, with 57 violations in each block of ten,
and the script exits 1.

## 6. The planted errors

`controls.py` runs each one, and each run is logged separately in I1062-mac. A mutation counts as caught only by
exit 1 with a printed failed check. A traceback or any other exit code fails the control, so a crash cannot pass for
a caught error:

| id | mutation | check | result | exit |
|---|---|---|---|---|
| M1 | E × 3/2 in the explicit part | PC3 | six of seven families unbounded (deg_j N = 6 against deg_j D = 7; the ψ family is unaffected), and PC2 off by 14.2, 4.9×10⁻⁴ and 2.64 | 1 |
| M2 | far-end l = j term dropped | PC2 | off by 0.234, 0.140 and 0.0695 (μ = ν − 2, ν, ν + 2) | 1 |
| M3 | 3ω for (ν + 2)ω in B4 | prefactor check | fails | 1 |
| M4 | g₂ × 1.01 | PC2 | off by 0.021, 0.0074 and 0.0037 | 1 |
| M5 | S^s term i = j dropped from the majorant | ν = 1 equality test | off by 0.074 | 1 |
| M6 | one ordered lapse pair dropped from the model | PC1 | off by 3.35 | 1 |
| M7 | A₀/100 in B7 | PC7 | **did not fire**: worst ratio 0.372, 0.488 and 4.9×10⁻⁴, no violation | 0 |
| M7′ | all of B7 divided by 100 | PC7 | worst ratio 2.353, 2.389 and 8.4×10⁻⁴, violations | 1 |
| plant | three times C′ν/j² at five or six j per run | PC9 | 57 violations in each block of ten runs | 1 |

**M7 did not fire, and why.** A₀ multiplies only B7's convolution part, not the de-mixing and ψ part D_j. With A₀/100,
B7 still bounds every head term at ν = 3 to 21, and D_j carries most of it. At ν = 3, j = 0, μ = ν − 2,
D₀ = 2(1/(8ω) + 1/(4ω)) = 0.196, against |ΔR₀/K| = 0.0858 (model.darboux). So A₀/100 is not a planted error at all:
B7 stays true. That is a fact about B7's slack, not a fault in it. M7′, which divides all of B7, takes its place, and
M7 stays in `controls.py` as a record with the expected exit 0. No threshold was changed.

## 7. What is proved, and what rests on (H1) and (H2)

**What the code and the lemmas prove** (*certified*, both by the certificate and, independently, by this review). Fix
ω = 2π/1.64389375. For odd ν ≥ 3 and μ ∈ {ν − 2, ν, ν + 2}, let D_j be the coefficient of p in ΔR_j/K_ν, the increment
of the order-3 normalised multiplier sequence of harmonic ν. Its order-1 inputs are −i T^1, +i T^{−1} and p T^μ. Then:

- for every j ≥ max(2ν, 10), |D_j| ≤ C_μ ν/j², with the review's C′_μ of §5.5. Hence the bound also holds with the
  quoted C_μ, although the certificate by itself proves 28.4492, 56.874 and 4.76611×10⁶ (finding 1);
- for every odd ν ≥ 3, Σ_{j<j₀} |D_j| ≤ Λ e^{0.736ν}, with Λ = 0.011013, 0.058818 and 3.3073;
- hence Σ_{j≥0} |D_j| ≤ e^{0.736ν} times the constants of §5.7.

By §5.1, D_j is exactly the part of amp_flint's own late-order sequence at order |μ| + 2 that is linear in a Kummer
input ε T^μ added at order |μ| of an untuned expansion.

**What it does not prove.**

- Anything for tuned or forced inputs, whose late orders are not Kummer sequences (below).
- Terms outside the one-step class at the orders that matter. At harmonic ν's first tuned order the forcing also has
  products with two non-background factors, for example labels (3, 3, 1) at ν = 7. amp_flint forms that one through
  e^{(6)} ∋ d^{(3)} d^{(3)} and then g^{(6)} ∗ s^{(1)}. The forcing also has the nested lapse terms g^{(2)} ∗ e^{(2)} of
  `pass_even`. P6_route3.md lists this as "orders beyond one A² step".
- Any ω but this one (§10.5 item 4).
- A bound on the head that is independent of ν beyond ν = 99. The closed form G grows like ν² log ν. The
  certificate proves H ≤ Λ e^{0.736ν}, which is the form (★) allows. "The head does not grow with ν" is numerical
  evidence, and it is labelled so.
- (★) itself. B_tail_M7.md §8.3 explains why the lowest order does not settle it.

**What rests on (H1).** Under the envelope |X_l[μ]| ≤ X̄_μ |T^μ_l|, B3 holds with X̄_μ for |p|. So the interior bound
and B7's convolution and de-mixing parts carry over with that factor, for terms of the one-step form. B7's ψ term does
not. It uses the Kummer difference ψ(α_ν + j + 1) − ψ(α_ν + j) = 1/(α_ν + j), which holds only when the input's
ratio X_j/T^ν_j is constant. For a general input under (H1), the ψ-weighted term changes by up to about
X̄_ν(|ψ(α_ν + j + 1)| + |ψ(α_ν + j)|) per step, of order X̄_ν ln j. That would keep the head polynomial in ν, but it is
not B7. The ψ term is present only for μ = ν. (Credit: CodeRabbit's review of #1063.)

**What rests on (H2), and why it is not enough as stated.** The explicit part uses the exact Kummer form of the
perturbation twice:

- to write T^μ_{j−k}/T^ν_{j+1} as ρ^μ_j times a rational function, from which the S_λ are computed;
- for the O(1/j) cancellation.

For a tuned input, the near-end terms X_{j−1}/T^ν_{j+1}, X_{j−2}/T^ν_{j+1} and X_j/T^ν_{j+1} are each of order X̄/j
under (H1). (H2) says only that the full multiplier sequence has no O(1/j) increments. It gives neither the j⁻² rate
nor a constant independent of ν for what is left. So "under (H1) and (H2) the form C ν/j² survives, with C multiplied
by X̄_μ and recomputed" does not follow (finding 2). It would follow from a quantitative (H2′): the explicit part of
the tuned increment is at most C₂ X̄_μ ν/j² for j ≥ j₀, with C₂ independent of ν. That is the open content. X̄_μ's
own ν dependence is also not controlled.

**Assumed facts**, each named:

- |Γ(½ + iy)|² = π/cosh πy and |Γ(1 + iy)|² = πy/sinh πy (DLMF §5.4), for B2(ii);
- |Γ(L + α + iω)/Γ(L + α)| → 1 as L → ∞, for B2(ii)'s limit (DLMF §5.11). Not in §10's list of what the bound uses;
- ψ(x + 1) − ψ(x) = 1/x, for the Darboux form's ψ difference (DLMF §5.5). Not in that list either;
- ψ(2) = 1 − γ, ψ(½) = −γ − 2 ln 2 and ψ(3/2) = 2 − γ − 2 ln 2, for S_a and S_b (DLMF §5.4);
- the correctness of the interval libraries (arb in FLINT 3.6.0 through python-flint 0.9.0 for the certificate,
  mpmath 1.3.0's iv for the review) and of sympy 1.14.0's polynomial arithmetic over ℚ(i), which both use;
- amp_flint.py as the definition of the problem.

## 8. What the review shares with the code under test

| item | shared | how far it was checked here |
|---|---|---|
| the model's definition | both read it from amp_flint.py (`pass_even`, `pass_odd`, `S_at`) | the review's model is checked against amp_flint itself, run as a black box (§5.1); route3_cancel.py was not used |
| ω | the same decimal 1.64389375 | — |
| the split into explicit part and interior, and the assembly C = Σ S w + Cr P | the same definitions (from P6_route3.md §10) | PC2 checks that the split sums to the increment |
| B3, B4 and B7 | the statements, re-derived in §3 and then used by the review's code | B4 and the ν = 1 equality test check them pointwise; B7 against every head term |
| the convolution bound and log-convexity | the same fact | proved in §3, checked to n = 2000; the review uses it only above J₁ = 2000 and in B7 |
| sympy 1.14.0 | both build the C_λ with it | the review's C_λ are checked numerically against model.split() in mpmath (no sympy), to 1.4×10⁻⁹³ |
| mpmath 1.3.0 | the review's arithmetic; the certificate uses it only in checks | — |
| not shared | | the interval library (mpmath.iv against arb); the sup method (adaptive bisection against a fixed grid); the interior method (direct termwise sums against grouping); the recursion code; the head's exact sums |

The review's author wrote all of `review_p6/` in one session. The re-derivations were committed before the proofs
were read (§4). model.py, ampcheck.py, explicit.py, interior.py and head.py were written and run before the five
scripts were read in detail. constants.py, tailcheck.py and controls.py, which assemble and check, came after.

## 9. Findings

Numbered by severity: wrong, then unproved, unclear and minor.

**Wrong**

1. **Λ^head_{ν−2} = 0.011012 is not an upper bound.** At ν = 3, H_{ν−2}(3) ≥ 0.10018557 (review, mpmath.iv; §11.3's
   own table gives 0.10019). But 0.011012 e^{0.736·3} = 0.10018171. So "H_μ(ν) ≤ Λ^head_μ e^{0.736ν}" fails at ν = 3 for
   μ = ν − 2, by 3.9×10⁻⁵ relative. The sup is 0.01101243, so the value should read 0.011013. The cause is the
   printing. route3_head.py and route3_bound.py print certified upper bounds with arb's `.str(n, radius=False)`, which
   rounds to nearest. AGENTS.md asks for outward rounding. Λ^head_ν = 0.058818 and Λ^head_{ν+2} = 3.3073 are valid, at
   their last digit (sups 0.058817977 and 3.3072783). The same printing gives C_μ, S_λ, Cr, W, P and G(101).
   route3_bound.py's own upper ends are C_{ν−2} ≤ 28.4491411, C_ν ≤ 56.8734956 and C_{ν+2} ≤ 4766104.06 (printed to
   12 digits by the review). So two of the quoted constants, 28.449 and 4.7661×10⁶, are below what the certificate
   proves. route3_head.py hard-codes those two values (`C_TAIL`), and its whole-sum bound for μ = ν + 2 evaluates to
   1.74634×10⁵, against the quoted 1.7463×10⁵. Those constants lie far above the true sups (§5.5, §5.7), so the
   statements stay true by this review's certificate, but the certificate does not prove them as printed. The
   error changes no conclusion.

**Unproved**

2. **§10.4's conditional extension to tuned inputs does not follow from (H1) and (H2) as stated** (§7). (H1) carries the
   interior over, and B7 except its ψ term (μ = ν), which needs the Kummer form. But the explicit part's j⁻² rate and
   its ν-uniform constant also come from the perturbation's exact Kummer form. (H2) is qualitative, so it supplies
   neither, and the constants "recomputed for the families the tuned inputs carry" are not computed. In addition, the
   real first tuned order of harmonic ν ≥ 7 contains terms outside the one-step class: two non-background factors, and
   nested lapse terms. This matters for route 3's purpose. P6_scope.md §4 stage 3 asks for "the first tuned order of
   harmonic ν". What is certified is the one-step model with Kummer inputs, and the step from that to the first tuned
   order is exactly this conditional claim. The Result line "Tuned inputs keep the form, *conditional* on two named
   hypotheses" should read as a conjecture, or (H2) should be strengthened to the quantitative (H2′) of §7.

**Unclear**

3. **The evidence for (H2) at first tuned orders is two cases.** route3_tuned.py's ν = 3 row (order 3) has no tuned
   input: its only inputs are the order-1 background, the h3 case of §1. Of the four later orders, (ν, order) = (1, 3)
   has none either. So tuned inputs enter at ν = 5 and 7 and at (1, 5), (3, 5) and (5, 7). The window test cannot see
   a factor log j, which §10.4 says.
4. **The link from the three perturbations to (★) is not stated.** The μ = ν + 2 constant is 4.77×10⁶ (5.8×10⁵
   here). It is almost all the untuned weight P ≈ e^{πω}, which §10.5 says the tuned problem removes. The note does not
   say which μ (★)'s coupling needs. So it is not clear whether the certified constant that carries P is the one that
   matters.

**Minor**

5. **The interior grouping is 6.3 times looser than the termwise sum it bounds** (§5.3). The direct sum gives
   C_μ ≤ 4.874, 9.800 and 8.928×10⁵ with B4's per-term bounds, and 2.389, 5.396 and 5.782×10⁵ with exact prefactors. It
   makes the whole-sum constants 0.190, 0.418 and 3.27×10⁴ (B4) instead of 1.0534, 2.1427 and 1.7463×10⁵.
6. **The explicit sups' grid is 0.8% to 18% above the sup** (§5.2). This is valid, and the note's "within 20%" says
   so.
7. **B6's decrease above ν = 99.** route3_bound.py's `W_sup` docstring argues through ν q^{νω/4 − 1}. At ν = 101 that
   is about q^{−1/2} ≈ 2 times the quoted tail ν q^M = 1.2×10⁻⁵⁵. The quoted tail is the sup over odd ν ≥ 101 by a
   one-line step argument: M grows by at least 1 per step, q decreases, and (103/101) q < 1 (§3). It is harmless
   against W = 4.7×10⁻⁴.
8. **B3's wording.** "P_μ = 1 for μ ≤ ν" should read μ ∈ {ν − 2, ν}. The condition a + b + c < N is not used; only
   a ≤ N is.
9. **Implicit steps.** Two steps are left unwritten:
   - the convolution bound Σ_l t_l t_{n−l} ≤ (2 + t₁) t_n (B5, B7) is cited to D_entire.md §6.3, where it is stated
     without its line, U_n ≤ 2 + t₁(n − 1)/r_{n−1} (§3);
   - G's monotonicity uses G′/G ≤ max_k G_k′/G_k and that each logarithm is at least 1.
   Both hold.
10. **The list of facts the bound uses is incomplete.** §10 says it uses "only B1's elementary inequalities and two Γ
    identities (DLMF §5.4)". B2(ii) also uses the limit of a Γ ratio (DLMF §5.11), and the Darboux form uses
    ψ(x + 1) − ψ(x) = 1/x (§7).

## 10. Reproduce

From `solver/proof/review_p6/`, with the Mac's venv-314 (`/Users/nathan/venv-314/bin/python`: mpmath 1.3.0, sympy
1.14.0, python-flint 0.9.0 for amp_flint.py only). Each command runs one process. Set OUT to a folder for the JSON
files. The logs of record are in the raw-data repository, `git clone https://github.com/nmoinvaz/choptuik-large-d-raw
../choptuik-large-d-raw` from the checkout's root, folder I1062-mac.

```
python3 ampcheck.py                          # 5.1, PC1: the model against amp_flint.py; PASS, exit 0 (~10 s)
python3 explicit.py --json=$OUT/explicit.json   # 5.2: the seven sups; PASS (~20 s)
python3 interior.py --json=$OUT/interior.json   # 5.3, 5.4: Cr, P, W; PASS (~90 s)
python3 head.py --certify --json=$OUT/head.json # 5.6: H for odd nu = 3..99, G(101), Lambda_head (~8 min)
python3 head.py --check                      # 5.6, PC7: B7 against every head term, nu = 3..21; PASS (~15 s)
python3 constants.py $OUT/explicit.json $OUT/interior.json $OUT/head.json $OUT/constants.json   # 5.5, 5.7
python3 tailcheck.py $OUT/constants.json 1000   # 5.8, PC9: no violation (~12 min); PASS
python3 tailcheck.py $OUT/constants.json 200 --mutate=plant   # control: FAIL, exit 1
python3 head.py --check --mutate=b7          # M7': FAIL, exit 1; --mutate=a0 (M7) passes, as recorded in 6
python3 controls.py                          # 6: every control as expected (~3 min); PASS
```

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
  fixed grid: boxes are split until the upper bound is within 2% of the best point value.
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
So ν|ρ_{j₀}| ≤ ν q^M, since M ≤ νω/4 < 2ν ≤ j₀. q = ¼ + (νω/2 + 1 + w)/(4ν²w) decreases in ν. Going from ν to
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

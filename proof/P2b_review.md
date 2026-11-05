# P2b: second implementations of Lemma LC's bound algebra and Lemma J's secant coefficients

Completion of queue item P2 ([notes/PROOF_QUEUE.md](../notes/PROOF_QUEUE.md)) after the first review
([P2_review.md](P2_review.md)), which left two items only partly re-implemented (its §§3.2, 3.3). The reviewer is a
fresh agent that had not seen how P1 reached its results and did not read P1's transcripts. It worked from the
statements and derivations ([B_chain.md](B_chain.md) §§12–15, [B8_constants.md](B8_constants.md) §§1–6, P2_review.md
§§2–3) and wrote its own code before opening the certificate code or its logs. Written by Claude (Anthropic) under
Nathan Moinvaziri's direction; not checked by a mathematician. Status words as in [AGENTS.md](../AGENTS.md). Code and
logs are in `solver/proof/review_p2b/`; `run_p2b.sh` there runs everything.

Contents

- [0. Verdict](#0-verdict)
- [1. Method, arithmetic and what is shared](#1-method-arithmetic-and-what-is-shared)
- [2. Lemma LC: the second implementation](#2-lemma-lc-the-second-implementation)
- [3. Lemma LC: comparison with the certificate](#3-lemma-lc-comparison-with-the-certificate)
- [4. Lemma J: the second implementation and the comparison](#4-lemma-j-the-second-implementation-and-the-comparison)
- [5. Mutations](#5-mutations)
- [6. P2's findings F1–F7 applied, and the reruns](#6-p2s-findings-f1f7-applied-and-the-reruns)
- [7. New observations](#7-new-observations)
- [8. Still not independently checked](#8-still-not-independently-checked)
- [9. Reproduce](#9-reproduce)

## 0. Verdict

**Lemma LC is certified by a second implementation, and so is Lemma J; no quoted constant or threshold changes.**

- *Lemma LC, own class.* The second implementation (`p2b_lc.py`) closes on its own: P(h) < h for every component and
  band (ratios ≤ 0.990), d₀ = 6.55×10⁵ (the certificate: 3.59×10⁶), and its 40 × 40 Lipschitz matrix (41 × 41 with
  the component of finding F2) has a certified Perron weight with θ = 0.602.
- *Lemma LC, the certificate's class.* At the certified constants (h, h′, h_Sp of `b8m_lc_K4.log`), the second
  implementation's majorant gives every value ratio below 1 (largest 0.9900) and every derivative ratio at most 1, and
  its Lipschitz matrix there has θ = 0.621. So the certificate's class, with F1–F3 built in, is mapped into itself and
  T is a contraction on it, by code that shares only the formal-series recursion with the certificate.
- *Lemma J.* The second implementation (`p2b_lcj.py`) gives θ = 0.163 on its own class and θ = 0.164 on the
  certificate's class (the certificate: 0.176).
- *The comparison.* In a configuration close to the certificate's region boxes (§3.2), d₀ agrees to 2×10⁻⁵, the V, S,
  β, c bounds on N0–Nb to 0.1–1 %; every larger difference is located and explained (§§3.3, 4.2).
- *P2's findings F1–F7* are applied to the notes and the code (§6). The reruns of `b5_label.py`, `b5_jump.py`,
  `b7_ift.py`, `b8m_lc.py` and `b8m_lcj.py` change no quoted constant: ε_L ≤ 0.3527, ln C_E = 76.9399 (d₀ = 10⁷ data),
  n₀ = e^{89.75}, C₇ = e^{67.84}, θ_LC = 0.7355, d₀ = 3.59×10⁶, θ_J = 0.176.
- *New observations* (§7): no error. Two places where the notes should say what the code does (straight radial paths
  on L; the atoms' enclosures over the box), and two conservative choices.

## 1. Method, arithmetic and what is shared

**Order of work.** The second implementations were written from B_chain.md §§12–15 and B8_constants.md §§1–6, and
their numbers were committed (commit `d89f5ca`, "own Lemma LC and Lemma J numbers, before the comparison") before any
of `b8_alg.py`, `b8m_bounds.py`, `b8m_lc.py`, `b8m_lcj.py`, `review_p2/p2_lc_fix.py`, P2's unfinished
`review_p2/p2_alg.py`, `p2_atoms.py`, `p2_b8.py`, `p2_lc.py`, `p2_lcj.py`, or the certificate logs
(`b8m_lc_K4.log`, `.out`, `b8m_lc_lip.json`, `b8m_lcj.log`, `.json`, `p2_lc_fix.log`) was opened. The tables of
B_chain.md §13.4 and §14 (the printed h, d₀, θ) were known from the notes.

**Arithmetic.**
- The formal series Z_k (k ≤ 4) is computed exactly: coefficients are polynomials in (A, ω) over the Gaussian
  rationals (`p2b_series.py`). Lemma F's residual orders (τ_D, τ_b = O(ε⁴), τ_v, ρ₄ = O(ε⁵)), the gauge c_k(1) = 0 and
  the evenness of c_k are verified exactly.
- Every other quantity is in mpmath interval arithmetic (`iv`, 106 bits for the bound algebra, 200 for the series):
  coefficients are enclosed over B8's box (A, ω/ω∞ ∈ x̂ + [−2×10⁻⁶, 2×10⁻⁶] + i[−2×10⁻⁶, 2×10⁻⁶]); each bound is the
  upper endpoint of an interval operation (outward rounding); lower bounds are lower endpoints; φ_δ = arcsin(δ/2) is
  enclosed by checking sin at both ends (mpmath's iv has no arcsin). No python-flint.
- Perron weights are checked in exact rationals (v rationalized, Mv ≤ θv with the matrices' upper bounds).

**What is shared with the certificate code** (AGENTS.md: independent checks are independent).
- The order-by-order recursion of the formal series is P2's (`review_p2/p2_formal.py`, derived from the original
  p, m, b, c equations), re-typed for an exact coefficient ring. P2's recursion is itself independent of
  `b8m_formal.py`; the certificate's atoms agree with p2b's at the box centre (§3.4).
- The statements: the model (B8_constants.md §1.1), Lemma R's identities (§4), Lemma K's kernel bounds (§5), the
  bounded forms on L (B_chain.md §13.3), Lemma J's Γ formulas (§14), the regions and paths (B8_constants.md §1.3,
  B_chain.md §13.1), the box and n₀. P2 checked Lemma R and the L forms independently (P2_review.md §3.2).
- Not shared: the bound algebra (representation, sums, sups, inverses), the operator norms, the kernel gains, the
  integrals along the paths, the Lipschitz method (forward derivatives here, ball-mode secants there), Lemma J's
  coefficient extraction and endpoint. The comparison scripts (`p2b_compare_lc.py`, `p2b_cert_probe.py`,
  `p2b_atoms_cmp.py`) read the certificate's outputs or import its modules; they are not part of the check.

## 2. Lemma LC: the second implementation

**Representation** (`p2b_alg.py`). A bound on a cell is a weight monomial W = X^a θ^b ρ^e (ρ = X²/n) and a number v,
valid at every point of the cell and for every n ≥ n₀. Known quantities are kept as per-(j, m) coefficient lists and
bounded on each cell term by term; products add weights; sums pick a common weight; inverses use
‖(I + E)⁻¹‖ ≤ 1/(1 − sup‖E‖). Each bound carries a jet (the bound of x d/dx and (x d/dx)² where needed) and forward
derivatives with respect to the class constants.

**The class** is that of B_chain.md §13.1 with P2's F1–F3 from the start: R_c(1) = 0; |xc′| ≤ |xc^K′| + |xR_c′| in the
derivatives of C⁻¹ and D⁻¹; on L the second-derivative bound |x(xR_c′)′| ≤ h_cc nw as a 41st norm component. The
derivative constants D (per band and component) are unknowns of the closure like h.

**What is bounded.** Lemma R's G₁, W₂ (and G₂ = Λ⁻¹xW₂), ĝ, F_c, ρ₂ (as det C^K·ρ₂, a polynomial, divided by det C^K),
ρ₃, q, xq′, x(xq′)′, qΛ = (ℓ/x²)(I + e)⁻¹ε₁A^K, and on L the bounded forms with D̃ = D/(x − 1),
P_c = R_C/(x − 1) (|P_c| ≤ sup|R_c′|, |xP_c′| ≤ |x| sup|R_c″|/2).
- T_S off L: |G₁| plus Lemma K's gains split into the arc at |x| (integrand of the target cell) and the ray pieces
  (each with its own integrand and the gain from its top radius).
- T_S on L: |G| + |u(x′)| + sup|G′|·(δ/2)/(N_L − 1), with u(x′) = G₂(x′) + ∫_{P_x′} from the Na cells at radius
  1 ± δ/2; the derivative bound h_Sp from ‖(D̃^K)⁻¹A^K‖(|u(x′)|/(δ/2) + sup|G′|/(N_L − 1)).
- T_β: |ĝ|/(n − 1) + (1/(n₀ − 1)) sup_radial + φ_δ sup_arc of |xĝ′|/(n − 1).
- T_V, T_c: the offset d and the radial integrals with ∫w dt/t = ν⁵((1 + t_b)^{10} − (1 + t_a)^{10})/10 piece by piece,
  ρ₄ integrated term by term; on L the straight segment [0, x] for V and β and [1, x] for c (§7, O1).
- d₀ = n⁵ sup|d|: ρ₄ exactly on [0, 1], the rest cell by cell.

**Cells.** "cells" mode (the primary result) uses 181 cells: radial cells in each band, the ray at arg x = α in the
same radii, L as one cell, and on F cells in X up to 10⁸ plus X ≥ 10⁸ with ρ ≤ ρ_max. "box" mode uses one box per band
(B8_constants.md §1.3) plus the annulus 1 ± δ/2 of Na for the paths through the arc Ac.

**Closure and contraction.** Iterate (h, D) ↦ (P, Q) from 0, then descend h ← 1.01 P(h) from 1.1 times the fixed point
until 1.01 P(h) ≤ h (as B8_constants.md §7.1 describes). The Lipschitz matrix is the forward derivative of every output
with respect to every class constant, scaled by the constants: the product rule, the inverse rule and the linearity of
the path integrals bound the difference of every node at U, Ũ ∈ 𝒦_h by its derivative majorant (B_chain.md §13.4),
and maxima over cells are taken componentwise. This needs no convexity and no secant.

**Result** (`p2b_lc_cells.log`; box mode with the certificate-like options of §3.2 in `p2b_lc_box_crude_d_notesH.log`):

| band | h_V | h_S | h_β | h_c |
|---|---|---|---|---|
| N0 | 3.58×10⁶ | 965 | 7.03 | 7.03×10⁵ |
| Na | 2.20×10⁶ | 4.42×10³ | 4.92×10³ | 4.32×10⁵ |
| L | 4.35×10³ | 2.06×10³ | 1.75×10³ | 64.7 |
| Nb | 3.10×10⁴ | 1.16×10⁵ | 1.37×10⁵ | 2.53×10⁴ |
| F | 1.15×10⁶ | 3.00×10⁶ | 3.52×10⁵ | 9.35×10⁴ |

d₀ = 6.549×10⁵, h_Sp = 4.67×10⁻⁹, every P/h ≤ 0.990; s_* = 0.18757, N_L = 5.758×10³¹, η_L = 5.865;
Lipschitz matrix: spectral radius 0.5954, certified θ = 0.6022 (40 × 40 and 41 × 41 alike). d₀ ≤ 10⁷, so B8, (H8),
B5(1) and B5(2) with the data of d₀ = 10⁷ apply (B_chain.md §13.1, consequences).

## 3. Lemma LC: comparison with the certificate

### 3.1 The certificate's class under the second implementation

`p2b_compare_lc.py` evaluates p2b's majorant at the certificate's constants (h of `b8m_lc_K4.log`, h′ of
`b8m_lc_lip.json`, h_Sp; h_cc closed at 3517 from p2b's own value), in cells mode (`p2b_compare_lc_cells.log`) and in
the certificate-like box mode (`p2b_compare_lc_box.log`).

| | certificate | p2b, box mode | p2b, cells |
|---|---|---|---|
| largest value ratio P/h | 0.9901 | 0.9988 (S.N0) | 0.9900 (V.N0) |
| largest derivative ratio | 1 | 1.010 (S.Nb) | 1.000 (V, by equality) |
| d₀ | 3.58631×10⁶ | 3.58623×10⁶ | 3.58623×10⁶ |
| P_S on F (h_{S,F} = 4.121×10⁷) | 4.001×10⁷ | 3.693×10⁷ | 3.272×10⁷ |
| P_Sp (h_Sp = 5.968×10⁻⁷) | 5.91×10⁻⁷ | 3.16×10⁻⁸ | 3.16×10⁻⁸ |
| Lipschitz matrix: spectral radius (θ) | 0.7052 (0.7191) | 0.6536 (0.6665) | 0.6109 (0.6205) |

In cells mode every value ratio is below 1, every derivative ratio is at most 1 (the V-derivative bound is preserved
with equality, D_V = |ω|h_S, as in the certificate) and θ < 1. That is the independent confirmation of the certified
class. In box mode the derivative bound of S on Nb comes out 1 % above the certificate's h′ (p2b keeps one weight per
bound, and on the wide Nb box the conversion of P_S w + |G₁| to one weight loses 1 %); the cells resolve it.

### 3.2 Item by item, certificate-like settings

Two options bring p2b's box mode to the certificate's choices, both read from the notes: `--crude_d` uses
ρ^{K+1} ≥ n^{−K−1} for the offset terms (B8_constants.md §8.4), where p2b's own bound keeps X^{−10}; `--notesH` uses
Lemma K.3's H_F = sup‖η‖ℓ_F/(X²(1 − ‖Γ‖)), where p2b's own bound keeps |x|²/|x² − 1| ≤ 400/399 instead of ℓ_F = 21/19.
At the certificate's constants (P/h; the certificate's ratios are 0.9901 except on F):

| | V | S | β | c |
|---|---|---|---|---|
| N0 | 0.9900 | 0.9988 | 0.4104 | 0.9899 |
| Na | 0.9900 | 0.9868 | 0.9858 | 0.9931 |
| L | 0.9894 | 0.9890 | 0.9892 | 0.9655 |
| Nb | 0.9900 | 0.9898 | 0.9899 | 0.9901 |
| F | 0.9833 (cert 0.9833) | 0.8960 (0.9708) | 0.9120 (0.9897) | 0.9719 (0.9837) |

Own closure in these settings: d₀ = 3.572×10⁶ (certificate 3.586×10⁶), h_{S,F} = 3.12×10⁷ (4.12×10⁷),
h_Sp = 3.18×10⁻⁸ (5.97×10⁻⁷), θ = 0.666 (0.7355), H_F = 14.05 (14.044), N_F = 117.5 (117.475), γ_F = 0.415 (0.41504),
s_* = 0.18749 (0.18751).

### 3.3 The differences, located

Each was traced with `p2b_cert_probe.py` (it imports the certificate's modules and re-assembles its pieces) or by
reading the certificate code.

1. **d₀, V and c on N0–Nb, S and β on Na, L, Nb (0.03–0.4 %).** The atoms over the box: the certificate's enclosures of
   the formal coefficients come from ball arithmetic through the recursion and are 0.1–0.3 % wider than the exact
   polynomials' ranges for many entries (§3.4). The ρ₄ part of d₀: 1.90034×10⁵ here, 1.90109×10⁵ there.
2. **β on N0 (8.90 vs 21.48).** On N0 only the x² coefficients of τ_b matter. Their box enclosures in the certificate
   are much wider than the exact range: (j, m) = (4, 2): 276.8 vs 25.85; (5, 2): 242 vs exactly 0 (no product of the
   truncated series reaches ε⁵x²). The probe: the certificate's |ρ₃|/((n − 1)w) on N0 is 20.1, p2b's 8.9.
3. **S on N0 (+0.9 %).** Box mode has one N0 cell, so its arc gain is αe^{Hα} with H over all of N0 (0.960); the
   certificate takes the maximum over 300 radial cells (γ_N0 = 0.93540 ≈ α). In cells mode p2b's S.N0 is 0.25.
4. **S, β, c on F (1–8 % lower).** The local term (5.07×10⁶ vs 5.49×10⁶) and the integrand (7.67×10⁷ vs 8.32×10⁷).
   The probe splits G₁ on F: T₁ = (A₀ − A^K)S: 1.741×10⁶ vs 1.854×10⁶; T₂ = x²(B₀V − B^KV^K): 3.097×10⁶ vs 3.362×10⁶;
   ρ₂: 2.19×10⁵ vs 2.65×10⁵. The sources: p2b bounds the multiplication operator by c by its column sums (the induced
   l¹ norm), the certificate by |c|₁; p2b's ρ₂ is det C^K·ρ₂ (one polynomial) over det C^K, the certificate sums five
   terms.
5. **h_Sp (3.2×10⁻⁸ vs 5.9×10⁻⁷).** The datum u(x′) at the arc Ac uses Lemma K's arc gain at radius 1 ± δ/2 in p2b
   (≈ 3×10⁻¹⁶) and γ_Na, the maximum over all of Na (3.72×10⁻¹⁵, attained at |x| = 0.05), in the certificate. Both are
   valid; it feeds nothing back (the certificate's own row der/S/L).
6. **β and V on L.** The certificate bounds T_β and T_V on L along the straight segment [0, x] (no arc term:
   `fac = nf/n0` and `slowr` for r = L in `b8m_lc.evaluate`), which is valid because the integrands are analytic on the
   closed sector. p2b first used Q_x with its arc and got β.L = 1.18 at the certificate's h; with the straight segment,
   0.9892. (§7, O1.)
7. **The derivative ratio V.F = 0.952.** The certificate's h′_{V,F} = 1.285×10⁸ is |ω|h_{S,F} times θ_hi/θ_lo = 1.05,
   because its far-region representation replaces θ by its bounds at once; p2b's output is |ω|h_{S,F}.
8. **D_S on Na and Nb (box mode).** Before the comparison p2b's box-mode lower bound of |x² − 1|/X² was decoupled over
   the box (inf |x² − 1| over sup X²) and its near-disc bound used |x + 1| ≥ (r² + 2r cos α + 1)^{1/2}. Now the radius
   range is split into 200 sub-intervals and the near-disc bound is (δ/2)(2r² + 2 − δ²/4)^{1/2} (the parallelogram law,
   as the certificate's ℓ_N1 and as P2 checked). This changed box mode only.
9. **The Lipschitz entries.** The rows fed by d₀ (val/V/·, val/c/· ← val/β/Na) agree to 4 digits in box mode; in
   cells mode p2b's offset term carries X^{−10} and they are small except on N0. On F p2b's entries are 5–15 % smaller
   (item 4). The certificate's val/S/F ← val/β/Nb = 0.026 comes from its candidate γ_F × I_Rb (the full far gain on
   the Nb ray integrand); in p2b the Nb ray piece has the gain from radius 20, |D_20|/((n − 1)s_*) ≈ 10⁻²⁹.
   der/S/L ← val/β/Na: item 5.

### 3.4 The atoms

`p2b_atoms_cmp.py` compares the 325 descriptor coefficients of `b8m_atoms.atoms` with p2b's exact ones. At the box
centre they agree to all printed digits wherever both are nonzero; the certificate carries structurally zero entries as
rounding-level balls (≤ 2×10⁻⁶⁹). Over the box the certificate's bound is never below p2b's by more than 5×10⁻⁶
(relative; V (3, 6): 1504.5367 vs 1504.5440) and both contain the range sampled at 216 points of the box
(1504.4887). The certificate's bounds are wider for the low x-powers of τ_b and β (item 2) and it carries 25
structurally zero entries of τ_b as balls up to 4.4×10⁴ (j = 13); they enter only through weights with large powers of
ρ.

## 4. Lemma J: the second implementation and the comparison

### 4.1 Construction (`p2b_lcj.py`)

The secant coefficients are forward derivatives of p2b's LC majorant at the class constants, with each value input
seeded by its shape s_j as a derivative entry in its own weight (a pointwise derivative scales differently in n from the
node). So |f(R⁺) − f(R⁻)|(x) ≤ Σ_j ℓ_{f,j}(x)|v_j(x)| with ℓ_{f,j}s_j bounded cell by cell. Cells cover the Stokes
segment: L (whole), 12 real cells on [1 + δ/2, 1.5], 30 on [1.5, 20], and far cells to X = ∞ with ρ ≤ ρ* =
(n₀^{−1/2} + u*^{1/2})², the value of ρ at x* for every n ≥ n₀ (ρ_max = 0.0085 elsewhere; u* = 0.0081). On L the
pseudo-component p = sup_{[1,x]}|v_c′| ≤ Σ_j sup_L(ℓ_{F_c,j}s_j/|s|)·m(x)N_j is added. The endpoint is derived from the
label condition (Λ(v_S + δG) = −Ωv_S + δCΦ⁻ + δ[xc′P], δP bounded through C⁻¹ and Y), and Γ from §14's formulas
with n s_β ≥ nρ².

### 4.2 Results and comparison

| Γ (row ← column) | certificate (`b8m_lcj.json`) | p2b, own class | p2b, certificate's class |
|---|---|---|---|
| V ← S | 5.9396 | 5.9396 | 5.9396 |
| S ← V, S, β, c | 1.397×10⁻³, 4.603×10⁻², 2.17×10⁻⁵, 2.420×10⁻² | 1.196×10⁻³, 3.97×10⁻², 1.66×10⁻⁵, 2.055×10⁻² | 1.197×10⁻³, 4.11×10⁻², 1.66×10⁻⁵, 2.067×10⁻² |
| β ← V, S, β, c | 6.038, 3.320, 1.58×10⁻⁵, 34.33 | 5.806, 0.668, 5×10⁻¹⁹, 32.32 | 5.811, 0.668, 5×10⁻¹⁹, 32.36 |
| c ← S, β, c | 0.1085, 1.533×10⁻⁴, 2.181×10⁻³ | 0.0992, 1.385×10⁻⁴, 1.962×10⁻³ | 0.0993, 1.385×10⁻⁴, 1.965×10⁻³ |
| h_H, ‖Λ⁻¹(x*)‖, \|Φ⁻\| | 0.1080, 9.533×10⁻³, 2.001 | 0.1023, 9.023×10⁻³, 1.318 | 0.1023, 9.023×10⁻³, 1.576 |
| Perron root; certified θ | 0.17593 | 0.1597; 0.1630 | 0.1604; 0.1637 |

The differences:
- *ρ on the far part of the segment.* p2b uses ρ ≤ ρ* = 0.0081 (the segment ends at x*), the certificate the F box
  with ρ_max = 0.0085. This lowers h_H, ‖Λ⁻¹(x*)‖ and the S- and c-rows by 5–15 %.
- *Γ_{β,S} (0.668 vs 3.32).* The probe (`p2b_cert_probe.py lcj`) shows the certificate's value comes from band Na,
  where it takes the supremum over the whole band box (|x| from 0.05, X^{−4} at X = 1.05); the Stokes segment meets Na
  only on [1 + δ/2, 1.5], where p2b's cells give at most 0.668 (L gives 0.668 in both). Both are valid.
- *Γ_{β,β} (5×10⁻¹⁹ vs 1.6×10⁻⁵).* The certificate's value is its F-box bound of ℓ_{ĝ,β}s_β/(n s_β) with s_β as the
  two-monomial sum ρ/n + ρ²; the quantity is O(1/n), as p2b finds. It does not affect θ.
- *|Φ⁻|.* p2b bounds it at x* with ρ*, the certificate on the F box.

Lemma J's inputs from Lemma LC are the class constants; θ changes by 0.0007 between p2b's class and the certificate's.

## 5. Mutations

Criteria written before the comparison: every certificate-like run must close with θ < 1, d₀ within 2 % of the
certificate's 3.59×10⁶, S on N0–Nb within 2 % of the certificate's table, h_Sp ≤ 6.0×10⁻⁷; Lemma J must give θ < 1
and Γ_{S,S} within a factor 2 of 0.046. Each mutation below violates one (`p2b_lc_mutations.log`,
`p2b_lcj_mutations.log`; LC mutations in the certificate-like box mode):

| mutation | effect | caught by |
|---|---|---|
| `tauD`: ρ₂ × 1.5 | h_S on N0–Nb × 1.5 (Na 7.7×10⁴) | the S comparison |
| `taub`: τ_b × 1.5 | d₀ = 5.27×10⁶ | the d₀ comparison |
| `rho4`: the exact ρ₄ part of d₀ dropped | d₀ = 3.39×10⁶ (−5.6 %) | the d₀ comparison |
| `kernelH`: H_r = 0 in Lemma K | h_{S,F} = 1.62×10⁷, θ = 0.54 | the S.F comparison |
| `NL`: N_L = 2 | h_Sp = 1.1×10⁵ | h_Sp ≤ 6.0×10⁻⁷ |
| `nod0`: the offset d dropped | h_V ≈ 4×10⁵ | the V comparison |
| `gamma3`: arc gains × 3 | no closure ("inverse: lower bound lost") | closure |
| J `noend`: endpoint coefficients 0 | Γ_{S,S} = 7×10⁻³ | Γ_{S,S} comparison |
| J `omega`: \|ω\| × 10⁴ | θ = 8.66 | θ < 1 |
| J `nodecay`: backward factor × n₀ | θ = 8.8×10²⁹ | θ < 1 |
| J `shape`: s_V = 1 | unbounded weight ratio (the run stops) | the run |
| series `--mutate=sign` (Ω sign) | Lemma F's orders fail exactly | `p2b_series.py` |

## 6. P2's findings F1–F7 applied, and the reruns

The second implementation confirms F1–F3 independently: its class contains F1 and F2's conditions and its derivative
bounds use F3's class bound, and the certificate's constants close under it (§3.1); the 41st norm component (F2) leaves
θ unchanged (0.6022 in 40 × 40 and 41 × 41). It also confirms F5 and F6 by derivation (the z^{−Λ} factor is largest at
μ₊, the ψ_C bound at μ₋).

| finding | applied in | rerun | result |
|---|---|---|---|
| F1: R_c(1) = 0 in 𝒦_h | B_chain.md §13.1 | — | statement only (the code imposes it) |
| F2: the R_c″ bound on L in 𝒦_h and in N | B_chain.md §§13.1, 13.4 | — | the code's bound is the class bound; p2b's 41 × 41 θ = 40 × 40 θ |
| F3: \|xc′\| ≤ \|xc^K′\| + \|xR_c′\| in `b8m_lc.build` | `b8m_lc.py`; B_chain.md §13.1 | `b8m_lc.py --lip`, `--lipcheck`, `b8m_lcj.py` | see below |
| F4: name the uniqueness class | B_chain.md §15, C_selection.md §13.1 | — | wording |
| F5: lemma L's C-term t^{2−δ}, ψ_C with z^{μ₋+3} | `b5_label.py`; B_chain.md §2.3 | `b5_label.py` | ∫g 0.302096486 → 0.302096493, ε_L 0.3526917 (unchanged) |
| F6: C_E with u*^{−μ₊}; ψ_C in `ell_bound` with μ₋ | `b5_jump.py`, `b7_ift.py`; B_chain.md §3.2 | `b5_jump.py` (three data sets), `b7_ift.py`, its mutation | ln C_E 76.91427 → 76.91446 (d₀ = 0), 76.91613 → 76.91633 (10⁶), 76.93966 → 76.93985 (10⁷); n₀ = e^{89.75}; C₇ 2.8954×10²⁹ → 2.8959×10²⁹ = e^{67.838}; `--mutate=JD` caught |
| F7: u′ = (Λ/x)u + G′; s_b = ρ/n + ρ² | B_chain.md §13.3; `b8m_lcj.py` docstring | — | text |

F3 rerun of the certificate (`b8m_lc_K4.log`, `.out`, `b8m_lc_lip.json`, then `b8m_lcj.py`): F3_PLACEHOLDER

The old outputs are kept beside the new ones as `*_preP2b.*` (`b5_label`, `b5_jump*`, `b7_ift`, `b7_ift_mutJD`,
`b8m_lc_K4`, `b8m_lc_lip`, `b8m_lc_lipcheck`, `b8m_lcj`, `b8m_lc_numcheck_n1e3`). The quoted constants C_E = e^{76.914}
(d₀ = 0), e^{76.916} (10⁶), e^{76.940} (10⁷), u*^{1−μ₊} = 33.84 (B_chain.md §3), ε_L ≤ 0.3527, n₀ = e^{89.75},
C₇ = e^{67.84} are unchanged at their printed precision.

## 7. New observations

None changes a number or breaks a lemma.

1. **O1 (notes): the paths on L.** B_chain.md §13.2 defines T_V and T_β with Q_x, but `b8m_lc.evaluate` bounds them on
   L along the straight segment [0, x] and T_c along [1, x]. That is valid (the integrands are analytic on the closed
   sector, which contains L; along [0, x], |(s/x)^{n−1}| ≤ 1 and ∫₀^{|x|}(t/|x|)^{n−1}dt/t = 1/(n − 1)), and with Q_x's
   arc the β bound on L would exceed h_{β,L} by 18 % at the certified constants (§3.3 item 6). *Proposed text* for
   §13.2, after the definition of T: "For x ∈ L the bounds of T_V and T_β use the segment [0, x], and that of T_c the
   segment [1, x]; this is allowed because the integrands are analytic on Σ_n."
2. **O2 (notes): the atoms over the box.** B8_constants.md §2 says the structural zeros of Lemma F are imposed exactly;
   over the box the certificate carries some structurally zero coefficients (τ_b at ε^{6..13}x^{2,4}) as balls up to
   4.4×10⁴, and its enclosures of the low x-powers of τ_b and β are up to 10× wider than the exact range (§3.4). This is
   valid (every ball contains the true value) and costs only in β on N0 (21.7 where 9.0 suffices).
3. **O3 (conservative).** h_Sp uses γ_Na for the data at Ac (§3.3 item 5) and the certificate's Lemma J takes band
   suprema over all of Na and over the F box with ρ_max (§4.2). Valid; noted because they are the main reasons the
   certificate's h_Sp and Γ_{β,S} are larger than needed.
4. **O4 (old log).** None found: the stored logs of `b5_jump.py` (three data sets) and `b7_ift.py` are reproduced by the
   unchanged code before the fixes were applied (checked by rerunning the committed version of `b5_jump.py`).

## 8. Still not independently checked

- B5(1)'s code: `b5_bounds.py`, `b5_formulas.py` (the inner enclosures, lemma S over 𝒦, the super-solutions x_Φ).
- B8 beyond `--d0`: `b8m_bounds.py`'s bootstrap and `b8m_scan.py` (p2b's LC majorant covers B8's estimates on the same
  bands in the light-cone gauge, but B8's own certificate with c(0) = c^K(0) + d and its (H8) conversion were not
  re-implemented), and `b8m_h8.py` ((H8)).
- Lemma L (`b5_label.py`) and Lemma P (`b7_adjoint.py`): read and rerun, not re-implemented (P2_review.md §4).
- Lemma 6b₁'s disc covering at M = 1 (Theorem A), and the M = 3 items of the queue (lemmas S and 6b₀'s majorant code,
  the 6b₁ discs).
- Lemma J's a priori norm θ₀ on (1, 1 + δ/2] (the certificate's θ₀ = 5.8×10⁻⁵) was not recomputed.

## 9. Reproduce

In `solver/proof/review_p2b/` (`run_p2b.sh` runs all of it, about 20 minutes on the Mac):

```
python3 p2b_series.py                     # exact series over the box; Lemma F's orders (p2b_series.json)
python3 p2b_lc.py --mode=cells --log=p2b_lc_cells.log                          # own closure, theta 0.6022
python3 p2b_lc.py --mode=box --crude_d --notesH --log=p2b_lc_box_crude_d_notesH.log   # certificate-like, d0 3.57e6
python3 p2b_lc.py --mode=box --crude_d --notesH --mutate=rho4                  # d0 3.39e6: caught (also tauD, taub,
                                                                                # kernelH, NL, nod0, gamma3)
python3 p2b_lcj.py --h=p2b_lc_cells.json --log=p2b_lcj_cells.log               # Lemma J, theta 0.163
python3 p2b_lcj.py --h=cert --tag=cert --log=p2b_lcj_cert.log                   # on the certificate's class, 0.164
python3 p2b_lcj.py --h=p2b_lc_cells.json --mutate=omega                        # theta 8.66: caught (also noend,
                                                                                # nodecay, shape)
python3 p2b_compare_lc.py --mode=cells --log=p2b_compare_lc_cells.log           # the certified class under p2b
python3 p2b_cert_probe.py; python3 p2b_cert_probe.py lcj                        # pieces inside the certificate
python3 p2b_atoms_cmp.py                                                        # atoms, centre and box
```

In `solver/proof/` the reruns of §6 are the commands of B_chain.md §11.

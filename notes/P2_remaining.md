# P2's remaining items: a review plan, and Lemma J's θ₀ checked independently

Issue #988, queue item P2 of [PROOF_QUEUE.md](PROOF_QUEUE.md). P2 ([proof/P2_review.md](../proof/P2_review.md)) and
P2b ([proof/P2b_review.md](../proof/P2b_review.md)) left six certificate items without an independent check
(P2_review.md §5, P2b_review.md §8). This note plans a check of each (§1) and does the smallest (§2). Written
2026-11-19 by Claude (Anthropic) under Nathan Moinvaziri's direction. The reviewer, a helper session, had not worked
on Theorem B's certificates or read P1's transcripts. It read P2_review.md and P2b_review.md first. Not reviewed.
Status words as in [AGENTS.md](../AGENTS.md).

## 1. The six items

Sizes are line counts of the certificate scripts at main b081fbb2. Costs are estimates.

| # | Item | What it certifies, and where | The link that rests on it | Code | What an independent check needs | Cost |
|---|---|---|---|---|---|---|
| 1 | Lemma J's θ₀ | The a priori step of Lemma J. On (1, 1 + δ/2], with m = 1, the affine map with the endpoint value v_S(1 + δ/2) contracts (θ₀ = 5.8×10⁻⁵), so Ψ_out is a Neumann sum there and N(Ψ_out) < ∞. `b8m_lcj.py` (one block near its end), B_chain.md §14, last paragraph | (H-J), and through it Theorem B(1) (B_chain.md §15). Without it, T₀'s contraction does not reach Ψ_out | about 35 lines, on top of Lemma J's secant coefficients | the Γ⁰ entries re-derived from the statement, evaluated on secant coefficients that share no code with `b8m_lcj.py`. P2b's are such coefficients (P2b_review.md §4) | hours (done here, §2) |
| 2 | Lemma 6b₁'s discs, M = 1 | y⁺ analytic on the annular sector 0.0624 ≤ \|ξ\| ≤ 0.2, by 178 validated Taylor discs and their gluing. `lemma6b1_disc.py`, `lemma6b1_cover.py`, `lemma6b1_estimate.py`, A_certify.md §12 | Theorem A(M = 1): lemma 6b, that the continued y⁺ is the Borel sum. Everything in Theorem B(1) starts from Theorem A's box | 215 + 478 + 140 | (a) the covering's geometry and gluing in exact rationals, which is cheap; (b) a second disc certificate in mpmath intervals (Taylor coefficients and a remainder bound for the ODE on each disc), for every disc or a stated sample. A rerun on Windows (#889) reproduced the rows, but a rerun is not an independent check | (a) hours, (b) 2–4 days |
| 3 | (H8) | B8's continuation turned into B5(1)'s data on the inner arc \|x\| = n^γ, δ = (8.2×10⁻⁵, 79.03, 3.231, 6.463), and 122.31 for δ_Φ at d₀ = 10⁷. `b8m_h8.py`, B8_constants.md §8.1–8.3 | B5(1), so B5(2)'s C_E and B7, for n ≥ e^{73.89} (C_selection.md §13.1) | 339 | a reread of §8.2's three pieces by hand, then the conversion in mpmath intervals from B8's printed h. The input h is item 4's, so the check would share it; that is recorded, not avoided | 1–2 days |
| 4 | B8 beyond `--d0` | Lemma B8(1): the bootstrap P(h) < h on 𝒟′_n (K = 4, κ = 0.3, κ₁ = 0.6), closing for ln n₀ ≥ 62 and with the normalization offset for ln n₀ ≥ 73.89. `b8_alg.py`, `b8m_bounds.py`, `b8m_scan.py`, B8_constants.md §§3–7 | B8(1), (H8), B5(1) and the gauge range of Lemma LC (C_selection.md §13.1) | 227 + 402 + 84 | a second bootstrap in other arithmetic. P2b's LC majorant covers B8's estimates on the same bands in the light-cone gauge (P2b_review.md §8), and P2 reproduced the formal series and atoms (P2_review.md §5). What remains is B8's own closure with c(0) = c^K(0) + d. P2b's bound algebra could serve, with the sharing recorded | 3–5 days |
| 5 | B5(1) | The inner fixed point W = O(1/n) on the matching domain (κ = 0.3, Z = 0.0085), with the super-solutions x_Φ = (39784, 38776) that enter C_E. `b5_bounds.py`, `b5_formulas.py`, B5_inner.md §§3–5 | B5(1), and so B5(2) and B7. x_Φ is a factor of C_E, the dominant constant of n₀ (notes/P7_scope.md §4.1) | 692 + 288 | a reread of B5_inner.md §4's proof and a second super-solution certificate (b + Lx < x) in mpmath intervals. Its input is lemma S's enclosure over 𝒦 (A_certify.md §4.5), which would be shared | 4–7 days |
| 6 | The M = 3 items | Lemma S(3) and lemma 6b₀(3), which share the majorant code of `lemma6_sector_M.py` and the helpers of `fixed_point.py`, and the 183 discs of lemma 6b₁(3), in `lemma6b1_disc_M.py`, `lemma6b1_cover_M.py` and `lemma6b1_estimate_M.py` (A_lemmas_M.md §§3, 5). A_M_review.md checked the gluing, not the discs | Theorem A(M = 3). It is not on Theorem B(1)'s path. It anchors the truncation sequence that P6 uses | 189 + 520 (helpers), 141 + 598 + 207 | a second majorant implementation for lemma S (the start-value lemma's second implementation, #427, does not cover it), and the disc work of item 2 at M = 3 | 3–5 days |

**Order, by cost against the weight of the link.**
1. Item 1. It is the cheapest, and it is on Theorem B(1)'s path (done in §2).
2. Item 2(a), the covering's geometry. It costs hours and sits under the whole chain. Then item 2(b).
3. Item 3, (H8). It is moderate, and it sets the n ≥ e^{73.89} links.
4. Item 5, B5(1). It is the largest, and it carries the dominant constant.
5. Item 4, B8. It is large, and P2b's light-cone majorant already covers part of it.
6. Item 6. It is large and off Theorem B(1)'s path.

## 2. Item 1: Lemma J's θ₀, checked independently

### 2.1 Protocol, fixed before the run

- **Independence.** `b8m_lcj.py` is not opened before the numbers are committed. The Γ⁰ entries are derived here from
  B_chain.md §14's equations. The secant coefficients come from P2b's code (`review_p2b/p2b_lc.py` and
  `p2b_lcj.py`), which shares no code with the certificate (P2b_review.md §1). They are evaluated on P2b's L cell
  (1, 1.025] with every shape set to 1. One fact about the certificate was seen in a search before this protocol
  was written: a comment of `b8m_lcj.py` names the a priori norm "m = 1 on (1, 1 + delta/2], s_V = s_c = 1".
- **The derivation** (the docstring of `solver/proof/review_p2c/p2c_theta0.py`). Plain sup norms on I = (1, 1 + δ/2],
  with m = 1 and unit shapes.
  - S, backward, with B_chain.md §14's kernel bound: Γ⁰_{S,j} = (1 + h_H)/(1 − h_H) sup_I ℓ_{G,j}.
  - V: Γ⁰_{V,S} = |ω| ln(1 + δ/2).
  - c: Γ⁰_{c,j} = ln(1 + δ/2) sup_I ℓ_{F_c,j}.
  - β: Γ⁰_{β,j} = sup_I ℓ_{ĝ,j}/(n − 1).
  - The pseudo-component of δD/(x − 1) is added to ℓ_G and ℓ_ĝ, as P2b adds it.
- **Pass criteria.** θ₀ < 1 on the certificate's class constants (`--h=cert`) and on P2b's own
  (`--h=p2b_lc_cells.json`). The certificate's θ₀ = 5.8×10⁻⁵ is compared, not required. P2_review.md §3.3's
  θ₀ ≤ 1.8×10⁻⁵, from the saved Γ₀, is compared too. A difference larger than a factor 10 is located after the commit,
  by reading the certificate's block.
- **Controls.** `--planted`: the Perron certification on a rank-one matrix of known root, and the Γ⁰ assembly against
  hand values, must pass. `--planted --mutate=perron` must fail. `--mutate=nodecay`, the backward kernel times n₀
  (the certificate's own mutation in B_chain.md §14), must give θ₀ > 1 and exit 1.

### 2.2 Results

**θ₀ < 1 is confirmed. The certificate's Γ⁰ agrees with this one entry by entry, up to the differences explained
below, and its Perron root is 1.08×10⁻¹⁶ against 1.03×10⁻¹⁶ here. The printed θ₀ = 5.8×10⁻⁵ is a valid upper bound,
about 10¹¹ above the root.** The own numbers were committed (59591319) before `b8m_lcj.py` was opened. Logs are in
`solver/proof/review_p2c/`, run with `/Users/nathan/venv-314/bin/python` (mpmath 1.3.0), in about 3 s each.

| run | Perron root of Γ⁰ | certified θ₀ | exit |
|---|---|---|---|
| `--h=cert` (the certificate's class) | 1.029×10⁻¹⁶ | 1.000×10⁻¹² | 0 |
| `--h=p2b_lc_cells.json` (P2b's own class) | 1.029×10⁻¹⁶ | 1.000×10⁻¹² | 0 |
| `--h=cert --rate=half` (the weighted form's kernel factor) | 1.198×10⁻¹⁶ | 1.000×10⁻¹² | 0 |
| `--planted` | 0.2070 (planted 0.2070) | 0.2104 | 0 |
| `--planted --mutate=perron` | 2.054 | 2.069 | 1 (fails as it must) |
| `--h=cert --mutate=nodecay` (pre-set control) | 0.553 | 0.564 | 0 (**did not fire**) |
| `--h=cert --mutate=gaininv` (added after the run) | 9.08×10¹⁵ | 9.27×10¹⁵ | 1 (fails as it must) |

The certified θ₀ sits at 10⁻¹² because the certification's weight search adds 10⁻¹² to its trial root.

- **On L every loop of Γ⁰ is tiny.** The O(1) entries are Γ⁰_{V,S} = |ω| ln(1.025) = 0.0733 and Γ⁰_{c,β} = 0.0247,
  since ℓ_{F_c,β} = 1. Every entry that closes a loop with them is O(ρ) or O(1/n), and ρ = (1 + |x|)²/n ≈ 4/n on L.
  So θ₀ is about the square root of the largest two-cycle, Γ⁰_{c,β}Γ⁰_{β,c} = 0.0247 × 2.2×10⁻³¹.
- **The pre-set control did not fire, and that is reported, not hidden.** The backward kernel times n₀ multiplies the
  S row's O(ρ) coefficients by n₀ and leaves them O(1). The largest loop, S → V → S, then reaches 0.31, and θ₀ = 0.564.
  The certificate's own mutation fires in the weighted norm of the main contraction, where the S row is not O(ρ).
  Here it is not a test. The control added after the run, with that reason recorded in the script's docstring,
  inverts the β row's gain at n = n₀. It is the factor (n₀ − 1)² that a multiplication in place of a division would
  make, and it fails at θ₀ = 9.3×10¹⁵. The inversion for every n ≥ n₀ is unbounded, and P2b's algebra refuses it
  with an error.

### 2.3 The comparison with the certificate

`b8m_lcj.py`'s block (the comment "the a priori norm (m = 1 on (1, 1 + delta/2], s_V = s_c = 1)" and the lines after
it) and its saved Γ₀ (`b8m_lcj.json`) were read after the commit. The certificate's Perron root was recomputed with
this review's certification (`p2c_cert_gamma0.py`, `p2c_cert_gamma0.log`).

| entry | certificate | here | the difference |
|---|---|---|---|
| Γ⁰_{V,S} | 0.07424 | 0.07333 | the certificate integrates dy/y over (1, 1 + δ/2] as δ/2, here as ln(1 + δ/2) |
| Γ⁰_{S,V}, Γ⁰_{S,c} | 4.21×10⁻³², 6.58×10⁻³¹ | 3.39×10⁻³², 5.30×10⁻³¹ | the ratio, 1.242, is the kernel factor (next row). With it removed, the secant coefficients agree to four digits (4.2110×10⁻³² and 6.5832×10⁻³¹) |
| the β row and column | Γ⁰_{β,V} = 5.34, Γ⁰_{β,c} = 27.2; Γ⁰_{S,β} = 3.7×10⁻⁶³ | 4.3×10⁻³², 2.2×10⁻³¹; 3.6×10⁻³¹ | the certificate's shape is s_β = 1/n, here 1: a diagonal scaling, which leaves the Perron root unchanged |
| Γ⁰_{c,S} | 4.24×10⁻³³ | 5.17×10⁻³⁵ | the certificate's c row carries F_c's pseudo-component (fcpL × cpj), which P2b's F_c node does not have (§2.4) |
| the kernel factor | (1 + h_H)/(1 − h_H) = 1.242, with h_H = 0.108 over the whole segment | the same form, with h_H on L, 9.7×10⁻³², so 1.000 | both use the full decay rate, which m = 1 allows. The certificate's h_H is conservative on L |
| Perron root | 1.08×10⁻¹⁶ | 1.03×10⁻¹⁶ | agreement to 5% |
| printed θ₀ | 5.84×10⁻⁵ | 1.0×10⁻¹² | the certificate's power iteration floors each weight at 10⁻¹², and the tight weight needs a ratio of about 4×10²³ (`p2c_cert_gamma0.log`). Its θ₀ is a valid bound, only loose. P2_review.md §3.3's 1.8×10⁻⁵ is a bound of the same kind |

### 2.4 Discrepancies (reported, not fixed)

1. **F_c's pseudo-component on L.** The certificate's c row adds fcpL × cpj (`b8m_lcj.py`, the Gam0["c"] line): the
   derivative of δF_c with respect to sup_{[1,x]}|v_c′|. P2b's F_c node has no such entry (its pseudo-component
   derivative is 0, `p2b_lc.cell_eval`), so P2b's Lemma J and this check omit a term that the certificate includes.
   Which form is right needs the bounded form of F_c on L, which B_chain.md §13.3 does not write out. The term cannot
   change θ₀'s verdict, since the c row's loops are O(ρ). Whether it changes P2b's main θ (0.164, against the
   certificate's 0.176) was not checked.
2. **The printed θ₀ is loose by a factor of about 10¹¹.** That is harmless, because the lemma needs only θ₀ < 1.
   B_chain.md §14 quotes 5.8×10⁻⁵, and P2_review.md §3.3 quotes ≤ 1.8×10⁻⁵. A proposed wording for B_chain.md §14:
   "(θ₀ ≤ 6×10⁻⁵, a loose bound; the Perron root of Γ⁰ is about 10⁻¹⁶, #988)".
3. **This review's pre-set real-data control did not fire** (§2.2). That is a defect of the control, not of the
   certificate.

### 2.5 Reproduce

From `solver/proof/review_p2c/`, with python and mpmath (P2b's modules are imported from `../review_p2b/`):

```
python3 p2c_theta0.py --planted                       # PASS, exit 0
python3 p2c_theta0.py --planted --mutate=perron       # FAIL, exit 1
python3 p2c_theta0.py --h=cert --log=p2c_theta0_cert.log              # theta0 1.0e-12, exit 0
python3 p2c_theta0.py --h=p2b_lc_cells.json --log=p2c_theta0_own.log  # theta0 1.0e-12, exit 0
python3 p2c_theta0.py --h=cert --mutate=gaininv       # theta0 9.3e15, exit 1
python3 p2c_theta0.py --h=cert --mutate=nodecay       # theta0 0.564, exit 0 (the pre-set control; it did not fire)
python3 p2c_cert_gamma0.py                            # the certificate's Gamma0: Perron root 1.08e-16
```

## 3. Item 2(a): lemma 6b₁'s M = 1 covering, its geometry checked independently

Issue #991. Lemma 6b₁ (A_certify.md §12.3–12.4) glues 178 discs of lemma D into an analytic continuation of y⁺ over
A₁ = {25 ≤ |u| ≤ 260, −3π/4 ≤ arg u ≤ 0}. This section checks the geometry of that covering. The discs' own
certificates are item 2(b).

### 3.1 Protocol, fixed before the run

- **Independence.** `lemma6b1_cover.py` and the certificate's check output `lemma6b1_check_1e-9.json` are not opened
  before the numbers are committed. Only the top-level key names of that output were seen while finding the data. The
  inputs are the stored runs `lemma6b1_runs/row_00.json` to `row_11.json`. They give per row ρ_j (the decimal the run
  used), r_j and n_j, and per disc t_jk and its certificate flag. The definitions of the tiles, the gluing order and
  the agreement points are A_certify.md §12.3's statements: a_j = ρ_j − 0.6r_j, b_j = ρ_j + 0.6r_j,
  α_j = 0.6 · (3π/4)/K_j, φ_jk = −(3π/4)(1 − t_jk) and G₀ = {|u| > 257, −3π/4 − 0.4 < arg u < 0}. Here a_j, b_j and
  α_j are taken as exact rationals and π-multiples, not as the certificate's binary floats. A lemma G proof may
  choose its tiles inside the discs, so passing with these tiles proves the gluing on its own.
- **Arithmetic.** mpmath intervals (`iv`, 200 bits) for every quantity involving π, sin or cos, and exact rationals
  elsewhere. A comparison counts only when it is proven. A strict inequality needs the upper end of one side below the
  lower end of the other.
- **What is checked** (`solver/proof/review_p2c/p2c_cover.py`).
  - G1. The data: 12 rows, 178 discs, every disc flagged certified, t_j0 = 0 and t_jK = 1, and r_j < ρ_j.
  - G2. Each closed tile lies in its open disc. The distance from the centre is largest at a corner of the closed
    polar rectangle, so the four corners are checked.
  - G3. The open tiles cover the closed A₁. The bands cover [25, 260] with strict overlaps, and in every row the
    angular intervals cover [−3π/4, 0] with strict overlaps.
  - G4. A second, tile-free route: A₁ is split into closed polar cells, and each cell is proven to lie in one open
    disc. Cells are bisected until that holds, with a cap of 10⁵ cells.
  - G5. The gluing, tile by tile in §12.3's order. Each meeting set M_m (T_m against G₀ and every earlier tile) is
    built from the pairwise intersections not proven empty. M_m must be connected through proven-nonempty overlaps,
    which is undecided if a needed overlap is not proven. The agreement point q_m must lie in T_m and in the partner
    piece. The partner is G₀ for (0, 0), tile (j, k − 1) for k ≥ 1 with q at mid-angle on the arc, and tile
    (j − 1, 0) for k = 0 with q on the ray at radius (a_{j−1} + b_j)/2. The curve along which the two functions agree
    must lie in both discs: for k ≥ 1 the half-arcs, 2ρ_j sin(Δφ/4) < r_j; for k = 0 the ray segments,
    ρ_{j−1} − |q| < r_{j−1} and |q| − ρ_j < r_j.
  - G6. Reachability: every tile's partner comes earlier in the order, so every tile is reached from (0, 0) on the
    start ray, and s₀ = 262e^{−3iπ/4} lies in G₀.
- **Pass criteria.** G1 to G6 all hold. The counts of pieces and links are reported and compared afterwards with
  A_certify.md §12.4's 566 and 450, which are not required to match, since the meeting sets are counted here in a
  different way.
- **Planted failures, which must each fail.** (P1) Disc (5, 6) is removed. (P2) Row 3's α is set to 0.49 Δφ, so
  consecutive tiles of that row no longer overlap and the links between them break. Both run through the same code.

### 3.2 Results

**The geometry is confirmed. The open tiles cover A₁, each closed tile lies in its open disc, A₁ is also covered by
the discs directly, and every link of lemma G holds. Both planted coverings fail.** The own numbers were committed
(148a63c4) before `lemma6b1_cover.py` and `lemma6b1_check_1e-9.json` were opened. Logs and JSON are in
`solver/proof/review_p2c/` (`p2c_cover_real.log`, `_drop.log`, `_gap.log`), run with
`/Users/nathan/venv-314/bin/python` (mpmath 1.3.0) in 1 to 11 s each.

| check | real covering | P1: disc (5, 6) removed | P2: row 3's α = 0.49 Δφ |
|---|---|---|---|
| G1 data | ok: 12 rows, 178 discs, all certified | ok (177 tiles) | ok |
| G2 tiles in discs | ok, smallest margin r² − d² = 21.4 | ok | ok |
| G3 tiles cover A₁ | ok | **FAIL**: row 5's tiles 5 and 7 do not overlap | **FAIL**: row 3's consecutive tiles do not overlap |
| G4 cells in discs | ok: 8930 cells, none bisected | **FAIL**: a sliver near \|u\| = 133, arg u = −1.162 lies in no disc | ok (the discs are unchanged) |
| G5 gluing | ok: 566 pieces, 450 proven links (388 in spanning trees) | **FAIL**: tile (5, 7) has no partner; the meeting set of tile (6, 5) is not shown connected | **FAIL**, 32 failures: tiles (3, 1) to (3, 17) have no piece with their partner, and 15 tiles of row 4 have meeting sets not shown connected |
| G6 reach, s₀ ∈ G₀ | ok | ok | ok |
| exit | 0 | 1 | 1 |

- The tiles here are exact (a_j = ρ_j − 3r_j/5, α_j = (3/5)(3π/4)/K_j). Lemma G may use any tiles inside the discs,
  so these checks prove the gluing by themselves. Their agreement with the certificate's float tiles is a cross-check.
- G4 is the tile-free route. It covers A₁ with closed polar cells of 2.5 in |u| and 3π/380 in angle, and each is
  proven inside one open disc with no bisection needed. Under P1 it finds the uncovered sliver where disc (5, 6) was,
  which shows that the discs, and not only the tiles, would then fail to cover.
- P1 fails in the same three places the certificate's own control `check --drop=5,6` reports (A_certify.md §12.5):
  tile (5, 7) has no agreement point, tile (6, 5)'s meeting set is disconnected, and (5, 6)'s direction is not
  covered.

### 3.3 The comparison with the certificate

The certificate's check output (`lemma6b1_check_1e-9.json`) reports ok, no failures, 178 tiles, 566 pieces and 450
links. These are the same counts as here, from code that shares nothing with it but the statements of A_certify.md
§12.3 and the stored runs. Its tiles are the same formulas in binary floats (α_j = 0.6 Δφ_j, a_j = ρ_j − 0.6 r_j,
`lemma6b1_cover.py`), which differ from the exact ones here in the last bits. No discrepancy was found.

One wording point, not a discrepancy: PROOF.md §4 and issue #991 describe the annular sector as
0.0624 ≤ |ξ| ≤ 0.2. A₁ = {25 ≤ |u| ≤ 260} is 0.0620 ≤ |ξ| ≤ 0.2, which contains it. G₀ = {|u| > 257} covers the rest
down to |ξ| = 0.06238.

### 3.4 Reproduce

From `solver/proof/review_p2c/`, with python and mpmath:

```
python3 p2c_cover.py --log=p2c_cover_real.log        # PASS, exit 0 (about 2 s)
python3 p2c_cover.py --plant=drop --log=p2c_cover_drop.log   # FAIL, exit 1 (about 11 s)
python3 p2c_cover.py --plant=gap --log=p2c_cover_gap.log     # FAIL, exit 1
```

## 4. Item 2(b): lemma 6b₁'s M = 1 disc certificates, a second implementation

Issue #992. Lemma D (A_certify.md §12.2) certifies, for each of the 178 discs, that y⁺ continues analytically from the
disc's centre to the whole disc, with a bound M on |y − y₀|. This section re-implements that certificate.

### 4.1 Protocol, fixed before the run

- **Independence.** `lemma6b1_disc.py` and `lemma6b1_estimate.py` are not opened. The code
  (`solver/proof/review_p2c/p2c_disc.py`) is written from A_certify.md §12.2's statement of lemma D and from
  §4.4's equations in the u = 1/z variables of §11.4. The only shared things are the inputs: the stored runs, the
  equation's form and coefficients, κ = 100, the order n_j and the radius r_j. The equation is compared with
  `lemma6_usys.SysU.rhs` at random points, the only use of the certificate's code. Seen before this protocol: the
  stored Λ and sup bounds of discs (0, 0) to (0, 4) and (11, 0) to (11, 2) (the first lines of `rows.log` and
  `row_11.log`, and row_00's first record), and A_certify.md §12.4's per-row maxima.
- **Method, own.** The arithmetic is python-flint 0.9 balls (acb) at 192 bits, in code written here.
  - Taylor coefficients: Picard iteration on truncated power series (acb_series), started from the input box, n + 1
    times.
  - Defect: in the form with denominators cleared, d̃ = u^m (p′ − F(u, p)) with m = 1 for P and V and m = 2 for C. It
    is an exact polynomial in τ (acb_poly products), so no expansion of 1/u with a remainder is needed. Its coefficients
    below order n must contain 0, and |d(τ)| ≤ |τ|ⁿ Δ̃/(|c| − r)^m with Δ̃ = Σ_{k≥n} |d̃_k| r^{k−n}.
  - Lipschitz matrix: forward-mode derivatives (series of length 2) of F over the box B, the z-ball 1/(c + disc(r))
    and the ω-ball.
  - Check: lemma D's contraction with Perron weights, Λ = r/(n + 2) · max_i Σ_j ω_i L_ij/ω_j < 1,
    ρ = 2F₀/(1 − Λ), and b_j ≥ ρ r^{n+1}/ω_j, iterated with B = Y₀ + disc(M⁰ + b) until it holds. M_j = M⁰_j + b_j.
- **Inputs.**
  - The 12 discs on the start ray (k = 0) use the stored row-start boxes (`lemma6b1_runs/ray.json`, `starts`), exactly.
    They include the corners (0, 0) and (11, 0) and the disc with the largest stored Λ, (11, 0).
  - The 166 arc discs have no stored box. Their boxes are reconstructed by this review: its own high-precision,
    non-rigorous integration along the arc from the stored row-start box's midpoint, inflated to 4 times the stored
    box radius plus 10⁻¹⁵. A certificate there covers lemma D's computation on that box. It covers the certificate's
    box only if that box lies inside, which is assumed, not proven. The integrator is first checked against the stored
    ray boxes: from s₀ along the ray to s₁, its value must lie within the stored box at s₁.
- **Sample, then all.** First the 12 ray discs, all of row 11 (the smallest margins), and the real-axis corner discs
  (0, 24) and (11, 6). Then all 178 if a disc costs under 30 s.
- **Pass criteria.** Every disc certified (Λ < 1, the vanishing orders hold, b closes). The margins (Λ and max_j M_j)
  are recorded and committed before the stored ones are read in full, then compared.
- **Controls, which must each fail.**
  - (K1) A too-large disc: disc (11, 0)'s stored input with radius 16. The branch point u* = 9.6746e^{−2.1198i}
    (A_certify.md §12.1) lies at distance about 15.7 from its centre, so it is inside.
  - (K2) The sign mutation: the defect computed with the fast term's sign flipped (V_u = +V/2 + …), while the Taylor
    coefficients use the true equation, on the 12 ray discs. Every certificate must fail.
- **Planted checks, which must pass.** The equation against `SysU.rhs` (relative 10⁻⁴⁰ at 10 random points), and the
  integrator against the stored ray box at s₁.

### 4.2 Results

The own numbers were committed (ce097b2b) before the stored per-disc margins were read for the comparison (§4.3).
The work paused there at the coordinator's word on 2026-11-19 and resumed the same day.

- **Planted checks pass.** The equation matches `SysU.rhs` at 10 random points to a relative 3.5×10⁻⁵⁸
  (`p2c_disc_checkint.log`). The integrator lands inside each stored ray box from s_j to s_{j+1}, at deviations of
  10⁻²⁰ to 1.7×10⁻⁹ against box radii of 2×10⁻⁹ to 8×10⁻⁸.
- **All 178 discs, protocol inputs: 177 certified** (`p2c_disc_all.log`, 512 s).
  - All 12 start-ray discs, with exact stored inputs, are certified. Their Λ are 0.1699 for (0, 0) and 0.4397 for
    (11, 0).
  - **Disc (11, 1) is not certified (Λ = 1.494)**, on its reconstructed box (4 × the stored radius 1.0×10⁻⁶).
- **Located.** It is input size, not a certificate error.
  - The reconstructed input at (6, 7) is y⁺: three integration paths agree to 10⁻²².
  - This implementation's Picard iteration on ball series amplifies the input box's radius more than the certificate
    does. On (6, 7) a box of 10⁻¹² gives max M⁰ = 0.88, and the protocol's 4 × box radius gives 5.08.
  - Rerunning (11, 1) at 0, 1, 2 and 3 times the stored radius certifies it with Λ = 0.393, 0.453, 0.541 and 0.674
    (`p2c_disc_diag_11_1.log`, outside the protocol).
- **Controls fail as they must.** K1, disc (11, 0) at radius 16 with u* inside, gives Λ = 4.7×10⁸. K2, the sign
  mutation, fails all 12 ray discs at order 0 of the V defect.
- **Development fixes**, found on discs (0, 0) and (11, 0) before the full run, whose stored Λ and sup had been seen
  (§4.1):
  - python-flint's series cap, 10 by default, which truncated the Taylor series;
  - the weight orientation, ω_i = 1/v_i;
  - the z-ball, which now uses lemma D's own enclosure 1/c + disc(r/(|c|(|c| − r))). python-flint's inverse of the
    box was 8 times looser.
- **Disc (11, 1)** is finished in §4.4, and the comparison is in §4.3.

### 4.3 The comparison with the certificate

`p2c_disc_compare.py` reads the stored per-disc Λ, sup (the bound M) and box radius from `lemma6b1_runs/row_XX.json`
and sets them beside this review's (`p2c_disc_compare.log`).

- **Same inputs, same margins.** On the 12 start-ray discs the input is the stored box itself. Λ agrees to at most
  5.6×10⁻⁶ relative and M to at most 3.1×10⁻¹¹ on rows 0 to 10. On (11, 0) both are larger here by 1.6×10⁻³ (Λ 0.43968
  against 0.43897, M 15.489 against 15.469). That is the input-size effect of §4.2 on the stored box of 7.9×10⁻⁸
  (attributed, not proven).
- **Arc discs.** Their inputs here are larger than the certificate's: 4 times (protocol) or 2 times (§4.4) the stored
  radius, centred on this review's midpoint. Λ agrees to 0.2% on rows 0 to 10 at both sizes. M agrees to 0.3% on
  rows 0 to 3 and 7 to 10. On rows 4 to 6 and 11 it is larger, and more so at 4 times than at 2 times (row 4 agrees
  at 2 times):

| row | max Λ, stored | here, 4× | here, 2× | max M, stored | here, 4× | here, 2× |
|---|---|---|---|---|---|---|
| 4 | 0.1793 | 0.1793 | 0.1793 | 0.4023 | 0.4347 | 0.4023 |
| 5 | 0.1833 | 0.1833 | 0.1833 | 0.6173 | 1.19 | 0.6372 |
| 6 | 0.1909 | 0.1910 | 0.1909 | 1.067 | 5.081 | 2.614 |
| 11 | 0.4390 | 0.6482 (6 of 7) | 0.5410 | 15.47 | 84.45 (6 of 7) | 43.01 |

- **The stored files against A_certify.md §12.4.** The row maxima of Λ, box radius and sup in the stored files
  round to the table's entries (for example row 11: 0.4390, 1.77×10⁻⁵, 15.47 against 0.439, 1.8×10⁻⁵, 15.5).

### 4.4 Disc (11, 1)

- **At the protocol's input** (4 × its stored box radius 1.024×10⁻⁶, plus 10⁻¹⁵) it is not certified: Λ = 1.494.
- **Cause: the input box.** The bound M grows by 9.85×10⁶ per unit of input radius, between 0 and 1 times the stored
  radius (`p2c_disc_11_1_growth.log`). The growth allowed by the Lipschitz bound over the disc,
  exp((n + 2)Λ) at the thin input, is 1.49×10⁷.
  - So the growth comes from the Lipschitz-based bound, not from the ball series alone.
  - A centred (Gronwall) form would carry the same factor. On the 4.1×10⁻⁶ box it would allow a drift of about 60,
    so it was not pursued.
- **The input that is needed.** The certificate's box at (11, 1) contains y⁺(c) and has radius at most 1.024×10⁻⁶
  (A_certify.md §12.4: the Lohner box at the centre, the input of lemma D).
  - It therefore lies inside this review's midpoint ± (2 × 1.024×10⁻⁶ + |midpoint − y⁺(c)|) on each axis.
  - The midpoint agrees with y⁺(c) to 10⁻²² by three paths. That is the protocol's non-rigorous assumption, and its
    10⁻¹⁵ covers it.
  - So 2 times the stored radius plus 10⁻¹⁵ contains the certificate's box. The 4 times of §4.1 was a margin chosen
    beforehand, not a need.
- **At 2 times, all 178 discs are certified** (`p2c_disc_all_infl2.log`, 212 s). Disc (11, 1) has Λ = 0.541 and max
  M = 43.0, against the stored 0.418 and 14.6.

**Status.** The protocol's inputs certify 177 of 178 discs. The smallest inputs that contain the certificate's boxes,
under the same assumption, certify 178 of 178, and both controls fail.
- **The 12 start-ray discs** are confirmed outright, on the stored inputs.
- **The 166 arc discs:** the review covers lemma D's computation on boxes assumed to contain the certificate's. That
  assumption, the midpoint's accuracy, is not proven (§4.1).
- **Versions:** python-flint 0.9.0 and FLINT 3.6.0.

### 4.5 Reproduce

From `solver/proof/review_p2c/`, with python-flint 0.9.0:

```
python3 p2c_disc.py --checkeq                                  # PASS, exit 0
python3 p2c_disc.py --checkint                                 # PASS, exit 0
python3 p2c_disc.py --discs=all > p2c_disc_all.log             # 177 of 178, exit 1 (about 510 s)
python3 p2c_disc.py --discs=all --infl=2 --tag=all_infl2 > p2c_disc_all_infl2.log   # 178 of 178, exit 0
python3 p2c_disc.py --discs=11:1 --infl=0 --tag=diag_11_1_infl0  # and --infl=1, 2, 3 (tags ..._infl1, 2, 3)
python3 p2c_disc.py --discs=11:0 --radius=16 > p2c_disc_K1_toolarge.log   # K1: not certified, exit 1
python3 p2c_disc.py --discs=ray --mutate=sign > p2c_disc_K2_sign.log      # K2: 0 of 12, exit 1
python3 p2c_disc_compare.py p2c_disc_all.jsonl p2c_disc_all_infl2.jsonl > p2c_disc_compare.log
python3 p2c_disc_11_1_growth.py > p2c_disc_11_1_growth.log
```
